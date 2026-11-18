"""Explicit saved-preparation loop: one immutable regional model cell at a time."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.neural_analysis.encoding.configuration import EncodingConfig
from src.neural_analysis.encoding import modeling, results
from src.neural_analysis.encoding.preparation import PreparedRegion, response_activity
from src.neural_analysis.encoding.predictors import (
    build_joint_predictors,
    OUTCOME_COLUMNS,
)


def run_prepared_encoding(run_directory: Path | str) -> dict:
    """Resume or execute coarse model cells using only immutable saved preparation.

    Parameters
    ----------
    run_directory : Path or str
        Existing session-local run containing config.json/progress.json and a
        manifest reference to published preparation. Live CSVs, metadata and
        spike files need not remain mounted after preparation publication.

    Returns
    -------
    dict
        JSON-compatible progress with status complete and completed_cells
        mapping IDs to run-relative NPZ names. Cells may be scientifically
        unavailable and still complete. Existing compatible published cells
        are adopted before fitting, even if their prior index update crashed.
        Corrupt/mismatched published cells or changed numerical code raise
        ValueError and are never overwritten. No PCA is fitted here.
    """
    started = perf_counter()
    run = Path(run_directory)
    manifest = results.load_run_manifest(run)
    if results.numerical_code_identity() != manifest["identities"]["code_identity"]:
        raise ValueError(
            "Numerical code differs from this run; start a new run instead of resume"
        )
    if (
        results.numerical_runtime_versions()
        != manifest["identities"]["runtime_versions"]
    ):
        raise ValueError(
            "Numerical runtime versions differ from this run; start a new run instead of resume"
        )
    config = results.restore_config(manifest["config"])
    snapshot = results.load_preparation(
        Path(manifest["session_root"]) / manifest["preparation_path"]
    )
    if snapshot["meta"]["preparation_id"] != manifest["identities"]["preparation_id"]:
        raise ValueError("Preparation identity differs from run manifest")
    # Validate each completed file once in the loop instead of a second status pass.
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    cells = _planned_cells(config, snapshot["regions"], manifest["identities"])
    designs = {
        (role, encoding): build_joint_predictors(
            snapshot["trials"],
            region.bin_rows.iloc[:: region.unit_counts.shape[1]].row_position.to_numpy(
                dtype=int
            ),
            region.bin_edges_s,
            encoding,
        )
        for role, region in snapshot["regions"].items()
        for encoding in config.encodings
    }
    planned_ids = {cell["cell_id"] for cell in cells}
    if not set(progress["completed_cells"]).issubset(planned_ids):
        raise ValueError("Progress contains unexpected cell identities")
    progress.update(
        {
            "status": "running",
            "planned_cells": len(cells),
            "unavailable_regions": snapshot["meta"].get("unavailable", {}),
        }
    )
    results.write_progress(run, progress)
    for description in cells:
        settings = description["settings"]
        region = snapshot["regions"][settings["region"]]
        design = designs[settings["region"], settings["encoding"]]
        cell_started = perf_counter()
        save_s = None
        cell_id = description["cell_id"]
        relative = str(Path("cells") / f"{cell_id}.npz")
        path = run / relative
        if (
            cell_id in progress["completed_cells"]
            and progress["completed_cells"][cell_id] != relative
        ):
            raise ValueError("Published cell path differs from planned cell identity")
        if path.exists():
            cell = results.load_cell(path)
            _validate_existing_cell(
                cell, description, manifest["identities"], region, design
            )
            action = "reused"
        else:
            if cell_id in progress["completed_cells"]:
                raise ValueError(
                    "Completed cell file is missing; refusing silent refit"
                )
            cell = _compute_cell(
                config, region, description, manifest["identities"], design
            )
            cell["meta"]["elapsed_s"] = perf_counter() - cell_started
            save_started = perf_counter()
            results.save_cell(path, cell)
            save_s = perf_counter() - save_started
            action = "fitted"
        _append_cell_log(run, cell, action, perf_counter() - cell_started, save_s)
        # The durable NPZ always precedes this small index publication.
        progress["completed_cells"][cell_id] = relative
        results.write_progress(run, progress)
    progress["status"] = "complete"
    progress["elapsed_s"] = perf_counter() - started
    results.write_progress(run, progress)
    return progress


# Small planning and numeric assembly helpers; no source discovery or PCA.
def _planned_cells(
    config: EncodingConfig, regions: dict[str, PreparedRegion], identities: dict
) -> list[dict]:
    """Input current controls/region records/IDs; return unique region/response/encoding/estimator cells.

    Only signed ridge shares penalty-mode labels; OLS runs once per design.
    No activity arrays are copied and no legacy model selection occurs.
    """
    descriptions = []
    for role in regions:
        for representation in config.response_representations:
            for encoding in config.encodings:
                for estimator in config.estimators:
                    if estimator == "ols":
                        modes = [(None, [])]
                    elif encoding == "signed_context":
                        modes = [(config.penalty_modes[0], list(config.penalty_modes))]
                    else:
                        modes = [(mode, [mode]) for mode in config.penalty_modes]
                    for mode, aliases in modes:
                        settings = {
                            "region": role,
                            "representation": representation,
                            "encoding": encoding,
                            "estimator": estimator,
                            "penalty_mode": mode,
                        }
                        key = {**settings, "model_id": identities["model_id"]}
                        identifier = hashlib.sha256(
                            json.dumps(key, sort_keys=True).encode("ascii")
                        ).hexdigest()[:20]
                        descriptions.append(
                            {
                                "cell_id": identifier,
                                "settings": settings,
                                "penalty_aliases": aliases,
                            }
                        )
    return descriptions


def _compute_cell(
    config: EncodingConfig,
    region: PreparedRegion,
    description: dict,
    identities: dict,
    design: dict,
) -> dict:
    """Input shared activity/design/controls; return one native unit-Hz or PC result cell without observed duplication.

    Eligible X(N,P) and Y(N,R) use identical row indices/folds for all response
    representations. Slopes(P,R), predictions(N,R), contrasts(2,R), fold
    contrasts(5,2,R) retain original response units and support flags.
    """
    settings = description["settings"]
    indices = np.flatnonzero(design["valid"])
    x = design["predictors"][indices]
    names = design["column_names"]
    folds = region.bin_rows.fold_id.to_numpy(dtype=int)[indices]
    reason = ""
    try:
        activity, ids, units = response_activity(region, settings["representation"])
        y = activity.reshape(-1, len(ids))[indices]
    except ValueError as error:
        ids = ()
        units = "PC score"
        y = np.empty((len(indices), 0))
        reason = str(error)
    if not len(indices):
        reason = "No eligible joint predictor windows"
    p, r = len(names), len(ids)
    weights = np.ones(p)
    if (
        settings["encoding"] == "separate_context"
        and settings["penalty_mode"] == "matched_context"
    ):
        weights[:2] = 0.5
    penalties = (
        config.alpha * weights if settings["estimator"] == "ridge" else np.zeros(p)
    )
    meta = {
        "cell_id": description["cell_id"],
        "settings": settings,
        "preparation_id": identities["preparation_id"],
        "model_id": identities["model_id"],
        "penalty_aliases": description["penalty_aliases"],
        "row_count": len(indices),
        "predictor_count": p,
        "response_count": r,
        "trial_count": int(region.bin_rows.iloc[indices].row_position.nunique()),
        "bin_width_ms": config.bin_width_ms,
        "window_s": list(config.window_s),
        "response_units": units,
        "raw_slope_units": units + " per native predictor unit",
        "available": not bool(reason),
        "unavailable_reason": reason,
        "reference_valid": False,
        "configured_alpha": config.alpha if settings["estimator"] == "ridge" else None,
        "feature_penalties": penalties.tolist(),
        "objective": "summed SSE plus feature-specific L2 slope penalties; unpenalized intercept",
        "predictor_scaling": "native; no standardization",
        "response_scaling": "unsmoothed Hz"
        if units == "Hz"
        else "fixed unwhitened PC scores",
        "reference_scope": "all-eligible-data reference fit",
        "score_scope": "held-out regression on unit rates"
        if units == "Hz"
        else "held-out regression conditional on fixed descriptive session PCA; includes test-block activity",
        "condition_counts": design["condition_counts"],
        "excluded_bin_count": int((~design["valid"]).sum()),
        "cv_available": False,
        "cv_reason": reason or "Not fitted",
        "fit_elapsed_s": 0.0,
    }
    cell = _empty_cell_arrays(len(indices), p, r)
    cell.update(
        {
            "selected_row_indices": indices,
            "response_ids": np.asarray(ids, dtype=str),
            "column_names": np.asarray(names),
            "predictors": x,
            "fold_ids": folds,
            "penalty_vector": penalties,
            "meta": meta,
        }
    )
    if reason:
        return cell
    started = perf_counter()
    try:
        evaluated = modeling.evaluate_block_cv(
            x,
            y,
            folds,
            estimator=settings["estimator"],
            alpha=config.alpha,
            penalty_weights=weights,
        )
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        meta.update(
            available=False,
            unavailable_reason=f"Reference fit unavailable: {error}",
            cv_reason=str(error),
            fit_elapsed_s=perf_counter() - started,
        )
        cell["fold_reasons"][:] = str(error)
        return cell
    meta["fit_elapsed_s"] = perf_counter() - started
    reference = evaluated["reference_fit"]
    cell.update(
        reference_slopes=reference.slopes,
        reference_intercept=reference.intercept,
        standardized_slopes=modeling.standardize_coefficients(reference, x, y),
    )
    for name in (
        "reference_predictions",
        "held_out_predictions",
        "fold_slopes",
        "fold_intercepts",
        "fold_ids",
        "fold_valid",
        "fold_train_counts",
        "fold_test_counts",
        "fold_mse",
        "fold_r2",
        "mse",
        "r2",
    ):
        cell[name] = evaluated[name]
    for name in ("fold_reasons", "fold_r2_reason", "r2_reason"):
        cell[name] = np.asarray(evaluated[name], dtype=str)
    contrast = modeling.compute_reward_contrasts(reference, x, names)
    cell.update(
        contrasts=contrast["values"],
        contrast_valid=contrast["valid"],
        contrast_reasons=contrast["reasons"],
        predictor_supported=contrast["predictor_supported"],
        predictor_nonzero_counts=contrast["predictor_nonzero_counts"],
    )
    fold_counts = []
    for fold, fitted in enumerate(evaluated["fold_fits"]):
        train = (folds >= 0) & (folds != fold)
        fold_counts.append(
            {
                "train": _condition_counts(x[train], names, config.bin_width_ms),
                "test": _condition_counts(x[folds == fold], names, config.bin_width_ms),
            }
        )
        if fitted is not None:
            item = modeling.compute_reward_contrasts(fitted, x[train], names)
            cell["fold_contrasts"][fold] = item["values"]
            cell["fold_contrast_valid"][fold] = item["valid"]
            cell["fold_contrast_reasons"][fold] = item["reasons"]
            cell["fold_predictor_supported"][fold] = item["predictor_supported"]
            cell["fold_predictor_nonzero_counts"][fold] = item[
                "predictor_nonzero_counts"
            ]
    meta.update(
        reference_valid=True,
        cv_available=evaluated["cv_available"],
        cv_reason=region.cv_unavailable_reason or evaluated["cv_reason"],
        design_rank=contrast["design_rank"],
        contrast_names=list(contrast["names"]),
        fold_condition_counts=fold_counts,
        coefficient_convention=reference.coefficient_convention,
        reference_fit_metadata=_fit_metadata(reference),
        fold_fit_metadata=[
            None if fit is None else _fit_metadata(fit)
            for fit in evaluated["fold_fits"]
        ],
    )
    return cell


def _condition_counts(x: np.ndarray, names: tuple[str, ...], bin_width_ms: int) -> dict:
    """Input paired native X(N,P), names(P,), width in ms; return original trial counts per outcome."""
    post_bins = 1000 // bin_width_ms
    return {
        name: int(np.count_nonzero(x[:, names.index(name)])) // post_bins
        for name in OUTCOME_COLUMNS
    }


def _empty_cell_arrays(n: int, p: int, k: int) -> dict[str, np.ndarray]:
    """Input bin/predictor/response counts; return NaN unavailable fit arrays with documented native axes.

    Slopes(P,K), predictions(N,K), fold slopes(5,P,K), and score(K) arrays use
    response/native predictor units, response units, and squared-response/dimensionless metrics.
    """
    return {
        "reference_slopes": np.full((p, k), np.nan),
        "reference_intercept": np.full(k, np.nan),
        "standardized_slopes": np.full((p, k), np.nan),
        "reference_predictions": np.full((n, k), np.nan),
        "held_out_predictions": np.full((n, k), np.nan),
        "fold_slopes": np.full((5, p, k), np.nan),
        "fold_intercepts": np.full((5, k), np.nan),
        "fold_valid": np.zeros(5, dtype=bool),
        "fold_train_counts": np.zeros(5, dtype=int),
        "fold_test_counts": np.zeros(5, dtype=int),
        "fold_mse": np.full((5, k), np.nan),
        "fold_r2": np.full((5, k), np.nan),
        "mse": np.full(k, np.nan),
        "r2": np.full(k, np.nan),
        "fold_reasons": np.full(5, "Unavailable fit", dtype="U512"),
        "fold_r2_reason": np.full((5, k), "Unavailable fit", dtype="U512"),
        "r2_reason": np.full(k, "Unavailable fit", dtype="U512"),
        "contrasts": np.full((2, k), np.nan),
        "contrast_valid": np.zeros(2, dtype=bool),
        "contrast_reasons": np.full(2, "Unavailable fit", dtype="U512"),
        "fold_contrasts": np.full((5, 2, k), np.nan),
        "fold_contrast_valid": np.zeros((5, 2), dtype=bool),
        "fold_contrast_reasons": np.full((5, 2), "Unavailable fit", dtype="U512"),
        "predictor_supported": np.zeros(p, dtype=bool),
        "predictor_nonzero_counts": np.zeros(p, dtype=int),
        "fold_predictor_supported": np.zeros((5, p), dtype=bool),
        "fold_predictor_nonzero_counts": np.zeros((5, p), dtype=int),
    }


def _fit_metadata(fitted: modeling.LinearFit) -> dict:
    """Input LinearFit with slopes(P,K); return scalar convention/penalty/limitation metadata, no arrays."""
    return {
        "estimator": fitted.estimator,
        "configured_alpha": fitted.configured_alpha,
        "feature_penalties": fitted.penalties.tolist(),
        "coefficient_convention": fitted.coefficient_convention,
        "limitations": list(fitted.limitations),
    }


def _append_cell_log(
    run: Path, cell: dict, action: str, elapsed_s: float, save_s: float | None
) -> None:
    """Append one cell's settings/counts/status/reasons to run Path; wall time is scalar seconds.

    save_s is measured NPZ publication seconds, or None for reused cells.
    Input cell contains native coefficient/prediction arrays, which are not
    duplicated in the log. Return None; plain JSON text preserves errors and
    scientific unavailability alongside execution progress.
    """
    meta = cell["meta"]
    record = {
        name: meta[name]
        for name in (
            "cell_id",
            "settings",
            "row_count",
            "trial_count",
            "configured_alpha",
            "feature_penalties",
            "cv_available",
            "cv_reason",
            "unavailable_reason",
        )
    }
    record.update(
        {
            "action": action,
            "status": "available" if meta["available"] else "unavailable",
            "elapsed_s": elapsed_s,
            "save_s": save_s,
            "fold_reasons": cell["fold_reasons"].tolist(),
        }
    )
    with (run / "run.log").open("a", encoding="ascii") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def _validate_existing_cell(
    cell: dict,
    description: dict,
    identities: dict,
    region: PreparedRegion,
    design: dict,
) -> None:
    """Check cell named-array meta against planned IDs/settings; return None, no axis/unit changes."""
    meta = cell["meta"]
    if (
        meta["cell_id"] != description["cell_id"]
        or meta["settings"] != description["settings"]
        or meta["penalty_aliases"] != description["penalty_aliases"]
    ):
        raise ValueError(
            "Existing published cell settings/identity differ from requested model"
        )
    if any(meta.get(key) != identities[key] for key in ("preparation_id", "model_id")):
        raise ValueError("Existing published cell preparation/model identity differs")

    ids = (
        region.unit_ids
        if description["settings"]["representation"] == "units"
        else tuple(f"PC{rank}" for rank in region.pc_ranks)
    )
    indices = np.flatnonzero(design["valid"])
    if (
        not np.array_equal(cell["response_ids"], ids)
        or not np.array_equal(cell["selected_row_indices"], indices)
        or not np.array_equal(cell["column_names"], design["column_names"])
        or not np.array_equal(cell["predictors"], design["predictors"][indices])
    ):
        raise ValueError(
            "Published cell response/row/predictor identity differs from preparation"
        )
