"""Explicit saved-preparation loop: one immutable regional model cell at a time."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.neural_analysis.encoding.configuration import EncodingConfig
from src.neural_analysis.encoding import modeling, results
from src.neural_analysis.encoding.preparation import PreparedRegion
from src.neural_analysis.encoding.predictors import encode_context


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
        raise ValueError("Numerical code differs from this run; start a new run instead of resume")
    if results.numerical_runtime_versions() != manifest["identities"]["runtime_versions"]:
        raise ValueError("Numerical runtime versions differ from this run; start a new run instead of resume")
    config = results.restore_config(manifest["config"])
    snapshot = results.load_preparation(Path(manifest["session_root"]) / manifest["preparation_path"])
    if snapshot["meta"]["preparation_id"] != manifest["identities"]["preparation_id"]:
        raise ValueError("Preparation identity differs from run manifest")
    # Validate each completed file once in the loop instead of a second status pass.
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    cells = _planned_cells(config, snapshot["regions"], manifest["identities"])
    planned_ids = {cell["cell_id"] for cell in cells}
    if not set(progress["completed_cells"]).issubset(planned_ids):
        raise ValueError("Progress contains unexpected cell identities")
    progress.update({"status": "running", "planned_cells": len(cells),
                     "unavailable_regions": snapshot["meta"].get("unavailable", {})})
    results.write_progress(run, progress)
    for description in cells:
        cell_started = perf_counter()
        cell_id = description["cell_id"]
        relative = str(Path("cells") / f"{cell_id}.npz")
        path = run / relative
        if cell_id in progress["completed_cells"] and progress["completed_cells"][cell_id] != relative:
            raise ValueError("Published cell path differs from planned cell identity")
        if path.exists():
            cell = results.load_cell(path)
            _validate_existing_cell(cell, description, manifest["identities"])
            action = "reused"
        else:
            if cell_id in progress["completed_cells"]:
                raise ValueError("Completed cell file is missing; refusing silent refit")
            region = snapshot["regions"][description["settings"]["region"]]
            cell = _compute_cell(config, region, description, manifest["identities"])
            cell["meta"]["elapsed_s"] = perf_counter() - cell_started
            results.save_cell(path, cell)
            action = "fitted"
        _append_cell_log(run, cell, action, perf_counter() - cell_started)
        # The durable NPZ always precedes this small index publication.
        progress["completed_cells"][cell_id] = relative
        results.write_progress(run, progress)
    progress["status"] = "complete"
    progress["elapsed_s"] = perf_counter() - started
    results.write_progress(run, progress)
    return progress


# Small planning and numeric assembly helpers; no source discovery or PCA.
def _planned_cells(config: EncodingConfig, regions: dict[str, PreparedRegion], identities: dict) -> list[dict]:
    """Input controls/available regions/IDs; return distinct model settings and verified penalty aliases.

    No numerical arrays are copied. Signed same/matched ridge have identical
    effective alpha and share one fit; OLS has no penalty mode and runs once.
    """
    descriptions = []
    for role in regions:
        for timing_mode in config.timing_modes:
            for window in config.windows:
                for encoding in config.encodings:
                    for estimator in config.estimators:
                        if estimator == "ols":
                            modes = [(None, [], None)]
                        elif encoding == "signed_context":
                            modes = [(config.penalty_modes[0], list(config.penalty_modes), config.alpha)]
                        else:
                            modes = [(mode, [mode], config.alpha / 2 if mode == "matched_context" else config.alpha)
                                     for mode in config.penalty_modes]
                        for mode, aliases, effective_alpha in modes:
                            settings = {"region": role, "timing_mode": timing_mode, "window": window,
                                "encoding": encoding, "estimator": estimator, "penalty_mode": mode}
                            identity = {**settings, "effective_alpha": effective_alpha,
                                        "model_id": identities["model_id"]}
                            cell_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("ascii")).hexdigest()[:20]
                            descriptions.append({"cell_id": cell_id, "settings": settings, "penalty_aliases": aliases})
    return descriptions


def _compute_cell(config: EncodingConfig, region: PreparedRegion, description: dict, identities: dict) -> dict:
    """Assemble one model cell from shared scores(W,B,K_full)/context(W,B), preserving native units.

    Inputs are run controls, a prepared region, cell settings and scientific IDs.
    Return named arrays with selected row indices(N,), X(N,P), coefficient and
    prediction products, plus meta. Observed Y remains solely in preparation.
    """
    settings = description["settings"]
    mode, encoding, window = settings["timing_mode"], settings["encoding"], settings["window"]
    context = region.context_values[mode].reshape(-1)
    predictors, encoded_valid, names = encode_context(context, encoding)
    valid = region.context_valid[mode].reshape(-1) & encoded_valid
    if window != "whole":
        valid &= region.bin_rows[window].to_numpy(dtype=bool)
    indices = np.flatnonzero(valid)
    x = predictors[indices]
    full_scores = region.timeline.scores.reshape(-1, region.timeline.scores.shape[-1])
    y = full_scores[indices][:, region.pc_ranks - 1]
    folds = region.bin_rows.fold_id.to_numpy(dtype=int)[indices]
    penalty_mode = settings["penalty_mode"] or "same_numeric"
    meta = {"cell_id": description["cell_id"], "settings": settings,
        "preparation_id": identities["preparation_id"], "model_id": identities["model_id"],
        "penalty_aliases": description["penalty_aliases"],
        "row_count": len(indices), "predictor_count": len(names), "response_count": len(region.pc_ranks),
        "trial_count": int(region.bin_rows.iloc[indices].row_position.nunique()),
        "bin_width_ms": config.bin_width_ms,
        "available": bool(len(indices)), "unavailable_reason": "" if len(indices) else "No eligible predictor bins in this model cell",
        "configured_alpha": config.alpha if settings["estimator"] == "ridge" else None,
        "effective_alpha": (config.alpha / 2 if encoding == "separate_context" and penalty_mode == "matched_context" else config.alpha)
            if settings["estimator"] == "ridge" else None,
        "objective": "summed squared errors plus effective_alpha times squared slope norm; unpenalized intercept",
        "predictor_scaling": "native; no normalization", "response_scaling": "fixed unwhitened PC scores",
        "raw_slope_units": "PC-score units per native predictor unit",
        "reference_scope": "all-eligible-data reference fit",
        "score_scope": "held-out regression conditional on fixed descriptive session PCA",
        "cv_available": False, "cv_reason": "No eligible predictor bins", "reference_valid": False,
    }
    cell = _empty_cell_arrays(len(indices), len(names), len(region.pc_ranks))
    cell.update({"selected_row_indices": indices, "pc_ranks": region.pc_ranks,
                 "predictors": x, "fold_ids": folds, "column_names": np.asarray(names), "meta": meta})
    if not len(indices):
        return cell
    try:
        evaluated = modeling.evaluate_block_cv(x, y, folds, estimator=settings["estimator"], alpha=config.alpha,
                                              encoding=encoding, penalty_mode=penalty_mode)
    except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
        reason = f"Reference fit unavailable: {error}"
        meta.update({"available": False, "unavailable_reason": reason, "cv_reason": reason})
        cell["fold_reasons"] = np.full(5, reason)
        cell["fold_r2_reason"] = np.full((5, len(region.pc_ranks)), reason)
        cell["r2_reason"] = np.full(len(region.pc_ranks), reason)
        return cell
    reference = evaluated["reference_fit"]
    cell.update({"reference_slopes": reference.slopes, "reference_intercept": reference.intercept,
                 "standardized_slopes": modeling.standardize_coefficients(reference, x, y)})
    for name in ("reference_predictions", "held_out_predictions", "fold_slopes", "fold_intercepts",
                 "fold_ids", "fold_valid", "fold_train_counts", "fold_test_counts", "fold_mse", "fold_r2", "mse", "r2"):
        cell[name] = evaluated[name]
    for name in ("fold_reasons", "fold_r2_reason", "r2_reason"):
        cell[name] = np.asarray(evaluated[name], dtype=str)
    meta.update({"cv_available": evaluated["cv_available"],
        "cv_reason": region.cv_unavailable_reason or evaluated["cv_reason"], "reference_valid": True,
        "coefficient_convention": reference.coefficient_convention,
        "reference_fit_metadata": _fit_metadata(reference),
        "fold_fit_metadata": [None if fitted is None else _fit_metadata(fitted) for fitted in evaluated["fold_fits"]]})
    return cell


def _empty_cell_arrays(n: int, p: int, k: int) -> dict[str, np.ndarray]:
    """Input bin/predictor/PC counts; return NaN unavailable fit arrays with documented native axes.

    Slopes(P,K), predictions(N,K), fold slopes(5,P,K), and score(K) arrays use
    PC/native predictor units, PC units, and squared-PC/dimensionless metrics.
    """
    return {"reference_slopes": np.full((p, k), np.nan), "reference_intercept": np.full(k, np.nan),
        "standardized_slopes": np.full((p, k), np.nan), "reference_predictions": np.full((n, k), np.nan),
        "held_out_predictions": np.full((n, k), np.nan), "fold_slopes": np.full((5, p, k), np.nan),
        "fold_intercepts": np.full((5, k), np.nan), "fold_valid": np.zeros(5, dtype=bool),
        "fold_train_counts": np.zeros(5, dtype=int), "fold_test_counts": np.zeros(5, dtype=int),
        "fold_mse": np.full((5, k), np.nan), "fold_r2": np.full((5, k), np.nan),
        "mse": np.full(k, np.nan), "r2": np.full(k, np.nan),
        "fold_reasons": np.full(5, "No eligible bins"),
        "fold_r2_reason": np.full((5, k), "No eligible bins"), "r2_reason": np.full(k, "No eligible bins")}


def _fit_metadata(fitted: modeling.LinearFit) -> dict:
    """Input LinearFit with slopes(P,K); return scalar convention/penalty/limitation metadata, no arrays."""
    return {"estimator": fitted.estimator, "configured_alpha": fitted.configured_alpha,
        "effective_alpha": fitted.effective_alpha, "penalty_mode": fitted.penalty_mode,
        "coefficient_convention": fitted.coefficient_convention, "limitations": list(fitted.limitations)}


def _append_cell_log(run: Path, cell: dict, action: str, elapsed_s: float) -> None:
    """Append one cell's settings/counts/status/reasons to run Path; wall time is scalar seconds.

    Input cell contains native coefficient/prediction arrays, which are not
    duplicated in the log. Return None; plain JSON text preserves errors and
    scientific unavailability alongside execution progress.
    """
    meta = cell["meta"]
    record = {name: meta[name] for name in ("cell_id", "settings", "row_count", "trial_count",
        "configured_alpha", "effective_alpha", "cv_available", "cv_reason", "unavailable_reason")}
    record.update({"action": action, "status": "available" if meta["available"] else "unavailable",
                   "elapsed_s": elapsed_s, "fold_reasons": cell["fold_reasons"].tolist()})
    with (run / "run.log").open("a", encoding="ascii") as stream:
        stream.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def _validate_existing_cell(cell: dict, description: dict, identities: dict) -> None:
    """Check cell named-array meta against planned IDs/settings; return None, no axis/unit changes."""
    meta = cell["meta"]
    if (meta["cell_id"] != description["cell_id"] or meta["settings"] != description["settings"]
            or meta["penalty_aliases"] != description["penalty_aliases"]):
        raise ValueError("Existing published cell settings/identity differ from requested model")
    if any(meta.get(key) != identities[key] for key in ("preparation_id", "model_id")):
        raise ValueError("Existing published cell preparation/model identity differs")
