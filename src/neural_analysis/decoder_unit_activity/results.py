"""Validate saved decoder inputs and persist small, source-referenced selections."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from src.neural_analysis.population_decoding import results as decoder_results
from src.neural_analysis.population_decoding.preparation import PreparedPopulation, PreparedPopulationRegion
from .selection import GROUP_IDS, GROUP_LABELS, MODEL_NAMES, build_group_weights, summarize_group_weights, assign_unit_groups

ANALYSIS_VERSION = "decoder-unit-activity-v1"


def load_decoder_inputs(run_directory: Path | str, region: str,
                        *, prepared: PreparedPopulation | None = None) -> dict:
    """Load one trusted saved run/region; return ready/pending/unavailable input mapping.

    A ready mapping contains preparation (regional W,B,U Hz), coefficient cells,
    weights(5,10,U), source hashes, session_root, manifest and diagnostics. An
    unavailable/pending mapping has a reason and no fabricated weights. Optional
    prepared reuses an already loaded preparation with matching saved identity.
    No raw sources, fits or writes occur. Structural disagreement raises ValueError.
    """
    directory = Path(run_directory).resolve(strict=True)
    if region not in ("PFC", "HPC") or directory.parent.name != "analysis_runs":
        raise ValueError("Expected PFC/HPC and a saved run under its session's analysis_runs")
    manifest = decoder_results.load_manifest(directory)
    root = directory.parent.parent
    prep_path = Path(manifest["preparation_path"])
    if not prep_path.is_absolute():
        prep_path = directory / prep_path
    prepared = prepared if prepared is not None else decoder_results.load_preparation(prep_path)
    if prepared.meta["preparation_id"] != manifest["identities"]["preparation_id"]:
        raise ValueError("Saved preparation identity differs from source run")
    loaded = {"status": "ready", "reason": "", "directory": directory, "region": region,
        "session_root": root, "manifest": manifest, "prepared": prepared, "cells": {}}
    regional = prepared.regions.get(region)
    if regional is None:
        reason = prepared.unavailable.get(region)
        if not reason:
            raise ValueError("Missing region has no recorded unavailable reason")
        _validate_absent_region(directory, region, manifest)
        return {**loaded, "status": "unavailable", "reason": reason}
    if regional.probe_id != manifest["config"]["probe_roles"].get(region):
        raise ValueError("Source region/probe identities disagree")
    names = decoder_results.model_targets_for_run(manifest)
    requested = [name for name in MODEL_NAMES if name in names]
    if not set(MODEL_NAMES[:3]) <= set(requested):
        raise ValueError("Source run must declare context, choice and outcome models")
    missing = [name for name in requested if not (directory / "cells" / f"{region}_{name}.npz").is_file()]
    if missing:
        return {**loaded, "status": "pending", "reason": "Unpublished required cells: " + ", ".join(missing)}
    source_paths = [directory / "config.json", prep_path]
    for name in requested:
        path = directory / "cells" / f"{region}_{name}.npz"
        cell = decoder_results.load_cell(path)
        _validate_source_cell(cell, name, region, regional, manifest, prepared)
        # Public source readers validate predictions; retain only small direct-unit inputs.
        keep = {"meta", "class_codes", "labels", "row_positions", "bin_indices", "fold_ids"}
        for fold in range(5):
            keep.update(f"fold_{fold}_{suffix}" for suffix in ("retained_mask", "mean_hz", "scale_hz", "units_coefficients"))
        loaded["cells"][name] = {key: value for key, value in cell.items() if key in keep}
        source_paths.append(path)
    loaded["weights"] = build_group_weights(loaded["cells"], regional.unit_ids)
    loaded["diagnostics"] = source_diagnostics(loaded["cells"])
    loaded["sources"] = [{"path": os.path.relpath(path, root), "sha256": file_digest(path)} for path in source_paths]
    return loaded


def source_diagnostics(cells: dict[str, dict]) -> dict[str, list[dict]]:
    """Return saved model/outcome score rows with unique-trial versus required-bin support.

    cells contain labels(W,), original row_positions(W,), and source metric
    dictionaries. Scores stay dimensionless and unchanged; counts distinguish
    trials from pooled bin observations. No rescoring, filtering or fitting.
    """
    models, outcomes = [], []
    for name, cell in cells.items():
        summary = cell["meta"]["representations"]["units"]
        score = summary["pooled"]
        models.append({"Target": name, "Eligible trials": len(cell["labels"]),
            "Class trials": np.bincount(cell["labels"], minlength=len(cell["class_codes"])).tolist(),
            "Class bins": score["class_counts"], "Required bins": score["required_rows"],
            "Predicted bins": score["predicted_rows"], "Balanced accuracy": score["balanced_accuracy"],
            "Log loss": score["log_loss"], "Unavailable reasons": score["metric_reasons"]})
        if name == "outcome":
            for side, rewarded, absent in (("left", 0, 2), ("right", 1, 3)):
                conditional = summary["conditional_outcomes"][side]
                outcomes.append({"Side": side, "Reward trials": int(np.sum(cell["labels"] == rewarded)),
                    "No-reward trials": int(np.sum(cell["labels"] == absent)),
                    "Reward bins": conditional["class_counts"][1], "No-reward bins": conditional["class_counts"][0],
                    "Required bins": conditional["required_rows"], "Predicted bins": conditional["predicted_rows"],
                    "Balanced accuracy": conditional["balanced_accuracy"], "Log loss": conditional["log_loss"],
                    "Reward ROC-AUC": conditional["roc_auc"][1], "Unavailable reasons": conditional["metric_reasons"]})
    return {"models": models, "outcomes": outcomes}


def build_unit_selection(inputs: dict, max_units_per_group: int = 10) -> dict:
    """Compute a small selection from ready saved inputs, no writes or model fits.

    Return arrays, meta dictionary, summary(10*U rows), assignments(U rows).
    Arrays retain weights(5,10,U), dimensionless membership/ranks and selected
    display means/SDs(S,) in Hz. Activity remains solely in the source archive.
    """
    if inputs["status"] != "ready":
        raise ValueError("Selection requires ready saved decoder inputs")
    regional = inputs["prepared"].regions[inputs["region"]]
    weights = inputs["weights"]
    summary = summarize_group_weights(weights, GROUP_IDS, regional.unit_ids)
    assignments = assign_unit_groups(summary, max_units_per_group)
    display = displayed_units(assignments)
    indices = display.unit_index.to_numpy(dtype=int)
    rates = regional.rates_hz[:, :, indices]
    arrays = {"weights": weights.copy(), "unit_ids": np.asarray(regional.unit_ids), "group_ids": np.asarray(GROUP_IDS),
        "assigned_group_indices": assignments.group_index.to_numpy(dtype=int), "display_unit_indices": indices,
        "display_group_indices": display.group_index.to_numpy(dtype=int), "row_positions": regional.row_positions.copy(),
        "bin_edges_s": regional.bin_edges_s.copy(), "mean_hz": rates.mean(axis=(0, 1)),
        "scale_hz": rates.std(axis=(0, 1)), "constant_mask": np.ptp(rates, axis=(0, 1)) == 0}
    for name in ("mean_weight", "percentile", "available_folds", "positive_folds", "candidate"):
        arrays[name] = summary[name].to_numpy().reshape(10, len(regional.unit_ids))
    identity_payload = {"sources": inputs["sources"], "region": inputs["region"],
                        "cap": max_units_per_group, "code": code_identity(), "version": ANALYSIS_VERSION}
    selection_id = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode()).hexdigest()
    meta = {"analysis_version": ANALYSIS_VERSION, "schema_version": 1, "selection_id": selection_id,
        "generator": "decoder_unit_activity.results.build_unit_selection", "code_identity": identity_payload["code"],
        "source_model_id": inputs["manifest"]["identities"]["model_id"],
        "source_preparation_id": inputs["manifest"]["identities"]["preparation_id"],
        "source_run": os.path.relpath(inputs["directory"], inputs["session_root"]),
        "session_root": str(inputs["session_root"]), "session_id": inputs["prepared"].session_id,
        "region": inputs["region"], "probe_id": regional.probe_id, "sources": inputs["sources"],
        "max_units_per_group": max_units_per_group, "bin_width_ms": inputs["prepared"].meta["bin_width_ms"],
        "units": {"activity": "Hz", "display_mean_sd": "Hz", "weights": "logit contrast per training unit SD", "time": "seconds"},
        "axes": {"weights": ["fold", "group", "unit"], "activity_reference": ["window", "bin", "unit"]},
        "rules": {"required_folds": 5, "positive_folds": 4, "mean_positive": True,
                  "percentile": "(average_rank-0.5)/rankable_units", "assignment_before_cap": True},
        "group_labels": list(GROUP_LABELS), "diagnostics": inputs["diagnostics"],
        "fit_reasons": {name: [record["reason"] for record in cell["meta"]["representations"]["units"]["folds"]
                              if not record["available"]] for name, cell in inputs["cells"].items()},
        "runtime": decoder_results.runtime_versions(), "randomness_used": False, "random_seed": None}
    return {"arrays": arrays, "meta": meta, "summary": summary, "assignments": assignments}


def save_unit_selection(path: Path | str, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Save validated named numeric/string arrays and JSON meta to a new scalar NPZ path.

    Contracts match build_unit_selection. Existing files fail; a sibling
    temporary prevents interrupted archives from being published as complete.
    No pickle objects or raw activity are written. Return None.
    """
    _validate_selection(arrays, meta)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays, meta=np.array(json.dumps(meta, allow_nan=False, sort_keys=True)))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_unit_selection(path: Path | str) -> dict:
    """Load derived NPZ without pickle and validate shapes, membership and exact saved references.

    Return the same arrays/meta/summary/assignments mapping as build_unit_selection.
    No original recordings or fits are accessed. Corruption/stale references fail.
    """
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files if key != "meta"}
            meta = json.loads(str(archive["meta"].item()))
        summary, assignments = _validate_selection(arrays, meta)
        for reference in meta["sources"]:
            source = Path(meta["session_root"]) / reference["path"]
            if not source.is_file() or file_digest(source) != reference["sha256"]:
                raise ValueError(f"Changed or missing source reference: {source}")
        return {"arrays": arrays, "meta": meta, "summary": summary, "assignments": assignments}
    except (OSError, KeyError, TypeError, EOFError, BadZipFile) as error:
        raise ValueError(f"Invalid unit selection archive/reference: {error}") from error


def displayed_units(assignments: pd.DataFrame) -> pd.DataFrame:
    """Return selected rows(S,) in display order from full U assignment table, no mutation."""
    return assignments.loc[assignments.displayed.to_numpy(dtype=bool)].sort_values("display_order").reset_index(drop=True)


def file_digest(path: Path | str) -> str:
    """Hash a scalar file Path using bounded byte chunks; return SHA256 string, no writes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024*1024):
            digest.update(chunk)
    return digest.hexdigest()


def code_identity() -> str:
    """Fingerprint this derived package and the reused normalization source; return SHA256 string."""
    digest = hashlib.sha256()
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths.append(Path(__file__).parents[1] / "population/timeline_plotting.py")
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


# Validation helpers reject ordinary saved-input errors, without changing source products.
def _validate_absent_region(directory: Path, region: str, manifest: dict) -> None:
    """Check any published cells for a region absent from preparation; no available fits may contradict it."""
    for name in MODEL_NAMES:
        path = directory / "cells" / f"{region}_{name}.npz"
        if not path.is_file():
            continue
        meta = decoder_results.load_cell(path)["meta"]
        if (meta["region"] != region or meta["model_id"] != manifest["identities"]["model_id"]
                or meta["preparation_id"] != manifest["identities"]["preparation_id"]
                or any(record["available"] for record in meta["representations"]["units"]["folds"])):
            raise ValueError("Absent region contradicts published source identity or available fits")


def _validate_source_cell(cell: dict, name: str, region: str, regional: PreparedPopulationRegion,
                           manifest: dict, prepared: PreparedPopulation) -> None:
    """Check saved target identity and original row/native-bin geometry; return None or ValueError."""
    meta = cell["meta"]
    edges = regional.bin_edges_s
    bins = np.flatnonzero(edges[:-1] >= 0) if name == "outcome" else np.arange(len(edges)-1)
    rows, labels = cell["row_positions"], cell["labels"]
    if (meta["model_id"] != manifest["identities"]["model_id"]
            or meta["preparation_id"] != manifest["identities"]["preparation_id"]
            or meta["target"] != name or meta["region"] != region
            or meta["bin_width_ms"] != prepared.meta["bin_width_ms"]
            or meta["supervised_interval_s"] != ([0, 1] if name == "outcome" else [-1, 1])):
        raise ValueError("Saved source identity/target/window disagreement")
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any(np.diff(rows) <= 0)
            or not np.isin(rows, regional.row_positions).all() or labels.shape != rows.shape
            or not np.issubdtype(labels.dtype, np.integer) or not np.isin(labels, cell["class_codes"]).all()
            or not np.array_equal(cell["bin_indices"], bins)
            or not np.array_equal(cell["fold_ids"], prepared.fold_ids[rows])
            or not np.allclose(np.diff(edges), prepared.meta["bin_width_ms"]/1000)):
        raise ValueError("Saved target row/bin/fold axes disagree with preparation")


def _validate_selection(arrays: dict, meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate F,G,U/S named arrays and recomputed assignments; return two labeled tables."""
    if meta.get("analysis_version") != ANALYSIS_VERSION or meta.get("schema_version") != 1:
        raise ValueError("Unsupported unit selection version")
    summary = summarize_group_weights(arrays["weights"], tuple(arrays["group_ids"]), tuple(arrays["unit_ids"]))
    assigned = assign_unit_groups(summary, meta["max_units_per_group"])
    display = displayed_units(assigned)
    expected = {"assigned_group_indices": assigned.group_index.to_numpy(dtype=int),
        "display_unit_indices": display.unit_index.to_numpy(dtype=int),
        "display_group_indices": display.group_index.to_numpy(dtype=int)}
    for name in ("mean_weight", "percentile", "available_folds", "positive_folds", "candidate"):
        expected[name] = summary[name].to_numpy().reshape(10, len(arrays["unit_ids"]))
    for name, values in expected.items():
        if arrays[name].shape != values.shape or not np.array_equal(arrays[name], values, equal_nan=True):
            raise ValueError(f"Invalid saved selection {name}")
    count = len(display)
    if any(arrays[name].shape != (count,) for name in ("mean_hz", "scale_hz", "constant_mask")):
        raise ValueError("Invalid saved display normalization axes")
    if (not np.isfinite(arrays["mean_hz"]).all() or not np.isfinite(arrays["scale_hz"]).all()
            or (arrays["scale_hz"] < 0).any() or arrays["constant_mask"].dtype != bool):
        raise ValueError("Invalid saved display normalization values")
    return summary, assigned
