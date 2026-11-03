"""Ordinary named-array persistence and scientific identities for context encoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd

from src.neural_analysis.encoding.configuration import ANALYSIS_VERSION, EncodingConfig
from src.neural_analysis.encoding.preparation import PreparedRegion
from src.neural_analysis.interregional.records import RegionalPCATransform
from src.neural_analysis.population.session_timeline import RegionalTimelinePCA, SessionTimelineResult
from src.neural_analysis.population import session_timeline_io
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata


SCHEMA_VERSION = 1
REPOSITORY = Path(__file__).resolve().parents[3]
PREPARATION_SOURCES = (
    "src/neural_analysis/encoding/configuration.py", "src/neural_analysis/encoding/predictors.py",
    "src/neural_analysis/encoding/preparation.py", "src/neural_analysis/encoding/results.py",
    "src/behavior_analysis/project_utils.py", "src/neural_analysis/session_metadata.py",
    "src/neural_analysis/population/session_timeline.py", "src/neural_analysis/population/session_timeline_io.py",
    "src/neural_analysis/population/pca.py", "src/neural_analysis/interregional/pca.py",
    "src/neural_analysis/interregional/records.py", "src/neural_analysis/spike_behavior/loading.py",
    "src/neural_analysis/task_decoding/activity.py",
)
MODELING_SOURCES = (
    "src/neural_analysis/encoding/modeling.py", "src/neural_analysis/encoding/pipeline.py",
    "src/neural_analysis/encoding/configuration.py", "src/neural_analysis/encoding/results.py",
)


# Public control-file and identity functions.
def serialize_config(config: EncodingConfig) -> dict:
    """Input EncodingConfig; return JSON-compatible paths/selections/scalars, no arrays or units changed."""
    payload = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if isinstance(value, Path):
            value = str(value)
        elif isinstance(value, tuple):
            value = list(value)
        elif field.name == "probe_roles":
            value = dict(value)
        payload[field.name] = value
    return payload


def restore_config(payload: dict) -> EncodingConfig:
    """Restore saved JSON config to Path/tuple fields without checking live files.

    Input is a serialize_config dict of scalar paths and scientific controls.
    Return EncodingConfig; bin milliseconds/window seconds/penalty remain native.
    Used only for trusted saved manifests, never to validate new user input.
    """
    values = dict(payload)
    for name in ("session_metadata", "augmented_trials", "feature_parameters", "session_root",
                 "intermediate_root", "run_root", "timeline_result"):
        values[name] = None if values.get(name) is None else Path(values[name])
    for name in ("windows", "timing_modes", "encodings", "estimators", "penalty_modes", "window_s"):
        values[name] = tuple(values[name])
    return EncodingConfig(**values)


def numerical_code_identity() -> dict[str, dict[str, str]]:
    """Return preparation/modeling maps of used repository-relative Python paths to SHA256 strings.

    No inputs or physical units. This hashes the small numerical/source-loading
    modules actually used; unrelated application and synchronization edits are
    excluded. Scopes let penalty changes reuse unchanged preparation.
    """
    return {scope: {name: _sha256_file(REPOSITORY / name) for name in names}
            for scope, names in (("preparation", PREPARATION_SOURCES), ("modeling", MODELING_SOURCES))}


def numerical_runtime_versions() -> dict[str, str]:
    """Return Python/numerical package versions used in scientific identity; no inputs, arrays or units."""
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "pynapple"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def compute_run_identities(config: EncodingConfig) -> dict:
    """Read small source records and produce separate preparation/model identities.

    Input config carries file paths, native bin milliseconds and dimensionless
    model controls. Return JSON-compatible preparation_id/model_id, scoped
    code_identity, input_records and runtime_versions. Small control/table files
    are hashed; aligned spikes/assignments/timeline arrays use path/size/mtime
    records. No raw spike array, PCA or model is loaded or fitted.
    """
    session = resolve_session_metadata(load_session_metadata(config.session_metadata), config.session_metadata)
    small = {config.session_metadata, config.augmented_trials, config.feature_parameters}
    small.update(path for path in (session.behavior.trial_table_file, session.behavior.event_table_file) if path is not None)
    large = set()
    for probe in session.probes:
        if probe.probe_id not in config.probe_roles.values():
            continue
        small.update(path for path in (probe.channel_quality_file, probe.cluster_info_file) if path is not None)
        large.update(path for path in (probe.alignment_file, probe.spike_clusters_file) if path is not None)
    if config.timeline_result is not None:
        large.add(config.timeline_result)
    records = [_source_record(path, path in small) for path in sorted(small | large)]
    code = numerical_code_identity()
    runtime = numerical_runtime_versions()
    source_settings = session_timeline_io.timeline_configuration(session, dict(config.probe_roles), config.bin_width_ms / 1000)
    preparation_id = _digest({"analysis_version": ANALYSIS_VERSION, "code": code["preparation"],
        "inputs": records, "source_settings": source_settings, "runtime_versions": runtime, "window_s": config.window_s,
        "bin_width_ms": config.bin_width_ms, "probe_roles": dict(config.probe_roles)})
    model_settings = {name: getattr(config, name) for name in (
        "windows", "timing_modes", "encodings", "estimators", "alpha", "penalty_modes")}
    model_id = _digest({"preparation_id": preparation_id, "code": code["modeling"],
                        "runtime_versions": runtime, "settings": model_settings})
    return {"preparation_id": preparation_id, "model_id": model_id, "code_identity": code,
            "input_records": records, "runtime_versions": runtime}


def initialize_run(
    config: EncodingConfig, run_directory: Path, preparation_path: Path, identities: dict,
) -> dict:
    """Create a new run manifest and initial progress before neural preparation.

    Inputs are a validated config, new run directory under session/analysis_runs,
    immutable NPZ path under session/processed/context_encoding, and identity
    dict. Return JSON manifest with session-relative preparation_path. Existing
    run paths and repository/out-of-session outputs are rejected before writing.
    Time/array units are unchanged; no source arrays are loaded.
    """
    root = config.session_root.resolve()
    run, prep = Path(run_directory).resolve(), Path(preparation_path).resolve()
    if root.is_relative_to(REPOSITORY):
        raise ValueError("Analysis output session must be outside the source repository")
    if not run.is_relative_to(root / "analysis_runs") or run == root / "analysis_runs":
        raise ValueError("Run output must be inside this session's analysis_runs")
    if not prep.is_relative_to(root / "processed/context_encoding") or prep.suffix != ".npz":
        raise ValueError("Preparation output must be an NPZ inside this session's processed/context_encoding")
    if run.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run}")
    manifest = {"schema_version": SCHEMA_VERSION, "analysis_version": ANALYSIS_VERSION,
        "config": serialize_config(config), "session_root": str(root), "run_directory": str(run),
        "preparation_path": str(prep.relative_to(root)), "identities": identities}
    run.mkdir(parents=True, exist_ok=False)
    _write_json(run / "config.json", manifest)
    write_progress(run, {"status": "preparing", "completed_cells": {}})
    return manifest


def load_run_manifest(run_directory: Path | str) -> dict:
    """Input saved run Path/string; return version-checked JSON manifest, no source reads or units."""
    path = Path(run_directory) / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    _check_version(payload)
    return payload


def write_progress(run_directory: Path | str, progress: dict) -> None:
    """Atomically replace run progress from Path and JSON dict; return None, no scientific units."""
    _write_json(Path(run_directory) / "progress.json", progress)


# Public named-array publication and read-only loading.
def save_preparation(
    path: Path | str, trials: pd.DataFrame, behavior: SessionTimelineResult,
    regions: dict[str, PreparedRegion], meta: dict,
) -> Path:
    """Publish one immutable snapshot containing full behavior and regional bases.

    Inputs: new NPZ Path; complete trials (T,C); behavior with choices/events
    in UTC/elapsed seconds; regions with scores (W,B,K), native unit scaling,
    original ranks and (W,B) context values; provenance meta dictionary.
    Return saved Path. Named arrays preserve shapes/units; meta is a dictionary
    called 'meta'. A temporary sibling is published before any run-index update.
    Existing files raise FileExistsError and remain unchanged.
    """
    metadata = _archive_meta(meta, "preparation")
    metadata.update({"session_id": behavior.session_id, "time_origin_s": behavior.time_origin_s,
        "behavior_meta": behavior.meta, "behavior_unavailable": behavior.unavailable, "regions": {},
        "units": {"absolute_time": "UTC Unix seconds", "relative_time": "seconds",
                  "scores": "unwhitened standardized-unit scores", "counts": "spike counts per bin"},
        "axes": {"scores": ["window", "bin", "original_pc"], "context": ["window", "bin"]}})
    arrays = {"trials": trials.to_records(index=False), "choices": behavior.choices.to_records(index=False),
              "events": behavior.events.to_records(index=False)}
    for role, prepared in regions.items():
        regional, transform = prepared.timeline, prepared.timeline.transform
        prefix = role.lower() + "_"
        metadata["regions"][role] = {"probe_id": regional.probe_id,
            "retained_unit_ids": transform.retained_unit_ids, "omitted_unit_ids": transform.omitted_unit_ids,
            "input_unit_ids": transform.input_unit_ids, "n_training_observations": transform.n_training_observations,
            "total_session_variance": regional.total_session_variance, "color_limit": regional.color_limit,
            "timings_s": regional.timings_s, "cv_unavailable_reason": prepared.cv_unavailable_reason,
            "reused": prepared.reused, "meta": prepared.meta, "timing_modes": list(prepared.context_values)}
        for name in ("scores", "choice_positions", "bin_edges_s", "window_total_variance"):
            arrays[prefix + name] = getattr(regional, name)
        for name in ("components", "training_mean", "training_scale", "explained_variance", "explained_variance_ratio"):
            arrays[prefix + name] = getattr(transform, name)
        arrays[prefix + "pc_ranks"] = prepared.pc_ranks
        arrays[prefix + "window_fold_ids"] = prepared.window_fold_ids
        arrays[prefix + "bin_rows"] = prepared.bin_rows.to_records(index=False)
        for mode in prepared.context_values:
            arrays[prefix + mode + "_values"] = prepared.context_values[mode]
            arrays[prefix + mode + "_valid"] = prepared.context_valid[mode]
    arrays["meta"] = np.asarray(metadata, dtype=object)
    return _publish_npz(Path(path), arrays)


def load_preparation(path: Path | str) -> dict:
    """Load trusted saved NPZ Path; return trials/behavior/regions/meta with original shapes and units.

    Complete trials (T,C), UTC origin/events and fixed scores (W,B,K) are restored
    without raw source access. Invalid versions, identities or array axes raise
    ValueError. No saved arrays/files are modified.
    """
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        _check_version(meta, "preparation")
        trials = pd.DataFrame.from_records(archive["trials"])
        choices = pd.DataFrame.from_records(archive["choices"])
        events = pd.DataFrame.from_records(archive["events"])
        regions = {role: _load_region(archive, role, info) for role, info in meta["regions"].items()}
    behavior = SessionTimelineResult(meta["session_id"], meta["time_origin_s"], choices, events,
        {role: region.timeline for role, region in regions.items()}, meta["behavior_unavailable"], meta["behavior_meta"])
    return {"trials": trials, "behavior": behavior, "regions": regions, "meta": meta}


def save_cell(path: Path | str, cell: dict) -> Path:
    """Publish new cell NPZ from named arrays and meta dict; return Path without overwriting.

    Cell arrays use N selected bins, P native predictor columns and K original
    PC responses; slopes(P,K), predictions(N,K), fold slopes(5,P,K). Raw scores
    remain in shared preparation. Undefined outputs stay NaN with meta reasons.
    """
    arrays = {name: value for name, value in cell.items() if name != "meta"}
    meta = _archive_meta(cell["meta"], "cell")
    checked = {**arrays, "meta": meta}
    _validate_cell(checked)
    arrays["meta"] = np.asarray(meta, dtype=object)
    return _publish_npz(Path(path), arrays)


def load_cell(path: Path | str) -> dict:
    """Input trusted cell NPZ Path; return named arrays/meta with native axes/units, validating shapes."""
    with np.load(path, allow_pickle=True) as archive:
        cell = {name: archive[name] for name in archive.files if name != "meta"}
        cell["meta"] = archive["meta"].item()
    _check_version(cell["meta"], "cell")
    _validate_cell(cell)
    return cell


def inspect_run_status(run_directory: Path | str) -> dict:
    """Read progress and validate published named cells one at a time; return JSON dict.

    Input is a run directory Path/string. Published preparation metadata is
    checked without reading scores. No raw sources or fitting are needed.
    completed_cells maps IDs to run-relative NPZ names;
    scientific unavailability does not change completed execution status.
    """
    run = Path(run_directory)
    manifest = load_run_manifest(run)
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    path = Path(manifest["session_root"]) / manifest["preparation_path"]
    if path.exists() or progress["status"] != "preparing":
        with np.load(path, allow_pickle=True) as archive:
            meta = archive["meta"].item()
        _check_version(meta, "preparation")
        if meta.get("preparation_id") != manifest["identities"]["preparation_id"]:
            raise ValueError("Published preparation identity differs from run manifest")
    for cell_id, relative in progress["completed_cells"].items():
        cell = load_cell(run / relative)
        _check_cell_identity(cell["meta"], cell_id, manifest)
    return progress


def load_encoding_run(run_directory: Path | str, cell_ids: Sequence[str] | None = None) -> dict:
    """Read a saved run, optionally loading only selected model cells for the UI.

    Input run Path and optional (C_selected,) cell-ID sequence. Return manifest,
    restored config, immutable preparation, progress and cell-ID mapping.
    Each cell's observed(N,K) is derived from shared scores and selected row/PC
    IDs rather than duplicated on disk. All responses remain in PC-score units.
    No live metadata, CSVs, spike arrays or estimators are accessed.
    """
    run = Path(run_directory)
    manifest = load_run_manifest(run)
    config = restore_config(manifest["config"])
    progress = json.loads((run / "progress.json").read_text(encoding="utf-8"))
    prepared = load_preparation(Path(manifest["session_root"]) / manifest["preparation_path"])
    if prepared["meta"]["preparation_id"] != manifest["identities"]["preparation_id"]:
        raise ValueError("Saved preparation identity differs from run manifest")
    selected = list(progress["completed_cells"]) if cell_ids is None else list(cell_ids)
    cells = {}
    for cell_id in selected:
        if cell_id not in progress["completed_cells"]:
            raise ValueError(f"Requested cell is not completed: {cell_id}")
        cell = load_cell(run / progress["completed_cells"][cell_id])
        _check_cell_identity(cell["meta"], cell_id, manifest)
        region = prepared["regions"][cell["meta"]["settings"]["region"]]
        scores = region.timeline.scores.reshape(-1, region.timeline.scores.shape[-1])
        indices, ranks = cell["selected_row_indices"], cell["pc_ranks"]
        if np.any(indices < 0) or np.any(indices >= len(scores)) or not np.array_equal(ranks, region.pc_ranks):
            raise ValueError("Cell observation/PC identity differs from preparation")
        cell["observed"] = scores[indices][:, ranks - 1]
        cells[cell_id] = cell
    return {"manifest": manifest, "config": config, "preparation": prepared, "progress": progress, "cells": cells}


# Small archive/identity helpers.
def _load_region(archive: object, role: str, info: dict) -> PreparedRegion:
    """Restore one regional NPZ prefix to fixed scores(W,B,K)/native-scale PreparedRegion."""
    prefix = role.lower() + "_"
    transform = RegionalPCATransform(
        retained_unit_ids=tuple(info["retained_unit_ids"]), omitted_unit_ids=tuple(info["omitted_unit_ids"]),
        input_unit_ids=tuple(info["input_unit_ids"]), n_training_observations=info["n_training_observations"],
        **{name: archive[prefix + name] for name in ("components", "training_mean", "training_scale",
            "explained_variance", "explained_variance_ratio")})
    regional = RegionalTimelinePCA(role, info["probe_id"], archive[prefix + "choice_positions"],
        archive[prefix + "bin_edges_s"], archive[prefix + "scores"], transform,
        info["total_session_variance"], archive[prefix + "window_total_variance"], info["color_limit"], info["timings_s"])
    ranks, folds = archive[prefix + "pc_ranks"], archive[prefix + "window_fold_ids"]
    rows = pd.DataFrame.from_records(archive[prefix + "bin_rows"])
    values = {mode: archive[prefix + mode + "_values"] for mode in info["timing_modes"]}
    valid = {mode: archive[prefix + mode + "_valid"] for mode in info["timing_modes"]}
    shape = regional.scores.shape
    if (len(shape) != 3 or not np.isfinite(regional.scores).all() or len(rows) != shape[0] * shape[1]
            or folds.shape != (shape[0],) or len(regional.bin_edges_s) != shape[1] + 1
            or shape[2] != transform.actual_components or regional.choice_positions.shape != (shape[0],)
            or ranks.ndim != 1 or not np.issubdtype(ranks.dtype, np.integer)
            or np.any(ranks < 1) or np.any(ranks > shape[2])):
        raise ValueError("Saved preparation regional array shapes/PC identities are invalid")
    if any(values[mode].shape != shape[:2] or valid[mode].shape != shape[:2] for mode in values):
        raise ValueError("Saved context shape differs from regional window/bin axes")
    return PreparedRegion(regional, ranks, rows, values, valid, folds, info["cv_unavailable_reason"], info["reused"], info["meta"])


def _validate_cell(cell: dict) -> None:
    """Validate named cell arrays with N bins/P predictors/K PCs; no units changed, return None."""
    meta = cell["meta"]
    n, p, k = meta["row_count"], meta["predictor_count"], meta["response_count"]
    if any(type(value) is not int for value in (n, p, k)) or n < 0 or p < 1 or k < 1:
        raise ValueError("Saved cell row/predictor/response counts must be valid integers")
    shapes = {"selected_row_indices": (n,), "pc_ranks": (k,), "predictors": (n, p),
        "reference_slopes": (p, k), "reference_intercept": (k,), "standardized_slopes": (p, k),
        "reference_predictions": (n, k), "held_out_predictions": (n, k),
        "fold_slopes": (5, p, k), "fold_intercepts": (5, k), "fold_ids": (n,),
        "mse": (k,), "r2": (k,), "fold_mse": (5, k), "fold_r2": (5, k),
        "fold_valid": (5,), "fold_train_counts": (5,), "fold_test_counts": (5,),
        "fold_reasons": (5,), "fold_r2_reason": (5, k), "r2_reason": (k,), "column_names": (p,)}
    for name, shape in shapes.items():
        if name not in cell or np.asarray(cell[name]).shape != shape:
            raise ValueError(f"Saved cell {name} shape must be {shape}")
    for name in ("selected_row_indices", "pc_ranks", "fold_ids", "fold_train_counts", "fold_test_counts"):
        if not np.issubdtype(cell[name].dtype, np.integer):
            raise ValueError(f"Saved {name} identities/counts must be integer")
    if (np.any(cell["selected_row_indices"] < 0) or len(np.unique(cell["selected_row_indices"])) != n
            or np.any(cell["pc_ranks"] < 1) or len(np.unique(cell["pc_ranks"])) != k
            or not np.isin(cell["fold_ids"], [-1, 0, 1, 2, 3, 4]).all()):
        raise ValueError("Saved row/PC/fold identity values are invalid")
    if cell["fold_valid"].dtype != np.dtype(bool):
        raise ValueError("Saved fold_valid flags must be Boolean")
    for name in ("fold_train_counts", "fold_test_counts"):
        if np.any(cell[name] < 0) or np.any(cell[name] > n):
            raise ValueError(f"Saved {name} must be nonnegative bin counts")
    numeric = ("predictors", "reference_slopes", "reference_intercept", "standardized_slopes",
               "reference_predictions", "held_out_predictions", "fold_slopes", "fold_intercepts",
               "mse", "r2", "fold_mse", "fold_r2")
    for name in numeric:
        if not np.issubdtype(cell[name].dtype, np.number) or np.isinf(cell[name]).any():
            raise ValueError(f"Saved {name} must be numeric and cannot contain infinite values")
    if not np.isfinite(cell["predictors"]).all():
        raise ValueError("Saved native predictors must be finite")
    if meta["available"] and (not np.isfinite(cell["reference_slopes"]).all()
            or not np.isfinite(cell["reference_intercept"]).all()
            or not np.isfinite(cell["reference_predictions"]).all()):
        raise ValueError("Available cell reference predictions/coefficients must be finite")
    if meta["cv_available"] and (not meta["available"] or not cell["fold_valid"].all()
            or not np.isfinite(cell["held_out_predictions"]).all()
            or not np.isfinite(cell["mse"]).all()):
        raise ValueError("CV-available cells require finite held-out predictions and MSE from all five folds")


def _check_cell_identity(meta: dict, cell_id: str, manifest: dict) -> None:
    """Input cell meta/ID/run manifest; require matching scientific identities, return None, no units."""
    if meta.get("cell_id") != cell_id:
        raise ValueError("Published cell identity differs from progress")
    for key in ("preparation_id", "model_id"):
        if meta.get(key) != manifest["identities"][key]:
            raise ValueError(f"Published cell {key} identity differs from run")


def _archive_meta(meta: dict, kind: str) -> dict:
    """Input provenance dict/archive-kind string; return versioned dict, no shape/unit conversion."""
    return {**meta, "schema_version": SCHEMA_VERSION, "analysis_version": ANALYSIS_VERSION, "kind": kind,
            "generator": meta.get("generator", "src.neural_analysis.encoding"),
            "random_seed": meta.get("random_seed"), "randomness_used": False}


def _check_version(meta: dict, kind: str | None = None) -> None:
    """Validate scalar archive/manifest version fields; return None or ValueError, no units."""
    if (not isinstance(meta, dict) or meta.get("schema_version") != SCHEMA_VERSION
            or meta.get("analysis_version") != ANALYSIS_VERSION or (kind is not None and meta.get("kind") != kind)):
        raise ValueError("Unsupported encoding schema/analysis version or archive kind")


def _publish_npz(path: Path, arrays: dict) -> Path:
    """Publish named arrays/meta to a new NPZ Path via temporary sibling; preserve shapes/units."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if path.suffix != ".npz":
        raise ValueError("Published encoding arrays must use an .npz filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp.npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _write_json(path: Path, payload: dict) -> None:
    """Atomically write scalar Path/JSON dict; return None, no arrays or physical units."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", encoding="ascii", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    """Input small file Path; return SHA256 hex string without numerical shape/unit conversion."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_record(path: Path, hash_contents: bool) -> dict:
    """Input source Path/hash flag; return path/bytes/mtime_ns and optional small-file hash, no arrays."""
    resolved = path.resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    record = {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if hash_contents:
        record["sha256"] = _sha256_file(resolved)
    return record


def _digest(payload: dict) -> str:
    """Input canonicalizable control dict; return deterministic SHA256 identity string, no physical units."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()[:20]


