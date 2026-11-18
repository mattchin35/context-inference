"""Versioned named-array products and simple immutable publication boundaries."""

from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path
from typing import TYPE_CHECKING
import platform
import pickle
from importlib.metadata import version
import tempfile
import os
from zipfile import BadZipFile
import numpy as np
import pandas as pd
import sklearn

from .configuration import (ANALYSIS_VERSION, LEGACY_ANALYSIS_VERSION, READABLE_ANALYSIS_VERSIONS,
                            SCHEMA_VERSION, PopulationDecodingConfig)
from .targets import TARGET_NAMES, MODEL_TARGET_NAMES, CONDITIONAL_SWITCH_TARGETS

if TYPE_CHECKING:
    from .preparation import PreparedPopulation


def save_preparation(path: Path, prepared: PreparedPopulation, identities: dict) -> None:
    """Publish prepared session Hz/identity arrays once at scalar NPZ Path; return None.

    Tables are JSON strings with explicit field schemas. meta is a dictionary
    with versions, source/code identities, units and randomness provenance.
    Existing files are never replaced; no sklearn estimators are serialized.
    """
    meta = {**prepared.meta, **identities, "session_id": prepared.session_id,
            "time_origin_s": prepared.behavior.time_origin_s, "unavailable": prepared.unavailable,
            "regions": {}, "generator": "population_decoding.results.save_preparation",
            "units": "rates_hz: Hz; times: seconds UTC/relative/elapsed",
            "axes": "regional activity: window,bin,unit", "randomness_used": False,
            "random_seed": prepared.meta.get("random_seed")}
    arrays = {"baseline": prepared.baseline, "targets": prepared.targets.values,
              "target_valid": prepared.targets.valid, "target_reasons": prepared.targets.reasons,
              "fold_ids": prepared.fold_ids}
    for name, frame in (("trials", prepared.trials), ("choices", prepared.behavior.choices), ("events", prepared.behavior.events)):
        arrays[f"{name}_json"] = np.array(frame.to_json(orient="table", index=False, double_precision=15))
    for role, region in prepared.regions.items():
        meta["regions"][role] = {"probe_id": region.probe_id, "unit_ids": list(region.unit_ids), "meta": region.meta}
        for name in ("rates_hz", "row_positions", "bin_edges_s"):
            arrays[f"{role}_{name}"] = getattr(region, name)
        arrays[f"{role}_bin_rows_json"] = np.array(region.bin_rows.to_json(orient="table", index=False, double_precision=15))
    _save_archive(Path(path), arrays, meta)


def load_preparation(path: Path) -> PreparedPopulation:
    """Load trusted session-local preparation Path; return self-contained PreparedPopulation.

    Rates retain (W,B,U) Hz and source row identities; metadata must match the
    current schema. No original source files or estimator fitting are accessed.
    """
    from .preparation import PreparedPopulation, PreparedPopulationRegion
    from .targets import PopulationTargets
    from src.neural_analysis.population.session_timeline import SessionTimelineResult

    data = _load_archive(Path(path))
    meta = data["meta"]
    tables = {name: pd.read_json(StringIO(str(data[f"{name}_json"].item())), orient="table")
              for name in ("trials", "choices", "events")}
    n = len(tables["trials"])
    for name, shape in (("targets", (n, 5)), ("target_valid", (n, 5)), ("target_reasons", (n, 5)),
                        ("baseline", (n,)), ("fold_ids", (n,))):
        if data[name].shape != shape:
            raise ValueError(f"Invalid saved {name} axes")
    if (data["baseline"].dtype != bool or data["target_valid"].dtype != bool
            or not np.issubdtype(data["fold_ids"].dtype, np.integer)
            or not np.isin(data["fold_ids"], [-1, 0, 1, 2, 3, 4]).all()
            or not np.array_equal(data["target_valid"], data["targets"] >= 0)):
        raise ValueError("Invalid saved target validity/fold axes")
    regions = {}
    for role, info in meta["regions"].items():
        rates, rows, edges = [data[f"{role}_{name}"] for name in ("rates_hz", "row_positions", "bin_edges_s")]
        if (rates.ndim != 3 or rates.shape != (len(rows), len(edges)-1, len(info["unit_ids"]))
                or not np.isfinite(rates).all() or (rates < 0).any()
                or not np.issubdtype(rows.dtype, np.integer) or np.any((rows < 0) | (rows >= n))
                or np.any(np.diff(rows) <= 0) or not np.all(np.diff(edges) > 0)
                or len(set(info["unit_ids"])) != len(info["unit_ids"])
                or not np.isclose(edges[0], -1) or not np.isclose(edges[-1], 1)):
            raise ValueError("Invalid saved regional activity shape/units/identity")
        bin_rows = pd.read_json(StringIO(str(data[f"{role}_bin_rows_json"].item())), orient="table")
        regions[role] = PreparedPopulationRegion(role, info["probe_id"], rates, tuple(info["unit_ids"]),
                                                 rows, edges, bin_rows, info["meta"])
    behavior = SessionTimelineResult(meta["session_id"], meta["time_origin_s"], tables["choices"], tables["events"], {}, {}, {})
    return PreparedPopulation(meta["session_id"], tables["trials"], behavior, data["baseline"],
        PopulationTargets(data["targets"], data["target_valid"], data["target_reasons"]), data["fold_ids"],
        regions, meta["unavailable"], meta)


def save_cell(path: Path, cell: dict, identities: dict) -> None:
    """Publish one paired target cell's named arrays/meta; refuse existing scalar Path.

    Probability axes are (eligible_window,supervised_bin,class); parameters are
    separate per fold. Metadata includes the generating function and provenance.
    """
    meta = {**cell["meta"], **identities, "generator": "population_decoding.pipeline.fit_target",
            "probability_axes": ["window", "supervised_bin", "class"],
            "parameter_units": "logits per standardized-unit/PC-score unit; Hz scaling",
            "randomness_used": False, "random_seed": None}
    _save_archive(Path(path), {key: value for key, value in cell.items() if key != "meta"}, meta)


def load_cell(path: Path) -> dict:
    """Read trusted scalar cell Path; return named arrays/meta after axis/version checks."""
    cell = _load_archive(Path(path))
    c = len(cell["meta"]["class_names"])
    expected = (len(cell["row_positions"]), len(cell["bin_indices"]), c)
    if any(cell[f"{representation}_probabilities"].shape != expected for representation in ("units", "pcs")):
        raise ValueError("Invalid saved probability axes")
    if cell["meta"]["target"] in CONDITIONAL_SWITCH_TARGETS:
        _validate_switch_comparison(cell)
    return cell


def _validate_switch_comparison(cell: dict) -> None:
    """Validate saved conditional cell's W,B,2 scope/count/provenance metadata; return None.

    Counts describe dimensionless trial/bin observations. This reader checks
    structure and required support without loading reference data or rescoring.
    """
    comparison = cell["meta"].get("switch_comparison")
    mapping = cell.get("comparison_reference_window_indices")
    windows, bins = len(cell["row_positions"]), len(cell["bin_indices"])
    side = "left" if cell["meta"]["target"] == "left_switch_vs_stay" else "right"
    if (not isinstance(comparison, dict) or mapping is None or mapping.shape != (windows,)
            or not np.issubdtype(mapping.dtype, np.integer) or np.any(mapping < 0)
            or np.any(np.diff(mapping) <= 0) or comparison.get("required_trials") != windows
            or comparison.get("reference_target") != f"{side}_choice_switch"
            or comparison.get("reference_model_id") != cell["meta"].get("model_id")
            or comparison.get("reference_cell") != f"cells/{cell['meta']['region']}_{side}_choice_switch.npz"
            or comparison.get("conditioning_side") != side or cell["meta"].get("conditioning_side") != side
            or comparison.get("class_names") != ["stay", "switch"]
            or cell["meta"]["class_names"] != ["stay", "switch"]):
        raise ValueError("Invalid saved switch comparison mapping/provenance")
    try:
        for representation in ("units", "pcs"):
            record = comparison["representations"][representation]
            total = record["reference_all_trials"]["trial_count"]
            if total < windows or (windows and mapping[-1] >= total):
                raise ValueError("Invalid saved switch comparison reference support")
            for scope in ("reference_all_trials", "reference_matched_side", "within_side"):
                count = total if scope == "reference_all_trials" else windows
                if record[scope]["trial_count"] != count:
                    raise ValueError("Invalid saved switch comparison trial support")
                _validate_comparison_score(record[scope]["scores"], count * bins)
            if [fold["fold"] for fold in record["folds"]] != list(range(5)):
                raise ValueError("Invalid saved switch comparison folds")
            for fold in record["folds"]:
                count = int(np.sum(cell["fold_ids"] == fold["fold"]))
                if fold["trial_count"] != count:
                    raise ValueError("Invalid saved switch comparison fold support")
                for scope in ("reference_matched_side", "within_side"):
                    _validate_comparison_score(fold[scope], count * bins)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Invalid saved switch comparison summaries") from error


def _validate_comparison_score(scores: dict, required_bins: int) -> None:
    """Check binary score dict against integer required bin count; return None, no rescoring."""
    if (scores["required_rows"] != required_bins or not 0 <= scores["predicted_rows"] <= required_bins
            or len(scores["class_counts"]) != 2 or sum(scores["class_counts"]) != required_bins
            or any(len(scores[name]) != 2 for name in ("prevalence", "roc_auc", "average_precision"))):
        raise ValueError("Invalid saved switch comparison score support")


def compute_identities(config: PopulationDecodingConfig) -> dict:
    """Fingerprint small controls/source provenance and code/runtime; return string IDs.

    Small CSV/JSON inputs are hashed; large spike/alignment inputs use absolute
    path, byte count and nanosecond mtime. No neural arrays are read. Identity
    changes select new immutable preparation/run locations, never overwrite.
    """
    from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata, resolve_probe_sources
    session = resolve_session_metadata(load_session_metadata(config.session_metadata), config.session_metadata)
    paths = [config.session_metadata, config.augmented_trials, session.behavior.trial_table_file,
             session.behavior.event_table_file]
    for probe in config.probe_roles.values():
        source = resolve_probe_sources(session, probe)
        paths.extend(getattr(source, field) for field in ("alignment_file", "cluster_info_file", "channel_quality_file", "spike_clusters_file"))
    sources = []
    for value in paths:
        path = Path(value) if value is not None else None
        if path is None or not path.is_file():
            sources.append({"path": str(path), "missing": True})
            continue
        stat = path.stat()
        item = {"path": str(path), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if path.suffix in (".json", ".csv", ".tsv"):
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources.append(item)
    runtime = runtime_versions()
    code = code_identity()
    payload = {"config": config_payload(config), "sources": sources, "code": code, "runtime": runtime}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"preparation_id": identity, "model_id": identity, "code_identity": code,
            "runtime_identity": hashlib.sha256(json.dumps(runtime, sort_keys=True).encode()).hexdigest(),
            "source_provenance": sources, "runtime": runtime}


def code_identity() -> str:
    """Hash package and directly reused numerical Python sources; return hexadecimal string."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [root / name for name in ("encoding/predictors.py", "encoding/preparation.py",
        "population/session_timeline.py", "population/session_timeline_io.py", "population/pca.py",
        "task_decoding/activity.py", "task_decoding/pooled/decoding.py", "session_metadata.py")]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def runtime_versions() -> dict:
    """Return installed numerical runtime version strings; no data or randomness."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "scipy": version("scipy"), "pynapple": version("pynapple")}


def config_payload(config: PopulationDecodingConfig) -> dict:
    """Serialize config control paths/roles/bin width to portable JSON-compatible dict."""
    return {"analysis_version": config.analysis_version, "session_metadata": str(config.session_metadata),
            "augmented_trials": str(config.augmented_trials), "probe_roles": dict(config.probe_roles),
            "bin_width_ms": config.bin_width_ms}


def discover_runs(run_root: Path) -> list[Path]:
    """Return newest-first saved decoder directories under scalar run-root Path; no sources."""
    return sorted((path for path in Path(run_root).glob("population_task_decoding_*")
                   if path.is_dir() and (path / "config.json").is_file()), reverse=True)


def load_manifest(directory: Path) -> dict:
    """Read run-directory config.json; return version-checked manifest dict, no sources."""
    manifest = json.loads((Path(directory) / "config.json").read_text())
    if manifest.get("analysis_version") not in READABLE_ANALYSIS_VERSIONS or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported population decoding manifest version")
    model_targets_for_run(manifest)
    return manifest


def model_targets_for_run(manifest: dict) -> tuple[str, ...]:
    """Return ordered model IDs declared by v1/v2 saved manifest, without accessing sources.

    Legacy manifests infer the five-model set from cell identities. Extended
    manifests explicitly declare seven models. Unknown IDs/versions fail rather
    than being reinterpreted using the installed default target list.
    """
    version_name = manifest.get("analysis_version")
    if version_name not in READABLE_ANALYSIS_VERSIONS:
        raise ValueError("Unsupported population decoding version")
    allowed = TARGET_NAMES if version_name == LEGACY_ANALYSIS_VERSION else MODEL_TARGET_NAMES
    cells = manifest.get("cells", [])
    declared = set()
    for cell in cells:
        if not isinstance(cell, str) or "_" not in cell:
            raise ValueError("Invalid saved cell identity")
        region, target = cell.split("_", 1)
        if region not in ("PFC", "HPC") or target not in allowed:
            raise ValueError("Invalid saved model target")
        declared.add(target)
    names = tuple(name for name in allowed if name in declared)
    if not names or (version_name == ANALYSIS_VERSION and manifest.get("model_targets") != list(MODEL_TARGET_NAMES)):
        raise ValueError("Invalid saved model target declaration")
    return names


def inspect_run_status(directory: Path) -> dict:
    """Inspect expected paired result files; return completion counts/status and completed IDs.

    The immutable manifest/results are authoritative even if progress.json was
    interrupted. No fits or source reads occur. Corrupt/incompatible files raise
    ValueError instead of being treated as absent or overwritten.
    """
    directory = Path(directory)
    manifest = load_manifest(directory)
    completed = []
    for entry in manifest["cells"]:
        path = directory / "cells" / f"{entry}.npz"
        if not path.exists():
            continue
        cell = load_cell(path)
        if cell["meta"]["model_id"] != manifest["identities"]["model_id"]:
            raise ValueError("Published cell identity differs from this run")
        if f"{cell['meta']['region']}_{cell['meta']['target']}" != entry:
            raise ValueError("Published cell region/target differs from its filename")
        completed.append(entry)
    return {"status": "complete" if len(completed) == len(manifest["cells"]) else "partial",
            "completed_cells": len(completed), "expected_cells": len(manifest["cells"]),
            "completed_ids": completed, "run_directory": str(directory)}


def write_json(path: Path, payload: dict) -> None:
    """Atomically publish small JSON dict at Path (progress may replace prior state); no units."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, suffix=".tmp") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _save_archive(path: Path, arrays: dict, meta: dict) -> None:
    """Write named NumPy arrays/meta to a new scalar NPZ Path; never replace existing files."""
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **arrays, meta=np.array({**meta, "analysis_version": ANALYSIS_VERSION,
                                                            "schema_version": SCHEMA_VERSION}, dtype=object))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_archive(path: Path) -> dict:
    """Read trusted project-generated arrays/meta from scalar NPZ Path; corruption raises ValueError."""
    try:
        with np.load(path, allow_pickle=True) as archive:
            data = {key: archive[key] for key in archive.files}
        meta = data["meta"].item()
        if meta["analysis_version"] not in READABLE_ANALYSIS_VERSIONS or meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported population decoding version")
        data["meta"] = meta
        return data
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError, BadZipFile) as error:
        raise ValueError(f"Invalid saved population decoding archive {path}: {error}") from error
