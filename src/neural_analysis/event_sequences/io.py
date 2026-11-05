"""Immutable event products, exact portable PCA references and bounded spike reads.

Saved display uses only this product and its explicitly referenced PCA archive.
Current metadata/source readiness belongs to preparation, not saved loading.
"""

from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from pickle import UnpicklingError
import tempfile
from zipfile import BadZipFile
from zlib import error as DecompressionError

import numpy as np
import pandas as pd

from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import ResolvedSession, resolve_probe_sources
from src.neural_analysis.spike_behavior import loading
from .activity import SourceActivity, select_pc_activity
from .selection import (
    CENTRAL_CONDITIONS, TRANSITION_CONDITIONS, _ordinary_choices,
    prepare_sequence_trials, select_central_sequences, select_ideal_transitions,
)


ANALYSIS_VERSION = "event-sequence-heatmaps-v1"
SCHEMA_VERSION = 1
_TABLES = ("trials", "central_summary", "transition_summary")
_ARRAYS = ("bin_edges_s", "activity_row_positions", "full_to_activity")


def load_probe_spike_arrays(session: ResolvedSession, probe_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Input: resolved session/probe ID. Return read-only UTC spikes and integer assignments (S,).

    The two-entry LRU is keyed by actual alignment/assignment paths, byte sizes
    and mtime_ns, independent of selected unit/bin width. No selected-cluster
    presence requirement is imposed: a manually good unit may have zero spikes.
    """
    try:
        source = resolve_probe_sources(session, probe_id)
    except KeyError as error:
        raise ValueError(str(error)) from error
    if source.alignment_file is None or source.spike_clusters_file is None:
        raise ValueError(f"{probe_id}: aligned spikes and sorter assignments are required.")
    aligned = _existing_file_key(source.alignment_file)
    assignments = _existing_file_key(source.spike_clusters_file)
    return _cached_probe_arrays(aligned, assignments)


def clear_probe_spike_cache() -> None:
    """No inputs; return None after releasing both cached raw-probe array entries."""
    _cached_probe_arrays.cache_clear()


def source_file_identity(path: Path | str) -> dict:
    """Input: scalar file path. Return resolved path/size bytes/mtime_ns or explicit missing status."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def configuration_digest(configuration: dict) -> str:
    """Input: JSON-compatible source/parameter config. Return 16-character digest excluding its ID.

    Physical units remain in the declared parameters; this pure hash performs
    no reads. Exact PCA artifact identity is recorded separately from this
    canonical source configuration and never silently replaced during reuse.
    """
    payload = {key: value for key, value in configuration.items() if key != "configuration_id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()[:16]


def make_pca_reference(session_root: Path, path: Path, region: str, probe_id: str, configuration_id: str) -> dict:
    """Input: session root/PCA Path, role/probe/config strings. Return relative path/size/SHA256 identity.

    The archive must lie inside the session; sizes are bytes. No score tensors
    are copied into this record. Hashing streams file bytes with bounded memory.
    """
    root = Path(session_root).resolve()
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("PCA reference must be inside the session root.")
    return {
        "path": str(resolved.relative_to(root)), "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved), "configuration_id": configuration_id,
        "region": region, "probe_id": probe_id,
    }


def summarize_match_coverage(matches: dict, activity_row_positions: np.ndarray, *, transitions: bool = False) -> pd.DataFrame:
    """Input: ordered motif->int position arrays and covered positions (W,). Return count table.

    Central arrays are (M,), episodes (E,S). Each table retains every condition
    with unitless behavioral_count, excluded_count and n. Episodes require every
    segment covered; central conditions require only their displayed central row.
    """
    columns = ("direction", "repeats") if transitions else (
        "central_outcome", "side", "previous_outcome", "next_action",
    )
    rows = []
    for key, positions in matches.items():
        present = np.isin(positions, activity_row_positions)
        covered = present.all(axis=1) if transitions else present
        count = int(np.count_nonzero(covered))
        rows.append(dict(zip(columns, key), behavioral_count=len(positions), excluded_count=len(positions) - count, n=count))
    return pd.DataFrame(rows, columns=[*columns, "behavioral_count", "excluded_count", "n"])


def save_event_sequences(output_path: Path | str, packet: dict) -> Path:
    """Publish a new final named-array NPZ; never overwrite an existing product.

    packet holds full trials (N,C), all 24/8 match arrays, covered source rows
    (W,), inverse map (N,), relative-second edges (B+1,), count tables and meta.
    Units save Hz (W,B); PCs save shared choices and regional map (W,) plus exact
    relative PCA reference, with no copied PC tensor. Runtime session_root and
    timeline_result are not serialized. Return the new final Path.
    """
    path = Path(output_path).resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if path.suffix != ".npz":
        raise ValueError("Event sequence product must be an .npz file.")
    root = Path(packet["session_root"]).resolve()
    if path.name != "results.npz" or path.parent.parent != root / "analysis_runs":
        raise ValueError("Product must be results.npz in a session-local analysis_runs directory.")
    _validate_packet(packet)
    arrays = {name: packet[name].to_records(index=False) for name in _TABLES}
    arrays.update({name: packet[name] for name in _ARRAYS})
    for key, positions in packet["central_matches"].items():
        arrays["central__" + "__".join(key)] = positions
    for (direction, repeats), positions in packet["transition_matches"].items():
        arrays[f"transition__{direction}__{repeats}"] = positions
    if packet["meta"]["source_kind"] == "unit":
        arrays["unit_rates_hz"] = packet["unit_rates_hz"]
    else:
        arrays["shared_choices"] = packet["shared_choices"].to_records(index=False)
        arrays["regional_choice_positions"] = packet["regional_choice_positions"]
    arrays["meta"] = np.asarray(packet["meta"], dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_event_sequences(input_path: Path | str) -> dict:
    """Load a trusted saved product without current metadata/CSV/quality/spike access.

    input_path is session/analysis_runs/event_sequence_heatmaps_*/results.npz.
    Return the plain packet with original saved shapes/units and runtime-bound
    session_root. PCs verify exact referenced bytes/config/maps and load that
    fixed timeline once into runtime-only timeline_result for live rank changes.
    Missing PCA raises FileNotFoundError; malformed products raise ValueError.
    """
    path = Path(input_path).resolve(strict=True)
    try:
        with np.load(path, allow_pickle=True) as archive:
            meta = archive["meta"].item()
            _check_meta(meta)
            packet = {name: pd.DataFrame.from_records(archive[name]) for name in _TABLES}
            packet.update({name: archive[name] for name in _ARRAYS})
            packet.update(
                meta=meta, session_root=path.parents[2], timeline_result=None,
                central_matches={key: archive["central__" + "__".join(key)] for key in CENTRAL_CONDITIONS},
                transition_matches={key: archive[f"transition__{key[0]}__{key[1]}"] for key in TRANSITION_CONDITIONS},
                unit_rates_hz=archive["unit_rates_hz"] if meta["source_kind"] == "unit" else None,
                shared_choices=pd.DataFrame.from_records(archive["shared_choices"]) if meta["source_kind"] == "pc" else pd.DataFrame(),
                regional_choice_positions=archive["regional_choice_positions"] if meta["source_kind"] == "pc" else np.empty(0, dtype=int),
            )
        _validate_packet(packet)
        return packet
    except (KeyError, TypeError, IndexError, AttributeError, EOFError, BadZipFile, UnpicklingError, DecompressionError) as error:
        raise ValueError(f"Invalid saved event sequence product: {error}") from error


def discover_event_sequences(session_root: Path | str) -> list[dict]:
    """Input: session root Path. Return newest saved {result_path,meta,available,reason} entries.

    Discovery inspects saved metadata/member names and referenced PCA path/size
    only, never current sources or full PC tensors/hashes. Missing-PCA products
    remain selectable with explicit unavailable_reason; full selection performs
    exact validation. Interrupted/malformed/unsupported products are skipped.
    """
    root = Path(session_root).resolve()
    products = []
    for path in sorted((root / "analysis_runs").glob("event_sequence_heatmaps_*/results.npz"), reverse=True):
        try:
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
                _check_meta(meta)
                required = set(_TABLES + _ARRAYS)
                required.update("central__" + "__".join(key) for key in CENTRAL_CONDITIONS)
                required.update(f"transition__{direction}__{repeats}" for direction, repeats in TRANSITION_CONDITIONS)
                required.update(("unit_rates_hz",) if meta["source_kind"] == "unit" else ("shared_choices", "regional_choice_positions"))
                if not required.issubset(archive.files):
                    continue
            reason = ""
            if meta["source_kind"] == "pc":
                reference = meta["pca_reference"]
                pca = _reference_path(root, reference)
                if not pca.is_file():
                    reason = "Referenced PCA archive is missing."
                elif pca.stat().st_size != reference["size"]:
                    reason = "Referenced PCA archive byte size changed."
            products.append({"result_path": path, "meta": meta, "available": not reason, "unavailable_reason": reason})
        except (OSError, ValueError, KeyError, TypeError, AttributeError, EOFError, BadZipFile, UnpicklingError, DecompressionError):
            continue
    return products


def get_source_activity(packet: dict, *, pc_rank: int = 1) -> SourceActivity:
    """Input: loaded plain packet/one-based PC rank. Return selected SourceActivity (W,B).

    Unit values are saved Hz; PC values are the saved signed unwhitened axis.
    Uses bound arrays and labels only, with no raw reads, binning, fitting or
    publication. Full packet loading must precede display; rank is not a cache key.
    """
    meta = packet["meta"]
    if meta["source_kind"] == "unit":
        return SourceActivity(packet["unit_rates_hz"], packet["activity_row_positions"], packet["bin_edges_s"], "unit", meta["source_label"])
    if packet.get("timeline_result") is None:
        raise ValueError("PCA reference is not bound; load the saved product before display.")
    regional = packet["timeline_result"].regions[meta["pca_reference"]["region"]]
    return select_pc_activity(packet["trials"], packet["shared_choices"], regional, pc_rank)


# Persistence/source helpers, all limited to the explicitly supplied inputs.
@lru_cache(maxsize=2)
def _cached_probe_arrays(aligned_key: tuple, assignment_key: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Input: (path,size bytes,mtime ns) keys. Return immutable (S,) UTC spikes/integer assignments."""
    times = loading.load_aligned_spikes(Path(aligned_key[0]))
    assignments, _ = loading.load_sorter_metadata(Path(assignment_key[0]).parent)
    loading.validate_aligned_spike_inputs(times, assignments)
    result = (np.array(times, dtype=float, copy=True), np.array(assignments, dtype=int, copy=True))
    for values in result:
        values.setflags(write=False)
    return result


def _existing_file_key(path: Path) -> tuple:
    """Input: existing file Path. Return hashable path/size bytes/mtime-ns key; missing files raise FileNotFoundError."""
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return str(resolved), stat.st_size, stat.st_mtime_ns


def _sha256_file(path: Path) -> str:
    """Input: file Path. Return content SHA256 hex digest, reading bounded one-MiB byte blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _check_meta(meta: dict) -> None:
    """Input: saved metadata dict. Return None for supported version/source or raise ValueError."""
    if meta.get("analysis_version") != ANALYSIS_VERSION or meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported event sequence analysis/schema version.")
    if meta.get("source_kind") not in ("unit", "pc"):
        raise ValueError("Invalid saved event source kind.")
    if not isinstance(meta.get("configuration_id"), str) or not meta["configuration_id"]:
        raise ValueError("Missing event source configuration identity.")
    configuration = meta["configuration"]
    if configuration["analysis_version"] != meta["analysis_version"] or configuration["session_id"] != meta["session_id"]:
        raise ValueError("Saved event configuration version/session identity differs.")
    if configuration_digest(configuration) != meta["configuration_id"]:
        raise ValueError("Saved event source configuration identity differs from its parameters/sources.")
    if configuration["parameters"] != meta["parameters"] or meta["parameters"]["source_kind"] != meta["source_kind"]:
        raise ValueError("Saved event source parameters are inconsistent.")
    expected_units = {
        "absolute_time": "UTC Unix seconds", "relative_time": "seconds",
        "unit_rates_hz": "Hz", "scores": "unwhitened standardized-unit scores",
    }
    if meta.get("units") != expected_units:
        raise ValueError("Unsupported saved event activity/time units.")


def _reference_path(root: Path, reference: dict) -> Path:
    """Input: bound session root/reference dict. Return contained relative PCA Path or raise ValueError."""
    relative = Path(reference["path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("PCA reference must be relative and contained in the session.")
    return path


def _bind_pca(packet: dict) -> None:
    """Input: PC packet. Return None after exact bytes/config/mapping checks and runtime timeline binding.

    Saved choices (C,columns), regional map (W,) and native (W,B,K) referenced
    scores retain UTC seconds and signed unwhitened units. No current inputs load.
    """
    reference = packet["meta"]["pca_reference"]
    path = _reference_path(Path(packet["session_root"]), reference)
    if not path.is_file():
        raise FileNotFoundError(f"Referenced PCA archive is missing: {path}")
    if path.stat().st_size != reference["size"] or _sha256_file(path) != reference["sha256"]:
        raise ValueError("PCA reference byte identity changed.")
    result = packet.get("timeline_result")
    if result is None:
        result = timeline_io.load_timeline_result(path)
    if result.meta["configuration_id"] != reference["configuration_id"]:
        raise ValueError("PCA reference configuration identity differs.")
    regional = result.regions.get(reference["region"])
    if regional is None:
        raise ValueError("Referenced PCA archive lacks the requested region.")
    if regional.probe_id != reference["probe_id"]:
        raise ValueError("PCA reference probe identity differs.")
    if not np.array_equal(regional.choice_positions, packet["regional_choice_positions"]):
        raise ValueError("PCA regional choice mapping differs.")
    fitted = regional.transform
    observations = len(regional.choice_positions) * (len(regional.bin_edges_s) - 1)
    components = min(observations, len(fitted.retained_unit_ids))
    if (fitted.input_unit_ids != tuple(packet["meta"]["configuration"]["unit_ids"])
            or fitted.n_training_observations != observations or fitted.actual_components != components
            or regional.scores.shape != (len(regional.choice_positions), len(regional.bin_edges_s) - 1, components)
            or not np.isfinite(regional.scores).all()
            or result.meta.get("units", {}).get("scores") != "unwhitened standardized-unit scores"):
        raise ValueError("Referenced PCA full population/basis/score contract differs.")
    previous, saved = result.choices, packet["shared_choices"]
    if len(previous) != len(saved) or not timeline.trial_identities_match(previous["trial_id"], saved["trial_id"]):
        raise ValueError("PCA shared trial identity mapping differs.")
    for column in ("row_position", "choice_time", "action"):
        if not np.array_equal(previous[column].to_numpy(), saved[column].to_numpy()):
            raise ValueError(f"PCA shared {column} mapping differs.")
    packet["timeline_result"] = result
    source = get_source_activity(packet)
    if not np.array_equal(source.row_positions, packet["activity_row_positions"]):
        raise ValueError("PCA original source row mapping differs.")
    if not np.array_equal(source.bin_edges_s, packet["bin_edges_s"]):
        raise ValueError("PCA referenced bin geometry differs.")


def _validate_packet(packet: dict) -> None:
    """Input: plain saved/prepared packet. Return None after native arrays/labels/count/reference checks.

    Trials (N,C), source map (W,), inverse map (N,), matches (M,)/(E,S) and Hz
    (W,B) use their declared axes. Validation never consults current source files.
    """
    meta = packet["meta"]
    _check_meta(meta)
    trials = packet["trials"]
    if not {"row_position", "trial_id", "index_label"}.issubset(trials.columns):
        raise ValueError("Saved trials lack original positions/IDs/labels.")
    if not np.issubdtype(trials.row_position.dtype, np.integer) or not np.array_equal(trials.row_position, np.arange(len(trials))):
        raise ValueError("Saved original trial positions are invalid.")
    if "cur_trial" in trials and not timeline.trial_identities_match(trials.cur_trial, trials.trial_id):
        raise ValueError("Saved original trial identities are inconsistent.")
    rows = np.asarray(packet["activity_row_positions"])
    if (rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or np.any(rows < 0) or np.any(rows >= len(trials)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Invalid source activity row map.")
    inverse = np.full(len(trials), -1, dtype=int)
    inverse[rows] = np.arange(len(rows))
    if not np.issubdtype(np.asarray(packet["full_to_activity"]).dtype, np.integer) or not np.array_equal(inverse, packet["full_to_activity"]):
        raise ValueError("Invalid full-to-activity inverse mapping.")
    normalized = prepare_sequence_trials(trials)
    ordinary = np.flatnonzero(_ordinary_choices(normalized))
    covered = timeline.select_complete_choice_windows(normalized.choice_time.to_numpy()[ordinary], meta["coverage_s"])
    if not np.array_equal(rows, ordinary[covered]):
        raise ValueError("Saved source rows do not match the full eligible session reference pool.")
    edges = timeline.choice_bin_edges(meta["parameters"]["bin_width_s"])
    if not np.array_equal(edges, packet["bin_edges_s"]):
        raise ValueError("Invalid saved choice-bin geometry.")
    if meta["source_kind"] == "unit":
        get_source_activity(packet)
    else:
        _bind_pca(packet)
    expected_central = select_central_sequences(trials)
    absent = sorted({"state_int", "cur_block"} - set(trials.columns))
    if absent:
        if not all(column in meta["transition_unavailable_reason"] for column in absent):
            raise ValueError("Saved Part II availability reason omits missing fields.")
        expected_transitions = {key: np.empty((0, key[1] + 2), dtype=int) for key in TRANSITION_CONDITIONS}
    else:
        expected_transitions = select_ideal_transitions(trials)
    for name, expected in (("central_matches", expected_central), ("transition_matches", expected_transitions)):
        actual = packet[name]
        if list(actual) != list(expected):
            raise ValueError("Saved motif condition identities are incomplete.")
        for key, positions in expected.items():
            if not np.issubdtype(actual[key].dtype, np.integer) or not np.array_equal(actual[key], positions):
                raise ValueError(f"Invalid saved motif positions: {key}")
    for name, matches, transitions in (("central_summary", expected_central, False), ("transition_summary", expected_transitions, True)):
        expected = summarize_match_coverage(matches, rows, transitions=transitions)
        actual = packet[name]
        if list(actual.columns) != list(expected.columns) or len(actual) != len(expected):
            raise ValueError("Invalid saved condition summary axes.")
        for column in expected.columns:
            if column in ("behavioral_count", "excluded_count", "n") and not np.issubdtype(actual[column].dtype, np.integer):
                raise ValueError("Saved condition counts must be integer values.")
            if not np.array_equal(actual[column].to_numpy(), expected[column].to_numpy()):
                raise ValueError("Invalid saved behavioral/coverage counts or condition labels.")
