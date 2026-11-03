"""Small portable input configuration for the PC-first context encoding analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType

from src.neural_analysis.session_metadata import SessionMetadata, load_session_metadata


ANALYSIS_VERSION = "context-encoding-v1"
SUPPORTED_BIN_WIDTH_MS = (500, 100, 50, 20)
WINDOWS = ("before", "after", "whole")
TIMING_MODES = ("choice_updated", "trial_window")
ENCODINGS = ("signed_context", "separate_context")
ESTIMATORS = ("ridge", "ols")
PENALTY_MODES = ("same_numeric", "matched_context")


@dataclass(frozen=True)
class EncodingConfig:
    """Validated scalar controls and absolute paths; no neural arrays are loaded.

    Path fields name input files or derived session-local output directories.
    probe_roles maps PFC/HPC to declared probe IDs; an omitted region is absent.
    Selection tuples contain case-sensitive identifiers in requested order.
    bin_width_ms is integer milliseconds; window_s is choice-relative seconds.
    alpha is the positive L2 penalty for summed SSE, not mean SSE. alignment,
    fold_count and window_s are fixed contract fields, not JSON controls.
    """

    session_metadata: Path
    augmented_trials: Path
    feature_parameters: Path
    session_root: Path
    intermediate_root: Path
    run_root: Path
    probe_roles: Mapping[str, str]
    timeline_result: Path | None = None
    bin_width_ms: int = 500
    windows: tuple[str, ...] = WINDOWS
    timing_modes: tuple[str, ...] = TIMING_MODES
    encodings: tuple[str, ...] = ENCODINGS
    estimators: tuple[str, ...] = ESTIMATORS
    alpha: float = 1.0
    penalty_modes: tuple[str, ...] = ("same_numeric",)
    alignment: str = "choice_time"
    fold_count: int = 5
    window_s: tuple[float, float] = (-2.0, 2.0)


# Public loading entry point.
def load_encoding_config(path: Path | str) -> EncodingConfig:
    """Read one JSON config and small provenance inputs without loading spikes.

    Parameters
    ----------
    path : Path or str
        Scalar configuration filename. Declared relative paths resolve against
        its parent, independent of the process working directory.

    Returns
    -------
    EncodingConfig
        Absolute paths, immutable selections, native millisecond bin width and
        dimensionless penalty. Output directories are derived but not created.
        Missing inputs, unknown fields, or unsupported controls raise ValueError.
    """
    config_path = Path(path).expanduser().resolve()
    payload = _read_json_object(config_path, "configuration")
    required = {"session_metadata", "augmented_trials", "feature_parameters"}
    optional = {"probe_roles", "timeline_result", "bin_width_ms", "windows",
                "timing_modes", "encodings", "estimators", "alpha", "penalty_modes"}
    unknown = set(payload) - required - optional
    if unknown:
        raise ValueError(f"Unsupported configuration fields: {sorted(unknown)}")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Missing configuration fields: {sorted(missing)}")

    inputs = {field: _input_path(config_path.parent, payload[field], field) for field in required}
    metadata = load_session_metadata(inputs["session_metadata"])
    _read_json_object(inputs["feature_parameters"], "feature_parameters")
    roles = _probe_roles(payload.get("probe_roles"), metadata)
    width = payload.get("bin_width_ms", 500)
    if type(width) is not int or width not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError(f"bin_width_ms must be one of {SUPPORTED_BIN_WIDTH_MS}")
    alpha = payload.get("alpha", 1.0)
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive number")
    selections = {}
    for field, allowed, default in (
        ("windows", WINDOWS, WINDOWS),
        ("timing_modes", TIMING_MODES, TIMING_MODES),
        ("encodings", ENCODINGS, ENCODINGS),
        ("estimators", ESTIMATORS, ESTIMATORS),
        ("penalty_modes", PENALTY_MODES, ("same_numeric",)),
    ):
        selections[field] = _selection(payload.get(field, list(default)), field, allowed)

    timeline = payload.get("timeline_result")
    timeline_path = None if timeline is None else _input_path(config_path.parent, timeline, "timeline_result")
    root = inputs["session_metadata"].parent
    return EncodingConfig(
        **inputs, session_root=root,
        intermediate_root=root / "processed/context_encoding", run_root=root / "analysis_runs",
        probe_roles=MappingProxyType(roles), timeline_result=timeline_path,
        bin_width_ms=width, alpha=float(alpha), **selections,
    )


# Small validation helpers; all operate on control files, never neural sources.
def _read_json_object(path: Path, field: str) -> dict:
    """Read scalar Path/field string; return unitless JSON dict or raise ValueError."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field}: could not read JSON object: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value


def _input_path(directory: Path, value: object, field: str) -> Path:
    """Resolve directory Path/value string/field name to existing file Path; no units."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must name an existing input file")
    path = (directory / Path(value).expanduser()).resolve()
    if not path.is_file():
        raise ValueError(f"{field} must name an existing input file: {path}")
    return path


def _selection(value: object, field: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
    """Validate JSON list/field/allowed tuple; return nonempty unique string tuple, no units."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list from {allowed}")
    if any(not isinstance(item, str) or item not in allowed for item in value):
        raise ValueError(f"{field} supports only {allowed}")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} must not contain duplicates")
    return tuple(value)


def _probe_roles(value: object, metadata: SessionMetadata) -> dict[str, str]:
    """Validate optional JSON mapping/metadata; return PFC/HPC probe-ID dict, no units."""
    if value is None:
        if metadata.analysis is None:
            raise ValueError("probe_roles is required for version-2 metadata")
        roles = metadata.analysis.probe_roles
        value = {"PFC": roles.pfc, "HPC": roles.hpc}
    if not isinstance(value, dict) or not value or not set(value).issubset({"PFC", "HPC"}):
        raise ValueError("probe_roles must be a nonempty PFC/HPC mapping")
    declared = {probe.probe_id for probe in metadata.probes}
    if any(not isinstance(probe, str) or probe not in declared for probe in value.values()):
        raise ValueError("probe_roles must reference declared metadata probes")
    if len(set(value.values())) != len(value):
        raise ValueError("probe_roles must use distinct probes")
    return dict(value)
