"""Portable controls for the fixed population task-decoding protocol."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping

from src.neural_analysis.session_metadata import load_session_metadata

LEGACY_ANALYSIS_VERSION = "population-task-decoding-v1"
ANALYSIS_VERSION = "population-task-decoding-v2"
READABLE_ANALYSIS_VERSIONS = (LEGACY_ANALYSIS_VERSION, ANALYSIS_VERSION)
SCHEMA_VERSION = 1
SUPPORTED_BIN_WIDTH_MS = (500, 100, 50, 20)
REPRESENTATIONS = ("units", "pcs")


@dataclass(frozen=True)
class PopulationDecodingConfig:
    """Absolute input/output paths and fixed controls; no neural data loaded.

    probe_roles maps PFC/HPC to distinct declared probes. bin_width_ms is in
    milliseconds; window_s is choice-relative seconds. Fold count, both feature
    representations, and the analysis version are fixed scientific contracts.
    """

    session_metadata: Path
    augmented_trials: Path
    probe_roles: Mapping[str, str]
    bin_width_ms: int = 500
    window_s: tuple[float, float] = (-1.0, 1.0)
    fold_count: int = 5
    representations: tuple[str, str] = REPRESENTATIONS
    analysis_version: str = ANALYSIS_VERSION

    @property
    def session_root(self) -> Path:
        """Return session-directory Path from the metadata location; no units."""
        return self.session_metadata.parent

    @property
    def intermediate_root(self) -> Path:
        """Return session-local preparation directory Path without creating it."""
        return self.session_root / "processed/population_task_decoding"

    @property
    def run_root(self) -> Path:
        """Return session-local human-readable run directory Path, no writes."""
        return self.session_root / "analysis_runs"


def load_config(path: Path | str) -> PopulationDecodingConfig:
    """Read scalar JSON Path; return validated controls and absolute input paths.

    Relative paths resolve against the config directory. Input files must exist;
    no output directories or neural arrays are created. Unknown controls fail.
    """
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text())
    required = {"analysis_version", "session_metadata", "augmented_trials"}
    allowed = required | {"probe_roles", "bin_width_ms"}
    if not isinstance(payload, dict) or required - payload.keys() or payload.keys() - allowed:
        raise ValueError("Missing or unsupported population decoding configuration fields")
    if payload["analysis_version"] != ANALYSIS_VERSION:
        if payload["analysis_version"] == LEGACY_ANALYSIS_VERSION:
            raise ValueError("Legacy v1 is read-only; create a v2 config with init-config --output <new.json>")
        raise ValueError(f"analysis_version must be {ANALYSIS_VERSION}")
    inputs = {}
    for name in ("session_metadata", "augmented_trials"):
        value = payload[name]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must name an input file")
        inputs[name] = (path.parent / Path(value).expanduser()).resolve()
        if not inputs[name].is_file():
            raise ValueError(f"Missing {name}: {inputs[name]}")
    metadata = load_session_metadata(inputs["session_metadata"])
    roles = payload.get("probe_roles")
    if roles is None and metadata.analysis is not None:
        roles = {"PFC": metadata.analysis.probe_roles.pfc, "HPC": metadata.analysis.probe_roles.hpc}
    declared = {probe.probe_id for probe in metadata.probes}
    if (not isinstance(roles, dict) or not roles or not roles.keys() <= {"PFC", "HPC"}
            or any(not isinstance(value, str) or value not in declared for value in roles.values())
            or len(set(roles.values())) != len(roles)):
        raise ValueError("probe_roles must map PFC/HPC to distinct declared probes")
    width = payload.get("bin_width_ms", 500)
    if type(width) is not int or width not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError(f"bin_width_ms must be one of {SUPPORTED_BIN_WIDTH_MS}")
    return PopulationDecodingConfig(**inputs, probe_roles=MappingProxyType(dict(roles)), bin_width_ms=width)
