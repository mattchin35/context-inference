"""Create editable context-encoding controls from small existing session inputs."""

from collections.abc import Mapping
import json
from pathlib import Path

from src.neural_analysis.session_metadata import load_session_metadata

from .configuration import (
    ENCODINGS, ESTIMATORS, PENALTY_MODES, SUPPORTED_BIN_WIDTH_MS, TIMING_MODES, WINDOWS,
    _input_path, _probe_roles, _read_json_object,
)


def initialize_encoding_config(
    metadata_path: Path | str, *, augmented_trials: Path | str | None = None,
    feature_parameters: Path | str | None = None, bin_width_ms: int = 500,
    probe_roles: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Create sibling context_encoding_config.json or preserve an existing file.

    metadata_path names a session metadata JSON. Optional input locators resolve
    relative to its directory; omitted locators come from sibling
    task_decoding_config.json without validating unrelated decoder settings.
    bin_width_ms is a supported integer millisecond width. probe_roles optionally
    maps PFC/HPC to distinct declared probe IDs; None uses v3 metadata roles.

    Return status (created/preserved) and absolute config_path strings. Existing
    files are never edited or validated, even if source inputs have disappeared.
    New files use portable relative locators for contained inputs, the reviewed
    all-variant selections, and alpha=1. Validate named files/feature JSON before
    writing. No neural arrays, analysis directories, or fits are created.
    """
    metadata_path = Path(metadata_path).expanduser().resolve()
    root = metadata_path.parent
    destination = root / "context_encoding_config.json"
    report = {"status": "preserved", "config_path": str(destination)}
    if destination.exists():
        if not destination.is_file():
            raise ValueError(f"Configuration destination is not a file: {destination}")
        return report
    metadata = load_session_metadata(metadata_path)
    if type(bin_width_ms) is not int or bin_width_ms not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError(f"bin_width_ms must be one of {SUPPORTED_BIN_WIDTH_MS}")
    if probe_roles is None and metadata.analysis is None:
        raise ValueError("Version-2 metadata requires --pfc-probe and/or --hpc-probe")
    roles = None if probe_roles is None else dict(probe_roles)
    _probe_roles(roles, metadata)
    task = {}
    if augmented_trials is None or feature_parameters is None:
        task_path = root / "task_decoding_config.json"
        if not task_path.is_file():
            raise ValueError("No task_decoding_config.json; supply --augmented-trials and --feature-parameters")
        task = _read_json_object(task_path, "task_decoding_config.json (or supply explicit input paths)")
    augmented = _input_path(root, str(augmented_trials) if augmented_trials is not None
                            else task.get("augmented_trial_path"), "--augmented-trials")
    features = _input_path(root, str(feature_parameters) if feature_parameters is not None
                           else task.get("trial_feature_parameter_path"), "--feature-parameters")
    _read_json_object(features, "feature_parameters (--feature-parameters)")
    payload = {
        "session_metadata": metadata_path.name,
        "augmented_trials": _portable_locator(augmented, root),
        "feature_parameters": _portable_locator(features, root),
        "probe_roles": roles, "timeline_result": None, "bin_width_ms": bin_width_ms,
        "windows": list(WINDOWS), "timing_modes": list(TIMING_MODES),
        "encodings": list(ENCODINGS), "estimators": list(ESTIMATORS),
        "alpha": 1.0, "penalty_modes": list(PENALTY_MODES),
    }
    encoded = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"
    try:
        with destination.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(encoded)
    except FileExistsError:
        return report
    return {**report, "status": "created"}


# Path formatting is separate from input validation and exclusive publication.
def _portable_locator(path: Path, session_root: Path) -> str:
    """Return a session-relative POSIX locator when contained, otherwise an absolute path string."""
    return path.relative_to(session_root).as_posix() if path.is_relative_to(session_root) else str(path)
