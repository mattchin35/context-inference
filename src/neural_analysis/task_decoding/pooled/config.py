"""Configuration contracts for pooled-window task-variable decoding."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from src.neural_analysis.task_decoding import config as base_config


ANALYSIS_VERSION = "pooled-task-variable-decoding-v1"
REFERENCE_CONDITION = "correct_rewarded"
_ALLOWED_OVERLAY_FIELDS = frozenset({"base_config_path", "training_window_names"})
_CANONICAL_WINDOW_BOUNDS = (
    ("pre_choice", -0.5, 0.0),
    ("post_choice", 0.0, 0.5),
    ("full", base_config.WINDOW_START_SECONDS, base_config.WINDOW_END_SECONDS),
)
_CANONICAL_WINDOW_NAMES = tuple(item[0] for item in _CANONICAL_WINDOW_BOUNDS)
_BASE_SCIENTIFIC_FIELDS = frozenset(
    {
        "analysis_version",
        "session_metadata_path",
        "augmented_trial_path",
        "trial_feature_parameter_path",
        "pfc_region",
        "hpc_region",
        "alignment",
        "bin_width_ms",
        "pfc_pc_count",
        "hpc_pc_count",
        "target_names",
        "condition_names",
        "regularization_mode",
        "outer_fold_count",
        "inner_fold_count",
        "trusted_utc_bounds",
        "frozen_controls",
    }
)
_POOLED_SCIENTIFIC_FIELDS = frozenset(
    {
        "analysis_version",
        "base_config",
        "condition_names",
        "reference_condition",
        "training_windows",
    }
)


@dataclass(frozen=True)
class PooledDecodingConfig:
    """Validated configuration for one pooled-window analysis.

    Parameters
    ----------
    base_config : TaskDecodingConfig
        Validated existing task-decoding settings with absolute execution paths.
    base_config_path : pathlib.Path
        Absolute submitted base-configuration path. For prepared runs this is
        the inert audit-copy location and is never parsed.
    training_window_names : tuple[str, ...]
        Nonempty canonical-order window identifiers.
    condition_names : tuple[str, ...]
        Effective canonical conditions, including ``correct_rewarded``.
    """

    base_config: base_config.TaskDecodingConfig
    base_config_path: Path
    training_window_names: tuple[str, ...]
    condition_names: tuple[str, ...]


@dataclass(frozen=True)
class TrainingWindow:
    """One resolved choice-relative pooled training interval.

    Parameters
    ----------
    name : str
        Canonical window identifier.
    start_s, stop_s : float
        Half-open choice-relative bounds in seconds.
    bin_indices : numpy.ndarray
        Read-only ascending integer positions on the evaluation-time axis,
        shape ``(training_time,)``.
    """

    name: str
    start_s: float
    stop_s: float
    bin_indices: np.ndarray


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    """Read one UTF-8 JSON object without mutating the filesystem.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSON file.
    description : str
        Human-readable name used in validation errors.

    Returns
    -------
    dict[str, object]
        Decoded JSON mapping.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not decode {description} {path}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


def _parse_training_window_names(value: object) -> tuple[str, ...]:
    """Validate and canonicalize a pooled window selection.

    Parameters
    ----------
    value : object
        Omitted value or decoded nonempty JSON list of window names.

    Returns
    -------
    tuple[str, ...]
        Selected identifiers restored to canonical order.
    """
    if value is None:
        return ("pre_choice",)
    if not isinstance(value, list) or not value:
        raise ValueError("training_window_names must be a nonempty JSON list.")
    if any(not isinstance(name, str) or not name for name in value):
        raise ValueError("training_window_names must contain nonempty strings.")
    if len(set(value)) != len(value):
        raise ValueError("training_window_names must not contain duplicates.")
    unknown = [name for name in value if name not in _CANONICAL_WINDOW_NAMES]
    if unknown:
        raise ValueError(f"Unknown training window names: {unknown}")
    selected = set(value)
    return tuple(name for name in _CANONICAL_WINDOW_NAMES if name in selected)


def _effective_condition_names(
    task_config: base_config.TaskDecodingConfig,
) -> tuple[str, ...]:
    """Add the fixed reference condition to the base condition subset.

    Parameters
    ----------
    task_config : TaskDecodingConfig
        Validated base configuration with canonical condition order.

    Returns
    -------
    tuple[str, ...]
        Canonical effective conditions including ``correct_rewarded``.
    """
    selected = set(task_config.condition_names)
    selected.add(REFERENCE_CONDITION)
    return tuple(
        name for name in base_config.CONDITION_IDENTIFIERS if name in selected
    )


def _validate_base_mode(task_config: base_config.TaskDecodingConfig) -> None:
    """Enforce the revision-1 alignment and regularization restrictions.

    Parameters
    ----------
    task_config : TaskDecodingConfig
        Validated base configuration.

    Returns
    -------
    None
        Raises ``ValueError`` for unsupported settings.
    """
    if task_config.alignment != "choice_time":
        raise ValueError("Pooled decoding requires choice_time alignment.")
    if task_config.regularization_mode != "fixed":
        raise ValueError("Pooled decoding revision 1 requires fixed regularization.")


def load_pooled_decoding_config(path: Path | str) -> PooledDecodingConfig:
    """Load a pooled overlay and its existing base configuration.

    Parameters
    ----------
    path : pathlib.Path or str
        Pooled JSON overlay path. A relative ``base_config_path`` resolves from
        this file's parent.

    Returns
    -------
    PooledDecodingConfig
        Frozen validated settings with absolute execution paths and canonical
        window/condition order.
    """
    overlay_path = Path(path).resolve()
    payload = _read_json_object(overlay_path, "pooled decoding configuration")
    unknown_fields = set(payload) - _ALLOWED_OVERLAY_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown pooled configuration fields: {sorted(unknown_fields)}")
    if "base_config_path" not in payload:
        raise ValueError("base_config_path is required.")
    base_path_value = payload["base_config_path"]
    if not isinstance(base_path_value, str) or not base_path_value:
        raise ValueError("base_config_path must be a nonempty string.")
    supplied_base_path = Path(base_path_value)
    base_path = (
        supplied_base_path
        if supplied_base_path.is_absolute()
        else overlay_path.parent / supplied_base_path
    ).resolve()
    task_config = base_config.load_task_decoding_config(base_path)
    _validate_base_mode(task_config)
    window_names = _parse_training_window_names(payload.get("training_window_names"))
    return PooledDecodingConfig(
        base_config=task_config,
        base_config_path=base_path,
        training_window_names=window_names,
        condition_names=_effective_condition_names(task_config),
    )


def _canonical_time_bin_edges(bin_width_ms: int) -> np.ndarray:
    """Construct the base decoder's complete choice-relative edge axis.

    Parameters
    ----------
    bin_width_ms : int
        Supported positive bin width in milliseconds.

    Returns
    -------
    numpy.ndarray
        Float64 edge vector in seconds with shape ``(evaluation_time + 1,)``.
    """
    window_ms = int(
        round(
            1000
            * (base_config.WINDOW_END_SECONDS - base_config.WINDOW_START_SECONDS)
        )
    )
    if bin_width_ms not in base_config.SUPPORTED_BIN_WIDTH_MS:
        raise ValueError("Unsupported base bin width for pooled decoding.")
    if window_ms % bin_width_ms != 0:
        raise ValueError("Base bin width does not divide the complete analysis window.")
    bin_count = window_ms // bin_width_ms
    return np.linspace(
        base_config.WINDOW_START_SECONDS,
        base_config.WINDOW_END_SECONDS,
        bin_count + 1,
        dtype=np.float64,
    )


def resolve_training_windows(
    config: PooledDecodingConfig,
    time_bin_edges_s: np.ndarray,
) -> tuple[TrainingWindow, ...]:
    """Resolve configured half-open windows onto an exact time-bin edge axis.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated pooled settings.
    time_bin_edges_s : numpy.ndarray
        Finite strictly ascending choice-relative bin edges in seconds, shape
        ``(evaluation_time + 1,)``.

    Returns
    -------
    tuple[TrainingWindow, ...]
        Canonical selected windows with read-only ascending bin indices.

    Raises
    ------
    ValueError
        If the edge axis differs from the base configuration or a window
        boundary does not coincide exactly with an edge.
    """
    if not isinstance(config, PooledDecodingConfig):
        raise ValueError("config must be a PooledDecodingConfig.")
    raw_edges = np.asarray(time_bin_edges_s)
    if raw_edges.ndim != 1 or raw_edges.size < 2:
        raise ValueError("time_bin_edges_s must be a one-dimensional edge vector.")
    if not np.issubdtype(raw_edges.dtype, np.number) or np.issubdtype(
        raw_edges.dtype, np.bool_
    ):
        raise ValueError("time_bin_edges_s must contain numeric seconds.")
    edges = np.asarray(raw_edges, dtype=np.float64)
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0.0):
        raise ValueError("time_bin_edges_s must be finite and strictly ascending.")
    expected_edges = _canonical_time_bin_edges(config.base_config.bin_width_ms)
    if not np.array_equal(edges, expected_edges):
        raise ValueError("time-bin edges do not match the canonical base edge axis.")

    bounds_by_name = {
        name: (start_s, stop_s)
        for name, start_s, stop_s in _CANONICAL_WINDOW_BOUNDS
    }
    resolved: list[TrainingWindow] = []
    for name in config.training_window_names:
        start_s, stop_s = bounds_by_name[name]
        start_positions = np.flatnonzero(edges == start_s)
        stop_positions = np.flatnonzero(edges == stop_s)
        if start_positions.size != 1 or stop_positions.size != 1:
            raise ValueError(f"Training-window boundary does not align exactly: {name}")
        start_index = int(start_positions[0])
        stop_index = int(stop_positions[0])
        bin_indices = np.arange(start_index, stop_index, dtype=np.int64)
        if bin_indices.size == 0:
            raise ValueError(f"Training window contains no complete bins: {name}")
        bin_indices.setflags(write=False)
        resolved.append(
            TrainingWindow(
                name=name,
                start_s=start_s,
                stop_s=stop_s,
                bin_indices=bin_indices,
            )
        )
    return tuple(resolved)


def _validated_training_windows(
    config: PooledDecodingConfig,
    training_windows: Sequence[TrainingWindow],
) -> tuple[TrainingWindow, ...]:
    """Validate supplied resolved windows against canonical configuration.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated pooled settings.
    training_windows : sequence[TrainingWindow]
        Proposed resolved window records.

    Returns
    -------
    tuple[TrainingWindow, ...]
        Records after exact name, bound, and bin-index validation.
    """
    if isinstance(training_windows, (str, bytes)):
        raise ValueError("training_windows must be a sequence of TrainingWindow records.")
    supplied = tuple(training_windows)
    expected = resolve_training_windows(
        config,
        _canonical_time_bin_edges(config.base_config.bin_width_ms),
    )
    if len(supplied) != len(expected):
        raise ValueError("Resolved training windows do not match configuration.")
    for observed, canonical in zip(supplied, expected, strict=True):
        if not isinstance(observed, TrainingWindow):
            raise ValueError("training_windows contains an invalid record.")
        if (
            observed.name != canonical.name
            or observed.start_s != canonical.start_s
            or observed.stop_s != canonical.stop_s
            or not np.array_equal(observed.bin_indices, canonical.bin_indices)
        ):
            raise ValueError("Resolved training-window payload drifted from configuration.")
    return supplied


def scientific_pooled_config_payload(
    config: PooledDecodingConfig,
    training_windows: Sequence[TrainingWindow],
) -> dict[str, object]:
    """Serialize deterministic pooled scientific configuration.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated pooled settings. Absolute execution paths are excluded.
    training_windows : sequence[TrainingWindow]
        Canonically resolved choice-relative windows with bin indices.

    Returns
    -------
    dict[str, object]
        JSON-safe portable scientific payload used in run identity.
    """
    windows = _validated_training_windows(config, training_windows)
    return {
        "analysis_version": ANALYSIS_VERSION,
        "base_config": base_config.scientific_config_payload(config.base_config),
        "condition_names": list(config.condition_names),
        "reference_condition": REFERENCE_CONDITION,
        "training_windows": [
            {
                "name": window.name,
                "start_s": window.start_s,
                "stop_s": window.stop_s,
                "bin_indices": window.bin_indices.tolist(),
            }
            for window in windows
        ],
    }


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether an absolute path is contained by an absolute directory.

    Parameters
    ----------
    path, directory : pathlib.Path
        Resolved candidate and allowed root paths.

    Returns
    -------
    bool
        True when ``path`` equals or is below ``directory``.
    """
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _resolve_prepared_input_path(
    value: object,
    session_root: Path,
    field_name: str,
) -> Path:
    """Resolve one canonical session-relative prepared scientific input.

    Parameters
    ----------
    value : object
        Portable POSIX-like relative path from canonical ``config.json``.
    session_root : pathlib.Path
        Existing absolute saved session root.
    field_name : str
        Configuration field used in validation errors.

    Returns
    -------
    pathlib.Path
        Existing absolute regular file contained by ``session_root``.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty relative path.")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field_name} must be a contained relative path.")
    resolved = (session_root / relative).resolve()
    if not _is_within(resolved, session_root):
        raise ValueError(f"{field_name} escapes the saved session_root.")
    if not resolved.is_file():
        raise ValueError(f"Prepared scientific input is missing: {resolved}")
    return resolved


def _reconstruct_base_config(
    payload: object,
    *,
    session_root: Path,
    output_root: Path,
) -> base_config.TaskDecodingConfig:
    """Rebuild a validated base configuration from canonical scientific JSON.

    Parameters
    ----------
    payload : object
        Saved base scientific payload from ``config.json``.
    session_root : pathlib.Path
        Existing absolute root used to restore portable input paths.
    output_root : pathlib.Path
        Absolute run-directory parent used only for execution.

    Returns
    -------
    TaskDecodingConfig
        Reconstructed validated base settings.
    """
    if not isinstance(payload, dict) or set(payload) != _BASE_SCIENTIFIC_FIELDS:
        raise ValueError("Prepared base configuration has invalid canonical fields.")
    metadata_path = _resolve_prepared_input_path(
        payload["session_metadata_path"],
        session_root,
        "session_metadata_path",
    )
    if metadata_path.name != base_config.CANONICAL_FILENAME:
        raise ValueError("Prepared metadata path is not canonical.")
    if metadata_path.parent != session_root:
        raise ValueError("Prepared metadata path must directly define session_root.")
    augmented_path = _resolve_prepared_input_path(
        payload["augmented_trial_path"],
        session_root,
        "augmented_trial_path",
    )
    parameter_path = _resolve_prepared_input_path(
        payload["trial_feature_parameter_path"],
        session_root,
        "trial_feature_parameter_path",
    )
    _read_json_object(parameter_path, "trial feature parameter file")
    pfc_region = base_config._parse_region(payload["pfc_region"], "PFC")
    hpc_region = base_config._parse_region(payload["hpc_region"], "HPC")
    if pfc_region.probe_id == hpc_region.probe_id:
        raise ValueError("Prepared PFC and HPC probe IDs must be distinct.")
    alignment = payload["alignment"]
    regularization_mode = payload["regularization_mode"]
    if alignment not in {"choice_time", "start_time"}:
        raise ValueError("Prepared base alignment is invalid.")
    if regularization_mode not in {"fixed", "tuned"}:
        raise ValueError("Prepared base regularization mode is invalid.")
    if not _is_within(output_root, session_root) or output_root == session_root:
        raise ValueError("Prepared output root must be below the saved session_root.")
    source_root = base_config._find_source_git_root()
    if source_root is not None and _is_within(output_root, source_root):
        raise ValueError("Prepared output root must not be inside the source checkout.")
    for input_path in (metadata_path, augmented_path, parameter_path):
        if _is_within(input_path, output_root) or _is_within(output_root, input_path):
            raise ValueError("Prepared output root overlaps a scientific input path.")

    reconstructed = base_config.TaskDecodingConfig(
        session_metadata_path=metadata_path,
        augmented_trial_path=augmented_path,
        trial_feature_parameter_path=parameter_path,
        pfc_region=pfc_region,
        hpc_region=hpc_region,
        alignment=alignment,
        bin_width_ms=base_config._parse_count(
            payload["bin_width_ms"],
            "bin_width_ms",
            set(base_config.SUPPORTED_BIN_WIDTH_MS),
            100,
        ),
        pfc_pc_count=base_config._parse_positive_count(
            payload["pfc_pc_count"],
            "pfc_pc_count",
            10,
        ),
        hpc_pc_count=base_config._parse_positive_count(
            payload["hpc_pc_count"],
            "hpc_pc_count",
            10,
        ),
        target_names=base_config._parse_target_names(payload["target_names"]),
        regularization_mode=regularization_mode,
        outer_fold_count=base_config._parse_count(
            payload["outer_fold_count"],
            "outer_fold_count",
            {3, 5},
            5,
        ),
        inner_fold_count=base_config._parse_count(
            payload["inner_fold_count"],
            "inner_fold_count",
            {3},
            3,
        ),
        trusted_utc_bounds=base_config._parse_trusted_bounds(
            payload["trusted_utc_bounds"],
            {pfc_region.probe_id, hpc_region.probe_id},
        ),
        output_root=output_root,
        session_root=session_root,
        condition_names=base_config._parse_condition_names(payload["condition_names"]),
    )
    if base_config.scientific_config_payload(reconstructed) != payload:
        raise ValueError("Prepared base scientific payload has canonical or frozen-control drift.")
    return reconstructed


def load_prepared_pooled_decoding_config(
    run_directory: Path | str,
) -> tuple[PooledDecodingConfig, tuple[TrainingWindow, ...]]:
    """Reconstruct prepared settings without rereading submitted audit copies.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Existing immutable pooled run directory containing canonical
        ``config.json`` and execution-only ``execution.json``.

    Returns
    -------
    tuple[PooledDecodingConfig, tuple[TrainingWindow, ...]]
        Reconstructed base/pooled configuration and canonical resolved windows.

    Raises
    ------
    ValueError
        If saved roots, paths, controls, ordering, or canonical payload content
        have drifted.
    """
    directory = Path(run_directory).resolve()
    if not directory.is_dir():
        raise ValueError("Prepared pooled run directory does not exist.")
    saved_payload = _read_json_object(directory / "config.json", "prepared config")
    execution = _read_json_object(directory / "execution.json", "execution metadata")
    session_root_value = execution.get("session_root")
    if not isinstance(session_root_value, str) or not session_root_value:
        raise ValueError("execution.json requires an absolute session_root.")
    declared_root = Path(session_root_value)
    if not declared_root.is_absolute():
        raise ValueError("execution.json session_root must be absolute.")
    session_root = declared_root.resolve()
    if not session_root.is_dir():
        raise ValueError("Saved session_root is missing or is not a directory.")
    if set(saved_payload) != _POOLED_SCIENTIFIC_FIELDS:
        raise ValueError("Prepared pooled config has invalid canonical fields.")

    task_config = _reconstruct_base_config(
        saved_payload["base_config"],
        session_root=session_root,
        output_root=directory.parent,
    )
    saved_windows = saved_payload["training_windows"]
    if not isinstance(saved_windows, list) or not saved_windows:
        raise ValueError("Prepared training_windows must be a nonempty list.")
    window_names: list[str] = []
    for window in saved_windows:
        if not isinstance(window, dict) or not isinstance(window.get("name"), str):
            raise ValueError("Prepared training-window record is invalid.")
        window_names.append(window["name"])
    config = PooledDecodingConfig(
        base_config=task_config,
        base_config_path=(directory / "base_config.input.json").resolve(),
        training_window_names=_parse_training_window_names(window_names),
        condition_names=_effective_condition_names(task_config),
    )
    _validate_base_mode(task_config)
    windows = resolve_training_windows(
        config,
        _canonical_time_bin_edges(task_config.bin_width_ms),
    )
    if scientific_pooled_config_payload(config, windows) != saved_payload:
        raise ValueError("Prepared pooled scientific payload has canonical drift.")
    return config, windows
