"""Frozen configuration and scientific-identity contracts for task decoding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType

from src.neural_analysis.session_metadata import CANONICAL_FILENAME


ANALYSIS_VERSION = "task-variable-decoding-v1"
TARGET_IDENTIFIERS = (
    "current_state",
    "current_action",
    "current_action_is_correct",
    "previous_action",
    "previous_action_was_rewarded",
    "next_action",
    "current_choice_switch_stay",
    "next_choice_switch_stay",
    "consecutive_omissions",
    "consecutive_rewards",
    "session_trial_index",
    "trial_index_in_block",
    "rewards_in_block",
    "qlearning_relative_value",
    "forgetting_q_relative_value",
    "hmm_signed_belief",
    "hmm_decay_signed_belief",
    "relative_doubt",
)

WINDOW_START_SECONDS = -2.0
WINDOW_END_SECONDS = 2.0
RANDOM_SEED = 0
COEFFICIENT_TOLERANCE = 1e-8
SUPPORTED_BIN_WIDTH_MS = (500, 100, 50, 20)


@dataclass(frozen=True)
class TargetDefinition:
    """Frozen source and labeling metadata for one task-variable target.

    Parameters
    ----------
    identifier : str
        Case-sensitive JSON and saved-array target identifier.
    family : str
        Either ``"categorical"`` or ``"numerical"``.
    display_label : str
        Human-readable figure and result label.
    source_column : str
        Augmented-table source column, including the source for derived labels.
    derivation : str
        Frozen ``"source"`` or chronological derivation rule used to encode
        the target without bridging invalid adjacent choices.
    positive_label : str or None
        Human-readable label for categorical integer class one, or None for a
        numerical target.
    """

    identifier: str
    family: str
    display_label: str
    source_column: str
    derivation: str
    positive_label: str | None = None


TARGET_DEFINITIONS = (
    TargetDefinition(
        "current_state", "categorical", "Current state", "state_int", "source", "left"
    ),
    TargetDefinition(
        "current_action", "categorical", "Current action", "action", "source", "left"
    ),
    TargetDefinition(
        "current_action_is_correct",
        "categorical",
        "Current action is correct",
        "correct",
        "source",
        "correct",
    ),
    TargetDefinition(
        "previous_action",
        "categorical",
        "Previous action",
        "action",
        "previous_valid_action",
        "left",
    ),
    TargetDefinition(
        "previous_action_was_rewarded",
        "categorical",
        "Previous action was rewarded",
        "reward",
        "previous_valid_action_reward",
        "rewarded",
    ),
    TargetDefinition(
        "next_action",
        "categorical",
        "Next action",
        "action",
        "next_valid_action",
        "left",
    ),
    TargetDefinition(
        "current_choice_switch_stay",
        "categorical",
        "Current choice switch/stay",
        "action",
        "current_vs_previous_action",
        "switch",
    ),
    TargetDefinition(
        "next_choice_switch_stay",
        "categorical",
        "Next choice switch/stay",
        "action",
        "next_vs_current_action",
        "switch",
    ),
    TargetDefinition(
        "consecutive_omissions",
        "numerical",
        "Consecutive omissions",
        "consecutive_omissions",
        "source",
    ),
    TargetDefinition(
        "consecutive_rewards",
        "numerical",
        "Consecutive rewards",
        "consecutive_rewards",
        "source",
    ),
    TargetDefinition(
        "session_trial_index",
        "numerical",
        "Session trial index",
        "cur_trial",
        "source",
    ),
    TargetDefinition(
        "trial_index_in_block",
        "numerical",
        "Trial index in block",
        "cur_trial_in_block",
        "source",
    ),
    TargetDefinition(
        "rewards_in_block",
        "numerical",
        "Rewards in block",
        "rewards_in_block",
        "source",
    ),
    TargetDefinition(
        "qlearning_relative_value",
        "numerical",
        "Q-learning relative value",
        "Qlearning_rel_value",
        "source",
    ),
    TargetDefinition(
        "forgetting_q_relative_value",
        "numerical",
        "Forgetting-Q relative value",
        "FQlearning_rel_value",
        "source",
    ),
    TargetDefinition(
        "hmm_signed_belief",
        "numerical",
        "HMM signed belief",
        "HMM_rel_value_logodds",
        "source",
    ),
    TargetDefinition(
        "hmm_decay_signed_belief",
        "numerical",
        "HMM-decay signed belief",
        "HMM_rel_value_logodds_decay",
        "source",
    ),
    TargetDefinition(
        "relative_doubt",
        "numerical",
        "Relative doubt",
        "relative_doubt_index",
        "source",
    ),
)
TARGET_DEFINITION_BY_IDENTIFIER = MappingProxyType(
    {definition.identifier: definition for definition in TARGET_DEFINITIONS}
)


@dataclass(frozen=True)
class RegionConfig:
    """Selection settings for one configured PFC or HPC population.

    Parameters
    ----------
    region : str
        Canonical display region, exactly ``"PFC"`` or ``"HPC"``.
    probe_id : str
        Metadata probe identifier for this region.
    channel_labels : tuple[str, ...]
        Accepted channel-quality labels.
    require_inside_brain : bool
        Whether channel selection requires ``inside_brain``.
    cluster_groups : tuple[str, ...]
        Accepted normalized cluster-quality groups.
    channel_ids : tuple[int, ...]
        Optional ascending explicit zero-based channel restriction.
    """

    region: str
    probe_id: str
    channel_labels: tuple[str, ...]
    require_inside_brain: bool
    cluster_groups: tuple[str, ...]
    channel_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class TaskDecodingConfig:
    """Validated scientific and execution settings for one session analysis.

    Paths are resolved absolute paths for execution. Scientific serialization
    converts them back to canonical paths relative to ``session_root``.

    Parameters
    ----------
    session_metadata_path : Path
        Absolute canonical ``neural_session.json`` path.
    augmented_trial_path : Path
        Absolute augmented-trial CSV path.
    trial_feature_parameter_path : Path
        Absolute behavioral feature-parameter JSON path.
    pfc_region, hpc_region : RegionConfig
        Explicit region-to-probe and unit-selection settings.
    alignment : {"choice_time", "start_time"}
        UTC Unix timestamp column used as time zero, in seconds.
    bin_width_ms : int
        Event-relative firing-rate bin width in milliseconds.
    pfc_pc_count, hpc_pc_count : int
        Positive requested regional PCA component counts.
    target_names : tuple[str, ...]
        Nonempty selected target identifiers in canonical order.
    regularization_mode : {"fixed", "tuned"}
        Decoder regularization policy.
    outer_fold_count, inner_fold_count : int
        Grouped cross-validation fold counts; dimensionless.
    trusted_utc_bounds : Mapping[str, tuple[float, float]]
        Immutable optional probe-to-``(start, end)`` UTC seconds mapping.
    output_root : Path
        Absolute execution-only run-directory parent inside ``session_root``.
    session_root : Path
        Absolute canonical parent of ``session_metadata_path``.
    """

    session_metadata_path: Path
    augmented_trial_path: Path
    trial_feature_parameter_path: Path
    pfc_region: RegionConfig
    hpc_region: RegionConfig
    alignment: str
    bin_width_ms: int
    pfc_pc_count: int
    hpc_pc_count: int
    target_names: tuple[str, ...]
    regularization_mode: str
    outer_fold_count: int
    inner_fold_count: int
    trusted_utc_bounds: Mapping[str, tuple[float, float]]
    output_root: Path
    session_root: Path


_ALLOWED_TOP_LEVEL_FIELDS = {
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
    "regularization_mode",
    "outer_fold_count",
    "inner_fold_count",
    "trusted_utc_bounds",
    "output_root",
}
_REQUIRED_TOP_LEVEL_FIELDS = {
    "session_metadata_path",
    "augmented_trial_path",
    "trial_feature_parameter_path",
    "pfc_region",
    "hpc_region",
}


def _is_json_integer(value: object) -> bool:
    """Identify an exact non-Boolean JSON integer scalar.

    Parameters
    ----------
    value : object
        Decoded JSON value to classify.

    Returns
    -------
    bool
        True only for Python ``int`` values that are not Boolean subclasses.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _resolve_path(value: object, config_parent: Path, field_name: str) -> Path:
    """Resolve one nonempty JSON path relative to a configuration file.

    Parameters
    ----------
    value : object
        Decoded JSON path string.
    config_parent : Path
        Absolute parent directory of the configuration file.
    field_name : str
        Field name used in a validation error.

    Returns
    -------
    Path
        Absolute normalized execution path. It may not yet exist.

    Raises
    ------
    ValueError
        If ``value`` is not a nonempty string.
    """
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = config_parent / path
    return path.resolve()


def _is_within(path: Path, directory: Path) -> bool:
    """Test session-path containment without filesystem mutation.

    Parameters
    ----------
    path : Path
        Absolute candidate path.
    directory : Path
        Absolute allowed root directory.

    Returns
    -------
    bool
        True when ``path`` equals or is nested below ``directory``.
    """
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _validate_string_list(value: object, field_name: str) -> tuple[str, ...]:
    """Validate one nonempty JSON string-list field.

    Parameters
    ----------
    value : object
        Decoded JSON value expected to be a list of strings.
    field_name : str
        Field name used in validation errors.

    Returns
    -------
    tuple[str, ...]
        Immutable ordered string values.

    Raises
    ------
    ValueError
        If the field is empty, not a list, or contains an empty/non-string item.
    """
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a nonempty JSON list.")
    if any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{field_name} must contain nonempty strings.")
    return tuple(value)


def _parse_region(value: object, expected_region: str) -> RegionConfig:
    """Parse one typed PFC or HPC region selection from JSON.

    Parameters
    ----------
    value : object
        Decoded region object containing probe and channel/cluster rules.
    expected_region : str
        Required canonical region token, ``"PFC"`` or ``"HPC"``.

    Returns
    -------
    RegionConfig
        Immutable region selection with sorted zero-based channel IDs.

    Raises
    ------
    ValueError
        If fields are unknown, missing, malformed, or scientifically invalid.
    """
    if not isinstance(value, dict):
        raise ValueError(f"{expected_region.lower()}_region must be a JSON object.")
    allowed_fields = {
        "region",
        "probe_id",
        "channel_labels",
        "require_inside_brain",
        "cluster_groups",
        "channel_ids",
    }
    unknown_fields = set(value) - allowed_fields
    if unknown_fields:
        raise ValueError(f"Unknown region configuration fields: {sorted(unknown_fields)}")
    if value.get("region") != expected_region:
        raise ValueError(f"Region must be exactly {expected_region!r}.")
    probe_id = value.get("probe_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise ValueError("Region probe_id must be a nonempty string.")
    inside_brain = value.get("require_inside_brain")
    if not isinstance(inside_brain, bool):
        raise ValueError("require_inside_brain must be a Boolean.")
    channel_ids_value = value.get("channel_ids", [])
    if not isinstance(channel_ids_value, list):
        raise ValueError("channel_ids must be a JSON list when provided.")
    if any(not _is_json_integer(channel_id) or channel_id < 0 for channel_id in channel_ids_value):
        raise ValueError("channel_ids must contain nonnegative JSON integers only.")
    if len(set(channel_ids_value)) != len(channel_ids_value):
        raise ValueError("channel_ids must not contain duplicates.")
    return RegionConfig(
        region=expected_region,
        probe_id=probe_id,
        channel_labels=_validate_string_list(value.get("channel_labels"), "channel_labels"),
        require_inside_brain=inside_brain,
        cluster_groups=_validate_string_list(value.get("cluster_groups"), "cluster_groups"),
        channel_ids=tuple(sorted(channel_ids_value)),
    )


def _parse_target_names(value: object) -> tuple[str, ...]:
    """Validate a selected target list and normalize it to canonical order.

    Parameters
    ----------
    value : object
        Optional decoded JSON list of case-sensitive target identifiers.

    Returns
    -------
    tuple[str, ...]
        Nonempty canonical-order target identifiers.

    Raises
    ------
    ValueError
        If identifiers are aliases, duplicated, unknown, or malformed.
    """
    if value is None:
        return TARGET_IDENTIFIERS
    if not isinstance(value, list) or not value:
        raise ValueError("target_names must be a nonempty JSON list.")
    if any(not isinstance(name, str) for name in value):
        raise ValueError("target_names must contain strings.")
    if len(set(value)) != len(value):
        raise ValueError("target_names must not contain duplicates.")
    unknown_names = [name for name in value if name not in TARGET_DEFINITION_BY_IDENTIFIER]
    if unknown_names:
        raise ValueError(f"Unknown target_names: {unknown_names}")
    selected_names = set(value)
    return tuple(name for name in TARGET_IDENTIFIERS if name in selected_names)


def _parse_count(value: object, field_name: str, allowed_values: set[int], default: int) -> int:
    """Parse one strict integer count with a finite allowed vocabulary.

    Parameters
    ----------
    value : object
        Optional decoded JSON scalar.
    field_name : str
        Count name used in validation errors.
    allowed_values : set[int]
        Accepted non-Boolean integer values.
    default : int
        Value used when ``value`` is None.

    Returns
    -------
    int
        Valid configured or default count.
    """
    if value is None:
        return default
    if not _is_json_integer(value) or value not in allowed_values:
        raise ValueError(
            f"{field_name} must be one of {sorted(allowed_values)} "
            "as a JSON integer."
        )
    return value


def _parse_positive_count(value: object, field_name: str, default: int) -> int:
    """Parse one requested-PC count as a positive JSON integer.

    Parameters
    ----------
    value : object
        Optional decoded JSON scalar.
    field_name : str
        Component-count name used in validation errors.
    default : int
        Default requested component count.

    Returns
    -------
    int
        Positive requested component count, dimensionless.
    """
    if value is None:
        return default
    if not _is_json_integer(value) or value <= 0:
        raise ValueError(f"{field_name} must be a positive JSON integer.")
    return value


def _parse_trusted_bounds(
    value: object,
    probe_ids: set[str],
) -> Mapping[str, tuple[float, float]]:
    """Validate optional manual-alignment coverage bounds.

    Parameters
    ----------
    value : object
        Optional JSON object mapping configured probe IDs to two numeric UTC
        Unix timestamps in seconds.
    probe_ids : set[str]
        Exact configured probe identifiers allowed as mapping keys.

    Returns
    -------
    Mapping[str, tuple[float, float]]
        Immutable probe mapping in sorted key order. Each tuple is finite
        ``(start, end)`` UTC seconds with ``start < end``.

    Raises
    ------
    ValueError
        If a key is unknown or a bound is Boolean, nonnumeric, non-finite, or
        not strictly ordered.
    """
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, dict):
        raise ValueError("trusted_utc_bounds must be a JSON object.")
    parsed_bounds: dict[str, tuple[float, float]] = {}
    for probe_id, bounds in value.items():
        if probe_id not in probe_ids:
            raise ValueError(f"trusted_utc_bounds has unknown probe {probe_id!r}.")
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError("trusted_utc_bounds values must be [start, end] lists.")
        if any(
            isinstance(bound, bool) or not isinstance(bound, (int, float))
            for bound in bounds
        ):
            raise ValueError("trusted_utc_bounds must contain finite numeric seconds.")
        start, end = float(bounds[0]), float(bounds[1])
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("trusted_utc_bounds must be finite and strictly ordered.")
        parsed_bounds[probe_id] = (start, end)
    return MappingProxyType(
        {probe_id: parsed_bounds[probe_id] for probe_id in sorted(parsed_bounds)}
    )


def _find_source_git_root() -> Path | None:
    """Find the source checkout that owns this module.

    Returns
    -------
    Path or None
        Nearest absolute ancestor containing ``.git``, or None for an
        installed source tree without checkout metadata.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / ".git").exists():
            return candidate
    return None


def load_task_decoding_config(path: Path | str) -> TaskDecodingConfig:
    """Load and validate one portable task-decoding JSON configuration.

    Parameters
    ----------
    path : Path or str
        Configuration JSON path. Relative input paths resolve from its parent.

    Returns
    -------
    TaskDecodingConfig
        Immutable validated settings with absolute execution paths and a
        canonical session root derived from the metadata file's parent.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If JSON, values, paths, containment, or cross-field settings violate
        the task-decoding contract.
    """
    config_path = Path(path).resolve()
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not decode task-decoding configuration {config_path}.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Task-decoding configuration must be a JSON object.")
    unknown_fields = set(payload) - _ALLOWED_TOP_LEVEL_FIELDS
    if unknown_fields:
        raise ValueError(f"Unknown task-decoding configuration fields: {sorted(unknown_fields)}")
    missing_fields = _REQUIRED_TOP_LEVEL_FIELDS - set(payload)
    if missing_fields:
        raise ValueError(f"Missing required configuration fields: {sorted(missing_fields)}")

    config_parent = config_path.parent
    metadata_path = _resolve_path(
        payload["session_metadata_path"],
        config_parent,
        "session_metadata_path",
    )
    if metadata_path.name != CANONICAL_FILENAME:
        raise ValueError(
            f"session_metadata_path must name canonical {CANONICAL_FILENAME}."
        )
    augmented_path = _resolve_path(
        payload["augmented_trial_path"],
        config_parent,
        "augmented_trial_path",
    )
    feature_parameter_path = _resolve_path(
        payload["trial_feature_parameter_path"],
        config_parent,
        "trial_feature_parameter_path",
    )
    session_root = metadata_path.parent
    required_input_paths = (metadata_path, augmented_path, feature_parameter_path)
    for required_path in required_input_paths:
        if not _is_within(required_path, session_root):
            raise ValueError(
                "Required task-decoding input path escapes the metadata session root."
            )
        if not required_path.is_file():
            raise ValueError(f"Required task-decoding input is not a file: {required_path}")
    try:
        feature_parameters = json.loads(feature_parameter_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("trial feature parameter file must decode as JSON.") from exc
    if not isinstance(feature_parameters, dict):
        raise ValueError("trial feature parameter file must decode to a JSON object.")

    pfc_region = _parse_region(payload["pfc_region"], "PFC")
    hpc_region = _parse_region(payload["hpc_region"], "HPC")
    if pfc_region.probe_id == hpc_region.probe_id:
        raise ValueError("PFC and HPC must use distinct configured probe IDs.")

    alignment = payload.get("alignment", "choice_time")
    if not isinstance(alignment, str) or alignment not in {"choice_time", "start_time"}:
        raise ValueError("alignment must be 'choice_time' or 'start_time'.")
    bin_width_ms = _parse_count(
        payload.get("bin_width_ms"),
        "bin_width_ms",
        set(SUPPORTED_BIN_WIDTH_MS),
        100,
    )
    regularization_mode = payload.get("regularization_mode", "fixed")
    if not isinstance(regularization_mode, str) or regularization_mode not in {
        "fixed",
        "tuned",
    }:
        raise ValueError("regularization_mode must be 'fixed' or 'tuned'.")
    if "output_root" in payload:
        output_root = _resolve_path(payload["output_root"], config_parent, "output_root")
    else:
        output_root = session_root / "analysis_runs"
    if not _is_within(output_root, session_root) or output_root == session_root:
        raise ValueError("output_root must be a proper subdirectory of the session root.")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("output_root must be a directory when it already exists.")
    if output_root == config_path or _is_within(config_path, output_root):
        raise ValueError("output_root must not contain the configuration file.")
    for input_path in required_input_paths:
        if (
            output_root == input_path
            or _is_within(input_path, output_root)
            or _is_within(output_root, input_path)
        ):
            raise ValueError("output_root must not overlap a required input path or directory.")
    source_root = _find_source_git_root()
    if source_root is not None and _is_within(output_root, source_root):
        raise ValueError("output_root must not be inside the executing source-code Git checkout.")

    return TaskDecodingConfig(
        session_metadata_path=metadata_path,
        augmented_trial_path=augmented_path,
        trial_feature_parameter_path=feature_parameter_path,
        pfc_region=pfc_region,
        hpc_region=hpc_region,
        alignment=alignment,
        bin_width_ms=bin_width_ms,
        pfc_pc_count=_parse_positive_count(
            payload.get("pfc_pc_count"), "pfc_pc_count", 10
        ),
        hpc_pc_count=_parse_positive_count(
            payload.get("hpc_pc_count"), "hpc_pc_count", 10
        ),
        target_names=_parse_target_names(payload.get("target_names")),
        regularization_mode=regularization_mode,
        outer_fold_count=_parse_count(
            payload.get("outer_fold_count"), "outer_fold_count", {3, 5}, 5
        ),
        inner_fold_count=_parse_count(
            payload.get("inner_fold_count"), "inner_fold_count", {3}, 3
        ),
        trusted_utc_bounds=_parse_trusted_bounds(
            payload.get("trusted_utc_bounds"),
            {pfc_region.probe_id, hpc_region.probe_id},
        ),
        output_root=output_root,
        session_root=session_root,
    )


def _portable_path(path: Path, session_root: Path) -> str:
    """Convert an absolute contained path to portable POSIX text.

    Parameters
    ----------
    path : Path
        Absolute path contained by ``session_root``.
    session_root : Path
        Absolute canonical session root.

    Returns
    -------
    str
        Canonical session-relative path with POSIX separators and no units.
    """
    return path.relative_to(session_root).as_posix()


def _scientific_region_payload(region: RegionConfig) -> dict[str, object]:
    """Return one JSON-safe region selection record in deterministic field order.

    Parameters
    ----------
    region : RegionConfig
        Immutable region selection with zero-based channel IDs.

    Returns
    -------
    dict[str, object]
        JSON-serializable region record preserving the configured selection
        rules. Channel IDs remain dimensionless zero-based indices.
    """
    return {
        "region": region.region,
        "probe_id": region.probe_id,
        "channel_labels": list(region.channel_labels),
        "require_inside_brain": region.require_inside_brain,
        "cluster_groups": list(region.cluster_groups),
        "channel_ids": list(region.channel_ids),
    }


def scientific_config_payload(config: TaskDecodingConfig) -> dict[str, object]:
    """Return a deterministic JSON-safe scientific configuration mapping.

    Parameters
    ----------
    config : TaskDecodingConfig
        Validated configuration whose resolved paths share ``session_root``.

    Returns
    -------
    dict[str, object]
        Portable scientific settings and code-owned frozen controls. Execution
        fields such as ``output_root`` are deliberately excluded.
    """
    return {
        "analysis_version": ANALYSIS_VERSION,
        "session_metadata_path": _portable_path(
            config.session_metadata_path,
            config.session_root,
        ),
        "augmented_trial_path": _portable_path(
            config.augmented_trial_path,
            config.session_root,
        ),
        "trial_feature_parameter_path": _portable_path(
            config.trial_feature_parameter_path,
            config.session_root,
        ),
        "pfc_region": _scientific_region_payload(config.pfc_region),
        "hpc_region": _scientific_region_payload(config.hpc_region),
        "alignment": config.alignment,
        "bin_width_ms": config.bin_width_ms,
        "pfc_pc_count": config.pfc_pc_count,
        "hpc_pc_count": config.hpc_pc_count,
        "target_names": list(config.target_names),
        "regularization_mode": config.regularization_mode,
        "outer_fold_count": config.outer_fold_count,
        "inner_fold_count": config.inner_fold_count,
        "trusted_utc_bounds": {
            probe_id: list(bounds)
            for probe_id, bounds in sorted(config.trusted_utc_bounds.items())
        },
        "frozen_controls": {
            "window_start_s": WINDOW_START_SECONDS,
            "window_end_s": WINDOW_END_SECONDS,
            "seed": RANDOM_SEED,
            "coefficient_tolerance": COEFFICIENT_TOLERANCE,
            "tuning_grid": {
                "LogisticRegression": {
                    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                },
                "ElasticNet": {
                    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
                    "l1_ratio": [0.1, 0.5, 0.9],
                },
            },
            "estimators": {
                "LogisticRegression": {
                    "C": 1.0,
                    "l1_ratio": 0.5,
                    "solver": "saga",
                    "tol": 1e-4,
                    "max_iter": 100,
                    "fit_intercept": True,
                    "class_weight": None,
                    "warm_start": False,
                    "n_jobs": None,
                    "random_state": RANDOM_SEED,
                },
                "ElasticNet": {
                    "alpha": 1.0,
                    "l1_ratio": 0.5,
                    "tol": 1e-4,
                    "max_iter": 1000,
                    "fit_intercept": True,
                    "selection": "cyclic",
                    "positive": False,
                    "warm_start": False,
                },
                "PCA": {
                    "whiten": False,
                    "svd_solver": "auto",
                    "random_state": RANDOM_SEED,
                },
            },
        },
    }
