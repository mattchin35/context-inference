"""Fixed transfer definitions and contained, metadata-only run controls."""

from dataclasses import dataclass
import json
from pathlib import Path

from src.neural_analysis.session_metadata import CANONICAL_FILENAME, load_session_metadata
from src.neural_analysis.task_decoding.config import SUPPORTED_BIN_WIDTH_MS, load_task_decoding_config


ANALYSIS_VERSION = "cross-condition-generalization-v1"
SCHEMA_VERSION = 1
TARGET_NAMES = ("current_state", "current_action", "current_reward", "current_action_is_correct")
WINDOW_NAMES = ("full", "before", "after")
WEIGHTING_NAMES = ("equal_trial", "equal_cell")


@dataclass(frozen=True)
class TransferDefinition:
    """One directed split, with source/test cell names ordered by actual class.

    split_id and target_name are stable dimensionless identifiers. source_cells
    and test_cells are tuples of two distinct named cells, class 0 then class 1.
    The four cells are disjoint within one split; no fold or fitted state exists.
    """

    split_id: str
    target_name: str
    source_cells: tuple[str, str]
    test_cells: tuple[str, str]


TRANSFERS = (
    TransferDefinition("context_correct_to_incorrect", "current_state",
                       ("context_right_correct", "context_left_correct"),
                       ("context_right_incorrect", "context_left_incorrect")),
    TransferDefinition("context_left_choice_to_right_choice", "current_state",
                       ("context_right_incorrect", "context_left_correct"),
                       ("context_right_correct", "context_left_incorrect")),
    TransferDefinition("context_right_choice_to_left_choice", "current_state",
                       ("context_right_correct", "context_left_incorrect"),
                       ("context_right_incorrect", "context_left_correct")),
    TransferDefinition("context_incorrect_to_correct", "current_state",
                       ("context_right_incorrect", "context_left_incorrect"),
                       ("context_right_correct", "context_left_correct")),
    TransferDefinition("choice_correct_to_incorrect", "current_action",
                       ("choice_right_correct", "choice_left_correct"),
                       ("choice_right_incorrect", "choice_left_incorrect")),
    TransferDefinition("choice_left_context_to_right_context", "current_action",
                       ("choice_right_incorrect", "choice_left_correct"),
                       ("choice_right_correct", "choice_left_incorrect")),
    TransferDefinition("choice_right_context_to_left_context", "current_action",
                       ("choice_right_correct", "choice_left_incorrect"),
                       ("choice_right_incorrect", "choice_left_correct")),
    TransferDefinition("choice_incorrect_to_correct", "current_action",
                       ("choice_right_incorrect", "choice_left_incorrect"),
                       ("choice_right_correct", "choice_left_correct")),
    TransferDefinition("reward_left_choice_to_right_choice", "current_reward",
                       ("choice_left_unrewarded", "choice_left_rewarded"),
                       ("choice_right_unrewarded", "choice_right_rewarded")),
    TransferDefinition("reward_left_rewarded_right_unrewarded", "current_reward",
                       ("choice_right_unrewarded", "choice_left_rewarded"),
                       ("choice_left_unrewarded", "choice_right_rewarded")),
    TransferDefinition("reward_right_rewarded_left_unrewarded", "current_reward",
                       ("choice_left_unrewarded", "choice_right_rewarded"),
                       ("choice_right_unrewarded", "choice_left_rewarded")),
    TransferDefinition("reward_right_choice_to_left_choice", "current_reward",
                       ("choice_right_unrewarded", "choice_right_rewarded"),
                       ("choice_left_unrewarded", "choice_left_rewarded")),
    TransferDefinition("value_left_choice_to_right_choice", "current_action_is_correct",
                       ("choice_left_incorrect", "choice_left_correct"),
                       ("choice_right_incorrect", "choice_right_correct")),
    TransferDefinition("value_left_correct_right_incorrect", "current_action_is_correct",
                       ("choice_right_incorrect", "choice_left_correct"),
                       ("choice_left_incorrect", "choice_right_correct")),
    TransferDefinition("value_right_correct_left_incorrect", "current_action_is_correct",
                       ("choice_left_incorrect", "choice_right_correct"),
                       ("choice_right_incorrect", "choice_left_correct")),
    TransferDefinition("value_right_choice_to_left_choice", "current_action_is_correct",
                       ("choice_right_incorrect", "choice_right_correct"),
                       ("choice_left_incorrect", "choice_left_correct")),
)


@dataclass(frozen=True)
class CrossConditionConfig:
    """Absolute metadata/base-config Paths, integer bin-width ms and regional roles.

    probe_roles maps PFC/HPC to version-3 metadata probe identifiers. No files
    containing neural arrays are opened and no estimator settings are configurable.
    """

    session_metadata_path: Path
    base_config_path: Path
    bin_width_ms: int
    probe_roles: dict[str, str]


def load_cross_condition_config(
    metadata_path: Path | str,
    *,
    config_path: Path | str | None = None,
    bin_width_ms: int | None = None,
) -> CrossConditionConfig:
    """Read small controls/base JSON and return validated contained run settings.

    metadata_path names version-3 neural_session.json; optional config_path names
    contained controls, default cross_condition_config.json. An absent default
    uses defaults; an absent explicit path raises FileNotFoundError. A contained
    base_config_path resolves relative to controls and must reference the selected
    metadata. bin_width_ms is a supported exact integer override, or None.
    Returns CrossConditionConfig with absolute Paths, milliseconds and role IDs;
    reads no neural or behavioral numerical data and never creates output files.
    """
    metadata_path = Path(metadata_path).resolve(strict=True)
    if metadata_path.name != CANONICAL_FILENAME:
        raise ValueError(f"metadata_path must name {CANONICAL_FILENAME}")
    metadata = load_session_metadata(metadata_path)
    if metadata.schema_version != "3" or metadata.analysis is None:
        raise ValueError("Cross-condition generalization requires version-3 metadata with analysis roles")
    root = metadata_path.parent
    controls_path = (Path(config_path).resolve() if config_path is not None
                     else root / "cross_condition_config.json")
    if not controls_path.is_relative_to(root):
        raise ValueError("Cross-condition controls must be inside the metadata session")
    payload = {}
    if controls_path.exists() or config_path is not None:
        try:
            payload = json.loads(controls_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Cross-condition controls must decode as JSON") from error
    if not isinstance(payload, dict) or set(payload) - {"base_config_path", "bin_width_ms"}:
        raise ValueError("Cross-condition controls must be an object with only base_config_path/bin_width_ms")
    width = _validated_bin_width(payload.get("bin_width_ms", 500))
    if bin_width_ms is not None:
        width = _validated_bin_width(bin_width_ms)
    locator = payload.get("base_config_path", "task_decoding_config.json")
    if not isinstance(locator, str) or not locator.strip():
        raise ValueError("base_config_path must be a nonempty path string")
    base_path = (controls_path.parent / locator).resolve()
    if not base_path.is_relative_to(root):
        raise ValueError("base_config_path must remain inside the metadata session")
    base = load_task_decoding_config(base_path)
    if base.session_metadata_path != metadata_path:
        raise ValueError("Base configuration must reference the selected session metadata")
    roles = metadata.analysis.probe_roles
    return CrossConditionConfig(metadata_path, base_path, width, {"PFC": roles.pfc, "HPC": roles.hpc})


def _validated_bin_width(value: object) -> int:
    """Return a supported exact integer millisecond width; reject other scalar types/values."""
    if type(value) is not int or value not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError("bin_width_ms must be one of 500, 100, 50, 20")
    return value
