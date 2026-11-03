"""Small preparation controls; scientific readout settings are revision constants."""

from dataclasses import dataclass
import json
from pathlib import Path

from src.neural_analysis.task_decoding.config import SUPPORTED_BIN_WIDTH_MS


ANALYSIS_VERSION = "condition-session-prediction-v1"
SCHEMA_VERSION = 1
RIDGE_ALPHA = 1.0
LOGISTIC_C = 1.0
FOLD_COUNT = 5
TARGET_NAMES = (
    "current_state", "current_action", "qlearning_relative_value",
    "forgetting_q_relative_value", "hmm_signed_belief",
    "hmm_decay_signed_belief", "relative_doubt",
)


@dataclass(frozen=True)
class SessionPredictionConfig:
    """Resolved contained input paths and a bin width in integer milliseconds.

    session_metadata_path/base_config_path are absolute paths; bin_width_ms
    selects the matching saved PCA, never a resampling of another basis.
    """

    session_metadata_path: Path
    base_config_path: Path
    bin_width_ms: int


def load_prediction_config(
    metadata_path: Path | str,
    *,
    config_path: Path | str | None = None,
    bin_width_ms: int | None = None,
) -> SessionPredictionConfig:
    """Load optional JSON controls and return resolved paths and milliseconds.

    metadata_path names an existing neural_session.json. An absent default
    controls file uses defaults; an explicit config_path must exist. Relative
    base-config locators resolve from the control file, inside this session.
    An explicit supported integer bin_width_ms overrides the file's value.
    This function reads only controls; base decoder validation occurs at prepare.
    """
    metadata = Path(metadata_path).resolve(strict=True)
    if metadata.name != "neural_session.json":
        raise ValueError("metadata_path must name neural_session.json")
    root = metadata.parent
    controls = Path(config_path).resolve() if config_path is not None else root / "session_prediction_config.json"
    if not controls.is_relative_to(root):
        raise ValueError("Prediction controls must be inside the metadata session")
    payload = {}
    if controls.exists() or config_path is not None:
        try:
            payload = json.loads(controls.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("Prediction controls must decode as JSON") from error
    if not isinstance(payload, dict) or set(payload) - {"base_config_path", "bin_width_ms"}:
        raise ValueError("Prediction controls have unknown fields or are not an object")
    width = payload.get("bin_width_ms", 500)
    if type(width) is not int or width not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError("bin_width_ms must be one of 500, 100, 50, 20")
    if bin_width_ms is not None:
        if type(bin_width_ms) is not int or bin_width_ms not in SUPPORTED_BIN_WIDTH_MS:
            raise ValueError("bin_width_ms override must be one of 500, 100, 50, 20")
        width = bin_width_ms
    locator = payload.get("base_config_path", "task_decoding_config.json")
    if not isinstance(locator, str) or not locator.strip():
        raise ValueError("base_config_path must be a nonempty path")
    base_path = (controls.parent / locator).resolve()
    if not base_path.is_relative_to(root):
        raise ValueError("base_config_path must remain inside the metadata session")
    return SessionPredictionConfig(metadata, base_path, width)
