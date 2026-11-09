"""Small controls have fixed scientific settings and contained input locators."""

import json

import pytest

from src.neural_analysis.task_decoding.session_prediction import config


def test_defaults_and_explicit_bin_override(prediction_inputs):
    """Metadata path input returns 500-ms defaults; explicit width wins over JSON."""
    base, _, _ = prediction_inputs
    result = config.load_prediction_config(base.session_metadata_path)
    assert result.bin_width_ms == 500
    assert result.analysis_mode == "correct_rewarded_generalization"
    assert result.base_config_path == base.session_root / "task_decoding_config.json"
    assert config.RIDGE_ALPHA == config.LOGISTIC_C == 1.0
    controls = base.session_root / "session_prediction_config.json"
    controls.write_text(json.dumps({"bin_width_ms": 100}))
    assert config.load_prediction_config(base.session_metadata_path).bin_width_ms == 100
    assert config.load_prediction_config(base.session_metadata_path, bin_width_ms=20).bin_width_ms == 20


@pytest.mark.parametrize("payload", [
    {"ridge_alpha": 1}, {"logistic_C": 1}, {"unknown": 1},
    {"bin_width_ms": 200}, {"bin_width_ms": True},
    {"base_config_path": "../outside.json"}, [],
])
def test_invalid_controls_raise(prediction_inputs, payload):
    """Malformed controls fail rather than silently applying different science."""
    base, _, _ = prediction_inputs
    (base.session_root / "session_prediction_config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        config.load_prediction_config(base.session_metadata_path)


def test_explicit_missing_or_malformed_control_file_fails(prediction_inputs):
    """Only an absent default controls file uses defaults; explicit paths must exist."""
    base, _, _ = prediction_inputs
    with pytest.raises(FileNotFoundError):
        config.load_prediction_config(base.session_metadata_path, config_path=base.session_root / "missing.json")
    path = base.session_root / "session_prediction_config.json"
    path.write_text("{")
    with pytest.raises(ValueError):
        config.load_prediction_config(base.session_metadata_path)


def test_mode_controls_override_and_fixed_target_sets(prediction_inputs):
    """Mode strings select fixed ordered targets; explicit mode overrides JSON."""
    base, _, _ = prediction_inputs
    controls = base.session_root / "session_prediction_config.json"
    controls.write_text(json.dumps({"analysis_mode": "mixed_outcome_decoding"}))
    result = config.load_prediction_config(base.session_metadata_path)
    assert result.analysis_mode == "mixed_outcome_decoding"
    assert config.target_names_for_mode(result.analysis_mode) == ("current_state", "current_action", "current_reward")
    result = config.load_prediction_config(
        base.session_metadata_path, analysis_mode="correct_rewarded_generalization",
    )
    assert config.target_names_for_mode(result.analysis_mode) == config.TARGET_NAMES


@pytest.mark.parametrize("mode", ["mixed", "", None, 1, ["mixed_outcome_decoding"]])
def test_invalid_mode_in_controls_fails(prediction_inputs, mode):
    """Malformed mode controls cannot silently select another scientific analysis."""
    base, _, _ = prediction_inputs
    (base.session_root / "session_prediction_config.json").write_text(
        json.dumps({"analysis_mode": mode}),
    )
    with pytest.raises(ValueError, match="mode"):
        config.load_prediction_config(base.session_metadata_path)


@pytest.mark.parametrize("payload", [
    {"target_names": ["current_action"]},
    {"training_condition_names": ["all"]},
    {"analysis_mode": "mixed_outcome_decoding", "logistic_C": 2},
])
def test_mode_does_not_allow_arbitrary_scientific_controls(prediction_inputs, payload):
    """Mode selection introduces no target, condition, or penalty knobs."""
    base, _, _ = prediction_inputs
    (base.session_root / "session_prediction_config.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        config.load_prediction_config(base.session_metadata_path)


def test_invalid_explicit_mode_fails(prediction_inputs):
    """A noncanonical explicit mode fails before preparation or any raw reads."""
    base, _, _ = prediction_inputs
    with pytest.raises(ValueError, match="mode"):
        config.load_prediction_config(base.session_metadata_path, analysis_mode="mixed")


def test_supervised_window_defaults_and_precedence(prediction_inputs):
    """Old controls retain Full; an explicit window overrides the saved setting."""
    base, _, _ = prediction_inputs
    assert config.load_prediction_config(base.session_metadata_path).training_window == "full"
    controls = base.session_root / "session_prediction_config.json"
    for window in ("full", "before", "after"):
        controls.write_text(json.dumps({"training_window": window}))
        assert config.load_prediction_config(base.session_metadata_path).training_window == window
        assert config.load_prediction_config(base.session_metadata_path, training_window="before").training_window == "before"


@pytest.mark.parametrize("window", [None, "pre_choice", "whole", "", 1, [-1, 0]])
def test_invalid_supervised_windows_fail(prediction_inputs, window):
    """Only the three fixed identifiers are scientific controls; arbitrary bounds fail."""
    base, _, _ = prediction_inputs
    (base.session_root / "session_prediction_config.json").write_text(json.dumps({"training_window": window}))
    with pytest.raises(ValueError, match="window"):
        config.load_prediction_config(base.session_metadata_path)
