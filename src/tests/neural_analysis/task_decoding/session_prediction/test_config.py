"""Small controls have fixed scientific settings and contained input locators."""

import json

import pytest

from src.neural_analysis.task_decoding.session_prediction import config


def test_defaults_and_explicit_bin_override(prediction_inputs):
    """Metadata path input returns 500-ms defaults; explicit width wins over JSON."""
    base, _, _ = prediction_inputs
    result = config.load_prediction_config(base.session_metadata_path)
    assert result.bin_width_ms == 500
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
