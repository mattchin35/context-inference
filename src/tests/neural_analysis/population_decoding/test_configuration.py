"""Portable configuration and immutable scientific defaults."""

import json
import pytest

from src.neural_analysis.population_decoding.configuration import load_config


def test_defaults_and_session_local_paths(population_config_path):
    """Input temporary config Path; assert fixed timing/representations and paths."""
    config = load_config(population_config_path)
    assert config.bin_width_ms == 500
    assert config.window_s == (-1.0, 1.0)
    assert config.representations == ("units", "pcs")
    assert config.fold_count == 5
    assert config.intermediate_root == config.session_root / "processed/population_task_decoding"
    assert config.probe_roles == {"PFC": "A", "HPC": "B"}
    assert not config.intermediate_root.exists()


@pytest.mark.parametrize("field,value", [
    ("bin_width_ms", True), ("bin_width_ms", 200), ("old_count_archive", "old.npz"),
    ("analysis_version", "joint-task-encoding-v2"), ("C", 2),
    ("probe_roles", {"PFC": "A", "HPC": "A"}),
])
def test_invalid_controls_rejected(population_config_path, field, value):
    """Input JSON field/value; invalid analysis controls raise ValueError."""
    payload = json.loads(population_config_path.read_text())
    payload[field] = value
    population_config_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_config(population_config_path)


@pytest.mark.parametrize("width", [500, 100, 50, 20])
def test_supported_resolutions(population_config_path, width):
    """Input width in ms; accepted config retains integer bin width."""
    payload = json.loads(population_config_path.read_text())
    payload["bin_width_ms"] = width
    population_config_path.write_text(json.dumps(payload))
    assert load_config(population_config_path).bin_width_ms == width
