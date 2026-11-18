"""Named-array persistence with provenance and no overwritten results."""

import numpy as np
import pytest
from src.neural_analysis.population_decoding import results, pipeline


def test_preparation_roundtrip_is_self_contained(synthetic_population, tmp_path):
    """Input known population/temp dir; all Hz data, IDs and source labels survive NPZ."""
    path = tmp_path / "preparation.npz"
    results.save_preparation(path, synthetic_population, {"preparation_id": "fixture"})
    loaded = results.load_preparation(path)
    np.testing.assert_array_equal(loaded.regions["PFC"].rates_hz, synthetic_population.regions["PFC"].rates_hz)
    np.testing.assert_array_equal(loaded.targets.values, synthetic_population.targets.values)
    assert loaded.regions["PFC"].unit_ids == synthetic_population.regions["PFC"].unit_ids
    assert loaded.meta["preparation_id"] == "fixture"
    with np.load(path, allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        results.save_preparation(path, loaded, {})
    assert path.read_bytes() == before


def test_cell_roundtrip_and_corruption_failure(synthetic_population, tmp_path):
    """Input population/temp dir; numeric fold parameters and metrics round-trip, corruption fails."""
    cell = pipeline.fit_target(synthetic_population, "PFC", "outcome")
    path = tmp_path / "PFC_outcome.npz"
    results.save_cell(path, cell, {"model_id": "test"})
    loaded = results.load_cell(path)
    np.testing.assert_array_equal(loaded["pcs_probabilities"], cell["pcs_probabilities"])
    assert loaded["meta"]["model_id"] == "test"
    assert loaded["meta"]["class_names"] == ["left_reward", "right_reward", "left_no_reward", "right_no_reward"]
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        results.load_cell(path)
