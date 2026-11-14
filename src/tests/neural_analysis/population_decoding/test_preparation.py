"""IRIG coverage, regional window identity, and source-field minimality."""

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population_decoding.configuration import load_config
from src.neural_analysis.population_decoding.preparation import prepare_session


@pytest.mark.parametrize("width", [500, 100, 50, 20])
def test_regional_rates_geometry_and_shared_folds(population_config_path, width):
    """Input synthetic session/width ms; count-derived Hz arrays have (W,2000/ms,U)."""
    from dataclasses import replace

    config = replace(load_config(population_config_path), bin_width_ms=width)
    for path in (config.augmented_trials, config.session_root / "trials.csv"):
        table = pd.read_csv(path).drop(columns="correct")
        table.to_csv(path, index=False)
    prepared = prepare_session(config)
    assert set(prepared.regions) == {"PFC", "HPC"}
    assert not prepared.unavailable
    pfc, hpc = prepared.regions["PFC"], prepared.regions["HPC"]
    assert len(pfc.row_positions) > len(hpc.row_positions)
    for region in prepared.regions.values():
        assert region.rates_hz.shape == (len(region.row_positions), 2000 // width, 2)
        np.testing.assert_allclose(region.bin_edges_s[[0, -1]], [-1, 1])
        np.testing.assert_allclose(region.rates_hz * width / 1000,
                                   np.round(region.rates_hz * width / 1000), atol=1e-10)
        assert set(region.unit_ids) <= {"A:7", "A:8", "B:7", "B:8"}
        np.testing.assert_array_equal(region.bin_rows.fold_id.to_numpy().reshape(-1, 2000 // width)[:, 0],
                                      prepared.fold_ids[region.row_positions])


def test_manual_alignment_unavailable_region_not_guessed(population_config_path):
    """Input session; remove HPC IRIG and retain independent PFC rates, return None."""
    config = load_config(population_config_path)
    path = config.session_root / "B/aligned.npz"
    with np.load(path) as archive:
        spikes = archive["spike_utc_unix"]
    np.savez(path, spike_utc_unix=spikes)
    prepared = prepare_session(config)
    assert "PFC" in prepared.regions and "HPC" not in prepared.regions
    assert "IRIG" in prepared.unavailable["HPC"]
