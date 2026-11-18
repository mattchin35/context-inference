"""Round-1 configuration and independent input-selection contracts (WP1)."""

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.phase_synchrony.results import PhaseSynchronyConfig
from src.neural_analysis.phase_synchrony.preparation import (
    component_availability,
    select_metadata_inputs,
)
from src.neural_analysis.population.session_timeline import prepare_timeline_choices
from src.neural_analysis.population.session_timeline_io import load_timeline_behavior
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)


@pytest.fixture
def metadata_session(tmp_path: Path):
    """Return a resolved 3-site session; paths exist, numerical signals do not load."""
    payload = {
        "schema_version": "2",
        "session": "synthetic_2030-01-02_030405",
        "acquisition": "open_ephys",
        "behavior": {"trials": "trials.csv", "events": "events.csv"},
        "probes": {
            "ProbeA": {
                "lfp": "a/lfp.dat", "alignment": "a/sync.npz",
                "sorter": "a/ks", "quality": "a/quality.csv",
                "sites": {"Cortex": 3},
            },
            "ProbeB": {
                "lfp": "b/lfp.dat", "alignment": "b/sync.npz",
                "sorter": "b/ks", "quality": "b/quality.csv",
                "sites": {"Hippocampus1": 7, "Hippocampus2": 1},
            },
        },
        "site_pairs": [["Hippocampus2", "Cortex"]],
    }
    for folder in ("a", "b"):
        directory = tmp_path / folder
        directory.mkdir()
        for filename in ("lfp.dat", "lfp_preprocessing.json", "sync.npz", "quality.csv"):
            (directory / filename).touch()
        (directory / "ks").mkdir()
    path = tmp_path / "neural_session.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(path), path)


def test_round_one_defaults_are_explicit_and_frozen():
    """No inputs; default record contains agreed seconds/Hz/settings, and is frozen."""
    config = PhaseSynchronyConfig()
    assert config.intervals == ("primary",)
    assert config.components == ("power", "coherence", "lfp-phase")
    assert config.sites == config.pairs == config.unit_ids == ()
    assert config.lfp_site is None
    assert config.phase_rate_hz == 500.0
    assert config.phase_frequencies_hz == tuple(float(f) for f in range(2, 101, 2))
    assert config.notch_enabled is True
    assert config.notch_quality_factor == 30.0
    assert config.numerical_amplitude_threshold == 0.0
    assert config.absolute_amplitude_threshold == 0.0
    assert config.welch_segment_s == 0.5
    assert config.welch_overlap_fraction == 0.5
    assert [(b.name, b.lower_hz, b.upper_hz, b.excluded_intervals_hz) for b in config.bands] == [
        ("theta", 6.0, 10.0, ()), ("gamma", 30.0, 80.0, ((58.0, 62.0),))
    ]
    with pytest.raises(FrozenInstanceError):
        config.notch_enabled = False


@pytest.mark.parametrize("settings", [
    {"intervals": ("before",)},
    {"intervals": ()},
    {"components": ("itpc",)},
    {"phase_rate_hz": 0.0},
    {"phase_frequencies_hz": (8.0, 6.0)},
    {"phase_frequencies_hz": (0.0, 8.0)},
    {"notch_quality_factor": 0.0},
    {"numerical_amplitude_threshold": -1.0},
    {"absolute_amplitude_threshold": float("nan")},
    {"welch_segment_s": 0.0},
    {"welch_overlap_fraction": 1.0},
    {"components": ("spike-ppc",)},
    {"components": ("spike-ppc",), "unit_ids": ("17",), "lfp_site": "Cortex"},
    {"components": ("spike-ppc",), "unit_ids": ("ProbeA:17",)},
])
def test_invalid_configuration_is_rejected(settings: dict):
    """A settings mapping with invalid units/choices/selection must raise ValueError."""
    with pytest.raises(ValueError):
        PhaseSynchronyConfig(**settings)


def test_explicit_spike_selection_and_both_centered_intervals():
    """Explicit probe-qualified categorical IDs are preserved without pooling."""
    config = PhaseSynchronyConfig(
        components=("spike-ppc",), intervals=("primary", "secondary"),
        unit_ids=("ProbeB:17", "ProbeA:17"), lfp_site="Hippocampus1",
    )
    assert config.unit_ids == ("ProbeB:17", "ProbeA:17")
    assert config.intervals == ("primary", "secondary")


def test_metadata_sites_and_ordered_pairs_remain_authoritative(metadata_session):
    """Resolved metadata input returns declared SiteMetadata tuple and ordered pairs."""
    sites, pairs = select_metadata_inputs(metadata_session)
    assert [(s.site_id, s.probe_id, s.saved_channel_index) for s in sites] == [
        ("Cortex", "ProbeA", 3), ("Hippocampus1", "ProbeB", 7),
        ("Hippocampus2", "ProbeB", 1),
    ]
    assert pairs == (("Hippocampus2", "Cortex"),)


def test_absent_pair_definitions_use_distinct_metadata_site_pairs(metadata_session):
    """Empty pair metadata yields combinations in site declaration order, no self pairs."""
    from dataclasses import replace

    sites, pairs = select_metadata_inputs(replace(metadata_session, site_pairs=()))
    assert len(sites) == 3
    assert pairs == (
        ("Cortex", "Hippocampus1"), ("Cortex", "Hippocampus2"),
        ("Hippocampus1", "Hippocampus2"),
    )


def test_missing_site_is_never_fabricated_or_substituted(metadata_session):
    """Unknown categorical site selection raises instead of reusing another channel."""
    with pytest.raises(ValueError, match="HPC2"):
        select_metadata_inputs(metadata_session, site_ids=("HPC2",))


def test_missing_other_probe_and_spikes_leave_independent_power_available(metadata_session):
    """Availability maps component-selection keys to None or a concrete missing reason."""
    metadata_session.probes[1].lfp_file.unlink()
    config = PhaseSynchronyConfig()
    status = component_availability(metadata_session, config)
    assert status["power:Cortex"] is None
    assert status["power:Hippocampus1"]
    assert status["coherence:Hippocampus2:Cortex"]
    assert status["lfp-phase:Hippocampus2:Cortex"]
    assert all("spike-ppc" not in key for key in status)


def test_one_selected_site_power_needs_no_pair_spikes_or_other_probe(metadata_session):
    """Power-only availability checks the selected site's LFP/sync prerequisites only."""
    metadata_session.probes[1].lfp_file.unlink()
    status = component_availability(
        metadata_session, PhaseSynchronyConfig(components=("power",), sites=("Cortex",))
    )
    assert status == {"power:Cortex": None}


def test_ordinary_choices_reuse_original_rows_utc_origin_and_actions(metadata_session):
    """Real behavior CSV loader preserves (choice,) IDs and UTC/elapsed-second conventions."""
    origin = 1_800_000_000.0
    trials = pd.DataFrame({
        "cur_trial": [21, 23, 29, 31, 37],
        "start_time": origin + np.array([0, 5, 10, 15, 20]),
        "choice_time": origin + np.array([2.125, 7, 12, np.nan, 22.375]),
        "action": [0, 1, 1, 0, 1],
        "reward": [1, 0, 1, 1, 2],
        "experimenter_reward_given": [0, 1, 0, 0, 0],
    }, index=[101, 103, 107, 109, 113])
    original = trials.copy(deep=True)
    trials.to_csv(metadata_session.behavior.trial_table_file, index=False)
    pd.DataFrame({"Time": [origin + 22.25, origin + 1], "Event": ["lick", "LED"]}).to_csv(
        metadata_session.behavior.event_table_file, index=False
    )
    expected_origin, expected_choices = prepare_timeline_choices(trials)
    actual_origin, choices, events = load_timeline_behavior(metadata_session)
    assert actual_origin == expected_origin == origin
    pd.testing.assert_frame_equal(choices, expected_choices)
    pd.testing.assert_frame_equal(trials, original)
    assert choices.trial_id.tolist() == [21, 29, 37]
    assert choices.row_position.tolist() == [0, 2, 4]
    assert choices.action.tolist() == [0, 1, 1]
    np.testing.assert_allclose(choices.elapsed_s, [2.125, 12, 22.375], rtol=0, atol=1e-7)
    assert np.isnan(choices.reward.iloc[-1])
    np.testing.assert_array_equal(events.elapsed_s, [1, 22.25])
