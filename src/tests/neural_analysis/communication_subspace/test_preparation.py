"""Input selection and binning contracts for communication subspace analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.communication_subspace.configuration import (
    CommunicationConfig,
    validate_configuration,
)
from src.neural_analysis.communication_subspace.preparation import (
    inspect_session_inputs,
    load_session_inputs,
    draw_populations,
)


ROLES = {"PFC": "A", "HPC": "B"}


@pytest.mark.parametrize("width,bins", [(100, 40), (500, 8)])
def test_good_only_paired_counts_and_original_trial_annotations(
    synthetic_communication_session, width, bins,
):
    """A metadata session produces paired counts (W,B,U), original rows, and IDs."""
    config = CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=width)
    prepared = load_session_inputs(config)

    assert prepared.session_id == "synthetic-communication"
    assert prepared.time_origin_s == 1_700_000_000.0 + 6.0
    np.testing.assert_array_equal(prepared.choice_positions, np.arange(45))
    np.testing.assert_array_equal(prepared.selected_trials.cur_trial, np.arange(100, 145))
    assert prepared.selected_trials.context_label.iloc[4] == "right"
    assert prepared.selected_trials.outcome_note.iloc[4] == "outcome-4"
    np.testing.assert_allclose(prepared.bin_edges_s, np.linspace(-2, 2, bins + 1))
    for region, probe in ROLES.items():
        assert prepared.counts[region].shape == (45, bins, 8)
        assert prepared.counts[region].dtype == np.float64
        assert prepared.unit_ids[region] == tuple(f"{probe}:{unit}" for unit in range(8))
        np.testing.assert_array_equal(prepared.counts[region].sum(axis=1), np.ones((45, 8)))
        assert prepared.coverage_s[region] == (1_700_000_000.0, 1_700_000_470.0)
    assert prepared.configuration["population_quality"] == "good"
    assert prepared.configuration["missing_label_policy"] == "exclude"


def test_metadata_inspection_is_spike_free_and_events_optional(
    synthetic_communication_session, monkeypatch,
):
    """Readiness sees labels, coverage, choices, and counts without raw spike loading."""
    config = CommunicationConfig(synthetic_communication_session, ROLES)
    def forbidden_spike_load(*args, **kwargs):
        """Reject raw spike loading during metadata-only inspection."""
        raise AssertionError("inspection must not load raw spikes")

    monkeypatch.setattr(
        "src.neural_analysis.population.session_timeline_io.load_timeline_population",
        forbidden_spike_load,
    )
    inspection = inspect_session_inputs(config)

    assert inspection.session_id == "synthetic-communication"
    assert inspection.probe_roles == ROLES
    assert len(inspection.choices) == 45
    assert len(inspection.selected_trials) == 45
    np.testing.assert_array_equal(inspection.choice_positions, np.arange(45))
    assert inspection.unit_ids["PFC"] == tuple(f"A:{unit}" for unit in range(8))
    assert inspection.unit_ids["HPC"] == tuple(f"B:{unit}" for unit in range(8))
    assert set(inspection.coverage_s) == {"PFC", "HPC"}


def test_missing_spike_source_disables_readiness(synthetic_communication_session):
    """Readiness checks required source presence without reading raw arrays."""
    (synthetic_communication_session.parent / "A" / "spike_clusters.npy").unlink()
    with pytest.raises(ValueError, match="spike|source|missing|file"):
        inspect_session_inputs(CommunicationConfig(synthetic_communication_session, ROLES))


def test_common_coverage_excludes_only_unsupported_choice_windows(synthetic_communication_session):
    """The two probes share exactly the windows fully inside both IRIG ranges."""
    root = synthetic_communication_session.parent
    origin = 1_700_000_000.0
    npz = root / "A" / "aligned.npz"
    with np.load(npz) as archive:
        times = archive["spike_utc_unix"]
    np.savez(npz, spike_utc_unix=times, irig_utc_unix=[origin + 9, origin + 442])
    npz = root / "B" / "aligned.npz"
    with np.load(npz) as archive:
        times = archive["spike_utc_unix"]
    np.savez(npz, spike_utc_unix=times, irig_utc_unix=[origin, origin + 449])

    prepared = load_session_inputs(CommunicationConfig(synthetic_communication_session, ROLES))

    np.testing.assert_array_equal(prepared.choice_positions, np.arange(1, 44))
    np.testing.assert_array_equal(prepared.selected_trials.cur_trial, np.arange(101, 144))
    assert len(prepared.choices) == 45
    assert prepared.counts["PFC"].shape == prepared.counts["HPC"].shape


def test_overlapping_and_touching_windows_preserve_choice_axis_and_half_open_bins(
    synthetic_communication_session,
):
    """A spike at a shared endpoint is counted for the later window only."""
    root = synthetic_communication_session.parent
    trials = pd.read_csv(root / "trials.csv")
    first = float(trials.choice_time.iloc[0])
    trials.loc[1, "choice_time"] = first + 2.0
    trials.loc[2, "choice_time"] = first + 4.0
    trials.to_csv(root / "trials.csv", index=False)
    # Other selected units must still be represented by sorter assignments.
    # A disjoint gap spike per unit does not affect the checked first windows.
    for probe in ROLES.values():
        clusters = np.r_[np.array([0, 0, 0]), np.arange(1, 11)]
        times = np.r_[np.array([first - 2, first + 2, first + 4]), first + 460 + np.arange(1, 11) * .01]
        order = np.argsort(times, kind="stable")
        np.save(root / probe / "spike_clusters.npy", clusters[order])
        np.savez(root / probe / "aligned.npz", spike_utc_unix=times[order],
                 irig_utc_unix=[1_700_000_000.0, 1_700_000_470.0])

    prepared = load_session_inputs(CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=500))

    for region in ROLES:
        assert prepared.counts[region].shape == (45, 8, 8)
        np.testing.assert_array_equal(prepared.counts[region][:3, :, 0].sum(axis=1), [1, 1, 2])
        assert prepared.counts[region][0, -1, 0] == 0
        assert prepared.counts[region][1, 0, 0] == 0
        assert prepared.counts[region][2, 0, 0] == 1


def test_config_and_population_draws_require_valid_inputs(synthetic_communication_session):
    """Controls reject invalid widths/roles and draws use all four stable axes."""
    for width in (50, 250, True):
        with pytest.raises(ValueError, match="bin|width"):
            validate_configuration(CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=width))
    for roles in ({"PFC": "A"}, {"PFC": "A", "HPC": "A"}, {}):
        with pytest.raises(ValueError, match="role|probe|PFC|HPC"):
            validate_configuration(CommunicationConfig(synthetic_communication_session, roles))
    ids = {"PFC": tuple(f"A:{i}" for i in range(9)), "HPC": tuple(f"B:{i}" for i in range(8))}
    first = draw_populations(ids, seed=0)
    second = draw_populations(ids, seed=0)
    assert len(first) == 5
    for left, right in zip(first, second):
        for region in ("PFC", "HPC"):
            np.testing.assert_array_equal(left.source_indices[region], right.source_indices[region])
            np.testing.assert_array_equal(left.target_indices[region], right.target_indices[region])
            assert len(left.source_indices[region]) == len(left.target_indices[region]) == 4
            assert not set(left.source_indices[region]) & set(left.target_indices[region])
    with pytest.raises(ValueError, match="unit|population|n"):
        draw_populations({"PFC": ids["PFC"][:3], "HPC": ids["HPC"]})


def test_missing_good_labels_and_nonfinite_spikes_fail_explicitly(synthetic_communication_session):
    """KSLabel cannot rescue absent manual good labels; invalid times fail loading."""
    root = synthetic_communication_session.parent
    labels = root / "A" / "cluster_info.tsv"
    table = pd.read_csv(labels, sep="\t")
    table["group"] = ""
    table.to_csv(labels, sep="\t", index=False)
    config = CommunicationConfig(synthetic_communication_session, ROLES)
    with pytest.raises(ValueError, match="good|manual|unit"):
        inspect_session_inputs(config)

    table.loc[:7, "group"] = "good"
    table.to_csv(labels, sep="\t", index=False)
    aligned = root / "A" / "aligned.npz"
    with np.load(aligned) as archive:
        spikes = archive["spike_utc_unix"].copy()
        coverage = archive["irig_utc_unix"].copy()
    spikes[0] = np.nan
    np.savez(aligned, spike_utc_unix=spikes, irig_utc_unix=coverage)
    with pytest.raises(ValueError, match="finite|nan|spike"):
        load_session_inputs(config)
