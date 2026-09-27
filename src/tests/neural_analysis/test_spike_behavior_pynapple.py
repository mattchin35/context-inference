import numpy as np
import pandas as pd
import pynapple as nap
from pathlib import Path
import matplotlib.pyplot as plt

import src.neural_analysis.spike_behavior_pynapple as spike_behavior_pynapple


def test_normalize_region_name_for_filename_accepts_known_regions():
    """Region labels used in filenames should have one canonical spelling."""
    assert spike_behavior_pynapple.normalize_region_name_for_filename("hpc") == "HPC"
    assert spike_behavior_pynapple.normalize_region_name_for_filename(" V1 ") == "V1"
    assert spike_behavior_pynapple.normalize_region_name_for_filename("pfc") == "PFC"


def test_normalize_region_name_for_filename_rejects_unknown_regions():
    """Unexpected region labels should fail before creating ambiguous CSV names."""
    try:
        spike_behavior_pynapple.normalize_region_name_for_filename("M2")
    except ValueError as error:
        assert "Unsupported region" in str(error)
    else:
        raise AssertionError("Expected unsupported region to raise ValueError.")


def test_make_region_analysis_filename_prefixes_region():
    """Per-session analysis filenames should include the region tag."""
    filename = spike_behavior_pynapple.make_region_analysis_filename(
        region_name="hpc",
        base_filename="state_decodability_analysis.csv",
    )

    assert filename == "HPC_state_decodability_analysis.csv"


def test_normalize_region_channels_accepts_list_and_array_inputs():
    """Region channels should normalize to a one-dimensional integer NumPy array."""
    from_list = spike_behavior_pynapple.normalize_region_channels([200, 192, 210])
    from_array = spike_behavior_pynapple.normalize_region_channels(np.arange(192, 195, dtype=int))

    np.testing.assert_array_equal(from_list, np.array([200, 192, 210], dtype=int))
    np.testing.assert_array_equal(from_array, np.array([192, 193, 194], dtype=int))


def test_normalize_region_channels_rejects_empty_and_non_1d_inputs():
    """Region channel input should fail fast when shape is invalid."""
    try:
        spike_behavior_pynapple.normalize_region_channels([])
    except ValueError as error:
        assert "must contain at least one channel" in str(error)
    else:
        raise AssertionError("Expected empty region channels to raise ValueError.")

    try:
        spike_behavior_pynapple.normalize_region_channels(np.array([[192, 193], [194, 195]], dtype=int))
    except ValueError as error:
        assert "must be one-dimensional" in str(error)
    else:
        raise AssertionError("Expected non-1D region channels to raise ValueError.")


def test_build_spike_tsgroup_groups_spikes_by_cluster_id():
    """Spike groups should map each cluster id to a sorted Pynapple Ts."""
    spike_times = np.array([0.4, 0.1, 0.7, 0.3, 0.2], dtype=float)
    spike_clusters = np.array([2, 1, 2, 1, 1], dtype=int)

    spike_group = spike_behavior_pynapple.build_spike_tsgroup(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=np.array([1, 2], dtype=int),
    )

    assert isinstance(spike_group, nap.TsGroup)
    assert list(spike_group.keys()) == [1, 2]
    np.testing.assert_allclose(spike_group[1].index.to_numpy(), np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(spike_group[2].index.to_numpy(), np.array([0.4, 0.7]))


def test_plot_trial_raster_rejects_invalid_trial_index():
    """Trial raster plotting should fail cleanly for out-of-range trial indices."""
    trial_df = pd.DataFrame({"choice_time": [1.0], "start_time": [0.5]})
    region_spike_group = nap.TsGroup({11: nap.Ts(t=np.array([0.9, 1.1], dtype=float))})

    try:
        spike_behavior_pynapple.plot_trial_raster(
            region_spike_group=region_spike_group,
            trial_df=trial_df,
            trial_ix=2,
            show=False,
        )
    except ValueError as error:
        assert "trial_ix" in str(error)
    else:
        raise AssertionError("Expected invalid trial index to raise ValueError.")


def test_plot_trial_raster_requires_session_start_time_for_session_mode():
    """Session-time plots should require an explicit session start timestamp."""
    trial_df = pd.DataFrame({"choice_time": [1.0], "start_time": [0.5]})
    region_spike_group = nap.TsGroup({11: nap.Ts(t=np.array([0.9, 1.1], dtype=float))})

    try:
        spike_behavior_pynapple.plot_trial_raster(
            region_spike_group=region_spike_group,
            trial_df=trial_df,
            trial_ix=0,
            time_mode="session",
            show=False,
        )
    except ValueError as error:
        assert "session_start_time" in str(error)
    else:
        raise AssertionError("Expected missing session_start_time to raise ValueError.")


def test_plot_trial_raster_returns_two_axes_and_titles_trial_index():
    """Trial raster plotting should return spike and lick axes with a trial-index title."""
    trial_df = pd.DataFrame(
        {
            "choice_time": [10.0],
            "start_time": [9.5],
            "reward_time": [10.1],
        }
    )
    region_spike_group = nap.TsGroup(
        {
            11: nap.Ts(t=np.array([9.8, 10.0, 10.2], dtype=float)),
            12: nap.Ts(t=np.array([9.9, 10.3], dtype=float)),
        }
    )
    lick_times = {
        "right_entry": nap.Ts(t=np.array([9.85, 10.15], dtype=float)),
        "left_entry": nap.Ts(t=np.array([9.95], dtype=float)),
    }

    figure, axes = spike_behavior_pynapple.plot_trial_raster(
        region_spike_group=region_spike_group,
        trial_df=trial_df,
        trial_ix=0,
        lick_times=lick_times,
        event="choice_time",
        time_mode="event",
        pre_time=0.5,
        post_time=0.5,
        show=False,
    )

    assert len(axes) == 2
    assert "trial 0" in axes[0].get_title().lower()
    spike_reference_lines = [line for line in axes[0].lines if np.allclose(line.get_xdata(), [0.0, 0.0])]
    lick_reference_lines = [line for line in axes[1].lines if np.allclose(line.get_xdata(), [0.0, 0.0])]
    assert spike_reference_lines
    assert lick_reference_lines
    plt.close(figure)


def test_load_aligned_spikes_returns_required_aligned_time_array(tmp_path: Path):
    """Aligned spike loading should return a one-dimensional float time array."""
    aligned_path = tmp_path / "imec1_sync.npz"
    np.savez(aligned_path, spike_utc_unix=np.array([0.1, 0.2, 0.4], dtype=float))

    aligned_spike_times = spike_behavior_pynapple.load_aligned_spikes(aligned_path)

    np.testing.assert_allclose(aligned_spike_times, np.array([0.1, 0.2, 0.4], dtype=float))
    assert aligned_spike_times.ndim == 1


def test_load_aligned_spikes_fails_when_required_time_key_is_missing(tmp_path: Path):
    """Aligned spike loading should fail loudly if the aligned-time array is unavailable."""
    aligned_path = tmp_path / "imec1_sync.npz"
    np.savez(aligned_path, other_key=np.array([1.0, 2.0], dtype=float))

    try:
        spike_behavior_pynapple.load_aligned_spikes(aligned_path)
    except ValueError as error:
        assert "spike_utc_unix" in str(error)
    else:
        raise AssertionError("Expected missing aligned time key to raise ValueError.")


def test_validate_aligned_spike_inputs_rejects_shape_mismatch():
    """Aligned spike times must correspond one-to-one with sorter cluster ids."""
    try:
        spike_behavior_pynapple.validate_aligned_spike_inputs(
            aligned_spike_times=np.array([0.1, 0.2], dtype=float),
            spike_clusters=np.array([10, 11, 12], dtype=int),
        )
    except ValueError as error:
        assert "same length" in str(error)
    else:
        raise AssertionError("Expected mismatched aligned spikes and clusters to raise ValueError.")


def test_select_units_by_channels_returns_only_clusters_on_requested_channels():
    """Region selection should return cluster ids whose main channels match the user input."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12, 13],
            "ch": [191, 192, 200, 240],
            "group": ["good", "good", "mua", "noise"],
        }
    )

    selected_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info,
        region_channels=np.array([192, 200], dtype=int),
    )

    np.testing.assert_array_equal(selected_cluster_ids, np.array([11, 12], dtype=int))


def test_select_units_by_channels_accepts_arange_style_input():
    """Region selection should work the same for contiguous channel arrays."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12, 13],
            "ch": [191, 192, 193, 240],
            "group": ["good", "good", "mua", "noise"],
        }
    )

    selected_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info,
        region_channels=np.arange(192, 194, dtype=int),
    )

    np.testing.assert_array_equal(selected_cluster_ids, np.array([11, 12], dtype=int))


def test_select_units_by_channels_returns_empty_array_when_no_channels_match():
    """Region selection should return an empty integer array when nothing matches."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11],
            "ch": [5, 6],
            "group": ["good", "mua"],
        }
    )

    selected_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info,
        region_channels=np.array([192, 193], dtype=int),
    )

    np.testing.assert_array_equal(selected_cluster_ids, np.array([], dtype=int))


def test_select_units_by_channels_excludes_non_good_non_mua_groups():
    """Channel-matching clusters should still be excluded unless group is good or mua."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12, 13],
            "ch": [192, 192, 192, 192],
            "group": ["good", "mua", "noise", "unsorted"],
        }
    )

    selected_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info,
        region_channels=np.array([192], dtype=int),
    )

    np.testing.assert_array_equal(selected_cluster_ids, np.array([10, 11], dtype=int))


def test_select_units_by_channels_normalizes_group_labels_before_filtering():
    """Group filtering should tolerate simple case and whitespace differences."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12],
            "ch": [192, 192, 192],
            "group": [" Good ", "MUA", "noise"],
        }
    )

    selected_cluster_ids = spike_behavior_pynapple.select_units_by_channels(
        cluster_info,
        region_channels=np.array([192], dtype=int),
    )

    np.testing.assert_array_equal(selected_cluster_ids, np.array([10, 11], dtype=int))


def test_select_units_by_channels_requires_group_column():
    """Unit selection should fail loudly when cluster quality labels are unavailable."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11],
            "ch": [192, 193],
        }
    )

    try:
        spike_behavior_pynapple.select_units_by_channels(
            cluster_info,
            region_channels=np.array([192, 193], dtype=int),
        )
    except ValueError as error:
        assert "group" in str(error)
    else:
        raise AssertionError("Expected missing group column to raise ValueError.")


def test_resolve_trial_end_prefers_reward_then_choice_then_start():
    """Trial end should match the legacy event priority order."""
    reward_trial = pd.Series({"start_time": 1.0, "choice_time": 2.0, "reward_time": 3.0})
    choice_trial = pd.Series({"start_time": 1.0, "choice_time": 2.0, "reward_time": np.nan})
    start_only_trial = pd.Series({"start_time": 1.0, "choice_time": np.nan, "reward_time": np.nan})

    assert spike_behavior_pynapple.resolve_trial_end(reward_trial) == 3.0
    assert spike_behavior_pynapple.resolve_trial_end(choice_trial) == 2.0
    assert spike_behavior_pynapple.resolve_trial_end(start_only_trial) == 1.0


def test_bin_spikes_to_trial_pynapple_matches_manual_histogram():
    """Per-trial spike counts should match a direct NumPy histogram baseline."""
    spike_group = nap.TsGroup(
        {
            11: nap.Ts(t=np.array([0.2, 0.9, 1.3, 1.8], dtype=float)),
            12: nap.Ts(t=np.array([0.6, 1.1, 1.4], dtype=float)),
        }
    )

    binned_spikes, bin_edges = spike_behavior_pynapple.bin_spikes_to_trial_pynapple(
        trial_start=1.0,
        trial_end=1.5,
        spike_group=spike_group,
        cluster_ids=np.array([11, 12], dtype=int),
        bin_size=0.5,
        pre_time=0.5,
        post_time=0.5,
    )

    expected_edges = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)
    expected_counts = np.array([[1, 1, 1], [1, 2, 0]], dtype=float)

    np.testing.assert_allclose(bin_edges, expected_edges)
    np.testing.assert_allclose(binned_spikes, expected_counts)


def test_bin_licks_to_trial_pynapple_matches_manual_histogram():
    """Right and left lick bins should follow the legacy column ordering."""
    lick_times = {
        "right_entry": nap.Ts(t=np.array([0.6, 1.4], dtype=float)),
        "left_entry": nap.Ts(t=np.array([0.8, 1.1, 1.7], dtype=float)),
    }

    binned_licks, bin_edges = spike_behavior_pynapple.bin_licks_to_trial_pynapple(
        trial_start=1.0,
        trial_end=1.5,
        lick_times=lick_times,
        bin_size=0.5,
        pre_time=0.5,
        post_time=0.5,
    )

    expected_edges = np.array([0.5, 1.0, 1.5, 2.0], dtype=float)
    expected_counts = np.array([[1, 1], [1, 1], [0, 1]], dtype=float)

    np.testing.assert_allclose(bin_edges, expected_edges)
    np.testing.assert_allclose(binned_licks, expected_counts)


def test_bin_region_trials_returns_legacy_style_trial_dicts():
    """Trial binning should preserve the legacy list-of-dicts structure."""
    trial_df = pd.DataFrame(
        {
            "start_time": [1.0, 3.0],
            "choice_time": [1.5, 3.5],
            "reward_time": [2.0, np.nan],
            "state_int": [0, 1],
            "action": [1, 0],
        }
    )
    spike_group = nap.TsGroup(
        {
            11: nap.Ts(t=np.array([0.9, 1.1, 1.6, 3.2], dtype=float)),
            12: nap.Ts(t=np.array([1.4, 1.8, 3.1, 3.7], dtype=float)),
        }
    )

    trial_bins = spike_behavior_pynapple.bin_region_trials(
        trial_df=trial_df,
        spike_group=spike_group,
        cluster_ids=np.array([11, 12], dtype=int),
        bin_size=0.5,
        pre_time=0.5,
        post_time=0.5,
    )

    assert len(trial_bins) == 2
    assert trial_bins[0]["trial_ix"] == 0
    assert trial_bins[0]["binned_spikes"].shape == (2, 4)
    np.testing.assert_allclose(trial_bins[0]["bin_states"], np.zeros(4, dtype=float))
    np.testing.assert_allclose(trial_bins[0]["bin_choices"], np.ones(4, dtype=float))
    assert trial_bins[1]["trial_ix"] == 1
    assert trial_bins[1]["binned_spikes"].shape == (2, 3)
    np.testing.assert_allclose(trial_bins[1]["bin_states"], np.ones(3, dtype=float))
    np.testing.assert_allclose(trial_bins[1]["bin_choices"], np.zeros(3, dtype=float))


def test_bin_region_trials_matches_manual_histogram_baseline():
    """The Pynapple trial bins should match a direct histogram baseline."""
    trial_df = pd.DataFrame(
        {
            "start_time": [1.0],
            "choice_time": [1.5],
            "reward_time": [2.0],
            "state_int": [1],
            "action": [0],
        }
    )
    cluster_ids = np.array([11, 12], dtype=int)
    spike_times = np.array([0.9, 1.2, 1.7, 2.1, 1.1, 1.6], dtype=float)
    spike_clusters = np.array([11, 11, 11, 11, 12, 12], dtype=int)
    spike_group = spike_behavior_pynapple.build_spike_tsgroup(
        spike_times=spike_times,
        spike_clusters=spike_clusters,
        cluster_ids=cluster_ids,
    )

    trial_bins = spike_behavior_pynapple.bin_region_trials(
        trial_df=trial_df,
        spike_group=spike_group,
        cluster_ids=cluster_ids,
        bin_size=0.5,
        pre_time=0.5,
        post_time=0.5,
    )

    expected_edges = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)
    expected_counts = np.array(
        [
            np.histogram(spike_times[spike_clusters == 11], bins=expected_edges)[0],
            np.histogram(spike_times[spike_clusters == 12], bins=expected_edges)[0],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(trial_bins[0]["bin_edges"], expected_edges)
    np.testing.assert_allclose(trial_bins[0]["binned_spikes"], expected_counts)


def test_make_trial_type_masks_excludes_experimenter_reward_trials_from_all_conditions():
    """All neural-analysis masks should exclude experimenter-given reward trials."""
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": [0, 1, 0, 0, 0, 0, 0],
            "correct": [1, 1, 1, 0, 0, 1, 0],
            "reward": [1, 1, 0, 0, 0, 0, 0],
            "action": [0, 1, 1, 0, 1, 1, 1],
        }
    )

    masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)

    for mask_name, mask in masks.items():
        if mask_name == "valid":
            continue
        assert not bool(mask.iloc[1]), f"Mask {mask_name} incorrectly included manual reward trial."


def test_make_trial_type_masks_normalizes_legacy_give_reward_column():
    """Old trial CSVs should remain usable for neural trial-type masks."""
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 1, 0],
            "correct": [1, 1, 0],
            "reward": [1, 1, 0],
            "action": [0, 1, 1],
        }
    )

    masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)

    np.testing.assert_array_equal(masks["valid"].to_numpy(), np.array([True, False, True]))


def test_make_trial_type_masks_matches_base_condition_definitions():
    """Base condition masks should match the agreed simple trial definitions."""
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 1, 0, 0, 0, 0, 0],
            "correct": [1, 1, 1, 0, 0, 1, 0],
            "reward": [1, 1, 0, 0, 0, 0, 0],
            "action": [0, 1, 1, 0, 1, 1, 1],
        }
    )

    masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)

    np.testing.assert_array_equal(masks["valid"].to_numpy(), np.array([True, False, True, True, True, True, True]))
    np.testing.assert_array_equal(
        masks["correct_rewarded"].to_numpy(),
        np.array([True, False, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks["incorrect"].to_numpy(),
        np.array([False, False, False, True, True, False, True]),
    )
    np.testing.assert_array_equal(
        masks["omission"].to_numpy(),
        np.array([False, False, True, False, False, True, False]),
    )


def test_make_trial_type_masks_switch_and_stay_require_valid_next_trial_and_actions():
    """Switch/stay should exclude last trials, invalid next trials, and missing next actions."""
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 1, 0, 0, 0],
            "correct": [1, 1, 1, 0, 0, 1, 0],
            "reward": [0, 0, 0, 0, 0, 0, 0],
            "action": [0, 0, 1, 1, 1, np.nan, 0],
        }
    )

    masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)

    np.testing.assert_array_equal(
        masks["switch"].to_numpy(),
        np.array([False, True, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks["stay"].to_numpy(),
        np.array([True, False, False, False, False, False, False]),
    )


def test_make_trial_type_masks_combined_conditions_are_strict_intersections():
    """Combined masks should be exact AND combinations of the base masks."""
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0, 0],
            "correct": [1, 1, 0, 0, 1],
            "reward": [0, 0, 0, 0, 1],
            "action": [0, 1, 1, 0, 1],
        }
    )

    masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)

    np.testing.assert_array_equal(
        masks["omission_switch"].to_numpy(),
        (masks["omission"] & masks["switch"]).to_numpy(),
    )
    np.testing.assert_array_equal(
        masks["omission_stay"].to_numpy(),
        (masks["omission"] & masks["stay"]).to_numpy(),
    )
    np.testing.assert_array_equal(
        masks["incorrect_switch"].to_numpy(),
        (masks["incorrect"] & masks["switch"]).to_numpy(),
    )
    np.testing.assert_array_equal(
        masks["incorrect_stay"].to_numpy(),
        (masks["incorrect"] & masks["stay"]).to_numpy(),
    )


def test_make_classifier_bins_extracts_choice_and_start_aligned_windows():
    """Classifier-bin extraction should match the legacy edge-based window selection."""
    trial_df = pd.DataFrame(
        {
            "start_time": [1.0, 3.0],
            "choice_time": [1.5, 3.5],
            "reward_time": [2.0, 4.0],
            "state_int": [0, 1],
            "action": [1, 0],
        }
    )
    spike_group = nap.TsGroup(
        {
            11: nap.Ts(t=np.array([0.9, 1.2, 1.6, 3.1, 3.4, 3.8], dtype=float)),
            12: nap.Ts(t=np.array([1.1, 1.4, 1.9, 3.2, 3.6, 3.9], dtype=float)),
        }
    )
    trial_bins = spike_behavior_pynapple.bin_region_trials(
        trial_df=trial_df,
        spike_group=spike_group,
        cluster_ids=np.array([11, 12], dtype=int),
        bin_size=0.5,
        pre_time=0.5,
        post_time=0.5,
    )
    trial_mask = pd.Series([True, True])

    choice_spikes, choice_states, choice_actions = spike_behavior_pynapple.make_classifier_bins(
        region_trial_binned=trial_bins,
        trial_df=trial_df,
        trial_mask=trial_mask,
        event="choice_time",
        bounds=(-0.5, 0.0),
    )
    start_spikes, start_states, start_actions = spike_behavior_pynapple.make_classifier_bins(
        region_trial_binned=trial_bins,
        trial_df=trial_df,
        trial_mask=trial_mask,
        event="start_time",
        bounds=(0.0, 0.5),
    )

    np.testing.assert_allclose(choice_spikes, np.array([[1.0, 2.0], [2.0, 1.0]]))
    np.testing.assert_allclose(choice_states, np.array([0.0, 1.0]))
    np.testing.assert_allclose(choice_actions, np.array([1.0, 0.0]))
    np.testing.assert_allclose(start_spikes, np.array([[1.0, 2.0], [2.0, 1.0]]))
    np.testing.assert_allclose(start_states, np.array([0.0, 1.0]))
    np.testing.assert_allclose(start_actions, np.array([1.0, 0.0]))


def test_make_classifier_bins_raises_for_empty_trial_selection():
    """Empty selections should fail explicitly instead of returning misleading arrays."""
    trial_df = pd.DataFrame(
        {
            "start_time": [1.0],
            "choice_time": [1.5],
            "reward_time": [2.0],
            "state_int": [0],
            "action": [1],
        }
    )
    spike_group = nap.TsGroup({11: nap.Ts(t=np.array([1.1, 1.6], dtype=float))})
    trial_bins = spike_behavior_pynapple.bin_region_trials(
        trial_df=trial_df,
        spike_group=spike_group,
        cluster_ids=np.array([11], dtype=int),
        bin_size=0.5,
        pre_time=0.5,
        post_time=0.5,
    )

    try:
        spike_behavior_pynapple.make_classifier_bins(
            region_trial_binned=trial_bins,
            trial_df=trial_df,
            trial_mask=pd.Series([False]),
            event="choice_time",
            bounds=(-0.5, 0.0),
        )
    except ValueError as error:
        assert "No trials were selected" in str(error)
    else:
        raise AssertionError("Expected empty trial selection to raise ValueError.")


def test_summarize_trial_masks_returns_ordered_counts_and_indices():
    """Trial-mask summaries should preserve a user-provided condition order."""
    trial_masks = {
        "correct_rewarded": pd.Series([True, False, True, False]),
        "incorrect": pd.Series([False, True, False, True]),
        "omission": pd.Series([False, False, False, False]),
    }

    summary_df, condition_trial_indices = spike_behavior_pynapple.summarize_trial_masks(
        trial_masks=trial_masks,
        condition_names=["incorrect", "correct_rewarded", "omission"],
    )

    assert summary_df["condition"].tolist() == ["incorrect", "correct_rewarded", "omission"]
    assert summary_df["n_trials"].tolist() == [2, 2, 0]
    assert condition_trial_indices["incorrect"].tolist() == [1, 3]
    assert condition_trial_indices["correct_rewarded"].tolist() == [0, 2]
    assert condition_trial_indices["omission"].tolist() == []


def test_collect_condition_classifier_bins_returns_structured_results_for_requested_conditions():
    """Classifier-bin collection should preserve condition/window order and report failures explicitly."""
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0],
            "correct": [1, 1, 0, 1],
            "reward": [1, 0, 0, 1],
            "action": [0, 1, 0, 1],
            "state_int": [0, 1, 0, 1],
            "start_time": [0.0, 2.0, 4.0, 6.0],
            "choice_time": [0.5, 2.5, 4.5, 6.5],
            "reward_time": [1.0, 3.0, 5.0, 7.0],
        }
    )
    trial_masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)
    region_trial_binned = [
        {
            "trial_ix": trial_index,
            "binned_spikes": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "bin_edges": np.array([float(trial_df.iloc[trial_index]["choice_time"]) - 0.5, float(trial_df.iloc[trial_index]["choice_time"]), float(trial_df.iloc[trial_index]["choice_time"]) + 0.5]),
            "bin_states": np.array([float(trial_df.iloc[trial_index]["state_int"])] * 2),
            "bin_choices": np.array([float(trial_df.iloc[trial_index]["action"])] * 2),
        }
        for trial_index in range(len(trial_df))
    ]

    collected = spike_behavior_pynapple.collect_condition_classifier_bins(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        trial_masks=trial_masks,
        condition_names=["correct_rewarded", "omission", "stay"],
        windows={"pre_choice": (-0.5, 0.0), "post_choice": (0.0, 0.5)},
        event="choice_time",
    )

    assert list(collected.keys()) == [
        ("correct_rewarded", "pre_choice"),
        ("correct_rewarded", "post_choice"),
        ("omission", "pre_choice"),
        ("omission", "post_choice"),
        ("stay", "pre_choice"),
        ("stay", "post_choice"),
    ]
    assert collected[("correct_rewarded", "pre_choice")]["status"] == "ok"
    assert collected[("correct_rewarded", "pre_choice")]["spike_bins"].shape == (2, 2)
    assert collected[("omission", "post_choice")]["status"] == "ok"
    assert collected[("stay", "pre_choice")]["status"] == "failed"


def test_summarize_decoding_results_returns_compact_rows_for_display():
    """Decoding-result summaries should keep the key display columns in a stable order."""
    results_df = pd.DataFrame(
        [
            {
                "condition": "correct_rewarded",
                "window": "pre_choice",
                "status": "ok",
                "cv_score": 0.75,
                "cv_pvalue": 0.05,
                "test_accuracy": 0.8,
                "reason": "",
            },
            {
                "condition": "switch",
                "window": "post_choice",
                "status": "failed",
                "cv_score": np.nan,
                "cv_pvalue": np.nan,
                "test_accuracy": np.nan,
                "reason": "insufficient_classes",
            },
        ]
    )

    summary_df = spike_behavior_pynapple.summarize_decoding_results(
        results_df,
        value_columns=["cv_score", "cv_pvalue", "test_accuracy"],
    )

    assert summary_df.columns.tolist() == [
        "condition",
        "window",
        "status",
        "reason",
        "cv_score",
        "cv_pvalue",
        "test_accuracy",
    ]
    assert summary_df.iloc[0]["condition"] == "correct_rewarded"
    assert summary_df.iloc[1]["reason"] == "insufficient_classes"


def test_get_decode_target_returns_requested_column_and_rejects_unknown_target():
    """Decode targets should be limited to supported trial_df columns."""
    trial_df = pd.DataFrame({"state_int": [0, 1], "action": [1, 0]})

    np.testing.assert_allclose(
        spike_behavior_pynapple.get_decode_target(trial_df, target="state_int"),
        np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(
        spike_behavior_pynapple.get_decode_target(trial_df, target="action"),
        np.array([1.0, 0.0]),
    )

    try:
        spike_behavior_pynapple.get_decode_target(trial_df, target="reward")
    except ValueError as error:
        assert "target must be" in str(error)
    else:
        raise AssertionError("Expected unsupported decode target to raise ValueError.")


def test_cv_decodeability_score_returns_metrics_for_separable_data():
    """Cross-validated decodeability should succeed on a simple separable dataset."""
    binned_spikes = np.array(
        [
            [0.0, 0.0, 0.0, 3.0, 3.0, 3.0],
            [0.0, 1.0, 0.0, 3.0, 4.0, 3.0],
        ]
    )
    target_values = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    result = spike_behavior_pynapple.cv_decodeability_score(
        binned_spikes=binned_spikes,
        target_values=target_values,
        cv=2,
        n_permutations=4,
        random_state=42,
        label="sample",
    )

    assert result["status"] == "ok"
    assert result["label"] == "sample"
    assert result["n_samples"] == 6
    assert result["n_classes"] == 2
    assert 0.0 <= result["cv_score"] <= 1.0
    assert 0.0 <= result["cv_pvalue"] <= 1.0
    assert len(result["permutation_scores"]) == 4


def test_cv_decodeability_score_returns_failure_for_one_class_data():
    """Cross-validated decodeability should fail cleanly when only one class is present."""
    binned_spikes = np.array([[0.0, 1.0, 2.0]])
    target_values = np.array([1.0, 1.0, 1.0])

    result = spike_behavior_pynapple.cv_decodeability_score(
        binned_spikes=binned_spikes,
        target_values=target_values,
        cv=2,
        n_permutations=4,
        random_state=42,
    )

    assert result["status"] == "failed"
    assert result["reason"] == "insufficient_classes"


def test_train_single_decoder_with_shuffle_null_returns_classifier_and_metrics():
    """Single-decoder training should return a fitted classifier and held-out metrics."""
    binned_spikes = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0],
            [0.0, 1.0, 0.0, 1.0, 3.0, 4.0, 3.0, 4.0],
        ]
    )
    target_values = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

    classifier, result = spike_behavior_pynapple.train_single_decoder_with_shuffle_null(
        binned_spikes=binned_spikes,
        target_values=target_values,
        test_size=0.25,
        n_shuffles=8,
        random_state=42,
        label="train",
    )

    assert classifier is not None
    assert result["status"] == "ok"
    assert result["label"] == "train"
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["test_accuracy"] <= 1.0
    assert 0.0 <= result["shuffle_pvalue"] <= 1.0


def test_evaluate_decoder_on_condition_returns_accuracy_for_valid_input():
    """Evaluation on a held condition should return an accuracy score and counts."""
    training_spikes = np.array(
        [
            [0.0, 0.0, 1.0, 1.0, 3.0, 3.0, 4.0, 4.0],
            [0.0, 1.0, 0.0, 1.0, 3.0, 4.0, 3.0, 4.0],
        ]
    )
    training_targets = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    classifier, _ = spike_behavior_pynapple.train_single_decoder_with_shuffle_null(
        binned_spikes=training_spikes,
        target_values=training_targets,
        test_size=0.25,
        n_shuffles=4,
        random_state=42,
    )

    evaluation_result = spike_behavior_pynapple.evaluate_decoder_on_condition(
        classifier=classifier,
        binned_spikes=np.array([[0.0, 4.0], [1.0, 3.0]]),
        target_values=np.array([0.0, 1.0]),
        label="eval",
    )

    assert evaluation_result["status"] == "ok"
    assert evaluation_result["label"] == "eval"
    assert evaluation_result["n_samples"] == 2
    assert 0.0 <= evaluation_result["test_accuracy"] <= 1.0


def test_run_base_condition_decoding_returns_structured_results_and_failure_reasons():
    """Base-condition decoding should return training, CV, and evaluation records without crashing on sparse conditions."""
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0, 0, 0],
            "correct": [1, 1, 1, 1, 1, 0],
            "reward": [1, 1, 1, 1, 0, 0],
            "action": [0, 0, 1, 1, 0, 1],
            "state_int": [0, 0, 1, 1, 0, 1],
            "choice_time": [0.5] * 6,
            "start_time": [0.0] * 6,
        }
    )
    region_trial_binned = [
        {
            "trial_ix": trial_index,
            "binned_spikes": np.array(
                [
                    [float(trial_df.iloc[trial_index]["state_int"]), float(trial_df.iloc[trial_index]["state_int"]) + 1.0],
                    [float(trial_df.iloc[trial_index]["action"]), float(trial_df.iloc[trial_index]["action"]) + 1.0],
                ]
            ),
            "bin_edges": np.array([0.0, 0.5, 1.0]),
            "bin_states": np.array([float(trial_df.iloc[trial_index]["state_int"])] * 2),
            "bin_choices": np.array([float(trial_df.iloc[trial_index]["action"])] * 2),
        }
        for trial_index in range(len(trial_df))
    ]

    result = spike_behavior_pynapple.run_base_condition_decoding(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        target="state_int",
        cv=2,
        n_permutations=4,
        n_shuffles=4,
        random_state=42,
    )

    assert result["training_result"]["status"] == "ok"
    assert result["training_result"]["condition"] == "correct_rewarded"
    assert result["training_result"]["window"] == "post_choice"
    assert isinstance(result["decodeability_results"], pd.DataFrame)
    assert isinstance(result["generalization_results"], pd.DataFrame)
    assert result["decodeability_results"].shape[0] == 10
    assert result["generalization_results"].shape[0] == 10
    post_choice_row = result["generalization_results"].loc[
        (result["generalization_results"]["condition"] == "correct_rewarded")
        & (result["generalization_results"]["window"] == "post_choice")
    ].iloc[0]
    pre_choice_row = result["generalization_results"].loc[
        (result["generalization_results"]["condition"] == "correct_rewarded")
        & (result["generalization_results"]["window"] == "pre_choice")
    ].iloc[0]
    assert post_choice_row["label"] == result["training_result"]["label"]
    assert post_choice_row["test_accuracy"] == result["training_result"]["test_accuracy"]
    assert pre_choice_row["label"] == "correct_rewarded_pre_choice"
    assert set(result["decodeability_results"]["condition"]) == {
        "correct_rewarded", "incorrect", "omission", "switch", "stay"
    }
    assert "failed" in set(result["decodeability_results"]["status"])


def test_build_state_decodability_session_table_merges_pre_and_post_into_one_row_per_condition():
    decodeability_results = pd.DataFrame(
        [
            {
                "condition": "correct_rewarded",
                "window": "pre_choice",
                "status": "ok",
                "reason": "",
                "n_samples": 6,
                "n_classes": 2,
                "cv_score": 0.75,
                "cv_pvalue": 0.05,
                "permutation_score_mean": 0.52,
                "permutation_score_std": 0.08,
            },
            {
                "condition": "correct_rewarded",
                "window": "post_choice",
                "status": "failed",
                "reason": "insufficient_classes",
                "n_samples": 3,
                "n_classes": 1,
                "cv_score": np.nan,
                "cv_pvalue": np.nan,
                "permutation_score_mean": np.nan,
                "permutation_score_std": np.nan,
            },
            {
                "condition": "incorrect",
                "window": "pre_choice",
                "status": "ok",
                "reason": "",
                "n_samples": 4,
                "n_classes": 2,
                "cv_score": 0.6,
                "cv_pvalue": 0.2,
                "permutation_score_mean": 0.5,
                "permutation_score_std": 0.05,
            },
            {
                "condition": "incorrect",
                "window": "post_choice",
                "status": "ok",
                "reason": "",
                "n_samples": 4,
                "n_classes": 2,
                "cv_score": 0.7,
                "cv_pvalue": 0.1,
                "permutation_score_mean": 0.51,
                "permutation_score_std": 0.06,
            },
        ]
    )
    session = spike_behavior_pynapple.Session(
        sess_id_full="CT014_2025-12-23_163505",
        mouse="CT014",
        date="2025-12-23",
    )

    session_table = spike_behavior_pynapple.build_state_decodability_session_table(
        session=session,
        region_name="HPC",
        decodeability_results=decodeability_results,
    )

    assert session_table["condition"].tolist() == ["correct_rewarded", "incorrect"]
    assert session_table.iloc[0]["session_id"] == "CT014_2025-12-23_163505"
    assert session_table.iloc[0]["region"] == "HPC"
    assert session_table.iloc[0]["status_pre"] == "ok"
    assert session_table.iloc[0]["status_post"] == "failed"
    assert session_table.iloc[0]["cv_score_pre"] == 0.75
    assert np.isnan(session_table.iloc[0]["cv_score_post"])
    assert session_table.iloc[1]["cv_score_post"] == 0.7


def test_save_state_decodability_session_csv_writes_expected_filename_and_columns(tmp_path: Path):
    session_table = pd.DataFrame(
        [
            {
                "session_id": "CT014_2025-12-23_163505",
                "mouse": "CT014",
                "date": "2025-12-23",
                "region": "HPC",
                "condition": "correct_rewarded",
                "status_pre": "ok",
                "reason_pre": "",
                "n_trials_pre": 6,
                "n_classes_pre": 2,
                "cv_score_pre": 0.75,
                "p_value_pre": 0.05,
                "score_mean_pre": 0.52,
                "score_std_pre": 0.08,
                "status_post": "ok",
                "reason_post": "",
                "n_trials_post": 6,
                "n_classes_post": 2,
                "cv_score_post": 0.7,
                "p_value_post": 0.1,
                "score_mean_post": 0.51,
                "score_std_post": 0.06,
            }
        ]
    )

    saved_path = spike_behavior_pynapple.save_state_decodability_session_csv(
        output_dir=tmp_path,
        state_decodability_table=session_table,
        region_name="HPC",
    )

    assert saved_path == tmp_path / "HPC_state_decodability_analysis.csv"
    loaded_table = pd.read_csv(saved_path)
    assert loaded_table.columns.tolist() == session_table.columns.tolist()
    assert loaded_table.iloc[0]["condition"] == "correct_rewarded"


def test_run_repeated_correct_rewarded_decoder_returns_requested_runs_and_training_window():
    trial_df = pd.DataFrame(
        {
            "give_reward": [0, 0, 0, 0, 0, 0],
            "correct": [1, 1, 1, 1, 1, 0],
            "reward": [1, 1, 1, 1, 0, 0],
            "action": [0, 0, 1, 1, 0, 1],
            "state_int": [0, 0, 1, 1, 0, 1],
            "choice_time": [0.5] * 6,
            "start_time": [0.0] * 6,
        }
    )
    region_trial_binned = [
        {
            "trial_ix": trial_index,
            "binned_spikes": np.array(
                [
                    [float(trial_df.iloc[trial_index]["state_int"]), float(trial_df.iloc[trial_index]["state_int"]) + 1.0],
                    [float(trial_df.iloc[trial_index]["action"]), float(trial_df.iloc[trial_index]["action"]) + 1.0],
                ]
            ),
            "bin_edges": np.array([0.0, 0.5, 1.0]),
            "bin_states": np.array([float(trial_df.iloc[trial_index]["state_int"])] * 2),
            "bin_choices": np.array([float(trial_df.iloc[trial_index]["action"])] * 2),
        }
        for trial_index in range(len(trial_df))
    ]

    repeated_results = spike_behavior_pynapple.run_repeated_correct_rewarded_decoder(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        target="state_int",
        n_decoder_runs=3,
        n_shuffles=4,
        random_state=42,
    )

    assert repeated_results["training_results"].shape[0] == 3
    assert repeated_results["generalization_results"].shape[0] == 30
    assert set(repeated_results["training_results"]["decoder_run"]) == {0, 1, 2}
    assert set(repeated_results["training_results"]["training_condition"]) == {"correct_rewarded"}
    assert set(repeated_results["training_results"]["training_window"]) == {"post_choice"}
    assert set(repeated_results["generalization_results"]["condition"]) == {
        "correct_rewarded", "incorrect", "omission", "switch", "stay"
    }


def test_build_correct_rewarded_decoding_performance_session_table_uses_one_row_per_decoder_run():
    session = spike_behavior_pynapple.Session(
        sess_id_full="CT014_2025-12-23_163505",
        mouse="CT014",
        date="2025-12-23",
    )
    training_results = pd.DataFrame(
        [
            {
                "decoder_run": 0,
                "training_condition": "correct_rewarded",
                "training_window": "post_choice",
                "status": "ok",
                "reason": "",
                "train_accuracy": 0.9,
                "test_accuracy": 0.8,
                "shuffle_pvalue": 0.05,
                "shuffle_accuracy_mean": 0.5,
                "shuffle_accuracy_std": 0.1,
            },
            {
                "decoder_run": 1,
                "training_condition": "correct_rewarded",
                "training_window": "post_choice",
                "status": "ok",
                "reason": "",
                "train_accuracy": 0.85,
                "test_accuracy": 0.75,
                "shuffle_pvalue": 0.08,
                "shuffle_accuracy_mean": 0.48,
                "shuffle_accuracy_std": 0.09,
            },
        ]
    )
    generalization_results = pd.DataFrame(
        [
            {
                "decoder_run": 0,
                "condition": "correct_rewarded",
                "window": "pre_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.7,
                "n_samples": 6,
                "n_classes": 2,
            },
            {
                "decoder_run": 0,
                "condition": "correct_rewarded",
                "window": "post_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.8,
                "n_samples": 6,
                "n_classes": 2,
            },
            {
                "decoder_run": 0,
                "condition": "incorrect",
                "window": "pre_choice",
                "status": "failed",
                "reason": "insufficient_classes",
                "test_accuracy": np.nan,
                "n_samples": 1,
                "n_classes": 1,
            },
            {
                "decoder_run": 0,
                "condition": "incorrect",
                "window": "post_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.6,
                "n_samples": 2,
                "n_classes": 2,
            },
            {
                "decoder_run": 1,
                "condition": "correct_rewarded",
                "window": "pre_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.65,
                "n_samples": 6,
                "n_classes": 2,
            },
            {
                "decoder_run": 1,
                "condition": "correct_rewarded",
                "window": "post_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.75,
                "n_samples": 6,
                "n_classes": 2,
            },
            {
                "decoder_run": 1,
                "condition": "incorrect",
                "window": "pre_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.55,
                "n_samples": 2,
                "n_classes": 2,
            },
            {
                "decoder_run": 1,
                "condition": "incorrect",
                "window": "post_choice",
                "status": "ok",
                "reason": "",
                "test_accuracy": 0.58,
                "n_samples": 2,
                "n_classes": 2,
            },
        ]
    )

    session_table = spike_behavior_pynapple.build_correct_rewarded_decoding_performance_session_table(
        session=session,
        region_name="HPC",
        training_results=training_results,
        generalization_results=generalization_results,
    )

    assert session_table.shape[0] == 2
    assert session_table["decoder_run"].tolist() == [0, 1]
    assert session_table.iloc[0]["train_accuracy"] == 0.9
    assert session_table.iloc[0]["correct_rewarded_test_accuracy_pre"] == 0.7
    assert session_table.iloc[0]["correct_rewarded_test_accuracy_post"] == 0.8
    assert session_table.iloc[0]["incorrect_status_pre"] == "failed"
    assert np.isnan(session_table.iloc[0]["incorrect_test_accuracy_pre"])
    assert session_table.iloc[1]["incorrect_test_accuracy_post"] == 0.58


def test_save_correct_rewarded_decoding_performance_session_csv_writes_expected_filename(tmp_path: Path):
    session_table = pd.DataFrame(
        [
            {
                "session_id": "CT014_2025-12-23_163505",
                "mouse": "CT014",
                "date": "2025-12-23",
                "region": "HPC",
                "decoder_run": 0,
                "training_condition": "correct_rewarded",
                "training_window": "post_choice",
                "train_accuracy": 0.9,
                "heldout_test_accuracy": 0.8,
                "shuffle_pvalue": 0.05,
                "shuffle_accuracy_mean": 0.5,
                "shuffle_accuracy_std": 0.1,
                "correct_rewarded_test_accuracy_pre": 0.7,
                "correct_rewarded_test_accuracy_post": 0.8,
            }
        ]
    )

    saved_path = spike_behavior_pynapple.save_correct_rewarded_decoding_performance_session_csv(
        output_dir=tmp_path,
        decoding_performance_table=session_table,
        region_name="PFC",
    )

    assert saved_path == tmp_path / "PFC_correct_rewarded_decoding_performance.csv"
    loaded_table = pd.read_csv(saved_path)
    assert loaded_table.columns.tolist() == session_table.columns.tolist()
    assert loaded_table.iloc[0]["decoder_run"] == 0
