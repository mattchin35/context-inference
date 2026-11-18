import numpy as np
import pandas as pd
import pynapple as nap
import matplotlib.pyplot as plt

import src.neural_analysis.psth_behavior as psth_behavior


def _lick_times(left_times=None, right_times=None):
    left_times = [] if left_times is None else left_times
    right_times = [] if right_times is None else right_times
    return {
        "left_entry": nap.Ts(t=np.asarray(left_times, dtype=float)),
        "right_entry": nap.Ts(t=np.asarray(right_times, dtype=float)),
    }


def _synthetic_lick_peth_data(n_trials=3):
    time_bin_edges = np.array([-1.0, 0.0, 1.0], dtype=float)
    time_bin_centers = np.array([-0.5, 0.5], dtype=float)
    return psth_behavior.LickPethData(
        alignment="trial_start",
        trial_indices=np.arange(n_trials, dtype=int),
        time_bin_edges=time_bin_edges,
        time_bin_centers=time_bin_centers,
        left_raster_times_by_trial=[np.array([-0.2, 0.2], dtype=float) for _ in range(n_trials)],
        right_raster_times_by_trial=[np.array([0.4], dtype=float) for _ in range(n_trials)],
        left_rate_by_trial=np.ones((n_trials, time_bin_centers.size), dtype=float),
        right_rate_by_trial=np.ones((n_trials, time_bin_centers.size), dtype=float),
        valid_rate_bins=np.ones((n_trials, time_bin_centers.size), dtype=bool),
        trial_start_offsets=np.zeros(n_trials, dtype=float),
        led_offsets=np.full(n_trials, 0.2, dtype=float),
        choice_offsets=np.full(n_trials, 1.0, dtype=float),
    )


def _synthetic_spike_peth_data(n_trials=3):
    time_bin_edges = np.array([-1.0, 0.0, 1.0], dtype=float)
    time_bin_centers = np.array([-0.5, 0.5], dtype=float)
    return psth_behavior.SpikePethData(
        unit_cluster_id=7,
        alignment="trial_start",
        trial_indices=np.arange(n_trials, dtype=int),
        time_bin_edges=time_bin_edges,
        time_bin_centers=time_bin_centers,
        spike_raster_times_by_trial=[np.array([-0.2, 0.2], dtype=float) for _ in range(n_trials)],
        spike_rate_by_trial=np.ones((n_trials, time_bin_centers.size), dtype=float),
        valid_rate_bins=np.ones((n_trials, time_bin_centers.size), dtype=bool),
        trial_start_offsets=np.zeros(n_trials, dtype=float),
        led_offsets=np.full(n_trials, 0.2, dtype=float),
        choice_offsets=np.full(n_trials, 1.0, dtype=float),
    )


def test_select_first_valid_unit_cluster_id_uses_good_or_mua_units():
    """Single-unit spike plots should use the first non-noise unit in metadata order."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [2, 4, 6],
            "group": ["noise", " Good ", "mua"],
        }
    )

    cluster_id = psth_behavior.select_first_valid_unit_cluster_id(cluster_info)

    assert cluster_id == 4


def test_build_single_unit_spike_peth_data_respects_trial_mask():
    """Single-unit spike PETHs should share lick PETH trial masking behavior."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0, 30.0, 40.0],
            "choice_time": [11.0, 21.0, 31.0, 41.0],
            "led_on_time": [10.2, 20.2, 30.2, 40.2],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    spike_peth = psth_behavior.build_single_unit_spike_peth_data(
        trial_df=trial_df,
        unit_spikes=nap.Ts(t=np.array([10.1, 20.1, 30.1, 40.1], dtype=float)),
        unit_cluster_id=7,
        alignment="trial_start",
        rate_bin_size=1.0,
        pre_time=1.0,
        post_time=1.0,
        trial_mask=np.array([True, False, True, False]),
    )

    np.testing.assert_array_equal(spike_peth.trial_indices, np.array([0, 2], dtype=int))


def test_build_single_unit_spike_peth_data_trial_start_caps_long_trials_without_dropping_them():
    """Trial-start spike plots should cap the visible positive x-window without dropping trials."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0],
            "choice_time": [11.0, 28.0],
            "led_on_time": [10.2, 20.2],
            "experimenter_reward_given": [0, 0],
        }
    )

    spike_peth = psth_behavior.build_single_unit_spike_peth_data(
        trial_df=trial_df,
        unit_spikes=nap.Ts(t=np.array([10.1, 20.5, 26.0], dtype=float)),
        unit_cluster_id=7,
        alignment="trial_start",
        rate_bin_size=1.0,
        pre_time=1.0,
        post_time=1.0,
        max_time_after_trial_start=5.0,
    )

    np.testing.assert_array_equal(spike_peth.trial_indices, np.array([0, 1], dtype=int))
    np.testing.assert_allclose(spike_peth.time_bin_edges, np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(spike_peth.choice_offsets, np.array([1.0, 8.0]))
    assert 6.0 not in spike_peth.spike_raster_times_by_trial[1]


def test_plot_single_unit_spike_peth_returns_two_eventplot_axes():
    """Single-unit spike rasters should use eventplot collections on one raster axis."""
    spike_peth = _synthetic_spike_peth_data(n_trials=3)

    fig, axes = psth_behavior.plot_single_unit_spike_peth(spike_peth, show_led_lines=True)
    raster_collections = axes[0].collections
    collection_type_names = {type(collection).__name__ for collection in raster_collections}

    assert len(np.ravel(axes)) == 2
    assert "EventCollection" in collection_type_names
    assert "PathCollection" not in collection_type_names
    plt.close(fig)


def test_select_valid_lick_peth_trials_excludes_experimenter_reward_and_missing_choice():
    """Valid lick-PETH trials should have animal choices and no experimenter reward."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0, 30.0, 40.0, 50.0],
            "choice_time": [11.0, "None", 31.0, np.nan, 52.0],
            "led_on_time": [10.2, 20.2, 30.2, 40.2, 50.2],
            "give_reward": [0, 0, 1, 0, 0],
        }
    )

    valid_mask = psth_behavior.select_valid_lick_peth_trials(trial_df)

    np.testing.assert_array_equal(valid_mask.to_numpy(), np.array([True, False, False, False, True]))


def test_make_lick_peth_trial_type_masks_separates_left_and_right_correct_rewarded():
    """Correct side masks should include rewarded choices and exclude omissions/invalid trials."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            "choice_time": [11.0, 21.0, 31.0, 41.0, 51.0, "None"],
            "led_on_time": [10.2, 20.2, 30.2, 40.2, 50.2, 60.2],
            "experimenter_reward_given": [0, 0, 0, 0, 1, 0],
            "correct": [1, 1, 1, 1, 1, 1],
            "reward": [1, 1, 0, 0, 1, 1],
            "action": [1, 0, 1, 0, 1, 0],
        }
    )

    masks = psth_behavior.make_lick_peth_trial_type_masks(trial_df)

    np.testing.assert_array_equal(masks["valid"].to_numpy(), np.array([True, True, True, True, False, False]))
    np.testing.assert_array_equal(masks["left_correct"].to_numpy(), np.array([True, False, False, False, False, False]))
    np.testing.assert_array_equal(masks["right_correct"].to_numpy(), np.array([False, True, False, False, False, False]))


def test_make_lick_peth_trial_type_masks_splits_unrewarded_conditions_by_side():
    """Unrewarded switch/stay masks should mark the current trial using the next trial's action."""
    trial_df = pd.DataFrame(
        {
            "start_time": np.arange(10.0, 19.0),
            "choice_time": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, "None"],
            "led_on_time": np.arange(10.2, 19.2),
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 0, 1, 0],
            "correct": [1, 1, 0, 0, 1, 1, 1, 0, 0],
            "reward": [1, 1, 0, 0, 0, 0, 1, 0, 0],
            "action": [1, 0, 1, 0, 0, 1, 1, 0, 1],
        }
    )

    masks = psth_behavior.make_lick_peth_trial_type_masks(trial_df)

    expected_false = np.zeros(trial_df.shape[0], dtype=bool)
    expected_left_incorrect = expected_false.copy()
    expected_left_incorrect[2] = True
    expected_right_incorrect = expected_false.copy()
    expected_right_incorrect[3] = True
    expected_right_omission = expected_false.copy()
    expected_right_omission[4] = True
    expected_left_omission = expected_false.copy()
    expected_left_omission[5] = True
    expected_left_switch = expected_false.copy()
    expected_left_switch[2] = True
    expected_right_stay = expected_false.copy()
    expected_right_stay[3] = True
    expected_right_switch = expected_false.copy()
    expected_right_switch[4] = True
    expected_left_stay = expected_false.copy()
    expected_left_stay[5] = True

    np.testing.assert_array_equal(masks["left_incorrect"].to_numpy(), expected_left_incorrect)
    np.testing.assert_array_equal(masks["right_incorrect"].to_numpy(), expected_right_incorrect)
    np.testing.assert_array_equal(masks["left_omission"].to_numpy(), expected_left_omission)
    np.testing.assert_array_equal(masks["right_omission"].to_numpy(), expected_right_omission)
    np.testing.assert_array_equal(masks["left_switch"].to_numpy(), expected_left_switch)
    np.testing.assert_array_equal(masks["right_switch"].to_numpy(), expected_right_switch)
    np.testing.assert_array_equal(masks["left_stay"].to_numpy(), expected_left_stay)
    np.testing.assert_array_equal(masks["right_stay"].to_numpy(), expected_right_stay)


def test_build_lick_peth_data_trial_start_uses_longest_choice_window_and_nan_masks_short_trials():
    """Trial-start lick means should ignore bins after each trial's choice-centered end."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0],
            "choice_time": [11.0, 22.0],
            "led_on_time": [10.2, 20.2],
            "experimenter_reward_given": [0, 0],
        }
    )
    lick_times = _lick_times(left_times=[9.5, 10.1, 11.8, 20.5, 22.5], right_times=[10.6, 21.2])

    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=trial_df,
        lick_times=lick_times,
        alignment="trial_start",
        rate_bin_size=1.0,
        pre_time=1.0,
        post_time=1.0,
    )

    np.testing.assert_allclose(lick_peth.time_bin_edges, np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(lick_peth.choice_offsets, np.array([1.0, 2.0]))
    np.testing.assert_array_equal(
        lick_peth.valid_rate_bins,
        np.array(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        ),
    )
    assert np.isnan(lick_peth.left_rate_by_trial[0, -1])
    assert not np.isnan(lick_peth.left_rate_by_trial[1, -1])


def test_build_lick_peth_data_trial_mask_limits_trials_chronologically():
    """Trial masks should select non-contiguous trials while preserving original order."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0, 30.0, 40.0],
            "choice_time": [11.0, 21.0, 31.0, 41.0],
            "led_on_time": [10.2, 20.2, 30.2, 40.2],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=trial_df,
        lick_times=_lick_times(left_times=[10.1, 30.1], right_times=[10.3, 30.3]),
        alignment="trial_start",
        rate_bin_size=1.0,
        pre_time=1.0,
        post_time=1.0,
        trial_mask=np.array([True, False, True, False]),
    )

    np.testing.assert_array_equal(lick_peth.trial_indices, np.array([0, 2], dtype=int))


def test_build_lick_peth_data_trial_mask_uses_selected_trials_for_trial_start_window():
    """Trial-start x-window should be computed from selected trials, not unselected long trials."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0, 30.0],
            "choice_time": [11.0, 28.0, 32.0],
            "led_on_time": [10.2, 20.2, 30.2],
            "experimenter_reward_given": [0, 0, 0],
        }
    )

    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=trial_df,
        lick_times=_lick_times(left_times=[10.1, 20.1, 30.1], right_times=[10.3, 20.3, 30.3]),
        alignment="trial_start",
        rate_bin_size=1.0,
        pre_time=1.0,
        post_time=1.0,
        trial_mask=np.array([True, False, True]),
    )

    np.testing.assert_array_equal(lick_peth.trial_indices, np.array([0, 2], dtype=int))
    np.testing.assert_allclose(lick_peth.time_bin_edges, np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))


def test_build_lick_peth_data_trial_start_caps_long_trials_without_dropping_them():
    """Trial-start plots should keep long trials but cap the visible positive x-window."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0, 20.0],
            "choice_time": [11.0, 28.0],
            "led_on_time": [10.2, 20.2],
            "experimenter_reward_given": [0, 0],
        }
    )
    lick_times = _lick_times(left_times=[10.1, 20.5, 26.0], right_times=[10.3, 24.0, 29.0])

    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=trial_df,
        lick_times=lick_times,
        alignment="trial_start",
        rate_bin_size=1.0,
        pre_time=1.0,
        post_time=1.0,
        max_time_after_trial_start=5.0,
    )

    np.testing.assert_array_equal(lick_peth.trial_indices, np.array([0, 1], dtype=int))
    np.testing.assert_allclose(lick_peth.time_bin_edges, np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
    np.testing.assert_allclose(lick_peth.choice_offsets, np.array([1.0, 8.0]))
    assert 6.0 not in lick_peth.left_raster_times_by_trial[1]
    assert 9.0 not in lick_peth.right_raster_times_by_trial[1]


def test_build_lick_peth_data_choice_alignment_uses_fixed_window_and_offsets():
    """Choice-aligned lick PETHs should keep fixed windows and trial event offsets."""
    trial_df = pd.DataFrame(
        {
            "start_time": [8.0, 17.0],
            "choice_time": [10.0, 20.0],
            "led_on_time": [8.5, 17.5],
            "experimenter_reward_given": [0, 0],
        }
    )
    lick_times = _lick_times(left_times=[9.5, 19.5], right_times=[10.2, 20.2])

    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=trial_df,
        lick_times=lick_times,
        alignment="choice",
        rate_bin_size=1.0,
        pre_time=2.0,
        post_time=1.0,
    )

    np.testing.assert_allclose(lick_peth.time_bin_edges, np.array([-2.0, -1.0, 0.0, 1.0]))
    np.testing.assert_allclose(lick_peth.trial_start_offsets, np.array([-2.0, -3.0]))
    np.testing.assert_allclose(lick_peth.led_offsets, np.array([-1.5, -2.5]))
    np.testing.assert_allclose(lick_peth.choice_offsets, np.array([0.0, 0.0]))
    assert np.all(lick_peth.valid_rate_bins)


def test_build_lick_peth_data_choice_alignment_ignores_trial_start_x_cap():
    """Choice alignment should keep its fixed window when the trial-start cap is provided."""
    trial_df = pd.DataFrame(
        {
            "start_time": [10.0],
            "choice_time": [20.0],
            "led_on_time": [10.2],
            "experimenter_reward_given": [0],
        }
    )

    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=trial_df,
        lick_times=_lick_times(left_times=[19.5], right_times=[20.2]),
        alignment="choice",
        rate_bin_size=1.0,
        pre_time=2.0,
        post_time=1.0,
        max_time_after_trial_start=5.0,
    )

    np.testing.assert_allclose(lick_peth.time_bin_edges, np.array([-2.0, -1.0, 0.0, 1.0]))


def test_plot_lick_peth_overlay_returns_two_axes():
    """Overlay lick plots should return one raster axis and one mean-rate axis."""
    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=pd.DataFrame(
            {
                "start_time": [10.0],
                "choice_time": [11.0],
                "led_on_time": [10.2],
                "experimenter_reward_given": [0],
            }
        ),
        lick_times=_lick_times(left_times=[10.1], right_times=[10.3]),
        alignment="trial_start",
        rate_bin_size=0.5,
        pre_time=1.0,
        post_time=1.0,
    )

    fig, axes = psth_behavior.plot_lick_peth(lick_peth, lick_layout="overlay", show_led_lines=True)

    assert len(np.ravel(axes)) == 2
    plt.close(fig)


def test_plot_lick_peth_separate_returns_four_axes():
    """Separate lick plots should return left and right raster/mean axes."""
    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=pd.DataFrame(
            {
                "start_time": [10.0],
                "choice_time": [11.0],
                "led_on_time": [10.2],
                "experimenter_reward_given": [0],
            }
        ),
        lick_times=_lick_times(left_times=[10.1], right_times=[10.3]),
        alignment="choice",
        rate_bin_size=0.5,
        pre_time=1.0,
        post_time=1.0,
    )

    fig, axes = psth_behavior.plot_lick_peth(lick_peth, lick_layout="separate", show_led_lines=True)

    assert len(np.ravel(axes)) == 4
    plt.close(fig)


def test_plot_lick_peth_can_hide_led_lines():
    """Disabling LED markers should remove LED-labeled event markers from the axes."""
    lick_peth = psth_behavior.build_lick_peth_data(
        trial_df=pd.DataFrame(
            {
                "start_time": [10.0],
                "choice_time": [11.0],
                "led_on_time": [10.2],
                "experimenter_reward_given": [0],
            }
        ),
        lick_times=_lick_times(left_times=[10.1], right_times=[10.3]),
        alignment="trial_start",
        rate_bin_size=0.5,
        pre_time=1.0,
        post_time=1.0,
    )

    fig, axes = psth_behavior.plot_lick_peth(lick_peth, lick_layout="overlay", show_led_lines=False)
    labels = [artist.get_label() for axis in np.ravel(axes) for artist in [*axis.lines, *axis.collections]]

    assert "LED" not in labels
    plt.close(fig)


def test_plot_lick_peth_uses_eventplot_raster_rows():
    """Lick rasters should use eventplot collections instead of scatter marker rows."""
    lick_peth = _synthetic_lick_peth_data(n_trials=3)

    fig, axes = psth_behavior.plot_lick_peth(lick_peth, lick_layout="overlay", show_led_lines=True)
    raster_collections = axes[0].collections
    collection_type_names = {type(collection).__name__ for collection in raster_collections}

    assert "EventCollection" in collection_type_names
    assert "PathCollection" not in collection_type_names
    plt.close(fig)


def test_plot_lick_peth_scales_figure_height_with_trial_count():
    """Many-trial rasters should grow vertically instead of compressing rows into a fixed height."""
    lick_peth = _synthetic_lick_peth_data(n_trials=100)

    fig, _ = psth_behavior.plot_lick_peth(
        lick_peth,
        lick_layout="overlay",
        raster_row_height_inches=0.1,
        min_fig_height=6.0,
    )

    assert fig.get_size_inches()[1] > 6.0
    plt.close(fig)


def test_plot_lick_peth_accepts_raster_style_settings():
    """Raster line length and width should be user-adjustable for dense trial plots."""
    lick_peth = _synthetic_lick_peth_data(n_trials=3)

    fig, axes = psth_behavior.plot_lick_peth(
        lick_peth,
        lick_layout="overlay",
        raster_line_length=0.4,
        raster_line_width=0.2,
    )

    assert len(np.ravel(axes)) == 2
    plt.close(fig)


def test_save_figure_with_message_saves_png_and_prints_path(tmp_path, capsys):
    """Figure saving should leave a simple console breadcrumb for interactive runs."""
    fig, _ = plt.subplots()
    save_path = tmp_path / "lick_peth.png"

    psth_behavior.save_figure_with_message(fig, save_path)

    assert save_path.exists()
    captured = capsys.readouterr()
    assert f"Saved figure {save_path.name} to {save_path}" in captured.out
    plt.close(fig)
