"""Time-weighted treadmill speed, block-purged CCA and fixed replay tests."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population_decoding.targets import build_population_targets
from src.neural_analysis.presentation_figures.movement import (
    bin_speed_magnitude,
    compute_movement,
)
from src.neural_analysis.presentation_figures.preparation import (
    PAIRED_WINDOW_S,
    prepare_choices,
)


def test_bin_speed_integrates_absolute_velocity_and_does_not_cross_gaps():
    """Use actual update durations, preserve lone held values, and reject partial bins."""
    times = np.array([10., 10.25, 11., 13., 16., 17.])
    velocity = np.array([-2., 6., -4., np.nan, 10., 1.])
    starts = np.array([[10., 11.25, 12.5, 9.5, 16.], [10.2, 15.5, 16.5, 17., 16.75]])
    stops = np.array([[11., 11.75, 13., 10.25, 16.5], [10.5, 16.5, 17., 17.25, 17.25]])
    result = bin_speed_magnitude(
        times, velocity, starts, stops, invalid_speed_intervals_s=np.array([[10.25, 10.5]]),
    )
    np.testing.assert_allclose(result, [
        [np.nan, 4., 4., np.nan, 10.],
        [np.nan, np.nan, 10., np.nan, np.nan],
    ])
    weighted = bin_speed_magnitude(times, velocity, starts, stops)
    assert weighted[0, 0] == pytest.approx(5.)
    assert weighted[1, 0] == pytest.approx((.05 * 2 + .25 * 6) / .3)


@pytest.fixture
def movement_inputs(synthetic_presentation_session):
    """Return prepared counts and (speed record,) UTC times/signed mm/s.

    Four choices each in three blocks have identical first canonical pairs.
    Speed increases by 2 mm/s across the four choices in every block. Speed
    varies opposite to PFC outside block 20 and in the same direction in it.
    """
    prepared = prepare_choices(synthetic_presentation_session)
    rows = np.array([0, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13])
    trials = prepared.trials.copy()
    trial_times = trials.choice_time.iloc[0] + 5 * np.arange(len(trials))
    trials["start_time"] = trial_times - 4.
    trials["choice_time"] = trial_times
    trials.loc[rows, ["action", "reward"]] = [1, 1]
    targets = build_population_targets(
        trials.state_int, trials.action, trials.reward, trials.experimenter_reward_given,
    )
    choices = prepared.choices.copy()
    choices["choice_time"] = trial_times[choices.row_position]
    events = pd.DataFrame({
        "Event": ["pump2_reward(1)", "pump3_reward(1)"],
        "Time": [trial_times[rows[0]] + .05, trial_times[rows[0]] + .025],
    })
    edges = np.linspace(-2., 2., 41)
    activity = np.broadcast_to(np.arange(40)[None, :, None] % 2, (len(rows), 40, 1)).copy()
    window = replace(prepared.windows[PAIRED_WINDOW_S, 100],
                     row_positions=rows, bin_edges_s=edges, counts={"PFC": activity, "HPC": activity})
    prepared = replace(prepared, trials=trials, choices=choices, events=events, targets=targets,
                       unit_ids={"PFC": ("B:0",), "HPC": ("A:0",)},
                       windows={(PAIRED_WINDOW_S, 100): window})
    starts = trial_times[rows, None] + edges[:-1]
    sign = np.where(np.arange(len(rows)) // 4 == 0, 1, -1)[:, None]
    speeds = 10 + sign * 4 * (2 * activity[..., 0] - 1) + (np.arange(len(rows)) % 4)[:, None]
    utc = np.r_[starts.ravel(), trial_times[rows[-1]] + edges[-1]]
    return prepared, utc, -np.r_[speeds.ravel(), 0.]


def test_joint_training_sign_keeps_negative_test_correlation_and_earliest_replay(movement_inputs):
    """A training-only sign flips both regional scores and retains measured speed."""
    prepared, utc, velocity = movement_inputs
    result = compute_movement(prepared, utc, velocity)
    block = result["block_scores"].query("block_id == 20").set_index("metric")
    assert block.loc["pfc_hpc", "correlation"] == pytest.approx(1.)
    assert block.loc["pfc_speed", "correlation"] < -.9
    assert block.loc["hpc_speed", "correlation"] < -.9
    assert (block.train_anchor < -.9).all()
    assert (block.joint_sign == -1).all()
    assert (block.n_train_choices == 8).all()
    assert (block.n_test_choices == 4).all()
    assert (block.n_pairs == 160).all()
    replay = result["replay_tracks"]
    assert replay.row_position.drop_duplicates().tolist() == [0, 2, 3, 4]
    assert replay.trial_id.drop_duplicates().tolist() == [100, 102, 103, 104]
    assert replay.block_id.nunique() == 1
    assert len(replay) == 160
    assert replay.pfc_score.iloc[1] < replay.pfc_score.iloc[0]
    np.testing.assert_allclose(replay.pfc_score, replay.hpc_score)
    assert (replay.speed_mm_s > 0).all()
    assert result["replay_speed_increase_mm_s"] == pytest.approx(2.)
    assert result["speed_units"] == "mm/s"
    assert result["replay_reason"] == ""
    markers = result["replay_events"]
    assert markers.query("event == 'choice'").row_position.tolist() == [0, 2, 3, 4]
    delivery = markers.query("event == 'reward_delivery'")
    assert delivery.row_position.tolist() == [0]
    assert delivery.time_s.iloc[0] == pytest.approx(
        prepared.trials.choice_time.iloc[0] + .05, abs=1e-6, rel=0,
    )


@pytest.mark.parametrize("missing", [True, False])
def test_undefined_training_anchor_preserves_neural_coupling(movement_inputs, missing):
    """Missing or constant speed leaves movement scores undefined for every block."""
    prepared, utc, velocity = movement_inputs
    result = compute_movement(
        prepared, utc[:0] if missing else utc, velocity[:0] if missing else np.ones_like(velocity),
        speed_reason="No treadmill source." if missing else "",
    )
    scores = result["block_scores"]
    np.testing.assert_allclose(scores.query("metric == 'pfc_hpc'").correlation, np.ones(3))
    assert scores.query("metric != 'pfc_hpc'").correlation.isna().all()
    assert scores.query("metric != 'pfc_hpc'").reason.str.len().gt(0).all()
    assert scores.train_anchor.isna().all()
    assert result["replay_tracks"].empty
    assert result["replay_reason"]


def test_holdout_purges_an_overlapping_choice_from_another_block(movement_inputs):
    """Use physical four-second windows for the finite training support count."""
    prepared, _, _ = movement_inputs
    trials = prepared.trials.copy()
    trials.loc[5, "choice_time"] = trials.loc[4, "choice_time"] + 3.0
    result = compute_movement(replace(prepared, trials=trials), np.empty(0), np.empty(0))
    scores = result["block_scores"].query("metric == 'pfc_hpc'").set_index("block_id")
    assert scores.n_purged_choices.to_dict() == {20: 1, 21: 1, 22: 0}
    assert scores.n_train_choices.to_dict() == {20: 7, 21: 7, 22: 8}
