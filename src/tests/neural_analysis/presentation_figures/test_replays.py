"""Behavior-first motif and fixed-readout tests for elapsed-time replays."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population_decoding.targets import build_population_targets


@pytest.fixture
def replay_source():
    """Return synthetic prepared tables/windows with six blocks and a manual row."""
    action = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1,
                       1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0])
    reward = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
                       0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1])
    positions = np.arange(len(action))
    blocks = positions // 6
    context = (blocks + 1) % 2
    manual = (positions == 21).astype(int)
    choices_s = 101. + 3 * positions
    choices_s[12] = choices_s[11] + 1
    start_s = choices_s - .7
    correct = (action == context).astype(int)
    trials = pd.DataFrame(dict(row_position=positions, cur_trial=positions + 1000,
                               cur_block=blocks, start_time=start_s, choice_time=choices_s,
                               state_int=context, action=action, correct=correct, reward=reward,
                               experimenter_reward_given=manual))
    choices = trials.loc[manual == 0].assign(elapsed_s=choices_s[manual == 0] - 100.)
    events = [(start_s[i] - .1, f"enter_{'left' if context[i] else 'right'}_patch")
              for i in range(0, len(trials), 6)]
    for i in positions[(reward == 1) & (positions != 9)]:
        pump = 2 if action[i] else 3
        events.extend([(choices_s[i] + .01, f"pump{5 - pump} reward_amount: 3"),
                       (choices_s[i] + .05, f"pump{pump} reward_amount: 3"),
                       (choices_s[i] + .07, f"pump{pump} reward_amount: 3")])
    events = pd.DataFrame(sorted(events), columns=["Time", "Event"])
    events["elapsed_s"] = events.Time - 100.
    covered = choices.row_position.to_numpy()[1:]
    windows = {}
    for width_ms in (100, 500):
        edges = np.linspace(-1, 1, 2000 // width_ms + 1)
        per_choice = np.column_stack([covered + 1, 1 + 15 * action[covered], 1 + 15 * reward[covered]])
        counts = np.broadcast_to(per_choice[:, None, :], (len(covered), len(edges) - 1, 3)).copy()
        windows[((-1., 1.), width_ms)] = SimpleNamespace(
            row_positions=covered.copy(), bin_edges_s=edges, counts={"PFC": counts, "HPC": counts.copy()},
        )
    return SimpleNamespace(
        session_id="synthetic-01", time_origin_s=100., trials=trials, choices=choices, events=events,
        baseline=manual == 0, targets=build_population_targets(context, action, reward, manual),
        condition_masks={"overall": manual == 0, "correct_rewarded": (correct == 1) & (reward == 1),
                         "omission": (correct == 1) & (reward == 0), "incorrect": correct == 0},
        unit_ids={"PFC": ("a", "b", "c"), "HPC": ("d", "e", "f")}, windows=windows,
        coverage_s={"PFC": (101., 211.), "HPC": (101., 211.)}, unavailable={}, meta={},
    )


def test_deliveries_use_first_matching_pump_inside_rewarded_trial(replay_source):
    """A wrong-side pump or second matching event cannot replace the delivery."""
    from src.neural_analysis.presentation_figures.replays import recorded_deliveries

    deliveries = recorded_deliveries(replay_source, np.array([8, 9, 7, 14, 21]))
    np.testing.assert_allclose(deliveries, [125.05, np.nan, np.nan, 143.05, np.nan])


def test_motifs_use_original_positions_logged_switch_and_exact_choice_padding(replay_source):
    """Validate both switch rules and earliest shortest outcome excerpt."""
    from src.neural_analysis.presentation_figures.replays import compute_replays

    result = compute_replays(replay_source)
    episodes = result["episodes"].set_index("motif")
    assert set(episodes.status) == {"available"}
    trials = result["trials"]
    assert trials.query("motif == 'new_side_omission'").row_position.tolist() == list(range(3, 12))
    assert trials.query("motif == 'ideal_transition'").row_position.tolist() == list(range(9, 18))
    assert episodes.loc["ideal_transition", "repeat_count"] == 2
    assert trials.query("motif == 'four_outcome' and in_motif").row_position.tolist() == [4, 5, 6, 7]
    assert trials.query("motif == 'four_outcome'").row_position.tolist() == list(range(1, 11))
    a1 = result["events"].query("motif == 'new_side_omission'").set_index("event")
    assert a1.loc["switch", "elapsed_s"] == pytest.approx(18.2)
    assert a1.loc["first_new_side_choice", "trial_id"] == 1007
    assert a1.loc["first_new_side_delivery", "elapsed_s"] == pytest.approx(25.05)
    assert a1.loc["delivery_missing", "trial_id"] == 1009


def test_outcome_padding_retains_intervening_nonordinary_rows(replay_source):
    """Find motif on choice chronology while retaining every original display row."""
    from src.neural_analysis.presentation_figures.replays import compute_replays

    for window in replay_source.windows.values():
        keep = window.row_positions >= 19
        window.row_positions = window.row_positions[keep]
        window.counts = {region: counts[keep] for region, counts in window.counts.items()}
    result = compute_replays(replay_source)
    rows = result["trials"].query("motif == 'four_outcome'")
    assert rows.row_position.tolist() == list(range(19, 30))
    assert rows.query("in_motif").row_position.tolist() == [23, 24, 25, 26]
    assert 21 not in set(result["tracks"].row_position)


def test_tracks_share_fixed_held_out_block_fit_and_width_specific_bins(replay_source, monkeypatch):
    """Purging drops overlapping row 12, and right is probability column zero."""
    from src.neural_analysis.presentation_figures import replays

    fits_seen = []
    fit_readout = replays.fit_readout

    def capture(rates_hz, labels, unit_ids, classes, dimensions):
        """Capture (choice, bin, unit) synthetic Hz before each fixed fit."""
        bins = rates_hz.shape[1]
        if len(classes) == 2:
            width_s = 2 / bins
            training_ids = np.rint(rates_hz[:, 0, 0] * width_s).astype(int) - 1
            fits_seen.append((bins, tuple(training_ids)))
        return fit_readout(rates_hz, labels, unit_ids, classes, dimensions)

    monkeypatch.setattr(replays, "fit_readout", capture)
    result = replays.compute_replays(replay_source)
    expected = tuple(i for i in range(13, 36) if i != 21)
    assert fits_seen.count((4, expected)) == fits_seen.count((20, expected)) == 4
    excluded = result["held_out_blocks"].query("motif == 'new_side_omission'")
    assert excluded.block_id.tolist() == [0, 1]
    tracks = result["tracks"].query("motif == 'new_side_omission' and region == 'PFC' and target == 'choice'")
    for width_ms, group in tracks.groupby("width_ms"):
        assert set(group.trial_id) == set(range(1003, 1012))
        np.testing.assert_allclose(group.bin_right_elapsed_s - group.bin_left_elapsed_s, width_ms / 1000)
        assert len(group) == 9 * (2000 // width_ms)
        assert group.query("trial_id == 1007").p_0.mean() > group.query("trial_id == 1006").p_0.mean()
    outcome = result["tracks"].query("motif == 'four_outcome' and region == 'PFC' and width_ms == 500")
    assert set(outcome.target) == {"outcome"}
    assert (outcome.bin_left_elapsed_s >= (outcome.row_position * 3 + 1)).all()
