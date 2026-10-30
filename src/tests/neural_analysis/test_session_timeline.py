"""Scientific contracts for the session-time viewer (seconds, counts, fixed PCs)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src.neural_analysis.population import session_timeline as timeline


def test_completed_choices_keep_origin_ids_and_last_saved_choice():
    """A processed trial table defines choices; flag/action filtering never resets time."""
    trials = pd.DataFrame({
        "cur_trial": [10, 11, 12, 13, 14, 15],
        "start_time": [100., 110., 120., 130., 140., 150.],
        "choice_time": [np.nan, 112., 122., 132., np.inf, 152.],
        "action": ["no_choice", "1", 0, 1, 0, 0],
        "experimenter_reward_given": [1, "0", 0, 1, 0, 0],
        "correct": [0, 0, 1, 0, 0, 0],
        "reward": [1, 0, 0, 1, 0, 0],
    }, index=[20, 30, 40, 50, 60, 70])
    origin, choices = timeline.prepare_timeline_choices(trials)
    assert origin == 100.
    assert choices["row_position"].tolist() == [1, 2, 5]
    assert choices["trial_id"].tolist() == [11, 12, 15]
    assert choices["action"].tolist() == [1, 0, 0]
    np.testing.assert_array_equal(choices["elapsed_s"], [12., 22., 52.])


def test_legacy_manual_reward_flags_use_existing_convention():
    """Legacy give_reward flags exclude manual rewards without condition filters."""
    trials = pd.DataFrame({"cur_trial": [0, 1], "start_time": [0., 10.],
                           "choice_time": [2., 12.], "action": [1, 0],
                           "give_reward": ["1", "0"]})
    _, choices = timeline.prepare_timeline_choices(trials)
    assert choices.trial_id.tolist() == [1]


def test_coverage_is_regional_and_does_not_exclude_elapsed_zero():
    """Recording bounds, rather than behavioral origin or spike presence, decide coverage."""
    times = np.array([101., 110., 120.])
    np.testing.assert_array_equal(
        timeline.select_complete_choice_windows(times, (98., 123.)), [0, 1, 2])
    np.testing.assert_array_equal(
        timeline.select_complete_choice_windows(times, (105., 123.)), [1, 2])


@pytest.mark.parametrize("bin_width, bin_count", [(0.5, 8), (0.1, 40), (0.05, 80), (0.02, 200)])
def test_binning_preserves_half_open_edges_at_real_utc(bin_width, bin_count):
    """Each event-relative count tensor is (window, bin, unit), without smoothing."""
    choice = 1_700_000_100.123
    spikes = np.array([choice - 2, choice - 2 + bin_width, choice,
                       choice + 2 - bin_width / 2, choice + 2])
    group = nap.TsGroup({7: nap.Ts(t=spikes)})
    counts, edges = timeline.bin_choice_counts(group, np.array([7]), np.array([choice]), bin_width)
    assert counts.shape == (1, bin_count, 1)
    np.testing.assert_allclose(edges, -2 + np.arange(bin_count + 1) * bin_width)
    expected = np.zeros(bin_count)
    expected[[0, 1, bin_count // 2, bin_count - 1]] = 1
    np.testing.assert_array_equal(counts[0, :, 0], expected)


def test_silent_covered_window_remains_valid_counts():
    """An absence of spikes within a covered window is measured zero activity."""
    group = nap.TsGroup({7: nap.Ts(t=[90., 110., 130.])})
    counts, _ = timeline.bin_choice_counts(group, np.array([7]), np.array([100., 120.]), .5)
    np.testing.assert_array_equal(counts, np.zeros((2, 8, 1)))


def test_pca_uses_pooled_scaling_full_basis_and_direct_variance():
    """The retained transform and scores conserve standardized-unit sample variance."""
    rng = np.random.default_rng(20261030)
    counts = rng.poisson(3, size=(4, 8, 4)).astype(float)
    counts[:, :, 3] = 0.
    result = timeline.fit_timeline_activity(
        counts, ("A:7", "A:8", "A:9", "A:10"), np.arange(4), "PFC", "A", .5)
    transform = result.transform
    assert transform.retained_unit_ids == ("A:7", "A:8", "A:9")
    assert transform.omitted_unit_ids == ("A:10",)
    assert result.scores.shape == (4, 8, 3)
    flat = counts[:, :, :3].reshape(-1, 3)
    standardized = (flat - flat.mean(axis=0)) / flat.std(axis=0, ddof=0)
    expected_scores = standardized @ transform.components.T
    np.testing.assert_allclose(result.scores.reshape(-1, 3), expected_scores, atol=1e-12)
    np.testing.assert_allclose(result.total_session_variance, standardized.var(axis=0, ddof=1).sum())
    expected_window = standardized.reshape(4, 8, 3).var(axis=1, ddof=1).sum(axis=1)
    np.testing.assert_allclose(result.window_total_variance, expected_window)
    np.testing.assert_allclose(result.scores.var(axis=1, ddof=1).sum(axis=1), expected_window)
    assert result.color_limit == np.abs(result.scores).max()


def test_footer_uses_total_variance_including_unretained_axes():
    """A truncated spectrum is never renormalized to make a threshold reachable."""
    assert timeline.summarize_session_variance(np.array([6., 2., 1., 1.]), 10.) == {
        .8: 2, .9: 3, .95: 4}
    assert timeline.summarize_session_variance(np.array([6., 2.]), 10.) == {
        .8: 2, .9: None, .95: None}


def test_one_variable_unit_can_be_viewed_but_all_constant_is_unavailable():
    """A one-PC population is valid; no variable units is an explicit input error."""
    counts = np.arange(8, dtype=float).reshape(1, 8, 1)
    result = timeline.fit_timeline_activity(counts, ("A:7",), np.array([0]), "PFC", "A", .5)
    assert result.scores.shape == (1, 8, 1)
    with pytest.raises(ValueError, match="variable"):
        timeline.fit_timeline_activity(np.zeros_like(counts), ("A:7",), np.array([0]), "PFC", "A", .5)


def test_nonfinite_activity_is_not_silently_repaired_or_dropped():
    """Missing neural observations must not become zero measurements or removed units."""
    counts = np.arange(16, dtype=float).reshape(1, 8, 2)
    counts[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        timeline.fit_timeline_activity(counts, ("A:1", "A:2"), np.array([0]), "PFC", "A", .5)


@pytest.mark.parametrize("width", [0., .03, 1., np.nan])
def test_only_specified_bin_widths_are_supported(width):
    """The viewer's four resolutions are explicit, not arbitrary partial-bin windows."""
    with pytest.raises(ValueError, match="bin"):
        timeline.bin_choice_counts(nap.TsGroup({1: nap.Ts(t=[0., 1.])}),
                                   np.array([1]), np.array([.5]), width)
