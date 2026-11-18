"""Selected-source activity, original trial joins, and session-wide display scaling."""

from dataclasses import replace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences.activity import (
    SourceActivity, assemble_heatmap, extract_unit_activity,
    scale_activity, select_pc_activity,
)
from src.neural_analysis.population import session_timeline as timeline


ORIGIN_S = 1_700_000_000.123


def _trials(size: int = 7) -> pd.DataFrame:
    """Input: integer N. Return (N,8) chronological trials, UTC seconds and unitless labels."""
    return pd.DataFrame({
        "cur_trial": np.arange(101, 101 + size),
        "start_time": ORIGIN_S + np.arange(size) * 10 - 3,
        "choice_time": ORIGIN_S + np.arange(size) * 10,
        "action": np.arange(size) % 2,
        "correct": np.ones(size, dtype=int),
        "reward": np.arange(size) % 2,
        "experimenter_reward_given": np.zeros(size, dtype=int),
        "cur_block": np.zeros(size, dtype=int),
    }, index=[f"source-{position // 2}" for position in range(size)])


def _source(values: np.ndarray, kind: str = "unit", positions: np.ndarray | None = None) -> SourceActivity:
    """Input: float (W,8) values, kind, optional int (W,) positions. Return selected source.

    Unit values are Hz; PC values are signed unwhitened standardized-unit scores.
    Positions are original table positions; edges (9,) are choice-relative seconds.
    """
    if positions is None:
        positions = np.arange(len(values), dtype=int)
    return SourceActivity(
        values=np.asarray(values, dtype=float), row_positions=positions,
        bin_edges_s=np.linspace(-2, 2, 9), source_kind=kind,
        source_label="A:7" if kind == "unit" else "PFC/A PC1",
    )


@pytest.fixture
def pc_inputs() -> tuple[pd.DataFrame, pd.DataFrame, timeline.RegionalTimelinePCA]:
    """Return full (7,C) trials, (4,C) shared choices and regional (3,8,2) scores.

    regional positions [1,2,3] map to original positions [3,4,6], separate from
    trial IDs [104,105,107] and duplicated string DataFrame labels. Times are UTC seconds.
    """
    trials = _trials().astype(object)
    trials.iloc[0, trials.columns.get_loc("experimenter_reward_given")] = 1
    trials.iloc[2, trials.columns.get_loc("action")] = "no_choice"
    trials.iloc[5, trials.columns.get_loc("experimenter_reward_given")] = "None"
    _, choices = timeline.prepare_timeline_choices(trials)
    counts = np.random.default_rng(20261105).poisson(3, size=(3, 8, 2))
    regional = timeline.fit_timeline_activity(
        counts, ("A:7", "A:8"), np.array([1, 2, 3]), "PFC", "A", 0.5,
    )
    return trials, choices, regional


@pytest.mark.parametrize("width,bins", [(0.5, 8), (0.1, 40), (0.05, 80), (0.02, 200)])
def test_unit_rates_reuse_half_open_edges_at_real_utc_without_smoothing(width: float, bins: int) -> None:
    """Input: seconds/bin count. Return None; compare (1,B) unsmoothed Hz with exact edge spikes."""
    trials = _trials(1)
    spikes = ORIGIN_S + np.array([-2, -2 + width, 0, 2 - width / 2, 2])
    source = extract_unit_activity(spikes, trials, (ORIGIN_S - 2, ORIGIN_S + 2), width, source_label="A:7")
    expected = np.zeros(bins)
    expected[[0, 1, bins // 2, bins - 1]] = 1 / width
    assert source.values.shape == (1, bins)
    np.testing.assert_array_equal(source.values[0], expected)
    np.testing.assert_allclose(source.bin_edges_s, np.linspace(-2, 2, bins + 1))
    np.testing.assert_array_equal(source.row_positions, [0])
    assert source.source_kind == "unit" and source.source_label == "A:7"


@pytest.mark.parametrize("spikes", [np.array([]), np.array([ORIGIN_S + 10]), np.array([ORIGIN_S - 100, ORIGIN_S + 100])])
def test_covered_silence_and_single_timestamp_trains_are_valid_raw_sources(spikes: np.ndarray) -> None:
    """Input: (S,) UTC spikes. Return None; trusted bounds retain silent covered (1,8) window."""
    trials = _trials(1)
    source = extract_unit_activity(spikes, trials, (ORIGIN_S - 2, ORIGIN_S + 2), 0.5, source_label="B:7")
    np.testing.assert_array_equal(source.values, np.zeros((1, 8)))
    assert source.source_label == "B:7"


def test_unit_extraction_uses_all_ordinary_outcomes_and_preserves_original_positions() -> None:
    """No inputs; return None after verifying ordinary full-table pool and immutable inputs."""
    trials = _trials().astype(object)
    trials.iloc[0, trials.columns.get_loc("experimenter_reward_given")] = 1
    trials.iloc[2, trials.columns.get_loc("action")] = "no_choice"
    trials.iloc[4, trials.columns.get_loc("correct")] = "None"
    trials.iloc[5, trials.columns.get_loc("experimenter_reward_given")] = "None"
    trials.iloc[6, trials.columns.get_loc("correct")] = 0
    original = trials.copy(deep=True)
    spikes = ORIGIN_S + np.array([10, 30, 40, 60])
    source = extract_unit_activity(spikes, trials, (ORIGIN_S - 2, ORIGIN_S + 62), 0.5, source_label="A:7")
    np.testing.assert_array_equal(source.row_positions, [1, 3, 4, 6])
    np.testing.assert_array_equal(source.values[:, 4], [2, 2, 2, 2])
    pd.testing.assert_frame_equal(trials, original)
    np.testing.assert_array_equal(spikes, ORIGIN_S + np.array([10, 30, 40, 60]))


def test_unit_extraction_bins_only_one_requested_unit_and_complete_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Input: patch fixture. Return None; existing routine receives U=1 and only covered UTC choices."""
    record_binning = Mock(wraps=timeline.bin_choice_counts)
    monkeypatch.setattr(timeline, "bin_choice_counts", record_binning)
    trials = _trials(3)
    source = extract_unit_activity(np.array([ORIGIN_S + 10]), trials, (ORIGIN_S + 8, ORIGIN_S + 12), 0.5, source_label="A:7")
    record_binning.assert_called_once()
    assert np.asarray(record_binning.call_args.args[1]).shape == (1,)
    np.testing.assert_array_equal(record_binning.call_args.args[2], [ORIGIN_S + 10])
    np.testing.assert_array_equal(source.row_positions, [1])


@pytest.mark.parametrize("coverage", [(np.nan, ORIGIN_S), (ORIGIN_S + 2, ORIGIN_S - 2), None])
def test_unit_extraction_rejects_untrusted_or_invalid_coverage(coverage) -> None:
    """Input: invalid UTC pair/None. Return None; finite spikes never substitute recording bounds."""
    with pytest.raises(ValueError, match="coverage"):
        extract_unit_activity(np.array([ORIGIN_S - 3, ORIGIN_S + 3]), _trials(1), coverage, 0.5, source_label="A:7")


@pytest.mark.parametrize("spikes", [np.array([[ORIGIN_S]]), np.array([np.nan]), np.array([np.inf])])
def test_unit_spikes_require_finite_one_dimensional_seconds(spikes: np.ndarray) -> None:
    """Input: malformed UTC spike array. Return None; never reshape or repair neural observations."""
    with pytest.raises(ValueError):
        extract_unit_activity(spikes, _trials(1), (ORIGIN_S - 3, ORIGIN_S + 3), 0.5, source_label="A:7")


@pytest.mark.parametrize("rank", [1, 2])
def test_pc_selection_joins_regional_shared_and_original_position_identities(pc_inputs, rank: int) -> None:
    """Input: fixture tuple and one-based rank. Return None; verify (3,8) signed original scores."""
    trials, choices, regional = pc_inputs
    originals = (trials.copy(deep=True), choices.copy(deep=True), regional.scores.copy(), regional.transform.components.copy())
    source = select_pc_activity(trials, choices, regional, rank)
    np.testing.assert_array_equal(source.row_positions, [3, 4, 6])
    np.testing.assert_array_equal(source.values, regional.scores[:, :, rank - 1])
    np.testing.assert_array_equal(source.bin_edges_s, regional.bin_edges_s)
    assert source.source_kind == "pc" and "A" in source.source_label and str(rank) in source.source_label
    pd.testing.assert_frame_equal(trials, originals[0])
    pd.testing.assert_frame_equal(choices, originals[1])
    np.testing.assert_array_equal(regional.scores, originals[2])
    np.testing.assert_array_equal(regional.transform.components, originals[3])


@pytest.mark.parametrize("rank", [0, -1, 3, 1.5, True])
def test_pc_rank_is_a_valid_one_based_integer(pc_inputs, rank) -> None:
    """Input: fixture tuple and invalid scalar rank. Return None; reject out-of-range/noninteger ranks."""
    with pytest.raises(ValueError, match="rank"):
        select_pc_activity(*pc_inputs, rank)


@pytest.mark.parametrize("column", ["row_position", "trial_id", "choice_time", "action"])
def test_pc_join_rejects_original_trial_identity_mismatches(pc_inputs, column: str) -> None:
    """Input: fixture tuple/identity field. Return None; a selected-row mismatch cannot silently join."""
    trials, choices, regional = pc_inputs
    changed = choices.copy(deep=True)
    changed.loc[1, column] += 1
    with pytest.raises(ValueError, match="identity|row_position|trial_id|choice_time|action"):
        select_pc_activity(trials, changed, regional, 1)


@pytest.mark.parametrize("identity", [np.nan, "None"])
def test_matching_missing_trial_ids_preserve_eligible_pc_reference(pc_inputs, identity) -> None:
    """Input: fixture tuple/supported missing ID scalar. Return None; all three eligible windows survive.

    Missing trial IDs break behavioral motif adjacency, but do not remove an
    otherwise ordinary complete choice from the session-wide neural reference.
    """
    trials, choices, regional = pc_inputs
    trials = trials.copy(deep=True)
    choices = choices.copy(deep=True)
    trials.iloc[3, trials.columns.get_loc("cur_trial")] = identity
    choices.iloc[1, choices.columns.get_loc("trial_id")] = identity
    source = select_pc_activity(trials, choices, regional, 1)
    np.testing.assert_array_equal(source.row_positions, [3, 4, 6])
    np.testing.assert_array_equal(source.values, regional.scores[:, :, 0])
    scaled = scale_activity(source)
    assert scaled["session_mean"] == pytest.approx(regional.scores[:, :, 0].mean())
    assert scaled["session_sd"] == pytest.approx(regional.scores[:, :, 0].std(ddof=0))


@pytest.mark.parametrize("source_id,saved_id,compatible", [
    (104, "104", True), ("104", 104, True),
    ("invalid-source", "invalid-saved", False), ("None", "invalid-saved", False),
])
def test_pc_trial_id_comparison_preserves_numeric_and_missing_semantics(
    pc_inputs, source_id, saved_id, compatible: bool,
) -> None:
    """Input: fixture/ID scalars/expected bool. Return None; numeric strings agree, distinct bad tokens do not."""
    trials, choices, regional = pc_inputs
    trials = trials.copy(deep=True)
    choices = choices.copy(deep=True)
    trials.iloc[3, trials.columns.get_loc("cur_trial")] = source_id
    choices.iloc[1, choices.columns.get_loc("trial_id")] = saved_id
    if compatible:
        source = select_pc_activity(trials, choices, regional, 1)
        np.testing.assert_array_equal(source.row_positions, [3, 4, 6])
    else:
        with pytest.raises(ValueError, match="trial_id|identity"):
            select_pc_activity(trials, choices, regional, 1)


@pytest.mark.parametrize("corruption", ["map_bounds", "map_float", "map_duplicate", "score_shape", "nonfinite", "edges"])
def test_pc_selection_rejects_invalid_saved_axes(pc_inputs, corruption: str) -> None:
    """Input: fixture tuple/corruption name. Return None; malformed (W,B,K) and (W,) mappings fail."""
    trials, choices, regional = pc_inputs
    if corruption == "map_bounds":
        regional = replace(regional, choice_positions=np.array([1, 2, 99]))
    elif corruption == "map_float":
        regional = replace(regional, choice_positions=np.array([1.0, 2.0, 3.0]))
    elif corruption == "map_duplicate":
        regional = replace(regional, choice_positions=np.array([1, 1, 3]))
    elif corruption == "score_shape":
        regional = replace(regional, scores=regional.scores[:, :-1])
    elif corruption == "nonfinite":
        scores = regional.scores.copy()
        scores[0, 0, 0] = np.nan
        regional = replace(regional, scores=scores)
    else:
        regional = replace(regional, bin_edges_s=regional.bin_edges_s[:-1])
    with pytest.raises(ValueError):
        select_pc_activity(trials, choices, regional, 1)


def test_unrelated_pc_amplitude_cannot_change_selected_pc_scaling_or_limits(pc_inputs) -> None:
    """Input: fixture tuple. Return None; scaling uses selected (W,B) PC, never regional max over K."""
    trials, choices, regional = pc_inputs
    first = scale_activity(select_pc_activity(trials, choices, regional, 1))
    scores = regional.scores.copy()
    scores[:, :, 1] *= 10_000
    changed = replace(regional, scores=scores, color_limit=float(np.abs(scores).max()))
    second = scale_activity(select_pc_activity(trials, choices, changed, 1))
    np.testing.assert_array_equal(first["values"], second["values"])
    assert first["color_limits"] == second["color_limits"]
    assert first["session_mean"] == second["session_mean"]
    assert first["session_sd"] == second["session_sd"]


@pytest.mark.parametrize("kind", ["unit", "pc"])
def test_session_scaling_uses_all_eligible_bins_population_sd_and_native_units(kind: str) -> None:
    """Input: source kind. Return None; verify all (4,8) bins define raw/z-scored session references."""
    values = np.arange(32, dtype=float).reshape(4, 8)
    if kind == "pc":
        values -= 23
    source = _source(values, kind)
    original = source.values.copy()
    raw = scale_activity(source)
    scaled = scale_activity(source, zscore=True)
    assert raw["session_mean"] == pytest.approx(values.mean())
    assert raw["session_sd"] == pytest.approx(values.std(ddof=0))
    np.testing.assert_array_equal(raw["values"], values)
    np.testing.assert_allclose(scaled["values"], (values - values.mean()) / values.std(ddof=0))
    maximum = float(np.abs(values).max())
    assert raw["color_limits"] == ((0.0, maximum) if kind == "unit" else (-maximum, maximum))
    scaled_max = float(np.abs(scaled["values"]).max())
    assert scaled["color_limits"] == (-scaled_max, scaled_max)
    assert raw["value_units"] == ("Hz" if kind == "unit" else "unwhitened standardized-unit scores")
    assert scaled["value_units"] == "source SD units"
    assert scaled["zscore_available"] and not scaled["unavailable_reason"]
    np.testing.assert_array_equal(source.values, original)


def test_overlapping_episodes_and_panel_selection_cannot_reweight_session_reference() -> None:
    """No inputs; return None after repeated requested windows leave the unique (W,B) reference unchanged."""
    source = _source(np.arange(32, dtype=float).reshape(4, 8), "pc")
    scaled = scale_activity(source, zscore=True)
    episodes = np.array([[0, 1, 2], [1, 2, 3]])
    shown = assemble_heatmap(scaled, episodes)
    np.testing.assert_allclose(shown["values"], scaled["values"][episodes])
    one_panel = assemble_heatmap(scaled, np.array([3]))
    np.testing.assert_allclose(one_panel["values"], scaled["values"][[3]])
    assert scaled["session_mean"] == pytest.approx(source.values.mean())
    assert scaled["session_sd"] == pytest.approx(source.values.std(ddof=0))


@pytest.mark.parametrize("kind,value", [("unit", 0.0), ("unit", 2.0), ("pc", 0.0), ("pc", -5.0)])
def test_zero_sd_keeps_raw_constant_activity_and_explains_unavailable_zscore(kind: str, value: float) -> None:
    """Input: kind/scalar activity. Return None; raw (2,8) constant values valid, z-scoring disabled."""
    source = _source(np.full((2, 8), value), kind)
    raw = scale_activity(source)
    np.testing.assert_array_equal(raw["values"], source.values)
    assert raw["session_sd"] == 0 and raw["constant_activity"]
    assert not raw["zscore_available"] and not raw["unavailable_reason"]
    maximum = abs(value) or 1.0
    assert raw["color_limits"] == ((0.0, maximum) if kind == "unit" else (-maximum, maximum))
    scaled = scale_activity(source, zscore=True)
    assert scaled["values"] is None and scaled["color_limits"] is None
    assert "zero" in scaled["unavailable_reason"].lower()
    shown = assemble_heatmap(scaled, np.array([0, 1]))
    assert shown["n"] == 0 and shown["excluded_count"] == 0
    assert shown["unavailable_reason"]


@pytest.mark.parametrize("zscore", [False, True])
def test_zero_eligible_windows_have_no_guessed_reference_or_limits(zscore: bool) -> None:
    """Input: bool display mode. Return None; zero-W source is explicitly unavailable without mean/SD guesses."""
    source = extract_unit_activity(np.array([]), _trials(1), (ORIGIN_S + 10, ORIGIN_S + 20), 0.5, source_label="A:7")
    assert source.values.shape == (0, 8)
    scaled = scale_activity(source, zscore=zscore)
    assert scaled["values"] is None and scaled["color_limits"] is None
    assert scaled["session_mean"] is None and scaled["session_sd"] is None
    assert scaled["unavailable_reason"] and not scaled["zscore_available"]
    shown = assemble_heatmap(scaled, np.array([0]))
    assert shown["behavioral_count"] == 1 and shown["n"] == 0 and shown["excluded_count"] == 1
    assert shown["unavailable_reason"]


def test_central_coverage_uses_only_displayed_trial_and_original_positions() -> None:
    """No inputs; return None after covered central activity survives uncovered neighboring trial windows."""
    trials = _trials(3)
    source = extract_unit_activity(np.array([]), trials, (ORIGIN_S + 8, ORIGIN_S + 12), 0.5, source_label="A:7")
    shown = assemble_heatmap(scale_activity(source), np.array([1]))
    assert shown["n"] == shown["behavioral_count"] == 1 and shown["excluded_count"] == 0
    np.testing.assert_array_equal(shown["row_positions"], [1])
    np.testing.assert_array_equal(shown["values"], np.zeros((1, 8)))


def test_one_missing_episode_segment_excludes_entire_episode_without_padding() -> None:
    """No inputs; return None after logical (E,3,B) assembly removes one incomplete whole episode."""
    source = _source(np.arange(32, dtype=float).reshape(4, 8), positions=np.array([10, 11, 12, 14]))
    matches = np.array([[10, 11, 12], [11, 12, 13]])
    original = matches.copy()
    shown = assemble_heatmap(scale_activity(source), matches)
    assert shown["behavioral_count"] == 2 and shown["n"] == 1 and shown["excluded_count"] == 1
    assert shown["values"].shape == (1, 3, 8)
    np.testing.assert_array_equal(shown["row_positions"], [[10, 11, 12]])
    np.testing.assert_array_equal(shown["values"][0], source.values[:3])
    np.testing.assert_array_equal(matches, original)


@pytest.mark.parametrize("matches,expected_shape", [(np.array([], dtype=int), (0, 8)), (np.empty((0, 6), dtype=int), (0, 6, 8)), (np.array([99]), (0, 8)), (np.array([[99, 100, 101]]), (0, 3, 8))])
def test_empty_and_all_uncovered_matches_retain_logical_shapes(matches: np.ndarray, expected_shape: tuple) -> None:
    """Input: empty/uncovered position arrays and result shape. Return None; never fabricate zero observations."""
    shown = assemble_heatmap(scale_activity(_source(np.ones((1, 8)))), matches)
    assert shown["values"].shape == expected_shape
    assert shown["n"] == 0 and shown["behavioral_count"] == len(matches)
    assert shown["excluded_count"] == len(matches)


@pytest.mark.parametrize("corruption", ["values_ndim", "values_nonfinite", "positions_shape", "positions_float", "duplicate_positions", "edges_shape", "source_kind", "negative_unit"])
def test_source_activity_validates_its_data_contract(corruption: str) -> None:
    """Input: invalid field name. Return None; reject malformed/nonfinite/ambiguous source axes."""
    fields = {
        "values": np.ones((2, 8)), "row_positions": np.array([10, 11]),
        "bin_edges_s": np.linspace(-2, 2, 9), "source_kind": "unit", "source_label": "A:7",
    }
    if corruption == "values_ndim":
        fields["values"] = np.ones((2, 8, 1))
    elif corruption == "values_nonfinite":
        fields["values"][0, 0] = np.nan
    elif corruption == "positions_shape":
        fields["row_positions"] = np.array([10])
    elif corruption == "positions_float":
        fields["row_positions"] = np.array([10.0, 11.0])
    elif corruption == "duplicate_positions":
        fields["row_positions"] = np.array([10, 10])
    elif corruption == "edges_shape":
        fields["bin_edges_s"] = np.linspace(-2, 2, 8)
    elif corruption == "source_kind":
        fields["source_kind"] = "population"
    else:
        fields["values"][0, 0] = -1
    with pytest.raises(ValueError):
        scale_activity(SourceActivity(**fields))
