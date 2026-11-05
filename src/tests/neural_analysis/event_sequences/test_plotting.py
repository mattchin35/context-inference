"""Common scientific display payloads for interactive and opaque PNG heatmaps."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import matplotlib.image
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import activity, io, plotting
from src.neural_analysis.event_sequences.selection import (
    CENTRAL_CONDITIONS, TRANSITION_CONDITIONS, prepare_sequence_trials,
    select_central_sequences, select_ideal_transitions,
)
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.spike_behavior import loading


def _trials(actions: list, correct: list, reward: list, blocks: list | None = None, states: list | None = None) -> pd.DataFrame:
    """Input: unitless (N,) behavioral labels. Return normalized (N,C) trials, UTC seconds and distinct identities."""
    size = len(actions)
    table = pd.DataFrame({
        "cur_trial": np.arange(101, 101 + size),
        "start_time": 1_700_000_000.123 + np.arange(size) * 20 - 3,
        "choice_time": 1_700_000_000.123 + np.arange(size) * 20,
        "action": actions, "correct": correct, "reward": reward,
        "experimenter_reward_given": [int(action == "no_choice") for action in actions],
        "cur_block": blocks if blocks is not None else [0] * size,
        "state_int": states if states is not None else [1] * size,
    }, index=[f"source-label-{index // 2}" for index in range(size)])
    return prepare_sequence_trials(table)


def _all_trials() -> pd.DataFrame:
    """No inputs; return (N,C) trials containing every motif, separated by manual no-choice rows.

    Times are UTC seconds, choices use 0=right/1=left. A final unmatched
    ordinary trial contributes to the global source scale but no display motif.
    """
    actions, correct, reward, blocks, states = [], [], [], [], []
    for central, side, previous, following in CENTRAL_CONDITIONS:
        action = int(side == "left")
        actions.extend([1 - action if previous == "incorrect" else action, action, action if following == "repeat" else 1 - action, "no_choice"])
        correct.extend([int(previous != "incorrect"), 1, 0, 0])
        reward.extend([int(previous == "correct_rewarded"), int(central == "correct_rewarded"), 0, 1])
        blocks.extend([0] * 4)
        states.extend([1] * 4)
    for episode, (direction, repeats) in enumerate(TRANSITION_CONDITIONS):
        old = int(direction == "left_to_right")
        actions.extend([old] * (repeats + 1) + [1 - old, "no_choice"])
        correct.extend([1] + [0] * repeats + [1, 0])
        reward.extend([1] + [0] * repeats + [1, 1])
        blocks.extend([100 + episode * 2] + [101 + episode * 2] * (repeats + 2))
        states.extend([old] + [1 - old] * (repeats + 1) + [2])
    actions.append(0)
    correct.append(0)
    reward.append(0)
    blocks.append(999)
    states.append(0)
    return _trials(actions, correct, reward, blocks, states)


def _packet(kind: str = "unit", width: float = 0.5, trials: pd.DataFrame | None = None, drop: tuple = ()) -> dict:
    """Input: source kind/width seconds/optional (N,C) trials/excluded original positions. Return saved-display packet.

    Unit Hz are deterministic (W,B); PC scores come from seeded actual pooled
    full-SVD PCA (W,B,2). Runtime-only fixed timeline avoids all source files.
    """
    if trials is None:
        trials = _all_trials()
    _, choices = timeline.prepare_timeline_choices(trials)
    selected = np.flatnonzero(~choices.row_position.isin(drop).to_numpy())
    rows = choices.row_position.to_numpy()[selected]
    edges = timeline.choice_bin_edges(width)
    bins = len(edges) - 1
    reference = {"region": "PFC", "probe_id": "A", "configuration_id": "fixed-pca-basis-01", "sha256": "a" * 64, "path": "analysis_runs/fixed/results.npz", "size": 123}
    packet = {
        "trials": trials, "central_matches": select_central_sequences(trials),
        "transition_matches": select_ideal_transitions(trials),
        "activity_row_positions": rows, "bin_edges_s": edges,
        "unit_rates_hz": np.arange(len(rows) * bins, dtype=float).reshape(len(rows), bins),
        "shared_choices": choices, "timeline_result": None,
        "meta": {
            "session_id": "synthetic-plotting", "configuration_id": "event-source-01",
            "source_kind": kind, "source_label": "A:7" if kind == "unit" else "PFC/A PCs (good_mua)",
            "parameters": {"probe_id": "A", "region": None if kind == "unit" else "PFC", "population_quality": "good" if kind == "unit" else "good_mua", "bin_width_s": width, "unit_id": "A:7" if kind == "unit" else None},
            "pca_reference": None if kind == "unit" else reference,
            "transition_unavailable_reason": "",
        },
    }
    if kind == "pc":
        regional = timeline.fit_timeline_activity(
            np.random.default_rng(20261105).poisson(3, size=(len(rows), bins, 2)),
            ("A:7", "A:8"), selected, "PFC", "A", width,
        )
        packet["timeline_result"] = timeline.SessionTimelineResult(
            "synthetic-plotting", 1_700_000_000.123 - 3, choices, pd.DataFrame(),
            {"PFC": regional}, {}, {"configuration_id": reference["configuration_id"]},
        )
    return packet


def _forbid_scientific_work(*args: object, **kwargs: object) -> None:
    """Input: arbitrary reader/fitter/publisher arguments. Raise if a pure display path performs scientific work."""
    raise AssertionError("Display selectors/rendering must use the loaded saved packet only")


def _field_values(panel: dict, name: str) -> np.ndarray:
    """Input: panel/name. Return object customdata (...,B) field view; local times seconds and identities unitless."""
    return np.asarray(panel["customdata"], dtype=object)[..., panel["hover_fields"].index(name)]


def _assert_oldest_row_is_at_top(interactive, static) -> None:
    """Input: Plotly/Matplotlib figures with (2,B) traces/images. Return None; verify initial oldest-first y orientation."""
    trace = next(trace for trace in interactive.data if trace.type == "heatmap")
    suffix = (trace.yaxis or "y")[1:]
    axis = interactive.layout["yaxis" + suffix]
    assert axis.autorange == "reversed" or (axis.range is not None and axis.range[0] > axis.range[1])
    for axis in static.axes:
        if axis.images:
            assert axis.get_ylim()[0] > axis.get_ylim()[1]


@pytest.mark.parametrize("kind", ["central", "transition"])
def test_two_displayed_rows_preserve_original_ids_values_and_oldest_first_orientation(kind: str) -> None:
    """Input: view kind. Return None; both renderers place the two chronological source motifs oldest-first.

    Central positions are [1,5]; transition episodes are [[0,1,2],[4,5,6]].
    Manual no-choice position 3 separates motifs without closing original IDs.
    Values remain Hz (M,B)/(E,S,B), and times remain local choice-relative seconds.
    """
    if kind == "central":
        trials = _trials([1, 1, 1, "no_choice", 1, 1, 1], [1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0])
        expected_positions = np.array([1, 5])
    else:
        trials = _trials(
            [1, 1, 0, "no_choice", 1, 1, 0], [1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 2, 3, 3], [1, 0, 0, 2, 1, 0, 0],
        )
        expected_positions = np.array([[0, 1, 2], [4, 5, 6]])
    packet = _packet(trials=trials)
    source = plotting.prepare_display_source(packet)
    view = plotting.build_central_view(packet, source) if kind == "central" else plotting.build_transition_view(packet, source)
    panel = view["panels"][0]
    assert panel["n"] == panel["behavioral_count"] == 2 and panel["excluded_count"] == 0
    np.testing.assert_array_equal(panel["row_positions"], expected_positions)
    source_indices = np.searchsorted(packet["activity_row_positions"], expected_positions)
    expected_values = packet["unit_rates_hz"][source_indices]
    np.testing.assert_array_equal(panel["values"], expected_values)
    if kind == "central":
        np.testing.assert_array_equal(_field_values(panel, "trial_id")[:, 0], [102, 106])
    else:
        np.testing.assert_array_equal(_field_values(panel, "trigger_trial_id")[:, 0, 0], [101, 105])
        np.testing.assert_array_equal(_field_values(panel, "segment_trial_id")[:, :, 0], [[101, 102, 103], [105, 106, 107]])
    interactive = plotting.plotly_figure(view)
    static = plotting.matplotlib_figure(view)
    traces = [trace for trace in interactive.data if trace.type == "heatmap"]
    images = [image for axis in static.axes for image in axis.images]
    if kind == "central":
        assert len(traces) == len(images) == 1
        np.testing.assert_array_equal(traces[0].z, expected_values)
        np.testing.assert_array_equal(images[0].get_array(), expected_values)
    else:
        assert len(traces) == len(images) == 3
        for segment, (trace, image) in enumerate(zip(traces, images)):
            np.testing.assert_array_equal(trace.z, expected_values[:, segment])
            np.testing.assert_array_equal(image.get_array(), expected_values[:, segment])
    _assert_oldest_row_is_at_top(interactive, static)


@pytest.mark.parametrize("central,side", [("correct_rewarded", "left"), ("omission", "right")])
def test_six_central_panels_keep_common_row_column_order_and_original_chronology(central: str, side: str) -> None:
    """Input: outcome/side labels. Return None; six assembled (M,B) panels retain original trial chronology."""
    packet = _packet()
    source = plotting.prepare_display_source(packet)
    view = plotting.build_central_view(packet, source, central_outcome=central, side=side)
    expected = [(central, side, previous, following) for previous in ("correct_rewarded", "omission", "incorrect") for following in ("repeat", "switch")]
    assert view["kind"] == "central" and len(view["panels"]) == 6
    assert [panel["condition"] for panel in view["panels"]] == expected
    assert [(panel["row"], panel["col"]) for panel in view["panels"]] == [(row, column) for row in range(3) for column in range(2)]
    for panel in view["panels"]:
        positions = packet["central_matches"][panel["condition"]]
        np.testing.assert_array_equal(panel["row_positions"], positions)
        assert panel["n"] == panel["behavioral_count"] == len(positions)
        assert panel["excluded_count"] == 0 and panel["values"].shape == (len(positions), 8)
        assert f"n={panel['n']}" in panel["title"]
        assert all(np.diff(panel["row_positions"]) > 0)
        for label, position in zip(panel["row_labels"], positions):
            assert str(packet["trials"].iloc[position].trial_id) in label


def test_central_hover_exposes_positions_ids_labels_utc_side_outcome_and_local_times() -> None:
    """No inputs; return None after checking named customdata (M,B,H) preserves independent source identities."""
    packet = _packet()
    view = plotting.build_central_view(packet, plotting.prepare_display_source(packet), central_outcome="correct_rewarded", side="left")
    panel = view["panels"][0]
    required = {"row_position", "trial_id", "index_label", "choice_time", "side", "outcome", "local_time_s"}
    assert required.issubset(panel["hover_fields"])
    positions = panel["row_positions"]
    for index, position in enumerate(positions):
        row = packet["trials"].iloc[position]
        assert _field_values(panel, "row_position")[index, 0] == position
        assert _field_values(panel, "trial_id")[index, 0] == row.trial_id
        assert _field_values(panel, "index_label")[index, 0] == row.index_label
        assert _field_values(panel, "choice_time")[index, 0] == row.choice_time
        assert _field_values(panel, "side")[index, 0] == "left"
        assert _field_values(panel, "outcome")[index, 0] == "correct_rewarded"
    np.testing.assert_allclose(_field_values(panel, "local_time_s").astype(float), np.broadcast_to((view["bin_edges_s"][:-1] + view["bin_edges_s"][1:]) / 2, (panel["n"], 8)))


@pytest.mark.parametrize("kind,zscore", [("unit", False), ("unit", True), ("pc", False), ("pc", True)])
def test_both_views_share_one_selected_source_session_reference_limits_and_exact_identities(kind: str, zscore: bool) -> None:
    """Input: kind/mode. Return None; global limits/scale remain fixed across both tools and all live motifs."""
    packet = _packet(kind)
    source = plotting.prepare_display_source(packet, pc_rank=1, zscore=zscore)
    reference = activity.scale_activity(io.get_source_activity(packet, pc_rank=1), zscore=zscore)
    views = [
        plotting.build_central_view(packet, source, central_outcome="correct_rewarded", side="left"),
        plotting.build_central_view(packet, source, central_outcome="omission", side="right"),
        plotting.build_transition_view(packet, source, direction="left_to_right", repeats=1),
        plotting.build_transition_view(packet, source, direction="right_to_left", repeats=4),
    ]
    for view in views:
        assert view["color_limits"] == reference["color_limits"]
        assert view["value_units"] == reference["value_units"]
        assert packet["meta"]["configuration_id"] in view["caption"]
        assert "A" in view["caption"] and "500" in view["caption"]
        assert reference["value_units"] in view["caption"]
        if kind == "pc":
            assert "PC1" in view["caption"] and "good_mua" in view["caption"]
            assert packet["meta"]["pca_reference"]["configuration_id"] in view["caption"]
            assert packet["meta"]["pca_reference"]["sha256"][:8] in view["caption"]
    if kind == "pc" or zscore:
        assert reference["values"].min() < 0 < reference["values"].max()
        assert reference["color_limits"][0] == -reference["color_limits"][1]


@pytest.mark.parametrize("width", [0.5, 0.1, 0.05, 0.02])
@pytest.mark.parametrize("repeats", [1, 4])
def test_transition_segments_keep_full_width_fixed_display_gaps_and_local_zero_lines(width: float, repeats: int) -> None:
    """Input: width seconds/x. Return None; logical (E,x+2,B) has no separator samples or fake elapsed time."""
    packet = _packet(width=width)
    source = plotting.prepare_display_source(packet)
    view = plotting.build_transition_view(packet, source, direction="left_to_right", repeats=repeats)
    assert view["kind"] == "transition" and len(view["panels"]) == 1
    panel = view["panels"][0]
    segments, bins = repeats + 2, round(4 / width)
    assert panel["values"].shape == (panel["n"], segments, bins)
    edges = np.asarray(view["segment_edges"])
    assert edges.shape == (segments, bins + 1)
    np.testing.assert_allclose(edges[:, -1] - edges[:, 0], 4)
    np.testing.assert_allclose(edges[1:, 0] - edges[:-1, -1], plotting.SEGMENT_GAP_WIDTH)
    assert plotting.SEGMENT_GAP_WIDTH == 0.25
    np.testing.assert_allclose(view["zero_positions"], edges[:, 0] + 2)
    assert "separator" in view["caption"].lower() and "local" in view["caption"].lower()
    figures = plotting.plotly_figure(view), plotting.matplotlib_figure(view)
    heatmaps = [trace for trace in figures[0].data if trace.type == "heatmap"]
    images = [image for axis in figures[1].axes for image in axis.images]
    assert len(heatmaps) == len(images) == segments
    for segment, (trace, image) in enumerate(zip(heatmaps, images)):
        np.testing.assert_array_equal(trace.z, panel["values"][:, segment])
        np.testing.assert_array_equal(image.get_array(), panel["values"][:, segment])
        np.testing.assert_allclose(trace.x, edges[segment])
        np.testing.assert_allclose(image.get_extent()[:2], edges[segment, [0, -1]])
    assert len(figures[0].layout.shapes) == segments
    np.testing.assert_allclose([shape.x0 for shape in figures[0].layout.shapes], view["zero_positions"])


def test_transition_hover_keeps_trigger_full_episode_and_segment_identity_with_local_times() -> None:
    """No inputs; return None after checking (E,S,B,H) customdata and physical/local-time labeling."""
    packet = _packet()
    view = plotting.build_transition_view(packet, plotting.prepare_display_source(packet), direction="right_to_left", repeats=2)
    panel = view["panels"][0]
    required = {"trigger_row_position", "trigger_trial_id", "episode_trial_ids", "direction", "segment_index", "segment_row_position", "segment_trial_id", "segment_index_label", "segment_choice_time", "local_time_s"}
    assert required.issubset(panel["hover_fields"])
    for episode, positions in enumerate(panel["row_positions"]):
        ids = packet["trials"].iloc[positions].trial_id.tolist()
        for segment, position in enumerate(positions):
            row = packet["trials"].iloc[position]
            assert _field_values(panel, "trigger_row_position")[episode, segment, 0] == positions[0]
            assert _field_values(panel, "trigger_trial_id")[episode, segment, 0] == ids[0]
            saved_ids = str(_field_values(panel, "episode_trial_ids")[episode, segment, 0])
            assert all(str(identity) in saved_ids for identity in ids)
            assert _field_values(panel, "direction")[episode, segment, 0] == "right_to_left"
            assert _field_values(panel, "segment_index")[episode, segment, 0] == segment
            assert _field_values(panel, "segment_row_position")[episode, segment, 0] == position
            assert _field_values(panel, "segment_trial_id")[episode, segment, 0] == row.trial_id
            assert _field_values(panel, "segment_index_label")[episode, segment, 0] == row.index_label
            assert _field_values(panel, "segment_choice_time")[episode, segment, 0] == row.choice_time
    centers = (view["bin_edges_s"][:-1] + view["bin_edges_s"][1:]) / 2
    np.testing.assert_allclose(_field_values(panel, "local_time_s").astype(float), np.broadcast_to(centers, panel["values"].shape))
    for trace in plotting.plotly_figure(view).data:
        if trace.type == "heatmap":
            assert "%{x" not in trace.hovertemplate
            assert "local" in trace.hovertemplate.lower()


@pytest.mark.parametrize("kind,zscore", [("unit", False), ("unit", True), ("pc", False), ("pc", True)])
def test_interactive_and_matplotlib_central_arrays_limits_and_light_style_agree(kind: str, zscore: bool) -> None:
    """Input: kind/mode. Return None; both renderers consume identical (M,B) values, shared limits and chronology."""
    packet = _packet(kind)
    source = plotting.prepare_display_source(packet, zscore=zscore)
    view = plotting.build_central_view(packet, source)
    interactive = plotting.plotly_figure(view)
    static = plotting.matplotlib_figure(view)
    panels = [panel for panel in view["panels"] if panel["n"]]
    traces = [trace for trace in interactive.data if trace.type == "heatmap"]
    images = [image for axis in static.axes for image in axis.images]
    assert len(traces) == len(images) == len(panels)
    assert len(list(interactive.select_xaxes())) == len(list(interactive.select_yaxes())) == 6
    assert len(interactive.layout.shapes) == 6
    assert all(shape.x0 == shape.x1 == 0 for shape in interactive.layout.shapes)
    for panel, trace, image in zip(panels, traces, images):
        np.testing.assert_array_equal(trace.z, panel["values"])
        np.testing.assert_array_equal(image.get_array(), panel["values"])
        assert image.get_clim() == view["color_limits"]
        assert trace.coloraxis == "coloraxis" and trace.zsmooth is False
    assert (interactive.layout.coloraxis.cmin, interactive.layout.coloraxis.cmax) == view["color_limits"]
    assert interactive.layout.coloraxis.colorbar.title.text == view["value_units"]
    assert interactive.layout.paper_bgcolor == interactive.layout.plot_bgcolor == "white"
    assert interactive.layout.font.color == "black" and interactive.layout.font.size >= 12
    assert to_rgba(static.get_facecolor()) == (1, 1, 1, 1)
    for axis in static.axes:
        assert to_rgba(axis.get_facecolor()) == (1, 1, 1, 1)
        assert to_rgba(axis.xaxis.label.get_color()) == (0, 0, 0, 1)


def test_n_one_and_requested_empty_panels_remain_visible_with_distinct_coverage_messages() -> None:
    """No inputs; return None after distinguishing one real row, five absent motifs and a removed covered row."""
    trials = _trials([1, 1, 1], [1, 1, 0], [1, 1, 0])
    packet = _packet(trials=trials)
    view = plotting.build_central_view(packet, plotting.prepare_display_source(packet))
    assert view["panels"][0]["values"].shape == (1, 8)
    for panel in view["panels"][1:]:
        assert panel["values"].shape == (0, 8) and panel["n"] == 0
        assert "no matching" in panel["message"].lower() and "n=0" in panel["message"]
    figure = plotting.plotly_figure(view)
    assert len(list(figure.select_xaxes())) == 6
    assert len([trace for trace in figure.data if trace.type == "heatmap"]) == 1
    assert sum("no matching" in annotation.text.lower() for annotation in figure.layout.annotations) >= 5
    missing = _packet(trials=trials, drop=(1,))
    panel = plotting.build_central_view(missing, plotting.prepare_display_source(missing))["panels"][0]
    assert panel["behavioral_count"] == 1 and panel["excluded_count"] == 1 and panel["n"] == 0
    assert "coverage" in panel["message"].lower() and "no matching" not in panel["message"].lower()
    assert "1" in panel["message"] and "n=0" in panel["message"]
    assert "behavioral" in panel["message"].lower() and "excluded" in panel["message"].lower()


def test_absent_transition_condition_remains_visible_with_no_matching_message() -> None:
    """No inputs; return None after a valid session with zero transitions retains its selected empty condition."""
    packet = _packet(trials=_trials([1, 1, 1], [1, 1, 0], [1, 1, 0]))
    view = plotting.build_transition_view(packet, plotting.prepare_display_source(packet))
    assert len(view["panels"]) == 1
    panel = view["panels"][0]
    assert panel["behavioral_count"] == panel["excluded_count"] == panel["n"] == 0
    assert panel["values"].shape == (0, 3, 8)
    assert "no matching" in panel["message"].lower() and "n=0" in panel["message"]
    assert len(list(plotting.plotly_figure(view).select_xaxes())) == 1


def test_no_eligible_source_windows_preserves_coverage_counts_and_reference_unavailable_reason() -> None:
    """No inputs; return None after zero-W source separates unavailable reference from one real coverage exclusion."""
    packet = _packet(trials=_trials([1, 1, 1], [1, 1, 0], [1, 1, 0]), drop=(0, 1, 2))
    view = plotting.build_central_view(packet, plotting.prepare_display_source(packet))
    panel = view["panels"][0]
    assert panel["values"] is None and panel["n"] == 0
    assert panel["behavioral_count"] == panel["excluded_count"] == 1
    assert "eligible" in panel["message"].lower() and "reference" in panel["message"].lower()


def test_missing_episode_segment_excludes_whole_rendered_sequence_and_keeps_condition() -> None:
    """No inputs; return None after whole-episode coverage exclusion leaves an explicit empty transition view."""
    trials = _trials([1, 1, 0], [1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0])
    packet = _packet(trials=trials, drop=(1,))
    view = plotting.build_transition_view(packet, plotting.prepare_display_source(packet), direction="left_to_right", repeats=1)
    panel = view["panels"][0]
    assert panel["condition"] == ("left_to_right", 1)
    assert panel["values"].shape == (0, 3, 8)
    assert panel["behavioral_count"] == panel["excluded_count"] == 1 and panel["n"] == 0
    assert "coverage" in panel["message"].lower()
    figure = plotting.plotly_figure(view)
    assert not any(trace.type == "heatmap" for trace in figure.data)


def test_constant_raw_activity_is_labeled_and_zero_sd_zscore_unavailable_does_not_change_coverage() -> None:
    """No inputs; return None after constant zeros remain visible raw and unavailable z-score is explained separately."""
    packet = _packet(trials=_trials([1, 1, 1], [1, 1, 0], [1, 1, 0]))
    packet["unit_rates_hz"][:] = 0
    raw = plotting.build_central_view(packet, plotting.prepare_display_source(packet))
    np.testing.assert_array_equal(raw["panels"][0]["values"], np.zeros((1, 8)))
    assert raw["color_limits"] == (0, 1) and raw["constant_activity"]
    assert "constant" in raw["caption"].lower()
    zscored = plotting.build_central_view(packet, plotting.prepare_display_source(packet, zscore=True))
    assert zscored["panels"][0]["values"] is None
    assert zscored["panels"][0]["excluded_count"] == 0
    assert "zero" in zscored["panels"][0]["message"].lower()
    assert not any(trace.type == "heatmap" for trace in plotting.plotly_figure(zscored).data)


def test_missing_part_two_context_reason_is_not_reported_as_no_matches() -> None:
    """No inputs; return None after saved transition availability reason remains explicit in the selected panel."""
    packet = _packet()
    packet["meta"]["transition_unavailable_reason"] = "Part II unavailable: missing state_int."
    packet["transition_matches"] = {key: np.empty((0, key[1] + 2), dtype=int) for key in TRANSITION_CONDITIONS}
    view = plotting.build_transition_view(packet, plotting.prepare_display_source(packet))
    assert "state_int" in view["panels"][0]["message"]
    assert "no matching" not in view["panels"][0]["message"].lower()


@pytest.mark.parametrize("mismatch", ["configuration_id", "bin_edges_s", "row_positions", "source_label"])
def test_view_rejects_display_source_bound_to_different_packet_identity_or_maps(mismatch: str) -> None:
    """Input: mismatch field. Return None; never attach another source's values to this packet's hover labels."""
    packet = _packet()
    source = deepcopy(plotting.prepare_display_source(packet))
    if mismatch == "configuration_id":
        source[mismatch] = "other-source-configuration"
    elif mismatch == "source_label":
        source[mismatch] = "B:99"
    else:
        source[mismatch][0] += 99
    with pytest.raises(ValueError, match="source|identity|configuration|mapping|bin"):
        plotting.build_central_view(packet, source)
    with pytest.raises(ValueError, match="source|identity|configuration|mapping|bin"):
        plotting.build_transition_view(packet, source)


def test_live_selectors_and_both_renderers_do_not_load_bin_fit_or_publish(monkeypatch) -> None:
    """Input: patch fixture. Return None; one selected-source preparation serves all motif/render switches immutably."""
    packet = _packet("pc")
    scores = packet["timeline_result"].regions["PFC"].scores.copy()
    reader = Mock(wraps=io.get_source_activity)
    monkeypatch.setattr(io, "get_source_activity", reader)
    monkeypatch.setattr(io, "load_probe_spike_arrays", _forbid_scientific_work)
    monkeypatch.setattr(io, "save_event_sequences", _forbid_scientific_work)
    monkeypatch.setattr(loading, "load_aligned_spikes", _forbid_scientific_work)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbid_scientific_work)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbid_scientific_work)
    monkeypatch.setattr(pd, "read_csv", _forbid_scientific_work)
    source = plotting.prepare_display_source(packet, pc_rank=2, zscore=True)
    central = plotting.build_central_view(packet, source, central_outcome="omission", side="right")
    transition = plotting.build_transition_view(packet, source, direction="right_to_left", repeats=3)
    for view in (central, transition):
        plotting.plotly_figure(view)
        plotting.matplotlib_figure(view)
    reader.assert_called_once()
    np.testing.assert_array_equal(packet["timeline_result"].regions["PFC"].scores, scores)


def test_explicit_png_export_is_opaque_and_refuses_overwrite(tmp_path: Path) -> None:
    """Input: temporary Path. Return None; explicit Agg export writes opaque labeled PNG without changing an existing file."""
    packet = _packet(trials=_trials([1, 1, 1], [1, 1, 0], [1, 1, 0]))
    view = plotting.build_central_view(packet, plotting.prepare_display_source(packet))
    path = tmp_path / "central.png"
    assert plotting.export_png(view, path, dpi=120) == path
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pixels = matplotlib.image.imread(path)
    assert pixels.shape[0] > 400 and pixels.shape[1] > 400
    if pixels.shape[2] == 4:
        assert np.all(pixels[:, :, 3] == 1)
    np.testing.assert_array_equal(pixels[0, 0, :3], [1, 1, 1])
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        plotting.export_png(view, path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", ["central", "transition"])
def test_partial_coverage_counts_do_not_obscure_real_activity(kind: str) -> None:
    """Input: view kind. Return None; report two matches/one exclusion outside the surviving activity bins."""
    if kind == "central":
        trials = _trials([1, 1, 1, "no_choice", 1, 1, 1], [1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 1, 1, 0])
        missing = 5
    else:
        trials = _trials(
            [1, 1, 0, "no_choice", 1, 1, 0], [1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 1, 1, 0, 1],
            [0, 1, 1, 1, 2, 3, 3], [1, 0, 0, 2, 1, 0, 0],
        )
        missing = 5
    packet = _packet(trials=trials, drop=(missing,))
    source = plotting.prepare_display_source(packet)
    view = plotting.build_central_view(packet, source) if kind == "central" else plotting.build_transition_view(packet, source)
    panel = view["panels"][0]
    assert (panel["behavioral_count"], panel["excluded_count"], panel["n"]) == (2, 1, 1)
    interactive = plotting.plotly_figure(view)
    static = plotting.matplotlib_figure(view)
    # Coverage counts belong in panel titles; empty-panel explanations may
    # remain inside their axes, but text must not cover surviving activity.
    assert "excluded" in static.axes[0].get_title().lower()
    assert not any("excluded" in text.get_text().lower() for text in static.axes[0].texts)
    title = next(annotation for annotation in interactive.layout.annotations if "n=1" in annotation.text and annotation.yref == "paper")
    assert "excluded" in title.text.lower()
    assert not any("excluded" in annotation.text.lower() and annotation.yref != "paper" for annotation in interactive.layout.annotations)
