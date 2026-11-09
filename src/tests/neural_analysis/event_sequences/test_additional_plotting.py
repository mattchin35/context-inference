"""Family-aware displays preserve saved values, trial identities and local geometry."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import matplotlib.image
import matplotlib.text
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import activity, io, plotting, selection
from src.neural_analysis.population import session_timeline as timeline
from src.tests.neural_analysis.event_sequences.test_additional_io import _all_motifs
from src.tests.neural_analysis.event_sequences.test_additional_selection import (
    ADDITIONAL_KEYS, LONG_KEYS, _episode, _join_trials,
)
from src.tests.neural_analysis.event_sequences.test_plotting import (
    _field_values, _forbid_scientific_work, _packet,
)


def _plot_packet(
    key: tuple[str, str, int | None], *, width: float = 0.5,
    kind: str = "unit", drop: tuple[int, ...] = (), trials: pd.DataFrame | None = None,
) -> dict:
    """Return existing plotting fixture plus explicitly derived runtime fields.

    key is unitless family/side/x; width is seconds, kind unit/pc, drop original
    row positions, and optional trials are full chronological (N,C) raw rows.
    Normalize identities before _packet. Values are Hz or signed PC scores
    (W,B), with B=4/width; saved trial choices retain UTC seconds. No files load.
    """
    raw = _episode(key) if trials is None else trials.copy(deep=True)
    raw.index = [f"source-label-{index // 2}" for index in range(len(raw))]
    raw["start_time"] = raw["choice_time"] - 3
    normalized = selection.prepare_sequence_trials(raw)
    packet = _packet(kind, width, normalized, drop)
    packet.update(
        additional_matches=selection.select_additional_sequences(normalized),
        additional_unavailable_reason="",
        additional_selector_version=selection.ADDITIONAL_SELECTOR_VERSION,
    )
    return packet


def _build(packet: dict, source: dict, key: tuple[str, str, int | None]) -> dict:
    """Return family-aware common view from packet/source and unitless family/side/x.

    Values retain (E,S,B), positions (E,S) and source units. Attribute lookup
    makes the absent public builder a meaningful RED failure, not a collection error.
    """
    family, side, repeats = key
    return getattr(plotting, "build_sequence_view")(
        packet, source, family=family, initial_side=side, repeats=repeats,
    )


def _assert_renderer_values(view: dict) -> None:
    """Check Plotly/Matplotlib use identical (E,S,B) source values and global limits.

    Input is one populated common sequence view; return None. Display edges
    retain shifted local-second coordinates and do not add data bins for gaps.
    """
    panel = view["panels"][0]
    interactive = plotting.plotly_figure(view)
    static = plotting.matplotlib_figure(view)
    traces = [trace for trace in interactive.data if trace.type == "heatmap"]
    images = [image for axis in static.axes for image in axis.images]
    assert len(traces) == len(images) == panel["values"].shape[1]
    for segment, (trace, image) in enumerate(zip(traces, images)):
        np.testing.assert_array_equal(trace.z, panel["values"][:, segment])
        np.testing.assert_array_equal(image.get_array(), panel["values"][:, segment])
        np.testing.assert_allclose(trace.x, view["segment_edges"][segment])
        np.testing.assert_allclose(image.get_extent()[:2], view["segment_edges"][segment, [0, -1]])
        assert image.get_clim() == view["color_limits"]
        assert "%{x" not in trace.hovertemplate
        assert "local" in trace.hovertemplate.lower()
    assert (interactive.layout.coloraxis.cmin, interactive.layout.coloraxis.cmax) == view["color_limits"]
    assert interactive.layout.coloraxis.colorbar.title.text == view["value_units"]
    assert len(interactive.layout.shapes) == len(view["zero_positions"])
    np.testing.assert_allclose([shape.x0 for shape in interactive.layout.shapes], view["zero_positions"])
    assert interactive.layout.yaxis.range[0] > interactive.layout.yaxis.range[1]
    assert static.axes[0].get_ylim()[0] > static.axes[0].get_ylim()[1]
    assert interactive.layout.paper_bgcolor == interactive.layout.plot_bgcolor == "white"
    assert to_rgba(static.get_facecolor()) == (1, 1, 1, 1)


@pytest.mark.parametrize("key", ADDITIONAL_KEYS)
def test_all_conditions_keep_actual_values_full_episode_hover_and_correct_boundary(key):
    """Input: family/side/x key. Return None; verify actual (1,S,8) values/identities.

    All 24 conditions cover both orientations and every x. Probes carry choice
    paths without fabricated triggers; lucky's pre-boundary row is its omission.
    """
    packet = _plot_packet(key)
    source = plotting.prepare_display_source(packet)
    view = _build(packet, source, key)
    family, side, repeats = key
    panel = view["panels"][0]
    size = len(packet["trials"])
    assert view["kind"] == "transition" and len(view["panels"]) == 1
    assert panel["condition"] == key
    assert (panel["behavioral_count"], panel["excluded_count"], panel["n"]) == (1, 0, 1)
    np.testing.assert_array_equal(panel["row_positions"], [list(range(size))])
    np.testing.assert_array_equal(panel["values"], source["values"][None, :, :])
    assert panel["values"].shape == (1, size, 8)
    required = {
        "family", "initial_side", "repeats", "episode_start_row_position",
        "episode_start_trial_id", "episode_start_index_label", "episode_trial_ids",
        "segment_index", "segment_row_position", "segment_trial_id",
        "segment_index_label", "segment_choice_time", "local_time_s",
    }
    assert required.issubset(panel["hover_fields"])
    assert np.asarray(panel["customdata"]).shape == (1, size, 8, len(panel["hover_fields"]))
    ids = packet["trials"].trial_id.tolist()
    for segment in range(size):
        row = packet["trials"].iloc[segment]
        expected = {
            "family": family, "initial_side": side, "repeats": repeats,
            "episode_start_row_position": 0, "episode_start_trial_id": ids[0],
            "episode_start_index_label": "source-label-0", "segment_index": segment,
            "segment_row_position": segment, "segment_trial_id": row.trial_id,
            "segment_index_label": row.index_label, "segment_choice_time": row.choice_time,
        }
        for name, value in expected.items():
            assert _field_values(panel, name)[0, segment, 0] == value
        full_ids = str(_field_values(panel, "episode_trial_ids")[0, segment, 0])
        assert all(str(identity) in full_ids for identity in ids)
    centers = (source["bin_edges_s"][:-1] + source["bin_edges_s"][1:]) / 2
    np.testing.assert_allclose(
        _field_values(panel, "local_time_s").astype(float), np.broadcast_to(centers, (1, size, 8)),
    )
    path = "L -> R -> L" if side == "left" else "R -> L -> R"
    if family in ("probe", "omission_probe"):
        assert _field_values(panel, "choice_path")[0, 0, 0] == path
        assert path in view["caption"]
        assert not any("trigger" in field for field in panel["hover_fields"])
        assert not any("trigger" in label.lower() for label in panel["row_labels"] + view["segment_labels"])
        assert "episode" in panel["row_labels"][0].lower()
        for name in panel["hover_fields"]:
            if name.startswith("pre_boundary"):
                assert all(value is None for value in _field_values(panel, name).flat)
    else:
        direction = "left_to_right" if side == "left" else "right_to_left"
        assert _field_values(panel, "direction")[0, 0, 0] == direction
        boundary = 1 if family == "lucky_omission_switch" else 0
        required_boundary = {
            "pre_boundary_segment_index", "pre_boundary_row_position",
            "pre_boundary_trial_id", "pre_boundary_index_label",
        }
        assert required_boundary.issubset(panel["hover_fields"])
        assert _field_values(panel, "pre_boundary_segment_index")[0, 0, 0] == boundary
        assert _field_values(panel, "pre_boundary_row_position")[0, 0, 0] == boundary
        assert _field_values(panel, "pre_boundary_trial_id")[0, 0, 0] == ids[boundary]
        assert _field_values(panel, "pre_boundary_index_label")[0, 0, 0] == packet["trials"].iloc[boundary].index_label
        if family == "lucky_omission_switch":
            assert "omission" in view["segment_labels"][1].lower()
            assert "switch" in view["segment_labels"][2].lower()
            assert "trigger" not in view["segment_labels"][0].lower()
    if family == "omission_probe":
        assert "omission" in view["segment_labels"][1].lower()
    assert str(ids[0]) in panel["row_labels"][0]
    assert family.replace("_", " ") in view["caption"].lower()
    assert packet["additional_selector_version"] in view["caption"]
    assert (f"x{repeats}" in view["caption"]) if repeats is not None else ("xNone" not in view["caption"])


@pytest.mark.parametrize("key", [("omission_probe", "left", 4), ("indecisive_transition", "right", 3)])
@pytest.mark.parametrize("width", [0.5, 0.1, 0.05, 0.02])
def test_seven_segments_keep_full_windows_at_every_supported_bin_width(key, width):
    """Input: seven-segment key/width seconds. Return None; geometry is (7,B+1), B=4/width."""
    packet = _plot_packet(key, width=width)
    source = plotting.prepare_display_source(packet)
    view = _build(packet, source, key)
    bins = round(4 / width)
    edges = np.asarray(view["segment_edges"])
    assert view["panels"][0]["values"].shape == (1, 7, bins)
    assert len(view["segment_labels"]) == 7
    assert edges.shape == (7, bins + 1)
    np.testing.assert_allclose(edges[:, -1] - edges[:, 0], 4)
    np.testing.assert_allclose(edges[1:, 0] - edges[:-1, -1], 0.25)
    np.testing.assert_allclose(view["zero_positions"], np.arange(7) * 4.25)
    np.testing.assert_allclose(edges - np.asarray(view["zero_positions"])[:, None], np.broadcast_to(source["bin_edges_s"], (7, bins + 1)))
    assert "local" in view["caption"].lower() and "separator" in view["caption"].lower()


@pytest.mark.parametrize("side,repeats", [("left", 1), ("left", 4), ("right", 1), ("right", 4)])
def test_unified_ideal_view_preserves_every_legacy_builder_field(side, repeats):
    """Input: initial side/x. Return None; ideal wrapper agrees with unchanged legacy result."""
    packet = _packet()
    source = plotting.prepare_display_source(packet)
    direction = "left_to_right" if side == "left" else "right_to_left"
    legacy = plotting.build_transition_view(packet, source, direction=direction, repeats=repeats)
    unified = _build(packet, source, ("ideal_transition", side, repeats))
    for name, expected in legacy.items():
        if name == "panels":
            for field, value in expected[0].items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(unified[name][0][field], value)
                else:
                    assert unified[name][0][field] == value
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(unified[name], expected)
        else:
            assert unified[name] == expected


def test_default_unified_builder_selects_the_existing_default_ideal_condition():
    """No inputs; return None; the default still selects left-to-right x1 (E,3,B)."""
    packet = _packet()
    source = plotting.prepare_display_source(packet)
    view = getattr(plotting, "build_sequence_view")(packet, source)
    assert view["panels"][0]["condition"] == ("left_to_right", 1)
    np.testing.assert_array_equal(view["panels"][0]["values"], plotting.build_transition_view(packet, source)["panels"][0]["values"])


@pytest.mark.parametrize("key", LONG_KEYS)
def test_available_empty_condition_retains_full_shape_and_explicit_no_match_message(key):
    """Input: populated left key. Return None; opposite condition retains (0,S,8) data/geometry."""
    packet = _plot_packet(key)
    source = plotting.prepare_display_source(packet)
    empty_key = (key[0], "right", key[2])
    view = _build(packet, source, empty_key)
    panel = view["panels"][0]
    assert panel["condition"] == empty_key
    assert (panel["behavioral_count"], panel["excluded_count"], panel["n"]) == (0, 0, 0)
    assert panel["values"].shape == (0, len(packet["trials"]), 8)
    assert "no matching" in panel["message"].lower() and "n=0" in panel["message"]
    assert not any(trace.type == "heatmap" for trace in plotting.plotly_figure(view).data)


@pytest.mark.parametrize("key", LONG_KEYS)
def test_missing_context_reason_is_distinct_from_an_available_empty_condition(key):
    """Input: key. Return None; named absent fields explain the same shaped empty panel."""
    packet = _plot_packet(key)
    packet["trials"] = packet["trials"].drop(columns=["cur_block", "state_int"])
    packet["additional_matches"] = {condition: rows[:0] for condition, rows in packet["additional_matches"].items()}
    packet["additional_unavailable_reason"] = "Additional sequences unavailable; missing cur_block, state_int."
    view = _build(packet, plotting.prepare_display_source(packet), key)
    panel = view["panels"][0]
    assert panel["values"].shape == (0, len(packet["trials"]), 8)
    assert "cur_block" in panel["message"] and "state_int" in panel["message"]
    assert "no matching" not in panel["message"].lower()


@pytest.mark.parametrize("key", LONG_KEYS)
def test_missing_coverage_on_any_segment_excludes_the_whole_episode(key):
    """Input: key. Return None; each missing segment yields one exclusion and (0,S,8)."""
    size = len(_episode(key))
    for segment in range(size):
        packet = _plot_packet(key, drop=(segment,))
        view = _build(packet, plotting.prepare_display_source(packet), key)
        panel = view["panels"][0]
        assert (panel["behavioral_count"], panel["excluded_count"], panel["n"]) == (1, 1, 0)
        assert panel["values"].shape == (0, size, 8)
        assert panel["row_positions"].shape == (0, size)
        assert "coverage" in panel["message"].lower() and "no matching" not in panel["message"].lower()


def test_covered_ideal_prefix_remains_visible_when_later_indecisive_segment_is_uncovered():
    """No inputs; return None; ideal (1,5,8) survives exclusion of full (0,7,8) episode."""
    key = ("indecisive_transition", "left", 3)
    packet = _plot_packet(key, drop=(6,))
    source = plotting.prepare_display_source(packet)
    indecisive = _build(packet, source, key)["panels"][0]
    ideal = _build(packet, source, ("ideal_transition", "left", 3))["panels"][0]
    assert (indecisive["behavioral_count"], indecisive["excluded_count"], indecisive["n"]) == (1, 1, 0)
    assert (ideal["behavioral_count"], ideal["excluded_count"], ideal["n"]) == (1, 0, 1)
    np.testing.assert_array_equal(ideal["row_positions"], [[0, 1, 2, 3, 4]])
    np.testing.assert_array_equal(ideal["values"], source["values"][None, :5])


def test_indecisive_caption_identifies_x_as_only_the_initial_incorrect_run():
    """No inputs; return None; x3 counts initial errors, with one later error separate."""
    key = ("indecisive_transition", "left", 3)
    packet = _plot_packet(key)
    view = _build(packet, plotting.prepare_display_source(packet), key)
    assert "x3 initial incorrect trials" in view["panels"][0]["title"]
    assert "x3 initial incorrect trials" in view["caption"]


@pytest.mark.parametrize("kind,zscore", [("unit", False), ("unit", True), ("pc", False), ("pc", True)])
def test_all_families_and_renderers_share_one_immutable_selected_source_pool(kind, zscore, monkeypatch):
    """Input: source/scaling/patch fixtures. Return None; family changes reuse (W,B) values.

    Real seeded fixed PCA precedes patches; all builders/renderers then forbid
    source reads, refitting, behavioral reselection and publication.
    """
    trials, _ = _all_motifs()
    trials = _join_trials([trials])
    packet = _plot_packet(("probe", "left", 1), kind=kind, trials=trials)
    reader = Mock(wraps=io.get_source_activity)
    monkeypatch.setattr(io, "get_source_activity", reader)
    source = plotting.prepare_display_source(packet, pc_rank=2, zscore=zscore)
    before_values = source["values"].copy()
    before_rows = source["row_positions"].copy()
    before_scores = None if kind == "unit" else packet["timeline_result"].regions["PFC"].scores.copy()
    monkeypatch.setattr(io, "load_probe_spike_arrays", _forbid_scientific_work)
    monkeypatch.setattr(io, "load_event_sequences", _forbid_scientific_work)
    monkeypatch.setattr(io, "save_event_sequences", _forbid_scientific_work)
    monkeypatch.setattr(timeline, "bin_choice_counts", _forbid_scientific_work)
    monkeypatch.setattr(timeline, "fit_timeline_activity", _forbid_scientific_work)
    monkeypatch.setattr(selection, "select_additional_sequences", _forbid_scientific_work)
    monkeypatch.setattr(pd, "read_csv", _forbid_scientific_work)
    central = plotting.build_central_view(packet, source)
    ideal = _build(packet, source, ("ideal_transition", "left", 1))
    assert central["color_limits"] == ideal["color_limits"] == source["color_limits"]
    for key in LONG_KEYS:
        view = _build(packet, source, key)
        assert view["color_limits"] == source["color_limits"]
        assert view["value_units"] == source["value_units"]
        assert view["configuration_id"] == source["configuration_id"]
        assert source["configuration_id"] in view["caption"]
        assert source["value_units"] in view["caption"]
        _assert_renderer_values(view)
    reader.assert_called_once()
    np.testing.assert_array_equal(source["values"], before_values)
    np.testing.assert_array_equal(source["row_positions"], before_rows)
    if kind == "pc":
        np.testing.assert_array_equal(packet["timeline_result"].regions["PFC"].scores, before_scores)


@pytest.mark.parametrize("drop", [(), (9,)])
def test_two_episodes_preserve_chronology_and_partial_coverage_counts(drop):
    """Input: missing positions tuple. Return None; (2,7,B) or (1,7,B) keeps oldest row first."""
    key = ("omission_probe", "left", 4)
    episode = _episode(key)
    blocker = episode.iloc[[-1]].astype(object)
    blocker["action"] = "no_choice"
    blocker["experimenter_reward_given"] = 1
    trials = _join_trials([episode, blocker, episode])
    packet = _plot_packet(key, trials=trials, drop=drop)
    source = plotting.prepare_display_source(packet)
    view = _build(packet, source, key)
    panel = view["panels"][0]
    expected = [list(range(7))] if drop else [list(range(7)), list(range(8, 15))]
    np.testing.assert_array_equal(panel["row_positions"], expected)
    expected_values = source["values"][np.searchsorted(source["row_positions"], expected)]
    np.testing.assert_array_equal(panel["values"], expected_values)
    assert (panel["behavioral_count"], panel["excluded_count"], panel["n"]) == (2, int(bool(drop)), len(expected))
    assert str(packet["trials"].iloc[0].trial_id) in panel["row_labels"][0]
    if drop:
        assert "excluded" in panel["title"].lower()
    _assert_renderer_values(view)


@pytest.mark.parametrize("key", [("omission_probe", "left", 4), ("indecisive_transition", "right", 3)])
def test_seven_segment_png_text_fits_canvas_with_readable_opaque_light_style(key, tmp_path: Path, monkeypatch):
    """Input: key/temp Path/patch fixtures. Return None; actual PNG text fits physical canvas.

    Use PC z-scores and long source identities. Caption must clear the time
    label; segment headings must clear each other and the condition title.
    """
    packet = _plot_packet(key, kind="pc", width=0.02)
    packet["meta"]["session_id"] = "CT026_2026-08-28_134630"
    view = _build(packet, plotting.prepare_display_source(packet, zscore=True), key)
    figure = plotting.matplotlib_figure(view)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = figure.bbox
    texts = [text for text in figure.findobj(matplotlib.text.Text) if text.get_visible() and text.get_text().strip()]
    assert texts
    for text in texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.x1 <= bounds.width + 1, text.get_text()
        assert box.y0 >= -1 and box.y1 <= bounds.height + 1, text.get_text()
        assert text.get_fontsize() >= 12, text.get_text()
        assert to_rgba(text.get_color()) == (0, 0, 0, 1)
    caption = figure.texts[-1]
    assert packet["additional_selector_version"] in caption.get_text()
    assert key[0].replace("_", " ") in caption.get_text().lower()
    caption_box = caption.get_window_extent(renderer)
    time_label_box = figure.axes[0].xaxis.label.get_window_extent(renderer)
    assert caption_box.y1 < time_label_box.y0
    headings = figure.axes[0].texts
    assert len(headings) == 7
    heading_boxes = [heading.get_window_extent(renderer) for heading in headings]
    title_box = figure.axes[0].title.get_window_extent(renderer)
    assert all(not box.overlaps(title_box) for box in heading_boxes)
    assert all(first.x1 < second.x0 for first, second in zip(heading_boxes, heading_boxes[1:]))
    assert to_rgba(figure.get_facecolor()) == (1, 1, 1, 1)
    assert all(to_rgba(axis.get_facecolor()) == (1, 1, 1, 1) for axis in figure.axes)
    monkeypatch.setattr(plotting, "matplotlib_figure", Mock(return_value=figure))
    path = tmp_path / f"{key[0]}_seven_segments.png"
    assert plotting.export_png(view, path, dpi=100) == path
    pixels = matplotlib.image.imread(path)
    assert pixels.shape[0] >= 600 and pixels.shape[1] >= 1800
    np.testing.assert_array_equal(pixels[0, 0, :3], [1, 1, 1])
    if pixels.shape[2] == 4:
        assert np.all(pixels[..., 3] == 1)


@pytest.mark.parametrize("settings", [
    {"family": "unknown"}, {"initial_side": "both"},
    {"family": "lucky_omission_switch", "repeats": 1},
    {"family": "probe", "repeats": 0}, {"family": "probe", "repeats": None},
    {"family": "indecisive_transition", "repeats": 4},
    {"family": "ideal_transition", "repeats": True},
    {"family": "probe", "repeats": True}, {"family": "probe", "repeats": 1.5},
])
def test_family_specific_selector_validation_rejects_ambiguous_or_impossible_conditions(settings):
    """Input: selector-override dict. Return None; reject invalid family/side/unitless x."""
    packet = _plot_packet(("probe", "left", 1))
    with pytest.raises(ValueError, match="family|side|repeats|length"):
        getattr(plotting, "build_sequence_view")(packet, plotting.prepare_display_source(packet), **settings)


@pytest.mark.parametrize("mismatch", ["configuration_id", "row_positions"])
def test_new_sequence_builder_rejects_source_bound_to_other_identity_or_rows(mismatch):
    """Input: source-field string. Return None; never combine foreign values and trial identities."""
    key = ("probe", "left", 1)
    packet = _plot_packet(key)
    source = deepcopy(plotting.prepare_display_source(packet))
    if mismatch == "configuration_id":
        source[mismatch] = "another-source"
    else:
        source[mismatch][0] = 999
    with pytest.raises(ValueError, match="source|identity|configuration|mapping"):
        _build(packet, source, key)
