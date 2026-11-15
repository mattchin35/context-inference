"""Native-bin activity, zero semantics and exact non-occluding overlap geometry."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.neural_analysis.decoder_unit_activity.plotting import (
    build_unit_activity_data, plot_unit_activity, export_unit_activity_png,
)
from src.neural_analysis.population.session_timeline import SessionTimelineResult


@pytest.fixture
def activity_example():
    """Return rates(3,2,2) Hz, chronological choices and unit assignments with two overlap lanes."""
    rates = np.array([[[0, 5], [2, 5]], [[4, 5], [6, 5]], [[8, 5], [10, 5]]], float)
    choices = pd.DataFrame({"trial_id": [10, 20, 90], "row_position": [0, 2, 7],
        "choice_time": [102., 103., 108.], "elapsed_s": [2., 3., 8.],
        "action": [1, 0, 1], "reward": [1, 0, 1]})
    behavior = SessionTimelineResult("example", 100., choices,
        pd.DataFrame(columns=["Time", "Event", "elapsed_s"]), {}, {}, {})
    display = pd.DataFrame({"unit_id": ["u0", "u1"], "group_index": [0, 2],
        "mean_weight": [2., 1.], "percentile": [.9, .8], "positive_folds": [5, 4], "display_order": [0, 1]})
    return rates, choices, behavior, display


def make_data(example, scale="hz", time_range_s=None):
    """Input fixture/scale/range; return plot data from unchanged (3,2,2) Hz values."""
    rates, choices, _, _ = example
    return build_unit_activity_data(rates, ("u0", "u1"), ("u0", "u1"), choices,
        np.arange(3), np.array([-1., 0, 1]), scale=scale, time_range_s=time_range_s)


def test_zero_and_constant_activity_have_scale_specific_meaning(activity_example):
    """Input Hz rates; zero/constant values remain measured and normalize correctly only in SD mode."""
    raw = make_data(activity_example)
    z = make_data(activity_example, "zscore")
    np.testing.assert_equal(raw["values"], activity_example[0])
    assert raw["values"][0, 0, 0] == 0 and (raw["values"][..., 1] == 5).all()
    assert z["values"][0, 0, 0] == pytest.approx(-5 / np.std([0, 2, 4, 6, 8, 10]))
    assert (z["values"][..., 1] == 0).all() and np.isfinite(z["values"]).all()
    assert z["constant_mask"].tolist() == [False, True]
    np.testing.assert_equal(activity_example[0][..., 1], np.full((3, 2), 5.))


def test_two_units_two_lanes_have_explicit_disjoint_rows(activity_example):
    """Input overlapping windows; lane padding, y edges, native time edges and trial IDs are exact."""
    data = make_data(activity_example)
    assert data["lane_ids"].tolist() == [0, 1, 0] and data["lane_count"] == 2
    first, second = data["lanes"]
    np.testing.assert_equal(first["y_edges"], np.arange(5))
    assert first["z"].shape == (4, 5) and second["z"].shape == (4, 2)
    np.testing.assert_equal(first["x"], [1, 2, 3, 7, 8, 9])
    np.testing.assert_equal(second["x"], [2, 3, 4])
    assert np.isnan(first["z"][[1, 3]]).all() and np.isnan(second["z"][[0, 2]]).all()
    assert np.isnan(first["z"][:, 2]).all()
    np.testing.assert_equal(first["z"][0, [0, 1, 3, 4]], [0, 2, 8, 10])
    np.testing.assert_equal(second["z"][1], [4, 6])
    assert first["customdata"][0, 0, 0] == 10 and second["customdata"][1, 0, 0] == 20
    assert first["customdata"][1, 0, 0] is None


def test_panning_keeps_global_lanes_statistics_and_color_limits(activity_example):
    """Input two viewports; no lane reassignment, local z-scoring or automatic recoloring."""
    full = make_data(activity_example, "zscore")
    tail = make_data(activity_example, "zscore", (7, 9))
    assert tail["lane_count"] == 2 and len(tail["lanes"]) == 1
    np.testing.assert_equal(tail["lane_ids"], full["lane_ids"])
    np.testing.assert_equal(tail["values"], full["values"])
    assert tail["color_limits"] == full["color_limits"]
    assert tail["bin_width_s"] == 1


def test_touching_windows_need_one_lane_and_ids_are_not_indices(activity_example):
    """Input adjacent half-open intervals with irregular IDs; they do not overlap or reorder."""
    rates, choices, _, _ = activity_example
    choices = choices.assign(choice_time=[102., 104., 106.], elapsed_s=[2., 4., 6.])
    data = build_unit_activity_data(rates, ("u0", "u1"), ("u1",), choices,
        np.arange(3), np.array([-1., 0., 1.]), scale="hz")
    assert data["lane_count"] == 1 and data["lanes"][0]["z"].shape == (1, 6)
    assert data["unit_ids"] == ("u1",) and data["lanes"][0]["customdata"][0, -1, 0] == 90


def test_plot_and_png_share_exact_edges_and_rows(activity_example, tmp_path):
    """Input prepared plot data; both renderers preserve one label per unit and non-overwrite export."""
    data = make_data(activity_example)
    _, _, behavior, display = activity_example
    figure = plot_unit_activity(data, behavior, display)
    heatmaps = [trace for trace in figure.data if trace.type == "heatmap"]
    assert len(heatmaps) == 2 and all(trace.hoverongaps is False for trace in heatmaps)
    np.testing.assert_equal(heatmaps[0].z, data["lanes"][0]["z"])
    np.testing.assert_equal(heatmaps[0].y, np.arange(5))
    assert tuple(figure.layout.yaxis2.range) == (4, 0)
    assert list(figure.layout.yaxis2.ticktext) == ["u0", "u1"]
    assert "1000 ms" in figure.layout.meta["caption"]
    path = tmp_path / "units.png"
    export_unit_activity_png(path, data, behavior, display)
    with Image.open(path) as exported:
        assert exported.getpixel((0, 0))[:3] == (255, 255, 255)
    with pytest.raises(FileExistsError):
        export_unit_activity_png(path, data, behavior, display)


def test_empty_viewport_and_invalid_unit_selection(activity_example):
    """Input viewport without windows or unknown selected ID; no fabricated activity or ID fallback."""
    assert not make_data(activity_example, time_range_s=(20, 30))["lanes"]
    rates, choices, _, _ = activity_example
    with pytest.raises(ValueError):
        build_unit_activity_data(rates, ("u0", "u1"), ("missing",), choices,
            np.arange(3), np.array([-1., 0., 1.]))


def test_logged_context_and_led_tracks_export(activity_example, tmp_path):
    """Input recorded context/LED events; both renderers accept the public interval schemas."""
    _, _, behavior, display = activity_example
    behavior.events = pd.DataFrame({"Event": ["enter_left_patch", "LED_on", "left_entry", "LED_off", "enter_right_patch"],
        "elapsed_s": [0., 1., 2., 3., 5.], "Time": [100., 101., 102., 103., 105.]})
    data = make_data(activity_example)
    figure = plot_unit_activity(data, behavior, display, ("LED", "left_entry"))
    assert len([trace for trace in figure.data if trace.type == "heatmap"]) == 2
    export_unit_activity_png(tmp_path / "events.png", data, behavior, display, ("LED", "left_entry"))
