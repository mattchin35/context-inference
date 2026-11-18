"""Replay renderers preserve saved values, measured gaps and event times."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.presentation_figures.replay_plotting import (
    interactive_replays,
    render_replays,
)


@pytest.fixture
def saved_replays():
    """Return synthetic Figure 4/6 field dictionaries and a UTC origin/session identity."""
    motif = "new_side_omission"
    replay = {
        "episodes": pd.DataFrame([{
            "motif": motif, "status": "available", "reason": "",
            "start_elapsed_s": 1., "end_elapsed_s": 6.,
            "first_trial_id": 20, "last_trial_id": 21, "repeat_count": 0,
            "missing_deliveries": 0,
        }]),
        "tracks": pd.DataFrame.from_records([
            (motif, width, region, target, row, 20 + row, start, start + width / 1000, value, 1 - value)
            for width, value in [(500, .27), (100, .81)]
            for region in ["PFC", "HPC"]
            for target in ["context", "choice"]
            for row, start in [(0, 1.0), (1, 4.0)]
        ], columns=["motif", "width_ms", "region", "target", "row_position", "trial_id",
                    "bin_left_elapsed_s", "bin_right_elapsed_s", "p_0", "p_1"]),
        "events": pd.DataFrame.from_records([
            (motif, 1.1, "context", "left", 20),
            (motif, 1.3, "choice", "left", 20),
            (motif, 3.9, "switch", "right", 21),
            (motif, 3.9, "context", "right", 21),
            (motif, 4.1, "choice", "right", 21),
            (motif, 4.15, "delivery", "right", 21),
        ], columns=["motif", "elapsed_s", "event", "label", "trial_id"]),
        "trials": pd.DataFrame.from_records([
            (motif, 0, 20, 2, 1.3, 1, 1, 2, "omission", np.nan, False, True),
            (motif, 1, 21, 3, 4.1, 0, 0, 1, "correct rewarded", 4.15, False, True),
        ], columns=["motif", "row_position", "trial_id", "block_id", "choice_elapsed_s",
                    "context", "choice", "outcome", "trial_type", "delivery_elapsed_s",
                    "delivery_missing", "in_motif"]),
    }
    origin = 1_700_000_000.
    movement = {
        "replay_tracks": pd.DataFrame({
            "row_position": [2, 3], "trial_id": [22, 23], "block_id": [3, 3],
            "start_s": origin + np.array([8., 12.]), "stop_s": origin + np.array([8.1, 12.1]),
            "pfc_score": [-.55, 1.23], "hpc_score": [-.4, 1.5], "speed_mm_s": [13.2, 22.5],
        }),
        "replay_events": pd.DataFrame({
            "row_position": [2, 3], "trial_id": [22, 23], "event": ["choice", "reward_delivery"],
            "time_s": origin + np.array([8.05, 12.05]),
        }),
        "speed_units": "mm/s", "replay_reason": "", "replay_speed_increase_mm_s": 9.3,
    }
    return replay, movement, {"session_id": "synthetic replay", "time_origin_utc_s": origin}


def test_regional_width_selector_uses_saved_probabilities_and_gaps(saved_replays):
    """The normal app chart starts at 500 ms and switches to held-out 100-ms values."""
    replay, movement, meta = saved_replays
    fig = interactive_replays(replay, movement, meta, "light")["figure4_new_side_omission"]
    one_trace = [
        trace for trace in fig.data
        if isinstance(trace.meta, dict) and trace.meta.get("series") == "PFC:context"
    ]
    by_width = {trace.meta["width_ms"]: trace for trace in one_trace}
    assert set(by_width) == {100, 500}
    assert by_width[500].visible is True
    assert by_width[100].visible is False
    assert tuple(by_width[500].x) == (1., 1.5, None, 4., 4.5, None)
    assert tuple(by_width[500].y) == (.27, .27, None, .27, .27, None)
    assert tuple(by_width[100].x) == (1., 1.1, None, 4., 4.1, None)
    assert tuple(by_width[100].y) == (.81, .81, None, .81, .81, None)
    assert [button.label for button in fig.layout.updatemenus[0].buttons] == ["500 ms", "100 ms"]
    assert fig.layout.hovermode == "x"
    assert any(axis.showspikes for axis in fig.select_xaxes())
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    assert "<script src=" not in html
    assert '"series":"PFC:context"' in html
    assert '"y":[0.81,0.81,null,0.81,0.81,null]' in html


def test_matplotlib_and_interactive_movement_preserve_measured_values_and_events(
    saved_replays,
):
    """Static and live displays share elapsed axes, speed, neural traces and UTC markers."""
    replay, movement, meta = saved_replays
    fig = interactive_replays(replay, movement, meta, "light")["movement"]
    speed = next(trace for trace in fig.data if isinstance(trace.meta, dict)
                 and trace.meta.get("series") == "speed")
    np.testing.assert_allclose([value for value in speed.x if value is not None], [8., 8.1, 12., 12.1])
    assert tuple(speed.y) == (13.2, 13.2, None, 22.5, 22.5, None)
    mark = next(trace for trace in fig.data if isinstance(trace.meta, dict)
                and trace.meta.get("series") == "reward_delivery")
    np.testing.assert_allclose(mark.x, [12.05])

    figures = render_replays(replay, movement, meta, "light")
    assert set(figures) == {"figure4_new_side_omission", "movement"}
    try:
        static = figures["movement"]
        lines = [line for axis in static.axes for line in axis.get_lines()]
        speed_line = next(line for line in lines if line.get_gid() == "speed")
        np.testing.assert_allclose(speed_line.get_ydata(), [13.2, 13.2, np.nan, 22.5, 22.5, np.nan])
        regional_lines = [line for axis in figures["figure4_new_side_omission"].axes for line in axis.get_lines()]
        regional = next(line for line in regional_lines if line.get_gid() == "PFC:context")
        np.testing.assert_allclose(regional.get_ydata(), [.27, .27, np.nan, .27, .27, np.nan])
        static.canvas.draw()
    finally:
        for figure in figures.values():
            plt.close(figure)
