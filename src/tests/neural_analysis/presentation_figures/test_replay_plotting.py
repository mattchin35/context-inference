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
            (motif, width, region, target, row, 20 + row, start, start + width / 1000,
             value + .1 * (index % 2), 1 - value - .1 * (index % 2))
            for width, value in [(500, .27), (100, .81)]
            for region in ["PFC", "HPC"]
            for target in ["context", "choice"]
            for row, choice in [(0, 1.3), (1, 4.1)]
            for index in range(2000 // width)
            for start in [choice - 1. + index * width / 1000]
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
    for width, expected in [(500, .32), (100, .86)]:
        assert tuple(by_width[width].x) == (1.3, 4.1)
        np.testing.assert_allclose(by_width[width].y, [expected, expected])
        assert by_width[width].mode == "markers"
    assert [button.label for button in fig.layout.updatemenus[0].buttons] == ["500 ms", "100 ms"]
    assert fig.layout.hovermode == "x"
    assert any(axis.showspikes for axis in fig.select_xaxes())
    html = fig.to_html(full_html=True, include_plotlyjs=True)
    assert "<script src=" not in html
    assert '"series":"PFC:context"' in html


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
        np.testing.assert_allclose(regional.get_ydata(), [.32, .32])
        np.testing.assert_allclose(regional.get_xdata(), [1.3, 4.1])
        static.canvas.draw()
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_outcome_replay_stacks_post_choice_means_and_labels_observed_choice(saved_replays):
    """Both renderers show one four-state composition per complete choice and region."""
    replay, _, meta = saved_replays
    for name in ("episodes", "trials", "events"):
        replay[name]["motif"] = "four_outcome"
    rows = []
    for width in (100, 500):
        for region in ("PFC", "HPC"):
            for trial in replay["trials"].itertuples():
                for index in range(1000 // width):
                    start = trial.choice_elapsed_s + index * width / 1000
                    values = [.2, .3, .4, .1] if index % 2 else [.6, .1, .1, .2]
                    rows.append(("four_outcome", width, region, "outcome", trial.row_position,
                                 trial.trial_id, start, start + width / 1000, *values))
    replay["tracks"] = pd.DataFrame(rows, columns=[*replay["tracks"].columns, "p_2", "p_3"])
    # Incomplete class predictions cannot produce a four-state composition.
    replay["tracks"].loc[0, "p_2"] = np.nan
    fig = interactive_replays(replay, {}, meta, "light")["figure4_four_outcome"]
    assert fig.layout.barmode == "stack"
    for trace in [trace for trace in fig.data if trace.meta["width_ms"] is not None]:
        assert trace.type == "bar"
        region, _, state = trace.meta["series"].split(":")
        incomplete = region == "PFC" and trace.meta["width_ms"] == 100
        assert list(trace.x) == ([4.1] if incomplete else [1.3, 4.1])
        np.testing.assert_allclose(trace.y, [.4, .2, .25, .15][int(state)])
    assert any("L-" in text for text in fig.layout.xaxis3.ticktext)
    assert any("R+" in text for text in fig.layout.xaxis3.ticktext)
    static = render_replays(replay, {}, meta, "light")["figure4_four_outcome"]
    try:
        for axis in static.axes[1:]:
            assert len(axis.patches) == 8
            heights = np.array([bar.get_height() for bar in axis.patches]).reshape(4, 2)
            np.testing.assert_allclose(heights.sum(axis=0), 1)
            terminal = axis.patches[-1]
            assert terminal.get_y() + terminal.get_height() == pytest.approx(1.)
        assert any("L-" in tick.get_text() for tick in static.axes[-1].get_xticklabels())
    finally:
        plt.close(static)


def test_movement_block_summary_shows_finite_dots_medians_and_support(saved_replays):
    """Figure 6A includes negative values and reports a metric's own denominator."""
    replay, movement, meta = saved_replays
    movement["block_scores"] = pd.DataFrame({
        "metric": ["pfc_hpc"] * 3 + ["pfc_speed"] * 3 + ["hpc_speed"] * 3,
        "block_id": [1, 2, 3] * 3,
        "correlation": [.2, .5, .8, -.8, .3, np.nan, np.nan, np.nan, np.nan],
        "reason": [""] * 5 + ["Missing support"] * 4,
    })
    figures = render_replays(replay, movement, meta, "light")
    try:
        summary = figures["figure6_block_correlations"]
        axis = summary.axes[0]
        assert [tick.get_text().split("\n")[-1] for tick in axis.get_xticklabels()] == [
            "3/3 blocks", "2/3 blocks", "0/3 blocks",
        ]
        medians = {line.get_gid(): line.get_ydata()[0] for line in axis.get_lines() if line.get_gid()}
        assert medians["pfc_hpc:median"] == pytest.approx(.5)
        assert medians["pfc_speed:median"] == pytest.approx(-.25)
        assert "hpc_speed:median" not in medians
        dots = np.concatenate([collection.get_offsets()[:, 1] for collection in axis.collections])
        np.testing.assert_allclose(np.sort(dots), [-.8, .2, .3, .5, .8])
    finally:
        for figure in figures.values():
            plt.close(figure)
