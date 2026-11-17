"""Scientific rank-curve plots and incomplete paired support."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.communication_subspace.plotting import plot_rank_curves, save_rank_figure


def _plot_result() -> SimpleNamespace:
    """Return four directions of dimensionless draw/curve R2 with unequal support."""
    draw_rows = []
    for direction in ("PFC->PFC", "PFC->HPC", "HPC->HPC", "HPC->PFC"):
        for draw in range(3):
            draw_rows.append(dict(draw=draw, direction=direction, model="ridge", rank=None,
                                  mean_r2=0.1 * draw, n_folds=5, status="ok"))
            draw_rows.append(dict(draw=draw, direction=direction, model="rrr", rank=1,
                                  mean_r2=0.2 + 0.1 * draw, n_folds=5, status="ok"))
            supported = draw != 0 if direction == "PFC->HPC" else True
            draw_rows.append(dict(draw=draw, direction=direction, model="rrr", rank=2,
                                  mean_r2=0.4 + 0.1 * draw if supported else np.nan,
                                  n_folds=5 if supported else 4,
                                  status="ok" if supported else "incomplete"))
    draw_scores = pd.DataFrame(draw_rows)
    draw_scores["rank"] = draw_scores["rank"].astype("Int64")
    curve_rows = []
    for (direction, model, rank), group in draw_scores.groupby(
        ["direction", "model", "rank"], dropna=False, sort=False,
    ):
        complete = group.loc[group.status == "ok"]
        curve_rows.append(dict(direction=direction, model=model, rank=rank,
                               mean_r2=float(complete.mean_r2.mean()),
                               n_draws=len(complete),
                               draw_ids=tuple(complete.draw),
                               status="ok" if len(complete) == 3 else "incomplete"))
    curve_scores = pd.DataFrame(curve_rows)
    curve_scores["rank"] = curve_scores["rank"].astype("Int64")
    return SimpleNamespace(
        computation=SimpleNamespace(draw_scores=draw_scores, curve_scores=curve_scores,
                                    status="partial"),
        meta={"session_id": "synthetic-plot", "configuration": {"bin_width_ms": 500}},
    )


def test_two_light_panels_show_faint_draws_means_ridge_and_zero_reference():
    """No inputs; four directions and dimensionless R2 use two source panels."""
    figure = plot_rank_curves(_plot_result())
    try:
        assert len(figure.axes) == 2
        assert figure.get_facecolor() == (1., 1., 1., 1.)
        assert "PFC" in figure.axes[0].get_title()
        assert "HPC" in figure.axes[1].get_title()
        for axis in figure.axes:
            assert "R" in axis.get_ylabel()
            assert [int(tick) for tick in axis.get_xticks() if tick in (1, 2)] == [1, 2]
            assert any(np.allclose(line.get_ydata(), 0) for line in axis.lines)
            assert any(line.get_alpha() is not None and line.get_alpha() < 0.5
                       for line in axis.lines)
            labels = [line.get_label() for line in axis.lines]
            assert any("within" in label.lower() for label in labels)
            assert any("cross" in label.lower() for label in labels)
            assert any("ridge" in label.lower() for label in labels)
        caption = " ".join(text.get_text() for text in figure.texts).lower()
        for phrase in ("synthetic-plot", "500 ms", "raw activity", "five", "negative"):
            assert phrase in caption
    finally:
        plt.close(figure)


def test_incomplete_rank_marks_paired_rrr_and_baseline_on_same_draw_ids():
    """No inputs; rank-2 PFC cross markers use draws 1,2 for both predictions."""
    figure = plot_rank_curves(_plot_result())
    try:
        axis = figure.axes[0]
        paired_lines = [line for line in axis.lines if "paired" in line.get_label().lower()
                        and "cross" in line.get_label().lower() and "rank 2" in line.get_label().lower()]
        assert len(paired_lines) == 2
        assert all("2 draws" in line.get_label().lower() for line in paired_lines)
        paired_values = sorted(float(line.get_ydata()[0]) for line in paired_lines)
        np.testing.assert_allclose(paired_values, [0.15, 0.55])
        assert all(np.allclose(line.get_xdata(), 2) for line in paired_lines)
    finally:
        plt.close(figure)


def test_unavailable_result_renders_diagnostic_and_export_closes_figure(tmp_path):
    """Temporary output path; empty scores still yield a two-panel PNG and no open figure."""
    empty = _plot_result()
    empty.computation.draw_scores = empty.computation.draw_scores.iloc[:0]
    empty.computation.curve_scores = empty.computation.curve_scores.iloc[:0]
    empty.computation.status = "unavailable"
    figure = plot_rank_curves(empty)
    try:
        assert len(figure.axes) == 2
        assert "unavailable" in " ".join(text.get_text() for text in figure.texts).lower()
    finally:
        plt.close(figure)
    open_before = set(plt.get_fignums())
    path = tmp_path / "rank_curves.png"
    assert save_rank_figure(empty, path) == path
    assert path.stat().st_size > 1000
    assert set(plt.get_fignums()) == open_before
