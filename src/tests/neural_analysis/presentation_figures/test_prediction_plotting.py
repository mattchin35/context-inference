"""Headless rendering tests for saved population prediction curves."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prediction_curves():
    """Return test curves with dimensionless R2 and three-of-five rank support."""
    rows = []
    for source, target in (("PFC", "PFC"), ("HPC", "PFC"), ("HPC", "HPC"), ("PFC", "HPC")):
        for panel in "ABCD":
            models = {"A": ["ridge"], "B": ["ridge"], "C": ["rrr", "ridge_units"],
                      "D": ["rrr", "ridge_units", "ridge_10pc"]}[panel]
            for model in models:
                for count in (1, 2):
                    score = count * 0.1 if model == "rrr" else count * 0.12
                    support = 3 if panel == "D" and count == 2 else 5
                    rows.append(dict(panel=panel, count=count, direction=f"{source}->{target}",
                                     source_region=source, target_region=target, model=model,
                                     mean_r2=score, min_r2=score - 0.1, max_r2=score + 0.1,
                                     n_draws=support, source_pool_size=12, status="ok", reason=""))
    return pd.DataFrame(rows)


def test_figure5_renders_matched_references_ranges_and_support():
    """Render all branches by target, using rank-specific references and draw counts."""
    from src.neural_analysis.presentation_figures.prediction_plotting import render_prediction

    fields = {
        "curves": prediction_curves(), "row_positions": np.arange(18), "source_pool_size": 12,
        "available_units": pd.DataFrame({"region": ["PFC", "HPC"], "n_units": [24, 30]}),
    }
    figure = render_prediction(fields, {"session_id": "synthetic-session"}, "light")["figure5"]
    try:
        figure.canvas.draw()
        rank = next(axis for axis in figure.axes if axis.get_gid() == "D")
        assert "ten source PCs" in rank.get_title()
        for model, expected in (("rrr", [0.1, 0.2]), ("ridge_units", [0.12, 0.24]),
                                ("ridge_10pc", [0.12, 0.24])):
            line = next(line for line in rank.lines if line.get_gid() == f"D:PFC->PFC:{model}")
            np.testing.assert_allclose(line.get_ydata(), expected)
        assert rank.collections
        support = next(axis for axis in figure.axes if axis.get_gid() == "D:support")
        assert sorted(tuple(line.get_ydata()) for line in support.lines) == [(5, 3), (5, 3)]
        caption = " ".join(element.get_text() for element in figure.texts)
        assert all(value in caption for value in ("synthetic-session", "18", "100 ms", "24", "30", "12"))
    finally:
        plt.close(figure)


def test_figure8_draws_partial_session_curves_without_partial_roster_median():
    """Use all supported session scores and show medians only at full-roster counts."""
    from src.neural_analysis.presentation_figures.prediction_plotting import render_prediction_cohort

    curves = prediction_curves()
    sessions = ["day 1", "day 2", "day 3", "day 4"]
    by_session = []
    for index, session in enumerate(sessions):
        rows = curves.assign(session_id=session)
        if index == 3:
            rows.loc[rows.panel.eq("D") & rows["count"].eq(2), "mean_r2"] = np.nan
        by_session.append(rows)
    support = curves[["panel", "count", "direction", "model", "mean_r2"]].copy()
    support["roster_size"] = 4
    support["n_sessions"] = 4
    support.loc[support.panel.eq("D") & support["count"].eq(2), "n_sessions"] = 3
    fields = {"curves": pd.concat(by_session), "prediction_medians": support,
              "roster": pd.DataFrame({"session_id": sessions, "status": ["ok"] * 4,
                                      "reason": [""] * 4})}
    figures = render_prediction_cohort(fields, {"mouse_id": "synthetic-mouse"}, "light")
    try:
        assert set(figures) == {"figure8_counts", "figure8_ranks"}
        ranks = figures["figure8_ranks"]
        ranks.canvas.draw()
        panel = next(axis for axis in ranks.axes if axis.get_gid() == "D:PFC->PFC")
        lines = {line.get_gid(): line for line in panel.lines}
        np.testing.assert_allclose(lines["day 1:rrr"].get_ydata(), [0.1, 0.2])
        np.testing.assert_allclose(lines["day 4:rrr"].get_ydata(), [0.1, np.nan], equal_nan=True)
        np.testing.assert_allclose(lines["median:rrr"].get_ydata(), [0.1, np.nan], equal_nan=True)
        support_axis = next(axis for axis in ranks.axes if axis.get_gid() == "D:PFC->PFC:support")
        np.testing.assert_array_equal(support_axis.lines[0].get_ydata(), [4, 3])
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_figure8_keeps_roster_visible_when_no_sessions_have_saved_curves():
    """Render the intended roster and exclusion reasons even when all sessions fail."""
    from src.neural_analysis.presentation_figures.prediction_plotting import render_prediction_cohort

    fields = {"roster": pd.DataFrame({"session_id": ["day 1", "day 2"],
                                      "status": ["failed", "excluded"],
                                      "reason": ["Missing alignment", "Too few good units"]})}
    figures = render_prediction_cohort(fields, {"mouse_id": "synthetic-mouse"}, "light")
    try:
        for figure in figures.values():
            figure.canvas.draw()
            caption = " ".join(element.get_text() for element in figure.texts)
            assert "day 1 (Missing alignment)" in caption
            assert "day 2 (Too few good units)" in caption
            assert any(text.get_text() == "No supported scores."
                       for axis in figure.axes for text in axis.texts)
    finally:
        for figure in figures.values():
            plt.close(figure)


def test_current_figure5_and_8_label_saved_thirty_pc_ridge():
    """Use the saved thirty-PC reference and show why smaller sessions lack it."""
    from src.neural_analysis.presentation_figures.prediction_plotting import (
        render_prediction,
        render_prediction_cohort,
    )

    curves = prediction_curves().replace({"model": {"ridge_10pc": "ridge_30pc"}})
    curves["source_pool_size"] = 32
    session_fields = {
        "curves": curves, "row_positions": np.arange(18), "source_pool_size": 32,
        "panel_d_pcs": 30, "available_units": pd.DataFrame({"region": ["PFC", "HPC"], "n_units": [64, 66]}),
    }
    unsupported = curves.assign(session_id="Small", source_pool_size=20)
    unsupported.loc[unsupported.panel.eq("D"), "mean_r2"] = np.nan
    cohort_curves = pd.concat([curves.assign(session_id="Large"), unsupported])
    medians = curves[["panel", "count", "direction", "model", "mean_r2"]].assign(n_sessions=1, roster_size=2)
    cohort_fields = {
        "curves": cohort_curves, "prediction_medians": medians,
        "roster": pd.DataFrame({"session_id": ["Large", "Small"], "status": ["ok", "ok"], "reason": ["", ""]}),
    }
    figures = {}
    try:
        figures.update(render_prediction(session_fields, {"session_id": "Large"}, "light"))
        figures.update(render_prediction_cohort(cohort_fields, {"mouse_id": "SYNTH"}, "light"))
        for name in ("figure5", "figure8_ranks"):
            figure = figures[name]
            figure.canvas.draw()
            assert any("30 source PCs" in axis.get_title() for axis in figure.axes)
            legend_labels = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
            assert "30-PC ridge" in legend_labels
            assert "Ten-PC ridge" not in legend_labels
        caption = " ".join(element.get_text() for element in figures["figure8_ranks"].texts)
        assert "Small, panel D (only 20 sampled source units)" in caption
    finally:
        for figure in figures.values():
            plt.close(figure)
