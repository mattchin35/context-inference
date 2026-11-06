"""Cross-session window agreement follows the existing agreement figure style."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def summary():
    """Return four session/size summaries, shape (4, 7), with unitless quartiles."""
    return pd.DataFrame({
        "session_id": ["s1", "s2", "s1", "s2"],
        "date": ["2026-11-01"] * 4,
        "window_size": [10, 10, 20, 20],
        "agent": ["qlearning_mouse_agreement"] * 4,
        "agreement_q1": [0.25, 0.5, 0.1, 0.3],
        "agreement_median": [0.5, 0.75, 0.2, 0.6],
        "agreement_q3": [0.75, 1.0, 0.4, 0.8],
    })


@pytest.mark.parametrize("size,medians", [(10, [0.5, 0.75]), (20, [0.2, 0.6])])
def test_agreement_only_plot_matches_model_style_and_distinguishes_sessions(
    tmp_path, monkeypatch, summary, size, medians
):
    """One agreement axis uses existing colors, quartiles, and separate same-day ticks."""
    from src.behavior_analysis.performance_plots import plot_multisession_choice_window_agreement

    captured = []
    original_save = plt.Figure.savefig

    def capture(figure, *args, **kwargs):
        """Capture the Figure and forward original savefig inputs unchanged."""
        captured.append(figure)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    original = summary.copy(deep=True)
    path = plot_multisession_choice_window_agreement(summary, tmp_path, "mouse_run", size)
    assert path.parent == tmp_path
    assert path.name == f"mouse_run_choice-window-{size}_agreement.png"
    assert path.exists()
    figure = captured[0]
    assert len(figure.axes) == 1
    assert figure.get_facecolor() == (1, 1, 1, 1)
    axis = figure.axes[0]
    median = next(line for line in axis.lines if line.get_label() == "QL median")
    np.testing.assert_array_equal(median.get_ydata(), medians)
    assert median.get_color() == "#1f77b4"
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["2026-11-01\ns1", "2026-11-01\ns2"]
    assert "coverage" not in axis.get_title().lower()
    assert "agreement" in axis.get_ylabel().lower()
    pd.testing.assert_frame_equal(summary, original)


def test_no_complete_windows_still_saves_agreement_placeholder(tmp_path, monkeypatch, summary):
    """Empty summaries yield a readable figure instead of aborting multisession work."""
    from src.behavior_analysis.performance_plots import plot_multisession_choice_window_agreement

    saved_figures = []
    original_save = plt.Figure.savefig

    def capture(figure, *args, **kwargs):
        """Capture the Figure and forward original savefig inputs unchanged."""
        saved_figures.append(figure)
        return original_save(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture)
    path = plot_multisession_choice_window_agreement(summary.iloc[:0], tmp_path, "empty", 20)
    assert path.exists()
    assert len(saved_figures[0].axes) == 1
    assert any("No complete" in text.get_text() for text in saved_figures[0].axes[0].texts)
