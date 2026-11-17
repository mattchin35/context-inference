"""Cross-session window agreement follows the existing agreement figure style."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from src.behavior_analysis import performance_plots as plots
from src.tests.behavior_analysis.test_choice_window_panel_plotting import closed_figures, _figures


RETAINED = ("qlearning_mouse_agreement", "fql_mouse_agreement", "hmm_logodds_mouse_agreement",
            "perseveration_mouse_agreement", "wsls_mouse_agreement", "observer_mouse_agreement",
            "home_referenced_expectancy_doubt_mouse_agreement", "win_switch_lose_stay_mouse_agreement")


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


@pytest.mark.parametrize("size", [10, 20])
def test_restricted_cross_session_models_preserve_medians_iqr_and_block_registry(summary, closed_figures, tmp_path, size):
    """Saved (session,size,agent) rows retain eight model series without altering block defaults."""
    original_registry = dict(plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS)
    table = pd.concat([summary.assign(agent=agent) for agent in plots.CHOICE_WINDOW_AGENT_LABELS], ignore_index=True)
    original = table.copy(deep=True)
    plots.plot_multisession_choice_window_agreement(table, tmp_path, "run", size)
    axis = _figures(closed_figures)[0].axes[0]
    expected_labels = [plots.CHOICE_WINDOW_AGENT_LABELS[agent] for agent in RETAINED]
    medians = [line for line in axis.lines if line.get_label().endswith(" median")]
    assert {line.get_label() for line in medians} == {f"{label} median" for label in expected_labels}
    for line in medians:
        label = line.get_label().removesuffix(" median")
        np.testing.assert_allclose(line.get_ydata(), summary.loc[summary.window_size.eq(size), "agreement_median"])
        assert line.get_color() == plots.AGENT_MOUSE_AGREEMENT_COLORS[label]
        bars = [bar for bar in axis.lines if len(bar.get_xdata()) == 2
                and np.all(np.asarray(bar.get_xdata()) == bar.get_xdata()[0])
                and bar.get_color() == line.get_color()]
        assert len(bars) == 2
        for bar, row in zip(bars, summary.loc[summary.window_size.eq(size)].itertuples()):
            np.testing.assert_allclose(bar.get_ydata(), [row.agreement_q1, row.agreement_q3])
    assert plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS == original_registry
    assert "HMM decay" in original_registry and "Probe+P+D" in original_registry
    pd.testing.assert_frame_equal(table, original)


@pytest.mark.parametrize("empty", [False, True])
def test_short_subject_title_keeps_timestamped_filename_and_fits(summary, closed_figures, tmp_path, empty):
    """Display-only subject label gives a short fitting title; the full run ID stays in the filename."""
    identifier = "CT016_2026-11-17_16-59-59_123456"
    table = summary.iloc[:0] if empty else summary
    path = plots.plot_multisession_choice_window_agreement(table, tmp_path, identifier, 10, subject_label="CT016")
    figure = _figures(closed_figures)[0]
    assert path.name == f"{identifier}_choice-window-10_agreement.png"
    axis = figure.axes[0]
    assert axis.get_title() == "CT016: 10-choice agreement"
    figure.canvas.draw()
    bounds = axis.title.get_window_extent(figure.canvas.get_renderer())
    assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.x1


def test_filter_preserves_sessions_with_only_excluded_models(summary, closed_figures, tmp_path):
    """A session with no retained model keeps its x position rather than shifting later points."""
    table = summary.loc[summary.window_size.eq(10)].copy()
    table.loc[table.session_id.eq("s1"), "agent"] = "hmm_logodds_decay_mouse_agreement"
    plots.plot_multisession_choice_window_agreement(table, tmp_path, "run", 10)
    figure = _figures(closed_figures)[0]
    axis = figure.axes[0]
    median = next(line for line in axis.lines if line.get_label() == "QL median")
    assert len(median.get_xdata()) == 1 and median.get_xdata()[0] > 1.5
    assert len(axis.get_xticklabels()) == 2
    assert "Unavailable models:" in figure.texts[-1].get_text()
    assert "Win-switch/lose-stay" in figure.texts[-1].get_text()


def test_filtered_only_input_produces_placeholder_without_removed_curves(summary, closed_figures, tmp_path):
    """Saved rows containing only a removed model still preserve session labels and explain empty display."""
    plots.plot_multisession_choice_window_agreement(
        summary.assign(agent="hmm_logodds_decay_mouse_agreement"), tmp_path, "run", 20)
    axis = _figures(closed_figures)[0].axes[0]
    assert not any(line.get_label().endswith(" median") for line in axis.lines)
    assert len(axis.get_xticklabels()) == 2
    assert any("No complete" in text.get_text() for text in axis.texts)
