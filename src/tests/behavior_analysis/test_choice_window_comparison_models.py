"""All standard single-session window views share the retained comparison set."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import performance_plots as plots
from src.tests.behavior_analysis.test_choice_window_panel_plotting import closed_figures, _figures
from src.tests.behavior_analysis.test_choice_window_summary_plotting import summary_results


RETAINED = ("qlearning_mouse_agreement", "fql_mouse_agreement", "hmm_logodds_mouse_agreement",
            "perseveration_mouse_agreement", "wsls_mouse_agreement", "observer_mouse_agreement",
            "home_referenced_expectancy_doubt_mouse_agreement", "win_switch_lose_stay_mouse_agreement")
VERSION = "comparison_models_v1"


@pytest.fixture
def complete_results(summary_results):
    """Return 13-model saved rows with trial bounds/counts; no numerical analysis occurs."""
    table, metadata = summary_results
    table["trial_start"] = table.window_index * table.window_size
    table["trial_stop"] = table.trial_start + table.window_size
    metadata["context_transition_trial_positions"] = [7, 23]
    return table, metadata


@pytest.mark.parametrize("view", ["diagnostic", "timeline", "summary"])
def test_default_views_share_eight_models_and_keep_saved_values(complete_results, closed_figures, tmp_path, view):
    """Saved (T,C) inputs retain agreement/count values and colors in all default view types."""
    table, metadata = complete_results
    original, original_meta = table.copy(deep=True), deepcopy(metadata)
    block_registry = dict(plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS)
    if view == "diagnostic":
        path = plots.plot_session_choice_window_agreement(table, metadata, tmp_path, "session")
    else:
        exporter = plots.plot_session_choice_window_agreement_panel if view == "timeline" else plots.plot_session_choice_window_agreement_summary
        path = exporter(table, metadata, tmp_path, "session", 10)
    assert path.parent == tmp_path / VERSION
    figure = _figures(closed_figures)[0]
    expected_labels = [plots.CHOICE_WINDOW_AGENT_LABELS[agent] for agent in RETAINED]
    if view == "summary":
        axis = figure.axes[0]
        assert len(axis.collections) == 8
        assert "Model Comparison" in figure.texts[0].get_text()
        for agent, dots in zip(RETAINED, axis.collections):
            expected = table.loc[table.agent.eq(agent) & table.window_size.eq(10), "agreement"].dropna()
            np.testing.assert_allclose(dots.get_offsets()[:, 1], expected)
    else:
        axes = figure.axes if view == "diagnostic" else [figure.axes[0]]
        for index, axis in enumerate(axes):
            models = [line for line in axis.lines if not line.get_label().startswith("_")]
            assert [line.get_label() for line in models] == expected_labels
            size = (10, 10, 20, 20)[index] if view == "diagnostic" else 10
            for agent, line in zip(RETAINED, models):
                rows = table.loc[table.agent.eq(agent) & table.window_size.eq(size)]
                column = "n_comparisons" if view == "diagnostic" and index % 2 else "agreement"
                np.testing.assert_allclose(line.get_ydata(), rows[column], equal_nan=True)
                np.testing.assert_allclose(line.get_xdata(), (rows.trial_start + rows.trial_stop - 1) / 2)
                assert line.get_color() == plots.AGENT_MOUSE_AGREEMENT_COLORS[plots.CHOICE_WINDOW_AGENT_LABELS[agent]]
    assert tuple(plots.MULTISESSION_CHOICE_WINDOW_AGENTS) == RETAINED
    assert plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS == block_registry
    pd.testing.assert_frame_equal(table, original)
    assert metadata == original_meta


@pytest.mark.parametrize("view", ["diagnostic", "timeline", "summary"])
def test_legacy_cache_does_not_suppress_versioned_exports(complete_results, closed_figures, tmp_path, view):
    """Legacy PNG bytes survive; new-profile files are cached independently on repeat calls."""
    table, metadata = complete_results
    if view == "diagnostic":
        exporter, extra, name = plots.plot_session_choice_window_agreement, (), "session_choice_window_agreement.png"
    elif view == "timeline":
        exporter, extra, name = plots.plot_session_choice_window_agreement_panel, (10,), "session_choice_window_10_agreement.png"
    else:
        exporter, extra, name = plots.plot_session_choice_window_agreement_summary, (10,), "session_choice_window_10_summary.png"
    legacy = tmp_path / name
    legacy.write_bytes(b"earlier full-model figure")
    output = exporter(table, metadata, tmp_path, "session", *extra)
    assert output != legacy and output.parent == tmp_path / VERSION
    contents = output.read_bytes()
    assert exporter(table, metadata, tmp_path, "session", *extra) == output
    assert len(_figures(closed_figures)) == 1 and output.read_bytes() == contents
    assert legacy.read_bytes() == b"earlier full-model figure"


@pytest.mark.parametrize("view", ["diagnostic", "timeline", "summary"])
def test_default_views_report_missing_home_and_win_switch(complete_results, closed_figures, tmp_path, view):
    """Unavailable retained models are explicitly named rather than replaced by excluded models."""
    table, metadata = complete_results
    missing = RETAINED[-2:]
    for agent in missing:
        metadata["parameters"]["agent_value_columns"].pop(agent)
    table = table.loc[~table.agent.isin(missing)]
    if view == "diagnostic":
        plots.plot_session_choice_window_agreement(table, metadata, tmp_path, "missing")
    else:
        exporter = plots.plot_session_choice_window_agreement_panel if view == "timeline" else plots.plot_session_choice_window_agreement_summary
        exporter(table, metadata, tmp_path, "missing", 10)
    caption = _figures(closed_figures)[0].texts[-1].get_text()
    assert "Unavailable models:" in caption and "Home E + Doubt" in caption and "Win-switch/lose-stay" in caption


@pytest.mark.parametrize("view", ["diagnostic", "timeline", "summary"])
def test_only_excluded_models_produce_a_labeled_empty_view(complete_results, closed_figures, tmp_path, view):
    """Existing rows for excluded models cannot masquerade as available comparison data."""
    table, metadata = complete_results
    excluded = set(metadata["parameters"]["agent_value_columns"]) - set(RETAINED)
    metadata["parameters"]["agent_value_columns"] = dict.fromkeys(excluded, "saved_value")
    table = table.loc[table.agent.isin(excluded)]
    if view == "diagnostic":
        plots.plot_session_choice_window_agreement(table, metadata, tmp_path, "empty")
    else:
        exporter = plots.plot_session_choice_window_agreement_panel if view == "timeline" else plots.plot_session_choice_window_agreement_summary
        exporter(table, metadata, tmp_path, "empty", 10)
    figure = _figures(closed_figures)[0]
    for axis in figure.axes:
        assert not any(not line.get_label().startswith("_") for line in axis.lines)
        assert not axis.collections
    assert any("No requested models available" in text.get_text() for axis in figure.axes for text in axis.texts)


@pytest.mark.parametrize("count", [85, 100])
def test_summary_count_labels_fit_with_presentation_fonts(complete_results, closed_figures, tmp_path, count):
    """Two/three-digit window counts remain inside the canvas with the app's large tick fonts."""
    import matplotlib.pyplot as plt

    table, metadata = complete_results
    seed = table.loc[table.window_size.eq(10)].drop_duplicates("agent")
    table = pd.concat([seed.assign(window_index=index, agreement=.5) for index in range(count)], ignore_index=True)
    with plt.rc_context({"axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16}):
        plots.plot_session_choice_window_agreement_summary(table, metadata, tmp_path, "CT016_2026-05-29_132812", 10)
    figure = _figures(closed_figures)[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for label in figure.axes[0].get_xticklabels():
        bounds = label.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.x1
