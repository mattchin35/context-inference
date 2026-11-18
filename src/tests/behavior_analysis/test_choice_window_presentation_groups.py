"""Presentation subsets retain saved agreement values and consistent model colors."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import performance_plots as plots
from src.tests.behavior_analysis.test_choice_window_panel_plotting import closed_figures, _figures


GROUPS = {
    "main_comparison_v1": ("qlearning_mouse_agreement", "hmm_logodds_mouse_agreement",
                           "home_referenced_expectancy_doubt_mouse_agreement", "observer_mouse_agreement"),
    "strategy_comparison_v2": ("perseveration_mouse_agreement", "wsls_mouse_agreement",
                               "win_switch_lose_stay_mouse_agreement", "observer_mouse_agreement"),
}
LABELS = {
    "qlearning_mouse_agreement": "QL", "hmm_logodds_mouse_agreement": "HMM",
    "home_referenced_expectancy_doubt_mouse_agreement": "Home E + Doubt",
    "observer_mouse_agreement": "Ideal", "perseveration_mouse_agreement": "Perseveration",
    "wsls_mouse_agreement": "WSLS", "win_switch_lose_stay_mouse_agreement": "Win-switch/lose-stay",
}


@pytest.fixture
def presentation_results() -> tuple[pd.DataFrame, dict]:
    """Return synthetic (48,6) rows for eight agents and two sizes; no RNG or scoring.

    Bounds are zero-based half-open trial rows; agreement is a unitless fraction,
    counts are choices. FQL is a distractor excluded from both presentation groups.
    """
    agents = [*LABELS, "fql_mouse_agreement"]
    records = []
    for size in (10, 20):
        for agent_index, agent in enumerate(agents):
            for window in range(3):
                records.append((size, agent, size * window, size * (window + 1), .1 * agent_index + .05 * window, size))
    table = pd.DataFrame(records, columns=["window_size", "agent", "trial_start", "trial_stop", "agreement", "n_comparisons"])
    meta = {"parameters": {"window_sizes": [10, 20], "agent_value_columns": dict.fromkeys(agents, "saved_value")},
            "context_transition_trial_positions": [7, 23], "discarded_choices": {"10": 3, "20": 13}}
    return table, meta


def test_named_groups_define_exact_requested_models():
    """Input: none. Public preset definitions must contain exactly the two ordered comparisons."""
    assert getattr(plots, "choice_window_presentation_groups")() == GROUPS


@pytest.mark.parametrize("profile", GROUPS)
@pytest.mark.parametrize("size", [10, 20])
def test_group_curves_order_colors_and_saved_values(presentation_results, closed_figures, tmp_path, profile, size):
    """Input: saved-style (48,6) data. Restrict display only, preserving values/colors/earlier PNGs."""
    table, meta = presentation_results
    original, original_meta = table.copy(deep=True), deepcopy(meta)
    full = plots.plot_session_choice_window_agreement_panel(table, meta, tmp_path, "session", size)
    full_bytes = full.read_bytes()
    path = plots.plot_session_choice_window_agreement_panel(
        table, meta, tmp_path, "session", size,
        agent_selection=GROUPS[profile], presentation_profile=profile,
    )
    full_figure, grouped = _figures(closed_figures)
    axis = grouped.axes[0]
    assert len(grouped.axes) == 1
    assert path.name == f"session_choice_window_{size}_{profile}_agreement.png"
    assert path != full and path.stat().st_size > 0 and full.read_bytes() == full_bytes
    assert [line.get_label() for line in axis.lines[:4]] == [LABELS[key] for key in GROUPS[profile]]
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [LABELS[key] for key in GROUPS[profile]]
    assert len(axis.lines) == 6  # Four model curves and two existing context boundaries.
    full_lines = {line.get_label(): line for line in full_figure.axes[0].lines}
    base_labels = {column: label for label, column in plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS.items()}
    for key, line in zip(GROUPS[profile], axis.lines):
        rows = table.loc[table.window_size.eq(size) & table.agent.eq(key)]
        np.testing.assert_allclose(line.get_ydata(), rows.agreement)
        np.testing.assert_allclose(line.get_xdata(), (rows.trial_start + rows.trial_stop - 1) / 2)
        assert line.get_color() == full_lines[base_labels.get(key, LABELS[key])].get_color()
    assert grouped.get_facecolor() == (1, 1, 1, 1)
    pd.testing.assert_frame_equal(table, original)
    assert meta == original_meta


@pytest.mark.parametrize("all_missing", [False, True])
def test_missing_group_models_are_reported_without_substitution(presentation_results, closed_figures, tmp_path, all_missing):
    """Input: saved metadata lacking selected models. Plot available curves and explicitly name omissions."""
    table, meta = presentation_results
    missing = GROUPS["main_comparison_v1"] if all_missing else ("home_referenced_expectancy_doubt_mouse_agreement",)
    for key in missing:
        meta["parameters"]["agent_value_columns"].pop(key)
    table = table.loc[~table.agent.isin(missing)]
    plots.plot_session_choice_window_agreement_panel(
        table, meta, tmp_path, "missing", 10,
        agent_selection=GROUPS["main_comparison_v1"], presentation_profile="main_comparison_v1",
    )
    figure = _figures(closed_figures)[0]
    caption = figure.texts[-1].get_text()
    assert "Unavailable models:" in caption
    assert "Home E + Doubt" in caption
    curves = [line for line in figure.axes[0].lines if not line.get_label().startswith("_")]
    assert len(curves) == (0 if all_missing else 3)
    if all_missing:
        assert any("No requested models available" in text.get_text() for text in figure.axes[0].texts)


@pytest.mark.parametrize("profile", GROUPS)
@pytest.mark.parametrize("large_labels", [False, True])
def test_group_layout_and_existing_file_reuse(presentation_results, closed_figures, tmp_path, profile, large_labels):
    """Input: full session ID and four-curve selection. Legend clears data/caption; repeated exports preserve bytes."""
    table, meta = presentation_results
    options = {"agent_selection": GROUPS[profile], "presentation_profile": profile}
    style = {"axes.labelsize": 18, "axes.titlesize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16} if large_labels else {}
    with plt.rc_context(style):
        path = plots.plot_session_choice_window_agreement_panel(table, meta, tmp_path, "CT016_2026-05-29_132812", 10, **options)
    contents = path.read_bytes()
    assert plots.plot_session_choice_window_agreement_panel(table, meta, tmp_path, "CT016_2026-05-29_132812", 10, **options) == path
    assert path.read_bytes() == contents and len(_figures(closed_figures)) == 1
    figure = _figures(closed_figures)[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = figure.axes[0]
    legend = axis.get_legend().get_window_extent(renderer)
    assert legend.y1 < axis.get_window_extent(renderer).y0
    assert legend.y1 < axis.xaxis.label.get_window_extent(renderer).y0 - 5
    assert figure.texts[-1].get_window_extent(renderer).y1 < legend.y0
    for artist in [*figure.texts, axis.get_legend()]:
        bounds = artist.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.x1
        assert bounds.y0 >= 0 and bounds.y1 <= figure.bbox.y1


def test_selection_requires_distinct_output_identity(presentation_results, tmp_path):
    """Input: selection without a profile, or a typo. Reject ambiguous output paths and unknown models."""
    table, meta = presentation_results
    with pytest.raises(ValueError, match="profile"):
        plots.plot_session_choice_window_agreement_panel(table, meta, tmp_path, "invalid", 10, agent_selection=GROUPS["main_comparison_v1"])
    with pytest.raises(ValueError, match="agent"):
        plots.plot_session_choice_window_agreement_panel(table, meta, tmp_path, "invalid", 10, agent_selection=("typo",), presentation_profile="custom_v1")


def test_older_results_report_missing_inverse_rule(presentation_results, closed_figures, tmp_path):
    """Old saved rows[T,C] have no inverse rule; caption names it without inventing data."""
    table, meta = presentation_results
    key = "win_switch_lose_stay_mouse_agreement"
    meta["parameters"]["agent_value_columns"].pop(key)
    plots.plot_session_choice_window_agreement_panel(
        table.loc[~table.agent.eq(key)], meta, tmp_path, "old", 10,
        agent_selection=GROUPS["strategy_comparison_v2"], presentation_profile="strategy_comparison_v2")
    figure = _figures(closed_figures)[0]
    assert "Unavailable models: Win-switch/lose-stay." in figure.texts[-1].get_text()
