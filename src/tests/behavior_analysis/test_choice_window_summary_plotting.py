"""Single-session window distributions preserve raw values and descriptive quartiles."""

from copy import deepcopy

import matplotlib.colors as colors
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import performance_plots as plots
from src.tests.behavior_analysis.test_choice_window_panel_plotting import closed_figures, _figures


MAIN = ("qlearning_mouse_agreement", "hmm_logodds_mouse_agreement",
        "home_referenced_expectancy_doubt_mouse_agreement", "observer_mouse_agreement")
STRATEGY = ("perseveration_mouse_agreement", "wsls_mouse_agreement",
            "win_switch_lose_stay_mouse_agreement", "observer_mouse_agreement")
LABELS = {**plots.CHOICE_WINDOW_AGENT_LABELS,
          "perseveration_mouse_agreement": "Perseveration",
          "home_referenced_expectancy_doubt_mouse_agreement": "Home E + Doubt",
          "simple_probe_persistence_mouse_agreement": "Probe + persistence"}
COMPARISON = ("qlearning_mouse_agreement", "fql_mouse_agreement", "hmm_logodds_mouse_agreement",
              "perseveration_mouse_agreement", "wsls_mouse_agreement", "observer_mouse_agreement",
              "home_referenced_expectancy_doubt_mouse_agreement", "win_switch_lose_stay_mouse_agreement")
SELECTIONS = {None: None, "main_comparison_v1": MAIN, "strategy_comparison_v2": STRATEGY}


@pytest.fixture
def summary_results():
    """Return (104,6) saved-style rows for 13 agents, two sizes and four windows.

    Agreements are unitless fractions; n_comparisons counts usable predictions.
    For each model, ten-choice values are [0,.2,.8,missing] and twenty-choice
    values are [.1,.3,.5,.9]. Deliberately unequal denominators must not weight
    quartiles or trigger additional filtering of finite saved agreement values.
    """
    records = []
    for size, values in ((10, [0., .2, .8, np.nan]), (20, [.1, .3, .5, .9])):
        for agent in plots.CHOICE_WINDOW_AGENT_LABELS:
            for window, value in enumerate(values):
                records.append(("session", size, window, agent, value, [size, 1, size, 0 if size == 10 else size][window]))
    table = pd.DataFrame(records, columns=["session_id", "window_size", "window_index", "agent", "agreement", "n_comparisons"])
    meta = {"parameters": {"window_sizes": [10, 20],
                           "agent_value_columns": dict.fromkeys(plots.CHOICE_WINDOW_AGENT_LABELS, "saved_value")},
            "discarded_choices": {"10": 3, "20": 13}}
    return table, meta


@pytest.mark.parametrize("profile", SELECTIONS)
@pytest.mark.parametrize("size,expected,quartiles", [(10, [0., .2, .8], [.1, .2, .5]),
                                                   (20, [.1, .3, .5, .9], [.25, .4, .6])])
def test_summary_dots_quartiles_selection_colors_and_inputs(
    summary_results, closed_figures, tmp_path, profile, size, expected, quartiles,
):
    """Saved rows[T,C] produce unweighted quartiles and one dot per finite window."""
    table, meta = summary_results
    original, original_meta = table.copy(deep=True), deepcopy(meta)
    options = {} if profile is None else {"agent_selection": SELECTIONS[profile], "presentation_profile": profile}
    path = plots.plot_session_choice_window_agreement_summary(table, meta, tmp_path, "session", size, **options)
    figure = _figures(closed_figures)[0]
    axis = figure.axes[0]
    agents = COMPARISON if profile is None else SELECTIONS[profile]
    suffix = "" if profile is None else f"_{profile}"
    assert path.name == f"session_choice_window_{size}{suffix}_summary.png"
    assert path.is_file() and path.stat().st_size > 0
    assert len(figure.axes) == 1 and figure.get_facecolor() == (1., 1., 1., 1.)
    assert len(axis.collections) == len(agents)
    ticks = [tick.get_text() for tick in axis.get_xticklabels()]
    for index, (agent, collection) in enumerate(zip(agents, axis.collections)):
        np.testing.assert_allclose(collection.get_offsets()[:, 1], expected)
        assert np.abs(collection.get_offsets()[:, 0] - index).max() <= .08
        label = plots.CHOICE_WINDOW_AGENT_LABELS[agent]
        expected_color = colors.to_rgba(plots.AGENT_MOUSE_AGREEMENT_COLORS[label], alpha=.65)
        np.testing.assert_allclose(collection.get_facecolors()[0], expected_color)
        np.testing.assert_allclose(axis.lines[4 * index].get_ydata(), [quartiles[0], quartiles[2]])
        np.testing.assert_allclose(axis.lines[4 * index + 3].get_ydata(), [quartiles[1]] * 2)
        assert ticks[index] == f"{LABELS[agent]}\nn={len(expected)} windows"
    assert axis.get_ylim() == (-.05, 1.05)
    caption = figure.texts[-1].get_text()
    assert "equal weight" in caption and "not a confidence interval" in caption
    assert "seed=0" in caption and "no statistical tests" in caption
    assert str(meta["discarded_choices"][str(size)]) in caption
    pd.testing.assert_frame_equal(table, original)
    assert meta == original_meta


@pytest.mark.parametrize("case", ["empty", "all_missing", "one_window", "missing_model", "all_unavailable"])
def test_summary_sparse_data_and_missing_models(summary_results, closed_figures, tmp_path, case):
    """Empty/sparse saved rows[T,C] yield honest counts and explicit unavailable labels."""
    table, meta = summary_results
    if case == "empty":
        table = table.iloc[:0]
    elif case == "all_missing":
        table = table.assign(agreement=np.nan)
    elif case == "one_window":
        table = table.loc[table.window_index.eq(0)]
    elif case in ("missing_model", "all_unavailable"):
        missing = STRATEGY if case == "all_unavailable" else (STRATEGY[2],)
        for agent in missing:
            meta["parameters"]["agent_value_columns"].pop(agent)
        table = table.loc[~table.agent.isin(missing)]
    plots.plot_session_choice_window_agreement_summary(
        table, meta, tmp_path, case, 10,
        agent_selection=STRATEGY, presentation_profile="strategy_comparison_v2")
    figure = _figures(closed_figures)[0]
    axis = figure.axes[0]
    if case in ("empty", "all_missing", "all_unavailable"):
        assert len(axis.collections) == 0
        assert any("No " in text.get_text() for text in axis.texts)
    if case in ("empty", "all_missing"):
        assert all("n=0 windows" in text.get_text() for text in axis.get_xticklabels())
    if case == "one_window":
        assert len(axis.collections) == 4
        assert all(len(collection.get_offsets()) == 1 for collection in axis.collections)
        np.testing.assert_allclose(axis.lines[0].get_ydata(), [0., 0.])
    if case in ("missing_model", "all_unavailable"):
        assert "Unavailable models:" in figure.texts[-1].get_text()
        assert "Win-switch/lose-stay" in figure.texts[-1].get_text()


def test_summary_jitter_reproducibility_and_existing_file_reuse(summary_results, closed_figures, tmp_path):
    """Same saved values[T,C] give identical jitter; existing PNGs are reused unchanged."""
    table, meta = summary_results
    exporter = plots.plot_session_choice_window_agreement_summary
    first = exporter(table, meta, tmp_path / "first", "session", 10)
    before = first.read_bytes()
    assert exporter(table, meta, tmp_path / "first", "session", 10) == first
    assert first.read_bytes() == before and len(_figures(closed_figures)) == 1
    exporter(table, meta, tmp_path / "second", "session", 10)
    figures = _figures(closed_figures)
    for left, right in zip(figures[0].axes[0].collections, figures[1].axes[0].collections):
        np.testing.assert_array_equal(left.get_offsets(), right.get_offsets())


@pytest.mark.parametrize("profile", SELECTIONS)
def test_summary_layout_fits_long_labels_and_full_session_title(summary_results, closed_figures, tmp_path, profile):
    """All-model and restricted figures keep titles, counts and captions inside the canvas."""
    table, meta = summary_results
    options = {} if profile is None else {"agent_selection": SELECTIONS[profile], "presentation_profile": profile}
    plots.plot_session_choice_window_agreement_summary(table, meta, tmp_path, "CT016_2026-05-29_132812", 10, **options)
    figure = _figures(closed_figures)[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axis = figure.axes[0]
    caption = figure.texts[-1].get_window_extent(renderer)
    assert caption.y1 < axis.get_tightbbox(renderer).y0 - 5
    for text in [*figure.texts, *axis.get_xticklabels(), axis.xaxis.label, axis.yaxis.label, axis.title]:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.x1
        assert bounds.y0 >= 0 and bounds.y1 <= figure.bbox.y1
    if profile is None:
        assert figure.get_figwidth() >= 10


def test_summary_validation_reuses_profile_rules(summary_results, tmp_path):
    """Malformed size/columns/selection fail clearly before writing a PNG."""
    table, meta = summary_results
    exporter = plots.plot_session_choice_window_agreement_summary
    with pytest.raises(ValueError, match="positive"):
        exporter(table, meta, tmp_path, "bad_size", 0)
    with pytest.raises(ValueError, match="agreement"):
        exporter(table.drop(columns="agreement"), meta, tmp_path, "bad_columns", 10)
    with pytest.raises(ValueError, match="profile"):
        exporter(table, meta, tmp_path, "bad_profile", 10, agent_selection=STRATEGY)
    with pytest.raises(ValueError, match="Unknown agent"):
        exporter(table, meta, tmp_path, "bad_agent", 10, agent_selection=("typo",), presentation_profile="custom_v1")
