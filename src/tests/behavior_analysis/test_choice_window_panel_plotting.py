"""Standalone window agreement must reproduce the existing diagnostic panels."""

from copy import deepcopy
from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import performance_plots as plots


@pytest.fixture
def window_results() -> tuple[pd.DataFrame, dict]:
    """Return (10,6) saved-style rows and metadata for two agents/sizes.

    Bounds are zero-based half-open original trial positions. Agreement is
    unitless; counts are usable choices. One agreement is missing deliberately.
    """
    table = pd.DataFrame([
        (10, "qlearning_mouse_agreement", 0, 10, .2, 10),
        (10, "qlearning_mouse_agreement", 10, 20, .5, 8),
        (10, "qlearning_mouse_agreement", 20, 30, np.nan, 0),
        (10, "fql_mouse_agreement", 0, 10, .7, 10),
        (10, "fql_mouse_agreement", 10, 20, .3, 10),
        (10, "fql_mouse_agreement", 20, 30, .6, 10),
        (20, "qlearning_mouse_agreement", 0, 20, .35, 18),
        (20, "qlearning_mouse_agreement", 20, 40, .8, 20),
        (20, "fql_mouse_agreement", 0, 20, .5, 20),
        (20, "fql_mouse_agreement", 20, 40, .65, 20),
    ], columns=["window_size", "agent", "trial_start", "trial_stop", "agreement", "n_comparisons"])
    metadata = {
        "parameters": {"window_sizes": [10, 20], "agent_value_columns": {
            "qlearning_mouse_agreement": "Qlearning_rel_value", "fql_mouse_agreement": "FQlearning_rel_value",
        }},
        "context_transition_trial_positions": [7, 23], "discarded_choices": {"10": 3, "20": 13},
    }
    return table, metadata


@pytest.fixture
def closed_figures(monkeypatch) -> Mock:
    """Capture real Figure close calls while preserving normal PNG saving/cleanup; return Mock."""
    close = Mock(wraps=plt.close)
    monkeypatch.setattr(plt, "close", close)
    return close


def _figures(close: Mock) -> list[plt.Figure]:
    """Extract saved-and-closed Figure objects from a close Mock; return ordered figure list."""
    return [call.args[0] for call in close.call_args_list if call.args and isinstance(call.args[0], plt.Figure)]


def test_standalone_panels_match_combined_curves_and_keep_inputs(window_results, closed_figures, tmp_path):
    """Input: (10,6) table/meta. Require exact panel coordinates/styles and unchanged four-axis diagnostic."""
    table, metadata = window_results
    original, original_meta = table.copy(deep=True), deepcopy(metadata)
    combined = plots.plot_session_choice_window_agreement(table, metadata, tmp_path, "session")
    exporter = getattr(plots, "plot_session_choice_window_agreement_panel")
    outputs = [exporter(table, metadata, tmp_path, "session", size) for size in (10, 20)]
    figures = _figures(closed_figures)
    assert len(figures[0].axes) == 4
    assert combined.name == "session_choice_window_agreement.png"
    np.testing.assert_array_equal(figures[0].axes[1].lines[0].get_ydata(), [10, 8, 0])
    for row, (size, output, standalone) in enumerate(zip((10, 20), outputs, figures[1:])):
        assert output.name == f"session_choice_window_{size}_agreement.png"
        assert output.stat().st_size > 0
        assert len(standalone.axes) == 1
        assert standalone.get_facecolor() == (1, 1, 1, 1)
        expected, actual = figures[0].axes[2 * row], standalone.axes[0]
        assert len(actual.lines) == len(expected.lines) == 4
        for reference, line in zip(expected.lines, actual.lines):
            np.testing.assert_allclose(line.get_xdata(), reference.get_xdata(), equal_nan=True)
            np.testing.assert_allclose(line.get_ydata(), reference.get_ydata(), equal_nan=True)
            assert (line.get_color(), line.get_label(), line.get_marker()) == (
                reference.get_color(), reference.get_label(), reference.get_marker())
        assert actual.get_ylim() == expected.get_ylim() == (-.05, 1.05)
        assert actual.get_xlabel() == expected.get_xlabel()
        assert actual.get_ylabel() == expected.get_ylabel()
        assert [text.get_text() for text in actual.get_legend().get_texts()] == ["QL", "FQL"]
        np.testing.assert_allclose(actual.lines[0].get_xdata(), [4.5, 14.5, 24.5] if size == 10 else [9.5, 29.5])
        assert str(metadata["discarded_choices"][str(size)]) in standalone.texts[-1].get_text()
    pd.testing.assert_frame_equal(table, original)
    assert metadata == original_meta


@pytest.mark.parametrize("empty", [True, False])
def test_standalone_handles_empty_windows_and_missing_predictions(window_results, closed_figures, tmp_path, empty):
    """Input: empty or all-NaN agreement rows. Save one labeled axis without inventing values."""
    table, metadata = window_results
    table = table.iloc[:0].copy() if empty else table.assign(agreement=np.nan)
    output = getattr(plots, "plot_session_choice_window_agreement_panel")(table, metadata, tmp_path, "empty", 20)
    assert output.is_file()
    axis = _figures(closed_figures)[0].axes[0]
    if empty:
        assert any("No complete choice windows" in text.get_text() for text in axis.texts)
    else:
        assert np.isnan(axis.lines[0].get_ydata()).all()


def test_standalone_reuses_existing_png_and_validates_size(window_results, closed_figures, tmp_path):
    """Input: saved-style rows. Existing PNG bytes remain unchanged and malformed sizes are rejected."""
    table, metadata = window_results
    exporter = getattr(plots, "plot_session_choice_window_agreement_panel")
    first = exporter(table, metadata, tmp_path, "session", 10)
    original = first.read_bytes()
    assert exporter(table, metadata, tmp_path, "session", 10) == first
    assert len(_figures(closed_figures)) == 1
    assert first.read_bytes() == original
    with pytest.raises(ValueError):
        exporter(table, metadata, tmp_path, "invalid", 0)
    with pytest.raises(ValueError, match="agreement"):
        exporter(table.drop(columns="agreement"), metadata, tmp_path, "invalid", 20)


def test_full_session_title_legend_and_caption_fit_standalone_canvas(window_results, closed_figures, tmp_path):
    """Input: full session ID and saved rows. Figure text fits without overlap at presentation size."""
    table, metadata = window_results
    getattr(plots, "plot_session_choice_window_agreement_panel")(
        table, metadata, tmp_path, "CT016_2026-05-29_132812", 10,
    )
    figure = _figures(closed_figures)[0]
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for text in figure.texts:
        bounds = text.get_window_extent(renderer)
        assert bounds.x0 >= 0 and bounds.x1 <= figure.bbox.x1
        assert bounds.y0 >= 0 and bounds.y1 <= figure.bbox.y1
    caption = figure.texts[-1].get_window_extent(renderer)
    assert caption.y1 < figure.axes[0].get_tightbbox(renderer).y0
