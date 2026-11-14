"""True-time probability tracks and independent saved-only PNG export."""

import numpy as np
import pytest
from PIL import Image

from src.neural_analysis.population_decoding.pipeline import fit_target
from src.neural_analysis.population_decoding.plotting import plot_decoding_timeline, export_png


def test_outcomes_are_post_choice_and_real_gaps_remain(synthetic_population, tmp_path):
    """Input prepared data/temp dir; outcome probability segments never extend before choice."""
    cell = fit_target(synthetic_population, "PFC", "outcome")
    figure = plot_decoding_timeline(synthetic_population, cell, "units", viewport_revision=2)
    traces = [trace for trace in figure.data if trace.name.startswith("P(")]
    assert len(traces) == 4
    assert list(figure.layout.yaxis3.range) == [0, 1]
    for trace in traces:
        assert any(value is None or np.isnan(value) for value in trace.x)
        custom = [row for row in trace.customdata if row[0] is not None]
        assert all(row[1] >= 0 for row in custom)
    assert figure.layout.meta["target"] == "outcome"
    assert figure.layout.meta["primary_time_axis"] == "xaxis"
    other = plot_decoding_timeline(synthetic_population, cell, "pcs", viewport_revision=2)
    assert figure.layout.uirevision == other.layout.uirevision
    path = tmp_path / "outcome.png"
    export_png(path, synthetic_population, cell, "units")
    with Image.open(path) as image:
        assert image.getpixel((0, 0))[:3] == (255, 255, 255)
    with pytest.raises(FileExistsError):
        export_png(path, synthetic_population, cell, "units")


def test_whole_window_switch_tracks_use_current_trial(synthetic_population):
    """Input population; switch OOF labels/probabilities occupy current choice [-1,+1)."""
    cell = fit_target(synthetic_population, "PFC", "left_choice_switch")
    figure = plot_decoding_timeline(synthetic_population, cell, "units")
    trace = next(trace for trace in figure.data if trace.name == "P(left_switch)")
    relative = np.array([row[1] for row in trace.customdata if row[0] is not None])
    assert relative.min() == -1 and relative.max() == .5
