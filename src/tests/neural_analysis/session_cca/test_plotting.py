"""CCA display contracts: real bins, paired coordinates and stable session scales."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import cca
from src.neural_analysis.population import cca_plotting as plotting
from src.neural_analysis.population.session_cca_io import SessionCCAResult
from src.tests.neural_analysis.session_cca.test_computation import _analytic_counts


@pytest.fixture
def cca_display_result():
    """Return seeded (2,8,3) scores at elapsed 100/114 s plus one unpaired choice."""
    rng = np.random.default_rng(20261113)
    fitted = cca.fit_paired_cca(
        rng.poisson(2, (2, 8, 3)), rng.poisson(3, (2, 8, 4)),
        ("A:7", "A:8", "A:9"), ("B:7", "B:8", "B:9", "B:10"),
    )
    origin = 1_700_000_000.0
    choices = pd.DataFrame({
        "row_position": [4, 7, 10], "trial_id": [105, 106, 108],
        "choice_time": origin + np.array([100., 106., 114.]),
        "elapsed_s": [100., 106., 114.], "action": [1, 1, 0], "reward": [1, 0, 0],
    })
    events = pd.DataFrame({
        "Time": origin + np.array([0., 99., 101., 113., 115.]),
        "elapsed_s": [0., 99., 101., 113., 115.],
        "Event": ["enter_left_patch", "LED_on", "LED_off", "left_entry", "LED_on"],
    })
    return SessionCCAResult(
        "display-synthetic", origin, choices, events, np.array([0, 2]),
        {"PFC": np.array([0, 2]), "HPC": np.array([0, 2])}, np.linspace(-2, 2, 9),
        {"PFC": "A", "HPC": "B"}, fitted, None, {"bin_width_s": .5},
    )


def test_heatmaps_use_real_edges_paired_ranks_rho_and_measured_zero(cca_display_result):
    """Two (rank,cell) arrays share edges; only the ten-second gap is missing."""
    result = cca_display_result
    result.fit.scores["PFC"][0, 0, 0] = 0.
    expected_edges = np.r_[np.arange(98., 102.5, .5), np.arange(112., 116.5, .5)]
    for region in ("PFC", "HPC"):
        data = plotting.build_cca_heatmap_data(result, region)
        np.testing.assert_array_equal(data["x"], expected_edges)
        np.testing.assert_array_equal(data["y"], [1, 2, 3])
        assert data["z"].shape == (3, 17)
        assert np.isnan(data["z"][:, 8]).all()
        np.testing.assert_array_equal(data["z"][:, :8], result.fit.scores[region][0].T)
        np.testing.assert_array_equal(data["z"][:, 9:], result.fit.scores[region][1].T)
        assert data["customdata"][0, 0, 0] == 105
        assert data["customdata"][0, 9, 0] == 108
        np.testing.assert_allclose(data["customdata"][0, 0, 1:7].astype(float),
                                   [result.time_origin_s + 98, result.time_origin_s + 98.5,
                                    -2, -1.5, 98, 98.5])
        np.testing.assert_allclose(data["customdata"][:, 0, 7].astype(float), result.fit.correlations)
    assert plotting.build_cca_heatmap_data(result, "PFC")["z"][0, 0] == 0.


def test_default_first_ten_and_inclusive_pair_range_preserve_original_scores(cca_display_result):
    """Thirteen fitted pairs remain available; matched heatmap rows show original ranks."""
    fitted = cca.fit_paired_cca(*_analytic_counts(tuple(np.linspace(.05, .95, 13))))
    choices = pd.concat([cca_display_result.choices.iloc[[0]]] * 8, ignore_index=True)
    choices["elapsed_s"] = 100 + 14 * np.arange(8)
    choices["choice_time"] = cca_display_result.time_origin_s + choices.elapsed_s
    result = replace(cca_display_result, choices=choices, choice_positions=np.arange(8), fit=fitted)
    for region in ("PFC", "HPC"):
        default = plotting.build_cca_heatmap_data(result, region)
        np.testing.assert_array_equal(default["y"], np.arange(1, 11))
        selected = plotting.build_cca_heatmap_data(result, region, pair_range=(11, 13))
        np.testing.assert_array_equal(selected["y"], [11, 12, 13])
        np.testing.assert_array_equal(selected["z"][:, :8], fitted.scores[region][0, :, 10:13].T)
        np.testing.assert_allclose(selected["customdata"][:, 0, 7].astype(float), fitted.correlations[10:13])
    assert fitted.scores["PFC"].shape[-1] == 13


def test_bounded_payload_shared_scale_and_manual_saturation(cca_display_result):
    """Intersecting complete windows retain true edges and one session-wide color scale."""
    result = cca_display_result
    full = plotting.plot_session_cca(result)
    bounded = plotting.plot_session_cca(result, loaded_range_s=(98.2, 100.1), viewport_s=(98.2, 100.1))
    maps = _heatmaps(bounded)
    assert len(maps) == 2
    for trace in maps:
        np.testing.assert_array_equal(trace.x, np.arange(98., 102.5, .5))
        assert trace.zsmooth is False and trace.connectgaps is False
        assert _color_bounds(bounded, trace) == (-result.fit.color_limit, result.fit.color_limit)
    assert _colorbar_count(full) == 1
    assert bounded.layout.xaxis.range == (98.2, 100.1)
    manual = plotting.plot_session_cca(result, color_limit=.25)
    for before, after in zip(_heatmaps(full), _heatmaps(manual), strict=True):
        np.testing.assert_array_equal(before.z, after.z)
        assert _color_bounds(manual, after) == (-.25, .25)
    assert "saturat" in _figure_text(manual).lower()


def test_trace_values_are_bin_steps_with_gap_and_pair_wide_fixed_scale(cca_display_result):
    """CC3 traces remain exact when heatmaps show CC1; bounded load cannot rescale them."""
    result = cca_display_result
    full = plotting.plot_session_cca(result, pair_range=(1, 1), selected_pair=3)
    bounded = plotting.plot_session_cca(result, pair_range=(1, 1), selected_pair=3,
                                        loaded_range_s=(98, 102))
    regional_traces = _canonical_traces(full)
    assert len(regional_traces) == 2
    assert len({trace.line.color for trace in regional_traces}) == 2
    for region, trace in zip(("PFC", "HPC"), regional_traces, strict=True):
        segments = _finite_segments(trace)
        assert [(x[0], x[-1]) for x, _ in segments] == [(98, 102), (112, 116)]
        assert trace.connectgaps is not True
        for window, (x, y) in enumerate(segments):
            midpoints = [100, 114][window] + (result.bin_edges_s[:-1] + result.bin_edges_s[1:]) / 2
            np.testing.assert_allclose(_step_values(trace, x, y, midpoints), result.fit.scores[region][window, :, 2])
        axis_name = "yaxis" + trace.yaxis[1:]
        axis = full.layout[axis_name]
        assert axis.range == bounded.layout[axis_name].range
        pooled = np.concatenate([result.fit.scores[r][:, :, 2].ravel() for r in ("PFC", "HPC")])
        assert axis.range[0] <= pooled.min() and axis.range[1] >= pooled.max()
    assert all(list(trace.y) == [1] for trace in _heatmaps(full))


def test_display_controls_revisions_hover_and_light_caption_do_not_fit_or_mutate(cca_display_result, monkeypatch):
    """Display choices preserve saved values and time zoom while resetting neural row state."""
    result = cca_display_result
    before = {region: values.copy() for region, values in result.fit.scores.items()}
    monkeypatch.setattr(cca, "fit_paired_cca", _forbid_fit)
    first = plotting.plot_session_cca(result, viewport_revision=7)
    changed = plotting.plot_session_cca(result, pair_range=(2, 3), selected_pair=2, show_traces=False,
                                        event_tracks=("LED",), show_context=False, viewport_revision=7)
    assert first.layout.uirevision == changed.layout.uirevision
    assert first.layout.uirevision != plotting.plot_session_cca(result, viewport_revision=8).layout.uirevision
    shifted = replace(result, time_origin_s=result.time_origin_s + 1)
    assert first.layout.uirevision != plotting.plot_session_cca(shifted, viewport_revision=7).layout.uirevision
    first_map, changed_map = _heatmaps(first)[0], _heatmaps(changed)[0]
    assert first.layout["yaxis" + first_map.yaxis[1:]].uirevision != changed.layout["yaxis" + changed_map.yaxis[1:]].uirevision
    assert not _canonical_traces(changed)
    assert first.layout.paper_bgcolor == first.layout.plot_bgcolor == "white"
    assert first.layout.font.color == "black" and first.layout.font.size >= 12
    text = _figure_text(first).lower()
    assert "in-sample" in text and "canonical" in text and "task" in text
    for region, trace in zip(("PFC", "HPC"), _heatmaps(first), strict=True):
        hover = trace.hovertemplate.lower()
        assert region.lower() in hover and result.probe_roles[region] in trace.hovertemplate
        assert "rho" in hover or "correlation" in hover
        assert "utc" in hover and "relative" in hover
        # Explicit numeric formatting avoids Plotly prepending the CC tick label
        # to the already present CC prefix in the canonical-pair hover label.
        assert "CC%{y:.0f}" in trace.hovertemplate
        np.testing.assert_array_equal(result.fit.scores[region], before[region])


def test_behavior_only_figure_retains_unpaired_choices(cca_display_result):
    """Missing paired fit still displays all behavior without inventing neural cells."""
    result = replace(cca_display_result, fit=None, unavailable_reason="HPC population unavailable")
    figure = plotting.plot_session_cca(result, event_tracks=("LED",))
    assert not _heatmaps(figure) and not _canonical_traces(figure)
    choices = [trace for trace in figure.data if trace.name in ("Left choices", "Right choices")]
    assert sorted(float(x) for trace in choices for x in trace.x) == [100, 106, 114]
    assert "unavailable" in _figure_text(figure).lower()


def _forbid_fit(*args, **kwargs):
    """Fail if plotting attempts computational work; accept any API arguments."""
    pytest.fail("CCA display must consume saved scores without fitting")


def _heatmaps(figure):
    """Return ordered PFC/HPC heatmap traces from a Plotly Figure."""
    return [trace for trace in figure.data if trace.type == "heatmap"]


def _canonical_traces(figure):
    """Return the two labeled canonical line traces, excluding behavior/navigation."""
    return [trace for trace in figure.data if trace.type == "scatter" and "lines" in (trace.mode or "")
            and any(region in (trace.name or "") for region in ("PFC", "HPC"))]


def _color_bounds(figure, trace):
    """Return symmetric score limits from a heatmap's local or shared color axis."""
    if trace.coloraxis:
        axis = figure.layout[trace.coloraxis]
        return axis.cmin, axis.cmax
    return trace.zmin, trace.zmax


def _colorbar_count(figure):
    """Count visible independent colorbars, counting a shared color axis only once."""
    axes = {trace.coloraxis for trace in _heatmaps(figure) if trace.coloraxis}
    return (sum(figure.layout[axis].showscale is not False for axis in axes)
            + sum(trace.showscale is not False for trace in _heatmaps(figure) if not trace.coloraxis))


def _figure_text(figure):
    """Collect title, annotation and axis-title strings without serializing arrays."""
    return " ".join([figure.layout.title.text or "",
                     *(annotation.text or "" for annotation in figure.layout.annotations),
                     *(axis.title.text or "" for axis in figure.select_yaxes())])


def _finite_segments(trace):
    """Split elapsed-second/score line points at explicit None/NaN gap markers."""
    x, y = np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    boundaries = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return [(x[start:end], y[start:end]) for start, end in boundaries.reshape(-1, 2)]


def _step_values(trace, x, y, midpoints):
    """Evaluate Plotly hv/vh or explicit repeated-edge steps at bin midpoints."""
    if trace.line.shape == "hv":
        return y[np.searchsorted(x, midpoints, side="right") - 1]
    if trace.line.shape == "vh":
        return y[np.searchsorted(x, midpoints, side="left")]
    assert trace.line.shape in (None, "linear")
    return np.interp(midpoints, x, y)
