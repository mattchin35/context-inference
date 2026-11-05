"""WP6a saved-only scientific timeline, inspection and opaque PNG export contracts."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from functools import partial

import matplotlib.figure
import matplotlib.image
import matplotlib.text
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.neural_analysis.phase_synchrony import computation, preparation, results, plotting
from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.tests.neural_analysis.phase_synchrony.test_results import scientific_records, identity_for
from src.tests.neural_analysis.phase_synchrony.test_phase_metrics import UTC_ORIGIN, make_native_source


@pytest.fixture
def saved_view(tmp_path, scientific_records):
    """Return behavior-only timeline and actually saved/loaded typed products with original UTC IDs."""
    config, choices, records = scientific_records
    choices = choices.assign(action=[0, 1], reward=[1, 0], elapsed_s=choices.choice_time - UTC_ORIGIN)
    events = pd.DataFrame({"Event": ["trial_start", "left_entry", "LED_on", "LED_off"],
                           "Time": UTC_ORIGIN + np.array([1, 2, 2.5, 3]), "elapsed_s": [1, 2, 2.5, 3]})
    behavior = SessionTimelineResult("synthetic-view", UTC_ORIGIN, choices, events, {}, {}, {})
    products = []
    for family, record in records.items():
        if family == "power":
            record = replace(record, band_power_integrated=np.array([[2., 10.], [200., 400.]]))
        path = results.save_product(tmp_path / f"{family}.npz", record, identity_for(family, config, records),
                                    run_id="source-run", completed_utc="2030-01-02T03:04:05Z")
        products.append(results.load_product(path))
    return behavior, tuple(products)


def _metric_traces(figure, family=None, metric=None):
    """Return scientific scalar traces tagged by categorical family/metric, excluding behavior."""
    return [trace for trace in figure.data if isinstance(trace.meta, dict)
            and trace.meta.get("kind") == "metric"
            and (family is None or trace.meta.get("family") == family)
            and (metric is None or trace.meta.get("metric") == metric)]


def _find_product(products, family):
    """Return the sole saved typed product for a categorical family in this fixture."""
    return next(product for product in products if product.meta["identity"]["family"] == family)


def _custom_column(trace, name):
    """Read a named semantic hover field from trace customdata, preserving original scalar types."""
    fields = trace.meta["customdata_fields"]
    return np.asarray(trace.customdata, dtype=object)[:, fields.index(name)]


def _axis(figure, trace):
    """Return the Plotly y-axis object associated with a Cartesian trace."""
    return figure.layout["yaxis" + (trace.yaxis or "y")[1:]]


def _texts(figure):
    """Join readable caption/annotation text from a Plotly figure, with no numerical work."""
    return " ".join([str(figure.layout.meta.get("caption", ""))]
                    + [annotation.text for annotation in figure.layout.annotations])


def test_behavior_only_light_timeline_preserves_navigation_and_existing_events(saved_view):
    """No neural products still gives original left/right choices, optional events and persistent time axes."""
    behavior, _ = saved_view
    figure = plotting.plot_synchrony_timeline(behavior, (), event_tracks=("left_entry", "LED"))
    assert isinstance(figure, go.Figure)
    right = next(trace for trace in figure.data if trace.name == "Right choices")
    left = next(trace for trace in figure.data if trace.name == "Left choices")
    np.testing.assert_array_equal(right.x, behavior.choices.elapsed_s.iloc[:1])
    np.testing.assert_array_equal(left.x, behavior.choices.elapsed_s.iloc[1:])
    assert "Left licks" in [trace.name for trace in figure.data]
    assert figure.layout.paper_bgcolor == figure.layout.plot_bgcolor == "white"
    assert figure.layout.font.color == "black" and figure.layout.font.size >= 14
    assert figure.layout.meta["primary_time_axis"] in figure.layout.meta["time_axes"]
    assert "unavailable" in _texts(figure).lower() or "not prepared" in _texts(figure).lower()


def test_scalar_tracks_equal_saved_values_at_real_choices_and_identify_window_sources(saved_view):
    """Every site/pair/unit band is a marker-only saved scalar with true UTC interval/count provenance."""
    behavior, products = saved_view
    figure = plotting.plot_synchrony_timeline(behavior, products)
    expected_fields = {"power": ("band_power_integrated", "power"), "coherence": ("band_mean", "coherence"),
                       "lfp-phase": ("band_plv", "plv"), "spike-ppc": ("band_ppc", "ppc")}
    for family, (field, metric) in expected_fields.items():
        product = _find_product(products, family)
        traces = _metric_traces(figure, family, metric)
        assert len(traces) == len(product.record.band_names)
        for trace in traces:
            index = product.record.band_names.index(trace.meta["band"])
            np.testing.assert_array_equal(trace.x, product.record.choice_times_s - behavior.time_origin_s)
            np.testing.assert_allclose(trace.y, getattr(product.record, field)[:, index], equal_nan=True)
            assert trace.mode == "markers"
            np.testing.assert_array_equal(_custom_column(trace, "trial_id"), behavior.choices.trial_id)
            np.testing.assert_array_equal(_custom_column(trace, "row_position"), behavior.choices.row_position)
            np.testing.assert_allclose(_custom_column(trace, "interval_start_s").astype(float), product.record.choice_times_s - 1, rtol=0, atol=1e-7)
            np.testing.assert_allclose(_custom_column(trace, "interval_stop_s").astype(float), product.record.choice_times_s + 1, rtol=0, atol=1e-7)
            assert set(_custom_column(trace, "source_run_id")) == {"source-run"}
            assert set(_custom_column(trace, "completed_utc")) == {"2030-01-02T03:04:05Z"}
            assert "source_details" in trace.meta["customdata_fields"]
            if family == "power":
                assert "uV^2" in _axis(figure, trace).title.text
                np.testing.assert_array_equal(_custom_column(trace, "segment_count"), product.record.segment_count)
                np.testing.assert_array_equal(_custom_column(trace, "bandwidth_hz"), [product.record.bandwidth_hz[index]] * 2)
    assert "window" in _texts(figure).lower() and "[-1" in _texts(figure)


def test_power_scales_and_time_uirevision_stay_fixed_while_panning_and_toggling(saved_view):
    """Viewport/event/metric controls preserve saved values and full-product physical scales; explicit revision navigates."""
    behavior, products = saved_view
    before = deepcopy(_find_product(products, "power").record.band_power_integrated)
    first = plotting.plot_synchrony_timeline(behavior, products, time_range_s=(1, 3), viewport_revision=7)
    moved = plotting.plot_synchrony_timeline(behavior, products, time_range_s=(4, 7), viewport_revision=7,
                                            event_tracks=("left_entry",), phase_metric="ppc")
    assert first.layout.uirevision == moved.layout.uirevision
    changed = plotting.plot_synchrony_timeline(behavior, products, viewport_revision=8)
    assert changed.layout.uirevision != first.layout.uirevision
    for a, b in zip(_metric_traces(first, "power"), _metric_traces(moved, "power")):
        assert tuple(_axis(first, a).range) == tuple(_axis(moved, b).range)
        assert _axis(first, a).range[0] == 0
        assert _axis(first, a).range[1] >= np.nanmax(before[:, (0 if a.meta["band"] == "theta" else 1)])
    for trace in _metric_traces(first, "coherence") + _metric_traces(first, "lfp-phase", "plv"):
        assert tuple(_axis(first, trace).range) == (0, 1)
    assert _metric_traces(moved, "lfp-phase", "ppc")
    np.testing.assert_array_equal(_find_product(products, "power").record.band_power_integrated, before)


def test_frequency_heatmap_uses_actual_interval_edges_and_blank_gap_columns(saved_view):
    """Each heatmap column is a whole-window scalar; explicit edges leave real inter-choice gaps blank."""
    behavior, products = saved_view
    product = _find_product(products, "coherence")
    figure = plotting.plot_synchrony_timeline(behavior, products, heatmap_selection=("coherence", ("SiteA", "SiteB")))
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap" and trace.meta.get("kind") == "frequency-heatmap")
    np.testing.assert_allclose(heatmap.x, [1.125, 3.125, 4.125, 6.125], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(heatmap.y, product.record.frequency_hz)
    values = np.asarray(heatmap.z, dtype=float)
    assert values.shape == (len(product.record.frequency_hz), 3) and np.isnan(values[:, 1]).all()
    excluded = np.isin(product.record.frequency_hz, [58, 60, 62])
    assert np.isnan(values[excluded]).all()
    np.testing.assert_allclose(values[~excluded][:, [0, 2]], product.record.coherence[:, ~excluded].T, equal_nan=True)
    assert heatmap.zsmooth is False and heatmap.connectgaps is False
    assert heatmap.zmin == 0 and heatmap.zmax == 1
    assert "window" in heatmap.hovertemplate.lower()


def test_frequency_heatmap_hover_binds_each_tile_to_original_trial_support_and_source(saved_view):
    """A scalar tile retains IDs, actual UTC bounds/counts and completed-source provenance in hover."""
    behavior, products = saved_view
    figure = plotting.plot_synchrony_timeline(behavior, products, heatmap_selection=("coherence", ("SiteA", "SiteB")))
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
    fields = heatmap.meta["customdata_fields"]
    required = {"trial_id", "row_position", "choice_time_s", "interval_start_s", "interval_stop_s",
                "sample_count", "segment_count", "source_run_id", "completed_utc", "source_details"}
    assert required.issubset(fields)
    data = np.asarray(heatmap.customdata, dtype=object)
    assert data.shape[:2] == np.asarray(heatmap.z).shape
    for column, trial in ((0, 0), (2, 1)):
        assert data[0, column, fields.index("trial_id")] == behavior.choices.trial_id.iloc[trial]
        assert data[0, column, fields.index("row_position")] == behavior.choices.row_position.iloc[trial]
        assert data[0, column, fields.index("source_run_id")] == "source-run"
    assert "customdata" in heatmap.hovertemplate and "UTC" in heatmap.hovertemplate


def test_frequency_heatmap_honors_saved_validity_even_for_finite_stored_values(saved_view):
    """Invalid saved support is unavailable; a finite numeric buffer must not override its explicit mask."""
    behavior, products = saved_view
    product = _find_product(products, "coherence")
    valid = product.record.valid.copy()
    frequency = int(np.flatnonzero(product.record.frequency_hz == 8)[0])
    assert np.isfinite(product.record.coherence[0, frequency])
    valid[0, frequency] = False
    product = replace(product, record=replace(product.record, valid=valid))
    figure = plotting.plot_synchrony_timeline(behavior, (product,), heatmap_selection=("coherence", ("SiteA", "SiteB")))
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap")
    assert np.isnan(np.asarray(heatmap.z)[frequency, 0])


def test_lfp_band_hover_counts_only_retained_band_frequencies(saved_view, scientific_records):
    """Unsupported 2-Hz samples cannot change the support count shown for a fully supported theta band."""
    behavior, products = saved_view
    config, choices, records = scientific_records
    config = replace(config, phase_frequencies_hz=(2., 6., 8., 10.))
    phase = preparation.prepare_site_phase(make_native_source(offset_s=-5, duration_s=18), choices, config)
    validity = phase.valid.copy()
    validity[0] = False
    phase = replace(phase, valid=validity)
    record = computation.compute_lfp_phase_summary(phase, replace(phase, site_id="SiteB"), config)
    source = _find_product(products, "lfp-phase")
    identity = results.product_identity("lfp-phase", ("SiteA", "SiteB"), "primary", config,
                                       source_ids=source.meta["identity"]["source_ids"],
                                       preparation_ids=(results.preparation_identity(phase),))
    path = results.save_product(source.path.with_name("band-support.npz"), record, identity,
                               run_id="band-support", completed_utc="2030-01-02T03:04:05Z")
    figure = plotting.plot_synchrony_timeline(behavior, (results.load_product(path),))
    theta = next(trace for trace in _metric_traces(figure, "lfp-phase") if trace.meta["band"] == "theta")
    assert theta.y[0] == pytest.approx(1)
    assert str(_custom_column(theta, "sample_count")[0]) in {"1000", "1000-1000"}


def test_overlapping_choice_windows_keep_each_true_interval(saved_view):
    """Overlapping windows need separate tiles rather than fabricated monotonic shared edges."""
    behavior, products = saved_view
    product = _find_product(products, "coherence")
    times = product.record.choice_times_s.copy()
    times[1] = times[0] + 0.5
    choices = behavior.choices.copy()
    choices["choice_time"] = times
    choices["elapsed_s"] = times - behavior.time_origin_s
    behavior = replace(behavior, choices=choices)
    product = replace(product, record=replace(product.record, choice_times_s=times))
    figure = plotting.plot_synchrony_timeline(
        behavior, (product,), heatmap_selection=("coherence", ("SiteA", "SiteB")),
    )
    tiles = [trace for trace in figure.data if trace.type == "heatmap"
             and trace.meta.get("kind") == "frequency-heatmap"]
    intervals = []
    for tile in tiles:
        edges = np.asarray(tile.x, dtype=float)
        values = np.asarray(tile.z, dtype=float)
        for column in range(values.shape[1]):
            if np.isfinite(values[:, column]).any():
                intervals.append((edges[column], edges[column + 1]))
    expected = [(time - behavior.time_origin_s - 1, time - behavior.time_origin_s + 1) for time in times]
    np.testing.assert_allclose(sorted(intervals), sorted(expected), rtol=0, atol=1e-7)


@pytest.mark.parametrize("changed", ["trial_id", "row_position", "utc", "origin"])
def test_original_trial_identity_and_absolute_time_mismatch_cannot_be_plotted(saved_view, changed):
    """Same trial count does not justify joining different row IDs/choices or silently shifting the UTC origin."""
    behavior, products = saved_view
    power = _find_product(products, "power")
    if changed == "origin":
        behavior = replace(behavior, time_origin_s=behavior.time_origin_s + 10)
    else:
        field = {"trial_id": "trial_ids", "row_position": "row_positions", "utc": "choice_times_s"}[changed]
        values = getattr(power.record, field).copy()
        values[0] += 0.001 if changed == "utc" else 1
        power = replace(power, record=replace(power.record, **{field: values}))
    with pytest.raises(ValueError, match="trial|row|UTC|time|origin|identity"):
        plotting.plot_synchrony_timeline(behavior, (power,))


def test_selected_choice_inspection_uses_saved_waveform_phase_spikes_and_polar_counts(saved_view):
    """Original row_position selects the same saved choice; inspection phase excludes the metric guard."""
    behavior, products = saved_view
    figure = plotting.plot_choice_inspection(behavior, products, row_position=2, frequency_hz=8, site_id="SiteA", unit_id="ProbeA:17")
    traces = {trace.meta["kind"]: trace for trace in figure.data if isinstance(trace.meta, dict) and "kind" in trace.meta}
    waveform = _find_product(products, "waveforms").record
    phase = _find_product(products, "phase").record
    spike = _find_product(products, "spike-ppc").record
    np.testing.assert_array_equal(traces["waveform"].x, waveform.relative_time_s)
    np.testing.assert_array_equal(traces["waveform"].y, waveform.values_uv[0])
    np.testing.assert_array_equal(traces["phase"].x, phase.relative_time_s[phase.metric_mask])
    np.testing.assert_allclose(traces["phase"].y, np.angle(phase.phase[0, 0, phase.metric_mask]), equal_nan=True)
    assert len(traces["phase"].x) == 1000 and max(traces["phase"].x) < 1
    np.testing.assert_allclose(traces["spikes"].x, spike.spike_times_by_trial[0] - spike.choice_times_s[0], rtol=0, atol=1e-7)
    np.testing.assert_array_equal(traces["polar-histogram"].r, spike.polar_histogram_count[0, 0])
    assert "8" in _texts(figure) and "21" in _texts(figure)
    assert "PPC" in _texts(figure) and "spike" in _texts(figure).lower()
    assert phase.probe_id in _texts(figure) and "channel" in _texts(figure).lower()


def test_missing_inspection_inputs_and_secondary_products_remain_explicitly_unavailable(saved_view):
    """Partial saved products never trigger analysis or silently substitute a different interval/frequency."""
    behavior, products = saved_view
    power = _find_product(products, "power")
    inspection = plotting.plot_choice_inspection(behavior, (power,), row_position=2, frequency_hz=8)
    assert "unavailable" in _texts(inspection).lower()
    secondary = plotting.plot_synchrony_timeline(behavior, products, interval="secondary")
    assert not _metric_traces(secondary)
    assert "secondary" in _texts(secondary).lower() and "unavailable" in _texts(secondary).lower()


def test_spike_hover_reports_frequency_count_range_and_signed_ppc(saved_view, scientific_records):
    """Frequency-specific 2/3/2 counts are exposed as range 2-3, never summed to seven."""
    behavior, products = saved_view
    config, choices, records = scientific_records
    config = replace(config, phase_frequencies_hz=(6., 8., 10.))
    phase = preparation.prepare_site_phase(records["native-lfp"], choices, config)
    mask = phase.valid.copy()
    mask[0, 0, 562:564] = False
    mask[2, 0, 500] = False
    phase = replace(phase, valid=mask)
    unit = results.SelectedUnitSpikes("ProbeA:17", "ProbeA", 17, phase.choice_times_s[0] + np.array([0, .0625, .125]), {})
    record = computation.compute_selected_unit_ppc(unit, phase, config)
    source = _find_product(products, "spike-ppc")
    identity = results.product_identity("spike-ppc", ("ProbeA:17", "SiteA"), "primary", config,
                                        source_ids=source.meta["identity"]["source_ids"],
                                        preparation_ids=(results.preparation_identity(unit), results.preparation_identity(phase)))
    path = results.save_product(source.path.with_name("variable-counts.npz"), record, identity,
                                run_id="variable-counts", completed_utc="2030-01-02T03:04:05Z")
    product = results.load_product(path)
    figure = plotting.plot_synchrony_timeline(behavior, (product,))
    trace = next(trace for trace in _metric_traces(figure, "spike-ppc") if trace.meta["band"] == "theta")
    assert _custom_column(trace, "spike_count_min")[0] == 2
    assert _custom_column(trace, "spike_count_max")[0] == 3
    assert trace.y[0] == record.band_ppc[0, 0] < 0
    assert _axis(figure, trace).range[0] <= trace.y[0]


def _forbid_work(*args, **kwargs):
    """Reject any raw loading/numerical computation arguments during saved-only plotting/export."""
    raise AssertionError("Plotting requested analysis or raw signal inputs")


def test_plotting_never_loads_raw_inputs_or_calls_numerical_analysis(saved_view, monkeypatch):
    """Timeline and selected-choice figures operate on supplied loaded records only."""
    behavior, products = saved_view
    for name in ("load_site_lfp", "prepare_site_phase", "load_selected_unit_spikes"):
        monkeypatch.setattr(preparation, name, _forbid_work)
    for name in ("compute_site_psd", "compute_band_power", "compute_welch_coherence", "compute_lfp_phase_summary", "compute_selected_unit_ppc"):
        monkeypatch.setattr(computation, name, _forbid_work)
    plotting.plot_synchrony_timeline(behavior, products)
    plotting.plot_choice_inspection(behavior, products, row_position=2, frequency_hz=8)


def _capture_export_text(captured, original, figure, *args, **kwargs):
    """Capture actual Matplotlib rendered text while saving the real figure to its requested PNG."""
    captured.append(" ".join(text.get_text() for text in figure.findobj(matplotlib.text.Text)))
    return original(figure, *args, **kwargs)


def test_validation_export_writes_captioned_opaque_white_pngs_without_kaleido_or_overwrite(saved_view, tmp_path, monkeypatch):
    """Actual Agg exports contain readable content/captions under run/figures and retain immutable old images."""
    behavior, products = saved_view
    monkeypatch.setattr(go.Figure, "write_image", _forbid_work)
    for name in ("load_site_lfp", "prepare_site_phase", "load_selected_unit_spikes"):
        monkeypatch.setattr(preparation, name, _forbid_work)
    for name in ("compute_site_psd", "compute_band_power", "compute_welch_coherence", "compute_lfp_phase_summary", "compute_selected_unit_ppc"):
        monkeypatch.setattr(computation, name, _forbid_work)
    captured = []
    original = matplotlib.figure.Figure.savefig
    with patch.object(matplotlib.figure.Figure, "savefig", autospec=True,
                      side_effect=partial(_capture_export_text, captured, original)):
        paths = plotting.export_validation_figures(behavior, products, tmp_path / "run", row_position=2, frequency_hz=8)
    assert len(paths) >= 2 and all(path.parent == tmp_path / "run" / "figures" for path in paths)
    for path in paths:
        assert path.suffix == ".png"
        pixels = matplotlib.image.imread(path)
        assert pixels.shape[0] >= 400 and pixels.shape[1] >= 600
        assert np.min(pixels[..., :3]) < 0.5
        np.testing.assert_allclose(pixels[0, 0, :3], [1, 1, 1], rtol=0, atol=0)
        if pixels.shape[2] == 4:
            np.testing.assert_allclose(pixels[..., 3], 1, rtol=0, atol=0)
    assert captured and all("primary" in text.lower() and "[-1" in text for text in captured)
    assert "SiteA" in captured[0] and "theta" in captured[0] and "gamma" in captured[0]
    assert "8 Hz" in captured[1] and "ProbeA:17" in captured[1]
    before = {path: path.read_bytes() for path in paths}
    with pytest.raises(FileExistsError):
        plotting.export_validation_figures(behavior, products, tmp_path / "run", row_position=2, frequency_hz=8)
    assert all(path.read_bytes() == content for path, content in before.items())
