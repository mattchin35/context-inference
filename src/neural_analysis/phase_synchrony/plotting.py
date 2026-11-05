"""Saved-only choice-window figures and captioned opaque Matplotlib PNG exports."""

import json
from pathlib import Path
import re
import textwrap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure as MatplotlibFigure
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.population import timeline_plotting as behavior_plotting
from src.neural_analysis.population.session_timeline import SessionTimelineResult

from .results import INTERVAL_BOUNDS_S, SavedProduct


METRIC_FAMILIES = ("power", "coherence", "lfp-phase", "spike-ppc")
CUSTOM_FIELDS = ["trial_id", "row_position", "choice_time_s", "interval_start_s", "interval_stop_s",
                 "sample_count", "segment_count", "bandwidth_hz", "spike_count_min", "spike_count_max",
                 "source_run_id", "completed_utc", "source_details", "valid"]


def plot_synchrony_timeline(
    behavior: SessionTimelineResult,
    products: tuple[SavedProduct, ...],
    *, interval: str = "primary", phase_metric: str = "plv",
    time_range_s: tuple[float, float] | None = None, viewport_revision: int = 0,
    event_tracks: tuple[str, ...] = (), show_context: bool = True,
    heatmap_selection: tuple[str, tuple[str, ...]] | None = None,
) -> go.Figure:
    """Plot saved per-choice scalar results on the original elapsed-session clock.

    Behavior has original trial rows/IDs, UTC seconds and first-trial origin;
    products are loaded typed artifacts, never paths to raw data. interval is
    primary [-1,+1) or secondary [-2,+2) seconds; phase_metric is plv/ppc.
    Optional viewport uses elapsed seconds. Return a Plotly Figure with separate
    fixed physical uV^2 power axes and dimensionless pair/unit axes. Metric
    traces are marker-only; all scales use full products independently of pan.
    Optional (family,selection) heatmap uses actual interval edges/Hz and blank
    gaps, or separate true tiles when intervals overlap. No analysis or file
    reads occur. Time-axis uirevision changes only with explicit navigation.
    """
    if phase_metric not in {"plv", "ppc"}:
        raise ValueError("phase_metric must be plv or ppc")
    selected = _validated_products(behavior, tuple(products), interval)
    rows = _metric_rows(selected, phase_metric)
    tracks = tuple(track for track in event_tracks if track in behavior_plotting.available_event_tracks(behavior.events))
    heatmap = next((product for product in selected if heatmap_selection is not None
                    and product.meta["identity"]["family"] == heatmap_selection[0]
                    and tuple(product.meta["identity"]["selection"]) == tuple(heatmap_selection[1])), None)
    titles = ["Choices"] + [behavior_plotting.EVENT_LABELS[track] for track in tracks]
    titles.extend(row["title"] for row in rows)
    if heatmap is not None:
        titles.append("Frequency-by-choice window summaries")
    titles.append("Session navigator")
    figure = make_subplots(rows=len(titles), cols=1, shared_xaxes=True, subplot_titles=titles,
                           vertical_spacing=min(.025, .15 / len(titles)))
    behavior_plotting.add_timeline_behavior(figure, behavior, tracks, show_context)
    first_metric_row = 2 + len(tracks)
    for row_index, row in enumerate(rows, start=first_metric_row):
        product, band = row["product"], row["band"]
        record = product.record
        values = np.asarray(row["values"], dtype=float).copy()
        valid = np.asarray(record.valid[:, band] if row["family"] == "power" else record.band_valid[:, band])
        values[~valid] = np.nan
        figure.add_trace(go.Scattergl(
            x=record.choice_times_s - behavior.time_origin_s, y=values, mode="markers", name=row["title"],
            marker={"size": 8}, customdata=_metric_customdata(product, band, selected),
            meta={"kind": "metric", "family": row["family"], "selection": product.meta["identity"]["selection"],
                  "band": record.band_names[band], "metric": row["metric"], "customdata_fields": CUSTOM_FIELDS,
                  "source_settings": product.meta["identity"]["parameters"]},
            hovertemplate=_metric_hover(row), connectgaps=False,
        ), row=row_index, col=1)
        figure.update_yaxes(title_text=row["unit"], range=_metric_limits(row["values"], row["metric"]),
                            autorange=False, row=row_index, col=1)
    if heatmap is not None:
        _add_frequency_heatmap(figure, heatmap, behavior.time_origin_s,
                               first_metric_row + len(rows), phase_metric, selected)
    figure.add_trace(go.Scattergl(x=behavior.choices.elapsed_s, y=np.zeros(len(behavior.choices)), mode="markers",
                                  marker={"size": 3, "color": "#777777"}, name="Session navigator"), row=len(titles), col=1)
    bounds = behavior_plotting.session_time_bounds(behavior)
    visible = time_range_s or bounds
    if len(visible) != 2 or not np.isfinite(visible).all() or visible[0] >= visible[1]:
        raise ValueError("time_range_s must contain increasing finite elapsed seconds")
    figure.update_xaxes(range=visible, minallowed=bounds[0], maxallowed=bounds[1])
    figure.update_xaxes(title_text="Elapsed session time (s; zero = first trial start)",
                        rangeslider={"visible": True, "range": bounds}, row=len(titles), col=1)
    missing = [family for family in METRIC_FAMILIES if not any(row["family"] == family for row in rows)]
    caption = _caption(interval) + (" Unavailable saved tracks: " + ", ".join(missing) + "." if missing else "")
    _light_layout(figure, caption, height=max(600, 140 * len(titles) + 200))
    figure.update_layout(uirevision=f"{behavior.session_id}:{behavior.time_origin_s}:{viewport_revision}",
                         meta={"caption": caption, "interval": interval, "unavailable": missing,
                               "time_axes": ["xaxis" + (str(index) if index > 1 else "") for index in range(1, len(titles) + 1)],
                               "primary_time_axis": "xaxis"})
    return figure


def plot_choice_inspection(
    behavior: SessionTimelineResult, products: tuple[SavedProduct, ...], row_position: int,
    *, frequency_hz: float = 8.0, interval: str = "primary", site_id: str | None = None,
    unit_id: str | None = None,
) -> go.Figure:
    """Plot one original row's saved waveform, identified phase, spikes and polar counts.

    Input products retain their typed (trial,time)/(frequency,trial,time) arrays,
    calibrated uV, unit phase, UTC spikes and fixed-bin counts. row_position is
    the original source row, not trial ordinal or trial ID. Frequency is exact
    saved Hz, never silently substituted. Return a Plotly Figure using relative
    seconds for waveform/phase/spikes and degrees for the polar display. Real
    phase guards are excluded from inspection metrics; missing inputs/frequency
    are explicitly unavailable. No raw reading, transform or PPC calculation.
    """
    selected = _validated_products(behavior, tuple(products), interval)
    choices = behavior.choices.loc[behavior.choices.row_position == row_position]
    if len(choices) != 1:
        raise ValueError("Unknown original trial row_position")
    trial_id = choices.trial_id.iloc[0]
    site_id = site_id or next((getattr(product.record, "site_id", None) for product in selected
                              if product.meta["identity"]["family"] in {"phase", "waveforms", "spike-ppc"}), None)
    figure = make_subplots(rows=4, cols=1, specs=[[{}], [{}], [{}], [{"type": "polar"}]],
                           subplot_titles=["Saved LFP (uV)", f"Saved phase at {frequency_hz:g} Hz (rad)",
                                           "Selected-unit UTC spikes", "Saved spike-phase histogram"],
                           vertical_spacing=.07)
    unavailable = []
    ppc_text = "PPC unavailable"
    for family, kind in (("waveforms", "waveform"), ("phase", "phase"), ("spike-ppc", "spikes")):
        product = next((product for product in selected if product.meta["identity"]["family"] == family
                        and getattr(product.record, "site_id", None) == site_id
                        and (family != "spike-ppc" or unit_id is None or product.record.unit_id == unit_id)), None)
        if product is None:
            unavailable.append(kind)
            continue
        record = product.record
        trial = int(np.flatnonzero(record.row_positions == row_position)[0])
        if family == "waveforms":
            values = np.where(record.valid[trial], record.values_uv[trial], np.nan)
            figure.add_trace(go.Scattergl(x=record.relative_time_s, y=values, mode="lines", name=f"{site_id} LFP",
                                          connectgaps=False, meta={"kind": kind}), row=1, col=1)
        elif family == "phase":
            frequency = _frequency_index(record.frequency_hz, frequency_hz)
            if frequency is None:
                unavailable.append(f"phase at {frequency_hz:g} Hz")
                continue
            mask = record.metric_mask
            values = np.where(record.valid[frequency, trial, mask], np.angle(record.phase[frequency, trial, mask]), np.nan)
            figure.add_trace(go.Scattergl(x=record.relative_time_s[mask], y=values, mode="lines", name=f"{frequency_hz:g} Hz phase",
                                          connectgaps=False, meta={"kind": kind}), row=2, col=1)
        else:
            spikes = record.spike_times_by_trial[trial] - record.choice_times_s[trial]
            figure.add_trace(go.Scattergl(x=spikes, y=np.zeros(len(spikes)), mode="markers", name=record.unit_id,
                                          marker={"symbol": "line-ns", "size": 15}, meta={"kind": kind}), row=3, col=1)
            frequency = _frequency_index(record.frequency_hz, frequency_hz)
            if frequency is None:
                unavailable.append(f"polar histogram at {frequency_hz:g} Hz")
                continue
            edges = record.phase_bin_edges_rad
            figure.add_trace(go.Barpolar(r=record.polar_histogram_count[frequency, trial],
                                         theta=np.rad2deg((edges[:-1] + edges[1:]) / 2), width=np.rad2deg(np.diff(edges)),
                                         name="Spike-phase counts", meta={"kind": "polar-histogram"}), row=4, col=1)
            ppc_text = (f"{record.unit_id}: PPC {record.ppc[trial, frequency]:.5g}; "
                        f"{record.spike_count[trial, frequency]} valid spikes, "
                        f"{record.excluded_spike_count[trial, frequency]} excluded")
    bounds = INTERVAL_BOUNDS_S[interval]
    for row in (1, 2, 3):
        figure.update_xaxes(range=bounds, title_text="Time relative to choice (s)", row=row, col=1)
    figure.update_yaxes(title_text="uV", row=1, col=1)
    figure.update_yaxes(title_text="Phase (rad)", range=(-np.pi, np.pi), row=2, col=1)
    figure.update_yaxes(title_text="Spikes", row=3, col=1)
    caption = (f"Trial {trial_id}, original row {row_position}; {_site_label(site_id, selected)}, "
               f"{frequency_hz:g} Hz. " + _caption(interval) + " " + ppc_text + ".")
    if unavailable:
        caption += " Unavailable: " + ", ".join(unavailable) + "."
    _light_layout(figure, caption, height=1100)
    figure.update_layout(meta={"caption": caption, "unavailable": unavailable,
                               "time_axes": ["xaxis", "xaxis2", "xaxis3"], "primary_time_axis": "xaxis"},
                         uirevision=f"{behavior.session_id}:inspection:{row_position}:{interval}:{frequency_hz}")
    return figure


def export_validation_figures(
    behavior: SessionTimelineResult, products: tuple[SavedProduct, ...], run_directory: Path | str,
    *, row_position: int, frequency_hz: float = 8.0, interval: str = "primary",
    site_id: str | None = None, unit_id: str | None = None,
) -> tuple[Path, ...]:
    """Export saved overview/inspection data as captioned opaque-white PNGs via Agg.

    Input axes/units are identical to the two interactive functions; run_directory
    is a filesystem run path. Return two Paths under run/figures. Preflight all
    target names before creating any output and refuse existing files. Rendering
    uses actual prepared figure data with Matplotlib, never Kaleido/browser image
    export, raw sources or numerical analysis. Timing is the caller's export stage.
    """
    directory = Path(run_directory) / "figures"
    paths = (directory / f"timeline_{interval}.png", directory / f"choice_row_{row_position}_{frequency_hz:g}Hz_{interval}.png")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    products = tuple(products)
    figures = (plot_synchrony_timeline(behavior, products, interval=interval),
               plot_choice_inspection(behavior, products, row_position, frequency_hz=frequency_hz,
                                      interval=interval, site_id=site_id, unit_id=unit_id))
    directory.mkdir(parents=True, exist_ok=True)
    for figure, path in zip(figures, paths):
        _save_matplotlib_figure(figure, path)
    return paths


# Saved-data presentation helpers; none imports or invokes scientific analysis.
def _validated_products(behavior: SessionTimelineResult, products: tuple, interval: str) -> tuple:
    """Check original row/ID/UTC joins and common origin; return products for the requested seconds interval."""
    if interval not in INTERVAL_BOUNDS_S:
        raise ValueError("interval must be primary or secondary")
    choices = behavior.choices
    if (not np.isfinite(behavior.time_origin_s) or choices.row_position.duplicated().any()
            or not np.array_equal(choices.elapsed_s.to_numpy(), choices.choice_time.to_numpy() - behavior.time_origin_s)):
        raise ValueError("Behavior UTC origin/elapsed time or original trial row identity is inconsistent")
    by_row = choices.set_index("row_position")
    selected = []
    for product in products:
        if product.meta["identity"]["interval"] != interval:
            continue
        record = product.record
        if hasattr(record, "row_positions"):
            try:
                corresponding = by_row.loc[record.row_positions]
            except KeyError as error:
                raise ValueError("Saved product contains an unknown original trial row") from error
            if (not np.array_equal(record.trial_ids, corresponding.trial_id.to_numpy())
                    or not np.array_equal(record.choice_times_s, corresponding.choice_time.to_numpy())):
                raise ValueError("Saved product original trial ID/row/UTC time identity does not match behavior")
        selected.append(product)
    return tuple(selected)


def _metric_rows(products: tuple, phase_metric: str) -> list[dict]:
    """Return one marker-row description per saved site/pair/unit band, preserving stored scalars/units."""
    rows = []
    for family in METRIC_FAMILIES:
        for product in products:
            if product.meta["identity"]["family"] != family:
                continue
            record = product.record
            field, metric = {"power": ("band_power_integrated", "power"), "coherence": ("band_mean", "coherence"),
                             "lfp-phase": ("band_plv" if phase_metric == "plv" else "band_ppc", phase_metric),
                             "spike-ppc": ("band_ppc", "ppc")}[family]
            for band, name in enumerate(record.band_names):
                label = " -> ".join(product.meta["identity"]["selection"])
                metric_label = {"power": "Integrated power", "coherence": "Magnitude-squared coherence",
                                "lfp-phase": f"Within-window LFP-LFP {metric.upper()}",
                                "spike-ppc": "Selected-unit spike-LFP PPC"}[family]
                rows.append({"product": product, "family": family, "band": band, "metric": metric,
                             "values": getattr(record, field)[:, band], "unit": "uV^2" if family == "power" else "dimensionless",
                             "title": f"{label} - {name} {metric_label}"})
    return rows


def _metric_limits(values: np.ndarray, metric: str) -> tuple[float, float]:
    """Return stable full-product physical/dimensionless limits; signed PPC is never rectified."""
    finite = np.asarray(values)[np.isfinite(values)]
    if metric == "power":
        return 0.0, float(finite.max()) * 1.05 if finite.size and finite.max() > 0 else 1.0
    if metric == "ppc":
        return min(-1.0, float(finite.min())) if finite.size else -1.0, max(1.0, float(finite.max())) if finite.size else 1.0
    return 0.0, 1.0


def _metric_customdata(product: SavedProduct, band: int, products: tuple) -> np.ndarray:
    """Build (trial,field) hover values with original UTC bounds, physical/support counts and source provenance."""
    record = product.record
    start, stop = INTERVAL_BOUNDS_S[product.meta["identity"]["interval"]]
    data = np.empty((len(record.trial_ids), len(CUSTOM_FIELDS)), dtype=object)
    details = _source_details(product, products)
    family = product.meta["identity"]["family"]
    for trial in range(len(record.trial_ids)):
        sample_count = getattr(record, "sample_count", None)
        count = None if sample_count is None else sample_count[trial]
        if np.ndim(count):
            band_spec = product.meta["identity"]["parameters"]["bands"][band]
            retained = _retained_band_mask(record.frequency_hz, band_spec)
            count = f"{np.min(count[retained])}-{np.max(count[retained])}" if retained.any() else None
        valid = record.valid[trial, band] if family == "power" else record.band_valid[trial, band]
        data[trial] = [record.trial_ids[trial], record.row_positions[trial], record.choice_times_s[trial],
                       record.choice_times_s[trial] + start, record.choice_times_s[trial] + stop, count,
                       record.segment_count[trial] if hasattr(record, "segment_count") else None,
                       record.bandwidth_hz[band] if hasattr(record, "bandwidth_hz") else None,
                       record.band_spike_count_min[trial, band] if hasattr(record, "band_spike_count_min") else None,
                       record.band_spike_count_max[trial, band] if hasattr(record, "band_spike_count_max") else None,
                       product.meta["run_id"], product.meta["completed_utc"], details,
                       bool(valid)]
    return data


def _source_details(product: SavedProduct, products: tuple) -> str:
    """Summarize actual supplied site/probe/channel IDs and effective settings without repeating whole source metadata."""
    sites = []
    selection = product.meta["identity"]["selection"]
    for site, source in product.meta["identity"].get("source_ids", {}).items():
        if isinstance(source, dict) and "probe_id" in source:
            sites.append({"site": site, "probe": source["probe_id"], "saved_channel": source["saved_channel_index"]})
    meta = product.record.meta
    if "site_id" in meta and "probe_id" in meta:
        sites.append({"site": meta["site_id"], "probe": meta["probe_id"], "saved_channel": meta["saved_channel_index"]})
    for site, probe, channel in zip(meta.get("sites", ()), meta.get("probes", ()), meta.get("saved_channels", ())):
        sites.append({"site": site, "probe": probe, "saved_channel": channel})
    for related in products:
        record = related.record
        if hasattr(record, "site_id") and hasattr(record, "probe_id") and record.site_id in selection:
            sites.append({"site": record.site_id, "probe": record.probe_id, "saved_channel": record.saved_channel_index})
    return json.dumps({"selection": product.meta["identity"]["selection"], "sites": sites,
                       "notch_enabled": product.meta["identity"]["parameters"].get("notch_enabled"),
                       "welch": product.record.meta.get("welch")}, sort_keys=True)


def _site_label(site_id: str | None, products: tuple) -> str:
    """Return actual selected site/probe/saved-channel text from supplied records or saved source identity."""
    for product in products:
        record = product.record
        if getattr(record, "site_id", None) == site_id and hasattr(record, "probe_id"):
            return f"{site_id} ({record.probe_id}, saved channel {record.saved_channel_index})"
        source = product.meta["identity"].get("source_ids", {}).get(site_id)
        if isinstance(source, dict) and "probe_id" in source:
            return f"{site_id} ({source['probe_id']}, saved channel {source['saved_channel_index']})"
        meta = record.meta
        if meta.get("site_id") == site_id and "probe_id" in meta:
            return f"{site_id} ({meta['probe_id']}, saved channel {meta['saved_channel_index']})"
    return f"{site_id or 'site unavailable'} (probe/channel unavailable)"


def _retained_band_mask(frequencies_hz: np.ndarray, band: dict) -> np.ndarray:
    """Select saved frequency centers for display support labels, using inclusive configured exclusions."""
    retained = (frequencies_hz >= band["lower_hz"]) & (frequencies_hz <= band["upper_hz"])
    for low, high in band["excluded_intervals_hz"]:
        retained &= ~((frequencies_hz >= low) & (frequencies_hz <= high))
    return retained


def _metric_hover(row: dict) -> str:
    """Return named-field window-summary hover HTML; numerical values remain in customdata."""
    return (f"{row['title']}<br>Window summary; value %{{y:.6g}} {row['unit']}"
            "<br>Trial %{customdata[0]}, original row %{customdata[1]}"
            "<br>UTC [%{customdata[3]:.6f}, %{customdata[4]:.6f}) s; choice %{customdata[2]:.6f}"
            "<br>Samples %{customdata[5]}; segments %{customdata[6]}; retained bandwidth %{customdata[7]} Hz"
            "<br>Valid spike count range %{customdata[8]}-%{customdata[9]} across retained frequencies"
            "<br>Source run %{customdata[10]}, completed %{customdata[11]}<br>%{customdata[12]}<extra></extra>")


def _add_frequency_heatmap(figure: go.Figure, product: SavedProduct, origin: float, row: int,
                           phase_metric: str, products: tuple) -> None:
    """Add actual-Hz scalar window columns; nonoverlap gets NaN gaps, overlap gets independent true tiles."""
    record = product.record
    family = product.meta["identity"]["family"]
    field = "coherence" if family == "coherence" else phase_metric if family == "lfp-phase" else "ppc"
    metric = "coherence" if family == "coherence" else phase_metric if family == "lfp-phase" else "ppc"
    values = np.asarray(getattr(record, field), dtype=float).T.copy()
    values[~record.valid.T] = np.nan
    fields = CUSTOM_FIELDS + ["frequency_hz", "spike_count", "excluded_spike_count"]
    custom = _frequency_customdata(product, fields, products)
    for band in product.meta["identity"]["parameters"].get("bands", []):
        for low, high in band["excluded_intervals_hz"]:
            values[(record.frequency_hz >= low) & (record.frequency_hz <= high)] = np.nan
    start, stop = INTERVAL_BOUNDS_S[product.meta["identity"]["interval"]]
    starts, stops = record.choice_times_s - origin + start, record.choice_times_s - origin + stop
    overlap = np.any(starts[1:] < stops[:-1])
    tiles = []
    if overlap:
        tiles = [(np.array([left, right]), values[:, index:index + 1], custom[:, index:index + 1])
                 for index, (left, right) in enumerate(zip(starts, stops))]
    else:
        edges, columns, custom_columns = [], [], []
        for index, (left, right) in enumerate(zip(starts, stops)):
            if edges and left > edges[-1]:
                edges.append(left)
                columns.append(np.full(values.shape[0], np.nan))
                custom_columns.append(np.full((values.shape[0], len(fields)), None, dtype=object))
            if not edges:
                edges.append(left)
            edges.append(right)
            columns.append(values[:, index])
            custom_columns.append(custom[:, index])
        tiles = [(np.asarray(edges), np.column_stack(columns), np.stack(custom_columns, axis=1))] if columns else []
    limits = _metric_limits(values, metric)
    for edges, scalar_values, tile_custom in tiles:
        figure.add_trace(go.Heatmap(x=edges, y=record.frequency_hz, z=scalar_values,
                                    zmin=limits[0], zmax=limits[1], zauto=False, zsmooth=False, connectgaps=False,
                                    hoverongaps=False, colorscale="Viridis" if metric != "ppc" else "RdBu",
                                    customdata=tile_custom,
                                    meta={"kind": "frequency-heatmap", "family": family, "customdata_fields": fields,
                                          "selection": product.meta["identity"]["selection"]},
                                    hovertemplate="Window summary<br>Trial %{customdata[0]}, original row %{customdata[1]}"
                                    "<br>UTC [%{customdata[3]:.6f}, %{customdata[4]:.6f}) s; choice %{customdata[2]:.6f}"
                                    "<br>%{y:g} Hz: %{z:.6g}<br>Samples %{customdata[5]}, segments %{customdata[6]}"
                                    "<br>Valid spikes %{customdata[15]}, excluded %{customdata[16]}"
                                    "<br>Source run %{customdata[10]}, completed %{customdata[11]}"
                                    "<br>%{customdata[12]}<extra></extra>"),
                          row=row, col=1)
    figure.update_yaxes(title_text="Frequency (Hz)", row=row, col=1)


def _frequency_customdata(product: SavedProduct, fields: list[str], products: tuple) -> np.ndarray:
    """Build (frequency,trial,field) original-identity/UTC/support/source hover data, without reducing metrics."""
    record = product.record
    start, stop = INTERVAL_BOUNDS_S[product.meta["identity"]["interval"]]
    data = np.empty((len(record.frequency_hz), len(record.trial_ids), len(fields)), dtype=object)
    details = _source_details(product, products)
    for frequency, hz in enumerate(record.frequency_hz):
        for trial in range(len(record.trial_ids)):
            samples = getattr(record, "sample_count", None)
            count = None if samples is None else samples[trial] if samples.ndim == 1 else samples[trial, frequency]
            spikes = record.spike_count[trial, frequency] if hasattr(record, "spike_count") else None
            excluded = record.excluded_spike_count[trial, frequency] if hasattr(record, "excluded_spike_count") else None
            data[frequency, trial] = [record.trial_ids[trial], record.row_positions[trial], record.choice_times_s[trial],
                                      record.choice_times_s[trial] + start, record.choice_times_s[trial] + stop,
                                      count, record.segment_count[trial] if hasattr(record, "segment_count") else None,
                                      None, None, None, product.meta["run_id"], product.meta["completed_utc"],
                                      details, bool(record.valid[trial, frequency]), hz, spikes, excluded]
    return data


def _frequency_index(frequencies: np.ndarray, requested: float) -> int | None:
    """Return exact saved Hz index or None, with no nearest-frequency substitution."""
    match = np.flatnonzero(np.isclose(frequencies, requested, rtol=0, atol=1e-12))
    return int(match[0]) if match.size else None


def _caption(interval: str) -> str:
    """Return explicit half-open seconds and descriptive estimator/unit interpretation for figure captions."""
    half = 1 if interval == "primary" else 2
    return (f"{interval} [-{half},+{half}) s window summaries at actual choices. "
            "Power is absolute integrated uV^2; PLV/PPC describe unit-phase consistency and coherence averages spectra. "
            "LFP PPC is descriptive within-trial consistency; temporal dependence is not corrected.")


def _light_layout(figure: go.Figure, caption: str, height: int) -> None:
    """Apply opaque white backgrounds, readable black text and a visible caption; return None."""
    figure.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                         font={"size": 14, "color": "black"}, height=height,
                         margin={"l": 85, "r": 100, "t": 65, "b": 150}, showlegend=False, dragmode="pan")
    figure.add_annotation(text="<br>".join(textwrap.wrap(caption, 135)), x=0, y=-.1, xref="paper", yref="paper",
                           showarrow=False, xanchor="left", yanchor="top", align="left")


def _save_matplotlib_figure(prepared: go.Figure, path: Path) -> None:
    """Render the same prepared Cartesian/polar data with Agg; save an opaque captioned PNG at path."""
    axis_names = sorted((name for name in prepared.layout if re.fullmatch(r"yaxis\d*", name)),
                        key=_axis_number)
    polar = [trace for trace in prepared.data if trace.type == "barpolar"]
    count = len(axis_names) + bool(polar)
    figure = MatplotlibFigure(figsize=(12, max(6, 2.2 * count + 2)), facecolor="white")
    FigureCanvasAgg(figure)
    axes = [figure.add_subplot(count, 1, index + 1) for index in range(len(axis_names))]
    for index, name in enumerate(axis_names):
        layout = prepared.layout[name]
        axes[index].set_title(_subplot_title(prepared, layout.domain[1]), loc="left", fontsize=14, pad=9)
        axes[index].set_ylabel(layout.title.text or "", fontsize=13)
        if layout.range is not None:
            axes[index].set_ylim(layout.range)
        if layout.tickvals is not None and layout.ticktext is not None:
            axes[index].set_yticks(layout.tickvals, labels=layout.ticktext)
        time_layout = prepared.layout["xaxis" + name[5:]]
        if time_layout.range is not None:
            axes[index].set_xlim(time_layout.range)
        if index == len(axis_names) - 1:
            axes[index].set_xlabel(time_layout.title.text or "Elapsed session time (s)", fontsize=13)
        axes[index].tick_params(labelsize=12)
    for trace in prepared.data:
        if trace.type not in {"scatter", "scattergl"}:
            continue
        index = int((trace.yaxis or "y")[1:] or "1") - 1
        marker = "." if "markers" in (trace.mode or "") else None
        axes[index].plot(np.asarray(trace.x), np.asarray(trace.y, dtype=float),
                         linestyle="-" if "lines" in (trace.mode or "") else "None", marker=marker)
    if polar:
        polar_axis = figure.add_subplot(count, 1, count, projection="polar")
        trace = polar[0]
        polar_axis.bar(np.deg2rad(trace.theta), trace.r, width=np.deg2rad(trace.width), alpha=.8)
        polar_axis.set_title("Saved spike-phase histogram (counts)", fontsize=14, pad=22)
        polar_axis.tick_params(labelsize=12)
    caption = prepared.layout.meta["caption"]
    figure.text(.03, .015, "\n".join(textwrap.wrap(caption, 125)), fontsize=12, color="black")
    figure.subplots_adjust(left=.1, right=.95, top=.96, bottom=.14, hspace=.95)
    figure.savefig(path, dpi=120, facecolor="white", edgecolor="white", transparent=False)
    figure.clear()


def _subplot_title(prepared: go.Figure, domain_top: float) -> str:
    """Recover a public Plotly subplot annotation at its saved vertical-domain top for Matplotlib labels."""
    for annotation in prepared.layout.annotations:
        if annotation.xref == "paper" and annotation.yref == "paper" and np.isclose(annotation.y, domain_top, rtol=0, atol=1e-10):
            return re.sub(r"<[^>]+>", " ", annotation.text)
    return ""


def _axis_number(name: str) -> int:
    """Return the one-based numeric suffix of a Plotly Cartesian axis name."""
    suffix = re.search(r"\d+$", name)
    return int(suffix.group()) if suffix else 1
