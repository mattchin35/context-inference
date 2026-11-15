"""Native-window unit activity with explicit, non-occluding temporal sublanes."""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.neural_analysis.population.timeline_plotting import (
    CONTEXT_COLORS, add_timeline_behavior, available_event_tracks, build_window_heatmap_data,
    context_intervals, verified_led_intervals, zscore_window_activity,
)
from .selection import GROUP_LABELS


def build_unit_activity_data(
    rates_hz: np.ndarray, unit_ids: tuple[str, ...], selected_unit_ids: tuple[str, ...],
    choices: pd.DataFrame, choice_positions: np.ndarray, bin_edges_s: np.ndarray,
    *, scale: str = "zscore", time_range_s: tuple[float, float] | None = None,
) -> dict:
    """Prepare Hz(W,B,U) for selected IDs(S,), returning shared renderer data.

    choices supplies trial_id, UTC choice_time and elapsed_s; positions(W,) are
    integer indices into that table, never trial IDs. Edges(B+1,) are uniform
    choice-relative seconds. Full-pool normalization returns values(W,B,S), Hz
    or unit SDs, and means/SDs(S,) Hz. Global lane IDs(W,) precede viewport
    filtering. Each visible lane has x(M+1,), y_edges(S*L+1,), z(S*L,M) with
    transparent NaN padding, and original bin metadata(S*L,M,7). No mutation,
    resampling or interpolation occurs. time_range_s is elapsed seconds.
    """
    rates = np.asarray(rates_hz, dtype=float)
    ids, selected = tuple(unit_ids), tuple(selected_unit_ids)
    positions, edges = np.asarray(choice_positions), np.asarray(bin_edges_s, dtype=float)
    if (rates.ndim != 3 or rates.shape[2] != len(ids) or len(set(ids)) != len(ids)
            or len(set(selected)) != len(selected) or not set(selected) <= set(ids)
            or not np.isfinite(rates).all() or (rates < 0).any()
            or positions.shape != (rates.shape[0],) or not np.issubdtype(positions.dtype, np.integer)
            or np.any((positions < 0) | (positions >= len(choices))) or np.any(np.diff(positions) <= 0)
            or edges.shape != (rates.shape[1]+1,) or not np.isfinite(edges).all()
            or (np.diff(edges) <= 0).any() or not np.allclose(np.diff(edges), np.diff(edges)[0])):
        raise ValueError("Activity/IDs/choice positions/bin edges must have matching native geometry")
    if scale not in ("zscore", "hz"):
        raise ValueError("Activity scale must be zscore or hz")
    centers = choices.iloc[positions].elapsed_s.to_numpy(dtype=float)
    if not np.isfinite(centers).all() or np.any(np.diff(centers) < 0):
        raise ValueError("Choice times must be finite and chronological")
    if time_range_s is not None and (len(time_range_s) != 2 or not np.isfinite(time_range_s).all()
                                     or time_range_s[1] <= time_range_s[0]):
        raise ValueError("Viewport must be increasing finite elapsed seconds")
    indices = [ids.index(unit) for unit in selected]
    native = rates[:, :, indices]
    means = native.mean(axis=(0, 1)) if len(native) else np.zeros(len(selected))
    standard_deviations = native.std(axis=(0, 1)) if len(native) else np.zeros(len(selected))
    constants = np.ptp(native, axis=(0, 1)) == 0 if len(native) else np.ones(len(selected), bool)
    values = (zscore_window_activity(native) if scale == "zscore" and native.size else native.copy())
    magnitude = float(np.max(np.abs(values), initial=0)) or 1.
    color_limits = (-magnitude, magnitude) if scale == "zscore" else (0., magnitude)
    lane_ids = assign_temporal_lanes(centers + edges[0], centers + edges[-1])
    lane_count = int(lane_ids.max(initial=-1)) + 1
    lanes = []
    for lane in range(lane_count):
        window_indices = np.flatnonzero(lane_ids == lane)
        geometry = build_window_heatmap_data(values[window_indices], choices, positions[window_indices],
                                             edges, time_range_s)
        if not geometry["z"].shape[1] or not selected:
            continue
        columns = geometry["z"].shape[1]
        padded = np.full((len(selected)*lane_count, columns), np.nan)
        custom = np.full((*padded.shape, 7), None, dtype=object)
        rows = np.arange(len(selected))*lane_count + lane
        padded[rows] = geometry["z"]
        custom[rows] = geometry["customdata"]
        lanes.append({"lane_id": lane, "x": geometry["x"], "z": padded,
                      "y_edges": np.arange(len(selected)*lane_count+1), "customdata": custom})
    bounds = time_range_s or ((float(centers[0]+edges[0]), float(centers[-1]+edges[-1])) if len(centers) else (0., 1.))
    return {"values": values, "unit_ids": selected, "mean_hz": means, "scale_hz": standard_deviations,
        "constant_mask": constants, "scale": scale, "lane_ids": lane_ids, "lane_count": lane_count,
        "lanes": lanes, "color_limits": color_limits, "bin_width_s": float(edges[1]-edges[0]),
        "window_s": (float(edges[0]), float(edges[-1])), "time_range_s": tuple(bounds),
        "overlap_count": int(np.sum(lane_ids > 0))}


def assign_temporal_lanes(starts_s: np.ndarray, ends_s: np.ndarray) -> np.ndarray:
    """Assign chronological half-open intervals(W,) seconds to lowest free lane; return int(W,).

    Ends must exceed starts. Equal starts are permitted, touching intervals
    reuse a lane. Assignment is deterministic over the full pool, no mutation.
    """
    starts, ends = np.asarray(starts_s), np.asarray(ends_s)
    if (starts.ndim != 1 or ends.shape != starts.shape or not np.isfinite(starts).all()
            or not np.isfinite(ends).all() or np.any(ends <= starts) or np.any(np.diff(starts) < 0)):
        raise ValueError("Expected finite chronological half-open intervals")
    latest_ends = []
    lanes = np.empty(len(starts), dtype=int)
    for index, (start, end) in enumerate(zip(starts, ends)):
        free = next((lane for lane, previous_end in enumerate(latest_ends) if previous_end <= start), len(latest_ends))
        if free == len(latest_ends):
            latest_ends.append(float(end))
        else:
            latest_ends[free] = float(end)
        lanes[index] = free
    return lanes


def viewport_unit_activity_data(data: dict, time_range_s: tuple[float, float]) -> dict:
    """Slice cached full-pool plot data to elapsed-second viewport; return a shallow new mapping.

    Keep normalized values(W,B,S), statistics, color limits and global lanes by
    reference. Only visible lane columns/edges/customdata are sliced, preserving
    full native bins at viewport boundaries. No recomputation or input mutation.
    """
    if len(time_range_s) != 2 or not np.isfinite(time_range_s).all() or time_range_s[1] <= time_range_s[0]:
        raise ValueError("Expected increasing finite elapsed-second viewport")
    lanes = []
    for lane in data["lanes"]:
        edges = lane["x"]
        columns = np.flatnonzero((edges[:-1] < time_range_s[1]) & (edges[1:] > time_range_s[0]))
        if not len(columns):
            continue
        start, stop = int(columns[0]), int(columns[-1]+1)
        if not np.isfinite(lane["z"][:, start:stop]).any():
            continue
        lanes.append({**lane, "x": edges[start:stop+1], "z": lane["z"][:, start:stop],
                      "customdata": lane["customdata"][:, start:stop]})
    return {**data, "lanes": lanes, "time_range_s": tuple(time_range_s)}


def plot_unit_activity(data: dict, behavior: SessionTimelineResult, display: pd.DataFrame,
                       event_tracks: tuple[str, ...] = (), viewport_revision: int = 0,
                       time_range_s: tuple[float, float] | None = None) -> go.Figure:
    """Render prepared lane data plus behavior and S ordered assignment rows, no fitting.

    data is build_unit_activity_data's mapping; display unit IDs must match its
    S axis. X is elapsed seconds; heatmap Y is padded unit/lane coordinates.
    Return a Plotly Figure retaining shared time-axis navigation metadata.
    """
    _validate_display(data, display)
    tracks = tuple(track for track in event_tracks if track in available_event_tracks(behavior.events))
    row = 2 + len(tracks)
    figure = make_subplots(rows=row, cols=1, shared_xaxes=True, vertical_spacing=.035,
        row_heights=[.7, *([.25]*len(tracks)), max(1., len(display)*max(1, data["lane_count"])/12)],
        subplot_titles=["Recorded choices", *tracks, "Actual unit activity"])
    add_timeline_behavior(figure, behavior, tracks)
    for lane in data["lanes"]:
        custom = _weight_hover(lane["customdata"], data["lane_count"], display)
        figure.add_trace(go.Heatmap(x=lane["x"], y=lane["y_edges"], z=lane["z"],
            customdata=custom, coloraxis="coloraxis", zsmooth=False, connectgaps=False, hoverongaps=False,
            hovertemplate="Unit %{customdata[7]} / %{customdata[8]}<br>Trial %{customdata[0]}"
                "<br>Choice-relative [%{customdata[3]},%{customdata[4]}) s"
                "<br>Elapsed [%{customdata[5]},%{customdata[6]}) s"
                "<br>Weight %{customdata[9]:.4g}; percentile %{customdata[10]:.3f}"
                "<br>Positive folds %{customdata[11]}/5<br>Activity %{z:.4g}<extra></extra>"), row=row, col=1)
    lanes = max(1, data["lane_count"])
    ticks = np.arange(len(display))*lanes + lanes/2
    figure.update_yaxes(range=[max(1, len(display)*lanes), 0], tickvals=ticks,
                       uirevision=repr((data["unit_ids"], lanes)),
                       ticktext=list(data["unit_ids"]), row=row, col=1)
    for start, end, group in _group_bounds(display):
        figure.add_hline(y=end*lanes, line_color="#94a3b8", line_width=1, row=row, col=1)
        figure.add_annotation(x=0, xshift=-110, y=(start+end)*lanes/2, xref="paper", yref=f"y{row}",
            text=GROUP_LABELS[group], showarrow=False, xanchor="right", font={"size": 13})
    viewport = time_range_s or data["time_range_s"]
    caption = activity_caption(data, len(display))
    figure.update_xaxes(range=list(viewport))
    figure.update_xaxes(title_text="Session elapsed time (s)", row=row, col=1)
    figure.add_annotation(text="<br>".join(textwrap.wrap(caption, 80)), x=0, xshift=-220, y=0, yshift=-60,
        xref="paper", yref="paper", showarrow=False, xanchor="left", yanchor="top", align="left", font={"size": 12})
    heatmap_domain = figure.layout[f"yaxis{row}"].domain
    figure.update_layout(template="plotly_white", font={"size": 15},
        height=400 + max(160, len(display)*lanes*20) + 60*len(tracks), margin={"l": 235, "b": 170},
        coloraxis={"cmin": data["color_limits"][0], "cmax": data["color_limits"][1],
                   "colorscale": "RdBu_r" if data["scale"] == "zscore" else "Viridis",
                   "colorbar": {"title": "Unit SD" if data["scale"] == "zscore" else "Hz",
                                "y": sum(heatmap_domain)/2, "len": heatmap_domain[1]-heatmap_domain[0]}},
        uirevision=f"decoder-units-{behavior.session_id}-{viewport_revision}",
        meta={"primary_time_axis": "xaxis", "time_axes": ["xaxis" if i == 1 else f"xaxis{i}" for i in range(1, row+1)],
              "loaded_range_s": list(data["time_range_s"]), "caption": caption, "lane_count": data["lane_count"]})
    return figure


def export_unit_activity_png(path: Path, data: dict, behavior: SessionTimelineResult,
                             display: pd.DataFrame, event_tracks: tuple[str, ...] = (),
                             time_range_s: tuple[float, float] | None = None) -> None:
    """Export shared native lane data to a new PNG Path; no source loading or mutation.

    Inputs match plot_unit_activity. X is elapsed seconds, activity Hz/unit SD.
    NaN padding is transparent over an opaque white figure. Existing files fail.
    """
    from matplotlib import colormaps
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    _validate_display(data, display)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    tracks = tuple(track for track in event_tracks if track in available_event_tracks(behavior.events))
    lanes = max(1, data["lane_count"])
    figure = Figure(figsize=(16, max(7, 3 + len(display)*lanes*.22 + len(tracks))), facecolor="white")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(2+len(tracks), 1, sharex=True,
        gridspec_kw={"height_ratios": [1, *([.5]*len(tracks)), max(2, len(display)*lanes*.25)]}))
    viewport = time_range_s or data["time_range_s"]
    _draw_png_behavior(axes, behavior, tracks, viewport)
    heatmap = axes[-1]
    palette = colormaps["RdBu_r" if data["scale"] == "zscore" else "viridis"].copy()
    palette.set_bad(alpha=0)
    artist = None
    for lane in data["lanes"]:
        artist = heatmap.pcolormesh(lane["x"], lane["y_edges"], np.ma.masked_invalid(lane["z"]),
            cmap=palette, norm=Normalize(*data["color_limits"]), shading="flat", rasterized=True)
    heatmap.set(ylim=(max(1, len(display)*lanes), 0), xlim=viewport,
        yticks=np.arange(len(display))*lanes+lanes/2, yticklabels=list(data["unit_ids"]),
        xlabel="Session elapsed time (s)")
    for start, end, group in _group_bounds(display):
        heatmap.axhline(end*lanes, color="#94a3b8", linewidth=.7)
        heatmap.text(-.10, (start+end)*lanes/2, GROUP_LABELS[group],
            transform=heatmap.get_yaxis_transform(), ha="right", va="center", fontsize=12)
    for axis in axes:
        axis.tick_params(labelsize=13)
    figure.text(.02, .02, textwrap.fill(activity_caption(data, len(display)), 150), fontsize=11, color="black")
    figure.subplots_adjust(left=.22, right=.86, bottom=.18, hspace=.22)
    if artist is not None:
        position = heatmap.get_position()
        colorbar_axis = figure.add_axes((.90, position.y0, .015, position.height))
        figure.colorbar(artist, cax=colorbar_axis, label="Unit SD" if data["scale"] == "zscore" else "Hz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=140, facecolor="white", transparent=False)


def activity_caption(data: dict, unit_count: int) -> str:
    """Return caption from plot-data mapping and dimensionless selected-unit count."""
    source = f"{data['source_label']}: " if data.get("source_label") else ""
    return (f"{source}{unit_count} unique units; native {data['bin_width_s']*1000:g} ms bins; "
        f"choice window {list(data['window_s'])} s; actual {'unit z-scores' if data['scale'] == 'zscore' else 'Hz'}; "
        f"colors {data['color_limits']}. Five usable folds, >=4 positive; highest relative rank, one group/unit. "
        f"{data['lane_count']} sublanes; {data['overlap_count']} windows assigned to additional lanes. "
        "Outcome weights fitted only [0,1) s. Groups and same-session activity are descriptive.")


# Rendering helpers operate only on already prepared data.
def _validate_display(data: dict, display: pd.DataFrame) -> None:
    """Check S ordered assignment rows against selected IDs(S,); return None or ValueError."""
    if tuple(display.unit_id) != tuple(data["unit_ids"]):
        raise ValueError("Display assignments must match the ordered selected-unit axis")


def _weight_hover(custom: np.ndarray, lane_count: int, display: pd.DataFrame) -> np.ndarray:
    """Enrich native bin customdata(R,M,7) with selected-unit/weight fields -> (R,M,12)."""
    enriched = np.full((*custom.shape[:2], 12), None, dtype=object)
    enriched[:, :, :7] = custom
    for index, unit in enumerate(display.itertuples(index=False)):
        for row in range(index*lane_count, (index+1)*lane_count):
            valid = np.array([value is not None for value in custom[row, :, 0]])
            enriched[row, valid, 7:] = [unit.unit_id, GROUP_LABELS[unit.group_index], unit.mean_weight,
                                       unit.percentile, unit.positive_folds]
    return enriched


def _group_bounds(display: pd.DataFrame) -> list[tuple[int, int, int]]:
    """Return contiguous (start,end,group) ordinal unit ranges from S ordered rows."""
    bounds = []
    for group in display.group_index.drop_duplicates():
        positions = np.flatnonzero(display.group_index.to_numpy() == group)
        bounds.append((int(positions[0]), int(positions[-1]+1), int(group)))
    return bounds


def _draw_png_behavior(axes: np.ndarray, behavior: SessionTimelineResult,
                       tracks: tuple[str, ...], viewport: tuple[float, float]) -> None:
    """Draw shared recorded choices/context/events on Matplotlib axes; X elapsed seconds."""
    choices = behavior.choices
    for action, color in ((0, "#991b1b"), (1, "#2563eb")):
        for rewarded in (False, True):
            mask = (choices.action == action) & ((choices.reward == 1) if rewarded else (choices.reward != 1))
            axes[0].scatter(choices.loc[mask, "elapsed_s"], np.full(mask.sum(), action),
                            s=32 if rewarded else 16, facecolors=color if rewarded else "none", edgecolors=color)
    for interval in context_intervals(behavior.events, viewport[1]).itertuples(index=False):
        axes[0].axvspan(interval.start_s, interval.end_s, color=CONTEXT_COLORS[interval.context], alpha=.12)
    axes[0].set(yticks=[0, 1], yticklabels=["Right", "Left"], title="Recorded choices", xlim=viewport)
    for index, track in enumerate(tracks, start=1):
        if track == "LED":
            for interval in verified_led_intervals(behavior.events).itertuples(index=False):
                axes[index].axvspan(interval.start_s, interval.end_s, color="#ca8a04", alpha=.4)
        else:
            times = behavior.events.loc[behavior.events.Event == track, "elapsed_s"]
            axes[index].vlines(times, 0, 1, color="#475569")
        axes[index].set(yticks=[], ylabel=track)
