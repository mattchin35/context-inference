"""Plotly session-time tracks with optional score z-scoring; never bin or fit PCA."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.population.session_timeline import (
    RegionalTimelinePCA,
    SessionTimelineResult,
)
from src.neural_analysis.population.window_variance import WindowVarianceSummary
from src.neural_analysis.task_decoding.session_prediction.plotting import (
    PredictionDisplay, add_prediction_tracks, enrich_prediction_heatmap,
    prediction_display_label, prediction_heatmap_values, selected_pc_ids,
)
from src.mouse_behavior_preprocessing.treadmill_speed import TreadmillSpeed


EVENT_TRACKS = ("trial_start", "left_entry", "right_entry", "LED")
EVENT_LABELS = {
    "trial_start": "Trial start",
    "left_entry": "Left licks",
    "right_entry": "Right licks",
    "LED": "LED",
}
CONTEXT_EVENTS = {
    "enter_left_patch": "Left active",
    "enter_right_patch": "Right active",
    "enter_dark_period": "Dark period",
}
CONTEXT_COLORS = {
    "Left active": "blue",
    "Right active": "darkred",
    "Dark period": "#6b7280",
}


def build_timeline_heatmap_data(
    regional: RegionalTimelinePCA,
    choices: pd.DataFrame,
    pc_range: tuple[int, int] | None = None,
    loaded_range_s: tuple[float, float] | None = None,
    display_scores: np.ndarray | None = None,
    pc_order: tuple[int, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Return explicit temporal edges, signed scores, and per-cell hover information.

    regional scores are (W,B,K); choices supplies elapsed_s, choice_time (UTC
    seconds), and original trial_id. pc_range is inclusive one-based ranks.
    loaded_range_s optionally selects intersecting windows in elapsed seconds.
    pc_order optionally supplies explicit unique one-based original PC ranks
    in display order, instead of pc_range. Existing range defaults are unchanged.
    display_scores optionally replaces selected PC values, shape (W,B,P), in
    raw score or PC-SD units. It must retain the regional window/bin order.
    Return x (M+1,) edges, y (P,) ranks, z (P,M), customdata (P,M,8). Each actual
    inter-window gap is one NaN column, regardless of its elapsed duration.
    """
    component_count = regional.scores.shape[2]
    if pc_order is None:
        first, last = pc_range or (1, min(10, component_count))
        if first < 1 or last < first or last > component_count:
            raise ValueError("PC range must be within the available one-based ranks.")
        ranks = np.arange(first, last + 1)
    else:
        ranks = np.asarray(pc_order)
        if (pc_range is not None or ranks.ndim != 1 or not len(ranks)
                or not np.issubdtype(ranks.dtype, np.integer)
                or len(np.unique(ranks)) != len(ranks) or np.any(ranks < 1)
                or np.any(ranks > component_count)):
            raise ValueError("PC order must contain unique available one-based integer ranks")
    scores = (
        regional.scores[:, :, ranks - 1]
        if display_scores is None
        else display_scores
    )
    if scores.shape != (*regional.scores.shape[:2], len(ranks)):
        raise ValueError(
            "Display scores must match the windows, bins, and selected PCs."
        )
    edges = []
    columns = []
    custom = []
    for window_index, choice_position in enumerate(regional.choice_positions):
        choice = choices.iloc[int(choice_position)]
        elapsed_edges = float(choice.elapsed_s) + regional.bin_edges_s
        if loaded_range_s is not None:
            if (
                elapsed_edges[-1] <= loaded_range_s[0]
                or elapsed_edges[0] >= loaded_range_s[1]
            ):
                continue
        if edges and elapsed_edges[0] > edges[-1]:
            edges.append(float(elapsed_edges[0]))
            columns.append(np.full((len(ranks), 1), np.nan))
            custom.append(np.full((len(ranks), 1, 8), None, dtype=object))
        if not edges:
            edges.extend(elapsed_edges.tolist())
        else:
            edges.extend(elapsed_edges[1:].tolist())
        values = scores[window_index].T
        columns.append(values)
        bin_count = values.shape[1]
        info = np.empty((len(ranks), bin_count, 8), dtype=object)
        info[:, :, 0] = choice.trial_id
        info[:, :, 1] = float(choice.choice_time) + regional.bin_edges_s[:-1]
        info[:, :, 2] = float(choice.choice_time) + regional.bin_edges_s[1:]
        info[:, :, 3] = regional.bin_edges_s[:-1]
        info[:, :, 4] = regional.bin_edges_s[1:]
        info[:, :, 5] = elapsed_edges[:-1]
        info[:, :, 6] = elapsed_edges[1:]
        info[:, :, 7] = (
            regional.transform.explained_variance_ratio[ranks - 1, None] * 100
        )
        custom.append(info)
    return {
        "x": np.asarray(edges),
        "y": ranks,
        "z": np.concatenate(columns, axis=1) if columns else np.empty((len(ranks), 0)),
        "customdata": np.concatenate(custom, axis=1)
        if custom
        else np.empty((len(ranks), 0, 8), dtype=object),
    }


def zscore_pc_scores(scores: np.ndarray) -> np.ndarray:
    """Return display-only per-PC z-scores with unchanged (windows,bins,PCs) shape.

    scores is a nonempty finite float array of saved PCA scores. Each PC uses
    its own mean and population SD (ddof=0) across all window/bin observations.
    Return a new float64 array in dimensionless PC-SD units. Constant PCs
    become zero; no input is modified and no gaps are included in the pool.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 3 or not all(values.shape):
        raise ValueError("PC scores must have nonempty (windows, bins, PCs) axes.")
    means = values.mean(axis=(0, 1))
    scales = values.std(axis=(0, 1), ddof=0)
    # Explicitly detect constants even if floating-point mean rounding leaves a tiny SD.
    variable = values.max(axis=(0, 1)) > values.min(axis=(0, 1))
    return np.divide(
        values - means,
        scales,
        out=np.zeros_like(values),
        where=variable & (scales > 0),
    )


def verified_led_intervals(events: pd.DataFrame) -> pd.DataFrame:
    """Pair verified LED on/off events by source identity, in elapsed seconds.

    events contains Event and elapsed_s. Return identity/start_s/end_s columns.
    Source stems such as LED or left_LED stay separate. An unmatched offset
    adds no interval; a repeated onset replaces the uncertain earlier onset.
    """
    intervals = []
    pending = {}
    if not {"Event", "elapsed_s"}.issubset(events.columns):
        return pd.DataFrame(columns=["identity", "start_s", "end_s"])
    for event in events.sort_values("elapsed_s", kind="stable").itertuples():
        match = re.fullmatch(r"(.*LED.*)_(on|off)", str(event.Event))
        if match is None:
            continue
        identity, transition = match.groups()
        if transition == "on":
            pending[identity] = float(event.elapsed_s)
        elif identity in pending:
            start = pending.pop(identity)
            if float(event.elapsed_s) > start:
                intervals.append(
                    {
                        "identity": identity,
                        "start_s": start,
                        "end_s": float(event.elapsed_s),
                    }
                )
    return pd.DataFrame(intervals, columns=["identity", "start_s", "end_s"])


def context_intervals(events: pd.DataFrame, session_end_s: float) -> pd.DataFrame:
    """Return logged task-context intervals in elapsed-session seconds.

    events contains Event and elapsed_s; session_end_s is the displayed session
    extent. Return context/start_s/end_s columns, one half-open interval per
    context-entry event. Each state lasts until the next entry or session end,
    matching the existing behavioral color-block raster. Times before the first
    logged entry remain unknown. No context is inferred from choices or LEDs.
    """
    columns = ["context", "start_s", "end_s"]
    if not {"Event", "elapsed_s"}.issubset(events.columns):
        return pd.DataFrame(columns=columns)
    entries = events.loc[
        events.Event.isin(CONTEXT_EVENTS), ["Event", "elapsed_s"]
    ].copy()
    entries["elapsed_s"] = pd.to_numeric(entries["elapsed_s"], errors="coerce")
    entries = entries.loc[np.isfinite(entries.elapsed_s)].sort_values(
        "elapsed_s", kind="stable"
    )
    starts = entries.elapsed_s.to_numpy(dtype=float)
    ends = (
        np.minimum(np.r_[starts[1:], session_end_s], session_end_s)
        if len(starts)
        else np.array([])
    )
    intervals = pd.DataFrame(
        {
            "context": entries.Event.map(CONTEXT_EVENTS).to_numpy(),
            "start_s": starts,
            "end_s": ends,
        }
    )
    return intervals.loc[intervals.end_s > intervals.start_s].reset_index(drop=True)


def session_time_bounds(result: SessionTimelineResult) -> tuple[float, float]:
    """Return full elapsed-second extent of choices/windows/events, including time zero.

    The same common origin is used for every population; negative window edges
    are retained. An otherwise empty session receives a one-second display span.
    """
    bounds = [0.0]
    if not result.choices.empty:
        bounds.extend(
            [
                float(result.choices.elapsed_s.min()) - 2.0,
                float(result.choices.elapsed_s.max()) + 2.0,
            ]
        )
    if not result.events.empty:
        bounds.extend(
            [float(result.events.elapsed_s.min()), float(result.events.elapsed_s.max())]
        )
    low, high = min(bounds), max(bounds)
    return low, max(high, low + 1.0)


def available_event_tracks(events: pd.DataFrame) -> tuple[str, ...]:
    """Return existing optional track keys from an event table, without creating data."""
    if "Event" not in events:
        return ()
    names = set(events.Event.astype(str))
    tracks = [name for name in EVENT_TRACKS[:-1] if name in names]
    if any(re.fullmatch(r".*LED.*_(on|off)", name) for name in names):
        tracks.append("LED")
    return tuple(tracks)


def add_timeline_behavior(
    figure: go.Figure,
    result: SessionTimelineResult,
    event_tracks: tuple[str, ...] = (),
    show_context: bool = True,
) -> None:
    """Add existing choice/context/event traces to preallocated subplot rows.

    figure has choices at row1/column1 and one subsequent row per available
    requested event track. result retains choices(W,C), events(E,C), original
    IDs and UTC/elapsed-second times. Logged shading keeps its original timing.
    Return None; this public helper shares existing rendering without fitting
    or modifying result arrays. Unavailable requested event tracks are skipped.
    """
    contexts = context_intervals(result.events, session_time_bounds(result)[1])
    if show_context:
        _add_context_overlay(figure, contexts)
    _add_choices(figure, result.choices, 1, contexts)
    tracks = [track for track in event_tracks if track in available_event_tracks(result.events)]
    for row, track in enumerate(tracks, start=2):
        _add_event_track(figure, result.events, track, row)


def plot_session_timeline(
    result: SessionTimelineResult,
    time_range_s: tuple[float, float] | None = None,
    pc_ranges: dict[str, tuple[int, int]] | None = None,
    event_tracks: tuple[str, ...] = EVENT_TRACKS,
    loaded_range_s: tuple[float, float] | None = None,
    color_limits: dict[str, float] | None = None,
    viewport_revision: int = 0,
    show_context: bool = True,
    zscore_pcs: bool = False,
    window_variance: dict[str, WindowVarianceSummary] | None = None,
    speed: TreadmillSpeed | None = None,
    prediction_display: PredictionDisplay | None = None,
) -> go.Figure:
    """Build synchronized tracks from a prepared result, without modifying its arrays.

    Time ranges are elapsed seconds, PC ranges inclusive one-based ranks,
    and color_limits are positive symmetric display limits keyed by region.
    loaded_range_s bounds the neural payload only. show_context toggles logged
    task-context shading on the choices row; context hover remains available.
    zscore_pcs uses each displayed PC's mean/population SD over all prepared
    window/bin observations, before loaded-range filtering. Color limits are
    then in PC-SD units, with automatic limits spanning the selected PCs' full
    session z-score range. Saved scores and variance percentages are unchanged.
    window_variance optionally adds saved fixed-axis count points after each
    regional heatmap; it is independent of PC selection and display scaling.
    speed optionally adds a signed-mm/s post-step row below events. Its entire
    UTC series is converted to the common elapsed origin, without neural-window
    filtering or changing the established session/navigator bounds.
    Return a Plotly figure;
    native chart navigation does not send arbitrary range events to Python.
    prediction_display optionally adds saved PFC/HPC target readouts and actual
    overlays, with hidden/raw/weighted PC rows. It never changes fitted models.
    """
    full_bounds = session_time_bounds(result)
    contexts = context_intervals(result.events, full_bounds[1])
    visible = time_range_s or full_bounds
    pc_ranges = pc_ranges or {}
    color_limits = color_limits or {}
    window_variance = window_variance or {}
    tracks = [
        track
        for track in event_tracks
        if track in available_event_tracks(result.events)
    ]
    titles = ["Choices"] + [EVENT_LABELS[track] for track in tracks]
    if show_context and not contexts.empty:
        titles[0] = "Choices - context: blue=left, red=right, gray=dark"
    heights = [1.0] + [0.65] * len(tracks)
    if speed is not None:
        titles.append("Signed speed (mm/s)")
        heights.append(1.1)
    if prediction_display is not None:
        label = prediction_display_label(prediction_display)
        titles.extend([f"PFC - {label}", f"HPC - {label}"])
        heights.extend([1.5, 1.5])
    display_regions = result.regions if prediction_display is None or prediction_display.pc_mode != "hidden" else {}
    for role, regional in display_regions.items():
        first, last = pc_ranges.get(role, (1, min(10, regional.scores.shape[2])))
        if prediction_display is None:
            titles.append(f"{role} ({regional.probe_id}) - session PCs {first}-{last}")
            visible_pc_count = last - first + 1
        else:
            ids = selected_pc_ids(prediction_display, role, pc_ranges.get(role))
            titles.append(f"{role} ({regional.probe_id}) - {prediction_display.pc_mode} PCs; {prediction_display.pc_order} order")
            visible_pc_count = len(ids)
        heights.append(max(2.0, visible_pc_count * 0.2))
        if role in window_variance:
            titles.append(f"{role} - fixed session axes for 95% window variance")
            heights.append(0.9)
    titles.append("Session navigator")
    heights.append(0.45)
    figure = make_subplots(
        rows=len(titles),
        cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=min(0.025, 0.16 / len(titles)),
        subplot_titles=titles,
    )
    if show_context:
        _add_context_overlay(figure, contexts)
    _add_choices(figure, result.choices, 1, contexts)
    row = 2
    for track in tracks:
        _add_event_track(figure, result.events, track, row)
        row += 1
    speed_row = None
    if speed is not None:
        speed_row = row
        _add_speed_track(figure, speed, result.time_origin_s, row)
        row += 1
    if prediction_display is not None:
        add_prediction_tracks(figure, prediction_display, row, loaded_range_s)
        row += 2
    for role, regional in display_regions.items():
        display_scores = None
        default_limit = regional.color_limit
        weighted = prediction_display is not None and prediction_display.pc_mode == "weighted"
        scaled_raw = zscore_pcs and not weighted
        ids = None
        if prediction_display is not None:
            ids = selected_pc_ids(prediction_display, role, pc_ranges.get(role))
            sliced, windows, display_scores = prediction_heatmap_values(prediction_display, role, ids, loaded_range_s)
            if weighted:
                target = prediction_display.result.inputs.target_names.index(prediction_display.target_name)
                default_limit = prediction_display.result.regions[role].contribution_color_limit[target]
            elif scaled_raw:
                all_values = zscore_pc_scores(regional.scores[:, :, np.asarray(ids) - 1])
                default_limit = float(np.abs(all_values).max()) or 1.0
                display_scores = all_values[windows]
            data = build_timeline_heatmap_data(sliced, result.choices,
                                              display_scores=display_scores, pc_order=ids)
            data = enrich_prediction_heatmap(data, prediction_display, role, ids)
        elif scaled_raw:
            first, last = pc_ranges.get(role, (1, min(10, regional.scores.shape[2])))
            display_scores = zscore_pc_scores(regional.scores[:, :, first - 1 : last])
            default_limit = float(np.abs(display_scores).max())
            if default_limit == 0.0:
                default_limit = 1.0
        if prediction_display is None:
            data = build_timeline_heatmap_data(
                regional, result.choices, pc_ranges.get(role), loaded_range_s, display_scores,
            )
        limit = float(default_limit if weighted else color_limits.get(role, default_limit))
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                "Manual color limits must be positive finite score values."
            )
        axis_domain = figure.layout[f"yaxis{row}"].domain
        colorbar_title = f"{role} z-score<br>(PC SD)" if scaled_raw else f"{role} score"
        value_label = "Z-score (PC SD)" if scaled_raw else "Score"
        if weighted:
            units = "log-odds" if prediction_display.result.inputs.target_families[target] == "categorical" else "target units"
            colorbar_title = f"{role} contribution<br>({units})"
            value_label = f"Contribution ({units})"
        hover = (
            f"{role} ({regional.probe_id}) %{{y}} - %{{customdata[7]:.2f}}% session variance"
            "<br>Trial %{customdata[0]}<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
            "<br>UTC [%{customdata[1]:.6f}, %{customdata[2]:.6f}) s"
            "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
            f"<br>{value_label} %{{z:.5g}}<extra></extra>"
        )
        if prediction_display is not None:
            target_label = prediction_display_label(prediction_display)
            hover = (f"{role}: {target_label}; PC%{{customdata[8]}}<br>{value_label} %{{z:.5g}}"
                     "<br>Prediction %{customdata[12]:.5g}; actual %{customdata[13]:.5g}"
                     "<br>Trial %{customdata[0]}; held-out fold %{customdata[10]}"
                     "<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
                     "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
                     "<br>Weight %{customdata[9]:.5g}<br>Conditions: %{customdata[11]}<extra></extra>")
        figure.add_trace(
            go.Heatmap(
                **data,
                name=role,
                colorscale=[[0.0, "#2166ac"], [0.5, "#d1d5db"], [1.0, "#b2182b"]],
                zmin=-limit,
                zmax=limit,
                zauto=False,
                zsmooth=False,
                connectgaps=False,
                hoverongaps=False,
                colorbar={
                    "title": {"text": colorbar_title},
                    "len": axis_domain[1] - axis_domain[0],
                    "y": sum(axis_domain) / 2,
                    "thickness": 12,
                },
                hovertemplate=hover,
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(
            autorange="reversed",
            tickmode="array",
            tickvals=data["y"],
            ticktext=[f"PC{rank}" for rank in (ids if ids is not None else data["y"])],
            row=row,
            col=1,
        )
        row += 1
        if role in window_variance:
            _add_window_counts(figure, window_variance[role], result.choices, row)
            row += 1
    figure.add_trace(
        go.Scatter(
            x=result.choices.elapsed_s,
            y=result.choices.action,
            mode="markers",
            marker={"size": 3, "color": "#555555"},
            hoverinfo="skip",
            showlegend=False,
            name="Session overview",
        ),
        row=row,
        col=1,
    )
    figure.update_yaxes(visible=False, row=row, col=1)
    figure.update_xaxes(range=visible, showgrid=False)
    figure.update_yaxes(fixedrange=True, showgrid=False, zeroline=False)
    if speed_row is not None:
        figure.update_yaxes(
            zeroline=True, zerolinecolor="#9ca3af", row=speed_row, col=1
        )
    figure.update_xaxes(
        title_text="Elapsed session time (s; zero = first trial start)",
        row=row,
        col=1,
        rangeslider={"visible": True, "range": full_bounds, "thickness": 0.07},
    )
    allowed = loaded_range_s or full_bounds
    figure.update_xaxes(minallowed=allowed[0], maxallowed=allowed[1])
    identity = result.meta.get("configuration_id", result.session_id)
    time_identity = f"{result.session_id}:{result.time_origin_s}"
    # Only time navigation persists; PC/track changes must recalculate vertical ranges.
    figure.update_yaxes(
        uirevision=f"{identity}:rows:{tuple(tracks)}:{sorted(pc_ranges.items())}:{tuple(window_variance)}:speed:{speed is not None}"
    )
    if prediction_display is not None:
        saved = prediction_display.result
        figure.update_yaxes(uirevision=f"{identity}:{saved.inputs.analysis_mode}:{saved.meta.get('identity', '')}:prediction:{prediction_display.target_name}:"
                             f"{prediction_display.pc_mode}:{prediction_display.pc_order}:{prediction_display.pc_count}:"
                             f"{sorted(pc_ranges.items())}:{tuple(tracks)}:speed:{speed is not None}")
    figure.update_layout(
        height=max(600, int(sum(heights) * 90 + 180)),
        margin={"l": 85, "r": 135, "t": 50, "b": 90},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "black"},
        dragmode="pan",
        showlegend=False,
        uirevision=f"{time_identity}:{viewport_revision}",
    )
    return figure


# Plotting-only helpers. All x coordinates below are elapsed seconds.
def _add_speed_track(
    figure: go.Figure, speed: TreadmillSpeed, origin_s: float, row: int
) -> None:
    """Plot S actual speed updates as signed-mm/s steps in the indicated subplot.

    utc_s/speed_mm_s/source_rows retain (S,) order, including equal timestamps.
    Only x is transformed to elapsed seconds. Use a fixed full-series y range;
    no smoothing, sign change, cropping, or synthetic endpoint is introduced.
    """
    low = min(0.0, float(speed.speed_mm_s.min()))
    high = max(0.0, float(speed.speed_mm_s.max()))
    margin = 0.05 * (high - low) if high > low else 1.0
    figure.add_trace(
        go.Scattergl(
            x=speed.utc_s - origin_s,
            y=speed.speed_mm_s,
            mode="lines",
            line={"shape": "hv", "width": 1.5, "color": "#047857"},
            name="Signed speed (mm/s)",
            connectgaps=False,
            customdata=np.column_stack([speed.utc_s, speed.source_rows]),
            hovertemplate="Signed speed %{y:.2f} mm/s<br>Elapsed %{x:.3f} s"
            "<br>UTC %{customdata[0]:.6f} s<br>Parsed source row %{customdata[1]:.0f}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    figure.update_yaxes(
        title_text="mm/s",
        range=(low - margin, high + margin),
        autorange=False,
        row=row,
        col=1,
    )


def _add_window_counts(
    figure: go.Figure,
    summary: WindowVarianceSummary,
    choices: pd.DataFrame,
    row: int,
) -> None:
    """Add marker-only counts at actual choice times to one subplot row.

    summary has W aligned counts (NaN unavailable); choices supplies original
    IDs and UTC/elapsed seconds. Hover includes status, denominator, and basis.
    """
    selected = choices.iloc[summary.choice_positions]
    bins = len(summary.bin_edges_s) - 1
    basis = "complete" if summary.basis_complete else "restricted"
    # Keep numeric hover values numeric when mixed with status strings.
    custom = np.empty((len(selected), 7), dtype=object)
    custom[:, 0] = selected.trial_id
    custom[:, 1] = selected.choice_time
    custom[:, 2] = summary.status
    custom[:, 3] = summary.total_variance
    custom[:, 4] = bins
    custom[:, 5] = len(summary.unit_ids)
    custom[:, 6] = basis
    figure.add_trace(
        go.Scattergl(
            x=selected.elapsed_s,
            y=summary.counts95,
            mode="markers",
            name=f"{summary.region} fixed-axis count (95%)",
            marker={
                "size": 6,
                "color": "#7c3aed",
                "symbol": "circle" if summary.basis_complete else "diamond-open",
            },
            customdata=custom,
            hovertemplate=(
                "Trial %{customdata[0]}<br>Elapsed %{x:.3f} s; UTC %{customdata[1]:.6f} s"
                "<br>Fixed-axis count (95%): %{y}<br>Status: %{customdata[2]}"
                "<br>Total variance: %{customdata[3]:.5g}<br>%{customdata[4]} bins; %{customdata[5]} units"
                "<br>%{customdata[6]} basis<extra></extra>"
            ),
        ),
        row=row,
        col=1,
    )
    finite_counts = summary.counts95[np.isfinite(summary.counts95)]
    step = max(1, int(np.ceil(finite_counts.max() / 4))) if len(finite_counts) else 1
    figure.update_yaxes(
        title_text="PC count", rangemode="tozero", dtick=step, row=row, col=1
    )


def _add_choices(
    figure: go.Figure,
    choices: pd.DataFrame,
    row: int,
    contexts: pd.DataFrame,
) -> None:
    """Add outcome-styled choice lanes and context hover to a one-based subplot row.

    choices retains original trial IDs and elapsed/UTC-second choice times.
    contexts is the elapsed-second interval table from context_intervals.
    Optional reward is 1/0/NaN (rewarded/unrewarded/unknown). Marker sizes are
    pixels: 12 filled, 7 open, or 9 gray cross for unknown. Custom data has
    shape (selected choices, 4): trial ID, UTC time, context, reward status.
    """
    for action, label, color in ((1, "Left", "#2563eb"), (0, "Right", "#c2410c")):
        selected = choices.loc[choices.action == action]
        rewards = np.full(len(selected), np.nan)
        if "reward" in selected:
            rewards = pd.to_numeric(selected.reward, errors="coerce").to_numpy(
                dtype=float, na_value=np.nan
            )
        rewarded, unrewarded = rewards == 1, rewards == 0
        symbol = "triangle-up" if action else "triangle-down"
        custom = np.empty((len(selected), 4), dtype=object)
        custom[:, :2] = selected[["trial_id", "choice_time"]].to_numpy()
        custom[:, 2] = _choice_context_labels(selected.elapsed_s.to_numpy(), contexts)
        custom[:, 3] = np.select(
            [rewarded, unrewarded], ["Rewarded", "Unrewarded"], default="Unknown"
        )
        figure.add_trace(
            go.Scatter(
                x=selected.elapsed_s,
                y=np.full(len(selected), action),
                mode="markers",
                name=f"{label} choices",
                customdata=custom,
                marker={
                    "color": np.where(rewarded | unrewarded, color, "#6b7280").tolist(),
                    "size": np.select(
                        [rewarded, unrewarded], [12, 7], default=9
                    ).tolist(),
                    "symbol": np.select(
                        [rewarded, unrewarded], [symbol, symbol + "-open"], default="x"
                    ).tolist(),
                    "line": {"width": 1.5},
                },
                hovertemplate=f"{label} choice<br>Trial %{{customdata[0]}}<br>Elapsed %{{x:.3f}} s"
                "<br>UTC %{customdata[1]:.6f} s<br>Active context: %{customdata[2]}"
                "<br>Reward: %{customdata[3]}<extra></extra>",
            ),
            row=row,
            col=1,
        )
    figure.update_yaxes(
        tickmode="array",
        tickvals=[0, 1],
        ticktext=["Right", "Left"],
        range=[-0.5, 1.5],
        row=row,
        col=1,
    )


def _add_context_overlay(figure: go.Figure, contexts: pd.DataFrame) -> None:
    """Shade only the choices subplot from an elapsed-second context interval table.

    Each rectangle spans the top y-axis domain, below markers, with actual
    transition x coordinates. Keeping all intervals preserves native panning.
    """
    for interval in contexts.itertuples():
        figure.add_shape(
            type="rect",
            xref="x",
            yref="y domain",
            x0=interval.start_s,
            x1=interval.end_s,
            y0=0,
            y1=1,
            fillcolor=CONTEXT_COLORS[interval.context],
            opacity=0.18,
            line={"width": 0},
            layer="below",
        )


def _choice_context_labels(times_s: np.ndarray, contexts: pd.DataFrame) -> np.ndarray:
    """Return (N,) context strings at (N,) elapsed-second choice times.

    Context intervals are sorted and half-open; an exact switch uses the new
    state. Times outside logged intervals receive Unknown, never a choice-based
    guess. The result is separate from the saved choice table.
    """
    labels = np.full(len(times_s), "Unknown", dtype=object)
    if contexts.empty:
        return labels
    positions = np.searchsorted(contexts.start_s.to_numpy(), times_s, side="right") - 1
    known = (positions >= 0) & (
        times_s < contexts.end_s.to_numpy()[np.maximum(positions, 0)]
    )
    labels[known] = contexts.context.to_numpy()[positions[known]]
    return labels


def _add_event_track(
    figure: go.Figure, events: pd.DataFrame, track: str, row: int
) -> None:
    """Add source event timestamps and verified LED intervals without inferring missing states."""
    if track == "LED":
        selected = events.loc[
            events.Event.astype(str).str.fullmatch(r".*LED.*_(on|off)")
        ]
        identities = selected.Event.astype(str).str.replace(
            r"_(on|off)$", "", regex=True
        )
        names = list(dict.fromkeys(identities))
        for interval in verified_led_intervals(events).itertuples():
            lane = names.index(interval.identity)
            figure.add_trace(
                go.Scatter(
                    x=[interval.start_s, interval.end_s],
                    y=[lane, lane],
                    mode="lines",
                    line={"width": 12, "color": "#16a34a"},
                    opacity=0.3,
                    name=f"{interval.identity} on interval",
                    customdata=[[interval.start_s, interval.end_s]] * 2,
                    hovertemplate="LED on [%{customdata[0]:.3f}, %{customdata[1]:.3f}) s<extra></extra>",
                ),
                row=row,
                col=1,
            )
        y = [names.index(identity) for identity in identities]
        figure.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(names))),
            ticktext=names,
            row=row,
            col=1,
        )
    else:
        selected = events.loc[events.Event == track]
        y = np.zeros(len(selected))
        figure.update_yaxes(visible=False, row=row, col=1)
    notes = (
        selected["Note"].fillna("").astype(str).to_numpy()
        if "Note" in selected
        else np.full(len(selected), "")
    )
    custom = np.column_stack([selected.Event, selected.Time, notes])
    figure.add_trace(
        go.Scattergl(
            x=selected.elapsed_s,
            y=y,
            mode="markers",
            marker={"symbol": "line-ns", "size": 12, "color": "#333333"},
            name=EVENT_LABELS[track],
            customdata=custom,
            hovertemplate="%{customdata[0]}<br>Elapsed %{x:.3f} s"
            "<br>UTC %{customdata[1]:.6f} s<br>%{customdata[2]}<extra></extra>",
        ),
        row=row,
        col=1,
    )
