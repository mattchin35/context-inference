"""Plotly session-time tracks from prepared data; never bin, standardize, or fit."""

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
) -> dict[str, np.ndarray]:
    """Return explicit temporal edges, signed scores, and per-cell hover information.

    regional scores are (W,B,K); choices supplies elapsed_s, choice_time (UTC
    seconds), and original trial_id. pc_range is inclusive one-based ranks.
    loaded_range_s optionally selects intersecting windows in elapsed seconds.
    Return x (M+1,) edges, y (P,) ranks, z (P,M), customdata (P,M,8). Each actual
    inter-window gap is one NaN column, regardless of its elapsed duration.
    """
    component_count = regional.scores.shape[2]
    first, last = pc_range or (1, min(10, component_count))
    if first < 1 or last < first or last > component_count:
        raise ValueError("PC range must be within the available one-based ranks.")
    ranks = np.arange(first, last + 1)
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
        values = regional.scores[window_index, :, first - 1 : last].T
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
            regional.transform.explained_variance_ratio[first - 1 : last, None] * 100
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


def plot_session_timeline(
    result: SessionTimelineResult,
    time_range_s: tuple[float, float] | None = None,
    pc_ranges: dict[str, tuple[int, int]] | None = None,
    event_tracks: tuple[str, ...] = EVENT_TRACKS,
    loaded_range_s: tuple[float, float] | None = None,
    color_limits: dict[str, float] | None = None,
    viewport_revision: int = 0,
    show_context: bool = True,
) -> go.Figure:
    """Build synchronized tracks from a prepared result, without modifying its arrays.

    Time ranges are elapsed seconds, PC ranges inclusive one-based ranks,
    and color_limits are positive symmetric score limits keyed by region.
    loaded_range_s bounds the neural payload only. show_context toggles logged
    task-context shading on the choices row; context hover remains available.
    Return a Plotly figure;
    native chart navigation does not send arbitrary range events to Python.
    """
    full_bounds = session_time_bounds(result)
    contexts = context_intervals(result.events, full_bounds[1])
    visible = time_range_s or full_bounds
    pc_ranges = pc_ranges or {}
    color_limits = color_limits or {}
    tracks = [
        track
        for track in event_tracks
        if track in available_event_tracks(result.events)
    ]
    titles = ["Choices"] + [EVENT_LABELS[track] for track in tracks]
    if show_context and not contexts.empty:
        titles[0] = "Choices - context: blue=left, red=right, gray=dark"
    heights = [1.0] + [0.65] * len(tracks)
    for role, regional in result.regions.items():
        first, last = pc_ranges.get(role, (1, min(10, regional.scores.shape[2])))
        titles.append(f"{role} ({regional.probe_id}) - session PCs {first}-{last}")
        heights.append(max(2.0, (last - first + 1) * 0.2))
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
    for role, regional in result.regions.items():
        data = build_timeline_heatmap_data(
            regional, result.choices, pc_ranges.get(role), loaded_range_s
        )
        limit = float(color_limits.get(role, regional.color_limit))
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError(
                "Manual color limits must be positive finite score values."
            )
        axis_domain = figure.layout[f"yaxis{row}"].domain
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
                    "title": {"text": f"{role} score"},
                    "len": axis_domain[1] - axis_domain[0],
                    "y": sum(axis_domain) / 2,
                    "thickness": 12,
                },
                hovertemplate=(
                    f"{role} ({regional.probe_id}) %{{y}} - %{{customdata[7]:.2f}}% session variance"
                    "<br>Trial %{customdata[0]}<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
                    "<br>UTC [%{customdata[1]:.6f}, %{customdata[2]:.6f}) s"
                    "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
                    "<br>Score %{z:.5g}<extra></extra>"
                ),
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(
            autorange="reversed",
            tickmode="array",
            tickvals=data["y"],
            ticktext=[f"PC{rank}" for rank in data["y"]],
            row=row,
            col=1,
        )
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
    figure.update_xaxes(
        title_text="Elapsed session time (s; zero = first trial start)",
        row=row,
        col=1,
        rangeslider={"visible": True, "range": full_bounds, "thickness": 0.07},
    )
    if loaded_range_s is not None:
        figure.update_xaxes(minallowed=loaded_range_s[0], maxallowed=loaded_range_s[1])
    identity = result.meta.get("configuration_id", result.session_id)
    figure.update_layout(
        height=max(600, int(sum(heights) * 90 + 180)),
        margin={"l": 85, "r": 135, "t": 50, "b": 90},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "black"},
        dragmode="pan",
        showlegend=False,
        uirevision=f"{identity}:{viewport_revision}",
    )
    return figure


# Plotting-only helpers. All x coordinates below are elapsed seconds.
def _add_choices(
    figure: go.Figure,
    choices: pd.DataFrame,
    row: int,
    contexts: pd.DataFrame,
) -> None:
    """Add choice lanes and logged-context hover to a one-based subplot row.

    choices retains original trial IDs and elapsed/UTC-second choice times.
    contexts is the elapsed-second interval table from context_intervals.
    Custom data has shape (selected choices, 3): trial ID, UTC time, context.
    """
    for action, label, color in ((1, "Left", "#2563eb"), (0, "Right", "#c2410c")):
        selected = choices.loc[choices.action == action]
        custom = np.empty((len(selected), 3), dtype=object)
        custom[:, :2] = selected[["trial_id", "choice_time"]].to_numpy()
        custom[:, 2] = _choice_context_labels(selected.elapsed_s.to_numpy(), contexts)
        figure.add_trace(
            go.Scatter(
                x=selected.elapsed_s,
                y=np.full(len(selected), action),
                mode="markers",
                name=f"{label} choices",
                customdata=custom,
                marker={
                    "color": color,
                    "size": 9,
                    "symbol": "triangle-up" if action else "triangle-down",
                },
                hovertemplate=f"{label} choice<br>Trial %{{customdata[0]}}<br>Elapsed %{{x:.3f}} s"
                "<br>UTC %{customdata[1]:.6f} s<br>Active context: %{customdata[2]}<extra></extra>",
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
