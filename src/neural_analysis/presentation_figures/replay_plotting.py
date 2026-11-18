"""Render saved Figure 4 and Figure 6 replay bins and block correlations."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MOTIF_NAMES = {
    "new_side_omission": "Switch with a new-side omission",
    "ideal_transition": "Ideal context switch",
    "four_outcome": "Four-outcome sequence",
}
OUTCOMES = ("Left rewarded", "Right rewarded", "Left unrewarded", "Right unrewarded")
PALETTE = ("#3679bd", "#d96d20", "#188b67", "#ab56a2")
EVENT_STYLES = {
    "choice": ("Choice", "circle", "o", "#758393"),
    "delivery": ("Recorded reward", "diamond", "D", "#188b67"),
    "reward_delivery": ("Recorded reward", "diamond", "D", "#188b67"),
    "delivery_missing": ("Reward time missing", "x", "x", "#d96d20"),
    "switch": ("Logged switch", "triangle-down", "v", "#ab56a2"),
    "first_new_side_choice": ("First new-side choice", "triangle-up", "^", "#3679bd"),
    "first_new_side_delivery": ("First new-side reward", "star", "*", "#188b67"),
    "same_side_reward_pair": ("Same-side outcome contrast", "square", "s", "#ab56a2"),
}
VERTICAL_EVENTS = ("switch", "first_new_side_choice", "first_new_side_delivery")


@dataclass(frozen=True)
class _Trace:
    """One saved series on an elapsed-second lane, with null separators in bin traces."""

    lane: int
    series: str
    label: str
    color: str
    x: list[float | None]
    y: list[float | None]
    detail: list[str | None]
    width_ms: int | None = None
    marker: bool = False


@dataclass(frozen=True)
class _Replay:
    """Common display values for both plotting backends, in seconds from trial start origin."""

    title: str
    caption: str
    lanes: tuple[str, ...]
    traces: tuple[_Trace, ...]
    extent: tuple[float, float] | None = None
    widths_ms: tuple[int, ...] = ()
    regional: bool = False


def render_replays(replay_fields: dict, movement_fields: dict, meta: dict, theme: str) -> dict[str, Figure]:
    """Return static figures from saved replay/CCA table dictionaries and session metadata.

    Figure 4 opens at the 500-ms fit. Theme is light/dark. The shared clock is
    elapsed seconds and all intervals and values come from the saved tables.
    """
    figures = {
        key: _static(view, theme)
        for key, view in _replay_views(replay_fields, movement_fields, meta).items()
    }
    if "block_scores" in movement_fields:
        figures["figure6_block_correlations"] = _block_correlations(movement_fields, meta, theme)
    return figures


def interactive_replays(
    replay_fields: dict, movement_fields: dict, meta: dict, theme: str,
) -> dict[str, go.Figure]:
    """Return Plotly replay Figures for the app and self-contained HTML export.

    The tables and axes are identical to render_replays, with a width selector
    for separately fitted regional replay results and an elapsed-time cursor.
    """
    return {
        key: _interactive(view, theme)
        for key, view in _replay_views(replay_fields, movement_fields, meta).items()
    }


def _replay_views(replay: dict, movement: dict, meta: dict) -> dict[str, _Replay]:
    """Normalize saved elapsed/UTC table coordinates to the common replay records."""
    session = str(meta.get("session_id", "Session"))
    views = {}
    for episode in replay.get("episodes", pd.DataFrame()).itertuples(index=False):
        views[f"figure4_{episode.motif}"] = _regional_view(replay, episode, session)
    if "replay_tracks" in movement:
        views["movement"] = _movement_view(movement, meta, session)
    return views


def _regional_view(fields: dict, episode: object, session: str) -> _Replay:
    """Build one episode's behavior markers and per-region/width probability traces."""
    motif = episode.motif
    title = MOTIF_NAMES.get(motif, motif.replace("_", " ").capitalize())
    lanes = ("Recorded behavior", "PFC probability", "HPC probability")
    if episode.status != "available":
        return _Replay(title, f"{session}. {episode.reason.replace('_', ' ').capitalize()}.", (), ())
    extent = (float(episode.start_elapsed_s), float(episode.end_elapsed_s))
    selected = {name: fields[name].loc[fields[name].motif.eq(motif)] for name in ("tracks", "events", "trials")}
    traces = _regional_behavior(selected["events"], selected["trials"], extent)
    tracks = selected["tracks"]
    widths = (500, 100)
    for (width, region, target), rows in tracks.groupby(["width_ms", "region", "target"], sort=False):
        lane = 1 if region == "PFC" else 2
        edges = (rows.bin_left_elapsed_s, rows.bin_right_elapsed_s)
        if target == "outcome":
            for class_id, label in enumerate(OUTCOMES):
                traces.append(_bin_trace(lane, f"{region}:outcome:{class_id}", label, PALETTE[class_id],
                                         edges, rows[f"p_{class_id}"], rows.trial_id, int(width)))
        else:
            label = "Right context" if target == "context" else "Right choice"
            color = PALETTE[0] if target == "context" else PALETTE[1]
            traces.append(_bin_trace(lane, f"{region}:{target}", label, color,
                                     edges, rows.p_0, rows.trial_id, int(width)))
    caption = (
        f"{session}, trials {episode.first_trial_id} to {episode.last_trial_id}. "
        "Probabilities come from fixed fits that held out the entire excerpt blocks. "
        "Each segment shows a measured bin."
    )
    if motif == "ideal_transition" and pd.notna(episode.repeat_count):
        caption += f" {int(episode.repeat_count)} incorrect old-side repeat choices."
    if motif == "four_outcome":
        types = selected["trials"].loc[selected["trials"].in_motif, "trial_type"].drop_duplicates()
        caption += " Trial types in the sequence include " + ", ".join(types.str.replace("_", " ")) + "."
    if episode.missing_deliveries:
        caption += f" {int(episode.missing_deliveries)} rewarded choices lack a recorded delivery time."
    if episode.reason:
        caption += " " + str(episode.reason).replace("_", " ").capitalize() + "."
    if tracks.empty:
        caption += " The saved episode has no supported decoder tracks."
    return _Replay(title, caption, lanes, tuple(traces), extent, widths, regional=True)


def _regional_behavior(
    events: pd.DataFrame, trials: pd.DataFrame, extent: tuple[float, float],
) -> list[_Trace]:
    """Build logged context steps and choice/reward event traces on elapsed-second axes."""
    context = events.loc[events.event.eq("context")].sort_values("elapsed_s", kind="stable")
    traces = []
    if not context.empty:
        heights = context.label.map({"left": 2.25, "right": 1.75, "dark": 2.}).to_numpy(dtype=float)
        times = np.maximum(context.elapsed_s.to_numpy(dtype=float), extent[0])
        x = np.column_stack((times, np.r_[times[1:], extent[1]])).ravel().tolist()
        y = np.repeat(heights, 2).tolist()
        detail = np.repeat(context.label, 2).tolist()
        traces.append(_Trace(0, "context", "Logged context", PALETTE[0], x, y, detail))
    lookup = trials.set_index("trial_id")
    for event in EVENT_STYLES:
        subset = events.loc[events.event.eq(event)]
        if subset.empty:
            continue
        event_x = subset.elapsed_s
        if event == "delivery_missing":
            event_x = subset.trial_id.map(lookup.choice_elapsed_s)
        side = subset.label.map({"left": 1.25, "right": .75}).fillna(1.)
        if event in ("delivery", "delivery_missing", "first_new_side_delivery"):
            side = np.full(len(subset), 0.) if event != "delivery_missing" else np.full(len(subset), .22)
        elif event == "switch":
            side = np.full(len(subset), 2.68)
        kinds = subset.trial_id.map(lookup.trial_type).fillna("").str.replace("_", " ")
        detail = []
        for trial_id, side_name, kind in zip(subset.trial_id, subset.label, kinds):
            detail.append(" · ".join(str(value) for value in (
                f"Trial {trial_id}" if pd.notna(trial_id) else "", side_name, kind,
            ) if value))
        traces.append(_event_trace(event, event_x, side, detail))
    return traces


def _movement_view(fields: dict, meta: dict, session: str) -> _Replay:
    """Build one held-out movement replay with UTC bins/markers converted to elapsed seconds."""
    title = "Movement-selected canonical replay"
    tracks, events = fields["replay_tracks"], fields["replay_events"]
    if tracks.empty:
        reason = fields.get("replay_reason", "Replay unavailable.")
        return _Replay(title, f"{session}. {reason}", (), ())
    origin = float(meta["time_origin_utc_s"])
    starts, stops = (tracks[name] - origin for name in ("start_s", "stop_s"))
    traces = [
        _bin_trace(1, "PFC:cca", "PFC", PALETTE[0], (starts, stops), tracks.pfc_score, tracks.trial_id),
        _bin_trace(1, "HPC:cca", "HPC", PALETTE[1], (starts, stops), tracks.hpc_score, tracks.trial_id),
        _bin_trace(2, "speed", "Speed magnitude", PALETTE[2], (starts, stops), tracks.speed_mm_s, tracks.trial_id),
    ]
    for event, subset in events.groupby("event", sort=False):
        height = 1. if event == "choice" else 0.
        traces.append(_event_trace(event, subset.time_s - origin, np.full(len(subset), height),
                                   [f"Trial {trial_id}" for trial_id in subset.trial_id]))
    increase = fields["replay_speed_increase_mm_s"]
    units = fields["speed_units"]
    caption = (
        f"{session}, block {tracks.block_id.iloc[0]}, trials {tracks.trial_id.iloc[0]} to {tracks.trial_id.iloc[-1]}. "
        f"Mean speed magnitude increased by {increase:.3g} {units} from the first two choices to the last two. "
        "The canonical pair was fitted and oriented using the remaining blocks. "
        "Each segment shows a measured 100-ms bin."
    )
    return _Replay(title, caption, ("Recorded behavior", "Canonical score (SD)", f"Speed ({units})"),
                   tuple(traces), (float(starts.min()), float(stops.max())))


def _event_trace(event: str, times: pd.Series, heights: np.ndarray, detail: list[str]) -> _Trace:
    """Return an elapsed-second marker trace for finite event positions on the behavior lane."""
    label, _, _, color = EVENT_STYLES[event]
    valid = np.isfinite(times)
    return _Trace(0, event, label, color, np.asarray(times)[valid].tolist(),
                  np.asarray(heights)[valid].tolist(), np.asarray(detail)[valid].tolist(), marker=True)


def _bin_trace(
    lane: int, series: str, label: str, color: str,
    edges: tuple[pd.Series, pd.Series], values: pd.Series, trial_ids: pd.Series, width_ms: int | None = None,
) -> _Trace:
    """Represent each saved bin as a horizontal segment followed by a null separator."""
    x, y, detail = [], [], []
    for left, right, value, trial_id in zip(*edges, values, trial_ids):
        if not np.isfinite(value):
            continue
        x.extend((float(left), float(right), None))
        y.extend((float(value), float(value), None))
        description = f"Trial {trial_id} · [{left:.3f}, {right:.3f}) s"
        detail.extend((description, description, None))
    return _Trace(lane, series, label, color, x, y, detail, width_ms)


def _interactive(view: _Replay, theme: str) -> go.Figure:
    """Build a Plotly figure from common traces, with fixed event markers and width controls."""
    dark = theme == "dark"
    ink, paper = ("white", "#15191f") if dark else ("#17232f", "white")
    figure = make_subplots(
        rows=max(1, len(view.lanes)), cols=1, shared_xaxes=True, vertical_spacing=.1,
        row_heights=[.85, 1.15, 1.15] if view.lanes else None,
    )
    for trace in view.traces:
        shape = EVENT_STYLES[trace.series][1] if trace.marker else "circle"
        figure.add_trace(go.Scatter(
            x=trace.x, y=trace.y, name=trace.label, text=trace.detail,
            meta={"series": trace.series, "width_ms": trace.width_ms},
            mode="markers" if trace.marker else "lines", connectgaps=False,
            line={"color": trace.color, "width": 2,
                  "dash": "dot" if trace.series.endswith(":context") else "solid"},
            marker={"color": trace.color, "symbol": shape, "size": 9},
            visible=trace.width_ms in (None, 500),
            showlegend=trace.lane != 2 or not view.regional,
            legendgroup=trace.label,
            hovertemplate="%{text}<extra>%{fullData.name}</extra>" if trace.marker
            else "%{text}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
        ), row=trace.lane + 1, col=1)
        if trace.series in VERTICAL_EVENTS:
            for time in trace.x:
                figure.add_vline(x=time, line={"color": trace.color, "width": 1, "dash": "dot"})
    for lane, label in enumerate(view.lanes, 1):
        figure.update_yaxes(title_text=label, row=lane, col=1)
    if view.lanes:
        ticks, labels, limits = _behavior_axis(view.regional)
        figure.update_yaxes(tickvals=ticks, ticktext=labels, range=limits, title_text="", row=1, col=1)
        figure.update_xaxes(title_text="Elapsed session time (s)", row=len(view.lanes), col=1)
    if view.regional:
        figure.update_yaxes(range=[-.025, 1.025], row=2, col=1)
        figure.update_yaxes(range=[-.025, 1.025], row=3, col=1)
    if view.extent is not None:
        figure.update_xaxes(range=list(view.extent))
    figure.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor=ink)
    title = view.title + (" · 500 ms" if view.widths_ms else "")
    if view.widths_ms:
        buttons = [
            dict(label=f"{width} ms", method="update", args=[
                {"visible": [trace.width_ms in (None, width) for trace in view.traces]},
                {"title.text": f"{view.title} · {width} ms"},
            ]) for width in view.widths_ms
        ]
        figure.update_layout(updatemenus=[dict(buttons=buttons, x=1., xanchor="right", y=1.17)])
    if not view.lanes:
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    figure.update_layout(
        title=title, template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper, plot_bgcolor=paper, font={"color": ink, "size": 14},
        hovermode="x", hoversubplots="axis", height=635 if view.lanes else 240,
        margin={"l": 150, "b": 110, "t": 115, "r": 55},
        legend={"orientation": "h", "y": -.18},
        annotations=[dict(x=0, y=-.29 if view.lanes else .5, xref="paper", yref="paper",
                          xanchor="left", align="left", showarrow=False, text=fill(view.caption, 118).replace("\n", "<br>"))],
    )
    return figure


def _static(view: _Replay, theme: str) -> Figure:
    """Draw the default-width replay as a Matplotlib Figure with the same gaps and markers."""
    style = "dark_background" if theme == "dark" else "default"
    with plt.style.context(style):
        figure, axes = plt.subplots(
            max(1, len(view.lanes)), 1, sharex=True, squeeze=False,
            figsize=(12, 7 if view.lanes else 2.6),
            height_ratios=[.85, 1.15, 1.15] if view.lanes else None,
        )
        axes = axes[:, 0]
        for trace in view.traces:
            if trace.width_ms not in (None, 500):
                continue
            glyph = EVENT_STYLES[trace.series][2] if trace.marker else None
            linestyle = ":" if trace.series.endswith(":context") else "-"
            axes[trace.lane].plot(np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float),
                                  color=trace.color, marker=glyph, linestyle="None" if glyph else linestyle,
                                  label=trace.label, gid=trace.series, markersize=6., linewidth=1.7)
            if trace.series in VERTICAL_EVENTS:
                for time in trace.x:
                    for axis in axes:
                        axis.axvline(time, color=trace.color, lw=.8, ls=":", alpha=.8)
        for lane, (axis, label) in enumerate(zip(axes, view.lanes)):
            axis.set_ylabel(label, fontsize=11)
            axis.grid(axis="x", alpha=.2)
            handles, labels = axis.get_legend_handles_labels()
            if labels and lane == 0:
                axis.legend(handles, labels, loc="lower left", bbox_to_anchor=(0., 1.01),
                            fontsize=8, ncol=min(4, len(labels)), frameon=False)
            elif labels and (lane != 2 or not view.regional):
                axis.legend(handles, labels, loc="upper right", fontsize=8, ncol=min(4, len(labels)))
        if view.lanes:
            ticks, labels, limits = _behavior_axis(view.regional)
            axes[0].set(yticks=ticks, yticklabels=labels, ylim=limits, ylabel="")
            axes[-1].set_xlabel("Elapsed session time (s)")
        else:
            axes[0].set_axis_off()
        if view.regional:
            for axis in axes[1:]:
                axis.set_ylim(-.025, 1.025)
        if view.extent is not None:
            axes[-1].set_xlim(*view.extent)
        suffix = " · 500 ms" if view.widths_ms else ""
        figure.suptitle(view.title + suffix, fontsize=17, fontweight="bold")
        figure.text(.1, .027, fill(view.caption, 123), fontsize=10, ha="left", va="bottom")
        figure.subplots_adjust(left=.1, right=.97, bottom=.17, top=.855, hspace=.24)
    return figure


def _behavior_axis(regional: bool) -> tuple[list[float], list[str], tuple[float, float]]:
    """Return categorical behavior lane tick coordinates, human labels and plot bounds."""
    if regional:
        return [0., .75, 1.25, 1.75, 2.25], ["Reward", "Right choice", "Left choice", "Right context",
                                            "Left context"], (-.25, 3.05)
    return [0., 1.], ["Reward", "Choice"], (-.4, 1.6)


def _block_correlations(fields: dict, meta: dict, theme: str) -> Figure:
    """Plot one dot per finite held-out block Pearson r and each metric's median/support."""
    scores = fields["block_scores"]
    style = "dark_background" if theme == "dark" else "default"
    categories = (("pfc_hpc", "PFC – HPC"), ("pfc_speed", "PFC – speed"), ("hpc_speed", "HPC – speed"))
    with plt.style.context(style):
        figure, axis = plt.subplots(figsize=(8, 5.5))
        labels = []
        for index, (metric, label) in enumerate(categories):
            r = scores.loc[scores.metric.eq(metric), "correlation"].to_numpy(dtype=float)
            finite = r[np.isfinite(r)]
            labels.append(f"{label}\n{len(finite)}/{len(r)} blocks")
            if len(finite):
                x = np.linspace(index - .12, index + .12, len(finite)) if len(finite) > 1 else [index]
                axis.scatter(x, finite, color=PALETTE[index], alpha=.75, s=24)
                median = float(np.median(finite))
                axis.plot([index - .23, index + .23], [median, median], color=PALETTE[index],
                          lw=3, gid=f"{metric}:median")
            elif len(r):
                axis.text(index, 0., "Unavailable", ha="center", va="bottom", fontsize=9)
        axis.axhline(0., alpha=.25, lw=1)
        axis.set(xticks=range(len(categories)), xticklabels=labels, ylim=(-1.05, 1.05),
                 xlim=(-.5, 2.5), ylabel="Held-out Pearson r")
        axis.set_title("Shared activity and movement by held-out block", fontsize=15, fontweight="bold")
        speed_bins = fields.get("n_complete_speed_bins")
        coverage = "" if speed_bins is None else f" Speed covers {speed_bins:,}/{fields['n_paired_bins']:,} neural bins."
        reason = fields.get("speed_reason", "")
        caption = (
            f"{meta.get('session_id', 'Session')}. Horizontal bars show medians. "
            f"Speed is time-weighted magnitude ({fields.get('speed_units', 'mm/s')}).{coverage}"
        )
        if reason:
            caption += " " + reason
        if scores.empty:
            caption += " " + fields.get("replay_reason", "No paired blocks were available.")
        figure.text(.11, .035, fill(caption, 90), fontsize=9, va="bottom")
        figure.subplots_adjust(left=.13, right=.96, bottom=.23, top=.9)
    return figure
