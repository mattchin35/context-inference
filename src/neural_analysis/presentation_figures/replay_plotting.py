"""Render Figure 4 choice summaries and Figure 6 movement replays and correlations."""

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
OUTCOME_CODES = ("L+", "R+", "L-", "R-")
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
    kind: str = "line"


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
    observed: tuple[tuple[float, str], ...] = ()


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
    """Build behavior markers and one mean probability vector per choice/region/width."""
    motif = episode.motif
    title = MOTIF_NAMES.get(motif, motif.replace("_", " ").capitalize())
    lanes = ("Recorded behavior", "PFC probability", "HPC probability")
    if episode.status != "available":
        return _Replay(title, f"{session}. {episode.reason.replace('_', ' ').capitalize()}.", (), ())
    extent = (float(episode.start_elapsed_s), float(episode.end_elapsed_s))
    selected = {name: fields[name].loc[fields[name].motif.eq(motif)] for name in ("tracks", "events", "trials")}
    traces = _regional_behavior(selected["events"], selected["trials"], extent)
    tracks = selected["tracks"]
    trials = selected["trials"].set_index("row_position")
    for (width, region, target), rows in tracks.groupby(["width_ms", "region", "target"], sort=False):
        lane = 1 if region == "PFC" else 2
        # A choice contributes only if every bin and class in its fitted window is available.
        outcome = target == "outcome"
        columns = [f"p_{i}" for i in range(4 if outcome else 2)]
        grouped = rows.groupby("row_position")[columns]
        n_bins = (1000 if outcome else 2000) // width
        complete = grouped.count().eq(n_bins).all(axis=1)
        means = grouped.mean().loc[complete]
        if means.empty:
            continue
        choices = trials.loc[means.index]
        x = choices.choice_elapsed_s.to_list()
        detail = [f"Trial {trial_id}" for trial_id in choices.trial_id]
        if target == "outcome":
            for class_id, label in enumerate(OUTCOMES):
                traces.append(_Trace(lane, f"{region}:outcome:{class_id}", label, PALETTE[class_id],
                                     x, means[f"p_{class_id}"].to_list(), detail, int(width), "bar"))
        else:
            label = "Right context" if target == "context" else "Right choice"
            color = PALETTE[0] if target == "context" else PALETTE[1]
            traces.append(_Trace(lane, f"{region}:{target}", label, color,
                                 x, means.p_0.to_list(), detail, int(width), "point"))
    widths = tuple(w for w in (500, 100) if any(t.width_ms == w for t in traces))
    observed = ()
    caption = (
        f"{session}, trials {episode.first_trial_id} to {episode.last_trial_id}. "
        "The decoders held out all blocks in this excerpt. "
    )
    if motif != "four_outcome":
        caption += "Each point averages right-side probability from 1 s before to 1 s after the marked choice."
    if motif == "ideal_transition" and pd.notna(episode.repeat_count):
        caption += f" {int(episode.repeat_count)} incorrect old-side repeat choices."
    if motif == "four_outcome":
        caption += ("Each bar averages four outcome probabilities over the first second after choice. "
                    "Labels show observed side (L/R) and reward (+/-).")
        observed = tuple((float(t.choice_elapsed_s), OUTCOME_CODES[int(t.outcome)]
                          if t.outcome in range(4) else "?")
                         for t in trials.itertuples() if pd.notna(t.choice_elapsed_s))
        types = selected["trials"].loc[selected["trials"].in_motif, "trial_type"].drop_duplicates()
        caption += " Trial types in the sequence include " + ", ".join(types.str.replace("_", " ")) + "."
    if episode.missing_deliveries:
        caption += f" {int(episode.missing_deliveries)} rewarded choices lack a recorded delivery time."
    if episode.reason:
        caption += " " + str(episode.reason).replace("_", " ").capitalize() + "."
    if not widths:
        caption += " No choices have a complete decoder window."
    return _Replay(title, caption, lanes, tuple(traces), extent, widths, regional=True, observed=observed)


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
            detail.append(" | ".join(str(value) for value in (
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
                  np.asarray(heights)[valid].tolist(), np.asarray(detail)[valid].tolist(), kind="event")


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
        description = f"Trial {trial_id} | [{left:.3f}, {right:.3f}) s"
        detail.extend((description, description, None))
    return _Trace(lane, series, label, color, x, y, detail, width_ms)


def _interactive(view: _Replay, theme: str) -> go.Figure:
    """Build a Plotly figure from common traces, with fixed event markers and width controls."""
    dark = theme == "dark"
    ink, paper = ("white", "#15191f") if dark else ("#17232f", "white")
    figure = make_subplots(
        rows=max(1, len(view.lanes)), cols=1, shared_xaxes=True, vertical_spacing=.1,
        row_heights=[1., 1.15, 1.15] if view.lanes else None,
    )
    default_width = view.widths_ms[0] if view.widths_ms else None
    for trace in view.traces:
        settings = dict(
            x=trace.x, y=trace.y, name=trace.label, text=trace.detail,
            meta={"series": trace.series, "width_ms": trace.width_ms},
            visible=trace.width_ms in (None, default_width),
            showlegend=trace.lane != 2 or not view.regional,
            legendgroup=trace.label,
            hovertemplate="%{text}<extra>%{fullData.name}</extra>" if trace.kind == "event"
            else "%{text}<br>%{y:.3f}<extra>%{fullData.name}</extra>",
        )
        if trace.kind == "bar":
            glyph = go.Bar(**settings, marker_color=trace.color, width=_outcome_bar_width(view.observed))
        else:
            symbol = EVENT_STYLES[trace.series][1] if trace.kind == "event" else (
                "diamond-open" if trace.series.endswith(":context") else "circle")
            glyph = go.Scatter(**settings, mode="lines" if trace.kind == "line" else "markers",
                               line={"color": trace.color, "width": 2},
                               marker={"color": trace.color, "symbol": symbol,
                                       "size": 13 if symbol == "diamond-open" else 8})
        figure.add_trace(glyph, row=trace.lane + 1, col=1)
    for trace in view.traces:
        if trace.series in VERTICAL_EVENTS:
            for time in trace.x:
                figure.add_vline(x=time, exclude_empty_subplots=False,
                                 line={"color": trace.color, "width": 1, "dash": "dot"})
    for lane, label in enumerate(view.lanes, 1):
        figure.update_yaxes(title_text=label, row=lane, col=1)
    if view.lanes:
        ticks, labels, limits = _behavior_axis(view.regional)
        figure.update_yaxes(tickvals=ticks, ticktext=labels, range=limits, title_text="", row=1, col=1)
        figure.update_xaxes(title_text="Elapsed session time (s)", row=len(view.lanes), col=1)
    if view.regional:
        figure.update_yaxes(range=[-.025, 1.025], row=2, col=1)
        figure.update_yaxes(range=[-.025, 1.025], row=3, col=1)
    if view.observed:
        figure.update_layout(barmode="stack")
        figure.update_xaxes(tickvals=[t for t, _ in view.observed],
                            ticktext=[f"{t:.1f}<br>{label}" for t, label in view.observed],
                            row=3, col=1)
    if view.extent is not None:
        figure.update_xaxes(range=list(view.extent))
    figure.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor=ink)
    title = view.title + (f" | {default_width} ms decoder" if view.widths_ms else "")
    if view.widths_ms:
        buttons = [
            dict(label=f"{width} ms", method="update", args=[
                {"visible": [trace.width_ms in (None, width) for trace in view.traces]},
                {"title.text": f"{view.title} | {width} ms decoder"},
            ]) for width in view.widths_ms
        ]
        figure.update_layout(updatemenus=[dict(buttons=buttons, x=1., xanchor="right", y=1.17)])
    if not view.lanes:
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
    figure.update_layout(
        title=title, template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=paper, plot_bgcolor=paper, font={"color": ink, "size": 14},
        hovermode="x", hoversubplots="axis", height=740 if view.lanes else 240,
        margin={"l": 150, "b": 205 if view.lanes else 30, "t": 115, "r": 55},
        legend={"orientation": "h", "y": -.16, "yanchor": "top"},
        annotations=[dict(x=0, y=-.33 if view.lanes else .5, xref="paper", yref="paper",
                          xanchor="left", yanchor="top", align="left", showarrow=False,
                          text=fill(view.caption, 107).replace("\n", "<br>"))],
    )
    return figure


def _outcome_bar_width(observed: tuple[tuple[float, str], ...]) -> float:
    """Return a bar width in seconds below the minimum spacing of labeled choices."""
    gaps = np.diff(np.unique([time for time, _ in observed]))
    return min(1.2, .7 * gaps.min()) if len(gaps) else 1.2


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
        default_width = view.widths_ms[0] if view.widths_ms else None
        bottoms = {}
        for trace in view.traces:
            if trace.width_ms not in (None, default_width):
                continue
            if trace.kind == "bar":
                bottom = bottoms.setdefault(trace.lane, np.zeros(len(trace.x)))
                axes[trace.lane].bar(trace.x, trace.y, bottom=bottom, color=trace.color,
                                     width=_outcome_bar_width(view.observed), label=trace.label)
                bottoms[trace.lane] = bottom + trace.y
                continue
            glyph = None
            if trace.kind == "event":
                glyph = EVENT_STYLES[trace.series][2]
            elif trace.kind == "point":
                glyph = "D" if trace.series.endswith(":context") else "o"
            axes[trace.lane].plot(np.asarray(trace.x, dtype=float), np.asarray(trace.y, dtype=float),
                                  color=trace.color, marker=glyph, linestyle="None" if glyph else "-",
                                  label=trace.label, gid=trace.series, linewidth=1.7,
                                  markerfacecolor="none" if glyph == "D" and trace.kind == "point" else trace.color,
                                  markersize=8 if trace.series.endswith(":context") else 5)
            if trace.series in VERTICAL_EVENTS:
                for time in trace.x:
                    for axis in axes:
                        axis.axvline(time, color=trace.color, lw=.8, ls=":", alpha=.8)
        for lane, (axis, label) in enumerate(zip(axes, view.lanes)):
            axis.set_ylabel(label, fontsize=11)
            axis.grid(axis="x", alpha=.2)
            handles, labels = axis.get_legend_handles_labels()
            if labels and (lane == 0 or view.regional and lane == 1):
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
        if view.observed:
            axes[-1].set_xticks([t for t, _ in view.observed],
                                [f"{t:.1f}\n{label}" for t, label in view.observed],
                                fontsize=9)
        if view.extent is not None:
            axes[-1].set_xlim(*view.extent)
        suffix = f" | {default_width} ms decoder" if view.widths_ms else ""
        figure.suptitle(view.title + suffix, fontsize=17, fontweight="bold")
        figure.text(.1, .027, fill(view.caption, 123), fontsize=10, ha="left", va="bottom")
        figure.subplots_adjust(left=.1, right=.97, bottom=.22 if view.observed else .17,
                               top=.855, hspace=.24)
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
    categories = (("pfc_hpc", "PFC / HPC"), ("pfc_speed", "PFC / speed"), ("hpc_speed", "HPC / speed"))
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
