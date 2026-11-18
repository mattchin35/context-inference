"""Interactive and PNG choice figures built from the same source-row payload."""

from __future__ import annotations

from pathlib import Path
import textwrap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .examples import SequenceCondition, extract_example_trials


def build_example_view(
    trial_df: pd.DataFrame, episode_positions: np.ndarray, *, session_id: str,
    condition: SequenceCondition, trials_before: int = 5, trials_after: int = 5,
    probe_return_outcome: str = "rewarded_or_omission",
) -> dict:
    """Build an in-memory, light figure payload from full (N,C) source trials.

    episode_positions is an integer (S,) original-row vector; condition is
    (family, initial_side, repeats-or-None). Context counts are source rows.
    Source action/state use 0=right, 1=left; choice_time remains UTC seconds.
    Return a dict with display rows (M,C'), original-position x, choice y in
    {0,1,NaN}, separate outcome/annotation masks, context spans and caption.
    Matching, time conversion, source mutation and file IO never occur here.
    """
    rows = extract_example_trials(trial_df, episode_positions, trials_before=trials_before, trials_after=trials_after)
    x = rows.row_position.to_numpy(dtype=float)
    actions = rows.action.to_numpy(dtype=float)
    valid = np.isin(actions, [0, 1])
    y = np.where(valid, actions, np.nan)
    reward = rows.reward.to_numpy(dtype=float)
    manual = rows.experimenter_reward_given.to_numpy(dtype=float)
    series = [
        {"name": "Rewarded", "mask": valid & (reward == 1), "color": "#202020", "symbol": "circle", "mpl": "o", "open": False},
        {"name": "Unrewarded", "mask": valid & (reward == 0), "color": "#202020", "symbol": "circle-open", "mpl": "o", "open": True},
        {"name": "Unknown reward", "mask": valid & ~np.isin(reward, [0, 1]), "color": "#888888", "symbol": "diamond", "mpl": "D", "open": False},
        {"name": "Incorrect", "mask": valid & (rows.correct.to_numpy() == 0), "color": "#bd3030", "symbol": "x", "mpl": "x", "open": False},
        {"name": "Manual reward", "mask": valid & np.isfinite(manual) & (manual != 0), "color": "#a46c00", "symbol": "star", "mpl": "*", "open": False},
    ]
    # Insert separators at identity gaps without removing either choice marker.
    line_x, line_y = [], []
    ids = pd.to_numeric(rows.trial_id, errors="coerce").to_numpy(dtype=float)
    for index in range(len(rows)):
        if index and (not np.isfinite(ids[index - 1:index + 1]).all() or ids[index] != ids[index - 1] + 1):
            line_x.append(np.nan)
            line_y.append(np.nan)
        line_x.append(x[index])
        line_y.append(y[index])
    episode_ids = rows.loc[rows.in_episode, "trial_id"].tolist()
    family, side, repeats = condition
    title = f"{session_id}: {family.replace('_', ' ').title()} ({side} start"
    title += f", x={repeats})" if repeats is not None else ")"
    definition = _pattern_definition(family, probe_return_outcome)
    caption = (
        f"Session {session_id}; episode trials {episode_ids[0]}-{episode_ids[-1]}. "
        f"{definition} Equal spacing represents trial order, not elapsed time. "
        "Filled/open circles show reward/no reward; red crosses show incorrect choices. "
        "Background shows active context; top band marks the episode."
    )
    if rows.attrs["context_truncated"]:
        caption += " Requested surrounding context was truncated at a session boundary."
    return {
        "rows": rows, "x": x, "y": y, "line_x": line_x, "line_y": line_y,
        "series": series, "context_spans": _context_spans(rows), "title": title, "caption": caption,
        "episode_trial_ids": episode_ids, "episode_bounds": (float(episode_positions[0]) - .5, float(episode_positions[-1]) + .5),
        "condition": condition, "trials_before": trials_before, "trials_after": trials_after,
    }


def plotly_figure(view: dict) -> go.Figure:
    """Render a build_example_view payload as an interactive light Plotly Figure.

    Choices retain 0=right/1=left on y; x is original row position labeled by
    source trial ID. Hover retains UTC choice time and source identities.
    Return a new Figure without IO, selection or mutation of view arrays.
    """
    figure = go.Figure()
    for start, stop, color, _label in view["context_spans"]:
        figure.add_vrect(x0=start, x1=stop, fillcolor=color, opacity=.13, line_width=0, layer="below")
    figure.add_shape(type="rect", x0=view["episode_bounds"][0], x1=view["episode_bounds"][1],
                     y0=1.04, y1=1.08, yref="paper", fillcolor="#c49000", line_width=0)
    labels_seen = set()
    for _start, _stop, color, label in view["context_spans"]:
        if label not in labels_seen:
            figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=label,
                                        marker={"symbol": "square", "color": color, "size": 12}, hoverinfo="skip"))
            labels_seen.add(label)
    figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="Matched episode",
                                marker={"symbol": "square", "color": "#c49000", "size": 12}, hoverinfo="skip"))
    figure.add_trace(go.Scatter(x=view["line_x"], y=view["line_y"], mode="lines", name="Choices",
                                line={"color": "#777777", "width": 1.5}, connectgaps=False, hoverinfo="skip"))
    rows = view["rows"]
    fields = ["trial_id", "row_position", "index_label", "choice_time", "correct", "reward", "experimenter_reward_given", "state_int", "cur_block"]
    for series in view["series"]:
        mask = series["mask"]
        if not mask.any():
            continue
        figure.add_trace(go.Scatter(
            x=view["x"][mask], y=view["y"][mask], mode="markers", name=series["name"],
            marker={"symbol": series["symbol"], "color": series["color"], "size": 13,
                    "line": {"width": 1.6, "color": series["color"]}},
            customdata=rows.loc[mask, fields].to_numpy(),
            hovertemplate=("Trial %{customdata[0]} (row %{customdata[1]}, label %{customdata[2]})"
                           "<br>Choice time %{customdata[3]} UTC Unix s<br>Correct %{customdata[4]}, reward %{customdata[5]}"
                           "<br>Manual reward %{customdata[6]}<br>Context %{customdata[7]}, block %{customdata[8]}<extra>%{fullData.name}</extra>"),
        ))
    figure.update_layout(template="plotly_white", title=view["title"], paper_bgcolor="white", plot_bgcolor="white",
                         font={"size": 17, "color": "black"}, height=440,
                         margin={"l": 65, "r": 25, "t": 95, "b": 65}, legend={"orientation": "h", "y": -.25})
    figure.update_xaxes(title_text="Trial", tickvals=view["x"], ticktext=[str(item) for item in rows.trial_id],
                        range=[view["x"][0] - .6, view["x"][-1] + .6], showgrid=False)
    figure.update_yaxes(title_text="Choice", tickvals=[0, 1], ticktext=["Right", "Left"], range=[-.3, 1.3], showgrid=False)
    return figure


def matplotlib_figure(view: dict) -> Figure:
    """Render the same (M,) choices/masks/spans as an opaque white Matplotlib Figure.

    Input is build_example_view's dict; x remains source-row position labeled
    with original trial IDs, y remains 0=right/1=left. Return an independent
    Figure with Agg canvas, readable labels, legend and caption; no file IO.
    """
    figure = Figure(figsize=(13, 5.8), facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set_facecolor("white")
    labels_seen = set()
    for start, stop, color, label in view["context_spans"]:
        axis.axvspan(start, stop, color=color, alpha=.13, label=label if label not in labels_seen else None)
        labels_seen.add(label)
    axis.axvspan(*view["episode_bounds"], ymin=.96, ymax=1, color="#c49000", label="Matched episode")
    axis.plot(view["line_x"], view["line_y"], color="#777777", linewidth=1.3)
    for series in view["series"]:
        mask = series["mask"]
        if not mask.any():
            continue
        style = {"facecolors": "white", "edgecolors": series["color"]} if series["open"] else {"color": series["color"]}
        axis.scatter(view["x"][mask], view["y"][mask], marker=series["mpl"], s=90,
                     linewidths=1.6, label=series["name"], zorder=4, **style)
    axis.set_xticks(view["x"], labels=[str(item) for item in view["rows"].trial_id])
    axis.set_yticks([0, 1], labels=["Right", "Left"])
    axis.set_xlim(view["x"][0] - .6, view["x"][-1] + .6)
    axis.set_ylim(-.3, 1.3)
    axis.set_xlabel("Trial", fontsize=16, color="black")
    axis.set_ylabel("Choice", fontsize=16, color="black")
    axis.set_title(view["title"], fontsize=18, color="black", pad=18)
    axis.tick_params(labelsize=13, colors="black")
    for spine in axis.spines.values():
        spine.set_color("black")
    legend = axis.legend(loc="upper center", bbox_to_anchor=(.5, -.22), ncol=4, fontsize=12,
                         facecolor="white", edgecolor="#bbbbbb", labelcolor="black", framealpha=1)
    for text in legend.get_texts():
        text.set_color("black")
    figure.subplots_adjust(left=.08, right=.98, top=.86, bottom=.40)
    # Leave horizontal room for full session IDs and wider caption words.
    figure.text(.04, .035, "\n".join(textwrap.wrap(view["caption"], 125)), fontsize=12, color="black", va="bottom")
    return figure


def export_png(view: dict, output_path: Path | str, *, dpi: float = 200) -> Path:
    """Save a new opaque PNG from a view payload and return its Path.

    dpi is positive finite dots/inch. Physical trial/time units are unchanged.
    Existing paths raise FileExistsError; only .png destinations are accepted.
    """
    path = Path(output_path)
    if path.suffix.lower() != ".png" or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("Export requires a .png path and positive finite dpi.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = matplotlib_figure(view)
    with path.open("xb") as stream:
        figure.savefig(stream, format="png", dpi=dpi, facecolor="white", transparent=False)
    return path


def _context_spans(rows: pd.DataFrame) -> list[tuple[float, float, str, str]]:
    """Map aligned (M,C) display rows to source-position context spans and labels.

    state_int uses 0=right/1=left/2=dark; cur_block is unitless. Missing/dark
    context is shaded neutral gray. Return (left,right,color,label) tuples;
    block transitions split spans even when the active side is unchanged.
    """
    styles = {0: ("#d07926", "Right context"), 1: ("#386cb0", "Left context")}
    spans = []
    start = 0
    states = rows.state_int.to_numpy()
    blocks = rows.cur_block.to_numpy()
    for stop in range(1, len(rows) + 1):
        if stop < len(rows) and states[stop] == states[start] and blocks[stop] == blocks[start]:
            continue
        color, label = styles.get(states[start], ("#999999", "Dark/unknown context"))
        spans.append((float(rows.row_position.iloc[start]) - .5, float(rows.row_position.iloc[stop - 1]) + .5, color, label))
        start = stop
    return spans


def _pattern_definition(family: str, probe_return_outcome: str) -> str:
    """Map unitless family/return-policy strings to the actual scientific figure definition."""
    if family in ("probe", "omission_probe"):
        outcome = "correct rewarded or omission (known reward 0/1)" if probe_return_outcome == "rewarded_or_omission" else "correct rewarded"
        prefix = "An omission precedes the excursion. " if family == "omission_probe" else ""
        return f"{prefix}Probe return: {outcome}; block and active context stay unchanged."
    if family == "ideal_transition":
        return "Context changes after the initial reward; incorrect repeats end with a rewarded opposite-side choice."
    if family == "lucky_omission_switch":
        return "An omission precedes the context change and the immediate rewarded opposite-side choice."
    return "Context changes after the initial reward; incorrect repeats precede rewarded switching, a wrong-side return, and rewarded switching again."
