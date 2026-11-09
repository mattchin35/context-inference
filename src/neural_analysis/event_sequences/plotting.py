"""Shared event heatmap payloads, Plotly inspection and opaque Matplotlib PNGs.

Source scaling happens once across the eligible session reference. View builders
only assemble saved motifs/identities; renderers consume those same arrays.
Transition offsets are display coordinates, never physical elapsed session time.
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path
import textwrap

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure as MatplotlibFigure
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import activity, io


SEGMENT_GAP_WIDTH = 0.25  # Display-coordinate units; never an elapsed-time interval.
_PREVIOUS = ("correct_rewarded", "omission", "incorrect")
_FOLLOWING = ("repeat", "switch")
_CENTRAL_FIELDS = ("row_position", "trial_id", "index_label", "choice_time", "side", "outcome", "local_time_s")
_TRANSITION_FIELDS = (
    "trigger_row_position", "trigger_trial_id", "trigger_index_label", "episode_trial_ids",
    "direction", "trigger_side", "final_side", "segment_index", "segment_row_position",
    "segment_trial_id", "segment_index_label", "segment_choice_time", "local_time_s",
)


def prepare_display_source(packet: dict, *, pc_rank: int = 1, zscore: bool = False) -> dict:
    """Select/scale one loaded source once before changing behavioral view selectors.

    Input packet is the saved-data dict from io.load_event_sequences; pc_rank
    is a one-based fixed-axis rank and zscore a boolean display option. Return
    scale_activity's values (W,B)/original row_positions (W,)/relative-second
    edges (B+1,), global limits/units/reference availability, plus saved source
    identity fields. Units are Hz, unwhitened PC scores or source SD units.
    No source files, binning, fitting or result publication occur.
    """
    selected = io.get_source_activity(packet, pc_rank=pc_rank)
    source = activity.scale_activity(selected, zscore=zscore)
    meta = packet["meta"]
    parameters = meta["parameters"]
    source.update(
        session_id=meta["session_id"], configuration_id=meta["configuration_id"],
        probe_id=parameters["probe_id"], population_quality=parameters["population_quality"],
        bin_width_s=parameters["bin_width_s"], pc_rank=pc_rank if selected.source_kind == "pc" else None,
        zscore=zscore, pca_reference=deepcopy(meta["pca_reference"]),
        packet_source_label=meta["source_label"], bound_source_label=selected.source_label,
    )
    return source


def build_central_view(
    packet: dict, display_source: dict, *,
    central_outcome: str = "correct_rewarded", side: str = "left",
) -> dict:
    """Return the common six-panel central view without changing session scaling.

    Inputs are a loaded packet and matching prepared display_source; selectors
    are correct_rewarded/omission and left/right. Return a view dict with six
    row-major panels: previous correct_rewarded/omission/incorrect by next
    repeat/switch. Panels contain values (M,B) or None when unavailable,
    original row_positions (M,), labels, counts, message/title, and object
    customdata (M,B,H) with named hover_fields. UTC choice_time is absolute;
    local_time_s is choice-relative seconds. Global limits/units/caption are
    common to both renderers and preserved for empty conditions.
    """
    _validate_source_binding(packet, display_source)
    if central_outcome not in ("correct_rewarded", "omission") or side not in ("left", "right"):
        raise ValueError("Central view requires correct_rewarded/omission and side left/right.")
    panels = []
    for row, previous in enumerate(_PREVIOUS):
        for col, following in enumerate(_FOLLOWING):
            key = (central_outcome, side, previous, following)
            panel = activity.assemble_heatmap(display_source, packet["central_matches"][key])
            panel.update(condition=key, row=row, col=col)
            relation = "opposite side" if previous == "incorrect" else "same side"
            panel["title"] = f"Prev. {previous.replace('_', ' ')}\n({relation}) / {following}; n={panel['n']}"
            panel["title"] = _coverage_title(panel)
            panel["message"] = _panel_message(panel, "trials")
            positions = panel["row_positions"]
            rows = packet["trials"].iloc[positions]
            panel["row_labels"] = [f"Trial {identity}" for identity in rows.trial_id]
            panel["hover_fields"] = _CENTRAL_FIELDS
            panel["customdata"] = _central_customdata(rows, display_source["bin_edges_s"], side, central_outcome)
            panels.append(panel)
    view = _base_view(display_source, "central", panels)
    view["caption"] = _caption(display_source, panels, f"Central {central_outcome.replace('_', ' ')}; side={side}")
    return view


def build_transition_view(
    packet: dict, display_source: dict, *, direction: str = "left_to_right", repeats: int = 1,
) -> dict:
    """Return one ideal-transition condition with logical values (E,x+2,B).

    Inputs are a loaded packet/matching prepared display source, direction
    left_to_right/right_to_left and repeats x=1..4. Return one-panel view with
    episode positions (E,x+2), trigger labels, counts/reasons and named object
    hover data (E,x+2,B,H). segment_edges (x+2,B+1) and zero_positions (x+2,)
    use shifted display coordinates with fixed SEGMENT_GAP_WIDTH. Every segment
    is its own [-2,+2)-second window; hover/tick times are local seconds. Gaps
    contain no data bins and have no physical elapsed-time interpretation.
    """
    _validate_source_binding(packet, display_source)
    if direction not in ("left_to_right", "right_to_left") or isinstance(repeats, bool) or repeats not in (1, 2, 3, 4):
        raise ValueError("Transition view requires direction left_to_right/right_to_left and repeats 1..4.")
    key = (direction, repeats)
    panel = activity.assemble_heatmap(display_source, packet["transition_matches"][key])
    panel.update(condition=key, row=0, col=0)
    label = "L -> R" if direction == "left_to_right" else "R -> L"
    panel["title"] = f"{label}, x{repeats} incorrect repeats; n={panel['n']}"
    panel["title"] = _coverage_title(panel)
    panel["message"] = _panel_message(panel, "transitions", packet["meta"]["transition_unavailable_reason"])
    positions = panel["row_positions"]
    trials = packet["trials"]
    panel["row_labels"] = [f"Trigger {identity}" for identity in trials.iloc[positions[:, 0]].trial_id]
    panel["hover_fields"] = _TRANSITION_FIELDS
    panel["customdata"] = _transition_customdata(trials, positions, display_source["bin_edges_s"], direction)
    view = _base_view(display_source, "transition", [panel])
    labels = ["Trigger"] + [f"Incorrect\nrepeat {index}" for index in range(1, repeats + 1)] + ["Rewarded\nswitch"]
    view.update(_sequence_geometry(display_source["bin_edges_s"], labels))
    selection = f"Ideal transition {label}; x{repeats} incorrect repeats"
    view["caption"] = _caption(display_source, [panel], selection) + (
        " Each segment uses local choice-relative seconds in [-2,+2). Fixed gaps are display separators; they do not represent elapsed time."
    )
    return view


def build_sequence_view(
    packet: dict, display_source: dict, *, family: str = "ideal_transition",
    initial_side: str = "left", repeats: int | None = 1,
) -> dict:
    """Build one complete multi-trial condition from a loaded source product.

    Parameters
    ----------
    packet : dict
        Loaded io.load_event_sequences packet, including runtime-only
        additional_matches (integer (E,S) original positions), availability
        reason and selector version. Saved trials retain original IDs/labels
        and UTC Unix-second choices. No files are read or matches recomputed.
    display_source : dict
        Matching prepare_display_source output: all-session values (W,B),
        global limits and units (Hz, signed PC scores or source SD units).
    family : str, keyword-only
        ideal_transition, probe, omission_probe, lucky_omission_switch or
        indecisive_transition. The ideal family delegates to the unchanged
        build_transition_view contract and does not need additional fields.
    initial_side : str, keyword-only
        left or right: probe starting side or transition's old active side.
    repeats : int or None, keyword-only
        Exactly x initial incorrect trials: 1..4 for ideal/probes, 1..3 for
        indecisive. Lucky omission-switch requires None because x is absent.

    Returns
    -------
    dict
        Common one-panel kind=transition view with logical values (E_covered,S,B)
        and original row_positions (E_covered,S). Every segment retains its
        [-2,+2) local-second window, zero line and fixed display separator.
        Hover data (E_covered,S,B,H) identify family, episode start, full IDs,
        segment identities and choice path/context direction. Boundary fields
        identify lucky's omission (segment index 1) or indecisive's trigger
        (index 0); probes have no trigger or boundary fields. Counts distinguish
        behavioral matches from whole-episode coverage exclusions. Empty and
        unavailable conditions remain explicit. Source arrays/scaling and
        persisted identities are unchanged; no fitting or publication occurs.
    """
    _validate_source_binding(packet, display_source)
    allowed = ("ideal_transition", "probe", "omission_probe", "lucky_omission_switch", "indecisive_transition")
    if family not in allowed or initial_side not in ("left", "right"):
        raise ValueError("Sequence view requires a supported family and initial side left/right.")
    if family == "lucky_omission_switch":
        if repeats is not None:
            raise ValueError("Lucky omission-switch repeats must be None.")
    else:
        maximum = 3 if family == "indecisive_transition" else 4
        if isinstance(repeats, bool) or not isinstance(repeats, Integral) or not 1 <= repeats <= maximum:
            raise ValueError(f"Sequence family {family} requires integer repeats 1..{maximum}.")
    direction = "left_to_right" if initial_side == "left" else "right_to_left"
    if family == "ideal_transition":
        return build_transition_view(packet, display_source, direction=direction, repeats=repeats)

    key = (family, initial_side, repeats)
    panel = activity.assemble_heatmap(display_source, packet["additional_matches"][key])
    panel.update(condition=key, row=0, col=0)
    path = "L -> R" if initial_side == "left" else "R -> L"
    if family in ("probe", "omission_probe"):
        path += " -> L" if initial_side == "left" else " -> R"
    description = f"{family.replace('_', ' ').capitalize()} {path}"
    if repeats is not None:
        run_label = "initial incorrect trials" if family == "indecisive_transition" else "incorrect trials"
        description += f"; x{repeats} {run_label}"
    panel["title"] = f"{description}; n={panel['n']}"
    panel["title"] = _coverage_title(panel)
    panel["message"] = _panel_message(panel, "sequences", packet["additional_unavailable_reason"])
    positions = panel["row_positions"]
    starts = packet["trials"].iloc[positions[:, 0]]
    panel["row_labels"] = [f"Episode start {identity}" for identity in starts.trial_id]
    panel["hover_fields"], panel["customdata"] = _sequence_customdata(
        packet["trials"], positions, display_source["bin_edges_s"], family, initial_side, repeats,
    )
    view = _base_view(display_source, "transition", [panel])
    view.update(_sequence_geometry(display_source["bin_edges_s"], _sequence_labels(family, repeats)))
    view["caption"] = _caption(display_source, [panel], description) + (
        f" Additional selector version={packet['additional_selector_version']}."
        " Each segment uses local choice-relative seconds in [-2,+2). Fixed gaps are display separators; they do not represent elapsed time."
    )
    return view


def plotly_figure(view: dict) -> go.Figure:
    """Input: common central/transition view. Return light opaque Plotly figure using its exact arrays/limits.

    Central traces use (M,B) values; transitions use one (E,B) trace per trial
    segment, never gap samples. Hover identities/time come only from the common
    payload. Rows are oldest-first at the top. No loading, fitting or writes.
    """
    central = view["kind"] == "central"
    rows, cols = (3, 2) if central else (1, 1)
    figure = make_subplots(rows=rows, cols=cols,
                           subplot_titles=[panel["title"].replace("\n", "<br>") for panel in view["panels"]],
                           vertical_spacing=.12 if central else .08, horizontal_spacing=.22 if central else .1)
    # Long headings and horizontal PC units otherwise squeeze the two-column
    # app grid. Preserve exact unit text while putting it beside the colorbar.
    for annotation in figure.layout.annotations:
        annotation.font.size = 13
    if not central:
        figure.layout.annotations[0].y = 1.22
    for panel in view["panels"]:
        row, col = panel["row"] + 1, panel["col"] + 1
        number = panel["row"] * cols + panel["col"] + 1
        suffix = str(number) if number > 1 else ""
        if panel["n"] and panel["values"] is not None:
            segments = 1 if central else panel["values"].shape[1]
            for segment in range(segments):
                values = panel["values"] if central else panel["values"][:, segment]
                customdata = panel["customdata"] if central else panel["customdata"][:, segment]
                edges = view["bin_edges_s"] if central else view["segment_edges"][segment]
                figure.add_trace(go.Heatmap(
                    x=edges, y=np.arange(panel["n"]), z=values, customdata=customdata,
                    coloraxis="coloraxis", zsmooth=False, connectgaps=False,
                    hovertemplate=_hover_template(panel["hover_fields"], view["value_units"]), showscale=False,
                ), row=row, col=col)
        zeros = [0.0] if central else view["zero_positions"]
        for zero in zeros:
            figure.add_shape(type="line", x0=zero, x1=zero, y0=0, y1=1,
                             xref="x" + suffix, yref="y" + suffix + " domain",
                             line={"color": "black", "width": 1, "dash": "dash"})
        ticks = _row_ticks(panel["n"])
        figure.update_yaxes(range=[max(panel["n"], 1) - .5, -.5], tickvals=ticks,
                            ticktext=[panel["row_labels"][index] for index in ticks],
                            title_text="Trials (oldest first)" if central else "Episodes (oldest first)", row=row, col=col)
        if panel["message"] and not panel["n"]:
            message_lines = textwrap.wrap(panel["message"], 32 if central else 75,
                                          break_long_words=False, break_on_hyphens=False)
            figure.add_annotation(text="<br>".join(message_lines), x=.5, y=.5,
                                  xref="x" + suffix + " domain", yref="y" + suffix + " domain",
                                  showarrow=False, align="center", font={"color": "black", "size": 13})
        _plotly_time_axis(figure, view, row, col)
    coloraxis = {"colorscale": "Viridis" if _positive_colors(view) else "RdBu_r", "colorbar": {"title": {"text": view["value_units"], "side": "right"}}}
    if view["color_limits"] is not None:
        coloraxis.update(cmin=view["color_limits"][0], cmax=view["color_limits"][1])
    figure.update_layout(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 13, "color": "black"},
        coloraxis=coloraxis, height=1150 if central else 600, width=1500 if central else max(1200, len(view["zero_positions"]) * 270),
        margin={"l": 140, "r": 130, "t": 100, "b": 210},
        meta={"caption": view["caption"], "kind": view["kind"]},
    )
    figure.add_annotation(text="<br>".join(_caption_lines(view["caption"], width=85)), x=0, y=-.10 if central else -.25,
                          xref="paper", yref="paper", xanchor="left", yanchor="top",
                          showarrow=False, align="left", font={"size": 12, "color": "black"})
    return figure


def matplotlib_figure(view: dict) -> MatplotlibFigure:
    """Input: common view. Return direct Figure+Agg with the same (M,B)/(E,S,B) arrays and limits.

    No pyplot-managed figures are registered. Images retain actual signed
    activity and oldest-first row order; transition image extents leave white
    display separators. All backgrounds are opaque white and text is black.
    This function renders in memory only; export_png is the explicit writer.
    """
    central = view["kind"] == "central"
    width = 16 if central else max(12, len(view["zero_positions"]) * 3)
    figure = MatplotlibFigure(figsize=(width, 12 if central else 7), facecolor="white")
    FigureCanvasAgg(figure)
    axes = []
    for index, panel in enumerate(view["panels"]):
        axis = figure.add_subplot(3, 2, index + 1) if central else figure.add_subplot(1, 1, 1)
        axes.append(axis)
        axis.set_facecolor("white")
        axis.set_title(panel["title"], fontsize=13, color="black", pad=15 if central else 38)
        if panel["n"] and panel["values"] is not None:
            segments = 1 if central else panel["values"].shape[1]
            for segment in range(segments):
                values = panel["values"] if central else panel["values"][:, segment]
                edges = view["bin_edges_s"] if central else view["segment_edges"][segment]
                axis.imshow(values, extent=(edges[0], edges[-1], panel["n"] - .5, -.5),
                            origin="upper", aspect="auto", interpolation="nearest",
                            cmap="viridis" if _positive_colors(view) else "RdBu_r",
                            vmin=view["color_limits"][0], vmax=view["color_limits"][1])
        for zero in ([0.0] if central else view["zero_positions"]):
            axis.axvline(zero, color="black", linewidth=1, linestyle="--")
        ticks = _row_ticks(panel["n"])
        axis.set_yticks(ticks, labels=[panel["row_labels"][position] for position in ticks])
        axis.set_ylim(max(panel["n"], 1) - .5, -.5)
        axis.set_ylabel("Trials (oldest first)" if central else "Episodes (oldest first)", fontsize=13, color="black")
        axis.tick_params(labelsize=12, colors="black")
        if panel["message"] and not panel["n"]:
            axis.text(.5, .5, "\n".join(textwrap.wrap(panel["message"], 60)), transform=axis.transAxes,
                      ha="center", va="center", fontsize=13, color="black")
        if central:
            axis.set_xlim(-2, 2)
            axis.set_xticks([-2, -1, 0, 1, 2])
            axis.set_xlabel("Time from choice (s)", fontsize=13, color="black")
        else:
            ticks, labels = _transition_ticks(view)
            axis.set_xticks(ticks, labels=labels)
            axis.set_xlim(view["segment_edges"][0, 0], view["segment_edges"][-1, -1])
            axis.set_xlabel("Local time from each choice (s); gaps are display separators", fontsize=13, color="black")
            for zero, label in zip(view["zero_positions"], view["segment_labels"]):
                axis.text(zero, 1.02, label, transform=axis.get_xaxis_transform(), ha="center", va="bottom", fontsize=12, color="black")
    figure.subplots_adjust(left=.13, right=.88, top=.92 if central else .83, bottom=.18 if central else .30, hspace=.65, wspace=.43)
    if view["color_limits"] is not None:
        mappable = ScalarMappable(norm=Normalize(*view["color_limits"]), cmap="viridis" if _positive_colors(view) else "RdBu_r")
        colorbar = figure.colorbar(mappable, ax=axes, fraction=.025, pad=.035)
        colorbar.set_label(view["value_units"], fontsize=13, color="black")
        colorbar.ax.tick_params(labelsize=12, colors="black")
        colorbar.ax.set_facecolor("white")
    # A conservative character budget follows the physical canvas width;
    # narrow transition exports must also fit long source and unit identities.
    figure.text(.03, .025, "\n".join(_caption_lines(view["caption"], width=round(width * 8))),
                fontsize=12, color="black", va="bottom")
    return figure


def export_png(view: dict, output_path: Path | str, *, dpi: float = 150) -> Path:
    """Input: common view/new PNG Path/positive dots-per-inch. Return saved opaque PNG Path.

    Uses matplotlib_figure's exact arrays/limits; no interactive screenshot or
    new rendering dependency. Existing files raise FileExistsError unchanged.
    """
    path = Path(output_path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    if path.suffix.lower() != ".png" or not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("Export requires a new .png path and positive finite dpi.")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = matplotlib_figure(view)
    figure.savefig(path, dpi=dpi, facecolor="white", edgecolor="white", transparent=False)
    return path


# Small payload/format helpers; scientific selection and scaling stay upstream.
def _sequence_geometry(bin_edges_s: np.ndarray, labels: list[str]) -> dict:
    """Return shared geometry from float (B+1,) local-second edges and S labels.

    segment_edges (S,B+1) and zero_positions (S,) use shifted display
    coordinates with fixed gaps. Labels stay in chronological segment order.
    Gaps contain no samples and have no physical elapsed-time interpretation.
    """
    offsets = np.arange(len(labels)) * (4.0 + SEGMENT_GAP_WIDTH)
    return {
        "segment_edges": bin_edges_s[None, :] + offsets[:, None],
        "zero_positions": offsets, "segment_labels": labels,
    }


def _sequence_labels(family: str, repeats: int | None) -> list[str]:
    """Return S readable labels for a validated unitless additional family/x.

    x is 1..4 for probes, 1..3 for indecisive, None for lucky. Labels describe
    actual displayed trials; only indecisive starts at a rewarded trigger.
    """
    if family == "lucky_omission_switch":
        return ["Rewarded\nstart", "Omission\npre-change", "Rewarded\nswitch"]
    if family == "indecisive_transition":
        return ["Trigger"] + [f"Incorrect\nrepeat {index}" for index in range(1, repeats + 1)] + [
            "Rewarded\nswitch", "Incorrect\nreturn", "Rewarded\nswitch",
        ]
    labels = ["Rewarded\nstart"]
    if family == "omission_probe":
        labels.append("Correct\nomission")
    return labels + [f"Incorrect\nexcursion {index}" for index in range(1, repeats + 1)] + ["Rewarded\nreturn"]


def _sequence_customdata(
    trials, positions: np.ndarray, edges_s: np.ndarray,
    family: str, initial_side: str, repeats: int | None,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return named fields and object hover data (E,S,B,H) for one condition.

    trials is the full pandas DataFrame (N,C); positions are integer (E,S)
    original rows and edges_s float (B+1,) local-second edges. family/initial_side/x are
    validated unitless selectors. Original IDs/labels, UTC-second choices and
    zero-based segment indices are preserved, including empty E=0 conditions.
    Probes expose their choice path; boundary families expose context direction
    and the actual pre-boundary row. No source values or arrays are modified.
    """
    episodes, segments = positions.shape
    rows = trials.iloc[positions.reshape(-1)]
    ids = rows.trial_id.to_numpy().reshape(episodes, segments)
    labels = rows.index_label.to_numpy().reshape(episodes, segments)
    full_ids = np.asarray([", ".join(str(identity) for identity in episode) for episode in ids], dtype=object)
    fields = {
        "family": family, "initial_side": initial_side, "repeats": repeats,
        "episode_start_row_position": positions[:, :1, None],
        "episode_start_trial_id": ids[:, :1, None],
        "episode_start_index_label": labels[:, :1, None],
        "episode_trial_ids": full_ids[:, None, None],
        "segment_index": np.arange(segments)[None, :, None],
        "segment_row_position": positions[:, :, None],
        "segment_trial_id": ids[:, :, None],
        "segment_index_label": labels[:, :, None],
        "segment_choice_time": rows.choice_time.to_numpy().reshape(episodes, segments, 1),
        "local_time_s": ((edges_s[:-1] + edges_s[1:]) / 2)[None, None, :],
    }
    if family in ("probe", "omission_probe"):
        fields["choice_path"] = "L -> R -> L" if initial_side == "left" else "R -> L -> R"
    else:
        fields["direction"] = "left_to_right" if initial_side == "left" else "right_to_left"
        boundary = 1 if family == "lucky_omission_switch" else 0
        fields.update(
            pre_boundary_segment_index=boundary,
            pre_boundary_row_position=positions[:, boundary, None, None],
            pre_boundary_trial_id=ids[:, boundary, None, None],
            pre_boundary_index_label=labels[:, boundary, None, None],
        )
    data = np.empty((episodes, segments, len(edges_s) - 1, len(fields)), dtype=object)
    for index, values in enumerate(fields.values()):
        data[..., index] = values
    return tuple(fields), data


def _validate_source_binding(packet: dict, source: dict) -> None:
    """Input: saved packet/prepared source. Return None or reject mixed config/labels/(W,) maps/(B+1,) edges."""
    meta = packet["meta"]
    parameters = meta["parameters"]
    if (source["configuration_id"] != meta["configuration_id"] or source["session_id"] != meta["session_id"]
            or source["source_kind"] != meta["source_kind"] or source["packet_source_label"] != meta["source_label"]
            or source["source_label"] != source["bound_source_label"]
            or source["probe_id"] != parameters["probe_id"] or source["population_quality"] != parameters["population_quality"]
            or source["pca_reference"] != meta["pca_reference"]):
        raise ValueError("Display source identity/configuration does not match this saved packet.")
    if not np.array_equal(source["bin_edges_s"], packet["bin_edges_s"]) or source["bin_width_s"] != parameters["bin_width_s"]:
        raise ValueError("Display source bin geometry does not match this saved packet.")
    if not np.array_equal(source["row_positions"], packet["activity_row_positions"]):
        raise ValueError("Display source row mapping does not match this saved packet.")


def _base_view(source: dict, kind: str, panels: list[dict]) -> dict:
    """Input: prepared source/kind/assembled panels. Return common value units/limits/identity view dict."""
    return {
        "kind": kind, "panels": panels, "bin_edges_s": source["bin_edges_s"].copy(),
        "color_limits": source["color_limits"], "value_units": source["value_units"],
        "source_label": source["source_label"], "source_kind": source["source_kind"],
        "zscore": source["zscore"], "constant_activity": source["constant_activity"],
        "configuration_id": source["configuration_id"], "pca_reference": deepcopy(source["pca_reference"]),
    }


def _panel_message(panel: dict, noun: str, context_reason: str = "") -> str:
    """Input: assembled counts/source availability, trial noun/context reason. Return explicit display message."""
    counts = f"behavioral matches={panel['behavioral_count']}; excluded for incomplete coverage={panel['excluded_count']}; n={panel['n']}"
    if context_reason:
        return context_reason + " " + counts
    if panel["unavailable_reason"]:
        return panel["unavailable_reason"] + " " + counts
    if panel["behavioral_count"] == 0:
        return f"No matching {noun} in this session - n=0"
    if panel["excluded_count"]:
        return (f"No {noun} with complete neural coverage; " if not panel["n"] else "") + counts
    return ""


def _coverage_title(panel: dict) -> str:
    """Input: panel with title/counts. Return title with partial-coverage counts above populated activity."""
    title = panel["title"]
    if panel["n"] and panel["excluded_count"]:
        title += f"\nMatches={panel['behavioral_count']}; coverage excluded={panel['excluded_count']}"
    return title


def _central_customdata(rows, edges_s: np.ndarray, side: str, outcome: str) -> np.ndarray:
    """Input: selected (M,C) rows/(B+1,) relative-second edges/labels. Return object hover array (M,B,H)."""
    data = np.empty((len(rows), len(edges_s) - 1, len(_CENTRAL_FIELDS)), dtype=object)
    for index, field in enumerate(_CENTRAL_FIELDS[:4]):
        data[:, :, index] = rows[field].to_numpy()[:, None]
    data[:, :, 4] = side
    data[:, :, 5] = outcome
    data[:, :, 6] = ((edges_s[:-1] + edges_s[1:]) / 2)[None, :]
    return data


def _transition_customdata(trials, positions: np.ndarray, edges_s: np.ndarray, direction: str) -> np.ndarray:
    """Input: full (N,C) saved rows/int(E,S) map/(B+1,) seconds/direction. Return object hover (E,S,B,H)."""
    episodes, segments = positions.shape
    data = np.empty((episodes, segments, len(edges_s) - 1, len(_TRANSITION_FIELDS)), dtype=object)
    if not episodes:
        return data
    rows = trials.iloc[positions.reshape(-1)]
    ids = rows.trial_id.to_numpy().reshape(episodes, segments)
    labels = rows.index_label.to_numpy().reshape(episodes, segments)
    data[..., 0] = positions[:, :1, None]
    data[..., 1] = ids[:, :1, None]
    data[..., 2] = labels[:, :1, None]
    full_ids = np.asarray([", ".join(str(identity) for identity in episode) for episode in ids], dtype=object)
    data[..., 3] = full_ids[:, None, None]
    data[..., 4] = direction
    data[..., 5] = "left" if direction == "left_to_right" else "right"
    data[..., 6] = "right" if direction == "left_to_right" else "left"
    data[..., 7] = np.arange(segments)[None, :, None]
    data[..., 8] = positions[:, :, None]
    data[..., 9] = ids[:, :, None]
    data[..., 10] = labels[:, :, None]
    data[..., 11] = rows.choice_time.to_numpy().reshape(episodes, segments, 1)
    data[..., 12] = ((edges_s[:-1] + edges_s[1:]) / 2)[None, None, :]
    return data


def _caption(source: dict, panels: list[dict], selection: str) -> str:
    """Input: prepared source/panels/selection. Return saved scientific identities, counts, units and availability caption."""
    caption = (
        f"Session {source['session_id']}; {source['source_label']}; probe={source['probe_id']}; "
        f"population={source['population_quality']}; bin={source['bin_width_s'] * 1000:g} ms; "
        f"{'session-wide z-score' if source['zscore'] else 'raw'} ({source['value_units']}). {selection}. "
        f"n={sum(panel['n'] for panel in panels)}; behavioral matches={sum(panel['behavioral_count'] for panel in panels)}; "
        f"excluded for incomplete coverage={sum(panel['excluded_count'] for panel in panels)}. "
        f"Canonical source configuration={source['configuration_id']}."
    )
    if source["pca_reference"] is not None:
        reference = source["pca_reference"]
        caption += f" Exact PCA reference configuration={reference['configuration_id']}; archive SHA256={reference['sha256'][:12]}."
    if source["constant_activity"]:
        caption += " Constant activity; session SD is zero."
    if source["unavailable_reason"]:
        caption += " " + source["unavailable_reason"]
    return caption


def _hover_template(fields: tuple[str, ...], units: str) -> str:
    """Input: named hover fields/activity units. Return Plotly template using local seconds, never shifted x coordinates."""
    lines = []
    for index, field in enumerate(fields):
        label = field.replace("_", " ")
        suffix = " UTC Unix s" if field in ("choice_time", "segment_choice_time") else " s" if field == "local_time_s" else ""
        lines.append(f"{label}: %{{customdata[{index}]}}{suffix}")
    return "<br>".join(lines) + f"<br>activity: %{{z}} {units}<extra></extra>"


def _row_ticks(count: int) -> np.ndarray:
    """Input: nonnegative row count. Return up to ten uniformly spaced integer trial ticks (T,)."""
    # Integer stride avoids adjacent labels when an uneven count (e.g. 17)
    # cannot support twelve large tick labels without overlap.
    stride = max(1, int(np.ceil(count / 10)))
    return np.arange(0, count, stride, dtype=int)


def _transition_ticks(view: dict) -> tuple[np.ndarray, list[str]]:
    """Input: transition geometry. Return display tick coordinates (3*S,) and repeated local-second labels."""
    ticks = (view["zero_positions"][:, None] + np.array([-1.0, 0.0, 1.0])[None, :]).reshape(-1)
    return ticks, ["-1", "0", "+1"] * len(view["zero_positions"])


def _plotly_time_axis(figure: go.Figure, view: dict, row: int, col: int) -> None:
    """Input: figure/view/one-based subplot coordinates. Return None after setting physical local-time tick labels."""
    if view["kind"] == "central":
        figure.update_xaxes(range=[-2, 2], tickvals=[-2, -1, 0, 1, 2], title_text="Time from choice (s)", row=row, col=col)
    else:
        ticks, labels = _transition_ticks(view)
        figure.update_xaxes(range=[view["segment_edges"][0, 0], view["segment_edges"][-1, -1]], tickvals=ticks,
                            ticktext=labels, title_text="Local time from each choice (s); display separators", row=row, col=col)
        for zero, label in zip(view["zero_positions"], view["segment_labels"]):
            figure.add_annotation(x=zero, y=1.04, xref="x", yref="paper", text=label.replace("\n", "<br>"),
                                  yanchor="bottom", showarrow=False, font={"size": 12, "color": "black"})


def _positive_colors(view: dict) -> bool:
    """Input: common view. Return True for nonnegative raw-unit palette, False for signed PC/z-score palette."""
    return view["source_kind"] == "unit" and not view["zscore"]


def _caption_lines(caption: str, *, width: int = 135) -> list[str]:
    """Input: caption text/maximum character count per line. Return wrapped lines preserving configuration/hash tokens."""
    return textwrap.wrap(caption, width, break_long_words=False, break_on_hyphens=False)
