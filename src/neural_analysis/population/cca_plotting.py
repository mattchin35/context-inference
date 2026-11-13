"""Session-time CCA displays from saved canonical scores; no loading or fitting."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.population.session_cca_io import SessionCCAResult
from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.neural_analysis.population.timeline_plotting import (
    EVENT_LABELS,
    add_timeline_behavior,
    available_event_tracks,
    session_time_bounds,
)


def build_cca_heatmap_data(
    result: SessionCCAResult,
    region: str,
    pair_range: tuple[int, int] | None = None,
    loaded_range_s: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Return true bin edges and selected saved scores for one regional heatmap.

    result contains scores (W,B,K) in canonical session-SD units. region is
    PFC/HPC; pair_range is inclusive one-based ranks, default first min(10,K).
    loaded_range_s optionally selects intersecting complete windows in elapsed
    seconds. Return x (M+1,) elapsed-second edges, y (P,) original ranks, z
    (P,M) scores, and customdata (P,M,8). Hover fields are trial ID, UTC bin
    start/end, choice-relative bin start/end, elapsed bin start/end, and rho.
    Each unmeasured inter-window gap is one NaN column; measured zero remains
    zero. Empty selections return zero columns. Inputs are never modified.
    """
    if result.fit is None or region not in ("PFC", "HPC"):
        raise ValueError("A paired CCA fit and PFC/HPC region are required.")
    ranks = _selected_ranks(result, pair_range)
    edges, columns, custom = [], [], []
    for window_index, choice, elapsed_edges in _display_windows(result, loaded_range_s):
        if edges and elapsed_edges[0] > edges[-1]:
            edges.append(float(elapsed_edges[0]))
            columns.append(np.full((len(ranks), 1), np.nan))
            custom.append(np.full((len(ranks), 1, 8), None, dtype=object))
        edges.extend(elapsed_edges.tolist() if not edges else elapsed_edges[1:].tolist())
        values = result.fit.scores[region][window_index][:, ranks - 1].T
        columns.append(values)
        info = np.empty((*values.shape, 8), dtype=object)
        info[:, :, 0] = choice.trial_id
        info[:, :, 1] = float(choice.choice_time) + result.bin_edges_s[:-1]
        info[:, :, 2] = float(choice.choice_time) + result.bin_edges_s[1:]
        info[:, :, 3] = result.bin_edges_s[:-1]
        info[:, :, 4] = result.bin_edges_s[1:]
        info[:, :, 5] = elapsed_edges[:-1]
        info[:, :, 6] = elapsed_edges[1:]
        info[:, :, 7] = result.fit.correlations[ranks - 1, None]
        custom.append(info)
    return {
        "x": np.asarray(edges), "y": ranks,
        "z": np.concatenate(columns, axis=1) if columns else np.empty((len(ranks), 0)),
        "customdata": np.concatenate(custom, axis=1) if custom else np.empty((len(ranks), 0, 8), dtype=object),
    }


def plot_session_cca(
    result: SessionCCAResult,
    pair_range: tuple[int, int] | None = None,
    selected_pair: int = 1,
    show_traces: bool = True,
    event_tracks: tuple[str, ...] = (),
    color_limit: float | None = None,
    loaded_range_s: tuple[float, float] | None = None,
    viewport_s: tuple[float, float] | None = None,
    viewport_revision: int = 0,
    show_context: bool = True,
) -> go.Figure:
    """Build a light session figure using saved behavior and (W,B,K) CCA scores.

    result is SessionCCAResult; fit=None displays behavior/unavailability only.
    pair_range selects inclusive one-based heatmap ranks (first ten default),
    while selected_pair independently selects the two regional step traces.
    show_traces/show_context and available event_tracks control display rows.
    color_limit is a positive symmetric canonical-SD limit, default the fixed
    full-session maximum across both regions/all pairs. loaded_range_s bounds
    neural payload by intersecting windows; viewport_s sets visible time. Both
    are elapsed-second (low,high) pairs. viewport_revision resets time zoom on
    explicit navigation; row/rank changes reset vertical state independently.
    Return Plotly Figure. Scores, transforms and metadata remain unchanged;
    neither raw sources nor numerical fitting are accessed.
    """
    loaded_range_s = _time_range(loaded_range_s)
    viewport_s = _time_range(viewport_s)
    behavior = SessionTimelineResult(
        result.session_id, result.time_origin_s, result.choices, result.events,
        {}, {}, result.meta,
    )
    full_bounds = session_time_bounds(behavior)
    tracks = tuple(track for track in event_tracks if track in available_event_tracks(result.events))
    titles = ["Choices"] + [EVENT_LABELS[track] for track in tracks]
    heights = [1.] + [.65] * len(tracks)
    ranks = _selected_ranks(result, pair_range) if result.fit is not None else np.array([], dtype=int)
    if result.fit is not None:
        if not isinstance(selected_pair, (int, np.integer)) or not 1 <= selected_pair <= len(result.fit.correlations):
            raise ValueError("Trace pair must be an available one-based canonical rank.")
        for region in ("PFC", "HPC"):
            titles.append(f"{region} ({result.probe_roles[region]}) - canonical pairs {ranks[0]}-{ranks[-1]}")
            heights.append(max(2., len(ranks) * .2))
        if show_traces:
            titles.append(f"CC{selected_pair} - paired traces; in-sample rho = {result.fit.correlations[selected_pair - 1]:.4f}")
            heights.append(1.5)
    titles.append("Session navigator")
    heights.append(.45)
    figure_height = max(740, int(sum(heights) * 90 + 300))
    figure = make_subplots(
        rows=len(titles), cols=1, shared_xaxes=True, row_heights=heights,
        vertical_spacing=min(.025, .16 / len(titles)), subplot_titles=titles,
    )
    add_timeline_behavior(figure, behavior, tracks, show_context)
    # Behavior already has outcome/context hover; only paired traces need legends.
    for trace in figure.data:
        trace.showlegend = False
    row = 2 + len(tracks)
    if result.fit is not None:
        limit = float(result.fit.color_limit if color_limit is None else color_limit)
        if not np.isfinite(limit) or limit <= 0:
            raise ValueError("Color limit must be a positive finite canonical score.")
        for region in ("PFC", "HPC"):
            data = build_cca_heatmap_data(result, region, pair_range, loaded_range_s)
            figure.add_trace(go.Heatmap(
                **data, name=region, coloraxis="coloraxis", zsmooth=False,
                connectgaps=False, hoverongaps=False,
                hovertemplate=_score_hover(result, region, "%{z:.5g}", "%{y}", "%{customdata[7]:.4f}"),
            ), row=row, col=1)
            figure.update_yaxes(
                autorange="reversed", tickmode="array", tickvals=ranks,
                ticktext=[f"CC{rank}" for rank in ranks], row=row, col=1,
            )
            row += 1
        figure.update_layout(coloraxis={
            "cmin": -limit, "cmax": limit, "cauto": False,
            "colorscale": [[0., "#2166ac"], [.5, "#d1d5db"], [1., "#b2182b"]],
            "showscale": True, "colorbar": {"title": "Canonical score<br>(session SD)", "thickness": 14},
        })
        if show_traces:
            _add_canonical_traces(figure, result, selected_pair, loaded_range_s, row)
            row += 1
    figure.add_trace(go.Scatter(
        x=result.choices.elapsed_s, y=result.choices.action, mode="markers",
        marker={"size": 3, "color": "#555555"}, hoverinfo="skip",
        showlegend=False, name="Session overview",
    ), row=row, col=1)
    figure.update_yaxes(visible=False, row=row, col=1)
    figure.update_xaxes(range=viewport_s or loaded_range_s or full_bounds, showgrid=False)
    allowed = loaded_range_s or full_bounds
    figure.update_xaxes(minallowed=allowed[0], maxallowed=allowed[1])
    figure.update_xaxes(
        title_text="Elapsed session time (s; zero = first trial start)", row=row, col=1,
        rangeslider={"visible": True, "range": full_bounds,
                     "thickness": min(.07, 60 / figure_height)},
    )
    vertical_revision = f"{result.meta.get('configuration_id', result.session_id)}:{tracks}:{tuple(ranks)}:{show_traces}:{selected_pair}"
    figure.update_yaxes(fixedrange=True, showgrid=False, zeroline=False, uirevision=vertical_revision)
    caption = ("In-sample canonical correlations; task activity retained.<br>"
               "Scores use session sample SD; pooled bins are temporally dependent.")
    if result.fit is None:
        caption = f"Paired CCA unavailable: {result.unavailable_reason or 'not prepared'}. Behavior remains available."
    elif color_limit is not None and color_limit < result.fit.color_limit:
        caption += f"<br>Manual color limit +/-{color_limit:g}; larger magnitudes saturate."
    figure.add_annotation(
        text=caption, x=0, y=0, yshift=-145, xref="paper", yref="paper", xanchor="left",
        yanchor="top", showarrow=False, align="left", font={"size": 13},
    )
    figure.update_layout(
        height=figure_height,
        margin={"l": 85, "r": 145, "t": 60, "b": 210}, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white", font={"size": 14, "color": "black"},
        dragmode="pan", showlegend=bool(result.fit is not None and show_traces),
        legend={"orientation": "h", "y": 1.07},
        uirevision=f"{result.session_id}:{result.time_origin_s}:{viewport_revision}",
    )
    return figure


# Display helpers: no file reads, numerical fits or saved-array mutations.
def _selected_ranks(result: SessionCCAResult, pair_range: tuple[int, int] | None) -> np.ndarray:
    """Return (P,) integer one-based ranks from an inclusive range and saved K pairs."""
    count = len(result.fit.correlations)
    first, last = pair_range or (1, min(10, count))
    if (not isinstance(first, (int, np.integer)) or not isinstance(last, (int, np.integer))
            or first < 1 or last < first or last > count):
        raise ValueError("Canonical pair range must use available one-based ranks.")
    return np.arange(first, last + 1)


def _time_range(bounds: tuple[float, float] | None) -> tuple[float, float] | None:
    """Validate an optional two-float increasing elapsed-second range; return it or None."""
    if bounds is None:
        return None
    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError("Time range must contain two increasing finite elapsed seconds.")
    return float(bounds[0]), float(bounds[1])


def _display_windows(result: SessionCCAResult, bounds: tuple[float, float] | None) -> list:
    """Return (index, choice Series, (B+1,) elapsed edges) for intersecting complete windows.

    result retains chronological common choice_positions (W,); bounds is an
    optional elapsed-second range. Selection uses half-open intersection and
    never clips bins or changes saved scores. Return a list in saved order.
    """
    bounds = _time_range(bounds)
    windows = []
    for window_index, position in enumerate(result.choice_positions):
        choice = result.choices.iloc[int(position)]
        edges = float(choice.elapsed_s) + result.bin_edges_s
        if bounds is None or (edges[-1] > bounds[0] and edges[0] < bounds[1]):
            windows.append((window_index, choice, edges))
    return windows


def _score_hover(result: SessionCCAResult, region: str, score: str, rank: str, rho: str) -> str:
    """Return Plotly hover markup for scores with eight-field bin metadata.

    result supplies probe roles; region is PFC/HPC. score/rank/rho are Plotly
    template strings or formatted constants. Return text with original trial,
    UTC/elapsed/choice-relative seconds and dimensionless in-sample rho.
    """
    return (
        f"{region} ({result.probe_roles[region]}) CC{rank}; in-sample rho {rho}"
        "<br>Trial %{customdata[0]}<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
        "<br>UTC [%{customdata[1]:.6f}, %{customdata[2]:.6f}) s"
        "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
        f"<br>Canonical score {score} (session SD)<extra></extra>"
    )


def _add_canonical_traces(
    figure: go.Figure, result: SessionCCAResult, pair: int,
    loaded_range_s: tuple[float, float] | None, row: int,
) -> None:
    """Add two raw-score bin steps to a preallocated row, with full-pair y limits.

    figure is modified in place at one-based row. result scores (W,B,K) stay
    unchanged; pair is one-based, loaded_range_s is elapsed seconds. Points
    cover actual B+1 edges, repeat only the final bin value, and break at
    unmeasured gaps. Hover follows the heatmap's eight-field metadata. Return
    None; score units are canonical session SD, with no viewport rescaling.
    """
    for region, color in (("PFC", "#2563eb"), ("HPC", "#c2410c")):
        x, y, custom = [], [], []
        data = build_cca_heatmap_data(result, region, (pair, pair), loaded_range_s)
        column = 0
        for index, choice, edges in _display_windows(result, loaded_range_s):
            if x and edges[0] > x[-1]:
                x.append(None)
                y.append(None)
                custom.append([None] * 8)
                column += 1
            scores = result.fit.scores[region][index, :, pair - 1]
            x.extend(edges.tolist())
            y.extend(np.r_[scores, scores[-1]].tolist())
            info = data["customdata"][0, column:column + len(scores)]
            custom.extend(np.concatenate([info, info[-1:]], axis=0).tolist())
            column += len(scores)
        figure.add_trace(go.Scatter(
            x=x, y=y, customdata=custom, mode="lines", name=f"{region} CC{pair}",
            line={"shape": "hv", "width": 1.7, "color": color}, connectgaps=False,
            hovertemplate=_score_hover(result, region, "%{y:.5g}", str(pair),
                                       f"{result.fit.correlations[pair - 1]:.4f}"),
            showlegend=True,
        ), row=row, col=1)
    low = min(float(result.fit.scores[region][:, :, pair - 1].min()) for region in ("PFC", "HPC"))
    high = max(float(result.fit.scores[region][:, :, pair - 1].max()) for region in ("PFC", "HPC"))
    margin = .05 * (high - low) if high > low else 1.
    figure.update_yaxes(range=(low - margin, high + margin), autorange=False,
                        title_text="Canonical SD", row=row, col=1)
