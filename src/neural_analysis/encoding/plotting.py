"""Display-only coefficient ordering and coupled fixed-PC activity inspection."""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.encoding import results
from src.neural_analysis.encoding.predictors import encode_context
from src.neural_analysis.population import timeline_plotting as timeline_plotting


SIGNED_COLORS = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]


def order_responses(
    cell: dict,
    *,
    coefficient_scale: str = "raw",
    ordering: str = "original",
    sorting_predictor: str | None = None,
) -> np.ndarray:
    """Select a display order without changing original fitted response identities.

    Input cell retains pc_ranks(K,), slopes(P,K) and column_names(P,).
    coefficient_scale is raw or standardized; ordering is original or
    coefficient_strength. sorting_predictor names one native model column.
    Return integer response-column indices(K,), strongest absolute coefficients
    first with original-PC-rank ties and unavailable values after valid zeros.
    Raw coefficients retain PC/native predictor units; standardized are unitless.
    """
    ranks = np.asarray(cell["pc_ranks"])
    if ordering == "original":
        return np.argsort(ranks, kind="stable")
    if ordering != "coefficient_strength":
        raise ValueError("ordering must be original or coefficient_strength")
    slopes = _display_slopes(cell, coefficient_scale)
    names = list(cell["column_names"])
    name = sorting_predictor or names[0]
    if name not in names:
        raise ValueError("Sorting predictor must be an available model column")
    strength = np.abs(slopes[names.index(name)])
    missing = ~np.isfinite(strength)
    return np.lexsort((ranks, -np.where(missing, 0, strength), missing))


def plot_encoding_timeline(
    preparation: dict,
    cell: dict,
    *,
    coefficient_scale: str = "raw",
    ordering: str = "original",
    sorting_predictor: str | None = None,
    row_start: int = 0,
    row_count: int = 10,
    selected_pc: int | None = None,
    prediction_scope: str = "held_out",
    time_range_s: tuple[float, float] | None = None,
    loaded_range_s: tuple[float, float] | None = None,
    event_tracks: tuple[str, ...] = (),
    show_context: bool = True,
    viewport_revision: int = 0,
    inspection_penalty_mode: str | None = None,
) -> go.Figure:
    """Build one figure with behavioral, coefficient, activity and prediction panels.

    preparation is a saved dict with behavior and regional scores(W,B,K_full).
    cell is a raw saved cell with row indices(N,), original pc_ranks(K_fit),
    coefficients(P,K_fit) and reference/held-out predictions(N,K_fit). Observed
    values are read from the shared basis; no observed field is required.
    Controls change display only. Row start is zero-based; row_count defaults10.
    Time/loaded ranges are elapsed seconds, defaulting to the first60 seconds.
    selected_pc is an original one-based fitted rank. prediction_scope selects
    held_out or reference explicitly; unavailable predictions never fall back.
    inspection_penalty_mode labels a verified alias without mutating saved meta.

    Return a Plotly Figure. Coefficient/activity panels use identical consecutive
    display-row coordinates with original PC labels. Activity remains raw signed
    PC scores, preserving fixed limits, actual bin edges and blank gaps. Layout
    metadata declares only elapsed-time axes for browser range propagation.
    """
    results.cell_observations(preparation, cell)
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    behavior = preparation["behavior"]
    display = order_responses(cell, coefficient_scale=coefficient_scale,
        ordering=ordering, sorting_predictor=sorting_predictor)
    if type(row_start) is not int or type(row_count) is not int or row_start < 0 or row_count < 1:
        raise ValueError("Display row start/count must be nonnegative/positive integers")
    display = display[row_start:row_start + row_count]
    if not len(display):
        raise ValueError("Requested display rows are outside the fitted PC range")
    ranks = cell["pc_ranks"][display]
    rank = int(ranks[0] if selected_pc is None else selected_pc)
    _response_column(cell, rank)
    visible = _visible_range(behavior, time_range_s)
    loaded = _visible_range(behavior, loaded_range_s) if loaded_range_s is not None else visible
    tracks = tuple(track for track in event_tracks if track in timeline_plotting.available_event_tracks(behavior.events))
    predictor_row, activity_row, detail_row = len(tracks) + 2, len(tracks) + 3, len(tracks) + 4
    specs = [[{"colspan": 2}, None] for _ in range(detail_row)]
    specs[activity_row - 1] = [{}, {}]
    titles = ["Choices and logged context"] + [timeline_plotting.EVENT_LABELS[track] for track in tracks]
    titles += ["Native model predictors and selected fit bins", "Coefficients", "Fixed regional PC activity",
               f"Selected response: PC{rank}"]
    figure = make_subplots(rows=detail_row, cols=2, specs=specs, column_widths=[0.2, 0.8],
        row_heights=[0.8] + [0.45] * len(tracks) + [0.95, max(2.0, len(display) * 0.18), 1.25],
        horizontal_spacing=0.10, vertical_spacing=0.06, subplot_titles=titles)
    timeline_plotting.add_timeline_behavior(figure, behavior, tracks, show_context)
    _add_predictors(figure, preparation, cell, predictor_row, loaded)
    slopes = _display_slopes(cell, coefficient_scale)
    coefficient_limit = _symmetric_limit(slopes)
    coefficient_custom = np.empty((len(display), slopes.shape[0], 2), dtype=object)
    coefficient_custom[:, :, 0] = np.asarray([f"PC{value}" for value in ranks])[:, None]
    coefficient_custom[:, :, 1] = cell["column_names"][None, :]
    figure.add_trace(go.Heatmap(x=cell["column_names"], y=np.arange(len(display)), z=slopes[:, display].T,
        customdata=coefficient_custom, name="Coefficients", colorscale=SIGNED_COLORS,
        zmin=-coefficient_limit, zmax=coefficient_limit, zauto=False, zsmooth=False,
        hoverongaps=False, colorbar={"title": f"{coefficient_scale} slope", "x": -0.06, "len": 0.25},
        hovertemplate="%{customdata[0]} / %{customdata[1]}<br>Coefficient %{z:.6g}<extra></extra>"),
        row=activity_row, col=1)
    data = timeline_plotting.build_timeline_heatmap_data(region.timeline, behavior.choices,
        loaded_range_s=loaded, pc_order=tuple(int(value) for value in ranks))
    fit_data = _bin_data(preparation, cell, _fit_mask(region, cell), loaded)
    custom = np.empty((*data["customdata"].shape[:2], 11), dtype=object)
    custom[:, :, :8] = data["customdata"]
    custom[:, :, 8] = np.asarray([f"PC{value}" for value in ranks])[:, None]
    custom[:, :, 9] = np.asarray(fit_data["z"])[0][None, :]
    predictor_values = _predictor_values(region, cell)
    predictor_series = [_bin_data(preparation, cell, predictor_values[:, :, column:column + 1], loaded)["z"][0]
                        for column in range(predictor_values.shape[2])]
    assigned = ["; ".join(f"{name}={value:g}" if np.isfinite(value) else f"{name}=unavailable"
                          for name, value in zip(cell["column_names"], values))
                for values in zip(*predictor_series)]
    custom[:, :, 10] = np.asarray(assigned, dtype=object)[None, :]
    figure.add_trace(go.Heatmap(x=data["x"], y=np.arange(len(display)), z=data["z"], customdata=custom,
        name="PC activity", colorscale=SIGNED_COLORS, zmin=-region.timeline.color_limit,
        zmax=region.timeline.color_limit, zauto=False, zsmooth=False, connectgaps=False, hoverongaps=False,
        colorbar={"title": "PC score", "len": 0.30},
        hovertemplate="%{customdata[8]} - %{customdata[7]:.2f}% session variance"
        "<br>Trial %{customdata[0]}<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
        "<br>UTC [%{customdata[1]:.6f}, %{customdata[2]:.6f}) s"
        "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
        "<br>Assigned predictors: %{customdata[10]}"
        "<br>Selected for fit: %{customdata[9]}<br>PC score %{z:.6g}<extra></extra>"),
        row=activity_row, col=2)
    coefficient_axis = figure.get_subplot(activity_row, 1).xaxis.plotly_name
    coefficient_y = figure.get_subplot(activity_row, 1).yaxis.plotly_name
    for column in (1, 2):
        figure.update_yaxes(tickmode="array", tickvals=np.arange(len(display)),
            ticktext=[f"PC{value}" for value in ranks], range=[len(display) - 0.5, -0.5],
            row=activity_row, col=column)
    figure.update_yaxes(matches=coefficient_y.replace("axis", ""), row=activity_row, col=2)
    figure.layout[coefficient_axis].update(type="category", fixedrange=True)
    _add_prediction_traces(figure, preparation, cell, rank, prediction_scope, loaded, detail_row)
    time_axes = sorted((name for name in figure.to_plotly_json()["layout"]
                        if name.startswith("xaxis") and name != coefficient_axis), key=_axis_number)
    primary = figure.get_subplot(1, 1).xaxis.plotly_name
    activity_domain = figure.get_subplot(activity_row, 2).xaxis.domain
    for name in time_axes:
        figure.layout[name].update(range=visible, type="linear", showgrid=False,
            domain=activity_domain,
            matches=None if name == primary else primary.replace("axis", ""),
            minallowed=loaded[0], maxallowed=loaded[1])
    # Equal elapsed ranges also need equal pixel domains to align all time tracks.
    for annotation in figure.layout.annotations:
        if annotation.xref == "paper" and annotation.x == 0.5:
            annotation.x = sum(activity_domain) / 2
    figure.update_xaxes(title_text="Elapsed session time (s; zero = first trial start)", row=detail_row, col=1)
    figure.update_yaxes(fixedrange=True, showgrid=False, zeroline=False)
    figure.update_layout(title={"text": _configuration_label(cell, inspection_penalty_mode), "font": {"size": 16}},
        height=max(780, 680 + len(display) * 14 + len(tracks) * 55),
        margin={"l": 100, "r": 110, "t": 100, "b": 80}, template="plotly_white",
        paper_bgcolor="white", plot_bgcolor="white", font={"size": 14, "color": "black"},
        dragmode="pan", showlegend=True, legend={"orientation": "h", "y": -0.08},
        uirevision=f"{cell['meta']['preparation_id']}:{region.timeline.region}:{viewport_revision}",
        meta={"primary_time_axis": primary, "time_axes": time_axes, "coefficient_axis": coefficient_axis,
              "display_pc_ranks": ranks.tolist(), "selected_pc_rank": rank,
              "coefficient_scale": coefficient_scale, "prediction_scope": prediction_scope,
              "inspection_penalty_mode": inspection_penalty_mode or cell["meta"]["settings"]["penalty_mode"],
              "loaded_range_s": list(loaded)})
    return figure


def plot_selected_prediction(
    preparation: dict, cell: dict, *, pc_rank: int, prediction_scope: str = "held_out",
    time_range_s: tuple[float, float] | None = None,
) -> go.Figure:
    """Input shared scores(W,B,K)/cell predictions(N,K), original PC rank and elapsed range; return trace figure.

    Observed and selected prediction retain original PC-score units and gaps.
    held_out/reference is explicit; this standalone convenience uses the same
    trace builder embedded in plot_encoding_timeline, without any refitting.
    """
    visible = _visible_range(preparation["behavior"], time_range_s)
    figure = make_subplots(rows=1, cols=1)
    _add_prediction_traces(figure, preparation, cell, pc_rank, prediction_scope, visible, 1)
    figure.update_layout(template="plotly_white", font={"size": 14}, height=320,
        meta={"primary_time_axis": "xaxis", "time_axes": ["xaxis"],
              "selected_pc_rank": pc_rank, "prediction_scope": prediction_scope})
    figure.update_xaxes(range=visible, title_text="Elapsed session time (s)")
    return figure


def save_inspection_figures(
    output_directory: Path | str,
    preparation: dict,
    cell: dict,
    *,
    time_range_s: tuple[float, float] | None = None,
    selected_pc: int | None = None,
    coefficient_scale: str = "raw",
    inspection_penalty_mode: str | None = None,
    prediction_scope: str = "held_out",
) -> tuple[Path, Path]:
    """Export light-mode coefficient and bounded activity/prediction PNGs without refitting.

    Inputs share the interactive figure's saved arrays/units. output_directory
    is a session run's figures directory; time range is elapsed seconds and
    selected_pc is one original rank; prediction_scope explicitly selects
    held_out or reference without substitution. Return two new cell/scale/PC/scope-identified
    PNG Paths. Captions identify fit settings, scope and fixed PCA. Existing
    filenames raise FileExistsError. No full-session neural raster is exported.
    """
    import matplotlib.pyplot as plt

    order = order_responses(cell, coefficient_scale=coefficient_scale,
        ordering="coefficient_strength", sorting_predictor=str(cell["column_names"][0]))[:10]
    rank = int(cell["pc_ranks"][order[0]] if selected_pc is None else selected_pc)
    _response_column(cell, rank)
    visible = _visible_range(preparation["behavior"], time_range_s)
    observed, predicted = _prediction_data(preparation, cell, rank, prediction_scope, visible)
    scope_label = "held-out predictions" if prediction_scope == "held_out" else "reference-fit predictions"
    scope_status = "available" if np.isfinite(predicted["z"]).any() else "unavailable"
    directory = Path(output_directory)
    stem = f"{cell['meta']['cell_id']}_{coefficient_scale}_pc{rank}_{prediction_scope}"
    paths = directory / f"{stem}_coefficients.png", directory / f"{stem}_timeline_prediction.png"
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite existing inspection PNGs")
    directory.mkdir(parents=True, exist_ok=True)
    caption = (_configuration_label(cell, inspection_penalty_mode) +
        f"; {coefficient_scale} coefficients; selected PC{rank}; {scope_label} {scope_status}. "
        f"Elapsed [{visible[0]:g}, {visible[1]:g}] s. "
        "Regression uses fixed descriptive session PCA. Full choice windows are shown; "
        "only selected fit bins contribute. Missing predictions and inter-window gaps remain blank.")
    if not cell["meta"]["cv_available"]:
        caption += f" Complete CV unavailable: {cell['meta'].get('cv_reason', 'see saved fit details')}."
    slopes = _display_slopes(cell, coefficient_scale)[:, order].T
    with plt.rc_context({"font.size": 14, "axes.labelsize": 14, "figure.facecolor": "white",
                         "axes.facecolor": "white", "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black"}):
        figure, axis = plt.subplots(figsize=(10, 7))
        limit = _symmetric_limit(_display_slopes(cell, coefficient_scale))
        image = axis.imshow(slopes, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_xticks(np.arange(slopes.shape[1]), cell["column_names"])
        axis.set_yticks(np.arange(len(order)), [f"PC{value}" for value in cell["pc_ranks"][order]])
        axis.set_title(f"{cell['meta']['settings']['region']} context coefficients ({coefficient_scale})")
        figure.colorbar(image, ax=axis, label=f"{coefficient_scale} slope")
        for row in range(slopes.shape[0]):
            for column in range(slopes.shape[1]):
                value = slopes[row, column]
                axis.text(column, row, f"{value:.3g}" if np.isfinite(value) else "unavailable",
                          ha="center", va="center", fontsize=11,
                          color="white" if abs(value) > limit * 0.6 else "black")
        _save_png(figure, paths[0], caption)
        plt.close(figure)
        region = preparation["regions"][cell["meta"]["settings"]["region"]]
        data = timeline_plotting.build_timeline_heatmap_data(region.timeline, preparation["behavior"].choices,
            loaded_range_s=visible, pc_order=tuple(int(value) for value in cell["pc_ranks"][order]))
        figure, grid = plt.subplots(3, 2, figsize=(13, 10), sharex="col",
            gridspec_kw={"height_ratios": [1, 2.8, 1.5], "width_ratios": [1, 0.025]})
        axes = grid[:, 0]
        grid[0, 1].set_axis_off()
        grid[2, 1].set_axis_off()
        context = _predictor_values(region, cell)
        for column, name in enumerate(cell["column_names"]):
            values = _bin_data(preparation, cell, context[:, :, column:column + 1], visible)
            x, y = _steps(values)
            axes[0].plot(x, y, label=str(name))
        mask = _bin_data(preparation, cell, _fit_mask(region, cell), visible)
        x, y = _steps(mask)
        axes[0].plot(x, y * 0.25 - 1.5, linestyle=":", color="black", label="Selected fit bins")
        axes[0].set_ylabel("Native predictors")
        axes[0].legend(fontsize=10, loc="upper right")
        if len(data["x"]):
            heatmap = axes[1].pcolormesh(data["x"], np.arange(len(order) + 1) - 0.5,
                np.ma.masked_invalid(data["z"]), cmap="RdBu_r", vmin=-region.timeline.color_limit,
                vmax=region.timeline.color_limit, shading="flat")
            figure.colorbar(heatmap, cax=grid[1, 1], label="PC score")
        else:
            grid[1, 1].set_axis_off()
        axes[1].set_yticks(np.arange(len(order)), [f"PC{value}" for value in cell["pc_ranks"][order]])
        axes[1].set_ylim(len(order) - 0.5, -0.5)
        axes[2].plot(*_steps(observed), label=f"Observed PC{rank}", color="#2563eb")
        prediction_label = "Held-out prediction" if prediction_scope == "held_out" else "Reference prediction"
        axes[2].plot(*_steps(predicted), label=prediction_label, color="#b91c1c", linestyle="--")
        axes[2].set_ylabel("PC score")
        axes[2].set_xlabel("Elapsed session time (s)")
        axes[2].legend(fontsize=11)
        axes[2].set_xlim(*visible)
        axes[0].set_title(textwrap.fill(_configuration_label(cell, inspection_penalty_mode), width=90),
                          fontsize=13, pad=14)
        _save_png(figure, paths[1], caption)
        plt.close(figure)
    return paths


# Plotting-only array/geometry helpers.
def _display_slopes(cell: dict, scale: str) -> np.ndarray:
    """Input saved cell/scale string; return slopes(P,K), PC/native or dimensionless units, unchanged."""
    if scale not in ("raw", "standardized"):
        raise ValueError("coefficient_scale must be raw or standardized")
    return cell["reference_slopes" if scale == "raw" else "standardized_slopes"]


def _symmetric_limit(values: np.ndarray) -> float:
    """Input coefficient array of any shape; return positive finite absolute scale in same units."""
    finite = np.abs(values[np.isfinite(values)])
    return max(float(finite.max(initial=0)), np.finfo(float).eps) if finite.size else 1.0


def _visible_range(behavior: object, requested: tuple[float, float] | None) -> tuple[float, float]:
    """Input saved behavior/optional elapsed-second pair; return finite ordered bounds, default first60s."""
    low, high = timeline_plotting.session_time_bounds(behavior)
    bounds = (low, min(high, low + 60.0)) if requested is None else tuple(requested)
    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("Time range must contain finite increasing elapsed seconds")
    return bounds


def _axis_number(name: str) -> int:
    """Input Plotly xaxis layout-name string; return dimensionless numeric subplot index."""
    return int(name[5:] or "1")


def _configuration_label(cell: dict, inspection_penalty_mode: str | None) -> str:
    """Input saved cell/optional verified alias; return exact fit label, native alpha/bin units retained."""
    meta, settings = cell["meta"], cell["meta"]["settings"]
    mode = inspection_penalty_mode or settings["penalty_mode"]
    if inspection_penalty_mode is not None and inspection_penalty_mode not in meta.get("penalty_aliases", []):
        raise ValueError("Requested inspection penalty mode is not an alias of this saved fit")
    label = " | ".join(str(settings[key]) for key in ("region", "window", "timing_mode", "encoding", "estimator"))
    if mode is not None:
        label += f" | {mode}; alpha={meta['effective_alpha']:g} (configured {meta['configured_alpha']:g})"
    return label + f" | {meta['bin_width_ms']} ms"


def _fit_mask(region: object, cell: dict) -> np.ndarray:
    """Input region scores(W,B,K)/cell row indices(N,); return float fit mask(W,B,1), values0/1."""
    mask = np.zeros(region.timeline.scores.shape[:2], dtype=float)
    mask.reshape(-1)[cell["selected_row_indices"]] = 1.0
    return mask[:, :, None]


def _predictor_values(region: object, cell: dict) -> np.ndarray:
    """Return native predictors(W,B,P), using saved X(N,P) on selected fit rows.

    Input region supplies full saved context(W,B); the cell supplies its exact
    selected-row design(N,P). Context-derived values fill only outside-fit
    inspection bins. Missing context stays NaN; inputs are never mutated.
    """
    mode = cell["meta"]["settings"]["timing_mode"]
    states = region.context_values[mode]
    encoded, valid, _ = encode_context(states.reshape(-1), cell["meta"]["settings"]["encoding"])
    encoded[~(valid & region.context_valid[mode].reshape(-1))] = np.nan
    saved = np.asarray(cell["predictors"], dtype=float)
    indices = cell["selected_row_indices"]
    if saved.shape != (len(indices), encoded.shape[1]) or not np.isfinite(saved).all():
        raise ValueError("Saved predictor design must match selected rows/native column shape")
    encoded[indices] = saved
    return encoded.reshape(*states.shape, encoded.shape[1])


def _bin_data(preparation: dict, cell: dict, values: np.ndarray, loaded: tuple[float, float]) -> dict:
    """Map one display-only series(W,B,1) to existing explicit bin geometry; retain values' native units."""
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    return timeline_plotting.build_timeline_heatmap_data(region.timeline, preparation["behavior"].choices,
        loaded_range_s=loaded, display_scores=values, pc_order=(int(region.pc_ranks[0]),))


def _add_predictors(figure: go.Figure, preparation: dict, cell: dict, row: int, loaded: tuple[float, float]) -> None:
    """Input allocated figure/sharedprep/cell/row/elapsed range; add native predictor and fit-mask heatmaps, return None."""
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    values = _predictor_values(region, cell)
    series = [_bin_data(preparation, cell, values[:, :, column:column + 1], loaded) for column in range(values.shape[2])]
    custom = np.empty((len(series), len(series[0]["x"]) - 1 if len(series[0]["x"]) else 0, 9), dtype=object)
    custom[:, :, :8] = np.concatenate([data["customdata"] for data in series], axis=0)
    custom[:, :, 8] = cell["column_names"][:, None]
    figure.add_trace(go.Heatmap(x=series[0]["x"], y=np.arange(len(series)),
        z=np.concatenate([data["z"] for data in series]), name="Model predictors", colorscale=SIGNED_COLORS,
        customdata=custom, zmin=-1, zmax=1, showscale=False, zsmooth=False, hoverongaps=False,
        hovertemplate="%{customdata[8]} = %{z}<br>Trial %{customdata[0]}"
        "<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
        "<br>UTC [%{customdata[1]:.6f}, %{customdata[2]:.6f}) s"
        "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s<extra></extra>"), row=row, col=1)
    fit = _bin_data(preparation, cell, _fit_mask(region, cell), loaded)
    figure.add_trace(go.Heatmap(x=fit["x"], y=[len(series)], z=fit["z"], name="Fit-bin inclusion",
        colorscale=[[0, "white"], [1, "#444444"]], zmin=0, zmax=1, showscale=False,
        zsmooth=False, hoverongaps=False, hovertemplate="Selected fit bin: %{z}<extra></extra>"), row=row, col=1)
    figure.update_yaxes(tickmode="array", tickvals=np.arange(len(series) + 1),
        ticktext=[*cell["column_names"], "Selected fit bins"], range=[len(series) + 0.5, -0.5], row=row, col=1)


def _response_column(cell: dict, rank: int) -> int:
    """Input saved cell/original one-based PC rank; return matching response-column index, no unit changes."""
    found = np.flatnonzero(cell["pc_ranks"] == rank)
    if len(found) != 1:
        raise ValueError("Selected PC rank must be one original fitted response")
    return int(found[0])


def _prediction_data(preparation: dict, cell: dict, rank: int, scope: str, loaded: tuple[float, float]) -> tuple[dict, dict]:
    """Input shared scores/cell/rank/scope/elapsed bounds; return observed/predicted bin geometry in PC-score units."""
    results.cell_observations(preparation, cell)
    if scope not in ("held_out", "reference"):
        raise ValueError("prediction_scope must be held_out or reference")
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    column = _response_column(cell, rank)
    observed = timeline_plotting.build_timeline_heatmap_data(region.timeline, preparation["behavior"].choices,
        loaded_range_s=loaded, pc_order=(rank,))
    full = np.full(region.timeline.scores.shape[:2], np.nan)
    predictions = cell["held_out_predictions" if scope == "held_out" else "reference_predictions"]
    full.reshape(-1)[cell["selected_row_indices"]] = predictions[:, column]
    predicted = _bin_data(preparation, cell, full[:, :, None], loaded)
    return observed, predicted


def _steps(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Input bin edges(M+1,)/one-series z(1,M); return explicit flat-segment vertices(2M,), same units."""
    return np.column_stack((data["x"][:-1], data["x"][1:])).reshape(-1), np.repeat(data["z"][0], 2)


def _add_prediction_traces(figure: go.Figure, preparation: dict, cell: dict, rank: int,
                           scope: str, loaded: tuple[float, float], row: int) -> None:
    """Input figure/sharedprep/cell/rank/scope/elapsed range/row; add observed and explicit-scope PC traces, return None."""
    observed, predicted = _prediction_data(preparation, cell, rank, scope, loaded)
    label = "Held-out prediction" if scope == "held_out" else "Reference prediction"
    for name, data, color, dash in ((f"Observed PC{rank}", observed, "#2563eb", "solid"),
                                     (f"{label} PC{rank}", predicted, "#b91c1c", "dash")):
        x, y = _steps(data)
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, connectgaps=False,
            line={"color": color, "dash": dash}, hovertemplate="Elapsed %{x:.3f} s<br>PC score %{y:.5g}<extra></extra>"),
            row=row, col=1)
    figure.update_yaxes(title_text=f"PC{rank} score", row=row, col=1)
    if not np.isfinite(predicted["z"]).any():
        figure.add_annotation(text=f"{label} unavailable for PC{rank} in this time range", showarrow=False,
                              x=0.5, y=0.0, xref="paper", yref="paper")


def _save_png(figure: object, path: Path, caption: str) -> None:
    """Input Matplotlib figure/new PNG Path/caption string; save opaque light PNG with visible/embedded caption, return None."""
    figure.text(0.02, 0.02, textwrap.fill(caption, width=int(figure.get_figwidth() * 10)),
                ha="left", va="bottom", fontsize=10)
    figure.tight_layout(rect=(0, 0.14, 1, 0.97))
    figure.savefig(path, dpi=130, facecolor="white", transparent=False, metadata={"Description": caption})
