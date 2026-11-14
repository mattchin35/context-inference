"""Saved-only joint-model inspection in native unit-Hz or fixed PC-score units."""

from __future__ import annotations

from pathlib import Path
import re
import textwrap

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.neural_analysis.encoding.predictors import build_joint_predictors
from src.neural_analysis.population import timeline_plotting

SIGNED_COLORS = [[0.0, "#2166ac"], [0.5, "#f7f7f7"], [1.0, "#b2182b"]]
CONTRAST_NAMES = ("left_reward_minus_omission", "right_reward_minus_omission")


def contrast_values(cell: dict) -> np.ndarray:
    """Return supported raw contrasts(2,R), masking unavailable rows with NaN.

    cell contains contrasts(2,R) in Hz or PC-score units and contrast_valid(2,).
    Return a float copy; original values remain saved even when unsupported.
    """
    values = np.asarray(cell["contrasts"], dtype=float).copy()
    values[~cell["contrast_valid"]] = np.nan
    return values


def order_responses(
    cell: dict,
    *,
    coefficient_scale: str = "raw",
    ordering: str = "original",
    sorting_predictor: str | None = None,
) -> np.ndarray:
    """Return response-column indices(R,) for display without changing response IDs.

    cell supplies response_ids(R,), slopes(P,R), predictor/contrast support.
    Original means saved unit order or original PC rank. Strength sorts absolute
    native/standardized coefficients or raw contrasts, with missing values last
    and saved order breaking ties. Contrasts always retain native response units.
    """
    original = np.arange(len(cell["response_ids"]))
    if ordering == "original":
        return original
    if ordering != "coefficient_strength":
        raise ValueError("ordering must be original or coefficient_strength")
    name = sorting_predictor or str(cell["column_names"][0])
    if name in CONTRAST_NAMES:
        strength = np.abs(contrast_values(cell)[CONTRAST_NAMES.index(name)])
    elif name in cell["column_names"]:
        strength = np.abs(
            _display_slopes(cell, coefficient_scale)[
                list(cell["column_names"]).index(name)
            ]
        )
    else:
        raise ValueError(
            "Sorting predictor must name a model column or raw reward contrast"
        )
    missing = ~np.isfinite(strength)
    return np.lexsort((original, -np.where(missing, 0, strength), missing))


def plot_encoding_timeline(
    preparation: dict,
    cell: dict,
    *,
    coefficient_scale: str = "raw",
    ordering: str = "original",
    sorting_predictor: str | None = None,
    row_start: int = 0,
    row_count: int = 10,
    selected_response: str | None = None,
    prediction_scope: str = "held_out",
    time_range_s: tuple[float, float] | None = None,
    loaded_range_s: tuple[float, float] | None = None,
    event_tracks: tuple[str, ...] = (),
    show_context: bool = True,
    viewport_revision: int = 0,
    inspection_penalty_mode: str | None = None,
) -> go.Figure:
    """Build coupled behavioral/predictor/coefficient/activity/prediction panels.

    preparation owns counts(W,B,U), optional fixed PCA(W,B,K), trial identities
    and second-valued edges. cell owns native X(N,P), coefficients(P,R), exact
    response_ids(R,) and predictions(N,R). selected_response is probe:cluster
    or PC rank text. Row controls are zero-based offsets/counts; time ranges
    are elapsed seconds. Controls consume saved arrays only. Return a Plotly
    Figure with identical response rows and time-axis pixel domains. Activity
    stays Hz or PC scores; coefficient scaling never rescales neural activity.
    """
    region = _region_for_cell(preparation, cell)
    behavior = preparation["behavior"]
    display = order_responses(
        cell,
        coefficient_scale=coefficient_scale,
        ordering=ordering,
        sorting_predictor=sorting_predictor,
    )
    if (
        type(row_start) is not int
        or type(row_count) is not int
        or row_start < 0
        or row_count < 1
    ):
        raise ValueError(
            "Display row start/count must be nonnegative/positive integers"
        )
    display = display[row_start : row_start + row_count]
    if not len(display):
        raise ValueError("Requested display rows are outside the fitted response range")
    ids = cell["response_ids"][display]
    selected = str(ids[0] if selected_response is None else selected_response)
    _response_column(cell, selected)
    visible = _visible_range(behavior, time_range_s)
    loaded = (
        _visible_range(behavior, loaded_range_s)
        if loaded_range_s is not None
        else visible
    )
    tracks = tuple(
        track
        for track in event_tracks
        if track in timeline_plotting.available_event_tracks(behavior.events)
    )
    predictor_row, activity_row, detail_row = (
        len(tracks) + 2,
        len(tracks) + 3,
        len(tracks) + 4,
    )
    specs = [[{"colspan": 2}, None] for _ in range(detail_row)]
    specs[activity_row - 1] = [{}, {}]
    units = cell["meta"]["response_units"]
    titles = ["Choices and logged context"] + [
        timeline_plotting.EVENT_LABELS[track] for track in tracks
    ]
    titles += [
        "Native joint predictors and fit bins",
        "Coefficients",
        f"Neural activity ({units})",
        f"Selected response: {selected}",
    ]
    figure = make_subplots(
        rows=detail_row,
        cols=2,
        specs=specs,
        column_widths=[0.28, 0.72],
        row_heights=[0.8]
        + [0.45] * len(tracks)
        + [2.3, max(2.2, len(display) * 0.2), 1.4],
        horizontal_spacing=0.10,
        vertical_spacing=0.055,
        subplot_titles=titles,
    )
    timeline_plotting.add_timeline_behavior(figure, behavior, tracks, show_context)
    _add_predictors(figure, preparation, cell, predictor_row, loaded)
    slopes = _display_slopes(cell, coefficient_scale)
    limit = _symmetric_limit(slopes)
    custom = np.empty((len(display), len(cell["column_names"]), 2), dtype=object)
    custom[:, :, 0], custom[:, :, 1] = ids[:, None], cell["column_names"][None, :]
    panel = figure.get_subplot(activity_row, 1)
    figure.add_trace(
        go.Heatmap(
            x=cell["column_names"],
            y=np.arange(len(display)),
            z=slopes[:, display].T,
            customdata=custom,
            name="Coefficients",
            colorscale=SIGNED_COLORS,
            zmin=-limit,
            zmax=limit,
            zauto=False,
            zsmooth=False,
            hoverongaps=False,
            colorbar={
                "title": {"text": f"{coefficient_scale} slope", "side": "top"},
                "orientation": "h",
                "x": sum(panel.xaxis.domain) / 2,
                "xanchor": "center",
                "y": panel.yaxis.domain[0] - 0.14,
                "yanchor": "top",
                "thickness": 10,
                "len": panel.xaxis.domain[1] - panel.xaxis.domain[0],
            },
            hovertemplate="%{customdata[0]} / %{customdata[1]}<br>Coefficient %{z:.6g}<extra></extra>",
        ),
        row=activity_row,
        col=1,
    )
    data = _bin_data(preparation, cell, _response_values(region, cell, display), loaded)
    custom = np.empty((*data["customdata"].shape[:2], 9), dtype=object)
    custom[:, :, :7], custom[:, :, 7] = data["customdata"], ids[:, None]
    custom[:, :, 8] = _bin_data(preparation, cell, _fit_mask(region, cell), loaded)[
        "z"
    ][0][None, :]
    variance = ""
    if cell["meta"]["settings"]["representation"] == "pcs":
        expanded = np.empty((*custom.shape[:2], 10), dtype=object)
        expanded[:, :, :9] = custom
        expanded[:, :, 9] = (
            region.pca.transform.explained_variance_ratio[
                region.pc_ranks[display] - 1, None
            ]
            * 100
        )
        custom = expanded
        variance = " - %{customdata[9]:.2f}% session variance"
    low, high, colors = _activity_scale(region, cell)
    figure.add_trace(
        go.Heatmap(
            x=data["x"],
            y=np.arange(len(display)),
            z=data["z"],
            customdata=custom,
            name="Neural activity",
            colorscale=colors,
            zmin=low,
            zmax=high,
            zauto=False,
            zsmooth=False,
            connectgaps=False,
            hoverongaps=False,
            colorbar={
                "title": {"text": units, "side": "right"},
                "y": sum(panel.yaxis.domain) / 2,
                "len": panel.yaxis.domain[1] - panel.yaxis.domain[0],
                "thickness": 16,
            },
            hovertemplate="%{customdata[7]}" + variance + "<br>Trial %{customdata[0]}"
            "<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
            "<br>UTC [%{customdata[1]:.6f}, %{customdata[2]:.6f}) s"
            "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s"
            "<br>Selected for fit: %{customdata[8]}<br>Activity %{z:.6g} "
            + units
            + "<extra></extra>",
        ),
        row=activity_row,
        col=2,
    )
    coefficient_axis = panel.xaxis.plotly_name
    for column in (1, 2):
        figure.update_yaxes(
            tickmode="array",
            tickvals=np.arange(len(display)),
            ticktext=ids.tolist(),
            range=[len(display) - 0.5, -0.5],
            row=activity_row,
            col=column,
        )
    figure.update_yaxes(
        matches=panel.yaxis.plotly_name.replace("axis", ""), row=activity_row, col=2
    )
    figure.layout[coefficient_axis].update(
        type="category", fixedrange=True, tickangle=-55, tickfont={"size": 11}
    )
    _add_prediction_traces(
        figure, preparation, cell, selected, prediction_scope, loaded, detail_row
    )
    time_axes = sorted(
        (
            name
            for name in figure.to_plotly_json()["layout"]
            if name.startswith("xaxis") and name != coefficient_axis
        ),
        key=_axis_number,
    )
    primary = figure.get_subplot(1, 1).xaxis.plotly_name
    activity_domain = figure.get_subplot(activity_row, 2).xaxis.domain
    for name in time_axes:
        figure.layout[name].update(
            range=visible,
            type="linear",
            showgrid=False,
            domain=activity_domain,
            matches=None if name == primary else primary.replace("axis", ""),
            minallowed=loaded[0],
            maxallowed=loaded[1],
        )
    for annotation in figure.layout.annotations:
        if annotation.xref == "paper" and annotation.x == 0.5:
            annotation.x = sum(activity_domain) / 2
    figure.update_xaxes(
        title_text="Elapsed session time (s; zero = first trial start)",
        row=detail_row,
        col=1,
    )
    figure.update_yaxes(fixedrange=True, showgrid=False, zeroline=False)
    legend_names = set()
    for item in figure.data:
        if item.showlegend is not False and item.name:
            if item.name in legend_names:
                item.showlegend = False
            legend_names.add(item.name)
    representation = cell["meta"]["settings"]["representation"]
    figure.update_layout(
        title={
            "text": _configuration_label(cell, inspection_penalty_mode),
            "font": {"size": 16},
        },
        height=max(1050, 950 + len(display) * 16 + len(tracks) * 55),
        margin={"l": 130, "r": 100, "t": 110, "b": 90},
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"size": 14, "color": "black"},
        dragmode="pan",
        legend={"orientation": "h", "y": -0.08},
        uirevision=f"{cell['meta']['preparation_id']}:{region.region}:{viewport_revision}",
        meta={
            "primary_time_axis": primary,
            "time_axes": time_axes,
            "coefficient_axis": coefficient_axis,
            "display_response_ids": ids.tolist(),
            "selected_response": selected,
            "representation": representation,
            "coefficient_scale": coefficient_scale,
            "prediction_scope": prediction_scope,
            "inspection_penalty_mode": inspection_penalty_mode
            or cell["meta"]["settings"]["penalty_mode"],
            "loaded_range_s": list(loaded),
        },
    )
    return figure


def save_inspection_figures(
    output_directory: Path | str,
    preparation: dict,
    cell: dict,
    *,
    time_range_s: tuple[float, float] | None = None,
    selected_response: str | None = None,
    coefficient_scale: str = "raw",
    inspection_penalty_mode: str | None = None,
    prediction_scope: str = "held_out",
) -> tuple[Path, Path]:
    """Export two opaque light PNGs from saved native activity and coefficients.

    Inputs follow plot_encoding_timeline: counts(W,B,U), optional scores(W,B,K),
    response_ids(R,), coefficients(P,R), elapsed-second bounds and explicit
    prediction scope. Return two new Paths (coefficient summary and coupled
    coefficients/activity/prediction). Existing filenames raise FileExistsError.
    No fitting or raw input loading occurs. Captions state native units/scope.
    """
    import matplotlib.pyplot as plt

    region = _region_for_cell(preparation, cell)
    order = order_responses(
        cell, coefficient_scale=coefficient_scale, ordering="coefficient_strength"
    )[:10]
    selected = str(
        cell["response_ids"][order[0]]
        if selected_response is None
        else selected_response
    )
    _response_column(cell, selected)
    visible = _visible_range(preparation["behavior"], time_range_s)
    observed, predicted = _prediction_data(
        preparation, cell, selected, prediction_scope, visible
    )
    directory = Path(output_directory)
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", selected)
    stem = f"{cell['meta']['cell_id']}_{coefficient_scale}_{safe_id}_{prediction_scope}"
    paths = (
        directory / f"{stem}_coefficients.png",
        directory / f"{stem}_timeline_prediction.png",
    )
    if any(path.exists() for path in paths):
        raise FileExistsError("Refusing to overwrite existing inspection PNGs")
    directory.mkdir(parents=True, exist_ok=True)
    units = cell["meta"]["response_units"]
    scope_label = (
        "held-out predictions"
        if prediction_scope == "held_out"
        else "reference-fit predictions"
    )
    status = "available" if np.isfinite(predicted["z"]).any() else "unavailable"
    caption = (
        _configuration_label(cell, inspection_penalty_mode)
        + f"; {coefficient_scale} coefficients; selected {selected}; response units {units}; {scope_label} {status}. "
        f"Elapsed [{visible[0]:g}, {visible[1]:g}] s. Choice-aligned outcomes use [0,+1); neural pool [-1,+1). "
        "Reward-minus-omission contrasts describe conditional associations; a positive outcome slope alone is not reward selectivity. "
        "Missing predictions and inter-window gaps remain blank."
    )
    if cell["meta"]["settings"]["representation"] == "pcs":
        caption += " Fixed descriptive session PCA includes test-block neural activity."
    if not cell["meta"]["cv_available"]:
        caption += f" Complete CV unavailable: {cell['meta'].get('cv_reason', 'see saved fit details')}."
    slopes = _display_slopes(cell, coefficient_scale)[:, order].T
    ids = cell["response_ids"][order]
    limit = _symmetric_limit(_display_slopes(cell, coefficient_scale))
    data = _bin_data(preparation, cell, _response_values(region, cell, order), visible)
    low, high, _ = _activity_scale(region, cell)
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
        }
    ):
        figure, axis = plt.subplots(figsize=(12, 8))
        image = axis.imshow(
            slopes, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
        )
        axis.set_xticks(
            np.arange(slopes.shape[1]), cell["column_names"], rotation=45, ha="right"
        )
        axis.set_yticks(np.arange(len(ids)), ids)
        axis.set_title(f"{region.region} joint coefficients ({coefficient_scale})")
        figure.colorbar(image, ax=axis, label=f"{coefficient_scale} slope")
        _save_png(figure, paths[0], caption)
        plt.close(figure)
        # Dedicated colorbar columns preserve identical coefficient/activity rows
        # and identical elapsed-time pixel domains for the activity and trace.
        figure = plt.figure(figsize=(16, 11))
        grid = figure.add_gridspec(
            2, 4, width_ratios=[3, 0.12, 7, 0.15], height_ratios=[3, 1.5], wspace=0.65
        )
        coefficients, activity = (
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 2]),
        )
        detail = figure.add_subplot(grid[1, 2], sharex=activity)
        image = coefficients.pcolormesh(
            np.arange(slopes.shape[1] + 1),
            np.arange(len(ids) + 1),
            np.ma.masked_invalid(slopes),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        figure.colorbar(
            image,
            cax=figure.add_subplot(grid[0, 1]),
            label=f"{coefficient_scale} slope",
        )
        coefficients.set_xticks(
            np.arange(slopes.shape[1]) + 0.5,
            cell["column_names"],
            rotation=65,
            ha="right",
            fontsize=10,
        )
        coefficients.set_yticks(np.arange(len(ids)) + 0.5, ids)
        coefficients.set_ylim(len(ids), 0)
        coefficients.set_title("Joint coefficients")
        if len(data["x"]):
            image = activity.pcolormesh(
                data["x"],
                np.arange(len(ids) + 1),
                np.ma.masked_invalid(data["z"]),
                cmap="viridis" if units == "Hz" else "RdBu_r",
                vmin=low,
                vmax=high,
            )
            figure.colorbar(image, cax=figure.add_subplot(grid[0, 3]), label=units)
        activity.set_yticks(np.arange(len(ids)) + 0.5, ids)
        activity.set_ylim(len(ids), 0)
        activity.set_title(f"Neural activity ({units})")
        detail.plot(*_steps(observed), label=f"Observed {selected}", color="#2563eb")
        detail.plot(
            *_steps(predicted), label=scope_label, color="#b91c1c", linestyle="--"
        )
        detail.set_ylabel(units)
        detail.set_xlabel("Elapsed session time (s)")
        detail.legend(fontsize=11)
        detail.set_xlim(*visible)
        _save_png(figure, paths[1], caption, use_tight_layout=False)
        plt.close(figure)
    return paths


# Saved-array and geometry helpers. These never load live data or estimate a model.
def _region_for_cell(preparation: dict, cell: dict) -> object:
    """Validate saved preparation/response IDs and row indices; return the regional activity record.

    Counts are (W,B,U), optional PCs are (W,B,K), indices are (N,). Validation
    avoids allocating an unused full observation matrix on every display edit.
    """
    if cell["meta"]["preparation_id"] != preparation["meta"]["preparation_id"]:
        raise ValueError("Cell preparation identity differs from shared activity")
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    representation = cell["meta"]["settings"]["representation"]
    ids = (
        region.unit_ids
        if representation == "units"
        else tuple(f"PC{rank}" for rank in region.pc_ranks)
    )
    indices = np.asarray(cell["selected_row_indices"])
    if not np.array_equal(cell["response_ids"], ids):
        raise ValueError("Cell response identity differs from preparation")
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= np.prod(region.unit_counts.shape[:2]))
    ):
        raise ValueError("Cell observation row identity differs from preparation")
    return region


def _response_values(region: object, cell: dict, columns: np.ndarray) -> np.ndarray:
    """Return selected saved activity(W,B,R_selected) in Hz or PC-score units; never change source arrays."""
    if cell["meta"]["settings"]["representation"] == "units":
        return region.unit_counts[:, :, columns] / np.diff(region.bin_edges_s)[0]
    return region.pca.scores[:, :, region.pc_ranks[columns] - 1]


def _activity_scale(region: object, cell: dict) -> tuple[float, float, object]:
    """Input saved activity/cell; return fixed full-regional native lower/upper limits and colorscale."""
    if cell["meta"]["settings"]["representation"] == "units":
        high = float(region.unit_counts.max(initial=0)) / np.diff(region.bin_edges_s)[0]
        return 0.0, max(1.0, high), "Viridis"
    return -region.pca.color_limit, region.pca.color_limit, SIGNED_COLORS


def _display_slopes(cell: dict, scale: str) -> np.ndarray:
    """Return supported slopes(P,R), raw in native response units or standardized dimensionless; no mutation."""
    if scale not in ("raw", "standardized"):
        raise ValueError("coefficient_scale must be raw or standardized")
    slopes = np.array(
        cell["reference_slopes" if scale == "raw" else "standardized_slopes"], copy=True
    )
    slopes[~cell["predictor_supported"]] = np.nan
    return slopes


def _symmetric_limit(values: np.ndarray) -> float:
    """Input coefficient array in native units; return a safe positive absolute color limit in those units."""
    finite = np.abs(values[np.isfinite(values)])
    return (
        max(float(finite.max(initial=0)), np.finfo(float).eps) if finite.size else 1.0
    )


def _visible_range(
    behavior: object, requested: tuple[float, float] | None
) -> tuple[float, float]:
    """Input saved behavior/optional elapsed-second bounds; return finite increasing bounds, default first 60 seconds."""
    low, high = timeline_plotting.session_time_bounds(behavior)
    bounds = (low, min(high, low + 60.0)) if requested is None else tuple(requested)
    if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[1] <= bounds[0]:
        raise ValueError("Time range must contain finite increasing elapsed seconds")
    return bounds


def _axis_number(name: str) -> int:
    """Input Plotly xaxis layout-name string; return its dimensionless subplot index."""
    return int(name[5:] or "1")


def _configuration_label(cell: dict, inspection_penalty_mode: str | None) -> str:
    """Input saved cell/optional verified penalty alias; return model label with alpha and bin/window units."""
    meta, settings = cell["meta"], cell["meta"]["settings"]
    mode = inspection_penalty_mode or settings["penalty_mode"]
    if inspection_penalty_mode is not None and inspection_penalty_mode not in meta.get(
        "penalty_aliases", []
    ):
        raise ValueError(
            "Requested inspection penalty mode is not an alias of this saved fit"
        )
    label = " | ".join(
        str(settings[key])
        for key in ("region", "representation", "encoding", "estimator")
    )
    if mode is not None:
        label += f" | {mode}; alpha={meta['configured_alpha']:g}"
    return label + f" | {meta['bin_width_ms']} ms | [-1,+1)"


def _fit_mask(region: object, cell: dict) -> np.ndarray:
    """Input counts(W,B,U)/selected row indices(N,); return float included-bin mask(W,B,1), values 0/1."""
    mask = np.zeros(region.unit_counts.shape[:2], dtype=float)
    mask.reshape(-1)[cell["selected_row_indices"]] = 1.0
    return mask[:, :, None]


def _predictor_values(preparation: dict, cell: dict) -> np.ndarray:
    """Return native predictors(W,B,P) from saved trials and exact fitted X on eligible rows.

    Original row_position maps each window to the complete source trial table;
    unknown outcomes stay NaN for the whole window. No trial is reclassified
    from activity, and the saved design overrides all selected fit rows.
    """
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    bins = region.unit_counts.shape[1]
    design = build_joint_predictors(
        preparation["trials"],
        region.bin_rows.iloc[::bins].row_position.to_numpy(),
        region.bin_edges_s,
        cell["meta"]["settings"]["encoding"],
    )
    values = design["predictors"].copy()
    values[cell["selected_row_indices"]] = cell["predictors"]
    return values.reshape(*region.unit_counts.shape[:2], -1)


def _bin_data(
    preparation: dict, cell: dict, values: np.ndarray, loaded: tuple[float, float]
) -> dict:
    """Input native display values(W,B,R)/elapsed bounds; return actual-time edges and blank-gap geometry."""
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    return timeline_plotting.build_window_heatmap_data(
        values,
        preparation["behavior"].choices,
        region.choice_positions,
        region.bin_edges_s,
        loaded,
    )


def _add_predictors(
    figure: go.Figure,
    preparation: dict,
    cell: dict,
    row: int,
    loaded: tuple[float, float],
) -> None:
    """Input allocated Plotly figure/saved arrays/row/elapsed range; add joint X and fit-mask heatmaps, return None."""
    region = preparation["regions"][cell["meta"]["settings"]["region"]]
    values = _predictor_values(preparation, cell)
    data = _bin_data(preparation, cell, values, loaded)
    count = values.shape[2]
    custom = np.empty((*data["customdata"].shape[:2], 8), dtype=object)
    custom[:, :, :7], custom[:, :, 7] = (
        data["customdata"],
        cell["column_names"][:, None],
    )
    figure.add_trace(
        go.Heatmap(
            x=data["x"],
            y=np.arange(count),
            z=data["z"],
            name="Model predictors",
            colorscale=SIGNED_COLORS,
            customdata=custom,
            zmin=-1,
            zmax=1,
            showscale=False,
            zsmooth=False,
            hoverongaps=False,
            hovertemplate="%{customdata[7]} = %{z}<br>Trial %{customdata[0]}"
            "<br>Elapsed [%{customdata[5]:.3f}, %{customdata[6]:.3f}) s"
            "<br>Choice-relative [%{customdata[3]:.3f}, %{customdata[4]:.3f}) s<extra></extra>",
        ),
        row=row,
        col=1,
    )
    fit = _bin_data(preparation, cell, _fit_mask(region, cell), loaded)
    figure.add_trace(
        go.Heatmap(
            x=fit["x"],
            y=[count],
            z=fit["z"],
            name="Fit-bin inclusion",
            colorscale=[[0, "white"], [1, "#444444"]],
            zmin=0,
            zmax=1,
            showscale=False,
            zsmooth=False,
            hoverongaps=False,
            hovertemplate="Selected fit bin: %{z}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    figure.update_yaxes(
        tickmode="array",
        tickvals=np.arange(count + 1),
        ticktext=[*cell["column_names"], "Selected fit bins"],
        range=[count + 0.5, -0.5],
        row=row,
        col=1,
    )


def _response_column(cell: dict, response_id: str) -> int:
    """Input saved response_ids(R,) and original ID string; return the unique response column index."""
    found = np.flatnonzero(cell["response_ids"] == response_id)
    if len(found) != 1:
        raise ValueError("Selected response must be one original fitted response ID")
    return int(found[0])


def _prediction_data(
    preparation: dict,
    cell: dict,
    response_id: str,
    scope: str,
    loaded: tuple[float, float],
) -> tuple[dict, dict]:
    """Input saved arrays/ID/scope/elapsed bounds; return observed/predicted bin geometry in native response units."""
    region = _region_for_cell(preparation, cell)
    if scope not in ("held_out", "reference"):
        raise ValueError("prediction_scope must be held_out or reference")
    column = _response_column(cell, response_id)
    observed = _bin_data(
        preparation, cell, _response_values(region, cell, np.array([column])), loaded
    )
    full = np.full((*region.unit_counts.shape[:2], 1), np.nan)
    predictions = cell[
        "held_out_predictions" if scope == "held_out" else "reference_predictions"
    ]
    full.reshape(-1)[cell["selected_row_indices"]] = predictions[:, column]
    return observed, _bin_data(preparation, cell, full, loaded)


def _steps(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Input edges(M+1,)/one-series z(1,M); return flat-segment vertices(2M,) retaining seconds/native units."""
    return np.column_stack((data["x"][:-1], data["x"][1:])).reshape(-1), np.repeat(
        data["z"][0], 2
    )


def _add_prediction_traces(
    figure: go.Figure,
    preparation: dict,
    cell: dict,
    response_id: str,
    scope: str,
    loaded: tuple[float, float],
    row: int,
) -> None:
    """Input Plotly figure/saved arrays/ID/scope/elapsed bounds/row; add observed and explicit-scope traces, return None."""
    observed, predicted = _prediction_data(
        preparation, cell, response_id, scope, loaded
    )
    label = "Held-out prediction" if scope == "held_out" else "Reference prediction"
    units = cell["meta"]["response_units"]
    for name, data, color, dash in (
        (f"Observed {response_id}", observed, "#2563eb", "solid"),
        (f"{label} {response_id}", predicted, "#b91c1c", "dash"),
    ):
        x, y = _steps(data)
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                connectgaps=False,
                line={"color": color, "dash": dash},
                hovertemplate="Elapsed %{x:.3f} s<br>%{y:.5g} "
                + units
                + "<extra></extra>",
            ),
            row=row,
            col=1,
        )
    figure.update_yaxes(title_text=units, row=row, col=1)
    if not np.isfinite(predicted["z"]).any():
        figure.add_annotation(
            text=f"{label} unavailable for {response_id} in this time range",
            showarrow=False,
            x=0.5,
            y=0.0,
            xref="paper",
            yref="paper",
        )


def _save_png(
    figure: object, path: Path, caption: str, *, use_tight_layout: bool = True
) -> None:
    """Input Matplotlib figure/new PNG Path/caption; save opaque light PNG with embedded and visible caption, return None."""
    figure.text(
        0.02,
        0.02,
        textwrap.fill(caption, width=int(figure.get_figwidth() * 11)),
        ha="left",
        va="bottom",
        fontsize=10,
    )
    if use_tight_layout:
        figure.tight_layout(rect=(0, 0.2, 1, 0.97))
    else:
        figure.subplots_adjust(left=0.08, right=0.95, bottom=0.26, top=0.94, hspace=0.8)
    figure.savefig(
        path,
        dpi=130,
        facecolor="white",
        transparent=False,
        metadata={"Description": caption},
    )
