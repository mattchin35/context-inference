"""Light-mode task-decoding figures derived only from validated saved results."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile
import textwrap

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from src.neural_analysis.task_decoding.results import (
    CONDITION_RESULT_SCHEMA_VERSION,
    condition_labels,
    load_task_decoding_run,
    select_condition_arrays,
)


NONZERO_COEFFICIENT_TOLERANCE = 1e-8

_FAMILY_METRICS = {
    "categorical": ("balanced_accuracy", "auc"),
    "numerical": ("r2",),
}
_METRIC_TITLES = {
    "balanced_accuracy": "Categorical balanced accuracy",
    "auc": "Categorical ROC AUC",
    "r2": "Numerical R²",
}
_METRIC_REFERENCES = {"balanced_accuracy": 0.5, "auc": 0.5, "r2": 0.0}
_REPRESENTATION_TITLES = {"pca": "PCA", "units": "direct units"}


def _saved_arrays(saved_run: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    """Return the primitive array mapping from one loader-shaped saved run.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Mapping returned by ``load_task_decoding_run``.

    Returns
    -------
    mapping[str, numpy.ndarray]
        Saved primitive arrays with their declared axes and units unchanged.
    """
    arrays = saved_run.get("arrays")
    if not isinstance(arrays, Mapping):
        raise ValueError("Saved run does not contain an array mapping.")
    return arrays


def _label_index(values: np.ndarray, label: str, axis_name: str) -> int:
    """Return one exact Unicode-label position or raise a readable error.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional saved Unicode label axis.
    label : str
        Exact requested label.
    axis_name : str
        Human-readable axis name used in validation errors.

    Returns
    -------
    int
        Zero-based position of ``label``.
    """
    matches = np.flatnonzero(np.asarray(values) == label)
    if matches.size != 1:
        raise ValueError(f"Unknown {axis_name}: {label}")
    return int(matches[0])


def _mean_available(values: np.ndarray) -> np.ndarray:
    """Average the final fold axis while retaining all-unavailable cells as NaN.

    Parameters
    ----------
    values : numpy.ndarray
        Floating scores with outer fold on the final axis.

    Returns
    -------
    numpy.ndarray
        Float64 mean scores on the leading axes; no runtime warning is emitted
        for cells whose folds are all unavailable.
    """
    numeric = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(numeric)
    counts = finite.sum(axis=-1)
    total = np.where(finite, numeric, 0.0).sum(axis=-1)
    means = np.full(total.shape, np.nan, dtype=np.float64)
    np.divide(total, counts, out=means, where=counts > 0)
    return means


def _target_display_label(identifier: str) -> str:
    """Convert one stable target identifier to a concise plot label.

    Parameters
    ----------
    identifier : str
        Saved snake-case target identifier.

    Returns
    -------
    str
        Human-readable label without changing the saved identifier itself.
    """
    replacements = {
        "hmm": "HMM",
        "qlearning": "Q-learning",
    }
    words = [replacements.get(word, word) for word in identifier.split("_")]
    label = " ".join(words)
    return label[0].upper() + label[1:] if label else label


def _condition_display_label(identifier: str) -> str:
    """Convert one stable condition identifier to a concise display label.

    Parameters
    ----------
    identifier : str
        Saved snake-case condition identifier.

    Returns
    -------
    str
        Human-readable title preserving the underlying saved identifier.
    """
    label = identifier.replace("_", " ")
    return label[0].upper() + label[1:] if label else label


def _heatmap_normalization(values: np.ndarray, reference: float) -> TwoSlopeNorm:
    """Build a reference-centered scale that never clips finite saved scores.

    Parameters
    ----------
    values : numpy.ndarray
        Fold-averaged dimensionless metric values; NaN marks unavailability.
    reference : float
        Descriptive baseline, 0.5 for categorical metrics or 0 for R².

    Returns
    -------
    matplotlib.colors.TwoSlopeNorm
        Diverging normalization centered exactly at ``reference``.
    """
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    lower = float(np.min(finite)) if finite.size else reference
    upper = float(np.max(finite)) if finite.size else reference
    return TwoSlopeNorm(
        vmin=min(reference - 0.5, lower),
        vcenter=reference,
        vmax=max(reference + 0.5, upper),
    )


def _heatmap_caption(
    arrays: Mapping[str, np.ndarray],
    target_indices: np.ndarray,
    metric: str,
    condition: str,
) -> str:
    """Compose a concise scientific caption from saved counts and fold coverage.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Valid saved result arrays.
    target_indices : numpy.ndarray
        One-dimensional positions of targets shown in the heatmaps.
    metric : {"balanced_accuracy", "auc", "r2"}
        Saved held-out metric displayed.
    condition : str
        Exact saved condition whose scores and eligibility counts are shown.

    Returns
    -------
    str
        Figure caption containing interpretation, trial counts, and fold count.
    """
    labels = arrays["target_labels"]
    counts = arrays["eligibility_counts"]
    trial_summary = ", ".join(
        f"{labels[index]}: {int(counts[index])} eligible trials"
        for index in target_indices
    )
    fold_count = int(arrays["fold_labels"].size)
    if metric in {"balanced_accuracy", "auc"}:
        reference_text = "0.5 is a descriptive reference, not a significance threshold."
    else:
        reference_text = "0 is a descriptive reference; negative R² values remain visible."
    return (
        f"Condition: {condition}. Held-out outer-fold "
        f"{_METRIC_TITLES[metric].lower()}; {reference_text} "
        f"Gray cells are unavailable. {trial_summary}; {fold_count} outer folds."
    )


def plot_decoding_heatmap(
    saved_run: Mapping[str, object],
    *,
    family: str,
    metric: str,
    condition: str = "all",
    _legacy_panel_titles: bool = False,
) -> plt.Figure:
    """Plot one saved metric across all region/representation combinations.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Valid loader-shaped run. Score axes are target, region, representation,
        metric, time bin, and outer fold; scores are dimensionless.
    family : {"categorical", "numerical"}
        Target family to facet on the heatmap row axis.
    metric : {"balanced_accuracy", "auc", "r2"}
        Held-out metric. Categorical metrics use a 0.5 reference; R² uses 0.
    condition : str, default="all"
        Exact saved condition to display. Historical schema-1 runs expose only
        ``all``; schema-2 runs select one leading condition-axis cell.
    _legacy_panel_titles : bool, default=False
        Internal compatibility presentation for the pre-WP7 pipeline test seam.
        User-facing figures always use the clearer WP7 titles.

    Returns
    -------
    matplotlib.figure.Figure
        Opaque-white figure with three region rows by two representation
        columns. No model fitting or source-data loading occurs.
    """
    if family not in _FAMILY_METRICS or metric not in _FAMILY_METRICS[family]:
        raise ValueError("Metric is not valid for the requested target family.")
    arrays = select_condition_arrays(saved_run, condition)
    target_indices = np.flatnonzero(arrays["target_families"] == family)
    if target_indices.size == 0:
        raise ValueError(f"Saved run has no {family} targets.")
    metric_index = _label_index(arrays["metric_labels"], metric, "metric")
    scores = _mean_available(arrays["fold_scores"][target_indices, :, :, metric_index])
    reference = _METRIC_REFERENCES[metric]
    normalization = _heatmap_normalization(scores, reference)
    color_map = plt.get_cmap("RdBu_r").copy()
    color_map.set_bad("#bdbdbd")
    figure, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), squeeze=False)
    figure.patch.set_facecolor("white")
    target_labels = [
        _target_display_label(str(arrays["target_labels"][index]))
        for index in target_indices
    ]
    time_centers = np.asarray(arrays["time_bin_centers_s"], dtype=float)
    tick_positions = np.unique(
        np.linspace(0, time_centers.size - 1, min(5, time_centers.size), dtype=int)
    )
    image = None
    for region_index, region in enumerate(arrays["region_labels"].tolist()):
        for representation_index, representation in enumerate(
            arrays["representation_labels"].tolist()
        ):
            axis = axes[region_index, representation_index]
            axis.set_facecolor("white")
            panel = np.ma.masked_invalid(scores[:, region_index, representation_index, :])
            image = axis.imshow(
                panel,
                aspect="auto",
                interpolation="nearest",
                cmap=color_map,
                norm=normalization,
            )
            if _legacy_panel_titles:
                axis.set_title(f"{region} {representation}")
            else:
                region_title = str(region).replace("+", " + ")
                representation_title = _REPRESENTATION_TITLES.get(
                    str(representation), str(representation)
                )
                axis.set_title(f"{region_title} - {representation_title}")
            alignment = str(saved_run["scientific_config"]["alignment"])
            axis.set_xlabel(f"Time from {alignment} (s)")
            axis.set_xticks(tick_positions)
            axis.set_xticklabels([f"{time_centers[index]:g}" for index in tick_positions])
            axis.set_yticks(np.arange(target_indices.size))
            axis.set_yticklabels(target_labels if representation_index == 0 else [])
            axis.tick_params(colors="black", labelsize=9)
            for spine in axis.spines.values():
                spine.set_color("black")
    assert image is not None
    figure.suptitle(
        f"{_METRIC_TITLES[metric]} - {_condition_display_label(condition)}",
        color="black",
        fontsize=15,
    )
    caption = _heatmap_caption(arrays, target_indices, metric, condition)
    figure.text(
        0.02,
        0.015,
        textwrap.fill(caption, width=130),
        color="black",
        fontsize=9,
    )
    figure.subplots_adjust(left=0.15, right=0.82, top=0.91, bottom=0.15, hspace=0.42)
    colorbar_axis = figure.add_axes((0.86, 0.18, 0.025, 0.66))
    figure.colorbar(image, cax=colorbar_axis, label=_METRIC_TITLES[metric])
    return figure


def _atomic_save_png(figure: plt.Figure, destination: Path) -> None:
    """Publish one opaque PNG without exposing partial bytes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Fully rendered light-mode figure.
    destination : pathlib.Path
        Final PNG path. Existing immutable files are left unchanged.

    Returns
    -------
    None
        Writes a sibling temporary PNG, fsyncs it, and atomically replaces the
        absent destination. Temporary files are removed on failure.
    """
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        figure.savefig(
            temporary,
            format="png",
            dpi=150,
            facecolor="white",
            transparent=False,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def save_default_decoding_figures(
    run_directory: Path | str,
    *,
    saved_run: Mapping[str, object] | None = None,
    _legacy_panel_titles: bool = False,
) -> tuple[Path, ...]:
    """Save the bounded default PNG set for target families present in one run.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Completed or currently reporting task-decoding run directory.
    saved_run : mapping[str, object] or None, default=None
        Optional already validated loader-shaped mapping. If omitted,
        ``results.npz`` and its small sidecars are loaded from ``run_directory``.
    _legacy_panel_titles : bool, default=False
        Internal compatibility presentation for the pre-WP7 private test seam.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Required categorical balanced-accuracy/AUC and/or numerical R² paths,
        in deterministic order. Existing immutable PNGs are not overwritten.
    """
    directory = Path(run_directory)
    loaded = load_task_decoding_run(directory) if saved_run is None else saved_run
    arrays = _saved_arrays(loaded)
    families = set(arrays["target_families"].tolist())
    metric_specifications: list[tuple[str, str, str]] = []
    if "categorical" in families:
        metric_specifications.extend(
            [
                ("categorical", "balanced_accuracy", "categorical_balanced_accuracy.png"),
                ("categorical", "auc", "categorical_auc.png"),
            ]
        )
    if "numerical" in families:
        metric_specifications.append(("numerical", "r2", "numerical_r2.png"))
    schema_version = loaded.get("meta", {}).get("schema_version")
    specifications = [
        (
            condition,
            family,
            metric,
            (
                f"{condition}--{filename}"
                if schema_version == CONDITION_RESULT_SCHEMA_VERSION
                else filename
            ),
        )
        for condition in condition_labels(loaded)
        for family, metric, filename in metric_specifications
    ]
    paths = []
    for condition, family, metric, filename in specifications:
        destination = directory / "figures" / filename
        figure = plot_decoding_heatmap(
            loaded,
            family=family,
            metric=metric,
            condition=condition,
            _legacy_panel_titles=_legacy_panel_titles,
        )
        try:
            _atomic_save_png(figure, destination)
        finally:
            plt.close(figure)
        paths.append(destination)
    return tuple(paths)


def required_default_figure_filenames(
    saved_run: Mapping[str, object],
) -> tuple[str, ...]:
    """Return the exact deterministic completion-gate PNG names for one run.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Valid loader-shaped schema-1 or schema-2 run.

    Returns
    -------
    tuple[str, ...]
        Present-family filenames in condition-major then metric order. Pooled
        schema-1 names remain unchanged; schema-2 names use
        ``condition--metric.png`` qualification.
    """
    arrays = _saved_arrays(saved_run)
    families = set(arrays["target_families"].tolist())
    base_names: list[str] = []
    if "categorical" in families:
        base_names.extend(
            ["categorical_balanced_accuracy.png", "categorical_auc.png"]
        )
    if "numerical" in families:
        base_names.append("numerical_r2.png")
    if saved_run.get("meta", {}).get("schema_version") != CONDITION_RESULT_SCHEMA_VERSION:
        return tuple(base_names)
    return tuple(
        f"{condition}--{filename}"
        for condition in condition_labels(saved_run)
        for filename in base_names
    )


def summarize_unit_coefficients(
    saved_run: Mapping[str, object],
    *,
    target: str,
    region: str,
    time_bin_index: int,
    condition: str = "all",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize selected direct-unit coefficients across saved outer folds.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Valid loader-shaped saved run. Coefficients use axes target, region,
        representation, time bin, outer fold, and stable feature identity.
    target : str
        Exact saved target identifier.
    region : {"PFC", "HPC", "PFC+HPC"}
        Exact saved regional configuration.
    time_bin_index : int
        Zero-based saved time-bin position.
    condition : str, default="all"
        Exact saved condition whose direct-unit folds are summarized.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Feature rows sorted by mean absolute coefficient and outer-fold rows
        with fit status, reason, counts, and nonzero coefficient fractions.
        Coefficients retain their saved model scale and unit labels.
    """
    arrays = select_condition_arrays(saved_run, condition)
    target_index = _label_index(arrays["target_labels"], target, "target")
    region_index = _label_index(arrays["region_labels"], region, "region")
    representation_index = _label_index(
        arrays["representation_labels"], "units", "representation"
    )
    time_count = int(arrays["time_bin_centers_s"].size)
    if isinstance(time_bin_index, bool) or not 0 <= int(time_bin_index) < time_count:
        raise ValueError("time_bin_index is outside the saved time axis.")
    time_index = int(time_bin_index)
    active = arrays["coefficient_active_masks"][region_index, representation_index]
    feature_ids = arrays["coefficient_feature_ids"][
        region_index, representation_index, active
    ]
    feature_regions = arrays["coefficient_feature_regions"][
        region_index, representation_index, active
    ]
    all_coefficients = arrays["coefficient_values"][
        target_index, region_index, representation_index, time_index
    ]
    coefficients = np.asarray(all_coefficients[:, active], dtype=float)
    if coefficients.ndim == 1:
        coefficients = coefficients.reshape(1, -1)
    statuses = arrays["fit_status"][
        target_index, region_index, representation_index, time_index
    ]
    reasons = arrays["fit_reason_codes"][
        target_index, region_index, representation_index, time_index
    ]
    feature_rows = []
    for feature_index, (feature_id, feature_region) in enumerate(
        zip(feature_ids.tolist(), feature_regions.tolist())
    ):
        values = coefficients[:, feature_index]
        finite = np.isfinite(values)
        valid_fit = statuses == "valid"
        unavailable_fit = statuses == "unavailable"
        selected = finite & (np.abs(values) > NONZERO_COEFFICIENT_TOLERANCE)
        contributing_count = int(finite.sum())
        finite_values = values[finite]
        feature_rows.append(
            {
                "feature_id": str(feature_id),
                "region": str(feature_region),
                "contributing_fold_count": contributing_count,
                "excluded_fold_count": int((valid_fit & ~finite).sum()),
                "unavailable_fold_count": int(unavailable_fit.sum()),
                "median_coefficient": (
                    float(np.median(finite_values)) if contributing_count else np.nan
                ),
                "iqr_low": (
                    float(np.percentile(finite_values, 25)) if contributing_count else np.nan
                ),
                "iqr_high": (
                    float(np.percentile(finite_values, 75)) if contributing_count else np.nan
                ),
                "mean_absolute_coefficient": (
                    float(np.mean(np.abs(finite_values))) if contributing_count else np.nan
                ),
                "selection_count": int(selected.sum()),
                "selection_frequency": (
                    float(selected.sum() / contributing_count)
                    if contributing_count
                    else np.nan
                ),
            }
        )
    feature_table = pd.DataFrame(
        feature_rows,
        columns=[
            "feature_id",
            "region",
            "contributing_fold_count",
            "excluded_fold_count",
            "unavailable_fold_count",
            "median_coefficient",
            "iqr_low",
            "iqr_high",
            "mean_absolute_coefficient",
            "selection_count",
            "selection_frequency",
        ],
    )
    if not feature_table.empty:
        feature_table = feature_table.sort_values(
            ["mean_absolute_coefficient", "feature_id"],
            ascending=[False, True],
            na_position="last",
            ignore_index=True,
        )
    fold_rows = []
    fold_labels = arrays["fold_labels"]
    effective_counts = arrays["effective_feature_counts"][
        target_index, region_index, representation_index, time_index
    ]
    train_counts = arrays["train_counts"][
        target_index, region_index, representation_index, time_index
    ]
    test_counts = arrays["test_counts"][
        target_index, region_index, representation_index, time_index
    ]
    for fold_position, fold_label in enumerate(fold_labels.tolist()):
        values = coefficients[fold_position]
        if statuses[fold_position] == "valid":
            nonzero_count = int(
                np.count_nonzero(
                    np.isfinite(values)
                    & (np.abs(values) > NONZERO_COEFFICIENT_TOLERANCE)
                )
            )
            denominator = int(effective_counts[fold_position])
            fraction = nonzero_count / denominator if denominator else np.nan
        else:
            nonzero_count = 0
            fraction = np.nan
        fold_rows.append(
            {
                "outer_fold": int(fold_label),
                "fit_status": str(statuses[fold_position]),
                "failure_reason": str(reasons[fold_position]),
                "train_trial_count": int(train_counts[fold_position]),
                "test_trial_count": int(test_counts[fold_position]),
                "effective_feature_count": int(effective_counts[fold_position]),
                "nonzero_feature_count": nonzero_count,
                "nonzero_feature_fraction": fraction,
            }
        )
    return feature_table, pd.DataFrame(fold_rows)


def plot_unit_coefficients(
    saved_run: Mapping[str, object],
    *,
    target: str,
    region: str,
    time_bin_index: int,
    condition: str = "all",
) -> plt.Figure:
    """Plot direct-unit outer-fold coefficients with median and IQR summaries.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Valid loader-shaped saved run.
    target : str
        Exact saved target identifier.
    region : {"PFC", "HPC", "PFC+HPC"}
        Selected regional configuration.
    time_bin_index : int
        Zero-based saved time-bin position.
    condition : str, default="all"
        Exact saved condition whose direct-unit coefficients are plotted.

    Returns
    -------
    matplotlib.figure.Figure
        Opaque-white reliance summary. Finite per-fold points, exact zeros,
        fold-local excluded features, and unavailable fits use distinct legend
        entries. IQR is descriptive across CV fits, not a confidence interval.
    """
    arrays = select_condition_arrays(saved_run, condition)
    feature_table, fold_table = summarize_unit_coefficients(
        saved_run,
        target=target,
        region=region,
        time_bin_index=time_bin_index,
        condition=condition,
    )
    target_index = _label_index(arrays["target_labels"], target, "target")
    region_index = _label_index(arrays["region_labels"], region, "region")
    representation_index = _label_index(
        arrays["representation_labels"], "units", "representation"
    )
    active = arrays["coefficient_active_masks"][region_index, representation_index]
    feature_ids = arrays["coefficient_feature_ids"][region_index, representation_index, active]
    all_values = arrays["coefficient_values"][
        target_index, region_index, representation_index, time_bin_index
    ]
    values = np.asarray(all_values[:, active], dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    original_positions = {str(identifier): index for index, identifier in enumerate(feature_ids)}
    figure_height = max(4.5, 0.42 * max(1, feature_table.shape[0]) + 2.5)
    figure, axis = plt.subplots(figsize=(10.5, figure_height))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    y_positions = np.arange(feature_table.shape[0], dtype=float)
    for y_position, row in feature_table.iterrows():
        original = original_positions[str(row["feature_id"])]
        fold_values = values[:, original]
        finite = np.isfinite(fold_values)
        zero = finite & (np.abs(fold_values) <= NONZERO_COEFFICIENT_TOLERANCE)
        nonzero = finite & ~zero
        axis.scatter(
            fold_values[nonzero],
            np.full(int(nonzero.sum()), y_position),
            color="#2166ac",
            alpha=0.65,
            s=24,
        )
        axis.scatter(
            fold_values[zero],
            np.full(int(zero.sum()), y_position),
            facecolors="white",
            edgecolors="black",
            s=30,
        )
        if int(row["excluded_fold_count"]) > 0:
            axis.scatter(
                [1.02],
                [y_position],
                transform=axis.get_yaxis_transform(),
                clip_on=False,
                marker="x",
                color="#f4a261",
                s=35,
            )
        if int(row["unavailable_fold_count"]) > 0:
            axis.scatter(
                [1.06],
                [y_position],
                transform=axis.get_yaxis_transform(),
                clip_on=False,
                marker="x",
                color="#7f7f7f",
                s=35,
            )
        if np.isfinite(row["median_coefficient"]):
            axis.errorbar(
                float(row["median_coefficient"]),
                y_position,
                xerr=np.array(
                    [
                        [float(row["median_coefficient"] - row["iqr_low"])],
                        [float(row["iqr_high"] - row["median_coefficient"])],
                    ]
                ),
                color="black",
                marker="D",
                markersize=4,
                capsize=3,
            )
    axis.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.scatter([], [], color="#2166ac", label="nonzero")
    axis.scatter([], [], facecolors="white", edgecolors="black", label="zero")
    axis.scatter([], [], marker="x", color="#f4a261", label="excluded")
    axis.scatter([], [], marker="x", color="#7f7f7f", label="unavailable")
    axis.set_yticks(y_positions)
    axis.set_yticklabels(feature_table["feature_id"].tolist())
    axis.invert_yaxis()
    scale = str(saved_run["meta"]["units"]["coefficient_values"][target])
    time_s = float(arrays["time_bin_centers_s"][time_bin_index])
    axis.set_xlabel(scale)
    axis.set_ylabel("Stable unit identity")
    axis.set_title(
        f"{_target_display_label(target)} - {region} - "
        f"{_condition_display_label(condition)} at {time_s:g} s"
    )
    axis.legend(loc="best")
    axis.tick_params(colors="black")
    if feature_table.empty:
        axis.text(
            0.5,
            0.5,
            "No active direct-unit features are available for this region.",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="black",
        )
    caption_prefix = (
        "No active direct-unit features. " if feature_table.empty else ""
    )
    caption = caption_prefix + (
        "Points are signed outer-fold coefficients; diamonds and bars show median and IQR. "
        "IQR is variation across CV fits, not a confidence interval. Coefficients describe "
        f"decoder reliance, not causal contribution. Scale: {scale}. "
        f"Condition: {condition}. Outer fits: {fold_table.shape[0]}."
    )
    figure.text(0.02, 0.015, caption, color="black", fontsize=9, wrap=True)
    figure.subplots_adjust(left=0.26, right=0.90, top=0.90, bottom=0.18)
    return figure
