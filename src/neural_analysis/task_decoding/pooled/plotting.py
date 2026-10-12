"""Compact saved-result-only figures for pooled task-variable decoding."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.task_decoding.pooled import results


FIGURE_KINDS = ("native", "transfer", "gap")


def _coordinate_index(values: np.ndarray, name: str, coordinate: str) -> int:
    """Return one exact coordinate position from a one-dimensional array.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional string coordinate array.
    name : str
        Human-readable axis name used in errors.
    coordinate : str
        Exact requested coordinate value.

    Returns
    -------
    int
        Zero-based position of ``coordinate``.
    """
    matches = np.flatnonzero(np.asarray(values).astype(str) == coordinate)
    if matches.size != 1:
        raise ValueError(f"Unknown or duplicate {name} coordinate: {coordinate}")
    return int(matches[0])


def _fold_mean(scores: np.ndarray, statuses: np.ndarray) -> np.ndarray | None:
    """Average valid fold scores without treating unavailable cells as zero.

    Parameters
    ----------
    scores : numpy.ndarray
        Float scores shaped ``(fold, time)`` in native metric units.
    statuses : numpy.ndarray
        UInt8 status indices shaped ``(fold, time)``; index one means valid.

    Returns
    -------
    numpy.ndarray or None
        Mean time course shaped ``(time,)`` or ``None`` when every cell is
        unavailable/not-applicable. Fold variation is not a confidence interval.
    """
    valid_scores = np.where(np.asarray(statuses) == 1, scores, np.nan)
    if not np.isfinite(valid_scores).any():
        return None
    counts = np.sum(np.isfinite(valid_scores), axis=0)
    totals = np.nansum(valid_scores, axis=0)
    return np.divide(
        totals,
        counts,
        out=np.full(totals.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )


def _metric_indices(arrays: Mapping[str, np.ndarray], target_index: int) -> tuple[int, ...]:
    """Return applicable saved metric positions for one target family.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Validated pooled result arrays.
    target_index : int
        Zero-based target coordinate.

    Returns
    -------
    tuple[int, ...]
        Balanced-accuracy/AUC positions for categorical targets, otherwise R2.
    """
    family = str(arrays["target_family_names"][target_index])
    names = arrays["metric_names"].astype(str).tolist()
    wanted = ("balanced_accuracy", "auc") if family == "categorical" else ("r2",)
    return tuple(names.index(name) for name in wanted)


def _line_specs(
    arrays: Mapping[str, np.ndarray],
    *,
    figure_kind: str,
    window_index: int,
    target_index: int,
    region_index: int,
    representation_index: int,
    metric_index: int,
) -> tuple[tuple[str, np.ndarray | None], ...]:
    """Extract labeled fold-mean curves for one reporting panel.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Validated saved arrays using schema-1 axis order.
    figure_kind : {"native", "transfer", "gap"}
        Required reporting family.
    window_index, target_index, region_index, representation_index, metric_index : int
        Zero-based coordinates for the fixed panel slice.

    Returns
    -------
    tuple[tuple[str, numpy.ndarray or None], ...]
        Label/time-course pairs. ``None`` records explicit unavailability.
    """
    conditions = arrays["condition_names"].astype(str).tolist()
    output: list[tuple[str, np.ndarray | None]] = []
    if figure_kind == "gap":
        for condition_index, condition in enumerate(conditions):
            cell = (
                window_index,
                condition_index,
                target_index,
                slice(None),
                region_index,
                representation_index,
                slice(None),
                metric_index,
            )
            output.append(
                (
                    condition,
                    _fold_mean(arrays["gap_scores"][cell], arrays["gap_status"][cell]),
                )
            )
        return tuple(output)

    train = arrays["evaluation_train_condition"].astype(str).tolist()
    test = arrays["evaluation_test_condition"].astype(str).tolist()
    for pair_index, (train_name, test_name) in enumerate(zip(train, test, strict=True)):
        is_native = train_name == test_name
        if figure_kind == "native" and not is_native:
            continue
        cell = (
            window_index,
            pair_index,
            target_index,
            slice(None),
            region_index,
            representation_index,
            slice(None),
            metric_index,
        )
        output.append(
            (
                f"{train_name} -> {test_name}",
                _fold_mean(
                    arrays["evaluation_scores"][cell],
                    arrays["evaluation_status"][cell],
                ),
            )
        )
    return tuple(output)


def plot_pooled_timecourses(
    saved_result: Mapping[str, object],
    *,
    figure_kind: str,
    window_name: str,
    target_name: str,
) -> matplotlib.figure.Figure:
    """Create one six-panel light-mode figure from validated saved arrays.

    Parameters
    ----------
    saved_result : mapping[str, object]
        Value returned by :func:`results.load_pooled_run`; arrays retain their
        declared schema-1 axes and native score units.
    figure_kind : {"native", "transfer", "gap"}
        Requested required completion figure.
    window_name, target_name : str
        Exact saved coordinates selecting one window/target cell.

    Returns
    -------
    matplotlib.figure.Figure
        Light-mode figure with rows for PFC/HPC/PFC+HPC and columns for
        PCA/direct-unit representations. Evaluation time is in seconds.
    """
    if figure_kind not in FIGURE_KINDS:
        raise ValueError(f"Unknown pooled figure kind: {figure_kind}")
    arrays_value = saved_result.get("arrays")
    metadata = saved_result.get("metadata")
    if not isinstance(arrays_value, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("saved_result must be a validated pooled result mapping.")
    arrays = arrays_value
    results.validate_pooled_result_arrays(arrays, metadata)
    window_index = _coordinate_index(
        arrays["training_window_names"], "training window", window_name
    )
    target_index = _coordinate_index(arrays["target_names"], "target", target_name)
    metric_indices = _metric_indices(arrays, target_index)
    metric_names = arrays["metric_names"].astype(str)
    time_s = arrays["evaluation_time_centers_s"]
    start_s = float(arrays["training_window_start_s"][window_index])
    stop_s = float(arrays["training_window_stop_s"][window_index])

    plt.style.use("default")
    figure, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    figure.patch.set_facecolor("white")
    for region_index, region in enumerate(arrays["region_names"].astype(str)):
        for representation_index, representation in enumerate(
            arrays["representation_names"].astype(str)
        ):
            axis = axes[region_index, representation_index]
            axis.set_facecolor("white")
            axis.axvspan(start_s, stop_s, color="#d8e8f5", alpha=0.7, label="training interval")
            if figure_kind == "gap":
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            plotted = False
            for metric_index in metric_indices:
                for label, curve in _line_specs(
                    arrays,
                    figure_kind=figure_kind,
                    window_index=window_index,
                    target_index=target_index,
                    region_index=region_index,
                    representation_index=representation_index,
                    metric_index=metric_index,
                ):
                    if curve is None:
                        continue
                    axis.plot(time_s, curve, label=f"{label} | {metric_names[metric_index]}")
                    plotted = True
            if not plotted:
                axis.text(
                    0.5,
                    0.5,
                    "Unavailable: no valid held-out metric cells",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            axis.set_title(f"{region} | {representation}")
            axis.set_ylabel(", ".join(metric_names[list(metric_indices)]))
            axis.grid(alpha=0.2)
            if plotted:
                axis.legend(fontsize=7, loc="best")
    for axis in axes[-1, :]:
        axis.set_xlabel("Evaluation time from choice (seconds)")
    temporal_scope = (
        "All displayed evaluation times are in-window for the full training-time "
        "distribution."
        if window_name == "full"
        else (
            "Shaded times are in-window and unshaded times are out-of-window "
            "held-out temporal generalization."
        )
    )
    caption = (
        f"{figure_kind.title()} pooled-window decoding for target {target_name}. "
        "Conditions/evaluation pairs: "
        f"{', '.join(arrays['condition_names'].astype(str))}; these are overlapping "
        "contrasts, not mutually exclusive levels. "
        f"Window {window_name} trains from {start_s:.1f} to {stop_s:.1f} seconds. "
        f"{temporal_scope} One decoder is repeatedly evaluated through time; native "
        "curves test temporal stability of one readout. Folds are "
        "grouped by behavioral block, fold variation is not a confidence interval, "
        "and unavailable cells are labeled. Native and transfer scores use the same "
        "destination held-out trials. Native and transfer are complete pipelines with "
        "separately trained preprocessing, so their gap cannot isolate estimator weights. "
        "An independently cross-validated time-specific curve can use a different "
        "transform-training window. Transfer failure or a descriptive curve difference "
        "alone does not prove remapping."
    )
    figure.suptitle(
        f"{figure_kind.title()} | {target_name} | {window_name}",
        fontsize=15,
    )
    figure.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8, wrap=True)
    figure.tight_layout(rect=(0.0, 0.09, 1.0, 0.95))
    return figure


def _validate_png(path: Path) -> None:
    """Validate one required nonempty PNG completion artifact.

    Parameters
    ----------
    path : pathlib.Path
        Required figure path.

    Returns
    -------
    None
        Returns only when the file has the standard PNG signature.
    """
    if not path.is_file() or not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Required pooled figure is not a valid PNG: {path.name}")


def save_pooled_decoding_figures(run_directory: Path | str) -> tuple[Path, ...]:
    """Publish exactly three required PNGs per saved window/target cell.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Directory containing a validated ``results.npz``. Plotting reads no
        spike arrays and performs no transforms or model fits.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Required PNG paths in window, target, then native/transfer/gap order.
    """
    directory = Path(run_directory)
    saved = results.load_pooled_run(directory)
    arrays = saved["arrays"]
    figures_directory = directory / "figures"
    figures_directory.mkdir(exist_ok=True)
    output: list[Path] = []
    for window_name in arrays["training_window_names"].astype(str):
        for target_name in arrays["target_names"].astype(str):
            for figure_kind in FIGURE_KINDS:
                destination = figures_directory / (
                    f"{figure_kind}__{window_name}__{target_name}.png"
                )
                if destination.exists():
                    _validate_png(destination)
                else:
                    figure = plot_pooled_timecourses(
                        saved,
                        figure_kind=figure_kind,
                        window_name=str(window_name),
                        target_name=str(target_name),
                    )
                    temporary = destination.with_name(f".{destination.name}.temporary")
                    try:
                        figure.savefig(temporary, format="png", dpi=140, facecolor="white")
                        temporary.replace(destination)
                    finally:
                        plt.close(figure)
                        if temporary.exists():
                            temporary.unlink()
                    _validate_png(destination)
                output.append(destination)
    required_names = {path.name for path in output}
    actual_names = {path.name for path in figures_directory.glob("*.png")}
    if actual_names != required_names:
        raise ValueError("Pooled figure directory contains missing or unexpected PNGs.")
    return tuple(output)
