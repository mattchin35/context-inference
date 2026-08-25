"""Pure Matplotlib views for validated cached LFP-summary arrays."""

from __future__ import annotations
from dataclasses import dataclass
import re
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class PlotContext:
    """Immutable provenance shared by every cache-backed summary figure.

    Attributes
    ----------
    session_id : str
        Human-readable recording identifier.
    alignment_event : str
        Event at zero on every relative-time axis.
    epoch_bounds_s : dict[str, tuple[float, float]]
        Half-open epoch bounds in seconds relative to ``alignment_event``.
    notch_enabled : bool
        Whether the cached analysis applied the configured 60 Hz notch.
    gamma_exclusion_hz : tuple[float, float]
        Open interval in Hz omitted from gamma integration.
    reference_description : str
        LFP reference and preprocessing provenance shown verbatim.
    source_voltage_unit : str
        Physical unit of cached source traces, normally ``"uV"``.
    """

    session_id: str
    alignment_event: str
    epoch_bounds_s: dict[str, tuple[float, float]]
    notch_enabled: bool
    gamma_exclusion_hz: tuple[float, float]
    reference_description: str
    source_voltage_unit: str


def _figure(names: tuple[str, ...]) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Create opaque-white axes keyed by unique display names.

    Parameters
    ----------
    names : tuple[str, ...]
        Nonempty ordered axis names.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure and its named axes. No numerical data are transformed.
    """
    figure, array = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)))
    array = np.atleast_1d(array)
    figure.patch.set_facecolor("white")
    axes = {name: axis for name, axis in zip(names, array, strict=True)}
    for axis in axes.values():
        axis.set_facecolor("white")
        axis.tick_params(colors="black", labelsize=10)
    return figure, axes


def _caption(figure: plt.Figure, context: PlotContext, text: str) -> None:
    """Add the final bottom provenance caption without saving the figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Figure receiving the caption.
    context : PlotContext
        Session, timing, frequency, reference, and source-unit metadata.
    text : str
        Plot-specific count and interpretation statement.

    Returns
    -------
    None
        The supplied figure is modified in place; cached arrays are untouched.
    """
    notch = "enabled" if context.notch_enabled else "disabled"
    bounds = ", ".join(
        f"{key}={value}" for key, value in context.epoch_bounds_s.items()
    )
    figure.subplots_adjust(bottom=0.20, hspace=0.55)
    figure.text(
        0.01,
        0.015,
        (
            f"{text}. Alignment={context.alignment_event}; windows={bounds}; "
            f"60-Hz notch {notch}; gamma excludes {context.gamma_exclusion_hz[0]:g}-"
            f"{context.gamma_exclusion_hz[1]:g} Hz; {context.reference_description}; "
            f"source unit={context.source_voltage_unit}."
        ),
        fontsize=9,
        va="bottom",
    )


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    """Return a finite one-dimensional coordinate vector.

    Parameters
    ----------
    values : numpy.ndarray
        Coordinate values in the physical unit specified by the caller.
    name : str
        Coordinate name used in validation errors.

    Returns
    -------
    numpy.ndarray
        Float64 copy/view with shape ``(coordinate,)``.

    Raises
    ------
    ValueError
        If values are empty, nonfinite, or not one-dimensional.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def plot_condition_psd(
    frequency_hz: np.ndarray,
    condition_trial_psd_db: np.ndarray,
    condition_names: tuple[str, ...],
    contributing_trial_counts: np.ndarray,
    site_label: str,
    epoch_name: str,
    normalization: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot condition PSD medians and interquartile ranges.

    Parameters
    ----------
    frequency_hz : numpy.ndarray
        Finite shape ``(frequency,)`` canonical coordinates in Hz.
    condition_trial_psd_db : numpy.ndarray
        Shape ``(condition, trial, frequency)`` PSD normalized in dB. NaN marks
        an unavailable trial-frequency value.
    condition_names : tuple[str, ...]
        Labels for the condition axis.
    contributing_trial_counts : numpy.ndarray
        Integer shape ``(condition,)`` effective trial counts.
    site_label, epoch_name, normalization : str
        Display labels for site, half-open epoch, and dB reference method.
    context : PlotContext
        Figure provenance and source-voltage units.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure and the ``"spectrum"`` axis.

    Raises
    ------
    ValueError
        If coordinate, condition, trial-count, or frequency axes disagree.
    """
    frequency = _vector(frequency_hz, "frequency_hz")
    values = np.asarray(condition_trial_psd_db, float)
    counts = np.asarray(contributing_trial_counts, int)
    if (
        values.ndim != 3
        or values.shape[0] != len(condition_names)
        or values.shape[2] != frequency.size
        or counts.shape != (len(condition_names),)
    ):
        raise ValueError("condition PSD axes are invalid")
    figure, axes = _figure(("spectrum",))
    axis = axes["spectrum"]
    labels = []
    for index, name in enumerate(condition_names):
        median = np.nanmedian(values[index], axis=0)
        low = np.nanpercentile(values[index], 25, axis=0)
        high = np.nanpercentile(values[index], 75, axis=0)
        label = f"{name} (n={counts[index]})"
        labels.append(label)
        axis.plot(frequency, median, label=label)
        axis.fill_between(frequency, low, high, alpha=0.2)
    axis.set(
        xlabel="Frequency (Hz)",
        ylabel="PSD (dB)",
        title=f"{site_label}: {epoch_name} ({normalization})",
    )
    axis.legend(fontsize=9)
    _caption(
        figure,
        context,
        f"Conditions: {', '.join(labels)}; normalization={normalization}",
    )
    return figure, axes


def plot_band_power_summary(
    condition_trial_band_power_db: np.ndarray,
    condition_names: tuple[str, ...],
    epoch_names: tuple[str, ...],
    band_names: tuple[str, ...],
    contributing_trial_counts: np.ndarray,
    site_label: str,
    normalization: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot condition-level theta/gamma band-power medians.

    Parameters
    ----------
    condition_trial_band_power_db : numpy.ndarray
        Shape ``(condition, trial, epoch, band)`` normalized band power in dB;
        NaN represents unavailable integration.
    condition_names, epoch_names, band_names : tuple[str, ...]
        Labels for the corresponding axes. Epochs must include ``before`` and
        ``after``; bands must include ``theta`` and ``gamma``.
    contributing_trial_counts : numpy.ndarray
        Integer shape ``(condition, epoch)`` effective trial counts.
    site_label, normalization : str
        Site label and dB reference method.
    context : PlotContext
        Figure provenance and source-voltage units.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure and the ``"band_power"`` axis.

    Raises
    ------
    ValueError
        If axes disagree or required epoch/band labels are absent.
    """
    data = np.asarray(condition_trial_band_power_db, float)
    counts = np.asarray(contributing_trial_counts)
    if (
        data.ndim != 4
        or data.shape[0] != len(condition_names)
        or data.shape[2:] != (len(epoch_names), len(band_names))
        or counts.shape != (len(condition_names), len(epoch_names))
    ):
        raise ValueError("band-power axes are invalid")
    if not {"before", "after"}.issubset(epoch_names) or not {"theta", "gamma"}.issubset(
        band_names
    ):
        raise ValueError("required band/epoch labels missing")
    figure, axes = _figure(("band_power",))
    axis = axes["band_power"]
    labels = []
    x = []
    y = []
    order = [
        ("theta", "before"),
        ("theta", "after"),
        ("gamma", "before"),
        ("gamma", "after"),
    ]
    for ci, name in enumerate(condition_names):
        for bi, (band, epoch) in enumerate(order):
            ei = epoch_names.index(epoch)
            gi = band_names.index(band)
            x.append(ci * 5 + bi)
            y.append(np.nanmedian(data[ci, :, ei, gi]))
            labels.append(f"{name} {band}-{epoch}")
    axis.bar(x, y)
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Band power (dB)")
    axis.set_title(f"{site_label} ({normalization})")
    _caption(
        figure,
        context,
        f"Band medians; counts={counts.tolist()}; normalization={normalization}",
    )
    return figure, axes


def plot_phase_map(
    metric: np.ndarray,
    effective_trial_count: np.ndarray,
    frequency_hz: np.ndarray,
    relative_time_s: np.ndarray,
    metric_name: str,
    entity_label: str,
    condition_name: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot a phase metric beside its effective-trial count map.

    Parameters
    ----------
    metric : numpy.ndarray
        Dimensionless ITPC/ISPC values with shape ``(frequency, time)``.
    effective_trial_count : numpy.ndarray
        Nonnegative counts with shape ``(frequency, time)``.
    frequency_hz, relative_time_s : numpy.ndarray
        Finite coordinate vectors in Hz and seconds, respectively.
    metric_name, entity_label, condition_name : str
        Display labels for the statistic, site/site-pair, and trial condition.
    context : PlotContext
        Figure provenance and analysis-window metadata.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure with ``"metric"`` and ``"effective_count"`` axes.

    Raises
    ------
    ValueError
        If either map disagrees with the frequency/time coordinates.
    """
    f = _vector(frequency_hz, "frequency_hz")
    t = _vector(relative_time_s, "relative_time_s")
    m = np.asarray(metric, float)
    c = np.asarray(effective_trial_count, float)
    if m.shape != (f.size, t.size) or c.shape != m.shape:
        raise ValueError("phase-map axes are invalid")
    figure, axes = _figure(("metric", "effective_count"))
    for axis, data, label in (
        (axes["metric"], m, metric_name),
        (axes["effective_count"], c, "Effective trials"),
    ):
        mesh = axis.pcolormesh(t, f, np.ma.masked_invalid(data), shading="auto")
        figure.colorbar(mesh, ax=axis)
        axis.set(ylabel="Frequency (Hz)", title=label)
        axis.axvline(0, color="black", ls="--")
    axes["effective_count"].set_xlabel("Time from alignment (s)")
    _caption(
        figure,
        context,
        f"{metric_name} {entity_label}, condition={condition_name}; effective counts shown",
    )
    return figure, axes


def plot_phase_band_summary(
    estimates: np.ndarray,
    ci_low: np.ndarray,
    ci_high: np.ndarray,
    contributing_trial_counts: np.ndarray,
    labels: tuple[str, ...],
    band_name: str,
    epoch_name: str,
    metric_name: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot scalar phase estimates, bootstrap intervals, and counts.

    Parameters
    ----------
    estimates, ci_low, ci_high : numpy.ndarray
        Dimensionless shape ``(summary,)`` estimates and bootstrap 95% bounds.
    contributing_trial_counts : numpy.ndarray
        Integer shape ``(summary,)`` effective trial counts.
    labels : tuple[str, ...]
        Labels for the summary axis.
    band_name, epoch_name, metric_name : str
        Frequency-band, half-open epoch, and statistic labels.
    context : PlotContext
        Figure provenance and source-unit metadata.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure and the ``"summary"`` axis.

    Raises
    ------
    ValueError
        If estimates, intervals, counts, and labels do not share one axis.
    """
    e = np.asarray(estimates, float)
    lo = np.asarray(ci_low, float)
    hi = np.asarray(ci_high, float)
    n = np.asarray(contributing_trial_counts, int)
    if (
        e.shape != lo.shape
        or e.shape != hi.shape
        or e.shape != n.shape
        or e.shape != (len(labels),)
    ):
        raise ValueError("phase-summary axes are invalid")
    figure, axes = _figure(("summary",))
    axis = axes["summary"]
    x = np.arange(e.size)
    axis.errorbar(x, e, yerr=np.vstack((e - lo, hi - e)), fmt="o")
    axis.set_xticks(x, labels)
    axis.set_ylabel(metric_name)
    axis.set_title(f"{band_name} {epoch_name}")
    unstable = [labels[i] for i in np.flatnonzero(n < 10)]
    _caption(
        figure,
        context,
        f"95% bootstrap CI; counts={n.tolist()}; unstable low trial count: {unstable}",
    )
    return figure, axes


def plot_plv_distribution(
    trial_band_plv: np.ndarray,
    trial_valid_sample_counts: np.ndarray,
    trial_valid_sample_fractions: np.ndarray,
    epoch_names: tuple[str, ...],
    pair_label: str,
    band_name: str,
    condition_name: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot trial PLV distributions and paired-sample coverage.

    Parameters
    ----------
    trial_band_plv : numpy.ndarray
        Dimensionless shape ``(trial, epoch)`` band PLV; NaN is unavailable.
    trial_valid_sample_counts : numpy.ndarray
        Integer-valued shape ``(trial, epoch)`` paired sample counts.
    trial_valid_sample_fractions : numpy.ndarray
        Shape ``(trial, epoch)`` fractions of each configured epoch grid.
    epoch_names : tuple[str, ...]
        Labels for the epoch axis.
    pair_label, band_name, condition_name : str
        Site-pair, frequency-band, and trial-condition labels.
    context : PlotContext
        Figure provenance and window metadata.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure with ``"distribution"`` and ``"coverage"`` axes.

    Raises
    ------
    ValueError
        If PLV and coverage arrays do not share ``(trial, epoch)`` axes.
    """
    p = np.asarray(trial_band_plv, float)
    c = np.asarray(trial_valid_sample_counts, float)
    q = np.asarray(trial_valid_sample_fractions, float)
    if (
        p.ndim != 2
        or c.shape != p.shape
        or p.shape[1] != len(epoch_names)
    ):
        raise ValueError("PLV axes are invalid")
    figure, axes = _figure(("distribution", "coverage"))
    x = np.arange(p.shape[1])
    axes["distribution"].boxplot(
        [p[:, i][np.isfinite(p[:, i])] for i in x], positions=x
    )
    axes["distribution"].set_xticks(x, epoch_names)
    axes["distribution"].set_ylabel("PLV")
    axes["coverage"].plot(x, np.nanmedian(q, axis=0), marker="o")
    axes["coverage"].set_xticks(x, epoch_names)
    axes["coverage"].set(ylabel="Valid sample fraction", xlabel="Epoch")
    _caption(
        figure,
        context,
        f"PLV {pair_label}, {band_name}, {condition_name}; sample counts={c.astype(int).tolist()}",
    )
    return figure, axes


def plot_plv_exemplar(
    relative_time_s: np.ndarray,
    source_traces: np.ndarray,
    filtered_traces: np.ndarray,
    phase_rad: np.ndarray,
    site_labels: tuple[str, ...],
    pair_label: str,
    trial_index: int,
    percentile_label: str,
    pooled_plv: float,
    illustrative_trial_plv: float,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot one cached illustrative trial for a pooled PLV exemplar.

    Parameters
    ----------
    relative_time_s : numpy.ndarray
        Finite shape ``(time,)`` coordinate in seconds from alignment.
    source_traces, filtered_traces : numpy.ndarray
        Shape ``(site, time)`` traces in ``context.source_voltage_unit``.
    phase_rad : numpy.ndarray
        Shape ``(site, time)`` instantaneous phase in radians.
    site_labels : tuple[str, ...]
        Labels for the site axis.
    pair_label, percentile_label : str
        Site-pair and pooled-percentile labels.
    trial_index : int
        Table-row index of the illustrative trial.
    pooled_plv, illustrative_trial_plv : float
        Dimensionless pooled and trial-specific PLV values.
    context : PlotContext
        Figure provenance and source-voltage units.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure with ``"source"``, ``"filtered"``, and ``"phase"`` axes.

    Raises
    ------
    ValueError
        If trace/phase site or time axes disagree.
    """
    t = _vector(relative_time_s, "relative_time_s")
    s = np.asarray(source_traces, float)
    b = np.asarray(filtered_traces, float)
    p = np.asarray(phase_rad, float)
    if (
        s.shape != b.shape
        or s.shape != p.shape
        or s.shape != (len(site_labels), t.size)
    ):
        raise ValueError("PLV exemplar axes are invalid")
    figure, axes = _figure(("source", "filtered", "phase"))
    for i, label in enumerate(site_labels):
        axes["source"].plot(t, s[i], label=label)
        axes["filtered"].plot(t, b[i], label=label)
        axes["phase"].plot(t, p[i], label=label)
    for axis, name in (
        (axes["source"], "Source trace"),
        (axes["filtered"], "Filtered trace"),
        (axes["phase"], "Phase (rad)"),
    ):
        axis.set(ylabel=name)
        axis.legend(fontsize=8)
    axes["phase"].set_xlabel("Time from alignment (s)")
    _caption(
        figure,
        context,
        (
            f"{percentile_label} PLV exemplar, pair={pair_label}; "
            f"pooled={pooled_plv:g}; illustrative trial "
            f"{trial_index}={illustrative_trial_plv:g}"
        ),
    )
    return figure, axes


def plot_unit_ppc_map(
    ppc: np.ndarray,
    computable: np.ndarray,
    reliable: np.ndarray,
    spike_count: np.ndarray,
    frequency_hz: np.ndarray,
    unit_ids: tuple[str, ...],
    condition_name: str,
    site_label: str,
    epoch_name: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot unit PPC with computability, reliability, and spike diagnostics.

    Parameters
    ----------
    ppc : numpy.ndarray
        Dimensionless shape ``(unit, frequency)`` PPC in reference unit order.
    computable, reliable : numpy.ndarray
        Boolean shape ``(unit, frequency)`` masks for two-spike computability
        and the configured reliable-spike threshold.
    spike_count : numpy.ndarray
        Nonnegative shape ``(unit, frequency)`` valid phase-sample counts.
    frequency_hz : numpy.ndarray
        Finite shape ``(frequency,)`` coordinates in Hz.
    unit_ids : tuple[str, ...]
        Probe-qualified labels for the unit axis.
    condition_name, site_label, epoch_name : str
        Trial-condition, LFP-site, and half-open epoch labels.
    context : PlotContext
        Figure provenance and analysis-window metadata.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure with ``"ppc"``, ``"reliability"``, and
        ``"spike_count"`` axes.

    Raises
    ------
    ValueError
        If any unit/frequency axis disagrees.
    """
    f = _vector(frequency_hz, "frequency_hz")
    p = np.asarray(ppc, float)
    c = np.asarray(computable, bool)
    r = np.asarray(reliable, bool)
    n = np.asarray(spike_count, float)
    if (
        p.shape != (len(unit_ids), f.size)
        or c.shape != p.shape
        or r.shape != p.shape
        or n.shape != p.shape
    ):
        raise ValueError("PPC map axes are invalid")
    if np.any(r & ~c):
        raise ValueError("reliable PPC entries must also be computable")
    figure, axes = _figure(("ppc", "reliability", "spike_count"))
    reliability_state = c.astype(np.int8) + r.astype(np.int8)
    for key, data, title in (
        ("ppc", p, "PPC"),
        ("reliability", reliability_state, "0 unavailable, 1 computable, 2 reliable"),
        ("spike_count", n, "Spike count"),
    ):
        mesh = axes[key].pcolormesh(
            f,
            np.arange(len(unit_ids)),
            np.ma.masked_invalid(data),
            shading="auto",
        )
        figure.colorbar(mesh, ax=axes[key])
        axes[key].set(ylabel="Unit", title=title)
        axes[key].set_yticks(np.arange(len(unit_ids)), unit_ids)
    axes["spike_count"].set_xlabel("Frequency (Hz)")
    _caption(
        figure,
        context,
        f"PPC {condition_name}, {site_label}, {epoch_name}; unreliable entries shown separately",
    )
    return figure, axes


def plot_population_ppc_maps(
    median_ppc: np.ndarray,
    significant_fraction: np.ndarray,
    eligible_unit_count: np.ndarray,
    total_unit_count: np.ndarray,
    condition_names: tuple[str, ...],
    frequency_hz: np.ndarray,
    site_label: str,
    epoch_name: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot population PPC medians and FDR-significant prevalence.

    Parameters
    ----------
    median_ppc, significant_fraction : numpy.ndarray
        Dimensionless shape ``(condition, frequency)`` reliable-unit median PPC
        and eligible-denominator significant fraction. Zero eligible units are
        represented by NaN prevalence.
    eligible_unit_count, total_unit_count : numpy.ndarray
        Integer shape ``(condition, frequency)`` denominator diagnostics.
    condition_names : tuple[str, ...]
        Labels for the condition axis.
    frequency_hz : numpy.ndarray
        Finite shape ``(frequency,)`` coordinates in Hz.
    site_label, epoch_name : str
        LFP-site and half-open epoch labels.
    context : PlotContext
        Figure provenance and analysis-window metadata.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure with ``"median_ppc"`` and ``"prevalence"`` axes.

    Raises
    ------
    ValueError
        If condition/frequency axes or count axes disagree.
    """
    f = _vector(frequency_hz, "frequency_hz")
    m = np.asarray(median_ppc, float)
    q = np.asarray(significant_fraction, float)
    e = np.asarray(eligible_unit_count)
    total = np.asarray(total_unit_count)
    if (
        m.shape != (len(condition_names), f.size)
        or q.shape != m.shape
        or e.shape != m.shape
        or total.shape != m.shape
    ):
        raise ValueError("population PPC axes are invalid")
    figure, axes = _figure(("median_ppc", "prevalence"))
    for key, data, title in (
        ("median_ppc", m, "Median PPC"),
        ("prevalence", q, "Significant fraction"),
    ):
        mesh = axes[key].pcolormesh(
            f,
            np.arange(len(condition_names)),
            np.ma.masked_invalid(data),
            shading="auto",
        )
        figure.colorbar(mesh, ax=axes[key])
        axes[key].set(ylabel="Condition", title=title)
        axes[key].set_yticks(np.arange(len(condition_names)), condition_names)
    axes["prevalence"].set_xlabel("Frequency (Hz)")
    _caption(
        figure,
        context,
        (
            f"{site_label} {epoch_name}; eligible/total="
            f"{e.tolist()}/{total.tolist()}; NaN means no eligible units"
        ),
    )
    return figure, axes


def plot_ppc_band_summary(
    condition_unit_band_ppc: np.ndarray,
    reliable: np.ndarray,
    condition_names: tuple[str, ...],
    epoch_names: tuple[str, ...],
    band_names: tuple[str, ...],
    unit_count: int,
    site_label: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot reliable-only PPC medians by condition, epoch, and band.

    Parameters
    ----------
    condition_unit_band_ppc : numpy.ndarray
        Dimensionless shape ``(condition, unit, epoch, band)`` PPC summaries.
    reliable : numpy.ndarray
        Boolean array with the same axes; false entries do not contribute.
    condition_names, epoch_names, band_names : tuple[str, ...]
        Labels for the corresponding axes.
    unit_count : int
        Expected size of the reference unit axis.
    site_label : str
        LFP-site label.
    context : PlotContext
        Figure provenance and analysis-window metadata.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure and the ``"band_ppc"`` axis.

    Raises
    ------
    ValueError
        If the four cache axes or reliability mask disagree.
    """
    p = np.asarray(condition_unit_band_ppc, float)
    r = np.asarray(reliable, bool)
    if (
        p.ndim != 4
        or r.shape != p.shape
        or p.shape
        != (len(condition_names), unit_count, len(epoch_names), len(band_names))
    ):
        raise ValueError("PPC band axes are invalid")
    figure, axes = _figure(("band_ppc",))
    axis = axes["band_ppc"]
    labels = []
    values = []
    for ci, name in enumerate(condition_names):
        for bi, band in enumerate(band_names):
            for ei, epoch in enumerate(epoch_names):
                labels.append(f"{name} {band}-{epoch}")
                values.append(
                    np.nanmedian(np.where(r[ci, :, ei, bi], p[ci, :, ei, bi], np.nan))
                )
    axis.bar(np.arange(len(values)), values)
    axis.set_xticks(np.arange(len(values)), labels, rotation=35, ha="right")
    axis.set_ylabel("PPC")
    _caption(figure, context, f"{site_label}; reliable units only (total={unit_count})")
    return figure, axes


def plot_ppc_exemplar(
    frequency_hz: np.ndarray,
    pooled_ppc: np.ndarray,
    preferred_phase_rad: np.ndarray,
    representative_phase_hist_count: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
    relative_time_s: np.ndarray,
    source_trace: np.ndarray,
    filtered_trace: np.ndarray,
    spike_times_relative_s: np.ndarray,
    unit_id: str,
    trial_index: int,
    band_name: str,
    representative_frequency_hz: float,
    percentile_label: str,
    context: PlotContext,
) -> tuple[plt.Figure, dict[str, plt.Axes]]:
    """Plot a pooled PPC exemplar beside one illustrative trial.

    Parameters
    ----------
    frequency_hz, pooled_ppc, preferred_phase_rad : numpy.ndarray
        Shape ``(frequency,)`` coordinates in Hz, dimensionless PPC, and
        preferred spike phase in radians.
    representative_phase_hist_count : numpy.ndarray
        Integer shape ``(phase_bin,)`` pooled spike counts.
    phase_bin_edges_rad : numpy.ndarray
        Finite shape ``(phase_bin + 1,)`` bin edges in radians.
    relative_time_s : numpy.ndarray
        Finite shape ``(time,)`` coordinate in seconds from alignment.
    source_trace, filtered_trace : numpy.ndarray
        Shape ``(time,)`` illustrative traces in
        ``context.source_voltage_unit``.
    spike_times_relative_s : numpy.ndarray
        Finite shape ``(spike,)`` illustrative spike times in seconds.
    unit_id, band_name, percentile_label : str
        Probe-qualified unit, frequency-band, and pooled-percentile labels.
    trial_index : int
        Table-row index of the illustrative trial.
    representative_frequency_hz : float
        Polar-histogram frequency in Hz.
    context : PlotContext
        Figure provenance and source-voltage units.

    Returns
    -------
    tuple[matplotlib.figure.Figure, dict[str, matplotlib.axes.Axes]]
        Unsaved figure with ``"ppc"``, ``"polar"``, and ``"trace"`` axes.

    Raises
    ------
    ValueError
        If frequency, phase-bin, time, or spike-time contracts are invalid.
    """
    f = _vector(frequency_hz, "frequency_hz")
    p = np.asarray(pooled_ppc, float)
    edges = _vector(phase_bin_edges_rad, "phase_bin_edges_rad")
    t = _vector(relative_time_s, "relative_time_s")
    if (
        p.shape != f.shape
        or np.asarray(preferred_phase_rad).shape != f.shape
        or np.asarray(representative_phase_hist_count).shape != (edges.size - 1,)
        or np.asarray(source_trace).shape != t.shape
        or np.asarray(filtered_trace).shape != t.shape
    ):
        raise ValueError("PPC exemplar axes are invalid")
    preferred_phase = np.asarray(preferred_phase_rad, dtype=float)
    spike_times = np.asarray(spike_times_relative_s, dtype=float)
    if spike_times.ndim != 1 or not np.all(np.isfinite(spike_times)):
        raise ValueError("spike_times_relative_s must be a finite shape (spike,) array")
    if not band_name:
        raise ValueError("band_name must be nonempty")
    figure = plt.figure(figsize=(10, 7))
    figure.patch.set_facecolor("white")
    grid = figure.add_gridspec(2, 2)
    axes = {
        "ppc": figure.add_subplot(grid[0, 0]),
        "polar": figure.add_subplot(grid[:, 1], projection="polar"),
        "trace": figure.add_subplot(grid[1, 0]),
    }
    axes["ppc"].plot(f, p)
    axes["ppc"].set(xlabel="Frequency (Hz)", ylabel="PPC", title="Pooled PPC")
    centers = (edges[:-1] + edges[1:]) / 2
    axes["polar"].bar(centers, representative_phase_hist_count, width=np.diff(edges))
    phase_index = int(np.argmin(np.abs(f - representative_frequency_hz)))
    axes["polar"].set_title(
        f"{band_name}, {representative_frequency_hz:g} Hz; "
        f"preferred phase={preferred_phase[phase_index]:.2f} rad"
    )
    axes["trace"].plot(t, source_trace, label="source")
    axes["trace"].plot(t, filtered_trace, label="filtered")
    axes["trace"].vlines(
        spike_times, *axes["trace"].get_ylim(), color="black"
    )
    axes["trace"].set(
        xlabel="Time from alignment (s)", ylabel=context.source_voltage_unit
    )
    axes["trace"].legend(fontsize=8)
    for axis in axes.values():
        axis.set_facecolor("white")
    _caption(
        figure,
        context,
        (
            f"{percentile_label} pooled PPC exemplar, unit={unit_id}; "
            f"band={band_name}; illustrative trial {trial_index}; "
            f"{representative_frequency_hz:g} Hz"
        ),
    )
    return figure, axes


def build_summary_figure_filename(
    session_id: str,
    component: str,
    entity_label: str,
    condition_name: str,
    choice_filter: str,
    context_filter: str,
    epoch_name: str,
    normalization: str,
) -> str:
    """Build a deterministic, selection-specific PNG basename.

    Parameters
    ----------
    session_id, component, entity_label, condition_name : str
        Session, cache component, site/site-pair/unit, and trial-condition labels.
    choice_filter, context_filter : str
        Active trial-filter labels.
    epoch_name, normalization : str
        Half-open epoch and power-normalization labels.

    Returns
    -------
    str
        ASCII-safe basename ending in ``.png``. No directory is created and no
        figure is written.
    """
    parts = (
        session_id,
        component,
        entity_label,
        condition_name,
        choice_filter,
        context_filter,
        epoch_name,
        normalization,
    )
    return "__".join(_slug_filename_part(value) for value in parts) + ".png"


def _slug_filename_part(value: str) -> str:
    """Convert one selection label to an ASCII-safe filename segment.

    Parameters
    ----------
    value : str
        Human-readable selection label without physical units.

    Returns
    -------
    str
        Nonempty basename segment without path separators.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unspecified"
