"""Production-facing CT026 Power validation runner without Streamlit dependencies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.lfp_summary import power_runtime
from src.neural_analysis.lfp_summary.runtime_common import load_configured_trial_table
from src.neural_analysis.lfp_summary.cache import (
    ComponentStatus,
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
)
from src.neural_analysis.lfp_summary.models import (
    LFPSiteConfig,
    LFPSummaryConfig,
    TrialFilterConfig,
    canonical_config_json,
    default_lfp_summary_config,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary.pipeline import (
    ComponentRunResult,
    PipelineDependencies,
    compute_power_component,
)
from src.neural_analysis.lfp_summary.plotting import (
    POWER_BASE_CONDITIONS,
    POWER_SUBDIVISION_CONDITIONS,
    PlotContext,
    plot_band_power_summary,
    plot_condition_psd,
)


_SESSION_ID = "CT026_2026-08-01_130853"
_RUN_LABEL = "lfp_power_validation"


@dataclass(frozen=True)
class PowerValidationDependencies:
    """Injected Power execution, cache, plotting, timing, and reporting seams.

    Every callable receives cache-backed arrays or immutable configuration only.
    ``monotonic_seconds`` returns seconds and ``peak_memory_bytes`` returns bytes.
    The plotting seams return unsaved figures; ``save_png`` and ``close_figure``
    own their respective side effects.
    """

    pipeline_dependencies: PipelineDependencies | object
    compute_power_component: Callable[[LFPSummaryConfig, object], ComponentRunResult]
    load_manifest: Callable[[Path, LFPSummaryConfig], dict[str, object]]
    assess_component_status: Callable[
        [Path, str, LFPSummaryConfig, Mapping[str, Any]], ComponentStatus
    ]
    load_power_arrays: Callable[[Path, Mapping[str, Any]], dict[str, np.ndarray]]
    plot_condition_psd: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_band_power_summary: Callable[..., tuple[Any, Mapping[str, Any]]]
    save_png: Callable[[Any, Path], None]
    close_figure: Callable[[Any], None]
    now_utc: Callable[[], str]
    monotonic_seconds: Callable[[], float]
    peak_memory_bytes: Callable[[], int]
    source_identifiers: Callable[[LFPSummaryConfig], dict[str, str]]


@dataclass(frozen=True)
class PowerValidationResult:
    """Completed immutable-report paths and measured Power validation metadata.

    Paths are all direct children of ``run_directory``. Times are seconds,
    memory and file sizes are bytes, and trial/exclusion counts are categorical.
    ``report`` is JSON-compatible metadata without raw traces or PSD arrays.
    """

    component: str
    run_directory: Path
    manifest_snapshot_path: Path
    configuration_snapshot_path: Path
    summary_path: Path
    log_path: Path
    source_identifiers_path: Path
    png_paths: tuple[Path, ...]
    wall_time_s: float
    peak_memory_bytes: int
    cache_size_bytes: int
    component_size_bytes: int
    trial_count: int
    exclusion_count: int
    report: dict[str, object]


def build_ct026_power_config(session_path: Path) -> LFPSummaryConfig:
    """Build the fixed first-inspection CT026 Open Ephys Power configuration.

    Parameters
    ----------
    session_path : pathlib.Path
        CT026 session root containing ``processed`` and ``ephys``. It is a
        filesystem path; this function does not open any source files.

    Returns
    -------
    LFPSummaryConfig
        Validated configuration using the augmented trial CSV, generic processed
        Power cache, Open Ephys ProbeA/ProbeB ``lfp.dat`` files, probe sync NPZs,
        PFC=5/HPC1=222/HPC2=14, all/all filters, and fixed seed zero.
    """
    root = Path(session_path)
    derived = root / "ephys" / "derived"
    aligned = root / "ephys" / "aligned" / "aligned_open_ephys"
    probe_a = derived / "Record_Node_101_Neuropix-PXI-110.ProbeA"
    probe_b = derived / "Record_Node_101_Neuropix-PXI-110.ProbeB"
    sites = (
        LFPSiteConfig(
            "PFC", "PFC channel 5", "open_ephys", probe_a / "lfp.dat",
            aligned / "probeA_sync.npz", "ProbeA", 5, "uV", 2500.0,
        ),
        LFPSiteConfig(
            "HPC1", "HPC1 channel 222", "open_ephys", probe_b / "lfp.dat",
            aligned / "probeB_sync.npz", "ProbeB", 222, "uV", 2500.0,
        ),
        LFPSiteConfig(
            "HPC2", "HPC2 channel 14", "open_ephys", probe_b / "lfp.dat",
            aligned / "probeB_sync.npz", "ProbeB", 14, "uV", 2500.0,
        ),
    )
    default = default_lfp_summary_config()
    config = replace(
        default,
        session_id=_SESSION_ID,
        session_path=root,
        output_directory=root / "processed" / "lfp_summary_cache",
        sites=sites,
        site_pairs=(("PFC", "HPC1"), ("PFC", "HPC2"), ("HPC1", "HPC2")),
        trial_filter=TrialFilterConfig(),
        trial_table_path=root / "processed" / f"{_SESSION_ID}_augmented_trials.csv",
        random_seed=0,
    )
    validate_lfp_summary_config(config)
    return config


def run_power_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: PowerValidationDependencies,
) -> PowerValidationResult:
    """Compute only Power, validate its cache, then write one immutable report run.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated active session configuration. Its generic cache path is
        disposable; source paths, units, and numerical axes are unchanged.
    run_parent : pathlib.Path
        Parent directory for timestamped human-readable reports. All report
        writes remain beneath this path; it is not created before Power/cache
        success.
    dependencies : PowerValidationDependencies
        Real production seams or test fakes. Plotting receives only reloaded
        cache arrays; timings are seconds and memory is bytes.

    Returns
    -------
    PowerValidationResult
        Run paths, twelve cache-backed PNG paths, measurements, counts, and report
        metadata. No raw LFP arrays or wavelets are returned.

    Raises
    ------
    FileExistsError
        If the timestamped report path already exists, before Power computation.
    RuntimeError
        If Power fails or its committed cache is not compatible. No report
        directory is created in either failure case.
    """
    parent, run_directory = _planned_run_directory(config, run_parent, dependencies)

    start_s = float(dependencies.monotonic_seconds())
    component_result = dependencies.compute_power_component(
        config,
        dependencies.pipeline_dependencies,
    )
    stop_s = float(dependencies.monotonic_seconds())
    if component_result.state != "complete":
        raise RuntimeError(component_result.error or "Power component failed")

    wall_time_s = stop_s - start_s
    peak_memory_bytes = int(dependencies.peak_memory_bytes())
    manifest, status, arrays = _load_compatible_power(config, dependencies)
    return _write_power_validation_report(
        config,
        parent,
        run_directory,
        manifest,
        arrays,
        status,
        wall_time_s,
        peak_memory_bytes,
        dependencies,
        power_recomputed=True,
    )


def render_cached_power_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: PowerValidationDependencies,
    *,
    power_wall_time_s: float,
    power_peak_memory_bytes: int,
) -> PowerValidationResult:
    """Write revised Power figures from a compatible cache without recomputation.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active immutable configuration used to assess cache compatibility.
    run_parent : pathlib.Path
        Parent of the new timestamped human-readable report directory.
    dependencies : PowerValidationDependencies
        Cache loading, plotting, reporting, and source-identity seams. The
        compute and monotonic-clock seams are deliberately not called.
    power_wall_time_s : float
        Seconds measured for the Power computation that produced the cache.
    power_peak_memory_bytes : int
        Peak resident memory in bytes measured for that Power computation.

    Returns
    -------
    PowerValidationResult
        New twelve-PNG report paths and preserved Power performance metrics.

    Raises
    ------
    FileExistsError
        If the timestamped report directory already exists.
    RuntimeError
        If the existing Power cache is not compatible with ``config``.
    ValueError
        If either supplied performance measurement is negative or nonfinite.
    """
    wall_time_s = float(power_wall_time_s)
    peak_memory_bytes = int(power_peak_memory_bytes)
    if not np.isfinite(wall_time_s) or wall_time_s < 0.0:
        raise ValueError("power_wall_time_s must be finite and nonnegative")
    if peak_memory_bytes < 0:
        raise ValueError("power_peak_memory_bytes must be nonnegative")
    parent, run_directory = _planned_run_directory(config, run_parent, dependencies)
    manifest, status, arrays = _load_compatible_power(config, dependencies)
    return _write_power_validation_report(
        config,
        parent,
        run_directory,
        manifest,
        arrays,
        status,
        wall_time_s,
        peak_memory_bytes,
        dependencies,
        power_recomputed=False,
    )


def _planned_run_directory(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: PowerValidationDependencies,
) -> tuple[Path, Path]:
    """Return the report parent and unused timestamped child path.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active session identity used in the directory name.
    run_parent : pathlib.Path
        Parent filesystem path for human-readable outputs.
    dependencies : PowerValidationDependencies
        Supplies a filesystem-safe UTC timestamp string.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Parent and planned run paths. Neither path is created.

    Raises
    ------
    FileExistsError
        If the planned immutable run directory already exists.
    """
    parent = Path(run_parent)
    timestamp = dependencies.now_utc()
    run_directory = parent / f"{config.session_id}_{_RUN_LABEL}_{timestamp}"
    if run_directory.exists():
        raise FileExistsError(f"validation run already exists: {run_directory}")
    return parent, run_directory


def _load_compatible_power(
    config: LFPSummaryConfig,
    dependencies: PowerValidationDependencies,
) -> tuple[dict[str, object], ComponentStatus, dict[str, np.ndarray]]:
    """Load one compatible Power manifest and its validated cache arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active cache path and fingerprint inputs.
    dependencies : PowerValidationDependencies
        Manifest, status-assessment, and safe component-loading seams.

    Returns
    -------
    tuple[dict[str, object], ComponentStatus, dict[str, numpy.ndarray]]
        Compatible manifest, status, and named Power arrays with cached units
        and axes unchanged.

    Raises
    ------
    RuntimeError
        If the Power cache is missing, stale, running, or failed.
    """
    manifest = dependencies.load_manifest(config.output_directory, config)
    status = dependencies.assess_component_status(
        config.output_directory,
        "power",
        config,
        manifest,
    )
    if status.status != "compatible":
        detail = "; ".join(status.differences)
        message = f"Power cache is not compatible: {status.status} {detail}"
        raise RuntimeError(message.strip())
    arrays = dependencies.load_power_arrays(config.output_directory, manifest)
    return manifest, status, arrays


def _write_power_validation_report(
    config: LFPSummaryConfig,
    parent: Path,
    run_directory: Path,
    manifest: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    status: ComponentStatus,
    wall_time_s: float,
    peak_memory_bytes: int,
    dependencies: PowerValidationDependencies,
    *,
    power_recomputed: bool,
) -> PowerValidationResult:
    """Render cache-only figures and write one complete immutable report.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active session configuration and generic cache path.
    parent, run_directory : pathlib.Path
        Report parent and unused timestamped child filesystem paths.
    manifest : Mapping[str, object]
        Compatible manifest metadata without raw numerical arrays.
    arrays : Mapping[str, numpy.ndarray]
        Validated Power cache arrays retaining their named axes and units.
    status : ComponentStatus
        Compatible Power status used in report metadata.
    wall_time_s : float
        Power computation duration in seconds.
    peak_memory_bytes : int
        Power computation peak resident memory in bytes.
    dependencies : PowerValidationDependencies
        Plot, save, close, and source-identity seams.
    power_recomputed : bool
        Whether this report invocation recomputed Power before cache loading.

    Returns
    -------
    PowerValidationResult
        Paths to twelve figures and report artifacts plus measured metadata.
    """
    parent.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir()
    png_paths = _render_cached_power_pngs(config, arrays, run_directory, dependencies)
    cache_size_bytes = _directory_size_bytes(config.output_directory)
    component_size_bytes = _component_size_bytes(config.output_directory, manifest)
    report = _build_report(
        config,
        arrays,
        status,
        wall_time_s,
        peak_memory_bytes,
        cache_size_bytes,
        component_size_bytes,
        power_recomputed=power_recomputed,
    )
    paths = _write_report_artifacts(
        run_directory,
        manifest,
        config,
        report,
        dependencies.source_identifiers(config),
    )
    return PowerValidationResult(
        component="power",
        run_directory=run_directory,
        manifest_snapshot_path=paths["manifest"],
        configuration_snapshot_path=paths["configuration"],
        summary_path=paths["summary"],
        log_path=paths["log"],
        source_identifiers_path=paths["source_identifiers"],
        png_paths=png_paths,
        wall_time_s=wall_time_s,
        peak_memory_bytes=peak_memory_bytes,
        cache_size_bytes=cache_size_bytes,
        component_size_bytes=component_size_bytes,
        trial_count=int(report["trial_count"]),
        exclusion_count=int(report["exclusion_count"]),
        report=report,
    )


def make_production_power_validation_dependencies() -> PowerValidationDependencies:
    """Bind the validation runner to runtime, cache, Matplotlib, and process seams.

    Returns
    -------
    PowerValidationDependencies
        Production dependencies with no Streamlit import. Cache arrays are loaded
        with the existing safe I/O API; clock values are seconds and peak memory
        is a Linux-independent best-effort byte measurement.
    """
    pipeline = power_runtime.make_power_pipeline_dependencies(
        trial_table_loader=load_configured_trial_table,
    )

    def load_manifest(directory: Path, config: LFPSummaryConfig) -> dict[str, object]:
        """Load a Power manifest using the active immutable configuration."""
        return load_or_initialize_manifest(directory, config)

    def load_power(directory: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Load validated Power arrays from the generic disposable cache."""
        return load_component_arrays(directory / "power.npz", manifest, "power")

    def save_png(figure: Any, path: Path) -> None:
        """Save one opaque-light Matplotlib figure as a PNG report artifact."""
        figure.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")

    def now_utc() -> str:
        """Return a filesystem-safe UTC timestamp for an immutable run directory."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    return PowerValidationDependencies(
        pipeline,
        compute_power_component,
        load_manifest,
        assess_component_status,
        load_power,
        plot_condition_psd,
        plot_band_power_summary,
        save_png,
        plt.close,
        now_utc,
        time.monotonic,
        _production_peak_memory_bytes,
        _production_source_identifiers,
    )


def _render_cached_power_pngs(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    run_directory: Path,
    dependencies: PowerValidationDependencies,
) -> tuple[Path, ...]:
    """Render base and subdivision PSD/band summaries for each cached site.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable labels, epoch bounds, band settings, and source units.
    arrays : Mapping[str, numpy.ndarray]
        Reloaded Power cache arrays. Required axes are site/trial/epoch/frequency
        and site/trial/epoch/band; no raw file loader is used.
    run_directory : pathlib.Path
        Existing timestamped report directory receiving twelve PNGs.
    dependencies : PowerValidationDependencies
        Cache-only plotting, saving, and closing seams.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Twelve paths: PSD and band summaries for both condition groups per site.
    """
    site_ids = tuple(str(value) for value in arrays["site_ids"])
    condition_names = tuple(str(value) for value in arrays["condition_names"])
    epoch_names = tuple(str(value) for value in arrays["epoch_names"])
    band_names = tuple(str(value) for value in arrays["band_names"])
    membership = np.asarray(arrays["condition_membership"], dtype=bool)
    filter_membership = np.asarray(arrays["filter_membership"], dtype=bool)
    objective_valid = np.asarray(arrays["objective_valid"], dtype=bool)
    site_valid = np.asarray(arrays["site_valid"], dtype=bool)
    excluded = np.asarray(arrays["user_excluded"], dtype=bool)
    context = _plot_context(config)
    frequency_hz = np.asarray(arrays["frequency_hz"], dtype=float)
    display_frequency = frequency_hz <= 100.0
    if not np.any(display_frequency):
        raise ValueError("Power cache frequency grid does not include 0-100 Hz")
    condition_groups = (
        ("base", POWER_BASE_CONDITIONS),
        ("subdivisions", POWER_SUBDIVISION_CONDITIONS),
    )
    cached_name_to_index = {name: index for index, name in enumerate(condition_names)}
    if len(cached_name_to_index) != len(condition_names):
        raise ValueError("Power cache condition_names must be unique")
    paths: list[Path] = []
    for site_index, site_id in enumerate(site_ids):
        valid_condition = (
            membership
            & filter_membership[:, None]
            & objective_valid[site_index, :, None]
            & site_valid[site_index, :, None]
            & ~excluded[:, None]
        )
        site_whole_psd = np.asarray(arrays["normalized_psd_session_db"])[
            site_index,
            :,
            0,
        ]
        site_band_power = np.asarray(arrays["band_power_session_db"])[site_index]
        for group_label, group_names in condition_groups:
            try:
                group_indices = np.array(
                    [cached_name_to_index[name] for name in group_names],
                    dtype=np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    f"Power cache lacks required condition {error.args[0]!r}"
                ) from error
            group_membership = valid_condition[:, group_indices]
            counts = group_membership.sum(axis=0, dtype=np.int64)
            condition_psd = _condition_trials(
                site_whole_psd[:, display_frequency],
                group_membership,
            )
            figure, _ = dependencies.plot_condition_psd(
                frequency_hz=frequency_hz[display_frequency],
                condition_trial_psd_db=condition_psd,
                condition_names=group_names,
                contributing_trial_counts=counts,
                site_label=site_id,
                epoch_name="whole",
                normalization="session_median",
                context=context,
            )
            path = run_directory / f"{site_id}_{group_label}_condition_psd.png"
            try:
                dependencies.save_png(figure, path)
            finally:
                dependencies.close_figure(figure)
            paths.append(path)

            condition_band = _condition_trials(site_band_power, group_membership)
            epoch_counts = np.repeat(counts[:, None], len(epoch_names), axis=1)
            figure, _ = dependencies.plot_band_power_summary(
                condition_trial_band_power_db=condition_band,
                condition_names=group_names,
                epoch_names=epoch_names,
                band_names=band_names,
                contributing_trial_counts=epoch_counts,
                site_label=site_id,
                normalization="session_median",
                context=context,
            )
            path = run_directory / f"{site_id}_{group_label}_band_power.png"
            try:
                dependencies.save_png(figure, path)
            finally:
                dependencies.close_figure(figure)
            paths.append(path)
    return tuple(paths)


def _condition_trials(values: np.ndarray, membership: np.ndarray) -> np.ndarray:
    """Stack condition-selected trial cache values without changing inner axes.

    Parameters
    ----------
    values : numpy.ndarray
        Float cache values with shape ``(trial,) + trailing_axes`` in dB.
    membership : numpy.ndarray
        Boolean shape ``(trial, condition)`` mask for one valid site.

    Returns
    -------
    numpy.ndarray
        Float shape ``(condition, trial) + trailing_axes``. Unselected trials are
        NaN, preserving cache trial positions for plotting without raw inputs.
    """
    output = np.full((membership.shape[1],) + values.shape, np.nan, dtype=float)
    for condition_index in range(membership.shape[1]):
        output[condition_index, membership[:, condition_index]] = values[
            membership[:, condition_index]
        ]
    return output


def _plot_context(config: LFPSummaryConfig) -> PlotContext:
    """Build immutable caption provenance from a Power configuration.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable session, window, notch, band, and site-unit metadata.

    Returns
    -------
    PlotContext
        Caption metadata with epoch seconds, gamma exclusion Hz, and first-site
        source voltage units. It contains no cached numerical arrays.
    """
    gamma = next(band for band in config.power.bands if band.name == "gamma")
    exclusion = gamma.excluded_intervals_hz[0] if gamma.excluded_intervals_hz else (58.0, 62.0)
    windows = config.analysis_windows
    return PlotContext(
        config.session_id,
        windows.alignment_event,
        {
            "whole": (windows.whole_start_s, windows.whole_stop_s),
            "before": (windows.before_start_s, windows.before_stop_s),
            "after": (windows.after_start_s, windows.after_stop_s),
        },
        config.power.notch_enabled,
        exclusion,
        "external brain reference; no CAR or bipolar rereference",
        config.sites[0].voltage_unit,
    )


def _build_report(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    status: ComponentStatus,
    wall_time_s: float,
    peak_memory_bytes: int,
    cache_size_bytes: int,
    component_size_bytes: int,
    *,
    power_recomputed: bool,
) -> dict[str, object]:
    """Summarize cache validity, workload, exclusions, and nine-filter projection.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active immutable configuration with no raw signal arrays.
    arrays : Mapping[str, numpy.ndarray]
        Reloaded Power cache used only for categorical counts and reason strings.
    status : ComponentStatus
        Compatible Power cache classification.
    wall_time_s : float
        Measured elapsed Power execution seconds.
    peak_memory_bytes, cache_size_bytes, component_size_bytes : int
        Process peak and persisted size measurements in bytes.
    power_recomputed : bool
        Whether this report invocation recomputed Power before cache loading.

    Returns
    -------
    dict[str, object]
        JSON-compatible metrics including trial/exclusion/warning counts and a
        linear nine-filter time/size projection; no raw arrays are retained.
    """
    condition_names = tuple(str(value) for value in arrays["condition_names"])
    site_ids = tuple(str(value) for value in arrays["site_ids"])
    membership = np.asarray(arrays["condition_membership"], dtype=bool)
    filter_membership = np.asarray(arrays["filter_membership"], dtype=bool)
    objective_valid = np.asarray(arrays["objective_valid"], dtype=bool)
    site_valid = np.asarray(arrays["site_valid"], dtype=bool)
    excluded = np.asarray(arrays["user_excluded"], dtype=bool)
    reasons = np.asarray(arrays["exclusion_reason_code"]).astype(str)
    warning_values = sorted({str(value) for value in reasons.ravel() if value})
    warnings = list(status.differences) + warning_values
    presession_available = np.asarray(
        arrays["presession_reference_available"],
        dtype=bool,
    )
    missing_presession = [
        site_ids[index]
        for index in range(len(site_ids))
        if not presession_available[index]
    ]
    if missing_presession:
        warnings.append(
            "presession reference unavailable: " + ", ".join(missing_presession)
        )
    unstable = np.asarray(arrays["condition_unstable"], dtype=bool)
    unstable_labels = [
        f"{condition_names[condition_index]}@{site_ids[site_index]}"
        for condition_index, site_index in np.argwhere(unstable)
    ]
    if unstable_labels:
        warnings.append("unstable low trial count: " + ", ".join(unstable_labels))
    effective_counts = np.asarray(
        arrays["condition_effective_trial_count"],
        dtype=np.int64,
    )
    condition_counts = {
        site_id: {
            condition_name: int(effective_counts[condition_index, site_index])
            for condition_index, condition_name in enumerate(condition_names)
        }
        for site_index, site_id in enumerate(site_ids)
    }
    site_valid_counts = {
        site_id: int(
            np.count_nonzero(
                filter_membership
                & objective_valid[site_index]
                & site_valid[site_index]
                & ~excluded
            )
        )
        for site_index, site_id in enumerate(site_ids)
    }
    has_exclusion_reason = np.any(reasons != "", axis=0)
    analyzed_excluded = filter_membership & (excluded | has_exclusion_reason)
    filtered_trial_count = int(np.count_nonzero(filter_membership))
    return {
        "session_id": config.session_id,
        "component": "power",
        "power_recomputed": power_recomputed,
        "cache_status": status.status,
        "wall_time_s": float(wall_time_s),
        "peak_memory_bytes": int(peak_memory_bytes),
        "cache_size_bytes": int(cache_size_bytes),
        "component_size_bytes": int(component_size_bytes),
        "trial_count": filtered_trial_count,
        "total_trial_count": int(membership.shape[0]),
        "filtered_trial_count": filtered_trial_count,
        "condition_effective_trial_count": condition_counts,
        "site_valid_trial_count": site_valid_counts,
        "exclusion_count": int(np.count_nonzero(analyzed_excluded)),
        "warnings": warnings,
        "nine_filter_projection": {
            "filter_count": 9,
            "projected_wall_time_s": float(wall_time_s) * 9.0,
            "projected_component_size_bytes": int(component_size_bytes) * 9,
        },
    }


def _write_report_artifacts(
    run_directory: Path,
    manifest: Mapping[str, object],
    config: LFPSummaryConfig,
    report: Mapping[str, object],
    source_identifiers: Mapping[str, str],
) -> dict[str, Path]:
    """Write report snapshots and readable summaries inside one new run directory.

    Parameters
    ----------
    run_directory : pathlib.Path
        Newly created timestamped directory beneath the caller's run parent.
    manifest, report, source_identifiers : Mapping[str, object]
        JSON-compatible cache metadata, metrics, and source identity strings.
    config : LFPSummaryConfig
        Immutable configuration serialized without raw signal arrays.

    Returns
    -------
    dict[str, pathlib.Path]
        Direct paths for manifest/configuration/source JSON, Markdown summary,
        and plain-text log. All writes remain inside ``run_directory``.
    """
    paths = {
        "manifest": run_directory / "manifest.json",
        "configuration": run_directory / "configuration.json",
        "source_identifiers": run_directory / "source_identifiers.json",
        "summary": run_directory / "run_summary.md",
        "log": run_directory / "run.log",
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["configuration"], json.loads(canonical_config_json(config)))
    _write_json(paths["source_identifiers"], source_identifiers)
    paths["summary"].write_text(_markdown_summary(report), encoding="ascii")
    paths["log"].write_text(json.dumps(dict(report), sort_keys=True) + "\n", encoding="ascii")
    return paths


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write one JSON-compatible mapping using ASCII and deterministic key order."""
    path.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="ascii")


def _markdown_summary(report: Mapping[str, object]) -> str:
    """Render concise human-readable Power validation metrics without raw arrays."""
    projection = report["nine_filter_projection"]
    return (
        "# LFP Power validation\n\n"
        "Goal: inspect condition-resolved Power summaries before running "
        "Synchrony or spike-phase analyses.\n\n"
        f"Session: {report['session_id']}\n\n"
        f"Power recomputed for this report: {report['power_recomputed']}\n\n"
        f"Wall time: {report['wall_time_s']} s\n\n"
        f"Peak memory: {report['peak_memory_bytes']} bytes\n\n"
        f"Trials: {report['trial_count']}; exclusions: {report['exclusion_count']}\n\n"
        f"Condition counts: {report['condition_effective_trial_count']}\n\n"
        f"Site-valid counts: {report['site_valid_trial_count']}\n\n"
        f"Warnings: {report['warnings']}\n\n"
        f"Nine-filter projection: {projection}\n"
    )


def _directory_size_bytes(directory: Path) -> int:
    """Return recursive regular-file size in bytes, or zero for a missing cache."""
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _component_size_bytes(directory: Path, manifest: Mapping[str, object]) -> int:
    """Return Power component NPZ size in bytes, or zero when unavailable."""
    entry = manifest.get("components", {}).get("power", {})
    filename = entry.get("file_name", "power.npz") if isinstance(entry, Mapping) else "power.npz"
    path = directory / str(filename)
    return path.stat().st_size if path.is_file() else 0


def _production_peak_memory_bytes() -> int:
    """Return a best-effort current-process peak resident-set size in bytes."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return 0
    if value <= 0:
        return 0
    return value if sys.platform == "darwin" else value * 1024


def _production_source_identifiers(config: LFPSummaryConfig) -> dict[str, str]:
    """Return current code identifiers and relevant module paths without raw arrays."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = "unavailable"
    return {
        "git_commit": commit or "unavailable",
        "runtime_module": str(Path(__file__).resolve().parents[1] / "lfp_summary_runtime.py"),
        "pipeline_module": str(Path(__file__).resolve().parents[1] / "lfp_summary_pipeline.py"),
        "validation_module": str(Path(__file__).resolve().parents[1] / "lfp_power_validation.py"),
        "session_id": config.session_id,
    }
