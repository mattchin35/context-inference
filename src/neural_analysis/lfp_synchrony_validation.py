"""Production CT026 Synchrony validation without Streamlit dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis import lfp_summary_runtime
from src.neural_analysis.lfp_power_validation import build_ct026_power_config
from src.neural_analysis.lfp_summary_io import (
    ComponentStatus,
    assess_component_status,
    load_component_arrays,
    load_or_initialize_manifest,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    canonical_config_json,
)
from src.neural_analysis.lfp_summary_pipeline import (
    ComponentRunResult,
    PipelineDependencies,
    compute_synchrony_component,
)
from src.neural_analysis.lfp_summary_plotting import (
    PlotContext,
    plot_phase_band_summary,
    plot_phase_map,
    plot_plv_distribution,
    plot_plv_exemplar,
)


_RUN_LABEL = "lfp_synchrony_validation"


@dataclass(frozen=True)
class SynchronyValidationDependencies:
    """Injected Synchrony compute, cache, plot, timing, and report seams.

    Every callable receives immutable configuration or validated cache arrays.
    Clock values are seconds, memory values are bytes, source traces retain the
    cached voltage units, and phase values retain radians.
    """

    pipeline_dependencies: PipelineDependencies | object
    compute_synchrony_component: Callable[
        [LFPSummaryConfig, object], ComponentRunResult
    ]
    load_manifest: Callable[[Path, LFPSummaryConfig], dict[str, object]]
    assess_component_status: Callable[
        [Path, str, LFPSummaryConfig, Mapping[str, Any]], ComponentStatus
    ]
    load_synchrony_arrays: Callable[[Path, Mapping[str, Any]], dict[str, np.ndarray]]
    plot_phase_map: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_phase_band_summary: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_plv_distribution: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_plv_exemplar: Callable[..., tuple[Any, Mapping[str, Any]]]
    save_png: Callable[[Any, Path], None]
    close_figure: Callable[[Any], None]
    now_utc: Callable[[], str]
    monotonic_seconds: Callable[[], float]
    peak_memory_bytes: Callable[[], int]
    source_identifiers: Callable[[LFPSummaryConfig], dict[str, str]]


@dataclass(frozen=True)
class SynchronyValidationResult:
    """Immutable report paths and measured Synchrony validation metadata.

    Paths are direct children of ``run_directory``. Times use seconds, sizes
    use bytes, counts are categorical, and ``report`` contains no LFP or phase
    arrays.
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


def build_ct026_synchrony_config(session_path: Path) -> LFPSummaryConfig:
    """Build the fixed approved CT026 Synchrony configuration.

    Parameters
    ----------
    session_path : pathlib.Path
        CT026 session root containing the active trial CSV and Open Ephys data.

    Returns
    -------
    LFPSummaryConfig
        Validated PFC=5, HPC1=222, HPC2=14 configuration with three ordered
        pairs, 500-Hz phase output, configured 2-Hz frequencies, and seed zero.
    """
    return build_ct026_power_config(session_path)


def run_synchrony_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SynchronyValidationDependencies,
) -> SynchronyValidationResult:
    """Compute only Synchrony and write one cache-backed immutable report.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active site, pair, time, frequency, band, filter, and seed settings.
    run_parent : pathlib.Path
        Parent for a new timestamped human-readable report directory.
    dependencies : SynchronyValidationDependencies
        Production bindings or test fakes. Plot seams receive cache arrays only.

    Returns
    -------
    SynchronyValidationResult
        Figure/report paths, wall seconds, peak bytes, cache sizes, and counts.

    Raises
    ------
    FileExistsError
        If the immutable timestamped report directory already exists.
    RuntimeError
        If Synchrony computation fails or its committed cache is incompatible.
    """
    run_directory = _planned_run_directory(config, run_parent, dependencies)
    start_s = float(dependencies.monotonic_seconds())
    component_result = dependencies.compute_synchrony_component(
        config,
        dependencies.pipeline_dependencies,
    )
    stop_s = float(dependencies.monotonic_seconds())
    if component_result.state != "complete":
        raise RuntimeError(component_result.error or "Synchrony component failed")

    manifest, status, arrays = _load_compatible_synchrony(config, dependencies)
    return _write_validation_report(
        config,
        Path(run_parent),
        run_directory,
        manifest,
        status,
        arrays,
        stop_s - start_s,
        int(dependencies.peak_memory_bytes()),
        dependencies,
    )


def make_production_synchrony_validation_dependencies(
) -> SynchronyValidationDependencies:
    """Bind real Synchrony runtime, safe cache loading, Matplotlib, and clocks.

    Returns
    -------
    SynchronyValidationDependencies
        Production dependencies with no Streamlit or raw-data plotting seam.
    """
    pipeline = lfp_summary_runtime.make_synchrony_pipeline_dependencies(
        trial_table_loader=lfp_summary_runtime.load_configured_trial_table,
    )

    def load_manifest(directory: Path, config: LFPSummaryConfig) -> dict[str, object]:
        """Load the cache manifest for the active immutable configuration."""
        return load_or_initialize_manifest(directory, config)

    def load_synchrony(
        directory: Path,
        manifest: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        """Load and validate the named Synchrony NPZ arrays without pickle."""
        return load_component_arrays(
            directory / "synchrony.npz",
            manifest,
            "synchrony",
        )

    def save_png(figure: Any, path: Path) -> None:
        """Save one opaque-light cached summary as a 150-dpi PNG."""
        figure.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")

    def now_utc() -> str:
        """Return a filesystem-safe UTC timestamp."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    return SynchronyValidationDependencies(
        pipeline,
        compute_synchrony_component,
        load_manifest,
        assess_component_status,
        load_synchrony,
        plot_phase_map,
        plot_phase_band_summary,
        plot_plv_distribution,
        plot_plv_exemplar,
        save_png,
        plt.close,
        now_utc,
        time.monotonic,
        _production_peak_memory_bytes,
        _production_source_identifiers,
    )


def _planned_run_directory(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SynchronyValidationDependencies,
) -> Path:
    """Return one unused report path without creating directories.

    Parameters
    ----------
    config : LFPSummaryConfig
        Session identity used in the child directory name.
    run_parent : pathlib.Path
        Existing or future parent for immutable validation reports.
    dependencies : SynchronyValidationDependencies
        Supplies the filesystem-safe UTC timestamp string.

    Returns
    -------
    pathlib.Path
        Uncreated timestamped child path beneath ``run_parent``.

    Raises
    ------
    FileExistsError
        If that immutable report path already exists.
    """
    run_directory = Path(run_parent) / (
        f"{config.session_id}_{_RUN_LABEL}_{dependencies.now_utc()}"
    )
    if run_directory.exists():
        raise FileExistsError(f"validation run already exists: {run_directory}")
    return run_directory


def _load_compatible_synchrony(
    config: LFPSummaryConfig,
    dependencies: SynchronyValidationDependencies,
) -> tuple[dict[str, object], ComponentStatus, dict[str, np.ndarray]]:
    """Reload one compatible manifest and validated Synchrony cache arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active cache path and fingerprint inputs.
    dependencies : SynchronyValidationDependencies
        Manifest, status, and safe NPZ loading seams.

    Returns
    -------
    tuple[dict, ComponentStatus, dict[str, numpy.ndarray]]
        Compatible manifest/status and named arrays with cache axes and units
        unchanged.

    Raises
    ------
    RuntimeError
        If Synchrony is missing, stale, running, or failed.
    """
    manifest = dependencies.load_manifest(config.output_directory, config)
    status = dependencies.assess_component_status(
        config.output_directory,
        "synchrony",
        config,
        manifest,
    )
    if status.status != "compatible":
        detail = "; ".join(status.differences)
        raise RuntimeError(
            f"Synchrony cache is not compatible: {status.status} {detail}".strip()
        )
    arrays = dependencies.load_synchrony_arrays(config.output_directory, manifest)
    return manifest, status, arrays


def _write_validation_report(
    config: LFPSummaryConfig,
    run_parent: Path,
    run_directory: Path,
    manifest: Mapping[str, object],
    status: ComponentStatus,
    arrays: Mapping[str, np.ndarray],
    wall_time_s: float,
    peak_memory_bytes: int,
    dependencies: SynchronyValidationDependencies,
) -> SynchronyValidationResult:
    """Render cached figures and write snapshots after successful computation.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable active configuration and generic cache path.
    run_parent, run_directory : pathlib.Path
        Report parent and unused timestamped child paths.
    manifest : Mapping[str, object]
        Compatible cache metadata without numerical arrays.
    status : ComponentStatus
        Compatible Synchrony state.
    arrays : Mapping[str, numpy.ndarray]
        Validated Synchrony arrays with documented cache axes and units.
    wall_time_s : float
        Measured Synchrony computation duration in seconds.
    peak_memory_bytes : int
        Measured process peak resident memory in bytes.
    dependencies : SynchronyValidationDependencies
        Cache-only plot, file, and source-identity seams.

    Returns
    -------
    SynchronyValidationResult
        Immutable report paths, sizes, times, and categorical counts.
    """
    run_parent.mkdir(parents=True, exist_ok=True)
    run_directory.mkdir()
    png_paths = _render_cached_synchrony_pngs(
        config,
        arrays,
        run_directory,
        dependencies,
    )
    cache_size = _directory_size_bytes(config.output_directory)
    component_size = _component_size_bytes(config.output_directory, manifest)
    report = _build_report(
        config,
        arrays,
        status,
        wall_time_s,
        peak_memory_bytes,
        cache_size,
        component_size,
    )
    paths = _write_report_artifacts(
        run_directory,
        manifest,
        config,
        report,
        dependencies.source_identifiers(config),
    )
    return SynchronyValidationResult(
        "synchrony",
        run_directory,
        paths["manifest"],
        paths["configuration"],
        paths["summary"],
        paths["log"],
        paths["source_identifiers"],
        png_paths,
        float(wall_time_s),
        int(peak_memory_bytes),
        cache_size,
        component_size,
        int(report["trial_count"]),
        int(report["exclusion_count"]),
        report,
    )


def _render_cached_synchrony_pngs(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    run_directory: Path,
    dependencies: SynchronyValidationDependencies,
) -> tuple[Path, ...]:
    """Render maps, band summaries, PLV distributions, and cached exemplars.

    Parameters
    ----------
    config : LFPSummaryConfig
        Analysis windows, bands, notch state, and source units.
    arrays : Mapping[str, numpy.ndarray]
        Reloaded cache arrays on condition/site/pair/frequency/time/trial axes.
    run_directory : pathlib.Path
        Existing immutable directory receiving PNG files.
    dependencies : SynchronyValidationDependencies
        Plot, save, and close seams; no raw loader is available here.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Saved PNG paths in deterministic map/summary/PLV order.
    """
    site_ids = tuple(str(value) for value in arrays["site_ids"])
    condition_names = tuple(str(value) for value in arrays["condition_names"])
    pair_labels = _pair_labels(arrays)
    frequencies_hz = np.asarray(arrays["frequency_hz"], dtype=float)
    relative_time_s = np.asarray(arrays["relative_time_s"], dtype=float)
    context = _plot_context(config)
    paths: list[Path] = []

    for condition_index, condition_name in enumerate(condition_names):
        for site_index, site_id in enumerate(site_ids):
            path = run_directory / f"{site_id}_{condition_name}_itpc_map.png"
            figure, _ = dependencies.plot_phase_map(
                np.asarray(arrays["itpc"])[condition_index, site_index],
                np.asarray(arrays["itpc_effective_trial_count"])[
                    condition_index, site_index
                ],
                frequencies_hz,
                relative_time_s,
                "ITPC",
                site_id,
                condition_name,
                context,
            )
            _save_and_close(figure, path, dependencies)
            paths.append(path)
        for pair_index, pair_label in enumerate(pair_labels):
            path = run_directory / f"{pair_label}_{condition_name}_ispc_map.png"
            figure, _ = dependencies.plot_phase_map(
                np.asarray(arrays["ispc"])[condition_index, pair_index],
                np.asarray(arrays["ispc_effective_trial_count"])[
                    condition_index, pair_index
                ],
                frequencies_hz,
                relative_time_s,
                "ISPC",
                pair_label.replace("_", "-"),
                condition_name,
                context,
            )
            _save_and_close(figure, path, dependencies)
            paths.append(path)

    paths.extend(
        _render_phase_band_summaries(
            arrays,
            condition_names,
            site_ids,
            pair_labels,
            context,
            run_directory,
            dependencies,
        )
    )
    paths.extend(
        _render_plv_figures(
            config,
            arrays,
            condition_names,
            site_ids,
            pair_labels,
            context,
            run_directory,
            dependencies,
        )
    )
    return tuple(paths)


def _render_phase_band_summaries(
    arrays: Mapping[str, np.ndarray],
    condition_names: tuple[str, ...],
    site_ids: tuple[str, ...],
    pair_labels: tuple[str, ...],
    context: PlotContext,
    run_directory: Path,
    dependencies: SynchronyValidationDependencies,
) -> tuple[Path, ...]:
    """Render one combined ITPC/ISPC bootstrap summary per epoch and band.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Cache estimates/intervals on condition, entity, epoch, and band axes.
    condition_names, site_ids, pair_labels : tuple[str, ...]
        Ordered categorical labels matching those cache axes.
    context : PlotContext
        Caption provenance with seconds, Hz, and source-voltage units.
    run_directory : pathlib.Path
        Existing output directory receiving six phase-summary PNGs.
    dependencies : SynchronyValidationDependencies
        Cache-only plot, save, and close seams.

    Returns
    -------
    tuple[pathlib.Path, ...]
        One saved PNG path for every epoch-by-band combination.
    """
    epoch_names = tuple(str(value) for value in arrays["epoch_names"])
    band_names = tuple(str(value) for value in arrays["band_names"])
    membership = np.asarray(arrays["condition_membership"], dtype=bool)
    site_valid = np.asarray(arrays["site_valid"], dtype=bool)
    pair_valid = np.asarray(arrays["pair_valid"], dtype=bool)
    labels = tuple(
        f"ITPC {site}:{condition}"
        for condition in condition_names
        for site in site_ids
    ) + tuple(
        f"ISPC {pair}:{condition}"
        for condition in condition_names
        for pair in pair_labels
    )
    paths = []
    for epoch_index, epoch_name in enumerate(epoch_names):
        for band_index, band_name in enumerate(band_names):
            values, lows, highs, counts = _combined_phase_summary_vectors(
                arrays,
                membership,
                site_valid,
                pair_valid,
                epoch_index,
                band_index,
            )
            figure, _ = dependencies.plot_phase_band_summary(
                values,
                lows,
                highs,
                counts,
                labels,
                band_name,
                epoch_name,
                "ITPC / ISPC",
                context,
            )
            path = run_directory / f"{band_name}_{epoch_name}_phase_band_summary.png"
            _save_and_close(figure, path, dependencies)
            paths.append(path)
    return tuple(paths)


def _combined_phase_summary_vectors(
    arrays: Mapping[str, np.ndarray],
    membership: np.ndarray,
    site_valid: np.ndarray,
    pair_valid: np.ndarray,
    epoch_index: int,
    band_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten ITPC then ISPC cache values and contributing trial counts.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Synchrony estimates and confidence bounds with condition first.
    membership : numpy.ndarray
        Boolean ``(trial, condition)`` selected-trial membership.
    site_valid, pair_valid : numpy.ndarray
        Boolean ``(site, trial)`` and ``(pair, trial)`` availability.
    epoch_index, band_index : int
        Zero-based categorical coordinates.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Dimensionless estimate/lower/upper vectors and integer trial counts,
        ordered condition-by-site followed by condition-by-pair.
    """
    values = np.concatenate((
        np.asarray(arrays["itpc_band_mean"])[:, :, epoch_index, band_index].ravel(),
        np.asarray(arrays["ispc_band_mean"])[:, :, epoch_index, band_index].ravel(),
    ))
    lows = np.concatenate((
        np.asarray(arrays["itpc_ci_low"])[:, :, epoch_index, band_index].ravel(),
        np.asarray(arrays["ispc_ci_low"])[:, :, epoch_index, band_index].ravel(),
    ))
    highs = np.concatenate((
        np.asarray(arrays["itpc_ci_high"])[:, :, epoch_index, band_index].ravel(),
        np.asarray(arrays["ispc_ci_high"])[:, :, epoch_index, band_index].ravel(),
    ))
    site_counts = np.stack(
        [np.count_nonzero(membership & valid[:, None], axis=0) for valid in site_valid],
        axis=1,
    )
    pair_counts = np.stack(
        [np.count_nonzero(membership & valid[:, None], axis=0) for valid in pair_valid],
        axis=1,
    )
    return values, lows, highs, np.concatenate((site_counts.ravel(), pair_counts.ravel()))


def _render_plv_figures(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    condition_names: tuple[str, ...],
    site_ids: tuple[str, ...],
    pair_labels: tuple[str, ...],
    context: PlotContext,
    run_directory: Path,
    dependencies: SynchronyValidationDependencies,
) -> tuple[Path, ...]:
    """Render condition PLV distributions and low/high cached trace exemplars.

    Parameters
    ----------
    config : LFPSummaryConfig
        Phase-band definitions in Hz, including excluded intervals.
    arrays : Mapping[str, numpy.ndarray]
        Trial PLV, coverage, and exemplar arrays from the Synchrony cache.
    condition_names, site_ids, pair_labels : tuple[str, ...]
        Ordered labels matching cache condition/site/pair axes.
    context : PlotContext
        Figure provenance and source voltage units.
    run_directory : pathlib.Path
        Existing directory receiving PLV PNG files.
    dependencies : SynchronyValidationDependencies
        Cache-only plot, save, and close seams.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Distribution and available exemplar PNG paths in deterministic order.
    """
    epoch_names = tuple(str(value) for value in arrays["epoch_names"])
    band_names = tuple(str(value) for value in arrays["band_names"])
    membership = np.asarray(arrays["condition_membership"], dtype=bool)
    pair_valid = np.asarray(arrays["pair_valid"], dtype=bool)
    trial_indices = np.asarray(arrays["trial_indices"], dtype=np.int64)
    plv_band = np.asarray(arrays["plv_band_mean"], dtype=float)
    paths: list[Path] = []
    pair_site_indices = _pair_site_indices(arrays, site_ids)
    for pair_index, pair_label in enumerate(pair_labels):
        for condition_index, condition_name in enumerate(condition_names):
            selected = membership[:, condition_index] & pair_valid[pair_index]
            for band_index, band_name in enumerate(band_names):
                counts, fractions = _band_coverage(config, arrays, band_index)
                figure, _ = dependencies.plot_plv_distribution(
                    plv_band[selected, pair_index, :, band_index],
                    counts[selected, pair_index],
                    fractions[selected, pair_index],
                    epoch_names,
                    pair_label.replace("_", "-"),
                    band_name,
                    condition_name,
                    context,
                )
                path = run_directory / (
                    f"{pair_label}_{condition_name}_{band_name}_plv_distribution.png"
                )
                _save_and_close(figure, path, dependencies)
                paths.append(path)
                paths.extend(
                    _render_plv_exemplars(
                        arrays,
                        selected,
                        pair_index,
                        pair_site_indices[pair_index],
                        pair_label,
                        condition_name,
                        band_name,
                        band_index,
                        trial_indices,
                        plv_band,
                        site_ids,
                        context,
                        run_directory,
                        dependencies,
                    )
                )
    return tuple(paths)


def _render_plv_exemplars(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    pair_index: int,
    site_indices: tuple[int, int],
    pair_label: str,
    condition_name: str,
    band_name: str,
    band_index: int,
    trial_indices: np.ndarray,
    plv_band: np.ndarray,
    site_ids: tuple[str, ...],
    context: PlotContext,
    run_directory: Path,
    dependencies: SynchronyValidationDependencies,
) -> tuple[Path, ...]:
    """Render nearest 25th/75th percentile whole-epoch PLV cached trials.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Cached source, band-filtered, Hilbert-phase, and relative-time arrays.
    selected : numpy.ndarray
        Boolean ``(trial,)`` condition/pair selection mask.
    pair_index, band_index : int
        Zero-based pair and band coordinates.
    site_indices : tuple[int, int]
        Ordered cache sites defining the pair.
    pair_label, condition_name, band_name : str
        Filesystem/display labels without physical units.
    trial_indices : numpy.ndarray
        Int64 ``(trial,)`` original table rows.
    plv_band : numpy.ndarray
        Dimensionless ``(trial, pair, epoch, band)`` values.
    site_ids : tuple[str, ...]
        Ordered labels matching the cache site axis.
    context : PlotContext
        Caption provenance and cached trace voltage unit.
    run_directory : pathlib.Path
        Existing output directory receiving zero or two PNGs.
    dependencies : SynchronyValidationDependencies
        Cache-only exemplar plot, save, and close seams.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Empty when no finite selected PLV exists; otherwise low/high PNG paths.
    """
    whole_values = plv_band[:, pair_index, 0, band_index]
    positions = np.flatnonzero(selected & np.isfinite(whole_values))
    if not positions.size:
        return ()
    finite_values = whole_values[positions]
    pooled = float(np.nanmedian(finite_values))
    paths = []
    for percentile, label in ((25.0, "low"), (75.0, "high")):
        target = float(np.nanpercentile(finite_values, percentile))
        position = int(positions[np.argmin(np.abs(finite_values - target))])
        figure, _ = dependencies.plot_plv_exemplar(
            np.asarray(arrays["relative_time_s"], dtype=float),
            np.asarray(arrays["source_trace"])[list(site_indices), position],
            np.asarray(arrays["band_filtered_trace"])[
                list(site_indices), position, band_index
            ],
            np.asarray(arrays["hilbert_phase_rad"])[
                list(site_indices), position, band_index
            ],
            tuple(site_ids[index] for index in site_indices),
            pair_label.replace("_", "-"),
            int(trial_indices[position]),
            label,
            pooled,
            float(whole_values[position]),
            context,
        )
        path = run_directory / (
            f"{pair_label}_{condition_name}_{band_name}_{label}_plv_exemplar.png"
        )
        _save_and_close(figure, path, dependencies)
        paths.append(path)
    return tuple(paths)


def _band_coverage(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    band_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative minimum sample coverage across retained band bins.

    Parameters
    ----------
    config : LFPSummaryConfig
        Frequency-band bounds and open line-noise exclusions in Hz.
    arrays : Mapping[str, numpy.ndarray]
        Frequency coordinates plus sample count/fraction arrays with frequency
        on the final axis.
    band_index : int
        Zero-based phase-band coordinate.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Minimum integer sample counts and dimensionless fractions with axes
        ``(trial, pair, epoch)``.

    Raises
    ------
    ValueError
        If the cached frequency grid retains no sample for the requested band.
    """
    frequency_hz = np.asarray(arrays["frequency_hz"], dtype=float)
    band = config.phase.bands[band_index]
    retained = (frequency_hz >= band.lower_hz) & (frequency_hz <= band.upper_hz)
    for low_hz, high_hz in band.excluded_intervals_hz:
        retained &= ~((frequency_hz >= low_hz) & (frequency_hz <= high_hz))
    if not np.any(retained):
        raise ValueError(f"no cached frequencies retained for band {band.name}")
    counts = np.min(
        np.asarray(arrays["plv_valid_sample_count"])[..., retained],
        axis=-1,
    )
    fractions = np.min(
        np.asarray(arrays["plv_valid_sample_fraction"])[..., retained],
        axis=-1,
    )
    return counts, fractions


def _pair_labels(arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return ordered cache pair labels as filesystem-safe ``site_a_site_b``.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Cache identity arrays ``pair_site_a_ids`` and ``pair_site_b_ids``.

    Returns
    -------
    tuple[str, ...]
        One ordered label per pair axis coordinate.

    Raises
    ------
    ValueError
        If the two identity arrays have different lengths.
    """
    site_a = tuple(str(value) for value in arrays["pair_site_a_ids"])
    site_b = tuple(str(value) for value in arrays["pair_site_b_ids"])
    if len(site_a) != len(site_b):
        raise ValueError("Synchrony cache pair identity axes disagree")
    return tuple(f"{a}_{b}" for a, b in zip(site_a, site_b, strict=True))


def _pair_site_indices(
    arrays: Mapping[str, np.ndarray],
    site_ids: tuple[str, ...],
) -> tuple[tuple[int, int], ...]:
    """Map cached pair ids to zero-based cached site coordinates.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Cache pair identity arrays with shape ``(pair,)``.
    site_ids : tuple[str, ...]
        Ordered stable identities matching the cache site axis.

    Returns
    -------
    tuple[tuple[int, int], ...]
        Ordered zero-based ``(site_a, site_b)`` coordinates per pair.

    Raises
    ------
    ValueError
        If a pair references a site absent from ``site_ids``.
    """
    position = {site_id: index for index, site_id in enumerate(site_ids)}
    try:
        return tuple(
            (position[str(site_a)], position[str(site_b)])
            for site_a, site_b in zip(
                arrays["pair_site_a_ids"],
                arrays["pair_site_b_ids"],
                strict=True,
            )
        )
    except KeyError as error:
        raise ValueError("Synchrony pair references an unknown cached site") from error


def _save_and_close(
    figure: Any,
    path: Path,
    dependencies: SynchronyValidationDependencies,
) -> None:
    """Save one report PNG and always release its figure resources.

    Parameters
    ----------
    figure : object
        Unsaved plotting-backend figure.
    path : pathlib.Path
        Destination PNG path inside the active report directory.
    dependencies : SynchronyValidationDependencies
        Save and close seams.

    Returns
    -------
    None
        The figure is closed even when saving raises.
    """
    try:
        dependencies.save_png(figure, path)
    finally:
        dependencies.close_figure(figure)


def _plot_context(config: LFPSummaryConfig) -> PlotContext:
    """Return immutable seconds/Hz/reference provenance for captions.

    Parameters
    ----------
    config : LFPSummaryConfig
        Session, half-open window, band, notch, reference, and voltage metadata.

    Returns
    -------
    PlotContext
        Cache-plot provenance with seconds, Hz, and source voltage units.
    """
    gamma = next(band for band in config.phase.bands if band.name == "gamma")
    exclusion = (
        gamma.excluded_intervals_hz[0]
        if gamma.excluded_intervals_hz
        else (58.0, 62.0)
    )
    windows = config.analysis_windows
    return PlotContext(
        config.session_id,
        windows.alignment_event,
        {
            "whole": (windows.whole_start_s, windows.whole_stop_s),
            "before": (windows.before_start_s, windows.before_stop_s),
            "after": (windows.after_start_s, windows.after_stop_s),
        },
        config.phase.notch_enabled,
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
) -> dict[str, object]:
    """Build JSON-compatible workload, validity, warning, and projection metrics.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active session and filter settings.
    arrays : Mapping[str, numpy.ndarray]
        Cache membership, validity, exclusion, and instability arrays.
    status : ComponentStatus
        Compatible cache state and warning differences.
    wall_time_s : float
        Measured Synchrony computation duration in seconds.
    peak_memory_bytes, cache_size_bytes, component_size_bytes : int
        Process peak and persisted sizes in bytes.

    Returns
    -------
    dict[str, object]
        JSON-compatible metrics and nine-filter projection without signal data.
    """
    membership = np.asarray(arrays["condition_membership"], dtype=bool)
    filter_membership = np.asarray(arrays["filter_membership"], dtype=bool)
    site_exclusions = np.asarray(arrays["site_exclusion_count"], dtype=np.int64)
    pair_exclusions = np.asarray(arrays["pair_exclusion_count"], dtype=np.int64)
    unstable_count = int(
        np.count_nonzero(arrays["itpc_unstable"])
        + np.count_nonzero(arrays["ispc_unstable"])
    )
    warnings = list(status.differences)
    if unstable_count:
        warnings.append(f"unstable low trial count summaries: {unstable_count}")
    trial_count = int(np.count_nonzero(filter_membership))
    return {
        "session_id": config.session_id,
        "component": "synchrony",
        "cache_status": status.status,
        "wall_time_s": float(wall_time_s),
        "peak_memory_bytes": int(peak_memory_bytes),
        "cache_size_bytes": int(cache_size_bytes),
        "component_size_bytes": int(component_size_bytes),
        "trial_count": trial_count,
        "total_trial_count": int(membership.shape[0]),
        "exclusion_count": int(site_exclusions.sum() + pair_exclusions.sum()),
        "site_exclusion_count": site_exclusions.tolist(),
        "pair_exclusion_count": pair_exclusions.tolist(),
        "unstable_summary_count": unstable_count,
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
    """Write manifest/config/source snapshots plus Markdown and JSON-line log.

    Parameters
    ----------
    run_directory : pathlib.Path
        Existing immutable report directory.
    manifest, report, source_identifiers : Mapping[str, object]
        JSON-compatible cache metadata, metrics, and code identities.
    config : LFPSummaryConfig
        Immutable configuration serialized without numerical source arrays.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths for manifest, configuration, source identities, summary, and log.
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
    paths["log"].write_text(
        json.dumps(dict(report), sort_keys=True) + "\n",
        encoding="ascii",
    )
    return paths


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Write one JSON mapping with deterministic keys and ASCII encoding.

    Parameters
    ----------
    path : pathlib.Path
        Destination file path.
    value : Mapping[str, object]
        JSON-compatible mapping containing no numerical signal arrays.

    Returns
    -------
    None
        The destination is written once with a trailing newline.
    """
    path.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="ascii")


def _markdown_summary(report: Mapping[str, object]) -> str:
    """Render a concise human-readable Synchrony validation summary.

    Parameters
    ----------
    report : Mapping[str, object]
        JSON-compatible timing, memory, count, warning, and projection metrics.

    Returns
    -------
    str
        ASCII Markdown without cached numerical arrays.
    """
    return (
        "# LFP Synchrony validation\n\n"
        "Goal: inspect cache-backed ITPC, ISPC, phase offsets, counts, PLV, "
        "and exemplars before spike-phase analysis.\n\n"
        f"Session: {report['session_id']}\n\n"
        f"Wall time: {report['wall_time_s']} s\n\n"
        f"Peak memory: {report['peak_memory_bytes']} bytes\n\n"
        f"Trials: {report['trial_count']}; exclusions: {report['exclusion_count']}\n\n"
        f"Unstable summaries: {report['unstable_summary_count']}\n\n"
        f"Warnings: {report['warnings']}\n\n"
        f"Nine-filter projection: {report['nine_filter_projection']}\n"
    )


def _directory_size_bytes(directory: Path) -> int:
    """Return recursive regular-file size in bytes, or zero when absent.

    Parameters
    ----------
    directory : pathlib.Path
        Generic cache directory.

    Returns
    -------
    int
        Sum of regular-file sizes in bytes.
    """
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _component_size_bytes(
    directory: Path,
    manifest: Mapping[str, object],
) -> int:
    """Return the Synchrony NPZ size in bytes, or zero when unavailable.

    Parameters
    ----------
    directory : pathlib.Path
        Generic cache directory.
    manifest : Mapping[str, object]
        Compatible manifest whose Synchrony entry may override the filename.

    Returns
    -------
    int
        Component file size in bytes, or zero if the file is absent.
    """
    components = manifest.get("components", {})
    entry = components.get("synchrony", {}) if isinstance(components, Mapping) else {}
    filename = (
        entry.get("file_name", "synchrony.npz")
        if isinstance(entry, Mapping)
        else "synchrony.npz"
    )
    path = directory / str(filename)
    return path.stat().st_size if path.is_file() else 0


def _production_peak_memory_bytes() -> int:
    """Return best-effort current-process peak resident-set size in bytes.

    Returns
    -------
    int
        Nonnegative process peak RSS in bytes, or zero when unavailable.
    """
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return 0
    if value <= 0:
        return 0
    return value if sys.platform == "darwin" else value * 1024


def _production_source_identifiers(config: LFPSummaryConfig) -> dict[str, str]:
    """Return code revision/module identities without numerical source arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active session identity; no raw files are opened.

    Returns
    -------
    dict[str, str]
        Git revision, module paths, and session id for reproduction.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            timeout=5.0,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        commit = "unavailable"
    return {
        "git_commit": commit or "unavailable",
        "runtime_module": str(Path(lfp_summary_runtime.__file__).resolve()),
        "pipeline_module": str(Path(__file__).with_name("lfp_summary_pipeline.py")),
        "validation_module": str(Path(__file__).resolve()),
        "session_id": config.session_id,
    }
