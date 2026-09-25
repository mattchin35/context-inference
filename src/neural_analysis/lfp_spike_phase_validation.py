"""CT026 100-shuffle Spike-phase preview validation without Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.lfp_summary import spike_phase_runtime
from src.neural_analysis.lfp_summary.runtime_common import load_configured_trial_table
from src.neural_analysis.lfp_power_validation import build_ct026_power_config
from src.neural_analysis.lfp_summary.cache import ComponentStatus
from src.neural_analysis.lfp_summary.models import (
    LFPSummaryConfig,
    UnitPopulationConfig,
    canonical_config_json,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary.pipeline import ComponentRunResult, PipelineDependencies
from src.neural_analysis.lfp_summary_plotting import (
    PPCExemplarPanel,
    PlotContext,
    build_summary_figure_filename,
    plot_population_ppc_maps,
    plot_ppc_band_summary,
    plot_ppc_exemplar,
    plot_ppc_exemplar_pair,
    plot_unit_ppc_map,
)


_PREVIEW_SHUFFLE_COUNT = 100
_RUN_LABEL = "lfp_spike_phase_preview_validation"
_REPORT_SCHEMA_VERSION = "spike_phase_preview_report.v1"
_GENERAL_REPORT_SCHEMA_VERSION = "spike_phase_report.v1"
_PRIMARY_CONDITION = "correct_rewarded"
_PRIMARY_EPOCHS = ("before", "after")
_PRIMARY_BANDS = ("theta", "gamma")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class SpikePhasePreviewValidationDependencies:
    """Injected cache-only Spike-phase preview execution and report seams.

    All arrays retain their cache axes and units. The compute seam is limited to
    the Spike-phase component; Power and Synchrony are intentionally absent.
    """

    pipeline_dependencies: PipelineDependencies | object
    compute_spike_phase_component: Callable[[LFPSummaryConfig, object], ComponentRunResult]
    load_manifest: Callable[[Path, LFPSummaryConfig], dict[str, object]]
    assess_component_status: Callable[
        [Path, str, LFPSummaryConfig, Mapping[str, Any]], ComponentStatus
    ]
    load_spike_phase_arrays: Callable[[Path, Mapping[str, Any]], dict[str, np.ndarray]]
    now_utc: Callable[[], str]
    monotonic_seconds: Callable[[], float]
    peak_memory_bytes: Callable[[], int]
    plot_unit_ppc_map: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_population_ppc_maps: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_ppc_band_summary: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_ppc_exemplar: Callable[..., tuple[Any, Mapping[str, Any]]]
    plot_ppc_exemplar_pair: Callable[..., tuple[Any, Mapping[str, Any]]]
    save_png: Callable[[Any, Path], None]
    close_figure: Callable[[Any], None]
    source_identifiers: Callable[[LFPSummaryConfig], dict[str, str]]


@dataclass(frozen=True)
class SpikePhaseFilterBenchmark:
    """One explicit filter run used to project nine-filter cost.

    Attributes
    ----------
    source_label : str
        Nonempty provenance label identifying the representative measurement.
    wall_time_s : float
        Finite nonnegative measured seconds for one filter.
    final_cache_size_bytes : int
        Nonnegative measured final component/cache storage in bytes.
    intermediate_cache_size_bytes : int or None
        Nonnegative measured resumable-work storage in bytes, or ``None`` when
        unavailable. Measured zero is preserved as zero.
    """

    source_label: str
    wall_time_s: float
    final_cache_size_bytes: int
    intermediate_cache_size_bytes: int | None

    def __post_init__(self) -> None:
        """Validate categorical provenance, seconds, and byte units in place."""
        if not self.source_label:
            raise ValueError("benchmark source_label must be nonempty")
        if self.wall_time_s is None:
            raise ValueError("benchmark wall_time_s must be available")
        _optional_nonnegative_float(self.wall_time_s, "benchmark wall_time_s")
        if self.final_cache_size_bytes is None:
            raise ValueError("benchmark final_cache_size_bytes must be available")
        _optional_nonnegative_int(
            self.final_cache_size_bytes,
            "benchmark final_cache_size_bytes",
        )
        _optional_nonnegative_int(
            self.intermediate_cache_size_bytes,
            "benchmark intermediate_cache_size_bytes",
        )


@dataclass(frozen=True)
class SpikePhaseReportMeasurements:
    """Explicit scalar provenance for one cache-backed Spike report.

    ``*_seconds`` fields are finite nonnegative seconds or ``None``;
    ``*_bytes`` fields are nonnegative bytes or ``None``. Worker counts are
    positive requested workers and optional nonnegative planned/active workers.
    ``prepared_phase_cache_state`` is ``"cold"``, ``"warm"``, or ``None``.
    Warning/exclusion tuples contain short categorical text only. No numerical
    arrays, schedules, phases, or spike trains are retained by this record.
    """

    requested_worker_count: int
    phase_preparation_seconds: float | None = None
    ppc_planning_seconds: float | None = None
    grouped_execution_seconds: float | None = None
    component_total_seconds: float | None = None
    planned_worker_count: int | None = None
    active_worker_count: int | None = None
    prepared_phase_cache_state: str | None = None
    peak_process_rss_bytes: int | None = None
    peak_process_tree_rss_bytes: int | None = None
    peak_process_tree_pss_bytes: int | None = None
    memory_provenance: str | None = None
    final_component_size_bytes: int | None = None
    final_cache_size_bytes: int | None = None
    intermediate_work_size_bytes: int | None = None
    warnings: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()
    benchmark: SpikePhaseFilterBenchmark | None = None

    def __post_init__(self) -> None:
        """Validate scalar axes/units without replacing unavailable values."""
        if (
            not isinstance(self.requested_worker_count, int)
            or isinstance(self.requested_worker_count, bool)
            or self.requested_worker_count < 1
        ):
            raise ValueError("requested_worker_count must be a positive integer")
        for name in (
            "phase_preparation_seconds",
            "ppc_planning_seconds",
            "grouped_execution_seconds",
            "component_total_seconds",
        ):
            _optional_nonnegative_float(getattr(self, name), name)
        for name in (
            "planned_worker_count",
            "active_worker_count",
            "peak_process_rss_bytes",
            "peak_process_tree_rss_bytes",
            "peak_process_tree_pss_bytes",
            "final_component_size_bytes",
            "final_cache_size_bytes",
            "intermediate_work_size_bytes",
        ):
            _optional_nonnegative_int(getattr(self, name), name)
        if self.prepared_phase_cache_state not in {None, "cold", "warm"}:
            raise ValueError("prepared_phase_cache_state must be cold, warm, or None")
        if self.memory_provenance is not None and not self.memory_provenance:
            raise ValueError("memory_provenance must be nonempty or None")
        if any(not isinstance(value, str) or not value for value in self.warnings):
            raise ValueError("warnings must contain nonempty strings")
        if any(not isinstance(value, str) or not value for value in self.exclusions):
            raise ValueError("exclusions must contain nonempty strings")
        if self.benchmark is not None and not isinstance(
            self.benchmark,
            SpikePhaseFilterBenchmark,
        ):
            raise ValueError("benchmark must be SpikePhaseFilterBenchmark or None")


@dataclass(frozen=True)
class SpikePhasePreviewValidationResult:
    """Immutable cache-backed preview report metadata.

    Paths are direct children of ``run_directory``. The report mapping contains
    categorical counts only and never exposes LFP phase or spike-time arrays.
    """

    component: str
    run_directory: Path
    manifest_snapshot_path: Path
    configuration_snapshot_path: Path
    summary_path: Path
    trial_count: int
    unit_count: int
    reliable_cell_count: int
    png_paths: tuple[Path, ...]
    report: dict[str, object]
    report_path: Path
    log_path: Path
    source_identifiers_path: Path
    deferred_cleanup: Callable[[], None] | None


def build_ct026_spike_phase_preview_config(
    session_path: Path,
    unit_population: UnitPopulationConfig,
) -> LFPSummaryConfig:
    """Build the fixed CT026 preview configuration for an active unit population.

    Parameters
    ----------
    session_path : pathlib.Path
        CT026 session root. This function does not open recordings or sorter data.
    unit_population : UnitPopulationConfig
        Active probe-qualified units, selected channels, and quality settings.

    Returns
    -------
    LFPSummaryConfig
        Power/Synchrony-compatible CT026 settings with the supplied population
        unchanged and exactly 100 same-condition trial shuffles for Spike phase.
    """
    base = build_ct026_power_config(Path(session_path))
    preview_ppc = replace(base.ppc, shuffle_count=_PREVIEW_SHUFFLE_COUNT)
    config = replace(base, unit_population=unit_population, ppc=preview_ppc)
    validate_lfp_summary_config(config)
    return config


def build_ct026_default_active_population(
    session_path: Path,
    cluster_metadata_loader: Callable[[Path], pd.DataFrame],
    channel_metadata_loader: Callable[[Path], pd.DataFrame],
) -> UnitPopulationConfig:
    """Load the fixed CT026 ProbeB active population from injected metadata.

    Parameters
    ----------
    session_path : pathlib.Path
        CT026 root used only to form the ProbeB sorter and aligned-spike paths.
    cluster_metadata_loader, channel_metadata_loader : callable
        Loaders accepting the sorter path and returning cluster/channel tables.

    Returns
    -------
    UnitPopulationConfig
        ProbeB good/mua units on good inside-brain channels, ordered by cluster.
    """
    return build_ct026_active_population(
        session_path,
        "ProbeB",
        cluster_metadata_loader,
        channel_metadata_loader,
    )


def build_ct026_active_population(
    session_path: Path,
    probe_label: str,
    cluster_metadata_loader: Callable[[Path], pd.DataFrame],
    channel_metadata_loader: Callable[[Path], pd.DataFrame],
) -> UnitPopulationConfig:
    """Load one explicit CT026 ProbeA or ProbeB active population.

    Parameters
    ----------
    session_path : pathlib.Path
        CT026 root used only to form sorter and aligned-spike paths.
    probe_label : str
        Exact categorical value ``"ProbeA"`` or ``"ProbeB"``. Combined
        populations are rejected.
    cluster_metadata_loader, channel_metadata_loader : callable
        Loaders accepting the selected sorter path and returning cluster and
        channel tables. Cluster ids/channels are categorical integer values.

    Returns
    -------
    UnitPopulationConfig
        Good/MUA units on good inside-brain channels for exactly one probe,
        ordered by cluster id with probe-qualified stable identities.
    """
    if probe_label not in {"ProbeA", "ProbeB"}:
        raise ValueError("probe_label must be ProbeA or ProbeB")
    root = Path(session_path)
    sorter = (
        root
        / "ephys"
        / "derived"
        / f"Record_Node_101_Neuropix-PXI-110.{probe_label}"
        / "kilosort4"
    )
    clusters = cluster_metadata_loader(sorter)
    channels = channel_metadata_loader(sorter)
    required_clusters = {"cluster_id", "ch", "group"}
    if not required_clusters.issubset(clusters) or "inside_brain" not in channels:
        raise ValueError(
            f"CT026 {probe_label} metadata is missing required quality columns"
        )
    if {"channel_id", "label"}.issubset(channels):
        channel_numbers = pd.to_numeric(
            channels["channel_id"].astype(str).str.replace("CH", "", regex=False),
            errors="coerce",
        )
        channel_labels = channels["label"]
    elif {"channel", "channel_quality"}.issubset(channels):
        channel_numbers = pd.to_numeric(channels["channel"], errors="coerce")
        channel_labels = channels["channel_quality"]
    else:
        raise ValueError(
            f"CT026 {probe_label} metadata lacks channel id or label columns"
        )
    good_channels = channels.loc[
        channel_labels.astype(str).str.lower().eq("good")
        & channels["inside_brain"].astype(bool)
        & channel_numbers.notna()
    ].copy()
    good_channels = channel_numbers.loc[good_channels.index].astype(int)
    selected_channels = tuple(sorted(good_channels.unique().tolist()))
    selected = clusters.loc[
        clusters["ch"].isin(selected_channels)
        & clusters["group"].astype(str).str.lower().isin(("good", "mua"))
    ].sort_values("cluster_id")
    unit_ids = tuple(
        f"{probe_label}:{int(value)}" for value in selected["cluster_id"]
    )
    aligned_name = f"probe{probe_label[-1]}_sync.npz"
    return UnitPopulationConfig(
        f"CT026 {probe_label} active", probe_label, sorter,
        root / "ephys/aligned/aligned_open_ephys" / aligned_name,
        selected_channels,
        (("channel_quality", "good"), ("inside_brain", "true"), ("unit_quality", "good,mua")),
        unit_ids,
    )


def make_production_spike_phase_preview_dependencies(
) -> SpikePhasePreviewValidationDependencies:
    """Bind the production Spike cache boundary, public plots, and clocks.

    Returns
    -------
    SpikePhasePreviewValidationDependencies
        Production seams. The numerical Spike runtime is required separately;
        this factory never calculates PPC or opens raw data for plotting.
    """
    factory = getattr(spike_phase_runtime, "make_spike_phase_pipeline_dependencies", None)
    if factory is None:
        pipeline: object = _unavailable_spike_pipeline()
    else:
        pipeline = factory(trial_table_loader=load_configured_trial_table)

    def load_manifest(directory: Path, config: LFPSummaryConfig) -> dict[str, object]:
        """Load the manifest matching the immutable active configuration."""
        from src.neural_analysis.lfp_summary.cache import load_or_initialize_manifest
        return load_or_initialize_manifest(directory, config)

    def load_arrays(directory: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Load the validated Spike NPZ without pickle or raw recordings."""
        from src.neural_analysis.lfp_summary.cache import load_component_arrays
        return load_component_arrays(directory / "spike_phase.npz", manifest, "spike_phase")

    return SpikePhasePreviewValidationDependencies(
        pipeline, _compute_spike, load_manifest, _assess_spike, load_arrays,
        lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        time.monotonic, _production_peak_memory_bytes, _plot_cached_unit_map,
        _plot_cached_population_map, _plot_cached_band_summary,
        _plot_cached_exemplar, _plot_cached_exemplar_pair, _save_png, plt.close,
        _source_identifiers,
    )


def run_spike_phase_preview_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
    *,
    report_measurements: SpikePhaseReportMeasurements | None = None,
) -> SpikePhasePreviewValidationResult:
    """Compute only a compatible 100-shuffle Spike-phase preview and report it.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable active configuration with exactly 100 PPC shuffles.
    run_parent : pathlib.Path
        Parent for one new immutable cache-backed report directory.
    dependencies : SpikePhasePreviewValidationDependencies
        Injected Spike-only compute and cache seams. Plots, if added, must load
        only cached arrays rather than triggering numerical recomputation.
    report_measurements : SpikePhaseReportMeasurements or None, default=None
        Optional explicit stage/worker/memory/cache provenance. Missing fields
        remain unavailable except component-total seconds, current-process RSS,
        and final persisted sizes directly measured by this function.

    Returns
    -------
    SpikePhasePreviewValidationResult
        Atomic report paths and scalar trial/unit/reliability counts.

    Raises
    ------
    ValueError
        If the configuration is not the explicit 100-shuffle preview.
    RuntimeError
        If Spike-phase calculation fails or its cache is incompatible.
    FileExistsError
        If the immutable report directory already exists.
    """
    _validate_preview_config(config)
    run_directory = _planned_run_directory(config, Path(run_parent), dependencies.now_utc())
    clock = getattr(dependencies, "monotonic_seconds", time.monotonic)
    start_s = float(clock())
    result = dependencies.compute_spike_phase_component(config, dependencies.pipeline_dependencies)
    wall_time_s = float(clock()) - start_s
    if result.state != "complete":
        raise RuntimeError(result.error or "Spike-phase component failed")
    manifest, arrays = _load_compatible_spike_phase(config, dependencies)
    return _write_report(
        config, Path(run_parent), run_directory, manifest, arrays, wall_time_s,
        int(getattr(dependencies, "peak_memory_bytes", lambda: 0)()), dependencies,
        report_measurements=report_measurements,
        deferred_cleanup=getattr(result, "deferred_cleanup", None),
        schema_version=_REPORT_SCHEMA_VERSION,
        run_kind="preview",
    )


def render_cached_spike_phase_preview_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
    *,
    spike_phase_wall_time_s: float,
    spike_phase_peak_memory_bytes: int,
    report_measurements: SpikePhaseReportMeasurements | None = None,
) -> SpikePhasePreviewValidationResult:
    """Render a compatible preview cache without dispatching numerical compute.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable 100-shuffle preview configuration.
    run_parent : pathlib.Path
        Parent for one new immutable cache-backed report directory.
    dependencies : SpikePhasePreviewValidationDependencies
        Cache loading and public plotting seams; the compute seam is not called.
    spike_phase_wall_time_s, spike_phase_peak_memory_bytes : float, int
        Previously measured compute seconds and peak resident bytes.
    report_measurements : SpikePhaseReportMeasurements or None, default=None
        Optional richer provenance. The cached path never calls compute or
        fabricates unavailable planner/process-tree values.

    Returns
    -------
    SpikePhasePreviewValidationResult
        New report paths using only compatible cached arrays.
    """
    _validate_preview_config(config)
    if not np.isfinite(spike_phase_wall_time_s) or spike_phase_wall_time_s < 0:
        raise ValueError("spike_phase_wall_time_s must be finite and nonnegative")
    if spike_phase_peak_memory_bytes < 0:
        raise ValueError("spike_phase_peak_memory_bytes must be nonnegative")
    run_directory = _planned_run_directory(config, Path(run_parent), dependencies.now_utc())
    manifest, arrays = _load_compatible_spike_phase(config, dependencies)
    return _write_report(
        config, Path(run_parent), run_directory, manifest, arrays,
        float(spike_phase_wall_time_s), int(spike_phase_peak_memory_bytes), dependencies,
        report_measurements=report_measurements,
        deferred_cleanup=None,
        schema_version=_REPORT_SCHEMA_VERSION,
        run_kind="preview",
    )


def render_cached_spike_phase_report(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
    *,
    run_kind: str,
    component_wall_time_s: float,
    component_peak_memory_bytes: int,
    report_measurements: SpikePhaseReportMeasurements | None = None,
    deferred_cleanup: Callable[[], None] | None = None,
) -> SpikePhasePreviewValidationResult:
    """Publish a cache-only launcher report for preview or final shuffles.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable active-population configuration using exactly 100 preview or
        1,000 final shuffles. Cached numerical axes/units remain unchanged.
    run_parent : pathlib.Path
        Parent for one immutable timestamped report directory.
    dependencies : SpikePhasePreviewValidationDependencies
        Cache compatibility/loading, bounded plotting, clock, and writer seams.
        Its compute callable is never invoked.
    run_kind : str
        Exact categorical value ``"preview"`` or ``"final"`` matching 100 or
        1,000 configured shuffles respectively.
    component_wall_time_s : float
        Finite nonnegative complete component duration in seconds.
    component_peak_memory_bytes : int
        Nonnegative current-process peak RSS bytes; zero means unavailable.
    report_measurements : SpikePhaseReportMeasurements or None
        Optional detailed scalar timing/worker/memory/storage measurements.
    deferred_cleanup : callable or None
        Exact in-memory cleanup action returned only after report publication;
        this function never invokes it.

    Returns
    -------
    SpikePhasePreviewValidationResult
        Validated immutable report paths using schema
        ``spike_phase_report.v1`` and the actual configured shuffle count.
    """
    validate_lfp_summary_config(config)
    if config.unit_population is None:
        raise ValueError("Spike-phase report requires an active unit population")
    expected_shuffles = {"preview": 100, "final": 1000}
    if run_kind not in expected_shuffles:
        raise ValueError("run_kind must be preview or final")
    if config.ppc.shuffle_count != expected_shuffles[run_kind]:
        raise ValueError("run_kind does not match the configured shuffle count")
    _optional_nonnegative_float(component_wall_time_s, "component_wall_time_s")
    _optional_nonnegative_int(
        component_peak_memory_bytes,
        "component_peak_memory_bytes",
    )
    timestamp = dependencies.now_utc()
    label = f"lfp_spike_phase_{run_kind}_report"
    run_directory = _planned_run_directory(
        config,
        Path(run_parent),
        timestamp,
        run_label=label,
    )
    manifest, arrays = _load_compatible_spike_phase(config, dependencies)
    return _write_report(
        config,
        Path(run_parent),
        run_directory,
        manifest,
        arrays,
        float(component_wall_time_s),
        int(component_peak_memory_bytes),
        dependencies,
        report_measurements=report_measurements,
        deferred_cleanup=deferred_cleanup,
        schema_version=_GENERAL_REPORT_SCHEMA_VERSION,
        run_kind=run_kind,
    )


def _validate_preview_config(config: LFPSummaryConfig) -> None:
    """Reject final-count or unitless configurations before any compute dispatch.

    Parameters
    ----------
    config : LFPSummaryConfig
        Candidate immutable summary configuration.

    Returns
    -------
    None
        Validation does not alter configuration values or source identities.
    """
    validate_lfp_summary_config(config)
    if config.unit_population is None:
        raise ValueError("Spike-phase preview requires an active unit population")
    if config.ppc.shuffle_count != _PREVIEW_SHUFFLE_COUNT:
        raise ValueError("Spike-phase preview requires exactly 100 shuffles")


def _planned_run_directory(
    config: LFPSummaryConfig,
    run_parent: Path,
    timestamp: str,
    *,
    run_label: str = _RUN_LABEL,
) -> Path:
    """Return an unused immutable report path without creating it.

    Parameters
    ----------
    config : LFPSummaryConfig
        Session identity used in the report directory name.
    run_parent : pathlib.Path
        Existing or future report parent.
    timestamp : str
        Filesystem-safe UTC identifier supplied by the dependency seam.

    Returns
    -------
    pathlib.Path
        Uncreated directory beneath ``run_parent``.
    """
    path = run_parent / f"{config.session_id}_{run_label}_{timestamp}"
    if path.exists():
        raise FileExistsError(f"validation run already exists: {path}")
    return path


def _load_compatible_spike_phase(
    config: LFPSummaryConfig,
    dependencies: SpikePhasePreviewValidationDependencies,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Load only one compatible Spike-phase cache component.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable cache location and Spike-phase fingerprint inputs.
    dependencies : SpikePhasePreviewValidationDependencies
        Manifest, compatibility, and safe NPZ-array loading seams.

    Returns
    -------
    tuple[dict[str, object], dict[str, numpy.ndarray]]
        Manifest and validated cache arrays without array transformations.
    """
    manifest = dependencies.load_manifest(config.output_directory, config)
    status = dependencies.assess_component_status(
        config.output_directory, "spike_phase", config, manifest
    )
    if status.status != "compatible":
        detail = "; ".join(status.differences)
        raise RuntimeError(f"Spike-phase cache is not compatible: {status.status} {detail}".strip())
    return manifest, dependencies.load_spike_phase_arrays(config.output_directory, manifest)


def _write_report(
    config: LFPSummaryConfig,
    run_parent: Path,
    run_directory: Path,
    manifest: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    wall_time_s: float,
    peak_memory_bytes: int,
    dependencies: SpikePhasePreviewValidationDependencies,
    *,
    report_measurements: SpikePhaseReportMeasurements | None,
    deferred_cleanup: Callable[[], None] | None,
    schema_version: str,
    run_kind: str,
) -> SpikePhasePreviewValidationResult:
    """Atomically write scalar report artifacts from already-cached arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable preview configuration.
    run_parent, run_directory : pathlib.Path
        Parent and prevalidated uncreated final report directories.
    manifest : Mapping[str, object]
        Compatible component metadata without raw numerical values.
    arrays : Mapping[str, numpy.ndarray]
        Validated Spike-phase cache arrays with their documented axes unchanged.
    wall_time_s : float
        Finite nonnegative component-total seconds measured or supplied by the
        caller.
    peak_memory_bytes : int
        Legacy current-process peak RSS bytes; zero means the measurement was
        unavailable and is converted to ``None`` rather than reported as zero.
    dependencies : SpikePhasePreviewValidationDependencies
        Cache-only plot, clock, save, close, and source-identity seams.
    report_measurements : SpikePhaseReportMeasurements or None
        Explicit optional stage/worker/memory/cache/projection provenance.
    deferred_cleanup : callable or None
        Exact committed PPC-work cleanup. It is never called here and is
        returned only after staged and published artifacts validate.

    Returns
    -------
    SpikePhasePreviewValidationResult
        Final immutable report paths and scalar summaries.
    """
    measurements = _resolved_report_measurements(
        config,
        manifest,
        wall_time_s,
        peak_memory_bytes,
        report_measurements,
    )
    run_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{run_directory.name}.staging-", dir=run_parent) as temp:
        staging = Path(temp)
        report_start_s = float(dependencies.monotonic_seconds())
        png_paths, rendered, unavailable = _render_cached_pngs(
            config,
            arrays,
            staging,
            dependencies,
        )
        report_seconds = float(dependencies.monotonic_seconds()) - report_start_s
        if not np.isfinite(report_seconds) or report_seconds < 0.0:
            raise ValueError("report clock must produce finite nonnegative seconds")
        report = _scalar_report(
            config,
            arrays,
            measurements,
            report_seconds,
            rendered,
            unavailable,
            cleanup_ready=deferred_cleanup is not None,
            schema_version=schema_version,
            run_kind=run_kind,
        )
        staged_paths = _write_report_artifacts(
            staging,
            manifest,
            config,
            report,
            dependencies.source_identifiers(config),
        )
        _validate_staged_report(staging, png_paths)
        _publish_staged_report(staging, run_directory)
        published_pngs = tuple(run_directory / path.name for path in png_paths)
        try:
            _validate_staged_report(run_directory, published_pngs)
        except Exception:
            # Publication is part of this transaction. Restore the known
            # temporary path so its context manager removes the failed report.
            if run_directory.exists() and not staging.exists():
                run_directory.replace(staging)
            raise
    final_paths = {
        name: run_directory / path.name for name, path in staged_paths.items()
    }
    return SpikePhasePreviewValidationResult(
        component="spike_phase",
        run_directory=run_directory,
        manifest_snapshot_path=final_paths["manifest"],
        configuration_snapshot_path=final_paths["configuration"],
        summary_path=final_paths["summary"],
        trial_count=int(report["trial_count"]),
        unit_count=int(report["unit_count"]),
        reliable_cell_count=int(report["reliable_cell_count"]),
        png_paths=published_pngs,
        report=report,
        report_path=final_paths["report"],
        log_path=final_paths["log"],
        source_identifiers_path=final_paths["source_identifiers"],
        deferred_cleanup=deferred_cleanup,
    )


def _scalar_report(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    measurements: SpikePhaseReportMeasurements,
    report_seconds: float,
    rendered_selections: tuple[dict[str, object], ...],
    unavailable_selections: tuple[dict[str, object], ...],
    *,
    cleanup_ready: bool,
    schema_version: str,
    run_kind: str,
) -> dict[str, object]:
    """Build one JSON-safe detailed report without retaining numerical arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable session, population, filter, worker, and cache identity.
    arrays : Mapping[str, numpy.ndarray]
        Validated Spike-phase arrays. Only categorical axis lengths and scalar
        Boolean/count reductions are retained.
    measurements : SpikePhaseReportMeasurements
        Validated seconds, worker counts, memory bytes, cache bytes, warnings,
        exclusions, and representative benchmark.
    report_seconds : float
        Finite nonnegative cache-only figure/report generation seconds.
    rendered_selections, unavailable_selections : tuple[dict[str, object], ...]
        JSON-safe categorical selection/file records containing no arrays.
    cleanup_ready : bool
        Whether an exact deferred action may be handed to the launcher after
        publication. It has no effect on report artifacts.

    Returns
    -------
    dict[str, object]
        JSON-ready scalar/categorical report using ``None`` for unavailable
        measurements and preserving measured numeric zero.
    """
    benchmark = measurements.benchmark
    projection = {
        "filter_count": 9,
        "benchmark_source": benchmark.source_label if benchmark else None,
        "benchmark_wall_time_s": benchmark.wall_time_s if benchmark else None,
        "benchmark_final_cache_size_bytes": (
            benchmark.final_cache_size_bytes if benchmark else None
        ),
        "benchmark_intermediate_cache_size_bytes": (
            benchmark.intermediate_cache_size_bytes if benchmark else None
        ),
        "projected_wall_time_s": benchmark.wall_time_s * 9.0 if benchmark else None,
        "projected_final_cache_size_bytes": (
            benchmark.final_cache_size_bytes * 9 if benchmark else None
        ),
        "projected_intermediate_cache_size_bytes": (
            None
            if benchmark is None or benchmark.intermediate_cache_size_bytes is None
            else benchmark.intermediate_cache_size_bytes * 9
        ),
    }
    membership = np.asarray(arrays["filter_membership"], dtype=bool)
    population = config.unit_population
    assert population is not None
    return {
        "schema_version": schema_version,
        "run_kind": run_kind,
        "session_id": config.session_id,
        "component": "spike_phase",
        "population_label": population.label,
        "probe_label": population.probe_label,
        "trial_count": int(np.asarray(arrays["trial_indices"]).size),
        "selected_trial_count": int(np.count_nonzero(membership)),
        "unit_count": int(np.asarray(arrays["unit_ids"]).size),
        "computable_cell_count": int(
            np.count_nonzero(np.asarray(arrays["computable"], dtype=bool))
        ),
        "reliable_cell_count": int(
            np.count_nonzero(np.asarray(arrays["reliable"], dtype=bool))
        ),
        "null_eligible_cell_count": int(
            np.count_nonzero(np.asarray(arrays["null_eligible"], dtype=bool))
        ),
        "significant_cell_count": int(
            np.count_nonzero(np.asarray(arrays["significant"], dtype=bool))
        ),
        "spike_count": _unique_cached_spike_count(arrays),
        "shuffle_count": int(config.ppc.shuffle_count),
        "wall_time_s": measurements.component_total_seconds,
        "peak_memory_bytes": measurements.peak_process_rss_bytes,
        "cache_size_bytes": measurements.final_cache_size_bytes,
        "component_size_bytes": measurements.final_component_size_bytes,
        "timing_seconds": {
            "phase_preparation": measurements.phase_preparation_seconds,
            "ppc_planning": measurements.ppc_planning_seconds,
            "grouped_execution": measurements.grouped_execution_seconds,
            "component_total": measurements.component_total_seconds,
            "report": float(report_seconds),
        },
        "workers": {
            "requested": measurements.requested_worker_count,
            "planned": measurements.planned_worker_count,
            "active": measurements.active_worker_count,
        },
        "prepared_phase_cache_state": measurements.prepared_phase_cache_state,
        "memory_bytes": {
            "process_rss": measurements.peak_process_rss_bytes,
            "process_tree_rss": measurements.peak_process_tree_rss_bytes,
            "process_tree_pss": measurements.peak_process_tree_pss_bytes,
            "provenance": measurements.memory_provenance,
        },
        "sizes_bytes": {
            "final_component": measurements.final_component_size_bytes,
            "final_cache": measurements.final_cache_size_bytes,
            "intermediate_work": measurements.intermediate_work_size_bytes,
        },
        "warnings": list(measurements.warnings),
        "exclusions": list(measurements.exclusions),
        "rendered_selections": list(rendered_selections),
        "unavailable_selections": list(unavailable_selections),
        "cleanup_ready": bool(cleanup_ready),
        "nine_filter_projection": projection,
    }


def _unique_cached_spike_count(arrays: Mapping[str, np.ndarray]) -> int:
    """Count packed unit/trial spikes once, with a legacy metric-cell fallback."""
    if "relative_spike_times_s" in arrays:
        return int(np.asarray(arrays["relative_spike_times_s"]).size)
    return int(np.asarray(arrays["spike_count"], dtype=np.int64).sum())


def _render_cached_pngs(
    config: LFPSummaryConfig,
    arrays: Mapping[str, np.ndarray],
    directory: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
) -> tuple[
    tuple[Path, ...],
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
]:
    """Render the fixed bounded PPC report plan from cached arrays only.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable plot provenance without raw recording handles.
    arrays : Mapping[str, numpy.ndarray]
        Reloaded compatible Spike-phase cache arrays, passed unchanged to seams.
    directory : pathlib.Path
        Existing staging directory receiving PNG files.
    dependencies : SpikePhasePreviewValidationDependencies
        Public PPC plotting, saving, and closing seams.

    Returns
    -------
    tuple[tuple[pathlib.Path, ...], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]
        Staged paths, rendered selection records, and scientifically unavailable
        exemplar records. For three sites, paths contain six unit maps, six
        population maps, three band summaries, and at most 12 paired exemplars.
    """
    condition_names = tuple(np.asarray(arrays["condition_names"]).astype(str))
    site_ids = tuple(np.asarray(arrays["site_ids"]).astype(str))
    epoch_names = tuple(np.asarray(arrays["epoch_names"]).astype(str))
    band_names = tuple(np.asarray(arrays["band_names"]).astype(str))
    condition_index = _required_label_index(
        condition_names,
        _PRIMARY_CONDITION,
        "condition",
    )
    epoch_indices = tuple(
        _required_label_index(epoch_names, name, "epoch") for name in _PRIMARY_EPOCHS
    )
    band_indices = tuple(
        _required_label_index(band_names, name, "band") for name in _PRIMARY_BANDS
    )
    paths: list[Path] = []
    rendered: list[dict[str, object]] = []
    unavailable: list[dict[str, object]] = []
    for site_index, site_id in enumerate(site_ids):
        for epoch_index in epoch_indices:
            epoch_name = epoch_names[epoch_index]
            unit_path = directory / _report_figure_filename(
                config,
                "unit_ppc_map",
                site_id,
                _PRIMARY_CONDITION,
                epoch_name,
                "reference_order",
            )
            figure, _ = dependencies.plot_unit_ppc_map(
                arrays,
                config,
                condition_index=condition_index,
                site_index=site_index,
                epoch_index=epoch_index,
            )
            _save_and_close_report_figure(figure, unit_path, dependencies)
            paths.append(unit_path)
            rendered.append(
                _selection_record(
                    "unit_ppc_map", site_id, _PRIMARY_CONDITION, epoch_name, None,
                    unit_path.name,
                )
            )

            population_path = directory / _report_figure_filename(
                config,
                "population_ppc_map",
                site_id,
                "all_conditions",
                epoch_name,
                "eligible_prevalence",
            )
            figure, _ = dependencies.plot_population_ppc_maps(
                arrays,
                config,
                site_index=site_index,
                epoch_index=epoch_index,
            )
            _save_and_close_report_figure(figure, population_path, dependencies)
            paths.append(population_path)
            rendered.append(
                _selection_record(
                    "population_ppc_map", site_id, "all_conditions", epoch_name,
                    None, population_path.name,
                )
            )

        band_path = directory / _report_figure_filename(
            config,
            "ppc_band_summary",
            site_id,
            "all_conditions",
            "all_epochs",
            "reliable_units",
        )
        figure, _ = dependencies.plot_ppc_band_summary(
            arrays,
            config,
            site_index=site_index,
        )
        _save_and_close_report_figure(figure, band_path, dependencies)
        paths.append(band_path)
        rendered.append(
            _selection_record(
                "ppc_band_summary", site_id, "all_conditions", "all_epochs",
                None, band_path.name,
            )
        )

        for epoch_index in epoch_indices:
            epoch_name = epoch_names[epoch_index]
            for band_index in band_indices:
                band_name = band_names[band_index]
                selection = _selection_record(
                    "ppc_exemplar_pair",
                    site_id,
                    _PRIMARY_CONDITION,
                    epoch_name,
                    band_name,
                    None,
                )
                reason = _paired_exemplar_unavailable_reason(
                    arrays,
                    condition_index,
                    site_index,
                    epoch_index,
                    band_index,
                )
                if reason is not None:
                    unavailable.append({**selection, "reason": reason})
                    continue
                exemplar_path = directory / _report_figure_filename(
                    config,
                    "ppc_exemplar_pair",
                    site_id,
                    _PRIMARY_CONDITION,
                    epoch_name,
                    f"{band_name}_5th_95th_percentiles",
                )
                figure, _ = dependencies.plot_ppc_exemplar_pair(
                    arrays,
                    config,
                    condition_index=condition_index,
                    site_index=site_index,
                    epoch_index=epoch_index,
                    band_index=band_index,
                )
                _save_and_close_report_figure(figure, exemplar_path, dependencies)
                paths.append(exemplar_path)
                rendered.append({**selection, "file_name": exemplar_path.name})
    return tuple(paths), tuple(rendered), tuple(unavailable)


def _required_label_index(labels: tuple[str, ...], required: str, axis: str) -> int:
    """Return one required categorical position without changing axis order.

    Parameters
    ----------
    labels : tuple[str, ...]
        Ordered unique categorical coordinate.
    required : str
        Required nonempty coordinate value.
    axis : str
        Human-readable axis name used in failures.

    Returns
    -------
    int
        Zero-based categorical position without physical units.
    """
    if required not in labels:
        raise ValueError(f"Spike report requires {axis} {required!r}")
    return labels.index(required)


def _report_figure_filename(
    config: LFPSummaryConfig,
    component: str,
    site_id: str,
    condition_name: str,
    epoch_name: str,
    normalization: str,
) -> str:
    """Return one collision-safe basename for the fixed report selection.

    Parameters
    ----------
    config : LFPSummaryConfig
        Session and active choice/context filter labels.
    component, site_id, condition_name, epoch_name, normalization : str
        Categorical plot identity; no field has a physical unit.

    Returns
    -------
    str
        Deterministic ASCII PNG basename without a directory component.
    """
    return build_summary_figure_filename(
        config.session_id,
        component,
        site_id,
        condition_name,
        config.trial_filter.choice,
        config.trial_filter.context,
        epoch_name,
        normalization,
    )


def _selection_record(
    kind: str,
    site_id: str,
    condition_name: str,
    epoch_name: str,
    band_name: str | None,
    file_name: str | None,
) -> dict[str, object]:
    """Build one JSON-safe categorical figure selection record.

    Parameters
    ----------
    kind, site_id, condition_name, epoch_name : str
        Nonempty categorical report coordinates.
    band_name, file_name : str or None
        Optional band and published PNG basename.

    Returns
    -------
    dict[str, object]
        Scalar categorical record containing no cached numerical values.
    """
    return {
        "kind": kind,
        "site_id": site_id,
        "condition_name": condition_name,
        "epoch_name": epoch_name,
        "band_name": band_name,
        "file_name": file_name,
    }


def _paired_exemplar_unavailable_reason(
    arrays: Mapping[str, np.ndarray],
    condition_index: int,
    site_index: int,
    epoch_index: int,
    band_index: int,
) -> str | None:
    """Return why one exact low/high cache selection cannot be plotted.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Validated cache arrays with categorical unit/trial identities.
    condition_index, site_index, epoch_index, band_index : int
        Zero-based categorical positions without physical units.

    Returns
    -------
    str or None
        Stable categorical reason, or ``None`` when both unit identities and
        unit-specific nonnegative trial-table rows are available.
    """
    index = (condition_index, site_index, epoch_index, band_index)
    low_unit = str(np.asarray(arrays["selected_low_unit_ids"])[index])
    high_unit = str(np.asarray(arrays["selected_high_unit_ids"])[index])
    low_trial = int(np.asarray(arrays["illustrative_low_trial_indices"])[index])
    high_trial = int(np.asarray(arrays["illustrative_high_trial_indices"])[index])
    unit_ids = set(np.asarray(arrays["unit_ids"]).astype(str))
    trial_ids = set(int(value) for value in np.asarray(arrays["trial_indices"]))
    if not low_unit or not high_unit:
        return "fewer_than_two_finite_band_ppc_units"
    if low_unit not in unit_ids or high_unit not in unit_ids:
        return "selected_unit_missing_from_cache_axis"
    if low_trial not in trial_ids or high_trial not in trial_ids:
        return "unit_specific_illustrative_trial_unavailable"
    return None


def _save_and_close_report_figure(
    figure: Any,
    path: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
) -> None:
    """Save one cache-backed figure and always release its Matplotlib memory.

    Parameters
    ----------
    figure : matplotlib.figure.Figure or injected opaque object
        One bounded report figure containing no full component copy.
    path : pathlib.Path
        PNG path inside the staging directory.
    dependencies : SpikePhasePreviewValidationDependencies
        Injected PNG writer and figure closer.

    Returns
    -------
    None
        The figure is closed even when saving raises.
    """
    try:
        dependencies.save_png(figure, path)
    finally:
        dependencies.close_figure(figure)


def _plot_context(config: LFPSummaryConfig, arrays: Mapping[str, np.ndarray]) -> PlotContext:
    """Build cache-plot provenance without opening raw data.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable alignment, window, notch, and band settings.
    arrays : Mapping[str, numpy.ndarray]
        Cache arrays whose first site supplies the displayed source unit.

    Returns
    -------
    PlotContext
        Caption metadata in seconds, Hz, radians, and source voltage units.
    """
    windows = config.analysis_windows
    units = np.asarray(arrays["site_voltage_units"]).astype(str)
    gamma = next(band for band in config.phase.bands if band.name == "gamma")
    exclusion = gamma.excluded_intervals_hz[0] if gamma.excluded_intervals_hz else None
    return PlotContext(
        session_id=config.session_id,
        alignment_event=windows.alignment_event,
        epoch_bounds_s={
            "whole": (windows.whole_start_s, windows.whole_stop_s),
            "before": (windows.before_start_s, windows.before_stop_s),
            "after": (windows.after_start_s, windows.after_stop_s),
        },
        notch_enabled=config.phase.notch_enabled,
        gamma_exclusion_hz=exclusion,
        reference_description="independent cached LFP sites; no rereference",
        source_voltage_unit=str(units[0]),
    )


def _plot_cached_unit_map(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    *,
    condition_index: int | None = None,
    site_index: int | None = None,
    epoch_index: int | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Render one selected or representative unit PPC map from cache views.

    Optional indices are zero-based categorical positions. If all are ``None``,
    the legacy maximal-reliability selection is used. Mixed ``None``/integer
    inputs are rejected. PPC is dimensionless, frequency is Hz, and counts are
    spikes on ``(unit, frequency)`` views.
    """
    if condition_index is site_index is epoch_index is None:
        condition_index, site_index, epoch_index = _representative_metric_cell(arrays)
    elif None in (condition_index, site_index, epoch_index):
        raise ValueError("unit map indices must be supplied together")
    assert condition_index is not None and site_index is not None and epoch_index is not None
    return plot_unit_ppc_map(
        np.asarray(arrays["ppc"])[:, condition_index, site_index, epoch_index],
        np.asarray(arrays["computable"])[:, condition_index, site_index, epoch_index],
        np.asarray(arrays["reliable"])[:, condition_index, site_index, epoch_index],
        np.asarray(arrays["spike_count"])[:, condition_index, site_index, epoch_index],
        np.asarray(arrays["frequency_hz"], dtype=float),
        tuple(np.asarray(arrays["unit_ids"]).astype(str)),
        str(np.asarray(arrays["condition_names"])[condition_index]),
        str(np.asarray(arrays["site_ids"])[site_index]),
        str(np.asarray(arrays["epoch_names"])[epoch_index]),
        _plot_context(config, arrays),
    )


def _plot_cached_population_map(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    *,
    site_index: int | None = None,
    epoch_index: int | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Render selected reliable median PPC and eligible FDR prevalence.

    Optional site/epoch values are zero-based categorical positions. Both must
    be supplied together or both omitted for the legacy representative cell.
    Values are dimensionless on ``(condition, frequency)`` views.
    """
    if site_index is None and epoch_index is None:
        _, site_index, epoch_index = _representative_metric_cell(arrays)
    elif site_index is None or epoch_index is None:
        raise ValueError("population map site and epoch indices must be supplied together")
    assert site_index is not None and epoch_index is not None
    ppc = np.asarray(arrays["ppc"])
    reliable = np.asarray(arrays["reliable"], dtype=bool)
    selected_ppc = np.asarray(
        ppc[:, :, site_index, epoch_index],
        dtype=float,
    )
    median = np.nanmedian(
        np.where(
            reliable[:, :, site_index, epoch_index],
            selected_ppc,
            np.nan,
        ),
        axis=0,
    )
    eligible = np.asarray(arrays["null_eligible"], dtype=bool)[
        :, :, site_index, epoch_index
    ]
    significant = np.asarray(arrays["significant"], dtype=bool)[
        :, :, site_index, epoch_index
    ]
    eligible_count = eligible.sum(axis=0, dtype=np.int64)
    total_count = np.full(eligible_count.shape, ppc.shape[0], dtype=np.int64)
    prevalence = np.full(eligible_count.shape, np.nan)
    np.divide(
        (significant & eligible).sum(axis=0),
        eligible_count,
        out=prevalence,
        where=eligible_count > 0,
    )
    return plot_population_ppc_maps(
        median,
        prevalence,
        eligible_count,
        total_count,
        tuple(np.asarray(arrays["condition_names"]).astype(str)),
        np.asarray(arrays["frequency_hz"], dtype=float),
        str(np.asarray(arrays["site_ids"])[site_index]),
        str(np.asarray(arrays["epoch_names"])[epoch_index]),
        _plot_context(config, arrays),
    )


def _plot_cached_band_summary(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    *,
    site_index: int | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Render one selected site's dimensionless band PPC summary.

    ``site_index`` is a zero-based categorical position. ``None`` preserves the
    legacy maximal-reliability site selection.
    """
    if site_index is None:
        _, site_index, _ = _representative_metric_cell(arrays)
    ppc_band = np.asarray(arrays["ppc_band_mean"], dtype=float)[:, :, site_index]
    reliability = _band_reliability(arrays, config)[:, :, site_index]
    epoch_names = tuple(np.asarray(arrays["epoch_names"]).astype(str))
    band_names = tuple(np.asarray(arrays["band_names"]).astype(str))
    epoch_indices = tuple(
        _required_label_index(epoch_names, name, "epoch") for name in _PRIMARY_EPOCHS
    )
    band_indices = tuple(
        _required_label_index(band_names, name, "band") for name in _PRIMARY_BANDS
    )
    ppc_band = np.take(
        np.take(ppc_band, epoch_indices, axis=2),
        band_indices,
        axis=3,
    )
    reliability = np.take(
        np.take(reliability, epoch_indices, axis=2),
        band_indices,
        axis=3,
    )
    return plot_ppc_band_summary(
        np.moveaxis(ppc_band, 0, 1),
        np.moveaxis(reliability, 0, 1),
        tuple(np.asarray(arrays["condition_names"]).astype(str)),
        _PRIMARY_EPOCHS,
        _PRIMARY_BANDS,
        ppc_band.shape[0],
        str(np.asarray(arrays["site_ids"])[site_index]),
        _plot_context(config, arrays),
    )


def _plot_cached_exemplar(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
) -> tuple[Any, Mapping[str, Any]]:
    """Render one deterministic cached PPC unit/trial exemplar."""
    condition_index, site_index, epoch_index = _representative_metric_cell(arrays)
    unit_index = _representative_unit_index(arrays, condition_index, site_index, epoch_index)
    trial_position = _representative_trial_position(
        arrays, condition_index, site_index, epoch_index
    )
    band_index = 0
    band_name = str(np.asarray(arrays["band_names"])[band_index])
    frequency_hz = np.asarray(arrays["frequency_hz"], dtype=float)
    representative_hz = 8.0 if band_name == "theta" else 40.0
    return plot_ppc_exemplar(
        frequency_hz,
        np.asarray(arrays["ppc"])[
            unit_index, condition_index, site_index, epoch_index
        ],
        np.asarray(arrays["preferred_phase_rad"])[
            unit_index, condition_index, site_index, epoch_index
        ],
        np.asarray(arrays["representative_phase_hist_count"])[
            unit_index, condition_index, site_index, epoch_index, band_index
        ],
        np.asarray(arrays["phase_bin_edges_rad"], dtype=float),
        np.asarray(arrays["relative_time_s"], dtype=float),
        np.asarray(arrays["source_trace"])[site_index, trial_position],
        np.asarray(arrays["band_filtered_trace"])[
            site_index, trial_position, band_index
        ],
        _unpacked_trial_spikes(arrays, unit_index, trial_position),
        str(np.asarray(arrays["unit_ids"])[unit_index]),
        int(np.asarray(arrays["trial_indices"])[trial_position]),
        band_name,
        representative_hz,
        "representative reliable unit",
        _plot_context(config, arrays),
    )


def _plot_cached_exemplar_pair(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    *,
    condition_index: int,
    site_index: int,
    epoch_index: int,
    band_index: int,
) -> tuple[Any, Mapping[str, Any]]:
    """Render one exact cached low/high unit and trial selection.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Validated final cache arrays. Views retain unit/condition/site/epoch/
        frequency, band/phase-bin, site/trial/time, and packed spike axes.
    config : LFPSummaryConfig
        Immutable session/plot provenance.
    condition_index, site_index, epoch_index, band_index : int
        Zero-based categorical positions without physical units.

    Returns
    -------
    tuple[matplotlib.figure.Figure, Mapping[str, matplotlib.axes.Axes]]
        Unsaved paired figure constructed without raw-data access or PPC work.
    """
    index = (condition_index, site_index, epoch_index, band_index)
    unit_ids = tuple(np.asarray(arrays["unit_ids"]).astype(str))
    trial_indices = np.asarray(arrays["trial_indices"], dtype=np.int64)
    low_unit_id = str(np.asarray(arrays["selected_low_unit_ids"])[index])
    high_unit_id = str(np.asarray(arrays["selected_high_unit_ids"])[index])
    low_trial_index = int(np.asarray(arrays["illustrative_low_trial_indices"])[index])
    high_trial_index = int(np.asarray(arrays["illustrative_high_trial_indices"])[index])
    low_unit_index = unit_ids.index(low_unit_id)
    high_unit_index = unit_ids.index(high_unit_id)
    low_trial_position = int(np.flatnonzero(trial_indices == low_trial_index)[0])
    high_trial_position = int(np.flatnonzero(trial_indices == high_trial_index)[0])
    panels = []
    for unit_index, unit_id, percentile, trial_index, trial_position in (
        (low_unit_index, low_unit_id, "5th percentile", low_trial_index, low_trial_position),
        (
            high_unit_index,
            high_unit_id,
            "95th percentile",
            high_trial_index,
            high_trial_position,
        ),
    ):
        panels.append(
            PPCExemplarPanel(
                unit_id=unit_id,
                percentile_label=percentile,
                pooled_ppc=np.asarray(arrays["ppc"])[
                    unit_index, condition_index, site_index, epoch_index
                ],
                preferred_phase_rad=np.asarray(arrays["preferred_phase_rad"])[
                    unit_index, condition_index, site_index, epoch_index
                ],
                representative_phase_hist_count=np.asarray(
                    arrays["representative_phase_hist_count"]
                )[
                    unit_index,
                    condition_index,
                    site_index,
                    epoch_index,
                    band_index,
                ],
                trial_index=trial_index,
                source_trace=np.asarray(arrays["source_trace"])[
                    site_index, trial_position
                ],
                filtered_trace=np.asarray(arrays["band_filtered_trace"])[
                    site_index, trial_position, band_index
                ],
                hilbert_phase_rad=np.asarray(arrays["hilbert_phase_rad"])[
                    site_index, trial_position, band_index
                ],
                spike_times_relative_s=_unpacked_trial_spikes(
                    arrays,
                    unit_index,
                    trial_position,
                ),
            )
        )
    band_name = str(np.asarray(arrays["band_names"])[band_index])
    representative_frequency_hz = 8.0 if band_name == "theta" else 40.0
    return plot_ppc_exemplar_pair(
        frequency_hz=np.asarray(arrays["frequency_hz"], dtype=float),
        phase_bin_edges_rad=np.asarray(arrays["phase_bin_edges_rad"], dtype=float),
        relative_time_s=np.asarray(arrays["relative_time_s"], dtype=float),
        low=panels[0],
        high=panels[1],
        condition_name=str(np.asarray(arrays["condition_names"])[condition_index]),
        site_label=str(np.asarray(arrays["site_ids"])[site_index]),
        epoch_name=str(np.asarray(arrays["epoch_names"])[epoch_index]),
        band_name=band_name,
        representative_frequency_hz=representative_frequency_hz,
        context=_plot_context(config, arrays),
    )


def _representative_metric_cell(
    arrays: Mapping[str, np.ndarray],
) -> tuple[int, int, int]:
    """Select a deterministic condition/site/epoch with maximal reliability."""
    reliable = np.asarray(arrays["reliable"], dtype=bool)
    counts = reliable.sum(axis=(0, 4))
    return tuple(int(value) for value in np.unravel_index(np.argmax(counts), counts.shape))


def _representative_unit_index(
    arrays: Mapping[str, np.ndarray],
    condition_index: int,
    site_index: int,
    epoch_index: int,
) -> int:
    """Return the most reliable unit for one representative cache cell."""
    reliable = np.asarray(arrays["reliable"], dtype=bool)[
        :, condition_index, site_index, epoch_index
    ]
    return int(np.argmax(reliable.sum(axis=1)))


def _representative_trial_position(
    arrays: Mapping[str, np.ndarray],
    condition_index: int,
    site_index: int,
    epoch_index: int,
) -> int:
    """Map a cached illustrative row id to a full trial-axis position."""
    selected = np.asarray(arrays["illustrative_trial_indices"])[
        condition_index, site_index, epoch_index, 0
    ]
    trial_indices = np.asarray(arrays["trial_indices"])
    positions = np.flatnonzero(trial_indices == selected)
    return int(positions[0]) if positions.size else 0


def _band_reliability(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
) -> np.ndarray:
    """Require every retained band frequency to meet reliable spike counts."""
    frequency_hz = np.asarray(arrays["frequency_hz"], dtype=float)
    reliable = np.asarray(arrays["reliable"], dtype=bool)
    summaries = []
    for band in config.phase.bands:
        selected = (frequency_hz >= band.lower_hz) & (frequency_hz <= band.upper_hz)
        for lower_hz, upper_hz in band.excluded_intervals_hz:
            selected &= ~((frequency_hz >= lower_hz) & (frequency_hz <= upper_hz))
        summaries.append(np.all(reliable[..., selected], axis=-1))
    return np.stack(summaries, axis=-1)


def _unpacked_trial_spikes(
    arrays: Mapping[str, np.ndarray],
    unit_index: int,
    trial_position: int,
) -> np.ndarray:
    """Return one cached unit/trial event-relative seconds spike vector."""
    offsets = np.asarray(arrays["relative_spike_time_offsets"], dtype=np.int64)
    start = int(offsets[unit_index, trial_position])
    stop = int(offsets[unit_index, trial_position + 1])
    return np.asarray(arrays["relative_spike_times_s"], dtype=float)[start:stop].copy()


def _compute_spike(config: LFPSummaryConfig, pipeline: object) -> ComponentRunResult:
    """Dispatch only the pipeline's Spike-phase component.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable preview configuration.
    pipeline : object
        Production Spike pipeline dependency bundle.

    Returns
    -------
    ComponentRunResult
        Atomic Spike-phase pipeline outcome; no Power or Synchrony is invoked.
    """
    from src.neural_analysis.lfp_summary.pipeline import compute_spike_phase_component
    return compute_spike_phase_component(
        config,
        pipeline,
        defer_post_commit_cleanup=True,
    )


def _assess_spike(
    directory: Path,
    component: str,
    config: LFPSummaryConfig,
    manifest: Mapping[str, Any],
) -> ComponentStatus:
    """Assess only the named Spike-phase cache component.

    Parameters
    ----------
    directory : pathlib.Path
        Generic numerical cache directory.
    component : str
        Required literal ``"spike_phase"``.
    config : LFPSummaryConfig
        Immutable compatibility inputs.
    manifest : Mapping[str, Any]
        Loaded cache manifest.

    Returns
    -------
    ComponentStatus
        Compatibility state without loading raw recordings.
    """
    from src.neural_analysis.lfp_summary.cache import assess_component_status
    return assess_component_status(directory, component, config, manifest)


def _save_png(figure: Any, path: Path) -> None:
    """Save one public PPC figure as opaque 150-dpi PNG.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        Unsaved cache-backed public PPC figure.
    path : pathlib.Path
        PNG destination inside the report staging directory.
    """
    figure.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")


def _unavailable_spike_pipeline() -> PipelineDependencies:
    """Return a dependency bundle that fails explicitly until runtime wiring exists.

    Returns
    -------
    PipelineDependencies
        A no-placeholder bundle whose Spike preparation fails before cache writes.
    """
    def unavailable(*_: object) -> object:
        """Raise rather than fabricate any unavailable production calculation."""
        raise NotImplementedError("Spike-phase runtime dependencies are unavailable")

    return PipelineDependencies(
        unavailable, unavailable, unavailable, unavailable, unavailable, unavailable,
        unavailable, unavailable,
    )


def _directory_size_bytes(directory: Path) -> int:
    """Return recursive cache-file size in bytes, or zero when absent."""
    if not directory.is_dir():
        return 0
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())


def _component_size_bytes(
    directory: Path,
    manifest: Mapping[str, object],
) -> int:
    """Return the committed Spike NPZ size in bytes, or zero when absent."""
    components = manifest.get("components", {})
    entry = components.get("spike_phase", {}) if isinstance(components, Mapping) else {}
    filename = (
        entry.get("file_name", "spike_phase.npz")
        if isinstance(entry, Mapping)
        else "spike_phase.npz"
    )
    path = directory / str(filename)
    return path.stat().st_size if path.is_file() else 0


def _resolved_report_measurements(
    config: LFPSummaryConfig,
    manifest: Mapping[str, object],
    wall_time_s: float,
    peak_memory_bytes: int,
    supplied: SpikePhaseReportMeasurements | None,
) -> SpikePhaseReportMeasurements:
    """Merge directly observed values into explicit report measurements.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable configuration whose worker count and cache path are used.
    manifest : Mapping[str, object]
        Compatible component metadata used to locate the final NPZ.
    wall_time_s : float
        Directly measured or caller-supplied component duration in seconds.
    peak_memory_bytes : int
        Best-effort current-process peak RSS in bytes; zero means unavailable.
    supplied : SpikePhaseReportMeasurements or None
        Optional richer launcher measurements. Explicit measured zeros and all
        non-``None`` values take precedence over locally observed fallbacks.

    Returns
    -------
    SpikePhaseReportMeasurements
        Validated scalar measurements with unavailable values retained as
        ``None`` and no numerical cache arrays.
    """
    _optional_nonnegative_float(wall_time_s, "wall_time_s")
    _optional_nonnegative_int(peak_memory_bytes, "peak_memory_bytes")
    requested_workers = int(config.ppc_execution.worker_count)
    if supplied is not None and not isinstance(supplied, SpikePhaseReportMeasurements):
        raise ValueError("report_measurements must be SpikePhaseReportMeasurements or None")
    if supplied is not None and supplied.requested_worker_count != requested_workers:
        raise ValueError(
            "report requested_worker_count must match the active PPC configuration"
        )

    component_bytes = _component_size_bytes(config.output_directory, manifest)
    cache_bytes = _directory_size_bytes(config.output_directory)
    base = supplied or SpikePhaseReportMeasurements(
        requested_worker_count=requested_workers,
    )
    process_rss = base.peak_process_rss_bytes
    memory_provenance = base.memory_provenance
    if process_rss is None and peak_memory_bytes > 0:
        process_rss = int(peak_memory_bytes)
        if memory_provenance is None:
            memory_provenance = "current process peak RSS from resource usage"
    component_total = (
        float(wall_time_s)
        if base.component_total_seconds is None
        else base.component_total_seconds
    )
    final_component = (
        component_bytes
        if base.final_component_size_bytes is None
        else base.final_component_size_bytes
    )
    final_cache = (
        cache_bytes
        if base.final_cache_size_bytes is None
        else base.final_cache_size_bytes
    )
    benchmark = base.benchmark
    if benchmark is None:
        benchmark = SpikePhaseFilterBenchmark(
            source_label="current active filter run",
            wall_time_s=component_total,
            final_cache_size_bytes=final_cache,
            intermediate_cache_size_bytes=base.intermediate_work_size_bytes,
        )
    return replace(
        base,
        component_total_seconds=component_total,
        peak_process_rss_bytes=process_rss,
        memory_provenance=memory_provenance,
        final_component_size_bytes=final_component,
        final_cache_size_bytes=final_cache,
        benchmark=benchmark,
    )


def _write_report_artifacts(
    staging: Path,
    manifest: Mapping[str, object],
    config: LFPSummaryConfig,
    report: Mapping[str, object],
    source_identifiers: Mapping[str, str],
) -> dict[str, Path]:
    """Write the fixed scalar artifact set inside one staging directory.

    Parameters
    ----------
    staging : pathlib.Path
        Existing temporary report directory that is not yet published.
    manifest, report : Mapping[str, object]
        JSON-compatible cache metadata and scalar/categorical report content.
    config : LFPSummaryConfig
        Immutable configuration serialized with its documented physical units.
    source_identifiers : Mapping[str, str]
        Short code/session provenance strings containing no numerical arrays.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths keyed by report, log, manifest, configuration, source identifiers,
        and summary. Every path is a direct child of ``staging``.
    """
    paths = {
        "report": staging / "report.json",
        "log": staging / "run.log",
        "manifest": staging / "manifest_snapshot.json",
        "configuration": staging / "configuration.json",
        "source_identifiers": staging / "source_identifiers.json",
        "summary": staging / "run_summary.md",
    }
    _write_json_ascii(paths["report"], report)
    _write_json_ascii(paths["manifest"], manifest)
    _write_json_ascii(
        paths["configuration"],
        json.loads(canonical_config_json(config)),
    )
    _write_json_ascii(paths["source_identifiers"], source_identifiers)
    paths["summary"].write_text(
        _summary_markdown(config, report),
        encoding="ascii",
    )
    log_entry = {
        "event": "spike_phase_preview_report_complete",
        "report": dict(report),
    }
    paths["log"].write_text(
        json.dumps(log_entry, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return paths


def _write_json_ascii(path: Path, value: Mapping[str, object]) -> None:
    """Write one deterministic ASCII JSON mapping with a trailing newline.

    Parameters
    ----------
    path : pathlib.Path
        Destination file inside the report staging directory.
    value : Mapping[str, object]
        JSON-safe scalar/categorical mapping containing no signal arrays.

    Returns
    -------
    None
        Nonfinite values raise before a valid report can be published.
    """
    path.write_text(
        json.dumps(dict(value), sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _validate_staged_report(directory: Path, png_paths: tuple[Path, ...]) -> None:
    """Reload and validate the exact staged or published artifact set.

    Parameters
    ----------
    directory : pathlib.Path
        Existing staging or final report directory.
    png_paths : tuple[pathlib.Path, ...]
        Expected PNG paths in ``directory`` in deterministic planner order.

    Returns
    -------
    None
        Success proves JSON/log reload, ASCII Markdown, PNG signatures, figure
        selection agreement, and absence of missing or unexpected artifacts.
    """
    fixed_names = {
        "report.json",
        "run.log",
        "manifest_snapshot.json",
        "configuration.json",
        "source_identifiers.json",
        "run_summary.md",
    }
    expected_png_names = tuple(path.name for path in png_paths)
    if len(expected_png_names) != len(set(expected_png_names)):
        raise ValueError("report PNG names must be unique")
    expected_names = fixed_names | set(expected_png_names)
    actual_names = {path.name for path in directory.iterdir()}
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ValueError(
            f"report artifact set mismatch: missing={missing}, unexpected={unexpected}"
        )

    decoded: dict[str, object] = {}
    for name in (
        "report.json",
        "manifest_snapshot.json",
        "configuration.json",
        "source_identifiers.json",
    ):
        value = json.loads((directory / name).read_text(encoding="ascii"))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain one JSON mapping")
        decoded[name] = value
    report = decoded["report.json"]
    assert isinstance(report, dict)
    if report.get("schema_version") not in {
        _REPORT_SCHEMA_VERSION,
        _GENERAL_REPORT_SCHEMA_VERSION,
    }:
        raise ValueError("report schema_version is incompatible")

    rendered = report.get("rendered_selections")
    unavailable = report.get("unavailable_selections")
    if not isinstance(rendered, list) or not isinstance(unavailable, list):
        raise ValueError("report selection records must be JSON lists")
    rendered_names = []
    for record in rendered:
        if not isinstance(record, dict) or not isinstance(record.get("file_name"), str):
            raise ValueError("each rendered selection must name one PNG")
        rendered_names.append(record["file_name"])
    if set(rendered_names) != set(expected_png_names) or len(rendered_names) != len(
        expected_png_names
    ):
        raise ValueError("rendered selections do not agree with the PNG artifact set")
    if any(
        not isinstance(record, dict) or record.get("file_name") is not None
        for record in unavailable
    ):
        raise ValueError("unavailable selections must not name a PNG")

    log_lines = (directory / "run.log").read_text(encoding="ascii").splitlines()
    if not log_lines:
        raise ValueError("run.log must contain at least one JSON line")
    log_entries = []
    for line in log_lines:
        entry = json.loads(line)
        if not isinstance(entry, dict):
            raise ValueError("each run.log line must contain one JSON mapping")
        log_entries.append(entry)
    if not any(
        entry.get("event") == "spike_phase_preview_report_complete"
        and entry.get("report") == report
        for entry in log_entries
    ):
        raise ValueError("run.log does not contain the complete machine report")
    summary = (directory / "run_summary.md").read_text(encoding="ascii")
    if not summary.strip():
        raise ValueError("run_summary.md must be nonempty ASCII Markdown")
    for path in png_paths:
        if path.parent != directory or not path.is_file():
            raise ValueError(f"report PNG is missing or outside its directory: {path}")
        with path.open("rb") as stream:
            if stream.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
                raise ValueError(f"report PNG has an invalid signature: {path.name}")


def _publish_staged_report(staging: Path, final: Path) -> None:
    """Atomically rename one validated staging directory into final position.

    Parameters
    ----------
    staging : pathlib.Path
        Existing validated temporary sibling directory.
    final : pathlib.Path
        Uncreated immutable report directory on the same filesystem.

    Returns
    -------
    None
        The staging directory becomes ``final`` without exposing partial files.
    """
    if final.exists():
        raise FileExistsError(f"validation run already exists: {final}")
    staging.replace(final)


def _optional_nonnegative_float(value: float | None, name: str) -> None:
    """Validate optional finite nonnegative seconds without coercing ``None``.

    Parameters
    ----------
    value : float or None
        Optional scalar in seconds.
    name : str
        Human-readable field name used in validation failures.

    Returns
    -------
    None
        Measured numeric zero is accepted; Booleans and nonfinite values fail.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite nonnegative number or None")
    if not np.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number or None")


def _optional_nonnegative_int(value: int | None, name: str) -> None:
    """Validate optional nonnegative counts/bytes without changing units.

    Parameters
    ----------
    value : int or None
        Optional scalar count or byte quantity.
    name : str
        Human-readable field name used in validation failures.

    Returns
    -------
    None
        Measured integer zero is accepted; Booleans and negative values fail.
    """
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")


def _production_peak_memory_bytes() -> int:
    """Return best-effort current-process peak resident-set size in bytes."""
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, AttributeError):
        return 0
    if value <= 0:
        return 0
    return value if sys.platform == "darwin" else value * 1024


def _source_identifiers(config: LFPSummaryConfig) -> dict[str, str]:
    """Return code revision/module identities without numerical arrays."""
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
        "runtime_module": str(Path(__file__).resolve().with_name("lfp_summary_runtime.py")),
        "validation_module": str(Path(__file__).resolve()),
        "session_id": config.session_id,
    }


def _summary_markdown(config: LFPSummaryConfig, report: Mapping[str, object]) -> str:
    """Return a detailed ASCII report summary without raw numerical arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable preview configuration.
    report : Mapping[str, object]
        Scalar/categorical counts, measurements, warnings, selections, and
        projection inputs from the validated machine report.

    Returns
    -------
    str
        Markdown text with explicit ``unavailable`` markers, physical units,
        and no numerical cache arrays.
    """
    timings = report["timing_seconds"]
    workers = report["workers"]
    memory = report["memory_bytes"]
    sizes = report["sizes_bytes"]
    projection = report["nine_filter_projection"]
    assert isinstance(timings, Mapping)
    assert isinstance(workers, Mapping)
    assert isinstance(memory, Mapping)
    assert isinstance(sizes, Mapping)
    assert isinstance(projection, Mapping)
    text = (
        "# Spike-phase preview report\n\n"
        "Goal: inspect cache-backed PPC summaries and exact low/high "
        "exemplars before any final-shuffle run.\n\n"
        f"Session: {_markdown_value(config.session_id)}\n\n"
        f"Population: {_markdown_value(report['population_label'])}; "
        f"probe: {_markdown_value(report['probe_label'])}\n\n"
        "## Execution\n\n"
        f"Workers (requested/planned/active): {_markdown_value(workers['requested'])} / "
        f"{_markdown_value(workers['planned'])} / "
        f"{_markdown_value(workers['active'])}\n\n"
        "Prepared-phase cache state: "
        f"{_markdown_value(report['prepared_phase_cache_state'])}\n\n"
        "Timing seconds (phase preparation/planning/grouped/component/report): "
        f"{_markdown_value(timings['phase_preparation'])} / "
        f"{_markdown_value(timings['ppc_planning'])} / "
        f"{_markdown_value(timings['grouped_execution'])} / "
        f"{_markdown_value(timings['component_total'])} / "
        f"{_markdown_value(timings['report'])}\n\n"
        "Peak memory bytes (process RSS/tree RSS/tree PSS): "
        f"{_markdown_value(memory['process_rss'])} / "
        f"{_markdown_value(memory['process_tree_rss'])} / "
        f"{_markdown_value(memory['process_tree_pss'])}\n\n"
        f"Memory provenance: {_markdown_value(memory['provenance'])}\n\n"
        "Storage bytes (final component/final cache/intermediate work): "
        f"{_markdown_value(sizes['final_component'])} / "
        f"{_markdown_value(sizes['final_cache'])} / "
        f"{_markdown_value(sizes['intermediate_work'])}\n\n"
        "## Scientific summary\n\n"
        f"Shuffles: {_markdown_value(report['shuffle_count'])}\n\n"
        f"Trials (total/selected): {_markdown_value(report['trial_count'])} / "
        f"{_markdown_value(report['selected_trial_count'])}\n\n"
        f"Units: {_markdown_value(report['unit_count'])}; unique packed spikes: "
        f"{_markdown_value(report['spike_count'])}\n\n"
        "PPC cells (computable/reliable/null-eligible/significant): "
        f"{_markdown_value(report['computable_cell_count'])} / "
        f"{_markdown_value(report['reliable_cell_count'])} / "
        f"{_markdown_value(report['null_eligible_cell_count'])} / "
        f"{_markdown_value(report['significant_cell_count'])}\n\n"
        f"Rendered selections: {len(report['rendered_selections'])}; unavailable "
        f"selections: {len(report['unavailable_selections'])}\n\n"
        f"Warnings: {_markdown_value(report['warnings'])}\n\n"
        f"Exclusions: {_markdown_value(report['exclusions'])}\n\n"
        "## Nine-filter linear projection\n\n"
        f"Benchmark source: {_markdown_value(projection['benchmark_source'])}\n\n"
        f"Benchmark wall time: {_markdown_value(projection['benchmark_wall_time_s'])} s\n\n"
        "Benchmark final/intermediate bytes: "
        f"{_markdown_value(projection['benchmark_final_cache_size_bytes'])} / "
        f"{_markdown_value(projection['benchmark_intermediate_cache_size_bytes'])}\n\n"
        f"Projected wall time: {_markdown_value(projection['projected_wall_time_s'])} s\n\n"
        "Projected final/intermediate bytes: "
        f"{_markdown_value(projection['projected_final_cache_size_bytes'])} / "
        f"{_markdown_value(projection['projected_intermediate_cache_size_bytes'])}\n"
    )
    return text.encode("ascii", errors="backslashreplace").decode("ascii")


def _markdown_value(value: object) -> str:
    """Return one deterministic ASCII scalar/container display value.

    Parameters
    ----------
    value : object
        JSON-compatible scalar or short categorical container; no arrays.

    Returns
    -------
    str
        ``"unavailable"`` for ``None`` or an ASCII JSON-style representation.
    """
    if value is None:
        return "unavailable"
    if isinstance(value, str):
        return value.encode("ascii", errors="backslashreplace").decode("ascii")
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)
