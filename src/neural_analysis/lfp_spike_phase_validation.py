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

from src.neural_analysis import lfp_summary_runtime
from src.neural_analysis.lfp_power_validation import build_ct026_power_config
from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    UnitPopulationConfig,
    canonical_config_json,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult, PipelineDependencies
from src.neural_analysis.lfp_summary_plotting import (
    PlotContext,
    plot_population_ppc_maps,
    plot_ppc_band_summary,
    plot_ppc_exemplar,
    plot_unit_ppc_map,
)


_PREVIEW_SHUFFLE_COUNT = 100
_RUN_LABEL = "lfp_spike_phase_preview_validation"


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
    save_png: Callable[[Any, Path], None]
    close_figure: Callable[[Any], None]
    source_identifiers: Callable[[LFPSummaryConfig], dict[str, str]]


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
    root = Path(session_path)
    sorter = root / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeB/kilosort4"
    clusters = cluster_metadata_loader(sorter)
    channels = channel_metadata_loader(sorter)
    required_clusters = {"cluster_id", "ch", "group"}
    if not required_clusters.issubset(clusters) or "inside_brain" not in channels:
        raise ValueError("CT026 ProbeB metadata is missing required quality columns")
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
        raise ValueError("CT026 ProbeB metadata lacks channel id or label columns")
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
    unit_ids = tuple(f"ProbeB:{int(value)}" for value in selected["cluster_id"])
    return UnitPopulationConfig(
        "CT026 ProbeB active", "ProbeB", sorter,
        root / "ephys/aligned/aligned_open_ephys/probeB_sync.npz",
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
    factory = getattr(lfp_summary_runtime, "make_spike_phase_pipeline_dependencies", None)
    if factory is None:
        pipeline: object = _unavailable_spike_pipeline()
    else:
        pipeline = factory(trial_table_loader=lfp_summary_runtime.load_configured_trial_table)

    def load_manifest(directory: Path, config: LFPSummaryConfig) -> dict[str, object]:
        """Load the manifest matching the immutable active configuration."""
        from src.neural_analysis.lfp_summary_io import load_or_initialize_manifest
        return load_or_initialize_manifest(directory, config)

    def load_arrays(directory: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Load the validated Spike NPZ without pickle or raw recordings."""
        from src.neural_analysis.lfp_summary_io import load_component_arrays
        return load_component_arrays(directory / "spike_phase.npz", manifest, "spike_phase")

    return SpikePhasePreviewValidationDependencies(
        pipeline, _compute_spike, load_manifest, _assess_spike, load_arrays,
        lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        time.monotonic, _production_peak_memory_bytes, _plot_cached_unit_map,
        _plot_cached_population_map, _plot_cached_band_summary,
        _plot_cached_exemplar, _save_png, plt.close, _source_identifiers,
    )


def run_spike_phase_preview_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
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
    )


def render_cached_spike_phase_preview_validation(
    config: LFPSummaryConfig,
    run_parent: Path,
    dependencies: SpikePhasePreviewValidationDependencies,
    *,
    spike_phase_wall_time_s: float,
    spike_phase_peak_memory_bytes: int,
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


def _planned_run_directory(config: LFPSummaryConfig, run_parent: Path, timestamp: str) -> Path:
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
    path = run_parent / f"{config.session_id}_{_RUN_LABEL}_{timestamp}"
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

    Returns
    -------
    SpikePhasePreviewValidationResult
        Final immutable report paths and scalar summaries.
    """
    report = _scalar_report(
        arrays,
        wall_time_s,
        peak_memory_bytes,
        _directory_size_bytes(config.output_directory),
        _component_size_bytes(config.output_directory, manifest),
    )
    run_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{run_directory.name}.staging-", dir=run_parent) as temp:
        staging = Path(temp)
        manifest_path = staging / "manifest_snapshot.json"
        config_path = staging / "configuration.json"
        sources_path = staging / "source_identifiers.json"
        summary_path = staging / "run_summary.md"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii")
        config_path.write_text(canonical_config_json(config), encoding="ascii")
        sources_path.write_text(
            json.dumps(dependencies.source_identifiers(config), indent=2, sort_keys=True),
            encoding="ascii",
        )
        summary_path.write_text(_summary_markdown(config, report), encoding="ascii")
        png_paths = _render_cached_pngs(config, arrays, staging, dependencies)
        staging.replace(run_directory)
    return SpikePhasePreviewValidationResult(
        "spike_phase", run_directory, run_directory / manifest_path.name,
        run_directory / config_path.name, run_directory / summary_path.name,
        int(report["trial_count"]), int(report["unit_count"]),
        int(report["reliable_cell_count"]),
        tuple(run_directory / path.name for path in png_paths), report,
    )


def _scalar_report(
    arrays: Mapping[str, np.ndarray],
    wall_time_s: float,
    peak_memory_bytes: int,
    cache_size_bytes: int,
    component_size_bytes: int,
) -> dict[str, object]:
    """Summarize cache axes and reliable PPC cells without retaining raw values.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Spike-phase arrays containing ``trial_indices``, ``unit_ids``, and
        ``reliable`` on their documented cache axes.

    Returns
    -------
    dict[str, object]
        JSON-ready categorical counts for the immutable report.
    """
    return {
        "trial_count": int(np.asarray(arrays["trial_indices"]).size),
        "unit_count": int(np.asarray(arrays["unit_ids"]).size),
        "reliable_cell_count": int(
            np.count_nonzero(np.asarray(arrays["reliable"], dtype=bool))
        ),
        "null_eligible_cell_count": int(
            np.count_nonzero(np.asarray(arrays["null_eligible"], dtype=bool))
        ),
        "spike_count": _unique_cached_spike_count(arrays),
        "shuffle_count": _PREVIEW_SHUFFLE_COUNT,
        "wall_time_s": float(wall_time_s),
        "peak_memory_bytes": int(peak_memory_bytes),
        "cache_size_bytes": int(cache_size_bytes),
        "component_size_bytes": int(component_size_bytes),
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
) -> tuple[Path, ...]:
    """Request the four public PPC figure classes using cached arrays only.

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
    tuple[pathlib.Path, ...]
        Staged PNG paths for unit, population, band-summary, and exemplar plots.
    """
    plotters = (
        ("unit_ppc_map", dependencies.plot_unit_ppc_map),
        ("population_ppc_map", dependencies.plot_population_ppc_maps),
        ("ppc_band_summary", dependencies.plot_ppc_band_summary),
        ("ppc_exemplar", dependencies.plot_ppc_exemplar),
    )
    paths: list[Path] = []
    for name, plotter in plotters:
        figure, _ = plotter(arrays, config)
        path = directory / f"{name}.png"
        try:
            dependencies.save_png(figure, path)
        finally:
            dependencies.close_figure(figure)
        paths.append(path)
    return tuple(paths)


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
) -> tuple[Any, Mapping[str, Any]]:
    """Render one representative unit PPC map from cache axes only."""
    condition_index, site_index, epoch_index = _representative_metric_cell(arrays)
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
) -> tuple[Any, Mapping[str, Any]]:
    """Render reliable median PPC and eligible FDR prevalence from cache."""
    _, site_index, epoch_index = _representative_metric_cell(arrays)
    ppc = np.asarray(arrays["ppc"], dtype=float)
    reliable = np.asarray(arrays["reliable"], dtype=bool)
    median = np.nanmedian(
        np.where(reliable[:, :, site_index, epoch_index],
                 ppc[:, :, site_index, epoch_index], np.nan),
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
) -> tuple[Any, Mapping[str, Any]]:
    """Render one site's band PPC summary with reliable-frequency gating."""
    _, site_index, _ = _representative_metric_cell(arrays)
    ppc_band = np.asarray(arrays["ppc_band_mean"], dtype=float)[:, :, site_index]
    reliability = _band_reliability(arrays, config)[:, :, site_index]
    return plot_ppc_band_summary(
        np.moveaxis(ppc_band, 0, 1),
        np.moveaxis(reliability, 0, 1),
        tuple(np.asarray(arrays["condition_names"]).astype(str)),
        tuple(np.asarray(arrays["epoch_names"]).astype(str)),
        tuple(np.asarray(arrays["band_names"]).astype(str)),
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
    from src.neural_analysis.lfp_summary_pipeline import compute_spike_phase_component
    return compute_spike_phase_component(config, pipeline)


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
    from src.neural_analysis.lfp_summary_io import assess_component_status
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
        "runtime_module": str(Path(lfp_summary_runtime.__file__).resolve()),
        "validation_module": str(Path(__file__).resolve()),
        "session_id": config.session_id,
    }


def _summary_markdown(config: LFPSummaryConfig, report: Mapping[str, object]) -> str:
    """Return a small ASCII report summary without raw arrays.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable preview configuration.
    report : Mapping[str, object]
        Scalar report counts and approved shuffle count.

    Returns
    -------
    str
        Markdown text with categorical counts and no numerical cache arrays.
    """
    return (
        "# CT026 Spike-phase preview\n\n"
        f"Session: {config.session_id}\n\n"
        f"Shuffles: {report['shuffle_count']}\n\n"
        f"Trials: {report['trial_count']}\n\n"
        f"Units: {report['unit_count']}\n\n"
        f"Reliable PPC cells: {report['reliable_cell_count']}\n"
    )
