"""CT026 100-shuffle Spike-phase preview validation without Streamlit."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping

import numpy as np

from src.neural_analysis.lfp_power_validation import build_ct026_power_config
from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    UnitPopulationConfig,
    canonical_config_json,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult, PipelineDependencies


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
    result = dependencies.compute_spike_phase_component(config, dependencies.pipeline_dependencies)
    if result.state != "complete":
        raise RuntimeError(result.error or "Spike-phase component failed")
    manifest, arrays = _load_compatible_spike_phase(config, dependencies)
    return _write_report(config, Path(run_parent), run_directory, manifest, arrays)


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
    report = _scalar_report(arrays)
    run_parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{run_directory.name}.staging-", dir=run_parent) as temp:
        staging = Path(temp)
        manifest_path = staging / "manifest_snapshot.json"
        config_path = staging / "configuration.json"
        summary_path = staging / "run_summary.md"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="ascii")
        config_path.write_text(canonical_config_json(config), encoding="ascii")
        summary_path.write_text(_summary_markdown(config, report), encoding="ascii")
        staging.replace(run_directory)
    return SpikePhasePreviewValidationResult(
        "spike_phase", run_directory, run_directory / manifest_path.name,
        run_directory / config_path.name, run_directory / summary_path.name,
        int(report["trial_count"]), int(report["unit_count"]),
        int(report["reliable_cell_count"]), report,
    )


def _scalar_report(arrays: Mapping[str, np.ndarray]) -> dict[str, object]:
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
        "reliable_cell_count": int(np.count_nonzero(np.asarray(arrays["reliable"], dtype=bool))),
        "shuffle_count": _PREVIEW_SHUFFLE_COUNT,
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
