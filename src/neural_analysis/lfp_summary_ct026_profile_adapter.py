"""Injected production adapter for nonpublishing CT026 PPC profile runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Callable, Mapping

from src.neural_analysis.lfp_summary_ppc_profile import RepresentativePPCProfileJob


_SCENARIO_UNITS = {
    "low": lambda job: (job.low_unit_id,),
    "median": lambda job: (job.median_unit_id,),
    "high": lambda job: (job.high_unit_id,),
    "combined": lambda job: (job.low_unit_id, job.median_unit_id, job.high_unit_id),
}
_RSS_SOURCE = "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss_kib"


@dataclass(frozen=True)
class CT026PPCProfileAdapterDependencies:
    """Injected boundaries for a CT026 profile adapter.

    All preparation, slicing, isolated child execution, identity generation,
    recovery, and orchestration behavior is injected. ``run_profile`` must be
    the existing interruption-safe generic runner. No dependency builds a
    component, manifest, payload, preview, or scientific report artifact.
    """

    build_config: Callable[[Path, object], object]
    load_active_population: Callable[[Path], object]
    prepare_phase: Callable[..., object]
    prepare_spikes: Callable[[object, object], object]
    select_profile_job: Callable[[object, object, object], RepresentativePPCProfileJob]
    slice_profile_job: Callable[..., object]
    run_isolated_profile_job: Callable[..., Mapping[str, object]]
    run_profile: Callable[..., object]
    config_fingerprint: Callable[[object], str]
    source_fingerprint: Callable[[object], str]
    git_fingerprint: Callable[[], str]
    monotonic_seconds: Callable[[], float]
    acquire_run_lock: Callable[..., AbstractContextManager[None]]
    recover_profile_work: Callable[..., None]


def run_ct026_ppc_profile_adapter(
    *,
    session_path: Path,
    analysis_root: Path,
    dependencies: CT026PPCProfileAdapterDependencies,
    run_directory: Path | None = None,
) -> object:
    """Bind injected CT026 inputs to the generic interruption-safe profile runner.

    Parameters
    ----------
    session_path : pathlib.Path
        CT026 session root, passed only to injected population/config builders.
    analysis_root : pathlib.Path
        Parent of generic runner work-only run directories; no final component,
        manifest, NPZ, or preview output is written by this adapter.
    dependencies : CT026PPCProfileAdapterDependencies
        Explicit production or test seams. Child work is sequential because the
        generic runner invokes one scenario callback at a time.
    run_directory : pathlib.Path or None
        Exact existing generic-runner directory for resume, or ``None`` for a
        new run. A resumed child invokes the injected audited recovery seam.

    Returns
    -------
    object
        The unchanged result returned by the existing generic runner.
    """
    if not isinstance(dependencies, CT026PPCProfileAdapterDependencies):
        raise ValueError("dependencies must be CT026PPCProfileAdapterDependencies")
    _validate_callables(dependencies)
    session = Path(session_path)
    root = Path(analysis_root)
    resumed_directory = Path(run_directory) if run_directory is not None else None
    population = dependencies.load_active_population(session)
    config = dependencies.build_config(session, population)
    identity = {
        "config_fingerprint": _identity_value(dependencies.config_fingerprint(config), "config_fingerprint"),
        "source_fingerprint": _identity_value(dependencies.source_fingerprint(config), "source_fingerprint"),
        "git_fingerprint": _identity_value(dependencies.git_fingerprint(), "git_fingerprint"),
    }
    phase_work_root = root / "ct026_ppc_profile_work" / "phase"
    phase_seconds: dict[str, float] = {"cold": 0.0, "warm": 0.0}
    selected_job: RepresentativePPCProfileJob | None = None

    def prepare_phase(*, warm: bool) -> object:
        """Prepare cold or mmap-warm phase under one stable work-only path."""
        if not isinstance(warm, bool):
            raise ValueError("warm must be Boolean")
        before = _read_clock(dependencies.monotonic_seconds)
        phase = dependencies.prepare_phase(config, work_cache_root=phase_work_root)
        after = _read_clock(dependencies.monotonic_seconds)
        if after < before:
            raise ValueError("monotonic_seconds must not decrease")
        phase_seconds["warm" if warm else "cold"] = after - before
        return phase

    def select_spikes(phase: object) -> object:
        """Prepare spikes and select exact metadata from the warm phase object."""
        nonlocal selected_job
        spikes = dependencies.prepare_spikes(config, phase)
        job = dependencies.select_profile_job(config, phase, spikes)
        if not isinstance(job, RepresentativePPCProfileJob):
            raise ValueError("select_profile_job must return RepresentativePPCProfileJob")
        selected_job = job
        return spikes

    def profile_job(
        *, scenario: str, phase: object, spikes: object, shuffle_count: int, work_root: Path,
    ) -> dict[str, object]:
        """Slice one exact scenario and run it through the isolated-child seam."""
        nonlocal selected_job
        if shuffle_count != 100:
            raise ValueError("CT026 PPC profiling requires exactly 100 shuffles")
        if scenario not in _SCENARIO_UNITS:
            raise ValueError("unknown CT026 PPC profile scenario")
        if selected_job is None:
            # The generic runner normally calls select_spikes first. This fallback
            # keeps an injected resume seam self-contained without reloading spikes.
            selected = dependencies.select_profile_job(config, phase, spikes)
            if not isinstance(selected, RepresentativePPCProfileJob):
                raise ValueError("select_profile_job must return RepresentativePPCProfileJob")
            selected_job = selected
        scenario_work_root = Path(work_root)
        if resumed_directory is not None:
            dependencies.recover_profile_work(
                work_root=scenario_work_root,
                identity=dict(identity),
            )
        unit_ids = _SCENARIO_UNITS[scenario](selected_job)
        sliced = dependencies.slice_profile_job(
            profile_job=selected_job,
            scenario=scenario,
            unit_ids=unit_ids,
            phase=phase,
            spikes=spikes,
        )
        child_metrics = dependencies.run_isolated_profile_job(
            job=sliced,
            shuffle_count=shuffle_count,
            work_root=scenario_work_root,
        )
        if not isinstance(child_metrics, Mapping):
            raise ValueError("isolated profile child must return a scalar metric mapping")
        metrics = dict(child_metrics)
        metrics.update(
            {
                "phase_cold_seconds": phase_seconds["cold"],
                "phase_warm_seconds": phase_seconds["warm"],
                "peak_memory_source": _RSS_SOURCE,
                **identity,
            }
        )
        return metrics

    with dependencies.acquire_run_lock(
        analysis_root=root,
        identity=dict(identity),
        run_directory=resumed_directory,
    ):
        return dependencies.run_profile(
            config=config,
            config_fingerprint=identity["config_fingerprint"],
            source_fingerprint=identity["source_fingerprint"],
            git_fingerprint=identity["git_fingerprint"],
            analysis_root=root,
            run_directory=resumed_directory,
            prepare_phase=prepare_phase,
            select_spikes=select_spikes,
            profile_job=profile_job,
        )


def _validate_callables(dependencies: CT026PPCProfileAdapterDependencies) -> None:
    """Reject missing injected seams before configuration or run-state work."""
    for field_name in dependencies.__dataclass_fields__:
        if not callable(getattr(dependencies, field_name)):
            raise ValueError(f"{field_name} must be callable")


def _identity_value(value: object, name: str) -> str:
    """Validate one adapter-generated nonempty categorical identity."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _read_clock(clock: Callable[[], float]) -> float:
    """Return one finite injected monotonic seconds sample."""
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError("monotonic_seconds must return finite seconds")
    return float(value)
