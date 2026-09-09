"""Injected production adapter for nonpublishing CT026 PPC profile runs."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import socket
import subprocess
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from src.neural_analysis import lfp_summary_runtime, lfp_summary_work_cache, spike_behavior_pynapple, unit_spike_loading
from src.neural_analysis.lfp_spike_phase_validation import build_ct026_default_active_population, build_ct026_spike_phase_preview_config
from src.neural_analysis.lfp_summary_ct026_profile_locks import acquire_ct026_profile_run_lock
from src.neural_analysis.lfp_summary_models import PPCExecutionConfig, canonical_config_json, fingerprint_source_files
from src.neural_analysis.lfp_summary_ppc_profile import RepresentativePPCProfileJob, profile_production_ppc_job, select_representative_ppc_profile_job


_SCENARIO_UNITS = {
    "low": lambda job: (job.low_unit_id,),
    "median": lambda job: (job.median_unit_id,),
    "high": lambda job: (job.high_unit_id,),
    "combined": lambda job: (job.low_unit_id, job.median_unit_id, job.high_unit_id),
}
_RSS_SOURCE = "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss_kib"


@dataclass(frozen=True)
class CT026PPCProfileSlice:
    """One bounded PPC child input with phase axes ``(1,freq,trial,time)``."""
    scenario: str
    unit_ids: tuple[str, ...]
    trial_indices: tuple[int, ...]
    site_id: str
    epoch_bounds_s: tuple[float, float]
    overlap_trial_indices: tuple[int, ...]
    phase_tensor: np.ndarray
    phase_valid: np.ndarray
    prepared_phase: object
    prepared_spikes: object
    schedule: np.ndarray


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
            config=config,
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


def _ct026_probe_b_sorter(session_path: Path) -> Path:
    """Return the established CT026 ProbeB Kilosort directory without I/O."""
    return Path(session_path) / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeB/kilosort4"


def make_production_ct026_profile_dependencies(*, work_cache_root: Path) -> CT026PPCProfileAdapterDependencies:
    """Bind existing CT026 preparation/profile APIs without executing them."""
    def load_population(session: Path) -> object:
        sorter = _ct026_probe_b_sorter(session)
        return build_ct026_default_active_population(
            session,
            lambda path: spike_behavior_pynapple.load_sorter_metadata(path)[1],
            unit_spike_loading.load_channel_quality,
        )
    def build_config(session: Path, population: object) -> object:
        config = build_ct026_spike_phase_preview_config(session, population)
        return replace(config, ppc=replace(config.ppc, shuffle_count=100), ppc_execution=replace(config.ppc_execution, worker_count=1))
    def prepare_phase(config: object, *, work_cache_root: Path) -> object:
        del work_cache_root
        return lfp_summary_runtime.prepare_phase_run(config, lfp_summary_runtime.load_configured_trial_table, work_cache_root=Path(shared_work_root))
    def prepare_spikes(config: object, phase: object) -> object:
        return lfp_summary_runtime.prepare_spike_run(config, phase, lfp_summary_runtime.load_configured_unit_spikes)
    shared_work_root = Path(work_cache_root)
    return CT026PPCProfileAdapterDependencies(build_config, load_population, prepare_phase, prepare_spikes, select_representative_ppc_profile_job, slice_ct026_profile_job, run_isolated_ct026_profile_job, __import__("src.neural_analysis.lfp_summary_ct026_profile_runner", fromlist=["run_ct026_ppc_profile"]).run_ct026_ppc_profile, lambda c: sha256(canonical_config_json(c).encode()).hexdigest(), lambda c: sha256(json.dumps(fingerprint_source_files(c, "spike_phase"), sort_keys=True).encode()).hexdigest(), production_git_fingerprint, __import__("time").monotonic, _production_run_lock, recover_ct026_profile_work)


def slice_ct026_profile_job(*, config: object, profile_job: RepresentativePPCProfileJob, scenario: str, unit_ids: tuple[str, ...], phase: object, spikes: object) -> CT026PPCProfileSlice:
    """Slice exact site/trial/unit PPC inputs, preserving stable IDs and overlaps."""
    positions = [int(np.flatnonzero(np.asarray(phase.trial_indices) == trial_id)[0]) for trial_id in profile_job.trial_indices]
    site_index = [site.stable_id for site in config.sites].index(profile_job.site_id)
    valid_positions = [p for p in positions if bool(phase.site_valid[site_index, p])]
    stable = tuple(int(phase.trial_indices[p]) for p in valid_positions)
    unit_positions = [spikes.unit_ids.index(unit) for unit in unit_ids]
    trains = tuple(spikes.trial_spike_trains[p] for p in unit_positions)
    overlap = sorted(set(stable).intersection(*(set(map(int, train.overlap_trial_indices)) for train in trains))) if trains else []
    prepared_phase = SimpleNamespace(phase_tensor=np.asarray(phase.phase_tensor)[site_index:site_index+1, :, valid_positions, :], phase_valid=np.asarray(phase.phase_valid)[site_index:site_index+1, :, valid_positions, :], relative_time_s=getattr(phase, "relative_time_s", np.array([0.0, 1.0])), trial_indices=np.asarray(stable, dtype=np.int64))
    prepared_spikes = SimpleNamespace(unit_ids=unit_ids, trial_spike_trains=trains)
    seed = lfp_summary_runtime._ppc_schedule_seed(config, 0, site_index, 0)
    schedule = lfp_summary_runtime._shared_derangement_schedule(len(stable), 100, seed)
    return CT026PPCProfileSlice(scenario, unit_ids, stable, profile_job.site_id, profile_job.epoch_bounds_s, tuple(overlap), prepared_phase.phase_tensor, prepared_phase.phase_valid, prepared_phase, prepared_spikes, schedule)


def run_isolated_ct026_profile_job(*, job: CT026PPCProfileSlice, shuffle_count: int, work_root: Path, process_launcher: Callable[..., Mapping[str, object]] | None = None) -> dict[str, object]:
    """Run one child seam and retain only scalar elapsed/RSS metrics."""
    launcher = process_launcher or _default_child_launcher
    result = launcher(_profile_child_worker, {"job": job, "shuffle_count": shuffle_count, "work_root": Path(work_root)})
    if "exit_code" in result and int(result["exit_code"]) != 0:
        raise RuntimeError(f"isolated PPC child exit {result['exit_code']}: {result.get('error', '')}")
    return {"total_elapsed_seconds": float(result["total_elapsed_seconds"]), "peak_memory_bytes": int(result["ru_maxrss"]) * 1024, "peak_memory_source": _RSS_SOURCE}


def _profile_child_worker(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Execute existing PPC profiler in a fresh process and return scalar metrics."""
    raise RuntimeError("production child launcher must supply configured profile inputs")


def _default_child_launcher(target: Callable[..., Mapping[str, object]], payload: Mapping[str, object]) -> Mapping[str, object]:
    """Reject implicit in-process execution; production callers inject process transport."""
    del target, payload
    raise RuntimeError("a fresh-process launcher is required for CT026 PPC profiling")


def recover_ct026_profile_work(*, work_root: Path, identity: Mapping[str, str]) -> None:
    """Audit direct PPC fingerprint directories and delegate exact stale-lock recovery."""
    del identity
    root = Path(work_root) / "ppc"
    if not root.is_dir(): return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for name in ("executor.lock", "writer.lock"):
            lock = directory / name
            if lock.exists(): lfp_summary_work_cache.recover_stale_lock(lock, directory.name)


def production_git_fingerprint(*, repository_root: Path | None = None, head_reader: Callable[[], str] | None = None, tracked_source_reader: Callable[[], Mapping[str, str]] | None = None) -> str:
    """Hash HEAD plus relevant tracked neural source content, excluding untracked files."""
    head = (head_reader or (lambda: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository_root, text=True).strip()))()
    sources = (tracked_source_reader or (lambda: {str(path): path.read_text(encoding="utf-8") for path in Path(repository_root or Path.cwd()).glob("src/neural_analysis/*.py")}))()
    return sha256(json.dumps({"head": head, "sources": dict(sources)}, sort_keys=True).encode()).hexdigest()


def _production_run_lock(*, analysis_root: Path, identity: Mapping[str, str], run_directory: Path | None) -> AbstractContextManager[None]:
    """Use a global new-run lock or exact resumed-run lock with OS identity."""
    directory = Path(run_directory) if run_directory is not None else Path(analysis_root)
    directory.mkdir(parents=True, exist_ok=True)
    return acquire_ct026_profile_run_lock(directory, identity, pid=os.getpid(), hostname=socket.gethostname(), process_exists=lambda pid: pid == os.getpid())
