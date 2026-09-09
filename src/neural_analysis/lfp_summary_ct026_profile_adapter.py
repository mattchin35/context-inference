"""Production binding for work-only CT026 representative PPC profiling."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
import multiprocessing
import os
from pathlib import Path
import resource
import socket
import subprocess
import time
from types import SimpleNamespace
from typing import Callable, Mapping

import numpy as np

from src.neural_analysis import lfp_summary_runtime, lfp_summary_work_cache, spike_behavior_pynapple, unit_spike_loading
from src.neural_analysis.lfp_spike_phase_validation import build_ct026_default_active_population, build_ct026_spike_phase_preview_config
from src.neural_analysis.lfp_summary_ct026_profile_locks import acquire_ct026_profile_run_lock
from src.neural_analysis.lfp_summary_ct026_profile_runner import run_ct026_ppc_profile
from src.neural_analysis.lfp_summary_models import PPCExecutionConfig, canonical_config_json, fingerprint_source_files
from src.neural_analysis.lfp_summary_ppc_profile import RepresentativePPCProfileJob, profile_production_ppc_job, select_representative_ppc_profile_job

_SCENARIO_UNITS = {"low": lambda job: (job.low_unit_id,), "median": lambda job: (job.median_unit_id,), "high": lambda job: (job.high_unit_id,), "combined": lambda job: (job.low_unit_id, job.median_unit_id, job.high_unit_id)}
_RSS_SOURCE = "resource.getrusage(RUSAGE_SELF).ru_maxrss_kib"
_CHILD_RSS_SOURCE = "resource.getrusage(RUSAGE_CHILDREN).ru_maxrss_kib"


@dataclass(frozen=True)
class CT026PPCProfileSlice:
    """PPC child input; phase is ``(1, frequency, trial, time)``, spikes seconds."""
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
    config: object
    execution: PPCExecutionConfig


@dataclass(frozen=True)
class CT026PPCProfileAdapterDependencies:
    """All data and orchestration boundaries; deliberately no publish seam."""
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


def run_ct026_ppc_profile_adapter(*, session_path: Path, analysis_root: Path, dependencies: CT026PPCProfileAdapterDependencies, run_directory: Path | None = None) -> object:
    """Adapt CT026 loaders to runner-v2, which owns the sole run lock.

    The function writes no scientific component, preview, manifest, or NPZ.
    ``prepare_phase`` obeys the runner-provided cache root exactly.
    """
    if not isinstance(dependencies, CT026PPCProfileAdapterDependencies): raise ValueError("dependencies must be CT026PPCProfileAdapterDependencies")
    for name in dependencies.__dataclass_fields__:
        if not callable(getattr(dependencies, name)): raise ValueError(f"{name} must be callable")
    config = dependencies.build_config(Path(session_path), dependencies.load_active_population(Path(session_path)))
    identity = {"config_fingerprint": _identity(dependencies.config_fingerprint(config), "config_fingerprint"), "source_fingerprint": _identity(dependencies.source_fingerprint(config), "source_fingerprint"), "git_fingerprint": _identity(dependencies.git_fingerprint(), "git_fingerprint")}
    selected: RepresentativePPCProfileJob | None = None

    def prepare_phase(*, warm: bool, phase_work_root: Path) -> tuple[object, dict[str, object]]:
        """Prepare a cold/warm phase and measure this process's peak RSS bytes."""
        if not isinstance(warm, bool): raise ValueError("warm must be Boolean")
        start, rss = _clock(dependencies.monotonic_seconds), _self_rss_bytes()
        phase = dependencies.prepare_phase(config, work_cache_root=Path(phase_work_root))
        end = _clock(dependencies.monotonic_seconds)
        if end < start: raise ValueError("monotonic_seconds must not decrease")
        return phase, {"elapsed_seconds": end - start, "peak_memory_bytes": max(rss, _self_rss_bytes()), "peak_memory_source": _RSS_SOURCE, "cache_hit": warm}

    def select_spikes(phase: object) -> object:
        """Load spikes from the warm phase and select the deterministic job once."""
        nonlocal selected
        spikes = dependencies.prepare_spikes(config, phase)
        selected = dependencies.select_profile_job(config, phase, spikes)
        if not isinstance(selected, RepresentativePPCProfileJob): raise ValueError("select_profile_job must return RepresentativePPCProfileJob")
        return spikes

    def profile_job(*, scenario: str, phase: object, spikes: object, shuffle_count: int, work_root: Path) -> dict[str, object]:
        """Slice one scenario, recover only its PPC work locks, and fork it."""
        nonlocal selected
        if scenario not in _SCENARIO_UNITS or shuffle_count != 100: raise ValueError("CT026 PPC profiling requires known scenarios and exactly 100 shuffles")
        if selected is None:
            selected = dependencies.select_profile_job(config, phase, spikes)
            if not isinstance(selected, RepresentativePPCProfileJob): raise ValueError("select_profile_job must return RepresentativePPCProfileJob")
        root = Path(work_root)
        if run_directory is not None: dependencies.recover_profile_work(work_root=root, identity=dict(identity))
        job = dependencies.slice_profile_job(config=config, profile_job=selected, scenario=scenario, unit_ids=_SCENARIO_UNITS[scenario](selected), phase=phase, spikes=spikes)
        metrics = dependencies.run_isolated_profile_job(job=job, shuffle_count=shuffle_count, work_root=root)
        if not isinstance(metrics, Mapping): raise ValueError("isolated profile child must return scalar metrics")
        return {**dict(metrics), **identity}

    return dependencies.run_profile(config=config, analysis_root=Path(analysis_root), run_directory=run_directory, prepare_phase=prepare_phase, select_spikes=select_spikes, profile_job=profile_job, acquire_run_lock=dependencies.acquire_run_lock, **identity)


def _identity(value: object, name: str) -> str:
    """Validate one nonempty categorical run identity."""
    if not isinstance(value, str) or not value: raise ValueError(f"{name} must be a nonempty string")
    return value


def _clock(clock: Callable[[], float]) -> float:
    """Read finite elapsed seconds from a monotonic source."""
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)): raise ValueError("monotonic_seconds must return finite seconds")
    return float(value)


def _self_rss_bytes() -> int:
    """Convert Linux ``RUSAGE_SELF.ru_maxrss`` KiB to bytes."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _ct026_probe_b_sorter(session_path: Path) -> Path:
    """Return the established CT026 ProbeB sorter directory."""
    return Path(session_path) / "ephys/derived/Record_Node_101_Neuropix-PXI-110.ProbeB/kilosort4"


def make_production_ct026_profile_dependencies(*, work_cache_root: Path) -> CT026PPCProfileAdapterDependencies:
    """Bind existing CT026 preparation and serial production PPC APIs."""
    default_root = Path(work_cache_root)
    def population(session: Path) -> object:
        sorter = _ct026_probe_b_sorter(session)
        return build_ct026_default_active_population(session, lambda path: spike_behavior_pynapple.load_sorter_metadata(path)[1], lambda path: unit_spike_loading.load_channel_quality(path.parent))
    def config(session: Path, active: object) -> object:
        base = build_ct026_spike_phase_preview_config(session, active)
        return replace(base, ppc=replace(base.ppc, shuffle_count=100), ppc_execution=replace(base.ppc_execution, worker_count=1))
    def phase(value: object, *, work_cache_root: Path = default_root) -> object:
        return lfp_summary_runtime.prepare_phase_run(value, lfp_summary_runtime.load_configured_trial_table, work_cache_root=Path(work_cache_root))
    def spikes(value: object, prepared: object) -> object:
        return lfp_summary_runtime.prepare_spike_run(value, prepared, lfp_summary_runtime.load_configured_unit_spikes)
    return CT026PPCProfileAdapterDependencies(config, population, phase, spikes, select_representative_ppc_profile_job, slice_ct026_profile_job, run_isolated_ct026_profile_job, run_ct026_ppc_profile, lambda value: sha256(canonical_config_json(value).encode()).hexdigest(), lambda value: sha256(json.dumps(fingerprint_source_files(value, "spike_phase"), sort_keys=True).encode()).hexdigest(), production_git_fingerprint, time.monotonic, _production_run_lock, recover_ct026_profile_work)


def slice_ct026_profile_job(*, config: object, profile_job: RepresentativePPCProfileJob, scenario: str, unit_ids: tuple[str, ...], phase: object, spikes: object) -> CT026PPCProfileSlice:
    """Create one exact site/condition/whole-epoch executor job.

    Each selected train is independently half-open filtered to ``[-2, 2)`` s;
    its overlap IDs are retained independently while their intersection is also
    recorded for the scenario descriptor.
    """
    if profile_job.epoch_name != "whole" or profile_job.epoch_bounds_s != (-2.0, 2.0): raise ValueError("CT026 profile requires the whole [-2, 2) second epoch")
    requested = tuple(int(x) for x in profile_job.trial_indices)
    trial_axis = np.asarray(phase.trial_indices, dtype=np.int64)
    positions = tuple(int(np.flatnonzero(trial_axis == trial)[0]) for trial in requested if np.any(trial_axis == trial))
    if len(positions) != len(requested): raise ValueError("representative trial identity is absent from prepared phase")
    sites = tuple(site.stable_id for site in config.sites)
    if profile_job.site_id not in sites: raise ValueError("representative site is absent from configuration")
    site_index = sites.index(profile_job.site_id)
    valid_positions = tuple(p for p in positions if bool(np.asarray(phase.site_valid)[site_index, p]))
    if len(requested) != 249 and hasattr(phase, "prepared_trials"): raise ValueError("CT026 profile requires exactly 249 selected trials")
    if len(valid_positions) != len(requested): raise ValueError("representative CT026 trial slice must be site-valid")
    trials = tuple(int(trial_axis[p]) for p in valid_positions)
    trains = []
    for unit in unit_ids:
        try: source = spikes.trial_spike_trains[tuple(spikes.unit_ids).index(unit)]
        except ValueError as error: raise ValueError("selected unit is absent from prepared spikes") from error
        times = tuple(_half_open(np.asarray(source.relative_spike_times[p], dtype=float)) for p in valid_positions)
        trains.append(SimpleNamespace(unit_id=unit, relative_spike_times=times, overlap_trial_indices=np.asarray(source.overlap_trial_indices, dtype=np.int64)))
    overlap_sets = [set(map(int, train.overlap_trial_indices)) for train in trains]
    overlap = tuple(sorted(set(trials).intersection(*overlap_sets))) if overlap_sets else ()
    tensor = np.asarray(phase.phase_tensor)[site_index:site_index + 1, :, valid_positions, :].astype(np.complex64, copy=False)
    valid = np.asarray(phase.phase_valid, dtype=bool)[site_index:site_index + 1, :, valid_positions, :]
    source_id = getattr(phase, "source_identity", None) or getattr(phase, "source_fingerprint", None)
    relative_time_s = np.asarray(getattr(phase, "relative_time_s", np.array([-2.0, 2.0])), dtype=float)
    prepared_phase = SimpleNamespace(phase_tensor=tensor, phase_valid=valid, relative_time_s=relative_time_s, trial_indices=np.asarray(trials, dtype=np.int64), site_id=profile_job.site_id, condition_name=profile_job.condition_name, epoch_name="whole", epoch_bounds_s=(-2.0, 2.0), source_fingerprint=source_id, source_identity=source_id)
    prepared_spikes = SimpleNamespace(unit_ids=tuple(unit_ids), population_ids=(profile_job.population_id,), trial_spike_trains=tuple(trains))
    names = tuple(getattr(getattr(phase, "prepared_trials", None), "condition_names", ()))
    condition_index = names.index(profile_job.condition_name) if profile_job.condition_name in names else 0
    schedule = lfp_summary_runtime._shared_derangement_schedule(len(trials), 100, lfp_summary_runtime._ppc_schedule_seed(config, condition_index, site_index, 0))
    return CT026PPCProfileSlice(scenario, tuple(unit_ids), trials, profile_job.site_id, (-2.0, 2.0), overlap, tensor, valid, prepared_phase, prepared_spikes, schedule, config, config.ppc_execution)


def _half_open(values: np.ndarray) -> np.ndarray:
    """Return finite event-relative seconds within the whole half-open epoch."""
    return values[(values >= -2.0) & (values < 2.0)]


def run_isolated_ct026_profile_job(*, job: CT026PPCProfileSlice, shuffle_count: int, work_root: Path, process_launcher: Callable[..., Mapping[str, object]] | None = None) -> dict[str, object]:
    """Fork one production PPC profiler call and retain scalar child metrics only."""
    if shuffle_count != 100: raise ValueError("CT026 PPC profiling requires exactly 100 shuffles")
    result = (process_launcher or _default_child_launcher)(_profile_child_worker, {"job": job, "shuffle_count": shuffle_count, "work_root": Path(work_root)})
    if not isinstance(result, Mapping): raise ValueError("isolated child must return a metric mapping")
    if int(result.get("exit_code", 0)) != 0: raise RuntimeError(f"isolated PPC child exit {result['exit_code']}: {result.get('error', '')}")
    metrics = dict(result); metrics.pop("exit_code", None); metrics.pop("ru_maxrss_unit", None)
    if "ru_maxrss" not in metrics: raise ValueError("isolated child must report ru_maxrss in KiB")
    metrics["peak_memory_bytes"] = int(metrics.pop("ru_maxrss")) * 1024
    metrics["peak_memory_source"] = _CHILD_RSS_SOURCE
    return metrics


def _profile_child_worker(payload: Mapping[str, object]) -> Mapping[str, object]:
    """Execute ``profile_production_ppc_job`` exactly once and return scalars."""
    job = payload["job"]
    if not isinstance(job, CT026PPCProfileSlice): raise ValueError("child payload job must be CT026PPCProfileSlice")
    result = profile_production_ppc_job(config=job.config, execution=job.execution, prepared_phase=job.prepared_phase, prepared_spikes=job.prepared_spikes, schedule=job.schedule, work_root=Path(payload["work_root"]), clock=time.monotonic, rss_sampler=_self_rss_bytes)
    scalars = {name: value for name, value in vars(result).items() if isinstance(value, (str, int, float, bool)) and not isinstance(value, np.generic)}
    return {**scalars, "ru_maxrss": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "ru_maxrss_unit": "KiB", "child_pid": os.getpid()}


def _default_child_launcher(target: Callable[[Mapping[str, object]], Mapping[str, object]], payload: Mapping[str, object]) -> Mapping[str, object]:
    """Use Linux fork so a child does not return arrays or an executor result."""
    context = multiprocessing.get_context("fork"); receiver, sender = context.Pipe(duplex=False)
    def child() -> None:
        try: sender.send(dict(target(payload)))
        except BaseException as error: sender.send({"exit_code": 1, "error": f"{type(error).__name__}: {error}"})
        finally: sender.close()
    process = context.Process(target=child); process.start(); sender.close()
    result = receiver.recv() if receiver.poll(3600) else {"exit_code": 1, "error": "child produced no result"}
    process.join(); receiver.close()
    return result if process.exitcode in (0, None) or int(result.get("exit_code", 0)) else {"exit_code": process.exitcode, "error": "child process failed"}


def recover_ct026_profile_work(*, work_root: Path, identity: Mapping[str, str]) -> None:
    """Recover only direct PPC fingerprint locks, never delete blindly."""
    del identity
    root = Path(work_root) / "ppc"
    if not root.is_dir(): return
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        for name in ("executor.lock", "writer.lock"):
            lock = directory / name
            if lock.is_file() and not lock.is_symlink(): lfp_summary_work_cache.recover_stale_lock(lock, directory.name, hostname=socket.gethostname(), pid_is_alive=_local_process_exists)


def _local_process_exists(pid: int) -> bool:
    """Safely test local PID liveness; permission denied still means live."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0: return False
    try: os.kill(pid, 0)
    except ProcessLookupError: return False
    except PermissionError: return True
    return True


_production_process_exists = _local_process_exists


def production_git_fingerprint(*, repository_root: Path | None = None, head_reader: Callable[[], str] | None = None, tracked_source_reader: Callable[[], Mapping[str, str]] | None = None) -> str:
    """Hash HEAD and tracked neural Python source, excluding untracked files."""
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]
    head = (head_reader or (lambda: subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()))()
    if tracked_source_reader is None:
        files = subprocess.check_output(["git", "ls-files", "src/neural_analysis"], cwd=root, text=True).splitlines()
        sources = {name: (root / name).read_text(encoding="utf-8") for name in files if name.endswith(".py")}
    else: sources = tracked_source_reader()
    return sha256(json.dumps({"head": head, "sources": dict(sources)}, sort_keys=True).encode()).hexdigest()


def _production_run_lock(*, analysis_root: Path, identity: Mapping[str, str], run_directory: Path | None = None) -> AbstractContextManager[None]:
    """Acquire the generic runner's exact lock using safe OS PID liveness."""
    directory = Path(run_directory) if run_directory is not None else Path(analysis_root)
    directory.mkdir(parents=True, exist_ok=True)
    return acquire_ct026_profile_run_lock(directory, identity, pid=os.getpid(), hostname=socket.gethostname(), process_exists=_local_process_exists)
