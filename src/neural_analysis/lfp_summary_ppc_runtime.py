"""Serial, restartable sufficient-statistic PPC execution for one job.

This module owns work-only checkpoints. It never writes a final component or
manifest, and never keeps a whole-session shuffle tensor in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Callable, Iterator, Mapping, Sequence

import numpy as np

from src.neural_analysis import spike_lfp_summary
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    PPCExecutionConfig,
    ProgressEvent,
    _validate_ppc_execution,
    component_fingerprint,
    fingerprint_source_files,
)
from src.neural_analysis.lfp_summary_work_cache import (
    _ownership_record,
    _write_lock_exclusive,
    write_ppc_checkpoint,
)


_FLOAT_FIELDS = (
    "ppc", "resultant_length", "preferred_phase_rad", "p_value", "q_value",
    "null_mean", "null_std", "null_p025", "null_p50", "null_p975",
)
_INTEGER_FIELDS = (
    "spike_count", "eligible_trial_count", "null_exceedance_count",
    "permutation_count",
)
_BOOLEAN_FIELDS = ("computable", "reliable", "null_eligible", "significant")
_SUMMARY_FIELDS = _FLOAT_FIELDS + _INTEGER_FIELDS + _BOOLEAN_FIELDS


@dataclass(frozen=True)
class PPCExecutionResult:
    """Work-only result for one selected condition/site/epoch PPC job.

    Attributes
    ----------
    run_fingerprint : str
        SHA-256 work identity binding the selected trials, source and phase
        representation, unit population, scientific settings, schedule, and
        execution representation. It has no physical units.
    run_directory : pathlib.Path
        Exact ``ppc/<run_fingerprint>`` work directory. It is absent when
        checkpoints are disabled.
    schedule : numpy.ndarray
        Owned int64 ``(shuffle, trial)`` source-to-target derangement matrix.
        It is empty with shape ``(0, trial)`` when fewer than two trials exist.
    summary_arrays : dict[str, numpy.ndarray]
        Owned arrays with axes ``(unit, frequency)``. ``ppc``, resultant
        length, p/q values, and null moments/percentiles are dimensionless;
        preferred phase is radians; count fields are int64 spikes, trials, or
        shuffles; flags are Boolean. Unavailable numerical entries are NaN and
        unavailable/ineligible flags are false. No raw phase or full-session
        shuffle-by-unit result is returned.
    completed_block_ids, resumed_block_ids : tuple[str, ...]
        Stable unit-block identities completed by this execution and the valid
        subset loaded from work checkpoints, respectively.
    """

    run_fingerprint: str
    run_directory: Path
    schedule: np.ndarray
    summary_arrays: dict[str, np.ndarray]
    completed_block_ids: tuple[str, ...]
    resumed_block_ids: tuple[str, ...]


@dataclass(frozen=True)
class _PhaseWorkDescriptor:
    """Read-only on-disk phase inputs shared by process workers.

    ``phase_path`` and ``valid_path`` name validated NPY arrays with axes
    ``(site=1, frequency, trial, time)``. Phase is complex64 unit vectors;
    validity is Boolean. Time remains in seconds in the lightweight task
    because it is a one-dimensional coordinate rather than the large payload.
    """

    phase_path: Path
    valid_path: Path


@dataclass(frozen=True)
class _ParallelBlockTask:
    """Pickle-safe, phase-free description of one contiguous unit block.

    ``trains`` contains finite event-relative spike-time arrays in seconds,
    with ``(unit in block, trial)`` nesting. ``schedule`` has int64
    ``(shuffle, trial)`` axes and ``frequencies_hz`` has shape ``(frequency,)``.
    No phase tensor or validity mask is embedded in this task.
    """

    block_id: str
    trains: tuple[tuple[np.ndarray, ...], ...]
    time_s: np.ndarray
    frequencies_hz: np.ndarray
    schedule: np.ndarray
    execution: PPCExecutionConfig


@dataclass(frozen=True)
class _ParallelBlockResult:
    """One worker's in-memory summary arrays for its named unit block."""

    block_id: str
    arrays: dict[str, np.ndarray]


def execute_ppc_blocks(
    *,
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    schedule: np.ndarray | None,
    work_root: Path,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> PPCExecutionResult:
    """Execute one serial PPC job with bounded restart checkpoints.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated scientific configuration. ``config.phase.frequency_hz`` is
        the exact ascending Hz coordinate used by the phase frequency axis;
        PPC reliability, shuffle, seed, and FDR settings define the estimator.
    execution : PPCExecutionConfig
        Validated execution-only block sizes, checkpoint policy, and progress
        cadence. These values affect work identity and memory only, never the
        final component scientific fingerprint.
    prepared_phase : object
        Must expose complex64 ``phase_tensor`` and Boolean ``phase_valid`` with
        identical ``(site=1, frequency, trial, time)`` axes, float64 finite
        increasing ``relative_time_s`` in event-relative seconds with shape
        ``(time,)``, and int64 stable ``trial_indices`` with shape ``(trial,)``.
        Optional ``site_id``, ``condition_name``, ``epoch_bounds_s`` in seconds,
        and source/representation fingerprints refine job identity. Invalid
        phase samples are marked by ``phase_valid`` rather than inferred from
        a collapsed mask.
    prepared_spikes : object
        Must expose unique stable ``unit_ids`` and ``trial_spike_trains``. Each
        train exposes finite event-relative seconds arrays on ``(trial,)`` and
        integer overlap trial identities. A physical spike may occur in more
        than one trial-local array when windows overlap.
    schedule : numpy.ndarray or None
        Caller-owned int64 ``(shuffle, trial)`` derangements. ``None`` creates
        one schedule from the saved PPC seed; the returned result owns its copy.
    work_root : pathlib.Path
        Session-local parent for execution-only work artifacts. No final cache
        component or manifest is written below this path by this function.
    progress_callback : callable or None
        Receives ``ProgressEvent`` records with this job's stable ``job_id``;
        records contain counts/time metadata only, never raw arrays.

    Returns
    -------
    PPCExecutionResult
        Exact observed and null summaries on ``(unit, frequency)`` axes. PPC
        preserves negative values; unavailable numerical outputs are NaN.
        Checkpoints contain summary arrays only and are never scientific cache
        components.

    Raises
    ------
    ValueError
        If phase, spike, time, frequency, trial, or schedule contracts differ.
    FileExistsError
        If another writer holds the exact ``executor.lock``.
    OSError
        If schedule/checkpoint creation, validation, or atomic replacement
        fails. Such a failure cannot publish a final scientific component.
    """
    # Validate execution settings before inspecting phase inputs or creating a
    # work directory.  In particular, an invalid process count must be a
    # side-effect-free programming error.
    _validate_ppc_execution(execution)
    phase, valid, time_s, frequencies_hz, trial_indices = _validated_phase_inputs(
        config, prepared_phase
    )
    trains, unit_ids, overlap_indices = _validated_spike_inputs(
        prepared_spikes, phase.shape[2]
    )
    schedule_array = _resolve_schedule(config, schedule, phase.shape[2])
    metadata = _build_run_metadata(
        config,
        execution,
        prepared_phase,
        prepared_spikes,
        phase,
        valid,
        time_s,
        frequencies_hz,
        trial_indices,
        trains,
        unit_ids,
        overlap_indices,
        schedule_array,
    )
    run_fingerprint = str(metadata["run_fingerprint"])
    run_directory = Path(work_root) / "ppc" / run_fingerprint
    progress = _ProgressReporter(progress_callback, str(metadata["job_id"]))
    progress.emit("prepare_phase", 1, 1, "validated prepared phase")

    if not execution.checkpoint_enabled:
        summary = _compute_unit_blocks(
            trains, phase, valid, time_s, frequencies_hz, schedule_array,
            execution, progress, None,
        )
        _apply_bh_and_significance(summary, config.ppc.fdr_alpha)
        progress.emit("fdr", 1, 1, "adjusted p values", timed=True)
        progress.emit("commit", 1, 1, "PPC summaries ready")
        return PPCExecutionResult(
            run_fingerprint, run_directory, schedule_array.copy(), summary, (), ()
        )

    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_executor_lock(run_directory, run_fingerprint)
    try:
        schedule_matches = _stored_schedule_matches(
            run_directory,
            schedule_array,
        )
        if not schedule_matches:
            _write_schedule(run_directory, schedule_array)
        resumed = _load_resumable_blocks(
            run_directory,
            metadata,
            schedule_matches,
            len(trains),
            frequencies_hz.size,
        )
        summary = _empty_summary_arrays(len(trains), frequencies_hz.size)
        for block_id, arrays in resumed.items():
            start, stop = _block_bounds(block_id)
            _copy_block_into_summary(summary, arrays, start, stop)
        resumed_ids = tuple(sorted(resumed))

        def checkpoint(
            block_id: str,
            arrays: Mapping[str, np.ndarray],
            complete_ids: tuple[str, ...],
        ) -> None:
            """Persist one validated unit block while the executor owns its lock."""
            block_metadata = dict(metadata)
            block_metadata["completed_block_ids"] = list(complete_ids)
            write_ppc_checkpoint(run_directory, block_id, arrays, block_metadata)

        if execution.worker_count == 1:
            summary = _compute_unit_blocks(
                trains, phase, valid, time_s, frequencies_hz, schedule_array,
                execution, progress, checkpoint, summary, set(resumed_ids),
            )
        else:
            summary = _compute_parallel_unit_blocks(
                trains=trains,
                phase=phase,
                valid=valid,
                time_s=time_s,
                frequencies_hz=frequencies_hz,
                schedule=schedule_array,
                execution=execution,
                run_directory=run_directory,
                progress=progress,
                checkpoint_writer=checkpoint,
                initial_summary=summary,
                resumed_block_ids=set(resumed_ids),
            )
        _apply_bh_and_significance(summary, config.ppc.fdr_alpha)
        all_ids = _block_ids(len(trains), execution.unit_block_size)
        for block_id in all_ids:
            start, stop = _block_bounds(block_id)
            checkpoint(block_id, _summary_block(summary, start, stop), all_ids)
        _atomic_json(
            run_directory / "complete.json",
            {"run_fingerprint": run_fingerprint},
        )
        progress.emit("fdr", 1, 1, "adjusted p values", timed=True)
        progress.emit("checkpoint", 1, 2, "wrote validated checkpoints", timed=True)
        progress.emit("checkpoint", 2, 2, "checkpoint set complete", timed=True)
        progress.emit("commit", 1, 1, "PPC summaries ready")
        return PPCExecutionResult(
            run_fingerprint, run_directory, schedule_array.copy(),
            {name: values.copy() for name, values in summary.items()}, all_ids,
            resumed_ids,
        )
    finally:
        _release_executor_lock(lock_path)


def _validated_phase_inputs(
    config: LFPSummaryConfig, prepared_phase: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact complex phase axes without repeating frequency data."""
    phase = np.asarray(prepared_phase.phase_tensor)
    valid = np.asarray(prepared_phase.phase_valid, dtype=bool)
    time_s = np.asarray(prepared_phase.relative_time_s, dtype=float)
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    trial_indices = np.asarray(
        getattr(prepared_phase, "trial_indices", ()),
        dtype=np.int64,
    )
    expected_shape = (1, frequencies_hz.size, trial_indices.size, time_s.size)
    if phase.dtype != np.dtype(np.complex64) or phase.shape != expected_shape:
        raise ValueError("prepared PPC phase frequency and axes must match configuration")
    if valid.shape != phase.shape or time_s.ndim != 1 or trial_indices.ndim != 1:
        raise ValueError("prepared PPC validity, trial, or time axes are invalid")
    if time_s.size and (
        not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0)
    ):
        raise ValueError("prepared PPC time axis must be finite and strictly increasing")
    return phase, valid, time_s, frequencies_hz, trial_indices


def _validated_spike_inputs(
    prepared_spikes: object, trial_count: int
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
]:
    """Validate trial-local seconds arrays and preserve overlap identities."""
    unit_ids = tuple(str(value) for value in prepared_spikes.unit_ids)
    trains: list[tuple[np.ndarray, ...]] = []
    overlaps: list[tuple[int, ...]] = []
    for train in prepared_spikes.trial_spike_trains:
        times = tuple(
            np.asarray(value, dtype=float) for value in train.relative_spike_times
        )
        if len(times) != trial_count or any(
            value.ndim != 1 or not np.isfinite(value).all() for value in times
        ):
            raise ValueError("prepared PPC spikes must be finite (unit, trial) seconds arrays")
        trains.append(times)
        overlaps.append(
            tuple(
                int(value)
                for value in np.asarray(train.overlap_trial_indices, dtype=np.int64)
            )
        )
    if len(trains) != len(unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("prepared PPC units must have unique matching stable identifiers")
    return tuple(trains), unit_ids, tuple(overlaps)


def _resolve_schedule(
    config: LFPSummaryConfig, schedule: np.ndarray | None, trial_count: int
) -> np.ndarray:
    """Return one validated int64 derangement schedule for the selected trials."""
    if schedule is None:
        if trial_count < 2:
            return np.empty((0, trial_count), dtype=np.int64)
        return generate_trial_derangement_schedule(
            trial_count, config.ppc.shuffle_count, seed=config.ppc.seed
        )
    candidate = np.asarray(schedule)
    if trial_count < 2 and candidate.shape == (0, trial_count):
        return candidate.astype(np.int64, copy=True)
    return spike_lfp_summary._derangement_schedule(candidate, trial_count)


def _build_run_metadata(
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    phase: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    trial_indices: np.ndarray,
    trains: tuple[tuple[np.ndarray, ...], ...],
    unit_ids: tuple[str, ...],
    overlap_indices: tuple[tuple[int, ...], ...],
    schedule: np.ndarray,
) -> dict[str, object]:
    """Build complete canonical identity metadata for one resumable PPC job."""
    site_id = str(getattr(prepared_phase, "site_id", "site-0"))
    condition_name = str(getattr(prepared_phase, "condition_name", "condition"))
    epoch_bounds = tuple(float(value) for value in getattr(prepared_phase, "epoch_bounds_s", ()))
    job_id = ":".join((site_id, condition_name, *(str(value) for value in epoch_bounds)))
    source = getattr(prepared_phase, "source_fingerprint", None)
    source_fingerprint = source if isinstance(source, str) else _hash_json(fingerprint_source_files(config, "spike_phase"))
    representation = getattr(prepared_phase, "representation_fingerprint", None)
    representation_fingerprint = representation if isinstance(representation, str) else _array_fingerprint(phase, valid)
    populations = tuple(str(value) for value in getattr(prepared_spikes, "population_ids", ()))
    metadata: dict[str, object] = {
        "job_id": job_id,
        "source_fingerprint": source_fingerprint,
        "scientific_fingerprint": component_fingerprint("spike_phase", config),
        "representation_fingerprint": representation_fingerprint,
        "site_id": site_id,
        "condition_name": condition_name,
        "epoch_bounds_s": list(epoch_bounds),
        "trial_indices": trial_indices.tolist(),
        "unit_ids": list(unit_ids),
        "population_id": populations[0] if populations else "selected_population",
        "spike_fingerprint": _hash_spikes(trains, overlap_indices),
        "overlap_trial_indices": [list(row) for row in overlap_indices],
        "schedule_seed": int(config.ppc.seed),
        "schedule_fingerprint": _array_fingerprint(schedule),
        "schema_version": config.schema_version,
        "code_version": "wp5c-4-v1",
        "axes": ["site", "frequency", "trial", "time"],
        "shapes": {"phase": list(phase.shape), "valid": list(valid.shape), "schedule": list(schedule.shape)},
        "dtypes": {"phase": str(phase.dtype), "valid": str(valid.dtype), "schedule": str(schedule.dtype)},
        "execution_settings": {
            "unit_block_size": execution.unit_block_size,
            "shuffle_block_size": execution.shuffle_block_size,
            "trial_edge_block_size": execution.trial_edge_block_size,
            "worker_count": execution.worker_count,
            "checkpoint_enabled": execution.checkpoint_enabled,
            "checkpoint_retention": execution.checkpoint_retention,
            "progress_update_interval": execution.progress_update_interval,
        },
        "phase_content_fingerprint": _array_fingerprint(phase, valid),
        "completed_block_ids": [],
    }
    identity = dict(metadata)
    identity.pop("completed_block_ids")
    metadata["run_fingerprint"] = _hash_json(identity)
    return metadata


def _compute_unit_blocks(
    trains: tuple[tuple[np.ndarray, ...], ...],
    phase: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    schedule: np.ndarray,
    execution: PPCExecutionConfig,
    progress: "_ProgressReporter",
    checkpoint_writer: Callable[[str, Mapping[str, np.ndarray], tuple[str, ...]], None] | None,
    initial_summary: dict[str, np.ndarray] | None = None,
    resumed_block_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute/reuse deterministic unit blocks while retaining bounded draws only."""
    summary = _empty_summary_arrays(len(trains), frequencies_hz.size) if initial_summary is None else initial_summary
    block_ids = _block_ids(len(trains), execution.unit_block_size)
    resumed = resumed_block_ids or set()
    progress.emit("observed_reduction", 0, len(block_ids), "starting observed reduction")
    for completed, block_id in enumerate(block_ids, start=1):
        start, stop = _block_bounds(block_id)
        if block_id not in resumed:
            block = _empty_summary_arrays(stop - start, frequencies_hz.size)
            _compute_observed_block(
                block,
                trains[start:stop],
                phase,
                valid,
                time_s,
                frequencies_hz,
                execution.trial_edge_block_size,
            )
            _compute_null_block(block, trains[start:stop], phase, valid, time_s, frequencies_hz, schedule, execution)
            _copy_block_into_summary(summary, block, start, stop)
            if checkpoint_writer is not None:
                checkpoint_writer(block_id, block, tuple(sorted(resumed | set(block_ids[:completed]))))
        progress.emit("observed_reduction", completed, len(block_ids), "completed unit block", timed=True)
        progress.emit("trial_edge_reduction", completed, len(block_ids), "reduced scheduled edges", timed=True)
        progress.emit("shuffle_aggregation", completed, len(block_ids), "aggregated shuffled PPC", timed=True)
    return summary


def _compute_parallel_unit_blocks(
    *,
    trains: tuple[tuple[np.ndarray, ...], ...],
    phase: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    schedule: np.ndarray,
    execution: PPCExecutionConfig,
    run_directory: Path,
    progress: "_ProgressReporter",
    checkpoint_writer: Callable[[str, Mapping[str, np.ndarray], tuple[str, ...]], None],
    initial_summary: dict[str, np.ndarray],
    resumed_block_ids: set[str],
) -> dict[str, np.ndarray]:
    """Compute pending unit blocks in processes while the parent owns output.

    Inputs have the same axes and units as :func:`_compute_unit_blocks`.  The
    parent alone emits progress and writes checkpoints.  Worker tasks contain
    only their unit-local spike arrays and scalar/coordinate inputs; workers
    open the common complex64/Boolean phase arrays read-only from ``.npy``.
    """
    block_ids = _block_ids(len(trains), execution.unit_block_size)
    pending = tuple(block_id for block_id in block_ids if block_id not in resumed_block_ids)
    descriptor = _materialize_phase_work_inputs(run_directory, phase, valid)
    tasks = tuple(
        _ParallelBlockTask(
            block_id=block_id,
            trains=trains[_block_bounds(block_id)[0]:_block_bounds(block_id)[1]],
            time_s=time_s,
            frequencies_hz=frequencies_hz,
            schedule=schedule,
            execution=execution,
        )
        for block_id in pending
    )
    compute_block = partial(_compute_parallel_block, phase_descriptor=descriptor)
    results = iter(
        _run_parallel_worker_batches(
            phase_descriptor=descriptor,
            block_tasks=tasks,
            worker_count=execution.worker_count,
            compute_block=compute_block,
        )
    )
    completed_ids = set(resumed_block_ids)
    progress.emit("observed_reduction", 0, len(block_ids), "starting observed reduction")
    for completed, block_id in enumerate(block_ids, start=1):
        if block_id not in resumed_block_ids:
            result = next(results)
            if result.block_id != block_id:
                raise RuntimeError("parallel PPC worker returned blocks out of canonical order")
            start, stop = _block_bounds(block_id)
            _copy_block_into_summary(initial_summary, result.arrays, start, stop)
            completed_ids.add(block_id)
            # Checkpoint before requesting the next generator value. This
            # makes every yielded block independently resumable on a failure.
            checkpoint_writer(block_id, result.arrays, tuple(sorted(completed_ids)))
        progress.emit("observed_reduction", completed, len(block_ids), "completed unit block", timed=True)
        progress.emit("trial_edge_reduction", completed, len(block_ids), "reduced scheduled edges", timed=True)
        progress.emit("shuffle_aggregation", completed, len(block_ids), "aggregated shuffled PPC", timed=True)
    return initial_summary


def _materialize_phase_work_inputs(
    run_directory: Path,
    phase: np.ndarray,
    valid: np.ndarray,
) -> _PhaseWorkDescriptor:
    """Atomically materialize validated shared phase arrays for worker mmap.

    Parameters are complex64 and Boolean arrays on ``(site, frequency, trial,
    time)`` axes. The returned paths are under the exact locked run directory;
    they are execution-only inputs, not analysis components.
    """
    phase_path = run_directory / "worker_phase.npy"
    valid_path = run_directory / "worker_valid.npy"
    _atomic_npy(phase_path, phase)
    _atomic_npy(valid_path, valid)
    return _PhaseWorkDescriptor(phase_path=phase_path, valid_path=valid_path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    """Write and re-open one exact NPY array before atomic publication."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        loaded = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if loaded.dtype != values.dtype or loaded.shape != values.shape or not np.array_equal(loaded, values):
            raise ValueError("temporary PPC worker input validation failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compute_parallel_block(
    task: _ParallelBlockTask,
    *,
    phase_descriptor: _PhaseWorkDescriptor,
) -> _ParallelBlockResult:
    """Compute one pure worker block from read-only phase memory maps.

    ``task`` has unit-local trial spike arrays in seconds and no phase payload.
    The returned arrays have ``(unit in block, frequency)`` axes with the same
    documented units as the serial summary arrays.
    """
    phase = np.load(phase_descriptor.phase_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(phase_descriptor.valid_path, mmap_mode="r", allow_pickle=False)
    block = _empty_summary_arrays(len(task.trains), task.frequencies_hz.size)
    _compute_observed_block(
        block, task.trains, phase, valid, task.time_s, task.frequencies_hz,
        task.execution.trial_edge_block_size,
    )
    _compute_null_block(
        block, task.trains, phase, valid, task.time_s, task.frequencies_hz,
        task.schedule, task.execution,
    )
    return _ParallelBlockResult(task.block_id, block)


def _run_parallel_worker_batches(
    *,
    phase_descriptor: _PhaseWorkDescriptor,
    block_tasks: tuple[_ParallelBlockTask, ...],
    worker_count: int,
    compute_block: Callable[[_ParallelBlockTask], _ParallelBlockResult],
) -> Iterator[_ParallelBlockResult]:
    """Yield process-worker block results in input order, one at a time.

    At most ``worker_count`` tasks are submitted at once.  The initial window
    lets independent unit blocks use all requested workers, while ordered
    yielding lets the parent checkpoint each result before a replacement task
    is submitted.  On a worker failure, every submitted future is cancelled
    before the executor is shut down; already-yielded results remain available
    to the parent checkpoint writer.

    ``compute_block`` is a pickle-safe top-level-function partial in production
    and is injectable solely for deterministic executor-contract tests.
    """
    del phase_descriptor  # The descriptor is captured by the pickle-safe partial.
    task_iterator = iter(block_tasks)
    pending: list[object] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        # Fill the bounded initial submission window before waiting on work.
        for _ in range(worker_count):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.append(executor.submit(compute_block, task))

        while pending:
            # Retain the current future until its result has succeeded so a
            # failure can cancel both it and the remaining submission window.
            future = pending[0]
            try:
                result = future.result()
            except BaseException:
                for submitted_future in pending:
                    submitted_future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            pending.pop(0)
            yield result

            # The parent has now copied and checkpointed ``result``. Submit no
            # more than one replacement, retaining the bounded work window.
            try:
                task = next(task_iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(compute_block, task))


def _compute_observed_block(
    block: dict[str, np.ndarray], trains: Sequence[tuple[np.ndarray, ...]],
    phase: np.ndarray, valid: np.ndarray, time_s: np.ndarray, frequencies_hz: np.ndarray,
    trial_edge_block_size: int,
) -> None:
    """Populate observed PPC/count statistics for one bounded unit block."""
    phase_by_trial = np.moveaxis(phase[0], 1, 0)
    valid_by_trial = np.moveaxis(valid[0], 1, 0)
    for unit_index, spikes in enumerate(trains):
        _observed(
            block,
            unit_index,
            spikes,
            time_s,
            phase_by_trial,
            valid_by_trial,
            frequencies_hz,
            trial_edge_block_size,
        )


def _compute_null_block(
    block: dict[str, np.ndarray], trains: Sequence[tuple[np.ndarray, ...]],
    phase: np.ndarray, valid: np.ndarray, time_s: np.ndarray, frequencies_hz: np.ndarray,
    schedule: np.ndarray, execution: PPCExecutionConfig,
) -> None:
    """Infer each unit only over its reliable, two-trial frequency subset."""
    if schedule.size == 0:
        return
    phase_by_trial = np.moveaxis(phase[0], 1, 0)
    valid_by_trial = np.moveaxis(valid[0], 1, 0)
    for unit_index, spikes in enumerate(trains):
        eligible = block["reliable"][unit_index] & (block["eligible_trial_count"][unit_index] >= 2)
        positions = np.flatnonzero(eligible)
        if positions.size == 0:
            continue
        draws = _stream_null_draws(
            (spikes,), time_s, phase_by_trial[:, positions], valid_by_trial[:, positions],
            frequencies_hz[positions], schedule, execution,
        )[:, 0]
        null = spike_lfp_summary.summarize_permutation_null(
            observed_ppc=block["ppc"][unit_index, positions],
            null_ppc_chunks=(draws,),
            spike_count=block["spike_count"][unit_index, positions],
            eligible_trial_count=block["eligible_trial_count"][unit_index, positions],
        )
        for name in ("null_exceedance_count", "permutation_count", "p_value", "null_mean", "null_std", "null_p025", "null_p50", "null_p975", "null_eligible"):
            block[name][unit_index, positions] = getattr(null, name)


def _stream_null_draws(
    trains: Sequence[Sequence[np.ndarray]], time_s: np.ndarray, phase: np.ndarray,
    valid: np.ndarray, frequencies_hz: np.ndarray, schedule: np.ndarray,
    execution: PPCExecutionConfig,
) -> np.ndarray:
    """Stream edge and shuffle blocks into one current unit-block draw array."""
    source_edges, target_edges = _scheduled_edges(schedule)
    draws = np.full(
        (schedule.shape[0], len(trains), frequencies_hz.size),
        np.nan,
        dtype=float,
    )
    phase_vectors = np.where(valid, phase, 0.0j)
    # These current-unit-block accumulators let each costly edge be sampled
    # once, while schedule rows are still applied in bounded shuffle slices.
    vector_sum = np.zeros(draws.shape, dtype=np.complex128)
    valid_count = np.zeros(draws.shape, dtype=np.int64)
    for edge_start in range(0, source_edges.size, execution.trial_edge_block_size):
        edge_stop = min(edge_start + execution.trial_edge_block_size, source_edges.size)
        edge = spike_lfp_summary.compute_edge_sufficient_statistics(
            trial_relative_spike_times_s=trains,
            phase_time_s=time_s,
            trial_phase_vectors=phase_vectors,
            frequencies_hz=frequencies_hz,
            source_trial_position=source_edges[edge_start:edge_stop],
            target_trial_position=target_edges[edge_start:edge_stop],
        )
        for shuffle_start in range(0, schedule.shape[0], execution.shuffle_block_size):
            shuffle_stop = min(
                shuffle_start + execution.shuffle_block_size, schedule.shape[0]
            )
            schedule_block = schedule[shuffle_start:shuffle_stop]
            for edge_index, (source, target) in enumerate(
                zip(
                    edge.source_trial_position,
                    edge.target_trial_position,
                    strict=True,
                )
            ):
                rows = np.flatnonzero(
                    schedule_block[:, int(source)] == int(target)
                )
                vector_sum[shuffle_start + rows] += edge.phase_vector_sum[edge_index]
                valid_count[shuffle_start + rows] += edge.valid_spike_count[edge_index]
    np.divide(
        np.abs(vector_sum) ** 2 - valid_count,
        valid_count.astype(float) * (valid_count - 1),
        out=draws,
        where=valid_count >= 2,
    )
    return draws


def _observed(
    arrays: dict[str, np.ndarray], unit: int, spikes: tuple[np.ndarray, ...],
    time_s: np.ndarray, phase: np.ndarray, valid: np.ndarray, frequencies_hz: np.ndarray,
    trial_edge_block_size: int | None = None,
) -> None:
    """Compute observed same-trial PPC for one unit on every configured frequency."""
    positions = np.arange(len(spikes), dtype=np.int64)
    if positions.size == 0:
        return
    edge_size = positions.size if trial_edge_block_size is None else trial_edge_block_size
    count = np.zeros(frequencies_hz.size, dtype=np.int64)
    vector = np.zeros(frequencies_hz.size, dtype=np.complex128)
    for start in range(0, positions.size, edge_size):
        stop = min(start + edge_size, positions.size)
        edge = spike_lfp_summary.compute_edge_sufficient_statistics(
            trial_relative_spike_times_s=(spikes,), phase_time_s=time_s,
            trial_phase_vectors=np.where(valid, phase, 0.0j), frequencies_hz=frequencies_hz,
            source_trial_position=positions[start:stop], target_trial_position=positions[start:stop],
        )
        count += edge.valid_spike_count[:, 0].sum(axis=0, dtype=np.int64)
        vector += edge.phase_vector_sum[:, 0].sum(axis=0, dtype=np.complex128)
    ppc = np.full(count.shape, np.nan, dtype=float)
    np.divide(np.abs(vector) ** 2 - count, count.astype(float) * (count - 1), out=ppc, where=count >= 2)
    resultant = np.full(count.shape, np.nan, dtype=float)
    np.divide(np.abs(vector), count, out=resultant, where=count > 0)
    preferred = np.full(count.shape, np.nan, dtype=float)
    preferred[count > 0] = np.angle(vector[count > 0])
    arrays["ppc"][unit] = ppc
    arrays["resultant_length"][unit] = resultant
    arrays["preferred_phase_rad"][unit] = preferred
    arrays["spike_count"][unit] = count
    arrays["computable"][unit] = count >= 2
    arrays["reliable"][unit] = count >= 50
    # Re-evaluate one trial at a time, preserving trial-local overlap membership.
    eligible_trials = np.zeros(frequencies_hz.size, dtype=np.int64)
    for position in positions:
        edge = spike_lfp_summary.compute_edge_sufficient_statistics(
            trial_relative_spike_times_s=(spikes,), phase_time_s=time_s,
            trial_phase_vectors=np.where(valid, phase, 0.0j), frequencies_hz=frequencies_hz,
            source_trial_position=np.array([position]), target_trial_position=np.array([position]),
        )
        eligible_trials += edge.valid_spike_count[0, 0] > 0
    arrays["eligible_trial_count"][unit] = eligible_trials


def _apply_bh_and_significance(arrays: dict[str, np.ndarray], alpha: float) -> None:
    """Apply BH along frequency independently for each unit's eligible spectrum."""
    q_value = spike_lfp_summary.adjust_ppc_pvalues_bh(
        p_value=arrays["p_value"][:, None, None, None, :],
        null_eligible=arrays["null_eligible"][:, None, None, None, :],
    )[:, 0, 0, 0, :]
    arrays["q_value"] = q_value
    arrays["significant"] = arrays["null_eligible"] & (q_value <= alpha)


def _empty_summary_arrays(unit_count: int, frequency_count: int) -> dict[str, np.ndarray]:
    """Allocate documented work-only arrays with axes ``(unit, frequency)``."""
    shape = (unit_count, frequency_count)
    arrays = {name: np.full(shape, np.nan, dtype=float) for name in _FLOAT_FIELDS}
    arrays.update({name: np.zeros(shape, dtype=np.int64) for name in _INTEGER_FIELDS})
    arrays.update({name: np.zeros(shape, dtype=bool) for name in _BOOLEAN_FIELDS})
    return arrays


def _block_ids(unit_count: int, unit_block_size: int) -> tuple[str, ...]:
    """Return stable contiguous unit-block IDs in ascending unit order."""
    return tuple(
        f"unit-{start:06d}-{min(start + unit_block_size, unit_count) - 1:06d}"
        for start in range(0, unit_count, unit_block_size)
    )


def _block_bounds(block_id: str) -> tuple[int, int]:
    """Decode a stable inclusive unit-block ID to half-open unit bounds."""
    _, start, stop = block_id.split("-")
    return int(start), int(stop) + 1


def _summary_block(summary: Mapping[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    """Copy one unit slice for atomic checkpoint serialization."""
    return {name: np.asarray(values[start:stop]).copy() for name, values in summary.items()}


def _copy_block_into_summary(summary: dict[str, np.ndarray], block: Mapping[str, np.ndarray], start: int, stop: int) -> None:
    """Validate and copy one checkpoint's documented unit/frequency arrays."""
    for name in _SUMMARY_FIELDS:
        values = np.asarray(block[name])
        if values.shape != summary[name][start:stop].shape or values.dtype != summary[name].dtype:
            raise ValueError("PPC checkpoint summary axes or dtypes are invalid")
        summary[name][start:stop] = values


def _scheduled_edges(schedule: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique lexicographic nonself source-target pairs used by schedule."""
    pairs = sorted({(source, int(target)) for row in schedule for source, target in enumerate(row)})
    return (
        np.asarray([source for source, _ in pairs], dtype=np.int64),
        np.asarray([target for _, target in pairs], dtype=np.int64),
    )


def _stored_schedule_matches(run_directory: Path, schedule: np.ndarray) -> bool:
    """Return whether an existing schedule NPZ exactly matches this job schedule."""
    try:
        with np.load(run_directory / "schedule.npz", allow_pickle=False) as loaded:
            return set(loaded.files) == {"schedule"} and loaded["schedule"].dtype == np.dtype(np.int64) and np.array_equal(loaded["schedule"], schedule)
    except (OSError, ValueError):
        return False


def _load_resumable_blocks(
    run_directory: Path, metadata: Mapping[str, object], schedule_matches: bool,
    unit_count: int, frequency_count: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Load validated unit-block siblings without accepting partial work.

    ``metadata`` is the exact current JSON-safe job identity and
    ``schedule_matches`` confirms the persisted int64 ``(shuffle, trial)``
    schedule before any block is eligible for reuse. Each accepted NPZ has only
    documented ``(unit_block, frequency)`` float/int64/Boolean summary arrays.
    Corrupt, orphaned, markerless, wrong-axis, wrong-dtype, or incompatible
    blocks are ignored independently so valid siblings remain resumable.
    """
    if not schedule_matches:
        return {}
    try:
        stored = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    expected = dict(metadata)
    expected.pop("completed_block_ids")
    stored.pop("completed_block_ids", None)
    if stored != expected:
        return {}
    valid_blocks: dict[str, dict[str, np.ndarray]] = {}
    for marker_path in sorted((run_directory / "blocks").glob("*.complete.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            block_id = str(marker["block_id"])
            if marker != {"block_id": block_id, "run_fingerprint": metadata["run_fingerprint"]}:
                continue
            start, stop = _block_bounds(block_id)
            if start < 0 or stop > unit_count:
                continue
            with np.load(run_directory / "blocks" / f"{block_id}.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in loaded.files}
            if set(arrays) != set(_SUMMARY_FIELDS):
                continue
            template = _empty_summary_arrays(stop - start, frequency_count)
            _copy_block_into_summary(template, arrays, 0, stop - start)
            valid_blocks[block_id] = arrays
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return valid_blocks


def _acquire_executor_lock(
    run_directory: Path,
    run_fingerprint: str,
) -> Path:
    """Create a complete exclusive JSON lock before any exact-job mutation."""
    lock_path = run_directory / "executor.lock"
    _write_lock_exclusive(
        lock_path,
        _ownership_record(run_fingerprint, "executor"),
        f"PPC executor.lock already exists: {lock_path}",
    )
    return lock_path


def _release_executor_lock(lock_path: Path) -> None:
    """Remove only the exact lock created by this invocation."""
    lock_path.unlink(missing_ok=True)


def _write_schedule(run_directory: Path, schedule: np.ndarray) -> None:
    """Atomically write the validated int64 schedule while executor.lock exists."""
    temporary = run_directory / ".schedule.npz.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, schedule=schedule)
        with np.load(temporary, allow_pickle=False) as loaded:
            if not np.array_equal(loaded["schedule"], schedule):
                raise ValueError("temporary PPC schedule validation failed")
        os.replace(temporary, run_directory / "schedule.npz")
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a small JSON completion marker in the same directory."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_fingerprint(*arrays: np.ndarray) -> str:
    """Hash dtype, shape, and bytes for exact numerical work-input identity."""
    digest = sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _hash_spikes(trains: object, overlaps: object) -> str:
    """Hash trial-local spike seconds and overlap identities without object arrays."""
    digest = sha256()
    for unit in trains:
        for trial in unit:
            digest.update(_array_fingerprint(np.asarray(trial, dtype=float)).encode("ascii"))
    digest.update(_hash_json(overlaps).encode("ascii"))
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    """Return a SHA-256 digest for one JSON-safe canonical identity value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class _ProgressReporter:
    """Emit ordered job-scoped progress records without numerical payloads.

    ``job_id`` identifies one selected site/condition/epoch work item.
    ``emit`` carries nonnegative completed/total work units, elapsed seconds,
    and an ETA only after two timed updates in the same stage. It never changes
    phase, spike, unit, frequency, or missingness data.
    """

    callback: Callable[[ProgressEvent], None] | None
    job_id: str
    start_time: float = 0.0
    timed_by_stage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.start_time = time.perf_counter()
        self.timed_by_stage = {}

    def emit(self, stage: str, completed: int, total: int, message: str, *, timed: bool = False) -> None:
        """Forward one monotonic event without raw phase/spike data."""
        assert self.timed_by_stage is not None
        if timed:
            self.timed_by_stage[stage] = self.timed_by_stage.get(stage, 0) + 1
        elapsed = max(0.0, time.perf_counter() - self.start_time)
        eta = None
        if self.timed_by_stage.get(stage, 0) >= 2:
            eta = max(0.0, elapsed / completed * (total - completed))
        if self.callback is not None:
            self.callback(ProgressEvent("spike_phase", stage, completed, total, message, elapsed, eta, self.job_id))


generate_trial_derangement_schedule = spike_lfp_summary.generate_trial_derangement_schedule
# Test sentinel only: production execution must never call this full-draw helper.
_compute_scheduled_shuffle_draws = spike_lfp_summary._compute_scheduled_shuffle_draws
