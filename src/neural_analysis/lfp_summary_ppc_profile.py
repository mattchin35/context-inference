"""Work-only, serial instrumentation for representative PPC profiling.

The profiler deliberately does not implement a PPC estimator or write cache
components. It either records an injected descriptor runner or temporarily
instruments the existing production serial executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re
from typing import Callable

import numpy as np

from src.neural_analysis import lfp_summary_ppc_runtime
from src.neural_analysis.lfp_summary_models import LFPSummaryConfig, PPCExecutionConfig


_SAFE_WORKLOAD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_STAGES = (
    "phase_preparation",
    "edge_reduction",
    "shuffle_aggregation",
    "checkpoint",
)


@dataclass(frozen=True)
class PPCProfileWorkload:
    """Immutable synthetic PPC benchmark descriptor.

    ``trial_count`` counts selected trials, ``unit_count`` counts units, and
    ``frequency_count`` counts phase frequencies; each is a positive Python
    integer, never Boolean. ``shuffle_count`` is the positive number of
    derangement rows. ``spike_rate_hz`` is a positive finite expected spike
    rate in Hz. ``epoch_duration_s`` is positive finite seconds and defaults to
    the approved four-second Spike-phase epoch. ``seed`` is a nonnegative
    Python integer used by a future synthetic input builder. The descriptor has
    no arrays and creating it does not run phase preparation or PPC.
    """

    name: str
    trial_count: int
    unit_count: int
    frequency_count: int
    shuffle_count: int
    spike_rate_hz: float
    seed: int
    epoch_duration_s: float = 4.0

    def __post_init__(self) -> None:
        """Validate safe identities and finite positive benchmark dimensions."""
        if not isinstance(self.name, str) or not _SAFE_WORKLOAD_NAME.fullmatch(self.name):
            raise ValueError("name must be a safe non-empty workload identifier")
        for field_name in (
            "trial_count",
            "unit_count",
            "frequency_count",
            "shuffle_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(self.spike_rate_hz, (int, float)) or isinstance(self.spike_rate_hz, bool):
            raise ValueError("spike_rate_hz must be a finite positive Hz value")
        if not isfinite(float(self.spike_rate_hz)) or float(self.spike_rate_hz) <= 0.0:
            raise ValueError("spike_rate_hz must be a finite positive Hz value")
        if not isinstance(self.epoch_duration_s, (int, float)) or isinstance(self.epoch_duration_s, bool):
            raise ValueError("epoch_duration_s must be finite positive seconds")
        if not isfinite(float(self.epoch_duration_s)) or float(self.epoch_duration_s) <= 0.0:
            raise ValueError("epoch_duration_s must be finite positive seconds")


@dataclass(frozen=True)
class PPCProfileResult:
    """Measured serial PPC work for one descriptor.

    ``workload`` records the benchmark axes and units. All stage and total
    fields are elapsed seconds; an unreported optional checkpoint is ``None``.
    ``peak_memory_bytes`` is the maximum nonnegative sampled resident bytes.
    Count fields repeat the descriptor's unit/trial/shuffle axes. Throughput is
    ``unit * trial * shuffle / total_elapsed_seconds`` in operations per second.
    The result contains no phase, spike, null, or scientific component arrays.
    """

    workload: PPCProfileWorkload
    phase_preparation_seconds: float | None
    edge_reduction_seconds: float | None
    shuffle_aggregation_seconds: float | None
    checkpoint_overhead_seconds: float | None
    total_elapsed_seconds: float
    peak_memory_bytes: int
    unit_count: int
    trial_count: int
    shuffle_count: int
    worker_count: int
    throughput_unit_trial_shuffle_per_second: float


@dataclass(frozen=True)
class PPCProductionProfileResult:
    """Instrumentation result for one existing serial ``execute_ppc_blocks`` call.

    ``execution_result`` is the unchanged production summary-only result. All
    duration fields are nonnegative seconds measured around direct runtime
    callables. ``observed_reduction_seconds`` and
    ``shuffle_aggregation_seconds`` are inclusive wall times: each can overlap
    ``edge_reduction_seconds`` because observed and shuffle work invoke the edge
    reducer. They are therefore intentionally nonadditive. ``null_summary`` is
    measured separately and may also occur inside shuffle work. ``peak_memory``
    is the maximum value returned by the caller's RSS sampler and its source is
    explicitly ``'injected_rss_sampler'``. Counts use the production arrays:
    spike samples are raw trial-local events, phase tensor values are complex
    entries on ``(site, frequency, trial, time)``, and edge-valid samples sum
    ``valid_spike_count`` over every direct edge reducer result (including
    observed, eligibility re-evaluation, and null calls). No raw draws or final
    component artifacts are created by this dataclass or profiler.
    """

    execution_result: object
    phase_validation_seconds: float
    observed_reduction_seconds: float
    edge_reduction_seconds: float
    shuffle_aggregation_seconds: float
    null_summarization_seconds: float
    checkpoint_overhead_seconds: float
    total_elapsed_seconds: float
    peak_memory_bytes: int
    peak_memory_source: str
    worker_count: int
    edge_call_count: int
    unique_scheduled_edge_count: int
    eligible_unit_frequency_count: int
    spike_sample_count: int
    phase_tensor_value_count: int
    edge_valid_phase_sample_count: int
    checkpoint_block_count: int
    checkpoint_write_call_count: int


WorkloadRunner = Callable[[PPCProfileWorkload, Callable[[str], None], Path], None]


def representative_serial_ppc_workloads() -> tuple[PPCProfileWorkload, ...]:
    """Return deterministic low/median/high 249-trial profile descriptors.

    Each descriptor is a 249-trial, eight-unit, 50-frequency, 1,000-shuffle
    synthetic job on the approved four-second epoch. Expected rates are 2, 10,
    and 40 Hz, respectively. This function creates descriptors only; callers
    explicitly choose whether to build and execute the potentially expensive
    synthetic workloads.
    """
    return (
        PPCProfileWorkload("low", 249, 8, 50, 1000, 2.0, 5101),
        PPCProfileWorkload("median", 249, 8, 50, 1000, 10.0, 5102),
        PPCProfileWorkload("high", 249, 8, 50, 1000, 40.0, 5103),
    )


def profile_serial_ppc_workload(
    *,
    workload: PPCProfileWorkload,
    execution: PPCExecutionConfig,
    work_root: Path,
    workload_runner: WorkloadRunner,
    clock: Callable[[], float],
    memory_sampler: Callable[[], int],
) -> PPCProfileResult:
    """Run injected serial PPC work while recording completed stage measurements.

    Parameters
    ----------
    workload : PPCProfileWorkload
        Validated descriptor; its axes are unit/trial/frequency and its rate is
        Hz. It carries no raw experimental data.
    execution : PPCExecutionConfig
        Work-only settings. ``worker_count`` must be exactly integer ``1``;
        this module neither creates workers nor changes the production setting.
    work_root : pathlib.Path
        Session-local parent for a runner's work-only artifacts. The profiler
        passes ``work_root / 'ppc_profile' / workload.name`` without creating it.
    workload_runner : callable
        Receives ``(workload, stage_complete, work_directory)`` and performs
        the selected production-backed synthetic work. It calls
        ``stage_complete`` once after each completed ordered stage from
        ``phase_preparation``, ``edge_reduction``, ``shuffle_aggregation``, and
        optionally ``checkpoint``. It returns no value used by this profiler.
    clock : callable
        Returns finite monotonic elapsed seconds from an injected time source.
    memory_sampler : callable
        Returns a nonnegative integer resident-memory sample in bytes.

    Returns
    -------
    PPCProfileResult
        Stage/total seconds, peak bytes, dimension counts, and serial
        unit-trial-shuffle throughput. No final component, manifest, estimator,
        or scientific output is created by this function.

    Raises
    ------
    ValueError
        If serial execution, clock/memory samples, or stage order is invalid.
    """
    _validate_serial_execution(execution)
    if not isinstance(workload, PPCProfileWorkload):
        raise ValueError("workload must be a PPCProfileWorkload")
    if not callable(workload_runner) or not callable(clock) or not callable(memory_sampler):
        raise ValueError("runner, clock, and memory sampler must be callable")

    work_directory = Path(work_root) / "ppc_profile" / workload.name
    start_time = _read_clock(clock, "start")
    last_time = start_time
    samples = [_read_memory(memory_sampler)]
    stage_seconds: dict[str, float | None] = {stage: None for stage in _STAGES}
    next_stage_index = 0

    def stage_complete(stage: str) -> None:
        """Store one completed ordered stage using injected seconds and bytes."""
        nonlocal last_time, next_stage_index
        if next_stage_index >= len(_STAGES) or stage != _STAGES[next_stage_index]:
            raise ValueError("PPC profile stages must be completed once in approved order")
        now = _read_clock(clock, stage)
        if now < last_time:
            raise ValueError("PPC profile clock must be monotonic")
        stage_seconds[stage] = now - last_time
        last_time = now
        samples.append(_read_memory(memory_sampler))
        next_stage_index += 1

    workload_runner(workload, stage_complete, work_directory)
    end_time = _read_clock(clock, "end")
    if end_time < last_time:
        raise ValueError("PPC profile clock must be monotonic")
    samples.append(_read_memory(memory_sampler))
    total = end_time - start_time
    if total <= 0.0:
        raise ValueError("PPC profile total elapsed time must be positive")
    operations = workload.unit_count * workload.trial_count * workload.shuffle_count
    return PPCProfileResult(
        workload=workload,
        phase_preparation_seconds=stage_seconds["phase_preparation"],
        edge_reduction_seconds=stage_seconds["edge_reduction"],
        shuffle_aggregation_seconds=stage_seconds["shuffle_aggregation"],
        checkpoint_overhead_seconds=stage_seconds["checkpoint"],
        total_elapsed_seconds=total,
        peak_memory_bytes=max(samples),
        unit_count=workload.unit_count,
        trial_count=workload.trial_count,
        shuffle_count=workload.shuffle_count,
        worker_count=execution.worker_count,
        throughput_unit_trial_shuffle_per_second=operations / total,
    )


def profile_production_ppc_job(
    *,
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    schedule: np.ndarray | None,
    work_root: Path,
    clock: Callable[[], float],
    rss_sampler: Callable[[], int],
) -> PPCProductionProfileResult:
    """Profile exactly one existing serial PPC executor call without publication.

    Parameters
    ----------
    config, execution, prepared_phase, prepared_spikes, schedule, work_root
        The unchanged keyword arguments accepted by
        :func:`lfp_summary_ppc_runtime.execute_ppc_blocks`. Prepared phase has
        complex64/Boolean ``(site, frequency, trial, time)`` axes; prepared
        spikes contain trial-local seconds arrays. The profiler forwards these
        caller-owned objects exactly once and does not construct a final payload
        or manifest.
    clock : callable
        Finite monotonic seconds source. It is sampled around direct production
        functions and at the call boundary.
    rss_sampler : callable
        Nonnegative integer resident-memory bytes source. It is sampled beside
        every timing boundary; the result records its explicit source.

    Returns
    -------
    PPCProductionProfileResult
        The unmodified production result plus nested, deliberately nonadditive
        timings and exact counters. Temporary callable replacements are restored
        to their precise prior objects even when the executor raises.

    Raises
    ------
    ValueError
        If execution is nonserial or instrumentation samples are invalid.
    Exception
        Any exception from the unchanged production executor is propagated after
        temporary instrumentation has been restored.
    """
    _validate_serial_execution(execution)
    if not callable(clock) or not callable(rss_sampler):
        raise ValueError("clock and RSS sampler must be callable")

    samples: list[int] = []
    durations = {
        "phase_validation": 0.0,
        "observed": 0.0,
        "edge": 0.0,
        "shuffle": 0.0,
        "null": 0.0,
        "checkpoint": 0.0,
    }
    counters = {"edge_calls": 0, "edge_valid": 0, "checkpoint_writes": 0}
    checkpoint_ids: set[str] = set()

    def mark_memory() -> None:
        """Append one validated caller-supplied resident-byte sample."""
        samples.append(_read_memory(rss_sampler))

    def measure(name: str, function: Callable[..., object], *args: object, **kwargs: object) -> object:
        """Time one direct callable inclusively and retain an adjacent RSS sample."""
        before = _read_clock(clock, name)
        mark_memory()
        try:
            return function(*args, **kwargs)
        finally:
            after = _read_clock(clock, name)
            if after < before:
                raise ValueError("PPC profile clock must be monotonic")
            durations[name] += after - before
            mark_memory()

    originals = {
        "validation": lfp_summary_ppc_runtime._validated_phase_inputs,
        "observed": lfp_summary_ppc_runtime._compute_observed_block,
        "edge": lfp_summary_ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics,
        "shuffle": lfp_summary_ppc_runtime._stream_null_draws,
        "null": lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null,
        "checkpoint": lfp_summary_ppc_runtime.write_ppc_checkpoint,
    }

    def validation_wrapper(*args: object, **kwargs: object) -> object:
        """Time existing prepared-phase validation without changing its inputs."""
        return measure("phase_validation", originals["validation"], *args, **kwargs)

    def observed_wrapper(*args: object, **kwargs: object) -> object:
        """Time the inclusive observed unit-block reduction."""
        return measure("observed", originals["observed"], *args, **kwargs)

    def edge_wrapper(*args: object, **kwargs: object) -> object:
        """Count and time each exact sufficient-statistics reducer result once."""
        edge = measure("edge", originals["edge"], *args, **kwargs)
        counters["edge_calls"] += 1
        counters["edge_valid"] += int(np.asarray(edge.valid_spike_count).sum())
        return edge

    def shuffle_wrapper(*args: object, **kwargs: object) -> object:
        """Time inclusive scheduled shuffle aggregation around its direct helper."""
        return measure("shuffle", originals["shuffle"], *args, **kwargs)

    def null_wrapper(*args: object, **kwargs: object) -> object:
        """Time direct null-summary creation without retaining any draw arrays."""
        return measure("null", originals["null"], *args, **kwargs)

    def checkpoint_wrapper(*args: object, **kwargs: object) -> object:
        """Time checkpoint writes and count stable block identities, not retries."""
        if len(args) >= 2:
            checkpoint_ids.add(str(args[1]))
        elif "block_id" in kwargs:
            checkpoint_ids.add(str(kwargs["block_id"]))
        counters["checkpoint_writes"] += 1
        return measure("checkpoint", originals["checkpoint"], *args, **kwargs)

    start = _read_clock(clock, "start")
    mark_memory()
    lfp_summary_ppc_runtime._validated_phase_inputs = validation_wrapper
    lfp_summary_ppc_runtime._compute_observed_block = observed_wrapper
    lfp_summary_ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics = edge_wrapper
    lfp_summary_ppc_runtime._stream_null_draws = shuffle_wrapper
    lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null = null_wrapper
    lfp_summary_ppc_runtime.write_ppc_checkpoint = checkpoint_wrapper
    try:
        execution_result = lfp_summary_ppc_runtime.execute_ppc_blocks(
            config=config,
            execution=execution,
            prepared_phase=prepared_phase,
            prepared_spikes=prepared_spikes,
            schedule=schedule,
            work_root=work_root,
        )
    finally:
        lfp_summary_ppc_runtime._validated_phase_inputs = originals["validation"]
        lfp_summary_ppc_runtime._compute_observed_block = originals["observed"]
        lfp_summary_ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics = originals["edge"]
        lfp_summary_ppc_runtime._stream_null_draws = originals["shuffle"]
        lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null = originals["null"]
        lfp_summary_ppc_runtime.write_ppc_checkpoint = originals["checkpoint"]
    end = _read_clock(clock, "end")
    if end < start:
        raise ValueError("PPC profile clock must be monotonic")
    mark_memory()
    total = end - start
    if total <= 0.0:
        raise ValueError("PPC profile total elapsed time must be positive")

    phase_values = int(np.asarray(prepared_phase.phase_tensor).size)
    spike_count = sum(
        int(np.asarray(trial_times).size)
        for train in prepared_spikes.trial_spike_trains
        for trial_times in train.relative_spike_times
    )
    source_edges, _ = lfp_summary_ppc_runtime._scheduled_edges(execution_result.schedule)
    null_eligible = np.asarray(execution_result.summary_arrays["null_eligible"], dtype=bool)
    return PPCProductionProfileResult(
        execution_result=execution_result,
        phase_validation_seconds=durations["phase_validation"],
        observed_reduction_seconds=durations["observed"],
        edge_reduction_seconds=durations["edge"],
        shuffle_aggregation_seconds=durations["shuffle"],
        null_summarization_seconds=durations["null"],
        checkpoint_overhead_seconds=durations["checkpoint"],
        total_elapsed_seconds=total,
        peak_memory_bytes=max(samples),
        peak_memory_source="injected_rss_sampler",
        worker_count=execution.worker_count,
        edge_call_count=counters["edge_calls"],
        unique_scheduled_edge_count=int(source_edges.size),
        eligible_unit_frequency_count=int(null_eligible.sum()),
        spike_sample_count=spike_count,
        phase_tensor_value_count=phase_values,
        edge_valid_phase_sample_count=counters["edge_valid"],
        checkpoint_block_count=len(checkpoint_ids),
        checkpoint_write_call_count=counters["checkpoint_writes"],
    )


def _validate_serial_execution(execution: PPCExecutionConfig) -> None:
    """Reject non-serial or malformed worker settings before runner execution."""
    worker_count = execution.worker_count
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count != 1:
        raise ValueError("worker_count must be exactly 1 for serial PPC profiling")


def _read_clock(clock: Callable[[], float], label: str) -> float:
    """Return one finite seconds sample from the injected monotonic clock."""
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
        raise ValueError(f"PPC profile {label} clock sample must be finite seconds")
    return float(value)


def _read_memory(memory_sampler: Callable[[], int]) -> int:
    """Return one nonnegative integer resident-memory sample in bytes."""
    value = memory_sampler()
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("PPC profile memory samples must be nonnegative integer bytes")
    return value
