"""Work-only, serial instrumentation for representative PPC profiling.

The profiler deliberately does not implement a PPC estimator or write cache
components.  A caller supplies the production-backed workload runner; this
module records bounded stage timing and sampled resident memory around it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import re
from typing import Callable

from src.neural_analysis.lfp_summary_models import PPCExecutionConfig


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
