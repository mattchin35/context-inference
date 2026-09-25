"""Work-only, serial instrumentation for representative PPC profiling.

The profiler deliberately does not implement a PPC estimator or write cache
components. It either records an injected descriptor runner or temporarily
instruments the existing production serial executor.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from pathlib import Path
import re
from typing import Callable, Mapping

import numpy as np

from src.neural_analysis.lfp_summary import ppc_execution as lfp_summary_ppc_runtime
from src.neural_analysis.lfp_summary.models import LFPSummaryConfig, PPCExecutionConfig


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


@dataclass(frozen=True)
class GroupedPPCProfileMetadata:
    """Scalar-only cost and memory metadata for one grouped PPC plan.

    Every count is dimensionless, byte field is bytes, and memory samples are
    resident bytes. Projection fields report the supplied same-workload
    100- and 1,000-shuffle plans; no plan, schedule, phase, spike, null, or
    histogram array is retained in this public profiling product.
    """

    schema_version: str
    profile_kind: str
    run_fingerprint: str
    scheduled_edge_count: int
    independent_edge_count: int
    unique_site_qualified_union_edge_count: int
    edge_union_saturation: float
    edge_reuse_ratio: float
    planned_parent_private_bytes: int
    planned_worker_private_bytes: int
    shared_phase_mmap_bytes: int
    planned_aggregate_array_bytes: int
    measured_peak_process_rss_bytes: int
    measured_peak_aggregate_rss_bytes: int | None
    measured_peak_aggregate_pss_bytes: int | None
    measured_memory_source: str
    projection_100_scheduled_edge_count: int
    projection_100_independent_edge_count: int
    projection_100_unique_site_qualified_union_edge_count: int
    projection_100_edge_union_saturation: float
    projection_100_edge_reuse_ratio: float
    projection_1000_scheduled_edge_count: int
    projection_1000_independent_edge_count: int
    projection_1000_unique_site_qualified_union_edge_count: int
    projection_1000_edge_union_saturation: float
    projection_1000_edge_reuse_ratio: float


@dataclass(frozen=True)
class GroupedPPCProfileResult:
    """Scalar serial timing plus grouped plan metadata for one component.

    Stage fields are seconds. Observed, shuffle, null, and representative
    histogram durations are inclusive and therefore deliberately nonadditive.
    Throughput divides scheduled physical edges by total elapsed seconds.
    """

    schema_version: str
    profile_kind: str
    run_fingerprint: str
    geometry_build_seconds: float
    observed_reduction_seconds: float
    union_edge_reduction_seconds: float
    shuffle_aggregation_seconds: float
    null_summarization_seconds: float
    representative_histogram_seconds: float
    checkpoint_overhead_seconds: float
    total_elapsed_seconds: float
    throughput_scheduled_edge_per_second: float
    scheduled_edge_count: int
    independent_edge_count: int
    unique_site_qualified_union_edge_count: int
    edge_union_saturation: float
    edge_reuse_ratio: float
    planned_parent_private_bytes: int
    planned_worker_private_bytes: int
    shared_phase_mmap_bytes: int
    planned_aggregate_array_bytes: int
    measured_peak_process_rss_bytes: int
    measured_peak_aggregate_rss_bytes: int | None
    measured_peak_aggregate_pss_bytes: int | None
    measured_memory_source: str
    projection_100_scheduled_edge_count: int
    projection_100_independent_edge_count: int
    projection_100_unique_site_qualified_union_edge_count: int
    projection_100_edge_union_saturation: float
    projection_100_edge_reuse_ratio: float
    projection_1000_scheduled_edge_count: int
    projection_1000_independent_edge_count: int
    projection_1000_unique_site_qualified_union_edge_count: int
    projection_1000_edge_union_saturation: float
    projection_1000_edge_reuse_ratio: float


@dataclass(frozen=True)
class RepresentativePPCProfileJob:
    """Immutable CT026 profile-job selection from prepared experimental inputs.

    ``condition_name`` and ``site_id`` are configured categorical identities;
    ``epoch_name`` is always ``"whole"`` and ``epoch_bounds_s`` gives its
    half-open event-relative seconds interval. ``trial_indices`` is an ordered
    tuple of exactly 249 stable trial-table row identities for the selected
    condition. It intentionally preserves the full condition selection rather
    than copying phase or spike arrays; ``site_valid_trial_count`` records how
    many of those rows have usable phase at ``site_id``.

    ``low_*``, ``median_*``, and ``high_*`` identify the nearest-rank 10th,
    50th, and 90th eligible units. Counts are raw spikes in the whole epoch
    across selected site-valid trials; rates are count divided by
    ``site_valid_trial_count * epoch_duration_s`` in Hz. ``trial_count``,
    ``site_valid_trial_count``, ``unit_count``, and ``frequency_count`` are
    provenance dimensions for the caller-owned prepared inputs. This descriptor
    runs neither a phase transform nor a PPC executor and owns no scientific
    array payloads.
    """

    condition_name: str
    site_id: str
    epoch_name: str
    epoch_bounds_s: tuple[float, float]
    trial_indices: tuple[int, ...]
    trial_count: int
    site_valid_trial_count: int
    unit_count: int
    frequency_count: int
    population_id: str
    low_unit_id: str
    median_unit_id: str
    high_unit_id: str
    low_spike_count: int
    median_spike_count: int
    high_spike_count: int
    low_spike_rate_hz: float
    median_spike_rate_hz: float
    high_spike_rate_hz: float


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


def select_representative_ppc_profile_job(
    config: LFPSummaryConfig,
    prepared_phase: object,
    prepared_spikes: object,
) -> RepresentativePPCProfileJob:
    """Select one deterministic, metadata-only CT026 PPC profiling job.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated analysis configuration. Its configured condition and site
        order breaks membership ties, and its ``whole`` analysis-window bounds
        are event-relative seconds.
    prepared_phase : PreparedPhaseRun-like object
        Must expose ``trial_indices`` as one-dimensional stable integer trial
        identities, ``prepared_trials.condition_names`` and trial-by-condition
        Boolean membership, ``site_valid`` as Boolean ``(site, trial)``, and
        ``phase_tensor`` with ``(site, frequency, trial, time)`` axes. Phase
        values are not transformed or copied; only the frequency-axis length is
        retained as provenance.
    prepared_spikes : PreparedSpikeRun-like object
        Must expose configured-order ``unit_ids``, one or more ``population_ids``,
        and one trial-local spike train per unit. Each train provides a float
        seconds array for each full trial axis. Only selected site-valid trials
        are counted in the half-open whole epoch.

    Returns
    -------
    RepresentativePPCProfileJob
        Frozen categorical identity, exactly 249 stable trial identities, and
        low/median/high eligible-unit raw counts and Hz rates. No executor,
        final artifact, phase transform, or data-array copy is invoked.

    Raises
    ------
    ValueError
        If prepared metadata axes/identities are inconsistent, if the largest
        selected condition is not exactly 249 trials, if no configured site has
        selected valid trials, or if fewer than three units satisfy the reliable
        count and contributing-trial gates.
    """
    if not isinstance(config, LFPSummaryConfig):
        raise ValueError("config must be an LFPSummaryConfig")

    trial_indices = _integer_trial_indices(prepared_phase)
    trial_count = trial_indices.size
    condition_names, condition_membership = _prepared_condition_metadata(
        prepared_phase,
        trial_count,
    )
    selected_positions, condition_name = _largest_condition_positions(
        prepared_phase,
        condition_names,
        condition_membership,
        trial_count,
    )
    if len(selected_positions) != 249:
        raise ValueError("CT026 PPC profiling requires exactly 249 selected trials")

    site_valid, site_id = _selected_profile_site(
        config,
        prepared_phase,
        selected_positions,
        trial_count,
    )
    site_valid_positions = tuple(
        int(position) for position in selected_positions if site_valid[position]
    )
    if not site_valid_positions:
        raise ValueError("selected CT026 trials contain no site-valid phase rows")

    epoch_bounds_s = _whole_epoch_bounds(config)
    eligible_units = _eligible_profile_units(
        config,
        prepared_spikes,
        trial_count,
        site_valid_positions,
        epoch_bounds_s,
    )
    if len(eligible_units) < 3:
        raise ValueError("CT026 PPC profiling requires at least three eligible units")

    ranked_units = sorted(eligible_units, key=lambda item: (item[1], item[0]))
    low_unit = _nearest_rank_unit(ranked_units, 0.10)
    median_unit = _nearest_rank_unit(ranked_units, 0.50)
    high_unit = _nearest_rank_unit(ranked_units, 0.90)
    epoch_duration_s = epoch_bounds_s[1] - epoch_bounds_s[0]
    population_id = _profile_population_id(prepared_spikes)
    frequency_count = _profile_frequency_count(prepared_phase, trial_count)
    stable_trial_ids = tuple(int(trial_indices[position]) for position in selected_positions)
    return RepresentativePPCProfileJob(
        condition_name=condition_name,
        site_id=site_id,
        epoch_name="whole",
        epoch_bounds_s=epoch_bounds_s,
        trial_indices=stable_trial_ids,
        trial_count=len(stable_trial_ids),
        site_valid_trial_count=len(site_valid_positions),
        unit_count=len(eligible_units),
        frequency_count=frequency_count,
        population_id=population_id,
        low_unit_id=low_unit[0],
        median_unit_id=median_unit[0],
        high_unit_id=high_unit[0],
        low_spike_count=low_unit[1],
        median_spike_count=median_unit[1],
        high_spike_count=high_unit[1],
        low_spike_rate_hz=low_unit[1] / (len(site_valid_positions) * epoch_duration_s),
        median_spike_rate_hz=median_unit[1] / (len(site_valid_positions) * epoch_duration_s),
        high_spike_rate_hz=high_unit[1] / (len(site_valid_positions) * epoch_duration_s),
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


def summarize_grouped_ppc_profile_metadata(
    *,
    component_plan: object,
    run_fingerprint: str,
    measured_memory: Mapping[str, object],
    projection_plans: tuple[tuple[int, object], ...],
) -> GroupedPPCProfileMetadata:
    """Return pure scalar cost metadata for one grouped component plan.

    Parameters
    ----------
    component_plan : PPCComponentPlan
        Immutable same-workload base plan. Its categorical jobs, stable trial
        rows, site-qualified union, allocation estimates, and schedules are
        inspected only to derive scalar cost metadata.
    run_fingerprint : str
        Exact 64-character hexadecimal grouped execution identity.
    measured_memory : mapping
        Exact scalar process/aggregate RSS/PSS byte fields and a nonempty
        source string. Aggregate RSS/PSS may both be ``None`` when unavailable.
    projection_plans : tuple[(int, PPCComponentPlan), ...]
        Ordered 100- and 1,000-shuffle plans for the exact same workload.
        No schedule rows are copied into the result.

    Returns
    -------
    GroupedPPCProfileMetadata
        Scalar edge, allocation, measured-memory, and 100/1,000 projection
        accounting. The function is pure and performs no executor, file, or
        scientific-array work.
    """
    _validate_grouped_fingerprint(run_fingerprint)
    memory = _validated_grouped_memory(measured_memory)
    base = _validated_grouped_plan(component_plan, "component_plan")
    projections = _validated_projection_plans(base, projection_plans)
    plan_100 = projections[100]
    plan_1000 = projections[1000]
    allocation = base.allocation_estimate
    return GroupedPPCProfileMetadata(
        schema_version="grouped_ppc_profile_metadata.v1",
        profile_kind="grouped_serial_ppc",
        run_fingerprint=run_fingerprint,
        scheduled_edge_count=int(base.scheduled_edge_count),
        independent_edge_count=int(base.independent_edge_count),
        unique_site_qualified_union_edge_count=int(base.union_edge_count),
        edge_union_saturation=float(base.edge_union_saturation),
        edge_reuse_ratio=float(base.edge_reuse_ratio),
        planned_parent_private_bytes=int(allocation.planned_parent_private_bytes),
        planned_worker_private_bytes=int(allocation.planned_worker_private_bytes),
        shared_phase_mmap_bytes=int(allocation.shared_phase_mmap_bytes),
        planned_aggregate_array_bytes=int(allocation.planned_aggregate_array_bytes),
        measured_peak_process_rss_bytes=memory["peak_process_rss_bytes"],
        measured_peak_aggregate_rss_bytes=memory["peak_aggregate_rss_bytes"],
        measured_peak_aggregate_pss_bytes=memory["peak_aggregate_pss_bytes"],
        measured_memory_source=memory["memory_source"],
        **_projection_metric_fields(100, plan_100),
        **_projection_metric_fields(1000, plan_1000),
    )


def profile_grouped_ppc_component(
    *,
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    work_root: Path,
    clock: Callable[[], float],
    memory_sampler: Callable[[], Mapping[str, object]],
    projection_plans: tuple[tuple[int, object], ...],
    component_plan: object | None = None,
) -> GroupedPPCProfileResult:
    """Profile one production grouped component without retaining scientific data.

    ``prepared_phase`` and ``prepared_spikes`` retain the production axes and
    seconds units accepted by :func:`execute_grouped_ppc_component`. The grouped
    executor is invoked exactly once. Seven narrow runtime seams are restored
    after success or failure; inclusive stage timings intentionally overlap.
    ``projection_plans`` are pure same-workload planner products used only for
    scalar 100/1,000 shuffle estimates.
    """
    _validate_serial_execution(execution)
    if not callable(clock) or not callable(memory_sampler):
        raise ValueError("clock and memory_sampler must be callable")

    # Read and validate the start time before replacing any production seam.
    # A bad profiling clock must not leave the runtime partially instrumented.
    start = _read_clock(clock, "start")

    durations = {
        "geometry": 0.0,
        "observed": 0.0,
        "union": 0.0,
        "shuffle": 0.0,
        "null": 0.0,
        "histogram": 0.0,
        "checkpoint": 0.0,
    }

    def measure(name: str, function: Callable[..., object], *args: object, **kwargs: object) -> object:
        """Time one direct seam inclusively using the caller's monotonic clock."""
        before = _read_clock(clock, name)
        try:
            return function(*args, **kwargs)
        finally:
            after = _read_clock(clock, name)
            if after < before:
                raise ValueError("PPC profile clock must be monotonic")
            durations[name] += after - before

    originals = {
        "geometry": lfp_summary_ppc_runtime.build_source_trial_spike_geometry,
        "observed": lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics,
        "union": lfp_summary_ppc_runtime.compute_segmented_edge_statistics,
        "shuffle": lfp_summary_ppc_runtime._execute_grouped_condition_batch,
        "null": lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null,
        "histogram": lfp_summary_ppc_runtime._record_grouped_representative_histogram,
        "checkpoint": lfp_summary_ppc_runtime.write_ppc_checkpoint,
    }

    lfp_summary_ppc_runtime.build_source_trial_spike_geometry = (
        lambda *args, **kwargs: measure("geometry", originals["geometry"], *args, **kwargs)
    )
    lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics = (
        lambda *args, **kwargs: measure("observed", originals["observed"], *args, **kwargs)
    )
    lfp_summary_ppc_runtime.compute_segmented_edge_statistics = (
        lambda *args, **kwargs: measure("union", originals["union"], *args, **kwargs)
    )
    lfp_summary_ppc_runtime._execute_grouped_condition_batch = (
        lambda *args, **kwargs: measure("shuffle", originals["shuffle"], *args, **kwargs)
    )
    lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null = (
        lambda *args, **kwargs: measure("null", originals["null"], *args, **kwargs)
    )
    lfp_summary_ppc_runtime._record_grouped_representative_histogram = (
        lambda *args, **kwargs: measure("histogram", originals["histogram"], *args, **kwargs)
    )
    lfp_summary_ppc_runtime.write_ppc_checkpoint = (
        lambda *args, **kwargs: measure("checkpoint", originals["checkpoint"], *args, **kwargs)
    )

    try:
        execution_result = lfp_summary_ppc_runtime.execute_grouped_ppc_component(
            config=config,
            execution=execution,
            prepared_phase=prepared_phase,
            prepared_spikes=prepared_spikes,
            work_root=Path(work_root),
        )
    finally:
        lfp_summary_ppc_runtime.build_source_trial_spike_geometry = originals["geometry"]
        lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics = originals["observed"]
        lfp_summary_ppc_runtime.compute_segmented_edge_statistics = originals["union"]
        lfp_summary_ppc_runtime._execute_grouped_condition_batch = originals["shuffle"]
        lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null = originals["null"]
        lfp_summary_ppc_runtime._record_grouped_representative_histogram = originals["histogram"]
        lfp_summary_ppc_runtime.write_ppc_checkpoint = originals["checkpoint"]
    end = _read_clock(clock, "end")
    if end < start:
        raise ValueError("PPC profile clock must be monotonic")
    total = end - start
    if total <= 0.0:
        raise ValueError("PPC profile total elapsed time must be positive")

    result_fingerprint = getattr(execution_result, "run_fingerprint", None)
    result_plan = getattr(execution_result, "component_plan", None)
    _validated_grouped_plan(result_plan, "execution_result.component_plan")
    if component_plan is not None and not _grouped_plans_are_equivalent(
        component_plan,
        result_plan,
    ):
        raise ValueError("component_plan does not match the executed grouped plan")
    metadata = summarize_grouped_ppc_profile_metadata(
        component_plan=result_plan,
        run_fingerprint=result_fingerprint,
        measured_memory=memory_sampler(),
        projection_plans=projection_plans,
    )
    return GroupedPPCProfileResult(
        schema_version="grouped_ppc_profile_result.v1",
        profile_kind="grouped_serial_ppc",
        run_fingerprint=metadata.run_fingerprint,
        geometry_build_seconds=durations["geometry"],
        observed_reduction_seconds=durations["observed"],
        union_edge_reduction_seconds=durations["union"],
        shuffle_aggregation_seconds=durations["shuffle"],
        null_summarization_seconds=durations["null"],
        representative_histogram_seconds=durations["histogram"],
        checkpoint_overhead_seconds=durations["checkpoint"],
        total_elapsed_seconds=total,
        throughput_scheduled_edge_per_second=(
            metadata.scheduled_edge_count / total
        ),
        **{
            name: value
            for name, value in vars(metadata).items()
            if name not in {"schema_version", "profile_kind", "run_fingerprint"}
        },
    )


def _validated_grouped_plan(value: object, name: str) -> object:
    """Return a runtime grouped plan after its public immutable type is checked."""
    if not isinstance(value, lfp_summary_ppc_runtime.PPCComponentPlan):
        raise ValueError(f"{name} must be a PPCComponentPlan")
    return value


def _validate_grouped_fingerprint(value: object) -> None:
    """Reject noncanonical grouped run identities before any scalar reporting."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("grouped profile fingerprint must be 64 lowercase hexadecimal characters")


def _validated_grouped_memory(value: Mapping[str, object]) -> dict[str, int | str | None]:
    """Validate measured process/aggregate RSS/PSS scalar memory boundaries."""
    required = {
        "peak_process_rss_bytes",
        "peak_aggregate_rss_bytes",
        "peak_aggregate_pss_bytes",
        "memory_source",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("grouped profile memory fields are incomplete")
    process = _nonnegative_memory_scalar(value["peak_process_rss_bytes"], "process")
    aggregate_rss = _optional_memory_scalar(value["peak_aggregate_rss_bytes"], "aggregate RSS")
    aggregate_pss = _optional_memory_scalar(value["peak_aggregate_pss_bytes"], "aggregate PSS")
    source = value["memory_source"]
    if not isinstance(source, str) or not source:
        raise ValueError("grouped profile memory source must be a nonempty string")
    if (aggregate_rss is None) != (aggregate_pss is None):
        raise ValueError("grouped profile memory aggregate RSS/PSS availability must agree")
    if aggregate_rss is not None and (process > aggregate_pss or aggregate_pss > aggregate_rss):
        raise ValueError("grouped profile memory must satisfy process RSS <= aggregate PSS <= aggregate RSS")
    return {
        "peak_process_rss_bytes": process,
        "peak_aggregate_rss_bytes": aggregate_rss,
        "peak_aggregate_pss_bytes": aggregate_pss,
        "memory_source": source,
    }


def _nonnegative_memory_scalar(value: object, name: str) -> int:
    """Return one nonnegative byte scalar and reject arrays or Boolean values."""
    if isinstance(value, np.ndarray):
        raise ValueError("grouped profile memory values must be scalar")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"grouped profile memory {name} must be a nonnegative integer")
    return value


def _optional_memory_scalar(value: object, name: str) -> int | None:
    """Return one optional nonnegative byte scalar without coercing floats."""
    if value is None:
        return None
    return _nonnegative_memory_scalar(value, name)


def _validated_projection_plans(
    base: object,
    value: tuple[tuple[int, object], ...],
) -> dict[int, object]:
    """Return ordered same-workload 100/1,000-shuffle plans keyed by label."""
    if not isinstance(value, tuple) or not value:
        raise ValueError("projection plans must be a nonempty ordered tuple")
    projections: dict[int, object] = {}
    previous = 0
    for item in value:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError("projection plans must contain (shuffle_count, plan) pairs")
        label, candidate = item
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise ValueError("projection shuffle labels must be positive integers")
        if label <= previous:
            raise ValueError("projection shuffle labels must be ordered and unique")
        previous = label
        plan = _validated_grouped_plan(candidate, "projection plan")
        _validate_projection_workload(base, plan, label)
        projections[label] = plan
    if tuple(projections) != (100, 1000):
        raise ValueError("projection plans must be ordered 100 and 1000 shuffle plans")
    return projections


def _validate_projection_workload(base: object, candidate: object, label: int) -> None:
    """Reject a projection whose categorical workload differs from the base plan."""
    positive_rows = {
        int(job.schedule_shape[0])
        for job in candidate.job_plans
        if int(job.schedule_shape[0]) > 0
    }
    if positive_rows and positive_rows != {label}:
        raise ValueError("projection shuffle label does not match its plan schedule")
    if _plan_workload_identity(base) != _plan_workload_identity(candidate):
        raise ValueError("projection plan must describe the same grouped workload")


def _plan_workload_identity(plan: object) -> tuple[object, ...]:
    """Return the shuffle-independent identity of a grouped workload.

    Realized derangements and their site-qualified union are intentionally not
    part of this identity: 100- and 1,000-shuffle plans can legitimately
    realize different reusable physical edges while still profiling the same
    selected trials, sites, conditions, epochs, units, frequencies, and mmap.
    """
    allocation = plan.allocation_estimate
    jobs = tuple(
        (
            job.condition_index,
            job.condition_name,
            job.site_index,
            job.site_id,
            job.epoch_index,
            job.epoch_name,
            job.segment_expression,
            tuple(int(row) for row in job.selected_trial_rows),
            job.base_ppc_seed,
            job.schedule_seed,
            job.condition_derivation_identity,
            job.site_derivation_identity,
            job.epoch_derivation_identity,
        )
        for job in plan.job_plans
    )
    allocation_identity = tuple(
        int(getattr(allocation, name))
        for name in (
            "observed_trial_statistics_bytes",
            "source_trial_spike_count_bytes",
            "worker_summary_bytes",
            "shared_phase_mmap_bytes",
        )
    )
    return (
        jobs,
        allocation_identity,
    )


def _grouped_plans_are_equivalent(left: object, right: object) -> bool:
    """Return whether two plans are exact products of the same grouped planner.

    This comparison is stricter than same-workload projection validation: it
    includes realized schedules, unions, batching, and every allocation field
    because the executor result is authoritative for measured profiling.
    """
    if not isinstance(left, lfp_summary_ppc_runtime.PPCComponentPlan):
        return False
    if not isinstance(right, lfp_summary_ppc_runtime.PPCComponentPlan):
        return False
    scalar_names = (
        "condition_batches",
        "scheduled_edge_count",
        "independent_edge_count",
        "union_edge_count",
        "edge_union_saturation",
        "edge_reuse_ratio",
    )
    if any(getattr(left, name) != getattr(right, name) for name in scalar_names):
        return False
    if vars(left.allocation_estimate) != vars(right.allocation_estimate):
        return False
    for name in (
        "edge_site_index",
        "stable_edge_source_trial_row",
        "stable_edge_target_trial_row",
    ):
        if not np.array_equal(getattr(left, name), getattr(right, name)):
            return False
    if len(left.job_plans) != len(right.job_plans):
        return False
    scalar_job_names = (
        "condition_index", "condition_name", "site_index", "site_id",
        "epoch_index", "epoch_name", "segment_expression", "base_ppc_seed",
        "schedule_seed", "condition_derivation_identity",
        "site_derivation_identity", "epoch_derivation_identity", "schedule_shape",
        "schedule_fingerprint",
    )
    array_job_names = (
        "selected_trial_rows", "schedule", "stable_edge_source_trial_row",
        "stable_edge_target_trial_row", "edge_union_position",
    )
    for left_job, right_job in zip(left.job_plans, right.job_plans, strict=True):
        if any(
            getattr(left_job, name) != getattr(right_job, name)
            for name in scalar_job_names
        ):
            return False
        if any(
            not np.array_equal(getattr(left_job, name), getattr(right_job, name))
            for name in array_job_names
        ):
            return False
    return True


def _projection_metric_fields(label: int, plan: object) -> dict[str, int | float]:
    """Return five flat scalar fields for one validated projection plan."""
    prefix = f"projection_{label}"
    return {
        f"{prefix}_scheduled_edge_count": int(plan.scheduled_edge_count),
        f"{prefix}_independent_edge_count": int(plan.independent_edge_count),
        f"{prefix}_unique_site_qualified_union_edge_count": int(plan.union_edge_count),
        f"{prefix}_edge_union_saturation": float(plan.edge_union_saturation),
        f"{prefix}_edge_reuse_ratio": float(plan.edge_reuse_ratio),
    }


def _integer_trial_indices(prepared_phase: object) -> np.ndarray:
    """Validate and return the caller-owned one-dimensional trial identity axis."""
    trial_indices = getattr(prepared_phase, "trial_indices", None)
    if (
        not isinstance(trial_indices, np.ndarray)
        or trial_indices.ndim != 1
        or not np.issubdtype(trial_indices.dtype, np.integer)
        or trial_indices.size == 0
        or len({int(trial_index) for trial_index in trial_indices}) != trial_indices.size
    ):
        raise ValueError("prepared phase trial_indices must be unique one-dimensional integers")
    return trial_indices


def _prepared_condition_metadata(
    prepared_phase: object,
    trial_count: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Validate and return configured condition identities and Boolean membership."""
    prepared_trials = getattr(prepared_phase, "prepared_trials", None)
    condition_names = getattr(prepared_trials, "condition_names", None)
    membership = getattr(prepared_trials, "condition_membership", None)
    if (
        not isinstance(condition_names, tuple)
        or not condition_names
        or len(set(condition_names)) != len(condition_names)
        or any(not isinstance(name, str) or not name for name in condition_names)
        or not isinstance(membership, np.ndarray)
        or membership.dtype != bool
        or membership.shape != (trial_count, len(condition_names))
    ):
        raise ValueError("prepared condition identities and membership have invalid axes")
    return condition_names, membership


def _largest_condition_positions(
    prepared_phase: object,
    condition_names: tuple[str, ...],
    membership: np.ndarray,
    trial_count: int,
) -> tuple[tuple[int, ...], str]:
    """Return stable positions for the largest fully analysis-eligible condition."""
    prepared_trials = prepared_phase.prepared_trials
    filter_membership = getattr(prepared_trials, "filter_membership", None)
    if (
        not isinstance(filter_membership, np.ndarray)
        or filter_membership.dtype != bool
        or filter_membership.shape != (trial_count,)
    ):
        raise ValueError("prepared filter_membership must be Boolean on the trial axis")
    objective_valid = _optional_trial_mask(
        prepared_trials,
        "objective_valid",
        trial_count,
    )
    user_excluded = _optional_trial_mask(
        prepared_trials,
        "user_excluded",
        trial_count,
    )

    best_positions: tuple[int, ...] = ()
    best_condition_index = 0
    for condition_index, _condition_name in enumerate(condition_names):
        positions = tuple(
            trial_index
            for trial_index in range(trial_count)
            if membership[trial_index, condition_index]
            and filter_membership[trial_index]
            and (objective_valid is None or objective_valid[trial_index])
            and (user_excluded is None or not user_excluded[trial_index])
        )
        if len(positions) > len(best_positions):
            best_positions = positions
            best_condition_index = condition_index
    return best_positions, condition_names[best_condition_index]


def _optional_trial_mask(
    prepared_trials: object,
    attribute: str,
    trial_count: int,
) -> np.ndarray | None:
    """Return one optional Boolean trial mask without creating a complement copy."""
    value = getattr(prepared_trials, attribute, None)
    if value is None:
        return None
    if not isinstance(value, np.ndarray) or value.dtype != bool or value.shape != (trial_count,):
        raise ValueError(f"prepared {attribute} must be Boolean on the trial axis")
    return value


def _selected_profile_site(
    config: LFPSummaryConfig,
    prepared_phase: object,
    selected_positions: tuple[int, ...],
    trial_count: int,
) -> tuple[np.ndarray, str]:
    """Return the configured-first site with the most selected valid phase rows."""
    site_valid = getattr(prepared_phase, "site_valid", None)
    phase_tensor = getattr(prepared_phase, "phase_tensor", None)
    if (
        not isinstance(site_valid, np.ndarray)
        or site_valid.dtype != bool
        or site_valid.shape != (len(config.sites), trial_count)
        or not isinstance(phase_tensor, np.ndarray)
        or phase_tensor.ndim != 4
        or phase_tensor.shape[0] != len(config.sites)
        or phase_tensor.shape[2] != trial_count
        or phase_tensor.shape[1] <= 0
    ):
        raise ValueError("prepared phase site-valid and phase-tensor axes are invalid")

    best_site_index = 0
    best_valid_count = -1
    for site_index, _site in enumerate(config.sites):
        valid_count = sum(bool(site_valid[site_index, position]) for position in selected_positions)
        if valid_count > best_valid_count:
            best_site_index = site_index
            best_valid_count = valid_count
    return site_valid[best_site_index], config.sites[best_site_index].stable_id


def _whole_epoch_bounds(config: LFPSummaryConfig) -> tuple[float, float]:
    """Return finite half-open whole-epoch event-relative seconds bounds."""
    start_s = config.analysis_windows.whole_start_s
    stop_s = config.analysis_windows.whole_stop_s
    if (
        isinstance(start_s, bool)
        or isinstance(stop_s, bool)
        or not isinstance(start_s, (int, float))
        or not isinstance(stop_s, (int, float))
        or not isfinite(float(start_s))
        or not isfinite(float(stop_s))
        or start_s >= stop_s
    ):
        raise ValueError("whole PPC epoch bounds must be finite ascending seconds")
    return float(start_s), float(stop_s)


def _eligible_profile_units(
    config: LFPSummaryConfig,
    prepared_spikes: object,
    trial_count: int,
    site_valid_positions: tuple[int, ...],
    epoch_bounds_s: tuple[float, float],
) -> list[tuple[str, int]]:
    """Return unit IDs and raw whole-epoch counts passing PPC reliability gates."""
    unit_ids = getattr(prepared_spikes, "unit_ids", None)
    trains = getattr(prepared_spikes, "trial_spike_trains", None)
    if (
        not isinstance(unit_ids, tuple)
        or len(set(unit_ids)) != len(unit_ids)
        or any(not isinstance(unit_id, str) or not unit_id for unit_id in unit_ids)
        or not isinstance(trains, tuple)
        or len(trains) != len(unit_ids)
    ):
        raise ValueError("prepared spike units and trial trains must agree")

    eligible_units: list[tuple[str, int]] = []
    start_s, stop_s = epoch_bounds_s
    for unit_id, train in zip(unit_ids, trains, strict=True):
        if getattr(train, "unit_id", None) != unit_id:
            raise ValueError("prepared spike train identity does not match its unit")
        trial_times = getattr(train, "relative_spike_times", None)
        if not isinstance(trial_times, tuple) or len(trial_times) != trial_count:
            raise ValueError("prepared spike trains must have one array per trial")
        count = 0
        contributing_trials = 0
        for trial_index in site_valid_positions:
            times = trial_times[trial_index]
            if not isinstance(times, np.ndarray) or times.ndim != 1:
                raise ValueError("trial-local spike times must be one-dimensional arrays")
            if not np.isfinite(times).all():
                raise ValueError("trial-local spike times must be finite seconds")
            in_epoch_count = int(np.count_nonzero((times >= start_s) & (times < stop_s)))
            count += in_epoch_count
            contributing_trials += int(in_epoch_count > 0)
        if (
            count >= config.ppc.minimum_reliable_spikes
            and contributing_trials >= config.ppc.minimum_computable_spikes
        ):
            eligible_units.append((unit_id, count))
    return eligible_units


def _nearest_rank_unit(
    ranked_units: list[tuple[str, int]],
    proportion: float,
) -> tuple[str, int]:
    """Return a one-indexed nearest-rank unit from count/ID-sorted candidates."""
    rank = max(1, ceil(proportion * len(ranked_units)))
    return ranked_units[rank - 1]


def _profile_population_id(prepared_spikes: object) -> str:
    """Return the sole representative population identity without copying it."""
    population_ids = getattr(prepared_spikes, "population_ids", None)
    if (
        not isinstance(population_ids, tuple)
        or len(population_ids) != 1
        or not isinstance(population_ids[0], str)
        or not population_ids[0]
    ):
        raise ValueError("prepared spikes must have exactly one population identity")
    return population_ids[0]


def _profile_frequency_count(prepared_phase: object, trial_count: int) -> int:
    """Return the validated positive phase-frequency provenance dimension."""
    phase_tensor = prepared_phase.phase_tensor
    if phase_tensor.shape[2] != trial_count:
        raise ValueError("prepared phase tensor trial axis disagrees with trial identities")
    return int(phase_tensor.shape[1])


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
