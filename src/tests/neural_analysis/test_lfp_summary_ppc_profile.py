"""Contracts for deterministic, serial-only PPC profiling.

The representative 249-trial inputs are descriptors, not an ordinary unit-test
benchmark.  Tests inject both execution and instrumentation so they remain
fast and do not make wall-clock or resident-memory assertions.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ppc_runtime
from src.neural_analysis.lfp_summary_models import (
    PPCExecutionConfig,
    default_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_ppc_profile import (
    PPCProfileWorkload,
    profile_production_ppc_job,
    profile_serial_ppc_workload,
    representative_serial_ppc_workloads,
    select_representative_ppc_profile_job,
)
from src.neural_analysis import lfp_summary_ppc_profile as ppc_profile


def _serial_execution(*, checkpoints: bool = True) -> PPCExecutionConfig:
    """Return valid serial execution settings for a profiling-only test job."""
    return PPCExecutionConfig(
        unit_block_size=2,
        shuffle_block_size=2,
        trial_edge_block_size=3,
        worker_count=1,
        checkpoint_enabled=checkpoints,
    )


def _production_inputs() -> tuple[object, object, np.ndarray]:
    """Return a finite one-unit, two-trial PPC job for production instrumentation.

    Returns complex64/Boolean phase and validity arrays on
    ``(site=1, frequency=50, trial=2, time=2)`` axes, float64 event-relative
    seconds, int64 trial identities, one unit's finite trial-local seconds, and
    an int64 ``(shuffle=3, trial=2)`` derangement schedule.  The arrays are
    intentionally tiny and never represent the 249-trial benchmark.
    """
    phase = np.ones((1, 50, 2, 2), dtype=np.complex64)
    prepared_phase = SimpleNamespace(
        phase_tensor=phase,
        phase_valid=np.ones(phase.shape, dtype=bool),
        relative_time_s=np.array([0.0, 1.0]),
        trial_indices=np.array([3, 4], dtype=np.int64),
    )
    prepared_spikes = SimpleNamespace(
        unit_ids=("PFC:1",),
        trial_spike_trains=(
            SimpleNamespace(
                relative_spike_times=(np.zeros(50), np.zeros(50)),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            ),
        ),
    )
    schedule = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int64)
    return prepared_phase, prepared_spikes, schedule


def _production_config() -> object:
    """Return a valid checkpointed one-worker configuration for the tiny job."""
    defaults = default_lfp_summary_config()
    return replace(
        defaults,
        ppc=replace(defaults.ppc, shuffle_count=3),
        ppc_execution=PPCExecutionConfig(
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
            worker_count=1,
            checkpoint_enabled=True,
        ),
    )


def _selection_inputs(
    *,
    largest_condition_trials: int = 249,
    eligible_unit_count: int = 5,
) -> tuple[object, object]:
    """Return metadata-only inputs for deterministic 249-trial job selection.

    The arrays preserve production axis meanings but contain no transformed
    phase result or PPC execution. Stable input unit ordering is deliberately
    non-sorted so percentile ties must use stable unit identifiers.
    """
    trial_count = 259
    condition_membership = np.zeros((trial_count, 2), dtype=bool)
    condition_membership[:largest_condition_trials, 1] = True
    condition_membership[largest_condition_trials:, 0] = True
    prepared_trials = SimpleNamespace(
        condition_names=("minor", "largest"),
        condition_membership=condition_membership,
        filter_membership=np.ones(trial_count, dtype=bool),
    )
    site_valid = np.zeros((3, trial_count), dtype=bool)
    site_valid[0, :largest_condition_trials] = True
    site_valid[1, : largest_condition_trials - 1] = True
    site_valid[2, :largest_condition_trials] = True
    prepared_phase = SimpleNamespace(
        prepared_trials=prepared_trials,
        trial_indices=np.arange(1000, 1000 + trial_count, dtype=np.int64),
        site_valid=site_valid,
        phase_tensor=np.ones((3, 50, trial_count, 2), dtype=np.complex64),
        phase_valid=np.ones((3, 50, trial_count, 2), dtype=bool),
        relative_time_s=np.array([-2.0, 0.0], dtype=float),
    )
    counts_by_unit = (60, 80, 100, 100, 140)[:eligible_unit_count]
    unit_ids = ("PFC:5", "PFC:4", "PFC:3", "PFC:2", "PFC:1")[:eligible_unit_count]
    trains = []
    for count_value, unit_id in zip(counts_by_unit, unit_ids, strict=True):
        first_count = count_value // 2
        second_count = count_value - first_count
        trial_spikes = tuple(
            np.zeros(first_count, dtype=float)
            if trial_position == 0
            else np.ones(second_count, dtype=float)
            if trial_position == 1
            else np.empty(0, dtype=float)
            for trial_position in range(trial_count)
        )
        trains.append(
            SimpleNamespace(
                unit_id=unit_id,
                relative_spike_times=trial_spikes,
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            )
        )
    prepared_spikes = SimpleNamespace(
        unit_ids=unit_ids,
        population_ids=("ProbeB",),
        trial_spike_trains=tuple(trains),
    )
    return prepared_phase, prepared_spikes


def test_select_representative_ppc_profile_job_uses_exact_249_trials_and_stable_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selection is metadata-only and fixes the CT026 benchmark job identity.

    The largest condition has exactly 249 stable trial rows. PFC and HPC2 tie
    for most site-valid rows, so configured site order selects PFC. Candidate
    units have raw whole-epoch counts 60, 80, 100, 100, and 140; the nearest
    rank 10th/50th/90th selections are therefore PFC:5, PFC:2, and PFC:1.
    The two 100-count units verify stable unit-ID tie handling.
    """
    config = _production_config()
    phase, spikes = _selection_inputs()
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_ppc_blocks",
        lambda **_: (_ for _ in ()).throw(AssertionError("selection ran PPC")),
    )

    job = select_representative_ppc_profile_job(config, phase, spikes)

    assert job.condition_name == "largest"
    assert job.site_id == "PFC"
    assert job.epoch_name == "whole"
    assert job.epoch_bounds_s == (-2.0, 2.0)
    np.testing.assert_array_equal(job.trial_indices, np.arange(1000, 1249))
    assert job.trial_count == 249
    assert job.unit_count == 5
    assert job.frequency_count == 50
    assert job.low_unit_id == "PFC:5"
    assert job.median_unit_id == "PFC:2"
    assert job.high_unit_id == "PFC:1"
    assert (job.low_spike_count, job.median_spike_count, job.high_spike_count) == (
        60,
        100,
        140,
    )
    assert job.low_spike_rate_hz == pytest.approx(60.0 / (249 * 4.0))
    assert job.median_spike_rate_hz == pytest.approx(100.0 / (249 * 4.0))
    assert job.high_spike_rate_hz == pytest.approx(140.0 / (249 * 4.0))
    with pytest.raises(FrozenInstanceError):
        job.site_id = "HPC1"


def test_select_representative_ppc_profile_job_rejects_non_249_or_insufficient_units() -> None:
    """The benchmark requires exactly 249 trials and three eligible units."""
    config = _production_config()
    short_phase, short_spikes = _selection_inputs(largest_condition_trials=248)
    with pytest.raises(ValueError, match="249"):
        select_representative_ppc_profile_job(config, short_phase, short_spikes)

    phase, sparse_spikes = _selection_inputs(eligible_unit_count=2)
    with pytest.raises(ValueError, match="three"):
        select_representative_ppc_profile_job(config, phase, sparse_spikes)


def test_representative_serial_workloads_are_deterministic_249_trial_rate_levels() -> None:
    """Representative descriptors are fixed low/median/high 249-trial synthetic jobs.

    The returned tuple contains immutable workload specifications only: no phase
    transform, permutation sampling, cache write, or final component publication
    occurs while these descriptors are created.
    """
    first = representative_serial_ppc_workloads()
    second = representative_serial_ppc_workloads()

    assert first == second
    assert [workload.name for workload in first] == ["low", "median", "high"]
    assert {workload.trial_count for workload in first} == {249}
    assert all(workload.unit_count > 0 for workload in first)
    assert all(workload.frequency_count > 0 for workload in first)
    assert all(workload.shuffle_count > 0 for workload in first)
    assert [workload.spike_rate_hz for workload in first] == sorted(
        workload.spike_rate_hz for workload in first
    )
    assert len({workload.seed for workload in first}) == 3


def test_profile_serial_execution_records_injected_stage_time_memory_and_throughput(
    tmp_path: Path,
) -> None:
    """A serial profile records dimensioned stage metrics from injected instrumentation.

    ``workload_runner`` receives one validated workload, a stage-completion
    callback accepting the exact stage names ``phase_preparation``,
    ``edge_reduction``, ``shuffle_aggregation``, and optionally ``checkpoint``,
    plus a work-only directory.  It returns no scientific arrays.  ``clock``
    returns monotonically increasing seconds and ``memory_sampler`` returns
    non-negative resident bytes.  The result reports seconds, bytes, counts,
    and unit-trial-shuffle operations per second without writing a component or
    manifest.
    """
    workload = PPCProfileWorkload(
        name="unit-test",
        trial_count=3,
        unit_count=2,
        frequency_count=4,
        shuffle_count=4,
        spike_rate_hz=8.0,
        seed=17,
    )
    clock_values = iter((0.0, 1.0, 3.0, 6.0, 7.0, 10.0))
    memory_values = iter((100, 200, 500, 400, 450, 300))

    def workload_runner(
        received: PPCProfileWorkload,
        stage_complete: Callable[[str], None],
        work_directory: Path,
    ) -> None:
        """Report deterministic completed stages and use only the supplied work directory."""
        assert received == workload
        assert work_directory == tmp_path / "ppc_profile" / "unit-test"
        stage_complete("phase_preparation")
        stage_complete("edge_reduction")
        stage_complete("shuffle_aggregation")
        stage_complete("checkpoint")

    result = profile_serial_ppc_workload(
        workload=workload,
        execution=_serial_execution(),
        work_root=tmp_path,
        workload_runner=workload_runner,
        clock=lambda: next(clock_values),
        memory_sampler=lambda: next(memory_values),
    )

    assert result.workload == workload
    assert result.worker_count == 1
    assert result.unit_count == 2
    assert result.trial_count == 3
    assert result.shuffle_count == 4
    assert result.phase_preparation_seconds == pytest.approx(1.0)
    assert result.edge_reduction_seconds == pytest.approx(2.0)
    assert result.shuffle_aggregation_seconds == pytest.approx(3.0)
    assert result.checkpoint_overhead_seconds == pytest.approx(1.0)
    assert result.total_elapsed_seconds == pytest.approx(10.0)
    assert result.peak_memory_bytes == 500
    assert result.throughput_unit_trial_shuffle_per_second == pytest.approx(2.4)


def test_profile_serial_execution_validates_workload_and_rejects_workers(
    tmp_path: Path,
) -> None:
    """Profiling accepts positive finite workload axes/rates and exactly one worker."""
    with pytest.raises(ValueError, match="trial_count"):
        PPCProfileWorkload(
            name="bad", trial_count=0, unit_count=1, frequency_count=1,
            shuffle_count=1, spike_rate_hz=1.0, seed=1,
        )
    with pytest.raises(ValueError, match="spike_rate_hz"):
        PPCProfileWorkload(
            name="bad", trial_count=1, unit_count=1, frequency_count=1,
            shuffle_count=1, spike_rate_hz=float("nan"), seed=1,
        )

    workload = PPCProfileWorkload(
        name="small", trial_count=2, unit_count=1, frequency_count=1,
        shuffle_count=1, spike_rate_hz=1.0, seed=1,
    )
    parallel = replace(_serial_execution(), worker_count=2)
    with pytest.raises(ValueError, match="worker_count"):
        profile_serial_ppc_workload(
            workload=workload,
            execution=parallel,
            work_root=tmp_path,
            workload_runner=lambda *_: None,
            clock=lambda: 0.0,
            memory_sampler=lambda: 0,
        )


def test_profile_never_publishes_a_final_component_or_manifest(tmp_path: Path) -> None:
    """Profiling creates no final cache payload, component record, or manifest artifact."""
    workload = PPCProfileWorkload(
        name="no-publish", trial_count=2, unit_count=1, frequency_count=1,
        shuffle_count=1, spike_rate_hz=1.0, seed=2,
    )
    clock_values = iter((0.0, 1.0, 2.0))
    memory_values = iter((1, 2, 1))

    def runner(_: PPCProfileWorkload, stage_complete: Callable[[str], None], __: Path) -> None:
        """Complete one measured phase stage without producing analysis outputs."""
        stage_complete("phase_preparation")

    profile_serial_ppc_workload(
        workload=workload,
        execution=_serial_execution(checkpoints=False),
        work_root=tmp_path,
        workload_runner=runner,
        clock=lambda: next(clock_values),
        memory_sampler=lambda: next(memory_values),
    )

    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("component.json"))
    assert not list(tmp_path.rglob("*.npz"))


def test_production_ppc_job_profile_wraps_one_real_serial_execution_and_restores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production profiling measures one real executor call without publication.

    The profiler receives the validated small prepared objects and the caller's
    int64 derangement schedule unchanged, calls ``execute_ppc_blocks`` exactly
    once, and instruments only its temporary wrappers around production phase
    validation, observed reduction, edge sufficient statistics, shuffle
    aggregation, null summarization, and checkpoint writing. The result records
    seconds, peak sampled resident bytes, serial execution counts, unique
    scheduled directed edges, eligible `(unit, frequency)` entries, exact raw
    spike counts, phase-tensor value count, edge-valid phase sample count, and
    unique checkpoint block count. The edge-valid count sums
    ``valid_spike_count`` from every actual reducer call, including observed,
    eligibility re-evaluation, and null work. It must leave no final component
    or manifest artifact and restore every wrapped callable.
    """
    config = _production_config()
    phase, spikes, schedule = _production_inputs()
    original_execute = lfp_summary_ppc_runtime.execute_ppc_blocks
    execute_calls: list[dict[str, object]] = []

    def recording_execute(**kwargs: object):
        """Count the profile's one unmodified production executor invocation."""
        execute_calls.append(kwargs)
        return original_execute(**kwargs)

    originals = {
        "validate": lfp_summary_ppc_runtime._validated_phase_inputs,
        "observed": lfp_summary_ppc_runtime._compute_observed_block,
        "edge": lfp_summary_ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics,
        "shuffle": lfp_summary_ppc_runtime._stream_null_draws,
        "null": lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null,
        "checkpoint": lfp_summary_ppc_runtime.write_ppc_checkpoint,
    }
    monotonic_seconds = count(start=0)
    resident_bytes = count(start=100, step=100)
    monkeypatch.setattr(lfp_summary_ppc_runtime, "execute_ppc_blocks", recording_execute)

    result = profile_production_ppc_job(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
        clock=lambda: float(next(monotonic_seconds)),
        rss_sampler=lambda: next(resident_bytes),
    )

    assert len(execute_calls) == 1
    assert execute_calls[0]["schedule"] is schedule
    assert result.execution_result.summary_arrays["ppc"].shape == (1, 50)
    assert result.worker_count == 1
    assert result.edge_call_count > 0
    assert result.unique_scheduled_edge_count == 2
    assert result.eligible_unit_frequency_count == 50
    assert result.spike_sample_count == 100
    assert result.phase_tensor_value_count == 200
    assert result.edge_valid_phase_sample_count == 15000
    assert result.checkpoint_block_count == 1
    assert result.checkpoint_overhead_seconds >= 0.0
    assert result.phase_validation_seconds >= 0.0
    assert result.observed_reduction_seconds >= 0.0
    assert result.edge_reduction_seconds >= 0.0
    assert result.shuffle_aggregation_seconds >= 0.0
    assert result.total_elapsed_seconds > 0.0
    assert result.peak_memory_bytes > 0
    assert result.peak_memory_source == "injected_rss_sampler"
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("component.json"))
    assert lfp_summary_ppc_runtime._validated_phase_inputs is originals["validate"]
    assert lfp_summary_ppc_runtime._compute_observed_block is originals["observed"]
    assert lfp_summary_ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics is originals["edge"]
    assert lfp_summary_ppc_runtime._stream_null_draws is originals["shuffle"]
    assert lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null is originals["null"]
    assert lfp_summary_ppc_runtime.write_ppc_checkpoint is originals["checkpoint"]


def test_production_ppc_job_profile_restores_wrappers_after_executor_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrumentation restores the prior callable even when real execution raises.

    The injected failure replaces only observed block reduction; the profiler's
    temporary wrapper must restore that exact failing callable in ``finally``.
    The exception propagates, and no final component or manifest is published.
    """
    config = _production_config()
    phase, spikes, schedule = _production_inputs()

    def fail_observed(*_: object, **__: object) -> None:
        """Abort the real executor after its profile wrappers have been installed."""
        raise RuntimeError("injected observed failure")

    monkeypatch.setattr(lfp_summary_ppc_runtime, "_compute_observed_block", fail_observed)
    with pytest.raises(RuntimeError, match="injected observed failure"):
        profile_production_ppc_job(
            config=config,
            execution=config.ppc_execution,
            prepared_phase=phase,
            prepared_spikes=spikes,
            schedule=schedule,
            work_root=tmp_path,
            clock=lambda: 1.0,
            rss_sampler=lambda: 0,
        )

    assert lfp_summary_ppc_runtime._compute_observed_block is fail_observed
    assert not list(tmp_path.rglob("manifest.json"))
    assert not list(tmp_path.rglob("component.json"))


def _grouped_plan_for_profile_metadata(
    *,
    selected_trial_count: int = 3,
    shuffle_count: int = 3,
    seed: int = 19,
    site_ids: tuple[str, ...] = ("PFC", "HPC"),
    stable_row_start: int = 101,
    unit_count: int = 1,
    frequency_count: int = 2,
    shared_phase_mmap_bytes: int = 4096,
) -> object:
    """Return a pure two-site grouped plan with no phase or spike payloads.

    The returned plan has categorical trial/site identities, a Boolean
    ``(trial, condition)`` membership matrix, and int64
    ``(trial, unit, before_after)`` source counts only.  It is intentionally
    sufficient for schedule-union accounting while containing no CT026 or
    scientific input arrays.
    """
    defaults = default_lfp_summary_config()
    config = replace(
        defaults,
        ppc=replace(defaults.ppc, shuffle_count=shuffle_count, seed=seed),
        ppc_execution=replace(
            defaults.ppc_execution,
            worker_count=1,
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
        ),
    )
    membership = np.zeros((selected_trial_count, 2), dtype=bool)
    if selected_trial_count >= 1:
        membership[0, 0] = True
    if selected_trial_count >= 2:
        membership[1, :] = True
    if selected_trial_count >= 3:
        membership[2, 1] = True
    return lfp_summary_ppc_runtime.plan_grouped_ppc_component(
        config=config,
        condition_names=("condition-a", "condition-b"),
        condition_membership=membership,
        site_ids=site_ids,
        site_trial_valid=np.ones((len(site_ids), selected_trial_count), dtype=bool),
        stable_trial_rows=np.arange(
            stable_row_start,
            stable_row_start + selected_trial_count,
            dtype=np.int64,
        ),
        source_trial_spike_count=np.zeros(
            (selected_trial_count, unit_count, 2), dtype=np.int64
        ),
        frequency_count=frequency_count,
        shared_phase_mmap_bytes=shared_phase_mmap_bytes,
    )


def test_grouped_profile_metadata_reports_scalar_union_cost_memory_and_projections() -> None:
    """Grouped profile metadata is scalar-only and derives plan cost exactly.

    The supplied pure plan has 12 result jobs (two sites, two conditions, and
    three epochs), three scheduled derangements per job, 24 independently
    requested physical edges, and eight unique site-qualified union edges.
    The 100/1000-shuffle projections use separately supplied real planner
    outputs rather than linearly scaling the three-shuffle union. Planned
    private bytes remain planner estimates, while measured RSS/PSS is a
    separate observed boundary with its source.
    """
    plan = _grouped_plan_for_profile_metadata()
    plan_100 = _grouped_plan_for_profile_metadata(shuffle_count=100)
    plan_1000 = _grouped_plan_for_profile_metadata(shuffle_count=1000)

    first = ppc_profile.summarize_grouped_ppc_profile_metadata(
        component_plan=plan,
        run_fingerprint="a" * 64,
        measured_memory={
            "peak_process_rss_bytes": 8192,
            "peak_aggregate_rss_bytes": 12288,
            "peak_aggregate_pss_bytes": 10240,
            "memory_source": "injected_rss_pss_sampler",
        },
        projection_plans=((100, plan_100), (1000, plan_1000)),
    )
    second = ppc_profile.summarize_grouped_ppc_profile_metadata(
        component_plan=plan,
        run_fingerprint="a" * 64,
        measured_memory={
            "peak_process_rss_bytes": 8192,
            "peak_aggregate_rss_bytes": 12288,
            "peak_aggregate_pss_bytes": 10240,
            "memory_source": "injected_rss_pss_sampler",
        },
        projection_plans=((100, plan_100), (1000, plan_1000)),
    )

    assert first == second
    assert first.schema_version == "grouped_ppc_profile_metadata.v1"
    assert first.profile_kind == "grouped_serial_ppc"
    assert first.run_fingerprint == "a" * 64
    assert first.scheduled_edge_count == 72
    assert first.independent_edge_count == 24
    assert first.unique_site_qualified_union_edge_count == 8
    assert first.edge_union_saturation == 1.0
    assert first.edge_reuse_ratio == 3.0
    assert first.planned_parent_private_bytes == plan.allocation_estimate.planned_parent_private_bytes
    assert first.planned_worker_private_bytes == plan.allocation_estimate.planned_worker_private_bytes
    assert first.shared_phase_mmap_bytes == 4096
    assert first.planned_aggregate_array_bytes == plan.allocation_estimate.planned_aggregate_array_bytes
    assert first.measured_peak_process_rss_bytes == 8192
    assert first.measured_peak_aggregate_rss_bytes == 12288
    assert first.measured_peak_aggregate_pss_bytes == 10240
    assert first.measured_memory_source == "injected_rss_pss_sampler"
    assert first.projection_100_scheduled_edge_count == 2400
    assert first.projection_100_independent_edge_count == 24
    assert first.projection_100_unique_site_qualified_union_edge_count == 8
    assert first.projection_100_edge_union_saturation == 1.0
    assert first.projection_100_edge_reuse_ratio == 3.0
    assert first.projection_1000_scheduled_edge_count == 24000
    assert first.projection_1000_independent_edge_count == 24
    assert first.projection_1000_unique_site_qualified_union_edge_count == 8
    assert first.projection_1000_edge_union_saturation == 1.0
    assert first.projection_1000_edge_reuse_ratio == 3.0
    assert all(
        not isinstance(value, (np.ndarray, dict, list, tuple))
        for value in vars(first).values()
    )


def test_grouped_profile_metadata_accepts_real_249_trial_projection_union_growth() -> None:
    """Same-workload validation ignores realized shuffle rows and edge unions.

    A 249-trial component has the same site/condition/epoch identities,
    selected stable rows, unit/frequency dimensions, and phase-mmap accounting
    at 100 and 1,000 shuffles. Its independently seeded schedules legitimately
    realize different site-qualified physical unions, so requiring those arrays
    to match would reject the very projections the profiler is meant to report.
    """
    defaults = default_lfp_summary_config()
    config = replace(
        defaults,
        ppc=replace(defaults.ppc, shuffle_count=100, seed=313),
        ppc_execution=replace(
            defaults.ppc_execution,
            worker_count=1,
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
        ),
    )

    def plan(shuffle_count: int) -> object:
        """Build one deterministic nondegenerate plan without sampled inputs."""
        return lfp_summary_ppc_runtime.plan_grouped_ppc_component(
            config=replace(config, ppc=replace(config.ppc, shuffle_count=shuffle_count)),
            condition_names=("condition-a",),
            condition_membership=np.ones((249, 1), dtype=bool),
            site_ids=("PFC",),
            site_trial_valid=np.ones((1, 249), dtype=bool),
            stable_trial_rows=np.arange(1000, 1249, dtype=np.int64),
            source_trial_spike_count=np.zeros((249, 1, 2), dtype=np.int64),
            frequency_count=2,
            shared_phase_mmap_bytes=4096,
        )

    plan_100 = plan(100)
    plan_1000 = plan(1000)
    assert not np.array_equal(
        plan_100.stable_edge_target_trial_row,
        plan_1000.stable_edge_target_trial_row,
    )
    assert plan_100.union_edge_count != plan_1000.union_edge_count

    metadata = ppc_profile.summarize_grouped_ppc_profile_metadata(
        component_plan=plan_100,
        run_fingerprint="1" * 64,
        measured_memory={
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": None,
            "peak_aggregate_pss_bytes": None,
            "memory_source": "aggregate_rss_pss_unavailable",
        },
        projection_plans=((100, plan_100), (1000, plan_1000)),
    )

    assert metadata.scheduled_edge_count == plan_100.scheduled_edge_count
    assert metadata.projection_1000_scheduled_edge_count == plan_1000.scheduled_edge_count
    assert (
        metadata.projection_1000_unique_site_qualified_union_edge_count
        == plan_1000.union_edge_count
    )
    assert (
        metadata.projection_1000_unique_site_qualified_union_edge_count
        != metadata.projection_100_unique_site_qualified_union_edge_count
    )


def test_grouped_profile_metadata_rejects_empty_singleton_malformed_and_overflow_cases() -> None:
    """Metadata-only accounting rejects ambiguous projections before any execution.

    Empty/singleton plans report zero demand and reuse. Malformed fingerprints,
    unordered projection requests, raw arrays in measured-memory metadata, and
    mismatched projection labels fail before any grouped executor or artifact
    operation is possible.
    """
    singleton_plan = _grouped_plan_for_profile_metadata(selected_trial_count=1)
    empty = ppc_profile.summarize_grouped_ppc_profile_metadata(
        component_plan=singleton_plan,
        run_fingerprint="b" * 64,
        measured_memory={
            "peak_process_rss_bytes": 0,
            "peak_aggregate_rss_bytes": 0,
            "peak_aggregate_pss_bytes": 0,
            "memory_source": "not_sampled",
        },
        projection_plans=(
            (100, _grouped_plan_for_profile_metadata(selected_trial_count=1, shuffle_count=100)),
            (1000, _grouped_plan_for_profile_metadata(selected_trial_count=1, shuffle_count=1000)),
        ),
    )
    assert empty.scheduled_edge_count == 0
    assert empty.independent_edge_count == 0
    assert empty.unique_site_qualified_union_edge_count == 0
    assert empty.edge_union_saturation == 0.0
    assert empty.edge_reuse_ratio == 0.0
    for prefix in ("projection_100", "projection_1000"):
        assert getattr(empty, f"{prefix}_scheduled_edge_count") == 0
        assert getattr(empty, f"{prefix}_independent_edge_count") == 0
        assert getattr(empty, f"{prefix}_unique_site_qualified_union_edge_count") == 0
        assert getattr(empty, f"{prefix}_edge_union_saturation") == 0.0
        assert getattr(empty, f"{prefix}_edge_reuse_ratio") == 0.0
    unavailable = ppc_profile.summarize_grouped_ppc_profile_metadata(
        component_plan=_grouped_plan_for_profile_metadata(),
        run_fingerprint="f" * 64,
        measured_memory={
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": None,
            "peak_aggregate_pss_bytes": None,
            "memory_source": "aggregate_rss_pss_unavailable",
        },
        projection_plans=(
            (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
            (1000, _grouped_plan_for_profile_metadata(shuffle_count=1000)),
        ),
    )
    assert unavailable.measured_peak_aggregate_rss_bytes is None
    assert unavailable.measured_peak_aggregate_pss_bytes is None

    common = {
        "component_plan": _grouped_plan_for_profile_metadata(),
        "run_fingerprint": "c" * 64,
        "measured_memory": {
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": 1,
            "peak_aggregate_pss_bytes": 1,
            "memory_source": "sampler",
        },
    }
    with pytest.raises(ValueError, match="fingerprint"):
        ppc_profile.summarize_grouped_ppc_profile_metadata(
            **{**common, "run_fingerprint": "not-a-fingerprint"},
            projection_plans=((100, _grouped_plan_for_profile_metadata(shuffle_count=100)),),
        )
    with pytest.raises(ValueError, match="ordered"):
        ppc_profile.summarize_grouped_ppc_profile_metadata(
            **common,
            projection_plans=(
                (1000, _grouped_plan_for_profile_metadata(shuffle_count=1000)),
                (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
            ),
        )
    with pytest.raises(ValueError, match="scalar"):
        ppc_profile.summarize_grouped_ppc_profile_metadata(
            **{
                **common,
                "measured_memory": {
                    "peak_process_rss_bytes": np.array([1], dtype=np.int64),
                    "peak_aggregate_rss_bytes": 1,
                    "peak_aggregate_pss_bytes": 1,
                    "memory_source": "sampler",
                },
            },
            projection_plans=((100, _grouped_plan_for_profile_metadata(shuffle_count=100)),),
        )
    for invalid_memory in (
        {
            "peak_process_rss_bytes": True,
            "peak_aggregate_rss_bytes": 1,
            "peak_aggregate_pss_bytes": 1,
            "memory_source": "sampler",
        },
        {
            "peak_process_rss_bytes": -1,
            "peak_aggregate_rss_bytes": 1,
            "peak_aggregate_pss_bytes": 1,
            "memory_source": "sampler",
        },
        {
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": float("inf"),
            "peak_aggregate_pss_bytes": 1,
            "memory_source": "sampler",
        },
        {
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": 1,
            "memory_source": "sampler",
        },
        {
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": 1,
            "peak_aggregate_pss_bytes": 1,
            "memory_source": "sampler",
            "unexpected": 1,
        },
        {
            "peak_process_rss_bytes": 1,
            "peak_aggregate_rss_bytes": 4,
            "peak_aggregate_pss_bytes": 5,
            "memory_source": "sampler",
        },
    ):
        with pytest.raises(ValueError, match="memory"):
            ppc_profile.summarize_grouped_ppc_profile_metadata(
                **{**common, "measured_memory": invalid_memory},
                projection_plans=(
                    (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
                ),
            )
    for invalid_projections in (
        ((True, _grouped_plan_for_profile_metadata(shuffle_count=100)),),
        ((0, _grouped_plan_for_profile_metadata(shuffle_count=100)),),
        (
            (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
            (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
        ),
    ):
        with pytest.raises(ValueError, match="projection"):
            ppc_profile.summarize_grouped_ppc_profile_metadata(
                **common, projection_plans=invalid_projections
            )
    for mismatched_plan in (
        _grouped_plan_for_profile_metadata(shuffle_count=100, selected_trial_count=1),
        _grouped_plan_for_profile_metadata(shuffle_count=100, site_ids=("PFC",)),
        _grouped_plan_for_profile_metadata(shuffle_count=100, stable_row_start=901),
        _grouped_plan_for_profile_metadata(shuffle_count=100, seed=99),
        _grouped_plan_for_profile_metadata(shuffle_count=100, unit_count=2),
        _grouped_plan_for_profile_metadata(shuffle_count=100, frequency_count=3),
        _grouped_plan_for_profile_metadata(shuffle_count=100, shared_phase_mmap_bytes=8192),
    ):
        with pytest.raises(ValueError, match="projection"):
            ppc_profile.summarize_grouped_ppc_profile_metadata(
                **common, projection_plans=((100, mismatched_plan),)
            )
    with pytest.raises(ValueError, match="shuffle"):
        ppc_profile.summarize_grouped_ppc_profile_metadata(
            **common,
            projection_plans=((1000, _grouped_plan_for_profile_metadata(shuffle_count=100)),),
        )


def test_grouped_profile_binds_one_grouped_executor_and_returns_scalar_product(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grouped serial profiler invokes only the grouped production seam.

    The injected grouped result deliberately contains a scientific summary
    array. The profiler may inspect its immutable plan and run fingerprint but
    must return scalar stage, union, planned-memory, and measured-memory
    fields only; it cannot leak phase, spike, schedule, null, histogram, or
    final-artifact tensors into the profile product. The legacy single-job
    executor is forbidden on this grouped path.

    S6 source scope is deliberately limited to extracting the narrow private
    ``_record_grouped_representative_histogram`` seam in
    ``lfp_summary_ppc_runtime.py``. It exists only to time histogram counting;
    no broader grouped-runtime behavior or science is changed.
    """
    config = _production_config()
    plan = _grouped_plan_for_profile_metadata()
    grouped_calls: list[dict[str, object]] = []

    def geometry(*_: object, **__: object) -> None:
        """Act as the production geometry seam without creating sampled arrays."""

    def histogram(*_: object, **__: object) -> None:
        """Act as the histogram-only count seam nested in observation."""

    def observed(*_: object, **__: object) -> None:
        """Invoke the nested histogram seam to prove inclusive stage timing."""
        lfp_summary_ppc_runtime._record_grouped_representative_histogram()

    def union_edge(*_: object, **__: object) -> None:
        """Act as one grouped site-qualified edge-union reducer call."""

    def null_summary(*_: object, **__: object) -> None:
        """Act as a null-summary reduction nested in shuffle aggregation."""

    def shuffle(*_: object, **__: object) -> None:
        """Invoke the nested null-summary seam for inclusive timing semantics."""
        lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null()

    def checkpoint(*_: object, **__: object) -> None:
        """Act as one grouped checkpoint-publication seam."""

    originals = {
        "geometry": geometry,
        "observed": observed,
        "union_edge": union_edge,
        "shuffle": shuffle,
        "null": null_summary,
        "histogram": histogram,
        "checkpoint": checkpoint,
    }
    monkeypatch.setattr(lfp_summary_ppc_runtime, "build_source_trial_spike_geometry", geometry)
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        observed,
    )
    monkeypatch.setattr(lfp_summary_ppc_runtime, "compute_segmented_edge_statistics", union_edge)
    monkeypatch.setattr(lfp_summary_ppc_runtime, "_execute_grouped_condition_batch", shuffle)
    monkeypatch.setattr(
        lfp_summary_ppc_runtime.spike_lfp_summary,
        "summarize_permutation_null",
        null_summary,
    )
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "_record_grouped_representative_histogram",
        histogram,
        raising=False,
    )
    monkeypatch.setattr(lfp_summary_ppc_runtime, "write_ppc_checkpoint", checkpoint)

    def grouped_executor(**kwargs: object) -> object:
        """Record the one production grouped invocation and return test metadata."""
        grouped_calls.append(kwargs)
        lfp_summary_ppc_runtime.build_source_trial_spike_geometry()
        lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics()
        lfp_summary_ppc_runtime.compute_segmented_edge_statistics()
        lfp_summary_ppc_runtime._execute_grouped_condition_batch()
        lfp_summary_ppc_runtime.write_ppc_checkpoint()
        return SimpleNamespace(
            run_fingerprint="d" * 64,
            component_plan=plan,
            summary_arrays={"ppc": np.ones((1, 1, 1, 1, 1), dtype=float)},
        )

    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_grouped_ppc_component",
        grouped_executor,
    )
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_ppc_blocks",
        lambda **_: (_ for _ in ()).throw(AssertionError("legacy executor called")),
    )
    clock_values = count(start=0)

    result = ppc_profile.profile_grouped_ppc_component(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=SimpleNamespace(phase_tensor=np.ones((1,), dtype=np.complex64)),
        prepared_spikes=SimpleNamespace(trial_spike_trains=()),
        work_root=tmp_path,
        clock=lambda: float(next(clock_values)),
        memory_sampler=lambda: {
            "peak_process_rss_bytes": 4096,
            "peak_aggregate_rss_bytes": 6144,
            "peak_aggregate_pss_bytes": 5120,
            "memory_source": "injected_rss_pss_sampler",
        },
        projection_plans=(
            (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
            (1000, _grouped_plan_for_profile_metadata(shuffle_count=1000)),
        ),
    )

    assert len(grouped_calls) == 1
    assert grouped_calls[0]["config"] is config
    assert grouped_calls[0]["execution"] is config.ppc_execution
    assert grouped_calls[0]["work_root"] == tmp_path
    assert result.schema_version == "grouped_ppc_profile_result.v1"
    assert result.run_fingerprint == "d" * 64
    assert result.total_elapsed_seconds == pytest.approx(15.0)
    assert result.measured_peak_process_rss_bytes == 4096
    assert result.measured_peak_aggregate_rss_bytes == 6144
    assert result.measured_peak_aggregate_pss_bytes == 5120
    assert result.geometry_build_seconds == pytest.approx(1.0)
    assert result.observed_reduction_seconds == pytest.approx(3.0)
    assert result.union_edge_reduction_seconds == pytest.approx(1.0)
    assert result.shuffle_aggregation_seconds == pytest.approx(3.0)
    assert result.null_summarization_seconds == pytest.approx(1.0)
    assert result.representative_histogram_seconds == pytest.approx(1.0)
    assert result.checkpoint_overhead_seconds == pytest.approx(1.0)
    assert (
        result.observed_reduction_seconds
        + result.representative_histogram_seconds
        + result.shuffle_aggregation_seconds
        + result.null_summarization_seconds
        > 2 * result.total_elapsed_seconds / 5
    )
    assert result.throughput_scheduled_edge_per_second == pytest.approx(
        result.scheduled_edge_count / result.total_elapsed_seconds
    )
    for name, value in vars(result).items():
        assert not isinstance(value, np.ndarray), name
        assert name not in {"execution_result", "summary_arrays", "component_plan"}
    assert lfp_summary_ppc_runtime.build_source_trial_spike_geometry is originals["geometry"]
    assert lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics is originals["observed"]
    assert lfp_summary_ppc_runtime.compute_segmented_edge_statistics is originals["union_edge"]
    assert lfp_summary_ppc_runtime._execute_grouped_condition_batch is originals["shuffle"]
    assert lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null is originals["null"]
    assert lfp_summary_ppc_runtime._record_grouped_representative_histogram is originals["histogram"]
    assert lfp_summary_ppc_runtime.write_ppc_checkpoint is originals["checkpoint"]

    def failing_null_summary(*_: object, **__: object) -> None:
        """Raise from the nested wrapped null-summary seam, not the executor shell."""
        raise RuntimeError("injected grouped failure")

    def failing_grouped_executor(**_: object) -> object:
        """Reach the wrapped shuffle/null nesting before propagating its failure."""
        lfp_summary_ppc_runtime._execute_grouped_condition_batch()
        raise AssertionError("nested null failure did not propagate")

    monkeypatch.setattr(
        lfp_summary_ppc_runtime.spike_lfp_summary,
        "summarize_permutation_null",
        failing_null_summary,
    )
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_grouped_ppc_component",
        failing_grouped_executor,
    )
    with pytest.raises(RuntimeError, match="injected grouped failure"):
        ppc_profile.profile_grouped_ppc_component(
            config=config,
            execution=config.ppc_execution,
            prepared_phase=SimpleNamespace(phase_tensor=np.ones((1,), dtype=np.complex64)),
            prepared_spikes=SimpleNamespace(trial_spike_trains=()),
            work_root=tmp_path,
            clock=lambda: float(next(count(start=0))),
            memory_sampler=lambda: {
                "peak_process_rss_bytes": 1,
                "peak_aggregate_rss_bytes": 1,
                "peak_aggregate_pss_bytes": 1,
                "memory_source": "sampler",
            },
            projection_plans=((100, _grouped_plan_for_profile_metadata(shuffle_count=100)),),
        )
    assert lfp_summary_ppc_runtime.build_source_trial_spike_geometry is originals["geometry"]
    assert lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics is originals["observed"]
    assert lfp_summary_ppc_runtime.compute_segmented_edge_statistics is originals["union_edge"]
    assert lfp_summary_ppc_runtime._execute_grouped_condition_batch is originals["shuffle"]
    assert lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null is failing_null_summary
    assert lfp_summary_ppc_runtime._record_grouped_representative_histogram is originals["histogram"]
    assert lfp_summary_ppc_runtime.write_ppc_checkpoint is originals["checkpoint"]

    caller_plan = _grouped_plan_for_profile_metadata(seed=91)
    authoritative_plan = _grouped_plan_for_profile_metadata(seed=92)
    monkeypatch.setattr(
        lfp_summary_ppc_runtime,
        "execute_grouped_ppc_component",
        lambda **_: SimpleNamespace(
            run_fingerprint="e" * 64,
            component_plan=authoritative_plan,
            summary_arrays={},
        ),
    )
    mismatch_clock = iter((0.0, 1.0))
    with pytest.raises(ValueError, match="component_plan"):
        ppc_profile.profile_grouped_ppc_component(
            config=config,
            execution=config.ppc_execution,
            prepared_phase=SimpleNamespace(),
            prepared_spikes=SimpleNamespace(),
            work_root=tmp_path,
            clock=lambda: next(mismatch_clock),
            memory_sampler=lambda: {
                "peak_process_rss_bytes": 1,
                "peak_aggregate_rss_bytes": None,
                "peak_aggregate_pss_bytes": None,
                "memory_source": "aggregate_rss_pss_unavailable",
            },
            component_plan=caller_plan,
            projection_plans=(
                (100, _grouped_plan_for_profile_metadata(seed=91, shuffle_count=100)),
                (1000, _grouped_plan_for_profile_metadata(seed=91, shuffle_count=1000)),
            ),
        )


def test_grouped_profile_restores_every_wrapper_when_start_clock_is_invalid(
    tmp_path: Path,
) -> None:
    """A nonfinite start sample cannot leave process-global runtime seams wrapped."""
    config = _production_config()
    before = {
        "geometry": lfp_summary_ppc_runtime.build_source_trial_spike_geometry,
        "observed": lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics,
        "union": lfp_summary_ppc_runtime.compute_segmented_edge_statistics,
        "shuffle": lfp_summary_ppc_runtime._execute_grouped_condition_batch,
        "null": lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null,
        "histogram": lfp_summary_ppc_runtime._record_grouped_representative_histogram,
        "checkpoint": lfp_summary_ppc_runtime.write_ppc_checkpoint,
    }

    try:
        with pytest.raises(ValueError, match="clock"):
            ppc_profile.profile_grouped_ppc_component(
                config=config,
                execution=config.ppc_execution,
                prepared_phase=SimpleNamespace(),
                prepared_spikes=SimpleNamespace(),
                work_root=tmp_path,
                clock=lambda: float("nan"),
                memory_sampler=lambda: {
                    "peak_process_rss_bytes": 1,
                    "peak_aggregate_rss_bytes": None,
                    "peak_aggregate_pss_bytes": None,
                    "memory_source": "aggregate_rss_pss_unavailable",
                },
                projection_plans=(
                    (100, _grouped_plan_for_profile_metadata(shuffle_count=100)),
                    (1000, _grouped_plan_for_profile_metadata(shuffle_count=1000)),
                ),
            )
        after = {
            "geometry": lfp_summary_ppc_runtime.build_source_trial_spike_geometry,
            "observed": lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics,
            "union": lfp_summary_ppc_runtime.compute_segmented_edge_statistics,
            "shuffle": lfp_summary_ppc_runtime._execute_grouped_condition_batch,
            "null": lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null,
            "histogram": lfp_summary_ppc_runtime._record_grouped_representative_histogram,
            "checkpoint": lfp_summary_ppc_runtime.write_ppc_checkpoint,
        }
    finally:
        lfp_summary_ppc_runtime.build_source_trial_spike_geometry = before["geometry"]
        lfp_summary_ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics = before["observed"]
        lfp_summary_ppc_runtime.compute_segmented_edge_statistics = before["union"]
        lfp_summary_ppc_runtime._execute_grouped_condition_batch = before["shuffle"]
        lfp_summary_ppc_runtime.spike_lfp_summary.summarize_permutation_null = before["null"]
        lfp_summary_ppc_runtime._record_grouped_representative_histogram = before["histogram"]
        lfp_summary_ppc_runtime.write_ppc_checkpoint = before["checkpoint"]

    assert after == before
