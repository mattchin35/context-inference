"""Contracts for deterministic, serial-only PPC profiling.

The representative 249-trial inputs are descriptors, not an ordinary unit-test
benchmark.  Tests inject both execution and instrumentation so they remain
fast and do not make wall-clock or resident-memory assertions.
"""

from __future__ import annotations

from dataclasses import replace
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
)


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
