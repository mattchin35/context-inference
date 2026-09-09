"""Contracts for deterministic, serial-only PPC profiling.

The representative 249-trial inputs are descriptors, not an ordinary unit-test
benchmark.  Tests inject both execution and instrumentation so they remain
fast and do not make wall-clock or resident-memory assertions.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from src.neural_analysis.lfp_summary_models import PPCExecutionConfig
from src.neural_analysis.lfp_summary_ppc_profile import (
    PPCProfileWorkload,
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
