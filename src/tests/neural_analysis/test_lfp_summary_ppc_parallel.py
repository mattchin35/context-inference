"""RED contracts for process-worker PPC execution after the serial benchmark."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ppc_runtime as ppc_runtime
from src.neural_analysis.lfp_summary_models import (
    PPCExecutionConfig,
    default_lfp_summary_config,
)


def _inputs(unit_count: int = 8) -> tuple[object, object, np.ndarray]:
    """Return a small deterministic PPC job with eight independent unit rows.

    Returns
    -------
    prepared_phase : object
        Complex64/Boolean arrays with ``(site, frequency, trial, time)`` axes,
        relative time in seconds, and stable trial IDs.
    prepared_spikes : object
        ``unit_count`` units, each containing one finite seconds array per trial.
    schedule : numpy.ndarray
        Int64 ``(shuffle, trial)`` derangement schedule.
    """
    phase = np.ones((1, 50, 2, 2), dtype=np.complex64)
    prepared_phase = SimpleNamespace(
        phase_tensor=phase,
        phase_valid=np.ones(phase.shape, dtype=bool),
        relative_time_s=np.array([0.0, 1.0]),
        trial_indices=np.array([3, 4], dtype=np.int64),
    )
    trains = tuple(
        SimpleNamespace(
            relative_spike_times=(np.zeros(50), np.zeros(50)),
            overlap_trial_indices=np.empty(0, dtype=np.int64),
        )
        for _ in range(unit_count)
    )
    prepared_spikes = SimpleNamespace(
        unit_ids=tuple(f"ProbeB:{index}" for index in range(unit_count)),
        trial_spike_trains=trains,
    )
    schedule = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int64)
    return prepared_phase, prepared_spikes, schedule


def _config(worker_count: int) -> object:
    """Return a valid deterministic execution configuration for one worker count."""
    config = default_lfp_summary_config()
    return replace(
        config,
        ppc=replace(config.ppc, shuffle_count=3),
        ppc_execution=PPCExecutionConfig(
            unit_block_size=2,
            shuffle_block_size=1,
            trial_edge_block_size=1,
            worker_count=worker_count,
        ),
    )


def _execute(
    tmp_path: Path,
    worker_count: int,
    *,
    progress_callback: object | None = None,
) -> object:
    """Run the shared synthetic job under one requested worker count."""
    phase, spikes, schedule = _inputs()
    config = _config(worker_count)
    return ppc_runtime.execute_ppc_blocks(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
        progress_callback=progress_callback,
    )


@pytest.mark.parametrize("worker_count", (1, 2, 4, 8))
def test_worker_counts_preserve_serial_schedule_and_summary_values(
    tmp_path: Path,
    worker_count: int,
) -> None:
    """One, two, four, and eight workers preserve serial PPC values exactly."""
    serial = _execute(tmp_path / "serial", 1)
    candidate = _execute(tmp_path / f"workers-{worker_count}", worker_count)

    np.testing.assert_array_equal(candidate.schedule, serial.schedule)
    assert candidate.completed_block_ids == serial.completed_block_ids
    for name, serial_values in serial.summary_arrays.items():
        np.testing.assert_allclose(
            candidate.summary_arrays[name], serial_values, rtol=0.0, atol=0.0,
            equal_nan=True,
        )


def test_parallel_dispatch_uses_disjoint_deterministic_block_tasks_and_shared_mmaps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parent dispatches each block once without embedding phase arrays in tasks."""
    observed: dict[str, object] = {}

    def inspect_then_compute(*, phase_descriptor: object, block_tasks: tuple[object, ...],
                             worker_count: int, compute_block: object) -> Iterator[object]:
        """Yield one result at a time after checking the shared-worker contract."""
        observed["worker_count"] = worker_count
        observed["block_ids"] = tuple(task.block_id for task in block_tasks)
        observed["task_attributes"] = tuple(
            tuple(sorted(vars(task))) for task in block_tasks
        )
        phase = np.load(phase_descriptor.phase_path, mmap_mode="r")
        valid = np.load(phase_descriptor.valid_path, mmap_mode="r")
        assert not phase.flags.writeable
        assert not valid.flags.writeable
        assert phase.shape == (1, 50, 2, 2)
        assert valid.shape == phase.shape
        assert all("phase" not in attribute for attributes in observed["task_attributes"] for attribute in attributes)
        assert all("valid" not in attribute for attributes in observed["task_attributes"] for attribute in attributes)
        for task in block_tasks:
            yield compute_block(task)

    monkeypatch.setattr(ppc_runtime, "_run_parallel_worker_batches", inspect_then_compute)
    result = _execute(tmp_path, 4)

    assert observed["worker_count"] == 4
    assert observed["block_ids"] == result.completed_block_ids
    assert len(set(observed["block_ids"])) == len(result.completed_block_ids)
    assert result.run_directory.joinpath("complete.json").is_file()
    assert not list(result.run_directory.rglob("manifest.*"))


def test_parallel_failure_keeps_completed_blocks_resumable_and_never_publishes_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed worker leaves only canonical completed work for an exact retry."""
    calls = 0

    def fail_after_first(*, phase_descriptor: object, block_tasks: tuple[object, ...],
                         worker_count: int, compute_block: object) -> Iterator[object]:
        """Yield one block, then model a worker failure before its successor."""
        nonlocal calls
        for task in block_tasks:
            calls += 1
            if calls == 2:
                raise RuntimeError("injected worker failure")
            yield compute_block(task)

    monkeypatch.setattr(ppc_runtime, "_run_parallel_worker_batches", fail_after_first)
    with pytest.raises(RuntimeError, match="injected worker failure"):
        _execute(tmp_path, 2)

    run_directory = next((tmp_path / "ppc").iterdir())
    completed_before_retry = sorted(
        path.stem.replace(".complete", "")
        for path in (run_directory / "blocks").glob("*.complete.json")
    )
    assert completed_before_retry == ["unit-000000-000001"]
    assert not (run_directory / "complete.json").exists()
    assert not list(run_directory.rglob("manifest.*"))

    monkeypatch.undo()
    resumed = _execute(tmp_path, 2)
    assert resumed.resumed_block_ids == ("unit-000000-000001",)
    assert resumed.completed_block_ids == (
        "unit-000000-000001", "unit-000002-000003",
        "unit-000004-000005", "unit-000006-000007",
    )


def test_parallel_progress_is_parent_only_and_monotonic(tmp_path: Path) -> None:
    """Parallel workers do not emit callback events; parent reports monotonic blocks."""
    events: list[object] = []
    _execute(tmp_path, 4, progress_callback=events.append)

    block_events = [
        event for event in events
        if event.stage in {"observed_reduction", "trial_edge_reduction", "shuffle_aggregation"}
    ]
    for stage in ("observed_reduction", "trial_edge_reduction", "shuffle_aggregation"):
        counts = [event.completed_count for event in block_events if event.stage == stage]
        assert counts == sorted(counts)
        expected = [0, 1, 2, 3, 4] if stage == "observed_reduction" else [1, 2, 3, 4]
        assert counts == expected
    assert all(event.job_id is not None for event in events)


def test_one_worker_uses_exact_serial_path_without_phase_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The approved one-worker path remains the prior serial implementation."""
    def fail_if_parallel(*_: object, **__: object) -> object:
        """Fail if the one-worker implementation creates parallel resources."""
        raise AssertionError("worker_count=1 entered parallel dispatch")

    monkeypatch.setattr(ppc_runtime, "_run_parallel_worker_batches", fail_if_parallel)
    result = _execute(tmp_path, 1)
    assert result.completed_block_ids == (
        "unit-000000-000001", "unit-000002-000003",
        "unit-000004-000005", "unit-000006-000007",
    )


@pytest.mark.parametrize("worker_count", (-1, 0, 1.5, True))
def test_invalid_worker_counts_are_rejected_before_phase_or_work_writes(
    tmp_path: Path,
    worker_count: int | float | bool,
) -> None:
    """Invalid process counts fail validation before workers, phase files, or checkpoints."""
    phase, spikes, schedule = _inputs()
    config = _config(1)
    execution = replace(config.ppc_execution, worker_count=worker_count)

    with pytest.raises(ValueError, match="invalid PPC execution settings"):
        ppc_runtime.execute_ppc_blocks(
            config=config,
            execution=execution,
            prepared_phase=phase,
            prepared_spikes=spikes,
            schedule=schedule,
            work_root=tmp_path,
        )
    assert not tmp_path.exists()
