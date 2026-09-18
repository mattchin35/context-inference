"""RED contracts for process-worker PPC execution after the serial benchmark."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import fields, replace
import inspect
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


def test_parallel_workers_use_spawn_and_keep_a_bounded_ordered_submission_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Workers use ``spawn``, fill the window, then yield canonical task order.

    The fake executor does not execute subprocesses.  It records the parent-side
    executor protocol so this test remains deterministic while guarding the
    production scheduling contract.
    """
    submitted: list[str] = []
    result_waits: list[str] = []
    executor_arguments: dict[str, object] = {}

    class FakeFuture:
        """Future whose result wait records ordering relative to submissions."""

        def __init__(self, task: str) -> None:
            self.task = task

        def result(self) -> str:
            result_waits.append(self.task)
            if self.task == "first":
                assert submitted == ["first", "second"]
            return self.task

    class FakeExecutor:
        """Minimal process-executor double retaining constructor arguments."""

        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            executor_arguments["max_workers"] = max_workers
            executor_arguments["mp_context"] = mp_context

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def submit(self, _compute_block: object, task: str) -> FakeFuture:
            submitted.append(task)
            return FakeFuture(task)

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", FakeExecutor)
    results = ppc_runtime._run_parallel_worker_batches(
        phase_descriptor=object(),
        block_tasks=("first", "second", "third"),
        worker_count=2,
        compute_block=lambda task: task,
    )

    assert next(results) == "first"
    assert submitted == ["first", "second"]
    assert result_waits == ["first"]
    assert next(results) == "second"
    assert submitted == ["first", "second", "third"]
    assert list(results) == ["third"]
    assert result_waits == ["first", "second", "third"]
    assert executor_arguments["max_workers"] == 2
    assert executor_arguments["mp_context"].get_start_method() == "spawn"


def test_parallel_worker_failure_cancels_pending_work_after_prior_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure cancels queued work while preserving the already-yielded block."""
    submitted: list[str] = []
    cancelled: list[str] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakeFuture:
        """Future double with one injected failure and observable cancellation."""

        def __init__(self, task: str) -> None:
            self.task = task

        def result(self) -> str:
            if self.task == "second":
                raise RuntimeError("injected worker failure")
            return self.task

        def cancel(self) -> bool:
            cancelled.append(self.task)
            return True

    class FakeExecutor:
        """Executor double that requires explicit shutdown on exceptional exit."""

        def __init__(self, *, max_workers: int, mp_context: object) -> None:
            del max_workers, mp_context

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def submit(self, _compute_block: object, task: str) -> FakeFuture:
            submitted.append(task)
            return FakeFuture(task)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", FakeExecutor)
    results = ppc_runtime._run_parallel_worker_batches(
        phase_descriptor=object(),
        block_tasks=("first", "second", "third"),
        worker_count=2,
        compute_block=lambda task: task,
    )

    assert next(results) == "first"
    with pytest.raises(RuntimeError, match="injected worker failure"):
        next(results)

    assert submitted == ["first", "second", "third"]
    assert cancelled == ["second", "third"]
    assert shutdown_calls == [(True, True)]


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
    assert not (tmp_path / "ppc").exists()


# S7 grouped-worker contracts.  The legacy ``execute_ppc_blocks`` cases above
# intentionally remain frozen S0 regressions; these tests name the new grouped
# executor seam rather than extending its single-job task contract.


def test_grouped_parallel_worker_task_and_initializer_are_spawn_safe_top_level() -> None:
    """Grouped workers use phase-free site/unit tasks and top-level callables.

    ``_GroupedParallelBlockTask`` describes exactly one stable site and one
    half-open unit block.  It deliberately owns neither prepared tensors nor
    compact common axes/ragged spike arrays: workers open those once from the
    shared input descriptor.  The private names are frozen because process
    targets must remain importable under the ``spawn`` start method.
    """
    task_type = ppc_runtime._GroupedParallelBlockTask
    task_fields = {field.name for field in fields(task_type)}
    assert {
        "block_id",
        "site_index",
        "unit_start",
        "unit_stop",
        "site_jobs",
        "site_condition_batches",
        "site_union_source_trial_row",
        "site_union_target_trial_row",
        "staged_result_path",
    } <= task_fields
    assert not {
        "phase",
        "valid",
        "prepared_phase",
        "component_plan",
        "prepared_spikes",
        "relative_time_s",
        "stable_trial_rows",
        "unit_spike_trains",
        "worker_input_descriptor",
        "progress_callback",
        "run_directory",
        "checkpoint_writer",
    } & task_fields
    descriptor_fields = {field.name for field in fields(ppc_runtime._PhaseWorkDescriptor)}
    assert descriptor_fields == {
        "phase_path",
        "valid_path",
        "relative_time_s_path",
        "stable_trial_rows_path",
        "spike_times_s_path",
        "spike_offsets_path",
    }
    result_fields = {field.name for field in fields(ppc_runtime._GroupedParallelBlockResult)}
    assert {
        "block_id",
        "site_index",
        "unit_start",
        "unit_stop",
        "staged_result_path",
    } == result_fields
    for callable_name in (
        "_initialize_grouped_parallel_worker",
        "_compute_grouped_parallel_block",
    ):
        callable_value = getattr(ppc_runtime, callable_name)
        assert inspect.isfunction(callable_value)
        assert callable_value.__module__ == ppc_runtime.__name__
        assert callable_value.__qualname__ == callable_name


def test_grouped_parallel_runner_uses_spawn_initializer_and_cancels_failed_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The grouped runner bounds work, preserves yields, and cleans up failure.

    This pure executor double avoids process creation while binding the parent
    protocol: fill a two-task window, yield a completed first block, then on
    the second block's failure cancel the current and queued work before a
    wait-for-workers shutdown.  The shared mmap descriptor is supplied once as
    the process initializer argument rather than copied into every task.
    """
    submitted: list[str] = []
    cancelled: list[str] = []
    shutdown_calls: list[tuple[bool, bool]] = []
    executor_arguments: dict[str, object] = {}
    descriptor = object()
    worker = object()

    class FakeFuture:
        """Future double with deterministic first success and second failure."""

        def __init__(self, task: str) -> None:
            self.task = task

        def result(self) -> str:
            if self.task == "second":
                raise RuntimeError("injected grouped worker failure")
            return self.task

        def cancel(self) -> bool:
            cancelled.append(self.task)
            return True

    class FakeExecutor:
        """Process-pool double retaining spawn and initializer arguments."""

        def __init__(
            self,
            *,
            max_workers: int,
            mp_context: object,
            initializer: object,
            initargs: tuple[object, ...],
        ) -> None:
            executor_arguments.update(
                {
                    "max_workers": max_workers,
                    "mp_context": mp_context,
                    "initializer": initializer,
                    "initargs": initargs,
                }
            )

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def submit(self, submitted_worker: object, task: str) -> FakeFuture:
            assert submitted_worker is worker
            submitted.append(task)
            return FakeFuture(task)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", FakeExecutor)
    results = ppc_runtime._run_grouped_parallel_block_batches(
        phase_descriptor=descriptor,
        block_tasks=("first", "second", "third"),
        worker_count=2,
        compute_block=worker,
    )

    assert next(results) == "first"
    # The parent owns the first result until it asks for another one, which is
    # its opportunity to copy/checkpoint before the runner submits a replacement.
    assert submitted == ["first", "second"]
    with pytest.raises(RuntimeError, match="injected grouped worker failure"):
        next(results)
    assert submitted == ["first", "second", "third"]
    assert cancelled == ["second", "third"]
    assert shutdown_calls == [(True, True)]
    assert executor_arguments["max_workers"] == 2
    assert executor_arguments["mp_context"].get_start_method() == "spawn"
    assert executor_arguments["initializer"] is ppc_runtime._initialize_grouped_parallel_worker
    assert executor_arguments["initargs"] == (descriptor,)


def test_grouped_parallel_runner_yields_canonical_tasks_despite_out_of_order_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner, rather than the parent, hides out-of-order worker readiness.

    The second submitted future is marked ready before the first.  The runner
    still waits/yields in task order, so a parent holding one yielded result can
    merge and checkpoint it before requesting the next canonical block.
    """
    readiness_order: list[str] = []
    result_waits: list[str] = []
    descriptor = object()
    worker = object()

    class FakeFuture:
        """Future with an externally recorded readiness order."""

        def __init__(self, task: str) -> None:
            self.task = task

        def result(self) -> str:
            assert readiness_order.index("second") < readiness_order.index("first")
            result_waits.append(self.task)
            return self.task

    class FakeExecutor:
        """Process-pool double that makes task two ready before task one."""

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def submit(self, submitted_worker: object, task: str) -> FakeFuture:
            assert submitted_worker is worker
            if task == "second":
                readiness_order.extend(("second", "first"))
            elif task == "third":
                readiness_order.append("third")
            return FakeFuture(task)

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", FakeExecutor)
    results = ppc_runtime._run_grouped_parallel_block_batches(
        phase_descriptor=descriptor,
        block_tasks=("first", "second", "third"),
        worker_count=2,
        compute_block=worker,
    )

    assert next(results) == "first"
    assert result_waits == ["first"]
    assert next(results) == "second"
    assert result_waits == ["first", "second"]
    assert list(results) == ["third"]
    assert result_waits == ["first", "second", "third"]


def test_component_site_union_views_scalar_scan_sorted_site_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One site union is found without component-length index temporaries.

    The component union is already site-sorted.  The private helper must scan
    its scalar categorical positions and return borrowed source/target slices,
    rather than construct Boolean masks or position arrays proportional to all
    component edges.
    """
    plan = SimpleNamespace(
        edge_site_index=np.array([0, 0, 1, 1, 2], dtype=np.int64),
        stable_edge_source_trial_row=np.array([4, 5, 8, 9, 12], dtype=np.int64),
        stable_edge_target_trial_row=np.array([6, 7, 10, 11, 13], dtype=np.int64),
    )

    def reject_vector_temporary(*_: object, **__: object) -> object:
        raise AssertionError("site-union lookup allocated a component-length temporary")

    monkeypatch.setattr(ppc_runtime.np, "flatnonzero", reject_vector_temporary)
    monkeypatch.setattr(ppc_runtime.np, "arange", reject_vector_temporary)
    monkeypatch.setattr(ppc_runtime.np, "array_equal", reject_vector_temporary)

    source, target, offset = ppc_runtime._component_site_union_views(plan, 1)

    assert offset == 2
    assert source.tolist() == [8, 9]
    assert target.tolist() == [10, 11]
    assert np.shares_memory(source, plan.stable_edge_source_trial_row)
    assert np.shares_memory(target, plan.stable_edge_target_trial_row)


def test_component_site_union_views_empty_middle_and_end_borrow_insertion_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty sites borrow zero-edge slices at their sorted insertion positions.

    A component may have no physical edges for an interior or trailing site.
    The helper must retain a non-owning empty slice of the component arrays
    and report the number of preceding union edges, so later job positions
    remain correctly rebased without allocating a component-length mask.
    """
    source_rows = np.array([4, 5, 12, 13], dtype=np.int64)
    target_rows = np.array([6, 7, 14, 15], dtype=np.int64)
    source_rows.setflags(write=False)
    target_rows.setflags(write=False)
    plan = SimpleNamespace(
        edge_site_index=np.array([0, 0, 2, 2], dtype=np.int64),
        stable_edge_source_trial_row=source_rows,
        stable_edge_target_trial_row=target_rows,
    )

    def reject_vector_temporary(*_: object, **__: object) -> object:
        raise AssertionError("site-union lookup allocated a component-length temporary")

    monkeypatch.setattr(ppc_runtime.np, "flatnonzero", reject_vector_temporary)
    monkeypatch.setattr(ppc_runtime.np, "arange", reject_vector_temporary)
    monkeypatch.setattr(ppc_runtime.np, "array_equal", reject_vector_temporary)

    middle_source, middle_target, middle_offset = ppc_runtime._component_site_union_views(
        plan,
        1,
    )
    end_source, end_target, end_offset = ppc_runtime._component_site_union_views(plan, 3)

    for values, parent in (
        (middle_source, source_rows),
        (middle_target, target_rows),
        (end_source, source_rows),
        (end_target, target_rows),
    ):
        assert values.dtype == np.dtype(np.int64)
        assert values.shape == (0,)
        assert not values.flags.writeable
        assert not values.flags.owndata
        base_chain: list[object] = []
        candidate: object | None = values
        while candidate is not None:
            base_chain.append(candidate)
            candidate = getattr(candidate, "base", None)
        assert any(candidate is parent for candidate in base_chain)

    assert middle_offset == 2
    assert end_offset == 4


@pytest.mark.parametrize("empty_site", (1, 2))
def test_planned_zero_edge_middle_and_end_sites_keep_scalar_union_offsets(
    empty_site: int,
) -> None:
    """Real planner output preserves insertion offsets for zero-edge sites.

    This integrates the scalar view contract with a valid three-site component
    plan, rather than relying only on a hand-built union.  The empty middle
    site must rebase after site zero; the empty final site must rebase after
    every preceding physical edge.
    """
    config = _config(worker_count=2)
    trial_count = 3
    site_valid = np.ones((len(config.sites), trial_count), dtype=bool)
    site_valid[empty_site] = False
    plan = ppc_runtime.plan_grouped_ppc_component(
        config=config,
        condition_names=("all",),
        condition_membership=np.ones((trial_count, 1), dtype=bool),
        site_ids=tuple(site.stable_id for site in config.sites),
        site_trial_valid=site_valid,
        stable_trial_rows=np.arange(10, 10 + trial_count, dtype=np.int64),
        source_trial_spike_count=np.zeros((trial_count, 1, 2), dtype=np.int64),
        frequency_count=2,
        shared_phase_mmap_bytes=0,
    )

    source, target, offset = ppc_runtime._component_site_union_views(plan, empty_site)

    assert source.shape == target.shape == (0,)
    for values, parent in (
        (source, plan.stable_edge_source_trial_row),
        (target, plan.stable_edge_target_trial_row),
    ):
        # ``shares_memory`` reports false for zero-length views, so follow the
        # base chain to prove that the empty result still borrows its union.
        assert not values.flags.owndata
        base_chain: list[object] = []
        candidate: object | None = values
        while candidate is not None:
            base_chain.append(candidate)
            candidate = getattr(candidate, "base", None)
        assert any(candidate is parent for candidate in base_chain)
    assert offset == int(np.count_nonzero(plan.edge_site_index < empty_site))


def test_grouped_parallel_runner_cancels_pending_window_when_generator_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing after one publication cancels the remaining current-site work.

    A parent merge/checkpoint/progress failure closes the runner while it is
    paused at ``yield``.  The runner itself must cancel the pending future and
    wait for worker cleanup; relying on executor context-manager exit leaves
    queued work alive and risks a false later publication.
    """
    submitted: list[str] = []
    cancelled: list[str] = []
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakeFuture:
        """Future double that records cancellation without holding an array."""

        def __init__(self, task: str) -> None:
            self.task = task

        def result(self) -> str:
            return self.task

        def cancel(self) -> bool:
            cancelled.append(self.task)
            return True

    class FakeExecutor:
        """Executor double whose context exit does not provide cleanup."""

        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "FakeExecutor":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def submit(self, _worker: object, task: str) -> FakeFuture:
            submitted.append(task)
            return FakeFuture(task)

        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            shutdown_calls.append((wait, cancel_futures))

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", FakeExecutor)
    results = ppc_runtime._run_grouped_parallel_block_batches(
        phase_descriptor=object(),
        block_tasks=("first", "second", "third"),
        worker_count=2,
        compute_block=object(),
    )

    assert next(results) == "first"
    assert submitted == ["first", "second"]
    results.close()

    assert cancelled == ["second"]
    assert shutdown_calls == [(True, True)]
