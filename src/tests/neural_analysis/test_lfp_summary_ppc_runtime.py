"""RED contracts for serial PPC checkpoint/restart execution and progress."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, fields, replace
import gc
import inspect
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
from typing import Mapping
import weakref

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ppc_runtime as ppc_runtime
from src.neural_analysis import lfp_summary_runtime
from src.neural_analysis import lfp_summary_work_cache as work_cache
from src.neural_analysis import spike_lfp_summary
from src.neural_analysis.lfp_summary_models import (
    PPCExecutionConfig,
    UnitPopulationConfig,
    component_fingerprint,
    default_lfp_summary_config,
    fingerprint_source_files,
)
from src.neural_analysis.lfp_summary_preparation import (
    PreparedTrials,
    TrialRelativeSpikeTrains,
    build_common_event_grid,
)


def _inputs() -> tuple[object, object, np.ndarray]:
    """Return finite synthetic one-job phase/spike inputs and a deterministic schedule.

    Returns
    -------
    prepared_phase : object
        Namespace with complex64 phase `(site, frequency, trial, time)`, Boolean
        validity on identical axes, float64 relative seconds, and stable IDs.
    prepared_spikes : object
        Namespace with one `(unit, trial)` collection of finite relative seconds.
    schedule : numpy.ndarray
        Int64 `(shuffle, trial)` derangements with no fixed points.
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
    return prepared_phase, prepared_spikes, np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int64)


def _config() -> object:
    """Return a valid serial execution configuration for one synthetic PPC job."""
    return replace(
        default_lfp_summary_config(),
        ppc=replace(default_lfp_summary_config().ppc, shuffle_count=3),
        ppc_execution=PPCExecutionConfig(
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
            worker_count=1,
        ),
    )


def test_run_identity_and_block_ids_are_deterministic_and_schedule_is_saved(tmp_path: Path) -> None:
    """Equivalent jobs use one stable run identity, block identity, and persisted schedule."""
    config = _config()
    phase, spikes, schedule = _inputs()

    first = ppc_runtime.execute_ppc_blocks(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )
    second = ppc_runtime.execute_ppc_blocks(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )

    assert first.run_fingerprint == second.run_fingerprint
    assert first.completed_block_ids == second.completed_block_ids
    np.testing.assert_array_equal(first.schedule, schedule)
    assert (first.run_directory / "schedule.npz").is_file()


def test_validated_resume_equals_cold_and_corrupt_or_orphan_blocks_recompute(tmp_path: Path) -> None:
    """Only NPZ-plus-marker exact checkpoints resume; corrupt/orphan work is recomputed."""
    config = _config()
    phase, spikes, schedule = _inputs()
    cold = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "cold",
    )
    resumed = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "warm",
    )
    original_sampler = ppc_runtime._compute_scheduled_shuffle_draws

    def fail_if_resampled(*_: object, **__: object) -> None:
        """Fail if a fully validated checkpoint is sampled again on exact resume."""
        raise AssertionError("validated checkpoint was recomputed")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(ppc_runtime, "_compute_scheduled_shuffle_draws", fail_if_resampled)
    resumed_exact = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "warm",
    )
    monkeypatch.undo()
    assert resumed_exact.resumed_block_ids == resumed_exact.completed_block_ids
    (resumed.run_directory / "blocks" / f"{resumed.completed_block_ids[0]}.complete.json").unlink()
    resumed_again = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "warm",
    )

    assert not hasattr(cold, "null_ppc")
    for name in cold.summary_arrays:
        np.testing.assert_allclose(
            cold.summary_arrays[name], resumed.summary_arrays[name], equal_nan=True,
        )
        np.testing.assert_allclose(
            cold.summary_arrays[name], resumed_again.summary_arrays[name], equal_nan=True,
        )
    assert resumed_again.resumed_block_ids == ()


def test_progress_is_ordered_monotonic_and_eta_starts_after_two_timed_blocks(tmp_path: Path) -> None:
    """Structured runtime progress follows approved stages and delayed ETA semantics."""
    config = _config()
    phase, spikes, schedule = _inputs()
    events: list[object] = []
    ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
        progress_callback=events.append,
    )

    stages = [event.stage for event in events]
    assert stages == sorted(stages, key=("prepare_phase", "observed_reduction", "trial_edge_reduction", "shuffle_aggregation", "fdr", "checkpoint", "commit").index)
    assert all(event.elapsed_seconds >= 0.0 for event in events)
    assert all(event.eta_seconds is None for event in events[:2])
    assert any(event.eta_seconds is not None for event in events[2:])


def test_inference_ineligible_entries_never_invoke_permutation_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entries below fifty phases or two spike-contributing trials bypass null sampling."""
    config = _config()
    phase, spikes, schedule = _inputs()
    spikes.trial_spike_trains[0].relative_spike_times = (np.array([0.0]), np.array([]))

    def fail(*_: object, **__: object) -> None:
        """Fail if the runtime attempts scheduled permutation sampling for an ineligible entry."""
        raise AssertionError("ineligible entry sampled permutation phases")

    monkeypatch.setattr(ppc_runtime, "_compute_scheduled_shuffle_draws", fail)
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )

    assert np.isnan(result.summary_arrays["p_value"]).all()


def test_execution_result_is_work_only_and_cleanup_obeys_retention_policy(tmp_path: Path) -> None:
    """Execution leaves final manifest/schema untouched and cleans only its exact successful run."""
    config = _config()
    phase, spikes, schedule = _inputs()
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )

    assert not (tmp_path / "manifest.json").exists()
    assert not (tmp_path / "spike_phase.npz").exists()
    assert not hasattr(result, "final_payload_arrays")
    assert result.run_directory.exists()


def test_execution_result_contains_only_final_summary_arrays_not_shuffle_draws(tmp_path: Path) -> None:
    """Checkpoints/results retain PPC null summaries, never a full shuffle-by-unit tensor."""
    config = _config()
    phase, spikes, schedule = _inputs()
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )

    required = {
        "ppc", "spike_count", "reliable", "null_eligible",
        "computable", "resultant_length", "preferred_phase_rad",
        "null_exceedance_count", "permutation_count", "p_value", "q_value",
        "significant", "null_mean", "null_std", "null_p025", "null_p50", "null_p975",
    }
    assert required <= set(result.summary_arrays)
    assert not hasattr(result, "null_ppc")
    assert all(array.ndim <= 2 for array in result.summary_arrays.values())


def test_checkpoint_schedule_is_generated_once_and_blocks_store_summaries_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One job persists one schedule and checkpoint NPZs never contain raw shuffle draws."""
    config = _config()
    phase, spikes, schedule = _inputs()
    calls = 0
    original = ppc_runtime.generate_trial_derangement_schedule

    def recording_schedule(*args: object, **kwargs: object) -> np.ndarray:
        """Count deterministic schedule creation while delegating unchanged scientific inputs."""
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(ppc_runtime, "generate_trial_derangement_schedule", recording_schedule)
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=None, work_root=tmp_path,
    )

    assert calls == 1
    with np.load(next((result.run_directory / "blocks").glob("*.npz")), allow_pickle=False) as block:
        assert "null_ppc" not in block.files
        assert all("shuffle" not in name for name in block.files)


def test_checkpoint_corrupt_and_orphan_blocks_recompute_without_losing_valid_siblings(tmp_path: Path) -> None:
    """Marker-backed invalid shape/dtype blocks and orphan files never resume; valid blocks do."""
    config = _config()
    phase, spikes, schedule = _inputs()
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    blocks = result.run_directory / "blocks"
    block_id = result.completed_block_ids[0]
    np.savez(blocks / f"{block_id}.npz", ppc=np.array(["wrong dtype"], dtype="<U16"))
    np.savez(blocks / "orphan.npz", ppc=np.array([1.0]))
    (blocks / "orphan.complete.json").write_text('{"block_id":"orphan"}', encoding="ascii")

    rerun = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )

    assert block_id not in rerun.resumed_block_ids


def test_schedule_mismatch_and_invalid_checkpoint_metadata_shape_dtype_recompute(tmp_path: Path) -> None:
    """Only checkpoints matching run metadata, schedule, shape, and dtype can resume."""
    config = _config()
    phase, spikes, schedule = _inputs()
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    np.savez(result.run_directory / "schedule.npz", schedule=np.array([[1, 0]], dtype=np.int64))
    recomputed = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )

    assert recomputed.resumed_block_ids == ()


def test_execution_block_sizes_preserve_summarized_results(tmp_path: Path) -> None:
    """Execution-only unit/edge/shuffle sizes do not alter any scientific summary array."""
    phase, spikes, schedule = _inputs()
    first_config = _config()
    second_config = replace(
        first_config,
        ppc_execution=replace(first_config.ppc_execution, shuffle_block_size=3),
    )
    first = ppc_runtime.execute_ppc_blocks(
        config=first_config, execution=first_config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "one",
    )
    second = ppc_runtime.execute_ppc_blocks(
        config=second_config, execution=second_config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "two",
    )

    for name in first.summary_arrays:
        np.testing.assert_allclose(first.summary_arrays[name], second.summary_arrays[name], equal_nan=True)


@pytest.mark.parametrize(
    "spike_trains",
    (
        (np.zeros(24), np.zeros(25)),
        (np.zeros(50), np.array([], dtype=float)),
    ),
)
def test_execution_ineligible_cases_bypass_permutation_sampling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spike_trains: tuple[np.ndarray, np.ndarray],
) -> None:
    """Low spike count and too-few contributing trials independently bypass null edges."""
    config = _config()
    phase, spikes, schedule = _inputs()
    spikes.trial_spike_trains[0].relative_spike_times = spike_trains
    monkeypatch.setattr(
        ppc_runtime,
        "_compute_scheduled_shuffle_draws",
        lambda **_: (_ for _ in ()).throw(AssertionError("ineligible sampled")),
    )
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert not result.summary_arrays["null_eligible"].any()


def test_checkpoint_progress_counts_are_stage_monotonic_and_eta_waits_for_two_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ETA is based on two completed timed blocks, with stable totals within every stage."""
    config = replace(_config(), ppc_execution=replace(_config().ppc_execution, shuffle_block_size=1))
    phase, spikes, schedule = _inputs()
    timestamps = iter(float(value) for value in range(100))
    monkeypatch.setattr(ppc_runtime.time, "perf_counter", lambda: next(timestamps))
    events: list[object] = []
    ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
        progress_callback=events.append,
    )
    by_stage: dict[str, list[object]] = {}
    for event in events:
        by_stage.setdefault(event.stage, []).append(event)
    for stage_events in by_stage.values():
        totals = {event.total_count for event in stage_events}
        assert len(totals) == 1
        assert all(0 <= event.completed_count <= event.total_count for event in stage_events)
        assert [event.completed_count for event in stage_events] == sorted(event.completed_count for event in stage_events)
    completed_blocks = [event for event in events if event.stage == "checkpoint" and event.completed_count > 0]
    assert all(event.eta_seconds is None for event in completed_blocks[:1])
    assert any(event.eta_seconds is not None for event in completed_blocks[1:])


def test_run_identity_binds_selected_job_inputs_and_source_representation(
    tmp_path: Path,
) -> None:
    """Equal schedules cannot reuse work across distinct sites, trials, units, or phase values."""
    config = _config()
    phase, spikes, schedule = _inputs()
    first = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    exact = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert exact.resumed_block_ids == exact.completed_block_ids
    phase.site_id = "HPC1"
    phase.condition_name = "omission"
    phase.epoch_bounds_s = (-1.0, 1.0)
    phase.trial_indices = np.array([8, 9], dtype=np.int64)
    phase.phase_tensor[0, 0, 0, 0] = 1.0j
    spikes.unit_ids = ("HPC1:1",)
    changed = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert changed.run_fingerprint != first.run_fingerprint
    assert changed.resumed_block_ids == ()


def test_frequency_axis_mismatch_is_rejected_without_repeating_phase_data(
    tmp_path: Path,
) -> None:
    """A one-frequency phase tensor cannot be silently broadcast to the configured grid."""
    config = _config()
    phase, spikes, schedule = _inputs()
    phase.phase_tensor = phase.phase_tensor[:, :1]
    phase.phase_valid = phase.phase_valid[:, :1]
    with pytest.raises(ValueError, match="frequency"):
        ppc_runtime.execute_ppc_blocks(
            config=config, execution=config.ppc_execution, prepared_phase=phase,
            prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
        )


def test_executor_uses_bounded_edge_reduction_not_full_shuffle_draw_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production execution streams scheduled edges and never calls the full-draw helper."""
    config = _config()
    phase, spikes, schedule = _inputs()
    edge_shapes: list[tuple[int, int, int]] = []
    edge_reducer = ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics

    def recording_edge_reducer(**kwargs: object):
        """Record bounded unit/frequency/edge dimensions before delegating."""
        trains = kwargs["trial_relative_spike_times_s"]
        phase_vectors = kwargs["trial_phase_vectors"]
        source_edges = kwargs["source_trial_position"]
        edge_shapes.append((len(trains), phase_vectors.shape[1], source_edges.size))
        return edge_reducer(**kwargs)

    monkeypatch.setattr(
        ppc_runtime,
        "_compute_scheduled_shuffle_draws",
        lambda **_: (_ for _ in ()).throw(AssertionError("full shuffle helper called")),
    )
    monkeypatch.setattr(
        ppc_runtime.spike_lfp_summary,
        "compute_edge_sufficient_statistics",
        recording_edge_reducer,
    )
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert result.summary_arrays["permutation_count"].max() == schedule.shape[0]
    assert edge_shapes
    assert all(unit <= config.ppc_execution.unit_block_size for unit, _, _ in edge_shapes)
    assert all(edge <= config.ppc_execution.trial_edge_block_size for _, _, edge in edge_shapes)
    assert not hasattr(result, "null_ppc")


def test_execution_samples_each_scheduled_edge_once_per_unit_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduled cross-trial edges are sampled once per eligible `(unit block, edge)`.

    The synthetic job has one eligible unit block, two trial positions, and four
    shuffle rows aggregated in two shuffle blocks.  The wrapped reducer records
    only cross-trial calls, excluding observed same-trial reductions.  Each
    directed edge must therefore occur exactly once regardless of the number of
    shuffle aggregation blocks.
    """
    config = replace(
        _config(),
        ppc_execution=replace(_config().ppc_execution, shuffle_block_size=2),
    )
    phase, spikes, _ = _inputs()
    schedule = np.tile(np.array([[1, 0]], dtype=np.int64), (4, 1))
    sampled_edges: Counter[tuple[int, int]] = Counter()
    original_reducer = ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics

    def recording_reducer(**kwargs: object):
        """Record scheduled `(source trial, target trial)` int64 edge pairs."""
        source = np.asarray(kwargs["source_trial_position"], dtype=np.int64)
        target = np.asarray(kwargs["target_trial_position"], dtype=np.int64)
        for source_trial, target_trial in zip(source, target, strict=True):
            if source_trial != target_trial:
                sampled_edges[(int(source_trial), int(target_trial))] += 1
        return original_reducer(**kwargs)

    monkeypatch.setattr(
        ppc_runtime.spike_lfp_summary,
        "compute_edge_sufficient_statistics",
        recording_reducer,
    )
    ppc_runtime.execute_ppc_blocks(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )

    assert sampled_edges == Counter({(0, 1): 1, (1, 0): 1})


def test_checkpoint_disabled_and_lock_contracts_prevent_work_artifacts(
    tmp_path: Path,
) -> None:
    """Disabled checkpoints create no work tree, while an active exact lock blocks mutation."""
    config = _config()
    phase, spikes, schedule = _inputs()
    disabled = replace(
        config,
        ppc_execution=replace(config.ppc_execution, checkpoint_enabled=False),
    )
    ppc_runtime.execute_ppc_blocks(
        config=disabled, execution=disabled.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "disabled",
    )
    assert not (tmp_path / "disabled" / "ppc").exists()
    run = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "locked",
    )
    lock = run.run_directory / "executor.lock"
    lock.write_text("active\n", encoding="ascii")
    with pytest.raises(FileExistsError, match="executor.lock"):
        ppc_runtime.execute_ppc_blocks(
            config=config, execution=config.ppc_execution, prepared_phase=phase,
            prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "locked",
        )
    assert lock.exists()


def test_progress_events_identify_each_ppc_job_and_outer_commit_is_separate(
    tmp_path: Path,
) -> None:
    """Every executor event carries a stable job identity rather than a component-only label."""
    config = _config()
    phase, spikes, schedule = _inputs()
    phase.site_id = "PFC"
    phase.condition_name = "correct_rewarded"
    phase.epoch_bounds_s = (-2.0, 2.0)
    events: list[object] = []
    ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
        progress_callback=events.append,
    )
    assert events
    assert {event.job_id for event in events} == {"PFC:correct_rewarded:-2.0:2.0"}
    assert all(event.stage != "commit_component" for event in events)


def test_run_metadata_declares_complete_job_identity_and_execution_contract(
    tmp_path: Path,
) -> None:
    """A resumable work directory records source, job, schedule, and block identities."""
    import json

    config = _config()
    phase, spikes, schedule = _inputs()
    phase.site_id = "PFC"
    phase.condition_name = "correct_rewarded"
    phase.epoch_bounds_s = (-2.0, 2.0)
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    metadata = json.loads((result.run_directory / "metadata.json").read_text())
    required = {
        "job_id", "source_fingerprint", "scientific_fingerprint",
        "representation_fingerprint", "site_id", "condition_name",
        "epoch_bounds_s", "trial_indices", "unit_ids", "population_id",
        "spike_fingerprint", "schedule_seed", "schedule_fingerprint",
        "schema_version", "code_version", "axes", "shapes", "dtypes",
        "execution_settings", "completed_block_ids",
    }
    assert required <= set(metadata)


def test_mixed_unit_frequency_eligibility_samples_only_eligible_subsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Null reduction excludes sub-reliable frequencies and one-trial units before edge sampling."""
    config = _config()
    config = replace(
        config,
        phase=replace(config.phase, frequency_hz=tuple(config.phase.frequency_hz[:2])),
    )
    phase, spikes, schedule = _inputs()
    phase.phase_tensor = phase.phase_tensor[:, :2]
    phase.phase_valid = np.ones(phase.phase_tensor.shape, dtype=bool)
    # Unit one has 100 valid frequency-zero phases.  Frequency one has only
    # twenty-five valid phases, while unit two has valid spikes in one trial.
    phase.phase_valid[0, 1, :, :] = False
    phase.phase_valid[0, 1, 0, 0] = True
    spikes.trial_spike_trains = (
        SimpleNamespace(
            relative_spike_times=(
                np.concatenate((np.zeros(25), np.ones(25))),
                np.concatenate((np.zeros(25), np.ones(25))),
            ),
            overlap_trial_indices=np.empty(0, dtype=np.int64),
        ),
    )
    spikes.unit_ids = ("PFC:1", "PFC:2")
    spikes.trial_spike_trains = (
        spikes.trial_spike_trains[0],
        SimpleNamespace(
            relative_spike_times=(np.zeros(50), np.empty(0)),
            overlap_trial_indices=np.empty(0, dtype=np.int64),
        ),
    )
    sampled: list[tuple[int, tuple[float, ...]]] = []
    original_reducer = ppc_runtime.spike_lfp_summary.compute_edge_sufficient_statistics

    def recording_reducer(**kwargs: object):
        """Record only shuffled, nonself trial edges before computing statistics."""
        source = np.asarray(kwargs["source_trial_position"])
        target = np.asarray(kwargs["target_trial_position"])
        if np.any(source != target):
            sampled.append(
                (
                    len(kwargs["trial_relative_spike_times_s"]),
                    tuple(np.asarray(kwargs["frequencies_hz"], dtype=float)),
                ),
            )
        return original_reducer(**kwargs)

    monkeypatch.setattr(
        ppc_runtime.spike_lfp_summary,
        "compute_edge_sufficient_statistics",
        recording_reducer,
    )
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert sampled
    assert all(
        unit_count == 1 and frequencies == (config.phase.frequency_hz[0],)
        for unit_count, frequencies in sampled
    )
    assert np.isnan(result.summary_arrays["p_value"][0, 1])
    assert np.isnan(result.summary_arrays["q_value"][0, 1])
    assert np.isnan(result.summary_arrays["null_mean"][0, 1])
    assert result.summary_arrays["permutation_count"][0, 1] == 0
    assert not result.summary_arrays["null_eligible"][0, 1]
    assert not result.summary_arrays["null_eligible"][1].any()


def test_executor_matches_wp5b_reference_and_keeps_zero_valid_phase_nan(
    tmp_path: Path,
) -> None:
    """Seeded serial summaries equal the established WP5B observed/null reference numerics."""
    config = _config()
    config = replace(
        config,
        phase=replace(config.phase, frequency_hz=tuple(config.phase.frequency_hz[:2])),
    )
    phase, spikes, schedule = _inputs()
    phase.phase_tensor = phase.phase_tensor[:, :2]
    phase.phase_tensor[0, 1] = 0.0j
    phase.phase_valid = np.ones(phase.phase_tensor.shape, dtype=bool)
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    reference = spike_lfp_summary.compute_trial_shuffle_ppc(
        trial_relative_spike_times_s=spikes.trial_spike_trains[0].relative_spike_times,
        phase_time_s=phase.relative_time_s,
        trial_phase_vectors=phase.phase_tensor[0].transpose(1, 0, 2),
        frequencies_hz=np.asarray(config.phase.frequency_hz[:2]),
        schedule=schedule,
    )
    observed = spike_lfp_summary.compute_observed_ppc(
        probe_label="PFC",
        cluster_id=1,
        spike_phase_vectors=np.vstack((np.ones(100, dtype=np.complex64), np.zeros(100, dtype=np.complex64))),
        valid_mask=np.vstack((np.ones(100, dtype=bool), np.zeros(100, dtype=bool))),
        frequencies_hz=np.asarray(config.phase.frequency_hz),
        spike_times_s=np.zeros(100, dtype=float),
    )
    reference_q = spike_lfp_summary.adjust_ppc_pvalues_bh(
        p_value=reference.null_summary.p_value[None, None, None, None, :],
        null_eligible=reference.null_summary.null_eligible[None, None, None, None, :],
    )[0, 0, 0, 0]
    np.testing.assert_allclose(result.summary_arrays["ppc"][0], reference.observed_ppc, equal_nan=True)
    np.testing.assert_array_equal(result.summary_arrays["spike_count"][0], reference.spike_count)
    np.testing.assert_allclose(result.summary_arrays["resultant_length"][0], observed.resultant_length, equal_nan=True)
    np.testing.assert_allclose(result.summary_arrays["preferred_phase_rad"][0], observed.preferred_phase_rad, equal_nan=True)
    for field in ("p_value", "null_mean", "null_std", "null_p025", "null_p50", "null_p975"):
        np.testing.assert_allclose(
            result.summary_arrays[field][0], getattr(reference.null_summary, field), equal_nan=True,
        )
    np.testing.assert_allclose(result.summary_arrays["q_value"][0], reference_q, equal_nan=True)
    np.testing.assert_array_equal(
        result.summary_arrays["significant"][0],
        reference.null_summary.null_eligible & (reference_q <= config.ppc.fdr_alpha),
    )
    assert np.isnan(result.summary_arrays["preferred_phase_rad"][0, 1])


def test_unit_block_checkpoints_resume_valid_siblings_after_interruption_and_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit blocks checkpoint independently, so only unfinished or corrupt blocks recompute."""
    config = _config()
    phase, spikes, schedule = _inputs()
    unit = spikes.trial_spike_trains[0]
    spikes.unit_ids = ("PFC:1", "PFC:2", "PFC:3")
    spikes.trial_spike_trains = (unit, unit, unit)
    written: list[str] = []
    original_write = ppc_runtime.write_ppc_checkpoint

    def interrupt_after_first(run_directory: Path, block_id: str, *args: object, **kwargs: object):
        """Persist the first valid block, then simulate interruption before block two."""
        if written:
            raise RuntimeError("interrupted after first block")
        written.append(block_id)
        return original_write(run_directory, block_id, *args, **kwargs)

    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", interrupt_after_first)
    with pytest.raises(RuntimeError, match="interrupted after first block"):
        ppc_runtime.execute_ppc_blocks(
            config=config, execution=config.ppc_execution, prepared_phase=phase,
            prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
        )
    assert written == ["unit-000000-000000"]
    run_directories = list((tmp_path / "ppc").iterdir())
    assert len(run_directories) == 1
    interrupted_directory = run_directories[0]
    assert not (interrupted_directory / "complete.json").exists()
    assert not (tmp_path / "spike_phase.npz").exists()

    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", original_write)
    resumed = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert resumed.completed_block_ids == (
        "unit-000000-000000", "unit-000001-000001", "unit-000002-000002",
    )
    assert resumed.resumed_block_ids == ("unit-000000-000000",)
    corrupt_block = resumed.completed_block_ids[1]
    (resumed.run_directory / "blocks" / f"{corrupt_block}.npz").write_bytes(b"corrupt")
    repaired = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert corrupt_block not in repaired.resumed_block_ids
    assert resumed.completed_block_ids[0] in repaired.resumed_block_ids
    assert resumed.completed_block_ids[2] in repaired.resumed_block_ids


def test_execution_settings_change_work_identity_but_not_component_fingerprint(
    tmp_path: Path,
) -> None:
    """Execution block sizes isolate resumable work without changing final science identity."""
    config = _config()
    changed = replace(
        config,
        ppc_execution=replace(
            config.ppc_execution,
            unit_block_size=2,
            shuffle_block_size=2,
            trial_edge_block_size=2,
        ),
    )
    phase, spikes, schedule = _inputs()
    first = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    second = ppc_runtime.execute_ppc_blocks(
        config=changed, execution=changed.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert first.run_fingerprint != second.run_fingerprint
    assert second.resumed_block_ids == ()
    assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed)


def test_executor_lock_precedes_schedule_and_preserves_external_lock_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The executor owns only its lock and cannot mutate a locked run directory."""
    config = _config()
    phase, spikes, schedule = _inputs()
    original_schedule_write = ppc_runtime._write_schedule
    lock_seen: list[Path] = []

    def recording_schedule_write(run_directory: Path, schedule_array: np.ndarray) -> None:
        """Require an executor-owned lock before the first schedule mutation."""
        lock = run_directory / "executor.lock"
        assert lock.is_file()
        lock_seen.append(lock)
        original_schedule_write(run_directory, schedule_array)

    monkeypatch.setattr(ppc_runtime, "_write_schedule", recording_schedule_write)
    result = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "work",
    )
    assert lock_seen
    assert not (result.run_directory / "executor.lock").exists()
    before_schedule = (result.run_directory / "schedule.npz").read_bytes()
    before_markers = {
        path.name: path.read_bytes()
        for path in (result.run_directory / "blocks").glob("*.complete.json")
    }
    external_lock = result.run_directory / "executor.lock"
    external_lock.write_bytes(b"external executor lock\n")
    with pytest.raises(FileExistsError, match="executor.lock"):
        ppc_runtime.execute_ppc_blocks(
            config=config, execution=config.ppc_execution, prepared_phase=phase,
            prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "work",
        )
    assert external_lock.read_bytes() == b"external executor lock\n"
    assert (result.run_directory / "schedule.npz").read_bytes() == before_schedule
    assert {
        path.name: path.read_bytes()
        for path in (result.run_directory / "blocks").glob("*.complete.json")
    } == before_markers

    monkeypatch.setattr(
        ppc_runtime,
        "_observed",
        lambda *_: (_ for _ in ()).throw(RuntimeError("injected computation failure")),
    )
    with pytest.raises(RuntimeError, match="injected computation failure"):
        ppc_runtime.execute_ppc_blocks(
            config=config, execution=config.ppc_execution, prepared_phase=phase,
            prepared_spikes=spikes, schedule=schedule, work_root=tmp_path / "failure",
        )
    assert not list((tmp_path / "failure").rglob("executor.lock"))


def test_executor_lock_is_an_exact_json_ownership_record_and_live_collision_is_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor locking publishes complete ownership before mutating scheduled work.

    The record binds schema, process, host, scope, and exact run fingerprint.
    A second live owner cannot acquire the same path or alter its bytes.
    """
    config = _config()
    phase, spikes, schedule = _inputs()
    original_schedule_write = ppc_runtime._write_schedule
    observed_record: dict[str, object] = {}

    def inspect_lock_before_schedule(
        run_directory: Path,
        schedule_array: np.ndarray,
    ) -> None:
        """Read the fully published ownership record before schedule mutation."""
        observed_record.update(
            json.loads((run_directory / "executor.lock").read_text(encoding="utf-8"))
        )
        original_schedule_write(run_directory, schedule_array)

    monkeypatch.setattr(ppc_runtime, "_write_schedule", inspect_lock_before_schedule)
    result = ppc_runtime.execute_ppc_blocks(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )

    assert observed_record == {
        "schema_version": "1",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "run_fingerprint": result.run_fingerprint,
        "scope": "executor",
    }
    lock_path = result.run_directory / "executor.lock"
    live_bytes = (json.dumps(observed_record, sort_keys=True) + "\n").encode("utf-8")
    lock_path.write_bytes(live_bytes)
    with pytest.raises(FileExistsError, match="executor.lock"):
        ppc_runtime.execute_ppc_blocks(
            config=config,
            execution=config.ppc_execution,
            prepared_phase=phase,
            prepared_spikes=spikes,
            schedule=schedule,
            work_root=tmp_path,
        )
    assert lock_path.read_bytes() == live_bytes


def test_run_metadata_preserves_selected_trials_and_overlap_identity(
    tmp_path: Path,
) -> None:
    """Selected stable trial IDs and overlap warnings are both resumable job identity."""
    import json

    config = _config()
    phase, spikes, schedule = _inputs()
    phase.trial_indices = np.array([31, 47], dtype=np.int64)
    spikes.trial_spike_trains[0].overlap_trial_indices = np.array([31], dtype=np.int64)
    overlapped = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    metadata = json.loads((overlapped.run_directory / "metadata.json").read_text())
    assert metadata["trial_indices"] == [31, 47]
    assert metadata["overlap_trial_indices"] == [[31]]
    spikes.trial_spike_trains[0].overlap_trial_indices = np.empty(0, dtype=np.int64)
    without_overlap = ppc_runtime.execute_ppc_blocks(
        config=config, execution=config.ppc_execution, prepared_phase=phase,
        prepared_spikes=spikes, schedule=schedule, work_root=tmp_path,
    )
    assert without_overlap.run_fingerprint != overlapped.run_fingerprint


# S3 planner and allocation contracts.  These tests intentionally exercise only
# pure schedule/identity/byte planning; grouped execution remains an S4 task.


def _planner_config(
    *,
    shuffle_count: int = 3,
    seed: int = 17,
    unit_block_size: int = 2,
    shuffle_block_size: int = 2,
    trial_edge_block_size: int = 4,
    worker_count: int = 1,
    maximum_worker_allocation_bytes: int = 2 * 1024**3,
    maximum_aggregate_allocation_bytes: int = 12 * 1024**3,
) -> object:
    """Return small valid settings for a pure grouped-PPC planner fixture.

    The selected source-trial count arrays used below have axes ``(trial,
    unit, segment=2)`` in before/after order.  All memory values are private
    planned NumPy bytes rather than measured process residency.
    """
    base = default_lfp_summary_config()
    return replace(
        base,
        ppc=replace(base.ppc, shuffle_count=shuffle_count, seed=seed),
        ppc_execution=replace(
            base.ppc_execution,
            unit_block_size=unit_block_size,
            shuffle_block_size=shuffle_block_size,
            trial_edge_block_size=trial_edge_block_size,
            worker_count=worker_count,
            maximum_worker_allocation_bytes=maximum_worker_allocation_bytes,
            maximum_aggregate_allocation_bytes=maximum_aggregate_allocation_bytes,
        ),
    )


def _planner_inputs(
    *,
    stable_trial_rows: np.ndarray | None = None,
    condition_membership: np.ndarray | None = None,
    site_trial_valid: np.ndarray | None = None,
    source_trial_spike_count: np.ndarray | None = None,
    condition_names: tuple[str, ...] | None = None,
    site_ids: tuple[str, ...] = ("site-a",),
    frequency_count: int = 2,
    shared_phase_mmap_bytes: int = 4096,
) -> dict[str, object]:
    """Build purely categorical S3 planner inputs with documented axes.

    ``condition_membership`` is Boolean ``(full trial, condition)`` and
    ``site_trial_valid`` is Boolean ``(site, full trial)``.  Stable rows are
    full trial-table identities and deliberately need not equal local positions.
    """
    rows = (
        np.array([101, 303, 709], dtype=np.int64)
        if stable_trial_rows is None
        else np.asarray(stable_trial_rows, dtype=np.int64)
    )
    membership = (
        np.ones((rows.size, 1), dtype=bool)
        if condition_membership is None
        else np.asarray(condition_membership, dtype=bool)
    )
    valid = (
        np.ones((len(site_ids), rows.size), dtype=bool)
        if site_trial_valid is None
        else np.asarray(site_trial_valid, dtype=bool)
    )
    spike_counts = (
        np.zeros((rows.size, 2, 2), dtype=np.int64)
        if source_trial_spike_count is None
        else np.asarray(source_trial_spike_count, dtype=np.int64)
    )
    names = (
        tuple(f"condition-{index}" for index in range(membership.shape[1]))
        if condition_names is None
        else condition_names
    )
    return {
        "condition_names": names,
        "condition_membership": membership,
        "site_ids": site_ids,
        "site_trial_valid": valid,
        "stable_trial_rows": rows,
        "source_trial_spike_count": spike_counts,
        "frequency_count": frequency_count,
        "shared_phase_mmap_bytes": shared_phase_mmap_bytes,
    }


def _plan(config: object, **inputs: object) -> object:
    """Call the frozen pure S3 planner with only categorical/byte inputs."""
    return ppc_runtime.plan_grouped_ppc_component(config=config, **inputs)


def _job_by_identity(
    plan: object,
    *,
    condition_index: int,
    site_index: int,
    epoch_name: str,
) -> object:
    """Return one planned result cell without relying on incidental job order."""
    return next(
        job
        for job in plan.job_plans
        if job.condition_index == condition_index
        and job.site_index == site_index
        and job.epoch_name == epoch_name
    )


def test_grouped_plan_contracts_freeze_fields_and_pure_keyword_interfaces() -> None:
    """S3 exposes immutable planner records with explicit stable-identity axes.

    ``PPCJobPlan`` maps every schedule cell to its stable physical source/target
    trial rows and then to the component-wide unique edge union.  Its schedule
    itself remains the legacy local int64 derangement.  ``PPCComponentPlan`` is
    a planner-only aggregate; it contains no phase/spike samples or work paths.
    """
    assert tuple(field.name for field in fields(ppc_runtime.PPCJobPlan)) == (
        "condition_index",
        "condition_name",
        "site_index",
        "site_id",
        "epoch_index",
        "epoch_name",
        "selected_trial_rows",
        "schedule",
        "stable_edge_source_trial_row",
        "stable_edge_target_trial_row",
        "edge_union_position",
        "segment_expression",
        "base_ppc_seed",
        "schedule_seed",
        "condition_derivation_identity",
        "site_derivation_identity",
        "epoch_derivation_identity",
        "schedule_shape",
        "schedule_fingerprint",
    )
    assert tuple(field.name for field in fields(ppc_runtime.PPCAllocationEstimate)) == (
        "job_accumulator_bytes",
        "observed_trial_statistics_bytes",
        "observed_gather_temporary_bytes",
        "kernel_working_bytes",
        "geometry_bytes",
        "condition_membership_bytes",
        "source_trial_spike_count_bytes",
        "planning_working_bytes",
        "planner_array_bytes",
        "planned_planning_private_bytes",
        "summary_assembly_bytes",
        "worker_plan_bytes",
        "worker_summary_bytes",
        "checkpoint_block_bytes",
        "planned_computation_private_bytes",
        "planned_parent_private_bytes",
        "planned_worker_private_bytes",
        "shared_phase_mmap_bytes",
        "planned_aggregate_array_bytes",
        "active_worker_count",
    )
    assert tuple(field.name for field in fields(ppc_runtime.PPCComponentPlan)) == (
        "job_plans",
        "edge_site_index",
        "stable_edge_source_trial_row",
        "stable_edge_target_trial_row",
        "condition_batches",
        "scheduled_edge_count",
        "independent_edge_count",
        "union_edge_count",
        "edge_union_saturation",
        "edge_reuse_ratio",
        "allocation_estimate",
    )
    plan_signature = inspect.signature(ppc_runtime.plan_grouped_ppc_component)
    assert tuple(plan_signature.parameters) == (
        "config",
        "condition_names",
        "condition_membership",
        "site_ids",
        "site_trial_valid",
        "stable_trial_rows",
        "source_trial_spike_count",
        "frequency_count",
        "shared_phase_mmap_bytes",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in plan_signature.parameters.values()
    )
    allocation_signature = inspect.signature(
        ppc_runtime.estimate_grouped_ppc_allocation
    )
    assert tuple(allocation_signature.parameters) == (
        "active_job_count",
        "worker_result_job_count",
        "component_job_count",
        "shuffle_count",
        "total_unit_count",
        "unit_block_size",
        "source_trial_spike_count",
        "edge_source_trial_position",
        "observed_source_trial_position",
        "frequency_count",
        "representative_band_count",
        "phase_bin_count",
        "planner_array_bytes",
        "worker_plan_bytes",
        "worker_count",
        "pending_unit_block_count",
        "shared_phase_mmap_bytes",
        "condition_membership_bytes",
        "source_trial_spike_count_bytes",
        "planning_working_bytes",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in allocation_signature.parameters.values()
    )


@pytest.mark.parametrize("invalid_bytes", (-1, True, 1.5))
def test_grouped_allocation_records_full_planning_masks_and_spike_counts(
    invalid_bytes: object,
) -> None:
    """Planning bytes are explicit, checked, and separate from steady execution.

    The executor constructs one gated Boolean ``(trial, condition)`` membership
    and one int64 ``(trial, full_unit, before_after=2)`` source-count table
    before it can call the pure planner.  They are released before parent
    summary computation, so only the named planning lifetime contains them.
    """
    arguments = {
        "active_job_count": 1,
        "worker_result_job_count": 1,
        "component_job_count": 1,
        "shuffle_count": 1,
        "total_unit_count": 3,
        "unit_block_size": 1,
        "source_trial_spike_count": np.zeros((2, 1, 2), dtype=np.int64),
        "edge_source_trial_position": np.array([0], dtype=np.int64),
        "observed_source_trial_position": np.array([0], dtype=np.int64),
        "frequency_count": 1,
        "representative_band_count": 2,
        "phase_bin_count": 2,
        "planner_array_bytes": 100,
        "worker_plan_bytes": 0,
        "worker_count": 1,
        "pending_unit_block_count": 1,
        "shared_phase_mmap_bytes": 0,
        "condition_membership_bytes": 4,
        "source_trial_spike_count_bytes": 96,
        "planning_working_bytes": 0,
    }
    field_name = {
        -1: "condition_membership_bytes",
        True: "source_trial_spike_count_bytes",
        1.5: "planning_working_bytes",
    }[invalid_bytes]
    with pytest.raises(ValueError, match="planning|bytes|integer"):
        ppc_runtime.estimate_grouped_ppc_allocation(
            **(arguments | {field_name: invalid_bytes})
        )


def test_grouped_plan_accounts_full_construction_peak_and_preflights_it() -> None:
    """Planner centrally derives a conservative construction-stage peak.

    The retained table bytes are ``M = trial * condition`` and
    ``Q = 16 * trial * unit``.  Planner construction retains either final plan
    arrays ``P`` or its schedule/map scratch: ``D`` for every selected-trial
    and schedule int64 cell plus ``A`` for scalar local-position/edge lookup
    vectors and one copied ``(trial, unit_block, segment)`` int64 count block.
    ``A`` is a conservative worst-case bound, rather than an exact allocation
    trace.  A sparse selection over a larger trial axis makes ``D + A`` exceed
    ``P``. The plan derives all values from its own validated inputs; callers
    cannot underreport them.
    """
    config = _planner_config(shuffle_count=2)
    trial_count = 100
    membership = np.zeros((trial_count, 2), dtype=bool)
    membership[:2, 0] = True
    membership[2:4, 1] = True
    inputs = _planner_inputs(
        stable_trial_rows=np.arange(11, 11 + trial_count, dtype=np.int64),
        condition_names=("all", "late"),
        condition_membership=membership,
        source_trial_spike_count=np.zeros((trial_count, 1, 2), dtype=np.int64),
    )
    plan = _plan(config, **inputs)
    estimate = plan.allocation_estimate
    assert estimate.condition_membership_bytes == trial_count * 2
    assert estimate.source_trial_spike_count_bytes == trial_count * 1 * 2 * 8
    selected_row_cells = sum(job.selected_trial_rows.size for job in plan.job_plans)
    schedule_cells = sum(job.schedule.size for job in plan.job_plans)
    D = 8 * selected_row_cells + 8 * schedule_cells
    Emax = trial_count * (trial_count - 1)
    A = 8 * (2 * trial_count + Emax) + 16 * trial_count * 1
    assert estimate.planning_working_bytes == D + A
    assert estimate.planning_working_bytes > estimate.planner_array_bytes
    assert estimate.planned_planning_private_bytes == (
        estimate.condition_membership_bytes
        + estimate.source_trial_spike_count_bytes
        + max(estimate.planner_array_bytes, estimate.planning_working_bytes)
    )
    # Counts/membership and their construction peak are released before the
    # serial steady stage, but must still gate the parent lifetime.
    steady_parent = (
        estimate.planner_array_bytes + estimate.summary_assembly_bytes
        + max(
            estimate.planned_computation_private_bytes,
            estimate.checkpoint_block_bytes,
        )
    )
    assert estimate.planned_parent_private_bytes == max(
        estimate.planned_planning_private_bytes,
        steady_parent,
    )
    construction_limit = (
        estimate.condition_membership_bytes
        + estimate.source_trial_spike_count_bytes
        + estimate.planning_working_bytes
        - 1
    )
    assert (
        estimate.condition_membership_bytes
        + estimate.source_trial_spike_count_bytes
        + estimate.planner_array_bytes
        <= construction_limit
        < estimate.condition_membership_bytes
        + estimate.source_trial_spike_count_bytes
        + estimate.planning_working_bytes
    )

    def forbidden_map(*_: object, **__: object) -> None:
        """Construction-stage preflight must precede final edge-map materialization."""
        raise AssertionError("unsafe plan materialized final edge maps")

    unsafe_config = replace(
        config,
        ppc_execution=replace(
            config.ppc_execution,
            maximum_worker_allocation_bytes=construction_limit,
        ),
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ppc_runtime.np, "broadcast_to", forbidden_map)
        with pytest.raises(ValueError, match="planning|parent|allocation"):
            _plan(unsafe_config, **inputs)


def test_plan_translates_condition_local_schedule_positions_to_stable_trial_rows() -> None:
    """Local derangement positions cannot leak into a cross-condition edge key."""
    config = _planner_config(shuffle_count=2, seed=29)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([17, 53, 89, 149], dtype=np.int64),
        condition_names=("nested",),
        condition_membership=np.array([[False], [True], [True], [True]], dtype=bool),
        source_trial_spike_count=np.zeros((4, 1, 2), dtype=np.int64),
    )

    plan = _plan(config, **inputs)
    job = _job_by_identity(plan, condition_index=0, site_index=0, epoch_name="before")
    expected_source = np.broadcast_to(
        job.selected_trial_rows,
        job.schedule.shape,
    )
    expected_target = job.selected_trial_rows[job.schedule]

    np.testing.assert_array_equal(job.selected_trial_rows, np.array([53, 89, 149], dtype=np.int64))
    np.testing.assert_array_equal(job.stable_edge_source_trial_row, expected_source)
    np.testing.assert_array_equal(job.stable_edge_target_trial_row, expected_target)
    assert set(job.stable_edge_source_trial_row.ravel()) <= {53, 89, 149}
    assert set(job.stable_edge_target_trial_row.ravel()) <= {53, 89, 149}
    assert not np.any(job.stable_edge_source_trial_row == job.stable_edge_target_trial_row)
    np.testing.assert_array_equal(
        np.column_stack((
            plan.stable_edge_source_trial_row[job.edge_union_position.ravel()],
            plan.stable_edge_target_trial_row[job.edge_union_position.ravel()],
        )),
        np.column_stack((expected_source.ravel(), expected_target.ravel())),
    )


def test_edge_union_uses_nested_and_overlapping_condition_pools_without_widening() -> None:
    """The stable union deduplicates physical edges but never admits cross-pool pairs."""
    config = _planner_config(shuffle_count=3, seed=13)
    membership = np.array(
        [
            [True, False, False],
            [True, True, False],
            [True, True, True],
            [False, False, True],
        ],
        dtype=bool,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([101, 303, 709, 911], dtype=np.int64),
        condition_names=("outer", "nested", "overlap"),
        condition_membership=membership,
        source_trial_spike_count=np.zeros((4, 1, 2), dtype=np.int64),
    )

    plan = _plan(config, **inputs)
    all_job_edges: list[tuple[int, int, int]] = []
    allowed_rows = {
        0: {101, 303, 709},
        1: {303, 709},
        2: {709, 911},
    }
    for job in plan.job_plans:
        source = job.stable_edge_source_trial_row.ravel()
        target = job.stable_edge_target_trial_row.ravel()
        assert set(source) <= allowed_rows[job.condition_index]
        assert set(target) <= allowed_rows[job.condition_index]
        all_job_edges.extend(
            (job.site_index, int(left), int(right))
            for left, right in zip(source, target, strict=True)
        )

    union_edges = list(
        zip(
            plan.edge_site_index.tolist(),
            plan.stable_edge_source_trial_row.tolist(),
            plan.stable_edge_target_trial_row.tolist(),
            strict=True,
        )
    )
    assert union_edges == sorted(set(all_job_edges))
    assert plan.union_edge_count == len(union_edges)
    assert plan.union_edge_count < plan.independent_edge_count
    assert (0, 101, 911) not in union_edges
    assert (0, 911, 101) not in union_edges


def test_plan_preserves_distinct_epoch_schedule_seeds_segments_and_schedule_identity() -> None:
    """Whole/before/after retain their exact legacy derived seeds and schedules."""
    config = _planner_config(shuffle_count=4, seed=31)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([11, 23, 47], dtype=np.int64),
        condition_names=("left", "right"),
        condition_membership=np.array([[True, False], [True, True], [True, True]], dtype=bool),
        site_ids=("site-a", "site-b"),
        site_trial_valid=np.ones((2, 3), dtype=bool),
        source_trial_spike_count=np.zeros((3, 1, 2), dtype=np.int64),
    )

    plan = _plan(config, **inputs)
    expected_expression = {
        "whole": "before + after",
        "before": "before",
        "after": "after",
    }
    expected_seed = {}
    for epoch_index, epoch_name in enumerate(config.ppc.epochs):
        seed = lfp_summary_runtime._ppc_schedule_seed(config, 1, 1, epoch_index)
        expected_seed[epoch_name] = seed
        job = _job_by_identity(
            plan,
            condition_index=1,
            site_index=1,
            epoch_name=epoch_name,
        )
        expected_schedule = spike_lfp_summary.generate_trial_derangement_schedule(
            2,
            config.ppc.shuffle_count,
            seed=seed,
        )
        assert job.base_ppc_seed == config.ppc.seed
        assert job.schedule_seed == seed
        assert job.condition_derivation_identity == 1
        assert job.site_derivation_identity == 1
        assert job.epoch_derivation_identity == epoch_index
        assert job.segment_expression == expected_expression[epoch_name]
        assert job.schedule_shape == expected_schedule.shape
        assert job.schedule_fingerprint == ppc_runtime._array_fingerprint(expected_schedule)
        np.testing.assert_array_equal(job.schedule, expected_schedule)

    assert len(set(expected_seed.values())) == 3


def test_plan_reuses_generator_owned_schedule_buffers_in_site_major_job_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner jobs seal, but do not copy, their private generator schedules."""
    config = _planner_config(shuffle_count=2, seed=41)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([11, 29], dtype=np.int64),
        condition_names=("all",),
        condition_membership=np.ones((2, 1), dtype=bool),
        site_ids=("site-a", "site-b"),
        site_trial_valid=np.ones((2, 2), dtype=bool),
        source_trial_spike_count=np.zeros((2, 1, 2), dtype=np.int64),
        frequency_count=1,
    )
    captured: dict[int, np.ndarray] = {}

    def owned_schedule(
        trial_count: int,
        shuffle_count: int,
        *,
        seed: int,
    ) -> np.ndarray:
        """Return one distinct owned int64 derangement buffer for each job call."""
        assert trial_count == 2
        schedule = np.tile(
            np.array([1, 0], dtype=np.int64), (shuffle_count, 1)
        )
        captured[seed] = schedule
        return schedule

    monkeypatch.setattr(
        ppc_runtime, "generate_trial_derangement_schedule", owned_schedule
    )
    plan = _plan(config, **inputs)
    jobs = tuple(job for job in plan.job_plans if job.selected_trial_rows.size == 2)

    assert len(captured) == len(jobs)
    assert {job.schedule_seed for job in jobs} == set(captured)
    for job in jobs:
        generated = captured[job.schedule_seed]
        assert np.shares_memory(job.schedule, generated)
        assert not job.schedule.flags.writeable
        assert not generated.flags.writeable


def test_plan_order_is_deterministic_and_planner_does_not_execute_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planning is a side-effect-free ordered description, not an executor seam."""
    config = _planner_config(shuffle_count=2, worker_count=2)
    inputs = _planner_inputs(
        condition_names=("first", "second"),
        condition_membership=np.ones((3, 2), dtype=bool),
        site_ids=("site-a", "site-b"),
        site_trial_valid=np.ones((2, 3), dtype=bool),
        source_trial_spike_count=np.zeros((3, 3, 2), dtype=np.int64),
    )

    def forbidden(*_: object, **__: object) -> None:
        """Fail if a planner attempts computation, checkpointing, or process creation."""
        raise AssertionError("S3 planning must not execute PPC work")

    monkeypatch.setattr(ppc_runtime, "execute_ppc_blocks", forbidden)
    monkeypatch.setattr(ppc_runtime, "_compute_unit_blocks", forbidden)
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", forbidden)
    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden)
    first = _plan(config, **inputs)
    second = _plan(config, **inputs)

    expected_order = [
        (condition_index, site_index, epoch_index)
        for site_index in range(2)
        for condition_index in range(2)
        for epoch_index in range(3)
    ]
    assert [
        (job.condition_index, job.site_index, job.epoch_index)
        for job in first.job_plans
    ] == expected_order
    assert [
        (job.condition_index, job.site_index, job.epoch_index)
        for job in second.job_plans
    ] == expected_order
    assert first.condition_batches == second.condition_batches
    np.testing.assert_array_equal(
        first.stable_edge_source_trial_row,
        second.stable_edge_source_trial_row,
    )
    np.testing.assert_array_equal(
        first.stable_edge_target_trial_row,
        second.stable_edge_target_trial_row,
    )


def test_plan_empty_and_single_trial_conditions_do_not_request_derangements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed-only empty/singleton pools keep empty schedule axes and no invalid shuffle."""
    config = _planner_config(shuffle_count=2)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([5, 8, 13], dtype=np.int64),
        condition_names=("empty", "single"),
        condition_membership=np.array([[False, False], [False, True], [False, False]], dtype=bool),
        source_trial_spike_count=np.zeros((3, 1, 2), dtype=np.int64),
    )

    def forbidden_schedule(*_: object, **__: object) -> np.ndarray:
        """Fail if a condition with fewer than two trials requests a derangement."""
        raise AssertionError("empty/single-trial plan requested a derangement")

    monkeypatch.setattr(
        ppc_runtime,
        "generate_trial_derangement_schedule",
        forbidden_schedule,
    )
    plan = _plan(config, **inputs)

    for epoch_name in config.ppc.epochs:
        empty = _job_by_identity(plan, condition_index=0, site_index=0, epoch_name=epoch_name)
        single = _job_by_identity(plan, condition_index=1, site_index=0, epoch_name=epoch_name)
        assert empty.schedule.shape == (0, 0)
        assert single.schedule.shape == (0, 1)
        assert empty.stable_edge_source_trial_row.shape == (0, 0)
        assert single.stable_edge_target_trial_row.shape == (0, 1)
        assert empty.edge_union_position.shape == (0, 0)
        assert single.edge_union_position.shape == (0, 1)

    # The site/unit result retains every final condition/epoch cell even when
    # none has a legal null schedule. Only the batch-local null state is zero.
    assert plan.allocation_estimate.job_accumulator_bytes == 0
    assert plan.allocation_estimate.kernel_working_bytes == 0
    assert plan.allocation_estimate.worker_summary_bytes == 6 * (116 * 2 + 2 * 2 * 8)
    assert plan.scheduled_edge_count == 0
    assert plan.independent_edge_count == 0
    assert plan.union_edge_count == 0
    assert plan.edge_union_saturation == 0.0
    assert plan.edge_reuse_ratio == 0.0


@pytest.mark.parametrize("shuffle_count", (100, 1000))
def test_edge_union_counts_saturation_and_reuse_match_two_trial_hand_example(
    shuffle_count: int,
) -> None:
    """Repeated schedules saturate the two directed physical edges without duplication.

    Three epoch jobs independently need both directions, but the grouped union
    samples each stable pair once.  Saturation is the fraction of allowed
    condition-local physical pairs reached by the schedule union; reuse is the
    independent unique-edge demand divided by that shared union.
    """
    config = _planner_config(shuffle_count=shuffle_count, seed=5)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([41, 97], dtype=np.int64),
        source_trial_spike_count=np.zeros((2, 1, 2), dtype=np.int64),
        frequency_count=1,
    )

    plan = _plan(config, **inputs)

    assert plan.scheduled_edge_count == 3 * shuffle_count * 2
    assert plan.independent_edge_count == 3 * 2
    assert plan.union_edge_count == 2
    assert plan.edge_union_saturation == 1.0
    assert plan.edge_reuse_ratio == 3.0
    np.testing.assert_array_equal(plan.edge_site_index, np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(plan.stable_edge_source_trial_row, np.array([41, 97], dtype=np.int64))
    np.testing.assert_array_equal(plan.stable_edge_target_trial_row, np.array([97, 41], dtype=np.int64))


def test_edge_union_three_trial_incomplete_schedule_has_hand_saturation_and_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-row three-trial schedule need not fill all six directed pairs.

    With base seed one, each unchanged epoch derivation produces the same
    derangement ``[2, 0, 1]``. The three epoch jobs therefore independently use
    nine edge cells, while their stable union contains only three of the six
    allowed directed physical pairs.
    """
    config = _planner_config(shuffle_count=1, seed=1)

    def fixed_three_trial_schedule(
        trial_count: int,
        shuffle_count: int,
        *,
        seed: int,
    ) -> np.ndarray:
        """Keep this hand-count fixture independent of NumPy RNG implementation."""
        assert trial_count == 3
        assert shuffle_count == 1
        assert isinstance(seed, int)
        return np.array([[2, 0, 1]], dtype=np.int64)

    monkeypatch.setattr(
        ppc_runtime,
        "generate_trial_derangement_schedule",
        fixed_three_trial_schedule,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([10, 20, 30], dtype=np.int64),
        source_trial_spike_count=np.zeros((3, 1, 2), dtype=np.int64),
        frequency_count=1,
    )

    plan = _plan(config, **inputs)

    assert plan.scheduled_edge_count == 9
    assert plan.independent_edge_count == 9
    assert plan.union_edge_count == 3
    assert plan.edge_union_saturation == 0.5
    assert plan.edge_reuse_ratio == 3.0
    np.testing.assert_array_equal(
        plan.stable_edge_source_trial_row,
        np.array([10, 20, 30], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        plan.stable_edge_target_trial_row,
        np.array([30, 10, 20], dtype=np.int64),
    )


def test_edge_union_keeps_identical_trial_pairs_separate_for_each_site() -> None:
    """A physical pair is reusable across conditions but never across sites."""
    config = _planner_config(shuffle_count=1, seed=3)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([41, 97], dtype=np.int64),
        site_ids=("site-a", "site-b"),
        site_trial_valid=np.ones((2, 2), dtype=bool),
        source_trial_spike_count=np.zeros((2, 1, 2), dtype=np.int64),
        frequency_count=1,
    )

    plan = _plan(config, **inputs)

    assert plan.condition_batches == (((0,),), ((0,),))
    assert plan.union_edge_count == 4
    assert plan.independent_edge_count == 12
    assert plan.edge_reuse_ratio == 3.0
    np.testing.assert_array_equal(plan.edge_site_index, np.array([0, 0, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(
        plan.stable_edge_source_trial_row,
        np.array([41, 97, 41, 97], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        plan.stable_edge_target_trial_row,
        np.array([97, 41, 97, 41], dtype=np.int64),
    )
    for job in plan.job_plans:
        np.testing.assert_array_equal(
            plan.edge_site_index[job.edge_union_position.ravel()],
            np.full(job.edge_union_position.size, job.site_index, dtype=np.int64),
        )


def test_plan_selects_condition_and_site_valid_trial_intersection_per_site() -> None:
    """Each site's plan uses exactly its own valid rows within a condition pool."""
    config = _planner_config(shuffle_count=1, seed=7)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([11, 29, 47, 71], dtype=np.int64),
        condition_names=("selected",),
        condition_membership=np.array([[True], [True], [False], [True]], dtype=bool),
        site_ids=("site-a", "site-b"),
        site_trial_valid=np.array(
            [[True, False, True, True], [False, True, True, True]],
            dtype=bool,
        ),
        source_trial_spike_count=np.zeros((4, 1, 2), dtype=np.int64),
        frequency_count=1,
    )

    plan = _plan(config, **inputs)

    for epoch_name in config.ppc.epochs:
        first_site = _job_by_identity(
            plan, condition_index=0, site_index=0, epoch_name=epoch_name
        )
        second_site = _job_by_identity(
            plan, condition_index=0, site_index=1, epoch_name=epoch_name
        )
        np.testing.assert_array_equal(
            first_site.selected_trial_rows,
            np.array([11, 71], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            second_site.selected_trial_rows,
            np.array([29, 71], dtype=np.int64),
        )


def test_plan_batches_conditions_deterministically_under_worker_allocation_limit() -> None:
    """Batching reduces only null accumulators, not retained site-level state."""
    config = _planner_config(
        shuffle_count=25,
        unit_block_size=10,
        shuffle_block_size=25,
        worker_count=2,
        maximum_worker_allocation_bytes=60_000,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([2, 7], dtype=np.int64),
        condition_names=("a", "b", "c"),
        condition_membership=np.ones((2, 3), dtype=bool),
        source_trial_spike_count=np.zeros((2, 10, 2), dtype=np.int64),
        frequency_count=1,
    )

    plan = _plan(config, **inputs)

    assert plan.condition_batches == (((0,), (1,), (2,)),)
    assert [job.condition_index for job in plan.job_plans] == [
        0, 0, 0, 1, 1, 1, 2, 2, 2,
    ]
    assert plan.allocation_estimate.worker_plan_bytes == 14_592
    assert plan.allocation_estimate.worker_summary_bytes == 13_320
    assert plan.allocation_estimate.planned_worker_private_bytes == 59_256
    assert plan.allocation_estimate.planned_parent_private_bytes <= 60_000
    assert plan.allocation_estimate.planned_worker_private_bytes <= 60_000


def test_plan_batches_conditions_to_satisfy_aggregate_allocation_limit() -> None:
    """Aggregate preflight splits an otherwise valid all-condition worker batch."""
    config = _planner_config(
        shuffle_count=25,
        unit_block_size=10,
        shuffle_block_size=25,
        worker_count=2,
        maximum_worker_allocation_bytes=150_000,
        maximum_aggregate_allocation_bytes=105_000,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([2, 7], dtype=np.int64),
        condition_names=("a", "b", "c"),
        condition_membership=np.ones((2, 3), dtype=bool),
        source_trial_spike_count=np.zeros((2, 10, 2), dtype=np.int64),
        frequency_count=1,
    )

    plan = _plan(config, **inputs)

    assert plan.condition_batches == (((0,), (1,), (2,)),)
    assert plan.allocation_estimate.planned_worker_private_bytes == 59_256
    assert plan.allocation_estimate.checkpoint_block_bytes == 13_344
    assert plan.allocation_estimate.planned_aggregate_array_bytes == 104_608


def test_plan_rejects_aggregate_limit_when_one_condition_batch_cannot_fit() -> None:
    """Deterministic batching cannot rescue an aggregate-unsafe singleton batch."""
    config = _planner_config(
        shuffle_count=25,
        unit_block_size=10,
        shuffle_block_size=25,
        worker_count=2,
        maximum_worker_allocation_bytes=150_000,
        maximum_aggregate_allocation_bytes=90_000,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([2, 7], dtype=np.int64),
        condition_names=("a", "b", "c"),
        condition_membership=np.ones((2, 3), dtype=bool),
        source_trial_spike_count=np.zeros((2, 10, 2), dtype=np.int64),
        frequency_count=1,
    )

    with pytest.raises(ValueError, match="aggregate"):
        _plan(config, **inputs)


def test_plan_allocation_uses_largest_bounded_unit_block_and_active_workers() -> None:
    """Component planning charges the largest unit block, never all units at once."""
    config = _planner_config(
        shuffle_count=1,
        unit_block_size=2,
        shuffle_block_size=1,
        worker_count=3,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([2, 7], dtype=np.int64),
        source_trial_spike_count=np.zeros((2, 3, 2), dtype=np.int64),
        frequency_count=1,
        shared_phase_mmap_bytes=4096,
    )

    plan = _plan(config, **inputs)

    assert plan.allocation_estimate.job_accumulator_bytes == 3 * 1 * 2 * 1 * 40
    assert plan.allocation_estimate.geometry_bytes == 2 * (2 * 2 + 1) * 8 + 2 * 8
    assert plan.allocation_estimate.kernel_working_bytes == 2 * 2 * 1 * 48 + 2 * 16
    assert plan.allocation_estimate.observed_trial_statistics_bytes == (
        2 * 8
        + 2 * 2 * 2 * 1 * (16 + 8)
        + 2 * 8
        + 2 * 8
        + (2 + 1) * 8
        + 2 * 2 * 2 * 2 * 2 * 8
    )
    assert plan.allocation_estimate.planner_array_bytes == 6 * 8 + 6 * 32 + 2 * 3 * 8
    assert plan.allocation_estimate.summary_assembly_bytes == 3 * 3 * (116 + 2 * 2 * 8)
    assert plan.allocation_estimate.worker_plan_bytes == 6 * 8 + 6 * 32 + 2 * 3 * 8
    assert plan.allocation_estimate.worker_summary_bytes == 2 * 3 * (116 + 2 * 2 * 8)
    assert plan.allocation_estimate.checkpoint_block_bytes == 2 * 3 * (116 + 2 * 2 * 8) + 3 * 8
    assert plan.allocation_estimate.planned_computation_private_bytes == 616
    assert plan.allocation_estimate.planned_parent_private_bytes == 288 + 1332 + 912
    assert plan.allocation_estimate.planned_worker_private_bytes == 288 + 888 + 616
    assert plan.allocation_estimate.active_worker_count == 2
    assert plan.allocation_estimate.planned_aggregate_array_bytes == 4096 + 2532 + 2 * 1792


def test_plan_allocation_uses_the_true_later_site_unit_and_edge_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Planner allocation is the true maximum, not the first site/block estimate.

    Site B has the complete three-trial pool while site A has only the first
    two trials.  The partial final unit block contains nearly all spikes, and
    the last two-edge block has two distinct targets from that high-spike
    source.  This makes the final site/unit/edge task unambiguously largest.
    """
    config = _planner_config(
        shuffle_count=1,
        unit_block_size=2,
        shuffle_block_size=1,
        trial_edge_block_size=2,
        worker_count=2,
    )

    def fixed_schedule(
        trial_count: int,
        shuffle_count: int,
        *,
        seed: int,
    ) -> np.ndarray:
        """Make condition-specific unions independent of RNG-version details."""
        assert shuffle_count == 1
        if trial_count == 2:
            return np.array([[1, 0]], dtype=np.int64)
        assert trial_count == 3
        if seed // 1_000_000 == 0:
            return np.array([[1, 2, 0]], dtype=np.int64)
        return np.array([[2, 0, 1]], dtype=np.int64)

    monkeypatch.setattr(
        ppc_runtime,
        "generate_trial_derangement_schedule",
        fixed_schedule,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([10, 20, 30], dtype=np.int64),
        condition_names=("first", "second"),
        condition_membership=np.ones((3, 2), dtype=bool),
        site_ids=("site-a", "site-b"),
        site_trial_valid=np.array(
            [[True, True, False], [True, True, True]],
            dtype=bool,
        ),
        source_trial_spike_count=np.array(
            [
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [1, 1]],
                [[1, 1], [1, 1], [1000, 1000]],
            ],
            dtype=np.int64,
        ),
        frequency_count=1,
        shared_phase_mmap_bytes=0,
    )

    plan = _plan(config, **inputs)
    estimate = plan.allocation_estimate

    # Full-component parent arrays retain two sites, while the worker holds
    # only the maximum site plan and current partial unit block.
    assert estimate.planner_array_bytes == 1_392
    assert estimate.summary_assembly_bytes == 5_328
    assert estimate.worker_plan_bytes == 864
    assert estimate.worker_summary_bytes == 888
    assert estimate.checkpoint_block_bytes == 912
    assert estimate.job_accumulator_bytes == 240

    # The selected peak is site B's final one-unit block. Its last edge block
    # has two site-qualified high-spike source edges, unlike site A or block 0.
    assert estimate.geometry_bytes == 52_200
    assert estimate.observed_trial_statistics_bytes == 416
    assert estimate.observed_gather_temporary_bytes == 102_204
    assert estimate.kernel_working_bytes == 204_128
    assert estimate.planned_computation_private_bytes == 256_568
    assert estimate.planned_parent_private_bytes == 7_632
    assert estimate.planned_worker_private_bytes == 258_320
    assert estimate.active_worker_count == 2
    assert estimate.planned_aggregate_array_bytes == 524_272


def test_grouped_allocation_estimate_matches_every_named_category_and_lifetime_peak() -> None:
    """Byte accounting uses kernel arrays, retained observed arrays, and 40-byte job cells.

    The job cell is one complex128 sum, one int64 count, one float64 PPC draw,
    and one float64 calculation scratch. The observed stage retains geometry, its per-trial statistics,
    and gather scratch; the later null stage retains geometry, job accumulators,
    and complete kernel working arrays.  Those stages do not overlap, so each
    worker reports their maximum rather than their sum. The parallel parent
    concurrently retains component-wide planner and final-summary arrays; each
    worker retains its complete site-plan copy and site/block summaries across
    condition batches. The shared mmap is counted exactly once.
    """
    source_counts = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int64,
    )
    edge_source_position = np.array([0, 1, 1], dtype=np.int64)

    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=3,
        worker_result_job_count=6,
        component_job_count=6,
        shuffle_count=5,
        total_unit_count=4,
        unit_block_size=2,
        source_trial_spike_count=source_counts,
        edge_source_trial_position=edge_source_position,
        observed_source_trial_position=np.array([0, 1], dtype=np.int64),
        frequency_count=2,
        representative_band_count=2,
        phase_bin_count=4,
        planner_array_bytes=480,
        worker_plan_bytes=480,
        worker_count=4,
        pending_unit_block_count=2,
        shared_phase_mmap_bytes=1000,
    )

    geometry_bytes = 26 * 36 + 2 * (2 * 2 + 1) * 8 + 2 * 8
    segmented_bytes = 3 * 2 * 2 * 48 + 3 * 2 * 8
    gather_bytes = (10 + 26 + 26) * 2 * 51
    job_bytes = 3 * 5 * 2 * 2 * 40
    observed_bytes = (
        2 * 8
        + 2 * 2 * 2 * 2 * (16 + 8)
        + 2 * 8
        + 2 * 8
        + (4 + 1) * 8
        + 2 * 2 * 2 * 2 * 4 * 8
    )
    observed_gather_bytes = (10 + 26) * 2 * 51
    kernel_bytes = segmented_bytes + gather_bytes
    observed_stage_bytes = geometry_bytes + observed_bytes + observed_gather_bytes
    null_stage_bytes = geometry_bytes + job_bytes + kernel_bytes
    component_summary_bytes = 4 * 6 * (116 * 2 + 2 * 4 * 8)
    worker_summary_bytes = 2 * 6 * (116 * 2 + 2 * 4 * 8)
    checkpoint_block_bytes = worker_summary_bytes + 3 * 8
    computation_bytes = max(observed_stage_bytes, null_stage_bytes)
    worker_bytes = 480 + worker_summary_bytes + computation_bytes
    parent_bytes = 480 + component_summary_bytes + checkpoint_block_bytes

    assert estimate.job_accumulator_bytes == job_bytes
    assert estimate.observed_trial_statistics_bytes == observed_bytes
    assert estimate.observed_gather_temporary_bytes == observed_gather_bytes
    assert estimate.kernel_working_bytes == kernel_bytes
    assert estimate.geometry_bytes == geometry_bytes
    assert estimate.planner_array_bytes == 480
    assert estimate.summary_assembly_bytes == component_summary_bytes
    assert estimate.worker_plan_bytes == 480
    assert estimate.worker_summary_bytes == worker_summary_bytes
    assert estimate.checkpoint_block_bytes == checkpoint_block_bytes
    assert estimate.planned_computation_private_bytes == computation_bytes
    assert estimate.planned_parent_private_bytes == parent_bytes
    assert estimate.planned_worker_private_bytes == worker_bytes
    assert estimate.planned_computation_private_bytes == null_stage_bytes
    assert estimate.planned_computation_private_bytes < (
        geometry_bytes + job_bytes + observed_bytes + kernel_bytes
    )
    assert estimate.shared_phase_mmap_bytes == 1000
    assert estimate.active_worker_count == 2
    assert estimate.planned_aggregate_array_bytes == 1000 + parent_bytes + 2 * worker_bytes
    assert all(
        isinstance(getattr(estimate, field.name), int) and getattr(estimate, field.name) >= 0
        for field in fields(estimate)
    )
    with pytest.raises(FrozenInstanceError):
        estimate.geometry_bytes = 0


def test_allocation_estimate_uses_serial_parent_peak_and_checked_integer_arithmetic() -> None:
    """A serial plan charges no worker and rejects overflow instead of wrapping bytes."""
    serial = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=1,
        worker_result_job_count=3,
        component_job_count=3,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((2, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.array([0, 1], dtype=np.int64),
        observed_source_trial_position=np.array([0, 1], dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=100,
        worker_plan_bytes=0,
        worker_count=1,
        pending_unit_block_count=2,
        shared_phase_mmap_bytes=64,
    )
    assert serial.active_worker_count == 0
    assert serial.summary_assembly_bytes == 3 * (116 + 2 * 2 * 8)
    assert serial.checkpoint_block_bytes == 3 * (116 + 2 * 2 * 8) + 3 * 8
    assert serial.planned_computation_private_bytes == 360
    assert serial.planned_parent_private_bytes == 100 + 3 * (116 + 2 * 2 * 8) + 468
    assert serial.planned_aggregate_array_bytes == 64 + serial.planned_parent_private_bytes
    with pytest.raises(ValueError):
        ppc_runtime.estimate_grouped_ppc_allocation(
            active_job_count=np.iinfo(np.int64).max,
            worker_result_job_count=np.iinfo(np.int64).max,
            component_job_count=np.iinfo(np.int64).max,
            shuffle_count=2,
            total_unit_count=1,
            unit_block_size=1,
            source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            observed_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=1,
            representative_band_count=2,
            phase_bin_count=2,
            planner_array_bytes=0,
            worker_plan_bytes=0,
            worker_count=1,
            pending_unit_block_count=1,
            shared_phase_mmap_bytes=0,
        )


def test_allocation_estimate_uses_compute_dominant_serial_parent_peak() -> None:
    """Serial parent lifetime retains the larger compute or checkpoint stage.

    A large observed source is intentionally more expensive than retaining one
    bounded serialized block.  The parent must therefore charge computation,
    not add the two non-overlapping stages together or always prefer the
    checkpoint block.
    """
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=1,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.array([[[1_000, 1_000]]], dtype=np.int64),
        edge_source_trial_position=np.array([0], dtype=np.int64),
        observed_source_trial_position=np.array([0], dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=100,
        worker_plan_bytes=0,
        worker_count=1,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )

    assert estimate.planned_computation_private_bytes > estimate.checkpoint_block_bytes
    assert estimate.planned_parent_private_bytes == (
        estimate.planner_array_bytes
        + estimate.summary_assembly_bytes
        + estimate.planned_computation_private_bytes
    )


def test_allocation_estimate_checks_checkpoint_identity_addition_overflow() -> None:
    """A fitting summary still rejects an overflowing checkpoint identity suffix.

    The component summary is exactly ``(int64_max - 7) / 2`` bytes.  Adding
    its bounded checkpoint copy plus the three int64 block identity cells
    overflows the serial parent lifetime only because of that compact suffix.
    """
    int64_max = np.iinfo(np.int64).max
    summary_bytes = (int64_max - 7) // 2
    frequency_count = (summary_bytes // 4 - 2 * 2 * 11) // 29
    assert summary_bytes == 116 * frequency_count + 8 * 2 * 11

    with pytest.raises(ValueError, match="overflow|bytes"):
        ppc_runtime.estimate_grouped_ppc_allocation(
            active_job_count=0,
            worker_result_job_count=1,
            component_job_count=1,
            shuffle_count=1,
            total_unit_count=1,
            unit_block_size=1,
            source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
            edge_source_trial_position=np.empty(0, dtype=np.int64),
            observed_source_trial_position=np.empty(0, dtype=np.int64),
            frequency_count=frequency_count,
            representative_band_count=2,
            phase_bin_count=11,
            planner_array_bytes=0,
            worker_plan_bytes=0,
            worker_count=1,
            pending_unit_block_count=1,
            shared_phase_mmap_bytes=0,
        )


def test_allocation_estimate_counts_checkpoint_when_inactive_serial_compute_is_smaller() -> None:
    """Serial parent publication includes one checkpoint copy even without null work."""
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=0,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.empty(0, dtype=np.int64),
        observed_source_trial_position=np.empty(0, dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=100,
        worker_plan_bytes=0,
        worker_count=1,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )

    summary_bytes = 116 + 2 * 2 * 8
    assert estimate.summary_assembly_bytes == summary_bytes
    assert estimate.worker_summary_bytes == summary_bytes
    assert estimate.checkpoint_block_bytes == summary_bytes + 3 * 8
    assert estimate.planned_computation_private_bytes < estimate.checkpoint_block_bytes
    assert estimate.planned_parent_private_bytes == 100 + summary_bytes + estimate.checkpoint_block_bytes
    assert estimate.planned_aggregate_array_bytes == estimate.planned_parent_private_bytes


def test_allocation_estimate_uses_observed_stage_peak_without_summing_null_stage() -> None:
    """A histogram-heavy observed stage can be the private peak by itself.

    This guards against the overly conservative and incorrect sum of observed
    per-trial arrays with null-stage job accumulators after observed statistics
    have been reduced and released.
    """
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=1,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.array([0], dtype=np.int64),
        observed_source_trial_position=np.array([0], dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=10,
        planner_array_bytes=100,
        worker_plan_bytes=0,
        worker_count=2,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )

    geometry_bytes = 3 * 8 + 8
    kernel_bytes = 48 + 2 * 8
    job_bytes = 40
    observed_bytes = 8 + 2 * (16 + 8) + 2 * 8 + 2 * 8 + 11 * 8 + 2 * 2 * 10 * 8
    observed_stage_bytes = geometry_bytes + observed_bytes
    null_stage_bytes = geometry_bytes + job_bytes + kernel_bytes

    assert estimate.observed_trial_statistics_bytes == observed_bytes
    assert observed_stage_bytes > null_stage_bytes
    assert estimate.planned_computation_private_bytes == observed_stage_bytes
    assert estimate.planned_computation_private_bytes < observed_stage_bytes + null_stage_bytes
    assert estimate.summary_assembly_bytes > job_bytes
    assert estimate.worker_summary_bytes == 276
    assert estimate.checkpoint_block_bytes == 300
    assert estimate.planned_parent_private_bytes == 100 + 276 + 300
    assert estimate.planned_worker_private_bytes == 276 + observed_stage_bytes
    assert estimate.active_worker_count == 1
    assert estimate.planned_aggregate_array_bytes == 100 + 276 + 300 + 276 + observed_stage_bytes

    planning_dominant = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=1,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((100, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.array([0], dtype=np.int64),
        observed_source_trial_position=np.array([0], dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=10,
        planner_array_bytes=1000,
        worker_plan_bytes=0,
        worker_count=2,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
        condition_membership_bytes=100,
        source_trial_spike_count_bytes=1_600,
        planning_working_bytes=5_000,
    )
    planning_bytes = 100 + 1_600 + max(1000, 5_000)
    steady_parent_bytes = (
        planning_dominant.planner_array_bytes
        + planning_dominant.summary_assembly_bytes
        + planning_dominant.checkpoint_block_bytes
    )
    steady_parallel_bytes = (
        steady_parent_bytes
        + planning_dominant.active_worker_count
        * planning_dominant.planned_worker_private_bytes
    )
    assert planning_dominant.planning_working_bytes == 5_000
    assert planning_dominant.planned_planning_private_bytes == planning_bytes
    assert planning_dominant.planned_parent_private_bytes == max(
        planning_bytes,
        steady_parent_bytes,
    )
    assert planning_dominant.planned_aggregate_array_bytes == max(
        planning_bytes,
        steady_parallel_bytes,
    )


@pytest.mark.parametrize("worker_count", (1, 2))
def test_allocation_estimate_uses_planning_peak_before_or_alongside_steady_execution(
    worker_count: int,
) -> None:
    """Planning tables overlap neither summary compute nor parallel workers.

    The planning lifetime contains gated membership ``M``, full source counts
    ``Q``, and the larger of retained plan arrays ``P`` or construction scratch
    ``D + A``.  It is an alternative peak: the parent chooses its larger
    planning or steady stage, while aggregate memory chooses planning alone or
    steady parent plus actually active workers.
    """
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=1,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((100, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.array([0], dtype=np.int64),
        observed_source_trial_position=np.array([0], dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=100,
        worker_plan_bytes=20,
        worker_count=worker_count,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=17,
        condition_membership_bytes=100,
        source_trial_spike_count_bytes=1_600,
        planning_working_bytes=5_000,
    )
    planning_bytes = 100 + 1_600 + max(100, 5_000)
    steady_parent_bytes = estimate.planner_array_bytes + estimate.summary_assembly_bytes
    if estimate.active_worker_count == 0:
        steady_parent_bytes += max(
            estimate.planned_computation_private_bytes,
            estimate.checkpoint_block_bytes,
        )
    else:
        steady_parent_bytes += estimate.checkpoint_block_bytes
    steady_aggregate_bytes = (
        steady_parent_bytes
        + estimate.active_worker_count * estimate.planned_worker_private_bytes
    )
    assert estimate.planning_working_bytes == 5_000
    assert estimate.planned_planning_private_bytes == planning_bytes
    assert estimate.planned_parent_private_bytes == max(
        planning_bytes,
        steady_parent_bytes,
    )
    assert estimate.planned_aggregate_array_bytes == 17 + max(
        planning_bytes,
        steady_aggregate_bytes,
    )


@pytest.mark.parametrize(
    "description, mutate",
    (
        (
            "non-Boolean condition membership",
            lambda values: values | {"condition_membership": np.ones((3, 1), dtype=np.int64)},
        ),
        (
            "site validity shape",
            lambda values: values | {"site_trial_valid": np.ones((1, 2), dtype=bool)},
        ),
        (
            "duplicate stable trial rows",
            lambda values: values | {"stable_trial_rows": np.array([3, 3, 7], dtype=np.int64)},
        ),
        (
            "negative stable trial rows",
            lambda values: values | {"stable_trial_rows": np.array([-3, 4, 7], dtype=np.int64)},
        ),
        (
            "duplicate site identifiers",
            lambda values: values | {"site_ids": ("site-a", "site-a")},
        ),
        (
            "condition name count",
            lambda values: values | {"condition_names": ("only-one", "extra")},
        ),
        (
            "Boolean source spike count",
            lambda values: values | {"source_trial_spike_count": np.zeros((3, 2, 2), dtype=bool)},
        ),
        (
            "negative source spike count",
            lambda values: values | {"source_trial_spike_count": -np.ones((3, 2, 2), dtype=np.int64)},
        ),
        (
            "source spike count segment axis",
            lambda values: values | {"source_trial_spike_count": np.zeros((3, 2, 1), dtype=np.int64)},
        ),
        (
            "Boolean frequency count",
            lambda values: values | {"frequency_count": True},
        ),
        (
            "Boolean mmap byte count",
            lambda values: values | {"shared_phase_mmap_bytes": False},
        ),
    ),
)
def test_plan_rejects_malformed_identity_and_allocation_inputs(
    description: str,
    mutate: object,
) -> None:
    """Pure planning rejects malformed categorical axes and byte inputs before work."""
    del description
    config = _planner_config()
    inputs = _planner_inputs()
    with pytest.raises(ValueError):
        _plan(config, **mutate(inputs))


@pytest.mark.parametrize(
    "overrides",
    (
        {"active_job_count": True},
        {"worker_result_job_count": 0},
        {"component_job_count": 0},
        {"active_job_count": 2, "worker_result_job_count": 1},
        {"worker_result_job_count": 2, "component_job_count": 1},
        {"shuffle_count": 0},
        {"total_unit_count": 0},
        {"unit_block_size": False},
        {"total_unit_count": 3, "source_trial_spike_count": np.zeros((2, 2, 2), dtype=np.int64)},
        {"source_trial_spike_count": np.zeros((2, 1, 1), dtype=np.int64)},
        {"edge_source_trial_position": np.array([2], dtype=np.int64)},
        {"observed_source_trial_position": np.array([2], dtype=np.int64)},
        {"observed_source_trial_position": np.array([0, 0], dtype=np.int64)},
        {"observed_source_trial_position": np.array([True], dtype=bool)},
        {"frequency_count": 0},
        {"representative_band_count": 0},
        {"phase_bin_count": False},
        {"planner_array_bytes": True},
        {"planner_array_bytes": np.iinfo(np.int64).max + 1},
        {"condition_membership_bytes": True},
        {"source_trial_spike_count_bytes": -1},
        {"planning_working_bytes": 1.5},
        {"worker_plan_bytes": -1},
        {"worker_count": 0},
        {"pending_unit_block_count": -1},
        {"shared_phase_mmap_bytes": -1},
    ),
)
def test_allocation_estimate_rejects_malformed_axes_counts_and_positions(
    overrides: dict[str, object],
) -> None:
    """The estimator validates every array/count contract before byte arithmetic."""
    arguments: dict[str, object] = {
        "active_job_count": 1,
        "worker_result_job_count": 1,
        "component_job_count": 1,
        "shuffle_count": 1,
        "total_unit_count": 1,
        "unit_block_size": 1,
        "source_trial_spike_count": np.zeros((2, 1, 2), dtype=np.int64),
        "edge_source_trial_position": np.array([0, 1], dtype=np.int64),
        "observed_source_trial_position": np.array([0, 1], dtype=np.int64),
        "frequency_count": 1,
        "representative_band_count": 2,
        "phase_bin_count": 2,
        "planner_array_bytes": 0,
        "worker_plan_bytes": 0,
        "worker_count": 1,
        "pending_unit_block_count": 1,
        "shared_phase_mmap_bytes": 0,
        "condition_membership_bytes": 0,
        "source_trial_spike_count_bytes": 0,
        "planning_working_bytes": 0,
    }
    with pytest.raises(ValueError):
        ppc_runtime.estimate_grouped_ppc_allocation(**(arguments | overrides))


def test_allocation_estimate_rejects_retained_observed_statistics_overflow() -> None:
    """Retained S2 trial statistics use checked arithmetic before multiplying bytes."""
    with pytest.raises(ValueError):
        ppc_runtime.estimate_grouped_ppc_allocation(
            active_job_count=1,
            worker_result_job_count=1,
            component_job_count=1,
            shuffle_count=1,
            total_unit_count=1,
            unit_block_size=1,
            source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
            edge_source_trial_position=np.array([0], dtype=np.int64),
            observed_source_trial_position=np.array([0], dtype=np.int64),
            frequency_count=1,
            representative_band_count=2,
            phase_bin_count=np.iinfo(np.int64).max,
            planner_array_bytes=0,
            worker_plan_bytes=0,
            worker_count=1,
            pending_unit_block_count=1,
            shared_phase_mmap_bytes=0,
        )


@pytest.mark.parametrize(
    ("active_job_count", "worker_result_job_count", "component_job_count"),
    (
        (1, 0, 1),
        (2, 1, 2),
        (1, 2, 1),
    ),
)
def test_allocation_estimate_rejects_incoherent_job_lifetimes(
    active_job_count: int,
    worker_result_job_count: int,
    component_job_count: int,
) -> None:
    """Null-batch jobs are a bounded subset of retained site/block result jobs."""
    with pytest.raises(ValueError):
        ppc_runtime.estimate_grouped_ppc_allocation(
            active_job_count=active_job_count,
            worker_result_job_count=worker_result_job_count,
            component_job_count=component_job_count,
            shuffle_count=1,
            total_unit_count=1,
            unit_block_size=1,
            source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
            edge_source_trial_position=np.empty(0, dtype=np.int64),
            observed_source_trial_position=np.empty(0, dtype=np.int64),
            frequency_count=1,
            representative_band_count=2,
            phase_bin_count=2,
            planner_array_bytes=0,
            worker_plan_bytes=0,
            worker_count=1,
            pending_unit_block_count=1,
            shared_phase_mmap_bytes=0,
        )


@pytest.mark.parametrize("shuffle_count", (3, 100, 1000))
def test_allocation_job_accumulators_use_full_schedule_not_shuffle_block(
    shuffle_count: int,
) -> None:
    """Every active job retains its full null draw array for all schedule rows.

    The configured shuffle block controls traversal only.  It cannot reduce the
    accumulator needed to summarize all planned shuffles, including a three-row
    schedule smaller than the default block size.
    """
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=2,
        worker_result_job_count=2,
        component_job_count=2,
        shuffle_count=shuffle_count,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.array([0], dtype=np.int64),
        observed_source_trial_position=np.array([0], dtype=np.int64),
        frequency_count=3,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=0,
        worker_plan_bytes=0,
        worker_count=1,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )

    assert estimate.job_accumulator_bytes == 40 * 2 * shuffle_count * 1 * 3


def test_allocation_estimate_accepts_zero_observed_and_null_edge_sets() -> None:
    """Empty observed and shuffled source sets are valid planned no-work states."""
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=0,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=100,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.empty(0, dtype=np.int64),
        observed_source_trial_position=np.empty(0, dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=0,
        worker_plan_bytes=0,
        worker_count=2,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )

    assert estimate.observed_trial_statistics_bytes == 0
    assert estimate.observed_gather_temporary_bytes == 0
    assert estimate.job_accumulator_bytes == 0
    assert estimate.kernel_working_bytes == 0
    assert estimate.active_worker_count == 1


def test_allocation_estimate_retains_singleton_observed_summary_without_null_edges() -> None:
    """A singleton job has observed work and complete site summaries but no null work.

    The worker result arrays span all three epoch jobs at the current site/unit
    block even though their condition batch has no null-eligible schedule cells.
    """
    estimate = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=0,
        worker_result_job_count=3,
        component_job_count=3,
        shuffle_count=100,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.array([[[3, 4]]], dtype=np.int64),
        edge_source_trial_position=np.empty(0, dtype=np.int64),
        observed_source_trial_position=np.array([0], dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=100,
        worker_plan_bytes=40,
        worker_count=2,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )

    geometry_bytes = 26 * 7 + (1 * 2 + 1) * 8 + 8
    observed_statistics_bytes = 8 + 2 * (16 + 8) + 2 * 8 + 2 * 8 + 3 * 8 + 2 * 2 * 2 * 8
    observed_gather_bytes = 7 * 51
    computation_bytes = geometry_bytes + observed_statistics_bytes + observed_gather_bytes
    summary_bytes = 3 * (116 + 2 * 2 * 8)

    assert estimate.job_accumulator_bytes == 0
    assert estimate.kernel_working_bytes == 0
    assert estimate.observed_trial_statistics_bytes == observed_statistics_bytes
    assert estimate.observed_gather_temporary_bytes == observed_gather_bytes
    assert estimate.planned_computation_private_bytes == computation_bytes
    assert estimate.summary_assembly_bytes == summary_bytes
    assert estimate.worker_summary_bytes == summary_bytes
    assert estimate.checkpoint_block_bytes == summary_bytes + 3 * 8
    assert estimate.planned_parent_private_bytes == 100 + summary_bytes + estimate.checkpoint_block_bytes
    assert estimate.planned_worker_private_bytes == 40 + summary_bytes + computation_bytes


@pytest.mark.parametrize(
    "overrides",
    (
        {"planner_array_bytes": np.iinfo(np.int64).max},
        {
            "condition_membership_bytes": np.iinfo(np.int64).max,
            "source_trial_spike_count_bytes": 1,
        },
        {
            "condition_membership_bytes": 1,
            "planning_working_bytes": np.iinfo(np.int64).max,
        },
        {"worker_plan_bytes": np.iinfo(np.int64).max},
        {"shared_phase_mmap_bytes": np.iinfo(np.int64).max},
        {
            "total_unit_count": 2,
            "worker_count": 2,
            "pending_unit_block_count": 2,
            "worker_plan_bytes": np.iinfo(np.int64).max // 2,
        },
        {
            "shared_phase_mmap_bytes": 1,
            "planning_working_bytes": np.iinfo(np.int64).max,
        },
    ),
)
def test_allocation_estimate_checks_parent_worker_shared_and_aggregate_overflow(
    overrides: dict[str, object],
) -> None:
    """Each parent/worker/aggregate lifetime addition is checked before wrapping."""
    arguments: dict[str, object] = {
        "active_job_count": 1,
        "worker_result_job_count": 1,
        "component_job_count": 1,
        "shuffle_count": 1,
        "total_unit_count": 1,
        "unit_block_size": 1,
        "source_trial_spike_count": np.zeros((1, 1, 2), dtype=np.int64),
        "edge_source_trial_position": np.array([0], dtype=np.int64),
        "observed_source_trial_position": np.array([0], dtype=np.int64),
        "frequency_count": 1,
        "representative_band_count": 2,
        "phase_bin_count": 2,
        "planner_array_bytes": 0,
        "worker_plan_bytes": 0,
        "worker_count": 2,
        "pending_unit_block_count": 1,
        "shared_phase_mmap_bytes": 0,
        "condition_membership_bytes": 0,
        "source_trial_spike_count_bytes": 0,
        "planning_working_bytes": 0,
    }
    with pytest.raises(ValueError):
        ppc_runtime.estimate_grouped_ppc_allocation(**(arguments | overrides))


def test_allocation_estimate_accepts_exact_int64_planning_boundary() -> None:
    """An exact planning boundary is valid until a concurrent byte is added."""
    boundary = ppc_runtime.estimate_grouped_ppc_allocation(
        active_job_count=0,
        worker_result_job_count=1,
        component_job_count=1,
        shuffle_count=1,
        total_unit_count=1,
        unit_block_size=1,
        source_trial_spike_count=np.zeros((1, 1, 2), dtype=np.int64),
        edge_source_trial_position=np.empty(0, dtype=np.int64),
        observed_source_trial_position=np.empty(0, dtype=np.int64),
        frequency_count=1,
        representative_band_count=2,
        phase_bin_count=2,
        planner_array_bytes=0,
        worker_plan_bytes=0,
        worker_count=1,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
        condition_membership_bytes=np.iinfo(np.int64).max,
        source_trial_spike_count_bytes=0,
        planning_working_bytes=0,
    )
    assert boundary.planned_planning_private_bytes == np.iinfo(np.int64).max
    assert boundary.planned_parent_private_bytes == np.iinfo(np.int64).max
    assert boundary.planned_aggregate_array_bytes == np.iinfo(np.int64).max


def test_plan_returns_owned_immutable_schedule_and_union_arrays() -> None:
    """Planner outputs cannot be changed through caller inputs or mutable result buffers."""
    config = _planner_config(shuffle_count=2)
    inputs = _planner_inputs(
        stable_trial_rows=np.array([2, 7, 13], dtype=np.int64),
        source_trial_spike_count=np.zeros((3, 1, 2), dtype=np.int64),
    )
    plan = _plan(config, **inputs)
    job = _job_by_identity(plan, condition_index=0, site_index=0, epoch_name="before")
    before_rows = job.selected_trial_rows.copy()
    before_schedule = job.schedule.copy()
    before_source = job.stable_edge_source_trial_row.copy()
    before_target = job.stable_edge_target_trial_row.copy()
    before_position = job.edge_union_position.copy()
    before_edge_site = plan.edge_site_index.copy()
    before_union_source = plan.stable_edge_source_trial_row.copy()
    before_union_target = plan.stable_edge_target_trial_row.copy()

    inputs["stable_trial_rows"][0] = 999
    inputs["source_trial_spike_count"][0, 0, 0] = 999

    np.testing.assert_array_equal(job.selected_trial_rows, before_rows)
    np.testing.assert_array_equal(job.schedule, before_schedule)
    np.testing.assert_array_equal(job.stable_edge_source_trial_row, before_source)
    np.testing.assert_array_equal(job.stable_edge_target_trial_row, before_target)
    np.testing.assert_array_equal(job.edge_union_position, before_position)
    np.testing.assert_array_equal(plan.edge_site_index, before_edge_site)
    np.testing.assert_array_equal(plan.stable_edge_source_trial_row, before_union_source)
    np.testing.assert_array_equal(plan.stable_edge_target_trial_row, before_union_target)
    assert not np.shares_memory(job.selected_trial_rows, inputs["stable_trial_rows"])
    assert not np.shares_memory(job.stable_edge_source_trial_row, inputs["stable_trial_rows"])
    assert not np.shares_memory(job.stable_edge_target_trial_row, inputs["stable_trial_rows"])
    assert not np.shares_memory(plan.stable_edge_source_trial_row, inputs["stable_trial_rows"])
    assert not np.shares_memory(plan.stable_edge_target_trial_row, inputs["stable_trial_rows"])
    protected_arrays = (
        job.selected_trial_rows,
        job.schedule,
        job.stable_edge_source_trial_row,
        job.stable_edge_target_trial_row,
        job.edge_union_position,
        plan.edge_site_index,
        plan.stable_edge_source_trial_row,
        plan.stable_edge_target_trial_row,
    )
    for array in protected_arrays:
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]
    with pytest.raises(FrozenInstanceError):
        job.schedule = np.empty((0, 0), dtype=np.int64)


def test_direct_planner_records_own_freeze_and_validate_array_contracts() -> None:
    """Public plan records defend their advertised immutable identity arrays."""
    selected_rows = np.array([10, 20], dtype=np.int64)
    schedule = np.array([[1, 0]], dtype=np.int64)
    stable_source = np.array([[10, 20]], dtype=np.int64)
    stable_target = np.array([[20, 10]], dtype=np.int64)
    union_position = np.array([[0, 1]], dtype=np.int64)
    job = ppc_runtime.PPCJobPlan(
        condition_index=0,
        condition_name="condition",
        site_index=0,
        site_id="site",
        epoch_index=0,
        epoch_name="whole",
        selected_trial_rows=selected_rows,
        schedule=schedule,
        stable_edge_source_trial_row=stable_source,
        stable_edge_target_trial_row=stable_target,
        edge_union_position=union_position,
        segment_expression="before + after",
        base_ppc_seed=11,
        schedule_seed=11,
        condition_derivation_identity=0,
        site_derivation_identity=0,
        epoch_derivation_identity=0,
        schedule_shape=(1, 2),
        schedule_fingerprint=ppc_runtime._array_fingerprint(schedule),
    )
    allocation = ppc_runtime.PPCAllocationEstimate(
        job_accumulator_bytes=0,
        observed_trial_statistics_bytes=0,
        observed_gather_temporary_bytes=0,
        kernel_working_bytes=0,
        geometry_bytes=0,
        condition_membership_bytes=0,
        source_trial_spike_count_bytes=0,
        planner_array_bytes=0,
        planning_working_bytes=0,
        planned_planning_private_bytes=0,
        summary_assembly_bytes=0,
        worker_plan_bytes=0,
        worker_summary_bytes=0,
        checkpoint_block_bytes=0,
        planned_computation_private_bytes=0,
        planned_parent_private_bytes=0,
        planned_worker_private_bytes=0,
        shared_phase_mmap_bytes=0,
        planned_aggregate_array_bytes=0,
        active_worker_count=0,
    )
    edge_site = np.array([0, 0], dtype=np.int64)
    edge_source = np.array([10, 20], dtype=np.int64)
    edge_target = np.array([20, 10], dtype=np.int64)
    component = ppc_runtime.PPCComponentPlan(
        job_plans=(job,),
        edge_site_index=edge_site,
        stable_edge_source_trial_row=edge_source,
        stable_edge_target_trial_row=edge_target,
        condition_batches=(((0,),),),
        scheduled_edge_count=2,
        independent_edge_count=2,
        union_edge_count=2,
        edge_union_saturation=1.0,
        edge_reuse_ratio=1.0,
        allocation_estimate=allocation,
    )

    selected_rows[0] = 99
    schedule[0, 0] = 0
    stable_source[0, 0] = 99
    stable_target[0, 0] = 99
    union_position[0, 0] = 1
    edge_site[0] = 1
    edge_source[0] = 99
    edge_target[0] = 99

    np.testing.assert_array_equal(job.selected_trial_rows, np.array([10, 20], dtype=np.int64))
    np.testing.assert_array_equal(job.schedule, np.array([[1, 0]], dtype=np.int64))
    np.testing.assert_array_equal(job.stable_edge_source_trial_row, np.array([[10, 20]], dtype=np.int64))
    np.testing.assert_array_equal(job.stable_edge_target_trial_row, np.array([[20, 10]], dtype=np.int64))
    np.testing.assert_array_equal(job.edge_union_position, np.array([[0, 1]], dtype=np.int64))
    np.testing.assert_array_equal(component.edge_site_index, np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(component.stable_edge_source_trial_row, np.array([10, 20], dtype=np.int64))
    np.testing.assert_array_equal(component.stable_edge_target_trial_row, np.array([20, 10], dtype=np.int64))
    for array in (
        job.selected_trial_rows,
        job.schedule,
        job.stable_edge_source_trial_row,
        job.stable_edge_target_trial_row,
        job.edge_union_position,
        component.edge_site_index,
        component.stable_edge_source_trial_row,
        component.stable_edge_target_trial_row,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]

    with pytest.raises(ValueError):
        replace(
            job,
            stable_edge_target_trial_row=np.array([[10, 20]], dtype=np.int64),
        )
    with pytest.raises(ValueError):
        replace(component, edge_site_index=np.array([0], dtype=np.int64))
    with pytest.raises(ValueError):
        replace(
            component,
            stable_edge_target_trial_row=np.array([10, 20], dtype=np.int64),
        )


def _direct_job_plan(
    *,
    selected_trial_rows: np.ndarray | None = None,
    schedule: np.ndarray | None = None,
    condition_index: int = 0,
    site_index: int = 0,
    epoch_index: int = 0,
    **overrides: object,
) -> ppc_runtime.PPCJobPlan:
    """Construct one valid direct public job record before a targeted override."""
    selected = (
        np.array([10, 20], dtype=np.int64)
        if selected_trial_rows is None
        else selected_trial_rows
    )
    local_schedule = (
        np.array([[1, 0]], dtype=np.int64) if schedule is None else schedule
    )
    epoch_name, segment_expression = (
        ("whole", "before + after")
        if epoch_index == 0
        else ("before", "before")
        if epoch_index == 1
        else ("after", "after")
    )
    source = np.broadcast_to(selected[np.newaxis, :], local_schedule.shape).copy()
    target = (
        selected[local_schedule].copy()
        if local_schedule.size
        else np.empty(local_schedule.shape, dtype=np.int64)
    )
    values: dict[str, object] = {
        "condition_index": condition_index,
        "condition_name": f"condition-{condition_index}",
        "site_index": site_index,
        "site_id": f"site-{site_index}",
        "epoch_index": epoch_index,
        "epoch_name": epoch_name,
        "selected_trial_rows": selected,
        "schedule": local_schedule,
        "stable_edge_source_trial_row": source,
        "stable_edge_target_trial_row": target,
        "edge_union_position": np.arange(local_schedule.size, dtype=np.int64).reshape(
            local_schedule.shape
        ),
        "segment_expression": segment_expression,
        "base_ppc_seed": 11,
        "schedule_seed": 11 + condition_index * 1_000_000 + site_index * 10_000 + epoch_index * 100,
        "condition_derivation_identity": condition_index,
        "site_derivation_identity": site_index,
        "epoch_derivation_identity": epoch_index,
        "schedule_shape": local_schedule.shape,
        "schedule_fingerprint": ppc_runtime._array_fingerprint(local_schedule),
    }
    values.update(overrides)
    return ppc_runtime.PPCJobPlan(**values)


def _direct_allocation_estimate() -> ppc_runtime.PPCAllocationEstimate:
    """Return a zero-byte direct-record allocation placeholder."""
    return ppc_runtime.PPCAllocationEstimate(**dict.fromkeys(
        (field.name for field in fields(ppc_runtime.PPCAllocationEstimate)), 0
    ))


def _direct_component_fields(
    job_plans: tuple[ppc_runtime.PPCJobPlan, ...],
) -> dict[str, object]:
    """Return exact direct component fields derived from valid public jobs."""
    union_edges = sorted(
        {
            (job.site_index, int(source), int(target))
            for job in job_plans
            for source, target in zip(
                job.stable_edge_source_trial_row.ravel(),
                job.stable_edge_target_trial_row.ravel(),
                strict=True,
            )
        }
    )
    allowed_edges = {
        (job.site_index, int(source), int(target))
        for job in job_plans
        for source in job.selected_trial_rows
        for target in job.selected_trial_rows
        if source != target
    }
    scheduled_edge_count = sum(job.schedule.size for job in job_plans)
    independent_edge_count = sum(
        len(
            set(
                zip(
                    job.stable_edge_source_trial_row.ravel(),
                    job.stable_edge_target_trial_row.ravel(),
                    strict=True,
                )
            )
        )
        for job in job_plans
    )
    site_count = 1 + max(job.site_index for job in job_plans)
    condition_batches = tuple(
        (
            tuple(
                sorted(
                    {
                        job.condition_index
                        for job in job_plans
                        if job.site_index == site_index
                    }
                )
            ),
        )
        for site_index in range(site_count)
    )
    union_count = len(union_edges)
    return {
        "job_plans": job_plans,
        "edge_site_index": np.asarray(
            [site for site, _, _ in union_edges], dtype=np.int64
        ),
        "stable_edge_source_trial_row": np.asarray(
            [source for _, source, _ in union_edges], dtype=np.int64
        ),
        "stable_edge_target_trial_row": np.asarray(
            [target for _, _, target in union_edges], dtype=np.int64
        ),
        "condition_batches": condition_batches,
        "scheduled_edge_count": scheduled_edge_count,
        "independent_edge_count": independent_edge_count,
        "union_edge_count": union_count,
        "edge_union_saturation": (
            float(union_count) / float(len(allowed_edges)) if allowed_edges else 0.0
        ),
        "edge_reuse_ratio": (
            float(independent_edge_count) / float(union_count) if union_count else 0.0
        ),
        "allocation_estimate": _direct_allocation_estimate(),
    }


@pytest.mark.parametrize(
    ("selected_trial_rows", "schedule", "overrides"),
    (
        (
            np.array([10, 20, 30, 40], dtype=np.int64),
            np.empty((0, 4), dtype=np.int64),
            {},
        ),
        (
            np.array([10, 20, 30, 40], dtype=np.int64),
            np.array([[1, 0, 3, 1]], dtype=np.int64),
            {},
        ),
        (None, None, {"base_ppc_seed": 11.5}),
        (None, None, {"schedule_seed": 12}),
        (
            None,
            None,
            {"condition_derivation_identity": 1, "schedule_seed": 1_000_011},
        ),
        (None, None, {"site_derivation_identity": 1, "schedule_seed": 10_011}),
        (None, None, {"epoch_derivation_identity": 1, "schedule_seed": 111}),
        (
            None,
            None,
            {
                "epoch_index": 1,
                "epoch_name": "whole",
                "segment_expression": "before + after",
                "epoch_derivation_identity": 1,
                "schedule_seed": 111,
            },
        ),
    ),
)
def test_direct_job_plan_rejects_permutation_and_provenance_incoherence(
    selected_trial_rows: np.ndarray | None,
    schedule: np.ndarray | None,
    overrides: dict[str, object],
) -> None:
    """Public jobs require nonempty derangement permutations and exact provenance."""
    with pytest.raises(ValueError):
        _direct_job_plan(
            selected_trial_rows=selected_trial_rows,
            schedule=schedule,
            **overrides,
        )


@pytest.mark.parametrize("schedule_shape", ((True, 2), (1.5, 2.0)))
def test_direct_job_plan_rejects_noninteger_schedule_shape_elements(
    schedule_shape: tuple[object, object],
) -> None:
    """Checkpoint schedule shapes require exact non-Boolean integer elements."""
    with pytest.raises(ValueError):
        _direct_job_plan(schedule_shape=schedule_shape)


@pytest.mark.parametrize("kind", ("duplicate", "unsorted", "extra"))
def test_direct_component_plan_requires_exact_sorted_unique_job_union(kind: str) -> None:
    """Public unions cannot contain duplicate, reordered, or surplus job edges."""
    job = _direct_job_plan()
    fields_by_name = _direct_component_fields((job,))
    if kind == "duplicate":
        fields_by_name.update(
            edge_site_index=np.array([0, 0, 0], dtype=np.int64),
            stable_edge_source_trial_row=np.array([10, 20, 10], dtype=np.int64),
            stable_edge_target_trial_row=np.array([20, 10, 20], dtype=np.int64),
            union_edge_count=3,
        )
    elif kind == "unsorted":
        reordered_job = replace(
            job,
            edge_union_position=np.array([[1, 0]], dtype=np.int64),
        )
        fields_by_name = _direct_component_fields((reordered_job,))
        fields_by_name.update(
            edge_site_index=np.array([0, 0], dtype=np.int64),
            stable_edge_source_trial_row=np.array([20, 10], dtype=np.int64),
            stable_edge_target_trial_row=np.array([10, 20], dtype=np.int64),
        )
    else:
        fields_by_name.update(
            edge_site_index=np.array([0, 0, 0], dtype=np.int64),
            stable_edge_source_trial_row=np.array([10, 20, 30], dtype=np.int64),
            stable_edge_target_trial_row=np.array([20, 10, 40], dtype=np.int64),
            union_edge_count=3,
        )
    with pytest.raises(ValueError):
        ppc_runtime.PPCComponentPlan(**fields_by_name)


@pytest.mark.parametrize(
    "overrides",
    (
        {"scheduled_edge_count": 1},
        {"independent_edge_count": 1, "edge_reuse_ratio": 0.5},
        {"edge_union_saturation": True},
        {"edge_union_saturation": -0.1},
        {"edge_union_saturation": 1.1},
        {"edge_union_saturation": 0.5},
        {"edge_reuse_ratio": True},
        {"edge_reuse_ratio": -0.1},
        {"edge_reuse_ratio": 0.5},
    ),
)
def test_direct_component_plan_rejects_count_and_metric_incoherence(
    overrides: dict[str, object],
) -> None:
    """Public component counts and derived dimensionless metrics are exact."""
    fields_by_name = _direct_component_fields((_direct_job_plan(),))
    fields_by_name.update(overrides)
    with pytest.raises(ValueError):
        ppc_runtime.PPCComponentPlan(**fields_by_name)


@pytest.mark.parametrize(
    "condition_batches",
    (
        (((0, 0),),),
        (((0,),),),
        (((1, 0),),),
    ),
)
def test_direct_component_plan_requires_ordered_complete_site_condition_batches(
    condition_batches: tuple[tuple[tuple[int, ...], ...], ...],
) -> None:
    """Every site's condition batches must cover each job condition exactly once."""
    jobs = (_direct_job_plan(), _direct_job_plan(condition_index=1))
    fields_by_name = _direct_component_fields(jobs)
    fields_by_name["condition_batches"] = condition_batches
    with pytest.raises(ValueError):
        ppc_runtime.PPCComponentPlan(**fields_by_name)


def test_plan_rejects_unsafe_aggregate_workers_before_any_process_or_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 12-GiB-style aggregate guard fails from the pure plan before spawn."""
    config = _planner_config(
        shuffle_count=1,
        unit_block_size=1,
        shuffle_block_size=1,
        worker_count=2,
        maximum_aggregate_allocation_bytes=1,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([1, 4], dtype=np.int64),
        source_trial_spike_count=np.zeros((2, 2, 2), dtype=np.int64),
        frequency_count=1,
    )

    def forbidden(*_: object, **__: object) -> None:
        """Fail if an aggregate preflight performs computation or process setup."""
        raise AssertionError("unsafe worker count reached execution")

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(ppc_runtime, "_compute_unit_blocks", forbidden)
    with pytest.raises(ValueError, match="aggregate"):
        _plan(config, **inputs)


def test_plan_rejects_parent_private_peak_even_when_bounded_workers_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 2-GiB-style per-process limit applies to parent summary ownership too.

    With 20 total units, the parent retains 288 bytes of component planning
    arrays and 8,880 bytes of final component summaries. A one-unit worker
    task needs only 1,092 bytes, so condition batching cannot make this parent
    allocation legal under the deliberately small 9,000-byte limit.
    """
    config = _planner_config(
        shuffle_count=1,
        unit_block_size=1,
        shuffle_block_size=1,
        worker_count=2,
        maximum_worker_allocation_bytes=9_000,
    )
    inputs = _planner_inputs(
        stable_trial_rows=np.array([1, 4], dtype=np.int64),
        source_trial_spike_count=np.zeros((2, 20, 2), dtype=np.int64),
        frequency_count=1,
    )

    def forbidden(*_: object, **__: object) -> None:
        """A pure planning rejection must occur before execution/process setup."""
        raise AssertionError("unsafe parent allocation reached execution")

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden)
    monkeypatch.setattr(ppc_runtime, "_compute_unit_blocks", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "broadcast_to", forbidden)
    with pytest.raises(ValueError, match="parent"):
        _plan(config, **inputs)


def test_allocation_limits_enter_existing_work_metadata_and_fingerprint(
    tmp_path: Path,
) -> None:
    """Execution-only limits are recorded in resumable work identity, not final science."""
    config = _planner_config(
        maximum_worker_allocation_bytes=123_456,
        maximum_aggregate_allocation_bytes=654_321,
    )
    changed_worker = replace(
        config,
        ppc_execution=replace(
            config.ppc_execution,
            maximum_worker_allocation_bytes=123_457,
        ),
    )
    changed_aggregate = replace(
        config,
        ppc_execution=replace(
            config.ppc_execution,
            maximum_aggregate_allocation_bytes=654_322,
        ),
    )
    phase, spikes, schedule = _inputs()
    first = ppc_runtime.execute_ppc_blocks(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )
    second = ppc_runtime.execute_ppc_blocks(
        config=changed_worker,
        execution=changed_worker.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )
    third = ppc_runtime.execute_ppc_blocks(
        config=changed_aggregate,
        execution=changed_aggregate.ppc_execution,
        prepared_phase=phase,
        prepared_spikes=spikes,
        schedule=schedule,
        work_root=tmp_path,
    )
    metadata = json.loads((first.run_directory / "metadata.json").read_text(encoding="utf-8"))

    assert metadata["execution_settings"]["maximum_worker_allocation_bytes"] == 123_456
    assert metadata["execution_settings"]["maximum_aggregate_allocation_bytes"] == 654_321
    assert first.run_fingerprint != second.run_fingerprint
    assert first.run_fingerprint != third.run_fingerprint
    assert second.run_fingerprint != third.run_fingerprint
    assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed_worker)
    assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed_aggregate)


# S4 grouped serial execution contracts.  The S3 planner above deliberately
# stops before phase/spike sampling; these tests freeze the first executor that
# consumes production prepared records and emits component-axis summaries.


_GROUPED_SUMMARY_FIELDS = frozenset(
    (*ppc_runtime._SUMMARY_FIELDS, "representative_phase_histogram_count")
)


def _grouped_config(
    *,
    worker_count: int = 1,
    maximum_worker_allocation_bytes: int = 2 * 1024**3,
    maximum_aggregate_allocation_bytes: int = 12 * 1024**3,
) -> object:
    """Return a valid two-site production configuration for grouped PPC tests.

    The configured phase grid has 8 and 40 Hz coordinates and retains the
    canonical 500-Hz event-relative sampling rate.  One-unit, one-edge, and
    one-shuffle blocks expose every bounded S4 reduction seam without changing
    the scientific estimator.
    """
    base = default_lfp_summary_config()
    return replace(
        base,
        sites=base.sites[:2],
        site_pairs=(base.site_pairs[0],),
        unit_population=UnitPopulationConfig(
            label="selected_population",
            probe_label="PFC",
            sorter_path=None,
            aligned_spike_path=None,
            selected_channels=(0, 1),
            quality_settings=(),
            stable_unit_ids=("PFC:1", "PFC:2"),
        ),
        phase=replace(base.phase, frequency_hz=(8.0, 40.0)),
        ppc=replace(base.ppc, shuffle_count=3, seed=31),
        ppc_execution=PPCExecutionConfig(
            unit_block_size=1,
            shuffle_block_size=1,
            trial_edge_block_size=1,
            worker_count=worker_count,
            maximum_worker_allocation_bytes=maximum_worker_allocation_bytes,
            maximum_aggregate_allocation_bytes=maximum_aggregate_allocation_bytes,
            checkpoint_enabled=True,
        ),
    )


def _grouped_inputs(config: object) -> tuple[object, object]:
    """Return real prepared records with overlapping and excluded conditions.

    The full trial axis contains six stable int64 trial rows.  Filter,
    objective-validity, and user-exclusion masks retain only its first three
    rows.  ``all`` therefore selects three rows, ``late`` selects rows 59/83,
    ``empty`` selects none, and ``single`` selects only row 41 at both sites.
    Unit two is whole-window null eligible but each half is ineligible for the
    two-trial late condition.  Phase has complex64/Boolean
    ``(site, frequency, trial, time)`` axes on the configured 500-Hz grid;
    spike trains have one finite, trial-distinct seconds vector per full trial.
    """
    stable_rows = np.array([41, 59, 83, 97, 101, 113], dtype=np.int64)
    time_s = build_common_event_grid(-2.0, 2.0, float(config.phase.output_rate_hz))
    assert time_s.shape == (2000,)
    frequency_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    phase_angle = (
        np.arange(len(config.sites), dtype=float)[:, None, None, None] * 0.11
        + frequency_hz[None, :, None, None] * time_s[None, None, None, :] * 0.03
        + np.arange(stable_rows.size, dtype=float)[None, None, :, None] * 0.17
    )
    phase = np.exp(1j * phase_angle).astype(np.complex64)
    condition_membership = np.array(
        [
            [True, False, False, True],
            [True, True, False, False],
            [True, True, False, False],
            [True, False, True, False],
            [True, False, False, False],
            [True, False, False, False],
        ],
        dtype=bool,
    )
    filter_membership = np.array([True, True, True, False, True, True], dtype=bool)
    objective_valid = np.array([True, True, True, True, False, True], dtype=bool)
    user_excluded = np.array([False, False, False, False, False, True], dtype=bool)
    # Objective-invalid trial 101 is deliberately unavailable at every layer,
    # rather than merely excluded by a categorical planner mask.
    site_valid = np.ones((len(config.sites), stable_rows.size), dtype=bool)
    site_valid[:, 4] = False
    phase_valid = np.ones(phase.shape, dtype=bool)
    phase_valid[:, :, 4] = False
    phase[:, :, 4] = 0.0j
    site_validity = {
        site.stable_id: site_valid[site_index].copy()
        for site_index, site in enumerate(config.sites)
    }
    prepared_trials = PreparedTrials(
        condition_names=("all", "late", "empty", "single"),
        condition_membership=condition_membership,
        filter_membership=filter_membership,
        user_excluded=user_excluded,
        objective_valid=objective_valid,
        objective_exclusion_reason=np.where(~objective_valid, "objective", "").astype("<U9"),
        user_exclusion_reason=np.where(user_excluded, "user", "").astype("<U4"),
        site_validity=site_validity,
        pair_validity={
            tuple(config.site_pairs[0]): site_valid[0] & site_valid[1],
        },
    )
    prepared_phase = lfp_summary_runtime.PreparedPhaseRun(
        trial_indices=stable_rows,
        alignment_times_s=np.array([0.0, 1.0, 2.0, 3.0, np.nan, 5.0]),
        prepared_trials=prepared_trials,
        phase_tensor=phase,
        phase_valid=phase_valid,
        relative_time_s=time_s,
        site_valid=site_valid,
        pair_valid=(site_valid[0] & site_valid[1])[None, :],
        source_trace=np.where(
            site_valid[:, :, None],
            np.zeros((len(config.sites), stable_rows.size, time_s.size), dtype=float),
            np.nan,
        ),
    )
    dense_before = np.linspace(-1.75, -0.25, 30)
    dense_after = np.linspace(0.25, 1.75, 30)
    sparse_before = np.linspace(-1.70, -0.30, 13)
    sparse_after = np.linspace(0.30, 1.70, 13)
    # Source-trial-specific half-window shifts break the degenerate null in
    # which every source train is identical.  They preserve both spike count
    # and the half-open before/after windows for every valid trial.
    # Each shift is an odd half-sample (1 ms) offset from the canonical 500-Hz
    # grid. This makes interpolation intentional rather than relying on a
    # decimal value that might accidentally compare equal to a grid sample.
    before_shift_s = np.array([-0.181, 0.051, 0.161, -0.101, 0.001, 0.121])
    after_shift_s = np.array([0.151, -0.081, 0.031, 0.101, 0.001, -0.141])

    def shifted_trial_spikes(
        before: np.ndarray,
        after: np.ndarray,
        trial_index: int,
    ) -> np.ndarray:
        """Return one trial-distinct train without crossing PPC half windows."""
        return np.concatenate(
            (
                before + before_shift_s[trial_index],
                after + after_shift_s[trial_index],
            )
        )

    dense_by_trial = tuple(
        np.empty(0, dtype=float)
        if trial_index == 4
        else shifted_trial_spikes(dense_before, dense_after, trial_index)
        for trial_index in range(stable_rows.size)
    )
    sparse_by_trial = tuple(
        np.empty(0, dtype=float)
        if trial_index == 4
        else shifted_trial_spikes(sparse_before, sparse_after, trial_index)
        for trial_index in range(stable_rows.size)
    )
    for values, half_count in (
        (dense_by_trial, dense_before.size),
        (sparse_by_trial, sparse_before.size),
    ):
        for trial_index, trial_spikes in enumerate(values):
            if trial_index == 4:
                assert trial_spikes.size == 0
                continue
            assert trial_spikes.size == 2 * half_count
            assert np.count_nonzero(trial_spikes < 0.0) == half_count
            assert np.count_nonzero(trial_spikes >= 0.0) == half_count
            assert np.all(trial_spikes >= time_s[0])
            assert np.all(trial_spikes < time_s[-1])
            for spike_time_s in trial_spikes:
                insertion = int(np.searchsorted(time_s, spike_time_s, side="left"))
                for grid_index in (insertion - 1, insertion):
                    if not 0 <= grid_index < time_s.size:
                        continue
                    if abs(float(spike_time_s - time_s[grid_index])) <= 1e-12:
                        assert (
                            np.asarray(spike_time_s, dtype=np.float64)
                            .view(np.uint64)
                            .item()
                            == np.asarray(time_s[grid_index], dtype=np.float64)
                            .view(np.uint64)
                            .item()
                        )
    prepared_spikes = lfp_summary_runtime.PreparedSpikeRun(
        unit_ids=("PFC:1", "PFC:2"),
        population_ids=("selected_population",),
        trial_spike_trains=(
            TrialRelativeSpikeTrains(
                unit_id="PFC:1",
                relative_spike_times=tuple(values.copy() for values in dense_by_trial),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            ),
            TrialRelativeSpikeTrains(
                unit_id="PFC:2",
                relative_spike_times=tuple(values.copy() for values in sparse_by_trial),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            ),
        ),
    )
    return prepared_phase, prepared_spikes


def _sparse_large_trial_grouped_inputs(config: object) -> tuple[object, object]:
    """Return real grouped records with a sparse 100-trial planning axis.

    Returns
    -------
    prepared_phase : PreparedPhaseRun
        Complex64 phase and Boolean validity with axes ``(site, frequency,
        trial=100, time)``. Every trial and site is analysis-valid.
    prepared_spikes : PreparedSpikeRun
        Two units with one finite seconds spike-time vector for each of the
        same 100 full-axis trials.

    Notes
    -----
    Only two ``all`` rows, two ``late`` rows, no ``empty`` rows, and one
    ``single`` row are selected. This leaves the full trial axis deliberately
    much larger than schedule demand, so the conservative planner scratch
    bound dominates retained plan arrays before execution allocates anything.
    """
    source_phase, source_spikes = _grouped_inputs(config)
    trial_count = 100
    stable_rows = np.arange(1_000, 1_000 + trial_count, dtype=np.int64)
    site_count, frequency_count, _, time_count = source_phase.phase_tensor.shape
    phase_tensor = np.repeat(source_phase.phase_tensor[:, :, :1, :], trial_count, axis=2)
    phase_valid = np.repeat(source_phase.phase_valid[:, :, :1, :], trial_count, axis=2)
    site_valid = np.ones((site_count, trial_count), dtype=bool)
    condition_membership = np.zeros((trial_count, 4), dtype=bool)
    condition_membership[:2, 0] = True
    condition_membership[2:4, 1] = True
    condition_membership[4, 3] = True
    prepared_trials = PreparedTrials(
        condition_names=("all", "late", "empty", "single"),
        condition_membership=condition_membership,
        filter_membership=np.ones(trial_count, dtype=bool),
        user_excluded=np.zeros(trial_count, dtype=bool),
        objective_valid=np.ones(trial_count, dtype=bool),
        objective_exclusion_reason=np.full(trial_count, "", dtype="<U1"),
        user_exclusion_reason=np.full(trial_count, "", dtype="<U1"),
        site_validity={
            site.stable_id: site_valid[site_index].copy()
            for site_index, site in enumerate(config.sites)
        },
        pair_validity={tuple(config.site_pairs[0]): np.ones(trial_count, dtype=bool)},
    )
    prepared_phase = lfp_summary_runtime.PreparedPhaseRun(
        trial_indices=stable_rows,
        alignment_times_s=np.arange(trial_count, dtype=float),
        prepared_trials=prepared_trials,
        phase_tensor=phase_tensor,
        phase_valid=phase_valid,
        relative_time_s=source_phase.relative_time_s,
        site_valid=site_valid,
        pair_valid=np.ones((1, trial_count), dtype=bool),
        source_trace=np.zeros((site_count, trial_count, time_count), dtype=float),
    )
    prepared_spikes = lfp_summary_runtime.PreparedSpikeRun(
        unit_ids=source_spikes.unit_ids,
        population_ids=source_spikes.population_ids,
        trial_spike_trains=tuple(
            TrialRelativeSpikeTrains(
                unit_id=train.unit_id,
                relative_spike_times=tuple(
                    train.relative_spike_times[0].copy() for _ in range(trial_count)
                ),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            )
            for train in source_spikes.trial_spike_trains
        ),
    )
    assert phase_tensor.shape == (site_count, frequency_count, trial_count, time_count)
    return prepared_phase, prepared_spikes


def _run_grouped_component(
    config: object,
    prepared_phase: object,
    prepared_spikes: object,
    work_root: Path,
    *,
    execution: object | None = None,
    progress_callback: object | None = None,
) -> object:
    """Execute the frozen grouped serial entry point with production records.

    Parameters have the data contracts of ``PreparedPhaseRun`` and
    ``PreparedSpikeRun``.  ``work_root`` is a session-local execution-only
    parent.  A supplied callback receives ``ProgressEvent`` records only.
    """
    selected_execution = config.ppc_execution if execution is None else execution
    return ppc_runtime.execute_grouped_ppc_component(
        config=config,
        execution=selected_execution,
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        work_root=work_root,
        progress_callback=progress_callback,
    )


def _legacy_grouped_reference(
    config: object,
    prepared_phase: object,
    prepared_spikes: object,
    plan: object,
    work_root: Path,
) -> tuple[dict[tuple[int, int, int], object], np.ndarray]:
    """Run independent legacy jobs and build their exact component histogram.

    Every legacy job receives the corresponding planned local schedule unchanged.
    The mapping key is ``(condition, site, epoch)``; the histogram has int64
    ``(unit, condition, site, epoch, band, phase_bin)`` axes.
    """
    epoch_windows = lfp_summary_runtime._selected_ppc_epoch_windows(config)
    row_position = {
        int(row): position
        for position, row in enumerate(np.asarray(prepared_phase.trial_indices))
    }
    histogram = np.zeros(
        (
            len(prepared_spikes.unit_ids),
            len(prepared_phase.prepared_trials.condition_names),
            prepared_phase.phase_tensor.shape[0],
            len(epoch_windows),
            len(config.phase.bands),
            len(config.ppc.phase_bin_edges_rad) - 1,
        ),
        dtype=np.int64,
    )
    execution = replace(config.ppc_execution, worker_count=1, checkpoint_enabled=False)
    source_fingerprint = lfp_summary_runtime._work_fingerprint(
        fingerprint_source_files(config, component="spike_phase")
    )
    results: dict[tuple[int, int, int], object] = {}
    for job in plan.job_plans:
        full_positions = np.asarray(
            [row_position[int(row)] for row in job.selected_trial_rows],
            dtype=np.int64,
        )
        epoch_window = epoch_windows[job.epoch_name]
        job_phase = lfp_summary_runtime._PPCPhaseJob(
            phase_tensor=prepared_phase.phase_tensor[
                job.site_index : job.site_index + 1, :, full_positions, :
            ],
            phase_valid=prepared_phase.phase_valid[
                job.site_index : job.site_index + 1, :, full_positions, :
            ],
            relative_time_s=prepared_phase.relative_time_s,
            trial_indices=np.asarray(job.selected_trial_rows, dtype=np.int64),
            site_id=job.site_id,
            condition_name=job.condition_name,
            epoch_bounds_s=epoch_window,
            source_fingerprint=source_fingerprint,
        )
        trial_spikes_by_unit = tuple(
            tuple(
                lfp_summary_runtime._spikes_in_epoch(
                    train.relative_spike_times[int(position)], epoch_window
                )
                for position in full_positions
            )
            for train in prepared_spikes.trial_spike_trains
        )
        job_spikes = lfp_summary_runtime.PreparedSpikeRun(
            unit_ids=prepared_spikes.unit_ids,
            population_ids=prepared_spikes.population_ids,
            trial_spike_trains=tuple(
                TrialRelativeSpikeTrains(
                    unit_id=unit_id,
                    relative_spike_times=trial_spikes_by_unit[unit_index],
                    overlap_trial_indices=np.empty(0, dtype=np.int64),
                )
                for unit_index, unit_id in enumerate(prepared_spikes.unit_ids)
            ),
        )
        legacy = ppc_runtime.execute_ppc_blocks(
            config=config,
            execution=execution,
            prepared_phase=job_phase,
            prepared_spikes=job_spikes,
            schedule=job.schedule,
            work_root=work_root,
        )
        results[(job.condition_index, job.site_index, job.epoch_index)] = legacy
        phase_by_trial = np.moveaxis(prepared_phase.phase_tensor[job.site_index], 1, 0)
        valid_by_trial = np.moveaxis(prepared_phase.phase_valid[job.site_index], 1, 0)
        for unit_index, selected_spikes in enumerate(trial_spikes_by_unit):
            sampled, sampled_valid, sampled_rows = lfp_summary_runtime._sample_observed_trial_phase(
                prepared_phase.relative_time_s,
                phase_by_trial[full_positions],
                valid_by_trial[full_positions],
                selected_spikes,
                np.asarray(job.selected_trial_rows, dtype=np.int64),
            )
            histogram[
                unit_index,
                job.condition_index,
                job.site_index,
                job.epoch_index,
            ] = spike_lfp_summary.build_representative_phase_histograms(
                frequencies_hz=np.asarray(config.phase.frequency_hz, dtype=float),
                spike_phase_vectors=sampled,
                valid_mask=sampled_valid,
                trial_indices=sampled_rows,
                phase_bin_edges_rad=np.asarray(config.ppc.phase_bin_edges_rad, dtype=float),
            ).spike_count_by_band
    return results, histogram


def _assert_legacy_inferential_margins(
    config: object,
    prepared_phase: object,
    prepared_spikes: object,
    plan: object,
) -> None:
    """Require every exact legacy p/q decision to avoid a numerical tie.

    The grouped implementation reduces source-trial sufficient statistics,
    whereas legacy execution concatenates source-trial phase samples before
    reduction.  Exact inferential comparisons therefore use this fixture only
    when every finite null draw contributing to an eligible decision is at
    least ``1e-5`` from the corresponding observed PPC.
    """
    epoch_windows = lfp_summary_runtime._selected_ppc_epoch_windows(config)
    row_position = {
        int(row): position
        for position, row in enumerate(np.asarray(prepared_phase.trial_indices))
    }
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    checked_decision_count = 0
    for job in plan.job_plans:
        if not job.schedule.size:
            continue
        full_positions = tuple(
            row_position[int(row)] for row in job.selected_trial_rows
        )
        phase_by_trial = np.moveaxis(
            prepared_phase.phase_tensor[job.site_index], 1, 0
        )
        epoch_window = epoch_windows[job.epoch_name]
        for unit_index, train in enumerate(prepared_spikes.trial_spike_trains):
            selected_spikes = tuple(
                lfp_summary_runtime._spikes_in_epoch(
                    train.relative_spike_times[position], epoch_window
                )
                for position in full_positions
            )
            legacy = spike_lfp_summary.compute_trial_shuffle_ppc(
                trial_relative_spike_times_s=selected_spikes,
                phase_time_s=prepared_phase.relative_time_s,
                trial_phase_vectors=phase_by_trial[list(full_positions)],
                frequencies_hz=frequencies_hz,
                schedule=job.schedule,
            )
            eligible = legacy.null_summary.null_eligible
            if not np.any(eligible):
                continue
            sampled = spike_lfp_summary._precompute_trial_phase_samples(
                prepared_phase.relative_time_s,
                phase_by_trial[list(full_positions)],
                selected_spikes,
            )
            null_draws = np.empty(
                (job.schedule.shape[0], frequencies_hz.size), dtype=float
            )
            for shuffle_index, schedule_row in enumerate(job.schedule):
                null_draws[shuffle_index] = spike_lfp_summary._phase_metrics_from_vectors(
                    spike_lfp_summary._pooled_sampled_phase_vectors(
                        sampled, schedule_row
                    ),
                    frequencies_hz,
                ).ppc
            for frequency_index in range(frequencies_hz.size):
                finite_draws = null_draws[
                    np.isfinite(null_draws[:, frequency_index]), frequency_index
                ]
                observed_value = float(legacy.observed_ppc[frequency_index])
                exceedance_count = int(
                    np.count_nonzero(finite_draws >= observed_value)
                )
                assert finite_draws.size == int(
                    legacy.null_summary.permutation_count[frequency_index]
                )
                assert exceedance_count == int(
                    legacy.null_summary.null_exceedance_count[frequency_index]
                )
                if not bool(eligible[frequency_index]):
                    assert np.isnan(legacy.null_summary.p_value[frequency_index])
                    continue
                assert finite_draws.size
                expected_p_value = (1.0 + exceedance_count) / (1.0 + finite_draws.size)
                assert expected_p_value == float(
                    legacy.null_summary.p_value[frequency_index]
                )
                checked_decision_count += 1
                margins = np.abs(finite_draws - observed_value)
                assert np.all(margins >= 1e-5), (
                    "exact inferential fixture is numerically tied: "
                    f"condition={job.condition_name!r}, site={job.site_id!r}, "
                    f"epoch={job.epoch_name!r}, unit={unit_index}, "
                    f"frequency_hz={frequencies_hz[frequency_index]!r}, "
                    f"minimum_margin={float(np.min(margins))!r}"
                )
    assert checked_decision_count > 0, "fixture did not exercise an exact inferential decision"


def _assert_float_summary_equal(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    """Compare one PPC float field with its approved numerical policy."""
    if name == "preferred_phase_rad":
        finite = np.isfinite(actual) & np.isfinite(expected)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        circular_difference = np.angle(np.exp(1j * (actual[finite] - expected[finite])))
        assert np.all(np.abs(circular_difference) <= 1e-6)
    elif name in {"p_value", "q_value"}:
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)
    else:
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7, equal_nan=True)


def _assert_component_summaries_equal(first: object, second: object) -> None:
    """Compare exact/count fields and approved-tolerance grouped float fields."""
    assert set(first.summary_arrays) == _GROUPED_SUMMARY_FIELDS
    assert set(second.summary_arrays) == _GROUPED_SUMMARY_FIELDS
    for name, values in first.summary_arrays.items():
        candidate = second.summary_arrays[name]
        if values.dtype.kind == "f":
            _assert_float_summary_equal(name, values, candidate)
        else:
            np.testing.assert_array_equal(values, candidate)


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load one checkpoint's non-object arrays for intentional corruption tests."""
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in loaded.files}


def test_grouped_component_contract_is_keyword_only_serial_and_work_only() -> None:
    """S4 adds one serial component executor without changing legacy job execution."""
    assert tuple(field.name for field in fields(ppc_runtime.PPCComponentExecutionResult)) == (
        "run_fingerprint",
        "run_directory",
        "component_plan",
        "summary_arrays",
        "completed_block_ids",
        "resumed_block_ids",
    )
    signature = inspect.signature(ppc_runtime.execute_grouped_ppc_component)
    assert tuple(signature.parameters) == (
        "config",
        "execution",
        "prepared_phase",
        "prepared_spikes",
        "work_root",
        "progress_callback",
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in signature.parameters.values()
    )
    assert signature.parameters["progress_callback"].default is None
    assert ppc_runtime.execute_ppc_blocks is not ppc_runtime.execute_grouped_ppc_component


def test_grouped_component_matches_legacy_and_reduces_each_physical_edge_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Grouped output equals legacy cells while physical shuffled edges are reused.

    Whole epochs retain their unchanged derived schedules but consume before/after
    sufficient statistics.  With one-edge blocks, every active physical union
    edge is reduced exactly once per site/unit block and immediately consumed in
    one-shuffle chunks before the next edge reduction starts.
    """
    config = _grouped_config()
    prepared_phase, prepared_spikes = _grouped_inputs(config)
    dense_train = prepared_spikes.trial_spike_trains[0]
    prepared_spikes = replace(
        prepared_spikes,
        trial_spike_trains=(
            dense_train,
            TrialRelativeSpikeTrains(
                unit_id="PFC:2",
                relative_spike_times=tuple(value.copy() for value in dense_train.relative_spike_times),
                overlap_trial_indices=np.empty(0, dtype=np.int64),
            ),
        ),
    )
    observed_calls: list[tuple[int, int]] = []
    edge_sizes: list[int] = []
    event_order: list[str] = []
    shuffle_sizes: list[int] = []
    previous_observed_arrays: tuple[weakref.ReferenceType[np.ndarray], ...] = ()
    previous_edge_arrays: tuple[weakref.ReferenceType[np.ndarray], ...] = ()
    original_observed = (
        ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics
    )
    original_reduce = ppc_runtime.compute_segmented_edge_statistics
    original_consume = ppc_runtime._consume_grouped_edge_block
    original_writer = ppc_runtime.write_ppc_checkpoint

    def recording_observed(**kwargs: object):
        """Record one bounded site/unit observed reduction before delegating."""
        nonlocal previous_observed_arrays
        phase = np.asarray(kwargs["trial_phase_vectors"])
        geometries = tuple(kwargs["source_trial_geometries"])
        observed_calls.append((len(geometries), len(geometries)))
        assert phase.shape[0] >= len(geometries)
        statistics = original_observed(**kwargs)
        previous_observed_arrays = (
            weakref.ref(statistics.phase_trial_index),
            weakref.ref(statistics.phase_vector_sum),
            weakref.ref(statistics.valid_spike_count),
            weakref.ref(statistics.representative_frequency_index),
            weakref.ref(statistics.representative_frequency_hz),
            weakref.ref(statistics.phase_bin_edges_rad),
            weakref.ref(statistics.representative_phase_histogram_count),
        )
        return statistics

    def recording_reduce(**kwargs: object):
        """Record one bounded physical edge reduction before delegating."""
        nonlocal previous_edge_arrays
        # S2 arrays must be released after observed aggregation, before S1 null
        # edges start; completed S1 edge arrays must likewise be gone before
        # the next bounded physical edge is reduced.
        assert all(reference() is None for reference in previous_observed_arrays)
        assert all(reference() is None for reference in previous_edge_arrays)
        source = np.asarray(kwargs["source_trial_index"])
        target = np.asarray(kwargs["target_trial_index"])
        assert source.shape == target.shape
        assert source.size == config.ppc_execution.trial_edge_block_size
        edge_sizes.append(source.size)
        event_order.append("reduce")
        statistics = original_reduce(**kwargs)
        previous_edge_arrays = (
            weakref.ref(statistics.source_trial_index),
            weakref.ref(statistics.target_trial_index),
            weakref.ref(statistics.phase_vector_sum),
            weakref.ref(statistics.valid_spike_count),
        )
        return statistics

    def recording_consume(*args: object, **kwargs: object) -> object:
        """Require immediate bounded shuffle consumption of one reduced edge block."""
        schedule_block = np.asarray(kwargs["schedule_block"])
        assert schedule_block.shape[0] <= config.ppc_execution.shuffle_block_size
        shuffle_sizes.append(schedule_block.shape[0])
        event_order.append("consume")
        job_plan = kwargs["job_plan"]
        if job_plan.epoch_name != "whole":
            return original_consume(*args, **kwargs)

        # Whole segments are the direct before+after sufficient-statistic sum;
        # they cannot create a temporary reduction over a segment axis or a
        # temporary ``before + after`` array.  Four in-place adds are required:
        # before/after for complex sums and before/after for int64 counts.
        original_sum = ppc_runtime.np.sum
        original_add = ppc_runtime.np.add
        whole_add_count = 0

        def forbidden_sum(*_: object, **__: object) -> object:
            """Whole grouped consumption must not materialize an axis reduction."""
            raise AssertionError("whole grouped consume reduced a temporary segment array")

        def guarded_add(
            left: object,
            right: object,
            *args: object,
            **add_kwargs: object,
        ) -> object:
            """Require each whole-segment addition to write a retained accumulator."""
            nonlocal whole_add_count
            out = add_kwargs.get("out")
            output = out[0] if isinstance(out, tuple) else out
            if output is None:
                raise AssertionError("whole grouped consume formed a before+after temporary")
            vector_sum = np.asarray(kwargs["vector_sum"])
            valid_count = np.asarray(kwargs["valid_count"])
            if not (
                np.shares_memory(np.asarray(output), vector_sum)
                or np.shares_memory(np.asarray(output), valid_count)
            ):
                raise AssertionError("whole grouped consume wrote an unaccounted temporary")
            whole_add_count += 1
            return original_add(left, right, *args, **add_kwargs)

        ppc_runtime.np.sum = forbidden_sum
        ppc_runtime.np.add = guarded_add
        try:
            result = original_consume(*args, **kwargs)
            assert whole_add_count >= 4
            return result
        finally:
            ppc_runtime.np.sum = original_sum
            ppc_runtime.np.add = original_add

    def recording_writer(*args: object, **kwargs: object) -> Path:
        """Require every bounded S1/S2 reducer buffer to die before publication."""
        assert all(reference() is None for reference in previous_observed_arrays)
        assert all(reference() is None for reference in previous_edge_arrays)
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        recording_observed,
    )
    monkeypatch.setattr(ppc_runtime, "compute_segmented_edge_statistics", recording_reduce)
    monkeypatch.setattr(ppc_runtime, "_consume_grouped_edge_block", recording_consume)
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", recording_writer)
    grouped = _run_grouped_component(config, prepared_phase, prepared_spikes, tmp_path / "grouped")
    legacy, expected_histogram = _legacy_grouped_reference(
        config, prepared_phase, prepared_spikes, grouped.component_plan, tmp_path / "legacy"
    )
    _assert_legacy_inferential_margins(
        config, prepared_phase, prepared_spikes, grouped.component_plan
    )

    assert set(grouped.summary_arrays) == _GROUPED_SUMMARY_FIELDS
    for job in grouped.component_plan.job_plans:
        reference = legacy[(job.condition_index, job.site_index, job.epoch_index)]
        index = (slice(None), job.condition_index, job.site_index, job.epoch_index, slice(None))
        for name, expected in reference.summary_arrays.items():
            actual = grouped.summary_arrays[name][index]
            if expected.dtype.kind == "f":
                _assert_float_summary_equal(name, actual, expected)
            else:
                np.testing.assert_array_equal(actual, expected)
        if job.epoch_name == "whole" and job.selected_trial_rows.size >= 2:
            expected_schedule = ppc_runtime.generate_trial_derangement_schedule(
                job.selected_trial_rows.size,
                config.ppc.shuffle_count,
                seed=job.schedule_seed,
            )
            np.testing.assert_array_equal(job.schedule, expected_schedule)
    np.testing.assert_array_equal(grouped.summary_arrays["representative_phase_histogram_count"], expected_histogram)
    assert len(observed_calls) == len(config.sites) * len(prepared_spikes.unit_ids)
    assert all(trial_count >= geometry_count > 0 for trial_count, geometry_count in observed_calls)
    assert sum(edge_sizes) == len(prepared_spikes.unit_ids) * grouped.component_plan.union_edge_count
    assert shuffle_sizes and max(shuffle_sizes) == config.ppc_execution.shuffle_block_size
    for event_index, event in enumerate(event_order):
        if event == "reduce":
            assert event_index + 1 < len(event_order)
            assert event_order[event_index + 1] == "consume"


def test_grouped_component_axes_progress_and_safe_block_identity_are_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4 returns immutable component axes and canonical serial progress cycles."""
    config = _grouped_config()
    prepared_phase, prepared_spikes = _grouped_inputs(config)
    events: list[object] = []

    def forbidden_spawn(*_: object, **__: object) -> object:
        """Reject any accidental S4 process-pool construction."""
        raise AssertionError("S4 grouped execution must not spawn workers")

    original_writer = ppc_runtime.write_ppc_checkpoint
    writer_block_ids: list[str] = []

    def recording_writer(
        run_directory: Path,
        block_id: str,
        arrays: Mapping[str, np.ndarray],
        metadata: Mapping[str, object],
        *,
        copy_arrays: bool = True,
    ) -> Path:
        """Require parent publication to transfer bounded frozen arrays directly."""
        assert copy_arrays is False
        assert arrays
        assert all(not np.asarray(values).dtype.hasobject for values in arrays.values())
        assert all(not np.asarray(values).flags.writeable for values in arrays.values())
        writer_block_ids.append(block_id)
        return original_writer(
            run_directory,
            block_id,
            arrays,
            metadata,
            copy_arrays=copy_arrays,
        )

    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden_spawn)
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", recording_writer)
    result = _run_grouped_component(config, prepared_phase, prepared_spikes, tmp_path, progress_callback=events.append)
    metric_shape = (2, 4, 2, 3, 2)
    histogram_shape = metric_shape[:-1] + (2, 2)
    assert set(result.summary_arrays) == _GROUPED_SUMMARY_FIELDS
    for name in ppc_runtime._FLOAT_FIELDS:
        assert result.summary_arrays[name].shape == metric_shape
        assert result.summary_arrays[name].dtype == np.dtype(float)
    for name in ppc_runtime._INTEGER_FIELDS:
        assert result.summary_arrays[name].shape == metric_shape
        assert result.summary_arrays[name].dtype == np.dtype(np.int64)
    for name in ppc_runtime._BOOLEAN_FIELDS:
        assert result.summary_arrays[name].shape == metric_shape
        assert result.summary_arrays[name].dtype == np.dtype(bool)
    assert result.summary_arrays["representative_phase_histogram_count"].shape == histogram_shape
    assert result.summary_arrays["representative_phase_histogram_count"].dtype == np.dtype(np.int64)
    assert not np.shares_memory(result.summary_arrays["ppc"], prepared_phase.phase_tensor)
    assert all(not values.flags.writeable for values in result.summary_arrays.values())

    assert len(result.completed_block_ids) == 4
    assert writer_block_ids == list(result.completed_block_ids)
    assert len(set(result.completed_block_ids)) == len(result.completed_block_ids)
    assert all(Path(block_id).name == block_id and "/" not in block_id and "\\" not in block_id for block_id in result.completed_block_ids)
    metadata = json.loads((result.run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["summary_axes"] == ["unit", "condition", "site", "epoch", "frequency"]
    assert metadata["histogram_axes"] == ["unit", "condition", "site", "epoch", "band", "phase_bin"]
    identities = metadata["block_identities"]
    assert tuple(identities) == result.completed_block_ids
    assert [
        (identities[block_id]["site_id"], identities[block_id]["unit_start"], identities[block_id]["unit_stop"])
        for block_id in result.completed_block_ids
    ] == [
        (config.sites[0].stable_id, 0, 1),
        (config.sites[0].stable_id, 1, 2),
        (config.sites[1].stable_id, 0, 1),
        (config.sites[1].stable_id, 1, 2),
    ]
    checkpoint_keys = set(ppc_runtime._SUMMARY_FIELDS) | {
        "representative_phase_histogram_count",
        "site_index",
        "unit_bounds",
    }
    for block_id in result.completed_block_ids:
        arrays = _checkpoint_arrays(result.run_directory / "blocks" / f"{block_id}.npz")
        assert set(arrays) == checkpoint_keys
        for name in ppc_runtime._FLOAT_FIELDS:
            assert arrays[name].shape == (1, 4, 3, 2)
            assert arrays[name].dtype == np.dtype(float)
        for name in ppc_runtime._INTEGER_FIELDS:
            assert arrays[name].shape == (1, 4, 3, 2)
            assert arrays[name].dtype == np.dtype(np.int64)
        for name in ppc_runtime._BOOLEAN_FIELDS:
            assert arrays[name].shape == (1, 4, 3, 2)
            assert arrays[name].dtype == np.dtype(bool)
        assert arrays["representative_phase_histogram_count"].shape == (1, 4, 3, 2, 2)
        assert arrays["representative_phase_histogram_count"].dtype == np.dtype(np.int64)
        assert arrays["site_index"].shape == (1,)
        assert arrays["site_index"].dtype == np.dtype(np.int64)
        assert arrays["unit_bounds"].shape == (2,)
        assert arrays["unit_bounds"].dtype == np.dtype(np.int64)
        assert arrays["site_index"][0] == identities[block_id]["site_index"]
        np.testing.assert_array_equal(
            arrays["unit_bounds"],
            np.array(
                [identities[block_id]["unit_start"], identities[block_id]["unit_stop"]],
                dtype=np.int64,
            ),
        )
    assert events[0].stage == "prepare_phase"
    assert events[-1].stage == "commit"
    cycle = ("observed_reduction", "trial_edge_reduction", "shuffle_aggregation", "fdr", "checkpoint")
    completed_events = [event for event in events if event.stage in cycle and event.completed_count > 0]
    assert [event.stage for event in completed_events] == list(cycle) * len(result.completed_block_ids)
    for stage in cycle:
        stage_events = [event for event in completed_events if event.stage == stage]
        assert [event.completed_count for event in stage_events] == list(range(1, len(result.completed_block_ids) + 1))
        assert all(event.total_count == len(result.completed_block_ids) for event in stage_events)

    unsafe_site_config = replace(
        config,
        sites=(replace(config.sites[0], stable_id="PFC/unsafe"), config.sites[1]),
        site_pairs=(("PFC/unsafe", config.sites[1].stable_id),),
    )
    unsafe_phase, unsafe_spikes = _grouped_inputs(unsafe_site_config)
    unsafe = _run_grouped_component(unsafe_site_config, unsafe_phase, unsafe_spikes, tmp_path / "unsafe")
    unsafe_metadata = json.loads((unsafe.run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert all("/" not in block_id and "\\" not in block_id for block_id in unsafe.completed_block_ids)
    assert any(identity["site_id"] == "PFC/unsafe" for identity in unsafe_metadata["block_identities"].values())


def test_grouped_component_uses_condition_intersection_and_bypasses_empty_or_ineligible_nulls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Condition/site selections retain stable rows and avoid needless shuffled work."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    result = _run_grouped_component(config, phase, spikes, tmp_path / "normal")
    for site_index in range(len(config.sites)):
        assert np.array_equal(
            next(job for job in result.component_plan.job_plans if job.condition_index == 0 and job.site_index == site_index).selected_trial_rows,
            np.array([41, 59, 83], dtype=np.int64),
        )
        assert np.array_equal(
            next(job for job in result.component_plan.job_plans if job.condition_index == 1 and job.site_index == site_index).selected_trial_rows,
            np.array([59, 83], dtype=np.int64),
        )
        assert next(job for job in result.component_plan.job_plans if job.condition_index == 2 and job.site_index == site_index).selected_trial_rows.size == 0
        assert np.array_equal(
            next(job for job in result.component_plan.job_plans if job.condition_index == 3 and job.site_index == site_index).selected_trial_rows,
            np.array([41], dtype=np.int64),
        )
    whole = result.summary_arrays["null_eligible"][1, 1, 1, 0]
    before = result.summary_arrays["null_eligible"][1, 1, 1, 1]
    after = result.summary_arrays["null_eligible"][1, 1, 1, 2]
    assert whole.all() and not before.any() and not after.any()
    # The sparse late condition has enough spikes only after whole-window
    # before+after aggregation.  Its whole result must remain numerically
    # identical to the legacy job despite both standalone halves being null
    # ineligible.
    legacy, _ = _legacy_grouped_reference(
        config, phase, spikes, result.component_plan, tmp_path / "late-legacy"
    )
    _assert_legacy_inferential_margins(
        config, phase, spikes, result.component_plan
    )
    late_whole = legacy[(1, 1, 0)]
    late_index = (slice(None), 1, 1, 0, slice(None))
    for name, expected in late_whole.summary_arrays.items():
        actual = result.summary_arrays[name][late_index]
        if expected.dtype.kind == "f":
            _assert_float_summary_equal(name, actual, expected)
        else:
            np.testing.assert_array_equal(actual, expected)
    whole_observed = result.summary_arrays["spike_count"][1, 1, 1, 0]
    before_observed = result.summary_arrays["spike_count"][1, 1, 1, 1]
    after_observed = result.summary_arrays["spike_count"][1, 1, 1, 2]
    np.testing.assert_array_equal(whole_observed, before_observed + after_observed)
    assert whole_observed.all()
    for condition_index in (2, 3):
        assert not result.summary_arrays["null_eligible"][:, condition_index].any()
        assert not result.summary_arrays["permutation_count"][:, condition_index].any()
        assert np.isnan(result.summary_arrays["p_value"][:, condition_index]).all()
        assert np.isnan(result.summary_arrays["q_value"][:, condition_index]).all()

    sparse_trains = tuple(
        TrialRelativeSpikeTrains(
            unit_id=train.unit_id,
            relative_spike_times=tuple(np.array([-0.5], dtype=float) for _ in train.relative_spike_times),
            overlap_trial_indices=np.empty(0, dtype=np.int64),
        )
        for train in spikes.trial_spike_trains
    )
    ineligible_spikes = replace(spikes, trial_spike_trains=sparse_trains)

    def forbidden_null_edge(*_: object, **__: object) -> object:
        """Fail if inference-ineligible grouped cells request shuffled reduction."""
        raise AssertionError("ineligible grouped PPC cell sampled shuffled edges")

    monkeypatch.setattr(ppc_runtime, "compute_segmented_edge_statistics", forbidden_null_edge)
    bypassed = _run_grouped_component(config, phase, ineligible_spikes, tmp_path / "ineligible")
    assert not bypassed.summary_arrays["null_eligible"].any()
    assert not bypassed.summary_arrays["permutation_count"].any()
    assert np.isnan(bypassed.summary_arrays["p_value"]).all()


@pytest.mark.parametrize("rejected_execution", [
    lambda config: replace(config.ppc_execution, worker_count=2),
    lambda config: replace(config.ppc_execution, unit_block_size=2),
])
def test_grouped_component_rejects_nonserial_or_mismatched_execution_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rejected_execution: object,
) -> None:
    """S4 accepts only the exact serial config execution before planning or I/O."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)

    def forbidden(*_: object, **__: object) -> object:
        """Reject planning, sampling, checkpointing, or worker construction on bad execution."""
        raise AssertionError("invalid grouped execution reached a side-effect seam")

    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", forbidden)
    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        forbidden,
    )
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", forbidden)
    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden)
    with pytest.raises(ValueError, match="serial|execution"):
        _run_grouped_component(
            config,
            phase,
            spikes,
            tmp_path,
            execution=rejected_execution(config),
        )
    assert not (tmp_path / "ppc").exists()


def test_grouped_component_rejects_matching_nonserial_config_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4 rejects an otherwise matching multi-worker configuration before any work."""
    config = _grouped_config(worker_count=2)
    phase, spikes = _grouped_inputs(config)

    def forbidden(*_: object, **__: object) -> object:
        """Reject a planning, sampling, checkpoint, or process side effect."""
        raise AssertionError("nonserial S4 execution reached a side-effect seam")

    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", forbidden)
    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        forbidden,
    )
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", forbidden)
    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden)
    with pytest.raises(ValueError, match="serial|worker"):
        _run_grouped_component(config, phase, spikes, tmp_path)
    assert not (tmp_path / "ppc").exists()


@pytest.mark.parametrize("limit_name", [
    "maximum_worker_allocation_bytes",
    "maximum_aggregate_allocation_bytes",
])
def test_grouped_component_preflights_unsafe_memory_before_summary_or_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    """Unsafe process or aggregate limits fail before sampling, output, or spawn."""
    config = _grouped_config()
    unsafe_execution = replace(config.ppc_execution, **{limit_name: 1})
    unsafe_config = replace(config, ppc_execution=unsafe_execution)
    phase, spikes = _grouped_inputs(unsafe_config)

    def forbidden(*_: object, **__: object) -> object:
        """Reject summary allocation, phase sampling, checkpoint publication, and spawn."""
        raise AssertionError("unsafe grouped allocation reached execution")

    monkeypatch.setattr(ppc_runtime, "_empty_grouped_summary_arrays", forbidden)
    monkeypatch.setattr(ppc_runtime, "build_source_trial_spike_geometry", forbidden)
    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        forbidden,
    )
    monkeypatch.setattr(ppc_runtime, "compute_segmented_edge_statistics", forbidden)
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", forbidden)
    monkeypatch.setattr(ppc_runtime, "ProcessPoolExecutor", forbidden)
    with pytest.raises(ValueError, match="allocation|parent|aggregate|worker"):
        _run_grouped_component(unsafe_config, phase, spikes, tmp_path)
    assert not (tmp_path / "ppc").exists()


def test_grouped_component_honors_forced_condition_batches_without_changing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deterministic split condition batch preserves full component output/order."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    unbatched = _run_grouped_component(config, phase, spikes, tmp_path / "unbatched")
    original_plan = ppc_runtime.plan_grouped_ppc_component

    def split_plan(**kwargs: object) -> object:
        """Return the ordinary plan with every site split in input condition order."""
        plan = original_plan(**kwargs)
        return replace(plan, condition_batches=tuple(((0,), (1, 2, 3)) for _ in config.sites))

    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", split_plan)
    split = _run_grouped_component(config, phase, spikes, tmp_path / "split")
    assert split.component_plan.condition_batches == tuple(((0,), (1, 2, 3)) for _ in config.sites)
    assert split.completed_block_ids == unbatched.completed_block_ids
    _assert_component_summaries_equal(unbatched, split)


def test_grouped_component_run_identity_and_metadata_bind_real_inputs_and_execution_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run identity binds prepared content, schedule inputs, segments, and execution.

    The production records deliberately have no ad-hoc phase fingerprint fields:
    changing their numerical phase, trial, spike, or condition content must still
    invalidate work identity.  All execution settings are serialized in baseline
    work metadata; executable setting changes receive independent run identities.
    """
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    baseline = _run_grouped_component(config, phase, spikes, tmp_path / "baseline")
    metadata = json.loads((baseline.run_directory / "metadata.json").read_text(encoding="utf-8"))
    expected_execution = {
        "unit_block_size": config.ppc_execution.unit_block_size,
        "shuffle_block_size": config.ppc_execution.shuffle_block_size,
        "trial_edge_block_size": config.ppc_execution.trial_edge_block_size,
        "worker_count": config.ppc_execution.worker_count,
        "maximum_worker_allocation_bytes": config.ppc_execution.maximum_worker_allocation_bytes,
        "maximum_aggregate_allocation_bytes": config.ppc_execution.maximum_aggregate_allocation_bytes,
        "prepared_phase_cache_enabled": config.ppc_execution.prepared_phase_cache_enabled,
        "checkpoint_enabled": config.ppc_execution.checkpoint_enabled,
        "checkpoint_retention": config.ppc_execution.checkpoint_retention,
        "progress_update_interval": config.ppc_execution.progress_update_interval,
    }
    assert metadata["execution_settings"] == expected_execution
    assert isinstance(metadata["grouped_kernel_version"], str) and metadata["grouped_kernel_version"]
    assert isinstance(metadata["grouped_code_version"], str) and metadata["grouped_code_version"]
    fingerprints = {baseline.run_fingerprint}

    changed_rows_phase, changed_rows_spikes = _grouped_inputs(config)
    fingerprints.add(_run_grouped_component(
        config,
        replace(changed_rows_phase, trial_indices=np.array([43, 61, 89, 101, 107, 127], dtype=np.int64)),
        changed_rows_spikes,
        tmp_path / "rows",
    ).run_fingerprint)
    changed_phase, changed_phase_spikes = _grouped_inputs(config)
    phase_values = changed_phase.phase_tensor.copy()
    phase_values[0, 0, 0, 0] *= np.complex64(1.0j)
    fingerprints.add(_run_grouped_component(config, replace(changed_phase, phase_tensor=phase_values), changed_phase_spikes, tmp_path / "phase").run_fingerprint)
    changed_valid_phase, changed_valid_spikes = _grouped_inputs(config)
    phase_valid = changed_valid_phase.phase_valid.copy()
    phase_valid[0, 0, 0, 0] = False
    fingerprints.add(_run_grouped_component(
        config,
        replace(changed_valid_phase, phase_valid=phase_valid),
        changed_valid_spikes,
        tmp_path / "phase-valid",
    ).run_fingerprint)
    changed_spike_phase, changed_spikes = _grouped_inputs(config)
    first_train = changed_spikes.trial_spike_trains[0]
    fingerprints.add(_run_grouped_component(
        config,
        changed_spike_phase,
        replace(changed_spikes, trial_spike_trains=(
            replace(first_train, relative_spike_times=(np.concatenate((first_train.relative_spike_times[0], [0.125])), *first_train.relative_spike_times[1:])),
            changed_spikes.trial_spike_trains[1],
        )),
        tmp_path / "spikes",
    ).run_fingerprint)
    changed_membership_phase, changed_membership_spikes = _grouped_inputs(config)
    membership = changed_membership_phase.prepared_trials.condition_membership.copy()
    membership[0, 1] = True
    fingerprints.add(_run_grouped_component(
        config,
        replace(changed_membership_phase, prepared_trials=replace(changed_membership_phase.prepared_trials, condition_membership=membership)),
        changed_membership_spikes,
        tmp_path / "membership",
    ).run_fingerprint)
    changed_schedule = replace(config, ppc=replace(config.ppc, seed=config.ppc.seed + 1))
    schedule_phase, schedule_spikes = _grouped_inputs(changed_schedule)
    fingerprints.add(_run_grouped_component(changed_schedule, schedule_phase, schedule_spikes, tmp_path / "schedule").run_fingerprint)
    changed_windows = replace(
        config,
        analysis_windows=replace(
            config.analysis_windows,
            before_stop_s=0.25,
            after_start_s=0.25,
        ),
    )
    window_phase, window_spikes = _grouped_inputs(changed_windows)
    fingerprints.add(_run_grouped_component(changed_windows, window_phase, window_spikes, tmp_path / "windows").run_fingerprint)
    changed_site = replace(
        config,
        sites=(replace(config.sites[0], stable_id="PFC-alternate"), config.sites[1]),
        site_pairs=(("PFC-alternate", config.sites[1].stable_id),),
    )
    site_phase, site_spikes = _grouped_inputs(changed_site)
    fingerprints.add(_run_grouped_component(changed_site, site_phase, site_spikes, tmp_path / "site").run_fingerprint)

    executable_changes = (
        {"unit_block_size": 2},
        {"shuffle_block_size": 2},
        {"trial_edge_block_size": 2},
        {"maximum_worker_allocation_bytes": 3 * 1024**3},
        {"maximum_aggregate_allocation_bytes": 11 * 1024**3},
        {"prepared_phase_cache_enabled": False},
        {"checkpoint_enabled": False},
        {"checkpoint_retention": "retain"},
        {"progress_update_interval": 2},
    )
    for change_index, change in enumerate(executable_changes):
        changed_config = replace(config, ppc_execution=replace(config.ppc_execution, **change))
        changed_phase, changed_spikes = _grouped_inputs(changed_config)
        fingerprints.add(_run_grouped_component(changed_config, changed_phase, changed_spikes, tmp_path / f"execution-{change_index}").run_fingerprint)
        assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed_config)

    original_schedule = ppc_runtime.generate_trial_derangement_schedule

    def alternate_valid_schedule(
        trial_count: int,
        shuffle_count: int,
        *,
        seed: int,
    ) -> np.ndarray:
        """Return valid but deliberately different deterministic derangements."""
        generated = original_schedule(trial_count, shuffle_count, seed=seed)
        if trial_count >= 3:
            alternate_row = (np.arange(trial_count, dtype=np.int64) + 2) % trial_count
            return np.broadcast_to(alternate_row, generated.shape).copy()
        return generated

    monkeypatch.setattr(ppc_runtime, "generate_trial_derangement_schedule", alternate_valid_schedule)
    alternate_phase, alternate_spikes = _grouped_inputs(config)
    alternate = _run_grouped_component(config, alternate_phase, alternate_spikes, tmp_path / "alternate-schedule")
    alternate_metadata = json.loads((alternate.run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert alternate_metadata["execution_plan_fingerprint"] != metadata["execution_plan_fingerprint"]
    assert alternate.run_fingerprint != baseline.run_fingerprint
    fingerprints.add(alternate.run_fingerprint)
    monkeypatch.setattr(ppc_runtime, "generate_trial_derangement_schedule", original_schedule)

    original_code_version = ppc_runtime._GROUPED_PPC_CODE_VERSION
    monkeypatch.setattr(ppc_runtime, "_GROUPED_PPC_CODE_VERSION", "test-code-version-change")
    version_phase, version_spikes = _grouped_inputs(config)
    version_changed = _run_grouped_component(config, version_phase, version_spikes, tmp_path / "baseline")
    assert version_changed.run_fingerprint != baseline.run_fingerprint
    assert version_changed.resumed_block_ids == ()
    fingerprints.add(version_changed.run_fingerprint)
    monkeypatch.setattr(ppc_runtime, "_GROUPED_PPC_CODE_VERSION", original_code_version)

    original_kernel_version = ppc_runtime._GROUPED_PPC_KERNEL_VERSION
    monkeypatch.setattr(ppc_runtime, "_GROUPED_PPC_KERNEL_VERSION", "test-kernel-version-change")
    kernel_phase, kernel_spikes = _grouped_inputs(config)
    kernel_changed = _run_grouped_component(config, kernel_phase, kernel_spikes, tmp_path / "baseline")
    kernel_metadata = json.loads((kernel_changed.run_directory / "metadata.json").read_text(encoding="utf-8"))
    assert kernel_metadata["grouped_kernel_version"] == "test-kernel-version-change"
    assert kernel_metadata["execution_plan_fingerprint"] != metadata["execution_plan_fingerprint"]
    assert kernel_changed.run_fingerprint != baseline.run_fingerprint
    assert kernel_changed.resumed_block_ids == ()
    fingerprints.add(kernel_changed.run_fingerprint)
    monkeypatch.setattr(ppc_runtime, "_GROUPED_PPC_KERNEL_VERSION", original_kernel_version)

    assert len(fingerprints) == 12 + len(executable_changes)


def test_grouped_component_resumes_valid_siblings_and_rejects_tampered_blocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only exact grouped block identity/axes/plan checkpoints resume independently."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    cold = _run_grouped_component(config, phase, spikes, tmp_path / "cold")
    warm = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    original_observed = (
        ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics
    )
    original_single_loader = ppc_runtime.load_valid_ppc_checkpoint
    previous_checkpoint_arrays: tuple[weakref.ReferenceType[np.ndarray], ...] = ()

    def forbidden_recompute(*_: object, **__: object) -> object:
        """Fail if a fully valid grouped checkpoint is sampled again."""
        raise AssertionError("valid grouped checkpoint was recomputed")

    def streaming_loader(*args: object, **kwargs: object) -> object:
        """Require prior merged checkpoint arrays to be released before next load."""
        nonlocal previous_checkpoint_arrays
        assert all(reference() is None for reference in previous_checkpoint_arrays)
        checkpoint = original_single_loader(*args, **kwargs)
        if checkpoint is not None:
            assert all(not values.flags.writeable for values in checkpoint.arrays.values())
            previous_checkpoint_arrays = tuple(
                weakref.ref(values) for values in checkpoint.arrays.values()
            )
        return checkpoint

    def forbidden_eager_loader(*_: object, **__: object) -> object:
        """Grouped resume must not eagerly load every valid checkpoint sibling."""
        raise AssertionError("grouped resume used eager checkpoint loading")

    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        forbidden_recompute,
    )
    monkeypatch.setattr(ppc_runtime, "load_valid_ppc_checkpoint", streaming_loader)
    monkeypatch.setattr(ppc_runtime, "load_valid_ppc_checkpoints", forbidden_eager_loader)
    exact = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert exact.resumed_block_ids == exact.completed_block_ids
    _assert_component_summaries_equal(cold, exact)
    assert all(reference() is None for reference in previous_checkpoint_arrays)
    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        original_observed,
    )
    monkeypatch.setattr(ppc_runtime, "load_valid_ppc_checkpoint", original_single_loader)

    warm_metadata = json.loads((warm.run_directory / "metadata.json").read_text(encoding="utf-8"))
    same_bounds = [
        block_id
        for block_id, identity in warm_metadata["block_identities"].items()
        if identity["unit_start"] == 0 and identity["unit_stop"] == 1
    ]
    assert len(same_bounds) == 2
    first_swap, second_swap = same_bounds
    first_path = warm.run_directory / "blocks" / f"{first_swap}.npz"
    second_path = warm.run_directory / "blocks" / f"{second_swap}.npz"
    first_bytes = first_path.read_bytes()
    second_bytes = second_path.read_bytes()
    first_path.write_bytes(second_bytes)
    second_path.write_bytes(first_bytes)
    swapped = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert first_swap not in swapped.resumed_block_ids
    assert second_swap not in swapped.resumed_block_ids
    assert set(warm.completed_block_ids) - {first_swap, second_swap} <= set(swapped.resumed_block_ids)
    _assert_component_summaries_equal(cold, swapped)
    warm = swapped

    corrupt_id = warm.completed_block_ids[0]
    corrupt_path = warm.run_directory / "blocks" / f"{corrupt_id}.npz"
    corrupt_path.write_bytes(b"corrupt")
    np.savez(warm.run_directory / "blocks" / "orphan.npz", ppc=np.zeros(1, dtype=float))
    repaired = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert corrupt_id not in repaired.resumed_block_ids
    assert set(warm.completed_block_ids[1:]) <= set(repaired.resumed_block_ids)
    _assert_component_summaries_equal(cold, repaired)

    tampered_id = repaired.completed_block_ids[-1]
    tampered_path = repaired.run_directory / "blocks" / f"{tampered_id}.npz"
    tampered_arrays = _checkpoint_arrays(tampered_path)
    tampered_arrays["unit_bounds"] = np.array([0, 1], dtype=np.int64)
    np.savez(tampered_path, **tampered_arrays)
    identity_repaired = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert tampered_id not in identity_repaired.resumed_block_ids
    assert set(repaired.completed_block_ids) - {tampered_id} <= set(identity_repaired.resumed_block_ids)
    _assert_component_summaries_equal(cold, identity_repaired)

    wrong_axis_id = identity_repaired.completed_block_ids[0]
    np.savez(identity_repaired.run_directory / "blocks" / f"{wrong_axis_id}.npz", ppc=np.zeros((1,), dtype=float))
    axis_repaired = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert wrong_axis_id not in axis_repaired.resumed_block_ids
    _assert_component_summaries_equal(cold, axis_repaired)

    metadata_path = axis_repaired.run_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "execution_plan_fingerprint" in metadata
    metadata["execution_plan_fingerprint"] = "mismatched-plan"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    mismatched_plan = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert mismatched_plan.resumed_block_ids == ()
    _assert_component_summaries_equal(cold, mismatched_plan)


def test_grouped_component_checkpoint_publication_is_parent_atomic_and_failure_resumes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed parent checkpoint preserves its exact valid sibling for rerun."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    original_write = ppc_runtime.write_ppc_checkpoint
    written: list[str] = []

    def fail_after_one_checkpoint(run_directory: Path, block_id: str, *args: object, **kwargs: object) -> Path:
        """Write one valid parent-owned block and fail before the next publication."""
        if written:
            raise RuntimeError("injected grouped checkpoint failure")
        written.append(block_id)
        return original_write(run_directory, block_id, *args, **kwargs)

    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", fail_after_one_checkpoint)
    with pytest.raises(RuntimeError, match="injected grouped checkpoint failure"):
        _run_grouped_component(config, phase, spikes, tmp_path)
    run_directories = tuple((tmp_path / "ppc").iterdir())
    assert len(run_directories) == 1
    run_directory = run_directories[0]
    assert not (run_directory / "executor.lock").exists()
    assert not (run_directory / "complete.json").exists()
    assert not (run_directory / "failed.json").exists()
    assert not (tmp_path / "spike_phase.npz").exists()
    assert not (tmp_path / "manifest.json").exists()
    metadata = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    valid_siblings = work_cache.load_valid_ppc_checkpoints(run_directory, metadata)
    assert [checkpoint.block_id for checkpoint in valid_siblings] == written

    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", original_write)
    cold_phase, cold_spikes = _grouped_inputs(config)
    cold = _run_grouped_component(config, cold_phase, cold_spikes, tmp_path / "cold")
    resumed = _run_grouped_component(config, phase, spikes, tmp_path)
    assert resumed.resumed_block_ids == tuple(written)
    _assert_component_summaries_equal(cold, resumed)
    assert not (resumed.run_directory / "executor.lock").exists()


def test_grouped_component_passes_full_phase_views_to_selection_aware_reducers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4 never gathers selected phase/valid trial rows before S1 or S2 reduction.

    The selection-aware observed reducer receives complete site-local phase and
    validity views plus explicit full-axis positions.  The existing S1 reducer
    already permits a selected geometry tuple with complete phase views, so its
    input must also share prepared storage.  No call order is prescribed.
    """
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    original_observed = (
        ppc_runtime.compute_selected_observed_trial_segmented_ppc_statistics
    )
    original_edge = ppc_runtime.compute_segmented_edge_statistics
    original_plan = ppc_runtime.plan_grouped_ppc_component
    observed_calls: list[np.ndarray] = []
    edge_calls: list[np.ndarray] = []
    component_plans: list[object] = []

    def recording_plan(**kwargs: object) -> object:
        """Retain the immutable planner union only for source/target view checks."""
        plan = original_plan(**kwargs)
        component_plans.append(plan)
        return plan

    def recording_observed(**kwargs: object) -> object:
        """Require one full site-local view and explicit selected full positions."""
        phase_view = np.asarray(kwargs["trial_phase_vectors"])
        valid_view = np.asarray(kwargs["phase_valid_mask"])
        phase_trial_index = np.asarray(kwargs["phase_trial_index"])
        geometries = tuple(kwargs["source_trial_geometries"])
        assert np.shares_memory(phase_view, phase.phase_tensor)
        assert np.shares_memory(valid_view, phase.phase_valid)
        assert np.shares_memory(phase_trial_index, phase.trial_indices)
        assert phase_view.shape == valid_view.shape == (
            phase.phase_tensor.shape[2],
            phase.phase_tensor.shape[1],
            phase.phase_tensor.shape[3],
        )
        assert phase_trial_index.shape == (phase.trial_indices.size,)
        assert geometries
        observed_calls.append(
            np.asarray(
                [geometry.source_trial_index for geometry in geometries],
                dtype=np.int64,
            )
        )
        return original_observed(**kwargs)

    def recording_edge(**kwargs: object) -> object:
        """Require S1 target phase and validity to remain complete storage views."""
        phase_view = np.asarray(kwargs["trial_phase_vectors"])
        valid_view = np.asarray(kwargs["phase_valid_mask"])
        phase_trial_index = np.asarray(kwargs["phase_trial_index"])
        source_view = np.asarray(kwargs["source_trial_index"])
        target_view = np.asarray(kwargs["target_trial_index"])
        assert np.shares_memory(phase_view, phase.phase_tensor)
        assert np.shares_memory(valid_view, phase.phase_valid)
        assert np.shares_memory(phase_trial_index, phase.trial_indices)
        assert phase_view.shape == valid_view.shape == (
            phase.phase_tensor.shape[2],
            phase.phase_tensor.shape[1],
            phase.phase_tensor.shape[3],
        )
        assert phase_trial_index.shape == (phase.trial_indices.size,)
        assert component_plans
        assert np.shares_memory(
            source_view,
            component_plans[0].stable_edge_source_trial_row,
        )
        assert np.shares_memory(
            target_view,
            component_plans[0].stable_edge_target_trial_row,
        )
        edge_calls.append(np.asarray(kwargs["source_trial_index"], dtype=np.int64).copy())
        return original_edge(**kwargs)

    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", recording_plan)
    monkeypatch.setattr(
        ppc_runtime,
        "compute_selected_observed_trial_segmented_ppc_statistics",
        recording_observed,
    )
    monkeypatch.setattr(ppc_runtime, "compute_segmented_edge_statistics", recording_edge)
    _run_grouped_component(config, phase, spikes, tmp_path)
    assert observed_calls and edge_calls
    assert all(
        np.all(np.isin(source_ids, phase.trial_indices))
        for source_ids in observed_calls
    )


def test_grouped_component_condition_batches_bound_active_null_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forced deterministic batches retain only that batch's null accumulators."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    unbatched = _run_grouped_component(config, phase, spikes, tmp_path / "unbatched")
    original_plan = ppc_runtime.plan_grouped_ppc_component
    original_batch = ppc_runtime._execute_grouped_condition_batch
    original_consume = ppc_runtime._consume_grouped_edge_block
    original_finalizer = ppc_runtime._finalize_grouped_null_job
    active_conditions: tuple[int, ...] = ()
    batch_calls: list[tuple[int, ...]] = []
    consumed_conditions: list[int] = []
    finalized_conditions: list[int] = []
    released_batch_arrays: list[weakref.ReferenceType[np.ndarray]] = []

    def split_plan(**kwargs: object) -> object:
        """Force two input-order batches for every site without changing schedules."""
        plan = original_plan(**kwargs)
        return replace(plan, condition_batches=tuple(((0,), (1, 2, 3)) for _ in config.sites))

    def recording_batch(*args: object, **kwargs: object) -> object:
        """Expose exactly one live planner batch around its bounded null work."""
        nonlocal active_conditions
        gc.collect()
        # A completed batch cannot retain complete shuffle draw/accumulator
        # arrays while the next condition batch begins.
        assert all(reference() is None for reference in released_batch_arrays)
        assert active_conditions == ()
        active_conditions = tuple(kwargs["condition_indices"])
        batch_calls.append(active_conditions)
        try:
            return original_batch(*args, **kwargs)
        finally:
            active_conditions = ()
            gc.collect()
            assert all(reference() is None for reference in released_batch_arrays)

    def recording_consume(*args: object, **kwargs: object) -> object:
        """Every edge consume must belong to the single currently live batch."""
        job = kwargs["job_plan"]
        assert job.condition_index in active_conditions
        consumed_conditions.append(job.condition_index)
        return original_consume(*args, **kwargs)

    def recording_finalizer(*args: object, **kwargs: object) -> object:
        """Capture only weak refs to arrays that must die with this batch."""
        job = kwargs["job_plan"]
        assert job.condition_index in active_conditions
        finalized_conditions.append(job.condition_index)
        accumulator = np.asarray(kwargs["vector_sum"])
        valid_count = np.asarray(kwargs["valid_count"])
        draw_scratch = np.asarray(kwargs["draw_scratch"])
        released_batch_arrays.extend(
            (weakref.ref(accumulator), weakref.ref(valid_count), weakref.ref(draw_scratch))
        )
        return original_finalizer(*args, **kwargs)

    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", split_plan)
    monkeypatch.setattr(ppc_runtime, "_execute_grouped_condition_batch", recording_batch)
    monkeypatch.setattr(ppc_runtime, "_consume_grouped_edge_block", recording_consume)
    monkeypatch.setattr(ppc_runtime, "_finalize_grouped_null_job", recording_finalizer)
    split = _run_grouped_component(config, phase, spikes, tmp_path / "split")
    assert batch_calls == [((0,), (1, 2, 3))[index] for _ in split.completed_block_ids for index in range(2)]
    assert consumed_conditions and finalized_conditions
    assert set(consumed_conditions) <= {0, 1}
    assert set(finalized_conditions) <= {0, 1}
    _assert_component_summaries_equal(unbatched, split)


@pytest.mark.parametrize("shuffle_count", (3, 4))
def test_grouped_null_finalizer_uses_one_scratch_without_legacy_draw_copy(
    monkeypatch: pytest.MonkeyPatch,
    shuffle_count: int,
) -> None:
    """Finalize predefined mixed draws exactly without a copied draw matrix.

    The first three/four retained accumulator rows yield PPC draws
    ``[-1, NaN, 0]``/``[-1, NaN, 0, 1]``.  This is deliberately a direct
    finalizer contract: S1 sufficient-statistic reductions can differ from the
    legacy concatenate-and-sum path by a few binary64 ulps, while the summary
    operation itself must reproduce the legacy finite-draw decision exactly.
    """
    complete_sums = np.array([0.0j, 0.0j, 1.0 + 1.0j, 2.0 + 0.0j]).reshape(4, 1, 1)
    complete_counts = np.array([2, 1, 2, 2], dtype=np.int64).reshape(4, 1, 1)
    vector_sum = complete_sums[:shuffle_count].copy()
    valid_count = complete_counts[:shuffle_count].copy()
    observed_ppc = np.array([[0.5]], dtype=float)
    legacy_draws = np.array([-1.0, np.nan, 0.0, 1.0], dtype=float)[:shuffle_count]
    expected = spike_lfp_summary.summarize_permutation_null(
        observed_ppc=observed_ppc,
        null_ppc_chunks=(legacy_draws.reshape(shuffle_count, 1, 1),),
        spike_count=np.array([[50]], dtype=np.int64),
        eligible_trial_count=np.array([[2]], dtype=np.int64),
    )
    output_arrays = {
        "null_exceedance_count": np.zeros((1, 1), dtype=np.int64),
        "permutation_count": np.zeros((1, 1), dtype=np.int64),
        "p_value": np.full((1, 1), np.nan, dtype=float),
        "null_mean": np.full((1, 1), np.nan, dtype=float),
        "null_std": np.full((1, 1), np.nan, dtype=float),
        "null_p025": np.full((1, 1), np.nan, dtype=float),
        "null_p50": np.full((1, 1), np.nan, dtype=float),
        "null_p975": np.full((1, 1), np.nan, dtype=float),
        "null_eligible": np.zeros((1, 1), dtype=bool),
    }
    draw_scratch = np.empty_like(valid_count, dtype=float)
    job_plan = _direct_job_plan()
    eligibility = np.array([[True]], dtype=bool)
    expected_scratch = (
        np.array([-1.0, 0.0, np.nan], dtype=float)
        if shuffle_count == 3
        else np.array([-1.0, 0.0, 1.0, np.nan], dtype=float)
    )
    original_copy = ppc_runtime.np.copy
    original_array = ppc_runtime.np.array
    original_copyto = ppc_runtime.np.copyto
    original_sort = ppc_runtime.np.sort
    original_partition = ppc_runtime.np.partition
    original_percentile = ppc_runtime.np.percentile
    original_nanpercentile = ppc_runtime.np.nanpercentile
    original_quantile = ppc_runtime.np.quantile
    original_nanquantile = ppc_runtime.np.nanquantile
    original_isfinite = ppc_runtime.np.isfinite
    original_isnan = ppc_runtime.np.isnan
    original_concatenate = ppc_runtime.np.concatenate

    def forbidden(*_: object, **__: object) -> object:
        """Reject a top-level full-draw copying or order-statistic helper."""
        raise AssertionError("grouped finalization copied or materialized null draws")

    def forbid_full_mask(function: object) -> object:
        """Permit scalar finite checks while rejecting a full Boolean draw mask."""
        def guarded(values: object, *args: object, **kwargs: object) -> object:
            """Delegate scalar checks after enforcing the direct scratch contract."""
            if np.asarray(values).size > 1:
                raise AssertionError("grouped finalization materialized a full null mask")
            return function(values, *args, **kwargs)

        return guarded

    monkeypatch.setattr(ppc_runtime.np, "copy", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "array", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "copyto", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "sort", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "partition", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "percentile", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "nanpercentile", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "quantile", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "nanquantile", forbidden)
    monkeypatch.setattr(ppc_runtime.np, "isfinite", forbid_full_mask(original_isfinite))
    monkeypatch.setattr(ppc_runtime.np, "isnan", forbid_full_mask(original_isnan))
    monkeypatch.setattr(ppc_runtime.np, "concatenate", forbidden)
    ppc_runtime._finalize_grouped_null_job(
        job_plan=job_plan,
        vector_sum=vector_sum,
        valid_count=valid_count,
        draw_scratch=draw_scratch,
        eligible_cell_mask=eligibility,
        observed_ppc=observed_ppc,
        output_arrays=output_arrays,
    )
    monkeypatch.undo()
    assert draw_scratch.dtype == np.dtype(float)
    assert draw_scratch.shape == vector_sum.shape
    assert draw_scratch.nbytes == vector_sum.size * 8
    np.testing.assert_allclose(
        draw_scratch[:, 0, 0],
        expected_scratch,
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    for name in (
        "null_exceedance_count",
        "permutation_count",
        "p_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
        "null_eligible",
    ):
        expected_values = getattr(expected, name)
        actual_values = output_arrays[name]
        if actual_values.dtype.kind == "f":
            np.testing.assert_allclose(actual_values, expected_values, rtol=0.0, atol=0.0)
        else:
            np.testing.assert_array_equal(actual_values, expected_values)


def test_grouped_component_only_accumulates_and_finalizes_eligible_unit_frequency_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed active job excludes ineligible unit-frequency null cells entirely."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    phase_valid = phase.phase_valid.copy()
    phase_valid[:, 1] = False
    mixed_phase = replace(phase, phase_valid=phase_valid)
    original_consume = ppc_runtime._consume_grouped_edge_block
    original_finalizer = ppc_runtime._finalize_grouped_null_job
    consumed_masks: list[np.ndarray] = []
    finalized_masks: list[np.ndarray] = []

    def recording_consume(*args: object, **kwargs: object) -> object:
        """Use the summary-owned Boolean eligibility view, not an int64 map."""
        cells = np.asarray(kwargs["eligible_cell_mask"])
        vector_sum = np.asarray(kwargs["vector_sum"])
        assert cells.dtype == np.dtype(bool)
        assert cells.shape == (1, 2)
        assert np.array_equal(cells, np.array([[True, False]], dtype=bool))
        assert vector_sum.shape == (config.ppc.shuffle_count, 1, 2)
        assert np.shares_memory(cells, result_summary_arrays["null_eligible"])
        consumed_masks.append(cells)
        result = original_consume(*args, **kwargs)
        # No edge contribution may accumulate in a false eligibility cell.
        assert np.all(vector_sum[:, ~cells] == 0.0j)
        assert np.all(np.asarray(kwargs["valid_count"])[:, ~cells] == 0)
        return result

    def recording_finalizer(*args: object, **kwargs: object) -> object:
        """Finalize exactly the same selected null metric cells as consumption."""
        cells = np.asarray(kwargs["eligible_cell_mask"])
        assert cells.dtype == np.dtype(bool)
        assert np.array_equal(cells, np.array([[True, False]], dtype=bool))
        assert np.shares_memory(cells, result_summary_arrays["null_eligible"])
        finalized_masks.append(cells)
        return original_finalizer(*args, **kwargs)

    result_summary_arrays: dict[str, np.ndarray] = {}

    def recording_summary(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        """Retain only output arrays that own the Boolean eligibility storage."""
        values = original_summary(*args, **kwargs)
        result_summary_arrays.update(values)
        return values

    original_summary = ppc_runtime._empty_grouped_summary_arrays
    monkeypatch.setattr(ppc_runtime, "_consume_grouped_edge_block", recording_consume)
    monkeypatch.setattr(ppc_runtime, "_finalize_grouped_null_job", recording_finalizer)
    monkeypatch.setattr(ppc_runtime, "_empty_grouped_summary_arrays", recording_summary)
    result = _run_grouped_component(config, mixed_phase, spikes, tmp_path)
    assert consumed_masks and finalized_masks
    assert not result.summary_arrays["null_eligible"][..., 1].any()
    assert not result.summary_arrays["permutation_count"][..., 1].any()
    assert not result.summary_arrays["null_exceedance_count"][..., 1].any()
    assert np.isnan(result.summary_arrays["p_value"][..., 1]).all()
    assert np.isnan(result.summary_arrays["q_value"][..., 1]).all()
    assert np.isnan(result.summary_arrays["null_mean"][..., 1]).all()


def test_grouped_component_removes_complete_marker_before_failed_corrupt_block_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed repair cannot leave a stale run-level completion certification."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    complete = _run_grouped_component(config, phase, spikes, tmp_path)
    corrupt_id = complete.completed_block_ids[0]
    corrupt_path = complete.run_directory / "blocks" / f"{corrupt_id}.npz"
    corrupt_path.write_bytes(b"corrupt")
    sibling_ids = complete.completed_block_ids[1:]
    sibling_bytes = {
        block_id: (
            (complete.run_directory / "blocks" / f"{block_id}.npz").read_bytes(),
            (complete.run_directory / "blocks" / f"{block_id}.complete.json").read_bytes(),
        )
        for block_id in sibling_ids
    }
    original_writer = ppc_runtime.write_ppc_checkpoint

    def fail_repair(
        run_directory: Path,
        block_id: str,
        *args: object,
        **kwargs: object,
    ) -> Path:
        """Fail only the corrupt block's replacement publication."""
        if block_id == corrupt_id:
            raise RuntimeError("injected corrupt-block repair failure")
        return original_writer(run_directory, block_id, *args, **kwargs)

    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", fail_repair)
    with pytest.raises(RuntimeError, match="corrupt-block repair failure"):
        _run_grouped_component(config, phase, spikes, tmp_path)
    assert not (complete.run_directory / "complete.json").exists()
    assert not (complete.run_directory / "executor.lock").exists()
    for block_id, expected in sibling_bytes.items():
        assert (
            (complete.run_directory / "blocks" / f"{block_id}.npz").read_bytes(),
            (complete.run_directory / "blocks" / f"{block_id}.complete.json").read_bytes(),
        ) == expected


def test_grouped_component_releases_gated_membership_and_full_source_counts_after_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S4 deletes planner-only full tables before parent summary allocation."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    original_membership = lfp_summary_runtime._analysis_condition_membership
    original_counts = ppc_runtime._grouped_source_trial_spike_counts
    original_summary = ppc_runtime._empty_grouped_summary_arrays
    original_plan = ppc_runtime.plan_grouped_ppc_component
    memberships: list[weakref.ReferenceType[np.ndarray]] = []
    counts: list[weakref.ReferenceType[np.ndarray]] = []
    planning_estimates: list[object] = []

    def recording_membership(*args: object, **kwargs: object) -> np.ndarray:
        """Keep only a weak reference to the executor-created gated mask."""
        values = original_membership(*args, **kwargs)
        memberships.append(weakref.ref(values))
        return values

    def recording_counts(*args: object, **kwargs: object) -> np.ndarray:
        """Keep only a weak reference to the full trial/unit count table."""
        values = original_counts(*args, **kwargs)
        counts.append(weakref.ref(values))
        return values

    def recording_summary(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        """Require both planning tables to be unreachable before output allocation."""
        gc.collect()
        assert memberships and counts
        assert all(reference() is None for reference in memberships)
        assert all(reference() is None for reference in counts)
        return original_summary(*args, **kwargs)

    def recording_plan(**kwargs: object) -> object:
        """Record the planner-derived construction peak without caller input."""
        assert "planning_working_bytes" not in kwargs
        plan = original_plan(**kwargs)
        planning_estimates.append(plan.allocation_estimate)
        return plan

    monkeypatch.setattr(
        lfp_summary_runtime,
        "_analysis_condition_membership",
        recording_membership,
    )
    monkeypatch.setattr(
        ppc_runtime,
        "_grouped_source_trial_spike_counts",
        recording_counts,
    )
    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", recording_plan)
    monkeypatch.setattr(ppc_runtime, "_empty_grouped_summary_arrays", recording_summary)
    _run_grouped_component(config, phase, spikes, tmp_path)
    assert len(planning_estimates) == 1
    estimate = planning_estimates[0]
    selected_row_cells = 2 * 3 * (3 + 2 + 0 + 1)
    schedule_cells = 2 * 3 * config.ppc.shuffle_count * (3 + 2)
    D = 8 * selected_row_cells + 8 * schedule_cells
    T = phase.trial_indices.size
    Emax = T * (T - 1)
    A = 8 * (2 * T + Emax) + 16 * T * config.ppc_execution.unit_block_size
    assert estimate.planning_working_bytes == D + A


def test_grouped_component_passes_exact_bounded_schema_to_single_checkpoint_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Streaming resume gives the cache loader exact block axes and byte bound."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    cold = _run_grouped_component(config, phase, spikes, tmp_path)
    original_loader = ppc_runtime.load_valid_ppc_checkpoint
    schemas: list[Mapping[str, object]] = []
    byte_bounds: list[int] = []

    def bounded_loader(*args: object, **kwargs: object) -> object:
        """Require every resumed site/unit block to declare its full NPZ contract."""
        schema = kwargs["expected_array_schema"]
        schemas.append(schema)
        byte_bounds.append(kwargs["maximum_array_bytes"])
        return original_loader(*args, **kwargs)

    monkeypatch.setattr(ppc_runtime, "load_valid_ppc_checkpoint", bounded_loader)
    warm = _run_grouped_component(config, phase, spikes, tmp_path)
    expected_metric_shape = (1, 4, 3, 2)
    expected_histogram_shape = (1, 4, 3, 2, 2)
    expected_schema = {
        **{
            name: (np.dtype(float), expected_metric_shape)
            for name in ppc_runtime._FLOAT_FIELDS
        },
        **{
            name: (np.dtype(np.int64), expected_metric_shape)
            for name in ppc_runtime._INTEGER_FIELDS
        },
        **{
            name: (np.dtype(bool), expected_metric_shape)
            for name in ppc_runtime._BOOLEAN_FIELDS
        },
        "representative_phase_histogram_count": (
            np.dtype(np.int64),
            expected_histogram_shape,
        ),
        "site_index": (np.dtype(np.int64), (1,)),
        "unit_bounds": (np.dtype(np.int64), (2,)),
    }
    assert schemas == [expected_schema] * len(cold.completed_block_ids)
    assert byte_bounds == [
        cold.component_plan.allocation_estimate.checkpoint_block_bytes
    ] * len(cold.completed_block_ids)
    assert warm.resumed_block_ids == cold.completed_block_ids


def test_grouped_component_recomputes_after_non_utf8_run_metadata(
    tmp_path: Path,
) -> None:
    """Malformed text metadata is an invalid cache state, never an execution error."""
    config = _grouped_config()
    phase, spikes = _grouped_inputs(config)
    cold = _run_grouped_component(config, phase, spikes, tmp_path / "cold")
    warm = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    (warm.run_directory / "metadata.json").write_bytes(b"\xff\xfe\x80")
    repaired = _run_grouped_component(config, phase, spikes, tmp_path / "warm")
    assert repaired.resumed_block_ids == ()
    _assert_component_summaries_equal(cold, repaired)


@pytest.mark.parametrize(
    "limit_name",
    ("maximum_worker_allocation_bytes", "maximum_aggregate_allocation_bytes"),
)
def test_grouped_component_preflights_full_scalar_construction_scratch_before_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
) -> None:
    """Scalar preflight rejects map scratch before membership/count/schedule arrays.

    The complete conservative planning bound is ``M + Q + D + A``. Sparse
    selection over the 100-trial input makes it larger than ``M + Q + P``.
    The selected limit sits between those two values, so the executor must
    derive selected job sizes from scalar masks before invoking the
    array-producing membership, count, or schedule seams.
    """
    config = _grouped_config()
    phase, spikes = _sparse_large_trial_grouped_inputs(config)
    membership_bytes = phase.trial_indices.size * len(
        phase.prepared_trials.condition_names
    )
    source_count_bytes = phase.trial_indices.size * len(spikes.unit_ids) * 2 * 8
    selected_per_site_epoch = 2 + 2 + 0 + 1
    scheduled_per_site_epoch = 2 + 2 + 0
    selected_row_cells = len(config.sites) * 3 * selected_per_site_epoch
    schedule_cells = (
        len(config.sites)
        * 3
        * config.ppc.shuffle_count
        * scheduled_per_site_epoch
    )
    D = 8 * selected_row_cells + 8 * schedule_cells
    trial_count = phase.trial_indices.size
    Emax = trial_count * (trial_count - 1)
    A = (
        8 * (2 * trial_count + Emax)
        + 16 * trial_count * config.ppc_execution.unit_block_size
    )
    planning_probe = _plan(
        config,
        **_planner_inputs(
            stable_trial_rows=phase.trial_indices,
            condition_names=phase.prepared_trials.condition_names,
            condition_membership=phase.prepared_trials.condition_membership,
            site_ids=tuple(site.stable_id for site in config.sites),
            site_trial_valid=phase.site_valid,
            source_trial_spike_count=np.zeros(
                (trial_count, len(spikes.unit_ids), 2), dtype=np.int64
            ),
            frequency_count=phase.phase_tensor.shape[1],
            shared_phase_mmap_bytes=phase.phase_tensor.nbytes + phase.phase_valid.nbytes,
        ),
    )
    planner_bytes = planning_probe.allocation_estimate.planner_array_bytes
    shared_phase_bytes = phase.phase_tensor.nbytes + phase.phase_valid.nbytes
    construction_bytes = membership_bytes + source_count_bytes + D + A
    limit_bytes = construction_bytes - 1
    assert membership_bytes + source_count_bytes + planner_bytes <= limit_bytes
    assert limit_bytes < construction_bytes
    if limit_name == "maximum_aggregate_allocation_bytes":
        limit_bytes += shared_phase_bytes
    unsafe_execution = replace(
        config.ppc_execution,
        **{limit_name: limit_bytes},
    )
    unsafe_config = replace(config, ppc_execution=unsafe_execution)

    def forbidden(*_: object, **__: object) -> object:
        """Construction/allocation must not occur after a dimension-only rejection."""
        raise AssertionError("unsafe planning table was allocated")

    monkeypatch.setattr(lfp_summary_runtime, "_analysis_condition_membership", forbidden)
    monkeypatch.setattr(ppc_runtime, "_grouped_source_trial_spike_counts", forbidden)
    monkeypatch.setattr(ppc_runtime, "generate_trial_derangement_schedule", forbidden)
    monkeypatch.setattr(ppc_runtime, "plan_grouped_ppc_component", forbidden)
    monkeypatch.setattr(ppc_runtime, "_empty_grouped_summary_arrays", forbidden)
    monkeypatch.setattr(ppc_runtime, "write_ppc_checkpoint", forbidden)
    with pytest.raises(ValueError, match="planning|membership|source.*count|allocation"):
        _run_grouped_component(unsafe_config, phase, spikes, tmp_path)
    assert not (tmp_path / "ppc").exists()
