"""RED contracts for serial PPC checkpoint/restart execution and progress."""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError, fields, replace
import inspect
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ppc_runtime as ppc_runtime
from src.neural_analysis import lfp_summary_runtime
from src.neural_analysis import spike_lfp_summary
from src.neural_analysis.lfp_summary_models import (
    PPCExecutionConfig,
    component_fingerprint,
    default_lfp_summary_config,
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
        "planner_array_bytes",
        "summary_assembly_bytes",
        "worker_plan_bytes",
        "worker_summary_bytes",
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
    )
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in allocation_signature.parameters.values()
    )


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
        maximum_aggregate_allocation_bytes=100_000,
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
    assert plan.allocation_estimate.planned_aggregate_array_bytes == 91_264


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
    assert plan.allocation_estimate.planned_computation_private_bytes == 616
    assert plan.allocation_estimate.planned_parent_private_bytes == 288 + 1332
    assert plan.allocation_estimate.planned_worker_private_bytes == 288 + 888 + 616
    assert plan.allocation_estimate.active_worker_count == 2
    assert plan.allocation_estimate.planned_aggregate_array_bytes == 4096 + 1620 + 2 * 1792


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
    assert estimate.job_accumulator_bytes == 240

    # The selected peak is site B's final one-unit block. Its last edge block
    # has two site-qualified high-spike source edges, unlike site A or block 0.
    assert estimate.geometry_bytes == 52_200
    assert estimate.observed_trial_statistics_bytes == 416
    assert estimate.observed_gather_temporary_bytes == 102_204
    assert estimate.kernel_working_bytes == 204_128
    assert estimate.planned_computation_private_bytes == 256_568
    assert estimate.planned_parent_private_bytes == 6_720
    assert estimate.planned_worker_private_bytes == 258_320
    assert estimate.active_worker_count == 2
    assert estimate.planned_aggregate_array_bytes == 523_360


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
    computation_bytes = max(observed_stage_bytes, null_stage_bytes)
    worker_bytes = 480 + worker_summary_bytes + computation_bytes
    parent_bytes = 480 + component_summary_bytes

    assert estimate.job_accumulator_bytes == job_bytes
    assert estimate.observed_trial_statistics_bytes == observed_bytes
    assert estimate.observed_gather_temporary_bytes == observed_gather_bytes
    assert estimate.kernel_working_bytes == kernel_bytes
    assert estimate.geometry_bytes == geometry_bytes
    assert estimate.planner_array_bytes == 480
    assert estimate.summary_assembly_bytes == component_summary_bytes
    assert estimate.worker_plan_bytes == 480
    assert estimate.worker_summary_bytes == worker_summary_bytes
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
    assert serial.planned_computation_private_bytes == 360
    assert serial.planned_parent_private_bytes == 100 + 3 * (116 + 2 * 2 * 8) + 360
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
    assert estimate.planned_parent_private_bytes == 100 + 276
    assert estimate.planned_worker_private_bytes == 276 + observed_stage_bytes
    assert estimate.active_worker_count == 1
    assert estimate.planned_aggregate_array_bytes == 100 + 276 + 276 + observed_stage_bytes

    planning_dominant = ppc_runtime.estimate_grouped_ppc_allocation(
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
        planner_array_bytes=1000,
        worker_plan_bytes=0,
        worker_count=2,
        pending_unit_block_count=1,
        shared_phase_mmap_bytes=0,
    )
    assert planning_dominant.planned_parent_private_bytes == 1000 + 276
    assert planning_dominant.planned_aggregate_array_bytes == 1000 + 276 + 276 + observed_stage_bytes


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
    assert estimate.planned_parent_private_bytes == 100 + summary_bytes
    assert estimate.planned_worker_private_bytes == 40 + summary_bytes + computation_bytes


@pytest.mark.parametrize(
    "overrides",
    (
        {"planner_array_bytes": np.iinfo(np.int64).max},
        {"worker_plan_bytes": np.iinfo(np.int64).max},
        {"shared_phase_mmap_bytes": np.iinfo(np.int64).max},
        {
            "total_unit_count": 2,
            "worker_count": 2,
            "pending_unit_block_count": 2,
            "worker_plan_bytes": np.iinfo(np.int64).max // 2,
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
    }
    with pytest.raises(ValueError):
        ppc_runtime.estimate_grouped_ppc_allocation(**(arguments | overrides))


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
        planner_array_bytes=0,
        summary_assembly_bytes=0,
        worker_plan_bytes=0,
        worker_summary_bytes=0,
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
