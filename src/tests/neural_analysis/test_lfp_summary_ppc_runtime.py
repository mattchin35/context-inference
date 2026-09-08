"""RED contracts for serial PPC checkpoint/restart execution and progress."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_ppc_runtime as ppc_runtime
from src.neural_analysis import spike_lfp_summary
from src.neural_analysis.lfp_summary_models import (
    PPCExecutionConfig,
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
