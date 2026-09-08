"""RED contracts for serial PPC checkpoint/restart execution and progress."""

from __future__ import annotations

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
    phase = np.ones((1, 1, 2, 2), dtype=np.complex64)
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
    assert result.final_payload_arrays is None
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
        "null_exceedance_count", "permutation_count", "p_value", "q_value",
        "significant", "null_mean", "null_std", "null_p025", "null_p50", "null_p975",
    }
    assert required <= set(result.summary_arrays)
    assert not hasattr(result, "null_ppc")
    assert all(array.ndim <= 2 for array in result.summary_arrays.values())


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
