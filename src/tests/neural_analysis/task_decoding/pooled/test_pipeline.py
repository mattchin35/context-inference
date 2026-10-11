"""Synchronous pooled computation, assembly, and resume contracts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.neural_analysis.task_decoding.pooled import config as pooled_config
from src.neural_analysis.task_decoding.pooled import decoding, pipeline, results
from src.tests.neural_analysis.task_decoding.pooled.test_config import (
    canonical_edges,
    write_configuration_tree,
)
from src.tests.neural_analysis.task_decoding.pooled.test_results import make_result_template


def _empty_cell(target_identifier: str) -> decoding.PooledWindowTargetResult:
    """Return one target-outer-unavailable checkpoint result.

    Parameters
    ----------
    target_identifier : str
        Stable configured target name. The runtime result contains no fitted
        estimators, transforms, evaluations, or physical-unit conversions.

    Returns
    -------
    decoding.PooledWindowTargetResult
        Empty dense-cell precursor with explicit outer-split unavailability.
    """
    return decoding.PooledWindowTargetResult(
        models={},
        evaluations={},
        gaps={},
        transforms={},
        unavailable_reason="target_outer_unavailable",
        target_identifier=target_identifier,
        target_family=(
            "categorical" if target_identifier == "current_action" else "numerical"
        ),
        outer_fold_ids=np.full(4, -1, dtype=int),
    )


def _context_for_config(
    config: pooled_config.PooledDecodingConfig,
    windows: tuple[pooled_config.TrainingWindow, ...],
) -> pipeline.PooledComputationContext:
    """Build an unavailable result template aligned to one test configuration.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated overlay and base target/condition settings.
    windows : tuple[TrainingWindow, ...]
        Canonical windows resolved on the 40-bin base time axis.

    Returns
    -------
    pipeline.PooledComputationContext
        Independent schema arrays, canonical metadata, and fixed fingerprint.
    """
    edges = canonical_edges()
    centers = (edges[:-1] + edges[1:]) / 2.0
    arrays, metadata = results.initialize_pooled_result_arrays(
        training_windows=windows,
        condition_names=config.condition_names,
        target_names=config.base_config.target_names,
        target_family_names=("categorical", "numerical"),
        target_unit_names=("encoded class", "unitless"),
        outer_fold_count=config.base_config.outer_fold_count,
        evaluation_time_centers_s=centers,
        time_bin_edges_s=edges,
        trial_positions=np.arange(5, dtype=np.int64),
        trial_ids=np.arange(101, 106, dtype=np.int64),
        behavioral_block_ids=np.array([1, 1, 2, 2, 3], dtype=object),
        trial_row_indices=np.array([0, 1, 3, 4], dtype=np.int64),
        pfc_unit_ids=("pfc:1", "pfc:2"),
        hpc_unit_ids=("hpc:1", "hpc:2"),
        pfc_pc_count=2,
        hpc_pc_count=2,
        unit_selection_rules=({"region": "PFC"}, {"region": "HPC"}),
        target_estimator_class_names=("LogisticRegression", "ElasticNet"),
        target_estimator_parameters=({"C": 1.0}, {"alpha": 1.0}),
        run_fingerprint="run-fingerprint",
        source_fingerprint="source-fingerprint",
        random_seed=1729,
    )
    return pipeline.PooledComputationContext(arrays, metadata, "run-fingerprint")


def test_target_audit_projection_preserves_full_table_and_tensor_row_axes():
    """Target audit arrays retain N while fold identities retain common-row M."""
    arrays, _metadata = make_result_template()
    target_values = np.array([0.0, 1.0, np.nan, 0.0, 1.0])
    target_eligible = np.array([True, True, False, True, True])
    baseline_eligible = np.array([True, True, True, False, True])
    bilateral_coverage = np.array([True, True, False, True, True])
    trial_row_indices = arrays["trial_row_indices"]
    outer_fold_ids = np.array([0, 1, -1, 2], dtype=int)

    pipeline.populate_target_audit_arrays(
        arrays,
        target_index=0,
        target_values=target_values,
        target_eligible=target_eligible,
        baseline_eligible=baseline_eligible,
        bilateral_full_window_coverage=bilateral_coverage,
        trial_row_indices=trial_row_indices,
        outer_fold_ids=outer_fold_ids,
    )

    np.testing.assert_array_equal(arrays["target_values"][0], target_values)
    np.testing.assert_array_equal(arrays["target_eligible"][0], target_eligible)
    np.testing.assert_array_equal(
        arrays["master_eligible"][0],
        baseline_eligible & target_eligible & bilateral_coverage,
    )
    np.testing.assert_array_equal(arrays["outer_fold_ids"][0], outer_fold_ids)
    assert arrays["target_values"].shape[1] == 5
    assert arrays["outer_fold_ids"].shape[1] == 4
    reason_names = arrays["reason_code_names"]
    decoded = [
        "" if index < 0 else str(reason_names[index])
        for index in arrays["eligibility_reason_index"][0]
    ]
    assert decoded == ["", "", "target_ineligible", "baseline_ineligible", ""]


def test_one_shot_and_interrupted_resume_have_identical_arrays_and_metadata(
    tmp_path,
    monkeypatch,
):
    """Resume restores exact checkpoints and computes only missing window-target cells."""
    config_path, _base_path, _session_root = write_configuration_tree(tmp_path / "config")
    config = pooled_config.load_pooled_decoding_config(config_path)
    windows = pooled_config.resolve_training_windows(config, canonical_edges())
    context = _context_for_config(config, windows)
    monkeypatch.setattr(
        pipeline,
        "_prepare_pooled_computation_context",
        lambda supplied_config, supplied_windows: replace(
            context,
            arrays={name: value.copy() for name, value in context.arrays.items()},
            metadata=dict(context.metadata),
        ),
    )
    calls: list[str] = []

    def compute_cell(_context, _window, target_identifier):
        """Return deterministic unavailable cells while recording new computation."""
        calls.append(target_identifier)
        return _empty_cell(target_identifier)

    monkeypatch.setattr(pipeline, "_compute_window_target", compute_cell)
    one_shot_directory = tmp_path / "one-shot"
    one_shot_result = pipeline.compute_or_resume_pooled_results(
        config,
        windows,
        one_shot_directory,
    )
    assert calls == list(config.base_config.target_names)
    one_shot = results.load_pooled_run(one_shot_directory)

    calls.clear()
    interrupted_directory = tmp_path / "interrupted"
    original_save = results.save_pooled_checkpoint
    saved_count = 0

    def interrupt_after_first(path, *, checkpoint_key, run_fingerprint, result):
        """Publish the first valid checkpoint and interrupt before the second."""
        nonlocal saved_count
        original_save(
            path,
            checkpoint_key=checkpoint_key,
            run_fingerprint=run_fingerprint,
            result=result,
        )
        saved_count += 1
        if saved_count == 1:
            raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(results, "save_pooled_checkpoint", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        pipeline.compute_or_resume_pooled_results(config, windows, interrupted_directory)
    assert not (interrupted_directory / "results.npz").exists()
    assert len(list((interrupted_directory / "checkpoints").glob("*.npz"))) == 1

    monkeypatch.setattr(results, "save_pooled_checkpoint", original_save)
    calls.clear()
    resumed_result = pipeline.compute_or_resume_pooled_results(
        config,
        windows,
        interrupted_directory,
    )
    assert calls == [config.base_config.target_names[1]]
    resumed = results.load_pooled_run(interrupted_directory)

    assert one_shot_result == one_shot_directory / "results.npz"
    assert resumed_result == interrupted_directory / "results.npz"
    assert resumed["metadata"] == one_shot["metadata"]
    assert set(resumed["arrays"]) == set(one_shot["arrays"])
    for name in one_shot["arrays"]:
        np.testing.assert_array_equal(resumed["arrays"][name], one_shot["arrays"][name])


def test_checkpoint_key_order_is_window_then_target_and_reference_pair_is_unique(
    tmp_path,
    monkeypatch,
):
    """Synchronous computation uses canonical W-by-T checkpoint and pair order."""
    config_path, _base_path, _session_root = write_configuration_tree(
        tmp_path / "config",
        pooled_payload={
            "base_config_path": "task_decoding_config.json",
            "training_window_names": ["pre_choice", "post_choice"],
        },
    )
    config = pooled_config.load_pooled_decoding_config(config_path)
    windows = pooled_config.resolve_training_windows(config, canonical_edges())
    context = _context_for_config(config, windows)
    monkeypatch.setattr(
        pipeline,
        "_prepare_pooled_computation_context",
        lambda *_: replace(
            context,
            arrays={name: value.copy() for name, value in context.arrays.items()},
            metadata=dict(context.metadata),
        ),
    )
    observed: list[str] = []

    def compute_cell(_context, window, target_identifier):
        """Record canonical cell traversal and return deterministic unavailability."""
        observed.append(f"{window.name}::{target_identifier}")
        return _empty_cell(target_identifier)

    monkeypatch.setattr(pipeline, "_compute_window_target", compute_cell)
    pipeline.compute_or_resume_pooled_results(config, windows, tmp_path / "work")

    assert observed == [
        f"{window.name}::{target}"
        for window in windows
        for target in config.base_config.target_names
    ]
    pair_coordinates = list(
        zip(
            context.arrays["evaluation_train_condition"].tolist(),
            context.arrays["evaluation_test_condition"].tolist(),
            strict=True,
        )
    )
    assert len(pair_coordinates) == len(set(pair_coordinates))
    assert pair_coordinates.count(("correct_rewarded", "correct_rewarded")) == 1
