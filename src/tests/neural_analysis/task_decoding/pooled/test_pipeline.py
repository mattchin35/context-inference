"""Synchronous pooled computation, assembly, and resume contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.neural_analysis.task_decoding.pooled import config as pooled_config
from src.neural_analysis.task_decoding.pooled import decoding, pipeline, results
from src.tests.neural_analysis.task_decoding.test_pipeline import write_session_inputs
from src.tests.neural_analysis.task_decoding.pooled.test_config import (
    canonical_edges,
    write_configuration_tree,
)
from src.tests.neural_analysis.task_decoding.pooled.test_results import make_result_template


def _write_real_pooled_overlay(
    tmp_path: Path,
    *,
    training_window_names: tuple[str, ...] = ("pre_choice",),
) -> tuple[Path, dict[str, Path]]:
    """Create one metadata-valid base session plus a portable pooled overlay.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory for the complete synthetic session tree.
    training_window_names : tuple[str, ...], default=("pre_choice",)
        Canonical pooled window identifiers to request.

    Returns
    -------
    tuple[pathlib.Path, dict[str, pathlib.Path]]
        Pooled overlay path and the established small session fixture paths.
    """
    paths = write_session_inputs(tmp_path)
    overlay = paths["session_root"] / "pooled_task_decoding_config.json"
    overlay.write_text(
        json.dumps(
            {
                "base_config_path": paths["config"].name,
                "training_window_names": list(training_window_names),
            }
        ),
        encoding="ascii",
    )
    return overlay, paths


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
        Canonical windows resolved on the configured base time axis.

    Returns
    -------
    pipeline.PooledComputationContext
        Independent schema arrays, canonical metadata, and fixed fingerprint.
    """
    edges = canonical_edges(config.base_config.bin_width_ms)
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


def test_dry_run_reports_exact_pooled_counts_and_bytes_without_scientific_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Planning uses small inputs and exact formulas, never spikes, tensors, or fits."""
    overlay, _paths = _write_real_pooled_overlay(tmp_path)

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Fail if dry planning reaches a full neural or model allocation path."""
        raise AssertionError("dry run reached a forbidden scientific operation")

    monkeypatch.setattr(pipeline.activity, "load_region_activity", forbidden)
    monkeypatch.setattr(pipeline.activity, "build_session_rate_tensors", forbidden)
    monkeypatch.setattr(pipeline.modeling, "make_estimator", forbidden)
    monkeypatch.setattr(results, "initialize_pooled_result_arrays", forbidden)
    monkeypatch.setattr(results, "validate_pooled_source_cleanliness", lambda *_: None)

    plan = pipeline.plan_pooled_decoding_session(overlay)

    envelope = plan["resource_envelope"]
    assert plan["training_window_names"] == ("pre_choice",)
    assert plan["condition_names"] == ("all", "correct_rewarded")
    assert plan["target_names"] == ("current_action", "relative_doubt")
    assert envelope == {
        "full_trial_count": 12,
        "tensor_trial_count": 12,
        "pfc_unit_count": 2,
        "hpc_unit_count": 2,
        "time_bin_count": 8,
        "training_window_count": 1,
        "condition_count": 2,
        "evaluation_pair_count": 3,
        "target_count": 2,
        "outer_fold_count": 3,
        "coefficient_feature_capacity": 4,
        "categorical_fit_count": 36,
        "numerical_fit_count": 36,
        "transform_fit_count": 24,
        "evaluation_metric_cell_count": 1296,
        "tensor_allocation_bytes": envelope["tensor_allocation_bytes"],
        "planned_major_array_bytes": envelope["planned_major_array_bytes"],
    }
    coefficient_cells = 1 * 2 * 2 * 3 * 3 * 2 * 4
    evaluation_cells = 3 * 2 * 3 * 3 * 2 * 8 * 3
    gap_cells = 2 * 2 * 3 * 3 * 2 * 8 * 3
    assert envelope["planned_major_array_bytes"] == (
        envelope["tensor_allocation_bytes"]
        + coefficient_cells * (8 + 1)
        + evaluation_cells * (8 + 1 + 2)
        + gap_cells * (8 + 1 + 2)
    )


def _complete_resource_payload() -> dict[str, object]:
    """Return one minimal complete lean pooled resource payload.

    Returns
    -------
    dict[str, object]
        Exact schema-1 JSON mapping for two completed checkpoint cells.
    """
    return {
        "schema_version": 1,
        "analysis_type": "pooled_task_variable_decoding",
        "measurement_method": "resource.getrusage",
        "status": "complete",
        "peak_rss_bytes": 1024,
        "wall_time_seconds": 2.0,
        "user_cpu_seconds": 1.0,
        "system_cpu_seconds": 0.5,
        "cpu_efficiency": 0.75,
        "resource_envelope": {
            "full_trial_count": 12,
            "tensor_trial_count": 12,
            "pfc_unit_count": 2,
            "hpc_unit_count": 2,
            "time_bin_count": 8,
            "training_window_count": 1,
            "condition_count": 2,
            "evaluation_pair_count": 3,
            "target_count": 2,
            "outer_fold_count": 3,
            "coefficient_feature_capacity": 4,
            "categorical_fit_count": 36,
            "numerical_fit_count": 36,
            "transform_fit_count": 24,
            "evaluation_metric_cell_count": 1296,
            "tensor_allocation_bytes": 4096,
            "planned_major_array_bytes": 65536,
        },
        "completed_checkpoint_keys": [
            "pre_choice::current_action",
            "pre_choice::relative_doubt",
        ],
        "restored_checkpoint_keys": ["pre_choice::current_action"],
        "stage_timing_seconds": {
            "input_loading": 0.1,
            "modeling": 0.2,
            "final_assembly": 0.1,
            "reporting": 0.1,
        },
        "result_counts": {
            "requested_transform_cells": 24,
            "valid_transform_cells": 0,
            "unavailable_transform_cells": 24,
            "requested_model_cells": 72,
            "valid_model_cells": 0,
            "unavailable_model_cells": 72,
            "requested_evaluation_metric_cells": 1296,
            "valid_evaluation_metric_cells": 0,
            "unavailable_evaluation_metric_cells": 1296,
        },
        "checkpoint_size_bytes": 2048,
        "output_size_bytes": 8192,
    }


def test_lean_resource_payload_has_exact_schema_order_and_reconciliation() -> None:
    """Pooled telemetry rejects drift, invalid subsets, and partial complete counts."""
    canonical = (
        "pre_choice::current_action",
        "pre_choice::relative_doubt",
    )
    payload = _complete_resource_payload()

    assert pipeline.validate_pooled_resource_usage_payload(
        payload,
        canonical_checkpoint_keys=canonical,
    ) == payload

    with pytest.raises(ValueError, match="field|schema|unexpected"):
        pipeline.validate_pooled_resource_usage_payload(
            {**payload, "extra": True},
            canonical_checkpoint_keys=canonical,
        )
    invalid_order = {**payload, "completed_checkpoint_keys": list(reversed(canonical))}
    with pytest.raises(ValueError, match="checkpoint|canonical|order"):
        pipeline.validate_pooled_resource_usage_payload(
            invalid_order,
            canonical_checkpoint_keys=canonical,
        )
    invalid_restored = {
        **payload,
        "restored_checkpoint_keys": ["pre_choice::relative_doubt", canonical[0]],
    }
    with pytest.raises(ValueError, match="restored|order|subset"):
        pipeline.validate_pooled_resource_usage_payload(
            invalid_restored,
            canonical_checkpoint_keys=canonical,
        )
    invalid_counts = {
        **payload,
        "result_counts": {
            **payload["result_counts"],
            "unavailable_model_cells": 71,
        },
    }
    with pytest.raises(ValueError, match="count|reconcil"):
        pipeline.validate_pooled_resource_usage_payload(
            invalid_counts,
            canonical_checkpoint_keys=canonical,
        )


def test_output_size_excludes_resource_sidecar_and_atomic_temporaries(tmp_path: Path) -> None:
    """Self-reporting bytes remain stable while the resource sidecar is replaced."""
    (tmp_path / "results.npz").write_bytes(b"result")
    (tmp_path / "summary.md").write_bytes(b"summary")
    (tmp_path / "resource_usage.json").write_bytes(b"old-resource")
    (tmp_path / ".resource_usage.json.temporary").write_bytes(b"temporary")
    expected = len(b"result") + len(b"summary")

    assert pipeline.measure_pooled_output_size_bytes(tmp_path) == expected


def test_prepared_lifecycle_interrupts_resumes_completes_and_reenters_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Lifecycle publication preserves checkpoints and makes completion immutable."""
    overlay, _paths = _write_real_pooled_overlay(tmp_path)
    monkeypatch.setattr(results, "validate_pooled_source_cleanliness", lambda *_: None)
    run_directory = pipeline.prepare_pooled_decoding_run(overlay)
    prepared_config, prepared_windows = pooled_config.load_prepared_pooled_decoding_config(
        run_directory
    )
    context = _context_for_config(prepared_config, prepared_windows)
    execution_identity = json.loads(
        (run_directory / "execution.json").read_text(encoding="utf-8")
    )["run_fingerprint"]
    context_metadata = dict(context.metadata)
    context_metadata["provenance"] = {
        **context.metadata["provenance"],
        "run_fingerprint": execution_identity,
    }
    monkeypatch.setattr(
        pipeline,
        "_prepare_pooled_computation_context",
        lambda *_: replace(
            context,
            arrays={name: value.copy() for name, value in context.arrays.items()},
            metadata=dict(context_metadata),
            run_fingerprint=execution_identity,
        ),
    )

    required_prepared = {
        "config.json",
        "pooled_config.input.json",
        "base_config.input.json",
        "input_manifest.json",
        "scientific_source.json",
        "execution.json",
        "run_state.json",
        "run.log",
        "run_command.txt",
        "resume_command.txt",
        "status_command.txt",
        "run_session.py",
        "pipeline.py",
        "plotting.py",
        "resource_usage.json",
    }
    assert required_prepared.issubset(path.name for path in run_directory.iterdir())
    execution = json.loads((run_directory / "execution.json").read_text())
    for filename in ("run_session.py", "pipeline.py", "plotting.py"):
        snapshot = run_directory / filename
        record = execution["source_snapshots"][filename]
        assert snapshot.stat().st_size == record["size_bytes"]
        assert pipeline.sha256_file(snapshot) == record["sha256"]

    calls: list[str] = []

    def compute_cell(_context, _window, target_identifier):
        """Return deterministic unavailable work while recording resume behavior."""
        calls.append(target_identifier)
        return _empty_cell(target_identifier)

    monkeypatch.setattr(pipeline, "_compute_window_target", compute_cell)
    original_save = results.save_pooled_checkpoint
    saved_count = 0

    def interrupt_after_first(path, *, checkpoint_key, run_fingerprint, result):
        """Publish one valid checkpoint before simulating operator interruption."""
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
        pipeline.run_prepared_pooled_decoding(run_directory)
    interrupted_state = json.loads((run_directory / "run_state.json").read_text())
    assert interrupted_state["lifecycle"] == "interrupted"
    assert interrupted_state["final_results_published"] is False
    assert len(list((run_directory / "checkpoints").glob("*.npz"))) == 1

    monkeypatch.setattr(results, "save_pooled_checkpoint", original_save)
    calls.clear()
    pipeline.run_prepared_pooled_decoding(run_directory)
    assert calls == ["relative_doubt"]
    completed_state = json.loads((run_directory / "run_state.json").read_text())
    assert completed_state["lifecycle"] == "complete"
    assert completed_state["final_results_published"] is True
    assert (run_directory / "results.npz").is_file()
    assert (run_directory / "summary.md").is_file()
    assert len(list((run_directory / "figures").glob("*.png"))) == 6
    summary = (run_directory / "summary.md").read_text(encoding="utf-8").lower()
    for required in (
        "goal",
        "session",
        "parameters",
        "scripts",
        "warnings",
        "errors",
        "elapsed",
        "complete pipelines",
        "cannot isolate estimator weights",
    ):
        assert required in summary

    tracked = {
        path.relative_to(run_directory): (path.stat().st_mtime_ns, path.read_bytes())
        for path in run_directory.rglob("*")
        if path.is_file()
    }
    assert pipeline.run_prepared_pooled_decoding(run_directory) is None
    assert tracked == {
        path.relative_to(run_directory): (path.stat().st_mtime_ns, path.read_bytes())
        for path in run_directory.rglob("*")
        if path.is_file()
    }
    status = pipeline.inspect_pooled_decoding_status(run_directory)
    assert status["lifecycle"] == "complete"
    assert status["final_results_published"] is True
    assert status["result_valid"] is True


def test_prepared_execution_revalidates_inputs_before_accepting_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A changed original input fails before checkpoint loading or computation."""
    overlay, paths = _write_real_pooled_overlay(tmp_path)
    monkeypatch.setattr(results, "validate_pooled_source_cleanliness", lambda *_: None)
    run_directory = pipeline.prepare_pooled_decoding_run(overlay)
    paths["augmented"].write_text("changed\n", encoding="ascii")

    def forbidden(*_args: object, **_kwargs: object) -> None:
        """Reject checkpoint or neural work after manifest validation fails."""
        raise AssertionError("execution passed the input-manifest boundary")

    monkeypatch.setattr(results, "load_pooled_checkpoint", forbidden)
    monkeypatch.setattr(pipeline, "compute_or_resume_pooled_results", forbidden)

    with pytest.raises(ValueError, match="input|manifest|changed|identity"):
        pipeline.run_prepared_pooled_decoding(run_directory)
    state = json.loads((run_directory / "run_state.json").read_text())
    assert state["lifecycle"] == "failed"
    assert state["final_results_published"] is False
    resource_payload = json.loads(
        (run_directory / "resource_usage.json").read_text(encoding="utf-8")
    )
    assert resource_payload["status"] == "failed"
    assert resource_payload["stage_timing_seconds"]["input_loading"] is not None
    assert resource_payload["completed_checkpoint_keys"] == []


def test_active_execution_owner_rejection_is_read_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A losing local writer cannot regress another process's durable state."""
    overlay, _paths = _write_real_pooled_overlay(tmp_path)
    monkeypatch.setattr(results, "validate_pooled_source_cleanliness", lambda *_: None)
    run_directory = pipeline.prepare_pooled_decoding_run(overlay)
    (run_directory / "execution_guard.json").write_text(
        '{"pid":123,"claimed_at":"synthetic"}\n',
        encoding="ascii",
    )
    before = {
        path.relative_to(run_directory): path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file()
    }

    with pytest.raises(RuntimeError, match="active execution owner"):
        pipeline.run_prepared_pooled_decoding(run_directory)

    assert before == {
        path.relative_to(run_directory): path.read_bytes()
        for path in run_directory.rglob("*")
        if path.is_file()
    }


def test_thin_cli_routes_dry_run_new_resume_detached_and_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The pooled command surface delegates without owning scientific logic."""
    from src.neural_analysis.task_decoding.pooled import run_session

    overlay, _paths = _write_real_pooled_overlay(tmp_path)
    prepared = tmp_path / "prepared-pooled-run"
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        pipeline,
        "plan_pooled_decoding_session",
        lambda path: calls.append(("plan", Path(path))) or {"target_count": 2},
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_pooled_decoding_run",
        lambda path: calls.append(("prepare", Path(path))) or prepared,
    )
    monkeypatch.setattr(
        pipeline,
        "run_prepared_pooled_decoding",
        lambda path: calls.append(("run", Path(path))),
    )
    monkeypatch.setattr(
        pipeline,
        "inspect_pooled_decoding_status",
        lambda path: calls.append(("status", Path(path))) or {"lifecycle": "prepared"},
    )
    monkeypatch.setattr(
        run_session,
        "_launch_detached",
        lambda path: calls.append(("detach", Path(path))),
    )

    assert run_session.main(["dry-run", "--config", str(overlay)]) == 0
    assert run_session.main(["new", "--config", str(overlay)]) == 0
    assert run_session.main(["resume", "--run-directory", str(prepared)]) == 0
    assert run_session.main(
        ["resume", "--run-directory", str(prepared), "--detach"]
    ) == 0
    assert run_session.main(["status", "--run-directory", str(prepared)]) == 0
    assert calls == [
        ("plan", overlay),
        ("prepare", overlay),
        ("run", prepared),
        ("run", prepared),
        ("detach", prepared),
        ("status", prepared),
    ]


def test_detached_private_child_waits_for_parent_receipt_before_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Detached output accounting starts only after the launch receipt exists."""
    from src.neural_analysis.task_decoding.pooled import run_session

    prepared = tmp_path / "prepared-pooled-run"
    prepared.mkdir()
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        run_session,
        "_wait_for_detached_receipt",
        lambda path: calls.append(("receipt", Path(path))),
    )
    monkeypatch.setattr(
        pipeline,
        "run_prepared_pooled_decoding",
        lambda path: calls.append(("run", Path(path))),
    )

    assert run_session.main(
        ["_execute-prepared", "--run-directory", str(prepared)]
    ) == 0
    assert calls == [("receipt", prepared), ("run", prepared)]
