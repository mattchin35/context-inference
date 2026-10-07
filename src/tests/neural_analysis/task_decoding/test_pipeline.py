"""WP5B single-session preparation, execution, and CLI contracts.

The fixtures are intentionally small but scientifically coherent: six
contiguous behavioral blocks supply three grouped folds, both configured probes
have ordinary metadata, and the assembly test uses real model records and the
released WP5A result serializer.  No test reads experimental data, launches a
scheduler, or allocates a large neural array.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from threading import Barrier, Thread

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding import activity, config as decoding_config
from src.neural_analysis.task_decoding import modeling, pipeline, results, run_session, targets
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.tests.neural_analysis.task_decoding.test_results import (
    save_run_fixture,
    write_input_fixture,
)


def read_json(path: Path) -> dict[str, object]:
    """Read one test-owned UTF-8 JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Existing small JSON object in a temporary session or run directory.

    Returns
    -------
    dict[str, object]
        Decoded durable state, manifest, provenance, or receipt mapping.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def fail_if_called(*_args: object, **_kwargs: object) -> None:
    """Fail a test if a forbidden loader, poller, or scientific operation runs.

    Parameters
    ----------
    *_args, **_kwargs : object
        Arbitrary dependency arguments supplied through monkeypatch.

    Returns
    -------
    None
        This function always raises AssertionError.
    """
    raise AssertionError("This lifecycle path must not perform the guarded operation.")


def write_probe_source(session_root: Path, probe_id: str) -> dict[str, str]:
    """Create one small probe's metadata and raw-array file names.

    Parameters
    ----------
    session_root : pathlib.Path
        Temporary session root containing the probe directory.
    probe_id : str
        Explicit metadata probe key, such as ``"probe-pfc"``.

    Returns
    -------
    dict[str, str]
        Session-relative metadata paths for a version-2 neural-session record.
        The IRIG archive is deliberately small; normal planning may inspect it,
        whereas sorter spike arrays remain forbidden before execution.
    """
    probe_root = session_root / "probes" / probe_id
    sorter = probe_root / "sorter"
    sorter.mkdir(parents=True)
    (probe_root / "lfp.dat").write_bytes(b"small-lfp-placeholder")
    pd.DataFrame(
        {
            "channel_id": [0, 1],
            "label": ["good", "good"],
            "inside_brain": [True, True],
            "x_um": [0.0, 20.0],
            "y_um": [0.0, 0.0],
        }
    ).to_csv(probe_root / "channel_quality.csv", index=False)
    pd.DataFrame(
        {"cluster_id": [1, 2], "ch": [0, 1], "group": ["good", "good"]}
    ).to_csv(sorter / "cluster_info.tsv", sep="\t", index=False)
    np.save(sorter / "spike_clusters.npy", np.array([1, 2], dtype=np.int64))
    np.save(sorter / "spike_times.npy", np.array([10.1, 10.2], dtype=float))
    np.savez(
        probe_root / "aligned_spikes.npz",
        spike_utc_unix=np.array([10.1, 10.2], dtype=float),
        # The final choice at 21 s requires full [-2, 2] s coverage through 23 s.
        irig_utc_unix=np.array([7.0, 24.0], dtype=float),
    )
    prefix = f"probes/{probe_id}"
    return {
        "lfp": f"{prefix}/lfp.dat",
        "alignment": f"{prefix}/aligned_spikes.npz",
        "sorter": f"{prefix}/sorter",
        "quality": f"{prefix}/channel_quality.csv",
        "sites": {},
        "unit_channels": [0, 1],
    }


def write_session_inputs(tmp_path: Path) -> dict[str, Path]:
    """Create a coherent twelve-trial, two-probe session for pipeline tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned parent directory.

    Returns
    -------
    dict[str, pathlib.Path]
        Existing config, metadata, augmented CSV, feature parameters, and
        output root. Six contiguous two-trial blocks provide both action
        classes in each group and support three grouped outer folds.
    """
    session_root = tmp_path / "session"
    processed = session_root / "processed"
    processed.mkdir(parents=True)
    pfc = write_probe_source(session_root, "probe-pfc")
    hpc = write_probe_source(session_root, "probe-hpc")
    augmented = processed / "augmented_trials.csv"
    trial_count = 12
    pd.DataFrame(
        {
            "cur_trial": np.arange(trial_count, dtype=int),
            "cur_block": np.repeat(np.arange(6, dtype=int), 2),
            "action": np.tile([0, 1], 6),
            "reward": np.tile([0, 1], 6),
            "experimenter_reward_given": np.zeros(trial_count, dtype=int),
            "choice_time": 10.0 + np.arange(trial_count, dtype=float),
            # The selected numerical target is intentionally outer-unavailable.
            "relative_doubt_index": np.ones(trial_count, dtype=float),
        }
    ).to_csv(augmented, index=False)
    (processed / "behavior_trials.csv").write_text("trial\n0\n", encoding="ascii")
    feature_parameters = processed / "trial_feature_params.json"
    feature_parameters.write_text('{"feature_schema":"synthetic-v1"}\n', encoding="ascii")
    metadata = session_root / "neural_session.json"
    metadata.write_text(
        json.dumps(
            {
                "schema_version": "2",
                "session": "synthetic-session",
                "acquisition": "open_ephys",
                "behavior": {"trials": "processed/behavior_trials.csv"},
                "probes": {"probe-pfc": pfc, "probe-hpc": hpc},
                "site_pairs": [],
            }
        ),
        encoding="ascii",
    )
    config_path = session_root / "task_decoding_config.json"
    config_path.write_text(
        json.dumps(
            {
                "session_metadata_path": "neural_session.json",
                "augmented_trial_path": "processed/augmented_trials.csv",
                "trial_feature_parameter_path": "processed/trial_feature_params.json",
                "pfc_region": {
                    "region": "PFC",
                    "probe_id": "probe-pfc",
                    "channel_labels": ["good"],
                    "require_inside_brain": True,
                    "cluster_groups": ["good"],
                },
                "hpc_region": {
                    "region": "HPC",
                    "probe_id": "probe-hpc",
                    "channel_labels": ["good"],
                    "require_inside_brain": True,
                    "cluster_groups": ["good"],
                },
                "alignment": "choice_time",
                "bin_width_ms": 500,
                "pfc_pc_count": 1,
                "hpc_pc_count": 1,
                "target_names": ["current_action", "relative_doubt"],
                "outer_fold_count": 3,
                "inner_fold_count": 3,
                "output_root": "analysis_runs",
            }
        ),
        encoding="ascii",
    )
    return {
        "session_root": session_root,
        "config": config_path,
        "metadata": metadata,
        "augmented": augmented,
        "feature_parameters": feature_parameters,
        "output_root": session_root / "analysis_runs",
    }


def set_regularization_mode(paths: dict[str, Path], regularization_mode: str) -> None:
    """Persist one requested fixed or tuned mode into a test-owned config file.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Session fixture mapping containing the JSON config path.
    regularization_mode : {"fixed", "tuned"}
        Decoder regularization route to persist as a scientific setting.

    Returns
    -------
    None
        Rewrites only the temporary fixture's canonical JSON configuration.
    """
    if regularization_mode not in {"fixed", "tuned"}:
        raise ValueError("regularization_mode must be fixed or tuned.")
    payload = read_json(paths["config"])
    payload["regularization_mode"] = regularization_mode
    paths["config"].write_text(json.dumps(payload), encoding="ascii")


def make_coherent_model_records(
    paths: dict[str, Path],
    *,
    regularization_mode: str = "fixed",
) -> dict[str, object]:
    """Build real target tables, tensors, and one available/unavailable record pair.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Session files returned by :func:`write_session_inputs`.
    regularization_mode : {"fixed", "tuned"}, default="fixed"
        Decoder-record mode used to prove primitive checkpoint conversion.

    Returns
    -------
    dict[str, object]
        Typed config, a 12-row target table, float64 Hz tensors shaped
        ``(12, 8, 2)``, and ordered ``TargetDecodingResult`` values. The first
        record is a genuine available categorical result; the second is a
        genuine outer-unavailable numerical result. ``regularization_mode``
        controls fixed versus nested tuned model records.
    """
    typed_config = replace(
        decoding_config.load_task_decoding_config(paths["config"]),
        regularization_mode=regularization_mode,
    )
    trial_df = pd.read_csv(paths["augmented"])
    target_table = targets.build_target_table(trial_df, typed_config)
    # Tensor construction consumes the execution table, not the original CSV.
    target_table[typed_config.alignment] = trial_df[typed_config.alignment].to_numpy(
        dtype=float,
    )
    rng = np.random.default_rng(7)
    rates_pfc = rng.normal(5.0, 0.2, size=(12, 8, 2)).astype(np.float64)
    rates_hpc = rng.normal(4.0, 0.2, size=(12, 8, 2)).astype(np.float64)
    rate_tensors = activity.SessionRateTensors(
        trial_row_indices=np.arange(12, dtype=np.int64),
        trial_ids=np.arange(12, dtype=np.int64),
        pfc_rate_tensor_hz=rates_pfc,
        hpc_rate_tensor_hz=rates_hpc,
        pfc_bin_centers_s=np.linspace(-1.75, 1.75, 8),
        hpc_bin_centers_s=np.linspace(-1.75, 1.75, 8),
        rate_units="Hz",
    )
    available = modeling.decode_target(
        target_identifier="current_action",
        target_family="categorical",
        target_values=target_table["current_action"].to_numpy(dtype=float),
        block_ids=target_table["block_id"].to_numpy(),
        pfc_rate_tensor_hz=rates_pfc,
        hpc_rate_tensor_hz=rates_hpc,
        pfc_unit_ids=("probe-pfc:1", "probe-pfc:2"),
        hpc_unit_ids=("probe-hpc:1", "probe-hpc:2"),
        pfc_requested_pc_count=1,
        hpc_requested_pc_count=1,
        outer_fold_count=3,
        inner_fold_count=3,
        regularization_mode=regularization_mode,
    )
    unavailable = modeling.decode_target(
        target_identifier="relative_doubt",
        target_family="numerical",
        target_values=target_table["relative_doubt"].to_numpy(dtype=float),
        block_ids=target_table["block_id"].to_numpy(),
        pfc_rate_tensor_hz=rates_pfc,
        hpc_rate_tensor_hz=rates_hpc,
        pfc_unit_ids=("probe-pfc:1", "probe-pfc:2"),
        hpc_unit_ids=("probe-hpc:1", "probe-hpc:2"),
        pfc_requested_pc_count=1,
        hpc_requested_pc_count=1,
        outer_fold_count=3,
        inner_fold_count=3,
        regularization_mode=regularization_mode,
    )
    assert available.status == "available"
    assert len(available.fold_records) == 3 * 8 * 3 * 2
    assert unavailable.status == "unavailable"
    return {
        "config": typed_config,
        "target_table": target_table,
        "rate_tensors": rate_tensors,
        "records": (available, unavailable),
    }


def test_small_session_fixture_is_valid_for_metadata_and_activity_dry_run(tmp_path):
    """The shared fixture satisfies real metadata resolution and bounded dry-run inspection."""
    paths = write_session_inputs(tmp_path)
    typed_config = decoding_config.load_task_decoding_config(paths["config"])
    resolved = resolve_session_metadata(
        load_session_metadata(paths["metadata"]),
        paths["metadata"],
    )
    target_table = targets.build_target_table(pd.read_csv(paths["augmented"]), typed_config)
    target_table[typed_config.alignment] = pd.read_csv(paths["augmented"])[
        typed_config.alignment
    ]
    report = activity.inspect_activity_dry_run(
        target_table,
        typed_config.pfc_region,
        typed_config.hpc_region,
        resolved.probes,
        alignment_event=typed_config.alignment,
        window_s=(-2.0, 2.0),
        bin_width_s=0.5,
    )
    assert report.tensor_allocation_bytes > 0
    assert all(path.is_file() for path in report.source_file_sizes_bytes)


def prepare_with_clean_identity(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    *,
    mode: str,
) -> Path:
    """Prepare one run while isolating ordinary lifecycle tests from repository dirtiness.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch only the owning results identity dependencies.
    paths : dict[str, pathlib.Path]
        Valid session input paths.
    mode : {"foreground", "detached", "slurm"}
        Requested execution route recorded in prepared provenance.

    Returns
    -------
    pathlib.Path
        Newly created immutable prepared run directory.
    """
    monkeypatch.setattr(pipeline.results, "validate_scientific_source_cleanliness", lambda *_: None)
    monkeypatch.setattr(
        pipeline.results,
        "scientific_source_fingerprint",
        lambda *_: {"files": [], "fingerprint": "synthetic-clean"},
    )
    return pipeline.prepare_task_decoding_run(paths["config"], rerun=True, execution_mode=mode)


def prepared_full_fingerprint(run_directory: Path) -> str:
    """Read the immutable prepared run's full scientific fingerprint text.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory created by the public preparation API.

    Returns
    -------
    str
        Nonempty full run fingerprint used by every target-local checkpoint.
    """
    value = (run_directory / "run_fingerprint.txt").read_text(encoding="ascii").strip()
    assert value
    return value


def expected_follow_up_commands(run_directory: Path) -> tuple[str, str]:
    """Return the immutable unattended resume and read-only status commands.

    Parameters
    ----------
    run_directory : pathlib.Path
        Immutable single-session decoding run directory.

    Returns
    -------
    tuple[str, str]
        Exact newline-terminated ``(resume, status)`` commands. Both command
        strings use the public CLI and retain the absolute immutable run path.
    """
    prefix = "uv run python -m src.neural_analysis.task_decoding.run_session"
    quoted_directory = shlex.quote(str(run_directory))
    resume = f"{prefix} resume --run-directory {quoted_directory} --detach\n"
    status = f"{prefix} status --run-directory {quoted_directory}\n"
    return resume, status


def assert_target_result_checkpoint_equivalent(
    expected: modeling.TargetDecodingResult,
    restored: modeling.TargetDecodingResult,
) -> None:
    """Assert full fixed or tuned target-checkpoint fidelity.

    Parameters
    ----------
    expected, restored : modeling.TargetDecodingResult
        Original and checkpoint-restored results for one configured target.
        Fold coefficients are one-dimensional standardized model-scale arrays;
        outer and inner indices are dimensionless matched-row positions.

    Returns
    -------
    None
        Verifies target state, grouped plans, all audit matrices, fold records,
        summaries, coefficients, intercepts, and selected tuning parameters.
    """
    assert restored.target_identifier == expected.target_identifier
    assert restored.inner_fold_count == expected.inner_fold_count
    assert restored.is_available == expected.is_available
    assert restored.status == expected.status
    assert restored.unavailable_reason == expected.unavailable_reason
    np.testing.assert_array_equal(restored.outer_fold_ids, expected.outer_fold_ids)
    assert set(restored.inner_split_plans) == set(expected.inner_split_plans)
    for fold_id, expected_plan in expected.inner_split_plans.items():
        restored_plan = restored.inner_split_plans[fold_id]
        assert restored_plan.is_available == expected_plan.is_available
        assert restored_plan.unavailable_reason == expected_plan.unavailable_reason
        np.testing.assert_array_equal(restored_plan.fold_ids, expected_plan.fold_ids)
        assert len(restored_plan.splits) == len(expected_plan.splits)
        for actual_split, expected_split in zip(
            restored_plan.splits,
            expected_plan.splits,
            strict=True,
        ):
            assert actual_split.fold_id == expected_split.fold_id
            np.testing.assert_array_equal(actual_split.train_indices, expected_split.train_indices)
            np.testing.assert_array_equal(actual_split.test_indices, expected_split.test_indices)
    assert restored.candidate_inner_scores == expected.candidate_inner_scores
    assert restored.candidate_inner_statuses == expected.candidate_inner_statuses
    assert restored.candidate_inner_reasons == expected.candidate_inner_reasons
    assert restored.selected_candidate_indices == expected.selected_candidate_indices
    assert restored.cell_summaries == expected.cell_summaries
    assert len(restored.fold_records) == len(expected.fold_records)
    for actual_record, expected_record in zip(
        restored.fold_records,
        expected.fold_records,
        strict=True,
    ):
        for field_name in (
            "record_key",
            "target_identifier",
            "inner_fold_count",
            "outer_fold_id",
            "time_bin_index",
            "region_configuration",
            "representation",
            "regularization_mode",
            "is_valid",
            "status",
            "reason",
            "train_count",
            "test_count",
            "train_class_counts",
            "test_class_counts",
            "requested_feature_count",
            "effective_feature_count",
            "estimator_class",
            "estimator_parameters",
            "convergence_status",
            "metrics",
            "feature_ids",
            "positive_class",
            "coefficient_scale_label",
            "selected_parameters",
            "candidate_inner_scores",
            "candidate_inner_statuses",
            "candidate_inner_reasons",
            "selected_candidate_index",
        ):
            assert getattr(actual_record, field_name) == getattr(expected_record, field_name)
        np.testing.assert_array_equal(
            actual_record.outer_test_indices,
            expected_record.outer_test_indices,
        )
        np.testing.assert_array_equal(
            actual_record.inner_selection_indices,
            expected_record.inner_selection_indices,
        )
        np.testing.assert_allclose(
            actual_record.coefficients,
            expected_record.coefficients,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            actual_record.intercept,
            expected_record.intercept,
            equal_nan=True,
        )


def write_real_checkpoint(
    path: Path,
    target_result: modeling.TargetDecodingResult,
    full_fingerprint: str,
) -> bytes:
    """Write one full model-result checkpoint and return its immutable payload.

    Parameters
    ----------
    path : pathlib.Path
        New target-checkpoint ``.npz`` destination below a temporary run.
    target_result : modeling.TargetDecodingResult
        Complete fixed or tuned target record. Its fold arrays, coefficients,
        intercepts, summaries, and nested candidate audits are converted to
        primitive checkpoint members by the pipeline's owning helper.
    full_fingerprint : str
        Exact prepared full-run fingerprint required for resume matching.

    Returns
    -------
    bytes
        Exact checkpoint bytes after successful real results-module publication.
    """
    results.save_target_checkpoint(
        path,
        target_label=target_result.target_identifier,
        target_arrays=pipeline._target_result_to_checkpoint_arrays(target_result),
        full_run_fingerprint=full_fingerprint,
    )
    return path.read_bytes()


def patch_coherent_execution(
    monkeypatch: pytest.MonkeyPatch,
    model_inputs: dict[str, object],
) -> None:
    """Patch only owning module loaders to supply typed bounded scientific outputs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing selected module-owned I/O boundaries.
    model_inputs : dict[str, object]
        Real table, tensors, and ordered model records from
        :func:`make_coherent_model_records`.

    Returns
    -------
    None
        Subsequent pipeline execution receives real target/model data rather
        than opaque stand-ins while avoiding raw-array I/O in lifecycle tests.
    """
    typed_config = model_inputs["config"]
    resolved_session = resolve_session_metadata(
        load_session_metadata(typed_config.session_metadata_path),
        typed_config.session_metadata_path,
    )
    source_by_probe = {source.probe_id: source for source in resolved_session.probes}
    pfc_metadata = pd.DataFrame(
        {
            "unit_id": ["probe-pfc:1", "probe-pfc:2"],
            "region": ["PFC", "PFC"],
            "probe_id": ["probe-pfc", "probe-pfc"],
            "cluster_id": [1, 2],
            "channel": [0, 1],
            "quality_label": ["good", "good"],
        }
    )
    hpc_metadata = pd.DataFrame(
        {
            "unit_id": ["probe-hpc:1", "probe-hpc:2"],
            "region": ["HPC", "HPC"],
            "probe_id": ["probe-hpc", "probe-hpc"],
            "cluster_id": [1, 2],
            "channel": [0, 1],
            "quality_label": ["good", "good"],
        }
    )
    def region_activity(region_config, *_args, **_kwargs):
        """Return a typed minimal region activity record without loading spikes."""
        is_pfc = region_config.region == "PFC"
        return activity.RegionActivity(
            region=region_config.region,
            probe_id=region_config.probe_id,
            selected_channel_ids=np.array([0, 1], dtype=np.int64),
            selection_rules={"channel_labels": ["good"], "cluster_groups": ["good"]},
            cluster_metadata=pd.DataFrame({"cluster_id": [1, 2]}),
            unit_metadata=pfc_metadata if is_pfc else hpc_metadata,
            spike_group=None,
            coverage=activity.ProbeCoverage("irig_utc_unix", 7.0, 24.0),
            source=source_by_probe[region_config.probe_id],
        )

    records = {record.target_identifier: record for record in model_inputs["records"]}
    monkeypatch.setattr(
        pipeline.targets,
        "build_target_table",
        lambda *_: model_inputs["target_table"],
    )
    monkeypatch.setattr(pipeline.activity, "load_region_activity", region_activity)
    def build_rate_tensors(
        execution_target_table,
        _pfc_activity,
        _hpc_activity,
        *,
        alignment_event,
        window_s,
        bin_width_s,
        **_kwargs,
    ):
        """Verify the exact execution table before returning bounded test tensors."""
        assert alignment_event == typed_config.alignment
        assert window_s == (-2.0, 2.0)
        assert bin_width_s == typed_config.bin_width_ms / 1000.0
        assert execution_target_table.equals(model_inputs["target_table"])
        assert alignment_event in execution_target_table
        common_rows = model_inputs["rate_tensors"].trial_row_indices
        np.testing.assert_array_equal(common_rows, np.arange(execution_target_table.shape[0]))
        np.testing.assert_array_equal(
            model_inputs["rate_tensors"].trial_ids,
            execution_target_table.loc[common_rows, "trial_id"].to_numpy(),
        )
        return model_inputs["rate_tensors"]

    monkeypatch.setattr(pipeline.activity, "build_session_rate_tensors", build_rate_tensors)
    monkeypatch.setattr(
        pipeline.modeling,
        "decode_target",
        lambda *, target_identifier, **_kwargs: records[target_identifier],
    )


def patch_final_reporting_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
) -> None:
    """Patch every final reporting boundary while preserving execution-stage assertions.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing all result publication, summary, figure, and final
        directory-validation boundaries owned by the pipeline.
    events : list[str]
        Mutable ordered event labels. These are dimensionless lifecycle events.

    Returns
    -------
    None
        Prevents tests about target decoding/checkpoint resume from depending on
        unimplemented reporting while proving that execution reaches publication.
    """
    monkeypatch.setattr(
        pipeline.results,
        "save_task_decoding_run",
        lambda *_args, **_kwargs: events.append("npz"),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_run_summary",
        lambda *_args, **_kwargs: events.append("summary"),
    )
    monkeypatch.setattr(
        pipeline,
        "_write_minimal_family_heatmaps",
        lambda *_args, **_kwargs: events.append("figures"),
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_published_run_directory",
        lambda *_args, **_kwargs: events.append("validate"),
    )


def test_plan_reads_only_small_inputs_and_reports_dirtiness(monkeypatch, tmp_path):
    """Dry planning reads metadata/CSV/IRIG only, never spike arrays or model fitting."""
    paths = write_session_inputs(tmp_path)
    original_load = np.load

    def forbid_spike_array_open(path, *args, **kwargs):
        """Allow small IRIG archives but reject sorter spike-array opens during planning."""
        if Path(path).name in {"spike_clusters.npy", "spike_times.npy"}:
            raise AssertionError("dry planning must not open sorter spike arrays")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    monkeypatch.setattr(pipeline.activity, "build_session_rate_tensors", fail_if_called)
    monkeypatch.setattr(pipeline.modeling, "decode_target", fail_if_called)
    monkeypatch.setattr(np, "load", forbid_spike_array_open)
    monkeypatch.setattr(
        pipeline.results,
        "validate_scientific_source_cleanliness",
        lambda *_: (_ for _ in ()).throw(ValueError("synthetic dirty source")),
    )

    plan = pipeline.plan_task_decoding_session(paths["config"])

    assert tuple(plan["target_names"]) == ("current_action", "relative_doubt")
    assert plan["fit_count"] == 2 * 3 * 8 * 3 * 2
    assert plan["tensor_allocation_bytes"] > 0
    assert "synthetic dirty source" in plan["source_cleanliness"]
    assert not paths["output_root"].exists()


def test_prepare_rejects_dirty_scientific_source(monkeypatch, tmp_path):
    """Preparation has one dedicated causal scientific-source cleanliness failure."""
    paths = write_session_inputs(tmp_path)
    monkeypatch.setattr(
        pipeline.results,
        "validate_scientific_source_cleanliness",
        lambda *_: (_ for _ in ()).throw(ValueError("uncommitted analysis source")),
    )
    with pytest.raises(ValueError, match="uncommitted analysis source"):
        pipeline.prepare_task_decoding_run(paths["config"], rerun=True, execution_mode="foreground")
    assert not paths["output_root"].exists()


def test_prepare_discovery_uses_exact_fingerprint_not_latest(monkeypatch, tmp_path):
    """Only a matching validated complete run skips; every recovery state is reported."""
    paths = write_session_inputs(tmp_path)
    run_root = paths["output_root"]
    expected = "same-fingerprint"
    states = {
        "task_variable_decoding_matching-complete": ("complete", expected, True),
        "task_variable_decoding_matching-failed": ("failed", expected, False),
        "task_variable_decoding_matching-interrupted": ("interrupted", expected, False),
        "task_variable_decoding_matching-corrupt": ("complete", expected, False),
        "task_variable_decoding_unrelated-complete": ("complete", "other-fingerprint", True),
        "not_a_task_decoding_run": ("failed", expected, False),
    }
    for name, (lifecycle, fingerprint, _valid) in states.items():
        directory = run_root / name
        directory.mkdir(parents=True)
        (directory / "run_state.json").write_text(json.dumps({"lifecycle": lifecycle}))
        (directory / "execution.json").write_text("{}")
        (directory / "run_fingerprint.txt").write_text(fingerprint, encoding="ascii")
    monkeypatch.setattr(pipeline.results, "validate_scientific_source_cleanliness", lambda *_: None)
    monkeypatch.setattr(
        pipeline,
        "_build_preparation_identity",
        lambda *_: {"fingerprint": expected},
    )
    monkeypatch.setattr(
        pipeline,
        "_validate_complete_prepared_run",
        lambda directory: Path(directory).name == "task_variable_decoding_matching-complete",
    )

    assert pipeline.prepare_task_decoding_run(
        paths["config"], rerun=False, execution_mode="foreground"
    ) == (run_root / "task_variable_decoding_matching-complete")
    monkeypatch.setattr(pipeline, "_validate_complete_prepared_run", lambda _directory: False)
    with pytest.raises(RuntimeError) as error:
        pipeline.prepare_task_decoding_run(
            paths["config"],
            rerun=False,
            execution_mode="foreground",
        )
    message = str(error.value)
    assert {
        "task_variable_decoding_matching-failed",
        "task_variable_decoding_matching-interrupted",
        "task_variable_decoding_matching-corrupt",
    } <= set(message.split()) or all(
        name in message
        for name in (
            "task_variable_decoding_matching-failed",
            "task_variable_decoding_matching-interrupted",
            "task_variable_decoding_matching-corrupt",
        )
    )
    assert "task_variable_decoding_unrelated-complete" not in message
    assert "not_a_task_decoding_run" not in message


def test_rerun_claims_same_second_suffix_without_overwriting(monkeypatch, tmp_path):
    """Concurrent same-second reruns claim distinct immutable siblings atomically."""
    paths = write_session_inputs(tmp_path)
    monkeypatch.setattr(pipeline, "_utc_timestamp", lambda: "2026-10-07T01-02-03Z")
    first = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    before = (first / "run_state.json").read_bytes()
    second = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    assert first != second
    assert first.parent == second.parent
    assert first.name.endswith("2026-10-07T01-02-03Z")
    assert second.name.endswith("2026-10-07T01-02-03Z-01")
    assert (first / "run_state.json").read_bytes() == before


def test_prepare_writes_immutable_artifacts_and_complete_provenance(monkeypatch, tmp_path):
    """Preparation saves all bounded artifacts before any large scientific loader can run."""
    paths = write_session_inputs(tmp_path)
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    required = {
        "config.json", "trial_feature_params.json", "input_manifest.json", "scientific_source.json",
        "execution.json",
        "run_fingerprint.txt",
        "run_state.json",
        "resume_command.txt",
        "status_command.txt",
        "run_session.py",
        "checkpoints", "figures", "run.log",
    }
    assert required <= {member.name for member in run_directory.iterdir()}
    assert (run_directory / "trial_feature_params.json").read_bytes() == paths[
        "feature_parameters"
    ].read_bytes()
    execution = read_json(run_directory / "execution.json")
    assert {
        "mode",
        "pid",
        "host",
        "platform",
        "architecture",
        "cpu_count",
        "runtime_versions",
    } <= set(execution)
    assert execution["thread_limits"] == {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert {
        "python",
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "pynapple",
        "matplotlib",
    } <= set(execution["runtime_versions"])
    assert read_json(run_directory / "run_state.json")["lifecycle"] == "initialized"
    expected_resume, expected_status = expected_follow_up_commands(run_directory)
    assert (run_directory / "resume_command.txt").read_text(encoding="utf-8") == expected_resume
    assert (run_directory / "status_command.txt").read_text(encoding="utf-8") == expected_status


def test_prepare_shell_quotes_command_files_for_run_directory_with_spaces(monkeypatch, tmp_path):
    """Actual prepared command files safely round-trip an immutable path with spaces."""
    paths = write_session_inputs(tmp_path)
    config_payload = read_json(paths["config"])
    config_payload["output_root"] = "analysis runs with spaces"
    paths["config"].write_text(json.dumps(config_payload), encoding="ascii")
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    assert " " in str(run_directory)
    prefix = "uv run python -m src.neural_analysis.task_decoding.run_session"
    quoted_directory = shlex.quote(str(run_directory))
    expected_resume = f"{prefix} resume --run-directory {quoted_directory} --detach\n"
    expected_status = f"{prefix} status --run-directory {quoted_directory}\n"
    resume_command = (run_directory / "resume_command.txt").read_text(encoding="utf-8")
    status_command = (run_directory / "status_command.txt").read_text(encoding="utf-8")
    assert resume_command == expected_resume
    assert status_command == expected_status
    assert shlex.split(resume_command) == [
        "uv",
        "run",
        "python",
        "-m",
        "src.neural_analysis.task_decoding.run_session",
        "resume",
        "--run-directory",
        str(run_directory),
        "--detach",
    ]
    assert shlex.split(status_command) == [
        "uv",
        "run",
        "python",
        "-m",
        "src.neural_analysis.task_decoding.run_session",
        "status",
        "--run-directory",
        str(run_directory),
    ]


@pytest.mark.parametrize("mutation", ("source", "analysis_version", "runtime"))
@pytest.mark.parametrize("route", ("foreground", "detached_private", "mocked_slurm"))
def test_shared_execution_revalidates_live_identity_before_guard_or_loader(
    monkeypatch,
    tmp_path,
    mutation,
    route,
):
    """Every foreground/private/Slurm execution route revalidates before a guard."""
    paths = write_session_inputs(tmp_path)
    execution_mode = {
        "foreground": "foreground",
        "detached_private": "detached",
        "mocked_slurm": "slurm",
    }[route]
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode=execution_mode)
    calls: list[str] = []
    identity_calls: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "_claim_execution_guard",
        lambda *_args, **_kwargs: calls.append("guard"),
    )
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    def current_source_identity(*_args, **_kwargs):
        """Return one recorded live source identity for route-causality assertions."""
        identity_calls.append("identity")
        fingerprint = "changed-source" if mutation == "source" else "synthetic-clean"
        return {"files": [], "fingerprint": fingerprint}

    monkeypatch.setattr(
        pipeline.results,
        "scientific_source_fingerprint",
        current_source_identity,
    )
    if mutation == "analysis_version":
        monkeypatch.setattr(pipeline.decoding_config, "ANALYSIS_VERSION", "changed-version")
    elif mutation == "runtime":
        monkeypatch.setattr(
            pipeline,
            "_runtime_version_mapping",
            lambda: {"python": "changed-runtime"},
        )
    if route == "foreground":
        with pytest.raises(ValueError, match="source|version|runtime|fingerprint"):
            pipeline.run_prepared_task_decoding(run_directory)
    else:
        if route == "mocked_slurm":
            monkeypatch.setenv("SLURM_JOB_ID", "synthetic-job")
        assert run_session.main(["_execute-prepared", "--run-directory", str(run_directory)]) != 0
    assert identity_calls == ["identity"]
    assert calls == []


@pytest.mark.parametrize("route", ("foreground", "detached_private", "mocked_slurm"))
def test_shared_execution_identity_control_reaches_guard(monkeypatch, tmp_path, route):
    """Every execution route reaches the same private guard after revalidation."""
    paths = write_session_inputs(tmp_path)
    execution_mode = {
        "foreground": "foreground",
        "detached_private": "detached",
        "mocked_slurm": "slurm",
    }[route]
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode=execution_mode)
    events: list[str] = []

    def current_source_identity(*_args, **_kwargs):
        """Record the live identity check before returning the prepared fingerprint."""
        events.append("identity")
        return {"files": [], "fingerprint": "synthetic-clean"}

    def guarded_claim(*_args, **_kwargs):
        """Prove that the common execution guard follows successful revalidation."""
        events.append("guard")
        raise AssertionError("guarded operation")

    monkeypatch.setattr(
        pipeline.results,
        "scientific_source_fingerprint",
        current_source_identity,
    )
    monkeypatch.setattr(pipeline, "_claim_execution_guard", guarded_claim)
    if route == "foreground":
        with pytest.raises(AssertionError, match="guarded operation"):
            pipeline.run_prepared_task_decoding(run_directory)
    else:
        if route == "mocked_slurm":
            monkeypatch.setenv("SLURM_JOB_ID", "synthetic-job")
        assert run_session.main(["_execute-prepared", "--run-directory", str(run_directory)]) != 0
    assert events == ["identity", "guard"]


def test_revalidation_detects_original_input_mutation_not_only_sidecars(monkeypatch, tmp_path):
    """Changing a manifested source after preparation fails before acquiring a run guard."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    paths["augmented"].write_text(paths["augmented"].read_text() + "\n", encoding="ascii")
    monkeypatch.setattr(pipeline, "_claim_execution_guard", fail_if_called)
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    with pytest.raises(ValueError, match="manifest|fingerprint|input"):
        pipeline.run_prepared_task_decoding(run_directory)


def test_private_assembly_preserves_real_target_records_and_round_trips(monkeypatch, tmp_path):
    """Assembly converts ordered real model records to the exact WP5A payload once."""
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    input_manifest = results.build_input_manifest(
        session_root=paths["session_root"],
        files={
            "neural_session": paths["metadata"],
            "augmented_trials": paths["augmented"],
            "trial_feature_parameters": paths["feature_parameters"],
        },
        file_lists={
            "pfc_sorter": (
                paths["session_root"] / "probes/probe-pfc/sorter/cluster_info.tsv",
            ),
            "hpc_sorter": (
                paths["session_root"] / "probes/probe-hpc/sorter/cluster_info.tsv",
            ),
        },
    )
    region_activity_metadata = {
        "PFC": {
            "unit_ids": ["probe-pfc:1", "probe-pfc:2"],
            "selection_rules": {"channel_labels": ["good"], "cluster_groups": ["good"]},
        },
        "HPC": {
            "unit_ids": ["probe-hpc:1", "probe-hpc:2"],
            "selection_rules": {"channel_labels": ["good"], "cluster_groups": ["good"]},
        },
    }
    arrays, meta = pipeline._assemble_run_result_payload(
        config=model_inputs["config"],
        target_table=model_inputs["target_table"],
        rate_tensors=model_inputs["rate_tensors"],
        region_activity_metadata=region_activity_metadata,
        ordered_target_results=model_inputs["records"],
        stage_timing_seconds={"targets": 0.01, "activity": 0.02, "modeling": 0.03},
        input_manifest=input_manifest,
        scientific_source={"fingerprint": "source-fingerprint-test"},
        execution_provenance={"mode": "foreground", "runtime_versions": {}},
        run_fingerprint="pipeline-assembly-test",
    )
    assert tuple(arrays["target_labels"]) == ("current_action", "relative_doubt")
    assert arrays["outer_fold_ids"].shape == (2, 12)
    assert arrays["target_status"].tolist() == ["available", "unavailable"]
    run_directory = tmp_path / "assembled-run"
    run_directory.mkdir()
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        meta=meta,
        input_manifest=input_manifest,
        scientific_config=decoding_config.scientific_config_payload(model_inputs["config"]),
        feature_parameter_source=paths["feature_parameters"],
        run_fingerprint="pipeline-assembly-test",
    )
    loaded = results.load_task_decoding_run(
        run_directory,
        expected_run_fingerprint="pipeline-assembly-test",
    )
    assert tuple(loaded["arrays"]["target_labels"]) == ("current_action", "relative_doubt")


def test_checkpoint_roundtrip_uses_real_results_api_and_configured_order(tmp_path):
    """Target checkpoints are primitive, fingerprint-bound, and restored in configured order."""
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    first = checkpoint_root / "current_action.npz"
    second = checkpoint_root / "relative_doubt.npz"
    payload = {"outer_fold_ids": np.array([0, 1, 2], dtype=np.int64)}
    results.save_target_checkpoint(
        first,
        target_label="current_action",
        target_arrays=payload,
        full_run_fingerprint="full",
    )
    results.save_target_checkpoint(
        second,
        target_label="relative_doubt",
        target_arrays=payload,
        full_run_fingerprint="full",
    )
    assert results.load_target_checkpoint(
        first,
        expected_full_run_fingerprint="full",
    )["target_label"] == "current_action"
    before = first.read_bytes()
    with pytest.raises(ValueError, match="fingerprint"):
        results.load_target_checkpoint(first, expected_full_run_fingerprint="different")
    assert first.read_bytes() == before
    restored = [
        results.load_target_checkpoint(
            checkpoint_root / f"{label}.npz",
            expected_full_run_fingerprint="full",
        )["target_label"]
        for label in ("current_action", "relative_doubt")
    ]
    assert restored == ["current_action", "relative_doubt"]


def test_target_result_checkpoint_primitives_restore_in_configured_order(tmp_path):
    """A real model record serializes to primitive checkpoint arrays and restores exactly once."""
    paths = write_session_inputs(tmp_path)
    records = make_coherent_model_records(paths)["records"]
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    restored_identifiers: list[str] = []
    for record in records:
        target_arrays = pipeline._target_result_to_checkpoint_arrays(record)
        checkpoint_path = checkpoint_root / f"{record.target_identifier}.npz"
        results.save_target_checkpoint(
            checkpoint_path,
            target_label=record.target_identifier,
            target_arrays=target_arrays,
            full_run_fingerprint="full",
        )
    for label in ("current_action", "relative_doubt"):
        saved = results.load_target_checkpoint(
            checkpoint_root / f"{label}.npz",
            expected_full_run_fingerprint="full",
        )
        restored = pipeline._target_result_from_checkpoint(saved)
        expected = next(record for record in records if record.target_identifier == label)
        assert_target_result_checkpoint_equivalent(expected, restored)
        restored_identifiers.append(restored.target_identifier)
    assert restored_identifiers == ["current_action", "relative_doubt"]


def test_tuned_target_checkpoint_round_trips_inner_plans_and_candidate_audits(tmp_path):
    """Tuned checkpoint conversion preserves nested split plans and candidate audit primitives."""
    paths = write_session_inputs(tmp_path)
    tuned = make_coherent_model_records(paths, regularization_mode="tuned")["records"][0]
    checkpoint_path = tmp_path / "current_action_tuned.npz"
    results.save_target_checkpoint(
        checkpoint_path,
        target_label=tuned.target_identifier,
        target_arrays=pipeline._target_result_to_checkpoint_arrays(tuned),
        full_run_fingerprint="full",
    )
    restored = pipeline._target_result_from_checkpoint(
        results.load_target_checkpoint(
            checkpoint_path,
            expected_full_run_fingerprint="full",
        )
    )
    assert_target_result_checkpoint_equivalent(tuned, restored)


@pytest.mark.parametrize("regularization_mode", ("fixed", "tuned"))
def test_target_result_checkpoint_roundtrips_all_fixed_or_tuned_values(
    tmp_path,
    regularization_mode,
):
    """Checkpoint conversion preserves all fixed and tuned target audit primitives."""
    paths = write_session_inputs(tmp_path)
    expected = make_coherent_model_records(
        paths,
        regularization_mode=regularization_mode,
    )["records"][0]
    checkpoint_path = tmp_path / f"{regularization_mode}.npz"
    results.save_target_checkpoint(
        checkpoint_path,
        target_label=expected.target_identifier,
        target_arrays=pipeline._target_result_to_checkpoint_arrays(expected),
        full_run_fingerprint="full",
    )
    loaded = results.load_target_checkpoint(
        checkpoint_path,
        expected_full_run_fingerprint="full",
    )
    assert_target_result_checkpoint_equivalent(
        expected,
        pipeline._target_result_from_checkpoint(loaded),
    )


def test_execution_orders_real_module_owned_stages_and_resume(monkeypatch, tmp_path):
    """Execution uses owning module seams and resumes matching target checkpoints in order."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    calls: list[str] = []
    build_target_table = pipeline.targets.build_target_table
    load_region_activity = pipeline.activity.load_region_activity
    build_rate_tensors = pipeline.activity.build_session_rate_tensors
    decode_target = pipeline.modeling.decode_target

    def record_target_table(*args, **kwargs):
        """Record target construction while returning its real typed table."""
        calls.append("targets")
        return build_target_table(*args, **kwargs)

    def record_region_activity(*args, **kwargs):
        """Record typed region activity loading without replacing its value."""
        calls.append("activity")
        return load_region_activity(*args, **kwargs)

    def record_rate_tensors(*args, **kwargs):
        """Record typed rate-tensor construction without replacing its value."""
        calls.append("tensors")
        return build_rate_tensors(*args, **kwargs)

    def record_decode_target(*args, **kwargs):
        """Record real model result retrieval without replacing its record."""
        calls.append("model")
        return decode_target(*args, **kwargs)

    monkeypatch.setattr(
        pipeline.targets,
        "build_target_table",
        record_target_table,
    )
    monkeypatch.setattr(
        pipeline.activity,
        "load_region_activity",
        record_region_activity,
    )
    monkeypatch.setattr(
        pipeline.activity,
        "build_session_rate_tensors",
        record_rate_tensors,
    )
    monkeypatch.setattr(
        pipeline.modeling,
        "decode_target",
        record_decode_target,
    )
    monkeypatch.setattr(
        pipeline.results,
        "save_target_checkpoint",
        lambda *_args, **_kwargs: calls.append("checkpoint"),
    )
    patch_final_reporting_boundaries(monkeypatch, calls)
    monkeypatch.setattr(pipeline, "_assemble_run_result_payload", lambda **_: ({}, {}))
    pipeline.run_prepared_task_decoding(run_directory)
    assert (
        calls.index("targets")
        < calls.index("activity")
        < calls.index("tensors")
        < calls.index("model")
    )
    assert calls[-4:] == ["npz", "summary", "figures", "validate"]


@pytest.mark.parametrize("regularization_mode", ("fixed", "tuned"))
def test_execution_resumes_matching_real_checkpoint_and_assembles_configured_order(
    monkeypatch,
    tmp_path,
    regularization_mode,
):
    """Fixed and tuned resume restores exact checkpoints before decoding remaining targets."""
    paths = write_session_inputs(tmp_path)
    set_regularization_mode(paths, regularization_mode)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    model_inputs = make_coherent_model_records(paths, regularization_mode=regularization_mode)
    patch_coherent_execution(monkeypatch, model_inputs)
    first, second = model_inputs["records"]
    results.save_target_checkpoint(
        run_directory / "checkpoints" / f"{first.target_identifier}.npz",
        target_label=first.target_identifier,
        target_arrays=pipeline._target_result_to_checkpoint_arrays(first),
        full_run_fingerprint=prepared_full_fingerprint(run_directory),
    )
    decoded: list[str] = []
    assembled: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        pipeline.modeling,
        "decode_target",
        lambda *, target_identifier, **_kwargs: decoded.append(target_identifier)
        or second,
    )
    monkeypatch.setattr(
        pipeline,
        "_assemble_run_result_payload",
        lambda *, ordered_target_results, **_kwargs: assembled.append(
            tuple(record.target_identifier for record in ordered_target_results)
        ) or ({}, {}),
    )
    reporting_events: list[str] = []
    patch_final_reporting_boundaries(monkeypatch, reporting_events)
    pipeline.run_prepared_task_decoding(run_directory)
    assert decoded == ["relative_doubt"]
    assert assembled == [("current_action", "relative_doubt")]
    assert reporting_events == ["npz", "summary", "figures", "validate"]


@pytest.mark.parametrize("regularization_mode", ("fixed", "tuned"))
def test_execution_rejects_mismatched_checkpoint_before_decoding(
    monkeypatch,
    tmp_path,
    regularization_mode,
):
    """Fixed and tuned checkpoints with another full fingerprint cannot resume."""
    paths = write_session_inputs(tmp_path)
    set_regularization_mode(paths, regularization_mode)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    record = make_coherent_model_records(
        paths,
        regularization_mode=regularization_mode,
    )["records"][0]
    results.save_target_checkpoint(
        run_directory / "checkpoints" / f"{record.target_identifier}.npz",
        target_label=record.target_identifier,
        target_arrays=pipeline._target_result_to_checkpoint_arrays(record),
        full_run_fingerprint="different-fingerprint",
    )
    monkeypatch.setattr(pipeline.modeling, "decode_target", fail_if_called)
    with pytest.raises(ValueError, match="fingerprint|checkpoint"):
        pipeline.run_prepared_task_decoding(run_directory)


@pytest.mark.parametrize("regularization_mode", ("fixed", "tuned"))
def test_execution_rejects_saved_scientific_parameter_mismatch_before_decoding(
    monkeypatch,
    tmp_path,
    regularization_mode,
):
    """Both fixed and tuned resume reject a saved scientific parameter mismatch early."""
    paths = write_session_inputs(tmp_path)
    set_regularization_mode(paths, regularization_mode)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    saved_config = read_json(run_directory / "config.json")
    saved_config["pfc_pc_count"] = 2
    (run_directory / "config.json").write_text(json.dumps(saved_config), encoding="ascii")
    monkeypatch.setattr(pipeline.modeling, "decode_target", fail_if_called)
    with pytest.raises(ValueError, match="config|parameter|fingerprint|scientific"):
        pipeline.run_prepared_task_decoding(run_directory)


@pytest.mark.parametrize("error", (RuntimeError("bug"), OSError("disk"), ValueError("schema")))
def test_unexpected_error_marks_failed_flushes_and_keeps_checkpoints(monkeypatch, tmp_path, error):
    """Unexpected errors retain completed checkpoints but never claim final publication."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    checkpoint = run_directory / "checkpoints" / "current_action.npz"
    completed_target = make_coherent_model_records(paths)["records"][0]
    checkpoint_bytes = write_real_checkpoint(
        checkpoint,
        completed_target,
        prepared_full_fingerprint(run_directory),
    )
    monkeypatch.setattr(
        pipeline.modeling,
        "decode_target",
        lambda **_: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(pipeline.results, "save_task_decoding_run", fail_if_called)
    with pytest.raises(type(error), match=str(error)):
        pipeline.run_prepared_task_decoding(run_directory)
    state = read_json(run_directory / "run_state.json")
    assert state["lifecycle"] == "failed"
    assert state["last_error"]
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert (run_directory / "run.log").read_text(encoding="utf-8")
    assert not (run_directory / "results.npz").exists()


@pytest.mark.parametrize("interrupt", (KeyboardInterrupt, SystemExit))
def test_interruption_records_resumable_state_and_nonzero_cli(monkeypatch, tmp_path, interrupt):
    """Keyboard and system interruption preserve checkpoints and never become complete."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    checkpoint = run_directory / "checkpoints" / "current_action.npz"
    completed_target = make_coherent_model_records(paths)["records"][0]
    checkpoint_bytes = write_real_checkpoint(
        checkpoint,
        completed_target,
        prepared_full_fingerprint(run_directory),
    )
    monkeypatch.setattr(
        pipeline.modeling,
        "decode_target",
        lambda **_: (_ for _ in ()).throw(interrupt()),
    )
    with pytest.raises(interrupt):
        pipeline.run_prepared_task_decoding(run_directory)
    assert read_json(run_directory / "run_state.json")["lifecycle"] == "interrupted"
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert "interrupted" in (run_directory / "run.log").read_text(encoding="utf-8").lower()


@pytest.mark.parametrize("failure_stage", ("results", "summary", "figure", "complete"))
def test_atomic_final_publication_failure_preserves_truthful_lifecycle(
    monkeypatch,
    tmp_path,
    failure_stage,
):
    """Each final artifact uses a sibling replace and a failed publish never completes."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    replaced: list[tuple[Path, Path]] = []
    injected: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source, destination):
        """Record sibling publication and fail only the requested final destination."""
        source_path = Path(source)
        destination_path = Path(destination)
        replaced.append((source_path, destination_path))
        should_fail = (
            (failure_stage == "results" and destination_path.name == "results.npz")
            or (failure_stage == "summary" and destination_path.name == "summary.md")
            or (failure_stage == "figure" and destination_path.suffix == ".png")
        )
        if failure_stage == "complete" and destination_path.name == "run_state.json":
            state = read_json(source_path)
            should_fail = state.get("lifecycle") == "complete"
        if should_fail:
            injected.append((source_path, destination_path))
            raise OSError(f"{failure_stage} final publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", record_replace)
    with pytest.raises(OSError, match=f"{failure_stage} final publication failure"):
        pipeline.run_prepared_task_decoding(run_directory)
    assert len(injected) == 1
    assert all(source.parent == destination.parent for source, destination in replaced)
    assert len({source for source, _ in replaced}) == len(replaced)
    assert all(not source.exists() for source, _ in replaced)
    assert not injected[0][0].exists()
    assert not any(
        path.name.startswith(".") and path.is_file()
        for path in run_directory.rglob("*")
    )
    state = read_json(run_directory / "run_state.json")
    assert state["lifecycle"] != "complete"
    assert not state["final_results_published"]
    expected_presence = {
        "results": (False, False, False),
        "summary": (True, False, False),
        "figure": (True, True, False),
        "complete": (True, True, True),
    }[failure_stage]
    assert (run_directory / "results.npz").exists() is expected_presence[0]
    assert (run_directory / "summary.md").exists() is expected_presence[1]
    assert bool(list((run_directory / "figures").glob("*.png"))) is expected_presence[2]


@pytest.mark.parametrize("failure_stage", ("summary", "figure", "complete"))
def test_resume_after_final_reporting_failure_publishes_only_missing_artifacts(
    monkeypatch,
    tmp_path,
    failure_stage,
):
    """Resume preserves immutable finals after a late reporting publication failure.

    The first execution publishes all artifacts before the chosen final boundary.
    Its resume must restore checkpoints and validate existing finals rather than
    decode again or replace immutable result, summary, or figure files.
    """
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    real_replace = os.replace
    initial_replacements: list[tuple[Path, Path]] = []

    def fail_one_final_replace(source, destination):
        """Inject one late artifact or final-complete replace failure."""
        source_path = Path(source)
        destination_path = Path(destination)
        initial_replacements.append((source_path, destination_path))
        should_fail = (
            failure_stage == "summary" and destination_path.name == "summary.md"
        ) or (failure_stage == "figure" and destination_path.suffix == ".png")
        if failure_stage == "complete" and destination_path.name == "run_state.json":
            should_fail = read_json(source_path).get("lifecycle") == "complete"
        if should_fail:
            raise OSError(f"{failure_stage} final publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", fail_one_final_replace)
    with pytest.raises(OSError, match=f"{failure_stage} final publication failure"):
        pipeline.run_prepared_task_decoding(run_directory)

    published_paths = [run_directory / "results.npz", run_directory / "summary.md"]
    published_paths.extend(sorted((run_directory / "figures").glob("*.png")))
    preserved_bytes = {
        path.relative_to(run_directory): path.read_bytes()
        for path in published_paths
        if path.exists()
    }
    assert (Path("results.npz")) in preserved_bytes
    checkpoint_bytes = {
        path.relative_to(run_directory): path.read_bytes()
        for path in (run_directory / "checkpoints").glob("*.npz")
    }
    assert checkpoint_bytes

    retry_replacements: list[tuple[Path, Path]] = []

    def record_retry_replace(source, destination):
        """Record the fault-free resume publications without overwriting finals."""
        source_path = Path(source)
        destination_path = Path(destination)
        retry_replacements.append((source_path, destination_path))
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", record_retry_replace)
    monkeypatch.setattr(pipeline.modeling, "decode_target", fail_if_called)
    pipeline.run_prepared_task_decoding(run_directory)

    for relative_path, content in preserved_bytes.items():
        assert (run_directory / relative_path).read_bytes() == content
    for relative_path, content in checkpoint_bytes.items():
        assert (run_directory / relative_path).read_bytes() == content
    retry_final_destinations = {
        destination.relative_to(run_directory)
        for _, destination in retry_replacements
        if destination.name in {"results.npz", "summary.md"}
        or destination.suffix == ".png"
    }
    assert Path("results.npz") not in retry_final_destinations
    if failure_stage == "summary":
        assert Path("summary.md") in retry_final_destinations
        assert len([path for path in retry_final_destinations if path.suffix == ".png"]) == 3
    elif failure_stage == "figure":
        assert retry_final_destinations == {
            Path("figures/categorical_balanced_accuracy.png"),
            Path("figures/categorical_auc.png"),
            Path("figures/numerical_r2.png"),
        }
    else:
        assert retry_final_destinations == set()
    assert read_json(run_directory / "run_state.json")["lifecycle"] == "complete"
    assert not any(source.exists() for source, _ in retry_replacements)


def test_complete_is_written_last_after_summary_log_and_required_family_png(monkeypatch, tmp_path):
    """Complete is the final atomic replacement after every required final artifact."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    replaced: list[tuple[Path, Path, str | None]] = []
    validation_positions: list[int] = []
    real_replace = os.replace
    real_validate = pipeline._validate_published_run_directory

    def record_replace(source, destination):
        """Record every final atomic replacement while retaining real publication."""
        source_path = Path(source)
        destination_path = Path(destination)
        lifecycle = None
        if destination_path.name == "run_state.json":
            lifecycle = str(read_json(source_path).get("lifecycle"))
        replaced.append((source_path, destination_path, lifecycle))
        return real_replace(source, destination)

    monkeypatch.setattr(pipeline.os, "replace", record_replace)
    monkeypatch.setattr(
        pipeline,
        "_validate_published_run_directory",
        lambda *args, **kwargs: validation_positions.append(len(replaced))
        or real_validate(*args, **kwargs),
    )
    pipeline.run_prepared_task_decoding(run_directory)
    final_destinations = [destination.name for _, destination, _ in replaced]
    assert "results.npz" in final_destinations
    assert "summary.md" in final_destinations
    assert any(destination.suffix == ".png" for _, destination, _ in replaced)
    complete_positions = [
        index
        for index, (_, destination, lifecycle) in enumerate(replaced)
        if destination.name == "run_state.json" and lifecycle == "complete"
    ]
    assert complete_positions == [len(replaced) - 1]
    assert validation_positions and validation_positions[-1] < complete_positions[0]
    assert read_json(run_directory / "run_state.json")["lifecycle"] == "complete"


def test_completed_run_writes_required_summary_log_and_minimal_family_pngs(monkeypatch, tmp_path):
    """WP5B completion writes the factual summary and exactly three default metric PNGs."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    pipeline.run_prepared_task_decoding(run_directory)
    summary = (run_directory / "summary.md").read_text(encoding="utf-8").lower()
    log = (run_directory / "run.log").read_text(encoding="utf-8").lower()
    for token in (
        "utc",
        "analysis goal",
        "task_variable_decoding",
        "synthetic-session",
        "launch command",
        "foreground",
        "run_session.py",
        "config",
        "parameter",
        "eligibility",
        "fold",
        "warning",
        "unavailable",
        "stage timing",
        "total timing",
        "saved output",
    ):
        assert token in summary
    assert re.search(r"20\d{2}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}z", summary)
    assert "stage" in log and "complete" in log
    figures = sorted((run_directory / "figures").glob("*.png"))
    assert {path.name for path in figures} == {
        "categorical_balanced_accuracy.png",
        "categorical_auc.png",
        "numerical_r2.png",
    }
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures)


@pytest.mark.parametrize(
    "owner_state",
    (
        "live",
        "pid_reused",
        "dead",
        "slurm_pending",
        "slurm_running",
        "slurm_terminal",
        "foreign_unresolved",
    ),
)
def test_execution_guard_matrix_is_bounded_and_owner_only(monkeypatch, tmp_path, owner_state):
    """Guard handling refuses active/foreign owners and replaces stale local owners."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    guard = run_directory / "execution_guard.json"
    owner = {
        "mode": "slurm" if owner_state.startswith("slurm") else "foreground",
        "host": platform.node(),
        "pid": 88,
        "start_token": "old",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": "1234" if owner_state.startswith("slurm") else None,
    }
    guard.write_text(json.dumps(owner), encoding="ascii")
    monkeypatch.setattr(
        pipeline,
        "_inspect_execution_owner",
        lambda *_args, **_kwargs: owner_state,
    )
    if owner_state in {
        "live",
        "slurm_pending",
        "slurm_running",
        "foreign_unresolved",
    }:
        before = guard.read_bytes()
        with pytest.raises(RuntimeError, match="guard|live|Slurm|foreign|receipt"):
            pipeline.run_prepared_task_decoding(run_directory)
        assert guard.read_bytes() == before
    else:
        monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
        with pytest.raises(AssertionError, match="guarded"):
            pipeline.run_prepared_task_decoding(run_directory)
        assert read_json(run_directory / "execution.json")["history"]


@pytest.mark.parametrize(
    ("start_token", "expected"),
    (("current", "live"), ("reused", "pid_reused"), (None, "dead")),
)
def test_inspect_execution_owner_uses_local_liveness_and_start_token(
    start_token,
    expected,
):
    """Owner inspection distinguishes a live PID from reuse and confirmed death."""
    owner = {
        "mode": "foreground",
        "host": platform.node(),
        "pid": 88,
        "start_token": "current",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    assert pipeline._inspect_execution_owner(
        owner,
        process_start_token=lambda _pid: start_token,
        scheduler_state=fail_if_called,
    ) == expected


@pytest.mark.parametrize(
    ("scheduler_value", "expected"),
    (("PENDING", "slurm_pending"), ("RUNNING", "slurm_running"), ("COMPLETED", "slurm_terminal")),
)
def test_inspect_execution_owner_uses_one_scheduler_state_for_slurm(
    scheduler_value,
    expected,
):
    """Slurm owner inspection maps one injected scheduler fact without polling."""
    owner = {
        "mode": "slurm",
        "host": "cluster-host",
        "pid": None,
        "start_token": None,
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": "1234",
    }
    assert pipeline._inspect_execution_owner(
        owner,
        process_start_token=fail_if_called,
        scheduler_state=lambda job_id: scheduler_value if job_id == "1234" else "UNKNOWN",
    ) == expected


def test_inspect_execution_owner_refuses_unresolved_foreign_host():
    """A foreign same-run owner is not guessed from a local PID lookup."""
    owner = {
        "mode": "foreground",
        "host": "other-host",
        "pid": 88,
        "start_token": "current",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    assert pipeline._inspect_execution_owner(
        owner,
        process_start_token=fail_if_called,
        scheduler_state=fail_if_called,
    ) == "foreign_unresolved"


def test_guard_claim_is_exclusive_and_orderly_release_keeps_history(monkeypatch, tmp_path):
    """Racing claims never clobber an owner, while that owner alone removes its guard."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    first_owner = {
        "mode": "foreground",
        "host": platform.node(),
        "pid": 1,
        "start_token": "a",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    wrong_owner = {**first_owner, "pid": 2, "start_token": "b"}
    first = pipeline._claim_execution_guard(run_directory, owner=first_owner)
    with pytest.raises(FileExistsError, match="guard|owner"):
        pipeline._claim_execution_guard(run_directory, owner=wrong_owner)
    pipeline._release_execution_guard(run_directory, owner=wrong_owner)
    assert first.exists()
    pipeline._release_execution_guard(run_directory, owner=first_owner)
    assert not first.exists()
    assert read_json(run_directory / "execution.json")["history"]


def test_concurrent_guard_claims_have_exactly_one_owner(monkeypatch, tmp_path):
    """Two simultaneous local claimants use exclusive creation rather than check-then-write."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    barrier = Barrier(2)
    outcomes: list[str] = []

    def claim(pid: int) -> None:
        """Synchronize one claimant then record whether atomic guard creation succeeded."""
        barrier.wait()
        owner = {
            "mode": "foreground",
            "host": platform.node(),
            "pid": pid,
            "start_token": str(pid),
            "started_at": "2026-10-07T00:00:00Z",
            "job_id": None,
        }
        try:
            pipeline._claim_execution_guard(run_directory, owner=owner)
        except FileExistsError:
            outcomes.append("lost")
        else:
            outcomes.append("won")

    workers = [Thread(target=claim, args=(pid,)) for pid in (1, 2)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=2.0)
    assert sorted(outcomes) == ["lost", "won"]


@pytest.mark.parametrize("resume", (False, True))
def test_detached_launch_uses_private_bridge_receipt_and_no_poll(
    monkeypatch,
    tmp_path,
    resume,
):
    """Detached new/resume atomically saves a live receipt and exact follow-up actions."""
    paths = write_session_inputs(tmp_path)
    monkeypatch.setattr(pipeline.results, "validate_scientific_source_cleanliness", lambda *_: None)
    monkeypatch.setattr(
        pipeline.results,
        "scientific_source_fingerprint",
        lambda *_: {"files": [], "fingerprint": "synthetic-clean"},
    )
    launched: dict[str, object] = {}

    class FakeProcess:
        """Minimal Popen return object retaining a stable child PID."""
        pid = 4321

    def fake_popen(argv, **kwargs):
        """Record one private prepared-execution launch without creating a child."""
        launched["argv"] = argv
        launched["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(pipeline, "_process_start_token", lambda _pid: "child-token")
    monkeypatch.setattr(pipeline, "inspect_task_decoding_status", fail_if_called)
    events: list[tuple[str, str]] = []
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def record_replace(source, destination):
        """Record the receipt's completed sibling publication without changing behavior."""
        source_path = Path(source)
        destination_path = Path(destination)
        replaced.append((source_path, destination_path))
        if destination_path.name == "local_launch.json":
            events.append(("receipt", str(destination_path)))
        return real_replace(source, destination)

    def record_print(*values, sep=" ", **_kwargs):
        """Record successful detached-launch guidance after receipt publication."""
        events.append(("print", sep.join(str(value) for value in values)))

    monkeypatch.setattr(run_session.os, "replace", record_replace)
    monkeypatch.setattr(run_session, "print", record_print, raising=False)
    if resume:
        prepared = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
        code = run_session.main(["resume", "--run-directory", str(prepared), "--detach"])
    else:
        code = run_session.main(["new", "--config", str(paths["config"]), "--detach", "--rerun"])
        prepared = Path(launched["kwargs"]["cwd"])
    assert code == 0
    assert launched["argv"][-3:] == ["_execute-prepared", "--run-directory", str(prepared)]
    assert launched["kwargs"]["stdin"] is subprocess.DEVNULL
    assert launched["kwargs"]["start_new_session"] is True
    assert launched["kwargs"]["stdout"] is launched["kwargs"]["stderr"] or launched[
        "kwargs"
    ]["stderr"] is subprocess.STDOUT
    receipt = read_json(prepared / "local_launch.json")
    assert receipt["mode"] == "detached"
    assert receipt["host"] == platform.node()
    assert receipt["pid"] == 4321
    assert receipt["start_token"] == "child-token"
    assert isinstance(receipt["started_at"], str) and receipt["started_at"]
    receipt_replacements = [
        (source, destination)
        for source, destination in replaced
        if destination == prepared / "local_launch.json"
    ]
    assert len(receipt_replacements) == 1
    receipt_source, receipt_destination = receipt_replacements[0]
    assert receipt_source.parent == receipt_destination.parent
    assert receipt_source != receipt_destination
    assert not receipt_source.exists()
    expected_resume, expected_status = expected_follow_up_commands(prepared)
    output = "\n".join(value for kind, value in events if kind == "print")
    assert str(prepared) in output
    assert expected_resume.strip() in output
    assert expected_status.strip() in output
    receipt_position = next(
        index for index, (kind, _) in enumerate(events) if kind == "receipt"
    )
    print_positions = [
        index for index, (kind, _value) in enumerate(events) if kind == "print"
    ]
    assert print_positions and receipt_position < min(print_positions)
    launch_count = len(
        [entry for entry in replaced if entry[1] == prepared / "local_launch.json"]
    )
    monkeypatch.setattr(pipeline, "_inspect_execution_owner", lambda *_a, **_k: "live")
    assert run_session.main(["resume", "--run-directory", str(prepared)]) != 0
    assert len(
        [entry for entry in replaced if entry[1] == prepared / "local_launch.json"]
    ) == launch_count


def test_dead_detached_receipt_is_read_only_until_explicit_resume_rechecks_owner(
    monkeypatch,
    tmp_path,
):
    """A child dying after its receipt leaves facts visible but never auto-restarts work."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    receipt = {
        "mode": "detached",
        "host": platform.node(),
        "pid": 4321,
        "start_token": "dead-child",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    (run_directory / "local_launch.json").write_text(json.dumps(receipt), encoding="ascii")
    calls: list[str] = []
    real_run_prepared = pipeline.run_prepared_task_decoding
    monkeypatch.setattr(
        pipeline,
        "_inspect_execution_owner",
        lambda *_args, **_kwargs: calls.append("inspect") or "dead",
    )
    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", fail_if_called)
    status = pipeline.inspect_task_decoding_status(run_directory, verify_results=False)
    assert status["lifecycle"] == "initialized"
    assert "dead" in json.dumps(status, sort_keys=True).lower()
    assert calls == ["inspect"]
    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", real_run_prepared)

    def inspected_guard(*_args, **_kwargs):
        """Require explicit resume to inspect the dead receipt a second time."""
        assert calls == ["inspect", "inspect"]
        raise AssertionError("guarded operation")

    monkeypatch.setattr(pipeline, "_claim_execution_guard", inspected_guard)
    with pytest.raises(AssertionError, match="guarded operation"):
        pipeline.run_prepared_task_decoding(run_directory)
    assert calls == ["inspect", "inspect"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux detached-session test")
def test_linux_actual_launcher_returns_before_prepared_child_completes(monkeypatch, tmp_path):
    """The actual CLI launcher returns before a terminal-independent prepared child completes."""
    paths = write_session_inputs(tmp_path)
    prepared = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    release = tmp_path / "release"
    child_state = prepared / "child_state.json"
    child_program = """
import json
import os
import pathlib
import sys
import time
state = pathlib.Path(sys.argv[1])
release = pathlib.Path(sys.argv[2])
temporary = state.with_suffix('.tmp')
temporary.write_text(json.dumps(
    {'lifecycle':'running','pid':os.getpid(),'session_id':os.getsid(0)}
))
os.replace(temporary, state)
deadline = time.monotonic() + 5.0
while not release.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(2)
    time.sleep(0.01)
temporary.write_text(json.dumps({'lifecycle':'complete','pid':os.getpid()}))
os.replace(temporary, state)
print('child complete', flush=True)
"""
    child_argv = [sys.executable, "-c", child_program, str(child_state), str(release)]
    receipt_path = run_session._launch_detached_prepared(
        run_directory=prepared,
        child_argv=child_argv,
    )
    receipt = read_json(prepared / "local_launch.json")
    assert receipt_path == prepared / "local_launch.json"
    deadline = time.monotonic() + 3.0
    while not child_state.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_facts = read_json(child_state)
    assert child_facts["lifecycle"] == "running"
    assert child_facts["session_id"] == child_facts["pid"]
    assert receipt["pid"] != os.getpid()
    release.write_text("release\n", encoding="ascii")
    deadline = time.monotonic() + 5.0
    while read_json(child_state)["lifecycle"] != "complete" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert read_json(child_state)["lifecycle"] == "complete"
    assert (prepared / "console.log").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "lifecycle",
    ("initialized", "running", "interrupted", "failed"),
)
def test_status_is_read_only_for_nonterminal_lifecycles(
    monkeypatch,
    tmp_path,
    lifecycle,
):
    """Status reads saved nonterminal state/log facts without source access or mutation."""
    run_directory = tmp_path / lifecycle
    run_directory.mkdir()
    (run_directory / "run_state.json").write_text(
        json.dumps(
            {
                "lifecycle": lifecycle,
                "warnings": ["w"],
                "last_error": "e",
                "completed_targets": [],
                "total_targets": ["current_action"],
                "final_results_published": False,
            }
        )
    )
    (run_directory / "execution.json").write_text(json.dumps({"pid": 9, "history": []}))
    (run_directory / "run.log").write_text("durable log\n")
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", fail_if_called)
    monkeypatch.setattr(pipeline, "_poll_scheduler", fail_if_called)
    monkeypatch.setattr(pipeline.results, "load_task_decoding_run", fail_if_called)
    status = pipeline.inspect_task_decoding_status(run_directory, verify_results=False)
    assert status["lifecycle"] == lifecycle
    assert status["log_path"] == run_directory / "run.log"
    assert status["recovery_command"]


def test_status_verify_results_uses_real_loader_for_complete_run(monkeypatch, tmp_path):
    """Complete status optionally validates a real WP5A result without source or execution work."""
    inputs = write_input_fixture(tmp_path)
    run_directory = tmp_path / "complete"
    save_run_fixture(run_directory, inputs, regularization_mode="fixed")
    (run_directory / "run_state.json").write_text(
        json.dumps(
            {
                "lifecycle": "complete",
                "warnings": ["retained warning"],
                "last_error": "",
                "completed_targets": ["current_action", "relative_doubt"],
                "total_targets": ["current_action", "relative_doubt"],
                "final_results_published": True,
            }
        )
    )
    (run_directory / "execution.json").write_text(json.dumps({"pid": 9, "history": []}))
    (run_directory / "run.log").write_text("durable complete log\n")
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", fail_if_called)
    before = {path: path.read_bytes() for path in run_directory.iterdir() if path.is_file()}
    status = pipeline.inspect_task_decoding_status(run_directory, verify_results=True)
    assert status["lifecycle"] == "complete"
    assert status["result_validation"] == "valid"
    assert before == {path: path.read_bytes() for path in before}


def test_status_reports_complete_artifact_validation_failure_without_writing(
    monkeypatch,
    tmp_path,
):
    """An absent/corrupt complete result reports validation failure without repair writes."""
    run_directory = tmp_path / "corrupt-complete"
    run_directory.mkdir()
    (run_directory / "run_state.json").write_text(
        json.dumps({"lifecycle": "complete", "final_results_published": True})
    )
    (run_directory / "execution.json").write_text(json.dumps({"pid": 9, "history": []}))
    (run_directory / "run.log").write_text("durable log\n")
    before = {path: path.read_bytes() for path in run_directory.iterdir() if path.is_file()}
    status = pipeline.inspect_task_decoding_status(run_directory, verify_results=True)
    assert status["lifecycle"] == "complete"
    assert status["result_validation"] == "invalid"
    assert before == {path: path.read_bytes() for path in before}


def test_cli_routes_public_and_private_modes_through_one_prepare_execute_seam(
    monkeypatch,
    tmp_path,
):
    """CLI returns nonzero on setup failure and private execute never reparses science."""
    paths = write_session_inputs(tmp_path)
    calls: list[tuple[str, Path]] = []
    prepared = tmp_path / "prepared"
    monkeypatch.setattr(
        pipeline,
        "prepare_task_decoding_run",
        lambda config_path, rerun, execution_mode: calls.append(
            ("prepare", Path(config_path))
        ) or prepared,
    )
    monkeypatch.setattr(
        pipeline,
        "run_prepared_task_decoding",
        lambda directory: calls.append(("execute", Path(directory))),
    )
    assert run_session.main(["_prepare", "--config", str(paths["config"])]) == 0
    assert run_session.main(["_execute-prepared", "--run-directory", str(prepared)]) == 0
    assert calls == [("prepare", paths["config"]), ("execute", prepared)]
    monkeypatch.setattr(
        pipeline,
        "prepare_task_decoding_run",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("bad config")),
    )
    assert run_session.main(["new", "--config", str(paths["config"])]) != 0


def test_cli_dry_run_foreground_resume_status_and_private_bridge_routes(
    monkeypatch,
    tmp_path,
):
    """CLI keeps scientific preparation/execution in the pipeline's two owned seams."""
    paths = write_session_inputs(tmp_path)
    prepared = tmp_path / "task_variable_decoding_cli"
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        pipeline,
        "plan_task_decoding_session",
        lambda config_path: calls.append(("dry-run", Path(config_path)))
        or {"target_names": ["current_action"]},
    )
    monkeypatch.setattr(
        pipeline,
        "prepare_task_decoding_run",
        lambda config_path, rerun, execution_mode: calls.append(
            ("prepare", (Path(config_path), rerun, execution_mode))
        )
        or prepared,
    )
    monkeypatch.setattr(
        pipeline,
        "run_prepared_task_decoding",
        lambda run_directory: calls.append(("execute", Path(run_directory))),
    )
    monkeypatch.setattr(
        pipeline,
        "inspect_task_decoding_status",
        lambda run_directory, verify_results=False: calls.append(
            ("status", (Path(run_directory), verify_results))
        )
        or {"lifecycle": "initialized"},
    )
    assert run_session.main(["dry-run", "--config", str(paths["config"])]) == 0
    assert run_session.main(["new", "--config", str(paths["config"]), "--rerun"]) == 0
    assert run_session.main(["resume", "--run-directory", str(prepared)]) == 0
    assert run_session.main(["status", "--run-directory", str(prepared)]) == 0
    assert run_session.main(["_prepare", "--config", str(paths["config"])]) == 0
    assert run_session.main(["_execute-prepared", "--run-directory", str(prepared)]) == 0
    assert calls == [
        ("dry-run", paths["config"]),
        ("prepare", (paths["config"], True, "foreground")),
        ("execute", prepared),
        ("execute", prepared),
        ("status", (prepared, False)),
        ("prepare", (paths["config"], False, "foreground")),
        ("execute", prepared),
    ]


@pytest.mark.parametrize("exception", (RuntimeError("failure"), KeyboardInterrupt()))
def test_cli_foreground_failures_and_interruptions_return_nonzero(monkeypatch, tmp_path, exception):
    """Foreground new execution returns nonzero for unexpected failure or interruption."""
    paths = write_session_inputs(tmp_path)
    prepared = tmp_path / "task_variable_decoding_failure"
    monkeypatch.setattr(
        pipeline,
        "prepare_task_decoding_run",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        pipeline,
        "run_prepared_task_decoding",
        lambda _directory: (_ for _ in ()).throw(exception),
    )
    assert run_session.main(["new", "--config", str(paths["config"])]) != 0


def test_cli_detached_and_mocked_slurm_submission_only_report_launch_acceptance(
    monkeypatch,
    tmp_path,
):
    """Detached and scheduled handoffs report accepted launch without scientific completion."""
    paths = write_session_inputs(tmp_path)
    prepared = tmp_path / "task_variable_decoding_detached"
    calls: list[object] = []
    monkeypatch.setattr(
        pipeline,
        "prepare_task_decoding_run",
        lambda _config_path, rerun, execution_mode: calls.append(
            ("prepare", rerun, execution_mode)
        )
        or prepared,
    )
    monkeypatch.setattr(
        pipeline,
        "run_prepared_task_decoding",
        lambda *_args, **_kwargs: calls.append("execute"),
    )
    monkeypatch.setattr(
        run_session,
        "_launch_detached_prepared",
        lambda *, run_directory, child_argv: calls.append(("detached", run_directory))
        or run_directory / "local_launch.json",
    )
    assert run_session.main(["new", "--config", str(paths["config"]), "--detach"]) == 0
    assert calls == [("prepare", False, "detached"), ("detached", prepared)]
    calls.clear()
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-job")
    assert run_session.main(["_execute-prepared", "--run-directory", str(prepared)]) == 0
    assert calls == ["execute"]


def test_entry_sets_thread_limits_before_lazy_numerical_pipeline_import():
    """A fresh interpreter sets all thread limits before a CLI route imports pipeline/numerics."""
    script = """
import builtins
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.pop(key, None)
seen = []
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('numpy', 'scipy', 'sklearn', 'src.neural_analysis.task_decoding.pipeline')):
        assert all(os.environ.get(key) == '1' for key in
                   ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
        seen.append(name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from src.neural_analysis.task_decoding import run_session
assert all(os.environ.get(key) == '1' for key in
           ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
try:
    run_session.main(['_execute-prepared', '--run-directory', '/missing'])
except Exception:
    pass
assert any('pipeline' in name for name in seen)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
