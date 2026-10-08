"""WP5B single-session preparation, execution, and CLI contracts.

The fixtures are intentionally small but scientifically coherent: six
contiguous behavioral blocks supply three grouped folds, both configured probes
have ordinary metadata, and the assembly test uses real model records and the
released WP5A result serializer.  No test reads experimental data, launches a
scheduler, or allocates a large neural array.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import re
import signal
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from threading import Barrier, Event, Thread

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


def set_condition_names(paths: dict[str, Path], condition_names: tuple[str, ...]) -> None:
    """Persist a selected canonical condition list in one test configuration.

    Parameters
    ----------
    paths : dict[str, pathlib.Path]
        Session fixture mapping containing the JSON configuration path.
    condition_names : tuple[str, ...]
        Nonempty canonical condition identifiers in requested input order.

    Returns
    -------
    None
        Rewrites only the test-owned configuration JSON.
    """
    payload = read_json(paths["config"])
    payload["condition_names"] = list(condition_names)
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


def write_matching_detached_receipt(run_directory: Path) -> None:
    """Publish the receipt a real detached parent writes before child execution.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared detached run directory receiving ``local_launch.json``.

    Returns
    -------
    None
        Atomically writes the current process identity used by in-process CLI
        bridge tests. The values are execution facts without physical units.
    """
    run_session._atomic_receipt(
        run_directory / "local_launch.json",
        {
            "mode": "detached",
            "host": platform.node(),
            "pid": os.getpid(),
            "start_token": pipeline._process_start_token(os.getpid()),
            "started_at": "2026-10-07T00:00:00Z",
            "job_id": None,
        },
    )


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
    assert tuple(plan["condition_names"]) == ("all",)
    assert plan["fit_count"] == 2 * 3 * 8 * 3 * 2
    assert plan["tensor_allocation_bytes"] > 0
    assert "synthetic dirty source" in plan["source_cleanliness"]
    assert plan["analysis_version"] == pipeline.decoding_config.ANALYSIS_VERSION
    assert plan["regularization_mode"] == "fixed"
    assert plan["resource_envelope"] == {
        "full_trial_count": 12,
        "tensor_trial_count": 12,
        "pfc_unit_count": 2,
        "hpc_unit_count": 2,
        "time_bin_count": 8,
        "target_count": 2,
        "condition_count": 1,
        "outer_fold_count": 3,
        "inner_fold_count": 3,
        "coefficient_feature_capacity": 4,
        "categorical_fit_count": 144,
        "numerical_fit_count": 144,
        "tensor_allocation_bytes": plan["tensor_allocation_bytes"],
    }
    assert plan["target_diagnostics"] == [
        {
            "target_identifier": "current_action",
            "target_family": "categorical",
            "eligible_trial_count": 12,
            "block_count": 6,
            "class_counts": {"0": 6, "1": 6},
            "value_min": None,
            "value_max": None,
            "outer_split_available": True,
            "outer_split_unavailable_reason": None,
        },
        {
            "target_identifier": "relative_doubt",
            "target_family": "numerical",
            "eligible_trial_count": 12,
            "block_count": 6,
            "class_counts": {},
            "value_min": 1.0,
            "value_max": 1.0,
            "outer_split_available": False,
            "outer_split_unavailable_reason": "constant target",
        },
    ]


def test_condition_resolved_plan_reports_expanded_fit_envelope_and_diagnostics(
    monkeypatch,
    tmp_path,
) -> None:
    """Dry run should count every configured condition-target analysis cell."""
    paths = write_session_inputs(tmp_path)
    set_condition_names(
        paths,
        ("stay", "all", "incorrect", "correct_rewarded", "omission", "switch"),
    )
    monkeypatch.setattr(
        pipeline.results,
        "validate_scientific_source_cleanliness",
        lambda *_args: None,
    )

    plan = pipeline.plan_task_decoding_session(paths["config"])

    assert tuple(plan["condition_names"]) == (
        "all",
        "correct_rewarded",
        "omission",
        "incorrect",
        "switch",
        "stay",
    )
    assert plan["fit_count"] == 6 * 2 * 3 * 8 * 3 * 2
    assert plan["resource_envelope"]["condition_count"] == 6
    assert plan["resource_envelope"]["categorical_fit_count"] == 6 * 144
    assert plan["resource_envelope"]["numerical_fit_count"] == 6 * 144
    diagnostics = plan["condition_target_diagnostics"]
    assert len(diagnostics) == 6 * 2
    assert [
        (entry["condition_identifier"], entry["target_identifier"])
        for entry in diagnostics
    ] == [
        (condition, target)
        for condition in plan["condition_names"]
        for target in plan["target_names"]
    ]
    condition_counts = plan["condition_trial_counts"]
    assert condition_counts["all"] == 12
    assert set(condition_counts) == set(plan["condition_names"])


def test_condition_target_common_masks_intersect_before_model_splitting(tmp_path) -> None:
    """Each decoder should receive the condition/target/common-row intersection."""
    paths = write_session_inputs(tmp_path)
    set_condition_names(paths, ("all", "correct_rewarded", "incorrect"))
    config = decoding_config.load_task_decoding_config(paths["config"])
    trial_table = pd.read_csv(paths["augmented"])
    target_table = targets.build_target_table(trial_table, config)
    common_rows = np.array([0, 1, 2, 4, 5, 7, 8, 10, 11], dtype=np.int64)

    observed = pipeline._condition_target_common_masks(
        config,
        trial_table,
        target_table,
        common_rows,
    )

    shared = pipeline.conditions.build_condition_masks(
        trial_table,
        config.condition_names,
    )
    assert tuple(observed) == tuple(
        (condition, target)
        for condition in config.condition_names
        for target in config.target_names
    )
    for condition, target in observed:
        expected = (
            shared[condition][common_rows]
            & target_table[f"{target}_valid"].to_numpy(dtype=bool)[common_rows]
        )
        np.testing.assert_array_equal(observed[(condition, target)], expected)


def test_condition_target_checkpoint_identity_is_unambiguous(tmp_path) -> None:
    """Condition-target checkpoints should have stable state keys and safe filenames."""
    run_directory = tmp_path / "run"

    assert pipeline._condition_target_key("correct_rewarded", "current_action") == (
        "correct_rewarded::current_action"
    )
    assert pipeline._condition_target_checkpoint_path(
        run_directory,
        "correct_rewarded",
        "current_action",
    ) == run_directory / "checkpoints" / "correct_rewarded--current_action.npz"
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
        "run_batch.py",
        "pipeline.py",
        "checkpoints", "figures", "run.log",
    }
    assert required <= {member.name for member in run_directory.iterdir()}
    assert (run_directory / "trial_feature_params.json").read_bytes() == paths[
        "feature_parameters"
    ].read_bytes()
    assert (run_directory / "run_session.py").read_bytes() == Path(
        run_session.__file__
    ).read_bytes()
    assert (run_directory / "run_batch.py").read_bytes() == Path(
        pipeline.__file__
    ).with_name("run_batch.py").read_bytes()
    assert (run_directory / "pipeline.py").read_bytes() == Path(pipeline.__file__).read_bytes()
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
    if route == "detached_private":
        write_matching_detached_receipt(run_directory)
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
    if route == "detached_private":
        write_matching_detached_receipt(run_directory)
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
        session_id="synthetic-session",
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


def test_wp5b_private_assembly_requires_explicit_nonempty_session_id(tmp_path):
    """Assembly cannot invent session provenance when direct callers omit session ID.

    The private seam receives the canonical resolved metadata session identifier
    explicitly.  Its signature must not provide a fallback value, omitted
    calls must be rejected by Python, and an empty identity must be rejected
    before result metadata can claim a fabricated session.
    """
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    assembly_kwargs = {
        "config": model_inputs["config"],
        "target_table": model_inputs["target_table"],
        "rate_tensors": model_inputs["rate_tensors"],
        "region_activity_metadata": {
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": {}},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        "ordered_target_results": model_inputs["records"],
        "stage_timing_seconds": {"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        "input_manifest": {"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        "scientific_source": {"fingerprint": "source"},
        "execution_provenance": {"mode": "foreground"},
        "run_fingerprint": "fingerprint",
    }
    signature = inspect.signature(pipeline._assemble_run_result_payload)
    assert signature.parameters["session_id"].default is inspect.Parameter.empty
    with pytest.raises(TypeError, match="session_id"):
        pipeline._assemble_run_result_payload(**assembly_kwargs)
    for invalid_session_id in ("", "   "):
        with pytest.raises(ValueError, match="session_id|session"):
            pipeline._assemble_run_result_payload(
                **assembly_kwargs,
                session_id=invalid_session_id,
            )


@pytest.mark.parametrize("inner", (False, True))
def test_wp5b_compact_reason_mapping_rejects_unknown_model_prose(inner):
    """Only reviewed model reasons may become persisted compact reason codes.

    Parameters are deliberately unrecognized outer and inner prose so a future
    model/API change cannot be silently mislabeled as an estimator or candidate
    failure in the fixed saved schema.
    """
    with pytest.raises(ValueError, match="reason|unknown|unsupported"):
        pipeline._compact_reason_code("unrecognized model failure prose", inner=inner)


@pytest.mark.parametrize(
    ("reason", "inner"),
    (
        ("unexpected parser bug in regional training features: PFC", False),
        ("unavailable regional training features: PFC, unexpected", False),
        ("unavailable inner regional training features: PFC, unexpected", True),
    ),
)
def test_wp5b_compact_reason_mapping_rejects_unreviewed_regional_prose(reason, inner):
    """A familiar regional substring cannot disguise an unreviewed model reason."""
    with pytest.raises(ValueError, match="reason|unknown|unsupported"):
        pipeline._compact_reason_code(reason, inner=inner)


@pytest.mark.parametrize(
    ("reason", "inner", "expected"),
    (
        ("convergence_warning", False, "fit_convergence_failure"),
        ("convergence_warning", True, "candidate_fit_failed"),
        ("invalid coefficient or intercept shape", False, "outer_estimator_failure"),
        ("invalid coefficient or intercept shape", True, "candidate_fit_failed"),
        ("non-finite coefficient or intercept", False, "outer_estimator_failure"),
        ("non-finite coefficient or intercept", True, "candidate_fit_failed"),
        (
            "categorical estimator must expose binary classes including positive class 1",
            False,
            "outer_estimator_failure",
        ),
        (
            "categorical estimator must expose binary classes including positive class 1",
            True,
            "candidate_fit_failed",
        ),
        ("invalid probability shape", False, "outer_estimator_failure"),
        ("invalid probability shape", True, "candidate_fit_failed"),
        ("non-finite probability or score", False, "outer_estimator_failure"),
        ("non-finite probability or score", True, "candidate_fit_failed"),
        ("positive class is ambiguous", False, "outer_estimator_failure"),
        ("positive class is ambiguous", True, "candidate_fit_failed"),
        (
            "positive_scores must be finite and align with target_values.",
            False,
            "outer_estimator_failure",
        ),
        (
            "positive_scores must be finite and align with target_values.",
            True,
            "candidate_fit_failed",
        ),
        (
            "Categorical metrics require held-out values from both classes.",
            False,
            "outer_estimator_failure",
        ),
        (
            "Categorical metrics require held-out values from both classes.",
            True,
            "candidate_fit_failed",
        ),
        ("non-finite prediction or score", False, "outer_estimator_failure"),
        ("non-finite prediction or score", True, "candidate_fit_failed"),
        (
            "constant, singleton, or non-finite held-out numerical target",
            False,
            "outer_estimator_failure",
        ),
        (
            "constant, singleton, or non-finite held-out numerical target",
            True,
            "candidate_fit_failed",
        ),
        ("inner split unavailable: constant target", False, "inner_plan_unavailable"),
        ("inner split unavailable: constant target", True, "inner_plan_unavailable"),
        (
            "unavailable inner regional training features: PFC",
            True,
            "inner_candidate_unavailable",
        ),
        (
            "unavailable outer regional training features: HPC",
            False,
            "outer_hpc_features_unavailable",
        ),
        (
            "unavailable regional training features: PFC, HPC",
            False,
            "outer_features_unavailable",
        ),
        ("no valid inner tuning candidate", False, "no_valid_tuning_candidate"),
        ("non-finite inner score", True, "nonfinite_inner_score"),
    ),
)
def test_wp5b_reviewed_model_reasons_map_to_declared_compact_codes(reason, inner, expected):
    """Every reviewed modeling invalidity maps losslessly to one declared code."""
    compact = pipeline._compact_reason_code(reason, inner=inner)
    assert compact == expected
    assert compact in results._COMPACT_REASON_CODES


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
    assert calls[-4:] == ["npz", "figures", "summary", "validate"]


def test_summary_receives_final_resource_snapshot_after_figures(monkeypatch, tmp_path):
    """Human summary reports the complete post-figure resource measurement."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    events: list[str] = []
    observed_summary_evidence: dict[str, object] = {}
    original_snapshot = pipeline.resource_usage.ResourceUsageTracker.snapshot

    def record_snapshot(self, *, status, completed_targets, stage_timing_seconds=None):
        """Publish real evidence while recording its lifecycle order."""
        payload = original_snapshot(
            self,
            status=status,
            completed_targets=completed_targets,
            stage_timing_seconds=stage_timing_seconds,
        )
        events.append(f"snapshot:{status}")
        return payload

    def record_summary(*_args, resource_evidence, **_kwargs):
        """Capture the exact resource mapping supplied to the human summary."""
        observed_summary_evidence.update(resource_evidence)
        events.append("summary")

    monkeypatch.setattr(
        pipeline.resource_usage.ResourceUsageTracker,
        "snapshot",
        record_snapshot,
    )
    monkeypatch.setattr(
        pipeline.results,
        "save_task_decoding_run",
        lambda *_args, **_kwargs: events.append("npz"),
    )
    monkeypatch.setattr(pipeline, "_write_run_summary", record_summary)
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
    monkeypatch.setattr(pipeline, "_assemble_run_result_payload", lambda **_: ({}, {}))

    pipeline.run_prepared_task_decoding(run_directory)

    assert observed_summary_evidence["status"] == "complete"
    assert "reporting" in observed_summary_evidence["stage_timing_seconds"]
    assert events.index("figures") < events.index("snapshot:complete")
    assert events.index("snapshot:complete") < events.index("summary")
    assert events.index("summary") < events.index("validate")


def test_each_target_checkpoint_publishes_partial_resource_evidence(monkeypatch, tmp_path):
    """A completed checkpoint has matching atomic evidence before the next target starts."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    model_inputs = make_coherent_model_records(paths)
    patch_coherent_execution(monkeypatch, model_inputs)
    records = {record.target_identifier: record for record in model_inputs["records"]}
    observed_partial: dict[str, object] = {}

    def decode_with_checkpoint_observation(*, target_identifier, **_kwargs):
        """Inspect first-target evidence immediately before decoding the second."""
        if target_identifier == "relative_doubt":
            observed_partial.update(read_json(run_directory / "resource_usage.json"))
        return records[target_identifier]

    monkeypatch.setattr(pipeline.modeling, "decode_target", decode_with_checkpoint_observation)
    reporting_events: list[str] = []
    patch_final_reporting_boundaries(monkeypatch, reporting_events)
    monkeypatch.setattr(pipeline, "_assemble_run_result_payload", lambda **_: ({}, {}))

    pipeline.run_prepared_task_decoding(run_directory)

    assert observed_partial["status"] == "running"
    assert observed_partial["completed_targets"] == ["current_action"]
    assert set(observed_partial["target_measurements"]) == {"current_action"}
    assert observed_partial["target_measurements"]["current_action"]["timing_available"]
    complete = read_json(run_directory / "resource_usage.json")
    assert complete["status"] == "complete"
    assert complete["completed_targets"] == ["current_action", "relative_doubt"]
    assert set(complete["target_measurements"]) == {
        "current_action",
        "relative_doubt",
    }
    assert complete["peak_rss_bytes"] > 0
    assert complete["wall_time_seconds"] >= 0.0
    assert complete["user_cpu_seconds"] >= 0.0
    assert complete["system_cpu_seconds"] >= 0.0


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
    assert reporting_events == ["npz", "figures", "summary", "validate"]


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
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
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
    usage = read_json(run_directory / "resource_usage.json")
    assert usage["status"] == "failed"
    assert usage["completed_targets"] == ["current_action"]
    restored = usage["target_measurements"]["current_action"]
    assert restored["timing_available"] is False
    assert restored["total_seconds"] is None


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
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
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
    usage = read_json(run_directory / "resource_usage.json")
    assert usage["status"] == "interrupted"
    assert usage["completed_targets"] == ["current_action"]
    assert usage["target_measurements"]["current_action"]["timing_available"] is False


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
        "summary": (True, False, True),
        "figure": (True, False, False),
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
        assert not [path for path in retry_final_destinations if path.suffix == ".png"]
    elif failure_stage == "figure":
        assert retry_final_destinations == {
            Path("figures/categorical_balanced_accuracy.png"),
            Path("figures/categorical_auc.png"),
            Path("figures/numerical_r2.png"),
            Path("summary.md"),
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


@pytest.mark.parametrize("owner_file", ("local_launch.json", "execution_guard.json"))
def test_live_owner_rejection_cannot_mutate_an_active_run_state(
    monkeypatch,
    tmp_path,
    owner_file,
):
    """A losing receipt/guard contender must not publish failure for the owner."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    active_owner = {
        "mode": "detached",
        "host": platform.node(),
        "pid": 4321,
        "start_token": "active-owner",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    owner_path = run_directory / owner_file
    owner_path.write_text(json.dumps(active_owner), encoding="ascii")
    state_path = run_directory / "run_state.json"
    state_before = state_path.read_bytes()
    owner_before = owner_path.read_bytes()
    monkeypatch.setattr(
        pipeline,
        "_inspect_execution_owner",
        lambda *_args, **_kwargs: "live",
    )
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)

    with pytest.raises(RuntimeError, match="active|live|owner|receipt|guard"):
        pipeline.run_prepared_task_decoding(run_directory)

    assert state_path.read_bytes() == state_before
    assert owner_path.read_bytes() == owner_before
    assert not (run_directory / "resource_usage.json").exists()
    expected_event = "receipt_live" if owner_file == "local_launch.json" else "guard_live"
    assert read_json(run_directory / "execution.json")["history"][-1]["event"] == expected_event


def test_atomic_guard_claim_race_cannot_mutate_the_winning_run_state(monkeypatch, tmp_path):
    """Losing exclusive guard creation must leave the winner's lifecycle untouched."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    state_path = run_directory / "run_state.json"
    execution_path = run_directory / "execution.json"
    state_before = state_path.read_bytes()
    execution_before = execution_path.read_bytes()
    monkeypatch.setattr(
        pipeline,
        "_claim_execution_guard",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileExistsError("synthetic guard race")
        ),
    )
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)

    with pytest.raises(RuntimeError, match="guard|owner|race"):
        pipeline.run_prepared_task_decoding(run_directory)

    assert state_path.read_bytes() == state_before
    assert execution_path.read_bytes() == execution_before
    assert not (run_directory / "resource_usage.json").exists()


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


def test_wp11_private_prepare_accepts_explicit_slurm_mode_and_prints_directory(
    monkeypatch,
    tmp_path,
    capsys,
):
    """The cluster wrapper prepares through the same runner with Slurm provenance."""
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="ascii")
    prepared = tmp_path / "session with spaces" / "analysis_runs" / "exact-run"
    calls: list[tuple[Path, bool, str]] = []
    monkeypatch.setattr(
        pipeline,
        "prepare_task_decoding_run",
        lambda path, rerun, execution_mode: calls.append(
            (Path(path), rerun, execution_mode)
        )
        or prepared,
    )

    return_code = run_session.main(
        [
            "_prepare",
            "--config",
            str(config_path),
            "--execution-mode",
            "slurm",
        ]
    )

    assert return_code == 0
    assert calls == [(config_path, False, "slurm")]
    assert capsys.readouterr().out.strip() == str(prepared)


def test_wp11_sigterm_becomes_existing_keyboard_interruption_and_restores_handler(
    monkeypatch,
    tmp_path,
):
    """TERM reaches the durable pipeline interruption boundary, never false completion."""
    prepared = tmp_path / "exact-run"
    prepared.mkdir()
    original_handler = object()
    installed: dict[int, object] = {signal.SIGTERM: original_handler}
    calls: list[Path] = []

    def fake_signal(signal_number: int, handler: object) -> object:
        """Install or restore a process handler without changing pytest signals."""
        previous = installed.get(signal_number, signal.SIG_DFL)
        installed[signal_number] = handler
        return previous

    class FakePipeline:
        """Deliver TERM while execution is inside the prepared-run boundary."""

        @staticmethod
        def run_prepared_task_decoding(run_directory: Path) -> None:
            calls.append(Path(run_directory))
            handler = installed[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)
            raise AssertionError("SIGTERM handler returned")

    monkeypatch.setattr(run_session.signal, "signal", fake_signal)

    with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
        run_session._execute_prepared_with_signal_handling(prepared, FakePipeline)

    assert calls == [prepared]
    assert installed[signal.SIGTERM] is original_handler


def test_wp11_slurm_execution_checks_effective_memory_before_spike_load(
    monkeypatch,
    tmp_path,
):
    """The active allocation guard uses its effective limit before large arrays."""
    slurm = importlib.import_module("src.neural_analysis.task_decoding.slurm")
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="slurm")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(slurm, "effective_memory_budget_bytes", lambda: 1)
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)

    with pytest.raises(ValueError, match="50%|memory"):
        pipeline.run_prepared_task_decoding(run_directory)

    state = read_json(run_directory / "run_state.json")
    assert state["lifecycle"] == "failed"
    assert not (run_directory / "results.npz").exists()


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


def test_wp5b_detached_child_consumes_its_matching_receipt_before_guard(
    monkeypatch,
    tmp_path,
):
    """A detached child must consume its own published receipt, then claim its guard."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    execution = read_json(run_directory / "execution.json")
    owner = {
        "mode": "detached",
        "host": platform.node(),
        "pid": os.getpid(),
        "start_token": pipeline._process_start_token(os.getpid()),
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    (run_directory / "local_launch.json").write_text(json.dumps(owner), encoding="utf-8")
    real_claim = pipeline._claim_execution_guard

    def claim_after_receipt(directory, *, owner):
        """Prove the receipt remains durable until its child claims the guard."""
        assert (directory / "local_launch.json").exists()
        claimed = real_claim(directory, owner=owner)
        assert (directory / "execution_guard.json").exists()
        assert (directory / "local_launch.json").exists()
        return claimed

    def fail_after_handoff(*_args, **_kwargs):
        """Stop after checking that claim and receipt handoff leave no unowned gap."""
        assert (run_directory / "execution_guard.json").exists()
        assert not (run_directory / "local_launch.json").exists()
        raise AssertionError("guarded operation")

    monkeypatch.setattr(pipeline, "_current_owner", lambda _execution: owner)
    monkeypatch.setattr(pipeline, "_claim_execution_guard", claim_after_receipt)
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_after_handoff)
    with pytest.raises(AssertionError, match="guarded operation"):
        pipeline.run_prepared_task_decoding(run_directory)
    assert execution["mode"] == "detached"


@pytest.mark.parametrize("initial_receipt", ("absent", "mismatched"))
def test_wp5b_private_detached_child_waits_for_matching_launch_receipt(
    monkeypatch,
    tmp_path,
    initial_receipt,
):
    """The private detached bridge cannot enter execution before its receipt exists.

    The child runs in a thread only to make the parent/child interleaving
    deterministic: the execution seam records entry, the test first proves no
    entry without a receipt, then atomically publishes the current process's
    matching PID/start-token receipt and requires a successful bridge return.
    """
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    receipt_path = run_directory / "local_launch.json"
    entered = Event()
    observed_receipts: list[bool] = []
    codes: list[int] = []

    def record_execution(directory: Path) -> None:
        """Record whether the private bridge entered only after receipt publication."""
        assert directory == run_directory
        observed_receipts.append(receipt_path.is_file())
        entered.set()

    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", record_execution)
    monkeypatch.setattr(run_session, "_pipeline_module", lambda: pipeline)
    if initial_receipt == "mismatched":
        run_session._atomic_receipt(
            receipt_path,
            {
                "mode": "detached",
                "host": platform.node(),
                "pid": os.getpid(),
                "start_token": "not-the-child-token",
                "started_at": "2026-10-07T00:00:00Z",
                "job_id": None,
            },
        )
    child = Thread(
        target=lambda: codes.append(
            run_session.main(["_execute-prepared", "--run-directory", str(run_directory)])
        )
    )
    child.start()
    assert not entered.wait(timeout=0.2)
    run_session._atomic_receipt(
        receipt_path,
        {
            "mode": "detached",
            "host": platform.node(),
            "pid": os.getpid(),
            "start_token": pipeline._process_start_token(os.getpid()),
            "started_at": "2026-10-07T00:00:00Z",
            "job_id": None,
        },
    )
    child.join(timeout=1.0)
    assert not child.is_alive()
    assert codes == [0]
    assert observed_receipts == [True]


def test_wp5b_second_detached_resume_refuses_before_popen(monkeypatch, tmp_path):
    """An active detached receipt rejects immediate resume without starting another child."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    (run_directory / "local_launch.json").write_text(
        json.dumps(
            {
                "mode": "detached",
                "host": platform.node(),
                "pid": os.getpid(),
                "start_token": pipeline._process_start_token(os.getpid()),
                "started_at": "2026-10-07T00:00:00Z",
                "job_id": None,
            }
        ),
        encoding="utf-8",
    )
    calls: list[object] = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: calls.append("popen"))
    assert run_session.main(["resume", "--run-directory", str(run_directory), "--detach"]) != 0
    assert calls == []


@pytest.mark.parametrize("wait_timeout", (False, True))
def test_wp5b_receipt_replace_failure_terminates_child(monkeypatch, tmp_path, wait_timeout):
    """Receipt failure terminates/reaps a child, escalating to kill after bounded wait timeout."""
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    calls: list[str] = []

    class FakeProcess:
        """Bounded fake child recording termination and wait after receipt failure."""

        pid = 4321

        def terminate(self) -> None:
            """Record child termination requested by failed parent handoff."""
            calls.append("terminate")

        def wait(self, timeout: float) -> int:
            """Record bounded reaping of a child whose receipt could not publish."""
            assert timeout > 0
            calls.append("wait")
            if wait_timeout and calls.count("wait") == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return 0

        def kill(self) -> None:
            """Record escalation when graceful bounded termination does not finish."""
            calls.append("kill")

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: FakeProcess())
    monkeypatch.setattr(run_session, "_pipeline_module", lambda: pipeline)
    monkeypatch.setattr(
        run_session,
        "_atomic_receipt",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError, match="replace failed"):
        run_session._launch_detached_prepared(
            run_directory=run_directory,
            child_argv=[sys.executable, "-c", "pass"],
        )
    expected_calls = (
        ["terminate", "wait"]
        if not wait_timeout
        else ["terminate", "wait", "kill", "wait"]
    )
    assert calls == expected_calls


def test_wp5b_preflight_never_checkpoints_before_common_tensor_rows(monkeypatch, tmp_path):
    """Preflight cannot declare rows unavailable before bilateral common-row coverage."""
    values = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    blocks = np.repeat(np.arange(5), 2)
    assert not modeling.make_outer_splits(
        values,
        blocks,
        target_family="categorical",
        fold_count=3,
    ).is_available
    assert modeling.make_outer_splits(
        values[2:],
        blocks[2:],
        target_family="categorical",
        fold_count=3,
    ).is_available
    paths = write_session_inputs(tmp_path)
    config = replace(decoding_config.load_task_decoding_config(paths["config"]), outer_fold_count=3)
    table = pd.DataFrame(
        {
            "current_action": values,
            "current_action_valid": np.ones(values.size, dtype=bool),
            "relative_doubt": np.ones(values.size, dtype=float),
            "relative_doubt_valid": np.ones(values.size, dtype=bool),
            "block_id": blocks,
        }
    )
    run_directory = tmp_path / "run"
    (run_directory / "checkpoints").mkdir(parents=True)
    existing = {"relative_doubt": make_coherent_model_records(paths)["records"][1]}
    result = pipeline._preflight_structurally_unavailable_targets(
        config=config,
        target_table=table,
        existing=existing,
        run_directory=run_directory,
        full_fingerprint="test",
    )
    assert "current_action" not in result
    assert not list((run_directory / "checkpoints").iterdir())


def test_wp5b_assembly_preserves_dynamic_selection_rules_and_explicit_session_id(tmp_path):
    """Assembly receives lossless rules and session ID explicitly without filesystem reads."""
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    long_rules = {
        "channel_labels": ["good"],
        "require_inside_brain": True,
        "metadata_unit_channels": [0, 1],
        "config_channel_ids": [0, 1],
        "cluster_groups": ["good"],
    }
    long_rules["long_explanation"] = "quality rule " * 40
    canonical_rules = json.dumps(long_rules, sort_keys=True, separators=(",", ":"))
    assert len(canonical_rules) > 256
    arrays, meta = pipeline._assemble_run_result_payload(
        config=model_inputs["config"],
        target_table=model_inputs["target_table"],
        rate_tensors=model_inputs["rate_tensors"],
        region_activity_metadata={
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": long_rules},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": long_rules},
        },
        ordered_target_results=model_inputs["records"],
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={
            "pid": 123,
            "start_token": "token",
            "mode": "foreground",
            "history": [],
        },
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    assert arrays["unit_selection_rules_json"].dtype.kind == "U"
    assert arrays["unit_selection_rules_json"].dtype.itemsize // 4 >= len(canonical_rules)
    assert json.loads(arrays["unit_selection_rules_json"][0]) == long_rules
    assert meta["provenance"]["session_id"] == "synthetic-session"


def test_wp5b_eligibility_distinguishes_target_invalid_from_not_common_rows(tmp_path):
    """Assembly labels target-invalid rows separately from target-valid rows absent from tensors."""
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    config = replace(model_inputs["config"], target_names=("relative_doubt",))
    result = model_inputs["records"][1]
    table = model_inputs["target_table"].copy()
    table.loc[0, "relative_doubt_valid"] = False
    tensors = replace(
        model_inputs["rate_tensors"],
        trial_row_indices=np.arange(2, 12, dtype=np.int64),
        trial_ids=np.arange(2, 12, dtype=np.int64),
        pfc_rate_tensor_hz=model_inputs["rate_tensors"].pfc_rate_tensor_hz[2:],
        hpc_rate_tensor_hz=model_inputs["rate_tensors"].hpc_rate_tensor_hz[2:],
    )
    arrays, _meta = pipeline._assemble_run_result_payload(
        config=config,
        target_table=table,
        rate_tensors=tensors,
        region_activity_metadata={
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": {}},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        ordered_target_results=(result,),
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={},
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    assert arrays["eligibility_reason_codes"][0, 0] == "target_ineligible"
    assert arrays["eligibility_reason_codes"][0, 1] == "not_common"


@pytest.mark.parametrize("state", ("UNKNOWN", ""))
def test_wp5b_slurm_unknown_or_query_failure_never_allows_takeover(monkeypatch, tmp_path, state):
    """Only explicit terminal Slurm states permit stale-owner replacement after bounded lookup."""
    owner = {
        "mode": "slurm",
        "host": "submit-host",
        "pid": None,
        "start_token": None,
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": "1234",
    }
    assert pipeline._inspect_execution_owner(
        owner,
        scheduler_state=lambda _job: state,
    ) == "foreign_unresolved"


@pytest.mark.parametrize("state", ("FAILED", "COMPLETED", "CANCELLED"))
def test_wp5b_known_terminal_slurm_states_allow_stale_replacement(state):
    """Explicit terminal scheduler states classify as replaceable rather than foreign/active."""
    owner = {
        "mode": "slurm",
        "host": "submit-host",
        "pid": None,
        "start_token": None,
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": "1234",
    }
    assert pipeline._inspect_execution_owner(
        owner,
        scheduler_state=lambda _job: state,
    ) == "slurm_terminal"


def test_wp5b_scheduler_lookup_uses_bounded_timeout(monkeypatch):
    """A one-shot Slurm query supplies an explicit timeout rather than hanging ownership checks."""
    observed: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        """Capture scheduler invocation and return an explicit terminal state."""
        observed.update(kwargs)
        return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="COMPLETED\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert pipeline._scheduler_state("1234") == "COMPLETED"
    assert isinstance(observed.get("timeout"), (int, float))
    assert observed["timeout"] > 0


@pytest.mark.parametrize("failure", ("nonzero", "timeout"))
def test_wp5b_scheduler_query_failures_are_unknown_and_refuse_takeover(monkeypatch, failure):
    """Nonzero or timed-out scheduler lookup remains unknown rather than terminal/stale."""
    if failure == "nonzero":
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: subprocess.CompletedProcess(
                args=args[0],
                returncode=1,
                stdout="",
                stderr="scheduler unavailable",
            ),
        )
    else:
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                subprocess.TimeoutExpired(args[0], kwargs["timeout"])
            ),
        )
    owner = {
        "mode": "slurm",
        "host": "submit-host",
        "pid": None,
        "start_token": None,
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": "1234",
    }
    assert pipeline._scheduler_state("1234") == "UNKNOWN"
    assert pipeline._inspect_execution_owner(owner) == "foreign_unresolved"


def test_wp5b_fallback_commands_shell_quote_special_run_paths(capsys, tmp_path):
    """Status and detached-print fallbacks shell-quote paths when saved command files are absent."""
    run_directory = tmp_path / "a path with spaces" / "$(unsafe)"
    run_directory.mkdir(parents=True)
    (run_directory / "run_state.json").write_text(json.dumps({"lifecycle": "initialized"}))
    (run_directory / "execution.json").write_text(json.dumps({"mode": "detached"}))
    (run_directory / "run.log").write_text("log\n")
    status = pipeline.inspect_task_decoding_status(run_directory)
    assert shlex.split(status["recovery_command"])[-1] == str(run_directory)
    run_session._print_detached_success(run_directory, run_directory / "local_launch.json")
    lines = capsys.readouterr().out.splitlines()
    resume = next(line.removeprefix("Resume: ") for line in lines if line.startswith("Resume: "))
    printed_status = next(
        line.removeprefix("Status: ") for line in lines if line.startswith("Status: ")
    )
    assert shlex.split(resume)[-2:] == [str(run_directory), "--detach"]
    assert shlex.split(printed_status)[-1] == str(run_directory)


def test_wp5b_process_start_token_parses_proc_comm_with_spaces(monkeypatch):
    """Linux /proc stat parsing must read field 22 after the final closing parenthesis."""
    stat = "123 (name with spaces) S " + " ".join(str(value) for value in range(4, 31))

    class FakeStatPath:
        """Small /proc stat stand-in returning a command name with embedded spaces."""

        def read_text(self, **_kwargs) -> str:
            """Return one synthetic proc stat line."""
            return stat

    class FakeProcPath:
        """Path-like stand-in that resolves a PID/stat lookup to FakeStatPath."""

        def __truediv__(self, _component):
            """Return self until the final stat member then provide synthetic text."""
            return FakeStatPath() if _component == "stat" else self

    monkeypatch.setattr(pipeline, "Path", lambda _path: FakeProcPath())
    assert pipeline._process_start_token(123) == "22"


def test_wp5b_direct_feature_axis_uses_full_region_metadata_order(tmp_path):
    """Direct-unit arrays retain dropped stable IDs from region metadata rather than fit subsets."""
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    original = model_inputs["records"][0]
    mutated_records = []
    for record in original.fold_records:
        if record.region_configuration == "PFC" and record.representation == "units":
            mutated_records.append(
                replace(
                    record,
                    feature_ids=("probe-pfc:2",),
                    coefficients=np.asarray(record.coefficients[-1:], dtype=float),
                    effective_feature_count=1,
                )
            )
        elif record.region_configuration == "PFC+HPC" and record.representation == "units":
            mutated_records.append(
                replace(
                    record,
                    feature_ids=("probe-pfc:2", "probe-hpc:1", "probe-hpc:2"),
                    coefficients=np.asarray(record.coefficients[[1, 2, 3]], dtype=float),
                    effective_feature_count=3,
                )
            )
        else:
            mutated_records.append(record)
    result = replace(original, fold_records=tuple(mutated_records))
    arrays, _meta = pipeline._assemble_run_result_payload(
        config=replace(model_inputs["config"], target_names=("current_action",)),
        target_table=model_inputs["target_table"],
        rate_tensors=model_inputs["rate_tensors"],
        region_activity_metadata={
            "PFC": {
                "unit_ids": ["probe-pfc:1", "probe-pfc:2"],
                "selection_rules": {},
            },
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        ordered_target_results=(result,),
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={},
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    direct_ids = arrays["coefficient_feature_ids"][0, 1, :2].tolist()
    assert direct_ids == ["probe-pfc:1", "probe-pfc:2"]
    assert arrays["requested_feature_counts"][0, 0, 1, 0, 0] == 2
    assert arrays["requested_feature_counts"][0, 1, 1, 0, 0] == 2
    assert arrays["requested_feature_counts"][0, 2, 1, 0, 0] == 4
    assert np.isnan(arrays["coefficient_values"][0, 0, 1, 0, 0, 0])
    assert np.isfinite(arrays["coefficient_values"][0, 0, 1, 0, 0, 1])
    assert np.isnan(arrays["coefficient_values"][0, 2, 1, 0, 0, 0])
    assert np.isfinite(arrays["coefficient_values"][0, 2, 1, 0, 0, 1:4]).all()


def test_wp5b_equal_width_fold_subsets_keep_distinct_metadata_feature_slots(tmp_path):
    """Equal-size fold subsets map coefficients by stable full unit identity, not longest subset."""
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    original = model_inputs["records"][0]
    changed = []
    for record in original.fold_records:
        if record.region_configuration == "PFC" and record.representation == "units":
            unit_id = "probe-pfc:1" if record.outer_fold_id % 2 == 0 else "probe-pfc:2"
            changed.append(
                replace(
                    record,
                    feature_ids=(unit_id,),
                    coefficients=np.asarray(record.coefficients[:1], dtype=float),
                    effective_feature_count=1,
                )
            )
        elif record.region_configuration == "PFC+HPC" and record.representation == "units":
            unit_id = "probe-pfc:1" if record.outer_fold_id % 2 == 0 else "probe-pfc:2"
            pfc_position = 0 if unit_id.endswith(":1") else 1
            changed.append(
                replace(
                    record,
                    feature_ids=(unit_id, "probe-hpc:1", "probe-hpc:2"),
                    coefficients=np.asarray(
                        record.coefficients[[pfc_position, 2, 3]],
                        dtype=float,
                    ),
                    effective_feature_count=3,
                )
            )
        else:
            changed.append(record)
    result = replace(original, fold_records=tuple(changed))
    arrays, _meta = pipeline._assemble_run_result_payload(
        config=replace(model_inputs["config"], target_names=("current_action",)),
        target_table=model_inputs["target_table"],
        rate_tensors=model_inputs["rate_tensors"],
        region_activity_metadata={
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": {}},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        ordered_target_results=(result,),
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={},
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    assert arrays["coefficient_feature_ids"][0, 1, :2].tolist() == [
        "probe-pfc:1",
        "probe-pfc:2",
    ]
    assert arrays["requested_feature_counts"][0, 0, 1, 0, 0] == 2
    assert arrays["requested_feature_counts"][0, 1, 1, 0, 0] == 2
    assert arrays["requested_feature_counts"][0, 2, 1, 0, 0] == 4
    assert np.isfinite(arrays["coefficient_values"][0, 0, 1, 0, 0, 0])
    assert np.isfinite(arrays["coefficient_values"][0, 0, 1, 0, 1, 1])


def test_wp5b_guard_still_exists_during_atomic_complete_publication(monkeypatch, tmp_path):
    """A second owner cannot interleave between directory validation and complete publication."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    observed: list[bool] = []
    real_update = pipeline._update_state

    def observe_complete(directory, **changes):
        """Assert guard ownership while the final complete state is atomically written."""
        if changes.get("lifecycle") == "complete":
            observed.append((directory / "execution_guard.json").exists())
        return real_update(directory, **changes)

    monkeypatch.setattr(pipeline, "_update_state", observe_complete)
    pipeline.run_prepared_task_decoding(run_directory)
    assert observed == [True]


def test_wp5b_successful_completion_clears_a_stale_midrun_error(monkeypatch, tmp_path):
    """A valid final publication cannot retain an error from a losing contender."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    real_write_figures = pipeline._write_minimal_family_heatmaps

    def write_figures_after_stale_error(directory, *, arrays):
        """Inject a competing-invocation error after this owner has done its work."""
        real_write_figures(directory, arrays=arrays)
        pipeline._update_state(
            directory,
            last_error="RuntimeError: rejected competing owner",
        )

    monkeypatch.setattr(
        pipeline,
        "_write_minimal_family_heatmaps",
        write_figures_after_stale_error,
    )

    pipeline.run_prepared_task_decoding(run_directory)

    state = read_json(run_directory / "run_state.json")
    assert state["lifecycle"] == "complete"
    assert state["final_results_published"] is True
    assert state["last_error"] == ""


def test_wp5b_valid_complete_reentry_skips_heavy_loading(monkeypatch, tmp_path):
    """Reentering a validated complete run returns without loading activity or decoding again."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    pipeline.run_prepared_task_decoding(run_directory)
    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    monkeypatch.setattr(pipeline.modeling, "decode_target", fail_if_called)
    assert pipeline.run_prepared_task_decoding(run_directory) is None


def test_wp5b_final_meta_uses_actual_execution_owner_and_nonzero_target_timing(
    monkeypatch,
    tmp_path,
):
    """Published provenance and timings record the actual claimed owner and wall time.

    The synthetic monotonic clock supplies seconds for the outer execution,
    target preparation, activity, modeling, and reporting boundaries.  The
    final result and ``execution.json`` must retain the detached child's
    owner identity rather than the preparing parent's identity.
    """
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    owner = {
        "mode": "detached",
        "host": "child-host",
        "pid": 4321,
        "start_token": "child-token",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": None,
    }
    captured: dict[str, object] = {}
    real_assemble = pipeline._assemble_run_result_payload

    def capture_payload(**kwargs):
        """Capture final meta/arrays while preserving the real assembly implementation."""
        assert kwargs["session_id"] == "synthetic-session"
        arrays, meta = real_assemble(**kwargs)
        captured["arrays"] = arrays
        captured["meta"] = meta
        return arrays, meta

    monkeypatch.setattr(pipeline, "_current_owner", lambda _execution: owner)
    monkeypatch.setattr(pipeline, "_assemble_run_result_payload", capture_payload)
    monotonic_values = iter(range(100))
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: float(next(monotonic_values)))
    pipeline.run_prepared_task_decoding(run_directory)
    arrays = captured["arrays"]
    meta = captured["meta"]
    assert meta["provenance"]["execution"]["pid"] == 4321
    assert meta["provenance"]["execution"]["start_token"] == "child-token"
    assert meta["provenance"]["execution"]["mode"] == "detached"
    execution = read_json(run_directory / "execution.json")
    assert execution["pid"] == 4321
    assert execution["start_token"] == "child-token"
    assert execution["mode"] == "detached"
    event_names = [entry["event"] for entry in execution["history"]]
    assert all(entry["timestamp"] for entry in execution["history"])
    assert event_names.index("guard_claimed") < event_names.index("guard_released")
    saved_history = meta["provenance"]["execution"]["history"]
    assert saved_history
    assert all(entry["timestamp"] for entry in saved_history)
    saved_event_names = [entry["event"] for entry in saved_history]
    assert saved_event_names[-1] == "guard_claimed"
    assert "guard_released" not in saved_event_names
    timing = dict(zip(arrays["stage_timing_labels"], arrays["stage_timing_seconds"], strict=True))
    assert timing == {"targets": 1.0, "activity": 1.0, "modeling": 1.0}
    assert arrays["total_timing_seconds"] == 5.0


def test_wp5b_metric_figures_render_three_region_by_two_representation_panels(
    monkeypatch,
    tmp_path,
):
    """Completion heatmaps require six region/representation panels per metric with time columns."""
    from matplotlib.figure import Figure

    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    arrays, _meta = pipeline._assemble_run_result_payload(
        config=model_inputs["config"],
        target_table=model_inputs["target_table"],
        rate_tensors=model_inputs["rate_tensors"],
        region_activity_metadata={
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": {}},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        ordered_target_results=model_inputs["records"],
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={},
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    observed_axes: list[tuple[tuple[tuple[str, np.ndarray, object], ...], str]] = []
    real_savefig = Figure.savefig

    def inspect_figure(figure, *args, **kwargs):
        """Record panel count and labels immediately before the genuine PNG renderer runs."""
        image_axes = tuple(
            (
                axis.get_title(),
                np.asarray(axis.images[0].get_array()),
                axis.images[0].norm,
            )
            for axis in figure.axes
            if axis.images
        )
        observed_axes.append((image_axes, figure._suptitle.get_text() if figure._suptitle else ""))
        return real_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_figure)
    pipeline._write_minimal_family_heatmaps(tmp_path, arrays=arrays)
    expected_shape = (1, arrays["time_bin_centers_s"].size)
    assert [len(image_axes) for image_axes, _caption in observed_axes] == [6, 6, 6]
    expected_titles = {
        f"{region} {representation}"
        for region in ("PFC", "HPC", "PFC+HPC")
        for representation in ("pca", "units")
    }
    assert all(
        {title for title, _image, _norm in image_axes} == expected_titles
        for image_axes, _ in observed_axes
    )
    assert all(
        "categorical" in caption.lower() or "numerical" in caption.lower()
        for _, caption in observed_axes
    )
    assert all(
        all(
            image.shape == expected_shape
            for _title, image, _norm in image_axes
        )
        for image_axes, _caption in observed_axes
    )
    family_rows = ((np.array([0]), 0), (np.array([0]), 1), (np.array([1]), 2))
    for (image_axes, _caption), (family_targets, metric) in zip(
        observed_axes,
        family_rows,
        strict=True,
    ):
        by_title = {title: image for title, image, _norm in image_axes}
        for region_index, region in enumerate(("PFC", "HPC", "PFC+HPC")):
            for representation_index, representation in enumerate(("pca", "units")):
                expected = np.nanmean(
                    arrays["fold_scores"][
                        family_targets,
                        region_index,
                        representation_index,
                        metric,
                    ],
                    axis=-1,
                )
                np.testing.assert_allclose(
                    by_title[f"{region} {representation}"],
                    expected,
                    equal_nan=True,
                )
        expected_reference = 0.0 if metric == 2 else 0.5
        assert all(
            np.isclose(float(norm(expected_reference)), 0.5)
            for _title, _image, norm in image_axes
        )


def test_wp5b_long_model_reasons_map_to_compact_codes_without_truncation(tmp_path):
    """Long fit and tuned-audit text must map to defined compact saved reason codes."""
    paths = write_session_inputs(tmp_path)
    fixed_inputs = make_coherent_model_records(paths)
    first_record = fixed_inputs["records"][0].fold_records[0]
    fixed_reason = "unavailable regional training features: PFC"
    affected_fold = first_record.outer_fold_id

    def rebuild_categorical_summaries(records):
        """Recompute complete-fold summaries after replacing a shared transform family."""
        summaries = {}
        for time_index, region, representation in {
            (record.time_bin_index, record.region_configuration, record.representation)
            for record in records
        }:
            ordered = sorted(
                (
                    record
                    for record in records
                    if (
                        record.time_bin_index,
                        record.region_configuration,
                        record.representation,
                    )
                    == (time_index, region, representation)
                ),
                key=lambda record: record.outer_fold_id,
            )
            summaries[(time_index, region, representation)] = modeling.aggregate_complete_folds(
                [
                    record.metrics.get("balanced_accuracy") if record.is_valid else None
                    for record in ordered
                ],
                expected_fold_count=3,
                invalid_reasons=[record.reason for record in ordered],
            )
        return summaries

    def unavailable_fixed_record(record):
        """Return one shared-PFC-transform unavailable fixed fit with exact sentinels."""
        return replace(
            record,
            is_valid=False,
            status="unavailable",
            reason=fixed_reason,
            effective_feature_count=0,
            estimator_class=None,
            estimator_parameters=record.estimator_parameters,
            convergence_status="not_run",
            metrics={},
            coefficients=np.empty(0, dtype=float),
            intercept=np.nan,
            feature_ids=(),
            selected_parameters=None,
        )

    fixed_records = [
        unavailable_fixed_record(record)
        if record.outer_fold_id == affected_fold
        and record.region_configuration in {"PFC", "PFC+HPC"}
        else record
        for record in fixed_inputs["records"][0].fold_records
    ]
    fixed_result = replace(
        fixed_inputs["records"][0],
        fold_records=tuple(fixed_records),
        cell_summaries=rebuild_categorical_summaries(fixed_records),
    )
    arrays, _meta = pipeline._assemble_run_result_payload(
        config=replace(fixed_inputs["config"], target_names=("current_action",)),
        target_table=fixed_inputs["target_table"],
        rate_tensors=fixed_inputs["rate_tensors"],
        region_activity_metadata={
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": {}},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        ordered_target_results=(fixed_result,),
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={},
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    assert arrays["fit_reason_codes"][0, 0, 0, 0, 0] == "outer_pfc_features_unavailable"

    tuned_inputs = make_coherent_model_records(paths, regularization_mode="tuned")
    tuned = tuned_inputs["records"][0]
    key = next(iter(tuned.candidate_inner_reasons))
    inner_reason = "unavailable inner regional training features: PFC"
    invalid_scores = tuple(tuple(None for _ in range(3)) for _ in range(15))
    invalid_statuses = tuple(tuple("invalid" for _ in range(3)) for _ in range(15))
    invalid_reasons = tuple(tuple(inner_reason for _ in range(3)) for _ in range(15))
    matching_record = next(record for record in tuned.fold_records if record.record_key == key)
    assert matching_record.estimator_parameters != first_record.estimator_parameters
    tuned_fold = matching_record.outer_fold_id

    def unavailable_tuned_record(record):
        """Return a no-selection outer record after shared PFC inner-transform failure."""
        return replace(
            record,
            is_valid=False,
            status="unavailable",
            reason="no valid inner tuning candidate",
            effective_feature_count=0,
            estimator_class=None,
            estimator_parameters=dict(first_record.estimator_parameters),
            convergence_status="not_run",
            metrics={},
            coefficients=np.empty(0, dtype=float),
            intercept=np.nan,
            feature_ids=(),
            selected_parameters=None,
            candidate_inner_scores=invalid_scores,
            candidate_inner_statuses=invalid_statuses,
            candidate_inner_reasons=invalid_reasons,
            selected_candidate_index=None,
        )

    scores = dict(tuned.candidate_inner_scores)
    statuses = dict(tuned.candidate_inner_statuses)
    reasons = dict(tuned.candidate_inner_reasons)
    selected = dict(tuned.selected_candidate_indices)
    affected_keys = {
        record.record_key
        for record in tuned.fold_records
        if record.outer_fold_id == tuned_fold
        and record.region_configuration in {"PFC", "PFC+HPC"}
    }
    for affected_key in affected_keys:
        scores[affected_key] = invalid_scores
        statuses[affected_key] = invalid_statuses
        reasons[affected_key] = invalid_reasons
        selected[affected_key] = None
    tuned = replace(
        tuned,
        fold_records=tuple(
            unavailable_tuned_record(record)
            if record.record_key in affected_keys
            else record
            for record in tuned.fold_records
        ),
        candidate_inner_scores=scores,
        candidate_inner_statuses=statuses,
        candidate_inner_reasons=reasons,
        selected_candidate_indices=selected,
        cell_summaries=rebuild_categorical_summaries(
            tuple(
                unavailable_tuned_record(record)
                if record.record_key in affected_keys
                else record
                for record in tuned.fold_records
            )
        ),
    )
    tuned_arrays, _meta = pipeline._assemble_run_result_payload(
        config=replace(tuned_inputs["config"], target_names=("current_action",)),
        target_table=tuned_inputs["target_table"],
        rate_tensors=tuned_inputs["rate_tensors"],
        region_activity_metadata={
            "PFC": {"unit_ids": ["probe-pfc:1", "probe-pfc:2"], "selection_rules": {}},
            "HPC": {"unit_ids": ["probe-hpc:1", "probe-hpc:2"], "selection_rules": {}},
        },
        ordered_target_results=(tuned,),
        stage_timing_seconds={"targets": 0.1, "activity": 0.2, "modeling": 0.3},
        input_manifest={"files": {"neural_session": {"relative_path": "neural_session.json"}}},
        scientific_source={"fingerprint": "source"},
        execution_provenance={},
        run_fingerprint="fingerprint",
        session_id="synthetic-session",
    )
    fold, time_index, region, representation = key
    region_index = ("PFC", "HPC", "PFC+HPC").index(region)
    representation_index = ("pca", "units").index(representation)
    assert (
        tuned_arrays["candidate_inner_reasons"][
            0,
            fold,
            region_index,
            representation_index,
            time_index,
            0,
            0,
        ]
        == "inner_candidate_unavailable"
    )


def test_wp5b_completed_new_skips_foreground_execution_and_detached_popen(
    monkeypatch,
    tmp_path,
):
    """A validated matching completed run is a CLI skip in both new launch modes."""
    paths = write_session_inputs(tmp_path)
    completed = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    pipeline.run_prepared_task_decoding(completed)
    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", fail_if_called)
    monkeypatch.setattr(run_session.subprocess, "Popen", fail_if_called)
    assert run_session.main(["new", "--config", str(paths["config"])]) == 0
    assert run_session.main(["new", "--config", str(paths["config"]), "--detach"]) == 0


@pytest.mark.parametrize(
    ("artifact", "replacement"),
    (
        ("summary.md", None),
        ("resource_usage.json", None),
        ("resource_usage.json", b"{}"),
        ("figures/categorical_auc.png", None),
        ("figures/categorical_auc.png", b"\x89PNG\r\n\x1a\n"),
    ),
)
def test_wp5b_completed_discovery_refuses_missing_or_corrupt_completion_artifacts(
    monkeypatch,
    tmp_path,
    artifact,
    replacement,
):
    """Default discovery and CLI new refuse a complete state lacking final artifacts.

    A lifecycle ``complete`` marker is insufficient: the immutable summary and
    every applicable PNG are part of the directory-level completion contract.
    The small mutations cover missing or invalid resource evidence, a missing
    summary, a missing PNG, and a PNG signature without decodable image data.
    """
    paths = write_session_inputs(tmp_path)
    completed = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    pipeline.run_prepared_task_decoding(completed)
    damaged_path = completed / artifact
    if replacement is None:
        damaged_path.unlink()
    else:
        damaged_path.write_bytes(replacement)
    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", fail_if_called)
    monkeypatch.setattr(run_session.subprocess, "Popen", fail_if_called)
    with pytest.raises(RuntimeError, match="recovery|rerun|resume|matching"):
        pipeline.prepare_task_decoding_run(
            paths["config"],
            rerun=False,
            execution_mode="foreground",
        )
    assert run_session.main(["new", "--config", str(paths["config"])]) != 0


def test_wp5b_complete_reentry_rejects_unpublished_terminal_state(monkeypatch, tmp_path):
    """A complete lifecycle marker cannot bypass its final-publication flag."""
    paths = write_session_inputs(tmp_path)
    completed = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    pipeline.run_prepared_task_decoding(completed)
    state_path = completed / "run_state.json"
    state = read_json(state_path)
    state["final_results_published"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="complete|published|state"):
        pipeline.run_prepared_task_decoding(completed)
    state["lifecycle"] = "complete"
    state["final_results_published"] = False
    state_path.write_text(json.dumps(state), encoding="utf-8")
    assert run_session.main(["resume", "--run-directory", str(completed)]) != 0


@pytest.mark.parametrize("corrupt_bytes", (b"", b"\x89PNG\r\n\x1a\n"))
def test_wp5b_completion_validation_rejects_empty_or_corrupt_png(
    monkeypatch,
    tmp_path,
    corrupt_bytes,
):
    """Completion validation decodes PNG structure, not empty/signature-only files."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    patch_coherent_execution(monkeypatch, make_coherent_model_records(paths))
    pipeline.run_prepared_task_decoding(run_directory)
    corrupt = run_directory / "figures" / "categorical_auc.png"
    corrupt.write_bytes(corrupt_bytes)
    with pytest.raises(ValueError, match="figure|PNG|invalid|corrupt"):
        pipeline._validate_published_run_directory(run_directory)


def test_wp5b_summary_uses_actual_launch_and_present_families_with_numeric_timings(
    monkeypatch,
    tmp_path,
):
    """Completion summary names only emitted families, actual mode, and numeric stage values."""
    paths = write_session_inputs(tmp_path)
    model_inputs = make_coherent_model_records(paths)
    config_payload = read_json(paths["config"])
    config_payload["target_names"] = ["current_action"]
    paths["config"].write_text(json.dumps(config_payload), encoding="ascii")
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="detached")
    patch_coherent_execution(
        monkeypatch,
        {
            **model_inputs,
        },
    )
    pipeline.run_prepared_task_decoding(run_directory)
    summary = (run_directory / "summary.md").read_text(encoding="utf-8").lower()
    assert "detached" in summary
    command_line = next(
        line.removeprefix("Launch command: ")
        for line in (run_directory / "summary.md").read_text(encoding="utf-8").splitlines()
        if line.startswith("Launch command: ")
    )
    assert shlex.split(command_line) == [
        "uv",
        "run",
        "python",
        "-m",
        "src.neural_analysis.task_decoding.run_session",
        "_execute-prepared",
        "--run-directory",
        str(run_directory),
    ]
    assert "run_session.py" in summary
    assert "pipeline.py" in summary
    assert "categorical ba/auc" in summary
    assert "numerical r2" not in summary
    for stage in ("targets", "activity", "modeling", "total"):
        assert re.search(rf"{stage}[^\n]*\d+\.\d+", summary)


def test_wp5b_status_exposes_saved_owner_identity_facts(monkeypatch, tmp_path):
    """Read-only status returns saved PID, start-token, and scheduler-job owner facts."""
    run_directory = tmp_path / "status-owner"
    run_directory.mkdir()
    (run_directory / "run_state.json").write_text(
        json.dumps({"lifecycle": "running", "final_results_published": False}),
        encoding="utf-8",
    )
    owner = {
        "mode": "slurm",
        "host": platform.node(),
        "pid": 123,
        "start_token": "saved-token",
        "started_at": "2026-10-07T00:00:00Z",
        "job_id": "777",
        "history": [{"timestamp": "2026-10-07T00:00:00Z", "event": "guard_claimed"}],
    }
    (run_directory / "execution.json").write_text(json.dumps(owner), encoding="utf-8")
    (run_directory / "execution_guard.json").write_text(json.dumps(owner), encoding="utf-8")
    (run_directory / "run.log").write_text("running\n", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_inspect_execution_owner", lambda _owner: "slurm_running")
    status = pipeline.inspect_task_decoding_status(run_directory, verify_results=False)
    assert status["execution_owner"]["pid"] == 123
    assert status["execution_owner"]["start_token"] == "saved-token"
    assert status["execution_owner"]["job_id"] == "777"
