"""WP5A RED contracts for task-decoding saved results and scientific identity.

This module intentionally covers only ``results.py``: primitive schema/NPZ,
input and scoped-source identity, complete-target checkpoints, and atomic
publication. Pipeline state, CLI, logs, summaries, plots, discovery, and raw
neural loading remain WP5B work.
"""

from __future__ import annotations

import builtins
from dataclasses import replace
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pytest

from src.neural_analysis.task_decoding import modeling, results, targets
from src.neural_analysis.task_decoding.config import (
    ANALYSIS_VERSION,
    RegionConfig,
    TaskDecodingConfig,
    scientific_config_payload,
)


_MEBIBYTE = 1024 * 1024
_RESULT_FILE = "results.npz"
_CONFIG_FILE = "config.json"
_MANIFEST_FILE = "input_manifest.json"
_FEATURE_COPY_FILE = "trial_feature_params.json"
_TARGETS = ("current_action", "relative_doubt")
_REGIONS = ("PFC", "HPC", "PFC+HPC")
_REPRESENTATIONS = ("pca", "units")
_METRICS = ("balanced_accuracy", "auc", "r2")
_TARGET_OUTER_UNAVAILABLE_CODE = "target_outer_unavailable"
_SCORE_AXES = ("target", "region", "representation", "metric", "time", "fold")
_FIT_AXES = ("target", "region", "representation", "time", "fold")
_BLOCK_VALUES = (
    100,
    100,
    "two",
    "two",
    200,
    200,
    "\u00b5-block-\u00e4",
    "\u00b5-block-\u00e4",
    300,
    300,
    "singleton-shared-label-longer-than-thirty-two-characters",
    "target-one-only",
    "target-zero-only",
    "not-common",
    "not-common",
)
_BLOCK_ID_JSON_BYTES = tuple(
    json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    for value in _BLOCK_VALUES
)
_BLOCK_ID_BYTE_COUNT = sum(len(value) for value in _BLOCK_ID_JSON_BYTES)
_DIRECT_RUNTIME_PATHS = (
    "src/__init__.py",
    "src/neural_analysis/__init__.py",
    "src/neural_analysis/session_metadata.py",
    "src/neural_analysis/spike_behavior/__init__.py",
    "src/neural_analysis/spike_behavior/loading.py",
    "src/neural_analysis/population/__init__.py",
    "src/neural_analysis/population/pca.py",
    "src/behavior_analysis/__init__.py",
    "src/behavior_analysis/project_utils.py",
    "pyproject.toml",
    "uv.lock",
)
_TASK_DECODING_FILES = (
    "__init__.py",
    "activity.py",
    "config.py",
    "modeling.py",
    "results.py",
    "targets.py",
)


def test_wp5b_selection_rules_json_uses_lossless_dynamic_unicode_width(tmp_path):
    """Selection-rule JSON round-trips canonically at its actual Unicode width, not a fixed cap.

    The two region values remain a ``(2,)`` Unicode array.  This bounded test
    uses a canonical rule object longer than 256 characters to freeze that the
    saved width is derived from content rather than a new arbitrary schema cap.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    rule = {
        "channel_labels": ["good"],
        "cluster_groups": ["good"],
        "require_inside_brain": True,
        "description": "lossless selected-channel provenance " * 12,
    }
    encoded = json.dumps(rule, sort_keys=True, separators=(",", ":"))
    arrays["unit_selection_rules_json"] = np.asarray(
        [encoded, encoded],
        dtype=f"U{len(encoded)}",
    )
    directory = tmp_path / "dynamic-selection-rules"
    results.save_task_decoding_run(
        directory,
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(directory)["arrays"]
    assert saved["unit_selection_rules_json"].shape == (2,)
    assert saved["unit_selection_rules_json"].dtype.kind == "U"
    assert saved["unit_selection_rules_json"].dtype.itemsize // 4 >= len(encoded)
    assert [json.loads(value) for value in saved["unit_selection_rules_json"]] == [rule, rule]


@pytest.mark.parametrize("rule_json", ('{"a":1,}', '{"b":1,"a":2}'))
def test_wp5b_loader_rejects_malformed_or_noncanonical_selection_rules_json(rule_json, tmp_path):
    """Saved region selection rules must be canonical JSON objects, not arbitrary Unicode text."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays["unit_selection_rules_json"] = np.asarray([rule_json, rule_json], dtype="U48")
    with pytest.raises(ValueError, match="selection|canonical|JSON"):
        results.save_task_decoding_run(
            tmp_path / f"bad-selection-{len(rule_json)}",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize("member", ("fit_reason_codes", "candidate_inner_reasons"))
def test_wp5b_loader_rejects_unknown_compact_reason_codes(member, tmp_path):
    """Saved compact reason fields reject unknown codes rather than accepting truncated prose."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    if member == "fit_reason_codes":
        fit_index = (0, 0, 0, 0, 0)
        arrays["fit_status"][fit_index] = "unavailable"
        arrays[member][fit_index] = "unknown_compact_reason"
        arrays["fold_scores"][0, 0, 0, :, 0, 0] = np.nan
        arrays["coefficient_values"][fit_index] = np.nan
        arrays["fitted_intercepts"][fit_index] = np.nan
    else:
        audit_index = (0, 0, 0, 0, 0, 14, 2)
        assert arrays["candidate_inner_statuses"][audit_index] == "invalid"
        assert arrays[member][audit_index] == "nonfinite_inner_score"
        arrays[member][audit_index] = "unknown_compact_reason"
    with pytest.raises(ValueError, match="reason|compact|unknown"):
        results.save_task_decoding_run(
            tmp_path / member,
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5b_declared_compact_reason_vocabulary_fits_fixed_fit_storage():
    """Every declared reason code fits losslessly in the U32 fit-reason member."""
    fit_reason_width = np.dtype("U32").itemsize // np.dtype("U1").itemsize
    assert all(len(code) <= fit_reason_width for code in results._COMPACT_REASON_CODES)


def write_input_fixture(
    tmp_path: Path,
    *,
    name: str = "session",
    include_threshold_files: bool = False,
) -> dict[str, Path]:
    """Create a portable session input tree with optional manifest boundary files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary parent directory.
    name : str, default="session"
        Session-root name varied by root-prefix portability tests.
    include_threshold_files : bool, default=False
        If true, create a sparse exactly-64-MiB JSON and a 64-MiB-plus-one
        binary only for the dedicated input-manifest boundary test.

    Returns
    -------
    dict[str, pathlib.Path]
        Contained explicit session inputs. Ordinary result fixtures remain tiny.
    """
    session_root = tmp_path / name
    processed_path = session_root / "processed"
    sorter_path = session_root / "probes" / "probe-a" / "sorter"
    processed_path.mkdir(parents=True)
    sorter_path.mkdir(parents=True)
    paths = {
        "session_root": session_root,
        "neural_session": session_root / "neural_session.json",
        "augmented_trials": processed_path / "augmented_trials.csv",
        "feature_parameters": processed_path / "trial_feature_params.json",
        "cluster_info": sorter_path / "cluster_info.tsv",
        "sorter_params": sorter_path / "params.py",
        "unlisted_neighbor": sorter_path / "unlisted.txt",
    }
    paths["neural_session"].write_bytes(b'{"session_id":"synthetic-01"}\n')
    paths["augmented_trials"].write_bytes(b"cur_trial,cur_block\n0,10\n1,10\n")
    paths["feature_parameters"].write_bytes(b'{"model":"synthetic","alpha":0.1}\n')
    paths["cluster_info"].write_bytes(b"cluster_id\tgroup\n11\tgood\n")
    paths["sorter_params"].write_bytes(b"sample_rate = 30000\n")
    paths["unlisted_neighbor"].write_bytes(b"not an explicit manifest input\n")
    if include_threshold_files:
        boundary_json = processed_path / "exactly_64_mib.json"
        spike_binary = sorter_path / "spike_clusters.npy"
        with boundary_json.open("wb") as stream:
            stream.truncate(64 * _MEBIBYTE)
        with spike_binary.open("wb") as stream:
            stream.truncate(64 * _MEBIBYTE + 1)
        paths["boundary_json"] = boundary_json
        paths["spike_binary"] = spike_binary
    return paths


def build_manifest(
    input_paths: Mapping[str, Path],
    *,
    include_threshold_files: bool = False,
) -> dict[str, object]:
    """Build one manifest from explicit files and an unsorted explicit file list.

    Parameters
    ----------
    input_paths : mapping[str, pathlib.Path]
        Contained source files created by :func:`write_input_fixture`.
    include_threshold_files : bool, default=False
        Include the 64-MiB boundary structured file and larger binary only for
        the dedicated manifest-performance test.

    Returns
    -------
    dict[str, object]
        Portable input identities and sorted explicit list membership.
    """
    files = {
        "neural_session": input_paths["neural_session"],
        "augmented_trials": input_paths["augmented_trials"],
        "trial_feature_parameters": input_paths["feature_parameters"],
    }
    if include_threshold_files:
        files.update(
            {
                "boundary_structured_json": input_paths["boundary_json"],
                "aligned_spike_binary": input_paths["spike_binary"],
            }
        )
    return results.build_input_manifest(
        session_root=input_paths["session_root"],
        files=files,
        file_lists={
            "pfc_sorter": (input_paths["sorter_params"], input_paths["cluster_info"])
        },
    )


def make_config(
    input_paths: Mapping[str, Path],
    *,
    regularization_mode: str,
    output_root_name: str = "analysis_runs",
) -> TaskDecodingConfig:
    """Create a real configuration record for the single config serializer owner.

    Parameters
    ----------
    input_paths : mapping[str, pathlib.Path]
        Contained session input files.
    regularization_mode : {"fixed", "tuned"}
        Frozen science mode matching the saved primitive arrays.
    output_root_name : str, default="analysis_runs"
        Execution-only output directory name, excluded by the serializer.

    Returns
    -------
    TaskDecodingConfig
        Real typed configuration passed only to ``scientific_config_payload``.
    """
    return TaskDecodingConfig(
        session_metadata_path=input_paths["neural_session"],
        augmented_trial_path=input_paths["augmented_trials"],
        trial_feature_parameter_path=input_paths["feature_parameters"],
        pfc_region=RegionConfig(
            region="PFC",
            probe_id="probe-pfc",
            channel_labels=("good",),
            require_inside_brain=True,
            cluster_groups=("good",),
        ),
        hpc_region=RegionConfig(
            region="HPC",
            probe_id="probe-hpc",
            channel_labels=("good",),
            require_inside_brain=True,
            cluster_groups=("good",),
        ),
        alignment="choice_time",
        bin_width_ms=500,
        pfc_pc_count=2,
        hpc_pc_count=2,
        target_names=_TARGETS,
        regularization_mode=regularization_mode,
        outer_fold_count=3,
        inner_fold_count=3,
        trusted_utc_bounds={"probe-pfc": (1.0, 2.0)},
        output_root=input_paths["session_root"] / output_root_name,
        session_root=input_paths["session_root"],
    )


def declared_candidates(
    scientific_config: Mapping[str, object],
    target_family: str,
) -> tuple[dict[str, float], ...]:
    """Expand the frozen serializer-owned tuning grid in declared deterministic order.

    Parameters
    ----------
    scientific_config : mapping[str, object]
        Complete mapping returned by ``scientific_config_payload``.
    target_family : {"categorical", "numerical"}
        Selects LogisticRegression ``C`` or ElasticNet ``alpha`` candidates.

    Returns
    -------
    tuple[dict[str, float], ...]
        Fifteen candidate mappings ordered strength then ``l1_ratio``.
    """
    estimator_name = "LogisticRegression" if target_family == "categorical" else "ElasticNet"
    strength_name = "C" if target_family == "categorical" else "alpha"
    grid = scientific_config["frozen_controls"]["tuning_grid"][estimator_name]
    return tuple(
        {strength_name: strength, "l1_ratio": l1_ratio}
        for strength in grid[strength_name]
        for l1_ratio in grid["l1_ratio"]
    )


def required_array_names() -> tuple[str, ...]:
    """Return the exact primitive NPZ schema member set for fixed and tuned runs.

    Returns
    -------
    tuple[str, ...]
        Every required saved array, including empty fixed-mode tuned members.
    """
    return (
        "target_labels",
        "target_families",
        "target_status",
        "target_unavailable_reasons",
        "target_source_labels",
        "target_positive_classes",
        "target_label_mappings_json",
        "region_labels",
        "representation_labels",
        "metric_labels",
        "time_bin_edges_s",
        "time_bin_centers_s",
        "fold_labels",
        "fold_scores",
        "fit_status",
        "fit_reason_codes",
        "requested_feature_counts",
        "effective_feature_counts",
        "train_counts",
        "test_counts",
        "train_class_counts",
        "test_class_counts",
        "full_table_row_positions",
        "full_table_trial_ids",
        "full_table_block_ids_utf8",
        "full_table_block_id_offsets",
        "encoded_target_values",
        "eligibility_masks",
        "eligibility_reason_codes",
        "eligibility_counts",
        "trial_row_indices",
        "outer_fold_ids",
        "unit_selection_rules_json",
        "coefficient_values",
        "coefficient_feature_ids",
        "coefficient_feature_regions",
        "coefficient_feature_statuses",
        "coefficient_active_masks",
        "fitted_intercepts",
        "fixed_parameters_json",
        "candidate_parameter_json",
        "selected_parameters_json",
        "inner_selection_fold_ids",
        "candidate_inner_scores",
        "candidate_inner_statuses",
        "candidate_inner_reasons",
        "selected_candidate_indices",
        "stage_timing_labels",
        "stage_timing_seconds",
        "total_timing_seconds",
    )


def make_result_arrays(
    scientific_config: Mapping[str, object],
    *,
    regularization_mode: str,
) -> dict[str, np.ndarray]:
    """Build scientifically consistent 15-row/13-common-row fixed or tuned arrays.

    Parameters
    ----------
    scientific_config : mapping[str, object]
        Complete serializer-owned scientific mapping, including the frozen grid.
    regularization_mode : {"fixed", "tuned"}
        Selects explicit no-inner sentinels or leakage-safe nested-CV records.

    Returns
    -------
    dict[str, numpy.ndarray]
        Complete primitive NPZ schema. Common row 11 is target-0 unavailable;
        common row 12 is target-1 unavailable. Each target has twelve eligible
        rows, four outer-test rows per fold, and eight outer-training rows.
    """
    target_count, region_count, representation_count = 2, 3, 2
    metric_count, time_count, fold_count, feature_capacity = 3, 8, 3, 4
    full_row_count, common_row_count = 15, 13
    score_shape = (
        target_count,
        region_count,
        representation_count,
        metric_count,
        time_count,
        fold_count,
    )
    fit_shape = (target_count, region_count, representation_count, time_count, fold_count)
    coefficient_shape = (*fit_shape, feature_capacity)
    parameter_shape = (target_count, fold_count, region_count, representation_count, time_count)
    feature_shape = (region_count, representation_count, feature_capacity)
    arrays = {
        "target_labels": np.array(_TARGETS, dtype="U32"),
        "target_families": np.array(["categorical", "numerical"], dtype="U16"),
        "target_status": np.array(["available", "available"], dtype="U16"),
        "target_unavailable_reasons": np.array(["", ""], dtype="U128"),
        "target_source_labels": np.array(["action", "relative_doubt_index"], dtype="U32"),
        "target_positive_classes": np.array([1, -1], dtype=np.int64),
        "target_label_mappings_json": np.array(
            ['{"0":"right","1":"left"}', '{"native_target":true}'], dtype="U48"
        ),
        "region_labels": np.array(_REGIONS, dtype="U16"),
        "representation_labels": np.array(_REPRESENTATIONS, dtype="U16"),
        "metric_labels": np.array(_METRICS, dtype="U32"),
        "time_bin_edges_s": np.linspace(-2.0, 2.0, time_count + 1, dtype=np.float64),
        "time_bin_centers_s": np.linspace(-1.75, 1.75, time_count, dtype=np.float64),
        "fold_labels": np.arange(fold_count, dtype=np.int64),
        "fold_scores": np.full(score_shape, np.nan, dtype=np.float64),
        "fit_status": np.full(fit_shape, "valid", dtype="U16"),
        "fit_reason_codes": np.full(fit_shape, "", dtype="U32"),
        "requested_feature_counts": np.full(fit_shape, feature_capacity, dtype=np.int64),
        "effective_feature_counts": np.full(fit_shape, feature_capacity, dtype=np.int64),
        "train_counts": np.full(fit_shape, 8, dtype=np.int64),
        "test_counts": np.full(fit_shape, 4, dtype=np.int64),
        "train_class_counts": np.full((*fit_shape, 2), -1, dtype=np.int64),
        "test_class_counts": np.full((*fit_shape, 2), -1, dtype=np.int64),
        "full_table_row_positions": np.arange(full_row_count, dtype=np.int64),
        "full_table_trial_ids": np.arange(full_row_count, dtype=np.int64),
        "full_table_block_ids_utf8": np.empty(_BLOCK_ID_BYTE_COUNT, dtype=np.uint8),
        "full_table_block_id_offsets": np.empty(full_row_count + 1, dtype=np.int64),
        "encoded_target_values": np.full(
            (target_count, full_row_count),
            np.nan,
            dtype=np.float64,
        ),
        "eligibility_masks": np.zeros((target_count, full_row_count), dtype=np.bool_),
        "eligibility_reason_codes": np.full(
            (target_count, full_row_count),
            "not_common",
            dtype="U32",
        ),
        "eligibility_counts": np.zeros(target_count, dtype=np.int64),
        "trial_row_indices": np.arange(common_row_count, dtype=np.int64),
        "outer_fold_ids": np.full(
            (target_count, common_row_count),
            -1,
            dtype=np.int64,
        ),
        "unit_selection_rules_json": np.array(
            ['{"PFC":"good_inside_brain"}', '{"HPC":"good_inside_brain"}'], dtype="U48"
        ),
        "coefficient_values": np.full(coefficient_shape, np.nan, dtype=np.float64),
        "coefficient_feature_ids": np.full(feature_shape, "", dtype="U32"),
        "coefficient_feature_regions": np.full(feature_shape, "", dtype="U16"),
        "coefficient_feature_statuses": np.full(feature_shape, "padding", dtype="U16"),
        "coefficient_active_masks": np.zeros(feature_shape, dtype=np.bool_),
        "fitted_intercepts": np.arange(np.prod(fit_shape), dtype=np.float64).reshape(fit_shape),
        "fixed_parameters_json": np.full(parameter_shape, "", dtype="U48"),
        "candidate_parameter_json": np.empty((0,), dtype="U48"),
        "selected_parameters_json": np.full(parameter_shape, "", dtype="U48"),
        "inner_selection_fold_ids": np.empty((0,), dtype=np.int64),
        "candidate_inner_scores": np.empty((0,), dtype=np.float64),
        "candidate_inner_statuses": np.empty((0,), dtype="U16"),
        "candidate_inner_reasons": np.empty((0,), dtype="U48"),
        "selected_candidate_indices": np.empty((0,), dtype=np.int64),
        "stage_timing_labels": np.array(["targets", "activity", "modeling"], dtype="U16"),
        "stage_timing_seconds": np.array([0.01, 0.02, 0.03], dtype=np.float64),
        "total_timing_seconds": np.array(0.06, dtype=np.float64),
    }
    arrays["full_table_block_ids_utf8"] = np.frombuffer(
        b"".join(_BLOCK_ID_JSON_BYTES),
        dtype=np.uint8,
    ).copy()
    arrays["full_table_block_id_offsets"] = np.r_[
        np.array([0], dtype=np.int64),
        np.cumsum([len(value) for value in _BLOCK_ID_JSON_BYTES], dtype=np.int64),
    ]
    target_zero_values = np.tile(np.array([0.0, 1.0, 0.0, 1.0]), 3)
    target_one_values = np.linspace(-1.0, 1.0, 12, dtype=np.float64)
    eligible_rows = (np.r_[np.arange(11), 12], np.arange(12))
    outer_fold_by_row = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        dtype=np.int64,
    )
    for target_index, rows in enumerate(eligible_rows):
        values = target_zero_values if target_index == 0 else target_one_values
        arrays["encoded_target_values"][target_index, rows] = values
        arrays["eligibility_masks"][target_index, rows] = True
        arrays["eligibility_reason_codes"][target_index, rows] = ""
        arrays["eligibility_counts"][target_index] = rows.size
        arrays["outer_fold_ids"][target_index, rows] = outer_fold_by_row[rows]
    arrays["eligibility_reason_codes"][0, 11] = "target_ineligible"
    arrays["eligibility_reason_codes"][1, 12] = "target_ineligible"
    arrays["fold_scores"][0, :, :, 0, :, :] = 0.75
    arrays["fold_scores"][0, :, :, 1, :, :] = 0.80
    arrays["fold_scores"][1, :, :, 2, :, :] = -0.25
    arrays["train_class_counts"][0, ...] = (4, 4)
    arrays["test_class_counts"][0, ...] = (2, 2)
    feature_values = {
        ("PFC", "pca"): (("PFC:PC1", "PFC:PC2"), ("PFC", "PFC")),
        ("PFC", "units"): (("probe-pfc:11", "probe-pfc:19"), ("PFC", "PFC")),
        ("HPC", "pca"): (("HPC:PC1", "HPC:PC2"), ("HPC", "HPC")),
        ("HPC", "units"): (("probe-hpc:5", "probe-hpc:17"), ("HPC", "HPC")),
        ("PFC+HPC", "pca"): (
            ("PFC:PC1", "PFC:PC2", "HPC:PC1", "HPC:PC2"),
            ("PFC", "PFC", "HPC", "HPC"),
        ),
        ("PFC+HPC", "units"): (
            ("probe-pfc:11", "probe-pfc:19", "probe-hpc:5", "probe-hpc:17"),
            ("PFC", "PFC", "HPC", "HPC"),
        ),
    }
    for region_index, region in enumerate(_REGIONS):
        for representation_index, representation in enumerate(_REPRESENTATIONS):
            feature_ids, feature_regions = feature_values[(region, representation)]
            width = len(feature_ids)
            feature_slice = (region_index, representation_index, slice(None, width))
            arrays["coefficient_feature_ids"][feature_slice] = feature_ids
            arrays["coefficient_feature_regions"][feature_slice] = feature_regions
            arrays["coefficient_feature_statuses"][feature_slice] = "available"
            arrays["coefficient_active_masks"][feature_slice] = True
            arrays["coefficient_values"][
                :, region_index, representation_index, :, :, :width
            ] = 0.5
            arrays["effective_feature_counts"][:, region_index, representation_index] = width
    requested_widths = np.array([[2, 2], [2, 2], [4, 4]], dtype=np.int64)
    for region_index, representation_index in np.ndindex(requested_widths.shape):
        arrays["requested_feature_counts"][:, region_index, representation_index] = (
            requested_widths[region_index, representation_index]
        )
    categorical_fixed = json.dumps({"C": 1.0, "l1_ratio": 0.5}, sort_keys=True)
    numerical_fixed = json.dumps({"alpha": 1.0, "l1_ratio": 0.5}, sort_keys=True)
    arrays["fixed_parameters_json"][0, ...] = categorical_fixed
    arrays["fixed_parameters_json"][1, ...] = numerical_fixed
    if regularization_mode == "tuned":
        categorical_candidates = declared_candidates(scientific_config, "categorical")
        numerical_candidates = declared_candidates(scientific_config, "numerical")
        candidate_parameters = np.empty((target_count, 15), dtype="U48")
        candidate_parameters[0] = [
            json.dumps(item, sort_keys=True) for item in categorical_candidates
        ]
        candidate_parameters[1] = [
            json.dumps(item, sort_keys=True) for item in numerical_candidates
        ]
        audit_shape = (
            target_count,
            fold_count,
            region_count,
            representation_count,
            time_count,
            15,
            3,
        )
        audit_scores = np.empty(audit_shape, dtype=np.float64)
        audit_statuses = np.full(audit_shape, "valid", dtype="U16")
        audit_reasons = np.full(audit_shape, "", dtype="U48")
        selected_indices = np.empty(audit_shape[:-2], dtype=np.int64)
        inner_assignments = np.full(
            (target_count, fold_count, common_row_count),
            -1,
            dtype=np.int64,
        )
        for target_index in range(target_count):
            outer_folds = arrays["outer_fold_ids"][target_index]
            for outer_fold in range(fold_count):
                eligible_train = np.flatnonzero(
                    (outer_folds != outer_fold) & (outer_folds >= 0)
                )
                inner_group_ids: dict[bytes, int] = {}
                for row_index in eligible_train:
                    block_json = _BLOCK_ID_JSON_BYTES[row_index]
                    if block_json not in inner_group_ids:
                        inner_group_ids[block_json] = len(inner_group_ids) % 3
                    inner_assignments[target_index, outer_fold, row_index] = (
                        inner_group_ids[block_json]
                    )
        for target_index in range(target_count):
            for outer_fold in range(fold_count):
                for region_index in range(region_count):
                    for representation_index in range(representation_count):
                        for time_index in range(time_count):
                            selected_index = (
                                target_index
                                + outer_fold
                                + region_index
                                + representation_index
                                + time_index
                            ) % 14
                            selected_indices[
                                target_index,
                                outer_fold,
                                region_index,
                                representation_index,
                                time_index,
                            ] = selected_index
                            for candidate_index in range(15):
                                for inner_fold in range(3):
                                    mixed_radix = (
                                        (((((target_index * fold_count + outer_fold)
                                        * region_count + region_index)
                                        * representation_count + representation_index)
                                        * time_count + time_index)
                                        * 15 + candidate_index)
                                        * 3 + inner_fold
                                    )
                                    score_index = (
                                        target_index,
                                        outer_fold,
                                        region_index,
                                        representation_index,
                                        time_index,
                                        candidate_index,
                                        inner_fold,
                                    )
                                    base_score = 0.10 + mixed_radix / 100_000.0
                                    audit_scores[score_index] = (
                                        base_score + 0.70
                                        if candidate_index == selected_index
                                        else base_score
                                    )
                            invalid_index = (
                                target_index,
                                outer_fold,
                                region_index,
                                representation_index,
                                time_index,
                                14,
                                2,
                            )
                            audit_scores[invalid_index] = np.nan
                            audit_statuses[invalid_index] = "invalid"
                            audit_reasons[invalid_index] = "nonfinite_inner_score"
                            arrays["selected_parameters_json"][
                                target_index,
                                outer_fold,
                                region_index,
                                representation_index,
                                time_index,
                            ] = candidate_parameters[target_index, selected_index]
        arrays.update(
            {
                "candidate_parameter_json": candidate_parameters,
                "inner_selection_fold_ids": inner_assignments,
                "candidate_inner_scores": audit_scores,
                "candidate_inner_statuses": audit_statuses,
                "candidate_inner_reasons": audit_reasons,
                "selected_candidate_indices": selected_indices,
            }
        )
    return arrays


def complete_axis_mapping(
    arrays: Mapping[str, np.ndarray],
    *,
    regularization_mode: str,
) -> dict[str, list[str]]:
    """Return an axis entry for every required primitive member, including scalars.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Complete fixed or tuned primitive array payload.
    regularization_mode : {"fixed", "tuned"}
        Determines whether inner/candidate arrays use real or not-applicable axes.

    Returns
    -------
    dict[str, list[str]]
        Exact ordered axes for every NPZ member. Scalars use ``[]``.
    """
    axes = {
        "target_labels": ["target"],
        "target_families": ["target"],
        "target_status": ["target"],
        "target_unavailable_reasons": ["target"],
        "target_source_labels": ["target"],
        "target_positive_classes": ["target"],
        "target_label_mappings_json": ["target"],
        "region_labels": ["region"],
        "representation_labels": ["representation"],
        "metric_labels": ["metric"],
        "time_bin_edges_s": ["time_edge"],
        "time_bin_centers_s": ["time"],
        "fold_labels": ["fold"],
        "fold_scores": list(_SCORE_AXES),
        "fit_status": list(_FIT_AXES),
        "fit_reason_codes": list(_FIT_AXES),
        "requested_feature_counts": list(_FIT_AXES),
        "effective_feature_counts": list(_FIT_AXES),
        "train_counts": list(_FIT_AXES),
        "test_counts": list(_FIT_AXES),
        "train_class_counts": [*_FIT_AXES, "class"],
        "test_class_counts": [*_FIT_AXES, "class"],
        "full_table_row_positions": ["full_table_row"],
        "full_table_trial_ids": ["full_table_row"],
        "full_table_block_ids_utf8": ["block_id_byte"],
        "full_table_block_id_offsets": ["full_table_row_boundary"],
        "encoded_target_values": ["target", "full_table_row"],
        "eligibility_masks": ["target", "full_table_row"],
        "eligibility_reason_codes": ["target", "full_table_row"],
        "eligibility_counts": ["target"],
        "trial_row_indices": ["common_neural_tensor_row"],
        "outer_fold_ids": ["target", "common_neural_tensor_row"],
        "unit_selection_rules_json": ["region_component"],
        "coefficient_values": [*_FIT_AXES, "feature"],
        "coefficient_feature_ids": ["region", "representation", "feature"],
        "coefficient_feature_regions": ["region", "representation", "feature"],
        "coefficient_feature_statuses": ["region", "representation", "feature"],
        "coefficient_active_masks": ["region", "representation", "feature"],
        "fitted_intercepts": list(_FIT_AXES),
        "fixed_parameters_json": ["target", "outer_fold", "region", "representation", "time"],
        "selected_parameters_json": ["target", "outer_fold", "region", "representation", "time"],
        "stage_timing_labels": ["stage"],
        "stage_timing_seconds": ["stage"],
        "total_timing_seconds": [],
    }
    if regularization_mode == "fixed":
        axes.update(
            {
                "candidate_parameter_json": ["not_applicable"],
                "inner_selection_fold_ids": ["not_applicable"],
                "candidate_inner_scores": ["not_applicable"],
                "candidate_inner_statuses": ["not_applicable"],
                "candidate_inner_reasons": ["not_applicable"],
                "selected_candidate_indices": ["not_applicable"],
            }
        )
    else:
        candidate_axes = [
            "target",
            "outer_fold",
            "region",
            "representation",
            "time",
            "candidate",
            "inner_fold",
        ]
        axes.update(
            {
                "candidate_parameter_json": ["target", "candidate"],
                "inner_selection_fold_ids": [
                    "target",
                    "outer_fold",
                    "common_neural_tensor_row",
                ],
                "candidate_inner_scores": candidate_axes,
                "candidate_inner_statuses": candidate_axes,
                "candidate_inner_reasons": candidate_axes,
                "selected_candidate_indices": candidate_axes[:-2],
            }
        )
    assert set(axes) == set(arrays)
    return axes


def make_meta(
    arrays: Mapping[str, np.ndarray],
    scientific_config: Mapping[str, object],
    *,
    regularization_mode: str,
    run_fingerprint: str = "run-fingerprint-test",
) -> dict[str, object]:
    """Create the named required schema/version/axis/units/provenance mapping.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Complete NPZ primitive schema.
    scientific_config : mapping[str, object]
        Sole serializer-owned scientific payload saved beside the NPZ.
    regularization_mode : {"fixed", "tuned"}
        Mode matching all config and mode-specific array contracts.
    run_fingerprint : str, default="run-fingerprint-test"
        Complete opaque scientific run identity required to agree with the
        separate immutable-save argument.

    Returns
    -------
    dict[str, object]
        JSON-safe ``meta`` dictionary returned by the directory-level loader.
    """
    return {
        "schema_version": results.RESULT_SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "random_seed": scientific_config["frozen_controls"]["seed"],
        "units": {
            "encoded_target_values": {
                "current_action": "encoded class",
                "relative_doubt": "unitless",
            },
            "fold_scores": {
                "current_action": {
                    "balanced_accuracy": "fraction",
                    "auc": "fraction",
                },
                "relative_doubt": {"r2": "coefficient_of_determination"},
            },
            "candidate_inner_scores": {
                "current_action": {"balanced_accuracy": "fraction"},
                "relative_doubt": {"r2": "coefficient_of_determination"},
            },
            "coefficient_values": {
                "current_action": "log-odds change per pooled training standard deviation",
                "relative_doubt": "unitless per pooled training standard deviation",
            },
            "fitted_intercepts": {
                "current_action": "log-odds",
                "relative_doubt": "unitless",
            },
            "time_bin_edges_s": "s",
            "time_bin_centers_s": "s",
            "stage_timing_seconds": "s",
            "total_timing_seconds": "s",
        },
        "axes": complete_axis_mapping(arrays, regularization_mode=regularization_mode),
        "paths": {"session_root": ".", "feature_parameter_copy": _FEATURE_COPY_FILE},
        "parameters": {
            "regularization_mode": regularization_mode,
            "outer_fold_count": 3,
            "inner_fold_count": 3,
            "bin_width_ms": 500,
            "candidate_selection_metrics": {
                "current_action": "balanced_accuracy",
                "relative_doubt": "r2",
            },
        },
        "provenance": {
            "session_id": "synthetic-01",
            "source_fingerprint": "source-fingerprint-test",
            "scientific_config_identity": hashlib.sha256(
                json.dumps(scientific_config, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "run_fingerprint": run_fingerprint,
        },
        "warnings": ["synthetic warning retained for provenance"],
    }


def save_run_fixture(
    run_directory: Path,
    input_paths: Mapping[str, Path],
    *,
    regularization_mode: str,
    output_root_name: str = "analysis_runs",
    run_fingerprint: str = "run-fingerprint-test",
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object], dict[str, object]]:
    """Publish one internally consistent fixed or tuned synthetic run.

    Parameters
    ----------
    run_directory : pathlib.Path
        New immutable result directory.
    input_paths : mapping[str, pathlib.Path]
        Contained source inputs.
    regularization_mode : {"fixed", "tuned"}
        Shared config/meta/array regularization policy.
    output_root_name : str, default="analysis_runs"
        Execution-only typed configuration value for serializer invariance tests.
    run_fingerprint : str, default="run-fingerprint-test"
        Full stable scientific run identity.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object], dict[str, object], dict[str, object]]
        Original arrays, meta, manifest, and complete scientific config payload.
    """
    config = make_config(
        input_paths,
        regularization_mode=regularization_mode,
        output_root_name=output_root_name,
    )
    scientific_config = scientific_config_payload(config)
    arrays = make_result_arrays(scientific_config, regularization_mode=regularization_mode)
    meta = make_meta(
        arrays,
        scientific_config,
        regularization_mode=regularization_mode,
        run_fingerprint=run_fingerprint,
    )
    manifest = build_manifest(input_paths)
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        meta=meta,
        input_manifest=manifest,
        scientific_config=scientific_config,
        feature_parameter_source=input_paths["feature_parameters"],
        run_fingerprint=run_fingerprint,
    )
    return arrays, meta, manifest, scientific_config


def make_run_save_arguments(
    input_paths: Mapping[str, Path],
    *,
    regularization_mode: str,
    run_fingerprint: str = "run-fingerprint-test",
) -> dict[str, object]:
    """Build a complete ``save_task_decoding_run`` argument mapping.

    Parameters
    ----------
    input_paths : mapping[str, pathlib.Path]
        Tiny explicit source inputs for one synthetic session.
    regularization_mode : {"fixed", "tuned"}
        Frozen nested-CV mode represented by the primitive arrays.
    run_fingerprint : str, default="run-fingerprint-test"
        Complete stable scientific identity written with the immutable run.

    Returns
    -------
    dict[str, object]
        Keyword-only save arguments, excluding the destination directory.
    """
    scientific_config = scientific_config_payload(
        make_config(input_paths, regularization_mode=regularization_mode)
    )
    arrays = make_result_arrays(scientific_config, regularization_mode=regularization_mode)
    meta = make_meta(
        arrays,
        scientific_config,
        regularization_mode=regularization_mode,
        run_fingerprint=run_fingerprint,
    )
    return {
        "arrays": arrays,
        "meta": meta,
        "input_manifest": build_manifest(input_paths),
        "scientific_config": scientific_config,
        "feature_parameter_source": input_paths["feature_parameters"],
        "run_fingerprint": run_fingerprint,
    }


def test_condition_payload_stacks_only_condition_dependent_axes_and_round_trips(
    tmp_path: Path,
) -> None:
    """Schema 2 should save one validated condition axis without duplicating identities."""
    input_paths = write_input_fixture(tmp_path)
    typed_config = replace(
        make_config(input_paths, regularization_mode="fixed"),
        condition_names=("all", "incorrect"),
    )
    scientific_config = scientific_config_payload(typed_config)
    all_arrays = make_result_arrays(scientific_config, regularization_mode="fixed")
    incorrect_arrays = {name: value.copy() for name, value in all_arrays.items()}
    incorrect_arrays["fold_scores"] = incorrect_arrays["fold_scores"].copy()
    incorrect_arrays["fold_scores"][0, :, :, 0, :, :] = 0.61
    all_meta = make_meta(
        all_arrays,
        scientific_config,
        regularization_mode="fixed",
    )
    incorrect_meta = make_meta(
        incorrect_arrays,
        scientific_config,
        regularization_mode="fixed",
    )
    full_count = all_arrays["full_table_row_positions"].size
    condition_masks = {
        "all": np.ones(full_count, dtype=np.bool_),
        "incorrect": np.ones(full_count, dtype=np.bool_),
    }

    stacked_arrays, stacked_meta = results.assemble_condition_result_payload(
        condition_names=("all", "incorrect"),
        condition_masks=condition_masks,
        condition_payloads={
            "all": (all_arrays, all_meta),
            "incorrect": (incorrect_arrays, incorrect_meta),
        },
    )

    assert results.CONDITION_RESULT_SCHEMA_VERSION == 2
    assert stacked_meta["schema_version"] == 2
    assert stacked_arrays["condition_labels"].tolist() == ["all", "incorrect"]
    assert stacked_arrays["condition_masks"].shape == (2, full_count)
    assert stacked_arrays["fold_scores"].shape == (2, *all_arrays["fold_scores"].shape)
    assert stacked_arrays["target_labels"].shape == all_arrays["target_labels"].shape
    assert stacked_arrays["encoded_target_values"].shape == (
        all_arrays["target_labels"].size,
        full_count,
    )
    assert stacked_meta["axes"]["fold_scores"][0] == "condition"
    assert stacked_meta["axes"]["target_labels"] == ["target"]

    manifest = build_manifest(input_paths)
    run_directory = tmp_path / "condition-run"
    results.save_task_decoding_run(
        run_directory,
        arrays=stacked_arrays,
        meta=stacked_meta,
        input_manifest=manifest,
        scientific_config=scientific_config,
        feature_parameter_source=input_paths["feature_parameters"],
        run_fingerprint="run-fingerprint-test",
    )
    loaded = results.load_task_decoding_run(run_directory)

    assert loaded["meta"]["schema_version"] == 2
    assert loaded["arrays"]["condition_labels"].tolist() == ["all", "incorrect"]
    np.testing.assert_array_equal(
        loaded["arrays"]["fold_scores"],
        stacked_arrays["fold_scores"],
    )


def test_schema_one_pooled_result_remains_loadable_after_condition_extension(
    tmp_path: Path,
) -> None:
    """The immutable completed pooled format must remain a supported read path."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "pooled-run"

    save_run_fixture(run_directory, input_paths, regularization_mode="fixed")
    loaded = results.load_task_decoding_run(run_directory)

    assert loaded["meta"]["schema_version"] == 1
    assert "condition_labels" not in loaded["arrays"]
    assert results.condition_labels(loaded) == ("all",)


def write_raw_npz(
    run_directory: Path,
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object] | None,
) -> None:
    """Write deliberately raw NPZ members for isolated loader-negative tests.

    Parameters
    ----------
    run_directory : pathlib.Path
        Existing saved run retaining valid supporting files.
    arrays : mapping[str, numpy.ndarray]
        Replacement primitive members, potentially intentionally invalid.
    meta : mapping[str, object] or None
        Named scalar JSON metadata or None to omit the required member.
    """
    members = dict(arrays)
    if meta is not None:
        members["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(run_directory / _RESULT_FILE, **members)


def expected_array_contract(
    regularization_mode: str,
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Return exact fixed/tuned shapes and safe dtypes for every NPZ member.

    Parameters
    ----------
    regularization_mode : {"fixed", "tuned"}
        Determines real nested-CV dimensions versus explicit empty sentinels.

    Returns
    -------
    dict[str, tuple[tuple[int, ...], numpy.dtype]]
        Required member name mapped to complete primitive shape and dtype.
    """
    fit_shape = (2, 3, 2, 8, 3)
    score_shape = (2, 3, 2, 3, 8, 3)
    parameter_shape = (2, 3, 3, 2, 8)
    feature_shape = (3, 2, 4)
    contract = {
        "target_labels": ((2,), np.dtype("U32")),
        "target_families": ((2,), np.dtype("U16")),
        "target_status": ((2,), np.dtype("U16")),
        "target_unavailable_reasons": ((2,), np.dtype("U128")),
        "target_source_labels": ((2,), np.dtype("U32")),
        "target_positive_classes": ((2,), np.dtype(np.int64)),
        "target_label_mappings_json": ((2,), np.dtype("U48")),
        "region_labels": ((3,), np.dtype("U16")),
        "representation_labels": ((2,), np.dtype("U16")),
        "metric_labels": ((3,), np.dtype("U32")),
        "time_bin_edges_s": ((9,), np.dtype(np.float64)),
        "time_bin_centers_s": ((8,), np.dtype(np.float64)),
        "fold_labels": ((3,), np.dtype(np.int64)),
        "fold_scores": (score_shape, np.dtype(np.float64)),
        "fit_status": (fit_shape, np.dtype("U16")),
        "fit_reason_codes": (fit_shape, np.dtype("U32")),
        "requested_feature_counts": (fit_shape, np.dtype(np.int64)),
        "effective_feature_counts": (fit_shape, np.dtype(np.int64)),
        "train_counts": (fit_shape, np.dtype(np.int64)),
        "test_counts": (fit_shape, np.dtype(np.int64)),
        "train_class_counts": ((*fit_shape, 2), np.dtype(np.int64)),
        "test_class_counts": ((*fit_shape, 2), np.dtype(np.int64)),
        "full_table_row_positions": ((15,), np.dtype(np.int64)),
        "full_table_trial_ids": ((15,), np.dtype(np.int64)),
        "full_table_block_ids_utf8": ((_BLOCK_ID_BYTE_COUNT,), np.dtype(np.uint8)),
        "full_table_block_id_offsets": ((16,), np.dtype(np.int64)),
        "encoded_target_values": ((2, 15), np.dtype(np.float64)),
        "eligibility_masks": ((2, 15), np.dtype(np.bool_)),
        "eligibility_reason_codes": ((2, 15), np.dtype("U32")),
        "eligibility_counts": ((2,), np.dtype(np.int64)),
        "trial_row_indices": ((13,), np.dtype(np.int64)),
        "outer_fold_ids": ((2, 13), np.dtype(np.int64)),
        "unit_selection_rules_json": ((2,), np.dtype("U48")),
        "coefficient_values": ((*fit_shape, 4), np.dtype(np.float64)),
        "coefficient_feature_ids": (feature_shape, np.dtype("U32")),
        "coefficient_feature_regions": (feature_shape, np.dtype("U16")),
        "coefficient_feature_statuses": (feature_shape, np.dtype("U16")),
        "coefficient_active_masks": (feature_shape, np.dtype(np.bool_)),
        "fitted_intercepts": (fit_shape, np.dtype(np.float64)),
        "fixed_parameters_json": (parameter_shape, np.dtype("U48")),
        "selected_parameters_json": (parameter_shape, np.dtype("U48")),
        "stage_timing_labels": ((3,), np.dtype("U16")),
        "stage_timing_seconds": ((3,), np.dtype(np.float64)),
        "total_timing_seconds": ((), np.dtype(np.float64)),
    }
    if regularization_mode == "fixed":
        contract.update(
            {
                "candidate_parameter_json": ((0,), np.dtype("U48")),
                "inner_selection_fold_ids": ((0,), np.dtype(np.int64)),
                "candidate_inner_scores": ((0,), np.dtype(np.float64)),
                "candidate_inner_statuses": ((0,), np.dtype("U16")),
                "candidate_inner_reasons": ((0,), np.dtype("U48")),
                "selected_candidate_indices": ((0,), np.dtype(np.int64)),
            }
        )
    else:
        audit_shape = (2, 3, 3, 2, 8, 15, 3)
        contract.update(
            {
                "candidate_parameter_json": ((2, 15), np.dtype("U48")),
                "inner_selection_fold_ids": ((2, 3, 13), np.dtype(np.int64)),
                "candidate_inner_scores": (audit_shape, np.dtype(np.float64)),
                "candidate_inner_statuses": (audit_shape, np.dtype("U16")),
                "candidate_inner_reasons": (audit_shape, np.dtype("U48")),
                "selected_candidate_indices": (audit_shape[:-2], np.dtype(np.int64)),
            }
        )
    assert set(contract) == set(required_array_names())
    return contract


def decoded_block_labels(arrays: Mapping[str, np.ndarray]) -> tuple[object, ...]:
    """Decode UTF-8 canonical JSON block labels without fixed-width text storage.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Primitive schema containing byte-concatenated block JSON and row offsets.

    Returns
    -------
    tuple[object, ...]
        Original mixed integer/string behavioral block labels in full-row order.
    """
    block_bytes = arrays["full_table_block_ids_utf8"]
    offsets = arrays["full_table_block_id_offsets"]
    return tuple(
        json.loads(block_bytes[start:stop].tobytes().decode("utf-8"))
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )


def encoded_block_labels(arrays: Mapping[str, np.ndarray]) -> tuple[bytes, ...]:
    """Return exact canonical JSON byte identities in full-table row order.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Primitive schema containing uint8 block-label bytes and int64 offsets.

    Returns
    -------
    tuple[bytes, ...]
        Canonical JSON byte identity for each full-table behavioral block label.
    """
    block_bytes = arrays["full_table_block_ids_utf8"]
    offsets = arrays["full_table_block_id_offsets"]
    return tuple(
        block_bytes[start:stop].tobytes()
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    )


def assign_pure_categorical_values_by_outer_group(
    arrays: dict[str, np.ndarray],
    *,
    target: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[bytes, float]]:
    """Assign binary labels to the eligible common rows of each outer group.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Complete mutable tuned result payload. ``outer_fold_ids`` has shape
        ``(target, common_neural_tensor_row)`` and target values have shape
        ``(target, full_table_row)``; all values are dimensionless.
    target : int
        Zero-based categorical target-axis position to update.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[bytes, float]]
        Target outer IDs, full-table canonical block byte labels, common-row
        full-table positions, and one pure ``0.0``/``1.0`` class per eligible
        behavioral group. Ineligible full-table rows are not modified.
    """
    outer = arrays["outer_fold_ids"][target]
    trial_rows = arrays["trial_row_indices"]
    # Merge rows 10--12 only in this local fixture. Target 0 uses rows 10 and
    # 12, while target 1 uses rows 10 and 11; each target therefore sees two
    # fold-2 groups without changing the shared common-row union.
    payloads = dynamic_block_payloads(arrays)
    payloads[11] = payloads[10]
    payloads[12] = payloads[10]
    replace_dynamic_block_payloads(arrays, payloads)
    arrays["inner_selection_fold_ids"] = rebuild_dynamic_inner_assignments(arrays)
    blocks = np.asarray(encoded_block_labels(arrays), dtype=object)
    eligible_common_rows = np.flatnonzero(outer >= 0)
    group_to_outer: dict[bytes, int] = {}
    for common_row in eligible_common_rows:
        group = blocks[int(trial_rows[common_row])]
        group_to_outer[group] = int(outer[common_row])

    group_values: dict[bytes, float] = {}
    for fold in arrays["fold_labels"]:
        groups = [
            group
            for group, group_fold in group_to_outer.items()
            if group_fold == int(fold)
        ]
        assert len(groups) == 2
        group_values[groups[0]] = 0.0
        group_values[groups[1]] = 1.0

    for common_row in eligible_common_rows:
        full_row = int(trial_rows[common_row])
        arrays["encoded_target_values"][target, full_row] = group_values[blocks[full_row]]
    return outer, blocks, trial_rows, group_values


def refresh_categorical_outer_class_counts(
    arrays: dict[str, np.ndarray],
    *,
    target: int,
) -> None:
    """Synchronize per-fit binary class counts with a target's outer assignments.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Complete mutable result payload with binary encoded target values in
        ``{0, 1}`` and outer IDs on the common-neural-tensor row axis.
    target : int
        Zero-based categorical target-axis position whose count arrays are
        overwritten. Counts are dimensionless trial counts.

    Returns
    -------
    None
        Mutates the target slice of ``train_class_counts`` and
        ``test_class_counts`` with shape ``(region, representation, time,
        fold, class)``.
    """
    outer = arrays["outer_fold_ids"][target]
    values = arrays["encoded_target_values"][target, arrays["trial_row_indices"]]
    for region, representation, time, fold in np.ndindex(arrays["fit_status"].shape[1:]):
        index = (target, region, representation, time, fold)
        train = (outer >= 0) & (outer != fold)
        test = outer == fold
        arrays["train_class_counts"][index] = (
            np.count_nonzero(values[train] == 0),
            np.count_nonzero(values[train] == 1),
        )
        arrays["test_class_counts"][index] = (
            np.count_nonzero(values[test] == 0),
            np.count_nonzero(values[test] == 1),
        )


def assert_grouped_split_integrity(arrays: Mapping[str, np.ndarray]) -> None:
    """Assert outer and nested assignments never split eligible behavioral groups.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Fixed or tuned primitive arrays with global common-row fold positions.

    Returns
    -------
    None
        Raises an assertion error if any target's eligible rows sharing one
        canonical JSON block identity receive different outer or inner folds.
    """
    block_ids = encoded_block_labels(arrays)
    for target_index, outer_ids in enumerate(arrays["outer_fold_ids"]):
        eligible_rows = np.flatnonzero(outer_ids >= 0)
        eligible_block_ids = {block_ids[row_index] for row_index in eligible_rows}
        for block_json in eligible_block_ids:
            matching_rows = np.array(
                [row_index for row_index in eligible_rows if block_ids[row_index] == block_json],
                dtype=np.int64,
            )
            assert np.unique(outer_ids[matching_rows]).size == 1
        if arrays["inner_selection_fold_ids"].size:
            for inner_ids in arrays["inner_selection_fold_ids"][target_index]:
                for block_json in eligible_block_ids:
                    matching_rows = np.array(
                        [
                            row_index
                            for row_index in eligible_rows
                            if block_ids[row_index] == block_json
                        ],
                        dtype=np.int64,
                    )
                    selected_rows = matching_rows[inner_ids[matching_rows] >= 0]
                    if selected_rows.size:
                        assert np.unique(inner_ids[selected_rows]).size == 1


def assert_primitive_schema(
    arrays: Mapping[str, np.ndarray],
    meta: Mapping[str, object],
    scientific_config: Mapping[str, object],
    *,
    regularization_mode: str,
) -> None:
    """Assert exact labels, dtypes, shapes, row identities, axes, units, and audit sentinels.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Loaded primitive results schema.
    meta : mapping[str, object]
        Loaded required schema/provenance mapping.
    scientific_config : mapping[str, object]
        Complete config serializer payload paired with this result.
    regularization_mode : {"fixed", "tuned"}
        Expected applicability of nested-CV members.
    """
    assert set(arrays) == set(required_array_names())
    for array_name, (expected_shape, expected_dtype) in expected_array_contract(
        regularization_mode
    ).items():
        assert arrays[array_name].shape == expected_shape
        assert arrays[array_name].dtype == expected_dtype
    assert arrays["target_labels"].tolist() == list(_TARGETS)
    assert arrays["target_families"].tolist() == ["categorical", "numerical"]
    assert arrays["target_status"].tolist() == ["available", "available"]
    assert arrays["target_unavailable_reasons"].tolist() == ["", ""]
    assert arrays["region_labels"].tolist() == list(_REGIONS)
    assert arrays["representation_labels"].tolist() == list(_REPRESENTATIONS)
    assert arrays["metric_labels"].tolist() == list(_METRICS)
    assert arrays["metric_labels"].dtype.kind == "U"
    assert arrays["time_bin_edges_s"].tolist() == [
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]
    np.testing.assert_allclose(
        arrays["time_bin_centers_s"],
        (arrays["time_bin_edges_s"][:-1] + arrays["time_bin_edges_s"][1:]) / 2.0,
    )
    assert scientific_config["bin_width_ms"] == 500
    assert scientific_config["frozen_controls"]["window_start_s"] == -2.0
    assert scientific_config["frozen_controls"]["window_end_s"] == 2.0
    assert arrays["fold_scores"].shape == (2, 3, 2, 3, 8, 3)
    assert arrays["fold_scores"].dtype == np.dtype(np.float64)
    assert arrays["fit_status"].shape == (2, 3, 2, 8, 3)
    assert arrays["fit_status"].dtype.kind == "U"
    assert arrays["fit_reason_codes"].shape == arrays["fit_status"].shape
    assert arrays["train_counts"].shape == arrays["fit_status"].shape
    assert arrays["test_counts"].shape == arrays["fit_status"].shape
    assert np.all(arrays["train_counts"] == 8)
    assert np.all(arrays["test_counts"] == 4)
    assert arrays["train_class_counts"].shape == (*arrays["fit_status"].shape, 2)
    assert arrays["test_class_counts"].shape == (*arrays["fit_status"].shape, 2)
    assert np.all(arrays["train_class_counts"][0] == (4, 4))
    assert np.all(arrays["test_class_counts"][0] == (2, 2))
    assert arrays["full_table_row_positions"].tolist() == list(range(15))
    assert arrays["full_table_trial_ids"].tolist() == list(range(15))
    assert decoded_block_labels(arrays) == _BLOCK_VALUES
    assert encoded_block_labels(arrays) == _BLOCK_ID_JSON_BYTES
    assert isinstance(decoded_block_labels(arrays)[0], int)
    assert isinstance(decoded_block_labels(arrays)[2], str)
    assert decoded_block_labels(arrays)[6:8] == ("\u00b5-block-\u00e4", "\u00b5-block-\u00e4")
    assert encoded_block_labels(arrays)[6] == b'"\xc2\xb5-block-\xc3\xa4"'
    assert decoded_block_labels(arrays)[10] == _BLOCK_VALUES[10]
    assert len(decoded_block_labels(arrays)[10]) > 32
    assert arrays["trial_row_indices"].tolist() == list(range(13))
    assert arrays["encoded_target_values"].shape == (2, 15)
    assert arrays["eligibility_masks"].shape == (2, 15)
    assert arrays["eligibility_reason_codes"].shape == (2, 15)
    assert arrays["eligibility_counts"].tolist() == [12, 12]
    assert np.all(arrays["eligibility_counts"] == arrays["eligibility_masks"].sum(axis=1))
    assert np.isnan(arrays["encoded_target_values"][0, 11])
    assert np.isnan(arrays["encoded_target_values"][1, 12])
    assert np.isnan(arrays["encoded_target_values"][:, 13:]).all()
    assert np.all(np.abs(arrays["encoded_target_values"][1, :12]) <= 1.0)
    assert arrays["outer_fold_ids"].shape == (2, 13)
    for target_index, outer_folds in enumerate(arrays["outer_fold_ids"]):
        eligible = arrays["eligibility_masks"][target_index, arrays["trial_row_indices"]]
        assert np.array_equal(outer_folds >= 0, eligible)
        assert np.count_nonzero(outer_folds == -1) == 1
        assert [np.count_nonzero(outer_folds == fold) for fold in range(3)] == [4, 4, 4]
        assert arrays["eligibility_counts"][target_index] - 4 == 8
    assert_grouped_split_integrity(arrays)
    expected_widths = np.array([[2, 2], [2, 2], [4, 4]], dtype=np.int64)
    expected_requested_widths = np.array([[2, 2], [2, 2], [4, 4]], dtype=np.int64)
    for region_index, representation_index in np.ndindex(expected_widths.shape):
        width = expected_widths[region_index, representation_index]
        feature_ids = arrays["coefficient_feature_ids"][region_index, representation_index]
        statuses = arrays["coefficient_feature_statuses"][region_index, representation_index]
        active = arrays["coefficient_active_masks"][region_index, representation_index]
        assert np.all(feature_ids[:width] != "")
        assert np.all(feature_ids[width:] == "")
        assert np.all(statuses[:width] == "available")
        assert np.all(statuses[width:] == "padding")
        assert np.all(active[:width])
        assert not np.any(active[width:])
        coefficients = arrays["coefficient_values"][:, region_index, representation_index]
        assert np.isfinite(coefficients[..., :width]).all()
        assert np.isnan(coefficients[..., width:]).all()
        assert np.all(
            arrays["effective_feature_counts"][:, region_index, representation_index]
            == width
        )
        assert np.all(
            arrays["requested_feature_counts"][:, region_index, representation_index]
            == expected_requested_widths[region_index, representation_index]
        )
    assert arrays["coefficient_feature_ids"][0, 0, :2].tolist() == ["PFC:PC1", "PFC:PC2"]
    assert arrays["coefficient_feature_ids"][0, 1, :2].tolist() == ["probe-pfc:11", "probe-pfc:19"]
    assert arrays["coefficient_feature_ids"][2, 0, :4].tolist() == [
        "PFC:PC1",
        "PFC:PC2",
        "HPC:PC1",
        "HPC:PC2",
    ]
    assert arrays["coefficient_feature_ids"][2, 1].tolist() == [
        "probe-pfc:11",
        "probe-pfc:19",
        "probe-hpc:5",
        "probe-hpc:17",
    ]
    assert arrays["fitted_intercepts"].shape == arrays["fit_status"].shape
    assert arrays["fixed_parameters_json"].shape == (2, 3, 3, 2, 8)
    assert '"C"' in arrays["fixed_parameters_json"][0, 0, 0, 0, 0]
    assert '"alpha"' in arrays["fixed_parameters_json"][1, 0, 0, 0, 0]
    assert arrays["stage_timing_labels"].shape == arrays["stage_timing_seconds"].shape
    assert arrays["total_timing_seconds"].ndim == 0
    assert meta["analysis_version"] == ANALYSIS_VERSION
    assert meta["parameters"]["regularization_mode"] == regularization_mode
    assert meta["parameters"]["bin_width_ms"] == scientific_config["bin_width_ms"] == 500
    assert meta["parameters"]["candidate_selection_metrics"] == {
        "current_action": "balanced_accuracy",
        "relative_doubt": "r2",
    }
    assert set(meta["axes"]) == set(arrays)
    assert meta["axes"] == complete_axis_mapping(
        arrays,
        regularization_mode=regularization_mode,
    )
    for array_name, array in arrays.items():
        assert len(meta["axes"][array_name]) == array.ndim
    assert meta["units"]["encoded_target_values"]["current_action"] == "encoded class"
    assert meta["units"]["encoded_target_values"]["relative_doubt"] == "unitless"
    assert meta["units"]["fold_scores"]["current_action"] == {
        "balanced_accuracy": "fraction",
        "auc": "fraction",
    }
    assert meta["units"]["fold_scores"]["relative_doubt"] == {
        "r2": "coefficient_of_determination"
    }
    assert meta["units"]["candidate_inner_scores"]["current_action"] == {
        "balanced_accuracy": "fraction"
    }
    assert meta["units"]["candidate_inner_scores"]["relative_doubt"] == {
        "r2": "coefficient_of_determination"
    }
    assert meta["units"]["coefficient_values"]["current_action"] == (
        "log-odds change per pooled training standard deviation"
    )
    assert meta["units"]["coefficient_values"]["relative_doubt"] == (
        "unitless per pooled training standard deviation"
    )
    assert meta["units"]["fitted_intercepts"] == {
        "current_action": "log-odds",
        "relative_doubt": "unitless",
    }
    if regularization_mode == "fixed":
        for array_name in (
            "candidate_parameter_json",
            "inner_selection_fold_ids",
            "candidate_inner_scores",
            "candidate_inner_statuses",
            "candidate_inner_reasons",
            "selected_candidate_indices",
        ):
            assert arrays[array_name].shape == (0,)
            assert meta["axes"][array_name] == ["not_applicable"]
        assert not np.any(arrays["selected_parameters_json"] != "")
    else:
        expected_audit_shape = (2, 3, 3, 2, 8, 15, 3)
        assert arrays["candidate_parameter_json"].shape == (2, 15)
        assert arrays["inner_selection_fold_ids"].shape == (2, 3, 13)
        assert arrays["candidate_inner_scores"].shape == expected_audit_shape
        assert arrays["candidate_inner_statuses"].shape == expected_audit_shape
        assert arrays["candidate_inner_reasons"].shape == expected_audit_shape
        assert arrays["selected_candidate_indices"].shape == expected_audit_shape[:-2]
        for target_index, target_family in enumerate(("categorical", "numerical")):
            candidates = declared_candidates(scientific_config, target_family)
            expected_order = [json.dumps(item, sort_keys=True) for item in candidates]
            assert arrays["candidate_parameter_json"][target_index].tolist() == expected_order
            outer_folds = arrays["outer_fold_ids"][target_index]
            for outer_fold in range(3):
                inner_ids = arrays["inner_selection_fold_ids"][target_index, outer_fold]
                assert np.all(inner_ids[outer_folds == outer_fold] == -1)
                assert np.all(inner_ids[outer_folds == -1] == -1)
                assert set(inner_ids[inner_ids >= 0]) == {0, 1, 2}
        invalid_position = (1, 2, 2, 1, 1, 14, 2)
        assert np.isnan(arrays["candidate_inner_scores"][invalid_position])
        assert arrays["candidate_inner_statuses"][invalid_position] == "invalid"
        assert arrays["candidate_inner_reasons"][invalid_position] == "nonfinite_inner_score"
        assert np.isnan(arrays["candidate_inner_scores"][..., 14, 2]).all()
        assert np.all(arrays["candidate_inner_statuses"][..., 14, 2] == "invalid")
        assert np.all(
            arrays["candidate_inner_reasons"][..., 14, 2] == "nonfinite_inner_score"
        )
        finite_audit_scores = arrays["candidate_inner_scores"][
            np.isfinite(arrays["candidate_inner_scores"])
        ]
        assert np.all((0.0 <= finite_audit_scores) & (finite_audit_scores <= 1.0))
        assert np.unique(finite_audit_scores).size == finite_audit_scores.size
        for target_index in range(2):
            for indexes in np.ndindex(3, 3, 2, 8):
                outer_fold, region_index, representation_index, time_index = indexes
                selected_index = arrays["selected_candidate_indices"][
                    target_index,
                    outer_fold,
                    region_index,
                    representation_index,
                    time_index,
                ]
                assert selected_index == sum((target_index, *indexes)) % 14
                selected_parameter = arrays["selected_parameters_json"][
                    target_index,
                    outer_fold,
                    region_index,
                    representation_index,
                    time_index,
                ]
                assert selected_parameter == arrays["candidate_parameter_json"][
                    target_index,
                    selected_index,
                ]
                scores = arrays["candidate_inner_scores"][
                    target_index,
                    outer_fold,
                    region_index,
                    representation_index,
                    time_index,
                ]
                candidate_means = np.nanmean(scores, axis=1)
                assert selected_index == int(np.argmax(candidate_means))
                other_means = np.delete(candidate_means, selected_index)
                assert candidate_means[selected_index] > np.max(other_means)


@pytest.mark.parametrize("regularization_mode", ("fixed", "tuned"))
def test_saved_run_round_trip_preserves_complete_primitive_schema(tmp_path, regularization_mode):
    """Fixed/tuned result directories preserve all scientific schema and provenance members."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / f"run-{regularization_mode}"
    arrays, meta, manifest, scientific_config = save_run_fixture(
        run_directory,
        input_paths,
        regularization_mode=regularization_mode,
    )

    loaded = results.load_task_decoding_run(run_directory)

    assert loaded["meta"] == meta
    assert loaded["input_manifest"] == manifest
    assert loaded["scientific_config"] == scientific_config
    assert loaded["run_fingerprint"] == "run-fingerprint-test"
    for array_name, expected in arrays.items():
        np.testing.assert_array_equal(loaded["arrays"][array_name], expected)
    assert_primitive_schema(
        loaded["arrays"],
        loaded["meta"],
        loaded["scientific_config"],
        regularization_mode=regularization_mode,
    )


def test_npz_meta_member_is_safe_json_and_covers_every_array_axis(tmp_path):
    """NPZ ``meta`` is named safe JSON and explicitly documents scalars and 1-D arrays."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "run"
    arrays, meta, _manifest, _config = save_run_fixture(
        run_directory,
        input_paths,
        regularization_mode="fixed",
    )

    with np.load(run_directory / _RESULT_FILE, allow_pickle=False) as archive:
        assert "meta" in archive.files
        assert archive["meta"].ndim == 0
        assert json.loads(str(archive["meta"].item())) == meta
    assert set(meta["axes"]) == set(required_array_names())
    assert meta["axes"]["target_labels"] == ["target"]
    assert meta["axes"]["total_timing_seconds"] == []
    assert meta["axes"]["candidate_inner_scores"] == ["not_applicable"]
    assert set(arrays) == set(required_array_names())


@pytest.mark.parametrize("array_name", required_array_names())
def test_loader_rejects_each_missing_required_primitive_member(array_name, tmp_path):
    """Directory loading rejects every absent primitive array rather than passing through."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "run"
    arrays, meta, _manifest, _config = save_run_fixture(
        run_directory,
        input_paths,
        regularization_mode="tuned",
    )
    altered_arrays = {name: value.copy() for name, value in arrays.items()}
    altered_arrays.pop(array_name)
    write_raw_npz(run_directory, altered_arrays, meta)

    with pytest.raises(ValueError, match="schema|array|required|missing"):
        results.load_task_decoding_run(run_directory)


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("bad_integer_dtype", "dtype|integer"),
        ("unsafe_object_dtype", "dtype|object|pickle"),
        ("bad_shape", "shape|schema"),
        ("bad_metric_order", "metric|label|order"),
        ("bad_region_order", "region|label|order"),
        ("bad_representation_order", "representation|label|order"),
        ("axes_order_mismatch", "axes|axis|order"),
        ("fixed_tuned_member", "mode|fixed|candidate"),
        ("tuned_malformed_shape", "shape|candidate|inner"),
        ("cross_count_inconsistency", "count|train|fold"),
        ("cross_identity_inconsistency", "identity|trial|row"),
        ("candidate_nan_valid", "candidate|nonfinite|status"),
        ("candidate_invalid_missing_reason", "candidate|reason|invalid"),
        ("selected_index_parameter_mismatch", "selected|candidate|parameter"),
        ("outer_test_inner_assignment", "inner|outer|test"),
        ("target_ineligible_inner_assignment", "inner|eligible|target"),
        ("outer_block_leakage", "outer|block|group"),
        ("inner_block_leakage", "inner|block|group"),
        ("coefficient_padding_mismatch", "coefficient|padding|feature"),
        ("requested_effective_disagreement", "requested|effective|feature"),
    ),
)
def test_loader_rejects_invalid_schema_members_and_cross_array_inconsistencies(
    kind,
    failure_pattern,
    tmp_path,
):
    """Each isolated malformed primitive fails for its own schema or leakage reason."""
    input_paths = write_input_fixture(tmp_path)
    tuned_kinds = {
        "tuned_malformed_shape",
        "candidate_nan_valid",
        "candidate_invalid_missing_reason",
        "selected_index_parameter_mismatch",
        "outer_test_inner_assignment",
        "target_ineligible_inner_assignment",
        "inner_block_leakage",
    }
    mode = "tuned" if kind in tuned_kinds else "fixed"
    run_directory = tmp_path / "run"
    arrays, meta, _manifest, _config = save_run_fixture(
        run_directory,
        input_paths,
        regularization_mode=mode,
    )
    altered_arrays = {name: value.copy() for name, value in arrays.items()}
    altered_meta = json.loads(json.dumps(meta))
    if kind == "bad_integer_dtype":
        altered_arrays["outer_fold_ids"] = altered_arrays["outer_fold_ids"].astype(np.float64)
    elif kind == "unsafe_object_dtype":
        altered_arrays["fit_status"] = altered_arrays["fit_status"].astype(object)
    elif kind == "bad_shape":
        altered_arrays["fit_status"] = altered_arrays["fit_status"][..., :2]
    elif kind == "bad_metric_order":
        altered_arrays["metric_labels"][[0, 1]] = altered_arrays["metric_labels"][[1, 0]]
    elif kind == "bad_region_order":
        altered_arrays["region_labels"][[0, 1]] = altered_arrays["region_labels"][[1, 0]]
    elif kind == "bad_representation_order":
        altered_arrays["representation_labels"][[0, 1]] = altered_arrays[
            "representation_labels"
        ][[1, 0]]
    elif kind == "axes_order_mismatch":
        altered_meta["axes"]["fold_scores"] = list(reversed(altered_meta["axes"]["fold_scores"]))
    elif kind == "fixed_tuned_member":
        altered_arrays["candidate_parameter_json"] = np.array([["bad"]], dtype="U16")
    elif kind == "tuned_malformed_shape":
        altered_arrays["candidate_inner_scores"] = altered_arrays["candidate_inner_scores"][..., :2]
    elif kind == "cross_count_inconsistency":
        altered_arrays["train_counts"][0, 0, 0, 0, 0] = 7
    elif kind == "cross_identity_inconsistency":
        altered_arrays["trial_row_indices"][0] = 14
    elif kind == "candidate_nan_valid":
        altered_arrays["candidate_inner_scores"][0, 0, 0, 0, 0, 1, 0] = np.nan
    elif kind == "candidate_invalid_missing_reason":
        altered_arrays["candidate_inner_reasons"][0, 0, 0, 0, 0, 14, 2] = ""
    elif kind == "selected_index_parameter_mismatch":
        altered_arrays["selected_parameters_json"][0, 0, 0, 0, 0] = altered_arrays[
            "candidate_parameter_json"
        ][0, 1]
    elif kind == "outer_test_inner_assignment":
        altered_arrays["inner_selection_fold_ids"][0, 0, 0] = 0
    elif kind == "target_ineligible_inner_assignment":
        altered_arrays["inner_selection_fold_ids"][0, 0, 11] = 0
    elif kind == "outer_block_leakage":
        altered_arrays["outer_fold_ids"][0, 1] = 1
        altered_arrays["outer_fold_ids"][0, 5] = 0
    elif kind == "inner_block_leakage":
        altered_arrays["inner_selection_fold_ids"][0, 0, 5] = 1
    elif kind == "coefficient_padding_mismatch":
        altered_arrays["coefficient_values"][0, 0, 0, 0, 0, 2] = 0.0
    else:
        altered_arrays["requested_feature_counts"][0, 0, 1, 0, 0] = 1
    write_raw_npz(run_directory, altered_arrays, altered_meta)

    with pytest.raises(ValueError, match=failure_pattern):
        results.load_task_decoding_run(run_directory)


def test_loader_rejects_missing_meta_versions_axes_and_config_identity_mismatch(tmp_path):
    """Loader validates named meta, schema/version, every axes key, and config/meta identity."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "run"
    arrays, meta, _manifest, _config = save_run_fixture(
        run_directory,
        input_paths,
        regularization_mode="fixed",
    )
    write_raw_npz(run_directory, arrays, None)
    with pytest.raises(ValueError, match="meta|schema"):
        results.load_task_decoding_run(run_directory)

    invalid_meta = json.loads(json.dumps(meta))
    invalid_meta["schema_version"] = results.RESULT_SCHEMA_VERSION + 1
    write_raw_npz(run_directory, arrays, invalid_meta)
    with pytest.raises(ValueError, match="schema"):
        results.load_task_decoding_run(run_directory)

    invalid_meta = json.loads(json.dumps(meta))
    invalid_meta["analysis_version"] = "other-analysis-version"
    write_raw_npz(run_directory, arrays, invalid_meta)
    with pytest.raises(ValueError, match="analysis.version|version"):
        results.load_task_decoding_run(run_directory)

    invalid_meta = json.loads(json.dumps(meta))
    invalid_meta["axes"].pop("fold_scores")
    write_raw_npz(run_directory, arrays, invalid_meta)
    with pytest.raises(ValueError, match="axes|schema"):
        results.load_task_decoding_run(run_directory)

    write_raw_npz(run_directory, arrays, meta)
    config_payload = json.loads((run_directory / _CONFIG_FILE).read_text(encoding="utf-8"))
    config_payload["regularization_mode"] = "tuned"
    (run_directory / _CONFIG_FILE).write_text(json.dumps(config_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="config|mode|identity"):
        results.load_task_decoding_run(run_directory)


@pytest.mark.parametrize("support_name", (_CONFIG_FILE, _MANIFEST_FILE, _FEATURE_COPY_FILE))
def test_loader_rejects_missing_required_supporting_artifact(support_name, tmp_path):
    """Directory loading owns supporting config, manifest, and exact feature-copy validation."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "run"
    save_run_fixture(run_directory, input_paths, regularization_mode="fixed")
    (run_directory / support_name).unlink()

    with pytest.raises(ValueError, match="config|manifest|feature|parameter|support"):
        results.load_task_decoding_run(run_directory)


def test_saved_feature_parameter_copy_is_exact_and_portable_without_original_session(tmp_path):
    """Transferred results load after source deletion, then reject a changed copied feature file."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "run"
    save_run_fixture(run_directory, input_paths, regularization_mode="fixed")
    copied_parameters = run_directory / _FEATURE_COPY_FILE

    assert copied_parameters.read_bytes() == input_paths["feature_parameters"].read_bytes()
    imported_directory = tmp_path / "portable-import" / "run"
    shutil.copytree(run_directory, imported_directory)
    shutil.rmtree(input_paths["session_root"])
    loaded_meta = results.load_task_decoding_run(imported_directory)["meta"]
    assert loaded_meta["analysis_version"] == ANALYSIS_VERSION

    copied_parameters.write_bytes(b'{"model":"altered"}\n')
    with pytest.raises(ValueError, match="feature|parameter|SHA|manifest"):
        results.load_task_decoding_run(run_directory)


def test_input_manifest_hashes_inclusive_boundary_and_never_opens_large_binary(
    tmp_path,
    monkeypatch,
):
    """Exactly 64 MiB streams SHA-256; larger binary content is never opened or read."""
    input_paths = write_input_fixture(tmp_path, include_threshold_files=True)
    original_open = Path.open
    original_read_bytes = Path.read_bytes
    original_builtin_open = builtins.open

    def guarded_open(path: Path, *args, **kwargs):
        """Forbid a large-binary open while allowing bounded structured-file streaming."""
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == input_paths["spike_binary"] and "r" in mode:
            raise AssertionError("Large binary content must not open for manifest identity.")
        return original_open(path, *args, **kwargs)

    def guarded_read_bytes(path: Path) -> bytes:
        """Forbid whole large-binary reads while keeping ordinary small fixture checks valid."""
        if path == input_paths["spike_binary"]:
            raise AssertionError("Large binary content must not be read for manifest identity.")
        return original_read_bytes(path)

    def guarded_builtin_open(file, *args, **kwargs):
        """Forbid alternate large-binary streaming seams using builtin ``open``."""
        mode = args[0] if args else kwargs.get("mode", "r")
        if isinstance(file, (str, os.PathLike)) and Path(file) == input_paths["spike_binary"]:
            if "r" in mode:
                raise AssertionError("Large binary content must not open for manifest identity.")
        return original_builtin_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    manifest = build_manifest(input_paths, include_threshold_files=True)

    assert manifest["identity_policy"] == {
        "hash_algorithm": "sha256",
        "hash_threshold_bytes": 64 * _MEBIBYTE,
        "mtime_resolution": "whole_seconds",
    }
    for input_name in (
        "neural_session",
        "augmented_trials",
        "trial_feature_parameters",
        "boundary_structured_json",
    ):
        identity = manifest["files"][input_name]
        resolved_path = input_paths["session_root"] / identity["relative_path"]
        expected_hash = hashlib.sha256()
        with resolved_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                expected_hash.update(chunk)
        assert identity["relative_path"]
        assert identity["size_bytes"] == resolved_path.stat().st_size
        assert identity["mtime_seconds"] == int(resolved_path.stat().st_mtime)
        assert identity["sha256"] == expected_hash.hexdigest()
    large_identity = manifest["files"]["aligned_spike_binary"]
    assert large_identity["relative_path"] == "probes/probe-a/sorter/spike_clusters.npy"
    assert large_identity["size_bytes"] == 64 * _MEBIBYTE + 1
    assert large_identity["mtime_seconds"] == int(input_paths["spike_binary"].stat().st_mtime)
    assert "sha256" not in large_identity
    assert [item["relative_path"] for item in manifest["file_lists"]["pfc_sorter"]] == [
        "probes/probe-a/sorter/cluster_info.tsv",
        "probes/probe-a/sorter/params.py",
    ]
    for item in manifest["file_lists"]["pfc_sorter"]:
        listed_path = input_paths["session_root"] / item["relative_path"]
        assert item["size_bytes"] == listed_path.stat().st_size
        assert item["mtime_seconds"] == int(listed_path.stat().st_mtime)
        assert item["sha256"] == hashlib.sha256(listed_path.read_bytes()).hexdigest()
    assert all(
        item["relative_path"] != "probes/probe-a/sorter/unlisted.txt"
        for item in manifest["file_lists"]["pfc_sorter"]
    )


@pytest.mark.parametrize("container_name", ("files", "file_lists"))
def test_input_manifest_rejects_outside_session_paths(container_name, tmp_path):
    """Both named files and explicit file-list members must be session-contained."""
    input_paths = write_input_fixture(tmp_path)
    outside_path = tmp_path / "outside.json"
    outside_path.write_text("{}\n", encoding="ascii")
    files = {"neural_session": input_paths["neural_session"]}
    file_lists = {"pfc_sorter": (input_paths["cluster_info"],)}
    if container_name == "files":
        files["outside"] = outside_path
    else:
        file_lists["pfc_sorter"] = (outside_path,)

    with pytest.raises(ValueError, match="session|relative|contain"):
        results.build_input_manifest(
            session_root=input_paths["session_root"],
            files=files,
            file_lists=file_lists,
        )


def write_scoped_source_tree(tmp_path: Path, *, name: str) -> Path:
    """Create the exact reviewed task package/direct-dependency source layout.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned parent directory.
    name : str
        Repository-root name varied for portable source identity comparisons.

    Returns
    -------
    pathlib.Path
        Synthetic repository containing every exact scoped file and one document.
    """
    repository_root = tmp_path / name
    for relative_path in _DIRECT_RUNTIME_PATHS:
        path = repository_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="ascii")
    task_directory = repository_root / "src" / "neural_analysis" / "task_decoding"
    task_directory.mkdir(parents=True, exist_ok=True)
    for filename in _TASK_DECODING_FILES:
        (task_directory / filename).write_text(f"# {filename}\n", encoding="ascii")
    document = repository_root / "docs" / "unrelated.md"
    document.parent.mkdir(parents=True)
    document.write_text("unrelated\n", encoding="ascii")
    return repository_root


def expected_scientific_source_paths() -> tuple[str, ...]:
    """Return every exact sorted Section-8.1 scoped source/dependency path.

    Returns
    -------
    tuple[str, ...]
        Dynamic task package Python paths plus explicit external dependencies,
        project metadata, and environment lock.
    """
    task_paths = tuple(
        f"src/neural_analysis/task_decoding/{filename}"
        for filename in _TASK_DECODING_FILES
    )
    return tuple(sorted((*task_paths, *_DIRECT_RUNTIME_PATHS)))


def test_scientific_source_fingerprint_is_portable_scoped_and_includes_extra_package_python(
    tmp_path,
):
    """Source identity is root-stable, ignores docs, and includes package Python files."""
    repository_a = write_scoped_source_tree(tmp_path, name="workstation-repository")
    repository_b = tmp_path / "cluster-repository"
    shutil.copytree(repository_a, repository_b)
    fingerprint_a = results.scientific_source_fingerprint(repository_root=repository_a)
    fingerprint_b = results.scientific_source_fingerprint(repository_root=repository_b)

    assert fingerprint_a == fingerprint_b
    assert [entry["relative_path"] for entry in fingerprint_a["files"]] == list(
        expected_scientific_source_paths()
    )
    baseline = fingerprint_a["fingerprint"]
    (repository_a / "docs" / "unrelated.md").write_text("changed\n", encoding="ascii")
    assert (
        results.scientific_source_fingerprint(repository_root=repository_a)["fingerprint"]
        == baseline
    )
    extra_path = repository_a / "src/neural_analysis/task_decoding/new_science.py"
    extra_path.write_text("# extra scoped science\n", encoding="ascii")
    extra_fingerprint = results.scientific_source_fingerprint(repository_root=repository_a)
    assert extra_fingerprint["fingerprint"] != baseline
    assert "src/neural_analysis/task_decoding/new_science.py" in [
        item["relative_path"] for item in extra_fingerprint["files"]
    ]


@pytest.mark.parametrize("relative_path", expected_scientific_source_paths())
def test_scientific_source_fingerprint_changes_when_each_scoped_file_changes(
    relative_path,
    tmp_path,
):
    """Every required task/dependency/project/lock path contributes content to identity."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    baseline = results.scientific_source_fingerprint(repository_root=repository_root)["fingerprint"]
    source_path = repository_root / relative_path
    source_path.write_text(
        source_path.read_text(encoding="ascii") + "# changed\n",
        encoding="ascii",
    )

    assert (
        results.scientific_source_fingerprint(repository_root=repository_root)["fingerprint"]
        != baseline
    )


@pytest.mark.parametrize("relative_path", expected_scientific_source_paths())
def test_scientific_source_fingerprint_rejects_each_missing_scoped_file(relative_path, tmp_path):
    """Missing any exact scoped package/dependency/project/lock file fails visibly."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    (repository_root / relative_path).unlink()

    with pytest.raises(ValueError, match="required|missing|source"):
        results.scientific_source_fingerprint(repository_root=repository_root)


def patch_git_commands(
    monkeypatch,
    *,
    status_output: str,
    omitted_dependency: str | None = None,
) -> list[tuple[str, ...]]:
    """Replace Git subprocess calls with deterministic cleanliness/tracking replies.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture replacing ``results.subprocess.run`` without repository mutation.
    status_output : str
        Porcelain records returned by the scoped-cleanliness implementation.
    omitted_dependency : str or None
        Scoped path omitted from mocked ``git ls-files`` output.

    Returns
    -------
    list[tuple[str, ...]]
        Commands observed by the test; dirty paths may short-circuit before tracking.
    """
    calls: list[tuple[str, ...]] = []

    def fake_run(command: Sequence[str], *args, **kwargs) -> subprocess.CompletedProcess[str]:
        """Return mocked Git replies while rejecting unrelated subprocess execution."""
        command_tuple = tuple(command)
        calls.append(command_tuple)
        if "status" in command_tuple:
            return subprocess.CompletedProcess(command, 0, stdout=status_output, stderr="")
        if "ls-files" in command_tuple:
            tracked_paths = [
                path for path in expected_scientific_source_paths() if path != omitted_dependency
            ]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout="\n".join(tracked_paths) + "\n",
                stderr="",
            )
        raise AssertionError(f"Unexpected subprocess command: {command_tuple!r}")

    monkeypatch.setattr(results.subprocess, "run", fake_run)
    return calls


@pytest.mark.parametrize("relative_path", _DIRECT_RUNTIME_PATHS)
def test_preparation_cleanliness_rejects_every_dirty_direct_dependency(
    relative_path,
    tmp_path,
    monkeypatch,
):
    """Every explicit direct dependency, including project utils/lock, blocks preparation."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    calls = patch_git_commands(monkeypatch, status_output=f" M {relative_path}\n")

    with pytest.raises(ValueError, match="clean|source|dirty"):
        results.validate_scientific_source_cleanliness(repository_root=repository_root)
    assert any("status" in call for call in calls)


@pytest.mark.parametrize(
    "status_output, should_raise",
    (
        (" M docs/unrelated.md\n", False),
        (" M src/neural_analysis/task_decoding/modeling.py\n", True),
        ("?? src/neural_analysis/task_decoding/new_science.py\n", True),
        (
            "?? src/neural_analysis/task_decoding/__pycache__/pipeline.cpython-314.pyc\n",
            False,
        ),
    ),
)
def test_preparation_cleanliness_handles_package_untracked_missing_and_unrelated_state(
    status_output,
    should_raise,
    tmp_path,
    monkeypatch,
):
    """Scientific Python blocks while unrelated docs and bytecode do not."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    calls = patch_git_commands(
        monkeypatch,
        status_output=status_output,
    )

    if should_raise:
        with pytest.raises(ValueError, match="clean|tracked|untracked|source"):
            results.validate_scientific_source_cleanliness(repository_root=repository_root)
    else:
        results.validate_scientific_source_cleanliness(repository_root=repository_root)
        assert any("ls-files" in call for call in calls)


@pytest.mark.parametrize("omitted_dependency", _DIRECT_RUNTIME_PATHS)
def test_preparation_cleanliness_requires_every_direct_dependency_to_be_tracked(
    omitted_dependency,
    tmp_path,
    monkeypatch,
):
    """Every explicit direct dependency must be tracked for persistent preparation."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    patch_git_commands(
        monkeypatch,
        status_output="",
        omitted_dependency=omitted_dependency,
    )

    with pytest.raises(ValueError, match="tracked|missing|source"):
        results.validate_scientific_source_cleanliness(repository_root=repository_root)


def test_run_fingerprint_uses_real_scientific_payload_and_is_root_and_key_order_stable(tmp_path):
    """Only serializer-owned science/input/source/session identity changes a run fingerprint."""
    input_paths = write_input_fixture(tmp_path, name="workstation-session")
    copied_root = tmp_path / "cluster-session"
    shutil.copytree(input_paths["session_root"], copied_root, copy_function=shutil.copy2)
    copied_paths = {
        key: (
            copied_root
            if key == "session_root"
            else copied_root / path.relative_to(input_paths["session_root"])
        )
        for key, path in input_paths.items()
    }
    workstation_config = scientific_config_payload(
        make_config(input_paths, regularization_mode="fixed", output_root_name="analysis_runs")
    )
    cluster_config = scientific_config_payload(
        make_config(copied_paths, regularization_mode="fixed", output_root_name="different_output")
    )
    manifest = build_manifest(input_paths)
    copied_manifest = build_manifest(copied_paths)
    source_fingerprint = "scoped-source-fingerprint"
    baseline = results.build_run_fingerprint(
        analysis_version=ANALYSIS_VERSION,
        scientific_source_fingerprint=source_fingerprint,
        scientific_config=workstation_config,
        input_manifest=manifest,
        session_id="synthetic-01",
    )

    assert workstation_config == cluster_config
    reordered_config = {key: workstation_config[key] for key in reversed(workstation_config)}
    assert baseline == results.build_run_fingerprint(
        analysis_version=ANALYSIS_VERSION,
        scientific_source_fingerprint=source_fingerprint,
        scientific_config=reordered_config,
        input_manifest=copied_manifest,
        session_id="synthetic-01",
    )
    changed_config = json.loads(json.dumps(workstation_config))
    changed_config["alignment"] = "start_time"
    changed_manifest = json.loads(json.dumps(manifest))
    changed_manifest["files"]["neural_session"]["sha256"] = "0" * 64
    variations = (
        {
            "analysis_version": "different-analysis-version",
            "scientific_source_fingerprint": source_fingerprint,
            "scientific_config": workstation_config,
            "input_manifest": manifest,
            "session_id": "synthetic-01",
        },
        {
            "analysis_version": ANALYSIS_VERSION,
            "scientific_source_fingerprint": "different-source-fingerprint",
            "scientific_config": workstation_config,
            "input_manifest": manifest,
            "session_id": "synthetic-01",
        },
        {
            "analysis_version": ANALYSIS_VERSION,
            "scientific_source_fingerprint": source_fingerprint,
            "scientific_config": changed_config,
            "input_manifest": manifest,
            "session_id": "synthetic-01",
        },
        {
            "analysis_version": ANALYSIS_VERSION,
            "scientific_source_fingerprint": source_fingerprint,
            "scientific_config": workstation_config,
            "input_manifest": changed_manifest,
            "session_id": "synthetic-01",
        },
        {
            "analysis_version": ANALYSIS_VERSION,
            "scientific_source_fingerprint": source_fingerprint,
            "scientific_config": workstation_config,
            "input_manifest": manifest,
            "session_id": "other-session",
        },
    )
    for changed_values in variations:
        assert results.build_run_fingerprint(**changed_values) != baseline


def test_revision_six_solver_budget_cannot_reuse_revision_five_fingerprint(tmp_path):
    """The repaired logistic ceiling must allocate a new scientific run identity."""
    input_paths = write_input_fixture(tmp_path, name="versioned-session")
    current_config = scientific_config_payload(
        make_config(input_paths, regularization_mode="fixed")
    )
    revision_five_config = json.loads(json.dumps(current_config))
    revision_five_config["analysis_version"] = "task-variable-decoding-v1"
    revision_five_config["frozen_controls"]["estimators"]["LogisticRegression"][
        "max_iter"
    ] = 100
    common_identity = {
        "scientific_source_fingerprint": "scoped-source-fingerprint",
        "input_manifest": build_manifest(input_paths),
        "session_id": "synthetic-01",
    }

    assert current_config["analysis_version"] == "task-variable-decoding-v3"
    assert current_config["frozen_controls"]["estimators"]["LogisticRegression"][
        "max_iter"
    ] == 5000
    assert results.build_run_fingerprint(
        analysis_version=ANALYSIS_VERSION,
        scientific_config=current_config,
        **common_identity,
    ) != results.build_run_fingerprint(
        analysis_version="task-variable-decoding-v1",
        scientific_config=revision_five_config,
        **common_identity,
    )


def _directory_entries(directory: Path) -> set[str]:
    """Return the exact immediate visible directory members for atomicity assertions.

    Parameters
    ----------
    directory : pathlib.Path
        Existing or absent destination directory.

    Returns
    -------
    set[str]
        Immediate entry names; an absent directory has no visible artifacts.
    """
    if not directory.exists():
        return set()
    return {path.name for path in directory.iterdir()}


def _assert_safe_complete_npz(path: Path, expected_members: set[str]) -> None:
    """Assert a publication temporary is already a safe, complete NPZ artifact.

    Parameters
    ----------
    path : pathlib.Path
        Temporary sibling submitted to ``os.replace``.
    expected_members : set[str]
        Required NPZ member names, including ``meta`` where applicable.

    Returns
    -------
    None
        Raises an assertion error for incomplete, unsafe, or malformed output.
    """
    assert path.is_file()
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == expected_members


def test_target_checkpoint_round_trip_is_safe_atomic_and_requires_exact_fingerprint(
    tmp_path,
    monkeypatch,
):
    """A complete-target checkpoint is safe NPZ, exact-identity gated, and atomically published.

    The target arrays are one completed categorical target's fold-level
    primitives: ``scores`` has axes ``(region, representation, time, fold)``
    in dimensionless score units and ``status`` has the same axes.
    """
    checkpoint_path = tmp_path / "checkpoints" / "current_action.npz"
    target_arrays = {
        "scores": np.full((3, 2, 2, 3), 0.75, dtype=np.float64),
        "status": np.full((3, 2, 2, 3), "valid", dtype="U16"),
    }
    fingerprint = "full-run-fingerprint-a"
    original_replace = results.os.replace
    seen_sources: list[Path] = []
    failed_source_paths: list[Path] = []

    def inspect_then_fail(source, destination):
        """Inspect the complete temp file immediately before injected failure."""
        if Path(destination) == checkpoint_path:
            temporary = Path(source)
            assert temporary != checkpoint_path
            assert temporary.parent == checkpoint_path.parent
            _assert_safe_complete_npz(
                temporary,
                {"checkpoint_meta", "scores", "status"},
            )
            with np.load(temporary, allow_pickle=False) as archive:
                checkpoint_meta = json.loads(str(archive["checkpoint_meta"].item()))
            assert checkpoint_meta == {
                "full_run_fingerprint": fingerprint,
                "target_label": "current_action",
            }
            seen_sources.append(temporary)
            failed_source_paths.append(temporary)
            raise OSError("checkpoint publication failed")
        return original_replace(source, destination)

    monkeypatch.setattr(results.os, "replace", inspect_then_fail)
    with pytest.raises(OSError, match="checkpoint publication failed"):
        results.save_target_checkpoint(
            checkpoint_path,
            target_label="current_action",
            target_arrays=target_arrays,
            full_run_fingerprint=fingerprint,
        )
    assert len(seen_sources) == 1
    assert not checkpoint_path.exists()
    assert _directory_entries(checkpoint_path.parent) == set()

    seen_sources.clear()

    def inspect_then_publish(source, destination):
        """Capture a distinct, complete checkpoint source while allowing publication."""
        if Path(destination) == checkpoint_path:
            temporary = Path(source)
            assert temporary != checkpoint_path
            assert temporary.parent == checkpoint_path.parent
            _assert_safe_complete_npz(
                temporary,
                {"checkpoint_meta", "scores", "status"},
            )
            seen_sources.append(temporary)
        return original_replace(source, destination)

    monkeypatch.setattr(results.os, "replace", inspect_then_publish)
    results.save_target_checkpoint(
        checkpoint_path,
        target_label="current_action",
        target_arrays=target_arrays,
        full_run_fingerprint=fingerprint,
    )
    assert len(seen_sources) == 1
    assert seen_sources[0] != failed_source_paths[0]
    assert not seen_sources[0].exists()
    assert _directory_entries(checkpoint_path.parent) == {checkpoint_path.name}

    loaded = results.load_target_checkpoint(
        checkpoint_path,
        expected_full_run_fingerprint=fingerprint,
    )
    assert loaded["target_label"] == "current_action"
    assert loaded["full_run_fingerprint"] == fingerprint
    np.testing.assert_array_equal(loaded["target_arrays"]["scores"], target_arrays["scores"])
    np.testing.assert_array_equal(loaded["target_arrays"]["status"], target_arrays["status"])
    with pytest.raises(ValueError, match="fingerprint|identity|checkpoint"):
        results.load_target_checkpoint(
            checkpoint_path,
            expected_full_run_fingerprint="different-full-run-fingerprint",
        )


def test_result_save_uses_complete_unique_sibling_and_leaves_no_partial_publication(
    tmp_path,
    monkeypatch,
):
    """Final NPZ publication preserves prepared sidecars through retry without temp residue.

    The final NPZ holds all primitive arrays plus scalar JSON ``meta``. Config,
    manifest, and exact feature-copy sidecars are prepared before decoding and
    must survive an NPZ replace failure unchanged for a retry in this directory.
    """
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "atomic-run"
    save_arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    original_replace = results.os.replace
    seen_sources: list[Path] = []
    failed_source_paths: list[Path] = []

    def inspect_then_fail(source, destination):
        """Prove final replacement receives a distinct, safe complete sibling."""
        if Path(destination) == run_directory / _RESULT_FILE:
            temporary = Path(source)
            assert temporary != Path(destination)
            assert temporary.parent == run_directory
            _assert_safe_complete_npz(temporary, set(required_array_names()) | {"meta"})
            seen_sources.append(temporary)
            failed_source_paths.append(temporary)
            raise OSError("final publication failed")
        return original_replace(source, destination)

    monkeypatch.setattr(results.os, "replace", inspect_then_fail)
    with pytest.raises(OSError, match="final publication failed"):
        results.save_task_decoding_run(run_directory, **save_arguments)
    assert len(seen_sources) == 1
    assert not (run_directory / _RESULT_FILE).exists()
    sidecar_names = {_CONFIG_FILE, _FEATURE_COPY_FILE, _MANIFEST_FILE}
    assert _directory_entries(run_directory) == sidecar_names
    sidecar_bytes = {
        sidecar_name: (run_directory / sidecar_name).read_bytes()
        for sidecar_name in sidecar_names
    }

    seen_sources.clear()

    def inspect_then_publish(source, destination):
        """Capture the normal publication source before its successful replacement."""
        if Path(destination) == run_directory / _RESULT_FILE:
            temporary = Path(source)
            assert temporary != Path(destination)
            assert temporary.parent == run_directory
            _assert_safe_complete_npz(temporary, set(required_array_names()) | {"meta"})
            seen_sources.append(temporary)
        return original_replace(source, destination)

    monkeypatch.setattr(results.os, "replace", inspect_then_publish)
    results.save_task_decoding_run(run_directory, **save_arguments)
    assert len(seen_sources) == 1
    assert seen_sources[0] != failed_source_paths[0]
    assert not seen_sources[0].exists()
    assert _directory_entries(run_directory) == {
        *sidecar_names,
        _RESULT_FILE,
    }
    assert {
        sidecar_name: (run_directory / sidecar_name).read_bytes()
        for sidecar_name in sidecar_names
    } == sidecar_bytes
    loaded_fingerprint = results.load_task_decoding_run(run_directory)["run_fingerprint"]
    assert loaded_fingerprint == "run-fingerprint-test"


def test_result_publication_cleanup_failure_keeps_temp_visible_and_chains_errors(
    tmp_path,
    monkeypatch,
):
    """A failed NPZ replace plus failed cleanup retains sidecars and a diagnosable temp.

    This uses a fresh destination, never overwrites an immutable final result,
    and requires the publication failure to remain available in exception
    context when cleanup itself also fails.
    """
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "cleanup-failure-run"
    save_arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    original_replace = results.os.replace
    original_unlink = results.os.unlink
    temporary_paths: list[Path] = []

    def fail_replace(source, destination):
        """Record the actual final temp source and inject the primary failure."""
        if Path(destination) == run_directory / _RESULT_FILE:
            temporary = Path(source)
            _assert_safe_complete_npz(temporary, set(required_array_names()) | {"meta"})
            temporary_paths.append(temporary)
            raise OSError("primary publication failure")
        return original_replace(source, destination)

    def fail_temp_cleanup(path, *args, **kwargs):
        """Prevent deletion of only the captured temp, leaving visible forensic evidence."""
        if temporary_paths and Path(path) == temporary_paths[0]:
            raise OSError("cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(results.os, "replace", fail_replace)
    monkeypatch.setattr(results.os, "unlink", fail_temp_cleanup)
    with pytest.raises(Exception) as caught:
        results.save_task_decoding_run(run_directory, **save_arguments)
    assert len(temporary_paths) == 1
    assert not (run_directory / _RESULT_FILE).exists()
    assert temporary_paths[0].exists()
    assert _directory_entries(run_directory) == {
        _CONFIG_FILE,
        _FEATURE_COPY_FILE,
        _MANIFEST_FILE,
        temporary_paths[0].name,
    }
    messages: list[str] = []
    exception: BaseException | None = caught.value
    while exception is not None:
        messages.append(str(exception))
        exception = exception.__context__ or exception.__cause__
    assert any("primary publication failure" in message for message in messages)
    assert any("cleanup failure" in message for message in messages)


def test_run_fingerprint_round_trips_from_meta_and_rejects_a_mismatched_load(tmp_path):
    """An arbitrary full run fingerprint is persisted in meta provenance and identity-gated.

    The fixed synthetic arrays retain their documented axes; the fingerprint is
    an opaque SHA-256-like scientific identity rather than execution metadata.
    """
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "fingerprint-run"
    fingerprint = "f" * 64
    save_run_fixture(
        run_directory,
        input_paths,
        regularization_mode="fixed",
        run_fingerprint=fingerprint,
    )

    loaded = results.load_task_decoding_run(
        run_directory,
        expected_run_fingerprint=fingerprint,
    )

    assert loaded["run_fingerprint"] == fingerprint
    assert loaded["meta"]["provenance"]["run_fingerprint"] == fingerprint
    with pytest.raises(ValueError, match="fingerprint|identity|run"):
        results.load_task_decoding_run(
            run_directory,
            expected_run_fingerprint="other" * 16,
        )


def test_save_rejects_run_fingerprint_disagreement_with_predeclared_meta(tmp_path):
    """Save rejects an argument whose identity disagrees with immutable provenance."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode="fixed",
        run_fingerprint="declared-fingerprint",
    )
    meta = json.loads(json.dumps(arguments["meta"]))
    assert meta["provenance"]["run_fingerprint"] == "declared-fingerprint"
    run_directory = tmp_path / "fingerprint-disagreement"

    with pytest.raises(ValueError, match="fingerprint|provenance|identity"):
        results.save_task_decoding_run(
            run_directory,
            arrays=arguments["arrays"],
            meta=meta,
            input_manifest=arguments["input_manifest"],
            scientific_config=arguments["scientific_config"],
            feature_parameter_source=arguments["feature_parameter_source"],
            run_fingerprint="different-fingerprint",
        )
    assert not (run_directory / _RESULT_FILE).exists()


def test_final_result_and_checkpoint_destinations_are_immutable_after_publication(tmp_path):
    """Second result/checkpoint saves fail without changing already-published safe bytes."""
    input_paths = write_input_fixture(tmp_path)
    run_directory = tmp_path / "immutable-run"
    save_arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    results.save_task_decoding_run(run_directory, **save_arguments)
    result_bytes = (run_directory / _RESULT_FILE).read_bytes()

    with pytest.raises(FileExistsError, match="immutable|exists|destination"):
        results.save_task_decoding_run(run_directory, **save_arguments)
    assert (run_directory / _RESULT_FILE).read_bytes() == result_bytes

    checkpoint_path = tmp_path / "checkpoints" / "current_action.npz"
    checkpoint_arrays = {"scores": np.array([0.75], dtype=np.float64)}
    results.save_target_checkpoint(
        checkpoint_path,
        target_label="current_action",
        target_arrays=checkpoint_arrays,
        full_run_fingerprint="checkpoint-fingerprint",
    )
    checkpoint_bytes = checkpoint_path.read_bytes()
    with pytest.raises(FileExistsError, match="immutable|exists|destination"):
        results.save_target_checkpoint(
            checkpoint_path,
            target_label="current_action",
            target_arrays=checkpoint_arrays,
            full_run_fingerprint="checkpoint-fingerprint",
        )
    assert checkpoint_path.read_bytes() == checkpoint_bytes


def test_small_binary_npy_manifest_identity_does_not_open_or_hash_content(tmp_path, monkeypatch):
    """Small binary NumPy inputs use portable metadata rather than content hashing.

    The fixture has a tiny ``.npy`` path to prove the classification depends on
    binary type, not only the 64-MiB threshold used for structured sources.
    """
    input_paths = write_input_fixture(tmp_path)
    binary_path = input_paths["session_root"] / "processed" / "small_binary.npy"
    binary_path.write_bytes(b"\x93NUMPY synthetic bytes")
    original_open = Path.open

    def forbid_binary_open(path: Path, *args, **kwargs):
        """Forbid any content read of the small binary input identity."""
        mode = args[0] if args else kwargs.get("mode", "r")
        if path == binary_path and "r" in mode:
            raise AssertionError("Small binary content must not be opened for manifest identity.")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", forbid_binary_open)
    manifest = results.build_input_manifest(
        session_root=input_paths["session_root"],
        files={"small_binary": binary_path},
        file_lists={},
    )
    identity = manifest["files"]["small_binary"]
    assert identity["relative_path"] == "processed/small_binary.npy"
    assert identity["size_bytes"] == binary_path.stat().st_size
    assert identity["mtime_seconds"] == int(binary_path.stat().st_mtime)
    assert "sha256" not in identity


def make_dynamic_tuned_payload(
    input_paths: Mapping[str, Path],
    *,
    run_fingerprint: str = "dynamic-run-fingerprint",
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    """Build a one-target tuned payload whose dimensions are not fixture constants.

    Parameters
    ----------
    input_paths : mapping[str, pathlib.Path]
        Contained source inputs used to create a real scientific config.
    run_fingerprint : str, default="dynamic-run-fingerprint"
        Opaque full scientific identity stored in provenance before publication.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object], dict[str, object]]
        Complete tuned arrays, matching JSON meta, and serializer-owned config.
        There are 19 full-table rows, 14 common rows, five unequal block-safe
        outer tests, eight 500-ms time bins, and five padded feature slots.
    """
    config = replace(
        make_config(input_paths, regularization_mode="tuned"),
        target_names=("relative_doubt",),
        outer_fold_count=5,
    )
    scientific_config = scientific_config_payload(config)
    target_n, region_n, representation_n = 1, 3, 2
    metric_n, time_n, fold_n, feature_n = 3, 8, 5, 5
    full_n, common_n = 19, 14
    fit_shape = (target_n, region_n, representation_n, time_n, fold_n)
    score_shape = (target_n, region_n, representation_n, metric_n, time_n, fold_n)
    parameter_shape = (target_n, fold_n, region_n, representation_n, time_n)
    trial_rows = np.array(
        [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18],
        dtype=np.int64,
    )
    block_values: tuple[object, ...] = (
        "alpha",
        "alpha",
        "not-common-2",
        "beta",
        "beta",
        "not-common-5",
        "gamma",
        "gamma",
        "not-common-8",
        "delta",
        "delta",
        "not-common-11",
        "delta",
        "epsilon",
        "not-common-14",
        "epsilon",
        "epsilon",
        "target-ineligible-zero",
        "target-ineligible-one",
    )
    block_bytes = tuple(
        json.dumps(value, separators=(",", ":")).encode("utf-8")
        for value in block_values
    )
    arrays: dict[str, np.ndarray] = {
        "target_labels": np.array(["relative_doubt"], dtype="U32"),
        "target_families": np.array(["numerical"], dtype="U16"),
        "target_status": np.array(["available"], dtype="U16"),
        "target_unavailable_reasons": np.array([""], dtype="U128"),
        "target_source_labels": np.array(["relative_doubt_index"], dtype="U32"),
        "target_positive_classes": np.array([-1], dtype=np.int64),
        "target_label_mappings_json": np.array(['{"native_target":true}'], dtype="U48"),
        "region_labels": np.array(_REGIONS, dtype="U16"),
        "representation_labels": np.array(_REPRESENTATIONS, dtype="U16"),
        "metric_labels": np.array(_METRICS, dtype="U32"),
        "time_bin_edges_s": np.linspace(-2.0, 2.0, time_n + 1, dtype=np.float64),
        "time_bin_centers_s": np.linspace(-1.75, 1.75, time_n, dtype=np.float64),
        "fold_labels": np.arange(fold_n, dtype=np.int64),
        "fold_scores": np.full(score_shape, np.nan, dtype=np.float64),
        "fit_status": np.full(fit_shape, "valid", dtype="U16"),
        "fit_reason_codes": np.full(fit_shape, "", dtype="U32"),
        "requested_feature_counts": np.empty(fit_shape, dtype=np.int64),
        "effective_feature_counts": np.empty(fit_shape, dtype=np.int64),
        "train_counts": np.empty(fit_shape, dtype=np.int64),
        "test_counts": np.empty(fit_shape, dtype=np.int64),
        "train_class_counts": np.full((*fit_shape, 2), -1, dtype=np.int64),
        "test_class_counts": np.full((*fit_shape, 2), -1, dtype=np.int64),
        "full_table_row_positions": np.arange(full_n, dtype=np.int64),
        "full_table_trial_ids": np.arange(full_n, dtype=np.int64),
        "full_table_block_ids_utf8": np.frombuffer(
            b"".join(block_bytes), dtype=np.uint8
        ).copy(),
        "full_table_block_id_offsets": np.r_[
            np.array([0], dtype=np.int64),
            np.cumsum([len(value) for value in block_bytes], dtype=np.int64),
        ],
        "encoded_target_values": np.full((target_n, full_n), np.nan, dtype=np.float64),
        "eligibility_masks": np.zeros((target_n, full_n), dtype=np.bool_),
        "eligibility_reason_codes": np.full(
            (target_n, full_n), "not_common", dtype="U32"
        ),
        "eligibility_counts": np.array([12], dtype=np.int64),
        "trial_row_indices": trial_rows,
        "outer_fold_ids": np.array(
            [[0, 0, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, -1, -1]],
            dtype=np.int64,
        ),
        "unit_selection_rules_json": np.array(
            ['{"PFC":"good_inside_brain"}', '{"HPC":"good_inside_brain"}'],
            dtype="U48",
        ),
        "coefficient_values": np.full((*fit_shape, feature_n), np.nan, dtype=np.float64),
        "coefficient_feature_ids": np.full(
            (region_n, representation_n, feature_n), "", dtype="U32"
        ),
        "coefficient_feature_regions": np.full(
            (region_n, representation_n, feature_n), "", dtype="U16"
        ),
        "coefficient_feature_statuses": np.full(
            (region_n, representation_n, feature_n), "padding", dtype="U16"
        ),
        "coefficient_active_masks": np.zeros(
            (region_n, representation_n, feature_n), dtype=np.bool_
        ),
        "fitted_intercepts": np.zeros(fit_shape, dtype=np.float64),
        "fixed_parameters_json": np.full(parameter_shape, "", dtype="U48"),
        "candidate_parameter_json": np.empty((target_n, 15), dtype="U48"),
        "selected_parameters_json": np.empty(parameter_shape, dtype="U48"),
        "inner_selection_fold_ids": np.full(
            (target_n, fold_n, common_n), -1, dtype=np.int64
        ),
        "candidate_inner_scores": np.empty(
            (target_n, fold_n, region_n, representation_n, time_n, 15, 3),
            dtype=np.float64,
        ),
        "candidate_inner_statuses": np.full(
            (target_n, fold_n, region_n, representation_n, time_n, 15, 3),
            "valid",
            dtype="U16",
        ),
        "candidate_inner_reasons": np.full(
            (target_n, fold_n, region_n, representation_n, time_n, 15, 3),
            "",
            dtype="U48",
        ),
        "selected_candidate_indices": np.empty(parameter_shape, dtype=np.int64),
        "stage_timing_labels": np.array(["targets", "activity", "modeling"], dtype="U16"),
        "stage_timing_seconds": np.array([0.01, 0.02, 0.03], dtype=np.float64),
        "total_timing_seconds": np.array(0.06, dtype=np.float64),
    }
    eligible_full_rows = trial_rows[:12]
    arrays["encoded_target_values"][0, eligible_full_rows] = np.linspace(
        -1.0, 1.0, eligible_full_rows.size
    )
    arrays["eligibility_masks"][0, eligible_full_rows] = True
    arrays["eligibility_reason_codes"][0, eligible_full_rows] = ""
    arrays["eligibility_reason_codes"][0, trial_rows[12:]] = "target_ineligible"
    test_sizes = np.array([2, 2, 2, 3, 3], dtype=np.int64)
    for fold_index, test_size in enumerate(test_sizes):
        arrays["train_counts"][..., fold_index] = 12 - test_size
        arrays["test_counts"][..., fold_index] = test_size
    feature_map = {
        ("PFC", "pca"): (("PFC:PC1", "PFC:PC2"), ("PFC", "PFC"), 2),
        ("PFC", "units"): (("probe-pfc:1", "probe-pfc:2"), ("PFC", "PFC"), 2),
        ("HPC", "pca"): (("HPC:PC1", "HPC:PC2"), ("HPC", "HPC"), 2),
        ("HPC", "units"): (("probe-hpc:1", "probe-hpc:2"), ("HPC", "HPC"), 2),
        ("PFC+HPC", "pca"): (
            ("PFC:PC1", "PFC:PC2", "HPC:PC1", "HPC:PC2"),
            ("PFC", "PFC", "HPC", "HPC"),
            4,
        ),
        ("PFC+HPC", "units"): (
            ("probe-pfc:1", "probe-pfc:2", "probe-hpc:1", "probe-hpc:2"),
            ("PFC", "PFC", "HPC", "HPC"),
            4,
        ),
    }
    for region_index, region in enumerate(_REGIONS):
        for representation_index, representation in enumerate(_REPRESENTATIONS):
            ids, regions, requested = feature_map[(region, representation)]
            width = len(ids)
            arrays["coefficient_feature_ids"][region_index, representation_index, :width] = ids
            arrays["coefficient_feature_regions"][
                region_index, representation_index, :width
            ] = regions
            arrays["coefficient_feature_statuses"][
                region_index, representation_index, :width
            ] = "available"
            arrays["coefficient_active_masks"][region_index, representation_index, :width] = True
            arrays["coefficient_values"][:, region_index, representation_index, :, :, :width] = 0.5
            arrays["requested_feature_counts"][:, region_index, representation_index] = requested
            arrays["effective_feature_counts"][:, region_index, representation_index] = width
    arrays["fold_scores"][0, :, :, 2, :, :] = 0.25
    arrays["fixed_parameters_json"][0, ...] = json.dumps(
        {"alpha": 1.0, "l1_ratio": 0.5},
        sort_keys=True,
    )
    candidates = declared_candidates(scientific_config, "numerical")
    arrays["candidate_parameter_json"][0] = [
        json.dumps(candidate, sort_keys=True) for candidate in candidates
    ]
    common_blocks = tuple(block_bytes[row] for row in trial_rows)
    for outer_fold in range(fold_n):
        block_to_inner: dict[bytes, int] = {}
        for common_row, block in enumerate(common_blocks):
            if arrays["outer_fold_ids"][0, common_row] != outer_fold and common_row < 12:
                block_to_inner.setdefault(block, len(block_to_inner) % 3)
                arrays["inner_selection_fold_ids"][0, outer_fold, common_row] = (
                    block_to_inner[block]
                )
        assert set(block_to_inner.values()) == {0, 1, 2}
    for index in np.ndindex(parameter_shape):
        _, outer_fold, region, representation, time = index
        selected = (outer_fold + region + representation + time) % 14
        arrays["selected_candidate_indices"][index] = selected
        arrays["selected_parameters_json"][index] = arrays["candidate_parameter_json"][0, selected]
        for candidate in range(15):
            arrays["candidate_inner_scores"][index + (candidate, slice(None))] = (
                0.10 + candidate / 100.0
            )
        arrays["candidate_inner_scores"][index + (selected, slice(None))] = 0.95
        invalid_index = index + (14, 2)
        arrays["candidate_inner_scores"][invalid_index] = np.nan
        arrays["candidate_inner_statuses"][invalid_index] = "invalid"
        arrays["candidate_inner_reasons"][invalid_index] = "nonfinite_inner_score"
    meta = make_meta(
        arrays,
        scientific_config,
        regularization_mode="tuned",
        run_fingerprint=run_fingerprint,
    )
    meta["parameters"] = {
        "regularization_mode": "tuned",
        "outer_fold_count": fold_n,
        "inner_fold_count": 3,
        "bin_width_ms": 500,
        "candidate_selection_metrics": {"relative_doubt": "r2"},
    }
    meta["units"] = {
        "encoded_target_values": {"relative_doubt": "unitless"},
        "fold_scores": {"relative_doubt": {"r2": "coefficient_of_determination"}},
        "candidate_inner_scores": {
            "relative_doubt": {"r2": "coefficient_of_determination"}
        },
        "coefficient_values": {"relative_doubt": "unitless per pooled training standard deviation"},
        "fitted_intercepts": {"relative_doubt": "unitless"},
        "time_bin_edges_s": "s",
        "time_bin_centers_s": "s",
        "stage_timing_seconds": "s",
        "total_timing_seconds": "s",
    }
    return arrays, meta, scientific_config


def rebuild_dynamic_inner_assignments(arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    """Build grouped inner IDs after an intentional dynamic-fixture outer mutation.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Dynamic tuned result members with global common-to-full row positions.

    Returns
    -------
    numpy.ndarray
        Int64 ``(target, outer_fold, common_row)`` assignment. Outer-test and
        target-ineligible common rows are -1; every retained group is assigned
        one inner fold without splitting that block.
    """
    payload = arrays["full_table_block_ids_utf8"].tobytes()
    offsets = arrays["full_table_block_id_offsets"]
    blocks = tuple(payload[start:stop] for start, stop in zip(offsets[:-1], offsets[1:]))
    outer = arrays["outer_fold_ids"]
    assignments = np.full(
        (outer.shape[0], arrays["fold_labels"].size, outer.shape[1]),
        -1,
        dtype=np.int64,
    )
    for target, outer_fold in np.ndindex(outer.shape[0], arrays["fold_labels"].size):
        group_ids: dict[bytes, int] = {}
        training_rows = np.flatnonzero((outer[target] >= 0) & (outer[target] != outer_fold))
        for common_row in training_rows:
            full_row = int(arrays["trial_row_indices"][common_row])
            group_ids.setdefault(blocks[full_row], len(group_ids) % 3)
            assignments[target, outer_fold, common_row] = group_ids[blocks[full_row]]
    return assignments


def synchronize_selected_dynamic_parameters(arrays: Mapping[str, np.ndarray]) -> None:
    """Match each nonnegative selected index to its target-family candidate JSON.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Mutable dynamic tuned payload with target-first candidate and selection
        arrays. Candidate text is canonical JSON and selection uses the
        ``(target, outer_fold, region, representation, time)`` order.

    Returns
    -------
    None
        Updates only ``selected_parameters_json`` in place.
    """
    selected = arrays["selected_candidate_indices"]
    parameters = arrays["selected_parameters_json"]
    candidates = arrays["candidate_parameter_json"]
    for index in np.ndindex(selected.shape):
        choice = int(selected[index])
        parameters[index] = "" if choice < 0 else candidates[index[0], choice]


def make_inner_plan_unavailable_tuned_payload(
    input_paths: Mapping[str, Path],
    *,
    run_fingerprint: str = "inner-plan-unavailable",
) -> tuple[dict[str, np.ndarray], dict[str, object], dict[str, object]]:
    """Build a valid outer-CV numerical payload whose inner three-fold plan is impossible.

    Parameters
    ----------
    input_paths : mapping[str, pathlib.Path]
        Contained source inputs used to create a real scientific config.
    run_fingerprint : str, default="inner-plan-unavailable"
        Opaque full scientific identity stored in provenance before publication.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object], dict[str, object]]
        Complete tuned arrays, matching JSON meta, and serializer-owned config.
        Three two-row behavioral blocks form valid outer folds, but every
        outer-training universe contains exactly two blocks, fewer than the
        required three grouped inner folds.
    """
    config = replace(
        make_config(input_paths, regularization_mode="tuned"),
        target_names=("relative_doubt",),
        outer_fold_count=3,
    )
    scientific_config = scientific_config_payload(config)
    base = make_result_arrays(scientific_config, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in base.items()}
    target_members = (
        "target_labels",
        "target_families",
        "target_status",
        "target_unavailable_reasons",
        "target_source_labels",
        "target_positive_classes",
        "target_label_mappings_json",
        "fold_scores",
        "fit_status",
        "fit_reason_codes",
        "requested_feature_counts",
        "effective_feature_counts",
        "train_counts",
        "test_counts",
        "train_class_counts",
        "test_class_counts",
        "encoded_target_values",
        "eligibility_masks",
        "eligibility_reason_codes",
        "eligibility_counts",
        "outer_fold_ids",
        "coefficient_values",
        "fitted_intercepts",
        "fixed_parameters_json",
        "candidate_parameter_json",
        "selected_parameters_json",
        "inner_selection_fold_ids",
        "candidate_inner_scores",
        "candidate_inner_statuses",
        "candidate_inner_reasons",
        "selected_candidate_indices",
    )
    for name in target_members:
        arrays[name] = arrays[name][1:2].copy()

    full_rows = 7
    common_rows = 6
    block_json = _BLOCK_ID_JSON_BYTES[:full_rows]
    arrays["full_table_row_positions"] = np.arange(full_rows, dtype=np.int64)
    arrays["full_table_trial_ids"] = np.arange(full_rows, dtype=np.int64)
    arrays["full_table_block_ids_utf8"] = np.frombuffer(
        b"".join(block_json),
        dtype=np.uint8,
    ).copy()
    arrays["full_table_block_id_offsets"] = np.r_[
        np.array([0], dtype=np.int64),
        np.cumsum([len(value) for value in block_json], dtype=np.int64),
    ]
    arrays["trial_row_indices"] = np.arange(common_rows, dtype=np.int64)
    arrays["encoded_target_values"] = np.full((1, full_rows), np.nan, dtype=np.float64)
    arrays["encoded_target_values"][0, :common_rows] = np.linspace(
        -1.0,
        1.0,
        common_rows,
        dtype=np.float64,
    )
    arrays["eligibility_masks"] = np.zeros((1, full_rows), dtype=np.bool_)
    arrays["eligibility_masks"][0, :common_rows] = True
    arrays["eligibility_reason_codes"] = np.full((1, full_rows), "not_common", dtype="U32")
    arrays["eligibility_reason_codes"][0, :common_rows] = ""
    arrays["eligibility_counts"] = np.array([common_rows], dtype=np.int64)
    arrays["outer_fold_ids"] = np.array([[0, 0, 1, 1, 2, 2]], dtype=np.int64)
    arrays["inner_selection_fold_ids"] = np.full(
        (1, 3, common_rows),
        -1,
        dtype=np.int64,
    )

    unavailable_reason = "inner_plan_unavailable"
    arrays["fit_status"].fill("unavailable")
    arrays["fit_reason_codes"].fill(unavailable_reason)
    arrays["train_counts"].fill(4)
    arrays["test_counts"].fill(2)
    arrays["effective_feature_counts"].fill(0)
    arrays["fold_scores"].fill(np.nan)
    arrays["coefficient_values"].fill(np.nan)
    arrays["fitted_intercepts"].fill(np.nan)
    arrays["candidate_inner_scores"].fill(np.nan)
    arrays["candidate_inner_statuses"].fill("invalid")
    arrays["candidate_inner_reasons"].fill(unavailable_reason)
    arrays["selected_candidate_indices"].fill(-1)
    arrays["selected_parameters_json"].fill("")

    meta = make_meta(
        arrays,
        scientific_config,
        regularization_mode="tuned",
        run_fingerprint=run_fingerprint,
    )
    meta["parameters"] = {
        "regularization_mode": "tuned",
        "outer_fold_count": 3,
        "inner_fold_count": 3,
        "bin_width_ms": 500,
        "candidate_selection_metrics": {"relative_doubt": "r2"},
    }
    meta["units"] = {
        "encoded_target_values": {"relative_doubt": "unitless"},
        "fold_scores": {"relative_doubt": {"r2": "coefficient_of_determination"}},
        "candidate_inner_scores": {
            "relative_doubt": {"r2": "coefficient_of_determination"}
        },
        "coefficient_values": {
            "relative_doubt": "unitless per pooled training standard deviation"
        },
        "fitted_intercepts": {"relative_doubt": "unitless"},
        "time_bin_edges_s": "s",
        "time_bin_centers_s": "s",
        "stage_timing_seconds": "s",
        "total_timing_seconds": "s",
    }
    return arrays, meta, scientific_config


def test_dynamic_numerical_tuned_round_trip_uses_saved_dimensions_and_global_row_map(tmp_path):
    """A valid tuned run is not limited to the small two-target fixture dimensions.

    One numerical target has 19 full rows, 14 gapped common tensor rows, five
    unequal block-safe outer folds, eight 500-ms bins, and five padded feature
    slots. ``trial_row_indices`` are global full-table positions, not a
    positional assumption about the common neural tensor.
    """
    input_paths = write_input_fixture(tmp_path)
    arrays, meta, scientific_config = make_dynamic_tuned_payload(input_paths)
    run_directory = tmp_path / "dynamic-tuned-run"
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        meta=meta,
        input_manifest=build_manifest(input_paths),
        scientific_config=scientific_config,
        feature_parameter_source=input_paths["feature_parameters"],
        run_fingerprint="dynamic-run-fingerprint",
    )

    loaded = results.load_task_decoding_run(
        run_directory,
        expected_run_fingerprint="dynamic-run-fingerprint",
    )
    saved = loaded["arrays"]
    assert saved["target_labels"].tolist() == ["relative_doubt"]
    assert saved["fold_labels"].tolist() == [0, 1, 2, 3, 4]
    assert saved["time_bin_centers_s"].shape == (8,)
    assert saved["time_bin_edges_s"].tolist() == [
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]
    assert scientific_config["bin_width_ms"] == 500
    assert saved["coefficient_values"].shape[-1] == 5
    assert saved["trial_row_indices"].tolist() == [
        0,
        1,
        3,
        4,
        6,
        7,
        9,
        10,
        12,
        13,
        15,
        16,
        17,
        18,
    ]
    assert saved["full_table_trial_ids"].tolist() == list(range(19))
    assert [np.count_nonzero(saved["outer_fold_ids"][0] == fold) for fold in range(5)] == [
        2,
        2,
        2,
        3,
        3,
    ]
    assert np.isnan(saved["fold_scores"][0, :, :, :2]).all()
    assert np.all(np.isfinite(saved["fold_scores"][0, :, :, 2]))
    assert np.all(
        saved["fixed_parameters_json"]
        == json.dumps({"alpha": 1.0, "l1_ratio": 0.5}, sort_keys=True)
    )
    offsets = saved["full_table_block_id_offsets"]
    block_bytes = saved["full_table_block_ids_utf8"]
    assert block_bytes.dtype == np.dtype(np.uint8)
    assert block_bytes.ndim == 1
    assert offsets.shape == (saved["full_table_trial_ids"].size + 1,)
    payload = block_bytes.tobytes()
    blocks = [
        json.loads(payload[start:stop].decode("utf-8"))
        for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    assert blocks[0:2] == ["alpha", "alpha"]
    assert blocks[17:] == ["target-ineligible-zero", "target-ineligible-one"]
    assert payload[offsets[0] : offsets[1]] == json.dumps(
        "alpha", separators=(",", ":")
    ).encode("utf-8")
    outer = saved["outer_fold_ids"][0]
    inner = saved["inner_selection_fold_ids"][0]
    for outer_fold in range(5):
        outer_test = outer == outer_fold
        assert np.all(inner[outer_fold, outer_test] == -1)
        assert np.all(inner[outer_fold, outer < 0] == -1)
        assert set(inner[outer_fold, inner[outer_fold] >= 0]) == {0, 1, 2}
        inner_counts = np.bincount(inner[outer_fold, inner[outer_fold] >= 0], minlength=3)
        assert np.all(inner_counts >= 2)
        for common_row, full_row in enumerate(saved["trial_row_indices"]):
            grouped_rows = [
                row
                for row, row_full in enumerate(saved["trial_row_indices"])
                if blocks[row_full] == blocks[full_row] and inner[outer_fold, row] >= 0
            ]
            if grouped_rows:
                assert len({int(inner[outer_fold, row]) for row in grouped_rows}) == 1
    common_rows = saved["trial_row_indices"]
    common_eligibility = saved["eligibility_masks"][0, common_rows]
    assert np.array_equal(outer >= 0, common_eligibility)
    for common_row in np.flatnonzero(outer >= 0):
        full_row = common_rows[common_row]
        same_block = [
            row
            for row, row_full in enumerate(common_rows)
            if blocks[row_full] == blocks[full_row] and outer[row] >= 0
        ]
        assert len({int(outer[row]) for row in same_block}) == 1
    expected_candidates = [
        json.dumps(candidate, sort_keys=True)
        for candidate in declared_candidates(scientific_config, "numerical")
    ]
    assert saved["candidate_parameter_json"][0].tolist() == expected_candidates
    candidate_scores = saved["candidate_inner_scores"]
    candidate_statuses = saved["candidate_inner_statuses"]
    assert candidate_scores.shape == (1, 5, 3, 2, 8, 15, 3)
    assert set(np.unique(candidate_statuses)) == {"invalid", "valid"}
    for fit_index in np.ndindex(saved["selected_candidate_indices"].shape):
        valid_candidates = np.all(candidate_statuses[fit_index] == "valid", axis=1)
        means = np.mean(candidate_scores[fit_index], axis=1)
        means[~valid_candidates] = -np.inf
        selected = int(saved["selected_candidate_indices"][fit_index])
        assert selected == int(np.argmax(means))
        assert saved["selected_parameters_json"][fit_index] == saved[
            "candidate_parameter_json"
        ][0, selected]
    assert loaded["meta"]["provenance"]["run_fingerprint"] == "dynamic-run-fingerprint"


def test_save_rejects_outer_group_leakage_using_gapped_common_to_full_mapping(tmp_path):
    """Outer leakage is checked through full-table block IDs, not common-row positions.

    The two ``"alpha"`` common rows map to full rows 0 and 1. Swapping one
    with the single ``"beta"`` fold preserves every fold cardinality, so the
    only violated scientific invariant is the split behavioral block.
    """
    input_paths = write_input_fixture(tmp_path)
    arrays, meta, scientific_config = make_dynamic_tuned_payload(
        input_paths,
        run_fingerprint="gapped-leakage-fingerprint",
    )
    arrays["outer_fold_ids"] = arrays["outer_fold_ids"].copy()
    arrays["outer_fold_ids"][0, [1, 2]] = arrays["outer_fold_ids"][0, [2, 1]]
    arrays["inner_selection_fold_ids"] = rebuild_dynamic_inner_assignments(arrays)
    run_directory = tmp_path / "gapped-leakage-run"

    with pytest.raises(ValueError, match="[Oo]uter.*[Bb]lock|[Bb]lock.*[Ll]eakage"):
        results.save_task_decoding_run(
            run_directory,
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint="gapped-leakage-fingerprint",
        )
    assert not (run_directory / _RESULT_FILE).exists()


def test_save_rejects_inner_group_leakage_using_gapped_common_to_full_mapping(tmp_path):
    """Inner validation blocks cannot split after mapping common rows to full-table groups."""
    input_paths = write_input_fixture(tmp_path)
    arrays, meta, scientific_config = make_dynamic_tuned_payload(
        input_paths,
        run_fingerprint="gapped-inner-leakage-fingerprint",
    )
    arrays["inner_selection_fold_ids"] = arrays["inner_selection_fold_ids"].copy()
    arrays["inner_selection_fold_ids"][0, 0, 3] = 1
    run_directory = tmp_path / "gapped-inner-leakage-run"

    with pytest.raises(ValueError, match="[Ii]nner.*[Bb]lock|[Bb]lock.*[Ll]eakage"):
        results.save_task_decoding_run(
            run_directory,
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint="gapped-inner-leakage-fingerprint",
        )
    assert not (run_directory / _RESULT_FILE).exists()


def test_tuned_audit_requires_declared_grid_and_never_selects_an_invalid_candidate(tmp_path):
    """Tuned audit JSON follows the frozen family grid and excludes invalid candidates.

    A candidate with one invalid inner fold has a NaN score, ``"invalid"``
    status, and a nonempty reason. It cannot be selected even when its
    parameter JSON matches the selected-parameter cell exactly.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode="tuned",
        run_fingerprint="invalid-candidate-selection",
    )
    arrays = arguments["arrays"]
    assert isinstance(arrays, dict)
    config = arguments["scientific_config"]
    assert isinstance(config, dict)
    for target, family in enumerate(("categorical", "numerical")):
        expected = [
            json.dumps(item, sort_keys=True)
            for item in declared_candidates(config, family)
        ]
        assert arrays["candidate_parameter_json"][target].tolist() == expected
    statuses = arrays["candidate_inner_statuses"]
    assert set(np.unique(statuses)) == {"invalid", "valid"}
    invalid_scores = arrays["candidate_inner_scores"][statuses == "invalid"]
    assert np.isnan(invalid_scores).all()
    assert np.all(arrays["candidate_inner_reasons"][statuses == "invalid"] != "")

    altered = {name: value.copy() for name, value in arrays.items()}
    altered["selected_candidate_indices"][0, 0, 0, 0, 0] = 14
    altered["selected_parameters_json"][0, 0, 0, 0, 0] = altered[
        "candidate_parameter_json"
    ][0, 14]
    run_directory = tmp_path / "invalid-selected-candidate"
    with pytest.raises(ValueError, match="candidate|invalid|selected"):
        results.save_task_decoding_run(
            run_directory,
            arrays=altered,
            meta=arguments["meta"],
            input_manifest=arguments["input_manifest"],
            scientific_config=config,
            feature_parameter_source=arguments["feature_parameter_source"],
            run_fingerprint="invalid-candidate-selection",
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("reordered_grid", "candidate|grid|order"),
        ("changed_grid", "candidate|grid|parameter"),
        ("wrong_family_grid", "candidate|family|parameter"),
        ("noncanonical_grid_json", "candidate|canonical|JSON"),
        ("nonbest_selection", "selected|candidate|best|mean"),
        ("finite_invalid_score", "candidate|invalid|nonfinite"),
        ("unknown_status", "candidate|status"),
        ("valid_nonempty_reason", "candidate|reason|valid"),
        ("missing_inner_training_assignment", "inner|training|assignment"),
        ("minus_one_selection_with_valid_candidate", "selected|candidate|valid"),
    ),
)
def test_save_rejects_causal_tuned_candidate_audit_violations(
    kind,
    failure_pattern,
    tmp_path,
):
    """Each malformed tuned audit state fails before publication for its own reason."""
    input_paths = write_input_fixture(tmp_path)
    fingerprint = f"candidate-audit-{kind}"
    arrays, meta, scientific_config = make_dynamic_tuned_payload(
        input_paths,
        run_fingerprint=fingerprint,
    )
    arrays = {name: value.copy() for name, value in arrays.items()}
    selection_index = (0, 0, 0, 0, 0)
    candidate_index = selection_index + (slice(None), slice(None))
    if kind == "reordered_grid":
        arrays["candidate_parameter_json"][0, [0, 1]] = arrays["candidate_parameter_json"][
            0, [1, 0]
        ]
        synchronize_selected_dynamic_parameters(arrays)
    elif kind == "changed_grid":
        arrays["candidate_parameter_json"][0, 0] = json.dumps(
            {"alpha": 0.123, "l1_ratio": 0.1},
            sort_keys=True,
        )
        synchronize_selected_dynamic_parameters(arrays)
    elif kind == "wrong_family_grid":
        arrays["candidate_parameter_json"][0, 0] = json.dumps(
            {"C": 0.01, "l1_ratio": 0.1},
            sort_keys=True,
        )
        synchronize_selected_dynamic_parameters(arrays)
    elif kind == "noncanonical_grid_json":
        arrays["candidate_parameter_json"][0, 0] = '{"l1_ratio":0.1,"alpha":0.001}'
        synchronize_selected_dynamic_parameters(arrays)
    elif kind == "nonbest_selection":
        arrays["selected_candidate_indices"][selection_index] = 1
        synchronize_selected_dynamic_parameters(arrays)
        assert np.all(arrays["candidate_inner_statuses"][candidate_index][1] == "valid")
    elif kind == "finite_invalid_score":
        arrays["candidate_inner_scores"][selection_index + (14, 2)] = 0.50
    elif kind == "unknown_status":
        arrays["candidate_inner_statuses"][selection_index + (0, 0)] = "unknown"
        arrays["candidate_inner_reasons"][selection_index + (0, 0)] = "unknown_status"
    elif kind == "valid_nonempty_reason":
        arrays["candidate_inner_reasons"][selection_index + (0, 0)] = "unexpected_reason"
    elif kind == "missing_inner_training_assignment":
        arrays["inner_selection_fold_ids"][0, 0, 2] = -1
    else:
        arrays["selected_candidate_indices"][selection_index] = -1
        synchronize_selected_dynamic_parameters(arrays)
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(
            run_directory,
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint=fingerprint,
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    "array_name",
    tuple(name for name in required_array_names() if name != "fold_scores"),
)
def test_save_requires_exact_declared_axes_for_every_non_score_array(array_name, tmp_path):
    """Every NPZ member has an exact named axis contract, not only fold scores."""
    input_paths = write_input_fixture(tmp_path)
    fingerprint = f"bad-axis-{array_name}"
    arrays, meta, scientific_config = make_dynamic_tuned_payload(
        input_paths,
        run_fingerprint=fingerprint,
    )
    meta = json.loads(json.dumps(meta))
    original_axes = meta["axes"][array_name]
    if original_axes:
        meta["axes"][array_name] = list(reversed(original_axes))
        if meta["axes"][array_name] == original_axes:
            meta["axes"][array_name] = [f"wrong_{original_axes[0]}"]
    else:
        meta["axes"][array_name] = ["not_a_scalar"]
    run_directory = tmp_path / array_name

    with pytest.raises(ValueError, match="axis|axes|metadata"):
        results.save_task_decoding_run(
            run_directory,
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint=fingerprint,
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("missing_units", "meta|unit|required"),
        ("wrong_target_units", "unit|target|relative_doubt"),
        ("wrong_score_units", "unit|score|r2"),
        ("wrong_candidate_score_units", "unit|candidate|score|r2"),
        ("wrong_coefficient_units", "unit|coefficient|relative_doubt"),
        ("wrong_intercept_units", "unit|intercept|relative_doubt"),
        ("wrong_edge_time_units", "unit|time"),
        ("wrong_center_time_units", "unit|time"),
        ("wrong_stage_timing_units", "unit|timing|stage"),
        ("wrong_total_timing_units", "unit|timing|total"),
        ("missing_provenance", "meta|provenance|required"),
        ("missing_parameters", "meta|parameter|required"),
        ("missing_paths", "path|meta|required"),
        ("missing_schema_version", "meta|schema|required"),
        ("missing_random_seed", "meta|seed|required"),
        ("nonportable_paths", "path|meta|portable"),
        ("missing_warnings", "warning|meta|required"),
        ("outer_fold_parameter_mismatch", "outer|fold|parameter|config"),
        ("inner_fold_parameter_mismatch", "inner|fold|parameter|config"),
        ("bin_width_parameter_mismatch", "bin|width|parameter|config"),
        ("selection_metric_parameter_mismatch", "selection|metric|parameter"),
    ),
)
def test_save_rejects_missing_or_target_misaligned_required_meta(kind, failure_pattern, tmp_path):
    """Meta is complete and units are keyed by the concrete saved target label."""
    input_paths = write_input_fixture(tmp_path)
    fingerprint = f"bad-meta-{kind}"
    arrays, meta, scientific_config = make_dynamic_tuned_payload(
        input_paths,
        run_fingerprint=fingerprint,
    )
    meta = json.loads(json.dumps(meta))
    if kind == "missing_units":
        meta.pop("units")
    elif kind == "wrong_target_units":
        meta["units"]["encoded_target_values"]["relative_doubt"] = "encoded class"
    elif kind == "wrong_score_units":
        meta["units"]["fold_scores"]["relative_doubt"] = {"r2": "fraction"}
    elif kind == "wrong_candidate_score_units":
        meta["units"]["candidate_inner_scores"]["relative_doubt"] = {"r2": "fraction"}
    elif kind == "wrong_coefficient_units":
        meta["units"]["coefficient_values"]["relative_doubt"] = "log-odds per SD"
    elif kind == "wrong_intercept_units":
        meta["units"]["fitted_intercepts"]["relative_doubt"] = "log-odds"
    elif kind == "wrong_edge_time_units":
        meta["units"]["time_bin_edges_s"] = "ms"
    elif kind == "wrong_center_time_units":
        meta["units"]["time_bin_centers_s"] = "ms"
    elif kind == "wrong_stage_timing_units":
        meta["units"]["stage_timing_seconds"] = "minutes"
    elif kind == "wrong_total_timing_units":
        meta["units"]["total_timing_seconds"] = "minutes"
    elif kind == "missing_provenance":
        meta.pop("provenance")
    elif kind == "missing_parameters":
        meta.pop("parameters")
    elif kind == "missing_paths":
        meta.pop("paths")
    elif kind == "missing_schema_version":
        meta.pop("schema_version")
    elif kind == "missing_random_seed":
        meta.pop("random_seed")
    elif kind == "nonportable_paths":
        meta["paths"]["session_root"] = "../another-session"
    elif kind == "missing_warnings":
        meta.pop("warnings")
    elif kind == "outer_fold_parameter_mismatch":
        meta["parameters"]["outer_fold_count"] = 3
    elif kind == "inner_fold_parameter_mismatch":
        meta["parameters"]["inner_fold_count"] = 2
    elif kind == "bin_width_parameter_mismatch":
        meta["parameters"]["bin_width_ms"] = 100
    else:
        meta["parameters"]["candidate_selection_metrics"] = {
            "relative_doubt": "balanced_accuracy"
        }
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(
            run_directory,
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint=fingerprint,
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("encoded", "unit|target|current_action"),
        ("score", "unit|score|current_action"),
        ("auc_score", "unit|score|auc|current_action"),
        ("candidate_score", "unit|candidate|score|current_action"),
        ("coefficient", "unit|coefficient|current_action"),
        ("intercept", "unit|intercept|current_action"),
    ),
)
def test_save_rejects_current_action_target_specific_unit_mismatches(
    kind,
    failure_pattern,
    tmp_path,
):
    """Categorical target units remain distinct from numerical native-target units."""
    input_paths = write_input_fixture(tmp_path)
    fingerprint = f"bad-current-action-units-{kind}"
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode="tuned",
        run_fingerprint=fingerprint,
    )
    meta = json.loads(json.dumps(arguments["meta"]))
    if kind == "encoded":
        meta["units"]["encoded_target_values"]["current_action"] = "unitless"
    elif kind == "score":
        meta["units"]["fold_scores"]["current_action"]["balanced_accuracy"] = "unitless"
    elif kind == "auc_score":
        meta["units"]["fold_scores"]["current_action"]["auc"] = "unitless"
    elif kind == "candidate_score":
        meta["units"]["candidate_inner_scores"]["current_action"][
            "balanced_accuracy"
        ] = "unitless"
    elif kind == "coefficient":
        meta["units"]["coefficient_values"]["current_action"] = "unitless per SD"
    else:
        meta["units"]["fitted_intercepts"]["current_action"] = "unitless"
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(
            run_directory,
            arrays=arguments["arrays"],
            meta=meta,
            input_manifest=arguments["input_manifest"],
            scientific_config=arguments["scientific_config"],
            feature_parameter_source=arguments["feature_parameter_source"],
            run_fingerprint=fingerprint,
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("array_name", "replacement"),
    (
        ("train_class_counts", (3, 5)),
        ("test_class_counts", (1, 3)),
    ),
)
def test_save_rejects_categorical_class_counts_inconsistent_with_target_fold_rows(
    array_name,
    replacement,
    tmp_path,
):
    """Categorical train/test class counts must equal the labels in each grouped fold."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays[array_name][0, 0, 0, 0, 0] = replacement
    run_directory = tmp_path / f"wrong-{array_name}"

    with pytest.raises(ValueError, match="class|count|fold"):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("eligible_nan", "eligibility|target|finite"),
        ("ineligible_finite", "eligibility|target|sentinel"),
        ("eligible_reason", "eligibility|reason"),
        ("ineligible_empty_reason", "eligibility|reason"),
        ("mask_outer_disagreement", "eligibility|outer|fold"),
    ),
)
def test_save_rejects_eligibility_value_reason_and_outer_sentinel_disagreements(
    kind,
    failure_pattern,
    tmp_path,
):
    """Target values, masks, reasons, and outer -1 sentinels describe one state."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    if kind == "eligible_nan":
        arrays["encoded_target_values"][1, 0] = np.nan
    elif kind == "ineligible_finite":
        arrays["encoded_target_values"][0, 11] = 0.0
    elif kind == "eligible_reason":
        arrays["eligibility_reason_codes"][1, 0] = "unexpected"
    elif kind == "ineligible_empty_reason":
        arrays["eligibility_reason_codes"][0, 11] = ""
    else:
        moved_value = arrays["encoded_target_values"][1, 0]
        arrays["eligibility_masks"][1, 0] = False
        arrays["encoded_target_values"][1, 0] = np.nan
        arrays["eligibility_reason_codes"][1, 0] = "target_ineligible"
        arrays["eligibility_masks"][1, 12] = True
        arrays["encoded_target_values"][1, 12] = moved_value
        arrays["eligibility_reason_codes"][1, 12] = ""
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("valid_nonempty_reason", "fit|status|reason"),
        ("unknown_fit_status", "fit|status"),
        ("unavailable_empty_reason", "fit|status|reason"),
        ("unavailable_finite_score", "fit|status|score"),
        ("unavailable_finite_coefficient", "fit|status|coefficient"),
        ("unavailable_finite_intercept", "fit|status|intercept"),
        ("valid_nan_metric", "fit|score|nonfinite"),
        ("valid_nan_active_coefficient", "coefficient|fit|nonfinite"),
    ),
)
def test_save_rejects_fit_status_reason_metric_coefficient_and_intercept_disagreements(
    kind,
    failure_pattern,
    tmp_path,
):
    """Fit status controls whether scores, active coefficients, and intercepts exist."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    fit_index = (0, 0, 0, 0, 0)
    if kind == "valid_nonempty_reason":
        arrays["fit_reason_codes"][fit_index] = "unexpected"
    elif kind == "unknown_fit_status":
        arrays["fit_status"][fit_index] = "mystery"
        arrays["fit_reason_codes"][fit_index] = "unknown_status"
    elif kind == "unavailable_empty_reason":
        arrays["fit_status"][fit_index] = "unavailable"
        arrays["fold_scores"][0, 0, 0, :, 0, 0] = np.nan
        arrays["coefficient_values"][fit_index] = np.nan
        arrays["fitted_intercepts"][fit_index] = np.nan
    elif kind == "unavailable_finite_score":
        arrays["fit_status"][fit_index] = "unavailable"
        arrays["fit_reason_codes"][fit_index] = "no_features"
        arrays["coefficient_values"][fit_index] = np.nan
        arrays["fitted_intercepts"][fit_index] = np.nan
    elif kind == "unavailable_finite_coefficient":
        arrays["fit_status"][fit_index] = "unavailable"
        arrays["fit_reason_codes"][fit_index] = "no_features"
        arrays["fold_scores"][0, 0, 0, :, 0, 0] = np.nan
        arrays["fitted_intercepts"][fit_index] = np.nan
    elif kind == "unavailable_finite_intercept":
        arrays["fit_status"][fit_index] = "unavailable"
        arrays["fit_reason_codes"][fit_index] = "no_features"
        arrays["fold_scores"][0, 0, 0, :, 0, 0] = np.nan
        arrays["coefficient_values"][fit_index] = np.nan
    elif kind == "valid_nan_metric":
        arrays["fold_scores"][0, 0, 0, 0, 0, 0] = np.nan
    else:
        arrays["coefficient_values"][0, 0, 0, 0, 0, 0] = np.nan
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("empty_active_id", "feature|identifier|active"),
        ("wrong_active_region", "feature|region"),
        ("active_padding_status", "feature|status|padding"),
        ("inactive_available_status", "feature|mask|status"),
        ("padding_nonempty_id", "feature|padding|identifier"),
        ("padding_nonempty_region", "feature|padding|region"),
    ),
)
def test_save_rejects_feature_identity_region_status_and_active_mask_disagreements(
    kind,
    failure_pattern,
    tmp_path,
):
    """Feature metadata describes active padded coefficient columns without ambiguity."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    if kind == "empty_active_id":
        arrays["coefficient_feature_ids"][0, 0, 0] = ""
    elif kind == "wrong_active_region":
        arrays["coefficient_feature_regions"][0, 0, 0] = "HPC"
    elif kind == "active_padding_status":
        arrays["coefficient_feature_statuses"][0, 0, 0] = "padding"
    elif kind == "inactive_available_status":
        arrays["coefficient_feature_statuses"][0, 0, 2] = "available"
    elif kind == "padding_nonempty_id":
        arrays["coefficient_feature_ids"][0, 0, 2] = "unexpected-padding-id"
    else:
        arrays["coefficient_feature_regions"][0, 0, 2] = "PFC"
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("regularization_mode", "target_index", "replacement"),
    (
        ("fixed", 0, '{"C": 0.1, "l1_ratio": 0.5}'),
        ("fixed", 1, '{"alpha": 0.1, "l1_ratio": 0.5}'),
        ("tuned", 0, '{"C": 0.1, "l1_ratio": 0.5}'),
        ("tuned", 1, '{"alpha": 0.1, "l1_ratio": 0.5}'),
        ("tuned", 0, '{"C": 1.0, "l1_ratio": 0.9}'),
        ("fixed", 1, '{"l1_ratio": 0.5, "alpha": 1.0}'),
    ),
)
def test_save_rejects_fixed_parameters_that_disagree_with_frozen_controls(
    regularization_mode,
    target_index,
    replacement,
    tmp_path,
):
    """Saved fixed controls match the exact family defaults in every mode and cell."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode=regularization_mode,
    )
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays["fixed_parameters_json"][target_index, 0, 0, 0, 0] = replacement
    run_directory = tmp_path / f"wrong-fixed-{regularization_mode}-{target_index}"

    with pytest.raises(ValueError, match="fixed|parameter|control|family"):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


def test_save_rejects_selected_parameters_in_fixed_mode(tmp_path):
    """Fixed-mode records do not fabricate tuned selected-parameter values."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays["selected_parameters_json"][0, 0, 0, 0, 0] = arrays[
        "fixed_parameters_json"
    ][0, 0, 0, 0, 0]
    run_directory = tmp_path / "fixed-selected-parameter"

    with pytest.raises(ValueError, match="fixed|selected|parameter|mode"):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    ("array_name", "bad_dtype"),
    (
        ("candidate_parameter_json", np.dtype(np.float64)),
        ("inner_selection_fold_ids", np.dtype(np.float64)),
        ("candidate_inner_scores", np.dtype(np.int64)),
        ("candidate_inner_statuses", np.dtype(np.float64)),
        ("candidate_inner_reasons", np.dtype(np.float64)),
        ("selected_candidate_indices", np.dtype(np.float64)),
    ),
)
def test_save_requires_portable_empty_tuned_member_dtypes_in_fixed_mode(
    array_name,
    bad_dtype,
    tmp_path,
):
    """Fixed-mode not-applicable tuned arrays retain their declared safe dtypes."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays[array_name] = np.empty((0,), dtype=bad_dtype)
    run_directory = tmp_path / array_name

    with pytest.raises(ValueError, match="dtype|fixed|candidate|inner"):
        results.save_task_decoding_run(run_directory, arrays=arrays, **{
            key: value for key, value in arguments.items() if key != "arrays"
        })
    assert not (run_directory / _RESULT_FILE).exists()


def replace_dynamic_block_payloads(
    arrays: Mapping[str, np.ndarray],
    payloads: Sequence[bytes],
) -> None:
    """Replace dynamic full-table block scalar bytes and recompute int64 offsets.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Mutable dynamic result payload containing full-table block members.
    payloads : sequence[bytes]
        One UTF-8 JSON byte sequence per full-table row, in row order.

    Returns
    -------
    None
        Updates the one-dimensional uint8 payload and ``(full_row + 1,)``
        int64 boundary offsets in place.
    """
    arrays["full_table_block_ids_utf8"] = np.frombuffer(
        b"".join(payloads),
        dtype=np.uint8,
    ).copy()
    arrays["full_table_block_id_offsets"] = np.r_[
        np.array([0], dtype=np.int64),
        np.cumsum([len(payload) for payload in payloads], dtype=np.int64),
    ]


def dynamic_block_payloads(arrays: Mapping[str, np.ndarray]) -> list[bytes]:
    """Return the exact saved canonical JSON byte sequence for each full row.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Dynamic result payload with concatenated block bytes and int64 offsets.

    Returns
    -------
    list[bytes]
        One raw block-label JSON value per full-table row, preserving type text.
    """
    payload = arrays["full_table_block_ids_utf8"].tobytes()
    offsets = arrays["full_table_block_id_offsets"]
    return [payload[start:stop] for start, stop in zip(offsets[:-1], offsets[1:])]


@pytest.mark.parametrize(
    ("kind", "failure_pattern"),
    (
        ("two_dimensional_payload", "block|payload|dimension|axis|schema"),
        ("noncanonical_json", "block|canonical|JSON"),
        ("bad_offset_start", "block|offset"),
        ("bad_offset_end", "block|offset"),
        ("descending_offsets", "block|offset"),
        ("empty_offset_segment", "block|offset|empty"),
        ("wrong_offset_length", "block|offset|shape"),
        ("nonscalar_block_label", "block|scalar|label"),
    ),
)
def test_save_rejects_isolated_dynamic_block_payload_and_offset_violations(
    kind,
    failure_pattern,
    tmp_path,
):
    """Dynamic behavioral block storage is 1-D canonical scalar JSON with exact offsets."""
    input_paths = write_input_fixture(tmp_path)
    fingerprint = f"bad-block-{kind}"
    arrays, meta, scientific_config = make_dynamic_tuned_payload(
        input_paths,
        run_fingerprint=fingerprint,
    )
    arrays = {name: value.copy() for name, value in arrays.items()}
    if kind == "two_dimensional_payload":
        arrays["full_table_block_ids_utf8"] = arrays["full_table_block_ids_utf8"].reshape(1, -1)
    elif kind == "noncanonical_json":
        payloads = dynamic_block_payloads(arrays)
        payloads[0] = b'"alpha" '
        replace_dynamic_block_payloads(arrays, payloads)
    elif kind == "bad_offset_start":
        arrays["full_table_block_id_offsets"][0] = 1
    elif kind == "bad_offset_end":
        arrays["full_table_block_id_offsets"][-1] -= 1
    elif kind == "descending_offsets":
        arrays["full_table_block_id_offsets"][2] = (
            arrays["full_table_block_id_offsets"][1] - 1
        )
    elif kind == "empty_offset_segment":
        arrays["full_table_block_id_offsets"][1] = 0
    elif kind == "wrong_offset_length":
        arrays["full_table_block_id_offsets"] = arrays["full_table_block_id_offsets"][:-1]
    else:
        payloads = dynamic_block_payloads(arrays)
        payloads[0] = b"[]"
        replace_dynamic_block_payloads(arrays, payloads)
    run_directory = tmp_path / kind

    with pytest.raises(ValueError, match=failure_pattern):
        results.save_task_decoding_run(
            run_directory,
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint=fingerprint,
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    "function_name",
    (
        "build_input_manifest",
        "scientific_source_fingerprint",
        "validate_scientific_source_cleanliness",
        "build_run_fingerprint",
        "save_target_checkpoint",
        "load_target_checkpoint",
        "save_task_decoding_run",
        "load_task_decoding_run",
    ),
)
def test_public_result_helpers_have_explicit_parameters_and_returns_docstrings(function_name):
    """Every public WP5A seam states input and output contracts for scientific review."""
    docstring = inspect.getdoc(getattr(results, function_name))
    assert docstring is not None
    assert "Parameters" in docstring
    assert "Returns" in docstring


@pytest.mark.parametrize(
    "function_name",
    (
        "save_target_checkpoint",
        "load_target_checkpoint",
        "save_task_decoding_run",
        "load_task_decoding_run",
    ),
)
def test_array_bearing_public_helpers_document_shapes_axes_and_units(function_name):
    """Array persistence helpers state shape/axis conventions and scientific units."""
    docstring = inspect.getdoc(getattr(results, function_name))
    assert docstring is not None
    lowered = docstring.lower()
    assert "shape" in lowered
    assert "axis" in lowered
    assert "unit" in lowered


@pytest.mark.parametrize("function_name", ("_require", "_write_json_if_matching"))
def test_private_validation_and_sidecar_helpers_document_returns(function_name):
    """Private validation/publication helpers state their explicit return contract."""
    docstring = inspect.getdoc(getattr(results, function_name))
    assert docstring is not None
    assert "Parameters" in docstring
    assert "Returns" in docstring


@pytest.mark.parametrize(
    "function_name",
    ("_atomic_npz", "_decode_blocks", "_require", "_validate_arrays"),
)
def test_array_bearing_private_helpers_document_shapes_and_axes(function_name):
    """Private array helpers state shape and axis contracts even without physical units."""
    docstring = inspect.getdoc(getattr(results, function_name))
    assert docstring is not None
    lowered = docstring.lower()
    assert "shape" in lowered
    assert "axis" in lowered


@pytest.mark.parametrize("function_name", ("_validate_arrays",))
def test_scientific_private_array_helpers_document_applicable_units(function_name):
    """Private schema helpers describe units when they interpret scientific arrays."""
    docstring = inspect.getdoc(getattr(results, function_name))
    assert docstring is not None
    assert "unit" in docstring.lower()


def test_tuned_no_valid_candidate_uses_explicit_unavailable_cell_sentinels(tmp_path):
    """No-valid tuning cells retain invalid audit rows and an explicit unavailable fit.

    The selected-index sentinel is -1 and the selected parameter text is empty;
    this is distinct from a successful candidate zero and avoids fictional
    fitted coefficients, intercepts, or outer scores.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode="tuned",
        run_fingerprint="no-valid-candidate",
    )
    arrays = arguments["arrays"]
    assert isinstance(arrays, dict)
    altered = {name: value.copy() for name, value in arrays.items()}
    fit_index = (0, 0, 0, 0, 0)
    audit_index = fit_index + (slice(None), slice(None))
    altered["fit_status"][fit_index] = "unavailable"
    altered["fit_reason_codes"][fit_index] = "no_valid_tuning_candidate"
    altered["fold_scores"][0, 0, 0, :, 0, 0] = np.nan
    altered["coefficient_values"][fit_index] = np.nan
    altered["fitted_intercepts"][fit_index] = np.nan
    requested_count = int(altered["requested_feature_counts"][fit_index])
    altered["effective_feature_counts"][fit_index] = 0
    altered["candidate_inner_scores"][audit_index] = np.nan
    altered["candidate_inner_statuses"][audit_index] = "invalid"
    altered["candidate_inner_reasons"][audit_index] = "candidate_fit_failed"
    altered["selected_candidate_indices"][fit_index] = -1
    altered["selected_parameters_json"][fit_index] = ""
    run_directory = tmp_path / "no-valid-candidate"
    results.save_task_decoding_run(
        run_directory,
        arrays=altered,
        meta=arguments["meta"],
        input_manifest=arguments["input_manifest"],
        scientific_config=arguments["scientific_config"],
        feature_parameter_source=arguments["feature_parameter_source"],
        run_fingerprint="no-valid-candidate",
    )
    loaded = results.load_task_decoding_run(run_directory)
    saved = loaded["arrays"]
    assert saved["fit_status"][fit_index] == "unavailable"
    assert saved["requested_feature_counts"][fit_index] == requested_count
    assert saved["effective_feature_counts"][fit_index] == 0
    assert saved["selected_candidate_indices"][fit_index] == -1
    assert saved["selected_parameters_json"][fit_index] == ""
    assert np.isnan(saved["candidate_inner_scores"][audit_index]).all()
    assert set(saved["candidate_inner_statuses"][audit_index].flat) == {"invalid"}
    assert set(saved["candidate_inner_reasons"][audit_index].flat) == {"candidate_fit_failed"}


def test_wp5a_save_rejects_no_valid_candidate_with_retained_effective_features(tmp_path):
    """No-valid-candidate cells must not claim retained transformed features."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    fit_index = (0, 0, 0, 0, 0)
    audit_index = fit_index + (slice(None), slice(None))
    arrays["fit_status"][fit_index] = "unavailable"
    arrays["fit_reason_codes"][fit_index] = "no_valid_tuning_candidate"
    arrays["fold_scores"][0, 0, 0, :, 0, 0] = np.nan
    arrays["coefficient_values"][fit_index] = np.nan
    arrays["fitted_intercepts"][fit_index] = np.nan
    arrays["effective_feature_counts"][fit_index] = 1
    arrays["candidate_inner_scores"][audit_index] = np.nan
    arrays["candidate_inner_statuses"][audit_index] = "invalid"
    arrays["candidate_inner_reasons"][audit_index] = "candidate_fit_failed"
    arrays["selected_candidate_indices"][fit_index] = -1
    arrays["selected_parameters_json"][fit_index] = ""

    with pytest.raises(ValueError, match="candidate|effective|unavailable|feature"):
        results.save_task_decoding_run(
            tmp_path / "no-valid-candidate-with-features",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_tuned_inner_plan_unavailable_preserves_complete_invalid_audit_without_outputs(tmp_path):
    """Unavailable grouped inner plans are distinct from evaluated no-valid candidates.

    Every outer/refit cell is unavailable because no inner partition can be
    formed. The candidate audit remains shaped and diagnosable, but contains no
    finite score, selected setting, fitted score, coefficient, or intercept.
    """
    input_paths = write_input_fixture(tmp_path)
    fingerprint = "inner-plan-unavailable"
    arrays, meta, scientific_config = make_inner_plan_unavailable_tuned_payload(
        input_paths,
        run_fingerprint=fingerprint,
    )
    requested_feature_counts = arrays["requested_feature_counts"].copy()
    unavailable_reason = "inner_plan_unavailable"
    blocks = encoded_block_labels(arrays)
    outer_ids = arrays["outer_fold_ids"][0]
    assert arrays["fold_labels"].tolist() == [0, 1, 2]
    assert [np.count_nonzero(outer_ids == fold) for fold in range(3)] == [2, 2, 2]
    for outer_fold in arrays["fold_labels"]:
        test_rows = np.flatnonzero(outer_ids == outer_fold)
        train_rows = np.flatnonzero(outer_ids != outer_fold)
        test_blocks = {blocks[arrays["trial_row_indices"][row]] for row in test_rows}
        train_blocks = {blocks[arrays["trial_row_indices"][row]] for row in train_rows}
        assert len(test_blocks) == 1
        assert len(train_blocks) == 2
        assert len(train_blocks) < meta["parameters"]["inner_fold_count"]
    run_directory = tmp_path / "inner-plan-unavailable"
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        meta=meta,
        input_manifest=build_manifest(input_paths),
        scientific_config=scientific_config,
        feature_parameter_source=input_paths["feature_parameters"],
        run_fingerprint=fingerprint,
    )

    loaded = results.load_task_decoding_run(
        run_directory,
        expected_run_fingerprint=fingerprint,
    )
    saved = loaded["arrays"]
    assert np.all(saved["inner_selection_fold_ids"] == -1)
    assert set(saved["fit_status"].flat) == {"unavailable"}
    assert set(saved["fit_reason_codes"].flat) == {unavailable_reason}
    np.testing.assert_array_equal(saved["requested_feature_counts"], requested_feature_counts)
    assert np.all(saved["effective_feature_counts"] == 0)
    assert np.isnan(saved["fold_scores"]).all()
    assert np.isnan(saved["coefficient_values"]).all()
    assert np.isnan(saved["fitted_intercepts"]).all()
    assert np.isnan(saved["candidate_inner_scores"]).all()
    assert set(saved["candidate_inner_statuses"].flat) == {"invalid"}
    assert set(saved["candidate_inner_reasons"].flat) == {unavailable_reason}
    assert np.all(saved["selected_candidate_indices"] == -1)
    assert not np.any(saved["selected_parameters_json"] != "")


def test_save_validates_schema_before_creating_a_result_npz(tmp_path):
    """Malformed fit-array dtypes fail before an NPZ can look complete on disk."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode="fixed",
        run_fingerprint="invalid-before-publication",
    )
    arrays = arguments["arrays"]
    assert isinstance(arrays, dict)
    altered = {name: value.copy() for name, value in arrays.items()}
    altered["fit_status"] = altered["fit_status"].astype(object)
    run_directory = tmp_path / "prevalidation-run"

    with pytest.raises(ValueError, match="dtype|schema|fit"):
        results.save_task_decoding_run(
            run_directory,
            arrays=altered,
            meta=arguments["meta"],
            input_manifest=arguments["input_manifest"],
            scientific_config=arguments["scientific_config"],
            feature_parameter_source=arguments["feature_parameter_source"],
            run_fingerprint="invalid-before-publication",
        )
    assert not (run_directory / _RESULT_FILE).exists()


@pytest.mark.parametrize(
    "status_output",
    (
        "R  docs/unrelated.md -> src/neural_analysis/task_decoding/renamed_model.py\n",
        '?? "src/neural_analysis/task_decoding/new science.py"\n',
    ),
)
def test_cleanliness_rejects_renamed_and_quoted_task_package_paths(
    status_output,
    tmp_path,
    monkeypatch,
):
    """Git porcelain parsing identifies relevant paths after rename and quoting syntax."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    patch_git_commands(monkeypatch, status_output=status_output)

    with pytest.raises(ValueError, match="clean|source|dirty|untracked"):
        results.validate_scientific_source_cleanliness(repository_root)


def test_cleanliness_requires_tracking_for_an_added_task_decoding_python_module(
    tmp_path,
    monkeypatch,
):
    """A dynamically discovered task package module is part of clean scientific source."""
    repository_root = write_scoped_source_tree(tmp_path, name="repository")
    added_module = repository_root / "src/neural_analysis/task_decoding/added_science.py"
    added_module.write_text("# dynamically discovered scientific module\n", encoding="ascii")
    patch_git_commands(monkeypatch, status_output="")

    with pytest.raises(ValueError, match="tracked|source|scientific"):
        results.validate_scientific_source_cleanliness(repository_root)


def configure_real_outer_unavailable_target(
    arrays: dict[str, np.ndarray],
    *,
    target: int,
) -> str:
    """Configure one target's persisted cells from a real unavailable outer plan.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Complete mutable fixed or tuned result payload. Target values have
        shape ``(target, full_table_row)`` and remain dimensionless; fold and
        fit members preserve their documented common-row and fit axes.
    target : int
        Zero-based target position. Its eligible values are made constant so
        the project grouped outer splitter must return an unavailable plan.

    Returns
    -------
    str
        Exact nonempty unavailable reason returned by
        ``modeling.make_outer_splits`` and stored at target scope. Per-fit and
        per-candidate arrays use the compact
        ``target_outer_unavailable`` reason code instead.
    """
    family = str(arrays["target_families"][target])
    outer_before = arrays["outer_fold_ids"][target]
    eligible_common_rows = np.flatnonzero(outer_before >= 0)
    trial_rows = arrays["trial_row_indices"]
    eligible_full_rows = trial_rows[eligible_common_rows]
    constant_value = 0.0 if family == "categorical" else 0.5
    arrays["encoded_target_values"][target, eligible_full_rows] = constant_value
    blocks = np.asarray(encoded_block_labels(arrays), dtype=object)[eligible_full_rows]
    plan = modeling.make_outer_splits(
        arrays["encoded_target_values"][target, eligible_full_rows],
        blocks,
        target_family=family,
        fold_count=int(arrays["fold_labels"].size),
    )
    assert not plan.is_available
    assert plan.unavailable_reason == "constant target"
    assert np.all(plan.fold_ids == -1)
    reason = plan.unavailable_reason
    arrays["target_status"][target] = "unavailable"
    arrays["target_unavailable_reasons"][target] = reason
    unavailable_outer_ids = np.full_like(outer_before, -1)
    unavailable_outer_ids[eligible_common_rows] = plan.fold_ids
    arrays["outer_fold_ids"][target] = unavailable_outer_ids
    arrays["fit_status"][target].fill("unavailable")
    arrays["fit_reason_codes"][target].fill(_TARGET_OUTER_UNAVAILABLE_CODE)
    arrays["effective_feature_counts"][target].fill(0)
    arrays["train_counts"][target].fill(0)
    arrays["test_counts"][target].fill(0)
    if family == "categorical":
        arrays["train_class_counts"][target].fill(0)
        arrays["test_class_counts"][target].fill(0)
    arrays["fold_scores"][target].fill(np.nan)
    arrays["coefficient_values"][target].fill(np.nan)
    arrays["fitted_intercepts"][target].fill(np.nan)
    if arrays["inner_selection_fold_ids"].size:
        arrays["inner_selection_fold_ids"][target].fill(-1)
        arrays["candidate_inner_scores"][target].fill(np.nan)
        arrays["candidate_inner_statuses"][target].fill("invalid")
        arrays["candidate_inner_reasons"][target].fill(_TARGET_OUTER_UNAVAILABLE_CODE)
        arrays["selected_candidate_indices"][target].fill(-1)
        arrays["selected_parameters_json"][target].fill("")
    return reason


@pytest.mark.parametrize(
    ("regularization_mode", "target"),
    (
        ("fixed", 0),
        ("tuned", 0),
        ("fixed", 1),
        ("tuned", 1),
    ),
)
def test_wp5a_target_outer_unavailability_round_trips_with_explicit_target_axis(
    regularization_mode,
    target,
    tmp_path,
):
    """Outer-split failures retain eligible values but publish no fitted cells.

    Categorical and numerical targets become constant on their eligible rows,
    producing a real project splitter unavailable plan with every common-row
    outer ID ``-1``. Its exact reason explains why unavailable sentinels
    replace otherwise possible fits; categorical class counts use zero while
    numerical ones retain their ``-1`` not-applicable sentinels.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(
        input_paths,
        regularization_mode=regularization_mode,
        run_fingerprint=f"outer-unavailable-{regularization_mode}",
    )
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    requested_counts = arrays["requested_feature_counts"][target].copy()
    reason = configure_real_outer_unavailable_target(arrays, target=target)
    retained_values = arrays["encoded_target_values"][target].copy()
    run_directory = tmp_path / f"outer-unavailable-{regularization_mode}"

    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        meta=arguments["meta"],
        input_manifest=arguments["input_manifest"],
        scientific_config=arguments["scientific_config"],
        feature_parameter_source=arguments["feature_parameter_source"],
        run_fingerprint=f"outer-unavailable-{regularization_mode}",
    )

    saved = results.load_task_decoding_run(run_directory)["arrays"]
    assert saved["target_status"].dtype == np.dtype("U16")
    assert saved["target_unavailable_reasons"].dtype == np.dtype("U128")
    expected_statuses = (
        ["unavailable", "available"] if target == 0 else ["available", "unavailable"]
    )
    expected_reasons = [reason, ""] if target == 0 else ["", reason]
    assert saved["target_status"].tolist() == expected_statuses
    assert saved["target_unavailable_reasons"].tolist() == expected_reasons
    np.testing.assert_array_equal(saved["encoded_target_values"][target], retained_values)
    np.testing.assert_array_equal(saved["requested_feature_counts"][target], requested_counts)
    assert np.all(saved["outer_fold_ids"][target] == -1)
    assert set(saved["fit_status"][target].flat) == {"unavailable"}
    assert set(saved["fit_reason_codes"][target].flat) == {_TARGET_OUTER_UNAVAILABLE_CODE}
    assert np.all(saved["train_counts"][target] == 0)
    assert np.all(saved["test_counts"][target] == 0)
    expected_class_counts = 0 if target == 0 else -1
    assert np.all(saved["train_class_counts"][target] == expected_class_counts)
    assert np.all(saved["test_class_counts"][target] == expected_class_counts)
    assert np.isnan(saved["fold_scores"][target]).all()
    assert np.isnan(saved["coefficient_values"][target]).all()
    assert np.isnan(saved["fitted_intercepts"][target]).all()
    if regularization_mode == "tuned":
        assert np.all(saved["inner_selection_fold_ids"][target] == -1)
        assert np.isnan(saved["candidate_inner_scores"][target]).all()
        assert set(saved["candidate_inner_statuses"][target].flat) == {"invalid"}
        assert set(saved["candidate_inner_reasons"][target].flat) == {
            _TARGET_OUTER_UNAVAILABLE_CODE
        }
        assert np.all(saved["selected_candidate_indices"][target] == -1)
        assert not np.any(saved["selected_parameters_json"][target] != "")


def test_wp5a_save_rejects_unavailable_target_when_values_restore_a_splittable_plan(tmp_path):
    """Unavailable sentinels cannot mask a categorical target with real outer splits."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target = 0
    original_values = arrays["encoded_target_values"][target].copy()
    target_outer = arrays["outer_fold_ids"][target]
    other_outer = arrays["outer_fold_ids"][1]
    payloads = dynamic_block_payloads(arrays)
    for common_row, full_row in enumerate(arrays["trial_row_indices"]):
        fold = target_outer[common_row]
        if fold < 0:
            fold = other_outer[common_row]
        label = f"fold-{int(fold)}" if fold >= 0 else f"ineligible-{int(full_row)}"
        payloads[int(full_row)] = json.dumps(label).encode("ascii")
    replace_dynamic_block_payloads(arrays, payloads)
    assert_grouped_split_integrity(arrays)
    configure_real_outer_unavailable_target(arrays, target=target)
    arrays["encoded_target_values"][target] = original_values
    trial_rows = arrays["trial_row_indices"]
    eligible_common = arrays["eligibility_masks"][target, trial_rows]
    decoded_blocks = np.asarray(decoded_block_labels(arrays), dtype=object)
    plan = modeling.make_outer_splits(
        arrays["encoded_target_values"][target, trial_rows[eligible_common]],
        decoded_blocks[trial_rows[eligible_common]],
        target_family="categorical",
        fold_count=int(arrays["fold_labels"].size),
    )
    assert plan.is_available
    assert np.any(plan.fold_ids >= 0)

    with pytest.raises(ValueError, match="unavailable|outer|target|split"):
        results.save_task_decoding_run(
            tmp_path / "unavailable-with-splittable-values",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_save_rejects_available_numerical_target_with_constant_values(tmp_path):
    """Available numerical target folds must remain constructible from saved values."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target = 1
    trial_rows = arrays["trial_row_indices"]
    eligible_common = arrays["eligibility_masks"][target, trial_rows]
    eligible_full_rows = trial_rows[eligible_common]
    arrays["encoded_target_values"][target, eligible_full_rows] = 0.5
    decoded_blocks = np.asarray(decoded_block_labels(arrays), dtype=object)
    plan = modeling.make_outer_splits(
        arrays["encoded_target_values"][target, eligible_full_rows],
        decoded_blocks[eligible_full_rows],
        target_family="numerical",
        fold_count=int(arrays["fold_labels"].size),
    )
    assert not plan.is_available
    assert plan.unavailable_reason == "constant target"
    assert np.all(plan.fold_ids == -1)

    with pytest.raises(ValueError, match="available|outer|target|constant|split"):
        results.save_task_decoding_run(
            tmp_path / "available-numerical-constant-target",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_scalar_block_labels_are_decoded_before_grouped_split_planning(tmp_path):
    """Unavailable scalar numeric groups round-trip without byte-based recomputation.

    The exact counterexample produces an unavailable plan for numeric labels,
    but a different available plan for their canonical UTF-8 bytes. The saved
    target therefore proves the serializer must preserve decoded scalar group
    semantics if it recomputes outer availability in the future.
    """
    numeric_blocks = np.array([2, 3, 3, 3, 10, 10, 10, 11, 11, 11, 11], dtype=np.int64)
    target_values = np.array([0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1], dtype=np.int64)
    numeric_plan = modeling.make_outer_splits(
        target_values,
        numeric_blocks,
        target_family="categorical",
        fold_count=3,
    )
    byte_plan = modeling.make_outer_splits(
        target_values,
        np.array([str(value).encode("ascii") for value in numeric_blocks], dtype="S2"),
        target_family="categorical",
        fold_count=3,
    )
    assert not numeric_plan.is_available
    assert numeric_plan.unavailable_reason == "categorical grouped fold lacks both classes"
    assert np.all(numeric_plan.fold_ids == -1)
    assert byte_plan.is_available
    np.testing.assert_array_equal(
        byte_plan.fold_ids,
        np.array([0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1], dtype=int),
    )

    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target = 0
    configure_real_outer_unavailable_target(arrays, target=target)
    payloads = dynamic_block_payloads(arrays)
    for full_row, numeric_block in enumerate(numeric_blocks):
        payloads[full_row] = json.dumps(int(numeric_block)).encode("ascii")
    replace_dynamic_block_payloads(arrays, payloads)
    configure_real_outer_unavailable_target(arrays, target=1)
    arrays["eligibility_masks"][target, 12] = False
    arrays["eligibility_reason_codes"][target, 12] = "target_ineligible"
    arrays["eligibility_counts"][target] = 11
    arrays["encoded_target_values"][target] = np.nan
    arrays["encoded_target_values"][target, :11] = target_values
    arrays["target_unavailable_reasons"][target] = numeric_plan.unavailable_reason
    arrays["fit_reason_codes"][target].fill(_TARGET_OUTER_UNAVAILABLE_CODE)
    run_directory = tmp_path / "numeric-scalar-unavailable"
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(run_directory)["arrays"]
    assert saved["target_status"][target] == "unavailable"
    assert saved["target_unavailable_reasons"][target] == numeric_plan.unavailable_reason
    assert set(saved["fit_reason_codes"][target].flat) == {_TARGET_OUTER_UNAVAILABLE_CODE}
    assert np.all(saved["outer_fold_ids"][target] == -1)
    assert decoded_block_labels(saved)[:11] == tuple(numeric_blocks.tolist())
    np.testing.assert_array_equal(saved["encoded_target_values"][target, :11], target_values)


@pytest.mark.parametrize(
    ("status", "reason"),
    (
        ("unknown", ""),
        ("available", "unexpected reason"),
        ("unavailable", ""),
        ("unavailable", "declared unavailable without matching folds"),
    ),
)
def test_wp5a_save_rejects_target_availability_status_reason_or_payload_contradictions(
    status,
    reason,
    tmp_path,
):
    """Target status/reason tokens must agree with otherwise-valid fold payloads.

    The unavailable case deliberately leaves valid outer IDs and fitted cells
    intact, proving that an unavailable declaration cannot coexist with a
    scientifically available target payload.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays["target_status"][0] = status
    arrays["target_unavailable_reasons"][0] = reason

    with pytest.raises(ValueError, match="target|status|unavailable|reason|available"):
        results.save_task_decoding_run(
            tmp_path / f"invalid-target-status-{status}",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_save_rejects_blank_reason_for_an_otherwise_coherent_unavailable_target(
    tmp_path,
):
    """An unavailable target requires its splitter-derived nonempty scientific reason."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    reason = configure_real_outer_unavailable_target(arrays, target=0)
    assert reason == "constant target"
    arrays["target_unavailable_reasons"][0] = ""

    with pytest.raises(ValueError, match="target|unavailable|reason"):
        results.save_task_decoding_run(
            tmp_path / "unavailable-target-without-reason",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize(
    ("target", "member", "replacement"),
    (
        (0, "target_source_labels", "reward"),
        (0, "target_families", "numerical"),
        (0, "target_positive_classes", -1),
        (1, "target_positive_classes", 1),
    ),
)
def test_wp5a_save_rejects_incoherent_target_definition_metadata(
    target,
    member,
    replacement,
    tmp_path,
):
    """Target source, family, and positive-class metadata must match target definitions.

    Each otherwise-complete payload changes exactly one target-definition
    member, so rejection cannot be attributed to stale fold assignments or
    unrelated schema fields.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    arrays[member][target] = replacement

    with pytest.raises(ValueError, match="target|source|family|positive|metadata|definition"):
        results.save_task_decoding_run(
            tmp_path / f"incoherent-{member}-{target}",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize(
    ("target", "replacement", "expected_mapping"),
    (
        (0, '{"1":"left","0":"right"}', "categorical"),
        (0, '{"0":"right","1":"wrong"}', "categorical"),
        (0, '{"native_target":true}', "categorical"),
        (0, "not-json", "categorical"),
        (1, '{"native_target":false}', "numerical"),
    ),
)
def test_wp5a_save_rejects_noncanonical_or_wrong_target_label_mapping(
    target,
    replacement,
    expected_mapping,
    tmp_path,
):
    """Target mapping JSON must match the existing targets-module family contract."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    if expected_mapping == "categorical":
        expected_text = json.dumps(
            targets._CATEGORICAL_CLASS_LABELS["current_action"],
            sort_keys=True,
            separators=(",", ":"),
        )
    else:
        expected_text = '{"native_target":true}'
    assert arrays["target_label_mappings_json"][target] == expected_text
    arrays["target_label_mappings_json"][target] = replacement

    with pytest.raises(ValueError, match="mapping|target|canonical|family|label"):
        results.save_task_decoding_run(
            tmp_path / "bad-target-mapping",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize(
    "kind",
    ("duplicate", "hole", "pca_number", "combined_order", "requested_small", "requested_large"),
)
def test_wp5a_save_rejects_fold_local_feature_map_contract_violations(kind, tmp_path):
    """Feature metadata must preserve prefix masks and canonical representation identities."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    if kind == "duplicate":
        arrays["coefficient_feature_ids"][0, 1, 1] = arrays["coefficient_feature_ids"][0, 1, 0]
    elif kind == "hole":
        arrays["coefficient_active_masks"][0, 1, 0] = False
        arrays["coefficient_feature_ids"][0, 1, 0] = ""
        arrays["coefficient_feature_regions"][0, 1, 0] = ""
        arrays["coefficient_feature_statuses"][0, 1, 0] = "padding"
        arrays["coefficient_values"][:, 0, 1, :, :, 0] = np.nan
        arrays["effective_feature_counts"][:, 0, 1, :, :] = 1
    elif kind == "pca_number":
        arrays["coefficient_feature_ids"][0, 0, 0] = "PFC:PC2"
    elif kind == "combined_order":
        arrays["coefficient_feature_ids"][2, 1, :4] = (
            "probe-hpc:5",
            "probe-hpc:17",
            "probe-pfc:11",
            "probe-pfc:19",
        )
        arrays["coefficient_feature_regions"][2, 1, :4] = ("HPC", "HPC", "PFC", "PFC")
    elif kind == "requested_small":
        arrays["requested_feature_counts"][:, 2, 1] = 3
    else:
        arrays["requested_feature_counts"][:, 0, 1] = 3

    with pytest.raises(ValueError, match="feature|active|prefix|PCA|requested|identity|order"):
        results.save_task_decoding_run(
            tmp_path / f"bad-feature-{kind}",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_save_rejects_direct_width_below_pca_transform_width(tmp_path):
    """Direct-unit width cannot fall below the shared regional PCA width.

    Across all time bins for one target/fold, PFC direct and its combined PFC
    segment retain one reusable unit while PCA retains two components. A shared
    regional transform requires ``effective_pca <= effective_direct``; the
    corresponding combined widths of four and three preserve the same mismatch.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target, fold = 0, 0
    time_count = arrays["time_bin_centers_s"].size
    for time in range(time_count):
        pfc_index = (target, 0, 1, time, fold)
        combined_index = (target, 2, 1, time, fold)
        arrays["effective_feature_counts"][pfc_index] = 1
        arrays["effective_feature_counts"][combined_index] = 3
        arrays["coefficient_values"][pfc_index + (1,)] = np.nan
        arrays["coefficient_values"][combined_index + (1,)] = np.nan
    with pytest.raises(ValueError, match="direct|PCA|effective|transform|feature"):
        results.save_task_decoding_run(
            tmp_path / "direct-width-below-pca-width",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def mark_fit_unavailable_without_features(
    arrays: dict[str, np.ndarray],
    fit_index: tuple[int, int, int, int, int],
    *,
    reason: str,
) -> None:
    """Record one fixed- or tuned-mode fit with no outer feature matrix.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Mutable result payload for either regularization mode. Fit members use axes
        ``(target, region, representation, time, fold)`` and score members
        insert the metric axis before time.
    fit_index : tuple[int, int, int, int, int]
        Address of one target/region/representation/time/fold cell.
    reason : str
        Nonempty compact unavailable-state code stored in ``fit_reason_codes``.

    Returns
    -------
    None
        Mutates only the addressed cell to an explicit unavailable, zero-feature
        state with NaN scores, coefficients, and intercept. This represents a
        failed regional transform or a tuned cell with no selected candidate.
    """
    target, region, representation, time, fold = fit_index
    arrays["fit_status"][fit_index] = "unavailable"
    arrays["fit_reason_codes"][fit_index] = reason
    arrays["effective_feature_counts"][fit_index] = 0
    arrays["fold_scores"][target, region, representation, :, time, fold] = np.nan
    arrays["coefficient_values"][fit_index] = np.nan
    arrays["fitted_intercepts"][fit_index] = np.nan


def mark_tuned_cell_no_valid_candidate(
    arrays: dict[str, np.ndarray],
    fit_index: tuple[int, int, int, int, int],
) -> None:
    """Record one tuned cell with no valid inner-CV candidate.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Mutable tuned-mode result payload. Candidate audit axes are
        ``(target, outer_fold, region, representation, time, candidate,
        inner_fold)``.
    fit_index : tuple[int, int, int, int, int]
        Address using fit axes ``(target, region, representation, time, fold)``.

    Returns
    -------
    None
        Mutates the fit, selection, and every candidate audit member for the
        addressed cell to the explicit unavailable/no-selection sentinels.
    """
    target, region, representation, time, fold = fit_index
    reason = "no_valid_tuning_candidate"
    mark_fit_unavailable_without_features(arrays, fit_index, reason=reason)
    selection_index = (target, fold, region, representation, time)
    arrays["selected_candidate_indices"][selection_index] = -1
    arrays["selected_parameters_json"][selection_index] = ""
    audit_index = selection_index + (slice(None), slice(None))
    arrays["candidate_inner_scores"][audit_index] = np.nan
    arrays["candidate_inner_statuses"][audit_index] = "invalid"
    arrays["candidate_inner_reasons"][audit_index] = reason


def test_wp4_fixed_regional_transform_unavailability_round_trips_with_hpc_only_features(tmp_path):
    """One regional transform failure leaves the other region usable after round-trip.

    For one categorical target and outer fold, all eight time bins mark PFC
    PCA/direct and both combined representations unavailable with zero
    effective features. HPC PCA/direct retain their two usable features, so
    combined failure is not incorrectly required to equal a regional sum.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target, fold = 0, 0
    reason = "pfc_transform_unavailable"
    unavailable_cells = ((0, 0), (0, 1), (2, 0), (2, 1))
    for time in range(arrays["time_bin_centers_s"].size):
        for region, representation in unavailable_cells:
            mark_fit_unavailable_without_features(
                arrays,
                (target, region, representation, time, fold),
                reason=reason,
            )

    run_directory = tmp_path / "pfc-transform-unavailable"
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(run_directory)["arrays"]
    for time in range(saved["time_bin_centers_s"].size):
        for region, representation in unavailable_cells:
            fit_index = (target, region, representation, time, fold)
            assert saved["fit_status"][fit_index] == "unavailable"
            assert saved["fit_reason_codes"][fit_index] == reason
            assert saved["effective_feature_counts"][fit_index] == 0
            assert np.isnan(saved["coefficient_values"][fit_index]).all()
        for representation in range(2):
            hpc_index = (target, 1, representation, time, fold)
            assert saved["fit_status"][hpc_index] == "valid"
            assert saved["effective_feature_counts"][hpc_index] == 2


def test_wp4_save_rejects_cross_representation_regional_availability_mismatch(tmp_path):
    """PFC PCA/direct availability must agree with their combined counterparts.

    This payload makes only PFC PCA and combined PCA unavailable at every time
    bin, while PFC direct and combined direct remain usable. Each individual
    representation is internally time-consistent, isolating cross-
    representation regional availability as the rejected condition.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    for time in range(arrays["time_bin_centers_s"].size):
        for region in (0, 2):
            mark_fit_unavailable_without_features(
                arrays,
                (0, region, 0, time, 0),
                reason="pfc_pca_transform_unavailable",
            )

    with pytest.raises(ValueError, match="representation|PCA|direct|region|combined|availability"):
        results.save_task_decoding_run(
            tmp_path / "cross-representation-availability",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp4_fold_local_one_unit_region_pca_and_combined_masks_round_trip(tmp_path):
    """A one-unit PFC transform retains one PCA/direct feature in both regions.

    For all time bins in one target/fold, PFC direct and PFC PCA each retain
    their first of two mapped features. Both combined representations retain
    the matching PFC feature plus both usable HPC features. The resulting
    regional PCA width is exactly ``min(2 requested, 1 direct, observations)``
    and the combined width is three without imposing an invalid global PCA
    prefix across the PFC/HPC segment boundary.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target, fold = 0, 0
    for time in range(arrays["time_bin_centers_s"].size):
        for representation in range(2):
            pfc_index = (target, 0, representation, time, fold)
            combined_index = (target, 2, representation, time, fold)
            arrays["effective_feature_counts"][pfc_index] = 1
            arrays["effective_feature_counts"][combined_index] = 3
            arrays["coefficient_values"][pfc_index + (1,)] = np.nan
            arrays["coefficient_values"][combined_index + (1,)] = np.nan

    run_directory = tmp_path / "fold-local-one-unit-pfc"
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(run_directory)["arrays"]
    pfc_mask = np.array([True, False])
    combined_mask = np.array([True, False, True, True])
    for time in range(saved["time_bin_centers_s"].size):
        for representation in range(2):
            pfc_index = (target, 0, representation, time, fold)
            hpc_index = (target, 1, representation, time, fold)
            combined_index = (target, 2, representation, time, fold)
            assert saved["effective_feature_counts"][pfc_index] == 1
            assert saved["effective_feature_counts"][hpc_index] == 2
            assert saved["effective_feature_counts"][combined_index] == 3
            np.testing.assert_array_equal(
                np.isfinite(saved["coefficient_values"][pfc_index][:2]),
                pfc_mask,
            )
            np.testing.assert_array_equal(
                np.isfinite(saved["coefficient_values"][combined_index][:4]),
                combined_mask,
            )


def test_wp4_save_rejects_pca_effective_count_below_transform_limit(tmp_path):
    """A usable two-unit regional PCA must retain its exact two-PC transform width.

    PFC and the PFC segment of combined PCA use finite prefix masks at every
    time bin, so neither non-prefix nor direct/combined identity validation is
    implicated. Their effective widths of one and three are below the required
    ``min(requested_pcs, direct_width, train_trials * time_bins)`` limits of
    two and four, respectively.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    for time in range(arrays["time_bin_centers_s"].size):
        pfc_index = (0, 0, 0, time, 0)
        combined_index = (0, 2, 0, time, 0)
        arrays["effective_feature_counts"][pfc_index] = 1
        arrays["effective_feature_counts"][combined_index] = 3
        arrays["coefficient_values"][pfc_index + (1,)] = np.nan
        arrays["coefficient_values"][combined_index + (1,)] = np.nan

    with pytest.raises(ValueError, match="PCA|effective|requested|transform|component"):
        results.save_task_decoding_run(
            tmp_path / "pca-width-below-transform-limit",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp4_tuned_staggered_selected_direct_masks_must_agree_across_regions(tmp_path):
    """Selected direct transforms cannot imply incompatible PFC masks across times.

    PFC direct has a selected one-feature fit only at time zero. Combined
    direct has a selected full PFC segment only at time one. Their opposite
    cells are complete no-valid-candidate sentinels, so rejection depends on
    cross-selected-time evidence rather than simultaneous duplicate fits.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    pfc_time_zero = (0, 0, 1, 0, 0)
    combined_time_one = (0, 2, 1, 1, 0)
    arrays["effective_feature_counts"][pfc_time_zero] = 1
    arrays["coefficient_values"][pfc_time_zero + (1,)] = np.nan
    for time in range(arrays["time_bin_centers_s"].size):
        mark_tuned_cell_no_valid_candidate(arrays, (0, 0, 0, time, 0))
        mark_tuned_cell_no_valid_candidate(arrays, (0, 2, 0, time, 0))
        if time != 0:
            mark_tuned_cell_no_valid_candidate(arrays, (0, 0, 1, time, 0))
        if time != 1:
            mark_tuned_cell_no_valid_candidate(arrays, (0, 2, 1, time, 0))
    selected = arrays["selected_candidate_indices"][0, 0]
    assert np.flatnonzero(selected[0, 0] >= 0).tolist() == []
    assert np.flatnonzero(selected[2, 0] >= 0).tolist() == []
    assert np.flatnonzero(selected[0, 1] >= 0).tolist() == [0]
    assert np.flatnonzero(selected[2, 1] >= 0).tolist() == [1]

    with pytest.raises(ValueError, match="combined|direct|PFC|feature|selection|time"):
        results.save_task_decoding_run(
            tmp_path / "staggered-selected-direct-masks",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize(
    "evidence_case",
    (
        "standalone_pca_to_combined_direct",
        "combined_pca_to_standalone_direct",
        "combined_only_unavailable_pca",
    ),
)
def test_wp4_tuned_sparse_pca_and_direct_evidence_must_share_pfc_width(
    evidence_case,
    tmp_path,
):
    """Sparse tuned PFC PCA/direct evidence still constrains one shared width.

    The first two cases retain a selected PFC-bearing PCA cell and selected
    PFC-bearing direct-unit cell from complementary representations, while the
    opposite cells are no-candidate sentinels. The third leaves only selected
    combined representations, whose PCA transform is unavailable but direct
    features remain usable. Every case is incompatible with one PFC transform.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    time_count = arrays["time_bin_centers_s"].size
    all_times = list(range(time_count))
    for time in all_times:
        if evidence_case == "standalone_pca_to_combined_direct":
            mark_tuned_cell_no_valid_candidate(arrays, (0, 0, 1, time, 0))
            mark_tuned_cell_no_valid_candidate(arrays, (0, 2, 0, time, 0))
            pca_index = (0, 0, 0, time, 0)
            direct_index = (0, 2, 1, time, 0)
            arrays["effective_feature_counts"][pca_index] = 1
            arrays["coefficient_values"][pca_index + (1,)] = np.nan
        elif evidence_case == "combined_pca_to_standalone_direct":
            mark_tuned_cell_no_valid_candidate(arrays, (0, 0, 0, time, 0))
            mark_tuned_cell_no_valid_candidate(arrays, (0, 2, 1, time, 0))
            pca_index = (0, 2, 0, time, 0)
            direct_index = (0, 0, 1, time, 0)
            arrays["effective_feature_counts"][pca_index] = 3
            arrays["coefficient_values"][pca_index + (1,)] = np.nan
        else:
            mark_tuned_cell_no_valid_candidate(arrays, (0, 0, 0, time, 0))
            mark_tuned_cell_no_valid_candidate(arrays, (0, 0, 1, time, 0))
            pca_index = (0, 2, 0, time, 0)
            direct_index = (0, 2, 1, time, 0)
            mark_fit_unavailable_without_features(
                arrays,
                pca_index,
                reason="pfc_transform_unavailable",
            )
        assert arrays["effective_feature_counts"][direct_index] in (2, 4)

    selected = arrays["selected_candidate_indices"][0, 0]
    if evidence_case == "standalone_pca_to_combined_direct":
        assert np.flatnonzero(selected[0, 0] >= 0).tolist() == all_times
        assert np.flatnonzero(selected[0, 1] >= 0).tolist() == []
        assert np.flatnonzero(selected[2, 0] >= 0).tolist() == []
        assert np.flatnonzero(selected[2, 1] >= 0).tolist() == all_times
        np.testing.assert_array_equal(
            np.isfinite(arrays["coefficient_values"][0, 0, 0, 0, 0, :2]),
            np.array([True, False]),
        )
        np.testing.assert_array_equal(
            np.isfinite(arrays["coefficient_values"][0, 2, 1, 0, 0, :2]),
            np.array([True, True]),
        )
        assert arrays["effective_feature_counts"][0, 0, 0, 0, 0] == 1
        assert arrays["effective_feature_counts"][0, 2, 1, 0, 0] == 4
    elif evidence_case == "combined_pca_to_standalone_direct":
        assert np.flatnonzero(selected[0, 0] >= 0).tolist() == []
        assert np.flatnonzero(selected[0, 1] >= 0).tolist() == all_times
        assert np.flatnonzero(selected[2, 0] >= 0).tolist() == all_times
        assert np.flatnonzero(selected[2, 1] >= 0).tolist() == []
        np.testing.assert_array_equal(
            np.isfinite(arrays["coefficient_values"][0, 2, 0, 0, 0, :2]),
            np.array([True, False]),
        )
        np.testing.assert_array_equal(
            np.isfinite(arrays["coefficient_values"][0, 0, 1, 0, 0, :2]),
            np.array([True, True]),
        )
        assert arrays["effective_feature_counts"][0, 2, 0, 0, 0] == 3
        assert arrays["effective_feature_counts"][0, 0, 1, 0, 0] == 2
    else:
        assert np.flatnonzero(selected[0, 0] >= 0).tolist() == []
        assert np.flatnonzero(selected[0, 1] >= 0).tolist() == []
        assert np.flatnonzero(selected[2, 0] >= 0).tolist() == all_times
        assert np.flatnonzero(selected[2, 1] >= 0).tolist() == all_times
        pca_index = (0, 2, 0, 0, 0)
        direct_index = (0, 2, 1, 0, 0)
        assert arrays["fit_status"][pca_index] == "unavailable"
        assert arrays["effective_feature_counts"][pca_index] == 0
        assert np.isnan(arrays["coefficient_values"][pca_index]).all()
        assert arrays["fit_status"][direct_index] == "valid"
        assert arrays["effective_feature_counts"][direct_index] == 4

    with pytest.raises(ValueError, match="PCA|direct|combined|transform|effective|feature"):
        results.save_task_decoding_run(
            tmp_path / f"sparse-{evidence_case}",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_save_rejects_time_varying_direct_feature_dropout_within_target_fold(tmp_path):
    """One fitted direct-unit transform cannot change its width across time bins."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    pfc_index = (0, 0, 1, 0, 0)
    combined_index = (0, 2, 1, 0, 0)
    arrays["effective_feature_counts"][pfc_index] = 1
    arrays["effective_feature_counts"][combined_index] = 3
    arrays["coefficient_values"][pfc_index + (1,)] = np.nan
    arrays["coefficient_values"][combined_index + (1,)] = np.nan

    with pytest.raises(ValueError, match="time|feature|effective|transform|coefficient"):
        results.save_task_decoding_run(
            tmp_path / "time-varying-direct-dropout",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_save_rejects_direct_unit_subset_mismatch_with_combined_segment(tmp_path):
    """Standalone PFC unit dropout must match the PFC segment of combined units."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    for time in range(arrays["time_bin_centers_s"].size):
        pfc_index = (0, 0, 1, time, 0)
        arrays["effective_feature_counts"][pfc_index] = 1
        arrays["coefficient_values"][pfc_index + (1,)] = np.nan

    with pytest.raises(ValueError, match="combined|direct|unit|segment|feature"):
        results.save_task_decoding_run(
            tmp_path / "inconsistent-combined-direct-segment",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_save_rejects_nonprefix_pca_finite_coefficient_mask(tmp_path):
    """A retained second PCA component cannot coexist with a dropped first component."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}

    for time in range(arrays["time_bin_centers_s"].size):
        pfc_index = (0, 0, 0, time, 0)
        combined_index = (0, 2, 0, time, 0)
        arrays["effective_feature_counts"][pfc_index] = 1
        arrays["effective_feature_counts"][combined_index] = 3
        arrays["coefficient_values"][pfc_index + (0,)] = np.nan
        arrays["coefficient_values"][combined_index + (0,)] = np.nan
    fit_index = (0, 0, 0, 0, 0)
    assert np.array_equal(
        np.isfinite(arrays["coefficient_values"][fit_index][:2]),
        np.array([False, True]),
    )

    with pytest.raises(ValueError, match="PCA|prefix|component|feature|coefficient"):
        results.save_task_decoding_run(
            tmp_path / "nonprefix-pca-coefficient-mask",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize(
    "target_arrays",
    (
        {"unsafe": np.array([{"x": 1}], dtype=object)},
        {"checkpoint_meta": np.array("caller-owned", dtype="U16")},
    ),
)
def test_wp5a_checkpoint_rejects_object_arrays_and_reserved_metadata_key(target_arrays, tmp_path):
    """Checkpoint writers reject unsafe arrays and caller attempts to replace metadata."""
    checkpoint_path = tmp_path / "checkpoint.npz"
    with pytest.raises(ValueError, match="object|metadata|checkpoint"):
        results.save_target_checkpoint(
            checkpoint_path,
            target_label="current_action",
            target_arrays=target_arrays,
            full_run_fingerprint="checkpoint-validation",
        )
    assert not checkpoint_path.exists()


@pytest.mark.parametrize(
    "metadata",
    (
        None,
        "not-json",
        '{"full_run_fingerprint":"same"}',
        '{"target_label":"","full_run_fingerprint":"same"}',
        "[]",
        '{"target_label":7,"full_run_fingerprint":"same"}',
    ),
)
def test_wp5a_checkpoint_loader_rejects_missing_or_invalid_target_metadata(metadata, tmp_path):
    """Checkpoint loading requires a JSON object with a nonempty string target label."""
    checkpoint_path = tmp_path / "checkpoint.npz"
    members = {"scores": np.array([0.5], dtype=np.float64)}
    if metadata is not None:
        members["checkpoint_meta"] = np.array(metadata, dtype="U128")
    np.savez(checkpoint_path, **members)

    with pytest.raises(ValueError, match="metadata|target|checkpoint"):
        results.load_target_checkpoint(
            checkpoint_path,
            expected_full_run_fingerprint="same",
        )


def test_wp5a_tuned_selected_candidate_survives_outer_feature_or_later_fit_failure(tmp_path):
    """Completed inner selection persists when later outer work cannot yield a score.

    A selected candidate persists if later estimator fitting or scoring fails
    after a complete regional transform with its unchanged mapped width.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    fit_index = (0, 0, 1, 0, 0)
    reason = "outer_estimator_failure"
    effective = int(arrays["effective_feature_counts"][fit_index])
    assert effective == 2
    arrays["fit_status"][fit_index] = "unavailable"
    arrays["fit_reason_codes"][fit_index] = reason
    arrays["fold_scores"][0, 0, 1, :, 0, 0] = np.nan
    arrays["coefficient_values"][fit_index] = np.nan
    arrays["fitted_intercepts"][fit_index] = np.nan
    selection_index = (0, 0, 0, 1, 0)
    selected_index = int(arrays["selected_candidate_indices"][selection_index])
    selected_parameters = arrays["selected_parameters_json"][selection_index]
    assert selected_index >= 0
    assert selected_parameters != ""

    run_directory = tmp_path / "outer-failure-after-selection"
    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(run_directory)["arrays"]
    assert saved["fit_status"][fit_index] == "unavailable"
    assert saved["fit_reason_codes"][fit_index] == reason
    assert saved["effective_feature_counts"][fit_index] == effective
    assert saved["selected_candidate_indices"][selection_index] == selected_index
    assert saved["selected_parameters_json"][selection_index] == selected_parameters
    assert np.isnan(saved["fold_scores"][0, 0, 1, :, 0, 0]).all()
    assert np.isnan(saved["coefficient_values"][fit_index]).all()
    assert np.isnan(saved["fitted_intercepts"][fit_index])


def test_wp5a_tuned_inner_class_coverage_unavailability_round_trips(tmp_path):
    """Pure class blocks make every categorical three-fold inner plan unavailable.

    Six eligible behavioral blocks provide both classes to every outer test
    split and four outer-training groups (two per class). Three inner folds
    cannot validate each pure block with both classes, so every inner plan is
    declared unavailable rather than serialized with partial assignments.
    """
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target = 0
    outer, blocks, trial_rows, group_values = assign_pure_categorical_values_by_outer_group(
        arrays,
        target=target,
    )
    refresh_categorical_outer_class_counts(arrays, target=target)
    eligible_common_rows = np.flatnonzero(outer >= 0)
    eligible_values = arrays["encoded_target_values"][target, trial_rows[eligible_common_rows]]
    eligible_blocks = np.asarray(
        [blocks[int(trial_rows[row])] for row in eligible_common_rows],
        dtype=object,
    )
    assert len(group_values) == 6
    assert np.isnan(arrays["encoded_target_values"][target, 11])
    for outer_fold in arrays["fold_labels"]:
        test_rows = np.flatnonzero(outer == outer_fold)
        train_rows = np.flatnonzero((outer >= 0) & (outer != outer_fold))
        assert set(arrays["encoded_target_values"][target, trial_rows[test_rows]]) == {0.0, 1.0}
        train_groups = {blocks[int(trial_rows[row])] for row in train_rows}
        assert len(train_groups) == 4
        assert [
            sum(group_values[group] == class_value for group in train_groups)
            for class_value in (0.0, 1.0)
        ] == [2, 2]
        eligible_outer = outer[eligible_common_rows]
        plan = modeling.make_inner_splits(
            eligible_values,
            eligible_blocks,
            np.flatnonzero(eligible_outer != outer_fold),
            target_family="categorical",
            fold_count=3,
        )
        assert not plan.is_available
        assert plan.unavailable_reason == "categorical grouped fold lacks both classes"
        assert np.all(plan.fold_ids == -1)

    reason = "inner_class_coverage_unavailable"
    arrays["inner_selection_fold_ids"][target].fill(-1)
    arrays["fit_status"][target].fill("unavailable")
    arrays["fit_reason_codes"][target].fill(reason)
    arrays["effective_feature_counts"][target].fill(0)
    arrays["fold_scores"][target].fill(np.nan)
    arrays["coefficient_values"][target].fill(np.nan)
    arrays["fitted_intercepts"][target].fill(np.nan)
    arrays["candidate_inner_scores"][target].fill(np.nan)
    arrays["candidate_inner_statuses"][target].fill("invalid")
    arrays["candidate_inner_reasons"][target].fill(reason)
    arrays["selected_candidate_indices"][target].fill(-1)
    arrays["selected_parameters_json"][target].fill("")
    assert_grouped_split_integrity(arrays)

    results.save_task_decoding_run(
        tmp_path / "inner-class-coverage-unavailable",
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(tmp_path / "inner-class-coverage-unavailable")["arrays"]
    assert np.all(saved["inner_selection_fold_ids"][target] == -1)
    assert set(saved["candidate_inner_statuses"][target].flat) == {"invalid"}
    assert set(saved["candidate_inner_reasons"][target].flat) == {reason}
    assert np.all(saved["selected_candidate_indices"][target] == -1)
    assert not np.any(saved["selected_parameters_json"][target] != "")


def test_wp5a_tuned_inner_plan_rejects_finite_valid_audit_when_plan_is_unavailable(tmp_path):
    """An impossible grouped inner plan cannot contain any finite valid audit result."""
    input_paths = write_input_fixture(tmp_path)
    arrays, meta, scientific_config = make_inner_plan_unavailable_tuned_payload(input_paths)
    arrays = {name: value.copy() for name, value in arrays.items()}
    arrays["candidate_inner_scores"][0, 0, 0, 0, 0, 0, 0] = 0.75
    arrays["candidate_inner_statuses"][0, 0, 0, 0, 0, 0, 0] = "valid"
    arrays["candidate_inner_reasons"][0, 0, 0, 0, 0, 0, 0] = ""

    with pytest.raises(ValueError, match="inner|candidate|unavailable|audit"):
        results.save_task_decoding_run(
            tmp_path / "invalid-inner-audit",
            arrays=arrays,
            meta=meta,
            input_manifest=build_manifest(input_paths),
            scientific_config=scientific_config,
            feature_parameter_source=input_paths["feature_parameters"],
            run_fingerprint="inner-plan-unavailable",
        )


def test_wp5a_categorical_outer_partition_requires_both_classes(tmp_path):
    """A globally binary, block-safe categorical outer fold requires both classes."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target = 0
    outer = arrays["outer_fold_ids"][target]
    trial_rows = arrays["trial_row_indices"]
    for common_row in np.flatnonzero(outer >= 0):
        full_row = int(trial_rows[common_row])
        arrays["encoded_target_values"][target, full_row] = (
            0.0 if outer[common_row] == 0 else 1.0
        )
    refresh_categorical_outer_class_counts(arrays, target=target)
    values = arrays["encoded_target_values"][target, trial_rows]
    fold_zero_test = outer == 0
    fold_zero_train = (outer >= 0) & (outer != 0)
    assert set(values[outer >= 0]) == {0.0, 1.0}
    assert np.all(values[fold_zero_test] == 0.0)
    assert np.all(values[fold_zero_train] == 1.0)
    assert np.isnan(arrays["encoded_target_values"][target, 11])
    assert_grouped_split_integrity(arrays)

    with pytest.raises(ValueError, match="class|categorical|coverage|fold"):
        results.save_task_decoding_run(
            tmp_path / "single-class-block-safe-folds",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_categorical_inner_partition_rejects_single_class_validation(tmp_path):
    """A block-safe three-group inner partition fails if one validation fold has one class."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="tuned")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    target = 0
    outer_fold = 0
    outer, blocks, trial_rows, group_values = assign_pure_categorical_values_by_outer_group(
        arrays,
        target=target,
    )
    refresh_categorical_outer_class_counts(arrays, target=target)

    train_rows = np.flatnonzero((outer >= 0) & (outer != outer_fold))
    train_groups = {blocks[int(trial_rows[row])] for row in train_rows}
    zero_groups = [group for group in train_groups if group_values[group] == 0.0]
    assert len(train_groups) == 4
    assert len(zero_groups) >= 2
    inner = arrays["inner_selection_fold_ids"][target, outer_fold]
    inner.fill(-1)
    for row in train_rows:
        group = blocks[int(trial_rows[row])]
        inner[row] = 0 if group == zero_groups[0] else (2 if group == zero_groups[1] else 1)
    assert set(inner[train_rows]) == {0, 1, 2}
    values = arrays["encoded_target_values"][target, trial_rows]
    assert np.isnan(arrays["encoded_target_values"][target, 11])
    validation_rows = inner == 0
    assert np.all(values[validation_rows] == 0.0)
    assert_grouped_split_integrity(arrays)

    with pytest.raises(ValueError, match="inner|class|categorical|coverage|fold"):
        results.save_task_decoding_run(
            tmp_path / "inner-single-class-validation",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


@pytest.mark.parametrize(
    "kind",
    ("pfc_requested", "hpc_requested", "combined_requested", "combined_pca", "combined_units"),
)
def test_wp5a_save_rejects_config_or_combined_feature_identity_mismatch(kind, tmp_path):
    """Configured PCA counts and combined feature maps are exact region-ordered contracts."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    if kind == "pfc_requested":
        arrays["requested_feature_counts"][:, 0, 0] = 1
    elif kind == "hpc_requested":
        arrays["requested_feature_counts"][:, 1, 0] = 3
    elif kind == "combined_requested":
        arrays["requested_feature_counts"][:, 2, 0] = 3
    elif kind == "combined_pca":
        arrays["coefficient_feature_ids"][2, 0, :2] = ("PFC:PC1", "HPC:PC2")
    else:
        arrays["coefficient_feature_ids"][2, 1, 2] = "probe-hpc:unexpected"

    with pytest.raises(ValueError, match="PCA|feature|combined|requested|identity|region"):
        results.save_task_decoding_run(
            tmp_path / f"bad-config-feature-{kind}",
            arrays=arrays,
            **{key: value for key, value in arguments.items() if key != "arrays"},
        )


def test_wp5a_fixed_post_transform_fit_failure_retains_effective_feature_count(tmp_path):
    """Fixed-mode post-transform failure preserves nonzero effective features but no outputs."""
    input_paths = write_input_fixture(tmp_path)
    arguments = make_run_save_arguments(input_paths, regularization_mode="fixed")
    arrays = {name: value.copy() for name, value in arguments["arrays"].items()}
    fit_index = (0, 0, 1, 0, 0)
    arrays["fit_status"][fit_index] = "unavailable"
    arrays["fit_reason_codes"][fit_index] = "fit_convergence_failure"
    assert arrays["effective_feature_counts"][fit_index] == 2
    arrays["fold_scores"][0, 0, 1, :, 0, 0] = np.nan
    arrays["coefficient_values"][fit_index] = np.nan
    arrays["fitted_intercepts"][fit_index] = np.nan
    run_directory = tmp_path / "fixed-post-transform-failure"

    results.save_task_decoding_run(
        run_directory,
        arrays=arrays,
        **{key: value for key, value in arguments.items() if key != "arrays"},
    )
    saved = results.load_task_decoding_run(run_directory)["arrays"]
    assert saved["fit_status"][fit_index] == "unavailable"
    assert saved["fit_reason_codes"][fit_index] == "fit_convergence_failure"
    assert saved["effective_feature_counts"][fit_index] == 2
    assert np.isnan(saved["fold_scores"][0, 0, 1, :, 0, 0]).all()
    assert np.isnan(saved["coefficient_values"][fit_index]).all()
    assert np.isnan(saved["fitted_intercepts"][fit_index])


def test_wp5a_round_trips_a_second_canonical_categorical_target(tmp_path):
    """Target metadata is derived from shared definitions rather than current-action hardcoding."""
    input_paths = write_input_fixture(tmp_path)
    replacement_target = "current_action_is_correct"
    config = replace(
        make_config(input_paths, regularization_mode="fixed"),
        target_names=(replacement_target, "relative_doubt"),
    )
    scientific_config = scientific_config_payload(config)
    arrays = make_result_arrays(scientific_config, regularization_mode="fixed")
    arrays["target_labels"][0] = replacement_target
    arrays["target_source_labels"][0] = "correct"
    arrays["target_label_mappings_json"][0] = json.dumps(
        targets._CATEGORICAL_CLASS_LABELS[replacement_target],
        sort_keys=True,
        separators=(",", ":"),
    )
    meta = make_meta(arrays, scientific_config, regularization_mode="fixed")
    for field in (
        "encoded_target_values",
        "fold_scores",
        "candidate_inner_scores",
        "coefficient_values",
        "fitted_intercepts",
    ):
        values = meta["units"][field].pop("current_action")
        meta["units"][field][replacement_target] = values
    meta["parameters"]["candidate_selection_metrics"] = {
        replacement_target: "balanced_accuracy",
        "relative_doubt": "r2",
    }
    results.save_task_decoding_run(
        tmp_path / "second-categorical-target",
        arrays=arrays,
        meta=meta,
        input_manifest=build_manifest(input_paths),
        scientific_config=scientific_config,
        feature_parameter_source=input_paths["feature_parameters"],
        run_fingerprint="run-fingerprint-test",
    )
    saved = results.load_task_decoding_run(tmp_path / "second-categorical-target")["arrays"]
    assert saved["target_labels"][0] == replacement_target
    assert saved["target_source_labels"][0] == "correct"
    assert saved["target_label_mappings_json"][0] == arrays["target_label_mappings_json"][0]
