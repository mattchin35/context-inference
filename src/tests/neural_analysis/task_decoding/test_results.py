"""WP5A RED contracts for task-decoding saved results and scientific identity.

This module intentionally covers only ``results.py``: primitive schema/NPZ,
input and scoped-source identity, complete-target checkpoints, and atomic
publication. Pipeline state, CLI, logs, summaries, plots, discovery, and raw
neural loading remain WP5B work.
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence

import numpy as np
import pytest

from src.neural_analysis.task_decoding import results
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
_SCORE_AXES = ("target", "region", "representation", "metric", "time", "fold")
_FIT_AXES = ("target", "region", "representation", "time", "fold")
_BLOCK_VALUES = (
    100,
    100,
    "two",
    "two",
    200,
    200,
    "four",
    "four",
    300,
    300,
    "singleton-shared-label-longer-than-thirty-two-characters",
    "target-one-only",
    "target-zero-only",
    "not-common",
    "not-common",
)
_BLOCK_ID_JSON_BYTES = tuple(
    json.dumps(value, separators=(",", ":")).encode("utf-8") for value in _BLOCK_VALUES
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
        bin_width_ms=100,
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
    metric_count, time_count, fold_count, feature_capacity = 3, 2, 3, 4
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
        "target_source_labels": np.array(["action", "relative_doubt_index"], dtype="U32"),
        "target_positive_classes": np.array([1, -1], dtype=np.int64),
        "target_label_mappings_json": np.array(
            ['{"0":"right","1":"left"}', '{"native_target":true}'], dtype="U48"
        ),
        "region_labels": np.array(_REGIONS, dtype="U16"),
        "representation_labels": np.array(_REPRESENTATIONS, dtype="U16"),
        "metric_labels": np.array(_METRICS, dtype="U32"),
        "time_bin_edges_s": np.array([-0.2, -0.1, 0.0], dtype=np.float64),
        "time_bin_centers_s": np.array([-0.15, -0.05], dtype=np.float64),
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
        ("PFC", "pca"): (("PFC:PC1",), ("PFC",)),
        ("PFC", "units"): (("probe-pfc:11", "probe-pfc:19"), ("PFC", "PFC")),
        ("HPC", "pca"): (("HPC:PC1",), ("HPC",)),
        ("HPC", "units"): (("probe-hpc:5", "probe-hpc:17"), ("HPC", "HPC")),
        ("PFC+HPC", "pca"): (("PFC:PC1", "HPC:PC1"), ("PFC", "HPC")),
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
    numerical_fixed = json.dumps({"alpha": 0.1, "l1_ratio": 0.5}, sort_keys=True)
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
            "bin_width_ms": 100,
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
    meta = make_meta(arrays, scientific_config, regularization_mode=regularization_mode)
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
    meta = make_meta(arrays, scientific_config, regularization_mode=regularization_mode)
    return {
        "arrays": arrays,
        "meta": meta,
        "input_manifest": build_manifest(input_paths),
        "scientific_config": scientific_config,
        "feature_parameter_source": input_paths["feature_parameters"],
        "run_fingerprint": run_fingerprint,
    }


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
    fit_shape = (2, 3, 2, 2, 3)
    score_shape = (2, 3, 2, 3, 2, 3)
    parameter_shape = (2, 3, 3, 2, 2)
    feature_shape = (3, 2, 4)
    contract = {
        "target_labels": ((2,), np.dtype("U32")),
        "target_families": ((2,), np.dtype("U16")),
        "target_source_labels": ((2,), np.dtype("U32")),
        "target_positive_classes": ((2,), np.dtype(np.int64)),
        "target_label_mappings_json": ((2,), np.dtype("U48")),
        "region_labels": ((3,), np.dtype("U16")),
        "representation_labels": ((2,), np.dtype("U16")),
        "metric_labels": ((3,), np.dtype("U32")),
        "time_bin_edges_s": ((3,), np.dtype(np.float64)),
        "time_bin_centers_s": ((2,), np.dtype(np.float64)),
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
        audit_shape = (2, 3, 3, 2, 2, 15, 3)
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
    assert arrays["region_labels"].tolist() == list(_REGIONS)
    assert arrays["representation_labels"].tolist() == list(_REPRESENTATIONS)
    assert arrays["metric_labels"].tolist() == list(_METRICS)
    assert arrays["metric_labels"].dtype.kind == "U"
    assert arrays["fold_scores"].shape == (2, 3, 2, 3, 2, 3)
    assert arrays["fold_scores"].dtype == np.dtype(np.float64)
    assert arrays["fit_status"].shape == (2, 3, 2, 2, 3)
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
    expected_widths = np.array([[1, 2], [1, 2], [2, 4]], dtype=np.int64)
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
    assert arrays["coefficient_feature_ids"][0, 0, 0] == "PFC:PC1"
    assert arrays["coefficient_feature_ids"][0, 1, :2].tolist() == ["probe-pfc:11", "probe-pfc:19"]
    assert arrays["coefficient_feature_ids"][2, 0, :2].tolist() == ["PFC:PC1", "HPC:PC1"]
    assert arrays["coefficient_feature_ids"][2, 1].tolist() == [
        "probe-pfc:11",
        "probe-pfc:19",
        "probe-hpc:5",
        "probe-hpc:17",
    ]
    assert arrays["fitted_intercepts"].shape == arrays["fit_status"].shape
    assert arrays["fixed_parameters_json"].shape == (2, 3, 3, 2, 2)
    assert '"C"' in arrays["fixed_parameters_json"][0, 0, 0, 0, 0]
    assert '"alpha"' in arrays["fixed_parameters_json"][1, 0, 0, 0, 0]
    assert arrays["stage_timing_labels"].shape == arrays["stage_timing_seconds"].shape
    assert arrays["total_timing_seconds"].ndim == 0
    assert meta["analysis_version"] == ANALYSIS_VERSION
    assert meta["parameters"]["regularization_mode"] == regularization_mode
    assert meta["parameters"]["bin_width_ms"] == scientific_config["bin_width_ms"] == 100
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
        expected_audit_shape = (2, 3, 3, 2, 2, 15, 3)
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
            for indexes in np.ndindex(3, 3, 2, 2):
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
        altered_arrays["coefficient_values"][0, 0, 0, 0, 0, 1] = 0.0
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
    ),
)
def test_preparation_cleanliness_handles_package_untracked_missing_and_unrelated_state(
    status_output,
    should_raise,
    tmp_path,
    monkeypatch,
):
    """Relevant package dirtiness or untracked Python blocks; unrelated docs do not."""
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
