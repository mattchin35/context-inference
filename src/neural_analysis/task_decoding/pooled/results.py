"""Safe schema-1 storage and scientific identity for pooled decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from src.neural_analysis.task_decoding import modeling
from src.neural_analysis.task_decoding import results as base_results
from src.neural_analysis.task_decoding.pooled import decoding
from src.neural_analysis.task_decoding.pooled.config import ANALYSIS_VERSION, TrainingWindow


POOLED_RESULT_SCHEMA_VERSION = 1
STATUS_NAMES = ("not_applicable", "valid", "unavailable")
CONVERGENCE_STATUS_NAMES = ("not_run", "converged", "warning", "invalid")
REGION_NAMES = ("PFC", "HPC", "PFC+HPC")
SOURCE_REGION_NAMES = ("PFC", "HPC")
REPRESENTATION_NAMES = ("pca", "units")
METRIC_NAMES = ("balanced_accuracy", "auc", "r2")
POOLED_SCIENTIFIC_SOURCE_PATHS = (
    "src/neural_analysis/task_decoding/pooled/__init__.py",
    "src/neural_analysis/task_decoding/pooled/config.py",
    "src/neural_analysis/task_decoding/pooled/decoding.py",
    "src/neural_analysis/task_decoding/pooled/results.py",
    "src/neural_analysis/task_decoding/pooled/pipeline.py",
    "src/neural_analysis/task_decoding/__init__.py",
    "src/neural_analysis/task_decoding/activity.py",
    "src/neural_analysis/task_decoding/conditions.py",
    "src/neural_analysis/task_decoding/config.py",
    "src/neural_analysis/task_decoding/modeling.py",
    "src/neural_analysis/task_decoding/results.py",
    "src/neural_analysis/task_decoding/targets.py",
    "src/neural_analysis/spike_behavior/trials.py",
    *base_results._DIRECT_RUNTIME_PATHS,
)


_AXES = {
    "training_window_names": ("window",),
    "condition_names": ("condition",),
    "target_names": ("target",),
    "fold_ids": ("fold",),
    "region_names": ("region",),
    "source_region_names": ("source_region",),
    "representation_names": ("representation",),
    "evaluation_time_centers_s": ("time",),
    "metric_names": ("metric",),
    "evaluation_train_condition": ("pair",),
    "evaluation_test_condition": ("pair",),
    "status_names": ("status",),
    "reason_code_names": ("reason",),
    "convergence_status_names": ("convergence_status",),
    "target_values": ("target", "trial"),
    "target_eligible": ("target", "trial"),
    "master_eligible": ("target", "trial"),
    "eligibility_reason_index": ("target", "trial"),
    "condition_masks": ("condition", "trial"),
    "trial_row_indices": ("tensor_trial",),
    "outer_fold_ids": ("target", "tensor_trial"),
    "training_window_bin_mask": ("window", "time"),
    "transform_status": ("window", "condition", "target", "fold", "source_region"),
    "transform_reason_index": ("window", "condition", "target", "fold", "source_region"),
    "transform_input_unit_count": ("window", "condition", "target", "fold", "source_region"),
    "transform_usable_unit_count": ("window", "condition", "target", "fold", "source_region"),
    "transform_component_limit": ("window", "condition", "target", "fold", "source_region"),
    "transform_requested_pc_count": ("window", "condition", "target", "fold", "source_region"),
    "transform_effective_pc_count": ("window", "condition", "target", "fold", "source_region"),
    "model_status": ("window", "condition", "target", "fold", "region", "representation"),
    "model_reason_index": ("window", "condition", "target", "fold", "region", "representation"),
    "model_train_trial_count": ("window", "condition", "target", "fold", "region", "representation"),
    "model_stacked_row_count": ("window", "condition", "target", "fold", "region", "representation"),
    "model_requested_feature_count": ("window", "condition", "target", "fold", "region", "representation"),
    "model_feature_count": ("window", "condition", "target", "fold", "region", "representation"),
    "model_convergence_status": ("window", "condition", "target", "fold", "region", "representation"),
    "model_intercept": ("window", "condition", "target", "fold", "region", "representation"),
    "model_coefficients": ("window", "condition", "target", "fold", "region", "representation", "feature"),
    "model_feature_active": ("window", "condition", "target", "fold", "region", "representation", "feature"),
    "evaluation_status": ("window", "pair", "target", "fold", "region", "representation", "time", "metric"),
    "evaluation_reason_index": ("window", "pair", "target", "fold", "region", "representation", "time", "metric"),
    "evaluation_scores": ("window", "pair", "target", "fold", "region", "representation", "time", "metric"),
    "evaluation_test_trial_count": ("window", "pair", "target", "fold"),
    "gap_status": ("window", "condition", "target", "fold", "region", "representation", "time", "metric"),
    "gap_reason_index": ("window", "condition", "target", "fold", "region", "representation", "time", "metric"),
    "gap_scores": ("window", "condition", "target", "fold", "region", "representation", "time", "metric"),
    "time_bin_edges_s": ("time_edge",),
    "training_window_start_s": ("window",),
    "training_window_stop_s": ("window",),
    "trial_positions": ("trial",),
    "trial_ids": ("trial",),
    "behavioral_block_payload": ("block_id_byte",),
    "behavioral_block_offsets": ("trial_offset",),
    "target_family_names": ("target",),
    "target_unit_names": ("target",),
    "target_positive_class": ("target",),
    "coefficient_feature_ids": ("region", "representation", "feature"),
    "coefficient_feature_regions": ("region", "representation", "feature"),
    "coefficient_feature_present": ("region", "representation", "feature"),
    "unit_selection_rules_json": ("source_region",),
    "model_train_class_counts": ("window", "condition", "target", "fold", "class"),
    "evaluation_test_class_counts": ("window", "pair", "target", "fold", "class"),
    "target_estimator_class_names": ("target",),
    "target_estimator_parameters_json": ("target",),
}


def _canonical_json(value: object) -> str:
    """Encode JSON-compatible values with deterministic ASCII-safe structure.

    Parameters
    ----------
    value : object
        JSON-compatible mapping, sequence, or scalar.

    Returns
    -------
    str
        Canonical UTF-8 JSON text with sorted mapping keys.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _unicode_array(values: Sequence[str]) -> np.ndarray:
    """Create a lossless one-dimensional Unicode coordinate array.

    Parameters
    ----------
    values : sequence[str]
        Ordered coordinate or canonical-JSON strings.

    Returns
    -------
    numpy.ndarray
        Unicode array shaped ``(len(values),)`` with content-derived width.
    """
    normalized = tuple(str(value) for value in values)
    width = max(1, *(len(value) for value in normalized))
    return np.asarray(normalized, dtype=f"U{width}")


def _json_scalar(value: object) -> object:
    """Convert one NumPy scalar to its JSON-native equivalent.

    Parameters
    ----------
    value : object
        Behavioral block identity scalar.

    Returns
    -------
    object
        JSON-compatible scalar retaining exact logical identity.
    """
    return value.item() if isinstance(value, np.generic) else value


def _encode_block_ids(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack exact canonical behavioral-block identities into safe byte arrays.

    Parameters
    ----------
    values : numpy.ndarray
        One-dimensional trial-aligned block identities.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Concatenated UTF-8 bytes shaped ``(B,)`` and int64 offsets shaped
        ``(trial + 1,)``.
    """
    blocks = np.asarray(values)
    if blocks.ndim != 1 or blocks.size == 0:
        raise ValueError("behavioral_block_ids must be a nonempty one-dimensional array.")
    encoded = [
        _canonical_json(_json_scalar(value)).encode("utf-8")
        for value in blocks.tolist()
    ]
    offsets = np.zeros(blocks.size + 1, dtype=np.int64)
    for index, item in enumerate(encoded):
        offsets[index + 1] = offsets[index] + len(item)
    payload = np.frombuffer(b"".join(encoded), dtype=np.uint8).copy()
    return payload, offsets


def _feature_coordinates(
    pfc_unit_ids: tuple[str, ...],
    hpc_unit_ids: tuple[str, ...],
    pfc_pc_count: int,
    hpc_pc_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded stable feature identities for every region/representation.

    Parameters
    ----------
    pfc_unit_ids, hpc_unit_ids : tuple[str, ...]
        Stable direct-unit identities.
    pfc_pc_count, hpc_pc_count : int
        Positive requested anatomical PC counts.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Unicode IDs, anatomical regions, and Boolean presence arrays with
        shape ``(region=3, representation=2, feature=G)``.
    """
    feature_sets = {
        ("PFC", "pca"): tuple(f"PFC:PC{i + 1}" for i in range(pfc_pc_count)),
        ("HPC", "pca"): tuple(f"HPC:PC{i + 1}" for i in range(hpc_pc_count)),
        ("PFC+HPC", "pca"): (
            *(f"PFC:PC{i + 1}" for i in range(pfc_pc_count)),
            *(f"HPC:PC{i + 1}" for i in range(hpc_pc_count)),
        ),
        ("PFC", "units"): pfc_unit_ids,
        ("HPC", "units"): hpc_unit_ids,
        ("PFC+HPC", "units"): (*pfc_unit_ids, *hpc_unit_ids),
    }
    capacity = max(1, *(len(values) for values in feature_sets.values()))
    width = max(1, *(len(value) for values in feature_sets.values() for value in values))
    identifiers = np.full((3, 2, capacity), "", dtype=f"U{width}")
    regions = np.full((3, 2, capacity), "", dtype="U3")
    present = np.zeros((3, 2, capacity), dtype=bool)
    for region_index, region_name in enumerate(REGION_NAMES):
        for representation_index, representation in enumerate(REPRESENTATION_NAMES):
            values = feature_sets[(region_name, representation)]
            for feature_index, identifier in enumerate(values):
                identifiers[region_index, representation_index, feature_index] = identifier
                regions[region_index, representation_index, feature_index] = identifier.split(":")[0]
                present[region_index, representation_index, feature_index] = True
    return identifiers, regions, present


def initialize_pooled_result_arrays(
    *,
    training_windows: Sequence[TrainingWindow],
    condition_names: Sequence[str],
    target_names: Sequence[str],
    target_family_names: Sequence[str],
    target_unit_names: Sequence[str],
    outer_fold_count: int,
    evaluation_time_centers_s: np.ndarray,
    time_bin_edges_s: np.ndarray,
    trial_positions: np.ndarray,
    trial_ids: np.ndarray,
    behavioral_block_ids: np.ndarray,
    trial_row_indices: np.ndarray,
    pfc_unit_ids: Sequence[str],
    hpc_unit_ids: Sequence[str],
    pfc_pc_count: int,
    hpc_pc_count: int,
    unit_selection_rules: Sequence[Mapping[str, object]],
    target_estimator_class_names: Sequence[str],
    target_estimator_parameters: Sequence[Mapping[str, object]],
    run_fingerprint: str,
    source_fingerprint: str,
    random_seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Allocate one exact complete schema initialized as outer-unavailable.

    Parameters
    ----------
    training_windows : sequence[TrainingWindow]
        Nonempty canonical windows on the evaluation-time axis.
    condition_names, target_names : sequence[str]
        Nonempty unique condition and target coordinates.
    target_family_names, target_unit_names : sequence[str]
        Target-aligned response families and native/dimensionless unit labels.
    outer_fold_count : int
        Positive requested grouped fold count.
    evaluation_time_centers_s, time_bin_edges_s : numpy.ndarray
        Choice-relative seconds shaped ``(Q,)`` and ``(Q+1,)``.
    trial_positions, trial_ids, behavioral_block_ids : numpy.ndarray
        Full chronological trial arrays shaped ``(N,)``.
    trial_row_indices : numpy.ndarray
        Common bilateral tensor-row mapping shaped ``(M,)``.
    pfc_unit_ids, hpc_unit_ids : sequence[str]
        Stable selected source-unit identities.
    pfc_pc_count, hpc_pc_count : int
        Positive requested regional component counts.
    unit_selection_rules : sequence[mapping[str, object]]
        Exactly two JSON-safe PFC/HPC selection-rule records.
    target_estimator_class_names : sequence[str]
        Target-aligned concrete fixed estimator classes.
    target_estimator_parameters : sequence[mapping[str, object]]
        Target-aligned fixed-control mappings.
    run_fingerprint, source_fingerprint : str
        Nonempty complete run and scientific-source identities.
    random_seed : int
        Fixed dimensionless estimator/PCA random seed.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, object]]
        Exact primitive arrays and canonical JSON-safe metadata. Requested
        cells begin unavailable with ``target_outer_unavailable`` and are
        overwritten only by completed checkpoints.
    """
    windows = tuple(training_windows)
    conditions = tuple(condition_names)
    targets = tuple(target_names)
    families = tuple(target_family_names)
    target_units = tuple(target_unit_names)
    if not windows or not conditions or not targets:
        raise ValueError("Window, condition, and target coordinates must be nonempty.")
    if "correct_rewarded" not in conditions:
        raise ValueError("condition_names must include correct_rewarded.")
    if len(set(conditions)) != len(conditions) or len(set(targets)) != len(targets):
        raise ValueError("Condition and target coordinates must be unique.")
    if len(families) != len(targets) or len(target_units) != len(targets):
        raise ValueError("Target family and unit coordinates must align with targets.")
    if any(family not in {"categorical", "numerical"} for family in families):
        raise ValueError("Target families must be categorical or numerical.")
    centers = np.asarray(evaluation_time_centers_s, dtype=np.float64)
    edges = np.asarray(time_bin_edges_s, dtype=np.float64)
    if (
        centers.ndim != 1
        or centers.size == 0
        or edges.shape != (centers.size + 1,)
        or not np.isfinite(centers).all()
        or not np.isfinite(edges).all()
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("Time centers and edges must be finite aligned axes in seconds.")
    positions = np.asarray(trial_positions)
    identifiers = np.asarray(trial_ids)
    tensor_rows = np.asarray(trial_row_indices)
    if (
        positions.dtype != np.int64
        or identifiers.dtype != np.int64
        or tensor_rows.dtype != np.int64
        or positions.ndim != 1
        or identifiers.shape != positions.shape
        or np.asarray(behavioral_block_ids).shape != positions.shape
        or tensor_rows.ndim != 1
        or tensor_rows.size == 0
        or (tensor_rows < 0).any()
        or (tensor_rows >= positions.size).any()
    ):
        raise ValueError("Trial identities and common tensor rows are invalid.")
    pfc_ids = tuple(pfc_unit_ids)
    hpc_ids = tuple(hpc_unit_ids)
    if not pfc_ids or not hpc_ids or not all(isinstance(value, str) and value for value in (*pfc_ids, *hpc_ids)):
        raise ValueError("Both regional unit coordinates must contain stable IDs.")
    if not isinstance(outer_fold_count, int) or outer_fold_count < 1:
        raise ValueError("outer_fold_count must be positive.")
    feature_ids, feature_regions, feature_present = _feature_coordinates(
        pfc_ids,
        hpc_ids,
        pfc_pc_count,
        hpc_pc_count,
    )
    payload, offsets = _encode_block_ids(np.asarray(behavioral_block_ids))
    pair_train = ["correct_rewarded" for _ in conditions]
    pair_test = list(conditions)
    for condition in conditions:
        if condition != "correct_rewarded":
            pair_train.append(condition)
            pair_test.append(condition)
    W, C, T, F = len(windows), len(conditions), len(targets), outer_fold_count
    R, A, P, Q, K, E = 3, 2, 2, centers.size, 3, len(pair_train)
    N, M, G = positions.size, tensor_rows.size, feature_ids.shape[2]
    outer_reason = decoding.POOLED_REASON_CODES.index("target_outer_unavailable")
    target_reason = decoding.POOLED_REASON_CODES.index("target_ineligible")

    arrays: dict[str, np.ndarray] = {
        "training_window_names": _unicode_array([window.name for window in windows]),
        "condition_names": _unicode_array(conditions),
        "target_names": _unicode_array(targets),
        "fold_ids": np.arange(F, dtype=np.int16),
        "region_names": _unicode_array(REGION_NAMES),
        "source_region_names": _unicode_array(SOURCE_REGION_NAMES),
        "representation_names": _unicode_array(REPRESENTATION_NAMES),
        "evaluation_time_centers_s": centers.copy(),
        "metric_names": _unicode_array(METRIC_NAMES),
        "evaluation_train_condition": _unicode_array(pair_train),
        "evaluation_test_condition": _unicode_array(pair_test),
        "status_names": _unicode_array(STATUS_NAMES),
        "reason_code_names": _unicode_array(decoding.POOLED_REASON_CODES),
        "convergence_status_names": _unicode_array(CONVERGENCE_STATUS_NAMES),
        "target_values": np.full((T, N), np.nan, dtype=np.float64),
        "target_eligible": np.zeros((T, N), dtype=bool),
        "master_eligible": np.zeros((T, N), dtype=bool),
        "eligibility_reason_index": np.full((T, N), target_reason, dtype=np.int16),
        "condition_masks": np.zeros((C, N), dtype=bool),
        "trial_row_indices": tensor_rows.copy(),
        "outer_fold_ids": np.full((T, M), -1, dtype=np.int16),
        "training_window_bin_mask": np.zeros((W, Q), dtype=bool),
        "transform_status": np.full((W, C, T, F, A), 2, dtype=np.uint8),
        "transform_reason_index": np.full((W, C, T, F, A), outer_reason, dtype=np.int16),
        "transform_input_unit_count": np.empty((W, C, T, F, A), dtype=np.int32),
        "transform_usable_unit_count": np.full((W, C, T, F, A), -1, dtype=np.int32),
        "transform_component_limit": np.full((W, C, T, F, A), -1, dtype=np.int32),
        "transform_requested_pc_count": np.empty((W, C, T, F, A), dtype=np.int32),
        "transform_effective_pc_count": np.full((W, C, T, F, A), -1, dtype=np.int32),
        "model_status": np.full((W, C, T, F, R, P), 2, dtype=np.uint8),
        "model_reason_index": np.full((W, C, T, F, R, P), outer_reason, dtype=np.int16),
        "model_train_trial_count": np.full((W, C, T, F, R, P), -1, dtype=np.int32),
        "model_stacked_row_count": np.full((W, C, T, F, R, P), -1, dtype=np.int32),
        "model_requested_feature_count": np.empty((W, C, T, F, R, P), dtype=np.int32),
        "model_feature_count": np.full((W, C, T, F, R, P), -1, dtype=np.int32),
        "model_convergence_status": np.zeros((W, C, T, F, R, P), dtype=np.uint8),
        "model_intercept": np.full((W, C, T, F, R, P), np.nan, dtype=np.float64),
        "model_coefficients": np.full((W, C, T, F, R, P, G), np.nan, dtype=np.float64),
        "model_feature_active": np.zeros((W, C, T, F, R, P, G), dtype=bool),
        "evaluation_status": np.zeros((W, E, T, F, R, P, Q, K), dtype=np.uint8),
        "evaluation_reason_index": np.full((W, E, T, F, R, P, Q, K), -1, dtype=np.int16),
        "evaluation_scores": np.full((W, E, T, F, R, P, Q, K), np.nan, dtype=np.float64),
        "evaluation_test_trial_count": np.full((W, E, T, F), -1, dtype=np.int32),
        "gap_status": np.zeros((W, C, T, F, R, P, Q, K), dtype=np.uint8),
        "gap_reason_index": np.full((W, C, T, F, R, P, Q, K), -1, dtype=np.int16),
        "gap_scores": np.full((W, C, T, F, R, P, Q, K), np.nan, dtype=np.float64),
        "time_bin_edges_s": edges.copy(),
        "training_window_start_s": np.asarray([window.start_s for window in windows], dtype=np.float64),
        "training_window_stop_s": np.asarray([window.stop_s for window in windows], dtype=np.float64),
        "trial_positions": positions.copy(),
        "trial_ids": identifiers.copy(),
        "behavioral_block_payload": payload,
        "behavioral_block_offsets": offsets,
        "target_family_names": _unicode_array(families),
        "target_unit_names": _unicode_array(target_units),
        "target_positive_class": np.asarray([1 if family == "categorical" else -1 for family in families], dtype=np.int8),
        "coefficient_feature_ids": feature_ids,
        "coefficient_feature_regions": feature_regions,
        "coefficient_feature_present": feature_present,
        "unit_selection_rules_json": _unicode_array([_canonical_json(value) for value in unit_selection_rules]),
        "model_train_class_counts": np.full((W, C, T, F, 2), -1, dtype=np.int32),
        "evaluation_test_class_counts": np.full((W, E, T, F, 2), -1, dtype=np.int32),
        "target_estimator_class_names": _unicode_array(target_estimator_class_names),
        "target_estimator_parameters_json": _unicode_array([_canonical_json(value) for value in target_estimator_parameters]),
    }
    for window_index, window in enumerate(windows):
        indices = np.asarray(window.bin_indices, dtype=int)
        if indices.ndim != 1 or indices.size == 0 or (indices < 0).any() or (indices >= Q).any():
            raise ValueError("Training-window bin indices are invalid.")
        arrays["training_window_bin_mask"][window_index, indices] = True
    arrays["transform_input_unit_count"][..., 0] = len(pfc_ids)
    arrays["transform_input_unit_count"][..., 1] = len(hpc_ids)
    arrays["transform_requested_pc_count"][..., 0] = pfc_pc_count
    arrays["transform_requested_pc_count"][..., 1] = hpc_pc_count
    requested = {
        (0, 0): pfc_pc_count,
        (1, 0): hpc_pc_count,
        (2, 0): pfc_pc_count + hpc_pc_count,
        (0, 1): len(pfc_ids),
        (1, 1): len(hpc_ids),
        (2, 1): len(pfc_ids) + len(hpc_ids),
    }
    for region_index in range(R):
        for representation_index in range(P):
            arrays["model_requested_feature_count"][..., region_index, representation_index] = requested[(region_index, representation_index)]
    for target_index, family in enumerate(families):
        metric_indices = (0, 1) if family == "categorical" else (2,)
        for metric_index in metric_indices:
            arrays["evaluation_status"][:, :, target_index, :, :, :, :, metric_index] = 2
            arrays["evaluation_reason_index"][:, :, target_index, :, :, :, :, metric_index] = outer_reason
            arrays["gap_status"][:, :, target_index, :, :, :, :, metric_index] = 2
            arrays["gap_reason_index"][:, :, target_index, :, :, :, :, metric_index] = (
                decoding.POOLED_REASON_CODES.index("source_score_unavailable")
            )

    metadata = {
        "schema_version": POOLED_RESULT_SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "random_seed": random_seed,
        "axes": {name: list(axes) for name, axes in _AXES.items()},
        "units": {
            "time": "seconds relative to choice",
            "rates": "Hz",
            "scores": "dimensionless metrics",
        },
        "weighting_policy": "each stacked row has weight 1 / n_training_bins",
        "provenance": {
            "run_fingerprint": run_fingerprint,
            "source_fingerprint": source_fingerprint,
        },
    }
    validate_pooled_result_arrays(arrays, metadata)
    return arrays, metadata


def _expected_shapes(arrays: Mapping[str, np.ndarray]) -> dict[str, tuple[int, ...]]:
    """Derive every schema shape from validated coordinate lengths.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Candidate result arrays containing all coordinate members.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Exact member-to-shape mapping.
    """
    sizes = {
        "window": len(arrays["training_window_names"]),
        "condition": len(arrays["condition_names"]),
        "target": len(arrays["target_names"]),
        "fold": len(arrays["fold_ids"]),
        "region": 3,
        "source_region": 2,
        "representation": 2,
        "time": len(arrays["evaluation_time_centers_s"]),
        "time_edge": len(arrays["evaluation_time_centers_s"]) + 1,
        "metric": 3,
        "pair": len(arrays["evaluation_train_condition"]),
        "trial": len(arrays["trial_positions"]),
        "tensor_trial": len(arrays["trial_row_indices"]),
        "feature": arrays["coefficient_feature_ids"].shape[2],
        "block_id_byte": len(arrays["behavioral_block_payload"]),
        "trial_offset": len(arrays["trial_positions"]) + 1,
        "class": 2,
        "status": 3,
        "reason": len(decoding.POOLED_REASON_CODES),
        "convergence_status": 4,
    }
    return {
        name: tuple(sizes[axis] for axis in axes)
        for name, axes in _AXES.items()
    }


def validate_pooled_result_arrays(
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> None:
    """Strictly validate complete pooled arrays, axes, dtypes, and sentinels.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Exact schema-1 primitive arrays with units and axes declared in metadata.
    metadata : mapping[str, object]
        Canonical schema, analysis, units, weighting, and provenance mapping.

    Returns
    -------
    None
        Raises ValueError before any publication when the schema drifts.
    """
    if set(arrays) != set(_AXES):
        raise ValueError("Pooled result array names do not match schema 1.")
    if metadata.get("schema_version") != POOLED_RESULT_SCHEMA_VERSION:
        raise ValueError("Pooled result schema version is invalid.")
    if metadata.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError("Pooled analysis version is invalid.")
    if metadata.get("axes") != {name: list(axes) for name, axes in _AXES.items()}:
        raise ValueError("Pooled result axis metadata is invalid.")
    if not isinstance(metadata.get("provenance"), Mapping):
        raise ValueError("Pooled result provenance is invalid.")
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.dtype == object:
            raise ValueError(f"Pooled result array {name!r} is unsafe or invalid.")
    expected_shapes = _expected_shapes(arrays)
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"Pooled result array {name!r} has invalid shape/axes.")
    exact_dtypes = {
        "fold_ids": np.dtype(np.int16),
        "target_values": np.dtype(np.float64),
        "target_eligible": np.dtype(bool),
        "master_eligible": np.dtype(bool),
        "eligibility_reason_index": np.dtype(np.int16),
        "condition_masks": np.dtype(bool),
        "trial_row_indices": np.dtype(np.int64),
        "outer_fold_ids": np.dtype(np.int16),
        "training_window_bin_mask": np.dtype(bool),
        "transform_status": np.dtype(np.uint8),
        "transform_reason_index": np.dtype(np.int16),
        "transform_component_limit": np.dtype(np.int32),
        "model_status": np.dtype(np.uint8),
        "model_reason_index": np.dtype(np.int16),
        "model_convergence_status": np.dtype(np.uint8),
        "model_intercept": np.dtype(np.float64),
        "model_coefficients": np.dtype(np.float64),
        "model_feature_active": np.dtype(bool),
        "evaluation_status": np.dtype(np.uint8),
        "evaluation_reason_index": np.dtype(np.int16),
        "evaluation_scores": np.dtype(np.float64),
        "gap_status": np.dtype(np.uint8),
        "gap_reason_index": np.dtype(np.int16),
        "gap_scores": np.dtype(np.float64),
        "evaluation_time_centers_s": np.dtype(np.float64),
        "time_bin_edges_s": np.dtype(np.float64),
        "training_window_start_s": np.dtype(np.float64),
        "training_window_stop_s": np.dtype(np.float64),
        "trial_positions": np.dtype(np.int64),
        "trial_ids": np.dtype(np.int64),
        "behavioral_block_payload": np.dtype(np.uint8),
        "behavioral_block_offsets": np.dtype(np.int64),
        "target_positive_class": np.dtype(np.int8),
        "coefficient_feature_present": np.dtype(bool),
    }
    count_names = {
        name
        for name in arrays
        if name.endswith("_count") or name.endswith("_counts")
    }
    for name in count_names:
        exact_dtypes[name] = np.dtype(np.int32)
    for name, dtype in exact_dtypes.items():
        if arrays[name].dtype != dtype:
            raise ValueError(f"Pooled result array {name!r} has invalid dtype.")
    string_names = set(arrays).difference(exact_dtypes)
    for name in string_names:
        if arrays[name].dtype.kind != "U":
            raise ValueError(f"Pooled coordinate {name!r} must use Unicode dtype.")
    if tuple(arrays["reason_code_names"].tolist()) != decoding.POOLED_REASON_CODES:
        raise ValueError("Pooled reason-code vocabulary is invalid.")
    if tuple(arrays["status_names"].tolist()) != STATUS_NAMES:
        raise ValueError("Pooled status vocabulary is invalid.")
    if tuple(arrays["convergence_status_names"].tolist()) != CONVERGENCE_STATUS_NAMES:
        raise ValueError("Pooled convergence vocabulary is invalid.")
    for name in ("transform_status", "model_status", "evaluation_status", "gap_status"):
        if (arrays[name] > 2).any():
            raise ValueError(f"Pooled status array {name!r} contains an unknown value.")
    reason_count = len(decoding.POOLED_REASON_CODES)
    for name in (
        "eligibility_reason_index",
        "transform_reason_index",
        "model_reason_index",
        "evaluation_reason_index",
        "gap_reason_index",
    ):
        if (arrays[name] < -1).any() or (arrays[name] >= reason_count).any():
            raise ValueError(f"Pooled reason array {name!r} contains an unknown index.")
    if any("prediction" in name for name in arrays):
        raise ValueError("Dense per-trial predictions are forbidden in schema 1.")


def _atomic_npz(path: Path, members: Mapping[str, np.ndarray]) -> Path:
    """Atomically publish one immutable non-pickle NPZ file.

    Parameters
    ----------
    path : pathlib.Path
        New destination path.
    members : mapping[str, numpy.ndarray]
        Safe primitive NPZ members.

    Returns
    -------
    pathlib.Path
        Published destination.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        np.savez_compressed(temporary, **members)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        raise
    return destination


def _record_payload(record: decoding.PooledModelRecord) -> dict[str, object]:
    """Convert one durable pooled model record to JSON-safe values.

    Parameters
    ----------
    record : decoding.PooledModelRecord
        Model counts, provenance, coefficients, and status.

    Returns
    -------
    dict[str, object]
        JSON-safe record without sklearn runtime objects.
    """
    return {
        **record.__dict__,
        "coefficients": record.coefficients.tolist(),
        "feature_ids": list(record.feature_ids),
    }


def _transform_payload(transforms: decoding.PooledTransforms) -> dict[str, object]:
    """Convert transform audit counts to JSON without fitted PCA objects.

    Parameters
    ----------
    transforms : decoding.PooledTransforms
        Two anatomical training-only transform states.

    Returns
    -------
    dict[str, object]
        JSON-safe unit identities, compact reasons, and transform counts.
    """
    def regional(transform: modeling.RegionTransform) -> dict[str, object]:
        """Return one regional transform's scalar audit values.

        Parameters
        ----------
        transform : modeling.RegionTransform
            One anatomical transform with dimensionless counts and unit-axis
            positions; fitted scaler/PCA arrays are deliberately excluded.

        Returns
        -------
        dict[str, object]
            JSON-safe availability, reason, count, and unit-position values.
        """
        return {
            "is_available": transform.is_available,
            "unavailable_reason": transform.unavailable_reason,
            "component_limit": transform.component_limit,
            "effective_component_count": transform.effective_component_count,
            "cap_warning": transform.cap_warning,
            "input_unit_count": transform.input_unit_count,
            "requested_pc_count": transform.requested_pc_count,
            "usable_unit_indices": transform.usable_unit_indices.tolist(),
        }

    return {
        "pfc": regional(transforms.pfc_transform),
        "hpc": regional(transforms.hpc_transform),
        "pfc_unit_ids": list(transforms.pfc_unit_ids),
        "hpc_unit_ids": list(transforms.hpc_unit_ids),
        "pfc_reason": transforms.pfc_reason,
        "hpc_reason": transforms.hpc_reason,
    }


def _result_payload(result: decoding.PooledWindowTargetResult) -> dict[str, object]:
    """Encode one runtime cell result without estimator or PCA objects.

    Parameters
    ----------
    result : decoding.PooledWindowTargetResult
        Completed window-target scientific result.

    Returns
    -------
    dict[str, object]
        JSON-safe durable checkpoint payload.
    """
    return {
        "target_identifier": result.target_identifier,
        "target_family": result.target_family,
        "outer_fold_ids": result.outer_fold_ids.tolist(),
        "unavailable_reason": result.unavailable_reason,
        "models": [
            {"key": list(key), "record": _record_payload(model.record)}
            for key, model in result.models.items()
        ],
        "evaluations": [
            {
                "key": list(key),
                "evaluation_time_centers_s": value.evaluation_time_centers_s.tolist(),
                "metric_names": list(value.metric_names),
                "status": value.status.tolist(),
                "reason_codes": value.reason_codes.tolist(),
                "scores": value.scores.tolist(),
                "test_trial_count": value.test_trial_count,
                "test_class_counts": value.test_class_counts,
                "test_indices": value.test_indices.tolist(),
            }
            for key, value in result.evaluations.items()
        ],
        "gaps": [
            {
                "key": list(key),
                "status": value.status.tolist(),
                "reason_codes": value.reason_codes.tolist(),
                "scores": value.scores.tolist(),
            }
            for key, value in result.gaps.items()
        ],
        "transforms": [
            {"key": list(key), "value": _transform_payload(value)}
            for key, value in result.transforms.items()
        ],
    }


def save_pooled_checkpoint(
    path: Path,
    *,
    checkpoint_key: str,
    run_fingerprint: str,
    result: decoding.PooledWindowTargetResult,
) -> None:
    """Validate and atomically save one exact window-target checkpoint.

    Parameters
    ----------
    path : pathlib.Path
        New immutable checkpoint destination.
    checkpoint_key : str
        Canonical ``<training_window>::<target>`` identity.
    run_fingerprint : str
        Exact source/config/input/schema identity.
    result : decoding.PooledWindowTargetResult
        Complete runtime cell result; fitted objects are deliberately omitted.

    Returns
    -------
    None
        Publishes one safe scalar-Unicode JSON NPZ member.
    """
    parts = checkpoint_key.split("::")
    if len(parts) != 2 or not all(parts) or parts[1] != result.target_identifier:
        raise ValueError("Checkpoint key does not match the result target identity.")
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("Checkpoint run fingerprint must be nonempty.")
    metadata = {
        "schema_version": POOLED_RESULT_SCHEMA_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "checkpoint_key": checkpoint_key,
        "run_fingerprint": run_fingerprint,
    }
    _atomic_npz(
        Path(path),
        {
            "checkpoint_meta": np.array(_canonical_json(metadata)),
            "result_json": np.array(_canonical_json(_result_payload(result))),
        },
    )


def _audit_region_transform(value: Mapping[str, object]) -> modeling.RegionTransform:
    """Restore transform counts without recreating fitted PCA/scaler objects.

    Parameters
    ----------
    value : mapping[str, object]
        JSON transform audit payload.

    Returns
    -------
    modeling.RegionTransform
        Non-evaluable audit state with scalar counts and unit positions intact.
    """
    usable = np.asarray(value["usable_unit_indices"], dtype=int)
    input_count = int(value["input_unit_count"])
    return modeling.RegionTransform(
        is_available=bool(value["is_available"]),
        unavailable_reason=value["unavailable_reason"],
        unit_means_hz=np.full(input_count, np.nan),
        unit_scales_hz=np.full(input_count, np.nan),
        usable_unit_indices=usable,
        pca_model=None,
        component_limit=int(value["component_limit"]),
        effective_component_count=int(value["effective_component_count"]),
        cap_warning=value["cap_warning"],
        input_unit_count=input_count,
        requested_pc_count=int(value["requested_pc_count"]),
    )


def _restore_result(value: Mapping[str, object]) -> decoding.PooledWindowTargetResult:
    """Restore one durable checkpoint result with no fitted objects.

    Parameters
    ----------
    value : mapping[str, object]
        Validated decoded checkpoint JSON payload.

    Returns
    -------
    decoding.PooledWindowTargetResult
        Audit/evaluation result whose model estimators and transform references
        are absent; transform count records remain in the result-level mapping.
    """
    models = {}
    for item in value["models"]:
        record_value = dict(item["record"])
        record_value["coefficients"] = np.asarray(record_value["coefficients"], dtype=float)
        record_value["feature_ids"] = tuple(record_value["feature_ids"])
        record = decoding.PooledModelRecord(**record_value)
        models[tuple(item["key"])] = decoding.FittedPooledModel(
            record=record,
            transforms=None,
            fitted_estimator=None,
        )
    evaluations = {
        tuple(item["key"]): decoding.PooledEvaluationResult(
            evaluation_time_centers_s=np.asarray(item["evaluation_time_centers_s"], dtype=float),
            metric_names=tuple(item["metric_names"]),
            status=np.asarray(item["status"], dtype="U16"),
            reason_codes=np.asarray(item["reason_codes"], dtype="U32"),
            scores=np.asarray(item["scores"], dtype=float),
            test_trial_count=int(item["test_trial_count"]),
            test_class_counts=(
                None
                if item["test_class_counts"] is None
                else tuple(int(count) for count in item["test_class_counts"])
            ),
            test_indices=np.asarray(item["test_indices"], dtype=int),
        )
        for item in value["evaluations"]
    }
    gaps = {
        tuple(item["key"]): decoding.PooledGapResult(
            status=np.asarray(item["status"], dtype="U16"),
            reason_codes=np.asarray(item["reason_codes"], dtype="U32"),
            scores=np.asarray(item["scores"], dtype=float),
        )
        for item in value["gaps"]
    }
    transforms = {}
    for item in value["transforms"]:
        transform_value = item["value"]
        transforms[tuple(item["key"])] = decoding.PooledTransforms(
            pfc_transform=_audit_region_transform(transform_value["pfc"]),
            hpc_transform=_audit_region_transform(transform_value["hpc"]),
            pfc_unit_ids=tuple(transform_value["pfc_unit_ids"]),
            hpc_unit_ids=tuple(transform_value["hpc_unit_ids"]),
            pfc_reason=transform_value["pfc_reason"],
            hpc_reason=transform_value["hpc_reason"],
        )
    return decoding.PooledWindowTargetResult(
        models=models,
        evaluations=evaluations,
        gaps=gaps,
        transforms=transforms,
        unavailable_reason=value["unavailable_reason"],
        target_identifier=str(value["target_identifier"]),
        target_family=str(value["target_family"]),
        outer_fold_ids=np.asarray(value["outer_fold_ids"], dtype=int),
    )


def load_pooled_checkpoint(
    path: Path,
    *,
    expected_checkpoint_key: str,
    expected_run_fingerprint: str,
) -> decoding.PooledWindowTargetResult:
    """Load one safe checkpoint only when its complete identity matches.

    Parameters
    ----------
    path : pathlib.Path
        Existing checkpoint NPZ.
    expected_checkpoint_key, expected_run_fingerprint : str
        Exact cell and run identities required for reuse.

    Returns
    -------
    decoding.PooledWindowTargetResult
        Durable result with no fitted sklearn estimator or PCA object.
    """
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"checkpoint_meta", "result_json"}:
            raise ValueError("Checkpoint members are invalid.")
        try:
            metadata = json.loads(str(archive["checkpoint_meta"].item()))
            value = json.loads(str(archive["result_json"].item()))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("Checkpoint JSON is invalid.") from error
    if (
        metadata.get("schema_version") != POOLED_RESULT_SCHEMA_VERSION
        or metadata.get("analysis_version") != ANALYSIS_VERSION
    ):
        raise ValueError("Checkpoint schema or analysis identity is invalid.")
    if metadata.get("checkpoint_key") != expected_checkpoint_key:
        raise ValueError("Checkpoint key identity does not match.")
    if metadata.get("run_fingerprint") != expected_run_fingerprint:
        raise ValueError("Checkpoint run fingerprint identity does not match.")
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint result payload is invalid.")
    result = _restore_result(value)
    if expected_checkpoint_key.split("::")[-1] != result.target_identifier:
        raise ValueError("Checkpoint key target identity does not match its payload.")
    return result


def save_pooled_run(
    run_directory: Path,
    *,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> Path:
    """Validate and atomically publish one immutable final pooled result.

    Parameters
    ----------
    run_directory : pathlib.Path
        Existing or creatable synchronous work directory.
    arrays : mapping[str, numpy.ndarray]
        Complete schema-1 scientific arrays.
    metadata : mapping[str, object]
        Canonical schema, axes, units, weighting, and provenance.

    Returns
    -------
    pathlib.Path
        Published ``results.npz`` path.
    """
    validate_pooled_result_arrays(arrays, metadata)
    members = {name: value for name, value in arrays.items()}
    members["metadata_json"] = np.array(_canonical_json(metadata))
    return _atomic_npz(Path(run_directory) / "results.npz", members)


def load_pooled_run(
    run_directory: Path,
    *,
    expected_run_fingerprint: str | None = None,
) -> dict[str, object]:
    """Load and strictly validate one complete pooled result.

    Parameters
    ----------
    run_directory : pathlib.Path
        Directory containing ``results.npz``.
    expected_run_fingerprint : str or None
        Optional exact complete identity.

    Returns
    -------
    dict[str, object]
        Copied arrays, canonical metadata, and run fingerprint.
    """
    with np.load(Path(run_directory) / "results.npz", allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("Pooled result metadata is missing.")
        try:
            metadata = json.loads(str(archive["metadata_json"].item()))
        except (ValueError, json.JSONDecodeError) as error:
            raise ValueError("Pooled result metadata JSON is invalid.") from error
        arrays = {
            name: archive[name].copy()
            for name in archive.files
            if name != "metadata_json"
        }
    validate_pooled_result_arrays(arrays, metadata)
    fingerprint = metadata["provenance"]["run_fingerprint"]
    if expected_run_fingerprint is not None and fingerprint != expected_run_fingerprint:
        raise ValueError("Pooled run fingerprint does not match expected identity.")
    return {"arrays": arrays, "metadata": metadata, "run_fingerprint": fingerprint}


def pooled_scientific_source_fingerprint(repository_root: Path) -> dict[str, object]:
    """Hash exactly the reviewed pooled scientific dependency list.

    Parameters
    ----------
    repository_root : pathlib.Path
        Source checkout root; its absolute prefix is excluded from identity.

    Returns
    -------
    dict[str, object]
        Sorted per-file SHA-256 records and their combined fingerprint.
    """
    root = Path(repository_root)
    files = []
    for relative_path in sorted(POOLED_SCIENTIFIC_SOURCE_PATHS):
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Required pooled scientific source is missing: {relative_path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"relative_path": relative_path, "sha256": digest})
    fingerprint = hashlib.sha256(_canonical_json(files).encode("utf-8")).hexdigest()
    return {"files": files, "fingerprint": fingerprint}


def validate_pooled_source_cleanliness(repository_root: Path) -> None:
    """Require every explicit pooled dependency to be tracked and clean.

    Parameters
    ----------
    repository_root : pathlib.Path
        Git checkout containing all pooled scientific dependencies.

    Returns
    -------
    None
        Unrelated neighboring paths are ignored; dirty/untracked explicit
        dependencies raise ValueError.
    """
    root = Path(repository_root)
    scoped = set(POOLED_SCIENTIFIC_SOURCE_PATHS)
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for line in status:
        path = line[3:].strip().split(" -> ")[-1].strip().strip('"')
        if path in scoped:
            raise ValueError(f"Pooled scientific source is not clean: {path}")
    tracked = set(
        subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    missing = scoped.difference(tracked)
    if missing:
        raise ValueError(f"Required pooled scientific source is not tracked: {sorted(missing)[0]}")
