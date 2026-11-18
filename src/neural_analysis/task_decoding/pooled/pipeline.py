"""Synchronous preparation, checkpoint resume, and result assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shlex
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)
from src.neural_analysis.task_decoding import activity, conditions, modeling, targets
from src.neural_analysis.task_decoding import config as base_config
from src.neural_analysis.task_decoding import results as base_results
from src.neural_analysis.task_decoding.resource_usage import normalize_peak_rss_bytes
from src.neural_analysis.task_decoding.pooled import decoding, results
from src.neural_analysis.task_decoding.pooled.config import (
    ANALYSIS_VERSION,
    PooledDecodingConfig,
    TrainingWindow,
    load_pooled_decoding_config,
    load_prepared_pooled_decoding_config,
    resolve_training_windows,
    scientific_pooled_config_payload,
)


@dataclass(frozen=True)
class PooledComputationContext:
    """Prepared scientific inputs for synchronous pooled computation.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Mutable complete schema-1 result arrays with axes declared by metadata.
    metadata : dict[str, object]
        Canonical JSON-safe result metadata and complete run provenance.
    run_fingerprint : str
        SHA-256 identity binding source, configuration, inputs, and session.
    target_table : pandas.DataFrame or None
        Full chronological target table shaped ``(trial, column)``.
    condition_masks : mapping[str, numpy.ndarray] or None
        Full-table Boolean masks, each shaped ``(trial,)``.
    outer_splits : mapping[str, modeling.SplitPlan] or None
        Target-keyed split plans on the common ``tensor_trial`` axis.
    pfc_rate_tensor_hz, hpc_rate_tensor_hz : numpy.ndarray or None
        Regional rates shaped ``(tensor_trial, time, unit)`` in Hz.
    pfc_unit_ids, hpc_unit_ids : tuple[str, ...]
        Stable identifiers aligned to the regional unit axes.
    pfc_pc_count, hpc_pc_count : int
        Requested regional component counts.
    """

    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]
    run_fingerprint: str
    target_table: pd.DataFrame | None = None
    condition_masks: Mapping[str, np.ndarray] | None = None
    outer_splits: Mapping[str, modeling.SplitPlan] | None = None
    pfc_rate_tensor_hz: np.ndarray | None = None
    hpc_rate_tensor_hz: np.ndarray | None = None
    pfc_unit_ids: tuple[str, ...] = ()
    hpc_unit_ids: tuple[str, ...] = ()
    pfc_pc_count: int = 0
    hpc_pc_count: int = 0


def populate_target_audit_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    target_index: int,
    target_values: np.ndarray,
    target_eligible: np.ndarray,
    baseline_eligible: np.ndarray,
    bilateral_full_window_coverage: np.ndarray,
    trial_row_indices: np.ndarray,
    outer_fold_ids: np.ndarray,
) -> None:
    """Project one target's full-table eligibility and tensor-row fold audit.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Mutable schema-1 arrays with ``target,trial`` and
        ``target,tensor_trial`` members.
    target_index : int
        Zero-based target-axis position.
    target_values : numpy.ndarray
        Encoded classes or native numerical values shaped ``(trial,)``.
    target_eligible, baseline_eligible, bilateral_full_window_coverage : numpy.ndarray
        Boolean full-table masks shaped ``(trial,)``.
    trial_row_indices : numpy.ndarray
        Int64 full-table positions shaped ``(tensor_trial,)``.
    outer_fold_ids : numpy.ndarray
        Fold IDs shaped ``(tensor_trial,)``; unavailable rows contain ``-1``.

    Returns
    -------
    None
        Mutates only the selected target rows in the supplied arrays.
    """
    values = np.asarray(target_values, dtype=np.float64)
    row_indices = np.asarray(trial_row_indices)
    folds = np.asarray(outer_fold_ids)
    if values.shape != arrays["target_values"][target_index].shape:
        raise ValueError("target_values does not match the full trial axis.")
    if row_indices.dtype != np.int64 or row_indices.shape != arrays["trial_row_indices"].shape:
        raise ValueError("trial_row_indices does not match the tensor-trial axis.")
    if not np.array_equal(row_indices, arrays["trial_row_indices"]):
        raise ValueError("trial_row_indices differs from the result coordinate.")
    if folds.shape != arrays["outer_fold_ids"][target_index].shape:
        raise ValueError("outer_fold_ids does not match the tensor-trial axis.")
    eligibility = decoding.build_master_eligibility(
        baseline_eligible,
        target_eligible,
        bilateral_full_window_coverage,
    )
    arrays["target_values"][target_index] = values
    arrays["target_eligible"][target_index] = eligibility.target_eligible
    arrays["master_eligible"][target_index] = eligibility.master_eligible
    arrays["eligibility_reason_index"][target_index] = eligibility.reason_index
    arrays["outer_fold_ids"][target_index] = folds.astype(np.int16, copy=False)


def _probe_source_files(resolved_session: object) -> dict[str, tuple[Path, ...]]:
    """List exact probe files opened by regional activity loading.

    Parameters
    ----------
    resolved_session : ResolvedSession
        Metadata-resolved session whose probe paths are absolute.

    Returns
    -------
    dict[str, tuple[pathlib.Path, ...]]
        Probe-keyed existing source paths in deterministic order.
    """
    output: dict[str, tuple[Path, ...]] = {}
    for source in resolved_session.probes:
        candidates = (
            source.alignment_file,
            source.channel_quality_file,
            source.cluster_info_file,
            source.spike_clusters_file,
            source.spike_times_file,
        )
        output[f"{source.probe_id}_sources"] = tuple(
            sorted({Path(path) for path in candidates if path is not None and path.is_file()})
        )
    return output


def _unit_ids(region_activity: activity.RegionActivity) -> tuple[str, ...]:
    """Read stable selected-unit identities in tensor-axis order.

    Parameters
    ----------
    region_activity : activity.RegionActivity
        Loaded regional population with one metadata row per tensor unit.

    Returns
    -------
    tuple[str, ...]
        Nonempty identifiers shaped logically as ``(unit,)``.
    """
    identifiers = tuple(region_activity.unit_metadata["unit_id"].astype(str).tolist())
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Selected regional unit identifiers must be nonempty and unique.")
    return identifiers


def _estimator_audit(target_names: Sequence[str]) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    """Resolve concrete fixed estimators for target-axis audit coordinates.

    Parameters
    ----------
    target_names : sequence[str]
        Configured target identifiers in target-axis order.

    Returns
    -------
    tuple[tuple[str, ...], tuple[dict[str, object], ...]]
        Concrete class names and JSON-safe fixed parameter mappings.
    """
    class_names: list[str] = []
    parameters: list[dict[str, object]] = []
    for target_name in target_names:
        family = base_config.TARGET_DEFINITION_BY_IDENTIFIER[target_name].family
        estimator = modeling.make_estimator(target_family=family)
        class_names.append(estimator.__class__.__name__)
        parameters.append(dict(estimator.get_params(deep=False)))
    return tuple(class_names), tuple(parameters)


def _target_units(target_names: Sequence[str]) -> tuple[str, ...]:
    """Return target-value units in target-axis order.

    Parameters
    ----------
    target_names : sequence[str]
        Configured target identifiers.

    Returns
    -------
    tuple[str, ...]
        ``encoded class`` for categorical targets and ``unitless`` for the
        currently supported native numerical targets.
    """
    return tuple(
        "encoded class"
        if base_config.TARGET_DEFINITION_BY_IDENTIFIER[name].family == "categorical"
        else "unitless"
        for name in target_names
    )


def _prepare_pooled_computation_context(
    config: PooledDecodingConfig,
    training_windows: Sequence[TrainingWindow],
) -> PooledComputationContext:
    """Load one session and construct all fixed synchronous scientific inputs.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated pooled overlay and base task-decoding configuration.
    training_windows : sequence[TrainingWindow]
        Canonical windows on the complete choice-relative time axis.

    Returns
    -------
    PooledComputationContext
        Complete initialized result schema plus loaded regional Hz tensors and
        target-wide fold plans. No model is fitted by this function.
    """
    windows = tuple(training_windows)
    scientific_config = scientific_pooled_config_payload(config, windows)
    repository_root = Path(__file__).resolve().parents[4]
    results.validate_pooled_source_cleanliness(repository_root)
    source_identity = results.pooled_scientific_source_fingerprint(repository_root)

    task_config = config.base_config
    resolved = resolve_session_metadata(
        load_session_metadata(task_config.session_metadata_path),
        task_config.session_metadata_path,
    )
    trial_table = pd.read_csv(task_config.augmented_trial_path)
    target_table = targets.build_target_table(trial_table, task_config)
    target_table[task_config.alignment] = trial_table[task_config.alignment].to_numpy(
        dtype=float
    )
    input_manifest = base_results.build_input_manifest(
        task_config.session_root,
        {
            "neural_session": task_config.session_metadata_path,
            "augmented_trials": task_config.augmented_trial_path,
            "trial_feature_parameters": task_config.trial_feature_parameter_path,
        },
        _probe_source_files(resolved),
    )
    run_fingerprint = base_results.build_run_fingerprint(
        ANALYSIS_VERSION,
        str(source_identity["fingerprint"]),
        scientific_config,
        input_manifest,
        resolved.session_id,
    )

    pfc_activity = activity.load_region_activity(
        task_config.pfc_region,
        resolved.probes,
        trusted_utc_bounds=task_config.trusted_utc_bounds.get(
            task_config.pfc_region.probe_id
        ),
    )
    hpc_activity = activity.load_region_activity(
        task_config.hpc_region,
        resolved.probes,
        trusted_utc_bounds=task_config.trusted_utc_bounds.get(
            task_config.hpc_region.probe_id
        ),
    )
    rate_tensors = activity.build_session_rate_tensors(
        target_table,
        pfc_activity,
        hpc_activity,
        alignment_event=task_config.alignment,
        window_s=(base_config.WINDOW_START_SECONDS, base_config.WINDOW_END_SECONDS),
        bin_width_s=task_config.bin_width_ms / 1000.0,
    )
    centers = rate_tensors.pfc_bin_centers_s
    half_width = task_config.bin_width_ms / 2000.0
    edges = np.concatenate((centers - half_width, [centers[-1] + half_width]))
    expected_windows = scientific_pooled_config_payload(config, windows)
    if expected_windows != scientific_config:
        raise ValueError("Resolved pooled scientific configuration drifted.")

    target_names = task_config.target_names
    target_families = tuple(
        base_config.TARGET_DEFINITION_BY_IDENTIFIER[name].family for name in target_names
    )
    estimator_names, estimator_parameters = _estimator_audit(target_names)
    pfc_unit_ids = _unit_ids(pfc_activity)
    hpc_unit_ids = _unit_ids(hpc_activity)
    arrays, metadata = results.initialize_pooled_result_arrays(
        training_windows=windows,
        condition_names=config.condition_names,
        target_names=target_names,
        target_family_names=target_families,
        target_unit_names=_target_units(target_names),
        outer_fold_count=task_config.outer_fold_count,
        evaluation_time_centers_s=centers,
        time_bin_edges_s=edges,
        trial_positions=target_table["row_position"].to_numpy(dtype=np.int64),
        trial_ids=target_table["trial_id"].to_numpy(dtype=np.int64),
        behavioral_block_ids=target_table["block_id"].to_numpy(copy=True),
        trial_row_indices=np.asarray(rate_tensors.trial_row_indices, dtype=np.int64),
        pfc_unit_ids=pfc_unit_ids,
        hpc_unit_ids=hpc_unit_ids,
        pfc_pc_count=task_config.pfc_pc_count,
        hpc_pc_count=task_config.hpc_pc_count,
        unit_selection_rules=(pfc_activity.selection_rules, hpc_activity.selection_rules),
        target_estimator_class_names=estimator_names,
        target_estimator_parameters=estimator_parameters,
        run_fingerprint=run_fingerprint,
        source_fingerprint=str(source_identity["fingerprint"]),
        random_seed=base_config.RANDOM_SEED,
    )
    full_condition_masks = conditions.build_condition_masks(
        trial_table,
        config.condition_names,
    )
    arrays["condition_masks"][:] = np.stack(
        [full_condition_masks[name] for name in config.condition_names]
    )
    bilateral = np.zeros(len(target_table), dtype=bool)
    bilateral[rate_tensors.trial_row_indices] = True
    baseline = target_table["baseline_valid"].to_numpy(dtype=bool)
    tensor_rows = arrays["trial_row_indices"]
    tensor_blocks = target_table["block_id"].to_numpy(copy=True)[tensor_rows]
    outer_splits: dict[str, modeling.SplitPlan] = {}
    for target_index, (target_name, target_family) in enumerate(
        zip(target_names, target_families, strict=True)
    ):
        values = target_table[target_name].to_numpy(dtype=float)
        eligible = target_table[f"{target_name}_valid"].to_numpy(dtype=bool)
        master = decoding.build_master_eligibility(baseline, eligible, bilateral)
        split = decoding.make_master_outer_split(
            values[tensor_rows],
            tensor_blocks,
            master.master_eligible[tensor_rows],
            target_family=target_family,
            fold_count=task_config.outer_fold_count,
        )
        outer_splits[target_name] = split
        populate_target_audit_arrays(
            arrays,
            target_index=target_index,
            target_values=values,
            target_eligible=eligible,
            baseline_eligible=baseline,
            bilateral_full_window_coverage=bilateral,
            trial_row_indices=tensor_rows,
            outer_fold_ids=split.fold_ids,
        )
    metadata.update(
        {
            "session_id": resolved.session_id,
            "scientific_config": scientific_config,
            "input_manifest": input_manifest,
            "scientific_source": source_identity,
        }
    )
    return PooledComputationContext(
        arrays=arrays,
        metadata=metadata,
        run_fingerprint=run_fingerprint,
        target_table=target_table,
        condition_masks=full_condition_masks,
        outer_splits=outer_splits,
        pfc_rate_tensor_hz=rate_tensors.pfc_rate_tensor_hz,
        hpc_rate_tensor_hz=rate_tensors.hpc_rate_tensor_hz,
        pfc_unit_ids=pfc_unit_ids,
        hpc_unit_ids=hpc_unit_ids,
        pfc_pc_count=task_config.pfc_pc_count,
        hpc_pc_count=task_config.hpc_pc_count,
    )


def _compute_window_target(
    context: PooledComputationContext,
    training_window: TrainingWindow,
    target_identifier: str,
) -> decoding.PooledWindowTargetResult:
    """Compute one complete training-window/target checkpoint cell.

    Parameters
    ----------
    context : PooledComputationContext
        Prepared full-table audit data and regional Hz tensors.
    training_window : TrainingWindow
        Selected training bins on the shared evaluation-time axis.
    target_identifier : str
        Configured target coordinate.

    Returns
    -------
    decoding.PooledWindowTargetResult
        All conditions, folds, regions, representations, pairs, and times for
        the requested checkpoint identity.
    """
    required = (
        context.target_table,
        context.condition_masks,
        context.outer_splits,
        context.pfc_rate_tensor_hz,
        context.hpc_rate_tensor_hz,
    )
    if any(value is None for value in required):
        raise ValueError("Pooled computation context lacks runtime scientific inputs.")
    assert context.target_table is not None
    assert context.condition_masks is not None
    assert context.outer_splits is not None
    assert context.pfc_rate_tensor_hz is not None
    assert context.hpc_rate_tensor_hz is not None
    target_names = context.arrays["target_names"].tolist()
    if target_identifier not in target_names:
        raise ValueError("Unknown pooled target identifier.")
    target_index = target_names.index(target_identifier)
    tensor_rows = context.arrays["trial_row_indices"]
    family = str(context.arrays["target_family_names"][target_index])
    return decoding.decode_pooled_window_target(
        training_window=training_window,
        target_identifier=target_identifier,
        target_family=family,
        target_values=context.arrays["target_values"][target_index, tensor_rows],
        master_eligible=context.arrays["master_eligible"][target_index, tensor_rows],
        block_ids=context.target_table["block_id"].to_numpy(copy=True)[tensor_rows],
        condition_masks={
            name: np.asarray(mask, dtype=bool)[tensor_rows]
            for name, mask in context.condition_masks.items()
        },
        outer_split=context.outer_splits[target_identifier],
        pfc_rate_tensor_hz=context.pfc_rate_tensor_hz,
        hpc_rate_tensor_hz=context.hpc_rate_tensor_hz,
        pfc_unit_ids=context.pfc_unit_ids,
        hpc_unit_ids=context.hpc_unit_ids,
        evaluation_time_centers_s=context.arrays["evaluation_time_centers_s"],
        pfc_pc_count=context.pfc_pc_count,
        hpc_pc_count=context.hpc_pc_count,
    )


def _reason_index(reason: str | None) -> int:
    """Convert one compact reason code to its schema coordinate index.

    Parameters
    ----------
    reason : str or None
        Pooled reason code, empty string, or absence.

    Returns
    -------
    int
        ``-1`` for no reason, otherwise its fixed vocabulary position.
    """
    if reason is None or reason == "":
        return -1
    try:
        return decoding.POOLED_REASON_CODES.index(reason)
    except ValueError as error:
        raise ValueError(f"Unknown pooled reason code: {reason}") from error


def _status_indices(values: np.ndarray) -> np.ndarray:
    """Convert dense textual statuses to schema uint8 indices.

    Parameters
    ----------
    values : numpy.ndarray
        Unicode status array with arbitrary scientific axes.

    Returns
    -------
    numpy.ndarray
        Same shape, uint8 encoded as not-applicable/valid/unavailable.
    """
    output = np.empty(np.asarray(values).shape, dtype=np.uint8)
    for index, name in enumerate(results.STATUS_NAMES):
        output[np.asarray(values) == name] = index
    if not np.isin(values, results.STATUS_NAMES).all():
        raise ValueError("Checkpoint contains an unknown status.")
    return output


def _reason_indices(values: np.ndarray) -> np.ndarray:
    """Convert dense textual reasons to schema int16 indices.

    Parameters
    ----------
    values : numpy.ndarray
        Unicode compact-reason array with arbitrary scientific axes.

    Returns
    -------
    numpy.ndarray
        Same shape with ``-1`` for empty entries and vocabulary indices else.
    """
    source = np.asarray(values)
    output = np.full(source.shape, -1, dtype=np.int16)
    for reason_index, reason in enumerate(decoding.POOLED_REASON_CODES):
        output[source == reason] = reason_index
    if not np.isin(source, ("", *decoding.POOLED_REASON_CODES)).all():
        raise ValueError("Checkpoint contains an unknown reason code.")
    return output


def _merge_transforms(
    arrays: Mapping[str, np.ndarray],
    *,
    window_index: int,
    target_index: int,
    result: decoding.PooledWindowTargetResult,
) -> None:
    """Project result-level anatomical transform audit fields into arrays.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Mutable schema-1 arrays.
    window_index, target_index : int
        Selected zero-based output cell coordinates.
    result : decoding.PooledWindowTargetResult
        Restored or newly computed checkpoint result.

    Returns
    -------
    None
        Mutates transform arrays for the selected cell.
    """
    conditions = arrays["condition_names"].tolist()
    if result.unavailable_reason is None:
        for condition_index, condition in enumerate(conditions):
            for fold in arrays["fold_ids"].tolist():
                if (condition, fold) in result.transforms:
                    continue
                model_reasons = {
                    fitted.record.reason
                    for key, fitted in result.models.items()
                    if key[0] == condition and key[1] == fold
                }
                if len(model_reasons) != 1:
                    raise ValueError(
                        "Missing transform cell has inconsistent model reasons."
                    )
                reason = model_reasons.pop()
                if reason is None:
                    raise ValueError("A valid model cannot lack its shared transforms.")
                cell = (window_index, condition_index, target_index, fold)
                arrays["transform_status"][cell] = 2
                arrays["transform_reason_index"][cell] = _reason_index(reason)
    for (condition, fold), transforms in result.transforms.items():
        condition_index = conditions.index(condition)
        for source_index, (transform, reason) in enumerate(
            (
                (transforms.pfc_transform, transforms.pfc_reason),
                (transforms.hpc_transform, transforms.hpc_reason),
            )
        ):
            cell = (window_index, condition_index, target_index, fold, source_index)
            arrays["transform_status"][cell] = 1 if transform.is_available else 2
            arrays["transform_reason_index"][cell] = _reason_index(reason)
            arrays["transform_input_unit_count"][cell] = transform.input_unit_count
            arrays["transform_usable_unit_count"][cell] = len(
                transform.usable_unit_indices
            )
            arrays["transform_component_limit"][cell] = transform.component_limit
            arrays["transform_requested_pc_count"][cell] = transform.requested_pc_count
            arrays["transform_effective_pc_count"][cell] = (
                transform.effective_component_count
            )


def _merge_models(
    arrays: Mapping[str, np.ndarray],
    *,
    window_index: int,
    target_index: int,
    result: decoding.PooledWindowTargetResult,
) -> None:
    """Project model audit fields and padded coefficients into result arrays.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Mutable schema-1 arrays.
    window_index, target_index : int
        Selected zero-based output cell coordinates.
    result : decoding.PooledWindowTargetResult
        Restored or newly computed checkpoint result.

    Returns
    -------
    None
        Mutates model and categorical training-count arrays for this cell.
    """
    condition_names = arrays["condition_names"].tolist()
    region_names = arrays["region_names"].tolist()
    representation_names = arrays["representation_names"].tolist()
    tensor_rows = arrays["trial_row_indices"]
    fold_ids = result.outer_fold_ids
    target_values = arrays["target_values"][target_index, tensor_rows]
    master = arrays["master_eligible"][target_index, tensor_rows]
    for (condition, fold, region, representation), fitted in result.models.items():
        record = fitted.record
        condition_index = condition_names.index(condition)
        region_index = region_names.index(region)
        representation_index = representation_names.index(representation)
        cell = (
            window_index,
            condition_index,
            target_index,
            fold,
            region_index,
            representation_index,
        )
        arrays["model_status"][cell] = 1 if record.is_valid else 2
        arrays["model_reason_index"][cell] = _reason_index(record.reason)
        arrays["model_train_trial_count"][cell] = record.train_trial_count
        arrays["model_stacked_row_count"][cell] = record.stacked_row_count
        arrays["model_requested_feature_count"][cell] = record.requested_feature_count
        arrays["model_feature_count"][cell] = record.feature_count
        arrays["model_convergence_status"][cell] = results.CONVERGENCE_STATUS_NAMES.index(
            record.convergence_status
        )
        arrays["model_intercept"][cell] = record.intercept
        coordinate_ids = arrays["coefficient_feature_ids"][
            region_index, representation_index
        ].tolist()
        for coefficient, feature_id in zip(
            record.coefficients,
            record.feature_ids,
            strict=True,
        ):
            try:
                feature_index = coordinate_ids.index(feature_id)
            except ValueError as error:
                raise ValueError(f"Unknown pooled coefficient feature: {feature_id}") from error
            coefficient_cell = (*cell, feature_index)
            arrays["model_coefficients"][coefficient_cell] = coefficient
            arrays["model_feature_active"][coefficient_cell] = True
        if result.target_family == "categorical":
            condition_mask = arrays["condition_masks"][condition_index, tensor_rows]
            training = master & condition_mask & (fold_ids >= 0) & (fold_ids != fold)
            arrays["model_train_class_counts"][
                window_index, condition_index, target_index, fold
            ] = (
                np.count_nonzero(target_values[training] == 0),
                np.count_nonzero(target_values[training] == 1),
            )


def _merge_evaluations_and_gaps(
    arrays: Mapping[str, np.ndarray],
    *,
    window_index: int,
    target_index: int,
    result: decoding.PooledWindowTargetResult,
) -> None:
    """Project dense evaluation and native-minus-transfer arrays.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Mutable schema-1 arrays.
    window_index, target_index : int
        Selected zero-based output cell coordinates.
    result : decoding.PooledWindowTargetResult
        Restored or newly computed checkpoint result.

    Returns
    -------
    None
        Mutates evaluation, gap, test-count, and class-count arrays.
    """
    pairs = list(
        zip(
            arrays["evaluation_train_condition"].tolist(),
            arrays["evaluation_test_condition"].tolist(),
            strict=True,
        )
    )
    regions = arrays["region_names"].tolist()
    representations = arrays["representation_names"].tolist()
    conditions = arrays["condition_names"].tolist()
    for key, evaluation in result.evaluations.items():
        train_condition, test_condition, fold, region, representation = key
        pair_index = pairs.index((train_condition, test_condition))
        cell = (
            window_index,
            pair_index,
            target_index,
            fold,
            regions.index(region),
            representations.index(representation),
        )
        arrays["evaluation_status"][cell] = _status_indices(evaluation.status)
        arrays["evaluation_reason_index"][cell] = _reason_indices(
            evaluation.reason_codes
        )
        arrays["evaluation_scores"][cell] = evaluation.scores
        count_cell = (window_index, pair_index, target_index, fold)
        arrays["evaluation_test_trial_count"][count_cell] = evaluation.test_trial_count
        if evaluation.test_class_counts is not None:
            arrays["evaluation_test_class_counts"][count_cell] = (
                evaluation.test_class_counts
            )
    for (condition, fold, region, representation), gap in result.gaps.items():
        cell = (
            window_index,
            conditions.index(condition),
            target_index,
            fold,
            regions.index(region),
            representations.index(representation),
        )
        arrays["gap_status"][cell] = _status_indices(gap.status)
        arrays["gap_reason_index"][cell] = _reason_indices(gap.reason_codes)
        arrays["gap_scores"][cell] = gap.scores


def _merge_window_target_result(
    context: PooledComputationContext,
    training_window: TrainingWindow,
    target_identifier: str,
    result: decoding.PooledWindowTargetResult,
) -> None:
    """Merge one identity-checked checkpoint into the complete dense schema.

    Parameters
    ----------
    context : PooledComputationContext
        Prepared mutable result arrays and exact coordinates.
    training_window : TrainingWindow
        Window coordinate associated with the checkpoint.
    target_identifier : str
        Target coordinate associated with the checkpoint.
    result : decoding.PooledWindowTargetResult
        Restored or newly computed durable cell.

    Returns
    -------
    None
        Mutates only the matching window-target slice.
    """
    if result.target_identifier != target_identifier:
        raise ValueError("Checkpoint result target identity does not match its cell.")
    window_index = context.arrays["training_window_names"].tolist().index(
        training_window.name
    )
    target_index = context.arrays["target_names"].tolist().index(target_identifier)
    if result.outer_fold_ids.shape != context.arrays["outer_fold_ids"][target_index].shape:
        raise ValueError("Checkpoint fold map does not match the tensor-trial axis.")
    context.arrays["outer_fold_ids"][target_index] = result.outer_fold_ids
    _merge_transforms(
        context.arrays,
        window_index=window_index,
        target_index=target_index,
        result=result,
    )
    _merge_models(
        context.arrays,
        window_index=window_index,
        target_index=target_index,
        result=result,
    )
    _merge_evaluations_and_gaps(
        context.arrays,
        window_index=window_index,
        target_index=target_index,
        result=result,
    )


def compute_or_resume_pooled_results(
    config: PooledDecodingConfig,
    training_windows: Sequence[TrainingWindow],
    work_directory: Path,
) -> Path:
    """Synchronously compute or resume pooled window-target checkpoints.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated pooled and base scientific settings.
    training_windows : sequence[TrainingWindow]
        Canonically resolved windows on the shared time axis.
    work_directory : pathlib.Path
        Working directory for immutable checkpoint NPZ files and final output.

    Returns
    -------
    pathlib.Path
        Immutable validated ``results.npz``. This function creates no reports,
        figures, terminal lifecycle state, or detached execution process.
    """
    windows = tuple(training_windows)
    scientific_pooled_config_payload(config, windows)
    directory = Path(work_directory)
    final_path = directory / "results.npz"
    context = _prepare_pooled_computation_context(config, windows)
    if final_path.exists():
        results.load_pooled_run(
            directory,
            expected_run_fingerprint=context.run_fingerprint,
        )
        return final_path
    checkpoint_directory = directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    for window in windows:
        for target_identifier in config.base_config.target_names:
            checkpoint_key = f"{window.name}::{target_identifier}"
            checkpoint_path = (
                checkpoint_directory / f"{window.name}__{target_identifier}.npz"
            )
            if checkpoint_path.exists():
                result = results.load_pooled_checkpoint(
                    checkpoint_path,
                    expected_checkpoint_key=checkpoint_key,
                    expected_run_fingerprint=context.run_fingerprint,
                )
            else:
                result = _compute_window_target(
                    context,
                    window,
                    target_identifier,
                )
                results.save_pooled_checkpoint(
                    checkpoint_path,
                    checkpoint_key=checkpoint_key,
                    run_fingerprint=context.run_fingerprint,
                    result=result,
                )
            _merge_window_target_result(
                context,
                window,
                target_identifier,
                result,
            )
    return results.save_pooled_run(
        directory,
        arrays=context.arrays,
        metadata=context.metadata,
    )


_RESOURCE_FIELDS = frozenset(
    {
        "schema_version",
        "analysis_type",
        "measurement_method",
        "status",
        "peak_rss_bytes",
        "wall_time_seconds",
        "user_cpu_seconds",
        "system_cpu_seconds",
        "cpu_efficiency",
        "resource_envelope",
        "completed_checkpoint_keys",
        "restored_checkpoint_keys",
        "stage_timing_seconds",
        "result_counts",
        "checkpoint_size_bytes",
        "output_size_bytes",
    }
)
_RESOURCE_ENVELOPE_FIELDS = frozenset(
    {
        "full_trial_count",
        "tensor_trial_count",
        "pfc_unit_count",
        "hpc_unit_count",
        "time_bin_count",
        "training_window_count",
        "condition_count",
        "evaluation_pair_count",
        "target_count",
        "outer_fold_count",
        "coefficient_feature_capacity",
        "categorical_fit_count",
        "numerical_fit_count",
        "transform_fit_count",
        "evaluation_metric_cell_count",
        "tensor_allocation_bytes",
        "planned_major_array_bytes",
    }
)
_RESULT_COUNT_FIELDS = frozenset(
    {
        "requested_transform_cells",
        "valid_transform_cells",
        "unavailable_transform_cells",
        "requested_model_cells",
        "valid_model_cells",
        "unavailable_model_cells",
        "requested_evaluation_metric_cells",
        "valid_evaluation_metric_cells",
        "unavailable_evaluation_metric_cells",
    }
)
_STAGE_FIELDS = frozenset(
    {"input_loading", "modeling", "final_assembly", "reporting"}
)
_RESOURCE_STATUSES = frozenset({"running", "failed", "interrupted", "complete"})
_RUN_PREFIX = "pooled_task_variable_decoding_"


def sha256_file(path: Path | str) -> str:
    """Hash one regular file without changing its bytes or metadata.

    Parameters
    ----------
    path : pathlib.Path or str
        Existing file path.

    Returns
    -------
    str
        Lower-case hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    """Encode one JSON-safe value deterministically.

    Parameters
    ----------
    value : object
        JSON-compatible mapping, sequence, or scalar.

    Returns
    -------
    str
        Sorted compact JSON text with ASCII-safe output.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Publish one small immutable/lifecycle artifact by sibling replacement.

    Parameters
    ----------
    path : pathlib.Path
        Destination below a prepared run directory.
    content : bytes
        Complete bytes to publish.

    Returns
    -------
    None
        Replaces the destination only after the temporary file is flushed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".temporary",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write one canonical JSON object.

    Parameters
    ----------
    path : pathlib.Path
        JSON destination.
    value : mapping[str, object]
        Complete JSON-safe object.

    Returns
    -------
    None
        Writes UTF-8 JSON with one trailing newline.
    """
    _atomic_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def _read_json(path: Path) -> dict[str, object]:
    """Read one required UTF-8 JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Existing sidecar path.

    Returns
    -------
    dict[str, object]
        Decoded mapping.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read required pooled artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Pooled artifact must be a JSON object: {path.name}")
    return value


def _utc_now() -> str:
    """Return the current whole-second UTC timestamp.

    Returns
    -------
    str
        ISO-8601 UTC text with a trailing ``Z``.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _thread_limit_mapping() -> dict[str, str]:
    """Return the exact numerical thread limits used by this process.

    Returns
    -------
    dict[str, str]
        OpenMP, MKL, and OpenBLAS thread-limit environment values. Values are
        dimensionless decimal strings and may be empty for non-CLI callers.
    """
    return {
        variable: os.environ.get(variable, "")
        for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }


def _canonical_edges(bin_width_ms: int) -> np.ndarray:
    """Build the frozen choice-relative time-bin edge axis.

    Parameters
    ----------
    bin_width_ms : int
        Supported bin width in milliseconds.

    Returns
    -------
    numpy.ndarray
        Float64 edge coordinates in seconds shaped ``(time + 1,)``.
    """
    bin_count = int(
        round(
            (base_config.WINDOW_END_SECONDS - base_config.WINDOW_START_SECONDS)
            / (bin_width_ms / 1000.0)
        )
    )
    return np.linspace(
        base_config.WINDOW_START_SECONDS,
        base_config.WINDOW_END_SECONDS,
        bin_count + 1,
        dtype=np.float64,
    )


def _build_live_manifest(
    config: PooledDecodingConfig,
    resolved_session: object,
) -> dict[str, object]:
    """Build the exact current input manifest for a pooled configuration.

    Parameters
    ----------
    config : PooledDecodingConfig
        Reconstructed or newly loaded pooled settings.
    resolved_session : ResolvedSession
        Metadata-resolved probe inputs below the session root.

    Returns
    -------
    dict[str, object]
        Portable file identities for every input opened by execution.
    """
    task_config = config.base_config
    return base_results.build_input_manifest(
        task_config.session_root,
        {
            "neural_session": task_config.session_metadata_path,
            "augmented_trials": task_config.augmented_trial_path,
            "trial_feature_parameters": task_config.trial_feature_parameter_path,
        },
        _probe_source_files(resolved_session),
    )


def _collect_small_inputs(config_path: Path | str) -> dict[str, object]:
    """Resolve bounded planning inputs without loading spike arrays or fitting.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Portable pooled overlay JSON path.

    Returns
    -------
    dict[str, object]
        Config/windows, session identity, tables, dry activity dimensions,
        manifest, source identity, and full run fingerprint. Tables retain
        chronological trial rows and no neural tensor is allocated.
    """
    config = load_pooled_decoding_config(config_path)
    windows = resolve_training_windows(
        config,
        _canonical_edges(config.base_config.bin_width_ms),
    )
    task_config = config.base_config
    resolved = resolve_session_metadata(
        load_session_metadata(task_config.session_metadata_path),
        task_config.session_metadata_path,
    )
    trial_table = pd.read_csv(task_config.augmented_trial_path)
    target_table = targets.build_target_table(trial_table, task_config)
    target_table[task_config.alignment] = trial_table[task_config.alignment].to_numpy(
        dtype=float
    )
    report = activity.inspect_activity_dry_run(
        target_table,
        task_config.pfc_region,
        task_config.hpc_region,
        resolved.probes,
        alignment_event=task_config.alignment,
        window_s=(base_config.WINDOW_START_SECONDS, base_config.WINDOW_END_SECONDS),
        bin_width_s=task_config.bin_width_ms / 1000.0,
        pfc_trusted_utc_bounds=task_config.trusted_utc_bounds.get(
            task_config.pfc_region.probe_id
        ),
        hpc_trusted_utc_bounds=task_config.trusted_utc_bounds.get(
            task_config.hpc_region.probe_id
        ),
    )
    manifest = _build_live_manifest(config, resolved)
    repository_root = Path(__file__).resolve().parents[4]
    source = results.pooled_scientific_source_fingerprint(repository_root)
    scientific_config = scientific_pooled_config_payload(config, windows)
    fingerprint = base_results.build_run_fingerprint(
        ANALYSIS_VERSION,
        str(source["fingerprint"]),
        scientific_config,
        manifest,
        resolved.session_id,
    )
    return {
        "config": config,
        "windows": windows,
        "resolved_session": resolved,
        "trial_table": trial_table,
        "target_table": target_table,
        "dry_report": report,
        "input_manifest": manifest,
        "scientific_source": source,
        "scientific_config": scientific_config,
        "run_fingerprint": fingerprint,
    }


def _resource_envelope(identity: Mapping[str, object]) -> dict[str, int]:
    """Calculate exact pooled dimensions, requested cells, and major bytes.

    Parameters
    ----------
    identity : mapping[str, object]
        Internal bounded planning record returned by ``_collect_small_inputs``.

    Returns
    -------
    dict[str, int]
        Exact frozen resource-envelope fields. Counts are dimensionless and
        allocation fields are bytes.
    """
    config = identity["config"]
    windows = identity["windows"]
    target_table = identity["target_table"]
    report = identity["dry_report"]
    assert isinstance(config, PooledDecodingConfig)
    assert isinstance(target_table, pd.DataFrame)
    W = len(windows)
    C = len(config.condition_names)
    T = len(config.base_config.target_names)
    F = config.base_config.outer_fold_count
    Q = report.time_bin_count
    E = 2 * C - 1
    G = max(
        report.pfc_unit_count + report.hpc_unit_count,
        config.base_config.pfc_pc_count + config.base_config.hpc_pc_count,
    )
    family_names = tuple(
        base_config.TARGET_DEFINITION_BY_IDENTIFIER[name].family
        for name in config.base_config.target_names
    )
    categorical_targets = family_names.count("categorical")
    numerical_targets = family_names.count("numerical")
    models_per_target = W * C * F * 3 * 2
    categorical_fits = categorical_targets * models_per_target
    numerical_fits = numerical_targets * models_per_target
    transform_fits = W * C * T * F * 2
    metric_count_across_targets = 2 * categorical_targets + numerical_targets
    evaluation_cells = W * E * F * 3 * 2 * Q * metric_count_across_targets
    coefficient_cells = W * C * T * F * 3 * 2 * G
    stored_evaluation_cells = W * E * T * F * 3 * 2 * Q * 3
    stored_gap_cells = W * C * T * F * 3 * 2 * Q * 3
    planned_major_bytes = (
        report.tensor_allocation_bytes
        + coefficient_cells * (8 + 1)
        + stored_evaluation_cells * (8 + 1 + 2)
        + stored_gap_cells * (8 + 1 + 2)
    )
    return {
        "full_trial_count": int(len(target_table)),
        "tensor_trial_count": int(report.tensor_trial_count),
        "pfc_unit_count": int(report.pfc_unit_count),
        "hpc_unit_count": int(report.hpc_unit_count),
        "time_bin_count": int(Q),
        "training_window_count": int(W),
        "condition_count": int(C),
        "evaluation_pair_count": int(E),
        "target_count": int(T),
        "outer_fold_count": int(F),
        "coefficient_feature_capacity": int(G),
        "categorical_fit_count": int(categorical_fits),
        "numerical_fit_count": int(numerical_fits),
        "transform_fit_count": int(transform_fits),
        "evaluation_metric_cell_count": int(evaluation_cells),
        "tensor_allocation_bytes": int(report.tensor_allocation_bytes),
        "planned_major_array_bytes": int(planned_major_bytes),
    }


def _training_window_stack_shapes(
    config: PooledDecodingConfig,
    windows: Sequence[TrainingWindow],
    report: activity.ActivityDryRunReport,
) -> tuple[dict[str, object], ...]:
    """Report maximum requested model-input shapes for each training window.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated pooled and base settings containing requested component counts.
    windows : sequence[TrainingWindow]
        Canonical training windows whose bin indices define stacked time rows.
    report : activity.ActivityDryRunReport
        Bounded activity dimensions before condition, target, and fold filtering.

    Returns
    -------
    tuple[dict[str, object], ...]
        Canonical per-window records. Row counts are dimensionless maxima equal
        to ``tensor_trial_count * training_bin_count``. Each shape is
        ``(maximum_stacked_row, requested_feature)``; PCA widths are requested
        component counts and direct-unit widths are selected unit counts.
    """
    feature_counts = {
        "PFC": {
            "pca": config.base_config.pfc_pc_count,
            "units": report.pfc_unit_count,
        },
        "HPC": {
            "pca": config.base_config.hpc_pc_count,
            "units": report.hpc_unit_count,
        },
        "PFC+HPC": {
            "pca": config.base_config.pfc_pc_count + config.base_config.hpc_pc_count,
            "units": report.pfc_unit_count + report.hpc_unit_count,
        },
    }
    records: list[dict[str, object]] = []
    for window in windows:
        training_bin_count = int(window.bin_indices.size)
        maximum_row_count = report.tensor_trial_count * training_bin_count
        records.append(
            {
                "window_name": window.name,
                "training_bin_count": training_bin_count,
                "maximum_stacked_row_count": maximum_row_count,
                "maximum_requested_model_input_shapes": {
                    region_name: {
                        representation_name: (maximum_row_count, feature_count)
                        for representation_name, feature_count in representations.items()
                    }
                    for region_name, representations in feature_counts.items()
                },
            }
        )
    return tuple(records)


def plan_pooled_decoding_session(config_path: Path | str) -> dict[str, object]:
    """Inspect exact pooled work without loading spike arrays or fitting models.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Portable pooled overlay JSON path.

    Returns
    -------
    dict[str, object]
        Session/configuration coordinates, bounded per-window stack dimensions,
        exact resource envelope, input source sizes, fingerprints, and
        cleanliness text. Byte fields are bytes; dimensions are counts.
    """
    identity = _collect_small_inputs(config_path)
    config = identity["config"]
    windows = identity["windows"]
    resolved = identity["resolved_session"]
    report = identity["dry_report"]
    assert isinstance(config, PooledDecodingConfig)
    repository_root = Path(__file__).resolve().parents[4]
    try:
        results.validate_pooled_source_cleanliness(repository_root)
    except ValueError as error:
        cleanliness = str(error)
    else:
        cleanliness = "clean"
    envelope = _resource_envelope(identity)
    return {
        "config_path": str(Path(config_path).resolve()),
        "session_id": resolved.session_id,
        "analysis_version": ANALYSIS_VERSION,
        "training_window_names": tuple(window.name for window in windows),
        "condition_names": config.condition_names,
        "target_names": config.base_config.target_names,
        "training_window_stack_shapes": _training_window_stack_shapes(
            config,
            windows,
            report,
        ),
        "resource_envelope": envelope,
        "source_file_sizes_bytes": {
            str(path): int(size)
            for path, size in report.source_file_sizes_bytes.items()
        },
        "scientific_source_fingerprint": identity["scientific_source"]["fingerprint"],
        "run_fingerprint": identity["run_fingerprint"],
        "source_cleanliness": cleanliness,
    }


def _nonnegative_integer(value: object, field_name: str) -> int:
    """Validate one built-in nonnegative integer.

    Parameters
    ----------
    value : object
        Candidate dimensionless count or byte size.
    field_name : str
        Field label used in errors.

    Returns
    -------
    int
        Validated value.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer.")
    return value


def _nonnegative_seconds_or_none(value: object, field_name: str) -> float | None:
    """Validate one optional finite nonnegative duration.

    Parameters
    ----------
    value : object
        JSON number in seconds or ``None`` when a stage was not entered.
    field_name : str
        Field label used in errors.

    Returns
    -------
    float or None
        Validated seconds without changing null semantics.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be nonnegative seconds or null.")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{field_name} must be nonnegative seconds or null.")
    return converted


def _ordered_subset(values: Sequence[str], canonical: Sequence[str]) -> bool:
    """Return whether unique values preserve their canonical sequence order.

    Parameters
    ----------
    values, canonical : sequence[str]
        Candidate subset and complete canonical order.

    Returns
    -------
    bool
        True only for a duplicate-free ordered subset.
    """
    if len(set(values)) != len(values):
        return False
    positions = {value: index for index, value in enumerate(canonical)}
    if any(value not in positions for value in values):
        return False
    indices = [positions[value] for value in values]
    return indices == sorted(indices)


def validate_pooled_resource_usage_payload(
    payload: Mapping[str, object],
    *,
    canonical_checkpoint_keys: Sequence[str],
) -> dict[str, object]:
    """Validate the exact lean pooled resource sidecar contract.

    Parameters
    ----------
    payload : mapping[str, object]
        Decoded schema-1 pooled resource JSON.
    canonical_checkpoint_keys : sequence[str]
        Complete ``window::target`` identities in pipeline order.

    Returns
    -------
    dict[str, object]
        JSON round-tripped validated copy with unchanged values and units.
    """
    if not isinstance(payload, Mapping) or set(payload) != _RESOURCE_FIELDS:
        raise ValueError("Pooled resource payload has missing or unexpected fields.")
    if payload["schema_version"] != 1:
        raise ValueError("Pooled resource schema_version must be 1.")
    if payload["analysis_type"] != "pooled_task_variable_decoding":
        raise ValueError("Pooled resource analysis_type is invalid.")
    if payload["measurement_method"] != "resource.getrusage":
        raise ValueError("Pooled resource measurement method is invalid.")
    status = payload["status"]
    if status not in _RESOURCE_STATUSES:
        raise ValueError("Pooled resource status is invalid.")

    envelope = payload["resource_envelope"]
    if not isinstance(envelope, Mapping) or set(envelope) != _RESOURCE_ENVELOPE_FIELDS:
        raise ValueError("Pooled resource envelope has missing or unexpected fields.")
    for name, value in envelope.items():
        validated = _nonnegative_integer(value, f"resource_envelope.{name}")
        if name not in {
            "categorical_fit_count",
            "numerical_fit_count",
            "tensor_allocation_bytes",
            "planned_major_array_bytes",
        } and validated == 0:
            raise ValueError(f"resource_envelope.{name} must be positive.")
    if envelope["categorical_fit_count"] + envelope["numerical_fit_count"] <= 0:
        raise ValueError("Pooled resource envelope requires planned model fits.")

    for name in ("peak_rss_bytes", "checkpoint_size_bytes", "output_size_bytes"):
        _nonnegative_integer(payload[name], name)
    wall = _nonnegative_seconds_or_none(payload["wall_time_seconds"], "wall_time_seconds")
    user = _nonnegative_seconds_or_none(payload["user_cpu_seconds"], "user_cpu_seconds")
    system = _nonnegative_seconds_or_none(payload["system_cpu_seconds"], "system_cpu_seconds")
    if wall is None or user is None or system is None:
        raise ValueError("Process wall and CPU times cannot be null.")
    efficiency = payload["cpu_efficiency"]
    if wall == 0.0:
        if efficiency is not None:
            raise ValueError("cpu_efficiency must be null for zero wall time.")
    else:
        actual_efficiency = _nonnegative_seconds_or_none(efficiency, "cpu_efficiency")
        if actual_efficiency is None or not math.isclose(
            actual_efficiency,
            (user + system) / wall,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("cpu_efficiency does not reconcile with CPU and wall time.")

    canonical = tuple(canonical_checkpoint_keys)
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("Canonical checkpoint identities are invalid.")
    completed_value = payload["completed_checkpoint_keys"]
    restored_value = payload["restored_checkpoint_keys"]
    if not isinstance(completed_value, list) or not all(
        isinstance(value, str) for value in completed_value
    ):
        raise ValueError("completed checkpoint keys must be a JSON string list.")
    if not isinstance(restored_value, list) or not all(
        isinstance(value, str) for value in restored_value
    ):
        raise ValueError("restored checkpoint keys must be a JSON string list.")
    completed = tuple(completed_value)
    restored = tuple(restored_value)
    if not _ordered_subset(completed, canonical):
        raise ValueError("Completed checkpoint keys must preserve canonical order.")
    if status == "complete" and completed != canonical:
        raise ValueError("Complete resource payload requires every canonical checkpoint.")
    if not _ordered_subset(restored, completed):
        raise ValueError("Restored checkpoint keys must be an ordered completed subset.")

    timings = payload["stage_timing_seconds"]
    if not isinstance(timings, Mapping) or set(timings) != _STAGE_FIELDS:
        raise ValueError("Pooled stage timings have missing or unexpected fields.")
    for name, value in timings.items():
        timing = _nonnegative_seconds_or_none(value, f"stage_timing_seconds.{name}")
        if status == "complete" and timing is None:
            raise ValueError("Complete pooled resource timings cannot be null.")

    counts = payload["result_counts"]
    if not isinstance(counts, Mapping) or set(counts) != _RESULT_COUNT_FIELDS:
        raise ValueError("Pooled result counts have missing or unexpected fields.")
    for name, value in counts.items():
        _nonnegative_integer(value, f"result_counts.{name}")
    expected_requested = {
        "requested_transform_cells": envelope["transform_fit_count"],
        "requested_model_cells": (
            envelope["categorical_fit_count"] + envelope["numerical_fit_count"]
        ),
        "requested_evaluation_metric_cells": envelope["evaluation_metric_cell_count"],
    }
    for requested_name, expected in expected_requested.items():
        if counts[requested_name] != expected:
            raise ValueError("Requested pooled result counts do not reconcile with envelope.")
        stem = requested_name.removeprefix("requested_").removesuffix("_cells")
        valid_name = f"valid_{stem}_cells"
        unavailable_name = f"unavailable_{stem}_cells"
        observed = counts[valid_name] + counts[unavailable_name]
        if observed > expected or (status == "complete" and observed != expected):
            raise ValueError("Pooled result count reconciliation failed.")
    return json.loads(_canonical_json(payload))


def measure_pooled_output_size_bytes(run_directory: Path | str) -> int:
    """Measure current run bytes excluding the self-referential resource sidecar.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Existing prepared run directory.

    Returns
    -------
    int
        Sum of regular-file sizes in bytes, excluding ``resource_usage.json``
        and sibling atomic temporaries whose names start with that basename.
    """
    total = 0
    for path in Path(run_directory).rglob("*"):
        if not path.is_file():
            continue
        if path.name == "resource_usage.json" or path.name.startswith(
            ".resource_usage.json."
        ):
            continue
        total += path.stat().st_size
    return total


def _canonical_checkpoint_keys(
    config: PooledDecodingConfig,
    windows: Sequence[TrainingWindow],
) -> tuple[str, ...]:
    """Build canonical window-major, target-minor checkpoint identities.

    Parameters
    ----------
    config : PooledDecodingConfig
        Validated target order.
    windows : sequence[TrainingWindow]
        Canonical selected window order.

    Returns
    -------
    tuple[str, ...]
        Complete ``window::target`` identities.
    """
    return tuple(
        f"{window.name}::{target}"
        for window in windows
        for target in config.base_config.target_names
    )


def _checkpoint_path(run_directory: Path, checkpoint_key: str) -> Path:
    """Map one validated checkpoint identity to its durable NPZ path.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory.
    checkpoint_key : str
        Canonical ``window::target`` identity.

    Returns
    -------
    pathlib.Path
        Path below the run's ``checkpoints`` directory.
    """
    window, target = checkpoint_key.split("::", maxsplit=1)
    return run_directory / "checkpoints" / f"{window}__{target}.npz"


def _empty_result_counts(envelope: Mapping[str, int]) -> dict[str, int]:
    """Build requested counts with no completed scientific work.

    Parameters
    ----------
    envelope : mapping[str, int]
        Exact frozen resource dimensions.

    Returns
    -------
    dict[str, int]
        Exact result-count vocabulary; valid/unavailable observations are zero.
    """
    return {
        "requested_transform_cells": envelope["transform_fit_count"],
        "valid_transform_cells": 0,
        "unavailable_transform_cells": 0,
        "requested_model_cells": (
            envelope["categorical_fit_count"] + envelope["numerical_fit_count"]
        ),
        "valid_model_cells": 0,
        "unavailable_model_cells": 0,
        "requested_evaluation_metric_cells": envelope["evaluation_metric_cell_count"],
        "valid_evaluation_metric_cells": 0,
        "unavailable_evaluation_metric_cells": 0,
    }


def _complete_result_counts(saved: Mapping[str, object]) -> dict[str, int]:
    """Count valid/unavailable cells in one validated final pooled result.

    Parameters
    ----------
    saved : mapping[str, object]
        Value returned by :func:`results.load_pooled_run`.

    Returns
    -------
    dict[str, int]
        Exact requested/valid/unavailable result counts. Metric-family
        not-applicable padding is excluded.
    """
    arrays = saved["arrays"]
    transform = arrays["transform_status"]
    model = arrays["model_status"]
    evaluation = arrays["evaluation_status"]
    return {
        "requested_transform_cells": int(np.count_nonzero(transform != 0)),
        "valid_transform_cells": int(np.count_nonzero(transform == 1)),
        "unavailable_transform_cells": int(np.count_nonzero(transform == 2)),
        "requested_model_cells": int(np.count_nonzero(model != 0)),
        "valid_model_cells": int(np.count_nonzero(model == 1)),
        "unavailable_model_cells": int(np.count_nonzero(model == 2)),
        "requested_evaluation_metric_cells": int(np.count_nonzero(evaluation != 0)),
        "valid_evaluation_metric_cells": int(np.count_nonzero(evaluation == 1)),
        "unavailable_evaluation_metric_cells": int(np.count_nonzero(evaluation == 2)),
    }


def _checkpoint_result_counts(
    run_directory: Path,
    *,
    config: PooledDecodingConfig,
    canonical_keys: Sequence[str],
    completed_keys: Sequence[str],
    expected_fingerprint: str,
    envelope: Mapping[str, int],
) -> dict[str, int]:
    """Count exact scientific cells represented by validated checkpoints.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory containing immutable checkpoint NPZ files.
    config : PooledDecodingConfig
        Condition, target-family, and fold settings.
    canonical_keys, completed_keys : sequence[str]
        Complete order and validated completed subset.
    expected_fingerprint : str
        Exact checkpoint scientific identity.
    envelope : mapping[str, int]
        Immutable requested aggregate counts.

    Returns
    -------
    dict[str, int]
        Requested full-run counts and exact valid/unavailable observations for
        completed checkpoint cells. Not-applicable metric padding is excluded.
    """
    counts = _empty_result_counts(envelope)
    C = len(config.condition_names)
    F = config.base_config.outer_fold_count
    E = 2 * C - 1
    Q = envelope["time_bin_count"]
    completed_set = set(completed_keys)
    for key in canonical_keys:
        if key not in completed_set:
            continue
        result = results.load_pooled_checkpoint(
            _checkpoint_path(run_directory, key),
            expected_checkpoint_key=key,
            expected_run_fingerprint=expected_fingerprint,
        )
        transform_requested = C * F * 2
        model_requested = C * F * 3 * 2
        metric_count = 2 if result.target_family == "categorical" else 1
        evaluation_requested = E * F * 3 * 2 * Q * metric_count
        if result.unavailable_reason is not None:
            counts["unavailable_transform_cells"] += transform_requested
            counts["unavailable_model_cells"] += model_requested
            counts["unavailable_evaluation_metric_cells"] += evaluation_requested
            continue
        valid_transforms = sum(
            int(transforms.pfc_transform.is_available)
            + int(transforms.hpc_transform.is_available)
            for transforms in result.transforms.values()
        )
        valid_models = sum(
            fitted.record.is_valid for fitted in result.models.values()
        )
        valid_evaluations = sum(
            int(np.count_nonzero(evaluation.status == "valid"))
            for evaluation in result.evaluations.values()
        )
        counts["valid_transform_cells"] += int(valid_transforms)
        counts["unavailable_transform_cells"] += transform_requested - int(
            valid_transforms
        )
        counts["valid_model_cells"] += int(valid_models)
        counts["unavailable_model_cells"] += model_requested - int(valid_models)
        counts["valid_evaluation_metric_cells"] += int(valid_evaluations)
        counts["unavailable_evaluation_metric_cells"] += evaluation_requested - int(
            valid_evaluations
        )
    return counts


def _resource_snapshot(
    *,
    status: str,
    envelope: Mapping[str, int],
    canonical_keys: Sequence[str],
    completed_keys: Sequence[str],
    restored_keys: Sequence[str],
    stage_timings: Mapping[str, float | None],
    result_counts: Mapping[str, int],
    run_directory: Path,
    wall_start: float,
    usage_start: object,
) -> dict[str, object]:
    """Build one current-process lean resource measurement.

    Parameters
    ----------
    status : {"running", "failed", "interrupted", "complete"}
        Current durable lifecycle outcome.
    envelope : mapping[str, int]
        Immutable dry-run dimensions and byte plan.
    canonical_keys, completed_keys, restored_keys : sequence[str]
        Complete and observed checkpoint identities in pipeline order.
    stage_timings : mapping[str, float or None]
        Current-invocation aggregate seconds for the four declared stages.
    result_counts : mapping[str, int]
        Requested and observed validated cell counts.
    run_directory : pathlib.Path
        Prepared run whose checkpoint/output bytes are measured.
    wall_start : float
        ``time.monotonic`` value at invocation start.
    usage_start : resource.struct_rusage
        Process usage snapshot at invocation start.

    Returns
    -------
    dict[str, object]
        Validated exact schema-1 resource payload.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    wall = max(0.0, time.monotonic() - wall_start)
    user = max(0.0, float(usage.ru_utime - usage_start.ru_utime))
    system = max(0.0, float(usage.ru_stime - usage_start.ru_stime))
    checkpoint_bytes = sum(
        _checkpoint_path(run_directory, key).stat().st_size
        for key in completed_keys
        if _checkpoint_path(run_directory, key).is_file()
    )
    measured_output_bytes = measure_pooled_output_size_bytes(run_directory)
    guard_path = run_directory / "execution_guard.json"
    if guard_path.is_file():
        measured_output_bytes -= guard_path.stat().st_size
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_type": "pooled_task_variable_decoding",
        "measurement_method": "resource.getrusage",
        "status": status,
        "peak_rss_bytes": normalize_peak_rss_bytes(int(usage.ru_maxrss), sys.platform),
        "wall_time_seconds": wall,
        "user_cpu_seconds": user,
        "system_cpu_seconds": system,
        "cpu_efficiency": None if wall == 0.0 else (user + system) / wall,
        "resource_envelope": dict(envelope),
        "completed_checkpoint_keys": list(completed_keys),
        "restored_checkpoint_keys": list(restored_keys),
        "stage_timing_seconds": dict(stage_timings),
        "result_counts": dict(result_counts),
        "checkpoint_size_bytes": checkpoint_bytes,
        "output_size_bytes": measured_output_bytes,
    }
    return validate_pooled_resource_usage_payload(
        payload,
        canonical_checkpoint_keys=canonical_keys,
    )


def _initial_resource_payload(
    envelope: Mapping[str, int],
    canonical_keys: Sequence[str],
    run_directory: Path,
) -> dict[str, object]:
    """Build preparation-time zeroed resource evidence.

    Parameters
    ----------
    envelope : mapping[str, int]
        Immutable exact dry-run dimensions.
    canonical_keys : sequence[str]
        Complete checkpoint order for the prepared run.
    run_directory : pathlib.Path
        Run directory after all non-resource preparation artifacts exist.

    Returns
    -------
    dict[str, object]
        Valid running payload with no attributed execution work.
    """
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_type": "pooled_task_variable_decoding",
        "measurement_method": "resource.getrusage",
        "status": "running",
        "peak_rss_bytes": 0,
        "wall_time_seconds": 0.0,
        "user_cpu_seconds": 0.0,
        "system_cpu_seconds": 0.0,
        "cpu_efficiency": None,
        "resource_envelope": dict(envelope),
        "completed_checkpoint_keys": [],
        "restored_checkpoint_keys": [],
        "stage_timing_seconds": {name: None for name in sorted(_STAGE_FIELDS)},
        "result_counts": _empty_result_counts(envelope),
        "checkpoint_size_bytes": 0,
        "output_size_bytes": measure_pooled_output_size_bytes(run_directory),
    }
    return validate_pooled_resource_usage_payload(
        payload,
        canonical_checkpoint_keys=canonical_keys,
    )


def _new_run_directory(output_root: Path) -> Path:
    """Create one exclusive timestamped immutable run directory.

    Parameters
    ----------
    output_root : pathlib.Path
        Session-contained output parent.

    Returns
    -------
    pathlib.Path
        Newly created run directory with a collision-safe suffix.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for suffix in range(1000):
        name = f"{_RUN_PREFIX}{timestamp}"
        if suffix:
            name = f"{name}-{suffix:02d}"
        candidate = output_root / name
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not allocate a unique pooled run directory.")


def _snapshot_sources(run_directory: Path) -> dict[str, dict[str, object]]:
    """Copy and hash the three execution/reporting modules.

    Parameters
    ----------
    run_directory : pathlib.Path
        Newly created run directory.

    Returns
    -------
    dict[str, dict[str, object]]
        Filename-keyed SHA-256 and byte-size records.
    """
    source_directory = Path(__file__).resolve().parent
    records: dict[str, dict[str, object]] = {}
    for filename in ("run_session.py", "pipeline.py", "plotting.py"):
        source = source_directory / filename
        content = source.read_bytes()
        _atomic_bytes(run_directory / filename, content)
        records[filename] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    return records


def _validate_source_snapshots(run_directory: Path) -> None:
    """Verify saved execution snapshots against preparation hashes and bytes.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory containing ``execution.json`` and snapshots.

    Returns
    -------
    None
        Raises before execution/resume/reentry when any snapshot drifted.
    """
    execution = _read_json(run_directory / "execution.json")
    records = execution.get("source_snapshots")
    if not isinstance(records, Mapping) or set(records) != {
        "run_session.py",
        "pipeline.py",
        "plotting.py",
    }:
        raise ValueError("Prepared source snapshot records are invalid.")
    for filename, value in records.items():
        if not isinstance(value, Mapping) or set(value) != {"sha256", "size_bytes"}:
            raise ValueError("Prepared source snapshot record has invalid fields.")
        path = run_directory / filename
        if (
            not path.is_file()
            or path.stat().st_size != value["size_bytes"]
            or sha256_file(path) != value["sha256"]
        ):
            raise ValueError(f"Prepared source snapshot identity changed: {filename}")


def _append_log(run_directory: Path, message: str) -> None:
    """Append and flush one durable pooled lifecycle event.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory.
    message : str
        Single human-readable event line.

    Returns
    -------
    None
        Adds a UTC-prefixed UTF-8 line to ``run.log``.
    """
    with (run_directory / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_now()} {message}\n")
        stream.flush()
        os.fsync(stream.fileno())


def prepare_pooled_decoding_run(
    config_path: Path | str,
    *,
    execution_mode: str = "foreground",
) -> Path:
    """Create one immutable pooled identity and resumable lifecycle directory.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Portable pooled overlay JSON path.
    execution_mode : {"foreground", "detached", "slurm"}, default="foreground"
        Execution provenance recorded with the immutable prepared run. This
        changes orchestration only; scientific settings and array axes are
        unchanged.

    Returns
    -------
    pathlib.Path
        New session-contained run directory. No spike arrays are loaded and no
        estimator is fit during preparation.
    """
    if execution_mode not in {"foreground", "detached", "slurm"}:
        raise ValueError(f"Unsupported pooled execution mode: {execution_mode!r}")
    identity = _collect_small_inputs(config_path)
    config = identity["config"]
    windows = identity["windows"]
    assert isinstance(config, PooledDecodingConfig)
    repository_root = Path(__file__).resolve().parents[4]
    results.validate_pooled_source_cleanliness(repository_root)
    run_directory = _new_run_directory(config.base_config.output_root)
    try:
        _atomic_bytes(
            run_directory / "pooled_config.input.json",
            Path(config_path).resolve().read_bytes(),
        )
        _atomic_bytes(
            run_directory / "base_config.input.json",
            config.base_config_path.read_bytes(),
        )
        _write_json(run_directory / "config.json", identity["scientific_config"])
        _write_json(run_directory / "input_manifest.json", identity["input_manifest"])
        _write_json(run_directory / "scientific_source.json", identity["scientific_source"])
        snapshot_records = _snapshot_sources(run_directory)
        _write_json(
            run_directory / "execution.json",
            {
                "mode": execution_mode,
                "prepared_at": _utc_now(),
                "session_root": str(config.base_config.session_root),
                "run_fingerprint": identity["run_fingerprint"],
                "source_snapshots": snapshot_records,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "runtime_versions": results.collect_software_versions(),
                "thread_limits": _thread_limit_mapping(),
            },
        )
        _write_json(
            run_directory / "run_state.json",
            {
                "lifecycle": "prepared",
                "final_results_published": False,
                "error": None,
                "updated_at": _utc_now(),
            },
        )
        module = "src.neural_analysis.task_decoding.pooled.run_session"
        quoted = shlex.quote(str(run_directory))
        commands = {
            "run_command.txt": f"uv run python -m {module} resume --run-directory {quoted}\n",
            "resume_command.txt": f"uv run python -m {module} resume --run-directory {quoted}\n",
            "status_command.txt": f"uv run python -m {module} status --run-directory {quoted}\n",
        }
        for filename, command in commands.items():
            _atomic_bytes(run_directory / filename, command.encode("utf-8"))
        _atomic_bytes(run_directory / "run.log", b"")
        _append_log(run_directory, "Prepared pooled task-variable decoding run.")
        envelope = _resource_envelope(identity)
        canonical = _canonical_checkpoint_keys(config, windows)
        _write_json(
            run_directory / "resource_usage.json",
            _initial_resource_payload(envelope, canonical, run_directory),
        )
    except BaseException:
        # The exclusive directory is intentionally retained as failure evidence.
        raise
    return run_directory


def _validate_prepared_identity(
    run_directory: Path,
) -> tuple[PooledDecodingConfig, tuple[TrainingWindow, ...], str]:
    """Reconstruct and revalidate live inputs/source before checkpoint access.

    Parameters
    ----------
    run_directory : pathlib.Path
        Existing prepared pooled run directory.

    Returns
    -------
    tuple[PooledDecodingConfig, tuple[TrainingWindow, ...], str]
        Canonical reconstructed settings, windows, and expected run fingerprint.
    """
    _validate_source_snapshots(run_directory)
    execution = _read_json(run_directory / "execution.json")
    if execution.get("runtime_versions") != results.collect_software_versions():
        raise ValueError("Prepared runtime-version mapping differs from current runtime.")
    if execution.get("thread_limits") != _thread_limit_mapping():
        raise ValueError("Prepared numerical thread limits differ from current runtime.")
    config, windows = load_prepared_pooled_decoding_config(run_directory)
    task_config = config.base_config
    resolved = resolve_session_metadata(
        load_session_metadata(task_config.session_metadata_path),
        task_config.session_metadata_path,
    )
    live_manifest = _build_live_manifest(config, resolved)
    if live_manifest != _read_json(run_directory / "input_manifest.json"):
        raise ValueError("Live input manifest changed from the prepared identity.")
    repository_root = Path(__file__).resolve().parents[4]
    live_source = results.pooled_scientific_source_fingerprint(repository_root)
    if live_source != _read_json(run_directory / "scientific_source.json"):
        raise ValueError("Live scientific source identity changed after preparation.")
    expected_fingerprint = base_results.build_run_fingerprint(
        ANALYSIS_VERSION,
        str(live_source["fingerprint"]),
        scientific_pooled_config_payload(config, windows),
        live_manifest,
        resolved.session_id,
    )
    if execution.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("Prepared run fingerprint identity does not reconcile.")
    return config, windows, expected_fingerprint


def _validated_checkpoint_keys(
    run_directory: Path,
    canonical_keys: Sequence[str],
    expected_fingerprint: str,
) -> tuple[str, ...]:
    """Validate and list existing checkpoint files in canonical order.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory.
    canonical_keys : sequence[str]
        Complete window-major/target-minor identities.
    expected_fingerprint : str
        Exact scientific run fingerprint required by every checkpoint.

    Returns
    -------
    tuple[str, ...]
        Canonically ordered identities whose immutable checkpoint files exist
        and validate completely.
    """
    output: list[str] = []
    for key in canonical_keys:
        path = _checkpoint_path(run_directory, key)
        if not path.exists():
            continue
        results.load_pooled_checkpoint(
            path,
            expected_checkpoint_key=key,
            expected_run_fingerprint=expected_fingerprint,
        )
        output.append(key)
    checkpoint_directory = run_directory / "checkpoints"
    if checkpoint_directory.exists():
        expected_paths = {_checkpoint_path(run_directory, key) for key in canonical_keys}
        unexpected = set(checkpoint_directory.glob("*.npz")).difference(expected_paths)
        if unexpected:
            raise ValueError("Checkpoint directory contains an unexpected pooled cell.")
    return tuple(output)


def _write_summary(
    run_directory: Path,
    *,
    config: PooledDecodingConfig,
    windows: Sequence[TrainingWindow],
    session_id: str,
    elapsed_seconds: float,
) -> None:
    """Publish the required human-readable scientific/run summary.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory.
    config : PooledDecodingConfig
        Validated pooled/base settings.
    windows : sequence[TrainingWindow]
        Canonical selected training windows in seconds.
    session_id : str
        Stable session metadata identifier.
    elapsed_seconds : float
        Current execution invocation wall time in seconds.

    Returns
    -------
    None
        Atomically publishes ``summary.md`` without changing scientific arrays.
    """
    window_text = ", ".join(
        f"{window.name} [{window.start_s:g}, {window.stop_s:g}) s" for window in windows
    )
    text = f"""# Pooled task-variable decoding summary

## Goal

Measure temporal stability and condition transfer of complete pooled-window decoding pipelines.

## Session

- Session: `{session_id}`
- Analysis version: `{ANALYSIS_VERSION}`
- Immutable run directory: `{run_directory}`

## Parameters

- Training windows: {window_text}
- Conditions: {", ".join(config.condition_names)}
- Targets: {", ".join(config.base_config.target_names)}
- Outer folds: {config.base_config.outer_fold_count}, grouped by behavioral block
- Evaluation time units: seconds from choice

## Scripts

- `run_session.py`
- `pipeline.py`
- `plotting.py`

## Runtime evidence

- Elapsed time for this invocation: {elapsed_seconds:.6f} seconds
- Warnings: none recorded by the lifecycle layer; model-level convergence states remain in `results.npz`.
- Errors: none at terminal publication.

## Scientific limits

Native and transfer scores use the same destination held-out trials. They are complete pipelines with separately trained preprocessing, so their gap cannot isolate estimator weights. Fold variation is not a confidence interval. Out-of-window performance describes temporal generalization, and transfer failure alone does not prove remapping. Comparisons with independently cross-validated per-time-bin analyses can also differ because their transform-training windows differ.
"""
    _atomic_bytes(run_directory / "summary.md", text.encode("utf-8"))


def _validate_complete_artifacts(
    run_directory: Path,
    *,
    config: PooledDecodingConfig,
    windows: Sequence[TrainingWindow],
    expected_fingerprint: str,
) -> None:
    """Validate every terminal pooled artifact without mutating the run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory claiming terminal completion.
    config : PooledDecodingConfig
        Reconstructed target order.
    windows : sequence[TrainingWindow]
        Reconstructed selected windows.
    expected_fingerprint : str
        Exact final-result scientific identity.

    Returns
    -------
    None
        Returns only when result, summary, figures, checkpoints, and resource
        evidence are complete and mutually consistent.
    """
    results.load_pooled_run(
        run_directory,
        expected_run_fingerprint=expected_fingerprint,
    )
    summary = run_directory / "summary.md"
    if not summary.is_file() or not summary.read_text(encoding="utf-8").strip():
        raise ValueError("Completed pooled run is missing its summary.")
    expected_figures = {
        f"{kind}__{window.name}__{target}.png"
        for window in windows
        for target in config.base_config.target_names
        for kind in ("native", "transfer", "gap")
    }
    figure_directory = run_directory / "figures"
    actual_figures = {path.name for path in figure_directory.glob("*.png")}
    if actual_figures != expected_figures:
        raise ValueError("Completed pooled run has missing or unexpected figures.")
    for filename in expected_figures:
        if not (figure_directory / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Completed pooled run contains an invalid PNG.")
    canonical = _canonical_checkpoint_keys(config, windows)
    completed = _validated_checkpoint_keys(
        run_directory,
        canonical,
        expected_fingerprint,
    )
    if completed != canonical:
        raise ValueError("Completed pooled run is missing a validated checkpoint.")
    resource_payload = _read_json(run_directory / "resource_usage.json")
    validate_pooled_resource_usage_payload(
        resource_payload,
        canonical_checkpoint_keys=canonical,
    )
    if resource_payload["status"] != "complete":
        raise ValueError("Completed pooled run lacks complete resource evidence.")
    observed_output_bytes = measure_pooled_output_size_bytes(run_directory)
    guard_path = run_directory / "execution_guard.json"
    if guard_path.is_file():
        observed_output_bytes -= guard_path.stat().st_size
    if resource_payload["output_size_bytes"] != observed_output_bytes:
        raise ValueError("Completed pooled output byte measurement does not reconcile.")


def _claim_execution(run_directory: Path) -> int:
    """Claim one exclusive local writer using an atomic guard file.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory.

    Returns
    -------
    int
        Open guard file descriptor retained until execution exits.
    """
    path = run_directory / "execution_guard.json"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("Pooled run already has an active execution owner.") from error
    content = (_canonical_json({"pid": os.getpid(), "claimed_at": _utc_now()}) + "\n").encode()
    os.write(descriptor, content)
    os.fsync(descriptor)
    return descriptor


def _release_execution(run_directory: Path, descriptor: int) -> None:
    """Release the current process's local execution guard.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory.
    descriptor : int
        File descriptor returned by ``_claim_execution``.

    Returns
    -------
    None
        Closes the descriptor and removes the guard file.
    """
    os.close(descriptor)
    guard = run_directory / "execution_guard.json"
    if guard.exists():
        guard.unlink()


def run_prepared_pooled_decoding(run_directory: Path | str) -> None:
    """Execute/resume one prepared pooled run and publish terminal reporting.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact immutable directory returned by ``prepare_pooled_decoding_run``.

    Returns
    -------
    None
        Scientific output is published to the run directory. Interruption and
        failure preserve validated checkpoints and raise to the caller.
    """
    directory = Path(run_directory).resolve()
    initial_state = _read_json(directory / "run_state.json")
    if initial_state.get("lifecycle") == "complete":
        if initial_state.get("final_results_published") is not True:
            raise ValueError("Complete pooled state lacks final publication truth.")
        config, windows, expected_fingerprint = _validate_prepared_identity(directory)
        _validate_complete_artifacts(
            directory,
            config=config,
            windows=windows,
            expected_fingerprint=expected_fingerprint,
        )
        return
    wall_start = time.monotonic()
    usage_start = resource.getrusage(resource.RUSAGE_SELF)
    stage_timings: dict[str, float | None] = {
        "input_loading": None,
        "modeling": None,
        "final_assembly": None,
        "reporting": None,
    }
    config: PooledDecodingConfig | None = None
    windows: tuple[TrainingWindow, ...] = ()
    canonical: tuple[str, ...] = ()
    restored: tuple[str, ...] = ()
    completed: tuple[str, ...] = ()
    envelope: dict[str, int] | None = None
    descriptor: int | None = None
    ownership_denied = False
    modeling_started: float | None = None
    assembly_started: float | None = None
    reporting_started: float | None = None
    input_started = time.monotonic()
    try:
        config, windows, expected_fingerprint = _validate_prepared_identity(directory)
        canonical = _canonical_checkpoint_keys(config, windows)
        saved_resource = _read_json(directory / "resource_usage.json")
        envelope_value = saved_resource.get("resource_envelope")
        if not isinstance(envelope_value, Mapping):
            raise ValueError("Prepared pooled resource envelope is invalid.")
        envelope = {name: int(value) for name, value in envelope_value.items()}
        validate_pooled_resource_usage_payload(
            saved_resource,
            canonical_checkpoint_keys=canonical,
        )
        if os.environ.get("SLURM_JOB_ID"):
            from src.neural_analysis.task_decoding.pooled import slurm

            tensor_bytes = envelope["tensor_allocation_bytes"]
            if tensor_bytes * 2 > slurm.effective_memory_budget_bytes():
                raise ValueError(
                    "Pooled tensor allocation exceeds 50% of the effective "
                    "Slurm memory budget."
                )
        restored = _validated_checkpoint_keys(
            directory,
            canonical,
            expected_fingerprint,
        )
        stage_timings["input_loading"] = time.monotonic() - input_started
        try:
            descriptor = _claim_execution(directory)
        except RuntimeError:
            ownership_denied = True
            raise
        _write_json(
            directory / "run_state.json",
            {
                "lifecycle": "running",
                "final_results_published": False,
                "error": None,
                "updated_at": _utc_now(),
            },
        )
        _append_log(directory, "Started or resumed pooled computation.")

        modeling_started = time.monotonic()
        if (directory / "results.npz").exists():
            results.load_pooled_run(
                directory,
                expected_run_fingerprint=expected_fingerprint,
            )
        else:
            compute_or_resume_pooled_results(config, windows, directory)
        stage_timings["modeling"] = time.monotonic() - modeling_started
        completed = _validated_checkpoint_keys(
            directory,
            canonical,
            expected_fingerprint,
        )

        assembly_started = time.monotonic()
        saved = results.load_pooled_run(
            directory,
            expected_run_fingerprint=expected_fingerprint,
        )
        result_counts = _complete_result_counts(saved)
        stage_timings["final_assembly"] = time.monotonic() - assembly_started

        reporting_started = time.monotonic()
        from src.neural_analysis.task_decoding.pooled import plotting

        plotting.save_pooled_decoding_figures(directory)
        resolved = resolve_session_metadata(
            load_session_metadata(config.base_config.session_metadata_path),
            config.base_config.session_metadata_path,
        )
        _write_summary(
            directory,
            config=config,
            windows=windows,
            session_id=resolved.session_id,
            elapsed_seconds=time.monotonic() - wall_start,
        )
        stage_timings["reporting"] = time.monotonic() - reporting_started
        _append_log(directory, "Completed pooled result, summary, and required figures.")
        resource_payload = _resource_snapshot(
            status="complete",
            envelope=envelope,
            canonical_keys=canonical,
            completed_keys=completed,
            restored_keys=restored,
            stage_timings=stage_timings,
            result_counts=result_counts,
            run_directory=directory,
            wall_start=wall_start,
            usage_start=usage_start,
        )
        final_state = {
            "lifecycle": "complete",
            "final_results_published": True,
            "error": None,
            "updated_at": _utc_now(),
        }
        current_state_path = directory / "run_state.json"
        final_state_size = len((_canonical_json(final_state) + "\n").encode("utf-8"))
        resource_payload["output_size_bytes"] = (
            measure_pooled_output_size_bytes(directory)
            - current_state_path.stat().st_size
            - (directory / "execution_guard.json").stat().st_size
            + final_state_size
        )
        _write_json(directory / "resource_usage.json", resource_payload)
        _write_json(current_state_path, final_state)
        _validate_complete_artifacts(
            directory,
            config=config,
            windows=windows,
            expected_fingerprint=expected_fingerprint,
        )
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if stage_timings["input_loading"] is None:
            stage_timings["input_loading"] = time.monotonic() - input_started
        if modeling_started is not None and stage_timings["modeling"] is None:
            stage_timings["modeling"] = time.monotonic() - modeling_started
        if assembly_started is not None and stage_timings["final_assembly"] is None:
            stage_timings["final_assembly"] = time.monotonic() - assembly_started
        if reporting_started is not None and stage_timings["reporting"] is None:
            stage_timings["reporting"] = time.monotonic() - reporting_started
        if not ownership_denied and (config is None or not canonical or envelope is None):
            try:
                recovered_config, recovered_windows = (
                    load_prepared_pooled_decoding_config(directory)
                )
                recovered_canonical = _canonical_checkpoint_keys(
                    recovered_config,
                    recovered_windows,
                )
                saved_resource = _read_json(directory / "resource_usage.json")
                validate_pooled_resource_usage_payload(
                    saved_resource,
                    canonical_checkpoint_keys=recovered_canonical,
                )
                envelope_value = saved_resource["resource_envelope"]
                assert isinstance(envelope_value, Mapping)
                config = recovered_config
                windows = recovered_windows
                canonical = recovered_canonical
                envelope = {
                    name: int(value) for name, value in envelope_value.items()
                }
            except BaseException:
                pass
        if (
            not ownership_denied
            and config is not None
            and canonical
            and envelope is not None
        ):
            try:
                completed = _validated_checkpoint_keys(
                    directory,
                    canonical,
                    _read_json(directory / "execution.json")["run_fingerprint"],
                )
            except BaseException:
                completed = ()
            try:
                partial = _resource_snapshot(
                    status=status,
                    envelope=envelope,
                    canonical_keys=canonical,
                    completed_keys=completed,
                    restored_keys=restored,
                    stage_timings=stage_timings,
                    result_counts=_checkpoint_result_counts(
                        directory,
                        config=config,
                        canonical_keys=canonical,
                        completed_keys=completed,
                        expected_fingerprint=_read_json(
                            directory / "execution.json"
                        )["run_fingerprint"],
                        envelope=envelope,
                    ),
                    run_directory=directory,
                    wall_start=wall_start,
                    usage_start=usage_start,
                )
                _write_json(directory / "resource_usage.json", partial)
            except BaseException:
                pass
        if not ownership_denied and (directory / "run_state.json").is_file():
            _write_json(
                directory / "run_state.json",
                {
                    "lifecycle": status,
                    "final_results_published": False,
                    "error": f"{type(error).__name__}: {error}",
                    "updated_at": _utc_now(),
                },
            )
        if not ownership_denied and (directory / "run.log").is_file():
            _append_log(directory, f"{status.title()}: {type(error).__name__}: {error}")
        raise
    finally:
        if descriptor is not None:
            _release_execution(directory, descriptor)


def inspect_pooled_decoding_status(run_directory: Path | str) -> dict[str, object]:
    """Read bounded pooled lifecycle evidence without changing any artifact.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact prepared pooled run directory.

    Returns
    -------
    dict[str, object]
        Durable lifecycle/publication fields plus completion validation status,
        checkpoint counts, and resource status. No spike array is loaded.
    """
    directory = Path(run_directory).resolve()
    state = _read_json(directory / "run_state.json")
    resource_payload = _read_json(directory / "resource_usage.json")
    status: dict[str, object] = dict(state)
    status["run_directory"] = str(directory)
    status["resource_status"] = resource_payload.get("status")
    status["completed_checkpoint_count"] = len(
        resource_payload.get("completed_checkpoint_keys", [])
    )
    status["result_valid"] = False
    if state.get("lifecycle") == "complete":
        try:
            config, windows, fingerprint = _validate_prepared_identity(directory)
            _validate_complete_artifacts(
                directory,
                config=config,
                windows=windows,
                expected_fingerprint=fingerprint,
            )
        except (OSError, ValueError):
            status["result_valid"] = False
        else:
            status["result_valid"] = True
    return status
