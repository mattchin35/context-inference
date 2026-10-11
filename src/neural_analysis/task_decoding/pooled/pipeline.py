"""Synchronous preparation, checkpoint resume, and result assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)
from src.neural_analysis.task_decoding import activity, conditions, modeling, targets
from src.neural_analysis.task_decoding import config as base_config
from src.neural_analysis.task_decoding import results as base_results
from src.neural_analysis.task_decoding.pooled import decoding, results
from src.neural_analysis.task_decoding.pooled.config import (
    ANALYSIS_VERSION,
    PooledDecodingConfig,
    TrainingWindow,
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
