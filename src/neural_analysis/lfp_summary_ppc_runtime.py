"""Restartable PPC execution with serial and site-sequential worker paths.

This module owns work-only checkpoints. It never writes a final component or
manifest, and never keeps a whole-session shuffle tensor in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from hashlib import sha256
import json
import multiprocessing
import os
from pathlib import Path
import time
from typing import Callable, Iterator, Mapping, Sequence
from types import SimpleNamespace
import zipfile

import numpy as np

from src.neural_analysis.spike_lfp import ppc as spike_lfp_summary
from src.neural_analysis.spike_lfp.ppc_kernel import (
    aggregate_observed_trial_segmented_ppc_statistics,
    build_source_trial_spike_geometry,
    compose_observed_segmented_ppc_metrics,
    compute_observed_trial_segmented_ppc_statistics,
    compute_selected_observed_trial_segmented_ppc_statistics,
    compute_segmented_edge_statistics,
    estimate_segmented_kernel_allocation,
    reduce_segmented_schedule_to_ppc,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    PPCExecutionConfig,
    ProgressEvent,
    _validate_ppc_execution,
    component_fingerprint,
    fingerprint_source_files,
)
from src.neural_analysis.lfp_summary_work_cache import (
    _ownership_record,
    _write_lock_exclusive,
    load_valid_ppc_checkpoint,
    load_valid_ppc_checkpoints,
    write_ppc_checkpoint,
)


_FLOAT_FIELDS = (
    "ppc", "resultant_length", "preferred_phase_rad", "p_value", "q_value",
    "null_mean", "null_std", "null_p025", "null_p50", "null_p975",
)
_INTEGER_FIELDS = (
    "spike_count", "eligible_trial_count", "null_exceedance_count",
    "permutation_count",
)
_BOOLEAN_FIELDS = ("computable", "reliable", "null_eligible", "significant")
_SUMMARY_FIELDS = _FLOAT_FIELDS + _INTEGER_FIELDS + _BOOLEAN_FIELDS
# S1/S2 retain nearest 8 and 40 Hz representative histograms independently of
# configurable phase-analysis bands used elsewhere in the pipeline.
_S1_S2_REPRESENTATIVE_BAND_COUNT = 2


@dataclass(frozen=True)
class PPCExecutionResult:
    """Work-only result for one selected condition/site/epoch PPC job.

    Attributes
    ----------
    run_fingerprint : str
        SHA-256 work identity binding the selected trials, source and phase
        representation, unit population, scientific settings, schedule, and
        execution representation. It has no physical units.
    run_directory : pathlib.Path
        Exact ``ppc/<run_fingerprint>`` work directory. It is absent when
        checkpoints are disabled.
    schedule : numpy.ndarray
        Owned int64 ``(shuffle, trial)`` source-to-target derangement matrix.
        It is empty with shape ``(0, trial)`` when fewer than two trials exist.
    summary_arrays : dict[str, numpy.ndarray]
        Owned arrays with axes ``(unit, frequency)``. ``ppc``, resultant
        length, p/q values, and null moments/percentiles are dimensionless;
        preferred phase is radians; count fields are int64 spikes, trials, or
        shuffles; flags are Boolean. Unavailable numerical entries are NaN and
        unavailable/ineligible flags are false. No raw phase or full-session
        shuffle-by-unit result is returned.
    completed_block_ids, resumed_block_ids : tuple[str, ...]
        Stable unit-block identities completed by this execution and the valid
        subset loaded from work checkpoints, respectively.
    """

    run_fingerprint: str
    run_directory: Path
    schedule: np.ndarray
    summary_arrays: dict[str, np.ndarray]
    completed_block_ids: tuple[str, ...]
    resumed_block_ids: tuple[str, ...]


_PPC_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class PPCJobPlan:
    """Immutable schedule and physical-edge description for one PPC result cell.

    Parameters
    ----------
    condition_index, site_index, epoch_index : int
        Nonnegative categorical positions in the component condition, site,
        and ``("whole", "before", "after")`` epoch axes.
    condition_name, site_id, epoch_name, segment_expression : str
        Stable categorical labels. ``segment_expression`` is one of
        ``"before"``, ``"after"``, or ``"before + after"``; it has no units.
    selected_trial_rows : numpy.ndarray
        Unique nonnegative int64 ``(selected_trial,)`` stable full trial-table
        identities. These are categorical row IDs, not local positions.
    schedule : numpy.ndarray
        Int64 ``(shuffle, selected_trial)`` local target-position derangement.
        It has shape ``(0, selected_trial)`` below two selected trials.
    stable_edge_source_trial_row, stable_edge_target_trial_row : numpy.ndarray
        Int64 arrays with the same ``(shuffle, selected_trial)`` axes as
        ``schedule``. They translate every local schedule edge to a stable
        full-table source/target identity.
    edge_union_position : numpy.ndarray
        Nonnegative int64 ``(shuffle, selected_trial)`` positions into the
        owning component's site-qualified physical-edge union.
    base_ppc_seed, schedule_seed, condition_derivation_identity,
    site_derivation_identity, epoch_derivation_identity : int
        Deterministic categorical seed provenance with no physical units.
    schedule_shape : tuple[int, int]
        Exact ``schedule.shape`` stored for checkpoint identity. Both elements
        must be non-Boolean integers and are canonicalized to Python ``int``.
    schedule_fingerprint : str
        Exact SHA-256 fingerprint of the int64 schedule bytes.

    Raises
    ------
    ValueError
        If core categorical identities, array dtypes/shapes, stable-edge
        translation, schedule fingerprint, or local target positions disagree.

    Notes
    -----
    Public construction always copies and disables writes on every ndarray
    field.  A private trusted factory is used internally only after planner
    arrays have already been allocated as owned read-only buffers.
    """

    condition_index: int
    condition_name: str
    site_index: int
    site_id: str
    epoch_index: int
    epoch_name: str
    selected_trial_rows: np.ndarray
    schedule: np.ndarray
    stable_edge_source_trial_row: np.ndarray
    stable_edge_target_trial_row: np.ndarray
    edge_union_position: np.ndarray
    segment_expression: str
    base_ppc_seed: int
    schedule_seed: int
    condition_derivation_identity: int
    site_derivation_identity: int
    epoch_derivation_identity: int
    schedule_shape: tuple[int, int]
    schedule_fingerprint: str

    def __post_init__(self) -> None:
        """Validate public inputs and retain owned read-only planning arrays."""
        provenance = _validate_job_plan_fields(
            condition_index=self.condition_index,
            site_index=self.site_index,
            epoch_index=self.epoch_index,
            condition_name=self.condition_name,
            site_id=self.site_id,
            epoch_name=self.epoch_name,
            segment_expression=self.segment_expression,
            selected_trial_rows=self.selected_trial_rows,
            schedule=self.schedule,
            stable_source=self.stable_edge_source_trial_row,
            stable_target=self.stable_edge_target_trial_row,
            edge_union_position=self.edge_union_position,
            base_ppc_seed=self.base_ppc_seed,
            schedule_seed=self.schedule_seed,
            condition_derivation_identity=self.condition_derivation_identity,
            site_derivation_identity=self.site_derivation_identity,
            epoch_derivation_identity=self.epoch_derivation_identity,
            schedule_shape=self.schedule_shape,
            schedule_fingerprint=self.schedule_fingerprint,
        )
        for field_name, value in zip(
            (
                "base_ppc_seed",
                "schedule_seed",
                "condition_derivation_identity",
                "site_derivation_identity",
                "epoch_derivation_identity",
            ),
            provenance[:5],
            strict=True,
        ):
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "schedule_shape", provenance[5])
        for field_name in (
            "selected_trial_rows",
            "schedule",
            "stable_edge_source_trial_row",
            "stable_edge_target_trial_row",
            "edge_union_position",
        ):
            object.__setattr__(
                self, field_name, _freeze_planned_int64(getattr(self, field_name))
            )


@dataclass(frozen=True)
class PPCAllocationEstimate:
    """Checked byte accounting for one grouped-PPC planning candidate.

    Parameters
    ----------
    job_accumulator_bytes : int
        Bytes for 40-byte null cells on ``(active_job, shuffle, unit, frequency)``.
    observed_trial_statistics_bytes, observed_gather_temporary_bytes : int
        Bytes retained while reducing observed same-trial statistics and their
        source gathers. These arrays are released before null accumulation.
    kernel_working_bytes, geometry_bytes : int
        Bytes for null kernel edge/gather temporaries and source geometry,
        respectively. Geometry is present in both computational stages.
    condition_membership_bytes, source_trial_spike_count_bytes : int
        Executor-created Boolean ``(trial, condition)`` membership and int64
        ``(trial, unit, before_after=2)`` count tables retained only during
        planning.
    planning_working_bytes : int
        Conservatively bounded schedule/map construction scratch derived by the
        planner. It excludes the two named planning tables and is not a caller
        supplied public planner value.
    planner_array_bytes, planned_planning_private_bytes,
    summary_assembly_bytes : int
        Parent-retained plan arrays, the complete planning stage peak, and
        full-component publication arrays.
    worker_plan_bytes, worker_summary_bytes : int
        Worker-owned full-site plan and current site/unit-block result arrays.
    checkpoint_block_bytes : int
        One parent-owned serialized site/unit-block summary plus compact int64
        ``site_index`` and half-open ``unit_bounds`` identity arrays.
    planned_computation_private_bytes : int
        Maximum of observed and null computational stage bytes.
    planned_parent_private_bytes, planned_worker_private_bytes : int
        Lifetime-based private peaks for the parent and one active worker.
    shared_phase_mmap_bytes, planned_aggregate_array_bytes : int
        Shared mmap bytes counted once and total parent-plus-active-worker bytes.
    active_worker_count : int
        ``min(requested_workers, pending_unit_blocks)`` for parallel work, or
        zero in serial execution. All values are bytes except this count.

    Raises
    ------
    ValueError
        If a value is Boolean, nonintegral, negative, or outside int64 range.
    """

    job_accumulator_bytes: int
    observed_trial_statistics_bytes: int
    observed_gather_temporary_bytes: int
    kernel_working_bytes: int
    geometry_bytes: int
    condition_membership_bytes: int
    source_trial_spike_count_bytes: int
    planning_working_bytes: int
    planner_array_bytes: int
    planned_planning_private_bytes: int
    summary_assembly_bytes: int
    worker_plan_bytes: int
    worker_summary_bytes: int
    checkpoint_block_bytes: int
    planned_computation_private_bytes: int
    planned_parent_private_bytes: int
    planned_worker_private_bytes: int
    shared_phase_mmap_bytes: int
    planned_aggregate_array_bytes: int
    active_worker_count: int

    def __post_init__(self) -> None:
        """Reject non-integral or overflowing byte-count records."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                or int(value) > _PPC_INT64_MAX
            ):
                raise ValueError(f"{field_name} must be a nonnegative int64")
            object.__setattr__(self, field_name, int(value))


@dataclass(frozen=True)
class PPCComponentPlan:
    """Pure immutable grouped PPC component plan.

    Parameters
    ----------
    job_plans : tuple[PPCJobPlan, ...]
        Site-major then condition then epoch immutable result-cell plans.
    edge_site_index, stable_edge_source_trial_row,
    stable_edge_target_trial_row : numpy.ndarray
        Matching owned int64 ``(union_edge,)`` arrays for the site-qualified
        physical union. Trial rows and site positions are categorical IDs with
        no physical units.
    condition_batches : tuple[tuple[tuple[int, ...], ...], ...]
        Ordered ``(site, batch, condition)`` condition positions. Batching only
        bounds null accumulation; it does not alter job or output order.
    scheduled_edge_count, independent_edge_count, union_edge_count : int
        Nonnegative edge counts. ``independent`` sums per-job physical demand;
        ``union`` counts site-qualified reusable work items.
    edge_union_saturation, edge_reuse_ratio : float
        Dimensionless finite metrics. Empty unions use exactly ``0.0``.
    allocation_estimate : PPCAllocationEstimate
        Pure lifetime-based byte estimate for the selected worst bounded task.

    Raises
    ------
    ValueError
        If union arrays, job union positions, counts, batching axes, or finite
        metrics are incoherent. Public construction defensively owns/freezes
        all ndarray fields.
    """

    job_plans: tuple[PPCJobPlan, ...]
    edge_site_index: np.ndarray
    stable_edge_source_trial_row: np.ndarray
    stable_edge_target_trial_row: np.ndarray
    condition_batches: tuple[tuple[tuple[int, ...], ...], ...]
    scheduled_edge_count: int
    independent_edge_count: int
    union_edge_count: int
    edge_union_saturation: float
    edge_reuse_ratio: float
    allocation_estimate: PPCAllocationEstimate

    def __post_init__(self) -> None:
        """Validate direct public construction and own component union arrays."""
        _validate_component_plan_fields(
            job_plans=self.job_plans,
            edge_site_index=self.edge_site_index,
            stable_source=self.stable_edge_source_trial_row,
            stable_target=self.stable_edge_target_trial_row,
            condition_batches=self.condition_batches,
            scheduled_edge_count=self.scheduled_edge_count,
            independent_edge_count=self.independent_edge_count,
            union_edge_count=self.union_edge_count,
            edge_union_saturation=self.edge_union_saturation,
            edge_reuse_ratio=self.edge_reuse_ratio,
            allocation_estimate=self.allocation_estimate,
        )
        for field_name in (
            "edge_site_index",
            "stable_edge_source_trial_row",
            "stable_edge_target_trial_row",
        ):
            object.__setattr__(
                self, field_name, _freeze_planned_int64(getattr(self, field_name))
            )


def _checked_ppc_int(value: object, name: str, *, minimum: int = 0) -> int:
    """Validate and canonicalize one bounded integer.

    Parameters
    ----------
    value : int or numpy.integer
        Candidate scalar without physical units. Boolean and floating values
        are not valid integers.
    name : str
        Input label used in a raised error.
    minimum : int, default=0
        Inclusive lower bound.

    Returns
    -------
    int
        Python ``int`` in ``[minimum, int64_max]``.

    Raises
    ------
    ValueError
        If ``value`` is Boolean, nonintegral, below ``minimum``, or outside
        signed int64 range.
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < minimum
        or int(value) > _PPC_INT64_MAX
    ):
        raise ValueError(f"{name} must be an int64 greater than or equal to {minimum}")
    return int(value)


def _checked_ppc_add(*values: int) -> int:
    """Add nonnegative int64-sized scalar counts without wraparound.

    Parameters
    ----------
    *values : int
        Already validated nonnegative byte counts, categorical counts, or
        seeds. They are scalar and have no shared array axes.

    Returns
    -------
    int
        Exact nonnegative Python sum, limited to signed int64 range.

    Raises
    ------
    ValueError
        If a value is negative or the sum exceeds signed int64 range.
    """
    total = 0
    for value in values:
        if value < 0 or value > _PPC_INT64_MAX - total:
            raise ValueError("PPC allocation arithmetic overflow")
        total += value
    return total


def _checked_ppc_multiply(*values: int) -> int:
    """Multiply nonnegative int64-sized scalar counts without wraparound.

    Parameters
    ----------
    *values : int
        Already validated nonnegative dimension sizes, byte widths, or seed
        factors. Scalars have no physical array axes.

    Returns
    -------
    int
        Exact nonnegative Python product, limited to signed int64 range.

    Raises
    ------
    ValueError
        If a factor is negative or multiplication exceeds signed int64 range.
    """
    product = 1
    for value in values:
        if value < 0 or (value and product > _PPC_INT64_MAX // value):
            raise ValueError("PPC allocation arithmetic overflow")
        product *= value
    return product


def _freeze_planned_int64(value: np.ndarray) -> np.ndarray:
    """Copy one planner array into owned read-only int64 storage.

    Parameters
    ----------
    value : numpy.ndarray
        Integer-like planning array of arbitrary documented caller axes. Values
        are categorical positions or counts and have no physical units.

    Returns
    -------
    numpy.ndarray
        Independent read-only int64 array with the same shape and values.
    """
    array = np.array(value, dtype=np.int64, copy=True)
    array.setflags(write=False)
    return array


def _seal_owned_planned_int64(value: np.ndarray) -> np.ndarray:
    """Mark an already-owned int64 planner buffer read-only without copying.

    Parameters
    ----------
    value : numpy.ndarray
        Planner-owned int64 array with arbitrary documented axes. It must not
        alias caller-owned mutable input; values are categorical IDs/counts.

    Returns
    -------
    numpy.ndarray
        The same array object with writes disabled.

    Raises
    ------
    ValueError
        If ``value`` is not an int64 ndarray.
    """
    array = np.asarray(value)
    if array.dtype != np.dtype(np.int64):
        raise ValueError("trusted planning arrays must be int64")
    array.setflags(write=False)
    return array


def _canonical_schedule_shape(
    value: object,
    expected_shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate and canonicalize a checkpoint schedule shape.

    Parameters
    ----------
    value : tuple[int, int]
        Candidate two-element shape. Each element is a categorical axis length
        and must be a non-Boolean Python or NumPy integer.
    expected_shape : tuple[int, int]
        Actual int64 schedule ``(shuffle, selected_trial)`` shape.

    Returns
    -------
    tuple[int, int]
        Exact ``expected_shape`` represented by Python integers.

    Raises
    ------
    ValueError
        If the candidate is not a two-element tuple of non-Boolean integers or
        does not exactly match the schedule axes.
    """
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("schedule_shape must be a two-element integer tuple")
    shape = (
        _checked_ppc_int(value[0], "schedule_shape[0]"),
        _checked_ppc_int(value[1], "schedule_shape[1]"),
    )
    if shape != expected_shape:
        raise ValueError("PPC schedule shape does not match its array")
    return shape


def _validate_generated_derangement_schedule(
    value: object,
    *,
    trial_count: int,
    shuffle_count: int,
) -> np.ndarray:
    """Validate a generator-owned schedule in place without copying it.

    Parameters
    ----------
    value : numpy.ndarray
        Int64 ``(shuffle, selected_trial)`` candidate supplied by the internal
        generator seam. Its buffer is retained directly by the planner; trial
        positions are local categorical indexes, not stable IDs or times.
    trial_count : int
        Positive selected-trial axis size, at least two.
    shuffle_count : int
        Positive configured full schedule-row count.

    Returns
    -------
    numpy.ndarray
        The same owned int64 buffer after every row is verified as a complete
        no-fixed-point permutation. No schedule-sized copy is allocated.

    Raises
    ------
    ValueError
        If dtype, axes, or any local derangement row is invalid.
    """
    if not isinstance(value, np.ndarray):
        raise ValueError("generated PPC schedule must be an owned int64 ndarray")
    schedule = value
    if (
        schedule.dtype != np.dtype(np.int64)
        or schedule.shape != (shuffle_count, trial_count)
    ):
        raise ValueError("generated PPC schedule has invalid dtype or axes")
    identity = np.arange(trial_count, dtype=np.int64)
    for row in schedule:
        if (
            np.any(row < 0)
            or np.any(row >= trial_count)
            or np.any(row == identity)
            or not np.array_equal(np.sort(row), identity)
        ):
            raise ValueError("generated PPC schedule rows must be derangement permutations")
    return schedule


def _validate_job_plan_fields(
    *,
    condition_index: object,
    site_index: object,
    epoch_index: object,
    condition_name: object,
    site_id: object,
    epoch_name: object,
    segment_expression: object,
    selected_trial_rows: object,
    schedule: object,
    stable_source: object,
    stable_target: object,
    edge_union_position: object,
    base_ppc_seed: object,
    schedule_seed: object,
    condition_derivation_identity: object,
    site_derivation_identity: object,
    epoch_derivation_identity: object,
    schedule_shape: object,
    schedule_fingerprint: object,
    check_array_content: bool = True,
    check_schedule_fingerprint: bool = True,
) -> tuple[int, int, int, int, int, tuple[int, int]]:
    """Validate one job record's categorical, schedule, and stable-edge contract.

    Parameters
    ----------
    condition_index, site_index, epoch_index : int or numpy.integer
        Nonnegative categorical positions. Epoch position is exactly 0, 1, or
        2 for whole, before, or after, respectively.
    condition_name, site_id, epoch_name, segment_expression : str
        Stable labels; the epoch label/expression pair follows the fixed epoch
        index mapping and has no physical unit.
    selected_trial_rows : numpy.ndarray
        Unique nonnegative int64 ``(selected_trial,)`` stable trial-table IDs.
    schedule : numpy.ndarray
        Int64 ``(shuffle, selected_trial)`` local target positions. For two or
        more selected trials it has at least one row and every row is a full
        no-fixed-point permutation. Empty/singleton jobs have zero rows.
    stable_source, stable_target, edge_union_position : numpy.ndarray
        Int64 arrays on the same ``(shuffle, selected_trial)`` axes. Stable
        source/target arrays translate the local schedule; positions are
        nonnegative indexes into a component union.
    base_ppc_seed, schedule_seed, condition_derivation_identity,
    site_derivation_identity, epoch_derivation_identity : int or numpy.integer
        Nonnegative deterministic provenance. Identities equal their matching
        categorical index and schedule seed equals ``base + 1e6*condition +
        1e4*site + 100*epoch``.
    schedule_shape : tuple[int, int]
        Exact local schedule shape.
    schedule_fingerprint : str
        SHA-256 fingerprint of the int64 schedule bytes.
    check_array_content : bool, default=True
        Public construction checks stable-source/target content and each row
        permutation. The internal trusted factory passes ``False`` after it
        has constructed owned buffers, avoiding full-size validation arrays.
    check_schedule_fingerprint : bool, default=True
        Whether to recompute the schedule hash. The trusted factory receives a
        hash that the private planner just computed and passes ``False`` to
        avoid a second schedule-sized byte serialization.

    Returns
    -------
    tuple
        Canonical Python ``(base_seed, schedule_seed, condition_identity,
        site_identity, epoch_identity, schedule_shape)`` values.

    Raises
    ------
    ValueError
        If scalar provenance, epoch mapping, dtypes, axes, stable identities,
        schedule permutation/fingerprint, or checked seed arithmetic is
        incoherent.
    """
    for value, name in (
        (condition_index, "condition_index"),
        (site_index, "site_index"),
        (epoch_index, "epoch_index"),
    ):
        _checked_ppc_int(value, name)
    expected_epochs = {
        0: ("whole", "before + after"),
        1: ("before", "before"),
        2: ("after", "after"),
    }
    if (
        not isinstance(condition_name, str)
        or not condition_name
        or not isinstance(site_id, str)
        or not site_id
        or epoch_index not in expected_epochs
        or not isinstance(epoch_name, str)
        or not isinstance(segment_expression, str)
        or (epoch_name, segment_expression) != expected_epochs[epoch_index]
    ):
        raise ValueError("invalid PPC job categorical identity")
    base_seed = _checked_ppc_int(base_ppc_seed, "base_ppc_seed")
    derived_seed = _checked_ppc_int(schedule_seed, "schedule_seed")
    condition_identity = _checked_ppc_int(
        condition_derivation_identity, "condition_derivation_identity"
    )
    site_identity = _checked_ppc_int(site_derivation_identity, "site_derivation_identity")
    epoch_identity = _checked_ppc_int(
        epoch_derivation_identity, "epoch_derivation_identity"
    )
    if (
        condition_identity != int(condition_index)
        or site_identity != int(site_index)
        or epoch_identity != int(epoch_index)
    ):
        raise ValueError("PPC derivation identities must match categorical indices")
    expected_seed = _checked_ppc_add(
        base_seed,
        _checked_ppc_multiply(int(condition_index), 1_000_000),
        _checked_ppc_multiply(int(site_index), 10_000),
        _checked_ppc_multiply(int(epoch_index), 100),
    )
    if derived_seed != expected_seed:
        raise ValueError("PPC schedule_seed must match its categorical derivation")
    selected = np.asarray(selected_trial_rows)
    local_schedule = np.asarray(schedule)
    source = np.asarray(stable_source)
    target = np.asarray(stable_target)
    union_position = np.asarray(edge_union_position)
    if (
        selected.dtype != np.dtype(np.int64)
        or selected.ndim != 1
        or np.any(selected < 0)
        or np.unique(selected).size != selected.size
        or local_schedule.dtype != np.dtype(np.int64)
        or local_schedule.ndim != 2
        or local_schedule.shape[1] != selected.size
        or source.dtype != np.dtype(np.int64)
        or target.dtype != np.dtype(np.int64)
        or union_position.dtype != np.dtype(np.int64)
        or source.shape != local_schedule.shape
        or target.shape != local_schedule.shape
        or union_position.shape != local_schedule.shape
    ):
        raise ValueError("invalid PPC job planning array axes")
    if selected.size < 2:
        if local_schedule.shape[0] != 0:
            raise ValueError("empty/singleton PPC jobs cannot have a schedule")
    elif local_schedule.shape[0] == 0:
        raise ValueError("PPC jobs with two or more trials require a schedule")
    elif check_array_content:
        expected_positions = np.arange(selected.size, dtype=np.int64)
        for row in local_schedule:
            if (
                np.any(row < 0)
                or np.any(row >= selected.size)
                or np.any(row == expected_positions)
                or not np.array_equal(np.sort(row), expected_positions)
            ):
                raise ValueError("PPC schedule rows must be derangement permutations")
    if check_array_content:
        if np.any(union_position < 0):
            raise ValueError("PPC edge-union positions must be nonnegative")
        expected_source = np.broadcast_to(selected[np.newaxis, :], local_schedule.shape)
        expected_target = (
            selected[local_schedule]
            if local_schedule.size
            else np.empty(local_schedule.shape, dtype=np.int64)
        )
        if not np.array_equal(source, expected_source) or not np.array_equal(target, expected_target):
            raise ValueError("PPC stable edge arrays must match selected rows and schedule")
    canonical_shape = _canonical_schedule_shape(schedule_shape, local_schedule.shape)
    if not isinstance(schedule_fingerprint, str):
        raise ValueError("PPC schedule fingerprint must be a string")
    if (
        check_schedule_fingerprint
        and schedule_fingerprint != _array_fingerprint(local_schedule)
    ):
        raise ValueError("PPC schedule fingerprint does not match its array")
    return (
        base_seed,
        derived_seed,
        condition_identity,
        site_identity,
        epoch_identity,
        canonical_shape,
    )


def _trusted_ppc_job_plan(**values: object) -> PPCJobPlan:
    """Create one owned internal job without public full-array revalidation.

    Parameters
    ----------
    **values : object
        Complete :class:`PPCJobPlan` field mapping. Array fields are already
        planner-owned int64 buffers with documented job axes; scalar fields
        use the public record's categorical/seed units.

    Returns
    -------
    PPCJobPlan
        Frozen record that reuses and seals the owned buffers.

    Raises
    ------
    ValueError
        If scalar provenance, array axes/dtypes, or schedule fingerprint is
        invalid. It deliberately skips full stable-edge/permutation temporary
        construction because the private planner generated those arrays.
    """
    provenance = _validate_job_plan_fields(
        condition_index=values["condition_index"],
        site_index=values["site_index"],
        epoch_index=values["epoch_index"],
        condition_name=values["condition_name"],
        site_id=values["site_id"],
        epoch_name=values["epoch_name"],
        segment_expression=values["segment_expression"],
        selected_trial_rows=values["selected_trial_rows"],
        schedule=values["schedule"],
        stable_source=values["stable_edge_source_trial_row"],
        stable_target=values["stable_edge_target_trial_row"],
        edge_union_position=values["edge_union_position"],
        base_ppc_seed=values["base_ppc_seed"],
        schedule_seed=values["schedule_seed"],
        condition_derivation_identity=values["condition_derivation_identity"],
        site_derivation_identity=values["site_derivation_identity"],
        epoch_derivation_identity=values["epoch_derivation_identity"],
        schedule_shape=values["schedule_shape"],
        schedule_fingerprint=values["schedule_fingerprint"],
        check_array_content=False,
        check_schedule_fingerprint=False,
    )
    job = object.__new__(PPCJobPlan)
    for field_name in PPCJobPlan.__dataclass_fields__:
        value = values[field_name]
        if field_name in {
            "selected_trial_rows",
            "schedule",
            "stable_edge_source_trial_row",
            "stable_edge_target_trial_row",
            "edge_union_position",
        }:
            value = _seal_owned_planned_int64(np.asarray(value))
        elif field_name in {
            "base_ppc_seed",
            "schedule_seed",
            "condition_derivation_identity",
            "site_derivation_identity",
            "epoch_derivation_identity",
        }:
            value = provenance[
                (
                    "base_ppc_seed",
                    "schedule_seed",
                    "condition_derivation_identity",
                    "site_derivation_identity",
                    "epoch_derivation_identity",
                ).index(field_name)
            ]
        elif field_name == "schedule_shape":
            value = provenance[5]
        object.__setattr__(job, field_name, value)
    return job


def _validate_component_plan_fields(
    *,
    job_plans: object,
    edge_site_index: object,
    stable_source: object,
    stable_target: object,
    condition_batches: object,
    scheduled_edge_count: object,
    independent_edge_count: object,
    union_edge_count: object,
    edge_union_saturation: object,
    edge_reuse_ratio: object,
    allocation_estimate: object,
    check_derived: bool = True,
) -> None:
    """Validate a component record's union, metrics, and site batching contract.

    Parameters
    ----------
    job_plans : tuple[PPCJobPlan, ...]
        Immutable jobs in site-major, condition-major, epoch-major order. Job
        arrays use ``(shuffle, selected_trial)`` axes and stable row IDs.
    edge_site_index, stable_source, stable_target : numpy.ndarray
        Matching nonnegative int64 ``(union_edge,)`` site-qualified physical
        union vectors. Site/trial values are categorical identities.
    condition_batches : tuple[tuple[tuple[int, ...], ...], ...]
        ``(site, batch, condition)`` categorical positions. Flattened batches
        for each site exactly equal that site's job condition positions in
        ascending input order.
    scheduled_edge_count, independent_edge_count, union_edge_count : int
        Nonnegative scalar edge counts. Scheduled counts all schedule cells;
        independent counts unique physical edges within each job; union counts
        sorted unique site-qualified job demand.
    edge_union_saturation, edge_reuse_ratio : float
        Dimensionless non-Boolean finite metrics. Saturation is in ``[0, 1]``;
        reuse is nonnegative. Empty demand uses exactly ``0.0`` for both.
    allocation_estimate : PPCAllocationEstimate
        Immutable byte estimate with byte-valued allocation categories.
    check_derived : bool, default=True
        Public construction recomputes all union/count/metric/batch invariants.
        The trusted planner factory passes ``False`` after generating owned
        canonical values, avoiding full-size validation temporary structures.

    Raises
    ------
    ValueError
        If record types, vector axes, union mapping, scalar metrics, or any
        requested derived invariant is incoherent.
    """
    if (
        not isinstance(job_plans, tuple)
        or not all(isinstance(job, PPCJobPlan) for job in job_plans)
        or not isinstance(allocation_estimate, PPCAllocationEstimate)
    ):
        raise ValueError("component plan requires PPC job plans and an allocation estimate")
    site = np.asarray(edge_site_index)
    source = np.asarray(stable_source)
    target = np.asarray(stable_target)
    union_count = _checked_ppc_int(union_edge_count, "union_edge_count")
    if (
        site.dtype != np.dtype(np.int64)
        or source.dtype != np.dtype(np.int64)
        or target.dtype != np.dtype(np.int64)
        or site.ndim != 1
        or source.ndim != 1
        or target.ndim != 1
        or site.size != union_count
        or source.size != union_count
        or target.size != union_count
    ):
        raise ValueError("component PPC union arrays must be matching int64 vectors")
    scheduled_count = _checked_ppc_int(scheduled_edge_count, "scheduled_edge_count")
    independent_count = _checked_ppc_int(
        independent_edge_count, "independent_edge_count"
    )
    if (
        isinstance(edge_union_saturation, (bool, np.bool_))
        or not isinstance(edge_union_saturation, (float, int, np.floating, np.integer))
        or not np.isfinite(edge_union_saturation)
        or isinstance(edge_reuse_ratio, (bool, np.bool_))
        or not isinstance(edge_reuse_ratio, (float, int, np.floating, np.integer))
        or not np.isfinite(edge_reuse_ratio)
        or float(edge_union_saturation) < 0.0
        or float(edge_union_saturation) > 1.0
        or float(edge_reuse_ratio) < 0.0
    ):
        raise ValueError("component PPC edge metrics must be finite valid ratios")

    if check_derived:
        if np.any(site < 0) or np.any(source < 0) or np.any(target < 0):
            raise ValueError("component PPC union arrays must be nonnegative")
        actual_union = tuple(
            zip(site.tolist(), source.tolist(), target.tolist(), strict=True)
        )
        expected_union = tuple(
            sorted(
                {
                    (job.site_index, int(left), int(right))
                    for job in job_plans
                    for left, right in zip(
                        job.stable_edge_source_trial_row.ravel(),
                        job.stable_edge_target_trial_row.ravel(),
                        strict=True,
                    )
                }
            )
        )
        if actual_union != expected_union:
            raise ValueError("component union must be sorted unique site-qualified job demand")
        expected_scheduled_count = sum(job.schedule.size for job in job_plans)
        expected_independent_count = sum(
            len(
                set(
                    zip(
                        job.stable_edge_source_trial_row.ravel(),
                        job.stable_edge_target_trial_row.ravel(),
                        strict=True,
                    )
                )
            )
            for job in job_plans
        )
        if (
            scheduled_count != expected_scheduled_count
            or independent_count != expected_independent_count
            or union_count != len(expected_union)
        ):
            raise ValueError("component PPC edge counts must be derived from job demand")
        allowed_edges = {
            (job.site_index, int(left), int(right))
            for job in job_plans
            for left in job.selected_trial_rows
            for right in job.selected_trial_rows
            if left != right
        }
        expected_saturation = (
            float(union_count) / float(len(allowed_edges)) if allowed_edges else 0.0
        )
        expected_reuse = (
            float(independent_count) / float(union_count) if union_count else 0.0
        )
        if (
            float(edge_union_saturation) != expected_saturation
            or float(edge_reuse_ratio) != expected_reuse
        ):
            raise ValueError("component PPC edge metrics must be exact derived values")
        site_count = 1 + max((job.site_index for job in job_plans), default=-1)
        if not isinstance(condition_batches, tuple) or len(condition_batches) != site_count:
            raise ValueError("condition_batches must provide every component site")
        for site_index, site_batches in enumerate(condition_batches):
            if not isinstance(site_batches, tuple) or not site_batches:
                raise ValueError("condition_batches must contain ordered nonempty batches")
            flattened_conditions: list[int] = []
            for batch in site_batches:
                if (
                    not isinstance(batch, tuple)
                    or not batch
                    or any(
                        isinstance(condition, (bool, np.bool_))
                        or not isinstance(condition, (int, np.integer))
                        or int(condition) < 0
                        for condition in batch
                    )
                ):
                    raise ValueError("condition batches must contain nonnegative integer positions")
                flattened_conditions.extend(int(condition) for condition in batch)
            expected_conditions = tuple(
                sorted(
                    {
                        job.condition_index
                        for job in job_plans
                        if job.site_index == site_index
                    }
                )
            )
            if tuple(flattened_conditions) != expected_conditions:
                raise ValueError("condition batches must cover site conditions once in order")
        for job in job_plans:
            for union_position, left, right in zip(
                job.edge_union_position.ravel(),
                job.stable_edge_source_trial_row.ravel(),
                job.stable_edge_target_trial_row.ravel(),
                strict=True,
            ):
                position = int(union_position)
                if (
                    position >= union_count
                    or int(site[position]) != job.site_index
                    or int(source[position]) != int(left)
                    or int(target[position]) != int(right)
                ):
                    raise ValueError("job edge positions must resolve to matching union edges")
    elif not isinstance(condition_batches, tuple):
        raise ValueError("condition_batches must be a site/batch tuple")


def _trusted_ppc_component_plan(**values: object) -> PPCComponentPlan:
    """Create an owned internal component plan without public recomputation.

    Parameters
    ----------
    **values : object
        Complete :class:`PPCComponentPlan` field mapping. Union vectors are
        planner-owned int64 ``(union_edge,)`` buffers and job records are
        already canonical; all memory values are bytes.

    Returns
    -------
    PPCComponentPlan
        Frozen component record that reuses and seals existing union buffers.

    Raises
    ------
    ValueError
        If record/vector types or scalar metric bounds are invalid. Full union,
        metric, and batch recomputation is intentionally skipped because the
        private planner generated canonical values without caller aliases.
    """
    _validate_component_plan_fields(
        job_plans=values["job_plans"],
        edge_site_index=values["edge_site_index"],
        stable_source=values["stable_edge_source_trial_row"],
        stable_target=values["stable_edge_target_trial_row"],
        condition_batches=values["condition_batches"],
        scheduled_edge_count=values["scheduled_edge_count"],
        independent_edge_count=values["independent_edge_count"],
        union_edge_count=values["union_edge_count"],
        edge_union_saturation=values["edge_union_saturation"],
        edge_reuse_ratio=values["edge_reuse_ratio"],
        allocation_estimate=values["allocation_estimate"],
        check_derived=False,
    )
    component = object.__new__(PPCComponentPlan)
    for field_name in PPCComponentPlan.__dataclass_fields__:
        value = values[field_name]
        if field_name in {
            "edge_site_index",
            "stable_edge_source_trial_row",
            "stable_edge_target_trial_row",
        }:
            value = _seal_owned_planned_int64(np.asarray(value))
        object.__setattr__(component, field_name, value)
    return component


def _validate_planning_positions(
    value: object,
    *,
    name: str,
    source_count: int,
    unique: bool,
) -> np.ndarray:
    """Validate source-axis positions used only by allocation planning.

    Parameters
    ----------
    value : numpy.ndarray
        Candidate int64 ``(position,)`` source-trial indices. Positions are
        categorical, not physical time or frequency values.
    name : str
        Label included in a validation failure.
    source_count : int
        Positive length of the source-trial axis that bounds each position.
    unique : bool
        Whether the positions must occur at most once, as required by observed
        same-trial gathers. Null edge positions may repeat.

    Returns
    -------
    numpy.ndarray
        The original int64 one-dimensional position array without copying.

    Raises
    ------
    ValueError
        If dtype/axis/range is invalid or a unique position is repeated.
    """
    positions = np.asarray(value)
    if positions.dtype != np.dtype(np.int64) or positions.ndim != 1:
        raise ValueError(f"{name} must be valid int64 source positions")
    seen_positions: set[int] = set()
    for raw_position in positions:
        position = int(raw_position)
        if position < 0 or position >= source_count:
            raise ValueError(f"{name} must be valid int64 source positions")
        if unique:
            if position in seen_positions:
                raise ValueError(f"{name} must be valid int64 source positions")
            seen_positions.add(position)
    return positions


def _geometry_allocation_bytes(source_trial_spike_count: np.ndarray) -> int:
    """Count retained segmented-kernel source geometry bytes.

    Parameters
    ----------
    source_trial_spike_count : numpy.ndarray
        Nonnegative int64 ``(source_trial, unit_block, segment=2)`` spike
        counts. Segment 0/1 are before/after event; counts have no physical
        unit and are used to size geometry records.

    Returns
    -------
    int
        Exact nonnegative byte count for retained spike records, offsets, and
        stable source identities.

    Raises
    ------
    ValueError
        If checked byte arithmetic exceeds signed int64 range.
    """
    total_spikes = 0
    for count in source_trial_spike_count.flat:
        total_spikes = _checked_ppc_add(total_spikes, int(count))
    source_count, unit_count, _ = source_trial_spike_count.shape
    geometry_spikes = _checked_ppc_multiply(26, total_spikes)
    offsets = _checked_ppc_multiply(source_count, unit_count * 2 + 1, 8)
    identities = _checked_ppc_multiply(source_count, 8)
    return _checked_ppc_add(geometry_spikes, offsets, identities)


def estimate_grouped_ppc_allocation(
    *,
    active_job_count: int,
    worker_result_job_count: int,
    component_job_count: int,
    shuffle_count: int,
    total_unit_count: int,
    unit_block_size: int,
    source_trial_spike_count: np.ndarray,
    edge_source_trial_position: np.ndarray,
    observed_source_trial_position: np.ndarray,
    frequency_count: int,
    representative_band_count: int,
    phase_bin_count: int,
    planner_array_bytes: int,
    worker_plan_bytes: int,
    worker_count: int,
    pending_unit_block_count: int,
    shared_phase_mmap_bytes: int,
    condition_membership_bytes: int = 0,
    source_trial_spike_count_bytes: int = 0,
    planning_working_bytes: int = 0,
) -> PPCAllocationEstimate:
    """Estimate the bounded-memory cost of one grouped PPC work candidate.

    Parameters
    ----------
    active_job_count : int
        Nonnegative number of scheduled jobs in the current site/condition
        batch.  It sizes only the current null-shuffle accumulators.
    worker_result_job_count : int
        Positive number of all jobs retained by the current site/unit-block
        worker, including observed-only and empty jobs.  It sizes the worker
        checkpoint/result arrays and must be at least ``active_job_count``.
    component_job_count : int
        Positive number of result jobs across all sites, conditions, and
        epochs in the component.  It sizes parent summary assembly and must
        be at least ``worker_result_job_count``.
    shuffle_count : int
        Positive full configured number of null schedules.  This is not a
        chunk size and sizes retained null accumulator cells.
    total_unit_count : int
        Positive full component ``unit``-axis length.  Units are sorted source
        units and have no physical dimension in this allocator.
    unit_block_size : int
        Positive current ``unit_block`` length, at most ``total_unit_count``.
    source_trial_spike_count : numpy.ndarray
        Nonnegative int64 spike counts with shape ``(source_trial, unit_block,
        segment=2)``.  Segment position 0 is before-event and 1 is
        after-event; values are spike counts, not times or rates.
    edge_source_trial_position : numpy.ndarray
        Int64 ``(edge,)`` positions on the first axis of
        ``source_trial_spike_count`` for null edges. Repeated positions are
        valid and preserve source multiplicity in the null gather estimate.
    observed_source_trial_position : numpy.ndarray
        Unique int64 ``(observed_trial,)`` positions on the same source axis.
        Each position is gathered once for observed same-trial statistics.
    frequency_count : int
        Positive number of frequency bins; bins have units of Hz but this
        integer is a dimension count.
    representative_band_count : int
        Positive number of representative frequency bands.
    phase_bin_count : int
        Positive number of phase-histogram bins; the underlying phase unit is
        radians but this integer is a dimension count.
    planner_array_bytes : int
        Nonnegative parent-retained component-plan byte count.
    worker_plan_bytes : int
        Nonnegative worker-retained current-site plan byte count.
    worker_count : int
        Positive requested process count. ``1`` denotes serial execution.
    pending_unit_block_count : int
        Nonnegative number of pending unit blocks. Parallel active workers are
        ``min(worker_count, pending_unit_block_count)``; serial has zero.
    shared_phase_mmap_bytes : int
        Nonnegative shared read-only phase mmap bytes, counted exactly once.
    condition_membership_bytes : int
        Nonnegative bytes for the executor-created Boolean ``(trial,
        condition)`` membership table. It is retained only for planning.
    source_trial_spike_count_bytes : int
        Nonnegative bytes for the executor-created int64 ``(trial, unit,
        before_after=2)`` full spike-count table. It is retained only for
        planning.
    planning_working_bytes : int
        Nonnegative conservative planner construction scratch bytes after the
        two named planning tables. The pure planner derives this value from
        scalar input dimensions, schedules, and worst-case edge geometry.

    Returns
    -------
    PPCAllocationEstimate
        Immutable byte counts for observed/null computation stages, retained
        parent/worker lifetimes, and the aggregate process peak. Every memory
        field is bytes; ``active_worker_count`` is a process count.

    Raises
    ------
    ValueError
        If counts are Boolean, nonintegral, negative, or exceed int64; if an
        array dtype/shape/position is invalid; if job-count coherence or unit
        axes disagree; or if checked int64 allocation arithmetic overflows.

    Notes
    -----
    This function is pure: it does not sample phase, access files, start
    workers, or allocate numerical phase/spike buffers.
    """
    active_jobs = _checked_ppc_int(active_job_count, "active_job_count")
    worker_jobs = _checked_ppc_int(
        worker_result_job_count, "worker_result_job_count", minimum=1
    )
    component_jobs = _checked_ppc_int(
        component_job_count, "component_job_count", minimum=1
    )
    shuffles = _checked_ppc_int(shuffle_count, "shuffle_count", minimum=1)
    total_units = _checked_ppc_int(total_unit_count, "total_unit_count", minimum=1)
    block_units = _checked_ppc_int(unit_block_size, "unit_block_size", minimum=1)
    frequencies = _checked_ppc_int(frequency_count, "frequency_count", minimum=1)
    representative_bands = _checked_ppc_int(
        representative_band_count, "representative_band_count", minimum=1
    )
    phase_bins = _checked_ppc_int(phase_bin_count, "phase_bin_count", minimum=1)
    planned_arrays = _checked_ppc_int(planner_array_bytes, "planner_array_bytes")
    worker_plan = _checked_ppc_int(worker_plan_bytes, "worker_plan_bytes")
    requested_workers = _checked_ppc_int(worker_count, "worker_count", minimum=1)
    pending_blocks = _checked_ppc_int(
        pending_unit_block_count, "pending_unit_block_count"
    )
    shared_mmap = _checked_ppc_int(
        shared_phase_mmap_bytes, "shared_phase_mmap_bytes"
    )
    membership_bytes = _checked_ppc_int(
        condition_membership_bytes, "condition_membership_bytes"
    )
    full_count_bytes = _checked_ppc_int(
        source_trial_spike_count_bytes, "source_trial_spike_count_bytes"
    )
    planning_working = _checked_ppc_int(
        planning_working_bytes, "planning_working_bytes"
    )
    if active_jobs > worker_jobs or worker_jobs > component_jobs:
        raise ValueError("active_job_count <= worker_result_job_count <= component_job_count is required")
    if block_units > total_units:
        raise ValueError("unit_block_size cannot exceed total_unit_count")

    counts = np.asarray(source_trial_spike_count)
    if (
        counts.dtype != np.dtype(np.int64)
        or counts.ndim != 3
        or counts.shape[0] < 1
        or counts.shape[1] != block_units
        or counts.shape[2] != 2
    ):
        raise ValueError("source_trial_spike_count must be nonnegative int64 (source, unit_block, 2)")
    for raw_count in counts.flat:
        if int(raw_count) < 0:
            raise ValueError(
                "source_trial_spike_count must be nonnegative int64 (source, unit_block, 2)"
            )
    edge_positions = _validate_planning_positions(
        edge_source_trial_position,
        name="edge_source_trial_position",
        source_count=counts.shape[0],
        unique=False,
    )
    observed_positions = _validate_planning_positions(
        observed_source_trial_position,
        name="observed_source_trial_position",
        source_count=counts.shape[0],
        unique=True,
    )

    geometry_bytes = _geometry_allocation_bytes(counts)
    if edge_positions.size:
        kernel = estimate_segmented_kernel_allocation(
            source_trial_spike_count=counts,
            edge_source_trial_position=edge_positions,
            frequency_count=frequencies,
        )
        geometry_bytes = kernel.geometry_bytes
        kernel_working_bytes = _checked_ppc_add(
            kernel.segmented_edge_statistics_bytes,
            kernel.gather_temporary_bytes,
        )
    else:
        kernel_working_bytes = 0

    job_accumulator_bytes = _checked_ppc_multiply(
        40, active_jobs, shuffles, block_units, frequencies
    )
    if observed_positions.size:
        observed_trial_count = int(observed_positions.size)
        observed_trial_statistics_bytes = _checked_ppc_add(
            _checked_ppc_multiply(8, observed_trial_count),
            _checked_ppc_multiply(
                24, observed_trial_count, block_units, 2, frequencies
            ),
            _checked_ppc_multiply(8, representative_bands),
            _checked_ppc_multiply(8, representative_bands),
            _checked_ppc_multiply(8, phase_bins + 1),
            _checked_ppc_multiply(
                8,
                observed_trial_count,
                block_units,
                2,
                representative_bands,
                phase_bins,
            ),
        )
        observed_spikes = 0
        for position in observed_positions:
            for count in counts[int(position)].flat:
                observed_spikes = _checked_ppc_add(observed_spikes, int(count))
        observed_gather_temporary_bytes = _checked_ppc_multiply(
            51, frequencies, observed_spikes
        )
    else:
        observed_trial_statistics_bytes = 0
        observed_gather_temporary_bytes = 0

    observed_stage = _checked_ppc_add(
        geometry_bytes,
        observed_trial_statistics_bytes,
        observed_gather_temporary_bytes,
    )
    null_stage = _checked_ppc_add(
        geometry_bytes, job_accumulator_bytes, kernel_working_bytes
    )
    computation_private = max(observed_stage, null_stage)
    summary_cell_bytes = _checked_ppc_add(
        _checked_ppc_multiply(116, frequencies),
        _checked_ppc_multiply(8, representative_bands, phase_bins),
    )
    summary_assembly_bytes = _checked_ppc_multiply(
        total_units, component_jobs, summary_cell_bytes
    )
    worker_summary_bytes = _checked_ppc_multiply(
        block_units, worker_jobs, summary_cell_bytes
    )
    checkpoint_block_bytes = _checked_ppc_add(worker_summary_bytes, 3 * 8)
    planning_private = _checked_ppc_add(
        membership_bytes,
        full_count_bytes,
        max(planned_arrays, planning_working),
    )
    if requested_workers == 1:
        active_workers = 0
        steady_parent_private = _checked_ppc_add(
            planned_arrays,
            summary_assembly_bytes,
            max(computation_private, checkpoint_block_bytes),
        )
    else:
        active_workers = min(requested_workers, pending_blocks)
        steady_parent_private = _checked_ppc_add(
            planned_arrays, summary_assembly_bytes, checkpoint_block_bytes
        )
    # The local summary remains live while compute scratch is used, then its
    # bounded staged checkpoint replaces scratch before the scalar handoff.
    worker_private = _checked_ppc_add(
        worker_plan,
        max(
            _checked_ppc_add(worker_summary_bytes, computation_private),
            checkpoint_block_bytes,
        ),
    )
    parent_private = max(planning_private, steady_parent_private)
    steady_aggregate_private = _checked_ppc_add(
        steady_parent_private,
        _checked_ppc_multiply(active_workers, worker_private),
    )
    aggregate = _checked_ppc_add(
        shared_mmap,
        max(planning_private, steady_aggregate_private),
    )
    return PPCAllocationEstimate(
        job_accumulator_bytes=job_accumulator_bytes,
        observed_trial_statistics_bytes=observed_trial_statistics_bytes,
        observed_gather_temporary_bytes=observed_gather_temporary_bytes,
        kernel_working_bytes=kernel_working_bytes,
        geometry_bytes=geometry_bytes,
        condition_membership_bytes=membership_bytes,
        source_trial_spike_count_bytes=full_count_bytes,
        planning_working_bytes=planning_working,
        planner_array_bytes=planned_arrays,
        planned_planning_private_bytes=planning_private,
        summary_assembly_bytes=summary_assembly_bytes,
        worker_plan_bytes=worker_plan,
        worker_summary_bytes=worker_summary_bytes,
        checkpoint_block_bytes=checkpoint_block_bytes,
        planned_computation_private_bytes=computation_private,
        planned_parent_private_bytes=parent_private,
        planned_worker_private_bytes=worker_private,
        shared_phase_mmap_bytes=shared_mmap,
        planned_aggregate_array_bytes=aggregate,
        active_worker_count=active_workers,
    )


def _planning_array_bytes(
    drafts: Sequence[dict[str, object]],
    union_edge_count: int,
) -> int:
    """Count retained component or worker planner arrays in bytes.

    Parameters
    ----------
    drafts : Sequence[dict[str, object]]
        Planner-owned draft mappings containing int64 ``selected_trial_rows``
        ``(selected_trial,)`` and local ``schedule``
        ``(shuffle, selected_trial)`` arrays. Values are categorical IDs and
        local positions, with no physical units.
    union_edge_count : int
        Nonnegative number of site-qualified physical union edges.

    Returns
    -------
    int
        Exact byte count ``8*selected_rows + 32*schedule_cells +
        24*union_edges``. The 32-byte schedule term accounts for the local
        schedule plus three final stable-edge/union-position maps.

    Raises
    ------
    ValueError
        If checked count/byte arithmetic exceeds signed int64 range.
    """
    selected_rows = 0
    schedule_cells = 0
    for draft in drafts:
        selected_rows = _checked_ppc_add(
            selected_rows, int(np.asarray(draft["selected_trial_rows"]).size)
        )
        schedule_cells = _checked_ppc_add(
            schedule_cells, int(np.asarray(draft["schedule"]).size)
        )
    return _checked_ppc_add(
        _checked_ppc_multiply(8, selected_rows),
        _checked_ppc_multiply(32, schedule_cells),
        _checked_ppc_multiply(24, union_edge_count),
    )


def _stable_edge_union(
    drafts: Sequence[dict[str, object]],
    *,
    include_site: bool,
) -> list[tuple[int, int, int]]:
    """Derive a sorted physical-edge union from local schedule drafts.

    Parameters
    ----------
    drafts : Sequence[dict[str, object]]
        Planner-owned drafts containing stable int64 ``selected_trial_rows``
        ``(selected_trial,)``, local int64 ``schedule``
        ``(shuffle, selected_trial)``, and an integer ``site_index``. Trial
        rows and sites are categorical identities without physical units.
    include_site : bool
        If true, return site-qualified component edges. If false, collapse all
        drafts to site 0 for an isolated current-site worker-plan estimate.

    Returns
    -------
    list[tuple[int, int, int]]
        Lexicographically sorted unique ``(site, stable_source_row,
        stable_target_row)`` physical edge identities.

    Notes
    -----
    Drafts retain only selected rows and local schedules until final job arrays
    are needed, avoiding duplicate full stable-edge matrices during planning.
    """
    edges: set[tuple[int, int, int]] = set()
    for draft in drafts:
        selected = np.asarray(draft["selected_trial_rows"])
        schedule = np.asarray(draft["schedule"])
        site_index = int(draft["site_index"]) if include_site else 0
        for source_position, target_position in enumerate(schedule.T):
            edges.update(
                (site_index, int(selected[source_position]), int(selected[target]))
                for target in target_position
            )
    return sorted(edges)


def _fixed_edge_position_groups(
    *,
    positions: Sequence[int],
    maximum_group_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Partition one batch's physical-union positions without shifting boundaries.

    Parameters
    ----------
    positions : Sequence[int]
        Nonnegative site-local or component-union edge positions for one
        condition batch. Duplicate positions are allowed at input and denote
        the same physical edge.
    maximum_group_size : int
        Positive maximum number of sorted unique union positions in one fixed
        group. It is the configured ``trial_edge_block_size`` and has no
        physical units.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Sorted unique positions partitioned in deterministic fixed-size groups.
        Empty input returns an empty tuple. A later executor may discard
        inactive positions only within these returned group boundaries.

    Raises
    ------
    ValueError
        If a position is negative or the requested group size is not a
        positive non-Boolean signed-int64 integer.
    """
    group_size = _checked_ppc_int(
        maximum_group_size,
        "trial_edge_block_size",
        minimum=1,
    )
    unique_positions: set[int] = set()
    for raw_position in positions:
        position = _checked_ppc_int(raw_position, "edge union position")
        unique_positions.add(position)
    ordered_positions = tuple(sorted(unique_positions))
    return tuple(
        ordered_positions[start : start + group_size]
        for start in range(0, len(ordered_positions), group_size)
    )


def _validate_grouped_planning_inputs(
    *,
    condition_names: Sequence[str],
    condition_membership: np.ndarray,
    site_ids: Sequence[str],
    site_trial_valid: np.ndarray,
    stable_trial_rows: np.ndarray,
    source_trial_spike_count: np.ndarray,
    frequency_count: int,
    shared_phase_mmap_bytes: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    int,
    int,
]:
    """Validate grouped-planner categorical axes without changing ownership.

    Parameters
    ----------
    condition_names : Sequence[str]
        Unique nonempty condition labels in the input condition-axis order.
    condition_membership : numpy.ndarray
        Boolean ``(full_trial, condition)`` membership matrix.
    site_ids : Sequence[str]
        Unique nonempty site labels in input site-axis order.
    site_trial_valid : numpy.ndarray
        Boolean ``(site, full_trial)`` eligibility matrix.
    stable_trial_rows : numpy.ndarray
        Unique nonnegative int64 ``(full_trial,)`` trial-table identities.
    source_trial_spike_count : numpy.ndarray
        Nonnegative int64 ``(full_trial, unit, segment=2)`` before/after spike
        counts. Values are counts, not seconds or Hz.
    frequency_count : int
        Positive frequency-axis length; the coordinate itself is in Hz.
    shared_phase_mmap_bytes : int
        Nonnegative shared phase-cache allocation in bytes.

    Returns
    -------
    tuple
        Original non-copying membership, validity, stable-row, and count
        arrays; the original condition-name tuple when supplied as a tuple
        (otherwise a tuple copy); and canonical Python frequency count and
        mmap bytes.

    Raises
    ------
    ValueError
        If dtypes, shapes, categorical labels, unique trial IDs, negative
        values, or scalar bounds are invalid.
    """
    membership = np.asarray(condition_membership)
    valid = np.asarray(site_trial_valid)
    stable_rows = np.asarray(stable_trial_rows)
    source_counts = np.asarray(source_trial_spike_count)
    if (
        membership.dtype != np.dtype(bool)
        or membership.ndim != 2
        or membership.shape[0] < 1
        or membership.shape[1] < 1
    ):
        raise ValueError("condition_membership must be Boolean (trial, condition)")
    if (
        valid.dtype != np.dtype(bool)
        or valid.ndim != 2
        or valid.shape[0] != len(site_ids)
        or valid.shape[1] != membership.shape[0]
    ):
        raise ValueError("site_trial_valid must be Boolean (site, trial)")
    if (
        stable_rows.dtype != np.dtype(np.int64)
        or stable_rows.ndim != 1
        or stable_rows.size != membership.shape[0]
    ):
        raise ValueError("stable_trial_rows must be unique int64 trial identities")
    seen_stable_rows: set[int] = set()
    for raw_stable_row in stable_rows:
        stable_row = int(raw_stable_row)
        if stable_row < 0 or stable_row in seen_stable_rows:
            raise ValueError("stable_trial_rows must be unique int64 trial identities")
        seen_stable_rows.add(stable_row)
    if (
        source_counts.dtype != np.dtype(np.int64)
        or source_counts.ndim != 3
        or source_counts.shape[0] != membership.shape[0]
        or source_counts.shape[1] < 1
        or source_counts.shape[2] != 2
    ):
        raise ValueError("source_trial_spike_count must be nonnegative int64 (trial, unit, 2)")
    for raw_count in source_counts.flat:
        if int(raw_count) < 0:
            raise ValueError(
                "source_trial_spike_count must be nonnegative int64 (trial, unit, 2)"
            )
    if len(condition_names) != membership.shape[1] or (
        any(not isinstance(name, str) or not name for name in condition_names)
        or len(set(condition_names)) != len(condition_names)
    ):
        raise ValueError("condition_names must be unique names for every condition")
    if (
        len(site_ids) < 1
        or any(not isinstance(site_id, str) or not site_id for site_id in site_ids)
        or len(set(site_ids)) != len(site_ids)
    ):
        raise ValueError("site_ids must be unique nonempty stable identities")
    frequencies = _checked_ppc_int(frequency_count, "frequency_count", minimum=1)
    shared_mmap = _checked_ppc_int(
        shared_phase_mmap_bytes, "shared_phase_mmap_bytes"
    )
    return (
        membership,
        valid,
        stable_rows,
        source_counts,
        condition_names
        if isinstance(condition_names, tuple)
        else tuple(condition_names),
        frequencies,
        shared_mmap,
    )


def plan_grouped_ppc_component(
    *,
    config: LFPSummaryConfig,
    condition_names: Sequence[str],
    condition_membership: np.ndarray,
    site_ids: Sequence[str],
    site_trial_valid: np.ndarray,
    stable_trial_rows: np.ndarray,
    source_trial_spike_count: np.ndarray,
    frequency_count: int,
    shared_phase_mmap_bytes: int,
) -> PPCComponentPlan:
    """Build a deterministic immutable grouped PPC plan without numerical work.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated analysis configuration. This planner uses PPC seed, shuffle
        count, epoch order, phase-band/bin dimensions, and execution memory,
        batching, and worker settings. It does not modify the configuration.
    condition_names : Sequence[str]
        Unique nonempty condition labels in exact ``condition``-axis order.
    condition_membership : numpy.ndarray
        Boolean ``(full_trial, condition)`` membership matrix. ``full_trial``
        is the stable trial-table row order, and ``condition`` follows
        ``condition_names``.
    site_ids : Sequence[str]
        Unique nonempty site labels in exact ``site``-axis order.
    site_trial_valid : numpy.ndarray
        Boolean ``(site, full_trial)`` eligibility matrix. A job selects the
        intersection of one condition column and one site row; this prevents
        nested or overlapping conditions from widening a site's trial pool.
    stable_trial_rows : numpy.ndarray
        Unique nonnegative int64 ``(full_trial,)`` stable trial-table row IDs.
        These categorical IDs are copied into output jobs and are not local
        condition positions.
    source_trial_spike_count : numpy.ndarray
        Nonnegative int64 ``(full_trial, unit, segment=2)`` spike counts.
        Segment 0 is before-event and segment 1 is after-event. Counts have no
        physical unit; they are used only for allocation planning.
    frequency_count : int
        Positive number of frequency bins. The associated analysis coordinate
        is in Hz, while this argument is a dimension count.
    shared_phase_mmap_bytes : int
        Nonnegative bytes in the shared read-only phase cache/mmap. The count
        is included once in each aggregate allocation estimate.

    Returns
    -------
    PPCComponentPlan
        Pure site-major, then condition, then ``whole``/``before``/``after``
        job records with owned read-only local schedules, stable edge mappings,
        a site-qualified physical-edge union, deterministic condition batches,
        and the worst bounded allocation estimate. Phase values are never
        read or sampled.

    Raises
    ------
    ValueError
        If configuration or categorical axes are invalid; trial rows are not
        unique nonnegative int64 IDs; source count axes disagree; requested
        memory cannot fit even a singleton condition batch; or checked seed or
        allocation arithmetic overflows.

    Notes
    -----
    The function is pure and side-effect free: it performs no I/O, worker
    spawning, checkpointing, phase/spike sampling, or calls to
    :func:`execute_ppc_blocks`. Candidate batches are tested against both the
    applicable parent/worker process peak and aggregate peak, splitting in
    input condition order until singleton batches fit or are rejected.
    """
    _validate_ppc_execution(config.ppc_execution)
    (
        membership,
        site_valid,
        stable_rows,
        source_counts,
        names,
        frequencies,
        shared_mmap,
    ) = _validate_grouped_planning_inputs(
        condition_names=condition_names,
        condition_membership=condition_membership,
        site_ids=site_ids,
        site_trial_valid=site_trial_valid,
        stable_trial_rows=stable_trial_rows,
        source_trial_spike_count=source_trial_spike_count,
        frequency_count=frequency_count,
        shared_phase_mmap_bytes=shared_phase_mmap_bytes,
    )
    shuffle_count = _checked_ppc_int(
        config.ppc.shuffle_count, "config.ppc.shuffle_count", minimum=1
    )
    base_seed = _checked_ppc_int(config.ppc.seed, "config.ppc.seed")
    epochs = tuple(config.ppc.epochs)
    expressions = {"whole": "before + after", "before": "before", "after": "after"}
    if tuple(epochs) != ("whole", "before", "after"):
        raise ValueError("PPC epochs must remain whole, before, after")

    drafts_by_site: list[list[dict[str, object]]] = []
    allowed_edges: set[tuple[int, int, int]] = set()
    scheduled_edge_count = 0
    independent_edge_count = 0
    for site_index, site_id in enumerate(site_ids):
        site_drafts: list[dict[str, object]] = []
        for condition_index, condition_name in enumerate(names):
            local_positions = np.flatnonzero(
                membership[:, condition_index] & site_valid[site_index]
            ).astype(np.int64, copy=False)
            selected_rows = stable_rows[local_positions].astype(np.int64, copy=True)
            if selected_rows.size > 1:
                for source_row in selected_rows:
                    for target_row in selected_rows:
                        if source_row != target_row:
                            allowed_edges.add(
                                (site_index, int(source_row), int(target_row))
                            )
            for epoch_index, epoch_name in enumerate(epochs):
                schedule_seed = _checked_ppc_add(
                    base_seed,
                    _checked_ppc_multiply(condition_index, 1_000_000),
                    _checked_ppc_multiply(site_index, 10_000),
                    _checked_ppc_multiply(epoch_index, 100),
                )
                if selected_rows.size < 2:
                    schedule = np.empty((0, selected_rows.size), dtype=np.int64)
                else:
                    schedule = _validate_generated_derangement_schedule(
                        generate_trial_derangement_schedule(
                            selected_rows.size,
                            shuffle_count,
                            seed=schedule_seed,
                        ),
                        trial_count=int(selected_rows.size),
                        shuffle_count=shuffle_count,
                    )
                scheduled_edge_count = _checked_ppc_add(
                    scheduled_edge_count, int(schedule.size)
                )
                independent_edge_count = _checked_ppc_add(
                    independent_edge_count,
                    len(
                        {
                            (int(selected_rows[source]), int(selected_rows[target]))
                            for source, target_row in enumerate(schedule.T)
                            for target in target_row
                        }
                    ),
                )
                site_drafts.append(
                    {
                        "condition_index": condition_index,
                        "condition_name": str(condition_name),
                        "site_index": site_index,
                        "site_id": str(site_id),
                        "epoch_index": epoch_index,
                        "epoch_name": epoch_name,
                        "selected_trial_rows": selected_rows.copy(),
                        "schedule": schedule,
                        "schedule_seed": schedule_seed,
                    }
                )
        drafts_by_site.append(site_drafts)

    all_drafts = tuple(draft for site_drafts in drafts_by_site for draft in site_drafts)
    union_edges = _stable_edge_union(all_drafts, include_site=True)
    planner_array_bytes = _planning_array_bytes(all_drafts, len(union_edges))
    total_units = source_counts.shape[1]
    unit_block_size = config.ppc_execution.unit_block_size
    condition_membership_bytes = _checked_ppc_multiply(
        membership.shape[0], membership.shape[1]
    )
    source_trial_spike_count_bytes = _checked_ppc_multiply(
        16, membership.shape[0], total_units
    )
    selected_row_cells = 0
    schedule_cells = 0
    for draft in all_drafts:
        selected_row_cells = _checked_ppc_add(
            selected_row_cells,
            int(np.asarray(draft["selected_trial_rows"]).size),
        )
        schedule_cells = _checked_ppc_add(
            schedule_cells,
            int(np.asarray(draft["schedule"]).size),
        )
    draft_array_bytes = _checked_ppc_add(
        _checked_ppc_multiply(8, selected_row_cells),
        _checked_ppc_multiply(8, schedule_cells),
    )
    trial_count = membership.shape[0]
    maximum_edge_count = _checked_ppc_multiply(trial_count, trial_count - 1)
    largest_unit_block = min(unit_block_size, total_units)
    planning_scratch_bytes = _checked_ppc_add(
        _checked_ppc_multiply(8, _checked_ppc_add(2 * trial_count, maximum_edge_count)),
        _checked_ppc_multiply(16, trial_count, largest_unit_block),
    )
    planning_working_bytes = _checked_ppc_add(
        draft_array_bytes, planning_scratch_bytes
    )
    pending_blocks = (total_units + unit_block_size - 1) // unit_block_size
    component_job_count = len(all_drafts)
    representative_band_count = _S1_S2_REPRESENTATIVE_BAND_COUNT
    phase_bin_count = len(config.ppc.phase_bin_edges_rad) - 1
    summary_cell_bytes = _checked_ppc_add(
        _checked_ppc_multiply(116, frequencies),
        _checked_ppc_multiply(8, representative_band_count, phase_bin_count),
    )
    largest_site_job_count = max((len(site_drafts) for site_drafts in drafts_by_site), default=0)
    largest_checkpoint_block_bytes = _checked_ppc_add(
        _checked_ppc_multiply(
            largest_unit_block, largest_site_job_count, summary_cell_bytes
        ),
        3 * 8,
    )
    parent_parallel_bytes = _checked_ppc_add(
        planner_array_bytes,
        _checked_ppc_multiply(total_units, component_job_count, summary_cell_bytes),
        largest_checkpoint_block_bytes,
    )
    maximum_private = config.ppc_execution.maximum_worker_allocation_bytes
    maximum_aggregate = config.ppc_execution.maximum_aggregate_allocation_bytes
    if config.ppc_execution.worker_count > 1 and parent_parallel_bytes > maximum_private:
        raise ValueError("parent planned private allocation exceeds maximum_worker_allocation_bytes")

    def site_source_positions(site_index: int) -> np.ndarray:
        """Return one site's observed source-trial positions for allocation.

        Parameters
        ----------
        site_index : int
            Valid nonnegative position on the ``site`` axis of
            ``site_trial_valid``.

        Returns
        -------
        numpy.ndarray
            Int64 ``(source_trial,)`` positions on the full-trial axis that
            are site-valid and belong to at least one condition. Positions are
            categorical row offsets, not stable IDs or physical times.

        Notes
        -----
        The enclosing planner already validates the site axis; this helper has
        no I/O or numerical PPC side effects.
        """
        selected = site_valid[site_index] & np.any(membership, axis=1)
        return np.flatnonzero(selected).astype(np.int64, copy=False)

    def batch_estimates(
        site_index: int,
        condition_batch: tuple[int, ...],
    ) -> list[PPCAllocationEstimate]:
        """Estimate every bounded unit/edge work item for one site batch.

        Parameters
        ----------
        site_index : int
            Valid position on the input ``site`` axis.
        condition_batch : tuple[int, ...]
            Nonempty ordered positions on the input ``condition`` axis. These
            select null work only; all site job summaries remain retained.

        Returns
        -------
        list[PPCAllocationEstimate]
            One pure byte estimate for each current ``unit_block`` and bounded
            null ``edge_block``. Source counts retain axes
            ``(source_trial, unit_block, before_after=2)``; all result memory
            fields are bytes.

        Raises
        ------
        ValueError
            If delegated checked allocation arithmetic or planner positions
            are invalid. The enclosing function has already validated input
            categorical axes.
        """
        site_drafts = drafts_by_site[site_index]
        batch_drafts = tuple(
            draft
            for draft in site_drafts
            if int(draft["condition_index"]) in condition_batch
        )
        site_union = _stable_edge_union(site_drafts, include_site=False)
        worker_plan_bytes = _planning_array_bytes(site_drafts, len(site_union))
        batch_union = _stable_edge_union(batch_drafts, include_site=False)
        site_union_position = {
            edge: position for position, edge in enumerate(site_union)
        }
        fixed_batch_groups = _fixed_edge_position_groups(
            positions=tuple(site_union_position[edge] for edge in batch_union),
            maximum_group_size=config.ppc_execution.trial_edge_block_size,
        )
        source_positions = site_source_positions(site_index)
        active_job_count = sum(
            1 for draft in batch_drafts if np.asarray(draft["schedule"]).size
        )
        worker_result_job_count = len(site_drafts)
        if source_positions.size:
            source_row_to_position = {
                int(stable_rows[position]): local
                for local, position in enumerate(source_positions)
            }
        else:
            source_row_to_position = {}
        estimates: list[PPCAllocationEstimate] = []
        for unit_start in range(0, total_units, unit_block_size):
            unit_stop = min(unit_start + unit_block_size, total_units)
            if source_positions.size:
                block_counts = source_counts[
                    source_positions, unit_start:unit_stop, :
                ]
                observed_positions = np.arange(source_positions.size, dtype=np.int64)
            else:
                # The allocator requires one source-axis row; no observed/null
                # positions make this a harmless zero-work placeholder.
                block_counts = np.zeros((1, unit_stop - unit_start, 2), dtype=np.int64)
                observed_positions = np.empty(0, dtype=np.int64)
            edge_groups = fixed_batch_groups if fixed_batch_groups else ((),)
            for edge_group in edge_groups:
                edge_block = np.empty(len(edge_group), dtype=np.int64)
                for edge_offset, union_position in enumerate(edge_group):
                    _, source_row, _ = site_union[union_position]
                    edge_block[edge_offset] = source_row_to_position[source_row]
                estimates.append(
                    estimate_grouped_ppc_allocation(
                        active_job_count=active_job_count,
                        worker_result_job_count=worker_result_job_count,
                        component_job_count=component_job_count,
                        shuffle_count=shuffle_count,
                        total_unit_count=total_units,
                        unit_block_size=unit_stop - unit_start,
                        source_trial_spike_count=block_counts,
                        edge_source_trial_position=edge_block,
                        observed_source_trial_position=observed_positions,
                        frequency_count=frequencies,
                        representative_band_count=representative_band_count,
                        phase_bin_count=phase_bin_count,
                        planner_array_bytes=planner_array_bytes,
                        worker_plan_bytes=worker_plan_bytes,
                        worker_count=config.ppc_execution.worker_count,
                        pending_unit_block_count=pending_blocks,
                        shared_phase_mmap_bytes=shared_mmap,
                        condition_membership_bytes=condition_membership_bytes,
                        source_trial_spike_count_bytes=source_trial_spike_count_bytes,
                        planning_working_bytes=planning_working_bytes,
                    )
                )
        return estimates

    def batch_limit_failure(
        estimates: Sequence[PPCAllocationEstimate],
    ) -> str | None:
        """Return the first allocation bound violated by a candidate batch.

        Parameters
        ----------
        estimates : Sequence[PPCAllocationEstimate]
            Pure candidate work-item estimates with byte-valued parent, worker,
            and aggregate lifetime peaks.

        Returns
        -------
        str or None
            ``"parent"`` for a serial parent limit, ``"worker"`` for a
            parallel worker limit, ``"aggregate"`` for total allocation, or
            ``None`` when every item fits both configured byte limits.

        Notes
        -----
        The shared phase mmap is already counted exactly once by each
        aggregate estimate. This helper does not allocate phase/spike arrays
        or start workers.
        """
        process_peak = max(
            (
                estimate.planned_parent_private_bytes
                if config.ppc_execution.worker_count == 1
                else estimate.planned_worker_private_bytes
                for estimate in estimates
            ),
            default=0,
        )
        if process_peak > maximum_private:
            return "parent" if config.ppc_execution.worker_count == 1 else "worker"
        aggregate_peak = max(
            (estimate.planned_aggregate_array_bytes for estimate in estimates),
            default=0,
        )
        if aggregate_peak > maximum_aggregate:
            return "aggregate"
        return None

    condition_batches: list[tuple[tuple[int, ...], ...]] = []
    all_estimates: list[PPCAllocationEstimate] = []
    for site_index in range(len(site_ids)):
        site_batches: list[tuple[int, ...]] = []
        current: tuple[int, ...] = ()
        for condition_index in range(membership.shape[1]):
            candidate = current + (condition_index,)
            candidate_estimates = batch_estimates(site_index, candidate)
            failure = batch_limit_failure(candidate_estimates)
            if failure is None:
                current = candidate
            elif current:
                site_batches.append(current)
                current = (condition_index,)
                single_estimates = batch_estimates(site_index, current)
                single_failure = batch_limit_failure(single_estimates)
                if single_failure is not None:
                    raise ValueError(
                        f"{single_failure} planned allocation exceeds its configured limit"
                    )
            else:
                raise ValueError(
                    f"{failure} planned allocation exceeds its configured limit"
                )
        if current:
            site_batches.append(current)
        condition_batches.append(tuple(site_batches))
        for batch in site_batches:
            all_estimates.extend(batch_estimates(site_index, batch))

    if not all_estimates:
        raise ValueError("grouped PPC plan requires at least one site and condition")
    if config.ppc_execution.worker_count == 1:
        allocation_estimate = max(
            all_estimates, key=lambda estimate: estimate.planned_parent_private_bytes
        )
    else:
        allocation_estimate = max(
            all_estimates, key=lambda estimate: estimate.planned_worker_private_bytes
        )
    if allocation_estimate.planned_parent_private_bytes > maximum_private:
        raise ValueError("parent planned private allocation exceeds maximum_worker_allocation_bytes")
    if allocation_estimate.planned_worker_private_bytes > maximum_private:
        raise ValueError("worker planned private allocation exceeds maximum_worker_allocation_bytes")
    if (
        allocation_estimate.planned_aggregate_array_bytes
        > config.ppc_execution.maximum_aggregate_allocation_bytes
    ):
        raise ValueError("aggregate planned allocation exceeds maximum_aggregate_allocation_bytes")

    union_position = {edge: index for index, edge in enumerate(union_edges)}
    job_plans: list[PPCJobPlan] = []
    for draft in all_drafts:
        selected = np.asarray(draft["selected_trial_rows"])
        schedule = np.asarray(draft["schedule"])
        source = np.broadcast_to(selected[np.newaxis, :], schedule.shape).copy()
        target = (
            selected[schedule]
            if schedule.size
            else np.empty(schedule.shape, dtype=np.int64)
        )
        site_index = int(draft["site_index"])
        positions = np.fromiter(
            (
                union_position[(site_index, int(left), int(right))]
                for left, right in zip(source.ravel(), target.ravel(), strict=True)
            ),
            dtype=np.int64,
            count=source.size,
        ).reshape(source.shape)
        job_plans.append(
            _trusted_ppc_job_plan(
                condition_index=int(draft["condition_index"]),
                condition_name=str(draft["condition_name"]),
                site_index=site_index,
                site_id=str(draft["site_id"]),
                epoch_index=int(draft["epoch_index"]),
                epoch_name=str(draft["epoch_name"]),
                selected_trial_rows=selected,
                schedule=schedule,
                stable_edge_source_trial_row=source,
                stable_edge_target_trial_row=target,
                edge_union_position=positions,
                segment_expression=expressions[str(draft["epoch_name"])],
                base_ppc_seed=base_seed,
                schedule_seed=int(draft["schedule_seed"]),
                condition_derivation_identity=int(draft["condition_index"]),
                site_derivation_identity=site_index,
                epoch_derivation_identity=int(draft["epoch_index"]),
                schedule_shape=tuple(int(value) for value in schedule.shape),
                schedule_fingerprint=_array_fingerprint(schedule),
            )
        )

    # These new int64 vectors are already planner-owned; the trusted factory
    # seals them rather than making a second full union copy.
    union_site = np.asarray([site for site, _, _ in union_edges], dtype=np.int64)
    union_source = np.asarray([source for _, source, _ in union_edges], dtype=np.int64)
    union_target = np.asarray([target for _, _, target in union_edges], dtype=np.int64)
    union_edge_count = len(union_edges)
    saturation = (
        float(union_edge_count) / float(len(allowed_edges)) if allowed_edges else 0.0
    )
    reuse = (
        float(independent_edge_count) / float(union_edge_count)
        if union_edge_count
        else 0.0
    )
    return _trusted_ppc_component_plan(
        job_plans=tuple(job_plans),
        edge_site_index=union_site,
        stable_edge_source_trial_row=union_source,
        stable_edge_target_trial_row=union_target,
        condition_batches=tuple(condition_batches),
        scheduled_edge_count=scheduled_edge_count,
        independent_edge_count=independent_edge_count,
        union_edge_count=union_edge_count,
        edge_union_saturation=saturation,
        edge_reuse_ratio=reuse,
        allocation_estimate=allocation_estimate,
    )


_GROUPED_PPC_CODE_VERSION = "s4-grouped-serial-v1"
_GROUPED_PPC_KERNEL_VERSION = "segmented-kernel-v1"


@dataclass(frozen=True)
class PPCComponentExecutionResult:
    """Completed grouped PPC work result for one full Spike-phase component.

    Parameters
    ----------
    run_fingerprint : str
        SHA-256 execution identity binding full prepared phase/spike content,
        stable trial/site/job schedules, segment settings, grouped code/kernel
        versions, and all execution-only controls. It has no physical units.
    run_directory : pathlib.Path
        Exact ``<work_root>/ppc/<run_fingerprint>`` work directory containing
        metadata and, when enabled, independently resumable bounded blocks.
    component_plan : PPCComponentPlan
        Immutable site-major grouped plan used for this execution. It contains
        categorical trial/site identities and no phase or spike samples.
    summary_arrays : dict[str, numpy.ndarray]
        Owned read-only arrays. Metric/count/flag fields have axes
        ``(unit, condition, site, epoch, frequency)``; PPC/resultant/null
        values are dimensionless, preferred phase is radians, and count/flag
        units are documented by their names. ``representative_phase_histogram_count``
        has ``(unit, condition, site, epoch, band, phase_bin)`` axes.
    completed_block_ids, resumed_block_ids : tuple[str, ...]
        Safe site/unit-block checkpoint identities in stable site-major then
        increasing half-open unit-bound order. The second tuple is the subset
        loaded from exact valid single-block checkpoints.
    planning_seconds, grouped_execution_seconds : float
        Finite nonnegative wall-clock seconds spent before versus after the
        immutable grouped component plan was created.
    """

    run_fingerprint: str
    run_directory: Path
    component_plan: PPCComponentPlan
    summary_arrays: dict[str, np.ndarray]
    completed_block_ids: tuple[str, ...]
    resumed_block_ids: tuple[str, ...]
    planning_seconds: float
    grouped_execution_seconds: float


def execute_grouped_ppc_component(
    *,
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    work_root: Path,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> PPCComponentExecutionResult:
    """Execute one grouped PPC component with serial or site-local workers.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated scientific configuration. The prepared phase axes are site,
        frequency in Hz, full trial, and relative time in seconds. Its PPC
        epochs remain ``whole, before, after`` and scientific fingerprint is
        unchanged by this work-only executor.
    execution : PPCExecutionConfig
        Must equal ``config.ppc_execution``. Unit, edge, and shuffle blocks
        bound transient work only; all settings enter the execution identity.
        ``worker_count == 1`` preserves the serial path. Larger counts create
        a spawned pool for one current site at a time, capped by that site's
        pending unit blocks; sites and conditions never co-reside as tasks.
    prepared_phase : PreparedPhaseRun-like object
        Production record with complex64/Boolean ``(site, frequency, trial,
        time)`` phase/valid axes, int64 stable trial rows, per-site validity,
        and trial condition/filter/objective/user information.
    prepared_spikes : PreparedSpikeRun-like object
        Production record with configured stable unit order and one finite
        relative-seconds spike vector per full trial and unit.
    work_root : pathlib.Path
        Session-local parent for execution-only ``ppc/<fingerprint>`` files.
        This function never publishes a final component, manifest, or
        ``spike_phase.npz`` file.
    progress_callback : callable or None, default=None
        Receives parent-owned canonical ``ProgressEvent`` records only. Events
        carry no raw phase, spike, or full schedule tensors.

    Returns
    -------
    PPCComponentExecutionResult
        Read-only full-component summaries and an immutable plan. Whole
        observed statistics are before-plus-after sufficient statistics while
        whole null schedules retain their independently derived seeds.

    Raises
    ------
    ValueError
        If execution is mismatched, prepared records are malformed, or planner
        allocation limits fail. A cached block with incompatible
        documented axes, marker, metadata, or bounded NPZ schema is discarded
        and recomputed rather than raising. Memory preflight occurs before
        geometry, summary allocation, checkpoint I/O, or process construction.
    FileExistsError
        If another executor owns this exact run directory.
    OSError
        If metadata or a parent-owned checkpoint cannot be atomically written.
    """
    _validate_ppc_execution(execution)
    if execution != config.ppc_execution:
        raise ValueError("grouped PPC execution must equal config.ppc_execution")
    planning_started = time.perf_counter()

    # Import locally: the production runtime imports this module for legacy
    # job execution, while this serial bridge only needs its prepared-record
    # validation and condition intersection after execution was authorized.
    from src.neural_analysis import lfp_summary_runtime

    lfp_summary_runtime._validate_prepared_phase_run(config, prepared_phase)
    lfp_summary_runtime._validate_prepared_spike_run(
        config, prepared_phase, prepared_spikes
    )
    phase = np.asarray(prepared_phase.phase_tensor)
    valid = np.asarray(prepared_phase.phase_valid, dtype=bool)
    stable_rows = np.asarray(prepared_phase.trial_indices)
    site_valid = np.asarray(prepared_phase.site_valid, dtype=bool)
    parallel_shared_worker_input_bytes = _checked_ppc_add(
        int(phase.nbytes), int(valid.nbytes),
        int(np.asarray(prepared_phase.relative_time_s).nbytes), int(stable_rows.nbytes),
        _checked_ppc_multiply(
            8,
            sum(
                int(np.asarray(times).size)
                for train in prepared_spikes.trial_spike_trains
                for times in train.relative_spike_times
            ),
        ),
        _checked_ppc_multiply(8, len(prepared_spikes.unit_ids), stable_rows.size + 1),
    )
    shared_worker_input_bytes = (
        parallel_shared_worker_input_bytes
        if execution.worker_count > 1
        else _checked_ppc_add(int(phase.nbytes), int(valid.nbytes))
    )
    _preflight_grouped_planning_construction(
        config=config,
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        shared_phase_mmap_bytes=shared_worker_input_bytes,
    )
    membership = lfp_summary_runtime._analysis_condition_membership(
        prepared_phase.prepared_trials
    )
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    source_counts = _grouped_source_trial_spike_counts(
        config=config,
        prepared_spikes=prepared_spikes,
        trial_count=stable_rows.size,
    )
    plan = plan_grouped_ppc_component(
        config=config,
        condition_names=prepared_phase.prepared_trials.condition_names,
        condition_membership=membership,
        site_ids=tuple(site.stable_id for site in config.sites),
        site_trial_valid=site_valid,
        stable_trial_rows=stable_rows,
        source_trial_spike_count=source_counts,
        frequency_count=frequencies_hz.size,
        shared_phase_mmap_bytes=shared_worker_input_bytes,
    )
    metadata = _grouped_run_metadata(
        config=config,
        execution=execution,
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        plan=plan,
        condition_membership=membership,
    )
    # The run identity has consumed planning-only masks/counts. They are not
    # retained through full parent summary allocation or checkpoint publication.
    condition_count = int(membership.shape[1])
    del membership, source_counts
    run_fingerprint = str(metadata["run_fingerprint"])
    run_directory = Path(work_root) / "ppc" / run_fingerprint
    planning_seconds = max(0.0, time.perf_counter() - planning_started)
    execution_started = time.perf_counter()
    reporter = _ProgressReporter(progress_callback, "grouped-spike-phase")
    reporter.emit("prepare_phase", 1, 1, "validated grouped prepared inputs")

    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_executor_lock(run_directory, run_fingerprint)
    active_parallel_runners: list[object] = []
    try:
        # Any resumed/repaired run must re-earn the final completion marker.
        (run_directory / "complete.json").unlink(missing_ok=True)
        existing_metadata = _load_grouped_metadata(run_directory)
        resumable = existing_metadata == metadata
        _atomic_json(run_directory / "metadata.json", metadata)
        summary = _empty_grouped_summary_arrays(
            unit_count=len(prepared_spikes.unit_ids),
            condition_count=condition_count,
            site_count=len(config.sites),
            epoch_count=len(config.ppc.epochs),
            frequency_count=frequencies_hz.size,
            representative_band_count=_S1_S2_REPRESENTATIVE_BAND_COUNT,
            phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
        )
        block_identities = metadata["block_identities"]
        assert isinstance(block_identities, dict)
        completed_ids = tuple(block_identities)
        block_records: list[tuple[str, int, int, int]] = []
        resumed_ids: list[str] = []
        resumed_id_set: set[str] = set()
        for block_id in completed_ids:
            identity = block_identities[block_id]
            assert isinstance(identity, dict)
            site_index = int(identity["site_index"])
            unit_start = int(identity["unit_start"])
            unit_stop = int(identity["unit_stop"])
            block_records.append((block_id, site_index, unit_start, unit_stop))
            checkpoint_schema = _grouped_checkpoint_schema(
                unit_count=unit_stop - unit_start,
                condition_count=condition_count,
                epoch_count=len(config.ppc.epochs),
                frequency_count=frequencies_hz.size,
                representative_band_count=_S1_S2_REPRESENTATIVE_BAND_COUNT,
                phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
            )
            checkpoint = (
                load_valid_ppc_checkpoint(
                    run_directory,
                    block_id,
                    metadata,
                    expected_array_schema=checkpoint_schema,
                    maximum_array_bytes=_grouped_checkpoint_schema_bytes(
                        checkpoint_schema
                    ),
                )
                if resumable and execution.checkpoint_enabled
                else None
            )
            if checkpoint is not None and _merge_grouped_checkpoint(
                summary=summary,
                arrays=checkpoint.arrays,
                site_index=site_index,
                unit_start=unit_start,
                unit_stop=unit_stop,
                condition_count=condition_count,
                epoch_count=len(config.ppc.epochs),
                frequency_count=frequencies_hz.size,
                representative_band_count=_S1_S2_REPRESENTATIVE_BAND_COUNT,
                phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
            ):
                resumed_ids.append(block_id)
                resumed_id_set.add(block_id)
                del checkpoint
            else:
                del checkpoint

        def emit_completed_block(
            completed_count: int,
            checkpoint_message: str,
        ) -> None:
            """Emit parent-owned progress for one canonical completed block.

            ``completed_count`` is a one-based scalar in stable block order;
            ``checkpoint_message`` is metadata only. Returns ``None`` and
            never receives worker arrays or publishes checkpoints.
            """
            total_blocks = len(completed_ids)
            if (
                completed_count % execution.progress_update_interval == 0
                or completed_count == total_blocks
            ):
                reporter.emit("observed_reduction", completed_count, total_blocks, "completed grouped observed reduction", timed=True)
                reporter.emit("trial_edge_reduction", completed_count, total_blocks, "completed grouped edge reduction", timed=True)
                reporter.emit("shuffle_aggregation", completed_count, total_blocks, "completed grouped shuffle aggregation", timed=True)
                reporter.emit("fdr", completed_count, total_blocks, "applied grouped FDR", timed=True)
                reporter.emit("checkpoint", completed_count, total_blocks, checkpoint_message, timed=True)

        if execution.worker_count == 1:
            for completed_count, (block_id, site_index, unit_start, unit_stop) in enumerate(
                block_records,
                start=1,
            ):
                if block_id in resumed_id_set:
                    checkpoint_message = "resumed grouped checkpoint"
                else:
                    _compute_grouped_site_unit_block(
                        config=config,
                        prepared_phase=prepared_phase,
                        prepared_spikes=prepared_spikes,
                        plan=plan,
                        summary=summary,
                        site_index=site_index,
                        unit_start=unit_start,
                        unit_stop=unit_stop,
                    )
                    _apply_grouped_bh_block(
                        summary=summary,
                        site_index=site_index,
                        unit_start=unit_start,
                        unit_stop=unit_stop,
                        alpha=config.ppc.fdr_alpha,
                    )
                    if execution.checkpoint_enabled:
                        block_arrays = _grouped_checkpoint_arrays(
                            summary=summary,
                            site_index=site_index,
                            unit_start=unit_start,
                            unit_stop=unit_stop,
                        )
                        write_ppc_checkpoint(
                            run_directory,
                            block_id,
                            block_arrays,
                            metadata,
                            copy_arrays=False,
                        )
                        del block_arrays
                        checkpoint_message = "newly published grouped checkpoint"
                    else:
                        checkpoint_message = "checkpoint disabled"
                emit_completed_block(completed_count, checkpoint_message)
        else:
            pending_records = tuple(
                record for record in block_records if record[0] not in resumed_id_set
            )
            phase_descriptor = (
                _materialize_grouped_phase_work_inputs(
                    run_directory, phase, valid,
                    np.asarray(prepared_phase.relative_time_s, dtype=np.float64),
                    stable_rows, prepared_spikes,
                )
                if pending_records
                else None
            )
            _clear_grouped_parallel_staging_directory(run_directory / "worker-staging")
            completed_count = 0
            for site_index in range(len(config.sites)):
                site_records = tuple(
                    record for record in block_records if record[1] == site_index
                )
                site_pending = tuple(
                    record for record in site_records if record[0] not in resumed_id_set
                )
                results: Iterator[_GroupedParallelBlockResult] | None = None
                if site_pending:
                    assert phase_descriptor is not None
                    site_jobs = tuple(
                        job for job in plan.job_plans if job.site_index == site_index
                    )
                    (
                        site_union_source_trial_row,
                        site_union_target_trial_row,
                        site_union_position_offset,
                    ) = _component_site_union_views(plan, site_index)
                    tasks = tuple(
                        _GroupedParallelBlockTask(
                            block_id=block_id,
                            site_index=site_index,
                            unit_start=unit_start,
                            unit_stop=unit_stop,
                            site_jobs=site_jobs,
                            site_condition_batches=plan.condition_batches[site_index],
                            site_union_source_trial_row=site_union_source_trial_row,
                            site_union_target_trial_row=site_union_target_trial_row,
                            site_union_position_offset=site_union_position_offset,
                            staged_result_path=(
                                run_directory / "worker-staging" / f"{block_id}.npz"
                            ),
                            config=config,
                        )
                        for block_id, _, unit_start, unit_stop in site_pending
                    )
                    results = iter(
                        _run_grouped_parallel_block_batches(
                            phase_descriptor=phase_descriptor,
                            block_tasks=tasks,
                            worker_count=min(execution.worker_count, len(tasks)),
                            compute_block=_compute_grouped_parallel_block,
                        )
                    )
                    active_parallel_runners.append(results)
                for block_id, record_site_index, unit_start, unit_stop in site_records:
                    completed_count += 1
                    if block_id in resumed_id_set:
                        checkpoint_message = "resumed grouped checkpoint"
                    else:
                        assert results is not None
                        result = next(results)
                        if (
                            result.block_id != block_id
                            or result.site_index != record_site_index
                            or result.unit_start != unit_start
                            or result.unit_stop != unit_stop
                        ):
                            raise RuntimeError("grouped worker returned a noncanonical block")
                        expected_stage_path = (
                            run_directory / "worker-staging" / f"{block_id}.npz"
                        )
                        if result.staged_result_path != expected_stage_path:
                            raise RuntimeError("grouped worker returned a noncanonical staged path")
                        checkpoint_schema = _grouped_checkpoint_schema(
                            unit_count=unit_stop - unit_start,
                            condition_count=condition_count,
                            epoch_count=len(config.ppc.epochs),
                            frequency_count=frequencies_hz.size,
                            representative_band_count=_S1_S2_REPRESENTATIVE_BAND_COUNT,
                            phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
                        )
                        arrays = _load_grouped_staged_result(
                            staged_result_path=result.staged_result_path,
                            expected_array_schema=checkpoint_schema,
                            maximum_array_bytes=_grouped_checkpoint_schema_bytes(
                                checkpoint_schema
                            ),
                        )
                        try:
                            if not _merge_grouped_checkpoint(
                                summary=summary, arrays=arrays,
                                site_index=record_site_index, unit_start=unit_start,
                                unit_stop=unit_stop, condition_count=condition_count,
                                epoch_count=len(config.ppc.epochs),
                                frequency_count=frequencies_hz.size,
                                representative_band_count=_S1_S2_REPRESENTATIVE_BAND_COUNT,
                                phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
                            ):
                                raise RuntimeError("grouped worker block arrays are invalid")
                            if execution.checkpoint_enabled:
                                write_ppc_checkpoint(
                                    run_directory, block_id, arrays, metadata,
                                    copy_arrays=False,
                                )
                                checkpoint_message = "newly published grouped checkpoint"
                            else:
                                checkpoint_message = "checkpoint disabled"
                        finally:
                            del arrays
                            result.staged_result_path.unlink(missing_ok=True)
                            del result
                    emit_completed_block(completed_count, checkpoint_message)
                if results is not None:
                    # Resume the site-local generator after its last yielded
                    # block only after parent checkpoint publication. This
                    # releases the pool before the next site begins and keeps
                    # the yield/checkpoint protocol observable at site edges.
                    try:
                        unexpected = next(results)
                    except StopIteration:
                        pass
                    else:
                        del unexpected
                        raise RuntimeError("grouped worker yielded an extra block")
        _atomic_json(run_directory / "complete.json", {"run_fingerprint": run_fingerprint})
        reporter.emit("commit", 1, 1, "grouped PPC summaries ready")
        return PPCComponentExecutionResult(
            run_fingerprint=run_fingerprint,
            run_directory=run_directory,
            component_plan=plan,
            summary_arrays=_freeze_grouped_summary_arrays(summary),
            completed_block_ids=completed_ids,
            resumed_block_ids=tuple(resumed_ids),
            planning_seconds=planning_seconds,
            grouped_execution_seconds=max(
                0.0,
                time.perf_counter() - execution_started,
            ),
        )
    finally:
        try:
            for runner in active_parallel_runners:
                close = getattr(runner, "close", None)
                if callable(close):
                    close()
        finally:
            try:
                _clear_grouped_parallel_staging_directory(
                    run_directory / "worker-staging"
                )
            finally:
                _release_executor_lock(lock_path)


def _grouped_source_trial_spike_counts(
    *,
    config: LFPSummaryConfig,
    prepared_spikes: object,
    trial_count: int,
) -> np.ndarray:
    """Return exact full-trial before/after spike counts for allocation planning.

    Parameters
    ----------
    config : LFPSummaryConfig
        Contains half-open before/after windows in relative seconds.
    prepared_spikes : PreparedSpikeRun-like object
        One finite relative-seconds spike vector per configured unit/full trial.
    trial_count : int
        Positive full-trial axis length shared by all prepared spike trains.

    Returns
    -------
    numpy.ndarray
        Int64 ``(trial, unit, segment=2)`` nonnegative spike counts. Segment
        zero is before and segment one is after; counts have no physical units.

    Raises
    ------
    ValueError
        If a supplied spike record does not have the documented trial axis.
    """
    windows = config.analysis_windows
    bounds = (
        (float(windows.before_start_s), float(windows.before_stop_s)),
        (float(windows.after_start_s), float(windows.after_stop_s)),
    )
    counts = np.zeros((trial_count, len(prepared_spikes.unit_ids), 2), dtype=np.int64)
    for unit_index, train in enumerate(prepared_spikes.trial_spike_trains):
        if len(train.relative_spike_times) != trial_count:
            raise ValueError("prepared spike trial axis differs from prepared phase")
        for trial_index, values in enumerate(train.relative_spike_times):
            spikes = np.asarray(values, dtype=float)
            before_count = 0
            after_count = 0
            # Count scalars directly so planning does not retain a full-train
            # Boolean comparison mask alongside its int64 result table.
            for spike_time in spikes:
                time_s = float(spike_time)
                if bounds[0][0] <= time_s < bounds[0][1]:
                    before_count += 1
                if bounds[1][0] <= time_s < bounds[1][1]:
                    after_count += 1
            counts[trial_index, unit_index, 0] = before_count
            counts[trial_index, unit_index, 1] = after_count
    return counts


def _preflight_grouped_planning_construction(
    *,
    config: LFPSummaryConfig,
    prepared_phase: object,
    prepared_spikes: object,
    shared_phase_mmap_bytes: int,
) -> None:
    """Reject unsafe grouped planner construction from scalar prepared metadata.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated PPC configuration. Its execution process/aggregate limits are
        interpreted in bytes and its block sizes are categorical dimensions.
    prepared_phase : PreparedPhaseRun-like object
        Validated prepared phase record. Only scalar/indexed condition, filter,
        objective, user, and site-valid values are read; no membership array is
        constructed by this helper.
    prepared_spikes : PreparedSpikeRun-like object
        Validated full unit-axis spike record. Only ``unit_ids`` length is read.
    shared_phase_mmap_bytes : int
        Nonnegative bytes for complete read-only phase/valid prepared storage,
        counted once in the aggregate process peak.

    Returns
    -------
    None
        Raises before creation of the Boolean gated membership, int64 full
        spike-count table, schedules, planner maps, summaries, work directory,
        or checkpoints.

    Raises
    ------
    ValueError
        If checked int64 arithmetic overflows or the conservative planning
        construction peak exceeds either configured process or aggregate limit.

    Notes
    -----
    ``M = trial * condition`` and ``Q = 16 * trial * unit`` charge the two
    executor-created planning tables. ``D`` charges selected-row and schedule
    int64 drafts, and ``A`` conservatively charges local positions, worst-case
    full-trial edges, and one copied ``(trial, unit_block, segment)`` count
    block. The final planner later computes its exact retained ``P`` and gates
    ``M + Q + max(P, D + A)`` before final map materialization.
    """
    prepared_trials = prepared_phase.prepared_trials
    trial_count = _checked_ppc_int(
        int(np.asarray(prepared_phase.trial_indices).size),
        "prepared trial_count",
        minimum=1,
    )
    condition_count = _checked_ppc_int(
        len(prepared_trials.condition_names), "prepared condition_count", minimum=1
    )
    site_count = _checked_ppc_int(len(config.sites), "configured site_count", minimum=1)
    unit_count = _checked_ppc_int(len(prepared_spikes.unit_ids), "prepared unit_count", minimum=1)
    shuffle_count = _checked_ppc_int(
        config.ppc.shuffle_count, "config.ppc.shuffle_count", minimum=1
    )
    membership_bytes = _checked_ppc_multiply(trial_count, condition_count)
    source_count_bytes = _checked_ppc_multiply(16, trial_count, unit_count)
    selected_rows = 0
    scheduled_cells = 0
    condition_values = np.asarray(prepared_trials.condition_membership)
    filter_values = np.asarray(prepared_trials.filter_membership)
    objective_values = np.asarray(prepared_trials.objective_valid)
    excluded_values = np.asarray(prepared_trials.user_excluded)
    site_values = np.asarray(prepared_phase.site_valid)
    # Scalar indexing avoids the executor-created trial-by-condition Boolean
    # table until this full construction gate has accepted the workload.
    for site_index in range(site_count):
        for condition_index in range(condition_count):
            selected_count = 0
            for trial_index in range(trial_count):
                if (
                    bool(condition_values[trial_index, condition_index])
                    and bool(filter_values[trial_index])
                    and bool(objective_values[trial_index])
                    and not bool(excluded_values[trial_index])
                    and bool(site_values[site_index, trial_index])
                ):
                    selected_count += 1
            selected_rows = _checked_ppc_add(
                selected_rows,
                _checked_ppc_multiply(3, selected_count),
            )
            if selected_count >= 2:
                scheduled_cells = _checked_ppc_add(
                    scheduled_cells,
                    _checked_ppc_multiply(3, shuffle_count, selected_count),
                )
    draft_bytes = _checked_ppc_add(
        _checked_ppc_multiply(8, selected_rows),
        _checked_ppc_multiply(8, scheduled_cells),
    )
    maximum_edge_count = _checked_ppc_multiply(trial_count, trial_count - 1)
    largest_unit_block = min(config.ppc_execution.unit_block_size, unit_count)
    scratch_bytes = _checked_ppc_add(
        _checked_ppc_multiply(
            8,
            _checked_ppc_add(
                _checked_ppc_multiply(2, trial_count), maximum_edge_count
            ),
        ),
        _checked_ppc_multiply(16, trial_count, largest_unit_block),
    )
    construction_private = _checked_ppc_add(
        membership_bytes, source_count_bytes, draft_bytes, scratch_bytes
    )
    shared_bytes = _checked_ppc_int(
        shared_phase_mmap_bytes, "shared_phase_mmap_bytes"
    )
    if construction_private > config.ppc_execution.maximum_worker_allocation_bytes:
        raise ValueError("planning construction exceeds maximum_worker_allocation_bytes")
    if _checked_ppc_add(shared_bytes, construction_private) > (
        config.ppc_execution.maximum_aggregate_allocation_bytes
    ):
        raise ValueError("planning construction exceeds maximum_aggregate_allocation_bytes")


def _grouped_execution_settings(execution: PPCExecutionConfig) -> dict[str, object]:
    """Return all execution-only PPC settings as canonical JSON primitives.

    Parameters
    ----------
    execution : PPCExecutionConfig
        Validated bounded-work settings with byte limits in bytes and block
        dimensions in categorical axis entries.

    Returns
    -------
    dict[str, object]
        Exact JSON-safe mapping of all ten execution fields. It carries no raw
        phase, spike, schedule, or scientific component value.
    """
    return {
        "unit_block_size": execution.unit_block_size,
        "shuffle_block_size": execution.shuffle_block_size,
        "trial_edge_block_size": execution.trial_edge_block_size,
        "worker_count": execution.worker_count,
        "maximum_worker_allocation_bytes": execution.maximum_worker_allocation_bytes,
        "maximum_aggregate_allocation_bytes": execution.maximum_aggregate_allocation_bytes,
        "prepared_phase_cache_enabled": execution.prepared_phase_cache_enabled,
        "checkpoint_enabled": execution.checkpoint_enabled,
        "checkpoint_retention": execution.checkpoint_retention,
        "progress_update_interval": execution.progress_update_interval,
    }


def _grouped_execution_plan_fingerprint(plan: PPCComponentPlan) -> str:
    """Hash immutable grouped scheduling/union state and grouped kernel version.

    Parameters
    ----------
    plan : PPCComponentPlan
        Immutable site-major job plans with local int64 schedules and one
        site-qualified physical-edge union.

    Returns
    -------
    str
        SHA-256 identity for planned schedules, stable rows/edges, batches,
        and the currently selected grouped numerical-kernel version.
    """
    job_records: list[dict[str, object]] = []
    for job in plan.job_plans:
        job_records.append(
            {
                "condition_index": job.condition_index,
                "condition_name": job.condition_name,
                "site_index": job.site_index,
                "site_id": job.site_id,
                "epoch_index": job.epoch_index,
                "epoch_name": job.epoch_name,
                "selected_trial_rows": _array_fingerprint(job.selected_trial_rows),
                "schedule": _array_fingerprint(job.schedule),
                "stable_edge_source_trial_row": _array_fingerprint(
                    job.stable_edge_source_trial_row
                ),
                "stable_edge_target_trial_row": _array_fingerprint(
                    job.stable_edge_target_trial_row
                ),
                "edge_union_position": _array_fingerprint(job.edge_union_position),
                "segment_expression": job.segment_expression,
                "base_ppc_seed": job.base_ppc_seed,
                "schedule_seed": job.schedule_seed,
                "condition_derivation_identity": job.condition_derivation_identity,
                "site_derivation_identity": job.site_derivation_identity,
                "epoch_derivation_identity": job.epoch_derivation_identity,
                "schedule_shape": job.schedule_shape,
                "schedule_fingerprint": job.schedule_fingerprint,
            }
        )
    allocation = {
        field_name: getattr(plan.allocation_estimate, field_name)
        for field_name in PPCAllocationEstimate.__dataclass_fields__
    }
    return _hash_json(
        {
            "kernel_version": _GROUPED_PPC_KERNEL_VERSION,
            "jobs": job_records,
            "edge_site_index": _array_fingerprint(plan.edge_site_index),
            "stable_edge_source_trial_row": _array_fingerprint(
                plan.stable_edge_source_trial_row
            ),
            "stable_edge_target_trial_row": _array_fingerprint(
                plan.stable_edge_target_trial_row
            ),
            "condition_batches": plan.condition_batches,
            "scheduled_edge_count": plan.scheduled_edge_count,
            "independent_edge_count": plan.independent_edge_count,
            "union_edge_count": plan.union_edge_count,
            "edge_union_saturation": plan.edge_union_saturation,
            "edge_reuse_ratio": plan.edge_reuse_ratio,
            "allocation_estimate": allocation,
        }
    )


def _grouped_run_metadata(
    *,
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    plan: PPCComponentPlan,
    condition_membership: np.ndarray,
) -> dict[str, object]:
    """Build complete work metadata for serial grouped component execution.

    Parameters
    ----------
    config : LFPSummaryConfig
        Scientific configuration with seconds/Hz windows and component
        settings. Its execution-independent component fingerprint is retained.
    execution : PPCExecutionConfig
        Exact ten-field execution setting record.
    prepared_phase : PreparedPhaseRun-like object
        Full complex64/Boolean phase axes, stable trial/site validity, and
        relative-second coordinates.
    prepared_spikes : PreparedSpikeRun-like object
        Stable unit identities and full-trial relative-second spike vectors.
    plan : PPCComponentPlan
        Immutable scheduler output including derived schedule bytes.
    condition_membership : numpy.ndarray
        Boolean ``(trial, condition)`` mask after shared filter/objective/user
        gates; it is categorical membership, not a physical measurement.

    Returns
    -------
    dict[str, object]
        JSON-safe exact run metadata, including safe block identity mapping and
        SHA-256 ``execution_plan_fingerprint``/``run_fingerprint`` values.
    """
    phase = np.asarray(prepared_phase.phase_tensor)
    valid = np.asarray(prepared_phase.phase_valid)
    trial_rows = np.asarray(prepared_phase.trial_indices)
    site_valid = np.asarray(prepared_phase.site_valid)
    trains = tuple(
        tuple(np.asarray(values, dtype=float) for values in train.relative_spike_times)
        for train in prepared_spikes.trial_spike_trains
    )
    overlaps = tuple(
        tuple(int(value) for value in np.asarray(train.overlap_trial_indices, dtype=np.int64))
        for train in prepared_spikes.trial_spike_trains
    )
    block_identities: dict[str, dict[str, object]] = {}
    for site_index, site in enumerate(config.sites):
        for unit_start in range(0, len(prepared_spikes.unit_ids), execution.unit_block_size):
            unit_stop = min(unit_start + execution.unit_block_size, len(prepared_spikes.unit_ids))
            block_id = f"site-{site_index:03d}-unit-{unit_start:06d}-{unit_stop:06d}"
            block_identities[block_id] = {
                "site_id": site.stable_id,
                "site_index": site_index,
                "unit_start": unit_start,
                "unit_stop": unit_stop,
            }
    execution_plan_fingerprint = _grouped_execution_plan_fingerprint(plan)
    metadata: dict[str, object] = {
        "generator": "execute_grouped_ppc_component",
        "schema_version": config.schema_version,
        "source_fingerprint": _hash_json(fingerprint_source_files(config, "spike_phase")),
        "scientific_fingerprint": component_fingerprint("spike_phase", config),
        "execution_settings": _grouped_execution_settings(execution),
        "grouped_code_version": _GROUPED_PPC_CODE_VERSION,
        "grouped_kernel_version": _GROUPED_PPC_KERNEL_VERSION,
        "execution_plan_fingerprint": execution_plan_fingerprint,
        "phase_content_fingerprint": _array_fingerprint(
            phase,
            valid,
            np.asarray(prepared_phase.relative_time_s),
            trial_rows,
            site_valid,
        ),
        "spike_content_fingerprint": _hash_spikes(trains, overlaps),
        "condition_membership_fingerprint": _array_fingerprint(
            np.asarray(condition_membership, dtype=bool)
        ),
        "site_ids": [site.stable_id for site in config.sites],
        "unit_ids": list(prepared_spikes.unit_ids),
        "summary_axes": ["unit", "condition", "site", "epoch", "frequency"],
        "histogram_axes": ["unit", "condition", "site", "epoch", "band", "phase_bin"],
        "block_identities": block_identities,
    }
    metadata["run_fingerprint"] = _hash_json(metadata)
    return metadata


def _load_grouped_metadata(run_directory: Path) -> dict[str, object] | None:
    """Load one existing grouped metadata mapping without accepting malformed JSON.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact grouped work directory containing an optional metadata JSON file.

    Returns
    -------
    dict[str, object] or None
        Parsed metadata mapping only when the file is JSON object data; ``None``
        for missing, malformed, or nonmapping metadata.
    """
    try:
        value = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _empty_grouped_summary_arrays(
    *,
    unit_count: int,
    condition_count: int,
    site_count: int,
    epoch_count: int,
    frequency_count: int,
    representative_band_count: int,
    phase_bin_count: int,
) -> dict[str, np.ndarray]:
    """Allocate the documented full grouped PPC summary axes.

    Parameters
    ----------
    unit_count, condition_count, site_count, epoch_count, frequency_count : int
        Positive categorical axis sizes for unit, condition, site, epoch, and
        Hz frequency coordinates.
    representative_band_count, phase_bin_count : int
        Positive categorical histogram band/bin sizes; bins represent radians.

    Returns
    -------
    dict[str, numpy.ndarray]
        Writable parent arrays: metric/count/flag fields have
        ``(unit, condition, site, epoch, frequency)`` axes and the histogram
        has ``(unit, condition, site, epoch, band, phase_bin)`` axes.
    """
    metric_shape = (unit_count, condition_count, site_count, epoch_count, frequency_count)
    arrays = {name: np.full(metric_shape, np.nan, dtype=float) for name in _FLOAT_FIELDS}
    arrays.update({name: np.zeros(metric_shape, dtype=np.int64) for name in _INTEGER_FIELDS})
    arrays.update({name: np.zeros(metric_shape, dtype=bool) for name in _BOOLEAN_FIELDS})
    arrays["representative_phase_histogram_count"] = np.zeros(
        metric_shape[:-1] + (representative_band_count, phase_bin_count),
        dtype=np.int64,
    )
    return arrays


def _freeze_grouped_summary_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Freeze owned grouped output arrays without altering axes or units.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Parent-owned complete grouped summary arrays with documented metric or
        representative-histogram axes.

    Returns
    -------
    dict[str, numpy.ndarray]
        New mapping to the same owned arrays after element writes are disabled.
    """
    frozen: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        array.setflags(write=False)
        frozen[name] = array
    return frozen


def _grouped_checkpoint_arrays(
    *,
    summary: Mapping[str, np.ndarray],
    site_index: int,
    unit_start: int,
    unit_stop: int,
) -> dict[str, np.ndarray]:
    """Copy one bounded site/unit summary into no-copy checkpoint arrays.

    Parameters
    ----------
    summary : Mapping[str, numpy.ndarray]
        Full grouped parent arrays with documented component axes.
    site_index : int
        Nonnegative site axis position fixed for this checkpoint.
    unit_start, unit_stop : int
        Half-open unit bounds on the full unit axis.

    Returns
    -------
    dict[str, numpy.ndarray]
        Read-only copied metric arrays on ``(unit_block, condition, epoch,
        frequency)``, histogram on ``(unit_block, condition, epoch, band,
        phase_bin)``, plus int64 ``site_index[1]`` and ``unit_bounds[2]``.
    """
    arrays: dict[str, np.ndarray] = {}
    for name in _SUMMARY_FIELDS:
        arrays[name] = np.asarray(summary[name][unit_start:unit_stop, :, site_index]).copy()
    arrays["representative_phase_histogram_count"] = np.asarray(
        summary["representative_phase_histogram_count"][unit_start:unit_stop, :, site_index]
    ).copy()
    arrays["site_index"] = np.asarray([site_index], dtype=np.int64)
    arrays["unit_bounds"] = np.asarray([unit_start, unit_stop], dtype=np.int64)
    for values in arrays.values():
        values.setflags(write=False)
    return arrays


def _grouped_checkpoint_schema(
    *,
    unit_count: int,
    condition_count: int,
    epoch_count: int,
    frequency_count: int,
    representative_band_count: int,
    phase_bin_count: int,
) -> dict[str, tuple[np.dtype[object], tuple[int, ...]]]:
    """Return the exact bounded NPZ schema for one grouped site/unit block.

    Parameters
    ----------
    unit_count, condition_count, epoch_count, frequency_count : int
        Nonnegative/positive block axes in unit, condition, epoch, and Hz
        frequency order. ``unit_count`` is the half-open block length.
    representative_band_count, phase_bin_count : int
        Positive histogram axes. Bins represent radians, while the integers
        here are dimension counts.

    Returns
    -------
    dict[str, tuple[numpy.dtype, tuple[int, ...]]]
        Exact name-to-``(dtype, shape)`` mapping: all metric fields have
        ``(unit_block, condition, epoch, frequency)`` axes; the histogram has
        ``(unit_block, condition, epoch, band, phase_bin)``; identities are
        compact int64 ``site_index[1]`` and ``unit_bounds[2]`` arrays.

    Raises
    ------
    ValueError
        If a documented axis length is invalid. This helper allocates only a
        small mapping/tuple description, never a checkpoint member array.
    """
    block_units = _checked_ppc_int(unit_count, "checkpoint unit_count", minimum=1)
    conditions = _checked_ppc_int(
        condition_count, "checkpoint condition_count", minimum=1
    )
    epochs = _checked_ppc_int(epoch_count, "checkpoint epoch_count", minimum=1)
    frequencies = _checked_ppc_int(
        frequency_count, "checkpoint frequency_count", minimum=1
    )
    bands = _checked_ppc_int(
        representative_band_count, "checkpoint representative_band_count", minimum=1
    )
    bins = _checked_ppc_int(phase_bin_count, "checkpoint phase_bin_count", minimum=1)
    metric_shape = (block_units, conditions, epochs, frequencies)
    histogram_shape = metric_shape[:-1] + (bands, bins)
    return {
        **{name: (np.dtype(float), metric_shape) for name in _FLOAT_FIELDS},
        **{name: (np.dtype(np.int64), metric_shape) for name in _INTEGER_FIELDS},
        **{name: (np.dtype(bool), metric_shape) for name in _BOOLEAN_FIELDS},
        "representative_phase_histogram_count": (
            np.dtype(np.int64),
            histogram_shape,
        ),
        "site_index": (np.dtype(np.int64), (1,)),
        "unit_bounds": (np.dtype(np.int64), (2,)),
    }


def _grouped_checkpoint_schema_bytes(
    schema: Mapping[str, tuple[np.dtype[object], tuple[int, ...]]],
) -> int:
    """Return checked uncompressed numeric bytes for one checkpoint schema.

    Parameters
    ----------
    schema : Mapping[str, tuple[numpy.dtype, tuple[int, ...]]]
        Exact NPZ member declaration for one site/unit block. Each shape axis
        is a nonnegative Python integer dimension, and all member dtypes are
        numeric or Boolean.

    Returns
    -------
    int
        Exact nonnegative total element bytes across all declared members.
        The value is independent of ZIP compression and is suitable for the
        bounded single-block loader.

    Raises
    ------
    ValueError
        If a schema member dtype is object, an axis is invalid, or checked
        signed-int64 byte arithmetic overflows.
    """
    total_bytes = 0
    for dtype, shape in schema.values():
        member_dtype = np.dtype(dtype)
        if member_dtype.hasobject:
            raise ValueError("grouped checkpoint schema cannot contain object arrays")
        element_count = 1
        for axis, dimension in enumerate(shape):
            element_count = _checked_ppc_multiply(
                element_count,
                _checked_ppc_int(
                    dimension,
                    f"checkpoint schema axis {axis}",
                ),
            )
        total_bytes = _checked_ppc_add(
            total_bytes,
            _checked_ppc_multiply(element_count, int(member_dtype.itemsize)),
        )
    return total_bytes


def _clear_grouped_parallel_staging_directory(staging_directory: Path) -> None:
    """Remove disposable worker stages owned by one execution run.

    ``staging_directory`` contains only uncheckpointed worker handoffs.  It is
    cleared by the executor while holding its run lock, so no file here has
    resume semantics.

    Parameters
    ----------
    staging_directory : pathlib.Path
        Run-owned directory containing only scalar worker handoffs and partial
        files; it must never be a final cache or checkpoint directory.

    Returns
    -------
    None
        Leaves an existing empty directory. I/O failures propagate so the
        caller can still release its executor lock in an outer ``finally``.
    """
    staging_directory.mkdir(parents=True, exist_ok=True)
    for path in staging_directory.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)


def _load_grouped_staged_result(
    *,
    staged_result_path: Path,
    expected_array_schema: Mapping[str, tuple[np.dtype[object], tuple[int, ...]]],
    maximum_array_bytes: int,
) -> dict[str, np.ndarray]:
    """Load one exact bounded worker stage after validating ZIP NPY headers.

    The header pass rejects wrong keys, dtype, shape, or cap before
    ``numpy.load`` can materialize any member.  Every invalid/disposable stage
    is unlinked before its exception is raised.

    Parameters
    ----------
    staged_result_path : pathlib.Path
        Canonical run-owned NPZ path for one worker block.
    expected_array_schema : Mapping[str, tuple[numpy.dtype, tuple[int, ...]]]
        Exact flat member names, numeric dtypes, and checkpoint axes.
    maximum_array_bytes : int
        Exact uncompressed schema byte total, not a compression-size limit.

    Returns
    -------
    dict[str, numpy.ndarray]
        Read-only owned arrays matching the schema exactly.

    Raises
    ------
    ValueError
        If ZIP/NPY headers, members, dtype, shape, or byte cap are invalid;
        the disposable stage is removed first.
    """
    try:
        if maximum_array_bytes != _grouped_checkpoint_schema_bytes(expected_array_schema):
            raise ValueError("staged grouped worker result exceeds exact schema cap")
        with zipfile.ZipFile(staged_result_path) as archive:
            member_names = archive.namelist()
            if (
                any("/" in name or not name.endswith(".npy") for name in member_names)
                or len(member_names) != len(set(member_names))
            ):
                raise ValueError("staged grouped worker result schema is invalid")
            names = {name[:-4] for name in member_names}
            if names != set(expected_array_schema):
                raise ValueError("staged grouped worker result schema is invalid")
            declared_bytes = 0
            for name, (expected_dtype, expected_shape) in expected_array_schema.items():
                with archive.open(f"{name}.npy") as member:
                    version = np.lib.format.read_magic(member)
                    if version == (1, 0):
                        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
                    elif version in {(2, 0), (3, 0)}:
                        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
                    else:
                        raise ValueError("staged grouped worker result has unsupported NPY header")
                if (
                    fortran_order
                    or np.dtype(dtype) != np.dtype(expected_dtype)
                    or tuple(shape) != tuple(expected_shape)
                ):
                    raise ValueError("staged grouped worker result schema is invalid")
                declared_bytes = _checked_ppc_add(
                    declared_bytes,
                    _checked_ppc_multiply(int(np.prod(shape, dtype=np.int64)), int(np.dtype(dtype).itemsize)),
                )
            if declared_bytes != maximum_array_bytes:
                raise ValueError("staged grouped worker result exceeds exact schema cap")
        with np.load(staged_result_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in expected_array_schema}
        for values in arrays.values():
            values.setflags(write=False)
        return arrays
    except (
        OSError, ValueError, TypeError, EOFError, RuntimeError,
        NotImplementedError, zipfile.BadZipFile, KeyError,
    ) as error:
        staged_result_path.unlink(missing_ok=True)
        if isinstance(error, ValueError) and str(error).startswith("staged grouped worker result"):
            raise
        raise ValueError("staged grouped worker result is invalid") from error


def _merge_grouped_checkpoint(
    *,
    summary: dict[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    site_index: int,
    unit_start: int,
    unit_stop: int,
    condition_count: int,
    epoch_count: int,
    frequency_count: int,
    representative_band_count: int,
    phase_bin_count: int,
) -> bool:
    """Validate and merge one streaming grouped checkpoint into parent arrays.

    Parameters
    ----------
    summary : dict[str, numpy.ndarray]
        Writable full component arrays with documented axes.
    arrays : Mapping[str, numpy.ndarray]
        One frozen single-loader block mapping with scalar/compact identities.
    site_index : int
        Expected categorical site position.
    unit_start, unit_stop : int
        Expected half-open full-unit bounds.
    condition_count, epoch_count, frequency_count : int
        Expected block metric axis lengths; frequencies are Hz coordinates.
    representative_band_count, phase_bin_count : int
        Expected histogram band/radian-bin axis lengths.

    Returns
    -------
    bool
        ``True`` after exact dtype/shape/identity validation and parent copy;
        ``False`` when this one block is corrupt or incompatible. It never
        reads siblings or changes arrays on a rejected block.
    """
    expected_keys = set(_SUMMARY_FIELDS) | {
        "representative_phase_histogram_count", "site_index", "unit_bounds"
    }
    block_units = unit_stop - unit_start
    if set(arrays) != expected_keys:
        return False
    site_values = np.asarray(arrays["site_index"])
    bounds = np.asarray(arrays["unit_bounds"])
    if (
        site_values.dtype != np.dtype(np.int64)
        or site_values.shape != (1,)
        or int(site_values[0]) != site_index
        or bounds.dtype != np.dtype(np.int64)
        or bounds.shape != (2,)
        or int(bounds[0]) != unit_start
        or int(bounds[1]) != unit_stop
    ):
        return False
    metric_shape = (block_units, condition_count, epoch_count, frequency_count)
    histogram_shape = (
        block_units,
        condition_count,
        epoch_count,
        representative_band_count,
        phase_bin_count,
    )
    for name in _FLOAT_FIELDS:
        if np.asarray(arrays[name]).dtype != np.dtype(float) or np.asarray(arrays[name]).shape != metric_shape:
            return False
    for name in _INTEGER_FIELDS:
        if np.asarray(arrays[name]).dtype != np.dtype(np.int64) or np.asarray(arrays[name]).shape != metric_shape:
            return False
    for name in _BOOLEAN_FIELDS:
        if np.asarray(arrays[name]).dtype != np.dtype(bool) or np.asarray(arrays[name]).shape != metric_shape:
            return False
    histogram = np.asarray(arrays["representative_phase_histogram_count"])
    if histogram.dtype != np.dtype(np.int64) or histogram.shape != histogram_shape:
        return False
    for name in _SUMMARY_FIELDS:
        summary[name][unit_start:unit_stop, :, site_index] = arrays[name]
    summary["representative_phase_histogram_count"][unit_start:unit_stop, :, site_index] = histogram
    return True


def _grouped_job_key(job: PPCJobPlan) -> tuple[int, int, int]:
    """Return the categorical dictionary key for one immutable grouped job.

    Parameters
    ----------
    job : PPCJobPlan
        Validated immutable plan record. Its condition, site, and epoch
        indices address the component summary axes and have no physical units.

    Returns
    -------
    tuple[int, int, int]
        ``(condition_index, site_index, epoch_index)`` in the component's
        documented categorical-axis order. The tuple is hashable and contains
        no source phase, spike, schedule, or physical measurement values.
    """
    return job.condition_index, job.site_index, job.epoch_index


def _write_grouped_observed_job_in_place(
    *,
    observed_trial_statistics: object,
    job_plan: PPCJobPlan,
    output_arrays: Mapping[str, np.ndarray],
) -> None:
    """Write one job's observed PPC metrics into already-owned summary views.

    Parameters
    ----------
    observed_trial_statistics : ObservedTrialSegmentedPPCStatistics
        Frozen selected-trial sufficient statistics with int64 stable IDs,
        complex128 sums, int64 counts, and int64 representative histograms on
        ``(trial, unit, segment=2, frequency)`` / ``(trial, unit, segment=2,
        band=2, phase_bin)`` axes. Sums are dimensionless; histogram phase
        coordinates are radians in the record's metadata.
    job_plan : PPCJobPlan
        One immutable condition/site/epoch plan. Its ordered int64
        ``selected_trial_rows`` are stable identities in the statistic record;
        ``epoch_name`` selects before, after, or the additive whole window.
    output_arrays : Mapping[str, numpy.ndarray]
        Writable summary-owned views. Metric fields have ``(unit, frequency)``
        axes; ``representative_phase_histogram_count`` has
        ``(unit, band=2, phase_bin)`` axes. PPC/resultant are dimensionless,
        preferred phase is radians, count fields are spikes/trials, and flags
        are Boolean.

    Returns
    -------
    None
        Mutates only the supplied summary views. It creates no pooled observed
        statistic or metric record.

    Raises
    ------
    ValueError
        If statistic, job selection, epoch, or output dtypes/axes do not meet
        the grouped observed-summary contract.

    Notes
    -----
    The output preferred-phase, resultant, and spike-count views temporarily
    hold real sums, imaginary sums, and counts. Segment/trial scalar loops
    preserve the legacy aggregate-then-compose arithmetic without retaining
    an extra full condition record.
    """
    import math

    required = {
        "ppc",
        "resultant_length",
        "preferred_phase_rad",
        "spike_count",
        "computable",
        "reliable",
        "eligible_trial_count",
        "null_eligible",
        "representative_phase_histogram_count",
    }
    if set(output_arrays) != required:
        raise ValueError("grouped observed output fields are incomplete")
    stable_rows = np.asarray(observed_trial_statistics.phase_trial_index)
    sums = np.asarray(observed_trial_statistics.phase_vector_sum)
    counts = np.asarray(observed_trial_statistics.valid_spike_count)
    histograms = np.asarray(observed_trial_statistics.representative_phase_histogram_count)
    selected_rows = np.asarray(job_plan.selected_trial_rows)
    if (
        stable_rows.dtype != np.dtype(np.int64)
        or stable_rows.ndim != 1
        or sums.dtype != np.dtype(np.complex128)
        or counts.dtype != np.dtype(np.int64)
        or sums.ndim != 4
        or sums.shape != counts.shape
        or sums.shape[0] != stable_rows.size
        or sums.shape[2] != 2
        or histograms.dtype != np.dtype(np.int64)
        or histograms.ndim != 5
        or histograms.shape[:3] != (sums.shape[0], sums.shape[1], 2)
        or histograms.shape[3] != 2
        or selected_rows.dtype != np.dtype(np.int64)
        or selected_rows.ndim != 1
    ):
        raise ValueError("grouped observed statistic axes disagree")
    metric_shape = (sums.shape[1], sums.shape[3])
    phase_bin_count = histograms.shape[4]
    ppc = np.asarray(output_arrays["ppc"])
    resultant = np.asarray(output_arrays["resultant_length"])
    preferred_phase = np.asarray(output_arrays["preferred_phase_rad"])
    spike_count = np.asarray(output_arrays["spike_count"])
    computable = np.asarray(output_arrays["computable"])
    reliable = np.asarray(output_arrays["reliable"])
    contributing = np.asarray(output_arrays["eligible_trial_count"])
    null_eligible = np.asarray(output_arrays["null_eligible"])
    histogram = np.asarray(output_arrays["representative_phase_histogram_count"])
    if (
        ppc.dtype != np.dtype(float)
        or resultant.dtype != np.dtype(float)
        or preferred_phase.dtype != np.dtype(float)
        or spike_count.dtype != np.dtype(np.int64)
        or computable.dtype != np.dtype(bool)
        or reliable.dtype != np.dtype(bool)
        or contributing.dtype != np.dtype(np.int64)
        or null_eligible.dtype != np.dtype(bool)
        or any(
            values.shape != metric_shape
            for values in (
                ppc,
                resultant,
                preferred_phase,
                spike_count,
                computable,
                reliable,
                contributing,
                null_eligible,
            )
        )
        or histogram.dtype != np.dtype(np.int64)
        or histogram.shape != (metric_shape[0], 2, phase_bin_count)
    ):
        raise ValueError("grouped observed summary axes disagree")
    segments_by_epoch = {
        "before": (0,),
        "after": (1,),
        "whole": (0, 1),
    }
    try:
        segments = segments_by_epoch[job_plan.epoch_name]
    except KeyError as error:
        raise ValueError("grouped observed job has an unknown epoch") from error
    position_by_stable_row = {
        int(stable_row): position for position, stable_row in enumerate(stable_rows)
    }
    selected_positions: list[int] = []
    for stable_row in selected_rows:
        try:
            selected_positions.append(position_by_stable_row[int(stable_row)])
        except KeyError as error:
            raise ValueError("grouped observed job selects an unknown stable row") from error

    ppc.fill(math.nan)
    resultant.fill(0.0)
    preferred_phase.fill(0.0)
    spike_count.fill(0)
    computable.fill(False)
    reliable.fill(False)
    contributing.fill(0)
    null_eligible.fill(False)
    histogram.fill(0)
    # Segment-major order first completes all before sums, then adds after
    # sums for whole jobs, matching the legacy pooled whole-window expression.
    for segment in segments:
        for selected_position in selected_positions:
            trial_sums = sums[selected_position, :, segment]
            trial_counts = counts[selected_position, :, segment]
            np.add(preferred_phase, trial_sums.real, out=preferred_phase)
            np.add(resultant, trial_sums.imag, out=resultant)
            np.add(spike_count, trial_counts, out=spike_count)
            _record_grouped_representative_histogram(
                histogram=histogram,
                trial_histogram=histograms[selected_position, :, segment],
            )
    for selected_position in selected_positions:
        trial_counts = counts[selected_position]
        for unit_index in range(metric_shape[0]):
            for frequency_index in range(metric_shape[1]):
                if job_plan.epoch_name == "whole":
                    contributes = bool(
                        int(trial_counts[unit_index, 0, frequency_index]) > 0
                        or int(trial_counts[unit_index, 1, frequency_index]) > 0
                    )
                else:
                    contributes = bool(
                        int(
                            trial_counts[
                                unit_index,
                                segments[0],
                                frequency_index,
                            ]
                        )
                        > 0
                    )
                if contributes:
                    contributing[unit_index, frequency_index] += 1
    direction_epsilon = np.finfo(float).eps
    for unit_index in range(metric_shape[0]):
        for frequency_index in range(metric_shape[1]):
            count = int(spike_count[unit_index, frequency_index])
            vector = np.complex128(
                complex(
                    float(preferred_phase[unit_index, frequency_index]),
                    float(resultant[unit_index, frequency_index]),
                )
            )
            magnitude = float(np.abs(vector))
            if count >= 2:
                ppc[unit_index, frequency_index] = (
                    magnitude**2 - count
                ) / float(count * (count - 1))
                computable[unit_index, frequency_index] = True
            if count > 0:
                resultant[unit_index, frequency_index] = magnitude / float(count)
                if magnitude > direction_epsilon * max(count, 1):
                    preferred_phase[unit_index, frequency_index] = float(np.angle(vector))
                else:
                    preferred_phase[unit_index, frequency_index] = math.nan
            else:
                resultant[unit_index, frequency_index] = math.nan
                preferred_phase[unit_index, frequency_index] = math.nan
            reliable[unit_index, frequency_index] = count >= 50
            null_eligible[unit_index, frequency_index] = bool(
                reliable[unit_index, frequency_index]
                and contributing[unit_index, frequency_index] >= 2
                and math.isfinite(float(ppc[unit_index, frequency_index]))
            )


def _record_grouped_representative_histogram(
    *,
    histogram: np.ndarray,
    trial_histogram: np.ndarray,
) -> None:
    """Accumulate one selected-trial representative histogram in place.

    Parameters
    ----------
    histogram, trial_histogram : numpy.ndarray
        Int64 arrays with matching ``(unit, band=2, phase_bin)`` axes. Counts
        are observations in phase bins; phase coordinates themselves remain
        radians in the configuration metadata.

    Returns
    -------
    None
        Adds ``trial_histogram`` to the existing ``histogram`` buffer without
        allocating or changing any grouped observed-summary behavior.

    Notes
    -----
    This narrow private seam exists so the work-only S6 profiler can time the
    representative-histogram count separately from the inclusive observed
    reduction. It deliberately preserves the prior in-place ``numpy.add``.
    """
    np.add(histogram, trial_histogram, out=histogram)


def _component_site_union_views(
    plan: PPCComponentPlan,
    site_index: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one site's borrowed contiguous physical-edge union views.

    Parameters
    ----------
    plan : PPCComponentPlan
        Immutable component plan whose site-qualified union arrays have
        matching int64 ``(union_edge,)`` axes.
    site_index : int
        Valid nonnegative categorical site position.

    Returns
    -------
    source_trial_row, target_trial_row, position_offset : tuple
        Read-only int64 views of this site's contiguous component-union slice
        and its first global union position. Empty sites return non-owning
        empty views at their sorted insertion position. Stable row IDs are
        categorical, not physical times.

    Raises
    ------
    ValueError
        If the immutable component union is not site-contiguous, which would
        make the task's borrowed local edge map ambiguous.
    """
    site_positions = np.asarray(plan.edge_site_index, dtype=np.int64)
    start = 0
    while start < site_positions.size and int(site_positions[start]) < site_index:
        start += 1
    stop = start
    while stop < site_positions.size and int(site_positions[stop]) == site_index:
        stop += 1
    if stop < site_positions.size and int(site_positions[stop]) < site_index:
        raise ValueError("grouped component site union must be site-sorted")
    return (
        plan.stable_edge_source_trial_row[start:stop],
        plan.stable_edge_target_trial_row[start:stop],
        start,
    )


def _compute_grouped_site_unit_block(
    *,
    config: LFPSummaryConfig,
    prepared_phase: object,
    prepared_spikes: object,
    plan: PPCComponentPlan | None,
    summary: dict[str, np.ndarray],
    site_index: int,
    unit_start: int,
    unit_stop: int,
    site_jobs: tuple[PPCJobPlan, ...] | None = None,
    site_condition_batches: tuple[tuple[int, ...], ...] | None = None,
    site_union_source_trial_row: np.ndarray | None = None,
    site_union_target_trial_row: np.ndarray | None = None,
    site_union_position_offset: int | None = None,
    unit_spike_trains: tuple[object, ...] | None = None,
    summary_unit_start: int | None = None,
    summary_site_index: int | None = None,
) -> None:
    """Compute one bounded site/unit grouped block from segmented statistics.

    Parameters
    ----------
    config : LFPSummaryConfig
        Supplies seconds windows, Hz frequencies, PPC schedules, reliability,
        and bounded edge/shuffle sizes.
    prepared_phase, prepared_spikes : production prepared records
        Full site/frequency/trial/time complex64/Boolean phase and unit/trial
        relative-second spike inputs. Only one site/unit block is sampled.
    plan : PPCComponentPlan or None
        Full immutable scheduler output for serial work. Parallel workers pass
        ``None`` and supply borrowed current-site plan fields instead, never a
        full component plan.
    summary : dict[str, numpy.ndarray]
        Writable full output arrays. Only ``site_index`` and ``[unit_start,
        unit_stop)`` are changed.
    site_index : int
        Current categorical site position.
    unit_start, unit_stop : int
        Half-open full-unit bounds for this bounded computation.
    site_jobs, site_condition_batches : tuple or None
        Borrowed current-site jobs and ordered condition batches. Both are
        required when ``plan`` is ``None``; batches bound only null work.
    site_union_source_trial_row, site_union_target_trial_row : numpy.ndarray or None
        Borrowed read-only int64 ``(site_union_edge,)`` current-site union
        views. They share immutable component-plan storage in worker tasks.
    site_union_position_offset : int or None
        First global component-union position represented by the local union
        views. Job edge positions are rebased by this categorical offset.
    unit_spike_trains : tuple or None
        Exact bounded original spike records for a worker. ``None`` retains
        serial use of the full prepared record slice.
    summary_unit_start, summary_site_index : int or None
        Full coordinates represented by summary axis zero. ``None`` preserves
        full-component serial indexing; workers use local output axes.
    Returns
    -------
    None
        Observed metrics/histograms and scheduled null summaries are copied to
        the documented output axes. Per-trial observed and per-edge arrays are
        released before the next execution stage/checkpoint publication.
    """
    if site_jobs is None:
        if plan is None:
            raise ValueError("grouped worker requires current-site jobs")
        site_jobs = tuple(job for job in plan.job_plans if job.site_index == site_index)
    if site_condition_batches is None:
        if plan is None:
            raise ValueError("grouped worker requires current-site condition batches")
        site_condition_batches = plan.condition_batches[site_index]
    if (
        site_union_source_trial_row is None
        or site_union_target_trial_row is None
        or site_union_position_offset is None
    ):
        if plan is None:
            raise ValueError("grouped worker requires a current-site edge union")
        (
            site_union_source_trial_row,
            site_union_target_trial_row,
            site_union_position_offset,
        ) = _component_site_union_views(plan, site_index)
    if (
        site_union_source_trial_row.dtype != np.dtype(np.int64)
        or site_union_target_trial_row.dtype != np.dtype(np.int64)
        or site_union_source_trial_row.ndim != 1
        or site_union_target_trial_row.ndim != 1
        or site_union_source_trial_row.shape != site_union_target_trial_row.shape
    ):
        raise ValueError("grouped current-site union arrays are invalid")
    output_unit_start = (
        unit_start if summary_unit_start is None else unit_start - summary_unit_start
    )
    output_unit_stop = output_unit_start + (unit_stop - unit_start)
    output_site_index = site_index if summary_site_index is None else summary_site_index
    if (
        output_unit_start < 0
        or output_unit_stop > summary["ppc"].shape[0]
        or output_site_index < 0
        or output_site_index >= summary["ppc"].shape[2]
    ):
        raise ValueError("grouped worker summary/local block coordinates disagree")
    frequencies = np.asarray(config.phase.frequency_hz, dtype=float)
    stable_rows = np.asarray(prepared_phase.trial_indices, dtype=np.int64)
    selected_ids = {
        int(stable_id)
        for job in site_jobs
        for stable_id in np.asarray(job.selected_trial_rows)
    }
    if not selected_ids:
        return
    source_positions = [
        position
        for position, stable_id in enumerate(stable_rows)
        if int(stable_id) in selected_ids
    ]
    # Moving the trial axis is a view. S1/S2 index a row at a time and never
    # gather selected phase/validity arrays into a second full-trial buffer.
    phase_by_trial = np.moveaxis(
        np.asarray(prepared_phase.phase_tensor)[site_index], 1, 0
    )
    valid_by_trial = np.moveaxis(
        np.asarray(prepared_phase.phase_valid)[site_index], 1, 0
    )
    selected_trains = (
        tuple(prepared_spikes.trial_spike_trains[unit_start:unit_stop])
        if unit_spike_trains is None
        else unit_spike_trains
    )
    if len(selected_trains) != unit_stop - unit_start:
        raise ValueError("grouped worker unit spike bounds disagree")
    unit_ids = tuple(
        str(
            getattr(
                train,
                "unit_id",
                prepared_spikes.unit_ids[unit_start + offset]
                if unit_spike_trains is None
                else unit_start + offset,
            )
        )
        for offset, train in enumerate(selected_trains)
    )
    windows = config.analysis_windows
    segment_bounds = (
        (float(windows.before_start_s), float(windows.before_stop_s)),
        (float(windows.after_start_s), float(windows.after_stop_s)),
    )
    geometries = tuple(
        build_source_trial_spike_geometry(
            phase_time_s=np.asarray(prepared_phase.relative_time_s, dtype=float),
            phase_sampling_rate_hz=float(config.phase.output_rate_hz),
            source_trial_index=int(stable_rows[position]),
            unit_ids=unit_ids,
            unit_trial_spike_times_s=tuple(
                np.asarray(train.relative_spike_times[int(position)], dtype=float)
                for train in selected_trains
            ),
            segment_bounds_s=segment_bounds,
        )
        for position in source_positions
    )
    observed_trial_statistics = compute_selected_observed_trial_segmented_ppc_statistics(
        source_trial_geometries=geometries,
        trial_phase_vectors=phase_by_trial,
        phase_valid_mask=valid_by_trial,
        phase_trial_index=stable_rows,
        frequencies_hz=frequencies,
        phase_bin_edges_rad=np.asarray(config.ppc.phase_bin_edges_rad, dtype=float),
    )
    eligible_by_job: dict[tuple[int, int, int], np.ndarray] = {}
    for job in site_jobs:
        output_index = (
            slice(output_unit_start, output_unit_stop),
            job.condition_index,
            output_site_index,
            job.epoch_index,
            slice(None),
        )
        _write_grouped_observed_job_in_place(
            observed_trial_statistics=observed_trial_statistics,
            job_plan=job,
            output_arrays={
                name: summary[name][output_index]
                for name in (
                    "ppc",
                    "resultant_length",
                    "preferred_phase_rad",
                    "spike_count",
                    "computable",
                    "reliable",
                    "eligible_trial_count",
                    "null_eligible",
                )
            }
            | {
                "representative_phase_histogram_count": summary[
                    "representative_phase_histogram_count"
                ][
                    slice(output_unit_start, output_unit_stop),
                    job.condition_index,
                    output_site_index,
                    job.epoch_index,
                ]
            },
        )
        output_eligibility = summary["null_eligible"][output_index]
        eligible_by_job[_grouped_job_key(job)] = output_eligibility
    # No observed-trial statistic arrays remain live while S1 physical edges
    # are reduced; aggregation above owns its compact condition summaries.
    del observed_trial_statistics

    for condition_indices in site_condition_batches:
        _execute_grouped_condition_batch(
            config=config,
            summary=summary,
            site_jobs=site_jobs,
            condition_indices=condition_indices,
            source_trial_geometries=geometries,
            trial_phase_vectors=phase_by_trial,
            phase_valid_mask=valid_by_trial,
            phase_trial_index=stable_rows,
            site_index=site_index,
            unit_start=unit_start,
            unit_stop=unit_stop,
            site_union_source_trial_row=site_union_source_trial_row,
            site_union_target_trial_row=site_union_target_trial_row,
            site_union_position_offset=site_union_position_offset,
            summary_unit_start=summary_unit_start,
            summary_site_index=summary_site_index,
            eligible_by_job=eligible_by_job,
        )


def _execute_grouped_condition_batch(
    *,
    config: LFPSummaryConfig,
    summary: dict[str, np.ndarray],
    site_jobs: tuple[PPCJobPlan, ...],
    condition_indices: tuple[int, ...],
    source_trial_geometries: tuple[object, ...],
    trial_phase_vectors: np.ndarray,
    phase_valid_mask: np.ndarray,
    phase_trial_index: np.ndarray,
    site_index: int,
    unit_start: int,
    unit_stop: int,
    site_union_source_trial_row: np.ndarray,
    site_union_target_trial_row: np.ndarray,
    site_union_position_offset: int,
    summary_unit_start: int | None,
    summary_site_index: int | None,
    eligible_by_job: Mapping[tuple[int, int, int], np.ndarray],
) -> None:
    """Execute one site-local ordered condition batch and release its null work.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated seconds/Hz PPC configuration. Edge and shuffle blocks bound
        only the transient work for this condition batch.
    summary : dict[str, numpy.ndarray]
        Writable serial full-component or worker-local summary arrays. Only the
        supplied site/unit rows and batch conditions are changed.
    site_jobs : tuple[PPCJobPlan, ...]
        All result jobs for this site in stable condition/epoch order.
    condition_indices : tuple[int, ...]
        Nonempty ordered input condition positions for this active batch.
    source_trial_geometries : tuple[SourceTrialSpikeGeometry, ...]
        Selected source geometries for this site/unit block. Stable identities
        select rows from the complete phase views.
    trial_phase_vectors, phase_valid_mask : numpy.ndarray
        Complete site-local phase/validity views with axes ``(full_trial,
        frequency, time)``. Both share prepared input storage.
    phase_trial_index : numpy.ndarray
        Int64 ``(full_trial,)`` stable identities matching the phase first axis.
    site_index : int
        Fixed categorical site position.
    unit_start, unit_stop : int
        Half-open full unit-axis bounds for the block.
    site_union_source_trial_row, site_union_target_trial_row : numpy.ndarray
        Borrowed read-only int64 ``(site_union_edge,)`` stable source/target
        identities for this one site. They are a current-site view, not a full
        component union.
    site_union_position_offset : int
        First global component-union position represented by the local union
        views. Job positions are rebased before edge reduction.
    summary_unit_start, summary_site_index : int or None
        Full coordinates represented by summary axis zero. ``None`` selects
        serial full-component output coordinates; worker summaries use local
        unit axis zero and site axis zero.
    eligible_by_job : Mapping[tuple[int, int, int], numpy.ndarray]
        Summary-owned Boolean ``(unit_block, frequency)`` null eligibility
        views keyed by condition/site/epoch identity.

    Returns
    -------
    None
        Consumes every needed physical edge once per bounded union slice and
        finalizes one batch's accumulators. Accumulator/count/draw arrays are
        unreachable before the next batch begins.
    """
    active_jobs = tuple(
        job
        for job in site_jobs
        if (
            job.condition_index in condition_indices
            and job.schedule.size
            and bool(np.any(eligible_by_job[_grouped_job_key(job)]))
        )
    )
    if not active_jobs:
        return
    frequency_count = trial_phase_vectors.shape[1]
    accumulators: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] = {}
    for job in active_jobs:
        shape = (job.schedule.shape[0], unit_stop - unit_start, frequency_count)
        accumulators[_grouped_job_key(job)] = (
            np.zeros(shape, dtype=np.complex128),
            np.zeros(shape, dtype=np.int64),
        )
    batch_positions = tuple(
        int(position) - site_union_position_offset
        for job in site_jobs
        if job.condition_index in condition_indices
        for position in job.edge_union_position.ravel()
    )
    if any(
        position < 0 or position >= site_union_source_trial_row.size
        for position in batch_positions
    ):
        raise ValueError("grouped job edge positions disagree with current-site union")
    planned_edge_groups = _fixed_edge_position_groups(
        positions=batch_positions,
        maximum_group_size=config.ppc_execution.trial_edge_block_size,
    )
    active_positions = {
        int(position) - site_union_position_offset
        for job in active_jobs
        for position in job.edge_union_position.ravel()
    }
    for planned_group in planned_edge_groups:
        # Allocation is based on this whole fixed group.  Filtering happens
        # only afterwards, so discarded inactive edges cannot shift two costly
        # active edges into a newly unplanned kernel reduction.
        active_group = tuple(
            position for position in planned_group if position in active_positions
        )
        active_cursor = 0
        while active_cursor < len(active_group):
            edge_start = active_group[active_cursor]
            edge_stop = edge_start + 1
            active_cursor += 1
            while (
                active_cursor < len(active_group)
                and active_group[active_cursor] == edge_stop
            ):
                edge_stop += 1
                active_cursor += 1
            edge_statistics = compute_segmented_edge_statistics(
                source_trial_geometries=source_trial_geometries,
                trial_phase_vectors=trial_phase_vectors,
                phase_valid_mask=phase_valid_mask,
                phase_trial_index=phase_trial_index,
                frequencies_hz=np.asarray(config.phase.frequency_hz, dtype=float),
                source_trial_index=site_union_source_trial_row[edge_start:edge_stop],
                target_trial_index=site_union_target_trial_row[edge_start:edge_stop],
            )
            edge_identities = {
                (int(source), int(target))
                for source, target in zip(
                    site_union_source_trial_row[edge_start:edge_stop],
                    site_union_target_trial_row[edge_start:edge_stop],
                    strict=True,
                )
            }
            for job in active_jobs:
                # A condition job should consume only the physical edges it
                # actually schedules. Apart from avoiding needless scalar
                # work, this keeps whole-window consumption tied to retained
                # before/after sufficient statistics.
                if not any(
                    edge_start
                    <= int(position) - site_union_position_offset
                    < edge_stop
                    for position in job.edge_union_position.ravel()
                ):
                    continue
                vector_sum, valid_count = accumulators[_grouped_job_key(job)]
                eligibility = eligible_by_job[_grouped_job_key(job)]
                for shuffle_start in range(
                    0, job.schedule.shape[0], config.ppc_execution.shuffle_block_size
                ):
                    shuffle_stop = min(
                        shuffle_start + config.ppc_execution.shuffle_block_size,
                        job.schedule.shape[0],
                    )
                    schedule_block = job.schedule[shuffle_start:shuffle_stop]
                    # Do not invoke the immediate consumer for a schedule
                    # chunk that cannot use this physical edge block. This is
                    # scalar membership testing, not a retained edge map.
                    if not any(
                        (
                            int(job.selected_trial_rows[source_position]),
                            int(job.selected_trial_rows[int(target_position)]),
                        )
                        in edge_identities
                        for schedule_row in schedule_block
                        for source_position, target_position in enumerate(schedule_row)
                    ):
                        continue
                    _consume_grouped_edge_block(
                        edge_statistics=edge_statistics,
                        job_plan=job,
                        schedule_block=schedule_block,
                        shuffle_start=shuffle_start,
                        vector_sum=vector_sum,
                        valid_count=valid_count,
                        eligible_cell_mask=eligibility,
                    )
            del edge_statistics
    for job in active_jobs:
        vector_sum, valid_count = accumulators.pop(_grouped_job_key(job))
        draw_scratch = np.empty(vector_sum.shape, dtype=float)
        output_index = (
            slice(
                unit_start if summary_unit_start is None else unit_start - summary_unit_start,
                unit_stop if summary_unit_start is None else unit_stop - summary_unit_start,
            ),
            job.condition_index,
            site_index if summary_site_index is None else summary_site_index,
            job.epoch_index,
            slice(None),
        )
        _finalize_grouped_null_job(
            job_plan=job,
            vector_sum=vector_sum,
            valid_count=valid_count,
            draw_scratch=draw_scratch,
            eligible_cell_mask=eligible_by_job[_grouped_job_key(job)],
            observed_ppc=summary["ppc"][output_index],
            output_arrays={
                name: summary[name][output_index]
                for name in (
                    "null_exceedance_count",
                    "permutation_count",
                    "p_value",
                    "null_mean",
                    "null_std",
                    "null_p025",
                    "null_p50",
                    "null_p975",
                    "null_eligible",
                )
            },
        )
        del vector_sum, valid_count, draw_scratch


def _consume_grouped_edge_block(
    *,
    edge_statistics: object,
    job_plan: PPCJobPlan,
    schedule_block: np.ndarray,
    shuffle_start: int,
    vector_sum: np.ndarray,
    valid_count: np.ndarray,
    eligible_cell_mask: np.ndarray,
) -> None:
    """Immediately accumulate one physical-edge block into one job's draws.

    Parameters
    ----------
    edge_statistics : SegmentedEdgeStatistics
        Bounded unique physical edge sums/counts on ``(edge, unit, segment=2,
        frequency)`` axes. It is consumed before the next edge reduction.
    job_plan : PPCJobPlan
        One immutable local schedule and stable source/target edge mapping.
    schedule_block : numpy.ndarray
        Int64 ``(shuffle_block, selected_trial)`` contiguous local schedule
        rows. Its first global row is ``shuffle_start``.
    shuffle_start : int
        Nonnegative global shuffle-row offset into ``vector_sum`` and
        ``valid_count``.
    vector_sum : numpy.ndarray
        Complex128 ``(shuffle, unit, frequency)`` retained null accumulator.
    valid_count : numpy.ndarray
        Int64 array matching ``vector_sum`` of valid spike-phase counts.
    eligible_cell_mask : numpy.ndarray
        Summary-owned Boolean ``(unit, frequency)`` null-inference eligibility
        view. False cells are never accumulated and remain zero in both
        retained accumulator arrays.

    Returns
    -------
    None
        Adds only edges present in this bounded statistic block. Whole jobs add
        before and after segments before PPC calculation; half jobs use their
        named segment only.
    """
    edge_by_identity = {
        (int(source), int(target)): index
        for index, (source, target) in enumerate(
            zip(
                edge_statistics.source_trial_index,
                edge_statistics.target_trial_index,
                strict=True,
            )
        )
    }
    segment = {"before": 0, "after": 1, "whole": None}[job_plan.epoch_name]
    eligibility = np.asarray(eligible_cell_mask)
    if (
        eligibility.dtype != np.dtype(bool)
        or eligibility.shape != vector_sum.shape[1:]
        or valid_count.shape != vector_sum.shape
    ):
        raise ValueError("grouped null eligibility/accumulator axes disagree")
    for local_shuffle, global_targets in enumerate(np.asarray(schedule_block)):
        global_shuffle = shuffle_start + local_shuffle
        for source_position, target_position in enumerate(global_targets):
            source = int(job_plan.selected_trial_rows[source_position])
            target = int(job_plan.selected_trial_rows[int(target_position)])
            edge_index = edge_by_identity.get((source, target))
            if edge_index is None:
                continue
            for unit_index in range(eligibility.shape[0]):
                for frequency_index in range(eligibility.shape[1]):
                    if not bool(eligibility[unit_index, frequency_index]):
                        continue
                    vector_output = vector_sum[
                        global_shuffle,
                        unit_index : unit_index + 1,
                        frequency_index : frequency_index + 1,
                    ]
                    count_output = valid_count[
                        global_shuffle,
                        unit_index : unit_index + 1,
                        frequency_index : frequency_index + 1,
                    ]
                    if segment is None:
                        # These four explicit retained-output additions are the
                        # whole-window memory contract: no segment-axis sum and
                        # no transient ``before + after`` array is created.
                        np.add(
                            vector_output,
                            edge_statistics.phase_vector_sum[
                                edge_index,
                                unit_index : unit_index + 1,
                                0,
                                frequency_index : frequency_index + 1,
                            ],
                            out=vector_output,
                        )
                        np.add(
                            vector_output,
                            edge_statistics.phase_vector_sum[
                                edge_index,
                                unit_index : unit_index + 1,
                                1,
                                frequency_index : frequency_index + 1,
                            ],
                            out=vector_output,
                        )
                        np.add(
                            count_output,
                            edge_statistics.valid_spike_count[
                                edge_index,
                                unit_index : unit_index + 1,
                                0,
                                frequency_index : frequency_index + 1,
                            ],
                            out=count_output,
                        )
                        np.add(
                            count_output,
                            edge_statistics.valid_spike_count[
                                edge_index,
                                unit_index : unit_index + 1,
                                1,
                                frequency_index : frequency_index + 1,
                            ],
                            out=count_output,
                        )
                    else:
                        np.add(
                            vector_output,
                            edge_statistics.phase_vector_sum[
                                edge_index,
                                unit_index : unit_index + 1,
                                segment,
                                frequency_index : frequency_index + 1,
                            ],
                            out=vector_output,
                        )
                        np.add(
                            count_output,
                            edge_statistics.valid_spike_count[
                                edge_index,
                                unit_index : unit_index + 1,
                                segment,
                                frequency_index : frequency_index + 1,
                            ],
                            out=count_output,
                        )


def _finalize_grouped_null_job(
    *,
    job_plan: PPCJobPlan,
    vector_sum: np.ndarray,
    valid_count: np.ndarray,
    draw_scratch: np.ndarray,
    eligible_cell_mask: np.ndarray,
    observed_ppc: np.ndarray,
    output_arrays: Mapping[str, np.ndarray],
) -> None:
    """Finalize one job's bounded null accumulators without retaining draw copies.

    Parameters
    ----------
    job_plan : PPCJobPlan
        Immutable condition/site/epoch identity for the accumulated schedule.
    vector_sum : numpy.ndarray
        Complex128 ``(shuffle, unit, frequency)`` accumulated phase vectors.
    valid_count : numpy.ndarray
        Int64 array matching ``vector_sum`` of valid sampled phase counts.
    draw_scratch : numpy.ndarray
        Float64 writable array matching ``vector_sum``. It is the sole
        additional full-shuffle workspace beyond the required vector/count
        accumulators; each cell is compacted and sorted in place.
    eligible_cell_mask : numpy.ndarray
        Summary-owned Boolean ``(unit, frequency)`` null eligibility view.
    observed_ppc : numpy.ndarray
        Float64 ``(unit, frequency)`` observed PPC values.
    output_arrays : Mapping[str, numpy.ndarray]
        Writable summary views with ``(unit, frequency)`` axes for exact
        null counts, p values, moments, linear percentiles, and eligibility.

    Returns
    -------
    None
        Writes plus-one p values, finite-draw population moments (``ddof=0``),
        and NumPy-compatible linear 2.5/50/97.5 percentiles. Ineligible cells
        retain zero counts, false flags, and NaN floating values.

    Raises
    ------
    ValueError
        If private accumulator/scratch/output axes or dtypes disagree.

    Notes
    -----
    No legacy null summarizer, copied/concatenated draw matrix, top-level
    sort/partition, or percentile helper is used. Each one-dimensional scratch
    view is sorted with its ndarray in-place method after finite values have
    been compacted into its prefix.
    """
    import math

    del job_plan  # Identity is carried by the caller's output view selection.
    sums = np.asarray(vector_sum)
    counts = np.asarray(valid_count)
    scratch = np.asarray(draw_scratch)
    eligibility = np.asarray(eligible_cell_mask)
    observed = np.asarray(observed_ppc)
    required = {
        "null_exceedance_count",
        "permutation_count",
        "p_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
        "null_eligible",
    }
    output_dtypes = {
        "null_exceedance_count": np.dtype(np.int64),
        "permutation_count": np.dtype(np.int64),
        "p_value": np.dtype(np.float64),
        "null_mean": np.dtype(np.float64),
        "null_std": np.dtype(np.float64),
        "null_p025": np.dtype(np.float64),
        "null_p50": np.dtype(np.float64),
        "null_p975": np.dtype(np.float64),
        "null_eligible": np.dtype(bool),
    }
    if set(output_arrays) != required:
        raise ValueError("grouped null finalization output fields disagree")
    output_views = {name: np.asarray(values) for name, values in output_arrays.items()}
    if (
        sums.ndim != 3
        or sums.dtype != np.dtype(np.complex128)
        or counts.dtype != np.dtype(np.int64)
        or scratch.dtype != np.dtype(np.float64)
        or sums.shape != counts.shape
        or sums.shape != scratch.shape
        or eligibility.dtype != np.dtype(bool)
        or eligibility.shape != sums.shape[1:]
        or observed.dtype != np.dtype(np.float64)
        or observed.shape != sums.shape[1:]
        or any(
            output_views[name].dtype != output_dtypes[name]
            or output_views[name].shape != sums.shape[1:]
            for name in required
        )
    ):
        raise ValueError("grouped null finalization axes disagree")

    def linear_percentile(sorted_values: np.ndarray, count: int, quantile: float) -> float:
        """Return one linear percentile from an in-place sorted finite prefix."""
        position = (count - 1) * quantile
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        fraction = position - lower
        return float(
            sorted_values[lower]
            + fraction * (sorted_values[upper] - sorted_values[lower])
        )

    for unit_index in range(sums.shape[1]):
        for frequency_index in range(sums.shape[2]):
            values = scratch[:, unit_index, frequency_index]
            if not bool(eligibility[unit_index, frequency_index]):
                for shuffle_index in range(values.size):
                    values[shuffle_index] = math.nan
                continue
            finite_count = 0
            exceedance_count = 0
            observed_value = float(observed[unit_index, frequency_index])
            for shuffle_index in range(values.size):
                count = int(counts[shuffle_index, unit_index, frequency_index])
                if count < 2:
                    values[shuffle_index] = math.nan
                    continue
                vector = sums[shuffle_index, unit_index, frequency_index]
                value = (
                    float(vector.real * vector.real + vector.imag * vector.imag) - count
                ) / float(count * (count - 1))
                values[shuffle_index] = value
                if math.isfinite(value):
                    values[finite_count] = value
                    finite_count += 1
                    if value >= observed_value:
                        exceedance_count += 1
            for shuffle_index in range(finite_count, values.size):
                values[shuffle_index] = math.nan
            if finite_count == 0 or not math.isfinite(observed_value):
                output_arrays["null_eligible"][unit_index, frequency_index] = False
                continue
            values[:finite_count].sort()
            # ndarray reductions preserve the legacy NumPy pairwise floating
            # arithmetic while operating on the existing compact scratch
            # prefix; neither produces a retained shuffle-draw copy.
            mean = float(values[:finite_count].mean())
            standard_deviation = float(values[:finite_count].std())
            output_arrays["null_exceedance_count"][unit_index, frequency_index] = exceedance_count
            output_arrays["permutation_count"][unit_index, frequency_index] = finite_count
            output_arrays["p_value"][unit_index, frequency_index] = (
                (1.0 + exceedance_count) / (1.0 + finite_count)
            )
            output_arrays["null_mean"][unit_index, frequency_index] = mean
            output_arrays["null_std"][unit_index, frequency_index] = standard_deviation
            output_arrays["null_p025"][unit_index, frequency_index] = linear_percentile(
                values, finite_count, 0.025
            )
            output_arrays["null_p50"][unit_index, frequency_index] = linear_percentile(
                values, finite_count, 0.5
            )
            output_arrays["null_p975"][unit_index, frequency_index] = linear_percentile(
                values, finite_count, 0.975
            )
            output_arrays["null_eligible"][unit_index, frequency_index] = True


def _apply_grouped_bh_block(
    *,
    summary: dict[str, np.ndarray],
    site_index: int,
    unit_start: int,
    unit_stop: int,
    alpha: float,
) -> None:
    """Apply exact per-job frequency BH correction for one site/unit block.

    Parameters
    ----------
    summary : dict[str, numpy.ndarray]
        Writable full grouped arrays with five metric axes.
    site_index : int
        Fixed categorical site position for this block.
    unit_start, unit_stop : int
        Half-open full-unit bounds.
    alpha : float
        Dimensionless FDR significance threshold in ``[0, 1]``.

    Returns
    -------
    None
        Writes float64 q values and Boolean significant flags for this block;
        unavailable/ineligible entries stay NaN/false.
    """
    for unit_index in range(unit_start, unit_stop):
        for condition_index in range(summary["p_value"].shape[1]):
            for epoch_index in range(summary["p_value"].shape[3]):
                _apply_grouped_bh_frequency_family_in_place(
                    p_value=summary["p_value"][unit_index, condition_index, site_index, epoch_index],
                    null_eligible=summary["null_eligible"][unit_index, condition_index, site_index, epoch_index],
                    q_value=summary["q_value"][unit_index, condition_index, site_index, epoch_index],
                    significant=summary["significant"][unit_index, condition_index, site_index, epoch_index],
                    alpha=alpha,
                )


def _apply_grouped_bh_frequency_family_in_place(
    *, p_value: np.ndarray, null_eligible: np.ndarray, q_value: np.ndarray,
    significant: np.ndarray, alpha: float,
) -> None:
    """Apply BH to one bounded frequency family without a block-sized scratch.

    Parameters are float64 ``p_value``/``q_value`` and Boolean
    ``null_eligible``/``significant`` arrays with matching ``(frequency,)``
    axes; p and q values plus ``alpha`` are dimensionless. This mutates only
    the supplied q/significance views and returns ``None``.
    """
    q_value.fill(np.nan)
    significant.fill(False)
    eligible_indices = [
        index for index in range(p_value.size)
        if bool(null_eligible[index]) and np.isfinite(p_value[index])
    ]
    if not eligible_indices:
        return
    ordered = sorted(eligible_indices, key=lambda index: float(p_value[index]))
    running = 1.0
    family_size = len(ordered)
    for rank in range(family_size, 0, -1):
        index = ordered[rank - 1]
        running = min(running, float(p_value[index]) * family_size / rank)
        q_value[index] = running
        significant[index] = running <= alpha


@dataclass(frozen=True)
class _PhaseWorkDescriptor:
    """Read-only on-disk phase inputs shared by process workers.

    The six paths name read-only NPY arrays shared by all spawned workers:
    complex64/Boolean phase and validity with ``(site, frequency, trial,
    time)`` axes, float64 event-relative seconds ``(time,)``, int64 stable
    trial rows ``(trial,)``, packed float64 spike times ``(spike,)``, and
    int64 spike offsets ``(unit, trial + 1)``.  Tasks contain only scalar
    coordinates and immutable plan views, never any input payload.

    The four auxiliary paths are optional only for the frozen legacy two-array
    executor; grouped workers require the complete six-path descriptor.
    """

    phase_path: Path
    valid_path: Path
    relative_time_s_path: Path | None = None
    stable_trial_rows_path: Path | None = None
    spike_times_s_path: Path | None = None
    spike_offsets_path: Path | None = None


@dataclass(frozen=True)
class _GroupedParallelBlockTask:
    """Pickle-safe description of one current-site grouped unit block.

    Parameters
    ----------
    block_id : str
        Stable checkpoint identity for one site and half-open unit block.
    site_index, unit_start, unit_stop : int
        Categorical site position and half-open full-unit bounds. They have no
        physical units.
    site_jobs : tuple[PPCJobPlan, ...]
        Borrowed immutable result-cell plans for this site only, in canonical
        condition/epoch order. These are not a full component plan.
    site_condition_batches : tuple[tuple[int, ...], ...]
        Borrowed ordered condition batches for the current site. They bound
        null accumulation inside this one unit task.
    site_union_source_trial_row, site_union_target_trial_row : numpy.ndarray
        Borrowed read-only int64 views of the current contiguous site slice of
        the component physical-edge union. Both have ``(site_union_edge,)``
        axes of stable trial-row IDs.
    site_union_position_offset : int
        First global component-union position represented by the two site
        union views. Job union positions are rebased by this categorical
        offset before worker reduction.
    staged_result_path : pathlib.Path
        Unique disposable worker NPZ path. The worker publishes this exact
        bounded schema atomically; only the parent validates, merges, and
        writes durable checkpoints.
    config : LFPSummaryConfig
        Immutable settings metadata only. Numerical phase, coordinate, and
        spike payloads remain initializer-owned mmap arrays.
    """

    block_id: str
    site_index: int
    unit_start: int
    unit_stop: int
    site_jobs: tuple[PPCJobPlan, ...]
    site_condition_batches: tuple[tuple[int, ...], ...]
    site_union_source_trial_row: np.ndarray
    site_union_target_trial_row: np.ndarray
    site_union_position_offset: int
    staged_result_path: Path
    config: LFPSummaryConfig


@dataclass(frozen=True)
class _GroupedParallelBlockResult:
    """One bounded worker result awaiting parent merge/checkpoint publication.

    The result owns only scalar identity coordinates and the staged NPZ path.
    Numerical arrays remain in the worker staging file until the parent loads
    exactly one validated bounded block.
    """

    block_id: str
    site_index: int
    unit_start: int
    unit_stop: int
    staged_result_path: Path


_GROUPED_WORKER_PHASE: np.ndarray | None = None
_GROUPED_WORKER_VALID: np.ndarray | None = None
_GROUPED_WORKER_RELATIVE_TIME_S: np.ndarray | None = None
_GROUPED_WORKER_STABLE_TRIAL_ROWS: np.ndarray | None = None
_GROUPED_WORKER_SPIKE_TIMES_S: np.ndarray | None = None
_GROUPED_WORKER_SPIKE_OFFSETS: np.ndarray | None = None


def _materialize_grouped_phase_work_inputs(
    run_directory: Path,
    phase: np.ndarray,
    valid: np.ndarray,
    relative_time_s: np.ndarray,
    stable_trial_rows: np.ndarray,
    prepared_spikes: object,
) -> _PhaseWorkDescriptor:
    """Stream six shared grouped worker inputs into one mmap descriptor.

    Parameters
    ----------
    run_directory : pathlib.Path
        Locked grouped PPC work directory. All NPY files are disposable
        execution-only artifacts below this directory.
    phase, valid : numpy.ndarray
        Validated complex64 and Boolean ``(site, frequency, trial, time)``
        arrays.
    relative_time_s, stable_trial_rows : numpy.ndarray
        Float64 seconds ``(time,)`` and int64 stable IDs ``(trial,)``.
    prepared_spikes : PreparedSpikeRun-like
        Trial-local finite float seconds vectors for every unit/trial. Values
        are streamed directly into packed mmap storage, never concatenated.

    Returns
    -------
    _PhaseWorkDescriptor
        Paths to one shared six-array set. Files are direct execution work
        artifacts; worker initialization validates their on-disk contracts. The caller invokes
        this only when at least one grouped unit block remains pending, so warm
        resume has no mmap materialization side effect.
    """
    directory = run_directory / "worker-inputs"
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase": directory / "phase.npy",
        "valid": directory / "valid.npy",
        "relative_time_s": directory / "relative_time_s.npy",
        "stable_trial_rows": directory / "stable_trial_rows.npy",
        "spike_times_s": directory / "spike_times_s.npy",
        "spike_offsets": directory / "spike_offsets.npy",
    }

    def write_direct(path: Path, values: np.ndarray) -> None:
        """Stream one validated numeric array into ``path`` without a copy.

        ``values`` retains its documented dtype/axes; the returned value is
        ``None`` and ownership of the transient writable mmap ends here.
        """
        mapped = np.lib.format.open_memmap(
            path, mode="w+", dtype=values.dtype, shape=values.shape
        )
        mapped[...] = values
        mapped.flush()
        del mapped

    write_direct(paths["phase"], phase)
    write_direct(paths["valid"], valid)
    write_direct(paths["relative_time_s"], relative_time_s)
    write_direct(paths["stable_trial_rows"], stable_trial_rows)
    trial_count = stable_trial_rows.size
    trains = tuple(prepared_spikes.trial_spike_trains)
    total_spikes = sum(
        int(np.asarray(times).size)
        for train in trains
        for times in train.relative_spike_times
    )
    offsets = np.lib.format.open_memmap(
        paths["spike_offsets"], mode="w+", dtype=np.int64,
        shape=(len(trains), trial_count + 1),
    )
    packed = np.lib.format.open_memmap(
        paths["spike_times_s"], mode="w+", dtype=np.float64,
        shape=(total_spikes,),
    )
    cursor = 0
    for unit_index, train in enumerate(trains):
        offsets[unit_index, 0] = cursor
        for trial_index, times in enumerate(train.relative_spike_times):
            values = np.asarray(times, dtype=np.float64)
            stop = cursor + values.size
            packed[cursor:stop] = values
            cursor = stop
            offsets[unit_index, trial_index + 1] = cursor
    packed.flush()
    offsets.flush()
    del packed, offsets
    return _PhaseWorkDescriptor(
        phase_path=paths["phase"], valid_path=paths["valid"],
        relative_time_s_path=paths["relative_time_s"],
        stable_trial_rows_path=paths["stable_trial_rows"],
        spike_times_s_path=paths["spike_times_s"],
        spike_offsets_path=paths["spike_offsets"],
    )


def _initialize_grouped_parallel_worker(
    phase_descriptor: _PhaseWorkDescriptor,
) -> None:
    """Open six grouped shared input maps once in one spawned worker.

    Parameters
    ----------
    phase_descriptor : _PhaseWorkDescriptor
        Existing NPY paths for complex64 phase/Boolean validity ``(site,
        frequency, full_trial, time)``, float64 seconds ``(time,)``, int64
        stable rows ``(trial,)``, packed float64 spikes ``(spike,)``, and int64
        offsets ``(unit, trial + 1)``. All are opened ``mmap_mode="r"``.

    Returns
    -------
    None
        Stores all six read-only maps in module-private worker globals.
        Subsequent tasks reconstruct bounded spike views without reopening or
        copying any complete shared input.

    Raises
    ------
    ValueError
        If any of six mmap dtypes, axes, read-only flags, offsets, or the
        float64 ``(time,)`` length disagrees with phase ``time``. I/O failures
        propagate unchanged.
    """
    global _GROUPED_WORKER_PHASE, _GROUPED_WORKER_VALID
    global _GROUPED_WORKER_RELATIVE_TIME_S, _GROUPED_WORKER_STABLE_TRIAL_ROWS
    global _GROUPED_WORKER_SPIKE_TIMES_S, _GROUPED_WORKER_SPIKE_OFFSETS
    if any(path is None for path in (
        phase_descriptor.relative_time_s_path,
        phase_descriptor.stable_trial_rows_path,
        phase_descriptor.spike_times_s_path,
        phase_descriptor.spike_offsets_path,
    )):
        raise ValueError("grouped worker descriptor lacks shared input paths")
    phase = np.load(phase_descriptor.phase_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(phase_descriptor.valid_path, mmap_mode="r", allow_pickle=False)
    relative_time_s = np.load(
        phase_descriptor.relative_time_s_path, mmap_mode="r", allow_pickle=False
    )
    stable_trial_rows = np.load(
        phase_descriptor.stable_trial_rows_path, mmap_mode="r", allow_pickle=False
    )
    spike_times_s = np.load(
        phase_descriptor.spike_times_s_path, mmap_mode="r", allow_pickle=False
    )
    spike_offsets = np.load(
        phase_descriptor.spike_offsets_path, mmap_mode="r", allow_pickle=False
    )
    if (
        phase.dtype != np.dtype(np.complex64)
        or valid.dtype != np.dtype(bool)
        or phase.shape != valid.shape
        or phase.ndim != 4
        or relative_time_s.dtype != np.dtype(np.float64)
        or stable_trial_rows.dtype != np.dtype(np.int64)
        or spike_times_s.dtype != np.dtype(np.float64)
        or spike_offsets.dtype != np.dtype(np.int64)
        or relative_time_s.shape != (phase.shape[3],)
        or stable_trial_rows.shape != (phase.shape[2],)
        or spike_offsets.shape[1:] != (phase.shape[2] + 1,)
        or spike_offsets.shape[0] < 1
        or int(spike_offsets[-1, -1]) != spike_times_s.size
        or phase.flags.writeable
        or valid.flags.writeable
        or relative_time_s.flags.writeable
        or stable_trial_rows.flags.writeable
        or spike_times_s.flags.writeable
        or spike_offsets.flags.writeable
    ):
        raise ValueError("grouped worker phase mmap contract is invalid")
    _GROUPED_WORKER_PHASE = phase
    _GROUPED_WORKER_VALID = valid
    _GROUPED_WORKER_RELATIVE_TIME_S = relative_time_s
    _GROUPED_WORKER_STABLE_TRIAL_ROWS = stable_trial_rows
    _GROUPED_WORKER_SPIKE_TIMES_S = spike_times_s
    _GROUPED_WORKER_SPIKE_OFFSETS = spike_offsets


def _compute_grouped_parallel_block(
    task: _GroupedParallelBlockTask,
) -> _GroupedParallelBlockResult:
    """Compute one grouped site/unit block from initializer-owned phase mmaps.

    Parameters
    ----------
    task : _GroupedParallelBlockTask
        Pickle-safe site-local plan, scalar unit coordinates, staging path,
        and immutable configuration. It owns no phase, time, stable-row, or
        spike payload; those six arrays are initializer-owned mmaps.

    Returns
    -------
    _GroupedParallelBlockResult
        Scalar identity/path record for the atomically staged exact
        checkpoint-schema NPZ. The parent is the only checkpoint publisher.

    Raises
    ------
    RuntimeError
        If the spawned-worker initializer has not opened all six shared maps.
    ValueError
        If bounded task coordinates or grouped reduction contracts disagree.
    """
    phase = _GROUPED_WORKER_PHASE
    valid = _GROUPED_WORKER_VALID
    relative_time_s = _GROUPED_WORKER_RELATIVE_TIME_S
    stable_trial_rows = _GROUPED_WORKER_STABLE_TRIAL_ROWS
    spike_times_s = _GROUPED_WORKER_SPIKE_TIMES_S
    spike_offsets = _GROUPED_WORKER_SPIKE_OFFSETS
    if any(values is None for values in (
        phase, valid, relative_time_s, stable_trial_rows, spike_times_s, spike_offsets,
    )):
        raise RuntimeError("grouped parallel worker was not initialized")
    assert phase is not None and valid is not None
    assert relative_time_s is not None and stable_trial_rows is not None
    assert spike_times_s is not None and spike_offsets is not None
    if any(values.flags.writeable for values in (
        phase, valid, relative_time_s, stable_trial_rows, spike_times_s, spike_offsets,
    )):
        raise RuntimeError("grouped parallel worker mmap inputs must be read-only")
    block_units = task.unit_stop - task.unit_start
    if block_units <= 0 or task.unit_stop > spike_offsets.shape[0]:
        raise ValueError("grouped parallel task unit bounds/spike offsets disagree")
    if not task.site_jobs or not task.site_condition_batches:
        raise ValueError("grouped parallel task requires current-site jobs and batches")
    if any(job.site_index != task.site_index for job in task.site_jobs):
        raise ValueError("grouped parallel task includes a foreign-site job")
    condition_count = max(job.condition_index for job in task.site_jobs) + 1
    prepared_phase = SimpleNamespace(
        phase_tensor=phase,
        phase_valid=valid,
        relative_time_s=relative_time_s,
        trial_indices=stable_trial_rows,
    )
    trial_spike_trains = []
    for unit_index in range(task.unit_start, task.unit_stop):
        relative_spikes = tuple(
            spike_times_s[
                int(spike_offsets[unit_index, trial_index]):int(
                    spike_offsets[unit_index, trial_index + 1]
                )
            ]
            for trial_index in range(stable_trial_rows.size)
        )
        trial_spike_trains.append(
            SimpleNamespace(
                unit_id=str(task.config.unit_population.stable_unit_ids[unit_index]),
                relative_spike_times=relative_spikes,
            )
        )
    prepared_spikes = SimpleNamespace(
        unit_ids=tuple(train.unit_id for train in trial_spike_trains),
        trial_spike_trains=tuple(trial_spike_trains),
    )
    summary = _empty_grouped_summary_arrays(
        unit_count=block_units,
        condition_count=condition_count,
        site_count=1,
        epoch_count=len(task.config.ppc.epochs),
        frequency_count=len(task.config.phase.frequency_hz),
        representative_band_count=_S1_S2_REPRESENTATIVE_BAND_COUNT,
        phase_bin_count=len(task.config.ppc.phase_bin_edges_rad) - 1,
    )
    _compute_grouped_site_unit_block(
        config=task.config,
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        plan=None,
        summary=summary,
        site_index=task.site_index,
        unit_start=task.unit_start,
        unit_stop=task.unit_stop,
        site_jobs=task.site_jobs,
        site_condition_batches=task.site_condition_batches,
        site_union_source_trial_row=task.site_union_source_trial_row,
        site_union_target_trial_row=task.site_union_target_trial_row,
        site_union_position_offset=task.site_union_position_offset,
        unit_spike_trains=tuple(trial_spike_trains),
        summary_unit_start=task.unit_start,
        summary_site_index=0,
    )
    _apply_grouped_bh_block(
        summary=summary,
        site_index=0,
        unit_start=0,
        unit_stop=block_units,
        alpha=task.config.ppc.fdr_alpha,
    )
    # The worker summary has exactly one site, so dropping that singleton axis
    # yields C-contiguous borrowed checkpoint views rather than a second block.
    arrays = {name: values[:, :, 0] for name, values in summary.items()}
    arrays["site_index"] = np.asarray([task.site_index], dtype=np.int64)
    arrays["unit_bounds"] = np.asarray([task.unit_start, task.unit_stop], dtype=np.int64)
    task.staged_result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = task.staged_result_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary_path, task.staged_result_path)
    del arrays, summary
    return _GroupedParallelBlockResult(
        block_id=task.block_id,
        site_index=task.site_index,
        unit_start=task.unit_start,
        unit_stop=task.unit_stop,
        staged_result_path=task.staged_result_path,
    )


def _run_grouped_parallel_block_batches(
    *,
    phase_descriptor: _PhaseWorkDescriptor,
    block_tasks: tuple[_GroupedParallelBlockTask, ...],
    worker_count: int,
    compute_block: Callable[[_GroupedParallelBlockTask], _GroupedParallelBlockResult],
) -> Iterator[_GroupedParallelBlockResult]:
    """Yield one current-site worker pool's results in canonical task order.

    Parameters
    ----------
    phase_descriptor : _PhaseWorkDescriptor
        One shared read-only six-input mmap descriptor supplied once to each
        spawned worker initializer.
    block_tasks : tuple[_GroupedParallelBlockTask, ...]
        Canonically ordered, nonempty current-site unit blocks only. Conditions
        never become process tasks.
    worker_count : int
        Positive current-site active-worker bound, no greater than
        ``len(block_tasks)``; invalid values raise ``ValueError``.
    compute_block : callable
        Top-level spawn-safe worker function. Production uses
        :func:`_compute_grouped_parallel_block`; tests inject a deterministic
        callable.

    Yields
    ------
    _GroupedParallelBlockResult
        One completed result in input task order. The generator pauses after
        each yield so the parent can validate scalar identity, merge/checkpoint
        its staged arrays, and then submit a replacement.

    Raises
    ------
    ValueError
        If ``worker_count`` is outside ``[1, len(block_tasks)]``.
    Exception
        Worker exceptions propagate after current and queued futures are
        cancelled and the pool performs a wait-for-workers shutdown. The
        parent validates task/result identity after each yield.
    """
    if not block_tasks:
        return
    if worker_count < 1 or worker_count > len(block_tasks):
        raise ValueError("grouped parallel worker_count must fit pending site blocks")
    task_iterator = iter(block_tasks)
    pending: list[object] = []
    executor: object | None = None
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_grouped_parallel_worker,
        initargs=(phase_descriptor,),
    ) as executor:
        try:
            for _ in range(worker_count):
                try:
                    task = next(task_iterator)
                except StopIteration:
                    break
                pending.append(executor.submit(compute_block, task))
            while pending:
                future = pending[0]
                result = future.result()
                pending.pop(0)
                yield result
                try:
                    task = next(task_iterator)
                except StopIteration:
                    continue
                pending.append(executor.submit(compute_block, task))
        finally:
            if pending:
                for submitted_future in pending:
                    submitted_future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class _ParallelBlockTask:
    """Pickle-safe, phase-free description of one contiguous unit block.

    ``trains`` contains finite event-relative spike-time arrays in seconds,
    with ``(unit in block, trial)`` nesting. ``schedule`` has int64
    ``(shuffle, trial)`` axes and ``frequencies_hz`` has shape ``(frequency,)``.
    No phase tensor or validity mask is embedded in this task.
    """

    block_id: str
    trains: tuple[tuple[np.ndarray, ...], ...]
    time_s: np.ndarray
    frequencies_hz: np.ndarray
    schedule: np.ndarray
    execution: PPCExecutionConfig


@dataclass(frozen=True)
class _ParallelBlockResult:
    """One worker's in-memory summary arrays for its named unit block."""

    block_id: str
    arrays: dict[str, np.ndarray]


def execute_ppc_blocks(
    *,
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    schedule: np.ndarray | None,
    work_root: Path,
    progress_callback: Callable[[ProgressEvent], None] | None = None,
) -> PPCExecutionResult:
    """Execute one serial PPC job with bounded restart checkpoints.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated scientific configuration. ``config.phase.frequency_hz`` is
        the exact ascending Hz coordinate used by the phase frequency axis;
        PPC reliability, shuffle, seed, and FDR settings define the estimator.
    execution : PPCExecutionConfig
        Validated execution-only block sizes, checkpoint policy, and progress
        cadence. These values affect work identity and memory only, never the
        final component scientific fingerprint.
    prepared_phase : object
        Must expose complex64 ``phase_tensor`` and Boolean ``phase_valid`` with
        identical ``(site=1, frequency, trial, time)`` axes, float64 finite
        increasing ``relative_time_s`` in event-relative seconds with shape
        ``(time,)``, and int64 stable ``trial_indices`` with shape ``(trial,)``.
        Optional ``site_id``, ``condition_name``, ``epoch_bounds_s`` in seconds,
        and source/representation fingerprints refine job identity. Invalid
        phase samples are marked by ``phase_valid`` rather than inferred from
        a collapsed mask.
    prepared_spikes : object
        Must expose unique stable ``unit_ids`` and ``trial_spike_trains``. Each
        train exposes finite event-relative seconds arrays on ``(trial,)`` and
        integer overlap trial identities. A physical spike may occur in more
        than one trial-local array when windows overlap.
    schedule : numpy.ndarray or None
        Caller-owned int64 ``(shuffle, trial)`` derangements. ``None`` creates
        one schedule from the saved PPC seed; the returned result owns its copy.
    work_root : pathlib.Path
        Session-local parent for execution-only work artifacts. No final cache
        component or manifest is written below this path by this function.
    progress_callback : callable or None
        Receives ``ProgressEvent`` records with this job's stable ``job_id``;
        records contain counts/time metadata only, never raw arrays.

    Returns
    -------
    PPCExecutionResult
        Exact observed and null summaries on ``(unit, frequency)`` axes. PPC
        preserves negative values; unavailable numerical outputs are NaN.
        Checkpoints contain summary arrays only and are never scientific cache
        components.

    Raises
    ------
    ValueError
        If phase, spike, time, frequency, trial, or schedule contracts differ.
    FileExistsError
        If another writer holds the exact ``executor.lock``.
    OSError
        If schedule/checkpoint creation, validation, or atomic replacement
        fails. Such a failure cannot publish a final scientific component.
    """
    # Validate execution settings before inspecting phase inputs or creating a
    # work directory.  In particular, an invalid process count must be a
    # side-effect-free programming error.
    _validate_ppc_execution(execution)
    phase, valid, time_s, frequencies_hz, trial_indices = _validated_phase_inputs(
        config, prepared_phase
    )
    trains, unit_ids, overlap_indices = _validated_spike_inputs(
        prepared_spikes, phase.shape[2]
    )
    schedule_array = _resolve_schedule(config, schedule, phase.shape[2])
    metadata = _build_run_metadata(
        config,
        execution,
        prepared_phase,
        prepared_spikes,
        phase,
        valid,
        time_s,
        frequencies_hz,
        trial_indices,
        trains,
        unit_ids,
        overlap_indices,
        schedule_array,
    )
    run_fingerprint = str(metadata["run_fingerprint"])
    run_directory = Path(work_root) / "ppc" / run_fingerprint
    progress = _ProgressReporter(progress_callback, str(metadata["job_id"]))
    progress.emit("prepare_phase", 1, 1, "validated prepared phase")

    if not execution.checkpoint_enabled:
        summary = _compute_unit_blocks(
            trains, phase, valid, time_s, frequencies_hz, schedule_array,
            execution, progress, None,
        )
        _apply_bh_and_significance(summary, config.ppc.fdr_alpha)
        progress.emit("fdr", 1, 1, "adjusted p values", timed=True)
        progress.emit("commit", 1, 1, "PPC summaries ready")
        return PPCExecutionResult(
            run_fingerprint, run_directory, schedule_array.copy(), summary, (), ()
        )

    run_directory.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_executor_lock(run_directory, run_fingerprint)
    try:
        schedule_matches = _stored_schedule_matches(
            run_directory,
            schedule_array,
        )
        if not schedule_matches:
            _write_schedule(run_directory, schedule_array)
        resumed = _load_resumable_blocks(
            run_directory,
            metadata,
            schedule_matches,
            len(trains),
            frequencies_hz.size,
        )
        summary = _empty_summary_arrays(len(trains), frequencies_hz.size)
        for block_id, arrays in resumed.items():
            start, stop = _block_bounds(block_id)
            _copy_block_into_summary(summary, arrays, start, stop)
        resumed_ids = tuple(sorted(resumed))

        def checkpoint(
            block_id: str,
            arrays: Mapping[str, np.ndarray],
            complete_ids: tuple[str, ...],
        ) -> None:
            """Persist one validated unit block while the executor owns its lock."""
            block_metadata = dict(metadata)
            block_metadata["completed_block_ids"] = list(complete_ids)
            write_ppc_checkpoint(run_directory, block_id, arrays, block_metadata)

        if execution.worker_count == 1:
            summary = _compute_unit_blocks(
                trains, phase, valid, time_s, frequencies_hz, schedule_array,
                execution, progress, checkpoint, summary, set(resumed_ids),
            )
        else:
            summary = _compute_parallel_unit_blocks(
                trains=trains,
                phase=phase,
                valid=valid,
                time_s=time_s,
                frequencies_hz=frequencies_hz,
                schedule=schedule_array,
                execution=execution,
                run_directory=run_directory,
                progress=progress,
                checkpoint_writer=checkpoint,
                initial_summary=summary,
                resumed_block_ids=set(resumed_ids),
            )
        _apply_bh_and_significance(summary, config.ppc.fdr_alpha)
        all_ids = _block_ids(len(trains), execution.unit_block_size)
        for block_id in all_ids:
            start, stop = _block_bounds(block_id)
            checkpoint(block_id, _summary_block(summary, start, stop), all_ids)
        _atomic_json(
            run_directory / "complete.json",
            {"run_fingerprint": run_fingerprint},
        )
        progress.emit("fdr", 1, 1, "adjusted p values", timed=True)
        progress.emit("checkpoint", 1, 2, "wrote validated checkpoints", timed=True)
        progress.emit("checkpoint", 2, 2, "checkpoint set complete", timed=True)
        progress.emit("commit", 1, 1, "PPC summaries ready")
        return PPCExecutionResult(
            run_fingerprint, run_directory, schedule_array.copy(),
            {name: values.copy() for name, values in summary.items()}, all_ids,
            resumed_ids,
        )
    finally:
        _release_executor_lock(lock_path)


def _validated_phase_inputs(
    config: LFPSummaryConfig, prepared_phase: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate exact complex phase axes without repeating frequency data."""
    phase = np.asarray(prepared_phase.phase_tensor)
    valid = np.asarray(prepared_phase.phase_valid, dtype=bool)
    time_s = np.asarray(prepared_phase.relative_time_s, dtype=float)
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    trial_indices = np.asarray(
        getattr(prepared_phase, "trial_indices", ()),
        dtype=np.int64,
    )
    expected_shape = (1, frequencies_hz.size, trial_indices.size, time_s.size)
    if phase.dtype != np.dtype(np.complex64) or phase.shape != expected_shape:
        raise ValueError("prepared PPC phase frequency and axes must match configuration")
    if valid.shape != phase.shape or time_s.ndim != 1 or trial_indices.ndim != 1:
        raise ValueError("prepared PPC validity, trial, or time axes are invalid")
    if time_s.size and (
        not np.isfinite(time_s).all() or np.any(np.diff(time_s) <= 0.0)
    ):
        raise ValueError("prepared PPC time axis must be finite and strictly increasing")
    return phase, valid, time_s, frequencies_hz, trial_indices


def _validated_spike_inputs(
    prepared_spikes: object, trial_count: int
) -> tuple[
    tuple[tuple[np.ndarray, ...], ...],
    tuple[str, ...],
    tuple[tuple[int, ...], ...],
]:
    """Validate trial-local seconds arrays and preserve overlap identities."""
    unit_ids = tuple(str(value) for value in prepared_spikes.unit_ids)
    trains: list[tuple[np.ndarray, ...]] = []
    overlaps: list[tuple[int, ...]] = []
    for train in prepared_spikes.trial_spike_trains:
        times = tuple(
            np.asarray(value, dtype=float) for value in train.relative_spike_times
        )
        if len(times) != trial_count or any(
            value.ndim != 1 or not np.isfinite(value).all() for value in times
        ):
            raise ValueError("prepared PPC spikes must be finite (unit, trial) seconds arrays")
        trains.append(times)
        overlaps.append(
            tuple(
                int(value)
                for value in np.asarray(train.overlap_trial_indices, dtype=np.int64)
            )
        )
    if len(trains) != len(unit_ids) or len(set(unit_ids)) != len(unit_ids):
        raise ValueError("prepared PPC units must have unique matching stable identifiers")
    return tuple(trains), unit_ids, tuple(overlaps)


def _resolve_schedule(
    config: LFPSummaryConfig, schedule: np.ndarray | None, trial_count: int
) -> np.ndarray:
    """Return one validated int64 derangement schedule for the selected trials."""
    if schedule is None:
        if trial_count < 2:
            return np.empty((0, trial_count), dtype=np.int64)
        return generate_trial_derangement_schedule(
            trial_count, config.ppc.shuffle_count, seed=config.ppc.seed
        )
    candidate = np.asarray(schedule)
    if trial_count < 2 and candidate.shape == (0, trial_count):
        return candidate.astype(np.int64, copy=True)
    return spike_lfp_summary._derangement_schedule(candidate, trial_count)


def _build_run_metadata(
    config: LFPSummaryConfig,
    execution: PPCExecutionConfig,
    prepared_phase: object,
    prepared_spikes: object,
    phase: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    trial_indices: np.ndarray,
    trains: tuple[tuple[np.ndarray, ...], ...],
    unit_ids: tuple[str, ...],
    overlap_indices: tuple[tuple[int, ...], ...],
    schedule: np.ndarray,
) -> dict[str, object]:
    """Build complete canonical identity metadata for one resumable PPC job."""
    site_id = str(getattr(prepared_phase, "site_id", "site-0"))
    condition_name = str(getattr(prepared_phase, "condition_name", "condition"))
    epoch_bounds = tuple(float(value) for value in getattr(prepared_phase, "epoch_bounds_s", ()))
    job_id = ":".join((site_id, condition_name, *(str(value) for value in epoch_bounds)))
    source = getattr(prepared_phase, "source_fingerprint", None)
    source_fingerprint = source if isinstance(source, str) else _hash_json(fingerprint_source_files(config, "spike_phase"))
    representation = getattr(prepared_phase, "representation_fingerprint", None)
    representation_fingerprint = representation if isinstance(representation, str) else _array_fingerprint(phase, valid)
    populations = tuple(str(value) for value in getattr(prepared_spikes, "population_ids", ()))
    metadata: dict[str, object] = {
        "job_id": job_id,
        "source_fingerprint": source_fingerprint,
        "scientific_fingerprint": component_fingerprint("spike_phase", config),
        "representation_fingerprint": representation_fingerprint,
        "site_id": site_id,
        "condition_name": condition_name,
        "epoch_bounds_s": list(epoch_bounds),
        "trial_indices": trial_indices.tolist(),
        "unit_ids": list(unit_ids),
        "population_id": populations[0] if populations else "selected_population",
        "spike_fingerprint": _hash_spikes(trains, overlap_indices),
        "overlap_trial_indices": [list(row) for row in overlap_indices],
        "schedule_seed": int(config.ppc.seed),
        "schedule_fingerprint": _array_fingerprint(schedule),
        "schema_version": config.schema_version,
        "code_version": "wp5c-4-v1",
        "axes": ["site", "frequency", "trial", "time"],
        "shapes": {"phase": list(phase.shape), "valid": list(valid.shape), "schedule": list(schedule.shape)},
        "dtypes": {"phase": str(phase.dtype), "valid": str(valid.dtype), "schedule": str(schedule.dtype)},
        "execution_settings": {
            "unit_block_size": execution.unit_block_size,
            "shuffle_block_size": execution.shuffle_block_size,
            "trial_edge_block_size": execution.trial_edge_block_size,
            "worker_count": execution.worker_count,
            "maximum_worker_allocation_bytes": execution.maximum_worker_allocation_bytes,
            "maximum_aggregate_allocation_bytes": execution.maximum_aggregate_allocation_bytes,
            "checkpoint_enabled": execution.checkpoint_enabled,
            "checkpoint_retention": execution.checkpoint_retention,
            "progress_update_interval": execution.progress_update_interval,
        },
        "phase_content_fingerprint": _array_fingerprint(phase, valid),
        "completed_block_ids": [],
    }
    identity = dict(metadata)
    identity.pop("completed_block_ids")
    metadata["run_fingerprint"] = _hash_json(identity)
    return metadata


def _compute_unit_blocks(
    trains: tuple[tuple[np.ndarray, ...], ...],
    phase: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    schedule: np.ndarray,
    execution: PPCExecutionConfig,
    progress: "_ProgressReporter",
    checkpoint_writer: Callable[[str, Mapping[str, np.ndarray], tuple[str, ...]], None] | None,
    initial_summary: dict[str, np.ndarray] | None = None,
    resumed_block_ids: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute/reuse deterministic unit blocks while retaining bounded draws only."""
    summary = _empty_summary_arrays(len(trains), frequencies_hz.size) if initial_summary is None else initial_summary
    block_ids = _block_ids(len(trains), execution.unit_block_size)
    resumed = resumed_block_ids or set()
    progress.emit("observed_reduction", 0, len(block_ids), "starting observed reduction")
    for completed, block_id in enumerate(block_ids, start=1):
        start, stop = _block_bounds(block_id)
        if block_id not in resumed:
            block = _empty_summary_arrays(stop - start, frequencies_hz.size)
            _compute_observed_block(
                block,
                trains[start:stop],
                phase,
                valid,
                time_s,
                frequencies_hz,
                execution.trial_edge_block_size,
            )
            _compute_null_block(block, trains[start:stop], phase, valid, time_s, frequencies_hz, schedule, execution)
            _copy_block_into_summary(summary, block, start, stop)
            if checkpoint_writer is not None:
                checkpoint_writer(block_id, block, tuple(sorted(resumed | set(block_ids[:completed]))))
        progress.emit("observed_reduction", completed, len(block_ids), "completed unit block", timed=True)
        progress.emit("trial_edge_reduction", completed, len(block_ids), "reduced scheduled edges", timed=True)
        progress.emit("shuffle_aggregation", completed, len(block_ids), "aggregated shuffled PPC", timed=True)
    return summary


def _compute_parallel_unit_blocks(
    *,
    trains: tuple[tuple[np.ndarray, ...], ...],
    phase: np.ndarray,
    valid: np.ndarray,
    time_s: np.ndarray,
    frequencies_hz: np.ndarray,
    schedule: np.ndarray,
    execution: PPCExecutionConfig,
    run_directory: Path,
    progress: "_ProgressReporter",
    checkpoint_writer: Callable[[str, Mapping[str, np.ndarray], tuple[str, ...]], None],
    initial_summary: dict[str, np.ndarray],
    resumed_block_ids: set[str],
) -> dict[str, np.ndarray]:
    """Compute pending unit blocks in processes while the parent owns output.

    Inputs have the same axes and units as :func:`_compute_unit_blocks`.  The
    parent alone emits progress and writes checkpoints.  Worker tasks contain
    only their unit-local spike arrays and scalar/coordinate inputs; workers
    open the common complex64/Boolean phase arrays read-only from ``.npy``.
    """
    block_ids = _block_ids(len(trains), execution.unit_block_size)
    pending = tuple(block_id for block_id in block_ids if block_id not in resumed_block_ids)
    descriptor = _materialize_phase_work_inputs(run_directory, phase, valid)
    tasks = tuple(
        _ParallelBlockTask(
            block_id=block_id,
            trains=trains[_block_bounds(block_id)[0]:_block_bounds(block_id)[1]],
            time_s=time_s,
            frequencies_hz=frequencies_hz,
            schedule=schedule,
            execution=execution,
        )
        for block_id in pending
    )
    compute_block = partial(_compute_parallel_block, phase_descriptor=descriptor)
    results = iter(
        _run_parallel_worker_batches(
            phase_descriptor=descriptor,
            block_tasks=tasks,
            worker_count=execution.worker_count,
            compute_block=compute_block,
        )
    )
    completed_ids = set(resumed_block_ids)
    progress.emit("observed_reduction", 0, len(block_ids), "starting observed reduction")
    for completed, block_id in enumerate(block_ids, start=1):
        if block_id not in resumed_block_ids:
            result = next(results)
            if result.block_id != block_id:
                raise RuntimeError("parallel PPC worker returned blocks out of canonical order")
            start, stop = _block_bounds(block_id)
            _copy_block_into_summary(initial_summary, result.arrays, start, stop)
            completed_ids.add(block_id)
            # Checkpoint before requesting the next generator value. This
            # makes every yielded block independently resumable on a failure.
            checkpoint_writer(block_id, result.arrays, tuple(sorted(completed_ids)))
        progress.emit("observed_reduction", completed, len(block_ids), "completed unit block", timed=True)
        progress.emit("trial_edge_reduction", completed, len(block_ids), "reduced scheduled edges", timed=True)
        progress.emit("shuffle_aggregation", completed, len(block_ids), "aggregated shuffled PPC", timed=True)
    return initial_summary


def _materialize_phase_work_inputs(
    run_directory: Path,
    phase: np.ndarray,
    valid: np.ndarray,
) -> _PhaseWorkDescriptor:
    """Atomically materialize validated shared phase arrays for worker mmap.

    Parameters are complex64 and Boolean arrays on ``(site, frequency, trial,
    time)`` axes. The returned paths are under the exact locked run directory;
    they are execution-only inputs, not analysis components.
    """
    phase_path = run_directory / "worker_phase.npy"
    valid_path = run_directory / "worker_valid.npy"
    _atomic_npy(phase_path, phase)
    _atomic_npy(valid_path, valid)
    return _PhaseWorkDescriptor(phase_path=phase_path, valid_path=valid_path)


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    """Write and re-open one exact NPY array before atomic publication."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
        loaded = np.load(temporary, mmap_mode="r", allow_pickle=False)
        if loaded.dtype != values.dtype or loaded.shape != values.shape or not np.array_equal(loaded, values):
            raise ValueError("temporary PPC worker input validation failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _compute_parallel_block(
    task: _ParallelBlockTask,
    *,
    phase_descriptor: _PhaseWorkDescriptor,
) -> _ParallelBlockResult:
    """Compute one pure worker block from read-only phase memory maps.

    ``task`` has unit-local trial spike arrays in seconds and no phase payload.
    The returned arrays have ``(unit in block, frequency)`` axes with the same
    documented units as the serial summary arrays.
    """
    phase = np.load(phase_descriptor.phase_path, mmap_mode="r", allow_pickle=False)
    valid = np.load(phase_descriptor.valid_path, mmap_mode="r", allow_pickle=False)
    block = _empty_summary_arrays(len(task.trains), task.frequencies_hz.size)
    _compute_observed_block(
        block, task.trains, phase, valid, task.time_s, task.frequencies_hz,
        task.execution.trial_edge_block_size,
    )
    _compute_null_block(
        block, task.trains, phase, valid, task.time_s, task.frequencies_hz,
        task.schedule, task.execution,
    )
    return _ParallelBlockResult(task.block_id, block)


def _run_parallel_worker_batches(
    *,
    phase_descriptor: _PhaseWorkDescriptor,
    block_tasks: tuple[_ParallelBlockTask, ...],
    worker_count: int,
    compute_block: Callable[[_ParallelBlockTask], _ParallelBlockResult],
) -> Iterator[_ParallelBlockResult]:
    """Yield process-worker block results in input order, one at a time.

    At most ``worker_count`` tasks are submitted at once.  The initial window
    lets independent unit blocks use all requested workers, while ordered
    yielding lets the parent checkpoint each result before a replacement task
    is submitted.  On a worker failure, every submitted future is cancelled
    before the executor is shut down; already-yielded results remain available
    to the parent checkpoint writer.

    ``compute_block`` is a pickle-safe top-level-function partial in production
    and is injectable solely for deterministic executor-contract tests.
    """
    del phase_descriptor  # The descriptor is captured by the pickle-safe partial.
    task_iterator = iter(block_tasks)
    pending: list[object] = []
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        # Fill the bounded initial submission window before waiting on work.
        for _ in range(worker_count):
            try:
                task = next(task_iterator)
            except StopIteration:
                break
            pending.append(executor.submit(compute_block, task))

        while pending:
            # Retain the current future until its result has succeeded so a
            # failure can cancel both it and the remaining submission window.
            future = pending[0]
            try:
                result = future.result()
            except BaseException:
                for submitted_future in pending:
                    submitted_future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            pending.pop(0)
            yield result

            # The parent has now copied and checkpointed ``result``. Submit no
            # more than one replacement, retaining the bounded work window.
            try:
                task = next(task_iterator)
            except StopIteration:
                continue
            pending.append(executor.submit(compute_block, task))


def _compute_observed_block(
    block: dict[str, np.ndarray], trains: Sequence[tuple[np.ndarray, ...]],
    phase: np.ndarray, valid: np.ndarray, time_s: np.ndarray, frequencies_hz: np.ndarray,
    trial_edge_block_size: int,
) -> None:
    """Populate observed PPC/count statistics for one bounded unit block."""
    phase_by_trial = np.moveaxis(phase[0], 1, 0)
    valid_by_trial = np.moveaxis(valid[0], 1, 0)
    for unit_index, spikes in enumerate(trains):
        _observed(
            block,
            unit_index,
            spikes,
            time_s,
            phase_by_trial,
            valid_by_trial,
            frequencies_hz,
            trial_edge_block_size,
        )


def _compute_null_block(
    block: dict[str, np.ndarray], trains: Sequence[tuple[np.ndarray, ...]],
    phase: np.ndarray, valid: np.ndarray, time_s: np.ndarray, frequencies_hz: np.ndarray,
    schedule: np.ndarray, execution: PPCExecutionConfig,
) -> None:
    """Infer each unit only over its reliable, two-trial frequency subset."""
    if schedule.size == 0:
        return
    phase_by_trial = np.moveaxis(phase[0], 1, 0)
    valid_by_trial = np.moveaxis(valid[0], 1, 0)
    for unit_index, spikes in enumerate(trains):
        eligible = block["reliable"][unit_index] & (block["eligible_trial_count"][unit_index] >= 2)
        positions = np.flatnonzero(eligible)
        if positions.size == 0:
            continue
        draws = _stream_null_draws(
            (spikes,), time_s, phase_by_trial[:, positions], valid_by_trial[:, positions],
            frequencies_hz[positions], schedule, execution,
        )[:, 0]
        null = spike_lfp_summary.summarize_permutation_null(
            observed_ppc=block["ppc"][unit_index, positions],
            null_ppc_chunks=(draws,),
            spike_count=block["spike_count"][unit_index, positions],
            eligible_trial_count=block["eligible_trial_count"][unit_index, positions],
        )
        for name in ("null_exceedance_count", "permutation_count", "p_value", "null_mean", "null_std", "null_p025", "null_p50", "null_p975", "null_eligible"):
            block[name][unit_index, positions] = getattr(null, name)


def _stream_null_draws(
    trains: Sequence[Sequence[np.ndarray]], time_s: np.ndarray, phase: np.ndarray,
    valid: np.ndarray, frequencies_hz: np.ndarray, schedule: np.ndarray,
    execution: PPCExecutionConfig,
) -> np.ndarray:
    """Stream edge and shuffle blocks into one current unit-block draw array."""
    source_edges, target_edges = _scheduled_edges(schedule)
    draws = np.full(
        (schedule.shape[0], len(trains), frequencies_hz.size),
        np.nan,
        dtype=float,
    )
    phase_vectors = np.where(valid, phase, 0.0j)
    # These current-unit-block accumulators let each costly edge be sampled
    # once, while schedule rows are still applied in bounded shuffle slices.
    vector_sum = np.zeros(draws.shape, dtype=np.complex128)
    valid_count = np.zeros(draws.shape, dtype=np.int64)
    for edge_start in range(0, source_edges.size, execution.trial_edge_block_size):
        edge_stop = min(edge_start + execution.trial_edge_block_size, source_edges.size)
        edge = spike_lfp_summary.compute_edge_sufficient_statistics(
            trial_relative_spike_times_s=trains,
            phase_time_s=time_s,
            trial_phase_vectors=phase_vectors,
            frequencies_hz=frequencies_hz,
            source_trial_position=source_edges[edge_start:edge_stop],
            target_trial_position=target_edges[edge_start:edge_stop],
        )
        for shuffle_start in range(0, schedule.shape[0], execution.shuffle_block_size):
            shuffle_stop = min(
                shuffle_start + execution.shuffle_block_size, schedule.shape[0]
            )
            schedule_block = schedule[shuffle_start:shuffle_stop]
            for edge_index, (source, target) in enumerate(
                zip(
                    edge.source_trial_position,
                    edge.target_trial_position,
                    strict=True,
                )
            ):
                rows = np.flatnonzero(
                    schedule_block[:, int(source)] == int(target)
                )
                vector_sum[shuffle_start + rows] += edge.phase_vector_sum[edge_index]
                valid_count[shuffle_start + rows] += edge.valid_spike_count[edge_index]
    np.divide(
        np.abs(vector_sum) ** 2 - valid_count,
        valid_count.astype(float) * (valid_count - 1),
        out=draws,
        where=valid_count >= 2,
    )
    return draws


def _observed(
    arrays: dict[str, np.ndarray], unit: int, spikes: tuple[np.ndarray, ...],
    time_s: np.ndarray, phase: np.ndarray, valid: np.ndarray, frequencies_hz: np.ndarray,
    trial_edge_block_size: int | None = None,
) -> None:
    """Compute observed same-trial PPC for one unit on every configured frequency."""
    positions = np.arange(len(spikes), dtype=np.int64)
    if positions.size == 0:
        return
    edge_size = positions.size if trial_edge_block_size is None else trial_edge_block_size
    count = np.zeros(frequencies_hz.size, dtype=np.int64)
    vector = np.zeros(frequencies_hz.size, dtype=np.complex128)
    for start in range(0, positions.size, edge_size):
        stop = min(start + edge_size, positions.size)
        edge = spike_lfp_summary.compute_edge_sufficient_statistics(
            trial_relative_spike_times_s=(spikes,), phase_time_s=time_s,
            trial_phase_vectors=np.where(valid, phase, 0.0j), frequencies_hz=frequencies_hz,
            source_trial_position=positions[start:stop], target_trial_position=positions[start:stop],
        )
        count += edge.valid_spike_count[:, 0].sum(axis=0, dtype=np.int64)
        vector += edge.phase_vector_sum[:, 0].sum(axis=0, dtype=np.complex128)
    ppc = np.full(count.shape, np.nan, dtype=float)
    np.divide(np.abs(vector) ** 2 - count, count.astype(float) * (count - 1), out=ppc, where=count >= 2)
    resultant = np.full(count.shape, np.nan, dtype=float)
    np.divide(np.abs(vector), count, out=resultant, where=count > 0)
    preferred = np.full(count.shape, np.nan, dtype=float)
    preferred[count > 0] = np.angle(vector[count > 0])
    arrays["ppc"][unit] = ppc
    arrays["resultant_length"][unit] = resultant
    arrays["preferred_phase_rad"][unit] = preferred
    arrays["spike_count"][unit] = count
    arrays["computable"][unit] = count >= 2
    arrays["reliable"][unit] = count >= 50
    # Re-evaluate one trial at a time, preserving trial-local overlap membership.
    eligible_trials = np.zeros(frequencies_hz.size, dtype=np.int64)
    for position in positions:
        edge = spike_lfp_summary.compute_edge_sufficient_statistics(
            trial_relative_spike_times_s=(spikes,), phase_time_s=time_s,
            trial_phase_vectors=np.where(valid, phase, 0.0j), frequencies_hz=frequencies_hz,
            source_trial_position=np.array([position]), target_trial_position=np.array([position]),
        )
        eligible_trials += edge.valid_spike_count[0, 0] > 0
    arrays["eligible_trial_count"][unit] = eligible_trials


def _apply_bh_and_significance(arrays: dict[str, np.ndarray], alpha: float) -> None:
    """Apply BH along frequency independently for each unit's eligible spectrum."""
    q_value = spike_lfp_summary.adjust_ppc_pvalues_bh(
        p_value=arrays["p_value"][:, None, None, None, :],
        null_eligible=arrays["null_eligible"][:, None, None, None, :],
    )[:, 0, 0, 0, :]
    arrays["q_value"] = q_value
    arrays["significant"] = arrays["null_eligible"] & (q_value <= alpha)


def _empty_summary_arrays(unit_count: int, frequency_count: int) -> dict[str, np.ndarray]:
    """Allocate documented work-only arrays with axes ``(unit, frequency)``."""
    shape = (unit_count, frequency_count)
    arrays = {name: np.full(shape, np.nan, dtype=float) for name in _FLOAT_FIELDS}
    arrays.update({name: np.zeros(shape, dtype=np.int64) for name in _INTEGER_FIELDS})
    arrays.update({name: np.zeros(shape, dtype=bool) for name in _BOOLEAN_FIELDS})
    return arrays


def _block_ids(unit_count: int, unit_block_size: int) -> tuple[str, ...]:
    """Return stable contiguous unit-block IDs in ascending unit order."""
    return tuple(
        f"unit-{start:06d}-{min(start + unit_block_size, unit_count) - 1:06d}"
        for start in range(0, unit_count, unit_block_size)
    )


def _block_bounds(block_id: str) -> tuple[int, int]:
    """Decode a stable inclusive unit-block ID to half-open unit bounds."""
    _, start, stop = block_id.split("-")
    return int(start), int(stop) + 1


def _summary_block(summary: Mapping[str, np.ndarray], start: int, stop: int) -> dict[str, np.ndarray]:
    """Copy one unit slice for atomic checkpoint serialization."""
    return {name: np.asarray(values[start:stop]).copy() for name, values in summary.items()}


def _copy_block_into_summary(summary: dict[str, np.ndarray], block: Mapping[str, np.ndarray], start: int, stop: int) -> None:
    """Validate and copy one checkpoint's documented unit/frequency arrays."""
    for name in _SUMMARY_FIELDS:
        values = np.asarray(block[name])
        if values.shape != summary[name][start:stop].shape or values.dtype != summary[name].dtype:
            raise ValueError("PPC checkpoint summary axes or dtypes are invalid")
        summary[name][start:stop] = values


def _scheduled_edges(schedule: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return unique lexicographic nonself source-target pairs used by schedule."""
    pairs = sorted({(source, int(target)) for row in schedule for source, target in enumerate(row)})
    return (
        np.asarray([source for source, _ in pairs], dtype=np.int64),
        np.asarray([target for _, target in pairs], dtype=np.int64),
    )


def _stored_schedule_matches(run_directory: Path, schedule: np.ndarray) -> bool:
    """Return whether an existing schedule NPZ exactly matches this job schedule."""
    try:
        with np.load(run_directory / "schedule.npz", allow_pickle=False) as loaded:
            return set(loaded.files) == {"schedule"} and loaded["schedule"].dtype == np.dtype(np.int64) and np.array_equal(loaded["schedule"], schedule)
    except (OSError, ValueError):
        return False


def _load_resumable_blocks(
    run_directory: Path, metadata: Mapping[str, object], schedule_matches: bool,
    unit_count: int, frequency_count: int,
) -> dict[str, dict[str, np.ndarray]]:
    """Load validated unit-block siblings without accepting partial work.

    ``metadata`` is the exact current JSON-safe job identity and
    ``schedule_matches`` confirms the persisted int64 ``(shuffle, trial)``
    schedule before any block is eligible for reuse. Each accepted NPZ has only
    documented ``(unit_block, frequency)`` float/int64/Boolean summary arrays.
    Corrupt, orphaned, markerless, wrong-axis, wrong-dtype, or incompatible
    blocks are ignored independently so valid siblings remain resumable.
    """
    if not schedule_matches:
        return {}
    try:
        stored = json.loads((run_directory / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    expected = dict(metadata)
    expected.pop("completed_block_ids")
    stored.pop("completed_block_ids", None)
    if stored != expected:
        return {}
    valid_blocks: dict[str, dict[str, np.ndarray]] = {}
    for marker_path in sorted((run_directory / "blocks").glob("*.complete.json")):
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            block_id = str(marker["block_id"])
            if marker != {"block_id": block_id, "run_fingerprint": metadata["run_fingerprint"]}:
                continue
            start, stop = _block_bounds(block_id)
            if start < 0 or stop > unit_count:
                continue
            with np.load(run_directory / "blocks" / f"{block_id}.npz", allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in loaded.files}
            if set(arrays) != set(_SUMMARY_FIELDS):
                continue
            template = _empty_summary_arrays(stop - start, frequency_count)
            _copy_block_into_summary(template, arrays, 0, stop - start)
            valid_blocks[block_id] = arrays
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return valid_blocks


def _acquire_executor_lock(
    run_directory: Path,
    run_fingerprint: str,
) -> Path:
    """Create a complete exclusive JSON lock before any exact-job mutation."""
    lock_path = run_directory / "executor.lock"
    _write_lock_exclusive(
        lock_path,
        _ownership_record(run_fingerprint, "executor"),
        f"PPC executor.lock already exists: {lock_path}",
    )
    return lock_path


def _release_executor_lock(lock_path: Path) -> None:
    """Remove only the exact lock created by this invocation."""
    lock_path.unlink(missing_ok=True)


def _write_schedule(run_directory: Path, schedule: np.ndarray) -> None:
    """Atomically write the validated int64 schedule while executor.lock exists."""
    temporary = run_directory / ".schedule.npz.tmp"
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, schedule=schedule)
        with np.load(temporary, allow_pickle=False) as loaded:
            if not np.array_equal(loaded["schedule"], schedule):
                raise ValueError("temporary PPC schedule validation failed")
        os.replace(temporary, run_directory / "schedule.npz")
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a small JSON completion marker in the same directory."""
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_fingerprint(*arrays: np.ndarray) -> str:
    """Hash exact dtype/shape/C-order content without allocating full copies.

    Parameters
    ----------
    *arrays : numpy.ndarray
        Production numeric or Boolean arrays with fixed non-object dtypes.
        Their dtype, shape, and logical C-order bytes define one work-input
        identity. Axes and units are caller-defined.

    Returns
    -------
    str
        SHA-256 hexadecimal digest over supplied arrays in order. Noncontiguous
        inputs hash exactly as their logical C-order copies would hash.

    Notes
    -----
    NumPy 2.4's buffered ``nditer`` with ``order="C"`` exposes bounded
    C-order chunks for contiguous and noncontiguous arrays. Each temporary byte
    payload here is at most 64 KiB; this helper never materializes a whole-array
    contiguous copy or one whole-array byte string.
    """
    chunk_bytes = 64 * 1024
    digest = sha256()
    for array in arrays:
        value = np.asarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        if value.size == 0:
            continue
        chunk_elements = max(1, chunk_bytes // max(1, value.dtype.itemsize))
        iterator = np.nditer(
            value,
            flags=("external_loop", "buffered", "zerosize_ok", "refs_ok"),
            op_flags=("readonly",),
            order="C",
            buffersize=chunk_elements,
        )
        for chunk in iterator:
            digest.update(chunk.tobytes(order="C"))
    return digest.hexdigest()


def _hash_spikes(trains: object, overlaps: object) -> str:
    """Hash trial-local spike seconds and overlap identities without object arrays."""
    digest = sha256()
    for unit in trains:
        for trial in unit:
            digest.update(_array_fingerprint(np.asarray(trial, dtype=float)).encode("ascii"))
    digest.update(_hash_json(overlaps).encode("ascii"))
    return digest.hexdigest()


def _hash_json(value: object) -> str:
    """Return a SHA-256 digest for one JSON-safe canonical identity value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class _ProgressReporter:
    """Emit ordered job-scoped progress records without numerical payloads.

    ``job_id`` identifies one selected site/condition/epoch work item.
    ``emit`` carries nonnegative completed/total work units, elapsed seconds,
    and an ETA only after two timed updates in the same stage. It never changes
    phase, spike, unit, frequency, or missingness data.
    """

    callback: Callable[[ProgressEvent], None] | None
    job_id: str
    start_time: float = 0.0
    timed_by_stage: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self.start_time = time.perf_counter()
        self.timed_by_stage = {}

    def emit(self, stage: str, completed: int, total: int, message: str, *, timed: bool = False) -> None:
        """Forward one monotonic event without raw phase/spike data."""
        assert self.timed_by_stage is not None
        if timed:
            self.timed_by_stage[stage] = self.timed_by_stage.get(stage, 0) + 1
        elapsed = max(0.0, time.perf_counter() - self.start_time)
        eta = None
        if self.timed_by_stage.get(stage, 0) >= 2:
            eta = max(0.0, elapsed / completed * (total - completed))
        if self.callback is not None:
            self.callback(ProgressEvent("spike_phase", stage, completed, total, message, elapsed, eta, self.job_id))


generate_trial_derangement_schedule = spike_lfp_summary.generate_trial_derangement_schedule
# Test sentinel only: production execution must never call this full-draw helper.
_compute_scheduled_shuffle_draws = spike_lfp_summary._compute_scheduled_shuffle_draws
