"""Serial, restartable sufficient-statistic PPC execution for one job.

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

import numpy as np

from src.neural_analysis import spike_lfp_summary
from src.neural_analysis.lfp_summary_ppc_kernel import (
    estimate_segmented_kernel_allocation,
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
    planner_array_bytes, summary_assembly_bytes : int
        Parent-owned retained plan and full-component publication arrays.
    worker_plan_bytes, worker_summary_bytes : int
        Worker-owned full-site plan and current site/unit-block result arrays.
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
    planner_array_bytes: int
    summary_assembly_bytes: int
    worker_plan_bytes: int
    worker_summary_bytes: int
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
    if (
        positions.dtype != np.dtype(np.int64)
        or positions.ndim != 1
        or np.any(positions < 0)
        or np.any(positions >= source_count)
        or (unique and np.unique(positions).size != positions.size)
    ):
        raise ValueError(f"{name} must be valid int64 source positions")
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
        or np.any(counts < 0)
    ):
        raise ValueError("source_trial_spike_count must be nonnegative int64 (source, unit_block, 2)")
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
    if requested_workers == 1:
        active_workers = 0
        parent_private = _checked_ppc_add(
            planned_arrays, summary_assembly_bytes, computation_private
        )
    else:
        active_workers = min(requested_workers, pending_blocks)
        parent_private = _checked_ppc_add(planned_arrays, summary_assembly_bytes)
    worker_private = _checked_ppc_add(
        worker_plan, worker_summary_bytes, computation_private
    )
    aggregate = _checked_ppc_add(
        shared_mmap,
        parent_private,
        _checked_ppc_multiply(active_workers, worker_private),
    )
    return PPCAllocationEstimate(
        job_accumulator_bytes=job_accumulator_bytes,
        observed_trial_statistics_bytes=observed_trial_statistics_bytes,
        observed_gather_temporary_bytes=observed_gather_temporary_bytes,
        kernel_working_bytes=kernel_working_bytes,
        geometry_bytes=geometry_bytes,
        planner_array_bytes=planned_arrays,
        summary_assembly_bytes=summary_assembly_bytes,
        worker_plan_bytes=worker_plan,
        worker_summary_bytes=worker_summary_bytes,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
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
        arrays; an owned object ``(condition,)`` name array; and canonical
        Python frequency count and mmap bytes.

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
        or np.any(stable_rows < 0)
        or np.unique(stable_rows).size != stable_rows.size
    ):
        raise ValueError("stable_trial_rows must be unique int64 trial identities")
    if (
        source_counts.dtype != np.dtype(np.int64)
        or source_counts.ndim != 3
        or source_counts.shape[0] != membership.shape[0]
        or source_counts.shape[1] < 1
        or source_counts.shape[2] != 2
        or np.any(source_counts < 0)
    ):
        raise ValueError("source_trial_spike_count must be nonnegative int64 (trial, unit, 2)")
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
        np.asarray(tuple(condition_names), dtype=object),
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
        for condition_index, condition_name in enumerate(names.tolist()):
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
    pending_blocks = (total_units + unit_block_size - 1) // unit_block_size
    component_job_count = len(all_drafts)
    representative_band_count = len(config.phase.bands)
    phase_bin_count = len(config.ppc.phase_bin_edges_rad) - 1
    summary_cell_bytes = _checked_ppc_add(
        _checked_ppc_multiply(116, frequencies),
        _checked_ppc_multiply(8, representative_band_count, phase_bin_count),
    )
    parent_parallel_bytes = _checked_ppc_add(
        planner_array_bytes,
        _checked_ppc_multiply(total_units, component_job_count, summary_cell_bytes),
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
            batch_edge_positions = np.asarray(
                [source_row_to_position[source] for _, source, _ in batch_union],
                dtype=np.int64,
            )
        else:
            batch_edge_positions = np.empty(0, dtype=np.int64)
        estimates: list[PPCAllocationEstimate] = []
        for unit_start in range(0, total_units, unit_block_size):
            unit_stop = min(unit_start + unit_block_size, total_units)
            if source_positions.size:
                block_counts = source_counts[
                    source_positions, unit_start:unit_stop, :
                ].copy()
                observed_positions = np.arange(source_positions.size, dtype=np.int64)
            else:
                # The allocator requires one source-axis row; no observed/null
                # positions make this a harmless zero-work placeholder.
                block_counts = np.zeros((1, unit_stop - unit_start, 2), dtype=np.int64)
                observed_positions = np.empty(0, dtype=np.int64)
            edge_blocks = (
                tuple(
                    batch_edge_positions[start : start + config.ppc_execution.trial_edge_block_size]
                    for start in range(
                        0,
                        batch_edge_positions.size,
                        config.ppc_execution.trial_edge_block_size,
                    )
                )
                if batch_edge_positions.size
                else (np.empty(0, dtype=np.int64),)
            )
            for edge_block in edge_blocks:
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


@dataclass(frozen=True)
class _PhaseWorkDescriptor:
    """Read-only on-disk phase inputs shared by process workers.

    ``phase_path`` and ``valid_path`` name validated NPY arrays with axes
    ``(site=1, frequency, trial, time)``. Phase is complex64 unit vectors;
    validity is Boolean. Time remains in seconds in the lightweight task
    because it is a one-dimensional coordinate rather than the large payload.
    """

    phase_path: Path
    valid_path: Path


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
    """Hash dtype, shape, and bytes for exact numerical work-input identity."""
    digest = sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(value.shape).encode("ascii"))
        digest.update(value.tobytes())
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
