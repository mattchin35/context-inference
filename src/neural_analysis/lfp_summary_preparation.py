"""Shared, validated preparation for offline LFP summary analyses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from math import isfinite
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp_summary_models import LFPSiteConfig, ProgressEvent, TrialFilterConfig
from src.neural_analysis.spike_behavior.trials import make_trial_type_masks


CONDITION_NAMES = (
    "correct_rewarded",
    "omission",
    "incorrect",
    "switch",
    "stay",
    "omission_switch",
    "omission_stay",
    "incorrect_switch",
    "incorrect_stay",
)


@dataclass(frozen=True)
class PreparedTrials:
    """Prepared behavioral and site-validity masks.

    Attributes
    ----------
    condition_names : tuple[str, ...]
        Ordered condition axis labels.
    condition_membership : numpy.ndarray
        Boolean shape ``(trial, condition)``. Conditions intentionally overlap.
    filter_membership, user_excluded, objective_valid : numpy.ndarray
        Boolean shape ``(trial,)`` masks.
    objective_exclusion_reason, user_exclusion_reason : numpy.ndarray
        Fixed-Unicode shape ``(trial,)`` reason strings; empty string means none.
    site_validity : dict[str, numpy.ndarray]
        Stable-site-id keyed boolean ``(trial,)`` masks.
    pair_validity : dict[tuple[str, str], numpy.ndarray]
        Ordered-pair keyed boolean ``(trial,)`` intersections only.
    """

    condition_names: tuple[str, ...]
    condition_membership: np.ndarray
    filter_membership: np.ndarray
    user_excluded: np.ndarray
    objective_valid: np.ndarray
    objective_exclusion_reason: np.ndarray
    user_exclusion_reason: np.ndarray
    site_validity: dict[str, np.ndarray]
    pair_validity: dict[tuple[str, str], np.ndarray]


@dataclass(frozen=True)
class PreparedSiteTraces:
    """Canonical native-rate traces and per-trial quality control.

    Attributes
    ----------
    source_trace : numpy.ndarray
        Float64 shape ``(trial, time)`` in ``voltage_unit``. Invalid rows are NaN.
    relative_time_s : numpy.ndarray
        Float64 shape ``(time,)`` exact half-open event-relative seconds grid.
    valid : numpy.ndarray
        Boolean shape ``(trial,)`` objective trace-validity mask.
    exclusion_reason : numpy.ndarray
        Fixed-Unicode shape ``(trial,)``; empty string for valid traces.
    rms, peak_to_peak : numpy.ndarray
        Float64 shape ``(trial,)`` in ``voltage_unit``; invalid rows are NaN.
    voltage_unit : str
        Source voltage physical unit.
    sample_rate_hz : float
        Effective native LFP sample rate in samples/second.
    """

    source_trace: np.ndarray
    relative_time_s: np.ndarray
    valid: np.ndarray
    exclusion_reason: np.ndarray
    rms: np.ndarray
    peak_to_peak: np.ndarray
    voltage_unit: str
    sample_rate_hz: float


@dataclass(frozen=True)
class TrialRelativeSpikeTrains:
    """Trial-local spike trains for one stable unit.

    Attributes
    ----------
    unit_id : str
        Stable ``"probe_label:cluster_id"`` identifier.
    relative_spike_times : tuple[numpy.ndarray, ...]
        One float64 ``(spike,)`` seconds array per trial; empty arrays mean no spikes.
    overlap_trial_indices : numpy.ndarray
        Int64 ``(overlapping_trial,)`` indices. Empty means no overlapping windows.
    """

    unit_id: str
    relative_spike_times: tuple[np.ndarray, ...]
    overlap_trial_indices: np.ndarray


# User-facing preparation functions.
def build_prepared_trials(
    trial_df: pd.DataFrame,
    trial_filter: TrialFilterConfig,
    alignment_event: str,
    site_validity: dict[str, Sequence[bool]] | None = None,
) -> PreparedTrials:
    """Build trial masks using the project's authoritative condition definitions.

    Parameters
    ----------
    trial_df : pandas.DataFrame
        One row per trial. ``choice_time`` or ``start_time`` is seconds; action
        and state values use 0=right and 1=left.
    trial_filter : TrialFilterConfig
        Choice/context selections plus integer user-excluded row positions.
    alignment_event : str
        Supported ``"choice_time"`` or ``"start_time"`` column.
    site_validity : dict[str, Sequence[bool]] | None
        Optional site masks, each with shape ``(trial,)``.

    Returns
    -------
    PreparedTrials
        Stable trial-axis masks. Missing alignment is objective-invalid; user
        exclusions remain separate; condition overlap is preserved.

    Raises
    ------
    ValueError
        If alignment, filters, exclusion indices, or site mask axes are invalid.
    """
    if alignment_event not in {"choice_time", "start_time"}:
        raise ValueError("unsupported alignment event")
    if alignment_event not in trial_df:
        raise ValueError("trial table lacks alignment column")

    trial_count = len(trial_df)
    _validate_excluded_indices(trial_filter.excluded_trial_indices, trial_count)
    condition_membership = _condition_membership(trial_df)
    filter_membership = _filter_membership(trial_df, trial_filter)
    objective_valid = np.isfinite(
        pd.to_numeric(trial_df[alignment_event], errors="coerce").to_numpy(dtype=float)
    )
    user_excluded = np.zeros(trial_count, dtype=bool)
    user_excluded[list(trial_filter.excluded_trial_indices)] = True
    site_masks = _site_masks(site_validity, trial_count)

    return PreparedTrials(
        condition_names=CONDITION_NAMES,
        condition_membership=condition_membership,
        filter_membership=filter_membership,
        user_excluded=user_excluded,
        objective_valid=objective_valid,
        objective_exclusion_reason=np.where(
            objective_valid, "", f"missing_{alignment_event}"
        ).astype("<U32"),
        user_exclusion_reason=np.where(user_excluded, "user_excluded", "").astype("<U32"),
        site_validity=site_masks,
        pair_validity=_pair_masks(site_masks),
    )


def build_common_event_grid(
    start_s: float,
    stop_s: float,
    sample_rate_hz: float,
) -> np.ndarray:
    """Construct an exact half-open event-relative time grid.

    Parameters
    ----------
    start_s, stop_s : float
        Finite seconds bounds for ``[start_s, stop_s)``.
    sample_rate_hz : float
        Finite positive samples/second. Duration times rate must be integral.

    Returns
    -------
    numpy.ndarray
        Float64 shape ``(time,)`` grid in seconds. It contains no endpoint at stop.

    Raises
    ------
    ValueError
        If either bound or the rate is nonfinite, the half-open interval is
        empty or reversed, or its duration does not contain an integral number
        of samples within ``1e-9`` samples. No invalid grid is returned.
    """
    values = (start_s, stop_s, sample_rate_hz)
    if (
        not all(isfinite(value) for value in values)
        or stop_s <= start_s
        or sample_rate_hz <= 0
    ):
        raise ValueError("invalid event grid")
    sample_count_float = (stop_s - start_s) * sample_rate_hz
    sample_count = round(sample_count_float)
    if not np.isclose(sample_count_float, sample_count, rtol=0.0, atol=1e-9):
        raise ValueError("duration must contain integral samples")
    return float(start_s) + np.arange(sample_count, dtype=float) / float(sample_rate_hz)


def interpolate_complex_coefficients_to_common_grid(
    source_time_s: np.ndarray,
    coefficients: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    """Interpolate complex coefficients by real/imaginary parts and unit-normalize.

    Parameters
    ----------
    source_time_s : numpy.ndarray
        Nonempty, finite, strictly increasing float64 ``(source_time,)``
        coordinate in seconds.
    target_time_s : numpy.ndarray
        Finite, strictly increasing float64 ``(target_time,)`` coordinate in
        seconds. It may be empty, yielding an empty final time axis.
    coefficients : numpy.ndarray
        Numeric array with final axis ``(source_time,)``; leading axes are preserved.

    Returns
    -------
    numpy.ndarray
        Complex shape ``coefficients.shape[:-1] + (target_time,)`` unit phases.
        Values outside source support or with zero magnitude are complex NaN.

    Raises
    ------
    ValueError
        If a coordinate has the wrong axis, is nonfinite or nonmonotonic, the
        source coordinate is empty, or coefficients are nonnumeric or have an
        incompatible final source-time axis. Invalid interpolation locations
        are represented by complex NaN rather than raising.
    """
    source = _time_coordinate(source_time_s, "source", allow_empty=False)
    target = _time_coordinate(target_time_s, "target", allow_empty=True)
    complex_values = np.asarray(coefficients)
    if complex_values.ndim < 1 or complex_values.shape[-1] != source.size:
        raise ValueError("coefficient time axis does not match source time")
    if not np.issubdtype(complex_values.dtype, np.number):
        raise ValueError("coefficients must be numeric")

    flattened = complex_values.reshape((-1, source.size))
    interpolated = np.empty((flattened.shape[0], target.size), dtype=complex)
    for row_index, row in enumerate(flattened):
        real = np.interp(target, source, row.real, left=np.nan, right=np.nan)
        imaginary = np.interp(target, source, row.imag, left=np.nan, right=np.nan)
        combined = real + 1j * imaginary
        magnitude = np.abs(combined)
        interpolated[row_index] = np.divide(
            combined,
            magnitude,
            out=np.full(target.size, np.nan + 1j * np.nan),
            where=magnitude > 0,
        )
    return interpolated.reshape(complex_values.shape[:-1] + (target.size,))


def prepare_site_trial_traces(
    site: LFPSiteConfig,
    trial_indices: np.ndarray,
    alignment_times_s: np.ndarray,
    window: tuple[float, float],
    trace_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]],
) -> PreparedSiteTraces:
    """Load canonical native LFP traces and compute per-trial QC.

    Parameters
    ----------
    site : LFPSiteConfig
        One saved channel with source voltage unit and effective sample rate.
    trial_indices : numpy.ndarray
        Integer-like shape ``(trial,)`` row positions.
    alignment_times_s : numpy.ndarray
        Float shape ``(trial,)`` absolute seconds. NaN is ``missing_alignment``.
    window : tuple[float, float]
        Half-open event-relative seconds interval.
    trace_loader : callable
        Normalized seam called positionally as ``(site, alignment_s, window)``;
        returns ``(time_s, values, sample_rate_hz)`` with one-dimensional arrays.

    Returns
    -------
    PreparedSiteTraces
        Canonical `(trial,time)` float arrays. Invalid rows/QC are NaN with reasons.

    Raises
    ------
    ValueError
        If trial indices or alignment times are not matching one-dimensional
        trial axes, or the seconds window or native Hz rate cannot define a
        valid half-open grid. Missing alignments and invalid loader results
        instead remain NaN rows with explicit reason codes.
    """
    indices = _trial_indices(trial_indices)
    alignments = np.asarray(alignment_times_s, dtype=float)
    if alignments.ndim != 1 or alignments.size != indices.size:
        raise ValueError("alignment times must match trial axis")
    canonical_grid = build_common_event_grid(window[0], window[1], site.sample_rate_hz)
    traces, valid, reasons, rms, peak_to_peak = _empty_trace_outputs(
        indices.size,
        canonical_grid.size,
    )

    for trial_position, alignment_time_s in enumerate(alignments):
        if not np.isfinite(alignment_time_s):
            reasons[trial_position] = "missing_alignment"
            continue
        _load_and_validate_trial(
            trace_loader,
            site,
            float(alignment_time_s),
            window,
            canonical_grid,
            traces,
            valid,
            reasons,
            rms,
            peak_to_peak,
            trial_position,
        )
    return PreparedSiteTraces(
        source_trace=traces,
        relative_time_s=canonical_grid,
        valid=valid,
        exclusion_reason=reasons,
        rms=rms,
        peak_to_peak=peak_to_peak,
        voltage_unit=site.voltage_unit,
        sample_rate_hz=float(site.sample_rate_hz),
    )


def load_site_trial_traces(
    sites: Sequence[LFPSiteConfig],
    trial_indices: np.ndarray,
    alignment_times_s: np.ndarray,
    window: tuple[float, float],
    *,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_metadata_by_site: Mapping[str, dict[str, object]] | None = None,
) -> dict[str, PreparedSiteTraces]:
    """Prepare multiple sites via normalized injected seams or production adapters.

    Injected loaders receive exactly keyword arguments ``site``,
    ``alignment_time_s``, and ``window`` and return `(time_s, values, rate_hz)`.
    Production adapters call existing loader APIs with their real signatures.

    Parameters
    ----------
    sites : Sequence[LFPSiteConfig]
        Unique sites with explicit source voltage units and native sample rates.
    trial_indices : numpy.ndarray
        Integer `(trial,)` table-row positions.
    alignment_times_s : numpy.ndarray
        Absolute seconds `(trial,)`; nonfinite values produce invalid trace rows.
    window : tuple[float, float]
        Half-open event-relative seconds bounds.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized integration seams. Each callable receives exactly
        ``site: LFPSiteConfig``, ``alignment_time_s: float`` in absolute
        seconds, and ``window: tuple[float, float]`` in relative seconds as
        keyword arguments. It returns ``(time_s, values, sample_rate_hz)``:
        matching one-dimensional native seconds and voltage arrays plus a
        finite positive native rate in Hz. ``None`` selects the corresponding
        production adapter; invalid rows remain NaN with reason codes.
    open_ephys_metadata_by_site : mapping[str, dict[str, object]] or None
        Optional normalized production metadata keyed by stable site id. It is
        used only when ``open_ephys_loader`` is ``None`` and is validated
        before any trial numerical loading. Supplying an injected Open Ephys
        loader leaves that seam independent of production metadata I/O.

    Returns
    -------
    dict[str, PreparedSiteTraces]
        Stable-site keyed native `(trial,time)` results; invalid rows are NaN.

    Raises
    ------
    ValueError
        If site identities/formats or preparation inputs violate their contracts.
    """
    _validate_sites(sites)
    normalized_metadata = (
        preflight_open_ephys_site_metadata(sites, open_ephys_metadata_by_site)
        if open_ephys_loader is None
        else {}
    )
    # Construct all production adapters before any site enters its trial loop.
    # This makes a later invalid Open Ephys site fail before another site's
    # numerical reads can begin.
    site_adapters = [
        (
            site,
            *_site_loader(
                site,
                spikeglx_loader,
                open_ephys_loader,
                normalized_metadata.get(site.stable_id),
            ),
        )
        for site in sites
    ]
    prepared: dict[str, PreparedSiteTraces] = {}
    for site, loader, effective_site in site_adapters:
        prepared[site.stable_id] = prepare_site_trial_traces(
            effective_site,
            trial_indices,
            alignment_times_s,
            window,
            loader,
        )
    return prepared


def preflight_open_ephys_site_metadata(
    sites: Sequence[LFPSiteConfig],
    metadata_by_site: Mapping[str, dict[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Structurally preflight, then load/reuse every production Open Ephys sidecar.

    Parameters
    ----------
    sites : Sequence[LFPSiteConfig]
        Configured sites with unique stable ids. Only entries with
        ``acquisition_format == "open_ephys"`` are inspected. Their paths
        have no numerical axes; configured rates are Hz and voltage units are
        categorical physical-unit labels.
    metadata_by_site : mapping[str, dict[str, object]] or None, optional
        Already normalized sidecars keyed by Open Ephys stable id. A supplied
        entry is reused without reparsing its JSON file, but it is still
        checked against the configured saved-channel index, rate, and unit.

    Returns
    -------
    dict[str, dict[str, object]]
        The exact normalized metadata mappings keyed by each Open Ephys stable
        id. First, every selected aligned-sync path is checked as an existing
        regular file before any sidecar is parsed. The mapping contains no
        trace samples and does not parse synchronization data or perform
        binary-trace, cache, filter, interpolation, or transform I/O.

    Raises
    ------
    ValueError
        If a configured Open Ephys aligned-sync path is absent or not a regular
        file, a site lacks normalized metadata, or its saved channel,
        configured rate, or configured voltage unit is invalid. Structural
        failures are raised before any sidecar metadata parsing.
    OSError
        If an absent metadata entry cannot be read from its production sidecar.
    """
    open_ephys_sites = tuple(
        site for site in sites if site.acquisition_format == "open_ephys"
    )
    # Check every source path first so a later unusable OE site cannot permit
    # earlier metadata, cache, sync, or numerical work to begin.
    for site in open_ephys_sites:
        validate_open_ephys_aligned_sync_path(site)

    preloaded = metadata_by_site or {}
    normalized: dict[str, dict[str, object]] = {}
    for site in open_ephys_sites:
        if site.stable_id in preloaded:
            metadata = preloaded[site.stable_id]
            if not isinstance(metadata, dict):
                raise ValueError(f"Open Ephys site {site.stable_id}: normalized metadata is invalid")
        else:
            metadata = lfp_loading.load_open_ephys_lfp_metadata(site.lfp_path)
        lfp_loading.validate_open_ephys_site_metadata(
            site.stable_id,
            site.saved_channel_index,
            site.sample_rate_hz,
            site.voltage_unit,
            metadata,
        )
        normalized[site.stable_id] = metadata
    return normalized


def validate_open_ephys_aligned_sync_path(site: LFPSiteConfig) -> None:
    """Require one production Open Ephys site to reference a regular sync file.

    Parameters
    ----------
    site : LFPSiteConfig
        One configured derived Open Ephys source. ``aligned_sync_path`` has no
        numerical axis or physical unit and must identify the already-created
        aligned synchronization file; no NPZ contents are parsed.

    Returns
    -------
    None
        Returns after a non-mutating ``Path.is_file`` structural check. It does
        not read metadata, synchronization samples, binary traces, caches, or
        numerical arrays.

    Raises
    ------
    ValueError
        If ``aligned_sync_path`` is absent or is not an existing regular file.
        The message identifies the configured stable site and field name.
    """
    sync_path = site.aligned_sync_path
    if sync_path is None or not sync_path.is_file():
        raise ValueError(
            f"Open Ephys site {site.stable_id}: aligned_sync_path must be an "
            "existing regular file"
        )


def build_trial_relative_spike_trains(
    *,
    probe_label: str,
    cluster_id: int,
    unit_spike_times_s: np.ndarray,
    event_times_s: np.ndarray,
    window: tuple[float, float],
) -> TrialRelativeSpikeTrains:
    """Assign each spike to every containing half-open trial window.

    Parameters
    ----------
    probe_label : str
        Nonempty probe identity used with ``cluster_id`` to form the stable
        ``"probe:cluster"`` unit identifier.
    cluster_id : int
        Integer cluster identity; its units are categorical, not time.
    unit_spike_times_s : numpy.ndarray
        Finite float64 ``(spike,)`` absolute spike times in seconds.
    event_times_s : numpy.ndarray
        Finite float64 ``(trial,)`` absolute alignment-event times in seconds.
    window : tuple[float, float]
        Finite event-relative seconds bounds defining ``[start_s, stop_s)``.

    Returns
    -------
    TrialRelativeSpikeTrains
        Stable-unit metadata and one owned float64 ``(spike_in_trial,)`` array
        in event-relative seconds per trial. Empty arrays represent no spikes;
        spikes in overlapping absolute half-open windows occur in every member.

    Raises
    ------
    ValueError
        If the probe or cluster identity is missing, either time vector is not
        finite one-dimensional seconds data, or the interval is nonfinite,
        empty, or reversed. No partially prepared result is returned.
    """
    if not probe_label or not isinstance(cluster_id, (int, np.integer)):
        raise ValueError("probe label and cluster id are required")
    spikes = _finite_vector(unit_spike_times_s, "spike times")
    events = _finite_vector(event_times_s, "event times")
    start_s, stop_s = window
    if not all(isfinite(value) for value in window) or stop_s <= start_s:
        raise ValueError("invalid spike window")
    relative_spikes: list[np.ndarray] = []
    for event_time_s in events:
        in_window = (spikes >= event_time_s + start_s) & (spikes < event_time_s + stop_s)
        relative_spikes.append(np.asarray(spikes[in_window] - event_time_s, dtype=float))
    intervals = [(event + start_s, event + stop_s) for event in events]
    overlap_indices = _overlap_indices(intervals)
    return TrialRelativeSpikeTrains(
        f"{probe_label}:{int(cluster_id)}",
        tuple(relative_spikes),
        np.asarray(overlap_indices, dtype=int),
    )


def validate_progress_events(events: Sequence[ProgressEvent]) -> None:
    """Validate monotonic framework-independent progress events.

    Parameters
    ----------
    events : Sequence[ProgressEvent]
        Ordered framework-independent records. ``completed_count`` and optional
        ``total_count`` are nonnegative item counts; monotonicity is evaluated
        independently for each ``(component, stage)`` categorical key.

    Returns
    -------
    None
        Returns ``None`` after confirming all records. It does not mutate the
        event sequence and has no missing-value sentinel.

    Raises
    ------
    ValueError
        If a count is negative, completion exceeds a supplied total, completion
        decreases for one key, or that key changes its supplied total.
    """
    previous: dict[tuple[str, str], tuple[int, int | None]] = {}
    for event in events:
        if event.completed_count < 0:
            raise ValueError("progress completed count must be nonnegative")
        total_is_invalid = event.total_count is not None and (
            event.total_count < 0 or event.completed_count > event.total_count
        )
        if total_is_invalid:
            raise ValueError("invalid progress total")
        key = (event.component, event.stage)
        prior = previous.get(key)
        is_decreasing = prior is not None and event.completed_count < prior[0]
        has_changed_total = prior is not None and event.total_count != prior[1]
        if is_decreasing or has_changed_total:
            raise ValueError("progress must be monotonic with consistent total")
        previous[key] = (event.completed_count, event.total_count)


# Private validation and adapter helpers.
def _condition_membership(trial_df: pd.DataFrame) -> np.ndarray:
    """Return authoritative boolean `(trial, condition)` membership without changing overlap."""
    masks = make_trial_type_masks(trial_df)
    return np.column_stack([masks[name].to_numpy(dtype=bool) for name in CONDITION_NAMES])


def _filter_membership(trial_df: pd.DataFrame, trial_filter: TrialFilterConfig) -> np.ndarray:
    """Return boolean `(trial,)` independent choice/context filter membership."""
    choice_membership = _one_filter(trial_df, "action", trial_filter.choice)
    context_membership = _one_filter(
        trial_df,
        "state_int",
        trial_filter.context,
    )
    return choice_membership & context_membership


def _one_filter(trial_df: pd.DataFrame, column: str, selected: str) -> np.ndarray:
    """Return one categorical `(trial,)` filter; unknown/missing values fail left/right."""
    if selected == "all":
        return np.ones(len(trial_df), dtype=bool)
    if selected not in {"left", "right"}:
        raise ValueError("unsupported trial filter")
    desired = 1 if selected == "left" else 0
    values = trial_df.get(column, pd.Series(np.nan, index=trial_df.index))
    return values.eq(desired).to_numpy(dtype=bool)


def _validate_excluded_indices(indices: Sequence[int], trial_count: int) -> None:
    """Validate user exclusion row positions without changing trial order."""
    has_invalid_index = any(
        not isinstance(index, (int, np.integer))
        or index < 0
        or index >= trial_count
        for index in indices
    )
    if has_invalid_index:
        raise ValueError("excluded trial index is out of range")


def _site_masks(
    site_validity: dict[str, Sequence[bool]] | None,
    trial_count: int,
) -> dict[str, np.ndarray]:
    """Copy site validity masks as owned boolean `(trial,)` arrays."""
    output: dict[str, np.ndarray] = {}
    for site_id, values in (site_validity or {}).items():
        mask = np.asarray(values, dtype=bool)
        if not site_id or mask.ndim != 1 or mask.size != trial_count:
            raise ValueError("site validity must have a stable id and trial axis")
        output[site_id] = mask.copy()
    return output


def _pair_masks(site_masks: dict[str, np.ndarray]) -> dict[tuple[str, str], np.ndarray]:
    """Return pair-specific `(trial,)` intersections without global over-exclusion."""
    pair_masks: dict[tuple[str, str], np.ndarray] = {}
    for left_site_id, right_site_id in combinations(site_masks, 2):
        pair_masks[(left_site_id, right_site_id)] = (
            site_masks[left_site_id] & site_masks[right_site_id]
        )
    return pair_masks


def _time_coordinate(values: np.ndarray, name: str, allow_empty: bool) -> np.ndarray:
    """Validate a finite strictly increasing float seconds `(time,)` coordinate."""
    coordinate = np.asarray(values, dtype=float)
    is_empty = coordinate.size == 0
    is_invalid = (
        coordinate.ndim != 1
        or (is_empty and not allow_empty)
        or not np.isfinite(coordinate).all()
        or np.any(np.diff(coordinate) <= 0)
    )
    if is_invalid:
        raise ValueError(f"invalid {name} time coordinate")
    return coordinate


def _overlap_indices(intervals: list[tuple[float, float]]) -> list[int]:
    """Return indices whose half-open absolute intervals overlap another interval."""
    result: list[int] = []
    for index, (left, right) in enumerate(intervals):
        for other_index, (other_left, other_right) in enumerate(intervals):
            if index != other_index and left < other_right and other_left < right:
                result.append(index)
                break
    return result


def _trial_indices(values: np.ndarray) -> np.ndarray:
    """Validate integer-like one-dimensional trial row positions; duplicates are allowed."""
    indices = np.asarray(values)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("trial indices must be one-dimensional integers")
    return indices.astype(np.int64, copy=True)


def _empty_trace_outputs(
    trial_count: int,
    time_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocate owned NaN trace/QC and false/reason `(trial,)` outputs."""
    traces = np.full((trial_count, time_count), np.nan)
    valid = np.zeros(trial_count, dtype=bool)
    reasons = np.full(trial_count, "", dtype="<U32")
    rms = np.full(trial_count, np.nan)
    peak_to_peak = np.full(trial_count, np.nan)
    return traces, valid, reasons, rms, peak_to_peak


def _load_and_validate_trial(
    loader: Callable[..., tuple[np.ndarray, np.ndarray, float]],
    site: LFPSiteConfig,
    alignment_time_s: float,
    window: tuple[float, float],
    grid: np.ndarray,
    traces: np.ndarray,
    valid: np.ndarray,
    reasons: np.ndarray,
    rms: np.ndarray,
    peak: np.ndarray,
    row: int,
) -> None:
    """Load one trial, recording only expected I/O/data failures as a reason code."""
    try:
        time_s, values, rate_hz = loader(site, alignment_time_s, window)
    except (OSError, ValueError):
        reasons[row] = "loader_error"
        return
    try:
        time = np.asarray(time_s, dtype=float)
        trace = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        reasons[row] = "incomplete_window"
        return
    if not _matches_grid(time, trace, rate_hz, grid, site.sample_rate_hz):
        reasons[row] = "incomplete_window"
        return
    if not np.isfinite(trace).all():
        reasons[row] = "nonfinite_trace"
        return
    if np.ptp(trace) == 0:
        reasons[row] = "constant_trace"
        return
    traces[row] = trace
    valid[row] = True
    rms[row] = np.sqrt(np.mean(trace ** 2))
    peak[row] = np.ptp(trace)


def _matches_grid(
    time: np.ndarray,
    values: np.ndarray,
    rate_hz: float,
    grid: np.ndarray,
    expected_rate_hz: float,
) -> bool:
    """Return whether loader arrays are finite one-dimensional canonical native samples."""
    correct_axes = time.ndim == 1 and values.ndim == 1
    correct_length = time.size == grid.size and values.size == grid.size
    valid_rate = isinstance(rate_hz, (int, float, np.number))
    valid_rate = valid_rate and isfinite(float(rate_hz)) and float(rate_hz) > 0
    return bool(
        correct_axes
        and correct_length
        and np.isfinite(time).all()
        and valid_rate
        and np.isclose(float(rate_hz), expected_rate_hz)
        and np.allclose(time, grid, rtol=0.0, atol=1e-9)
    )


def _validate_sites(sites: Sequence[LFPSiteConfig]) -> None:
    """Validate nonempty, unique stable site identifiers and supported acquisition formats."""
    site_ids = [site.stable_id for site in sites]
    if any(not site_id for site_id in site_ids) or len(site_ids) != len(set(site_ids)):
        raise ValueError("site ids must be nonempty and unique")
    if any(site.acquisition_format not in {"spikeglx", "open_ephys"} for site in sites):
        raise ValueError("unsupported acquisition format")


def _site_loader(
    site: LFPSiteConfig,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None,
    normalized_open_ephys_metadata: dict[str, object] | None = None,
) -> tuple[Callable[..., tuple[np.ndarray, np.ndarray, float]], LFPSiteConfig]:
    """Return one normalized loader seam and effective-rate site configuration.

    Parameters
    ----------
    site : LFPSiteConfig
        One configured saved channel with categorical source units and a
        native sampling rate in Hz.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional injected ``(site, alignment_time_s, window)`` trace seams.
        Their time arrays are one-dimensional seconds and values retain source
        voltage units. ``None`` selects production loading.
    normalized_open_ephys_metadata : dict[str, object] or None, optional
        Preflighted normalized Open Ephys sidecar for this site. It has no
        sample array and is ignored by injected seams and SpikeGLX.

    Returns
    -------
    tuple[callable, LFPSiteConfig]
        A normalized trial loader returning ``(time_s, values, rate_hz)`` on
        one-dimensional time/value axes, plus its effective-rate site config.
    """
    injected = spikeglx_loader if site.acquisition_format == "spikeglx" else open_ephys_loader
    if injected is not None:
        def injected_adapter(
            _: LFPSiteConfig,
            alignment_time_s: float,
            window: tuple[float, float],
        ) -> tuple[np.ndarray, np.ndarray, float]:
            return injected(site=site, alignment_time_s=alignment_time_s, window=window)
        return injected_adapter, site
    if site.acquisition_format == "spikeglx":
        return _spikeglx_adapter(site)
    return _open_ephys_adapter(site, normalized_open_ephys_metadata)


def _spikeglx_adapter(
    site: LFPSiteConfig,
) -> tuple[Callable[..., tuple[np.ndarray, np.ndarray, float]], LFPSiteConfig]:
    """Decode one site's sync once and adapt the real SpikeGLX loader signature."""
    sync_df, sample_rate_hz = lfp_loading.decode_lfp_sync(site.lfp_path)
    effective_site = replace(site, sample_rate_hz=float(sample_rate_hz))

    def loader(
        _: LFPSiteConfig,
        alignment_time_s: float,
        window: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return lfp_loading.load_trial_lfp_trace_with_sample_rate(
            lfp_path=site.lfp_path,
            saved_channel_index=site.saved_channel_index,
            alignment_time_s=alignment_time_s,
            window=window,
            lfp_irig_df=sync_df,
            sample_rate_hz=float(sample_rate_hz),
        )
    return loader, effective_site


def _open_ephys_adapter(
    site: LFPSiteConfig,
    normalized_metadata: dict[str, object] | None = None,
) -> tuple[Callable[..., tuple[np.ndarray, np.ndarray, float]], LFPSiteConfig]:
    """Adapt one preflighted Open Ephys source to exact native trial grids.

    Parameters
    ----------
    site : LFPSiteConfig
        One configured Open Ephys saved channel. Its configured sampling rate
        is Hz and its source voltage unit must match sidecar physical uV.
    normalized_metadata : dict[str, object] or None, optional
        Normalized sidecar metadata from :func:`preflight_open_ephys_site_metadata`.
        When omitted, this private adapter loads and validates one sidecar for
        direct callers. It contains no trace samples.

    Returns
    -------
    tuple[callable, LFPSiteConfig]
        A loader accepting absolute alignment seconds and a relative seconds
        window, returning exact native ``(time_s, values_uV, rate_hz)``
        one-dimensional arrays/scalar. The effective site rate is authoritative
        sidecar Hz.

    Raises
    ------
    ValueError
        If sync, normalized metadata, saved-channel bounds, rate, or physical
        voltage units are incompatible. No trial numerical loading occurs.
    """
    validate_open_ephys_aligned_sync_path(site)
    metadata = (
        normalized_metadata
        if normalized_metadata is not None
        else lfp_loading.load_open_ephys_lfp_metadata(site.lfp_path)
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"Open Ephys site {site.stable_id}: normalized metadata is invalid")
    sample_rate_hz = float(metadata["sampling_frequency_hz"])
    lfp_loading.validate_open_ephys_site_metadata(
        site.stable_id,
        site.saved_channel_index,
        site.sample_rate_hz,
        site.voltage_unit,
        metadata,
    )
    effective_site = replace(site, sample_rate_hz=sample_rate_hz)

    def loader(
        _: LFPSiteConfig,
        alignment_time_s: float,
        window: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        time_s, values = lfp_loading.load_open_ephys_trial_lfp_trace(
            lfp_path=site.lfp_path,
            aligned_sync_npz_path=site.aligned_sync_path,
            saved_channel_index=site.saved_channel_index,
            alignment_time_s=alignment_time_s,
            window=window,
        )
        target_time_s = build_common_event_grid(window[0], window[1], sample_rate_hz)
        target_values = _interpolate_open_ephys_trace_to_native_grid(
            time_s,
            values,
            target_time_s,
        )
        return target_time_s, target_values, sample_rate_hz
    return loader, effective_site


def _interpolate_open_ephys_trace_to_native_grid(
    source_time_s: np.ndarray,
    source_values: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    """Interpolate one fractional Open Ephys trace onto an exact native grid.

    Parameters
    ----------
    source_time_s : numpy.ndarray
        Finite, strictly increasing `(source_time,)` event-relative seconds from
        the production Open Ephys loader. It may contain one extra edge sample
        because a fractional alignment residual selects whole source samples.
    source_values : numpy.ndarray
        Finite `(source_time,)` source-voltage values sharing ``source_time_s``.
    target_time_s : numpy.ndarray
        Finite, strictly increasing `(time,)` exact half-open native seconds
        grid requested by :func:`prepare_site_trial_traces`.

    Returns
    -------
    numpy.ndarray
        Float64 `(time,)` source-voltage values on ``target_time_s``. Linear
        interpolation corrects only the sub-sample alignment residual; units
        and target shape are unchanged.

    Raises
    ------
    ValueError
        If source axes are invalid, values are nonfinite, or target support is
        incomplete. The helper never extrapolates beyond finite continuous input
        support or bridges missing samples.
    """
    source_time = np.asarray(source_time_s, dtype=float)
    source = np.asarray(source_values, dtype=float)
    target_time = np.asarray(target_time_s, dtype=float)
    valid_axes = (
        source_time.ndim == 1
        and source.ndim == 1
        and target_time.ndim == 1
        and source_time.size == source.size
        and source_time.size >= 2
    )
    if not valid_axes:
        raise ValueError("Open Ephys trace must have matching one-dimensional support")
    if (
        not np.isfinite(source_time).all()
        or not np.isfinite(source).all()
        or not np.isfinite(target_time).all()
        or np.any(np.diff(source_time) <= 0)
        or np.any(np.diff(target_time) <= 0)
    ):
        raise ValueError("Open Ephys trace support must be finite and strictly increasing")
    if target_time[0] < source_time[0] or target_time[-1] > source_time[-1]:
        raise ValueError("Open Ephys trace does not continuously cover the native grid")
    return np.interp(target_time, source_time, source).astype(float, copy=False)


def _finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Validate and return a float64 finite one-dimensional seconds vector."""
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"invalid {name}")
    return vector
