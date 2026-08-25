"""Observed spike-LFP PPC summaries without shuffle-based inference.

This module deliberately delegates PPC and circular metrics to
``spike_lfp_phase_locking.compute_frequency_phase_metrics``. WP5B adds null
distributions and significance fields later; no shuffle calculation belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from src.neural_analysis import spike_lfp_phase_locking


THETA_FREQUENCIES_HZ = (6.0, 8.0, 10.0)
GAMMA_FREQUENCIES_HZ = tuple(
    float(value)
    for value in range(30, 81, 2)
    if value not in {58, 60, 62}
)
MINIMUM_COMPUTABLE_SPIKES = 2
MINIMUM_RELIABLE_SPIKES = 50
POOLED_METRIC_LABEL = "pooled eligible trials"
ILLUSTRATIVE_TRIAL_LABEL = "illustrative single trial"


@dataclass(frozen=True)
class ObservedPPCResult:
    """Observed frequency-resolved PPC metrics for one probe-qualified unit.

    Attributes
    ----------
    unit_id : str
        Stable ``"probe_label:cluster_id"`` categorical unit identity.
    frequencies_hz : numpy.ndarray
        Float64 shape ``(frequency,)`` Morlet frequency coordinates in Hz.
    ppc : numpy.ndarray
        Float64 shape ``(frequency,)`` dimensionless, unbiased PPC. It remains
        negative when the established PPC formula yields a negative value and
        is NaN with fewer than two valid spike phases.
    resultant_length : numpy.ndarray
        Float64 shape ``(frequency,)`` mean-vector magnitude in ``[0, 1]``;
        NaN with no valid spike phases.
    preferred_phase_rad : numpy.ndarray
        Float64 shape ``(frequency,)`` circular mean in radians; NaN when the
        mean vector has no direction.
    spike_count : numpy.ndarray
        Int64 shape ``(frequency,)`` count of valid spike-phase observations.
    computable : numpy.ndarray
        Boolean shape ``(frequency,)`` true for at least two valid phases.
    reliable : numpy.ndarray
        Boolean shape ``(frequency,)`` true for at least fifty valid phases.
    """

    unit_id: str
    frequencies_hz: np.ndarray
    ppc: np.ndarray
    resultant_length: np.ndarray
    preferred_phase_rad: np.ndarray
    spike_count: np.ndarray
    computable: np.ndarray
    reliable: np.ndarray


@dataclass(frozen=True)
class BandPPCSummary:
    """Arithmetic band means from observed frequency-resolved PPC.

    Attributes
    ----------
    band_names : tuple[str, str]
        Ordered categorical names ``("theta", "gamma")``.
    ppc_band_mean : numpy.ndarray
        Float64 shape ``leading_axes + (band,)`` dimensionless arithmetic mean.
        A band is NaN when none of its retained discrete frequency values has a
        finite PPC; it does not interpolate or bridge excluded gamma values.
    """

    band_names: tuple[str, str]
    ppc_band_mean: np.ndarray


@dataclass(frozen=True)
class PopulationPPCSummary:
    """Ordered unit heatmap and reliable-only population PPC summaries.

    Attributes
    ----------
    unit_order_ids : tuple[str, ...]
        Stable probe-qualified unit identities ordered once by descending
        correct-rewarded, selected-site, whole-window theta PPC, with stable-id
        ascending tie breaks. That same order indexes every displayed condition.
    computable, reliable : numpy.ndarray
        Boolean shape ``(unit, condition, site, epoch, frequency)`` masks in
        ``unit_order_ids`` order. Computable requires at least two spikes;
        reliable requires at least fifty.
    heatmap_ppc : numpy.ndarray
        Float64 shape ``(unit, condition, site, epoch, frequency)`` PPC for all
        computable entries. Entries below two spikes or with unavailable PPC are
        NaN, rather than zero.
    population_median_ppc : numpy.ndarray
        Float64 shape ``(condition, site, epoch, frequency)`` median over only
        reliable unit entries. It is NaN where no reliable unit contributes.
    """

    unit_order_ids: tuple[str, ...]
    computable: np.ndarray
    reliable: np.ndarray
    heatmap_ppc: np.ndarray
    population_median_ppc: np.ndarray


@dataclass(frozen=True)
class RepresentativePhaseHistograms:
    """Pooled fixed-bin phase histograms at representative theta/gamma frequencies.

    Attributes
    ----------
    representative_frequency_hz : tuple[float, float]
        Actual configured frequency values in Hz nearest requested 8 and 40 Hz.
    phase_bin_edges_rad : numpy.ndarray
        Float64 shape ``(phase_bin + 1,)`` strictly increasing bin edges in
        radians, copied from the caller.
    spike_count_by_band : numpy.ndarray
        Int64 shape ``(band, phase_bin)`` pooled valid spike counts for theta
        then gamma. Invalid phases are omitted rather than represented by zero.
    source_label : str
        Explicit label identifying the counts as a pooled, not illustrative,
        trial result.
    """

    representative_frequency_hz: tuple[float, float]
    phase_bin_edges_rad: np.ndarray
    spike_count_by_band: np.ndarray
    source_label: str


@dataclass(frozen=True)
class PPCExemplarSelection:
    """Deterministic high/low pooled PPC units and their illustrative trials.

    Attributes
    ----------
    low_unit_id, high_unit_id : str or None
        Probe-qualified identities nearest the requested 5th and 95th percentile
        of finite band PPC. Both are ``None`` when fewer than two units qualify.
    illustrative_trial_index_by_unit : Mapping[str, int]
        Selected table-row index for each available exemplar. Each is the
        positive-spike trial nearest that unit's median spike count, with the
        earliest trial index breaking equal-count ties.
    pooled_metric_label, illustrative_trial_label : str
        Distinct labels preventing pooled PPC from being confused with the
        single-trial trace.
    """

    low_unit_id: str | None
    high_unit_id: str | None
    illustrative_trial_index_by_unit: Mapping[str, int]
    pooled_metric_label: str
    illustrative_trial_label: str


def build_probe_qualified_unit_ids(
    probe_labels: Sequence[str],
    cluster_ids: Sequence[int],
) -> tuple[str, ...]:
    """Build unique stable ``probe:cluster`` identities.

    Parameters
    ----------
    probe_labels : Sequence[str]
        Nonempty probe labels with shape ``(unit,)``. They are categorical and
        have no physical units.
    cluster_ids : Sequence[int]
        Nonnegative integer sorter cluster ids with shape ``(unit,)``.

    Returns
    -------
    tuple[str, ...]
        One probe-qualified categorical identity per input unit. No missing-id
        sentinel is returned.

    Raises
    ------
    ValueError
        If axes differ, a label is empty or contains ``":"``, an id is not a
        nonnegative integer, or the qualified identities collide.
    """
    if len(probe_labels) != len(cluster_ids):
        raise ValueError("probe labels and cluster ids must have matching lengths")
    identities: list[str] = []
    for probe_label, cluster_id in zip(probe_labels, cluster_ids, strict=True):
        _validate_probe_and_cluster(probe_label, cluster_id)
        identities.append(f"{probe_label}:{int(cluster_id)}")
    if len(set(identities)) != len(identities):
        raise ValueError("probe-qualified unit ids must be unique")
    return tuple(identities)


def compute_observed_ppc(
    *,
    probe_label: str,
    cluster_id: int,
    spike_phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    frequencies_hz: np.ndarray,
    spike_times_s: np.ndarray,
) -> ObservedPPCResult:
    """Compute observed PPC by delegating to the established PPC helper.

    Parameters
    ----------
    probe_label : str
        Nonempty categorical probe label used in the returned stable identity.
    cluster_id : int
        Nonnegative categorical sorter cluster identifier.
    spike_phase_vectors : numpy.ndarray
        Complex shape ``(frequency, spike)`` sampled LFP phase vectors. NaN or
        zero-magnitude vectors are unavailable observations.
    valid_mask : numpy.ndarray
        Boolean shape ``(frequency, spike)`` candidate-validity mask.
    frequencies_hz : numpy.ndarray
        Finite positive float shape ``(frequency,)`` coordinates in Hz.
    spike_times_s : numpy.ndarray
        Finite float shape ``(spike,)`` synchronized absolute spike times in
        seconds. Times identify observations but do not affect PPC weighting.

    Returns
    -------
    ObservedPPCResult
        Frequency-resolved dimensionless PPC/circular metrics and spike counts.
        PPC below two valid phases is NaN; 2--49 is computable but unreliable;
        fifty or more is reliable.

    Raises
    ------
    ValueError
        If the identity is invalid, input axes do not match, or coordinates are
        nonfinite. The delegated helper validates its additional phase details.
    """
    _validate_probe_and_cluster(probe_label, cluster_id)
    phase = np.asarray(spike_phase_vectors)
    valid = np.asarray(valid_mask, dtype=bool)
    frequencies = _frequency_vector(frequencies_hz)
    spike_times = _finite_seconds_vector(spike_times_s, "spike_times_s")
    if phase.ndim != 2 or valid.shape != phase.shape:
        raise ValueError("phase vectors and valid mask must have shape (frequency, spike)")
    if phase.shape != (frequencies.size, spike_times.size):
        raise ValueError("phase vectors must match frequency and spike axes")

    metrics = spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=phase,
        valid_mask=valid,
        frequencies_hz=frequencies,
        spike_times_s=spike_times,
        unit_id=int(cluster_id),
        lfp_site_label=probe_label,
    )
    spike_count = np.asarray(metrics.n_spikes, dtype=np.int64).copy()
    computable = spike_count >= MINIMUM_COMPUTABLE_SPIKES
    reliable = spike_count >= MINIMUM_RELIABLE_SPIKES
    return ObservedPPCResult(
        unit_id=f"{probe_label}:{int(cluster_id)}",
        frequencies_hz=np.asarray(metrics.frequencies_hz, dtype=float).copy(),
        ppc=np.asarray(metrics.ppc, dtype=float).copy(),
        resultant_length=np.asarray(metrics.resultant_length, dtype=float).copy(),
        preferred_phase_rad=np.asarray(metrics.preferred_phase_rad, dtype=float).copy(),
        spike_count=spike_count,
        computable=computable,
        reliable=reliable,
    )


def compute_band_ppc_means(
    *,
    frequencies_hz: np.ndarray,
    ppc_by_frequency: np.ndarray,
) -> BandPPCSummary:
    """Average observed PPC over approved retained theta and gamma grid values.

    Parameters
    ----------
    frequencies_hz : numpy.ndarray
        Finite strictly increasing float shape ``(frequency,)`` grid in Hz with
        a 2-Hz step. The grid may omit values outside the requested analysis.
    ppc_by_frequency : numpy.ndarray
        Floating shape ``leading_axes + (frequency,)`` dimensionless PPC.
        NaN denotes an unavailable frequency and is ignored within a band.

    Returns
    -------
    BandPPCSummary
        Theta/gamma arithmetic means with shape ``leading_axes + (band,)``.
        A band with no finite retained values is NaN; gamma excludes 58/60/62 Hz.

    Raises
    ------
    ValueError
        If the frequency coordinate is invalid, not a 2-Hz grid, or the final
        PPC axis does not match. Missing PPC values are valid input data.
    """
    frequencies = _two_hz_frequency_vector(frequencies_hz)
    ppc = np.asarray(ppc_by_frequency, dtype=float)
    if ppc.ndim < 1 or ppc.shape[-1] != frequencies.size:
        raise ValueError("PPC final axis must match frequencies_hz")

    theta = _mean_retained_frequencies(ppc, frequencies, THETA_FREQUENCIES_HZ)
    gamma = _mean_retained_frequencies(ppc, frequencies, GAMMA_FREQUENCIES_HZ)
    return BandPPCSummary(
        band_names=("theta", "gamma"),
        ppc_band_mean=np.stack((theta, gamma), axis=-1),
    )


def build_population_ppc_summary(
    *,
    unit_ids: Sequence[str],
    condition_names: Sequence[str],
    site_ids: Sequence[str],
    epoch_names: Sequence[str],
    frequencies_hz: np.ndarray,
    ppc: np.ndarray,
    spike_count: np.ndarray,
    selected_site_id: str | None = None,
) -> PopulationPPCSummary:
    """Order observed unit PPC once and derive heatmap and reliable medians.

    Parameters
    ----------
    unit_ids, condition_names, site_ids, epoch_names : Sequence[str]
        Unique categorical axis labels for arrays shaped ``(unit, condition,
        site, epoch, frequency)``. ``correct_rewarded`` and ``whole`` must be
        present; ``selected_site_id`` defaults to the first site label.
    frequencies_hz : numpy.ndarray
        Finite strictly increasing float ``(frequency,)`` coordinate in Hz.
    ppc : numpy.ndarray
        Float shape ``(unit, condition, site, epoch, frequency)`` dimensionless
        observed PPC. NaN represents unavailable PPC.
    spike_count : numpy.ndarray
        Nonnegative integer array matching ``ppc``; values count valid phases.
    selected_site_id : str or None, default=None
        Site used for the fixed ordering reference. ``None`` selects
        ``site_ids[0]``; it has categorical rather than physical units.

    Returns
    -------
    PopulationPPCSummary
        Input unit axis reordered once for all output conditions. The heatmap
        retains finite PPC with at least two spikes; medians use only entries
        with at least fifty spikes and are NaN when no such unit exists.

    Raises
    ------
    ValueError
        If identities are malformed or duplicated, axes are incompatible, counts
        are noninteger/negative, required reference labels are absent, or the
        frequency coordinate is invalid. No partial population result is made.
    """
    ordered_unit_ids = _validate_stable_unit_ids(unit_ids)
    conditions = _unique_names(condition_names, "condition_names")
    sites = _unique_names(site_ids, "site_ids")
    epochs = _unique_names(epoch_names, "epoch_names")
    frequencies = _frequency_vector(frequencies_hz)
    ppc_values = np.asarray(ppc, dtype=float)
    counts = _count_array(spike_count)
    expected_shape = (
        len(ordered_unit_ids),
        len(conditions),
        len(sites),
        len(epochs),
        frequencies.size,
    )
    if ppc_values.shape != expected_shape or counts.shape != expected_shape:
        raise ValueError("PPC and spike counts must use documented five-dimensional axes")

    reference_site = sites[0] if selected_site_id is None else selected_site_id
    order = _reference_unit_order(
        ordered_unit_ids,
        ppc_values,
        counts,
        conditions,
        sites,
        epochs,
        frequencies,
        reference_site,
    )
    sorted_ppc = ppc_values[order].copy()
    sorted_counts = counts[order].copy()
    computable = (sorted_counts >= MINIMUM_COMPUTABLE_SPIKES) & np.isfinite(sorted_ppc)
    reliable = (sorted_counts >= MINIMUM_RELIABLE_SPIKES) & np.isfinite(sorted_ppc)
    heatmap = np.where(computable, sorted_ppc, np.nan)
    median = _reliable_population_median(sorted_ppc, reliable)
    return PopulationPPCSummary(
        unit_order_ids=tuple(ordered_unit_ids[index] for index in order),
        computable=computable,
        reliable=reliable,
        heatmap_ppc=heatmap,
        population_median_ppc=median,
    )


def build_representative_phase_histograms(
    *,
    frequencies_hz: np.ndarray,
    spike_phase_vectors: np.ndarray,
    valid_mask: np.ndarray,
    trial_indices: np.ndarray,
    phase_bin_edges_rad: np.ndarray,
) -> RepresentativePhaseHistograms:
    """Pool valid spike phases into fixed representative-frequency histograms.

    Parameters
    ----------
    frequencies_hz : numpy.ndarray
        Finite strictly increasing float shape ``(frequency,)`` in Hz. The
        configured values nearest requested 8 and 40 Hz are selected explicitly.
    spike_phase_vectors : numpy.ndarray
        Complex shape ``(frequency, spike)`` pooled phase vectors. Nonfinite or
        zero-magnitude vectors are unavailable.
    valid_mask : numpy.ndarray
        Boolean shape ``(frequency, spike)`` candidate-validity mask.
    trial_indices : numpy.ndarray
        Integer shape ``(spike,)`` source trial row for every pooled spike. It
        validates pooling provenance but does not split the output histogram.
    phase_bin_edges_rad : numpy.ndarray
        Finite strictly increasing float shape ``(phase_bin + 1,)`` bin edges
        in radians.

    Returns
    -------
    RepresentativePhaseHistograms
        Theta then gamma fixed-bin pooled spike counts at actual selected Hz.
        Invalid observations are excluded; no missing count sentinel is used.

    Raises
    ------
    ValueError
        If axes do not match, coordinates or edges are invalid, or trial indices
        are not integer-valued. Empty valid pools return all-zero count rows.
    """
    frequencies = _frequency_vector(frequencies_hz)
    phase = np.asarray(spike_phase_vectors)
    valid = np.asarray(valid_mask, dtype=bool)
    trials = np.asarray(trial_indices)
    edges = _phase_bin_edges(phase_bin_edges_rad)
    if phase.ndim != 2 or valid.shape != phase.shape:
        raise ValueError("phase vectors and valid mask must have shape (frequency, spike)")
    if phase.shape[0] != frequencies.size:
        raise ValueError("phase frequency axis must match frequencies_hz")
    if trials.ndim != 1 or trials.size != phase.shape[1]:
        raise ValueError("trial_indices must match the spike axis")
    if not np.issubdtype(trials.dtype, np.integer):
        raise ValueError("trial_indices must be integers")

    selected_indices = (
        _nearest_frequency_index(frequencies, 8.0),
        _nearest_frequency_index(frequencies, 40.0),
    )
    counts = np.empty((2, edges.size - 1), dtype=np.int64)
    for band_index, frequency_index in enumerate(selected_indices):
        phase_row = phase[frequency_index]
        usable = valid[frequency_index].copy()
        usable &= np.isfinite(phase_row.real) & np.isfinite(phase_row.imag)
        usable &= np.abs(phase_row) > 0.0
        angles = np.angle(phase_row[usable])
        counts[band_index] = np.histogram(angles, bins=edges)[0]
    return RepresentativePhaseHistograms(
        representative_frequency_hz=tuple(float(frequencies[index]) for index in selected_indices),
        phase_bin_edges_rad=edges.copy(),
        spike_count_by_band=counts,
        source_label=POOLED_METRIC_LABEL,
    )


def select_ppc_exemplars(
    *,
    unit_ids: Sequence[str],
    band_ppc: np.ndarray,
    trial_indices: np.ndarray,
    trial_spike_counts: np.ndarray,
) -> PPCExemplarSelection:
    """Select deterministic pooled-PPC unit exemplars and single trial traces.

    Parameters
    ----------
    unit_ids : Sequence[str]
        Unique probe-qualified unit labels with shape ``(unit,)``.
    band_ppc : numpy.ndarray
        Float shape ``(unit,)`` dimensionless band PPC. NaN units are ineligible
        for high/low percentile selection.
    trial_indices : numpy.ndarray
        Integer table-row positions with shape ``(trial,)``.
    trial_spike_counts : numpy.ndarray
        Nonnegative integer shape ``(unit, trial)`` counts of valid selected-epoch
        spikes. Only strictly positive counts can supply an illustrative trial.

    Returns
    -------
    PPCExemplarSelection
        Units nearest the default 5th/95th percentile and an illustrative trial
        per available unit. Fewer than two finite PPC values returns unavailable
        ``None`` ids and an empty mapping; labels remain explicit.

    Raises
    ------
    ValueError
        If identities, axes, or count values violate their contracts. Missing PPC
        is represented by NaN and does not raise.
    """
    identities = _validate_stable_unit_ids(unit_ids)
    ppc_values = np.asarray(band_ppc, dtype=float)
    trials = np.asarray(trial_indices)
    counts = _count_array(trial_spike_counts)
    if ppc_values.ndim != 1 or ppc_values.size != len(identities):
        raise ValueError("band_ppc must have a unit axis")
    if trials.ndim != 1 or not np.issubdtype(trials.dtype, np.integer):
        raise ValueError("trial_indices must be one-dimensional integers")
    if counts.shape != (len(identities), trials.size):
        raise ValueError("trial spike counts must have shape (unit, trial)")

    eligible_indices = np.flatnonzero(np.isfinite(ppc_values))
    if eligible_indices.size < 2:
        return _unavailable_exemplars()
    low_index = _percentile_unit_index(identities, ppc_values, eligible_indices, 5.0)
    high_index = _percentile_unit_index(identities, ppc_values, eligible_indices, 95.0)
    selected = (low_index, high_index)
    illustrative = {
        identities[index]: _illustrative_trial_index(trials, counts[index])
        for index in selected
        if _has_positive_spike_count(counts[index])
    }
    return PPCExemplarSelection(
        low_unit_id=identities[low_index],
        high_unit_id=identities[high_index],
        illustrative_trial_index_by_unit=MappingProxyType(illustrative),
        pooled_metric_label=POOLED_METRIC_LABEL,
        illustrative_trial_label=ILLUSTRATIVE_TRIAL_LABEL,
    )


# Private validation and numerical helpers.
def _validate_probe_and_cluster(probe_label: str, cluster_id: int) -> None:
    """Validate one categorical probe label and nonnegative integer cluster id.

    Parameters
    ----------
    probe_label : str
        Nonempty probe identity without the qualified-id separator ``":"``.
    cluster_id : int
        Nonnegative integer sorter label with categorical, not physical, units.

    Returns
    -------
    None
        Returns ``None`` when the identity is usable; no missing identity value
        is accepted.

    Raises
    ------
    ValueError
        If either part is malformed.
    """
    if not isinstance(probe_label, str) or not probe_label or ":" in probe_label:
        raise ValueError("probe_label must be a nonempty label without ':'")
    if not isinstance(cluster_id, (int, np.integer)) or int(cluster_id) < 0:
        raise ValueError("cluster_id must be a nonnegative integer")


def _frequency_vector(values: np.ndarray) -> np.ndarray:
    """Validate a finite strictly increasing one-dimensional Hz coordinate.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate float shape ``(frequency,)`` coordinate in Hz.

    Returns
    -------
    numpy.ndarray
        Owned float64 ``(frequency,)`` Hz coordinate. Missing/nonfinite values
        are rejected rather than converted to a sentinel.

    Raises
    ------
    ValueError
        If the axis is empty, nonfinite, nonpositive, or nonmonotonic.
    """
    frequencies = np.asarray(values, dtype=float)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or not np.isfinite(frequencies).all()
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise ValueError("frequencies_hz must be finite, positive, and increasing")
    return frequencies.copy()


def _two_hz_frequency_vector(values: np.ndarray) -> np.ndarray:
    """Validate a one-dimensional linear 2-Hz coordinate for band summaries.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate ``(frequency,)`` coordinate in Hz.

    Returns
    -------
    numpy.ndarray
        Owned float64 increasing Hz coordinate with every adjacent step 2 Hz.
        Missing values are rejected.

    Raises
    ------
    ValueError
        If the coordinate fails general frequency validation or has another step.
    """
    frequencies = _frequency_vector(values)
    if frequencies.size > 1 and not np.allclose(np.diff(frequencies), 2.0):
        raise ValueError("band PPC requires a linear 2-Hz frequency grid")
    return frequencies


def _finite_seconds_vector(values: np.ndarray, name: str) -> np.ndarray:
    """Validate an owned finite one-dimensional absolute-seconds vector.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate float shape ``(observation,)`` timestamps in seconds.
    name : str
        Human-readable parameter name for an error message.

    Returns
    -------
    numpy.ndarray
        Owned float64 seconds vector. Missing/nonfinite timestamps are rejected.

    Raises
    ------
    ValueError
        If the input is not one-dimensional finite seconds data.
    """
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite one-dimensional seconds vector")
    return vector.copy()


def _count_array(values: np.ndarray) -> np.ndarray:
    """Validate nonnegative integer count data without changing its axes.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate integer count array of caller-defined axes; counts are
        dimensionless nonnegative observation totals.

    Returns
    -------
    numpy.ndarray
        Owned int64 array with the original axes. Missing counts are rejected.

    Raises
    ------
    ValueError
        If values are noninteger, nonfinite, or negative.
    """
    counts = np.asarray(values)
    if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
        raise ValueError("spike counts must be nonnegative integers")
    return counts.astype(np.int64, copy=True)


def _unique_names(values: Sequence[str], name: str) -> tuple[str, ...]:
    """Validate one nonempty unique categorical axis-label sequence.

    Parameters
    ----------
    values : Sequence[str]
        Candidate shape ``(axis,)`` categorical labels with no physical units.
    name : str
        Axis name used in validation errors.

    Returns
    -------
    tuple[str, ...]
        Immutable axis labels in supplied order. Missing labels are rejected.

    Raises
    ------
    ValueError
        If labels are empty, nonstrings, or duplicated.
    """
    labels = tuple(values)
    if not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must be unique")
    return labels


def _validate_stable_unit_ids(values: Sequence[str]) -> tuple[str, ...]:
    """Validate a unique one-dimensional axis of probe-qualified unit strings.

    Parameters
    ----------
    values : Sequence[str]
        Candidate categorical ``(unit,)`` labels in ``"probe:cluster"`` form.

    Returns
    -------
    tuple[str, ...]
        Immutable supplied-order unit identities. Missing or duplicate labels are
        rejected; no unqualified fallback identity is introduced.

    Raises
    ------
    ValueError
        If a label lacks a nonempty probe/cluster part or identities collide.
    """
    identities = _unique_names(values, "unit_ids")
    for identity in identities:
        probe_label, separator, cluster_text = identity.partition(":")
        if not separator or not probe_label or not cluster_text or ":" in cluster_text:
            raise ValueError("unit_ids must use probe-qualified 'probe:cluster' strings")
        try:
            cluster_id = int(cluster_text)
        except ValueError as error:
            raise ValueError("unit_ids must have integer cluster identifiers") from error
        _validate_probe_and_cluster(probe_label, cluster_id)
    return identities


def _mean_retained_frequencies(
    ppc: np.ndarray,
    frequencies_hz: np.ndarray,
    retained_hz: Sequence[float],
) -> np.ndarray:
    """Take an arithmetic finite-value mean over named retained Hz coordinates.

    Parameters
    ----------
    ppc : numpy.ndarray
        Float shape ``leading_axes + (frequency,)`` dimensionless PPC values.
    frequencies_hz : numpy.ndarray
        Float shape ``(frequency,)`` Hz coordinate matching the final PPC axis.
    retained_hz : Sequence[float]
        Requested discrete frequency values in Hz.

    Returns
    -------
    numpy.ndarray
        Float shape ``leading_axes`` arithmetic mean. It is NaN if no requested
        finite PPC value is present for one output cell.
    """
    selected = np.isin(frequencies_hz, retained_hz)
    output = np.full(ppc.shape[:-1], np.nan, dtype=float)
    if not np.any(selected):
        return output
    values = ppc[..., selected]
    finite = np.isfinite(values)
    counts = finite.sum(axis=-1)
    sums = np.where(finite, values, 0.0).sum(axis=-1)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


def _reference_unit_order(
    unit_ids: tuple[str, ...],
    ppc: np.ndarray,
    spike_count: np.ndarray,
    conditions: tuple[str, ...],
    sites: tuple[str, ...],
    epochs: tuple[str, ...],
    frequencies_hz: np.ndarray,
    selected_site_id: str,
) -> np.ndarray:
    """Rank units by correct-rewarded whole-window theta observed PPC.

    Parameters
    ----------
    unit_ids : tuple[str, ...]
        Unique ``(unit,)`` probe-qualified categorical labels.
    ppc : numpy.ndarray
        Float ``(unit, condition, site, epoch, frequency)`` dimensionless PPC.
    spike_count : numpy.ndarray
        Int64 array matching ``ppc`` with valid spike-phase counts. Reference
        entries below two spikes are unavailable for ordering even if PPC is
        finite.
    conditions, sites, epochs : tuple[str, ...]
        Categorical labels for the corresponding PPC axes.
    frequencies_hz : numpy.ndarray
        Float ``(frequency,)`` coordinate in Hz.
    selected_site_id : str
        Categorical site label used for ranking.

    Returns
    -------
    numpy.ndarray
        Int64 ``(unit,)`` input-axis indices ranked descending by finite theta
        mean; unavailable means sort after finite values, then stable id breaks ties.

    Raises
    ------
    ValueError
        If the required condition, epoch, or selected site is absent.
    """
    if "correct_rewarded" not in conditions or "whole" not in epochs:
        raise ValueError("population ordering requires correct_rewarded and whole")
    if selected_site_id not in sites:
        raise ValueError("selected_site_id must name a configured site")
    condition_index = conditions.index("correct_rewarded")
    site_index = sites.index(selected_site_id)
    epoch_index = epochs.index("whole")
    reference = ppc[:, condition_index, site_index, epoch_index, :]
    reference_counts = spike_count[:, condition_index, site_index, epoch_index, :]
    reference_computable = (
        reference_counts >= MINIMUM_COMPUTABLE_SPIKES
    ) & np.isfinite(reference)
    reference = np.where(reference_computable, reference, np.nan)
    theta_mean = _mean_retained_frequencies(reference, frequencies_hz, THETA_FREQUENCIES_HZ)
    sort_keys = []
    for index, value in enumerate(theta_mean):
        unavailable = not np.isfinite(value)
        descending_value = -float(value) if not unavailable else 0.0
        sort_keys.append(
            (unavailable, descending_value, unit_ids[index], index)
        )
    return np.asarray(sorted(range(len(unit_ids)), key=sort_keys.__getitem__), dtype=np.int64)


def _reliable_population_median(ppc: np.ndarray, reliable: np.ndarray) -> np.ndarray:
    """Compute reliable-only unit medians while preserving non-unit axes.

    Parameters
    ----------
    ppc : numpy.ndarray
        Float ``(unit, condition, site, epoch, frequency)`` PPC values.
    reliable : numpy.ndarray
        Boolean array matching ``ppc``; true where a unit has at least fifty
        valid spike phases and finite PPC.

    Returns
    -------
    numpy.ndarray
        Float ``(condition, site, epoch, frequency)`` medians over reliable
        units. Cells without a reliable unit are NaN rather than zero.
    """
    output = np.full(ppc.shape[1:], np.nan, dtype=float)
    for index in np.ndindex(output.shape):
        values = ppc[(slice(None),) + index]
        selected = reliable[(slice(None),) + index]
        if np.any(selected):
            output[index] = np.median(values[selected])
    return output


def _phase_bin_edges(values: np.ndarray) -> np.ndarray:
    """Validate a strictly increasing one-dimensional radians bin coordinate.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate float shape ``(phase_bin + 1,)`` edge coordinate in radians.

    Returns
    -------
    numpy.ndarray
        Owned float64 radians edge vector. Missing/nonfinite edges are rejected.

    Raises
    ------
    ValueError
        If fewer than two edges, nonfinite values, or nonincreasing edges occur.
    """
    edges = np.asarray(values, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.isfinite(edges).all():
        raise ValueError("phase_bin_edges_rad must be finite one-dimensional edges")
    if np.any(np.diff(edges) <= 0.0):
        raise ValueError("phase_bin_edges_rad must be strictly increasing")
    return edges.copy()


def _nearest_frequency_index(frequencies_hz: np.ndarray, requested_hz: float) -> int:
    """Return the index of the configured Hz value nearest one requested frequency.

    Parameters
    ----------
    frequencies_hz : numpy.ndarray
        Finite increasing float shape ``(frequency,)`` coordinate in Hz.
    requested_hz : float
        Positive representative frequency in Hz.

    Returns
    -------
    int
        Index of the nearest configured frequency; equal-distance ties select the
        lower frequency through NumPy's first-index behavior.
    """
    return int(np.argmin(np.abs(frequencies_hz - requested_hz)))


def _percentile_unit_index(
    unit_ids: tuple[str, ...],
    values: np.ndarray,
    eligible_indices: np.ndarray,
    percentile: float,
) -> int:
    """Select a finite unit nearest one percentile with stable-id tie breaking.

    Parameters
    ----------
    unit_ids : tuple[str, ...]
        Categorical ``(unit,)`` probe-qualified labels.
    values : numpy.ndarray
        Float ``(unit,)`` dimensionless PPC values.
    eligible_indices : numpy.ndarray
        Int64 ``(eligible_unit,)`` indices where ``values`` is finite.
    percentile : float
        Requested percentile in percent, such as 5 or 95.

    Returns
    -------
    int
        Input unit-axis index nearest the percentile; ties use stable-id order.
    """
    target = float(np.percentile(values[eligible_indices], percentile))
    return min(
        eligible_indices.tolist(),
        key=lambda index: (abs(values[index] - target), unit_ids[index]),
    )


def _has_positive_spike_count(counts: np.ndarray) -> bool:
    """Report whether a one-dimensional trial count axis has an eligible spike.

    Parameters
    ----------
    counts : numpy.ndarray
        Int64 shape ``(trial,)`` nonnegative spike counts.

    Returns
    -------
    bool
        True when at least one count is positive; zero counts represent no spike.
    """
    return bool(np.any(counts > 0))


def _illustrative_trial_index(trial_indices: np.ndarray, counts: np.ndarray) -> int:
    """Select one positive-spike trial nearest its median count.

    Parameters
    ----------
    trial_indices : numpy.ndarray
        Int64 ``(trial,)`` source table-row positions.
    counts : numpy.ndarray
        Int64 ``(trial,)`` nonnegative valid spike counts for one unit.

    Returns
    -------
    int
        Earliest table-row index among positive-count trials nearest the median.

    Raises
    ------
    ValueError
        If no positive-count trial exists. Callers normally check this first.
    """
    eligible = counts > 0
    if not np.any(eligible):
        raise ValueError("an illustrative trial requires a positive spike count")
    median_count = float(np.median(counts[eligible]))
    candidates = np.flatnonzero(eligible)
    selected_position = min(
        candidates.tolist(),
        key=lambda index: (
            abs(float(counts[index]) - median_count),
            int(trial_indices[index]),
        ),
    )
    return int(trial_indices[selected_position])


def _unavailable_exemplars() -> PPCExemplarSelection:
    """Return explicit unavailable exemplar fields for fewer than two units.

    Returns
    -------
    PPCExemplarSelection
        ``None`` high/low ids and an empty immutable trial mapping. Labels still
        distinguish pooled metrics from an unavailable illustrative trace.
    """
    return PPCExemplarSelection(
        low_unit_id=None,
        high_unit_id=None,
        illustrative_trial_index_by_unit=MappingProxyType({}),
        pooled_metric_label=POOLED_METRIC_LABEL,
        illustrative_trial_label=ILLUSTRATIVE_TRIAL_LABEL,
    )
