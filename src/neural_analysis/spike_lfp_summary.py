"""Observed spike-LFP PPC summaries and trial-shuffle inference.

This module deliberately delegates PPC and circular metrics to
``spike_lfp_phase_locking.compute_frequency_phase_metrics`` so all observed and
null values use one established formula.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import false_discovery_control

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


@dataclass(frozen=True)
class PermutationNullSummary:
    """Cache-ready trial-shuffle null statistics without retained shuffle draws.

    Attributes
    ----------
    null_exceedance_count, permutation_count, eligible_trial_count : numpy.ndarray
        Int64 shape ``metric_axes`` counts of finite null PPC values greater
        than or equal to observed PPC, all finite null values, and trials with
        at least one valid same-trial spike phase, respectively.
    p_value, null_mean, null_std, null_p025, null_p50, null_p975 : numpy.ndarray
        Float64 shape ``metric_axes`` dimensionless PPC statistics. ``p_value``
        is NaN when null inference is ineligible; null moments are NaN when no
        finite draw contributes.
    null_eligible, significant : numpy.ndarray
        Boolean shape ``metric_axes``. Eligibility requires finite observed PPC,
        fifty spikes, two trials, and one finite permutation; significance is
        false here until frequency-family FDR is applied.
    """

    null_exceedance_count: np.ndarray
    permutation_count: np.ndarray
    eligible_trial_count: np.ndarray
    p_value: np.ndarray
    null_mean: np.ndarray
    null_std: np.ndarray
    null_p025: np.ndarray
    null_p50: np.ndarray
    null_p975: np.ndarray
    null_eligible: np.ndarray
    significant: np.ndarray


@dataclass(frozen=True)
class TrialShufflePPCResult:
    """Observed and same-condition trial-shuffle PPC for one unit/site job.

    Attributes
    ----------
    observed_ppc, spike_count : numpy.ndarray
        Float64 dimensionless PPC and int64 valid phase count, each shape
        ``(frequency,)``. PPC is NaN below two valid phases.
    schedule : numpy.ndarray
        Int64 shape ``(shuffle, trial)`` source-to-target derangements, copied
        from validated input.
    trial_relative_spike_times_s : tuple[numpy.ndarray, ...]
        Owned float64 ``(spike_in_trial,)`` relative-seconds arrays, one per
        trial. Overlapping memberships remain duplicated across their trials.
    overlap_warning : bool
        True when ``overlap_trial_indices`` is nonempty.
    overlap_trial_indices : numpy.ndarray
        Owned int64 ``(overlapping_trial,)`` table-row positions; empty when no
        overlap warning applies.
    null_summary : PermutationNullSummary
        Cache-ready null statistics with ``metric_axes == (frequency,)``.
    """

    observed_ppc: np.ndarray
    spike_count: np.ndarray
    schedule: np.ndarray
    trial_relative_spike_times_s: tuple[np.ndarray, ...]
    overlap_warning: bool
    overlap_trial_indices: np.ndarray
    null_summary: PermutationNullSummary

    @property
    def p_value(self) -> np.ndarray:
        """Return owned-elsewhere frequency PPC p values from the null summary.

        Returns
        -------
        numpy.ndarray
            Float64 ``(frequency,)`` plus-one p values; ineligible entries are
            NaN. The array belongs to ``null_summary`` and callers should copy
            before mutation.
        """
        return self.null_summary.p_value

    @property
    def null_mean(self) -> np.ndarray:
        """Return frequency-wise dimensionless null PPC mean values.

        Returns
        -------
        numpy.ndarray
            Float64 ``(frequency,)`` null means; no finite draw is NaN. The
            array belongs to ``null_summary`` and callers should copy to mutate.
        """
        return self.null_summary.null_mean


@dataclass(frozen=True)
class EdgeSufficientStatistics:
    """Reduced spike-phase observations for stable trial-pair edges.

    Attributes
    ----------
    source_trial_position, target_trial_position : numpy.ndarray
        Owned int64 shape ``(edge,)`` zero-based positions on the shared trial
        axis. Same-trial edges are permitted for observed calculations; null
        schedules select only nonself edges.
    phase_vector_sum : numpy.ndarray
        Owned complex128 shape ``(edge, unit, frequency)`` sums of valid unit
        phase vectors. Phase is dimensionless.
    valid_spike_count : numpy.ndarray
        Owned int64 with the same axes as ``phase_vector_sum``. Counts are
        nonnegative valid spike-phase observations; zero has no missing sentinel.
    """

    source_trial_position: np.ndarray
    target_trial_position: np.ndarray
    phase_vector_sum: np.ndarray
    valid_spike_count: np.ndarray

    def __post_init__(self) -> None:
        """Validate axes and retain owned canonical-dtype arrays.

        Inputs are the four public fields documented on the class. This method
        preserves edge, unit, and frequency order, casts coordinates/counts to
        int64 and sums to complex128, and raises ``ValueError`` for malformed,
        duplicate, nonfinite, or inconsistent arrays. It returns ``None``.
        """
        source, target = _edge_position_arrays(
            self.source_trial_position,
            self.target_trial_position,
        )
        vector_sum = np.asarray(self.phase_vector_sum, dtype=np.complex128)
        counts = _count_array(self.valid_spike_count)
        expected_leading_shape = (source.size,)
        if (
            vector_sum.ndim != 3
            or vector_sum.shape[0:1] != expected_leading_shape
            or vector_sum.shape[1] < 1
            or vector_sum.shape[2] < 1
            or counts.shape != vector_sum.shape
        ):
            raise ValueError(
                "edge statistics require matching (edge, unit, frequency) arrays"
            )
        if not np.isfinite(vector_sum.real).all() or not np.isfinite(
            vector_sum.imag
        ).all():
            raise ValueError("edge phase-vector sums must be finite")
        if np.any((counts == 0) & (vector_sum != 0.0j)):
            raise ValueError("zero-count edge statistics must have a zero vector sum")
        object.__setattr__(self, "source_trial_position", source)
        object.__setattr__(self, "target_trial_position", target)
        object.__setattr__(self, "phase_vector_sum", vector_sum.copy())
        object.__setattr__(self, "valid_spike_count", counts)


@dataclass(frozen=True)
class SignificantPrevalenceSummary:
    """FDR-significant PPC prevalence reduced over the unit axis only.

    Attributes
    ----------
    prevalence : numpy.ndarray
        Float64 shape ``(condition, site, epoch, frequency)`` fraction of
        eligible units with q below alpha. It is NaN when no unit is eligible.
    eligible_unit_count, total_unit_count : numpy.ndarray
        Int64 shape ``(condition, site, epoch, frequency)`` denominator and all
        selected-unit count. Counts have categorical unit totals, not Hz/seconds.
    """

    prevalence: np.ndarray
    eligible_unit_count: np.ndarray
    total_unit_count: np.ndarray


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


def generate_trial_derangement_schedule(
    trial_count: int,
    shuffle_count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Generate deterministic source-to-target trial derangements.

    Parameters
    ----------
    trial_count : int
        Number of eligible trial-local spike trains and phase traces. It must be
        at least two; this is a categorical axis size, not a duration.
    shuffle_count : int
        Positive number of schedule rows. Preview and final runs normally use
        100 and 1000 rows, respectively.
    seed : int
        Integer NumPy random-generator seed with no physical units.

    Returns
    -------
    numpy.ndarray
        Owned int64 shape ``(shuffle, trial)`` array. Every row is a permutation
        of trial positions and has no fixed point; no missing schedule exists.

    Raises
    ------
    ValueError
        If trial/shuffle counts are invalid or a seed is not an integer.
    """
    if not isinstance(trial_count, (int, np.integer)) or int(trial_count) < 2:
        raise ValueError("trial_count must be an integer of at least two")
    if not isinstance(shuffle_count, (int, np.integer)) or int(shuffle_count) < 1:
        raise ValueError("shuffle_count must be a positive integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    trial_count = int(trial_count)
    generator = np.random.default_rng(int(seed))
    schedule = np.empty((int(shuffle_count), trial_count), dtype=np.int64)
    identity = np.arange(trial_count)
    for shuffle_index in range(schedule.shape[0]):
        permutation = generator.permutation(trial_count)
        while np.any(permutation == identity):
            permutation = generator.permutation(trial_count)
        schedule[shuffle_index] = permutation
    return schedule


def compute_edge_sufficient_statistics(
    *,
    trial_relative_spike_times_s: Sequence[Sequence[np.ndarray]],
    phase_time_s: np.ndarray,
    trial_phase_vectors: np.ndarray,
    frequencies_hz: np.ndarray,
    source_trial_position: np.ndarray,
    target_trial_position: np.ndarray,
) -> EdgeSufficientStatistics:
    """Sample bounded source-target edges and immediately reduce by unit.

    Parameters
    ----------
    trial_relative_spike_times_s : Sequence[Sequence[numpy.ndarray]]
        Nested ``(unit, trial)`` collection. Every element is a finite float64
        ``(spike_in_unit_trial,)`` event-relative seconds array. Empty arrays
        represent no spike, and overlapping trial membership remains separate.
    phase_time_s : numpy.ndarray
        Finite strictly increasing float64 shape ``(time,)`` common relative
        seconds coordinate.
    trial_phase_vectors : numpy.ndarray
        Numeric complex coefficients with shape ``(trial, frequency, time)``.
        Interpolation follows the WP5B real/imaginary adjacent-support rule;
        nonfinite, zero-magnitude, and out-of-support samples are unavailable.
    frequencies_hz : numpy.ndarray
        Finite positive increasing float64 shape ``(frequency,)`` coordinates
        in Hz. Values validate the phase frequency axis and are not transformed.
    source_trial_position, target_trial_position : numpy.ndarray
        Matching nonempty integer shape ``(edge,)`` zero-based positions. Edge
        order is preserved. Same-trial edges are permitted for observed work.

    Returns
    -------
    EdgeSufficientStatistics
        Complex128 sums and int64 counts with axes ``(edge, unit, frequency)``.
        No per-spike phase vector survives this function.

    Raises
    ------
    ValueError
        If nested spike axes, phase/time/frequency axes, or edge coordinates are
        invalid. No partial statistics are returned.
    """
    phase_time = _strict_relative_seconds(phase_time_s, "phase_time_s")
    frequencies = _frequency_vector(frequencies_hz)
    phase = np.asarray(trial_phase_vectors)
    if (
        phase.ndim != 3
        or phase.shape[1:] != (frequencies.size, phase_time.size)
        or phase.shape[0] < 1
        or not np.issubdtype(phase.dtype, np.number)
    ):
        raise ValueError(
            "trial_phase_vectors must have numeric shape (trial, frequency, time)"
        )
    unit_trial_spikes = _unit_trial_local_spike_arrays(
        trial_relative_spike_times_s,
        phase.shape[0],
    )
    source_positions, target_positions = _edge_position_arrays(
        source_trial_position,
        target_trial_position,
    )
    if (
        np.any(source_positions >= phase.shape[0])
        or np.any(target_positions >= phase.shape[0])
    ):
        raise ValueError("trial-edge positions must index the phase trial axis")

    statistic_shape = (
        source_positions.size,
        len(unit_trial_spikes),
        frequencies.size,
    )
    vector_sum = np.zeros(statistic_shape, dtype=np.complex128)
    valid_count = np.zeros(statistic_shape, dtype=np.int64)
    for edge_index, (source_position, target_position) in enumerate(
        zip(source_positions, target_positions, strict=True)
    ):
        target_phase = phase[int(target_position)]
        for unit_index, unit_spikes in enumerate(unit_trial_spikes):
            sampled = _interpolate_unit_phase(
                phase_time,
                target_phase,
                unit_spikes[int(source_position)],
            )
            valid = np.isfinite(sampled.real) & np.isfinite(sampled.imag)
            # WP5B converts normalized phase to complex64 before complex128 sums.
            unit_phase = np.zeros(sampled.shape, dtype=np.complex64)
            unit_phase[valid] = sampled[valid].astype(np.complex64)
            vector_sum[edge_index, unit_index] = np.sum(
                unit_phase,
                axis=1,
                dtype=np.complex128,
            )
            valid_count[edge_index, unit_index] = np.sum(
                valid,
                axis=1,
                dtype=np.int64,
            )
    return EdgeSufficientStatistics(
        source_trial_position=source_positions,
        target_trial_position=target_positions,
        phase_vector_sum=vector_sum,
        valid_spike_count=valid_count,
    )


def reduce_scheduled_shuffle_block(
    *,
    edge_statistics: EdgeSufficientStatistics,
    schedule: np.ndarray,
    shuffle_positions: np.ndarray,
) -> np.ndarray:
    """Reduce selected derangements into temporary unit-by-frequency PPC draws.

    Parameters
    ----------
    edge_statistics : EdgeSufficientStatistics
        Stable edge sums/counts with axes ``(edge, unit, frequency)``. It must
        contain every source-target pair requested by selected schedule rows.
    schedule : numpy.ndarray
        Int64 shape ``(shuffle, source_trial_position)`` derangements. Each row
        maps every source trial position to one distinct target phase trial.
    shuffle_positions : numpy.ndarray
        Nonempty unique integer shape ``(shuffle_block,)`` positions selecting
        rows from ``schedule``. Supplied order defines the output shuffle order.

    Returns
    -------
    numpy.ndarray
        Float64 dimensionless PPC with axes ``(shuffle_block, unit, frequency)``.
        Counts zero or one produce NaN; negative PPC values remain unchanged.
        The temporary draws are not stored on ``edge_statistics``.

    Raises
    ------
    ValueError
        If schedule/selection axes are invalid, an edge lies outside the trial
        axis, or a selected scheduled pair is absent from ``edge_statistics``.
    """
    if not isinstance(edge_statistics, EdgeSufficientStatistics):
        raise ValueError("edge_statistics must be EdgeSufficientStatistics")
    candidate_schedule = np.asarray(schedule)
    if candidate_schedule.ndim != 2:
        raise ValueError("schedule must have shape (shuffle, trial)")
    validated_schedule = _derangement_schedule(
        candidate_schedule,
        candidate_schedule.shape[1],
    )
    positions = _shuffle_position_array(
        shuffle_positions,
        validated_schedule.shape[0],
    )
    if (
        np.any(edge_statistics.source_trial_position >= validated_schedule.shape[1])
        or np.any(edge_statistics.target_trial_position >= validated_schedule.shape[1])
    ):
        raise ValueError("edge statistics lie outside the schedule trial axis")
    edge_lookup = {
        (int(source), int(target)): edge_index
        for edge_index, (source, target) in enumerate(
            zip(
                edge_statistics.source_trial_position,
                edge_statistics.target_trial_position,
                strict=True,
            )
        )
    }
    selected_schedule = validated_schedule[positions]
    edge_indices = np.empty(selected_schedule.shape, dtype=np.int64)
    for block_position, target_row in enumerate(selected_schedule):
        for source_position, target_position in enumerate(target_row):
            try:
                edge_indices[block_position, source_position] = edge_lookup[
                    (source_position, int(target_position))
                ]
            except KeyError as error:
                raise ValueError(
                    "edge statistics lack a selected scheduled trial pair"
                ) from error
    pooled_sum = np.sum(
        edge_statistics.phase_vector_sum[edge_indices],
        axis=1,
        dtype=np.complex128,
    )
    pooled_count = np.sum(
        edge_statistics.valid_spike_count[edge_indices],
        axis=1,
        dtype=np.int64,
    )
    ppc = np.full(pooled_count.shape, np.nan, dtype=float)
    computable = pooled_count >= MINIMUM_COMPUTABLE_SPIKES
    numerator = np.abs(pooled_sum) ** 2 - pooled_count
    denominator = pooled_count.astype(float) * (pooled_count - 1)
    np.divide(numerator, denominator, out=ppc, where=computable)
    return ppc


def compute_trial_shuffle_ppc(
    *,
    trial_relative_spike_times_s: Sequence[np.ndarray],
    phase_time_s: np.ndarray,
    trial_phase_vectors: np.ndarray,
    frequencies_hz: np.ndarray,
    schedule: np.ndarray,
    overlap_trial_indices: np.ndarray | None = None,
) -> TrialShufflePPCResult:
    """Compute observed PPC and a same-condition trial-shuffle null.

    Parameters
    ----------
    trial_relative_spike_times_s : Sequence[numpy.ndarray]
        One finite float ``(spike_in_trial,)`` event-relative seconds array per
        eligible trial. Arrays are preserved separately; overlapping trials may
        legitimately contain the same physical spike.
    phase_time_s : numpy.ndarray
        Finite strictly increasing float64 ``(time,)`` common relative-seconds
        coordinate shared by all phase traces.
    trial_phase_vectors : numpy.ndarray
        Complex shape ``(trial, frequency, time)`` LFP coefficients. Nonfinite,
        zero-magnitude, or out-of-support interpolation results are invalid.
    frequencies_hz : numpy.ndarray
        Finite positive float shape ``(frequency,)`` coordinates in Hz.
    schedule : numpy.ndarray
        Int64 source-to-target shape ``(shuffle, trial)`` derangements. Source
        spike times are sampled on the target trial's phase trace unchanged.
    overlap_trial_indices : numpy.ndarray or None, default=None
        Optional integer ``(overlapping_trial,)`` source trial rows. It is copied
        and recorded as a warning, never merged into a pooled interval.

    Returns
    -------
    TrialShufflePPCResult
        Observed and cache-ready null dimensionless PPC per frequency. Null p
        values are NaN/ineligible below fifty spikes, two trials, or a finite
        null draw. Returned arrays and trial trains are owned copies.

    Raises
    ------
    ValueError
        If trial/frequency/time axes, seconds coordinates, phase coefficients,
        local spikes, schedule, or overlap indices violate their contracts.
    """
    phase_time = _strict_relative_seconds(phase_time_s, "phase_time_s")
    frequencies = _frequency_vector(frequencies_hz)
    phase = np.asarray(trial_phase_vectors)
    trial_spikes = _trial_local_spike_arrays(trial_relative_spike_times_s)
    expected_shape = (len(trial_spikes), frequencies.size, phase_time.size)
    if phase.shape != expected_shape:
        raise ValueError("trial_phase_vectors must have shape (trial, frequency, time)")
    if not np.issubdtype(phase.dtype, np.number):
        raise ValueError("trial_phase_vectors must be numeric complex coefficients")
    schedule_array = _derangement_schedule(schedule, len(trial_spikes))
    overlap_indices = _overlap_index_array(overlap_trial_indices)

    sampled = _precompute_trial_phase_samples(phase_time, phase, trial_spikes)
    observed_vectors = _pooled_sampled_phase_vectors(sampled, np.arange(len(trial_spikes)))
    observed_metrics = _phase_metrics_from_vectors(observed_vectors, frequencies)
    eligible_trial_count = _spike_contributing_trial_count(sampled)
    null_rows = np.empty((schedule_array.shape[0], frequencies.size), dtype=float)
    for shuffle_index, target_trials in enumerate(schedule_array):
        shuffled_vectors = _pooled_sampled_phase_vectors(sampled, target_trials)
        null_rows[shuffle_index] = _phase_metrics_from_vectors(
            shuffled_vectors,
            frequencies,
        ).ppc
    null_summary = summarize_permutation_null(
        observed_ppc=observed_metrics.ppc,
        null_ppc_chunks=(null_rows,),
        spike_count=observed_metrics.n_spikes,
        eligible_trial_count=eligible_trial_count,
    )
    return TrialShufflePPCResult(
        observed_ppc=np.asarray(observed_metrics.ppc, dtype=float).copy(),
        spike_count=np.asarray(observed_metrics.n_spikes, dtype=np.int64).copy(),
        schedule=schedule_array.copy(),
        trial_relative_spike_times_s=tuple(values.copy() for values in trial_spikes),
        overlap_warning=bool(overlap_indices.size),
        overlap_trial_indices=overlap_indices.copy(),
        null_summary=null_summary,
    )


def summarize_permutation_null(
    *,
    observed_ppc: np.ndarray,
    null_ppc_chunks: Sequence[np.ndarray],
    spike_count: np.ndarray,
    eligible_trial_count: int | np.ndarray,
) -> PermutationNullSummary:
    """Accumulate finite PPC null draws into cache-ready scalar statistics.

    Parameters
    ----------
    observed_ppc : numpy.ndarray
        Float shape ``metric_axes`` dimensionless observed PPC. NaN is an
        unavailable observation.
    null_ppc_chunks : Sequence[numpy.ndarray]
        Temporary arrays, each shape ``(shuffle_chunk,) + metric_axes`` of
        dimensionless null PPC. Nonfinite draws are ignored per metric cell;
        exact percentiles may concatenate these supplied chunks in memory.
    spike_count : numpy.ndarray
        Nonnegative integer shape ``metric_axes`` observed valid phase counts.
    eligible_trial_count : int or numpy.ndarray
        Nonnegative scalar or int64 ``metric_axes`` count of trials with at
        least one valid same-trial spike phase. A scalar is broadcast across
        metrics for legacy direct callers.

    Returns
    -------
    PermutationNullSummary
        Counts, plus-one p values, moments, and percentiles with ``metric_axes``.
        No full draw array is retained. Ineligible p values are NaN and their
        significant flags are false.

    Raises
    ------
    ValueError
        If axes do not match, a chunk has no shuffle axis, counts are invalid,
        or eligible_trial_count is invalid. A cell is ineligible below two
        spike-contributing trials even if its condition includes more trials.
    """
    observed = np.asarray(observed_ppc, dtype=float)
    counts = _count_array(spike_count)
    if observed.shape != counts.shape:
        raise ValueError("observed_ppc and spike_count must have matching axes")
    contributing_trials = _eligible_trial_counts(
        eligible_trial_count,
        observed.shape,
    )
    chunks = _null_chunk_arrays(null_ppc_chunks, observed.shape)
    draws = np.concatenate(chunks, axis=0)
    finite_draw = np.isfinite(draws)
    permutation_count = finite_draw.sum(axis=0, dtype=np.int64)
    exceedance_count = (
        finite_draw & (draws >= observed[np.newaxis, ...])
    ).sum(axis=0, dtype=np.int64)
    null_mean, null_std, null_p025, null_p50, null_p975 = _null_distribution_statistics(
        draws,
        finite_draw,
    )
    null_eligible = (
        np.isfinite(observed)
        & (counts >= MINIMUM_RELIABLE_SPIKES)
        & (contributing_trials >= 2)
        & (permutation_count > 0)
    )
    p_value = np.full(observed.shape, np.nan, dtype=float)
    p_value[null_eligible] = (
        (1.0 + exceedance_count[null_eligible])
        / (1.0 + permutation_count[null_eligible])
    )
    return PermutationNullSummary(
        null_exceedance_count=exceedance_count,
        permutation_count=permutation_count,
        eligible_trial_count=contributing_trials,
        p_value=p_value,
        null_mean=null_mean,
        null_std=null_std,
        null_p025=null_p025,
        null_p50=null_p50,
        null_p975=null_p975,
        null_eligible=null_eligible,
        significant=np.zeros(observed.shape, dtype=bool),
    )


def permutation_null_summary_to_arrays(
    summary: PermutationNullSummary,
) -> dict[str, np.ndarray]:
    """Convert a null summary into owned cache-safe numeric/boolean arrays.

    Parameters
    ----------
    summary : PermutationNullSummary
        Cache-ready dimensionless PPC statistics with one shared ``metric_axes``
        shape. It contains no retained shuffle-draw axis.

    Returns
    -------
    dict[str, numpy.ndarray]
        Owned numeric/boolean arrays for every summary field. The mapping omits
        full null draws by contract; unavailable statistics remain NaN.

    Raises
    ------
    ValueError
        If a field has inconsistent axes or unsupported numeric/boolean dtype.
    """
    arrays = {
        "null_exceedance_count": summary.null_exceedance_count,
        "permutation_count": summary.permutation_count,
        "eligible_trial_count": summary.eligible_trial_count,
        "p_value": summary.p_value,
        "null_mean": summary.null_mean,
        "null_std": summary.null_std,
        "null_p025": summary.null_p025,
        "null_p50": summary.null_p50,
        "null_p975": summary.null_p975,
        "null_eligible": summary.null_eligible,
        "significant": summary.significant,
    }
    return _validated_summary_arrays(arrays)


def permutation_null_summary_from_arrays(
    arrays: Mapping[str, np.ndarray],
) -> PermutationNullSummary:
    """Reconstruct a null summary from cache-safe numeric/boolean arrays.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Named numeric/boolean arrays from ``permutation_null_summary_to_arrays``
        with a common ``metric_axes`` shape. NaNs retain unavailable statistics.

    Returns
    -------
    PermutationNullSummary
        Frozen summary with owned copies of every input array and no null draws.

    Raises
    ------
    ValueError
        If names are missing, dtypes are wrong, or metric axes are inconsistent.
    """
    validated = _validated_summary_arrays(arrays)
    return PermutationNullSummary(**validated)


def adjust_ppc_pvalues_bh(
    *,
    p_value: np.ndarray,
    null_eligible: np.ndarray,
) -> np.ndarray:
    """Apply SciPy Benjamini-Hochberg separately within each frequency family.

    Parameters
    ----------
    p_value : numpy.ndarray
        Float shape ``(unit, condition, site, epoch, frequency)`` uncorrected
        permutation p values in ``[0, 1]``; NaN denotes unavailable inference.
    null_eligible : numpy.ndarray
        Boolean array matching ``p_value``. Only finite eligible frequencies are
        passed to SciPy within each unit/condition/site/epoch family.

    Returns
    -------
    numpy.ndarray
        Owned float64 shape matching ``p_value`` of BH q values. Ineligible or
        missing p values stay NaN and are never silently treated as tests.

    Raises
    ------
    ValueError
        If axes mismatch, input is not five-dimensional, or finite p values lie
        outside ``[0, 1]``.
    """
    p_values = np.asarray(p_value, dtype=float)
    eligible = np.asarray(null_eligible, dtype=bool)
    if p_values.ndim != 5 or eligible.shape != p_values.shape:
        raise ValueError("p_value and null_eligible require matching five-dimensional axes")
    finite = np.isfinite(p_values)
    if np.any((p_values[finite] < 0.0) | (p_values[finite] > 1.0)):
        raise ValueError("finite p values must lie in [0, 1]")
    q_value = np.full(p_values.shape, np.nan, dtype=float)
    for family_index in np.ndindex(p_values.shape[:-1]):
        family_p = p_values[family_index]
        family_mask = eligible[family_index] & np.isfinite(family_p)
        if np.any(family_mask):
            q_value[family_index][family_mask] = false_discovery_control(
                family_p[family_mask],
                axis=0,
                method="bh",
            )
    return q_value


def build_shuffle_run_metadata(
    *,
    shuffle_count: int,
    seed: int,
    mode: str,
) -> Mapping[str, int | str]:
    """Build canonical preview/final shuffle metadata and its SHA256 fingerprint.

    Parameters
    ----------
    shuffle_count : int
        Number of schedule rows: exactly 100 for ``"preview"`` or 1000 for
        ``"final"``.
    seed : int
        Integer schedule seed with no physical units.
    mode : str
        Explicit categorical run mode, either ``"preview"`` or ``"final"``.

    Returns
    -------
    Mapping[str, int | str]
        Immutable metadata containing mode, count, seed, and SHA256 fingerprint.
        It has no missing-value representation.

    Raises
    ------
    ValueError
        If mode/count combinations are not approved or seed is not an integer.
    """
    expected_counts = {"preview": 100, "final": 1000}
    if mode not in expected_counts or shuffle_count != expected_counts[mode]:
        raise ValueError("preview requires 100 and final requires 1000 shuffles")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    payload = {
        "mode": mode,
        "seed": int(seed),
        "shuffle_count": int(shuffle_count),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    metadata: dict[str, int | str] = dict(payload)
    metadata["fingerprint"] = sha256(canonical.encode("ascii")).hexdigest()
    return MappingProxyType(metadata)


def compute_significant_prevalence(
    *,
    q_value: np.ndarray,
    null_eligible: np.ndarray,
    alpha: float,
) -> SignificantPrevalenceSummary:
    """Compute FDR-significant unit prevalence over eligible units only.

    Parameters
    ----------
    q_value : numpy.ndarray
        Float shape ``(unit, condition, site, epoch, frequency)`` BH q values.
        NaN means unavailable/ineligible inference.
    null_eligible : numpy.ndarray
        Boolean array matching ``q_value``; it defines the denominator.
    alpha : float
        Finite significance threshold in ``(0, 1)`` with no physical units.

    Returns
    -------
    SignificantPrevalenceSummary
        Prevalence and integer eligible/total counts shaped ``(condition, site,
        epoch, frequency)``. Zero eligible units produce NaN, never zero.

    Raises
    ------
    ValueError
        If axes mismatch, q values are out of range, or alpha is invalid.
    """
    q_values = np.asarray(q_value, dtype=float)
    eligible = np.asarray(null_eligible, dtype=bool)
    if q_values.ndim != 5 or eligible.shape != q_values.shape:
        raise ValueError("q_value and null_eligible require matching five-dimensional axes")
    if not np.isfinite(alpha) or not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    finite = np.isfinite(q_values)
    if np.any((q_values[finite] < 0.0) | (q_values[finite] > 1.0)):
        raise ValueError("finite q values must lie in [0, 1]")
    eligible_count = eligible.sum(axis=0, dtype=np.int64)
    total_count = np.full(eligible_count.shape, q_values.shape[0], dtype=np.int64)
    significant = eligible & finite & (q_values < float(alpha))
    significant_count = significant.sum(axis=0, dtype=np.int64)
    prevalence = np.full(eligible_count.shape, np.nan, dtype=float)
    np.divide(
        significant_count,
        eligible_count,
        out=prevalence,
        where=eligible_count > 0,
    )
    return SignificantPrevalenceSummary(
        prevalence=prevalence,
        eligible_unit_count=eligible_count,
        total_unit_count=total_count,
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


def _strict_relative_seconds(values: np.ndarray, name: str) -> np.ndarray:
    """Validate an owned, nonempty, strictly increasing relative-seconds axis.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate float ``(time,)`` coordinate in event-relative seconds.
    name : str
        Parameter name used for validation errors.

    Returns
    -------
    numpy.ndarray
        Owned float64 ``(time,)`` seconds coordinate without missing values.

    Raises
    ------
    ValueError
        If the coordinate is empty, nonfinite, non-one-dimensional, or not
        strictly increasing.
    """
    coordinate = np.asarray(values, dtype=float)
    if (
        coordinate.ndim != 1
        or coordinate.size == 0
        or not np.isfinite(coordinate).all()
        or np.any(np.diff(coordinate) <= 0.0)
    ):
        raise ValueError(f"{name} must be a finite strictly increasing seconds axis")
    return coordinate.copy()


def _trial_local_spike_arrays(values: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    """Copy finite relative-seconds spike arrays without merging trial membership.

    Parameters
    ----------
    values : Sequence[numpy.ndarray]
        One candidate float ``(spike_in_trial,)`` array per trial in relative
        seconds. Empty arrays mean a trial has no spike.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Owned float64 arrays in original trial order; no pooled or merged axis.

    Raises
    ------
    ValueError
        If fewer than two trials exist or an array is nonfinite/non-1D.
    """
    if len(values) < 2:
        raise ValueError("trial-shuffle PPC requires at least two trial-local trains")
    copied: list[np.ndarray] = []
    for trial_values in values:
        spikes = np.asarray(trial_values, dtype=float)
        if spikes.ndim != 1 or not np.isfinite(spikes).all():
            raise ValueError("trial-local spikes must be finite one-dimensional seconds")
        copied.append(spikes.copy())
    return tuple(copied)


def _unit_trial_local_spike_arrays(
    values: Sequence[Sequence[np.ndarray]],
    trial_count: int,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Validate and copy nested event-relative spike arrays by unit and trial.

    Parameters
    ----------
    values : Sequence[Sequence[numpy.ndarray]]
        Nested ``(unit, trial)`` finite one-dimensional seconds arrays. Empty
        arrays represent no spike and overlapping trial membership is retained.
    trial_count : int
        Positive required length of every unit's shared trial axis.

    Returns
    -------
    tuple[tuple[numpy.ndarray, ...], ...]
        Owned float64 spike arrays in supplied unit/trial order and unchanged
        seconds units.

    Raises
    ------
    ValueError
        If no unit is supplied, trial axes differ, or any spike array is not
        finite one-dimensional seconds data.
    """
    if not isinstance(trial_count, (int, np.integer)) or int(trial_count) < 1:
        raise ValueError("trial_count must be a positive integer")
    if len(values) < 1:
        raise ValueError("edge statistics require at least one unit")
    units: list[tuple[np.ndarray, ...]] = []
    for unit_values in values:
        if len(unit_values) != int(trial_count):
            raise ValueError("every unit must share the phase trial axis")
        trial_spikes: list[np.ndarray] = []
        for trial_values in unit_values:
            spikes = np.asarray(trial_values, dtype=float)
            if spikes.ndim != 1 or not np.isfinite(spikes).all():
                raise ValueError(
                    "unit-trial spikes must be finite one-dimensional seconds"
                )
            trial_spikes.append(spikes.copy())
        units.append(tuple(trial_spikes))
    return tuple(units)


def _edge_position_arrays(
    source_trial_position: np.ndarray,
    target_trial_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and copy one stable nonduplicate trial-edge coordinate.

    Parameters
    ----------
    source_trial_position, target_trial_position : numpy.ndarray
        Matching nonempty integer shape ``(edge,)`` zero-based trial positions.
        Same-trial edges are valid observed-statistic inputs.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Owned int64 edge coordinates in unchanged supplied order.

    Raises
    ------
    ValueError
        If axes/dtypes differ, positions are negative, or an edge is duplicated.
    """
    source = np.asarray(source_trial_position)
    target = np.asarray(target_trial_position)
    if (
        source.ndim != 1
        or target.ndim != 1
        or source.size < 1
        or source.shape != target.shape
        or not np.issubdtype(source.dtype, np.integer)
        or not np.issubdtype(target.dtype, np.integer)
        or np.any(source < 0)
        or np.any(target < 0)
    ):
        raise ValueError(
            "trial-edge positions must be matching nonnegative integer vectors"
        )
    source_int64 = source.astype(np.int64, copy=True)
    target_int64 = target.astype(np.int64, copy=True)
    edge_pairs = set(
        zip(source_int64.tolist(), target_int64.tolist(), strict=True)
    )
    if len(edge_pairs) != source_int64.size:
        raise ValueError("trial-edge positions must not contain duplicate pairs")
    return source_int64, target_int64


def _shuffle_position_array(values: np.ndarray, shuffle_count: int) -> np.ndarray:
    """Validate a stable block of unique schedule-row positions.

    Parameters
    ----------
    values : numpy.ndarray
        Nonempty integer shape ``(shuffle_block,)`` schedule-row positions.
    shuffle_count : int
        Positive size of the available schedule shuffle axis.

    Returns
    -------
    numpy.ndarray
        Owned int64 positions preserving supplied block order.

    Raises
    ------
    ValueError
        If values are noninteger, empty, duplicated, negative, or out of range.
    """
    positions = np.asarray(values)
    if (
        positions.ndim != 1
        or positions.size < 1
        or not np.issubdtype(positions.dtype, np.integer)
    ):
        raise ValueError("shuffle_positions must be a nonempty integer vector")
    positions_int64 = positions.astype(np.int64, copy=True)
    if (
        np.any(positions_int64 < 0)
        or np.any(positions_int64 >= int(shuffle_count))
        or np.unique(positions_int64).size != positions_int64.size
    ):
        raise ValueError("shuffle_positions must be unique valid schedule rows")
    return positions_int64


def _derangement_schedule(values: np.ndarray, trial_count: int) -> np.ndarray:
    """Validate and copy source-to-target trial derangement rows.

    Parameters
    ----------
    values : numpy.ndarray
        Candidate integer shape ``(shuffle, trial)`` schedule.
    trial_count : int
        Expected trial-axis size, at least two.

    Returns
    -------
    numpy.ndarray
        Owned int64 schedule where every row permutes all trial positions and
        has no fixed point.

    Raises
    ------
    ValueError
        If axes, dtype, row permutations, or fixed-point constraints fail.
    """
    schedule = np.asarray(values)
    if (
        schedule.ndim != 2
        or schedule.shape[0] < 1
        or schedule.shape[1] != trial_count
        or not np.issubdtype(schedule.dtype, np.integer)
    ):
        raise ValueError("schedule must be an integer array with shape (shuffle, trial)")
    identity = np.arange(trial_count)
    for row in schedule:
        if not np.array_equal(np.sort(row), identity) or np.any(row == identity):
            raise ValueError("every schedule row must be a derangement")
    return schedule.astype(np.int64, copy=True)


def _overlap_index_array(values: np.ndarray | None) -> np.ndarray:
    """Validate and copy optional overlapping source trial-row indices.

    Parameters
    ----------
    values : numpy.ndarray or None
        Integer ``(overlapping_trial,)`` trial-table rows, or ``None`` for none.

    Returns
    -------
    numpy.ndarray
        Owned int64 one-dimensional rows. An empty array represents no warning.

    Raises
    ------
    ValueError
        If supplied rows are not one-dimensional nonnegative integers.
    """
    if values is None:
        return np.empty(0, dtype=np.int64)
    indices = np.asarray(values)
    if (
        indices.ndim != 1
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
    ):
        raise ValueError("overlap_trial_indices must be nonnegative integer rows")
    return indices.astype(np.int64, copy=True)


def _precompute_trial_phase_samples(
    phase_time_s: np.ndarray,
    phase_vectors: np.ndarray,
    trial_spikes_s: tuple[np.ndarray, ...],
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Sample every source train on every target trial phase trace once.

    Parameters
    ----------
    phase_time_s : numpy.ndarray
        Float64 ``(time,)`` common relative-seconds coordinate.
    phase_vectors : numpy.ndarray
        Complex ``(trial, frequency, time)`` coefficient tensor.
    trial_spikes_s : tuple[numpy.ndarray, ...]
        Float64 relative-seconds ``(spike_in_trial,)`` arrays by source trial.

    Returns
    -------
    tuple[tuple[numpy.ndarray, ...], ...]
        ``(source_trial, target_trial)`` nested tuples of complex normalized
        ``(frequency, spike_in_source_trial)`` samples. Invalid samples are NaN.
    """
    sampled_rows: list[tuple[np.ndarray, ...]] = []
    for source_spikes in trial_spikes_s:
        target_samples = tuple(
            _interpolate_unit_phase(phase_time_s, phase_vectors[target], source_spikes)
            for target in range(phase_vectors.shape[0])
        )
        sampled_rows.append(target_samples)
    return tuple(sampled_rows)


def _interpolate_unit_phase(
    source_time_s: np.ndarray,
    coefficients: np.ndarray,
    target_time_s: np.ndarray,
) -> np.ndarray:
    """Interpolate complex coefficients by components then normalize unit phase.

    Parameters
    ----------
    source_time_s : numpy.ndarray
        Float64 ``(time,)`` strictly increasing relative seconds.
    coefficients : numpy.ndarray
        Complex ``(frequency, time)`` coefficients matching source time.
    target_time_s : numpy.ndarray
        Float64 ``(spike,)`` relative seconds; outside support is unavailable.

    Returns
    -------
    numpy.ndarray
        Complex128 ``(frequency, spike)`` unit vectors. Nonfinite, zero-magnitude,
        and out-of-support samples are complex NaN. An exact source sample uses
        that sample's validity; an in-between sample requires two adjacent
        finite nonzero source coefficients.
    """
    source_time = np.asarray(source_time_s, dtype=float)
    coefficient_array = np.asarray(coefficients)
    target_time = np.asarray(target_time_s, dtype=float)
    if (
        source_time.ndim != 1
        or source_time.size < 1
        or not np.isfinite(source_time).all()
        or np.any(np.diff(source_time) <= 0.0)
        or coefficient_array.ndim != 2
        or coefficient_array.shape[1] != source_time.size
        or not np.issubdtype(coefficient_array.dtype, np.number)
        or target_time.ndim != 1
        or not np.isfinite(target_time).all()
    ):
        raise ValueError("phase interpolation requires valid frequency/time axes")

    output = np.full(
        (coefficient_array.shape[0], target_time.size),
        np.nan + 1j * np.nan,
    )
    source_valid = (
        np.isfinite(coefficient_array.real)
        & np.isfinite(coefficient_array.imag)
        & (np.abs(coefficient_array) > 0.0)
    )
    right_position = np.searchsorted(source_time, target_time, side="left")
    inside_right_support = right_position < source_time.size
    exact = np.zeros(target_time.shape, dtype=bool)
    exact[inside_right_support] = (
        source_time[right_position[inside_right_support]]
        == target_time[inside_right_support]
    )
    between = (
        ~exact
        & (right_position > 0)
        & (right_position < source_time.size)
    )
    support_valid = np.zeros(output.shape, dtype=bool)
    if np.any(exact):
        support_valid[:, exact] = source_valid[:, right_position[exact]]
    if np.any(between):
        left_position = right_position[between] - 1
        support_valid[:, between] = (
            source_valid[:, left_position]
            & source_valid[:, right_position[between]]
        )

    for frequency_index, row in enumerate(coefficient_array):
        real = np.interp(target_time, source_time, row.real, left=np.nan, right=np.nan)
        imaginary = np.interp(
            target_time,
            source_time,
            row.imag,
            left=np.nan,
            right=np.nan,
        )
        combined = real + 1j * imaginary
        magnitude = np.abs(combined)
        usable = (
            support_valid[frequency_index]
            & np.isfinite(magnitude)
            & (magnitude > 0.0)
        )
        output[frequency_index] = np.divide(
            combined,
            magnitude,
            out=np.full(target_time.size, np.nan + 1j * np.nan),
            where=usable,
        )
    return output


def _pooled_sampled_phase_vectors(
    sampled: tuple[tuple[np.ndarray, ...], ...],
    target_trials: np.ndarray,
) -> np.ndarray:
    """Pool one source train's sampled phases for each source-to-target mapping.

    Parameters
    ----------
    sampled : tuple[tuple[numpy.ndarray, ...], ...]
        Nested ``(source_trial, target_trial)`` normalized phase samples with
        arrays shaped ``(frequency, spike_in_source_trial)``.
    target_trials : numpy.ndarray
        Int64 ``(trial,)`` target phase-trial position for every source trial.

    Returns
    -------
    numpy.ndarray
        Complex ``(frequency, pooled_spike)`` concatenation. Empty source trains
        contribute zero columns and preserve no merged interval representation.
    """
    parts = [
        sampled[source_index][int(target_index)]
        for source_index, target_index in enumerate(target_trials)
    ]
    return np.concatenate(parts, axis=1)


def _phase_metrics_from_vectors(
    phase_vectors: np.ndarray,
    frequencies_hz: np.ndarray,
) -> spike_lfp_phase_locking.SpikePhaseLockingResult:
    """Delegate pooled normalized phase PPC to the established project helper.

    Parameters
    ----------
    phase_vectors : numpy.ndarray
        Complex ``(frequency, spike)`` normalized samples; complex NaN means
        unavailable phase observation.
    frequencies_hz : numpy.ndarray
        Float64 ``(frequency,)`` Morlet coordinates in Hz.

    Returns
    -------
    SpikePhaseLockingResult
        Existing-helper PPC/circular metrics with dimensionless PPC and counts.
        Missing phase values are excluded by its validity mask.
    """
    valid = np.isfinite(phase_vectors.real) & np.isfinite(phase_vectors.imag)
    spike_times = np.arange(phase_vectors.shape[1], dtype=float)
    return spike_lfp_phase_locking.compute_frequency_phase_metrics(
        spike_phase_vectors=phase_vectors,
        valid_mask=valid,
        frequencies_hz=frequencies_hz,
        spike_times_s=spike_times,
        unit_id=0,
        lfp_site_label="trial_shuffle",
    )


def _null_chunk_arrays(
    values: Sequence[np.ndarray],
    metric_shape: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Validate temporary null chunks with a leading shuffle axis.

    Parameters
    ----------
    values : Sequence[numpy.ndarray]
        Candidate float arrays shaped ``(shuffle_chunk,) + metric_axes``.
    metric_shape : tuple[int, ...]
        Required observed-PPC axes after the leading shuffle axis.

    Returns
    -------
    tuple[numpy.ndarray, ...]
        Owned float64 chunks. Nonfinite draw values remain as missing null draws.

    Raises
    ------
    ValueError
        If no chunk exists, a chunk has no rows, or axes do not match.
    """
    if not values:
        raise ValueError("null_ppc_chunks must contain at least one chunk")
    chunks: list[np.ndarray] = []
    for chunk in values:
        array = np.asarray(chunk, dtype=float)
        if array.ndim != len(metric_shape) + 1 or array.shape[1:] != metric_shape:
            raise ValueError("null chunks require shape (shuffle,) + observed_ppc.shape")
        if array.shape[0] < 1:
            raise ValueError("null chunks must contain at least one shuffle row")
        chunks.append(array.copy())
    return tuple(chunks)


def _eligible_trial_counts(
    values: int | np.ndarray,
    metric_shape: tuple[int, ...],
) -> np.ndarray:
    """Validate/broadcast spike-contributing trial counts to metric axes.

    Parameters
    ----------
    values : int or numpy.ndarray
        Nonnegative scalar or integer ``metric_axes`` count of trials with at
        least one valid same-trial spike phase at each frequency/metric cell.
    metric_shape : tuple[int, ...]
        Target observed-PPC axes, typically ``(frequency,)``.

    Returns
    -------
    numpy.ndarray
        Owned int64 ``metric_axes`` count array. No missing count sentinel is
        accepted because ineligibility is represented by values below two.

    Raises
    ------
    ValueError
        If counts are noninteger, negative, or incompatible with metric axes.
    """
    if isinstance(values, (int, np.integer)):
        if int(values) < 0:
            raise ValueError("eligible_trial_count must be nonnegative")
        return np.full(metric_shape, int(values), dtype=np.int64)
    counts = _count_array(values)
    if counts.shape != metric_shape:
        raise ValueError("eligible_trial_count must match observed PPC axes")
    return counts


def _spike_contributing_trial_count(
    sampled: tuple[tuple[np.ndarray, ...], ...],
) -> np.ndarray:
    """Count trials with a valid same-trial phase sample per frequency.

    Parameters
    ----------
    sampled : tuple[tuple[numpy.ndarray, ...], ...]
        ``(source_trial, target_trial)`` normalized complex samples shaped
        ``(frequency, spike_in_source_trial)`` from the precomputation cache.

    Returns
    -------
    numpy.ndarray
        Owned int64 ``(frequency,)`` count of source trials contributing at
        least one finite phase to the observed same-trial PPC. Empty/nonfinite
        trial samples contribute zero and are not eligible trial evidence.
    """
    frequency_count = sampled[0][0].shape[0]
    counts = np.zeros(frequency_count, dtype=np.int64)
    for trial_index, target_samples in enumerate(sampled):
        same_trial_samples = target_samples[trial_index]
        valid = np.isfinite(same_trial_samples.real) & np.isfinite(
            same_trial_samples.imag
        )
        counts += np.any(valid, axis=1)
    return counts


def _null_distribution_statistics(
    draws: np.ndarray,
    finite_draw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate finite-null moments and exact percentiles without returning draws.

    Parameters
    ----------
    draws : numpy.ndarray
        Float ``(shuffle,) + metric_axes`` temporary null PPC draws.
    finite_draw : numpy.ndarray
        Boolean mask matching ``draws`` for finite null values.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Float64 ``metric_axes`` null mean, population std, and 2.5/50/97.5
        percentiles. A cell with no finite draw is NaN in all fields.
    """
    output_shape = draws.shape[1:]
    mean = np.full(output_shape, np.nan, dtype=float)
    std = np.full(output_shape, np.nan, dtype=float)
    p025 = np.full(output_shape, np.nan, dtype=float)
    p50 = np.full(output_shape, np.nan, dtype=float)
    p975 = np.full(output_shape, np.nan, dtype=float)
    for index in np.ndindex(output_shape):
        values = draws[(slice(None),) + index]
        values = values[finite_draw[(slice(None),) + index]]
        if values.size:
            mean[index] = np.mean(values)
            std[index] = np.std(values)
            p025[index], p50[index], p975[index] = np.percentile(
                values,
                [2.5, 50.0, 97.5],
                method="linear",
            )
    return mean, std, p025, p50, p975


def _validated_summary_arrays(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Validate/copy the numeric cache representation of a null summary.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Named count, float statistic, and boolean eligibility arrays sharing
        ``metric_axes``. Full shuffle draws are unsupported by this contract.

    Returns
    -------
    dict[str, numpy.ndarray]
        Owned canonical int64, float64, and boolean arrays for reconstruction.

    Raises
    ------
    ValueError
        If required names, supported dtypes, nonnegative counts, or axes fail.
    """
    count_names = (
        "null_exceedance_count",
        "permutation_count",
        "eligible_trial_count",
    )
    float_names = ("p_value", "null_mean", "null_std", "null_p025", "null_p50", "null_p975")
    bool_names = ("null_eligible", "significant")
    required_names = set(count_names + float_names + bool_names)
    if set(arrays) != required_names:
        raise ValueError("null summary arrays must contain exactly the required fields")
    copied: dict[str, np.ndarray] = {}
    common_shape: tuple[int, ...] | None = None
    for name in count_names:
        value = _count_array(arrays[name])
        copied[name] = value
        common_shape = _matching_shape(common_shape, value.shape, name)
    for name in float_names:
        value = np.asarray(arrays[name], dtype=float)
        copied[name] = value.copy()
        common_shape = _matching_shape(common_shape, value.shape, name)
    for name in bool_names:
        value = np.asarray(arrays[name])
        if value.dtype != bool:
            raise ValueError(f"{name} must be a boolean array")
        copied[name] = value.copy()
        common_shape = _matching_shape(common_shape, value.shape, name)
    return copied


def _matching_shape(
    expected: tuple[int, ...] | None,
    actual: tuple[int, ...],
    name: str,
) -> tuple[int, ...]:
    """Check one cache field shape against the first observed metric axes.

    Parameters
    ----------
    expected : tuple[int, ...] or None
        First field's metric axes, or ``None`` before validation begins.
    actual : tuple[int, ...]
        Current field's metric axes.
    name : str
        Field name used in an error message.

    Returns
    -------
    tuple[int, ...]
        Shared metric axes after accepting the current field.

    Raises
    ------
    ValueError
        If the current field has incompatible axes.
    """
    if expected is not None and actual != expected:
        raise ValueError(f"{name} must match null summary metric axes")
    return actual
