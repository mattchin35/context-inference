"""Shared prepared records and validation for LFP-summary runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from src.neural_analysis.lfp_summary.models import LFPSummaryConfig, validate_lfp_summary_config
from src.neural_analysis.lfp_summary.preparation import (
    PreparedSiteTraces,
    PreparedTrials,
    TrialRelativeSpikeTrains,
)
from src.neural_analysis.spike_lfp import ppc as spike_lfp_summary


_CACHE_SAMPLE_RATE_HZ = 500.0


@dataclass(frozen=True)
class PreparedPowerRun:
    """Immutable native-rate inputs for one Power component execution.

    Attributes
    ----------
    trial_indices : numpy.ndarray
        Int64 `(trial,)` row positions in the loaded active trial table.
    prepared_trials : PreparedTrials
        Condition, filter, objective-validity, user-exclusion, and site masks.
    site_traces : dict[str, PreparedSiteTraces]
        Site-keyed native `(trial, time)` source-voltage traces for the whole
        event window. Invalid rows remain NaN and are never interpolated.
    presession_traces, presession_time_s : dict[str, numpy.ndarray]
        Site-keyed native `(sample,)` exact full-duration pre-session values and
        relative seconds coordinates. An unavailable baseline is NaN-valued.
    first_start_time_s : float
        Earliest finite trial ``start_time`` in absolute seconds, or NaN when
        the table cannot define the required pre-session interval.
    """

    trial_indices: np.ndarray
    prepared_trials: PreparedTrials
    site_traces: dict[str, PreparedSiteTraces]
    presession_traces: dict[str, np.ndarray]
    presession_time_s: dict[str, np.ndarray]
    first_start_time_s: float


@dataclass(frozen=True)
class PreparedPhaseRun:
    """Full-trial-axis phase and cached exemplar inputs for Synchrony.

    Attributes
    ----------
    trial_indices : numpy.ndarray
        Int64 shape ``(trial,)`` trial-table row positions.
    alignment_times_s : numpy.ndarray
        Float64 shape ``(trial,)`` absolute alignment-event seconds. Missing
        objective alignments are NaN and own no phase or trial-local spikes.
    prepared_trials : PreparedTrials
        Condition/filter/objective/user masks plus per-site/pair validity.
    phase_tensor : numpy.ndarray
        Complex64 unit phase with axes ``(site, frequency, trial, time)``.
        Invalid entries are zero and are interpreted only through
        ``phase_valid``.
    phase_valid : numpy.ndarray
        Boolean numerical validity with the same axes as ``phase_tensor``.
    relative_time_s : numpy.ndarray
        Exact 500-Hz shape ``(time,)`` half-open event-relative seconds.
    site_valid : numpy.ndarray
        Boolean shape ``(site, trial)`` retained continuous-transform trials.
    pair_valid : numpy.ndarray
        Boolean shape ``(pair, trial)`` ordered site-pair intersections.
    source_trace : numpy.ndarray
        Float shape ``(site, trial, time)`` unprocessed source-voltage samples
        anti-aliased onto the exact 500-Hz cache grid. Invalid rows are NaN.
    prepared_phase_cache_state : str or None
        ``"cold"`` when this invocation computed phase, ``"warm"`` when it
        loaded the exact prepared-phase cache, or ``None`` for injected legacy
        records that do not expose cache provenance. This field has no units.
    """

    trial_indices: np.ndarray
    alignment_times_s: np.ndarray
    prepared_trials: PreparedTrials
    phase_tensor: np.ndarray
    phase_valid: np.ndarray
    relative_time_s: np.ndarray
    site_valid: np.ndarray
    pair_valid: np.ndarray
    source_trace: np.ndarray
    prepared_phase_cache_state: str | None = None


@dataclass(frozen=True)
class PreparedSpikeRun:
    """Trial-local spike inputs for one configured stable unit population.

    Attributes
    ----------
    unit_ids : tuple[str, ...]
        Stable probe-qualified unit identities in configured order.
    population_ids : tuple[str, ...]
        One stable population label for the cache population axis.
    trial_spike_trains : tuple[TrialRelativeSpikeTrains, ...]
        One entry per unit. Each entry owns one event-relative seconds array per
        full trial-table row; invalid alignment rows contain empty arrays.
    """

    unit_ids: tuple[str, ...]
    population_ids: tuple[str, ...]
    trial_spike_trains: tuple[TrialRelativeSpikeTrains, ...]


def load_configured_trial_table(config: LFPSummaryConfig) -> pd.DataFrame:
    """Load the active session trial table from its explicit configured CSV path.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated summary configuration whose ``trial_table_path`` identifies
        the active CT026-style augmented trial CSV.

    Returns
    -------
    pandas.DataFrame
        One row per trial with values and column names preserved from the CSV.
        Time columns remain in their stored seconds until preparation validates
        and selects the active alignment event.

    Raises
    ------
    ValueError
        If the configuration does not identify a trial CSV or the CSV cannot be
        read as a tabular trial table.
    """
    if config.trial_table_path is None:
        raise ValueError("Power production runtime requires config.trial_table_path")
    try:
        return pd.read_csv(config.trial_table_path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError) as error:
        raise ValueError("unable to read configured trial-table CSV") from error


def _validate_composed_phase_amplitude_policy(config: LFPSummaryConfig) -> None:
    """Reject unsupported absolute phase-amplitude masking before production work.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable production configuration. Threshold values, when eventually
        supported by WP13, are expressed in each configured site's source
        voltage unit. No phase, spike, trial, or signal array is accepted here.

    Returns
    -------
    None
        Returns only when the complete configuration is valid and its absolute
        amplitude-threshold tuple is empty. The function does not open source
        files, create work caches, or transform data.

    Raises
    ------
    ValueError
        If general configuration validation fails or a nonempty absolute
        amplitude threshold requests the deferred WP13 masking behavior.
    """
    validate_lfp_summary_config(config)
    if config.phase.absolute_amplitude_thresholds:
        raise ValueError(
            "absolute amplitude thresholds are unsupported until WP13 is implemented"
        )


def _validate_prepared_phase_run(
    config: LFPSummaryConfig,
    prepared: PreparedPhaseRun,
) -> None:
    """Validate prepared Synchrony identities, dtypes, axes, seconds, and masks."""
    if not isinstance(prepared, PreparedPhaseRun):
        raise ValueError("prepared must be a PreparedPhaseRun")
    trial_count = prepared.trial_indices.size
    expected_phase = (
        len(config.sites),
        len(config.phase.frequency_hz),
        trial_count,
        prepared.relative_time_s.size,
    )
    if prepared.phase_tensor.shape != expected_phase:
        raise ValueError("prepared phase axes disagree with configuration")
    if prepared.phase_tensor.dtype != np.dtype(np.complex64):
        raise ValueError("prepared phase storage must remain complex64")
    if prepared.phase_valid.shape != expected_phase:
        raise ValueError("prepared phase validity axes are invalid")
    if prepared.site_valid.shape != (len(config.sites), trial_count):
        raise ValueError("prepared site validity axes are invalid")
    if prepared.pair_valid.shape != (len(config.site_pairs), trial_count):
        raise ValueError("prepared pair validity axes are invalid")
    expected_source = (len(config.sites), trial_count, prepared.relative_time_s.size)
    if prepared.source_trace.shape != expected_source:
        raise ValueError("prepared source trace axes are invalid")
    if not np.isfinite(prepared.relative_time_s).all():
        raise ValueError("prepared phase time must be finite seconds")
    if prepared.alignment_times_s.shape != (trial_count,):
        raise ValueError("prepared alignment times must share the trial axis")


def _validate_prepared_spike_run(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
) -> None:
    """Validate stable identities and full trial-local spike axes."""
    population = config.unit_population
    if not isinstance(prepared_spikes, PreparedSpikeRun) or population is None:
        raise ValueError("prepared spikes and configured population are required")
    if prepared_spikes.unit_ids != population.stable_unit_ids:
        raise ValueError("prepared unit order disagrees with configuration")
    if prepared_spikes.population_ids != (population.label,):
        raise ValueError("prepared population identity disagrees with configuration")
    trial_count = prepared_phase.trial_indices.size
    if len(prepared_spikes.trial_spike_trains) != len(prepared_spikes.unit_ids):
        raise ValueError("prepared spike train count must match unit identities")
    for unit_id, train in zip(
        prepared_spikes.unit_ids,
        prepared_spikes.trial_spike_trains,
        strict=True,
    ):
        if train.unit_id != unit_id or len(train.relative_spike_times) != trial_count:
            raise ValueError("prepared spike trains must share unit and trial axes")
    if config.ppc.minimum_computable_spikes != spike_lfp_summary.MINIMUM_COMPUTABLE_SPIKES:
        raise ValueError("runtime currently requires the established two-spike PPC threshold")
    if config.ppc.minimum_reliable_spikes != spike_lfp_summary.MINIMUM_RELIABLE_SPIKES:
        raise ValueError(
            "runtime currently requires the established 50-spike reliability threshold"
        )


def _analysis_condition_membership(prepared: PreparedTrials) -> np.ndarray:
    """Return trial-by-condition masks after shared filter/objective/user gates."""
    shared = (
        prepared.filter_membership
        & prepared.objective_valid
        & ~prepared.user_excluded
    )
    return prepared.condition_membership & shared[:, None]


def _work_fingerprint(value: object) -> str:
    """Return a stable SHA-256 identity for JSON-safe work-cache metadata."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _alignment_times(trial_table: pd.DataFrame, config: LFPSummaryConfig) -> np.ndarray:
    """Return one float `(trial,)` absolute-second alignment vector from the active table."""
    column = config.analysis_windows.alignment_event
    if column not in trial_table:
        raise ValueError("trial table lacks active alignment column")
    return pd.to_numeric(trial_table[column], errors="coerce").to_numpy(dtype=float)


def _resample_trace_rows(
    source_trace: np.ndarray,
    downsample_factor: int,
    expected_count: int,
) -> np.ndarray:
    """Polyphase-resample finite trace rows while retaining invalid rows as NaN.

    Parameters
    ----------
    source_trace : numpy.ndarray
        Float `(trial, native_time)` source-voltage array. A nonfinite row is
        invalid and must remain unavailable in the 500-Hz cache.
    downsample_factor : int
        Positive integer native-rate to 500-Hz reduction factor.
    expected_count : int
        Exact cached time-axis length expected from the half-open native grid.

    Returns
    -------
    numpy.ndarray
        Float `(trial, cached_time)` source-voltage array. Finite rows receive
        FIR anti-alias filtering through ``scipy.signal.resample_poly``; rows
        containing NaN never bridge their unavailable samples.
    """
    source = np.asarray(source_trace, dtype=float)
    if source.ndim != 2:
        raise ValueError("source traces must have axes (trial, native_time)")
    output = np.full((source.shape[0], expected_count), np.nan, dtype=float)
    for trial_index, row in enumerate(source):
        if not np.isfinite(row).all():
            continue
        resampled = signal.resample_poly(row, up=1, down=downsample_factor)
        if resampled.shape != (expected_count,):
            raise ValueError("polyphase resampling did not preserve the cached time axis")
        output[trial_index] = resampled
    return output


def _sample_count(duration_s: float, sample_rate_hz: float) -> int:
    """Return the exact integer sample count for one half-open seconds interval."""
    count = round(duration_s * sample_rate_hz)
    if not np.isclose(duration_s * sample_rate_hz, count, rtol=0.0, atol=1e-9):
        raise ValueError("pre-session duration must contain an integral sample count")
    return int(count)
