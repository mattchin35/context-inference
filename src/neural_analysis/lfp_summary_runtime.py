"""Production bridge for offline Power-summary preparation, calculation, and caching.

The bridge intentionally has no Streamlit dependency. It loads native-rate LFP
traces through the preparation adapters, performs PSD calculations before
decimation, and stores only the documented 500-Hz inspection traces in payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy import signal

from src.neural_analysis.lfp_power_summary import (
    compute_presession_reference_psd,
    compute_session_reference_psd,
    compute_trial_epoch_psds,
    interpolate_linear_psd_to_canonical_grid,
    mean_band_power_linear,
    normalize_psd_db,
)
from src.neural_analysis.lfp_summary_io import (
    load_or_initialize_manifest,
    write_component_transaction,
)
from src.neural_analysis.lfp_summary_models import LFPSummaryConfig, LFPSiteConfig
from src.neural_analysis.lfp_summary_payloads import build_component_payload
from src.neural_analysis.lfp_summary_pipeline import ComponentPayload, PipelineDependencies
from src.neural_analysis.lfp_summary_preparation import (
    PreparedSiteTraces,
    PreparedTrials,
    build_prepared_trials,
    load_site_trial_traces,
)


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


def prepare_power_run(
    config: LFPSummaryConfig,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> PreparedPowerRun:
    """Load active trials, native whole-window traces, and exact pre-session data.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated session configuration. Windows are seconds and LFP rates are
        samples/second; site traces retain each source voltage unit.
    trial_table_loader : callable
        Production or injected loader called once with ``config``. It returns a
        pandas table with one row per trial and documented behavior columns.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized trace seams forwarded to
        :func:`load_site_trial_traces`. Each accepts keyword ``site``, absolute
        ``alignment_time_s``, and event-relative ``window`` in seconds.

    Returns
    -------
    PreparedPowerRun
        Frozen native-rate preparation. The pre-session trace is available only
        when the full configured interval immediately before the first
        ``start_time`` loads on the exact expected grid.
    """
    trial_table = trial_table_loader(config)
    if not isinstance(trial_table, pd.DataFrame):
        raise ValueError("trial_table_loader must return a pandas DataFrame")
    trial_indices = np.arange(len(trial_table), dtype=np.int64)
    alignment_times_s = _alignment_times(trial_table, config)
    site_traces = load_site_trial_traces(
        config.sites,
        trial_indices,
        alignment_times_s,
        (config.analysis_windows.whole_start_s, config.analysis_windows.whole_stop_s),
        spikeglx_loader=spikeglx_loader,
        open_ephys_loader=open_ephys_loader,
    )
    prepared_trials = build_prepared_trials(
        trial_table,
        config.trial_filter,
        config.analysis_windows.alignment_event,
        site_validity={site_id: traces.valid for site_id, traces in site_traces.items()},
    )
    first_start_time_s = _first_start_time(trial_table)
    presession_traces, presession_time_s = _load_presession_traces(
        config,
        first_start_time_s,
        spikeglx_loader,
        open_ephys_loader,
    )
    return PreparedPowerRun(
        trial_indices=trial_indices,
        prepared_trials=prepared_trials,
        site_traces=site_traces,
        presession_traces=presession_traces,
        presession_time_s=presession_time_s,
        first_start_time_s=first_start_time_s,
    )


def build_power_payload(
    config: LFPSummaryConfig,
    prepared: PreparedPowerRun,
) -> ComponentPayload:
    """Compute a complete cache-ready Power payload from native-rate preparation.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable Power settings. PSDs use native samples, seconds, Hz, and
        each site's unchanged source voltage unit.
    prepared : PreparedPowerRun
        Frozen trial masks, whole-window traces, and pre-session inputs.

    Returns
    -------
    ComponentPayload
        Full ``power`` schema arrays. PSD axes are `(site, trial, epoch,
        frequency)` in source-voltage-unit-squared/Hz. Cached source traces use
        exact 500-Hz `(site, trial, time)` samples without filling invalid NaNs.
    """
    _validate_prepared_power_run(config, prepared)
    site_results = _compute_site_power_results(config, prepared)
    frequency_hz = _common_frequency_grid(site_results)
    arrays = _assemble_power_arrays(config, prepared, site_results, frequency_hz)
    return build_component_payload("power", arrays)


def make_power_pipeline_dependencies(
    *,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> PipelineDependencies:
    """Bind the Power pipeline to real preparation, payload, and atomic I/O seams.

    Parameters
    ----------
    trial_table_loader, spikeglx_loader, open_ephys_loader : callable
        Same production/injected seams accepted by :func:`prepare_power_run`.

    Returns
    -------
    PipelineDependencies
        Real Power preparation, numerical payload construction, manifest loading,
        and atomic component transactions. Synchrony and spike-phase operations
        explicitly raise until their production bridges are supplied.
    """
    active_config: LFPSummaryConfig | None = None

    def prepare_power(config: LFPSummaryConfig) -> PreparedPowerRun:
        """Prepare native Power inputs with the factory's fixed loader seams."""
        nonlocal active_config
        active_config = config
        return prepare_power_run(
            config,
            trial_table_loader,
            spikeglx_loader,
            open_ephys_loader,
        )

    def load_manifest(directory: Path) -> dict[str, object]:
        """Load the manifest for the config accepted by the preceding prepare stage.

        Parameters
        ----------
        directory : pathlib.Path
            Cache directory supplied unchanged by the component pipeline.

        Returns
        -------
        dict[str, object]
            JSON-ready cache manifest without any raw signal values.
        """
        if active_config is None:
            raise RuntimeError("Power manifest loading requires prepared configuration")
        return load_or_initialize_manifest(directory, active_config)

    def unsupported_phase(_: LFPSummaryConfig) -> object:
        """Reject unbound synchrony preparation instead of returning fake arrays."""
        raise NotImplementedError("synchrony preparation is unsupported by the Power runtime")

    def unsupported_spike(_: LFPSummaryConfig, __: object) -> object:
        """Reject unbound spike preparation instead of returning fake arrays."""
        raise NotImplementedError("spike preparation is unsupported by the Power runtime")

    def unsupported_synchrony(_: LFPSummaryConfig, __: object) -> ComponentPayload:
        """Reject unbound synchrony payload creation instead of returning fake arrays."""
        raise NotImplementedError("synchrony payload is unsupported by the Power runtime")

    def unsupported_spike_payload(
        _: LFPSummaryConfig,
        __: object,
        ___: object,
    ) -> ComponentPayload:
        """Reject unbound spike payload creation instead of returning fake arrays."""
        raise NotImplementedError("spike payload is unsupported by the Power runtime")

    return PipelineDependencies(
        prepare_power=prepare_power,
        prepare_phase=unsupported_phase,
        prepare_spike=unsupported_spike,
        build_power_payload=build_power_payload,
        build_synchrony_payload=unsupported_synchrony,
        build_spike_phase_payload=unsupported_spike_payload,
        load_manifest=load_manifest,
        write_component=write_component_transaction,
    )


@dataclass(frozen=True)
class _SitePowerResult:
    """Native site PSD products before cache-axis assembly."""

    site: LFPSiteConfig
    traces: PreparedSiteTraces
    frequency_hz: np.ndarray
    psd_linear: np.ndarray
    psd_valid: np.ndarray
    session_reference_psd_linear: np.ndarray
    presession_reference_psd_linear: np.ndarray
    presession_reference_available: bool


def _alignment_times(trial_table: pd.DataFrame, config: LFPSummaryConfig) -> np.ndarray:
    """Return one float `(trial,)` absolute-second alignment vector from the active table."""
    column = config.analysis_windows.alignment_event
    if column not in trial_table:
        raise ValueError("trial table lacks active alignment column")
    return pd.to_numeric(trial_table[column], errors="coerce").to_numpy(dtype=float)


def _first_start_time(trial_table: pd.DataFrame) -> float:
    """Return the earliest finite trial start in absolute seconds, or NaN when absent."""
    if "start_time" not in trial_table:
        return float("nan")
    starts = pd.to_numeric(trial_table["start_time"], errors="coerce").to_numpy(dtype=float)
    finite_starts = starts[np.isfinite(starts)]
    return float(np.min(finite_starts)) if finite_starts.size else float("nan")


def _load_presession_traces(
    config: LFPSummaryConfig,
    first_start_time_s: float,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load one exact pre-session interval through the established trace adapters."""
    duration_s = config.power.presession_reference_duration_s
    if np.isfinite(first_start_time_s):
        loaded = load_site_trial_traces(
            config.sites,
            np.array([0], dtype=np.int64),
            np.array([first_start_time_s], dtype=float),
            (-duration_s, 0.0),
            spikeglx_loader=spikeglx_loader,
            open_ephys_loader=open_ephys_loader,
        )
    else:
        loaded = {}
    traces: dict[str, np.ndarray] = {}
    time_s: dict[str, np.ndarray] = {}
    for site in config.sites:
        if site.stable_id in loaded:
            trace = loaded[site.stable_id]
            traces[site.stable_id] = trace.source_trace[0].copy()
            time_s[site.stable_id] = trace.relative_time_s.copy()
        else:
            count = _sample_count(duration_s, site.sample_rate_hz)
            time_s[site.stable_id] = -duration_s + np.arange(count) / site.sample_rate_hz
            traces[site.stable_id] = np.full(count, np.nan, dtype=float)
    return traces, time_s


def _validate_prepared_power_run(config: LFPSummaryConfig, prepared: PreparedPowerRun) -> None:
    """Validate immutable Power preparation dimensions without changing arrays or units."""
    if not isinstance(prepared, PreparedPowerRun):
        raise ValueError("prepared must be a PreparedPowerRun")
    trial_count = prepared.trial_indices.size
    valid_indices = prepared.trial_indices.ndim == 1 and np.issubdtype(
        prepared.trial_indices.dtype,
        np.integer,
    )
    if not valid_indices:
        raise ValueError("prepared trial indices must be one-dimensional integers")
    if prepared.prepared_trials.condition_membership.shape[0] != trial_count:
        raise ValueError("prepared trial masks must share the trial axis")
    for site in config.sites:
        traces = prepared.site_traces.get(site.stable_id)
        if traces is None or traces.source_trace.shape[0] != trial_count:
            raise ValueError("prepared site traces must share the trial axis")


def _compute_site_power_results(
    config: LFPSummaryConfig,
    prepared: PreparedPowerRun,
) -> tuple[_SitePowerResult, ...]:
    """Compute native PSDs and references per site before common-cache interpolation."""
    results: list[_SitePowerResult] = []
    for site in config.sites:
        traces = prepared.site_traces[site.stable_id]
        trial_psd = compute_trial_epoch_psds(
            traces.source_trace,
            traces.relative_time_s,
            traces.sample_rate_hz,
            config.analysis_windows,
            config.power,
        )
        valid_whole = (
            prepared.prepared_trials.objective_valid
            & ~prepared.prepared_trials.user_excluded
            & traces.valid
            & trial_psd.psd_valid[:, 0]
        )
        session_reference = compute_session_reference_psd(trial_psd.psd_linear[:, 0], valid_whole)
        presession_time = prepared.presession_time_s[site.stable_id]
        if np.isfinite(prepared.first_start_time_s):
            absolute_presession_time = presession_time + prepared.first_start_time_s
            _, presession_reference, presession_available = compute_presession_reference_psd(
                absolute_presession_time,
                prepared.presession_traces[site.stable_id],
                prepared.first_start_time_s,
                traces.sample_rate_hz,
                config.power,
            )
        else:
            presession_reference = np.full(trial_psd.frequency_hz.shape, np.nan)
            presession_available = False
        results.append(
            _SitePowerResult(
                site,
                traces,
                trial_psd.frequency_hz,
                trial_psd.psd_linear,
                trial_psd.psd_valid,
                session_reference,
                presession_reference,
                presession_available,
            )
        )
    return tuple(results)


def _common_frequency_grid(results: tuple[_SitePowerResult, ...]) -> np.ndarray:
    """Return the shared 2-Hz grid bounded by the lowest native Nyquist frequency."""
    maximum_hz = min(result.frequency_hz[-1] for result in results)
    first = results[0].frequency_hz
    return first[first <= maximum_hz].copy()


def _assemble_power_arrays(
    config: LFPSummaryConfig,
    prepared: PreparedPowerRun,
    results: tuple[_SitePowerResult, ...],
    frequency_hz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Assemble all Power-schema arrays while preserving named axes and source units."""
    trial_count = prepared.trial_indices.size
    site_count = len(results)
    epoch_count = 3
    band_count = len(config.power.bands)
    psd = np.stack([
        _interpolate_result(result.psd_linear, result.frequency_hz, frequency_hz)
        for result in results
    ])
    psd_valid = np.isfinite(psd).all(axis=-1)
    session_reference = np.stack(
        [
            _interpolate_result(
                result.session_reference_psd_linear,
                result.frequency_hz,
                frequency_hz,
            )
            for result in results
        ]
    )
    presession_reference = np.stack(
        [
            _interpolate_result(
                result.presession_reference_psd_linear,
                result.frequency_hz,
                frequency_hz,
            )
            for result in results
        ]
    )
    normalized_session = np.stack(
        [normalize_psd_db(psd[index], session_reference[index]) for index in range(site_count)]
    )
    normalized_presession = np.stack(
        [normalize_psd_db(psd[index], presession_reference[index]) for index in range(site_count)]
    )
    band_power = np.full((site_count, trial_count, epoch_count, band_count), np.nan)
    band_session_db = band_power.copy()
    band_presession_db = band_power.copy()
    for site_index, result in enumerate(results):
        for band_index, band in enumerate(config.power.bands):
            values, _ = mean_band_power_linear(psd[site_index], frequency_hz, band)
            session_value, _ = mean_band_power_linear(
                session_reference[site_index],
                frequency_hz,
                band,
            )
            presession_value, _ = mean_band_power_linear(
                presession_reference[site_index],
                frequency_hz,
                band,
            )
            band_power[site_index, :, :, band_index] = values
            band_session_db[site_index, :, :, band_index] = _normalize_scalar_db(
                values,
                session_value,
            )
            band_presession_db[site_index, :, :, band_index] = _normalize_scalar_db(
                values,
                presession_value,
            )
    site_valid = np.stack([result.traces.valid for result in results])
    objective_valid = np.broadcast_to(
        prepared.prepared_trials.objective_valid,
        (site_count, trial_count),
    ).copy()
    exclusion_reason = _exclusion_codes(prepared, results)
    effective_counts = _effective_condition_counts(prepared, site_valid)
    cached_time, cached_source = _cached_source_traces(results)
    return {
        "trial_indices": prepared.trial_indices.astype(np.int64, copy=True),
        "site_ids": np.asarray(
            [result.site.stable_id for result in results],
            dtype="<U64",
        ),
        "site_voltage_units": np.asarray(
            [result.traces.voltage_unit for result in results],
            dtype="<U64",
        ),
        "condition_names": np.asarray(prepared.prepared_trials.condition_names, dtype="<U64"),
        "condition_membership": prepared.prepared_trials.condition_membership.copy(),
        "condition_trial_count": prepared.prepared_trials.condition_membership.sum(
            axis=0
        ).astype(np.int64),
        "condition_effective_trial_count": effective_counts,
        "condition_unstable": effective_counts < 10,
        "frequency_hz": frequency_hz.astype(float, copy=True),
        "epoch_names": np.asarray(("whole", "before", "after"), dtype="<U16"),
        "objective_valid": objective_valid,
        "site_valid": site_valid,
        "user_excluded": prepared.prepared_trials.user_excluded.copy(),
        "exclusion_reason_code": exclusion_reason,
        "psd_linear": psd,
        "psd_valid": psd_valid,
        "session_reference_psd_linear": session_reference,
        "presession_reference_psd_linear": presession_reference,
        "presession_reference_available": np.asarray(
            [result.presession_reference_available for result in results],
            dtype=bool,
        ),
        "normalized_psd_session_db": normalized_session,
        "normalized_psd_presession_db": normalized_presession,
        "band_names": np.asarray([band.name for band in config.power.bands], dtype="<U64"),
        "band_power_linear": band_power,
        "band_power_session_db": band_session_db,
        "band_power_presession_db": band_presession_db,
        "trial_rms": np.stack([result.traces.rms for result in results]),
        "trial_peak_to_peak": np.stack([result.traces.peak_to_peak for result in results]),
        "relative_time_s": cached_time,
        "source_trace": cached_source,
    }


def _interpolate_result(
    values: np.ndarray,
    source_hz: np.ndarray,
    target_hz: np.ndarray,
) -> np.ndarray:
    """Interpolate a finite-or-NaN linear PSD result onto the shared cache grid."""
    return interpolate_linear_psd_to_canonical_grid(source_hz, values, target_hz)


def _normalize_scalar_db(values: np.ndarray, reference: float | np.ndarray) -> np.ndarray:
    """Return no-epsilon dB ratios for scalar band powers in unchanged source units."""
    numerator = np.asarray(values, dtype=float)
    denominator = np.asarray(reference, dtype=float)
    output = np.full(numerator.shape, np.nan, dtype=float)
    valid = (
        np.isfinite(numerator)
        & (numerator > 0)
        & np.isfinite(denominator)
        & (denominator > 0)
    )
    output[valid] = 10.0 * np.log10(numerator[valid] / denominator)
    return output


def _effective_condition_counts(prepared: PreparedPowerRun, site_valid: np.ndarray) -> np.ndarray:
    """Return `(condition, site)` contributing trial counts after all Power masks."""
    shared = (
        prepared.prepared_trials.condition_membership
        & prepared.prepared_trials.filter_membership[:, None]
        & prepared.prepared_trials.objective_valid[:, None]
        & ~prepared.prepared_trials.user_excluded[:, None]
    )
    counts = [
        np.count_nonzero(shared & site_valid[index, :, None], axis=0)
        for index in range(site_valid.shape[0])
    ]
    return np.stack(counts, axis=1).astype(np.int64)


def _exclusion_codes(
    prepared: PreparedPowerRun,
    results: tuple[_SitePowerResult, ...],
) -> np.ndarray:
    """Merge objective, user, and site-local exclusions into `(site, trial)` codes."""
    rows: list[np.ndarray] = []
    for result in results:
        codes = result.traces.exclusion_reason.astype("<U32", copy=True)
        objective = prepared.prepared_trials.objective_exclusion_reason
        user = prepared.prepared_trials.user_exclusion_reason
        codes[~prepared.prepared_trials.objective_valid] = objective[
            ~prepared.prepared_trials.objective_valid
        ]
        needs_user_code = (codes == "") & prepared.prepared_trials.user_excluded
        codes[needs_user_code] = user[needs_user_code]
        rows.append(codes)
    return np.stack(rows)


def _cached_source_traces(
    results: tuple[_SitePowerResult, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias and downsample whole traces without filling invalid rows."""
    cached: list[np.ndarray] = []
    time_s: np.ndarray | None = None
    for result in results:
        rate_hz = result.traces.sample_rate_hz
        ratio = rate_hz / _CACHE_SAMPLE_RATE_HZ
        factor = round(ratio)
        if not np.isclose(ratio, factor, rtol=0.0, atol=1e-9) or factor < 1:
            raise ValueError(
                "source traces require an integer native-to-500-Hz decimation factor"
            )
        selected_time = result.traces.relative_time_s[::factor]
        selected_trace = _resample_trace_rows(
            result.traces.source_trace,
            factor,
            selected_time.size,
        )
        if time_s is None:
            time_s = selected_time.copy()
        elif not np.array_equal(time_s, selected_time):
            raise ValueError("site source traces do not share an exact 500-Hz cache grid")
        cached.append(selected_trace.copy())
    if time_s is None:
        raise ValueError("Power preparation requires at least one site")
    return time_s, np.stack(cached)


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
