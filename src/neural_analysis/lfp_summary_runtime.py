"""Production bridge for offline Power and Synchrony preparation and caching.

The bridge intentionally has no Streamlit dependency. It loads native-rate LFP
traces through the preparation adapters, performs PSD calculations before
decimation, and stores only the documented 500-Hz inspection traces in payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd
from scipy import signal

from src.neural_analysis import (
    lfp_summary_ppc_runtime,
    spike_lfp_summary,
    spike_lfp_hilbert_phase,
)
from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp import phase as lfp_phase_clustering
from src.neural_analysis.lfp import spectrogram as lfp_spectrogram
from src.neural_analysis.lfp.power import (
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
from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    LFPSiteConfig,
    ProgressEvent,
    canonical_config_json,
    fingerprint_source_files,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary_payloads import build_component_payload
from src.neural_analysis.lfp_summary_work_cache import (
    PreparedPhaseCache,
    cleanup_ppc_run,
    load_prepared_phase_cache,
    write_prepared_phase_cache,
)
from src.neural_analysis.lfp_summary_pipeline import (
    ComponentPayload,
    PPCWorkCleanupTarget,
    PipelineDependencies,
)
from src.neural_analysis.lfp_summary_preparation import (
    PreparedSiteTraces,
    PreparedTrials,
    TrialRelativeSpikeTrains,
    build_common_event_grid,
    build_prepared_trials,
    build_trial_relative_spike_trains,
    load_site_trial_traces,
    preflight_open_ephys_site_metadata,
    validate_open_ephys_aligned_sync_path,
)
from src.neural_analysis.lfp.synchrony import (
    aggregate_trial_plv_bands,
    bootstrap_phase_clustering_bands,
    compute_phase_clustering_summary,
    compute_trial_plv_by_frequency,
)
from src.neural_analysis.spike_behavior import loading as spike_behavior_pynapple
from src.neural_analysis.spike_behavior import loading as unit_spike_loading


_CACHE_SAMPLE_RATE_HZ = 500.0


def _prepared_phase_work_metadata(
    config: LFPSummaryConfig,
    trial_indices: np.ndarray,
    alignment_times_s: np.ndarray,
) -> dict[str, object]:
    """Build deterministic work-cache metadata for one phase representation.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated configuration. Only transform-relevant settings enter the
        scientific cache identity; PPC execution block settings do not.
    trial_indices : numpy.ndarray
        Int64 shape ``(trial,)`` trial-table row positions on the cached phase
        axis.
    alignment_times_s : numpy.ndarray
        Float64 shape ``(trial,)`` absolute event times in seconds. These
        coordinates bind a cached trial axis to the current trial table.
        Finite seconds remain JSON numbers in the cache identity; unavailable
        or otherwise nonfinite entries are represented by reserved JSON
        strings at their original trial positions.

    Returns
    -------
    dict[str, object]
        JSON-safe metadata accepted by ``write_prepared_phase_cache``. Its
        fingerprints are stable across Python processes and reject changed
        sources, transform settings, trial layout, axes, shapes, or dtypes.
    """
    canonical_config = json.loads(canonical_config_json(config))
    scientific_inputs = {
        key: canonical_config[key]
        for key in (
            "session_id",
            "session_path",
            "trial_table_path",
            "sites",
            "trial_filter",
            "analysis_windows",
            "phase",
            "schema_version",
        )
    }
    # Cache metadata is strict JSON (``allow_nan=False``). Preserve the full
    # trial axis even when an objective alignment is unavailable, because its
    # position is part of the phase representation identity.
    encoded_alignment_times_s: list[float | str] = []
    for alignment_time_s in np.asarray(alignment_times_s, dtype=float):
        if np.isnan(alignment_time_s):
            encoded_alignment_times_s.append("__missing_alignment_nan__")
        elif np.isposinf(alignment_time_s):
            encoded_alignment_times_s.append("__nonfinite_alignment_posinf__")
        elif np.isneginf(alignment_time_s):
            encoded_alignment_times_s.append("__nonfinite_alignment_neginf__")
        else:
            encoded_alignment_times_s.append(float(alignment_time_s))
    trial_axis = {
        "trial_indices": np.asarray(trial_indices, dtype=np.int64).tolist(),
        "alignment_times_s": encoded_alignment_times_s,
    }
    source_fingerprint = _work_fingerprint(
        fingerprint_source_files(config, component="synchrony")
    )
    scientific_fingerprint = _work_fingerprint(
        {"transform_inputs": scientific_inputs, "trial_axis": trial_axis}
    )
    time_count = build_common_event_grid(
        config.analysis_windows.whole_start_s,
        config.analysis_windows.whole_stop_s,
        config.phase.output_rate_hz,
    ).size
    phase_shape = [
        len(config.sites),
        len(config.phase.frequency_hz),
        int(trial_indices.size),
        int(time_count),
    ]
    representation = {
        "generator": "prepare_phase_run",
        "analysis_version": "prepared-phase-cache-v1",
        "source_fingerprint": source_fingerprint,
        "scientific_fingerprint": scientific_fingerprint,
        "axes": ["site", "frequency", "trial", "time"],
        "shapes": {
            "phase": phase_shape,
            "valid": phase_shape,
            "site_trial_valid": [phase_shape[0], phase_shape[2]],
            "site_trial_exclusion_reason": [phase_shape[0], phase_shape[2]],
        },
        "dtypes": {
            "phase": "complex64",
            "valid": "bool",
            "site_trial_valid": "bool",
            "site_trial_exclusion_reason": "<U32",
        },
    }
    return {
        **representation,
        "schema_version": config.schema_version,
        "representation_fingerprint": _work_fingerprint(representation),
        "execution_settings": {"phase_storage": "complex64/bool"},
        "units": {
            "phase": "dimensionless",
            "relative_time_s": "s",
            "frequency_hz": "Hz",
        },
    }


def _work_fingerprint(value: object) -> str:
    """Return a stable SHA-256 identity for JSON-safe work-cache metadata."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _cached_phase_axes_match_current_run(
    cached: PreparedPhaseCache,
    config: LFPSummaryConfig,
    trial_indices: np.ndarray,
) -> bool:
    """Return whether a validated cache has this run's exact named axes.

    The work-cache loader checks array contracts and metadata. This additional
    check prevents reuse if a manually corrupted cache has correct metadata but
    different site, frequency, trial, or relative-time coordinates.
    """
    expected_time_s = build_common_event_grid(
        config.analysis_windows.whole_start_s,
        config.analysis_windows.whole_stop_s,
        config.phase.output_rate_hz,
    )
    return (
        np.array_equal(
            cached.axes["site_ids"],
            np.asarray([site.stable_id for site in config.sites], dtype="<U64"),
        )
        and np.array_equal(
            cached.axes["frequency_hz"],
            np.asarray(config.phase.frequency_hz, dtype=float),
        )
        and np.array_equal(cached.axes["trial_indices"], trial_indices)
        and np.array_equal(cached.axes["relative_time_s"], expected_time_s)
    )


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


@dataclass(frozen=True)
class _PPCPhaseJob:
    """One selected site/condition/epoch phase view for the PPC executor.

    ``phase_tensor`` and ``phase_valid`` have axes ``(site=1, frequency,
    selected_trial, time)``. Unit phase is complex64 and validity is Boolean;
    ``relative_time_s`` is the shared float64 event-relative seconds axis.
    ``trial_indices`` are stable int64 trial-table rows. The categorical job
    identity is deliberately retained after selecting the full phase tensor so
    work checkpoints cannot cross condition/site/epoch boundaries.
    """

    phase_tensor: np.ndarray
    phase_valid: np.ndarray
    relative_time_s: np.ndarray
    trial_indices: np.ndarray
    site_id: str
    condition_name: str
    epoch_bounds_s: tuple[float, float]
    source_fingerprint: str


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
    normalized_open_ephys_metadata = (
        preflight_open_ephys_site_metadata(config.sites)
        if open_ephys_loader is None
        else {}
    )
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
        open_ephys_metadata_by_site=normalized_open_ephys_metadata,
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
        normalized_open_ephys_metadata,
    )
    return PreparedPowerRun(
        trial_indices=trial_indices,
        prepared_trials=prepared_trials,
        site_traces=site_traces,
        presession_traces=presession_traces,
        presession_time_s=presession_time_s,
        first_start_time_s=first_start_time_s,
    )


def prepare_phase_run(
    config: LFPSummaryConfig,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    *,
    site_phase_tensor_builder: Callable[..., lfp_phase_clustering.PhaseTrialTensor] = (
        lfp_phase_clustering.compute_site_phase_trial_tensor
    ),
    block_loader_factory: Callable[
        [LFPSiteConfig],
        Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    ]
    | None = None,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    work_cache_root: Path | None = None,
) -> PreparedPhaseRun:
    """Prepare bounded continuous phase tensors on one exact full trial axis.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable phase settings. Window bounds are seconds, frequencies are
        Hz, and output rate is samples/second.
    trial_table_loader : callable
        Called once with ``config`` and returns one pandas row per trial.
    site_phase_tensor_builder : callable
        Existing/injected one-site bounded transform accepting the
        ``compute_site_phase_trial_tensor`` keyword contract.
    block_loader_factory : callable or None
        Optional site-to-continuous-loader seam. Each loader accepts absolute
        ``start_s, stop_s`` seconds and returns absolute time, source-voltage
        values, and native sample rate in Hz. ``None`` uses production files.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized per-trial loaders for cache exemplar source traces.
    work_cache_root : pathlib.Path or None, default=None
        Optional execution-only work-cache root. When supplied and
        ``prepared_phase_cache_enabled`` is true, the function loads an exact
        compatible read-only phase/valid mmap cache or atomically writes a cold
        transform. Trial, alignment, site, pair, and source-trace metadata are
        rebuilt on both cold and warm paths.

    Returns
    -------
    PreparedPhaseRun
        Complex64 phase and boolean validity on
        ``(site, frequency, trial, time)`` axes, pair-specific validity, and
        500-Hz source-voltage exemplar traces. No wavelet coefficients persist.

    Raises
    ------
    ValueError
        If the trial table, configured sites/pairs, transform axes, or common
        phase/source time grids are invalid.
    """
    # Preflight before the work-cache read so an invalid production sidecar
    # cannot be hidden behind a warm phase tensor.
    production_open_ephys_metadata_needed = (
        block_loader_factory is None or open_ephys_loader is None
    )
    normalized_open_ephys_metadata = (
        preflight_open_ephys_site_metadata(config.sites)
        if production_open_ephys_metadata_needed
        else {}
    )
    trial_table = trial_table_loader(config)
    if not isinstance(trial_table, pd.DataFrame):
        raise ValueError("trial_table_loader must return a pandas DataFrame")
    trial_indices = np.arange(len(trial_table), dtype=np.int64)
    alignment_times_s = _alignment_times(trial_table, config)
    preliminary_trials = build_prepared_trials(
        trial_table,
        config.trial_filter,
        config.analysis_windows.alignment_event,
    )
    transform_positions = np.flatnonzero(preliminary_trials.objective_valid)
    if not transform_positions.size:
        raise ValueError("phase preparation requires at least one finite alignment")
    factory = (
        partial(
            _preflighted_production_phase_block_loader_factory,
            normalized_open_ephys_metadata_by_site=normalized_open_ephys_metadata,
        )
        if block_loader_factory is None
        else block_loader_factory
    )
    frequency_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    whole_window = (
        config.analysis_windows.whole_start_s,
        config.analysis_windows.whole_stop_s,
    )
    cache_metadata = _prepared_phase_work_metadata(
        config,
        trial_indices,
        alignment_times_s,
    )
    cached: PreparedPhaseCache | None = None
    if work_cache_root is not None and config.ppc_execution.prepared_phase_cache_enabled:
        cached = load_prepared_phase_cache(work_cache_root, cache_metadata)
        if cached is not None and not _cached_phase_axes_match_current_run(
            cached,
            config,
            trial_indices,
        ):
            cached = None
    site_tensors = []
    if cached is None:
        for site in config.sites:
            tensor = site_phase_tensor_builder(
                event_times_s=alignment_times_s[transform_positions],
                trial_indices=trial_indices[transform_positions],
                block_loader=factory(site),
                frequencies_hz=frequency_hz,
                window=whole_window,
                output_sample_rate_hz=config.phase.output_rate_hz,
                gaussian_width=config.phase.morlet_gaussian_width,
                window_length=config.phase.morlet_window_length,
                precision=config.phase.morlet_precision,
                norm=config.phase.morlet_normalization,
                notch_60_hz=config.phase.notch_enabled,
                notch_quality_factor=config.phase.notch_quality_factor,
                minimum_relative_magnitude=config.phase.numerical_amplitude_threshold,
                maximum_core_duration_s=config.phase.block_duration_s,
            )
            _validate_site_phase_tensor(tensor, frequency_hz.size)
            site_tensors.append(tensor)
        relative_time_s, phase_tensor, phase_valid, site_valid = _full_phase_axes(
            site_tensors,
            trial_indices,
            frequency_hz.size,
        )
    else:
        relative_time_s = cached.axes["relative_time_s"]
        phase_tensor = cached.phase
        phase_valid = cached.valid
        site_valid = cached.axes["site_trial_valid"]
    canonical_time_s = build_common_event_grid(
        whole_window[0],
        whole_window[1],
        config.phase.output_rate_hz,
    )
    if not np.allclose(relative_time_s, canonical_time_s, rtol=0.0, atol=1e-12):
        raise ValueError("phase transform did not return the canonical output grid")
    relative_time_s = canonical_time_s
    if (
        cached is None
        and work_cache_root is not None
        and config.ppc_execution.prepared_phase_cache_enabled
    ):
        axes = {
            "site_ids": np.asarray(
                [site.stable_id for site in config.sites], dtype="<U64"
            ),
            "frequency_hz": frequency_hz.copy(),
            "trial_indices": trial_indices.copy(),
            "relative_time_s": relative_time_s.copy(),
            "site_trial_valid": site_valid.copy(),
            "site_trial_exclusion_reason": np.where(
                site_valid,
                "",
                "phase_unavailable",
            ).astype("<U32"),
        }
        write_prepared_phase_cache(
            work_cache_root,
            cache_metadata,
            phase_tensor,
            phase_valid,
            axes,
        )
    pair_indices = _configured_pair_indices(config)
    pair_valid = np.stack(
        [site_valid[site_a] & site_valid[site_b] for site_a, site_b in pair_indices]
    )
    prepared_trials = build_prepared_trials(
        trial_table,
        config.trial_filter,
        config.analysis_windows.alignment_event,
        site_validity={
            site.stable_id: site_valid[index]
            for index, site in enumerate(config.sites)
        },
    )
    native_traces = load_site_trial_traces(
        config.sites,
        trial_indices,
        alignment_times_s,
        whole_window,
        spikeglx_loader=spikeglx_loader,
        open_ephys_loader=open_ephys_loader,
        open_ephys_metadata_by_site=normalized_open_ephys_metadata,
    )
    source_time_s, source_trace = _cached_prepared_source_traces(
        config,
        native_traces,
    )
    if not np.array_equal(source_time_s, relative_time_s):
        raise ValueError("phase and cached source traces require the same exact 500-Hz grid")
    return PreparedPhaseRun(
        trial_indices=trial_indices,
        alignment_times_s=alignment_times_s.copy(),
        prepared_trials=prepared_trials,
        phase_tensor=phase_tensor,
        phase_valid=phase_valid,
        relative_time_s=relative_time_s,
        site_valid=site_valid,
        pair_valid=pair_valid,
        source_trace=source_trace,
        prepared_phase_cache_state="warm" if cached is not None else "cold",
    )


def load_configured_unit_spikes(
    config: LFPSummaryConfig,
) -> dict[str, np.ndarray]:
    """Load synchronized absolute spike times for the configured stable units.

    Parameters
    ----------
    config : LFPSummaryConfig
        Configuration whose unit population provides sorter and aligned-spike
        paths plus stable ``"probe:cluster"`` identities. Times are seconds.

    Returns
    -------
    dict[str, numpy.ndarray]
        Configured stable ids mapped to owned finite absolute seconds
        ``(spike_for_unit,)`` arrays, in the configured identity set.

    Raises
    ------
    ValueError
        If population paths, identities, aligned axes, metadata channels, or
        requested units are inconsistent. No partial mapping is returned.
    """
    population = config.unit_population
    if population is None:
        raise ValueError("Spike phase requires config.unit_population")
    if population.sorter_path is None or population.aligned_spike_path is None:
        raise ValueError("Spike phase requires sorter and aligned-spike paths")
    spike_clusters, cluster_info = spike_behavior_pynapple.load_sorter_metadata(
        population.sorter_path
    )
    aligned_spikes = spike_behavior_pynapple.load_aligned_spikes(
        population.aligned_spike_path
    )
    spike_behavior_pynapple.validate_aligned_spike_inputs(
        aligned_spikes,
        spike_clusters,
    )
    quality = dict(population.quality_settings)
    labels_text = quality.get("quality_labels", "good,mua")
    quality_labels = tuple(
        value.strip() for value in labels_text.split(",") if value.strip()
    )
    filtered = unit_spike_loading.filter_cluster_metadata(
        cluster_info,
        population.selected_channels,
        quality_labels=quality_labels,
        default_group=quality.get("default_group", "mua"),
        quality_column=quality.get("quality_column", "group"),
    )
    permitted_clusters = set(filtered["cluster_id"].astype(int).tolist())
    loaded: dict[str, np.ndarray] = {}
    for unit_id in population.stable_unit_ids:
        probe_label, cluster_id = _split_stable_unit_id(unit_id)
        if probe_label != population.probe_label or cluster_id not in permitted_clusters:
            raise ValueError(f"configured unit is outside active population: {unit_id}")
        values = np.asarray(aligned_spikes[spike_clusters == cluster_id], dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError(f"unit spike times must be finite seconds: {unit_id}")
        loaded[unit_id] = values.copy()
    return loaded


def prepare_spike_run(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    unit_spike_loader: Callable[[LFPSummaryConfig], Mapping[str, np.ndarray]],
) -> PreparedSpikeRun:
    """Assign configured absolute unit spikes to full-axis trial windows.

    Parameters
    ----------
    config : LFPSummaryConfig
        Stable unit identities and half-open whole event window in seconds.
    prepared_phase : PreparedPhaseRun
        Full trial row axis plus absolute alignment seconds. Missing alignment
        rows own empty spike trains and remain in the cache trial axis.
    unit_spike_loader : callable
        Called once with ``config`` and returns stable ids mapped to finite
        absolute seconds ``(spike_for_unit,)`` arrays.

    Returns
    -------
    PreparedSpikeRun
        Configured unit order and one event-relative seconds array per unit and
        trial. Overlapping trial windows retain independent spike ownership.
    """
    _validate_prepared_phase_run(config, prepared_phase)
    population = config.unit_population
    if population is None or not population.stable_unit_ids:
        raise ValueError("Spike phase requires at least one configured stable unit")
    loaded = unit_spike_loader(config)
    if not isinstance(loaded, Mapping):
        raise ValueError("unit_spike_loader must return a stable-id mapping")
    if set(loaded) != set(population.stable_unit_ids):
        raise ValueError("loaded unit identities must exactly match configuration")
    event_times = np.asarray(prepared_phase.alignment_times_s, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(event_times))
    whole_window = (
        config.analysis_windows.whole_start_s,
        config.analysis_windows.whole_stop_s,
    )
    trains = []
    for unit_id in population.stable_unit_ids:
        probe_label, cluster_id = _split_stable_unit_id(unit_id)
        finite_train = build_trial_relative_spike_trains(
            probe_label=probe_label,
            cluster_id=cluster_id,
            unit_spike_times_s=np.asarray(loaded[unit_id], dtype=float),
            event_times_s=event_times[finite_positions],
            window=whole_window,
        )
        relative = [np.empty(0, dtype=float) for _ in event_times]
        for position, values in zip(
            finite_positions,
            finite_train.relative_spike_times,
            strict=True,
        ):
            relative[int(position)] = values.copy()
        overlap = finite_positions[finite_train.overlap_trial_indices]
        trains.append(
            TrialRelativeSpikeTrains(
                unit_id=unit_id,
                relative_spike_times=tuple(relative),
                overlap_trial_indices=overlap.astype(np.int64, copy=True),
            )
        )
    return PreparedSpikeRun(
        unit_ids=tuple(population.stable_unit_ids),
        population_ids=(population.label,),
        trial_spike_trains=tuple(trains),
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


def build_synchrony_payload(
    config: LFPSummaryConfig,
    prepared: PreparedPhaseRun,
) -> ComponentPayload:
    """Compute the complete Synchrony cache payload from prepared unit phase.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable site/pair, phase-frequency, band, half-open epoch, bootstrap,
        filter, and seed settings. Times are seconds and frequencies are Hz.
    prepared : PreparedPhaseRun
        Full-trial complex64 phase, numerical validity, site/pair masks, and
        source-voltage 500-Hz traces.

    Returns
    -------
    ComponentPayload
        Exact ``SYNCHRONY_ARRAY_SCHEMA`` arrays. ITPC/ISPC/PLV are
        dimensionless, offsets/Hilbert phase are radians, counts are trials or
        samples, and no wavelet tensor is retained.

    Raises
    ------
    ValueError
        If prepared axes or identities disagree with the active configuration.
    """
    _validate_prepared_phase_run(config, prepared)
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    condition_membership = _analysis_condition_membership(prepared.prepared_trials)
    pair_indices = _configured_pair_indices(config)
    clustering = compute_phase_clustering_summary(
        prepared.phase_tensor,
        prepared.phase_valid,
        prepared.prepared_trials.condition_names,
        condition_membership,
        pair_indices,
    )
    epoch_windows = _epoch_windows(config)
    epoch_names = tuple(epoch_windows)
    band_names = tuple(band.name for band in config.phase.bands)
    condition_count = len(prepared.prepared_trials.condition_names)
    site_count = len(config.sites)
    pair_count = len(pair_indices)
    epoch_count = len(epoch_names)
    band_count = len(band_names)
    itpc_band = np.full(
        (condition_count, site_count, epoch_count, band_count),
        np.nan,
    )
    itpc_low = itpc_band.copy()
    itpc_q25 = itpc_band.copy()
    itpc_median = itpc_band.copy()
    itpc_q75 = itpc_band.copy()
    itpc_high = itpc_band.copy()
    itpc_selected_count = np.zeros(itpc_band.shape, dtype=np.int32)
    itpc_unstable = np.ones(itpc_band.shape, dtype=bool)
    ispc_band = np.full(
        (condition_count, pair_count, epoch_count, band_count),
        np.nan,
    )
    ispc_low = ispc_band.copy()
    ispc_q25 = ispc_band.copy()
    ispc_median = ispc_band.copy()
    ispc_q75 = ispc_band.copy()
    ispc_high = ispc_band.copy()
    ispc_selected_count = np.zeros(ispc_band.shape, dtype=np.int32)
    ispc_unstable = np.ones(ispc_band.shape, dtype=bool)
    for condition_index in range(condition_count):
        condition_mask = condition_membership[:, condition_index]
        condition_seed = int(config.phase.seed) + condition_index
        for site_index in range(site_count):
            summary = bootstrap_phase_clustering_bands(
                np.moveaxis(prepared.phase_tensor[site_index], 1, 0),
                np.moveaxis(prepared.phase_valid[site_index], 1, 0),
                condition_mask & prepared.site_valid[site_index],
                frequencies_hz,
                prepared.relative_time_s,
                epoch_windows,
                config.phase.bands,
                config.phase.bootstrap_count,
                condition_seed,
            )
            itpc_band[condition_index, site_index] = summary.estimate
            itpc_low[condition_index, site_index] = summary.ci_low
            itpc_q25[condition_index, site_index] = summary.q25
            itpc_median[condition_index, site_index] = summary.median
            itpc_q75[condition_index, site_index] = summary.q75
            itpc_high[condition_index, site_index] = summary.ci_high
            itpc_selected_count[condition_index, site_index] = summary.selected_trial_count
            itpc_unstable[condition_index, site_index] = summary.unstable

    plv_shape = (
        prepared.trial_indices.size,
        pair_count,
        epoch_count,
        frequencies_hz.size,
    )
    plv_by_frequency = np.full(plv_shape, np.nan)
    plv_offset = np.full(plv_shape, np.nan)
    plv_count = np.zeros(plv_shape, dtype=np.int32)
    plv_fraction = np.zeros(plv_shape, dtype=float)
    plv_computable = np.zeros(plv_shape, dtype=bool)
    plv_band = np.full(plv_shape[:3] + (band_count,), np.nan)
    for pair_index, (site_a, site_b) in enumerate(pair_indices):
        relative_phase = (
            prepared.phase_tensor[site_a] * np.conjugate(prepared.phase_tensor[site_b])
        )
        relative_valid = prepared.phase_valid[site_a] & prepared.phase_valid[site_b]
        pair_plv = compute_trial_plv_by_frequency(
            np.moveaxis(relative_phase, 1, 0)[None],
            np.moveaxis(relative_valid, 1, 0)[None],
            prepared.relative_time_s,
            epoch_windows,
        )
        plv_by_frequency[:, pair_index] = pair_plv.plv_by_frequency[0]
        plv_offset[:, pair_index] = pair_plv.plv_phase_offset_rad[0]
        plv_count[:, pair_index] = pair_plv.valid_sample_count[0]
        plv_fraction[:, pair_index] = pair_plv.valid_sample_fraction[0]
        plv_computable[:, pair_index] = pair_plv.computable[0]
        pair_band = aggregate_trial_plv_bands(
            pair_plv.plv_by_frequency,
            frequencies_hz,
            config.phase.bands,
        )
        plv_band[:, pair_index] = pair_band[0]
        for condition_index in range(condition_count):
            condition_mask = condition_membership[:, condition_index]
            summary = bootstrap_phase_clustering_bands(
                np.moveaxis(relative_phase, 1, 0),
                np.moveaxis(relative_valid, 1, 0),
                condition_mask & prepared.pair_valid[pair_index],
                frequencies_hz,
                prepared.relative_time_s,
                epoch_windows,
                config.phase.bands,
                config.phase.bootstrap_count,
                int(config.phase.seed) + condition_index,
            )
            ispc_band[condition_index, pair_index] = summary.estimate
            ispc_low[condition_index, pair_index] = summary.ci_low
            ispc_q25[condition_index, pair_index] = summary.q25
            ispc_median[condition_index, pair_index] = summary.median
            ispc_q75[condition_index, pair_index] = summary.q75
            ispc_high[condition_index, pair_index] = summary.ci_high
            ispc_selected_count[condition_index, pair_index] = summary.selected_trial_count
            ispc_unstable[condition_index, pair_index] = summary.unstable
    filtered_trace, hilbert_phase_rad = _band_hilbert_traces(config, prepared)
    arrays = {
        "trial_indices": prepared.trial_indices.astype(np.int64, copy=True),
        "site_ids": np.asarray([site.stable_id for site in config.sites], dtype="<U64"),
        "site_voltage_units": np.asarray(
            [site.voltage_unit for site in config.sites],
            dtype="<U64",
        ),
        "condition_names": np.asarray(
            prepared.prepared_trials.condition_names,
            dtype="<U64",
        ),
        "condition_membership": prepared.prepared_trials.condition_membership.copy(),
        "filter_membership": prepared.prepared_trials.filter_membership.copy(),
        "frequency_hz": frequencies_hz.copy(),
        "epoch_names": np.asarray(epoch_names, dtype="<U16"),
        "band_names": np.asarray(band_names, dtype="<U64"),
        "relative_time_s": prepared.relative_time_s.copy(),
        "site_valid": prepared.site_valid.copy(),
        "pair_valid": prepared.pair_valid.copy(),
        "site_exclusion_count": np.count_nonzero(~prepared.site_valid, axis=1).astype(
            np.int64
        ),
        "pair_exclusion_count": np.count_nonzero(~prepared.pair_valid, axis=1).astype(
            np.int64
        ),
        "pair_site_a_ids": np.asarray(
            [config.sites[index].stable_id for index, _ in pair_indices],
            dtype="<U64",
        ),
        "pair_site_b_ids": np.asarray(
            [config.sites[index].stable_id for _, index in pair_indices],
            dtype="<U64",
        ),
        "itpc": clustering.itpc,
        "itpc_effective_trial_count": clustering.itpc_effective_trial_count,
        "ispc": clustering.ispc,
        "ispc_phase_offset_rad": clustering.ispc_phase_offset_rad,
        "ispc_effective_trial_count": clustering.ispc_effective_trial_count,
        "itpc_band_mean": itpc_band,
        "itpc_ci_low": itpc_low,
        "itpc_bootstrap_q25": itpc_q25,
        "itpc_bootstrap_median": itpc_median,
        "itpc_bootstrap_q75": itpc_q75,
        "itpc_ci_high": itpc_high,
        "itpc_band_trial_count": itpc_selected_count,
        "itpc_unstable": itpc_unstable,
        "ispc_band_mean": ispc_band,
        "ispc_ci_low": ispc_low,
        "ispc_bootstrap_q25": ispc_q25,
        "ispc_bootstrap_median": ispc_median,
        "ispc_bootstrap_q75": ispc_q75,
        "ispc_ci_high": ispc_high,
        "ispc_band_trial_count": ispc_selected_count,
        "ispc_unstable": ispc_unstable,
        "plv_by_frequency": plv_by_frequency,
        "plv_phase_offset_rad": plv_offset,
        "plv_valid_sample_count": plv_count,
        "plv_valid_sample_fraction": plv_fraction,
        "plv_computable": plv_computable,
        "plv_band_mean": plv_band,
        "source_trace": prepared.source_trace.copy(),
        "band_filtered_trace": filtered_trace,
        "hilbert_phase_rad": hilbert_phase_rad,
    }
    return build_component_payload("synchrony", arrays)


def build_spike_phase_payload(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
) -> ComponentPayload:
    """Build the frozen public Spike-phase payload without progress injection."""
    return _build_spike_phase_payload(
        config,
        prepared_phase,
        prepared_spikes,
        progress_callback=None,
    )


def _build_spike_phase_payload(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
    *,
    progress_callback: Callable[[ProgressEvent], None] | None,
) -> ComponentPayload:
    """Compute the complete observed and shuffled Spike-phase cache payload.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable frequency, half-open epoch, band, reliability, FDR, shuffle,
        phase-bin, site, filter, and seed settings. Times are seconds and
        frequencies are Hz.
    prepared_phase : PreparedPhaseRun
        Full-axis complex unit phase and 500-Hz source traces with explicit
        numerical validity on ``(site, frequency, trial, time)`` axes.
    prepared_spikes : PreparedSpikeRun
        Stable units and one event-relative seconds array per unit/trial.
    progress_callback : callable or None
        Internal pipeline seam receiving executor ``ProgressEvent`` objects.
        The frozen public three-argument builder always supplies ``None``.

    Returns
    -------
    ComponentPayload
        Exact ``SPIKE_PHASE_ARRAY_SCHEMA`` arrays. PPC is dimensionless, phase
        is radians, counts retain spike/trial/shuffle units, and no shuffle or
        wavelet tensor is persisted.

    Raises
    ------
    ValueError
        If prepared phase/spike identities or axes disagree with configuration.
    """
    _validate_prepared_phase_run(config, prepared_phase)
    _validate_prepared_spike_run(config, prepared_phase, prepared_spikes)
    frequencies_hz = np.asarray(config.phase.frequency_hz, dtype=float)
    condition_membership = _analysis_condition_membership(
        prepared_phase.prepared_trials
    )
    epoch_windows = _selected_ppc_epoch_windows(config)
    epoch_names = tuple(epoch_windows)
    band_names = tuple(band.name for band in config.phase.bands)
    unit_count = len(prepared_spikes.unit_ids)
    condition_count = condition_membership.shape[1]
    site_count = len(config.sites)
    epoch_count = len(epoch_names)
    frequency_count = frequencies_hz.size
    metric_shape = (
        unit_count,
        condition_count,
        site_count,
        epoch_count,
        frequency_count,
    )
    phase_bin_edges = np.asarray(config.ppc.phase_bin_edges_rad, dtype=float)
    execution = lfp_summary_ppc_runtime.execute_grouped_ppc_component(
        config=config,
        execution=config.ppc_execution,
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        work_root=config.output_directory.parent / "lfp_summary_work",
        progress_callback=progress_callback,
    )
    arrays = _copy_grouped_ppc_component_summary(
        execution.summary_arrays,
        metric_shape=metric_shape,
        phase_bin_count=len(config.ppc.phase_bin_edges_rad) - 1,
    )
    trial_spike_counts = _grouped_exemplar_trial_spike_counts(
        prepared_phase=prepared_phase,
        prepared_spikes=prepared_spikes,
        condition_membership=condition_membership,
        epoch_windows=epoch_windows,
    )
    completed_ppc_runs = (
        [(Path(execution.run_directory), execution.run_fingerprint)]
        if execution.run_directory.exists()
        else []
    )
    arrays["ppc_band_mean"] = spike_lfp_summary.compute_band_ppc_means(
        frequencies_hz=frequencies_hz,
        ppc_by_frequency=arrays["ppc"],
    ).ppc_band_mean
    (
        selected_low,
        selected_high,
        illustrative,
        illustrative_low,
        illustrative_high,
    ) = _select_spike_exemplars(
        config,
        prepared_phase,
        prepared_spikes,
        arrays["ppc_band_mean"],
        trial_spike_counts,
    )
    packed_spikes, spike_offsets = _pack_trial_spike_trains(prepared_spikes)
    filtered_trace, hilbert_phase_rad = _band_hilbert_traces(
        config,
        prepared_phase,
    )
    arrays.update(
        {
            "unit_ids": np.asarray(prepared_spikes.unit_ids, dtype="<U64"),
            "population_ids": np.asarray(
                prepared_spikes.population_ids, dtype="<U64"
            ),
            "trial_indices": prepared_phase.trial_indices.astype(
                np.int64, copy=True
            ),
            "site_ids": np.asarray(
                [site.stable_id for site in config.sites], dtype="<U64"
            ),
            "site_voltage_units": np.asarray(
                [site.voltage_unit for site in config.sites], dtype="<U64"
            ),
            "condition_names": np.asarray(
                prepared_phase.prepared_trials.condition_names, dtype="<U64"
            ),
            "condition_membership": (
                prepared_phase.prepared_trials.condition_membership.copy()
            ),
            "filter_membership": (
                prepared_phase.prepared_trials.filter_membership.copy()
            ),
            "epoch_names": np.asarray(epoch_names, dtype="<U16"),
            "band_names": np.asarray(band_names, dtype="<U64"),
            "frequency_hz": frequencies_hz.copy(),
            "relative_time_s": prepared_phase.relative_time_s.copy(),
            "phase_bin_edges_rad": phase_bin_edges.copy(),
            "relative_spike_times_s": packed_spikes,
            "relative_spike_time_offsets": spike_offsets,
            "source_trace": prepared_phase.source_trace.copy(),
            "band_filtered_trace": filtered_trace,
            "hilbert_phase_rad": hilbert_phase_rad,
            "selected_low_unit_ids": selected_low,
            "selected_high_unit_ids": selected_high,
            "illustrative_low_trial_indices": illustrative_low,
            "illustrative_high_trial_indices": illustrative_high,
            "illustrative_trial_indices": illustrative,
        }
    )
    payload = build_component_payload("spike_phase", arrays)
    cleanup_targets = _ppc_cleanup_targets(config, completed_ppc_runs)
    cleanup = _ppc_post_commit_cleanup(config, completed_ppc_runs)
    component_plan = getattr(execution, "component_plan", None)
    allocation = getattr(component_plan, "allocation_estimate", None)
    execution_metadata = {
        "ppc_planning_seconds": getattr(execution, "planning_seconds", None),
        "grouped_execution_seconds": getattr(
            execution,
            "grouped_execution_seconds",
            None,
        ),
        "requested_worker_count": int(config.ppc_execution.worker_count),
        "planner_active_worker_count": getattr(allocation, "active_worker_count", None),
        "planned_parent_private_bytes": getattr(
            allocation,
            "planned_parent_private_bytes",
            None,
        ),
        "planned_worker_private_bytes": getattr(
            allocation,
            "planned_worker_private_bytes",
            None,
        ),
        "planned_aggregate_array_bytes": getattr(
            allocation,
            "planned_aggregate_array_bytes",
            None,
        ),
        "shared_phase_mmap_bytes": getattr(
            allocation,
            "shared_phase_mmap_bytes",
            None,
        ),
        "scheduled_edge_count": getattr(component_plan, "scheduled_edge_count", None),
        "independent_edge_count": getattr(
            component_plan,
            "independent_edge_count",
            None,
        ),
        "union_edge_count": getattr(component_plan, "union_edge_count", None),
        "completed_block_count": len(getattr(execution, "completed_block_ids", ())),
        "resumed_block_count": len(getattr(execution, "resumed_block_ids", ())),
        "prepared_phase_cache_state": prepared_phase.prepared_phase_cache_state,
        "run_fingerprint": execution.run_fingerprint,
        "run_directory": str(execution.run_directory),
    }
    return ComponentPayload(
        arrays=payload.arrays,
        manifest_entry=payload.manifest_entry,
        post_commit_cleanup=cleanup,
        post_commit_cleanup_targets=cleanup_targets,
        execution_metadata=execution_metadata,
    )


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


def make_lfp_summary_pipeline_dependencies(
    *,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    unit_spike_loader: Callable[
        [LFPSummaryConfig], Mapping[str, np.ndarray]
    ] = load_configured_unit_spikes,
    phase_preparer: Callable[[LFPSummaryConfig], PreparedPhaseRun] | None = None,
    site_phase_tensor_builder: Callable[..., lfp_phase_clustering.PhaseTrialTensor] = (
        lfp_phase_clustering.compute_site_phase_trial_tensor
    ),
    block_loader_factory: Callable[
        [LFPSiteConfig],
        Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    ]
    | None = None,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> PipelineDependencies:
    """Bind one production dependency bundle for every LFP-summary component.

    Parameters
    ----------
    trial_table_loader : callable
        Accepts one :class:`LFPSummaryConfig` and returns a trial table with one
        row per stable trial. Absolute event timestamps are seconds.
    unit_spike_loader : callable
        Accepts the same configuration and returns probe-qualified unit ids
        mapped to finite one-dimensional absolute spike-time arrays in seconds.
    phase_preparer : callable or None
        Optional complete configuration-to-:class:`PreparedPhaseRun` seam. Its
        phase and validity arrays have ``(site, frequency, trial, time)`` axes;
        frequency is Hz and event-relative time is seconds. ``None`` uses the
        production phase-preparation path and its validated work cache.
    site_phase_tensor_builder : callable
        Continuous-block Morlet seam used only when ``phase_preparer`` is
        ``None``. It preserves the configured site/frequency/trial/time axes.
    block_loader_factory : callable or None
        Optional site-specific continuous source loader factory. Returned
        traces retain their configured source voltage unit and time in seconds.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized per-trial LFP loaders. Each returns relative seconds,
        a one-dimensional source-voltage trace, and sample rate in Hz.

    Returns
    -------
    PipelineDependencies
        A single bundle supporting Power, Synchrony, Spike phase, and Compute
        All. Synchrony and Spike phase share the exact phase object supplied by
        the pipeline. Spike progress events retain all grouped-runtime fields.

    Raises
    ------
    ValueError
        Before any production work if absolute amplitude thresholds are
        nonempty, or if prepared phase/spike identities violate their contracts.
    RuntimeError
        If manifest loading is requested before a composed preparation call.

    Notes
    -----
    The bundle introduces no numerical transformations. It delegates to the
    established preparation, payload, and manifest-last transaction functions.
    """
    active_config: LFPSummaryConfig | None = None

    def bind_config(config: LFPSummaryConfig) -> None:
        """Validate supported settings and retain the exact active config."""
        nonlocal active_config
        _validate_composed_phase_amplitude_policy(config)
        active_config = config

    def prepare_power(config: LFPSummaryConfig) -> PreparedPowerRun:
        """Prepare Power inputs after fail-before-work policy validation."""
        bind_config(config)
        return prepare_power_run(
            config,
            trial_table_loader,
            spikeglx_loader,
            open_ephys_loader,
        )

    def prepare_phase(config: LFPSummaryConfig) -> PreparedPhaseRun:
        """Prepare or inject one full-axis phase record for shared reuse."""
        bind_config(config)
        prepared = (
            phase_preparer(config)
            if phase_preparer is not None
            else prepare_phase_run(
                config,
                trial_table_loader,
                site_phase_tensor_builder=site_phase_tensor_builder,
                block_loader_factory=block_loader_factory,
                spikeglx_loader=spikeglx_loader,
                open_ephys_loader=open_ephys_loader,
                work_cache_root=config.output_directory.parent / "lfp_summary_work",
            )
        )
        if not isinstance(prepared, PreparedPhaseRun):
            raise ValueError("composed phase preparation requires PreparedPhaseRun")
        return prepared

    def prepare_spike(
        config: LFPSummaryConfig,
        phase: object,
    ) -> PreparedSpikeRun:
        """Prepare trial-local unit spikes from the exact shared phase axis."""
        bind_config(config)
        if not isinstance(phase, PreparedPhaseRun):
            raise ValueError("composed Spike phase requires PreparedPhaseRun")
        return prepare_spike_run(config, phase, unit_spike_loader)

    def load_manifest(directory: Path) -> dict[str, object]:
        """Load metadata for the exact config accepted by preparation."""
        if active_config is None:
            raise RuntimeError("composed manifest loading requires preparation")
        return load_or_initialize_manifest(directory, active_config)

    def build_spike_payload_with_progress(
        config: LFPSummaryConfig,
        phase: object,
        spikes: object,
        progress_callback: Callable[[ProgressEvent], None] | None,
    ) -> ComponentPayload:
        """Forward grouped PPC progress without translating record fields."""
        if not isinstance(phase, PreparedPhaseRun) or not isinstance(
            spikes,
            PreparedSpikeRun,
        ):
            raise ValueError(
                "composed Spike payload requires prepared phase and spike records"
            )
        return _build_spike_phase_payload(
            config,
            phase,
            spikes,
            progress_callback=progress_callback,
        )

    return PipelineDependencies(
        prepare_power=prepare_power,
        prepare_phase=prepare_phase,
        prepare_spike=prepare_spike,
        build_power_payload=build_power_payload,
        build_synchrony_payload=build_synchrony_payload,
        build_spike_phase_payload=build_spike_phase_payload,
        load_manifest=load_manifest,
        write_component=write_component_transaction,
        build_spike_phase_payload_with_progress=build_spike_payload_with_progress,
    )


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


def make_synchrony_pipeline_dependencies(
    *,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    site_phase_tensor_builder: Callable[..., lfp_phase_clustering.PhaseTrialTensor] = (
        lfp_phase_clustering.compute_site_phase_trial_tensor
    ),
    block_loader_factory: Callable[
        [LFPSiteConfig],
        Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    ]
    | None = None,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> PipelineDependencies:
    """Bind real phase preparation, Synchrony calculation, and atomic cache I/O.

    Parameters
    ----------
    trial_table_loader, site_phase_tensor_builder, block_loader_factory : callable
        Production or injected seams accepted by :func:`prepare_phase_run`.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized per-trial trace loaders for exemplar cache arrays.

    Returns
    -------
    PipelineDependencies
        Synchrony-only production dependencies. Power and spike-phase seams
        raise explicitly instead of creating placeholder arrays.
    """
    active_config: LFPSummaryConfig | None = None

    def prepare_phase(config: LFPSummaryConfig) -> PreparedPhaseRun:
        """Prepare phase products using the factory's fixed loader seams."""
        nonlocal active_config
        active_config = config
        return prepare_phase_run(
            config,
            trial_table_loader,
            site_phase_tensor_builder=site_phase_tensor_builder,
            block_loader_factory=block_loader_factory,
            spikeglx_loader=spikeglx_loader,
            open_ephys_loader=open_ephys_loader,
            work_cache_root=config.output_directory.parent / "lfp_summary_work",
        )

    def load_manifest(directory: Path) -> dict[str, object]:
        """Load the active configuration's JSON-ready cache manifest."""
        if active_config is None:
            raise RuntimeError("Synchrony manifest loading requires prepared configuration")
        return load_or_initialize_manifest(directory, active_config)

    def unsupported_power(_: LFPSummaryConfig) -> object:
        """Reject unbound Power work without returning fake arrays."""
        raise NotImplementedError("power is unsupported by the Synchrony runtime")

    def unsupported_spike(_: LFPSummaryConfig, __: object) -> object:
        """Reject unbound spike preparation without returning fake arrays."""
        raise NotImplementedError("spike preparation is unsupported by Synchrony runtime")

    def unsupported_power_payload(_: LFPSummaryConfig, __: object) -> ComponentPayload:
        """Reject unbound Power payload work without placeholder values."""
        raise NotImplementedError("power payload is unsupported by Synchrony runtime")

    def unsupported_spike_payload(
        _: LFPSummaryConfig,
        __: object,
        ___: object,
    ) -> ComponentPayload:
        """Reject unbound spike payload work without placeholder values."""
        raise NotImplementedError("spike payload is unsupported by Synchrony runtime")

    return PipelineDependencies(
        prepare_power=unsupported_power,
        prepare_phase=prepare_phase,
        prepare_spike=unsupported_spike,
        build_power_payload=unsupported_power_payload,
        build_synchrony_payload=build_synchrony_payload,
        build_spike_phase_payload=unsupported_spike_payload,
        load_manifest=load_manifest,
        write_component=write_component_transaction,
    )


def make_spike_phase_pipeline_dependencies(
    *,
    trial_table_loader: Callable[[LFPSummaryConfig], pd.DataFrame],
    unit_spike_loader: Callable[
        [LFPSummaryConfig], Mapping[str, np.ndarray]
    ] = load_configured_unit_spikes,
    phase_preparer: Callable[[LFPSummaryConfig], PreparedPhaseRun] | None = None,
    site_phase_tensor_builder: Callable[..., lfp_phase_clustering.PhaseTrialTensor] = (
        lfp_phase_clustering.compute_site_phase_trial_tensor
    ),
    block_loader_factory: Callable[
        [LFPSiteConfig],
        Callable[[float, float], tuple[np.ndarray, np.ndarray, float]],
    ]
    | None = None,
    spikeglx_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
    open_ephys_loader: Callable[..., tuple[np.ndarray, np.ndarray, float]] | None = None,
) -> PipelineDependencies:
    """Bind real phase/spike preparation, PPC payload, and atomic cache I/O.

    Parameters
    ----------
    trial_table_loader, unit_spike_loader : callable
        Trial-table and stable-id-to-absolute-seconds spike loader seams.
    phase_preparer : callable or None
        Optional complete phase seam for tests or shared upstream preparation.
        ``None`` delegates to :func:`prepare_phase_run` with the remaining LFP
        seams.
    site_phase_tensor_builder, block_loader_factory : callable or None
        Bounded continuous phase-transform seams accepted by phase preparation.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized per-trial source loaders for exemplar cache traces.

    Returns
    -------
    PipelineDependencies
        Spike-only production dependencies. Power and Synchrony payload seams
        reject use rather than creating placeholder arrays.
    """
    active_config: LFPSummaryConfig | None = None

    def prepare_phase(config: LFPSummaryConfig) -> PreparedPhaseRun:
        """Prepare phase with the explicit seam or the production adapters."""
        nonlocal active_config
        active_config = config
        if phase_preparer is not None:
            return phase_preparer(config)
        return prepare_phase_run(
            config,
            trial_table_loader,
            site_phase_tensor_builder=site_phase_tensor_builder,
            block_loader_factory=block_loader_factory,
            spikeglx_loader=spikeglx_loader,
            open_ephys_loader=open_ephys_loader,
            work_cache_root=config.output_directory.parent / "lfp_summary_work",
        )

    def prepare_spike(
        config: LFPSummaryConfig,
        phase: object,
    ) -> PreparedSpikeRun:
        """Prepare configured trial-local spikes from the exact phase trial axis."""
        if not isinstance(phase, PreparedPhaseRun):
            raise ValueError("Spike runtime requires PreparedPhaseRun")
        return prepare_spike_run(config, phase, unit_spike_loader)

    def load_manifest(directory: Path) -> dict[str, object]:
        """Load the active configuration's JSON-ready cache manifest."""
        if active_config is None:
            raise RuntimeError("Spike manifest loading requires prepared configuration")
        return load_or_initialize_manifest(directory, active_config)

    def unsupported_power(_: LFPSummaryConfig) -> object:
        """Reject unbound Power preparation without fake arrays."""
        raise NotImplementedError("power is unsupported by Spike-phase runtime")

    def unsupported_power_payload(
        _: LFPSummaryConfig,
        __: object,
    ) -> ComponentPayload:
        """Reject unbound Power payload construction."""
        raise NotImplementedError("power payload is unsupported by Spike-phase runtime")

    def unsupported_synchrony_payload(
        _: LFPSummaryConfig,
        __: object,
    ) -> ComponentPayload:
        """Reject unbound Synchrony payload construction."""
        raise NotImplementedError(
            "synchrony payload is unsupported by Spike-phase runtime"
        )

    def build_spike_payload_with_progress(
        config: LFPSummaryConfig,
        phase: object,
        spikes: object,
        progress_callback: Callable[[ProgressEvent], None] | None,
    ) -> ComponentPayload:
        """Build Spike phase while forwarding pipeline progress to PPC jobs."""
        if not isinstance(phase, PreparedPhaseRun) or not isinstance(
            spikes,
            PreparedSpikeRun,
        ):
            raise ValueError("Spike progress payload requires prepared runtime products")
        return _build_spike_phase_payload(
            config,
            phase,
            spikes,
            progress_callback=progress_callback,
        )

    return PipelineDependencies(
        prepare_power=unsupported_power,
        prepare_phase=prepare_phase,
        prepare_spike=prepare_spike,
        build_power_payload=unsupported_power_payload,
        build_synchrony_payload=unsupported_synchrony_payload,
        build_spike_phase_payload=build_spike_phase_payload,
        load_manifest=load_manifest,
        write_component=write_component_transaction,
        build_spike_phase_payload_with_progress=build_spike_payload_with_progress,
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


def _validate_site_phase_tensor(
    tensor: lfp_phase_clustering.PhaseTrialTensor,
    frequency_count: int,
) -> None:
    """Validate one site's complex phase axes without changing its storage."""
    if not isinstance(tensor, lfp_phase_clustering.PhaseTrialTensor):
        raise ValueError("site phase builder must return PhaseTrialTensor")
    phase = np.asarray(tensor.phase)
    valid = np.asarray(tensor.valid)
    times = np.asarray(tensor.relative_time_s)
    indices = np.asarray(tensor.trial_indices)
    expected = (1, frequency_count, indices.size, times.size)
    if phase.shape != expected or valid.shape != phase.shape:
        raise ValueError("site phase tensor axes are invalid")
    if not np.iscomplexobj(phase) or valid.dtype != np.dtype(bool):
        raise ValueError("site phase and validity dtypes are invalid")
    if times.ndim != 1 or times.size < 2 or not np.isfinite(times).all():
        raise ValueError("site phase time must be a finite nontrivial vector")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("site phase time must be strictly increasing")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("site phase trial indices must be integer row positions")
    if np.unique(indices).size != indices.size:
        raise ValueError("site phase trial indices must be unique")


def _full_phase_axes(
    site_tensors: list[lfp_phase_clustering.PhaseTrialTensor],
    trial_indices: np.ndarray,
    frequency_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Place independent site tensors on one full trial axis without intersection.

    Parameters
    ----------
    site_tensors : list[PhaseTrialTensor]
        One independently retained tensor per configured site.
    trial_indices : numpy.ndarray
        Int64 shape ``(trial,)`` full trial-table row positions.
    frequency_count : int
        Positive size of the configured phase-frequency axis.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Exact relative seconds ``(time,)``, complex64 phase and boolean validity
        ``(site, frequency, trial, time)``, and boolean ``(site, trial)``
        retention. Missing sites/trials remain zero/false, never globally drop.
    """
    if not site_tensors:
        raise ValueError("phase preparation requires at least one site")
    relative_time_s = np.asarray(site_tensors[0].relative_time_s, dtype=float).copy()
    shape = (
        len(site_tensors),
        frequency_count,
        trial_indices.size,
        relative_time_s.size,
    )
    phase = np.zeros(shape, dtype=np.complex64)
    valid = np.zeros(shape, dtype=bool)
    site_valid = np.zeros((len(site_tensors), trial_indices.size), dtype=bool)
    position_by_row = {int(row): position for position, row in enumerate(trial_indices)}
    for site_index, tensor in enumerate(site_tensors):
        if not np.array_equal(relative_time_s, tensor.relative_time_s):
            raise ValueError("site phase tensors do not share an exact time grid")
        try:
            positions = np.asarray(
                [position_by_row[int(row)] for row in tensor.trial_indices],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError("site phase tensor contains an unknown trial row") from error
        phase[site_index][:, positions, :] = np.asarray(tensor.phase[0], np.complex64)
        valid[site_index][:, positions, :] = tensor.valid[0]
        site_valid[site_index, positions] = np.any(tensor.valid[0], axis=(0, 2))
    return relative_time_s, phase, valid, site_valid


def _configured_pair_indices(config: LFPSummaryConfig) -> tuple[tuple[int, int], ...]:
    """Map ordered stable site-pair ids to distinct zero-based site indices."""
    position = {site.stable_id: index for index, site in enumerate(config.sites)}
    pairs = []
    for site_a_id, site_b_id in config.site_pairs:
        try:
            site_a = position[site_a_id]
            site_b = position[site_b_id]
        except KeyError as error:
            raise ValueError("configured pair references an unknown site") from error
        if site_a == site_b:
            raise ValueError("configured Synchrony pairs require distinct sites")
        pairs.append((site_a, site_b))
    if not pairs:
        raise ValueError("Synchrony requires at least one configured site pair")
    return tuple(pairs)


def _cached_prepared_source_traces(
    config: LFPSummaryConfig,
    site_traces: dict[str, PreparedSiteTraces],
) -> tuple[np.ndarray, np.ndarray]:
    """Anti-alias native prepared traces onto the exact 500-Hz cache grid."""
    if not np.isclose(config.phase.output_rate_hz, _CACHE_SAMPLE_RATE_HZ):
        raise ValueError("Synchrony production output_rate_hz must be exactly 500")
    cached = []
    relative_time_s: np.ndarray | None = None
    for site in config.sites:
        traces = site_traces[site.stable_id]
        ratio = traces.sample_rate_hz / _CACHE_SAMPLE_RATE_HZ
        factor = round(ratio)
        if factor < 1 or not np.isclose(ratio, factor, rtol=0.0, atol=1e-9):
            raise ValueError("Synchrony source traces require integer decimation to 500 Hz")
        selected_time = traces.relative_time_s[::factor]
        selected_trace = _resample_trace_rows(
            traces.source_trace,
            factor,
            selected_time.size,
        )
        if relative_time_s is None:
            relative_time_s = selected_time.copy()
        elif not np.array_equal(relative_time_s, selected_time):
            raise ValueError("Synchrony source traces do not share an exact time grid")
        cached.append(selected_trace)
    if relative_time_s is None:
        raise ValueError("Synchrony source traces require at least one site")
    return relative_time_s, np.stack(cached)


def _production_phase_block_loader_factory(
    site: LFPSiteConfig,
    normalized_open_ephys_metadata: dict[str, object] | None = None,
) -> Callable[[float, float], tuple[np.ndarray, np.ndarray, float]]:
    """Bind one configured site's production continuous absolute-time loader.

    Parameters
    ----------
    site : LFPSiteConfig
        Saved LFP channel, acquisition format, sync path, and source units.
    normalized_open_ephys_metadata : dict[str, object] or None, optional
        Preflighted normalized Open Ephys sidecar for ``site``. It contains no
        trace samples and avoids metadata reparsing during phase preparation.
        ``None`` preserves the direct production-factory behavior.

    Returns
    -------
    Callable
        Loader accepting absolute start/stop seconds and returning matching
        absolute seconds, source-voltage samples, and native rate in Hz.

    Raises
    ------
    ValueError
        If the source configuration, normalized Open Ephys metadata,
        saved-channel bounds, rate, or physical unit is invalid before sync or
        binary-trace numerical I/O begins.
    """
    if site.acquisition_format == "open_ephys":
        validate_open_ephys_aligned_sync_path(site)
        metadata = (
            normalized_open_ephys_metadata
            if normalized_open_ephys_metadata is not None
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
        sync = lfp_loading.build_open_ephys_lfp_irig_df(
            site.aligned_sync_path,
            sample_rate_hz,
        )

        def load_open_ephys(
            start_s: float,
            stop_s: float,
        ) -> tuple[np.ndarray, np.ndarray, float]:
            """Read one Open Ephys channel block on its inferred absolute grid."""
            relative, start_sample, stop_sample = lfp_loading.map_lfp_time_window_to_samples(
                start_s,
                (0.0, stop_s - start_s),
                sync,
                sample_rate_hz,
            )
            values, file_rate_hz = lfp_loading.read_open_ephys_lfp_channel_window(
                site.lfp_path,
                site.saved_channel_index,
                start_sample,
                stop_sample,
            )
            return start_s + relative, values, file_rate_hz

        return load_open_ephys
    if site.acquisition_format == "spikeglx":
        metadata = lfp_loading.load_lfp_metadata(site.lfp_path)
        lfp_loading.validate_lfp_saved_channel(metadata, site.saved_channel_index)
        sync, sample_rate_hz = lfp_loading.decode_lfp_sync(site.lfp_path)

        def load_spikeglx(
            start_s: float,
            stop_s: float,
        ) -> tuple[np.ndarray, np.ndarray, float]:
            """Read one SpikeGLX channel block on its decoded absolute grid."""
            relative, start_sample, stop_sample = lfp_loading.map_lfp_time_window_to_samples(
                start_s,
                (0.0, stop_s - start_s),
                sync,
                sample_rate_hz,
            )
            values, file_rate_hz = lfp_loading.read_lfp_saved_channel_window(
                site.lfp_path,
                site.saved_channel_index,
                start_sample,
                stop_sample,
            )
            return start_s + relative, values, file_rate_hz

        return load_spikeglx
    raise ValueError(f"unsupported phase acquisition format: {site.acquisition_format}")


def _preflighted_production_phase_block_loader_factory(
    site: LFPSiteConfig,
    *,
    normalized_open_ephys_metadata_by_site: Mapping[str, dict[str, object]],
) -> Callable[[float, float], tuple[np.ndarray, np.ndarray, float]]:
    """Bind a phase block loader while reusing all-site preflighted OE metadata.

    Parameters
    ----------
    site : LFPSiteConfig
        One configured source. Its source time coordinates are absolute
        seconds, source voltage is in the configured physical unit, and its
        saved-channel index selects one channel from the binary channel axis.
    normalized_open_ephys_metadata_by_site : Mapping[str, dict[str, object]]
        Exact sidecar mappings returned by
        :func:`preflight_open_ephys_site_metadata`, keyed by Open Ephys stable
        site ID. The mappings contain no traces and are reused without JSON
        reparsing. SpikeGLX IDs are absent.

    Returns
    -------
    Callable[[float, float], tuple[numpy.ndarray, numpy.ndarray, float]]
        A loader accepting absolute half-open ``(start_s, stop_s)`` seconds and
        returning matching one-dimensional absolute seconds, one-dimensional
        physical-voltage samples, and a finite source sample rate in Hz.

    Raises
    ------
    ValueError
        If a configured source is unsupported or an Open Ephys metadata/path
        contract is invalid. No sidecar JSON is reloaded for a supplied
        Open Ephys mapping.
    """
    return _production_phase_block_loader_factory(
        site,
        normalized_open_ephys_metadata_by_site.get(site.stable_id),
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


def _split_stable_unit_id(unit_id: str) -> tuple[str, int]:
    """Parse one ``probe:cluster`` identity into categorical components."""
    try:
        probe_label, cluster_text = unit_id.rsplit(":", maxsplit=1)
        cluster_id = int(cluster_text)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid stable unit identity: {unit_id!r}") from error
    if not probe_label or cluster_id < 0 or f"{probe_label}:{cluster_id}" != unit_id:
        raise ValueError(f"invalid stable unit identity: {unit_id!r}")
    return probe_label, cluster_id


def _selected_ppc_epoch_windows(
    config: LFPSummaryConfig,
) -> dict[str, tuple[float, float]]:
    """Return configured PPC epochs in declared order with seconds bounds."""
    available = _epoch_windows(config)
    try:
        return {name: available[name] for name in config.ppc.epochs}
    except KeyError as error:
        raise ValueError("PPC epochs must name whole, before, or after") from error


def _spikes_in_epoch(
    relative_spike_times_s: np.ndarray,
    epoch_window: tuple[float, float],
) -> np.ndarray:
    """Return an owned half-open epoch selection from one trial in seconds."""
    spikes = np.asarray(relative_spike_times_s, dtype=float)
    start_s, stop_s = epoch_window
    return spikes[(spikes >= start_s) & (spikes < stop_s)].copy()


def _shared_derangement_schedule(
    trial_count: int,
    shuffle_count: int,
    seed: int,
) -> np.ndarray | None:
    """Return one unit-shared trial derangement or None below two trials."""
    if trial_count < 2:
        return None
    return spike_lfp_summary.generate_trial_derangement_schedule(
        trial_count,
        shuffle_count,
        seed=seed,
    )


def _ppc_schedule_seed(
    config: LFPSummaryConfig,
    condition_index: int,
    site_index: int,
    epoch_index: int,
) -> int:
    """Derive a deterministic condition/site/epoch seed without unit identity."""
    return int(
        config.ppc.seed
        + condition_index * 1_000_000
        + site_index * 10_000
        + epoch_index * 100
    )


def _copy_grouped_ppc_component_summary(
    summary_arrays: Mapping[str, np.ndarray],
    *,
    metric_shape: tuple[int, int, int, int, int],
    phase_bin_count: int,
) -> dict[str, np.ndarray]:
    """Copy grouped PPC outputs into the frozen public payload field names.

    Parameters
    ----------
    summary_arrays : Mapping[str, numpy.ndarray]
        Grouped executor arrays with metric axes ``(unit, condition, site,
        epoch, frequency)`` and fixed representative histogram axes ``(unit,
        condition, site, epoch, band=2, phase_bin)``. Float metrics are
        dimensionless except preferred phase in radians; counts and flags
        retain their documented public units.
    metric_shape : tuple[int, int, int, int, int]
        Exact final ``(unit, condition, site, epoch, frequency)`` dimensions.
    phase_bin_count : int
        Positive number of configured radians histogram bins.

    Returns
    -------
    dict[str, numpy.ndarray]
        Writable copies for final payload assembly. The histogram key is
        renamed from the executor-private
        ``representative_phase_histogram_count`` to the frozen public
        ``representative_phase_hist_count`` without changing axes or dtype.

    Raises
    ------
    ValueError
        If the executor omits a required field or returns an incompatible
        public axis/dtype contract.
    """
    float_fields = (
        "ppc",
        "resultant_length",
        "preferred_phase_rad",
        "p_value",
        "q_value",
        "null_mean",
        "null_std",
        "null_p025",
        "null_p50",
        "null_p975",
    )
    integer_fields = (
        "spike_count",
        "eligible_trial_count",
        "null_exceedance_count",
        "permutation_count",
    )
    boolean_fields = ("computable", "reliable", "null_eligible", "significant")
    arrays: dict[str, np.ndarray] = {}
    for name in float_fields + integer_fields + boolean_fields:
        if name not in summary_arrays:
            raise ValueError(f"grouped PPC summary lacks {name}")
        values = np.asarray(summary_arrays[name])
        if values.shape != metric_shape:
            raise ValueError("grouped PPC summary axes disagree with final payload")
        expected_dtype = (
            np.dtype(float)
            if name in float_fields
            else np.dtype(np.int64)
            if name in integer_fields
            else np.dtype(bool)
        )
        if values.dtype != expected_dtype:
            raise ValueError("grouped PPC summary dtype disagrees with final payload")
        arrays[name] = values.copy()
    histogram_name = "representative_phase_histogram_count"
    if histogram_name not in summary_arrays:
        raise ValueError("grouped PPC summary lacks representative histogram counts")
    histogram = np.asarray(summary_arrays[histogram_name])
    expected_histogram_shape = metric_shape[:-1] + (2, phase_bin_count)
    if (
        histogram.shape != expected_histogram_shape
        or histogram.dtype != np.dtype(np.int64)
    ):
        raise ValueError("grouped representative histogram axes disagree with final payload")
    arrays["representative_phase_hist_count"] = histogram.copy()
    return arrays


def _grouped_exemplar_trial_spike_counts(
    *,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
    condition_membership: np.ndarray,
    epoch_windows: Mapping[str, tuple[float, float]],
) -> np.ndarray:
    """Count phase-free selected spikes for frozen exemplar selection.

    Parameters
    ----------
    prepared_phase : PreparedPhaseRun
        Full stable trial and Boolean site-valid axes. No phase values are
        sampled by this helper.
    prepared_spikes : PreparedSpikeRun
        Full-unit ordered finite relative spike times in seconds for every
        trial.
    condition_membership : numpy.ndarray
        Boolean ``(trial, condition)`` analysis membership after filter,
        objective-validity, and user-exclusion gates.
    epoch_windows : Mapping[str, tuple[float, float]]
        Ordered half-open epoch seconds bounds, with keys defining the final
        epoch axis.

    Returns
    -------
    numpy.ndarray
        Int64 spike counts on ``(unit, condition, site, epoch, trial)`` axes.
        Nonselected rows remain zero. This preserves the legacy exemplar
        selection input without phase interpolation or histogram sampling.
    """
    counts = np.zeros(
        (
            len(prepared_spikes.unit_ids),
            condition_membership.shape[1],
            len(prepared_phase.site_valid),
            len(epoch_windows),
            prepared_phase.trial_indices.size,
        ),
        dtype=np.int64,
    )
    for condition_index in range(condition_membership.shape[1]):
        for site_index in range(len(prepared_phase.site_valid)):
            selected_positions = np.flatnonzero(
                condition_membership[:, condition_index]
                & prepared_phase.site_valid[site_index]
            )
            for epoch_index, epoch_window in enumerate(epoch_windows.values()):
                for unit_index, train in enumerate(prepared_spikes.trial_spike_trains):
                    for position in selected_positions:
                        counts[
                            unit_index,
                            condition_index,
                            site_index,
                            epoch_index,
                            position,
                        ] = _spikes_in_epoch(
                            train.relative_spike_times[int(position)], epoch_window
                        ).size
    return counts


def _ppc_post_commit_cleanup(
    config: LFPSummaryConfig,
    completed_runs: list[tuple[Path, str]],
) -> Callable[[], None] | None:
    """Return exact completed-run cleanup for the approved retention policy."""
    if (
        not config.ppc_execution.checkpoint_enabled
        or config.ppc_execution.checkpoint_retention != "incomplete_only"
        or not completed_runs
    ):
        return None
    exact_runs = tuple(dict.fromkeys(completed_runs))

    def cleanup() -> None:
        """Remove only executor runs that produced this successfully committed payload."""
        for run_directory, run_fingerprint in exact_runs:
            cleanup_ppc_run(run_directory, run_fingerprint)

    return cleanup


def _ppc_cleanup_targets(
    config: LFPSummaryConfig,
    completed_runs: list[tuple[Path, str]],
) -> tuple[PPCWorkCleanupTarget, ...]:
    """Return exact JSON-safe work identities for deferred launcher cleanup.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable execution policy; no numerical arrays are inspected.
    completed_runs : list[tuple[pathlib.Path, str]]
        PPC work directories and exact SHA-256 run fingerprints that produced
        the committed payload.

    Returns
    -------
    tuple[PPCWorkCleanupTarget, ...]
        Stable de-duplicated targets, or empty when cleanup is disabled/retained.
    """
    if (
        not config.ppc_execution.checkpoint_enabled
        or config.ppc_execution.checkpoint_retention != "incomplete_only"
    ):
        return ()
    return tuple(
        PPCWorkCleanupTarget(Path(run_directory), run_fingerprint)
        for run_directory, run_fingerprint in dict.fromkeys(completed_runs)
    )


def _sample_observed_trial_phase(
    phase_time_s: np.ndarray,
    trial_phase: np.ndarray,
    trial_valid: np.ndarray,
    trial_spikes: tuple[np.ndarray, ...],
    trial_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample complex phase without bridging invalid coordinates.

    Parameters
    ----------
    phase_time_s : numpy.ndarray
        Exact finite relative seconds ``(time,)`` grid.
    trial_phase, trial_valid : numpy.ndarray
        Complex and boolean ``(trial, frequency, time)`` arrays.
    trial_spikes : tuple[numpy.ndarray, ...]
        One event-relative seconds array per selected trial.
    trial_rows : numpy.ndarray
        Integer full-table row identity ``(trial,)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Complex samples and validity ``(frequency, spike)`` plus integer source
        trial rows ``(spike,)``. Invalid samples are zero/false.
    """
    time_s = np.asarray(phase_time_s, dtype=float)
    phase = np.asarray(trial_phase)
    valid = np.asarray(trial_valid, dtype=bool)
    rows = np.asarray(trial_rows)
    expected = (len(trial_spikes), phase.shape[1], time_s.size)
    if phase.ndim != 3 or phase.shape != expected or valid.shape != phase.shape:
        raise ValueError("trial phase sampling axes are inconsistent")
    if rows.shape != (len(trial_spikes),):
        raise ValueError("trial rows must match sampled spike trains")
    spike_count = sum(values.size for values in trial_spikes)
    sampled = np.zeros((phase.shape[1], spike_count), dtype=np.complex64)
    sampled_valid = np.zeros(sampled.shape, dtype=bool)
    source_rows = np.empty(spike_count, dtype=np.int64)
    offset = 0
    for trial_index, spikes in enumerate(trial_spikes):
        for spike in np.asarray(spikes, dtype=float):
            source_rows[offset] = int(rows[trial_index])
            _sample_one_phase_time(
                time_s,
                phase[trial_index],
                valid[trial_index],
                float(spike),
                sampled[:, offset],
                sampled_valid[:, offset],
            )
            offset += 1
    return sampled, sampled_valid, source_rows


def _sample_one_phase_time(
    time_s: np.ndarray,
    phase: np.ndarray,
    valid: np.ndarray,
    spike_time_s: float,
    output: np.ndarray,
    output_valid: np.ndarray,
) -> None:
    """Interpolate one phase column only between two valid neighboring samples."""
    right = int(np.searchsorted(time_s, spike_time_s, side="left"))
    if right < time_s.size and np.isclose(
        time_s[right], spike_time_s, rtol=0.0, atol=1e-12
    ):
        usable = valid[:, right]
        values = phase[:, right]
    elif right == 0 or right == time_s.size:
        return
    else:
        left = right - 1
        usable = valid[:, left] & valid[:, right]
        fraction = (spike_time_s - time_s[left]) / (time_s[right] - time_s[left])
        values = phase[:, left] + fraction * (phase[:, right] - phase[:, left])
    magnitude = np.abs(values)
    usable &= np.isfinite(values.real) & np.isfinite(values.imag) & (magnitude > 0.0)
    output[usable] = values[usable] / magnitude[usable]
    output_valid[usable] = True


def _pack_trial_spike_trains(
    prepared: PreparedSpikeRun,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack unit/trial relative seconds into one vector and global offsets."""
    trial_count = len(prepared.trial_spike_trains[0].relative_spike_times)
    offsets = np.zeros((len(prepared.unit_ids), trial_count + 1), dtype=np.int64)
    chunks: list[np.ndarray] = []
    running = 0
    for unit_index, train in enumerate(prepared.trial_spike_trains):
        offsets[unit_index, 0] = running
        for trial_index, values in enumerate(train.relative_spike_times):
            chunks.append(np.asarray(values, dtype=float).copy())
            running += values.size
            offsets[unit_index, trial_index + 1] = running
    packed = np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
    return packed, offsets


def _select_spike_exemplars(
    config: LFPSummaryConfig,
    prepared_phase: PreparedPhaseRun,
    prepared_spikes: PreparedSpikeRun,
    ppc_band_mean: np.ndarray,
    trial_spike_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select pooled low/high units and their exact illustrative trials.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable condition, site, epoch, and band axis definitions.
    prepared_phase : PreparedPhaseRun
        Full int64 trial-table row axis shared by cached phase/traces.
    prepared_spikes : PreparedSpikeRun
        Stable unit identities and event-relative spike seconds per trial.
    ppc_band_mean : numpy.ndarray
        Dimensionless float array with axes
        ``(unit, condition, site, epoch, band)``.
    trial_spike_counts : numpy.ndarray
        Nonnegative int64 array with axes
        ``(unit, condition, site, epoch, trial)`` in spike units.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Low/high stable-unit-id arrays followed by the legacy high-first trial,
        exact low-unit trial, and exact high-unit trial arrays. Every output has
        ``(condition, site, epoch, band)`` axes. Trial values are table-row ids;
        unavailable ids use ``""`` and unavailable trials use ``-1``.
    """
    output_shape = ppc_band_mean.shape[1:]
    low = np.full(output_shape, "", dtype="<U64")
    high = np.full(output_shape, "", dtype="<U64")
    illustrative = np.full(output_shape, -1, dtype=np.int64)
    illustrative_low = np.full(output_shape, -1, dtype=np.int64)
    illustrative_high = np.full(output_shape, -1, dtype=np.int64)
    for condition_index, site_index, epoch_index, band_index in np.ndindex(
        output_shape
    ):
        selection = spike_lfp_summary.select_ppc_exemplars(
            unit_ids=prepared_spikes.unit_ids,
            band_ppc=ppc_band_mean[
                :, condition_index, site_index, epoch_index, band_index
            ],
            trial_indices=prepared_phase.trial_indices,
            trial_spike_counts=trial_spike_counts[
                :, condition_index, site_index, epoch_index
            ],
        )
        index = (condition_index, site_index, epoch_index, band_index)
        if selection.low_unit_id is not None:
            low[index] = selection.low_unit_id
        if selection.high_unit_id is not None:
            high[index] = selection.high_unit_id
        if selection.low_unit_id in selection.illustrative_trial_index_by_unit:
            illustrative_low[index] = selection.illustrative_trial_index_by_unit[
                selection.low_unit_id
            ]
        if selection.high_unit_id in selection.illustrative_trial_index_by_unit:
            illustrative_high[index] = selection.illustrative_trial_index_by_unit[
                selection.high_unit_id
            ]
        for unit_id in (selection.high_unit_id, selection.low_unit_id):
            if unit_id in selection.illustrative_trial_index_by_unit:
                illustrative[index] = selection.illustrative_trial_index_by_unit[unit_id]
                break
    return low, high, illustrative, illustrative_low, illustrative_high


def _analysis_condition_membership(prepared: PreparedTrials) -> np.ndarray:
    """Return trial-by-condition masks after shared filter/objective/user gates."""
    shared = (
        prepared.filter_membership
        & prepared.objective_valid
        & ~prepared.user_excluded
    )
    return prepared.condition_membership & shared[:, None]


def _epoch_windows(config: LFPSummaryConfig) -> dict[str, tuple[float, float]]:
    """Return ordered whole/before/after half-open bounds in seconds."""
    windows = config.analysis_windows
    return {
        "whole": (windows.whole_start_s, windows.whole_stop_s),
        "before": (windows.before_start_s, windows.before_stop_s),
        "after": (windows.after_start_s, windows.after_stop_s),
    }


def _band_hilbert_traces(
    config: LFPSummaryConfig,
    prepared: PreparedPhaseRun,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cache-only bandpassed source traces and Hilbert phase in radians.

    Returns float32 arrays with axes ``(site, trial, band, time)``. Entirely
    unavailable source rows remain NaN and are never bridged by filtering.
    """
    shape = prepared.source_trace.shape[:2] + (
        len(config.phase.bands),
        prepared.source_trace.shape[2],
    )
    filtered = np.full(shape, np.nan, dtype=np.float32)
    phase_rad = np.full(shape, np.nan, dtype=np.float32)
    for site_index in range(shape[0]):
        for trial_index in range(shape[1]):
            source = prepared.source_trace[site_index, trial_index]
            if not np.isfinite(source).all():
                continue
            notched = lfp_spectrogram.apply_optional_60_hz_notch(
                source,
                sample_rate_hz=_CACHE_SAMPLE_RATE_HZ,
                enabled=config.phase.notch_enabled,
                quality_factor=config.phase.notch_quality_factor,
            )
            for band_index, band in enumerate(config.phase.bands):
                result = spike_lfp_hilbert_phase.compute_hilbert_phase_trace(
                    prepared.relative_time_s,
                    notched,
                    _CACHE_SAMPLE_RATE_HZ,
                    (band.lower_hz, band.upper_hz),
                )
                filtered[site_index, trial_index, band_index] = (
                    result.bandpassed_lfp.astype(np.float32)
                )
                phase_rad[site_index, trial_index, band_index] = (
                    result.phase_rad.astype(np.float32)
                )
    return filtered, phase_rad


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
    open_ephys_metadata_by_site: Mapping[str, dict[str, object]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load one exact pre-session interval through preflighted trace adapters.

    Parameters
    ----------
    config : LFPSummaryConfig
        Valid configuration with a positive pre-session duration in seconds.
    first_start_time_s : float
        Earliest finite absolute trial start in seconds, or NaN when absent.
    spikeglx_loader, open_ephys_loader : callable or None
        Optional normalized injected trace seams. Their returned time/value
        vectors are one-dimensional seconds/source-voltage axes and a rate in
        Hz. ``None`` selects production adapters.
    open_ephys_metadata_by_site : mapping[str, dict[str, object]] or None
        Preflighted normalized production Open Ephys sidecars keyed by stable
        site id. They contain no trace samples and are reused without reparsing.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], dict[str, numpy.ndarray]]
        Site-keyed pre-session voltage ``(sample,)`` arrays and matching
        relative-second coordinates. Missing first starts yield NaN traces.
    """
    duration_s = config.power.presession_reference_duration_s
    if np.isfinite(first_start_time_s):
        loaded = load_site_trial_traces(
            config.sites,
            np.array([0], dtype=np.int64),
            np.array([first_start_time_s], dtype=float),
            (-duration_s, 0.0),
            spikeglx_loader=spikeglx_loader,
            open_ephys_loader=open_ephys_loader,
            open_ephys_metadata_by_site=open_ephys_metadata_by_site,
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
        "filter_membership": prepared.prepared_trials.filter_membership.copy(),
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
