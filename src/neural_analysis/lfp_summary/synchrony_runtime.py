"""Phase preparation and Synchrony payload construction for LFP summaries."""

from __future__ import annotations

from functools import partial
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp import phase as lfp_phase_clustering
from src.neural_analysis.lfp import spectrogram as lfp_spectrogram
from src.neural_analysis.lfp.synchrony import (
    aggregate_trial_plv_bands,
    bootstrap_phase_clustering_bands,
    compute_phase_clustering_summary,
    compute_trial_plv_by_frequency,
)
from src.neural_analysis.lfp_summary.cache import load_or_initialize_manifest, write_component_transaction
from src.neural_analysis.lfp_summary.models import (
    LFPSiteConfig,
    LFPSummaryConfig,
    canonical_config_json,
    fingerprint_source_files,
)
from src.neural_analysis.lfp_summary.payloads import build_component_payload
from src.neural_analysis.lfp_summary.pipeline import ComponentPayload, PipelineDependencies
from src.neural_analysis.lfp_summary.preparation import (
    PreparedSiteTraces,
    build_common_event_grid,
    build_prepared_trials,
    load_site_trial_traces,
    preflight_open_ephys_site_metadata,
    validate_open_ephys_aligned_sync_path,
)
from src.neural_analysis.lfp_summary.runtime_common import (
    PreparedPhaseRun,
    _CACHE_SAMPLE_RATE_HZ,
    _alignment_times,
    _analysis_condition_membership,
    _resample_trace_rows,
    _validate_prepared_phase_run,
    _work_fingerprint,
)
from src.neural_analysis.lfp_summary.work_cache import (
    PreparedPhaseCache,
    load_prepared_phase_cache,
    write_prepared_phase_cache,
)
from src.neural_analysis.spike_lfp import hilbert as spike_lfp_hilbert_phase


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
