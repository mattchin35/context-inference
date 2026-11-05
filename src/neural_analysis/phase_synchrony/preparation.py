"""Independent calibrated source loading and drift-aware UTC waveform sampling."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.external_tools import readSGLX
from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp import phase as lfp_phase
from src.neural_analysis.lfp.spectrogram import (
    apply_optional_60_hz_notch, compute_decimation_factor, compute_wavelet_padding_s,
)
from src.neural_analysis.session_metadata import (
    ResolvedSession, SiteMetadata, resolve_probe_sources,
)
from src.neural_analysis.spike_behavior import loading as spike_loading

from .results import (
    INTERVAL_BOUNDS_S, WAVEFORM_ALIGNMENT_SEMANTICS,
    PHASE_ALIGNMENT_SEMANTICS, PHASE_SUPPORT_SEMANTICS,
    ContinuousSiteLFP, PhaseSynchronyConfig, SelectedUnitSpikes, SiteWaveforms, SitePhase,
    parse_unit_id,
)


def select_metadata_inputs(
    session: ResolvedSession,
    *,
    site_ids: tuple[str, ...] = (),
    pairs: tuple[tuple[str, str], ...] = (),
) -> tuple[tuple[SiteMetadata, ...], tuple[tuple[str, str], ...]]:
    """Select metadata site names/ordered pairs without reading numerical inputs.

    ``session`` is resolved session metadata; empty categorical selections use
    declared sites/pairs. If pairs are undeclared, distinct site combinations
    follow declaration order. Return (SiteMetadata tuple, ordered pair tuple),
    with original probe IDs and zero-based saved channels. Unknown IDs raise.
    """
    known_ids = {site.site_id for site in session.sites}
    unknown = set(site_ids) - known_ids
    if unknown:
        raise ValueError(f"Unknown metadata sites: {sorted(unknown)}")
    sites = tuple(site for site in session.sites if not site_ids or site.site_id in site_ids)
    selected_ids = {site.site_id for site in sites}
    default_pairs = session.site_pairs or tuple(combinations((site.site_id for site in session.sites), 2))
    selected_pairs = pairs or tuple(pair for pair in default_pairs if set(pair).issubset(selected_ids))
    for pair in selected_pairs:
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair).issubset(selected_ids):
            raise ValueError(f"Pair must name two distinct selected metadata sites: {pair}")
    return sites, selected_pairs


def product_source_ids(session: ResolvedSession, family: str, selection: tuple[str, ...]) -> dict:
    """Describe relevant metadata/source files without loading any scientific arrays.

    session is resolved metadata, family names a product category, and selection
    contains site IDs, ordered site pairs, a probe:cluster ID, or (unit,site).
    Return JSON-compatible source identities: paths, byte sizes, modification
    nanoseconds, explicit missing status, mapped channels and unit restrictions.
    Windowed products include the behavior trial table; native/unit preparations
    do not. Unknown selections raise ValueError. No clock/unit conversion occurs.
    """
    sources = {}
    site_ids = [] if family == "unit-spikes" else (list(selection[1:]) if family == "spike-ppc" else list(selection))
    sites = {site.site_id: site for site in session.sites}
    for site_id in site_ids:
        if site_id not in sites:
            raise ValueError(f"Unknown metadata LFP site: {site_id}")
        site = sites[site_id]
        probe = resolve_probe_sources(session, site.probe_id)
        sources[site_id] = {"probe_id": site.probe_id, "saved_channel_index": site.saved_channel_index,
                            "acquisition_family": probe.acquisition_family,
                            "files": [_file_identity(path) for path in (probe.lfp_file, probe.lfp_metadata_file, probe.alignment_file)]}
    if family in {"unit-spikes", "spike-ppc"}:
        probe_id, _ = parse_unit_id(selection[0])
        try:
            probe = resolve_probe_sources(session, probe_id)
        except KeyError as error:
            raise ValueError(f"Unknown selected unit probe: {probe_id}") from error
        sources[selection[0]] = {"unit_channels": probe.unit_channels,
                                "files": [_file_identity(path) for path in (probe.alignment_file, probe.spike_clusters_file,
                                                                          probe.cluster_info_file, probe.channel_quality_file)]}
    if family not in {"native-lfp", "unit-spikes"}:
        sources["behavior_trials"] = _file_identity(session.behavior.trial_table_file)
    return sources


def component_availability(session: ResolvedSession, config: PhaseSynchronyConfig) -> dict[str, str | None]:
    """Return path-level readiness by requested component/selection, without data I/O.

    Inputs are resolved metadata and categorical/numeric config. Dictionary
    keys identify ``power:site``, ``coherence:A:B``, ``lfp-phase:A:B``, or
    ``spike-ppc:probe:cluster:site``. Values are None or missing-input reasons.
    This checks file existence only, not sidecar contents or signal validity;
    actual loading validates those independently. Unrelated probes never gate.
    """
    sites, pairs = select_metadata_inputs(session, site_ids=config.sites, pairs=config.pairs)
    site_status = {}
    for site in session.sites:
        probe = resolve_probe_sources(session, site.probe_id)
        required = [probe.lfp_file, probe.lfp_metadata_file]
        if probe.acquisition_family == "open_ephys":
            required.append(probe.alignment_file)
        site_status[site.site_id] = _missing_files(required)
    status = {}
    for component in config.components:
        if component == "power":
            status.update({f"power:{site.site_id}": site_status[site.site_id] for site in sites})
        elif component in {"coherence", "lfp-phase"}:
            for left, right in pairs:
                status[f"{component}:{left}:{right}"] = site_status[left] or site_status[right]
        else:
            if config.lfp_site not in site_status:
                raise ValueError(f"Unknown metadata site: {config.lfp_site}")
            for unit_id in config.unit_ids:
                probe_id, _ = parse_unit_id(unit_id)
                try:
                    probe = resolve_probe_sources(session, probe_id)
                    missing = _missing_files([
                        probe.alignment_file, probe.spike_clusters_file,
                        probe.cluster_info_file, probe.channel_quality_file,
                    ])
                except KeyError:
                    missing = f"Unknown probe: {probe_id}"
                status[f"spike-ppc:{unit_id}:{config.lfp_site}"] = site_status[config.lfp_site] or missing
    return status


def load_site_lfp(
    session: ResolvedSession,
    site_id: str,
    *,
    start_sample: int,
    stop_sample: int,
    utc_offset_hours: float = 0.0,
) -> ContinuousSiteLFP:
    """Read one named native channel and its clock; apply calibration exactly once.

    ``session`` resolves metadata names; ``start_sample:stop_sample`` is a
    half-open integer native-LFP range, bounded by the caller's required real
    filter/metric support. ``utc_offset_hours`` adds hours to LFP anchors only.
    Return calibrated float (sample,) uV values, native indices/validity and
    IRIG UTC seconds. Neither notch nor UTC resampling occurs here. Known AP
    nominal acquisition rate from sync metadata converts AP indices without
    erasing IRIG drift. Missing/unverifiable nominal AP Hz is rejected: a rate
    inferred from UTC anchor slopes cannot establish the resampling ratio.
    """
    sites, _ = select_metadata_inputs(session, site_ids=(site_id,))
    site = sites[0]
    probe = resolve_probe_sources(session, site.probe_id)
    if not np.isfinite(utc_offset_hours):
        raise ValueError("utc_offset_hours must be finite")
    if probe.lfp_file is None:
        raise ValueError(f"Missing LFP source for {site_id}")
    if probe.acquisition_family == "open_ephys":
        metadata = lfp_loading.load_open_ephys_lfp_metadata(probe.lfp_file)
        rate_hz = float(metadata["sampling_frequency_hz"])
        if probe.alignment_file is None:
            raise ValueError(f"Missing alignment source for {site_id}")
        with np.load(probe.alignment_file, allow_pickle=True) as archive:
            sync_meta = archive["meta"].item()
        acquisition_rate = sync_meta.get("sample_rate_hz")
        if acquisition_rate is None or isinstance(acquisition_rate, bool):
            raise ValueError("Open Ephys synchronization needs verified nominal sample_rate_hz")
        acquisition_rate = float(acquisition_rate)
        if not np.isfinite(acquisition_rate) or acquisition_rate <= 0:
            raise ValueError("Open Ephys nominal acquisition sample_rate_hz must be finite and positive")
        sync = lfp_loading.build_open_ephys_lfp_irig_df(
            probe.alignment_file, rate_hz,
            continuous_sample_rate_hz=acquisition_rate,
            utc_offset_hours=utc_offset_hours,
        )
        values, file_rate_hz = lfp_loading.read_open_ephys_lfp_channel_window(
            probe.lfp_file, site.saved_channel_index, start_sample, stop_sample,
        )
        synchronization = {
            "file": _file_identity(probe.alignment_file),
            "acquisition_rate_hz": acquisition_rate,
            "acquisition_rate_source": "sync_metadata",
            "continuous_start_sample_ix": sync_meta.get("continuous_start_sample_ix", 0),
            "utc_offset_hours": utc_offset_hours,
        }
        calibration = lfp_loading.OPEN_EPHYS_AFFINE_UV_SEMANTICS
    elif probe.acquisition_family == "spikeglx":
        metadata = lfp_loading.load_lfp_metadata(probe.lfp_file)
        rate_hz = float(readSGLX.SampRate(metadata))
        sync, sync_rate_hz = lfp_loading.decode_lfp_sync(
            probe.lfp_file, utc_offset_hours=utc_offset_hours,
        )
        if sync_rate_hz != rate_hz:
            raise ValueError("SpikeGLX synchronization sample rate disagrees with metadata")
        values, file_rate_hz = lfp_loading.read_lfp_saved_channel_window(
            probe.lfp_file, site.saved_channel_index, start_sample, stop_sample,
        )
        synchronization = {"file": _file_identity(probe.lfp_file), "utc_offset_hours": utc_offset_hours}
        calibration = "spikeglx_gain_corrected_uV_v1"
    else:
        raise ValueError(f"Unsupported acquisition family: {probe.acquisition_family}")
    if file_rate_hz != rate_hz:
        raise ValueError("LFP reader sample rate disagrees with metadata")
    return ContinuousSiteLFP(
        site.site_id, site.probe_id, site.saved_channel_index,
        np.arange(start_sample, stop_sample, dtype=np.int64), values,
        np.isfinite(values), rate_hz, sync,
        {"voltage_unit": "uV", "calibration_semantics": calibration,
         "source": _file_identity(probe.lfp_file),
         "source_preprocessing": metadata, "synchronization": synchronization,
         "notch_enabled": False,
         "loaded_sample_bounds": (start_sample, stop_sample)},
    )


def preprocess_site_lfp(source: ContinuousSiteLFP, config: PhaseSynchronyConfig) -> ContinuousSiteLFP:
    """Apply one continuous native-rate notch before UTC sampling, preserving uV.

    Input is a bounded native (sample,) source with filter context and config.
    Return a new source with unchanged native indices/clock, copied voltages
    and boolean support. Never filter across invalid values or missing native
    indices. Valid runs shorter than the existing notch's padding requirement
    become unavailable locally. An already matching notch is reused; changing
    or undoing its settings requires reloading the original source.
    """
    _validate_source(source)
    metadata = dict(source.meta)
    already_notched = bool(metadata.get("notch_enabled", False))
    if already_notched:
        if not config.notch_enabled or metadata.get("notch_quality_factor") != config.notch_quality_factor:
            raise ValueError("Reload native source to change already applied notch settings")
        return replace(source, values_uv=source.values_uv.copy(), valid=source.valid.copy(), meta=metadata)
    values = np.asarray(source.values_uv, dtype=float).copy()
    valid = np.asarray(source.valid, dtype=bool).copy() & np.isfinite(values)
    if config.notch_enabled:
        # Gaps split continuous filter runs; one failed short run cannot discard another.
        positions = np.flatnonzero(valid)
        split_points = np.flatnonzero(
            (np.diff(positions) != 1) | (np.diff(source.sample_indices[positions]) != 1)
        ) + 1
        for run in np.split(positions, split_points):
            if not run.size:
                continue
            if run.size <= 9:  # Existing second-order notch filtfilt uses 3 * max(len(a), len(b)).
                valid[run] = False
                continue
            values[run] = apply_optional_60_hz_notch(
                values[run], source.sample_rate_hz, enabled=True,
                quality_factor=config.notch_quality_factor,
            )
    metadata.update({"notch_enabled": config.notch_enabled, "notch_hz": 60.0,
                     "notch_quality_factor": config.notch_quality_factor,
                     "preprocessing_order": ("native_calibration", "continuous_notch")})
    return replace(source, values_uv=values, valid=valid, meta=metadata)


def sample_site_lfp_utc(source: ContinuousSiteLFP, target_time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linearly sample native uV at float (target,) UTC-second timestamps.

    Evaluate this source's IRIG UTC-to-native-index map at every target.
    Return float uV values and bool validity with shape (target,), invalid
    values NaN. Exact samples use their own support; fractional samples need
    both contiguous valid native neighbors. No signal/synchronization
    extrapolation. UTC subtraction uses one real anchor origin, never default
    relative-tolerance comparisons on billion-second timestamps. A target that
    equals the representable UTC timestamp of its nearest native sample uses
    that exact sample; adjacent distinguishable UTC floats are not snapped.
    """
    targets = np.asarray(target_time_s, dtype=float)
    if targets.ndim != 1:
        raise ValueError("target_time_s must have one target axis")
    origin, anchor_time, anchor_indices = _clock_coordinates(source)
    return _sample_relative_utc(
        source, targets - origin, anchor_time, anchor_indices,
        absolute_targets=(origin, targets),
    )


def prepare_site_lfp(source: ContinuousSiteLFP, choices: pd.DataFrame, *, interval: str = "primary") -> SiteWaveforms:
    """Extract exact choice-centered native-rate (trial,time) uV rows and masks.

    ``source`` is already calibrated/continuously preprocessed; ``choices``
    contains original trial_id/row_position and UTC-second choice_time columns.
    ``interval`` selects primary [-1,+1) or secondary [-2,+2) seconds. Return
    SiteWaveforms without dropping any choice. Invalid support stays local to
    its samples/trial/interval. Targets use relative arithmetic to preserve
    the uniform choice grid without first rounding each target to UTC floats.
    """
    start_s, stop_s = interval_bounds(interval)
    rate_hz = source.sample_rate_hz
    count_float = (stop_s - start_s) * rate_hz
    count = round(count_float)
    if count < 1 or not np.isclose(count_float, count, rtol=0, atol=1e-9):
        raise ValueError("Interval duration is not representable at the native sample rate")
    required = {"trial_id", "row_position", "choice_time"}
    if not required.issubset(choices.columns):
        raise ValueError(f"Choices need columns {sorted(required)}")
    choice_times = choices.choice_time.to_numpy(dtype=float)
    if not np.isfinite(choice_times).all():
        raise ValueError("choice_time must contain finite UTC seconds")
    relative = start_s + np.arange(count, dtype=float) / rate_hz
    origin, anchor_time, anchor_indices = _clock_coordinates(source)
    values = np.full((len(choices), count), np.nan)
    valid = np.zeros(values.shape, dtype=bool)
    for row, choice in enumerate(choice_times):
        values[row], valid[row] = _sample_relative_utc(
            source, (choice - origin) + relative, anchor_time, anchor_indices,
        )
    metadata = dict(source.meta)
    metadata.update({"waveform_alignment_semantics": WAVEFORM_ALIGNMENT_SEMANTICS,
                     "interval": interval, "interval_bounds_s": (start_s, stop_s),
                     "utc_origin_s": origin, "sample_rate_hz": rate_hz,
                     "axes": ("trial", "time"), "voltage_unit": "uV"})
    return SiteWaveforms(
        source.site_id, source.probe_id, source.saved_channel_index, values, valid,
        relative, choice_times.copy(), choices.trial_id.to_numpy(copy=True),
        choices.row_position.to_numpy(dtype=np.int64, copy=True), rate_hz, metadata,
    )


def prepare_site_phase(
    source: ContinuousSiteLFP,
    choices: pd.DataFrame,
    config: PhaseSynchronyConfig,
    *,
    interval: str = "primary",
) -> SitePhase:
    """Prepare drift-corrected Morlet unit phase before extracting choice windows.

    Input is bounded calibrated native (sample,) uV with independent IRIG clock,
    real padding and masks, plus original trial IDs/rows/UTC-second choices.
    Apply/reuse the configured continuous notch once. Resample each bounded
    block onto uniform UTC native-rate voltages BEFORE existing integer
    decimation/Morlet. Split invalid source gaps into independent valid runs;
    frequency-specific real wavelet padding excludes run edges. Return complex64
    (frequency,trial,time) unit phase and validity on the exact 500-Hz choice
    grid, with one real right guard outside the half-open metric mask.

    No phase values are manufactured outside real source/transform support.
    Boundary trials/secondary support loss never discard other supported data.
    Morlet settings are the established Gaussian 1.5/window 1/precision 16/L1;
    all actual support, clocks and amplitude settings are retained in metadata.
    """
    source = preprocess_site_lfp(source, config)
    origin, anchor_time, anchor_indices = _clock_coordinates(source)
    start, stop = interval_bounds(interval)
    frequencies = np.asarray(config.phase_frequencies_hz, dtype=float)
    factor = compute_decimation_factor(source.sample_rate_hz, config.phase_rate_hz, float(frequencies.max()))
    actual_rate = source.sample_rate_hz / factor
    if actual_rate != config.phase_rate_hz or actual_rate != 500.0:
        raise ValueError("Actual integer-decimated phase rate must be exactly 500 Hz")
    times = choices.choice_time.to_numpy(dtype=float)
    positions = np.arange(len(choices), dtype=int)
    padding = compute_wavelet_padding_s(float(frequencies.min()), 1.0)
    frequency_padding = 8.0 / frequencies
    metric_count = round((stop - start) * actual_rate)
    relative = start + np.arange(metric_count + 1) / actual_rate
    metric_mask = np.arange(metric_count + 1) < metric_count
    shape = (len(frequencies), len(choices), len(relative))
    phase = np.full(shape, np.nan + 1j * np.nan, dtype=np.complex64)
    valid = np.zeros(shape, dtype=bool)
    blocks = lfp_phase.plan_continuous_processing_blocks(
        times - origin, positions, (start, stop + 1 / actual_rate), padding, 120.0,
    )
    block_metadata = []
    for block in blocks:
        native_count = int(np.ceil((block.load_end_s - block.load_start_s) * source.sample_rate_hz)) + 1
        block_time = np.arange(native_count) / source.sample_rate_hz
        voltages, support = _sample_relative_utc(
            source, block.load_start_s + block_time, anchor_time, anchor_indices,
        )
        for first, last in _valid_runs(support):
            # The existing IIR anti-alias decimator needs at least 28 native
            # samples. Short fragments cannot supply real wavelet support.
            if last - first <= 27 or block_time[last - 1] - block_time[first] <= 2 * frequency_padding.min():
                continue
            run_origin = block_time[first]
            run_time = np.arange(last - first) / source.sample_rate_hz
            transformed = lfp_phase.compute_wavelet_coefficients(
                run_time, voltages[first:last], source.sample_rate_hz, frequencies,
                gaussian_width=1.5, window_length=1.0, precision=16, norm="l1",
                target_sample_rate_hz=actual_rate, notch_60_hz=False,
            )
            if transformed.sample_rate_hz != actual_rate:
                raise ValueError("Morlet returned an incompatible actual phase rate")
            vectors, coefficient_valid = lfp_phase.normalize_wavelet_phase(
                transformed.coefficients,
                minimum_relative_magnitude=config.numerical_amplitude_threshold,
            )
            coefficient_valid &= np.abs(transformed.coefficients) >= config.absolute_amplitude_threshold
            # Only real continuous run support can justify a coefficient. A
            # source gap excludes its wavelet footprint, not merely one sample.
            coefficient_time = transformed.time_s
            coefficient_valid &= (
                (coefficient_time[None, :] >= frequency_padding[:, None])
                & (coefficient_time[None, :] <= run_time[-1] - frequency_padding[:, None])
            )
            for position in block.trial_indices:
                target = (times[position] - origin) - block.load_start_s - run_origin + relative
                sampled, sampled_valid = _interpolate_phase_vectors(
                    coefficient_time, vectors, coefficient_valid, target,
                )
                phase[:, position][sampled_valid] = sampled[sampled_valid].astype(np.complex64)
                valid[:, position] |= sampled_valid
            block_metadata.append({
                "clock_utc_origin_s": origin,
                "origin_relative_s": block.load_start_s + run_origin,
                "support_relative_bounds_s": (block.load_start_s + run_origin,
                                              block.load_start_s + block_time[last - 1]),
                "utc_origin_s": origin + (block.load_start_s + run_origin),
                "support_bounds_s": (origin + (block.load_start_s + run_origin),
                                     origin + (block.load_start_s + block_time[last - 1])),
                "native_rate_hz": source.sample_rate_hz, "phase_rate_hz": actual_rate,
                "native_sample_count": last - first,
            })
    metadata = dict(source.meta)
    metadata.update({
        "phase_support_semantics": PHASE_SUPPORT_SEMANTICS,
        "phase_alignment_semantics": PHASE_ALIGNMENT_SEMANTICS,
        "numerical_amplitude_threshold": config.numerical_amplitude_threshold,
        "absolute_amplitude_threshold": config.absolute_amplitude_threshold,
        "absolute_amplitude_unit": "uV Morlet coefficient magnitude with L1 normalization",
        "morlet": {"gaussian_width": 1.5, "window_length": 1.0, "precision": 16, "norm": "l1"},
        "padding_s": padding, "frequency_padding_s": frequency_padding.tolist(),
        "blocks": block_metadata, "phase_rate_hz": actual_rate,
        "native_rate_hz": source.sample_rate_hz, "decimation_factor": factor,
        "interval": interval, "interval_bounds_s": (start, stop),
        "synchronization_anchors": source.irig_df[["sample_ix", "utc_unix"]].to_dict("list"),
        "operation_order": ("native_calibration", "continuous_notch", "irig_uniform_utc_sampling",
                            "anti_alias_decimation", "morlet", "unit_phase", "choice_grid_sampling"),
        "axes": ("frequency", "trial", "time"), "guard_samples": 1,
    })
    return SitePhase(
        source.site_id, source.probe_id, source.saved_channel_index, frequencies,
        phase, valid, relative, metric_mask, actual_rate,
        choices.trial_id.to_numpy(copy=True), choices.row_position.to_numpy(dtype=np.int64, copy=True),
        times.copy(), metadata,
    )


def sample_site_phase_at_spikes(
    site_phase: SitePhase, trial_position: int, spike_times_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample prepared unit phase at already synchronized (spike,) UTC seconds.

    ``trial_position`` indexes the saved trial axis. Return complex128 unit
    vectors and bool validity (frequency,spike), with NaN vectors outside real
    support. Exact source timestamps require only their own validity; fractional
    timestamps need both neighbors. Interpolate normalized vectors and normalize
    again; never interpolate wrapped angles or fabricate coefficient amplitudes.
    The right guard is usable support but window selection remains half-open.
    Forward UTC equality recovers representable exact hits without snapping
    distinct fractional timestamps. No second LFP/spike clock correction occurs.
    """
    spikes = np.asarray(spike_times_s, dtype=float)
    if spikes.ndim != 1 or not np.isfinite(spikes).all():
        raise ValueError("Spike timestamps must be a finite (spike,) UTC-second vector")
    if not 0 <= trial_position < len(site_phase.choice_times_s):
        raise ValueError("trial_position is outside the saved trial axis")
    origin = site_phase.choice_times_s[trial_position]
    source_time = site_phase.relative_time_s
    targets = spikes - origin
    nearest = np.rint((targets - source_time[0]) * site_phase.sample_rate_hz).astype(np.int64)
    inside = (nearest >= 0) & (nearest < len(source_time))
    candidate = np.clip(nearest, 0, len(source_time) - 1)
    exact_utc = inside & (origin + source_time[candidate] == spikes)
    targets[exact_utc] = source_time[candidate[exact_utc]]
    return _interpolate_phase_vectors(
        source_time, site_phase.phase[:, trial_position],
        site_phase.valid[:, trial_position], targets,
    )


def validate_pair_sample_rates(site_a: ContinuousSiteLFP, site_b: ContinuousSiteLFP) -> None:
    """Require equal finite native Hz rates for two sources; return None or raise.

    No resampling or input mutation occurs. Independent site power remains
    possible when this pair's coherence rate contract is unsupported.
    """
    rates = np.asarray([site_a.sample_rate_hz, site_b.sample_rate_hz])
    if not np.isfinite(rates).all() or np.any(rates <= 0) or rates[0] != rates[1]:
        raise ValueError(f"Unsupported native sample rate mismatch: {rates.tolist()}")


def load_selected_unit_spikes(session: ResolvedSession, unit_id: str) -> SelectedUnitSpikes:
    """Load one explicit probe:cluster with existing Phy/channel-quality selection.

    Inputs are resolved metadata and categorical unit ID. Return float (spike,)
    synchronized UTC seconds, preserving all selected timestamps without bins
    or a second clock correction. Only good/mua clusters on good inside-brain
    channels satisfying metadata unit_channels qualify. Missing/excluded units
    raise rather than substituting another. No spike-count threshold is used.
    """
    probe_id, cluster_id = parse_unit_id(unit_id)
    try:
        probe = resolve_probe_sources(session, probe_id)
    except KeyError as error:
        raise ValueError(f"Unknown probe: {probe_id}") from error
    if probe.channel_quality_file is None or probe.sorter_directory is None or probe.alignment_file is None:
        raise ValueError(f"Missing selected-unit sources for {unit_id}")
    quality = spike_loading.load_channel_quality(probe.channel_quality_file)
    channels = spike_loading.select_channels_from_quality(quality)
    if probe.unit_channels is not None:
        channels = np.intersect1d(channels, probe.unit_channels)
    assignments, clusters = spike_loading.load_sorter_metadata(probe.sorter_directory)
    selected = spike_loading.filter_cluster_metadata(clusters, channels)
    if selected.cluster_id.duplicated().any():
        raise ValueError("Duplicate selected cluster IDs")
    if cluster_id not in selected.cluster_id.to_numpy():
        raise ValueError(f"Unknown or quality-excluded unit: {unit_id}")
    times = spike_loading.load_aligned_spikes(probe.alignment_file)
    spike_loading.validate_aligned_spike_inputs(times, assignments)
    unit_times = times[assignments == cluster_id].copy()
    if not np.isfinite(unit_times).all():
        raise ValueError(f"Nonfinite aligned timestamps for {unit_id}")
    return SelectedUnitSpikes(
        unit_id, probe_id, cluster_id, unit_times,
        {"time_unit": "UTC Unix seconds", "channels": channels.tolist(),
         "quality_rules": "Phy good/mua; good inside-brain channels; metadata unit_channels",
         "sources": [_file_identity(path) for path in (
             probe.alignment_file, probe.spike_clusters_file,
             probe.cluster_info_file, probe.channel_quality_file,
         )]},
    )


def select_unit_window_spikes(unit: SelectedUnitSpikes, choice_time_s: float, *, interval: str = "primary") -> np.ndarray:
    """Return this unit's (selected_spike,) UTC seconds inside [choice-a,choice+a).

    ``unit.times_s`` is float (spike,) UTC seconds; choice is finite UTC seconds;
    interval is primary/secondary. Preserve source precision/order, never pool
    units, round to bins, or correct the synchronized spike clock again.
    """
    start_s, stop_s = interval_bounds(interval)
    if not np.isfinite(choice_time_s):
        raise ValueError("choice_time_s must be finite UTC seconds")
    times = np.asarray(unit.times_s)
    return times[(times >= choice_time_s + start_s) & (times < choice_time_s + stop_s)].copy()


def interval_bounds(interval: str) -> tuple[float, float]:
    """Map categorical primary/secondary to half-open relative-second (start,stop)."""
    if interval not in INTERVAL_BOUNDS_S:
        raise ValueError("interval must be primary or secondary")
    return INTERVAL_BOUNDS_S[interval]


# Source/clock helpers: validation and sampling preserve every native axis.
def _valid_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open integer (start,stop) runs of a bool (time,) support vector."""
    starts = np.flatnonzero(mask & np.concatenate(([True], ~mask[:-1])))
    stops = np.flatnonzero(mask & np.concatenate((~mask[1:], [True]))) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def _interpolate_phase_vectors(
    source_time_s: np.ndarray, vectors: np.ndarray, valid: np.ndarray, targets_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate relative-second phase (frequency,time) at (target,); return unit vectors/mask.

    This narrow mask-aware adapter follows the existing complex-vector sampler's
    normalization, with the exact-hit validity contract used by PPC kernels.
    Inputs are already normalized phase, never amplitude-weighted coefficients.
    """
    times = np.asarray(source_time_s, dtype=float)
    phase = np.asarray(vectors, dtype=np.complex128)
    source_valid = np.asarray(valid, dtype=bool) & np.isfinite(phase) & (np.abs(phase) > 0)
    if (times.ndim != 1 or not times.size or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or phase.shape != source_valid.shape
            or phase.ndim != 2 or phase.shape[1] != times.size):
        raise ValueError("Phase vectors/validity must share an increasing (frequency,time) axis")
    right = np.searchsorted(times, targets_s)
    safe = np.clip(right, 0, times.size - 1)
    tolerance = 8 * np.finfo(float).eps * max(1.0, np.max(np.abs(times)))
    exact = (right < times.size) & np.isclose(times[safe], targets_s, rtol=0, atol=tolerance)
    output = np.full((phase.shape[0], len(targets_s)), np.nan + 1j * np.nan)
    supported = np.zeros(output.shape, dtype=bool)
    if exact.any():
        supported[:, exact] = source_valid[:, safe[exact]]
        output[:, exact] = phase[:, safe[exact]]
    between = ~exact & (right > 0) & (right < times.size)
    if between.any():
        left = right[between] - 1
        upper = right[between]
        weight = (targets_s[between] - times[left]) / (times[upper] - times[left])
        output[:, between] = phase[:, left] * (1 - weight) + phase[:, upper] * weight
        supported[:, between] = source_valid[:, left] & source_valid[:, upper]
    magnitude = np.abs(output)
    supported &= np.isfinite(magnitude) & (magnitude > 0)
    output = np.divide(output, magnitude, out=np.full_like(output, np.nan + 1j * np.nan), where=supported)
    return output, supported


def _validate_source(source: ContinuousSiteLFP) -> None:
    """Validate native (sample,) values/mask/indices and positive Hz; return None."""
    indices = np.asarray(source.sample_indices)
    if (indices.ndim != 1 or not indices.size
            or np.asarray(source.values_uv).shape != indices.shape
            or np.asarray(source.valid).shape != indices.shape
            or not np.isfinite(indices).all() or np.any(np.diff(indices) <= 0)
            or np.any(indices != np.floor(indices))):
        raise ValueError("Native samples, uV values and validity need matching increasing sample axes")
    if not np.isfinite(source.sample_rate_hz) or source.sample_rate_hz <= 0:
        raise ValueError("Native sample rate must be finite and positive")


def _clock_coordinates(source: ContinuousSiteLFP) -> tuple[float, np.ndarray, np.ndarray]:
    """Return UTC origin scalar, relative-second anchors and native (anchor,) indices."""
    _validate_source(source)
    times = source.irig_df.utc_unix.to_numpy(dtype=float)
    indices = source.irig_df.sample_ix.to_numpy(dtype=float)
    if (times.ndim != 1 or times.size < 2 or indices.shape != times.shape
            or not np.isfinite(times).all() or not np.isfinite(indices).all()
            or np.any(np.diff(times) <= 0) or np.any(np.diff(indices) <= 0)):
        raise ValueError("IRIG needs finite strictly increasing UTC and native sample anchors")
    return float(times[0]), times - times[0], indices


def _sample_relative_utc(
    source: ContinuousSiteLFP,
    targets_relative_s: np.ndarray,
    anchor_relative_s: np.ndarray,
    anchor_indices: np.ndarray,
    *,
    absolute_targets: tuple[float, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (target,) relative seconds via (anchor,) clock; return uV/mask vectors.

    Optional absolute_targets holds UTC origin and (target,) UTC floats when
    targets arrived already quantized to that representation. Recover an exact
    native hit only when forward clock mapping rounds to that identical float.
    Relative grids omit this option and retain their finer relative precision.
    """
    fractional = np.interp(targets_relative_s, anchor_relative_s, anchor_indices, left=np.nan, right=np.nan)
    if absolute_targets is not None:
        origin, utc_targets = absolute_targets
        nearest = np.rint(fractional)
        nearest_relative_s = np.interp(nearest, anchor_indices, anchor_relative_s, left=np.nan, right=np.nan)
        exact_utc = np.isfinite(fractional) & (origin + nearest_relative_s == utc_targets)
        fractional[exact_utc] = nearest[exact_utc]
    native = np.asarray(source.sample_indices)
    source_valid = np.asarray(source.valid, dtype=bool) & np.isfinite(source.values_uv)
    right = np.searchsorted(native, fractional)
    safe_right = np.clip(right, 0, native.size - 1)
    exact = np.isfinite(fractional) & (right < native.size) & np.isclose(
        native[safe_right], fractional, rtol=0, atol=1e-9,
    )
    values = np.full(fractional.shape, np.nan)
    valid = exact & source_valid[safe_right]
    values[valid] = source.values_uv[safe_right[valid]]
    left = np.clip(right - 1, 0, native.size - 1)
    between = (~exact & np.isfinite(fractional) & (right > 0) & (right < native.size)
               & (native[safe_right] - native[left] == 1)
               & source_valid[left] & source_valid[safe_right])
    weight = fractional[between] - native[left[between]]
    values[between] = (source.values_uv[left[between]] * (1 - weight)
                       + source.values_uv[safe_right[between]] * weight)
    valid[between] = True
    return values, valid


def _missing_files(paths: list[Path | None]) -> str | None:
    """Return missing path descriptions or None for file-list input; no data units."""
    missing = [str(path) if path is not None else "unspecified file" for path in paths if path is None or not path.is_file()]
    return "Missing inputs: " + ", ".join(missing) if missing else None


def _file_identity(path: Path | None) -> dict:
    """Return absolute file stat identity or explicit missing status, without opening arrays."""
    if path is None:
        return {"path": None, "missing": True}
    if not path.is_file():
        return {"path": str(path.resolve()), "missing": True}
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
