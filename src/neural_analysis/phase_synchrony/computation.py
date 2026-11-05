"""Within-choice Welch spectra, integrated band power and coherence."""

from dataclasses import asdict
from hashlib import sha256
import json

import numpy as np
from scipy import signal

from src.neural_analysis.lfp.config import FrequencyBandConfig, PowerAnalysisConfig
from src.neural_analysis.lfp.power import compute_welch_psd, mean_band_power_linear

from .results import (
    ANALYSIS_VERSION, INTERVAL_BOUNDS_S, WAVEFORM_ALIGNMENT_SEMANTICS,
    BandPower, PhaseSynchronyConfig, SitePSD, SiteWaveforms, WelchCoherence,
)


def compute_site_psd(waveforms: SiteWaveforms, config: PhaseSynchronyConfig) -> SitePSD:
    """Compute one Welch density per complete trial from native-rate uV rows.

    Inputs are already continuously preprocessed (trial,time) voltages/masks
    with exact native-rate choice-relative seconds and the effective config.
    Return SitePSD with actual Hz coordinates and (trial,frequency) uV^2/Hz.
    Partial/nonfinite rows remain unavailable; valid zero voltage is retained.
    No second notch, interpolation, trial pooling or segment shortening occurs.
    At least two complete configured segments must fit the metric row.
    """
    settings, segment_count = _waveform_welch_settings(waveforms, config)
    frequencies = np.fft.rfftfreq(settings["nperseg"], d=1.0 / waveforms.sample_rate_hz)
    supported = np.asarray(waveforms.valid, dtype=bool) & np.isfinite(waveforms.values_uv)
    sample_count = supported.sum(axis=1, dtype=np.int64)
    valid = supported.all(axis=1)
    density = np.full((len(valid), len(frequencies)), np.nan)
    power_config = PowerAnalysisConfig(
        welch_window_s=config.welch_segment_s,
        welch_overlap_fraction=config.welch_overlap_fraction,
        notch_enabled=False,
    )
    for trial in np.flatnonzero(valid):
        actual_hz, density[trial] = compute_welch_psd(
            waveforms.values_uv[trial], waveforms.sample_rate_hz, power_config,
        )
        if not np.array_equal(actual_hz, frequencies):
            raise ValueError("Welch returned an unexpected native frequency grid")
    metadata = dict(waveforms.meta)
    metadata.update({
        "analysis_version": ANALYSIS_VERSION, "site_id": waveforms.site_id,
        "probe_id": waveforms.probe_id, "saved_channel_index": waveforms.saved_channel_index,
        "sample_rate_hz": waveforms.sample_rate_hz, "welch": settings,
        "psd_unit": "uV^2/Hz", "axes": ("trial", "frequency"),
        "invalid_trial_reasons": tuple("" if okay else "incomplete waveform support" for okay in valid),
    })
    return SitePSD(
        frequencies, density, valid, sample_count,
        np.where(valid, segment_count, 0).astype(np.int64),
        waveforms.trial_ids.copy(), waveforms.row_positions.copy(),
        waveforms.choice_times_s.copy(), _psd_identity(waveforms, settings), metadata,
    )


def compute_band_power(psd: SitePSD, config: PhaseSynchronyConfig) -> BandPower:
    """Integrate each (trial,frequency) uV^2/Hz PSD into (trial,band) uV^2.

    Inputs are a shared SitePSD and configured Hz bands/exclusions. Return
    BandPower with retained bandwidths, original trial axes and validity. Reuse
    the existing disjoint-interval helper's one-sided gap-boundary convention;
    its mean density times bandwidth is total power. Required nonfinite
    retained support invalidates only its band/trial. Excluded bins never
    contaminate the integral. No division by window duration occurs.
    """
    if psd.meta.get("psd_unit") != "uV^2/Hz":
        raise ValueError("Band power requires verified calibrated uV^2/Hz density")
    values = np.full((len(psd.trial_ids), len(config.bands)), np.nan)
    bandwidth = np.zeros(len(config.bands))
    for index, band in enumerate(config.bands):
        mean_density, width = mean_band_power_linear(psd.psd_uv2_per_hz, psd.frequency_hz, band)
        values[:, index] = np.asarray(mean_density) * width
        bandwidth[index] = width
    valid = np.isfinite(values) & psd.valid[:, None]
    values[~valid] = np.nan
    metadata = dict(psd.meta)
    metadata.update({
        "power_unit": "uV^2", "axes": ("trial", "band"),
        "source_psd_id": psd.identity, "bands": [asdict(band) for band in config.bands],
        "integration": "mean_band_power_linear_times_retained_bandwidth_v1",
        "validity": "complete waveform and finite required retained PSD support",
    })
    return BandPower(
        values, valid, bandwidth, tuple(band.name for band in config.bands),
        psd.sample_count.copy(), psd.segment_count.copy(), psd.trial_ids.copy(),
        psd.row_positions.copy(), psd.choice_times_s.copy(), metadata,
    )


def compute_welch_coherence(
    site_a: SiteWaveforms,
    site_b: SiteWaveforms,
    config: PhaseSynchronyConfig,
    *,
    psd_a: SitePSD | None = None,
    psd_b: SitePSD | None = None,
) -> WelchCoherence:
    """Compute magnitude-squared coherence per matched native-rate trial window.

    Site inputs are already continuously preprocessed (trial,time) uV with
    identical UTC choices, IDs, native rates and relative-time grids. Optional
    shared PSDs must identify those exact samples/support/settings/axes; a
    mismatch raises instead of silently recomputing. Return dimensionless
    (trial,frequency) coherence/validity and strict (trial,band) summaries.
    Average matched cross/auto spectra before division. Zero/nonfinite spectral
    denominators are unavailable per frequency; no biological amplitude cutoff
    is imposed. Only roundoff excursions within 64 machine epsilons of [0,1]
    are clipped. No observations are pooled between trials.
    """
    settings, segments = _waveform_welch_settings(site_a, config)
    _waveform_welch_settings(site_b, config)
    _validate_pair_axes(site_a, site_b)
    auto_a = compute_site_psd(site_a, config) if psd_a is None else psd_a
    auto_b = compute_site_psd(site_b, config) if psd_b is None else psd_b
    _validate_shared_psd(auto_a, site_a, settings)
    _validate_shared_psd(auto_b, site_b, settings)
    coherence = np.full(auto_a.psd_uv2_per_hz.shape, np.nan)
    valid = np.zeros(coherence.shape, dtype=bool)
    trial_valid = auto_a.valid & auto_b.valid
    for trial in np.flatnonzero(trial_valid):
        frequency_hz, cross = signal.csd(
            site_a.values_uv[trial], site_b.values_uv[trial], fs=site_a.sample_rate_hz,
            window=settings["window"], nperseg=settings["nperseg"],
            noverlap=settings["noverlap"], nfft=settings["nfft"], detrend=settings["detrend"],
            return_onesided=True, scaling=settings["scaling"], average=settings["average"],
        )
        if not np.array_equal(frequency_hz, auto_a.frequency_hz):
            raise ValueError("Cross spectrum does not share the actual auto-PSD frequency grid")
        left = auto_a.psd_uv2_per_hz[trial]
        right = auto_b.psd_uv2_per_hz[trial]
        supported = np.isfinite(left) & (left > 0) & np.isfinite(right) & (right > 0) & np.isfinite(cross)
        ratio = np.full(left.shape, np.nan)
        # Sequential division matches SciPy and avoids multiplying tiny denominators.
        ratio[supported] = np.abs(cross[supported]) ** 2 / left[supported] / right[supported]
        tolerance = 64 * np.finfo(float).eps
        supported &= np.isfinite(ratio) & (ratio >= -tolerance) & (ratio <= 1 + tolerance)
        coherence[trial, supported] = np.clip(ratio[supported], 0.0, 1.0)
        valid[trial] = supported
    means, band_valid, _ = strict_band_mean(coherence, auto_a.frequency_hz, config.bands)
    metadata = {
        "analysis_version": ANALYSIS_VERSION, "sites": (site_a.site_id, site_b.site_id),
        "probes": (site_a.probe_id, site_b.probe_id),
        "saved_channels": (site_a.saved_channel_index, site_b.saved_channel_index),
        "interval": site_a.meta["interval"], "interval_bounds_s": site_a.meta["interval_bounds_s"],
        "sample_rate_hz": site_a.sample_rate_hz, "welch": settings,
        "auto_psd_ids": (auto_a.identity, auto_b.identity),
        "bands": [asdict(band) for band in config.bands], "units": "dimensionless",
        "validity": "complete matched waveforms and positive finite autos/defined cross ratio per frequency",
        "axes": ("trial", "frequency"), "roundoff_clip_tolerance": 64 * np.finfo(float).eps,
    }
    return WelchCoherence(
        auto_a.frequency_hz.copy(), coherence, valid, means, band_valid,
        tuple(band.name for band in config.bands),
        (site_a.valid & site_b.valid & np.isfinite(site_a.values_uv)
         & np.isfinite(site_b.values_uv)).sum(axis=1, dtype=np.int64),
        np.where(trial_valid, segments, 0).astype(np.int64),
        site_a.trial_ids.copy(), site_a.row_positions.copy(), site_a.choice_times_s.copy(), metadata,
    )


def strict_band_mean(
    values: np.ndarray,
    frequency_hz: np.ndarray,
    bands: tuple[FrequencyBandConfig, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce scalar dimensionless metrics over fixed retained frequency centers.

    ``values`` is float (...,frequency), ``frequency_hz`` is increasing float
    (frequency,) Hz, ``bands`` defines inclusive outer bounds and inclusive
    excluded centers. Return means and bool validity (...,band), plus bool
    (band,frequency) retained masks. Every retained center must be finite;
    empty bands are unavailable. Never average angles/vectors with this helper.
    Power uses frequency integration separately, not this frequency-center mean.
    """
    frequencies = np.asarray(frequency_hz, dtype=float)
    array = np.asarray(values, dtype=float)
    if (frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all()
            or np.any(np.diff(frequencies) <= 0) or array.ndim < 1 or array.shape[-1] != frequencies.size):
        raise ValueError("Values must share a finite increasing final frequency axis")
    masks = np.zeros((len(bands), frequencies.size), dtype=bool)
    means = np.full(array.shape[:-1] + (len(bands),), np.nan)
    valid = np.zeros(means.shape, dtype=bool)
    for index, band in enumerate(bands):
        mask = (frequencies >= band.lower_hz) & (frequencies <= band.upper_hz)
        for lower, upper in band.excluded_intervals_hz:
            mask &= ~((frequencies >= lower) & (frequencies <= upper))
        masks[index] = mask
        if np.any(mask):
            selected = array[..., mask]
            valid[..., index] = np.isfinite(selected).all(axis=-1)
            means[..., index] = np.where(valid[..., index], np.mean(selected, axis=-1), np.nan)
    return means, valid, masks


# Small validation/identity helpers; no estimators or numerical source loading.
def _waveform_welch_settings(waveforms: SiteWaveforms, config: PhaseSynchronyConfig) -> tuple[dict, int]:
    """Validate (trial,time) native uV/grid/config; return Welch settings and segment count."""
    values = np.asarray(waveforms.values_uv)
    time_s = np.asarray(waveforms.relative_time_s)
    rate = waveforms.sample_rate_hz
    if (values.ndim != 2 or waveforms.valid.shape != values.shape
            or time_s.ndim != 1 or values.shape[1] != time_s.size
            or not np.isfinite(rate) or rate <= 0):
        raise ValueError("Waveform values/masks must share (trial,time) axes and a positive native Hz rate")
    for array in (waveforms.trial_ids, waveforms.row_positions, waveforms.choice_times_s):
        if np.asarray(array).shape != (values.shape[0],):
            raise ValueError("Waveform identity axes must match trials")
    if not np.isfinite(waveforms.choice_times_s).all():
        raise ValueError("Choice UTC seconds must be finite")
    interval = waveforms.meta.get("interval")
    if interval not in INTERVAL_BOUNDS_S:
        raise ValueError("Waveforms need a supported centered interval")
    start, stop = INTERVAL_BOUNDS_S[interval]
    count_float = (stop - start) * rate
    count = round(count_float)
    if not np.isclose(count_float, count, rtol=0, atol=1e-9):
        raise ValueError("Interval duration is not representable at the native sample rate")
    if tuple(waveforms.meta.get("interval_bounds_s", ())) != (start, stop):
        raise ValueError("Waveform interval metadata does not match its centered interval")
    expected_time = start + np.arange(count) / rate
    if time_s.shape != expected_time.shape or not np.allclose(time_s, expected_time, rtol=0, atol=1e-12):
        raise ValueError("Waveforms must contain exactly the half-open native metric grid")
    if (waveforms.meta.get("voltage_unit") != "uV"
            or waveforms.meta.get("waveform_alignment_semantics") != WAVEFORM_ALIGNMENT_SEMANTICS):
        raise ValueError("Waveforms require calibrated uV and current IRIG alignment semantics")
    if bool(waveforms.meta.get("notch_enabled")) != config.notch_enabled:
        raise ValueError("Waveform continuous notch settings do not match requested config")
    if config.notch_enabled and waveforms.meta.get("notch_quality_factor") != config.notch_quality_factor:
        raise ValueError("Waveform continuous notch quality factor does not match config")
    nperseg = int(round(rate * config.welch_segment_s))
    noverlap = int(np.floor(nperseg * config.welch_overlap_fraction))
    if nperseg < 2 or noverlap >= nperseg:
        raise ValueError("Invalid Welch segment lengths")
    segments = 0 if count < nperseg else 1 + (count - nperseg) // (nperseg - noverlap)
    if segments < 2:
        raise ValueError("At least two complete configured Welch segments must fit each interval")
    return {
        "window": "hann_periodic", "detrend": "constant", "nperseg": nperseg,
        "noverlap": noverlap, "nfft": nperseg, "scaling": "density", "average": "mean",
    }, segments


def _validate_pair_axes(site_a: SiteWaveforms, site_b: SiteWaveforms) -> None:
    """Require equal native Hz, relative-second grid and original (trial,) axes; return None."""
    if site_a.sample_rate_hz != site_b.sample_rate_hz:
        raise ValueError("Unsupported native sample rate mismatch for coherence pair")
    for name in ("relative_time_s", "choice_times_s", "trial_ids", "row_positions"):
        if not np.array_equal(getattr(site_a, name), getattr(site_b, name)):
            raise ValueError(f"Coherence pair requires identical {name}")


def _psd_identity(waveforms: SiteWaveforms, settings: dict) -> str:
    """Digest exact uV/masks/time/ID arrays and relevant source/Welch metadata; return hex ID.

    Array shapes/dtypes and UTC seconds are retained, never rounded. Source
    metadata is JSON-compatible. Configured bands and other metric selections
    are absent, allowing power/coherence to reuse identical auto-PSDs.
    """
    metadata = {
        "analysis_version": ANALYSIS_VERSION, "site_id": waveforms.site_id,
        "probe_id": waveforms.probe_id, "saved_channel_index": waveforms.saved_channel_index,
        "sample_rate_hz": waveforms.sample_rate_hz, "source": waveforms.meta, "welch": settings,
    }
    digest = sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii"))
    for name in ("values_uv", "valid", "relative_time_s", "choice_times_s", "trial_ids", "row_positions"):
        array = np.asarray(getattr(waveforms, name))
        digest.update(json.dumps((name, array.shape, array.dtype.str)).encode("ascii"))
        if array.dtype.hasobject:
            digest.update(json.dumps(array.tolist(), separators=(",", ":")).encode("ascii"))
        else:
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _validate_shared_psd(psd: SitePSD, waveforms: SiteWaveforms, settings: dict) -> None:
    """Require supplied PSD identity/frequency/(trial,) axes to match exact waveform/settings."""
    expected_hz = np.fft.rfftfreq(settings["nperseg"], d=1.0 / waveforms.sample_rate_hz)
    expected_shape = (len(waveforms.trial_ids), len(expected_hz))
    if (psd.identity != _psd_identity(waveforms, settings)
            or psd.psd_uv2_per_hz.shape != expected_shape
            or not np.array_equal(psd.frequency_hz, expected_hz)
            or psd.meta.get("welch") != settings or psd.meta.get("psd_unit") != "uV^2/Hz"):
        raise ValueError("Supplied auto-PSD is incompatible with exact waveforms/settings/frequencies")
    supported = waveforms.valid & np.isfinite(waveforms.values_uv)
    count = supported.sum(axis=1, dtype=np.int64)
    valid = supported.all(axis=1)
    segments = 1 + (waveforms.values_uv.shape[1] - settings["nperseg"]) // (settings["nperseg"] - settings["noverlap"])
    if (not np.array_equal(psd.valid, valid) or not np.array_equal(psd.sample_count, count)
            or not np.array_equal(psd.segment_count, np.where(valid, segments, 0))):
        raise ValueError("Supplied auto-PSD support/segment counts do not match waveforms")
    for name in ("trial_ids", "row_positions", "choice_times_s"):
        if not np.array_equal(getattr(psd, name), getattr(waveforms, name)):
            raise ValueError(f"Supplied auto-PSD has incompatible {name}")
