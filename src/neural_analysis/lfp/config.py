"""Reusable configuration records for LFP windows, bands, and power analysis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisWindowConfig:
    """Half-open whole/before/after event-relative windows in seconds."""

    alignment_event: str = "choice_time"
    whole_start_s: float = -2.0
    whole_stop_s: float = 2.0
    before_start_s: float = -2.0
    before_stop_s: float = 0.0
    after_start_s: float = 0.0
    after_stop_s: float = 2.0


@dataclass(frozen=True)
class FrequencyBandConfig:
    """Named inclusive outer bounds and open excluded intervals, all in Hz."""

    name: str
    lower_hz: float
    upper_hz: float
    excluded_intervals_hz: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class PowerAnalysisConfig:
    """Welch PSD, line-noise, canonical-grid, and pre-session reference settings."""

    welch_window: str = "hann_periodic"
    welch_detrend: str = "constant"
    welch_window_s: float = 0.5
    welch_overlap_fraction: float = 0.5
    canonical_frequency_step_hz: float = 2.0
    notch_enabled: bool = True
    notch_hz: float = 60.0
    notch_quality_factor: float = 30.0
    presession_reference_duration_s: float = 10.0
    bands: tuple[FrequencyBandConfig, ...] = (
        FrequencyBandConfig("theta", 6.0, 10.0),
        FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),)),
    )
