"""Small data contracts for choice-centered phase synchrony and power.

Records are frozen containers; contained arrays/tables are not made read-only.
Computational functions preserve their inputs and return new output arrays.
"""

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd

from src.neural_analysis.lfp.config import FrequencyBandConfig


ANALYSIS_VERSION = "session-timeline-phase-synchrony-v1"
WAVEFORM_ALIGNMENT_SEMANTICS = "irig_uniform_utc_v1"
INTERVAL_BOUNDS_S = {"primary": (-1.0, 1.0), "secondary": (-2.0, 2.0)}
COMPONENTS = ("power", "coherence", "lfp-phase", "spike-ppc")


@dataclass(frozen=True)
class PhaseSynchronyConfig:
    """Round-1 settings: seconds, Hz, dimensionless thresholds, categorical IDs.

    ``sites`` and ordered ``pairs`` select metadata names; empty means defaults.
    ``unit_ids`` contains explicit ``probe:cluster`` identifiers; spikes are never
    pooled. ``absolute_amplitude_threshold`` is in coefficient-amplitude units
    (calibrated uV with the configured Morlet normalization), while numerical
    threshold is relative to the largest finite coefficient magnitude. Phase
    frequencies and bands are in Hz. No arrays or numerical results are stored.
    """

    intervals: tuple[str, ...] = ("primary",)
    components: tuple[str, ...] = ("power", "coherence", "lfp-phase")
    sites: tuple[str, ...] = ()
    pairs: tuple[tuple[str, str], ...] = ()
    unit_ids: tuple[str, ...] = ()
    lfp_site: str | None = None
    phase_rate_hz: float = 500.0
    phase_frequencies_hz: tuple[float, ...] = tuple(float(f) for f in range(2, 101, 2))
    notch_enabled: bool = True
    notch_quality_factor: float = 30.0
    numerical_amplitude_threshold: float = 0.0
    absolute_amplitude_threshold: float = 0.0
    welch_segment_s: float = 0.5
    welch_overlap_fraction: float = 0.5
    bands: tuple[FrequencyBandConfig, ...] = (
        FrequencyBandConfig("theta", 6.0, 10.0),
        FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),)),
    )

    def __post_init__(self) -> None:
        """Validate scalar units/selections in this record; return None or raise ValueError."""
        if not self.intervals or not set(self.intervals).issubset(INTERVAL_BOUNDS_S):
            raise ValueError("intervals must select primary and/or secondary")
        if not self.components or not set(self.components).issubset(COMPONENTS):
            raise ValueError(f"components must select from {COMPONENTS}")
        for name in ("phase_rate_hz", "notch_quality_factor", "welch_segment_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("numerical_amplitude_threshold", "absolute_amplitude_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        frequencies = np.asarray(self.phase_frequencies_hz, dtype=float)
        if (frequencies.ndim != 1 or not frequencies.size
                or not np.isfinite(frequencies).all() or np.any(frequencies <= 0)
                or np.any(np.diff(frequencies) <= 0)):
            raise ValueError("phase_frequencies_hz must be finite, positive, and increasing")
        if not math.isfinite(self.welch_overlap_fraction) or not 0 <= self.welch_overlap_fraction < 1:
            raise ValueError("welch_overlap_fraction must be in [0, 1)")
        for unit_id in self.unit_ids:
            parse_unit_id(unit_id)
        if "spike-ppc" in self.components and (not self.unit_ids or not self.lfp_site):
            raise ValueError("spike-ppc requires explicit unit_ids and lfp_site")


@dataclass(frozen=True)
class ContinuousSiteLFP:
    """One calibrated bounded native source and independent synchronization.

    ``sample_indices`` is integer (sample,), the original zero-based native LFP
    indices, with gaps permitted. ``values_uv`` is float (sample,) in uV;
    ``valid`` is bool (sample,). ``irig_df`` contains monotonic ``sample_ix``
    native coordinates and ``utc_unix`` UTC Unix seconds. ``sample_rate_hz`` is
    nominal samples/second. Metadata identifies sources, calibration, and actual
    preprocessing. No shape/clock/voltage conversion happens in this container.
    """

    site_id: str
    probe_id: str
    saved_channel_index: int
    sample_indices: np.ndarray
    values_uv: np.ndarray
    valid: np.ndarray
    sample_rate_hz: float
    irig_df: pd.DataFrame
    meta: dict


@dataclass(frozen=True)
class SiteWaveforms:
    """Choice-aligned native-rate voltages and support, without extra load samples.

    ``values_uv`` float uV and ``valid`` bool have (trial,time) axes.
    ``relative_time_s`` is float (time,), half-open choice-relative seconds.
    ``choice_times_s`` is float (trial,) UTC seconds; ``trial_ids`` preserves
    original categorical IDs, ``row_positions`` int (trial,) source positions.
    ``sample_rate_hz`` is the actual uniform target rate in Hz. Metadata records
    site/probe/channel, interval, clock alignment and preprocessing identities.
    """

    site_id: str
    probe_id: str
    saved_channel_index: int
    values_uv: np.ndarray
    valid: np.ndarray
    relative_time_s: np.ndarray
    choice_times_s: np.ndarray
    trial_ids: np.ndarray
    row_positions: np.ndarray
    sample_rate_hz: float
    meta: dict


@dataclass(frozen=True)
class SitePSD:
    """Reusable within-trial Welch density with physical uV^2/Hz units.

    ``psd_uv2_per_hz`` is float (trial,frequency); ``frequency_hz`` is (frequency,).
    ``valid`` is bool (trial,), requiring complete waveform support. Integer
    ``sample_count`` counts supported samples and ``segment_count`` counts
    contributing complete segments, each (trial,). Original categorical
    ``trial_ids``, integer ``row_positions`` and UTC-second ``choice_times_s``
    retain (trial,) axes. ``identity`` includes exact waveforms/axes/preprocessing
    and Welch settings, excluding bands. Metadata records site and unit details.
    """

    frequency_hz: np.ndarray
    psd_uv2_per_hz: np.ndarray
    valid: np.ndarray
    sample_count: np.ndarray
    segment_count: np.ndarray
    trial_ids: np.ndarray
    row_positions: np.ndarray
    choice_times_s: np.ndarray
    identity: str
    meta: dict


@dataclass(frozen=True)
class BandPower:
    """Per-site integrated power: float (trial,band) uV^2 and bool validity.

    ``bandwidth_hz`` is float (band,) retained Hz; ``band_names`` identifies
    that axis. Sample/segment counts are integer (trial,). Original trial IDs,
    row positions and UTC-second choices retain (trial,) axes. Metadata records
    the source auto-PSD identity, band definitions and integration convention.
    """

    band_power_integrated: np.ndarray
    valid: np.ndarray
    bandwidth_hz: np.ndarray
    band_names: tuple[str, ...]
    sample_count: np.ndarray
    segment_count: np.ndarray
    trial_ids: np.ndarray
    row_positions: np.ndarray
    choice_times_s: np.ndarray
    meta: dict


@dataclass(frozen=True)
class WelchCoherence:
    """Dimensionless magnitude-squared coherence on (trial,frequency) axes.

    ``frequency_hz`` is float (frequency,); ``coherence`` is float and ``valid``
    bool (trial,frequency), allowing zero-denominator bins to be unavailable
    independently. ``band_mean``/``band_valid`` have (trial,band) axes;
    ``band_names`` names that axis. ``sample_count``/``segment_count`` and
    original categorical IDs, row positions and UTC choices are (trial,).
    Metadata records ordered sites, actual settings and reused auto-PSD IDs.
    """

    frequency_hz: np.ndarray
    coherence: np.ndarray
    valid: np.ndarray
    band_mean: np.ndarray
    band_valid: np.ndarray
    band_names: tuple[str, ...]
    sample_count: np.ndarray
    segment_count: np.ndarray
    trial_ids: np.ndarray
    row_positions: np.ndarray
    choice_times_s: np.ndarray
    meta: dict


@dataclass(frozen=True)
class SelectedUnitSpikes:
    """One explicit unit's float (spike,) synchronized UTC-second ``times_s``.

    ``unit_id`` is probe-qualified; ``cluster_id`` is its integer sorter label.
    Metadata records selection/source identities. No clock correction or
    binning is applied by this container. Empty arrays represent true silence.
    """

    unit_id: str
    probe_id: str
    cluster_id: int
    times_s: np.ndarray
    meta: dict


def parse_unit_id(unit_id: str) -> tuple[str, int]:
    """Parse categorical ``probe:nonnegative_cluster`` text; return (str,int), no units."""
    if not isinstance(unit_id, str) or unit_id.count(":") != 1:
        raise ValueError("unit ID must be probe-qualified as PROBE:CLUSTER")
    probe_id, cluster = unit_id.split(":")
    if not probe_id or not cluster.isdecimal():
        raise ValueError("unit ID must be probe-qualified as PROBE:CLUSTER")
    return probe_id, int(cluster)
