"""Small data contracts for choice-centered phase synchrony and power.

Records are frozen containers; contained arrays/tables are not made read-only.
Computational functions preserve their inputs and return new output arrays.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import pickle
import socket
import tempfile
from typing import Iterator
import zipfile

import numpy as np
import pandas as pd

from src.neural_analysis.lfp.config import FrequencyBandConfig


ANALYSIS_VERSION = "session-timeline-phase-synchrony-v1"
WAVEFORM_ALIGNMENT_SEMANTICS = "irig_uniform_utc_v1"
PHASE_SUPPORT_SEMANTICS = "strict_support_v1"
PHASE_ALIGNMENT_SEMANTICS = "irig_uniform_utc_pre_morlet_v1"
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
class SitePhase:
    """One site's unit phase, complex64 (frequency,trial,time), with bool validity.

    ``frequency_hz`` is float (frequency,); ``relative_time_s`` is float (time,)
    choice-relative seconds including a real right guard. Boolean ``metric_mask``
    selects exactly the half-open metric samples, excluding guards. Actual rate
    is ``sample_rate_hz``. Original categorical IDs, integer rows and UTC-second
    choices retain (trial,) axes. Metadata records real transform/source support,
    synchronization, amplitude gates and operation order. Invalid vectors are
    complex NaN; phase carries no physical units or amplitude information.
    """

    site_id: str
    probe_id: str
    saved_channel_index: int
    frequency_hz: np.ndarray
    phase: np.ndarray
    valid: np.ndarray
    relative_time_s: np.ndarray
    metric_mask: np.ndarray
    sample_rate_hz: float
    trial_ids: np.ndarray
    row_positions: np.ndarray
    choice_times_s: np.ndarray
    meta: dict


@dataclass(frozen=True)
class LFPPhaseSummary:
    """Within-window dimensionless PLV/PPC on (trial,frequency) axes.

    ``phase_offset_rad`` is A-minus-B circular mean in radians or NaN for a
    vanishing resultant. ``sample_count`` is integer actual paired sample count;
    ``valid`` requires complete metric support and M>1. ``band_plv``,
    ``band_ppc`` and ``band_valid`` have (trial,band) axes, with fixed retained
    frequency centers. Frequencies are Hz; original IDs/rows/UTC-second choices
    retain (trial,) axes. No guards or observations from another trial enter M.
    """

    frequency_hz: np.ndarray
    plv: np.ndarray
    ppc: np.ndarray
    phase_offset_rad: np.ndarray
    sample_count: np.ndarray
    valid: np.ndarray
    band_plv: np.ndarray
    band_ppc: np.ndarray
    band_valid: np.ndarray
    band_names: tuple[str, ...]
    trial_ids: np.ndarray
    row_positions: np.ndarray
    choice_times_s: np.ndarray
    meta: dict


@dataclass(frozen=True)
class SelectedUnitPPC:
    """One probe-qualified unit/site's within-trial dimensionless spike PPC.

    Frequency results, bool ``valid`` and integer spike/exclusion counts have
    (trial,frequency) axes. Band PPC/validity and min/max retained-frequency
    spike counts have (trial,band) axes. ``polar_histogram_count`` is integer
    (frequency,trial,phase_bin); fixed float bin edges are radians. Selected
    UTC-second ``spike_times_by_trial`` is a tuple of (spike,) arrays, preserving
    input precision. Original trial IDs/rows/UTC choices retain (trial,) axes.
    Metadata identifies source phase settings/support; units are never pooled.
    """

    unit_id: str
    site_id: str
    frequency_hz: np.ndarray
    ppc: np.ndarray
    spike_count: np.ndarray
    excluded_spike_count: np.ndarray
    valid: np.ndarray
    band_ppc: np.ndarray
    band_valid: np.ndarray
    band_spike_count_min: np.ndarray
    band_spike_count_max: np.ndarray
    band_names: tuple[str, ...]
    phase_bin_edges_rad: np.ndarray
    polar_histogram_count: np.ndarray
    spike_times_by_trial: tuple[np.ndarray, ...]
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


def validate_phase_contract(site_phase: SitePhase) -> None:
    """Validate (frequency,trial,time) phase and exact 500-Hz centered metric/guard grid.

    Frequency coordinates are positive increasing Hz; trial axes retain IDs,
    integer source rows and finite UTC seconds. Return None or raise ValueError;
    no axes, masks, units or interval boundaries are inferred or changed.
    """
    frequencies = np.asarray(site_phase.frequency_hz)
    if (frequencies.ndim != 1 or not frequencies.size or not np.isfinite(frequencies).all()
            or np.any(frequencies <= 0) or np.any(np.diff(frequencies) <= 0)):
        raise ValueError("Phase frequency grid must be finite, positive and increasing")
    if site_phase.sample_rate_hz != 500.0:
        raise ValueError("Phase metric grid requires the actual 500-Hz rate")
    interval = site_phase.meta.get("interval")
    if interval not in INTERVAL_BOUNDS_S:
        raise ValueError("Phase metric interval must be primary or secondary")
    start, stop = INTERVAL_BOUNDS_S[interval]
    if tuple(site_phase.meta.get("interval_bounds_s", ())) != (start, stop):
        raise ValueError("Phase interval bounds disagree with its centered metric interval")
    metric_count = round((stop - start) * 500)
    expected_time = start + np.arange(metric_count + 1) / 500
    expected_mask = np.arange(metric_count + 1) < metric_count
    if (not np.array_equal(site_phase.relative_time_s, expected_time)
            or not np.array_equal(site_phase.metric_mask, expected_mask)
            or site_phase.metric_mask.dtype != np.dtype(bool)):
        raise ValueError("Phase grid/metric mask must retain exact half-open samples and one excluded real guard")
    expected = (len(site_phase.frequency_hz), len(site_phase.trial_ids), len(site_phase.relative_time_s))
    if (site_phase.phase.shape != expected or site_phase.valid.shape != expected
            or site_phase.valid.dtype != np.dtype(bool)):
        raise ValueError("Site phase must preserve (frequency,trial,time) axes and metric mask")
    for array in (site_phase.trial_ids, site_phase.row_positions, site_phase.choice_times_s):
        if np.asarray(array).shape != (expected[1],):
            raise ValueError("Phase trial ID/row/UTC axes must match the trial dimension")
    if (not np.isfinite(site_phase.choice_times_s).all()
            or not np.issubdtype(site_phase.row_positions.dtype, np.integer)):
        raise ValueError("Phase trial axes require finite UTC seconds and integer source rows")
    if (site_phase.meta.get("phase_support_semantics") != PHASE_SUPPORT_SEMANTICS
            or site_phase.meta.get("phase_alignment_semantics") != PHASE_ALIGNMENT_SEMANTICS):
        raise ValueError("Site phase has incompatible scientific support/alignment semantics")


def parse_unit_id(unit_id: str) -> tuple[str, int]:
    """Parse categorical ``probe:nonnegative_cluster`` text; return (str,int), no units."""
    if not isinstance(unit_id, str) or unit_id.count(":") != 1:
        raise ValueError("unit ID must be probe-qualified as PROBE:CLUSTER")
    probe_id, cluster = unit_id.split(":")
    if not probe_id or not cluster.isdecimal():
        raise ValueError("unit ID must be probe-qualified as PROBE:CLUSTER")
    return probe_id, int(cluster)


# Persistence uses an explicit family/type table, not arbitrary class loading.
_PRODUCT_TYPES = {
    "native-lfp": ContinuousSiteLFP, "unit-spikes": SelectedUnitSpikes,
    "waveforms": SiteWaveforms, "psd": SitePSD, "power": BandPower,
    "coherence": WelchCoherence, "phase": SitePhase,
    "lfp-phase": LFPPhaseSummary, "spike-ppc": SelectedUnitPPC,
}
_PHASE_FAMILIES = ("phase", "lfp-phase", "spike-ppc")
_PRODUCT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProductInfo:
    """Metadata-only artifact description: absolute filesystem ``path`` and dictionary ``meta``.

    No scientific arrays are loaded. Meta retains typed-record provenance,
    source/configuration identities and run/completion details, with units/axes
    belonging to the contained record contract.
    """

    path: Path
    meta: dict


@dataclass(frozen=True)
class SavedProduct:
    """One loaded scientific record plus independent artifact metadata and absolute path.

    ``record`` is one of the nine explicit data records above, with its original
    axes/units and unchanged record metadata. ``meta`` holds publication/run
    information, which is never stamped into the scientific record itself.
    """

    record: object
    meta: dict
    path: Path


def preparation_identity(record: ContinuousSiteLFP | SelectedUnitSpikes | SiteWaveforms | SitePhase) -> str:
    """Hash an actual prepared record's arrays/clock/support/settings; return a hex string.

    Inputs retain their documented native/UTC/unit-phase axes and units. The
    digest includes exact array dtypes/shapes/bytes, source/preprocessing metadata,
    valid masks, guards and precise phase-block relative origins. No display or
    publication state is added. Native clocks use only required native-index/UTC
    anchor columns; optional datetime display columns are not scientific inputs.
    """
    if type(record) not in (ContinuousSiteLFP, SelectedUnitSpikes, SiteWaveforms, SitePhase):
        raise ValueError("Preparation identity requires native, unit, waveform or phase preparation")
    arrays, scalar_values = _encode_record(record)
    digest = sha256(_json_bytes({"type": type(record).__name__, "values": scalar_values, "meta": record.meta}))
    for name, array in sorted(arrays.items()):
        digest.update(_json_bytes((name, array.shape, array.dtype.str)))
        digest.update(_json_bytes(array.tolist()) if array.dtype.hasobject else np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def product_identity(
    family: str, selection: tuple[str, ...], interval: str | None,
    config: PhaseSynchronyConfig, *, source_ids: dict, preparation_ids: tuple[str, ...] = (),
) -> dict:
    """Return one component's JSON-compatible scientific/source identity dictionary.

    Family/selection are categorical explicit site/pair/unit IDs. Native/unit
    preparations use interval None; windowed families use primary/secondary
    seconds. Source IDs identify relevant inputs only; preparation IDs reference
    exact actual support/arrays. Parameters include only this family's settings:
    native ignores phase/Welch, PSD ignores bands, phase ignores bands, summaries
    include their bands. Global requested families/units/sites/interval lists and
    run/display settings do not invalidate unrelated products. Inputs are copied.
    """
    if family not in _PRODUCT_TYPES or not selection or any(not isinstance(value, str) or not value for value in selection):
        raise ValueError("Product family and explicit nonempty selection are required")
    continuous = family in {"native-lfp", "unit-spikes"}
    if (continuous and interval is not None) or (not continuous and interval not in INTERVAL_BOUNDS_S):
        raise ValueError("Only continuous native/unit preparations use interval None")
    parameters = {}
    if family != "unit-spikes":
        parameters.update({"voltage_unit": "uV", "notch_enabled": config.notch_enabled,
                           "notch_quality_factor": config.notch_quality_factor if config.notch_enabled else None})
    else:
        parameters["time_unit"] = "UTC Unix seconds"
    if family in {"waveforms", "psd", "power", "coherence"}:
        parameters["waveform_alignment_semantics"] = WAVEFORM_ALIGNMENT_SEMANTICS
    if family in {"psd", "power", "coherence"}:
        parameters["welch"] = {"segment_s": config.welch_segment_s,
                               "overlap_fraction": config.welch_overlap_fraction,
                               "window": "hann_periodic", "detrend": "constant",
                               "fft_length": "segment_length", "scaling": "density", "average": "mean"}
    if family in _PHASE_FAMILIES:
        parameters.update({
            "phase_support_semantics": PHASE_SUPPORT_SEMANTICS,
            "phase_alignment_semantics": PHASE_ALIGNMENT_SEMANTICS,
            "numerical_amplitude_threshold": config.numerical_amplitude_threshold,
            "absolute_amplitude_threshold": config.absolute_amplitude_threshold,
            "phase_rate_hz": config.phase_rate_hz, "phase_frequencies_hz": config.phase_frequencies_hz,
            "morlet": {"gaussian_width": 1.5, "window_length": 1.0, "precision": 16, "norm": "l1"},
            "guard_samples": 1,
        })
    if family in {"power", "coherence", "lfp-phase", "spike-ppc"}:
        parameters["bands"] = [asdict(band) for band in config.bands]
    if family == "spike-ppc":
        parameters["phase_bin_count"] = 24
        parameters["estimator"] = "within_trial_all_pairs"
    identity = {"identity_version": 1, "analysis_version": ANALYSIS_VERSION,
                "family": family, "selection": tuple(selection), "interval": interval,
                "interval_bounds_s": None if interval is None else INTERVAL_BOUNDS_S[interval],
                "parameters": parameters, "source_ids": deepcopy(source_ids),
                "preparation_ids": tuple(preparation_ids)}
    identity["identity_id"] = sha256(_json_bytes(identity)).hexdigest()
    return identity


def save_product(
    path: Path | str, record: object, identity: dict, *, run_id: str, completed_utc: str,
) -> Path:
    """Atomically publish a new immutable named-array NPZ; return its absolute path.

    Record is an explicit scientific data type with its original axes/units.
    Identity comes from product_identity; run_id is categorical and completed_utc
    is a timezone-aware ISO instant. Artifact meta and arrays are separate from
    unchanged record.meta. Required native IRIG columns become named arrays;
    ragged selected UTC spikes become flat seconds plus integer offsets.
    A complete sibling is linked into place exclusively, then the temporary
    name is removed. Existing outputs never overwrite, even during publication.
    """
    target = Path(path).resolve()
    if target.suffix != ".npz":
        raise ValueError("Product path must have .npz suffix")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    _validate_identity(identity)
    _completion_time(completed_utc)
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be nonempty text")
    if type(record) is not _PRODUCT_TYPES[identity["family"]]:
        raise ValueError("Record type does not match product family")
    _validate_record_identity(record, identity)
    arrays, scalar_values = _encode_record(record)
    metadata = {"schema_version": _PRODUCT_SCHEMA_VERSION, "analysis_version": ANALYSIS_VERSION,
                "generator": "src.neural_analysis.phase_synchrony.results.save_product",
                "run_id": run_id, "completed_utc": completed_utc, "identity": deepcopy(identity),
                "record_type": type(record).__name__, "record_meta": deepcopy(record.meta),
                "record_values": scalar_values,
                "array_descriptors": {name: {"shape": array.shape, "dtype": array.dtype.str} for name, array in arrays.items()}}
    _validate_artifact_metadata(metadata)
    arrays["meta"] = np.asarray(metadata, dtype=object)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".", suffix=".tmp.npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def read_product_metadata(path: Path | str) -> dict:
    """Read/validate only NPZ dictionary meta and member names; return an independent dict.

    No native, phase, PSD, spike or other scientific arrays are decompressed.
    Metadata retains scientific units/axes, identities and UTC/run provenance.
    Temporary files, unsupported schemas and invalid phase semantics raise
    ValueError. Full numerical loading is an explicit separate operation.
    """
    target = Path(path)
    if target.name.startswith(".") or target.name.endswith(".tmp.npz"):
        raise ValueError("Temporary products are not completed artifacts")
    try:
        with np.load(target, allow_pickle=True) as archive:
            metadata = archive["meta"].item()
            _validate_artifact_metadata(metadata)
            if not set(metadata["array_descriptors"]).issubset(archive.files):
                raise ValueError("Product lacks required named arrays")
    except (KeyError, AttributeError, TypeError, pickle.UnpicklingError) as error:
        raise ValueError("Invalid product metadata") from error
    return deepcopy(metadata)


def load_product(path: Path | str, *, expected_identity: dict | None = None) -> SavedProduct:
    """Load one typed completed product, preserving all named scientific arrays/units.

    Optional expected_identity requires exact relevant configuration/source/
    preparation compatibility. Return SavedProduct with independent artifact
    meta and original record.meta; native clock display-only columns are not
    reconstructed. Array descriptors and ragged offsets are validated before
    construction. No raw numerical sources or scientific estimators are called.
    """
    target = Path(path).resolve()
    metadata = read_product_metadata(target)
    if expected_identity is not None and metadata["identity"] != expected_identity:
        raise ValueError("Product does not match expected scientific identity")
    values = deepcopy(metadata["record_values"])
    with np.load(target, allow_pickle=True) as archive:
        for name, descriptor in metadata["array_descriptors"].items():
            array = archive[name]
            if array.shape != tuple(descriptor["shape"]) or array.dtype.str != descriptor["dtype"]:
                raise ValueError(f"Product array descriptor mismatch: {name}")
            values[name] = array
    family = metadata["identity"]["family"]
    if family == "native-lfp":
        values["irig_df"] = pd.DataFrame({"sample_ix": values.pop("irig_sample_ix"), "utc_unix": values.pop("irig_utc_unix")})
    if family == "spike-ppc":
        flat = values.pop("spike_times_flat_s")
        offsets = values.pop("spike_times_offsets")
        if (offsets.ndim != 1 or offsets.dtype.kind not in "iu" or offsets.size < 1
                or offsets[0] != 0 or offsets[-1] != flat.size or np.any(np.diff(offsets) < 0)):
            raise ValueError("Invalid ragged spike offsets")
        values["spike_times_by_trial"] = tuple(flat[first:last].copy() for first, last in zip(offsets[:-1], offsets[1:]))
    values["meta"] = deepcopy(metadata["record_meta"])
    try:
        record = _PRODUCT_TYPES[family](**values)
    except TypeError as error:
        raise ValueError("Invalid typed record fields") from error
    _validate_record_identity(record, metadata["identity"])
    return SavedProduct(record, metadata, target)


def discover_product_metadata(session_root: Path | str, family: str | None = None) -> tuple[ProductInfo, ...]:
    """Discover ordered completed headers without needing a numerical preparation ID.

    Session root is a filesystem path; optional family is one of the nine
    categorical product families. Return an immutable tuple of ProductInfo,
    newest completion/run/path first. Only feature run components and processed
    preparations are searched, only meta is read, and nothing is written.
    Callers can compare known source/configuration fields, then obtain opaque
    actual preparation references from matching headers before numerical work.
    """
    if family is not None and family not in _PRODUCT_TYPES:
        raise ValueError("Unknown product family")
    root = Path(session_root)
    paths = list((root / "analysis_runs").glob("session_timeline_phase_synchrony_*/components/**/*.npz"))
    paths.extend((root / "processed" / "session_timeline_phase_synchrony").glob("**/*.npz"))
    candidates = []
    for path in paths:
        try:
            metadata = read_product_metadata(path)
            if family is None or metadata["identity"]["family"] == family:
                candidates.append(ProductInfo(path.resolve(), metadata))
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            continue
    candidates.sort(key=_product_order_key, reverse=True)
    return tuple(candidates)


def find_product_metadata(session_root: Path | str, expected_identity: dict) -> ProductInfo | None:
    """Find newest compatible metadata only under this feature's two session roots.

    Return ProductInfo or None, without numerical array reads or writes. Order
    is parsed UTC completion instant, run ID, then session-relative path string.
    Temporary/corrupt/incompatible products and newer failed/unfinished attempts
    never displace an older compatible completed product. Filesystem mtime is
    irrelevant; run measurements remain separate attempt-status evidence.
    """
    candidates = _product_candidates(Path(session_root), expected_identity)
    return candidates[0] if candidates else None


def find_product(session_root: Path | str, expected_identity: dict) -> SavedProduct | None:
    """Find/load the newest compatible typed product; fall back on corrupt numerical data.

    Metadata discovery is restricted to this feature. Only candidates in
    deterministic completion order are loaded; stop at the first readable
    completed record. Return SavedProduct or None. No scientific recomputation
    or source reopening occurs, and nothing is written.
    """
    for candidate in _product_candidates(Path(session_root), expected_identity):
        try:
            return load_product(candidate.path, expected_identity=expected_identity)
        except (OSError, ValueError, EOFError, zipfile.BadZipFile):
            continue
    return None


@contextmanager
def session_writer_lock(session_root: Path | str) -> Iterator[Path]:
    """Own one session-local feature writer lock until context exit; yield its path.

    Session root is a filesystem directory, not an analysis configuration. A
    live same-host PID, foreign host or malformed owner blocks acquisition.
    Recover only a validated same-session, same-host owner proven dead using
    signal-zero PID lookup. Permission uncertainty counts as live. Readers need
    no lock. On exit remove only this context's unchanged ownership record.
    """
    root = Path(session_root).resolve()
    path = root / "processed" / "session_timeline_phase_synchrony" / "writer.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"session_root": str(root), "hostname": socket.gethostname(), "pid": os.getpid(),
             "created_utc": datetime.now(timezone.utc).isoformat()}
    payload = json.dumps(owner, sort_keys=True).encode("ascii")
    try:
        handle = path.open("xb")
    except FileExistsError:
        _recover_writer_lock(path, root)
        try:
            handle = path.open("xb")
        except FileExistsError as error:
            raise RuntimeError("Session writer lock changed during recovery") from error
    with handle:
        handle.write(payload)
        handle.flush()
    try:
        yield path
    finally:
        if path.is_file() and path.read_bytes() == payload:
            path.unlink()


# Explicit persistence helpers; scientific records never gain workflow behavior.
def _encode_record(record: object) -> tuple[dict[str, np.ndarray], dict]:
    """Separate typed record into named arrays and scalar/tuple fields, preserving axes/units."""
    arrays = {}
    values = {}
    for field in fields(record):
        value = getattr(record, field.name)
        if field.name == "meta":
            continue
        if isinstance(value, np.ndarray):
            arrays[field.name] = value
        elif field.name == "irig_df":
            arrays["irig_sample_ix"] = value.sample_ix.to_numpy(copy=True)
            arrays["irig_utc_unix"] = value.utc_unix.to_numpy(copy=True)
        elif field.name == "spike_times_by_trial":
            lengths = np.asarray([len(times) for times in value], dtype=np.int64)
            arrays["spike_times_offsets"] = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
            arrays["spike_times_flat_s"] = np.concatenate(value) if len(value) else np.empty(0, dtype=float)
        else:
            values[field.name] = deepcopy(value)
    return arrays, values


def _json_bytes(value: object) -> bytes:
    """Encode JSON-compatible scientific metadata deterministically; no array/unit conversions."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _completion_time(value: str) -> datetime:
    """Parse timezone-aware ISO completion text; return the corresponding UTC datetime."""
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise ValueError("Completion time must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Completion time must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _validate_identity(identity: dict) -> None:
    """Validate supported scientific identity structure/digest; return None, preserving parameters."""
    if (not isinstance(identity, dict) or identity.get("identity_version") != 1
            or identity.get("analysis_version") != ANALYSIS_VERSION or identity.get("family") not in _PRODUCT_TYPES):
        raise ValueError("Unsupported scientific product identity")
    payload = {key: value for key, value in identity.items() if key != "identity_id"}
    if identity.get("identity_id") != sha256(_json_bytes(payload)).hexdigest():
        raise ValueError("Scientific product identity digest mismatch")


def _validate_artifact_metadata(metadata: dict) -> None:
    """Validate dictionary publication/type metadata and phase scientific provenance, without arrays."""
    if (not isinstance(metadata, dict) or metadata.get("schema_version") != _PRODUCT_SCHEMA_VERSION
            or metadata.get("analysis_version") != ANALYSIS_VERSION):
        raise ValueError("Unsupported product schema or analysis version")
    _validate_identity(metadata["identity"])
    _completion_time(metadata["completed_utc"])
    family = metadata["identity"]["family"]
    if (metadata.get("record_type") != _PRODUCT_TYPES[family].__name__
            or not isinstance(metadata.get("run_id"), str) or not metadata["run_id"]
            or not isinstance(metadata.get("record_meta"), dict)
            or not isinstance(metadata.get("record_values"), dict)
            or not isinstance(metadata.get("array_descriptors"), dict)):
        raise ValueError("Invalid product typed-record metadata")
    if family in _PHASE_FAMILIES:
        phase_metadata = metadata["record_meta"]
        if family == "lfp-phase":
            phase_metadata = phase_metadata["source_phase"]
        elif family == "spike-ppc":
            phase_metadata = (phase_metadata["source_phase"],)
        else:
            phase_metadata = (phase_metadata,)
        for item in phase_metadata:
            parameters = metadata["identity"]["parameters"]
            if (item.get("phase_support_semantics") != PHASE_SUPPORT_SEMANTICS
                    or item.get("phase_alignment_semantics") != PHASE_ALIGNMENT_SEMANTICS):
                raise ValueError("Phase product lacks current scientific support/alignment semantics")
            for name in ("numerical_amplitude_threshold", "absolute_amplitude_threshold"):
                if name not in item or item[name] != parameters[name]:
                    raise ValueError(f"Phase product has incompatible {name}")


def _product_candidates(root: Path, expected_identity: dict) -> list[ProductInfo]:
    """Return compatible metadata ordered by UTC/run/path under feature run/preparation roots only."""
    _validate_identity(expected_identity)
    return [product for product in discover_product_metadata(root, expected_identity["family"])
            if product.meta["identity"] == expected_identity]


def _validate_record_identity(record: object, identity: dict) -> None:
    """Bind an actual record's IDs, seconds, Hz and effective settings to its declared identity.

    Called only during save/full load, where scientific arrays are available.
    Metadata-only discovery does not inspect numerical arrays. Parent preparation
    IDs are opaque references; this does not invent a self-hash relationship.
    """
    family = identity["family"]
    parameters = identity["parameters"]
    meta = record.meta
    if family == "unit-spikes":
        selection = (record.unit_id,)
        if parse_unit_id(record.unit_id) != (record.probe_id, record.cluster_id):
            raise ValueError("Unit ID does not match actual probe/cluster")
    elif family == "spike-ppc":
        selection = (record.unit_id, record.site_id)
    elif family in {"coherence", "lfp-phase"}:
        selection = tuple(meta["sites"])
    else:
        selection = (record.site_id if hasattr(record, "site_id") else meta["site_id"],)
    if tuple(identity["selection"]) != selection:
        raise ValueError("Product selection does not match actual record site/unit IDs")
    if family not in {"native-lfp", "unit-spikes"}:
        if (meta.get("interval") != identity["interval"]
                or tuple(meta.get("interval_bounds_s", ())) != tuple(identity["interval_bounds_s"])):
            raise ValueError("Product interval/bounds do not match actual record")
    if "notch_enabled" in meta:
        if bool(meta["notch_enabled"]) != parameters["notch_enabled"]:
            raise ValueError("Product notch setting does not match actual preprocessing")
        if parameters["notch_enabled"] and meta.get("notch_quality_factor") != parameters["notch_quality_factor"]:
            raise ValueError("Product notch quality does not match actual preprocessing")
    if family in {"psd", "power", "coherence"}:
        requested = parameters["welch"]
        nperseg = round(meta["sample_rate_hz"] * requested["segment_s"])
        expected = {"window": requested["window"], "detrend": requested["detrend"],
                    "nperseg": nperseg, "noverlap": math.floor(nperseg * requested["overlap_fraction"]),
                    "nfft": nperseg, "scaling": requested["scaling"], "average": requested["average"]}
        if meta.get("welch") != expected:
            raise ValueError("Product Welch settings do not match actual spectra")
    if family in _PHASE_FAMILIES:
        if not np.array_equal(record.frequency_hz, parameters["phase_frequencies_hz"]):
            raise ValueError("Product phase frequencies do not match actual Hz coordinates")
        if family == "phase":
            validate_phase_contract(record)
            if record.sample_rate_hz != parameters["phase_rate_hz"] or meta.get("morlet") != parameters["morlet"]:
                raise ValueError("Product phase rate/Morlet settings do not match actual preparation")
        source_metadata = (meta,) if family == "phase" else meta["source_phase"]
        if family == "spike-ppc":
            source_metadata = (source_metadata,)
        for source in source_metadata:
            if (source.get("phase_rate_hz") != parameters["phase_rate_hz"]
                    or source.get("morlet") != parameters["morlet"]):
                raise ValueError("Product phase settings do not match its actual source preparation")
            if bool(source.get("notch_enabled")) != parameters["notch_enabled"]:
                raise ValueError("Product phase notch does not match actual source preparation")
            if parameters["notch_enabled"] and source.get("notch_quality_factor") != parameters["notch_quality_factor"]:
                raise ValueError("Product phase notch quality does not match actual source preparation")
    if "bands" in parameters and _json_bytes(meta.get("bands")) != _json_bytes(parameters["bands"]):
        raise ValueError("Product bands do not match actual band reduction")


def _product_order_key(product: ProductInfo) -> tuple[datetime, str, str]:
    """Return UTC/run/path tie keys; shared absolute root leaves relative path ordering unchanged."""
    return _completion_time(product.meta["completed_utc"]), product.meta["run_id"], str(product.path)


def _recover_writer_lock(path: Path, root: Path) -> None:
    """Remove only unchanged same-session/same-host positive PID ownership proven dead."""
    if path.is_symlink():
        raise RuntimeError("Session writer lock is not safely recoverable")
    try:
        payload = path.read_bytes()
        owner = json.loads(payload)
        if owner["session_root"] != str(root) or owner["hostname"] != socket.gethostname():
            raise RuntimeError("Session writer lock belongs to another session or host")
        pid = owner["pid"]
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise RuntimeError("Session writer lock has invalid PID")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError("Session writer lock is not safely recoverable") from error
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        raise RuntimeError("Session writer process liveness cannot be verified") from error
    else:
        raise RuntimeError("Session writer lock belongs to a live process")
    if path.read_bytes() != payload:
        raise RuntimeError("Session writer lock changed during PID verification")
    path.unlink()
