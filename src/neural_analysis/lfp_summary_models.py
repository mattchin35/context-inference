"""Validated immutable contracts and cache fingerprints for LFP summary analysis.

All times are seconds, frequencies are Hz, sample rates are samples/second, and
source fingerprints are inexpensive change detectors rather than content hashes
of large recordings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LFPSiteConfig:
    """One saved LFP channel with a zero-based channel index and explicit units."""

    stable_id: str
    label: str
    acquisition_format: str
    lfp_path: Path
    aligned_sync_path: Path | None
    probe_label: str
    saved_channel_index: int
    voltage_unit: str
    sample_rate_hz: float


@dataclass(frozen=True)
class UnitPopulationConfig:
    """One population of probe-qualified units and zero-based selected channels."""

    label: str
    probe_label: str
    sorter_path: Path | None
    aligned_spike_path: Path | None
    selected_channels: tuple[int, ...]
    quality_settings: tuple[tuple[str, str], ...]
    stable_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class TrialFilterConfig:
    """Choice/context filters and nonnegative trial-table row exclusions."""

    choice: str = "all"
    context: str = "all"
    excluded_trial_indices: tuple[int, ...] = ()


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


@dataclass(frozen=True)
class PhaseAnalysisConfig:
    """Morlet settings for a one-dimensional ascending frequency grid in Hz."""

    frequency_hz: tuple[float, ...] = tuple(float(value) for value in range(2, 101, 2))
    morlet_gaussian_width: float = 1.5
    morlet_window_length: float = 1.0
    morlet_precision: int = 16
    morlet_normalization: str = "l1"
    output_rate_hz: float = 500.0
    notch_enabled: bool = True
    notch_hz: float = 60.0
    notch_quality_factor: float = 30.0
    numerical_amplitude_threshold: float = 0.0
    absolute_amplitude_thresholds: tuple[tuple[str, float], ...] = ()
    block_duration_s: float = 120.0
    bands: tuple[FrequencyBandConfig, ...] = (
        FrequencyBandConfig("theta", 6.0, 10.0),
        FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),)),
    )
    bootstrap_count: int = 1000
    seed: int = 0


@dataclass(frozen=True)
class PPCAnalysisConfig:
    """PPC reliability, null-distribution, and phase-bin settings."""

    epochs: tuple[str, ...] = ("whole", "before", "after")
    minimum_computable_spikes: int = 2
    minimum_reliable_spikes: int = 50
    shuffle_count: int = 1000
    fdr_alpha: float = 0.05
    fdr_method: str = "bh"
    phase_bin_edges_rad: tuple[float, ...] = (-3.141592653589793, 0.0, 3.141592653589793)
    seed: int = 0


@dataclass(frozen=True)
class PPCExecutionConfig:
    """Work-only execution settings for bounded PPC computation and restart.

    These categorical settings control memory and progress behavior only. They
    are serialized in work metadata but deliberately excluded from final
    scientific component fingerprints.
    """

    unit_block_size: int = 8
    shuffle_block_size: int = 25
    trial_edge_block_size: int = 64
    worker_count: int = 1
    prepared_phase_cache_enabled: bool = True
    checkpoint_enabled: bool = True
    checkpoint_retention: str = "incomplete_only"
    progress_update_interval: int = 1


@dataclass(frozen=True)
class LFPSummaryConfig:
    """Complete immutable configuration; paths identify inputs and cache location."""

    session_id: str
    session_path: Path
    output_directory: Path
    sites: tuple[LFPSiteConfig, ...]
    site_pairs: tuple[tuple[str, str], ...]
    unit_population: UnitPopulationConfig | None
    trial_filter: TrialFilterConfig
    analysis_windows: AnalysisWindowConfig
    power: PowerAnalysisConfig
    phase: PhaseAnalysisConfig
    ppc: PPCAnalysisConfig
    ppc_execution: PPCExecutionConfig = PPCExecutionConfig()
    trial_table_path: Path | None = None
    schema_version: str = "1"
    random_seed: int = 0


@dataclass(frozen=True)
class ProgressEvent:
    """Framework-independent progress record with nonnegative work-unit counts."""

    component: str
    stage: str
    completed_count: int
    total_count: int | None
    message: str


def default_lfp_summary_config() -> LFPSummaryConfig:
    """Return an approved default configuration with placeholder source paths.

    Returns
    -------
    LFPSummaryConfig
        Frozen configuration using seconds, Hz, uV, and samples/second. Paths
        need not exist; missing sources are represented later by fingerprints.
    """
    sites = tuple(
        LFPSiteConfig(name, name, "spikeglx", Path(f"{name}.bin"), Path(f"{name}.sync.json"), name, channel, "uV", 2500.0)
        for name, channel in (("PFC", 5), ("HPC1", 222), ("HPC2", 14))
    )
    config = LFPSummaryConfig(
        "default-session", Path("."), Path("processed/lfp_summary_cache"), sites,
        (("PFC", "HPC1"), ("PFC", "HPC2"), ("HPC1", "HPC2")), None,
        TrialFilterConfig(), AnalysisWindowConfig(), PowerAnalysisConfig(), PhaseAnalysisConfig(), PPCAnalysisConfig(),
    )
    validate_lfp_summary_config(config)
    return config


def _primitive(value: Any) -> Any:
    """Convert configuration values to JSON primitives without numerical transforms.

    Parameters
    ----------
    value : Any
        Dataclass, ``Path``, tuple, list, mapping, or scalar configuration value.

    Returns
    -------
    Any
        JSON-compatible value retaining physical units and array-axis semantics;
        this helper never receives raw signal arrays or changes missing values.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, list):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _primitive(asdict(value))
    return value


def canonical_config_json(config: LFPSummaryConfig) -> str:
    """Serialize a validated configuration into deterministic JSON.

    Parameters
    ----------
    config : LFPSummaryConfig
        Frozen configuration with physical values expressed in documented units.

    Returns
    -------
    str
        Sorted-key JSON without raw recordings, numerical result arrays, or NaN.
    """
    validate_lfp_summary_config(config)
    return json.dumps(_primitive(config), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _construct(cls: type[Any], data: dict[str, Any]) -> Any:
    """Restore one configuration dataclass from decoded JSON primitives.

    Parameters
    ----------
    cls : type[Any]
        Frozen dataclass type to instantiate.
    data : dict[str, Any]
        JSON object with paths encoded as strings and no raw numerical arrays.

    Returns
    -------
    Any
        ``cls`` instance with declared path fields restored to ``Path`` objects.
    """
    path_fields = {
        "session_path", "output_directory", "lfp_path", "aligned_sync_path",
        "sorter_path", "aligned_spike_path", "trial_table_path",
    }
    kwargs: dict[str, Any] = {}
    for field in fields(cls):
        value = data[field.name]
        kwargs[field.name] = Path(value) if field.name in path_fields and value is not None else value
    return cls(**kwargs)


def lfp_summary_config_from_json(encoded: str) -> LFPSummaryConfig:
    """Restore and validate configuration paths and tuples from canonical JSON.

    Parameters
    ----------
    encoded : str
        JSON produced by :func:`canonical_config_json`; physical units remain in
        their serialized scalar fields and no raw data arrays are accepted.

    Returns
    -------
    LFPSummaryConfig
        Frozen validated configuration. Optional paths retain ``None`` missingness.
    """
    raw = json.loads(encoded)
    raw["sites"] = tuple(_construct(LFPSiteConfig, item) for item in raw["sites"])
    raw["site_pairs"] = tuple(tuple(item) for item in raw["site_pairs"])
    raw["trial_filter"] = _construct(TrialFilterConfig, {**raw["trial_filter"], "excluded_trial_indices": tuple(raw["trial_filter"]["excluded_trial_indices"])})
    raw["analysis_windows"] = _construct(AnalysisWindowConfig, raw["analysis_windows"])
    for key, cls in (("power", PowerAnalysisConfig), ("phase", PhaseAnalysisConfig)):
        item = raw[key]
        item["bands"] = tuple(FrequencyBandConfig(band["name"], band["lower_hz"], band["upper_hz"], tuple(tuple(interval) for interval in band["excluded_intervals_hz"])) for band in item["bands"])
        if key == "phase":
            item["frequency_hz"] = tuple(item["frequency_hz"])
            item["absolute_amplitude_thresholds"] = tuple(
                tuple(threshold) for threshold in item["absolute_amplitude_thresholds"]
            )
        raw[key] = _construct(cls, item)
    ppc = raw["ppc"]
    ppc["epochs"] = tuple(ppc["epochs"])
    ppc["phase_bin_edges_rad"] = tuple(ppc["phase_bin_edges_rad"])
    raw["ppc"] = _construct(PPCAnalysisConfig, ppc)
    raw["ppc_execution"] = _construct(PPCExecutionConfig, raw["ppc_execution"])
    raw["session_path"] = Path(raw["session_path"])
    raw["output_directory"] = Path(raw["output_directory"])
    if raw["trial_table_path"] is not None:
        raw["trial_table_path"] = Path(raw["trial_table_path"])
    if raw["unit_population"] is not None:
        population = raw["unit_population"]
        population["selected_channels"] = tuple(population["selected_channels"])
        population["quality_settings"] = tuple(tuple(item) for item in population["quality_settings"])
        population["stable_unit_ids"] = tuple(population["stable_unit_ids"])
        raw["unit_population"] = _construct(UnitPopulationConfig, population)
    config = _construct(LFPSummaryConfig, raw)
    validate_lfp_summary_config(config)
    return config


def _validate_bands(bands: tuple[FrequencyBandConfig, ...]) -> None:
    """Validate named frequency bands and excluded intervals.

    Parameters
    ----------
    bands : tuple[FrequencyBandConfig, ...]
        Band bounds and exclusions in Hz; this is a one-dimensional frequency contract.

    Returns
    -------
    None
        No frequency grid is resampled and no missing values are introduced.
    """
    names = [band.name for band in bands]
    if not bands or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("band names must be nonempty and unique")
    for band in bands:
        if not (isfinite(band.lower_hz) and isfinite(band.upper_hz) and 0 <= band.lower_hz < band.upper_hz):
            raise ValueError("invalid band bounds")
        previous_stop = band.lower_hz
        for low, high in sorted(band.excluded_intervals_hz):
            if not (isfinite(low) and isfinite(high) and band.lower_hz <= low < high <= band.upper_hz):
                raise ValueError("invalid excluded band interval")
            if low < previous_stop:
                raise ValueError("excluded band intervals must not overlap")
            previous_stop = high


def _validate_unit_population(population: UnitPopulationConfig) -> None:
    """Validate probe-qualified unit IDs and selected acquisition channels.

    Parameters
    ----------
    population : UnitPopulationConfig
        Unit identifiers plus one-dimensional zero-based channel indices; no spike
        times or ragged arrays are present at this configuration boundary.

    Returns
    -------
    None
        Validation does not change source paths or represent missing spikes.
    """
    channels = population.selected_channels
    if any(not isinstance(channel, int) or channel < 0 for channel in channels) or len(channels) != len(set(channels)):
        raise ValueError("unit selected channels must be unique nonnegative integers")
    unit_ids = population.stable_unit_ids
    prefix = f"{population.probe_label}:"
    if len(unit_ids) != len(set(unit_ids)) or any(not unit_id.startswith(prefix) or unit_id == prefix for unit_id in unit_ids):
        raise ValueError("unit ids must be unique and probe-qualified")


def validate_lfp_summary_config(config: LFPSummaryConfig) -> None:
    """Validate a complete configuration before any raw input is opened.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable configuration. Time values use seconds, frequency values use
        Hz, sample rates use samples/second, and channel indices are zero based.

    Returns
    -------
    None
        Validation has no value return and never changes configuration values.

    Raises
    ------
    ValueError
        If an identity, unit-bearing scalar, epoch, filter, or numerical
        contract is invalid. Missing source files are intentionally permitted;
        they are represented by source fingerprints later.
    """
    site_ids = _validate_sites(config)
    _validate_site_pairs(config.site_pairs, site_ids)
    _validate_windows(config.analysis_windows)
    _validate_trial_filter(config.trial_filter)
    _validate_power(config.power)
    _validate_phase(config.phase, site_ids)
    _validate_ppc(config.ppc)
    _validate_ppc_execution(config.ppc_execution)
    if config.unit_population is not None:
        _validate_unit_population(config.unit_population)


def _validate_sites(config: LFPSummaryConfig) -> set[str]:
    """Validate configured sites and return their stable identifiers.

    Parameters
    ----------
    config : LFPSummaryConfig
        Configuration whose sites represent scalar-voltage LFP traces.

    Returns
    -------
    set[str]
        Unique stable site identifiers. No source file is opened.
    """
    site_ids = [site.stable_id for site in config.sites]
    if not config.session_id or not site_ids or len(site_ids) != len(set(site_ids)):
        raise ValueError("session id and site stable ids must be nonempty and unique")
    for site in config.sites:
        if not all((site.label, site.acquisition_format, site.probe_label, site.voltage_unit)):
            raise ValueError("site identity and voltage unit must be nonempty")
        if not isinstance(site.saved_channel_index, int) or site.saved_channel_index < 0:
            raise ValueError("channel indices must be nonnegative integers")
        if not isfinite(site.sample_rate_hz) or site.sample_rate_hz <= 0:
            raise ValueError("site sample rates must be finite and positive")
    return set(site_ids)


def _validate_site_pairs(pairs: tuple[tuple[str, str], ...], site_ids: set[str]) -> None:
    """Validate ordered pair identities reference distinct configured site IDs.

    Parameters
    ----------
    pairs : tuple[tuple[str, str], ...]
        Ordered phase pairs; phase offset convention is first site minus second.
    site_ids : set[str]
        Site identifiers returned by :func:`_validate_sites`.

    Returns
    -------
    None
        Pair validation has no data or missing-value output.
    """
    if any(left == right or left not in site_ids or right not in site_ids for left, right in pairs):
        raise ValueError("site pairs must contain distinct configured sites")
    if len(pairs) != len(set(pairs)):
        raise ValueError("site pairs must be unique")


def _validate_windows(window: AnalysisWindowConfig) -> None:
    """Validate half-open event-relative windows expressed in seconds.

    Parameters
    ----------
    window : AnalysisWindowConfig
        Whole/before/after boundaries in seconds.

    Returns
    -------
    None
        No window values are transformed; NaN and infinite boundaries are rejected.
    """
    window_values = (window.whole_start_s, window.whole_stop_s, window.before_start_s, window.before_stop_s, window.after_start_s, window.after_stop_s)
    if window.alignment_event not in {"choice_time", "start_time"}:
        raise ValueError("unsupported alignment")
    if not all(isfinite(value) for value in window_values) or not (window.whole_start_s == window.before_start_s and window.before_stop_s == window.after_start_s and window.after_stop_s == window.whole_stop_s and window.whole_start_s < window.before_stop_s < window.whole_stop_s):
        raise ValueError("before and after windows must partition the whole window")


def _validate_trial_filter(trial_filter: TrialFilterConfig) -> None:
    """Validate categorical filters and nonnegative trial-table row indices.

    Parameters
    ----------
    trial_filter : TrialFilterConfig
        Choice/context labels and integer trial-table row positions.

    Returns
    -------
    None
        Validation does not infer missing behavioral values or alter indices.
    """
    if trial_filter.choice not in {"all", "left", "right"} or trial_filter.context not in {"all", "left", "right"}:
        raise ValueError("unsupported trial filter")
    if len(trial_filter.excluded_trial_indices) != len(set(trial_filter.excluded_trial_indices)) or any(not isinstance(index, int) or index < 0 for index in trial_filter.excluded_trial_indices):
        raise ValueError("invalid excluded trial indices")


def _validate_power(power: PowerAnalysisConfig) -> None:
    """Validate Welch, notch, reference, and band settings for linear PSD values.

    Parameters
    ----------
    power : PowerAnalysisConfig
        Durations in seconds and frequencies in Hz.

    Returns
    -------
    None
        No PSD array is created and no missing reference is substituted.
    """
    power_values = (
        power.welch_window_s, power.welch_overlap_fraction,
        power.canonical_frequency_step_hz, power.notch_hz,
        power.notch_quality_factor, power.presession_reference_duration_s,
    )
    valid_ranges = (
        power.welch_window_s > 0
        and 0 <= power.welch_overlap_fraction < 1
        and power.canonical_frequency_step_hz > 0
        and power.notch_hz > 0
        and power.notch_quality_factor > 0
        and power.presession_reference_duration_s > 0
    )
    if power.welch_window != "hann_periodic" or power.welch_detrend != "constant" or not all(isfinite(value) for value in power_values) or not valid_ranges:
        raise ValueError("invalid power parameters")
    _validate_bands(power.bands)


def _validate_phase(phase: PhaseAnalysisConfig, site_ids: set[str]) -> None:
    """Validate phase-transform settings, phase bands, and site thresholds.

    Parameters
    ----------
    phase : PhaseAnalysisConfig
        Frequencies in Hz, output rate in samples/second, and thresholds in each
        site's source voltage unit.
    site_ids : set[str]
        Valid stable site identities for threshold mappings.

    Returns
    -------
    None
        No wavelet values are created; unavailable phase remains a later NaN state.
    """
    phase_values = (
        phase.morlet_gaussian_width, phase.morlet_window_length,
        phase.output_rate_hz, phase.notch_hz, phase.notch_quality_factor,
        phase.numerical_amplitude_threshold, phase.block_duration_s,
    )
    valid_ranges = (
        phase.morlet_gaussian_width > 0
        and phase.morlet_window_length > 0
        and phase.output_rate_hz > 0
        and phase.notch_hz > 0
        and phase.notch_quality_factor > 0
        and phase.numerical_amplitude_threshold >= 0
        and phase.block_duration_s > 0
    )
    if not all(isfinite(value) for value in phase_values) or not valid_ranges or phase.morlet_precision <= 0 or phase.morlet_normalization != "l1":
        raise ValueError("invalid phase parameters")
    if not phase.frequency_hz or any(not isfinite(value) or value <= 0 for value in phase.frequency_hz) or any(left >= right for left, right in zip(phase.frequency_hz, phase.frequency_hz[1:])) or phase.bootstrap_count <= 0:
        raise ValueError("invalid phase frequency or bootstrap settings")
    _validate_bands(phase.bands)
    threshold_ids = [site_id for site_id, _ in phase.absolute_amplitude_thresholds]
    invalid_threshold = any(
        site_id not in site_ids or not isfinite(threshold) or threshold < 0
        for site_id, threshold in phase.absolute_amplitude_thresholds
    )
    if len(threshold_ids) != len(set(threshold_ids)) or invalid_threshold:
        raise ValueError("absolute amplitude thresholds must be unique, configured, and nonnegative")


def _validate_ppc(ppc: PPCAnalysisConfig) -> None:
    """Validate PPC reliability, permutation, FDR, and phase-bin contracts.

    Parameters
    ----------
    ppc : PPCAnalysisConfig
        Counts are integer samples/spikes; phase-bin edges are radians.

    Returns
    -------
    None
        Validation does not calculate PPC; unavailable PPC is later represented by NaN.
    """
    if ppc.epochs != ("whole", "before", "after") or not isfinite(ppc.fdr_alpha) or not 0 < ppc.fdr_alpha <= 1 or ppc.fdr_method != "bh":
        raise ValueError("invalid PPC epoch or FDR settings")
    if ppc.minimum_computable_spikes != 2 or ppc.minimum_reliable_spikes < 50 or ppc.shuffle_count <= 0:
        raise ValueError("invalid PPC count settings")
    if len(ppc.phase_bin_edges_rad) < 2 or any(not isfinite(edge) for edge in ppc.phase_bin_edges_rad) or any(left >= right for left, right in zip(ppc.phase_bin_edges_rad, ppc.phase_bin_edges_rad[1:])):
        raise ValueError("phase bin edges must be finite and ascending")


def _validate_ppc_execution(execution: PPCExecutionConfig) -> None:
    """Validate work-only PPC block, cache, checkpoint, and progress settings.

    Parameters
    ----------
    execution : PPCExecutionConfig
        Categorical block sizes/counts and Boolean work-artifact policies. No
        scientific phase, spike, time, frequency, or missing-value data occurs
        at this configuration boundary.

    Returns
    -------
    None
        Validation neither changes settings nor contributes to final scientific
        component fingerprints.
    """
    positive_values = (
        execution.unit_block_size,
        execution.shuffle_block_size,
        execution.trial_edge_block_size,
        execution.worker_count,
        execution.progress_update_interval,
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in positive_values
        )
        or not isinstance(execution.prepared_phase_cache_enabled, bool)
        or not isinstance(execution.checkpoint_enabled, bool)
        or execution.checkpoint_retention not in {"incomplete_only", "retain"}
    ):
        raise ValueError("invalid PPC execution settings")


def _phase_transform_payload(phase: PhaseAnalysisConfig, include_bootstrap: bool) -> dict[str, Any]:
    """Build phase settings relevant to one component fingerprint.

    Parameters
    ----------
    phase : PhaseAnalysisConfig
        Transform values in Hz, seconds, samples/second, and source voltage units.
    include_bootstrap : bool
        Whether Synchrony-only bootstrap count and seed are retained.

    Returns
    -------
    dict[str, Any]
        JSON primitives without raw arrays. No numerical missingness is created.
    """
    payload = _primitive(phase)
    if not include_bootstrap:
        payload.pop("bootstrap_count")
        payload.pop("seed")
    return payload


def component_fingerprint(component: str, config: LFPSummaryConfig) -> str:
    """Return a SHA-256 fingerprint for one component's configuration subset.

    Parameters
    ----------
    component : str
        ``power``, ``synchrony``, or ``spike_phase``.
    config : LFPSummaryConfig
        Valid configuration with all values retained in documented physical units.

    Returns
    -------
    str
        Hex digest. Raw arrays and source-file missingness are deliberately excluded.
    """
    validate_lfp_summary_config(config)
    shared = {"schema_version": config.schema_version, "session_id": config.session_id, "session_path": config.session_path, "trial_table_path": config.trial_table_path, "sites": config.sites, "trial_filter": config.trial_filter, "windows": config.analysis_windows}
    if component == "power":
        payload = {**shared, "power": config.power}
    elif component == "synchrony":
        payload = {**shared, "site_pairs": config.site_pairs, "phase": _phase_transform_payload(config.phase, True)}
    elif component == "spike_phase":
        payload = {**shared, "unit_population": config.unit_population, "phase": _phase_transform_payload(config.phase, False), "ppc": config.ppc}
    else:
        raise ValueError(f"unknown component: {component}")
    encoded = json.dumps(_primitive(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def fingerprint_source_files(
    config: LFPSummaryConfig,
    component: str | None = None,
) -> dict[str, dict[str, int | str]]:
    """Return component-scoped resolved-path, byte-size, and mtime fingerprints.

    Parameters
    ----------
    config : LFPSummaryConfig
        Configuration containing LFP, trial-table, and optional unit source paths.
    component : str | None
        Component name or ``None`` for all configured sources. Unit inputs apply
        only to ``spike_phase``; LFP and trial-table inputs apply to all.

    Returns
    -------
    dict[str, dict[str, int | str]]
        Resolved path metadata. Missing files have size and mtime ``-1``.
    """
    if component not in {None, "power", "synchrony", "spike_phase"}:
        raise ValueError(f"unknown component: {component}")
    paths = [path for site in config.sites for path in (site.lfp_path, site.aligned_sync_path) if path is not None]
    if config.trial_table_path is not None:
        paths.append(config.trial_table_path)
    if config.unit_population is not None and component in {None, "spike_phase"}:
        paths.extend(path for path in (config.unit_population.sorter_path, config.unit_population.aligned_spike_path) if path is not None)
    result: dict[str, dict[str, int | str]] = {}
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat() if resolved.exists() else None
        result[str(resolved)] = {"path": str(resolved), "size_bytes": stat.st_size if stat else -1, "mtime_ns": stat.st_mtime_ns if stat else -1}
    return result
