"""Contract tests for immutable LFP-summary configuration models."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from math import nan
import os
from pathlib import Path
from typing import Callable

import pytest

from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    FrequencyBandConfig,
    PPCExecutionConfig,
    ProgressEvent,
    UnitPopulationConfig,
    canonical_config_json,
    component_fingerprint,
    default_lfp_summary_config,
    fingerprint_source_files,
    lfp_summary_config_from_json,
    validate_lfp_summary_config,
)
from src.neural_analysis import lfp_summary_models


ConfigMutation = Callable[[LFPSummaryConfig], LFPSummaryConfig]


def _default_config_with_temporary_sources(tmp_path: Path) -> LFPSummaryConfig:
    """Return a valid config whose LFP and sync paths are files under ``tmp_path``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest-managed temporary directory.

    Returns
    -------
    LFPSummaryConfig
        Valid immutable configuration with one required sidecar per LFP site.
    """
    config = default_lfp_summary_config()
    sites = []
    for site in config.sites:
        lfp_path = tmp_path / f"{site.stable_id}.bin"
        sync_path = tmp_path / f"{site.stable_id}.sync.json"
        lfp_path.write_bytes(b"lfp")
        sync_path.write_text("{}", encoding="ascii")
        sites.append(replace(site, lfp_path=lfp_path, aligned_sync_path=sync_path))
    return replace(config, sites=tuple(sites))


def _open_ephys_config_with_temporary_sidecar(tmp_path: Path) -> LFPSummaryConfig:
    """Return a configuration with one authoritative Open Ephys sidecar.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed directory containing synthetic LFP, sync, and metadata
        source files only.

    Returns
    -------
    LFPSummaryConfig
        Valid configuration retaining the first site's stable identity while
        declaring the Open Ephys adapter and physical microvolt metadata.
    """
    config = _default_config_with_temporary_sources(tmp_path)
    first_site = config.sites[0]
    lfp_path = tmp_path / "open_ephys" / "lfp.dat"
    sync_path = tmp_path / "open_ephys" / "probe_sync.npz"
    lfp_path.parent.mkdir()
    lfp_path.write_bytes(b"lfp")
    sync_path.write_bytes(b"sync")
    sidecar_path = lfp_path.parent / "lfp_preprocessing.json"
    sidecar_path.write_text(
        '{"output_binary":"lfp.dat","sampling_frequency_hz":2500.0,'
        '"num_channels":1,"num_segments":1,"num_samples_by_segment":[1],'
        '"dtype":"float32","binary_layout":"time_major_channel_interleaved",'
        '"channel_ids_in_binary_order":["CH0"],"lfp_binary_scaling":'
        '{"data_units":"unscaled_binary_values","has_scaleable_traces":true,'
        '"channel_ids":["CH0"],"gain_to_uV_by_channel":[0.195],'
        '"offset_to_uV_by_channel":[0.0],"physical_unit_by_channel":["uV"],'
        '"export_scale_factor":1.0,'
        '"conversion":"trace_uV = trace_value * gain_to_uV + offset_to_uV"}}',
        encoding="ascii",
    )
    open_ephys_site = replace(
        first_site,
        acquisition_format="open_ephys",
        lfp_path=lfp_path,
        aligned_sync_path=sync_path,
        voltage_unit="uV",
        sample_rate_hz=2500.0,
    )
    return replace(config, sites=(open_ephys_site,) + config.sites[1:])


def test_default_configuration_is_frozen_deterministic_and_round_trips() -> None:
    """Default config is immutable and has stable lossless canonical JSON serialization."""
    config = default_lfp_summary_config()

    encoded_once = canonical_config_json(config)
    encoded_twice = canonical_config_json(config)
    decoded_config = lfp_summary_config_from_json(encoded_once)

    assert encoded_once == encoded_twice
    assert decoded_config == config
    with pytest.raises((AttributeError, TypeError)):
        config.schema_version = "mutated"  # type: ignore[misc]


def test_default_phase_frequency_grid_matches_approved_contract() -> None:
    """Default wavelet frequencies are the configured linear 2--100 Hz grid."""
    config = default_lfp_summary_config()

    assert config.analysis_windows.alignment_event == "choice_time"
    assert config.phase.frequency_hz == tuple(float(value) for value in range(2, 101, 2))


def test_default_wavelet_transform_settings_match_approved_contract() -> None:
    """Default Morlet settings preserve the approved transform implementation contract."""
    config = default_lfp_summary_config()

    assert config.phase.output_rate_hz == 500.0
    assert config.phase.morlet_gaussian_width == 1.5
    assert config.phase.morlet_window_length == 1.0
    assert config.phase.morlet_precision == 16
    assert config.phase.morlet_normalization == "l1"
    assert config.phase.block_duration_s == 120.0


def test_default_notch_and_presession_reference_settings_match_contract() -> None:
    """Default power settings explicitly retain the notch and exact reference duration."""
    config = default_lfp_summary_config()

    assert config.power.notch_enabled is True
    assert config.power.notch_hz == 60.0
    assert config.power.notch_quality_factor == 30.0
    assert config.power.presession_reference_duration_s == 10.0


def test_default_power_bands_and_welch_contract_match_approved_settings() -> None:
    """Power defaults retain approved bands, line-noise gap, and Welch estimator settings."""
    config = default_lfp_summary_config()

    assert config.power.bands == (
        FrequencyBandConfig("theta", 6.0, 10.0),
        FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),)),
    )
    assert config.power.welch_window == "hann_periodic"
    assert config.power.welch_detrend == "constant"
    assert config.power.welch_overlap_fraction == 0.5
    assert config.power.canonical_frequency_step_hz == 2.0


def test_default_phase_notch_and_site_sample_rate_contracts_are_explicit() -> None:
    """Phase preprocessing and every selected site expose their required physical settings."""
    config = default_lfp_summary_config()

    assert config.phase.notch_enabled is True
    assert config.phase.notch_hz == 60.0
    assert config.phase.notch_quality_factor == 30.0
    assert config.phase.bands == (
        FrequencyBandConfig("theta", 6.0, 10.0),
        FrequencyBandConfig("gamma", 30.0, 80.0, ((58.0, 62.0),)),
    )
    assert config.phase.absolute_amplitude_thresholds == ()
    assert all(site.sample_rate_hz > 0.0 for site in config.sites)


def test_per_site_absolute_amplitude_thresholds_are_validated_by_stable_site_id() -> None:
    """Absolute phase-amplitude thresholds are optional nonnegative values per stable site."""
    config = default_lfp_summary_config()
    configured_threshold = replace(
        config,
        phase=replace(config.phase, absolute_amplitude_thresholds=(("PFC", 25.0),)),
    )

    validate_lfp_summary_config(configured_threshold)
    for invalid_thresholds in (
        (("unknown-site", 25.0),),
        (("PFC", -1.0),),
        (("PFC", 1.0), ("PFC", 2.0)),
    ):
        with pytest.raises(ValueError):
            validate_lfp_summary_config(
                replace(config, phase=replace(config.phase, absolute_amplitude_thresholds=invalid_thresholds))
            )


def test_start_time_is_supported_and_unapproved_alignments_are_rejected() -> None:
    """Only the approved choice-time and start-time event columns are valid."""
    config = default_lfp_summary_config()

    validate_lfp_summary_config(
        replace(config, analysis_windows=replace(config.analysis_windows, alignment_event="start_time"))
    )
    for unapproved_event in ("cue_time", "outcome_time"):
        with pytest.raises(ValueError):
            validate_lfp_summary_config(
                replace(config, analysis_windows=replace(config.analysis_windows, alignment_event=unapproved_event))
            )


@pytest.mark.parametrize(
    "mutated_config",
    [
        lambda config: replace(config, analysis_windows=replace(config.analysis_windows, before_stop_s=-0.5)),
        lambda config: replace(config, phase=replace(config.phase, frequency_hz=(2.0, 8.0, 6.0))),
        lambda config: replace(config, phase=replace(config.phase, bootstrap_count=0)),
        lambda config: replace(config, ppc=replace(config.ppc, minimum_reliable_spikes=1)),
        lambda config: replace(config, sites=(replace(config.sites[0], saved_channel_index=-1),) + config.sites[1:]),
        lambda config: replace(
            config,
            unit_population=UnitPopulationConfig("units", "PFC", None, None, (1, 1), (), ("PFC:1",)),
        ),
        lambda config: replace(config, power=replace(config.power, welch_window_s=nan)),
        lambda config: replace(config, phase=replace(config.phase, output_rate_hz=nan)),
        lambda config: replace(config, ppc=replace(config.ppc, fdr_alpha=nan)),
        lambda config: replace(
            config,
            power=replace(
                config.power,
                bands=(config.power.bands[0], replace(config.power.bands[0], lower_hz=7.0)),
            ),
        ),
        lambda config: replace(
            config,
            power=replace(
                config.power,
                bands=(
                    replace(config.power.bands[1], excluded_intervals_hz=((40.0, 60.0), (50.0, 70.0))),
                ),
            ),
        ),
        lambda config: replace(
            config,
            unit_population=UnitPopulationConfig("units", "PFC", None, None, (-1,), (), ("PFC:1",)),
        ),
        lambda config: replace(
            config,
            unit_population=UnitPopulationConfig("units", "PFC", None, None, (1,), (), ("PFC:1", "PFC:1")),
        ),
        lambda config: replace(
            config,
            unit_population=UnitPopulationConfig("units", "PFC", None, None, (1,), (), ("unqualified-id",)),
        ),
    ],
)
def test_invalid_numeric_epoch_and_unit_contracts_are_rejected(mutated_config: ConfigMutation) -> None:
    """Reject malformed epoch partitions, numerical settings, and unit selections before I/O."""
    with pytest.raises(ValueError):
        validate_lfp_summary_config(mutated_config(default_lfp_summary_config()))


@pytest.mark.parametrize(
    ("description", "mutated_config"),
    [
        (
            "duplicate stable site id",
            lambda config: replace(
                config,
                sites=(config.sites[0], replace(config.sites[1], stable_id=config.sites[0].stable_id)),
            ),
        ),
        (
            "duplicate site pair",
            lambda config: replace(config, site_pairs=(config.site_pairs[0], config.site_pairs[0])),
        ),
        (
            "pair contains an unknown site",
            lambda config: replace(config, site_pairs=(("PFC", "unknown-site"),)),
        ),
        (
            "overlapping before and after windows",
            lambda config: replace(
                config,
                analysis_windows=replace(config.analysis_windows, after_start_s=-0.5),
            ),
        ),
        (
            "inverted whole window",
            lambda config: replace(
                config,
                analysis_windows=replace(config.analysis_windows, whole_start_s=2.0, whole_stop_s=1.0),
            ),
        ),
        (
            "invalid frequency band bounds",
            lambda config: replace(
                config,
                power=replace(config.power, bands=(replace(config.power.bands[0], lower_hz=9.0, upper_hz=8.0),)),
            ),
        ),
        (
            "excluded interval extends outside its band",
            lambda config: replace(
                config,
                power=replace(
                    config.power,
                    bands=(replace(config.power.bands[1], excluded_intervals_hz=((20.0, 30.0),)),),
                ),
            ),
        ),
        (
            "unsupported trial filter",
            lambda config: replace(config, trial_filter=replace(config.trial_filter, choice="up")),
        ),
        (
            "duplicate explicitly excluded trial index",
            lambda config: replace(config, trial_filter=replace(config.trial_filter, excluded_trial_indices=(3, 3))),
        ),
    ],
)
def test_invalid_configurations_are_rejected(
    description: str, mutated_config: ConfigMutation
) -> None:
    """Reject invalid site/pair/window/band/filter contracts before input files open.

    Parameters
    ----------
    description : str
        Human-readable reason the provided configuration is invalid.
    mutated_config : Callable[[LFPSummaryConfig], LFPSummaryConfig]
        Pure function producing one invalid immutable configuration.
    """
    del description
    with pytest.raises(ValueError):
        validate_lfp_summary_config(mutated_config(default_lfp_summary_config()))


def test_component_fingerprints_only_include_relevant_settings() -> None:
    """Each component fingerprint includes exactly its scientific dependencies."""
    config = default_lfp_summary_config()
    changed_ppc = replace(config, ppc=replace(config.ppc, shuffle_count=config.ppc.shuffle_count + 1))
    changed_power = replace(
        config,
        power=replace(
            config.power,
            canonical_frequency_step_hz=config.power.canonical_frequency_step_hz + 0.5,
        ),
    )

    assert component_fingerprint("power", config) == component_fingerprint("power", changed_ppc)
    assert component_fingerprint("synchrony", config) == component_fingerprint("synchrony", changed_ppc)
    assert component_fingerprint("spike_phase", config) != component_fingerprint("spike_phase", changed_ppc)
    assert component_fingerprint("power", config) != component_fingerprint("power", changed_power)
    assert component_fingerprint("synchrony", config) == component_fingerprint("synchrony", changed_power)
    assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed_power)


def test_ppc_execution_configuration_round_trips_without_staling_final_components() -> None:
    """Execution settings serialize for work metadata but not scientific cache identity."""
    config = default_lfp_summary_config()
    changed_execution = replace(
        config,
        ppc_execution=PPCExecutionConfig(
            unit_block_size=3,
            shuffle_block_size=7,
            trial_edge_block_size=11,
            worker_count=1,
            prepared_phase_cache_enabled=False,
            checkpoint_enabled=False,
            checkpoint_retention="retain",
            progress_update_interval=2,
            maximum_worker_allocation_bytes=123_456,
            maximum_aggregate_allocation_bytes=654_321,
        ),
    )

    encoded = canonical_config_json(changed_execution)
    decoded = lfp_summary_config_from_json(encoded)

    assert decoded == changed_execution
    assert '"unit_block_size":3' in encoded
    assert '"maximum_worker_allocation_bytes":123456' in encoded
    assert '"maximum_aggregate_allocation_bytes":654321' in encoded
    for component in ("power", "synchrony", "spike_phase"):
        assert component_fingerprint(component, config) == component_fingerprint(
            component,
            changed_execution,
        )


def test_ppc_execution_allocation_limits_default_round_trip_and_preserve_scientific_fingerprints() -> None:
    """The approved byte limits are exact execution-only integers in JSON.

    These limits constrain planned private NumPy arrays.  They deliberately
    remain outside every final scientific component fingerprint.
    """
    config = default_lfp_summary_config()
    execution = config.ppc_execution
    encoded = canonical_config_json(config)
    decoded = lfp_summary_config_from_json(encoded)
    changed = replace(
        config,
        ppc_execution=replace(
            execution,
            maximum_worker_allocation_bytes=2 * 1024**3 - 1,
            maximum_aggregate_allocation_bytes=12 * 1024**3 - 1,
        ),
    )

    assert execution.maximum_worker_allocation_bytes == 2 * 1024**3
    assert execution.maximum_aggregate_allocation_bytes == 12 * 1024**3
    assert decoded.ppc_execution.maximum_worker_allocation_bytes == 2 * 1024**3
    assert decoded.ppc_execution.maximum_aggregate_allocation_bytes == 12 * 1024**3
    for component in ("power", "synchrony", "spike_phase"):
        assert component_fingerprint(component, config) == component_fingerprint(
            component,
            changed,
        )


@pytest.mark.parametrize(
    "execution",
    (
        PPCExecutionConfig(unit_block_size=0),
        PPCExecutionConfig(unit_block_size=True),
        PPCExecutionConfig(shuffle_block_size=0),
        PPCExecutionConfig(shuffle_block_size=False),
        PPCExecutionConfig(trial_edge_block_size=0),
        PPCExecutionConfig(trial_edge_block_size=True),
        PPCExecutionConfig(worker_count=0),
        PPCExecutionConfig(worker_count=False),
        PPCExecutionConfig(checkpoint_retention="forever"),
        PPCExecutionConfig(progress_update_interval=0),
        PPCExecutionConfig(progress_update_interval=True),
    ),
)
def test_invalid_ppc_execution_configuration_is_rejected(
    execution: PPCExecutionConfig,
) -> None:
    """Work-only PPC execution settings are validated before any files are opened."""
    with pytest.raises(ValueError):
        validate_lfp_summary_config(
            replace(default_lfp_summary_config(), ppc_execution=execution)
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("maximum_worker_allocation_bytes", 0),
        ("maximum_worker_allocation_bytes", False),
        ("maximum_worker_allocation_bytes", 1.5),
        ("maximum_aggregate_allocation_bytes", 0),
        ("maximum_aggregate_allocation_bytes", True),
        ("maximum_aggregate_allocation_bytes", 1.5),
    ),
)
def test_ppc_execution_allocation_limits_reject_boolean_or_nonpositive_values(
    field_name: str,
    invalid_value: int | bool | float,
) -> None:
    """Both planned-byte limits are positive integers, never Boolean flags."""
    with pytest.raises(ValueError):
        execution = replace(
            PPCExecutionConfig(),
            **{field_name: invalid_value},
        )
        validate_lfp_summary_config(
            replace(default_lfp_summary_config(), ppc_execution=execution)
        )


def test_progress_event_execution_fields_preserve_legacy_positional_construction() -> None:
    """Elapsed/ETA fields are optional so existing five-position calls remain valid."""
    legacy = ProgressEvent("spike_phase", "run", 1, 4, "legacy")
    timed = ProgressEvent("spike_phase", "checkpoint", 2, 4, "timed", 1.5, None)

    assert legacy.elapsed_seconds is None
    assert legacy.eta_seconds is None
    assert timed.elapsed_seconds == 1.5
    assert timed.eta_seconds is None


def test_transform_fingerprint_change_affects_synchrony_and_spike_phase() -> None:
    """A phase-transform setting invalidates both consumers of Morlet coefficients."""
    config = default_lfp_summary_config()
    changed_transform = replace(
        config,
        phase=replace(config.phase, morlet_gaussian_width=config.phase.morlet_gaussian_width + 0.1),
    )

    assert component_fingerprint("power", config) == component_fingerprint("power", changed_transform)
    assert component_fingerprint("synchrony", config) != component_fingerprint("synchrony", changed_transform)
    assert component_fingerprint("spike_phase", config) != component_fingerprint("spike_phase", changed_transform)


def test_bootstrap_count_fingerprint_change_affects_synchrony_only() -> None:
    """Synchrony bootstrap resampling does not invalidate the spike-phase component."""
    config = default_lfp_summary_config()
    changed_bootstrap_count = replace(
        config,
        phase=replace(config.phase, bootstrap_count=config.phase.bootstrap_count + 1),
    )

    assert component_fingerprint("power", config) == component_fingerprint("power", changed_bootstrap_count)
    assert component_fingerprint("synchrony", config) != component_fingerprint("synchrony", changed_bootstrap_count)
    assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed_bootstrap_count)


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_component_fingerprints_include_session_identity(component: str) -> None:
    """Every component fingerprint distinguishes different active sessions."""
    config = default_lfp_summary_config()
    changed_session = replace(config, session_id="another-session")

    assert component_fingerprint(component, config) != component_fingerprint(component, changed_session)


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_component_fingerprints_include_affected_trial_filters(component: str) -> None:
    """Choice/context selection changes the cache identity of every summary component."""
    config = default_lfp_summary_config()
    changed_filter = replace(config, trial_filter=replace(config.trial_filter, choice="left"))

    assert component_fingerprint(component, config) != component_fingerprint(component, changed_filter)


def test_source_fingerprints_include_files_and_required_sidecars(tmp_path: Path) -> None:
    """Source fingerprints change when an LFP file or its sync sidecar changes.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest temporary directory used to create synthetic source files.
    """
    config = _default_config_with_temporary_sources(tmp_path)
    before = fingerprint_source_files(config)

    first_site = config.sites[0]
    Path(first_site.lfp_path).write_bytes(b"larger lfp input")
    after_lfp_change = fingerprint_source_files(config)
    Path(first_site.aligned_sync_path).write_text('{"version": 2}', encoding="ascii")
    after_sidecar_change = fingerprint_source_files(config)

    assert after_lfp_change != before
    assert after_sidecar_change != after_lfp_change
    assert str(Path(first_site.aligned_sync_path).resolve()) in after_sidecar_change


def test_source_fingerprint_records_resolved_path_size_and_mtime(tmp_path: Path) -> None:
    """Each fingerprint entry exposes the specified non-cryptographic file metadata."""
    config = _default_config_with_temporary_sources(tmp_path)

    fingerprints = fingerprint_source_files(config)
    lfp_entry = fingerprints[str(Path(config.sites[0].lfp_path).resolve())]

    assert lfp_entry["path"] == str(Path(config.sites[0].lfp_path).resolve())
    assert lfp_entry["size_bytes"] == 3
    assert isinstance(lfp_entry["mtime_ns"], int)


def test_component_source_fingerprints_scope_unit_and_trial_table_inputs(tmp_path: Path) -> None:
    """Unit sources affect Spike phase only; the trial table is shared by every component."""
    config = _default_config_with_temporary_sources(tmp_path)
    sorter_path = tmp_path / "spike_clusters.npy"
    aligned_spike_path = tmp_path / "aligned_spikes.npy"
    trial_table_path = tmp_path / "trials.csv"
    sorter_path.write_bytes(b"sorter")
    aligned_spike_path.write_bytes(b"aligned")
    trial_table_path.write_text("choice_time\n1.0\n", encoding="ascii")
    configured_population = UnitPopulationConfig(
        "units", "PFC", sorter_path, aligned_spike_path, (5,), (), ("PFC:1",)
    )
    config = replace(config, unit_population=configured_population, trial_table_path=trial_table_path)

    power_sources = fingerprint_source_files(config, component="power")
    synchrony_sources = fingerprint_source_files(config, component="synchrony")
    spike_sources = fingerprint_source_files(config, component="spike_phase")

    assert str(sorter_path.resolve()) not in power_sources
    assert str(sorter_path.resolve()) not in synchrony_sources
    assert str(sorter_path.resolve()) in spike_sources
    assert str(aligned_spike_path.resolve()) in spike_sources
    assert str(trial_table_path.resolve()) in power_sources
    assert str(trial_table_path.resolve()) in synchrony_sources
    assert str(trial_table_path.resolve()) in spike_sources


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_open_ephys_source_fingerprint_hashes_preprocessing_sidecar_content(
    tmp_path: Path,
    component: str,
) -> None:
    """A same-size, timestamp-restored sidecar edit stales all affected source identity."""
    config = _open_ephys_config_with_temporary_sidecar(tmp_path)
    lfp_path = Path(config.sites[0].lfp_path)
    sidecar_path = lfp_path.parent / "lfp_preprocessing.json"
    original_stat = sidecar_path.stat()
    original_digest = sha256(sidecar_path.read_bytes()).hexdigest()

    before = fingerprint_source_files(config, component=component)
    before_sidecar = before[str(sidecar_path.resolve())]
    lfp_entry = before[str(lfp_path.resolve())]
    sidecar = json.loads(sidecar_path.read_text(encoding="ascii"))
    sidecar["lfp_binary_scaling"]["gain_to_uV_by_channel"] = [0.196]
    sidecar_path.write_text(json.dumps(sidecar, separators=(",", ":")), encoding="ascii")
    os.utime(sidecar_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = fingerprint_source_files(config, component=component)
    after_sidecar = after[str(sidecar_path.resolve())]

    assert before_sidecar["path"] == str(sidecar_path.resolve())
    assert lfp_entry["value_semantics"] == "open_ephys_affine_uV_v1"
    assert before_sidecar["size_bytes"] == after_sidecar["size_bytes"]
    assert before_sidecar["mtime_ns"] == after_sidecar["mtime_ns"]
    assert before_sidecar["sha256"] == original_digest
    assert before != after
    assert before_sidecar["sha256"] != after_sidecar["sha256"]
    assert after_sidecar["sha256"] == sha256(sidecar_path.read_bytes()).hexdigest()


_PRE_NR0_OPEN_EPHYS_COMPONENT_FINGERPRINTS = {
    "power": "97d72e6bce66f44c89b5ddd11ce257fde21c5c68ac190adb92873750676f5a61",
    "synchrony": "5018dc585ccf8c1e3d2b559955f6a82005b2770c0dc76216d8dd5b23c0dec4db",
    "spike_phase": "856ecd69eeec22e44e2cca8c54fadfef55f516a3b5d2abb89138d65be5e19d3a",
}
_SPIKEGLX_COMPONENT_FINGERPRINTS = {
    "power": "c92331e34cc0252eacbc36b92c3beca0d2a99e5d35102325c88293c866da99b3",
    "synchrony": "28127adaf4318f33aa29aaa653b15ab950890fcb606eb4cd5fa5e4f75596ee43",
    "spike_phase": "3254279e2c37a4e238f5fe240524d2c9d914d1600af5adf0d934588009a1aab8",
}
_CORRECTED_OPEN_EPHYS_COMPONENT_FINGERPRINTS = {
    "power": "0e9b29b41f7ece65f6432fef3ad81acf7d5bc6dacd1fb4daa3528225daf7e1bb",
    "synchrony": "99bf31cd16fe74744e7b75526836eec07f14623aed7b6ad86d2772cc8a3e29e5",
    "spike_phase": "d2cfc02352b4792911b2754fde3ff8eac157eea71f66ac47aff7546490860d68",
}


def _fixed_open_ephys_identity_config() -> LFPSummaryConfig:
    """Return the path-stable pre-NR0 identity fixture with one Open Ephys site."""
    config = default_lfp_summary_config()
    first_site = replace(
        config.sites[0],
        acquisition_format="open_ephys",
        lfp_path=Path("pre_nr0_pfc_lfp.dat"),
        aligned_sync_path=Path("pre_nr0_pfc_sync.npz"),
        voltage_unit="uV",
        sample_rate_hz=2_500.0,
    )
    return replace(config, sites=(first_site,) + config.sites[1:])


def test_source_value_semantics_is_code_owned_and_open_ephys_scoped() -> None:
    """Live semantics are derived from acquisition metadata, never caller overrides."""
    open_ephys_config = _fixed_open_ephys_identity_config()
    spikeglx_config = default_lfp_summary_config()

    assert lfp_summary_models.source_value_semantics(open_ephys_config) == {
        "PFC": "open_ephys_affine_uV_v1"
    }
    assert lfp_summary_models.source_value_semantics(spikeglx_config) == {}


def _explicit_component_fingerprint_payload(
    component: str,
    config: LFPSummaryConfig,
    source_semantics: dict[str, str] | None,
) -> dict[str, object]:
    """Build the approved canonical fingerprint input without calling the production hasher."""
    canonical = json.loads(canonical_config_json(config))
    shared: dict[str, object] = {
        "schema_version": canonical["schema_version"],
        "session_id": canonical["session_id"],
        "session_path": canonical["session_path"],
        "trial_table_path": canonical["trial_table_path"],
        "sites": canonical["sites"],
        "trial_filter": canonical["trial_filter"],
        "windows": canonical["analysis_windows"],
    }
    if source_semantics:
        shared["source_value_semantics"] = source_semantics
    if component == "power":
        payload = {**shared, "power": canonical["power"]}
    elif component == "synchrony":
        phase = dict(canonical["phase"])
        payload = {
            **shared,
            "site_pairs": canonical["site_pairs"],
            "phase": phase,
        }
    elif component == "spike_phase":
        phase = dict(canonical["phase"])
        phase.pop("bootstrap_count")
        phase.pop("seed")
        payload = {
            **shared,
            "unit_population": canonical["unit_population"],
            "phase": phase,
            "ppc": canonical["ppc"],
        }
    else:
        raise AssertionError(f"unexpected component {component!r}")
    return payload


def _explicit_fingerprint_digest(payload: dict[str, object]) -> str:
    """Return the documented SHA-256 digest of one canonical JSON payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_open_ephys_component_identity_uses_only_the_resolved_semantics_payload_addition(
    component: str,
) -> None:
    """Every affected digest is the frozen canonical payload plus one OE semantics mapping."""
    config = _fixed_open_ephys_identity_config()
    legacy_payload = _explicit_component_fingerprint_payload(component, config, None)
    corrected_payload = _explicit_component_fingerprint_payload(
        component,
        config,
        {"PFC": "open_ephys_affine_uV_v1"},
    )

    assert set(corrected_payload) == set(legacy_payload) | {"source_value_semantics"}
    assert {
        key: value for key, value in corrected_payload.items() if key != "source_value_semantics"
    } == legacy_payload
    assert _explicit_fingerprint_digest(legacy_payload) == _PRE_NR0_OPEN_EPHYS_COMPONENT_FINGERPRINTS[component]
    assert _explicit_fingerprint_digest(corrected_payload) == _CORRECTED_OPEN_EPHYS_COMPONENT_FINGERPRINTS[component]
    assert component_fingerprint(component, config) == _CORRECTED_OPEN_EPHYS_COMPONENT_FINGERPRINTS[component]


@pytest.mark.parametrize("component", ("power", "synchrony", "spike_phase"))
def test_spikeglx_component_identity_retains_its_frozen_pre_nr0_digest(component: str) -> None:
    """Open Ephys provenance must not perturb an all-SpikeGLX component identity."""
    assert component_fingerprint(component, default_lfp_summary_config()) == (
        _SPIKEGLX_COMPONENT_FINGERPRINTS[component]
    )


def test_legacy_canonical_configuration_deserializes_without_value_semantics_field() -> None:
    """Adapter semantics stay in provenance so historical config JSON remains readable."""
    encoded = canonical_config_json(default_lfp_summary_config())

    decoded = lfp_summary_config_from_json(encoded)

    assert decoded == default_lfp_summary_config()
    assert "source_value_semantics" not in encoded
