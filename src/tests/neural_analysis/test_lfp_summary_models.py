"""Contract tests for immutable LFP-summary configuration models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable

import pytest

from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    canonical_config_json,
    component_fingerprint,
    default_lfp_summary_config,
    fingerprint_source_files,
    lfp_summary_config_from_json,
    validate_lfp_summary_config,
)


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
        power=replace(config.power, baseline_start_s=config.power.baseline_start_s - 0.1),
    )
    changed_phase = replace(
        config,
        phase=replace(config.phase, bootstrap_count=config.phase.bootstrap_count + 1),
    )

    assert component_fingerprint("power", config) == component_fingerprint("power", changed_ppc)
    assert component_fingerprint("synchrony", config) == component_fingerprint("synchrony", changed_ppc)
    assert component_fingerprint("spike_phase", config) != component_fingerprint("spike_phase", changed_ppc)
    assert component_fingerprint("power", config) != component_fingerprint("power", changed_power)
    assert component_fingerprint("synchrony", config) == component_fingerprint("synchrony", changed_power)
    assert component_fingerprint("spike_phase", config) == component_fingerprint("spike_phase", changed_power)
    assert component_fingerprint("power", config) == component_fingerprint("power", changed_phase)
    assert component_fingerprint("synchrony", config) != component_fingerprint("synchrony", changed_phase)
    assert component_fingerprint("spike_phase", config) != component_fingerprint("spike_phase", changed_phase)


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
