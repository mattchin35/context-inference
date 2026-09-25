"""Contracts for adapting resolved session metadata to LFP-summary configuration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.neural_analysis.lfp_summary_session import (
    LFPSummarySessionRequest,
    build_active_unit_population,
    build_lfp_summary_config,
    build_metadata_spike_phase_config,
)
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata


def _write_resolved_session(tmp_path: Path):
    """Write and resolve a small mixed-acquisition metadata fixture."""
    open_ephys_directory = tmp_path / "ephys/front"
    spikeglx_directory = tmp_path / "ephys/rear"
    sorter_a = open_ephys_directory / "kilosort4"
    sorter_b = spikeglx_directory / "kilosort4"
    behavior = tmp_path / "behavior"
    for directory in (open_ephys_directory, spikeglx_directory, sorter_a, sorter_b, behavior):
        directory.mkdir(parents=True, exist_ok=True)
    for sorter in (sorter_a, sorter_b):
        (sorter / "spike_times.npy").write_bytes(b"spike-times")
        (sorter / "spike_clusters.npy").write_bytes(b"spike-clusters")
        (sorter / "cluster_info.tsv").write_text("cluster_id\tch\tgroup\n", encoding="ascii")
    (behavior / "trials.csv").write_text("choice_time\n", encoding="ascii")
    open_ephys_lfp = open_ephys_directory / "lfp.dat"
    open_ephys_lfp.write_bytes(np.array([[1.0]], dtype=np.float32).tobytes())
    (open_ephys_directory / "lfp_preprocessing.json").write_text(
        json.dumps(
            {
                "output_binary": "lfp.dat",
                "sampling_frequency_hz": 1250.0,
                "num_channels": 1,
                "num_segments": 1,
                "num_samples_by_segment": [1],
                "dtype": "float32",
                "binary_layout": "time_major_channel_interleaved",
                "channel_ids_in_binary_order": ["CH0"],
                "lfp_binary_scaling": {
                    "data_units": "unscaled_binary_values",
                    "has_scaleable_traces": True,
                    "channel_ids": ["CH0"],
                    "gain_to_uV_by_channel": [0.195],
                    "offset_to_uV_by_channel": [0.0],
                    "physical_unit_by_channel": ["uV"],
                    "export_scale_factor": 1.0,
                    "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
                },
            }
        ),
        encoding="ascii",
    )
    spikeglx_lfp = spikeglx_directory / "recording.lf.bin"
    spikeglx_lfp.write_bytes(b"")
    (spikeglx_directory / "recording.lf.meta").write_text(
        "typeThis=imec\nimSampRate=2500\n",
        encoding="ascii",
    )
    for name in ("front_sync.npz", "rear_sync.npz"):
        (tmp_path / "ephys" / name).write_bytes(b"sync")
    payload = {
        "schema_version": "1",
        "subject_id": "Mouse-Z",
        "session_id": "Mouse-Z_2030-01-02_030405",
        "session_date": "2030-01-02",
        "session_label": None,
        "behavior": {
            "session_directory": "behavior",
            "trial_table_file": "behavior/trials.csv",
            "event_table_file": None,
            "treadmill_file": None,
        },
        "probes": [
            {
                "probe_id": "front-probe",
                "display_label": "frontal hardware",
                "acquisition_family": "open_ephys",
                "lfp_file": "ephys/front/lfp.dat",
                "lfp_metadata_file": "ephys/front/lfp_preprocessing.json",
                "synchronization_file": "ephys/front_sync.npz",
                "sorter_directory": "ephys/front/kilosort4",
                "aligned_spike_file": "ephys/front_sync.npz",
                "channel_quality_file": None,
            },
            {
                "probe_id": "rear-probe",
                "display_label": "rear hardware",
                "acquisition_family": "spikeglx",
                "lfp_file": "ephys/rear/recording.lf.bin",
                "lfp_metadata_file": "ephys/rear/recording.lf.meta",
                "synchronization_file": "ephys/rear_sync.npz",
                "sorter_directory": "ephys/rear/kilosort4",
                "aligned_spike_file": "ephys/rear_sync.npz",
                "channel_quality_file": None,
            },
        ],
        "sites": [
            {
                "site_id": "frontal-site",
                "display_label": "PFC",
                "probe_id": "front-probe",
                "saved_channel_index": 0,
            },
            {
                "site_id": "rear-site",
                "display_label": "HPC",
                "probe_id": "rear-probe",
                "saved_channel_index": 7,
            },
        ],
        "site_pairs": [["frontal-site", "rear-site"]],
        "channel_groups": [
            {
                "channel_group_id": "rear-hpc",
                "display_label": "HPC",
                "probe_id": "rear-probe",
                "channel_indices": [7, 8, 9],
            }
        ],
        "populations": [
            {
                "population_id": "rear-active",
                "display_label": "active HPC units",
                "probe_id": "rear-probe",
                "channel_group_id": "rear-hpc",
            }
        ],
        "lfp_summary_cache_directory": "processed/summary-cache",
        "lfp_summary_snapshot_directory": None,
    }
    metadata_path = tmp_path / "neural_session.json"
    metadata_path.write_text(json.dumps(payload), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)


def test_build_config_uses_metadata_sources_labels_and_authoritative_rates(tmp_path: Path) -> None:
    """Mixed Open Ephys/SpikeGLX sites retain metadata identities and physical units."""
    session = _write_resolved_session(tmp_path)

    config = build_lfp_summary_config(session, LFPSummarySessionRequest())

    assert config.session_id == "Mouse-Z_2030-01-02_030405"
    assert config.session_path == tmp_path.resolve()
    assert config.output_directory == (tmp_path / "processed/summary-cache").resolve()
    assert config.trial_table_path == (tmp_path / "behavior/trials.csv").resolve()
    assert [(site.stable_id, site.label) for site in config.sites] == [
        ("frontal-site", "PFC"),
        ("rear-site", "HPC"),
    ]
    assert [site.probe_label for site in config.sites] == ["front-probe", "rear-probe"]
    assert [site.acquisition_format for site in config.sites] == ["open_ephys", "spikeglx"]
    assert [site.sample_rate_hz for site in config.sites] == [1250.0, 2500.0]
    assert [site.voltage_unit for site in config.sites] == ["uV", "uV"]
    assert config.site_pairs == (("frontal-site", "rear-site"),)
    assert config.unit_population is None


def test_selected_population_uses_metadata_group_and_exact_probe_paths(tmp_path: Path) -> None:
    """Population requests preserve anatomical channels without deriving unit IDs."""
    session = _write_resolved_session(tmp_path)

    config = build_lfp_summary_config(
        session,
        LFPSummarySessionRequest(population_id="rear-active"),
    )

    population = config.unit_population
    assert population is not None
    assert population.label == "active HPC units"
    assert population.probe_label == "rear-probe"
    assert population.sorter_path == (tmp_path / "ephys/rear/kilosort4").resolve()
    assert population.aligned_spike_path == (tmp_path / "ephys/rear_sync.npz").resolve()
    assert population.selected_channels == (7, 8, 9)
    assert population.quality_settings == ()
    assert population.stable_unit_ids == ()


def test_request_output_directory_overrides_optional_metadata_cache(tmp_path: Path) -> None:
    """An explicit request chooses the live cache without mutating session metadata."""
    session = _write_resolved_session(tmp_path)
    requested_output = tmp_path / "another-cache"

    config = build_lfp_summary_config(
        session,
        LFPSummarySessionRequest(output_directory=requested_output),
    )

    assert config.output_directory == requested_output
    assert session.lfp_summary_cache_directory != requested_output


def test_unknown_population_is_rejected_clearly(tmp_path: Path) -> None:
    """Population selection never silently falls back to a CT-specific default."""
    session = _write_resolved_session(tmp_path)

    with pytest.raises(ValueError, match="unknown population.*not-present"):
        build_lfp_summary_config(
            session,
            LFPSummarySessionRequest(population_id="not-present"),
        )


def test_authoritative_sidecar_path_must_match_the_current_loader(tmp_path: Path) -> None:
    """A declared sidecar is checked against the selected acquisition adapter."""
    session = _write_resolved_session(tmp_path)
    declared = tmp_path / "ephys/front/lfp_preprocessing.json"
    renamed = tmp_path / "ephys/front/renamed.json"
    declared.rename(renamed)
    session = replace(
        session,
        probes=(replace(session.probes[0], lfp_metadata_file=renamed),) + session.probes[1:],
    )

    with pytest.raises(ValueError, match="lfp_metadata_file"):
        build_lfp_summary_config(session, LFPSummarySessionRequest())


def test_config_build_reads_metadata_but_never_loads_scientific_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configuration construction reads small sidecars without calling NumPy loaders."""
    session = _write_resolved_session(tmp_path)

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("configuration construction must not call numpy.load")

    monkeypatch.setattr(np, "load", forbidden_load)

    assert build_lfp_summary_config(session, LFPSummarySessionRequest()).sites


def test_active_population_uses_metadata_paths_and_existing_quality_rules(
    tmp_path: Path,
) -> None:
    """Launcher population selection is generic and records every consumed source."""
    session = _write_resolved_session(tmp_path)
    probe = session.probes[1]
    quality_path = tmp_path / "ephys/rear/channel_quality.csv"
    quality_path.write_text(
        "channel_id,label,inside_brain\nCH7,good,true\nCH8,bad,true\nCH9,good,true\n",
        encoding="ascii",
    )
    session = replace(
        session,
        probes=(session.probes[0], replace(probe, channel_quality_file=quality_path)),
    )
    clusters = __import__("pandas").DataFrame(
        {
            "cluster_id": [4, 5, 8],
            "ch": [7, 8, 9],
            "group": ["good", "good", "mua"],
        }
    )

    population = build_active_unit_population(
        session,
        "rear-probe",
        cluster_metadata_loader=lambda _path: clusters,
        channel_metadata_loader=lambda path: __import__("pandas").read_csv(path),
    )

    assert population.stable_unit_ids == ("rear-probe:4", "rear-probe:8")
    assert population.selected_channels == (7, 9)
    assert population.spike_times_path == probe.sorter_directory / "spike_times.npy"
    assert population.spike_clusters_path == probe.sorter_directory / "spike_clusters.npy"
    assert population.cluster_info_path == probe.sorter_directory / "cluster_info.tsv"
    assert population.channel_quality_path == quality_path


def test_metadata_population_requires_one_declared_population_for_probe(
    tmp_path: Path,
) -> None:
    """Probe selection fails clearly rather than silently choosing a population."""
    session = _write_resolved_session(tmp_path)

    with pytest.raises(ValueError, match="no population.*front-probe"):
        build_active_unit_population(
            session,
            "front-probe",
            cluster_metadata_loader=lambda _path: None,
            channel_metadata_loader=lambda _path: None,
        )


def test_metadata_spike_phase_config_keeps_user_choices_explicit(tmp_path: Path) -> None:
    """The launcher adapter changes only population, cache, and execution counts."""
    session = _write_resolved_session(tmp_path)
    probe = session.probes[1]
    quality_path = tmp_path / "ephys/rear/channel_quality.csv"
    quality_path.write_text(
        "channel_id,label,inside_brain\nCH7,good,true\nCH8,bad,true\nCH9,good,true\n",
        encoding="ascii",
    )
    session = replace(
        session,
        probes=(session.probes[0], replace(probe, channel_quality_file=quality_path)),
    )
    clusters = __import__("pandas").DataFrame(
        {"cluster_id": [4, 8], "ch": [7, 9], "group": ["good", "mua"]}
    )
    cache = tmp_path / "processed/new-cache"

    config = build_metadata_spike_phase_config(
        session,
        probe_id="rear-probe",
        cache_directory=cache,
        shuffle_count=100,
        worker_count=3,
        cluster_metadata_loader=lambda _path: clusters,
        channel_metadata_loader=lambda path: __import__("pandas").read_csv(path),
    )

    assert config.output_directory == cache
    assert config.ppc.shuffle_count == 100
    assert config.ppc_execution.worker_count == 3
    assert config.ppc_execution.checkpoint_enabled is True
    assert config.ppc_execution.checkpoint_retention == "incomplete_only"
    assert config.unit_population is not None
    assert config.unit_population.stable_unit_ids == ("rear-probe:4", "rear-probe:8")
