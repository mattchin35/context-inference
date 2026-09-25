"""Contracts for the compact, probe-centered neural-session metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from src.neural_analysis.session_metadata import (
    SessionMetadata,
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
    validate_session_for_action,
)


def _metadata_payload(
    *, acquisition: str = "open_ephys", restricted_channels: bool = True
) -> dict[str, Any]:
    """Return one complete version-2 metadata document."""
    probes: dict[str, Any] = {
        "ProbeA": {
            "lfp": "ephys/ProbeA/lfp.dat",
            "alignment": "ephys/ProbeA/alignment.npz",
            "sorter": "ephys/ProbeA/kilosort4",
            "quality": "ephys/ProbeA/channel_quality.csv",
            "sites": {"PFC": 5},
        },
        "ProbeB": {
            "lfp": "ephys/ProbeB/lfp.dat",
            "alignment": "ephys/ProbeB/alignment.npz",
            "sorter": "ephys/ProbeB/kilosort4",
            "quality": "ephys/ProbeB/channel_quality.csv",
            "sites": {"HPC": 12},
        },
    }
    if restricted_channels:
        probes["ProbeB"]["unit_channels"] = [10, 11, 12]
    if acquisition == "spikeglx":
        probes["ProbeA"]["lfp"] = "ephys/ProbeA/recording.lf.bin"
        probes["ProbeB"]["lfp"] = "ephys/ProbeB/recording.lf.bin"
    return {
        "schema_version": "2",
        "session": "Mouse-Z_2030-01-02_030405",
        "acquisition": acquisition,
        "behavior": {
            "trials": "behavior/trials.csv",
            "events": "behavior/events.csv",
        },
        "probes": probes,
        "site_pairs": [["PFC", "HPC"]],
        "cache": "processed/lfp-summary-cache",
    }


def _write_metadata(session_root: Path, payload: dict[str, Any]) -> Path:
    """Write one canonical metadata file and return its path."""
    session_root.mkdir(parents=True, exist_ok=True)
    metadata_path = session_root / "neural_session.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    return metadata_path


def _make_sources(session_root: Path, payload: dict[str, Any]) -> None:
    """Create the declared small files and fixed sorter children."""
    file_names = [payload["behavior"]["trials"], payload["behavior"].get("events")]
    for probe in payload["probes"].values():
        file_names.extend([probe["lfp"], probe["alignment"], probe["quality"]])
        sorter = session_root / probe["sorter"]
        sorter.mkdir(parents=True, exist_ok=True)
        for child in ("spike_times.npy", "spike_clusters.npy", "cluster_info.tsv"):
            (sorter / child).write_text("synthetic\n", encoding="ascii")
        lfp = session_root / probe["lfp"]
        sidecar = (
            lfp.with_suffix(".meta")
            if payload["acquisition"] == "spikeglx"
            else lfp.parent / "lfp_preprocessing.json"
        )
        file_names.append(str(sidecar.relative_to(session_root)))
    for relative_name in file_names:
        if relative_name is None:
            continue
        path = session_root / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic\n", encoding="ascii")
    cache = payload.get("cache")
    if cache is not None:
        (session_root / cache).mkdir(parents=True, exist_ok=True)


def test_compact_metadata_resolves_each_probe_as_one_complete_unit(tmp_path: Path) -> None:
    """Probe paths, sites, and optional channel restriction stay together."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)

    metadata = load_session_metadata(metadata_path)
    resolved = resolve_session_metadata(metadata, metadata_path)
    probe_a = resolve_probe_sources(resolved, "ProbeA")
    probe_b = resolve_probe_sources(resolved, "ProbeB")

    assert isinstance(metadata, SessionMetadata)
    assert metadata.session == "Mouse-Z_2030-01-02_030405"
    assert resolved.acquisition == "open_ephys"
    assert probe_a.sites[0].site_id == "PFC"
    assert probe_a.lfp_metadata_file.name == "lfp_preprocessing.json"
    assert probe_a.synchronization_file == probe_a.aligned_spike_file
    assert probe_a.unit_channels is None
    assert probe_b.unit_channels == (10, 11, 12)
    assert resolved.cache_directory == (session_root / payload["cache"]).resolve()


@pytest.mark.parametrize(
    ("family", "sidecar_name"),
    [("open_ephys", "lfp_preprocessing.json"), ("spikeglx", "recording.lf.meta")],
)
def test_one_session_acquisition_family_infers_lfp_sidecars(
    tmp_path: Path, family: str, sidecar_name: str
) -> None:
    """The sole acquisition label determines every probe sidecar path."""
    payload = _metadata_payload(acquisition=family)
    metadata_path = _write_metadata(tmp_path, payload)
    probe = resolve_probe_sources(
        resolve_session_metadata(load_session_metadata(metadata_path), metadata_path),
        "ProbeA",
    )
    assert probe.acquisition_family == family
    assert probe.lfp_metadata_file.name == sidecar_name


def test_optional_fields_can_be_omitted(tmp_path: Path) -> None:
    """Events, cache, and channel restrictions are genuinely optional."""
    payload = _metadata_payload(restricted_channels=False)
    del payload["behavior"]["events"]
    del payload["cache"]
    metadata_path = _write_metadata(tmp_path, payload)

    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)

    assert resolved.behavior.event_table_file is None
    assert resolved.cache_directory is None
    assert all(probe.unit_channels is None for probe in resolved.probes)


def test_version_one_is_rejected_with_a_replacement_message(tmp_path: Path) -> None:
    """The simpler schema replaces rather than coexists with version 1."""
    payload = _metadata_payload()
    payload["schema_version"] = "1"
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError, match="schema_version.*2|replace"):
        load_session_metadata(metadata_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(unexpected="value"),
        lambda value: value["probes"]["ProbeA"].update(display_label="duplicate"),
        lambda value: value.update(acquisition="other"),
        lambda value: value["probes"]["ProbeA"].update(unit_channels=[1, 1]),
        lambda value: value["probes"]["ProbeB"]["sites"].update(PFC=7),
        lambda value: value.update(site_pairs=[["PFC", "missing"]]),
    ],
)
def test_unknown_fields_and_inconsistent_identifiers_are_rejected(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    """The compact document rejects redundant or ambiguous metadata."""
    payload = _metadata_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        load_session_metadata(_write_metadata(tmp_path, payload))


@pytest.mark.parametrize(
    "relative_path", ["/absolute/lfp.dat", "../outside/lfp.dat", "ephys/../../outside.dat"]
)
def test_source_paths_must_remain_inside_the_session(
    tmp_path: Path, relative_path: str
) -> None:
    """Editable paths cannot escape the session root."""
    payload = _metadata_payload()
    payload["probes"]["ProbeA"]["lfp"] = relative_path
    metadata_path = _write_metadata(tmp_path / "session", payload)
    with pytest.raises(ValueError, match="relative|contained|session root"):
        resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)


def test_complete_sources_make_all_actions_available(tmp_path: Path) -> None:
    """The compact document supplies all four existing workflows."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)

    results = [
        validate_session_for_action(resolved, action)
        for action in ("webapp", "power", "synchrony", "spike-phase")
    ]
    assert all(result.available for result in results)


def test_validation_does_not_load_scientific_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata inspection uses paths and small text records only."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)

    def forbidden_array_load(*args: object, **kwargs: object) -> None:
        """Fail if metadata inspection attempts to load a numerical array."""
        raise AssertionError("metadata validation must not call numpy.load")

    monkeypatch.setattr(np, "load", forbidden_array_load)
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    assert validate_session_for_action(resolved, "spike-phase").available


def test_canonical_filename_and_exact_probe_lookup_are_required(tmp_path: Path) -> None:
    """One conventional filename and dictionary probe key define lookup."""
    renamed = tmp_path / "renamed.json"
    renamed.write_text(json.dumps(_metadata_payload()), encoding="ascii")
    with pytest.raises(ValueError, match="neural_session.json"):
        resolve_session_metadata(load_session_metadata(renamed), renamed)

    canonical = _write_metadata(tmp_path / "canonical", _metadata_payload())
    resolved = resolve_session_metadata(load_session_metadata(canonical), canonical)
    with pytest.raises(KeyError, match="unknown"):
        resolve_probe_sources(resolved, "unknown")
