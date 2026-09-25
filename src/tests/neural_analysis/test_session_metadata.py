"""Contracts for editable neural-session metadata and source resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.neural_analysis.session_metadata import (
    SessionMetadata,
    load_session_metadata,
    resolve_probe_sources,
    resolve_session_metadata,
    validate_session_for_action,
)


def _metadata_payload() -> dict[str, Any]:
    """Return complete synthetic metadata using session-relative paths."""
    return {
        "schema_version": "1",
        "subject_id": "Mouse-Z",
        "session_id": "Mouse-Z_2030-01-02_030405",
        "session_date": "2030-01-02",
        "session_label": "reversal learning",
        "behavior": {
            "session_directory": "behavior",
            "trial_table_file": "behavior/trials.csv",
            "event_table_file": "behavior/events.csv",
            "treadmill_file": None,
        },
        "probes": [
            {
                "probe_id": "front-probe",
                "display_label": "frontal probe",
                "acquisition_family": "open_ephys",
                "lfp_file": "ephys/front/lfp.dat",
                "lfp_metadata_file": "ephys/front/lfp_preprocessing.json",
                "synchronization_file": "ephys/aligned/front_sync.npz",
                "sorter_directory": "ephys/front/kilosort4",
                "aligned_spike_file": "ephys/aligned/front_sync.npz",
                "channel_quality_file": "ephys/front/channel_quality.csv",
            },
            {
                "probe_id": "rear-probe",
                "display_label": "rear probe",
                "acquisition_family": "spikeglx",
                "lfp_file": "ephys/rear/probe.lf.bin",
                "lfp_metadata_file": "ephys/rear/probe.lf.meta",
                "synchronization_file": "ephys/aligned/rear_sync.npz",
                "sorter_directory": "ephys/rear/kilosort4",
                "aligned_spike_file": "ephys/aligned/rear_sync.npz",
                "channel_quality_file": None,
            },
        ],
        "sites": [
            {
                "site_id": "PFC",
                "display_label": "prefrontal cortex",
                "probe_id": "front-probe",
                "saved_channel_index": 5,
            },
            {
                "site_id": "HPC",
                "display_label": "hippocampus",
                "probe_id": "rear-probe",
                "saved_channel_index": 12,
            },
        ],
        "site_pairs": [["PFC", "HPC"]],
        "channel_groups": [
            {
                "channel_group_id": "front-cortex",
                "display_label": "frontal cortex",
                "probe_id": "front-probe",
                "channel_indices": [2, 4, 6],
            },
            {
                "channel_group_id": "hippocampus",
                "display_label": "hippocampus",
                "probe_id": "rear-probe",
                "channel_indices": [10, 11, 12],
            },
        ],
        "populations": [
            {
                "population_id": "front-active",
                "display_label": "active frontal units",
                "probe_id": "front-probe",
                "channel_group_id": "front-cortex",
            },
            {
                "population_id": "hpc-active",
                "display_label": "active hippocampal units",
                "probe_id": "rear-probe",
                "channel_group_id": "hippocampus",
            },
        ],
        "lfp_summary_cache_directory": "processed/lfp-summary-cache",
        "lfp_summary_snapshot_directory": None,
    }


def _write_metadata(session_root: Path, payload: dict[str, Any]) -> Path:
    """Write one canonical metadata file and return its path."""
    session_root.mkdir(parents=True, exist_ok=True)
    metadata_path = session_root / "neural_session.json"
    metadata_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return metadata_path


def _make_sources(session_root: Path, payload: dict[str, Any]) -> None:
    """Create declared synthetic files and directories without scientific data."""
    behavior = payload["behavior"]
    directory_fields = [behavior["session_directory"]]
    file_fields = [
        behavior["trial_table_file"],
        behavior["event_table_file"],
        behavior["treadmill_file"],
    ]
    for probe in payload["probes"]:
        directory_fields.append(probe["sorter_directory"])
        file_fields.extend(
            [
                probe["lfp_file"],
                probe["lfp_metadata_file"],
                probe["synchronization_file"],
                probe["aligned_spike_file"],
                probe["channel_quality_file"],
            ]
        )
    directory_fields.extend(
        [
            payload["lfp_summary_cache_directory"],
            payload["lfp_summary_snapshot_directory"],
        ]
    )
    for relative_path in directory_fields:
        if relative_path:
            (session_root / relative_path).mkdir(parents=True, exist_ok=True)
    for relative_path in file_fields:
        if relative_path:
            path = session_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic\n", encoding="ascii")


def test_complete_metadata_loads_and_resolves_without_subject_specific_logic(
    tmp_path: Path,
) -> None:
    """A differently named two-probe session resolves every declared source."""
    payload = _metadata_payload()
    session_root = tmp_path / "unfamiliar-mouse" / "odd-session-name"
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)

    metadata = load_session_metadata(metadata_path)
    resolved = resolve_session_metadata(metadata, metadata_path)

    assert isinstance(metadata, SessionMetadata)
    assert metadata.subject_id == "Mouse-Z"
    assert resolved.session_root == session_root.resolve()
    assert resolved.behavior.trial_table_file == (session_root / "behavior/trials.csv").resolve()
    assert [probe.probe_id for probe in resolved.probes] == ["front-probe", "rear-probe"]
    assert resolve_probe_sources(resolved, "front-probe").lfp_file.name == "lfp.dat"
    assert resolve_probe_sources(resolved, "rear-probe").lfp_metadata_file.name == "probe.lf.meta"
    assert resolved.channel_groups[0].channel_indices == (2, 4, 6)


@pytest.mark.parametrize(
    ("family", "lfp_name", "metadata_name"),
    [
        ("open_ephys", "lfp.dat", "lfp_preprocessing.json"),
        ("spikeglx", "recording.lf.bin", "recording.lf.meta"),
    ],
)
def test_observed_acquisition_families_use_explicit_metadata_sidecars(
    tmp_path: Path,
    family: str,
    lfp_name: str,
    metadata_name: str,
) -> None:
    """Open Ephys and SpikeGLX are selected explicitly rather than guessed."""
    payload = _metadata_payload()
    payload["probes"] = [payload["probes"][0]]
    payload["probes"][0]["acquisition_family"] = family
    payload["probes"][0]["lfp_file"] = f"ephys/source/{lfp_name}"
    payload["probes"][0]["lfp_metadata_file"] = f"ephys/source/{metadata_name}"
    payload["sites"] = [payload["sites"][0]]
    payload["site_pairs"] = []
    payload["channel_groups"] = [payload["channel_groups"][0]]
    payload["populations"] = [payload["populations"][0]]
    session_root = tmp_path / family
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)

    probe = resolve_probe_sources(
        resolve_session_metadata(load_session_metadata(metadata_path), metadata_path),
        "front-probe",
    )

    assert probe.acquisition_family == family
    assert probe.lfp_file.name == lfp_name
    assert probe.lfp_metadata_file.name == metadata_name


def test_metadata_is_immutable_and_loading_is_deterministic(tmp_path: Path) -> None:
    """The same JSON produces equal immutable records with stable tuple ordering."""
    metadata_path = _write_metadata(tmp_path, _metadata_payload())

    first = load_session_metadata(metadata_path)
    second = load_session_metadata(metadata_path)

    assert first == second
    assert first.site_pairs == (("PFC", "HPC"),)
    with pytest.raises((AttributeError, TypeError)):
        first.subject_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="2"), "schema_version"),
        (lambda value: value.update(unexpected="value"), "unexpected"),
        (lambda value: value["probes"][0].update(unexpected="value"), "unexpected"),
        (lambda value: value["probes"][0].update(acquisition_family="other"), "acquisition_family"),
    ],
)
def test_unknown_versions_fields_and_acquisition_families_are_rejected(
    tmp_path: Path,
    mutation: Any,
    message: str,
) -> None:
    """Version 1 is exact and does not silently accept unknown meanings."""
    payload = _metadata_payload()
    mutation(payload)
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        load_session_metadata(metadata_path)


@pytest.mark.parametrize(
    "relative_path",
    ["/absolute/lfp.dat", "../outside/lfp.dat", "ephys/../../outside/lfp.dat"],
)
def test_source_paths_must_be_relative_and_contained(
    tmp_path: Path,
    relative_path: str,
) -> None:
    """Ordinary absolute and traversal paths cannot escape the session root."""
    payload = _metadata_payload()
    payload["probes"][0]["lfp_file"] = relative_path
    metadata_path = _write_metadata(tmp_path / "session", payload)
    metadata = load_session_metadata(metadata_path)

    with pytest.raises(ValueError, match="relative|session root|contained"):
        resolve_session_metadata(metadata, metadata_path)


def test_symlinked_source_cannot_escape_the_session_root(tmp_path: Path) -> None:
    """A declared in-session symlink may not resolve to an external source."""
    session_root = tmp_path / "session"
    outside_file = tmp_path / "outside.dat"
    outside_file.write_text("outside\n", encoding="ascii")
    payload = _metadata_payload()
    payload["probes"][0]["lfp_file"] = "ephys/linked.dat"
    metadata_path = _write_metadata(session_root, payload)
    (session_root / "ephys").mkdir()
    (session_root / "ephys/linked.dat").symlink_to(outside_file)

    with pytest.raises(ValueError, match="session root|contained"):
        resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)


def test_existing_sources_must_have_the_declared_file_or_directory_kind(
    tmp_path: Path,
) -> None:
    """An existing sorter file is rejected because sorter_directory is a directory."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    metadata_path = _write_metadata(session_root, payload)
    sorter_path = session_root / payload["probes"][0]["sorter_directory"]
    sorter_path.parent.mkdir(parents=True)
    sorter_path.write_text("not a directory\n", encoding="ascii")

    with pytest.raises(ValueError, match="sorter_directory.*directory"):
        resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["probes"].append(dict(value["probes"][0])),
        lambda value: value["sites"].append(dict(value["sites"][0])),
        lambda value: value["channel_groups"].append(dict(value["channel_groups"][0])),
        lambda value: value["populations"].append(dict(value["populations"][0])),
        lambda value: value["sites"][0].update(probe_id="missing-probe"),
        lambda value: value["site_pairs"][0].__setitem__(1, "missing-site"),
        lambda value: value["populations"][0].update(channel_group_id="missing-group"),
        lambda value: value["populations"][0].update(probe_id="rear-probe"),
    ],
)
def test_duplicate_ids_and_broken_references_are_rejected(
    tmp_path: Path,
    mutation: Any,
) -> None:
    """Every cross-reference is explicit, unique, and probe-consistent."""
    payload = _metadata_payload()
    mutation(payload)
    metadata_path = _write_metadata(tmp_path, payload)

    with pytest.raises(ValueError):
        load_session_metadata(metadata_path)


def test_missing_null_and_empty_values_remain_distinct(tmp_path: Path) -> None:
    """Absent fields are malformed while null and empty optional values are preserved."""
    payload = _metadata_payload()
    payload["behavior"]["treadmill_file"] = ""
    payload["probes"][0]["channel_quality_file"] = None
    metadata_path = _write_metadata(tmp_path / "present", payload)

    loaded = load_session_metadata(metadata_path)

    assert loaded.behavior.treadmill_file == ""
    assert loaded.probes[0].channel_quality_file is None

    del payload["behavior"]["treadmill_file"]
    missing_path = _write_metadata(tmp_path / "missing", payload)
    with pytest.raises(ValueError, match="treadmill_file"):
        load_session_metadata(missing_path)


def test_action_validation_reports_only_inputs_needed_by_each_action(
    tmp_path: Path,
) -> None:
    """Incomplete sources remain inspectable and produce concise action-specific gaps."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    metadata_path = _write_metadata(session_root, payload)
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)

    webapp = validate_session_for_action(resolved, "webapp")
    power = validate_session_for_action(resolved, "power")
    synchrony = validate_session_for_action(resolved, "synchrony")
    spike_phase = validate_session_for_action(resolved, "spike-phase")

    assert webapp.available is False
    assert "behavior.trial_table_file" in webapp.missing_inputs
    assert "probes[front-probe].lfp_file" in power.missing_inputs
    assert "site_pairs" not in power.missing_inputs
    assert "site_pairs" not in synchrony.missing_inputs
    assert "probes[front-probe].sorter_directory" in spike_phase.missing_inputs
    assert "probes[front-probe].channel_quality_file" not in spike_phase.missing_inputs


def test_complete_sources_make_all_actions_available(tmp_path: Path) -> None:
    """Complete ordinary metadata enables the four existing action categories."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)

    results = {
        action: validate_session_for_action(resolved, action)
        for action in ("webapp", "power", "synchrony", "spike-phase")
    }

    assert all(result.available for result in results.values())
    assert all(result.missing_inputs == () for result in results.values())


def test_metadata_validation_never_loads_scientific_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loading, resolving, and action validation use paths and small JSON only."""
    payload = _metadata_payload()
    session_root = tmp_path / "session"
    _make_sources(session_root, payload)
    metadata_path = _write_metadata(session_root, payload)

    def forbidden_array_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("metadata validation must not call numpy.load")

    monkeypatch.setattr(np, "load", forbidden_array_load)
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)

    assert validate_session_for_action(resolved, "spike-phase").available is True


def test_resolver_requires_the_canonical_metadata_filename(tmp_path: Path) -> None:
    """The production metadata boundary is the session-root neural_session.json file."""
    metadata_path = tmp_path / "renamed.json"
    metadata_path.write_text(json.dumps(_metadata_payload()), encoding="ascii")

    with pytest.raises(ValueError, match="neural_session.json"):
        resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)


def test_unknown_probe_lookup_has_a_clear_error(tmp_path: Path) -> None:
    """Probe lookup reports the requested stable ID rather than falling back."""
    metadata_path = _write_metadata(tmp_path, _metadata_payload())
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)

    with pytest.raises(KeyError, match="unknown-probe"):
        resolve_probe_sources(resolved, "unknown-probe")
