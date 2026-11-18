"""Immutable named-array products and portable exact fixed-PCA references."""

import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.event_sequences import io as event_io
from src.neural_analysis.event_sequences.run_session import EventSequenceRequest, prepare_event_sequences
from src.neural_analysis.event_sequences.selection import CENTRAL_CONDITIONS, TRANSITION_CONDITIONS
from src.neural_analysis.population import session_timeline_io as timeline_io


def _forbid_current_sources(*args: object, **kwargs: object) -> None:
    """Input: arbitrary current-source arguments. Raise if saved-only display reads or computes live inputs."""
    raise AssertionError("Saved event views must not inspect current source inputs")


def _rewrite_archive(path: Path, arrays: dict) -> None:
    """Input: synthetic NPZ Path/named arrays. Return None; replace only this fixture's archive for corruption tests."""
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("kind", ["unit", "pc"])
def test_saved_schema_preserves_all_conditions_maps_counts_and_metadata(event_session, kind: str) -> None:
    """Input: session/source kind. Return None; inspect named arrays and full source/session identity contracts."""
    request = EventSequenceRequest(
        event_session.metadata_path, kind,
        **({"unit_id": "A:7"} if kind == "unit" else {"region": "PFC"}),
    )
    report = prepare_event_sequences(request)
    packet = event_io.load_event_sequences(report["result_path"])
    assert len(packet["trials"]) == 9
    assert {"row_position", "trial_id", "index_label"}.issubset(packet["trials"].columns)
    np.testing.assert_array_equal(packet["trials"].row_position, np.arange(9))
    np.testing.assert_array_equal(packet["trials"].trial_id, np.arange(101, 110))
    np.testing.assert_array_equal(packet["activity_row_positions"], np.arange(1, 8))
    np.testing.assert_array_equal(packet["full_to_activity"], [-1, 0, 1, 2, 3, 4, 5, 6, -1])
    assert list(packet["central_matches"]) == list(CENTRAL_CONDITIONS)
    assert list(packet["transition_matches"]) == list(TRANSITION_CONDITIONS)
    assert len(packet["central_summary"]) == 24 and len(packet["transition_summary"]) == 8
    for table in (packet["central_summary"], packet["transition_summary"]):
        assert {"behavioral_count", "excluded_count", "n"}.issubset(table.columns)
        np.testing.assert_array_equal(table.behavioral_count, table.excluded_count + table.n)
    np.testing.assert_array_equal(packet["transition_matches"][("left_to_right", 1)], [[1, 2, 3]])
    meta = packet["meta"]
    assert meta["schema_version"] == event_io.SCHEMA_VERSION
    assert meta["analysis_version"] == event_io.ANALYSIS_VERSION
    assert meta["configuration_id"] == report["configuration_id"]
    assert meta["generator"] == "src.neural_analysis.event_sequences.run_session.prepare_event_sequences"
    assert meta["session_id"] == event_session.session_id
    assert meta["random_seed"] is None and meta["code_revision"]
    assert meta["parameters"]["missing_label_policy"] == "exclude"
    assert meta["units"]["absolute_time"] == "UTC Unix seconds"
    assert meta["units"]["unit_rates_hz"] == "Hz"
    assert not meta["transition_unavailable_reason"]
    with np.load(report["result_path"], allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
        for key, positions in packet["central_matches"].items():
            np.testing.assert_array_equal(archive["central__" + "__".join(key)], positions)
        for (direction, repeats), positions in packet["transition_matches"].items():
            saved = archive[f"transition__{direction}__{repeats}"]
            assert saved.shape == positions.shape == (len(positions), repeats + 2)
        for name in ("trials", "bin_edges_s", "activity_row_positions", "full_to_activity", "central_summary", "transition_summary"):
            assert name in archive.files
        assert "session_root" not in archive.files and "timeline_result" not in archive.files
        if kind == "unit":
            assert archive["unit_rates_hz"].shape == (7, 8)
        else:
            assert "unit_rates_hz" not in archive.files
            assert "shared_choices" in archive.files and "regional_choice_positions" in archive.files
            assert not any("scores" in name or "components" in name for name in archive.files)


def test_unit_round_trip_keeps_nontrivial_original_labels_and_never_overwrites(event_session) -> None:
    """Input: session. Return None; named (N,) source labels survive new NPZ without modifying published bytes."""
    report = prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:7"))
    packet = event_io.load_event_sequences(report["result_path"])
    labels = ["duplicate", "duplicate"] + [f"label-{index}" for index in range(2, 9)]
    packet["trials"]["index_label"] = labels
    target = event_session.session_root / "analysis_runs/event_sequence_heatmaps_roundtrip/results.npz"
    event_io.save_event_sequences(target, packet)
    loaded = event_io.load_event_sequences(target)
    assert loaded["trials"].index_label.tolist() == labels
    np.testing.assert_array_equal(loaded["unit_rates_hz"], packet["unit_rates_hz"])
    pd.testing.assert_frame_equal(loaded["central_summary"], packet["central_summary"])
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        event_io.save_event_sequences(target, packet)
    assert target.read_bytes() == before


@pytest.mark.parametrize("kind", ["unit", "pc"])
def test_saved_display_and_discovery_ignore_missing_current_raw_inputs(event_session, monkeypatch, kind: str) -> None:
    """Input: session/patch fixture/source kind. Return None; persisted labels/activity serve views without current CSVs."""
    request = EventSequenceRequest(event_session.metadata_path, kind, **({"unit_id": "A:7"} if kind == "unit" else {"region": "PFC"}))
    report = prepare_event_sequences(request)
    for filename in ("trials.csv", "events.csv", "A/cluster_info.tsv", "A/channel_quality.csv", "A/aligned.npz", "A/spike_clusters.npy"):
        (event_session.session_root / filename).unlink()
    metadata = json.loads(event_session.metadata_path.read_text())
    metadata["analysis"]["probe_roles"].update(pfc="B", hpc="A")
    metadata["probes"]["A"]["alignment"] = "A/missing-current-alignment.npz"
    event_session.metadata_path.write_text(json.dumps(metadata))
    monkeypatch.setattr(timeline_io, "timeline_configuration", _forbid_current_sources)
    monkeypatch.setattr(timeline_io, "load_timeline_behavior", _forbid_current_sources)
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_current_sources)
    monkeypatch.setattr(event_io, "load_probe_spike_arrays", _forbid_current_sources)
    reader = Mock(side_effect=_forbid_current_sources)
    monkeypatch.setattr(pd, "read_csv", reader)
    entries = event_io.discover_event_sequences(event_session.session_root)
    assert len(entries) == 1 and entries[0]["available"]
    assert entries[0]["result_path"] == report["result_path"]
    packet = event_io.load_event_sequences(report["result_path"])
    source = event_io.get_source_activity(packet, pc_rank=1)
    assert source.values.shape == (7, 8)
    assert "A" in source.source_label
    reader.assert_not_called()


def test_pc_reference_binds_exact_relative_archive_and_both_choice_maps(event_session) -> None:
    """Input: session. Return None; PC product stores exact identity/hash and mappings without copying score tensors."""
    report = prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "pc", region="PFC"))
    packet = event_io.load_event_sequences(report["result_path"])
    reference = packet["meta"]["pca_reference"]
    assert not Path(reference["path"]).is_absolute()
    path = event_session.session_root / reference["path"]
    assert path.is_file() and path.stat().st_size == reference["size"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
    timeline = timeline_io.load_timeline_result(path)
    assert timeline.meta["configuration_id"] == reference["configuration_id"]
    assert reference["region"] == "PFC" and reference["probe_id"] == "A"
    regional = timeline.regions["PFC"]
    np.testing.assert_array_equal(packet["regional_choice_positions"], regional.choice_positions)
    np.testing.assert_array_equal(packet["activity_row_positions"], packet["shared_choices"].row_position.to_numpy()[regional.choice_positions])
    for rank in (1, 2):
        source = event_io.get_source_activity(packet, pc_rank=rank)
        np.testing.assert_array_equal(source.values, regional.scores[:, :, rank - 1])


def test_portable_session_relocation_resolves_pc_without_original_root(event_session, tmp_path) -> None:
    """Input: session/temp Path. Return None; relative PCA reference works after entire session tree relocation."""
    report = prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "pc", region="PFC"))
    relative = report["result_path"].relative_to(event_session.session_root)
    moved = tmp_path / "relocated-session"
    shutil.copytree(event_session.session_root, moved)
    shutil.rmtree(event_session.session_root)
    packet = event_io.load_event_sequences(moved / relative)
    assert packet["session_root"] == moved
    assert event_io.get_source_activity(packet, pc_rank=1).values.shape == (7, 8)


@pytest.mark.parametrize("damage", ["missing", "bytes", "configuration", "mapping"])
def test_exact_pc_reference_failures_are_explicit_on_full_load(event_session, monkeypatch, damage: str) -> None:
    """Input: session/damage name. Return None; missing/changed PCA or incompatible saved maps cannot display."""
    report = prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "pc", region="PFC"))
    packet = event_io.load_event_sequences(report["result_path"])
    pca = event_session.session_root / packet["meta"]["pca_reference"]["path"]
    if damage == "missing":
        pca.unlink()
        entries = event_io.discover_event_sequences(event_session.session_root)
        assert len(entries) == 1 and not entries[0]["available"] and entries[0]["unavailable_reason"]
    elif damage == "bytes":
        payload = bytearray(pca.read_bytes())
        payload[len(payload) // 2] ^= 1
        pca.write_bytes(payload)
        # Byte count is unchanged: hash verification must precede score decoding.
        read_pca = Mock(wraps=timeline_io.load_timeline_result)
        monkeypatch.setattr(timeline_io, "load_timeline_result", read_pca)
    else:
        with np.load(report["result_path"], allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if damage == "configuration":
            arrays["meta"].item()["pca_reference"]["configuration_id"] = "different"
        else:
            arrays["regional_choice_positions"] = arrays["regional_choice_positions"][::-1]
        _rewrite_archive(report["result_path"], arrays)
    with pytest.raises((FileNotFoundError, ValueError), match="PCA|reference|identity|mapping"):
        event_io.load_event_sequences(report["result_path"])
    if damage == "bytes":
        read_pca.assert_not_called()


@pytest.mark.parametrize("damage", ["version", "schema", "missing_array", "unit_shape", "source_map", "condition_shape", "summary_count"])
def test_saved_packet_rejects_versions_missing_fields_and_invalid_axes(event_session, damage: str) -> None:
    """Input: session/damage name. Return None; invalid metadata/arrays never become reusable scientific products."""
    report = prepare_event_sequences(EventSequenceRequest(event_session.metadata_path, "unit", unit_id="A:7"))
    with np.load(report["result_path"], allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if damage in {"version", "schema"}:
        arrays["meta"].item()["analysis_version" if damage == "version" else "schema_version"] = "unsupported"
    elif damage == "missing_array":
        del arrays["activity_row_positions"]
    elif damage == "unit_shape":
        arrays["unit_rates_hz"] = arrays["unit_rates_hz"][:, :-1]
    elif damage == "source_map":
        arrays["full_to_activity"][1] = 999
    elif damage == "condition_shape":
        arrays["transition__left_to_right__4"] = np.empty((0, 3), dtype=int)
    else:
        arrays["central_summary"]["behavioral_count"][0] += 1
    _rewrite_archive(report["result_path"], arrays)
    with pytest.raises(ValueError):
        event_io.load_event_sequences(report["result_path"])


def test_discovery_ignores_interrupted_directories_and_malformed_final_npz(event_session) -> None:
    """Input: session. Return None; only published readable products appear in saved-run discovery."""
    root = event_session.session_root / "analysis_runs"
    for name in ("event_sequence_heatmaps_incomplete", "event_sequence_heatmaps_bad"):
        directory = root / name
        directory.mkdir(parents=True)
        (directory / "run.log").write_text("interrupted\n")
    (root / "event_sequence_heatmaps_bad/results.npz").write_bytes(b"incomplete NPZ")
    assert event_io.discover_event_sequences(event_session.session_root) == []
