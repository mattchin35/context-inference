"""Register immutable legacy PCA results before adding optional metadata."""

import hashlib
import json

import numpy as np
import pytest

from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as storage
from src.neural_analysis.population.run_session_timeline import find_saved_timeline, run_session_timeline
from src.neural_analysis.population.run_window_variance import run_window_variance, run_session_window_variance
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.tests.neural_analysis.test_session_timeline_io import synthetic_timeline_session as session_fixture

synthetic_timeline_session = session_fixture


def _legacy_result(session, width):
    """Write the original v1 configuration/NPZ independently of the new identity implementation."""
    current = storage.timeline_configuration(session, {"PFC": "A", "HPC": "B"}, width)
    fields = ("analysis_version", "session_id", "probe_roles", "bin_width_s", "window_s", "quality_rules", "selection", "sources")
    config = {key: current[key] for key in fields}
    path = session.metadata_path
    stat = path.stat()
    config["sources"] = [{"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}] + [entry for entry in config["sources"] if entry["path"] != str(path.resolve())]
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["configuration_id"] = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    origin, choices, events = storage.load_timeline_behavior(session)
    regions = {}
    for role, probe in (("PFC", "A"), ("HPC", "B")):
        regions[role] = timeline.fit_timeline_activity(
            np.random.default_rng(16).poisson(3, size=(3, round(4 / width), 2)),
            (f"{probe}:7", f"{probe}:8"), np.arange(3), role, probe, width,
        )
    directory = session.session_root / "analysis_runs" / f"session_timeline_pca_legacy_{width}_{config['configuration_id']}"
    directory.mkdir(parents=True)
    return storage.save_timeline_result(directory / "results.npz", timeline.SessionTimelineResult(session.session_id, origin, choices, events, regions, {}, config))


def _reload(session):
    """Resolve current synthetic metadata without loading scientific arrays."""
    return resolve_session_metadata(load_session_metadata(session.metadata_path), session.metadata_path)


def test_registration_preserves_both_old_bin_widths_and_exact_stage2_sources(synthetic_timeline_session, monkeypatch):
    """Optional treadmill edits reuse original paths and bytes without another neural operation."""
    session = synthetic_timeline_session
    products = []
    for width in (.5, .02):
        stage1 = _legacy_result(session, width)
        stage2 = run_window_variance(stage1).result_path
        products.append((width, stage1, stage2, stage1.read_bytes(), stage2.read_bytes(), stage1.stat().st_mtime_ns))
        registered = storage.register_timeline_scientific_identity(session, stage1)
        assert registered == storage.register_timeline_scientific_identity(session, stage1)
        assert registered.name == "scientific_identity.json"
    monkeypatch.setattr(storage, "load_timeline_population", pytest.fail)
    payload = json.loads(session.metadata_path.read_text())
    for source in ("speed.txt", "other.txt", None):
        payload["behavior"]["treadmill"] = source
        session.metadata_path.write_text(json.dumps(payload))
        for width, stage1, stage2, first_bytes, second_bytes, mtime in products:
            report = run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"}, width)
            assert report.status == "reused" and report.result_path == stage1
            summary = run_session_window_variance(session.metadata_path, {"PFC": "A", "HPC": "B"}, width)
            assert summary.status == "reused" and summary.result_path == stage2
            assert stage1.read_bytes() == first_bytes and stage2.read_bytes() == second_bytes
            assert stage1.stat().st_mtime_ns == mtime


@pytest.mark.parametrize("changed", ["quality", "trials", "events", "channels", "acquisition", "roles", "bin"])
def test_scientific_input_changes_still_invalidate_registered_results(synthetic_timeline_session, changed):
    """Do not weaken the source/scientific identity merely to tolerate an optional path."""
    session = synthetic_timeline_session
    path = _legacy_result(session, .5)
    storage.register_timeline_scientific_identity(session, path)
    before = storage.timeline_configuration(session, {"PFC": "A", "HPC": "B"}, .5)
    payload = json.loads(session.metadata_path.read_text())
    roles = {"PFC": "A", "HPC": "B"}
    width = .5
    if changed in ("quality", "trials", "events"):
        file = {"quality": session.session_root / "A/channel_quality.csv", "trials": session.behavior.trial_table_file, "events": session.behavior.event_table_file}[changed]
        file.write_text(file.read_text() + "\n")
    elif changed == "channels":
        payload["probes"]["A"]["unit_channels"] = [0]
    elif changed == "acquisition":
        payload["acquisition"] = "spikeglx"
    elif changed == "roles":
        roles = {"PFC": "B", "HPC": "A"}
    else:
        width = .1
    session.metadata_path.write_text(json.dumps(payload))
    config = storage.timeline_configuration(_reload(session), roles, width)
    assert config["configuration_id"] != before["configuration_id"]
    assert find_saved_timeline(session.session_root, config["configuration_id"]) is None


def test_edited_metadata_cannot_register_legacy_archive_and_stale_result_is_rejected(synthetic_timeline_session):
    """Registration proves an exact original match; registered files remain stat-bound."""
    session = synthetic_timeline_session
    path = _legacy_result(session, .5)
    storage.register_timeline_scientific_identity(session, path)
    config = storage.timeline_configuration(session, {"PFC": "A", "HPC": "B"}, .5)
    path.write_bytes(path.read_bytes() + b"stale")
    assert find_saved_timeline(session.session_root, config["configuration_id"]) is None
    other = _legacy_result(session, .02)
    session.metadata_path.write_text(session.metadata_path.read_text() + "\n")
    with pytest.raises(ValueError, match="legacy|metadata|match"):
        storage.register_timeline_scientific_identity(_reload(session), other)
    assert not (other.parent / "scientific_identity.json").exists()


def test_unregistered_legacy_result_is_not_guessed_after_metadata_edit(synthetic_timeline_session):
    """No shape/session-name heuristic substitutes for the required original identity."""
    session = synthetic_timeline_session
    _legacy_result(session, .5)
    session.metadata_path.write_text(session.metadata_path.read_text() + "\n")
    config = storage.timeline_configuration(_reload(session), {"PFC": "A", "HPC": "B"}, .5)
    assert find_saved_timeline(session.session_root, config["configuration_id"]) is None
