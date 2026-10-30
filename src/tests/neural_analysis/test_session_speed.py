"""Explicit independent speed preparation, named-array storage, and CLI contracts."""

from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as session_fixture,
)
from src.tests.neural_analysis.test_treadmill_speed import (
    synthetic_speed_messages as messages_fixture,
)
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)

synthetic_timeline_session = session_fixture
synthetic_speed_messages = messages_fixture


@pytest.fixture
def synthetic_speed_session(synthetic_timeline_session, synthetic_speed_messages):
    """Declare one small speed log independently of the two synthetic neural probes."""
    session = synthetic_timeline_session
    source = session.session_root / "teensy.txt"
    source.write_text(synthetic_speed_messages["text"])
    payload = json.loads(session.metadata_path.read_text())
    payload["behavior"]["treadmill"] = "teensy.txt"
    session.metadata_path.write_text(json.dumps(payload))
    return resolve_session_metadata(
        load_session_metadata(session.metadata_path), session.metadata_path
    )


def test_speed_round_trip_reuses_then_reruns_without_neural_loading(
    synthetic_speed_session, synthetic_speed_messages, monkeypatch
):
    """Signed UTC samples and source identity survive NPZ; preparation never touches spikes."""
    speed = importlib.import_module("src.neural_analysis.session_speed")
    from src.neural_analysis.population import session_timeline_io

    monkeypatch.setattr(session_timeline_io, "load_timeline_population", pytest.fail)
    session = synthetic_speed_session
    report = speed.run_session_speed(session.metadata_path)
    assert report.status == "prepared"
    loaded = speed.load_speed_result(report.result_path)
    np.testing.assert_array_equal(
        loaded.speed_mm_s, synthetic_speed_messages["speed_mm_s"]
    )
    np.testing.assert_allclose(
        loaded.utc_s, synthetic_speed_messages["utc_s"], atol=2e-6, rtol=0
    )
    assert loaded.meta["analysis_version"] == "treadmill-speed-v2"
    assert loaded.meta["random_seed"] is None
    assert loaded.meta["units"]["speed_mm_s"] == "signed mm/s"
    assert loaded.meta["units"]["frame_utc_s"] == "UTC Unix seconds"
    assert loaded.meta["axes"]["source_rows"] == ["speed_sample"]
    for name in (
        "config.json",
        "run.log",
        "summary.md",
        "run_session.py",
        "run_batch.py",
    ):
        assert (report.result_path.parent / name).is_file()
    with np.load(report.result_path, allow_pickle=True) as archive:
        assert set(
            (
                "utc_s",
                "speed_mm_s",
                "source_rows",
                "frame_clock_s",
                "frame_utc_s",
                "meta",
            )
        ).issubset(archive.files)
        assert isinstance(archive["meta"].item(), dict)
    before = report.result_path.read_bytes()
    with pytest.raises(FileExistsError):
        speed.save_speed_result(report.result_path, loaded)
    with monkeypatch.context() as cached:
        cached.setattr(speed.irig_serial_io, "load_coolterm_serial_txt", pytest.fail)
        repeated = speed.run_session_speed(session.metadata_path)
        assert (
            repeated.status == "reused" and repeated.result_path == report.result_path
        )
    rerun = speed.run_session_speed(session.metadata_path, rerun=True)
    assert rerun.result_path != report.result_path
    assert report.result_path.read_bytes() == before


def test_speed_identity_is_independent_of_trials_probes_and_metadata_stat(
    synthetic_speed_session,
):
    """UTC speed has no neural-bin or behavior-origin dependency; source/version changes invalidate it."""
    speed = importlib.import_module("src.neural_analysis.session_speed")
    session = synthetic_speed_session
    config = speed.speed_configuration(session)
    prepared = speed.run_session_speed(session.metadata_path)
    session.behavior.trial_table_file.write_text("changed behavior origin")
    session.metadata_path.write_text(session.metadata_path.read_text() + "\n")
    reduced = replace(
        session,
        probes=(),
        behavior=replace(
            session.behavior, trial_table_file=None, event_table_file=None
        ),
    )
    assert speed.speed_configuration(reduced) == config
    session.behavior.treadmill_file.write_text(
        session.behavior.treadmill_file.read_text() + "\n"
    )
    changed = speed.speed_configuration(session)
    assert changed["configuration_id"] != config["configuration_id"]
    assert (
        speed.find_saved_speed(session.session_root, changed["configuration_id"])
        is None
    )
    assert prepared.result_path.is_file()
    with pytest.MonkeyPatch.context() as versions:
        versions.setattr(speed, "ANALYSIS_VERSION", "changed-speed-version")
        assert (
            speed.speed_configuration(session)["configuration_id"]
            != changed["configuration_id"]
        )


def test_speed_dry_run_is_stat_only_and_failed_preparation_is_not_discovered(
    synthetic_speed_session, monkeypatch
):
    """A dry-run does not decode; a malformed log cannot masquerade as a prepared speed result."""
    speed = importlib.import_module("src.neural_analysis.session_speed")
    session = synthetic_speed_session
    with monkeypatch.context() as dry:
        dry.setattr(speed.irig_serial_io, "load_coolterm_serial_txt", pytest.fail)
        report = speed.run_session_speed(session.metadata_path, dry_run=True)
    assert report.status == "dry-run" and report.result_path is None
    assert not (session.session_root / "analysis_runs").exists()
    session.behavior.treadmill_file.write_text("broken")
    with pytest.raises(ValueError, match="line 1"):
        speed.run_session_speed(session.metadata_path)
    identity = speed.speed_configuration(session)["configuration_id"]
    assert speed.find_saved_speed(session.session_root, identity) is None
    assert list(
        (session.session_root / "analysis_runs").glob("treadmill_speed_*/run.log")
    )


def test_main_and_batch_speed_actions_are_separate_from_neural_work(
    synthetic_speed_session, tmp_path, monkeypatch
):
    """The existing entry points accept independent speed preparation and session-level workers."""
    from src.neural_analysis.population import run_session_timeline as main
    from src.neural_analysis.population import run_session_timeline_batch as batch
    from src.neural_analysis.population import session_timeline_io

    monkeypatch.setattr(session_timeline_io, "load_timeline_population", pytest.fail)
    path = str(synthetic_speed_session.metadata_path)
    assert main.main(["--session-metadata", path, "--speed", "--dry-run"]) == 0
    assert main.main(["--session-metadata", path, "--speed"]) == 0
    with pytest.raises(SystemExit):
        main.main(["--session-metadata", path, "--speed", "--window-summary"])
    listing = tmp_path / "sessions.txt"
    listing.write_text(path + "\n")
    assert batch.main(["--session-list", str(listing), "--speed", "--dry-run"]) == 0
    assert (
        batch.main(["--session-list", str(listing), "--speed", "--workers", "1"]) == 0
    )
