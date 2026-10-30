"""Stage 2 summaries reuse an exact saved Stage 1 result without modifying it."""

import importlib
import json

import numpy as np
import pytest

from src.neural_analysis.population.run_session_timeline import run_session_timeline
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as timeline_session_fixture,
)

synthetic_timeline_session = timeline_session_fixture


def test_summary_round_trip_reuse_and_explicit_rerun_preserve_stage1(
    synthetic_timeline_session, monkeypatch
):
    """Stage 2 has a versioned sidecar, scripts/logs, stable reuse, and explicit rerun behavior."""
    runner = importlib.import_module(
        "src.neural_analysis.population.run_window_variance"
    )
    storage = importlib.import_module(
        "src.neural_analysis.population.window_variance_io"
    )
    from src.neural_analysis.population import session_timeline_io

    session = synthetic_timeline_session
    stage1 = run_session_timeline(
        session.metadata_path, {"PFC": "A", "HPC": "B"}
    ).result_path
    before = stage1.read_bytes()
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", pytest.fail)
    first = runner.run_window_variance(stage1)
    assert first.status == "prepared"
    assert first.result_path.parent.parent == stage1.parent
    for name in ("run.log", "summary.md", "run_session.py", "run_batch.py"):
        assert (first.result_path.parent / name).is_file()
    loaded = storage.load_window_variance(first.result_path)
    assert loaded.choices.trial_id.tolist() == [101, 102, 103]
    assert set(loaded.regions) == {"PFC", "HPC"}
    assert loaded.meta["source_stage1"]["path"] == str(stage1)
    assert loaded.meta["parameters"]["variance_ddof"] == 1
    assert loaded.meta["parameters"]["threshold"] == 0.95
    assert loaded.meta["parameters"]["conservation_tolerance"] == 1e-10
    assert loaded.meta["parameters"]["threshold_tolerance"] == 1e-12
    with np.load(first.result_path, allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
        assert "pfc_pc_ids" in archive.files
    reused = runner.run_window_variance(stage1)
    assert reused.status == "reused" and reused.result_path == first.result_path
    rerun = runner.run_window_variance(stage1, rerun=True)
    assert rerun.result_path != first.result_path
    np.testing.assert_array_equal(
        storage.load_window_variance(rerun.result_path).regions["PFC"].counts95,
        loaded.regions["PFC"].counts95,
    )
    assert stage1.read_bytes() == before
    with pytest.raises(FileExistsError):
        storage.save_window_variance(first.result_path, loaded)


def test_dry_run_never_loads_scores_or_writes_and_exact_source_change_invalidates_summary(
    synthetic_timeline_session, monkeypatch
):
    """Different Stage 1 files and metric versions cannot silently reuse a stale sidecar."""
    runner = importlib.import_module(
        "src.neural_analysis.population.run_window_variance"
    )
    storage = importlib.import_module(
        "src.neural_analysis.population.window_variance_io"
    )
    session = synthetic_timeline_session
    stage1 = run_session_timeline(session.metadata_path, {"PFC": "A"}).result_path
    with monkeypatch.context() as dry:
        dry.setattr(runner.timeline_io, "load_timeline_result", pytest.fail)
        report = runner.run_window_variance(stage1, dry_run=True)
    assert report.status == "dry-run" and report.result_path is None
    assert not list(stage1.parent.glob("window_variance_*"))
    prepared = runner.run_window_variance(stage1)
    assert runner.find_saved_window_variance(stage1) == prepared.result_path
    config = storage.window_variance_configuration(stage1)
    copy = stage1.with_name("different_result.npz")
    copy.write_bytes(stage1.read_bytes())
    assert (
        storage.window_variance_configuration(copy)["configuration_id"]
        != config["configuration_id"]
    )
    assert runner.find_saved_window_variance(copy) is None
    with monkeypatch.context() as changed:
        changed.setattr(storage, "ANALYSIS_VERSION", "new-metric-version")
        assert runner.find_saved_window_variance(stage1) is None


def test_main_and_batch_window_summary_actions_reuse_prepared_stage1(
    synthetic_timeline_session, tmp_path, monkeypatch
):
    """Existing metadata entry points expose Stage 2 without loading spikes or refitting."""
    from src.neural_analysis.population import run_session_timeline as main
    from src.neural_analysis.population import run_session_timeline_batch as batch
    from src.neural_analysis.population import session_timeline_io

    session = synthetic_timeline_session
    run_session_timeline(session.metadata_path, {"PFC": "A"})
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", pytest.fail)
    assert (
        main.main(
            [
                "--session-metadata",
                str(session.metadata_path),
                "--pfc-probe",
                "A",
                "--window-summary",
            ]
        )
        == 0
    )
    listing = tmp_path / "sessions.txt"
    listing.write_text(str(session.metadata_path) + "\n")
    assert (
        batch.main(
            [
                "--session-list",
                str(listing),
                "--pfc-probe",
                "A",
                "--window-summary",
                "--dry-run",
            ]
        )
        == 0
    )


def test_benchmark_separates_stage2_reduction_and_plot_costs(
    synthetic_timeline_session,
):
    """Preparation timings, incremental metrics, cache loading, and payload are measured separately."""
    runner = importlib.import_module(
        "src.neural_analysis.population.run_window_variance"
    )
    stage1 = run_session_timeline(
        synthetic_timeline_session.metadata_path, {"PFC": "A"}
    ).result_path
    stage2 = runner.run_window_variance(stage1).result_path
    measured = runner.benchmark_window_variance(stage1, stage2)
    for name in (
        "stage1_load_s",
        "summary_load_s",
        "stage2_calculation_s",
        "figure_construction_s",
        "serialization_s",
        "json_bytes",
        "peak_process_rss_bytes",
    ):
        assert measured[name] >= 0
    assert measured["populations"]["PFC"]["windows"] == 3
    assert measured["populations"]["PFC"]["bins"] == 8
    assert measured["populations"]["PFC"]["basis_complete"]
    json.dumps(measured)
