"""Preparation, saved-result reuse, partial probes, and CLI/batch contracts."""

from __future__ import annotations

import numpy as np
import pytest

from src.neural_analysis.population import run_session_timeline as runner
from src.neural_analysis.population import run_session_timeline_batch as batch
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.tests.neural_analysis.test_session_timeline_io import synthetic_timeline_session


def _forbid_population_loading(*args, **kwargs):
    """Fail if a dry run or matching saved analysis loads raw neural arrays."""
    pytest.fail("This operation must not load neural populations")


def test_prepare_reuses_saved_results_and_explicit_rerun_is_new(synthetic_timeline_session, monkeypatch):
    """Both probes share choices while each has stable units and saved fixed loadings."""
    session = synthetic_timeline_session
    report = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"})
    assert report.status == "prepared"
    result = timeline_io.load_timeline_result(report.result_path)
    assert set(result.regions) == {"PFC", "HPC"}
    assert result.regions["PFC"].transform.input_unit_ids == ("A:7", "A:8")
    assert result.regions["HPC"].transform.input_unit_ids == ("B:7", "B:8")
    for filename in ("config.json", "run.log", "summary.md", "run_session.py", "run_batch.py"):
        assert (report.result_path.parent / filename).is_file()

    original = timeline_io.load_timeline_population
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_population_loading)
    again = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"})
    assert again.status == "reused" and again.result_path == report.result_path
    monkeypatch.setattr(timeline_io, "load_timeline_population", original)
    rerun = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"}, rerun=True)
    assert rerun.result_path != report.result_path
    np.testing.assert_array_equal(timeline_io.load_timeline_result(rerun.result_path).regions["PFC"].scores,
                                  result.regions["PFC"].scores)


def test_missing_probe_and_truncated_coverage_do_not_remove_behavior(synthetic_timeline_session):
    """Regional eligibility is independent; behavioral markers include unavailable windows."""
    session = synthetic_timeline_session
    (session.session_root / "B/aligned.npz").unlink()
    path = session.session_root / "A/aligned.npz"
    with np.load(path) as archive:
        times = archive["spike_utc_unix"]
    np.savez(path, spike_utc_unix=times, irig_utc_unix=[1_700_000_010., 1_700_000_040.])
    report = runner.run_session_timeline(session.metadata_path, {"PFC": "A", "HPC": "B"})
    result = timeline_io.load_timeline_result(report.result_path)
    assert result.choices.trial_id.tolist() == [101, 102, 103]
    assert result.regions["PFC"].choice_positions.tolist() == [1, 2]
    assert "HPC" in result.unavailable


def test_dry_run_does_not_load_spikes_or_write_outputs(synthetic_timeline_session, monkeypatch):
    """Source/configuration inspection is useful before allocating activity arrays."""
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_population_loading)
    report = runner.run_session_timeline(synthetic_timeline_session.metadata_path,
                                         {"PFC": "A"}, dry_run=True)
    assert report.status == "dry-run" and report.result_path is None
    assert not (synthetic_timeline_session.session_root / "analysis_runs").exists()


def test_source_or_bin_change_requires_another_configuration(synthetic_timeline_session):
    """A new bin size or changed quality source cannot reuse an old fitted basis."""
    session = synthetic_timeline_session
    first = runner.run_session_timeline(session.metadata_path, {"PFC": "A"})
    second = runner.run_session_timeline(session.metadata_path, {"PFC": "A"}, bin_width_s=.1)
    assert first.configuration_id != second.configuration_id
    path = session.session_root / "A/channel_quality.csv"
    path.write_text(path.read_text() + "\n")
    third = runner.run_session_timeline(session.metadata_path, {"PFC": "A"})
    assert first.configuration_id != third.configuration_id


def test_batch_worker_count_and_relative_list_paths(synthetic_timeline_session, tmp_path):
    """Batch preparation parallelizes across sessions and accepts an explicit worker cap."""
    assert batch.choose_worker_count(3, None, cpu_count=8) == 3
    assert batch.choose_worker_count(3, 1, cpu_count=8) == 1
    with pytest.raises(ValueError):
        batch.choose_worker_count(3, 0, cpu_count=8)
    listing = tmp_path / "sessions.txt"
    listing.write_text("# synthetic session\nsynthetic/neural_session.json\n")
    assert batch.read_session_list(listing) == (synthetic_timeline_session.metadata_path,)


def test_documented_cli_shapes_execute_dry_runs(synthetic_timeline_session, tmp_path):
    """Copyable main/batch usage parses and returns without computing neural arrays."""
    path = str(synthetic_timeline_session.metadata_path)
    assert runner.main(["--session-metadata", path, "--pfc-probe", "A", "--bin-width-ms", "500", "--dry-run"]) == 0
    listing = tmp_path / "sessions.txt"
    listing.write_text(path + "\n")
    assert batch.main(["--session-list", str(listing), "--pfc-probe", "A", "--dry-run"]) == 0
