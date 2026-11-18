"""Explicit CCA preparation, immutable reuse, and ordered session-level execution."""

from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import shutil
import tempfile
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.neural_analysis.population import cca
from src.neural_analysis.population import run_session_cca as runner
from src.neural_analysis.population import run_session_cca_batch as batch
from src.neural_analysis.population import run_session_timeline_batch as timeline_batch
from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as timeline_session_fixture,
)


synthetic_timeline_session = timeline_session_fixture
ROLES = {"PFC": "A", "HPC": "B"}


@pytest.fixture
def immediate_pool(monkeypatch):
    """Return an executor-constructor spy; submitted session calls run through real Futures."""
    constructor = MagicMock()
    executor = constructor.return_value.__enter__.return_value
    executor.submit.side_effect = _immediate_submit
    monkeypatch.setattr(batch, "ProcessPoolExecutor", constructor)
    return constructor


def test_explicit_prepare_writes_complete_run_and_utc_provenance(synthetic_timeline_session):
    """A fresh run publishes one paired fit, reproducibility files and a NY-stamped directory."""
    session = synthetic_timeline_session

    report = runner.run_session_cca(session.metadata_path, ROLES)

    assert report.status == "prepared" and report.elapsed_s >= 0
    assert report.result_path.is_file()
    result = cca_io.load_cca_result(report.result_path)
    assert result.meta["configuration_id"] == report.configuration_id
    assert result.choices.trial_id.tolist() == [101, 102, 103]
    assert result.fit.scores["PFC"].shape == (3, 8, 2)
    assert result.fit.scores["HPC"].shape == (3, 8, 2)
    for name in ("config.json", "run.log", "summary.md", "run_session.py", "run_batch.py"):
        assert (report.result_path.parent / name).is_file()
    config = json.loads((report.result_path.parent / "config.json").read_text())
    assert config["configuration_id"] == report.configuration_id
    started = datetime.fromisoformat(result.meta["started_utc"].replace("Z", "+00:00"))
    completed = datetime.fromisoformat(result.meta["completed_utc"].replace("Z", "+00:00"))
    assert started.utcoffset() == completed.utcoffset() == timedelta(0)
    stamp = started.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    assert report.result_path.parent.name == f"session_timeline_cca_{stamp}_{report.configuration_id}"
    first_log_line = (report.result_path.parent / "run.log").read_text().splitlines()[0]
    assert re.search(r"\d{2}:\d{2}:\d{2}.*(?:Z|\+00:00|UTC)", first_log_line)
    summary = (report.result_path.parent / "summary.md").read_text()
    assert session.session_id in summary and "run_session.py" in summary and "run_batch.py" in summary


def test_compatible_reuse_does_not_reload_spikes_or_refit(synthetic_timeline_session, monkeypatch):
    """Current small source checks can reuse the same immutable completed archive."""
    session = synthetic_timeline_session
    first = runner.run_session_cca(session.metadata_path, ROLES)
    original = first.result_path.read_bytes()
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_call)
    monkeypatch.setattr(cca, "CanCorr", _forbid_call)

    reused = runner.run_session_cca(session.metadata_path, ROLES)

    assert reused.status == "reused"
    assert reused.result_path == first.result_path
    assert reused.configuration_id == first.configuration_id
    assert first.result_path.read_bytes() == original


def test_explicit_rerun_creates_another_result_without_overwriting(synthetic_timeline_session):
    """Rerun repeats the same computation into a distinct timestamped output directory."""
    session = synthetic_timeline_session
    first = runner.run_session_cca(session.metadata_path, ROLES)
    original = first.result_path.read_bytes()

    again = runner.run_session_cca(session.metadata_path, ROLES, rerun=True)

    assert again.status == "prepared" and again.result_path != first.result_path
    assert again.configuration_id == first.configuration_id
    assert first.result_path.read_bytes() == original
    for region in ROLES:
        np.testing.assert_array_equal(cca_io.load_cca_result(again.result_path).fit.scores[region],
                                      cca_io.load_cca_result(first.result_path).fit.scores[region])


def test_dry_run_reads_no_spikes_fits_or_outputs(synthetic_timeline_session, monkeypatch):
    """Explicit planning validates metadata/IRIG but allocates no population or result files."""
    session = synthetic_timeline_session
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_call)
    monkeypatch.setattr(cca, "CanCorr", _forbid_call)

    report = runner.run_session_cca(session.metadata_path, ROLES, dry_run=True)

    assert report.status == "dry-run" and report.result_path is None
    assert report.configuration_id and report.elapsed_s >= 0
    assert not (session.session_root / "analysis_runs").exists()


@pytest.mark.parametrize("roles", [None, {}, {"PFC": "A"}, {"PFC": "A", "HPC": "A"},
                                   {"PFC": "A", "HPC": "unknown"}])
def test_version_two_requires_two_distinct_explicit_roles(synthetic_timeline_session, roles):
    """Invalid/missing assignments fail before a partial paired product is created."""
    with pytest.raises(ValueError):
        runner.run_session_cca(synthetic_timeline_session.metadata_path, roles)
    assert not (synthetic_timeline_session.session_root / "analysis_runs").exists()


def test_version_three_uses_existing_roles_but_not_for_partial_overrides(synthetic_timeline_session):
    """Metadata defaults apply only when the caller has not made an explicit selection."""
    session = synthetic_timeline_session
    _write_version_three(session.metadata_path)

    report = runner.run_session_cca(session.metadata_path, dry_run=True)

    assert report.status == "dry-run"
    with pytest.raises(ValueError):
        runner.run_session_cca(session.metadata_path, {"PFC": "A"}, dry_run=True)


def test_bin_and_population_changes_prepare_distinct_configurations(synthetic_timeline_session):
    """Real scientific settings select new count/basis products, not relabeled cached scores."""
    session = synthetic_timeline_session
    baseline = runner.run_session_cca(session.metadata_path, ROLES)
    finer = runner.run_session_cca(session.metadata_path, ROLES, 0.1)
    good_only = runner.run_session_cca(session.metadata_path, ROLES, population_quality="good")

    assert len({baseline.configuration_id, finer.configuration_id, good_only.configuration_id}) == 3
    assert cca_io.load_cca_result(finer.result_path).fit.scores["PFC"].shape == (3, 40, 2)
    assert cca_io.load_cca_result(good_only.result_path).fit.transforms["PFC"].input_unit_ids == ("A:7",)


@pytest.mark.parametrize("stage", ["loading", "fitting"])
def test_source_change_during_preparation_rejects_publication(synthetic_timeline_session, monkeypatch, stage):
    """A changed quality-file identity cannot be published under the initial source digest."""
    session = synthetic_timeline_session
    quality = session.session_root / "A/channel_quality.csv"
    original = timeline_io.load_timeline_population if stage == "loading" else cca.CanCorr

    def change_source_after_call(*args, **kwargs):
        """Delegate one real operation, then simulate an ordinary source edit in this fixture."""
        value = original(*args, **kwargs)
        quality.write_text(quality.read_text() + "\n")
        return value

    if stage == "loading":
        monkeypatch.setattr(timeline_io, "load_timeline_population", change_source_after_call)
    else:
        monkeypatch.setattr(cca, "CanCorr", change_source_after_call)

    with pytest.raises(ValueError):
        runner.run_session_cca(session.metadata_path, ROLES)
    assert cca_io.discover_cca_results(session.session_root) == ()
    assert list((session.session_root / "analysis_runs").glob("session_timeline_cca_*/run.log"))


def test_numerical_failure_logs_without_hiding_previous_completed_result(synthetic_timeline_session, monkeypatch):
    """A failed explicit rerun leaves the earlier compatible immutable result available."""
    session = synthetic_timeline_session
    first = runner.run_session_cca(session.metadata_path, ROLES)
    old_bytes = first.result_path.read_bytes()
    monkeypatch.setattr(cca, "CanCorr", _numerical_failure)

    with pytest.raises(ValueError):
        runner.run_session_cca(session.metadata_path, ROLES, rerun=True)

    headers = cca_io.discover_cca_results(session.session_root, first.configuration_id)
    assert tuple(header.path for header in headers) == (first.result_path,)
    assert first.result_path.read_bytes() == old_bytes
    logs = list((session.session_root / "analysis_runs").glob("session_timeline_cca_*/run.log"))
    assert len(logs) == 2
    assert any("synthetic numerical failure" in path.read_text() for path in logs)


def test_fresh_reuse_still_requires_current_source_validation(synthetic_timeline_session):
    """Historical loading remains possible, but missing current sources block fresh reuse."""
    session = synthetic_timeline_session
    first = runner.run_session_cca(session.metadata_path, ROLES)
    (session.session_root / "A/aligned.npz").unlink()

    with pytest.raises((ValueError, OSError)):
        runner.run_session_cca(session.metadata_path, ROLES)
    assert cca_io.load_cca_result(first.result_path).fit is not None


def test_runner_refuses_analysis_outputs_inside_the_repository(synthetic_timeline_session):
    """A genuinely repository-contained session is rejected before creating analysis outputs."""
    cache = Path(__file__).resolve().parents[4] / ".pytest_cache"
    with tempfile.TemporaryDirectory(dir=cache, prefix="cca_guard_") as folder:
        session_root = Path(folder) / "session"
        shutil.copytree(synthetic_timeline_session.session_root, session_root)

        with pytest.raises(ValueError):
            runner.run_session_cca(session_root / "neural_session.json", ROLES)
        assert not (session_root / "analysis_runs").exists()


def test_saved_benchmark_reports_only_measured_numerical_load_and_resource_products(synthetic_timeline_session):
    """Benchmarking works before a plotter exists and reports actual NPZ/score sizes and load time."""
    report = runner.run_session_cca(synthetic_timeline_session.metadata_path, ROLES)
    result = cca_io.load_cca_result(report.result_path)

    measured = runner.benchmark_cca_result(report.result_path)

    assert measured["measurement"] == "measured; filesystem cache state unspecified"
    assert measured["npz_load_s"] >= 0
    assert measured["npz_bytes"] == report.result_path.stat().st_size
    assert measured["score_bytes"] == sum(values.nbytes for values in result.fit.scores.values())
    assert measured["peak_process_rss_bytes"] > 0
    assert measured["preparation_timings_s"]


def test_single_cli_parses_units_roles_rerun_and_numeric_benchmark(synthetic_timeline_session):
    """The documented command uses milliseconds at its boundary and saves measured sidecars."""
    session = synthetic_timeline_session
    arguments = ["--session-metadata", str(session.metadata_path), "--pfc-probe", "A",
                 "--hpc-probe", "B", "--bin-width-ms", "100"]
    assert runner.main([*arguments, "--dry-run"]) == 0
    assert not (session.session_root / "analysis_runs").exists()

    assert runner.main([*arguments, "--benchmark"]) == 0
    assert runner.main([*arguments, "--rerun"]) == 0

    headers = cca_io.discover_cca_results(session.session_root)
    assert len(headers) == 2 and all(header.configuration["bin_width_s"] == 0.1 for header in headers)
    assert list((session.session_root / "analysis_runs").glob("session_timeline_cca_*/benchmark_*.json"))


def test_batch_dry_run_has_no_pool_spike_load_or_write(synthetic_timeline_session, monkeypatch):
    """Dry-run is sequential metadata inspection, regardless of requested worker count."""
    monkeypatch.setattr(batch, "ProcessPoolExecutor", _forbid_call)
    monkeypatch.setattr(timeline_io, "load_timeline_population", _forbid_call)

    reports = batch.run_batch((synthetic_timeline_session.metadata_path,), ROLES, workers=5, dry_run=True)

    assert [report.status for report in reports] == ["dry-run"]
    assert not (synthetic_timeline_session.session_root / "analysis_runs").exists()


def test_batch_caps_spawn_workers_and_isolates_failures_in_input_order(tmp_path, immediate_pool, monkeypatch):
    """One session failure does not discard successes or reorder reports by file name."""
    monkeypatch.setattr(timeline_batch.os, "cpu_count", _two_cpus)
    monkeypatch.setattr(batch, "run_session_cca", _fake_session_run)
    paths = tuple(tmp_path / name for name in ("third.json", "bad.json", "first.json"))

    reports = batch.run_batch(paths, ROLES, workers=8)

    kwargs = immediate_pool.call_args.kwargs
    assert kwargs["max_workers"] == 2
    assert kwargs["mp_context"].get_start_method() == "spawn"
    assert [report.status for report in reports] == ["prepared", "failed", "prepared"]
    assert reports[0].configuration_id == "third" and reports[2].configuration_id == "first"
    assert "synthetic session failure" in reports[1].error


def test_single_and_batch_entry_points_share_numerical_outputs(synthetic_timeline_session, immediate_pool):
    """Batch dispatch calls the same real runner; an explicit new fit reproduces saved scores."""
    session = synthetic_timeline_session
    first = runner.run_session_cca(session.metadata_path, ROLES)

    reports = batch.run_batch((session.metadata_path,), ROLES, workers=1, rerun=True)

    assert reports[0].status == "prepared" and reports[0].result_path != first.result_path
    baseline = cca_io.load_cca_result(first.result_path)
    batched = cca_io.load_cca_result(reports[0].result_path)
    for region in ROLES:
        np.testing.assert_array_equal(baseline.fit.scores[region], batched.fit.scores[region])


def test_batch_list_helpers_and_cli_failure_status(synthetic_timeline_session, tmp_path, immediate_pool, monkeypatch):
    """Relative session lists and worker validation are reused; any failed report makes CLI nonzero."""
    assert batch.choose_worker_count(3, None, cpu_count=8) == 3
    assert batch.choose_worker_count(3, 1, cpu_count=8) == 1
    with pytest.raises(ValueError):
        batch.choose_worker_count(3, 0, cpu_count=8)
    listing = tmp_path / "sessions.txt"
    listing.write_text("# relative fixture\nsynthetic/neural_session.json\n")
    assert batch.read_session_list(listing) == (synthetic_timeline_session.metadata_path,)
    assert batch.main(["--session-list", str(listing), "--pfc-probe", "A", "--hpc-probe", "B", "--dry-run"]) == 0
    monkeypatch.setattr(batch, "run_session_cca", _always_fail)
    assert batch.main(["--session-list", str(listing), "--pfc-probe", "A", "--hpc-probe", "B"]) != 0


def _write_version_three(path: Path) -> None:
    """Upgrade fixture metadata with verified native role/PPC fields; signal arrays are unchanged."""
    payload = json.loads(path.read_text())
    payload["schema_version"] = "3"
    payload["analysis"] = {
        "probe_roles": {"pfc": "A", "hpc": "B", "ppc": "B"},
        "ppc": {"prerequisite_cache": "processed/ppc_source", "preview_cache": "processed/ppc_preview",
                "final_cache": "processed/ppc_final", "preview_shuffles": 100, "final_shuffles": 1000},
    }
    path.write_text(json.dumps(payload), encoding="ascii")


def _immediate_submit(function, *args, **kwargs) -> Future:
    """Execute one submitted synthetic session and expose normal Future result/exception behavior."""
    future = Future()
    try:
        future.set_result(function(*args, **kwargs))
    except Exception as error:
        future.set_exception(error)
    return future


def _fake_session_run(metadata_path, probe_roles=None, bin_width_s=0.5, **kwargs):
    """Return a small prepared report or fail for the labeled synthetic bad session."""
    if Path(metadata_path).stem == "bad":
        raise ValueError("synthetic session failure")
    return runner.CCARunReport("prepared", None, Path(metadata_path).stem, 0.0)


def _two_cpus() -> int:
    """Return two logical CPUs for deterministic worker-cap checks; no physical units."""
    return 2


def _numerical_failure(*args, **kwargs):
    """Raise a declared numerical fit failure; return never."""
    raise ValueError("synthetic numerical failure")


def _always_fail(*args, **kwargs):
    """Raise one ordinary synthetic session error; return never."""
    raise ValueError("synthetic session failure")


def _forbid_call(*args, **kwargs):
    """Fail if reuse/dry-run starts a population load, fit or process pool; return never."""
    pytest.fail("Reuse/dry-run must not start neural loading, fitting or a process pool")
