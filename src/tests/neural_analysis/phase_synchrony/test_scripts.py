"""Thin CLI delegation and independent session-process contracts (WP5b)."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.neural_analysis.phase_synchrony import run_session as runner
from src.neural_analysis.phase_synchrony import run_batch as batch
from src.neural_analysis.phase_synchrony import results
from src.neural_analysis.phase_synchrony.results import PhaseSynchronyConfig
from src.tests.neural_analysis.phase_synchrony.test_preparation import binary_session
from src.tests.neural_analysis.phase_synchrony.test_runner import runnable_session, config


def _report(status=0):
    """Return an array-free execution report with dimensionless status and second timings."""
    return runner.RunReport(status, None, (), (), {"total_s": 0.0, "first_product_s": None})


def test_cli_preserves_explicit_selection_and_execution_flags(monkeypatch):
    """CLI converts categorical arguments to the frozen config without doing scientific work."""
    action = Mock(return_value=_report())
    monkeypatch.setattr(runner, "run_session", action)
    assert runner.main([
        "--session-metadata", "/session/neural_session.json", "--components", "spike-ppc", "power",
        "--sites", "HPC1", "--pair", "HPC1", "HPC2", "--unit", "ProbeB:8", "--unit", "ProbeB:12",
        "--lfp-site", "HPC1", "--intervals", "primary", "secondary", "--order", "power", "spike-ppc",
        "--rerun", "--dry-run",
    ]) == 0
    args, kwargs = action.call_args
    assert args[0] == Path("/session/neural_session.json")
    selected = args[1]
    assert selected.components == ("spike-ppc", "power")
    assert selected.sites == ("HPC1",) and selected.pairs == (("HPC1", "HPC2"),)
    assert selected.unit_ids == ("ProbeB:8", "ProbeB:12") and selected.lfp_site == "HPC1"
    assert selected.intervals == ("primary", "secondary")
    assert selected.numerical_amplitude_threshold == selected.absolute_amplitude_threshold == 0.0
    assert kwargs == {"order": ("power", "spike-ppc"), "rerun": True, "dry_run": True, "prepare_only": None}


@pytest.mark.parametrize("arguments", [
    ["--components", "spike-ppc"],
    ["--components", "spike-ppc", "--unit", "8", "--lfp-site", "HPC1"],
    ["--components", "power", "--order", "coherence"],
])
def test_invalid_cli_selection_fails_before_runner(arguments, monkeypatch, capsys):
    """Missing/ambiguous unit IDs and mismatched order cannot trigger expensive work."""
    action = Mock()
    monkeypatch.setattr(runner, "run_session", action)
    assert runner.main(["--session-metadata", "session.json", *arguments]) != 0
    action.assert_not_called()
    assert capsys.readouterr().err


def test_cli_defaults_and_preparation_only_flags(monkeypatch):
    """Default metrics remain primary with the accepted notch; preparation is explicit."""
    action = Mock(return_value=_report(1))
    monkeypatch.setattr(runner, "run_session", action)
    assert runner.main(["--session-metadata", "session.json", "--prepare-only", "native-lfp"]) == 1
    args, kwargs = action.call_args
    assert args[1] == PhaseSynchronyConfig()
    assert kwargs["prepare_only"] == "native-lfp"


def test_documented_commands_dry_run_without_signal_reads(runnable_session, monkeypatch, capsys):
    """Both module entry points execute real metadata-only plans and create no runs."""
    def forbid(*args, **kwargs):
        """Reject any array-loading call during metadata-only planning."""
        pytest.fail("Dry-run loaded scientific inputs")
    monkeypatch.setattr(runner.preparation, "load_site_lfp", forbid)
    assert runner.main(["--session-metadata", str(runnable_session), "--components", "power",
                        "--sites", "NamedSite", "--dry-run"]) == 0
    listing = runnable_session.parent / "sessions.txt"
    listing.write_text("neural_session.json\n", encoding="ascii")
    assert batch.main(["--session-list", str(listing), "--components", "power",
                       "--sites", "NamedSite", "--dry-run"]) == 0
    assert "planned" in capsys.readouterr().out
    assert not (runnable_session.parent / "analysis_runs").exists()


def test_batch_worker_limit_and_session_list_reuse(tmp_path):
    """CPU/session/request caps and duplicate-list rejection use existing shared helpers."""
    from src.neural_analysis.population import run_session_timeline_batch as existing
    assert batch.choose_worker_count is existing.choose_worker_count
    assert batch.read_session_list is existing.read_session_list
    assert batch.choose_worker_count(3, None, cpu_count=8) == 3
    assert batch.choose_worker_count(3, 1, cpu_count=8) == 1
    with pytest.raises(ValueError):
        batch.choose_worker_count(3, 0, cpu_count=8)
    metadata = tmp_path / "neural_session.json"
    metadata.write_text("{}", encoding="ascii")
    listing = tmp_path / "sessions.txt"
    listing.write_text("# sessions\nneural_session.json\n", encoding="ascii")
    assert batch.read_session_list(listing) == (metadata,)
    listing.write_text("neural_session.json\nneural_session.json\n", encoding="ascii")
    with pytest.raises(ValueError, match="Duplicate"):
        batch.read_session_list(listing)


def test_batch_submits_sessions_with_unchanged_config_and_flags(monkeypatch):
    """Real execution delegates once per session through a CPU-capped process pool."""
    paths = (Path("a.json"), Path("b.json"))
    selected = PhaseSynchronyConfig(components=("power",), sites=("HPC1",))
    pool = Mock()
    pool.submit.side_effect = [Mock(result=Mock(return_value=_report())), Mock(result=Mock(return_value=_report(1)))]
    constructor = Mock()
    constructor.return_value.__enter__ = Mock(return_value=pool)
    constructor.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", constructor)
    reports = batch.run_batch(paths, selected, workers=1, order=("power",), rerun=True, prepare_only="phase")
    assert constructor.call_count == 1
    assert constructor.call_args.kwargs["max_workers"] == 1
    assert constructor.call_args.kwargs["mp_context"].get_start_method() == "spawn"
    assert [report.status for report in reports] == [0, 1]
    assert len(pool.submit.call_args_list) == 2
    for call, path in zip(pool.submit.call_args_list, paths):
        assert call.args[1:] == (path, selected)
        assert call.kwargs == {"order": ("power",), "rerun": True, "dry_run": False, "prepare_only": "phase"}


def test_batch_dry_run_is_sequential_and_reports_failures(monkeypatch):
    """Metadata-only planning starts no workers and continues after an ordinary bad session."""
    selected = PhaseSynchronyConfig(components=("power",))
    action = Mock(side_effect=[ValueError("Bad metadata"), _report()])
    monkeypatch.setattr(batch, "run_session", action)
    pool = Mock(side_effect=AssertionError("Dry-run launched processes"))
    monkeypatch.setattr(batch, "ProcessPoolExecutor", pool)
    reports = batch.run_batch((Path("a.json"), Path("b.json")), selected, dry_run=True)
    assert [report.status for report in reports] == [1, 0]
    assert "Bad metadata" in reports[0].stages[0]["error"]
    assert all(call.kwargs["dry_run"] for call in action.call_args_list)


def test_real_run_saves_both_script_snapshots(runnable_session, config):
    """Completed single-session runs contain the actual main and batch scripts used."""
    report = runner.run_session(runnable_session, replace(config, sites=("NamedSite",), pairs=()))
    assert report.status == 0
    for filename in ("run_session.py", "run_batch.py"):
        expected = Path(runner.__file__).with_name(filename).read_bytes()
        assert (report.run_directory / filename).read_bytes() == expected


def test_writer_conflict_is_reported_by_cli_and_does_not_abort_batch(runnable_session, monkeypatch, capsys):
    """A live session writer is an ordinary input conflict; independent reports remain available."""
    with results.session_writer_lock(runnable_session.parent):
        assert runner.main(["--session-metadata", str(runnable_session), "--components", "power",
                            "--sites", "NamedSite"]) == 1
        assert "writer" in capsys.readouterr().err.lower()
        reports = batch.run_batch((runnable_session,), PhaseSynchronyConfig(components=("power",), sites=("NamedSite",)), workers=1)
        assert len(reports) == 1 and reports[0].status == 1
        assert "writer" in reports[0].stages[0]["error"].lower()


def test_unexpected_runtime_error_still_propagates(monkeypatch):
    """Boundary handling must not turn programming defects into ordinary unavailable-data reports."""
    action = Mock(side_effect=RuntimeError("Unexpected programming error"))
    monkeypatch.setattr(runner, "run_session", action)
    with pytest.raises(RuntimeError, match="Unexpected"):
        runner.main(["--session-metadata", "session.json"])
    monkeypatch.setattr(batch, "run_session", action)
    with pytest.raises(RuntimeError, match="Unexpected"):
        batch.run_batch((Path("session.json"),), dry_run=True)
