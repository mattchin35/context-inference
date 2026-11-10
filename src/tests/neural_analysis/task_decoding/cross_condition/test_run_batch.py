"""Scheduling and CLI contract tests for the cross-condition batch wrapper."""

from concurrent.futures import Future
from pathlib import Path

import pytest

from src.neural_analysis.task_decoding.cross_condition import run_batch
from src.neural_analysis.task_decoding.cross_condition.run_session import TransferRunReport
from src.neural_analysis.population import run_session_timeline_batch


def _report(path: Path, status: str = "dry-run") -> TransferRunReport:
    """Build a small runner report for metadata Path ``path``; no arrays or units apply."""
    return TransferRunReport(status, None, path.stem, 0.0, {})


def test_parallel_batch_forwards_runner_arguments_and_keeps_input_order(monkeypatch):
    """Run three Path inputs through a fake pool and preserve report order."""
    paths = (Path("first.json"), Path("second.json"), Path("third.json"))
    submitted = []

    class FakeExecutor:
        def __init__(self, max_workers):
            """Check the expected dimensionless process count."""
            assert max_workers == 2

        def __enter__(self):
            """Return this context manager without starting processes."""
            return self

        def __exit__(self, *args):
            """Close the fake context without suppressing exceptions."""
            return False

        def submit(self, function, path, **kwargs):
            """Return a completed future for one session metadata Path."""
            submitted.append((function, path, kwargs))
            future = Future()
            future.set_result(_report(path, "prepared"))
            return future

    def prepare(path, **kwargs):
        """Build a synthetic report for one metadata Path."""
        return _report(path, "prepared")

    def choose_count(count, workers):
        """Return a fixed worker count for this forwarding test."""
        return 2

    monkeypatch.setattr(run_batch, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(run_batch, "prepare_cross_condition_generalization", prepare)
    monkeypatch.setattr(run_batch, "choose_worker_count", choose_count)

    reports = run_batch.run_transfer_batch(
        paths, bin_width_ms=100, workers=2, rerun=True
    )

    assert [report.identity for report in reports] == [path.stem for path in paths]
    assert [path for _, path, _ in submitted] == list(paths)
    assert all(function is prepare for function, _, _ in submitted)
    assert [kwargs for _, _, kwargs in submitted] == [
        {"bin_width_ms": 100, "rerun": True, "dry_run": False}
    ] * 3


def test_batch_uses_shared_worker_count_contract(monkeypatch):
    """Cap default workers at available CPUs and sessions; allow a lower explicit cap."""
    paths = tuple(Path(f"session-{index}.json") for index in range(5))
    requested_counts = []

    class FakeExecutor:
        def __init__(self, max_workers):
            """Record the requested dimensionless process count."""
            requested_counts.append(max_workers)

        def __enter__(self):
            """Return this context manager without starting processes."""
            return self

        def __exit__(self, *args):
            """Close the fake context without suppressing exceptions."""
            return False

        def submit(self, function, path, **kwargs):
            """Return a completed future for one metadata Path."""
            future = Future()
            future.set_result(_report(path))
            return future

    monkeypatch.setattr(run_batch, "ProcessPoolExecutor", FakeExecutor)
    available_cpu_count = 3

    def prepare(path, **kwargs):
        """Build a synthetic report for one metadata Path."""
        return _report(path)

    def cpu_count():
        """Return the simulated dimensionless count of available CPUs."""
        return available_cpu_count

    monkeypatch.setattr(run_batch, "prepare_cross_condition_generalization", prepare)
    monkeypatch.setattr(run_session_timeline_batch.os, "cpu_count", cpu_count)

    run_batch.run_transfer_batch(paths)
    run_batch.run_transfer_batch(paths, workers=2)

    assert requested_counts == [3, 2]


def test_dry_run_is_sequential_and_never_creates_executor(monkeypatch):
    """Forward dry-run controls once per metadata Path without starting processes."""
    paths = (Path("one.json"), Path("two.json"))
    calls = []

    def forbidden_executor(*args, **kwargs):
        """Fail if dry-run attempts to create a process pool."""
        pytest.fail("dry-run must not create a process pool")

    def prepare(path, **kwargs):
        """Capture forwarded controls and return a synthetic session report."""
        calls.append((path, kwargs))
        return _report(path)

    monkeypatch.setattr(run_batch, "ProcessPoolExecutor", forbidden_executor)
    monkeypatch.setattr(run_batch, "prepare_cross_condition_generalization", prepare)

    reports = run_batch.run_transfer_batch(paths, bin_width_ms=50, dry_run=True)

    assert [report.identity for report in reports] == ["one", "two"]
    assert calls == [
        (path, {"bin_width_ms": 50, "rerun": False, "dry_run": True})
        for path in paths
    ]


def test_batch_cli_reads_session_list_and_prints_reports(monkeypatch, capsys):
    """CLI forwards parsed controls and returns success after reporting each session."""
    listing = Path("sessions.txt")
    paths = (Path("one.json"), Path("two.json"))
    calls = []

    def read_list(path):
        """Record the session-list Path and return fixed metadata Paths."""
        calls.append(("listing", path))
        return paths

    monkeypatch.setattr(run_batch, "read_session_list", read_list)

    def run(paths_arg, **kwargs):
        """Capture parsed controls and return synthetic reports in path order."""
        calls.append((paths_arg, kwargs))
        return [_report(path, "prepared") for path in paths_arg]

    monkeypatch.setattr(run_batch, "run_transfer_batch", run)

    status = run_batch.main([
        "--session-list", str(listing), "--bin-width-ms", "20", "--workers", "1",
        "--rerun", "--dry-run",
    ])

    assert status == 0
    assert calls == [
        ("listing", listing),
        (paths, {"bin_width_ms": 20, "workers": 1, "rerun": True, "dry_run": True}),
    ]
    assert capsys.readouterr().out.splitlines() == [
        "prepared: one", "prepared: two"
    ]


@pytest.mark.parametrize(
    "stage,error",
    [("list", OSError("missing list")), ("run", ValueError("invalid workers"))],
)
def test_batch_cli_reports_ordinary_errors_as_exit_one(monkeypatch, capsys, stage, error):
    """Convert ordinary input and runner failures into a concise CLI error."""

    def read_list(path):
        """Return a session Path or raise the configured list-stage error."""
        if stage == "list":
            raise error
        return (Path("one.json"),)

    def run(paths, **kwargs):
        """Raise the configured runner-stage error."""
        raise error

    monkeypatch.setattr(run_batch, "read_session_list", read_list)
    monkeypatch.setattr(run_batch, "run_transfer_batch", run)

    status = run_batch.main(["--session-list", "sessions.txt"])

    captured = capsys.readouterr()
    assert status == 1
    assert "Transfer batch failed:" in captured.err
    assert str(error) in captured.err

