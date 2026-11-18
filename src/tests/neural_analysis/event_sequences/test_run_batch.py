"""Contracts for the explicit event-sequence session batch wrapper."""

from __future__ import annotations

import shutil
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest

from src.neural_analysis.event_sequences.run_session import (
    EventSequenceRequest,
)
from src.neural_analysis.event_sequences import run_batch as batch_module
from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count


def _unit_request(session, unit_id: str = "A:7") -> EventSequenceRequest:
    """Input: resolved synthetic session and probe-qualified unit ID; return its 0.5-s request."""
    return EventSequenceRequest(session.metadata_path, "unit", unit_id=unit_id)


def _copy_session(session, destination: Path) -> EventSequenceRequest:
    """Input: resolved synthetic fixture and destination; copy files and return request for the copy."""
    shutil.copytree(session.session_root, destination)
    metadata_path = destination / session.metadata_path.name
    return EventSequenceRequest(metadata_path, "unit", unit_id="A:7")


def _completed_future(function, *args, **kwargs) -> Future:
    """Input: callable/arguments; return a Future completed with its value or exception."""
    future = Future()
    try:
        future.set_result(function(*args, **kwargs))
    except BaseException as error:
        future.set_exception(error)
    return future


def test_run_batch_is_public_callable():
    """No inputs; return None after requiring the public callable batch API."""
    assert callable(batch_module.run_batch)


@pytest.mark.parametrize("requests", [[], ()])
def test_run_batch_rejects_empty_request_sequences(requests):
    """Input: empty list/tuple; return None after requiring ValueError before preparation."""
    with pytest.raises(ValueError, match="empty"):
        batch_module.run_batch(requests)


def test_run_batch_rejects_non_request_items(event_session):
    """Input: session fixture; return None after rejecting a string in place of a request."""
    with pytest.raises((ValueError, TypeError), match="request|EventSequenceRequest"):
        batch_module.run_batch(["not a request"])


@pytest.mark.parametrize("alias_kind", ["normalized_path", "different_filename"])
def test_run_batch_rejects_duplicate_canonical_session_roots(event_session, alias_kind):
    """Input: session/alias kind; return None after rejecting paths sharing one canonical root."""
    request = _unit_request(event_session)
    alias = EventSequenceRequest(request.metadata_path.parent / "." / request.metadata_path.name,
                                 "unit", unit_id="A:7")
    if alias_kind == "different_filename":
        alias = EventSequenceRequest(
            request.metadata_path.parent / "alternate_neural_session.json", "unit", unit_id="A:7"
        )
    with pytest.raises(ValueError, match="duplicate|Duplicate"):
        batch_module.run_batch([request, alias], dry_run=True)


def test_run_batch_preserves_probe_qualified_unit_identity(event_session, monkeypatch):
    """Input: session/patch fixture; return None after checking A:8 reaches preparation unchanged."""
    prepared = Mock(return_value={"status": "dry-run", "result_path": None,
                                  "configuration_id": "cfg", "elapsed_s": 0.0,
                                  "timings_s": {}})
    monkeypatch.setattr(batch_module, "prepare_event_sequences", prepared, raising=False)
    result = batch_module.run_batch([_unit_request(event_session, "A:8")], dry_run=True)
    assert prepared.call_args_list[0].args[0].unit_id == "A:8"
    assert prepared.call_args_list[0].kwargs == {"rerun": False, "dry_run": True}
    assert result[0]["configuration_id"] == "cfg"


def test_run_batch_dry_run_is_sequential_and_never_creates_executor(event_session, tmp_path, monkeypatch):
    """Input: session/temp/patch fixtures; return None after checking ordered plans without workers."""
    requests = [
        _unit_request(event_session),
        _copy_session(event_session, tmp_path / "second"),
    ]
    prepared = Mock(side_effect=[
        {"status": "dry-run", "result_path": None, "configuration_id": "1", "elapsed_s": 0.0, "timings_s": {}},
        {"status": "dry-run", "result_path": None, "configuration_id": "2", "elapsed_s": 0.0, "timings_s": {}},
    ])
    executor = Mock(side_effect=AssertionError("dry-run must not start workers"))
    monkeypatch.setattr(batch_module, "prepare_event_sequences", prepared, raising=False)
    monkeypatch.setattr(batch_module, "ProcessPoolExecutor", executor, raising=False)
    reports = batch_module.run_batch(requests, workers=2, rerun=True, dry_run=True)
    assert [call.args[0] for call in prepared.call_args_list] == requests
    assert all(call.kwargs == {"rerun": True, "dry_run": True} for call in prepared.call_args_list)
    executor.assert_not_called()
    assert [report["configuration_id"] for report in reports] == ["1", "2"]


def test_run_batch_reuses_existing_worker_policy(event_session, tmp_path, monkeypatch):
    """Input: session/temp/patch fixtures; return None after verifying shared worker-policy delegation."""
    requests = [_unit_request(event_session), _copy_session(event_session, tmp_path / "second")]
    policy = Mock(wraps=choose_worker_count)
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.__exit__.return_value = False
    executor.submit.side_effect = _completed_future
    executor_factory = Mock(return_value=executor)
    monkeypatch.setattr(batch_module, "choose_worker_count", policy, raising=False)
    monkeypatch.setattr(batch_module, "ProcessPoolExecutor", executor_factory, raising=False)
    monkeypatch.setattr(batch_module, "prepare_event_sequences", Mock(return_value={}), raising=False)
    batch_module.run_batch(requests, workers=9)
    policy.assert_called_once_with(2, 9)
    executor_factory.assert_called_once()
    assert executor_factory.call_args.kwargs["max_workers"] == choose_worker_count(2, 9)
    assert executor_factory.call_args.kwargs["mp_context"].get_start_method() == "spawn"


@pytest.mark.parametrize("workers", [0, -1])
def test_run_batch_rejects_nonpositive_workers_before_work(event_session, workers, monkeypatch):
    """Input: session, worker count, patch fixture; return None after rejection before work."""
    prepared = Mock()
    executor = Mock()
    monkeypatch.setattr(batch_module, "prepare_event_sequences", prepared, raising=False)
    monkeypatch.setattr(batch_module, "ProcessPoolExecutor", executor, raising=False)
    with pytest.raises(ValueError):
        batch_module.run_batch([_unit_request(event_session)], workers=workers)
    prepared.assert_not_called()
    executor.assert_not_called()


@pytest.mark.filterwarnings(
    r"error:This process .* is multi-threaded, use of fork\(\) may lead to deadlocks in the child\.:DeprecationWarning"
)
def test_run_batch_parallel_prepares_two_synthetic_sessions(event_session, tmp_path):
    """Input: session/temp fixtures; return None after checking two parallel results and script snapshots."""
    first = _copy_session(event_session, tmp_path / "first")
    second = _copy_session(event_session, tmp_path / "second")
    reports = batch_module.run_batch([first, second], workers=2)
    assert len(reports) == 2
    assert all(report["status"] == "prepared" for report in reports)
    result_paths = [Path(report["result_path"]) for report in reports]
    assert [path.parents[2] for path in result_paths] == [first.metadata_path.parent, second.metadata_path.parent]
    for result_path in result_paths:
        run_directory = result_path.parent
        assert result_path.is_file()
        assert (run_directory / "run_batch.py").read_text() == Path(batch_module.__file__).read_text()
        assert (run_directory / "run_session.py").is_file()
