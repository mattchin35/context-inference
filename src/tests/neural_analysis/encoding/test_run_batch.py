"""Configuration-list batch commands with one process per canonical session."""

import json
from pathlib import Path
import shutil
from unittest.mock import MagicMock

import pytest

from src.neural_analysis.encoding import run_batch
from src.neural_analysis.encoding.configuration import EncodingConfig


def forbid_workers(*args: object, **kwargs: object) -> None:
    """Fail if arbitrary inputs start a process pool during dry-run; return no data."""
    pytest.fail("A dry-run must not start workers")


@pytest.fixture
def batch_configs(encoding_session: EncodingConfig, tmp_path: Path) -> tuple[Path, Path]:
    """Input session/config and temp Path; return two distinct config Paths, UTC source times."""
    second = tmp_path / "second-session"
    shutil.copytree(encoding_session.session_root, second)
    return encoding_session.session_root / "encoding.json", second / "encoding.json"


def test_worker_count_is_cpu_session_and_request_capped() -> None:
    """No inputs; dimensionless CPU/session/request counts cap workers, return None."""
    assert run_batch.choose_worker_count(3, None, cpu_count=8) == 3
    assert run_batch.choose_worker_count(8, 2, cpu_count=4) == 2
    assert run_batch.choose_worker_count(8, 9, cpu_count=4) == 4
    for request in (0, -1, True):
        with pytest.raises(ValueError, match="worker|positive"):
            run_batch.choose_worker_count(3, request, cpu_count=8)


def test_config_list_paths_are_relative_and_canonical_duplicates_fail(
    batch_configs: tuple[Path, Path], tmp_path: Path,
) -> None:
    """Input config Paths/temp directory; comments/relative paths retain order, return None."""
    listing = tmp_path / "configs.txt"
    listing.write_text("# sessions\nencoding-session/encoding.json\nsecond-session/encoding.json\n")
    assert run_batch.read_config_list(listing) == batch_configs
    listing.write_text("encoding-session/encoding.json\nencoding-session/../encoding-session/encoding.json\n")
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        run_batch.read_config_list(listing)


def test_distinct_config_files_for_same_session_are_rejected(
    batch_configs: tuple[Path, Path],
) -> None:
    """Input two config Paths; distinct JSON files cannot create two writers for one session."""
    alternate = batch_configs[0].with_name("alternate.json")
    alternate.write_bytes(batch_configs[0].read_bytes())
    with pytest.raises(ValueError, match="Duplicate|duplicate"):
        run_batch.run_encoding_batch((batch_configs[0], alternate), command="dry-run")


def test_batch_dry_run_is_sequential_and_writes_nothing(
    batch_configs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input two configs/patch; no process pool or output directories, return None."""
    monkeypatch.setattr(run_batch, "ProcessPoolExecutor", forbid_workers)
    report = run_batch.run_encoding_batch(batch_configs, command="dry-run")
    assert report["status"] == "dry-run"
    assert report["worker_count"] == 0
    assert len(report["reports"]) == 2
    assert all(item["status"] == "dry-run" for item in report["reports"])
    assert all(not (path.parent / "analysis_runs").exists() for path in batch_configs)


def test_batch_runs_two_synthetic_sessions_in_capped_processes(
    batch_configs: tuple[Path, Path],
) -> None:
    """Input two synthetic sessions; actual workers complete both saved runs, return None."""
    report = run_batch.run_encoding_batch(batch_configs, workers=2)
    assert report["status"] == "complete"
    assert 1 <= report["worker_count"] <= 2
    assert len(report["reports"]) == 2
    assert all(item["status"] == "complete" for item in report["reports"])
    assert len({item["run_directory"] for item in report["reports"]}) == 2


def test_batch_uses_spawn_instead_of_forking_loaded_scientific_threads(
    batch_configs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input configs/patch; executor gets explicit spawn context, return None, no physical units."""
    executor = MagicMock()
    executor.__enter__.return_value = executor
    executor.submit.return_value.result.return_value = {"status": "complete"}
    factory = MagicMock(return_value=executor)
    monkeypatch.setattr(run_batch, "ProcessPoolExecutor", factory)
    run_batch.run_encoding_batch(batch_configs, workers=2)
    assert factory.call_args.kwargs["mp_context"].get_start_method() == "spawn"


def test_batch_cli_parses_config_list_and_dry_run(
    batch_configs: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture,
) -> None:
    """Input session Paths/temp directory/terminal capture; CLI prints JSON plan, return None."""
    listing = tmp_path / "configs.txt"
    listing.write_text("\n".join(str(path) for path in batch_configs))
    assert run_batch.main(["dry-run", "--config-list", str(listing), "--workers", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"
