"""WP8 contracts for pooled single-session Slurm execution.

Scheduler commands are monkeypatched and shell execution uses only temporary
fixtures. These tests never contact Slurm, SSH, rsync, or experimental data.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
WRAPPER = (
    REPOSITORY_ROOT
    / "src"
    / "shell_scripts"
    / "pooled_task_variable_decoding_slurm.sh"
)
SLURM_MODULE = "src.neural_analysis.task_decoding.pooled.slurm"
RUNNER_MODULE = "src.neural_analysis.task_decoding.pooled.run_session"
EXPECTED_RESOURCES = {
    "partition": "normal",
    "job_name": "pooled_task_decoding",
    "tasks": 1,
    "cpus_per_task": 1,
    "memory": "3G",
    "memory_bytes": 3 * 1024**3,
    "time": "24:00:00",
    "signal": "B:TERM@300",
    "output": "/gs/gsfs0/users/mchin1/logs/pooled_task_decoding_%A_%a.log",
    "mail_type": "ALL",
    "mail_user": "matthew.chin@einsteinmed.edu",
}


def _slurm_module():
    """Import the pooled Slurm module inside a test.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pooled.slurm``.
    """
    return importlib.import_module(SLURM_MODULE)


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Write one test-owned JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Destination below a pytest temporary directory.
    value : dict[str, object]
        JSON-safe values without array axes or physical units unless named.

    Returns
    -------
    None
        Replaces the test-owned file with canonical-enough UTF-8 JSON.
    """
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _prepared_run(tmp_path: Path) -> Path:
    """Create one minimal Slurm-prepared pooled run directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned parent directory.

    Returns
    -------
    pathlib.Path
        Exact run directory with prepared lifecycle and Slurm provenance.
    """
    directory = tmp_path / "session with spaces" / "analysis_runs" / "pooled-run"
    directory.mkdir(parents=True)
    _write_json(
        directory / "run_state.json",
        {
            "lifecycle": "prepared",
            "final_results_published": False,
            "error": None,
        },
    )
    _write_json(directory / "execution.json", {"mode": "slurm"})
    return directory


def _completed_process(
    arguments: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Build one synthetic scheduler command result.

    Parameters
    ----------
    arguments : list[str]
        Exact subprocess tokens with no physical units.
    stdout, stderr : str
        Synthetic scheduler streams.
    returncode : int
        Synthetic process exit status.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Text-mode record accepted by the production command seam.
    """
    return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)


def test_wrapper_has_exact_resources_and_offline_one_thread_execution() -> None:
    """The wrapper freezes the approved normal/3-GiB/24-hour request."""
    syntax = subprocess.run(
        ("bash", "-n", str(WRAPPER)),
        capture_output=True,
        text=True,
        check=False,
    )
    text = WRAPPER.read_text(encoding="ascii")

    assert syntax.returncode == 0, syntax.stderr
    for directive in (
        "#SBATCH --partition=normal",
        "#SBATCH --job-name=pooled_task_decoding",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=3G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --output=/gs/gsfs0/users/mchin1/logs/pooled_task_decoding_%A_%a.log",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=matthew.chin@einsteinmed.edu",
    ):
        assert directive in text
    assert "uv run --frozen --no-sync --offline" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "MKL_NUM_THREADS=1" in text
    assert "OPENBLAS_NUM_THREADS=1" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "rsync" not in text


def test_python_resource_contract_matches_wrapper() -> None:
    """Python receipts and shell directives share one resource mapping."""
    assert _slurm_module().REQUESTED_RESOURCES == EXPECTED_RESOURCES


def test_submit_new_plans_prepares_slurm_and_submits_exact_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """New submission uses bounded planning and the shared preparation seam."""
    slurm = _slurm_module()
    config_path = tmp_path / "pooled.json"
    config_path.write_text("{}\n", encoding="ascii")
    prepared = _prepared_run(tmp_path)
    calls: list[object] = []

    class FakePipeline:
        """Record bounded planning and immutable preparation calls."""

        @staticmethod
        def plan_pooled_decoding_session(path: Path) -> dict[str, object]:
            """Return one admitted tensor envelope for ``path``."""
            calls.append(("dry-run", Path(path)))
            return {"resource_envelope": {"tensor_allocation_bytes": 96_951_680}}

        @staticmethod
        def prepare_pooled_decoding_run(path: Path, *, execution_mode: str) -> Path:
            """Return the exact synthetic prepared directory."""
            calls.append(("prepare", Path(path), execution_mode))
            return prepared

    monkeypatch.setattr(slurm, "_pipeline_module", lambda: FakePipeline)
    monkeypatch.setattr(
        slurm,
        "submit_prepared_run",
        lambda directory: calls.append(("submit", Path(directory)))
        or {"job_id": "12345"},
    )

    receipt = slurm.submit_new(config_path)

    assert receipt["job_id"] == "12345"
    assert calls == [
        ("dry-run", config_path),
        ("prepare", config_path, "slurm"),
        ("submit", prepared),
    ]


def test_submission_records_exact_commit_resources_and_follow_up_commands(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """One accepted sbatch produces an immutable exact-run receipt."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    commands: list[list[str]] = []

    def accept(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Capture one scheduler request and return one job identifier."""
        commands.append(arguments)
        return _completed_process(arguments, stdout="12345;einstein\n")

    monkeypatch.setattr(slurm, "_run_command", accept)
    monkeypatch.setattr(slurm, "_repository_commit", lambda: "approved-commit")

    receipt = slurm.submit_prepared_run(run_directory)

    assert len(commands) == 1
    assert commands[0][:2] == ["sbatch", "--parsable"]
    assert commands[0][-3:] == ["_execute-prepared", "--run-directory", str(run_directory)]
    assert receipt["job_id"] == "12345"
    assert receipt["repository_commit"] == "approved-commit"
    assert receipt["requested_resources"] == EXPECTED_RESOURCES
    assert receipt["scheduler_log_path"].endswith("pooled_task_decoding_12345_0.log")
    assert "submit-resume" in receipt["resume_command"]
    assert "status" in receipt["status_command"]
    saved = json.loads((run_directory / "slurm_submission.json").read_text())
    assert saved == receipt


def test_submission_failure_is_durable_and_not_retried(monkeypatch, tmp_path: Path) -> None:
    """One rejected sbatch marks failure and performs no automatic retry."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    calls = 0

    def reject(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Return one scheduler failure while counting calls."""
        nonlocal calls
        calls += 1
        return _completed_process(
            arguments,
            stderr="scheduler unavailable",
            returncode=1,
        )

    monkeypatch.setattr(slurm, "_run_command", reject)
    monkeypatch.setattr(slurm, "_repository_commit", lambda: "approved-commit")

    with pytest.raises(RuntimeError, match="scheduler unavailable|sbatch"):
        slurm.submit_prepared_run(run_directory)

    state = json.loads((run_directory / "run_state.json").read_text())
    assert state["lifecycle"] == "failed"
    assert state["error"]
    assert calls == 1
    assert not (run_directory / "slurm_submission.json").exists()


def test_status_combines_pipeline_state_with_one_read_only_sacct_query(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Status performs one bounded accounting query and changes no artifacts."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    _write_json(run_directory / "slurm_submission.json", {"job_id": "12345"})
    before = {path.name: path.read_bytes() for path in run_directory.iterdir()}
    commands: list[list[str]] = []

    class FakePipeline:
        """Expose one immutable running pooled status."""

        @staticmethod
        def inspect_pooled_decoding_status(directory: Path) -> dict[str, object]:
            """Return lifecycle state for the exact requested directory."""
            assert Path(directory) == run_directory
            return {"lifecycle": "running", "completed_checkpoint_count": 2}

    def accounting(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Return one parsable job-level accounting row."""
        commands.append(arguments)
        return _completed_process(
            arguments,
            stdout="12345|COMPLETED|03:35:20|03:35:17|1|1057M|3G|24:00:00|0:0\n",
        )

    monkeypatch.setattr(slurm, "_pipeline_module", lambda: FakePipeline)
    monkeypatch.setattr(slurm, "_run_command", accounting)

    status = slurm.inspect_status(run_directory)

    assert len(commands) == 1
    assert commands[0][0] == "sacct"
    assert status["scheduler"]["State"] == "COMPLETED"
    assert status["scheduler"]["MaxRSS"] == "1057M"
    after = {path.name: path.read_bytes() for path in run_directory.iterdir()}
    assert after == before

