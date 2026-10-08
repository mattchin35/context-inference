"""WP11 RED contracts for single-session Slurm execution.

All scheduler commands are replaced with temporary fake executables or
monkeypatched subprocess seams. These tests never contact Slurm, SSH, rsync,
the network, or experimental data.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
WRAPPER = REPOSITORY_ROOT / "src" / "shell_scripts" / "task_variable_decoding_slurm.sh"
SLURM_MODULE = "src.neural_analysis.task_decoding.slurm"
RUNNER_MODULE = "src.neural_analysis.task_decoding.run_session"
EXPECTED_RESOURCES = {
    "partition": "unlimited",
    "job_name": "task_decoding",
    "tasks": 1,
    "cpus_per_task": 1,
    "memory": "3G",
    "memory_bytes": 3 * 1024**3,
    "time": "2-00:00:00",
    "signal": "B:TERM@300",
    "output": "/gs/gsfs0/users/mchin1/logs/task_decoding_%j.log",
    "mail_type": "ALL",
    "mail_user": "matthew.chin@einsteinmed.edu",
}


def _slurm_module():
    """Import and return the WP11 operational module inside an individual test.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.slurm``. Importing inside tests
        keeps a missing RED module from aborting collection of other contracts.
    """
    return importlib.import_module(SLURM_MODULE)


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Write one small test-owned JSON mapping.

    Parameters
    ----------
    path : pathlib.Path
        Destination inside a pytest temporary directory.
    value : dict[str, object]
        JSON-safe lifecycle, execution, or submission mapping.

    Returns
    -------
    None
        Creates or replaces the test-owned file with UTF-8 JSON.
    """
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    """Return one test-owned JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Existing UTF-8 JSON file.

    Returns
    -------
    dict[str, object]
        Decoded mapping without physical units unless field names state them.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _prepared_run(tmp_path: Path) -> Path:
    """Create a minimal initialized Slurm run directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory.

    Returns
    -------
    pathlib.Path
        Exact run directory containing initialized state and Slurm provenance.
    """
    run_directory = tmp_path / "session with spaces" / "analysis_runs" / "exact-run"
    run_directory.mkdir(parents=True)
    _write_json(
        run_directory / "run_state.json",
        {
            "lifecycle": "initialized",
            "current_stage": "prepared",
            "final_results_published": False,
            "last_error": "",
        },
    )
    _write_json(
        run_directory / "execution.json",
        {"mode": "slurm", "history": [], "job_id": None},
    )
    (run_directory / "resume_command.txt").write_text(
        f"uv run python -m {RUNNER_MODULE} resume --run-directory "
        f"'{run_directory}' --detach\n",
        encoding="utf-8",
    )
    (run_directory / "status_command.txt").write_text(
        f"uv run python -m {RUNNER_MODULE} status --run-directory "
        f"'{run_directory}'\n",
        encoding="utf-8",
    )
    return run_directory


def _completed_process(
    arguments: list[str],
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess[str]:
    """Build one text subprocess result for a mocked scheduler call.

    Parameters
    ----------
    arguments : list[str]
        Exact dimensionless command tokens.
    stdout, stderr : str, default=""
        Synthetic scheduler text streams.
    returncode : int, default=0
        Synthetic process return status.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed command record accepted by the production subprocess seam.
    """
    return subprocess.CompletedProcess(arguments, returncode, stdout, stderr)


def _temporary_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Create a committed minimal repository containing the WP11 wrapper.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned parent directory.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Repository root and copied wrapper path.
    """
    repository = tmp_path / "repository"
    wrapper = repository / "src" / "shell_scripts" / WRAPPER.name
    wrapper.parent.mkdir(parents=True)
    shutil.copy2(WRAPPER, wrapper)
    required_files = (
        "src/__init__.py",
        "src/neural_analysis/__init__.py",
        "src/neural_analysis/session_metadata.py",
        "src/neural_analysis/spike_behavior/__init__.py",
        "src/neural_analysis/spike_behavior/loading.py",
        "src/neural_analysis/population/__init__.py",
        "src/neural_analysis/population/pca.py",
        "src/behavior_analysis/__init__.py",
        "src/behavior_analysis/project_utils.py",
        "src/neural_analysis/task_decoding/__init__.py",
        "src/neural_analysis/task_decoding/run_session.py",
        "src/neural_analysis/task_decoding/slurm.py",
        "pyproject.toml",
        "uv.lock",
    )
    for relative_path in required_files:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative_path}\n", encoding="ascii")
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "slurm-test@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Slurm Test"),
        cwd=repository,
        check=True,
    )
    subprocess.run(("git", "add", "."), cwd=repository, check=True)
    subprocess.run(
        ("git", "commit", "-q", "-m", "wrapper fixture"),
        cwd=repository,
        check=True,
    )
    return repository, wrapper


def _fake_uv(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake uv executable that records argv and thread limits.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory receiving the fake executable.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Fake-bin directory and NUL-delimited capture file.
    """
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    capture = tmp_path / "uv-capture.bin"
    executable = fake_bin / "uv"
    executable.write_text(
        """#!/bin/bash
set -u
if [[ "${1-}" == "--version" ]]; then
    echo "uv 0.test"
    exit 0
fi
{
    printf 'cwd\\0%s\\0' "$PWD"
    printf 'omp\\0%s\\0' "${OMP_NUM_THREADS-}"
    printf 'mkl\\0%s\\0' "${MKL_NUM_THREADS-}"
    printf 'openblas\\0%s\\0' "${OPENBLAS_NUM_THREADS-}"
    printf 'pybytecode\\0%s\\0' "${PYTHONDONTWRITEBYTECODE-}"
    for value in "$@"; do
        printf 'arg\\0%s\\0' "$value"
    done
} > "$TASK_DECODING_SLURM_TEST_CAPTURE"
exit "${TASK_DECODING_SLURM_TEST_EXIT_CODE:-0}"
""",
        encoding="ascii",
    )
    executable.chmod(0o755)
    return fake_bin, capture


def _wrapper_environment(
    fake_bin: Path,
    capture: Path,
    repository: Path,
    *,
    scheduled: bool,
) -> dict[str, str]:
    """Return a fake login-node or scheduled-job environment.

    Parameters
    ----------
    fake_bin, capture, repository : pathlib.Path
        Fake executable directory, capture file, and exact submission checkout.
    scheduled : bool
        If true, include one-task Slurm allocation metadata.

    Returns
    -------
    dict[str, str]
        Complete subprocess environment with one-thread execution controls.
    """
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "TASK_DECODING_SLURM_TEST_CAPTURE": str(capture),
        }
    )
    if scheduled:
        environment.update(
            {
                "SLURM_JOB_ID": "12345",
                "SLURM_JOB_NAME": "task_decoding",
                "SLURM_CPUS_PER_TASK": "1",
                "SLURM_SUBMIT_DIR": str(repository),
            }
        )
    else:
        for name in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_CPUS_PER_TASK",
            "SLURM_SUBMIT_DIR",
        ):
            environment.pop(name, None)
    return environment


def _captured_values(path: Path) -> list[str]:
    """Decode ordered fake-uv argument values.

    Parameters
    ----------
    path : pathlib.Path
        Existing NUL-delimited fake-uv capture.

    Returns
    -------
    list[str]
        Values for alternating field/value entries.
    """
    fields = path.read_bytes().decode("ascii").rstrip("\0").split("\0")
    assert fields[:10] == [
        "cwd",
        fields[1],
        "omp",
        "1",
        "mkl",
        "1",
        "openblas",
        "1",
        "pybytecode",
        "1",
    ]
    assert fields[10::2] == ["arg"] * len(fields[11::2])
    return fields[11::2]


def test_wp11_wrapper_has_exact_resources_and_valid_bash() -> None:
    """The wrapper freezes the approved one-CPU, 3-GiB, 48-hour request."""
    syntax = subprocess.run(
        ("bash", "-n", str(WRAPPER)),
        capture_output=True,
        text=True,
        check=False,
    )
    text = WRAPPER.read_text(encoding="ascii")

    assert syntax.returncode == 0, syntax.stderr
    for directive in (
        "#SBATCH --partition=unlimited",
        "#SBATCH --job-name=task_decoding",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=1",
        "#SBATCH --mem=3G",
        "#SBATCH --time=2-00:00:00",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --output=/gs/gsfs0/users/mchin1/logs/task_decoding_%j.log",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=matthew.chin@einsteinmed.edu",
    ):
        assert directive in text
    assert "uv run --frozen --no-sync --offline" in text
    assert "OMP_NUM_THREADS=1" in text
    assert "MKL_NUM_THREADS=1" in text
    assert "OPENBLAS_NUM_THREADS=1" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "solver" not in text and "max_iter" not in text
    assert "conda activate" not in text
    assert "source ~/.bashrc" not in text


def test_wp11_python_resource_contract_matches_wrapper() -> None:
    """Submission receipts and static SBATCH fields share one exact contract."""
    slurm = _slurm_module()
    assert slurm.REQUESTED_RESOURCES == EXPECTED_RESOURCES


@pytest.mark.parametrize(
    ("mode", "arguments"),
    (
        ("submit-new", ("--config", "/cluster/session with spaces/config.json")),
        ("submit-resume", ("--run-directory", "/cluster/session with spaces/exact-run")),
        ("status", ("--run-directory", "/cluster/session with spaces/exact-run")),
    ),
)
def test_wp11_login_wrapper_forwards_exact_public_mode(
    tmp_path: Path,
    mode: str,
    arguments: tuple[str, ...],
) -> None:
    """Login-node modes preserve spaced arguments and use the operational module."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _wrapper_environment(
        fake_bin,
        capture,
        repository,
        scheduled=False,
    )

    completed = subprocess.run(
        ("bash", str(wrapper), mode, *arguments),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert _captured_values(capture) == [
        "run",
        "--frozen",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        SLURM_MODULE,
        mode,
        *arguments,
    ]


def test_wp11_scheduled_wrapper_executes_only_exact_prepared_run(tmp_path: Path) -> None:
    """A scheduled private branch cannot submit and forwards one exact run."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _wrapper_environment(
        fake_bin,
        capture,
        repository,
        scheduled=True,
    )
    run_directory = "/cluster/session with spaces/exact-run"

    completed = subprocess.run(
        (
            "bash",
            str(wrapper),
            "_execute-prepared",
            "--run-directory",
            run_directory,
        ),
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert _captured_values(capture) == [
        "run",
        "--frozen",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        RUNNER_MODULE,
        "_execute-prepared",
        "--run-directory",
        run_directory,
    ]


def test_wp11_wrapper_rejects_dirty_or_wrong_repository_before_python(tmp_path: Path) -> None:
    """Wrong roots and relevant dirty/untracked source fail before fake uv."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _wrapper_environment(
        fake_bin,
        capture,
        repository,
        scheduled=False,
    )

    (repository / "pyproject.toml").write_text("dirty\n", encoding="ascii")
    dirty = subprocess.run(
        ("bash", str(wrapper), "status", "--run-directory", "/run"),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dirty.returncode != 0
    assert "tracked" in dirty.stderr.lower()
    assert not capture.exists()

    subprocess.run(
        ("git", "restore", "pyproject.toml"),
        cwd=repository,
        check=True,
    )
    untracked_python = repository / "src/neural_analysis/task_decoding/unreviewed.py"
    untracked_python.write_text("raise RuntimeError\n", encoding="ascii")
    untracked = subprocess.run(
        ("bash", str(wrapper), "status", "--run-directory", "/run"),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert untracked.returncode != 0
    assert "untracked" in untracked.stderr.lower()
    assert not capture.exists()

    untracked_python.unlink()
    nested = repository / "src"
    wrong_root = subprocess.run(
        ("bash", str(wrapper), "status", "--run-directory", "/run"),
        cwd=nested,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert wrong_root.returncode != 0
    assert "root" in wrong_root.stderr.lower()
    assert not capture.exists()


def test_wp11_login_wrapper_allows_unrelated_untracked_file_and_package_bytecode(
    tmp_path: Path,
) -> None:
    """Unrelated files and generated package bytecode do not mimic source."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _wrapper_environment(
        fake_bin,
        capture,
        repository,
        scheduled=False,
    )
    (repository / "notes.tmp").write_text("unrelated\n", encoding="ascii")
    bytecode = repository / "src/neural_analysis/task_decoding/__pycache__/slurm.pyc"
    bytecode.parent.mkdir()
    bytecode.write_bytes(b"generated-bytecode")

    completed = subprocess.run(
        ("bash", str(wrapper), "status", "--run-directory", "/run"),
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert capture.is_file()


def test_wp11_submit_new_dry_runs_prepares_slurm_and_submits_exact_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """New submission uses one dry run and the shared immutable preparation seam."""
    slurm = _slurm_module()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="ascii")
    prepared = _prepared_run(tmp_path)
    calls: list[object] = []

    class FakePipeline:
        """Record only the bounded planning and preparation calls."""

        @staticmethod
        def plan_task_decoding_session(path: Path) -> dict[str, int]:
            calls.append(("dry-run", Path(path)))
            return {"tensor_allocation_bytes": 96_951_680}

        @staticmethod
        def prepare_task_decoding_run(path: Path, rerun: bool, execution_mode: str) -> Path:
            calls.append(("prepare", Path(path), rerun, execution_mode))
            return prepared

        @staticmethod
        def _validate_complete_prepared_run(_directory: Path) -> bool:
            return False

    monkeypatch.setattr(slurm, "_pipeline_module", lambda: FakePipeline)
    monkeypatch.setattr(
        slurm,
        "submit_prepared_run",
        lambda run_directory: calls.append(("submit", Path(run_directory)))
        or {"job_id": "12345"},
    )

    receipt = slurm.submit_new(config_path)

    assert receipt["job_id"] == "12345"
    assert calls == [
        ("dry-run", config_path),
        ("prepare", config_path, False, "slurm"),
        ("submit", prepared),
    ]


def test_wp11_submit_resume_uses_only_exact_saved_directory(monkeypatch, tmp_path: Path) -> None:
    """Resume never searches a parent for the latest run or changes settings."""
    slurm = _slurm_module()
    exact = _prepared_run(tmp_path)
    sibling = exact.parent / "newer-run"
    sibling.mkdir()
    calls: list[Path] = []
    monkeypatch.setattr(
        slurm,
        "submit_prepared_run",
        lambda run_directory: calls.append(Path(run_directory)) or {"job_id": "44"},
    )

    receipt = slurm.submit_resume(exact)

    assert receipt["job_id"] == "44"
    assert calls == [exact]


def test_wp11_submission_receipt_cannot_regress_immediately_running_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """A fast job owns state after sbatch while the submitter writes only its receipt."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    commands: list[list[str]] = []

    def immediate_start(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Observe pending state, then model a job that publishes running state."""
        commands.append(arguments)
        state = _read_json(run_directory / "run_state.json")
        assert state["lifecycle"] == "submission-pending"
        state["lifecycle"] = "running"
        state["current_stage"] = "modeling"
        _write_json(run_directory / "run_state.json", state)
        return _completed_process(arguments, stdout="12345;cluster\n")

    monkeypatch.setattr(slurm, "_run_command", immediate_start)
    monkeypatch.setattr(slurm, "_repository_commit", lambda: "approved-commit")
    receipt = slurm.submit_prepared_run(run_directory)

    assert len(commands) == 1
    assert commands[0][0:2] == ["sbatch", "--parsable"]
    assert commands[0][-3:] == ["_execute-prepared", "--run-directory", str(run_directory)]
    assert receipt["job_id"] == "12345"
    assert receipt["requested_resources"] == EXPECTED_RESOURCES
    assert receipt["repository_commit"] == "approved-commit"
    assert receipt["scheduler_log_path"].endswith("task_decoding_12345.log")
    assert receipt["status_command"] == (
        "bash src/shell_scripts/task_variable_decoding_slurm.sh status "
        f"--run-directory '{run_directory}'"
    )
    assert receipt["resume_command"] == (
        "bash src/shell_scripts/task_variable_decoding_slurm.sh submit-resume "
        f"--run-directory '{run_directory}'"
    )
    assert _read_json(run_directory / "run_state.json")["lifecycle"] == "running"
    assert _read_json(run_directory / "slurm_submission.json") == receipt


def test_wp11_submission_failure_is_durable_and_never_retried(monkeypatch, tmp_path: Path) -> None:
    """One rejected sbatch marks failure and performs no automatic retry."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    calls = 0

    def reject(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Return one scheduler failure while counting exact invocation count."""
        nonlocal calls
        calls += 1
        return _completed_process(arguments, stderr="scheduler unavailable", returncode=1)

    monkeypatch.setattr(slurm, "_run_command", reject)
    monkeypatch.setattr(slurm, "_repository_commit", lambda: "approved-commit")

    with pytest.raises(RuntimeError, match="scheduler unavailable|sbatch"):
        slurm.submit_prepared_run(run_directory)

    state = _read_json(run_directory / "run_state.json")
    assert state["lifecycle"] == "failed"
    assert state["last_error"]
    assert calls == 1
    assert not (run_directory / "slurm_submission.json").exists()


def test_wp11_status_uses_one_sacct_query_and_never_writes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Status combines durable state with one bounded scheduler query read-only."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    submission = {
        "job_id": "12345",
        "requested_resources": EXPECTED_RESOURCES,
        "scheduler_log_path": "/gs/gsfs0/users/mchin1/logs/task_decoding_12345.log",
    }
    _write_json(run_directory / "slurm_submission.json", submission)
    before = {path: path.read_bytes() for path in run_directory.iterdir() if path.is_file()}
    commands: list[list[str]] = []

    class FakePipeline:
        """Return one immutable pipeline status mapping."""

        @staticmethod
        def inspect_task_decoding_status(directory: Path, verify_results: bool = False):
            assert Path(directory) == run_directory
            assert verify_results is False
            return {"lifecycle": "running", "current_stage": "modeling"}

    def accounting(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Return one parsable job-level sacct record."""
        commands.append(arguments)
        return _completed_process(
            arguments,
            stdout="12345|COMPLETED|01:45:53|01:45:40|1|956M|3G|05:00:00|0:0\n",
        )

    monkeypatch.setattr(slurm, "_pipeline_module", lambda: FakePipeline)
    monkeypatch.setattr(slurm, "_run_command", accounting)

    status = slurm.inspect_status(run_directory)

    assert len(commands) == 1
    assert commands[0][0] == "sacct"
    assert status["scheduler"] == {
        "State": "COMPLETED",
        "Elapsed": "01:45:53",
        "TotalCPU": "01:45:40",
        "AllocCPUS": "1",
        "MaxRSS": "956M",
        "ReqMem": "3G",
        "Timelimit": "05:00:00",
        "ExitCode": "0:0",
    }
    assert "scheduler_error" not in status
    after = {path: path.read_bytes() for path in run_directory.iterdir() if path.is_file()}
    assert after == before


@pytest.mark.parametrize("accounting_outcome", ("empty", "failed"))
def test_status_preserves_durable_state_when_accounting_is_unavailable(
    monkeypatch,
    tmp_path: Path,
    accounting_outcome: str,
) -> None:
    """Empty or failed sacct remains read-only and does not hide pipeline state."""
    slurm = _slurm_module()
    run_directory = _prepared_run(tmp_path)
    _write_json(
        run_directory / "slurm_submission.json",
        {
            "job_id": "30984881",
            "requested_resources": EXPECTED_RESOURCES,
            "scheduler_log_path": "/tmp/task_decoding_30984881.log",
        },
    )
    before = {path: path.read_bytes() for path in run_directory.iterdir() if path.is_file()}
    commands: list[list[str]] = []

    class FakePipeline:
        """Return a completed durable status without reading scientific inputs."""

        @staticmethod
        def inspect_task_decoding_status(directory: Path, verify_results: bool = False):
            """Return the immutable pipeline status for the requested run directory."""
            assert Path(directory) == run_directory
            assert verify_results is False
            return {
                "lifecycle": "complete",
                "current_stage": "complete",
                "final_results_published": True,
            }

    def unavailable_accounting(arguments: list[str]) -> subprocess.CompletedProcess[str]:
        """Return one bounded empty or failed accounting response."""
        commands.append(arguments)
        if accounting_outcome == "empty":
            return _completed_process(arguments, stdout="")
        return _completed_process(
            arguments,
            stderr="Slurm accounting temporarily unavailable",
            returncode=1,
        )

    monkeypatch.setattr(slurm, "_pipeline_module", lambda: FakePipeline)
    monkeypatch.setattr(slurm, "_run_command", unavailable_accounting)

    status = slurm.inspect_status(run_directory)

    assert status["lifecycle"] == "complete"
    assert status["current_stage"] == "complete"
    assert status["final_results_published"] is True
    assert status["scheduler"] is None
    if accounting_outcome == "empty":
        assert "no job-level record" in status["scheduler_error"]
    else:
        assert "temporarily unavailable" in status["scheduler_error"]
    assert len(commands) == 1
    assert commands[0][0] == "sacct"
    after = {path: path.read_bytes() for path in run_directory.iterdir() if path.is_file()}
    assert after == before


def test_wp11_effective_memory_uses_lower_memavailable_and_cgroup_limit(tmp_path: Path) -> None:
    """Scheduled execution never mistakes whole-node memory for its allocation."""
    slurm = _slurm_module()
    meminfo = tmp_path / "meminfo"
    cgroup = tmp_path / "memory.max"
    meminfo.write_text("MemAvailable:       4194304 kB\n", encoding="ascii")
    cgroup.write_text(str(3 * 1024**3) + "\n", encoding="ascii")

    assert slurm.effective_memory_budget_bytes(
        meminfo_path=meminfo,
        cgroup_limit_paths=(cgroup,),
    ) == 3 * 1024**3


@pytest.mark.parametrize("tensor_bytes", (EXPECTED_RESOURCES["memory_bytes"] // 2 + 1,))
def test_wp11_login_memory_guard_rejects_tensor_above_half_request(
    monkeypatch,
    tmp_path: Path,
    tensor_bytes: int,
) -> None:
    """An oversized dry-run tensor stops before preparation or sbatch."""
    slurm = _slurm_module()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}\n", encoding="ascii")

    class FakePipeline:
        """Expose only the oversized dry-run plan."""

        @staticmethod
        def plan_task_decoding_session(_path: Path) -> dict[str, int]:
            return {"tensor_allocation_bytes": tensor_bytes}

        @staticmethod
        def prepare_task_decoding_run(*_args: object, **_kwargs: object) -> Path:
            raise AssertionError("oversized tensor reached preparation")

    monkeypatch.setattr(slurm, "_pipeline_module", lambda: FakePipeline)
    monkeypatch.setattr(
        slurm,
        "submit_prepared_run",
        lambda _directory: (_ for _ in ()).throw(
            AssertionError("oversized tensor reached sbatch")
        ),
    )

    with pytest.raises(ValueError, match="50%|memory"):
        slurm.submit_new(config_path)


def test_wp11_unknown_cli_mode_fails_without_scheduler(monkeypatch) -> None:
    """The operational module rejects unknown modes before any command call."""
    slurm = _slurm_module()
    monkeypatch.setattr(
        slurm,
        "_run_command",
        lambda _arguments: (_ for _ in ()).throw(AssertionError("scheduler called")),
    )
    with pytest.raises(SystemExit):
        slurm.main(["unknown-mode"])
