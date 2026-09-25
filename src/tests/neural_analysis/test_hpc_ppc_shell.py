"""RED contracts for the thin uv/SLURM Spike-phase wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRACKED_WRAPPER = _REPOSITORY_ROOT / "src" / "shell_scripts" / "hpc_ppc.sh"


def _temporary_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Return a committed temporary checkout containing the tracked wrapper.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest-managed directory.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Temporary repository root and copied wrapper path. No project code is
        imported or executed.
    """
    repository = tmp_path / "repository"
    wrapper = repository / "src" / "shell_scripts" / "hpc_ppc.sh"
    launcher = repository / "src" / "neural_analysis" / "lfp_spike_phase_launcher.py"
    wrapper.parent.mkdir(parents=True)
    launcher.parent.mkdir(parents=True)
    shutil.copy2(_TRACKED_WRAPPER, wrapper)
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'wrapper-test'\nversion = '0.0.0'\n",
        encoding="ascii",
    )
    launcher.write_text('"""Synthetic launcher identity file."""\n', encoding="ascii")
    subprocess.run(("git", "init", "-q"), cwd=repository, check=True)
    subprocess.run(
        ("git", "config", "user.email", "wrapper-test@example.invalid"),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "config", "user.name", "Wrapper Test"),
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
    """Return a fake uv executable and its NUL-delimited capture path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed directory receiving executable test artifacts.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Fake-bin directory and capture file path. Capture fields are alternating
        names and values with no interpretation of launcher arguments.
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
    for value in "$@"; do
        printf 'arg\\0%s\\0' "$value"
    done
} > "$PPC_TEST_CAPTURE"
echo "fake uv stdout"
echo "fake uv stderr" >&2
if [[ -n "${PPC_TEST_SIGNAL_FILE-}" ]]; then
    trap 'printf TERM > "$PPC_TEST_SIGNAL_FILE"; exit 143' TERM
    : > "$PPC_TEST_READY_FILE"
    while true; do sleep 0.05; done
fi
exit "${PPC_TEST_EXIT_CODE:-0}"
""",
        encoding="ascii",
    )
    executable.chmod(0o755)
    return fake_bin, capture


def _environment(
    fake_bin: Path,
    capture: Path,
    repository: Path,
) -> dict[str, str]:
    """Return a synthetic eight-CPU Slurm environment using only fake uv.

    Parameters
    ----------
    fake_bin, capture, repository : pathlib.Path
        Fake executable directory, NUL-delimited output path, and exact
        repository represented by Slurm's submission-directory metadata.

    Returns
    -------
    dict[str, str]
        Process environment with no physical-unit values other than the
        categorical eight-CPU allocation.
    """
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "PPC_TEST_CAPTURE": str(capture),
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_JOB_ID": "12345",
            "SLURM_JOB_NAME": "ppc_cluster",
            "SLURM_SUBMIT_DIR": str(repository),
        }
    )
    return environment


def _run_wrapper(
    wrapper: Path,
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Run one copied wrapper and return captured text streams and status.

    Parameters
    ----------
    wrapper : pathlib.Path
        Shell wrapper inside a temporary tracked repository.
    environment : dict[str, str]
        Complete child environment including synthetic Slurm metadata.
    *arguments : str
        Exact launcher tokens forwarded without parsing except worker checks.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed Bash process with text stdout/stderr and integer return code.
    """
    return subprocess.run(
        ("bash", str(wrapper), *arguments),
        cwd=wrapper.parent.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _capture_fields(path: Path) -> list[str]:
    """Return decoded alternating field/value strings from fake uv output.

    Parameters
    ----------
    path : pathlib.Path
        Existing NUL-delimited ASCII capture file.

    Returns
    -------
    list[str]
        Ordered field names and values with the final empty delimiter removed.
    """
    return path.read_bytes().decode("ascii").rstrip("\0").split("\0")


def test_hpc_ppc_has_fixed_reviewed_slurm_resources_and_valid_bash() -> None:
    """Tracked wrapper requests one eight-CPU, 32-GB, 72-hour signaled job."""
    syntax = subprocess.run(
        ("bash", "-n", str(_TRACKED_WRAPPER)),
        capture_output=True,
        text=True,
        check=False,
    )
    text = _TRACKED_WRAPPER.read_text(encoding="ascii")

    assert syntax.returncode == 0, syntax.stderr
    for directive in (
        "#SBATCH --partition=unlimited",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=72:00:00",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --output=/gs/gsfs0/users/mchin1/logs/ppc_cluster_%j.log",
    ):
        assert directive in text
    assert "conda activate" not in text
    assert "source ~/.bashrc" not in text
    assert "#SBATCH -n 16" not in text
    assert "#SBATCH --mem=128gb" not in text


@pytest.mark.parametrize(
    "worker_arguments",
    ((), ("--workers", "8"), ("--workers=8",)),
)
def test_hpc_ppc_forwards_exact_arguments_and_fixed_uv_environment(
    tmp_path: Path,
    worker_arguments: tuple[str, ...],
) -> None:
    """Spaces, flags, cwd, thread limits, and offline uv invocation are exact."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    launcher_arguments = (
        "new",
        "--session-path",
        "/cluster/session with space",
        "--probe",
        "ProbeB",
        "--shuffles",
        "100",
        *worker_arguments,
    )

    result = _run_wrapper(wrapper, environment, *launcher_arguments)

    assert result.returncode == 0, result.stderr
    fields = _capture_fields(capture)
    assert fields[:8] == [
        "cwd",
        str(repository.resolve()),
        "omp",
        "1",
        "mkl",
        "1",
        "openblas",
        "1",
    ]
    forwarded = fields[9::2]
    assert fields[8::2] == ["arg"] * len(forwarded)
    assert forwarded == [
        "run",
        "--frozen",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        "src.neural_analysis.lfp_spike_phase_launcher",
        *launcher_arguments,
    ]
    assert "fake uv stdout" in result.stdout
    assert "fake uv stderr" in result.stderr
    for label in (
        "Cluster host:",
        "UTC start:",
        "Repository commit:",
        "SLURM_JOB_ID=12345",
        "Launcher arguments:",
        "uv 0.test",
    ):
        assert label in result.stdout


def test_hpc_ppc_forwards_metadata_driven_new_request(tmp_path: Path) -> None:
    """The existing Slurm wrapper forwards the metadata entry point unchanged."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    arguments = (
        "new",
        "--session-metadata",
        "/cluster/mouse-z/neural_session.json",
        "--cache-directory",
        "/cluster/mouse-z/processed/summary-cache",
        "--probe",
        "rear-probe",
        "--shuffles",
        "100",
        "--workers",
        "8",
    )

    result = _run_wrapper(wrapper, environment, *arguments)

    assert result.returncode == 0, result.stderr
    fields = _capture_fields(capture)
    assert fields[9::2] == [
        "run",
        "--frozen",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        "src.neural_analysis.lfp_spike_phase_launcher",
        *arguments,
    ]


@pytest.mark.parametrize(
    ("cpu_value", "worker_arguments", "expected_message"),
    (
        (None, (), "SLURM_CPUS_PER_TASK"),
        ("four", (), "SLURM_CPUS_PER_TASK"),
        ("4", (), "eight"),
        ("8", ("--workers", "4"), "--workers"),
        ("8", ("--workers=4",), "--workers"),
        ("8", ("--workers",), "value"),
    ),
)
def test_hpc_ppc_rejects_missing_or_mismatched_cpu_worker_contract(
    tmp_path: Path,
    cpu_value: str | None,
    worker_arguments: tuple[str, ...],
    expected_message: str,
) -> None:
    """Allocation metadata and explicit launcher workers must both equal eight."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    if cpu_value is None:
        environment.pop("SLURM_CPUS_PER_TASK")
    else:
        environment["SLURM_CPUS_PER_TASK"] = cpu_value

    result = _run_wrapper(
        wrapper,
        environment,
        "resume",
        "--run-directory",
        "/cluster/run",
        *worker_arguments,
    )

    assert result.returncode != 0
    assert expected_message in result.stderr
    assert not capture.exists()


def test_hpc_ppc_rejects_missing_or_dirty_repository_before_uv(tmp_path: Path) -> None:
    """An absent or tracked-dirty submission checkout fails before Python."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    (repository / "pyproject.toml").write_text("dirty\n", encoding="ascii")

    dirty = _run_wrapper(wrapper, environment, "resume", "--run-directory", "/run")

    assert dirty.returncode != 0
    assert "tracked" in dirty.stderr.lower()
    assert not capture.exists()

    orphan = tmp_path / "orphan" / "src" / "shell_scripts" / "hpc_ppc.sh"
    orphan.parent.mkdir(parents=True)
    shutil.copy2(_TRACKED_WRAPPER, orphan)
    environment["SLURM_SUBMIT_DIR"] = str(tmp_path / "not-a-repository")
    missing = _run_wrapper(orphan, environment, "resume", "--run-directory", "/run")
    assert missing.returncode != 0
    assert "repository" in missing.stderr.lower()
    assert not capture.exists()


def test_hpc_ppc_requires_slurm_submit_directory(tmp_path: Path) -> None:
    """An otherwise valid direct invocation cannot invent a repository root."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    environment.pop("SLURM_SUBMIT_DIR")

    result = _run_wrapper(wrapper, environment, "resume", "--run-directory", "/run")

    assert result.returncode != 0
    assert "SLURM_SUBMIT_DIR" in result.stderr
    assert not capture.exists()


def test_hpc_ppc_runs_from_slurm_spool_using_exact_submit_repository(
    tmp_path: Path,
) -> None:
    """A spooled script validates and enters Slurm's exact submission root."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    spooled_wrapper = tmp_path / "slurm-spool" / "slurm_script"
    spooled_wrapper.parent.mkdir()
    shutil.copy2(wrapper, spooled_wrapper)

    result = _run_wrapper(
        spooled_wrapper,
        environment,
        "resume",
        "--run-directory",
        "/cluster/exact-run",
    )

    assert result.returncode == 0, result.stderr
    fields = _capture_fields(capture)
    assert fields[:2] == ["cwd", str(repository.resolve())]


def test_hpc_ppc_propagates_uv_exit_status(tmp_path: Path) -> None:
    """The wrapper's process status is the existing Python launcher's status."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    environment["PPC_TEST_EXIT_CODE"] = "37"

    result = _run_wrapper(
        wrapper,
        environment,
        "resume",
        "--run-directory",
        "/cluster/exact-run",
    )

    assert result.returncode == 37
    assert capture.is_file()


def test_hpc_ppc_exec_forwards_term_to_uv(tmp_path: Path) -> None:
    """TERM reaches the exec-replaced uv process for launcher persistence."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    ready = tmp_path / "uv-ready"
    signal_record = tmp_path / "uv-signal"
    environment.update(
        {
            "PPC_TEST_READY_FILE": str(ready),
            "PPC_TEST_SIGNAL_FILE": str(signal_record),
        }
    )
    process = subprocess.Popen(
        (
            "bash",
            str(wrapper),
            "resume",
            "--run-directory",
            "/cluster/exact-run",
        ),
        cwd=wrapper.parent.parent.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 5.0
    while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert ready.is_file(), process.communicate(timeout=2.0)

    process.terminate()
    stdout, stderr = process.communicate(timeout=5.0)

    assert process.returncode == 143, (stdout, stderr)
    assert signal_record.read_text(encoding="ascii") == "TERM"
