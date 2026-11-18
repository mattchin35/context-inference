"""RED contracts for the one-shot inter-regional regression Slurm wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRACKED_WRAPPER = (
    _REPOSITORY_ROOT
    / "src"
    / "shell_scripts"
    / "interregional_regression_slurm.sh"
)


def _temporary_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Create a committed minimal checkout and return its root and wrapper.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest-managed directory.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Repository root and copied wrapper path. Paths are filesystem
        coordinates; no scientific arrays or physical units are involved.
    """
    repository = tmp_path / "repository"
    wrapper = repository / "src" / "shell_scripts" / _TRACKED_WRAPPER.name
    wrapper.parent.mkdir(parents=True)
    shutil.copy2(_TRACKED_WRAPPER, wrapper)
    required_files = (
        "pyproject.toml",
        "uv.lock",
        "src/__init__.py",
        "src/neural_analysis/__init__.py",
        "src/neural_analysis/interregional/__init__.py",
        "src/neural_analysis/interregional/run_session.py",
    )
    for relative_path in required_files:
        path = repository / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# launcher test fixture\n", encoding="ascii")
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
    """Create a fake uv command and return its bin directory and capture path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed directory receiving executable artifacts.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Fake-bin directory and NUL-delimited invocation capture path. Captured
        thread values are integer counts, not physical measurements.
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
} > "$INTERREGIONAL_TEST_CAPTURE"
echo "fake uv stdout"
echo "fake uv stderr" >&2
if [[ -n "${INTERREGIONAL_TEST_SIGNAL_FILE-}" ]]; then
    trap 'printf TERM > "$INTERREGIONAL_TEST_SIGNAL_FILE"; exit 143' TERM
    : > "$INTERREGIONAL_TEST_READY_FILE"
    while true; do sleep 0.05; done
fi
exit "${INTERREGIONAL_TEST_EXIT_CODE:-0}"
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
    """Return a complete synthetic eight-CPU Slurm environment.

    Parameters
    ----------
    fake_bin, capture, repository : pathlib.Path
        Fake executable directory, capture file, and exact submission root.

    Returns
    -------
    dict[str, str]
        Child-process environment. CPU fields are integer resource counts.
    """
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "INTERREGIONAL_TEST_CAPTURE": str(capture),
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_JOB_ID": "24680",
            "SLURM_JOB_NAME": "interregional_poisson",
            "SLURM_SUBMIT_DIR": str(repository),
        }
    )
    return environment


def _run_wrapper(
    wrapper: Path,
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Execute one wrapper and return its text streams and process status.

    Parameters
    ----------
    wrapper : pathlib.Path
        Wrapper path, either in a checkout or a simulated Slurm spool.
    environment : dict[str, str]
        Complete child environment with synthetic Slurm metadata.
    *arguments : str
        Exact CLI token sequence to forward.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Bash result with text stdout/stderr and integer exit status.
    """
    return subprocess.run(
        ("bash", str(wrapper), *arguments),
        cwd=wrapper.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _capture_fields(path: Path) -> list[str]:
    """Decode a fake-uv capture into alternating field and value strings.

    Parameters
    ----------
    path : pathlib.Path
        Existing NUL-delimited ASCII capture path.

    Returns
    -------
    list[str]
        Ordered scalar fields with the final empty delimiter removed.
    """
    return path.read_bytes().decode("ascii").rstrip("\0").split("\0")


def test_interregional_slurm_resources_and_bash_syntax() -> None:
    """The tracked wrapper requests the approved one-shot cluster resources."""
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
        "#SBATCH --job-name=interregional_poisson",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=72:00:00",
        "#SBATCH --signal=B:TERM@300",
        "#SBATCH --output=/gs/gsfs0/users/mchin1/logs/interregional_regression_%j.log",
        "#SBATCH --mail-type=ALL",
        "#SBATCH --mail-user=matthew.chin@einsteinmed.edu",
    ):
        assert directive in text


def test_interregional_slurm_forwards_exact_arguments_and_environment(
    tmp_path: Path,
) -> None:
    """Arguments, cwd, thread counts, and the frozen offline uv call are exact."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    arguments = (
        "new",
        "--config",
        "/cluster/session with space/interregional config.json",
        "--rerun",
    )

    result = _run_wrapper(wrapper, environment, *arguments)

    assert result.returncode == 0, result.stderr
    fields = _capture_fields(capture)
    assert fields[:8] == [
        "cwd",
        str(repository.resolve()),
        "omp",
        "8",
        "mkl",
        "8",
        "openblas",
        "8",
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
        "src.neural_analysis.interregional.run_session",
        *arguments,
    ]
    assert "fake uv stdout" in result.stdout
    assert "fake uv stderr" in result.stderr
    for label in (
        "Cluster host:",
        "UTC start:",
        "Repository root:",
        "Repository commit:",
        "SLURM_JOB_ID=24680",
        "Launcher arguments:",
        "uv 0.test",
    ):
        assert label in result.stdout


@pytest.mark.parametrize("cpu_value", (None, "four", "4"))
def test_interregional_slurm_rejects_invalid_cpu_contract(
    tmp_path: Path,
    cpu_value: str | None,
) -> None:
    """Missing, nonnumeric, or non-eight CPU allocations fail before uv."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    if cpu_value is None:
        environment.pop("SLURM_CPUS_PER_TASK")
    else:
        environment["SLURM_CPUS_PER_TASK"] = cpu_value

    result = _run_wrapper(wrapper, environment, "new", "--config", "/config.json")

    assert result.returncode != 0
    assert "SLURM_CPUS_PER_TASK" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize("submit_directory_case", ("missing", "absent", "subdirectory"))
def test_interregional_slurm_requires_exact_submission_root(
    tmp_path: Path,
    submit_directory_case: str,
) -> None:
    """Missing, nonexistent, and non-root submission paths fail before uv."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    if submit_directory_case == "missing":
        environment.pop("SLURM_SUBMIT_DIR")
    elif submit_directory_case == "absent":
        environment["SLURM_SUBMIT_DIR"] = str(tmp_path / "absent")
    else:
        environment["SLURM_SUBMIT_DIR"] = str(repository / "src")

    result = _run_wrapper(wrapper, environment, "new", "--config", "/config.json")

    assert result.returncode != 0
    assert "repository" in result.stderr.lower()
    assert not capture.exists()


def test_interregional_slurm_rejects_dirty_repository(tmp_path: Path) -> None:
    """A tracked modification in the submitted checkout fails before uv."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    (repository / "pyproject.toml").write_text("dirty\n", encoding="ascii")

    result = _run_wrapper(wrapper, environment, "new", "--config", "/config.json")

    assert result.returncode != 0
    assert "tracked" in result.stderr.lower()
    assert not capture.exists()


def test_interregional_slurm_requires_tracked_execution_dependencies(
    tmp_path: Path,
) -> None:
    """An untracked runner module fails before uv even when tracked state is clean."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    relative_runner = "src/neural_analysis/interregional/run_session.py"
    subprocess.run(
        ("git", "rm", "--cached", "-q", relative_runner),
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ("git", "commit", "-q", "-m", "remove runner from index"),
        cwd=repository,
        check=True,
    )

    result = _run_wrapper(wrapper, environment, "new", "--config", "/config.json")

    assert result.returncode != 0
    assert "not tracked" in result.stderr.lower()
    assert not capture.exists()


def test_interregional_slurm_runs_from_spooled_copy(tmp_path: Path) -> None:
    """A Slurm-spooled copy enters the exact submitted repository root."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    spooled_wrapper = tmp_path / "slurm-spool" / "slurm_script"
    spooled_wrapper.parent.mkdir()
    shutil.copy2(wrapper, spooled_wrapper)

    result = _run_wrapper(
        spooled_wrapper,
        environment,
        "new",
        "--config",
        "/cluster/config.json",
    )

    assert result.returncode == 0, result.stderr
    assert _capture_fields(capture)[:2] == ["cwd", str(repository.resolve())]


def test_interregional_slurm_propagates_uv_exit_status(tmp_path: Path) -> None:
    """The wrapper process returns the existing Python runner's exit status."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    environment["INTERREGIONAL_TEST_EXIT_CODE"] = "37"

    result = _run_wrapper(wrapper, environment, "new", "--config", "/config.json")

    assert result.returncode == 37
    assert capture.is_file()


def test_interregional_slurm_exec_forwards_term_to_uv(tmp_path: Path) -> None:
    """TERM reaches the exec-replaced uv process without a wrapper child process."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    ready = tmp_path / "uv-ready"
    signal_record = tmp_path / "uv-signal"
    environment.update(
        {
            "INTERREGIONAL_TEST_READY_FILE": str(ready),
            "INTERREGIONAL_TEST_SIGNAL_FILE": str(signal_record),
        }
    )
    process = subprocess.Popen(
        ("bash", str(wrapper), "new", "--config", "/cluster/config.json"),
        cwd=wrapper.parent,
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
