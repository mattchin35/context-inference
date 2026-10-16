"""Contracts for the thin Slurm LFP-preparation wrapper."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_TRACKED_WRAPPER = _REPOSITORY_ROOT / "src" / "shell_scripts" / "hpc_lfp_prepare.sh"


def _temporary_repository(tmp_path: Path) -> tuple[Path, Path]:
    """Create a committed minimal checkout and return its root and wrapper.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest-managed directory.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Repository root and copied wrapper. No scientific arrays are created.
    """
    repository = tmp_path / "repository"
    wrapper = repository / "src" / "shell_scripts" / _TRACKED_WRAPPER.name
    setup_module = repository / "src" / "neural_analysis" / "single_session_setup.py"
    wrapper.parent.mkdir(parents=True)
    setup_module.parent.mkdir(parents=True)
    shutil.copy2(_TRACKED_WRAPPER, wrapper)
    setup_module.write_text('"""Synthetic setup module."""\n', encoding="ascii")
    (repository / "pyproject.toml").write_text(
        "[project]\nname = 'wrapper-test'\nversion = '0.0.0'\n",
        encoding="ascii",
    )
    (repository / "uv.lock").write_text("version = 1\n", encoding="ascii")
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
    """Create a fake ``uv`` and return its bin directory and capture path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed directory receiving scalar test artifacts.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Executable directory and NUL-delimited invocation capture.
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
} > "$LFP_PREPARE_TEST_CAPTURE"
exit "${LFP_PREPARE_TEST_EXIT_CODE:-0}"
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
    """Return one synthetic eight-CPU Slurm environment.

    Parameters
    ----------
    fake_bin, capture, repository : pathlib.Path
        Fake executable directory, capture path, and submission checkout.

    Returns
    -------
    dict[str, str]
        Complete scalar process environment with no scientific data.
    """
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}{os.pathsep}{environment['PATH']}",
            "LFP_PREPARE_TEST_CAPTURE": str(capture),
            "SLURM_CPUS_PER_TASK": "8",
            "SLURM_JOB_ID": "24680",
            "SLURM_JOB_NAME": "lfp_prepare",
            "SLURM_SUBMIT_DIR": str(repository),
        }
    )
    return environment


def _run_wrapper(
    wrapper: Path,
    environment: dict[str, str],
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    """Run the copied wrapper and return its text streams and exit status.

    Parameters
    ----------
    wrapper : pathlib.Path
        Wrapper inside the temporary committed checkout.
    environment : dict[str, str]
        Synthetic Slurm environment.
    *arguments : str
        Exact preparation CLI tokens to forward.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Bash result containing scalar process metadata only.
    """
    return subprocess.run(
        ("bash", str(wrapper), *arguments),
        cwd=wrapper.parents[2],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def _capture_fields(path: Path) -> list[str]:
    """Decode the fake ``uv`` invocation into alternating names and values.

    Parameters
    ----------
    path : pathlib.Path
        Existing NUL-delimited ASCII capture.

    Returns
    -------
    list[str]
        Ordered scalar fields with the trailing delimiter removed.
    """
    return path.read_bytes().decode("ascii").rstrip("\0").split("\0")


def test_hpc_lfp_prepare_has_reviewed_resources_and_valid_bash() -> None:
    """The wrapper owns the normal-partition 24-hour resource request."""
    syntax = subprocess.run(
        ("bash", "-n", str(_TRACKED_WRAPPER)),
        capture_output=True,
        text=True,
        check=False,
    )
    text = _TRACKED_WRAPPER.read_text(encoding="ascii")

    assert syntax.returncode == 0, syntax.stderr
    for directive in (
        "#SBATCH --partition=normal",
        "#SBATCH --ntasks=1",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=32G",
        "#SBATCH --time=24:00:00",
        "#SBATCH --signal=B:TERM@300",
    ):
        assert directive in text
    assert any(
        line.startswith("#SBATCH --output=") and "%j" in line
        for line in text.splitlines()
    )


def test_hpc_lfp_prepare_forwards_exact_metadata_path_and_environment(
    tmp_path: Path,
) -> None:
    """The wrapper preserves path spaces and invokes only existing preparation."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    arguments = ("--session-metadata", "/cluster/session with space/neural_session.json")

    result = _run_wrapper(wrapper, environment, *arguments)

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
    assert fields[9::2] == [
        "run",
        "--frozen",
        "--no-sync",
        "--offline",
        "python",
        "-m",
        "src.neural_analysis.single_session_setup",
        "prepare",
        *arguments,
    ]
    assert "SLURM_JOB_ID=24680" in result.stdout
    assert "Preparation arguments:" in result.stdout


@pytest.mark.parametrize("cpu_value", (None, "invalid", "4"))
def test_hpc_lfp_prepare_rejects_invalid_allocation_before_python(
    tmp_path: Path,
    cpu_value: str | None,
) -> None:
    """Missing, malformed, or non-reviewed CPU allocations fail before ``uv``."""
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
        "--session-metadata",
        "/cluster/session/neural_session.json",
    )

    assert result.returncode != 0
    assert "SLURM_CPUS_PER_TASK" in result.stderr
    assert not capture.exists()


def test_hpc_lfp_prepare_rejects_dirty_checkout_and_propagates_uv_status(
    tmp_path: Path,
) -> None:
    """Tracked changes stop before computation and ``uv`` failures pass through."""
    repository, wrapper = _temporary_repository(tmp_path)
    fake_bin, capture = _fake_uv(tmp_path)
    environment = _environment(fake_bin, capture, repository)
    (repository / "pyproject.toml").write_text("dirty\n", encoding="ascii")

    dirty = _run_wrapper(
        wrapper,
        environment,
        "--session-metadata",
        "/cluster/session/neural_session.json",
    )

    assert dirty.returncode != 0
    assert "tracked" in dirty.stderr.lower()
    assert not capture.exists()

    subprocess.run(("git", "restore", "pyproject.toml"), cwd=repository, check=True)
    environment["LFP_PREPARE_TEST_EXIT_CODE"] = "37"
    failed = _run_wrapper(
        wrapper,
        environment,
        "--session-metadata",
        "/cluster/session/neural_session.json",
    )

    assert failed.returncode == 37


def test_cluster_documentation_uses_wrapper_not_login_node_preparation() -> None:
    """User-facing cluster instructions submit preparation through Slurm."""
    setup_guide = (
        _REPOSITORY_ROOT / "src" / "neural_analysis" / "SINGLE_SESSION_SETUP.md"
    ).read_text(encoding="ascii")
    readme = (
        _REPOSITORY_ROOT / "src" / "neural_analysis" / "README.md"
    ).read_text(encoding="ascii")

    command = "sbatch src/shell_scripts/hpc_lfp_prepare.sh"
    assert command in setup_guide
    assert command in readme
    assert "login node" in setup_guide.lower()
