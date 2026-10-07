"""One-shot Slurm submission and status operations for task decoding.

This module contains only standard-library orchestration. Scientific planning,
preparation, execution, and result validation remain in the shared pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
from typing import Mapping, Sequence


REQUESTED_RESOURCES = {
    "partition": "unlimited",
    "job_name": "task_decoding",
    "tasks": 1,
    "cpus_per_task": 1,
    "memory": "3G",
    "memory_bytes": 3 * 1024**3,
    "time": "05:00:00",
    "signal": "B:TERM@300",
    "output": "/gs/gsfs0/users/mchin1/logs/task_decoding_%j.log",
    "mail_type": "ALL",
    "mail_user": "matthew.chin@einsteinmed.edu",
}

_WRAPPER = "src/shell_scripts/task_variable_decoding_slurm.sh"
_RUNNER_MODULE = "src.neural_analysis.task_decoding.run_session"
_CGROUP_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)
_SACCT_FIELDS = (
    "JobIDRaw",
    "State",
    "Elapsed",
    "TotalCPU",
    "AllocCPUS",
    "MaxRSS",
    "ReqMem",
    "Timelimit",
    "ExitCode",
)


def _utc_now() -> str:
    """Return one whole-second UTC timestamp.

    Returns
    -------
    str
        ISO-8601 UTC text with a trailing ``Z`` and no physical units.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pipeline_module():
    """Import the shared scientific pipeline only when an operation needs it.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pipeline``.
    """
    return __import__(
        "src.neural_analysis.task_decoding.pipeline",
        fromlist=["pipeline"],
    )


def _run_command(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one bounded scheduler or source-control command.

    Parameters
    ----------
    arguments : list[str]
        Exact command tokens. Values have no physical units.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Completed text-mode process record without automatic retry.
    """
    return subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        timeout=15.0,
    )


def _repository_commit() -> str:
    """Return the exact commit of the current submission checkout.

    Returns
    -------
    str
        Full hexadecimal Git commit identifier.

    Raises
    ------
    RuntimeError
        If Git cannot resolve the current checkout.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not resolve the repository commit.") from error
    commit = completed.stdout.strip()
    if completed.returncode != 0 or not commit:
        detail = completed.stderr.strip() or "git rev-parse failed"
        raise RuntimeError(f"Could not resolve the repository commit: {detail}")
    return commit


def _read_json(path: Path) -> dict[str, object]:
    """Read one required JSON object without modifying its source.

    Parameters
    ----------
    path : pathlib.Path
        Existing UTF-8 JSON sidecar.

    Returns
    -------
    dict[str, object]
        Decoded mapping without array axes or physical-unit conversion.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read required JSON artifact: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Required JSON artifact is not an object: {path.name}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically publish one canonical JSON mapping.

    Parameters
    ----------
    path : pathlib.Path
        Destination below an existing prepared run directory.
    value : mapping[str, object]
        JSON-safe lifecycle or Slurm receipt fields. Byte-valued fields name
        their physical units explicitly.

    Returns
    -------
    None
        Fsyncs a unique sibling temporary and replaces ``path`` atomically.
    """
    temporary: Path | None = None
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException as publication_error:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except BaseException as cleanup_error:
                raise cleanup_error from publication_error
        raise


def _update_state(run_directory: Path, **changes: object) -> dict[str, object]:
    """Apply one atomic lifecycle transition to a prepared run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact immutable prepared run directory.
    **changes : object
        JSON-safe lifecycle, stage, and error fields.

    Returns
    -------
    dict[str, object]
        Complete saved state including a refreshed UTC timestamp.
    """
    state = _read_json(run_directory / "run_state.json")
    state.update(changes)
    state["updated_at"] = _utc_now()
    _write_json(run_directory / "run_state.json", state)
    return state


def _require_slurm_prepared_run(run_directory: Path) -> dict[str, object]:
    """Validate the exact directory and its Slurm execution provenance.

    Parameters
    ----------
    run_directory : pathlib.Path
        Candidate immutable prepared run directory.

    Returns
    -------
    dict[str, object]
        Current durable lifecycle state.

    Raises
    ------
    ValueError
        If the path is not an exact Slurm-prepared run or is already active or
        complete.
    """
    if not run_directory.is_dir():
        raise ValueError("The exact prepared run directory does not exist.")
    execution = _read_json(run_directory / "execution.json")
    if execution.get("mode") != "slurm":
        raise ValueError("The prepared run does not declare Slurm execution mode.")
    state = _read_json(run_directory / "run_state.json")
    lifecycle = state.get("lifecycle")
    if lifecycle in {"submission-pending", "running", "complete"}:
        raise ValueError(f"Prepared run cannot be submitted from lifecycle {lifecycle!r}.")
    return state


def _parse_job_id(stdout: str) -> str:
    """Parse one ``sbatch --parsable`` response.

    Parameters
    ----------
    stdout : str
        Scheduler stdout containing ``job_id`` or ``job_id;cluster``.

    Returns
    -------
    str
        Nonempty scheduler job identifier.
    """
    first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    job_id = first_line.split(";", maxsplit=1)[0].strip()
    if not job_id:
        raise RuntimeError("sbatch returned no job identifier.")
    return job_id


def _validate_login_memory_guard(tensor_bytes: object) -> None:
    """Enforce the reviewed 50-percent tensor/request guard before preparation.

    Parameters
    ----------
    tensor_bytes : object
        Dry-run exact regional tensor allocation in bytes. Boolean, fractional,
        negative, and otherwise malformed values are rejected.

    Returns
    -------
    None
        Returns only when the allocation is at most half the 3-GiB request.

    Raises
    ------
    ValueError
        If the value is malformed or exceeds 50 percent of requested memory.
    """
    if isinstance(tensor_bytes, bool):
        raise ValueError("Dry-run tensor memory must be a nonnegative integer.")
    try:
        normalized = int(tensor_bytes)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Dry-run tensor memory must be a nonnegative integer.") from error
    try:
        is_exact = bool(normalized == tensor_bytes)
    except (TypeError, ValueError) as error:
        raise ValueError("Dry-run tensor memory must be a nonnegative integer.") from error
    if not is_exact or normalized < 0:
        raise ValueError("Dry-run tensor memory must be a nonnegative integer.")
    if normalized * 2 > int(REQUESTED_RESOURCES["memory_bytes"]):
        raise ValueError("Tensor allocation exceeds 50% of requested Slurm memory.")


def submit_prepared_run(run_directory: Path | str) -> dict[str, object]:
    """Submit one exact Slurm-prepared directory exactly once.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact immutable prepared run directory. No latest-run discovery occurs.

    Returns
    -------
    dict[str, object]
        Atomically saved submission receipt with job, commit, resources, log,
        and exact resume/status commands.

    Raises
    ------
    ValueError
        If the directory is not eligible for a Slurm submission.
    RuntimeError
        If the single ``sbatch`` call fails or returns no job identifier.
    """
    directory = Path(run_directory)
    _require_slurm_prepared_run(directory)
    repository_commit = _repository_commit()
    _update_state(
        directory,
        lifecycle="submission-pending",
        current_stage="scheduler",
        last_error="",
    )
    arguments = [
        "sbatch",
        "--parsable",
        f"--export=ALL,TASK_DECODING_EXPECTED_COMMIT={repository_commit}",
        _WRAPPER,
        "_execute-prepared",
        "--run-directory",
        str(directory),
    ]
    try:
        completed = _run_command(arguments)
    except (OSError, subprocess.TimeoutExpired) as error:
        message = f"sbatch failed: {type(error).__name__}: {error}"
        _update_state(directory, lifecycle="failed", current_stage="scheduler", last_error=message)
        raise RuntimeError(message) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown scheduler error"
        message = f"sbatch failed: {detail}"
        _update_state(directory, lifecycle="failed", current_stage="scheduler", last_error=message)
        raise RuntimeError(message)
    try:
        job_id = _parse_job_id(completed.stdout)
    except BaseException as error:
        # Submission succeeded, so do not rewrite lifecycle state: the job may
        # already own it. The absent receipt makes the operational fault clear.
        raise RuntimeError(f"Slurm accepted the job but receipt creation failed: {error}") from error
    quoted_directory = shlex.quote(str(directory))
    receipt: dict[str, object] = {
        "job_id": job_id,
        "submitted_at": _utc_now(),
        "run_directory": str(directory),
        "requested_resources": dict(REQUESTED_RESOURCES),
        "repository_commit": repository_commit,
        "scheduler_log_path": str(REQUESTED_RESOURCES["output"]).replace("%j", job_id),
        "status_command": (
            f"bash {_WRAPPER} status --run-directory {quoted_directory}"
        ),
        "resume_command": (
            f"bash {_WRAPPER} submit-resume --run-directory {quoted_directory}"
        ),
    }
    _write_json(directory / "slurm_submission.json", receipt)
    return receipt


def submit_new(config_path: Path | str) -> dict[str, object]:
    """Dry-run, prepare, and submit one new scientific configuration.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Existing task-decoding JSON configuration on cluster-visible storage.

    Returns
    -------
    dict[str, object]
        Submission receipt, or a small complete-run record when immutable
        preparation discovers an already completed matching result.

    Raises
    ------
    ValueError
        If exact tensor allocation exceeds 50 percent of the 3-GiB request.
    """
    path = Path(config_path)
    pipeline = _pipeline_module()
    plan = pipeline.plan_task_decoding_session(path)
    _validate_login_memory_guard(plan.get("tensor_allocation_bytes"))
    prepared = Path(pipeline.prepare_task_decoding_run(path, False, "slurm"))
    if pipeline._validate_complete_prepared_run(prepared):
        return {
            "job_id": "",
            "run_directory": str(prepared),
            "already_complete": True,
        }
    return submit_prepared_run(prepared)


def submit_resume(run_directory: Path | str) -> dict[str, object]:
    """Submit only the exact saved prepared directory supplied by the caller.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact immutable run directory; parents and siblings are not searched.

    Returns
    -------
    dict[str, object]
        New Slurm submission receipt.
    """
    return submit_prepared_run(Path(run_directory))


def _read_mem_available_bytes(path: Path) -> int | None:
    """Read Linux ``MemAvailable`` from one proc-style file.

    Parameters
    ----------
    path : pathlib.Path
        Text file whose ``MemAvailable`` value is expressed in KiB.

    Returns
    -------
    int or None
        Available bytes, or None when the value is absent or unreadable.
    """
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] == "MemAvailable:":
            try:
                kibibytes = int(fields[1])
            except ValueError:
                return None
            return kibibytes * 1024 if kibibytes > 0 else None
    return None


def _read_cgroup_limit_bytes(paths: Sequence[Path]) -> int | None:
    """Return the lowest finite positive cgroup memory limit.

    Parameters
    ----------
    paths : sequence[pathlib.Path]
        Candidate cgroup-v2/v1 byte-limit files.

    Returns
    -------
    int or None
        Lowest finite positive limit in bytes, or None when none is readable.
    """
    limits: list[int] = []
    for path in paths:
        try:
            text = path.read_text(encoding="ascii").strip()
        except (OSError, UnicodeError):
            continue
        if text.lower() == "max":
            continue
        try:
            value = int(text)
        except ValueError:
            continue
        if value > 0:
            limits.append(value)
    return min(limits) if limits else None


def effective_memory_budget_bytes(
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_limit_paths: Sequence[Path] = _CGROUP_LIMIT_PATHS,
) -> int:
    """Return the conservative active memory budget for a Slurm process.

    Parameters
    ----------
    meminfo_path : pathlib.Path, default=/proc/meminfo
        Linux memory information; ``MemAvailable`` is expressed in KiB.
    cgroup_limit_paths : sequence[pathlib.Path]
        Candidate active allocation limit files whose values are bytes.

    Returns
    -------
    int
        Lowest positive value among available RAM, cgroup limit, and the
        reviewed 3-GiB request, in bytes.

    Raises
    ------
    ValueError
        If no positive memory budget can be determined.
    """
    candidates = [
        _read_mem_available_bytes(Path(meminfo_path)),
        _read_cgroup_limit_bytes(tuple(Path(path) for path in cgroup_limit_paths)),
        int(REQUESTED_RESOURCES["memory_bytes"]),
    ]
    positive = [value for value in candidates if value is not None and value > 0]
    if not positive:
        raise ValueError("No effective Slurm memory budget is available.")
    return min(positive)


def inspect_status(run_directory: Path | str) -> dict[str, object]:
    """Combine durable pipeline state with one read-only Slurm accounting query.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact run directory containing ``slurm_submission.json``.

    Returns
    -------
    dict[str, object]
        Pipeline status plus the parsed job-level scheduler accounting fields.

    Raises
    ------
    RuntimeError
        If the one-shot ``sacct`` query fails or lacks the requested job row.
    """
    directory = Path(run_directory)
    submission = _read_json(directory / "slurm_submission.json")
    job_id = submission.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("Slurm submission receipt has no job identifier.")
    pipeline_status = _pipeline_module().inspect_task_decoding_status(
        directory,
        verify_results=False,
    )
    arguments = [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        job_id,
        "--format=" + ",".join(_SACCT_FIELDS),
    ]
    try:
        completed = _run_command(arguments)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"sacct failed: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown scheduler error"
        raise RuntimeError(f"sacct failed: {detail}")
    scheduler: dict[str, str] | None = None
    for line in completed.stdout.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) != len(_SACCT_FIELDS) or fields[0] != job_id:
            continue
        scheduler = dict(zip(_SACCT_FIELDS[1:], fields[1:], strict=True))
        break
    if scheduler is None:
        raise RuntimeError(f"sacct returned no job-level record for {job_id}.")
    status = dict(pipeline_status)
    status["scheduler"] = scheduler
    return status


def _parser() -> argparse.ArgumentParser:
    """Build the login-node operational parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for one-shot new, resume, and status operations.
    """
    parser = argparse.ArgumentParser(prog="task-decoding-slurm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    submit_new_parser = subparsers.add_parser("submit-new")
    submit_new_parser.add_argument("--config", required=True)
    submit_resume_parser = subparsers.add_parser("submit-resume")
    submit_resume_parser.add_argument("--run-directory", required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded login-node Slurm operation.

    Parameters
    ----------
    argv : sequence[str] or None, default=None
        Command tokens excluding the program name; None uses ``sys.argv``.

    Returns
    -------
    int
        Zero after one successful submit/status operation. Validation and
        scheduler failures propagate to the module entry point as nonzero.
    """
    arguments = _parser().parse_args(argv)
    if arguments.command == "submit-new":
        payload = submit_new(Path(arguments.config))
    elif arguments.command == "submit-resume":
        payload = submit_resume(Path(arguments.run_directory))
    elif arguments.command == "status":
        payload = inspect_status(Path(arguments.run_directory))
    else:  # pragma: no cover - argparse owns this boundary.
        raise ValueError("Unknown Slurm operation.")
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
