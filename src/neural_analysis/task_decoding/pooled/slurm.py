"""One-shot Slurm submission and status operations for pooled decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from typing import Mapping, Sequence

from src.neural_analysis.task_decoding import slurm as shared_slurm


REQUESTED_RESOURCES = {
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

_WRAPPER = "src/shell_scripts/pooled_task_variable_decoding_slurm.sh"
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
_CGROUP_LIMIT_PATHS = (
    Path("/sys/fs/cgroup/memory.max"),
    Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
)

# Reuse the established bounded subprocess and atomic JSON primitives while
# retaining pooled-specific lifecycle and resource policy in this module.
_run_command = shared_slurm._run_command
_repository_commit = shared_slurm._repository_commit
_read_json = shared_slurm._read_json
_write_json = shared_slurm._write_json


def _utc_now() -> str:
    """Return a whole-second UTC timestamp.

    Returns
    -------
    str
        ISO-8601 UTC text ending in ``Z``.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pipeline_module():
    """Import the pooled numerical pipeline lazily.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pooled.pipeline``.
    """
    return __import__(
        "src.neural_analysis.task_decoding.pooled.pipeline",
        fromlist=["pipeline"],
    )


def _update_state(run_directory: Path, **changes: object) -> dict[str, object]:
    """Atomically apply a pooled lifecycle transition.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact prepared pooled run directory.
    **changes : object
        JSON-safe lifecycle fields without array axes or physical units.

    Returns
    -------
    dict[str, object]
        Complete updated state including a refreshed UTC timestamp.
    """
    state = _read_json(run_directory / "run_state.json")
    state.update(changes)
    state["updated_at"] = _utc_now()
    _write_json(run_directory / "run_state.json", state)
    return state


def _require_slurm_prepared_run(run_directory: Path) -> None:
    """Require one exact Slurm-prepared directory eligible for submission.

    Parameters
    ----------
    run_directory : pathlib.Path
        Candidate immutable run directory.

    Returns
    -------
    None
        Returns only for prepared, interrupted, or failed Slurm runs.
    """
    if not run_directory.is_dir():
        raise ValueError("The exact prepared pooled run directory does not exist.")
    execution = _read_json(run_directory / "execution.json")
    if execution.get("mode") != "slurm":
        raise ValueError("The pooled run was not prepared for Slurm execution.")
    lifecycle = _read_json(run_directory / "run_state.json").get("lifecycle")
    if lifecycle not in {"prepared", "interrupted", "failed"}:
        raise ValueError(f"Pooled run cannot be submitted from lifecycle {lifecycle!r}.")


def _validate_tensor_memory(plan: Mapping[str, object]) -> None:
    """Require planned tensors to use at most half the requested memory.

    Parameters
    ----------
    plan : mapping[str, object]
        Bounded pooled dry-run record containing a resource envelope in bytes.

    Returns
    -------
    None
        Returns when exact tensor allocation is within the 50-percent guard.
    """
    envelope = plan.get("resource_envelope")
    if not isinstance(envelope, Mapping):
        raise ValueError("Pooled dry-run resource envelope is missing.")
    tensor_bytes = envelope.get("tensor_allocation_bytes")
    if isinstance(tensor_bytes, bool) or not isinstance(tensor_bytes, int):
        raise ValueError("Pooled tensor allocation must be an integer byte count.")
    if tensor_bytes < 0 or tensor_bytes * 2 > int(REQUESTED_RESOURCES["memory_bytes"]):
        raise ValueError("Pooled tensor allocation exceeds 50% of requested Slurm memory.")


def _parse_job_id(stdout: str) -> str:
    """Parse one ``sbatch --parsable`` response.

    Parameters
    ----------
    stdout : str
        Scheduler text containing ``job_id`` or ``job_id;cluster``.

    Returns
    -------
    str
        Nonempty scheduler job identifier.
    """
    first_line = stdout.strip().splitlines()[0] if stdout.strip() else ""
    job_id = first_line.split(";", maxsplit=1)[0].strip()
    if not job_id:
        raise RuntimeError("sbatch returned no pooled job identifier.")
    return job_id


def submit_prepared_run(run_directory: Path | str) -> dict[str, object]:
    """Submit one exact pooled run directory once.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact Slurm-prepared pooled run directory.

    Returns
    -------
    dict[str, object]
        Saved job, commit, resource, log, resume, and status receipt.
    """
    directory = Path(run_directory)
    _require_slurm_prepared_run(directory)
    commit = _repository_commit()
    _update_state(
        directory,
        lifecycle="submission-pending",
        final_results_published=False,
        error=None,
    )
    arguments = [
        "sbatch",
        "--parsable",
        "--array=0",
        f"--export=ALL,POOLED_DECODING_EXPECTED_COMMIT={commit}",
        _WRAPPER,
        "_execute-prepared",
        "--run-directory",
        str(directory),
    ]
    try:
        completed = _run_command(arguments)
    except (OSError, subprocess.TimeoutExpired) as error:
        message = f"sbatch failed: {type(error).__name__}: {error}"
        _update_state(directory, lifecycle="failed", error=message)
        raise RuntimeError(message) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        message = f"sbatch failed: {detail}"
        _update_state(directory, lifecycle="failed", error=message)
        raise RuntimeError(message)
    job_id = _parse_job_id(completed.stdout)
    quoted = shlex.quote(str(directory))
    receipt: dict[str, object] = {
        "job_id": job_id,
        "array_index": 0,
        "submitted_at": _utc_now(),
        "run_directory": str(directory),
        "requested_resources": dict(REQUESTED_RESOURCES),
        "repository_commit": commit,
        "scheduler_log_path": str(REQUESTED_RESOURCES["output"])
        .replace("%A", job_id)
        .replace("%a", "0"),
        "status_command": f"bash {_WRAPPER} status --run-directory {quoted}",
        "resume_command": f"bash {_WRAPPER} submit-resume --run-directory {quoted}",
    }
    _write_json(directory / "slurm_submission.json", receipt)
    return receipt


def submit_new(config_path: Path | str) -> dict[str, object]:
    """Plan, prepare, and submit one new pooled configuration.

    Parameters
    ----------
    config_path : pathlib.Path or str
        Existing cluster-visible pooled overlay JSON path.

    Returns
    -------
    dict[str, object]
        Single-session Slurm submission receipt.
    """
    path = Path(config_path)
    pipeline = _pipeline_module()
    plan = pipeline.plan_pooled_decoding_session(path)
    _validate_tensor_memory(plan)
    directory = pipeline.prepare_pooled_decoding_run(path, execution_mode="slurm")
    return submit_prepared_run(directory)


def submit_resume(run_directory: Path | str) -> dict[str, object]:
    """Submit an exact interrupted or failed pooled run.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact saved directory; parents and siblings are never searched.

    Returns
    -------
    dict[str, object]
        Replacement submission receipt for the same immutable run.
    """
    return submit_prepared_run(Path(run_directory))


def inspect_status(run_directory: Path | str) -> dict[str, object]:
    """Combine durable pooled state with one read-only accounting query.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Exact submitted pooled run directory.

    Returns
    -------
    dict[str, object]
        Pipeline status plus job-level Slurm accounting or a concise error.
    """
    directory = Path(run_directory)
    submission = _read_json(directory / "slurm_submission.json")
    job_id = submission.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("Pooled Slurm receipt has no job identifier.")
    status = dict(_pipeline_module().inspect_pooled_decoding_status(directory))
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
        status["scheduler"] = None
        status["scheduler_error"] = f"sacct failed: {error}"
        return status
    if completed.returncode != 0:
        status["scheduler"] = None
        status["scheduler_error"] = (
            "sacct failed: " + (completed.stderr.strip() or "unknown scheduler error")
        )
        return status
    scheduler = None
    for line in completed.stdout.splitlines():
        fields = line.rstrip("|").split("|")
        if len(fields) != len(_SACCT_FIELDS) or fields[0] != job_id:
            continue
        scheduler = dict(zip(_SACCT_FIELDS[1:], fields[1:], strict=True))
        break
    status["scheduler"] = scheduler
    if scheduler is None:
        status["scheduler_error"] = f"sacct returned no job-level record for {job_id}."
    return status


def effective_memory_budget_bytes(
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    cgroup_limit_paths: Sequence[Path] = _CGROUP_LIMIT_PATHS,
) -> int:
    """Return the conservative available memory for one scheduled process.

    Parameters
    ----------
    meminfo_path : pathlib.Path
        Proc-style file whose ``MemAvailable`` value is expressed in KiB.
    cgroup_limit_paths : sequence[pathlib.Path]
        Candidate active allocation limits expressed in bytes.

    Returns
    -------
    int
        Lowest positive host, cgroup, and requested-memory budget in bytes.
    """
    candidates: list[int] = [int(REQUESTED_RESOURCES["memory_bytes"])]
    try:
        for line in Path(meminfo_path).read_text(encoding="ascii").splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] == "MemAvailable:":
                value = int(fields[1]) * 1024
                if value > 0:
                    candidates.append(value)
                break
    except (OSError, UnicodeError, ValueError):
        pass
    for path in cgroup_limit_paths:
        try:
            text = Path(path).read_text(encoding="ascii").strip()
            value = int(text)
        except (OSError, UnicodeError, ValueError):
            continue
        if value > 0:
            candidates.append(value)
    return min(candidates)


def _parser() -> argparse.ArgumentParser:
    """Build the public pooled Slurm command parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for new, resume, and status operations.
    """
    parser = argparse.ArgumentParser(prog="pooled-task-decoding-slurm")
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("submit-new")
    new.add_argument("--config", required=True)
    resume = commands.add_parser("submit-resume")
    resume.add_argument("--run-directory", required=True)
    status = commands.add_parser("status")
    status.add_argument("--run-directory", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded pooled Slurm operation.

    Parameters
    ----------
    argv : sequence[str] or None
        Command tokens excluding the program name; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Zero after a successful submission or status inspection.
    """
    arguments = _parser().parse_args(argv)
    if arguments.command == "submit-new":
        payload = submit_new(arguments.config)
    elif arguments.command == "submit-resume":
        payload = submit_resume(arguments.run_directory)
    elif arguments.command == "status":
        payload = inspect_status(arguments.run_directory)
    else:  # pragma: no cover
        raise ValueError("Unknown pooled Slurm operation.")
    print(json.dumps(payload, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
