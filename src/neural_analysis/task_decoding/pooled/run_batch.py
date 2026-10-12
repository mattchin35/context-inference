"""Bounded pooled planning and one-session-per-task Slurm arrays."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from src.neural_analysis.task_decoding.pooled import slurm
from src.neural_analysis.task_decoding.run_batch import read_config_list


_WRAPPER = "src/shell_scripts/pooled_task_variable_decoding_slurm.sh"
_run_command = slurm._run_command
_repository_commit = slurm._repository_commit
_read_json = slurm._read_json
_write_json = slurm._write_json


def _utc_now() -> str:
    """Return one whole-second UTC timestamp.

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


def _positive_int(value: str) -> int:
    """Parse one positive dimensionless command-line integer.

    Parameters
    ----------
    value : str
        Command-line integer token.

    Returns
    -------
    int
        Positive dimensionless value.
    """
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_envelope(value: object, description: str) -> dict[str, int]:
    """Validate a nonempty resource mapping of positive built-in integers.

    Parameters
    ----------
    value : object
        Candidate mapping whose keys name counts or byte-valued dimensions.
    description : str
        Human-readable source used in validation messages.

    Returns
    -------
    dict[str, int]
        Detached resource mapping; byte fields preserve byte units.
    """
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{description} resource envelope is invalid.")
    envelope: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{description} resource envelope has a non-string key.")
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(f"{description} resource field {key!r} must be positive.")
        envelope[key] = item
    return envelope


def _resource_evidence(run_directory: Path | str) -> dict[str, object]:
    """Load and validate one complete measured pooled benchmark.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Complete pooled run containing small JSON evidence sidecars.

    Returns
    -------
    dict[str, object]
        Analysis/source identity, peak RSS in bytes, and resource envelope.
    """
    directory = Path(run_directory).resolve()
    state = _read_json(directory / "run_state.json")
    usage = _read_json(directory / "resource_usage.json")
    source = _read_json(directory / "scientific_source.json")
    config = _read_json(directory / "config.json")
    if state.get("lifecycle") != "complete" or state.get("final_results_published") is not True:
        raise ValueError("Resource evidence must be a completely published pooled run.")
    if usage.get("status") != "complete":
        raise ValueError("Resource evidence usage must have complete status.")
    peak_rss = usage.get("peak_rss_bytes")
    if isinstance(peak_rss, bool) or not isinstance(peak_rss, int) or peak_rss <= 0:
        raise ValueError("Resource evidence peak_rss_bytes must be positive.")
    analysis_version = config.get("analysis_version")
    source_fingerprint = source.get("fingerprint")
    if not isinstance(analysis_version, str) or not analysis_version:
        raise ValueError("Resource evidence analysis version is invalid.")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise ValueError("Resource evidence source fingerprint is invalid.")
    return {
        "run_directory": str(directory),
        "analysis_version": analysis_version,
        "scientific_source_fingerprint": source_fingerprint,
        "peak_rss_bytes": peak_rss,
        "resource_envelope": _positive_envelope(
            usage.get("resource_envelope"),
            "Evidence",
        ),
    }


def _validated_plan(
    value: object,
    config_path: Path,
    evidence: Mapping[str, object],
) -> dict[str, object]:
    """Validate and admit one bounded session plan against measured evidence.

    Parameters
    ----------
    value : object
        Output from ``pipeline.plan_pooled_decoding_session``.
    config_path : pathlib.Path
        Exact resolved pooled overlay requested from the pipeline.
    evidence : mapping[str, object]
        Validated complete-run evidence with identical scientific identity.

    Returns
    -------
    dict[str, object]
        Detached plan preserving all dimension and byte units.
    """
    if not isinstance(value, Mapping):
        raise ValueError("Pooled session plan must be a mapping.")
    plan = dict(value)
    if plan.get("analysis_version") != evidence["analysis_version"]:
        raise ValueError("Pooled session plan differs from benchmark analysis version.")
    if (
        plan.get("scientific_source_fingerprint")
        != evidence["scientific_source_fingerprint"]
    ):
        raise ValueError("Pooled session plan differs from benchmark source evidence.")
    planned = _positive_envelope(plan.get("resource_envelope"), "Planned")
    measured = evidence["resource_envelope"]
    assert isinstance(measured, Mapping)
    if set(planned) != set(measured):
        raise ValueError("Planned and benchmark resource envelopes have different fields.")
    for name, amount in planned.items():
        if amount > measured[name]:
            raise ValueError(
                f"Planned resource {name!r} exceeds the completed benchmark evidence."
            )
    plan["config_path"] = str(config_path)
    plan["resource_envelope"] = planned
    return plan


def plan_batch(
    config_list_path: Path | str,
    resource_run_directory: Path | str,
    array_concurrency: int,
    pipeline: object | None = None,
) -> dict[str, object]:
    """Build one bounded multi-session array plan without loading spike arrays.

    Parameters
    ----------
    config_list_path : pathlib.Path or str
        UTF-8 file containing one portable pooled config path per session.
    resource_run_directory : pathlib.Path or str
        Complete measured pooled run used as an admission ceiling.
    array_concurrency : int
        Positive maximum number of simultaneous one-CPU/3-GiB array tasks.
    pipeline : object or None
        Optional module-like testing seam; ``None`` imports the pooled pipeline.

    Returns
    -------
    dict[str, object]
        JSON-safe session plans, evidence, and dimensionless concurrency cap.
    """
    if isinstance(array_concurrency, bool) or not isinstance(array_concurrency, int):
        raise ValueError("Array concurrency must be a positive integer.")
    if array_concurrency <= 0:
        raise ValueError("Array concurrency must be a positive integer.")
    paths = read_config_list(config_list_path)
    evidence = _resource_evidence(resource_run_directory)
    active_pipeline = _pipeline_module() if pipeline is None else pipeline
    sessions = [
        _validated_plan(
            active_pipeline.plan_pooled_decoding_session(path),
            path,
            evidence,
        )
        for path in paths
    ]
    return {
        "config_list_path": str(Path(config_list_path).resolve()),
        "array_concurrency": min(array_concurrency, len(sessions)),
        "session_count": len(sessions),
        "resource_evidence": evidence,
        "sessions": sessions,
    }


def prepare_array_batch(
    batch_plan: Mapping[str, object],
    config_list_path: Path | str,
    batch_root: Path | str,
    pipeline: object | None = None,
    timestamp: str | None = None,
) -> dict[str, object]:
    """Prepare exact independent runs and publish an immutable index manifest.

    Parameters
    ----------
    batch_plan : mapping[str, object]
        Validated output from :func:`plan_batch`.
    config_list_path : pathlib.Path or str
        Source ordered config-list path.
    batch_root : pathlib.Path or str
        Parent for one timestamped metadata-only batch directory.
    pipeline : object or None
        Optional module-like preparation seam.
    timestamp : str or None
        UTC directory token; ``None`` generates ``YYYYmmddTHHMMSSZ``.

    Returns
    -------
    dict[str, object]
        Manifest mapping including its absolute ``manifest_path``.
    """
    sessions_value = batch_plan.get("sessions")
    if not isinstance(sessions_value, Sequence) or isinstance(sessions_value, (str, bytes)):
        raise ValueError("Batch plan sessions are invalid.")
    active_pipeline = _pipeline_module() if pipeline is None else pipeline
    token = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    batch_directory = Path(batch_root).resolve() / token
    batch_directory.mkdir(parents=True, exist_ok=False)
    prepared_sessions: list[dict[str, object]] = []
    for index, value in enumerate(sessions_value):
        if not isinstance(value, Mapping):
            raise ValueError("Batch plan contains an invalid session.")
        config_path = Path(str(value["config_path"]))
        run_directory = active_pipeline.prepare_pooled_decoding_run(
            config_path,
            execution_mode="slurm",
        )
        prepared_sessions.append(
            {
                "array_index": index,
                "config_path": str(config_path),
                "run_directory": str(Path(run_directory)),
                "session_id": value.get("session_id"),
            }
        )
    manifest: dict[str, object] = {
        "created_at": _utc_now(),
        "config_list_path": str(Path(config_list_path).resolve()),
        "array_concurrency": batch_plan["array_concurrency"],
        "resource_evidence": batch_plan["resource_evidence"],
        "sessions": prepared_sessions,
    }
    manifest_path = batch_directory / "manifest.json"
    _write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _manifest_sessions(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """Validate the exact contiguous array-index mapping in one manifest.

    Parameters
    ----------
    manifest : mapping[str, object]
        Decoded batch manifest.

    Returns
    -------
    list[dict[str, object]]
        Detached sessions ordered by zero-based array index.
    """
    value = manifest.get("sessions")
    if not isinstance(value, list) or not value:
        raise ValueError("Batch manifest has no sessions.")
    sessions: list[dict[str, object]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or item.get("array_index") != index:
            raise ValueError("Batch manifest array indices are not contiguous.")
        run_directory = item.get("run_directory")
        if not isinstance(run_directory, str) or not run_directory:
            raise ValueError("Batch manifest run directory is invalid.")
        sessions.append(dict(item))
    return sessions


def _update_run_state(run_directory: Path, **changes: object) -> None:
    """Apply one durable state change to an exact prepared run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact session-contained pooled run directory.
    **changes : object
        JSON-safe lifecycle fields without array axes or unit conversion.

    Returns
    -------
    None
        Atomically replaces ``run_state.json``.
    """
    state = _read_json(run_directory / "run_state.json")
    state.update(changes)
    state["updated_at"] = _utc_now()
    _write_json(run_directory / "run_state.json", state)


def submit_array_batch(manifest_path: Path | str) -> dict[str, object]:
    """Submit one capped Slurm array and record every immutable index mapping.

    Parameters
    ----------
    manifest_path : pathlib.Path or str
        Exact prepared batch manifest path.

    Returns
    -------
    dict[str, object]
        Saved array job, commit, resources, and manifest receipt.
    """
    path = Path(manifest_path)
    manifest = _read_json(path)
    sessions = _manifest_sessions(manifest)
    concurrency = manifest.get("array_concurrency")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
        raise ValueError("Batch manifest array concurrency is invalid.")
    concurrency = min(concurrency, len(sessions))
    commit = _repository_commit()
    for item in sessions:
        _update_run_state(
            Path(str(item["run_directory"])),
            lifecycle="submission-pending",
            final_results_published=False,
            error=None,
        )
    arguments = [
        "sbatch",
        "--parsable",
        f"--array=0-{len(sessions) - 1}%{concurrency}",
        f"--export=ALL,POOLED_DECODING_EXPECTED_COMMIT={commit}",
        _WRAPPER,
        "_execute-array",
        "--manifest",
        str(path),
    ]
    try:
        completed = _run_command(arguments)
    except (OSError, subprocess.TimeoutExpired) as error:
        message = f"sbatch failed: {type(error).__name__}: {error}"
        for item in sessions:
            _update_run_state(
                Path(str(item["run_directory"])), lifecycle="failed", error=message
            )
        raise RuntimeError(message) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        message = f"sbatch failed: {detail}"
        for item in sessions:
            _update_run_state(
                Path(str(item["run_directory"])), lifecycle="failed", error=message
            )
        raise RuntimeError(message)
    job_id = slurm._parse_job_id(completed.stdout)
    receipt: dict[str, object] = {
        "job_id": job_id,
        "submitted_at": _utc_now(),
        "manifest_path": str(path),
        "repository_commit": commit,
        "requested_resources": dict(slurm.REQUESTED_RESOURCES),
        "array_concurrency": concurrency,
        "session_count": len(sessions),
    }
    _write_json(path.parent / "slurm_submission.json", receipt)
    for item in sessions:
        index = int(item["array_index"])
        directory = Path(str(item["run_directory"]))
        quoted = shlex.quote(str(directory))
        session_receipt = {
            **receipt,
            "array_index": index,
            "run_directory": str(directory),
            "scheduler_log_path": str(slurm.REQUESTED_RESOURCES["output"])
            .replace("%A", job_id)
            .replace("%a", str(index)),
            "status_command": f"bash {_WRAPPER} status --run-directory {quoted}",
            "resume_command": f"bash {_WRAPPER} submit-resume --run-directory {quoted}",
        }
        _write_json(directory / "slurm_submission.json", session_receipt)
    return receipt


def execute_array_element(
    manifest_path: Path | str,
    array_task_id: int,
    pipeline: object | None = None,
) -> None:
    """Execute only the exact run assigned to one zero-based array element.

    Parameters
    ----------
    manifest_path : pathlib.Path or str
        Exact immutable array manifest path.
    array_task_id : int
        Zero-based dimensionless Slurm array index.
    pipeline : object or None
        Optional module-like execution seam.

    Returns
    -------
    None
        Runs the selected prepared directory; no sibling search occurs.
    """
    if isinstance(array_task_id, bool) or not isinstance(array_task_id, int):
        raise ValueError("Slurm array task index must be an integer.")
    sessions = _manifest_sessions(_read_json(Path(manifest_path)))
    if array_task_id < 0 or array_task_id >= len(sessions):
        raise ValueError("Slurm array task index is outside the manifest.")
    active_pipeline = _pipeline_module() if pipeline is None else pipeline
    from src.neural_analysis.task_decoding.pooled.run_session import (
        _execute_with_signal_handling,
    )

    _execute_with_signal_handling(
        Path(str(sessions[array_task_id]["run_directory"])),
        active_pipeline,
    )


def inspect_batch_status(manifest_path: Path | str) -> dict[str, object]:
    """Read every exact session status without modifying batch artifacts.

    Parameters
    ----------
    manifest_path : pathlib.Path or str
        Exact prepared batch manifest path.

    Returns
    -------
    dict[str, object]
        Manifest path and per-index durable pooled status records.
    """
    path = Path(manifest_path)
    sessions = _manifest_sessions(_read_json(path))
    pipeline = _pipeline_module()
    statuses = []
    for item in sessions:
        statuses.append(
            {
                "array_index": item["array_index"],
                "run_directory": item["run_directory"],
                "pipeline": pipeline.inspect_pooled_decoding_status(
                    Path(str(item["run_directory"]))
                ),
            }
        )
    return {"manifest_path": str(path), "sessions": statuses}


def _parser() -> argparse.ArgumentParser:
    """Build the pooled batch/array command parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser for dry-run, submit, status, and private array execution.
    """
    parser = argparse.ArgumentParser(prog="pooled-task-decoding-batch")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("batch-dry-run", "submit-array"):
        command = commands.add_parser(name)
        command.add_argument("--config-list", required=True)
        command.add_argument("--resource-run-directory", required=True)
        command.add_argument("--array-concurrency", type=_positive_int, required=True)
        if name == "submit-array":
            command.add_argument("--batch-root")
    status = commands.add_parser("batch-status")
    status.add_argument("--manifest", required=True)
    execute = commands.add_parser("_execute-array")
    execute.add_argument("--manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded pooled batch operation.

    Parameters
    ----------
    argv : sequence[str] or None
        Command tokens excluding the program name; ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Zero after accepted work, 130 after interruption, otherwise one.
    """
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "batch-status":
            print(json.dumps(inspect_batch_status(arguments.manifest), indent=2))
            return 0
        if arguments.command == "_execute-array":
            task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if task_id is None:
                raise ValueError("SLURM_ARRAY_TASK_ID is required for array execution.")
            execute_array_element(arguments.manifest, int(task_id))
            return 0
        plan = plan_batch(
            arguments.config_list,
            arguments.resource_run_directory,
            arguments.array_concurrency,
        )
        if arguments.command == "batch-dry-run":
            print(json.dumps(plan, indent=2))
            return 0
        config_list = Path(arguments.config_list).resolve()
        batch_root = (
            Path(arguments.batch_root)
            if arguments.batch_root
            else config_list.parent / "pooled_batch_runs"
        )
        prepared = prepare_array_batch(plan, config_list, batch_root)
        receipt = submit_array_batch(str(prepared["manifest_path"]))
        print(json.dumps(receipt, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
