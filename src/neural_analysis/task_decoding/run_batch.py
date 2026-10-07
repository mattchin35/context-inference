"""Bounded multi-session task-decoding planning and foreground execution."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
from typing import Any


for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


_RESOURCE_USAGE_SCHEMA_VERSION = 1
_RESOURCE_ENVELOPE_KEYS = frozenset(
    {
        "full_trial_count",
        "tensor_trial_count",
        "pfc_unit_count",
        "hpc_unit_count",
        "time_bin_count",
        "target_count",
        "outer_fold_count",
        "inner_fold_count",
        "coefficient_feature_capacity",
        "categorical_fit_count",
        "numerical_fit_count",
        "tensor_allocation_bytes",
    }
)


def _pipeline_module():
    """Import the numerical pipeline only after thread limits are frozen.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pipeline`` imported lazily after
        OpenMP, MKL, and OpenBLAS are limited to one thread.
    """
    return __import__(
        "src.neural_analysis.task_decoding.pipeline",
        fromlist=["pipeline"],
    )


def _positive_int(text: str) -> int:
    """Parse one positive command-line integer.

    Parameters
    ----------
    text : str
        Command-line token representing a requested worker count.

    Returns
    -------
    int
        Positive dimensionless worker count.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``text`` is not a positive integer.
    """
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    """Build the batch CLI parser without importing numerical modules.

    Returns
    -------
    argparse.ArgumentParser
        Parser for bounded ``dry-run`` and foreground ``new`` commands.
    """
    parser = argparse.ArgumentParser(prog="task-decoding-batch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("dry-run", "new"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config-list", required=True)
        subparser.add_argument("--workers", type=_positive_int)
        subparser.add_argument("--resource-run-directory")
        if command == "new":
            subparser.add_argument("--rerun", action="store_true")
    return parser


def read_config_list(config_list_path: Path | str) -> tuple[Path, ...]:
    """Read unique session configs in declared order from one UTF-8 list.

    Parameters
    ----------
    config_list_path : pathlib.Path or str
        Text file with one path per nonblank, non-comment line. Relative paths
        resolve against the list file's directory.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Unique existing config files as resolved absolute paths in list order.

    Raises
    ------
    ValueError
        If the list is empty, a path is absent/not a file, or two entries
        resolve to the same configuration.
    """
    list_path = Path(config_list_path).resolve()
    try:
        lines = list_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Config list is unreadable: {list_path}") from error
    config_paths: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        text = raw_line.strip()
        if not text or text.startswith("#"):
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = list_path.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise ValueError(
                f"Config-list entry {line_number} is not an existing file: {candidate}"
            )
        if candidate in seen:
            raise ValueError(f"Duplicate resolved config path: {candidate}")
        seen.add(candidate)
        config_paths.append(candidate)
    if not config_paths:
        raise ValueError("Config list contains no session configurations.")
    return tuple(config_paths)


def _read_mem_available_bytes() -> int:
    """Read the Linux ``MemAvailable`` budget.

    Returns
    -------
    int
        Available memory in bytes.

    Raises
    ------
    RuntimeError
        If ``/proc/meminfo`` is unavailable or lacks a positive MemAvailable.
    """
    try:
        lines = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    except OSError as error:
        raise RuntimeError("Cannot determine MemAvailable from /proc/meminfo.") from error
    for line in lines:
        if not line.startswith("MemAvailable:"):
            continue
        fields = line.split()
        if len(fields) != 3 or fields[2] != "kB":
            break
        try:
            available = int(fields[1]) * 1024
        except ValueError:
            break
        if available > 0:
            return available
    raise RuntimeError("Cannot determine a positive MemAvailable memory budget.")


def _read_json_object(path: Path, description: str) -> dict[str, object]:
    """Read one required UTF-8 JSON object with a causal error.

    Parameters
    ----------
    path : pathlib.Path
        Existing small JSON sidecar path.
    description : str
        Human-readable sidecar role used in validation errors.

    Returns
    -------
    dict[str, object]
        Decoded JSON object.

    Raises
    ------
    ValueError
        If the sidecar is absent, unreadable, malformed, or not an object.
    """
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Resource evidence {description} is missing or invalid.") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"Resource evidence {description} must be a JSON object.")
    return decoded


def _positive_builtin_int(value: object, name: str) -> int:
    """Validate one positive built-in integer evidence field.

    Parameters
    ----------
    value : object
        Decoded JSON scalar.
    name : str
        Field name used in validation errors.

    Returns
    -------
    int
        Positive integer value.

    Raises
    ------
    ValueError
        If the value is Boolean, noninteger, or nonpositive.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Resource evidence {name} must be a positive integer.")
    return value


def _nonnegative_builtin_int(value: object, name: str) -> int:
    """Validate one nonnegative built-in integer evidence field.

    Parameters
    ----------
    value : object
        Decoded JSON scalar.
    name : str
        Field name used in validation errors.

    Returns
    -------
    int
        Nonnegative integer value.

    Raises
    ------
    ValueError
        If the value is Boolean, noninteger, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Resource evidence {name} must be a nonnegative integer.")
    return value


def _validate_resource_envelope(value: object, description: str) -> dict[str, int]:
    """Validate the exact comparable resource-envelope vocabulary.

    Parameters
    ----------
    value : object
        Decoded JSON resource-envelope candidate.
    description : str
        Context label used in validation errors.

    Returns
    -------
    dict[str, int]
        Exact positive dimension and byte fields.

    Raises
    ------
    ValueError
        If keys differ or any member is not a positive built-in integer.
    """
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_ENVELOPE_KEYS:
        raise ValueError(f"{description} resource envelope has missing or unexpected fields.")
    envelope: dict[str, int] = {}
    for key in sorted(_RESOURCE_ENVELOPE_KEYS):
        validator = (
            _nonnegative_builtin_int
            if key in {"categorical_fit_count", "numerical_fit_count"}
            else _positive_builtin_int
        )
        envelope[key] = validator(value[key], f"resource_envelope.{key}")
    if envelope["categorical_fit_count"] + envelope["numerical_fit_count"] <= 0:
        raise ValueError(f"{description} resource envelope must include planned fits.")
    return envelope


def _validate_planned_session(plan: object, config_path: Path) -> dict[str, object]:
    """Validate one bounded pipeline plan used by batch admission.

    Parameters
    ----------
    plan : object
        Output from ``pipeline.plan_task_decoding_session``.
    config_path : pathlib.Path
        Exact resolved path whose plan was requested.

    Returns
    -------
    dict[str, object]
        Shallow plan copy with a validated resource envelope.

    Raises
    ------
    ValueError
        If batch-required identity or resource fields are absent or invalid.
    """
    required = {
        "session_id",
        "fit_count",
        "tensor_allocation_bytes",
        "analysis_version",
        "scientific_source_fingerprint",
        "runtime_versions",
        "platform",
        "architecture",
        "thread_limits",
        "regularization_mode",
        "resource_envelope",
    }
    if not isinstance(plan, Mapping) or not required.issubset(plan):
        raise ValueError(f"Batch plan is incomplete for {config_path}.")
    validated = dict(plan)
    validated["config_path"] = str(config_path)
    validated["resource_envelope"] = _validate_resource_envelope(
        plan["resource_envelope"],
        f"Plan for {config_path}",
    )
    tensor_bytes = _positive_builtin_int(
        plan["tensor_allocation_bytes"],
        "tensor_allocation_bytes",
    )
    if tensor_bytes != validated["resource_envelope"]["tensor_allocation_bytes"]:
        raise ValueError(f"Batch plan tensor allocation disagrees for {config_path}.")
    for mapping_name in ("runtime_versions", "thread_limits"):
        if not isinstance(plan[mapping_name], Mapping):
            raise ValueError(f"Batch plan {mapping_name} is invalid for {config_path}.")
    for text_name in (
        "session_id",
        "analysis_version",
        "scientific_source_fingerprint",
        "platform",
        "architecture",
        "regularization_mode",
    ):
        if not isinstance(plan[text_name], str) or not str(plan[text_name]).strip():
            raise ValueError(f"Batch plan {text_name} is invalid for {config_path}.")
    return validated


def _load_resource_evidence(
    run_directory: Path,
    plans: Sequence[Mapping[str, object]],
    pipeline: Any,
) -> dict[str, object]:
    """Load and validate one explicit completed measured resource run.

    Parameters
    ----------
    run_directory : pathlib.Path
        Explicit evidence directory; implicit latest-run discovery is forbidden.
    plans : sequence[mapping[str, object]]
        Validated bounded plans for every proposed session.
    pipeline : module
        Lazy task-decoding pipeline used for complete-directory validation.

    Returns
    -------
    dict[str, object]
        JSON-safe evidence identity, measured peak RSS in bytes, and envelope.

    Raises
    ------
    ValueError
        If completion, measurement, identity, or envelope compatibility fails.
    """
    directory = run_directory.resolve()
    if not pipeline._validate_complete_prepared_run(directory):
        raise ValueError("Resource evidence run must be complete and fully valid.")
    config = _read_json_object(directory / "config.json", "scientific config")
    source = _read_json_object(directory / "scientific_source.json", "source identity")
    execution = _read_json_object(directory / "execution.json", "execution identity")
    usage = _read_json_object(directory / "resource_usage.json", "resource usage")
    if usage.get("schema_version") != _RESOURCE_USAGE_SCHEMA_VERSION:
        raise ValueError("Resource evidence usage schema version is invalid.")
    method = usage.get("measurement_method")
    if method != "resource.getrusage":
        raise ValueError("Resource evidence measurement method is invalid.")
    peak_rss_bytes = _positive_builtin_int(usage.get("peak_rss_bytes"), "peak_rss_bytes")
    evidence_envelope = _validate_resource_envelope(
        usage.get("resource_envelope"),
        "Evidence",
    )
    comparisons = {
        "analysis_version": config.get("analysis_version"),
        "scientific_source_fingerprint": source.get("fingerprint"),
        "runtime_versions": execution.get("runtime_versions"),
        "platform": execution.get("platform"),
        "architecture": execution.get("architecture"),
        "thread_limits": execution.get("thread_limits"),
        "regularization_mode": config.get("regularization_mode"),
    }
    for plan in plans:
        for field, evidence_value in comparisons.items():
            if plan.get(field) != evidence_value:
                raise ValueError(f"Resource evidence {field} identity does not match the plan.")
        plan_envelope = plan["resource_envelope"]
        for field in sorted(_RESOURCE_ENVELOPE_KEYS):
            if plan_envelope[field] > evidence_envelope[field]:
                raise ValueError(
                    f"Planned resource envelope exceeds evidence for {field}."
                )
    fingerprint_path = directory / "run_fingerprint.txt"
    try:
        fingerprint = fingerprint_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("Resource evidence run fingerprint is missing or invalid.") from error
    if not fingerprint:
        raise ValueError("Resource evidence run fingerprint is missing or invalid.")
    return {
        "run_directory": str(directory),
        "run_fingerprint": fingerprint,
        "measurement_method": method,
        "peak_rss_bytes": peak_rss_bytes,
        "resource_envelope": evidence_envelope,
        **comparisons,
    }


def _json_safe(value: object) -> object:
    """Recursively convert paths and mapping keys for stable JSON output.

    Parameters
    ----------
    value : object
        Nested batch report containing JSON scalars, mappings, sequences, or Paths.

    Returns
    -------
    object
        JSON-compatible value with paths and mapping keys rendered as strings.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _plan_batch(
    *,
    config_paths: Sequence[Path],
    requested_workers: int | None,
    resource_run_directory: Path | None,
    pipeline: Any,
) -> dict[str, object]:
    """Build one bounded ordered batch plan and concurrency calculation.

    Parameters
    ----------
    config_paths : sequence[pathlib.Path]
        Unique resolved config files in declared order.
    requested_workers : int or None
        Explicit positive worker request, or None to request available CPUs.
    resource_run_directory : pathlib.Path or None
        Explicit completed measured evidence run. None selects the safe one-worker path.
    pipeline : module
        Lazy task-decoding pipeline providing bounded planning and validation.

    Returns
    -------
    dict[str, object]
        Ordered session plans, memory budget, requested/effective workers, and
        optional evidence identity. Byte-valued fields explicitly end in ``_bytes``.

    Raises
    ------
    ValueError or RuntimeError
        If planning, memory, evidence, or worker admission is invalid.
    """
    available_cpus = os.cpu_count() or 1
    requested = available_cpus if requested_workers is None else requested_workers
    plans = tuple(
        _validate_planned_session(pipeline.plan_task_decoding_session(path), path)
        for path in config_paths
    )
    available_memory_bytes = _read_mem_available_bytes()
    memory_limit_bytes = available_memory_bytes // 2
    if memory_limit_bytes <= 0:
        raise RuntimeError("Half of MemAvailable does not permit a session allocation.")
    for plan in plans:
        if plan["tensor_allocation_bytes"] > memory_limit_bytes:
            raise ValueError(
                "Exact tensor allocation exceeds 50% of MemAvailable for "
                f"{plan['config_path']}."
            )
    evidence = None
    per_session_peak_rss_bytes = None
    if resource_run_directory is None:
        effective_workers = 1
    else:
        evidence = _load_resource_evidence(resource_run_directory, plans, pipeline)
        per_session_peak_rss_bytes = evidence["peak_rss_bytes"]
        memory_workers = memory_limit_bytes // per_session_peak_rss_bytes
        if memory_workers <= 0:
            raise ValueError("Measured peak RSS exceeds 50% of MemAvailable.")
        effective_workers = min(
            requested,
            available_cpus,
            len(plans),
            memory_workers,
        )
    return {
        "requested_workers": requested,
        "effective_workers": effective_workers,
        "available_cpu_count": available_cpus,
        "available_memory_bytes": available_memory_bytes,
        "memory_limit_bytes": memory_limit_bytes,
        "per_session_peak_rss_bytes": per_session_peak_rss_bytes,
        "resource_evidence": evidence,
        "sessions": plans,
    }


def _execute_prepared_worker(run_directory: str) -> dict[str, str]:
    """Execute one prepared session in an independent worker process.

    Parameters
    ----------
    run_directory : str
        Absolute prepared run directory. Scientific settings are loaded only
        from its saved sidecars.

    Returns
    -------
    dict[str, str]
        ``complete`` outcome or a concise failed outcome/error record.
    """
    try:
        _pipeline_module().run_prepared_task_decoding(Path(run_directory))
    except Exception as error:
        return {"outcome": "failed", "error": f"{type(error).__name__}: {error}"}
    return {"outcome": "complete", "error": ""}


def _execute_sessions(
    run_directories: Sequence[Path],
    *,
    worker_count: int,
    pipeline: Any,
) -> list[dict[str, str]]:
    """Execute independent prepared sessions with no within-session parallelism.

    Parameters
    ----------
    run_directories : sequence[pathlib.Path]
        Independent prepared run directories in config-list order.
    worker_count : int
        Positive admitted cross-session process count.
    pipeline : module
        Parent-process pipeline used directly only on the one-worker safe path.

    Returns
    -------
    list[dict[str, str]]
        Ordered complete/failed records, one per input run directory.
    """
    if worker_count <= 1:
        outcomes: list[dict[str, str]] = []
        for run_directory in run_directories:
            try:
                pipeline.run_prepared_task_decoding(run_directory)
            except Exception as error:
                outcomes.append(
                    {"outcome": "failed", "error": f"{type(error).__name__}: {error}"}
                )
            else:
                outcomes.append({"outcome": "complete", "error": ""})
        return outcomes
    outcomes = [
        {"outcome": "failed", "error": "Worker did not return an outcome."}
        for _run_directory in run_directories
    ]
    with ProcessPoolExecutor(max_workers=min(worker_count, len(run_directories))) as executor:
        futures: dict[Future[dict[str, str]], int] = {
            executor.submit(_execute_prepared_worker, str(run_directory)): index
            for index, run_directory in enumerate(run_directories)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcomes[index] = future.result()
            except Exception as error:
                outcomes[index] = {
                    "outcome": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
    return outcomes


def _batch_provenance(
    batch_plan: Mapping[str, object],
    *,
    config_list_path: Path,
    session_plan: Mapping[str, object],
) -> dict[str, object]:
    """Build per-run execution-only batch provenance.

    Parameters
    ----------
    batch_plan : mapping[str, object]
        Worker, memory, and optional evidence calculation from :func:`_plan_batch`.
    config_list_path : pathlib.Path
        Exact config-list path used for this batch invocation.
    session_plan : mapping[str, object]
        Bounded plan for the one prepared session receiving this record.

    Returns
    -------
    dict[str, object]
        JSON-safe execution-only batch facts and the session resource envelope.
    """
    return {
        "config_list_path": str(config_list_path.resolve()),
        "requested_workers": batch_plan["requested_workers"],
        "effective_workers": batch_plan["effective_workers"],
        "available_cpu_count": batch_plan["available_cpu_count"],
        "available_memory_bytes": batch_plan["available_memory_bytes"],
        "memory_limit_bytes": batch_plan["memory_limit_bytes"],
        "per_session_peak_rss_bytes": batch_plan["per_session_peak_rss_bytes"],
        "resource_evidence": batch_plan["resource_evidence"],
        "session_resource_envelope": session_plan["resource_envelope"],
    }


def _dry_run_report(batch_plan: Mapping[str, object]) -> dict[str, object]:
    """Return the public bounded dry-run report.

    Parameters
    ----------
    batch_plan : mapping[str, object]
        Validated plan from :func:`_plan_batch`.

    Returns
    -------
    dict[str, object]
        JSON-safe ordered session plans and worker/resource calculation.
    """
    return {"mode": "dry-run", **dict(batch_plan)}


def _run_new_batch(
    batch_plan: Mapping[str, object],
    *,
    config_list_path: Path,
    rerun: bool,
    pipeline: Any,
) -> tuple[dict[str, object], bool]:
    """Prepare and execute all independently valid sessions in one batch.

    Parameters
    ----------
    batch_plan : mapping[str, object]
        Validated bounded session/resource plan.
    config_list_path : pathlib.Path
        Exact ordered config-list path for execution provenance.
    rerun : bool
        True to create new immutable directories even when valid matches exist.
    pipeline : module
        Shared WP5B preparation/execution seams.

    Returns
    -------
    tuple[dict[str, object], bool]
        Ordered JSON-safe batch summary and whether any session failed.
    """
    session_reports: list[dict[str, object]] = []
    pending_indices: list[int] = []
    pending_directories: list[Path] = []
    for session_plan in batch_plan["sessions"]:
        report = {
            "config_path": session_plan["config_path"],
            "session_id": session_plan["session_id"],
            "run_directory": None,
            "outcome": "failed",
            "error": "",
        }
        session_reports.append(report)
        try:
            prepared = pipeline.prepare_task_decoding_run(
                session_plan["config_path"],
                rerun,
                "foreground",
            )
            report["run_directory"] = str(prepared)
            if pipeline._validate_complete_prepared_run(prepared):
                report["outcome"] = "skipped_complete"
                continue
            provenance = _batch_provenance(
                batch_plan,
                config_list_path=config_list_path,
                session_plan=session_plan,
            )
            pipeline._record_batch_execution_provenance(prepared, provenance)
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            continue
        pending_indices.append(len(session_reports) - 1)
        pending_directories.append(prepared)
    if pending_directories:
        execution_outcomes = _execute_sessions(
            pending_directories,
            worker_count=int(batch_plan["effective_workers"]),
            pipeline=pipeline,
        )
        for report_index, outcome in zip(
            pending_indices,
            execution_outcomes,
            strict=True,
        ):
            session_reports[report_index].update(outcome)
    failed = any(report["outcome"] == "failed" for report in session_reports)
    return (
        {
            "mode": "new",
            "requested_workers": batch_plan["requested_workers"],
            "effective_workers": batch_plan["effective_workers"],
            "available_cpu_count": batch_plan["available_cpu_count"],
            "available_memory_bytes": batch_plan["available_memory_bytes"],
            "memory_limit_bytes": batch_plan["memory_limit_bytes"],
            "per_session_peak_rss_bytes": batch_plan["per_session_peak_rss_bytes"],
            "resource_evidence": batch_plan["resource_evidence"],
            "sessions": session_reports,
        },
        failed,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded batch command and return a shell exit code.

    Parameters
    ----------
    argv : sequence[str] or None, default=None
        Command tokens excluding the program name; None uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero for a valid dry-run or all-success/skipped batch; one for planning,
        evidence, preparation, or per-session execution failure.
    """
    arguments = _parser().parse_args(argv)
    try:
        config_list_path = Path(arguments.config_list).resolve()
        config_paths = read_config_list(config_list_path)
        pipeline = _pipeline_module()
        resource_directory = (
            None
            if arguments.resource_run_directory is None
            else Path(arguments.resource_run_directory)
        )
        batch_plan = _plan_batch(
            config_paths=config_paths,
            requested_workers=arguments.workers,
            resource_run_directory=resource_directory,
            pipeline=pipeline,
        )
        if arguments.command == "dry-run":
            print(json.dumps(_json_safe(_dry_run_report(batch_plan)), sort_keys=True))
            return 0
        report, failed = _run_new_batch(
            batch_plan,
            config_list_path=config_list_path,
            rerun=arguments.rerun,
            pipeline=pipeline,
        )
        print(json.dumps(_json_safe(report), sort_keys=True))
        return 1 if failed else 0
    except BaseException as error:
        print(
            f"task-decoding batch failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
