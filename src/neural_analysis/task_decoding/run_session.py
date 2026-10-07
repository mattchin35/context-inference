"""Standard-library task-decoding session CLI with lazy scientific imports.

This entrypoint sets one numerical thread before importing the pipeline, then
offers bounded planning/preparation, foreground execution, detached handoff,
and read-only status inspection for one immutable run directory.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Sequence


for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def _utc_now() -> str:
    """Return one whole-second UTC receipt timestamp.

    Returns
    -------
    str
        ISO-8601 UTC timestamp with a trailing ``Z`` and no physical units.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _pipeline_module():
    """Import the numerical pipeline only after thread limits are frozen.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pipeline`` with all scientific
        dependencies imported after OMP/MKL/OpenBLAS are set to one thread.
    """
    return __import__(
        "src.neural_analysis.task_decoding.pipeline",
        fromlist=["pipeline"],
    )


def _wait_for_detached_launch_receipt(run_directory: Path, pipeline: object) -> None:
    """Wait briefly for the parent to publish this detached child's receipt.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared run directory whose execution mode and launch receipt are
        inspected. Paths and process identities have no physical units.
    pipeline : module
        Imported task-decoding pipeline providing owner and receipt identity
        helpers.

    Returns
    -------
    None
        Returns immediately for foreground, Slurm, or test seams without a
        prepared execution record. A detached child returns only after its
        PID/start-token receipt is atomically visible.

    Raises
    ------
    RuntimeError
        If no matching detached receipt appears within five seconds.
    """
    execution_path = run_directory / "execution.json"
    if os.environ.get("SLURM_JOB_ID") or not execution_path.is_file():
        return
    try:
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Prepared execution provenance is unreadable.") from error
    if not isinstance(execution, dict) or execution.get("mode") != "detached":
        return
    owner = pipeline._current_owner(execution)
    receipt_path = run_directory / "local_launch.json"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            receipt = None
        if isinstance(receipt, dict) and pipeline._is_matching_local_receipt(receipt, owner):
            return
        time.sleep(0.01)
    raise RuntimeError("Matching detached launch receipt was not published.")


def _atomic_receipt(path: Path, receipt: dict[str, object]) -> None:
    """Atomically publish one detached-child ownership receipt.

    Parameters
    ----------
    path : pathlib.Path
        ``local_launch.json`` destination in an immutable run directory.
    receipt : dict[str, object]
        JSON-safe child owner mode, host, PID, start-token, UTC timestamp, and
        optional job identity. Values are execution facts without physical units.

    Returns
    -------
    None
        Writes a unique sibling temporary, fsyncs it, and replaces the receipt.
    """
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(
                (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            )
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


def _launch_detached_prepared(*, run_directory: Path, child_argv: Sequence[str]) -> Path:
    """Launch exactly one terminal-independent private prepared-execution child.

    Parameters
    ----------
    run_directory : pathlib.Path
        Existing immutable prepared run directory.
    child_argv : sequence[str]
        Exact Python private-CLI argv ending in ``_execute-prepared`` and this
        run directory. It changes no scientific settings.

    Returns
    -------
    pathlib.Path
        Atomically published ``local_launch.json`` receipt. Child output is
        redirected to UTF-8 ``console.log`` below the same run directory.
    """
    console_path = run_directory / "console.log"
    with console_path.open("a", encoding="utf-8") as console:
        process = subprocess.Popen(
            list(child_argv),
            cwd=run_directory,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pipeline = _pipeline_module()
    receipt_path = run_directory / "local_launch.json"
    try:
        _atomic_receipt(
            receipt_path,
            {
                "mode": "detached",
                "host": platform.node(),
                "pid": process.pid,
                "start_token": pipeline._process_start_token(process.pid),
                "started_at": _utc_now(),
                "job_id": None,
            },
        )
    except BaseException:
        # A receipt is the handoff guarantee.  Without it, do not leave a
        # detached decoder running without an owner record the user can inspect.
        process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        raise
    return receipt_path


def _parser() -> argparse.ArgumentParser:
    """Build the small public/private command parser without numerical imports.

    Returns
    -------
    argparse.ArgumentParser
        Parser for dry-run, new, resume, status, and private bridge commands.
    """
    parser = argparse.ArgumentParser(prog="task-decoding-session")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--config", required=True)
    new = subparsers.add_parser("new")
    new.add_argument("--config", required=True)
    new.add_argument("--rerun", action="store_true")
    new.add_argument("--detach", action="store_true")
    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-directory", required=True)
    resume.add_argument("--detach", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--run-directory", required=True)
    status.add_argument("--verify-results", action="store_true")
    private_prepare = subparsers.add_parser("_prepare")
    private_prepare.add_argument("--config", required=True)
    private_execute = subparsers.add_parser("_execute-prepared")
    private_execute.add_argument("--run-directory", required=True)
    return parser


def _child_argv(run_directory: Path) -> list[str]:
    """Build the private detached-child argv for one prepared directory.

    Parameters
    ----------
    run_directory : pathlib.Path
        Immutable prepared run directory passed unchanged to the child.

    Returns
    -------
    list[str]
        Interpreter/module private-execute argv with no scientific overrides.
    """
    return [
        sys.executable,
        "-m",
        "src.neural_analysis.task_decoding.run_session",
        "_execute-prepared",
        "--run-directory",
        str(run_directory),
    ]


def _print_detached_success(run_directory: Path, receipt_path: Path) -> None:
    """Print bounded detached-launch guidance after atomic receipt publication.

    Parameters
    ----------
    run_directory : pathlib.Path
        Prepared immutable run directory.
    receipt_path : pathlib.Path
        Existing atomically published detached-child receipt.

    Returns
    -------
    None
        Prints directory, receipt, console log, and exact saved status/resume
        commands for a human shell; no polling or scientific work occurs.
    """
    prefix = "uv run python -m src.neural_analysis.task_decoding.run_session"
    resume_path = run_directory / "resume_command.txt"
    status_path = run_directory / "status_command.txt"
    resume = (
        resume_path.read_text(encoding="utf-8").strip()
        if resume_path.exists()
        else f"{prefix} resume --run-directory {shlex.quote(str(run_directory))} --detach"
    )
    status = (
        status_path.read_text(encoding="utf-8").strip()
        if status_path.exists()
        else f"{prefix} status --run-directory {shlex.quote(str(run_directory))}"
    )
    print(f"Detached task-decoding launch accepted: {run_directory}")
    print(f"Receipt: {receipt_path}")
    print(f"Console log: {run_directory / 'console.log'}")
    print(f"Resume: {resume}")
    print(f"Status: {status}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one bounded session CLI command and return a shell exit code.

    Parameters
    ----------
    argv : sequence[str] or None, default=None
        Command tokens excluding the program name; None uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Zero for accepted dry-run/preparation/execution/status/detached handoff,
        one for any validation, interruption, or execution error. Public
        detached/submission routes report launch acceptance only.
    """
    arguments = _parser().parse_args(argv)
    try:
        pipeline = _pipeline_module()
        if arguments.command == "dry-run":
            print(json.dumps(pipeline.plan_task_decoding_session(arguments.config), default=str))
            return 0
        if arguments.command == "_prepare":
            pipeline.prepare_task_decoding_run(arguments.config, False, "foreground")
            return 0
        if arguments.command == "_execute-prepared":
            prepared = Path(arguments.run_directory)
            _wait_for_detached_launch_receipt(prepared, pipeline)
            pipeline.run_prepared_task_decoding(prepared)
            return 0
        if arguments.command == "status":
            status = pipeline.inspect_task_decoding_status(
                Path(arguments.run_directory),
                verify_results=arguments.verify_results,
            )
            print(json.dumps(status, default=str, sort_keys=True))
            return 0
        if arguments.command == "new":
            mode = "detached" if arguments.detach else "foreground"
            prepared = pipeline.prepare_task_decoding_run(arguments.config, arguments.rerun, mode)
            if pipeline._validate_complete_prepared_run(prepared):
                print(f"Task-decoding run already complete: {prepared}")
                return 0
            if arguments.detach:
                pipeline._reject_live_owner(
                    prepared,
                    prepared / "local_launch.json",
                    "receipt",
                )
                pipeline._reject_live_owner(
                    prepared,
                    prepared / "execution_guard.json",
                    "guard",
                )
                receipt = _launch_detached_prepared(
                    run_directory=prepared,
                    child_argv=_child_argv(prepared),
                )
                _print_detached_success(prepared, receipt)
                return 0
            pipeline.run_prepared_task_decoding(prepared)
            return 0
        if arguments.command == "resume":
            prepared = Path(arguments.run_directory)
            if pipeline._validate_complete_prepared_run(prepared):
                print(f"Task-decoding run already complete: {prepared}")
                return 0
            if arguments.detach:
                pipeline._reject_live_owner(
                    prepared,
                    prepared / "local_launch.json",
                    "receipt",
                )
                pipeline._reject_live_owner(
                    prepared,
                    prepared / "execution_guard.json",
                    "guard",
                )
                receipt = _launch_detached_prepared(
                    run_directory=prepared,
                    child_argv=_child_argv(prepared),
                )
                _print_detached_success(prepared, receipt)
                return 0
            pipeline.run_prepared_task_decoding(prepared)
            return 0
        raise ValueError("Unknown task-decoding CLI command.")
    except BaseException as error:
        print(f"task-decoding command failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
