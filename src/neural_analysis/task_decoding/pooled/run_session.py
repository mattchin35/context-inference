"""Thin single-session CLI for pooled task-variable decoding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Sequence


for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def _pipeline_module():
    """Import the numerical pipeline after thread limits are frozen.

    Returns
    -------
    module
        ``src.neural_analysis.task_decoding.pooled.pipeline`` with numerical
        dependencies imported under one-thread execution limits.
    """
    return __import__(
        "src.neural_analysis.task_decoding.pooled.pipeline",
        fromlist=["pipeline"],
    )


def _parser() -> argparse.ArgumentParser:
    """Build the reviewed dry-run/new/resume/status command parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser with no scientific settings beyond the pooled config/run path.
    """
    parser = argparse.ArgumentParser(prog="pooled-task-decoding-session")
    commands = parser.add_subparsers(dest="command", required=True)
    dry_run = commands.add_parser("dry-run")
    dry_run.add_argument("--config", required=True)
    new = commands.add_parser("new")
    new.add_argument("--config", required=True)
    new.add_argument("--detach", action="store_true")
    resume = commands.add_parser("resume")
    resume.add_argument("--run-directory", required=True)
    resume.add_argument("--detach", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--run-directory", required=True)
    private = commands.add_parser("_execute-prepared")
    private.add_argument("--run-directory", required=True)
    return parser


def _launch_detached(run_directory: Path | str) -> None:
    """Start one terminal-independent child for an exact prepared run.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Existing immutable pooled run directory.

    Returns
    -------
    None
        Returns after process creation; the child owns scientific execution.
    """
    directory = Path(run_directory).resolve()
    console_path = directory / "console.log"
    command = [
        sys.executable,
        "-m",
        "src.neural_analysis.task_decoding.pooled.run_session",
        "_execute-prepared",
        "--run-directory",
        str(directory),
    ]
    pipeline = _pipeline_module()
    execution_path = directory / "execution.json"
    execution = pipeline._read_json(execution_path)
    execution["mode"] = "detached"
    pipeline._write_json(execution_path, execution)
    with console_path.open("a", encoding="utf-8") as console:
        process = subprocess.Popen(
            command,
            cwd=directory,
            stdin=subprocess.DEVNULL,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt = {
        "mode": "detached",
        "pid": process.pid,
        "run_directory": str(directory),
    }
    pipeline._write_json(directory / "local_launch.json", receipt)


def _wait_for_detached_receipt(run_directory: Path | str) -> None:
    """Wait briefly for the parent to publish this child's launch receipt.

    Parameters
    ----------
    run_directory : pathlib.Path or str
        Prepared run directory passed to the private detached child.

    Returns
    -------
    None
        Returns after ``local_launch.json`` names the current process. The
        receipt then participates in stable final output-byte measurement.
    """
    directory = Path(run_directory).resolve()
    execution = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
    if execution.get("mode") != "detached":
        return
    receipt_path = directory / "local_launch.json"
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            receipt = None
        if isinstance(receipt, dict) and receipt.get("pid") == os.getpid():
            return
        time.sleep(0.01)
    raise RuntimeError("Matching detached pooled launch receipt was not published.")


def _execute_with_signal_handling(run_directory: Path, pipeline: object) -> None:
    """Run one exact prepared directory with checkpoint-safe TERM handling.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact immutable prepared pooled run directory.
    pipeline : object
        Module-like object exposing ``run_prepared_pooled_decoding``.

    Returns
    -------
    None
        Converts Slurm's pre-timeout SIGTERM into ``KeyboardInterrupt`` so the
        pipeline durably records interruption before the process exits.
    """
    previous_handler = signal.getsignal(signal.SIGTERM)

    def interrupt_for_checkpoint(_signum: int, _frame: object) -> None:
        """Raise the pipeline's existing interruption path on SIGTERM."""
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_for_checkpoint)
    try:
        pipeline.run_prepared_pooled_decoding(run_directory)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse one pooled command and delegate to the public pipeline seams.

    Parameters
    ----------
    argv : sequence[str] or None
        Command arguments excluding the interpreter/module prefix. ``None``
        reads ``sys.argv``.

    Returns
    -------
    int
        Zero on accepted/successful work, 130 on interruption, otherwise one.
    """
    arguments = _parser().parse_args(argv)
    pipeline = _pipeline_module()
    try:
        if arguments.command == "dry-run":
            print(json.dumps(pipeline.plan_pooled_decoding_session(arguments.config), indent=2))
            return 0
        if arguments.command == "new":
            directory = pipeline.prepare_pooled_decoding_run(arguments.config)
            print(directory)
            if arguments.detach:
                _launch_detached(directory)
            else:
                pipeline.run_prepared_pooled_decoding(directory)
            return 0
        if arguments.command == "_execute-prepared":
            directory = Path(arguments.run_directory)
            _wait_for_detached_receipt(directory)
            _execute_with_signal_handling(directory, pipeline)
            return 0
        if arguments.command == "resume":
            directory = Path(arguments.run_directory)
            if arguments.detach:
                _launch_detached(directory)
            else:
                pipeline.run_prepared_pooled_decoding(directory)
            return 0
        if arguments.command == "status":
            print(
                json.dumps(
                    pipeline.inspect_pooled_decoding_status(arguments.run_directory),
                    indent=2,
                )
            )
            return 0
        raise ValueError(f"Unsupported pooled command: {arguments.command}")
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
