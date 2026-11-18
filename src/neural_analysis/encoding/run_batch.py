"""Session-separated context encoding execution with lightweight command imports."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys


for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def choose_worker_count(session_count: int, requested: int | None, *, cpu_count: int | None = None) -> int:
    """Input positive integer session/CPU/request counts; return capped worker count.

    None requested uses available CPUs, up to session count. Counts are
    dimensionless; reject Boolean/noninteger overrides before existing helper use.
    """
    values = (session_count, requested, cpu_count)
    if any(value is not None and (type(value) is not int or value < 1) for value in values):
        raise ValueError("Session, CPU and worker counts must be positive integers")
    from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count as existing_count

    return existing_count(session_count, requested, cpu_count=cpu_count)


def read_config_list(path: Path | str) -> tuple[Path, ...]:
    """Read a scalar list Path; return unique existing config Paths in listed order.

    Noncomment/nonempty lines resolve relative to the list file. Paths have no
    physical units; the existing path-list helper handles canonical duplicates.
    """
    from src.neural_analysis.population.run_session_timeline_batch import read_session_list

    paths = read_session_list(path)
    if not all(candidate.is_file() for candidate in paths):
        raise ValueError("Every configuration-list entry must name a file")
    return paths


def run_encoding_batch(
    config_paths: tuple[Path, ...], *, command: str = "new", workers: int | None = None,
    rerun: bool = False,
) -> dict:
    """Plan sequentially or execute one independent process per selected session.

    config_paths is a nonempty sequence of scalar JSON Paths, command is new or
    dry-run, workers is a positive dimensionless override, rerun creates new runs.
    Return ordered JSON-compatible reports/status and actual worker count.
    Canonically identical session roots are rejected even with distinct configs.
    Dry-run never creates a pool. Failures remain explicit alongside other reports.
    """
    from .configuration import load_encoding_config

    if command not in ("new", "dry-run"):
        raise ValueError("Batch command must be new or dry-run")
    paths = tuple(Path(path).resolve(strict=True) for path in config_paths)
    count = choose_worker_count(len(paths), workers)
    roots = [load_encoding_config(path).session_root for path in paths]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate canonical sessions in the batch configuration list")
    if command == "dry-run":
        from .run_session import run_encoding_session

        return {"status": "dry-run", "worker_count": 0, "planned_workers": count,
                "reports": [run_encoding_session(path, command="dry-run") for path in paths]}
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_worker, path, rerun) for path in paths]
        reports = []
        for path, future in zip(paths, futures):
            try:
                reports.append(future.result())
            except Exception as error:
                reports.append({"status": "failed", "config": str(path),
                                "error": f"{type(error).__name__}: {error}"})
    status = "complete" if all(item["status"] in ("complete", "reused") for item in reports) else "partial"
    return {"status": status, "worker_count": count, "reports": reports}


def main(argv: list[str] | None = None) -> int:
    """Input CLI string list or None; print JSON batch report and return integer status."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("new", "dry-run"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config-list", type=Path, required=True)
        subparser.add_argument("--workers", type=int)
        if command == "new":
            subparser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_encoding_batch(read_config_list(args.config_list), command=args.command,
                                    workers=args.workers, rerun=getattr(args, "rerun", False))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Context encoding batch failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in ("complete", "dry-run") else 1


def _run_worker(path: Path, rerun: bool) -> dict:
    """Input config Path/rerun flag; return one session's JSON report from a worker."""
    from .run_session import run_encoding_session

    return run_encoding_session(path, rerun=rerun)


if __name__ == "__main__":
    raise SystemExit(main())
