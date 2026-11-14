"""Session-level population decoding workers; no nested numerical parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def choose_worker_count(session_count: int, requested: int | None, *, cpu_count: int | None = None) -> int:
    """Return CPU/session-capped positive worker count from dimensionless integer controls."""
    from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count as existing_count
    if any(value is not None and (type(value) is not int or value < 1) for value in (session_count, requested, cpu_count)):
        raise ValueError("Worker/session/CPU counts must be positive integers")
    return existing_count(session_count, requested, cpu_count=cpu_count)


def run_batch(config_paths: tuple[Path, ...], *, command: str = "new", workers: int | None = None,
              rerun: bool = False) -> dict:
    """Run unique session config Paths in spawned workers; return ordered JSON reports.

    command is new/dry-run, workers is a positive count or CPU default. Dry-run
    uses no process pool or writes. Failed sessions retain explicit errors while
    other independent sessions finish. Each session runs sequentially internally.
    """
    from .configuration import load_config
    if command not in ("new", "dry-run"):
        raise ValueError("Batch command must be new or dry-run")
    paths = tuple(Path(path).resolve(strict=True) for path in config_paths)
    count = choose_worker_count(len(paths), workers)
    roots = [load_config(path).session_root for path in paths]
    if len(set(roots)) != len(roots):
        raise ValueError("Duplicate sessions in configuration list")
    if command == "dry-run":
        from .run_session import run_session
        return {"status": "dry-run", "worker_count": 0, "planned_workers": count,
                "reports": [run_session(path, command="dry-run") for path in paths]}
    reports = []
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_worker, path, rerun) for path in paths]
        for path, future in zip(paths, futures):
            try:
                reports.append(future.result())
            except Exception as error:
                reports.append({"status": "failed", "config": str(path), "error": f"{type(error).__name__}: {error}"})
    return {"status": "complete" if all(item["status"] in ("complete", "reused") for item in reports) else "partial",
            "worker_count": count, "reports": reports}


def main(argv: list[str] | None = None) -> int:
    """Input CLI strings or sys.argv; read config-list Path and print JSON batch status."""
    from src.neural_analysis.population.run_session_timeline_batch import read_session_list
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("new", "dry-run"):
        sub = commands.add_parser(name)
        sub.add_argument("--config-list", type=Path, required=True)
        sub.add_argument("--workers", type=int)
        if name == "new":
            sub.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_batch(read_session_list(args.config_list), command=args.command,
                           workers=args.workers, rerun=getattr(args, "rerun", False))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] in ("complete", "dry-run") else 1


def _run_worker(path: Path, rerun: bool) -> dict:
    """Input one config Path/rerun bool; return one independent session report."""
    from .run_session import run_session
    return run_session(path, rerun=rerun)


if __name__ == "__main__":
    raise SystemExit(main())
