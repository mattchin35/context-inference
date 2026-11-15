"""Thin session-level workers for saved decoder unit activity requests."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
import os
from pathlib import Path

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def read_requests(path: Path | str) -> tuple[dict, ...]:
    """Validate JSON list of {source_run,regions}; return absolute run Paths and region tuples.

    Paths resolve against the request file directory. One entry/session, with
    unique nonempty PFC/HPC region subset. No source arrays or outputs are loaded.
    """
    path = Path(path).resolve()
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("Requests must be a nonempty JSON array")
    requests, roots = [], set()
    for entry in entries:
        if (not isinstance(entry, dict) or set(entry) != {"source_run", "regions"}
                or not isinstance(entry["source_run"], str) or not entry["source_run"]
                or not isinstance(entry["regions"], list) or not entry["regions"]
                or any(region not in ("PFC", "HPC") for region in entry["regions"])
                or len(set(entry["regions"])) != len(entry["regions"])):
            raise ValueError("Each request needs source_run and a unique nonempty PFC/HPC regions list")
        directory = (path.parent / entry["source_run"]).resolve()
        if not directory.is_dir() or directory.parent.name != "analysis_runs":
            raise ValueError("Source run must exist under its session's analysis_runs directory")
        root = directory.parent.parent
        if root in roots:
            raise ValueError("Duplicate session roots in batch requests")
        roots.add(root)
        requests.append({"source_run": directory, "regions": tuple(entry["regions"])})
    return tuple(requests)


def run_batch(request_file: Path | str, *, command: str = "new", workers: int | None = None,
               max_units_per_group: int = 10, scale: str = "zscore") -> dict:
    """Run JSON session requests in spawned workers; return ordered JSON-safe region reports.

    command is new/dry-run; workers is positive or CPU/session default. Regions
    within a session run sequentially; independent failures produce partial
    status without canceling other sessions. Dry-run writes nothing/no workers.
    """
    from src.neural_analysis.population_decoding.run_batch import choose_worker_count
    if command not in ("new", "dry-run"):
        raise ValueError("Batch command must be new or dry-run")
    requests = read_requests(request_file)
    count = choose_worker_count(len(requests), workers)
    options = {"command": command, "max_units_per_group": max_units_per_group, "scale": scale}
    if command == "dry-run":
        reports = [report for request in requests for report in _run_worker(request, options)]
        return {"status": "dry-run", "worker_count": 0, "planned_workers": count, "reports": reports}
    reports = []
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_worker, request, options) for request in requests]
        for request, future in zip(requests, futures):
            try:
                reports.extend(future.result())
            except Exception as error:
                reports.append({"status": "failed", "source_run": str(request["source_run"]), "reason": str(error)})
    return {"status": "complete" if all(report["status"] == "complete" for report in reports) else "partial",
            "worker_count": count, "reports": reports}


def main(argv: list[str] | None = None) -> int:
    """Read CLI request-file/worker/cap/scale controls and print JSON report; return exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("new", "dry-run"))
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--max-units-per-group", type=int, default=10)
    parser.add_argument("--scale", choices=("zscore", "hz"), default="zscore")
    args = parser.parse_args(argv)
    try:
        report = run_batch(args.requests, command=args.command, workers=args.workers,
                           max_units_per_group=args.max_units_per_group, scale=args.scale)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] in ("complete", "dry-run") else 1


def _run_worker(request: dict, options: dict) -> list[dict]:
    """Run one session's requested regions sequentially; return status/error mappings, no raw input."""
    from .run_session import run_session
    reports = []
    for region in request["regions"]:
        try:
            reports.append(run_session(request["source_run"], region, **options))
        except Exception as error:
            reports.append({"status": "failed", "region": region, "source_run": str(request["source_run"]),
                            "reason": f"{type(error).__name__}: {error}"})
    return reports


if __name__ == "__main__":
    raise SystemExit(main())
