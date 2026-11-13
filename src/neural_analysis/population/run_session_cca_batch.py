"""Prepare independent CCA sessions with ordered, failure-isolated process reports."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

from src.neural_analysis.population.run_session_cca import CCARunReport, run_session_cca
from src.neural_analysis.population.run_session_timeline import probe_roles_from_arguments
from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count, read_session_list


def run_batch(
    metadata_paths: tuple[Path, ...],
    probe_roles: dict[str, str] | None = None,
    bin_width_s: float = 0.5,
    *,
    workers: int | None = None,
    rerun: bool = False,
    dry_run: bool = False,
) -> list[CCARunReport]:
    """Prepare sessions in separate spawn processes and preserve input report order.

    metadata_paths contains existing metadata paths, roles map PFC/HPC or None
    uses each session's defaults, and bin_width_s is seconds. workers defaults
    to CPU count capped by session count. Return one small report per session;
    ordinary failures become failed reports without cancelling other sessions.
    Dry-run is sequential metadata-only work with no pool or output writes.
    """
    count = choose_worker_count(len(metadata_paths), workers)
    if dry_run:
        return [_run_one(path, probe_roles, bin_width_s, rerun=rerun, dry_run=True)
                for path in metadata_paths]
    began = perf_counter()
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_one, path, probe_roles, bin_width_s, rerun=rerun)
                   for path in metadata_paths]
        reports = []
        for future in futures:
            try:
                reports.append(future.result())
            except Exception as error:
                # A process/serialization failure can occur before the worker's own boundary.
                reports.append(CCARunReport("failed", None, "", perf_counter() - began,
                                            f"{type(error).__name__}: {error}"))
    return reports


def main(argv: list[str] | None = None) -> int:
    """Execute a session-list CLI; return nonzero if input validation or any session fails.

    argv is CLI strings or None; bin widths enter in milliseconds and convert
    once to seconds. Relative metadata paths resolve against the list file.
    Reports retain list order and print explicit per-session status/error.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list", type=Path, required=True)
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20), default=500)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = read_session_list(args.session_list)
        reports = run_batch(paths, probe_roles_from_arguments(args.pfc_probe, args.hpc_probe),
                            args.bin_width_ms / 1000, workers=args.workers,
                            rerun=args.rerun, dry_run=args.dry_run)
    except (OSError, ValueError) as error:
        print(f"CCA batch failed: {error}", file=sys.stderr)
        return 1
    for path, report in zip(paths, reports):
        print(f"{path}: {report.status} {report.error or report.result_path or report.configuration_id}")
    return int(any(report.status == "failed" for report in reports))


# Worker boundary is module-level so it is importable/pickleable under spawn.
def _run_one(metadata_path: Path, probe_roles: dict[str, str] | None, bin_width_s: float,
             *, rerun: bool = False, dry_run: bool = False) -> CCARunReport:
    """Run one metadata session, returning a failed report for an ordinary exception.

    Arguments use the single-session runner's path/role/second contracts.
    elapsed_s records this action's wall seconds; no arrays cross the process
    boundary. User interrupts/SystemExit are not swallowed.
    """
    began = perf_counter()
    try:
        return run_session_cca(metadata_path, probe_roles, bin_width_s, rerun=rerun, dry_run=dry_run)
    except Exception as error:
        return CCARunReport("failed", None, "", perf_counter() - began,
                            f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    raise SystemExit(main())
