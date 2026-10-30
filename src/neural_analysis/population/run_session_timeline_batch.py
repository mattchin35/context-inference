"""Prepare timeline PCA or saved-score summaries across independent sessions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import sys

from src.neural_analysis.population.run_session_timeline import (
    TimelineRunReport,
    run_session_timeline,
    probe_roles_from_arguments,
)


def choose_worker_count(
    session_count: int, requested: int | None, *, cpu_count: int | None = None
) -> int:
    """Return a positive session/CPU/request-capped process count (dimensionless).

    session_count must be positive; None requested uses all available CPU cores
    up to session count. Explicit requested workers must be positive integers.
    """
    available = (os.cpu_count() or 1) if cpu_count is None else cpu_count
    if session_count < 1 or available < 1 or (requested is not None and requested < 1):
        raise ValueError("Session, CPU, and worker counts must be positive.")
    return min(
        session_count, available, requested if requested is not None else available
    )


def read_session_list(path: Path | str) -> tuple[Path, ...]:
    """Read existing metadata paths, one per noncomment line, relative to the list file.

    Return unique paths in input order. Empty lists and duplicate sessions raise
    ValueError; no scientific data are loaded or output directories created.
    """
    listing = Path(path).resolve(strict=True)
    paths = []
    for line in listing.read_text().splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        candidate = Path(text)
        resolved = (listing.parent / candidate).resolve(strict=True)
        if resolved in paths:
            raise ValueError(f"Duplicate session: {resolved}")
        paths.append(resolved)
    if not paths:
        raise ValueError("Session list is empty.")
    return tuple(paths)


def run_batch(
    metadata_paths: tuple[Path, ...],
    probe_roles: dict[str, str] | None = None,
    bin_width_s: float = 0.5,
    *,
    workers: int | None = None,
    rerun: bool = False,
    dry_run: bool = False,
    window_summary: bool = False,
) -> list[TimelineRunReport]:
    """Run session-level preparation and return reports in input order.

    metadata_paths are session JSON paths; bin_width_s is seconds and roles
    are explicit probe keys or metadata defaults. Dry-run is sequential and
    never launches workers. Real sessions run in independent processes.
    window_summary selects Stage 2 reduction of existing Stage 1 products;
    missing Stage 1 configurations raise rather than fitting implicitly.
    """
    count = choose_worker_count(len(metadata_paths), workers)
    action = run_session_timeline
    if window_summary:
        from src.neural_analysis.population.run_window_variance import (
            run_session_window_variance,
        )

        action = run_session_window_variance
    if dry_run:
        return [
            action(path, probe_roles, bin_width_s, dry_run=True)
            for path in metadata_paths
        ]
    with ProcessPoolExecutor(max_workers=count) as executor:
        futures = [
            executor.submit(action, path, probe_roles, bin_width_s, rerun=rerun)
            for path in metadata_paths
        ]
        return [future.result() for future in futures]


def main(argv: list[str] | None = None) -> int:
    """Parse session-list CLI arguments; print reports and return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list", type=Path, required=True)
    parser.add_argument(
        "--bin-width-ms", type=int, choices=(500, 100, 50, 20), default=500
    )
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--window-summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = read_session_list(args.session_list)
        roles = probe_roles_from_arguments(args.pfc_probe, args.hpc_probe)
        reports = run_batch(
            paths,
            roles,
            args.bin_width_ms / 1000.0,
            workers=args.workers,
            rerun=args.rerun,
            dry_run=args.dry_run,
            window_summary=args.window_summary,
        )
    except (OSError, ValueError) as error:
        print(f"Timeline batch failed: {error}", file=sys.stderr)
        return 1
    for path, report in zip(paths, reports):
        print(
            f"{path}: {report.status} {report.result_path or report.configuration_id}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
