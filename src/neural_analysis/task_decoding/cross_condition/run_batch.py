"""Prepare cross-condition generalization across independent metadata sessions."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

from src.neural_analysis.population.run_session_timeline_batch import (
    choose_worker_count,
    read_session_list,
)

from .run_session import TransferRunReport, prepare_cross_condition_generalization


def run_transfer_batch(
    metadata_paths: tuple[Path, ...],
    *,
    bin_width_ms: int | None = None,
    workers: int | None = None,
    rerun: bool = False,
    dry_run: bool = False,
) -> list[TransferRunReport]:
    """Prepare sessions and return one small report per input metadata Path.

    metadata_paths is an ordered tuple of version-3 session metadata Paths;
    bin_width_ms is an optional integer width in milliseconds, and workers is
    an optional positive dimensionless process count. Each session retains its
    own contained cross-condition configuration when bin_width_ms is None.
    dry_run plans sessions sequentially and starts no worker processes.
    Reports follow input order and contain paths/identities, elapsed seconds,
    and JSON-safe details without neural arrays.
    """
    worker_count = choose_worker_count(len(metadata_paths), workers)
    arguments = {
        "bin_width_ms": bin_width_ms,
        "rerun": rerun,
        "dry_run": dry_run,
    }
    if dry_run:
        return [
            prepare_cross_condition_generalization(path, **arguments)
            for path in metadata_paths
        ]

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(
                prepare_cross_condition_generalization, path, **arguments
            )
            for path in metadata_paths
        ]
        return [future.result() for future in futures]


def main(argv: list[str] | None = None) -> int:
    """Run the batch CLI and return 0 on success or 1 on ordinary input errors.

    argv is an optional sequence of command-line strings; output is text on
    stdout/stderr and the return value is a dimensionless process exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list", type=Path, required=True)
    parser.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20))
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        metadata_paths = read_session_list(args.session_list)
        reports = run_transfer_batch(
            metadata_paths,
            bin_width_ms=args.bin_width_ms,
            workers=args.workers,
            rerun=args.rerun,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"Transfer batch failed: {error}", file=sys.stderr)
        return 1

    for report in reports:
        print(f"{report.status}: {report.result_path or report.identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
