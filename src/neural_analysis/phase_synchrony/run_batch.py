"""Run independent choice-window phase synchrony sessions in separate processes."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter

from src.neural_analysis.population.run_session_timeline_batch import (
    choose_worker_count, read_session_list,
)
from .results import PhaseSynchronyConfig, WriterLockError
from .run_session import RunReport, argument_parser, config_from_arguments, run_session


def run_batch(
    metadata_paths: tuple[Path, ...], config: PhaseSynchronyConfig = PhaseSynchronyConfig(),
    *, workers: int | None = None, order: tuple[str, ...] | None = None,
    rerun: bool = False, dry_run: bool = False, prepare_only: str | None = None,
) -> list[RunReport]:
    """Return one report per metadata path in input order; isolate ordinary session errors.

    config preserves explicit units/sites/pairs and Hz/second settings. Real
    runs use independent session processes, capped by CPUs, session count and
    optional positive workers. Dry-run stays sequential and writes nothing.
    Use a lower worker count for recordings whose RAM/I/O needs justify it.
    Duplicate canonical session paths raise before any work, avoiding writers
    for the same session. This function never pools data across sessions.
    """
    count = choose_worker_count(len(metadata_paths), workers)
    if len({Path(path).resolve() for path in metadata_paths}) != len(metadata_paths):
        raise ValueError("Duplicate session metadata paths")
    options = dict(order=order, rerun=rerun, dry_run=dry_run, prepare_only=prepare_only)
    if dry_run:
        return [_run_one(path, config, **options) for path in metadata_paths]
    # Start fresh interpreters rather than inheriting scientific-library threads.
    with ProcessPoolExecutor(max_workers=count, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_run_one, path, config, **options) for path in metadata_paths]
        return [future.result() for future in futures]


def _run_one(path: Path, config: PhaseSynchronyConfig, **options) -> RunReport:
    """Run one metadata path/settings request; turn ordinary input errors into array-free reports."""
    started = perf_counter()
    try:
        return run_session(path, config, **options)
    except (OSError, ValueError, WriterLockError) as error:
        elapsed_s = perf_counter() - started
        stage = {"stage": "input-setup", "selection": (), "interval": None,
                 "status": "failed", "elapsed_s": elapsed_s, "counts": {}, "error": str(error)}
        return RunReport(1, None, (), (stage,), {"total_s": elapsed_s, "first_product_s": None})


def main(argv: list[str] | None = None) -> int:
    """Read a session-list CLI, print ordered array-free reports, and return nonzero on any failure.

    argv contains optional CLI strings. Session-list entries resolve relative
    to the list file; settings delegate unchanged to the single-session API.
    No experimental batch is launched merely by importing this module.
    """
    arguments = argument_parser(batch=True).parse_args(argv)
    try:
        config = config_from_arguments(arguments)
        paths = read_session_list(arguments.session_list)
        reports = run_batch(
            paths, config, workers=arguments.workers,
            order=None if arguments.order is None else tuple(arguments.order),
            rerun=arguments.rerun, dry_run=arguments.dry_run, prepare_only=arguments.prepare_only,
        )
    except (OSError, ValueError) as error:
        print(f"Phase synchrony batch failed: {error}", file=sys.stderr)
        return 1
    for path, report in zip(paths, reports):
        print(json.dumps({"session_metadata": str(path), **asdict(report)}, default=str))
    return int(any(report.status for report in reports))


if __name__ == "__main__":
    raise SystemExit(main())
