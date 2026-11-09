"""Thin session-level parallel wrapper for explicit prediction preparation."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

from src.neural_analysis.population.run_session_timeline import probe_roles_from_arguments
from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count, read_session_list

from .run_session import PredictionRunReport, prepare_session_predictions
from .config import ANALYSIS_MODES


def run_prediction_batch(
    metadata_paths: tuple[Path, ...], *, probe_roles: dict[str, str] | None = None,
    bin_width_ms: int | None = None, workers: int | None = None,
    analysis_mode: str | None = None,
    training_window: str | None = None,
    rerun: bool = False, dry_run: bool = False,
) -> list[PredictionRunReport]:
    """Return reports in input path order; parallelize sessions only.

    Paths name existing metadata files. Explicit roles and millisecond width
    apply per session. Default workers is CPU/session capped; dry-run is
    sequential, writes nothing and never starts a worker pool.
    analysis_mode optionally overrides each session's controls with one fixed
    mode identifier; None preserves per-session configuration/defaults.
    training_window optionally selects full/before/after choice-relative seconds
    for every session; None preserves each session's controls/defaults.
    """
    count = choose_worker_count(len(metadata_paths), workers)
    kwargs = {"probe_roles": probe_roles, "bin_width_ms": bin_width_ms,
              "analysis_mode": analysis_mode, "training_window": training_window,
              "rerun": rerun, "dry_run": dry_run}
    if dry_run:
        return [prepare_session_predictions(path, **kwargs) for path in metadata_paths]
    with ProcessPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(prepare_session_predictions, path, **kwargs) for path in metadata_paths]
        return [future.result() for future in futures]


def main(argv: list[str] | None = None) -> int:
    """Parse a metadata session list and return 0 on success, 1 on ordinary errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list", type=Path, required=True)
    parser.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20))
    parser.add_argument("--analysis-mode", choices=ANALYSIS_MODES)
    parser.add_argument("--training-window", choices=("full", "before", "after"))
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        reports = run_prediction_batch(
            read_session_list(args.session_list),
            probe_roles=probe_roles_from_arguments(args.pfc_probe, args.hpc_probe),
            bin_width_ms=args.bin_width_ms, analysis_mode=args.analysis_mode,
            training_window=args.training_window,
            workers=args.workers, rerun=args.rerun, dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"Prediction batch failed: {error}", file=sys.stderr)
        return 1
    for report in reports:
        print(f"{report.status}: {report.result_path or report.identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
