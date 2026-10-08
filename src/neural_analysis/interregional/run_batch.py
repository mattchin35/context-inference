"""Session-separated batch command boundary for inter-regional regression."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path

from .run_session import SessionRunReport, run_single_session


@dataclass(frozen=True)
class BatchRunReport:
    """Batch worker count and one small status record per configuration."""

    worker_count: int
    reports: tuple[SessionRunReport, ...]
    total_input_bytes: int
    max_session_input_bytes: int


def _positive_int(text: str) -> int:
    """Parse one positive dimensionless worker-count argument."""
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("workers must be a positive integer") from error
    if value <= 0:
        raise argparse.ArgumentTypeError("workers must be a positive integer")
    return value


def read_config_list(path: Path | str) -> tuple[Path, ...]:
    """Read unique configuration paths from nonblank, noncomment UTF-8 lines."""
    list_path = Path(path).resolve(strict=True)
    configs: list[Path] = []
    seen: set[Path] = set()
    for line_number, raw in enumerate(
        list_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = list_path.parent / candidate
        candidate = candidate.resolve(strict=True)
        if not candidate.is_file():
            raise ValueError(f"Config-list entry {line_number} is not a file.")
        if candidate in seen:
            raise ValueError(f"Duplicate resolved config path: {candidate}")
        seen.add(candidate)
        configs.append(candidate)
    if not configs:
        raise ValueError("Config list contains no session configurations.")
    return tuple(configs)


def choose_worker_count(
    session_count: int,
    *,
    requested: int | None,
    cpu_count: int | None = None,
) -> int:
    """Return the CPU/session/request minimum for session-level parallelism."""
    if isinstance(session_count, bool) or session_count <= 0:
        raise ValueError("session_count must be positive.")
    available = os.cpu_count() or 1 if cpu_count is None else cpu_count
    if available <= 0:
        raise ValueError("cpu_count must be positive.")
    limits = [session_count, available]
    if requested is not None:
        if isinstance(requested, bool) or requested <= 0:
            raise ValueError("requested workers must be positive.")
        limits.append(requested)
    return min(limits)


def _run_worker(config_path: Path, rerun: bool) -> SessionRunReport:
    """Execute one configuration in a process worker and return only its status."""
    return run_single_session(config_path, command="new", rerun=rerun)


def run_batch(
    config_paths: tuple[Path, ...],
    *,
    command: str,
    workers: int | None = None,
    rerun: bool = False,
    cpu_count: int | None = None,
) -> BatchRunReport:
    """Plan sequentially or execute configurations across session workers."""
    if command not in {"dry-run", "new"}:
        raise ValueError("command must be 'dry-run' or 'new'.")
    worker_count = choose_worker_count(
        len(config_paths), requested=workers, cpu_count=cpu_count
    )
    if command == "dry-run":
        reports = tuple(
            run_single_session(path, command="dry-run") for path in config_paths
        )
        input_sizes = [report.input_bytes for report in reports]
        return BatchRunReport(
            worker_count,
            reports,
            sum(input_sizes),
            max(input_sizes, default=0),
        )
    reports_by_path: dict[Path, SessionRunReport] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_run_worker, path, rerun): path for path in config_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                reports_by_path[path] = future.result()
            except Exception as error:
                reports_by_path[path] = SessionRunReport(
                    path.stem,
                    "failed",
                    None,
                    None,
                    0,
                    f"{type(error).__name__}: {error}",
                )
    ordered = tuple(reports_by_path[path] for path in config_paths)
    input_sizes = [report.input_bytes for report in ordered]
    return BatchRunReport(
        worker_count,
        ordered,
        sum(input_sizes),
        max(input_sizes, default=0),
    )


def _parser() -> argparse.ArgumentParser:
    """Build the documented dry-run/new batch command-line parser."""
    parser = argparse.ArgumentParser(prog="interregional-regression-batch")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("dry-run", "new"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config-list", required=True)
        subparser.add_argument("--workers", type=_positive_int)
        if command == "new":
            subparser.add_argument("--rerun", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the batch CLI and return failure when any session failed."""
    args = _parser().parse_args(argv)
    paths = read_config_list(args.config_list)
    report = run_batch(
        paths,
        command=args.command,
        workers=args.workers,
        rerun=bool(getattr(args, "rerun", False)),
    )
    for session_report in report.reports:
        print(session_report)
    return 1 if any(value.status == "failed" for value in report.reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
