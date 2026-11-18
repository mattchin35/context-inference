"""Session-level preparation of explicitly supplied behavioral example sessions."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

from .run_examples import prepare_behavior_examples


def run_batch(
    metadata_paths: Sequence[Path | str], *, workers: int | None = None,
    probe_return_outcome: str = "rewarded_or_omission", rerun: bool = False, dry_run: bool = False,
) -> list[dict]:
    """Prepare explicit sessions and return reports in input order.

    metadata_paths is a nonempty ordered sequence of distinct session JSON
    Paths/strings. workers is a positive integer, or None for CPU count capped
    by session count. The return policy/scalar run options follow
    prepare_behavior_examples; dry_run starts no workers and writes no files.
    Return list[dict] single-session status/paths/configuration/wall seconds.
    Parallelism is across sessions; no batch of experimental sessions is inferred.
    """
    if isinstance(metadata_paths, (str, bytes)) or not isinstance(metadata_paths, Sequence) or not metadata_paths:
        raise ValueError("metadata_paths must be a nonempty ordered sequence.")
    paths = [Path(path).resolve() for path in metadata_paths]
    if len({path.parent for path in paths}) != len(paths):
        raise ValueError("A batch cannot repeat a canonical session root.")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0):
        raise ValueError("workers must be a positive integer or None.")
    count = min(workers if workers is not None else (os.cpu_count() or 1), len(paths))
    options = {"probe_return_outcome": probe_return_outcome, "rerun": rerun, "dry_run": dry_run}
    if dry_run or count == 1:
        return [prepare_behavior_examples(path, **options) for path in paths]
    with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = [executor.submit(prepare_behavior_examples, path, **options) for path in paths]
        return [future.result() for future in futures]
