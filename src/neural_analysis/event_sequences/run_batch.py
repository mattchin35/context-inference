"""Callable session-level batch preparation for event-sequence heatmaps."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count
from .run_session import EventSequenceRequest, prepare_event_sequences


def run_batch(
    requests: Sequence[EventSequenceRequest],
    *,
    workers: int | None = None,
    rerun: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Prepare explicit event-sequence requests and return reports in input order.

    Parameters
    ----------
    requests : Sequence[EventSequenceRequest]
        Nonempty ordered sequence containing one unit or PC request per
        canonical session root. Request settings and physical units are
        defined by EventSequenceRequest; no arrays are accepted here.
    workers : int or None, keyword-only
        Positive requested process count, or None to use the shared
        CPU/session-capped policy. This is validated for dry runs too.
    rerun : bool, keyword-only
        Forward whether each session should create a fresh event-sequence run.
    dry_run : bool, keyword-only
        Plan sequentially without starting workers or loading neural sources.

    Returns
    -------
    list[dict]
        One single-session report per request, preserving input order. Each
        report contains status, result_path, configuration_id, elapsed_s, and
        timings_s as specified by prepare_event_sequences. Elapsed/timing
        values are wall seconds; configuration IDs are unitless strings.

    Raises
    ------
    TypeError
        If requests is not an ordered sequence of EventSequenceRequest values.
    ValueError
        If requests is empty, session roots repeat, or workers is invalid.
    """
    validated_requests = _validate_requests(requests)
    worker_count = choose_worker_count(len(validated_requests), workers)

    if dry_run:
        return [
            prepare_event_sequences(request, rerun=rerun, dry_run=True)
            for request in validated_requests
        ]

    # Spawn starts clean interpreters instead of inheriting the app's threaded state.
    process_context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=process_context) as executor:
        futures = [
            executor.submit(prepare_event_sequences, request, rerun=rerun)
            for request in validated_requests
        ]
        return [future.result() for future in futures]


def _validate_requests(requests: Sequence[EventSequenceRequest]) -> tuple[EventSequenceRequest, ...]:
    """Input: ordered request sequence; return immutable requests or raise for invalid/duplicate roots."""
    if isinstance(requests, (str, bytes)) or not isinstance(requests, Sequence):
        raise TypeError("requests must be a sequence of EventSequenceRequest values.")
    if not requests:
        raise ValueError("requests must not be empty.")

    validated = []
    session_roots = set()
    for request in requests:
        if not isinstance(request, EventSequenceRequest):
            raise TypeError("Every request must be an EventSequenceRequest.")
        session_root = request.metadata_path.resolve().parent
        if session_root in session_roots:
            raise ValueError(f"Duplicate canonical session root: {session_root}")
        session_roots.add(session_root)
        validated.append(request)
    return tuple(validated)
