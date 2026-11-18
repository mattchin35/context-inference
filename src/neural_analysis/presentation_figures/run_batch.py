"""Compute or reuse all session analyses and publish the named cohort summary."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

from src.neural_analysis.population.run_session_timeline_batch import choose_worker_count, read_session_list
from . import results
from .cohort import assemble_cohort
from .run_session import new_run_directory, run_session


def _run_one(path: Path, *, seed: int, dry_run: bool, rerun: bool) -> dict:
    """Return the session report or a roster entry explaining a failed input."""
    try:
        return run_session(path, seed=seed, dry_run=dry_run, rerun=rerun)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        return dict(status="failed", reason=f"{type(error).__name__}: {error}", manifest=None,
                    session_id=path.parent.name, metadata=str(path))


def run_batch(metadata_paths: list[Path] | tuple[Path, ...], *, output_root: Path,
              workers: int | None = None, seed: int = 0, dry_run: bool = False,
              rerun: bool = False) -> dict:
    """Run ordered session metadata Paths with one numerical-library thread per worker.

    Output root holds timestamped cohort directories. Session NPZs remain under
    each session. Return a JSON-compatible report with statuses in roster order.
    A dry run inspects metadata and coverage without creating run directories.
    """
    started = perf_counter()
    paths = [Path(p).resolve() for p in metadata_paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate metadata paths in cohort.")
    count = choose_worker_count(len(paths), workers)
    action = partial(_run_one, seed=seed, dry_run=dry_run, rerun=rerun)
    if dry_run or count == 1:
        sessions = [action(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=count) as pool:
            sessions = list(pool.map(action, paths))
    report = dict(sessions=sessions, workers=count, manifest=None)
    if dry_run:
        return dict(report, status="dry-run")
    # Exact constituent manifests make an explicit session rerun distinguishable.
    digest = hashlib.sha256(Path(__file__).with_name("cohort.py").read_bytes())
    digest.update(json.dumps([(s["session_id"], s["manifest"], s.get("reason", ""))
                              for s in sessions], sort_keys=True).encode())
    identity = digest.hexdigest()
    existing = results.discover_runs(output_root, identity)
    if existing and not rerun:
        return dict(report, status="reused", manifest=str(existing[0]))
    fields = assemble_cohort(sessions)
    ids = [s["session_id"] for s in sessions]
    directory = new_run_directory(output_root)
    meta = dict(producer="presentation_figures.assemble_cohort", seed=seed, session_ids=ids,
                parameters={"alpha": 0.1, "draws": 5}, axes={"tables": "one row per labeled observation"},
                units={"balanced_accuracy": "fraction", "mean_r2": "dimensionless"})
    results.save_result(directory / "cohort.npz", fields, meta)
    for name in ("run_session.py", "run_batch.py"):
        shutil.copyfile(Path(__file__).with_name(name), directory / name)
    elapsed = perf_counter() - started
    (directory / "run.log").write_text(json.dumps(dict(report, seed=seed, elapsed_s=elapsed), indent=2))
    lines = ["# PFC and HPC presentation cohort", "",
             "Figures 7 and 8 summarize held-out regional decoding and same-bin neural prediction.",
             "run_batch.py called run_session.py for each named session.", ""]
    lines.extend(f"- {s['session_id']}: {s['status']}. {s.get('reason', '')}" for s in sessions)
    (directory / "summary.md").write_text("\n".join(lines) + "\n")
    manifest = results.complete_run(directory, kind="cohort", identity=identity,
                                    session_ids=ids, archives={"cohort": "cohort.npz"})
    return dict(report, status="computed", manifest=str(manifest))


def main(argv: list[str] | None = None) -> int:
    """Parse CLI strings, print ordered reports and return nonzero for failed sessions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    report = run_batch(read_session_list(args.session_list), output_root=args.output_root,
                       workers=args.workers, seed=args.seed, dry_run=args.dry_run, rerun=args.rerun)
    print(json.dumps(report, indent=2))
    return int(any(s["status"] == "failed" for s in report["sessions"]))


if __name__ == "__main__":
    raise SystemExit(main())
