"""Offline context encoding commands; set numerical thread limits before imports."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
from types import ModuleType
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from .configuration import EncodingConfig


for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def run_encoding_session(
    config_path: Path | str | None = None,
    *,
    command: str = "new",
    run_directory: Path | str | None = None,
    rerun: bool = False,
) -> dict:
    """Plan, prepare, resume, or inspect one immutable session analysis.

    Inputs are a scalar config Path for new/dry-run, or a saved run-directory
    Path for resume/status. rerun requests a new timestamped run for new only.
    Return a JSON-compatible status/path/count mapping; time values are seconds.
    New work uses UTC source times and unsmoothed counts through preparation.
    Resume with published preparation and status never reload live inputs.
    Errors propagate to Python callers; main renders them as command errors.
    """
    from . import pipeline, results

    if command not in ("new", "dry-run", "resume", "status"):
        raise ValueError("command must be new, dry-run, resume, or status")
    if rerun and command != "new":
        raise ValueError("rerun applies only to new")
    if command in ("resume", "status"):
        if run_directory is None or config_path is not None:
            raise ValueError("resume/status require only a run_directory")
        directory = Path(run_directory).resolve(strict=True)
        if command == "status":
            return {**results.inspect_run_status(directory), "run_directory": str(directory)}
        manifest = results.load_run_manifest(directory)
        config = results.restore_config(manifest["config"])
        preparation_path = config.session_root / manifest["preparation_path"]
        if not preparation_path.exists():
            identities = results.compute_run_identities(config)
            if any(identities[key] != manifest["identities"][key]
                   for key in ("preparation_id", "model_id", "code_identity")):
                raise ValueError("Inputs or numerical code changed before preparation; start a new run")
            _publish_preparation(config, preparation_path, identities)
        return _execute(directory, pipeline, results)

    if config_path is None or run_directory is not None:
        raise ValueError("new/dry-run require only a config_path")
    from .configuration import load_encoding_config

    config = load_encoding_config(config_path)
    if command == "dry-run":
        return _plan(config)
    identities = results.compute_run_identities(config)
    if not rerun:
        for directory in sorted(config.run_root.glob("context_encoding_*"), reverse=True):
            if not (directory / "config.json").is_file():
                continue
            manifest = results.load_run_manifest(directory)
            if manifest["identities"]["model_id"] == identities["model_id"]:
                progress = results.inspect_run_status(directory)
                return {"status": "reused" if progress["status"] == "complete" else "partial",
                        "run_directory": str(directory), "progress": progress}
    preparation_path = config.intermediate_root / identities["preparation_id"] / "preparation.npz"
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = config.run_root / f"context_encoding_{stamp}_{identities['model_id'][:12]}"
    results.initialize_run(config, directory, preparation_path, identities)
    shutil.copyfile(__file__, directory / "run_session.py")
    shutil.copyfile(Path(__file__).with_name("run_batch.py"), directory / "run_batch.py")
    _append_log(directory, f"Session {config.session_root}; identities {json.dumps(identities, sort_keys=True)}")
    _write_summary(directory, config, {"status": "preparing", "completed_cells": {}})
    try:
        if preparation_path.exists():
            saved = results.load_preparation(preparation_path)
            if saved["meta"]["preparation_id"] != identities["preparation_id"]:
                raise ValueError("Published preparation identity differs; refusing to overwrite")
            del saved
            _append_log(directory, f"Reusing immutable preparation {preparation_path}")
        else:
            _publish_preparation(config, preparation_path, identities)
        return _execute(directory, pipeline, results)
    except Exception as error:
        _append_log(directory, f"ERROR {type(error).__name__}: {error}")
        raise


def main(argv: list[str] | None = None) -> int:
    """Parse CLI string arguments; print a JSON report and return integer exit status.

    argv is a sequence of command/control strings, or None for sys.argv.
    No units apply to arguments except the saved configuration's bin width.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("new", "dry-run", "resume", "status"):
        subparser = commands.add_parser(command)
        if command in ("new", "dry-run"):
            subparser.add_argument("--config", type=Path, required=True)
        else:
            subparser.add_argument("--run-directory", type=Path, required=True)
        if command == "new":
            subparser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_encoding_session(getattr(args, "config", None), command=args.command,
                                      run_directory=getattr(args, "run_directory", None),
                                      rerun=getattr(args, "rerun", False))
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"Context encoding failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


# Orchestration helpers; scientific calculations stay in preparation/modeling.
def _plan(config: EncodingConfig) -> dict:
    """Input EncodingConfig; return small-input plan/counts/blockers without outputs.

    Only metadata, behavioral tables and NPZ member names are inspected. Neural
    coverage, unit/PC counts and actual fit eligibility remain pending.
    """
    import numpy as np
    from .preparation import load_behavior_inputs
    from src.neural_analysis.session_metadata import resolve_probe_sources

    session, _, behavior = load_behavior_inputs(config)
    blockers = {}
    for role, probe in config.probe_roles.items():
        source = resolve_probe_sources(session, probe)
        if source.alignment_file is None or not source.alignment_file.is_file():
            blockers[role] = "Aligned spike archive is missing"
            continue
        try:
            with np.load(source.alignment_file, allow_pickle=False) as archive:
                if "irig_utc_unix" not in archive.files:
                    blockers[role] = "Manual alignment is deferred; IRIG-backed recording coverage is required"
                elif "spike_utc_unix" not in archive.files:
                    blockers[role] = "Aligned archive lacks spike_utc_unix"
        except (OSError, ValueError) as error:
            blockers[role] = str(error)
        for field in ("cluster_info_file", "channel_quality_file", "spike_clusters_file"):
            path = getattr(source, field)
            if path is None or not path.is_file():
                blockers.setdefault(role, f"Required {field} is missing")
    # Signed same/matched ridge and penalty-free OLS each need only one fit.
    per_scope = sum(int("ols" in config.estimators) + int("ridge" in config.estimators)
                    * (1 if encoding == "signed_context" else len(config.penalty_modes))
                    for encoding in config.encodings)
    return {"status": "dry-run", "session_id": session.session_id,
            "baseline_trials": len(behavior.choices), "input_blockers": blockers,
            "planned_cells": len(config.probe_roles) * len(config.windows) * len(config.timing_modes)
                             * per_scope,
            "neural_counts_pending": True,
            "context_event_diagnostics": behavior.meta["context_event_diagnostics"]}


def _publish_preparation(config: EncodingConfig, path: Path, identities: dict) -> None:
    """Input config/new NPZ Path/identities; save one fixed PCA snapshot, return None.

    Saved score axes are (window,bin,original PC), with UTC/elapsed/relative
    seconds retained by preparation. Runtime is recorded in elapsed seconds.
    """
    from . import preparation, results

    started = perf_counter()
    session, trials, behavior = preparation.load_behavior_inputs(config)
    regions, unavailable = preparation.prepare_regional_activity(config, session, trials, behavior)
    results.save_preparation(path, trials, behavior, regions,
                             {**identities, "generator": __name__, "unavailable": unavailable,
                              "preparation_elapsed_s": perf_counter() - started,
                              "randomness_used": False, "random_seed": None})


def _execute(directory: Path, pipeline_module: ModuleType, results_module: ModuleType) -> dict:
    """Input run Path/pipeline/results modules; fit unfinished cells and report progress.

    All fitting consumes immutable saved preparation. Wall time is seconds;
    reference/CV scope and scientific availability stay in individual products.
    """
    started = perf_counter()
    _append_log(directory, "Executing unfinished saved-preparation cells")
    progress = pipeline_module.run_prepared_encoding(directory)
    manifest = results_module.load_run_manifest(directory)
    config = results_module.restore_config(manifest["config"])
    _write_summary(directory, config, progress)
    _append_log(directory, f"Status {progress['status']}; elapsed {perf_counter() - started:.6f} s; "
                          f"completed cells {len(progress['completed_cells'])}")
    return {"status": progress["status"], "run_directory": str(directory), "progress": progress}


def _append_log(directory: Path, message: str) -> None:
    """Input run Path/message string; append timestamped execution text, return None."""
    with (directory / "run.log").open("a", encoding="ascii") as stream:
        stream.write(f"{datetime.now(ZoneInfo('America/New_York')).isoformat()} {message}\n")


def _write_summary(directory: Path, config: EncodingConfig, progress: dict) -> None:
    """Input run Path/config/progress; update this run's human-readable summary, return None.

    Bin width is milliseconds; coefficients/predictions preserve PC-score units.
    This mutable execution summary never replaces a scientific NPZ product.
    """
    text = ("# Context encoding\n\n"
            f"Session: {config.session_root.name}\n\n"
            "Goal: inspect pooled bin-level true-context associations on fixed regional PC axes.\n\n"
            "Scripts: run_session.py; run_batch.py.\n\n"
            f"Bin width: {config.bin_width_ms} ms; choice window [-2,+2) seconds.\n\n"
            f"Status: {progress['status']}; completed cells: {len(progress['completed_cells'])}.\n\n"
            "Reference coefficients use all eligible rows. Held-out scores evaluate regression "
            "conditional on descriptive session PCA, which includes test-block neural activity.\n\n"
            "No significance tests were performed. Missing CV remains distinct from zero scores.\n")
    (directory / "summary.md").write_text(text, encoding="ascii")


if __name__ == "__main__":
    raise SystemExit(main())
