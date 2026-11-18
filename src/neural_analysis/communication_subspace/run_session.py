"""Explicit single-session communication-subspace run and reuse lifecycle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np

from .computation import ALPHA_GRID, compute_communication_subspace
from .configuration import ANALYSIS_VERSION, CommunicationConfig
from .preparation import (
    build_validation_splits,
    draw_populations,
    inspect_session_inputs,
    load_session_inputs,
)
from .results import CommunicationResult, discover_results, load_result, save_result


@dataclass(frozen=True)
class RunReport:
    """Small result of an explicit run, reuse, or metadata-only dry run.

    status is dry-run/reused/prepared/partial/unavailable. result_path and
    run_directory are local Paths or None for dry-run; configuration_id is a
    scientific/source/code digest. elapsed_s is wall-clock seconds.
    """

    status: str
    result_path: Path | None
    run_directory: Path | None
    configuration_id: str
    elapsed_s: float


def run_session(
    config: CommunicationConfig, *, rerun: bool = False,
    dry_run: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
) -> RunReport:
    """Preflight, reuse or compute one paired session and save a new run.

    config supplies metadata Path, PFC/HPC roles, 100/500-ms bins and seed.
    dry_run checks labels, trusted coverage, n>=2 and five purged splits with
    no raw spikes/output writes. A compatible validated success archive is
    reused unless rerun=True. Computed runs save a trusted NPZ, log, script
    snapshot, configuration, summary and PNG under the session root. progress
    receives (completed,total,message) for 25 draw/fold tasks. Return RunReport.
    """
    started = perf_counter()
    inspection = inspect_session_inputs(config)
    root = inspection.session.session_root.resolve()
    _validate_session_output_root(root)
    draw_populations(inspection.unit_ids, seed=config.seed)
    times = inspection.choices.iloc[inspection.choice_positions].choice_time.to_numpy(dtype=float)
    build_validation_splits(times)
    effective = _effective_configuration(inspection.configuration)
    identity = effective["configuration_id"]
    if dry_run:
        return RunReport("dry-run", None, None, identity, perf_counter() - started)
    if not rerun:
        for info in discover_results(root, identity):
            try:
                saved = load_result(info.path)
            except (OSError, ValueError, KeyError, EOFError):
                continue
            if saved.computation.status == "success":
                run_dir = root / "communication_subspace" / "runs" / info.run_id
                return RunReport("reused", info.path, run_dir, identity, perf_counter() - started)

    now_local = datetime.now(ZoneInfo("America/New_York"))
    run_id = f"communication_subspace_{now_local:%Y%m%d_%H%M%S_%f}_{identity}"
    base = root / "communication_subspace"
    run_dir = base / "runs" / run_id
    result_path = base / "results" / f"{run_id}.npz"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(effective, indent=2, sort_keys=True), encoding="ascii")
    shutil.copyfile(Path(__file__), run_dir / "run_session.py")
    log_path = run_dir / "run.log"
    started_utc = datetime.now(timezone.utc).isoformat()
    log_path.write_text(
        f"started_utc={started_utc}\n"
        f"session_id={inspection.session_id}\n"
        f"parameters={json.dumps(effective, sort_keys=True)}\n",
        encoding="ascii",
    )
    try:
        prepared = load_session_inputs(config)
        if prepared.configuration["configuration_id"] != inspection.configuration["configuration_id"]:
            raise ValueError("Source configuration changed during preparation.")
        computation = compute_communication_subspace(prepared, seed=config.seed, progress=progress)
        _require_unchanged_identity(config, inspection.configuration["configuration_id"], identity)
        provenance = _run_provenance(effective, config.seed, started_utc)
        result = CommunicationResult(
            prepared.session_id, prepared.time_origin_s, prepared.choices,
            prepared.selected_trials, prepared.choice_positions, prepared.unit_ids,
            prepared.bin_edges_s, prepared.probe_roles, computation, provenance,
        )
        save_rank_figure(result, run_dir / "rank_curves.png")
        (run_dir / "summary.md").write_text(_summary(result), encoding="ascii")
        _require_unchanged_identity(config, inspection.configuration["configuration_id"], identity)
        elapsed = perf_counter() - started
        with log_path.open("a", encoding="ascii") as log:
            log.write(_run_log(result, elapsed))
        # The result is linked into place last. A failed plot or log cannot
        # leave a discoverable completed scientific archive.
        save_result(result_path, result)
    except Exception as error:
        with log_path.open("a", encoding="ascii") as log:
            log.write(f"error={type(error).__name__}: {error}\n")
            log.write(traceback.format_exc())
            log.write(f"elapsed_s={perf_counter() - started:.6f}\n")
        raise
    status = "prepared" if computation.status == "success" else computation.status
    return RunReport(status, result_path, run_dir, identity, elapsed)


def save_rank_figure(result: CommunicationResult, path: Path) -> Path:
    """Save one light PNG for a computed result at a session-local Path."""
    from .plotting import save_rank_figure as save_plot

    return save_plot(result, path)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI strings and run one session; return process exit status 0/1.

    argv is a list of shell-provided strings or None for sys.argv. Metadata is
    a JSON Path; roles are probe IDs; bin width is milliseconds. No batch mode
    or implicit run is provided. Errors propagate after argparse validation.
    """
    parser = argparse.ArgumentParser(description="Run one communication-subspace session")
    parser.add_argument("--session-metadata", required=True, type=Path)
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--bin-width-ms", type=int, choices=(100, 500), default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args(argv)
    if bool(args.pfc_probe) != bool(args.hpc_probe):
        parser.error("Both --pfc-probe and --hpc-probe are required together")
    roles = ({"PFC": args.pfc_probe, "HPC": args.hpc_probe}
             if args.pfc_probe is not None else None)
    try:
        report = run_session(
            CommunicationConfig(args.session_metadata, roles, args.bin_width_ms, args.seed),
            rerun=args.rerun, dry_run=args.dry_run,
        )
    except (OSError, ValueError, KeyError) as error:
        print(f"Communication subspace failed: {error}", file=sys.stderr)
        return 1
    print(f"{report.status}: {report.result_path or report.configuration_id}")
    return 0 if report.status in {"prepared", "reused", "dry-run"} else 1


# Helpers keep the run-control boundary separate from numerical computation.
def _validate_session_output_root(session_root: Path) -> None:
    """Reject a resolved session Path inside the project repository; return None."""
    repository = Path(__file__).resolve().parents[3]
    if session_root.is_relative_to(repository):
        raise ValueError("Analysis output must be outside the source repository.")


def _effective_configuration(source_configuration: dict) -> dict:
    """Return source settings with a code fingerprint and final digest."""
    source = dict(source_configuration)
    source["source_configuration_id"] = source.pop("configuration_id")
    source["code_fingerprint"] = _code_fingerprint()
    source["dependency_versions"] = _dependency_versions()
    payload = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False)
    source["configuration_id"] = hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]
    return source


def _code_fingerprint() -> str:
    """Return SHA256 of feature and reused neural/behavior source files."""
    digest = hashlib.sha256()
    for path in _code_source_paths():
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _code_source_paths() -> tuple[Path, ...]:
    """Return Paths whose implementation changes alter prepared/fitted outputs."""
    repository = Path(__file__).resolve().parents[3]
    feature = Path(__file__).parent
    return tuple(feature / name for name in (
        "configuration.py", "preparation.py", "modeling.py", "scoring.py",
        "computation.py", "results.py", "run_session.py",
    )) + tuple(repository / name for name in (
        "src/neural_analysis/population/session_timeline.py",
        "src/neural_analysis/population/session_timeline_io.py",
        "src/neural_analysis/population/pca.py",
        "src/neural_analysis/spike_behavior/loading.py",
        "src/neural_analysis/session_metadata.py",
        "src/neural_analysis/task_decoding/activity.py",
        "src/behavior_analysis/project_utils.py",
    ))


def _dependency_versions() -> dict[str, str]:
    """Return installed numerical/loader package versions as scalar strings."""
    return {name: version(name) for name in (
        "numpy", "pandas", "scipy", "pynapple", "scikit-learn",
    )}


def _require_unchanged_identity(
    config: CommunicationConfig, source_id: str, effective_id: str,
) -> None:
    """Raise if source files or relevant code changed since initial inspection."""
    rechecked = inspect_session_inputs(config)
    if (rechecked.configuration["configuration_id"] != source_id
            or _effective_configuration(rechecked.configuration)["configuration_id"] != effective_id):
        raise ValueError("Source or code identity changed during fitting; result was not published.")


def _run_provenance(configuration: dict, seed: int, started_utc: str) -> dict:
    """Return JSON-compatible generator, version, parameter and code metadata."""
    repository = Path(__file__).resolve().parents[3]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository,
                              capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src/neural_analysis/communication_subspace"],
                           cwd=repository, capture_output=True, text=True, check=False)
    return {
        "generator": "run_session", "analysis_version": ANALYSIS_VERSION,
        "configuration": configuration, "seed": seed,
        "alpha_grid": ALPHA_GRID.tolist(), "alpha_tie_tolerance": 1e-12,
        "fit_objective": "||Y-XB||_F^2/N + alpha*||B||_F^2",
        "rank_policy": "min(numerical_rank(training X), variable targets)",
        "scaling": "training mean and population SD, ddof=0; counts/bin",
        "axes": {"counts": "choice,bin,unit", "fit": "choice-major then bin-major observations"},
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "git_dirty": bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def _summary(result: CommunicationResult) -> str:
    """Return a short ASCII Markdown scientific summary for one session run."""
    computation = result.computation
    config = result.meta["configuration"]
    complete = computation.curve_scores.loc[
        (computation.curve_scores.model == "rrr") &
        (computation.curve_scores["rank"] == 1),
        ["direction", "mean_r2", "n_draws"],
    ]
    findings = "\n".join(
        f"- {row.direction}: rank-1 mean R2={row.mean_r2:.4g} over {row.n_draws} complete draws"
        for row in complete.itertuples(index=False)
    )
    return (
        f"# Communication subspace: {result.session_id}\n\n"
        "Single-session execution only; no batch processing was performed.\n\n"
        "Goal: quantify held-out population prediction with reduced-rank ridge regression.\n\n"
        f"Status: {computation.status}. Eligible choices: {len(result.choice_positions)}. "
        f"Bin width: {config['bin_width_ms']} ms. Seed: {result.meta['seed']}.\n\n"
        f"## Findings\n\n{findings or 'No complete rank-1 curve was available.'}\n\n"
        "## Limitations\n\nDraws share units and are not biological replicates. "
        "Negative R2 means worse prediction than the held-out mean reference. "
        "Overlapping test windows make scores descriptive, not independent uncertainty estimates.\n\n"
        "Scripts run: run_session.py, computation.py, preparation.py, modeling.py, scoring.py.\n"
    )


def _run_log(result: CommunicationResult, elapsed_s: float) -> str:
    """Return UTF-8 log text with session, parameters, status and runtime."""
    fold_scores = result.computation.fold_scores
    undefined = int((fold_scores.status != "ok").sum())
    selected = fold_scores.loc[fold_scores.status == "ok", "alpha"].dropna()
    boundary = int(np.isin(selected.to_numpy(dtype=float), [ALPHA_GRID[0], ALPHA_GRID[-1]]).sum())
    warning_counts = fold_scores.loc[fold_scores.status != "ok", "status"].value_counts().to_dict()
    return (
        f"completed_utc={datetime.now(timezone.utc).isoformat()}\n"
        f"session_id={result.session_id}\n"
        f"configuration_id={result.meta['configuration']['configuration_id']}\n"
        f"parameters={json.dumps(result.meta['configuration'], sort_keys=True)}\n"
        f"status={result.computation.status}\n"
        f"eligible_choices={len(result.choice_positions)}\n"
        f"fit_records={len(result.computation.fit_records)}\n"
        f"undefined_scores={undefined}\n"
        f"boundary_alphas={boundary}\n"
        f"warnings={json.dumps(warning_counts, sort_keys=True)}\n"
        f"elapsed_s={elapsed_s:.6f}\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
