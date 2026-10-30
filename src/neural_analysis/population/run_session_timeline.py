"""Prepare/reuse one session's fixed timeline PCA; no viewer or Stage 2 work."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
from time import perf_counter
from zipfile import BadZipFile
from zoneinfo import ZoneInfo

import numpy as np

from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.session_timeline import (
    SessionTimelineResult,
    compute_timeline_pca,
    select_complete_choice_windows,
    summarize_session_variance,
)
from src.neural_analysis.session_metadata import (
    ResolvedSession,
    load_session_metadata,
    resolve_session_metadata,
)


@dataclass(frozen=True)
class TimelineRunReport:
    """Small execution result: status, saved NPZ path, scientific configuration ID.

    status is prepared/reused/dry-run. result_path is None for dry-run; elapsed_s
    is wall time for the action. The record contains no large scientific arrays.
    """

    status: str
    result_path: Path | None
    configuration_id: str
    elapsed_s: float


def run_session_timeline(
    metadata_path: Path | str,
    probe_roles: dict[str, str] | None = None,
    bin_width_s: float = 0.5,
    *,
    rerun: bool = False,
    dry_run: bool = False,
) -> TimelineRunReport:
    """Prepare a session-local result or reuse an identical saved configuration.

    metadata_path is an existing neural_session.json. probe_roles maps PFC/HPC
    to probe keys; None uses existing version-3 role metadata. bin_width_s is
    .5/.1/.05/.02 seconds. Run flags apply only at this orchestration boundary.
    Return a small report; outputs are written under session/analysis_runs.
    Dry-run reads only metadata and small selection files and writes nothing.
    """
    started = perf_counter()
    session = resolve_session_metadata(
        load_session_metadata(metadata_path), metadata_path
    )
    roles = resolve_timeline_probe_roles(session, probe_roles)
    config = timeline_io.timeline_configuration(session, roles, bin_width_s)
    identity = config["configuration_id"]
    if dry_run:
        return TimelineRunReport("dry-run", None, identity, perf_counter() - started)
    if not rerun:
        saved = find_saved_timeline(session.session_root, identity)
        if saved is not None:
            return TimelineRunReport(
                "reused", saved, identity, perf_counter() - started
            )
    output_root = session.session_root / "analysis_runs"
    repository = Path(__file__).resolve().parents[3]
    if output_root.resolve().is_relative_to(repository):
        raise ValueError(
            "Experimental analysis outputs must be outside the source repository."
        )
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    run_directory = output_root / f"session_timeline_pca_{stamp}_{identity}"
    run_directory.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("session_timeline")
    handler = logging.FileHandler(run_directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info(
            "Session %s; configuration %s",
            session.session_id,
            json.dumps(config, sort_keys=True),
        )
        result = _prepare_session(session, roles, config, logger)
        result.meta["preparation_elapsed_s"] = perf_counter() - started
        result.meta["peak_process_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
        (run_directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="ascii"
        )
        shutil.copyfile(__file__, run_directory / "run_session.py")
        shutil.copyfile(
            Path(__file__).with_name("run_session_timeline_batch.py"),
            run_directory / "run_batch.py",
        )
        _write_summary(run_directory, result)
        saved = timeline_io.save_timeline_result(run_directory / "results.npz", result)
        logger.info(
            "Prepared %s; elapsed %.3f s; NPZ bytes %d; process peak RSS bytes %d",
            saved,
            perf_counter() - started,
            saved.stat().st_size,
            result.meta["peak_process_rss_bytes"],
        )
    except Exception:
        logger.exception("Preparation failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return TimelineRunReport("prepared", saved, identity, perf_counter() - started)


def resolve_timeline_probe_roles(
    session: ResolvedSession,
    probe_roles: dict[str, str] | None,
) -> dict[str, str]:
    """Return explicit role/probe keys, or existing metadata defaults, without guessing.

    An empty explicit dictionary selects behavior only. None with version-2
    metadata returns no assignments, requiring explicit CLI/UI choices for PCs.
    """
    if probe_roles is not None:
        return dict(probe_roles)
    if session.analysis is None:
        return {}
    roles = session.analysis.probe_roles
    return {"PFC": roles.pfc, "HPC": roles.hpc}


def find_saved_timeline(session_root: Path, configuration_id: str) -> Path | None:
    """Return the newest completed matching session-local NPZ, or None.

    Reads only archive metadata, not raw spikes or score arrays. Partial and
    unsupported files are skipped; configuration_id describes scientific inputs.
    """
    root = Path(session_root) / "analysis_runs"
    for path in sorted(
        root.glob(f"session_timeline_pca_*_{configuration_id}/results.npz"),
        reverse=True,
    ):
        try:
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
                if (
                    meta.get("configuration_id") == configuration_id
                    and meta.get("schema_version") == 1
                ):
                    return path
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            continue
    return None


def benchmark_timeline_result(result_path: Path | str) -> dict:
    """Measure NPZ load, default figure construction, serialization, and process memory.

    result_path is a completed timeline NPZ. Return JSON-compatible measured
    timings in seconds, sizes in bytes, and recorded source-preparation timings.
    This does not refit, change the result, measure the browser, or claim cold I/O.
    """
    from src.neural_analysis.population.timeline_plotting import plot_session_timeline

    started = perf_counter()
    result = timeline_io.load_timeline_result(result_path)
    loading_s = perf_counter() - started
    started = perf_counter()
    figure = plot_session_timeline(result)
    construction_s = perf_counter() - started
    started = perf_counter()
    payload = figure.to_json()
    serialization_s = perf_counter() - started
    populations = {}
    for role, regional in result.regions.items():
        windows, bins, pcs = regional.scores.shape
        populations[role] = {
            "windows": windows,
            "bins": bins,
            "pcs": pcs,
            "units": len(regional.transform.retained_unit_ids),
            "visible_pcs": min(10, pcs),
            "score_bytes": regional.scores.nbytes,
            "loadings_bytes": regional.transform.components.nbytes,
            "preparation_timings_s": regional.timings_s,
        }
    try:
        available_memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        available_memory = None
    return {
        "measurement": "measured; filesystem cache state unspecified",
        "result_path": str(result_path),
        "session_id": result.session_id,
        "bin_width_s": result.meta["bin_width_s"],
        "choices": len(result.choices),
        "npz_load_s": loading_s,
        "figure_construction_s": construction_s,
        "serialization_s": serialization_s,
        "json_bytes": len(payload.encode("utf-8")),
        "populations": populations,
        "machine": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "available_memory_bytes": available_memory,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "memory_note": "Peak RSS covers this process lifetime, including imports and prior work.",
        "browser_responsiveness": "requires inspection in the webapp",
    }


def main(argv: list[str] | None = None) -> int:
    """Execute CLI arguments and return zero on success, one on ordinary input failures.

    argv is a string argument list or None for process arguments. Bin width is
    entered in milliseconds and converted once to seconds at this boundary.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metadata", type=Path, required=True)
    parser.add_argument(
        "--bin-width-ms", type=int, choices=(500, 100, 50, 20), default=500
    )
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure saved-result loading and figure payload after preparation/reuse.",
    )
    args = parser.parse_args(argv)
    roles = probe_roles_from_arguments(args.pfc_probe, args.hpc_probe)
    try:
        report = run_session_timeline(
            args.session_metadata,
            roles,
            args.bin_width_ms / 1000.0,
            rerun=args.rerun,
            dry_run=args.dry_run,
        )
    except (OSError, ValueError) as error:
        print(f"Timeline preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"{report.status}: {report.result_path or report.configuration_id} ({report.elapsed_s:.3f} s)"
    )
    if args.benchmark and report.result_path is not None:
        measured = benchmark_timeline_result(report.result_path)
        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
        path = report.result_path.parent / f"benchmark_{stamp}.json"
        path.write_text(json.dumps(measured, indent=2) + "\n", encoding="ascii")
        print(f"Benchmark: {path}")
    return 0


# Orchestration helpers: scientific operations remain in session_timeline.py.
def _prepare_session(
    session: ResolvedSession,
    roles: dict[str, str],
    config: dict,
    logger: logging.Logger,
) -> SessionTimelineResult:
    """Assemble independently available populations and behavior; return fixed scores.

    config is the scientific/source identity; logger receives elapsed seconds,
    unit/window counts, parameters, and regional input failures.
    """
    start = perf_counter()
    origin, choices, events = timeline_io.load_timeline_behavior(session)
    meta = dict(config)
    meta.update(
        {
            "behavior_loading_s": perf_counter() - start,
            "code_revision": _code_revision(),
            "python": platform.python_version(),
            "library_versions": {
                name: version(name)
                for name in ("numpy", "pandas", "pynapple", "scikit-learn", "plotly")
            },
            "machine": platform.platform(),
            "cpu_count": os.cpu_count(),
        }
    )
    regions = {}
    unavailable = {}
    for role in ("PFC", "HPC"):
        if role not in roles:
            unavailable[role] = "No probe assigned"
            continue
        try:
            start = perf_counter()
            population = timeline_io.load_timeline_population(session, roles[role])
            loading_s = perf_counter() - start
            positions = select_complete_choice_windows(
                choices.choice_time.to_numpy(), population.coverage_s
            )
            if positions.size == 0:
                raise ValueError(
                    "No completed choices have full neural recording coverage"
                )
            regional = compute_timeline_pca(
                population.spike_group,
                population.cluster_ids,
                choices,
                positions,
                role,
                roles[role],
                config["bin_width_s"],
            )
            regional.timings_s["loading"] = loading_s
            regions[role] = regional
            logger.info(
                "%s/%s: %d/%d windows, %d retained units, %d omitted constants; timing seconds %s",
                role,
                roles[role],
                len(positions),
                len(choices),
                len(regional.transform.retained_unit_ids),
                len(regional.transform.omitted_unit_ids),
                regional.timings_s,
            )
        except (OSError, ValueError) as error:
            unavailable[role] = str(error)
            logger.warning("%s/%s unavailable: %s", role, roles[role], error)
    return SessionTimelineResult(
        session.session_id, origin, choices, events, regions, unavailable, meta
    )


def _write_summary(directory: Path, result: SessionTimelineResult) -> None:
    """Write a short session-scoped Markdown report of preparation, units, and coverage."""
    lines = [
        "# Session timeline PCA",
        "",
        f"Session: {result.session_id}",
        "",
        "Goal: inspect completed choices, events, and fixed regional PC scores on elapsed session time.",
        "",
        "Scripts: run_session.py (single-session preparation); run_batch.py (batch entry point copy).",
        "",
        f"Bin width: {result.meta['bin_width_s'] * 1000:g} ms; windows: [-2, +2) s.",
        f"Completed ordinary choices: {len(result.choices)}; origin: first trial start, UTC {result.time_origin_s} s.",
        "",
        "Variance coverage uses all eligible choice windows and total standardized-unit variance.",
        "",
    ]
    for role, regional in result.regions.items():
        coverage = summarize_session_variance(
            regional.transform.explained_variance, regional.total_session_variance
        )
        lines.append(
            f"- {role} ({regional.probe_id}): {len(regional.choice_positions)} windows, "
            f"{len(regional.transform.retained_unit_ids)} units; "
            f"80/90/95% leading-PC counts: {coverage[0.8]}/{coverage[0.9]}/{coverage[0.95]}."
        )
    for role, reason in result.unavailable.items():
        lines.append(f"- {role}: unavailable ({reason}).")
    lines.extend(
        [
            "",
            "No statistical tests or within-window participation counts were computed.",
            "PCs describe pooled choice-window activity; signs are arbitrary across independently fitted configurations.",
            "",
        ]
    )
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _code_revision() -> str:
    """Return the source Git revision string, or unknown outside a Git checkout."""
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return process.stdout.strip() if process.returncode == 0 else "unknown"


def probe_roles_from_arguments(
    pfc: str | None, hpc: str | None
) -> dict[str, str] | None:
    """Convert optional CLI probe keys to explicit roles, or None for metadata defaults."""
    if pfc is None and hpc is None:
        return None
    return {
        role: probe for role, probe in (("PFC", pfc), ("HPC", hpc)) if probe is not None
    }


if __name__ == "__main__":
    raise SystemExit(main())
