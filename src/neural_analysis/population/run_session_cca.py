"""Explicit single-session CCA preparation and saved-product numerical benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
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
from time import gmtime, perf_counter
from zoneinfo import ZoneInfo

from src.neural_analysis.population import cca
from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.population.cca_plotting import plot_session_cca
from src.neural_analysis.population.run_session_timeline import (
    probe_roles_from_arguments,
    resolve_timeline_probe_roles,
)
from src.neural_analysis.session_metadata import ResolvedSession, load_session_metadata, resolve_session_metadata


@dataclass(frozen=True)
class CCARunReport:
    """Small action report without numerical arrays.

    status is prepared/reused/dry-run, or failed for a batch error. result_path
    is a completed NPZ Path, or None; configuration_id is a unitless scientific
    identity (empty if failure prevented configuration). elapsed_s is wall time
    in seconds. error is None on success or a batch failure description.
    """

    status: str
    result_path: Path | None
    configuration_id: str
    elapsed_s: float
    error: str | None = None


def run_session_cca(
    metadata_path: Path | str,
    probe_roles: dict[str, str] | None = None,
    bin_width_s: float = 0.5,
    *,
    rerun: bool = False,
    dry_run: bool = False,
    population_quality: str = "good_mua",
    missing_label_policy: str = "mua",
) -> CCARunReport:
    """Prepare/reuse one session's paired CCA, or inspect inputs without writing.

    metadata_path names neural_session.json. Explicit roles map PFC/HPC to
    probes; None uses existing version-3 defaults. bin_width_s is .5/.1/.05/.02
    seconds. Quality strings use the accepted IO policy. Run flags act only at
    this boundary. Return a small CCARunReport; saved scores retain (W,B,K)
    canonical-SD axes. Source/input/numerical errors raise without publication.
    Dry-run reads small metadata/IRIG only and writes nothing. Rerun always
    creates a new session-local directory; no previous result is overwritten.
    """
    began = perf_counter()
    session = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    roles = resolve_timeline_probe_roles(session, probe_roles)
    policies = {"population_quality": population_quality, "missing_label_policy": missing_label_policy}
    configuration = cca_io.cca_configuration(session, roles, bin_width_s, **policies)
    identity = configuration["configuration_id"]
    if dry_run:
        return CCARunReport("dry-run", None, identity, perf_counter() - began)
    if not rerun:
        compatible = cca_io.load_latest_cca_result(session.session_root, identity)
        if compatible is not None:
            return CCARunReport("reused", compatible[0].path, identity, perf_counter() - began)

    output_root = session.session_root / "analysis_runs"
    if output_root.resolve().is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError("CCA analysis outputs must be outside the source repository.")
    started = datetime.now(timezone.utc)
    stamp = started.astimezone(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = output_root / f"session_timeline_cca_{stamp}_{identity}"
    directory.mkdir(parents=True, exist_ok=False)
    logger, handler = _run_logger(directory)
    try:
        logger.info("Session %s; configuration %s", session.session_id, json.dumps(configuration, sort_keys=True))
        result = _prepare_result(session, roles, bin_width_s, configuration, policies, started)
        _write_run_documents(directory, result, configuration)
        verify_began = perf_counter()
        current = cca_io.cca_configuration(session, roles, bin_width_s, **policies)
        if current["configuration_id"] != identity:
            raise ValueError("CCA sources changed during preparation; no result was published.")
        result.meta["preparation_timings_s"]["source_verification"] = perf_counter() - verify_began
        publish_began = perf_counter()
        saved = cca_io.save_cca_result(directory / "results.npz", result)
        measurements = {
            "preparation_timings_s": result.meta["preparation_timings_s"],
            "compression_and_publication_s": perf_counter() - publish_began,
            "wall_elapsed_s": perf_counter() - began,
            "peak_process_rss_bytes": _peak_rss_bytes(),
            "npz_bytes": saved.stat().st_size,
        }
        (directory / "measurements.json").write_text(json.dumps(measurements, indent=2) + "\n", encoding="ascii")
        logger.info("Prepared %s; timings %s", saved, json.dumps(measurements, sort_keys=True))
    except Exception:
        logger.exception("CCA preparation failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return CCARunReport("prepared", saved, identity, perf_counter() - began)


def benchmark_cca_result(
    result_path: Path | str, *, loaded_range_s: tuple[float, float] | None = None,
) -> dict:
    """Measure saved loading, actual figure construction/JSON and process memory.

    result_path names a completed CCA NPZ. loaded_range_s optionally bounds
    neural display windows in elapsed seconds; full scores still load. Return
    a JSON-compatible dictionary: seconds for measured load/preparation/figure
    and serialization times, UTF-8 bytes for JSON, bytes for file/memory, and
    dimensionless window/unit/rank/pair counts. Filesystem cache state is
    unspecified; RSS is process-lifetime peak, not incremental allocation.
    Browser rendering is not measured. No fitting or file modification occurs.
    """
    path = Path(result_path)
    began = perf_counter()
    result = cca_io.load_cca_result(path)
    loading_s = perf_counter() - began
    began = perf_counter()
    figure = plot_session_cca(result, loaded_range_s=loaded_range_s)
    construction_s = perf_counter() - began
    began = perf_counter()
    serialized = figure.to_json().encode("utf-8")
    serialization_s = perf_counter() - began
    return {
        "measurement": "measured; filesystem cache state unspecified",
        "result_path": str(path), "session_id": result.session_id,
        "bin_width_s": result.meta["bin_width_s"], "choices": len(result.choices),
        "paired_windows": len(result.choice_positions), "pairs": len(result.fit.correlations),
        "npz_load_s": loading_s, "npz_bytes": path.stat().st_size,
        "figure_construction_s": construction_s, "serialization_s": serialization_s,
        "json_bytes": len(serialized),
        "loaded_range_s": list(loaded_range_s) if loaded_range_s is not None else None,
        "browser_rendering": "not measured",
        "score_bytes": sum(values.nbytes for values in result.fit.scores.values()),
        "preparation_timings_s": result.meta.get("preparation_timings_s", {}),
        "regions": {region: {"units": len(transform.retained_unit_ids), "rank": transform.rank,
                             "score_shape": list(result.fit.scores[region].shape)}
                    for region, transform in result.fit.transforms.items()},
        "peak_process_rss_bytes": _peak_rss_bytes(),
        "available_memory_bytes": _available_memory_bytes(),
        "memory_note": "Process-lifetime peak RSS includes imports and prior operations.",
        "machine": platform.platform(), "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def main(argv: list[str] | None = None) -> int:
    """Parse one CLI action and return zero on success, one for ordinary action errors.

    argv is a list of CLI strings or None. Bin widths enter in milliseconds
    and convert once to seconds. --benchmark writes only measured saved-product
    metrics beside a prepared/reused result, never modifies its scientific NPZ.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-metadata", type=Path, required=True)
    parser.add_argument("--pfc-probe")
    parser.add_argument("--hpc-probe")
    parser.add_argument("--bin-width-ms", type=int, choices=(500, 100, 50, 20), default=500)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_session_cca(
            args.session_metadata, probe_roles_from_arguments(args.pfc_probe, args.hpc_probe),
            args.bin_width_ms / 1000, rerun=args.rerun, dry_run=args.dry_run,
        )
        if args.benchmark and report.result_path is not None:
            measured = benchmark_cca_result(report.result_path)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            target = report.result_path.parent / f"benchmark_{stamp}.json"
            target.write_text(json.dumps(measured, indent=2) + "\n", encoding="ascii")
        print(f"{report.status}: {report.result_path or report.configuration_id}")
    except (OSError, ValueError) as error:
        print(f"CCA preparation failed: {error}", file=sys.stderr)
        return 1
    return 0


# Orchestration helpers; all mathematical fitting remains in population.cca.
def _prepare_result(session: ResolvedSession, roles: dict[str, str], bin_width_s: float, configuration: dict,
                    policies: dict[str, str], started: datetime) -> cca_io.SessionCCAResult:
    """Load/fit one resolved session and return its unpersisted paired scientific record.

    session is ResolvedSession; roles/policies/configuration use the public
    runner contracts, started is aware UTC. Preserve count/bin/unit axes and
    add provenance/timings in seconds; raw counts are not retained in the result.
    """
    began = perf_counter()
    inputs = cca_io.load_cca_inputs(session, roles, bin_width_s, **policies)
    loading_s = perf_counter() - began
    if inputs.configuration["configuration_id"] != configuration["configuration_id"]:
        raise ValueError("CCA sources changed before paired input loading completed.")
    began = perf_counter()
    fit = cca.fit_paired_cca(inputs.counts["PFC"], inputs.counts["HPC"],
                             inputs.unit_ids["PFC"], inputs.unit_ids["HPC"])
    fitting_s = perf_counter() - began
    metadata_stat = session.metadata_path.stat()
    meta = dict(configuration)
    meta.update({
        "started_utc": started.isoformat().replace("+00:00", "Z"),
        "code_revision": _code_revision(), "python": platform.python_version(),
        "library_versions": {name: version(name) for name in ("numpy", "pandas", "pynapple", "scipy", "statsmodels")},
        "metadata_source": {"path": str(session.metadata_path), "size": metadata_stat.st_size,
                            "mtime_ns": metadata_stat.st_mtime_ns},
        "preparation_timings_s": {"input_loading": loading_s, "fit_total": fitting_s},
    })
    return cca_io.SessionCCAResult(
        inputs.session_id, inputs.time_origin_s, inputs.choices, inputs.events,
        inputs.choice_positions, inputs.coverage_positions, inputs.bin_edges_s,
        inputs.probe_roles, fit, None, meta,
    )


def _run_logger(directory: Path) -> tuple[logging.Logger, logging.FileHandler]:
    """Return a run-local logger/handler with explicitly UTC timestamps; no global setup.

    directory is the new run path. Log timestamps are ISO-like UTC seconds;
    caller closes/removes the handler after its action. No signal arrays involved.
    """
    logger = logging.Logger(f"session_cca.{directory.name}")
    handler = logging.FileHandler(directory / "run.log")
    formatter = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S")
    formatter.converter = gmtime
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger, handler


def _write_run_documents(directory: Path, result: cca_io.SessionCCAResult, configuration: dict) -> None:
    """Write config, main/batch script copies and a concise scientific summary; return None.

    directory is a new run folder; result contains canonical-SD scores and UTC
    origin, configuration is JSON-compatible scientific settings. No NPZ changes.
    """
    (directory / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="ascii")
    shutil.copyfile(__file__, directory / "run_session.py")
    shutil.copyfile(Path(__file__).with_name("run_session_cca_batch.py"), directory / "run_batch.py")
    lines = ["# Session timeline PFC-HPC CCA", "", f"Session: {result.session_id}", "",
             "Goal: inspect paired population activity in fixed session-wide canonical coordinates.",
             "Scripts: run_session.py and run_batch.py (copies of the executed project entry points).", "",
             f"All choices: {len(result.choices)}; paired windows: {len(result.choice_positions)}; "
             f"bins/window: {len(result.bin_edges_s) - 1}; bin width: {configuration['bin_width_s']} s.",
             "Scores use canonical session SD; correlations are in-sample and task activity is retained.",
             "No held-out assessment or significance tests were performed.", ""]
    for region, transform in result.fit.transforms.items():
        lines.append(f"- {region} ({result.probe_roles[region]}): {len(transform.input_unit_ids)} input units, "
                     f"{len(transform.retained_unit_ids)} nonconstant units, numerical rank {transform.rank}.")
    lines.extend(["", f"Retained canonical pairs: {len(result.fit.correlations)}.", ""])
    (directory / "summary.md").write_text("\n".join(lines), encoding="ascii")


def _peak_rss_bytes() -> int:
    """Return process-lifetime peak resident memory in bytes; no inputs or signal axes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak * (1 if sys.platform == "darwin" else 1024))


def _available_memory_bytes() -> int | None:
    """Return system-reported available memory in bytes, or None when unsupported."""
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def _code_revision() -> str:
    """Return current project Git HEAD, or unknown outside a checkout; no mutation."""
    try:
        process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                 capture_output=True, text=True, check=False)
    except OSError:
        return "unknown"
    return process.stdout.strip() if process.returncode == 0 else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
