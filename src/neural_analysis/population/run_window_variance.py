"""Prepare/reuse Stage 2 sidecars using saved Stage 1 scores only."""

from __future__ import annotations

from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
from zipfile import BadZipFile
from zoneinfo import ZoneInfo

import numpy as np

from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population import window_variance_io as storage
from src.neural_analysis.population.run_session_timeline import (
    TimelineRunReport,
    _code_revision,
    find_saved_timeline,
    resolve_timeline_probe_roles,
)
from src.neural_analysis.population.window_variance import (
    compute_session_window_variance,
)
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)


def run_window_variance(
    stage1_path: Path | str,
    *,
    rerun: bool = False,
    dry_run: bool = False,
) -> TimelineRunReport:
    """Prepare a new timestamped sidecar or reuse an identical Stage 2 result.

    stage1_path is an existing timeline NPZ. Dry-run reads only its file stat.
    Return a small report; outputs live inside that exact Stage 1 run directory.
    Neither spikes nor PCA fitting are accessed. Existing files are preserved.
    """
    started = perf_counter()
    config = storage.window_variance_configuration(stage1_path)
    identity = config["configuration_id"]
    if dry_run:
        return TimelineRunReport("dry-run", None, identity, perf_counter() - started)
    if not rerun:
        saved = find_saved_window_variance(stage1_path)
        if saved is not None:
            return TimelineRunReport(
                "reused", saved, identity, perf_counter() - started
            )
    source = Path(config["source_stage1"]["path"])
    if source.is_relative_to(Path(__file__).resolve().parents[3]):
        raise ValueError(
            "Experimental analysis outputs must be outside the source repository."
        )
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = source.parent / f"window_variance_{stamp}_{identity}"
    directory.mkdir(exist_ok=False)
    logger = logging.Logger("window_variance")
    handler = logging.FileHandler(directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Parameters/source: %s", json.dumps(config, sort_keys=True))
        start = perf_counter()
        stage1 = timeline_io.load_timeline_result(source)
        config.update(
            source_configuration_id=stage1.meta["configuration_id"],
            bin_width_s=stage1.meta["bin_width_s"],
            code_revision=_code_revision(),
            python=platform.python_version(),
            numpy=np.__version__,
            stage1_load_s=perf_counter() - start,
        )
        start = perf_counter()
        summary = compute_session_window_variance(stage1, config)
        summary.meta["calculation_s"] = perf_counter() - start
        for role, region in summary.regions.items():
            statuses, counts = np.unique(region.status, return_counts=True)
            logger.info(
                "%s/%s: %d units, %d bins, full basis %s, statuses %s",
                role,
                region.probe_id,
                len(region.unit_ids),
                len(region.bin_edges_s) - 1,
                region.basis_complete,
                dict(zip(statuses.tolist(), counts.tolist())),
            )
        summary.meta["preparation_elapsed_s"] = perf_counter() - started
        (directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="ascii"
        )
        shutil.copyfile(__file__, directory / "run_session.py")
        shutil.copyfile(
            Path(__file__).with_name("run_session_timeline_batch.py"),
            directory / "run_batch.py",
        )
        lines = [
            "# Fixed-axis choice-window participation",
            "",
            f"Session: {summary.session_id}",
            "",
            "Goal: smallest count of fixed session axes reaching 95% of each choice window's direct standardized-unit sample variance.",
            f"Source: {source}",
            "",
            "Scripts: run_session.py (Stage 2 runner); run_batch.py (--window-summary batch entry point).",
            "Counts are not local PCA dimensionality; there is no bin-count rank cap. No statistical tests were performed.",
            "",
        ]
        for role, region in summary.regions.items():
            statuses, counts = np.unique(region.status, return_counts=True)
            lines.append(
                f"- {role}: {len(region.choice_positions)} windows; {dict(zip(statuses.tolist(), counts.tolist()))}; complete basis: {region.basis_complete}."
            )
        lines.extend(
            [
                "",
                f"Parameters: {summary.meta['parameters']}",
                "",
                f"Calculation: {summary.meta['calculation_s']:.6f} seconds.",
                "",
            ]
        )
        (directory / "summary.md").write_text("\n".join(lines), encoding="ascii")
        saved = storage.save_window_variance(directory / "results.npz", summary)
        logger.info(
            "Prepared %s; elapsed %.3f seconds; bytes %d",
            saved,
            perf_counter() - started,
            saved.stat().st_size,
        )
    except Exception:
        logger.exception("Stage 2 preparation failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return TimelineRunReport("prepared", saved, identity, perf_counter() - started)


def find_saved_window_variance(stage1_path: Path | str) -> Path | None:
    """Return newest completed summary matching the exact Stage 1 file/version.

    Only small metadata is read; interrupted and unsupported sidecars are skipped.
    """
    config = storage.window_variance_configuration(stage1_path)
    identity = config["configuration_id"]
    root = Path(config["source_stage1"]["path"]).parent
    for path in sorted(
        root.glob(f"window_variance_*_{identity}/results.npz"), reverse=True
    ):
        try:
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
                if (
                    meta.get("configuration_id") == identity
                    and meta.get("analysis_version") == storage.ANALYSIS_VERSION
                    and meta.get("schema_version") == 1
                ):
                    return path
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            continue
    return None


def run_session_window_variance(
    metadata_path: Path | str,
    probe_roles: dict[str, str] | None = None,
    bin_width_s: float = 0.5,
    *,
    rerun: bool = False,
    dry_run: bool = False,
) -> TimelineRunReport:
    """Resolve metadata/roles/bin seconds to an EXISTING Stage 1 archive, then reduce.

    Signature matches the Stage 1 orchestration boundary for main/batch reuse.
    Raise ValueError if Stage 1 has not been prepared; never fit it implicitly.
    """
    session = resolve_session_metadata(
        load_session_metadata(metadata_path), metadata_path
    )
    roles = resolve_timeline_probe_roles(session, probe_roles)
    config = timeline_io.timeline_configuration(session, roles, bin_width_s)
    source = find_saved_timeline(session.session_root, config["configuration_id"])
    if source is None:
        raise ValueError(
            "Prepare this Stage 1 configuration before its window summary."
        )
    return run_window_variance(source, rerun=rerun, dry_run=dry_run)


def benchmark_window_variance(
    stage1_path: Path | str,
    summary_path: Path | str,
    *,
    loaded_range_s: tuple[float, float] | None = None,
) -> dict:
    """Measure saved loads, fresh reduction, figure/JSON costs; no files are written.

    Paths are a matching Stage 1/2 pair. Optional loaded_range_s bounds heatmap
    payload in elapsed seconds (as in the viewer); None sends the full session.
    Return JSON-compatible timings in seconds and sizes in bytes. Peak RSS is
    process lifetime, not incremental Stage 2 memory; cache state is unspecified.
    """
    from src.neural_analysis.population.timeline_plotting import plot_session_timeline

    started = perf_counter()
    stage1 = timeline_io.load_timeline_result(stage1_path)
    loading = perf_counter() - started
    started = perf_counter()
    summary = storage.load_window_variance(summary_path)
    summary_loading = perf_counter() - started
    if (
        summary.meta["configuration_id"]
        != storage.window_variance_configuration(stage1_path)["configuration_id"]
    ):
        raise ValueError("Summary does not match the exact Stage 1 archive.")
    started = perf_counter()
    compute_session_window_variance(stage1, {})
    calculation = perf_counter() - started
    started = perf_counter()
    figure = plot_session_timeline(
        stage1, loaded_range_s=loaded_range_s, window_variance=summary.regions
    )
    construction = perf_counter() - started
    started = perf_counter()
    payload = figure.to_json()
    return {
        "stage1_load_s": loading,
        "summary_load_s": summary_loading,
        "stage2_calculation_s": calculation,
        "figure_construction_s": construction,
        "serialization_s": perf_counter() - started,
        "json_bytes": len(payload.encode()),
        "loaded_range_s": loaded_range_s,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "measurement": "measured; filesystem cache unspecified; RSS covers process lifetime",
        "populations": {
            role: {
                "windows": len(region.choice_positions),
                "bins": len(region.bin_edges_s) - 1,
                "units": len(region.unit_ids),
                "pcs": region.pc_ids.shape[1],
                "basis_complete": region.basis_complete,
                "stage1_timings_s": stage1.regions[role].timings_s,
                "score_bytes": stage1.regions[role].scores.nbytes,
            }
            for role, region in summary.regions.items()
        },
    }
