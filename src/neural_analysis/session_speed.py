"""Explicit session-local preparation and storage of IRIG-aligned treadmill speed."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
from time import perf_counter
from zipfile import BadZipFile
from zoneinfo import ZoneInfo

import numpy as np

from src.irig_tools import irig_serial_io
from src.mouse_behavior_preprocessing.treadmill_speed import (
    MAX_REJECTED_INTERVAL_S,
    TreadmillSpeed,
    align_treadmill_speed,
)
from src.neural_analysis.session_metadata import (
    ResolvedSession,
    load_session_metadata,
    resolve_session_metadata,
)

ANALYSIS_VERSION = "treadmill-speed-v2"
ARRAY_NAMES = ("utc_s", "speed_mm_s", "source_rows", "frame_clock_s", "frame_utc_s")


@dataclass(frozen=True)
class SpeedRunReport:
    """Small action result: status prepared/reused/dry-run, NPZ path, ID, seconds.

    result_path is None for dry-run; no scientific arrays are carried here.
    """

    status: str
    result_path: Path | None
    configuration_id: str
    elapsed_s: float


def speed_configuration(session: ResolvedSession) -> dict:
    """Return source/version identity for signed mm/s speed without reading the log.

    session supplies an optional resolved text path. Require an existing file;
    return a JSON-compatible configuration with stat identity and parameters.
    Trial origins, neural inputs, bin widths, and metadata timestamps are excluded.
    """
    source = session.behavior.treadmill_file
    if source is None:
        raise ValueError("No behavior.treadmill source declared in session metadata.")
    if not source.is_file():
        raise FileNotFoundError(f"Treadmill source is missing: {source}")
    stat = source.stat()
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": 1,
        "session_id": session.session_id,
        "source": {
            "path": str(source.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "parameters": {
            "pc_timestamp_format": "ignore",
            "irig_format": "neurokairos",
            "clock": "sync intervals plus offsets",
            "speed_units": "signed mm/s",
            "mapping": "piecewise linear; endpoint extrapolation only to actual records",
            "timing_rejection": "omit reset-offset suffix before following sync; do not reassign time",
            "maximum_rejected_interval_s": MAX_REJECTED_INTERVAL_S,
        },
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["configuration_id"] = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return config


def find_saved_speed(session_root: Path, configuration_id: str) -> Path | None:
    """Return newest completed matching NPZ under session/analysis_runs, or None.

    Read only small archive metadata; never parse raw logs, decode, or write files.
    """
    for path in sorted(
        (Path(session_root) / "analysis_runs").glob(
            f"treadmill_speed_*_{configuration_id}/results.npz"
        ),
        reverse=True,
    ):
        try:
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
                if (
                    meta.get("configuration_id") == configuration_id
                    and meta.get("analysis_version") == ANALYSIS_VERSION
                    and meta.get("schema_version") == 1
                ):
                    return path
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            continue
    return None


def save_speed_result(path: Path | str, result: TreadmillSpeed) -> Path:
    """Write named UTC/mm/s/sample/frame arrays and meta to a NEW NPZ; return path.

    Array shapes/units remain TreadmillSpeed's contract. A temporary sibling
    prevents discovery of partial results. Never overwrite an existing archive.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    arrays = {name: getattr(result, name) for name in ARRAY_NAMES}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays, meta=result.meta)
        temporary.rename(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_speed_result(path: Path | str) -> TreadmillSpeed:
    """Load a trusted local NPZ into UTC/mm/s arrays with unchanged sample order.

    Reject unsupported analysis/schema versions. NumPy pickle is used only for
    the metadata dictionary, following the existing local analysis convention.
    No raw log or other scientific source is read.
    """
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        if (
            meta.get("analysis_version") != ANALYSIS_VERSION
            or meta.get("schema_version") != 1
        ):
            raise ValueError("Unsupported treadmill speed result version.")
        return TreadmillSpeed(
            **{name: archive[name] for name in ARRAY_NAMES}, meta=meta
        )


def run_session_speed(
    metadata_path: Path | str, *, rerun: bool = False, dry_run: bool = False
) -> SpeedRunReport:
    """Explicitly prepare/reuse one session's independent UTC signed-speed result.

    metadata_path is neural_session.json with optional behavior.treadmill.
    Dry-run reads metadata/stat only. Rerun creates a new timestamped directory;
    normal runs reuse a matching archive before loading the text. Return a small
    report. Errors are input/filesystem exceptions; no neural data are touched.
    """
    from src.neural_analysis.population.run_session_timeline import _code_revision

    started = perf_counter()
    session = resolve_session_metadata(
        load_session_metadata(metadata_path), metadata_path
    )
    config = speed_configuration(session)
    identity = config["configuration_id"]
    if dry_run:
        return SpeedRunReport("dry-run", None, identity, perf_counter() - started)
    if not rerun:
        saved = find_saved_speed(session.session_root, identity)
        if saved is not None:
            return SpeedRunReport("reused", saved, identity, perf_counter() - started)
    output_root = session.session_root / "analysis_runs"
    if output_root.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError(
            "Experimental analysis outputs must be outside the source repository."
        )
    stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
    directory = output_root / f"treadmill_speed_{stamp}_{identity}"
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.Logger("treadmill_speed")
    handler = logging.FileHandler(directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    try:
        logger.info("Configuration: %s", json.dumps(config, sort_keys=True))
        (directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="ascii"
        )
        source_directory = Path(__file__).parent / "population"
        for original, copy in (
            ("run_session_timeline.py", "run_session.py"),
            ("run_session_timeline_batch.py", "run_batch.py"),
        ):
            shutil.copyfile(source_directory / original, directory / copy)
        stage = perf_counter()
        parsed = irig_serial_io.load_coolterm_serial_txt(
            session.behavior.treadmill_file, pc_timestamp_format="ignore"
        )
        parsing_s = perf_counter() - stage
        stage = perf_counter()
        result = align_treadmill_speed(parsed)
        alignment_s = perf_counter() - stage
        result = replace(
            result,
            meta={
                **result.meta,
                **config,
                "random_seed": None,
                "code_revision": _code_revision(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "metadata_path": str(session.metadata_path),
                "preparation_timings_s": {
                    "parsing": parsing_s,
                    "alignment": alignment_s,
                },
            },
        )
        _write_speed_summary(directory, result)
        stage = perf_counter()
        saved = save_speed_result(directory / "results.npz", result)
        measurements = {
            "parsing_s": parsing_s,
            "alignment_s": alignment_s,
            "writing_s": perf_counter() - stage,
            "total_s": perf_counter() - started,
            "npz_bytes": saved.stat().st_size,
        }
        (directory / "measurements.json").write_text(
            json.dumps(measurements, indent=2) + "\n", encoding="ascii"
        )
        logger.info(
            "%d speed samples, %d frame anchors; diagnostics %s; timing %s",
            len(result.utc_s),
            len(result.frame_utc_s),
            result.meta,
            measurements,
        )
    except Exception:
        logger.exception("Speed preparation failed")
        raise
    finally:
        handler.close()
        logger.removeHandler(handler)
    return SpeedRunReport("prepared", saved, identity, perf_counter() - started)


def benchmark_speed_result(path: Path | str) -> dict:
    """Measure NPZ loading, saved preparation timings, and array/process bytes.

    path identifies a completed speed NPZ. Return JSON-compatible seconds/bytes;
    peak RSS is process lifetime and filesystem cache state is unspecified.
    No files or scientific arrays are changed; browser plotting is measured separately.
    """
    started = perf_counter()
    result = load_speed_result(path)
    return {
        "npz_load_s": perf_counter() - started,
        "samples": len(result.utc_s),
        "frame_anchors": len(result.frame_utc_s),
        "npz_bytes": Path(path).stat().st_size,
        "array_bytes": sum(getattr(result, name).nbytes for name in ARRAY_NAMES),
        "preparation_timings_s": result.meta["preparation_timings_s"],
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "machine": platform.platform(),
        "measurement": "measured; filesystem cache unspecified; RSS is process lifetime",
    }


def _write_speed_summary(directory: Path, result: TreadmillSpeed) -> None:
    """Write a session-scoped report of signed values, actual UTC bounds, and diagnostics."""
    lines = [
        "# IRIG-aligned signed treadmill speed",
        "",
        f"Session: {result.meta['session_id']}",
        "",
        "Goal: inspect actual signed mm/s speed on the common session timeline.",
        f"Source: {result.meta['source']['path']}",
        "Scripts: run_session.py --speed; run_batch.py --speed (entry point copies).",
        f"Samples: {len(result.utc_s)}; decoded frames: {len(result.frame_utc_s)}.",
        f"Rejected speed records: {len(result.meta['rejected_speed_rows'])}; parsed rows {result.meta['rejected_speed_rows']}.",
        f"Rejection ledger (seconds): {result.meta['rejected_speed_records']}",
        f"UTC Unix extent: {result.utc_s[0]:.6f} to {result.utc_s[-1]:.6f} seconds.",
        f"Signed speed range: {result.speed_mm_s.min():g} to {result.speed_mm_s.max():g} mm/s.",
        f"Positive/negative/zero samples: {(result.speed_mm_s > 0).sum()}/{(result.speed_mm_s < 0).sum()}/{(result.speed_mm_s == 0).sum()}.",
        f"Clock seconds per UTC second: {result.meta['clock_seconds_per_utc_second']:.12g}.",
        f"Invalid pulse indices: {result.meta['invalid_pulse_indices']}; incomplete boundary pulses: {result.meta['incomplete_boundary_pulses']}.",
        "No statistical tests, smoothing, resampling, sign changes, or artificial speed samples.",
        "",
    ]
    (directory / "summary.md").write_text("\n".join(lines), encoding="ascii")
