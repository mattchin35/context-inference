"""Sequential session-local execution and timing for choice-window LFP products.

This module owns execution only. Numerical estimators, source loading and
typed persistence retain their independent public contracts.
"""

import argparse
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
from time import perf_counter
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from src.external_tools import readSGLX
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import (
    ResolvedSession, load_session_metadata, resolve_probe_sources, resolve_session_metadata,
)

from . import computation, preparation, results
from .results import PhaseSynchronyConfig


FINAL_FAMILIES = ("power", "coherence", "lfp-phase", "spike-ppc")


@dataclass(frozen=True)
class RunReport:
    """Small execution report without scientific arrays.

    Status is integer 0 for success/data-unavailable or 1 for requested execution
    failures. run_directory is a session-local Path, None for dry-run. products
    contains requested completed/reused output Paths, excluding prerequisites.
    Stage dictionaries identify family/selection/interval, cache/failure state,
    exclusive elapsed seconds and measured counts. timings_s contains total
    wall seconds and first requested durable-output seconds (None until ready).
    """

    status: int
    run_directory: Path | None
    products: tuple[Path, ...]
    stages: tuple[dict, ...]
    timings_s: dict


@dataclass
class _RunState:
    """One invocation's ordinary inputs/cache/measurements; no scientific arrays in reports.

    Choices retain (trial,) original IDs/rows and UTC seconds. cache maps
    categorical (family,selection,interval) keys to typed record/Path or failed
    None. charged_s sums exclusive stage seconds; source/preparation identities
    never contain run timings or display state.
    """

    session: ResolvedSession
    config: PhaseSynchronyConfig
    tasks: list[tuple[str, tuple[str, ...], str | None]]
    started: float
    source_info: dict
    headers: list
    run_directory: Path | None = None
    run_id: str = ""
    choices: pd.DataFrame | None = None
    logger: logging.Logger | None = None
    stages: list[dict] = field(default_factory=list)
    products: list[Path] = field(default_factory=list)
    cache: dict = field(default_factory=dict)
    preparation_ids: dict = field(default_factory=dict)
    timings_s: dict = field(default_factory=dict)
    charged_s: float = 0.0
    failed: bool = False
    hardware: dict = field(default_factory=dict)


class _DependencyFailed(Exception):
    """An unavailable prerequisite skips its dependent without manufacturing a product."""


def run_session(
    metadata_path: Path | str,
    config: PhaseSynchronyConfig = PhaseSynchronyConfig(),
    *,
    order: tuple[str, ...] | None = None,
    rerun: bool = False,
    dry_run: bool = False,
    prepare_only: str | None = None,
) -> RunReport:
    """Execute requested components independently and publish primary outputs first.

    metadata_path names a resolved neural_session.json; config selects explicit
    metadata sites/pairs/units, seconds/Hz and scientific settings. order sets
    family order within each primary-then-secondary sweep. prepare_only selects
    native-lfp or phase and suppresses metrics. Return a small RunReport.

    Every real invocation creates a new session-local run. Reuse compatible
    prerequisites/final products unless rerun requests new final outputs;
    rerun still reuses prerequisites. With prepare_only, rerun forces that
    selected preparation. Native-only preparation does not read behavior.
    Historical products are never overwritten.

    Dry-run reads metadata/sidecars and saved headers only: no behavior CSV,
    scientific arrays, writer lock or output directories. Native I/O currently
    reads each selected full recording, bounded by authoritative sidecar sample
    count; saved preparation supports later components without raw reload.
    Timings state this actual work without claiming optimized bounds or cold OS
    caches. Ordinary I/O/input failures skip dependents and retain independent
    outputs; unexpected programming exceptions propagate.
    """
    started = perf_counter()
    session = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    tasks = _requested_tasks(session, config, order, prepare_only)
    source_info = _inspect_selected_sources(session, tasks)
    state = _RunState(session, config, tasks, started, source_info,
                      list(results.discover_product_metadata(session.session_root)),
                      timings_s={"total_s": 0.0, "first_product_s": None})
    if dry_run:
        for family, selection, interval in tasks:
            known = _known_identity(state, family, selection, interval)
            reusable = _matching_headers(state, known)
            state.stages.append({"stage": family, "selection": selection, "interval": interval,
                                 "status": "reusable" if reusable and not rerun else "planned",
                                 "elapsed_s": 0.0, "counts": {}})
        state.timings_s["total_s"] = perf_counter() - started
        return RunReport(0, None, (), tuple(state.stages), dict(state.timings_s))
    repository = Path(__file__).resolve().parents[3]
    if session.session_root.resolve().is_relative_to(repository):
        raise ValueError("Experimental outputs must be in a session outside the source repository")
    with results.session_writer_lock(session.session_root):
        _initialize_run(state)
        try:
            try:
                if prepare_only != "native-lfp":
                    _, state.choices, _ = timeline_io.load_timeline_behavior(session)
            except (OSError, ValueError) as error:
                state.failed = True
                _record_stage(state, "input-setup", (), None, "failed", perf_counter() - started,
                              error=str(error), counts={})
                for family, selection, interval in tasks:
                    _record_stage(state, family, selection, interval, "skipped", 0.0,
                                  error="Shared behavior input unavailable", counts={})
            else:
                _record_stage(state, "input-setup", (), None, "completed", perf_counter() - started,
                              counts={"trial_count": 0 if state.choices is None else len(state.choices)})
                for family, selection, interval in tasks:
                    _ensure(state, family, selection, interval, requested=True, force=rerun)
            state.timings_s["total_s"] = perf_counter() - started
            _write_measurements(state)
            _write_summary(state)
            return RunReport(int(state.failed), state.run_directory, tuple(state.products),
                             tuple(state.stages), dict(state.timings_s))
        finally:
            for handler in list(state.logger.handlers):
                handler.close()
                state.logger.removeHandler(handler)


def _requested_tasks(session: ResolvedSession, config: PhaseSynchronyConfig,
                     order: tuple[str, ...] | None, prepare_only: str | None) -> list:
    """Return categorical family/site/unit/interval work in primary-first order; no numerical I/O."""
    sites, pairs = preparation.select_metadata_inputs(session, site_ids=config.sites, pairs=config.pairs)
    intervals = [name for name in ("primary", "secondary") if name in config.intervals]
    if prepare_only is not None:
        if prepare_only not in {"native-lfp", "phase"}:
            raise ValueError("prepare_only must be native-lfp or phase")
        if prepare_only == "native-lfp":
            return [(prepare_only, (site.site_id,), None) for site in sites]
        return [(prepare_only, (site.site_id,), interval) for interval in intervals for site in sites]
    families = tuple(name for name in FINAL_FAMILIES if name in config.components) if order is None else tuple(order)
    if len(families) != len(set(families)) or set(families) != set(config.components):
        raise ValueError("order must contain every requested family exactly once")
    tasks = []
    for interval in intervals:
        for family in families:
            selections = [(site.site_id,) for site in sites] if family == "power" else list(pairs)
            if family == "spike-ppc":
                selections = [(unit_id, config.lfp_site) for unit_id in config.unit_ids]
            tasks.extend((family, tuple(selection), interval) for selection in selections)
    return tasks


def _inspect_selected_sources(session: ResolvedSession, tasks: list) -> dict:
    """Read only selected LFP sidecars; return native sample counts/rates or independent input errors."""
    site_ids = set()
    for family, selection, _ in tasks:
        site_ids.update(selection[1:] if family == "spike-ppc" else selection)
    info = {}
    sites = {site.site_id: site for site in session.sites}
    for site_id in site_ids:
        if site_id not in sites:
            raise ValueError(f"Unknown metadata LFP site: {site_id}")
        probe = resolve_probe_sources(session, sites[site_id].probe_id)
        try:
            if probe.lfp_file is None:
                raise ValueError(f"Missing LFP path for metadata site {site_id}")
            if probe.acquisition_family == "open_ephys":
                sidecar = preparation.lfp_loading.load_open_ephys_lfp_metadata(probe.lfp_file)
                info[site_id] = {"sample_count": sidecar["num_samples"], "sample_rate_hz": sidecar["sampling_frequency_hz"]}
            else:
                sidecar = preparation.lfp_loading.load_lfp_metadata(probe.lfp_file)
                info[site_id] = {"sample_count": int(sidecar["fileSizeBytes"]) // (2 * int(sidecar["nSavedChans"])),
                                 "sample_rate_hz": float(readSGLX.SampRate(sidecar))}
        except (OSError, ValueError) as error:
            info[site_id] = {"error": str(error)}
    return info


def _known_identity(state: _RunState, family: str, selection: tuple, interval: str | None,
                    preparation_ids: tuple[str, ...] = ()) -> dict:
    """Build relevant known source/config identity; optional actual preparation references complete it."""
    sources = {}
    site_ids = [] if family == "unit-spikes" else (list(selection[1:]) if family == "spike-ppc" else list(selection))
    sites = {site.site_id: site for site in state.session.sites}
    for site_id in site_ids:
        site = sites[site_id]
        probe = resolve_probe_sources(state.session, site.probe_id)
        sources[site_id] = {"probe_id": site.probe_id, "saved_channel_index": site.saved_channel_index,
                            "acquisition_family": probe.acquisition_family,
                            "files": [_file_identity(path) for path in (probe.lfp_file, probe.lfp_metadata_file, probe.alignment_file)]}
    if family in {"unit-spikes", "spike-ppc"}:
        probe_id, _ = results.parse_unit_id(selection[0])
        try:
            probe = resolve_probe_sources(state.session, probe_id)
        except KeyError as error:
            raise ValueError(f"Unknown selected unit probe: {probe_id}") from error
        sources[selection[0]] = {"unit_channels": probe.unit_channels,
                                 "files": [_file_identity(path) for path in (probe.alignment_file, probe.spike_clusters_file,
                                                                           probe.cluster_info_file, probe.channel_quality_file)]}
    if family not in {"native-lfp", "unit-spikes"}:
        sources["behavior_trials"] = _file_identity(state.session.behavior.trial_table_file)
    return results.product_identity(family, selection, interval, state.config, source_ids=sources,
                                    preparation_ids=preparation_ids)


def _matching_headers(state: _RunState, known: dict) -> list:
    """Match known scientific inputs without guessing or calculating actual preparation-array hashes."""
    keys = ("identity_version", "analysis_version", "family", "selection", "interval", "interval_bounds_s", "parameters", "source_ids")
    return [header for header in state.headers
            if all(header.meta["identity"][key] == known[key] for key in keys)]


def _ensure(state: _RunState, family: str, selection: tuple, interval: str | None,
            *, requested: bool = False, force: bool = False) -> tuple | None:
    """Load/compute one specific product once, recording exclusive seconds and independent failure state."""
    key = (family, selection, interval)
    if key in state.cache:
        return state.cache[key]
    started, children_before = perf_counter(), state.charged_s
    try:
        known = _known_identity(state, family, selection, interval)
        if not force:
            for header in _matching_headers(state, known):
                try:
                    loaded = results.load_product(header.path, expected_identity=header.meta["identity"])
                except (OSError, ValueError, EOFError, BadZipFile):
                    continue
                state.cache[key] = (loaded.record, loaded.path)
                if requested:
                    _requested_ready(state, loaded.path)
                _record_stage(state, family, selection, interval, "reused", perf_counter() - started,
                              counts=_record_counts(loaded.record), product_path=str(loaded.path), cache_state="reused")
                return state.cache[key]
        record, references = _perform(state, family, selection, interval)
        identity = _known_identity(state, family, selection, interval, references)
        root = state.run_directory / "components" / family if requested else state.session.session_root / "processed" / "session_timeline_phase_synchrony" / family
        path = root / f"{identity['identity_id']}_{state.run_id}.npz"
        publication_started = perf_counter()
        results.save_product(path, record, identity, run_id=state.run_id, completed_utc=datetime.now(timezone.utc).isoformat())
        if requested:
            _requested_ready(state, path)
        _record_stage(state, "publication", selection, interval, "completed", perf_counter() - publication_started,
                      counts={"file_bytes": path.stat().st_size}, product_path=str(path), family=family)
        state.headers.insert(0, results.ProductInfo(path, results.read_product_metadata(path)))
        state.cache[key] = (record, path)
        exclusive = max(0.0, perf_counter() - started - (state.charged_s - children_before))
        _record_stage(state, family, selection, interval, "completed", exclusive,
                      counts=_record_counts(record), product_path=str(path), cache_state="computed")
        return state.cache[key]
    except _DependencyFailed as error:
        status = "skipped"
        error_message = str(error)
    except (OSError, ValueError) as error:
        status = "failed"
        error_message = str(error)
    state.cache[key] = None
    state.failed = True
    exclusive = max(0.0, perf_counter() - started - (state.charged_s - children_before))
    _record_stage(state, family, selection, interval, status, exclusive, error=error_message, counts={})
    return None


def _require(state: _RunState, family: str, selection: tuple, interval: str | None):
    """Return one typed prerequisite or signal a skipped dependent; no requested-output rerun propagation."""
    available = _ensure(state, family, selection, interval)
    if available is None:
        raise _DependencyFailed(f"Unavailable prerequisite {family}:{selection}:{interval}")
    return available[0]


def _perform(state: _RunState, family: str, selection: tuple, interval: str | None) -> tuple:
    """Execute explicit family-specific prerequisites and math; return typed record and actual parent IDs."""
    if family == "native-lfp":
        info = state.source_info[selection[0]]
        if "error" in info:
            raise OSError(info["error"])
        started = perf_counter()
        native = preparation.load_site_lfp(state.session, selection[0], start_sample=0, stop_sample=info["sample_count"])
        _record_stage(state, "lfp-loading", selection, None, "completed", perf_counter() - started,
                      counts={"sample_count": len(native.sample_indices)}, cache_state="raw-full-recording")
        record = preparation.preprocess_site_lfp(native, state.config)
        return record, (_preparation_id(state, record),)
    if family == "unit-spikes":
        record = preparation.load_selected_unit_spikes(state.session, selection[0])
        return record, (_preparation_id(state, record),)
    if family in {"waveforms", "phase"}:
        native = _require(state, "native-lfp", selection, None)
        record = preparation.prepare_site_lfp(native, state.choices, interval=interval) if family == "waveforms" else preparation.prepare_site_phase(native, state.choices, state.config, interval=interval)
        references = (_preparation_id(state, native),)
        return record, references + ((_preparation_id(state, record),) if family == "phase" else ())
    if family == "psd":
        waveforms = _require(state, "waveforms", selection, interval)
        return computation.compute_site_psd(waveforms, state.config), (_preparation_id(state, waveforms),)
    if family == "power":
        psd = _require(state, "psd", selection, interval)
        return computation.compute_band_power(psd, state.config), (psd.identity,)
    if family == "coherence":
        left = _require(state, "waveforms", (selection[0],), interval)
        right = _require(state, "waveforms", (selection[1],), interval)
        auto_a = _require(state, "psd", (selection[0],), interval)
        auto_b = _require(state, "psd", (selection[1],), interval)
        return computation.compute_welch_coherence(left, right, state.config, psd_a=auto_a, psd_b=auto_b), (auto_a.identity, auto_b.identity)
    if family == "lfp-phase":
        left = _require(state, "phase", (selection[0],), interval)
        right = _require(state, "phase", (selection[1],), interval)
        return computation.compute_lfp_phase_summary(left, right, state.config), (_preparation_id(state, left), _preparation_id(state, right))
    unit = _require(state, "unit-spikes", (selection[0],), None)
    phase = _require(state, "phase", (selection[1],), interval)
    return computation.compute_selected_unit_ppc(unit, phase, state.config), (_preparation_id(state, unit), _preparation_id(state, phase))


def _preparation_id(state: _RunState, record: object) -> str:
    """Cache one actual array/support identity per live preparation object, avoiding repeated hash scans."""
    key = id(record)
    if key not in state.preparation_ids:
        state.preparation_ids[key] = results.preparation_identity(record)
    return state.preparation_ids[key]


def _requested_ready(state: _RunState, path: Path) -> None:
    """Record a requested durable/reused output and its first usable elapsed wall seconds."""
    state.products.append(path)
    if state.timings_s["first_product_s"] is None:
        state.timings_s["first_product_s"] = perf_counter() - state.started


def _record_stage(state: _RunState, stage: str, selection: tuple, interval: str | None,
                  status: str, elapsed_s: float, **details) -> None:
    """Append/persist one exclusive seconds stage with categorical status and measured dimensions."""
    state.charged_s += elapsed_s
    record = {"stage": stage, "selection": selection, "interval": interval,
              "status": status, "elapsed_s": elapsed_s, **details}
    state.stages.append(record)
    state.logger.info("%s %s %s %s %.6f s %s", stage, selection, interval, status, elapsed_s, details.get("error", ""))
    state.timings_s["total_s"] = perf_counter() - state.started
    _write_measurements(state)


def _record_counts(record: object) -> dict:
    """Measure major array shapes/dtypes/bytes and actual observation/support counts.

    Array byte counts describe numerical buffers, not Python-object memory or
    per-stage peak RSS. Phase metric samples exclude the saved real guard.
    Selected physical spikes are distinct from per-frequency valid/excluded
    counts, whose sums are reported separately without pooling frequencies.
    """
    arrays = {item.name: getattr(record, item.name) for item in fields(record)
              if isinstance(getattr(record, item.name), np.ndarray)}
    major = {name: {"shape": list(array.shape), "dtype": str(array.dtype), "nbytes": array.nbytes}
             for name, array in arrays.items()}
    counts = {"array_bytes": sum(array.nbytes for array in arrays.values()), "major_arrays": major}
    if hasattr(record, "trial_ids"):
        counts["trial_count"] = len(record.trial_ids)
    if hasattr(record, "frequency_hz"):
        counts["frequency_count"] = len(record.frequency_hz)
    if hasattr(record, "sample_count"):
        counts["sample_count"] = int(np.sum(record.sample_count))
    elif hasattr(record, "sample_indices"):
        counts["sample_count"] = len(record.sample_indices)
    if hasattr(record, "segment_count"):
        counts["segment_count"] = int(np.sum(record.segment_count))
    if hasattr(record, "metric_mask"):
        counts["metric_sample_count_per_trial"] = int(np.count_nonzero(record.metric_mask))
        counts["time_sample_count_including_guard"] = len(record.relative_time_s)
    if hasattr(record, "spike_times_by_trial"):
        counts["selected_spike_count"] = sum(len(times) for times in record.spike_times_by_trial)
        counts["valid_spike_count_by_frequency"] = record.spike_count.sum(axis=0).tolist()
        counts["excluded_spike_count_by_frequency"] = record.excluded_spike_count.sum(axis=0).tolist()
        spike_bytes = sum(times.nbytes for times in record.spike_times_by_trial)
        major["spike_times_flat_s"] = {"shape": [counts["selected_spike_count"]],
                                     "dtype": "float64", "nbytes": spike_bytes}
        counts["array_bytes"] += spike_bytes
    return counts


def _initialize_run(state: _RunState) -> None:
    """Create one dated immutable run directory, source snapshots, log and initial measured metadata."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    state.run_id = stamp + "_" + uuid4().hex[:8]
    state.run_directory = state.session.session_root / "analysis_runs" / f"session_timeline_phase_synchrony_{state.run_id}"
    state.run_directory.mkdir(parents=True, exist_ok=False)
    state.logger = logging.Logger("phase_synchrony." + state.run_id)
    handler = logging.FileHandler(state.run_directory / "run.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    state.logger.addHandler(handler)
    state.logger.info("Session %s; parameters %s", state.session.session_id, asdict(state.config))
    _write_json(state.run_directory / "config.json", {"session_id": state.session.session_id,
                "session_metadata": str(state.session.metadata_path), **asdict(state.config)})
    shutil.copyfile(__file__, state.run_directory / "run_session.py")
    batch = Path(__file__).with_name("run_batch.py")
    if batch.is_file():
        shutil.copyfile(batch, state.run_directory / "run_batch.py")
    state.hardware = {"platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
                      "cpu_count": os.cpu_count() or 1,
                      "versions": {package: version(package) for package in ("numpy", "scipy", "pandas", "pynapple")}}
    _write_measurements(state)


def _write_measurements(state: _RunState) -> None:
    """Atomically persist current stages/counts/hardware and actual seconds, without scientific arrays."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    _write_json(state.run_directory / "measurements.json", {
        "run_id": state.run_id, "session_id": state.session.session_id,
        "timings_s": state.timings_s, "stages": state.stages, "hardware": state.hardware,
        "process_peak_rss_bytes": rss, "memory_note": "Process lifetime high-water RSS, not a per-stage peak.",
        "cache_note": "Analysis cache states are recorded; OS filesystem cache state is unspecified.",
    })


def _write_summary(state: _RunState) -> None:
    """Write a short run-local Markdown goal/input/outcome summary from actual execution evidence."""
    requested = ", ".join(dict.fromkeys(family for family, _, _ in state.tasks))
    first = state.timings_s["first_product_s"]
    first_text = "not available" if first is None else f"{first:.6f} s"
    text = (f"# Choice-window neural analysis\n\nSession: {state.session.session_id}\n\n"
            f"Goal: prepare requested products: {requested}.\n\n"
            f"Script: run_session.py\n\nStatus: {'requested work failed' if state.failed else 'completed'}; "
            f"requested products: {len(state.products)}.\n\n"
            f"Total elapsed: {state.timings_s['total_s']:.6f} s\n\n"
            f"First requested product: {first_text}\n\n"
            "Measurements are in measurements.json. Native reads currently cover full selected recordings. "
            "Missing trial support is unavailable data; no significance tests were performed.\n")
    (state.run_directory / "summary.md").write_text(text, encoding="ascii")


def _write_json(path: Path, value: dict) -> None:
    """Atomically replace one invocation's mutable JSON measurement/config file, using filesystem paths."""
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="ascii")
    temporary.replace(path)


def _file_identity(path: Path | None) -> dict:
    """Return cheap absolute path/byte/mtime source identity, retaining explicit missing-input status."""
    if path is None:
        return {"path": None, "missing": True}
    if not path.is_file():
        return {"path": str(path.resolve()), "missing": True}
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def argument_parser(*, batch: bool = False) -> argparse.ArgumentParser:
    """Return the shared CLI parser; batch selects a session-list path instead of metadata.

    All numerical defaults come from PhaseSynchronyConfig (seconds/Hz). Site,
    ordered pair and probe:cluster selections are categorical metadata IDs.
    This constructs controls only and reads no files or scientific arrays.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-list" if batch else "--session-metadata", type=Path, required=True)
    defaults = PhaseSynchronyConfig()
    parser.add_argument("--components", nargs="+", choices=FINAL_FAMILIES, default=defaults.components)
    parser.add_argument("--sites", nargs="+", default=())
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("SITE_A", "SITE_B"))
    parser.add_argument("--unit", action="append", default=[], help="Explicit probe:cluster ID; repeat for multiple units")
    parser.add_argument("--lfp-site")
    parser.add_argument("--intervals", nargs="+", choices=("primary", "secondary"), default=defaults.intervals)
    parser.add_argument("--order", nargs="+", choices=FINAL_FAMILIES)
    parser.add_argument("--prepare-only", choices=("native-lfp", "phase"))
    parser.add_argument("--rerun", action="store_true", help="Recompute requested outputs, reusing prerequisites; --prepare-only forces that preparation")
    parser.add_argument("--dry-run", action="store_true")
    if batch:
        parser.add_argument("--workers", type=int, help="Session workers; defaults to CPU count capped by sessions")
    return parser


def config_from_arguments(arguments: argparse.Namespace) -> PhaseSynchronyConfig:
    """Convert parsed categorical selections to validated immutable scientific settings.

    Return PhaseSynchronyConfig with unchanged accepted Hz/second/threshold
    defaults; reject repeated selections and an incomplete/repeated work order
    before any session work. No files or arrays are read.
    """
    config = PhaseSynchronyConfig(
        components=tuple(arguments.components), sites=tuple(arguments.sites),
        pairs=tuple(tuple(pair) for pair in arguments.pair), unit_ids=tuple(arguments.unit),
        lfp_site=arguments.lfp_site, intervals=tuple(arguments.intervals),
    )
    for name in ("components", "sites", "pairs", "unit_ids", "intervals"):
        values = getattr(config, name)
        if len(values) != len(set(values)):
            raise ValueError(f"Repeated {name} selection")
    if arguments.order is not None:
        if len(arguments.order) != len(set(arguments.order)) or set(arguments.order) != set(config.components):
            raise ValueError("order must contain every requested family exactly once")
    return config


def main(argv: list[str] | None = None) -> int:
    """Execute one CLI request and print array-free JSON evidence; return integer exit status.

    argv is a list of CLI strings, or None for process arguments. Ordinary
    input/filesystem failures print to stderr and return 1; completed stages
    and paths are printed to stdout, including metadata-only dry-run plans.
    """
    arguments = argument_parser().parse_args(argv)
    try:
        config = config_from_arguments(arguments)
        report = run_session(
            arguments.session_metadata, config,
            order=None if arguments.order is None else tuple(arguments.order),
            rerun=arguments.rerun, dry_run=arguments.dry_run, prepare_only=arguments.prepare_only,
        )
    except (OSError, ValueError, results.WriterLockError) as error:
        print(f"Phase synchrony run failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(report), default=str))
    return report.status


if __name__ == "__main__":
    raise SystemExit(main())
