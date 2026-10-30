"""Existing metadata/source loading and session-local timeline NPZ persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
import pynapple as nap

from src.neural_analysis.interregional.records import RegionalPCATransform
from src.neural_analysis.population.session_timeline import (
    ANALYSIS_VERSION,
    RegionalTimelinePCA,
    SessionTimelineResult,
    choice_bin_edges,
    prepare_timeline_choices,
)
from src.neural_analysis.session_metadata import ResolvedSession, resolve_probe_sources
from src.neural_analysis.spike_behavior import loading
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage


@dataclass(frozen=True)
class TimelinePopulation:
    """Selected probe inputs: cluster_ids (U,), qualified unit_ids, UTC spikes/bounds.

    channels is a tuple of zero-based channel IDs; coverage_s is (start,end)
    UTC Unix seconds. spike_group keys are integer cluster IDs, never global IDs.
    """

    cluster_ids: np.ndarray
    unit_ids: tuple[str, ...]
    spike_group: nap.TsGroup
    coverage_s: tuple[float, float]
    channels: tuple[int, ...]


def load_timeline_behavior(
    session: ResolvedSession,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Load metadata behavior; return UTC origin, completed choices, source events.

    session contains resolved CSV paths. Events keep Time in UTC seconds and
    gain elapsed_s; missing optional event files produce an empty table.
    Choice/row identity and flag conventions are defined by prepare_timeline_choices.
    """
    trials = pd.read_csv(session.behavior.trial_table_file)
    origin, choices = prepare_timeline_choices(trials)
    path = session.behavior.event_table_file
    events = pd.DataFrame(columns=["Time", "Event", "Note", "elapsed_s"])
    if path is not None and path.is_file():
        events = pd.read_csv(path)
        if not {"Time", "Event"}.issubset(events.columns):
            raise ValueError("Event table needs Time and Event columns.")
        events["Time"] = pd.to_numeric(events["Time"], errors="coerce")
        events = (
            events.loc[np.isfinite(events["Time"])]
            .sort_values("Time", kind="stable")
            .reset_index(drop=True)
        )
        events["elapsed_s"] = events["Time"] - origin
    return origin, choices, events


def load_timeline_population(
    session: ResolvedSession, probe_id: str
) -> TimelinePopulation:
    """Load one metadata probe using existing Phy good/mua and channel-quality rules.

    Return selected units in cluster-ID order with unsmoothed UTC-second spikes
    and trusted IRIG bounds. No LFP samples or site-specific populations load.
    Missing sources, invalid quality data, or empty selections raise input errors.
    """
    source, channels, clusters = _select_population(session, probe_id)
    coverage = inspect_probe_coverage(source)
    assignments, _ = loading.load_sorter_metadata(source.sorter_directory)
    times = loading.load_aligned_spikes(source.alignment_file)
    loading.validate_aligned_spike_inputs(times, assignments)
    ids = clusters.cluster_id.to_numpy(dtype=int)
    if not np.isin(ids, assignments).all():
        raise ValueError("Selected cluster IDs are absent from spike assignments.")
    selected = np.isin(assignments, ids)
    if not np.isfinite(times[selected]).all():
        raise ValueError("Selected spike timestamps must be finite UTC seconds.")
    group = loading.build_spike_tsgroup(
        times[selected], assignments[selected], cluster_ids=ids
    )
    return TimelinePopulation(
        ids,
        tuple(f"{probe_id}:{unit}" for unit in ids),
        group,
        (coverage.start_s, coverage.end_s),
        tuple(int(ch) for ch in channels),
    )


def timeline_configuration(
    session: ResolvedSession,
    probe_roles: dict[str, str],
    bin_width_s: float,
) -> dict:
    """Return a JSON-compatible scientific/source cache identity, without spike loading.

    probe_roles maps PFC/HPC to metadata probe IDs; an absent role is unavailable.
    File identities use resolved paths, byte sizes, and mtime_ns. Display state
    is excluded. The configuration_id is a digest of these actual inputs.
    """
    choice_bin_edges(bin_width_s)
    if not set(probe_roles).issubset({"PFC", "HPC"}):
        raise ValueError("Probe roles must be PFC and/or HPC.")
    if len(set(probe_roles.values())) != len(probe_roles):
        raise ValueError("PFC and HPC must use distinct probes.")
    paths = [
        session.metadata_path,
        session.behavior.trial_table_file,
        session.behavior.event_table_file,
    ]
    selection = {}
    for role, probe_id in sorted(probe_roles.items()):
        source = resolve_probe_sources(session, probe_id)
        paths.extend(
            [
                source.alignment_file,
                source.spike_clusters_file,
                source.cluster_info_file,
                source.channel_quality_file,
            ]
        )
        try:
            _, channels, rows = _select_population(session, probe_id)
            selection[role] = {
                "unit_ids": [f"{probe_id}:{unit}" for unit in rows.cluster_id],
                "channels": channels.tolist(),
            }
        except (OSError, ValueError) as error:
            selection[role] = {"unavailable": str(error)}
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "session_id": session.session_id,
        "probe_roles": dict(probe_roles),
        "bin_width_s": bin_width_s,
        "window_s": [-2.0, 2.0],
        "quality_rules": "Phy group good/mua; good inside-brain channels; metadata unit_channels",
        "selection": selection,
        "sources": [_file_identity(path) for path in paths if path is not None],
    }
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    config["configuration_id"] = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return config


def save_timeline_result(
    output_path: Path | str, result: SessionTimelineResult
) -> Path:
    """Save a new named-array NPZ with a metadata dictionary; never overwrite.

    output_path is a new .npz file. Scientific arrays retain their existing
    shapes/units; choice/event DataFrames are stored as structured arrays.
    Return the saved path. A temporary sibling prevents loading a partial NPZ.
    """
    path = Path(output_path)
    if path.suffix != ".npz":
        raise ValueError("Timeline result must be an .npz file.")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    meta = dict(result.meta)
    meta.update(
        {
            "analysis_version": ANALYSIS_VERSION,
            "schema_version": 1,
            "generator": "src.neural_analysis.population.run_session_timeline",
            "session_id": result.session_id,
            "time_origin_s": result.time_origin_s,
            "unavailable": result.unavailable,
            "random_seed": None,
            "units": {
                "absolute_time": "UTC Unix seconds",
                "relative_time": "seconds",
                "activity": "spike counts per bin",
                "scores": "unwhitened standardized-unit scores",
                "choice_reward": "dimensionless flag: 1=rewarded, 0=unrewarded, NaN=unknown",
            },
            "axes": {
                "scores": ["choice_window", "time_bin", "session_pc"],
                "components": ["session_pc", "retained_unit"],
            },
            "regions": {},
        }
    )
    arrays = {
        "choices": result.choices.to_records(index=False),
        "events": result.events.to_records(index=False),
    }
    for role, regional in result.regions.items():
        prefix = role.lower() + "_"
        fitted = regional.transform
        meta["regions"][role] = {
            "probe_id": regional.probe_id,
            "input_unit_ids": fitted.input_unit_ids,
            "retained_unit_ids": fitted.retained_unit_ids,
            "omitted_unit_ids": fitted.omitted_unit_ids,
            "n_training_observations": fitted.n_training_observations,
            "total_session_variance": regional.total_session_variance,
            "color_limit": regional.color_limit,
            "timings_s": regional.timings_s,
        }
        for name in (
            "scores",
            "choice_positions",
            "bin_edges_s",
            "window_total_variance",
        ):
            arrays[prefix + name] = getattr(regional, name)
        for name in (
            "training_mean",
            "training_scale",
            "components",
            "explained_variance",
            "explained_variance_ratio",
        ):
            arrays[prefix + name] = getattr(fitted, name)
    arrays["meta"] = np.asarray(meta, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_timeline_result(input_path: Path | str) -> SessionTimelineResult:
    """Load a trusted project-generated NPZ without fitting or source-data access.

    input_path names an existing timeline archive. Return original table axes,
    float64 regional scores/transforms, and metadata. Object loading is used
    for the project's dictionary-valued meta and structured table columns.
    """
    with np.load(input_path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        if (
            meta.get("schema_version") != 1
            or meta.get("analysis_version") != ANALYSIS_VERSION
        ):
            raise ValueError("Unsupported timeline result version.")
        choices = pd.DataFrame.from_records(archive["choices"])
        events = pd.DataFrame.from_records(archive["events"])
        regions = {}
        for role, info in meta["regions"].items():
            prefix = role.lower() + "_"
            fitted = RegionalPCATransform(
                retained_unit_ids=tuple(info["retained_unit_ids"]),
                omitted_unit_ids=tuple(info["omitted_unit_ids"]),
                input_unit_ids=tuple(info["input_unit_ids"]),
                n_training_observations=info["n_training_observations"],
                **{
                    name: archive[prefix + name]
                    for name in (
                        "training_mean",
                        "training_scale",
                        "components",
                        "explained_variance",
                        "explained_variance_ratio",
                    )
                },
            )
            regions[role] = RegionalTimelinePCA(
                role,
                info["probe_id"],
                archive[prefix + "choice_positions"],
                archive[prefix + "bin_edges_s"],
                archive[prefix + "scores"],
                fitted,
                info["total_session_variance"],
                archive[prefix + "window_total_variance"],
                info["color_limit"],
                info["timings_s"],
            )
    return SessionTimelineResult(
        meta["session_id"],
        meta["time_origin_s"],
        choices,
        events,
        regions,
        meta["unavailable"],
        meta,
    )


# Source-boundary helpers; these deliberately do not read spike arrays.
def _select_population(session: ResolvedSession, probe_id: str) -> tuple:
    """Return resolved probe, (C,) channel IDs, and (U,columns) selected sorter rows."""
    source = resolve_probe_sources(session, probe_id)
    if source.channel_quality_file is None or source.cluster_info_file is None:
        raise ValueError(
            f"{probe_id}: metadata needs channel quality and manual Phy cluster_info.tsv."
        )
    quality = loading.load_channel_quality(source.channel_quality_file)
    channels = loading.select_channels_from_quality(
        quality, require_inside_brain=True, labels=("good",)
    )
    if source.unit_channels is not None:
        channels = np.intersect1d(channels, source.unit_channels)
    rows = loading.filter_cluster_metadata(
        pd.read_csv(source.cluster_info_file, sep="\t"),
        channels,
        quality_labels=("good", "mua"),
        quality_column="group",
    )
    if rows.empty:
        raise ValueError(
            f"{probe_id}: no manual good/mua units on selected good inside-brain channels."
        )
    if rows.cluster_id.duplicated().any():
        raise ValueError(f"{probe_id}: duplicate selected cluster IDs.")
    return source, channels, rows


def _file_identity(path: Path) -> dict:
    """Return resolved path/byte/mtime_ns identity, or missing status, without array reads."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
