"""Existing metadata/source loading and session-local timeline NPZ persistence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

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
    select_complete_choice_windows,
    trial_identities_match,
)
from src.neural_analysis.session_metadata import (
    ResolvedProbeSources, ResolvedSession, resolve_probe_sources,
)
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
    session: ResolvedSession, probe_id: str, *,
    population_quality: str = "good_mua", missing_label_policy: str = "mua",
) -> TimelinePopulation:
    """Load one metadata probe using existing Phy good/mua and channel-quality rules.

    Return selected units in cluster-ID order with unsmoothed UTC-second spikes
    and trusted IRIG bounds. No LFP samples or site-specific populations load.
    Missing sources, invalid quality data, or empty selections raise input errors.
    Keyword-only population_quality is good_mua/good; missing_label_policy is
    mua (historical fallback) or exclude (explicit manual labels only). Inputs
    are scalar policy strings; returned unit/spike axes and UTC units are unchanged.
    """
    source, channels, clusters = select_timeline_population(
        session, probe_id, population_quality=population_quality,
        missing_label_policy=missing_label_policy,
    )
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
    group = _build_timeline_spike_group(
        times[selected], assignments[selected], ids, (coverage.start_s, coverage.end_s),
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
    *,
    population_quality: str = "good_mua",
    missing_label_policy: str = "mua",
) -> dict:
    """Return a JSON-compatible scientific/source cache identity, without spike loading.

    probe_roles maps PFC/HPC to metadata probe IDs; an absent role is unavailable.
    File identities use resolved paths, byte sizes, and mtime_ns. Display state
    is excluded. The configuration_id is a digest of these actual inputs.
    bin_width_s is seconds; the keyword-only scalar quality/policy labels use
    the load_timeline_population contract. Both defaults preserve the previous
    dict/digest exactly; nondefault policies add explicit scientific identity.
    """
    _validate_population_policy(population_quality, missing_label_policy)
    choice_bin_edges(bin_width_s)
    if not set(probe_roles).issubset({"PFC", "HPC"}):
        raise ValueError("Probe roles must be PFC and/or HPC.")
    if len(set(probe_roles.values())) != len(probe_roles):
        raise ValueError("PFC and HPC must use distinct probes.")
    paths = [
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
            _, channels, rows = select_timeline_population(
                session, probe_id, population_quality=population_quality,
                missing_label_policy=missing_label_policy,
            )
            selection[role] = {
                "unit_ids": [f"{probe_id}:{unit}" for unit in rows.cluster_id],
                "channels": channels.tolist(),
            }
        except (OSError, ValueError) as error:
            selection[role] = {"unavailable": str(error)}
    config = {
        "analysis_version": ANALYSIS_VERSION,
        "configuration_identity_version": 2,
        "acquisition_families": {
            role: resolve_probe_sources(session, probe).acquisition_family
            for role, probe in sorted(probe_roles.items())
        },
        "session_id": session.session_id,
        "probe_roles": dict(probe_roles),
        "bin_width_s": bin_width_s,
        "window_s": [-2.0, 2.0],
        "quality_rules": "Phy group good/mua; good inside-brain channels; metadata unit_channels",
        "selection": selection,
        "sources": [_file_identity(path) for path in paths if path is not None],
    }
    if population_quality != "good_mua" or missing_label_policy != "mua":
        config.update(
            population_quality=population_quality, missing_label_policy=missing_label_policy,
            quality_rules=_quality_rule_description(population_quality, missing_label_policy),
        )
    config["configuration_id"] = _configuration_digest(config)
    return config


def select_timeline_population(
    session: ResolvedSession, probe_id: str, *,
    population_quality: str = "good_mua", missing_label_policy: str = "mua",
) -> tuple[ResolvedProbeSources, np.ndarray, pd.DataFrame]:
    """Select manual-quality metadata without loading spike arrays.

    Parameters
    ----------
    session : ResolvedSession
        Metadata/source paths for one session; no scientific arrays supplied.
    probe_id : str
        Metadata probe identity, used to qualify selected units downstream.
    population_quality : str, keyword-only
        good_mua includes manual Phy good/mua; good includes good only. Case
        and surrounding label whitespace are normalized; KSLabel is never used.
    missing_label_policy : str, keyword-only
        mua retains historical missing-to-MUA behavior; exclude drops real
        missing, blank and string missing labels from either population mode.

    Returns
    -------
    tuple[ResolvedProbeSources, np.ndarray, pd.DataFrame]
        Resolved source, integer channel IDs (C,), and cluster metadata (U,C').
        Channels are zero-based good inside-brain channels intersected with
        metadata unit_channels. Rows are sorted by integer cluster_id and add
        normalized quality_label. All identities/labels are unitless. Missing
        required manual metadata, no selected units or duplicate selected IDs
        raise ValueError; policy errors are validated before reading files.
    """
    _validate_population_policy(population_quality, missing_label_policy)
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
    labels = ("good", "mua") if population_quality == "good_mua" else ("good",)
    cluster_metadata = pd.read_csv(source.cluster_info_file, sep="\t")
    if "group" in cluster_metadata:
        # An entirely empty CSV group column is inferred as float. Give the
        # existing normalizer an object column before it fills string labels.
        cluster_metadata["group"] = cluster_metadata["group"].astype(object)
    rows = loading.filter_cluster_metadata(
        cluster_metadata, channels,
        quality_labels=labels, quality_column="group",
        default_group="mua" if missing_label_policy == "mua" else "",
    )
    if rows.empty:
        raise ValueError(
            f"{probe_id}: no manual {'/'.join(labels)} units on selected good inside-brain channels."
        )
    if rows.cluster_id.duplicated().any():
        raise ValueError(f"{probe_id}: duplicate selected cluster IDs.")
    return source, channels, rows


def find_compatible_regional_timeline(
    session: ResolvedSession, region: str, probe_id: str, bin_width_s: float, *,
    population_quality: str = "good_mua", missing_label_policy: str = "exclude",
) -> Path | None:
    """Find an exact existing regional basis without loading raw neural activity.

    Parameters
    ----------
    session : ResolvedSession
        Current metadata/source paths. Only trial CSV, small quality tables,
        file identities and trusted IRIG coverage are inspected; optional
        display events and aligned spike/assignment arrays are not loaded.
    region, probe_id : str
        PFC/HPC role and its requested metadata probe. Other probes need not
        match a candidate's whole-session configuration.
    bin_width_s : float
        Supported choice-window bin width in seconds; windows are [-2,+2) s.
    population_quality, missing_label_policy : str, keyword-only
        Same unitless policies as select_timeline_population. Default lookup
        requires explicit manual good/mua labels. A default/legacy basis is
        eligible only if its actual input IDs equal the requested population.

    Returns
    -------
    Path or None
        Newest compatible session-local NPZ, or None if unavailable. Reuse
        verifies source identities, full coverage-eligible (W,) trial pool,
        (W,B,K) signed unwhitened scores and complete (K,U_retained) basis.
        Old metadata-bound archives require their immutable registration
        receipt. Invalid candidates are skipped without modification or fitting.
    """
    current = timeline_configuration(
        session, {region: probe_id}, bin_width_s,
        population_quality=population_quality, missing_label_policy=missing_label_policy,
    )
    if "unavailable" in current["selection"][region]:
        return None
    try:
        trials = pd.read_csv(session.behavior.trial_table_file)
        origin, choices = prepare_timeline_choices(trials)
        source = resolve_probe_sources(session, probe_id)
        coverage = inspect_probe_coverage(source)
        positions = select_complete_choice_windows(
            choices.choice_time.to_numpy(), (coverage.start_s, coverage.end_s),
        )
        if positions.size == 0:
            return None
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return None

    paths = (session.session_root / "analysis_runs").glob("session_timeline_pca_*/results.npz")
    for path in sorted(paths, reverse=True):
        try:
            # Most archives differ in bin width or sources. Reject those using
            # metadata before decompressing any regional score tensors.
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
            if (meta.get("schema_version") != 1 or meta.get("analysis_version") != ANALYSIS_VERSION
                    or meta.get("regions", {}).get(region, {}).get("probe_id") != probe_id):
                continue
            configuration = _regional_candidate_configuration(path, meta)
            if configuration is None:
                continue
            if _regional_sources_match(configuration, current, session, source, region):
                candidate = load_timeline_result(path)
                if _regional_arrays_match(candidate, origin, choices, positions, current, region, probe_id):
                    return path
        except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError, EOFError, BadZipFile):
            continue
    return None


def register_timeline_scientific_identity(
    session: ResolvedSession, result_path: Path | str
) -> Path:
    """Register an exact legacy result BEFORE its original metadata is edited.

    session is the unchanged resolved metadata; result_path is a completed v1
    timeline NPZ under that session's analysis_runs. Read only archive metadata,
    verify the original full configuration, and write a small new JSON record.
    Return its path. Existing NPZ bytes/mtime and array axes/units are untouched.
    A conflicting record or an edited original source raises ValueError.
    """
    path = Path(result_path).resolve(strict=True)
    if path.name != "results.npz" or not path.is_relative_to(
        session.session_root / "analysis_runs"
    ):
        raise ValueError("Legacy result must be in this session's analysis_runs.")
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
    if (
        meta.get("schema_version") != 1
        or meta.get("analysis_version") != ANALYSIS_VERSION
    ):
        raise ValueError("Unsupported legacy timeline version.")
    current = timeline_configuration(session, meta["probe_roles"], meta["bin_width_s"])
    legacy = {
        key: current[key]
        for key in (
            "analysis_version",
            "session_id",
            "probe_roles",
            "bin_width_s",
            "window_s",
            "quality_rules",
            "selection",
            "sources",
        )
    }
    legacy["sources"] = [_file_identity(session.metadata_path)] + current["sources"]
    legacy["configuration_id"] = _configuration_digest(legacy)
    if any(meta.get(key) != value for key, value in legacy.items()):
        raise ValueError(
            "Legacy configuration does not match the original metadata and sources."
        )
    record = {
        "schema_version": 1,
        "analysis_version": ANALYSIS_VERSION,
        "configuration": current,
        "legacy_configuration_id": meta["configuration_id"],
        "result_identity": _file_identity(path),
    }
    target = path.parent / "scientific_identity.json"
    if target.exists():
        if json.loads(target.read_text()) != record:
            raise ValueError("Conflicting existing scientific identity record.")
    else:
        with target.open("x", encoding="ascii") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    return target


def find_registered_timeline(session_root: Path, configuration_id: str) -> Path | None:
    """Find a registered immutable legacy result by its new scientific ID.

    Only JSON and NPZ metadata are read. Require the recorded result to remain
    in that same run with unchanged path/bytes/mtime and supported versions.
    Invalid or stale records are ignored; no files are written.
    """
    for record_path in sorted(
        (Path(session_root) / "analysis_runs").glob(
            "session_timeline_pca_*/scientific_identity.json"
        ),
        reverse=True,
    ):
        try:
            record = json.loads(record_path.read_text())
            config = record["configuration"]
            if (
                record["schema_version"] != 1
                or record["analysis_version"] != ANALYSIS_VERSION
                or config["configuration_id"] != configuration_id
                or _configuration_digest(config) != configuration_id
            ):
                continue
            path = record_path.parent / "results.npz"
            if record["result_identity"] != _file_identity(path):
                continue
            with np.load(path, allow_pickle=True) as archive:
                meta = archive["meta"].item()
                if (
                    meta.get("schema_version") == 1
                    and meta.get("analysis_version") == ANALYSIS_VERSION
                    and meta.get("configuration_id")
                    == record["legacy_configuration_id"]
                ):
                    return path
        except (OSError, ValueError, KeyError, EOFError):
            continue
    return None


def _configuration_digest(config: dict) -> str:
    """Return a stable 16-character scientific digest, excluding its own ID field."""
    values = {key: value for key, value in config.items() if key != "configuration_id"}
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


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


# Source/compatibility helpers; raw spike file reads remain in the loader.
def _build_timeline_spike_group(
    times_s: np.ndarray, assignments: np.ndarray, cluster_ids: np.ndarray,
    coverage_s: tuple[float, float],
) -> nap.TsGroup:
    """Input: UTC spikes/assignments (S,), unit IDs (U,), trusted UTC bounds (2,).

    Return a TsGroup with sorted UTC-second Ts per cluster. Ordinary inputs use
    the existing shared builder. Empty/single-timestamp trains require explicit
    trusted support because Pynapple cannot infer positive-duration support
    there. Support includes the trusted bounds and every selected finite spike;
    the separate coverage bounds alone determine eligible choice windows.
    """
    spikes = {int(unit): np.sort(times_s[assignments == unit]) for unit in cluster_ids}
    if all(len(values) > 1 and values[-1] > values[0] for values in spikes.values()):
        return loading.build_spike_tsgroup(times_s, assignments, cluster_ids=cluster_ids)
    start_s, end_s = coverage_s
    if times_s.size:
        start_s = min(start_s, float(times_s.min()))
        end_s = max(end_s, float(times_s.max()))
    support = nap.IntervalSet(start=start_s, end=end_s, time_units="s")
    return nap.TsGroup(
        {unit: nap.Ts(t=values, time_units="s", time_support=support) for unit, values in spikes.items()},
        time_support=support,
    )


def _validate_population_policy(population_quality: str, missing_label_policy: str) -> None:
    """Validate two scalar unitless policy strings; return None or raise ValueError."""
    if population_quality not in ("good_mua", "good"):
        raise ValueError("population_quality must be good_mua or good.")
    if missing_label_policy not in ("mua", "exclude"):
        raise ValueError("missing_label_policy must be mua or exclude.")


def _quality_rule_description(population_quality: str, missing_label_policy: str) -> str:
    """Input: scalar quality/policy strings. Return unitless scientific rule description."""
    labels = "good/mua" if population_quality == "good_mua" else "good"
    rule = f"Phy group {labels}; good inside-brain channels; metadata unit_channels"
    if population_quality != "good_mua" or missing_label_policy != "mua":
        rule += f"; missing labels {missing_label_policy}"
    return rule


def _regional_candidate_configuration(path: Path, meta: dict) -> dict | None:
    """Input: saved NPZ Path/meta dict. Return verified scientific config or None.

    Configuration values are unitless identities except widths/windows in
    seconds and source file sizes/mtime in bytes/ns. Legacy receipts are checked
    using their original whole-session configuration, never a requested subset.
    """
    if meta.get("configuration_identity_version") == 2:
        fields = (
            "analysis_version", "configuration_identity_version", "acquisition_families",
            "session_id", "probe_roles", "bin_width_s", "window_s", "quality_rules",
            "selection", "sources", "configuration_id",
        )
        configuration = {key: meta[key] for key in fields}
        for key in ("population_quality", "missing_label_policy"):
            if key in meta:
                configuration[key] = meta[key]
        if _configuration_digest(configuration) != configuration["configuration_id"]:
            return None
    else:
        receipt = json.loads((path.parent / "scientific_identity.json").read_text())
        configuration = receipt["configuration"]
        registered = find_registered_timeline(path.parents[2], configuration["configuration_id"])
        if registered != path:
            return None
    quality = configuration.get("population_quality", "good_mua")
    missing = configuration.get("missing_label_policy", "mua")
    _validate_population_policy(quality, missing)
    if configuration["quality_rules"] != _quality_rule_description(quality, missing):
        return None
    return configuration


def _regional_sources_match(
    candidate: dict, current: dict, session: ResolvedSession,
    source: ResolvedProbeSources, region: str,
) -> bool:
    """Input: candidate/current configs, session/probe records, role. Return match bool.

    Only the requested probe's source identities, acquisition, selected unit
    IDs and allowed channels participate. Trial CSV identity is required;
    unrelated probes and optional event display files do not affect this basis.
    Width/window values are seconds; unit/channel/source labels are unitless.
    """
    for field in ("analysis_version", "session_id", "bin_width_s", "window_s"):
        if candidate.get(field) != current[field]:
            return False
    if candidate["probe_roles"].get(region) != current["probe_roles"][region]:
        return False
    if candidate["acquisition_families"].get(region) != source.acquisition_family:
        return False
    if candidate["selection"].get(region) != current["selection"][region]:
        return False
    previous_sources = {entry["path"]: entry for entry in candidate["sources"]}
    relevant = (
        session.behavior.trial_table_file, source.alignment_file,
        source.spike_clusters_file, source.cluster_info_file, source.channel_quality_file,
    )
    return all(
        path is not None and previous_sources.get(str(path.resolve())) == _file_identity(path)
        for path in relevant
    )


def _regional_arrays_match(
    candidate: SessionTimelineResult, origin_s: float, choices: pd.DataFrame,
    positions: np.ndarray, current: dict, region: str, probe_id: str,
) -> bool:
    """Input: saved result, current UTC origin/(N,C) choices/(W,) map/config/labels.

    Return bool for the exact current eligible trial pool, native (W,B,K)
    scores, and complete unwhitened (K,U_retained) basis. Bins are relative
    seconds. Dataclass loading already validates finite transform arrays and
    disjoint retained/omitted IDs; no raw-count reconstruction occurs here.
    """
    if candidate.time_origin_s != origin_s or candidate.session_id != current["session_id"]:
        return False
    saved = candidate.regions.get(region)
    if saved is None or saved.probe_id != probe_id:
        return False
    edges = choice_bin_edges(current["bin_width_s"])
    if not np.array_equal(saved.bin_edges_s, edges):
        return False
    old_positions = saved.choice_positions
    if (old_positions.ndim != 1 or not np.issubdtype(old_positions.dtype, np.integer)
            or np.any(old_positions < 0) or np.any(old_positions >= len(candidate.choices))):
        return False
    previous = candidate.choices.iloc[old_positions]
    expected = choices.iloc[positions]
    if "trial_id" not in previous or not trial_identities_match(previous["trial_id"], expected["trial_id"]):
        return False
    for column in ("row_position", "choice_time", "action"):
        if column not in previous or not np.array_equal(previous[column].to_numpy(), expected[column].to_numpy()):
            return False
    fitted = saved.transform
    unit_ids = tuple(current["selection"][region]["unit_ids"])
    observations = len(positions) * (len(edges) - 1)
    components = min(observations, len(fitted.retained_unit_ids))
    if (fitted.input_unit_ids != unit_ids or fitted.n_training_observations != observations
            or fitted.actual_components != components or fitted.whiten
            or saved.scores.shape != (len(positions), len(edges) - 1, components)
            or not np.isfinite(saved.scores).all()):
        return False
    for labels in (fitted.retained_unit_ids, fitted.omitted_unit_ids):
        if tuple(unit for unit in unit_ids if unit in labels) != labels:
            return False
    if not np.allclose(fitted.components @ fitted.components.T, np.eye(components), atol=1e-7):
        return False
    if (np.any(fitted.explained_variance < 0) or np.any(fitted.explained_variance_ratio < 0)
            or saved.window_total_variance.shape != (len(positions),)
            or not np.isfinite(saved.window_total_variance).all()
            or not np.isfinite(saved.total_session_variance) or saved.total_session_variance <= 0):
        return False
    return (
        candidate.meta.get("units", {}).get("scores") == "unwhitened standardized-unit scores"
        and candidate.meta.get("axes", {}).get("scores") == ["choice_window", "time_bin", "session_pc"]
    )


def _file_identity(path: Path) -> dict:
    """Return resolved path/byte/mtime_ns identity, or missing status, without array reads."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
