"""Load and resolve the editable metadata for one neural recording session.

The metadata names existing sources. It does not contain scientific settings,
load numerical arrays, or search a session directory for likely files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "1"
CANONICAL_FILENAME = "neural_session.json"
SUPPORTED_ACTIONS = ("webapp", "power", "synchrony", "spike-phase")
SUPPORTED_ACQUISITION_FAMILIES = ("open_ephys", "spikeglx")


@dataclass(frozen=True)
class BehaviorSources:
    """Session-relative behavior paths; values are strings with no physical units."""

    session_directory: str
    trial_table_file: str
    event_table_file: str | None
    treadmill_file: str | None


@dataclass(frozen=True)
class ProbeSources:
    """Source locations and labels for one physical acquisition probe."""

    probe_id: str
    display_label: str
    acquisition_family: str
    lfp_file: str
    lfp_metadata_file: str
    synchronization_file: str
    sorter_directory: str
    aligned_spike_file: str
    channel_quality_file: str | None


@dataclass(frozen=True)
class SiteMetadata:
    """One named LFP site with a zero-based saved-channel index."""

    site_id: str
    display_label: str
    probe_id: str
    saved_channel_index: int


@dataclass(frozen=True)
class ChannelGroupMetadata:
    """One anatomical group of zero-based acquisition-channel indices."""

    channel_group_id: str
    display_label: str
    probe_id: str
    channel_indices: tuple[int, ...]


@dataclass(frozen=True)
class PopulationMetadata:
    """A stable population label linked to one probe and anatomical group."""

    population_id: str
    display_label: str
    probe_id: str
    channel_group_id: str


@dataclass(frozen=True)
class SessionMetadata:
    """Structurally validated version-1 metadata with relative source paths."""

    schema_version: str
    subject_id: str
    session_id: str
    session_date: str | None
    session_label: str | None
    behavior: BehaviorSources
    probes: tuple[ProbeSources, ...]
    sites: tuple[SiteMetadata, ...]
    site_pairs: tuple[tuple[str, str], ...]
    channel_groups: tuple[ChannelGroupMetadata, ...]
    populations: tuple[PopulationMetadata, ...]
    lfp_summary_cache_directory: str | None
    lfp_summary_snapshot_directory: str | None


@dataclass(frozen=True)
class ResolvedBehaviorSources:
    """Absolute behavior paths for one resolved session."""

    session_directory: Path | None
    trial_table_file: Path | None
    event_table_file: Path | None
    treadmill_file: Path | None


@dataclass(frozen=True)
class ResolvedProbeSources:
    """Absolute paths for one probe, including fixed sorter child filenames."""

    probe_id: str
    display_label: str
    acquisition_family: str
    lfp_file: Path | None
    lfp_metadata_file: Path | None
    synchronization_file: Path | None
    sorter_directory: Path | None
    aligned_spike_file: Path | None
    channel_quality_file: Path | None
    spike_times_file: Path | None
    spike_clusters_file: Path | None
    cluster_info_file: Path | None


@dataclass(frozen=True)
class ResolvedSession:
    """Session metadata with contained absolute paths and unchanged labels."""

    metadata_path: Path
    session_root: Path
    schema_version: str
    subject_id: str
    session_id: str
    session_date: str | None
    session_label: str | None
    behavior: ResolvedBehaviorSources
    probes: tuple[ResolvedProbeSources, ...]
    sites: tuple[SiteMetadata, ...]
    site_pairs: tuple[tuple[str, str], ...]
    channel_groups: tuple[ChannelGroupMetadata, ...]
    populations: tuple[PopulationMetadata, ...]
    lfp_summary_cache_directory: Path | None
    lfp_summary_snapshot_directory: Path | None


@dataclass(frozen=True)
class ValidationResult:
    """Availability of one existing action and its ordered missing inputs."""

    action: str
    available: bool
    missing_inputs: tuple[str, ...]


_SESSION_KEYS = frozenset(
    {
        "schema_version",
        "subject_id",
        "session_id",
        "session_date",
        "session_label",
        "behavior",
        "probes",
        "sites",
        "site_pairs",
        "channel_groups",
        "populations",
        "lfp_summary_cache_directory",
        "lfp_summary_snapshot_directory",
    }
)
_BEHAVIOR_KEYS = frozenset(
    {"session_directory", "trial_table_file", "event_table_file", "treadmill_file"}
)
_PROBE_KEYS = frozenset(
    {
        "probe_id",
        "display_label",
        "acquisition_family",
        "lfp_file",
        "lfp_metadata_file",
        "synchronization_file",
        "sorter_directory",
        "aligned_spike_file",
        "channel_quality_file",
    }
)
_SITE_KEYS = frozenset({"site_id", "display_label", "probe_id", "saved_channel_index"})
_CHANNEL_GROUP_KEYS = frozenset(
    {"channel_group_id", "display_label", "probe_id", "channel_indices"}
)
_POPULATION_KEYS = frozenset(
    {"population_id", "display_label", "probe_id", "channel_group_id"}
)


def _mapping(value: object, field: str) -> Mapping[str, object]:
    """Return ``value`` as a string-keyed mapping or raise ``ValueError``."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], field: str) -> None:
    """Require exactly the documented keys for one metadata record."""
    missing = sorted(expected - value.keys())
    unexpected = sorted(value.keys() - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{field} has {' and '.join(details)}")


def _string(value: object, field: str, *, allow_empty: bool = True) -> str:
    """Return one string, optionally requiring nonempty content."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    """Return a string or ``None`` while preserving an explicit empty string."""
    if value is None:
        return None
    return _string(value, field)


def _sequence(value: object, field: str) -> Sequence[object]:
    """Return a JSON array while excluding strings and mappings."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _parse_behavior(value: object) -> BehaviorSources:
    """Parse the session-level behavior source record."""
    record = _mapping(value, "behavior")
    _exact_keys(record, _BEHAVIOR_KEYS, "behavior")
    return BehaviorSources(
        session_directory=_string(record["session_directory"], "behavior.session_directory"),
        trial_table_file=_string(record["trial_table_file"], "behavior.trial_table_file"),
        event_table_file=_optional_string(record["event_table_file"], "behavior.event_table_file"),
        treadmill_file=_optional_string(record["treadmill_file"], "behavior.treadmill_file"),
    )


def _parse_probe(value: object, index: int) -> ProbeSources:
    """Parse one probe source record at ``index``."""
    field = f"probes[{index}]"
    record = _mapping(value, field)
    _exact_keys(record, _PROBE_KEYS, field)
    family = _string(record["acquisition_family"], f"{field}.acquisition_family", allow_empty=False)
    if family not in SUPPORTED_ACQUISITION_FAMILIES:
        raise ValueError(
            f"{field}.acquisition_family must be one of {SUPPORTED_ACQUISITION_FAMILIES}"
        )
    return ProbeSources(
        probe_id=_string(record["probe_id"], f"{field}.probe_id", allow_empty=False),
        display_label=_string(record["display_label"], f"{field}.display_label", allow_empty=False),
        acquisition_family=family,
        lfp_file=_string(record["lfp_file"], f"{field}.lfp_file"),
        lfp_metadata_file=_string(record["lfp_metadata_file"], f"{field}.lfp_metadata_file"),
        synchronization_file=_string(
            record["synchronization_file"], f"{field}.synchronization_file"
        ),
        sorter_directory=_string(record["sorter_directory"], f"{field}.sorter_directory"),
        aligned_spike_file=_string(record["aligned_spike_file"], f"{field}.aligned_spike_file"),
        channel_quality_file=_optional_string(
            record["channel_quality_file"], f"{field}.channel_quality_file"
        ),
    )


def _parse_site(value: object, index: int) -> SiteMetadata:
    """Parse one site record with a nonnegative saved-channel index."""
    field = f"sites[{index}]"
    record = _mapping(value, field)
    _exact_keys(record, _SITE_KEYS, field)
    channel_index = record["saved_channel_index"]
    if isinstance(channel_index, bool) or not isinstance(channel_index, int) or channel_index < 0:
        raise ValueError(f"{field}.saved_channel_index must be a nonnegative integer")
    return SiteMetadata(
        site_id=_string(record["site_id"], f"{field}.site_id", allow_empty=False),
        display_label=_string(record["display_label"], f"{field}.display_label", allow_empty=False),
        probe_id=_string(record["probe_id"], f"{field}.probe_id", allow_empty=False),
        saved_channel_index=channel_index,
    )


def _parse_channel_group(value: object, index: int) -> ChannelGroupMetadata:
    """Parse one anatomical channel group with unique nonnegative indices."""
    field = f"channel_groups[{index}]"
    record = _mapping(value, field)
    _exact_keys(record, _CHANNEL_GROUP_KEYS, field)
    raw_indices = _sequence(record["channel_indices"], f"{field}.channel_indices")
    indices = tuple(raw_indices)
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in indices):
        raise ValueError(f"{field}.channel_indices must contain nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError(f"{field}.channel_indices must not contain duplicates")
    return ChannelGroupMetadata(
        channel_group_id=_string(
            record["channel_group_id"], f"{field}.channel_group_id", allow_empty=False
        ),
        display_label=_string(record["display_label"], f"{field}.display_label", allow_empty=False),
        probe_id=_string(record["probe_id"], f"{field}.probe_id", allow_empty=False),
        channel_indices=indices,
    )


def _parse_population(value: object, index: int) -> PopulationMetadata:
    """Parse one stable population reference."""
    field = f"populations[{index}]"
    record = _mapping(value, field)
    _exact_keys(record, _POPULATION_KEYS, field)
    return PopulationMetadata(
        population_id=_string(
            record["population_id"], f"{field}.population_id", allow_empty=False
        ),
        display_label=_string(record["display_label"], f"{field}.display_label", allow_empty=False),
        probe_id=_string(record["probe_id"], f"{field}.probe_id", allow_empty=False),
        channel_group_id=_string(
            record["channel_group_id"], f"{field}.channel_group_id", allow_empty=False
        ),
    )


def _require_unique(values: Sequence[str], field: str) -> None:
    """Require unique stable identifiers within one record collection."""
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate IDs")


def _validate_references(metadata: SessionMetadata) -> None:
    """Validate probe, site, group, pair, and population references."""
    probe_ids = {probe.probe_id for probe in metadata.probes}
    site_ids = {site.site_id for site in metadata.sites}
    group_by_id = {group.channel_group_id: group for group in metadata.channel_groups}
    for site in metadata.sites:
        if site.probe_id not in probe_ids:
            raise ValueError(f"site {site.site_id!r} references unknown probe {site.probe_id!r}")
    for first, second in metadata.site_pairs:
        if first not in site_ids or second not in site_ids:
            raise ValueError(f"site pair {(first, second)!r} references an unknown site")
        if first == second:
            raise ValueError("site pairs must contain two different sites")
    for group in metadata.channel_groups:
        if group.probe_id not in probe_ids:
            raise ValueError(
                f"channel group {group.channel_group_id!r} references unknown probe {group.probe_id!r}"
            )
    for population in metadata.populations:
        group = group_by_id.get(population.channel_group_id)
        if population.probe_id not in probe_ids:
            raise ValueError(
                f"population {population.population_id!r} references unknown probe"
            )
        if group is None:
            raise ValueError(
                f"population {population.population_id!r} references unknown channel group"
            )
        if group.probe_id != population.probe_id:
            raise ValueError(
                f"population {population.population_id!r} and its channel group use different probes"
            )


def load_session_metadata(path: Path | str) -> SessionMetadata:
    """Load structurally valid version-1 metadata without filesystem resolution.

    Parameters
    ----------
    path : pathlib.Path or str
        JSON metadata file. The file contains labels, integer channel indices,
        and session-relative path strings; it contains no numerical arrays.

    Returns
    -------
    SessionMetadata
        Frozen records preserving JSON list order and explicit ``None`` versus
        empty-string values.
    """
    metadata_path = Path(path)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read session metadata {metadata_path}: {error}") from error
    record = _mapping(payload, "session metadata")
    _exact_keys(record, _SESSION_KEYS, "session metadata")
    schema_version = _string(record["schema_version"], "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")

    probes = tuple(
        _parse_probe(item, index)
        for index, item in enumerate(_sequence(record["probes"], "probes"))
    )
    sites = tuple(
        _parse_site(item, index)
        for index, item in enumerate(_sequence(record["sites"], "sites"))
    )
    channel_groups = tuple(
        _parse_channel_group(item, index)
        for index, item in enumerate(_sequence(record["channel_groups"], "channel_groups"))
    )
    populations = tuple(
        _parse_population(item, index)
        for index, item in enumerate(_sequence(record["populations"], "populations"))
    )
    raw_pairs = _sequence(record["site_pairs"], "site_pairs")
    site_pairs: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(raw_pairs):
        pair = _sequence(raw_pair, f"site_pairs[{index}]")
        if len(pair) != 2:
            raise ValueError(f"site_pairs[{index}] must contain exactly two site IDs")
        site_pairs.append(
            (
                _string(pair[0], f"site_pairs[{index}][0]", allow_empty=False),
                _string(pair[1], f"site_pairs[{index}][1]", allow_empty=False),
            )
        )

    metadata = SessionMetadata(
        schema_version=schema_version,
        subject_id=_string(record["subject_id"], "subject_id"),
        session_id=_string(record["session_id"], "session_id"),
        session_date=_optional_string(record["session_date"], "session_date"),
        session_label=_optional_string(record["session_label"], "session_label"),
        behavior=_parse_behavior(record["behavior"]),
        probes=probes,
        sites=sites,
        site_pairs=tuple(site_pairs),
        channel_groups=channel_groups,
        populations=populations,
        lfp_summary_cache_directory=_optional_string(
            record["lfp_summary_cache_directory"], "lfp_summary_cache_directory"
        ),
        lfp_summary_snapshot_directory=_optional_string(
            record["lfp_summary_snapshot_directory"], "lfp_summary_snapshot_directory"
        ),
    )
    _require_unique([probe.probe_id for probe in probes], "probes")
    _require_unique([site.site_id for site in sites], "sites")
    _require_unique([group.channel_group_id for group in channel_groups], "channel_groups")
    _require_unique([population.population_id for population in populations], "populations")
    _validate_references(metadata)
    return metadata


def _resolve_path(
    session_root: Path,
    value: str | None,
    field: str,
    *,
    expected_kind: str,
) -> Path | None:
    """Resolve one optional relative path and check an existing path's kind."""
    if value is None or value == "":
        return None
    relative_path = Path(value)
    if relative_path.is_absolute():
        raise ValueError(f"{field} must be relative to the session root")
    resolved = (session_root / relative_path).resolve(strict=False)
    if not resolved.is_relative_to(session_root):
        raise ValueError(f"{field} must remain contained within the session root")
    if resolved.exists():
        if expected_kind == "file" and not resolved.is_file():
            raise ValueError(f"{field} must identify a file")
        if expected_kind == "directory" and not resolved.is_dir():
            raise ValueError(f"{field} must identify a directory")
    return resolved


def resolve_session_metadata(
    metadata: SessionMetadata,
    metadata_path: Path | str,
) -> ResolvedSession:
    """Resolve all declared paths below the canonical session metadata file.

    Parameters
    ----------
    metadata : SessionMetadata
        Structurally validated relative-path records.
    metadata_path : pathlib.Path or str
        Existing canonical ``<session-root>/neural_session.json`` path.

    Returns
    -------
    ResolvedSession
        Frozen records whose nonempty source locations are absolute paths.
        Missing files remain represented so action validation can report them.
    """
    canonical_path = Path(metadata_path)
    if canonical_path.name != CANONICAL_FILENAME:
        raise ValueError(f"metadata filename must be {CANONICAL_FILENAME}")
    session_root = canonical_path.parent.resolve(strict=True)
    behavior = metadata.behavior
    resolved_behavior = ResolvedBehaviorSources(
        session_directory=_resolve_path(
            session_root,
            behavior.session_directory,
            "behavior.session_directory",
            expected_kind="directory",
        ),
        trial_table_file=_resolve_path(
            session_root,
            behavior.trial_table_file,
            "behavior.trial_table_file",
            expected_kind="file",
        ),
        event_table_file=_resolve_path(
            session_root,
            behavior.event_table_file,
            "behavior.event_table_file",
            expected_kind="file",
        ),
        treadmill_file=_resolve_path(
            session_root,
            behavior.treadmill_file,
            "behavior.treadmill_file",
            expected_kind="file",
        ),
    )
    resolved_probes = []
    for probe in metadata.probes:
        prefix = f"probes[{probe.probe_id}]"
        sorter_directory = _resolve_path(
            session_root,
            probe.sorter_directory,
            f"{prefix}.sorter_directory",
            expected_kind="directory",
        )
        resolved_probes.append(
            ResolvedProbeSources(
                probe_id=probe.probe_id,
                display_label=probe.display_label,
                acquisition_family=probe.acquisition_family,
                lfp_file=_resolve_path(
                    session_root, probe.lfp_file, f"{prefix}.lfp_file", expected_kind="file"
                ),
                lfp_metadata_file=_resolve_path(
                    session_root,
                    probe.lfp_metadata_file,
                    f"{prefix}.lfp_metadata_file",
                    expected_kind="file",
                ),
                synchronization_file=_resolve_path(
                    session_root,
                    probe.synchronization_file,
                    f"{prefix}.synchronization_file",
                    expected_kind="file",
                ),
                sorter_directory=sorter_directory,
                aligned_spike_file=_resolve_path(
                    session_root,
                    probe.aligned_spike_file,
                    f"{prefix}.aligned_spike_file",
                    expected_kind="file",
                ),
                channel_quality_file=_resolve_path(
                    session_root,
                    probe.channel_quality_file,
                    f"{prefix}.channel_quality_file",
                    expected_kind="file",
                ),
                spike_times_file=sorter_directory / "spike_times.npy" if sorter_directory else None,
                spike_clusters_file=sorter_directory / "spike_clusters.npy" if sorter_directory else None,
                cluster_info_file=sorter_directory / "cluster_info.tsv" if sorter_directory else None,
            )
        )
    return ResolvedSession(
        metadata_path=canonical_path.resolve(strict=True),
        session_root=session_root,
        schema_version=metadata.schema_version,
        subject_id=metadata.subject_id,
        session_id=metadata.session_id,
        session_date=metadata.session_date,
        session_label=metadata.session_label,
        behavior=resolved_behavior,
        probes=tuple(resolved_probes),
        sites=metadata.sites,
        site_pairs=metadata.site_pairs,
        channel_groups=metadata.channel_groups,
        populations=metadata.populations,
        lfp_summary_cache_directory=_resolve_path(
            session_root,
            metadata.lfp_summary_cache_directory,
            "lfp_summary_cache_directory",
            expected_kind="directory",
        ),
        lfp_summary_snapshot_directory=_resolve_path(
            session_root,
            metadata.lfp_summary_snapshot_directory,
            "lfp_summary_snapshot_directory",
            expected_kind="directory",
        ),
    )


def resolve_probe_sources(session: ResolvedSession, probe_id: str) -> ResolvedProbeSources:
    """Return resolved sources for one exact stable probe ID.

    Parameters
    ----------
    session : ResolvedSession
        Resolved metadata for one session.
    probe_id : str
        Exact stable hardware/probe identifier.

    Returns
    -------
    ResolvedProbeSources
        Source paths and labels for the selected probe.
    """
    for probe in session.probes:
        if probe.probe_id == probe_id:
            return probe
    raise KeyError(f"unknown probe ID: {probe_id}")


def _missing_path(path: Path | None, expected_kind: str) -> bool:
    """Return whether one action input is absent or has the wrong path kind."""
    if path is None:
        return True
    if expected_kind == "file":
        return not path.is_file()
    return not path.is_dir()


def _append_missing(
    missing: list[str],
    label: str,
    path: Path | None,
    expected_kind: str,
) -> None:
    """Append ``label`` when a required path is not presently usable."""
    if _missing_path(path, expected_kind):
        missing.append(label)


def validate_session_for_action(session: ResolvedSession, action: str) -> ValidationResult:
    """Report the ordinary missing inputs for one existing user action.

    Parameters
    ----------
    session : ResolvedSession
        Resolved paths and labels for one session; no arrays are opened.
    action : {"webapp", "power", "synchrony", "spike-phase"}
        Existing application or computation category to check.

    Returns
    -------
    ValidationResult
        Availability and an ordered tuple of human-readable missing field names.
    """
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"action must be one of {SUPPORTED_ACTIONS}")
    missing: list[str] = []
    _append_missing(
        missing,
        "behavior.session_directory",
        session.behavior.session_directory,
        "directory",
    )
    _append_missing(
        missing,
        "behavior.trial_table_file",
        session.behavior.trial_table_file,
        "file",
    )
    if not session.probes:
        missing.append("probes")

    if action in {"power", "synchrony", "spike-phase"}:
        if not session.sites:
            missing.append("sites")
        site_probe_ids = {site.probe_id for site in session.sites}
        for probe in session.probes:
            if probe.probe_id not in site_probe_ids:
                continue
            prefix = f"probes[{probe.probe_id}]"
            _append_missing(missing, f"{prefix}.lfp_file", probe.lfp_file, "file")
            _append_missing(
                missing, f"{prefix}.lfp_metadata_file", probe.lfp_metadata_file, "file"
            )
            _append_missing(
                missing,
                f"{prefix}.synchronization_file",
                probe.synchronization_file,
                "file",
            )

    if action in {"synchrony", "spike-phase"} and not session.site_pairs:
        missing.append("site_pairs")

    if action == "spike-phase":
        if not session.populations:
            missing.append("populations")
        population_probe_ids = {population.probe_id for population in session.populations}
        for probe in session.probes:
            if probe.probe_id not in population_probe_ids:
                continue
            prefix = f"probes[{probe.probe_id}]"
            _append_missing(
                missing, f"{prefix}.sorter_directory", probe.sorter_directory, "directory"
            )
            _append_missing(
                missing,
                f"{prefix}.aligned_spike_file",
                probe.aligned_spike_file,
                "file",
            )

    return ValidationResult(
        action=action,
        available=not missing,
        missing_inputs=tuple(dict.fromkeys(missing)),
    )
