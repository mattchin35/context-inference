"""Load the compact editable metadata for one neural recording session.

The JSON names existing sources. It does not load scientific arrays or search
the session directory for likely files.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = "2"
CANONICAL_FILENAME = "neural_session.json"
SUPPORTED_ACTIONS = ("webapp", "power", "synchrony", "spike-phase")
SUPPORTED_ACQUISITION_FAMILIES = ("open_ephys", "spikeglx")


@dataclass(frozen=True)
class BehaviorSources:
    """Session-relative behavior files; values are paths with no physical units."""

    trials: str
    events: str | None


@dataclass(frozen=True)
class SiteMetadata:
    """One site name and its zero-based saved-channel index."""

    site_id: str
    probe_id: str
    saved_channel_index: int

    @property
    def display_label(self) -> str:
        """Return the sole user-facing site name."""
        return self.site_id


@dataclass(frozen=True)
class ProbeSources:
    """All editable paths, sites, and optional unit channels for one probe."""

    probe_id: str
    acquisition_family: str
    lfp: str
    alignment: str
    sorter: str
    quality: str
    sites: tuple[SiteMetadata, ...]
    unit_channels: tuple[int, ...] | None

    @property
    def display_label(self) -> str:
        """Return the probe dictionary key as its only label."""
        return self.probe_id


@dataclass(frozen=True)
class ChannelGroupMetadata:
    """Derived compatibility view of one probe's optional channel restriction."""

    channel_group_id: str
    display_label: str
    probe_id: str
    channel_indices: tuple[int, ...]


@dataclass(frozen=True)
class PopulationMetadata:
    """Derived compatibility view: one implicit population per probe."""

    population_id: str
    display_label: str
    probe_id: str
    channel_group_id: str


@dataclass(frozen=True)
class SessionMetadata:
    """Structurally validated version-2 metadata with relative paths."""

    schema_version: str
    session: str
    acquisition: str
    behavior: BehaviorSources
    probes: tuple[ProbeSources, ...]
    site_pairs: tuple[tuple[str, str], ...]
    cache: str | None

    @property
    def sites(self) -> tuple[SiteMetadata, ...]:
        """Return sites flattened in probe and site declaration order."""
        return tuple(site for probe in self.probes for site in probe.sites)

    @property
    def channel_groups(self) -> tuple[ChannelGroupMetadata, ...]:
        """Return one derived group per probe for existing internal adapters."""
        return tuple(
            ChannelGroupMetadata(
                probe.probe_id,
                probe.probe_id,
                probe.probe_id,
                probe.unit_channels or (),
            )
            for probe in self.probes
        )

    @property
    def populations(self) -> tuple[PopulationMetadata, ...]:
        """Return one derived population per probe for existing internal adapters."""
        return tuple(
            PopulationMetadata(probe.probe_id, probe.probe_id, probe.probe_id, probe.probe_id)
            for probe in self.probes
        )


@dataclass(frozen=True)
class ResolvedBehaviorSources:
    """Absolute behavior paths for one resolved session."""

    trial_table_file: Path | None
    event_table_file: Path | None

    @property
    def session_directory(self) -> Path | None:
        """Return the trial-table parent used by existing behavior loaders."""
        return None if self.trial_table_file is None else self.trial_table_file.parent

    @property
    def treadmill_file(self) -> None:
        """Return no treadmill source; schema version 2 does not declare one."""
        return None


@dataclass(frozen=True)
class ResolvedProbeSources:
    """Absolute sources and inferred fixed children for one probe."""

    probe_id: str
    acquisition_family: str
    lfp_file: Path | None
    lfp_metadata_file: Path | None
    alignment_file: Path | None
    sorter_directory: Path | None
    channel_quality_file: Path | None
    spike_times_file: Path | None
    spike_clusters_file: Path | None
    cluster_info_file: Path | None
    sites: tuple[SiteMetadata, ...]
    unit_channels: tuple[int, ...] | None

    @property
    def display_label(self) -> str:
        """Return the probe key as its only display label."""
        return self.probe_id

    @property
    def synchronization_file(self) -> Path | None:
        """Return the shared alignment file for LFP synchronization."""
        return self.alignment_file

    @property
    def aligned_spike_file(self) -> Path | None:
        """Return the shared alignment file for aligned spikes."""
        return self.alignment_file


@dataclass(frozen=True)
class ResolvedSession:
    """Version-2 metadata with contained absolute paths."""

    metadata_path: Path
    session_root: Path
    schema_version: str
    session: str
    acquisition: str
    behavior: ResolvedBehaviorSources
    probes: tuple[ResolvedProbeSources, ...]
    site_pairs: tuple[tuple[str, str], ...]
    cache_directory: Path | None

    @property
    def session_id(self) -> str:
        """Return the single session identifier for existing analysis configs."""
        return self.session

    @property
    def subject_id(self) -> str:
        """Return the session identifier where older UI code expects a subject label."""
        return self.session

    @property
    def session_date(self) -> None:
        """Return no separately duplicated session date."""
        return None

    @property
    def session_label(self) -> str:
        """Return the sole session identifier as its label."""
        return self.session

    @property
    def sites(self) -> tuple[SiteMetadata, ...]:
        """Return sites flattened in probe declaration order."""
        return tuple(site for probe in self.probes for site in probe.sites)

    @property
    def channel_groups(self) -> tuple[ChannelGroupMetadata, ...]:
        """Return one derived channel group per probe."""
        return tuple(
            ChannelGroupMetadata(
                probe.probe_id,
                probe.probe_id,
                probe.probe_id,
                probe.unit_channels or (),
            )
            for probe in self.probes
        )

    @property
    def populations(self) -> tuple[PopulationMetadata, ...]:
        """Return one derived population per probe."""
        return tuple(
            PopulationMetadata(probe.probe_id, probe.probe_id, probe.probe_id, probe.probe_id)
            for probe in self.probes
        )

    @property
    def lfp_summary_cache_directory(self) -> Path | None:
        """Return the optional session-wide cache path."""
        return self.cache_directory

    @property
    def lfp_summary_snapshot_directory(self) -> None:
        """Return no separate snapshot path; version 2 removes that metadata."""
        return None


@dataclass(frozen=True)
class ValidationResult:
    """Availability of one action and its ordered missing inputs."""

    action: str
    available: bool
    missing_inputs: tuple[str, ...]


_SESSION_REQUIRED = frozenset(
    {"schema_version", "session", "acquisition", "behavior", "probes", "site_pairs"}
)
_SESSION_OPTIONAL = frozenset({"cache"})
_BEHAVIOR_REQUIRED = frozenset({"trials"})
_BEHAVIOR_OPTIONAL = frozenset({"events"})
_PROBE_REQUIRED = frozenset({"lfp", "alignment", "sorter", "quality", "sites"})
_PROBE_OPTIONAL = frozenset({"unit_channels"})


def _mapping(value: object, field: str) -> Mapping[str, object]:
    """Return a string-keyed JSON object."""
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _keys(
    value: Mapping[str, object], required: frozenset[str], optional: frozenset[str], field: str
) -> None:
    """Require documented keys and reject metadata bloat."""
    missing = sorted(required - value.keys())
    unexpected = sorted(value.keys() - required - optional)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError(f"{field} has {' and '.join(details)}")


def _string(value: object, field: str, *, allow_empty: bool = True) -> str:
    """Return a string, optionally requiring content."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    """Return a string or ``None``."""
    return None if value is None else _string(value, field)


def _sequence(value: object, field: str) -> Sequence[object]:
    """Return a JSON array."""
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    return value


def _channel(value: object, field: str) -> int:
    """Return one nonnegative zero-based channel index."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _parse_probe(probe_id: str, value: object, acquisition: str) -> ProbeSources:
    """Parse one probe-centered record."""
    field = f"probes[{probe_id}]"
    _string(probe_id, "probe ID", allow_empty=False)
    record = _mapping(value, field)
    _keys(record, _PROBE_REQUIRED, _PROBE_OPTIONAL, field)
    site_record = _mapping(record["sites"], f"{field}.sites")
    sites = tuple(
        SiteMetadata(
            _string(site_id, f"{field}.sites key", allow_empty=False),
            probe_id,
            _channel(channel, f"{field}.sites[{site_id}]"),
        )
        for site_id, channel in site_record.items()
    )
    raw_channels = record.get("unit_channels")
    unit_channels: tuple[int, ...] | None = None
    if raw_channels is not None:
        unit_channels = tuple(
            _channel(item, f"{field}.unit_channels[{index}]")
            for index, item in enumerate(_sequence(raw_channels, f"{field}.unit_channels"))
        )
        if len(set(unit_channels)) != len(unit_channels):
            raise ValueError(f"{field}.unit_channels must not contain duplicates")
    return ProbeSources(
        probe_id=probe_id,
        acquisition_family=acquisition,
        lfp=_string(record["lfp"], f"{field}.lfp"),
        alignment=_string(record["alignment"], f"{field}.alignment"),
        sorter=_string(record["sorter"], f"{field}.sorter"),
        quality=_string(record["quality"], f"{field}.quality"),
        sites=sites,
        unit_channels=unit_channels,
    )


def load_session_metadata(path: Path | str) -> SessionMetadata:
    """Load version-2 JSON without resolving paths or opening arrays.

    Parameters
    ----------
    path : pathlib.Path or str
        Metadata JSON path.

    Returns
    -------
    SessionMetadata
        Frozen relative-path metadata in JSON declaration order.
    """
    metadata_path = Path(path)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read session metadata {metadata_path}: {error}") from error
    record = _mapping(payload, "session metadata")
    schema_version = _string(record.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}; version 2 replaces older metadata"
        )
    _keys(record, _SESSION_REQUIRED, _SESSION_OPTIONAL, "session metadata")
    acquisition = _string(record["acquisition"], "acquisition", allow_empty=False)
    if acquisition not in SUPPORTED_ACQUISITION_FAMILIES:
        raise ValueError(f"acquisition must be one of {SUPPORTED_ACQUISITION_FAMILIES}")
    behavior_record = _mapping(record["behavior"], "behavior")
    _keys(behavior_record, _BEHAVIOR_REQUIRED, _BEHAVIOR_OPTIONAL, "behavior")
    probe_record = _mapping(record["probes"], "probes")
    probes = tuple(
        _parse_probe(probe_id, probe, acquisition) for probe_id, probe in probe_record.items()
    )
    site_ids = [site.site_id for probe in probes for site in probe.sites]
    if len(site_ids) != len(set(site_ids)):
        raise ValueError("site names must be unique across probes")
    raw_pairs = _sequence(record["site_pairs"], "site_pairs")
    pairs: list[tuple[str, str]] = []
    for index, raw_pair in enumerate(raw_pairs):
        pair = _sequence(raw_pair, f"site_pairs[{index}]")
        if len(pair) != 2:
            raise ValueError(f"site_pairs[{index}] must contain exactly two site names")
        first = _string(pair[0], f"site_pairs[{index}][0]", allow_empty=False)
        second = _string(pair[1], f"site_pairs[{index}][1]", allow_empty=False)
        if first == second or first not in site_ids or second not in site_ids:
            raise ValueError(f"site_pairs[{index}] must reference two declared sites")
        pairs.append((first, second))
    return SessionMetadata(
        schema_version=schema_version,
        session=_string(record["session"], "session"),
        acquisition=acquisition,
        behavior=BehaviorSources(
            trials=_string(behavior_record["trials"], "behavior.trials"),
            events=_optional_string(behavior_record.get("events"), "behavior.events"),
        ),
        probes=probes,
        site_pairs=tuple(pairs),
        cache=_optional_string(record.get("cache"), "cache"),
    )


def _resolve_path(
    session_root: Path, value: str | None, field: str, *, expected_kind: str
) -> Path | None:
    """Resolve one optional path below the session root."""
    if value is None or value == "":
        return None
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be relative to the session root")
    resolved = (session_root / relative).resolve(strict=False)
    if not resolved.is_relative_to(session_root):
        raise ValueError(f"{field} must remain contained within the session root")
    if resolved.exists():
        if expected_kind == "file" and not resolved.is_file():
            raise ValueError(f"{field} must identify a file")
        if expected_kind == "directory" and not resolved.is_dir():
            raise ValueError(f"{field} must identify a directory")
    return resolved


def resolve_session_metadata(
    metadata: SessionMetadata, metadata_path: Path | str
) -> ResolvedSession:
    """Resolve all editable relative paths below one session root.

    Parameters
    ----------
    metadata : SessionMetadata
        Parsed version-2 metadata.
    metadata_path : pathlib.Path or str
        Canonical ``neural_session.json`` path.

    Returns
    -------
    ResolvedSession
        Absolute contained paths; absent optional paths remain ``None``.
    """
    canonical = Path(metadata_path)
    if canonical.name != CANONICAL_FILENAME:
        raise ValueError(f"metadata filename must be {CANONICAL_FILENAME}")
    session_root = canonical.parent.resolve(strict=True)
    probes = []
    for probe in metadata.probes:
        prefix = f"probes[{probe.probe_id}]"
        lfp = _resolve_path(session_root, probe.lfp, f"{prefix}.lfp", expected_kind="file")
        if lfp is None:
            lfp_metadata = None
        elif metadata.acquisition == "open_ephys":
            lfp_metadata = lfp.parent / "lfp_preprocessing.json"
        else:
            lfp_metadata = lfp.with_suffix(".meta")
        sorter = _resolve_path(
            session_root, probe.sorter, f"{prefix}.sorter", expected_kind="directory"
        )
        probes.append(
            ResolvedProbeSources(
                probe_id=probe.probe_id,
                acquisition_family=metadata.acquisition,
                lfp_file=lfp,
                lfp_metadata_file=lfp_metadata,
                alignment_file=_resolve_path(
                    session_root, probe.alignment, f"{prefix}.alignment", expected_kind="file"
                ),
                sorter_directory=sorter,
                channel_quality_file=_resolve_path(
                    session_root, probe.quality, f"{prefix}.quality", expected_kind="file"
                ),
                spike_times_file=sorter / "spike_times.npy" if sorter else None,
                spike_clusters_file=sorter / "spike_clusters.npy" if sorter else None,
                cluster_info_file=sorter / "cluster_info.tsv" if sorter else None,
                sites=probe.sites,
                unit_channels=probe.unit_channels,
            )
        )
    return ResolvedSession(
        metadata_path=canonical.resolve(strict=True),
        session_root=session_root,
        schema_version=metadata.schema_version,
        session=metadata.session,
        acquisition=metadata.acquisition,
        behavior=ResolvedBehaviorSources(
            trial_table_file=_resolve_path(
                session_root, metadata.behavior.trials, "behavior.trials", expected_kind="file"
            ),
            event_table_file=_resolve_path(
                session_root, metadata.behavior.events, "behavior.events", expected_kind="file"
            ),
        ),
        probes=tuple(probes),
        site_pairs=metadata.site_pairs,
        cache_directory=_resolve_path(
            session_root, metadata.cache, "cache", expected_kind="directory"
        ),
    )


def resolve_probe_sources(session: ResolvedSession, probe_id: str) -> ResolvedProbeSources:
    """Return sources for one exact probe dictionary key."""
    for probe in session.probes:
        if probe.probe_id == probe_id:
            return probe
    raise KeyError(f"unknown probe ID: {probe_id}")


def _append_missing(
    missing: list[str], label: str, path: Path | None, expected_kind: str
) -> None:
    """Append a label when a required path is absent or has the wrong kind."""
    if path is None or (expected_kind == "file" and not path.is_file()) or (
        expected_kind == "directory" and not path.is_dir()
    ):
        missing.append(label)


def validate_session_for_action(session: ResolvedSession, action: str) -> ValidationResult:
    """Report missing ordinary inputs without opening scientific arrays.

    Parameters
    ----------
    session : ResolvedSession
        Resolved paths for one session.
    action : {"webapp", "power", "synchrony", "spike-phase"}
        Existing workflow category.

    Returns
    -------
    ValidationResult
        Availability plus ordered human-readable missing fields.
    """
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"action must be one of {SUPPORTED_ACTIONS}")
    missing: list[str] = []
    _append_missing(missing, "behavior.trials", session.behavior.trial_table_file, "file")
    if not session.probes:
        missing.append("probes")
    if action in {"power", "synchrony", "spike-phase"}:
        if not session.sites:
            missing.append("sites")
        for probe in session.probes:
            if not probe.sites:
                continue
            prefix = f"probes[{probe.probe_id}]"
            _append_missing(missing, f"{prefix}.lfp", probe.lfp_file, "file")
            _append_missing(missing, f"{prefix}.lfp_metadata", probe.lfp_metadata_file, "file")
            _append_missing(missing, f"{prefix}.alignment", probe.alignment_file, "file")
    if action in {"synchrony", "spike-phase"} and not session.site_pairs:
        missing.append("site_pairs")
    if action == "spike-phase":
        for probe in session.probes:
            prefix = f"probes[{probe.probe_id}]"
            _append_missing(missing, f"{prefix}.sorter", probe.sorter_directory, "directory")
            _append_missing(missing, f"{prefix}.quality", probe.channel_quality_file, "file")
            _append_missing(missing, f"{prefix}.spike_times", probe.spike_times_file, "file")
            _append_missing(missing, f"{prefix}.spike_clusters", probe.spike_clusters_file, "file")
            _append_missing(missing, f"{prefix}.cluster_info", probe.cluster_info_file, "file")
    return ValidationResult(action, not missing, tuple(dict.fromkeys(missing)))
