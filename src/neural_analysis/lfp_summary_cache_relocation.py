"""Safely relocate completed Power/Synchrony cache bytes between session roots.

This command deliberately does not load numerical component arrays or recompute
analysis results.  It establishes equivalence of the configured source files,
copies the two completed NPZ archives as byte streams, and publishes a
cluster-bound compatibility manifest plus an external provenance receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import socket
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from src.neural_analysis.lfp_summary import models as lfp_summary_models
from src.neural_analysis.lfp_summary.cache import (
    assess_component_status,
    rebind_power_synchrony_manifest,
)
from src.neural_analysis.lfp_summary.models import (
    LFPSummaryConfig,
    canonical_config_json,
    component_fingerprint,
    fingerprint_source_files,
    lfp_summary_config_from_json,
    source_value_semantics,
)


_COPIED_COMPONENTS = ("power", "synchrony")
_FINAL_MEMBER_NAMES = frozenset({"manifest.json", "power.npz", "synchrony.npz"})
_NON_COMPONENT_CONFIGURATION_FIELDS = frozenset(
    {"output_directory", "unit_population", "ppc", "ppc_execution", "random_seed"}
)
_COPY_CHUNK_BYTES = 1024 * 1024
_MANIFEST_JSON_MAX_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CacheRelocationRequest:
    """Immutable paths, configurations, and provenance required for relocation.

    Parameters
    ----------
    source_cache_directory, destination_cache_directory : pathlib.Path
        Existing producer cache and absent final destination cache. Both must be
        direct nonsymlink children of their corresponding ``processed`` directory.
    source_config, destination_config : LFPSummaryConfig
        Producer and active destination configurations. Power/Synchrony source
        paths must differ only by the declared root substitution; no component
        arrays, physical units, or scientific parameters are changed.
    local_session_root, cluster_session_root : pathlib.Path
        Existing nonsymlink session roots for the single allowed path mapping.
    receipt_path : pathlib.Path
        Absent nonsymlink external JSON receipt path. It must not resolve into a
        source, destination, or protected legacy cache.
    command : tuple[str, ...]
        Exact launcher command retained as provenance, with no interpretation.
    git_commit : str
        Exact source revision identifier retained as provenance.
    """

    source_cache_directory: Path
    destination_cache_directory: Path
    source_config: LFPSummaryConfig
    destination_config: LFPSummaryConfig
    local_session_root: Path
    cluster_session_root: Path
    receipt_path: Path
    command: tuple[str, ...]
    git_commit: str


@dataclass(frozen=True)
class RelocationRuntime:
    """Injectable wall-clock, host, and resource providers for a relocation.

    All callables return scalar host/runtime metadata only. They never read LFP,
    synchrony, trial, or NPZ numerical values.

    Parameters
    ----------
    utc_now : Callable[[], datetime]
        Current timezone-aware UTC timestamp.
    hostname : Callable[[], str]
        Current executing hostname.
    monotonic_seconds : Callable[[], float]
        Monotonic seconds used only to report wall time.
    peak_rss_bytes : Callable[[], int]
        Current process peak resident bytes.
    """

    utc_now: Callable[[], datetime]
    hostname: Callable[[], str]
    monotonic_seconds: Callable[[], float]
    peak_rss_bytes: Callable[[], int]


@dataclass(frozen=True)
class RelocationResult:
    """Published cache and receipt paths returned after successful relocation.

    Both paths identify files/directories containing unchanged component array
    bytes and JSON provenance; the result carries no numerical data arrays.
    """

    destination_cache_directory: Path
    receipt_path: Path


@dataclass(frozen=True)
class _FilesystemIdentity:
    """Stable metadata used to detect a concurrent filesystem replacement.

    Parameters
    ----------
    device, inode : int
        Filesystem device and inode identity returned by ``stat``.
    size_bytes : int
        Current regular-file size in bytes. It is host metadata, not an LFP
        sample count, array shape, or physical-unit quantity.
    modified_ns, changed_ns : int
        Nanosecond modification and inode-change timestamps. Together with
        device/inode they detect same-name replacement or in-place mutation
        without streaming a second component hash during normal success.
    """

    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int


@dataclass(frozen=True)
class _PublishedDestinationIdentity:
    """Stable staging ownership expected after one atomic directory rename.

    Parameters
    ----------
    directory_device, directory_inode : int
        Device/inode identity of the staging directory. Directory modification
        and inode-change timestamps are intentionally excluded because a rename
        may update them while retaining the same directory ownership.
    members : tuple[tuple[str, _FilesystemIdentity], ...]
        Sorted exact file identities for ``manifest.json``, ``power.npz``, and
        ``synchrony.npz``. No numerical arrays, units, axes, or missing values
        are read or rehashed when ownership is captured or compared.
    """

    directory_device: int
    directory_inode: int
    members: tuple[tuple[str, _FilesystemIdentity], ...]


def _default_runtime() -> RelocationRuntime:
    """Return host-runtime providers without touching experiment data.

    Returns
    -------
    RelocationRuntime
        UTC, hostname, monotonic-clock, and Linux/macOS ``ru_maxrss`` providers.
        Linux reports KiB and macOS reports bytes, so the provider normalizes the
        documented result to bytes.
    """

    def peak_rss_bytes() -> int:
        """Return the current process peak resident set size in bytes."""
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value * 1024) if os.name == "posix" and "linux" in os.sys.platform else int(value)

    return RelocationRuntime(
        utc_now=lambda: datetime.now(timezone.utc),
        hostname=socket.gethostname,
        monotonic_seconds=time.monotonic,
        peak_rss_bytes=peak_rss_bytes,
    )


def _absolute_path(path: Path) -> Path:
    """Return an absolute lexical path without resolving possible symlinks.

    Parameters
    ----------
    path : pathlib.Path
        Candidate local filesystem path.

    Returns
    -------
    pathlib.Path
        Absolute path retaining each lexical component so symlink checks can
        reject aliases before canonical resolution.
    """
    return path.absolute()


def _has_symlink_component(path: Path) -> bool:
    """Return whether any existing lexical component of ``path`` is a symlink.

    Parameters
    ----------
    path : pathlib.Path
        Existing or prospective path. Missing trailing components are allowed;
        every existing parent through the filesystem root is examined.

    Returns
    -------
    bool
        ``True`` when the path itself or any existing parent is a symlink.
    """
    absolute = _absolute_path(path)
    return any(part.is_symlink() for part in (absolute, *absolute.parents))


def _require_nonsymlink(path: Path, label: str) -> Path:
    """Reject a path that has a symlink component and return its resolved form.

    Parameters
    ----------
    path : pathlib.Path
        Existing or prospective filesystem path.
    label : str
        Human-readable source/destination/receipt context for error messages.

    Returns
    -------
    pathlib.Path
        Fully resolved canonical path. Because every lexical component is
        nonsymlink, resolution cannot silently change containment.

    Raises
    ------
    ValueError
        If the path itself or any existing parent is a symlink.
    """
    if _has_symlink_component(path):
        raise ValueError(f"{label} must not contain a symlink")
    return _absolute_path(path).resolve()


def _require_direct_processed_child(path: Path, root: Path, label: str) -> Path:
    """Validate an absent/existing cache path is one direct child of ``processed``.

    Parameters
    ----------
    path : pathlib.Path
        Cache directory candidate with no component array contents inspected.
    root : pathlib.Path
        Canonical session root containing an existing nonsymlink ``processed``
        directory.
    label : str
        Error-message context.

    Returns
    -------
    pathlib.Path
        Canonical lexical cache path. It is not created by this validator.
    """
    checked = _require_nonsymlink(path, label)
    processed = _require_nonsymlink(root / "processed", f"{label} processed parent")
    if not processed.is_dir():
        raise ValueError(f"{label} processed parent must exist")
    if checked.parent != processed or checked.name in {"", ".", ".."}:
        raise ValueError(f"{label} must be a direct child of the session processed directory")
    return checked


def _require_path_within(path: Path, root: Path, label: str) -> Path:
    """Require one nonsymlink configured source path to be contained by a root.

    Parameters
    ----------
    path : pathlib.Path
        Existing source path such as LFP, aligned synchronization, or trial table.
    root : pathlib.Path
        Canonical session root that owns the source path.
    label : str
        Human-readable source role.

    Returns
    -------
    pathlib.Path
        Canonical path guaranteed to be below ``root`` with no symlink alias.
    """
    checked = _require_nonsymlink(path, label)
    try:
        checked.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must remain inside its declared session root") from error
    return checked


def _require_session_root(config: LFPSummaryConfig, root: Path, label: str) -> Path:
    """Require a configuration session path to equal its declared canonical root.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable producer or destination configuration.
    root : pathlib.Path
        Declared local or cluster session root.
    label : str
        Context used in errors.

    Returns
    -------
    pathlib.Path
        Canonical declared root with no symlink components.
    """
    checked_root = _require_nonsymlink(root, f"{label} session root")
    if not checked_root.is_dir():
        raise ValueError(f"{label} session root must exist")
    configured_root = _require_nonsymlink(config.session_path, f"{label} session path")
    if configured_root != checked_root:
        raise ValueError(f"{label} configuration session path differs from declared root")
    return checked_root


def _mapped_path(source_path: Path, local_root: Path, cluster_root: Path, label: str) -> Path:
    """Map one validated local source path under the sole declared root mapping.

    Parameters
    ----------
    source_path : pathlib.Path
        Local LFP, synchronization, trial-table, or preprocessing-sidecar path.
    local_root, cluster_root : pathlib.Path
        Canonical nonsymlink session roots.
    label : str
        Source-role context for errors.

    Returns
    -------
    pathlib.Path
        Expected canonical cluster path with the identical relative suffix.
    """
    local_path = _require_path_within(source_path, local_root, f"local {label}")
    return cluster_root / local_path.relative_to(local_root)


def _validate_single_root_mapping(request: CacheRelocationRequest) -> tuple[Path, Path]:
    """Prove all Power/Synchrony configured paths use exactly one root substitution.

    Parameters
    ----------
    request : CacheRelocationRequest
        Producer/destination configurations and declared roots. Unit populations,
        PPC settings, and top-level random seeds are intentionally not inputs to
        copied Power/Synchrony components.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path]
        Canonical local and cluster session roots.

    Raises
    ------
    ValueError
        If session roots, LFP paths, aligned sync paths, trial table paths, or
        source parent containment differ from the one declared mapping.
    """
    local_root = _require_session_root(request.source_config, request.local_session_root, "local")
    cluster_root = _require_session_root(
        request.destination_config, request.cluster_session_root, "cluster"
    )
    if len(request.source_config.sites) != len(request.destination_config.sites):
        raise ValueError("source and cluster configurations have different site mappings")
    for index, (local_site, cluster_site) in enumerate(
        zip(request.source_config.sites, request.destination_config.sites, strict=True)
    ):
        expected_lfp = _mapped_path(local_site.lfp_path, local_root, cluster_root, f"LFP site {index}")
        actual_lfp = _require_path_within(cluster_site.lfp_path, cluster_root, f"cluster LFP site {index}")
        if actual_lfp != expected_lfp:
            raise ValueError("LFP source paths differ from the declared root mapping")
        if (local_site.aligned_sync_path is None) != (cluster_site.aligned_sync_path is None):
            raise ValueError("aligned synchronization source paths differ from the declared mapping")
        if local_site.aligned_sync_path is not None and cluster_site.aligned_sync_path is not None:
            expected_sync = _mapped_path(
                local_site.aligned_sync_path, local_root, cluster_root, f"sync site {index}"
            )
            actual_sync = _require_path_within(
                cluster_site.aligned_sync_path, cluster_root, f"cluster sync site {index}"
            )
            if actual_sync != expected_sync:
                raise ValueError("aligned synchronization paths differ from the declared root mapping")
    if (request.source_config.trial_table_path is None) != (
        request.destination_config.trial_table_path is None
    ):
        raise ValueError("trial-table paths differ from the declared root mapping")
    if request.source_config.trial_table_path is not None:
        expected_trial_table = _mapped_path(
            request.source_config.trial_table_path, local_root, cluster_root, "trial table"
        )
        actual_trial_table = _require_path_within(
            request.destination_config.trial_table_path, cluster_root, "cluster trial table"
        )
        if actual_trial_table != expected_trial_table:
            raise ValueError("trial-table paths differ from the declared root mapping")
    return local_root, cluster_root


def _require_regular_source_file(path: Path, root: Path, label: str) -> Path:
    """Require one configured or derived source to be a contained regular file.

    Parameters
    ----------
    path : pathlib.Path
        Configured LFP, synchronization, trial-table, or derived Open Ephys
        preprocessing-sidecar path. Numerical file contents are not read.
    root : pathlib.Path
        Canonical owning session root.
    label : str
        Human-readable source role for error messages.

    Returns
    -------
    pathlib.Path
        Resolved contained regular file with no symlink in its lexical path.

    Raises
    ------
    ValueError
        If the file or any parent is a symlink, it falls outside ``root``, or it
        is absent or not a regular file. This validation occurs before any
        fingerprint or content hash is calculated.
    """
    checked = _require_path_within(path, root, label)
    if checked.is_symlink() or not checked.is_file():
        raise ValueError(f"{label} must be an existing regular nonsymlink file")
    return checked


def _validate_regular_source_pair(
    local_path: Path,
    cluster_path: Path,
    local_root: Path,
    cluster_root: Path,
    label: str,
) -> None:
    """Validate one local/cluster source pair before fingerprinting its bytes.

    Parameters
    ----------
    local_path, cluster_path : pathlib.Path
        Corresponding configured or derived source paths. Both must identify
        regular nonsymlink files; no source values, units, or array axes are read.
    local_root, cluster_root : pathlib.Path
        Canonical roots defining the only allowed path substitution.
    label : str
        Human-readable LFP, synchronization, trial-table, or sidecar role.

    Returns
    -------
    None
        Confirms exact relative path mapping and regular-file containment on both
        hosts before later metadata fingerprints and streamed SHA-256 passes.
    """
    local_checked = _require_regular_source_file(local_path, local_root, f"local {label}")
    expected_cluster = _mapped_path(local_checked, local_root, cluster_root, label)
    cluster_checked = _require_regular_source_file(
        cluster_path, cluster_root, f"cluster {label}"
    )
    if cluster_checked != expected_cluster:
        raise ValueError(f"{label} differs from the declared root mapping")


def _validate_configured_source_files(
    request: CacheRelocationRequest,
    local_root: Path,
    cluster_root: Path,
) -> None:
    """Validate all Power/Synchrony inputs and Open Ephys sidecars before hashes.

    Parameters
    ----------
    request : CacheRelocationRequest
        Local and cluster configurations whose explicit paths have already passed
        the one-root mapping check.
    local_root, cluster_root : pathlib.Path
        Canonical roots that contain every relevant LFP, synchronization,
        trial-table, and derived preprocessing-sidecar file.

    Returns
    -------
    None
        Every unique configured source and every Open Ephys
        ``lfp_preprocessing.json`` sidecar is proven to be a contained regular
        nonsymlink file on both roots before ``fingerprint_source_files`` can
        stat or stream any byte.
    """
    local_config = request.source_config
    cluster_config = request.destination_config
    for index, (local_site, cluster_site) in enumerate(
        zip(local_config.sites, cluster_config.sites, strict=True)
    ):
        _validate_regular_source_pair(
            local_site.lfp_path,
            cluster_site.lfp_path,
            local_root,
            cluster_root,
            f"LFP site {index}",
        )
        if local_site.aligned_sync_path is not None and cluster_site.aligned_sync_path is not None:
            _validate_regular_source_pair(
                local_site.aligned_sync_path,
                cluster_site.aligned_sync_path,
                local_root,
                cluster_root,
                f"sync site {index}",
            )
        if local_site.acquisition_format == "open_ephys":
            _validate_regular_source_pair(
                local_site.lfp_path.parent / "lfp_preprocessing.json",
                cluster_site.lfp_path.parent / "lfp_preprocessing.json",
                local_root,
                cluster_root,
                f"Open Ephys sidecar site {index}",
            )
    if local_config.trial_table_path is not None and cluster_config.trial_table_path is not None:
        _validate_regular_source_pair(
            local_config.trial_table_path,
            cluster_config.trial_table_path,
            local_root,
            cluster_root,
            "trial table",
        )


def _mapped_configuration_path(
    value: Any,
    local_root: Path,
    cluster_root: Path,
    label: str,
) -> str | None:
    """Map one declared configuration path and reject every other value type.

    Parameters
    ----------
    value : object
        JSON path value from one documented path field, or ``None`` for an
        optional path. Labels, stable identifiers, and other JSON strings are
        deliberately not accepted by this helper.
    local_root, cluster_root : pathlib.Path
        Canonical roots for the one permitted local-to-cluster substitution.
    label : str
        Configuration-field context used in validation errors.

    Returns
    -------
    str or None
        Cluster path with the same suffix below ``cluster_root``. No numerical
        arrays or source bytes are opened.

    Raises
    ------
    ValueError
        If the documented path field has a non-string value or falls outside the
        declared local root.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string path or null")
    return str(_mapped_path(Path(value), local_root, cluster_root, label))


def _normalize_component_configuration_paths(
    source_payload: dict[str, Any],
    local_root: Path,
    cluster_root: Path,
) -> dict[str, Any]:
    """Normalize only declared Power/Synchrony path fields in JSON metadata.

    Parameters
    ----------
    source_payload : dict[str, object]
        Canonical source configuration after fields irrelevant to copied
        Power/Synchrony components have been removed. All non-path strings,
        including site ``label`` and ``stable_id``, are retained verbatim.
    local_root, cluster_root : pathlib.Path
        Canonical roots defining the one allowed path substitution.

    Returns
    -------
    dict[str, object]
        A deep JSON copy whose ``session_path``, site ``lfp_path``, optional
        site ``aligned_sync_path``, and optional ``trial_table_path`` are mapped
        to the destination root. Scientific values, units, labels, and site IDs
        remain unchanged.

    Raises
    ------
    ValueError
        If the canonical configuration shape or a declared path field is invalid.
    """
    normalized = json.loads(json.dumps(source_payload, allow_nan=False))
    normalized["session_path"] = _mapped_configuration_path(
        normalized.get("session_path"), local_root, cluster_root, "session path"
    )
    sites = normalized.get("sites")
    if not isinstance(sites, list):
        raise ValueError("configuration sites must be a list")
    for index, site in enumerate(sites):
        if not isinstance(site, dict):
            raise ValueError("configuration site must be a mapping")
        site["lfp_path"] = _mapped_configuration_path(
            site.get("lfp_path"), local_root, cluster_root, f"LFP site {index}"
        )
        site["aligned_sync_path"] = _mapped_configuration_path(
            site.get("aligned_sync_path"), local_root, cluster_root, f"sync site {index}"
        )
    normalized["trial_table_path"] = _mapped_configuration_path(
        normalized.get("trial_table_path"), local_root, cluster_root, "trial table"
    )
    return normalized


def _validate_scientific_equivalence(
    request: CacheRelocationRequest,
    local_root: Path,
    cluster_root: Path,
) -> None:
    """Reject a non-path Power/Synchrony configuration difference.

    Parameters
    ----------
    request : CacheRelocationRequest
        Validated local/cluster configurations.
    local_root, cluster_root : pathlib.Path
        Canonical roots already proven to map each configured source path.

    Returns
    -------
    None
        Only output location, unit population, PPC settings, PPC execution, and
        top-level random seed are excluded because they do not enter Power or
        Synchrony component fingerprints. All physical units, axis-affecting
        configuration, filters, windows, sites, pairs, and analysis parameters
        remain equality requirements.
    """
    source_payload = json.loads(canonical_config_json(request.source_config))
    destination_payload = json.loads(canonical_config_json(request.destination_config))
    for field in _NON_COMPONENT_CONFIGURATION_FIELDS:
        source_payload.pop(field, None)
        destination_payload.pop(field, None)
    mapped_source = _normalize_component_configuration_paths(
        source_payload, local_root, cluster_root
    )
    if mapped_source != destination_payload:
        raise ValueError("source and destination scientific configurations differ")


def _validate_request_paths(
    request: CacheRelocationRequest,
    local_root: Path,
    cluster_root: Path,
) -> tuple[Path, Path, Path]:
    """Validate cache/receipt containment without creating destination artifacts.

    Parameters
    ----------
    request : CacheRelocationRequest
        Requested source, destination, and external receipt locations.
    local_root, cluster_root : pathlib.Path
        Canonical session roots established by the mapping validator.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Canonical source cache, absent destination cache, and absent external
        receipt path. No directories or files are created.
    """
    source_cache = _require_direct_processed_child(
        request.source_cache_directory, local_root, "source cache"
    )
    destination_cache = _require_direct_processed_child(
        request.destination_cache_directory, cluster_root, "destination cache"
    )
    configured_source_cache = _require_nonsymlink(
        request.source_config.output_directory, "source output directory"
    )
    configured_destination_cache = _require_nonsymlink(
        request.destination_config.output_directory, "destination output directory"
    )
    if configured_source_cache != source_cache:
        raise ValueError("source cache differs from the producer output directory")
    if configured_destination_cache != destination_cache:
        raise ValueError("destination cache differs from the active output directory")
    if source_cache.name == "lfp_summary_cache" or destination_cache.name == "lfp_summary_cache":
        raise ValueError("protected legacy lfp_summary_cache cannot be relocated")
    if not source_cache.is_dir():
        raise ValueError("source cache must exist as a regular directory")
    if source_cache.is_symlink():
        raise ValueError("source cache must not be a symlink")
    if destination_cache.exists() or destination_cache.is_symlink():
        raise FileExistsError("destination cache must be absent and nonsymlink")

    receipt_path = _require_nonsymlink(request.receipt_path, "receipt path")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("receipt path must be absent and nonsymlink")
    protected_paths = (
        source_cache,
        destination_cache,
        _require_nonsymlink(cluster_root / "processed" / "lfp_summary_cache", "legacy cache"),
    )
    for protected_path in protected_paths:
        try:
            receipt_path.relative_to(protected_path)
        except ValueError:
            continue
        raise ValueError("receipt path must be external to every cache")
    return source_cache, destination_cache, receipt_path


def _read_bounded_ascii_json_snapshot(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    """Read one bounded ASCII JSON mapping and retain its exact source bytes.

    Parameters
    ----------
    path : pathlib.Path
        Existing nonsymlink JSON file of at most 1 MiB. It contains metadata,
        not numerical component arrays.
    label : str
        Context used in validation errors.

    Returns
    -------
    tuple[dict[str, object], bytes]
        Decoded mapping plus the exact bounded ASCII bytes from the same open
        file snapshot. Callers can compute provenance digests without reopening
        a manifest that may have been atomically replaced.

    Raises
    ------
    ValueError
        If the path is unsafe, the mapping exceeds the metadata bound, or its
        bytes are not valid ASCII JSON. Numerical arrays and NPZ members are
        never opened by this helper.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    try:
        with path.open("rb") as handle:
            raw_bytes = handle.read(_MANIFEST_JSON_MAX_BYTES + 1)
        if len(raw_bytes) > _MANIFEST_JSON_MAX_BYTES:
            raise ValueError(f"{label} exceeds the metadata size limit")
        decoded = json.loads(raw_bytes.decode("ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} contains invalid ASCII JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must contain a JSON mapping")
    return decoded, raw_bytes


def _read_ascii_json(path: Path, label: str) -> dict[str, Any]:
    """Read one bounded ASCII JSON mapping without numerical arrays.

    Parameters
    ----------
    path : pathlib.Path
        Existing nonsymlink manifest or receipt JSON path of at most 1 MiB.
    label : str
        Context used in validation errors.

    Returns
    -------
    dict[str, object]
        Decoded mapping with Unicode producer metadata preserved semantically.
        The exact input bytes are intentionally discarded after validation.
    """
    decoded, _raw_bytes = _read_bounded_ascii_json_snapshot(path, label)
    return decoded


def _validate_source_cache_members(source_cache: Path) -> None:
    """Require exactly manifest, Power, and Synchrony regular source members.

    Parameters
    ----------
    source_cache : pathlib.Path
        Existing nonsymlink producer cache directory.

    Returns
    -------
    None
        Rejects work directories, Spike-phase members, symlinks, and every
        unexpected file before a destination staging directory is created.
    """
    members = list(source_cache.iterdir())
    names = {member.name for member in members}
    if names != _FINAL_MEMBER_NAMES:
        raise ValueError("source cache contains unexpected, Spike, or work members")
    if any(member.is_symlink() or not member.is_file() for member in members):
        raise ValueError("source cache members must be regular nonsymlink files")


def _validate_source_manifest(
    source_cache: Path,
    source_config: LFPSummaryConfig,
    source_fingerprints: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Prove a source manifest describes two publicly compatible components.

    Parameters
    ----------
    source_cache : pathlib.Path
        Existing producer cache containing only completed Power/Synchrony files.
    source_config : LFPSummaryConfig
        Producer configuration used to reconstruct expected compatibility identity.
    source_fingerprints : Mapping[str, Any]
        One precomputed Power/Synchrony source-fingerprint mapping. It is reused
        by public status checks so no sidecar receives a second content hash.

    Returns
    -------
    tuple[dict[str, Any], str]
        Original source manifest and a lowercase SHA-256 computed from the same
        bounded byte snapshot. The receipt therefore cannot combine decoded
        producer metadata from one manifest version with a digest from another.
    """
    manifest, manifest_bytes = _read_bounded_ascii_json_snapshot(
        source_cache / "manifest.json", "source manifest"
    )
    manifest_hash = sha256(manifest_bytes).hexdigest()
    expected_configuration = json.loads(canonical_config_json(source_config))
    if manifest.get("schema_version") != source_config.schema_version:
        raise ValueError("source manifest schema version differs from source configuration")
    if manifest.get("session_id") != source_config.session_id:
        raise ValueError("source manifest session differs from source configuration")
    if manifest.get("configuration") != expected_configuration:
        raise ValueError("source manifest configuration differs from source configuration")
    components = manifest.get("components")
    if not isinstance(components, Mapping) or set(components) != set(_COPIED_COMPONENTS):
        raise ValueError("source manifest must contain only complete Power and Synchrony components")
    expected_semantics = source_value_semantics(source_config)
    for component in _COPIED_COMPONENTS:
        entry = components.get(component)
        if not isinstance(entry, Mapping):
            raise ValueError(f"source manifest component {component} is invalid")
        if entry.get("file_name") != f"{component}.npz":
            raise ValueError(f"source manifest component filename is invalid: {component}")
        if entry.get("configuration_snapshot") != expected_configuration:
            raise ValueError(f"source manifest component snapshot is stale: {component}")
        if entry.get("configuration_fingerprint") != component_fingerprint(component, source_config):
            raise ValueError(f"source manifest component fingerprint is stale: {component}")
        if entry.get("source_fingerprints") != source_fingerprints:
            raise ValueError(f"source manifest component source fingerprints are stale: {component}")
        if entry.get("source_value_semantics") != expected_semantics:
            raise ValueError(f"source manifest component value semantics are stale: {component}")
        status = assess_component_status(
            source_cache,
            component,
            source_config,
            manifest,
            current_source_fingerprints=source_fingerprints,
            validate_headers_only=True,
        )
        if status.status != "compatible":
            raise ValueError(f"source component {component} is not compatible: {status.differences}")
    return manifest, manifest_hash


def _source_records_with_digests(
    source_fingerprints: Mapping[str, Mapping[str, Any]],
    destination_fingerprints: Mapping[str, Mapping[str, Any]],
    local_root: Path,
    cluster_root: Path,
) -> dict[str, dict[str, object]]:
    """Stream every unique Power/Synchrony input once per host and compare bytes.

    Parameters
    ----------
    source_fingerprints, destination_fingerprints : Mapping[str, Mapping[str, Any]]
        Existing component-scoped live fingerprint records. Open Ephys sidecar
        SHA-256 entries are reused because they were already streamed by the
        authoritative model fingerprint seam.
    local_root, cluster_root : pathlib.Path
        Canonical roots used for exact relative path mapping.

    Returns
    -------
    dict[str, dict[str, object]]
        Receipt-ready local-keyed records containing local/cluster paths,
        byte sizes, SHA-256 digests, and any fixed source-value semantics.

    Raises
    ------
    ValueError
        If a source is missing, maps to a different cluster path, has unequal
        byte size/digest, or lacks the required Open Ephys affine-uV semantics.
    """
    records: dict[str, dict[str, object]] = {}
    for local_key in sorted(source_fingerprints):
        local_fingerprint = source_fingerprints[local_key]
        if not isinstance(local_fingerprint, Mapping):
            raise ValueError("source fingerprint record is invalid")
        local_path = Path(local_key)
        cluster_path = _mapped_path(local_path, local_root, cluster_root, "source")
        cluster_fingerprint = destination_fingerprints.get(str(cluster_path))
        if not isinstance(cluster_fingerprint, Mapping):
            raise ValueError("cluster source fingerprint differs from declared mapping")
        local_size = local_fingerprint.get("size_bytes")
        cluster_size = cluster_fingerprint.get("size_bytes")
        if not isinstance(local_size, int) or local_size < 0 or not isinstance(cluster_size, int) or cluster_size < 0:
            raise ValueError("source size record is missing or invalid")
        local_digest = local_fingerprint.get("sha256")
        cluster_digest = cluster_fingerprint.get("sha256")
        if not isinstance(local_digest, str) or not local_digest:
            local_digest = lfp_summary_models.sha256_file_content(local_path)
        if not isinstance(cluster_digest, str) or not cluster_digest:
            cluster_digest = lfp_summary_models.sha256_file_content(cluster_path)
        if local_size != cluster_size or local_digest != cluster_digest:
            raise ValueError("source byte size or SHA-256 digest differs across roots")
        record: dict[str, object] = {
            "cluster_path": str(cluster_path),
            "local": {"size_bytes": local_size, "sha256": local_digest},
            "cluster": {"size_bytes": cluster_size, "sha256": cluster_digest},
        }
        value_semantics = local_fingerprint.get("value_semantics")
        if value_semantics is not None:
            if value_semantics != "open_ephys_affine_uV_v1":
                raise ValueError("source value semantics are not open_ephys_affine_uV_v1")
            if cluster_fingerprint.get("value_semantics") != value_semantics:
                raise ValueError("cluster source value semantics differ")
            record["value_semantics"] = value_semantics
        records[str(local_path)] = record
    return records


def _hash_file_record(path: Path) -> dict[str, object]:
    """Return one streamed byte-count/SHA-256 record for a regular file.

    Parameters
    ----------
    path : pathlib.Path
        Existing regular file. The content is streamed through the model's
        injectable SHA-256 seam; no ``Path.read_bytes`` call is used.

    Returns
    -------
    dict[str, object]
        ``size_bytes`` and lowercase SHA-256 digest for receipt provenance.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError("component file must be a regular nonsymlink file")
    return {
        "size_bytes": path.stat().st_size,
        "sha256": lfp_summary_models.sha256_file_content(path),
    }


def _copy_component_file(source_path: Path, destination_path: Path) -> None:
    """Copy one component byte stream without loading any NPZ member or array.

    Parameters
    ----------
    source_path, destination_path : pathlib.Path
        Regular source NPZ and new staging NPZ paths. Bytes, array shapes, axes,
        units, and NaN missingness are copied exactly in fixed-size chunks.

    Returns
    -------
    None
        Destination is fully written and closed; the caller hashes it before
        publication to prove byte identity.
    """
    with source_path.open("rb") as source_handle, destination_path.open("xb") as destination_handle:
        while chunk := source_handle.read(_COPY_CHUNK_BYTES):
            destination_handle.write(chunk)


def _write_ascii_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one JSON mapping as deterministic ASCII without numerical arrays.

    Parameters
    ----------
    path : pathlib.Path
        New file path whose parent already exists and is nonsymlink.
    payload : Mapping[str, Any]
        JSON-compatible manifest or receipt mapping. Non-ASCII producer strings
        are escaped so the bytes remain ASCII while JSON semantic values round-trip.

    Returns
    -------
    None
        Writes sorted-key UTF-8-compatible ASCII JSON with no NaN encoding.
    """
    with path.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _validate_git_commit(git_commit: str) -> None:
    """Require an exact lowercase forty-character Git revision before any I/O.

    Parameters
    ----------
    git_commit : str
        Producer revision recorded in the external receipt. It is provenance
        text only and cannot alter numerical arrays, units, or source paths.

    Returns
    -------
    None
        Confirms the value has exactly forty lowercase hexadecimal characters.

    Raises
    ------
    ValueError
        If provenance is empty, abbreviated, uppercase, or contains a non-hex
        character. Callers invoke this before fingerprinting, hashing, or staging.
    """
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError("git commit must be exactly 40 lowercase hexadecimal characters")


def _create_staging_directory(destination_cache: Path) -> Path:
    """Create one new nonsymlink direct sibling staging directory.

    Parameters
    ----------
    destination_cache : pathlib.Path
        Absent final cache path. Its existing parent is the target filesystem.

    Returns
    -------
    pathlib.Path
        New empty staging directory below the final destination parent.
    """
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination_cache.name}-stage-", dir=destination_cache.parent)
    )
    if staging.parent != destination_cache.parent or staging.is_symlink():
        raise ValueError("staging directory must be a nonsymlink destination sibling")
    return staging


def _publish_staged_cache(staging_directory: Path, destination_directory: Path) -> None:
    """Atomically publish a complete staging directory using ``os.replace``.

    Parameters
    ----------
    staging_directory : pathlib.Path
        Complete nonsymlink sibling containing exactly manifest, Power, and
        Synchrony files.
    destination_directory : pathlib.Path
        Absent final destination path on the same filesystem.

    Returns
    -------
    None
        A single directory rename exposes all component bytes and manifest together.
    """
    if staging_directory.parent != destination_directory.parent:
        raise ValueError("staging directory must be a destination sibling")
    if staging_directory.is_symlink() or destination_directory.exists() or destination_directory.is_symlink():
        raise ValueError("publication requires a regular staging directory and absent destination")
    if {member.name for member in staging_directory.iterdir()} != _FINAL_MEMBER_NAMES:
        raise ValueError("staging directory is incomplete")
    os.replace(staging_directory, destination_directory)


def _cleanup_created_path(path: Path | None) -> None:
    """Remove only a known private staging directory or temporary receipt file.

    Parameters
    ----------
    path : pathlib.Path or None
        Exact path allocated by this command. Public source, legacy, destination,
        and final receipt paths are never accepted by this cleanup helper.

    Returns
    -------
    None
        Best-effort removal of a private temporary artifact after an exception.
    """
    if path is None or path.is_symlink() or not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            if child.is_file() and not child.is_symlink():
                child.unlink()
        path.rmdir()
    elif path.is_file():
        path.unlink()


def _filesystem_identity(path: Path) -> _FilesystemIdentity:
    """Capture one path's non-content identity metadata without opening its bytes.

    Parameters
    ----------
    path : pathlib.Path
        Existing regular nonsymlink file or directory. No numerical component
        payload, physical unit, or array axis is read.

    Returns
    -------
    _FilesystemIdentity
        Device/inode plus size and nanosecond mutation metadata from one ``stat``
        record. This is used only to make rollback safe after publication.
    """
    metadata = path.stat()
    return _FilesystemIdentity(
        device=metadata.st_dev,
        inode=metadata.st_ino,
        size_bytes=metadata.st_size,
        modified_ns=metadata.st_mtime_ns,
        changed_ns=metadata.st_ctime_ns,
    )


def _capture_cache_ownership(
    cache_directory: Path,
) -> _PublishedDestinationIdentity:
    """Capture directory and member ownership without reading component bytes.

    Parameters
    ----------
    cache_directory : pathlib.Path
        Staging or published nonsymlink directory containing exactly the
        relocation manifest and two byte-preserved NPZ files. Numerical values
        are not opened, hashed, reshaped, or converted.

    Returns
    -------
    _PublishedDestinationIdentity
        Stable directory device/inode plus sorted exact identities for all three
        expected members. Directory mtime/ctime are not part of ownership because
        atomic publication may change them during rename.

    Raises
    ------
    ValueError
        If the staging or destination directory is a symlink or lacks the exact
        regular-member layout required for publication/rollback ownership checks.
    """
    if cache_directory.is_symlink() or not cache_directory.is_dir():
        raise ValueError("cache ownership directory must be a regular directory")
    members = tuple(cache_directory.iterdir())
    if (
        {member.name for member in members} != _FINAL_MEMBER_NAMES
        or any(member.is_symlink() or not member.is_file() for member in members)
    ):
        raise ValueError("cache ownership directory has unexpected members")
    directory_metadata = cache_directory.stat()
    return _PublishedDestinationIdentity(
        directory_device=directory_metadata.st_dev,
        directory_inode=directory_metadata.st_ino,
        members=tuple(
            (member.name, _filesystem_identity(member))
            for member in sorted(members, key=lambda item: item.name)
        ),
    )


def _require_cache_ownership(
    cache_directory: Path,
    expected_ownership: _PublishedDestinationIdentity,
    checkpoint: str,
) -> None:
    """Fail when a published cache no longer matches pre-rename staging ownership.

    Parameters
    ----------
    cache_directory : pathlib.Path
        Public destination directory checked without opening component bytes.
    expected_ownership : _PublishedDestinationIdentity
        Device/inode and exact-member metadata captured from the staging directory
        immediately before its atomic publication rename.
    checkpoint : str
        Human-readable publication or receipt-commit checkpoint in error text.

    Returns
    -------
    None
        Confirms the destination still has the staging directory identity and all
        original member identities. Directory timestamps are intentionally ignored
        because rename may change them without changing ownership.

    Raises
    ------
    ValueError
        If the public destination is missing, malformed, or changed by another
        actor. The caller then cleans private files and conditional rollback leaves
        the changed destination untouched.
    """
    try:
        current_ownership = _capture_cache_ownership(cache_directory)
    except (OSError, ValueError) as error:
        raise ValueError(f"published destination ownership changed {checkpoint}") from error
    if current_ownership != expected_ownership:
        raise ValueError(f"published destination ownership changed {checkpoint}")


def _rollback_published_destination(
    destination_cache: Path | None,
    published_identity: _PublishedDestinationIdentity | None,
) -> None:
    """Remove a published destination only when no actor changed any member.

    Parameters
    ----------
    destination_cache : pathlib.Path or None
        Exact final cache path after this command's successful staging rename.
        Source caches, legacy caches, receipts, and arbitrary directories are not
        accepted by this rollback helper.
    published_identity : _PublishedDestinationIdentity or None
        Directory/member ownership captured from staging immediately before this
        invocation's atomic publication. A mismatch means another actor changed
        or replaced at least one member, so the entire destination is preserved.

    Returns
    -------
    None
        Deletes only the unchanged nonsymlink directory containing exactly
        ``manifest.json``, ``power.npz``, and ``synchrony.npz``. Component bytes
        are not opened, hashed, or transformed. Any concurrent change preserves
        the whole destination untouched.
    """
    if destination_cache is None or published_identity is None:
        return
    try:
        current_identity = _capture_cache_ownership(destination_cache)
    except (OSError, ValueError):
        return
    if current_identity != published_identity:
        return
    members = tuple(destination_cache.iterdir())
    for member in members:
        member.unlink()
    destination_cache.rmdir()


def _utc_timestamp(value: datetime) -> str:
    """Return a parseable UTC ``Z`` timestamp without changing elapsed-time units.

    Parameters
    ----------
    value : datetime
        Timezone-aware or naive timestamp supplied by ``RelocationRuntime``.

    Returns
    -------
    str
        ISO-8601 UTC seconds timestamp ending in ``Z``.
    """
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return aware.isoformat(timespec="seconds").replace("+00:00", "Z")


def _receipt_payload(
    request: CacheRelocationRequest,
    local_root: Path,
    cluster_root: Path,
    source_cache: Path,
    destination_cache: Path,
    source_manifest: Mapping[str, Any],
    source_manifest_hash: str,
    rebound_manifest_hash: str,
    component_hashes: Mapping[str, Mapping[str, object]],
    source_hashes: Mapping[str, Mapping[str, object]],
    component_states: Mapping[str, str],
    runtime: RelocationRuntime,
    started_at_utc: str,
    completed_at_utc: str,
    wall_seconds: float,
) -> dict[str, object]:
    """Assemble deterministic external provenance for an already-validated copy.

    Parameters
    ----------
    request : CacheRelocationRequest
        Exact request command and Git provenance.
    local_root, cluster_root, source_cache, destination_cache : pathlib.Path
        Canonical published paths.
    source_manifest : Mapping[str, Any]
        Original producer metadata retained semantically, never presented as a
        cluster producer record.
    source_manifest_hash, rebound_manifest_hash : str
        SHA-256 digests of the bounded source-manifest byte snapshot and emitted
        staging manifest bytes, respectively. The source digest is not obtained
        by reopening the producer manifest after it has been decoded.
    component_hashes, source_hashes : Mapping[str, Mapping[str, object]]
        Local/destination and local/cluster byte records respectively.
    component_states : Mapping[str, str]
        Final public Power/Synchrony/Spike-phase status labels.
    runtime : RelocationRuntime
        Host/RSS provider only.
    started_at_utc, completed_at_utc : str
        Ordered parseable UTC timestamps.
    wall_seconds : float
        Monotonic elapsed seconds, not scientific timing data.

    Returns
    -------
    dict[str, object]
        JSON-ready ASCII-serializable receipt containing no numerical arrays.
    """
    return {
        "source_producer_manifest": source_manifest,
        "source_producer_manifest_sha256": source_manifest_hash,
        "rebound_manifest_sha256": rebound_manifest_hash,
        "component_hashes": component_hashes,
        "source_hashes": source_hashes,
        "command": list(request.command),
        "git_commit": request.git_commit,
        "local_session_root": str(local_root),
        "cluster_session_root": str(cluster_root),
        "source_cache_directory": str(source_cache),
        "destination_cache_directory": str(destination_cache),
        "component_states": dict(component_states),
        "hostname": runtime.hostname(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "wall_seconds": wall_seconds,
        "peak_rss_bytes": int(runtime.peak_rss_bytes()),
    }


def relocate_power_synchrony_cache(
    request: CacheRelocationRequest,
    runtime: RelocationRuntime | None = None,
) -> RelocationResult:
    """Validate, byte-copy, and atomically publish Power/Synchrony cache artifacts.

    Parameters
    ----------
    request : CacheRelocationRequest
        Immutable producer/destination paths, scientifically equivalent
        configurations, root mapping, and receipt provenance. No numerical LFP,
        phase, synchrony, PPC, or Spike-phase computation is performed.
    runtime : RelocationRuntime or None
        Optional injected scalar host-clock/resource providers. ``None`` uses
        local process defaults.

    Returns
    -------
    RelocationResult
        Final destination cache and external receipt paths after atomic directory
        publication. Power/Synchrony NPZ bytes retain their stored array shapes,
        axis order, units, values, and NaN missingness exactly.

    Raises
    ------
    ValueError, FileExistsError, OSError
        If source compatibility, source bytes, mapping, containment, schemas,
        copying, publication, or receipt invariants fail. Private staging and
        temporary receipt artifacts are removed on failure. A newly published
        destination is rolled back only when its captured directory and all
        member identities are still unchanged; any concurrent destination change
        preserves the whole destination for its other owner.
    """
    # Provenance must be valid before any filesystem hash or staging allocation.
    _validate_git_commit(request.git_commit)
    active_runtime = _default_runtime() if runtime is None else runtime
    started_timestamp = _utc_timestamp(active_runtime.utc_now())
    started_monotonic = active_runtime.monotonic_seconds()
    staging_directory: Path | None = None
    temporary_receipt: Path | None = None
    destination_cache: Path | None = None
    destination_published = False
    published_destination_identity: _PublishedDestinationIdentity | None = None
    try:
        local_root, cluster_root = _validate_single_root_mapping(request)
        _validate_scientific_equivalence(request, local_root, cluster_root)
        source_cache, destination_cache, receipt_path = _validate_request_paths(
            request, local_root, cluster_root
        )
        _validate_source_cache_members(source_cache)
        _validate_configured_source_files(request, local_root, cluster_root)

        # Power and Synchrony have identical configured file scopes. One source
        # fingerprint call per host preserves the model's single sidecar hash.
        source_fingerprints = fingerprint_source_files(request.source_config, component="power")
        destination_fingerprints = fingerprint_source_files(
            request.destination_config, component="power"
        )
        source_manifest, source_manifest_hash = _validate_source_manifest(
            source_cache, request.source_config, source_fingerprints
        )
        source_hashes = _source_records_with_digests(
            source_fingerprints, destination_fingerprints, local_root, cluster_root
        )
        rebound_manifest = rebind_power_synchrony_manifest(
            source_manifest,
            request.destination_config,
            destination_source_fingerprints={
                component: destination_fingerprints for component in _COPIED_COMPONENTS
            },
        )

        source_component_hashes = {
            component: _hash_file_record(source_cache / f"{component}.npz")
            for component in _COPIED_COMPONENTS
        }
        staging_directory = _create_staging_directory(destination_cache)
        staged_component_hashes: dict[str, dict[str, object]] = {}
        for component in _COPIED_COMPONENTS:
            source_component = source_cache / f"{component}.npz"
            staged_component = staging_directory / f"{component}.npz"
            _copy_component_file(source_component, staged_component)
            staged_hash = _hash_file_record(staged_component)
            if staged_hash != source_component_hashes[component]:
                raise ValueError(f"component byte hash differs after copy: {component}")
            staged_component_hashes[component] = staged_hash
        _write_ascii_json(staging_directory / "manifest.json", rebound_manifest)

        # Header-only public status validates the stage without allocating NPZ
        # payloads; the same live source records prevent a second sidecar hash.
        receipt_states: dict[str, str] = {}
        for component in _COPIED_COMPONENTS:
            stage_status = assess_component_status(
                staging_directory,
                component,
                request.destination_config,
                rebound_manifest,
                current_source_fingerprints=destination_fingerprints,
                validate_headers_only=True,
            )
            receipt_states[component] = stage_status.status
            if stage_status.status != "compatible":
                raise ValueError(f"staged component {component} is not compatible: {stage_status.differences}")
        staged_spike_status = assess_component_status(
            staging_directory,
            "spike_phase",
            request.destination_config,
            rebound_manifest,
            current_source_fingerprints=destination_fingerprints,
            validate_headers_only=True,
        )
        receipt_states["spike_phase"] = staged_spike_status.status
        if staged_spike_status.status != "missing":
            raise ValueError("staged cache must have no Spike-phase component")

        component_hashes = {
            component: {
                "local_path": str(source_cache / f"{component}.npz"),
                "destination_path": str(destination_cache / f"{component}.npz"),
                "local": source_component_hashes[component],
                "destination": staged_component_hashes[component],
            }
            for component in _COPIED_COMPONENTS
        }
        rebound_manifest_hash = lfp_summary_models.sha256_file_content(
            staging_directory / "manifest.json"
        )

        # Write and parse the external receipt before publishing either public
        # artifact. The temporary file stays private until cache publication wins.
        completed_timestamp = _utc_timestamp(active_runtime.utc_now())
        wall_seconds = active_runtime.monotonic_seconds() - started_monotonic
        receipt_payload = _receipt_payload(
            request,
            local_root,
            cluster_root,
            source_cache,
            destination_cache,
            source_manifest,
            source_manifest_hash,
            rebound_manifest_hash,
            component_hashes,
            source_hashes,
            receipt_states,
            active_runtime,
            started_timestamp,
            completed_timestamp,
            wall_seconds,
        )
        receipt_parent = receipt_path.parent
        _require_nonsymlink(receipt_parent, "receipt parent")
        receipt_parent.mkdir(parents=True, exist_ok=True)
        _require_nonsymlink(receipt_parent, "receipt parent")
        with tempfile.NamedTemporaryFile(
            prefix=f".{receipt_path.name}-", suffix=".tmp", dir=receipt_parent, delete=False
        ) as handle:
            temporary_receipt = Path(handle.name)
        temporary_receipt.unlink()
        _write_ascii_json(temporary_receipt, receipt_payload)
        if _read_ascii_json(temporary_receipt, "temporary receipt") != receipt_payload:
            raise ValueError("temporary receipt did not preserve provenance")

        # Capture staging ownership before rename. Directory timestamps are not
        # included because the atomic rename may update them after publication.
        published_destination_identity = _capture_cache_ownership(staging_directory)
        _publish_staged_cache(staging_directory, destination_cache)
        staging_directory = None
        destination_published = True
        _require_cache_ownership(
            destination_cache,
            published_destination_identity,
            "immediately after publication",
        )
        final_manifest = _read_ascii_json(destination_cache / "manifest.json", "destination manifest")
        for component in _COPIED_COMPONENTS:
            status = assess_component_status(
                destination_cache,
                component,
                request.destination_config,
                final_manifest,
                current_source_fingerprints=destination_fingerprints,
                validate_headers_only=True,
            )
            if status.status != "compatible":
                raise ValueError(f"published component {component} is not compatible: {status.differences}")
        spike_status = assess_component_status(
            destination_cache,
            "spike_phase",
            request.destination_config,
            final_manifest,
            current_source_fingerprints=destination_fingerprints,
            validate_headers_only=True,
        )
        if spike_status.status != "missing":
            raise ValueError("published cache must have no Spike-phase component")
        _require_cache_ownership(
            destination_cache,
            published_destination_identity,
            "immediately before receipt commit",
        )
        os.replace(temporary_receipt, receipt_path)
        temporary_receipt = None
        return RelocationResult(destination_cache, receipt_path)
    except Exception:
        _cleanup_created_path(staging_directory)
        _cleanup_created_path(temporary_receipt)
        if destination_published:
            _rollback_published_destination(destination_cache, published_destination_identity)
        raise


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit relocation command paths and provenance arguments.

    Parameters
    ----------
    arguments : Sequence[str] or None
        Explicit command arguments excluding the program name, or ``None`` to
        use process arguments. No path defaults or scientific configuration
        inference is permitted.

    Returns
    -------
    argparse.Namespace
        Required cache/config/root/receipt/Git values as strings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-directory", required=True)
    parser.add_argument("--destination-cache-directory", required=True)
    parser.add_argument("--source-config-json", required=True)
    parser.add_argument("--destination-config-json", required=True)
    parser.add_argument("--local-session-root", required=True)
    parser.add_argument("--cluster-session-root", required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument("--git-commit", required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the explicit relocation CLI and record the exact module invocation.

    Parameters
    ----------
    arguments : Sequence[str] or None
        Explicit option sequence excluding ``python -m`` or process arguments.
        Source/destination config JSON files contain paths and documented units,
        never raw component arrays.

    Returns
    -------
    int
        Zero after successful atomic publication and receipt replacement.
    """
    parsed = _parse_args(arguments)
    source_config = lfp_summary_config_from_json(
        Path(parsed.source_config_json).read_text(encoding="ascii")
    )
    destination_config = lfp_summary_config_from_json(
        Path(parsed.destination_config_json).read_text(encoding="ascii")
    )
    command_arguments = list(os.sys.argv[1:] if arguments is None else arguments)
    request = CacheRelocationRequest(
        source_cache_directory=Path(parsed.source_cache_directory),
        destination_cache_directory=Path(parsed.destination_cache_directory),
        source_config=source_config,
        destination_config=destination_config,
        local_session_root=Path(parsed.local_session_root),
        cluster_session_root=Path(parsed.cluster_session_root),
        receipt_path=Path(parsed.receipt_path),
        command=(
            os.sys.executable,
            "-m",
            "src.neural_analysis.lfp_summary_cache_relocation",
            *command_arguments,
        ),
        git_commit=parsed.git_commit,
    )
    relocate_power_synchrony_cache(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
