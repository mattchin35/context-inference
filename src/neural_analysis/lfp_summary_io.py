"""Safe NPZ cache validation, compatibility inspection, and manifest-last writes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import zipfile

import numpy as np

from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    canonical_config_json,
    component_fingerprint,
    fingerprint_source_files,
    source_value_semantics,
)


@dataclass(frozen=True)
class ComponentStatus:
    """One cache component state and concise explanations of incompatibility."""

    status: str
    differences: tuple[str, ...] = ()


def load_or_initialize_manifest(
    cache_directory: Path,
    config: LFPSummaryConfig,
) -> dict[str, object]:
    """Load a compatible manifest or return an unwritten JSON-ready initial mapping.

    Parameters
    ----------
    cache_directory : pathlib.Path
        Intended cache directory containing an optional ``manifest.json``. This
        function never creates the directory or writes a manifest.
    config : LFPSummaryConfig
        Validated active configuration supplying session and schema identities.

    Returns
    -------
    dict[str, object]
        Existing JSON manifest mapping, or an initial mapping with schema,
        generator, configuration, reference, processing metadata, and empty
        components ready for the first atomic write.

    Raises
    ------
    ValueError
        If an existing manifest is invalid JSON, is not an object, or is not
        compatible with the active session/schema/components merge boundary.
    """
    manifest_path = cache_directory / "manifest.json"
    if not manifest_path.exists():
        return _initial_manifest(config)
    try:
        decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest contains invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError("manifest must be a JSON mapping object")
    _validate_loaded_manifest(decoded, config)
    return decoded


def _initial_manifest(config: LFPSummaryConfig) -> dict[str, object]:
    """Build an unwritten manifest mapping with all top-level metadata fields.

    Parameters
    ----------
    config : LFPSummaryConfig
        Active configuration retained as a canonical JSON-compatible snapshot.

    Returns
    -------
    dict[str, object]
        Initial manifest with no component entries and no filesystem side effect.
    """
    return {
        "schema_version": config.schema_version,
        "session_id": config.session_id,
        "generator": {"module": "src.neural_analysis.lfp_summary_io"},
        "configuration": json.loads(canonical_config_json(config)),
        "reference": {
            "statement": "Component references and normalization metadata are stored per component."
        },
        "processing_metadata": {"status": "initialized"},
        "components": {},
    }


def _validate_loaded_manifest(manifest: Mapping[str, object], config: LFPSummaryConfig) -> None:
    """Validate fields required to safely merge one component into a loaded manifest.

    Parameters
    ----------
    manifest : Mapping[str, object]
        Existing JSON-decoded manifest mapping.
    config : LFPSummaryConfig
        Active schema and session identity.

    Returns
    -------
    None
        The decoded mapping is not modified.
    """
    if manifest.get("schema_version") != config.schema_version:
        raise ValueError("manifest schema version differs from active configuration")
    if manifest.get("session_id") != config.session_id:
        raise ValueError("manifest session id differs from active configuration")
    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("manifest components must be a mapping")


def _component_entry(manifest: Mapping[str, Any], component: str) -> Mapping[str, Any]:
    """Return one JSON manifest component entry.

    Parameters
    ----------
    manifest : Mapping[str, Any]
        Decoded JSON metadata; contains no NPZ arrays.
    component : str
        Component identity: ``power``, ``synchrony``, or ``spike_phase``.

    Returns
    -------
    Mapping[str, Any]
        Component metadata. Missing entries raise ``ValueError`` rather than a
        missing-value sentinel.
    """
    try:
        entry = manifest["components"][component]
    except (KeyError, TypeError) as error:
        raise ValueError(f"manifest lacks component {component}") from error
    if not isinstance(entry, Mapping):
        raise ValueError(f"manifest component {component} is not an object")
    return entry


def _component_filename(entry: Mapping[str, Any], component: str) -> str:
    """Validate a component filename is a local NPZ basename.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Manifest entry for one component.
    component : str
        Component name used only for the default filename.

    Returns
    -------
    str
        Safe basename with ``.npz`` suffix; no path traversal or missing path is allowed.
    """
    filename = entry.get("file_name", f"{component}.npz")
    if not isinstance(filename, str):
        raise ValueError("component file name must be a string")
    path = Path(filename)
    if path.name != filename or path.suffix != ".npz" or filename in {"", ".", ".."}:
        raise ValueError("component file name must be an NPZ basename inside the cache directory")
    return filename


def _validate_arrays(arrays: Mapping[str, np.ndarray], entry: Mapping[str, Any]) -> None:
    """Validate safe NPZ arrays against their named-axis manifest contract.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Numeric, boolean, complex, or fixed-Unicode arrays. Shapes are checked
        through shared named axes; physical units live in ``entry``.
    entry : Mapping[str, Any]
        Component metadata containing axes and physical-unit strings.

    Returns
    -------
    None
        Arrays remain unchanged; NaNs are valid numerical missingness where an
        individual component contract permits them.
    """
    schema = entry.get("array_schema")
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("manifest lacks array_schema")
    dimensions: dict[str, int] = {}
    for name, contract in schema.items():
        if not isinstance(name, str) or not isinstance(contract, Mapping):
            raise ValueError("invalid array schema entry")
        if name not in arrays:
            raise ValueError(f"missing required array: {name}")
        axes = contract.get("axes")
        units = contract.get("units")
        if not isinstance(axes, list) or not axes or not all(isinstance(axis, str) and axis for axis in axes) or not isinstance(units, str) or not units:
            raise ValueError(f"invalid array schema for {name}")
        array = arrays[name]
        if array.dtype.kind not in {"b", "i", "u", "f", "c", "U"}:
            raise ValueError(f"array {name} has object or pickle dtype")
        if array.ndim != len(axes):
            raise ValueError(f"array-axis mismatch for {name}")
        for axis, size in zip(axes, array.shape):
            if axis in dimensions and dimensions[axis] != size:
                raise ValueError(f"array-axis mismatch for {name}: axis {axis}")
            dimensions[axis] = size
    for name, array in arrays.items():
        if array.dtype.kind not in {"b", "i", "u", "f", "c", "U"}:
            raise ValueError(f"array {name} has object or pickle dtype")


def _validate_array_schema_header(
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    schema: Mapping[str, Any],
    dimensions: dict[str, int],
) -> None:
    """Check one NPZ member's NPY header against its named-axis contract.

    Parameters
    ----------
    name : str
        Manifest array name without the ``.npy`` archive suffix.
    shape : tuple[int, ...]
        Stored array shape in manifest axis order.  No numerical array values
        are loaded.
    dtype : numpy.dtype
        Stored NPY dtype. Object and pickle-bearing dtypes are rejected.
    schema : Mapping[str, Any]
        JSON-decoded ``array_schema`` mapping with named axes and physical-unit
        labels.
    dimensions : dict[str, int]
        Mutable named-axis size table shared by all members in one component.

    Returns
    -------
    None
        The header and schema are only inspected.  Array shapes, values, units,
        and missingness are not transformed.
    """
    contract = schema.get(name)
    if not isinstance(contract, Mapping):
        raise ValueError(f"invalid array schema for {name}")
    axes = contract.get("axes")
    units = contract.get("units")
    if (
        not isinstance(axes, list)
        or not axes
        or not all(isinstance(axis, str) and axis for axis in axes)
        or not isinstance(units, str)
        or not units
    ):
        raise ValueError(f"invalid array schema for {name}")
    if dtype.kind not in {"b", "i", "u", "f", "c", "U"}:
        raise ValueError(f"array {name} has object or pickle dtype")
    if len(shape) != len(axes):
        raise ValueError(f"array-axis rank mismatch for {name}")
    for axis, size in zip(axes, shape, strict=True):
        if size < 0:
            raise ValueError(f"array {name} has an invalid negative shape")
        prior_size = dimensions.get(axis)
        if prior_size is not None and prior_size != size:
            raise ValueError(f"array-axis size mismatch for {name}: axis {axis}")
        dimensions[axis] = size


def _read_npy_header(member: zipfile.ZipExtFile) -> tuple[tuple[int, ...], np.dtype[Any]]:
    """Read one bounded NPY header from an opened ZIP member.

    Parameters
    ----------
    member : zipfile.ZipExtFile
        Open compressed ``.npy`` member positioned at its magic prefix. Header
        reads are bounded by NumPy's fixed header limit; numerical payload bytes
        are never read.

    Returns
    -------
    tuple[tuple[int, ...], numpy.dtype]
        Stored shape and dtype. Axis order is represented by the companion
        manifest, while physical units remain manifest metadata.

    Raises
    ------
    ValueError
        If the member is not a supported NPY 1.0 or 2.0 stream or has an unsafe
        header. NPY 3.0 is deliberately rejected because the installed NumPy
        exposes no corresponding bounded public header reader.
    """
    version = np.lib.format.read_magic(member)
    if version == (1, 0):
        shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
    elif version == (2, 0):
        shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
    else:
        raise ValueError(f"unsupported NPY header version: {version}")
    if not isinstance(shape, tuple) or not all(isinstance(size, int) for size in shape):
        raise ValueError("invalid NPY array shape")
    return shape, dtype


def validate_component_npz_headers(
    component_path: Path,
    manifest: Mapping[str, Any],
    component: str,
) -> None:
    """Validate one component NPZ's schema using ZIP and NPY headers only.

    Parameters
    ----------
    component_path : pathlib.Path
        Existing local ``.npz`` component archive. Its compressed numerical
        arrays are not materialized and no pickle support is enabled.
    manifest : Mapping[str, Any]
        JSON-decoded component metadata with an ``array_schema`` defining
        named axes and physical-unit labels.
    component : str
        Component identity used to select the manifest entry.

    Returns
    -------
    None
        Confirms exact NPY member names, unique members, safe dtypes, shapes,
        shared named-axis sizes, and declared uncompressed payload sizes. Array
        values, axis order, units, and NaN missingness remain on disk unchanged.

    Raises
    ------
    ValueError
        If the archive, member names, headers, dtype, schema, or uncompressed
        payload lengths are invalid. The error is raised before any full array
        or 232 MB component payload is allocated.
    """
    entry = _component_entry(manifest, component)
    _component_filename(entry, component)
    schema = entry.get("array_schema")
    if not isinstance(schema, Mapping) or not schema:
        raise ValueError("manifest lacks array_schema")
    expected_names = []
    for name, contract in schema.items():
        if not isinstance(name, str) or not isinstance(contract, Mapping):
            raise ValueError("invalid array schema entry")
        expected_names.append(f"{name}.npy")
    if len(set(expected_names)) != len(expected_names):
        raise ValueError("array schema contains duplicate member names")

    try:
        with zipfile.ZipFile(component_path) as archive:
            members = archive.infolist()
            member_names = [member.filename for member in members]
            if len(member_names) != len(set(member_names)):
                raise ValueError("component archive contains duplicate members")
            if set(member_names) != set(expected_names):
                raise ValueError("component archive members differ from its array schema")
            dimensions: dict[str, int] = {}
            for info in members:
                if info.is_dir() or info.filename not in expected_names:
                    raise ValueError("component archive contains an unsafe member")
                name = info.filename.removesuffix(".npy")
                with archive.open(info, "r") as member:
                    shape, dtype = _read_npy_header(member)
                    header_size = member.tell()
                # Reject unsafe dtype/rank/axis contracts from bounded metadata
                # before interpreting any declared payload byte count.
                _validate_array_schema_header(name, shape, dtype, schema, dimensions)
                element_count = math.prod(shape)
                payload_size = element_count * dtype.itemsize
                if header_size + payload_size != info.file_size:
                    raise ValueError(f"truncated or invalid NPY payload for {name}")
    except (OSError, zipfile.BadZipFile, ValueError) as error:
        if isinstance(error, ValueError):
            raise
        raise ValueError("component contains invalid NPZ or NPY header data") from error


def load_component_arrays(component_path: Path, manifest: Mapping[str, Any], component: str) -> dict[str, np.ndarray]:
    """Load and validate one component NPZ without pickle support.

    Parameters
    ----------
    component_path : Path
        Local component NPZ path.
    manifest : Mapping[str, Any]
        JSON component schema defining axes and physical units.
    component : str
        Component identity selecting the schema.

    Returns
    -------
    dict[str, numpy.ndarray]
        Stored arrays with original shapes, axis order, units metadata external to
        the arrays, and preserved NaN missingness.
    """
    entry = _component_entry(manifest, component)
    _component_filename(entry, component)
    try:
        with np.load(component_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError) as error:
        raise ValueError("component contains invalid or pickle data") from error
    _validate_arrays(arrays, entry)
    return arrays


def _replace_component_file(source: Path, destination: Path) -> None:
    """Atomically replace one component file.

    Parameters
    ----------
    source, destination : Path
        Temporary and final paths on the same filesystem.

    Returns
    -------
    None
        Uses ``os.replace``; no array values, axes, units, or missingness are read.
    """
    os.replace(source, destination)


def _replace_manifest_file(source: Path, destination: Path) -> None:
    """Atomically replace the manifest transaction commit point.

    Parameters
    ----------
    source, destination : Path
        Temporary and final JSON paths on the same filesystem.

    Returns
    -------
    None
        Uses ``os.replace`` and does not inspect component arrays or missingness.
    """
    os.replace(source, destination)


def _temporary_path(parent: Path, suffix: str) -> Path:
    """Reserve a unique temporary path in a cache directory.

    Parameters
    ----------
    parent : Path
        Existing cache directory on the target filesystem.
    suffix : str
        Filename suffix such as ``.npz`` or ``.json``.

    Returns
    -------
    Path
        Empty unique path. No arrays, units, axes, or missing values are written.
    """
    handle = tempfile.NamedTemporaryFile(prefix=".lfp-summary-", suffix=suffix, dir=parent, delete=False)
    handle.close()
    return Path(handle.name)


def _write_temp_npz(parent: Path, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write arrays to a unique temporary NPZ.

    Parameters
    ----------
    parent : Path
        Cache directory receiving the temporary file.
    arrays : Mapping[str, numpy.ndarray]
        Numeric/boolean/fixed-Unicode arrays with caller-defined axes and units.

    Returns
    -------
    Path
        Temporary NPZ path. NaN missing values are stored unchanged.
    """
    path = _temporary_path(parent, ".npz")
    np.savez(path, **arrays)
    return path


def _write_temp_manifest(parent: Path, manifest: Mapping[str, Any]) -> Path:
    """Write JSON-compatible manifest metadata to a temporary file.

    Parameters
    ----------
    parent : Path
        Cache directory receiving the temporary JSON file.
    manifest : Mapping[str, Any]
        JSON-safe metadata describing arrays, axes, units, and source fingerprints.

    Returns
    -------
    Path
        Temporary manifest path. Metadata has no numerical missing-value encoding.
    """
    path = _temporary_path(parent, ".json")
    with path.open("w", encoding="ascii") as handle:
        json.dump(manifest, handle, sort_keys=True, allow_nan=False)
    return path


def _backup_existing_component(component_path: Path, cache_directory: Path) -> Path | None:
    """Move an existing component to a unique rollback path.

    Parameters
    ----------
    component_path : Path
        Current final NPZ component path.
    cache_directory : Path
        Directory used to allocate a same-filesystem temporary backup.

    Returns
    -------
    Path | None
        Backup path, or ``None`` when no previous component exists; no arrays are transformed.
    """
    if not component_path.exists():
        return None
    backup_path = _temporary_path(cache_directory, ".npz")
    os.replace(component_path, backup_path)
    return backup_path


def _restore_backup(component_path: Path, backup_path: Path | None) -> None:
    """Restore a prior component after a replacement failure.

    Parameters
    ----------
    component_path : Path
        Final component filename that may contain a failed replacement.
    backup_path : Path | None
        Prior component backup, or ``None`` if this was the first write.

    Returns
    -------
    None
        Restores bytes unchanged, preserving all array axes, units, and NaN values.
    """
    if component_path.exists():
        component_path.unlink()
    if backup_path is not None and backup_path.exists():
        os.replace(backup_path, component_path)


def write_component_transaction(
    cache_directory: Path,
    component: str,
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
) -> None:
    """Validate, atomically replace a component, then atomically commit its manifest.

    Parameters
    ----------
    cache_directory : Path
        Directory containing final NPZ components and ``manifest.json``.
    component : str
        Component identity naming the manifest entry.
    arrays : Mapping[str, numpy.ndarray]
        Safe arrays with manifest-defined axes, physical units, and preserved NaNs.
    manifest : Mapping[str, Any]
        New JSON-compatible metadata committed after component validation.

    Returns
    -------
    None
        The existing component is retained in a unique rollback file while the
        manifest replacement is attempted. Python-level replacement failures
        restore prior bytes unchanged.
    """
    cache_directory.mkdir(parents=True, exist_ok=True)
    entry = _component_entry(manifest, component)
    component_path = cache_directory / _component_filename(entry, component)
    manifest_path = cache_directory / "manifest.json"
    temporary_component = _write_temp_npz(cache_directory, arrays)
    temporary_manifest: Path | None = None
    backup_component: Path | None = None
    try:
        load_component_arrays(temporary_component, manifest, component)
        temporary_manifest = _write_temp_manifest(cache_directory, manifest)
        json.loads(temporary_manifest.read_text(encoding="ascii"))
        backup_component = _backup_existing_component(component_path, cache_directory)
        try:
            _replace_component_file(temporary_component, component_path)
            _replace_manifest_file(temporary_manifest, manifest_path)
        except Exception:
            _restore_backup(component_path, backup_component)
            raise
        if backup_component is not None and backup_component.exists():
            backup_component.unlink()
    finally:
        for path in (temporary_component, temporary_manifest, backup_component):
            if path is not None and path.exists():
                path.unlink()


def _source_fingerprint_differences(
    entry: Mapping[str, Any],
    config: LFPSummaryConfig,
    component: str,
    current_source_fingerprints: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Compare cached and active component-scoped source fingerprints.

    Parameters
    ----------
    entry : Mapping[str, Any]
        Manifest entry containing a source-fingerprint mapping.
    config : LFPSummaryConfig
        Active configuration identifying source paths.
    component : str
        Component that determines which sources are relevant.
    current_source_fingerprints : Mapping[str, Any] or None
        Optional already-validated live source records for this component. When
        supplied, they are compared directly and no source path is restatted or
        Open Ephys sidecar is hashed a second time.

    Returns
    -------
    tuple[str, ...]
        Empty when fingerprints match; otherwise human-readable stale reasons.
        Missing paths are compared using their documented ``-1`` metadata.
    """
    cached = entry.get("source_fingerprints")
    if not isinstance(cached, Mapping):
        return ("manifest lacks source fingerprints",)
    current = (
        current_source_fingerprints
        if current_source_fingerprints is not None
        else fingerprint_source_files(config, component=component)
    )
    if cached != current:
        return ("source fingerprints differ from active session inputs",)
    return ()


def assess_component_status(
    cache_directory: Path,
    component: str,
    config: LFPSummaryConfig,
    manifest: Mapping[str, Any],
    *,
    current_source_fingerprints: Mapping[str, Any] | None = None,
    validate_headers_only: bool = False,
) -> ComponentStatus:
    """Classify one component as missing, compatible, stale, running, or failed.

    Parameters
    ----------
    cache_directory : Path
        Directory containing component NPZ files and manifest metadata.
    component : str
        Component identity to inspect.
    config : LFPSummaryConfig
        Active configuration with paths, units, and axis contracts.
    manifest : Mapping[str, Any]
        Decoded cache manifest.
    current_source_fingerprints : Mapping[str, Any] or None, keyword-only
        Optional active component-scoped source records obtained by an earlier
        streamed verification pass. They preserve the live fingerprint format
        and avoid rehashing Open Ephys sidecars.
    validate_headers_only : bool, keyword-only
        When ``True``, validate only ZIP/NPY headers and declared payload sizes
        through :func:`validate_component_npz_headers`. This avoids materializing
        component arrays while retaining named-axis and unit-schema checks.

    Returns
    -------
    ComponentStatus
        Missing/running/failed/stale/compatible state. A corrupt array is failed;
        numerical NaNs inside otherwise valid arrays retain their normal meaning.
    """
    try:
        entry = _component_entry(manifest, component)
        filename = _component_filename(entry, component)
    except ValueError as error:
        return ComponentStatus("missing", (str(error),))
    component_path = cache_directory / filename
    if not component_path.is_file():
        return ComponentStatus("missing", (f"missing {filename}",))
    state = entry.get("state", "complete")
    if state == "running":
        return ComponentStatus("running", ("component computation is running",))
    if state == "failed":
        return ComponentStatus("failed", ("last computation failed",))
    if state != "complete":
        return ComponentStatus("failed", (f"invalid component state: {state}",))
    try:
        if validate_headers_only:
            validate_component_npz_headers(component_path, manifest, component)
        else:
            load_component_arrays(component_path, manifest, component)
    except ValueError as error:
        return ComponentStatus("failed", (f"invalid component arrays: {error}",))
    differences: list[str] = []
    if manifest.get("session_id") != config.session_id:
        differences.append("session id differs from active session")
    cached_snapshot = entry.get("configuration_snapshot")
    if not isinstance(cached_snapshot, Mapping):
        differences.append("manifest lacks configuration snapshot")
    expected = component_fingerprint(component, config)
    actual = entry.get("configuration_fingerprint")
    if actual != expected:
        differences.append(f"configuration fingerprint differs: cached={actual}, current={expected}")
    differences.extend(
        _source_fingerprint_differences(
            entry,
            config,
            component,
            current_source_fingerprints=current_source_fingerprints,
        )
    )
    if differences:
        return ComponentStatus("stale", tuple(differences))
    return ComponentStatus("compatible")


def rebind_power_synchrony_manifest(
    source_manifest: Mapping[str, Any],
    destination_config: LFPSummaryConfig,
    *,
    destination_source_fingerprints: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """Build a destination-compatible Power/Synchrony manifest without I/O.

    Parameters
    ----------
    source_manifest : Mapping[str, Any]
        JSON-decoded producer manifest. Completion time, generator, schema,
        units, scientific metadata, and all non-copied component metadata are
        retained as producer history.
    destination_config : LFPSummaryConfig
        Valid active cluster configuration. Its Power/Synchrony fingerprints
        and canonical snapshot become destination compatibility identity; no
        numerical array, source file, or cache path is opened by this helper.
    destination_source_fingerprints : Mapping[str, Mapping[str, Any]]
        Prevalidated active source records keyed by ``"power"`` and
        ``"synchrony"``. The caller owns their streamed verification.

    Returns
    -------
    dict[str, object]
        Deep-copied rebound manifest. Only top-level ``configuration`` and the
        copied components' configuration snapshot, fingerprint, source records,
        and source-value semantics are replaced. Array shapes, axes, physical
        units, completion status, and producer/scientific metadata are unchanged.

    Raises
    ------
    ValueError
        If Power/Synchrony entries or their required prevalidated destination
        source-record mappings are absent or malformed.
    """
    rebound = deepcopy(dict(source_manifest))
    components = rebound.get("components")
    if not isinstance(components, dict):
        raise ValueError("manifest components must be a mapping")
    # These JSON trees are destination compatibility identity, not shared
    # producer provenance. Keep each location independently mutable.
    snapshot = json.loads(canonical_config_json(destination_config))
    rebound["configuration"] = deepcopy(snapshot)
    for component in ("power", "synchrony"):
        entry = components.get(component)
        fingerprints = destination_source_fingerprints.get(component)
        if not isinstance(entry, dict) or not isinstance(fingerprints, Mapping):
            raise ValueError(f"manifest lacks prevalidated {component} metadata")
        entry["configuration_snapshot"] = deepcopy(snapshot)
        entry["configuration_fingerprint"] = component_fingerprint(
            component, destination_config
        )
        entry["source_fingerprints"] = deepcopy(dict(fingerprints))
        entry["source_value_semantics"] = source_value_semantics(destination_config)
    return rebound
