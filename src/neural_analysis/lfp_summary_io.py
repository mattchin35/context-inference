"""Safe NPZ cache validation, compatibility inspection, and manifest-last writes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from src.neural_analysis.lfp_summary_models import (
    LFPSummaryConfig,
    component_fingerprint,
    fingerprint_source_files,
)


@dataclass(frozen=True)
class ComponentStatus:
    """One cache component state and concise explanations of incompatibility."""

    status: str
    differences: tuple[str, ...] = ()


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

    Returns
    -------
    tuple[str, ...]
        Empty when fingerprints match; otherwise human-readable stale reasons.
        Missing paths are compared using their documented ``-1`` metadata.
    """
    cached = entry.get("source_fingerprints")
    if not isinstance(cached, Mapping):
        return ("manifest lacks source fingerprints",)
    current = fingerprint_source_files(config, component=component)
    if cached != current:
        return ("source fingerprints differ from active session inputs",)
    return ()


def assess_component_status(
    cache_directory: Path,
    component: str,
    config: LFPSummaryConfig,
    manifest: Mapping[str, Any],
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
    differences.extend(_source_fingerprint_differences(entry, config, component))
    if differences:
        return ComponentStatus("stale", tuple(differences))
    return ComponentStatus("compatible")
