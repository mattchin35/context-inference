"""Atomic, fingerprinted work artifacts for PPC phase preparation and checkpoints.

These files are execution-only intermediates. They never create or update an
inspection-cache manifest and cannot represent final scientific components.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PreparedPhaseCache:
    """Validated memory-mapped prepared phase representation.

    ``phase`` and ``valid`` have identical `(site, frequency, trial, time)`
    axes. Phase is read-only complex64 dimensionless unit phase; validity is
    read-only Boolean. ``axes`` includes stable site IDs, Hz frequencies,
    trial-table rows, relative seconds, `(site, trial)` Boolean validity, and
    fixed-width Unicode exclusion reasons. Invalid values are explicit through
    validity/reason arrays rather than inferred from collapsed masks.
    """

    cache_directory: Path
    metadata: dict[str, object]
    phase: np.memmap
    valid: np.memmap
    axes: dict[str, np.ndarray]


@dataclass(frozen=True)
class PPCCheckpoint:
    """One validated completed PPC execution block.

    ``block_id`` is a safe filename token. ``arrays`` maps names to numeric or
    Boolean arrays loaded with ``allow_pickle=False``; its axes/units are owned
    by checkpoint metadata. Checkpoints are execution-only and never final
    inspection components.
    """

    run_directory: Path
    block_id: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, object]


_PREPARED_METADATA_KEYS = frozenset(
    {
        "generator",
        "analysis_version",
        "schema_version",
        "source_fingerprint",
        "scientific_fingerprint",
        "representation_fingerprint",
        "execution_settings",
        "axes",
        "shapes",
        "dtypes",
        "units",
    }
)
_PREPARED_AXIS_KEYS = frozenset(
    {
        "site_ids",
        "frequency_hz",
        "trial_indices",
        "relative_time_s",
        "site_trial_valid",
        "site_trial_exclusion_reason",
    }
)


def write_prepared_phase_cache(
    cache_root: Path,
    metadata: Mapping[str, object],
    phase: np.ndarray,
    valid: np.ndarray,
    axes: Mapping[str, np.ndarray],
) -> Path:
    """Atomically write one fingerprinted prepared phase cache.

    Parameters
    ----------
    cache_root : pathlib.Path
        Work-cache root. This function writes only beneath
        ``prepared_phase/<representation_fingerprint>``.
    metadata : Mapping[str, object]
        Complete JSON-safe work metadata containing exact source/scientific/
        representation identities, axes, shapes, dtypes, units, and execution
        settings. It never enters a final component manifest.
    phase, valid : numpy.ndarray
        Complex64 and Boolean arrays with identical `(site, frequency, trial,
        time)` axes. Invalid phase locations are represented by ``valid``.
    axes : Mapping[str, numpy.ndarray]
        Coordinate arrays including `(site, trial)` Boolean validity and
        fixed-width Unicode exclusion reasons. Relative time is seconds and
        frequency is Hz.

    Returns
    -------
    pathlib.Path
        Exact completed cache directory. The completion marker is atomically
        replaced only after every data and metadata file is validated/written.

    Raises
    ------
    ValueError, FileExistsError, OSError
        For invalid contracts, another exact-directory writer, or atomic I/O
        failure. Failure never writes a valid completion marker.
    """
    normalized_metadata = _prepared_metadata(metadata)
    phase_array, valid_array, normalized_axes = _prepared_arrays(phase, valid, axes)
    _validate_prepared_metadata_arrays(normalized_metadata, phase_array, valid_array, normalized_axes)
    directory = Path(cache_root) / "prepared_phase" / str(
        normalized_metadata["representation_fingerprint"]
    )
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_lock(directory)
    try:
        # Invalidate an older transaction before replacing any of its data files.
        (directory / "complete.json").unlink(missing_ok=True)
        _atomic_npy(directory, "phase.npy", phase_array)
        _atomic_npy(directory, "valid.npy", valid_array)
        _atomic_npz(directory, "axes.npz", normalized_axes)
        _atomic_json(directory, "metadata.json", normalized_metadata)
        _atomic_json(
            directory,
            "complete.json",
            {"representation_fingerprint": normalized_metadata["representation_fingerprint"]},
        )
    finally:
        _release_lock(lock_path)
    return directory


def load_prepared_phase_cache(
    cache_root: Path,
    expected_metadata: Mapping[str, object],
) -> PreparedPhaseCache | None:
    """Load only an exact completed prepared-phase cache with read-only mmap arrays.

    Parameters
    ----------
    cache_root : pathlib.Path
        Work-cache root containing fingerprinted prepared-phase directories.
    expected_metadata : Mapping[str, object]
        Complete expected JSON-safe metadata. Any identity, schema, axis, shape,
        dtype, unit, or execution-setting mismatch rejects reuse.

    Returns
    -------
    PreparedPhaseCache or None
        Read-only mmap-backed complex64/Boolean phase arrays and exact axes, or
        ``None`` when work is absent, incomplete, unsafe, or incompatible.
    """
    try:
        expected = _prepared_metadata(expected_metadata)
        directory = Path(cache_root) / "prepared_phase" / str(
            expected["representation_fingerprint"]
        )
        completion = _load_json(directory / "complete.json")
        stored = _load_json(directory / "metadata.json")
        if stored != expected or completion != {
            "representation_fingerprint": expected["representation_fingerprint"]
        }:
            return None
        phase = np.load(directory / "phase.npy", mmap_mode="r", allow_pickle=False)
        valid = np.load(directory / "valid.npy", mmap_mode="r", allow_pickle=False)
        with np.load(directory / "axes.npz", allow_pickle=False) as loaded_axes:
            axes = {name: loaded_axes[name].copy() for name in loaded_axes.files}
        phase_array, valid_array, normalized_axes = _prepared_arrays(phase, valid, axes)
        _validate_prepared_metadata_arrays(expected, phase_array, valid_array, normalized_axes)
        if not isinstance(phase, np.memmap) or not isinstance(valid, np.memmap):
            return None
        if not np.array_equal(phase, phase_array) or not np.array_equal(valid, valid_array):
            return None
        return PreparedPhaseCache(directory, stored, phase, valid, normalized_axes)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def write_ppc_checkpoint(
    run_directory: Path,
    block_id: str,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
) -> Path:
    """Atomically write one validated PPC checkpoint and completion marker.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact ``ppc/<run_fingerprint>`` directory. Its basename must equal the
        metadata run fingerprint.
    block_id : str
        Safe nonempty filename token identifying one execution block.
    arrays : Mapping[str, numpy.ndarray]
        Non-object numeric or Boolean arrays. Their axes/units are declared in
        metadata and no pickle-bearing array is accepted.
    metadata : Mapping[str, object]
        Complete JSON-safe run metadata containing at least ``run_fingerprint``.

    Returns
    -------
    pathlib.Path
        Completed block NPZ path. The matching completion marker is written last.

    Raises
    ------
    ValueError, FileExistsError, OSError
        For unsafe identities/arrays, concurrent writers, or I/O failure. No
        partial block is returned as valid.
    """
    normalized_metadata = _checkpoint_metadata(run_directory, metadata)
    safe_block_id = _safe_block_id(block_id)
    normalized_arrays = _checkpoint_arrays(arrays)
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_lock(directory)
    try:
        blocks = directory / "blocks"
        blocks.mkdir(exist_ok=True)
        # An older marker must not certify a partially replaced block after failure.
        (blocks / f"{safe_block_id}.complete.json").unlink(missing_ok=True)
        _atomic_json(directory, "metadata.json", normalized_metadata)
        _atomic_npz(blocks, f"{safe_block_id}.npz", normalized_arrays)
        _atomic_json(
            blocks,
            f"{safe_block_id}.complete.json",
            {"block_id": safe_block_id, "run_fingerprint": normalized_metadata["run_fingerprint"]},
        )
    finally:
        _release_lock(lock_path)
    return directory / "blocks" / f"{safe_block_id}.npz"


def load_valid_ppc_checkpoints(
    run_directory: Path,
    expected_metadata: Mapping[str, object],
) -> tuple[PPCCheckpoint, ...]:
    """Return only exact completed safe PPC checkpoints in stable block-id order.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact fingerprinted PPC work directory.
    expected_metadata : Mapping[str, object]
        Complete expected run metadata. Any mismatch rejects all checkpoint reuse.

    Returns
    -------
    tuple[PPCCheckpoint, ...]
        Validated checkpoints sorted by block ID, or an empty tuple when the run
        is missing/incompatible. NPZ loading always uses ``allow_pickle=False``.
    """
    try:
        expected = _checkpoint_metadata(run_directory, expected_metadata)
        directory = Path(run_directory)
        if _load_json(directory / "metadata.json") != expected:
            return ()
        checkpoints: list[PPCCheckpoint] = []
        for marker_path in sorted((directory / "blocks").glob("*.complete.json")):
            marker = _load_json(marker_path)
            block_id = _safe_block_id(str(marker.get("block_id", "")))
            if marker != {"block_id": block_id, "run_fingerprint": expected["run_fingerprint"]}:
                continue
            npz_path = directory / "blocks" / f"{block_id}.npz"
            if not npz_path.is_file():
                continue
            with np.load(npz_path, allow_pickle=False) as loaded:
                arrays = {name: loaded[name].copy() for name in loaded.files}
            checkpoints.append(PPCCheckpoint(directory, block_id, _checkpoint_arrays(arrays), expected))
        return tuple(checkpoints)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return ()


def cleanup_ppc_run(run_directory: Path, expected_run_fingerprint: str) -> None:
    """Remove only one exact validated PPC run directory.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact run directory beneath a PPC work root.
    expected_run_fingerprint : str
        Required nonempty fingerprint matching both directory basename and stored
        metadata. It has no physical units or missing-value representation.

    Returns
    -------
    None
        Deletes only the exact validated directory. Sibling runs are untouched.

    Raises
    ------
    ValueError, OSError
        If the expected fingerprint, directory name, or stored metadata does not
        match. The target remains intact on validation failure.
    """
    directory = Path(run_directory)
    if not isinstance(expected_run_fingerprint, str) or not expected_run_fingerprint:
        raise ValueError("expected run fingerprint must be nonempty")
    stored = _load_json(directory / "metadata.json")
    if (
        directory.parent.name != "ppc"
        or directory.name != expected_run_fingerprint
        or stored.get("run_fingerprint") != expected_run_fingerprint
    ):
        raise ValueError("PPC cleanup requires the exact stored run fingerprint")
    shutil.rmtree(directory)


def _prepared_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Validate/copy exact JSON-safe prepared-cache metadata without I/O."""
    value = _json_mapping(metadata)
    if set(value) != _PREPARED_METADATA_KEYS:
        raise ValueError("prepared-phase metadata fields are incomplete")
    if not all(isinstance(value[key], str) and value[key] for key in _PREPARED_METADATA_KEYS - {"execution_settings", "axes", "shapes", "dtypes", "units"}):
        raise ValueError("prepared-phase metadata identifiers must be nonempty")
    if not isinstance(value["execution_settings"], dict):
        raise ValueError("prepared-phase execution settings must be a mapping")
    if value["axes"] != ["site", "frequency", "trial", "time"]:
        raise ValueError("prepared-phase metadata axes must be site/frequency/trial/time")
    shapes = value["shapes"]
    dtypes = value["dtypes"]
    if not isinstance(shapes, dict) or not isinstance(dtypes, dict):
        raise ValueError("prepared-phase metadata shapes and dtypes must be mappings")
    return value


def _prepared_arrays(
    phase: np.ndarray,
    valid: np.ndarray,
    axes: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Validate exact prepared phase arrays/axes while preserving their units."""
    phase_array = np.asarray(phase)
    valid_array = np.asarray(valid)
    if phase_array.dtype != np.dtype(np.complex64) or valid_array.dtype != np.dtype(bool):
        raise ValueError("prepared phase must be complex64 and validity must be Boolean")
    if phase_array.ndim != 4 or valid_array.shape != phase_array.shape:
        raise ValueError("prepared phase and validity require matching (site, frequency, trial, time) axes")
    if not np.isfinite(phase_array.real).all() or not np.isfinite(phase_array.imag).all():
        raise ValueError("prepared phase values must be finite")
    if set(axes) != _PREPARED_AXIS_KEYS:
        raise ValueError("prepared phase axes are incomplete")
    copied = {name: np.asarray(value).copy() for name, value in axes.items()}
    site_count, frequency_count, trial_count, time_count = phase_array.shape
    if (
        copied["site_ids"].dtype.kind != "U"
        or copied["site_ids"].shape != (site_count,)
        or copied["frequency_hz"].shape != (frequency_count,)
        or copied["trial_indices"].dtype != np.dtype(np.int64)
        or copied["trial_indices"].shape != (trial_count,)
        or copied["relative_time_s"].shape != (time_count,)
        or copied["site_trial_valid"].dtype != np.dtype(bool)
        or copied["site_trial_valid"].shape != (site_count, trial_count)
        or copied["site_trial_exclusion_reason"].dtype.kind != "U"
        or copied["site_trial_exclusion_reason"].shape != (site_count, trial_count)
    ):
        raise ValueError("prepared phase axes have incompatible dtype or shape")
    return phase_array, valid_array, copied


def _validate_prepared_metadata_arrays(
    metadata: Mapping[str, object],
    phase: np.ndarray,
    valid: np.ndarray,
    axes: Mapping[str, np.ndarray],
) -> None:
    """Require metadata shape/dtype declarations to exactly match prepared arrays.

    Parameters
    ----------
    metadata : Mapping[str, object]
        Validated JSON metadata declaring cache array shapes and dtypes.
    phase, valid : numpy.ndarray
        Prepared complex64/Boolean arrays on `(site, frequency, trial, time)`.
    axes : Mapping[str, numpy.ndarray]
        Exact coordinate/validity arrays, including `(site, trial)` flags and
        Unicode reasons. Relative time is seconds and frequency is Hz.

    Returns
    -------
    None
        Raises before a cache can be marked complete; it creates no arrays and
        does not alter any axis, units, or missingness representation.
    """
    expected_shapes = {
        "phase": list(phase.shape),
        "valid": list(valid.shape),
        "site_trial_valid": list(axes["site_trial_valid"].shape),
        "site_trial_exclusion_reason": list(axes["site_trial_exclusion_reason"].shape),
    }
    expected_dtypes = {
        "phase": str(phase.dtype),
        "valid": str(valid.dtype),
        "site_trial_valid": str(axes["site_trial_valid"].dtype),
        "site_trial_exclusion_reason": str(axes["site_trial_exclusion_reason"].dtype),
    }
    if metadata["shapes"] != expected_shapes or metadata["dtypes"] != expected_dtypes:
        raise ValueError("prepared-phase metadata shape or dtype declarations differ from arrays")


def _checkpoint_metadata(run_directory: Path, metadata: Mapping[str, object]) -> dict[str, object]:
    """Validate/copy JSON-safe checkpoint metadata bound to one directory."""
    value = _json_mapping(metadata)
    fingerprint = value.get("run_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint or Path(run_directory).name != fingerprint:
        raise ValueError("checkpoint metadata must match the exact run directory")
    return value


def _checkpoint_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Validate/copy safe checkpoint arrays without changing named axes or units."""
    if not arrays:
        raise ValueError("checkpoint arrays must be nonempty")
    copied: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint array names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("checkpoint arrays cannot contain pickle-bearing objects")
        copied[name] = array.copy()
    return copied


def _safe_block_id(block_id: str) -> str:
    """Validate one filename-safe checkpoint block identity without paths."""
    if (
        not isinstance(block_id, str)
        or not block_id
        or block_id in {".", ".."}
        or Path(block_id).name != block_id
        or "/" in block_id
        or "\\" in block_id
    ):
        raise ValueError("checkpoint block_id must be a safe filename token")
    return block_id


def _acquire_lock(directory: Path) -> Path:
    """Create an exact-directory exclusive writer lock or fail without sharing work."""
    lock_path = directory / "writer.lock"
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise FileExistsError(f"work cache writer already holds {lock_path}") from error
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write("active\n")
    return lock_path


def _release_lock(lock_path: Path) -> None:
    """Remove a lock created by this process after ordinary write completion/failure."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _atomic_npy(directory: Path, name: str, array: np.ndarray) -> None:
    """Write one NPY array via a same-directory temporary file and replacement."""
    temporary = _temporary_path(directory, name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
        loaded = np.load(temporary, allow_pickle=False)
        if loaded.dtype != array.dtype or loaded.shape != array.shape:
            raise ValueError("temporary NPY validation failed")
        os.replace(temporary, directory / name)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_npz(directory: Path, name: str, arrays: Mapping[str, np.ndarray]) -> None:
    """Write safe named NPZ arrays via a same-directory temporary replacement."""
    temporary = _temporary_path(directory, name)
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **arrays)
        with np.load(temporary, allow_pickle=False) as loaded:
            if set(loaded.files) != set(arrays):
                raise ValueError("temporary NPZ validation failed")
        os.replace(temporary, directory / name)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(directory: Path, name: str, value: Mapping[str, object]) -> None:
    """Write JSON through a same-directory temporary file and atomic replacement."""
    temporary = _temporary_path(directory, name)
    try:
        temporary.write_text(_canonical_json(value), encoding="utf-8")
        if _load_json(temporary) != dict(value):
            raise ValueError("temporary JSON validation failed")
        os.replace(temporary, directory / name)
    finally:
        temporary.unlink(missing_ok=True)


def _temporary_path(directory: Path, name: str) -> Path:
    """Return an unused same-directory temporary path for one final basename."""
    return directory / f".{name}.tmp-{uuid.uuid4().hex}"


def _json_mapping(value: Mapping[str, object]) -> dict[str, object]:
    """Copy a JSON-safe mapping through canonical encoding without lossy coercion."""
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be a mapping")
    try:
        decoded = json.loads(_canonical_json(value))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("metadata must be JSON-safe") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata must be a JSON object")
    return decoded


def _canonical_json(value: Mapping[str, object]) -> str:
    """Return deterministic JSON for work metadata without pickle/object serialization."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _load_json(path: Path) -> dict[str, object]:
    """Load one JSON object or raise a clear validation exception."""
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("work metadata must be a JSON object")
    return decoded
