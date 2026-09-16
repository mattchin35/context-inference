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
import socket
import stat
import uuid
from typing import Callable, Mapping
import zipfile

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


def recover_stale_lock(
    lock_path: Path,
    expected_run_fingerprint: str,
    *,
    hostname: str,
    pid_is_alive: Callable[[int], bool],
) -> bool:
    """Remove only a validated same-host dead-process PPC ownership lock.

    Parameters
    ----------
    lock_path : pathlib.Path
        Existing regular ``executor.lock`` or ``writer.lock`` at the exact
        ``<work_root>/ppc/<run_fingerprint>/`` level. Symbolic links are never
        followed. Lock data is a UTF-8 JSON ownership record without arrays.
    expected_run_fingerprint : str
        Exact categorical PPC run identity expected in both the directory name
        and ownership record. It has no physical units.
    hostname : str
        Current host identity. Recovery is forbidden for locks from other hosts
        because their process liveness cannot be established locally.
    pid_is_alive : callable
        Receives the positive integer owning PID and returns a Boolean. It is
        called only after path, schema, scope, host, and fingerprint validation.

    Returns
    -------
    bool
        ``True`` only after the exact validated dead-process lock is unlinked.

    Raises
    ------
    FileExistsError
        If the validated owning PID is still alive.
    ValueError
        If the path, ownership schema, hostname, fingerprint, scope, PID, or
        liveness callback result is unsafe or unverifiable. Rejected locks are
        never altered.
    OSError
        If an otherwise validated exact lock cannot be read or removed.
    """
    path = Path(lock_path)
    _validate_recovery_path(path, expected_run_fingerprint)
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("hostname must be a nonempty string")
    if not callable(pid_is_alive):
        raise ValueError("pid_is_alive must be callable")

    descriptor = _open_lock_without_following(path)
    try:
        descriptor_stat = os.fstat(descriptor)
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            try:
                record = json.load(handle)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("lock ownership record is malformed") from error
    finally:
        os.close(descriptor)
    _validate_lock_record(path, record, expected_run_fingerprint, hostname)

    owner_pid = record["pid"]
    assert isinstance(owner_pid, int)
    alive = pid_is_alive(owner_pid)
    if not isinstance(alive, bool):
        raise ValueError("pid_is_alive must return Boolean")
    if alive:
        raise FileExistsError(f"live PPC lock owner still holds {path}")

    current_stat = os.lstat(path)
    if (
        stat.S_ISLNK(current_stat.st_mode)
        or current_stat.st_dev != descriptor_stat.st_dev
        or current_stat.st_ino != descriptor_stat.st_ino
    ):
        raise ValueError("lock changed while stale ownership was validated")
    path.unlink()
    return True


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
    lock_path = _acquire_lock(
        directory,
        str(normalized_metadata["representation_fingerprint"]),
        "prepared_phase",
    )
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
    *,
    copy_arrays: bool = True,
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
    copy_arrays : bool, default=True
        ``True`` defensively copies every accepted input before publication,
        preserving the established public writer behavior. ``False`` transfers
        already-owned read-only non-object arrays directly to the synchronous
        NPZ publication path. It is intended only for a bounded parent-owned
        checkpoint block and never permits a writable input array.

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
    if not isinstance(copy_arrays, bool):
        raise ValueError("copy_arrays must be Boolean")
    normalized_metadata = _checkpoint_metadata(run_directory, metadata)
    safe_block_id = _safe_block_id(block_id)
    normalized_arrays = _checkpoint_arrays(arrays, copy_arrays=copy_arrays)
    directory = Path(run_directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = _acquire_lock(
        directory,
        str(normalized_metadata["run_fingerprint"]),
        f"checkpoint:{safe_block_id}",
    )
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


def load_valid_ppc_checkpoint(
    run_directory: Path,
    block_id: str,
    expected_metadata: Mapping[str, object],
    *,
    expected_array_schema: Mapping[str, tuple[np.dtype[object], tuple[int, ...]]] | None = None,
    maximum_array_bytes: int | None = None,
) -> PPCCheckpoint | None:
    """Load one exact completed PPC block without opening its siblings.

    Parameters
    ----------
    run_directory : pathlib.Path
        Exact ``ppc/<run_fingerprint>`` work directory.
    block_id : str
        Safe nonempty filename token for the single requested block.
    expected_metadata : Mapping[str, object]
        Exact JSON-safe run metadata. Its ``run_fingerprint`` must equal the
        directory name and stored run metadata.
    expected_array_schema : Mapping[str, tuple[numpy.dtype, tuple[int, ...]]] or None, default=None
        Optional exact key-to-``(dtype, shape)`` schema for a bounded caller.
        When supplied, ZIP/NPY headers are validated before NumPy is permitted
        to materialize any member array.
    maximum_array_bytes : int or None, default=None
        Optional nonnegative maximum total numeric member bytes for
        ``expected_array_schema``. It is compared to declared uncompressed NPY
        element bytes, not ZIP compressed bytes.

    Returns
    -------
    PPCCheckpoint or None
        One owned frozen numeric/Boolean array mapping when its run metadata,
        completion marker, and NPZ all match exactly; otherwise ``None``. The
        result contains no arrays from any sibling checkpoint.

    Notes
    -----
    NPZ members are read with ``allow_pickle=False`` and frozen in place after
    loading. NumPy creates each member as its own loaded array, so this avoids
    a redundant second full-block copy while keeping the returned checkpoint
    independent of its source file and caller arrays.
    """
    try:
        expected = _checkpoint_metadata(run_directory, expected_metadata)
        safe_block_id = _safe_block_id(block_id)
        directory = Path(run_directory)
        if _load_json(directory / "metadata.json") != expected:
            return None
        marker = _load_json(directory / "blocks" / f"{safe_block_id}.complete.json")
        if marker != {
            "block_id": safe_block_id,
            "run_fingerprint": expected["run_fingerprint"],
        }:
            return None
        npz_path = directory / "blocks" / f"{safe_block_id}.npz"
        if not npz_path.is_file():
            return None
        if not _checkpoint_npz_headers_match(
            npz_path,
            expected_array_schema=expected_array_schema,
            maximum_array_bytes=maximum_array_bytes,
        ):
            return None
        with np.load(npz_path, allow_pickle=False) as loaded:
            arrays = {name: loaded[name] for name in loaded.files}
        for value in arrays.values():
            array = np.asarray(value)
            if array.dtype.hasobject:
                return None
            array.setflags(write=False)
        frozen = _checkpoint_arrays(arrays, copy_arrays=False)
        return PPCCheckpoint(directory, safe_block_id, frozen, expected)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        EOFError,
        RuntimeError,
        NotImplementedError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ):
        return None


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
            checkpoint = load_valid_ppc_checkpoint(directory, block_id, expected)
            if checkpoint is not None:
                checkpoints.append(checkpoint)
        return tuple(checkpoints)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
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


def _checkpoint_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    copy_arrays: bool = True,
) -> dict[str, np.ndarray]:
    """Validate checkpoint arrays and optionally retain their existing storage.

    Parameters
    ----------
    arrays : Mapping[str, numpy.ndarray]
        Nonempty named numeric/Boolean arrays with caller-defined documented
        axes and units. Object dtype is never accepted.
    copy_arrays : bool, default=True
        ``True`` returns defensive writable copies for public writer inputs.
        ``False`` requires every supplied array to be read-only and returns
        the exact same ndarray objects for synchronous ownership transfer.

    Returns
    -------
    dict[str, numpy.ndarray]
        Validated named arrays. In no-copy mode each value is identical to the
        matching supplied ndarray.

    Raises
    ------
    ValueError
        If arrays are empty, names are invalid, an array is object dtype, the
        mode is not Boolean, or no-copy input is writable.
    """
    if not isinstance(copy_arrays, bool):
        raise ValueError("copy_arrays must be Boolean")
    if not arrays:
        raise ValueError("checkpoint arrays must be nonempty")
    copied: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name:
            raise ValueError("checkpoint array names must be nonempty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("checkpoint arrays cannot contain pickle-bearing objects")
        if not copy_arrays and array.flags.writeable:
            raise ValueError("copy_arrays=False requires read-only checkpoint arrays")
        copied[name] = array.copy() if copy_arrays else array
    return copied


def _checkpoint_npz_headers_match(
    npz_path: Path,
    *,
    expected_array_schema: Mapping[str, tuple[np.dtype[object], tuple[int, ...]]] | None,
    maximum_array_bytes: int | None,
) -> bool:
    """Validate optional exact NPZ member headers before member materialization.

    Parameters
    ----------
    npz_path : pathlib.Path
        Existing single checkpoint ZIP archive containing only ``.npy``
        members. The archive is read only through ZIP streams in this helper.
    expected_array_schema : mapping or None
        Optional exact mapping from array key to a NumPy dtype and tuple of
        nonnegative axis lengths. ``None`` preserves legacy loader behavior.
    maximum_array_bytes : int or None
        Optional nonnegative upper bound in uncompressed numeric element bytes
        across all declared members. It is only meaningful with a schema.

    Returns
    -------
    bool
        ``True`` for legacy no-schema loads, or when the archive has exactly
        the declared non-object numeric keys/dtypes/shapes within the byte
        bound. ``False`` rejects a malformed or oversized archive before
        ``numpy.load`` can request a member.

    Raises
    ------
    ValueError
        If optional schema/byte-bound arguments are malformed. Archive format
        errors are allowed to propagate to the public loader's safe rejection
        boundary.
    """
    if expected_array_schema is None:
        if maximum_array_bytes is not None:
            raise ValueError("maximum_array_bytes requires expected_array_schema")
        return True
    if not isinstance(expected_array_schema, Mapping) or not expected_array_schema:
        raise ValueError("expected_array_schema must be a nonempty mapping")
    if (
        maximum_array_bytes is None
        or isinstance(maximum_array_bytes, (bool, np.bool_))
        or not isinstance(maximum_array_bytes, (int, np.integer))
        or int(maximum_array_bytes) < 0
    ):
        raise ValueError("maximum_array_bytes must be a nonnegative integer")

    normalized_schema: dict[str, tuple[np.dtype[object], tuple[int, ...]]] = {}
    for name, value in expected_array_schema.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[1], tuple)
        ):
            raise ValueError("expected_array_schema must map names to (dtype, shape) tuples")
        dtype = np.dtype(value[0])
        shape = value[1]
        if dtype.hasobject or any(
            isinstance(length, (bool, np.bool_))
            or not isinstance(length, (int, np.integer))
            or int(length) < 0
            for length in shape
        ):
            raise ValueError("expected checkpoint schema must have numeric dtype and nonnegative shape")
        normalized_schema[name] = (dtype, tuple(int(length) for length in shape))

    with zipfile.ZipFile(npz_path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        archive_keys: list[str] = []
        for info in infos:
            if not info.filename.endswith(".npy") or "/" in info.filename:
                return False
            archive_keys.append(info.filename[:-4])
        if len(archive_keys) != len(set(archive_keys)) or set(archive_keys) != set(normalized_schema):
            return False
        total_bytes = 0
        for name, (expected_dtype, expected_shape) in normalized_schema.items():
            with archive.open(f"{name}.npy", "r") as member:
                version = np.lib.format.read_magic(member)
                if version == (1, 0):
                    shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(member)
                elif version in {(2, 0), (3, 0)}:
                    shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(member)
                else:
                    return False
            if (
                bool(fortran_order)
                or np.dtype(dtype) != expected_dtype
                or tuple(int(length) for length in shape) != expected_shape
                or np.dtype(dtype).hasobject
            ):
                return False
            element_count = 1
            for length in expected_shape:
                if length and element_count > np.iinfo(np.int64).max // length:
                    return False
                element_count *= length
            member_bytes = element_count * expected_dtype.itemsize
            if member_bytes > int(maximum_array_bytes) - total_bytes:
                return False
            total_bytes += member_bytes
    return True


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


def _acquire_lock(
    directory: Path,
    run_fingerprint: str,
    scope: str,
) -> Path:
    """Create one complete exact-directory JSON writer lock exclusively."""
    lock_path = directory / "writer.lock"
    record = _ownership_record(run_fingerprint, scope)
    _write_lock_exclusive(
        lock_path,
        record,
        f"work cache writer already holds {lock_path}",
    )
    return lock_path


def _release_lock(lock_path: Path) -> None:
    """Remove a lock created by this process after ordinary write completion/failure."""
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _ownership_record(run_fingerprint: str, scope: str) -> dict[str, object]:
    """Return one complete local-process lock record without numerical arrays."""
    if not isinstance(run_fingerprint, str) or not run_fingerprint:
        raise ValueError("lock run_fingerprint must be a nonempty string")
    if not isinstance(scope, str) or not scope:
        raise ValueError("lock scope must be a nonempty string")
    return {
        "schema_version": "1",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "run_fingerprint": run_fingerprint,
        "scope": scope,
    }


def _write_lock_exclusive(
    lock_path: Path,
    record: Mapping[str, object],
    collision_message: str,
) -> None:
    """Write complete JSON bytes through an exclusive descriptor and fsync them."""
    payload = (
        json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise FileExistsError(collision_message) from error
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("failed to write complete lock ownership record")
            offset += written
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def _validate_recovery_path(lock_path: Path, expected_run_fingerprint: str) -> None:
    """Require an exact nonsymlink ``ppc/<fingerprint>/<known lock>`` path."""
    if not isinstance(expected_run_fingerprint, str) or not expected_run_fingerprint:
        raise ValueError("expected run fingerprint must be a nonempty string")
    if lock_path.is_symlink():
        raise ValueError("lock path must not be a symbolic link")
    if lock_path.name not in {"executor.lock", "writer.lock"}:
        raise ValueError("lock path must identify executor.lock or writer.lock")
    if lock_path.parent.parent.name != "ppc":
        raise ValueError("lock path must be inside an exact ppc run directory")
    if lock_path.parent.name != expected_run_fingerprint:
        raise ValueError("lock directory fingerprint does not match expected fingerprint")
    if not lock_path.exists() or not lock_path.is_file():
        raise ValueError("lock path must identify an existing regular file")


def _open_lock_without_following(lock_path: Path) -> int:
    """Open one ownership record read-only without following a replaced symlink."""
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags)
    except OSError as error:
        raise ValueError("lock path could not be opened safely") from error
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise ValueError("lock path must identify a regular file")
    return descriptor


def _validate_lock_record(
    lock_path: Path,
    record: object,
    expected_run_fingerprint: str,
    hostname: str,
) -> None:
    """Validate exact lock schema, ownership, run identity, and filename scope."""
    expected_keys = {
        "schema_version",
        "pid",
        "hostname",
        "run_fingerprint",
        "scope",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("lock ownership record has invalid fields")
    owner_pid = record["pid"]
    if (
        record["schema_version"] != "1"
        or isinstance(owner_pid, bool)
        or not isinstance(owner_pid, int)
        or owner_pid <= 0
    ):
        raise ValueError("lock ownership schema or PID is invalid")
    if record["hostname"] != hostname:
        raise ValueError("lock ownership hostname differs from the current host")
    if record["run_fingerprint"] != expected_run_fingerprint:
        raise ValueError("lock ownership fingerprint differs from the expected fingerprint")
    scope = record["scope"]
    valid_scope = scope == "executor" if lock_path.name == "executor.lock" else (
        isinstance(scope, str) and scope.startswith("checkpoint:") and len(scope) > 11
    )
    if not valid_scope:
        raise ValueError("lock ownership scope does not match its lock filename")


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
