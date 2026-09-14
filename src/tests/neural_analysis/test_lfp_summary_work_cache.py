"""RED contracts for fingerprinted prepared-phase and PPC work artifacts."""

from __future__ import annotations

import json
import os
import inspect
from pathlib import Path
import socket

import numpy as np
import pytest

from src.neural_analysis import lfp_summary_work_cache as work_cache


def _metadata(fingerprint: str = "phase-a") -> dict[str, object]:
    """Return complete JSON-safe prepared-phase work metadata for one synthetic run.

    Parameters
    ----------
    fingerprint : str
        Exact source/scientific/representation work identity with no units.

    Returns
    -------
    dict[str, object]
        JSON-compatible metadata for complex64 phase, Boolean validity, seconds,
        Hz, and categorical axes. It contains no missing required field.
    """
    return {
        "generator": "test_prepared_phase",
        "analysis_version": "test-v1",
        "schema_version": "1",
        "source_fingerprint": "source-a",
        "scientific_fingerprint": "science-a",
        "representation_fingerprint": fingerprint,
        "execution_settings": {"unit_block_size": 2},
        "axes": ["site", "frequency", "trial", "time"],
        "shapes": {
            "phase": [1, 2, 3, 4],
            "valid": [1, 2, 3, 4],
            "site_trial_valid": [1, 3],
            "site_trial_exclusion_reason": [1, 3],
        },
        "dtypes": {
            "phase": "complex64",
            "valid": "bool",
            "site_trial_valid": "bool",
            "site_trial_exclusion_reason": "<U32",
        },
        "units": {"phase": "dimensionless", "relative_time_s": "s", "frequency_hz": "Hz"},
    }


def _axes() -> dict[str, np.ndarray]:
    """Return exact prepared-phase coordinate arrays with no missing values.

    Returns
    -------
    dict[str, numpy.ndarray]
        Stable site IDs, Hz frequencies, trial rows, and relative seconds arrays
        matching phase axes `(site, frequency, trial, time)`.
    """
    return {
        "site_ids": np.array(["PFC"], dtype="<U8"),
        "frequency_hz": np.array([8.0, 40.0]),
        "trial_indices": np.array([2, 5, 7], dtype=np.int64),
        "relative_time_s": np.array([-1.0, -0.5, 0.0, 0.5]),
        "site_trial_valid": np.array([[True, True, False]], dtype=bool),
        "site_trial_exclusion_reason": np.array([["", "", "missing_trace"]], dtype="<U32"),
    }


def _phase_and_valid() -> tuple[np.ndarray, np.ndarray]:
    """Return exact complex64 phase and Boolean validity on four named axes.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Complex64 unit phase and Boolean validity, each shape `(site, frequency,
        trial, time)`. All entries are finite and validity is explicit.
    """
    phase = np.ones((1, 2, 3, 4), dtype=np.complex64)
    valid = np.ones(phase.shape, dtype=bool)
    valid[0, 1, 2, 3] = False
    phase[~valid] = 0.0j
    return phase, valid


def test_prepared_phase_cold_write_and_mmap_warm_load_preserve_exact_arrays(
    tmp_path: Path,
) -> None:
    """A completed prepared cache round-trips complex64/bool axes through mmap files."""
    phase, valid = _phase_and_valid()
    written = work_cache.write_prepared_phase_cache(
        tmp_path,
        _metadata(),
        phase,
        valid,
        _axes(),
    )

    loaded = work_cache.load_prepared_phase_cache(tmp_path, _metadata())

    assert written == tmp_path / "prepared_phase" / "phase-a"
    assert loaded is not None
    assert isinstance(loaded.phase, np.memmap)
    assert isinstance(loaded.valid, np.memmap)
    assert loaded.phase.dtype == np.dtype(np.complex64)
    assert loaded.valid.dtype == np.dtype(bool)
    np.testing.assert_array_equal(loaded.phase, phase)
    np.testing.assert_array_equal(loaded.valid, valid)
    for name, expected in _axes().items():
        np.testing.assert_array_equal(loaded.axes[name], expected)
    assert loaded.axes["site_trial_valid"].dtype == np.dtype(bool)
    assert loaded.axes["site_trial_exclusion_reason"].dtype.kind == "U"


@pytest.mark.parametrize(
    "field,value",
    (
        ("source_fingerprint", "other-source"),
        ("scientific_fingerprint", "other-science"),
        ("analysis_version", "test-v2"),
        ("execution_settings", {"unit_block_size": 3}),
        ("schema_version", "2"),
        ("representation_fingerprint", "phase-b"),
        ("axes", ["site", "trial", "frequency", "time"]),
        ("shapes", {"phase": [1, 2, 4, 4], "valid": [1, 2, 4, 4]}),
        ("dtypes", {"phase": "complex128", "valid": "bool"}),
    ),
)
def test_prepared_phase_metadata_mismatch_rejects_reuse(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    """Every source/config/schema/axis/shape/dtype mismatch rejects warm reuse."""
    phase, valid = _phase_and_valid()
    metadata = _metadata()
    work_cache.write_prepared_phase_cache(tmp_path, metadata, phase, valid, _axes())
    expected = dict(metadata)
    expected[field] = value

    assert work_cache.load_prepared_phase_cache(tmp_path, expected) is None


def test_incomplete_or_atomic_write_failure_never_looks_reusable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completion-last transaction rejects interrupted work and leaves no complete marker."""
    phase, valid = _phase_and_valid()
    original_replace = work_cache.os.replace
    calls = 0

    def fail_before_completion(source: Path, destination: Path) -> None:
        """Fail one atomic replacement before any completion marker can publish."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(work_cache.os, "replace", fail_before_completion)
    with pytest.raises(OSError, match="injected replace failure"):
        work_cache.write_prepared_phase_cache(tmp_path, _metadata(), phase, valid, _axes())

    assert work_cache.load_prepared_phase_cache(tmp_path, _metadata()) is None
    assert not list(tmp_path.rglob("complete.json"))


def test_complete_marker_atomic_failure_never_publishes_prepared_phase_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure while replacing completion JSON leaves data files explicitly incomplete."""
    phase, valid = _phase_and_valid()
    original_replace = work_cache.os.replace

    def fail_completion_replace(source: Path, destination: Path) -> None:
        """Reject only the atomic replacement that would publish ``complete.json``."""
        if destination.name == "complete.json":
            raise OSError("injected completion-marker failure")
        original_replace(source, destination)

    monkeypatch.setattr(work_cache.os, "replace", fail_completion_replace)
    with pytest.raises(OSError, match="completion-marker failure"):
        work_cache.write_prepared_phase_cache(tmp_path, _metadata(), phase, valid, _axes())

    assert work_cache.load_prepared_phase_cache(tmp_path, _metadata()) is None
    assert not list(tmp_path.rglob("complete.json"))


def test_failed_prepared_phase_rewrite_never_pairs_new_files_with_old_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rewrite leaves the old cache intact or explicitly unavailable, never mixed."""
    old_phase, valid = _phase_and_valid()
    written = work_cache.write_prepared_phase_cache(tmp_path, _metadata(), old_phase, valid, _axes())
    new_phase = np.full(old_phase.shape, 2.0 + 0.0j, dtype=np.complex64)
    original_replace = work_cache.os.replace

    def fail_completion_replace(source: Path, destination: Path) -> None:
        """Fail only the final completion-marker replacement of the rewrite transaction."""
        if destination.name == "complete.json":
            raise OSError("injected prepared rewrite completion failure")
        original_replace(source, destination)

    monkeypatch.setattr(work_cache.os, "replace", fail_completion_replace)
    with pytest.raises(OSError, match="prepared rewrite completion failure"):
        work_cache.write_prepared_phase_cache(tmp_path, _metadata(), new_phase, valid, _axes())

    loaded = work_cache.load_prepared_phase_cache(tmp_path, _metadata())
    assert loaded is None or np.array_equal(loaded.phase, old_phase)
    assert written.exists()


def test_prepared_phase_loader_rejects_incomplete_and_pickle_bearing_axes(
    tmp_path: Path,
) -> None:
    """Unmarked caches and object-array axes cannot become reusable work inputs."""
    phase, valid = _phase_and_valid()
    written = work_cache.write_prepared_phase_cache(tmp_path, _metadata(), phase, valid, _axes())
    (written / "complete.json").unlink()
    assert work_cache.load_prepared_phase_cache(tmp_path, _metadata()) is None

    written = work_cache.write_prepared_phase_cache(tmp_path, _metadata(), phase, valid, _axes())
    np.savez(written / "axes.npz", object_axis=np.array([{"unsafe": True}], dtype=object))
    assert work_cache.load_prepared_phase_cache(tmp_path, _metadata()) is None


def test_writer_lock_stale_and_incomplete_artifacts_are_never_shared(tmp_path: Path) -> None:
    """An exact cache lock blocks another writer; stale/incomplete paths remain unreadable."""
    phase, valid = _phase_and_valid()
    cache_path = tmp_path / "prepared_phase" / "phase-a"
    cache_path.mkdir(parents=True)
    (cache_path / "writer.lock").write_text("other writer", encoding="ascii")

    with pytest.raises(FileExistsError):
        work_cache.write_prepared_phase_cache(tmp_path, _metadata(), phase, valid, _axes())
    assert work_cache.load_prepared_phase_cache(tmp_path, _metadata()) is None


def test_ppc_checkpoints_validate_completion_and_cleanup_is_exact_fingerprint(
    tmp_path: Path,
) -> None:
    """Checkpoint reuse and cleanup are isolated to one exact run fingerprint."""
    first_metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    second_metadata = {**_metadata("run-b"), "run_fingerprint": "run-b"}
    first_root = tmp_path / "ppc" / "run-a"
    second_root = tmp_path / "ppc" / "run-b"
    arrays = {"ppc": np.array([[0.1, 0.2]], dtype=float)}

    work_cache.write_ppc_checkpoint(first_root, "block-000", arrays, first_metadata)
    work_cache.write_ppc_checkpoint(second_root, "block-000", arrays, second_metadata)
    loaded = work_cache.load_valid_ppc_checkpoints(first_root, first_metadata)

    assert len(loaded) == 1
    assert loaded[0].block_id == "block-000"
    np.testing.assert_array_equal(loaded[0].arrays["ppc"], arrays["ppc"])
    (first_root / "blocks" / "block-000.complete.json").unlink()
    assert work_cache.load_valid_ppc_checkpoints(first_root, first_metadata) == ()
    work_cache.cleanup_ppc_run(second_root, "run-b")
    assert first_root.exists()
    assert not second_root.exists()


def test_single_ppc_checkpoint_loader_keeps_valid_siblings_independent(
    tmp_path: Path,
) -> None:
    """One block can load/reject without eagerly opening its checkpoint siblings.

    The single-block API accepts only the exact completed marker, safe NPZ, and
    run metadata.  Corrupting one sibling therefore cannot make an already
    valid sibling unavailable to a streaming grouped executor.
    """
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    first_arrays = {"ppc": np.array([[0.1]], dtype=float)}
    second_arrays = {"ppc": np.array([[0.2]], dtype=float)}
    work_cache.write_ppc_checkpoint(run_root, "block-a", first_arrays, metadata)
    work_cache.write_ppc_checkpoint(run_root, "block-b", second_arrays, metadata)

    first = work_cache.load_valid_ppc_checkpoint(run_root, "block-a", metadata)
    second = work_cache.load_valid_ppc_checkpoint(run_root, "block-b", metadata)
    assert first is not None and second is not None
    np.testing.assert_array_equal(first.arrays["ppc"], first_arrays["ppc"])
    np.testing.assert_array_equal(second.arrays["ppc"], second_arrays["ppc"])
    assert not first.arrays["ppc"].flags.writeable
    assert not second.arrays["ppc"].flags.writeable
    assert not np.shares_memory(first.arrays["ppc"], first_arrays["ppc"])

    # The streaming reader validates precisely the requested block, its marker,
    # and the complete run metadata.  A failure in one sibling cannot poison a
    # separate valid block.
    marker_path = run_root / "blocks" / "block-a.complete.json"
    marker_path.unlink()
    assert work_cache.load_valid_ppc_checkpoint(run_root, "block-a", metadata) is None
    work_cache.write_ppc_checkpoint(run_root, "block-a", first_arrays, metadata)
    marker_path.write_text(
        json.dumps({"block_id": "block-b", "run_fingerprint": "run-a"}),
        encoding="utf-8",
    )
    assert work_cache.load_valid_ppc_checkpoint(run_root, "block-a", metadata) is None
    work_cache.write_ppc_checkpoint(run_root, "block-a", first_arrays, metadata)
    assert work_cache.load_valid_ppc_checkpoint(
        run_root,
        "block-a",
        {**metadata, "scientific_fingerprint": "other-science"},
    ) is None
    assert work_cache.load_valid_ppc_checkpoint(run_root, "../block-a", metadata) is None
    marker_path.write_text(
        json.dumps({"block_id": "block-a", "run_fingerprint": "other-run"}),
        encoding="utf-8",
    )
    assert work_cache.load_valid_ppc_checkpoint(run_root, "block-a", metadata) is None
    work_cache.write_ppc_checkpoint(run_root, "block-a", first_arrays, metadata)

    (run_root / "blocks" / "block-b.npz").write_bytes(b"corrupt")
    preserved = work_cache.load_valid_ppc_checkpoint(run_root, "block-a", metadata)
    rejected = work_cache.load_valid_ppc_checkpoint(run_root, "block-b", metadata)
    assert preserved is not None
    np.testing.assert_array_equal(preserved.arrays["ppc"], first_arrays["ppc"])
    assert rejected is None
    assert work_cache.load_valid_ppc_checkpoint(run_root, "missing", metadata) is None

    work_cache.write_ppc_checkpoint(run_root, "block-b", second_arrays, metadata)
    np.savez(
        run_root / "blocks" / "block-b.npz",
        unsafe=np.array([{"pickle": True}], dtype=object),
    )
    assert work_cache.load_valid_ppc_checkpoint(run_root, "block-b", metadata) is None
    assert work_cache.load_valid_ppc_checkpoint(run_root, "block-a", metadata) is not None


@pytest.mark.parametrize(
    ("replacement", "schema", "maximum_array_bytes"),
    (
        (
            {"ppc": np.zeros((2, 2), dtype=np.float64)},
            {"ppc": (np.dtype(np.float64), (1, 1))},
            1024,
        ),
        (
            {"ppc": np.zeros((1, 1), dtype=np.float32)},
            {"ppc": (np.dtype(np.float64), (1, 1))},
            1024,
        ),
        (
            {"ppc": np.zeros((1, 1), dtype=np.float64)},
            {
                "ppc": (np.dtype(np.float64), (1, 1)),
                "required_identity": (np.dtype(np.int64), (1,)),
            },
            1024,
        ),
        (
            {"ppc": np.zeros((512, 512), dtype=np.float64)},
            {"ppc": (np.dtype(np.float64), (512, 512))},
            1024,
        ),
        (
            {
                "ppc": np.zeros((1, 1), dtype=np.float64),
                "unexpected": np.zeros((1,), dtype=np.int64),
            },
            {"ppc": (np.dtype(np.float64), (1, 1))},
            1024,
        ),
    ),
)
def test_single_checkpoint_loader_rejects_schema_or_compressed_size_before_member_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: dict[str, np.ndarray],
    schema: dict[str, tuple[np.dtype[object], tuple[int, ...]]],
    maximum_array_bytes: int,
) -> None:
    """A bounded grouped reader rejects invalid NPZ members before materializing them.

    ZIP metadata/header inspection may open the archive, but an incompatible
    dtype/shape/key set or compressed expansion beyond the exact supplied byte
    bound must not request any NumPy member array.  The legacy three-argument
    loader remains supported for unconstrained checkpoint callers.
    """
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    work_cache.write_ppc_checkpoint(
        run_root,
        "block-a",
        {"ppc": np.zeros((1, 1), dtype=np.float64)},
        metadata,
    )
    np.savez_compressed(run_root / "blocks" / "block-a.npz", **replacement)
    signature = inspect.signature(work_cache.load_valid_ppc_checkpoint)
    assert signature.parameters["expected_array_schema"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["maximum_array_bytes"].kind is inspect.Parameter.KEYWORD_ONLY
    original_load = work_cache.np.load
    member_names: list[str] = []

    class ArchiveWithoutMembers:
        """Allow archive metadata but fail any attempted NumPy member materialization."""

        def __init__(self, archive: object) -> None:
            self._archive = archive

        def __enter__(self) -> "ArchiveWithoutMembers":
            self._archive.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._archive.__exit__(*args)

        @property
        def files(self) -> object:
            return self._archive.files

        def __getitem__(self, name: str) -> np.ndarray:
            member_names.append(name)
            raise AssertionError("bounded loader materialized invalid NPZ member")

    def metadata_only_load(*args: object, **kwargs: object) -> ArchiveWithoutMembers:
        """Wrap NumPy's archive object while retaining only its name metadata."""
        return ArchiveWithoutMembers(original_load(*args, **kwargs))

    monkeypatch.setattr(work_cache.np, "load", metadata_only_load)
    assert work_cache.load_valid_ppc_checkpoint(
        run_root,
        "block-a",
        metadata,
        expected_array_schema=schema,
        maximum_array_bytes=maximum_array_bytes,
    ) is None
    assert member_names == []


def test_ppc_checkpoint_writer_default_copy_and_explicit_no_copy_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default checkpoint writes retain defensive copying; bounded no-copy is explicit.

    The no-copy path is reserved for grouped execution's already-owned,
    read-only bounded summary arrays.  It does not weaken the existing public
    defensive-copy behavior and still rejects writable or pickle-bearing input.
    """
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    writer_signature = inspect.signature(work_cache.write_ppc_checkpoint)
    assert writer_signature.parameters["copy_arrays"].kind is inspect.Parameter.KEYWORD_ONLY
    assert writer_signature.parameters["copy_arrays"].default is True
    original_atomic_npz = work_cache._atomic_npz
    published_arrays: dict[str, Mapping[str, np.ndarray]] = {}

    def recording_atomic_npz(
        directory: Path,
        name: str,
        arrays: Mapping[str, np.ndarray],
    ) -> None:
        """Capture exact arrays handed to synchronous NPZ publication."""
        if directory.name == "blocks":
            published_arrays[name] = dict(arrays)
        original_atomic_npz(directory, name, arrays)

    monkeypatch.setattr(work_cache, "_atomic_npz", recording_atomic_npz)
    mutable = np.array([[0.1]], dtype=float)
    work_cache.write_ppc_checkpoint(run_root, "default", {"ppc": mutable}, metadata)
    assert not np.shares_memory(published_arrays["default.npz"]["ppc"], mutable)
    mutable[0, 0] = 9.0
    default_loaded = work_cache.load_valid_ppc_checkpoint(run_root, "default", metadata)
    assert default_loaded is not None
    np.testing.assert_array_equal(default_loaded.arrays["ppc"], np.array([[0.1]], dtype=float))

    readonly = np.array([[0.2]], dtype=float)
    readonly.setflags(write=False)
    work_cache.write_ppc_checkpoint(
        run_root,
        "no-copy",
        {"ppc": readonly},
        metadata,
        copy_arrays=False,
    )
    assert published_arrays["no-copy.npz"]["ppc"] is readonly
    assert np.shares_memory(published_arrays["no-copy.npz"]["ppc"], readonly)
    no_copy_loaded = work_cache.load_valid_ppc_checkpoint(run_root, "no-copy", metadata)
    assert no_copy_loaded is not None
    np.testing.assert_array_equal(no_copy_loaded.arrays["ppc"], readonly)
    assert not no_copy_loaded.arrays["ppc"].flags.writeable

    with pytest.raises(ValueError, match="read-only|copy"):
        work_cache.write_ppc_checkpoint(
            run_root,
            "writable-no-copy",
            {"ppc": np.array([[0.3]], dtype=float)},
            metadata,
            copy_arrays=False,
        )
    with pytest.raises(ValueError, match="copy"):
        work_cache.write_ppc_checkpoint(
            run_root,
            "nonboolean-copy-mode",
            {"ppc": readonly},
            metadata,
            copy_arrays=0,
        )
    unsafe = np.array([{"pickle": True}], dtype=object)
    unsafe.setflags(write=False)
    with pytest.raises(ValueError, match="object|pickle"):
        work_cache.write_ppc_checkpoint(
            run_root,
            "object-no-copy",
            {"unsafe": unsafe},
            metadata,
            copy_arrays=False,
        )


def test_ppc_checkpoint_rejects_metadata_mismatch_and_pickle_arrays(tmp_path: Path) -> None:
    """Checkpoint loading accepts only exact metadata and safe non-object NPZ arrays."""
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    work_cache.write_ppc_checkpoint(
        run_root,
        "block-000",
        {"ppc": np.array([[0.1]], dtype=float)},
        metadata,
    )
    mismatched = {**metadata, "scientific_fingerprint": "other-science"}
    assert work_cache.load_valid_ppc_checkpoints(run_root, mismatched) == ()
    np.savez(run_root / "blocks" / "block-000.npz", unsafe=np.array([{"x": 1}], dtype=object))
    assert work_cache.load_valid_ppc_checkpoints(run_root, metadata) == ()


def test_failed_checkpoint_rewrite_never_pairs_new_npz_with_old_completion_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed checkpoint rewrite leaves the old block intact or unavailable, never mixed."""
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    old_arrays = {"ppc": np.array([[0.1]], dtype=float)}
    work_cache.write_ppc_checkpoint(run_root, "block-000", old_arrays, metadata)
    original_replace = work_cache.os.replace

    def fail_completion_replace(source: Path, destination: Path) -> None:
        """Fail only the block completion-marker publication for the rewrite."""
        if destination.name == "block-000.complete.json":
            raise OSError("injected checkpoint rewrite completion failure")
        original_replace(source, destination)

    monkeypatch.setattr(work_cache.os, "replace", fail_completion_replace)
    with pytest.raises(OSError, match="checkpoint rewrite completion failure"):
        work_cache.write_ppc_checkpoint(
            run_root,
            "block-000",
            {"ppc": np.array([[0.9]], dtype=float)},
            metadata,
        )

    loaded = work_cache.load_valid_ppc_checkpoints(run_root, metadata)
    assert not loaded or np.array_equal(loaded[0].arrays["ppc"], old_arrays["ppc"])


def test_ppc_checkpoint_writer_lock_and_unsafe_block_ids_are_rejected(tmp_path: Path) -> None:
    """Checkpoint writers cannot share a run directory or escape its blocks directory."""
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    run_root.mkdir(parents=True)
    (run_root / "writer.lock").write_text("other writer", encoding="ascii")

    with pytest.raises(FileExistsError):
        work_cache.write_ppc_checkpoint(
            run_root,
            "block-000",
            {"ppc": np.array([[0.1]], dtype=float)},
            metadata,
        )
    (run_root / "writer.lock").unlink()
    for unsafe_block_id in ("../sibling", "block/child", "", "."):
        with pytest.raises(ValueError):
            work_cache.write_ppc_checkpoint(
                run_root,
                unsafe_block_id,
                {"ppc": np.array([[0.1]], dtype=float)},
                metadata,
            )


def test_ppc_writer_lock_is_an_exact_json_ownership_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checkpoint mutation observes a complete exact-run writer ownership record."""
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    run_root = tmp_path / "ppc" / "run-a"
    original_atomic_json = work_cache._atomic_json
    observed_record: dict[str, object] = {}

    def inspect_writer_lock(
        directory: Path,
        name: str,
        value: dict[str, object],
    ) -> None:
        """Capture the writer record before the first checkpoint JSON mutation."""
        if not observed_record:
            observed_record.update(
                json.loads((run_root / "writer.lock").read_text(encoding="utf-8"))
            )
        original_atomic_json(directory, name, value)

    monkeypatch.setattr(work_cache, "_atomic_json", inspect_writer_lock)
    work_cache.write_ppc_checkpoint(
        run_root,
        "block-000",
        {"ppc": np.array([[0.1]], dtype=float)},
        metadata,
    )

    assert observed_record == {
        "schema_version": "1",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "run_fingerprint": "run-a",
        "scope": "checkpoint:block-000",
    }


def _lock_record(
    *,
    fingerprint: str = "run-a",
    hostname: str = "profile-host",
    pid: int = 4312,
    scope: str = "executor",
) -> dict[str, object]:
    """Return one complete PPC lock ownership record without physical units."""
    return {
        "schema_version": "1",
        "pid": pid,
        "hostname": hostname,
        "run_fingerprint": fingerprint,
        "scope": scope,
    }


def _write_lock(path: Path, record: dict[str, object]) -> bytes:
    """Write canonical test lock bytes and return them for immutability checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return payload


@pytest.mark.parametrize(
    ("lock_name", "scope"),
    (("executor.lock", "executor"), ("writer.lock", "checkpoint:block-000")),
)
def test_recover_stale_lock_removes_only_exact_same_host_dead_pid_ownership(
    tmp_path: Path,
    lock_name: str,
    scope: str,
) -> None:
    """An explicit recovery removes validated dead ownership for the exact PPC run."""
    lock_path = tmp_path / "ppc" / "run-a" / lock_name
    _write_lock(lock_path, _lock_record(scope=scope))
    checked_pids: list[int] = []

    def pid_is_alive(pid: int) -> bool:
        """Record the validated local PID and report that its process is dead."""
        checked_pids.append(pid)
        return False

    recovered = work_cache.recover_stale_lock(
        lock_path,
        "run-a",
        hostname="profile-host",
        pid_is_alive=pid_is_alive,
    )

    assert recovered is True
    assert checked_pids == [4312]
    assert not lock_path.exists()


def test_recover_stale_lock_rejects_live_or_foreign_ownership_without_mutation(
    tmp_path: Path,
) -> None:
    """Live and foreign-host locks remain byte-identical and cannot be recovered."""
    lock_path = tmp_path / "ppc" / "run-a" / "executor.lock"
    live_bytes = _write_lock(lock_path, _lock_record())
    with pytest.raises(FileExistsError, match="live"):
        work_cache.recover_stale_lock(
            lock_path,
            "run-a",
            hostname="profile-host",
            pid_is_alive=lambda pid: pid == 4312,
        )
    assert lock_path.read_bytes() == live_bytes

    foreign_bytes = _write_lock(lock_path, _lock_record(hostname="other-host"))
    with pytest.raises(ValueError, match="hostname"):
        work_cache.recover_stale_lock(
            lock_path,
            "run-a",
            hostname="profile-host",
            pid_is_alive=lambda _: False,
        )
    assert lock_path.read_bytes() == foreign_bytes


@pytest.mark.parametrize(
    "invalid_payload",
    (
        b"active\n",
        b'{"schema_version":"1","pid":"4312"}\n',
        b'{"schema_version":"old","pid":4312,"hostname":"profile-host",'
        b'"run_fingerprint":"run-a","scope":"executor"}\n',
    ),
)
def test_recover_stale_lock_rejects_malformed_or_legacy_records_unchanged(
    tmp_path: Path,
    invalid_payload: bytes,
) -> None:
    """Unverifiable legacy, incomplete, or wrong-schema locks require manual review."""
    lock_path = tmp_path / "ppc" / "run-a" / "executor.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(invalid_payload)

    with pytest.raises(ValueError, match="lock"):
        work_cache.recover_stale_lock(
            lock_path,
            "run-a",
            hostname="profile-host",
            pid_is_alive=lambda _: False,
        )

    assert lock_path.read_bytes() == invalid_payload


def test_recover_stale_lock_rejects_symlink_wrong_directory_and_fingerprint(
    tmp_path: Path,
) -> None:
    """Recovery cannot follow links, escape the PPC tree, or cross run identities."""
    target = tmp_path / "target.lock"
    target_bytes = _write_lock(target, _lock_record())
    symlink = tmp_path / "ppc" / "run-a" / "executor.lock"
    symlink.parent.mkdir(parents=True)
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        work_cache.recover_stale_lock(
            symlink,
            "run-a",
            hostname="profile-host",
            pid_is_alive=lambda _: False,
        )
    assert symlink.is_symlink()
    assert target.read_bytes() == target_bytes

    wrong_directory = tmp_path / "not-ppc" / "run-a" / "executor.lock"
    wrong_directory_bytes = _write_lock(wrong_directory, _lock_record())
    with pytest.raises(ValueError, match="ppc"):
        work_cache.recover_stale_lock(
            wrong_directory,
            "run-a",
            hostname="profile-host",
            pid_is_alive=lambda _: False,
        )
    assert wrong_directory.read_bytes() == wrong_directory_bytes

    wrong_fingerprint = tmp_path / "ppc" / "run-b" / "writer.lock"
    wrong_fingerprint_bytes = _write_lock(
        wrong_fingerprint,
        _lock_record(fingerprint="run-b", scope="checkpoint:block-000"),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        work_cache.recover_stale_lock(
            wrong_fingerprint,
            "run-a",
            hostname="profile-host",
            pid_is_alive=lambda _: False,
        )
    assert wrong_fingerprint.read_bytes() == wrong_fingerprint_bytes


def test_cleanup_requires_exact_fingerprint_and_preserves_siblings(tmp_path: Path) -> None:
    """Cleanup rejects mismatched expected fingerprints without removing any run directory."""
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    sibling_metadata = {**_metadata("run-b"), "run_fingerprint": "run-b"}
    run_root = tmp_path / "ppc" / "run-a"
    sibling_root = tmp_path / "ppc" / "run-b"
    arrays = {"ppc": np.array([[0.1]], dtype=float)}
    work_cache.write_ppc_checkpoint(run_root, "block-000", arrays, metadata)
    work_cache.write_ppc_checkpoint(sibling_root, "block-000", arrays, sibling_metadata)

    with pytest.raises(ValueError):
        work_cache.cleanup_ppc_run(run_root, "run-b")

    assert run_root.exists()
    assert sibling_root.exists()


def test_cleanup_refuses_run_directory_outside_direct_ppc_parent(tmp_path: Path) -> None:
    """Cleanup cannot remove a matching fingerprint directory outside the exact PPC root."""
    metadata = {**_metadata("run-a"), "run_fingerprint": "run-a"}
    unsafe_root = tmp_path / "not_ppc" / "run-a"
    work_cache.write_ppc_checkpoint(
        unsafe_root,
        "block-000",
        {"ppc": np.array([[0.1]], dtype=float)},
        metadata,
    )

    with pytest.raises(ValueError):
        work_cache.cleanup_ppc_run(unsafe_root, "run-a")

    assert unsafe_root.exists()


def test_work_artifacts_never_register_as_final_components(tmp_path: Path) -> None:
    """Prepared/cache work files never create manifest entries or inspection components."""
    phase, valid = _phase_and_valid()
    work_cache.write_prepared_phase_cache(tmp_path, _metadata(), phase, valid, _axes())

    assert not (tmp_path / "manifest.json").exists()
    assert not list(tmp_path.rglob("power.npz"))
    assert not list(tmp_path.rglob("synchrony.npz"))
    assert not list(tmp_path.rglob("spike_phase.npz"))
