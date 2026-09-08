"""RED contracts for fingerprinted prepared-phase and PPC work artifacts."""

from __future__ import annotations

from pathlib import Path

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
