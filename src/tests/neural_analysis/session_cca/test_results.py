"""CCA saved-array, source-independent loading, and metadata-discovery contracts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.population.cca import fit_paired_cca, transform_cca_activity
from src.neural_analysis.population.session_cca_io import CCAResultInfo, SessionCCAResult
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as timeline_session_fixture,
)


synthetic_timeline_session = timeline_session_fixture


@pytest.fixture
def cca_result(synthetic_timeline_session):
    """Return a fitted three-choice, eight-bin record using the existing seeded sources."""
    inputs = cca_io.load_cca_inputs(synthetic_timeline_session, {"PFC": "A", "HPC": "B"}, 0.5)
    fitted = fit_paired_cca(inputs.counts["PFC"], inputs.counts["HPC"],
                            inputs.unit_ids["PFC"], inputs.unit_ids["HPC"])
    return SessionCCAResult(
        session_id=inputs.session_id, time_origin_s=inputs.time_origin_s,
        choices=inputs.choices, events=inputs.events,
        choice_positions=inputs.choice_positions, coverage_positions=inputs.coverage_positions,
        bin_edges_s=inputs.bin_edges_s, probe_roles=inputs.probe_roles, fit=fitted,
        unavailable_reason=None, meta=dict(inputs.configuration),
    )


def test_round_trip_preserves_tables_axes_transforms_and_score_replay(cca_result, tmp_path):
    """NPZ reload preserves UTC/elapsed behavior and all (W,B,K)/(U,K) arrays exactly."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)

    loaded = cca_io.load_cca_result(path)

    assert loaded.session_id == cca_result.session_id
    assert loaded.time_origin_s == cca_result.time_origin_s
    assert loaded.probe_roles == cca_result.probe_roles
    assert loaded.unavailable_reason is None
    pd.testing.assert_frame_equal(loaded.choices, cca_result.choices)
    pd.testing.assert_frame_equal(loaded.events, cca_result.events)
    np.testing.assert_array_equal(loaded.choice_positions, cca_result.choice_positions)
    np.testing.assert_array_equal(loaded.bin_edges_s, cca_result.bin_edges_s)
    np.testing.assert_array_equal(loaded.fit.correlations, cca_result.fit.correlations)
    np.testing.assert_array_equal(loaded.fit.joint_signs, cca_result.fit.joint_signs)
    assert loaded.fit.n_observations == cca_result.fit.n_observations
    assert loaded.fit.color_limit == cca_result.fit.color_limit
    assert loaded.fit.diagnostics == cca_result.fit.diagnostics
    for region in ("PFC", "HPC"):
        np.testing.assert_array_equal(loaded.coverage_positions[region], cca_result.coverage_positions[region])
        np.testing.assert_array_equal(loaded.fit.scores[region], cca_result.fit.scores[region])
        original = cca_result.fit.transforms[region]
        transform = loaded.fit.transforms[region]
        for field in ("input_unit_ids", "retained_unit_ids", "omitted_unit_ids", "rank", "rank_cutoff"):
            assert getattr(transform, field) == getattr(original, field)
        for field in ("count_mean", "count_scale", "coefficients", "score_offset", "singular_values"):
            np.testing.assert_array_equal(getattr(transform, field), getattr(original, field))
        # Replay needs only saved transforms and a new count tensor, never raw spikes.
        counts = np.full((2, 3, len(transform.input_unit_ids)), 4.0)
        np.testing.assert_array_equal(
            transform_cca_activity(counts, transform.input_unit_ids, transform),
            transform_cca_activity(counts, original.input_unit_ids, original),
        )


def test_named_arrays_and_metadata_explain_units_axes_and_provenance(cca_result, tmp_path):
    """A trusted NPZ exposes named scientific arrays and a complete dictionary-valued meta."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)

    with np.load(path, allow_pickle=True) as archive:
        assert {"choices", "events", "choice_positions", "bin_edges_s", "correlations", "joint_signs",
                "pfc_scores", "hpc_scores", "pfc_coefficients", "pfc_score_offset",
                "pfc_count_mean", "pfc_count_scale", "pfc_coverage_positions"}.issubset(archive.files)
        meta = archive["meta"].item()
    assert isinstance(meta, dict)
    assert meta["analysis_version"] == "session-timeline-cca-v1"
    assert meta["schema_version"] == 1
    assert meta["numerical_version"] == cca_result.meta["numerical_version"]
    assert meta["generator"] == "src.neural_analysis.population.session_cca_io.save_cca_result"
    assert meta["random_seed"] is None
    assert meta["units"]["absolute_time"] == "UTC Unix seconds"
    assert meta["units"]["relative_time"] == "seconds"
    assert meta["units"]["scores"] == "canonical session SD"
    assert meta["axes"]["scores"] == ["choice_window", "time_bin", "canonical_pair"]
    assert meta["axes"]["coefficients"] == ["retained_unit", "canonical_pair"]
    assert meta["completed_utc"] and meta["run_id"]
    assert meta["configuration_id"] == cca_result.meta["configuration_id"]


def test_mixed_missing_and_nonconsecutive_trial_ids_remain_behavioral_data(cca_result, tmp_path):
    """Object trial IDs, including missing/duplicate values, gain no new scientific filter."""
    choices = cca_result.choices.copy()
    choices["trial_id"] = pd.Series(["legacy", None, "legacy"], dtype=object)
    choices["row_position"] = [2, 6, 20]
    result = replace(cca_result, choices=choices)

    loaded = cca_io.load_cca_result(cca_io.save_cca_result(tmp_path / "results.npz", result))

    pd.testing.assert_frame_equal(loaded.choices, choices)
    np.testing.assert_array_equal(loaded.choice_positions, [0, 1, 2])


def test_saving_never_overwrites_or_mutates_inputs(cca_result, tmp_path):
    """Saving enriches a metadata copy and refuses an existing destination."""
    before_meta = deepcopy(cca_result.meta)
    before_scores = {region: values.copy() for region, values in cca_result.fit.scores.items()}
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    before_bytes = path.read_bytes()

    with pytest.raises(FileExistsError):
        cca_io.save_cca_result(path, cca_result)

    assert path.read_bytes() == before_bytes
    assert cca_result.meta == before_meta
    for region in before_scores:
        np.testing.assert_array_equal(cca_result.fit.scores[region], before_scores[region])


def test_behavior_only_record_cannot_publish_a_completed_fit(cca_result, tmp_path):
    """UI-only unavailable states do not become discoverable completed numerical products."""
    result = replace(cca_result, fit=None, unavailable_reason="HPC unavailable")
    path = tmp_path / "results.npz"

    with pytest.raises(ValueError):
        cca_io.save_cca_result(path, result)
    assert not path.exists()


def test_invalid_fit_cannot_be_published(cca_result, tmp_path):
    """Nonfinite canonical values are rejected before an immutable result appears."""
    scores = {region: values.copy() for region, values in cca_result.fit.scores.items()}
    scores["PFC"][0, 0, 0] = np.nan
    result = replace(cca_result, fit=replace(cca_result.fit, scores=scores))
    path = tmp_path / "results.npz"

    with pytest.raises(ValueError):
        cca_io.save_cca_result(path, result)
    assert not path.exists()


@pytest.mark.parametrize("member", ["pfc_scores", "hpc_scores", "pfc_coefficients",
                                     "joint_signs", "choice_positions", "bin_edges_s"])
def test_loaded_array_shapes_and_shared_axes_must_match_metadata(cca_result, tmp_path, member):
    """Malformed named arrays are rejected rather than reshaped or relabeled during loading."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    arrays = _archive_arrays(path)
    if member == "pfc_scores":
        arrays[member] = arrays[member][:-1]
    elif member == "hpc_scores":
        arrays[member] = arrays[member][:, :-1]
    elif member in ("pfc_coefficients", "joint_signs"):
        arrays[member] = arrays[member][:-1]
    elif member == "choice_positions":
        arrays[member] = np.array([0, 1, 500])
    else:
        arrays[member][1] += 0.01
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError):
        cca_io.load_cca_result(path)


@pytest.mark.parametrize("member", ["pfc_scores", "pfc_count_mean", "correlations"])
def test_loading_rejects_nonfinite_scientific_arrays(cca_result, tmp_path, member):
    """NaN scientific measurements/scales cannot become a valid saved paired result."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    arrays = _archive_arrays(path)
    arrays[member].flat[0] = np.nan
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError):
        cca_io.load_cca_result(path)


@pytest.mark.parametrize("field,value", [("analysis_version", "future-analysis"),
                                          ("schema_version", 99), ("numerical_version", "future-numerics")])
def test_loading_rejects_unsupported_versions(cca_result, tmp_path, field, value):
    """Unsupported scientific/array/normalization conventions require an explicit loader update."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    arrays = _archive_arrays(path)
    arrays["meta"].item()[field] = value
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError):
        cca_io.load_cca_result(path)


def test_loading_rejects_duplicate_unit_ids(cca_result, tmp_path):
    """Unit IDs uniquely map every stored coefficient row; behavior trial IDs need not."""
    path = cca_io.save_cca_result(tmp_path / "results.npz", cca_result)
    arrays = _archive_arrays(path)
    arrays["meta"].item()["regions"]["PFC"]["input_unit_ids"] = ["A:7", "A:7"]
    np.savez_compressed(path, **arrays)

    with pytest.raises(ValueError):
        cca_io.load_cca_result(path)


def test_discovery_reads_metadata_only(cca_result, synthetic_timeline_session, monkeypatch):
    """Availability discovery returns small headers without decompressing scientific arrays."""
    root = synthetic_timeline_session.session_root
    path = _save_run(root, cca_result, "a", "2026-11-13T12:00:00Z", "a")
    with np.load(path, allow_pickle=True) as archive:
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__

    def metadata_only(archive, member):
        """Permit meta only; array access would make discovery an unintended score load."""
        assert member == "meta", f"Discovery read scientific member {member}"
        return original_getitem(archive, member)

    monkeypatch.setattr(archive_type, "__getitem__", metadata_only)

    headers = cca_io.discover_cca_results(root, cca_result.meta["configuration_id"])

    assert len(headers) == 1 and isinstance(headers[0], CCAResultInfo)
    assert headers[0].path == path
    assert headers[0].configuration["configuration_id"] == cca_result.meta["configuration_id"]
    assert headers[0].run_id == "a"


def test_discovery_orders_completion_run_and_path_independently_of_mtime(cca_result, synthetic_timeline_session):
    """Newest UTC completion wins; equal time/run ties use relative path, never filesystem mtime."""
    root = synthetic_timeline_session.session_root
    oldest = _save_run(root, cca_result, "z", "2026-11-13T11:00:00Z", "z")
    earlier_run = _save_run(root, cca_result, "y", "2026-11-13T12:00:00Z", "a")
    earlier_path = _save_run(root, cca_result, "a", "2026-11-13T12:00:00Z", "b")
    newest = _save_run(root, cca_result, "b", "2026-11-13T12:00:00Z", "b")
    for index, path in enumerate((newest, earlier_path, earlier_run, oldest)):
        os.utime(path, ns=(1_700_000_000_000_000_000 + index, 1_700_000_000_000_000_000 + index))

    headers = cca_io.discover_cca_results(root)

    assert tuple(header.path for header in headers) == (newest, earlier_path, earlier_run, oldest)
    assert cca_io.discover_cca_results(root, "different-configuration") == ()


def test_corrupt_newest_arrays_warn_and_fall_back_to_older_compatible_fit(cca_result, synthetic_timeline_session):
    """Metadata-only selection is followed by validation and fallback, not silent loss of results."""
    root = synthetic_timeline_session.session_root
    older = _save_run(root, cca_result, "old", "2026-11-13T11:00:00Z", "old")
    newest = _save_run(root, cca_result, "new", "2026-11-13T12:00:00Z", "new")
    arrays = _archive_arrays(newest)
    arrays["hpc_scores"] = arrays["hpc_scores"][:, :-1]
    np.savez_compressed(newest, **arrays)

    with pytest.warns(Warning):
        selected = cca_io.load_latest_cca_result(root, cca_result.meta["configuration_id"])

    assert selected is not None
    header, result = selected
    assert header.path == older
    assert result.meta["run_id"] == "old"
    assert cca_io.load_latest_cca_result(root, "absent-configuration") is None


def test_discovery_skips_interrupted_or_unsupported_files(cca_result, synthetic_timeline_session):
    """Temporary, incomplete and unsupported outputs do not hide a completed compatible run."""
    root = synthetic_timeline_session.session_root
    valid = _save_run(root, cca_result, "valid", "2026-11-13T11:00:00Z", "valid")
    partial = root / "analysis_runs/session_timeline_cca_partial/results.npz"
    partial.parent.mkdir()
    partial.write_bytes(b"incomplete archive")
    shutil.copyfile(valid, partial.with_name("results.temporary.npz"))
    unsupported = _save_run(root, cca_result, "unsupported", "2026-11-13T12:00:00Z", "unsupported")
    arrays = _archive_arrays(unsupported)
    arrays["meta"].item()["schema_version"] = 99
    np.savez_compressed(unsupported, **arrays)

    headers = cca_io.discover_cca_results(root, cca_result.meta["configuration_id"])

    assert tuple(header.path for header in headers) == (valid,)


def test_interrupted_write_does_not_publish_result(cca_result, synthetic_timeline_session, monkeypatch):
    """A failed NPZ write may leave a log/temp but never a completed target product."""
    path = synthetic_timeline_session.session_root / "analysis_runs/session_timeline_cca_interrupted/results.npz"
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(np, "savez_compressed", _failed_write)

    with pytest.raises(OSError):
        cca_io.save_cca_result(path, cca_result)

    assert not path.exists()
    assert cca_io.discover_cca_results(synthetic_timeline_session.session_root) == ()


def test_copied_archive_loads_after_original_raw_sources_are_removed(cca_result, synthetic_timeline_session, tmp_path):
    """Historical loading uses stored behavior and fit despite stale absolute source paths."""
    path = cca_io.save_cca_result(tmp_path / "original.npz", cca_result)
    copied = tmp_path / "copy/results.npz"
    copied.parent.mkdir()
    shutil.copyfile(path, copied)
    for source in _source_paths(cca_result.meta["sources"]):
        source.unlink()
    assert synthetic_timeline_session.metadata_path.is_file()

    loaded = cca_io.load_cca_result(copied)

    pd.testing.assert_frame_equal(loaded.choices, cca_result.choices)
    np.testing.assert_array_equal(loaded.fit.scores["PFC"], cca_result.fit.scores["PFC"])


def _save_run(root: Path, result: SessionCCAResult, name: str, completed: str, run_id: str) -> Path:
    """Save a fixture result with explicit UTC completion and run ID; return NPZ path."""
    directory = root / "analysis_runs" / f"session_timeline_cca_{name}_{result.meta['configuration_id']}"
    directory.mkdir(parents=True)
    saved = replace(result, meta=dict(result.meta, completed_utc=completed, run_id=run_id))
    return cca_io.save_cca_result(directory / "results.npz", saved)


def _archive_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read every fixture NPZ member for deliberate corruption; return independent arrays."""
    with np.load(path, allow_pickle=True) as archive:
        return {name: archive[name] for name in archive.files}


def _source_paths(sources: dict) -> tuple[Path, ...]:
    """Return present file paths from the named source map; values have no signal units."""
    paths = []
    for source in sources.values():
        if "path" in source:
            if source["path"] is not None and not source.get("missing", False):
                paths.append(Path(source["path"]))
        else:
            paths.extend(_source_paths(source))
    return tuple(paths)


def _failed_write(*args, **kwargs):
    """Simulate an ordinary I/O interruption during NPZ output; return never."""
    raise OSError("synthetic interrupted write")
