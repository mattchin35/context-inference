"""Trusted local archive replay and scientific provenance contracts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.communication_subspace.computation import compute_communication_subspace
from src.neural_analysis.communication_subspace.configuration import CommunicationConfig
from src.neural_analysis.communication_subspace.preparation import load_session_inputs
from src.neural_analysis.communication_subspace.results import (
    CommunicationResult,
    discover_results,
    load_result,
    save_result,
)


ROLES = {"PFC": "A", "HPC": "B"}


@pytest.fixture
def communication_result(synthetic_communication_session):
    """Return a real 500-ms synthetic fit and full prepared identity without writes."""
    prepared = load_session_inputs(
        CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=500),
    )
    computation = compute_communication_subspace(prepared, seed=0)
    return CommunicationResult(
        session_id=prepared.session_id,
        time_origin_s=prepared.time_origin_s,
        choices=prepared.choices,
        selected_trials=prepared.selected_trials,
        choice_positions=prepared.choice_positions,
        unit_ids=prepared.unit_ids,
        bin_edges_s=prepared.bin_edges_s,
        probe_roles=prepared.probe_roles,
        computation=computation,
        meta={"configuration": prepared.configuration, "generator": "test_results"},
    )


def test_named_archive_round_trip_preserves_scores_axes_and_compact_fits(
    communication_result, tmp_path,
):
    """NPZ replay restores table dtypes, all compact factors, and original identities."""
    path = save_result(tmp_path / "result.npz", communication_result)
    with np.load(path, allow_pickle=True) as archive:
        assert "meta" in archive.files
        assert isinstance(archive["meta"].item(), dict)
        assert {"choice_positions", "bin_edges_s", "fold_scores", "inner_scores"} <= set(archive.files)
        assert not any("prediction_tensor" in name or name == "counts" for name in archive.files)
    loaded = load_result(path)

    assert loaded.session_id == communication_result.session_id
    assert loaded.time_origin_s == communication_result.time_origin_s
    assert loaded.unit_ids == communication_result.unit_ids
    assert loaded.probe_roles == communication_result.probe_roles
    np.testing.assert_array_equal(loaded.choice_positions, communication_result.choice_positions)
    np.testing.assert_array_equal(loaded.bin_edges_s, communication_result.bin_edges_s)
    pd.testing.assert_frame_equal(loaded.choices, communication_result.choices)
    pd.testing.assert_frame_equal(loaded.selected_trials, communication_result.selected_trials)
    for field in ("fold_scores", "inner_scores", "draw_scores", "curve_scores"):
        pd.testing.assert_frame_equal(getattr(loaded.computation, field), getattr(communication_result.computation, field))
    assert loaded.computation.status == communication_result.computation.status
    assert len(loaded.computation.draws) == len(communication_result.computation.draws) == 5
    assert len(loaded.computation.splits) == len(communication_result.computation.splits) == 5
    assert len(loaded.computation.fit_records) == len(communication_result.computation.fit_records)
    for actual, expected in zip(loaded.computation.fit_records, communication_result.computation.fit_records):
        assert (actual.draw, actual.direction, actual.fold, actual.alpha) == (
            expected.draw, expected.direction, expected.fold, expected.alpha,
        )
        assert actual.source_unit_ids == expected.source_unit_ids
        assert actual.target_unit_ids == expected.target_unit_ids
        np.testing.assert_array_equal(actual.source_indices, expected.source_indices)
        np.testing.assert_array_equal(actual.target_indices, expected.target_indices)
        np.testing.assert_array_equal(actual.source_transform.means, expected.source_transform.means)
        np.testing.assert_array_equal(actual.target_transform.variable_mask, expected.target_transform.variable_mask)
        np.testing.assert_array_equal(actual.decomposition.ridge_coefficients,
                                      expected.decomposition.ridge_coefficients)
        np.testing.assert_array_equal(actual.decomposition.right_vectors,
                                      expected.decomposition.right_vectors)


def test_saved_result_reopens_without_original_source_files(communication_result, tmp_path):
    """Replay and discovery require only the trusted result file and its metadata."""
    session_root = tmp_path / "archive_only"
    destination = session_root / "communication_subspace" / "results" / "saved.npz"
    destination.parent.mkdir(parents=True)
    save_result(destination, communication_result)
    info = discover_results(session_root, communication_result.meta["configuration"]["configuration_id"])
    assert len(info) == 1
    assert info[0].path == destination
    assert info[0].configuration["configuration_id"] == communication_result.meta["configuration"]["configuration_id"]
    loaded = load_result(destination)
    assert loaded.computation.status == communication_result.computation.status


def test_save_never_overwrites_and_bad_axes_are_rejected(communication_result, tmp_path):
    """Existing paths and malformed saved choice/bin identity cannot masquerade as results."""
    path = save_result(tmp_path / "result.npz", communication_result)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_result(path, communication_result)
    assert path.read_bytes() == original

    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["choice_positions"] = np.array([999], dtype=np.int64)
    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, **arrays)
    with pytest.raises(ValueError, match="choice|axis|shape|position"):
        load_result(malformed)
    assert discover_results(tmp_path, communication_result.meta["configuration"]["configuration_id"]) == ()


def test_tampered_fit_factor_dimensions_are_rejected(communication_result, tmp_path):
    """A coefficient axis inconsistent with selected source/target masks fails load."""
    path = save_result(tmp_path / "result.npz", communication_result)
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    coefficient_keys = [name for name in arrays if name.endswith("ridge_coefficients")]
    assert coefficient_keys
    arrays[coefficient_keys[0]] = np.zeros((1, 1))
    malformed = tmp_path / "bad_fit.npz"
    np.savez(malformed, **arrays)
    with pytest.raises(ValueError, match="fit|coefficient|shape|axis"):
        load_result(malformed)


def test_corrupt_archive_is_reported_and_missing_required_field_rejected(
    communication_result, tmp_path,
):
    """Discovery warns on corrupt bytes; load reports absent required members cleanly."""
    result_root = tmp_path / "communication_subspace" / "results"
    result_root.mkdir(parents=True)
    broken = result_root / "broken.npz"
    broken.write_bytes(b"not an npz archive")
    with pytest.warns(RuntimeWarning, match="corrupt|invalid|unreadable"):
        assert discover_results(tmp_path) == ()
    with pytest.raises(ValueError, match="corrupt|invalid|unreadable|archive"):
        load_result(broken)

    valid = save_result(result_root / "valid.npz", communication_result)
    with np.load(valid, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "bin_edges_s"}
    missing = result_root / "missing.npz"
    np.savez(missing, **arrays)
    with pytest.raises(ValueError, match="missing|bin_edges_s|archive"):
        load_result(missing)
