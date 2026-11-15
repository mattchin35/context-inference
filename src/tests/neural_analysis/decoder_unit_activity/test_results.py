"""Derived selections validate published sources and never modify them."""

import hashlib
import json

import numpy as np
import pytest

from src.neural_analysis.decoder_unit_activity.results import (
    load_decoder_inputs, build_unit_selection, save_unit_selection, load_unit_selection,
    source_diagnostics,
)
from src.neural_analysis.population_decoding import results as source_results


def test_saved_selection_round_trip_source_free_and_no_overwrite(saved_decoder_run):
    """Input saved synthetic run; named-array selection reloads after raw inputs disappear."""
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in saved_decoder_run.rglob("*") if path.is_file()}
    code_before = source_results.code_identity()
    inputs = load_decoder_inputs(saved_decoder_run, "PFC")
    assert inputs["status"] == "ready" and inputs["weights"].shape[:2] == (5, 10)
    selection = build_unit_selection(inputs, 2)
    path = saved_decoder_run.parent.parent / "processed/decoder_unit_activity/test/selection.npz"
    save_unit_selection(path, selection["arrays"], selection["meta"])
    with np.load(path, allow_pickle=False) as archive:
        assert isinstance(json.loads(str(archive["meta"].item())), dict)
    for raw in (saved_decoder_run.parent.parent / "trials.csv", saved_decoder_run.parent.parent / "A/aligned.npz"):
        raw.unlink()
    loaded = load_unit_selection(path)
    np.testing.assert_equal(loaded["arrays"]["weights"], selection["arrays"]["weights"])
    np.testing.assert_equal(loaded["arrays"]["display_unit_indices"], selection["arrays"]["display_unit_indices"])
    assert loaded["meta"]["analysis_version"] == "decoder-unit-activity-v1"
    assert loaded["meta"]["randomness_used"] is False
    with pytest.raises(FileExistsError):
        save_unit_selection(path, selection["arrays"], selection["meta"])
    after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in saved_decoder_run.rglob("*") if path.is_file()}
    assert before == after and source_results.code_identity() == code_before


def test_missing_requested_cell_is_pending_without_writes(saved_decoder_run):
    """Input v2 with unpublished conditional cell; no all-trial substitution or partial artifact."""
    (saved_decoder_run / "cells/PFC_left_switch_vs_stay.npz").unlink()
    before = set(saved_decoder_run.parent.parent.rglob("*"))
    loaded = load_decoder_inputs(saved_decoder_run, "PFC")
    assert loaded["status"] == "pending" and "left_switch_vs_stay" in loaded["reason"]
    assert set(saved_decoder_run.parent.parent.rglob("*")) == before


def test_legacy_run_has_only_eight_supported_groups(legacy_population_run):
    """Input v1 fixture; absent conditional switch models remain unavailable."""
    run, _ = legacy_population_run
    loaded = load_decoder_inputs(run, "PFC")
    assert loaded["status"] == "ready"
    assert np.isnan(loaded["weights"][:, 8:]).all()


def test_missing_region_is_reported_before_coefficients(saved_decoder_run):
    """Input a preparation missing HPC with a reason; return unavailable instead of indexing absent data."""
    manifest = source_results.load_manifest(saved_decoder_run)
    prepared = source_results.load_preparation(manifest["preparation_path"])
    prepared.regions.pop("HPC")
    prepared.unavailable["HPC"] = "No regional coverage"
    loaded = load_decoder_inputs(saved_decoder_run, "HPC", prepared=prepared)
    assert loaded["status"] == "unavailable" and "coverage" in loaded["reason"]
    prepared.unavailable.pop("HPC")
    with pytest.raises(ValueError, match="region"):
        load_decoder_inputs(saved_decoder_run, "HPC", prepared=prepared)


def test_corrupt_or_changed_reference_is_not_reused(saved_decoder_run):
    """Input valid derived archive; a changed reference fails explicitly without being overwritten."""
    inputs = load_decoder_inputs(saved_decoder_run, "PFC")
    selection = build_unit_selection(inputs, 1)
    path = saved_decoder_run.parent.parent / "processed/decoder_unit_activity/test/selection.npz"
    save_unit_selection(path, selection["arrays"], selection["meta"])
    source = saved_decoder_run / "cells/PFC_context.npz"
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="reference|source"):
        load_unit_selection(path)


def test_source_identity_mismatch_fails(saved_decoder_run):
    """Input source whose declared preparation ID disagrees; no silent mixed-run join."""
    manifest_path = saved_decoder_run / "config.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identities"]["preparation_id"] = "different"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity|identities"):
        load_decoder_inputs(saved_decoder_run, "PFC")


def test_trial_support_is_not_bin_support(saved_decoder_run):
    """Input identical source labels at two bin counts; unique-trial counts stay fixed."""
    loaded = load_decoder_inputs(saved_decoder_run, "PFC")
    cell = loaded["cells"]["outcome"]
    before = source_diagnostics({"outcome": cell})["outcomes"]
    cell["meta"]["representations"]["units"]["conditional_outcomes"]["left"]["class_counts"] = [100, 200]
    after = source_diagnostics({"outcome": cell})["outcomes"]
    assert before[0]["Reward trials"] == after[0]["Reward trials"]
    assert before[0]["No-reward trials"] == after[0]["No-reward trials"]
    assert after[0]["Reward bins"] == 200 and after[0]["No-reward bins"] == 100
