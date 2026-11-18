"""Automatic saved-run selection remains explicit in provenance and strict across mice."""

import json
import os
import shutil
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.tests.behavior_analysis.test_cross_mouse_choice_windows import cohort, loader
from src.tests.test_main_multisession import load_main_module


OLD = "2026-01-01_12-00-00_000001"
NEW = "2026-01-02_12-00-00_000001"
INCOMPLETE = "2026-01-03_12-00-00_000001"


@pytest.fixture
def automatic_cohort(cohort):
    """Give each synthetic mouse two complete timestamped runs sharing trusted source NPZs."""
    root, explicit = cohort
    newest = {}
    for mouse, run in explicit.items():
        old = run.with_name(OLD)
        run.rename(old)
        new = old.with_name(NEW)
        shutil.copytree(old, new)
        os.utime(old, (2_000_000_000, 2_000_000_000))  # Modification time must not win over run timestamps.
        newest[mouse] = new
    return root, newest


def test_automatic_selection_matches_explicit_and_validates_once(automatic_cohort, monkeypatch):
    """Automatic and explicit calls give identical values; selected candidates are not reloaded."""
    from src.behavior_analysis import cross_mouse_choice_windows as module
    root, newest = automatic_cohort
    expected, explicit_meta = loader()(root, newest)
    loading = Mock(wraps=module._load_mouse_collection)
    monkeypatch.setattr(module, "_load_mouse_collection", loading)
    actual, meta = loader()(root, mice=["A", "B"])
    pd.testing.assert_frame_equal(actual, expected)
    assert loading.call_count == 2
    assert meta["selection_mode"] == "automatic"
    assert explicit_meta["selection_mode"] == "explicit"
    assert meta["collection_runs"] == {mouse: str(path) for mouse, path in newest.items()}
    assert meta["skipped_candidates"] == {"A": [], "B": []}


@pytest.mark.parametrize("problem", ["missing_file", "missing_session", "wrong_size", "invalid_json"])
def test_incomplete_newest_candidate_is_reported_and_skipped(automatic_cohort, problem):
    """An older complete roster-matching run is usable; every rejected newer run gets a reason."""
    root, newest = automatic_cohort
    bad = newest["A"].with_name(INCOMPLETE)
    shutil.copytree(newest["A"], bad)
    manifest_path = bad / "metadata.json"
    manifest = json.loads(manifest_path.read_text())
    if problem == "missing_file":
        (bad / "A_choice_window_summary.csv").unlink()
    elif problem == "missing_session":
        manifest["session_ids"].pop()
        manifest["sources"].pop()
        manifest_path.write_text(json.dumps(manifest))
    elif problem == "wrong_size":
        manifest["window_sizes"] = [10]
        manifest_path.write_text(json.dumps(manifest))
    else:
        manifest_path.write_text("{invalid")
    _, meta = loader()(root, mice=["A", "B"])
    assert meta["collection_runs"]["A"] == str(newest["A"])
    skipped = meta["skipped_candidates"]["A"]
    assert len(skipped) == 1 and skipped[0]["path"] == str(bad)
    assert skipped[0]["reason"]


def test_requested_size_can_select_a_newer_single_size_collection(automatic_cohort):
    """Completeness is evaluated against the requested sizes, rather than always requiring both."""
    root, newest = automatic_cohort
    single = newest["A"].with_name(INCOMPLETE)
    shutil.copytree(newest["A"], single)
    path = single / "metadata.json"
    meta = json.loads(path.read_text())
    meta["window_sizes"] = [10]
    path.write_text(json.dumps(meta))
    table, provenance = loader()(root, window_sizes=(10,), mice=["A", "B"])
    assert set(table.window_size) == {10}
    assert provenance["collection_runs"]["A"] == str(single)


def test_non_run_directories_are_not_candidates(automatic_cohort):
    """Unrelated directories and impossible date names cannot shadow real timestamped collections."""
    root, newest = automatic_cohort
    for name in ("exports", "latest", "2099-99-99_99-99-99_999999"):
        shutil.copytree(newest["A"], newest["A"].with_name(name))
    _, meta = loader()(root, mice=["A", "B"])
    assert meta["collection_runs"]["A"] == str(newest["A"])
    assert not meta["skipped_candidates"]["A"]


def test_no_qualifying_run_reports_mouse_candidates_and_creates_no_outputs(automatic_cohort, tmp_path):
    """Failure retains candidate-specific reasons instead of emitting a partial cohort export."""
    root, newest = automatic_cohort
    for run in newest["B"].parent.iterdir():
        (run / "B_choice_window_summary.csv").unlink()
    output = tmp_path / "exports"
    with pytest.raises(ValueError) as caught:
        load_main_module().run_cross_mouse_choice_window_agreement_curves(root, output, mice=["A", "B"])
    assert "B" in str(caught.value) and OLD in str(caught.value) and NEW in str(caught.value)
    assert not output.exists()


def test_missing_mouse_directory_is_an_error(automatic_cohort):
    """Discovery never silently drops a requested mouse that lacks saved collections."""
    root, _ = automatic_cohort
    with pytest.raises(ValueError, match="missing_mouse"):
        loader()(root, mice=["A", "missing_mouse"])


@pytest.mark.parametrize("options", [{}, {"mice": []}, {"mice": ["A", "A"]},
                                   {"mice": "A"}, {"collection_runs": {}},
                                   {"collection_runs": {}, "mice": ["A"]}])
def test_selection_mode_validation(automatic_cohort, options):
    """Exactly one nonempty explicit/discovery mode is required, with distinct mouse IDs."""
    root, _ = automatic_cohort
    with pytest.raises(ValueError):
        loader()(root, **options)


def test_both_nonempty_modes_are_rejected(automatic_cohort):
    """An explicit mapping and mouse-list discovery cannot ambiguously override one another."""
    root, newest = automatic_cohort
    with pytest.raises(ValueError):
        loader()(root, newest, mice=["A", "B"])


def test_explicit_bad_run_does_not_fall_back_to_discovery(automatic_cohort):
    """Explicit selection remains authoritative even when another valid run exists."""
    root, newest = automatic_cohort
    (newest["A"] / "A_choice_window_summary.csv").unlink()
    with pytest.raises(ValueError, match="A_choice_window_summary"):
        loader()(root, newest)


def test_version_conflict_does_not_downgrade_to_older_compatible_runs(automatic_cohort):
    """Choose the newest valid run per mouse, then stop if their numerical versions conflict."""
    root, newest = automatic_cohort
    path = newest["B"] / "metadata.json"
    manifest = json.loads(path.read_text())
    manifest["analysis_version"] = "newer_version"
    replacement = []
    for index, source in enumerate(manifest["sources"]):
        with np.load(source, allow_pickle=True) as saved:
            arrays = {key: saved[key] for key in saved.files}
        source_meta = arrays["meta"].item()
        source_meta["analysis_version"] = "newer_version"
        arrays["meta"] = source_meta
        destination = newest["B"] / f"source_{index}.npz"
        np.savez_compressed(destination, **arrays)
        replacement.append(str(destination))
    manifest["sources"] = replacement
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError) as caught:
        loader()(root, mice=["A", "B"])
    message = str(caught.value)
    assert "version" in message and "newer_version" in message and str(newest["B"]) in message


def test_automatic_entry_point_records_and_prints_selections(automatic_cohort, tmp_path, capsys):
    """The quick-analysis entry point saves its automatic selection mode and resolved paths."""
    root, newest = automatic_cohort
    main = load_main_module()
    table, directory = main.run_cross_mouse_choice_window_agreement_curves(
        root, tmp_path / "exports", window_sizes=(10,), mice=["A", "B"])
    metadata = json.loads((directory / "metadata.json").read_text())
    provenance = metadata["source_provenance"]
    assert provenance["selection_mode"] == "automatic"
    assert provenance["collection_runs"] == {mouse: str(path) for mouse, path in newest.items()}
    assert metadata["mice"] == ["A", "B"] and set(table.window_size) == {10}
    output = capsys.readouterr().out
    assert all(str(path) in output for path in newest.values())
    assert len(list(directory.glob("*.png"))) == 1
