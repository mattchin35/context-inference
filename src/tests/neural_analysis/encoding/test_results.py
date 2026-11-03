"""Immutable preparation, scientific identity and read-only saved-result contracts."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.encoding.configuration import EncodingConfig
from src.neural_analysis.encoding import pipeline, preparation, results


def forbid_source_access(*args: object, **kwargs: object) -> None:
    """Reject arbitrary raw/source-loading inputs; return no arrays and no physical units."""
    raise AssertionError("Saved inspection must not load live sources or fit activity")


def test_preparation_round_trip_retains_full_behavior_basis_and_row_identity(prepared_encoding_run: Path) -> None:
    """Input run Path; saved (W,B,K) axes/scales, original UTC origin and rows survive, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    config = results.restore_config(manifest["config"])
    path = config.session_root / manifest["preparation_path"]
    loaded = results.load_preparation(path)
    session, trials, behavior = preparation.load_behavior_inputs(config)
    original, unavailable = preparation.prepare_regional_activity(config, session, trials, behavior)
    pd.testing.assert_frame_equal(loaded["trials"], trials)
    pd.testing.assert_frame_equal(loaded["behavior"].choices, behavior.choices)
    pd.testing.assert_frame_equal(loaded["behavior"].events, behavior.events)
    assert loaded["behavior"].time_origin_s == behavior.time_origin_s
    assert loaded["meta"]["preparation_id"] == manifest["identities"]["preparation_id"]
    assert loaded["meta"]["random_seed"] == 20261103
    assert loaded["meta"]["unavailable"] == unavailable
    for role, expected in original.items():
        observed = loaded["regions"][role]
        for name in ("scores", "bin_edges_s", "choice_positions"):
            np.testing.assert_array_equal(getattr(observed.timeline, name), getattr(expected.timeline, name))
        for name in ("components", "training_mean", "training_scale", "explained_variance"):
            np.testing.assert_array_equal(getattr(observed.timeline.transform, name), getattr(expected.timeline.transform, name))
        np.testing.assert_array_equal(observed.pc_ranks, expected.pc_ranks)
        np.testing.assert_array_equal(observed.window_fold_ids, expected.window_fold_ids)
        pd.testing.assert_frame_equal(observed.bin_rows, expected.bin_rows)
        for mode in expected.context_values:
            np.testing.assert_array_equal(observed.context_values[mode], expected.context_values[mode])
            np.testing.assert_array_equal(observed.context_valid[mode], expected.context_valid[mode])
    with np.load(path, allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
        assert all(not name.startswith("arr_") for name in archive.files)


def test_config_restore_does_not_require_live_inputs(prepared_encoding_run: Path) -> None:
    """Input run Path; saved portable control fields restore without source existence, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    config = results.restore_config(manifest["config"])
    config.augmented_trials.unlink()
    config.feature_parameters.unlink()
    config.session_metadata.unlink()
    restored = results.restore_config(manifest["config"])
    assert restored == config
    assert isinstance(restored.session_root, Path)
    assert restored.penalty_modes == ("same_numeric", "matched_context")


def test_preparation_and_run_initialization_never_overwrite(prepared_encoding_run: Path) -> None:
    """Input run Path; existing published NPZ/manifest remain byte-identical, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    config = results.restore_config(manifest["config"])
    path = config.session_root / manifest["preparation_path"]
    saved = results.load_preparation(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        results.save_preparation(path, saved["trials"], saved["behavior"], saved["regions"], saved["meta"])
    with pytest.raises(FileExistsError):
        results.initialize_run(config, prepared_encoding_run, path, manifest["identities"])
    assert path.read_bytes() == original


def test_preparation_identity_excludes_model_settings_but_model_identity_includes_them(encoding_session: EncodingConfig) -> None:
    """Input config; source/bin identity stays fixed across alpha/fit settings, return None."""
    first = results.compute_run_identities(encoding_session)
    changed = results.compute_run_identities(replace(encoding_session, alpha=9,
        windows=("after",), estimators=("ols",), encodings=("separate_context",)))
    assert first["preparation_id"] == changed["preparation_id"]
    assert first["model_id"] != changed["model_id"]
    narrower = results.compute_run_identities(replace(encoding_session, bin_width_ms=20))
    assert first["preparation_id"] != narrower["preparation_id"]


def test_changed_small_input_is_hashed_and_changes_identity(encoding_session: EncodingConfig) -> None:
    """Input config; saved feature provenance content affects preparation/model IDs, return None."""
    first = results.compute_run_identities(encoding_session)
    encoding_session.feature_parameters.write_text('{"changed":true}', encoding="ascii")
    second = results.compute_run_identities(encoding_session)
    assert first["preparation_id"] != second["preparation_id"]
    assert first["model_id"] != second["model_id"]
    records = {record["path"]: record for record in second["input_records"]}
    assert records[str(encoding_session.feature_parameters)]["sha256"]
    aligned = records[str(encoding_session.session_root / "A/aligned.npz")]
    assert "size" in aligned and "mtime_ns" in aligned and "sha256" not in aligned


def test_numerical_code_identity_covers_used_sources_and_scopes_changes(encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input config/patch fixture; used preparation/model source hashes are distinct, return None."""
    first = results.compute_run_identities(encoding_session)
    code = first["code_identity"]
    prep_paths = set(code["preparation"])
    assert any(path.endswith("population/session_timeline.py") for path in prep_paths)
    assert any(path.endswith("population/pca.py") for path in prep_paths)
    assert any(path.endswith("interregional/pca.py") for path in prep_paths)
    assert any(path.endswith("interregional/records.py") for path in prep_paths)
    assert any(path.endswith("spike_behavior/loading.py") for path in prep_paths)
    assert any(path.endswith("task_decoding/activity.py") for path in prep_paths)
    assert not any(path.endswith("src/main.py") or path.endswith("sync_ephys.py") for path in prep_paths)
    changed = deepcopy(code)
    model_source = next(iter(changed["modeling"]))
    changed["modeling"][model_source] = "changed-test-source-hash"
    monkeypatch.setattr(results, "numerical_code_identity", Mock(return_value=changed))
    second = results.compute_run_identities(encoding_session)
    assert first["preparation_id"] == second["preparation_id"]
    assert first["model_id"] != second["model_id"]


def test_status_and_saved_loading_do_not_touch_live_sources(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input saved run Path/patch; status and inspection need only saved files, return None."""
    monkeypatch.setattr(preparation, "load_behavior_inputs", forbid_source_access)
    monkeypatch.setattr(preparation, "prepare_regional_activity", forbid_source_access)
    monkeypatch.setattr(results, "compute_run_identities", forbid_source_access)
    status = results.inspect_run_status(prepared_encoding_run)
    loaded = results.load_encoding_run(prepared_encoding_run)
    assert status["status"] == "preparing"
    assert status["completed_cells"] == {}
    assert loaded["preparation"]["regions"]
    assert loaded["cells"] == {}


@pytest.mark.parametrize("failure", ["outside_run_root", "outside_preparation_root", "repository_session"])
def test_output_locations_are_session_local_and_outside_repository(
    encoding_session: EncodingConfig, failure: str,
) -> None:
    """Input config/failure name; invalid output Paths fail before creation, return None."""
    config = encoding_session
    identities = results.compute_run_identities(config)
    run = config.run_root / "context_encoding_guard"
    prep = config.intermediate_root / "guard/preparation.npz"
    if failure == "outside_run_root":
        run = config.session_root.parent / "elsewhere"
    elif failure == "outside_preparation_root":
        prep = config.session_root.parent / "elsewhere.npz"
    else:
        repository = Path(__file__).resolve().parents[4]
        config = replace(config, session_root=repository, run_root=repository / "analysis_runs",
                         intermediate_root=repository / "processed/context_encoding")
        run, prep = config.run_root / "forbidden", config.intermediate_root / "forbidden.npz"
    with pytest.raises(ValueError, match="session|repository|output|preparation"):
        results.initialize_run(config, run, prep, identities)
    assert not run.exists()


def test_invalid_preparation_version_is_reported_without_overwrite(prepared_encoding_run: Path) -> None:
    """Input run Path; unsupported schema in trusted NPZ raises and preserves bytes, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    path = Path(manifest["session_root"]) / manifest["preparation_path"]
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    arrays["meta"] = np.asarray({**meta, "schema_version": 999}, dtype=object)
    np.savez_compressed(path, **arrays)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="version|schema"):
        results.load_preparation(path)
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["missing", "identity", "version"])
def test_complete_status_validates_published_preparation(prepared_encoding_run: Path, failure: str) -> None:
    """Input completed run Path/failure; missing or mismatched shared PCA cannot appear reusable, return None."""
    pipeline.run_prepared_encoding(prepared_encoding_run)
    manifest = results.load_run_manifest(prepared_encoding_run)
    path = Path(manifest["session_root"]) / manifest["preparation_path"]
    if failure == "missing":
        path.unlink()
    else:
        with np.load(path, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta = arrays["meta"].item()
        meta["preparation_id" if failure == "identity" else "schema_version"] = "wrong"
        arrays["meta"] = np.asarray(meta, dtype=object)
        np.savez_compressed(path, **arrays)
    with pytest.raises((ValueError, OSError), match="preparation|schema|version"):
        results.inspect_run_status(prepared_encoding_run)


def test_numeric_runtime_versions_are_provenance_and_scientific_identity(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input config/patch; changed numerical-library version changes both stage IDs, return None."""
    first = results.compute_run_identities(encoding_session)
    changed = {**first["runtime_versions"], "numpy": "different-numerical-version"}
    monkeypatch.setattr(results, "numerical_runtime_versions", Mock(return_value=changed), raising=False)
    second = results.compute_run_identities(encoding_session)
    assert second["preparation_id"] != first["preparation_id"]
    assert second["model_id"] != first["model_id"]
    assert second["runtime_versions"] == changed
    assert "python" in first["runtime_versions"]
