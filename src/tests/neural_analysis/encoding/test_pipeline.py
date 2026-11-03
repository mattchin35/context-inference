"""Cell-at-a-time encoding execution, immutable resume and publication recovery."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from src.neural_analysis.encoding import modeling, pipeline, results
from src.neural_analysis.encoding.modeling import evaluate_block_cv as original_evaluate
from src.neural_analysis.encoding.results import write_progress as original_write_progress


def crash_after_cell_publish(run_directory: Path, progress: dict) -> None:
    """Input run Path/unitless progress dict; simulate index failure after one published NPZ."""
    if progress["completed_cells"]:
        raise RuntimeError("synthetic index update crash")
    original_write_progress(run_directory, progress)


def forbid_work(*args: object, **kwargs: object) -> None:
    """Reject arbitrary fitting/source arguments; return no arrays or physical units."""
    raise AssertionError("Completed saved work must not run again")


def fail_one_reference_cell(
    predictors: np.ndarray, responses: np.ndarray, fold_ids: np.ndarray, **settings: object,
) -> dict:
    """Input X(N,P)/PC Y(N,K)/fold IDs; fail only one PFC signed ridge reference, return no repaired arrays."""
    if len(predictors) == 60 and settings.get("estimator") == "ridge" and settings.get("encoding") == "signed_context":
        raise ValueError("synthetic reference-cell fit failure")
    return original_evaluate(predictors, responses, fold_ids, **settings)


def test_pipeline_publishes_unique_fits_with_explicit_aliases_and_observation_identity(prepared_encoding_run: Path) -> None:
    """Input run Path; ten distinct cells retain pooled row IDs/PC identities and scopes, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    assert progress["status"] == "complete"
    assert len(progress["completed_cells"]) == 10
    loaded = results.load_encoding_run(prepared_encoding_run)
    assert len(loaded["cells"]) == 10
    assert len(list((prepared_encoding_run / "cells").glob("*.npz"))) == 10
    per_region = {"PFC": [], "HPC": []}
    for cell_id, cell in loaded["cells"].items():
        meta = cell["meta"]
        settings = meta["settings"]
        per_region[settings["region"]].append(cell)
        assert meta["cell_id"] == cell_id and meta["available"]
        assert meta["preparation_id"] == loaded["manifest"]["identities"]["preparation_id"]
        region = loaded["preparation"]["regions"][settings["region"]]
        indices = cell["selected_row_indices"]
        np.testing.assert_array_equal(cell["observed"], region.timeline.scores.reshape(-1, region.timeline.scores.shape[-1])[indices][:, region.pc_ranks - 1])
        np.testing.assert_array_equal(cell["pc_ranks"], region.pc_ranks)
        assert cell["reference_slopes"].shape == (cell["predictors"].shape[1], len(region.pc_ranks))
        assert cell["held_out_predictions"].shape == cell["reference_predictions"].shape == cell["observed"].shape
        assert "held" in meta["score_scope"] and "reference" in meta["reference_scope"]
        assert "scores" not in cell  # Shared full activity is not copied into model files.
        if settings["estimator"] == "ols":
            assert meta["effective_alpha"] is None and settings["penalty_mode"] is None
        elif settings["encoding"] == "signed_context":
            assert set(meta["penalty_aliases"]) == {"same_numeric", "matched_context"}
            assert meta["configured_alpha"] == meta["effective_alpha"] == 1.0
        elif settings["penalty_mode"] == "matched_context":
            assert meta["effective_alpha"] == 0.5
    assert all(len(cells) == 5 for cells in per_region.values())


def test_completed_cells_are_not_refit_or_rewritten_on_resume(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input run Path/patch; complete cells keep bytes/mtimes without refitting, return None."""
    first = pipeline.run_prepared_encoding(prepared_encoding_run)
    before = {name: (prepared_encoding_run / name).stat().st_mtime_ns for name in first["completed_cells"].values()}
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_work)
    second = pipeline.run_prepared_encoding(prepared_encoding_run)
    assert second["status"] == "complete" and second["completed_cells"] == first["completed_cells"]
    assert before == {name: (prepared_encoding_run / name).stat().st_mtime_ns for name in before}


def test_crash_between_npz_publish_and_progress_is_reconciled(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input run Path/patch; adopt published compatible NPZ and fit only unfinished cells, return None."""
    monkeypatch.setattr(results, "write_progress", crash_after_cell_publish)
    with pytest.raises(RuntimeError, match="index update crash"):
        pipeline.run_prepared_encoding(prepared_encoding_run)
    published = list((prepared_encoding_run / "cells").glob("*.npz"))
    assert len(published) == 1
    published_time = published[0].stat().st_mtime_ns
    assert results.inspect_run_status(prepared_encoding_run)["completed_cells"] == {}
    monkeypatch.undo()
    counter = Mock(wraps=modeling.evaluate_block_cv)
    monkeypatch.setattr(modeling, "evaluate_block_cv", counter)
    resumed = pipeline.run_prepared_encoding(prepared_encoding_run)
    assert resumed["status"] == "complete" and len(resumed["completed_cells"]) == 10
    assert counter.call_count == 9
    assert published[0].stat().st_mtime_ns == published_time


def test_temporary_unpublished_files_are_ignored(prepared_encoding_run: Path) -> None:
    """Input run Path; interrupted temporary bytes never become published results, return None."""
    directory = prepared_encoding_run / "cells"
    directory.mkdir()
    temporary = directory / "unfinished.tmp.npz"
    temporary.write_bytes(b"partial unpublished bytes")
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    assert progress["status"] == "complete"
    assert len(progress["completed_cells"]) == 10
    assert temporary.read_bytes() == b"partial unpublished bytes"


def test_resume_uses_saved_preparation_when_live_inputs_disappear(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input run Path/patch; immutable saved PCA/behavior replaces absent live files, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    config = results.restore_config(manifest["config"])
    for path in (config.session_metadata, config.augmented_trials, config.feature_parameters,
                 config.session_root / "A/aligned.npz", config.session_root / "B/aligned.npz"):
        path.unlink()
    monkeypatch.setattr(results, "compute_run_identities", forbid_work)
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    assert progress["status"] == "complete" and len(progress["completed_cells"]) == 10


def test_changed_numerical_code_blocks_resume_without_overwriting(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input run Path/patch; changed scoped code requires new run, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    changed = deepcopy(manifest["identities"]["code_identity"])
    path = next(iter(changed["modeling"]))
    changed["modeling"][path] = "different-code"
    monkeypatch.setattr(results, "numerical_code_identity", Mock(return_value=changed))
    with pytest.raises(ValueError, match="code|new run|resume"):
        pipeline.run_prepared_encoding(prepared_encoding_run)
    assert results.inspect_run_status(prepared_encoding_run)["completed_cells"] == {}


def test_corrupt_published_cell_is_reported_never_silently_refit(prepared_encoding_run: Path) -> None:
    """Input run Path; malformed stored (N,K) predictions fail saved loading/resume, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    path = prepared_encoding_run / next(iter(progress["completed_cells"].values()))
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["reference_predictions"] = arrays["reference_predictions"][:-1]
    np.savez_compressed(path, **arrays)
    corrupted = path.read_bytes()
    with pytest.raises(ValueError, match="shape|corrupt|identity|prediction"):
        pipeline.run_prepared_encoding(prepared_encoding_run)
    assert path.read_bytes() == corrupted


def test_cell_and_preparation_identity_conflicts_fail_read_only(prepared_encoding_run: Path) -> None:
    """Input run Path; published cell cannot claim a different preparation identity, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    path = prepared_encoding_run / next(iter(progress["completed_cells"].values()))
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["meta"] = np.asarray({**arrays["meta"].item(), "preparation_id": "wrong"}, dtype=object)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="identity|preparation"):
        results.load_encoding_run(prepared_encoding_run)


def test_completed_scientifically_unavailable_cell_is_finished_work(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input run Path/patch; empty cell and absent CV are completed, not endless retries, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    config = results.restore_config(manifest["config"])
    path = config.session_root / manifest["preparation_path"]
    saved = results.load_preparation(path)
    pfc = saved["regions"]["PFC"]
    pfc.context_valid["choice_updated"][:] = False
    hpc = saved["regions"]["HPC"]
    hpc_rows = hpc.bin_rows.copy(deep=True)
    hpc_rows["fold_id"] = -1
    saved["regions"]["HPC"] = replace(hpc, window_fold_ids=np.full_like(hpc.window_fold_ids, -1),
        bin_rows=hpc_rows, cv_unavailable_reason="Only four usable blocks")
    # Build a new immutable fixture product; the original preparation remains unchanged.
    new_path = path.with_name("unavailable_preparation.npz")
    results.save_preparation(new_path, saved["trials"], saved["behavior"], saved["regions"], saved["meta"])
    new_run = config.run_root / "context_encoding_unavailable"
    results.initialize_run(config, new_run, new_path, manifest["identities"])
    progress = pipeline.run_prepared_encoding(new_run)
    assert progress["status"] == "complete" and len(progress["completed_cells"]) == 10
    loaded = results.load_encoding_run(new_run)
    for cell in loaded["cells"].values():
        if cell["meta"]["settings"]["region"] == "PFC":
            assert not cell["meta"]["available"] and cell["meta"]["unavailable_reason"]
        else:
            assert cell["meta"]["available"] and not cell["meta"]["cv_available"]
            assert np.isfinite(cell["reference_predictions"]).all()
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_work)
    assert pipeline.run_prepared_encoding(new_run)["status"] == "complete"


def test_saved_cells_do_not_duplicate_full_activity_or_other_configuration_results(prepared_encoding_run: Path) -> None:
    """Input run Path; each named NPZ stores only its own selected rows/fits, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    for relative in progress["completed_cells"].values():
        with np.load(prepared_encoding_run / relative, allow_pickle=True) as archive:
            assert isinstance(archive["meta"].item(), dict)
            assert "reference_slopes" in archive.files and "fold_slopes" in archive.files
            assert "observed" not in archive.files  # Reconstructed from shared preparation on load.
            assert "pfc_scores" not in archive.files and "hpc_scores" not in archive.files


def test_completed_resume_validates_each_cell_once(prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Input run Path/patch; resume validates ten published cells once without retaining all arrays, return None."""
    pipeline.run_prepared_encoding(prepared_encoding_run)
    counter = Mock(wraps=results.load_cell)
    monkeypatch.setattr(results, "load_cell", counter)
    pipeline.run_prepared_encoding(prepared_encoding_run)
    assert counter.call_count == 10


@pytest.mark.parametrize("array_name,bad_value", [("held_out_predictions", np.nan), ("mse", np.inf)])
def test_cv_available_cell_rejects_nonfinite_primary_products(
    prepared_encoding_run: Path, array_name: str, bad_value: float,
) -> None:
    """Input run Path/product/scalar; a valid-CV claim requires finite OOF and MSE, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    path = prepared_encoding_run / next(iter(progress["completed_cells"].values()))
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    assert arrays["meta"].item()["cv_available"]
    arrays[array_name].flat[0] = bad_value
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="finite|prediction|MSE|mse"):
        results.load_cell(path)
    with pytest.raises(ValueError, match="finite|prediction|MSE|mse"):
        results.inspect_run_status(prepared_encoding_run)


@pytest.mark.parametrize("array_name", [
    "fold_valid", "fold_train_counts", "fold_test_counts", "fold_reasons", "fold_r2_reason",
    "r2_reason", "column_names", "fold_ids", "pc_ranks", "selected_row_indices",
])
def test_cell_loader_validates_metadata_axes_and_integer_identities(
    prepared_encoding_run: Path, array_name: str,
) -> None:
    """Input run Path/array name; malformed fold/predictor/response metadata cannot load, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    path = prepared_encoding_run / next(iter(progress["completed_cells"].values()))
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if array_name in ("fold_ids", "pc_ranks", "selected_row_indices"):
        arrays[array_name] = arrays[array_name].astype(float)
        arrays[array_name].flat[0] = 0.5
    else:
        arrays[array_name] = arrays[array_name][:-1]
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="shape|integer|identity|fold|column|rank"):
        results.load_cell(path)


def test_reference_cell_failure_is_saved_unavailable_and_other_cells_continue(
    prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input run Path/patch; failed reference preserves selected row/PC identity and never retries, return None."""
    monkeypatch.setattr(modeling, "evaluate_block_cv", fail_one_reference_cell)
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    assert progress["status"] == "complete" and len(progress["completed_cells"]) == 10
    saved = results.load_encoding_run(prepared_encoding_run)
    failed = [cell for cell in saved["cells"].values() if not cell["meta"]["available"]]
    assert len(failed) == 1
    cell = failed[0]
    assert "synthetic reference-cell" in cell["meta"]["unavailable_reason"]
    assert not cell["meta"]["reference_valid"]
    assert len(cell["selected_row_indices"]) == 60
    np.testing.assert_array_equal(cell["pc_ranks"], saved["preparation"]["regions"]["PFC"].pc_ranks)
    assert cell["reference_predictions"].shape == cell["observed"].shape
    assert np.isnan(cell["reference_predictions"]).all()
    assert "synthetic reference-cell fit failure" in (prepared_encoding_run / "run.log").read_text(encoding="ascii")
    monkeypatch.setattr(modeling, "evaluate_block_cv", forbid_work)
    assert pipeline.run_prepared_encoding(prepared_encoding_run)["status"] == "complete"


def test_runtime_version_change_refuses_resume_without_modifying_progress(
    prepared_encoding_run: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input run Path/patch; numerical runtime drift requires a new run before writes, return None."""
    manifest = results.load_run_manifest(prepared_encoding_run)
    changed = {**manifest["identities"]["runtime_versions"], "scipy": "different-numerical-version"}
    monkeypatch.setattr(results, "numerical_runtime_versions", Mock(return_value=changed), raising=False)
    original = (prepared_encoding_run / "progress.json").read_bytes()
    with pytest.raises(ValueError, match="runtime|version|new run"):
        pipeline.run_prepared_encoding(prepared_encoding_run)
    assert (prepared_encoding_run / "progress.json").read_bytes() == original


def test_pipeline_logs_each_cell_settings_counts_status_and_elapsed_seconds(prepared_encoding_run: Path) -> None:
    """Input run Path; cell audit records carry model controls/row counts/status/time, return None."""
    progress = pipeline.run_prepared_encoding(prepared_encoding_run)
    text = (prepared_encoding_run / "run.log").read_text(encoding="ascii")
    assert all(key in text for key in ("cell_id", "settings", "row_count", "status", "elapsed_s"))
    assert progress["elapsed_s"] >= 0
    loaded = results.load_encoding_run(prepared_encoding_run)
    for cell_id, cell in loaded["cells"].items():
        assert cell_id in text
        assert cell["meta"]["elapsed_s"] >= 0
