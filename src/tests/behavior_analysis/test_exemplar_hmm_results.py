"""Versioned fixed-exemplar persistence, cache identity and export regression tests."""

from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis import block_exemplar_models as exemplar_models
from src.behavior_analysis import exemplar_hmm_modeling as modeling


@pytest.fixture
def results_module():
    """Return the persistence module; no input or physical units."""
    return importlib.import_module("src.behavior_analysis.exemplar_hmm_results")


@pytest.fixture
def specs():
    """Return two ordered fixed exemplar specs with intercept and SD in trials."""
    return (
        exemplar_models.BlockExemplarModelSpec("fast", 2.0, 0.0, 2.0),
        exemplar_models.BlockExemplarModelSpec("slow", 6.0, 0.5, 4.0),
    )


@pytest.fixture
def blocks():
    """Return (4, 6) raw block features; TTC and preceding rewards are trials."""
    return pd.DataFrame({
        "block_ix": [3, 9, 15, 22], "trials_to_correct": [2.0, np.nan, 8.0, 10.0],
        "prev_n_rewarded": [3.0, 4.0, 5.0, 6.0],
        "bias_full_flag": [True, False, True, False],
        "fresh_block_feature": [1, 2, 3, 4], "source_block_row": [0, 1, 2, 3],
    }, index=pd.Index([81, 20, 81, 55], name="original_row"))


@pytest.fixture
def trials():
    """Return (5, 2) current trial features indexed independently from blocks."""
    return pd.DataFrame({"cur_block": [15, 3, 9, 22, 99], "fresh_feature": [1, 2, 3, 4, 5]}, index=[4, 2, 2, 9, 1])


def test_new_result_persists_named_plain_arrays_and_complete_provenance(
    results_module, blocks, trials, specs, tmp_path,
):
    """A run saves plain numerical data, immutable tables and readable provenance."""
    original_blocks, original_trials = blocks.copy(deep=True), trials.copy(deep=True)
    result, inherited, path = results_module.save_or_load_exemplar_hmm_results(
        blocks, trials, tmp_path, "mouse_session", specs=specs,
    )
    pd.testing.assert_frame_equal(blocks, original_blocks)
    pd.testing.assert_frame_equal(trials, original_trials)
    assert path.relative_to(tmp_path).parts[:2] == ("exemplar_hmm", results_module.ANALYSIS_VERSION)
    with np.load(path, allow_pickle=True) as saved:
        required = {"raw_residuals", "z_residuals", "log_likelihoods", "independent_probs", "valid_mask", "valid_positions", "initial_distribution", "transition_matrix", "smoothed_probs", "smoothed_states", "marginal_loglik", "meta"}
        assert required.issubset(saved.files)
        for name in required - {"meta"}:
            assert not saved[name].dtype.hasobject
        meta = saved["meta"].item()
    assert meta["analysis_version"] == results_module.ANALYSIS_VERSION
    assert meta["generator"].endswith("save_or_load_exemplar_hmm_results")
    assert meta["session_id"] == "mouse_session"
    assert meta["random_seed"] is None
    assert meta["model_names"] == ["fast", "slow"]
    assert meta["parameters"]["stay_prob"] == 0.9
    assert meta["parameters"]["specs"][0]["intercept"] == 2.0
    assert meta["input_digest"] and meta["analysis_source_digest"]
    assert meta["package_versions"]["dynamax"]
    assert meta["units"] and meta["axis_conventions"]
    assert meta["runtime_seconds"] >= 0
    assert "warnings" in meta and "errors" in meta
    assert isinstance(meta["source_manifest"], list)
    assert meta["scope"] == "single_session"
    run = path.parent
    for filename in ["block_performance.csv", "augmented_trials.csv", "metadata.json", "summary.md", "run.log", "main.py", "batch.py"]:
        assert (run / filename).exists(), filename
    assert "main.py" in (run / "summary.md").read_text()
    assert "source" in (run / "batch.py").read_text().lower()
    assert "def run_multisession_analysis(" in (run / "batch.py").read_text()
    assert result.valid_positions.tolist() == [0, 2, 3]
    assert inherited["fresh_feature"].tolist() == trials["fresh_feature"].tolist()
    assert (tmp_path / "mouse_session_block_performance.csv").exists()
    assert (tmp_path / "mouse_session_augmented_trials.csv").exists()


def test_cache_hit_rebuilds_current_tables_without_smoothing(
    results_module, blocks, trials, specs, tmp_path, monkeypatch,
):
    """A numerical hit reattaches only modeled outputs to fresh block/trial features."""
    first, _, first_path = results_module.save_or_load_exemplar_hmm_results(
        blocks, trials, tmp_path, "session", specs=specs,
    )
    first_bytes = {path: path.read_bytes() for path in first_path.parent.iterdir() if path.is_file()}

    def reject_smoother(*args, **kwargs):
        """Reject all recomputation calls; arguments/return have no units."""
        pytest.fail("Cache hit must not recompute the smoother")

    monkeypatch.setattr(modeling, "compute_fixed_exemplar_block_states", reject_smoother)
    monkeypatch.setattr(modeling, "smooth_fixed_exemplar_likelihoods_with_dynamax", reject_smoother)
    fresh_blocks = blocks.assign(fresh_block_feature=[11, 12, 13, 14], bias_full_flag=False)
    fresh_trials = trials.assign(fresh_feature=[11, 12, 13, 14, 15], new_feature=42)
    second, inherited, second_path = results_module.save_or_load_exemplar_hmm_results(
        fresh_blocks, fresh_trials, tmp_path, "session", specs=specs,
    )
    assert first_path == second_path
    assert second.block_performance["fresh_block_feature"].tolist() == [11, 12, 13, 14]
    assert inherited["fresh_feature"].tolist() == [11, 12, 13, 14, 15]
    assert inherited["new_feature"].tolist() == [42] * 5
    assert inherited["inherited_block_bias"].iloc[:4].tolist() == [False] * 4
    pd.testing.assert_index_equal(second.block_performance.index, blocks.index)
    pd.testing.assert_index_equal(inherited.index, trials.index)
    np.testing.assert_allclose(first.smoothing_result.smoothed_probs, second.smoothing_result.smoothed_probs)
    for path, content in first_bytes.items():
        assert path.read_bytes() == content
    records = list(first_path.parent.parent.glob("*/augmented_trials.csv"))
    assert len(records) == 2
    assert any(pd.read_csv(path)["new_feature"].tolist() == [42] * 5 for path in records if "new_feature" in pd.read_csv(path))


@pytest.mark.parametrize("change", ["parameter", "stay", "raw", "model_order", "source_row", "code", "block_order", "manifest_order", "manifest_added_empty", "manifest_removed"])
def test_cache_identity_changes_with_numerical_inputs(
    results_module, blocks, trials, specs, tmp_path, monkeypatch, change,
):
    """Parameters, source/code identity and ordered source sequence invalidate cache."""
    manifest = [{"session_id": "a", "n_blocks": 4}, {"session_id": "b", "n_blocks": 0}]
    first, _, first_path = results_module.save_or_load_exemplar_hmm_results(
        blocks, trials, tmp_path, "session", specs=specs, source_manifest=manifest,
    )
    kwargs = {"specs": specs, "source_manifest": manifest}
    if change == "parameter":
        kwargs["specs"] = (replace(specs[0], intercept=3.0), specs[1])
    elif change == "stay":
        kwargs["stay_prob"] = 0.8
    elif change == "raw":
        blocks = blocks.copy()
        blocks.iloc[0, blocks.columns.get_loc("trials_to_correct")] = 3.0
    elif change == "model_order":
        kwargs["specs"] = specs[::-1]
    elif change == "source_row":
        blocks = blocks.assign(source_block_row=[1, 0, 2, 3])
    elif change == "code":
        monkeypatch.setattr(results_module, "_analysis_source_digest", lambda: "different-source")
    elif change == "block_order":
        blocks = blocks.iloc[::-1]
    elif change == "manifest_order":
        kwargs["source_manifest"] = manifest[::-1]
    elif change == "manifest_added_empty":
        kwargs["source_manifest"] = manifest + [{"session_id": "empty", "n_blocks": 0}]
    elif change == "manifest_removed":
        kwargs["source_manifest"] = manifest[:1]
    second, _, second_path = results_module.save_or_load_exemplar_hmm_results(
        blocks, trials, tmp_path, "session", **kwargs,
    )
    assert first_path != second_path
    assert first_path.exists() and second_path.exists()
    if change == "model_order":
        assert second.model_names == first.model_names[::-1]


def test_residuals_regenerate_and_obsolete_columns_are_archived_then_removed(
    results_module, blocks, trials, specs, tmp_path,
):
    """Saved legacy exports survive while current model columns use current specs."""
    blocks = blocks.assign(cur_strategy_slope=0.7, hardcoded_strategy=1, inferred_strategy=77)
    blocks["removed_exemplar_residual_TTS"] = 999
    blocks["removed_exemplar_normalized_residual_TTS"] = 999
    blocks["fast_exemplar_residual_TTS"] = 999
    blocks["fast_exemplar_normalized_residual_TTS"] = 999
    blocks["exemplar_hmm_smoothed_prob_removed"] = 1.0
    trials = trials.assign(inherited_block_prob_removed=1.0, inherited_block_strategy=77)
    block_path = tmp_path / "session_block_performance.csv"
    trial_path = tmp_path / "session_augmented_trials.csv"
    blocks.to_csv(block_path, index=False)
    trials.to_csv(trial_path, index=False)
    before = {path.name: path.read_bytes() for path in (block_path, trial_path)}
    pickle_path = tmp_path / "session_block_statedict.pkl"
    pickle_path.write_bytes(b"old ssm payload must stay byte-identical")
    result, inherited, _ = results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path, "session", specs=specs)
    for name, content in before.items():
        archived_name = Path(name).with_suffix(".archived.csv").name
        assert any(path.read_bytes() == content for path in (tmp_path / "exemplar_hmm" / "archives").rglob(archived_name))
    assert pickle_path.read_bytes() == b"old ssm payload must stay byte-identical"
    assert not {"cur_strategy_slope", "hardcoded_strategy", "removed_exemplar_residual_TTS", "removed_exemplar_normalized_residual_TTS", "exemplar_hmm_smoothed_prob_removed"}.intersection(result.block_performance)
    assert "inherited_block_prob_removed" not in inherited
    assert result.block_performance["fast_exemplar_residual_TTS"].iloc[0] == 0.0
    assert result.block_performance["fast_exemplar_normalized_residual_TTS"].iloc[0] == 0.0
    assert result.block_performance["inferred_strategy"].iloc[0] != 77


def test_preserve_exports_is_content_addressed_and_does_not_touch_source_files(results_module, tmp_path):
    """Repeated archival preserves existing files once per distinct byte content."""
    block_path = tmp_path / "session_block_performance.csv"
    block_path.write_text("block_ix,inferred_strategy\n1,legacy\n")
    before = block_path.read_bytes()
    assert results_module.preserve_current_exports(tmp_path, "session") is None
    results_module.preserve_current_exports(tmp_path, "session")
    archived_name = "session_block_performance.archived.csv"
    archived = list((tmp_path / "exemplar_hmm" / "archives").rglob(archived_name))
    assert len(archived) == 1 and archived[0].read_bytes() == before
    assert block_path.read_bytes() == before
    block_path.write_text("block_ix,inferred_strategy\n1,new\n")
    results_module.preserve_current_exports(tmp_path, "session")
    archived = list((tmp_path / "exemplar_hmm" / "archives").rglob(archived_name))
    assert len(archived) == 2


def test_preserve_exports_reuses_legacy_archive_without_renaming_it(results_module, tmp_path):
    """Path/module fixtures preserve old archive bytes without a duplicate backup."""
    from hashlib import sha256

    current = tmp_path / "session_augmented_trials.csv"
    content = b"cur_block,action\n0,1\n"
    current.write_bytes(content)
    legacy = tmp_path / "exemplar_hmm" / "archives" / sha256(content).hexdigest() / current.name
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(content)
    results_module.preserve_current_exports(tmp_path, "session")
    assert legacy.read_bytes() == content and current.read_bytes() == content
    assert list(legacy.parent.iterdir()) == [legacy]


@pytest.mark.parametrize("empty", [False, True])
def test_empty_and_invalid_results_roundtrip_without_inventing_states(
    results_module, blocks, trials, specs, tmp_path, empty,
):
    """Zero-valid-block results roundtrip as full missing outputs and empty arrays."""
    blocks = blocks.iloc[:0] if empty else blocks.assign(trials_to_correct="None")
    trials = trials.iloc[:0] if empty else trials
    first, _, path = results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path, "session", specs=specs)
    second, inherited, reused = results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path, "session", specs=specs)
    assert path == reused
    assert second.smoothing_result.smoothed_probs.shape == (0, 2)
    assert second.block_performance["inferred_strategy"].isna().all()
    assert inherited["inherited_block_strategy"].isna().all()
    pd.testing.assert_frame_equal(first.block_performance, second.block_performance)
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert metadata["status"] == "no_valid_blocks"
    assert metadata["n_valid_blocks"] == 0


def test_rerun_creates_another_immutable_result(results_module, blocks, trials, specs, tmp_path):
    """Explicit rerun makes another timestamp directory without changing old files."""
    _, _, first = results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path, "session", specs=specs)
    previous = first.read_bytes()
    _, _, second = results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path, "session", specs=specs, rerun=True)
    assert first != second and first.parent.parent == second.parent.parent
    assert first.read_bytes() == previous


def test_default_manifest_distinguishes_same_date_sessions_and_preserves_scope(
    results_module, blocks, trials, specs, tmp_path,
):
    """Derived ordered manifest records full session IDs, and joint writes stay local."""
    blocks = blocks.assign(source_session_id=["same_day_a", "same_day_a", "same_day_b", "same_day_b"], source_date="2026-01-01")
    trials = trials.assign(source_session_id=["same_day_b", "same_day_a", "same_day_a", "same_day_b", "same_day_b"])
    result, inherited, path = results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path / "joint", "joint", specs=specs)
    with np.load(path, allow_pickle=True) as saved:
        meta = saved["meta"].item()
    assert [entry["session_id"] for entry in meta["source_manifest"]] == ["same_day_a", "same_day_b"]
    assert meta["scope"] == "multisession"
    assert inherited["source_session_id"].tolist() == trials["source_session_id"].tolist()
    assert not (tmp_path / "same_day_a").exists()
    assert not (tmp_path / "same_day_b").exists()


@pytest.mark.parametrize("explicit_manifest", [None, [], [{"session_id": "only_source", "n_blocks": 4}]])
def test_joint_scope_is_preserved_with_one_or_zero_manifest_entries(
    results_module, blocks, trials, specs, tmp_path, explicit_manifest,
):
    """Explicit joint source metadata defines scope even for one selected session."""
    blocks = blocks.assign(source_session_id="only_source")
    trials = trials.assign(source_session_id="only_source")
    _, _, path = results_module.save_or_load_exemplar_hmm_results(
        blocks, trials, tmp_path, "joint", specs=specs, source_manifest=explicit_manifest,
    )
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert metadata["scope"] == "multisession"
    assert metadata["export_record_directory"] == str(path.parent)
    assert metadata["numerical_artifact"] == str(path)


def test_failed_inference_writes_log_without_reusable_npz(
    results_module, blocks, trials, specs, tmp_path, monkeypatch,
):
    """An inference error propagates and leaves a diagnostic rather than a cache."""
    def fail_compute(*args, **kwargs):
        """Raise a synthetic computation error; inputs/output carry no units."""
        raise RuntimeError("synthetic inference failure")

    monkeypatch.setattr(modeling, "compute_fixed_exemplar_block_states", fail_compute)
    with pytest.raises(RuntimeError, match="synthetic inference failure"):
        results_module.save_or_load_exemplar_hmm_results(blocks, trials, tmp_path, "session", specs=specs)
    logs = list(tmp_path.rglob("run.log"))
    assert len(logs) == 1
    assert "synthetic inference failure" in logs[0].read_text()
    assert not list(tmp_path.rglob("results.npz"))
