"""Saved predictions bind the exact PCA archive while remaining relocatable."""

from dataclasses import fields, replace
import json
import shutil

import numpy as np
import pytest

from src.neural_analysis.population.session_timeline_io import save_timeline_result
from src.neural_analysis.task_decoding.session_prediction import results
from src.neural_analysis.task_decoding.session_prediction.preparation import prepare_prediction_inputs
from .test_modeling import fit_region


@pytest.fixture
def saved_prediction(prediction_inputs, request):
    """Return root, saved NPZ, inputs, models, metadata; scientific axes retain W,B,T/F,K."""
    base, trials, timeline = prediction_inputs
    mode = getattr(request, "param", "correct_rewarded_generalization")
    trials.loc[105, "correct"] = 0
    trials.loc[105, "state_int"] = 1 - trials.loc[105, "action"]
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode=mode)
    source = base.session_root / "analysis_runs/session_timeline_pca_fixture/results.npz"
    save_timeline_result(source, timeline)
    regions = {role: fit_region(prepared, role, use_training_membership=True) for role in timeline.regions}
    meta = results.build_prediction_metadata(
        prepared, base.session_root, source,
        {"augmented_trials": base.augmented_trial_path,
         "feature_parameters": base.trial_feature_parameter_path},
        {"bin_width_ms": 500, "probe_roles": {"PFC": "PFC", "HPC": "HPC"}},
    )
    path = base.session_root / "analysis_runs/condition_session_prediction_fixture/results.npz"
    results.save_prediction_result(path, prepared, regions, meta)
    return base.session_root, path, prepared, regions, meta


def test_roundtrip_keeps_axes_units_models_and_exact_pca(saved_prediction):
    """Round trip all named arrays without pickling estimators or copying PC scores."""
    root, path, prepared, regions, meta = saved_prediction
    loaded = results.load_prediction_result(path, session_root=root)
    assert loaded.meta["identity"] == meta["identity"]
    np.testing.assert_array_equal(loaded.inputs.target_values, prepared.target_values)
    np.testing.assert_array_equal(loaded.inputs.condition_masks, prepared.condition_masks)
    for role, model in regions.items():
        np.testing.assert_array_equal(loaded.regions[role].coefficients, model.coefficients)
        np.testing.assert_array_equal(loaded.regions[role].predictions, model.predictions)
        np.testing.assert_array_equal(loaded.inputs.timeline.regions[role].scores,
                                      prepared.timeline.regions[role].scores)
    with np.load(path, allow_pickle=True) as archive:
        assert archive["pfc_predictions"].shape == regions["PFC"].predictions.shape
        assert not any("scores" in name for name in archive.files)
        assert isinstance(archive["meta"].item(), dict)
    with pytest.raises(FileExistsError):
        results.save_prediction_result(path, prepared, regions, meta)


def test_relocation_needs_only_saved_products_not_raw_inputs(saved_prediction, tmp_path):
    """Relative source linkage survives cluster-to-workstation copies with no original CSVs."""
    root, path, _, _, _ = saved_prediction
    relocated = tmp_path / "copied"
    shutil.copytree(root / "analysis_runs", relocated / "analysis_runs")
    loaded = results.load_prediction_result(relocated / path.relative_to(root), session_root=relocated)
    assert loaded.inputs.timeline.session_id == "synthetic"
    assert not (relocated / "processed").exists()
    options = results.discover_prediction_runs(relocated)
    assert len(options) == 1
    assert results.find_saved_prediction(relocated, loaded.meta["identity"]) == relocated / path.relative_to(root)


@pytest.mark.parametrize("failure", ["missing", "changed"])
def test_referenced_pca_is_required_and_content_checked(saved_prediction, failure):
    """Matching names alone cannot substitute another PCA basis for the saved models."""
    root, path, _, _, meta = saved_prediction
    source = root / meta["source_pca"]["path"]
    if failure == "missing":
        source.unlink()
    else:
        source.write_bytes(source.read_bytes() + b"different")
    with pytest.raises((ValueError, FileNotFoundError)):
        results.load_prediction_result(path, session_root=root)


def test_identity_changes_for_science_not_display_or_unrelated_metadata(saved_prediction):
    """Effective numerical config and target bytes drive reuse; unrelated files do not."""
    root, _, prepared, _, meta = saved_prediction
    source = root / meta["source_pca"]["path"]
    sources = {name: root / info["path"] for name, info in meta["target_sources"].items()}
    (root / "display_settings.json").write_text("{}")
    identical = results.build_prediction_metadata(prepared, root, source, sources, meta["effective_config"])
    assert identical["identity"] == meta["identity"]
    changed = results.build_prediction_metadata(prepared, root, source, sources,
                                               {**meta["effective_config"], "bin_width_ms": 100})
    assert changed["identity"] != meta["identity"]
    next(iter(sources.values())).write_text("changed source")
    changed = results.build_prediction_metadata(prepared, root, source, sources, meta["effective_config"])
    assert changed["identity"] != meta["identity"]


@pytest.mark.parametrize("defect", ["probability", "nonfinite", "fold_leak", "counts", "conditions", "pc_ids"])
def test_invalid_saved_arrays_are_rejected_before_publication(saved_prediction, defect):
    """Common axis, availability, fold and probability corruption cannot publish a run."""
    root, _, prepared, regions, meta = saved_prediction
    model = regions["PFC"]
    if defect in {"probability", "nonfinite"}:
        values = model.predictions.copy()
        values[0, 0, 0] = 1.1 if defect == "probability" else np.inf
        model = replace(model, predictions=values)
    elif defect == "fold_leak":
        mask = model.training_choice_mask.copy()
        held = np.flatnonzero(prepared.fold_ids == 0)[0]
        mask[0, 0, held] = True
        model = replace(model, training_choice_mask=mask)
    elif defect == "counts":
        model = replace(model, training_row_count=model.training_row_count + 1)
    elif defect == "conditions":
        prepared = replace(prepared, condition_masks=~prepared.condition_masks)
    else:
        model = replace(model, pc_ids=np.array([1, 1, 3]))
    destination = root / "analysis_runs/condition_session_prediction_bad/results.npz"
    with pytest.raises(ValueError):
        results.save_prediction_result(destination, prepared, {**regions, "PFC": model}, meta)
    assert not destination.exists()


def test_discovery_reads_small_metadata_and_ignores_partial_runs(saved_prediction, monkeypatch):
    """Discovery avoids score/NPZ reads and excludes unpublished or malformed metadata."""
    root, _, _, _, _ = saved_prediction
    partial = root / "analysis_runs/condition_session_prediction_partial"
    partial.mkdir()
    (partial / "config.json").write_text(json.dumps({"analysis_version": "partial"}))

    def forbid_array_load(*args, **kwargs):
        """Any attempted NPZ discovery read is a regression; return never."""
        pytest.fail("Discovery must read only small metadata")

    monkeypatch.setattr(results.np, "load", forbid_array_load)
    assert len(results.discover_prediction_runs(root)) == 1


@pytest.mark.parametrize("saved_prediction", ["mixed_outcome_decoding", "correct_rewarded_generalization"], indirect=True)
def test_v2_mode_roundtrip_and_cross_mode_identity(saved_prediction):
    """New products persist selected membership and fixed mode metadata with correct T."""
    root, path, prepared, regions, meta = saved_prediction
    assert (meta["analysis_version"], meta["schema_version"]) == ("condition-session-prediction-v2", 2)
    config = meta["effective_config"]
    assert config["analysis_mode"] == prepared.analysis_mode
    assert config["target_names"] == list(prepared.target_names)
    expected = (["correct_rewarded"] if len(prepared.target_names) == 7
                else ["correct_rewarded", "omission", "incorrect"])
    assert config["training_condition_names"] == expected
    loaded = results.load_prediction_result(path, session_root=root)
    assert loaded.inputs.analysis_mode == prepared.analysis_mode
    np.testing.assert_array_equal(loaded.inputs.training_condition_mask, prepared.training_condition_mask)
    option = results.discover_prediction_runs(root)[0]
    assert option.analysis_mode == prepared.analysis_mode
    with np.load(path, allow_pickle=True) as archive:
        np.testing.assert_array_equal(archive["training_condition_mask"], prepared.training_condition_mask)
    if len(prepared.target_names) == 2:
        np.testing.assert_array_equal(loaded.regions["PFC"].predictions, regions["PFC"].predictions)
        original = replace(prepared, analysis_mode="correct_rewarded_generalization")
        # Different scientific modes cannot share identity even on the same PCA/target source bytes.
        sources = {name: root / info["path"] for name, info in meta["target_sources"].items()}
        with pytest.raises(ValueError, match="mode|target"):
            results.build_prediction_metadata(original, root, root / meta["source_pca"]["path"], sources, config)


@pytest.mark.parametrize("defect", ["mode", "target_set", "conditions", "mask", "missing_mode", "missing_mask", "version"])
def test_v2_rejects_mode_and_mask_contradictions(saved_prediction, defect):
    """New-schema products require explicit consistent mode, target, and mask contracts."""
    root, path, _, _, _ = saved_prediction
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    if defect == "mode":
        meta["effective_config"]["analysis_mode"] = "mixed_outcome_decoding"
    elif defect == "target_set":
        meta["effective_config"]["target_names"] = ["current_action"]
    elif defect == "conditions":
        meta["effective_config"]["training_condition_names"] = ["all"]
    elif defect == "mask":
        arrays["training_condition_mask"] = ~arrays["correct_rewarded"]
    elif defect == "missing_mode":
        meta["effective_config"].pop("analysis_mode", None)
    elif defect == "missing_mask":
        arrays.pop("training_condition_mask", None)
    else:
        meta["schema_version"] = 1
    arrays["meta"] = np.asarray(meta, dtype=object)
    # Mutate a synthetic archive directly to exercise the loader independently of publication.
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        results.load_prediction_result(path, session_root=root)


def test_legacy_v1_shaped_product_loads_without_rewrite_or_raw_data(saved_prediction):
    """Build the original schema explicitly, omitting every new mode/membership field."""
    root, path, prepared, regions, meta = saved_prediction
    legacy_shared = (
        "target_values", "target_valid", "baseline_valid", "correct_rewarded",
        "fold_ids", "block_values_json", "condition_masks", "training_bin_mask", "bin_edges_s",
    )
    meta = {**meta, "analysis_version": "condition-session-prediction-v1", "schema_version": 1,
            "identity": "original-v1-fixture", "effective_config": dict(meta["effective_config"])}
    for key in ("analysis_mode", "target_names", "training_condition_names"):
        meta["effective_config"].pop(key, None)
    meta["effective_config"]["target_names"] = list(prepared.target_names)
    arrays = {name: getattr(prepared, name) for name in legacy_shared}
    arrays.update({"choices": prepared.timeline.choices.to_records(index=False),
                   "target_names": np.asarray(prepared.target_names),
                   "target_families": np.asarray(prepared.target_families),
                   "condition_names": np.asarray(prepared.condition_names),
                   "meta": np.asarray(meta, dtype=object)})
    for role, model in regions.items():
        arrays[role.lower() + "_choice_positions"] = prepared.timeline.regions[role].choice_positions
        arrays.update({role.lower() + "_" + field.name: getattr(model, field.name)
                       for field in fields(results.RegionPredictions)})
    np.savez_compressed(path, **arrays)
    (path.parent / "config.json").write_text(json.dumps(meta))
    before = path.read_bytes()
    loaded = results.load_prediction_result(path, session_root=root)
    assert loaded.meta["identity"] == "original-v1-fixture"
    assert loaded.meta["analysis_version"] == "condition-session-prediction-v1"
    assert loaded.inputs.analysis_mode == "correct_rewarded_generalization"
    assert loaded.meta["effective_config"]["analysis_mode"] == loaded.inputs.analysis_mode
    np.testing.assert_array_equal(loaded.inputs.training_condition_mask, prepared.correct_rewarded)
    np.testing.assert_array_equal(loaded.regions["PFC"].predictions, regions["PFC"].predictions)
    assert path.read_bytes() == before
    assert results.discover_prediction_runs(root)[0].analysis_mode == loaded.inputs.analysis_mode


def test_publication_rejects_effective_mode_mismatch(saved_prediction):
    """Metadata must agree with arrays before a new product is published."""
    root, _, prepared, regions, meta = saved_prediction
    meta = {**meta, "effective_config": {**meta["effective_config"], "analysis_mode": "mixed_outcome_decoding"}}
    path = root / "analysis_runs/condition_session_prediction_bad_mode/results.npz"
    with pytest.raises(ValueError, match="mode|target"):
        results.save_prediction_result(path, prepared, regions, meta)
    assert not path.exists()
