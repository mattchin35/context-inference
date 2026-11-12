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
    selected = getattr(request, "param", "correct_rewarded_generalization")
    mode, window = selected if isinstance(selected, tuple) else (selected, "full")
    trials.loc[105, "correct"] = 0
    trials.loc[105, "state_int"] = 1 - trials.loc[105, "action"]
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode=mode, training_window=window)
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
    with pytest.raises(ValueError, match="bin|width|grid"):
        results.build_prediction_metadata(prepared, root, source, sources,
                                          {**meta["effective_config"], "bin_width_ms": 100})
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
def test_v4_mode_roundtrip_and_cross_mode_identity(saved_prediction):
    """New products persist selected membership and fixed mode metadata with correct T."""
    root, path, prepared, regions, meta = saved_prediction
    assert (meta["analysis_version"], meta["schema_version"]) == ("condition-session-prediction-v4", 3)
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
    if prepared.analysis_mode == "mixed_outcome_decoding":
        np.testing.assert_array_equal(loaded.regions["PFC"].predictions, regions["PFC"].predictions)
        original = replace(prepared, analysis_mode="correct_rewarded_generalization")
        # Different scientific modes cannot share identity even on the same PCA/target source bytes.
        sources = {name: root / info["path"] for name, info in meta["target_sources"].items()}
        with pytest.raises(ValueError, match="mode|target"):
            results.build_prediction_metadata(original, root, root / meta["source_pca"]["path"], sources, config)


@pytest.mark.parametrize("defect", ["mode", "target_set", "conditions", "mask", "missing_mode", "missing_mask", "version"])
def test_v4_rejects_mode_and_mask_contradictions(saved_prediction, defect):
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
    for key in ("analysis_mode", "target_names", "training_condition_names", "training_window", "scoring_window_s"):
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


@pytest.mark.parametrize("saved_prediction", [
    (mode, window) for mode in ("correct_rewarded_generalization", "mixed_outcome_decoding")
    for window in ("full", "before", "after")
], indirect=True)
def test_v4_window_roundtrip_and_identity(saved_prediction):
    """Every mode/window persists exact fitting/scoring bounds, mask and class metadata."""
    root, path, prepared, _, meta = saved_prediction
    bounds = {"full": [-1., 1.], "before": [-1., 0.], "after": [0., 1.]}
    config = meta["effective_config"]
    assert config["training_window"] == prepared.training_window
    assert config["training_window_s"] == config["scoring_window_s"] == bounds[prepared.training_window]
    assert config["prediction_window_s"] == [-2., 2.]
    loaded = results.load_prediction_result(path, session_root=root)
    assert loaded.inputs.training_window == prepared.training_window
    np.testing.assert_array_equal(loaded.inputs.training_bin_mask, prepared.training_bin_mask)
    option = results.discover_prediction_runs(root)[0]
    assert option.training_window == prepared.training_window
    assert prepared.training_window in option.display_label
    sources = {name: root / value["path"] for name, value in meta["target_sources"].items()}
    other = "after" if prepared.training_window != "after" else "before"
    from src.neural_analysis.task_decoding.session_prediction.preparation import training_bin_mask
    other_inputs = replace(prepared, training_window=other,
                           training_bin_mask=training_bin_mask(prepared.bin_edges_s, training_window=other))
    effective = {"bin_width_ms": 500, "probe_roles": config["probe_roles"]}
    other_meta = results.build_prediction_metadata(other_inputs, root, root / meta["source_pca"]["path"], sources, effective)
    assert other_meta["identity"] != meta["identity"]
    if prepared.analysis_mode == "mixed_outcome_decoding":
        assert loaded.inputs.target_names == ("current_state", "current_action", "current_reward",
                                         "qlearning_relative_value", "forgetting_q_relative_value",
                                         "hmm_signed_belief", "hmm_decay_signed_belief", "relative_doubt")
        assert meta["target_metadata"]["current_reward"]["positive_label"] == "rewarded"
        assert "left" not in meta["units"]["categorical_prediction"]


@pytest.mark.parametrize("defect", [
    "missing_window", "missing_scoring", "bounds", "scoring", "mask", "width", "extent", "reward_label", "binary_values",
])
@pytest.mark.parametrize("saved_prediction", [("mixed_outcome_decoding", "before")], indirect=True)
def test_v4_rejects_window_and_reward_contract_corruption(saved_prediction, defect):
    """Correct array length cannot disguise the wrong supervised interval or class meaning."""
    root, path, _, _, _ = saved_prediction
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    config = meta["effective_config"]
    if defect == "missing_window":
        config.pop("training_window", None)
    elif defect == "missing_scoring":
        config.pop("scoring_window_s", None)
    elif defect == "bounds":
        config["training_window_s"] = [0, 1]
    elif defect == "scoring":
        config["scoring_window_s"] = [-1, 1]
    elif defect == "mask":
        arrays["training_bin_mask"] = np.roll(arrays["training_bin_mask"], 2)
    elif defect == "width":
        config["bin_width_ms"] = 100
    elif defect == "extent":
        config["prediction_window_s"] = [-1, 1]
    elif defect == "binary_values":
        arrays["target_values"][0, 2] = 0.5
    else:
        meta["target_metadata"]["current_reward"]["positive_label"] = "left"
    arrays["meta"] = np.asarray(meta, dtype=object)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError):
        results.load_prediction_result(path, session_root=root)


def write_legacy_product(saved_prediction, version, mode):
    """Write a faithful old product and return its original identity; synthetic files only.

    Fixture tuple supplies root/path, N-choice inputs and T/F/K models. Version
    is 1, 2 or 3; mode selects seven generalization or two/three mixed targets.
    Versions 1/2 retain Full only; version 3 preserves the source window.
    """
    root, path, prepared, _, original = saved_prediction
    count = 7 if mode == "correct_rewarded_generalization" else (3 if version == 3 else 2)
    prepared = replace(prepared, target_names=prepared.target_names[:count],
                       target_families=prepared.target_families[:count],
                       target_values=prepared.target_values[:, :count], target_valid=prepared.target_valid[:, :count],
                       target_metadata={name: prepared.target_metadata[name] for name in prepared.target_names[:count]},
                       target_availability={name: prepared.target_availability[name] for name in prepared.target_names[:count]})
    models = {role: fit_region(prepared, role, use_training_membership=True) for role in prepared.timeline.regions}
    config = dict(original["effective_config"], target_names=list(prepared.target_names))
    if version < 3:
        for key in ("training_window", "scoring_window_s"):
            config.pop(key, None)
    if version == 1:
        config.pop("analysis_mode", None)
        config.pop("training_condition_names", None)
    meta = {**original, "analysis_version": f"condition-session-prediction-v{version}", "schema_version": version,
            "identity": f"original-v{version}-{mode}", "effective_config": config,
            "target_metadata": prepared.target_metadata, "target_availability": prepared.target_availability}
    shared = ("target_values", "target_valid", "baseline_valid", "correct_rewarded", "fold_ids",
              "block_values_json", "condition_masks", "training_bin_mask", "bin_edges_s")
    arrays = {name: getattr(prepared, name) for name in shared}
    if version >= 2:
        arrays["training_condition_mask"] = prepared.training_condition_mask
    arrays.update(choices=prepared.timeline.choices.to_records(index=False),
                  target_names=np.asarray(prepared.target_names), target_families=np.asarray(prepared.target_families),
                  condition_names=np.asarray(prepared.condition_names), meta=np.asarray(meta, dtype=object))
    for role, model in models.items():
        arrays[role.lower() + "_choice_positions"] = prepared.timeline.regions[role].choice_positions
        arrays.update({role.lower() + "_" + field.name: getattr(model, field.name) for field in fields(model)})
    np.savez_compressed(path, **arrays)
    (path.parent / "config.json").write_text(json.dumps(meta))
    return meta["identity"]


@pytest.mark.parametrize("version, saved_prediction", [
    (1, "correct_rewarded_generalization"), (2, "correct_rewarded_generalization"), (2, "mixed_outcome_decoding"),
    (3, "correct_rewarded_generalization"),
    (3, "mixed_outcome_decoding"), (3, ("mixed_outcome_decoding", "before")),
    (3, ("mixed_outcome_decoding", "after")),
], indirect=["saved_prediction"])
def test_original_version_contracts_load_relocate_and_score(saved_prediction, version, tmp_path):
    """Old schemas retain Full meaning and original bytes without raw sources or invented reward models."""
    from src.neural_analysis.task_decoding.session_prediction.scoring import compute_prediction_performance
    root, path, prepared, _, _ = saved_prediction
    identity = write_legacy_product(saved_prediction, version, prepared.analysis_mode)
    relocated = tmp_path / "legacy_copy"
    shutil.copytree(root / "analysis_runs", relocated / "analysis_runs")
    destination = relocated / path.relative_to(root)
    before = destination.read_bytes()
    loaded = results.load_prediction_result(destination, session_root=relocated)
    assert loaded.meta["identity"] == identity
    expected_window = prepared.training_window if version == 3 else "full"
    bounds = {"full": [-1., 1.], "before": [-1., 0.], "after": [0., 1.]}
    assert loaded.inputs.training_window == expected_window
    assert loaded.meta["effective_config"]["scoring_window_s"] == bounds[expected_window]
    expected_count = 7 if prepared.analysis_mode.endswith("generalization") else (3 if version == 3 else 2)
    assert len(loaded.inputs.target_names) == expected_count
    assert ("current_reward" in loaded.inputs.target_names) == (expected_count == 3)
    assert compute_prediction_performance(loaded.inputs, loaded.regions, "current_action")["held_out"]
    assert results.discover_prediction_runs(relocated)[0].training_window == expected_window
    assert results.find_saved_prediction(relocated, identity) == destination
    assert destination.read_bytes() == before


@pytest.mark.parametrize("saved_prediction", ["mixed_outcome_decoding"], indirect=True)
@pytest.mark.parametrize("defect", ["old_label_new_targets", "new_label_old_targets", "schema"])
def test_v4_rejects_version_target_set_mismatches(saved_prediction, defect):
    """Version labels cannot authorize a different mixed target axis or storage layout."""
    root, path, _, _, _ = saved_prediction
    if defect == "new_label_old_targets":
        write_legacy_product(saved_prediction, 3, "mixed_outcome_decoding")
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    meta["analysis_version"] = ("condition-session-prediction-v3" if defect == "old_label_new_targets"
                                else "condition-session-prediction-v4")
    if defect == "schema":
        meta["schema_version"] = 4
    arrays["meta"] = np.asarray(meta, dtype=object)
    np.savez_compressed(path, **arrays)
    (path.parent / "config.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        results.load_prediction_result(path, session_root=root)
    assert not results.discover_prediction_runs(root)


@pytest.mark.parametrize("defect", ["half_window", "reward_target", "unknown_pair", "new_schema_two_targets"])
@pytest.mark.parametrize("saved_prediction", ["mixed_outcome_decoding"], indirect=True)
def test_legacy_contract_rejects_new_scope_injection(saved_prediction, defect):
    """A v2 label cannot legitimize a half-window, reward model or unknown schema pair."""
    root, path, _, _, _ = saved_prediction
    write_legacy_product(saved_prediction, 2, "mixed_outcome_decoding")
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    if defect == "half_window":
        meta["effective_config"]["training_window"] = "before"
    elif defect == "reward_target":
        meta["effective_config"]["target_names"].append("current_reward")
    elif defect == "new_schema_two_targets":
        meta["analysis_version"] = "condition-session-prediction-v3"
        meta["schema_version"] = 3
        meta["effective_config"].update(training_window="full", scoring_window_s=[-1., 1.])
    else:
        meta["schema_version"] = 3
    arrays["meta"] = np.asarray(meta, dtype=object)
    np.savez_compressed(path, **arrays)
    (path.parent / "config.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        results.load_prediction_result(path, session_root=root)
    assert not results.discover_prediction_runs(root)
