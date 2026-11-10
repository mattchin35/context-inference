"""Self-contained named-array publication, portable provenance and normal saved contracts."""

from collections import Counter
from dataclasses import fields, replace
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import session_timeline_io
from src.neural_analysis.population.session_timeline import prepare_timeline_choices
from src.neural_analysis.task_decoding.cross_condition import modeling, results, scoring
from src.neural_analysis.task_decoding.cross_condition.preparation import prepare_transfer_inputs


@pytest.fixture
def metadata_args(transfer_data, regional_transfer_data, request):
    """Return builder args(inputs, root Path, regional source Paths, target Paths, config dict).

    Shared/separate PCA archives contain real seeded regional scores(W,B,K),
    but identity needs only their metadata and whole-file hashes. Behavioral
    raw numeric strings remain strings in choice identities/conflict records.
    Times use UTC/relative seconds; labels and weights are dimensionless.
    """
    trials, _, edges, base = transfer_data
    _, regions = regional_transfer_data
    raw = trials.copy(deep=True)
    raw["cur_trial"] = raw.cur_trial.astype(object)
    raw.loc[raw.cur_trial % 2 == 1, "cur_trial"] = raw.loc[raw.cur_trial % 2 == 1, "cur_trial"].map(str)
    raw["state_int"], raw["correct"] = raw.state_int.astype(object), raw.correct.astype(object)
    raw.loc[raw.index[14], "state_int"] = "0.0"
    raw.loc[raw.index[13], "correct"] = "0"
    _, choices = prepare_timeline_choices(raw)
    inputs = prepare_transfer_inputs(raw, choices, edges, base)
    raw.to_csv(base.augmented_trial_path, index=False)
    root = base.session_root
    separate = getattr(request, "param", "shared") == "separate"
    source_pcas = {}
    groups = ({role: (role,) for role in regions} if separate else {"shared": tuple(regions)})
    for name, roles in groups.items():
        source = root / "analysis_runs" / f"session_timeline_pca_{name}" / "results.npz"
        source.parent.mkdir(parents=True)
        meta = {"analysis_version": "session-timeline-pca-v1", "schema_version": 1,
                "configuration_id": f"pca-{name}", "session_id": "synthetic", "bin_width_s": .5}
        arrays = {"meta": np.asarray(meta, dtype=object)}
        for role in roles:
            scores, positions = regions[role]
            arrays[role.lower() + "_scores"] = scores
            arrays[role.lower() + "_choice_positions"] = positions
            source_pcas[role] = source
        np.savez_compressed(source, **arrays)
    target_sources = {"augmented_trials": base.augmented_trial_path,
                      "base_config": root / "task_decoding_config.json",
                      "feature_parameters": base.trial_feature_parameter_path}
    config = {"bin_width_ms": 500, "probe_roles": {"PFC": "ProbeA", "HPC": "ProbeB"}}
    return inputs, root, source_pcas, target_sources, config


@pytest.fixture
def fitted_transfer_bundle(metadata_args, regional_transfer_data):
    """Return CrossConditionResult built from real tiny seeded regional fits, not scoring placeholders."""
    inputs = metadata_args[0]
    regions = {role: modeling.fit_region_transfers(scores, positions, inputs)
               for role, (scores, positions) in regional_transfer_data[1].items()}
    meta = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    return results.CrossConditionResult(inputs, regions, {}, meta)


@pytest.fixture
def saved_transfer_bundle(metadata_args, fitted_transfer_bundle):
    """Return (result Path, packet) for a complete named-array schema-1 run inside the temporary session."""
    path = metadata_args[1] / "analysis_runs/cross_condition_generalization_20300102_030405_fixture/results.npz"
    assert results.save_transfer_result(path, fitted_transfer_bundle) == path
    return path, fitted_transfer_bundle


def assert_inputs_equal(actual, expected):
    """Compare all TransferInputs fields, retaining N/S/H/B axes, raw scalar types and DataFrame columns."""
    for field in fields(expected):
        observed, original = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(original, np.ndarray):
            assert observed.dtype == original.dtype
            np.testing.assert_array_equal(observed, original)
        elif isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(observed.reset_index(drop=True), original.reset_index(drop=True))
        else:
            assert observed == original


@pytest.mark.parametrize("metadata_args", ["shared", "separate"], indirect=True)
def test_complete_roundtrip_named_arrays_labels_axes_and_raw_types(saved_transfer_bundle):
    """Round trip every prepared/model field with dictionary meta and no neural-score copies."""
    path, packet = saved_transfer_bundle
    loaded = results.load_transfer_result(path)
    assert_inputs_equal(loaded.inputs, packet.inputs)
    assert loaded.unavailable_regions == {}
    assert loaded.meta["identity"] == packet.meta["identity"]
    for role, model in packet.regions.items():
        for field in fields(model):
            np.testing.assert_array_equal(getattr(loaded.regions[role], field.name), getattr(model, field.name))
    assert isinstance(loaded.inputs.choices.trial_id.iloc[0], (int, np.integer))
    assert isinstance(loaded.inputs.choices.trial_id.iloc[1], str)
    assert isinstance(loaded.inputs.conflicts.correct.iloc[0], str)
    assert isinstance(loaded.inputs.conflicts.state_int.iloc[1], str)
    with np.load(path, allow_pickle=True) as archive:
        required = {"choices", "conflicts", "target_values", "target_valid", "baseline_valid", "conflict_mask",
                    "condition_masks", "source_cells", "test_cells", "window_bin_masks", "bin_edges_s",
                    "target_names", "condition_names", "window_names", "weighting_names", "split_ids",
                    "split_target_names", "source_cell_names", "test_cell_names", "role_names", "meta"}
        assert required <= set(archive.files)
        assert not any("scores" in key or "fold" in key for key in archive.files)
        assert isinstance(archive["meta"].item(), dict)
        np.testing.assert_array_equal(archive["role_names"], ["training", "complementary_test"])
        np.testing.assert_array_equal(archive["pfc_trial_probabilities"], packet.regions["PFC"].trial_probabilities)
        np.testing.assert_array_equal(archive["pfc_eligible_cell_counts"], packet.regions["PFC"].eligible_cell_counts)
    manifest = json.loads((path.parent / "config.json").read_text())
    assert manifest["identity"] == loaded.meta["identity"]


def test_metadata_records_science_units_versions_and_separate_runtime_provenance(metadata_args):
    """Identity/provenance documents actual computation, all windows/weightings and deterministic fitting."""
    meta = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    assert (meta["analysis_version"], meta["schema_version"]) == ("cross-condition-generalization-v1", 1)
    assert meta["session_id"] == "synthetic" and meta["generator"].endswith("cross_condition.run_session")
    assert meta["randomness_used"] is False and meta["random_seed"] is None
    assert {"numpy", "pandas", "scipy", "scikit-learn"} <= set(meta["library_versions"])
    assert {"code_revision", "code_status", "runtime_s", "peak_process_rss_bytes"} <= set(meta)
    assert meta["session_metadata"]["path"] == "neural_session.json"
    assert meta["target_metadata"] == metadata_args[0].target_metadata
    config = meta["effective_config"]
    assert config["target_names"] == list(metadata_args[0].target_names)
    assert config["window_names"] == ["full", "before", "after"]
    assert config["weighting_names"] == ["equal_trial", "equal_cell"]
    assert len(config["transfers"]) == 16
    assert meta["axes"]["trial_probabilities"] == ["supervised_window", "weighting", "split", "regional_choice"]
    assert meta["axes"]["eligible_cell_counts"] == ["split", "role", "actual_class"]
    assert meta["units"]["absolute_time"] == "UTC Unix seconds"
    assert meta["units"]["coefficients"] == "log-odds per PC-score unit"
    assert meta["units"]["intercepts"] == "log-odds"
    dependencies = set(meta["scientific_sources"])
    assert "src/neural_analysis/task_decoding/cross_condition/modeling.py" in dependencies
    assert "src/neural_analysis/task_decoding/session_prediction/modeling.py" in dependencies
    assert "src/neural_analysis/task_decoding/pooled/decoding.py" in dependencies
    assert not any("scoring.py" in name or "webapp" in name or "run_session.py" in name for name in dependencies)


@pytest.mark.parametrize("metadata_args", ["shared", "separate"], indirect=True)
def test_source_identity_is_deduplicated_and_reads_only_npz_metadata(metadata_args, monkeypatch):
    """Each distinct archive is hashed/read once, with portable region links and configuration IDs."""
    _, root, source_pcas, _, _ = metadata_args
    calls, reads = Counter(), []
    original_identity = results.file_identity
    with np.load(next(iter(source_pcas.values())), allow_pickle=True) as archive:
        archive_type = type(archive)
    original_getitem = archive_type.__getitem__

    def counted_identity(path, session_root):
        """Count calls per resolved file Path; delegate portable relative/size/SHA-256 identity."""
        calls[Path(path).resolve()] += 1
        return original_identity(path, session_root)

    def metadata_only(archive, key):
        """Reject array reads during identity construction; allow only dictionary metadata."""
        assert key == "meta", f"Identity unexpectedly loaded PCA array {key}"
        reads.append(key)
        return original_getitem(archive, key)

    monkeypatch.setattr(results, "file_identity", counted_identity)
    monkeypatch.setattr(archive_type, "__getitem__", metadata_only)
    meta = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    distinct = {path.resolve() for path in source_pcas.values()}
    assert len(reads) == len(distinct) == len(meta["source_pcas"])
    for region, path in source_pcas.items():
        relative = path.relative_to(root).as_posix()
        assert calls[path.resolve()] == 1
        assert meta["region_sources"][region] == relative
        info = meta["source_pcas"][relative]
        assert info["path"] == relative and info["size"] == path.stat().st_size
        assert len(info["sha256"]) == 64
        assert info["configuration_id"] == "pca-" + path.parent.name.removeprefix("session_timeline_pca_")


def test_identity_survives_relocation_key_order_and_unrelated_metadata_changes(metadata_args, tmp_path):
    """Absolute prefixes and small-file presentation/PPC edits do not change fitting identity."""
    inputs, root, source_pcas, target_sources, config = metadata_args
    original = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    metadata = root / "neural_session.json"
    payload = json.loads(metadata.read_text())
    payload["analysis"]["ppc"]["preview_shuffles"] += 1
    metadata.write_text(json.dumps(payload, indent=4), encoding="ascii")
    (root / "display_settings.json").write_text('{"dark_mode": true}', encoding="ascii")
    identical = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    assert identical["identity"] == original["identity"]
    assert identical["session_metadata"]["sha256"] != original["session_metadata"]["sha256"]
    relocated = tmp_path / "relocated_session"
    shutil.copytree(root, relocated)
    moved = results.build_transfer_metadata(
        inputs, relocated,
        {region: relocated / path.relative_to(root) for region, path in reversed(tuple(source_pcas.items()))},
        {name: relocated / path.relative_to(root) for name, path in reversed(tuple(target_sources.items()))},
        dict(reversed(tuple(config.items()))), session_id="synthetic",
    )
    assert moved["identity"] == original["identity"]
    assert moved["source_pcas"] == original["source_pcas"]
    assert all(not Path(info["path"]).is_absolute() for info in moved["target_sources"].values())


@pytest.mark.parametrize("source_name", ["augmented_trials", "base_config", "feature_parameters", "pca"])
def test_whole_file_source_changes_change_fitting_identity(metadata_args, source_name):
    """Behavior/base/feature/PCA bytes matter even when file names and prepared arrays stay fixed."""
    original = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    source = next(iter(metadata_args[2].values())) if source_name == "pca" else metadata_args[3][source_name]
    if source_name == "pca":
        with np.load(source, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        meta = arrays["meta"].item()
        meta["configuration_id"] = "pca-revised"
        arrays["meta"] = np.asarray(meta, dtype=object)
        np.savez_compressed(source, **arrays)
    elif source_name == "augmented_trials":
        table = pd.read_csv(source)
        table.loc[0, "reward"] = 0
        table.to_csv(source, index=False)
    else:
        payload = json.loads(source.read_text())
        payload["bin_width_ms" if source_name == "base_config" else "test_parameter"] = 100
        source.write_text(json.dumps(payload), encoding="ascii")
    changed = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    assert changed["identity"] != original["identity"]


def test_region_source_map_effective_roles_and_computational_code_change_identity(metadata_args, monkeypatch):
    """Source-region assignment, effective roles and a reused readout's code hash are fitting inputs."""
    inputs, root, source_pcas, target_sources, config = metadata_args
    original = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    changed_config = {**config, "probe_roles": {"PFC": "DifferentProbe", "HPC": "ProbeB"}}
    changed = results.build_transfer_metadata(inputs, root, source_pcas, target_sources, changed_config, session_id="synthetic")
    assert changed["identity"] != original["identity"]
    pfc_only = results.build_transfer_metadata(inputs, root, {"PFC": source_pcas["PFC"]}, target_sources, config, session_id="synthetic")
    assert pfc_only["identity"] != original["identity"]
    original_identity = results.file_identity

    def changed_readout_identity(path, session_root):
        """Simulate a different scientific file hash without editing any project/dependency file."""
        value = original_identity(path, session_root)
        if Path(path).as_posix().endswith("task_decoding/session_prediction/modeling.py"):
            value = {**value, "sha256": "0" * 64}
        return value

    monkeypatch.setattr(results, "file_identity", changed_readout_identity)
    changed_code = results.build_transfer_metadata(*metadata_args, session_id="synthetic")
    assert changed_code["identity"] != original["identity"]


def test_contradictory_width_or_registry_cannot_build_new_metadata(metadata_args):
    """A 500-ms prepared grid cannot claim 100 ms, nor relabel the fixed sixteen split definitions."""
    inputs, root, source_pcas, target_sources, config = metadata_args
    with pytest.raises(ValueError, match="bin|width|grid"):
        results.build_transfer_metadata(inputs, root, source_pcas, target_sources, {**config, "bin_width_ms": 100}, session_id="synthetic")
    altered = replace(inputs, transfers=(replace(inputs.transfers[0], target_name="current_reward"), *inputs.transfers[1:]))
    with pytest.raises(ValueError):
        results.build_transfer_metadata(altered, root, source_pcas, target_sources, config, session_id="synthetic")


def test_runtime_timestamp_and_git_provenance_can_be_completed_without_new_fitting_identity(metadata_args, fitted_transfer_bundle):
    """Run measurements/creation time and Git receipt describe execution, without redefining scientific reuse."""
    original = fitted_transfer_bundle.meta
    meta = {**original, "runtime_s": 12.5, "peak_process_rss_bytes": 123456,
            "code_revision": "another-provenance-revision", "code_status": [" M notes.txt"],
            "created_at_utc": "2030-01-02T03:04:05Z"}
    path = metadata_args[1] / "analysis_runs/cross_condition_generalization_runtime/results.npz"
    results.save_transfer_result(path, replace(fitted_transfer_bundle, meta=meta))
    loaded = results.load_transfer_result(path)
    assert loaded.meta["identity"] == original["identity"]
    assert loaded.meta["runtime_s"] == 12.5 and loaded.meta["peak_process_rss_bytes"] == 123456
    assert loaded.meta["code_revision"] == "another-provenance-revision"


def test_copied_run_loads_and_scores_with_all_original_inputs_removed(saved_transfer_bundle, metadata_args, tmp_path, monkeypatch):
    """A copied final run alone supports loading, discovery, exact reuse and all summaries; no PCA/fit calls."""
    path, packet = saved_transfer_bundle
    expected = scoring.summarize_transfers(packet.inputs, packet.regions, packet.unavailable_regions)
    relocated = tmp_path / "saved_only"
    destination = relocated / "analysis_runs" / path.parent.name / path.name
    shutil.copytree(path.parent, destination.parent)
    shutil.rmtree(metadata_args[1])

    def forbidden(*args, **kwargs):
        """Reject all attempts to hash/reopen original sources, load PCA or fit during saved-only inspection."""
        raise AssertionError("Saved result loading must not use original sources or fitting")

    monkeypatch.setattr(results, "file_identity", forbidden)
    monkeypatch.setattr(session_timeline_io, "load_timeline_result", forbidden)
    monkeypatch.setattr(modeling, "fit_readout", forbidden)
    monkeypatch.setattr(modeling, "fit_region_transfers", forbidden)
    before = destination.read_bytes()
    loaded = results.load_transfer_result(destination)
    assert scoring.summarize_transfers(loaded.inputs, loaded.regions, loaded.unavailable_regions) == expected
    assert not (relocated / "processed").exists() and not (relocated / "task_decoding_config.json").exists()
    option, = results.discover_transfer_runs(relocated)
    assert option.result_path == destination and option.identity == packet.meta["identity"]
    assert option.session_id == "synthetic" and option.bin_width_ms == 500
    assert "500" in option.display_label
    assert results.find_saved_transfer(relocated, packet.meta["identity"]) == destination
    assert destination.read_bytes() == before


@pytest.mark.parametrize("availability", ["missing_class", "numerical_failure", "one_region", "all_regions"])
def test_roundtrip_local_failures_and_unavailable_regions(metadata_args, fitted_transfer_bundle, regional_transfer_data, availability):
    """Retain failed-fit NaNs/counts/weights and unavailable regional reasons, with no invented PC axes."""
    inputs, root, source_pcas, target_sources, config = metadata_args
    models, unavailable = dict(fitted_transfer_bundle.regions), {}
    if availability == "missing_class":
        scores, positions = regional_transfer_data[1]["PFC"]
        retained = ~inputs.choices.trial_id.astype(int).isin([2, 3]).to_numpy()
        models["PFC"] = modeling.fit_region_transfers(scores[retained], positions[retained], inputs)
        # Keep the referenced synthetic PCA's coverage identical to the fitted source.
        source = source_pcas["PFC"]
        with np.load(source, allow_pickle=True) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["pfc_scores"], arrays["pfc_choice_positions"] = scores[retained], positions[retained]
        source_meta = arrays["meta"].item()
        source_meta["configuration_id"] = "pca-missing-class"
        arrays["meta"] = np.asarray(source_meta, dtype=object)
        np.savez_compressed(source, **arrays)
    elif availability == "numerical_failure":
        model = models["PFC"]
        parameters, intercepts = model.coefficients.copy(), model.intercepts.copy()
        probability, status, reason = model.trial_probabilities.copy(), model.fit_status.copy(), model.fit_reason.copy()
        parameters[0, 0, 0], intercepts[0, 0, 0], probability[0, 0, 0] = np.nan, np.nan, np.nan
        status[0, 0, 0], reason[0, 0, 0] = "unavailable", "nonconvergence"
        models["PFC"] = replace(model, coefficients=parameters, intercepts=intercepts, trial_probabilities=probability,
                                fit_status=status, fit_reason=reason)
    else:
        unavailable = {"HPC": "regional source unavailable"}
        models.pop("HPC")
        if availability == "all_regions":
            unavailable["PFC"] = "regional PCA preparation failed"
            models = {}
        source_pcas = {role: path for role, path in source_pcas.items() if role in models}
    meta = results.build_transfer_metadata(inputs, root, source_pcas, target_sources, config, session_id="synthetic")
    packet = results.CrossConditionResult(inputs, models, unavailable, meta)
    path = root / f"analysis_runs/cross_condition_generalization_{availability}/results.npz"
    results.save_transfer_result(path, packet)
    loaded = results.load_transfer_result(path)
    assert loaded.unavailable_regions == unavailable and set(loaded.regions) == set(models)
    for role, model in models.items():
        for field in fields(model):
            np.testing.assert_array_equal(getattr(loaded.regions[role], field.name), getattr(model, field.name))
    assert scoring.summarize_transfers(loaded.inputs, loaded.regions, unavailable)
    with np.load(path, allow_pickle=True) as archive:
        for role in unavailable:
            assert not any(key.startswith(role.lower() + "_") for key in archive.files)
    if availability == "all_regions":
        assert loaded.meta["region_sources"] == loaded.meta["source_pcas"] == {}


@pytest.mark.parametrize("defect", [
    "target_order", "target_label", "invalid_target_value", "mask_dtype", "window_bins", "grid", "choice_identity",
    "regional_map", "pc_ids", "role_overlap", "cell_orientation", "cell_count", "weights", "off_source_weight",
    "probability", "missing_probability", "off_role_probability", "coefficient_shape", "available_parameter",
    "status", "fit_reason", "source_map", "schema", "analysis", "positive_label", "cell_name", "missing_array", "grid_width",
    "role_name", "weighting_order", "window_order", "split_ids", "conflict_ids", "baseline_cells", "condition_mask", "axis_names",
])
def test_loader_rejects_normal_malformed_saved_products(saved_transfer_bundle, defect):
    """Validate saved axes, label/cell/grid identities, counts/weights and model availability without PCA access."""
    path, _ = saved_transfer_bundle
    with np.load(path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    meta = arrays["meta"].item()
    if defect == "target_order":
        arrays["target_names"] = arrays["target_names"][::-1]
    elif defect == "target_label":
        arrays["target_values"][0, 0] = .5
    elif defect == "invalid_target_value":
        arrays["target_values"][12, 0] = 0
    elif defect == "mask_dtype":
        arrays["source_cells"] = arrays["source_cells"].astype(int)
    elif defect == "window_bins":
        arrays["window_bin_masks"][0, 0] = True
    elif defect == "grid":
        arrays["bin_edges_s"] += .1
    elif defect == "choice_identity":
        arrays["choices"]["row_position"][1] = 0
    elif defect == "regional_map":
        arrays["pfc_choice_positions"][1] = 0
    elif defect == "pc_ids":
        arrays["pfc_pc_ids"][1] = 1
    elif defect == "role_overlap":
        arrays["pfc_test_mask"][0, 0] = True
    elif defect == "cell_orientation":
        arrays["source_cells"][:, 0] = arrays["source_cells"][:, 0, ::-1]
    elif defect == "cell_count":
        arrays["pfc_eligible_cell_counts"][0, 0, 0] += 1
    elif defect == "weights":
        arrays["pfc_source_trial_weights"][1, 0, 0] += .1
    elif defect == "off_source_weight":
        arrays["pfc_source_trial_weights"][0, 0, 4] = 1
    elif defect == "probability":
        arrays["pfc_trial_probabilities"][0, 0, 0, 0] = 1.1
    elif defect == "missing_probability":
        arrays["pfc_trial_probabilities"][0, 0, 0, 0] = np.nan
    elif defect == "off_role_probability":
        arrays["pfc_trial_probabilities"][0, 0, 0, 8] = .4
    elif defect == "coefficient_shape":
        arrays["pfc_coefficients"] = arrays["pfc_coefficients"][..., :-1]
    elif defect == "available_parameter":
        arrays["pfc_coefficients"][0, 0, 0, 0] = np.nan
    elif defect == "status":
        arrays["pfc_fit_status"][0, 0, 0] = "unavailable"
    elif defect == "fit_reason":
        arrays["pfc_fit_reason"][0, 0, 0] = "nonconvergence"
    elif defect == "source_map":
        meta["region_sources"]["PFC"] = "missing/results.npz"
    elif defect == "schema":
        meta["schema_version"] = 2
    elif defect == "analysis":
        meta["analysis_version"] = "condition-session-prediction-v3"
    elif defect == "positive_label":
        meta["target_metadata"]["current_reward"]["positive_label"] = "left"
    elif defect == "cell_name":
        arrays["source_cell_names"][0, 0] = "wrong"
    elif defect == "missing_array":
        arrays.pop("window_bin_masks")
    elif defect == "grid_width":
        meta["effective_config"]["bin_width_ms"] = 100
    elif defect == "role_name":
        arrays["role_names"][0] = "held_out"
    elif defect == "weighting_order":
        arrays["weighting_names"] = arrays["weighting_names"][::-1]
    elif defect == "window_order":
        arrays["window_names"] = arrays["window_names"][::-1]
    elif defect == "split_ids":
        arrays["split_ids"] = arrays["split_ids"][::-1]
    elif defect == "conflict_ids":
        arrays["conflicts"]["row_position"][0] = 0
    elif defect == "baseline_cells":
        arrays["baseline_valid"][0] = False
    elif defect == "condition_mask":
        arrays["condition_masks"][1, 2] = False
    else:
        meta["axes"]["trial_probabilities"] = ["split", "weighting", "supervised_window", "regional_choice"]
    arrays["meta"] = np.asarray(meta, dtype=object)
    np.savez_compressed(path, **arrays)
    (path.parent / "config.json").write_text(json.dumps(meta), encoding="ascii")
    with pytest.raises(ValueError) as error:
        results.load_transfer_result(path)
    assert str(error.value)


@pytest.mark.parametrize("defect", ["probability", "weights", "counts", "failed_parameters", "unavailable_overlap"])
def test_invalid_records_are_rejected_before_final_publication(metadata_args, fitted_transfer_bundle, defect):
    """Validation precedes final publication; unsuccessful records cannot leave a discoverable NPZ."""
    packet = fitted_transfer_bundle
    model = packet.regions["PFC"]
    if defect == "probability":
        values = model.trial_probabilities.copy()
        values[0, 0, 0, 0] = -1
        model = replace(model, trial_probabilities=values)
    elif defect == "weights":
        model = replace(model, source_trial_weights=model.source_trial_weights * 2)
    elif defect == "counts":
        model = replace(model, eligible_cell_counts=model.eligible_cell_counts + 1)
    elif defect == "failed_parameters":
        status, reason = model.fit_status.copy(), model.fit_reason.copy()
        status[0, 0, 0], reason[0, 0, 0] = "unavailable", "nonconvergence"
        model = replace(model, fit_status=status, fit_reason=reason)
    else:
        packet = replace(packet, unavailable_regions={"PFC": "unavailable"})
    packet = replace(packet, regions={**packet.regions, "PFC": model})
    path = metadata_args[1] / "analysis_runs/cross_condition_generalization_invalid/results.npz"
    with pytest.raises(ValueError):
        results.save_transfer_result(path, packet)
    assert not path.exists()


def test_manifest_identity_must_match_archive_and_exact_lookup_skips_invalid(saved_transfer_bundle, metadata_args):
    """An explicitly selected inconsistent product has a clear error; exact reuse does not return it."""
    path, packet = saved_transfer_bundle
    manifest_path = path.parent / "config.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identity"] = "different-identity"
    manifest_path.write_text(json.dumps(manifest), encoding="ascii")
    with pytest.raises(ValueError, match="identity|manifest|config"):
        results.load_transfer_result(path)
    assert results.find_saved_transfer(metadata_args[1], packet.meta["identity"]) is None


def test_discovery_is_metadata_only_newest_first_and_skips_incomplete_runs(saved_transfer_bundle, metadata_args, monkeypatch):
    """Only completed prefix-matching runs with valid schema metadata appear; no archive reads during discovery."""
    path, packet = saved_transfer_bundle
    root = metadata_args[1]
    newer = root / "analysis_runs/cross_condition_generalization_20310102_030405_copy"
    shutil.copytree(path.parent, newer)
    for name, payload in (("partial", packet.meta), ("bad", {"analysis_version": "unsupported"})):
        directory = root / f"analysis_runs/cross_condition_generalization_{name}"
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps(payload), encoding="ascii")
        if name == "bad":
            (directory / "results.npz").write_bytes(b"incomplete archive")

    def forbidden(*args, **kwargs):
        """Fail if discovery opens any NPZ or original source instead of small manifest JSON."""
        raise AssertionError("Discovery must only read small metadata")

    monkeypatch.setattr(results.np, "load", forbidden)
    options = results.discover_transfer_runs(root)
    assert [option.result_path for option in options] == [newer / "results.npz", path]


def test_existing_results_are_never_overwritten(saved_transfer_bundle):
    """Repeated publication raises before replacing the NPZ or its manifest bytes."""
    path, packet = saved_transfer_bundle
    original, manifest = path.read_bytes(), (path.parent / "config.json").read_bytes()
    with pytest.raises(FileExistsError):
        results.save_transfer_result(path, packet)
    assert path.read_bytes() == original and (path.parent / "config.json").read_bytes() == manifest


def test_atomic_npz_publication_and_interrupted_write_are_not_discoverable(metadata_args, fitted_transfer_bundle, monkeypatch):
    """Publish via a complete temporary sibling rename; an interrupted NPZ leaves no final/temporary arrays."""
    root = metadata_args[1]
    path = root / "analysis_runs/cross_condition_generalization_atomic/results.npz"
    original_replace = Path.replace
    replacements = []

    def observe_replace(source, destination):
        """Observe a completed temporary sibling NPZ before its one final atomic rename."""
        if Path(destination) == path:
            assert source != path and source.parent == path.parent and source.suffix == ".npz"
            assert source.is_file() and not path.exists()
            with np.load(source, allow_pickle=True) as archive:
                assert archive["meta"].item()["identity"] == fitted_transfer_bundle.meta["identity"]
            replacements.append(source)
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", observe_replace)
    results.save_transfer_result(path, fitted_transfer_bundle)
    assert len(replacements) == 1 and path.is_file()

    def interrupt_write(*args, **kwargs):
        """Simulate a write-time filesystem failure; no result bytes are successfully published."""
        raise OSError("synthetic interrupted write")

    monkeypatch.setattr(results.np, "savez_compressed", interrupt_write)
    failed = root / "analysis_runs/cross_condition_generalization_interrupted/results.npz"
    with pytest.raises(OSError, match="interrupted"):
        results.save_transfer_result(failed, fitted_transfer_bundle)
    assert not failed.exists() and not list(failed.parent.glob("*.npz"))
    assert [option.result_path for option in results.discover_transfer_runs(root)] == [path]
