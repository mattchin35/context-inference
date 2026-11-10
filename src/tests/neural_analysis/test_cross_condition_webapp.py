"""Real summary-only Streamlit behavior on portable cross-condition products."""

from dataclasses import replace
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.neural_analysis.population import run_session_timeline as timeline_runner
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.task_decoding import config as task_config
from src.neural_analysis.task_decoding.cross_condition import config, modeling, results, run_session, scoring
from src.neural_analysis.webapp import app, cross_condition_view as view, session_inputs
from src.tests.neural_analysis.task_decoding.cross_condition.conftest import (
    regional_transfer_data as regional_fixture, transfer_data as data_fixture,
    transfer_session as session_fixture, transfer_trials as trials_fixture,
)
from src.tests.neural_analysis.task_decoding.cross_condition.test_results import (
    fitted_transfer_bundle as fitted_fixture, metadata_args as metadata_fixture,
    saved_transfer_bundle as saved_fixture,
)


transfer_session, transfer_trials = session_fixture, trials_fixture
transfer_data, regional_transfer_data = data_fixture, regional_fixture
metadata_args, fitted_transfer_bundle, saved_transfer_bundle = metadata_fixture, fitted_fixture, saved_fixture

DIRECTIONS = {
    "current_state": ("Correct to incorrect", "Left choice to right choice", "Right choice to left choice", "Incorrect to correct"),
    "current_action": ("Correct to incorrect", "Left context to right context", "Right context to left context", "Incorrect to correct"),
    "current_reward": ("Left choice to right choice", "Left rewarded / right unrewarded", "Right rewarded / left unrewarded", "Right choice to left choice"),
    "current_action_is_correct": ("Left choice to right choice", "Left correct / right incorrect", "Right correct / left incorrect", "Right choice to left choice"),
}


def render_transfer_app(metadata_path):
    """Render actual new view for a metadata Path/string, retaining saved N/W/H/G/S contracts."""
    import streamlit as st
    from src.neural_analysis.webapp.cross_condition_view import render_cross_condition_view
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    render_cross_condition_view(st, load_webapp_session(metadata_path))


def render_actual_app(metadata_path):
    """Run actual application metadata dispatch, rather than a test-owned route or fake renderer."""
    from src.neural_analysis.webapp.app import main

    main(["--session-metadata", str(metadata_path)])


def widget(test_app, label):
    """Return the unique real selectbox with a public label; values use stable saved identifiers."""
    matches = [item for item in test_app.selectbox if item.label == label]
    assert len(matches) == 1
    return matches[0]


def summary_table(test_app):
    """Return the displayed eight-row direction/region performance table, with explicit percent units."""
    matches = [item.value for item in test_app.dataframe if {"Direction", "Region", "Accuracy (%)", "Scored trials"} <= set(item.value.columns)]
    assert len(matches) == 1
    return matches[0]


def fit_table(test_app):
    """Return the regional fit-details dataframe with actual source/test class counts and stable split IDs."""
    required = {"Split ID", "Region", "Fit status", "Fit reason", "Source class 0", "Source class 1", "Test class 0", "Test class 1"}
    matches = [item.value for item in test_app.dataframe if required <= set(item.value.columns)]
    assert len(matches) == 1
    return matches[0]


def rendered_text(test_app):
    """Return public captions/info/warnings/errors/markdown as lowercase text for scientific wording assertions."""
    return "\n".join(str(item.value) for kind in (test_app.caption, test_app.info, test_app.warning, test_app.error, test_app.markdown)
                     for item in kind).lower()


def patch_view_dependency(monkeypatch, owner, name, replacement):
    """Patch one computation/I/O API and any direct view alias without fixing the view's import style."""
    monkeypatch.setattr(owner, name, replacement)
    if hasattr(view, name):
        monkeypatch.setattr(view, name, replacement)


def forbidden(*args, **kwargs):
    """Fail if a display-only selection attempts source loading, hashing, preparation or fitting."""
    raise AssertionError("Saved summary selectors cannot access raw/PCA sources or fit")


def protect_saved_only(monkeypatch):
    """Disable public raw/PCA/fitting/config preparation APIs while allowing saved manifest/NPZ reads."""
    for owner, name in ((timeline_io, "timeline_configuration"), (timeline_io, "load_timeline_result"),
                        (timeline_io, "load_timeline_behavior"), (timeline_runner, "run_session_timeline"),
                        (task_config, "load_task_decoding_config"), (config, "load_cross_condition_config"),
                        (modeling, "fit_region_transfers"), (run_session, "prepare_cross_condition_generalization"),
                        (results, "file_identity")):
        patch_view_dependency(monkeypatch, owner, name, forbidden)


@pytest.fixture
def portable_view(saved_transfer_bundle, metadata_args, tmp_path):
    """Return metadata/result Paths and fitted packet in a copy containing only metadata plus one final run.

    Raw behavior, base config, feature parameters and PCA archives are absent.
    Saved labels/probabilities retain N/W/H/G/S axes and dimensionless units.
    """
    path, packet = saved_transfer_bundle
    root = tmp_path / "saved_only"
    destination = root / "analysis_runs" / path.parent.name / path.name
    shutil.copytree(path.parent, destination.parent)
    metadata = root / "neural_session.json"
    shutil.copyfile(metadata_args[1] / "neural_session.json", metadata)
    return SimpleNamespace(root=root, metadata=metadata, path=destination, packet=packet)


def test_actual_metadata_app_dispatch_and_empty_availability_precede_raw_loading(portable_view, monkeypatch, tmp_path):
    """The distinct new view accepts both saved-only and empty sessions before generic spike/population loading."""
    selected = session_inputs.load_webapp_session(portable_view.metadata)
    name = session_inputs.PLOT_VIEW_CROSS_CONDITION_GENERALIZATION
    assert name == "Cross-condition generalization" and name in session_inputs.PLOT_VIEW_OPTIONS
    assert session_inputs.metadata_view_availability(selected)[name].available
    empty = tmp_path / "empty"
    empty.mkdir()
    shutil.copyfile(portable_view.metadata, empty / "neural_session.json")
    assert session_inputs.metadata_view_availability(session_inputs.load_webapp_session(empty / "neural_session.json"))[name].available
    protect_saved_only(monkeypatch)
    for function in ("load_metadata_viewer_data_cached", "load_viewer_data_cached", "load_metadata_behavior_session_cached"):
        monkeypatch.setattr(app, function, forbidden)
    at = AppTest.from_function(render_actual_app, args=(str(portable_view.metadata),)).run()
    widget(at, "Plot view").select(name).run()
    assert not at.exception and len(summary_table(at)) == 8
    assert not (portable_view.root / "task_decoding_config.json").exists()
    assert not selected.behavior.trial_table_file.exists()


def test_every_target_direction_and_saved_weight_uses_its_own_saved_summary(portable_view, monkeypatch):
    """All four directions and both regions use correct target/window/weight/role rows, without refitting."""
    expected = scoring.summarize_transfers(portable_view.packet.inputs, portable_view.packet.regions, {})
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    assert not at.exception
    for index, target in enumerate(config.TARGET_NAMES):
        widget(at, "Target").select(target).run()
        window = ("full", "before", "after", "full")[index]
        widget(at, "Supervised window").select(window).run()
        role = "training" if index % 2 else "complementary_test"
        widget(at, "Evaluation role").select(role).run()
        for weighting in config.WEIGHTING_NAMES:
            widget(at, "Weighting").select(weighting).run()
            assert not at.exception
            table = summary_table(at)
            assert set(table.Region) == {"PFC", "HPC"}
            assert set(table.Direction) == set(DIRECTIONS[target]) and len(table) == 8
            definitions = [item for item in portable_view.packet.inputs.transfers if item.target_name == target]
            for direction, definition in zip(DIRECTIONS[target], definitions, strict=True):
                for region in ("PFC", "HPC"):
                    record, = [row for row in expected if row["target_name"] == target and row["split_id"] == definition.split_id
                               and row["window"] == window and row["weighting"] == weighting and row["region"] == region
                               and row["role"] == role and row["population"] == "all"]
                    displayed = table.loc[(table.Direction == direction) & (table.Region == region)].iloc[0]
                    assert displayed["Eligible trials"] == record["eligible_trials"]
                    assert displayed["Scored trials"] == record["n_trials"]
                    assert displayed["Class 0"] == record["class_0_trials"] and displayed["Class 1"] == record["class_1_trials"]
                    assert displayed["Accuracy (%)"] == pytest.approx(100 * record["accuracy"])
                    assert displayed["Balanced accuracy (%)"] == pytest.approx(100 * record["balanced_accuracy"])
    text = rendered_text(at)
    assert "in-sample" in text and "complementary" in text and "session-wide" in text
    assert not at.get("plotly_chart") and not at.get("bidi_component") and not at.slider


def test_positive_class_captions_fit_details_and_excluded_id_audit(portable_view, monkeypatch):
    """Saved target orientations and source/test cell counts remain explicit, including conflicts and dark exclusions."""
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    for target, positive in (("current_state", "left context"), ("current_action", "left choice"),
                             ("current_reward", "rewarded"), ("current_action_is_correct", "correct")):
        widget(at, "Target").select(target).run()
        assert not at.exception
        text = rendered_text(at)
        assert "class 1" in text and positive in text
        details = fit_table(at)
        definitions = [(split, item) for split, item in enumerate(portable_view.packet.inputs.transfers) if item.target_name == target]
        assert set(details["Split ID"]) == {item.split_id for _, item in definitions}
        audit_text = details.astype(str).to_string()
        for split, definition in definitions:
            assert all(name in audit_text for name in (*definition.source_cells, *definition.test_cells))
            for region, model in portable_view.packet.regions.items():
                row = details.loc[(details["Split ID"] == definition.split_id) & (details.Region == region)].iloc[0]
                np.testing.assert_array_equal(row[["Source class 0", "Source class 1", "Test class 0", "Test class 1"]].to_numpy(),
                                              model.eligible_cell_counts[split].reshape(-1))
    text = rendered_text(at)
    assert "correctness-based" in text and "conflict" in text and "13" in text and "14" in text
    assert "dark" in text and "1" in text


def test_zero_accuracy_and_single_class_accuracy_are_displayed_as_available(portable_view, monkeypatch):
    """Zero is a valid metric; a single-class reason suppresses only balanced accuracy."""
    packet = portable_view.packet
    model = packet.regions["PFC"]
    probability = model.trial_probabilities.copy()
    selected = model.source_mask[0] | model.test_mask[0]
    probability[0, 0, 0, selected] = .9 - .8 * packet.inputs.target_values[selected, 0]
    changed = replace(packet, regions={**packet.regions, "PFC": replace(model, trial_probabilities=probability)})
    path = portable_view.root / "analysis_runs/cross_condition_generalization_20320102_zero/results.npz"
    results.save_transfer_result(path, changed)
    expected = scoring.summarize_transfers(changed.inputs, changed.regions, {})
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    assert not at.exception
    row = summary_table(at).loc[(summary_table(at).Direction == "Correct to incorrect") & (summary_table(at).Region == "PFC")].iloc[0]
    assert row["Accuracy (%)"] == 0 and row["Balanced accuracy (%)"] == 0
    widget(at, "Target").select("current_reward").run()
    widget(at, "Evaluation role").select("training").run()
    widget(at, "Outcome").select("omission").run()
    assert not at.exception
    table = summary_table(at)
    row = table.loc[(table.Direction == "Left choice to right choice") & (table.Region == "PFC")].iloc[0]
    record, = [item for item in expected if item["target_name"] == "current_reward"
               and item["split_id"] == "reward_left_choice_to_right_choice" and item["window"] == "full"
               and item["weighting"] == "equal_trial" and item["region"] == "PFC"
               and item["role"] == "training" and item["population"] == "omission"]
    assert row["Scored trials"] == row["Class 0"] == 1 and row["Class 1"] == 0
    assert row["Accuracy (%)"] == pytest.approx(100 * record["accuracy"])
    assert pd.isna(row["Balanced accuracy (%)"])
    assert "single-class" in rendered_text(at) and "balanced accuracy" in rendered_text(at)


def test_failed_fit_retains_eligible_cell_counts_while_scored_trials_are_zero(portable_view, monkeypatch):
    """A numerical failure does not turn five eligible test trials into an empty behavioral population."""
    packet, model = portable_view.packet, portable_view.packet.regions["PFC"]
    coefficients, intercepts = model.coefficients.copy(), model.intercepts.copy()
    probability, status, reason = model.trial_probabilities.copy(), model.fit_status.copy(), model.fit_reason.copy()
    coefficients[0, 0, 0], intercepts[0, 0, 0], probability[0, 0, 0] = np.nan, np.nan, np.nan
    status[0, 0, 0], reason[0, 0, 0] = "unavailable", "nonconvergence"
    failed = replace(model, coefficients=coefficients, intercepts=intercepts, trial_probabilities=probability,
                     fit_status=status, fit_reason=reason)
    path = portable_view.root / "analysis_runs/cross_condition_generalization_20320102_failed/results.npz"
    results.save_transfer_result(path, replace(packet, regions={**packet.regions, "PFC": failed}))
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    assert not at.exception
    table = summary_table(at)
    row = table.loc[(table.Direction == "Correct to incorrect") & (table.Region == "PFC")].iloc[0]
    assert row["Eligible trials"] == 5 and row["Scored trials"] == row["Class 0"] == row["Class 1"] == 0
    assert pd.isna(row["Accuracy (%)"]) and pd.isna(row["Balanced accuracy (%)"])
    assert "nonconvergence" in row["Reason"]
    details = fit_table(at)
    detail = details.loc[(details["Split ID"] == "context_correct_to_incorrect") & (details.Region == "PFC")].iloc[0]
    assert detail["Fit status"] == "unavailable"
    np.testing.assert_array_equal(detail[["Source class 0", "Source class 1", "Test class 0", "Test class 1"]].to_numpy(), [2, 3, 2, 3])


@pytest.mark.parametrize("both", [False, True])
def test_one_or_both_unavailable_regions_remain_visible_with_unknown_coverage(portable_view, monkeypatch, both):
    """Every direction retains PFC/HPC unavailable reasons, without invented regional counts or a crash."""
    reasons = {"HPC": "HPC source could not be loaded"}
    models = {"PFC": portable_view.packet.regions["PFC"]}
    if both:
        reasons["PFC"] = "No compatible PFC basis"
        models = {}
    path = portable_view.root / "analysis_runs/cross_condition_generalization_20320102_unavailable/results.npz"
    results.save_transfer_result(path, replace(portable_view.packet, regions=models, unavailable_regions=reasons))
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    assert not at.exception
    table = summary_table(at)
    assert len(table) == 8 and set(table.Region) == {"PFC", "HPC"}
    for region, reason in reasons.items():
        rows = table.loc[table.Region == region]
        assert len(rows) == 4 and rows["Eligible trials"].isna().all()
        assert rows["Scored trials"].eq(0).all() and rows["Accuracy (%)"].isna().all()
        assert rows.Reason.str.contains(reason, regex=False).all()
        assert reason.lower() in rendered_text(at)


def test_prepare_and_recompute_explicitly_request_the_full_bundle_at_selected_preparation_width(portable_view, monkeypatch):
    """Only buttons call the metadata runner; target/window/weight/outcome selectors never restrict preparation."""
    calls = []

    def prepare(metadata, **kwargs):
        """Capture one metadata Path and scalar preparation controls; publish an existing valid tiny packet as new output."""
        calls.append((Path(metadata), kwargs))
        path = portable_view.root / f"analysis_runs/cross_condition_generalization_20320102_prepared_{len(calls)}/results.npz"
        results.save_transfer_result(path, portable_view.packet)
        return run_session.TransferRunReport("prepared", path, portable_view.packet.meta["identity"], .1, {"max_model_count": 192})

    patch_view_dependency(monkeypatch, run_session, "prepare_cross_condition_generalization", prepare)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    widget(at, "Preparation bin width (ms)").select(100).run()
    widget(at, "Target").select("current_reward").run()
    widget(at, "Supervised window").select("after").run()
    widget(at, "Weighting").select("equal_cell").run()
    widget(at, "Outcome").select("omission").run()
    assert not calls and not at.exception
    assert "500 ms" in rendered_text(at)  # Current saved result keeps its own setting.
    assert "missing pca" in rendered_text(at)
    for label, rerun in (("Prepare cross-condition results", False), ("Recompute cross-condition results", True)):
        next(button for button in at.button if button.label == label).click().run()
        assert not at.exception
        metadata, kwargs = calls[-1]
        assert metadata == portable_view.metadata and kwargs["bin_width_ms"] == 100
        assert kwargs.get("rerun", False) is rerun
        assert not {"target", "target_name", "training_window", "weighting", "population", "probe_roles"} & set(kwargs)
        assert len(summary_table(at)) == 8
    assert len(calls) == 2


def test_prepare_error_preserves_previous_saved_selection_and_visible_metrics(portable_view, monkeypatch):
    """Failure after an explicit button reports its reason while leaving the selected valid run displayed."""
    newer = portable_view.root / "analysis_runs/cross_condition_generalization_20320102_newer/results.npz"
    results.save_transfer_result(newer, portable_view.packet)

    def fail_preparation(*args, **kwargs):
        """Raise an ordinary filesystem failure from an explicit metadata preparation request."""
        raise OSError("synthetic preparation unavailable")

    patch_view_dependency(monkeypatch, run_session, "prepare_cross_condition_generalization", fail_preparation)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    widget(at, "Saved run").select(str(portable_view.path)).run()
    before = summary_table(at).copy(deep=True)
    next(button for button in at.button if button.label == "Recompute cross-condition results").click().run()
    assert not at.exception and widget(at, "Saved run").value == str(portable_view.path)
    pd.testing.assert_frame_equal(summary_table(at), before)
    assert any("synthetic preparation unavailable" in item.value for item in at.error)


def test_outcome_selectors_never_borrow_trials_from_the_opposite_role(portable_view, monkeypatch):
    """The same combined-pool model supplies outcome rows; empty training incorrect remains empty."""
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    widget(at, "Evaluation role").select("training").run()
    for population, expected_count in (("correct", 5), ("correct_rewarded", 3), ("omission", 2), ("incorrect", 0)):
        widget(at, "Outcome").select(population).run()
        assert not at.exception
        table = summary_table(at)
        row = table.loc[(table.Direction == "Correct to incorrect") & (table.Region == "PFC")].iloc[0]
        assert row["Scored trials"] == row["Eligible trials"] == expected_count
        if expected_count == 0:
            assert pd.isna(row["Accuracy (%)"])
    widget(at, "Evaluation role").select("complementary_test").run()
    table = summary_table(at)
    row = table.loc[(table.Direction == "Correct to incorrect") & (table.Region == "PFC")].iloc[0]
    assert row["Scored trials"] == 5


def test_saved_controls_cache_small_products_and_invalidate_on_manifest_result_or_scoring_version(portable_view, monkeypatch):
    """Selectors share one loaded packet/report; cache identity includes both product stats and scoring version."""
    load = Mock(wraps=results.load_transfer_result)
    summarize = Mock(wraps=scoring.summarize_transfers)
    patch_view_dependency(monkeypatch, results, "load_transfer_result", load)
    patch_view_dependency(monkeypatch, scoring, "summarize_transfers", summarize)
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(portable_view.metadata),)).run()
    assert not at.exception and load.call_count == summarize.call_count == 1
    for label, value in (("Target", "current_reward"), ("Supervised window", "before"),
                         ("Weighting", "equal_cell"), ("Evaluation role", "training"), ("Outcome", "omission"),
                         ("Preparation bin width (ms)", 100)):
        widget(at, label).select(value).run()
        assert not at.exception
    assert load.call_count == summarize.call_count == 1
    assert "500 ms" in rendered_text(at) and "[-1,0)" in rendered_text(at)
    manifest = portable_view.path.parent / "config.json"
    manifest.write_text(json.dumps(json.loads(manifest.read_text()), indent=4), encoding="ascii")
    at.run()
    assert not at.exception and load.call_count == summarize.call_count == 2
    with np.load(portable_view.path, allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["meta"] = np.asarray({**arrays["meta"].item(), "ui_test_receipt": "changed-result-stat"}, dtype=object)
    np.savez_compressed(portable_view.path, **arrays)
    at.run()
    assert not at.exception and load.call_count == summarize.call_count == 3
    new_version = scoring.SCORING_VERSION + "-ui-test"
    monkeypatch.setattr(scoring, "SCORING_VERSION", new_version)
    if hasattr(view, "SCORING_VERSION"):
        monkeypatch.setattr(view, "SCORING_VERSION", new_version)
    at.run()
    assert not at.exception and summarize.call_count == 4


def test_empty_saved_view_explains_preparation_without_automatic_work(transfer_session, tmp_path, monkeypatch):
    """An empty metadata session shows useful preparation controls and missing-PCA guidance, with no source loading."""
    metadata = tmp_path / "empty" / "neural_session.json"
    metadata.parent.mkdir()
    shutil.copyfile(transfer_session[0], metadata)
    protect_saved_only(monkeypatch)
    at = AppTest.from_function(render_transfer_app, args=(str(metadata),)).run()
    assert not at.exception and not at.dataframe
    assert any(button.label == "Prepare cross-condition results" for button in at.button)
    assert "missing pca" in rendered_text(at)
    assert "no saved" in rendered_text(at) or "no completed" in rendered_text(at)
    assert not (metadata.parent / "analysis_runs").exists()


def test_manual_app_route_requests_metadata_before_generic_loading(monkeypatch):
    """Selecting the new view without metadata gives the launch instruction, without raw loading or renderer calls."""
    loaders = [Mock(side_effect=forbidden), Mock(side_effect=forbidden)]
    monkeypatch.setattr(app, "load_viewer_data_cached", loaders[0])
    monkeypatch.setattr(app, "load_metadata_viewer_data_cached", loaders[1])

    def manual_app():
        """Run the actual manual application route with no metadata arguments."""
        from src.neural_analysis.webapp.app import main

        main([])

    at = AppTest.from_function(manual_app).run()
    for load in loaders:
        load.reset_mock()
    widget(at, "Plot view").select("Cross-condition generalization").run()
    assert not at.exception and all(load.call_count == 0 for load in loaders)
    assert "--session-metadata" in rendered_text(at) and "metadata" in rendered_text(at)
