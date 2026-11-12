"""Real Streamlit prediction controls and raw-source-free app dispatch."""

import json
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pandas as pd

from streamlit.testing.v1 import AppTest

from src.neural_analysis.task_decoding.session_prediction.run_session import prepare_session_predictions
from src.neural_analysis.task_decoding.session_prediction.results import load_prediction_result
from src.neural_analysis.population.run_window_variance import run_window_variance
from src.neural_analysis.webapp import app, session_inputs
from src.tests.neural_analysis.task_decoding.session_prediction.test_scripts import (
    runnable_session as runnable_fixture, synthetic_timeline_session as timeline_fixture,
)

synthetic_timeline_session = timeline_fixture
runnable_session = runnable_fixture


def render_prediction_test_app(metadata_path):
    """Render the actual timeline view for one metadata path; all scientific data are saved."""
    import streamlit as st
    from src.neural_analysis.webapp.session_inputs import load_webapp_session
    from src.neural_analysis.webapp.timeline_view import render_session_timeline_view

    render_session_timeline_view(st, load_webapp_session(metadata_path))


def figure_spec(test_app):
    """Return the persistent timeline component's JSON figure, with numeric elapsed-second axes."""
    component = test_app.get("bidi_component")[0]
    return json.loads(json.loads(component.proto.json)["figure_json"])


def component_path(block):
    """Return tuple[int,...] chart position in an AppTest Block tree, or None if absent."""
    for index, child in block.children.items():
        if child.type == "bidi_component":
            return (index,)
        if hasattr(child, "children"):
            path = component_path(child)
            if path is not None:
                return (index, *path)
    return None


def test_explicit_prepare_and_display_changes_never_refit(runnable_session, monkeypatch):
    """Buttons prepare; subsequent target/mode/event changes load only saved arrays."""
    from src.neural_analysis.webapp import session_prediction_view

    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in at.selectbox if widget.label == "HPC probe").select("B").run()
    assert not list(runnable_session.parent.glob("analysis_runs/condition_session_prediction_*"))
    next(widget for widget in at.button if widget.label == "Prepare predictions").click().run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])

    def forbidden(*args, **kwargs):
        """Any fit on a display-only rerun is a failure; return never."""
        raise AssertionError("Display cannot prepare predictions")

    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    count = len(list(runnable_session.parent.glob("analysis_runs/*")))
    next(widget for widget in at.selectbox if widget.label == "Prediction target").select("relative_doubt").run()
    next(widget for widget in at.selectbox if widget.label == "Prediction PC view").select("weighted").run()
    assert any(widget.label == "PFC PC ranks" for widget in at.slider)
    assert not any("Default color limits" in item.value for item in at.caption)
    before = figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Prediction PC order").select("contribution").run()
    assert not at.exception
    assert figure_spec(at)["layout"]["uirevision"] == before
    assert len(list(runnable_session.parent.glob("analysis_runs/*"))) == count


def test_relocated_saved_display_and_actual_app_dispatch_without_raw_sources(runnable_session, tmp_path, monkeypatch):
    """The earlier availability gate and complete renderer both accept saved-only copies."""
    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"})
    copied = tmp_path / "copied"
    shutil.copytree(runnable_session.parent / "analysis_runs", copied / "analysis_runs")
    shutil.copyfile(runnable_session, copied / "neural_session.json")
    path = copied / "neural_session.json"
    session = session_inputs.load_webapp_session(path)
    assert not session.behavior.trial_table_file.exists()
    assert session_inputs.metadata_view_availability(session)[session_inputs.PLOT_VIEW_SESSION_TIMELINE].available
    fake_st = SimpleNamespace(sidebar=SimpleNamespace(caption=Mock(), selectbox=Mock(return_value=session_inputs.PLOT_VIEW_SESSION_TIMELINE)))
    renderer = Mock()
    monkeypatch.setattr(app, "st", fake_st)
    monkeypatch.setattr(app, "render_session_timeline_view", renderer)
    assert app._start_metadata_webapp(session) is None
    renderer.assert_called_once_with(fake_st, session)
    at = AppTest.from_function(render_prediction_test_app, args=(str(path),)).run()
    assert not at.exception
    names = [trace.get("name") for trace in figure_spec(at)["data"]]
    assert "PFC prediction" in names and "HPC prediction" in names
    assert (copied / report.result_path.relative_to(runnable_session.parent)).is_file()


def test_hidden_pc_view_does_not_offer_invisible_window_count_tracks(runnable_session):
    """Controls for heatmap-attached variance rows stay hidden when PC rows are hidden."""
    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"})
    saved = load_prediction_result(report.result_path)
    run_window_variance(runnable_session.parent / saved.meta["source_pca"]["path"])
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    assert not any(widget.label == "Show fixed-axis window counts" for widget in at.checkbox)


@pytest.mark.parametrize("mode", ["correct_rewarded_generalization", "mixed_outcome_decoding"])
def test_saved_only_unprepared_switch_and_hidden_predictions_keep_anchor(runnable_session, tmp_path, monkeypatch, mode):
    """Raw-free mode switches retain saved behavior/PCA and suppress every model overlay."""
    from src.neural_analysis.webapp import session_prediction_view, timeline_view

    report = prepare_session_predictions(
        runnable_session, probe_roles={"PFC": "A", "HPC": "B"}, analysis_mode=mode,
    )
    copied = tmp_path / "portable"
    shutil.copytree(runnable_session.parent / "analysis_runs", copied / "analysis_runs")
    shutil.copyfile(runnable_session, copied / "neural_session.json")

    def forbidden(*args, **kwargs):
        """Any preparation/raw lookup violates portable saved inspection; return never."""
        raise AssertionError("Saved-only switching cannot use raw inputs or fit")

    monkeypatch.setattr(timeline_view.timeline_io, "timeline_configuration", forbidden)
    monkeypatch.setattr(timeline_view, "load_timeline_behavior_cached", forbidden)
    monkeypatch.setattr(timeline_view, "run_session_timeline", forbidden)
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    at = AppTest.from_function(render_prediction_test_app, args=(str(copied / "neural_session.json"),)).run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])
    assert next(widget for widget in at.selectbox if widget.label == "Analysis").value == mode
    next(widget for widget in at.checkbox if widget.label == "Show predictions").uncheck().run()
    assert not at.exception
    assert not any("prediction" in trace.get("name", "") or "actual" in trace.get("name", "")
                   for trace in figure_spec(at)["data"])
    assert any(trace.get("name") == "PFC" for trace in figure_spec(at)["data"])
    next(widget for widget in at.checkbox if widget.label == "Show predictions").check().run()
    next(widget for widget in at.selectbox if widget.label == "Prediction PC view").select("weighted").run()
    other = "mixed_outcome_decoding" if mode == "correct_rewarded_generalization" else "correct_rewarded_generalization"
    before = figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Analysis").select(other).run()
    assert not at.exception
    spec = figure_spec(at)
    assert spec["layout"]["uirevision"] == before
    assert any("not prepared" in caption.value for caption in at.caption)
    assert any("500 ms" in caption.value and "background" in caption.value.lower() for caption in at.caption)
    assert not any("prediction" in trace.get("name", "") or "actual" in trace.get("name", "")
                   for trace in spec["data"])
    assert any(trace.get("name") == "PFC" for trace in spec["data"])
    assert next(widget for widget in at.button if widget.label == "Recompute predictions").disabled
    next(widget for widget in at.selectbox if widget.label == "Analysis").select(mode).run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])
    assert (copied / report.result_path.relative_to(runnable_session.parent)).exists()


def test_cross_configuration_mode_switch_retains_navigation_and_clear_labels(runnable_session, monkeypatch):
    """500/100-ms mode switches keep time navigation while controls follow saved basis."""
    from src.neural_analysis.webapp import session_prediction_view

    roles = {"PFC": "A", "HPC": "B"}
    reference = prepare_session_predictions(runnable_session, probe_roles=roles)
    mixed = prepare_session_predictions(runnable_session, probe_roles=roles, bin_width_ms=100,
                                        analysis_mode="mixed_outcome_decoding")
    assert load_prediction_result(reference.result_path).meta["source_pca"]["configuration_id"] != load_prediction_result(mixed.result_path).meta["source_pca"]["configuration_id"]
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", Mock(side_effect=AssertionError("No refit")))
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    next(widget for widget in at.number_input if widget.label == "Visible duration (s)").set_value(40.0).run()
    next(widget for widget in at.number_input if widget.label == "Window start (s)").set_value(50.0).run()
    next(widget for widget in at.selectbox if widget.label == "Jump to trial").select(9).run()
    initial = figure_spec(at)
    initial_component_path = component_path(at.main)
    assert any("State generalization (correct-rewarded training)" in item["text"]
               for item in initial["layout"]["annotations"])
    next(widget for widget in at.selectbox if widget.label == "Analysis").select("mixed_outcome_decoding").run()
    assert not at.exception
    updated = figure_spec(at)
    assert component_path(at.main) == initial_component_path
    assert updated["layout"]["uirevision"] == initial["layout"]["uirevision"]
    assert updated["layout"]["xaxis"]["range"] == initial["layout"]["xaxis"]["range"]
    assert updated["layout"]["yaxis"]["uirevision"] != initial["layout"]["yaxis"]["uirevision"]
    assert next(widget for widget in at.number_input if widget.label == "Visible duration (s)").value == 40
    assert next(widget for widget in at.number_input if widget.label == "Window start (s)").value == 50
    assert next(widget for widget in at.selectbox if widget.label == "Jump to trial").value == 9
    target_options = next(widget for widget in at.selectbox if widget.label == "Prediction target").options
    assert target_options[:3] == ["State decoding", "Action decoding", "Reward decoding"]
    assert len(target_options) == 8
    assert "Mixed-outcome decoding" in next(widget for widget in at.selectbox if widget.label == "Saved predictions").options[0]
    assert any("State decoding (mixed-outcome training)" in item["text"] for item in updated["layout"]["annotations"])
    assert any("sample size" in item.value and "regularization" in item.value for item in at.caption)
    saved = load_prediction_result(mixed.result_path)
    details = at.dataframe[0].value
    for fold in range(5):
        mask = saved.regions["PFC"].training_choice_mask[0, fold]
        assert details.loc[fold, "training_trials"] == mask.sum()
        for condition in ("correct_rewarded", "omission", "incorrect"):
            index = saved.inputs.condition_names.index(condition)
            assert details.loc[fold, condition] == (mask & saved.inputs.condition_masks[:, index]).sum()
        assert details.loc[fold, "left_class_1"] + details.loc[fold, "right_class_0"] == mask.sum()
    next(widget for widget in at.button if widget.label == "Jump").click().run()
    assert figure_spec(at)["layout"]["uirevision"] != updated["layout"]["uirevision"]


def test_gui_prepare_forwards_selected_mode_and_bounded_caption(runnable_session):
    """Explicit mixed Prepare yields eight-target products and explains bounded resets."""
    prepare_session_predictions(runnable_session, probe_roles={"PFC": "A"})
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    next(widget for widget in at.selectbox if widget.label == "PFC probe").select("A").run()
    next(widget for widget in at.selectbox if widget.label == "Timeline bin width (ms)").select(20).run()
    next(widget for widget in at.selectbox if widget.label == "Analysis").select("mixed_outcome_decoding").run()
    next(widget for widget in at.selectbox if widget.label == "Training/scoring window").select("before").run()
    next(widget for widget in at.button if widget.label == "Prepare predictions").click().run()
    assert not at.exception
    products = [load_prediction_result(path) for path in runnable_session.parent.glob("analysis_runs/condition_session_prediction_*/results.npz")]
    mixed = next(product for product in products if product.inputs.analysis_mode == "mixed_outcome_decoding")
    assert mixed.inputs.target_names == ("current_state", "current_action", "current_reward",
                                         "qlearning_relative_value", "forgetting_q_relative_value",
                                         "hmm_signed_belief", "hmm_decay_signed_belief", "relative_doubt")
    assert mixed.meta["effective_config"]["bin_width_ms"] == 20
    assert mixed.inputs.training_window == "before"
    assert next(widget for widget in at.checkbox if widget.label == "Load only visible neural interval").value
    next(widget for widget in at.checkbox if widget.label == "Load only visible neural interval").check().run()
    assert any("reset" in item.value.lower() and "loaded" in item.value.lower() for item in at.caption)


@pytest.mark.parametrize("mode", ["correct_rewarded_generalization", "mixed_outcome_decoding"])
def test_continuous_details_do_not_describe_values_as_binary_classes(runnable_session, mode):
    """Continuous targets show training/outcome counts without binary class labels."""
    prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"}, analysis_mode=mode)
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    next(widget for widget in at.selectbox if widget.label == "Prediction target").select("relative_doubt").run()
    assert not at.exception
    assert "left_class_1" not in at.dataframe[0].value.columns
    assert "right_class_0" not in at.dataframe[0].value.columns


@pytest.mark.parametrize("mode", ["correct_rewarded_generalization", "mixed_outcome_decoding"])
def test_saved_performance_panel_laterality_population_and_cache(runnable_session, tmp_path, monkeypatch, mode):
    """Portable saved scores show both lateralities and reuse small summaries on display changes."""
    from src.neural_analysis.webapp import session_prediction_view, timeline_view

    trials = pd.read_csv(runnable_session.parent / "augmented.csv")
    trials.loc[5, "correct"] = 0
    trials.loc[5, "state_int"] = 1 - trials.loc[5, "action"]
    trials.loc[5, "reward"] = 0
    for name in ("augmented.csv", "trials.csv"):
        trials.to_csv(runnable_session.parent / name, index=False)
    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"}, analysis_mode=mode)
    copied = tmp_path / "portable_performance"
    shutil.copytree(runnable_session.parent / "analysis_runs", copied / "analysis_runs")
    shutil.copyfile(runnable_session, copied / "neural_session.json")
    unchanged = {path: path.read_bytes() for path in copied.rglob("*") if path.is_file()}

    def forbidden(*args, **kwargs):
        """No fitting/raw path is allowed after copying portable products; return never."""
        raise AssertionError("Performance display cannot fit or use raw inputs")

    monkeypatch.setattr(timeline_view.timeline_io, "timeline_configuration", forbidden)
    monkeypatch.setattr(timeline_view, "load_timeline_behavior_cached", forbidden)
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    original = session_prediction_view.compute_prediction_performance
    compute = Mock(wraps=original)
    monkeypatch.setattr(session_prediction_view, "compute_prediction_performance", compute)
    at = AppTest.from_function(render_prediction_test_app, args=(str(copied / "neural_session.json"),)).run()
    assert not at.exception
    assert compute.call_count == 1
    assert any(expander.label == "Prediction performance" for expander in at.expander)
    overall = next(item.value for item in at.dataframe if "Population" in item.value.columns
                   and "Grouping" not in item.value.columns and "Fold" not in item.value.columns)
    assert set(overall["Population"]) >= {"All eligible choices", "Training outcomes", "Correct-rewarded", "Incorrect"}
    assert overall["Accuracy (%)"].dropna().between(0, 100).all()
    folds = next(item.value for item in at.dataframe if "Split" in item.value.columns and "Grouping" not in item.value.columns)
    assert set(folds["Split"]) == {"Training", "Held-out"}
    assert set(folds["Fold"]) == set(range(5))
    assert any("[-1,+1)" in item.value and "trial" in item.value for item in at.caption)
    before_path, before_revision = component_path(at.main), figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Laterality scoring population").select("incorrect").run()
    assert not at.exception
    sides = next(item.value for item in at.dataframe if "Side" in item.value.columns)
    assert set(sides["Grouping"]) == {"Choice", "State/context"}
    assert set(sides["Side"]) == {"Left", "Right"}
    assert set(sides["Region"]) == {"PFC", "HPC"}
    assert sides.groupby(["Region", "Grouping"])["Trials"].sum().eq(1).all()
    assert sides.loc[sides["Trials"] == 0, "Accuracy (%)"].isna().all()
    gaps = next(item.value for item in at.dataframe if "Accuracy difference (pp)" in item.value.columns)
    assert gaps["Accuracy difference (pp)"].isna().all()  # One side is absent in each one-trial group.
    assert compute.call_count == 1
    next(widget for widget in at.selectbox if widget.label == "Prediction PC view").select("weighted").run()
    assert compute.call_count == 1
    assert component_path(at.main) == before_path
    assert figure_spec(at)["layout"]["uirevision"] == before_revision
    next(widget for widget in at.selectbox if widget.label == "Prediction target").select("current_action").run()
    assert compute.call_count == 2
    assert not at.exception
    assert all(path.read_bytes() == content for path, content in unchanged.items())
    assert len(unchanged) == len([path for path in copied.rglob("*") if path.is_file()])
    assert (copied / report.result_path.relative_to(runnable_session.parent)).exists()


@pytest.mark.parametrize("mode", ["correct_rewarded_generalization", "mixed_outcome_decoding"])
def test_continuous_performance_panel_has_native_errors_and_no_class_accuracy(runnable_session, mode):
    """Continuous target changes expose R2/RMSE and meaningful side-error differences."""
    prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"}, analysis_mode=mode)
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    next(widget for widget in at.selectbox if widget.label == "Prediction target").select("relative_doubt").run()
    assert not at.exception
    overall = next(item.value for item in at.dataframe if "Population" in item.value.columns
                   and "Grouping" not in item.value.columns and "Fold" not in item.value.columns)
    assert "R2" in overall.columns and "RMSE" in overall.columns
    assert "Accuracy (%)" not in overall.columns
    assert overall["RMSE"].dropna().ge(0).all()
    assert any("target units" in item.value and "RMSE" in item.value for item in at.caption)
    assert not any("0.5 left/right threshold" in item.value for item in at.caption)
    assert not any("Single-class groups" in item.value for item in at.caption)
    gaps = next(item.value for item in at.dataframe if "RMSE difference" in item.value.columns)
    assert "R2 difference" in gaps.columns


def test_saved_window_pair_initialization_and_unprepared_switch_keep_anchor(runnable_session, tmp_path, monkeypatch):
    """A Before-only saved session initializes honestly; an unprepared After keeps background and pan scope."""
    from src.neural_analysis.webapp import session_prediction_view, timeline_view

    prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"},
                                analysis_mode="mixed_outcome_decoding", training_window="before")
    copied = tmp_path / "window_portable"
    shutil.copytree(runnable_session.parent / "analysis_runs", copied / "analysis_runs")
    shutil.copyfile(runnable_session, copied / "neural_session.json")
    forbidden = Mock(side_effect=AssertionError("Saved switching must not read raw inputs or fit"))
    monkeypatch.setattr(timeline_view.timeline_io, "timeline_configuration", forbidden)
    monkeypatch.setattr(timeline_view, "load_timeline_behavior_cached", forbidden)
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    at = AppTest.from_function(render_prediction_test_app, args=(str(copied / "neural_session.json"),)).run()
    assert not at.exception
    assert next(widget for widget in at.selectbox if widget.label == "Training/scoring window").value == "before"
    next(widget for widget in at.selectbox if widget.label == "Prediction target").select("current_reward").run()
    assert not at.exception
    assert any("Before [-1,0) s" in item.value and "trial" in item.value for item in at.caption)
    assert any("eventual" in item.value and "reward" in item.value for item in at.caption)
    assert not any("0.5 left/right threshold" in item.value for item in at.caption)
    details = at.dataframe[0].value
    assert "rewarded_class_1" in details.columns and "unrewarded_class_0" in details.columns
    assert "left_class_1" not in details.columns
    before_path, revision = component_path(at.main), figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Training/scoring window").select("after").run()
    assert not at.exception
    assert any("not prepared" in item.value for item in at.caption)
    assert component_path(at.main) == before_path
    assert figure_spec(at)["layout"]["uirevision"] == revision
    assert any(trace.get("name") == "PFC" for trace in figure_spec(at)["data"])
    assert not any("prediction" in trace.get("name", "") or "actual" in trace.get("name", "")
                   for trace in figure_spec(at)["data"])
    next(widget for widget in at.selectbox if widget.label == "Training/scoring window").select("before").run()
    assert not at.exception
    assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])
    forbidden.assert_not_called()


def test_performance_split_fold_controls_filter_cached_records(runnable_session, monkeypatch):
    """Training outcome/side rows remain per model and no filter repeats neural reconstruction."""
    from src.neural_analysis.webapp import session_prediction_view

    prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"})
    compute = Mock(wraps=session_prediction_view.compute_prediction_performance)
    monkeypatch.setattr(session_prediction_view, "compute_prediction_performance", compute)
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    next(widget for widget in at.selectbox if widget.label == "Scoring split").select("training").run()
    assert next(widget for widget in at.selectbox if widget.label == "Scoring fold").value == 0
    next(widget for widget in at.selectbox if widget.label == "Laterality scoring population").select("omission").run()
    assert not at.exception
    sides = next(item.value for item in at.dataframe if "Side" in item.value.columns)
    assert set(sides["Split"]) == {"Training"} and set(sides["Fold"]) == {0}
    assert sides["Trials"].eq(0).all()
    assert sides["Accuracy (%)"].isna().all()
    next(widget for widget in at.selectbox if widget.label == "Scoring fold").select(2).run()
    assert compute.call_count == 1
    next(widget for widget in at.selectbox if widget.label == "Scoring split").select("held_out").run()
    next(widget for widget in at.selectbox if widget.label == "Scoring fold").select(None).run()
    assert not at.exception
    sides = next(item.value for item in at.dataframe if "Side" in item.value.columns)
    assert set(sides["Split"]) == {"Held-out"}
    assert sides["Trials"].sum() > 0
    assert compute.call_count == 1


@pytest.mark.parametrize("version", [2, 3])
def test_legacy_mixed_view_keeps_saved_targets_and_explains_preparation(runnable_session, version):
    """Historical products stay inspectable without presenting additional targets as prepared."""
    from src.tests.neural_analysis.task_decoding.session_prediction.test_results import write_legacy_product

    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"},
                                        analysis_mode="mixed_outcome_decoding")
    saved = load_prediction_result(report.result_path)
    write_legacy_product((runnable_session.parent, report.result_path, saved.inputs, saved.regions, saved.meta),
                         version, "mixed_outcome_decoding")
    before = report.result_path.read_bytes()
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    expected = ["State decoding", "Action decoding"] + (["Reward decoding"] if version == 3 else [])
    assert next(widget for widget in at.selectbox if widget.label == "Prediction target").options == expected
    assert any("legacy" in item.value.lower() and "qlearning_relative_value" in item.value for item in at.caption)
    assert next(widget for widget in at.selectbox if widget.label == "Training/scoring window").value == "full"
    assert report.result_path.read_bytes() == before


@pytest.mark.parametrize("window", ["full", "before", "after"])
def test_mixed_continuous_target_switches_use_saved_predictions(runnable_session, monkeypatch, window):
    """All five numerical displays reuse saved W,B,T products and preserve navigation.

    Inputs are a synthetic metadata path and a fixed choice-relative window ID;
    no return. Assert native-unit scoring and weighted-PC display across target
    changes, with fitting forbidden and saved archive bytes unchanged.
    """
    from src.neural_analysis.webapp import session_prediction_view

    report = prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"},
                                         analysis_mode="mixed_outcome_decoding", training_window=window)
    before = report.result_path.read_bytes()
    forbidden = Mock(side_effect=AssertionError("Display must not fit"))
    monkeypatch.setattr(session_prediction_view, "prepare_session_predictions", forbidden)
    at = AppTest.from_function(render_prediction_test_app, args=(str(runnable_session),)).run()
    assert not at.exception
    revision = figure_spec(at)["layout"]["uirevision"]
    next(widget for widget in at.selectbox if widget.label == "Prediction PC view").select("weighted").run()
    for target in ("qlearning_relative_value", "forgetting_q_relative_value", "hmm_signed_belief",
                   "hmm_decay_signed_belief", "relative_doubt"):
        next(widget for widget in at.selectbox if widget.label == "Prediction target").select(target).run()
        assert not at.exception
        assert figure_spec(at)["layout"]["uirevision"] == revision
        assert any(trace.get("name") == "PFC prediction" for trace in figure_spec(at)["data"])
        assert any(trace.get("type") == "heatmap" for trace in figure_spec(at)["data"])
        overall = next(item.value for item in at.dataframe if "Population" in item.value.columns
                       and "Grouping" not in item.value.columns and "Fold" not in item.value.columns)
        assert "R2" in overall.columns and "RMSE" in overall.columns
        assert "Accuracy (%)" not in overall.columns
    forbidden.assert_not_called()
    assert report.result_path.read_bytes() == before
