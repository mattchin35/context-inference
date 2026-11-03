"""Prediction tracks retain real-time gaps, PC identity and held-out values."""

from dataclasses import replace

import numpy as np
import pytest

from src.neural_analysis.population.timeline_plotting import plot_session_timeline
from src.neural_analysis.task_decoding.session_prediction import plotting
from src.neural_analysis.task_decoding.session_prediction.results import SessionPredictions
from src.neural_analysis.task_decoding.session_prediction.preparation import prepare_prediction_inputs
from .test_modeling import fit_region


@pytest.fixture
def display_result(prediction_inputs):
    """Return fitted synthetic session records with N-choice labels and W,B,T outputs."""
    base, trials, timeline = prediction_inputs
    prepared = prepare_prediction_inputs(trials, timeline, base)
    regions = {role: fit_region(prepared, role) for role in timeline.regions}
    return SessionPredictions(prepared, regions, {"identity": "display-fixture"})


def test_prediction_steps_actual_overlays_and_intertrial_breaks(display_result):
    """Prediction segments occupy exact bins and actual values survive regional gaps."""
    display = plotting.PredictionDisplay(display_result, "qlearning_relative_value", pc_mode="hidden")
    timeline = display_result.inputs.timeline
    figure = plot_session_timeline(timeline, prediction_display=display)
    prediction = next(trace for trace in figure.data if trace.name == "PFC prediction")
    actual = next(trace for trace in figure.data if trace.name == "PFC actual")
    x = np.asarray(prediction.x, dtype=float)
    first_choice = timeline.choices.elapsed_s.iloc[0]
    np.testing.assert_array_equal(x[:4], first_choice + np.array([-2, -1.5, -1.5, -1]))
    assert np.isnan(x[16])
    assert prediction.connectgaps is False
    assert len(actual.x) == len(timeline.choices) * 3
    np.testing.assert_allclose(np.asarray(prediction.y[:16], dtype=float),
                               np.repeat(display_result.regions["PFC"].predictions[0, :, 2], 2))
    assert not any(trace.type == "heatmap" for trace in figure.data)


def test_binary_probability_axes_and_logit_contribution_labels(display_result):
    """Binary panels show P(left); weighted rows stay in additive log-odds units."""
    display = plotting.PredictionDisplay(display_result, "current_action", pc_mode="weighted")
    figure = plot_session_timeline(display_result.inputs.timeline, prediction_display=display)
    prediction = next(trace for trace in figure.data if trace.name == "PFC prediction")
    axis = figure.layout["yaxis" + prediction.yaxis.removeprefix("y")]
    assert tuple(axis.range) == (0, 1)
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap" and trace.name == "PFC")
    assert "log-odds" in heatmap.colorbar.title.text
    assert "weight" in heatmap.hovertemplate.lower()
    assert "PC" in heatmap.hovertemplate
    assert "actual" in heatmap.hovertemplate.lower()
    assert "prediction" in heatmap.hovertemplate.lower()
    assert "Action generalization" in heatmap.hovertemplate


def test_ordered_rows_use_sequential_centers_original_labels_and_correct_weights(display_result):
    """Display order (3,1,2) changes neither score-weight pairing nor color limits."""
    models = display_result.regions.copy()
    model = models["PFC"]
    order = model.contribution_pc_order.copy()
    order[2] = [3, 1, 2]
    models["PFC"] = replace(model, contribution_pc_order=order)
    saved = replace(display_result, regions=models)
    display = plotting.PredictionDisplay(saved, "qlearning_relative_value", pc_mode="weighted", pc_order="contribution", pc_count=3)
    figure = plot_session_timeline(saved.inputs.timeline, prediction_display=display)
    heatmap = next(trace for trace in figure.data if trace.type == "heatmap" and trace.name == "PFC")
    np.testing.assert_array_equal(heatmap.y, [0, 1, 2])
    axis = figure.layout["yaxis" + heatmap.yaxis.removeprefix("y")]
    assert tuple(axis.ticktext) == ("PC3", "PC1", "PC2")
    assert axis.autorange == "reversed"
    region = saved.inputs.timeline.regions["PFC"]
    fold = saved.inputs.fold_ids[region.choice_positions[0]]
    expected = (region.scores[0][:, [2, 0, 1]] * model.coefficients[2, fold, [2, 0, 1]]).T
    np.testing.assert_allclose(np.asarray(heatmap.z)[:, :8], expected)
    assert heatmap.zmax == model.contribution_color_limit[2]
    assert np.asarray(heatmap.customdata)[0, 0, 8] == 3


def test_display_changes_preserve_scores_time_revision_and_global_limits(display_result, monkeypatch):
    """Plotting has no fits/writes; target/PC changes retain horizontal navigation revision."""
    from src.neural_analysis.task_decoding.session_prediction import modeling

    def forbidden(*args, **kwargs):
        """Plotting must never reach fitting; fail rather than return."""
        pytest.fail("Display cannot fit")

    monkeypatch.setattr(modeling, "fit_region_prediction_models", forbidden)
    timeline = display_result.inputs.timeline
    original = timeline.regions["PFC"].scores.copy()
    display = plotting.PredictionDisplay(display_result, "qlearning_relative_value", pc_mode="weighted")
    full = plot_session_timeline(timeline, viewport_revision=5, prediction_display=display)
    small = plot_session_timeline(timeline, loaded_range_s=(0, 20), viewport_revision=5,
                                  prediction_display=replace(display, pc_count=1, pc_order="contribution"))
    assert full.layout.uirevision == small.layout.uirevision
    first = next(trace for trace in full.data if trace.type == "heatmap" and trace.name == "PFC")
    second = next(trace for trace in small.data if trace.type == "heatmap" and trace.name == "PFC")
    assert first.zmax == second.zmax
    assert len(second.x) < len(first.x)
    np.testing.assert_array_equal(timeline.regions["PFC"].scores, original)


def test_missing_region_keeps_actual_overlay_and_unavailable_panel(display_result):
    """A missing HPC model leaves all actual choices visible and PFC predictions usable."""
    timeline = replace(display_result.inputs.timeline,
                       regions={"PFC": display_result.inputs.timeline.regions["PFC"]},
                       unavailable={"HPC": "No recording"})
    saved = replace(display_result, inputs=replace(display_result.inputs, timeline=timeline),
                    regions={"PFC": display_result.regions["PFC"]})
    figure = plot_session_timeline(timeline, prediction_display=plotting.PredictionDisplay(saved, "current_state"))
    assert any(trace.name == "HPC actual" for trace in figure.data)
    assert any("unavailable" in annotation.text.lower() for annotation in figure.layout.annotations)


@pytest.mark.parametrize("mode, label, training", [
    ("correct_rewarded_generalization", "generalization", "correct-rewarded training"),
    ("mixed_outcome_decoding", "decoding", "mixed-outcome training"),
])
@pytest.mark.parametrize("target", ["current_state", "current_action"])
def test_binary_mode_labels_reach_titles_and_all_neural_hover(prediction_inputs, mode, label, training, target):
    """Visible labels disclose selected mode/training for predictions and contributions."""
    base, trials, timeline = prediction_inputs
    prepared = prepare_prediction_inputs(trials, timeline, base, analysis_mode=mode)
    saved = SessionPredictions(prepared, {role: fit_region(prepared, role, use_training_membership=True)
                                         for role in timeline.regions}, {})
    display = plotting.PredictionDisplay(saved, target, pc_mode="weighted")
    figure = plot_session_timeline(timeline, prediction_display=display)
    title = f"{target.removeprefix('current_').capitalize()} {label}"
    assert any(f"{title} ({training})" in annotation.text for annotation in figure.layout.annotations)
    for trace in figure.data:
        if trace.name == "PFC prediction" or trace.type == "heatmap":
            assert title in trace.hovertemplate
            assert training in trace.hovertemplate


def test_horizontal_identity_uses_session_origin_while_vertical_uses_basis(display_result):
    """Different PCA IDs/PC counts preserve time; different origins reset coordinates."""
    timeline = display_result.inputs.timeline
    changed_region = replace(timeline.regions["PFC"], scores=timeline.regions["PFC"].scores[:, :, :1])
    changed = replace(timeline, regions={**timeline.regions, "PFC": changed_region},
                      meta={**timeline.meta, "configuration_id": "another-basis", "bin_width_s": 0.1})
    first = plot_session_timeline(timeline, viewport_revision=2)
    second = plot_session_timeline(changed, viewport_revision=2)
    assert first.layout.uirevision == second.layout.uirevision
    assert first.layout.yaxis.uirevision != second.layout.yaxis.uirevision
    shifted = plot_session_timeline(replace(changed, time_origin_s=changed.time_origin_s + 1), viewport_revision=2)
    assert shifted.layout.uirevision != second.layout.uirevision
