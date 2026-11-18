"""Presentation-contract tests for benchmark report figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.behavior_analysis.behavior_model_benchmark import (
    build_score_table,
    run_benchmark,
)
from src.behavior_analysis.behavior_model_benchmark_plots import (
    plot_combined_sample_agreement,
    plot_model_behavior_atlas,
    save_benchmark_figure,
)
from src.behavior_modeling.benchmark_models import build_observable_models
from src.behavior_modeling.benchmark_scenarios import build_default_scenarios


def test_full_atlas_uses_two_balanced_pages_with_every_sample_once() -> None:
    """The 22-sample suite is balanced across two 11-panel pages."""
    scenarios = build_default_scenarios()
    model = build_observable_models()[0]
    result = run_benchmark(scenarios, (model,), include_oracle=False)

    figures = plot_model_behavior_atlas(
        scenarios,
        result.replays,
        model.model_id,
        model.display_label,
        model.parameters,
    )

    assert len(figures) == 2
    assert [len(figure.axes) for figure in figures] == [11, 11]
    plotted_titles = [axis.get_title() for figure in figures for axis in figure.axes]
    assert plotted_titles == [scenario.scenario_id for scenario in scenarios]
    for figure in figures:
        assert tuple(round(value, 1) for value in figure.get_size_inches()) == (
            24.0,
            12.0,
        )
        plt.close(figure)


def test_sample_panel_overlays_behavior_and_model_value_without_choice_line() -> None:
    """Choices are sized scatter markers on the same signed axis as model value."""
    scenarios = build_default_scenarios()[7:8]
    model = build_observable_models()[8]
    result = run_benchmark(scenarios, (model,), include_oracle=False)

    figure = plot_model_behavior_atlas(
        scenarios,
        result.replays,
        model.model_id,
        model.display_label,
        model.parameters,
    )[0]
    axis = figure.axes[0]

    assert axis.get_ylim()[0] <= -1.05
    assert axis.get_ylim()[1] >= 1.05
    model_lines = [line for line in axis.lines if line.get_label() == "Model value"]
    assert len(model_lines) == 1
    np.testing.assert_array_equal(
        model_lines[0].get_xdata(), np.arange(scenarios[0].action.size)
    )
    assert len(axis.collections) == 1
    choice_markers = axis.collections[0]
    assert choice_markers.get_offsets().shape[0] == scenarios[0].action.size
    assert np.unique(choice_markers.get_sizes()).size == 2
    assert all(line.get_label() != "Observed choice" for line in axis.lines)
    assert axis.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    plt.close(figure)


def test_context_is_labeled_without_continuous_correct_side_trace() -> None:
    """Initial context and reversals use text plus dashed vertical boundaries."""
    scenario = next(
        item for item in build_default_scenarios() if item.scenario_id == "EX-03"
    )
    model = build_observable_models()[0]
    result = run_benchmark((scenario,), (model,), include_oracle=False)
    figure = plot_model_behavior_atlas(
        (scenario,),
        result.replays,
        model.model_id,
        model.display_label,
        model.parameters,
    )[0]
    axis = figure.axes[0]

    context_labels = [text.get_text() for text in axis.texts]
    assert "L" in context_labels
    assert "R" in context_labels
    dashed_lines = [line for line in axis.lines if line.get_linestyle() == "--"]
    assert len(dashed_lines) == 1
    assert dashed_lines[0].get_xdata()[0] == 2.5
    assert all(line.get_label() != "Correct side" for line in axis.lines)
    plt.close(figure)


def test_single_trial_panel_keeps_model_value_visible() -> None:
    """A one-row sample needs a small value tick because no line segment exists."""
    scenario = build_default_scenarios()[0]
    model = build_observable_models()[0]
    result = run_benchmark((scenario,), (model,), include_oracle=False)
    figure = plot_model_behavior_atlas(
        (scenario,),
        result.replays,
        model.model_id,
        model.display_label,
        model.parameters,
    )[0]

    labels = [collection.get_label() for collection in figure.axes[0].collections]
    assert "Model value (single trial)" in labels
    plt.close(figure)


def test_trial_axes_use_only_integer_trial_indices() -> None:
    """Discrete trial counts must never be labeled with fractional ticks."""
    scenarios = build_default_scenarios()[:4]
    model = build_observable_models()[0]
    result = run_benchmark(scenarios, (model,), include_oracle=False)
    figure = plot_model_behavior_atlas(
        scenarios,
        result.replays,
        model.model_id,
        model.display_label,
        model.parameters,
    )[0]

    for axis, scenario in zip(figure.axes, scenarios):
        ticks = axis.get_xticks()
        np.testing.assert_array_equal(ticks, np.arange(scenario.action.size))
    plt.close(figure)


def test_combined_agreement_contains_each_model_and_chance_reference() -> None:
    """One descriptive figure compares all selected sample-level agreements."""
    scenarios = build_default_scenarios()[:4]
    models = build_observable_models()[:3]
    result = run_benchmark(scenarios, models, include_oracle=False)
    score_table = build_score_table(result)
    labels = {model.model_id: model.display_label for model in models}

    figure = plot_combined_sample_agreement(score_table, labels)
    axis = figure.axes[0]

    assert {line.get_label() for line in axis.lines} == {
        *labels.values(),
        "0.5 reference",
    }
    assert axis.get_ylim() == (0.0, 1.0)
    assert [tick.get_text() for tick in axis.get_xticklabels()] == [
        scenario.scenario_id for scenario in scenarios
    ]
    assert figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    plt.close(figure)


def test_save_benchmark_figure_writes_png_and_closes_only_target(tmp_path) -> None:
    """Saving is explicit, opaque, and does not close unrelated figures."""
    target_figure = plt.figure()
    other_figure = plt.figure()
    target_number = target_figure.number
    other_number = other_figure.number
    output_path = tmp_path / "diagnostic.png"

    save_benchmark_figure(target_figure, output_path)

    assert output_path.is_file()
    assert target_number not in plt.get_fignums()
    assert other_number in plt.get_fignums()
    plt.close(other_figure)


def test_save_benchmark_figure_preserves_precomputed_caption_layout(
    tmp_path,
    monkeypatch,
) -> None:
    """Saving must not rerun tight_layout after caption space was reserved."""
    figure = plt.figure()

    def fail_if_relaid_out(*_args, **_kwargs):
        raise AssertionError("save must preserve the figure's finalized layout")

    monkeypatch.setattr(figure, "tight_layout", fail_if_relaid_out)
    save_benchmark_figure(figure, tmp_path / "caption-layout.png")

    assert (tmp_path / "caption-layout.png").is_file()
