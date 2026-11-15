"""Choice-only figure semantics shared between interactive and static renderers."""

import importlib

import numpy as np
import pytest

from src.tests.neural_analysis.event_sequences.example_fixtures import example_trials


def test_both_renderers_show_original_trials_reward_correctness_and_context(example_trials, tmp_path):
    """Input: full (N,C) trials and one (3,) episode. Check original identities and independent outcomes."""
    plotting = importlib.import_module("src.neural_analysis.event_sequences.example_plotting")
    view = plotting.build_example_view(
        example_trials, np.array([4, 5, 6]), session_id="synthetic-behavior",
        condition=("probe", "left", 1), trials_before=5, trials_after=5,
    )
    assert view["episode_trial_ids"] == [505, 506, 507]
    assert view["rows"].trial_id.iloc[0] == 501
    assert "correct rewarded or omission" in view["caption"]
    assert "truncated" in view["caption"]
    interactive = plotting.plotly_figure(view)
    assert interactive.layout.plot_bgcolor == "white"
    names = {trace.name for trace in interactive.data}
    assert {"Rewarded", "Unrewarded", "Incorrect"}.issubset(names)
    assert {"Left context", "Right context", "Matched episode"}.issubset(names)
    assert list(interactive.layout.yaxis.ticktext) == ["Right", "Left"]
    assert interactive.layout.shapes
    static = plotting.matplotlib_figure(view)
    assert static.get_facecolor() == (1, 1, 1, 1)
    assert [label.get_text() for label in static.axes[0].get_yticklabels()] == ["Right", "Left"]
    assert static.axes[0].patches
    path = plotting.export_png(view, tmp_path / "example.png")
    assert path.stat().st_size > 5000
    with pytest.raises(FileExistsError):
        plotting.export_png(view, path)


def test_unknown_reward_and_manual_rows_are_distinct_and_missing_choices_break_lines(example_trials):
    """Input: context rows with unknown reward/manual reward/non-choice. Do not infer outcomes."""
    plotting = importlib.import_module("src.neural_analysis.event_sequences.example_plotting")
    trials = example_trials.copy()
    trials.iloc[0, trials.columns.get_loc("reward")] = np.nan
    trials.iloc[0, trials.columns.get_loc("experimenter_reward_given")] = 1
    view = plotting.build_example_view(trials, np.array([4, 5, 6]), session_id="test", condition=("probe", "left", 1))
    figure = plotting.plotly_figure(view)
    names = {trace.name for trace in figure.data}
    assert {"Unknown reward", "Manual reward"}.issubset(names)
    choices = next(trace for trace in figure.data if trace.name == "Choices")
    assert np.isnan(np.asarray(choices.y, dtype=float)[3])
    assert choices.connectgaps is False


def test_transition_caption_describes_its_own_reward_rule(example_trials):
    """Input: saved-style (N,C) trials and ideal transition (S,) positions. Keep its definition explicit."""
    plotting = importlib.import_module("src.neural_analysis.event_sequences.example_plotting")
    examples = importlib.import_module("src.neural_analysis.event_sequences.examples")
    key = ("ideal_transition", "right", 1)
    positions = examples.find_behavior_examples(example_trials)[key][0]
    view = plotting.build_example_view(example_trials, positions, session_id="test", condition=key)
    assert "rewarded opposite-side choice" in view["caption"]
    assert "correct rewarded or omission" not in view["caption"]


@pytest.mark.parametrize("family", ["probe", "omission_probe", "ideal_transition", "lucky_omission_switch", "indecisive_transition"])
def test_presentation_caption_fits_canvas_with_full_session_identity(example_trials, family):
    """Input: synthetic (N,C) rows and a full session label. Caption must fit and avoid the legend."""
    plotting = importlib.import_module("src.neural_analysis.event_sequences.example_plotting")
    examples = importlib.import_module("src.neural_analysis.event_sequences.examples")
    matches = examples.find_behavior_examples(example_trials)
    key = next(key for key, positions in matches.items() if key[0] == family and len(positions))
    view = plotting.build_example_view(
        example_trials, matches[key][0], session_id="CT026_2026-08-28_134630", condition=key,
    )
    figure = plotting.matplotlib_figure(view)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    caption = figure.texts[0].get_window_extent(renderer)
    legend = figure.axes[0].get_legend().get_window_extent(renderer)
    assert caption.x0 >= 0 and caption.x1 <= figure.bbox.x1 - 5
    assert caption.y0 >= 0 and caption.y1 < legend.y0 - 5
