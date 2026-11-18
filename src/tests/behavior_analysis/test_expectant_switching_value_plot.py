"""Tests for the mouse-history expectant-switching value plot."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.behavior_analysis import plot_model_values


def test_mouse_history_expectant_value_plot_is_one_light_panel(monkeypatch, tmp_path):
    """Both model values should share the ideal-observer-style action axis."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "left", "left", "right", "right"],
            "action": [1, "no_choice", 0, 0, 1],
            "reward": [1.0, 0.0, 0.0, 1.0, 1.0],
            "experimenter_reward_given": [0, 0, 1, 0, 0],
            "switch_trial": [False, False, False, False, False],
            "prev_correct": [False, False, False, False, False],
            "explore_trial": [False, False, False, False, False],
            "simple_probe_persistence_value": [0.0, None, None, -0.4, -0.2],
            "expectancy_persistence_doubt_value": [0.0, None, None, -0.6, 0.1],
        }
    )
    params = SimpleNamespace(
        expectancy_threshold=3.0,
        expectancy_scale=1.0,
        omission_lam=0.5,
        simple_persistence_weight=1.0,
        simple_probe_weight=1.0,
        full_persistence_weight=1.0,
        full_expectancy_weight=1.0,
        full_doubt_weight=1.0,
    )
    captured = {}

    def fake_plot_run_dataframe(run_df, **kwargs):
        captured["run_df"] = run_df.copy()
        captured["kwargs"] = kwargs
        fig, ax = plt.subplots()
        captured["fig"] = fig
        for column in kwargs["value_columns"]:
            ax.plot(run_df[column].to_numpy(), label=column)
        return fig, ax

    monkeypatch.setattr(plot_model_values, "plot_run_dataframe", fake_plot_run_dataframe)

    plotted_df, save_path = plot_model_values.plot_mouse_history_expectant_switching_values(
        trial_df=trial_df,
        figure_path=tmp_path,
        sess_id_full="CT016_2026-06-10_120000",
        params=params,
    )

    assert plotted_df.index.tolist() == [0, 3, 4]
    assert captured["kwargs"]["value_columns"] == [
        "simple_probe_persistence_value",
        "expectancy_persistence_doubt_value",
    ]
    assert captured["kwargs"]["theme"] == "light"
    assert captured["kwargs"]["show_legend"] is True
    assert captured["kwargs"]["figsize"] == (24, 5)
    assert len(captured["fig"].axes) == 1
    assert save_path == tmp_path / "CT016_2026-06-10_120000_mouse_history_expectant_switching_values.png"
    assert save_path.exists()


def test_expectant_value_parameter_caption_lists_tunable_values(monkeypatch, tmp_path):
    """Saved tuning plots should record every active exemplar parameter."""
    trial_df = pd.DataFrame(
        {
            "state": ["left"],
            "action": [1],
            "reward": [1.0],
            "experimenter_reward_given": [0],
            "switch_trial": [False],
            "prev_correct": [False],
            "explore_trial": [False],
            "simple_probe_persistence_value": [0.0],
            "expectancy_persistence_doubt_value": [0.0],
        }
    )
    params = SimpleNamespace(
        expectancy_threshold=4.0,
        expectancy_scale=1.5,
        omission_lam=0.25,
        simple_persistence_weight=0.8,
        simple_probe_weight=1.2,
        full_persistence_weight=0.9,
        full_expectancy_weight=1.3,
        full_doubt_weight=1.4,
    )
    captured = {}

    def fake_plot_run_dataframe(run_df, **kwargs):
        fig, ax = plt.subplots()
        captured["fig"] = fig
        return fig, ax

    monkeypatch.setattr(plot_model_values, "plot_run_dataframe", fake_plot_run_dataframe)

    plot_model_values.plot_mouse_history_expectant_switching_values(
        trial_df,
        tmp_path,
        "CT016_test",
        params,
    )

    caption = " ".join(text.get_text() for text in captured["fig"].texts)
    for expected in (
        "k0=4",
        "scale=1.5",
        "lambda=0.25",
        "simple(wP=0.8, wR=1.2)",
        "full(wP=0.9, wE=1.3, wD=1.4)",
    ):
        assert expected in caption


def test_prepare_expectant_switch_markers_uses_existing_trial_flags_only():
    """Existing flags should partition reward and omission switches without overlap."""
    trial_df = pd.DataFrame(
        {
            "action": [1, 0, 0, 1, 0],
            "switch_trial": [False, True, False, True, True],
            "prev_correct": [False, True, False, True, False],
            "explore_trial": [False, True, False, False, False],
        },
        index=[10, 20, 30, 40, 50],
    )

    markers = plot_model_values.prepare_expectant_switch_markers(
        trial_df,
        valid_rows=np.ones(trial_df.shape[0], dtype=bool),
    )

    assert markers.to_dict("records") == [
        {
            "source_position": 0,
            "plot_position": 0,
            "signed_action": 1.0,
            "marker_role": "correct_predecessor",
        },
        {
            "source_position": 1,
            "plot_position": 1,
            "signed_action": -1.0,
            "marker_role": "reward_to_switch",
        },
        {
            "source_position": 2,
            "plot_position": 2,
            "signed_action": -1.0,
            "marker_role": "correct_predecessor",
        },
        {
            "source_position": 3,
            "plot_position": 3,
            "signed_action": 1.0,
            "marker_role": "omission_to_switch",
        },
    ]


def test_expectant_value_plot_overlays_distinct_existing_switch_categories(
    monkeypatch,
    tmp_path,
):
    """The value plot should distinguish strict explore and omission switches."""
    trial_df = pd.DataFrame(
        {
            "state": ["left"] * 4,
            "action": [1, 0, 0, 1],
            "reward": [1.0, 0.0, 0.0, 0.0],
            "experimenter_reward_given": [0, 0, 0, 0],
            "switch_trial": [False, True, False, True],
            "prev_correct": [False, True, False, True],
            "explore_trial": [False, True, False, False],
            "simple_probe_persistence_value": [0.2, -0.3, -0.2, 0.4],
            "expectancy_persistence_doubt_value": [0.1, -0.4, -0.3, 0.5],
        }
    )
    params = SimpleNamespace(
        expectancy_threshold=3.0,
        expectancy_scale=1.0,
        omission_lam=0.5,
        simple_persistence_weight=1.0,
        simple_probe_weight=1.0,
        full_persistence_weight=1.0,
        full_expectancy_weight=1.0,
        full_doubt_weight=1.0,
    )
    captured = {}

    def fake_plot_run_dataframe(run_df, **kwargs):
        fig, ax = plt.subplots()
        captured["fig"] = fig
        return fig, ax

    monkeypatch.setattr(plot_model_values, "plot_run_dataframe", fake_plot_run_dataframe)

    plot_model_values.plot_mouse_history_expectant_switching_values(
        trial_df,
        tmp_path,
        "CT016_switch_overlay",
        params,
    )

    collections = {
        collection.get_label(): collection
        for collection in captured["fig"].axes[0].collections
    }
    assert set(collections) == {
        "Correct before switch",
        "Reward -> switch",
        "Omission -> switch",
    }
    assert collections["Correct before switch"].get_offsets().tolist() == [
        [0.0, 1.0],
        [2.0, -1.0],
    ]
    assert collections["Reward -> switch"].get_offsets().tolist() == [[1.0, -1.0]]
    assert collections["Omission -> switch"].get_offsets().tolist() == [[3.0, 1.0]]
    assert collections["Correct before switch"].get_facecolors().size == 0
    assert collections["Correct before switch"].get_sizes()[0] > (
        collections["Reward -> switch"].get_sizes()[0]
    )
    plt.close(captured["fig"])
