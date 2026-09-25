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
