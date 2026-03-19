from matplotlib.colors import to_rgba
import pandas as pd
import pytest

from src.visualization.agent_run_plot import (
    plot_run_dataframe,
    resolve_value_columns,
    validate_run_plot_columns,
)


def test_plot_helpers_validate_required_columns():
    run_df = pd.DataFrame({"state": ["left"], "action": [1]})

    with pytest.raises(ValueError, match="required columns"):
        validate_run_plot_columns(run_df)


def test_plot_helpers_resolve_value_columns():
    run_df = pd.DataFrame(
        {
            "state": ["left", "right"],
            "action": [1, 0],
            "reward": [1, 0],
            "agent_relative_value": [0.2, 0.1],
            "agent_hmm_value": [0.3, 0.2],
            "agent_doubt_value": [0.1, 0.1],
        }
    )

    assert resolve_value_columns(run_df) == [
        "agent_relative_value",
        "agent_hmm_value",
        "agent_doubt_value",
    ]


def make_run_df():
    return pd.DataFrame(
        {
            "state": ["left", "left", "right", "right"],
            "action": [1, 1, 0, 0],
            "reward": [1, 0, 0, 1],
            "agent_relative_value": [0.2, 0.1, -0.1, -0.2],
            "agent_hmm_value": [0.3, 0.2, -0.2, -0.3],
            "agent_doubt_value": [0.1, 0.1, 0.1, 0.1],
            "active_strategy": ["HMM", "HMM", "F-Qlearning", "F-Qlearning"],
        }
    )


def test_plot_run_dataframe_light_theme_uses_dark_combined_policy_line():
    fig, ax = plot_run_dataframe(make_run_df(), theme="light")

    combined_line = next(line for line in ax.lines if line.get_label() == "Combined policy value")
    assert to_rgba(combined_line.get_color()) == to_rgba("#3a3a3a")
    assert to_rgba(fig.get_facecolor()) == to_rgba("#ffffff")


def test_plot_run_dataframe_light_theme_uses_dark_action_and_reward_markers():
    _, ax = plot_run_dataframe(make_run_df(), theme="light")

    action_collection = next(collection for collection in ax.collections if collection.get_label() == "actions")
    reward_collection = next(collection for collection in ax.collections if collection.get_label() == "rewards")

    assert tuple(action_collection.get_facecolors()[0]) == to_rgba("#2f2f2f")
    assert tuple(reward_collection.get_facecolors()[0]) == to_rgba("#4a4a4a")


def test_plot_run_dataframe_accepts_dark_theme():
    fig, ax = plot_run_dataframe(make_run_df(), theme="dark")

    combined_line = next(line for line in ax.lines if line.get_label() == "Combined policy value")
    assert to_rgba(combined_line.get_color()) == to_rgba("#ffffff")
    assert to_rgba(fig.get_facecolor()) == to_rgba("#111111")


def test_plot_run_dataframe_rejects_unknown_theme():
    with pytest.raises(ValueError, match="theme"):
        plot_run_dataframe(make_run_df(), theme="sepia")
