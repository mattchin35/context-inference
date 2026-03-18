import pandas as pd
import pytest

from src.visualization.agent_run_plot import resolve_value_columns, validate_run_plot_columns


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
