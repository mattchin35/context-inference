"""Integration contracts for candidate expectant models in mouse analysis."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from src.behavior_analysis import gather_trial_features as gtf
from src.behavior_analysis import performance_plots
from src.behavior_analysis import plot_model_values
from src.behavior_analysis import session_analysis

sys.modules.setdefault("ssm", ModuleType("ssm"))
ssm_plots_stub = ModuleType("ssm.plots")
ssm_plots_stub.gradient_cmap = lambda *_args, **_kwargs: None
ssm_plots_stub.white_to_color_cmap = lambda *_args, **_kwargs: None
sys.modules.setdefault("ssm.plots", ssm_plots_stub)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import main  # noqa: E402


CANDIDATE_VALUE_COLUMNS = {
    "probe_persistence_doubt_mouse_agreement": "probe_persistence_doubt_value",
    "home_referenced_expectancy_doubt_mouse_agreement": (
        "home_referenced_expectancy_doubt_value"
    ),
}


def _candidate_trial_table() -> pd.DataFrame:
    """Build a minimal aligned real-session-style trial table."""
    return pd.DataFrame(
        {
            "state": ["left", "left", "right", "right"],
            "action": [1, 0, 0, 1],
            "reward": [1.0, 1.0, 0.0, 0.0],
            "experimenter_reward_given": [0, 1, 0, 0],
            "cur_block": [0, 0, 1, 1],
            "switch_trial": [False, True, False, True],
            "prev_correct": [False, True, False, True],
            "explore_trial": [False, True, False, False],
        }
    )


def test_candidate_features_use_explicit_defaults_and_skip_manual_rewards() -> None:
    """Mouse-session features expose canonical parameters and valid-row rules."""
    params = gtf.TaskParams()
    assert params.probe_doubt_persistence_weight == 1.0
    assert params.probe_doubt_probe_weight == 2.0
    assert params.probe_doubt_doubt_weight == 1.4
    assert params.probe_doubt_lambda == 0.5
    assert params.home_expectancy_threshold == 2.5
    assert params.home_expectancy_scale == 0.5
    assert params.home_expectancy_weight == 1.0
    assert params.home_doubt_weight == 1.0
    assert params.home_doubt_lambda == 0.5

    result = gtf.collect_candidate_expectant_switching_features(
        _candidate_trial_table(),
        params,
    )
    for value_column in CANDIDATE_VALUE_COLUMNS.values():
        assert value_column in result.columns
    assert result.loc[1, "probe_persistence_doubt_value"] is None
    assert result.loc[2, "probe_persistence_doubt_previous_choice"] == 1.0
    assert (
        result.loc[2, "home_referenced_expectancy_doubt_confirmed_side"]
        == 1.0
    )
    assert result.loc[2, "home_referenced_expectancy_doubt_reward_count"] == 1.0


def test_candidate_models_are_default_block_agreement_agents() -> None:
    """Per-session and cross-session agreement registries include both models."""
    for agreement_column, value_column in CANDIDATE_VALUE_COLUMNS.items():
        assert (
            session_analysis.DEFAULT_AGENT_MOUSE_AGREEMENT_VALUE_COLUMNS[
                agreement_column
            ]
            == value_column
        )
    assert (
        performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS["Probe+P+D"]
        == "probe_persistence_doubt_mouse_agreement"
    )
    assert (
        performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS["Home E+D"]
        == "home_referenced_expectancy_doubt_mouse_agreement"
    )


def test_candidate_value_plot_is_separate_wide_light_panel(monkeypatch, tmp_path) -> None:
    """The candidate pair receives its own readable behavior/value figure."""
    params = gtf.TaskParams()
    trial_df = gtf.collect_candidate_expectant_switching_features(
        _candidate_trial_table(),
        params,
    )
    captured = {}

    def fake_plot_run_dataframe(run_df, **kwargs):
        fig, axis = plt.subplots()
        captured["kwargs"] = kwargs
        captured["figure"] = fig
        for column in kwargs["value_columns"]:
            axis.plot(run_df[column].to_numpy(), label=column)
        return fig, axis

    monkeypatch.setattr(
        plot_model_values,
        "plot_run_dataframe",
        fake_plot_run_dataframe,
    )
    _plot_df, save_path = (
        plot_model_values.plot_mouse_history_candidate_expectant_switching_values(
            trial_df,
            tmp_path,
            "CT016_test",
            params,
        )
    )

    assert captured["kwargs"]["value_columns"] == [
        "probe_persistence_doubt_value",
        "home_referenced_expectancy_doubt_value",
    ]
    assert captured["kwargs"]["theme"] == "light"
    assert captured["kwargs"]["figsize"] == (24, 5)
    assert save_path.name == "CT016_test_mouse_history_candidate_expectant_values.png"
    assert save_path.exists()


def test_candidate_diagnostic_has_aligned_block_agreement(tmp_path) -> None:
    """The four-panel diagnostic uses trial-coordinate block spans."""
    params = gtf.TaskParams()
    trial_df = gtf.collect_candidate_expectant_switching_features(
        _candidate_trial_table(),
        params,
    )
    spans = performance_plots.prepare_candidate_expectant_block_agreement_spans(
        trial_df
    )
    save_path = performance_plots.plot_candidate_expectant_switching_diagnostics(
        trial_df,
        tmp_path,
        "CT016_test",
    )

    assert spans[["trial_start", "trial_end"]].to_numpy().tolist() == [
        [-0.5, 1.5],
        [1.5, 3.5],
    ]
    assert set(CANDIDATE_VALUE_COLUMNS).issubset(spans.columns)
    assert save_path.name == "CT016_test_candidate_expectant_switching.png"
    assert save_path.exists()


def test_saved_session_refresh_updates_only_model_outputs(monkeypatch, tmp_path) -> None:
    """Refresh loads saved CSVs and never enters preprocessing, HMM, or GLM code."""
    session = SimpleNamespace(
        sess_id_full="CT016_2026-05-29_132812",
        processed_data_path=tmp_path,
        figure_path=tmp_path / "figures",
    )
    session.figure_path.mkdir()
    trial_path = tmp_path / f"{session.sess_id_full}_augmented_trials.csv"
    block_path = tmp_path / f"{session.sess_id_full}_block_performance.csv"
    _candidate_trial_table().to_csv(trial_path, index=False)
    pd.DataFrame(
        {"block_ix": [0, 1], "block_type": ["left_uncued", "right_uncued"]}
    ).to_csv(block_path, index=False)

    monkeypatch.setattr(
        main,
        "plot_expectant_switching_for_session",
        lambda **_kwargs: {"candidate": session.figure_path / "candidate.png"},
    )
    monkeypatch.setattr(
        main.performance_plots,
        "plot_session_agent_mouse_agreement",
        lambda *_args, **_kwargs: None,
    )
    refreshed_trials, refreshed_blocks, plot_paths = (
        main.refresh_expectant_model_outputs_for_session(
            session,
            gtf.TaskParams(),
        )
    )

    for value_column in CANDIDATE_VALUE_COLUMNS.values():
        assert value_column in refreshed_trials.columns
    for agreement_column in CANDIDATE_VALUE_COLUMNS:
        assert agreement_column in refreshed_blocks.columns
    assert set(plot_paths) == {"candidate"}
    assert "probe_persistence_doubt_value" in pd.read_csv(trial_path).columns
    assert "probe_persistence_doubt_mouse_agreement" in pd.read_csv(block_path).columns
