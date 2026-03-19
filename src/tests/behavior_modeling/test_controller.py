import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.behavior_modeling import controller
from src.behavior_modeling.visualize_behavior import agent_run_plot as agent_run_plot_wrapper
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP


ROOT = Path(__file__).resolve().parents[3]


def make_params() -> tuple[task_config.AgentParams, task_config.TaskParams]:
    agent_params = task_config.AgentParams(
        HMM_transition_prob=0.2,
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=1.2,
        HMM_reward_decay_lambda=0.35,
        relative_doubt_lambda=0.4,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )
    task_params = task_config.TaskParams(
        block_transition_style="markov",
        state_transition_prob=0.2,
        active_reward_probability=0.8,
        inactive_reward_probability=0.0,
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
        n_trials=8,
    )
    return agent_params, task_params


def run_seeded_session(seed: Optional[int]) -> pd.DataFrame:
    agent_params, task_params = make_params()
    task_params.n_trials = 20
    task_rng, agent_rng = controller.create_run_rngs(seed)
    task = BaseMDP(task_params, rng=task_rng)
    agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        agent_params,
        task_params,
        rng=agent_rng,
    )
    _, _, performance_df = controller.run_agent_session(task, agent, task_params)
    return performance_df


def test_performance_dataframe_captures_component_values():
    agent_params, task_params = make_params()
    task = BaseMDP(task_params)
    agent = controller.select_agent("HMM_reward_decay_relative_doubt", agent_params, task_params)

    _, _, performance = controller.run_experiment(task, agent, task_params)
    performance_df = controller.performance_to_dataframe(performance)

    assert "agent_hmm_value" in performance_df.columns
    assert "agent_doubt_value" in performance_df.columns
    assert performance_df["agent_hmm_value"].notna().any()
    assert performance_df["agent_doubt_value"].notna().any()


def test_controller_plot_run_dataframe_saves_plot(tmp_path):
    save_path = tmp_path / "controller_run_plot.png"
    script = f"""
from pathlib import Path
from src.behavior_modeling import controller
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP

agent_params = task_config.AgentParams(
    HMM_transition_prob=0.2,
    HMM_value_mode='bayesian_log_odds',
    HMM_log_odds_tanh_scale=1.2,
    HMM_reward_decay_lambda=0.35,
    relative_doubt_lambda=0.4,
    HMM_active_reward_probability=0.8,
    HMM_inactive_reward_probability=0.0,
    logistic_alpha=0.0,
    action_temperature=1.0,
)
task_params = task_config.TaskParams(
    block_transition_style='markov',
    state_transition_prob=0.2,
    active_reward_probability=0.8,
    inactive_reward_probability=0.0,
    mean_correct_reward=1.0,
    mean_incorrect_reward=0.0,
    reward_std_dev=0.0,
    n_trials=8,
)
task = BaseMDP(task_params)
agent = controller.select_agent('HMM_reward_decay_relative_doubt', agent_params, task_params)
_, _, performance = controller.run_experiment(task, agent, task_params)
performance_df = controller.performance_to_dataframe(performance)
controller.plot_run_dataframe(
    performance_df,
    'HMM_reward_decay_relative_doubt',
    task_params,
    agent_params,
    plot_save_path=Path(r'{save_path}'),
    show_plot=False,
)
print('plot-ok')
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )

    if "Can't open SHM2" in result.stderr:
        pytest.skip("matplotlib subprocess plotting is blocked by sandboxed OpenMP shared-memory restrictions")
    assert result.returncode == 0, result.stderr
    assert save_path.exists()


def test_resolve_run_output_paths_separates_plot_and_run_targets(tmp_path):
    agent_params, task_params = make_params()
    options = controller.RunOutputOptions(
        save_run=False,
        save_plot=True,
        show_plot=False,
        run_data_dir=tmp_path / "runs",
    )

    run_path, params_path, plot_path = controller.resolve_run_output_paths(
        "HMM_reward_decay_relative_doubt",
        task_params,
        agent_params,
        options,
        sample_agent_data_dir=tmp_path / "sample_agents",
        experiment_data_dir=tmp_path / "experiments",
        session_note="unit-test",
    )

    assert run_path is None
    assert params_path is None
    assert plot_path is not None
    assert plot_path.suffix == ".png"


def test_handle_run_outputs_reports_saved_paths(tmp_path, monkeypatch, capsys):
    agent_params, task_params = make_params()
    task = BaseMDP(task_params)
    agent = controller.select_agent("HMM_reward_decay_relative_doubt", agent_params, task_params)
    _, _, performance = controller.run_experiment(task, agent, task_params)
    performance_df = controller.performance_to_dataframe(performance)

    def fake_plot_run_outputs(*args, **kwargs):
        save_path = kwargs.get("plot_path")
        if save_path is not None:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_text("plot")
        return object(), None

    monkeypatch.setattr(controller, "plot_run_outputs", fake_plot_run_outputs)

    artifacts = controller.handle_run_outputs(
        task,
        agent,
        performance_df,
        "HMM_reward_decay_relative_doubt",
        task_params,
        agent_params,
        controller.RunOutputOptions(
            save_run=True,
            save_plot=True,
            show_plot=False,
            run_data_dir=tmp_path / "sample_agents",
        ),
        sample_agent_data_dir=tmp_path / "sample_agents",
        experiment_data_dir=tmp_path / "experiments",
        session_note="unit-test",
    )

    output = capsys.readouterr().out
    assert artifacts.run_path is not None and artifacts.run_path.exists()
    assert artifacts.plot_path is not None and artifacts.plot_path.exists()
    assert "Agent run saved to:" in output
    assert "Plot saved to:" in output


def test_handle_run_outputs_reports_display_only(monkeypatch, capsys):
    agent_params, task_params = make_params()
    task = BaseMDP(task_params)
    agent = controller.select_agent("HMM_reward_decay_relative_doubt", agent_params, task_params)
    _, _, performance = controller.run_experiment(task, agent, task_params)
    performance_df = controller.performance_to_dataframe(performance)

    monkeypatch.setattr(controller, "plot_run_outputs", lambda *args, **kwargs: (object(), None))

    controller.handle_run_outputs(
        task,
        agent,
        performance_df,
        "HMM_reward_decay_relative_doubt",
        task_params,
        agent_params,
        controller.RunOutputOptions(
            save_run=False,
            save_plot=False,
            show_plot=True,
        ),
        sample_agent_data_dir=ROOT / "tmp_unused",
        experiment_data_dir=ROOT / "tmp_unused",
    )

    output = capsys.readouterr().out
    assert "Plot displayed (not saved to disk)." in output


def test_seeded_runs_are_reproducible():
    df1 = run_seeded_session(seed=123)
    df2 = run_seeded_session(seed=123)

    assert_frame_equal(df1, df2)


def test_different_seeds_change_run_trajectory():
    df1 = run_seeded_session(seed=123)
    df2 = run_seeded_session(seed=456)

    assert not df1.equals(df2)


def test_unseeded_runs_still_execute():
    df = run_seeded_session(seed=None)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 20


def test_controller_plot_run_dataframe_forwards_theme(monkeypatch):
    agent_params, task_params = make_params()
    performance_df = pd.DataFrame(
        {
            "state": ["left"],
            "action": [1],
            "reward": [1],
            "agent_relative_value": [0.2],
            "agent_hmm_value": [0.3],
            "agent_doubt_value": [0.1],
        }
    )
    captured = {}

    def fake_plot_run_dataframe(*args, **kwargs):
        captured["theme"] = kwargs.get("theme")
        return object(), None

    monkeypatch.setattr(agent_run_plot_wrapper, "plot_run_dataframe", fake_plot_run_dataframe)

    controller.plot_run_dataframe(
        performance_df,
        "HMM_reward_decay_relative_doubt",
        task_params,
        agent_params,
        theme="dark",
    )

    assert captured["theme"] == "dark"
