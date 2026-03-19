import importlib

import numpy as np
from pandas.testing import assert_frame_equal

from src.behavior_modeling import controller
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP


STRATEGY_NAMES = (
    "F-Qlearning",
    "HMM",
    "HMM_reward_decay_relative_doubt",
)


def load_switching_module():
    return importlib.import_module("src.behavior_modeling.agent_switching.switching")


def make_agent_params() -> task_config.AgentParams:
    return task_config.AgentParams(
        HMM_transition_prob=0.2,
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=1.2,
        HMM_reward_decay_lambda=0.35,
        relative_doubt_lambda=0.4,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        FQL_decay=0.7,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )


def make_task_params(n_trials: int = 6) -> task_config.TaskParams:
    return task_config.TaskParams(
        block_transition_style="markov",
        state_transition_prob=0.2,
        active_reward_probability=0.8,
        inactive_reward_probability=0.0,
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
        n_trials=n_trials,
    )


def make_schedule():
    return [
        {"start_trial": 0, "end_trial": 2, "strategy_name": "F-Qlearning"},
        {"start_trial": 2, "end_trial": 4, "strategy_name": "HMM"},
        {
            "start_trial": 4,
            "end_trial": 6,
            "strategy_name": "HMM_reward_decay_relative_doubt",
        },
    ]


def run_switched_session(seed: int, switching_mode: int):
    switching = load_switching_module()
    task_params = make_task_params(n_trials=6)
    agent_params = make_agent_params()
    seed_sequence = np.random.SeedSequence(seed)
    task_seed, *agent_seeds = seed_sequence.spawn(1 + len(STRATEGY_NAMES))
    task = BaseMDP(task_params, rng=np.random.default_rng(task_seed))
    agents_by_name = {
        strategy_name: controller.select_agent(
            strategy_name,
            agent_params,
            task_params,
            rng=np.random.default_rng(agent_seed),
        )
        for strategy_name, agent_seed in zip(STRATEGY_NAMES, agent_seeds)
    }

    _, _, run_df = switching.run_switched_agent_session(
        task=task,
        agents_by_name=agents_by_name,
        strategy_schedule=make_schedule(),
        switching_mode=switching_mode,
        task_params=task_params,
    )
    return run_df


def comparable_columns(run_df):
    columns = [
        "state",
        "state_int",
        "cur_trial",
        "cur_trial_in_block",
        "cur_block",
        "action",
        "correct",
        "reward",
        "model_stimulus",
        "agent_action_dist",
        "agent_hmm_value",
        "agent_doubt_value",
        "agent_relative_value",
        "active_strategy",
    ]
    return run_df.loc[:, columns]


def test_mode_1_seeded_run_is_reproducible():
    df1 = comparable_columns(run_switched_session(seed=123, switching_mode=1))
    df2 = comparable_columns(run_switched_session(seed=123, switching_mode=1))

    assert_frame_equal(df1, df2)


def test_mode_2_seeded_run_is_reproducible():
    df1 = comparable_columns(run_switched_session(seed=123, switching_mode=2))
    df2 = comparable_columns(run_switched_session(seed=123, switching_mode=2))

    assert_frame_equal(df1, df2)


def test_different_seed_changes_switch_run_trajectory():
    df1 = comparable_columns(run_switched_session(seed=123, switching_mode=1))
    df2 = comparable_columns(run_switched_session(seed=456, switching_mode=1))

    assert not df1.equals(df2)
