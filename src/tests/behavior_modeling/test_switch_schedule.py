import importlib

import numpy as np
import pytest

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


def make_agents(
    agent_params: task_config.AgentParams,
    task_params: task_config.TaskParams,
    seed: int = 123,
):
    seed_sequence = np.random.SeedSequence(seed)
    agent_rngs = [
        np.random.default_rng(child_seed)
        for child_seed in seed_sequence.spawn(len(STRATEGY_NAMES))
    ]
    return {
        strategy_name: controller.select_agent(
            strategy_name,
            agent_params,
            task_params,
            rng=agent_rng,
        )
        for strategy_name, agent_rng in zip(STRATEGY_NAMES, agent_rngs)
    }


def test_schedule_assigns_active_strategy_per_trial():
    switching = load_switching_module()

    labels = switching.build_active_strategy_labels(
        make_schedule(),
        n_trials=6,
        allowed_strategies=STRATEGY_NAMES,
    )

    assert list(labels) == [
        "F-Qlearning",
        "F-Qlearning",
        "HMM",
        "HMM",
        "HMM_reward_decay_relative_doubt",
        "HMM_reward_decay_relative_doubt",
    ]


def test_schedule_rejects_gaps():
    switching = load_switching_module()
    schedule = [
        {"start_trial": 0, "end_trial": 2, "strategy_name": "F-Qlearning"},
        {"start_trial": 3, "end_trial": 6, "strategy_name": "HMM"},
    ]

    with pytest.raises(ValueError, match="gap"):
        switching.build_active_strategy_labels(
            schedule,
            n_trials=6,
            allowed_strategies=STRATEGY_NAMES,
        )


def test_schedule_rejects_overlaps():
    switching = load_switching_module()
    schedule = [
        {"start_trial": 0, "end_trial": 3, "strategy_name": "F-Qlearning"},
        {"start_trial": 2, "end_trial": 6, "strategy_name": "HMM"},
    ]

    with pytest.raises(ValueError, match="overlap"):
        switching.build_active_strategy_labels(
            schedule,
            n_trials=6,
            allowed_strategies=STRATEGY_NAMES,
        )


def test_schedule_rejects_unknown_strategy():
    switching = load_switching_module()
    schedule = [
        {"start_trial": 0, "end_trial": 3, "strategy_name": "F-Qlearning"},
        {"start_trial": 3, "end_trial": 6, "strategy_name": "unknown"},
    ]

    with pytest.raises(ValueError, match="unknown"):
        switching.build_active_strategy_labels(
            schedule,
            n_trials=6,
            allowed_strategies=STRATEGY_NAMES,
        )


def test_schedule_must_cover_trial_range_exactly():
    switching = load_switching_module()
    schedule = [
        {"start_trial": 0, "end_trial": 2, "strategy_name": "F-Qlearning"},
        {"start_trial": 2, "end_trial": 5, "strategy_name": "HMM"},
    ]

    with pytest.raises(ValueError, match="cover"):
        switching.build_active_strategy_labels(
            schedule,
            n_trials=6,
            allowed_strategies=STRATEGY_NAMES,
        )


def test_run_dataframe_records_active_strategy_column():
    switching = load_switching_module()
    task_params = make_task_params(n_trials=6)
    agent_params = make_agent_params()
    task_rng, _ = controller.create_run_rngs(seed=123)
    task = BaseMDP(task_params, rng=task_rng)
    agents_by_name = make_agents(agent_params, task_params, seed=456)

    _, _, run_df = switching.run_switched_agent_session(
        task=task,
        agents_by_name=agents_by_name,
        strategy_schedule=make_schedule(),
        switching_mode=1,
        task_params=task_params,
    )

    assert "active_strategy" in run_df.columns
    assert list(run_df["active_strategy"]) == [
        "F-Qlearning",
        "F-Qlearning",
        "HMM",
        "HMM",
        "HMM_reward_decay_relative_doubt",
        "HMM_reward_decay_relative_doubt",
    ]
