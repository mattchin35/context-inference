import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src' / 'behavior_modeling'))

from src.behavior_analysis import trial_features
from src.behavior_modeling import controller
from src.behavior_modeling.agents import agents
from src.behavior_modeling.parameters import task_config


def make_reward_decay_params() -> tuple[task_config.AgentParams, task_config.TaskParams]:
    agent_params = task_config.AgentParams(
        HMM_transition_prob=0.2,
        HMM_value_mode='bayesian_log_odds',
        HMM_log_odds_tanh_scale=1.3,
        HMM_reward_decay_lambda=0.35,
        relative_doubt_lambda=0.4,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )
    task_params = task_config.TaskParams(
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
    )
    return agent_params, task_params


def test_hmm_reward_decay_requires_bayesian_log_odds():
    agent_params, task_params = make_reward_decay_params()
    agent_params.HMM_value_mode = 'expected_reward'

    with pytest.raises(ValueError, match="bayesian_log_odds"):
        agents.HMMRewardDecay(agent_params, task_params)


def test_hmm_reward_decay_matches_trial_feature_values():
    actions = np.array([1, 1, 1, 1, 0, 0, 1], dtype=int)
    rewards = np.array([1, 0, 0, 1, 0, 1, 0], dtype=float)
    agent_params, task_params = make_reward_decay_params()

    agent = agents.HMMRewardDecay(agent_params, task_params)

    online_values = []
    for action, reward in zip(actions, rewards):
        online_values.append(agent.value)
        agent.update_params(action, reward)

    expected_values = trial_features.hmm_relative_value_reward_decay(
        actions=actions,
        rewards=rewards,
        state_transition_prob=agent_params.HMM_transition_prob,
        active_reward_probability=agent_params.HMM_active_reward_probability,
        inactive_reward_probability=agent_params.HMM_inactive_reward_probability,
        correct_reward_size=task_params.mean_correct_reward,
        incorrect_reward_size=task_params.mean_incorrect_reward,
        lambda_decay=agent_params.HMM_reward_decay_lambda,
        value_mode='bayesian_log_odds',
        tanh_scale=agent_params.HMM_log_odds_tanh_scale,
    )

    npt.assert_allclose(online_values, expected_values, atol=1e-9)


def test_controller_selects_hmm_reward_decay():
    agent_params, task_params = make_reward_decay_params()

    agent = controller.select_agent('HMM_reward_decay', agent_params, task_params)

    assert agent.model_type == 'HMM_reward_decay'


def test_hmm_reward_decay_relative_doubt_matches_component_updates():
    actions = np.array([1, 1, 1, 0, 0, 1, 0], dtype=int)
    rewards = np.array([1, 0, 0, 1, 0, 0, 1], dtype=float)
    agent_params, task_params = make_reward_decay_params()

    agent = agents.HMMRewardDecayRelativeDoubt(agent_params, task_params)

    online_hmm_values = []
    online_doubt_values = []
    online_combined_values = []
    left_omissions_cf = 0
    right_omissions_cf = 0
    expected_doubt_values = []

    for action, reward in zip(actions, rewards):
        online_hmm_values.append(agent.hmm_value)
        online_doubt_values.append(agent.doubt_value)
        online_combined_values.append(agent.value)

        expected_doubt_values.append(
            trial_features.relative_doubt_index(
                R_omissions=right_omissions_cf,
                L_omissions=left_omissions_cf,
                lam=agent_params.relative_doubt_lambda,
            )
        )

        if action == 0:
            if np.isclose(reward, 0.0):
                right_omissions_cf += 1
            else:
                right_omissions_cf = 0
                left_omissions_cf = 0
        else:
            if np.isclose(reward, 0.0):
                left_omissions_cf += 1
            else:
                left_omissions_cf = 0
                right_omissions_cf = 0

        agent.update_params(action, reward)

    expected_hmm_values = trial_features.hmm_relative_value_reward_decay(
        actions=actions,
        rewards=rewards,
        state_transition_prob=agent_params.HMM_transition_prob,
        active_reward_probability=agent_params.HMM_active_reward_probability,
        inactive_reward_probability=agent_params.HMM_inactive_reward_probability,
        correct_reward_size=task_params.mean_correct_reward,
        incorrect_reward_size=task_params.mean_incorrect_reward,
        lambda_decay=agent_params.HMM_reward_decay_lambda,
        value_mode='bayesian_log_odds',
        tanh_scale=agent_params.HMM_log_odds_tanh_scale,
    )
    expected_combined_values = expected_hmm_values - np.asarray(expected_doubt_values)

    npt.assert_allclose(online_hmm_values, expected_hmm_values, atol=1e-9)
    npt.assert_allclose(online_doubt_values, expected_doubt_values, atol=1e-9)
    npt.assert_allclose(online_combined_values, expected_combined_values, atol=1e-9)


def test_controller_selects_hmm_reward_decay_relative_doubt():
    agent_params, task_params = make_reward_decay_params()

    agent = controller.select_agent('HMM_reward_decay_relative_doubt', agent_params, task_params)

    assert agent.model_type == 'HMM_reward_decay_relative_doubt'
