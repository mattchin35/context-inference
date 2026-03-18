import copy
import importlib

import numpy as np

from src.behavior_modeling import controller
from src.behavior_modeling.parameters import task_config
from src.behavior_modeling.task.context_task import BaseMDP


def load_switching_module():
    return importlib.import_module("src.behavior_modeling.switching")


def make_agent_params(
    *,
    fql_decay: float = 0.7,
    hmm_decay_lambda: float = 0.35,
    doubt_lambda: float = 0.4,
    hmm_transition_prob: float = 0.2,
) -> task_config.AgentParams:
    return task_config.AgentParams(
        HMM_transition_prob=hmm_transition_prob,
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=1.2,
        HMM_reward_decay_lambda=hmm_decay_lambda,
        relative_doubt_lambda=doubt_lambda,
        HMM_active_reward_probability=0.8,
        HMM_inactive_reward_probability=0.0,
        FQL_decay=fql_decay,
        logistic_alpha=0.0,
        action_temperature=1.0,
    )


def make_task_params(n_trials: int = 1) -> task_config.TaskParams:
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


def test_mode_1_only_active_agent_updates():
    switching = load_switching_module()
    task_params = make_task_params(n_trials=3)
    agent_params = make_agent_params()
    task_rng, _ = controller.create_run_rngs(seed=123)
    task = BaseMDP(task_params, rng=task_rng)

    active_agent = controller.select_agent("F-Qlearning", agent_params, task_params, rng=np.random.default_rng(1))
    inactive_agent = controller.select_agent("HMM", agent_params, task_params, rng=np.random.default_rng(2))
    initial_active_Q = active_agent.Q.copy()
    initial_inactive_prior = inactive_agent.prior.copy()

    _, agents_by_name, _ = switching.run_switched_agent_session(
        task=task,
        agents_by_name={
            "F-Qlearning": active_agent,
            "HMM": inactive_agent,
        },
        strategy_schedule=[
            {"start_trial": 0, "end_trial": 3, "strategy_name": "F-Qlearning"},
        ],
        switching_mode=1,
        task_params=task_params,
    )

    assert not np.allclose(agents_by_name["F-Qlearning"].Q, initial_active_Q)
    np.testing.assert_allclose(agents_by_name["HMM"].prior, initial_inactive_prior)


def test_mode_1_inactive_agents_keep_state_unchanged():
    switching = load_switching_module()
    task_params = make_task_params()
    agent_params = make_agent_params()

    fql_agent = controller.select_agent("F-Qlearning", agent_params, task_params, rng=np.random.default_rng(1))
    hmm_agent = controller.select_agent("HMM", agent_params, task_params, rng=np.random.default_rng(2))
    hmm_decay_doubt_agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        agent_params,
        task_params,
        rng=np.random.default_rng(3),
    )

    fql_agent.Q = np.array([0.2, 0.8], dtype=float)
    fql_agent.value = fql_agent.Q[1] - fql_agent.Q[0]
    hmm_agent.prior = np.array([0.8, 0.2], dtype=float)
    hmm_decay_doubt_agent.prior = np.array([0.75, 0.25], dtype=float)
    hmm_decay_doubt_agent.hmm_value = 0.4
    hmm_decay_doubt_agent.doubt_value = 0.3
    hmm_decay_doubt_agent.value = 0.1

    agents_before = copy.deepcopy(
        {
            "F-Qlearning": fql_agent,
            "HMM": hmm_agent,
            "HMM_reward_decay_relative_doubt": hmm_decay_doubt_agent,
        }
    )

    switching.apply_inactive_updates(
        agents_by_name={
            "F-Qlearning": fql_agent,
            "HMM": hmm_agent,
            "HMM_reward_decay_relative_doubt": hmm_decay_doubt_agent,
        },
        active_strategy="F-Qlearning",
        switching_mode=1,
    )

    np.testing.assert_allclose(fql_agent.Q, agents_before["F-Qlearning"].Q)
    np.testing.assert_allclose(hmm_agent.prior, agents_before["HMM"].prior)
    np.testing.assert_allclose(
        hmm_decay_doubt_agent.prior,
        agents_before["HMM_reward_decay_relative_doubt"].prior,
    )
    assert hmm_decay_doubt_agent.doubt_value == agents_before["HMM_reward_decay_relative_doubt"].doubt_value


def test_mode_2_fql_inactive_update_uses_fql_decay():
    switching = load_switching_module()
    task_params = make_task_params()
    agent_params = make_agent_params(fql_decay=0.6)
    fql_agent = controller.select_agent("F-Qlearning", agent_params, task_params, rng=np.random.default_rng(1))
    fql_agent.Q = np.array([0.8, 0.2], dtype=float)
    fql_agent.value = fql_agent.Q[1] - fql_agent.Q[0]

    switching.apply_inactive_updates(
        agents_by_name={"F-Qlearning": fql_agent},
        active_strategy="HMM",
        switching_mode=2,
    )

    expected_Q = np.array([0.8, 0.2], dtype=float) * agent_params.FQL_decay
    np.testing.assert_allclose(fql_agent.Q, expected_Q)
    np.testing.assert_allclose(fql_agent.value, expected_Q[1] - expected_Q[0])


def test_mode_2_hmm_inactive_update_applies_transition_matrix_to_prior():
    switching = load_switching_module()
    task_params = make_task_params()
    agent_params = make_agent_params(hmm_transition_prob=0.2)
    hmm_agent = controller.select_agent("HMM", agent_params, task_params, rng=np.random.default_rng(1))
    hmm_agent.prior = np.array([0.9, 0.1], dtype=float)
    expected_prior = np.dot(hmm_agent.transition_matrix.T, hmm_agent.prior)
    expected_prior /= expected_prior.sum()

    switching.apply_inactive_updates(
        agents_by_name={"HMM": hmm_agent},
        active_strategy="F-Qlearning",
        switching_mode=2,
    )

    np.testing.assert_allclose(hmm_agent.prior, expected_prior)


def test_mode_2_hmm_decay_doubt_inactive_update_uses_reward_and_doubt_lambdas():
    switching = load_switching_module()
    task_params = make_task_params()

    fast_belief_agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        make_agent_params(hmm_decay_lambda=0.8, doubt_lambda=0.2),
        task_params,
        rng=np.random.default_rng(1),
    )
    slow_belief_agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        make_agent_params(hmm_decay_lambda=0.2, doubt_lambda=0.2),
        task_params,
        rng=np.random.default_rng(2),
    )
    fast_doubt_agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        make_agent_params(hmm_decay_lambda=0.2, doubt_lambda=0.8),
        task_params,
        rng=np.random.default_rng(3),
    )
    slow_doubt_agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        make_agent_params(hmm_decay_lambda=0.2, doubt_lambda=0.2),
        task_params,
        rng=np.random.default_rng(4),
    )

    for agent in (fast_belief_agent, slow_belief_agent, fast_doubt_agent, slow_doubt_agent):
        agent.prior = np.array([0.9, 0.1], dtype=float)
        agent.hmm_value = agent.compute_relative_value()
        agent.left_omissions_cf = 3
        agent.right_omissions_cf = 0
        agent.doubt_value = agent.compute_doubt_value()
        agent.value = agent.hmm_value - agent.doubt_value

    initial_fast_belief_hmm_value = fast_belief_agent.hmm_value
    initial_slow_belief_hmm_value = slow_belief_agent.hmm_value
    initial_fast_doubt_value = fast_doubt_agent.doubt_value
    initial_slow_doubt_value = slow_doubt_agent.doubt_value

    switching.apply_inactive_updates(
        agents_by_name={
            "fast_belief": fast_belief_agent,
            "slow_belief": slow_belief_agent,
            "fast_doubt": fast_doubt_agent,
            "slow_doubt": slow_doubt_agent,
        },
        active_strategy="F-Qlearning",
        switching_mode=2,
    )

    assert abs(fast_belief_agent.hmm_value) < abs(initial_fast_belief_hmm_value)
    assert abs(slow_belief_agent.hmm_value) < abs(initial_slow_belief_hmm_value)
    assert abs(fast_doubt_agent.doubt_value) < abs(initial_fast_doubt_value)
    assert abs(slow_doubt_agent.doubt_value) < abs(initial_slow_doubt_value)
    assert abs(fast_belief_agent.hmm_value) < abs(slow_belief_agent.hmm_value)
    assert abs(fast_doubt_agent.doubt_value) < abs(slow_doubt_agent.doubt_value)
