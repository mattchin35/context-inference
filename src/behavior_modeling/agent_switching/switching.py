"""Switching helpers for multi-strategy behavior-model runs.

This module adds an explicit switching layer on top of the existing single-agent
controller workflow. The current implementation supports:
- mode 1: inactive agents do not update
- mode 2: inactive agents apply passive strategy-specific updates
- mode 3: all agents update from the executed action/reward stream
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.behavior_modeling import controller
from src.behavior_modeling.agent_switching import parallel_agents
from src.behavior_modeling.agents import agents
from src.behavior_modeling.task.context_task import BaseMDP


SWITCH_TRIAL_DF_COLUMNS = controller.TRIAL_DF_COLUMNS + ["active_strategy"]


def build_active_strategy_labels(
    strategy_schedule: Iterable[Mapping[str, int | str]],
    n_trials: int,
    allowed_strategies: Iterable[str],
) -> np.ndarray:
    """Build per-trial strategy labels from explicit trial segments.

    Parameters
    ----------
    strategy_schedule : iterable of mappings
        Each mapping must contain `start_trial` (int, inclusive), `end_trial`
        (int, exclusive), and `strategy_name` (str).
    n_trials : int
        Total number of trials in the run. Must be a positive integer.
    allowed_strategies : iterable of str
        Strategy names allowed in the schedule.

    Returns
    -------
    labels : np.ndarray of shape (n_trials,), dtype=object
        Active strategy label for each trial index in `[0, n_trials)`.

    Raises
    ------
    ValueError
        If the schedule contains gaps, overlaps, unknown strategies, invalid
        bounds, or does not cover the full trial range exactly.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be positive.")

    allowed_strategy_set = set(allowed_strategies)
    normalized_segments = []
    for segment in strategy_schedule:
        start_trial = int(segment["start_trial"])
        end_trial = int(segment["end_trial"])
        strategy_name = str(segment["strategy_name"])
        if strategy_name not in allowed_strategy_set:
            raise ValueError(f"unknown strategy in schedule: {strategy_name}")
        if start_trial < 0 or end_trial < 0 or end_trial <= start_trial:
            raise ValueError("schedule segments must have valid nonnegative bounds.")
        if end_trial > n_trials:
            raise ValueError("schedule must cover the trial range exactly.")
        normalized_segments.append((start_trial, end_trial, strategy_name))

    if not normalized_segments:
        raise ValueError("schedule must cover the trial range exactly.")

    normalized_segments.sort(key=lambda segment: segment[0])
    labels: list[str] = []
    current_trial = 0
    for start_trial, end_trial, strategy_name in normalized_segments:
        if start_trial > current_trial:
            raise ValueError("schedule contains a gap in trial coverage.")
        if start_trial < current_trial:
            raise ValueError("schedule contains an overlap in trial coverage.")
        labels.extend([strategy_name] * (end_trial - start_trial))
        current_trial = end_trial

    if current_trial != n_trials:
        raise ValueError("schedule must cover the trial range exactly.")

    return np.asarray(labels, dtype=object)


def apply_inactive_updates(
    agents_by_name: Mapping[str, agents.BehaviorAgent],
    active_strategy: str,
    switching_mode: int,
) -> None:
    """Apply per-trial inactive updates to non-active agents.

    Parameters
    ----------
    agents_by_name : mapping of str to BehaviorAgent
        Agent instances for the current switched run. Agents are updated
        in place.
    active_strategy : str
        Name of the strategy active on the current trial. Any agent whose
        mapping key matches this string is skipped.
    switching_mode : int
        Switching mode. Supported values are:
        - `1`: inactive agents do not update
        - `2`: inactive agents apply strategy-specific passive updates
        - `3`: all agents update from the executed action/reward stream

    Returns
    -------
    None
        Agents are modified in place.
    """
    _validate_switching_mode(switching_mode)
    if switching_mode == 1:
        return

    for strategy_name, agent in agents_by_name.items():
        if strategy_name == active_strategy:
            continue
        _apply_single_inactive_update(agent)


def run_switched_agent_session(
    task: BaseMDP,
    agents_by_name: Mapping[str, agents.BehaviorAgent],
    strategy_schedule: Iterable[Mapping[str, int | str]],
    switching_mode: int,
    task_params,
) -> tuple[BaseMDP, dict[str, agents.BehaviorAgent], pd.DataFrame]:
    """Run a behavior session with explicit strategy switching.

    Parameters
    ----------
    task : BaseMDP
        Task instance defining the environment dynamics for a single session.
    agents_by_name : mapping of str to BehaviorAgent
        Agents available for activation during the run. Keys are strategy names.
    strategy_schedule : iterable of mappings
        Explicit trial segments with keys `start_trial`, `end_trial`, and
        `strategy_name`.
    switching_mode : int
        Switching mode to use. Supported values are `1`, `2`, and `3`.
    task_params : TaskParams
        Task parameters for the run. `task_params.n_trials` defines the required
        schedule coverage.

    Returns
    -------
    task : BaseMDP
        The updated task after the session is complete.
    agents_by_name : dict[str, BehaviorAgent]
        The same agent mapping, updated in place and returned as a plain dict.
    performance_df : pd.DataFrame of shape (n_trials, n_columns)
        Trial-wise run dataframe with the standard controller columns plus
        `active_strategy`.
    """
    _validate_switching_mode(switching_mode)
    allowed_strategies = tuple(agents_by_name.keys())
    active_strategy_labels = build_active_strategy_labels(
        strategy_schedule,
        n_trials=task_params.n_trials,
        allowed_strategies=allowed_strategies,
    )
    mutable_agents = dict(agents_by_name)

    if switching_mode == 3:
        return parallel_agents.run_parallel_switched_session(
            task=task,
            agents_by_name=mutable_agents,
            active_strategy_labels=active_strategy_labels,
            task_params=task_params,
            switch_trial_df_columns=SWITCH_TRIAL_DF_COLUMNS,
        )

    performance: dict[str, list] = defaultdict(list)

    while task.cur_trial < task_params.n_trials:
        active_strategy = str(active_strategy_labels[task.cur_trial])
        if active_strategy not in mutable_agents:
            raise ValueError(f"schedule references missing agent: {active_strategy}")
        active_agent = mutable_agents[active_strategy]

        performance["state"].append(task.cur_state)
        performance["state_int"].append(task.state_dict[task.cur_state])
        performance["cur_trial"].append(task.cur_trial)
        performance["cur_block"].append(task.cur_block)
        performance["cur_trial_in_block"].append(task.cur_trial_in_block)
        performance["agent_relative_value"].append(active_agent.value)
        performance["p_active_rew"].append(task.params.active_reward_probability)
        performance["p_inactive_rew"].append(task.params.inactive_reward_probability)
        performance["p_switch"].append(task.params.state_transition_prob)
        performance["agent_prior"].append(controller.get_agent_prior(active_agent))
        performance["agent_hmm_value"].append(controller.get_agent_value_component(active_agent, "hmm_value"))
        performance["agent_doubt_value"].append(controller.get_agent_value_component(active_agent, "doubt_value"))
        performance["active_strategy"].append(active_strategy)

        stimulus = task.get_stimulus()
        action, action_dist = active_agent.choose_action(stimulus)
        reward, correct = task.step(action)

        performance["model_stimulus"].append(stimulus)
        performance["agent_action_dist"].append(action_dist[1])
        active_agent.update_params(action, reward)
        apply_inactive_updates(mutable_agents, active_strategy, switching_mode)

        performance["action"].append(action)
        performance["correct"].append(correct)
        performance["reward"].append(reward)

    performance_df = pd.DataFrame(performance).reindex(columns=SWITCH_TRIAL_DF_COLUMNS)
    return task, mutable_agents, performance_df


def _validate_switching_mode(switching_mode: int) -> None:
    """Validate the integer switching mode identifier.

    Parameters
    ----------
    switching_mode : int
        Requested switching mode.

    Returns
    -------
    None
        Raises on invalid mode values.
    """
    if switching_mode not in (1, 2, 3):
        raise ValueError("switching_mode must be 1, 2, or 3.")


def _apply_single_inactive_update(agent: agents.BehaviorAgent) -> None:
    """Apply a strategy-specific passive update to a single inactive agent.

    Parameters
    ----------
    agent : BehaviorAgent
        Agent to update in place.

    Returns
    -------
    None
        Agent state is modified in place.
    """
    if isinstance(agent, agents.HMMRewardDecayRelativeDoubt):
        _apply_hmm_decay_doubt_inactive_update(agent)
        return
    if isinstance(agent, agents.HMMRewardDecay):
        _apply_hmm_reward_decay_inactive_update(agent)
        return
    if isinstance(agent, agents.HMM):
        _apply_hmm_inactive_update(agent)
        return
    if isinstance(agent, agents.ForgettingQlearning):
        _apply_fql_inactive_update(agent)
        return

    raise TypeError(f"Unsupported inactive-update agent type: {type(agent).__name__}")


def _refresh_action_policy(agent: agents.BehaviorAgent) -> None:
    """Refresh an agent's action distribution using its last-known action state.

    Parameters
    ----------
    agent : BehaviorAgent
        Agent whose policy distribution should be refreshed in place.

    Returns
    -------
    None
        The agent action distribution is updated if the required state is
        available.
    """
    if not hasattr(agent, "update_action_dist"):
        return

    if hasattr(agent, "last_action"):
        agent.update_action_dist(int(agent.last_action))
        return

    if hasattr(agent, "last_action_ix"):
        last_action = int(getattr(agent, "last_action_ix") > 0)
        agent.update_action_dist(last_action)


def _apply_fql_inactive_update(agent: agents.ForgettingQlearning) -> None:
    """Apply a forgetting-only inactive update to a Forgetting Q-learning agent.

    Parameters
    ----------
    agent : ForgettingQlearning
        Agent updated in place.

    Returns
    -------
    None
        `Q`, `value`, and policy state are updated in place.
    """
    agent.Q *= agent.decay
    agent.value = agent.Q[agents.LEFT_IX] - agent.Q[agents.RIGHT_IX]
    _refresh_action_policy(agent)


def _apply_hmm_inactive_update(agent: agents.HMM) -> None:
    """Apply a transition-only inactive update to an HMM-like agent.

    Parameters
    ----------
    agent : HMM
        Agent updated in place.

    Returns
    -------
    None
        `prior`, `value`, and policy state are updated in place.
    """
    agent.prior = np.dot(agent.transition_matrix.T, agent.prior)
    agent.prior /= np.sum(agent.prior) + agents.eps
    agent.value = agent.compute_relative_value()
    _refresh_action_policy(agent)


def _apply_hmm_reward_decay_inactive_update(agent: agents.HMMRewardDecay) -> None:
    """Apply passive omission-style decay to an HMM reward-decay agent.

    Parameters
    ----------
    agent : HMMRewardDecay
        Agent updated in place.

    Returns
    -------
    None
        `prior`, `value`, and policy state are updated in place.
    """
    posterior = agent._omission_posterior()
    agent.prior = np.dot(agent.transition_matrix.T, posterior)
    agent.prior /= np.sum(agent.prior) + agents.eps
    agent.value = agent.compute_relative_value()
    _refresh_action_policy(agent)


def _apply_hmm_decay_doubt_inactive_update(agent: agents.HMMRewardDecayRelativeDoubt) -> None:
    """Apply passive belief and doubt decay to a composite HMM-doubt agent.

    Parameters
    ----------
    agent : HMMRewardDecayRelativeDoubt
        Agent updated in place.

    Returns
    -------
    None
        Belief state, doubt state, combined value, and policy state are updated
        in place.
    """
    posterior = agent._omission_posterior()
    agent.prior = np.dot(agent.transition_matrix.T, posterior)
    agent.prior /= np.sum(agent.prior) + agents.eps
    agent.hmm_value = agent.compute_relative_value()

    agent.apply_passive_doubt_decay()
    agent.value = agent.hmm_value - agent.doubt_value
    _refresh_action_policy(agent)
