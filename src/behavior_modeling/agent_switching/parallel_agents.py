"""Helpers for switching mode 3 parallel-model execution.

This module implements the near-term approximation for switching mode 3:
all models observe the same executed action/reward stream, while only the
active model contributes recorded outputs to the run dataframe.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

import pandas as pd

from src.behavior_modeling import controller
from src.behavior_modeling.agents import agents
from src.behavior_modeling.task.context_task import BaseMDP


def update_all_agents_from_executed_trial(
    agents_by_name: Mapping[str, agents.BehaviorAgent],
    action: int,
    reward: float,
) -> None:
    """Update all agents from one executed trial outcome.

    Parameters
    ----------
    agents_by_name : mapping of str to BehaviorAgent
        Agent instances participating in the switched run. All agents are
        updated in place from the same executed action and reward.
    action : int
        Executed action index for the trial. `0` denotes right, `1` denotes
        left.
    reward : float
        Observed reward on the executed trial.

    Returns
    -------
    None
        Agents are updated in place.
    """
    for agent in agents_by_name.values():
        agent.update_params(action, reward)


def run_parallel_switched_session(
    task: BaseMDP,
    agents_by_name: Mapping[str, agents.BehaviorAgent],
    active_strategy_labels,
    task_params,
    switch_trial_df_columns: list[str],
) -> tuple[BaseMDP, dict[str, agents.BehaviorAgent], pd.DataFrame]:
    """Run a switched behavior session in parallel-model mode.

    Parameters
    ----------
    task : BaseMDP
        Task instance defining the environment dynamics for a single session.
    agents_by_name : mapping of str to BehaviorAgent
        Agents available for parallel updates during the run. Keys are strategy
        names referenced by `active_strategy_labels`.
    active_strategy_labels : np.ndarray of shape (n_trials,), dtype=object
        Per-trial active strategy names. Labels are assumed to have already been
        validated against the run schedule and `task_params.n_trials`.
    task_params : TaskParams
        Task parameter bundle defining `n_trials` and reward/task dynamics.
    switch_trial_df_columns : list of str
        Ordered dataframe columns for the returned performance dataframe.

    Returns
    -------
    task : BaseMDP
        The updated task after the session is complete.
    agents_by_name : dict[str, BehaviorAgent]
        The same agent mapping, updated in place and returned as a plain dict.
    performance_df : pd.DataFrame of shape (n_trials, n_columns)
        Trial-wise run dataframe recording only the active model's pre-update
        outputs, along with the executed action/reward and `active_strategy`.
    """
    mutable_agents = dict(agents_by_name)
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

        update_all_agents_from_executed_trial(mutable_agents, action, reward)

        performance["action"].append(action)
        performance["correct"].append(correct)
        performance["reward"].append(reward)

    performance_df = pd.DataFrame(performance).reindex(columns=switch_trial_df_columns)
    return task, mutable_agents, performance_df
