from pathlib import Path
import pickle as pkl
from typing import Dict, Tuple, Any, Optional
import logging
from collections import defaultdict

import numpy as np
import pandas as pd

import agents
import context_task
import MDP
import controller


def run_figure_1_data(params, agent_name, max_rewards=6):
    reward_settings = np.arange(max_rewards)+1
    for i in reward_settings:
        rewards_before_switch = i
        session_name = 'pReward_{}_pSwitch_{}_rewardsBeforeSwitch_{}'.format(params.active_reward_probability,
                                                                             params.state_transition_prob,
                                                                             rewards_before_switch)

        # need to reset task and agent between runs
        params.fixed_block_lengths = [rewards_before_switch, 50]

        performance_df = collect_agent_performance(params, agent_name, n_agents=2)
        save_collected_runs(performance_df, agent_name, session_name)


def get_demo_action(task, agent):
    # select the correct action for an agent doing a demo task
    if task.cur_state == 'right':
        action = 0
    else:
        action = 1
    agent.last_action = action
    return action, agent


def select_agent(agent_name: str, params: MDP.TaskParams) -> \
        MDP.MarkovDecisionProcess:
    if agent_name == 'HMM':
        agent = agents.HMM(params)
    elif agent_name == 'HMM_RFLR':
        agent = agents.HMM_RFLR(params)
    elif agent_name == 'Qlearning':
        agent = agents.Qlearning(params)
    elif agent_name == 'F-Qlearning':
        agent = agents.ForgettingQlearning(params)
    elif agent_name == 'Logistic':
        agent = agents.Logistic(params)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def run_demo_experiment(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:
    performance = defaultdict(list)
    while task.cur_block < task.params.n_blocks:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance = run_demo_task_cycle(task, agent, performance)

    return task, agent, performance


def run_demo_task_cycle(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: Dict[str, list]) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:

    performance['states'].append(task.cur_state)
    performance['cur_trial'].append(task.cur_trial)
    performance['cur_block'].append(task.cur_block)
    performance['action_dist'].append(agent.action_dist)
    if agent.model_type in ['HMM', 'HMM_RFLR']:
        performance['prior'].append(agent.prior)
        performance['relative_value'].append(agent.value)
    elif agent.model_type in ['Qlearning', 'Forgetting Q-learning']:
        performance['q_values'].append(agent.Q)
        performance['relative_value'].append(agent.value)

    action, action_dist = agent.choose_action(task.get_stimulus())
    performance['stimulus'].append(task.get_stimulus())
    performance['action_dist'].append(action_dist[0])
    if task.cur_block == 0:  # choose correct action for demo runs
        action, agent = get_demo_action(task, agent)
    # choose demo action
    reward, correct = task.step(action)
    if task.cur_block == 0:  # guarantee reward for demo runs
        reward = task.params.mean_correct_reward

    agent.update_params(action, reward)

    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    return task, agent, performance


def collect_agent_performance(params: MDP.TaskParams, agent_name: str, n_agents: int, demo=True) -> \
        pd.DataFrame:
    agent_performance_list = []
    for i in range(n_agents):
        # must reset the task/agent here!!!
        task = context_task.BaseMDP(params)
        agent = select_agent(agent_name, params)
        if demo:
            task, agent, performance = run_demo_experiment(task, agent)
        else:
            task, agent, performance = controller.run_experiment(task, agent)

        performance['session_ID'] = np.ones(task.cur_trial) * i
        agent_performance_list.append(pd.DataFrame(performance))

    agent_performance_df = pd.concat(agent_performance_list)
    return agent_performance_df


def save_collected_runs(agent_performance_df: pd.DataFrame, agent_type: str, session_name: str) -> None:
    """Saves collected runs in a single dataframe. Paramters are not saved, so make sure to save/note those separately."""
    p = Path('../saved_run_collections') / agent_type  # make this dir
    if not p.exists():
        p.mkdir()

    p = p / (session_name + '.pkl')
    agent_performance_df.to_pickle(p)
    print("[***] Experiment saved in {} as: {}".format(agent_type, p.name))


def load_collected_runs(agent_type: str, p_reward: float, p_switch: float,
                        rewards_before_switch: Optional[int] = None,
                        n_agents: Optional[int] = None) -> pd.DataFrame:

    session_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(agent_type, p_reward, p_switch)
    if rewards_before_switch:
        session_name += '_rewardsBeforeSwitch_{}'.format(rewards_before_switch)
    if n_agents:
        session_name += '_nAgents_{}'.format(n_agents)

    p = Path('../saved_run_collections') / agent_type / (session_name + '.pkl')
    df = pd.read_pickle(p)
    return df


def run_performance_collection():
    """
    For the stats plots, need to collect
    1. A forgetting Q-learning model with decay .6, stickiness 0, temp .3
    2. A HMM with Psw .1, stickiness 0, temp 1
    """

    log_path = Path('collection_task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    params = MDP.TaskParams(blocks=1000)
    # params.fixed_block_sequence = ['left', 'right']

    params.mean_correct_reward = 1
    params.mean_incorrect_reward = 0
    params.active_reward_probability = .9
    params.inactive_reward_probability = 0
    params.reward_std_dev = 0
    params.p_cue = 0

    params.markov_block_transitions = False
    params.default_block_length = 15
    if params.markov_block_transitions:
        params.state_transition_prob = .3
    else:
        params.state_transition_prob = 1 / params.default_block_length
        # params.state_transition_prob = .1
    params.block_length_variation = 5
    params.max_consecutive_blocks = 1

    params.FQL_decay = .8  # for forgetting Q-learning agent
    # params.HMM_transition_prob = params.state_transition_prob  # HMM agent's belief of task dynamics
    params.HMM_transition_prob = .1  # HMM agent's belief of task dynamics
    # params.HMM_transition_prob = .001  # HMM agent's belief of task dynamics

    params.logistic_alpha = 1  # RFLR default 1 - action stickiness
    params.logistic_beta = 2  # RFLR default 2 - action/reward history update size
    params.logistic_tau = 1.5  # RFLR default 1.5 - memory for prior timestep settings, higher tau increases memory

    params.action_temperature = 1  # .2 or .3 seems to work for FQL  # temp=1 is default; temp->0 makes greedier; temp->inf increases randomness
    # params.action_stickiness = 0  # default for RFLR is 1; default for others is 0

    # agent_name = 'F-Qlearning'
    agent_name = 'HMM'
    # agent_name = 'HMM_RFLR'
    # agent_name = 'Logistic'

    # pick a file path based on agent type (HMM, Qlearning, etc.)
    # pick a filename based on task-specific params (pReward, pSwitch, rewards-before-state-switch, etc.)
    # rewards_before_switch = 1

    n_agents = 10
    session_name = '{}_pReward_{}_pSwitch_{:.2f}_nAgents_{}'.format(agent_name,
                                                                params.active_reward_probability,
                                                                params.state_transition_prob,
                                                                n_agents)
    extras_str = ''
    if extras_str:
        session_name += '_' + extras_str

    performance_df = collect_agent_performance(params, agent_name, n_agents=n_agents, demo=False)
    save_collected_runs(performance_df, agent_name, session_name)

    # make sure they're getting the rewards!!!
    # run_figure_1_data(params, agent_name)


if __name__ == '__main__':
    run_performance_collection()

