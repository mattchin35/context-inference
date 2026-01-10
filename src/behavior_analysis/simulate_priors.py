import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import copy
from pathlib import Path
import re
import pickle as pkl
import model_agents
import session_analysis
from typing import Protocol, Optional, Union
import time

SEED = 12345
rng = np.random.default_rng()


"""
Get the relative values and action distibutions for a set of model agents over a mouse's behavior session.
"""


@dataclass
class TaskParams:
    n_states: int = 2
    n_actions: int = 2

    # Probability parameters
    # For HHM inference model, you can play with using model parameters that are different from the true task parameters
    p_cue: float = .25
    state_transition_prob: float = .2
    active_reward_probability: float = .9
    inactive_reward_probability: float = 0
    correct_reward_size: float = 1
    incorrect_reward_size: float = 0

    # Reinforcement learning parameters
    FQL_decay: float = .9  # for forgetting Q-learning agent
    QL_learning_rate: float = .1  # for standard Q-learning agent
    greedy_epsilon: float = .1

    # Forgetting Q-learning parameters...
    # logistic_alpha: float = 1  # default 1. Equivalent to action stickiness
    weight_stickiness: float = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    weight_reward_history: float = 2  # default 2. Weight on reward history. "beta" in Beron PNAS 2022
    weight_decay: float = 1.5  # default 1.5. Reward history decay parameter (higher is faster decay, lower has longer "memory"). "tau" in Beron PNAS 2022

    action_temperature: float = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value
    greedy_action_selection: bool = False  # don't need this for re-runs - I'm not sampling


class RepeatMDP:
    """
    Repeat the exact trial conditions to obtain prior/choice distributions.
    """
    def __init__(self, params: TaskParams, task_df: pd.DataFrame):
        super().__init__(params)

        self.params = params
        self.task_df = task_df
        self.n_trials = task_df.shape[0]

        self.states: list[str] = ['right', 'left']
        self.state_dict = {s: i for i, s in enumerate(self.states)}
        self.state_stimulus_dict = copy.deepcopy(self.state_dict)
        self.state_stimulus_dict['uncued'] = -1
        self.p_cue = params.p_cue

        ## initialize task
        self.cur_state = self.task_df.loc[0, 'state']
        self.cur_block = 0
        self.cur_trial = 0
        self.cur_trial_in_block = 0
        self.correct_in_block = 0

    def get_stimulus(self) -> int:
        return self.task_df['stimulus'].loc[self.cur_trial]

    def step(self, action: int, correct: bool, reward: int) -> tuple[float, float]:
        self.cur_trial_in_block += 1
        self.cur_trial += 1
        self.correct_in_block += int(correct)
        if self.cur_trial < self.n_trials:
            self.cur_state = self.task_df['state'].loc[self.cur_trial]
            self.cur_block = self.task_df['cur_block'].loc[self.cur_trial]
            self.cur_trial_in_block = self.task_df['cur_trial_in_block'].loc[self.cur_trial]
        return reward, correct


def collect_agent_performance(trial_df: pd.DataFrame, agent: model_agents.BehaviorAgent, params: TaskParams) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    action_dist = []
    rel_value = []
    actions = []
    for action, reward in zip(trial_df['action'], trial_df['reward']):
        if action == 'None':  #skip any give_reward trials! This will also make output shorter, a 'None' needs to be added to this index...
            action_dist.append(pd.DataFrame([['None', 'None']], columns=['p_right', 'p_left']))
            rel_value.append('None')
            actions.append('None')
            continue

        action_dist.append(agent.get_action_dist())
        rel_value.append(agent.value)
        actions.append(np.argmax(agent.get_action_dist()))
        try:
            agent.update_params(int(action), int(reward))
        except ValueError as e:
            # nan rewards may occur for given rewards (non-operant, experimenter-given rewards) - just skip these
            pass

    action_dist = pd.concat(action_dist, ignore_index=True)
    return action_dist, np.array(actions), np.array(rel_value)


def run_performance_collection():
    """
    For the stats plots, need to collect a forgetting Q-learning model with decay .6, stickiness 0, temp .3;
    2. A HMM with Psw .1, stickiness 0, temp 1
    """
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
        sess_id_abbreviated = mouse + '_' + date
    else:
        print("Double-check the session name!")
        return

    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)
    params = TaskParams()

    # Task general params
    params.p_cue = 0
    params.state_transition_prob = .2
    params.active_reward_probability = .8
    params.inactive_reward_probability = 0
    params.correct_reward_size = 1
    params.incorrect_reward_size = 0

    # Reinforcement learning parameters
    params.greedy_action_selection = True
    params.greedy_epsilon = 1
    params.QL_learning_rate = .1  # for standard Q-learning agent
    params.FQL_decay = .9  # for forgetting Q-learning agent

    # Forgetting Q-learning/RFLR parameters...may need to fit this for best fit
    params.stickiness = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    params.weight_reward_history = 2  # default 2. Weight on reward history. "beta" in Beron PNAS 2022
    params.weight_decay = 1.5  # default 1.5. Reward history decay parameter (higher is faster decay, lower has longer "memory"). "tau" in Beron PNAS 2022
    params.action_temperature = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value

    agent = model_agents.Qlearning(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['Qlearning_prob_left'] = action_dist['p_left']
    augmented_trial_df['Qlearning_rel_value'] = rel_value
    augmented_trial_df['Qlearning_greedy_action'] = actions

    agent = model_agents.ForgettingQlearning(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['FQlearning_prob_left'] = action_dist['p_left']
    augmented_trial_df['FQlearning_rel_value'] = rel_value
    augmented_trial_df['FQlearning_greedy_action'] = actions

    agent = model_agents.Logistic(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['RFLR_prob_left'] = action_dist['p_left']
    augmented_trial_df['RFLR_rel_value'] = rel_value
    augmented_trial_df['RFLR_greedy_action'] = actions

    agent = model_agents.HMM(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['HMM_prob_left'] = action_dist['p_left']
    augmented_trial_df['HMM_rel_value'] = rel_value
    augmented_trial_df['HMM_greedy_action'] = actions

    # make sure all the "None"s carry over to the augemented trial dataframe!!
    augmented_trial_df.to_csv(augmented_trial_df_path, index=False)


def grid_search_priors():
    """
    Want to get optimal hyperparameters for RFLR and HMM stickiness. Use true HMM probability and optimize stickiness.
    Only optimize RFLR, which is mathematically equivalent to FQLearning.
    """
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
        sess_id_abbreviated = mouse + '_' + date
    else:
        print("Double-check the session name!")
        return

    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    trial_df = pd.read_csv(trial_df_path, sep=',', na_filter=False)
    params = TaskParams()

    # RFLR parameters
    params.stickiness = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    params.weight_reward_history = 2  # default 2. Weight on reward history. "beta" in Beron PNAS 2022
    params.weight_decay = 1.5  # default 1.5. Reward history decay parameter (higher is faster decay, lower has longer "memory"). "tau" in Beron PNAS 2022
    params.action_temperature = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value

    stickiness_grid = np.linspace(0, 1, 11)
    reward_hist_grid = np.linspace(0, 3, 21)
    temperature_grid = np.linspace(0, 2, 11)
    decay_grid = np.linspace(0, 2, 21)
    results = []

    st = time.time()
    agent = model_agents.Logistic(params)
    for s in stickiness_grid:
        for t in temperature_grid:
            for d in decay_grid:
                for r in reward_hist_grid:
                    action_dist, greedy_actions, rel_value = collect_agent_performance(trial_df, agent, params)
                    score = np.mean(greedy_actions == trial_df['action'])
                    results.append(dict(stickiness=s, temperature=t, weight_decay=d, weight_reward=r, score=score))
        print('starting stickiness value', s)
    print('runtime', time.time() - st)

    results = pd.DataFrame(results)
    best_params = results.iloc[results['score'].argmax()]
    print(best_params)

    # params.state_transition_prob = .2
    # params.active_reward_probability = .8
    # params.stickiness = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022

    # stickiness_grid = np.linspace(0, 1, 11)
    # temperature_grid = np.linspace(0, 2, 11)
    # # Probability structure - run with this (full hyperparameter check) and without it (just stickiness/temp)
    # Preward_grid = np.linspace(0, 1, 11)
    # Pswitch_grid = np.linspace(0, .5, 11)
    # results = []
    #
    # st = time.time()
    # agent = model_agents.HMM(params)
    # for s in stickiness_grid:
    #     for t in temperature_grid:
    #         for r in Preward_grid:
    #             for sw in Pswitch_grid:
    #                 action_dist, greedy_actions, rel_value = collect_agent_performance(trial_df, agent, params)
    #                 score = np.mean(greedy_actions == trial_df['action'])
    #                 results.append(dict(stickiness=s, temperature=t, decay=d, reward=r, score=score))
    #     print('starting stickiness value', s)
    # print('runtime HMM', time.time() - st)


if __name__ == '__main__':
    run_performance_collection()
    # grid_search_priors()

