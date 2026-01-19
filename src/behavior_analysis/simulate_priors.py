import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
import copy
from pathlib import Path
import re
import pickle as pkl
import model_agents
import session_analysis
from typing import Protocol, Optional, Union
import time
import json


SEED = 12345
rng = np.random.default_rng()
eps = np.finfo(float).eps


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


@dataclass
class QLearningParams:
    n_states: int = 2
    n_actions: int = 2
    QL_learning_rate: float = .9
    stickiness: float = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    action_temperature: float = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value


@dataclass
class ForgettingQLParams:
    n_states: int = 2
    n_actions: int = 2
    FQL_decay: float = .9
    stickiness: float = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    action_temperature: float = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value


@dataclass
class LogisticParams:
    n_states: int = 2
    n_actions: int = 2
    stickiness: float = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    weight_reward_history: float = 2  # default 2. Weight on reward history. "beta" in Beron PNAS 2022
    weight_decay: float = 1.5  # default 1.5. Reward history decay parameter (higher is faster decay, lower has longer "memory"). "tau" in Beron PNAS 2022
    action_temperature: float = 1  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value


@dataclass
class HmmParams:
    n_states: int = 2
    n_actions: int = 2
    p_cue: float = .25
    state_transition_prob: float = .2
    active_reward_probability: float = .8
    inactive_reward_probability: float = 0
    correct_reward_size: float = 1
    incorrect_reward_size: float = 0
    stickiness: float = 1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022


def collect_agent_performance(trial_df: pd.DataFrame, agent: model_agents.BehaviorAgent, params: TaskParams) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    action_dist = []
    rel_value = []
    actions = []
    for action, reward, give_reward in zip(trial_df['action'], trial_df['reward'], trial_df['give_reward']):
        # if action == 'None':  # skip any give_reward trials! This will also make output shorter, a 'None' needs to be added to this index...
        if give_reward == 1:  # skip any give_reward trials! This will also make output shorter, a 'None' needs to be added to this index...
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
    params.correct_reward_size = 1
    params.incorrect_reward_size = 0

    # Q-learning
    params = QLearningParams()
    params.QL_learning_rate = .35  # for standard Q-learning agent
    params.stickiness = 1.8  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    params.action_temperature = .2# + eps  # randomness parameter for choices - higher temp converges to random choice, lower is greedy to higher value

    agent = model_agents.Qlearning(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['Qlearning_prob_left'] = action_dist['p_left']
    augmented_trial_df['Qlearning_rel_value'] = rel_value
    augmented_trial_df['Qlearning_greedy_action'] = actions
    score = np.mean(actions == augmented_trial_df['action'])
    print(f"Qlearning score: {score}")

    # Save to JSON
    json_fname = processed_data_path / 'QLearning_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)

    # Forgetting Q-learning
    params = ForgettingQLParams()
    params.FQL_decay = .7  # for forgetting Q-learning agent
    params.stickiness = .1  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    params.action_temperature = 1.8 #+ eps  # randomness parameter for choices - higher temp converges to random choice, lower is greedy to higher value
    agent = model_agents.ForgettingQlearning(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['FQlearning_prob_left'] = action_dist['p_left']
    augmented_trial_df['FQlearning_rel_value'] = rel_value
    augmented_trial_df['FQlearning_greedy_action'] = actions
    score = np.mean(actions == augmented_trial_df['action'])
    print(f"Forgetting Qlearning score: {score}")

    # Save to JSON
    json_fname = processed_data_path / 'FQLearning_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)

    # RFLR parameters
    params = LogisticParams()
    params.stickiness = .3  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    params.weight_reward_history = .5  # default 2. Weight on reward history. "beta" in Beron PNAS 2022
    params.weight_decay = 2.6  # default 1.5. Reward history decay parameter (higher is faster decay, lower has longer "memory"). "tau" in Beron PNAS 2022
    params.action_temperature = 0 + eps  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value

    agent = model_agents.Logistic(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['RFLR_prob_left'] = action_dist['p_left']
    augmented_trial_df['RFLR_rel_value'] = rel_value
    augmented_trial_df['RFLR_greedy_action'] = actions
    score = np.mean(actions == augmented_trial_df['action'])
    print(f"RFLR score: {score}")

    # Save to JSON
    json_fname = processed_data_path / 'RFLR_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)

    # HMM
    params = HmmParams()
    params.state_transition_prob = .2
    params.active_reward_probability = .8
    params.inactive_reward_probability = 0
    params.stickiness = .3  # default 1. tendency to repeat last action. "alpha" in Beron PNAS 2022
    params.action_temperature = 1.5  # randomness parameter for choices - higher temp converges to random choice, lower is probabilisitic to higher value

    agent = model_agents.HMM(params)
    action_dist, actions, rel_value = collect_agent_performance(augmented_trial_df, agent, params)
    augmented_trial_df['HMM_prob_left'] = action_dist['p_left']
    augmented_trial_df['HMM_rel_value'] = rel_value
    augmented_trial_df['HMM_greedy_action'] = actions
    score = np.mean(actions == augmented_trial_df['action'])
    print(f"HMM score: {score}")

    # make sure all the "None"s carry over to the augemented trial dataframe!!
    augmented_trial_df.to_csv(augmented_trial_df_path, index=False)

    # Save to JSON
    json_fname = processed_data_path / 'HMM_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)

    # Load from JSON
    # with open("config.json", "r") as f:
    #     data = json.load(f)
    #     loaded_config = Config(**data)


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

    # QL parameters
    params = QLearningParams()
    learning_rate = np.linspace(0, 1, 21)  # for forgetting Q-learning agent
    stickiness_grid = np.linspace(0, 1, 11)
    temperature_grid = np.linspace(0, 2, 11)
    results = []

    print("Starting Q-Learning parameter search")
    st = time.time()
    for s in stickiness_grid:
        print('starting stickiness value', s)
        for t in temperature_grid:
            for lr in learning_rate:
                params.stickiness = s
                params.QL_learning_rate = lr
                params.action_temperature = t
                if params.action_temperature == 0:
                    params.action_temperature += eps

                agent = model_agents.Qlearning(params)
                action_dist, greedy_actions, rel_value = collect_agent_performance(trial_df, agent, params)
                score = np.mean(greedy_actions == trial_df['action'])
                results.append(dict(stickiness=s, temperature=t, learning_rate=lr, score=score))
    print('runtime', time.time() - st)

    results = pd.DataFrame(results)
    best_params = results.iloc[results['score'].argmax()]
    print(best_params)

    ### FQL parameters ######################
    # params = ForgettingQLParams()
    # decay_grid = np.linspace(0, 1, 21)  # for forgetting Q-learning agent
    # stickiness_grid = np.linspace(0, 1, 11)
    # temperature_grid = np.linspace(0, 2, 11)
    # results = []
    #
    # print("Starting Forgetting Q-Learning parameter search")
    # st = time.time()
    # for s in stickiness_grid:
    #     print('starting stickiness value', s)
    #     for t in temperature_grid:
    #         for d in decay_grid:
    #             params.stickiness = s
    #             params.FQL_decay = d
    #             params.action_temperature = t
    #
    #             agent = model_agents.ForgettingQlearning(params)
    #             action_dist, greedy_actions, rel_value = collect_agent_performance(trial_df, agent, params)
    #             score = np.mean(greedy_actions == trial_df['action'])
    #             results.append(dict(stickiness=s, temperature=t, weight_decay=d, score=score))
    # print('runtime', time.time() - st)
    #
    # results = pd.DataFrame(results)
    # best_params = results.iloc[results['score'].argmax()]
    # print(best_params)

    # params = TaskParams()
    ### RFLR parameters  ############################################33
    # stickiness_grid = np.linspace(0, 1, 11)
    # reward_hist_grid = np.linspace(0, 3, 21)
    # temperature_grid = np.linspace(0, 2, 11)
    # decay_grid = np.linspace(0, 2, 21)

    # stickiness_grid = np.linspace(0, .5, 6)
    # temperature_grid = np.linspace(0, .5, 6)
    # reward_hist_grid = np.linspace(0, 2, 21)
    # decay_grid = np.linspace(0, 4, 21)
    # results = []
    #
    # print("Starting Reduced Form Logistic Regression parameter search")
    # st = time.time()
    # for s in stickiness_grid:
    #     print('starting stickiness value', s)
    #     for t in temperature_grid:
    #         for d in decay_grid:
    #             for r in reward_hist_grid:
    #                 params.stickiness = s
    #                 params.weight_reward_history = r
    #                 params.weight_decay = d
    #                 params.action_temperature = t
    #
    #                 agent = model_agents.Logistic(params)
    #                 action_dist, greedy_actions, rel_value = collect_agent_performance(trial_df, agent, params)
    #                 score = np.mean(greedy_actions == trial_df['action'])
    #                 results.append(dict(stickiness=s, temperature=t, weight_decay=d, weight_reward=r, score=score))
    # print('runtime', time.time() - st)
    #
    # results = pd.DataFrame(results)
    # best_params = results.iloc[results['score'].argmax()]
    # print(best_params)


    #### HMM parameters ############################################33
    # stickiness_grid = np.linspace(0, 2, 21)
    # temperature_grid = np.linspace(0, 2, 21)
    # # Probability structure - run with this (full hyperparameter check) and without it (just stickiness/temp)
    # Preward_grid = np.linspace(0, 1, 11)
    # Pswitch_grid = np.linspace(0, .5, 11)
    # # Preward_grid = [.8]
    # # Pswitch_grid = [.2]
    # results = []
    #
    # optimal_model = dict(label='None',
    #                      stickiness='None', temperature='None',
    #                      p_reward='None', p_switch='None',
    #                      weight_decay='None', weight_reward='None',
    #                      score='None')
    #
    # print("Starting Hidden Markov Model parameter search")
    # st = time.time()
    # for stick in stickiness_grid:
    #     print('starting stickiness value', np.round(stick, decimals=1))
    #     for t in temperature_grid:
    #         for r in Preward_grid:
    #             for sw in Pswitch_grid:
    #                 params.state_transition_prob = sw
    #                 params.active_reward_probability = r
    #                 params.stickiness = stick
    #                 params.action_temperature = t
    #
    #                 agent = model_agents.HMM(params)
    #                 action_dist, greedy_actions, rel_value = collect_agent_performance(trial_df, agent, params)
    #                 score = np.mean(greedy_actions == trial_df['action'])
    #                 results.append(dict(stickiness=stick, temperature=t, p_reward=r, p_switch=sw, score=score))
    # print('runtime HMM', time.time() - st)
    #
    # results = pd.DataFrame(results)
    # best_params = results.iloc[results['score'].argmax()]
    # print(best_params)


if __name__ == '__main__':
    run_performance_collection()
    # grid_search_priors()

