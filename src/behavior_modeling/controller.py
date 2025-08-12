from pathlib import Path
import pickle as pkl
from typing import Dict, Tuple, Any, Optional
import logging
from collections import defaultdict
import time
import datetime as dt

import numpy as np
import pandas as pd
import torch

import agents.agents as agents
from task.context_task import BaseMDP
from parameters import task_config
# from config import AgentConfig

SEED = 12345  # 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def run_task_cycle(task: BaseMDP, agent: agents.BehaviorAgent, performance: Dict[str, list]) -> \
        Tuple[BaseMDP, agents.BehaviorAgent, Dict[str, list]]:

    performance['state'].append(task.cur_state)
    performance['cur_trial'].append(task.cur_trial)
    performance['cur_block'].append(task.cur_block)
    performance['cur_trial_in_block'].append(task.cur_trial_in_block)
    performance['relative_value'].append(agent.value)
    performance['p_active_reward'].append(task.params.active_reward_probability)
    performance['p_inactive_reward'].append(task.params.inactive_reward_probability)
    if agent.model_type == 'HMM':
        performance['prior'].append(agent.prior.copy())
    elif agent.model_type in ['Qlearning', 'Forgetting Q-learning']:
        performance['q_values'].append(agent.Q.copy())

    stimulus = task.get_stimulus()
    action, action_dist = agent.choose_action(stimulus)
    reward, correct = task.step(action)

    performance['stimulus'].append(stimulus)
    performance['action_dist'].append(action_dist[0])
    agent.update_params(action, reward)

    # this will get refactored out to a controller class/fxn
    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    return task, agent, performance


# def run_repeat_task_cycle(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: Dict[str, list],
#                           task_df: pd.DataFrame, repeat_decisions: int = False) -> Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:
#     """Repeat the trials from a prior session.
#     TODO - debug this with updates from April 2024"""
#     performance['state'].append(task.cur_state)
#     performance['cur_trial'].append(task.cur_trial)
#     performance['cur_block'].append(task.cur_block)
#     performance['cur_trial_in_block'].append(task.cur_trial_in_block)
#     performance['relative_value'].append(agent.value)
#     performance['p_active_reward'].append(task.params.active_reward_probability)
#     performance['p_inactive_reward'].append(task.params.inactive_reward_probability)
#     if agent.model_type == 'HMM':
#         performance['prior'].append(agent.prior.copy())
#     elif agent.model_type in ['Qlearning', 'Forgetting Q-learning']:
#         performance['q_values'].append(agent.Q.copy())
#
#     # for a repeat of a prior session, the action and its reward can be input rather than sampled
#     stimulus = task_df['stimulus'].values[task.cur_trial]
#     performance['stimulus'].append(stimulus)
#     if repeat_decisions:
#         action = task_df['action'].values[task.cur_trial]
#         reward = task_df['reward'].values[task.cur_trial]
#         _, action_dist = agent.choose_action(stimulus)
#         _, correct = task.step(action)
#     else:
#         action, action_dist = agent.choose_action(stimulus)
#         reward, correct = task.step(action)
#
#     performance['action_dist'].append(action_dist[0])
#     agent.update_params(action, reward)
#
#     # this will get refactored out to a controller class/fxn
#     performance['action'].append(action)
#     performance['correct'].append(correct)
#     performance['reward'].append(reward)
#     return task, agent, performance


def run_experiment(task: BaseMDP, agent: agents.BehaviorAgent, task_params: task_config.TaskParams) -> \
        Tuple[BaseMDP, agents.BehaviorAgent, Dict[str, list]]:

    performance = defaultdict(list)
    while task.cur_trial < task_params.n_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance = run_task_cycle(task, agent, performance)

    return task, agent, performance


# def run_repeat_experiment(agent_type: str, params: MDP.TaskParams, task_df: pd.DataFrame, repeat_decisions: bool = False) -> \
#             Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:
#
#     task = context_task.RepeatMDP(params, task_df)
#     agent = select_agent(agent_type, params)
#     performance = defaultdict(list)
#     while task.cur_trial < task_df.shape[0]:
#         logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
#         task, agent, performance = run_repeat_task_cycle(task, agent, performance, task_df, repeat_decisions)
#
#     return task, agent, performance


def select_agent(agent_name: str, agent_params: task_config.AgentParams, task_params: task_config.TaskParams) -> agents.BehaviorAgent:
    if agent_name == 'HMM':
        agent = agents.HMM(agent_params, task_params)
    elif agent_name == 'HMM_RFLR':
        agent = agents.HMM_RFLR(agent_params, task_params)
    elif agent_name == 'HMM_recursive':
        agent = agents.HMM_recursive(agent_params, task_params)
    elif agent_name == 'Qlearning':
        agent = agents.Qlearning(agent_params)
    elif agent_name == 'F-Qlearning':
        agent = agents.ForgettingQlearning(agent_params)
    elif agent_name == 'Logistic':
        agent = agents.Logistic(agent_params, task_params)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def set_params() -> tuple[task_config.TaskParams, task_config.AgentParams]:
    ## TASK PARAMETERS ##
    task_params = task_config.TaskParams()

    task_params.n_states = 2
    task_params.n_actions = 2

    # Task structure parameters, for markov and success_trigger. All between 0 and 1
    task_params.block_transition_style = 'success_trigger'  # 'success_trigger', 'markov', 'n_correct', 'fixed'
    task_params.p_cue = 0  # probability of cue being shown at a block transition
    task_params.t_cue = 2  # number of timesteps with cue present
    task_params.state_transition_prob = .1  # true task dynamics, for markov and success_trigger
    task_params.active_reward_probability = .9
    task_params.inactive_reward_probability = 0
    task_params.n_trials = 100

    # params for fixed block lengths
    task_params.default_fixed_block_length = 15
    task_params.fixed_block_length_variation = 0
    task_params.success_trials_to_block_transition = np.inf  # a maximum number of correct trials before a block transition is forced

    # reward size parameters
    task_params.mean_correct_reward = 1
    task_params.mean_incorrect_reward = 0
    task_params.reward_std_dev = 0
    task_params.ITI_lick_reward = 0  # for licking outside of go cue; don't think I'll use this
    task_params.wait_reward = 0  # for not licking during go cue. Pick 0 (no waiting penalty) or -1 (waiting penalty)


    ## AGENT PARAMETERS ##
    agent_params = task_config.AgentParams()

    # for the HMM agent, you can play with using model parameters that are different from the true task parameters
    agent_params.HMM_transition_prob = task_params.state_transition_prob  # HMM agent's belief of state change probability
    # agent_params.HMM_transition_prob = .1  # HMM agent's belief of task dynamics
    agent_params.HMM_active_reward_probability = .9  # HMM agent's belief of reward probability
    agent_params.HMM_inactive_reward_probability = 0  # HMM agent's belief of reward probability

    # Q-learning parameters
    agent_params.FQL_decay = .6  # for forgetting Q-learning agent
    agent_params.QL_learning_rate = .1  # for standard Q-learning agent
    agent_params.action_temperature = .2  # .3 seems to work for FQL  ## temp=1 is default; temp->0 makes greedier; temp->inf increases randomness
    agent_params.action_stickiness = 0  # tendency to repeat last action
    agent_params.greedy_action_selection = False
    agent_params.greedy_epsilon = .1

    # Logistic (and HMM-sticky) parameters
    agent_params.logistic_alpha = 1  # default 1 - action stickiness
    agent_params.logistic_beta = 2  # default 2 - action/reward history update size
    agent_params.logistic_tau = 1.5  # default 1.5 - memory for prior timestep settings, higher tau increases memory. 1.5 -> .5 memory

    # params.action_stickiness = 0  # use alpha instead. default for RFLR is 1; default for others is 0
    return task_params, agent_params


def save_experiment(save_name: str, data_dir: Path, task: BaseMDP, agent: agents.BehaviorAgent, performance: dict,
                    agent_params: task_config.AgentParams, task_params: task_config.TaskParams) -> None:
    exp = dict(task=vars(task), agent=vars(agent), agent_params=vars(agent_params), task_params=vars(task_params),
               performance=performance)
    date = str(dt.date.today().isoformat())
    save_dir = data_dir / date
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    p = save_dir / (save_name + '.pkl')
    with p.open('wb') as f:
        pkl.dump(exp, f)

    # print("[***] Experiment saved as: {}".format(p.name))
    print("[***] Experiment saved as: {}".format(p.resolve()))


def load_experiment(load_name: str) -> Tuple[BaseMDP, agents.BehaviorAgent, pd.DataFrame, task_config.TaskParams, task_config.AgentParams]:
    with open(load_name, 'rb') as f:
        exp = pkl.load(f)

    task_params = task_config.TaskParams()
    for k, v in exp['task_params'].items():
        setattr(task_params, k, v)

    agent_params = task_config.AgentParams()
    for k, v in exp['agent_params'].items():
        setattr(agent_params, k, v)

    task = BaseMDP(task_params)
    agent = agents.BehaviorAgent

    for k, v in exp['task'].items():
        setattr(task, k, v)

    for k, v in exp['agent'].items():
        setattr(agent, k, v)

    print("[***] Model restored from path: {}".format(load_name))
    return task, agent, exp['performance'], task_params, agent_params


def main():
    data_dir = Path('../../data/processed/model_experiments')
    model_dir = Path('../../saved_models/standard_models')

    log_path = Path('collection_task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    task_params, agent_params = set_params()
    session_note = ''

    # agent_name = 'F-Qlearning'
    # agent_param_str = 'alpha={}_temp={}_decay={}'.format(params.logistic_alpha, params.action_temperature, params.FQL_decay)

    # agent_name = 'HMM'
    # agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(params.logistic_alpha, params.action_temperature, params.HMM_transition_prob)

    # agent_name = 'HMM_recursive'
    # agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(params.logistic_alpha, params.action_temperature, params.HMM_transition_prob)

    agent_name = 'HMM_RFLR'
    agent_param_str = 'alpha={}_beta={}_tau={}'.format(agent_params.logistic_alpha, agent_params.logistic_beta,
                                                       agent_params.logistic_tau)

    # agent_name = 'Logistic'
    # agent_param_str = 'alpha={}_beta={}_tau={}'.format(agent_params.logistic_alpha, agent_params.logistic_beta,
    #                                                    agent_params.logistic_tau)

    # pick a file path based on agent type (HMM, Qlearning, etc.)
    # pick a filename based on task-specific params (pReward, pSwitch, rewards-before-state-switch, etc.)

    ### RUN A NEW EXPERIMENT ###
    session_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(agent_name,
                                                     task_params.active_reward_probability,
                                                     task_params.state_transition_prob)
    fname = session_name + '_' + agent_param_str
    if session_note:
        fname += '_' + session_note

    task = BaseMDP(task_params)
    agent = select_agent(agent_name, agent_params, task_params)
    task, agent, performance = run_experiment(task, agent, task_params)
    performance['session_ID'] = ['logistic_test'] * task.cur_trial
    performance = pd.DataFrame(performance)
    save_experiment(fname, data_dir, task, agent, performance, agent_params, task_params)

    ### REPEAT THE TRIALS FROM A MODEL SESSION ###
    p_reward = .9
    p_switch = .1
    alpha = .5
    beta = 2
    tau = 1.5
    temp = .2
    decay = .7
    HMM_transition_prob = .1

    # load_agent_name = 'F-Qlearning'
    # load_param_str = 'alpha={}_temp={}_decay={}'.format(alpha, temp, decay)
    # load_agent_name = 'HMM'
    # load_param_str = 'alpha={}_temp={}_model-Psw={}'.format(alpha, temp, HMM_transition_prob)
    # load_agent_name = 'Logistic'
    # load_param_str = 'alpha={}_beta={}_tau={}'.format(alpha, beta, tau)
    # extras_str = 'comparison-model'
    #
    # load_session_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(load_agent_name, p_reward, p_switch)
    # load_fname = load_session_name + '_' + load_param_str
    # if extras_str:
    #     load_fname += '_' + extras_str
    # p = Path('../saved_models') / (load_fname + '.pkl')
    # _, _, task_df, _ = load_experiment(p)


    ### REPEAT THE TRIALS FROM A MOUSE SESSION ###
    # mouse = 'MF03'
    # date = '2023-10-03'
    # sess_ID = mouse + '-' + date
    # p = Path('../mouse_behavior') / (sess_ID + '_performance.pkl')
    # with p.open('rb') as f:
    #     task_df = pkl.load(f)
    # fname = sess_ID + '_{}_{}'.format(agent_name, agent_param_str)
    # if extras_str:
    #     fname += '_' + extras_str
    # task_df['state'] = task_df['state'].astype(int)
    #
    #
    # task, agent, performance = run_repeat_experiment(agent_name, params, task_df, repeat_decisions=False)
    # performance['session_ID'] = np.zeros(task.cur_trial)
    # performance = pd.DataFrame(performance)
    # save_experiment(fname, task, agent, performance, params)


if __name__ == '__main__':
    main()
