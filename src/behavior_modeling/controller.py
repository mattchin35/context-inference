from pathlib import Path
import pickle as pkl
from typing import Dict, Tuple, Any, Optional
import logging
from collections import defaultdict
import time

import numpy as np
import pandas as pd
import torch

import agents
import context_task
import MDP
from config import AgentConfig
import pg_model

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def run_task_cycle(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: Dict[str, list]) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:

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


def run_repeat_task_cycle(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: Dict[str, list],
                          task_df: pd.DataFrame, repeat_decisions: int = False) -> Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:

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

    # for a repeat of a prior session, the action and its reward can be input rather than sampled
    stimulus = task_df['stimulus'].values[task.cur_trial]
    performance['stimulus'].append(stimulus)
    if repeat_decisions:
        action = task_df['action'].values[task.cur_trial]
        reward = task_df['reward'].values[task.cur_trial]
        _, action_dist = agent.choose_action(stimulus)
        _, correct = task.step(action)
    else:
        action, action_dist = agent.choose_action(stimulus)
        reward, correct = task.step(action)

    performance['action_dist'].append(action_dist[0])
    agent.update_params(action, reward)

    # this will get refactored out to a controller class/fxn
    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    return task, agent, performance


def run_experiment(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, max_trials: int = 1000) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:

    performance = defaultdict(list)
    while task.cur_block < task.params.n_blocks and task.cur_trial < max_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance = run_task_cycle(task, agent, performance)

    return task, agent, performance


def run_repeat_experiment(agent_type: str, params: MDP.TaskParams, task_df: pd.DataFrame, repeat_decisions: bool = False) -> \
            Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:

    task = context_task.RepeatMDP(params, task_df)
    agent = select_agent(agent_type, params)
    performance = defaultdict(list)
    while task.cur_trial < task_df.shape[0]:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance = run_repeat_task_cycle(task, agent, performance, task_df, repeat_decisions)

    return task, agent, performance


def select_agent(agent_name: str, params: MDP.TaskParams, opts: Optional[AgentConfig] = None) -> \
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
    elif agent_name == 'RNN_policy_gradient':
        data_dims = (1, 2, opts.rnn_size)  # dX, dY, rnn_size
        agent = pg_model.RNN(data_dims, opts)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def save_experiment(save_name: str, task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: dict,
                    params: MDP.TaskParams) -> None:
    exp = dict(task=vars(task), agent=vars(agent), params=vars(params), performance=performance)
    p = Path('../saved_models') / (save_name + '.pkl')
    with p.open('wb') as f:
        pkl.dump(exp, f)

    print("[***] Experiment saved as: {}".format(p.name))


def load_experiment(load_name: str) -> Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, pd.DataFrame, MDP.TaskParams]:
    with open(load_name, 'rb') as f:
        exp = pkl.load(f)

    params = MDP.TaskParams(blocks=1000)
    for k, v in exp['params'].items():
        setattr(params, k, v)

    task = context_task.BaseMDP(params)
    agent = agents.BehaviorAgent

    for k, v in exp['task'].items():
        setattr(task, k, v)

    for k, v in exp['agent'].items():
        setattr(agent, k, v)

    print("[***] Model restored from path: {}".format(load_name))
    return task, agent, exp['performance'], params


def main():
    log_path = Path('collection_task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    params = MDP.TaskParams(blocks=1000)

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
    params.block_length_variation = 15
    params.max_consecutive_blocks = 1

    params.FQL_decay = .6  # for forgetting Q-learning agent
    # params.HMM_transition_prob = params.state_transition_prob  # HMM agent's belief of task dynamics
    params.HMM_transition_prob = .1  # HMM agent's belief of task dynamics
    # params.HMM_transition_prob = .3  # HMM agent's belief of task dynamics

    params.logistic_alpha = 0  # default 1 - action stickiness
    params.logistic_beta = 1.5  # default 2 - action/reward history update size
    params.logistic_tau = 1  # default 1.5 - memory for prior timestep settings, higher tau increases memory. 1.5 -> .5 memory

    params.action_temperature = .2  # .3 seems to work for FQL  # temp=1 is default; temp->0 makes greedier; temp->inf increases randomness
    # params.action_stickiness = 0  # use alpha instead. default for RFLR is 1; default for others is 0

    # agent_name = 'F-Qlearning'
    # agent_param_str = 'alpha={}_temp={}_decay={}'.format(params.logistic_alpha, params.action_temperature, params.FQL_decay)

    agent_name = 'HMM'
    agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(params.logistic_alpha, params.action_temperature, params.HMM_transition_prob)

    # agent_name = 'HMM_RFLR'
    # agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(params.logistic_alpha, params.action_temperature, params.HMM_transition_prob)

    # agent_name = 'Logistic'
    # agent_param_str = 'alpha={}_beta={}_tau={}'.format(params.logistic_alpha, params.logistic_beta, params.logistic_tau)

    # pick a file path based on agent type (HMM, Qlearning, etc.)
    # pick a filename based on task-specific params (pReward, pSwitch, rewards-before-state-switch, etc.)

    ### RUN A NEW EXPERIMENT ###
    session_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(agent_name,
                                                     params.active_reward_probability,
                                                     params.state_transition_prob)
    fname = session_name + '_' + agent_param_str
    extras_str = 'comparison-model'
    if extras_str:
        fname += '_' + extras_str

    # task = context_task.BaseMDP(params)
    # agent = select_agent(agent_name, params)
    # task, agent, performance = run_experiment(task, agent, max_trials=10000)
    # performance['session_ID'] = np.zeros(task.cur_trial)
    # performance = pd.DataFrame(performance)
    # save_experiment(fname, task, agent, performance, params)

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
    mouse = 'MF03'
    date = '2023-10-03'
    sess_ID = mouse + '-' + date
    p = Path('../mouse_behavior') / (sess_ID + '_performance.pkl')
    with p.open('rb') as f:
        task_df = pkl.load(f)
    fname = sess_ID + '_{}_{}'.format(agent_name, agent_param_str)
    if extras_str:
        fname += '_' + extras_str
    task_df['state'] = task_df['state'].astype(int)


    task, agent, performance = run_repeat_experiment(agent_name, params, task_df, repeat_decisions=False)
    performance['session_ID'] = np.zeros(task.cur_trial)
    performance = pd.DataFrame(performance)
    save_experiment(fname, task, agent, performance, params)


if __name__ == '__main__':
    main()
