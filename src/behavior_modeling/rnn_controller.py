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

from src.behavior_modeling.task import rnn_task
from src.behavior_modeling.parameters import task_config, rnn_config
from src.behavior_modeling.agents import rnn_model
from src.behavior_modeling.fileIO import rnn_io

SEED = 12345
# torch.manual_seed(SEED)
# rng = np.random.default_rng(SEED)
rng = np.random.default_rng()


data_dir = Path('../../data/processed/rnn_experiments')
model_dir = Path('../../saved_models/RNN')


def select_agent(agent_name: str, rnn_params: rnn_config.AgentConfig) -> torch.nn.Module:
    """
    Refactor the data_dims into the rnn_config or the task_config; make it a dictionary for reader clarity
    """
    # data_dims = (4, 2, opts.rnn_size)  # dX, dY, rnn_size
    data_dims = dict(dim_ipt=4, dim_opt=2, rnn_size=rnn_params.rnn_size)
    if agent_name == 'RNN_reinforce':
        agent = rnn_model.RNN_reinforce(data_dims, rnn_params)
    elif agent_name == 'RNN_A2C':
        agent = rnn_model.RNN_A2C(data_dims, rnn_params)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def step_RNN(task: rnn_task.RnnMDP, agent: torch.nn.Module, performance: Dict[str, list],
             rnn_dict: Dict[str, np.ndarray], rnn_params: rnn_config.AgentConfig) -> Tuple[rnn_task.RnnMDP, torch.nn.Module, Dict[str, list], Dict[str, np.ndarray]]:
    """make this ready for policy gradient rnn and the state-inference version"""

    performance['states'].append(task.cur_state)
    performance['cur_trial'].append(task.cur_trial)
    performance['cur_block'].append(task.cur_block)
    performance['cur_trial_in_block'].append(task.cur_trial_in_block)
    performance['p_active_reward'].append(task.params.active_reward_probability)
    performance['p_inactive_reward'].append(task.params.inactive_reward_probability)

    # create input vector of last reward
    if performance['reward']:
        last_reward = performance['reward'][-1]
    else:
        last_reward = 0
    stimulus = task.get_stimulus()

    # inputs = np.array([last_reward, stimulus])[None, :]
    # inputs = torch.tensor([last_reward])[None, :]
    inputs = task.input_vector
    inputs_torch = torch.tensor(inputs)[None, :]
    noise = torch.randn(size=(1, rnn_params.rnn_size)) * .05

    # create an input vector of action and stimulus (if applicable)
    next_state, action_dist, action = agent.forward(inputs_torch.float(), noise.float())
    reward, correct = task.step(action)
    agent.rewards.append(reward)

    rnn_dict['inputs'].append(inputs)
    rnn_dict['noise'].append(np.squeeze(noise.numpy()))
    rnn_dict['action_dist'].append(np.squeeze(action_dist.detach().numpy()))

    performance['stimulus'].append(stimulus)
    performance['action_dist'].append(np.squeeze(action_dist.detach().numpy())[0])

    # this will get refactored out to a controller class/fxn
    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    return task, agent, performance, rnn_dict


def train_RNN(task: rnn_task.RnnMDP, agent: torch.nn.Module, rnn_params: rnn_config.AgentConfig, task_params: task_config.TaskParams):
    performance = defaultdict(list)
    rnn_dict = defaultdict(list)
    # while task.cur_block < task.params.n_blocks and task.cur_trial < max_trials:
    while task.cur_trial < task_params.n_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance, rnn_dict = step_RNN(task, agent, performance, rnn_dict, rnn_params)

    states = torch.stack(agent.state_series, dim=1)
    predictions = torch.stack(agent.prediction_series, dim=1)
    states, predictions = states.detach().numpy(), predictions.detach().numpy()

    total_loss, error_loss, activity_loss, weight_loss = agent.backward()
    assert not np.isnan(error_loss), f"Error is NaN, retry"

    rnn_dict['states'] = states
    rnn_dict['predictions'] = predictions
    rnn_dict['total_loss'] = total_loss
    rnn_dict['error_loss'] = error_loss
    rnn_dict['activity_loss'] = activity_loss
    rnn_dict['weight_loss'] = weight_loss
    # print("Performance: {}% correct".format(np.mean(performance['correct'])))
    return task, agent, performance, rnn_dict


def eval(task: rnn_task.RnnMDP, agent: torch.nn.Module, rnn_params: rnn_config.AgentConfig, save_name: str):
    task.initialize_task_state()
    agent.reset()

    # Generate and evaluate a test set for network analysis.
    print('[*] Testing [*]')
    performance = defaultdict(list)
    rnn_dict = defaultdict(list)
    while task.cur_trial < task.task_params.n_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance, rnn_dict = step_RNN(task, agent, performance, rnn_dict, rnn_params)

    states = torch.stack(agent.state_series, dim=1)
    predictions = torch.stack(agent.prediction_series, dim=1)
    states, predictions = states.detach().numpy(), predictions.detach().numpy()

    total_loss, error_loss, activity_loss, weight_loss = agent.backward(apply_gradient=False)
    assert not np.isnan(error_loss), f"Error is NaN, retry"

    rnn_dict['hidden_states'] = states
    rnn_dict['predictions'] = predictions
    rnn_dict['total_loss'] = total_loss
    rnn_dict['error_loss'] = error_loss
    rnn_dict['activity_loss'] = activity_loss
    rnn_dict['weight_loss'] = weight_loss
    # rnn_dict['performance'] = pd.DataFrame(performance)
    rnn_dict['task'] = vars(task)
    rnn_dict['agent_params'] = vars(rnn_params)
    rnn_dict['agent_name'] = agent.name
    rnn_dict['task_params'] = task.task_params

    # p_agent = Path('../saved_models') / (save_path + '.pkl')
    # with open(p_agent, 'wb') as f:
    #     pkl.dump(rnn_dict, f)
    rnn_io.save_experiment(save_name, data_dir, task, performance, rnn_dict, agent.name, task.task_params, rnn_params)

    # date = str(dt.date.today().isoformat())
    #
    # # save the experiment
    # save_dir = data_dir / date
    # if not save_dir.exists():
    #     save_dir.mkdir(parents=True)
    #
    # p_dict = save_dir / (save_name + '_dict.pkl')
    # with open(p_dict, 'wb') as f:
    #     pkl.dump(rnn_dict, f)
    #
    # print("[***] Eval saved as: {}".format(p_dict.name))


def set_params(task_params: task_config.TaskParams, rnn_params: rnn_config.AgentConfig) -> Tuple[task_config.TaskParams, rnn_config.AgentConfig]:
    # set the parameters for the task and the agent
    task_params.n_trials = 500
    task_params.mean_correct_reward = 1
    task_params.mean_incorrect_reward = -1
    task_params.active_reward_probability = .9
    task_params.inactive_reward_probability = 0
    task_params.reward_std_dev = 0
    task_params.p_cue = .5

    task_params.block_transition_style = 'success_trigger'  # markov, success_trigger, n_correct, fixed
    task_params.default_block_length = 10  # for non-probablistic block lengths
    task_params.state_transition_prob = .1
    task_params.block_length_variation = 1

    rnn_params.rnn_size = 200
    rnn_params.learning_rate = .0001
    rnn_params.decay = 1
    rnn_params.epoch = 1000
    rnn_params.weight_loss = .1
    rnn_params.activity_loss = .1
    rnn_params.baseline = 0  # alternately, use params.active_reward_probability
    return task_params, rnn_params


def set_eval_params(task_params: task_config.TaskParams) -> task_config.TaskParams:
    # set the parameters for the task and the agent
    task_params.n_trials = 10000
    task_params.mean_correct_reward = 1
    task_params.mean_incorrect_reward = -1
    task_params.active_reward_probability = .9
    task_params.inactive_reward_probability = 0
    task_params.reward_std_dev = 0
    task_params.p_cue = .5

    task_params.block_transition_style = 'success_trigger'  # markov, success_trigger, n_correct, fixed
    task_params.default_block_length = 10  # for non-probablistic block lengths
    task_params.state_transition_prob = .1
    task_params.block_length_variation = 3
    return task_params


def main():
    log_path = Path('./logs/rnn_training.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    task_params = task_config.TaskParams()
    rnn_params = rnn_config.AgentConfig()
    agent_name = 'RNN_reinforce'
    note = 'overtrain'

    ### ADJUST THE set_params FUNCTION TO SET THE PARAMETERS FOR THE TASK AND THE AGENT ###
    task_params, rnn_params = set_params(task_params, rnn_params)

    task = rnn_task.RnnMDP(task_params=task_params, rnn_params=rnn_params)
    agent = select_agent(agent_name, rnn_params)
    # pick a filename based on task-specific params (pReward, pSwitch, rewards-before-state-switch, etc.)

    if task_params.block_transition_style == 'success_trigger':
        switch_str = 'pSwitch_{}'.format(task_params.state_transition_prob)
    elif task_params.block_transition_style == 'fixed':
        switch_str = 'fixedBlocks_{}'.format(task_params.default_block_length)
    else:
        ValueError("Unprepared block transition style: {}".format(task_params.block_transition_style))

    session_name = '{}_pReward_{}_{}'.format(agent_name,
                                             task_params.active_reward_probability,
                                             switch_str)
    if note:
        session_name = session_name + '_' + note

    load_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name, .9, .1)
    # load_name = '{}_pReward_{}_fixedBlocks_{}'.format(agent_name, .9, 10)
    date = '2024-04-11'
    load_note = 'overtrain'
    if load_note:
        load_checkpoint = model_dir / date / (load_name + '_' + load_note + '_agent.pkl')
    else:
        load_checkpoint = model_dir / date / (load_name + '_agent.pkl')
        # load_checkpoint = ''

    if load_checkpoint:
        # rnn_io.load_agent_state(agent, load_checkpoint)
        print('Loading from checkpoint {}'.format(load_checkpoint.resolve()))
        agent.load_state_dict(torch.load(load_checkpoint))

    train = True
    evaluate = True
    if train:
        t = time.perf_counter()
        for ep in range(rnn_params.epoch):
            task.initialize_task_state()
            task, agent, performance, rnn_dict = train_RNN(task, agent, rnn_params, task_params)

            total_loss = rnn_dict['total_loss']
            error_loss = rnn_dict['error_loss']
            activity_loss = rnn_dict['activity_loss']
            weight_loss = rnn_dict['weight_loss']
            if ep > 0 and (ep+1) % 20 == 0:  # display in terminal
                print('[*] Epoch %d  total_loss=%.2f mse_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                      % (ep+1, total_loss, error_loss, activity_loss, weight_loss))
                print("Performance: {}% correct".format(np.mean(performance['correct'])))
                tnew = time.perf_counter()
                print(f'{tnew - t} seconds elapsed')
                t = tnew

            # if np.abs(error_loss) < 1:
            #     print('[*] Epoch %d  total_loss=%.2f mse_loss=%.2f a_loss=%.2f, w_loss=%.2f'
            #           % (ep + 1, total_loss, error_loss, activity_loss, weight_loss))
            #     tnew = time.perf_counter()
            #     print(f'{tnew - t} seconds elapsed')
            #     t = tnew
            #     break

            # performance['session_ID'] = np.zeros(task.cur_trial)
            if (ep > 0 and (ep + 1) % 100 == 0) or ep == rnn_params.epoch - 1:
                performance = pd.DataFrame(performance)
                rnn_io.save_experiment(session_name, data_dir, task, performance, rnn_dict, agent_name, task_params, rnn_params)
                rnn_io.save_agent(session_name, model_dir, agent)
                # save_experiment(session_name, task, agent, performance, task_params, rnn_params)

    if evaluate:
        # params.default_block_length = 10
        task_params = set_eval_params(task_params)
        task = rnn_task.RnnMDP(task_params=task_params, rnn_params=rnn_params)
        eval(task, agent, rnn_params, save_name='{}_eval'.format(session_name))

    # load
    # _, rnn_agent, performance, params, agent_opts = load_experiment(session_name)
    #
    # agent_name = 'HMM'
    # task = context_task.RepeatMDP(params, performance)
    # agent = select_agent(agent_name, params)
    # run_repeat_experiment(task, agent, performance)


if __name__ == '__main__':
    main()
