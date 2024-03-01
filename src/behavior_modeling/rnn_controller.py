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
import controller

SEED = 0
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def select_agent(agent_name: str, params: MDP.TaskParams, opts: Optional[AgentConfig] = None) -> \
        MDP.MarkovDecisionProcess:
    data_dims = (1, 2, opts.rnn_size)  # dX, dY, rnn_size
    if agent_name == 'RNN_reinforce':
        agent = pg_model.RNN(data_dims, opts)
    elif agent_name == 'RNN_A2C':
        agent = pg_model.RNN(data_dims, opts)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def run_RNN_task_cycle(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: Dict[str, list],
                       rnn_dict: Dict[str, np.ndarray], opts: AgentConfig) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list], Dict[str, np.ndarray]]:
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
    inputs = torch.tensor([last_reward])[None, :]
    noise = torch.randn(size=(1, opts.rnn_size)) * .05

    # create an input vector of action and stimulus (if applicable)
    next_state, action_dist, action = agent.forward(inputs.float(), noise.float())
    reward, correct = task.step(action)
    agent.rewards.append(reward)

    rnn_dict['inputs'].append(np.squeeze(inputs.numpy()))
    rnn_dict['noise'].append(np.squeeze(noise.numpy()))
    rnn_dict['action_dist'].append(np.squeeze(action_dist.detach().numpy()))
    performance['stimulus'].append(stimulus)
    performance['action_dist'].append(np.squeeze(action_dist.detach().numpy())[0])

    # this will get refactored out to a controller class/fxn
    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    return task, agent, performance, rnn_dict


def train_RNN(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, opts: AgentConfig, max_trials: int = 1000):
    performance = defaultdict(list)
    rnn_dict = defaultdict(list)
    while task.cur_block < task.params.n_blocks and task.cur_trial < max_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance, rnn_dict = run_RNN_task_cycle(task, agent, performance, rnn_dict, opts)

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

    return task, agent, performance, rnn_dict


def save_experiment(save_name: str, task: MDP.MarkovDecisionProcess, agent: torch.nn.Module, performance: dict,
                    params: MDP.TaskParams, agent_opts: AgentConfig) -> None:
    exp = dict(task=vars(task), params=vars(params), performance=performance, agent_opts=vars(agent_opts))
    p_dict = Path('../saved_models') / (save_name + '_dict.pkl')
    with p_dict.open('wb') as f:
        pkl.dump(exp, f)

    p_agent = Path('../saved_models') / (save_name + '_agent.pkl')
    torch.save(agent.state_dict(), p_agent)

    print("[***] Experiment saved as: {}; Agent saved as: {}".format(p_dict.name, p_agent.name))


def load_experiment(load_name: str) -> Tuple[MDP.MarkovDecisionProcess, torch.nn.Module, pd.DataFrame, MDP.TaskParams, AgentConfig]:
    p_dict = Path('../saved_models') / (load_name + '_dict.pkl')
    with open(p_dict, 'rb') as f:
        exp = pkl.load(f)

    params = MDP.TaskParams(blocks=1000)
    for k, v in exp['params'].items():
        setattr(params, k, v)

    task = context_task.BaseMDP(params)
    agent_opts = AgentConfig
    for k, v in exp['agent_opts'].items():
        setattr(agent_opts, k, v)

    p_agent = Path('../saved_models') / (load_name + '_agent.pkl')
    data_dims = (1, 2, agent_opts.rnn_size)  # dX, dY, rnn_size
    agent = pg_model.RNN_reinforce(data_dims, agent_opts)
    agent.load_state_dict(torch.load(p_agent))
    agent.eval()

    for k, v in exp['task'].items():
        setattr(task, k, v)

    print("[***] Model restored from path: {}".format(load_name))
    return task, agent, exp['performance'], params, agent_opts


def eval(task: MDP.MarkovDecisionProcess, agent: torch.nn.Module, agent_opts: AgentConfig, save_path: str, max_trials: int = 1000):
    # Generate and evaluate a test set for network analysis.
    print('[*] Testing')
    performance = defaultdict(list)
    rnn_dict = defaultdict(list)
    while task.cur_block < task.params.n_blocks and task.cur_trial < max_trials:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance, rnn_dict = run_RNN_task_cycle(task, agent, performance, rnn_dict, agent_opts)

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
    rnn_dict['performance'] = pd.DataFrame(performance)

    p_agent = Path('../saved_models') / (save_path + '.pkl')
    with open(p_agent, 'wb') as f:
        pkl.dump(rnn_dict, f)

    print("[***] Eval saved as: {}".format(p_agent.name))


def rnn_main():
    log_path = Path('collection_task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    params = MDP.TaskParams(blocks=10)
    # params.fixed_block_sequence = ['left', 'right']

    params.mean_correct_reward = 1
    params.mean_incorrect_reward = 0
    params.active_reward_probability = .9
    params.inactive_reward_probability = 0
    params.reward_std_dev = 0
    params.p_cue = 0

    params.markov_block_transitions = False
    params.default_block_length = 10
    if params.markov_block_transitions:
        params.state_transition_prob = .3
    else:
        params.state_transition_prob = 1 / params.default_block_length
        # params.state_transition_prob = .3
    params.block_length_variation = 3
    params.max_consecutive_blocks = 1

    agent_name = 'RNN_policy_gradient'

    # pick a file path based on agent type (HMM, Qlearning, etc.)
    # pick a filename based on task-specific params (pReward, pSwitch, rewards-before-state-switch, etc.)
    session_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name,
                                                     params.active_reward_probability,
                                                     params.state_transition_prob)

    task = context_task.BaseMDP(params)
    opts = AgentConfig()
    opts.rnn_size = 100
    opts.learning_rate = .0001
    opts.epoch = 5000
    opts.weight_loss = .1
    opts.activity_loss = .1
    opts.baseline = 0
    # opts.baseline = params.active_reward_probability

    agent = controller.select_agent(agent_name, params, opts)
    load_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name, .9, .05)
    load_checkpoint = Path('../saved_models') / (load_name + '_agent.pkl')
    # load_checkpoint = ''
    if load_checkpoint:
        print('Loading from checkpoint {}'.format(load_checkpoint))
        agent.load_state_dict(torch.load(load_checkpoint))

    train = True
    evaluate = True

    if train:
        t = time.perf_counter()
        for ep in range(opts.epoch):
            task.reset()
            task, agent, performance, rnn_dict = train_RNN(task, agent, opts, max_trials=100)

            total_loss = rnn_dict['total_loss']
            error_loss = rnn_dict['error_loss']
            activity_loss = rnn_dict['activity_loss']
            weight_loss = rnn_dict['weight_loss']
            if ep > 0 and (ep+1) % 20 == 0:  # display in terminal
                print('[*] Epoch %d  total_loss=%.2f mse_loss=%.2f a_loss=%.2f, w_loss=%.2f'
                      % (ep+1, total_loss, error_loss, activity_loss, weight_loss))
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

        performance['session_ID'] = np.zeros(task.cur_trial)
        performance = pd.DataFrame(performance)
        save_experiment(session_name, task, agent, performance, params, opts)

    if evaluate:
        # params.default_block_length = 10
        task = context_task.BaseMDP(params)
        agent.reset()
        eval(task, agent, opts, '{}_eval'.format(session_name), max_trials=1000)

    # load
    # _, rnn_agent, performance, params, agent_opts = load_experiment(session_name)
    #
    # agent_name = 'HMM'
    # task = context_task.RepeatMDP(params, performance)
    # agent = select_agent(agent_name, params)
    # run_repeat_experiment(task, agent, performance)


if __name__ == '__main__':
    rnn_main()