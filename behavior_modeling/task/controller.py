from pathlib import Path
import pickle as pkl

from typing import Dict, Tuple, Any
import logging

import agents
import context_task
import MDP


def run_task_cycle(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: Dict[str, list]) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:
    action = agent.choose_action()
    reward, correct = task.step(action)
    agent.update_params(action, reward)

    # this will get refactored out to a controller class/fxn
    performance['states'].append(task.cur_state)
    performance['action'].append(action)
    performance['correct'].append(correct)
    performance['reward'].append(reward)
    performance['cur_trial'].append(task.cur_trial)
    performance['cur_block'].append(task.cur_block)
    if agent.model_type == 'HMM':
        performance['prior'].append(agent.prior)
    return task, agent, performance


def run_experiment(task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, n_blocks: int) -> \
        Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list]]:
    performance = dict(states=[], action=[], correct=[], reward=[], cur_trial=[], cur_block=[], prior=[])
    while task.cur_block < n_blocks:
        logging.info("block {}, trial {}".format(task.cur_block, task.cur_trial))
        task, agent, performance = run_task_cycle(task, agent, performance)

    return task, agent, performance


def save_experiment(save_name: str, task: MDP.MarkovDecisionProcess, agent: agents.BehaviorAgent, performance: dict,
                    params: MDP.TaskParams) -> None:
    exp = dict(task=vars(task), agent=vars(agent), params=vars(params), performance=performance)
    p = Path('../saved_models') / save_name
    p = p.with_suffix('.pkl')
    with p.open('wb') as f:
        pkl.dump(exp, f)

    print("[***] Experiment saved as: {}".format(p.name))


def load_experiment(load_name: str) -> Tuple[MDP.MarkovDecisionProcess, agents.BehaviorAgent, Dict[str, list], MDP.TaskParams]:
    p = Path('../saved_models') / load_name
    p = p.with_suffix('.pkl')

    with open(p, 'rb') as f:
        exp = pkl.load(f)

    params = MDP.TaskParams
    for k, v in exp['params'].items():
        setattr(params, k, v)

    task = context_task.POMDP(params)
    agent = agents.BehaviorAgent
    performance = dict()

    for k, v in exp['task'].items():
        setattr(task, k, v)

    for k, v in exp['agent'].items():
        setattr(agent, k, v)

    for k, v in exp['performance'].items():
        performance[k] = v

    print("[***] Model restored from path: {}".format(p.name))

    return task, agent, performance, params


def main():
    log_path = Path('task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    params = MDP.TaskParams()
    # params.fixed_block_sequence = ['left_cued', 'right_cued', 'left_uncued', 'right_uncued'] * 2
    params.fixed_block_sequence = ['left_uncued', 'right_uncued'] * 2

    params.mean_correct_reward = 5
    params.mean_incorrect_reward = 5
    params.active_reward_probability = .5
    params.inactive_reward_probability = 0
    params.reward_std_dev = 0

    task = context_task.POMDP(params)

    # agent = agents.Qlearning()
    # name = 'Qlearning'

    # agent = agents.ForgettingQlearning()
    # name = 'F-Qlearning'

    transition_prob = 1 / params.default_block_length
    # agent = agents.HMM(task)
    agent = agents.HMM(task, transition_prob=.03, stickiness=0)
    name = 'HMM'

    # agent = agents.HMM(task, transition_prob=.03, stickiness=.25)
    # name = 'StickyHMM'

    task, agent, performance = run_experiment(task, agent, params.n_blocks)
    save_experiment(name, task, agent, performance, params)
    # task, agent, performance, params = load_experiment(name)


if __name__ == '__main__':
    main()

