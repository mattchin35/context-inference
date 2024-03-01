from pathlib import Path
import pickle as pkl

from typing import Dict, Tuple, Any
import logging

import agents
import context_task
import MDP
import controller


def run_figure_1D():
    log_path = Path('task.log')
    log_path.unlink(missing_ok=True)
    logging.basicConfig(format='%(message)s', filename='task.log', level=logging.INFO)

    params = MDP.TaskParams()
    params.fixed_block_sequence = ['left', 'right'] * 2

    params.mean_correct_reward = 1
    params.mean_incorrect_reward = 0
    params.active_reward_probability = .9
    params.inactive_reward_probability = 0
    params.reward_std_dev = 0
    params.p_cue = 0

    temp = 1
    stickiness = 0

    # task = context_task.POMDP(params)

    # agent = agents.Qlearning(learning_rate=.3)
    # name = 'Qlearning'

    # agent = agents.ForgettingQlearning(decay=.8)
    # name = 'F-Qlearning'

    # 2 state HMM only
    # agent = agents.HMM_2state(params, transition_prob=.05, stickiness=0)
    # name = 'HMM_2state'

    agent = agents.HMM_2state(params, transition_prob=.05, stickiness=.25)
    name = 'StickyHMM_2state'

    task = context_task.BaseMDP(params)
    task, agent, performance = controller.run_experiment(task, agent, params.n_blocks)
    controller.save_experiment(name, task, agent, performance, params)
    # task, agent, performance, params = load_experiment(name)