import numpy as np

from dataclasses import dataclass, field
from typing import Protocol, List, Tuple, Optional, Union
from abc import ABC
import copy

SEED = 12345
# rng = np.random.default_rng(SEED)
rng = np.random.default_rng()


@dataclass
class TaskParams:
    block_transition_style: str = 'success_trigger'  # 'success_trigger', 'markov', 'n_correct', 'fixed'
    n_states: int = 2
    n_actions = 2
    # give_observations: bool = False

    # Task probability parameters, for markov and success_trigger. All between 0 and 1
    p_cue: float = .25
    t_cue: int = 2
    # p_switch: float = .1
    state_transition_prob: float = .1  # true task dynamics, for markov and success_trigger
    active_reward_probability: float = .9
    inactive_reward_probability: float = 0
    reward_std_dev: float = 0

    # params for fixed block lengths
    default_fixed_block_length: int = 15
    fixed_block_length_variation: int = 0
    success_trials_to_block_transition: float = np.inf
    # max_consecutive_blocks: int = 2

    # reward size parameters
    mean_correct_reward: float = 1
    mean_incorrect_reward: float = 0
    ITI_lick_reward: float = 0  # for licking outside of go cue; don't think I'll use this
    wait_reward: float = 0  # for not licking during go cue. Pick 0 (no waiting penalty) or -1 (waiting penalty)

    # fixed_block_sequence = []
    # fixed_block_lengths = []  # if this variable is set, the list of blocklengths must equal number of fixed blocks

    n_trials: int = 100

    # @property
    # def n_blocks(self) -> int:
    #     if type(self.blocks) is list:
    #         return len(self.blocks)
    #     else:
    #         return self.blocks


@dataclass
class AgentParams:
    # behavior agent params for non-ML agents
    HMM_transition_prob: float = .05  # HMM agent's belief of task dynamics
    FQL_decay: float = .9  # for forgetting Q-learning agent
    QL_learning_rate: float = .1  # for standard Q-learning agent
    action_stickiness: float = 0  # tendency to repeat last action
    greedy_action_selection: bool = False
    greedy_epsilon: float = .1
    action_temperature: float = 1

    logistic_alpha: float = 1  # default 1
    logistic_beta: float = 2  # default 2
    logistic_tau: float = 1.5  # default 1.5

    # for the HMM agent, you can play with using model parameters that are different from the true task parameters
    HMM_active_reward_probability: float = .9
    HMM_inactive_reward_probability: float = 0


class MarkovDecisionProcess(Protocol):
    # states: List[str] = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'intercontext_interval']
    # state_stimulus_dict['right_uncued'] = 2
    # state_stimulus_dict['left_uncued'] = 2
    # state_stimulus_dict['intercontext_interval'] = 3

    states: List[str] = ['right', 'left']
    state_dict = {s: i for i, s in enumerate(states)}
    state_stimulus_dict = copy.deepcopy(state_dict)

    cur_state: str
    cur_trial: int
    cur_block: int
    params: TaskParams

    def get_stimulus(self) -> int:
        ...
        # return self.state_stimulus_dict[self.cur_state]

    def step(self, action: int) -> Tuple[float, float]:
        ...


if __name__ == '__main__':
    Params = TaskParams()
    print(Params.n_blocks)
    print('test')



