import numpy as np

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
from abc import ABC
import copy


@dataclass
class TaskParams:
    ### hold the parameters for an MDP task ###
    # block transition style - fixed order, random order but fixed length, fully markov
    blocks: Union[int, List[str]]

    markov_block_transitions: bool = False
    trigger_block_transitions: bool = False
    default_block_length: int = 30
    block_length_variation: int = 0
    success_trials_to_block_transition: float = np.inf
    max_consecutive_blocks: int = 2
    give_observations: bool = False

    mean_correct_reward: int = 1  # correct reward center, incorrect reward center
    mean_incorrect_reward: int = 0  # correct reward center, incorrect reward center
    effort_loss = -1  # don't think I'll use this
    n_actions = 2

    active_reward_probability = 1
    inactive_reward_probability = 0
    reward_std_dev = 0

    # Transitions for random block transitions
    state_transition_prob: float = .05  # true task dynamics
    n_states: int = 2
    p_cue = .5

    # some agent params
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

    fixed_block_sequence = []
    fixed_block_lengths = []  # if this variable is set, the list of block lengths must equal number of fixed blocks

    @property
    def n_blocks(self) -> int:
        if type(self.blocks) is list:
            return len(self.blocks)
        else:
            return self.blocks


class MarkovDecisionProcess(ABC):

    def __init__(self, params: TaskParams):
        self.params: TaskParams = params

        self.states: List[str] = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'intercontext_interval']
        self.state_dict = {s: i for i, s in enumerate(self.states)}
        self.state_stimulus_dict = copy.deepcopy(self.state_dict)
        self.state_stimulus_dict['right_uncued'] = 2
        self.state_stimulus_dict['left_uncued'] = 2
        self.state_stimulus_dict['intercontext_interval'] = 3
        self.cur_state = 0

    def get_stimulus(self) -> int:
        return self.state_stimulus_dict[self.cur_state]

    def step(self, action: int) -> Tuple[float, float]:
        ...


if __name__ == '__main__':
    Params = TaskParams()
    print(Params.n_blocks)
    print('test')



