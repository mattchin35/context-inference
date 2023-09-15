import numpy as np

from dataclasses import dataclass
from typing import Protocol, List, Tuple
import copy

SEED = 1000
np.random.seed(SEED)
N_ACTIONS = 2

"""
TODO - implement saving of parameters?
"""

@dataclass
class TaskParams:
    ### hold the parameters for an MDP task ###
    random_block_order: bool = False  # random, fixed
    markov_block_transitions: bool = False  # random, fixed

    fixed_block_sequence = ['left_cued', 'right_cued', 'left_uncued', 'right_uncued']

    default_block_length: int = 30
    block_length_variation: int = 0
    success_trials_to_block_transition: float = np.inf

    mean_correct_reward: int = 5  # correct reward center, incorrect reward center
    mean_incorrect_reward: int = 1  # correct reward center, incorrect reward center
    effort_loss = -1  # dunno if I'll use this
    n_actions = 2

    active_reward_probability = 1
    inactive_reward_probability = 1
    reward_std_dev = .25

    length_intercontext_interval: int = 0

    # Transitions for random block transitions
    transition_prob: float = .05
    n_states: int = 4

    @property
    def n_blocks(self) -> int:
        if self.random_block_order:
            n_blocks = 10
        else:
            n_blocks = len(self.fixed_block_sequence)
        return n_blocks


class MarkovDecisionProcess(Protocol):
    rng = np.random.default_rng(SEED)

    states: List[str] = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'intercontext_interval']
    state_dict = {s: i for i, s in enumerate(states)}
    state_stimulus_dict = copy.deepcopy(state_dict)
    state_stimulus_dict['right_uncued'] = 2
    state_stimulus_dict['left_uncued'] = 2
    state_stimulus_dict['intercontext_interval'] = 3

    # transitions: List[dict] = [
    #     {'trigger': 'to_intercontext', 'source': ['left_cued', 'right_cued', 'left_uncued', 'right_uncued'],
    #      'dest': 'intercontext_interval'},
    #     {'trigger': 'to_left_cued', 'source': ['right_cued', 'left_uncued', 'right_uncued', 'intercontext_interval'],
    #      'dest': 'intercontext_interval'},
    #     {'trigger': 'to_right_cued', 'source': ['left_cued', 'left_uncued', 'right_uncued', 'intercontext_interval'],
    #      'dest': 'intercontext_interval'},
    #     {'trigger': 'to_left_uncued', 'source': ['left_cued', 'right_cued', 'right_uncued', 'intercontext_interval'],
    #      'dest': 'intercontext_interval'},
    #     {'trigger': 'to_right_uncued', 'source': ['left_cued', 'right_cued', 'left_uncued', 'intercontext_interval'],
    #      'dest': 'intercontext_interval'},
    #     {'trigger': 'internal', 'source': ['left_cued', 'right_cued', 'left_uncued', 'right_uncued'], 'dest': None}
    # ]

    cur_state: str
    cur_trial: int
    cur_block: int
    params: TaskParams

    def get_stimulus(self) -> int:
        return self.state_stimulus_dict[self.cur_state]

    def step(self, action: int) -> Tuple[float, float]:
        ...


if __name__ == '__main__':
    Params = TaskParams()
    print(Params.n_blocks)
    print('test')



