import copy

import numpy as np
import pandas as pd
from typing import Tuple, List

import src.behavior_modeling.parameters.task_config as config
import logging

SEED = 12345
# rng = np.random.default_rng(SEED)
rng = np.random.default_rng()

N_ACTIONS = 2
RIGHT_IX = 0
LEFT_IX = 1


class BaseMDP:
    """
    Class for a 2 state version of the task, with a parameter for cue probability. 2 states (left active, right active),
    no intercontext intervals, p_cue probability of a cue.

    TODO
    - calculate the probability of each state as it goes.
    - make the transitions follow Vertechi 2020, requiring a correct choice to transition to the next block
    - fixed block sequence/fixed block length updates - mostly for repeats, not wanted for base task
    """

    def __init__(self, task_params: config.TaskParams):
        self.params = task_params

        self.states: List[str] = ['right', 'left']
        self.state_dict = {s: i for i, s in enumerate(self.states)}
        self.state_stimulus_dict = copy.deepcopy(self.state_dict)
        self.state_stimulus_dict['uncued'] = -1
        self.p_cue = task_params.p_cue
        self.initialize_task_state()

    def initialize_task_state(self):
        self.cur_state = rng.choice(self.states)
        self.cur_block = 0
        self.cur_trial = 0
        self.cur_trial_in_block = 0
        self.correct_in_block = 0

        if self.params.block_transition_style == 'fixed':
            self.cur_block_length = self.params.default_block_length

    # @property
    def transition_matrix(self) -> np.ndarray:
        """Return the transition matrix for the current state. Not used to calculate transitions, but can
        be used to calculate the probability of each state."""

        if self.params.block_transition_style == 'markov':
            # These transitions are the same regardless of the choice
            state_transition_matrix = np.ones((2, 2)) * self.params.state_transition_prob
            for i in range(2):
                state_transition_matrix[i, i] = 1 - self.params.state_transition_prob

        elif self.params.block_transition_style == 'success_trigger':
            if self.cur_state == 'right':
                # These transitions represent R-choice, R or L current state columns
                # transitions_asymmetric = np.ones((2, 2)) * self.params.state_transition_prob
                # transitions_asymmetric[0, 0] = 1 - self.params.state_transition_prob
                # transitions_asymmetric[0, 1] = 0
                # transitions_asymmetric[1, 1] = 1

                # These transitions represent R-active, R or L choice columns
                transitions_asymmetric = np.ones((2, 2)) * self.params.state_transition_prob
                transitions_asymmetric[0, 0] = 1 - self.params.state_transition_prob
                transitions_asymmetric[0, 1] = 1
                transitions_asymmetric[1, 0] = self.params.state_transition_prob
                transitions_asymmetric[1, 1] = 0

            elif self.cur_state == 'left':
                # These transitions represent L-choice, R or L current state columns
                # transitions_asymmetric = np.ones((2, 2)) * self.params.state_transition_prob
                # transitions_asymmetric[0, 0] = 1
                # transitions_asymmetric[0, 1] = self.params.state_transition_prob
                # transitions_asymmetric[1, 0] = 0
                # transitions_asymmetric[1, 1] = 1 - self.params.state_transition_prob

                # These transitions represent L-active, R or L choice columns
                transitions_asymmetric = np.ones((2, 2)) * self.params.state_transition_prob
                transitions_asymmetric[0, 0] = 0
                transitions_asymmetric[0, 1] = self.params.state_transition_prob
                transitions_asymmetric[1, 0] = 1
                transitions_asymmetric[1, 1] = 1 - self.params.state_transition_prob

            state_transition_matrix = transitions_asymmetric

        return state_transition_matrix

    def get_stimulus(self) -> int:
        cue = self.state_stimulus_dict[self.cur_state]
        if rng.random() < self.p_cue:
            return cue
        else:
            return -1  # indeterminate cue, either 2 or -1

    def is_action_correct(self, action) -> bool:
        # if self.cur_state == 0 and action == 0:
        if self.cur_state == 'right' and action == 0:
            correct = True

        # elif self.cur_state == 1 and action == 1:
        elif self.cur_state == 'left' and action == 1:
            correct = True

        else:
            correct = False

        return correct

    def determine_action_reward(self, correct: bool) -> float:
        if correct:
            if self.params.reward_std_dev > 0:
                reward = rng.normal(self.params.mean_correct_reward, self.params.reward_std_dev)
            else:
                reward = self.params.mean_correct_reward

        else:
            if self.params.reward_std_dev > 0:
                reward = rng.normal(self.params.mean_incorrect_reward, self.params.reward_std_dev)
            else:
                reward = self.params.mean_incorrect_reward

        reward = max(reward, 0)  # no negative rewards!
        return reward

    def to_next_state(self) -> None:
        logging.info("choosing next block type")
        # self.cur_state = 1 - self.cur_state
        prev_state = self.cur_state

        if self.cur_state == 'left':
            self.cur_state = 'right'
        elif self.cur_state == 'right':
            self.cur_state = 'left'

        logging.info("new block: {}; old block: {}".format(self.cur_state, prev_state))
        if self.params.block_transition_style in ['markov', 'success_trigger', 'n_correct']:
            return
        elif self.params.block_transition_style == 'fixed':
            pass
        else:
            raise AttributeError("block transition style not recognized")

        # if self.params.fixed_block_lengths:
        #     self.cur_block_length = self.params.fixed_block_lengths[self.cur_block]
        # else:
        self.cur_block_length = self.params.default_block_length
        if self.params.block_length_variation > 0:
            self.cur_block_length += rng.choice([-1, 1]) * \
                                     rng.integers(low=0, high=self.params.block_length_variation)

        # self.cur_trial_in_block = 0
        logging.info("new block: type {}, length {}".format(self.cur_state, self.cur_block_length))

    def step(self, action: int) -> Tuple[float, float]:
        correct = self.is_action_correct(action)
        if correct:
            give_reward = rng.random() < self.params.active_reward_probability
        else:
            give_reward = rng.random() < self.params.inactive_reward_probability

        if give_reward:
            reward = self.determine_action_reward(correct)
        else:
            reward = 0

        logging.info("action: {}; reward: {}".format(action, reward))

        self.cur_trial_in_block += 1
        self.cur_trial += 1
        self.correct_in_block += int(correct)

        change_block = self.flag_block_transition(correct)
        if change_block:
            logging.info('finished block {} with context {}'.format(self.cur_block, self.cur_state))
            logging.info('num correct in block: {}; num trials in block: {}'.format(self.correct_in_block, self.cur_trial_in_block))
            self.correct_in_block = 0
            self.cur_trial_in_block = 0

            self.cur_block += 1
            if self.cur_trial < self.params.n_trials:
                self.to_next_state()

        return reward, correct

    def flag_block_transition(self, correct: bool) -> bool:
        """Determine whether to move to the next block, returning a boolean."""
        if self.params.block_transition_style == 'markov':
            change_block = rng.random() < self.params.state_transition_prob

        elif self.params.block_transition_style == 'success_trigger' and correct:
            change_block = rng.random() < self.params.state_transition_prob
            # sample = rng.random()
            # change_block = sample < self.params.state_transition_prob
            # print(sample)
            # if correct:
            # else:
            #     change_block = False

        elif self.params.block_transition_style == 'n_correct':
            change_block = self.correct_in_block >= self.params.success_trials_to_block_transition

        elif self.params.block_transition_style == 'fixed':
            change_block = self.cur_trial_in_block >= self.cur_block_length

        else:
            change_block = False
            # raise AttributeError("block transition style not recognized")

        return change_block


class RepeatMDP(BaseMDP):
    """
    Repeat the exact trial conditions but with probabilistic rewards.
    """
    def __init__(self, params: config.TaskParams, task_df: pd.DataFrame):
        self.task_df = task_df
        self.n_trials = task_df.shape[0]
        super().__init__(params)

    def reset(self):
        self.cur_state = self.task_df.loc[0, 'state']
        self.cur_block = 0
        self.cur_trial = 0
        self.cur_trial_in_block = 0
        self.correct_in_block = 0
        self.state_counter = np.zeros(2)  # prevent the same state from being sampled more than twice in a row
        self.state_counter[self.cur_state] += 1

    def get_stimulus(self) -> int:
        return self.task_df['stimulus'].loc[self.cur_trial]

    def step(self, action) -> Tuple[float, float]:
        correct = self.is_action_correct(action)
        if correct:
            # give_reward = rng.random() < self.task_df.loc[self.cur_trial, 'active_reward_probability']
            give_reward = rng.random() < self.params.active_reward_probability
        else:
            # give_reward = rng.random() < self.task_df.loc[self.cur_trial, 'inactive_reward_probability']
            give_reward = rng.random() < self.params.inactive_reward_probability

        if give_reward:
            reward = self.determine_action_reward(correct)
        else:
            reward = 0

        self.cur_trial_in_block += 1
        self.cur_trial += 1
        self.correct_in_block += int(correct)
        if self.cur_trial < self.n_trials:
            self.cur_state = self.task_df['state'].loc[self.cur_trial]
            self.cur_block = self.task_df['cur_block'].loc[self.cur_trial]
            self.cur_trial_in_block = self.task_df['cur_trial_in_block'].loc[self.cur_trial]
        return reward, correct


if __name__ == '__main__':
    params = config.TaskParams()
    # task = POMDP(params)
    # task.params.random_block_order = True
    #
    # # task.to_intercontext()
    # task.to_next_state()
    # task.cur_trial_in_block = 50
    # rew = task.step_non_markov(0)
    # print(rew)
