from context_task import BaseMDP
import src.behavior_modeling.parameters.task_config as task_config
import src.behavior_modeling.parameters.rnn_config as rnn_config
import numpy as np
import logging
from typing import Tuple
from icecream import ic


SEED = 12345
rng = np.random.default_rng(SEED)


class RnnMDP(BaseMDP):

    def __init__(self, task_params: task_config.TaskParams, rnn_params: rnn_config.AgentConfig):
        super(RnnMDP, self).__init__(task_params)
        self.task_params = task_params
        self.rnn_params = rnn_params
        self.input_vector = np.zeros(4)  # reward, go_cue, right_cue, left_cue

        self.go_cue = False
        self.len_ITI = 1
        self.ITI_counter = 0

        self.input_vector = np.zeros(4)
        self.output_vector = np.zeros(2)
        self.loss_timesteps = []
        self.initialize_task_state()

    def initialize_task_state(self):
        self.cur_state = rng.choice(self.states)
        self.cur_block = 0
        self.cur_trial = 0
        self.cur_trial_in_block = 0
        self.correct_in_block = 0
        self.go_cue = True
        self.set_input_vector(reward=0)
        self.set_output_vector()
        self.loss_timesteps = []

        if self.params.fixed_block_lengths:
            self.cur_block_length = self.params.fixed_block_lengths[0]
        else:
            self.cur_block_length = self.params.default_block_length

    def set_input_vector(self, reward: float):
        """For update, sample the stimulus probability and leave it until len_stimulus time has passed."""
        if self.cur_trial_in_block == 0 and rng.random() < self.p_cue and self.go_cue:
            stimulus_vector = self.get_stimulus()
        else:
            stimulus_vector = np.zeros(2)

        self.input_vector = np.concatenate([np.array([reward, int(self.go_cue)]), stimulus_vector])

    def set_output_vector(self):
        self.output_vector = np.zeros(2)
        self.output_vector[self.state_dict[self.cur_state]] = 1

    def get_stimulus(self) -> np.ndarray:
        cue_ix = self.state_stimulus_dict[self.cur_state]
        cue_vector = np.zeros(2)
        cue_vector[cue_ix] = 1
        return cue_vector

    def is_action_correct(self, action: np.ndarray) -> bool:
        action = np.squeeze(np.argwhere(action))
        if self.cur_state == 'right' and action == 0:
            correct = True

        # elif self.cur_state == 1 and action == 1:
        elif self.cur_state == 'left' and action == 1:
            correct = True

        else:
            correct = False

        return correct

    def determine_action_reward(self, correct: bool) -> float:
        """Simple version to start: if correct at go cue, give reward. Else no reward."""
        reward = 0
        if self.go_cue:
            if correct:
                if rng.random() < self.params.active_reward_probability:
                    if self.params.reward_std_dev > 0:
                        reward = rng.normal(self.params.mean_correct_reward, self.params.reward_std_dev)
                    else:
                        reward = self.params.mean_correct_reward

            else:
                if rng.random() < self.params.inactive_reward_probability:
                    if self.params.reward_std_dev > 0:
                        reward = rng.normal(self.params.mean_incorrect_reward, self.params.reward_std_dev)
                    else:
                        reward = self.params.mean_incorrect_reward

        reward = max(reward, 0)  # no negative rewards!
        return reward

    def step(self, action: np.ndarray) -> Tuple[float, float]:
        correct = self.is_action_correct(action)
        reward = self.determine_action_reward(correct)

        ic(self.cur_trial, self.ITI_counter, self.go_cue)
        ic(self.cur_state, action)
        ic(correct, reward)

        # logging.info("action: {}; reward: {}".format(action, reward))

        if self.go_cue:
            self.loss_timesteps.append(self.cur_trial)
            self.cur_trial_in_block += 1
            self.cur_trial += 1
            self.correct_in_block += int(correct)
            self.go_cue = False
            change_block = self.flag_block_transition(correct)

        else:
            change_block = False
            self.ITI_counter += 1
            if self.ITI_counter >= self.len_ITI:
                self.ITI_counter = 0
                self.go_cue = True

        ic(change_block)

        if change_block:
            logging.info('finished block {} with context {}'.format(self.cur_block, self.cur_state))
            logging.info('num correct in block: {}; num trials in block: {}'.format(self.correct_in_block,
                                                                                    self.cur_trial_in_block))
            self.correct_in_block = 0
            self.cur_trial_in_block = 0

            self.cur_block += 1
            if self.cur_trial < self.params.n_trials:
                self.to_next_state()

        self.set_input_vector(reward)
        self.set_output_vector()

        return reward, correct


def main():
    mdp = RnnMDP(task_params=task_config.TaskParams(), rnn_params=rnn_config.AgentConfig())
    mdp.step(action=np.array([1, 0]))
    mdp.step(action=np.array([1, 0]))



if __name__ == '__main__':
    main()
