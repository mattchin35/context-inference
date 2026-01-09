class POMDP(MDP.MarkovDecisionProcess):
    """
    Class for the full 4-context version of the task, with options for intercontext intervals, probabilistic rewards, different reward sizes, etc.
    """
    def __init__(self, params: MDP.TaskParams):
        self.params = params

        if params.random_block_order:
            self.cur_state = self.states[rng.integers(4)]
        else:
            self.cur_state = params.fixed_block_sequence[0]
        self.cur_block = 0
        self.cur_trial = 0
        self.cur_trial_in_block = 0
        self.cur_block_length = self.params.default_block_length
        self.correct_in_block = 0

        # self.performance = dict(states=[], correct=[], reward=[], cur_trial=[], cur_block=[])  # will refactor this to controller later

    # from-to, ABC1C2
    @property
    def transition_matrix(self) -> np.ndarray:
        state_transition_matrix = np.ones((self.params.n_states, self.params.n_states)) * self.params.state_transition_prob
        for i in range(self.params.n_states):
            state_transition_matrix[i, i] = 1 - (self.params.n_states - 1) * self.params.state_transition_prob
        return state_transition_matrix

    @property
    def get_stimulus(self) -> int:
        return self.state_stimulus_dict[self.cur_state]

    def is_action_correct(self, action) -> bool:
        if self.cur_state in ['right_cued', 'right_uncued'] and action == 0:
            correct = True
        elif self.cur_state in ['left_cued', 'left_uncued'] and action == 1:
            correct = True
        # elif self.cur_state == 'intercontext_interval' and action == 2:
        #     correct = True
        else:
            correct = False
        return correct

    def determine_action_reward(self, correct: bool) -> float:
        # i.e. sample 0 to 1, give reward depending on outcome
        p = rng.random()
        if correct and self.cur_state != 'intercontext_interval':
            if p < self.params.active_reward_probability:
                if self.params.reward_std_dev > 0:
                    reward = rng.normal(self.params.mean_correct_reward, self.params.reward_std_dev)
                else:
                    reward = self.params.mean_correct_reward
            else:
                reward = 0

        elif not correct and self.cur_state != 'intercontext_interval':
            if p < self.params.inactive_reward_probability:
                if self.params.reward_std_dev > 0:
                    reward = rng.normal(self.params.mean_incorrect_reward, self.params.reward_std_dev)
                else:
                    reward = self.params.mean_incorrect_reward
            else:
                reward = 0

        else:  # self.cur_state == 'intercontext_interval'
            reward = 0
            # if self.params.n_actions == 2:
            #     reward = 0
            # elif correct and self.params.n_actions == 3:
            #     reward = self.params.effort_loss
            # else:
            #     raise AttributeError("The assigned number of actions is not possible")

        # if reward < 0:
        #     reward = 0
        return reward

    def to_next_state(self) -> None:
        logging.info("choosing next block type")
        if self.params.random_block_order:
            next_state = np.random.randint(4)
            self.cur_state = self.states[next_state]

        else:
            self.cur_state = self.params.fixed_block_sequence[self.cur_block]

        self.cur_block_length = self.params.default_block_length
        if self.params.block_length_variation > 0:
            self.cur_block_length += np.random.choice([-1, 1]) * np.random.randint(low=0, high=self.params.block_length_variation)

        logging.info("new block: type {}, length {}".format(self.cur_state, self.cur_block_length))

    def step_markov(self, action) -> Tuple[float, float]:
        # TODO
        # no ICI and markov block transitions
        # ICI and markov block transitions
        pass

    def step_non_markov(self, action) -> Tuple[float, float]:
        """
        Get reward based on action, move to next state.
        Blocks have a fixed length  +/- some variability.
        Can transition after X trials correct by changing the task parameters.

        Actions: 0 right choice, 1 left choice, 2 do nothing???
        """

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

        change_block = (self.cur_trial_in_block >= self.cur_block_length) or \
                (self.correct_in_block >= self.params.success_trials_to_block_transition)

        if change_block:
            logging.info('finished block {} with context {}'.format(self.cur_block, self.cur_state))
            logging.info('num correct in block: {}; num trials in block: {}'.format(self.correct_in_block, self.cur_trial_in_block))
            self.correct_in_block = 0
            self.cur_trial_in_block = 0
            #
            # to_ICI = self.params.length_intercontext_interval > 0 and self.cur_state != 'intercontext_interval'
            # if to_ICI:
            #     self.to_intercontext()
            # else:
            self.cur_block += 1
            if self.cur_block < self.params.n_blocks:
                self.to_next_state()

        return reward, correct

    def step(self, action: int) -> Tuple[float, float]:
        if self.params.markov_block_transitions:
            reward, correct = self.step_markov(action)
        else:
            reward, correct = self.step_non_markov(action)

        return reward, correct

    # def to_intercontext(self):
    #     logging.info("entering intercontext interval")
    #     self.cur_block_length = self.params.length_intercontext_interval
    #     self.cur_state = 'intercontext_interval'
        # self.cur_block += 1