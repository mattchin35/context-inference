# class OneLeverAuto:
#
#     def __int__(self):
#         self.npress = 60
#         self.std_dev = .5
#         self.means = [1, .25]
#
#     def create_block(self, blocktype):
#         assert blocktype in ['A', 'B', 'C1', 'C2']
#         if blocktype == 'A':
#             rewards = np.random.normal(self.means[0], std_dev, npress)
#             stimuli = np.zeros(npress)
#
#         elif blocktype == 'B':
#             rewards = np.random.normal(self.means[1], std_dev, npress)
#             stimuli = np.ones(npress)
#
#         elif blocktype == 'C1':
#             rewards = np.random.normal(self.means[0], std_dev, npress)
#             stimuli = np.ones(npress) * 2
#
#         elif blocktype == 'C2':
#             rewards = np.random.normal(self.means[1], std_dev, npress)
#             stimuli = np.ones(npress) * 2
#
#         else:
#             raise NotImplementedError("This block type does not exist!!")
#
#         rewards[rewards<0] = 0
#         outcome = np.stack((rewards, stimuli), axis=1)
#         return outcome
#
#     def block_probabilities(self, outcome, std=std_dev, include_stimulus=True):
#         pA = norm.pdf(outcome[:,0], self.means[0], std)
#         pB = norm.pdf(outcome[:,0], self.means[1], std)
#         pC1 = norm.pdf(outcome[:,0], self.means[0], std)
#         pC2 = norm.pdf(outcome[:,0], self.means[1], std)
#
#         if include_stimulus:
#             pA[outcome[:, 1] != 0] = 0
#             pB[outcome[:, 1] != 1] = 0
#             pC1[outcome[:, 1] != 2] = 0
#             pC2[outcome[:, 1] != 2] = 0
#
#         return np.stack((pA, pB, pC1, pC2), axis=1)


def stepOneLever(self, action):
    """One action with state-dependent outcome."""
    if self.cur_state == 'A':
        ix = 0
    elif self.cur_state == 'B':
        ix = 1
    elif self.cur_state == 'C1':
        ix = 0
    elif self.cur_state == 'C2':
        ix = 1
    elif self.cur_state == 'ITI':
        ix = 2
    else:
        raise NotImplementedError("This block type does not exist!!")

    if action == 0:
        reward = -.25
    else:
        if ix < 2:
            reward = np.random.normal(self.means[ix], self.std_dev) - self.cost
        else:
            reward = -self.cost

    self.t_block += 1
    if self.t_block >= self.cur_block_length:
        if self.ITI and self.cur_state != 'ITI':
            # enter ITI
            self.cur_state = 'ITI'
            self.cur_block_length = self.len_ITI
            self.cur_block += 1

        else:
            print(self.t_block)
            print('finished block {} with context {}'.format(self.cur_block, self.cur_state))
            self.cur_block += 1
            if self.cur_block >= self.nblocks:
                return reward

            self.t_block = 0
            if self.block_structure:
                self.cur_state = self.block_structure[self.cur_block]
            else:
                nextState = np.random.randint(4)
                self.cur_state = self.state_list[nextState]

            if self.random_blocksize:
                self.cur_block_length = self.default_block_length + np.random.choice([-1, 1]) * np.random.randint(0,
                                                                                                                  self.range_blocklength)
            else:
                self.cur_block_length = self.default_block_length

    return reward



    # for transition in transition_list:
    #     # if transition in session.keys():
    #     ix = session_df['Event'] == transition
    #     timestamps = session_df['Time'][ix]
    #     bins.extend(timestamps)  # collect the timestamps for every transition into one list
    #     bin_type.extend([transition] * len(timestamps))
    #
    # bins, bin_type = np.array(bins), np.array(bin_type)
    # ix = np.argsort(bins)  # make sure the bins are in ascending order
    # bins = bins[ix]
    # bins = np.round(bins, decimals=3)
    # bin_type = bin_type[ix][:-1]  # exclude the last bin - it's an ending intercontext interval
    # bin_centers = (bins[:-1] + bins[1:])/2
    #
    # # Iterate over the output list
    # # bin_dict = dict(bin_type=bin_type)#, bin_bounds=np.around(intervals, decimals=3))
    # bin_dict = dict()#, bin_bounds=np.around(intervals, decimals=3))
    # for output in output_list:
    #     if output in session.keys():
    #         # Bin the 'Time' values of the output
    #         # session[output]['Bin'] = pd.cut(session[output]['Time'], bins, include_lowest=True)
    #         session[output]['Bin'] = pd.cut(session[output]['Time'], bins)
    #
    #         # Count the number of 'Time' values in each bin
    #         counts = session[output].groupby('Bin').count()
    #         bin_dict[output] = counts['Time'].values

# Rreward_ix = session_df['pump1_reward_3']['Bin'] == iv
# Rreward = session_df['pump1_reward_3']['Time'][Rreward_ix].values
#
# Rfail_ix = session_df['pump1_reward_0']['Bin'] == iv
# Rfail = session_df['pump1_reward_0']['Time'][Rfail_ix].values
#
# Lreward_ix = session_df['pump2_reward_3']['Bin'] == iv
# Lreward = session_df['pump2_reward_3']['Time'][Lreward_ix].values
#
# Lfail_ix = session_df['pump2_reward_0']['Bin'] == iv
# Lfail = session_df['pump2_reward_0']['Time'][Lfail_ix].values
#
# Rall = np.sort(np.concatenate([Rreward, Rfail]))
# Lall = np.sort(np.concatenate([Lreward, Lfail]))
# Tix = np.argsort(np.concatenate([Rall, Lall]))
# T = np.concatenate([Rall, Lall])[Tix]

# all_rewards = (blocks['pump1_reward_3'] + blocks['pump2_reward_3'] + blocks['pump1_reward_0'] + blocks['pump1_reward_0']).values
    # active_ix = all_rewards > 0
    # Lreward = np.zeros_like(all_rewards)
    # Lreward[active_ix] = blocks['pump1_reward_3'].values[active_ix] / all_rewards[active_ix]
    # Rreward = np.zeros_like(all_rewards)
    # Rreward[active_ix] = blocks['pump2_reward_3'].values[active_ix] / all_rewards[active_ix]


    # event_labels = ['Left lick', 'Right lick',
    #                 'left reward 0', 'left reward 3',
    #                 'right reward 0', 'right reward 3',
    #                 'enter intercontext interval']
    #
    # if 'enter_ContextA' and 'enter_ContextC1' in unique_events:  # full ABC1C2 task
    #     session_type = 'full'
    #     event_list += ['enter_ContextA', 'enter_ContextB', 'enter_ContextC1', 'enter_ContextC2']
    #     event_labels += ['enter Context A', 'enter Context B', 'enter Context C1', 'enter Context C2']
    # elif 'enter_ContextA' in events:
    #     session_type = 'AB'
    #     event_list += ['enter_ContextA', 'enter_ContextB']
    #     event_labels += ['enter Context A', 'enter Context B']
    # else:  # 'enter_ContextC1' in events:
    #     session_type = 'C1C2'
    #     event_list += ['enter_ContextC1', 'enter_ContextC2']
    #     event_labels += ['enter Context C1', 'enter Context C2']


# def plot_multiple_experiments(data: []):
#     y = np.array(experiment1.actions)
#     yA = []
#     for a in y:
#         if a == 0:
#             yA.append('Left')
#         else:
#             yA.append('Right')
#
#     y2 = np.array(experiment2.actions)
#     yB = []
#     for b in y2:
#         if b == 0:
#             yB.append('Left')
#         else:
#             yB.append('Right')
#
#     x = np.arange(len(experiment1.actions))
#     dHMM = {'action': yA, 'trial': x, 'model_type': 'HMM'}
#     dfHMM = pd.DataFrame(data=dHMM)
#     dQ = {'action': yB, 'trial': x, 'model_type': 'Q-learning'}
#     dfQ = pd.DataFrame(data=dQ)
#     df = pd.concat([dfHMM, dfQ])
#
#     ### PLOT MODEL ACTIONS
#     f0 = plt.figure()
#     # plt.scatter(x, y, label='fast learning', s=4)
#     # sns.swarmplot(df, x='trial', y='action', hue='model_type',s=4)  # no dodge plotting
#     sns.swarmplot(df, x='trial', y='action', hue='model_type', s=4, dodge=True)
#
#     # FIGURE ADJUSTMENTS
#     # plt.yticks([0,1],['Left', 'Right'])
#     # plt.ylim([-.1,1.1])
#     plt.ylabel('Actions', fontsize=12)
#     plt.xticks(np.array([0,1,2,3,4]) * block_length)
#     plt.xlim([0,block_length*nBlocks])
#     plt.xlabel('Trial', fontsize=12)
#     plt.title('HMM vs Q-learning model behavior', fontsize=14)
#     ax = plt.gca()
#     ax.legend(fancybox=False)#, loc='lower right')
#
#     ### COLOR BLOCKS
#     colors = ['cyan', 'darkgreen', 'plum', 'indigo']
#     for i, c in enumerate(colors):
#         plt.axvspan(block_length*i, block_length * (i+1), color=c, alpha=.2)
#
#     format = 'png'
#     f0.tight_layout()
#     plt.show()
#     plt.savefig('../figures/HMM-vs-Q_actions.' + format, format=format, dpi=300)


class HMM_4state(BehaviorAgent):
    def __init__(self, params: MDP.TaskParams, temperature: float = 1, greedy: bool = False, epsilon: float = .1, stickiness: float = 0,
                 n_states: int = 4, transition_prob: float = .05):
        """
        Hidden Markov Model with Thompson sampling and a stickiness parameter. Requires full knowledge of the task
        structure to be an ideal observer. Consider updating this to allow custom parameters to test this on.

        stickiness ranges from (-inf, +inf)
        epsilon is for optional greedy action sampling
        temperature controls randomness, so that temp = 1 means an unchanged softmax over state probabilities,
            temp -> 0 makes a greedier/deterministic algorithm, and temp -> inf increases randomness to 50/50
        """
        assert temperature > 0, "must have positive temperature"

        self.temperature = temperature
        self.greedy = greedy
        self.epsilon = epsilon
        self.stickiness = stickiness
        self.transition_prob = transition_prob
        self.params = params
        self.model_type = 'HMM'

        self.n_states = n_states
        self.prior = np.ones(n_states) / n_states
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS

        self.last_action = 0
        self.last_stimulus = 0

    def choose_action(self, stimulus: int) -> int:
        if self.greedy:
            if rng.random() < self.epsilon:
                action = rng.choice(N_ACTIONS)
            else:
                action = np.argmax(self.action_dist)
        else:
            self.update_observation_posterior(stimulus)
            self.update_action_dist(self.last_action)

            logging.info("action dist: {}".format(self.action_dist))
            action = rng.choice(N_ACTIONS, p=self.action_dist)

        self.last_action = action
        self.last_stimulus = stimulus
        return action

    def update_params(self, action, reward) -> None:
        """
        Update the prior based on the action/reward outcome.
        """

        # right=0, left=1: map right to -1, left to +1
        # p_outcome = self.outcome_probability(action, reward, self.last_stimulus)

        p_observation = self.observation_probability(self.last_stimulus)
        # p_reward_delivery = self.reward_delivery_probability(reward=reward, action=action)
        # p_reward_size = self.reward_size_probability(reward=reward, action=action)
        # p_outcome = p_reward_delivery * p_reward_size * p_observation

        p_reward_delivery = self.reward_probability(reward=reward, action=action)
        p_outcome = p_reward_delivery * p_observation

        # posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)
        posterior = p_reward_delivery * self.prior  # transition matrix from the observation step to the reward step is identity - no state change in this interval
        # posterior = p_outcome * self.prior  # transition matrix from the observation step to the reward step is identity - no state change in this interval
        posterior /= np.sum(posterior)
        logging.info("reward -> updated posterior: {}".format(posterior))
        self.prior = posterior

        # action_ix = action * 2 - 1
        # dist = np.array([self.prior[0] + self.prior[2], self.prior[1] + self.prior[3]])
        # pR = sp.special.expit(
        #     (sp.special.logit(dist[0]) + action_ix * self.stickiness) / self.temperature)
        # pL = 1 - pR
        # self.action_dist = np.array([pR, pL])

    def update_action_dist(self, action: int):
        action_ix = action * 2 - 1
        dist = np.array([self.prior[0] + self.prior[2], self.prior[1] + self.prior[3]])
        pR = sp.special.expit(
            (sp.special.logit(dist[0]) + action_ix * self.stickiness) / self.temperature)
        pL = 1 - pR
        self.action_dist = np.array([pR, pL])

    # def outcome_probability(self, action: int, reward: float, stimulus: int) -> np.ndarray:
    #     # need the task parameters in this one
    #     p = np.zeros(4)
    #     if self.params.reward_std_dev > 0:
    #         if action == 0:  # right choice - find probability of the reward under each context
    #             p[0] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)
    #             p[1] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)
    #             p[2] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)
    #             p[3] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)
    #
    #         elif action == 1:  # left choice
    #             p[0] = norm.pdf(reward, self.params.mean_incorrect_reward, self.task.params.reward_std_dev)
    #             p[1] = norm.pdf(reward, self.params.mean_correct_reward, self.task.params.reward_std_dev)
    #             p[2] = norm.pdf(reward, self.params.mean_incorrect_reward, self.task.params.reward_std_dev)
    #             p[3] = norm.pdf(reward, self.params.mean_correct_reward, self.task.params.reward_std_dev)
    #
    #     else:  # potentially worth ignoring this case entirely
    #         if action == 0:  # right choice - find probability of the reward under each context
    #             p[0] = reward == self.task.params.active_reward_probability
    #             p[1] = reward == self.task.params.inactive_reward_probability
    #             p[2] = reward == self.task.params.active_reward_probability
    #             p[3] = reward == self.task.params.inactive_reward_probability
    #
    #         elif action == 1:  # left choice
    #             p[0] = reward == self.task.params.inactive_reward_probability
    #             p[1] = reward == self.task.params.active_reward_probability
    #             p[2] = reward == self.task.params.inactive_reward_probability
    #             p[3] = reward == self.task.params.active_reward_probability
    #
    #         p = p.astype(float)
    #
    #     p *= self.observation_probability(stimulus)
    #     return p

    def observation_probability(self, stimulus: int) -> np.ndarray:
        if stimulus == 0:  # i.e. right cued context
            p = np.array([1,0,0,0])
        elif stimulus == 1:  # i.e. left cued context
            p = np.array([0,1,0,0])
        else:  # ix > 1, either of the uncued contexts
            p = np.array([0,0,1,1])
        return p

    def nonzero_reward_size_probability(self, reward, action):
        p = np.zeros(4)
        if self.params.reward_std_dev > 0:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)
                p[2] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)
                p[3] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)

            elif action == 1:  # left choice
                p[0] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)
                p[2] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)
                p[3] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)

        else:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = reward == self.params.mean_correct_reward
                p[1] = reward == self.params.mean_incorrect_reward
                p[2] = reward == self.params.mean_correct_reward
                p[3] = reward == self.params.mean_incorrect_reward

            elif action == 1:  # left choice
                p[0] = reward == self.params.mean_incorrect_reward
                p[1] = reward == self.params.mean_correct_reward
                p[2] = reward == self.params.mean_incorrect_reward
                p[3] = reward == self.params.mean_correct_reward

            p = p.astype(float)
        return p

    def reward_probability(self, reward: float, action: int) -> np.ndarray:
        nonzero_reward_indicator = int(reward > 0)

        p_reward_delivery = self.reward_delivery_probability(action=action)
        p_reward_size = self.nonzero_reward_size_probability(reward=reward, action=action)
        p_no_reward = 1 - self.reward_delivery_probability(action=action)

        p_reward = nonzero_reward_indicator * p_reward_delivery * p_reward_size + \
                   (1-nonzero_reward_indicator) * p_no_reward
        return p_reward

    def update_observation_posterior(self, stimulus: int):
        p_outcome = self.observation_probability(stimulus)
        posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)  # todo - .T not be needed, it's symmetric...
        posterior /= np.sum(posterior)
        logging.info("observation -> updated posterior: {}".format(posterior))
        self.prior = posterior

    def reward_delivery_probability(self, action: int):
        p = np.zeros(4)
        if action == 0:  # right choice - find probability of the right reward under each context
            p[0] = self.params.active_reward_probability
            p[1] = self.params.inactive_reward_probability
            p[2] = self.params.active_reward_probability
            p[3] = self.params.inactive_reward_probability

        elif action == 1:  # left choice - find probability of the left reward under each context
            p[0] = self.params.inactive_reward_probability
            p[1] = self.params.active_reward_probability
            p[2] = self.params.inactive_reward_probability
            p[3] = self.params.active_reward_probability

        return p

    @property
    def transition_matrix(self) -> np.ndarray:
        state_transition_matrix = np.ones((self.n_states, self.n_states)) * self.transition_prob
        for i in range(self.n_states):
            state_transition_matrix[i, i] = 1 - (self.n_states - 1) * self.transition_prob

        return state_transition_matrix