import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy


# Random delivery rewards
# npress = 60
# std_dev = .25
# means = [1,.25]
np.random.seed(1)

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


class TwoLeverAction():

    def __init__(self, block_structure=None, default_blocklength=60, range_blocklength=15):
        self.std_dev = .25
        self.means = [1, -1]
        self.cost = 0
        self.t_block = 0
        self.t_total = 0
        self.cur_state = 'A'
        self.cur_block = 0
        self.cur_blocklength = 60
        self.ITI = False
        self.block_structure = block_structure
        if self.block_structure:
            self.nblocks = len(self.block_structure)
        else:
            self.nblocks = 4
        self.state_list = ['A', 'B', 'C1', 'C2', 'ITI', 'end']
        self.default_blocklength = 60
        self.range_blocklength = 15
        self.len_ITI = 10
        self.random_blocksize = False
        self.nActions = 2
        self.N = np.zeros(self.nActions)
        self.H = np.zeros(self.nActions)
        self.Q = np.zeros(self.nActions)


        self.actions = []
        self.baseline = []
        self.rewards = []
        self.states = []
        self.history = []
        self.count = []
        self.policy = []

    def stimulus(self):
        """Stimulus presentation"""
        if self.cur_state == 'A':
            ix = 0
        elif self.cur_state == 'B':
            ix = 1
        elif self.cur_state in ['C1', 'C2']:
            ix = 2
        elif self.cur_state == 'ITI':
            ix = 3

        else:
            raise NotImplementedError("This block type does not exist!!")

        return ix

    def stepOneLever(self, action):
        """Press lever, get outcome."""
        if self.cur_state == 'A':
            ix = 0
        elif self.cur_state == 'B':
            ix=1
        elif self.cur_state == 'C1':
            ix=0
        elif self.cur_state == 'C2':
            ix=1
        elif self.cur_state == 'ITI':
            ix=2
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
        if self.t_block >= self.cur_blocklength:
            if self.ITI and self.cur_state != 'ITI':
                    # enter ITI
                    self.cur_state = 'ITI'
                    self.cur_blocklength = self.len_ITI
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
                    self.cur_blocklength = self.default_blocklength + np.random.choice([-1, 1]) * np.random.randint(0,                                                                                       self.range_blocklength)
                else:
                    self.cur_blocklength = self.default_blocklength

        return reward

    def stepTwoLever(self, action):
        """Press lever, get outcome."""
        if self.cur_state in ['A', 'C1']:
            ix = 0
        elif self.cur_state in ['B', 'C2']:
            ix = 1
        elif self.cur_state == 'ITI':
            ix = 2
        else:
            raise NotImplementedError("This block type does not exist!!")

        correct=False
        # if action == 0:
        #     reward = -.25
        # else:
        if ix == 0 and action == 0:
            correct = True
        elif ix == 1 and action == 1:
            correct = True

        # if self.cur_state == 'B':
        #     b=1

        if correct:
            reward = np.random.normal(5, self.std_dev) - self.cost
        else:
            reward = np.random.normal(1, self.std_dev) - self.cost

        self.t_block += 1
        self.t_total += 1
        if self.t_block >= self.cur_blocklength:
            if self.ITI and self.cur_state != 'ITI':
                    # enter ITI
                    self.cur_state = 'ITI'
                    self.cur_blocklength = self.len_ITI
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
                    self.cur_blocklength = self.default_blocklength + np.random.choice([-1, 1]) * np.random.randint(0,                                                                                       self.range_blocklength)
                else:
                    self.cur_blocklength = self.default_blocklength

        return reward

    def policy_nonstationary(self, Q, N, alpha=.1, epsilon=.1):
        """
        The world is non-stationary (reward contingencies change over time).
        e-greedy action selection.
        Apply learning updates with magnitude alpha,
        choose random actions with probability epsilon
        """
        if np.random.rand() < epsilon:
            action = np.random.choice(self.nActions)
        else:
            action = np.argmax(Q)
        # reward = self.stepOneLever(action)
        reward = self.stepTwoLever(action)

        N[action] += 1
        Q[action] += alpha * (reward - Q[action])
        return action, reward, Q, N

    def policy_softmax(self, Q, N, H, nonstationary=True, alpha=.1, beta=.1):
        """
        Softmax action selection.
        N - count of times each action is selected
        H - history, the parameters used for softmax distribution
        Q - a baseline parameter updated with rewards. Can be used to speed up learning with a priori knowledge.
        alpha - learning size parameter for gradient updates
        beta - recency bias for non-stationary problems
        """
        Hdist = np.exp(H) / np.sum(np.exp(H))
        # Hdist = np.exp(Q) / np.sum(np.exp(Q))

        action = np.random.choice(self.nActions, p=Hdist)
        # reward = self.stepOneLever(action)
        reward = self.stepTwoLever(action)

        N[action] += 1
        ix = np.arange(self.nActions) != action
        # SOFTMAX DISTRIBUTION UPDATE
        # if self.t_total > 3 and self.cur_state != self.states[-3]:
        #     b=0
        #     pass
        ####
        H[ix] -= alpha * Hdist[ix] * (reward - Q[ix])
        H[action] += alpha * (1 - Hdist[action]) * (reward - Q[action])
        ####
        # H[ix] -= alpha * reward * Hdist[ix]
        # H[action] += alpha * reward * (1-Hdist[action])

        if nonstationary:
            Q[action] += beta * (reward - Q[action])  # nonstationary update, biased towards recent rewards
        else:
            Q[action] += (reward - Q[action]) / N[action]  # incremental average reward for stationary problems

        return action, reward, Q, N, H, Hdist

    def run_experiment(self, alpha=.1):
        # self.cur_state = self.state_list[np.random.randint(4)]
        self.cur_state = self.state_list[0]
        self.cur_block = 0
        if self.random_blocksize:
            self.cur_blocklength = self.default_blocklength + np.random.choice([-1,1]) * np.random.randint(0,self.range_blocklength)
        else:
            self.cur_blocklength = self.default_blocklength

        N = np.zeros(self.nActions)
        H = np.zeros(self.nActions)
        Q = np.zeros(self.nActions)
        Hdist= 0
        while self.cur_block < self.nblocks:
            stim = self.stimulus()
            action, reward, Q, N = self.policy_nonstationary(Q, N, alpha=alpha)
            # action, reward, Q, N, H, Hdist = self.policy_softmax(Q, N, H)

            self.actions.append(action)
            self.baseline.append(copy.deepcopy(Q))
            self.history.append(copy.deepcopy(H))
            self.policy.append(copy.deepcopy(Hdist))
            self.rewards.append(reward)
            self.states.append(self.cur_state)
            self.count.append(copy.deepcopy(N))

    def save(self):
        self.cur_state = 'A'
        self.cur_block = 0
        self.cur_blocklength = 60
        pass

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n










