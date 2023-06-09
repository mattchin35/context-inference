import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import os
import pickle as pkl


np.random.seed(0)

class TwoLeverAction():

    def __init__(self, block_structure=None, default_blocklength=60, range_blocklength=15, p_switch=None):
        self.std_dev = 1
        self.means = [3, 2]  # correct reward center, incorrect reward center

        self.t_block = 0
        self.t_total = 0

        self.cur_block = 0
        self.cur_blocklength = default_blocklength
        self.block_structure = block_structure
        if self.block_structure:
            self.nblocks = len(self.block_structure)
        else:
            self.nblocks = 4

        self.cur_state = 'A'
        self.state_list = ['A', 'B', 'C1', 'C2', 'ITI', 'end']
        self.default_blocklength = default_blocklength
        self.range_blocklength = 15
        self.random_blocksize = False

        self.len_ITI = 10
        self.ITI = False

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
        self.belief = []
        self.phi = []

        if p_switch is None:
            p_switch = 1 / default_blocklength
        # from-to, ABC1C2
        self.transition_matrix = np.array([[1 - 3 * p_switch, p_switch, p_switch, p_switch],
                                      [p_switch, 1 - 3 * p_switch, p_switch, p_switch],
                                      [p_switch, p_switch, 1 - 3 * p_switch, p_switch],
                                      [p_switch, p_switch, p_switch, 1 - 3 * p_switch]])

    def get_state_ix(self):
        if self.cur_state == 'A':
            ix = 0
        elif self.cur_state == 'B':
            ix = 1
        elif self.cur_state == 'C1':
            ix = 2
        elif self.cur_state == 'C2':
            ix = 3

        return ix

    def get_stimulus(self):
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

        correct = False
        # if action == 0:
        #     reward = -.25
        # else:
        if ix == 0 and action == 0:
            correct = True
        elif ix == 1 and action == 1:
            correct = True

        if correct:
            reward = np.random.normal(self.means[0], self.std_dev)
        else:
            reward = np.random.normal(self.means[1], self.std_dev)

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
                    self.cur_blocklength = self.default_blocklength + np.random.choice([-1, 1]) * np.random.randint(0,self.range_blocklength)
                else:
                    self.cur_blocklength = self.default_blocklength

        return reward

    def stateless_Qlearning(self, Q, N, alpha=.1, greedy=False, epsilon=.1):
        """
        Q-learning action selection.
        N - count of times each action is selected
        H - history, the parameters used for softmax distribution
        Q - a baseline parameter updated with rewards. Can be used to speed up learning with a priori knowledge.
        alpha - learning size parameter for gradient updates
        beta - recency bias for non-stationary problems
        """
        if greedy:
            if np.random.rand() < epsilon:
                action = np.random.choice(self.nActions)
            else:
                action = np.argmax(Q)
        else:
            dist = sp.special.softmax(Q)

            # logit = 1. / (1 + np.exp(-(Q[0] - Q[1])))
            # dist = [logit, 1-logit]

            action = np.random.choice(self.nActions, p=dist)

        reward = self.stepTwoLever(action)

        N[action] += 1
        Q[action] += alpha * (reward - Q[action])  # nonstationary update, biased towards recent rewards
        return action, reward, Q, N, dist

    def state_Qlearning(self, Q, N, alpha=.1, greedy=False, epsilon=.1):
        """
        Q-learning action selection for a model with complete state information.
        N - count of times each action is selected within a state
        Q - action-values updated with rewards. Can be used to speed up learning with a priori knowledge.
        alpha - learning size parameter Q updates
        """
        ix = self.get_state_ix()
        if greedy:
            if np.random.rand() < epsilon:
                action = np.random.choice(self.nActions)
            else:
                action = np.argmax(Q[ix])
        else:
            dist = sp.special.softmax(Q[ix])
            action = np.random.choice(self.nActions, p=dist)

        reward = self.stepTwoLever(action)
        N[ix, action] += 1
        Q[ix, action] += alpha * (reward - Q[ix, action])
        return action, reward, Q, N, dist

    def observation_Qlearning(self, Q, N, alpha=.1, greedy=False, epsilon=.1):
        """
        Q-learning action selection for a model with complete state information.
        N - the state is now being inferred; N is not used here.
        Q - action-values updated with rewards. Can be used to speed up learning with a priori knowledge.
        alpha - learning size parameter Q updates
        """
        ix = self.get_stimulus()
        if greedy:
            if np.random.rand() < epsilon:
                action = np.random.choice(self.nActions)
            else:
                action = np.argmax(Q[ix])  # both C1 and C2 will be loaded into the C1 index; C2 will have empty Q
        else:
            dist = sp.special.softmax(Q[ix])
            action = np.random.choice(self.nActions, p=dist)

        reward = self.stepTwoLever(action)
        N[ix, action] += 1

        # nonstationary Q update, biased towards recent rewards
        Q[ix, action] += alpha * (reward - Q[ix, action])
        return action, reward, Q, N, dist

    def POMDP(self):
        """
        SHEEEESH
        """
        pass

    def outcome_probability(self, action, reward):
        p = np.zeros(4)
        stim_ix = self.get_stimulus()
        if action == 0:
            p[0] = norm.pdf(reward, self.means[0], self.std_dev)
            p[1] = norm.pdf(reward, self.means[1], self.std_dev)
            p[2] = norm.pdf(reward, self.means[0], self.std_dev)
            p[3] = norm.pdf(reward, self.means[1], self.std_dev)
        elif action == 1:
            p[0] = norm.pdf(reward, self.means[1], self.std_dev)
            p[1] = norm.pdf(reward, self.means[0], self.std_dev)
            p[2] = norm.pdf(reward, self.means[1], self.std_dev)
            p[3] = norm.pdf(reward, self.means[0], self.std_dev)

        # adjust for blocks and stimuli
        if stim_ix == 0:
            p[1:] *= 0
        elif stim_ix == 1:
            p *= np.array([0,1,0,0])
        else:  # ix > 1:
            p *= np.array([0,0,1,1])

        return p

    def HMM(self, prior, temperature=1, greedy=False, epsilon=.1):
        """
        A hidden Markov model with Thompson-sampling action selection. Essentially the forward piece of the HMM algorithm.
        """
        # Thompson sampling of an action
        _dist = np.array([prior[0] + prior[2], prior[1] + prior[3]])
        dist = sp.special.softmax(_dist)
        # dist = sp.special.expit(sp.special.logit(_dist) / temperature) # temp = 1 means softmax, temp -> 0 makes greedy
        if greedy:
            if np.random.rand() < epsilon:
                action = np.random.choice(self.nActions)
            else:
                action = np.argmax(dist)
        else:
            # print(dist)
            action = np.random.choice(self.nActions, p=dist)

        reward = self.stepTwoLever(action)

        # update beliefs based on action outcome
        pOutcome = self.outcome_probability(action, reward)
        # condBeliefs = pOutcome * prior / np.dot(pOutcome, prior)
        # posterior = np.dot(self.transition_matrix.T, condBeliefs)
        # posterior /= np.sum(posterior)

        posterior = pOutcome * np.dot(self.transition_matrix.T, prior)
        posterior /= np.sum(posterior)

        return action, reward, posterior, dist

    def logistic(self, dist, phi, alpha=.1, beta=.1, tau=1):
        """
        Logistic regression model based on Beron et al, PNAS 2022. Switched 0 and 1 from their convention for
        consistency with the other code here. Note that this option only models actions, not beliefs.
        prior - prior log odds
        phi - previous recursion for choice-reward
        beta - weight for phi recursion
        alpha - weight for previous choice
        """
        print(dist)
        action = np.random.choice(self.nActions, p=dist)  # 0 left, 1 right
        reward = self.stepTwoLever(action)

        cbar = 2 * action - 1  # -1 left, 1 right
        phi_t = beta * cbar * reward + np.exp(1/tau) * phi
        log_posterior = alpha * cbar + phi_t
        p = sp.special.expit(log_posterior)
        posterior = np.array([p, 1-p])
        return action, reward, posterior, phi_t

    def policy_TDlearning(self, Q, N, S, alpha=.1, epsilon=.1):
        """
        e-greedy TD learning algorithm.
        Commentary: This is basically your nonstationary bandit with state information. Q-learning and Sarsa are the same for bandit tasks, in which episodes are a single-action long and
        take you to the terminal state. For Q-learning, the action could be selected from a behavior other than the goal policy.
        """
        if np.random.rand() < epsilon:
            action = np.random.choice(self.nActions)
        else:
            action = np.argmax(Q[S])

        reward = self.stepTwoLever(action)
        N[action] += 1
        Q[S, action] += alpha * (reward - Q[S, action])  # In a bandit task, every action takes you to the terminal state; no need for Q[S',A']
        return action, reward, Q, N

    def policy_fxnApprox(self):
        """
        Do this for the partially observable case. Use an RNN to do fxn approximation feeding in the stimulus set.
        """
        pass

    def policy_gradient(self):
        """
        Dunno if I'll get anything from this, but it'll be good for learning.
        """

        #
        # N[action] += 1
        # ix = np.arange(self.nActions) != action
        #
        # H[ix] -= alpha * Hdist[ix] * (reward - Q[ix])
        # H[action] += alpha * (1 - Hdist[action]) * (reward - Q[action])
        # ####
        # # H[ix] -= alpha * reward * Hdist[ix]
        # # H[action] += alpha * reward * (1-Hdist[action])
        #
        # Q[action] += beta * (reward - Q[action])  # nonstationary update, biased towards recent rewards
        pass

    def run_experiment(self, save_name, alpha=.1, model_name='stateless'):
        # self.cur_state = self.state_list[np.random.randint(4)]
        self.cur_state = self.state_list[0]
        self.cur_block = 0
        if self.random_blocksize:
            self.cur_blocklength = self.default_blocklength + np.random.choice([-1,1]) * np.random.randint(0, self.range_blocklength)
        else:
            self.cur_blocklength = self.default_blocklength

        assert model_name in ['stateless', 'state_Q', 'observation_Q', 'POMDP', 'HMM', 'logistic', 'belief_RNN'], 'Model type does not exist!!!'
        if model_name == 'stateless':
            Q = np.zeros(self.nActions)
            N = np.zeros(self.nActions)
        else:
            # Complete state information available
            Q = np.zeros((4,self.nActions))
            N = np.zeros((4,self.nActions))

        prior = np.ones(4)/4
        phi = 0
        dist = np.array([.5,.5])

        # Incomplete state information available
        while self.cur_block < self.nblocks:
            if model_name == 'stateless':
                action, reward, Q, N, dist = self.stateless_Qlearning(Q, N, alpha=alpha)
            elif model_name == 'state_Q':
                action, reward, Q, N, dist = self.state_Qlearning(Q, N, alpha=alpha)
            elif model_name == 'observation_Q':
                action, reward, Q, N, dist = self.observation_Qlearning(Q, N, alpha=alpha)
            elif model_name == 'POMDP':
                # self.HMM(B)
                pass
            elif model_name == 'HMM':
                action, reward, prior, dist = self.HMM(prior)
            elif model_name == 'logistic':
                # action, reward, dist, phi = self.logistic(dist, phi, alpha=.1, beta=.3, tau=10)  # closest I got so far to something decent
                action, reward, dist, phi = self.logistic(dist, phi, alpha=0, beta=.5, tau=10)
                # action, reward, dist, phi = self.logistic(dist, phi, alpha=1, beta=2, tau=1.5)  # in paper parameters
            elif model_name == 'belief_RNN':
                pass  #TODO

            # S = self.get_state_ix(self.cur_state)
            # action, reward, Q, N = self.policy_TDlearning(Q, N, S)

            self.actions.append(action)
            self.baseline.append(copy.deepcopy(Q))
            # self.history.append(copy.deepcopy(H))
            self.policy.append(copy.deepcopy(dist))
            self.rewards.append(reward)
            self.states.append(self.cur_state)
            self.count.append(copy.deepcopy(N))
            self.belief.append(copy.deepcopy(prior))
            self.phi.append(phi)


        self.save(save_name)

    def save(self, fname):
        self.cur_state = 'A'
        self.cur_block = 0
        self.cur_blocklength = 60

        save_dict = vars(self)
        save_path = os.path.join('..','saved_models', fname + '.pkl')
        with open(save_path, 'wb') as f:
            pkl.dump(save_dict, f)

        print("[***] Model saved as: {}".format(save_path))

    def load(self, fname):
        save_path = os.path.join('..', 'saved_models', fname + '.pkl')
        cur_dict = self.__dict__
        with open(save_path, 'rb') as f:
            load_dict = pkl.load(f)

        for k, v in load_dict.items():
            cur_dict[k] = v

        print("[***] Model restored from path: %s" % save_path)

    def reset(self):
        pass

    @staticmethod
    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n










