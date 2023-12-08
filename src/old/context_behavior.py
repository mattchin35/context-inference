import numpy as np
import scipy as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import copy
import os
import pickle as pkl


np.random.seed(0)

class TwoLeverAction():

    def __init__(self, block_structure=None, default_blocklength=60, range_blocklength=15, p_switch=None, Qalpha=.1,
                 logistic_params=np.array([1,2,1.5])):
        self.std_dev = 1
        self.means = [3, 2]  # correct reward center, incorrect reward center

        self.t_block = 0
        self.t_total = 0

        self.cur_block = 0
        self.cur_blocklength = default_blocklength
        self.block_structure = block_structure
        if block_structure is None:
            self.block_structure = ['A', 'B', 'C1', 'C2']

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

        self.Qalpha = Qalpha
        self.logistic_params = logistic_params  # alpha, beta, tau
        if p_switch is None:
            # p_switch = 1 / default_blocklength
            p_switch = .02
        self.p_switch = p_switch
        # from-to, ABC1C2

    def set_transition_matrix(self):
        p = self.p_switch
        self.transition_matrix = np.array([[1 - 3 * p, p, p, p],
                                           [p, 1 - 3 * p, p, p],
                                           [p, p, 1 - 3 * p, p],
                                           [p, p, p, 1 - 3 * p]])

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

    def stateless_Qlearning(self, Q, N, greedy=False, epsilon=.1):
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
        Q[action] += self.Qalpha * (reward - Q[action])  # nonstationary update, biased towards recent rewards
        return action, reward, Q, N, dist

    def state_Qlearning(self, Q, N, greedy=False, epsilon=.1):
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
        Q[ix, action] += self.Qalpha * (reward - Q[ix, action])
        return action, reward, Q, N, dist

    def observation_Qlearning(self, Q, N, alpha=.1, greedy=False, epsilon=.1):
        """
        Q-learning action selection for a model with incomplete state information.
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
        Q[ix, action] += self.Qalpha * (reward - Q[ix, action])
        return action, reward, Q, N, dist

    def POMDP(self):
        """
        SHEEEESH
        """
        pass

    def forgetting_Qlearning(self, Q, N, sticky=0, tau=5, temp=1):
        """
        Qlearning but with a decay for prior Q values.
        Parameters determinine:
        stickiness (repetition of prior choices)
        temperature (stochasticity of choice)
        decxay (higher tau
        """
        if self.t_total == 0:
            prev = 0
        else:
            prev = 1 - self.actions[-1] * 2  # maps left (0) to 1 and right (1) to -1

        dQ = Q[0] - Q[1]
        psi = (sticky * prev + dQ) / temp
        logit = sp.special.expit(psi)
        dist = [logit, 1-logit]

        action = np.random.choice(self.nActions, p=dist)
        reward = self.stepTwoLever(action)

        decay = np.exp(-1 / tau)
        Q[action] = decay * Q[action] + (1-decay) * reward
        Q[1-action] = decay * Q[1-action]
        N[action] += 1

        return action, reward, Q, N, dist

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

    def HMM(self, prior, temperature=1, greedy=False, epsilon=.1, sticky=0):
        """
        A hidden Markov model with Thompson-sampling action selection. Essentially the forward piece of the HMM algorithm.
        Temperature: a parameter for determinism; higher temperature is less deterministic/more stochastic
        sticky: parameter for "stickiness," bias to previous action
        """
        # Thompson sampling of an action. Get belief in left-optimal vs right-optimal state
        # If desired, adjust with temperature parameter
        if self.t_total == 0:
            prev = 0
        else:
            # prev = self.actions[-1] * 2 - 1
            prev = 1 - self.actions[-1] * 2

        dist = np.array([prior[0] + prior[2], prior[1] + prior[3]])
        pL = sp.special.expit((sp.special.logit(dist[0]) + prev * sticky) / temperature)  # temp = 1 means softmax, temp -> 0 makes greedy
        pR = 1-pL
        if greedy:
            if np.random.rand() < epsilon:
                action = np.random.choice(self.nActions)
            else:
                action = np.argmax([pL, pR])
        else:
            # print(dist)
            action = np.random.choice(self.nActions, p=[pL, pR])

        reward = self.stepTwoLever(action)

        # update beliefs based on action outcome
        pOutcome = self.outcome_probability(action, reward)
        # condBeliefs = pOutcome * prior / np.dot(pOutcome, prior)
        # posterior = np.dot(self.transition_matrix.T, condBeliefs)
        # posterior /= np.sum(posterior)

        posterior = pOutcome * np.dot(self.transition_matrix.T, prior)
        posterior /= np.sum(posterior)
        return action, reward, posterior, dist

    def logistic(self, dist, phi):
        """
        Logistic regression (RFLR) model based on Beron et al, PNAS 2022. Switched 0 and 1 from their convention for
        consistency with the other code here. Note that this option only agents actions, not beliefs.
        prior - prior log odds
        phi - previous recursion for choice-reward
        beta - weight for phi recursion
        alpha - weight for previous choice
        """
        print(dist)
        action = np.random.choice(self.nActions, p=dist)  # 0 left, 1 right
        reward = self.stepTwoLever(action)
        alpha, beta, tau = self.logistic_params

        cbar = 2 * action - 1  # -1 left, 1 right
        phi_t = beta * cbar * reward + np.exp(1/tau) * phi
        log_posterior = alpha * cbar + phi_t
        p = sp.special.expit(log_posterior)
        posterior = np.array([p, 1-p])
        return action, reward, posterior, phi_t

    def policy_TDlearning(self, Q, N, S, epsilon=.1):
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
        Q[S, action] += self.Qalpha * (reward - Q[S, action])  # In a bandit task, every action takes you to the terminal state; no need for Q[S',A']
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

    def run_experiment(self, save_name,  model_name='stateless'):
        # self.cur_state = self.state_list[np.random.randint(4)]
        self.cur_state = self.state_list[0]
        self.cur_block = 0
        if self.random_blocksize:
            self.cur_blocklength = self.default_blocklength + np.random.choice([-1,1]) * np.random.randint(0, self.range_blocklength)
        else:
            self.cur_blocklength = self.default_blocklength

        assert model_name in ['stateless', 'state_Q', 'observation_Q', 'POMDP', 'HMM', 'logistic', 'belief_RNN',
                              'forgetting_Q'], 'Model type does not exist!!!'
        self.set_transition_matrix()
        if model_name == 'stateless' or model_name == 'forgetting_Q':
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
                action, reward, Q, N, dist = self.stateless_Qlearning(Q, N)
            elif model_name == 'state_Q':
                action, reward, Q, N, dist = self.state_Qlearning(Q, N)
            elif model_name == 'observation_Q':
                action, reward, Q, N, dist = self.observation_Qlearning(Q, N)
            elif model_name == 'forgetting_Q':
                action, reward, Q, N, dist = self.forgetting_Qlearning(Q, N)
            elif model_name == 'POMDP':
                # self.HMM(B)
                pass
            elif model_name == 'HMM':
                action, reward, prior, dist = self.HMM(prior)
            elif model_name == 'logistic':
                # action, reward, dist, phi = self.logistic(dist, phi, alpha=.1, beta=.3, tau=10)  # closest I got so far to something decent
                action, reward, dist, phi = self.logistic(dist, phi, alpha=self.alpha, beta=.5, tau=10)
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










