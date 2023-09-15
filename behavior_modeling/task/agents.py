import numpy as np
import scipy as sp
from scipy.stats import norm
import os
import pickle as pkl

from dataclasses import dataclass
from typing import Protocol, Callable
import logging
import MDP


"""
refactor these codes to be independent of a class.
"""

SEED = 100000
N_ACTIONS = 2

@dataclass
class AgentParams:
    """Default parameters for all the agents. Should they only have their own parameters???"""
    n_actions: int = 2

    greedy: bool = False
    epsilon: float = .1
    Q_alpha: float = .1


class BehaviorAgent(Protocol):
    rng = np.random.default_rng(SEED)

    model_type: str
    prior = np.ones(4) / 4

    def choose_action(self) -> int:
        ...

    def update_params(self, action, reward) -> None:
        ...

    def save_params(self) -> None:
        ...

    def load_params(self) -> None:
        ...


class Qlearning(BehaviorAgent):

    def __init__(self, greedy: bool = False, epsilon: float = .1, learning_rate: float = .1):
        self.greedy = greedy
        self.epsilon = epsilon
        self.n_actions = N_ACTIONS
        self.learning_rate = learning_rate

        self.Q = np.zeros(N_ACTIONS)
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS

        self.model_type = 'Q-learning'

    def choose_action(self) -> int:
        if self.greedy:
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(self.n_actions)
            else:
                action = np.argmax(self.Q)
        else:
            action = self.rng.choice(self.n_actions, p=self.action_dist)

        return action

    def update_params(self, action: int, reward: float) -> None:
        self.Q[action] += self.learning_rate * (reward - self.Q[action])  # nonstationary update, biased towards recent rewards
        self.action_dist = sp.special.softmax(self.Q)


class ForgettingQlearning(BehaviorAgent):

    def __init__(self, greedy: bool = False, epsilon: float = .1, learning_rate: float = .1,
                 stickiness: float = 0, tau: float = 5, temperature: float = 1):
        """
        TODO - go back through the source (Beron et al, PNAS 2022) and find some better names for the variables...
        """
        self.N = np.zeros(N_ACTIONS)
        self.Q = np.zeros(N_ACTIONS)
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS

        self.greedy = greedy
        self.epsilon = epsilon
        self.n_actions = N_ACTIONS
        self.learning_rate = learning_rate
        # self.action_dict = dict(right=0, left=1)

        self.stickiness = stickiness
        self.tau = tau
        self.temperature = temperature
        self.prev_action = -1

        self.model_type = 'Forgetting Q-learning'

    def choose_action(self) -> int:
        action = self.rng.choice(self.n_actions, p=self.action_dist)
        return action

    def update_params(self, action, reward) -> None:
        decay = np.exp(-1 / self.tau)
        self.Q[action] = decay * self.Q[action] + (1 - decay) * reward
        self.Q[1 - action] = decay * self.Q[1 - action]

        # right=0, left=1: map right to -1, left to +1
        action_ix = action * 2 - 1

        dQ = self.Q[0] - self.Q[1]
        psi = (self.stickiness * action_ix + dQ) / self.temperature
        logit = sp.special.expit(psi)
        self.action_dist = [logit, 1 - logit]


class HMM(BehaviorAgent):
    def __init__(self, task: MDP.MarkovDecisionProcess, temperature: float = 1, greedy: bool = False, epsilon: float = .1, stickiness: float = 0,
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
        self.task = task
        self.model_type = 'HMM'

        self.n_states = n_states
        self.prior = np.ones(n_states) / n_states
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS

        self.last_action = 0
        self.last_stimulus = 0

    def choose_action(self) -> int:
        stimulus = self.task.get_stimulus
        if self.greedy:
            if self.rng.random() < self.epsilon:
                action = self.rng.choice(N_ACTIONS)
            else:
                action = np.argmax(self.action_dist)
        else:
            self.update_observation_posterior(stimulus)
            self.update_action_dist(self.last_action)

            logging.info("action dist: {}".format(self.action_dist))
            action = self.rng.choice(N_ACTIONS, p=self.action_dist)

        self.last_action = action
        self.last_stimulus = stimulus
        return action

    def update_params(self, action, reward) -> None:
        # right=0, left=1: map right to -1, left to +1
        # p_outcome = self.outcome_probability(action, reward, self.last_stimulus)

        p_observation = self.observation_probability(self.last_stimulus)
        # p_reward_delivery = self.reward_delivery_probability(reward=reward, action=action)
        # p_reward_size = self.reward_size_probability(reward=reward, action=action)
        # p_outcome = p_reward_delivery * p_reward_size * p_observation

        p_reward_delivery = self.reward_probability(reward=reward, action=action)
        p_outcome = p_reward_delivery * p_observation

        posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)
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

    def outcome_probability(self, action: int, reward: float, stimulus: int) -> np.ndarray:
        # need the task parameters in this one
        p = np.zeros(4)
        if self.task.params.reward_std_dev > 0:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)
                p[2] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)
                p[3] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)

            elif action == 1:  # left choice
                p[0] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)
                p[2] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)
                p[3] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)

        else:  # potentially worth ignoring this case entirely
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = reward == self.task.params.active_reward_probability
                p[1] = reward == self.task.params.inactive_reward_probability
                p[2] = reward == self.task.params.active_reward_probability
                p[3] = reward == self.task.params.inactive_reward_probability

            elif action == 1:  # left choice
                p[0] = reward == self.task.params.inactive_reward_probability
                p[1] = reward == self.task.params.active_reward_probability
                p[2] = reward == self.task.params.inactive_reward_probability
                p[3] = reward == self.task.params.active_reward_probability

            p = p.astype(float)

        p *= self.observation_probability(stimulus)
        return p

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
        if self.task.params.reward_std_dev > 0:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)
                p[2] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)
                p[3] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)

            elif action == 1:  # left choice
                p[0] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)
                p[2] = norm.pdf(reward, self.task.params.mean_incorrect_reward, self.task.params.reward_std_dev)
                p[3] = norm.pdf(reward, self.task.params.mean_correct_reward, self.task.params.reward_std_dev)

        else:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = reward == self.task.params.mean_correct_reward
                p[1] = reward == self.task.params.mean_incorrect_reward
                p[2] = reward == self.task.params.mean_correct_reward
                p[3] = reward == self.task.params.mean_incorrect_reward

            elif action == 1:  # left choice
                p[0] = reward == self.task.params.mean_incorrect_reward
                p[1] = reward == self.task.params.mean_correct_reward
                p[2] = reward == self.task.params.mean_incorrect_reward
                p[3] = reward == self.task.params.mean_correct_reward

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
        posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)
        posterior /= np.sum(posterior)
        logging.info("observation -> updated posterior: {}".format(posterior))
        self.prior = posterior

    def reward_delivery_probability(self, action: int):
        p = np.zeros(4)
        if action == 0:  # right choice - find probability of the reward under each context
            p[0] = self.task.params.active_reward_probability
            p[1] = self.task.params.inactive_reward_probability
            p[2] = self.task.params.active_reward_probability
            p[3] = self.task.params.inactive_reward_probability

        elif action == 1:  # left choice
            p[0] = self.task.params.inactive_reward_probability
            p[1] = self.task.params.active_reward_probability
            p[2] = self.task.params.inactive_reward_probability
            p[3] = self.task.params.active_reward_probability

        return p

    @property
    def transition_matrix(self) -> np.ndarray:
        state_transition_matrix = np.ones((self.n_states, self.n_states)) * self.transition_prob
        for i in range(self.n_states):
            state_transition_matrix[i, i] = 1 - (self.n_states - 1) * self.transition_prob

        return state_transition_matrix


class Logistic(BehaviorAgent):
    """Todo"""
    pass


def logistic(self, dist, phi):
    """
    Logistic regression (RFLR) model based on Beron et al, PNAS 2022. Switched 0 and 1 from their convention for
    consistency with the other code here. Note that this option only models actions, not beliefs.
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


if __name__ == '__main__':
    task = POMDP()
    task.params.random_block_order = True

    task.to_intercontext()
    task.to_next_state()
    task.cur_trial_in_block = 50
    rew = task.step_non_markov(0)
    print(rew)
