import numpy as np
import scipy as sp
from scipy.stats import norm

from typing import Protocol, Callable, Tuple
from abc import ABC, abstractmethod
import logging
from src.task import MDP


SEED = 1000
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

N_ACTIONS = 2
RIGHT_IX = 0
LEFT_IX = 1
eps = np.finfo(float).eps


def select_agent(agent_name: str, params: MDP.TaskParams) -> \
        MDP.MarkovDecisionProcess:
    if agent_name == 'HMM':
        agent = HMM(params)
    elif agent_name == 'Qlearning':
        agent = Qlearning(params)
    elif agent_name == 'F-Qlearning':
        agent = ForgettingQlearning(params)
    elif agent_name == 'Logistic':
        agent = Logistic(params)
    else:
        raise ValueError("Unknown agent name: {}".format(agent_name))
    return agent


def get_action_ix(action: int) -> int:
    return action * 2 - 1


class BehaviorAgent(ABC):
    rng = np.random.default_rng(SEED)

    model_type: str
    prior: np.ndarray
    Q: np.ndarray
    value: float
    greedy: bool
    epsilon: float
    action_dist: np.ndarray
    stickiness: float
    temperature: float

    def update_params(self, action, reward) -> None:
        ...

    def choose_action(self, stimulus: int) -> Tuple[int, np.ndarray]:
        if self.greedy:
            if rng.random() < self.epsilon:
                action = rng.choice(N_ACTIONS)
            else:
                action = np.argmax(self.action_dist)
        else:
            action = rng.choice(N_ACTIONS, p=self.action_dist)

        return action, self.action_dist.copy()

    def update_action_dist(self, action: int) -> None:
        # if action == -1:
        #     action_ix = 0
        # else:

        action_ix = action * 2 - 1
        # logit_left = (self.stickiness * action_ix + self.value) / self.temperature
        logit_left = self.stickiness * action_ix + self.value / self.temperature
        p_left = sp.special.expit(logit_left)
        self.action_dist = np.array([1 - p_left, p_left])


class Qlearning(BehaviorAgent):

    def __init__(self, params: MDP.TaskParams):

        self.greedy = params.greedy_action_selection
        self.epsilon = params.greedy_epsilon
        self.n_actions = N_ACTIONS
        self.learning_rate = params.QL_learning_rate
        self.temperature = params.action_temperature
        self.stickiness = params.action_stickiness

        self.N = np.zeros(N_ACTIONS)
        self.Q = np.zeros(N_ACTIONS)
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS
        self.value = 0  # relative value of L vs R actions

        self.model_type = 'Q-learning'

    def update_params(self, action: int, reward: float) -> None:
        self.Q[action] += self.learning_rate * (reward - self.Q[action])  # nonstationary update, biased towards recent rewards
        self.value = self.Q[LEFT_IX] - self.Q[RIGHT_IX]  # relative value of R vs L actions
        self.update_action_dist(action)


class ForgettingQlearning(BehaviorAgent):

    def __init__(self, params: MDP.TaskParams):
        self.n_actions = N_ACTIONS
        self.N = np.zeros(N_ACTIONS)
        self.Q = np.zeros(N_ACTIONS)
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS
        self.value = 0  # relative value of L vs R actions
        self.last_action_ix = 0

        self.greedy = params.greedy_action_selection
        self.epsilon = params.greedy_epsilon

        self.stickiness = params.action_stickiness
        self.temperature = params.action_temperature
        self.decay = params.FQL_decay

        self.alpha = params.logistic_alpha
        self.tau = params.logistic_tau

        self.model_type = 'Forgetting Q-learning'

    def update_params(self, action, reward) -> None:
        # Via Vertechi Neuron 2020
        self.Q *= self.decay  # "forgetting"
        self.Q[action] += (1 - self.decay) * reward  # update the chosen action
        self.value = self.Q[LEFT_IX] - self.Q[RIGHT_IX]  # relative value of R vs L actions

        # via Beron PNAS 2022; formulated for equivalence with Reduced-form logistic regression (RFLR)
        # decay = np.exp(-1 / self.tau)
        # self.Q *= decay  # "forgetting"
        # self.Q[action] += (1 - decay) * reward  # update the chosen action
        # action_ix = action * 2 - 1
        # self.value = decay * self.value + (1 - decay) * action_ix * reward  # recursive formulation

        self.update_action_dist(action)

    def update_action_dist(self, action: int) -> None:
        action_ix = get_action_ix(action)
        # logit_left = self.stickiness * action_ix + self.value / self.temperature
        logit_left = self.alpha * action_ix + self.value / self.temperature
        pL = sp.special.expit(logit_left)
        self.action_dist = np.array([1 - pL, pL])


class HMM(BehaviorAgent):

    def __init__(self, params: MDP.TaskParams, p_cue: float = .5, balanced_MDP: bool = True):
        assert params.action_temperature > 0, "must have positive temperature"

        self.temperature = params.action_temperature
        self.greedy = params.greedy_action_selection
        self.epsilon = params.greedy_epsilon
        self.stickiness = params.action_stickiness
        self.transition_prob = params.HMM_transition_prob
        self.params = params
        self.model_type = 'HMM'
        self.balanced_MDP = balanced_MDP

        n_states = 2
        self.n_states = n_states
        self.prior = np.ones(n_states) / n_states
        self.value = 0  # relative value of L vs R actions
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS
        self.p_cue = p_cue

        self.last_action_ix = 0
        self.last_stimulus = 0

        self.alpha = params.logistic_alpha
        self.tau = params.logistic_tau

    def choose_action(self, stimulus: int) -> Tuple[int, np.ndarray]:
        # self.update_observation_posterior(stimulus)
        # self.update_action_dist(self.last_action)
        logging.info("action dist: {}".format(self.action_dist))
        # action, _ = super().choose_action(stimulus)
        if self.greedy:
            if rng.random() < self.epsilon:
                action = rng.choice(N_ACTIONS)
            else:
                action = np.argmax(self.action_dist)
        else:
            action = rng.choice(N_ACTIONS, p=self.action_dist)

        self.last_action = action
        self.last_stimulus = stimulus
        return action, self.action_dist.copy()

    def update_params(self, action, reward) -> None:
        """
        Update the prior based on the action/reward outcome.
        This should actually work for any observation - so if those get added, update params after reward and
        again after observation.
        """
        # right=0, left=1: map right to -1, left to +1
        p_reward_delivery = self.reward_probability(reward=reward, action=action)

        # posterior = p_reward_delivery * self.prior  # transition matrix from the observation step to the reward step is identity - no state change in this interval
        # posterior = p_outcome * self.prior  # transition matrix from the observation step to the reward step is identity - no state change in this interval
        # if self.params.give_observations:
        #     p_observation = self.observation_probability(self.last_stimulus)
        #     p_outcome = p_reward_delivery * p_observation
        # else:
        #     p_outcome = p_reward_delivery

        # p_outcome = p_reward_delivery
        # posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)  # todo - .T not be needed, it's symmetric...
        # posterior /= (np.sum(posterior) + eps)

        p_outcome = p_reward_delivery * self.prior
        p_outcome /= (np.sum(p_outcome) + eps)
        posterior = np.dot(self.transition_matrix.T, p_outcome)  # todo - .T not be needed?, it's symmetric...
        posterior /= (np.sum(posterior) + eps)
        logging.info("reward -> updated posterior: {}".format(posterior))
        self.prior = posterior

        expected_rew_L = self.params.active_reward_probability * self.prior[1] + self.params.inactive_reward_probability * self.prior[0]
        expected_rew_R = self.params.active_reward_probability * self.prior[0] + self.params.inactive_reward_probability * self.prior[1]
        self.value = expected_rew_L - expected_rew_R  # relative value of L vs R actions

        # self.value = sp.special.logit(self.prior[1])

        self.update_action_dist(action)

    def update_action_dist(self, action: int):
        action_ix = get_action_ix(action)
        # logit_left = self.stickiness * action_ix + self.value / self.temperature
        logit_left = self.alpha * action_ix + self.value / self.temperature
        pL = sp.special.expit(logit_left)
        self.action_dist = np.array([1-pL, pL])

    def observation_probability(self, stimulus: int) -> np.ndarray:
        if stimulus == 0:  # i.e. right cued context
            p = np.array([self.p_cue, 0])
            # p = np.array([1, 0])
        elif stimulus == 1:  # i.e. left cued context
            p = np.array([0, self.p_cue])
            # p = np.array([0, 1])
        else:  #
            p = np.array([1-self.p_cue, 1-self.p_cue])
            # p = np.array([1, 1])
        return p

    def update_observation_posterior(self, stimulus: int) -> None:
        if stimulus == -1:
            return

        p_outcome = self.observation_probability(stimulus)
        posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)  # todo - .T not be needed, it's symmetric...
        posterior /= (np.sum(posterior) + eps)
        logging.info("observation -> updated posterior: {}".format(posterior))
        self.prior = posterior

    def nonzero_reward_size_probability(self, reward, action):
        p = np.zeros(2)
        if self.params.reward_std_dev > 0:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)

            elif action == 1:  # left choice
                p[0] = norm.pdf(reward, self.params.mean_incorrect_reward, self.params.reward_std_dev)
                p[1] = norm.pdf(reward, self.params.mean_correct_reward, self.params.reward_std_dev)

        else:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = reward == self.params.mean_correct_reward
                p[1] = reward == self.params.mean_incorrect_reward

            elif action == 1:  # left choice
                p[0] = reward == self.params.mean_incorrect_reward
                p[1] = reward == self.params.mean_correct_reward

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

    def reward_delivery_probability(self, action: int) -> np.ndarray:
        p = np.zeros(2)
        if action == 0:  # right choice - find probability of the right reward under each context
            p[0] = self.params.active_reward_probability
            p[1] = self.params.inactive_reward_probability

        elif action == 1:  # left choice - find probability of the left reward under each context
            p[0] = self.params.inactive_reward_probability
            p[1] = self.params.active_reward_probability

        return p

    @property
    def transition_matrix(self) -> np.ndarray:
        state_transition_matrix = np.ones((2, 2)) * self.transition_prob

        if self.balanced_MDP:
            for i in range(2):
                state_transition_matrix[i, i] = 1 - self.transition_prob
        else:
            state_transition_matrix[0, 1] = 0
            state_transition_matrix[1, 1] = 1

        return state_transition_matrix


class Logistic(BehaviorAgent):
    """Logistic regression (RFLR) model based on Beron et al, PNAS 2022."""
    def __init__(self, params: MDP.TaskParams):
        self.N = np.zeros(N_ACTIONS)
        self.Q = np.zeros(N_ACTIONS)
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS
        self.value = 0  # relative value of L vs R actions
        self.last_action_ix = 0

        self.n_actions = N_ACTIONS

        self.greedy = params.greedy_action_selection
        self.epsilon = params.greedy_epsilon

        self.alpha = params.logistic_alpha
        self.beta = params.logistic_beta
        self.tau = params.logistic_tau
        self.phi = 0

        self.stickiness = params.action_stickiness
        self.temperature = params.action_temperature

        self.model_type = 'Logistic regression'

    def update_params(self, action, reward) -> None:
        action_ix = get_action_ix(action)  # 1 left, -1 right
        decay = np.exp(-1 / self.tau)
        learning_rate = self.beta  # use this to set learning rate directly
        # learning_rate = (1 - decay) / self.temperature   # use this for equivalence with FQL

        # indirect recursion
        self.value = decay * self.value + learning_rate * action_ix * reward
        self.update_action_dist(action)

    def update_action_dist(self, action: int) -> None:
        action_ix = get_action_ix(action)
        logit_left = self.alpha * action_ix + self.value #/ self.temperature
        pL = sp.special.expit(logit_left)
        self.action_dist = np.array([1 - pL, pL])


if __name__ == '__main__':
    pass