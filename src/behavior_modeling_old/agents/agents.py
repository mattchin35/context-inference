import numpy as np
import scipy as sp
from scipy.stats import norm

from dataclasses import dataclass
from typing import Protocol, Callable, Tuple
from abc import ABC, abstractmethod
import logging
from src.behavior_modeling.parameters import task_config as config


"""
refactor these codes to be independent of a class.
"""

SEED = 12345
# np.random.seed(SEED)
rng = np.random.default_rng(SEED)

N_ACTIONS = 2
RIGHT_IX = 0
LEFT_IX = 1
eps = np.finfo(float).eps

states = ['right', 'left']
state_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]


def get_action_ix(action: int) -> int:
    # right=0, left=1: map right to -1, left to +1
    assert action in [0, 1], "action must be 0 or 1"
    return action * 2 - 1


class BehaviorAgent(ABC):
    model_type: str
    prior: np.ndarray
    Q: np.ndarray
    value: float
    greedy: bool
    epsilon: float
    action_dist: np.ndarray
    stickiness: float  # bias to previous action
    temperature: float  # randomness factor. Fully random has hi temp (logit->0), deterministic has low temp (logit->inf)

    @abstractmethod
    def update_params(self, action, reward) -> None:
        ...

    def choose_action(self, stimulus: int) -> tuple[int, np.ndarray]:
        # eps-greedy action selection
        if self.greedy:
            if rng.random() < self.epsilon:
                action = rng.choice(N_ACTIONS)
            else:
                action = np.argmax(self.action_dist)
        else:
            action = rng.choice(N_ACTIONS, p=self.action_dist)

        return action, self.action_dist.copy()

    def update_action_dist(self, action: int) -> None:
        action_ix = action * 2 - 1
        # logit_left = (self.stickiness * action_ix + self.value) / self.temperature  # temperature affects stickiness and value
        logit_left = self.stickiness * action_ix + self.value / self.temperature  # separate stickiness from temperature
        p_left = sp.special.expit(logit_left)
        self.action_dist = np.array([1 - p_left, p_left])


class Qlearning(BehaviorAgent):

    def __init__(self, params: config.TaskParams):
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

    def __init__(self, params: config.TaskParams):
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

    def __init__(self, agent_params: config.AgentParams, task_params: config.TaskParams):
        assert agent_params.action_temperature > 0, "must have positive temperature"

        self.temperature = agent_params.action_temperature
        self.greedy = agent_params.greedy_action_selection
        self.epsilon = agent_params.greedy_epsilon
        self.stickiness = agent_params.action_stickiness
        self.transition_prob = agent_params.HMM_transition_prob
        self.agent_params = agent_params
        self.task_params = task_params
        self.model_type = 'HMM'

        n_states = 2
        self.n_states = n_states
        self.prior = np.ones(n_states) / n_states
        self.value = 0  # relative value of L vs R actions
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS
        self.p_cue = task_params.p_cue

        self.last_action_ix = 0
        self.last_stimulus = 0
        self.last_action = 0

        self.alpha = agent_params.logistic_alpha
        self.beta = agent_params.logistic_beta
        self.tau = agent_params.logistic_tau

        # self.q = (np.exp(-1 / agent_params.logistic_tau) + 1) / 2  # q, probability of no system state change
        # self.p = sp.special.expit(
        #     agent_params.logistic_beta / (2 * (2 * self.q - 1)))  # p, probability of reward delivery
        # self.alpha = -(2 * self.q - 1) * sp.special.logit(self.p)  # stickiness
        # self.beta = agent_params.logistic_beta
        # self.tau = agent_params.logistic_tau
        # self.exp = np.exp(-1 / self.tau)
        # self.log_odds_L = sp.special.logit(.5)

    def choose_action(self, stimulus: int) -> Tuple[int, np.ndarray]:
        # self.update_observation_posterior(stimulus)
        # self.update_action_dist(self.last_action)
        # action, _ = super().choose_action(stimulus)
        logging.info("action dist: {}".format(self.action_dist))
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

        # value determination following Vertechi - uses expected reward, action sampling via relative value
        expected_rew_L = self.agent_params.HMM_active_reward_probability * self.prior[1] + self.agent_params.HMM_inactive_reward_probability * self.prior[0]
        expected_rew_R = self.agent_params.HMM_active_reward_probability * self.prior[0] + self.agent_params.HMM_inactive_reward_probability * self.prior[1]
        self.value = expected_rew_L - expected_rew_R  # relative value of L vs R actions

        # value determination following Beron - uses Bayesian posterior, action sampling via Thompson (belief) sampling
        # to match HMM and RFLR models, you need to use Thompson sampling
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

    def nonzero_reward_size_probability(self, reward, action):
        p = np.zeros(2)
        if self.task_params.reward_std_dev > 0:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = norm.pdf(reward, self.task_params.mean_correct_reward, self.task_params.reward_std_dev)
                p[1] = norm.pdf(reward, self.task_params.mean_incorrect_reward, self.task_params.reward_std_dev)

            elif action == 1:  # left choice
                p[0] = norm.pdf(reward, self.task_params.mean_incorrect_reward, self.task_params.reward_std_dev)
                p[1] = norm.pdf(reward, self.task_params.mean_correct_reward, self.task_params.reward_std_dev)

        else:
            if action == 0:  # right choice - find probability of the reward under each context
                p[0] = reward == self.task_params.mean_correct_reward
                p[1] = reward == self.task_params.mean_incorrect_reward

            elif action == 1:  # left choice
                p[0] = reward == self.task_params.mean_incorrect_reward
                p[1] = reward == self.task_params.mean_correct_reward

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

    def update_observation_posterior(self, stimulus: int) -> None:
        if stimulus == -1:
            return

        p_outcome = self.observation_probability(stimulus)
        posterior = p_outcome * np.dot(self.transition_matrix.T, self.prior)  # todo - .T not be needed, it's symmetric...
        posterior /= (np.sum(posterior) + eps)
        logging.info("observation -> updated posterior: {}".format(posterior))
        self.prior = posterior

    def reward_delivery_probability(self, action: int) -> np.ndarray:
        p = np.zeros(2)
        if action == 0:  # right choice - find probability of the right reward under each context
            p[0] = self.agent_params.HMM_active_reward_probability
            p[1] = self.agent_params.HMM_inactive_reward_probability

        elif action == 1:  # left choice - find probability of the left reward under each context
            p[0] = self.agent_params.HMM_inactive_reward_probability
            p[1] = self.agent_params.HMM_active_reward_probability

        return p

    @property
    def transition_matrix(self) -> np.ndarray:
        state_transition_matrix = np.ones((2, 2)) * self.transition_prob
        for i in range(2):
            state_transition_matrix[i, i] = 1 - self.transition_prob

        return state_transition_matrix


class HMM_recursive(HMM):
    """An HMM variant using the log-odds recursion described in Beron et al, PNAS 2022.
    Primarily meant as a proof-of-concept/verification that the paper's math is legit."""

    def __init__(self,  agent_params: config.AgentParams, task_params: config.TaskParams):
        super().__init__(agent_params, task_params)
        self.kappa = 0  # updated every trial

        # using HMM transition and reward probabilities, calculate recursive parameters
        self.q = 1 - agent_params.HMM_transition_prob  # q, probability of no system state change
        self.p = agent_params.HMM_active_reward_probability  # p, probability of reward delivery
        self.alpha = -(2*self.q - 1) * sp.special.logit(self.p)  # stickiness
        self.beta = 2 * (2*self.q - 1) * sp.special.logit(self.p)  # learning rate
        self.decay = 2 * self.q - 1
        self.tau = -1 / np.log(2*self.q - 1)

        self.model_type = 'HMM_recursive'
        self.log_odds_L = sp.special.logit(.5)

    def update_params(self, action, reward) -> None:
        # prior update
        self.log_odds_L = self.decay * self.log_odds_L + self.alpha * get_action_ix(action) + self.beta * reward * get_action_ix(action)
        self.kappa = (self.agent_params.logistic_alpha - self.alpha) * get_action_ix(action) - self.decay * self.agent_params.logistic_alpha * self.last_action_ix
        self.last_action_ix = get_action_ix(action)

        prior_L = sp.special.expit(self.log_odds_L + self.kappa)
        self.action_dist = np.array([1 - prior_L, prior_L])
        self.prior = np.array([1 - prior_L, prior_L])  # action dist and prior are the same in Thompson sampling


class HMM_RFLR(HMM):
    """
    Modification of the HMM to match the RFLR model from Beron et al, PNAS 2022.
    For details, check the paper supplementals.
    """

    def __init__(self,  agent_params: config.AgentParams, task_params: config.TaskParams):
        super().__init__(agent_params, task_params)
        self.kappa = 0  # updated every trial

        # using logistic tau and beta, calculate HMM probabilities (and thus a stickiness parameter)
        # that match the logistic regression model by construction
        self.q = (np.exp(-1 / agent_params.logistic_tau) + 1) / 2  # q, probability of no system state change
        self.p = sp.special.expit(agent_params.logistic_beta / (2 * (2*self.q - 1)))  # p, probability of reward delivery
        self.alpha = -(2*self.q - 1) * sp.special.logit(self.p)  # stickiness
        self.beta = agent_params.logistic_beta
        self.tau = agent_params.logistic_tau
        self.decay = np.exp(-1/self.tau)
        self.value = 0  # relative value of L vs R actions - not used in this model, use the log-odds or prior instead

        self.model_type = 'HMM_RFLR'
        self.log_odds_L = sp.special.logit(.5)

    def update_params(self, action, reward) -> None:
        # prior update
        action_ix = get_action_ix(action)
        self.log_odds_L = self.decay * self.log_odds_L + self.alpha * action_ix + self.beta * reward * action_ix
        self.kappa = (self.agent_params.logistic_alpha - self.alpha) * action_ix - self.decay * self.agent_params.logistic_alpha * self.last_action_ix
        self.last_action_ix = get_action_ix(action)

        prior_L = sp.special.expit(self.log_odds_L + self.kappa)
        self.action_dist = np.array([1 - prior_L, prior_L])
        self.prior = np.array([1 - prior_L, prior_L])  # action dist and prior are the same in Thompson sampling

    # def update_action_dist(self, action: int) -> None:
    #     # pL = sp.special.expit(self.value + self.kappa)
    #     pL = sp.special.expit(self.value)
    #     self.action_dist = np.array([1-pL, pL])


class Logistic(BehaviorAgent):
    """Logistic regression (RFLR) model based on Beron et al, PNAS 2022."""
    def __init__(self, agent_params: config.AgentParams, task_params: config.TaskParams):
        self.N = np.zeros(N_ACTIONS)
        self.Q = np.zeros(N_ACTIONS)
        self.action_dist = np.ones(N_ACTIONS) / N_ACTIONS
        self.value = 0  # relative value of L vs R actions
        self.last_action_ix = 0

        self.n_actions = N_ACTIONS

        self.greedy = agent_params.greedy_action_selection
        self.epsilon = agent_params.greedy_epsilon

        self.alpha = agent_params.logistic_alpha  # stickiness
        self.beta = agent_params.logistic_beta  # learning rate
        self.tau = agent_params.logistic_tau  # decay parameter
        self.phi = 0
        self.log_odds_L = sp.special.logit(.5)

        self.stickiness = agent_params.action_stickiness
        self.temperature = agent_params.action_temperature

        self.model_type = 'Logistic regression'

    def update_params(self, action, reward) -> None:
        action_ix = get_action_ix(action)  # 1 left, -1 right
        decay = np.exp(-1 / self.tau)

        self.value = decay * self.value + self.beta * action_ix * reward
        self.log_odds_L = self.alpha * action_ix + self.value
        pL = sp.special.expit(self.log_odds_L)
        self.action_dist = np.array([1 - pL, pL])

        # learning_rate = (1 - decay) / self.temperature   # use this for equivalence with FQL
        # self.update_action_dist(action)

    # def update_action_dist(self, action: int) -> None:
    #     action_ix = get_action_ix(action)
    #     logit_left = self.alpha * action_ix + self.value #/ self.temperature
    #     pL = sp.special.expit(logit_left)
    #     self.action_dist = np.array([1 - pL, pL])


if __name__ == '__main__':
    pass