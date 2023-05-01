import numpy as np
import scipy as sp
from scipy.stats import beta
import matplotlib.pyplot as plt
# from lever_behavior import create_block, block_probabilities

"""
Sutton and Barto Ch 2 and some extras from exploration vs exploitation.
"""


def greedy(epsilon=0):
    pass

def incremental(Q, N, epsilon=.1):
    # crude, simple bandit algorithm. Choose the argmax, but sample the
    # other options from time to time
    """

    :param Q: expected reward for each action
    :param N: number of times each action is selected
    :param epsilon: probability of random action selection
    :return:
    """
    if np.random.rand() < epsilon:
        action = np.random.choice([0,1])
    else:
        action = np.argmax(Q)
    reward = 0

    N[action] += 1
    Q[action] += (reward - Q[action]) / N[action]  # incremental average reward
    return Q, N, reward


def nonstationary(Q, N, alpha=.9, epsilon=.1):
    """The world is non-stationary (reward contingencies change over time)"""
    if np.random.rand() < epsilon:
        action = np.random.choice([0, 1])
    else:
        action = np.argmax(Q)
    reward = 0

    N[action] += 1
    Q[action] += alpha * (reward - Q[action])
    return Q, N, reward


def upper_confidence_bounds():
    """Interesting, but often not practical for non-stationary problems. Skip."""
    pass


def gradient_bandit(N, H, Q=0, nonstationary=True, alpha=.1, beta=.9):
    """
    Softmax action selection.
    N - count of times each action is selected
    H - history, the parameters used for softmax distribution
    Q - a baseline parameter updated with rewards. Can be used to speed up learning with a priori knowledge.
    alpha - learning size parameter for gradient updates
    beta - recency bias for non-stationary problems
    """
    Hdist = np.exp(H) / np.sum(np.exp(H))

    action = np.random.choice(2, p=Hdist)
    reward = 0

    N[action] += 1
    # for all other actions
    H -= alpha * Hdist * (reward - Q)
    # for chosen action
    H[action] += alpha * (reward - Q[action]) / N[action] * (1 - Hdist[action])

    if nonstationary:
        Q += beta * (reward - Q[action])  # nonstationary update, biased towards recent rewards
    else:
        Q[action] += (reward - Q[action]) / N[action]  # incremental average reward for stationary problems

    return N, H, Q, reward


if __name__ == '__main__':
    npress = 60
    std_dev = .5
    means = [1, .25]
    Y = np.concatenate([create_block('A'), create_block('B'),
                        create_block('C1'), create_block('C2'),
                        create_block('A'), create_block('C2')])
    pStates = block_probabilities(Y, include_stimulus=True)
    Ysize = np.shape(Y)[0]