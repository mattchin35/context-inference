"""
This module contains functions that update the value of a decision variable based on the outcome of a trial.
Source: Cazettes et al., Nature Neuroscience 2023
"""
import numpy as np

states = ['right', 'left']
state_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]


def consecutive_fail_counter(count: int, reward: int):
    if reward == 0:  # failure
        g = 1
        c = 1
    else:  # reward
        g = 0
        c = 0

    count = g * count + c
    return count


def consecutive_fail_renewal_counter(count: int, reward: int, last_rewarded: bool):
    """
    Count the number of consecutive omissions, resetting the count when a new chain of omissions begins.
    Hold the count during the string of rewards following the last omission.
    """
    if reward == 0:  # omission
        g = 1 - int(last_rewarded)
        c = 1
    else:  # reward
        g = 1
        c = 0

    count = g * count + c
    return count


def consecutive_reward_counter(count: int, reward: int):
    if reward == 0:  # failure
        g = 0
        c = 0
    else:  # reward
        g = 1
        c = 1

    count = g * count + c
    return count


def consecutive_reward_renewal_counter(count: int, reward: int, last_rewarded: bool):
    """
    Count the number of consecutive rewards, resetting the count when a new chain of rewards begins.
    Hold the count during the string of omissions following the last reward.
    """
    if reward == 0:  # omission
        g = 1
        c = 0
    else:  # reward
        g = int(last_rewarded)
        c = 1

    count = g * count + c
    return count


def negative_value_counter(count: int, reward: int):
    """
    This should be a one-sided reward count, so a mouse would switch when the value is too low/the negative
    value is too high. Increment the count on unrewarded trials.
    """
    if reward == 0:  # omission
        g = 1
        c = 1
    else:  # reward
        g = 1
        c = -1

    count = g * count + c
    return count


def counterfactual_value_counter(left_count: int, right_count: int, action: int, reward: int, zero_min=False) -> tuple[int, int]:
    if action == 0:
        if reward == 0:
            right_count -= 1
        elif reward == 1:
            right_count = np.amax([right_count, 0]) + 1
            left_count = 0

    elif action == 1:
        if reward == 0:
            left_count -= 1
        elif reward == 1:
            left_count = np.amax([left_count, 0]) + 1
            right_count = 0

    if zero_min:
        left_count = np.amax([left_count, 0])
        right_count = np.amax([right_count, 0])

    return left_count, right_count


def counterfactual_omissions_counter(left_count: int, right_count: int, action: int, reward: int) -> tuple[int, int]:
    if action == 0: #right
        if reward == 0:
            right_count += 1
        elif reward == 1:
            right_count = 0
            left_count = 0

    elif action == 1:
        if reward == 0:
            left_count += 1
        elif reward == 1:
            left_count = 0
            right_count = 0

    return left_count, right_count



