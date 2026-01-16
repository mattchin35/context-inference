"""
This module contains functions that update the value of a decision variable based on the outcome of a trial.
Source: Cazettes et al., Nature Neuroscience 2023
"""
import numpy as np


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

