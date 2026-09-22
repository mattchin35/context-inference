"""
This module contains functions that update the value of a decision variable based on the outcome of a trial.
Source: Cazettes et al., Nature Neuroscience 2023
"""
import numpy as np

from src.behavior_modeling.counterfactual_doubt import (
    CounterfactualDoubtState,
    update_counterfactual_doubt,
)

states = ['right', 'left']
side_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]


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


def value_counter(count: int, reward: int):
    """
    A simple reward count, increasing with reward and decreasing with omissions.
    """
    if reward == 0:  # omission
        g = 1
        c = -1
    elif reward == 1:
        g = 1
        c = 1
    else:
        raise ValueError('Reward must be 0 or 1.')

    count = g * count + c
    return count


def sided_value_counter(left_count: int, right_count: int, action: int, reward: int, zero_min=False, counterfactual=False, monotonic=False) -> tuple[int, int]:
    """
    This counter resets the value of the unchosen action on rewarded trials to 0.
    """
    if action == side_dict['right']:
        if reward == 0 and not monotonic:
            right_count -= 1
        elif reward == 1:
            right_count = np.amax([right_count, 0]) + 1
            left_count *= (0 if counterfactual else 1)

    elif action == side_dict['left']:
        if reward == 0 and not monotonic:
            left_count -= 1
        elif reward == 1:
            left_count = np.amax([left_count, 0]) + 1
            right_count *= (0 if counterfactual else 1)

    if zero_min:
        left_count = np.amax([left_count, 0])
        right_count = np.amax([right_count, 0])

    return left_count, right_count


def sided_omissions_counter(left_count: int, right_count: int, action: int, reward: int, counterfactual=False) -> tuple[int, int]:
    """
    A counter of consecutive omissions, increasing the count on unrewarded trials and resetting the count on rewarded trials.
    If counterfactual is True, also resets the value of the unchosen action on rewarded trials to 0.
    """
    if counterfactual:
        state = update_counterfactual_doubt(
            CounterfactualDoubtState(
                right_omissions=right_count,
                left_omissions=left_count,
            ),
            action=action,
            reward=reward,
        )
        # Preserve this legacy counter's integer tuple contract for ordinary
        # trial updates even though the shared state also supports fractional
        # effective counts used by passive agent decay.
        left_value = (
            int(state.left_omissions)
            if state.left_omissions.is_integer()
            else state.left_omissions
        )
        right_value = (
            int(state.right_omissions)
            if state.right_omissions.is_integer()
            else state.right_omissions
        )
        return left_value, right_value

    if action == side_dict['right']:
        if reward == 0:
            right_count += 1
        elif reward == 1:
            right_count = 0

    elif action == side_dict['left']:
        if reward == 0:
            left_count += 1
        elif reward == 1:
            left_count = 0

    return left_count, right_count
