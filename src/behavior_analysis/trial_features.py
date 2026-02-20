import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
from scipy.stats import norm
from typing import Protocol, Callable
import logging

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


def relative_omissions_index(R_omissions, L_omissions, epsilon=1e-8):
    """
    Compute relative omissions index in [-1, 1].

    Parameters
    ----------
    R_omissions : float or array-like
        Consecutive unrewarded trials on the right side.
    L_omissions : float or array-like
        Consecutive unrewarded trials on the left side.
    epsilon : float
        Small constant to prevent division by zero.

    Returns
    -------
    O : float or np.ndarray
        Relative omissions index in [-1, 1].
        +1 → only right accumulating omissions
        -1 → only left accumulating omissions
         0 → equal omissions
    """

    R = np.asarray(R_omissions, dtype=float)
    L = np.asarray(L_omissions, dtype=float)

    return (R - L) / (R + L + epsilon)


def signed_omission_regressor(loss_streak, lam, choice_side):
    """
    Compute signed omission regressor using exponential saturating doubt.

    Parameters
    ----------
    loss_streak : float or array-like
        Consecutive unrewarded trial count (>= 0).
    lam : float
        Saturation rate parameter (lambda > 0).
    choice_side : str or int or array-like
        Current choice.
        Accepts:
            - 'right' / 'left'
            - 1 (right) / -1 (left)

    Returns
    -------
    O : float or np.ndarray
        Signed omission regressor in [-1, 1].
    """

    # Convert to numpy array for vectorization
    D = np.asarray(loss_streak)

    # Unsigned doubt in [0, 1)
    H = 1 - np.exp(-lam * D)

    # Determine sign
    if isinstance(choice_side, str):
        sign = 1 if choice_side.lower() == 'right' else -1
    else:
        sign = np.asarray(choice_side)  # expects +1 (right) or -1 (left)

    return sign * H


def relative_doubt_index(R_omissions, L_omissions, lam):
    """
    Compute relative doubt index using exponential saturating omission
    for both sides, returning a value in [-1, 1].

    Parameters
    ----------
    R_omissions : float or array-like
        Consecutive unrewarded trials on the right side.
    L_omissions : float or array-like
        Consecutive unrewarded trials on the left side.
    lam : float
        Exponential saturation parameter (lambda > 0).

    Returns
    -------
    D : float or np.ndarray
        Relative doubt index in [-1, 1].
        +1 → strong doubt on right only
        -1 → strong doubt on left only
         0 → equal doubt
    """

    R = np.asarray(R_omissions, dtype=float)
    L = np.asarray(L_omissions, dtype=float)

    # Exponential saturating doubt per side
    H_R = 1 - np.exp(-lam * R)
    H_L = 1 - np.exp(-lam * L)

    return H_R - H_L

