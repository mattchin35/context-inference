"""Shared counterfactual side-specific omission state and readouts.

Choice encoding is ``0=right`` and ``1=left``. Raw doubt is left-minus-right
and therefore points toward the doubted side. Its negation points toward the
preferred next choice.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CounterfactualDoubtState:
    """Counterfactual omission counts for one sequential session.

    Parameters
    ----------
    right_omissions, left_omissions : float
        Nonnegative effective omission counts, in trials. Values are normally
        integral; passive inactive-agent decay can produce fractional counts.
    """

    right_omissions: float = 0.0
    left_omissions: float = 0.0

    def __post_init__(self) -> None:
        counts = np.asarray([self.right_omissions, self.left_omissions], dtype=float)
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("Counterfactual omission counts must be finite and nonnegative.")


def _validate_omission_lambda(omission_lambda: float) -> None:
    """Validate a scalar exponential saturation rate."""
    if not np.isfinite(omission_lambda) or omission_lambda <= 0:
        raise ValueError("omission_lambda must be finite and greater than zero.")


def counterfactual_doubt_raw_values(
    right_omissions: float | np.ndarray,
    left_omissions: float | np.ndarray,
    omission_lambda: float,
) -> float | np.ndarray:
    """Return left-minus-right counterfactual doubt.

    Parameters
    ----------
    right_omissions, left_omissions : float or ndarray
        Broadcast-compatible nonnegative effective counts, in trials.
    omission_lambda : float
        Positive exponential saturation rate, in inverse trials.

    Returns
    -------
    float or ndarray
        Broadcast shape of the inputs, bounded to ``[-1, 1]``. Positive values
        denote greater doubt about the left side.
    """
    _validate_omission_lambda(omission_lambda)
    right = np.asarray(right_omissions, dtype=float)
    left = np.asarray(left_omissions, dtype=float)
    if np.any(~np.isfinite(right)) or np.any(~np.isfinite(left)):
        raise ValueError("Omission counts must be finite.")
    if np.any(right < 0) or np.any(left < 0):
        raise ValueError("Omission counts must be nonnegative.")

    raw_value = (1.0 - np.exp(-omission_lambda * left)) - (
        1.0 - np.exp(-omission_lambda * right)
    )
    if raw_value.ndim == 0:
        return float(raw_value)
    return raw_value


def counterfactual_doubt_raw_value(
    state: CounterfactualDoubtState,
    omission_lambda: float,
) -> float:
    """Return doubted-side-oriented value for one state.

    Parameters
    ----------
    state : CounterfactualDoubtState
        Effective omission counts for one session.
    omission_lambda : float
        Positive exponential saturation rate, in inverse trials.

    Returns
    -------
    float
        Left-minus-right raw doubt in ``[-1, 1]``.
    """
    return float(
        counterfactual_doubt_raw_values(
            state.right_omissions,
            state.left_omissions,
            omission_lambda,
        )
    )


def counterfactual_doubt_choice_value(
    state: CounterfactualDoubtState,
    omission_lambda: float,
) -> float:
    """Return preferred-choice-oriented doubt for one state.

    Parameters and units match :func:`counterfactual_doubt_raw_value`.

    Returns
    -------
    float
        Negated raw doubt in ``[-1, 1]``; positive favors left.
    """
    return -counterfactual_doubt_raw_value(state, omission_lambda)


def update_counterfactual_doubt(
    state: CounterfactualDoubtState,
    action: int,
    reward: float,
) -> CounterfactualDoubtState:
    """Advance state after one valid completed trial.

    Parameters
    ----------
    state : CounterfactualDoubtState
        Pre-update state for one session.
    action : int
        Observed choice, encoded ``0=right`` or ``1=left``.
    reward : float
        Observed reward amount. Positive is rewarded; zero or negative is an
        omission. Units are task reward units.

    Returns
    -------
    CounterfactualDoubtState
        New state. Any reward clears both counters; otherwise only the chosen
        side increments by one trial.
    """
    if action not in (0, 1):
        raise ValueError("action must be 0 (right) or 1 (left).")
    if not np.isfinite(reward):
        raise ValueError("reward must be finite for a valid trial.")
    if reward > 0:
        return CounterfactualDoubtState()
    if action == 0:
        return CounterfactualDoubtState(
            right_omissions=state.right_omissions + 1.0,
            left_omissions=state.left_omissions,
        )
    return CounterfactualDoubtState(
        right_omissions=state.right_omissions,
        left_omissions=state.left_omissions + 1.0,
    )


def passively_decay_counterfactual_doubt(
    state: CounterfactualDoubtState,
    omission_lambda: float,
) -> CounterfactualDoubtState:
    """Apply the existing inactive-agent passive decay rule.

    Parameters
    ----------
    state : CounterfactualDoubtState
        Effective omission counts before one inactive update.
    omission_lambda : float
        Positive exponential saturation rate, in inverse trials.

    Returns
    -------
    CounterfactualDoubtState
        Fractional effective counts whose side-specific saturated doubt was
        multiplied by ``exp(-omission_lambda)``.
    """
    _validate_omission_lambda(omission_lambda)
    decay_factor = float(np.exp(-omission_lambda))

    def decayed_count(count: float) -> float:
        side_doubt = (1.0 - np.exp(-omission_lambda * count)) * decay_factor
        return float(-np.log(max(1.0 - side_doubt, np.finfo(float).eps)) / omission_lambda)

    return CounterfactualDoubtState(
        right_omissions=decayed_count(state.right_omissions),
        left_omissions=decayed_count(state.left_omissions),
    )
