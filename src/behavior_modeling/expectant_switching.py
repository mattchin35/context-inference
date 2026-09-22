"""Pure components and aligned replay for expectant-switching exemplars.

All signed quantities use ``positive=left`` and ``negative=right``. Replay
outputs are pre-trial: row ``t`` uses only valid observations through ``t-1``.
"""

from dataclasses import dataclass, field

import numpy as np

from src.behavior_modeling.counterfactual_doubt import (
    CounterfactualDoubtState,
    counterfactual_doubt_choice_value,
    counterfactual_doubt_raw_value,
    update_counterfactual_doubt,
)


def _validate_finite(name: str, value: float) -> None:
    """Reject a nonfinite scalar configuration value."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _validate_weight(name: str, value: float) -> None:
    """Reject a nonfinite or negative exemplar weight."""
    _validate_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


@dataclass(frozen=True)
class ExpectancyCurveParams:
    """Parameters for the saturating reward-count expectancy curve.

    Parameters
    ----------
    threshold : float
        Reward count at which expectancy equals 0.5, in rewarded trials.
    scale : float
        Positive transition width, in rewarded trials.
    """

    threshold: float = 3.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_finite("threshold", self.threshold)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and greater than zero.")


@dataclass(frozen=True)
class SimpleProbePersistenceParams:
    """Nonnegative weights for persistence and constant reward probe."""

    persistence_weight: float = 1.0
    probe_weight: float = 1.0

    def __post_init__(self) -> None:
        _validate_weight("persistence_weight", self.persistence_weight)
        _validate_weight("probe_weight", self.probe_weight)


@dataclass(frozen=True)
class ExpectancyPersistenceDoubtParams:
    """Configuration for the full expectancy-persistence-doubt exemplar."""

    curve: ExpectancyCurveParams = field(default_factory=ExpectancyCurveParams)
    doubt_lambda: float = 0.5
    persistence_weight: float = 1.0
    expectancy_weight: float = 1.0
    doubt_weight: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.doubt_lambda) or self.doubt_lambda <= 0:
            raise ValueError("doubt_lambda must be finite and greater than zero.")
        _validate_weight("persistence_weight", self.persistence_weight)
        _validate_weight("expectancy_weight", self.expectancy_weight)
        _validate_weight("doubt_weight", self.doubt_weight)


@dataclass(frozen=True)
class ExpectantSwitchingFeatureConfig:
    """Focused configuration for replaying both exemplar models."""

    simple: SimpleProbePersistenceParams = field(
        default_factory=SimpleProbePersistenceParams
    )
    full: ExpectancyPersistenceDoubtParams = field(
        default_factory=ExpectancyPersistenceDoubtParams
    )


@dataclass(frozen=True)
class PreviousOutcomeState:
    """Previous valid choice and binary outcome for a single session.

    ``previous_choice`` is ``None`` initially, otherwise ``0=right`` or
    ``1=left``. ``previous_reward`` is a binary indicator.
    """

    previous_choice: int | None = None
    previous_reward: int = 0


@dataclass(frozen=True)
class RewardConfirmedExpectancyState:
    """Most recently rewarded side and reward count for its current episode."""

    confirmed_side: int | None = None
    reward_count: int = 0


@dataclass(frozen=True)
class ExemplarReadout:
    """One pre-trial exemplar readout using left-positive encoding."""

    decision_drive: float
    signed_value: float
    p_right: float
    p_left: float


def expectancy_strength(
    reward_count: int | float | np.ndarray,
    params: ExpectancyCurveParams,
) -> float | np.ndarray:
    """Map reward count to nonnegative saturating expectancy.

    Parameters
    ----------
    reward_count : int, float, or ndarray
        Nonnegative count(s), in rewarded trials. Array shape is preserved.
    params : ExpectancyCurveParams
        Threshold and positive scale, both in rewarded trials.

    Returns
    -------
    float or ndarray
        Expectancy in ``[0, 1]`` with the input shape.
    """
    counts = np.asarray(reward_count, dtype=float)
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("reward_count must be finite and nonnegative.")
    strength = (1.0 + np.tanh((counts - params.threshold) / params.scale)) / 2.0
    if strength.ndim == 0:
        return float(strength)
    return strength


def previous_outcome_signals(state: PreviousOutcomeState) -> tuple[float, float]:
    """Return persistence and reward-triggered probe signals.

    Parameters
    ----------
    state : PreviousOutcomeState
        History through the previous valid trial.

    Returns
    -------
    persistence, probe : tuple[float, float]
        Left-positive signed values. Both are zero before any valid choice;
        probe is exactly zero unless the preceding valid trial was rewarded.
    """
    if state.previous_choice is None:
        return 0.0, 0.0
    choice_sign = 1.0 if state.previous_choice == 1 else -1.0
    return choice_sign, -choice_sign * state.previous_reward


def update_previous_outcome(
    state: PreviousOutcomeState,
    action: int,
    reward: float,
) -> PreviousOutcomeState:
    """Return state advanced by one valid action/outcome observation.

    ``action`` is ``0=right`` or ``1=left``; positive reward is encoded as one
    and zero or negative reward as zero. Reward units are task reward units.
    """
    del state
    _validate_valid_observation(action, reward)
    return PreviousOutcomeState(action, int(reward > 0))


def update_reward_confirmed_expectancy(
    state: RewardConfirmedExpectancyState,
    action: int,
    reward: float,
) -> RewardConfirmedExpectancyState:
    """Advance reward-confirmed episode memory by one valid observation.

    Omissions preserve both fields. A reward on the confirmed side increments
    the count; a reward on another side starts a new episode at one.
    """
    _validate_valid_observation(action, reward)
    if reward <= 0:
        return state
    if state.confirmed_side == action:
        return RewardConfirmedExpectancyState(action, state.reward_count + 1)
    return RewardConfirmedExpectancyState(action, 1)


def _validate_valid_observation(action: int, reward: float) -> None:
    """Validate a parsed valid trial observation."""
    if action not in (0, 1):
        raise ValueError("action must be 0 (right) or 1 (left).")
    if not np.isfinite(reward):
        raise ValueError("reward must be finite for a valid trial.")


def _readout_from_drive(decision_drive: float) -> ExemplarReadout:
    """Convert a finite drive into bounded value and direct probabilities."""
    if not np.isfinite(decision_drive):
        raise ValueError("decision_drive must be finite.")
    signed_value = float(np.tanh(decision_drive))
    p_left = (1.0 + signed_value) / 2.0
    return ExemplarReadout(
        decision_drive=float(decision_drive),
        signed_value=signed_value,
        p_right=1.0 - p_left,
        p_left=p_left,
    )


def read_simple_probe_persistence(
    persistence_signal: float,
    probe_signal: float,
    params: SimpleProbePersistenceParams,
) -> ExemplarReadout:
    """Compose the simple exemplar from left-positive scalar signals."""
    drive = (
        params.persistence_weight * persistence_signal
        + params.probe_weight * probe_signal
    )
    return _readout_from_drive(drive)


def read_expectancy_persistence_doubt(
    persistence_signal: float,
    expectant_switch_signal: float,
    doubt_choice_signal: float,
    params: ExpectancyPersistenceDoubtParams,
) -> ExemplarReadout:
    """Compose the full exemplar from left-positive scalar signals."""
    drive = (
        params.persistence_weight * persistence_signal
        + params.expectancy_weight * expectant_switch_signal
        + params.doubt_weight * doubt_choice_signal
    )
    return _readout_from_drive(drive)


def replay_expectant_switching(
    actions: np.ndarray,
    rewards: np.ndarray,
    valid_mask: np.ndarray,
    config: ExpectantSwitchingFeatureConfig,
) -> dict[str, np.ndarray]:
    """Replay both exemplars over one independently processed session.

    Parameters
    ----------
    actions : ndarray, shape (n_trials,)
        Choices encoded ``0=right`` and ``1=left`` on valid rows. Invalid-row
        values are ignored.
    rewards : ndarray, shape (n_trials,)
        Reward amounts in task units. Positive means rewarded. Invalid-row
        values are ignored.
    valid_mask : ndarray of bool, shape (n_trials,)
        Rows with a valid choice/outcome pair. False rows neither emit values
        nor update history.
    config : ExpectantSwitchingFeatureConfig
        Focused parameters for both models.

    Returns
    -------
    dict[str, ndarray]
        Named float arrays of shape ``(n_trials,)``. Values are pre-trial;
        invalid rows are ``NaN``. Signed columns are left-positive.
    """
    action_array = np.asarray(actions)
    reward_array = np.asarray(rewards, dtype=float)
    validity = np.asarray(valid_mask, dtype=bool)
    if action_array.ndim != 1 or reward_array.ndim != 1 or validity.ndim != 1:
        raise ValueError("actions, rewards, and valid_mask must be one-dimensional.")
    if not (action_array.shape == reward_array.shape == validity.shape):
        raise ValueError("actions, rewards, and valid_mask must have matching shapes.")

    column_names = (
        "previous_choice",
        "previous_reward",
        "reward_triggered_probe",
        "expectancy_confirmed_side",
        "expectancy_reward_count",
        "expectancy_strength",
        "expectant_switch",
        "doubt_raw_value",
        "doubt_choice_signal",
        "simple_probe_persistence_drive",
        "simple_probe_persistence_value",
        "simple_probe_persistence_prob_left",
        "expectancy_persistence_doubt_drive",
        "expectancy_persistence_doubt_value",
        "expectancy_persistence_doubt_prob_left",
    )
    outputs = {
        name: np.full(action_array.shape, np.nan, dtype=float) for name in column_names
    }
    previous_state = PreviousOutcomeState()
    expectancy_state = RewardConfirmedExpectancyState()
    doubt_state = CounterfactualDoubtState()

    for trial_index in np.flatnonzero(validity):
        action = int(action_array[trial_index])
        reward = float(reward_array[trial_index])
        _validate_valid_observation(action, reward)

        persistence, probe = previous_outcome_signals(previous_state)
        strength = float(expectancy_strength(expectancy_state.reward_count, config.full.curve))
        expectant_switch = probe * strength
        doubt_raw = counterfactual_doubt_raw_value(
            doubt_state, config.full.doubt_lambda
        )
        doubt_choice = -doubt_raw
        simple = read_simple_probe_persistence(persistence, probe, config.simple)
        full = read_expectancy_persistence_doubt(
            persistence,
            expectant_switch,
            doubt_choice,
            config.full,
        )

        outputs["previous_choice"][trial_index] = persistence
        outputs["previous_reward"][trial_index] = previous_state.previous_reward
        outputs["reward_triggered_probe"][trial_index] = probe
        if expectancy_state.confirmed_side is not None:
            outputs["expectancy_confirmed_side"][trial_index] = expectancy_state.confirmed_side
        outputs["expectancy_reward_count"][trial_index] = expectancy_state.reward_count
        outputs["expectancy_strength"][trial_index] = strength
        outputs["expectant_switch"][trial_index] = expectant_switch
        outputs["doubt_raw_value"][trial_index] = doubt_raw
        outputs["doubt_choice_signal"][trial_index] = doubt_choice
        outputs["simple_probe_persistence_drive"][trial_index] = simple.decision_drive
        outputs["simple_probe_persistence_value"][trial_index] = simple.signed_value
        outputs["simple_probe_persistence_prob_left"][trial_index] = simple.p_left
        outputs["expectancy_persistence_doubt_drive"][trial_index] = full.decision_drive
        outputs["expectancy_persistence_doubt_value"][trial_index] = full.signed_value
        outputs["expectancy_persistence_doubt_prob_left"][trial_index] = full.p_left

        previous_state = update_previous_outcome(previous_state, action, reward)
        expectancy_state = update_reward_confirmed_expectancy(
            expectancy_state, action, reward
        )
        doubt_state = update_counterfactual_doubt(doubt_state, action, reward)

    return outputs
