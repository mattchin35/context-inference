"""Deterministic model adapters for prescribed behavior benchmark samples.

All replay outputs are pre-trial and left-positive. Ordinary models receive
only action, reward, and validity histories; the hidden-context oracle is a
separate function with a deliberately different interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from scipy.special import expit

from src.behavior_modeling.agents.agents import HMMRewardDecayRelativeDoubt
from src.behavior_modeling.counterfactual_doubt import (
    CounterfactualDoubtState,
    counterfactual_doubt_raw_value,
    update_counterfactual_doubt,
)
from src.behavior_modeling.expectant_switching import (
    ExpectancyCurveParams,
    ExpectancyPersistenceDoubtParams,
    ExpectantSwitchingFeatureConfig,
    PreviousOutcomeState,
    RewardConfirmedExpectancyState,
    SimpleProbePersistenceParams,
    expectancy_strength,
    previous_outcome_signals,
    replay_expectant_switching,
    update_previous_outcome,
    update_reward_confirmed_expectancy,
)
from src.behavior_modeling.parameters.task_config import AgentParams, TaskParams


ORACLE_MODEL_ID = "oracle_correct_side"
OBSERVABLE_MODEL_IDS = (
    "chance",
    "immediate_persistence",
    "wsls",
    "win_switch_lose_stay",
    "simple_probe_persistence",
    "expectancy_persistence_doubt",
    "history_ideal_observer",
    "forgetting_q_learning",
    "probe_persistence_doubt",
    "direct_expectancy_switch_doubt",
    "global_reward_expectancy_doubt",
    "home_referenced_expectancy_doubt",
    "threshold_probe_return_3_3",
    "threshold_probe_return_1_1",
)


@dataclass(frozen=True)
class ObservableHistory:
    """Causally observable aligned history for one independent sample.

    Parameters
    ----------
    action : ndarray, shape (n_trials,)
        Choices with ``0=right`` and ``1=left`` on valid rows.
    reward : ndarray, shape (n_trials,)
        Rewards in task units; positive is rewarded and nonpositive is omission.
    valid : ndarray of bool, shape (n_trials,)
        Rows containing a valid completed choice/outcome pair.
    """

    action: np.ndarray
    reward: np.ndarray
    valid: np.ndarray

    def __post_init__(self) -> None:
        """Copy, validate, and freeze aligned observable arrays."""
        arrays = {
            "action": np.array(self.action, dtype=np.int8, copy=True),
            "reward": np.array(self.reward, dtype=float, copy=True),
            "valid": np.array(self.valid, dtype=bool, copy=True),
        }
        if any(values.ndim != 1 for values in arrays.values()):
            raise ValueError("Observable history arrays must be one-dimensional.")
        if not (arrays["action"].shape == arrays["reward"].shape == arrays["valid"].shape):
            raise ValueError("Observable history arrays must have matching shapes.")
        valid = arrays["valid"]
        if np.any(~np.isin(arrays["action"][valid], (0, 1))):
            raise ValueError("Valid actions must be 0 (right) or 1 (left).")
        if np.any(~np.isfinite(arrays["reward"][valid])):
            raise ValueError("Valid rewards must be finite.")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ModelReplay:
    """Aligned immutable pre-trial outputs for one model and one sample.

    Numeric arrays have shape ``(n_trials,)`` and are unitless. Signed outputs
    are left-positive. ``decision_drive`` is absent for models defined directly
    by choice probability.
    """

    model_id: str
    decision_drive: np.ndarray | None
    model_value: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    p_switch: np.ndarray
    exact_tie: np.ndarray
    internal_states: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        """Defensively copy and freeze common outputs and internal states."""
        if not self.model_id:
            raise ValueError("model_id must be nonempty.")
        normalized = {
            "model_value": np.array(self.model_value, dtype=float, copy=True),
            "p_left": np.array(self.p_left, dtype=float, copy=True),
            "p_right": np.array(self.p_right, dtype=float, copy=True),
            "p_switch": np.array(self.p_switch, dtype=float, copy=True),
            "exact_tie": np.array(self.exact_tie, dtype=bool, copy=True),
        }
        expected_shape = normalized["model_value"].shape
        if len(expected_shape) != 1:
            raise ValueError("Replay arrays must be one-dimensional.")
        if any(values.shape != expected_shape for values in normalized.values()):
            raise ValueError("Replay arrays must have matching shapes.")
        for name, values in normalized.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)

        if self.decision_drive is not None:
            drive = np.array(self.decision_drive, dtype=float, copy=True)
            if drive.shape != expected_shape:
                raise ValueError("decision_drive must align with replay outputs.")
            drive.setflags(write=False)
            object.__setattr__(self, "decision_drive", drive)

        frozen_states: dict[str, np.ndarray] = {}
        for state_name, state_values in self.internal_states.items():
            state = np.array(state_values, copy=True)
            if state.shape != expected_shape:
                raise ValueError(f"Internal state {state_name!r} is not aligned.")
            state.setflags(write=False)
            frozen_states[state_name] = state
        object.__setattr__(self, "internal_states", MappingProxyType(frozen_states))


ReplayFunction = Callable[[ObservableHistory], ModelReplay]


@dataclass(frozen=True)
class BenchmarkModelSpec:
    """One fixed model configuration and its deterministic replay function."""

    model_id: str
    display_label: str
    parameters: Mapping[str, float | int | str]
    replay: ReplayFunction

    def __post_init__(self) -> None:
        """Freeze a copied parameter record and validate identifying fields."""
        if not self.model_id or not self.display_label:
            raise ValueError("Model ID and display label must be nonempty.")
        if not callable(self.replay):
            raise ValueError("replay must be callable.")
        object.__setattr__(
            self,
            "parameters",
            MappingProxyType(dict(self.parameters)),
        )


def build_observable_models(
    parameters: Mapping[str, Mapping[str, float | int | str]] | None = None,
) -> tuple[BenchmarkModelSpec, ...]:
    """Construct the 14 canonical models restricted to observable history.

    Parameters
    ----------
    parameters : mapping or None
        Complete resolved parameter mapping for all 15 model IDs. ``None`` uses
        canonical defaults. Override validation occurs at the runner boundary.

    Returns
    -------
    tuple of BenchmarkModelSpec
        Ordered fresh specifications for the 14 observable models. Replay
        calls create new state, so specifications may be reused safely across
        independent samples.
    """
    resolved = _complete_parameter_mapping(parameters)
    simple_parameters = resolved["simple_probe_persistence"]
    full_parameters = resolved["expectancy_persistence_doubt"]
    exemplar_config = ExpectantSwitchingFeatureConfig(
        simple=SimpleProbePersistenceParams(
            persistence_weight=float(simple_parameters["persistence_weight"]),
            probe_weight=float(simple_parameters["probe_weight"]),
        ),
        full=ExpectancyPersistenceDoubtParams(
            curve=ExpectancyCurveParams(
                threshold=float(full_parameters["expectancy_threshold"]),
                scale=float(full_parameters["expectancy_scale"]),
            ),
            doubt_lambda=float(full_parameters["doubt_lambda"]),
            persistence_weight=float(full_parameters["persistence_weight"]),
            expectancy_weight=float(full_parameters["expectancy_weight"]),
            doubt_weight=float(full_parameters["doubt_weight"]),
        ),
    )
    observer_parameters = resolved["history_ideal_observer"]
    forgetting_q_parameters = resolved["forgetting_q_learning"]
    probe_doubt_parameters = resolved["probe_persistence_doubt"]
    direct_parameters = resolved["direct_expectancy_switch_doubt"]
    global_parameters = resolved["global_reward_expectancy_doubt"]
    home_expectancy_parameters = resolved["home_referenced_expectancy_doubt"]
    return (
        _spec("chance", "Chance", {}, _replay_chance),
        _spec(
            "immediate_persistence",
            "Immediate persistence",
            {},
            _replay_immediate_persistence,
        ),
        _spec("wsls", "Win-stay, lose-switch", {}, _replay_wsls),
        _spec(
            "win_switch_lose_stay",
            "Win-switch, lose-stay",
            {},
            _replay_win_switch_lose_stay,
        ),
        _spec(
            "simple_probe_persistence",
            "Probe + persistence",
            simple_parameters,
            lambda history: _replay_existing_exemplar(
                history, "simple_probe_persistence", exemplar_config
            ),
        ),
        _spec(
            "expectancy_persistence_doubt",
            "Expectancy + persistence + doubt",
            full_parameters,
            lambda history: _replay_existing_exemplar(
                history, "expectancy_persistence_doubt", exemplar_config
            ),
        ),
        _spec(
            "history_ideal_observer",
            "Mouse-history ideal observer",
            observer_parameters,
            lambda history: _replay_history_observer(history, observer_parameters),
        ),
        _spec(
            "forgetting_q_learning",
            "Forgetting Q-learning",
            forgetting_q_parameters,
            lambda history: _replay_forgetting_q(
                history, forgetting_q_parameters
            ),
        ),
        _spec(
            "probe_persistence_doubt",
            "Probe + persistence + doubt",
            probe_doubt_parameters,
            lambda history: _replay_probe_persistence_doubt(
                history, probe_doubt_parameters
            ),
        ),
        _spec(
            "direct_expectancy_switch_doubt",
            "Direct expectancy switch + doubt",
            direct_parameters,
            lambda history: _replay_direct_expectancy(history, direct_parameters),
        ),
        _spec(
            "global_reward_expectancy_doubt",
            "Global reward expectancy + doubt",
            global_parameters,
            lambda history: _replay_global_expectancy(history, global_parameters),
        ),
        _spec(
            "home_referenced_expectancy_doubt",
            "Home-referenced expectancy + doubt",
            home_expectancy_parameters,
            partial(
                _replay_home_referenced_expectancy_doubt,
                parameters=home_expectancy_parameters,
            ),
        ),
        _spec(
            "threshold_probe_return_3_3",
            "Threshold probe/return (3, 3)",
            resolved["threshold_probe_return_3_3"],
            lambda history: _replay_threshold(
                history, 3, 3, "threshold_probe_return_3_3"
            ),
        ),
        _spec(
            "threshold_probe_return_1_1",
            "Threshold probe/return (1, 1)",
            resolved["threshold_probe_return_1_1"],
            lambda history: _replay_threshold(
                history, 1, 1, "threshold_probe_return_1_1"
            ),
        ),
    )


def replay_oracle(
    history: ObservableHistory,
    correct_side: np.ndarray,
) -> ModelReplay:
    """Replay the hidden-context performance ceiling.

    Parameters
    ----------
    history : ObservableHistory
        Observable action history used only to define switch probability
        relative to the previous valid choice.
    correct_side : ndarray, shape (n_trials,)
        Hidden task context with ``0=right`` and ``1=left`` on valid rows.

    Returns
    -------
    ModelReplay
        Deterministic correct-side probabilities. The decision drive is absent.
    """
    sides = np.asarray(correct_side)
    validity = history.valid
    if sides.ndim != 1 or validity.ndim != 1 or sides.shape != validity.shape:
        raise ValueError("correct_side must align with the observable history.")
    if np.any(~np.isin(sides[validity], (0, 1))):
        raise ValueError("Valid correct sides must be 0 (right) or 1 (left).")
    p_left = np.full(sides.shape, np.nan, dtype=float)
    p_left[validity] = sides[validity]
    return _finalize(
        ORACLE_MODEL_ID,
        action=history.action,
        valid=validity,
        p_left=p_left,
        decision_drive=None,
        internal_states={},
    )


# ---------------------------------------------------------------------------
# Reference and existing adapters


def _replay_chance(history: ObservableHistory) -> ModelReplay:
    p_left = _empty_output(history)
    p_left[history.valid] = 0.5
    return _finalize("chance", history.action, history.valid, p_left, None, {})


def _replay_immediate_persistence(history: ObservableHistory) -> ModelReplay:
    previous_choice, previous_reward = _previous_history(history)
    value = _empty_output(history)
    value[history.valid] = previous_choice[history.valid]
    return _finalize_from_value(
        "immediate_persistence",
        history,
        value,
        {"previous_choice": previous_choice, "previous_reward": previous_reward},
    )


def _replay_wsls(history: ObservableHistory) -> ModelReplay:
    previous_choice, previous_reward = _previous_history(history)
    value = _empty_output(history)
    for index in np.flatnonzero(history.valid):
        choice = previous_choice[index]
        if choice == 0:
            value[index] = 0.0
        else:
            value[index] = choice if previous_reward[index] > 0 else -choice
    return _finalize_from_value(
        "wsls",
        history,
        value,
        {"previous_choice": previous_choice, "previous_reward": previous_reward},
    )


def _replay_win_switch_lose_stay(history: ObservableHistory) -> ModelReplay:
    previous_choice, previous_reward = _previous_history(history)
    value = _empty_output(history)
    for index in np.flatnonzero(history.valid):
        choice = previous_choice[index]
        if choice == 0:
            value[index] = 0.0
        else:
            value[index] = -choice if previous_reward[index] > 0 else choice
    return _finalize_from_value(
        "win_switch_lose_stay",
        history,
        value,
        {"previous_choice": previous_choice, "previous_reward": previous_reward},
    )


def _replay_existing_exemplar(
    history: ObservableHistory,
    model_id: str,
    config: ExpectantSwitchingFeatureConfig,
) -> ModelReplay:
    outputs = replay_expectant_switching(
        history.action, history.reward, history.valid, config
    )
    prefix = model_id
    drive = outputs[f"{prefix}_drive"]
    p_left = outputs[f"{prefix}_prob_left"]
    excluded = {
        "simple_probe_persistence_drive",
        "simple_probe_persistence_value",
        "simple_probe_persistence_prob_left",
        "expectancy_persistence_doubt_drive",
        "expectancy_persistence_doubt_value",
        "expectancy_persistence_doubt_prob_left",
    }
    states = {name: values for name, values in outputs.items() if name not in excluded}
    return _finalize(
        model_id,
        history.action,
        history.valid,
        p_left,
        drive,
        states,
    )


def _replay_history_observer(
    history: ObservableHistory,
    parameters: Mapping[str, float | int | str],
) -> ModelReplay:
    agent = HMMRewardDecayRelativeDoubt(
        AgentParams(
            HMM_transition_prob=float(
                parameters["context_transition_probability"]
            ),
            HMM_value_mode="bayesian_log_odds",
            HMM_log_odds_tanh_scale=float(
                parameters["log_odds_tanh_scale"]
            ),
            HMM_reward_decay_lambda=float(parameters["reward_decay_lambda"]),
            relative_doubt_lambda=float(parameters["doubt_lambda"]),
            HMM_active_reward_probability=float(
                parameters["active_reward_probability"]
            ),
            HMM_inactive_reward_probability=float(
                parameters["inactive_reward_probability"]
            ),
            logistic_alpha=float(parameters["action_history_bias"]),
            action_temperature=float(parameters["action_temperature"]),
        ),
        TaskParams(mean_correct_reward=1.0, mean_incorrect_reward=0.0),
        rng=np.random.default_rng(0),
    )
    drive = _empty_output(history)
    p_left = _empty_output(history)
    hmm_value = _empty_output(history)
    doubt_raw = _empty_output(history)
    belief_left = _empty_output(history)
    for index in np.flatnonzero(history.valid):
        drive[index] = agent.value
        p_left[index] = agent.action_dist[1]
        hmm_value[index] = agent.hmm_value
        doubt_raw[index] = agent.doubt_value
        belief_left[index] = agent.prior[1]
        agent.update_params(int(history.action[index]), float(history.reward[index]))
    return _finalize(
        "history_ideal_observer",
        history.action,
        history.valid,
        p_left,
        drive,
        {
            "hmm_value": hmm_value,
            "doubt_raw_value": doubt_raw,
            "belief_left": belief_left,
        },
    )


def _replay_forgetting_q(
    history: ObservableHistory,
    parameters: Mapping[str, float | int | str],
) -> ModelReplay:
    drive = _empty_output(history)
    p_left = _empty_output(history)
    q_right_values = _empty_output(history)
    q_left_values = _empty_output(history)
    q_values = np.zeros(2, dtype=float)
    last_action_sign = 0.0
    for index in np.flatnonzero(history.valid):
        q_right_values[index], q_left_values[index] = q_values
        drive[index] = q_values[1] - q_values[0]
        p_left[index] = expit(
            float(parameters["action_history_bias"]) * last_action_sign
            + drive[index] / float(parameters["action_temperature"])
        )
        q_values *= float(parameters["decay"])
        action = int(history.action[index])
        q_values[action] += float(parameters["reward_update_rate"]) * float(
            history.reward[index]
        )
        last_action_sign = 1.0 if action == 1 else -1.0
    return _finalize(
        "forgetting_q_learning",
        history.action,
        history.valid,
        p_left,
        drive,
        {"q_right": q_right_values, "q_left": q_left_values},
    )


# ---------------------------------------------------------------------------
# Proposed adapters


def _replay_probe_persistence_doubt(
    history: ObservableHistory,
    parameters: Mapping[str, float | int | str],
) -> ModelReplay:
    previous = PreviousOutcomeState()
    doubt = CounterfactualDoubtState()
    drive = _empty_output(history)
    p_left = _empty_output(history)
    states = _state_outputs(
        history,
        "previous_choice",
        "previous_reward",
        "reward_probe",
        "doubt_raw_value",
        "doubt_choice",
    )
    for index in np.flatnonzero(history.valid):
        persistence, probe = previous_outcome_signals(previous)
        doubt_raw = counterfactual_doubt_raw_value(
            doubt, float(parameters["doubt_lambda"])
        )
        doubt_choice = -doubt_raw
        drive[index] = (
            float(parameters["persistence_weight"]) * persistence
            + float(parameters["probe_weight"]) * probe
            + float(parameters["doubt_weight"]) * doubt_choice
        )
        p_left[index] = (1.0 + np.tanh(drive[index])) / 2.0
        states["previous_choice"][index] = persistence
        states["previous_reward"][index] = previous.previous_reward
        states["reward_probe"][index] = probe
        states["doubt_raw_value"][index] = doubt_raw
        states["doubt_choice"][index] = doubt_choice
        action = int(history.action[index])
        reward = float(history.reward[index])
        previous = update_previous_outcome(previous, action, reward)
        doubt = update_counterfactual_doubt(doubt, action, reward)
    return _finalize(
        "probe_persistence_doubt",
        history.action,
        history.valid,
        p_left,
        drive,
        states,
    )


def _replay_direct_expectancy(
    history: ObservableHistory,
    parameters: Mapping[str, float | int | str],
) -> ModelReplay:
    previous = PreviousOutcomeState()
    expectancy = RewardConfirmedExpectancyState()
    doubt = CounterfactualDoubtState()
    p_left = _empty_output(history)
    states = _state_outputs(
        history,
        "previous_choice",
        "previous_reward",
        "expectancy_confirmed_side",
        "expectancy_reward_count",
        "expectancy_strength",
        "doubt_raw_value",
        "omission_drive",
    )
    curve = ExpectancyCurveParams(
        threshold=float(parameters["expectancy_threshold"]),
        scale=float(parameters["expectancy_scale"]),
    )
    for index in np.flatnonzero(history.valid):
        persistence, _ = previous_outcome_signals(previous)
        strength = float(expectancy_strength(expectancy.reward_count, curve))
        doubt_raw = counterfactual_doubt_raw_value(
            doubt, float(parameters["doubt_lambda"])
        )
        if previous.previous_choice is None:
            p_left[index] = 0.5
            omission_drive = np.nan
        elif previous.previous_reward > 0:
            p_switch = float(parameters["minimum_switch_probability"]) + float(
                parameters["switch_probability_span"]
            ) * strength
            p_left[index] = _left_probability_from_switch(persistence, p_switch)
            omission_drive = np.nan
        else:
            omission_drive = (
                float(parameters["omission_persistence_weight"]) * persistence
                + float(parameters["doubt_weight"]) * (-doubt_raw)
            )
            p_left[index] = (1.0 + np.tanh(omission_drive)) / 2.0
        _record_expectancy_states(
            states, index, previous, expectancy, strength, doubt_raw, omission_drive
        )
        action = int(history.action[index])
        reward = float(history.reward[index])
        previous = update_previous_outcome(previous, action, reward)
        expectancy = update_reward_confirmed_expectancy(expectancy, action, reward)
        doubt = update_counterfactual_doubt(doubt, action, reward)
    return _finalize(
        "direct_expectancy_switch_doubt",
        history.action,
        history.valid,
        p_left,
        None,
        states,
    )


def _replay_global_expectancy(
    history: ObservableHistory,
    parameters: Mapping[str, float | int | str],
) -> ModelReplay:
    previous = PreviousOutcomeState()
    doubt = CounterfactualDoubtState()
    reward_accumulator = 0.0
    p_left = _empty_output(history)
    states = _state_outputs(
        history,
        "previous_choice",
        "previous_reward",
        "reward_accumulator",
        "expectancy_strength",
        "doubt_raw_value",
        "omission_drive",
    )
    curve = ExpectancyCurveParams(
        threshold=float(parameters["expectancy_threshold"]),
        scale=float(parameters["expectancy_scale"]),
    )
    for index in np.flatnonzero(history.valid):
        persistence, _ = previous_outcome_signals(previous)
        strength = float(expectancy_strength(reward_accumulator, curve))
        doubt_raw = counterfactual_doubt_raw_value(
            doubt, float(parameters["doubt_lambda"])
        )
        if previous.previous_choice is None:
            p_left[index] = 0.5
            omission_drive = np.nan
        elif previous.previous_reward > 0:
            p_switch = float(parameters["minimum_switch_probability"]) + float(
                parameters["switch_probability_span"]
            ) * strength
            p_left[index] = _left_probability_from_switch(persistence, p_switch)
            omission_drive = np.nan
        else:
            omission_drive = (
                float(parameters["omission_persistence_weight"]) * persistence
                + float(parameters["doubt_weight"]) * (-doubt_raw)
            )
            p_left[index] = (1.0 + np.tanh(omission_drive)) / 2.0
        states["previous_choice"][index] = persistence
        states["previous_reward"][index] = previous.previous_reward
        states["reward_accumulator"][index] = reward_accumulator
        states["expectancy_strength"][index] = strength
        states["doubt_raw_value"][index] = doubt_raw
        states["omission_drive"][index] = omission_drive
        action = int(history.action[index])
        reward = float(history.reward[index])
        previous = update_previous_outcome(previous, action, reward)
        reward_accumulator = (
            float(parameters["reward_decay"]) * reward_accumulator
            + float(reward > 0)
        )
        doubt = update_counterfactual_doubt(doubt, action, reward)
    return _finalize(
        "global_reward_expectancy_doubt",
        history.action,
        history.valid,
        p_left,
        None,
        states,
    )


def _replay_home_referenced_expectancy_doubt(
    history: ObservableHistory,
    parameters: Mapping[str, float | int | str],
) -> ModelReplay:
    """Replay persistent signed expectancy plus omission doubt.

    Parameters
    ----------
    history : ObservableHistory
        Aligned one-dimensional action, reward, and validity arrays. Actions
        use ``0=right, 1=left`` and rewards use task reward units.
    parameters : mapping
        Fixed threshold, scale, weights, and inverse-trial doubt rate for the
        complete independent sample.

    Returns
    -------
    ModelReplay
        Aligned pre-trial probabilities, left-positive values and drive, and
        component states. Every array has shape ``(n_trials,)``.
    """
    expectancy = RewardConfirmedExpectancyState()
    doubt = CounterfactualDoubtState()
    drive = _empty_output(history)
    p_left = _empty_output(history)
    states = _state_outputs(
        history,
        "expectancy_confirmed_side",
        "expectancy_reward_count",
        "expectancy_strength",
        "expectancy_choice",
        "doubt_raw_value",
        "doubt_choice",
    )
    curve = ExpectancyCurveParams(
        threshold=float(parameters["expectancy_threshold"]),
        scale=float(parameters["expectancy_scale"]),
    )
    expectancy_weight = float(parameters["expectancy_weight"])
    doubt_weight = float(parameters["doubt_weight"])
    doubt_lambda = float(parameters["doubt_lambda"])

    for index in np.flatnonzero(history.valid):
        strength = float(expectancy_strength(expectancy.reward_count, curve))
        if expectancy.confirmed_side is None:
            confirmed_side = np.nan
            expectancy_choice = 0.0
        else:
            confirmed_side = float(expectancy.confirmed_side)
            home_sign = 1.0 if expectancy.confirmed_side == 1 else -1.0
            expectancy_choice = home_sign * (1.0 - 2.0 * strength)
        doubt_raw = counterfactual_doubt_raw_value(doubt, doubt_lambda)
        doubt_choice = -doubt_raw
        drive[index] = (
            expectancy_weight * expectancy_choice + doubt_weight * doubt_choice
        )
        p_left[index] = (1.0 + np.tanh(drive[index])) / 2.0

        states["expectancy_confirmed_side"][index] = confirmed_side
        states["expectancy_reward_count"][index] = expectancy.reward_count
        states["expectancy_strength"][index] = strength
        states["expectancy_choice"][index] = expectancy_choice
        states["doubt_raw_value"][index] = doubt_raw
        states["doubt_choice"][index] = doubt_choice

        action = int(history.action[index])
        reward = float(history.reward[index])
        expectancy = update_reward_confirmed_expectancy(expectancy, action, reward)
        doubt = update_counterfactual_doubt(doubt, action, reward)

    return _finalize(
        "home_referenced_expectancy_doubt",
        history.action,
        history.valid,
        p_left,
        drive,
        states,
    )


def _replay_threshold(
    history: ObservableHistory,
    reward_threshold: int,
    omission_threshold: int,
    model_id: str,
) -> ModelReplay:
    home_side: int | None = None
    home_reward_count = 0
    in_excursion = False
    excursion_side: int | None = None
    excursion_omission_count = 0
    previous_reward = 0
    p_left = _empty_output(history)
    states = _state_outputs(
        history,
        "home_side",
        "home_reward_count",
        "in_excursion",
        "excursion_side",
        "excursion_omission_count",
        "previous_reward",
    )
    for index in np.flatnonzero(history.valid):
        if home_side is None:
            p_left[index] = 0.5
        elif in_excursion:
            desired_side = (
                excursion_side
                if excursion_omission_count < omission_threshold
                else home_side
            )
            p_left[index] = 0.9 if desired_side == 1 else 0.1
        else:
            should_probe = (
                previous_reward > 0 and home_reward_count >= reward_threshold
            )
            desired_side = 1 - home_side if should_probe else home_side
            p_left[index] = 0.9 if desired_side == 1 else 0.1

        states["home_side"][index] = np.nan if home_side is None else home_side
        states["home_reward_count"][index] = home_reward_count
        states["in_excursion"][index] = float(in_excursion)
        states["excursion_side"][index] = (
            np.nan if excursion_side is None else excursion_side
        )
        states["excursion_omission_count"][index] = excursion_omission_count
        states["previous_reward"][index] = previous_reward

        action = int(history.action[index])
        reward = float(history.reward[index])
        rewarded = reward > 0
        if home_side is None:
            if rewarded:
                home_side = action
                home_reward_count = 1
        elif not in_excursion:
            if action == home_side:
                if rewarded:
                    home_reward_count += 1
            elif rewarded:
                home_side = action
                home_reward_count = 1
            else:
                in_excursion = True
                excursion_side = action
                excursion_omission_count = 1
                home_reward_count = 0
        elif action == excursion_side:
            if rewarded:
                home_side = action
                home_reward_count = 1
                in_excursion = False
                excursion_side = None
                excursion_omission_count = 0
            else:
                excursion_omission_count += 1
        else:
            in_excursion = False
            excursion_side = None
            excursion_omission_count = 0
            home_reward_count = 1 if rewarded else 0
        previous_reward = int(rewarded)
    return _finalize(
        model_id,
        history.action,
        history.valid,
        p_left,
        None,
        states,
    )


# ---------------------------------------------------------------------------
# Shared replay helpers


def _spec(
    model_id: str,
    display_label: str,
    parameters: Mapping[str, float | int | str],
    replay: ReplayFunction,
) -> BenchmarkModelSpec:
    """Construct one immutable model specification."""
    return BenchmarkModelSpec(model_id, display_label, parameters, replay)


def _empty_output(history: ObservableHistory) -> np.ndarray:
    """Allocate one missing aligned floating-point output array."""
    return np.full(history.action.shape, np.nan, dtype=float)


def _state_outputs(history: ObservableHistory, *names: str) -> dict[str, np.ndarray]:
    """Allocate named missing internal-state arrays aligned to one history."""
    return {name: _empty_output(history) for name in names}


def _previous_history(history: ObservableHistory) -> tuple[np.ndarray, np.ndarray]:
    """Return signed previous-valid-choice and binary-reward arrays."""
    previous_choice = _empty_output(history)
    previous_reward = _empty_output(history)
    last_choice = 0.0
    last_reward = 0.0
    for index in np.flatnonzero(history.valid):
        previous_choice[index] = last_choice
        previous_reward[index] = last_reward
        last_choice = 1.0 if history.action[index] == 1 else -1.0
        last_reward = float(history.reward[index] > 0)
    return previous_choice, previous_reward


def _finalize_from_value(
    model_id: str,
    history: ObservableHistory,
    value: np.ndarray,
    internal_states: Mapping[str, np.ndarray],
) -> ModelReplay:
    """Create a replay for a policy that directly defines signed value."""
    p_left = _empty_output(history)
    p_left[history.valid] = (1.0 + value[history.valid]) / 2.0
    return _finalize(
        model_id,
        history.action,
        history.valid,
        p_left,
        None,
        internal_states,
    )


def _finalize(
    model_id: str,
    action: np.ndarray | None,
    valid: np.ndarray,
    p_left: np.ndarray,
    decision_drive: np.ndarray | None,
    internal_states: Mapping[str, np.ndarray],
) -> ModelReplay:
    """Derive normalized common outputs from aligned left probabilities."""
    validity = np.asarray(valid, dtype=bool)
    left = np.asarray(p_left, dtype=float)
    if left.shape != validity.shape:
        raise ValueError("p_left and valid must have matching shapes.")
    if np.any(~np.isfinite(left[validity])):
        raise ValueError("Valid left probabilities must be finite.")
    if np.any((left[validity] < 0) | (left[validity] > 1)):
        raise ValueError("Valid left probabilities must lie in [0, 1].")
    right = np.full(left.shape, np.nan, dtype=float)
    value = np.full(left.shape, np.nan, dtype=float)
    exact_tie = np.zeros(left.shape, dtype=bool)
    right[validity] = 1.0 - left[validity]
    value[validity] = left[validity] - right[validity]
    exact_tie[validity] = left[validity] == right[validity]

    p_switch = np.full(left.shape, np.nan, dtype=float)
    if action is not None:
        action_values = np.asarray(action)
        last_action: int | None = None
        for index in np.flatnonzero(validity):
            if last_action is not None:
                p_switch[index] = left[index] if last_action == 0 else right[index]
            last_action = int(action_values[index])
    return ModelReplay(
        model_id=model_id,
        decision_drive=decision_drive,
        model_value=value,
        p_left=left,
        p_right=right,
        p_switch=p_switch,
        exact_tie=exact_tie,
        internal_states=internal_states,
    )


def _left_probability_from_switch(previous_choice: float, p_switch: float) -> float:
    """Convert switch probability to left probability using prior choice sign."""
    return float(p_switch if previous_choice < 0 else 1.0 - p_switch)


def _record_expectancy_states(
    states: dict[str, np.ndarray],
    index: int,
    previous: PreviousOutcomeState,
    expectancy: RewardConfirmedExpectancyState,
    strength: float,
    doubt_raw: float,
    omission_drive: float,
) -> None:
    """Store shared direct-expectancy pre-trial state for one aligned row."""
    persistence, _ = previous_outcome_signals(previous)
    states["previous_choice"][index] = persistence
    states["previous_reward"][index] = previous.previous_reward
    states["expectancy_confirmed_side"][index] = (
        np.nan if expectancy.confirmed_side is None else expectancy.confirmed_side
    )
    states["expectancy_reward_count"][index] = expectancy.reward_count
    states["expectancy_strength"][index] = strength
    states["doubt_raw_value"][index] = doubt_raw
    states["omission_drive"][index] = omission_drive


def _history_observer_parameters() -> dict[str, float | str]:
    """Return complete report parameters for the canonical observer."""
    return {
        "context_transition_probability": 0.2,
        "active_reward_probability": 0.8,
        "inactive_reward_probability": 0.0,
        "reward_decay_lambda": 0.1,
        "doubt_lambda": 0.5,
        "log_odds_tanh_scale": 1.0,
        "action_history_bias": 0.0,
        "action_temperature": 1.0,
    }


def _direct_expectancy_parameters() -> dict[str, float]:
    """Return canonical direct-expectancy model parameters."""
    return {
        "expectancy_threshold": 2.5,
        "expectancy_scale": 0.5,
        "minimum_switch_probability": 0.1,
        "switch_probability_span": 0.8,
        "omission_persistence_weight": 1.0,
        "doubt_weight": 1.4,
        "doubt_lambda": 0.5,
    }


def _global_expectancy_parameters() -> dict[str, float]:
    """Return canonical global-reward model parameters."""
    return {
        "reward_decay": 0.8,
        **_direct_expectancy_parameters(),
        "expectancy_threshold": 2.0,
    }


def canonical_model_parameters() -> dict[str, dict[str, float | int | str]]:
    """Return independent canonical parameter dictionaries for all 15 models.

    Returns
    -------
    dict
        Mapping from stable model ID to its complete fixed parameter record.
        Numeric values are unitless unless their model documentation specifies
        trial, inverse-trial, or probability units.
    """
    return {
        "chance": {},
        "immediate_persistence": {},
        "wsls": {},
        "win_switch_lose_stay": {},
        ORACLE_MODEL_ID: {},
        "simple_probe_persistence": {
            "persistence_weight": 1.0,
            "probe_weight": 1.0,
        },
        "expectancy_persistence_doubt": {
            "expectancy_threshold": 3.0,
            "expectancy_scale": 1.0,
            "doubt_lambda": 0.5,
            "persistence_weight": 1.0,
            "expectancy_weight": 1.0,
            "doubt_weight": 1.0,
        },
        "history_ideal_observer": _history_observer_parameters(),
        "forgetting_q_learning": {
            "decay": 0.7,
            "reward_update_rate": 0.3,
            "action_history_bias": 0.0,
            "action_temperature": 1.0,
        },
        "probe_persistence_doubt": {
            "persistence_weight": 1.0,
            "probe_weight": 2.0,
            "doubt_weight": 1.4,
            "doubt_lambda": 0.5,
        },
        "direct_expectancy_switch_doubt": _direct_expectancy_parameters(),
        "global_reward_expectancy_doubt": _global_expectancy_parameters(),
        "home_referenced_expectancy_doubt": {
            "expectancy_threshold": 2.5,
            "expectancy_scale": 0.5,
            "expectancy_weight": 1.0,
            "doubt_weight": 1.0,
            "doubt_lambda": 0.5,
        },
        "threshold_probe_return_3_3": {
            "reward_threshold": 3,
            "omission_threshold": 3,
        },
        "threshold_probe_return_1_1": {
            "reward_threshold": 1,
            "omission_threshold": 1,
        },
    }


def _complete_parameter_mapping(
    parameters: Mapping[str, Mapping[str, float | int | str]] | None,
) -> dict[str, dict[str, float | int | str]]:
    """Copy a complete resolved mapping or return fresh canonical defaults."""
    canonical = canonical_model_parameters()
    if parameters is None:
        return canonical
    if set(parameters) != set(canonical):
        missing = sorted(set(canonical).difference(parameters))
        extra = sorted(set(parameters).difference(canonical))
        raise ValueError(
            "Resolved model parameters must contain every canonical model ID; "
            f"missing={missing}, extra={extra}."
        )
    resolved: dict[str, dict[str, float | int | str]] = {}
    for model_id, canonical_values in canonical.items():
        supplied_values = dict(parameters[model_id])
        if set(supplied_values) != set(canonical_values):
            raise ValueError(
                f"Resolved parameters for {model_id!r} do not match the "
                "canonical field set."
            )
        resolved[model_id] = supplied_values
    return resolved
