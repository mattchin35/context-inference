"""Prepare one observed session and replay causal, left-positive trial policies.

Only the input boundary knows about dataframes. Adapters reuse the existing
pre-choice feature implementations, with common history exclusions normalized
locally so legacy feature behavior is unchanged.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from src.behavior_analysis import trial_features
from src.behavior_analysis.project_utils import MISSING_STRINGS, NO_CHOICE_ACTION_LABELS
from src.behavior_modeling.counterfactual_doubt import (
    CounterfactualDoubtState,
    counterfactual_doubt_raw_value,
    update_counterfactual_doubt,
)

if TYPE_CHECKING:
    from src.behavior_analysis.gather_trial_features import TaskParams


DEFAULT_MODEL_IDS = (
    "forgetting_q_learning", "bayesian_hmm", "win_stay_lose_switch",
    "q_learning", "reward_decay_hmm", "ideal_observer",
)
FLAG_COLUMN_NAMES = ("give_reward", "experimenter_reward_given")
_BELIEF_PARAMETER_NAMES = (
    "state_transition_prob", "active_reward_probability", "inactive_reward_probability",
    "correct_reward_size", "incorrect_reward_size", "tanh_scale",
)


@dataclass(frozen=True)
class PreparedTrialHistory:
    """Data-only history in retained chronological row order.

    actions/rewards/manual_reward_flags are float64[T], dimensionless binary
    codes with NaN for missing/unknown values; actions use 0=Right, 1=Left.
    adapter_skip_flags and eligibility arrays are bool[T]. Reasons and original
    display identifiers are Unicode[T]; row_position is int64[T], zero-based.
    metadata is a plain provenance dictionary including schema-bearing digest.
    """

    session_id: str
    actions: np.ndarray
    rewards: np.ndarray
    manual_reward_flags: np.ndarray
    adapter_skip_flags: np.ndarray
    comparison_valid: np.ndarray
    history_update_valid: np.ndarray
    comparison_exclusion_reason: np.ndarray
    history_update_exclusion_reason: np.ndarray
    trial_id: np.ndarray
    block_id: np.ndarray
    row_position: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TrialPolicyCandidate:
    """Immutable policy definition with no learned state or file access.

    adapter receives PreparedTrialHistory and TaskParams and returns aligned
    dimensionless left-positive float64[T] scores and bool[T] availability.
    parameter_names records TaskParams attributes used by the adapter; labels
    and indifference_rule are strings for plain saved provenance.
    """

    model_id: str
    label: str
    adapter: Callable[[PreparedTrialHistory, "TaskParams"], tuple[np.ndarray, np.ndarray]]
    parameter_names: tuple[str, ...] = ()
    indifference_rule: str = "exact_zero"


# Public computational functions.
def prepare_trial_policy_history(
    trial_df: pd.DataFrame, *, session_id: str
) -> PreparedTrialHistory:
    """Normalize a single source session without sorting, dropping or mutating rows.

    Parameters
    ----------
    trial_df : pandas.DataFrame, shape (T, C)
        Chronological retained trials. Nonempty tables require action, reward,
        cur_trial, and source_cur_block or cur_block. Observations are binary
        delivery indicators, not water volume; actions use 0=Right, 1=Left.
        Optional trial_outcome labels distinguish choice, manual_reward,
        invalid_switch and unknown without changing eligibility conventions.
        Explicit invalid switches require no choice and zero reward/manual flag.
    session_id : str
        Full source-session ID, checked against any session_ID/source_session_id.

    Returns
    -------
    PreparedTrialHistory
        Aligned (T,) arrays with dimensionless values and original display IDs.
        Missing outcomes freeze updates but preserve current choice eligibility.
    """
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a nonempty full source-session string")
    if not isinstance(trial_df, pd.DataFrame):
        raise ValueError("trial_df must be a pandas DataFrame")
    n_trials = len(trial_df)
    block_column = "source_cur_block" if "source_cur_block" in trial_df else "cur_block"
    if n_trials:
        for column in ("action", "reward", "cur_trial", block_column):
            if column not in trial_df:
                raise ValueError(f"Session {session_id}: missing required column {column}")
    for column in ("session_ID", "source_session_id"):
        if column in trial_df and not trial_df[column].map(str).eq(session_id).all():
            raise ValueError(f"Session {session_id}: {column} contains another or missing session ID")

    trial_ids = trial_df["cur_trial"].map(str).to_numpy(dtype=str) if n_trials else np.array([], dtype=str)
    block_ids = trial_df[block_column].map(str).to_numpy(dtype=str) if n_trials else np.array([], dtype=str)
    actions = np.empty(n_trials, dtype=float)
    rewards = np.empty(n_trials, dtype=float)
    for position in range(n_trials):
        context = f"Session {session_id}, trial {trial_ids[position]}, row {position}"
        actions[position] = _parse_binary_value(trial_df["action"].iloc[position], context, "action")
        rewards[position] = _parse_binary_value(trial_df["reward"].iloc[position], context, "reward")

    source_flag_columns = [column for column in FLAG_COLUMN_NAMES if column in trial_df]
    flags = np.zeros(n_trials, dtype=float)
    if source_flag_columns:
        normalized_flags = []
        for column in source_flag_columns:
            normalized_flags.append(np.array([
                _parse_binary_value(value, "", "manual flag", unknown_allowed=True)
                for value in trial_df[column]
            ]))
        flags = normalized_flags[0].astype(float)
        if len(normalized_flags) == 2 and not np.array_equal(flags, normalized_flags[1], equal_nan=True):
            raise ValueError(f"Session {session_id}: conflicting manual-reward flag aliases")

    no_choice = np.isnan(actions)
    manual = flags == 1
    unknown_flag = np.isnan(flags)
    # Outcome labels add provenance; existing action/flag masks still determine
    # eligibility. Legacy missing outcomes are never inferred to be switches.
    outcomes = np.full(n_trials, "unknown", dtype=object)
    outcomes[~no_choice & (flags == 0)] = "choice"
    outcomes[manual] = "manual_reward"
    if "trial_outcome" in trial_df:
        outcomes = trial_df["trial_outcome"].fillna("unknown").astype(str).to_numpy()
        outcomes[np.isin(outcomes, list(MISSING_STRINGS))] = "unknown"
        unsupported = ~np.isin(outcomes, ["unknown", "choice", "manual_reward", "invalid_switch"])
        if unsupported.any():
            raise ValueError(f"Session {session_id}: unsupported trial_outcome {outcomes[unsupported][0]!r}")
    invalid_switch = outcomes == "invalid_switch"
    inconsistent_switch = invalid_switch & (~no_choice | (flags != 0) | (rewards != 0))
    if inconsistent_switch.any():
        position = np.flatnonzero(inconsistent_switch)[0]
        raise ValueError(f"Session {session_id}, trial {trial_ids[position]}: invalid_switch requires no choice and zero reward/manual flag")
    skip = no_choice | manual | unknown_flag
    missing_reward = np.isnan(rewards)
    comparison_valid = ~skip
    history_update_valid = comparison_valid & ~missing_reward
    comparison_reasons = _exclusion_reasons(no_choice, manual, unknown_flag)
    update_reasons = _exclusion_reasons(no_choice, manual, unknown_flag, missing_reward)
    comparison_reasons = np.array([
        f"invalid_switch;{reason}" if invalid else reason
        for reason, invalid in zip(comparison_reasons, invalid_switch)
    ], dtype=str)
    update_reasons = np.array([
        f"invalid_switch;{reason}" if invalid else reason
        for reason, invalid in zip(update_reasons, invalid_switch)
    ], dtype=str)
    positions = np.arange(n_trials, dtype=np.int64)

    # Include the original aliases, before normalization: equal histories from
    # absent, legacy and current flag schemas have different cache identities.
    identity = {
        "session_id": session_id,
        "actions": _missing_as_none(actions), "rewards": _missing_as_none(rewards),
        "manual_reward_flags": _missing_as_none(flags),
        "comparison_valid": comparison_valid.tolist(),
        "history_update_valid": history_update_valid.tolist(),
        "trial_id": trial_ids.tolist(), "block_id": block_ids.tolist(),
        "row_position": positions.tolist(), "initialization": "first_retained_row",
        "source_flag_columns": source_flag_columns,
        "trial_outcome": outcomes.tolist(),
        "source_outcome_column": "trial_outcome" in trial_df,
    }
    metadata = {
        "input_digest": hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest(),
        "source_flag_columns": source_flag_columns,
        "absent_flag_assumed_animal_controlled": not source_flag_columns,
        "initialization": "first_retained_row",
        "retained_trial_range": [trial_ids[0], trial_ids[-1]] if n_trials else [],
        "missing_reward_count": int((comparison_valid & missing_reward).sum()),
        "unknown_manual_flag_count": int(unknown_flag.sum()),
        "trial_outcome_counts": {
            "recorded_attempts": n_trials,
            "valid_animal_choices": int(comparison_valid.sum()),
            "invalid_switch": int(invalid_switch.sum()),
            "manual_reward": int(manual.sum()),
            "unknown": int((outcomes == "unknown").sum()),
        },
    }
    return PreparedTrialHistory(
        session_id, actions, rewards, flags, skip, comparison_valid,
        history_update_valid, comparison_reasons, update_reasons,
        trial_ids, block_ids, positions, metadata,
    )


def resolve_trial_policy_candidates(
    model_ids: Sequence[str] = DEFAULT_MODEL_IDS,
) -> tuple[TrialPolicyCandidate, ...]:
    """Resolve ordered IDs into immutable, ordinary adapter definitions.

    model_ids is a nonempty sequence[str], shape (K,), with unique implemented
    IDs. Returns tuple[TrialPolicyCandidate,...], length K, in supplied order;
    this operation has no physical units and does not run any policy.
    """
    if isinstance(model_ids, str) or not model_ids:
        raise ValueError("model_ids must be a nonempty sequence of unique policy IDs")
    if len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids contains duplicate policy IDs")
    definitions = (
        TrialPolicyCandidate("forgetting_q_learning", "Forgetting Q-learning", _replay_fql, ("FQL_decay", "n_actions")),
        TrialPolicyCandidate("bayesian_hmm", "Bayesian HMM", _replay_bayesian, _BELIEF_PARAMETER_NAMES),
        TrialPolicyCandidate("win_stay_lose_switch", "Win-stay lose-switch", _replay_wsls),
        TrialPolicyCandidate("q_learning", "Q-learning", _replay_q_learning, ("QL_learning_rate", "n_actions")),
        TrialPolicyCandidate("reward_decay_hmm", "Reward-decay HMM", _replay_reward_decay_hmm,
                             _BELIEF_PARAMETER_NAMES + ("hmm_reward_decay_lambda",),
                             "exact_zero_after_equal_belief_correction"),
        TrialPolicyCandidate("ideal_observer", "Ideal observer", _replay_ideal_observer,
                             _BELIEF_PARAMETER_NAMES + ("hmm_reward_decay_lambda", "omission_lam"),
                             "exact_zero_after_belief_correction_and_raw_doubt_subtraction"),
    )
    by_id = {definition.model_id: definition for definition in definitions}
    unknown = [model_id for model_id in model_ids if model_id not in by_id]
    if unknown:
        raise ValueError(f"Unknown or unimplemented trial policy IDs: {unknown}")
    return tuple(by_id[model_id] for model_id in model_ids)


def effective_trial_policy_parameters(
    candidates: Sequence[TrialPolicyCandidate], feature_params: "TaskParams"
) -> dict[str, dict[str, Any]]:
    """Return validated plain parameters for each selected policy in order.

    candidates is a sequence of K definitions; feature_params is explicit
    TaskParams. Returns dict[model_id,dict[str,scalar]], dimensionless policy
    probabilities/scales and binary reward-indicator units. Unrelated fields
    are excluded. Synthetic adapters may declare numeric parameter_names.
    """
    effective = {}
    for candidate in candidates:
        if candidate.model_id in ("forgetting_q_learning", "q_learning"):
            if feature_params.n_actions != 2:
                raise ValueError("n_actions must be 2 for the two-action trial policies")
            if candidate.model_id == "forgetting_q_learning":
                decay = _finite_parameter(feature_params, "FQL_decay", 0, 1)
                parameters = {"decay": decay, "reward_update_rate": 1 - decay, "n_actions": 2}
            else:
                parameters = {"learning_rate": _finite_parameter(feature_params, "QL_learning_rate", 0, 1),
                              "n_actions": 2}
        elif candidate.model_id in ("bayesian_hmm", "reward_decay_hmm", "ideal_observer"):
            parameters = {name: _finite_parameter(feature_params, name, 0, 1) for name in (
                "state_transition_prob", "active_reward_probability", "inactive_reward_probability")}
            for name in ("correct_reward_size", "incorrect_reward_size"):
                parameters[name] = _finite_parameter(feature_params, name, 0, np.inf)
            parameters["tanh_scale"] = _finite_parameter(feature_params, "tanh_scale", 0, np.inf)
            if parameters["tanh_scale"] == 0:
                raise ValueError("tanh_scale must be strictly positive")
            parameters["value_mode"] = "bayesian_log_odds"
            if candidate.model_id != "bayesian_hmm":
                parameters["lambda_decay"] = _finite_parameter(feature_params, "hmm_reward_decay_lambda", 0, 1)
            if candidate.model_id == "ideal_observer":
                parameters["omission_lam"] = _finite_parameter(feature_params, "omission_lam", 0, np.inf)
                if parameters["omission_lam"] == 0:
                    raise ValueError("omission_lam must be strictly positive")
        else:
            parameters = {name: _finite_parameter(feature_params, name, -np.inf, np.inf)
                          for name in candidate.parameter_names}
        effective[candidate.model_id] = parameters
    return effective


def replay_trial_policy_candidates(
    history: PreparedTrialHistory,
    candidates: Sequence[TrialPolicyCandidate],
    feature_params: "TaskParams",
) -> tuple[np.ndarray, np.ndarray]:
    """Replay selected policies independently using only observed usable history.

    history contains aligned (T,) inputs; candidates is an ordered sequence of
    K definitions and feature_params supplies explicit dimensionless settings.
    Returns float64 scores (T,K), left-positive with NaN unavailable, and bool
    prediction_valid (T,K). Exact zero remains available genuine indifference.
    No model selection or prediction gates another model's outcome updates.
    """
    if not candidates:
        raise ValueError("At least one trial policy candidate is required")
    effective_trial_policy_parameters(candidates, feature_params)
    shape = (len(history.actions), len(candidates))
    scores = np.full(shape, np.nan, dtype=float)
    prediction_valid = np.zeros(shape, dtype=bool)
    for column, candidate in enumerate(candidates):
        try:
            candidate_scores, available = candidate.adapter(history, feature_params)
        except ValueError as error:
            # The enabled Bayesian guard reports retained row position. Add
            # original IDs here without replaying or implementing learning twice.
            row_match = re.search(r"row (\d+)", str(error))
            trial = history.trial_id[int(row_match[1])] if row_match else "unknown"
            raise ValueError(f"Session {history.session_id}, trial {trial}, model {candidate.model_id}: {error}") from error
        candidate_scores = np.asarray(candidate_scores, dtype=float)
        available = np.asarray(available, dtype=bool)
        if candidate_scores.shape != (shape[0],) or available.shape != (shape[0],):
            raise ValueError(f"Policy {candidate.model_id} must return scores and validity with shape ({shape[0]},)")
        if not np.isfinite(candidate_scores[available]).all():
            raise ValueError(f"Session {history.session_id}, model {candidate.model_id}: non-finite available prediction")
        prediction_valid[:, column] = available
        scores[available, column] = candidate_scores[available]
    return scores, prediction_valid


# Private input and adapter helpers.
def _parse_binary_value(value: Any, context: str, field: str, *, unknown_allowed: bool = False) -> float:
    """Parse scalar dimensionless observation to exact float 0/1 or missing NaN.

    value is a scalar CSV/pandas representation; context/field are error strings.
    unknown_allowed maps unrecognized flag scalars to NaN; malformed actions or
    rewards raise ValueError. No arrays are reshaped and no units are converted.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value in MISSING_STRINGS or (field == "action" and value.lower() in NO_CHOICE_ACTION_LABELS):
            return np.nan
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = np.nan
    if np.isfinite(numeric) and numeric in (0, 1):
        return numeric
    if unknown_allowed:
        return np.nan
    raise ValueError(f"{context}: {field} must be exact finite binary 0/1 or a supported missing sentinel; got {value!r}")


def _exclusion_reasons(no_choice: np.ndarray, manual: np.ndarray, unknown: np.ndarray,
                       missing_reward: np.ndarray | None = None) -> np.ndarray:
    """Map bool[T] exclusion masks to Unicode[T] stable semicolon-joined names, unitless."""
    masks = [("manual_reward", manual), ("no_choice", no_choice), ("unknown_manual_flag", unknown)]
    if missing_reward is not None:
        masks.append(("missing_reward", missing_reward))
    return np.array([";".join(name for name, mask in masks if mask[row])
                     for row in range(len(no_choice))], dtype=str)


def _missing_as_none(values: np.ndarray) -> list[float | None]:
    """Convert float64[T] binary/missing values to JSON-compatible list[T], unitless."""
    return [None if np.isnan(value) else float(value) for value in values]


def _finite_parameter(params: "TaskParams", name: str, lower: float, upper: float) -> float:
    """Read named scalar TaskParams field, returning finite float in inclusive unitless bounds."""
    try:
        value = float(getattr(params, name))
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError(f"{name} must be a finite numeric policy parameter") from error
    if not np.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
    return value


def _feature_arrays(values: Any, skip_flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert object/float[T] outputs and bool[T] skips to unitless float64/bool[T].

    Existing policies must produce finite scores on every non-skipped row,
    including rows with missing rewards. Unexpected nonfinite scores raise
    ValueError with retained row position rather than becoming unavailability.
    """
    scores = np.asarray(values, dtype=float)
    available = ~skip_flags
    if scores.shape != available.shape:
        raise ValueError("Existing policy output must match retained history shape (T,)")
    invalid_rows = np.flatnonzero(available & ~np.isfinite(scores))
    if invalid_rows.size:
        raise ValueError(f"non-finite policy prediction at row {invalid_rows[0]}")
    return scores, available


def _replay_fql(history: PreparedTrialHistory, params: "TaskParams") -> tuple[np.ndarray, np.ndarray]:
    """Replay observed history[T] with TaskParams; return reward-scale left-positive float/bool[T]."""
    parameters = effective_trial_policy_parameters(resolve_trial_policy_candidates(("forgetting_q_learning",)), params)
    return _feature_arrays(trial_features.forgetting_qlearning_relative_value(
        actions=history.actions, rewards=history.rewards,
        experimenter_reward_given=history.adapter_skip_flags,
        **parameters["forgetting_q_learning"],
    ), history.adapter_skip_flags)


def _replay_bayesian(history: PreparedTrialHistory, params: "TaskParams") -> tuple[np.ndarray, np.ndarray]:
    """Replay history[T] and explicit belief parameters; return dimensionless left-positive float/bool[T]."""
    parameters = effective_trial_policy_parameters(resolve_trial_policy_candidates(("bayesian_hmm",)), params)
    return _feature_arrays(trial_features.hmm_relative_value(
        actions=history.actions, rewards=history.rewards,
        experimenter_reward_given=history.adapter_skip_flags,
        validate_outcome_evidence=True, **parameters["bayesian_hmm"],
    ), history.adapter_skip_flags)


def _replay_wsls(history: PreparedTrialHistory, params: "TaskParams") -> tuple[np.ndarray, np.ndarray]:
    """Replay usable choice/outcome history[T]; return unitless signed float/bool[T]; params unused."""
    return _feature_arrays(trial_features.win_stay_lose_switch_regressor(
        actions=history.actions, rewards=history.rewards,
        experimenter_reward_given=history.adapter_skip_flags,
    ), history.adapter_skip_flags)


def _replay_q_learning(history: PreparedTrialHistory, params: "TaskParams") -> tuple[np.ndarray, np.ndarray]:
    """Replay observed history[T]/TaskParams; return reward-scale left-positive float/bool[T]."""
    parameters = effective_trial_policy_parameters(resolve_trial_policy_candidates(("q_learning",)), params)
    return _feature_arrays(trial_features.qlearning_relative_value(
        actions=history.actions, rewards=history.rewards,
        experimenter_reward_given=history.adapter_skip_flags,
        **parameters["q_learning"],
    ), history.adapter_skip_flags)


def _replay_reward_decay_hmm(history: PreparedTrialHistory, params: "TaskParams") -> tuple[np.ndarray, np.ndarray]:
    """Replay history[T]/TaskParams; return dimensionless pre-choice belief scores and validity[T]."""
    parameters = effective_trial_policy_parameters(resolve_trial_policy_candidates(("reward_decay_hmm",)), params)
    return _feature_arrays(trial_features.hmm_relative_value_reward_decay(
        actions=history.actions, rewards=history.rewards,
        experimenter_reward_given=history.adapter_skip_flags,
        validate_outcome_evidence=True, **parameters["reward_decay_hmm"],
    ), history.adapter_skip_flags)


def _replay_ideal_observer(history: PreparedTrialHistory, params: "TaskParams") -> tuple[np.ndarray, np.ndarray]:
    """Replay history[T]/TaskParams; return dimensionless belief-minus-doubt scores and bool[T].

    Omission counts are in trials and omission_lam is in inverse trials. Read
    doubt before each observed outcome; skipped/missing outcomes freeze both
    belief and doubt history. Positive scores favor Left, as for every adapter.
    """
    scores, available = _replay_reward_decay_hmm(history, params)
    parameters = effective_trial_policy_parameters(resolve_trial_policy_candidates(("ideal_observer",)), params)
    omission_lam = parameters["ideal_observer"]["omission_lam"]
    doubt_state = CounterfactualDoubtState()
    for row in range(len(history.actions)):
        if not available[row]:
            continue
        scores[row] -= counterfactual_doubt_raw_value(doubt_state, omission_lam)
        if history.history_update_valid[row]:
            doubt_state = update_counterfactual_doubt(
                doubt_state, int(history.actions[row]), float(history.rewards[row]))
    return scores, available
