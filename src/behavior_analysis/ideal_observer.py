"""Ideal-observer performance summaries for fixed mouse sessions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior_analysis import general_behavior_assessment, trial_features
from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    get_experimenter_reward_flags,
    normalize_experimenter_reward_column,
)
from src.behavior_modeling import controller
from src.behavior_modeling.parameters import task_config

RIGHT_CHOICE = 0
LEFT_CHOICE = 1
MISSING_SUMMARY_VALUE = "None"


@dataclass
class IdealObserverParams:
    """Parameters controlling history-policy and fixed-state replay summaries."""
    hmm_reward_decay_lambda: float = 0.1
    relative_doubt_lambda: float = 0.5
    tanh_scale: float = 1.0
    n_fixed_replays: int = 100
    random_seed: int | None = 12345
    initial_tie_choice: int = RIGHT_CHOICE


MOUSE_HISTORY_RUN_COLUMNS = [
    "state",
    "state_int",
    "cur_trial",
    "cur_trial_in_block",
    "cur_block",
    "action",
    "correct",
    "reward",
    "p_active_rew",
    "p_inactive_rew",
    "p_switch",
    "model_stimulus",
    "agent_action_dist",
    "agent_hmm_value",
    "agent_doubt_value",
    "agent_relative_value",
    "agent_prior",
]


def state_to_oracle_choice(states) -> np.ndarray:
    """Map task state labels to oracle choices.

    Parameters
    ----------
    states : array-like
        State labels with shape `(n_trials,)`. Supported labels begin with
        `right` or `left`, including real-session `right_patch`/`left_patch`
        and simulated `right`/`left`.

    Returns
    -------
    np.ndarray
        Integer oracle choices with shape `(n_trials,)`; `0` is right and `1`
        is left.
    """
    state_series = pd.Series(states).astype(str).str.strip().str.lower()
    oracle_choice = np.full(state_series.shape[0], -1, dtype=int)
    oracle_choice[state_series.str.startswith("right").to_numpy()] = RIGHT_CHOICE
    oracle_choice[state_series.str.startswith("left").to_numpy()] = LEFT_CHOICE
    if np.any(oracle_choice < 0):
        bad_states = sorted(state_series.iloc[np.flatnonzero(oracle_choice < 0)].unique())
        raise ValueError(f"state values must begin with 'right' or 'left', got: {bad_states}")
    return oracle_choice


def choose_greedy_left_positive_actions(
    left_positive_values,
    initial_choice: int = RIGHT_CHOICE,
) -> np.ndarray:
    """Choose greedily from signed left-positive values with repeat-on-tie.

    Parameters
    ----------
    left_positive_values : array-like
        One-dimensional signed decision values with shape `(n_trials,)`.
        Positive values choose left (`1`), negative values choose right (`0`),
        and exact ties repeat the previous ideal choice.
    initial_choice : int, default=0
        Choice used for a tie on the first retained trial.

    Returns
    -------
    np.ndarray
        Integer choices with shape `(n_trials,)`, using `0` for right and `1`
        for left.
    """
    values = np.asarray(left_positive_values, dtype=float)
    choices = np.zeros(values.shape[0], dtype=int)
    previous_choice = int(initial_choice)
    for i, value in enumerate(values):
        if value > 0:
            choice = LEFT_CHOICE
        elif value < 0:
            choice = RIGHT_CHOICE
        else:
            choice = previous_choice
        choices[i] = choice
        previous_choice = choice
    return choices


def _valid_trials(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return normalized trial data and valid behavior mask."""
    normalized = normalize_experimenter_reward_column(trial_df)
    return normalized, general_behavior_assessment.make_valid_behavior_mask(normalized)


def _numeric_column_or_default(
    trial_df: pd.DataFrame,
    column_name: str,
    default_value: float,
) -> np.ndarray:
    """Return a numeric column or a default-filled array."""
    if column_name not in trial_df.columns:
        return np.full(trial_df.shape[0], float(default_value), dtype=float)

    numeric_values = pd.to_numeric(trial_df[column_name], errors="coerce")
    if numeric_values.isna().any():
        bad_values = sorted(trial_df.loc[numeric_values.isna(), column_name].astype(str).unique())
        raise ValueError(f"{column_name} values must be numeric, got: {bad_values}")
    return numeric_values.to_numpy(dtype=float)


def _first_valid_value(values: np.ndarray, valid_mask: np.ndarray, default_value: float) -> float:
    """Return the first valid-trial value, or a default when no valid row exists."""
    if valid_mask.any():
        return float(values[np.flatnonzero(valid_mask)[0]])
    return float(default_value)


def _history_hmm_decay_values(
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray,
    params: IdealObserverParams,
) -> np.ndarray:
    """Return mouse-history HMM reward-decay values."""
    if "HMM_rel_value_logodds_decay" in trial_df.columns:
        return pd.to_numeric(trial_df["HMM_rel_value_logodds_decay"], errors="coerce").to_numpy(dtype=float)

    p_active = _numeric_column_or_default(trial_df, "p_active_rew", 0.8)
    p_inactive = _numeric_column_or_default(trial_df, "p_inactive_rew", 0.0)
    p_switch = _numeric_column_or_default(trial_df, "p_switch", 0.2)
    experimenter_flags = get_experimenter_reward_flags(trial_df, default_zero=True)
    return np.asarray(
        trial_features.hmm_relative_value_reward_decay(
            actions=trial_df["action"].to_numpy(),
            rewards=trial_df["reward"].to_numpy(),
            state_transition_prob=_first_valid_value(p_switch, valid_mask, 0.2),
            active_reward_probability=_first_valid_value(p_active, valid_mask, 0.8),
            inactive_reward_probability=_first_valid_value(p_inactive, valid_mask, 0.0),
            lambda_decay=params.hmm_reward_decay_lambda,
            value_mode="bayesian_log_odds",
            tanh_scale=params.tanh_scale,
            experimenter_reward_given=experimenter_flags,
        ),
        dtype=object,
    )


def _history_relative_doubt_values(
    trial_df: pd.DataFrame,
    params: IdealObserverParams,
) -> np.ndarray:
    """Return mouse-history relative doubt values."""
    if "relative_doubt_index" in trial_df.columns:
        return pd.to_numeric(trial_df["relative_doubt_index"], errors="coerce").to_numpy(dtype=float)

    required_columns = {"right_cf_omissions", "left_cf_omissions"}
    missing_columns = sorted(required_columns.difference(trial_df.columns))
    if missing_columns:
        raise ValueError(
            "trial_df must contain 'relative_doubt_index' or the counterfactual "
            f"omission columns. Missing: {', '.join(missing_columns)}"
        )

    right_omissions = pd.to_numeric(trial_df["right_cf_omissions"], errors="coerce").to_numpy(dtype=float)
    left_omissions = pd.to_numeric(trial_df["left_cf_omissions"], errors="coerce").to_numpy(dtype=float)
    return trial_features.relative_doubt_index(
        R_omissions=right_omissions,
        L_omissions=left_omissions,
        lam=params.relative_doubt_lambda,
    )


def compute_history_ideal_values(
    trial_df: pd.DataFrame,
    params: IdealObserverParams | None = None,
) -> np.ndarray:
    """Compute mouse-history ideal-observer decision values.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `action`, `reward`, and either precomputed
        `HMM_rel_value_logodds_decay`/`relative_doubt_index` or the columns
        needed to compute them.
    params : IdealObserverParams or None
        Observer parameters. Defaults match the current trial-feature defaults.

    Returns
    -------
    np.ndarray
        Object array with shape `(n_trials,)`. Valid rows contain
        `HMM_rel_value_logodds_decay - relative_doubt_index`; skipped rows hold
        `None`.
    """
    if params is None:
        params = IdealObserverParams()
    normalized, valid_mask = _valid_trials(trial_df)

    hmm_values = _history_hmm_decay_values(normalized, valid_mask, params)
    doubt_values = _history_relative_doubt_values(normalized, params)
    ideal_values = np.full(normalized.shape[0], None, dtype=object)
    for i in np.flatnonzero(valid_mask):
        try:
            ideal_values[i] = float(hmm_values[i]) - float(doubt_values[i])
        except (TypeError, ValueError):
            ideal_values[i] = None
    return ideal_values


def summarize_history_ideal_policy(
    trial_df: pd.DataFrame,
    params: IdealObserverParams | None = None,
) -> dict:
    """Summarize the greedy ideal policy driven by mouse reward history.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `state`, `action`, `reward`, and `p_active_rew`; `p_inactive_rew`
        defaults to zero when absent. Experimenter rewards and no-choice trials
        are excluded.
    params : IdealObserverParams or None
        Observer parameters and tie rule.

    Returns
    -------
    dict
        Session-level history-policy summary. Reward values are in expected
        task reward units.
    """
    if params is None:
        params = IdealObserverParams()
    if "state" not in trial_df.columns or "p_active_rew" not in trial_df.columns:
        return _missing_history_summary()

    normalized, valid_mask = _valid_trials(trial_df)
    ideal_values = compute_history_ideal_values(normalized, params=params)
    numeric_ideal_values = pd.to_numeric(pd.Series(ideal_values), errors="coerce")
    summary_mask = valid_mask & numeric_ideal_values.notna().to_numpy()
    if not summary_mask.any():
        return _missing_history_summary()

    retained_values = numeric_ideal_values.loc[summary_mask].to_numpy(dtype=float)
    ideal_choices = choose_greedy_left_positive_actions(
        retained_values,
        initial_choice=params.initial_tie_choice,
    )
    oracle_choices = state_to_oracle_choice(normalized.loc[summary_mask, "state"])
    mouse_choices = pd.to_numeric(normalized.loc[summary_mask, "action"], errors="coerce").to_numpy(dtype=int)
    p_active = pd.to_numeric(normalized.loc[summary_mask, "p_active_rew"], errors="coerce").to_numpy(dtype=float)
    p_inactive = _numeric_column_or_default(normalized, "p_inactive_rew", 0.0)[summary_mask]
    reward = pd.to_numeric(normalized.loc[summary_mask, "reward"], errors="coerce").to_numpy(dtype=float)

    ideal_correct = ideal_choices == oracle_choices
    history_expected_reward = float(np.sum(np.where(ideal_correct, p_active, p_inactive)))
    actual_reward_collected = float(np.sum(reward))
    reward_fraction = (
        actual_reward_collected / history_expected_reward
        if history_expected_reward > 0
        else MISSING_SUMMARY_VALUE
    )
    reward_difference = actual_reward_collected - history_expected_reward

    return {
        "history_ideal_oracle_accuracy": float(np.mean(ideal_correct)),
        "history_ideal_mouse_agreement": float(np.mean(ideal_choices == mouse_choices)),
        "history_ideal_expected_reward": history_expected_reward,
        "history_ideal_reward_fraction": reward_fraction,
        "history_ideal_reward_difference": reward_difference,
    }


def _missing_history_summary() -> dict:
    """Return missing-value sentinels for history-policy summaries."""
    return {
        "history_ideal_oracle_accuracy": MISSING_SUMMARY_VALUE,
        "history_ideal_mouse_agreement": MISSING_SUMMARY_VALUE,
        "history_ideal_expected_reward": MISSING_SUMMARY_VALUE,
        "history_ideal_reward_fraction": MISSING_SUMMARY_VALUE,
        "history_ideal_reward_difference": MISSING_SUMMARY_VALUE,
    }


def _build_replay_agent_params(
    p_switch: float,
    p_active: float,
    p_inactive: float,
    params: IdealObserverParams,
) -> tuple[task_config.AgentParams, task_config.TaskParams]:
    """Build behavior-modeling params for fixed-state replay."""
    agent_params = task_config.AgentParams(
        HMM_transition_prob=float(p_switch),
        HMM_value_mode="bayesian_log_odds",
        HMM_log_odds_tanh_scale=params.tanh_scale,
        HMM_reward_decay_lambda=params.hmm_reward_decay_lambda,
        relative_doubt_lambda=params.relative_doubt_lambda,
        HMM_active_reward_probability=float(p_active),
        HMM_inactive_reward_probability=float(p_inactive),
        logistic_alpha=0.0,
        action_temperature=1.0,
        greedy_action_selection=False,
    )
    task_params = task_config.TaskParams(
        active_reward_probability=float(p_active),
        inactive_reward_probability=float(p_inactive),
        state_transition_prob=float(p_switch),
        mean_correct_reward=1.0,
        mean_incorrect_reward=0.0,
        reward_std_dev=0.0,
    )
    return agent_params, task_params


def run_fixed_state_ideal_replays(
    trial_df: pd.DataFrame,
    params: IdealObserverParams | None = None,
) -> pd.DataFrame:
    """Replay an ideal agent against the fixed mouse state sequence.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `state`, `action`, `reward`, `p_active_rew`, and `p_switch`;
        `p_inactive_rew` defaults to zero. Invalid mouse-action rows are
        excluded from the fixed sequence.
    params : IdealObserverParams or None
        Replay count, seed, and model parameters.

    Returns
    -------
    pd.DataFrame
        Replay table with shape `(n_fixed_replays, 3)` and columns
        `sampled_reward`, `expected_reward`, and `oracle_accuracy`.
    """
    if params is None:
        params = IdealObserverParams()
    if params.n_fixed_replays < 0:
        raise ValueError("n_fixed_replays must be non-negative.")
    required_columns = {"state", "action", "reward", "p_active_rew", "p_switch"}
    if missing_columns := sorted(required_columns.difference(trial_df.columns)):
        raise ValueError(f"trial_df is missing required replay columns: {', '.join(missing_columns)}")

    normalized, valid_mask = _valid_trials(trial_df)
    if not valid_mask.any():
        return pd.DataFrame(columns=["sampled_reward", "expected_reward", "oracle_accuracy"])

    replay_df = normalized.loc[valid_mask].reset_index(drop=True)
    oracle_choices = state_to_oracle_choice(replay_df["state"])
    p_active = pd.to_numeric(replay_df["p_active_rew"], errors="coerce").to_numpy(dtype=float)
    p_inactive = _numeric_column_or_default(replay_df, "p_inactive_rew", 0.0)
    p_switch = pd.to_numeric(replay_df["p_switch"], errors="coerce").to_numpy(dtype=float)

    rng = np.random.default_rng(params.random_seed)
    rows = []
    for _ in range(params.n_fixed_replays):
        agent_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        agent_params, task_params = _build_replay_agent_params(
            p_switch=_first_valid_value(p_switch, np.ones_like(p_switch, dtype=bool), 0.2),
            p_active=_first_valid_value(p_active, np.ones_like(p_active, dtype=bool), 0.8),
            p_inactive=_first_valid_value(p_inactive, np.ones_like(p_inactive, dtype=bool), 0.0),
            params=params,
        )
        agent = controller.select_agent(
            "HMM_reward_decay_relative_doubt",
            agent_params,
            task_params,
            rng=np.random.default_rng(agent_seed),
        )
        previous_choice = params.initial_tie_choice
        sampled_reward = 0.0
        expected_reward = 0.0
        correct_choices = []
        for oracle_choice, active_prob, inactive_prob in zip(oracle_choices, p_active, p_inactive):
            ideal_choice = choose_greedy_left_positive_actions(
                [agent.value],
                initial_choice=previous_choice,
            )[0]
            previous_choice = int(ideal_choice)
            ideal_correct = ideal_choice == oracle_choice
            correct_choices.append(ideal_correct)
            reward_probability = active_prob if ideal_correct else inactive_prob
            expected_reward += float(reward_probability)
            reward = float(rng.random() < reward_probability)
            sampled_reward += reward
            agent.update_params(int(ideal_choice), reward)

        rows.append(
            {
                "sampled_reward": sampled_reward,
                "expected_reward": expected_reward,
                "oracle_accuracy": float(np.mean(correct_choices)),
            }
        )

    return pd.DataFrame(rows, columns=["sampled_reward", "expected_reward", "oracle_accuracy"])


def compute_mouse_history_observer_run(
    trial_df: pd.DataFrame,
    params: IdealObserverParams | None = None,
) -> pd.DataFrame:
    """Run the observer agent over the mouse's actual choice and reward history.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        `state`, `action`, `reward`, `p_active_rew`, and `p_switch`;
        `p_inactive_rew` defaults to zero. Rows with no mouse choice or an
        experimenter reward are excluded. Choice convention is `0` right and
        `1` left; rewards are in task reward units.
    params : IdealObserverParams or None
        Observer parameters. These use the same defaults as the session-level
        ideal-observer summaries.

    Returns
    -------
    pd.DataFrame
        Trialwise observer trace with shape `(n_valid_trials, n_columns)`.
        Values are recorded before the current mouse trial updates the agent.
        `agent_action_dist` is the pre-update probability of choosing left,
        and `agent_relative_value` is left-positive combined policy value.
    """
    if params is None:
        params = IdealObserverParams()

    required_columns = {"state", "action", "reward", "p_active_rew", "p_switch"}
    if missing_columns := sorted(required_columns.difference(trial_df.columns)):
        raise ValueError(f"trial_df is missing required observer trace columns: {', '.join(missing_columns)}")

    normalized, valid_mask = _valid_trials(trial_df)
    if not valid_mask.any():
        return pd.DataFrame(columns=MOUSE_HISTORY_RUN_COLUMNS)

    history_df = normalized.loc[valid_mask].reset_index(drop=True)
    actions = pd.to_numeric(history_df["action"], errors="coerce").to_numpy(dtype=int)
    rewards = pd.to_numeric(history_df["reward"], errors="coerce").to_numpy(dtype=float)
    p_active = pd.to_numeric(history_df["p_active_rew"], errors="coerce").to_numpy(dtype=float)
    p_inactive = _numeric_column_or_default(history_df, "p_inactive_rew", 0.0)
    p_switch = pd.to_numeric(history_df["p_switch"], errors="coerce").to_numpy(dtype=float)

    agent_params, task_params = _build_replay_agent_params(
        p_switch=_first_valid_value(p_switch, np.ones_like(p_switch, dtype=bool), 0.2),
        p_active=_first_valid_value(p_active, np.ones_like(p_active, dtype=bool), 0.8),
        p_inactive=_first_valid_value(p_inactive, np.ones_like(p_inactive, dtype=bool), 0.0),
        params=params,
    )
    agent = controller.select_agent(
        "HMM_reward_decay_relative_doubt",
        agent_params,
        task_params,
        rng=np.random.default_rng(params.random_seed),
    )

    rows = []
    for row_index, row in history_df.iterrows():
        action = int(actions[row_index])
        reward = float(rewards[row_index])
        agent_action_dist = getattr(agent, "action_dist", np.array([np.nan, np.nan]))
        rows.append(
            {
                "state": row["state"],
                "state_int": row["state_int"] if "state_int" in history_df.columns else np.nan,
                "cur_trial": row["cur_trial"] if "cur_trial" in history_df.columns else row_index,
                "cur_trial_in_block": (
                    row["cur_trial_in_block"] if "cur_trial_in_block" in history_df.columns else np.nan
                ),
                "cur_block": row["cur_block"] if "cur_block" in history_df.columns else np.nan,
                "action": action,
                "correct": row["correct"] if "correct" in history_df.columns else np.nan,
                "reward": reward,
                "p_active_rew": float(p_active[row_index]),
                "p_inactive_rew": float(p_inactive[row_index]),
                "p_switch": float(p_switch[row_index]),
                "model_stimulus": row["active_stimulus"] if "active_stimulus" in history_df.columns else np.nan,
                "agent_action_dist": float(agent_action_dist[LEFT_CHOICE]),
                "agent_hmm_value": controller.get_agent_value_component(agent, "hmm_value"),
                "agent_doubt_value": controller.get_agent_value_component(agent, "doubt_value"),
                "agent_relative_value": float(agent.value),
                "agent_prior": controller.get_agent_prior(agent),
            }
        )
        agent.update_params(action, reward)

    return pd.DataFrame(rows, columns=MOUSE_HISTORY_RUN_COLUMNS)


def summarize_fixed_state_ideal_replay_distribution(replay_df: pd.DataFrame) -> dict:
    """Summarize fixed-state replay distributions with means and quartiles."""
    if replay_df.empty:
        return _missing_replay_summary()

    summary = {"fixed_replay_ideal_n_replays": int(replay_df.shape[0])}
    column_prefixes = {
        "sampled_reward": "fixed_replay_ideal_reward",
        "expected_reward": "fixed_replay_ideal_expected_reward",
        "oracle_accuracy": "fixed_replay_ideal_oracle_accuracy",
    }
    for column_name, prefix in column_prefixes.items():
        values = pd.to_numeric(replay_df[column_name], errors="coerce").dropna()
        if values.empty:
            summary.update(_missing_distribution_fields(prefix))
            continue
        quartiles = values.quantile([0.25, 0.5, 0.75])
        summary[f"{prefix}_mean"] = float(values.mean())
        summary[f"{prefix}_q1"] = float(quartiles.loc[0.25])
        summary[f"{prefix}_median"] = float(quartiles.loc[0.5])
        summary[f"{prefix}_q3"] = float(quartiles.loc[0.75])
    return summary


def _missing_distribution_fields(prefix: str) -> dict:
    """Return missing sentinels for one replay distribution."""
    return {
        f"{prefix}_mean": MISSING_SUMMARY_VALUE,
        f"{prefix}_q1": MISSING_SUMMARY_VALUE,
        f"{prefix}_median": MISSING_SUMMARY_VALUE,
        f"{prefix}_q3": MISSING_SUMMARY_VALUE,
    }


def _missing_replay_summary() -> dict:
    """Return missing sentinels for replay summaries."""
    summary = {"fixed_replay_ideal_n_replays": 0}
    for prefix in (
        "fixed_replay_ideal_reward",
        "fixed_replay_ideal_expected_reward",
        "fixed_replay_ideal_oracle_accuracy",
    ):
        summary.update(_missing_distribution_fields(prefix))
    return summary


def summarize_ideal_observer_behavior(
    trial_df: pd.DataFrame,
    params: IdealObserverParams | None = None,
) -> dict:
    """Return history-policy and fixed-state replay summaries for one session."""
    if params is None:
        params = IdealObserverParams()

    history_summary = summarize_history_ideal_policy(trial_df, params=params)
    try:
        replay_df = run_fixed_state_ideal_replays(trial_df, params=params)
        replay_summary = summarize_fixed_state_ideal_replay_distribution(replay_df)
    except ValueError:
        replay_summary = _missing_replay_summary()
    return history_summary | replay_summary
