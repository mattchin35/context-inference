import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import pickle as pkl
import src.behavior_analysis.trial_features as trial_features
import json
from typing import Optional


@dataclass
class TaskParams:
    n_states: int = 2
    n_actions: int = 2

    # Probability parameters
    # For HHM inference model, you can play with using model parameters that are different from the true task parameters
    p_cue: float = 0
    state_transition_prob: float = .2
    active_reward_probability: float = .8
    inactive_reward_probability: float = 0
    correct_reward_size: float = 1
    incorrect_reward_size: float = 0
    tanh_scale: float = 1
    hmm_reward_decay_lambda: float = 0.1
    # value_mode: str = 'bayesian_log_odds'  # 'expected_reward' or 'bayesian_log_odds'

    # Reinforcement learning parameters
    FQL_decay: float = .7  # for forgetting Q-learning agent
    FQL_reward_update_rate_fast_learn: float = .5
    QL_learning_rate: float = .3  # for standard Q-learning agent
    omission_lam: float = 0.5
    perseveration_decay: float = 0.25


def save_trial_features(augmented_trial_df: pd.DataFrame, params: TaskParams, processed_data_path: Path,
                        sess_id_full: str):
    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df.to_csv(augmented_trial_df_path, index=False, na_rep='None')

    # Save to JSON
    json_fname = processed_data_path / 'trial_feature_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)


def collect_and_save_trial_features(
    augmented_trial_df: pd.DataFrame,
    processed_data_path: Path,
    sess_id_full: str,
    params: Optional[TaskParams] = None,
) -> tuple[pd.DataFrame, TaskParams]:
    """Collect trial features from an already loaded dataframe and save outputs.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe already loaded by the caller. Rows index trials and
        must include the columns required by `collect_trial_features`.
    processed_data_path : Path
        Session-local processed-data directory where the updated augmented trial
        CSV and feature parameter JSON will be written.
    sess_id_full : str
        Full session identifier used to name the augmented trial CSV.
    params : Optional[TaskParams], default=None
        Optional feature-generation parameters. If omitted, defaults are used.

    Returns
    -------
    tuple[pd.DataFrame, TaskParams]
        - augmented_trial_df with feature columns added
        - parameters used to compute and save those features
    """
    augmented_trial_df, params = collect_trial_features(augmented_trial_df, params=params)
    save_trial_features(augmented_trial_df, params, processed_data_path, sess_id_full)
    return augmented_trial_df, params


def _get_column_or_raise(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col]
    raise ValueError(f"augmented_trial_df must contain one of columns: {candidates}")


def _is_give_reward_flag(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        parsed = value.strip().lower()
        if parsed in {"1", "true", "t", "yes", "y"}:
            return True
        if parsed in {"0", "false", "f", "no", "n", "none", ""}:
            return False
        try:
            return bool(int(float(parsed)))
        except (TypeError, ValueError):
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def collect_trial_index_features(
    augmented_trial_df: pd.DataFrame,
    omission_lam: float = 0.5,
    perseveration_decay: float = 0.25,
) -> pd.DataFrame:
    """
    Add trial index/regressor features derived from omissions and choices.
    """
    right_omissions = _get_column_or_raise(augmented_trial_df, ("right_omissions",)).to_numpy()
    left_omissions = _get_column_or_raise(augmented_trial_df, ("left_omissions",)).to_numpy()
    right_omissions_cf = _get_column_or_raise(augmented_trial_df, ("right_cf_omissions",)).to_numpy()
    left_omissions_cf = _get_column_or_raise(augmented_trial_df, ("left_cf_omissions",)).to_numpy()
    loss_streak = _get_column_or_raise(augmented_trial_df, ("consecutive_omissions",)).to_numpy()
    actions = _get_column_or_raise(augmented_trial_df, ("action",)).to_numpy()

    n_trials = augmented_trial_df.shape[0]
    give_reward = augmented_trial_df["give_reward"].to_numpy() if "give_reward" in augmented_trial_df.columns else np.zeros(n_trials)

    skip_mask = np.array(
        [
            _is_give_reward_flag(gr) or str(action).strip().lower() == "none"
            for gr, action in zip(give_reward, actions)
        ],
        dtype=bool,
    )
    valid_mask = ~skip_mask

    relative_omissions_index = np.full(n_trials, None, dtype=object)
    signed_omission_regressor = np.full(n_trials, None, dtype=object)
    relative_doubt_index = np.full(n_trials, None, dtype=object)
    perseveration_regressor = np.full(n_trials, None, dtype=object)

    rel_omission_valid = trial_features.relative_omissions_index(
        R_omissions=right_omissions_cf[valid_mask],
        L_omissions=left_omissions_cf[valid_mask],
    )
    signed_omission_valid = trial_features.signed_omission_regressor(
        loss_streak=loss_streak[valid_mask],
        lam=omission_lam,
        choice_side=actions[valid_mask],
    )
    rel_doubt_valid = trial_features.relative_doubt_index(
        R_omissions=right_omissions_cf[valid_mask],
        L_omissions=left_omissions_cf[valid_mask],
        lam=omission_lam,
    )
    perseveration_valid = trial_features.perseveration_regressor(
        choices=actions[valid_mask],
        decay=perseveration_decay,
    )

    relative_omissions_index[valid_mask] = rel_omission_valid
    signed_omission_regressor[valid_mask] = signed_omission_valid
    relative_doubt_index[valid_mask] = rel_doubt_valid
    perseveration_regressor[valid_mask] = perseveration_valid

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df["relative_omissions_index"] = relative_omissions_index
    augmented_trial_df["signed_omission_regressor"] = signed_omission_regressor
    augmented_trial_df["relative_doubt_index"] = relative_doubt_index
    augmented_trial_df["perseveration_regressor"] = perseveration_regressor
    return augmented_trial_df


def collect_trial_features(augmented_trial_df: pd.DataFrame, params: Optional[TaskParams] = None) -> tuple[pd.DataFrame, TaskParams]:
    """
    Add model-derived relative value features to an existing augmented trial dataframe.
    This function is intended for import/use from other files.
    """
    if params is None:
        params = TaskParams()

    if 'action' not in augmented_trial_df.columns or 'reward' not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df must contain 'action' and 'reward' columns.")

    actions = augmented_trial_df['action'].values
    rewards = augmented_trial_df['reward'].values
    give_reward = augmented_trial_df['give_reward'].values if 'give_reward' in augmented_trial_df.columns else None

    ql_rel_value = trial_features.qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        learning_rate=params.QL_learning_rate,
        n_actions=params.n_actions,
        give_reward=give_reward,
    )
    # Standard forgetting-Q: decay=.7 and default reward update rate (1 - decay).
    fql_rel_value = trial_features.forgetting_qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        decay=params.FQL_decay,
        n_actions=params.n_actions,
        give_reward=give_reward,
    )
    # Fast-learn forgetting-Q: same decay but explicit reward update rate.
    fql_rel_value_fast_learn = trial_features.forgetting_qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        decay=params.FQL_decay,
        reward_update_rate=params.FQL_reward_update_rate_fast_learn,
        n_actions=params.n_actions,
        give_reward=give_reward,
    )
    hmm_rel_value_logodds = trial_features.hmm_relative_value(
        actions=actions,
        rewards=rewards,
        state_transition_prob=params.state_transition_prob,
        active_reward_probability=params.active_reward_probability,
        inactive_reward_probability=params.inactive_reward_probability,
        correct_reward_size=params.correct_reward_size,
        incorrect_reward_size=params.incorrect_reward_size,
        value_mode='bayesian_log_odds',
        tanh_scale=params.tanh_scale,
        give_reward=give_reward,
    )
    hmm_rel_value_logodds_decay = trial_features.hmm_relative_value_reward_decay(
        actions=actions,
        rewards=rewards,
        state_transition_prob=params.state_transition_prob,
        active_reward_probability=params.active_reward_probability,
        inactive_reward_probability=params.inactive_reward_probability,
        correct_reward_size=params.correct_reward_size,
        incorrect_reward_size=params.incorrect_reward_size,
        lambda_decay=params.hmm_reward_decay_lambda,
        value_mode='bayesian_log_odds',
        tanh_scale=params.tanh_scale,
        give_reward=give_reward,
    )

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df['Qlearning_rel_value'] = ql_rel_value
    augmented_trial_df['FQlearning_rel_value'] = fql_rel_value
    augmented_trial_df['FQlearning_rel_value_fast_learn'] = fql_rel_value_fast_learn
    augmented_trial_df['HMM_rel_value_logodds'] = hmm_rel_value_logodds
    augmented_trial_df['HMM_rel_value_logodds_decay'] = hmm_rel_value_logodds_decay

    augmented_trial_df = collect_trial_index_features(
        augmented_trial_df,
        omission_lam=params.omission_lam,
        perseveration_decay=params.perseveration_decay,
    )
    return augmented_trial_df, params


def main():
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference/')
    # sess_id_full = 'CT014_2025-12-16_153200'
    # sess_id_full = 'CT014_2025-12-05_165240'
    sess_id_full = 'CT014_2025-12-23_163505'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
    else:
        print("Double-check the session name!")
        return

    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)

    params = TaskParams()
    params.p_cue = 0
    params.correct_reward_size = 1
    params.incorrect_reward_size = 0
    params.QL_learning_rate = .3

    augmented_trial_df, params = collect_and_save_trial_features(
        augmented_trial_df,
        processed_data_path=processed_data_path,
        sess_id_full=sess_id_full,
        params=params,
    )


if __name__ == '__main__':
    main()
