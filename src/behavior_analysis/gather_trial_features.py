import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from pathlib import Path
import src.behavior_analysis.trial_features as trial_features
from src.behavior_analysis.project_utils import (
    EXPERIMENTER_REWARD_GIVEN_COLUMN,
    get_experimenter_reward_flags,
    normalize_experimenter_reward_column,
)
import json
from typing import Optional
from src.behavior_analysis import residualization


@dataclass
class TaskParams:
    n_states: int = 2  # could be 4 if I use the stimulus version, but that could also be 2 states but in some cases having information
    n_actions: int = 2  # in practice I am not planning more than 2 choices, could I remove this?

    # Probability parameters
    # For HHM inference model, you can play with using model parameters that are different from the true task parameters
    p_cue: float = 0  # 0 for uncued task, 1 for cued, between 0 and 1 for mixed. Must be between 0 and 1
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
    hazard_lam: float = 0.5
    perseveration_decay: float = 0.25


def assert_saved_file(path: Path) -> None:
    """Verify that a save produced a non-empty file.

    Parameters
    ----------
    path : Path
        Expected output path.

    Returns
    -------
    None
        Returns None when the file exists and has nonzero byte size.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected saved file was not found: {path}")
    if path.stat().st_size == 0:
        raise OSError(f"Expected saved file is empty: {path}")


def save_trial_features(augmented_trial_df: pd.DataFrame, params: TaskParams, processed_data_path: Path,
                        sess_id_full: str) -> None:
    """Save augmented trial features and feature-generation parameters.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with feature columns, shape `(n_trials, n_columns)`.
    params : TaskParams
        Parameters used to generate feature columns.
    processed_data_path : Path
        Session processed-data directory.
    sess_id_full : str
        Full session identifier used in the augmented-trials filename.

    Returns
    -------
    None
        Writes `{sess_id_full}_augmented_trials.csv` and
        `trial_feature_params.json`.
    """
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    augmented_trial_df.to_csv(augmented_trial_df_path, index=False, na_rep='None')

    json_fname = processed_data_path / 'trial_feature_params.json'
    with open(json_fname, "w") as f:
        json.dump(asdict(params), f, indent=2)

    assert_saved_file(augmented_trial_df_path)
    assert_saved_file(json_fname)


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


def validate_trial_feature_inputs(augmented_trial_df: pd.DataFrame) -> None:
    """Validate columns required for model-value feature generation.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.

    Returns
    -------
    None
        Raises ValueError if required columns are missing.
    """
    if 'action' not in augmented_trial_df.columns or 'reward' not in augmented_trial_df.columns:
        raise ValueError("augmented_trial_df must contain 'action' and 'reward' columns.")


def collect_trial_index_features(
    augmented_trial_df: pd.DataFrame,
    omission_lam: float = 0.5,
    hazard_lam: float = 0.5,
    perseveration_decay: float = 0.25,
) -> pd.DataFrame:
    """Add trial index/regressor features derived from trial-history counters.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns include side-specific omission counters, `consecutive_omissions`,
        `relative_monotonic_cf_value`, `action`, and optional
        `experimenter_reward_given`.
    omission_lam : float, default=0.5
        Positive saturation parameter for omission-derived regressors.
    hazard_lam : float, default=0.5
        Positive saturation parameter for `relative_hazard_index`.
    perseveration_decay : float, default=0.25
        Exponential decay parameter for the perseveration regressor.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with omission, hazard, and perseveration
        regressor columns added. Regressors use left-positive sign convention.
    """
    right_omissions = _get_column_or_raise(augmented_trial_df, ("right_omissions",)).to_numpy()
    left_omissions = _get_column_or_raise(augmented_trial_df, ("left_omissions",)).to_numpy()
    right_omissions_counterfactual = _get_column_or_raise(augmented_trial_df, ("right_cf_omissions",)).to_numpy()
    left_omissions_counterfactual = _get_column_or_raise(augmented_trial_df, ("left_cf_omissions",)).to_numpy()
    loss_streak = _get_column_or_raise(augmented_trial_df, ("consecutive_omissions",)).to_numpy()
    relative_monotonic_cf_value = _get_column_or_raise(
        augmented_trial_df,
        ("relative_monotonic_cf_value",),
    ).to_numpy()
    actions = _get_column_or_raise(augmented_trial_df, ("action",)).to_numpy()

    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    n_trials = augmented_trial_df.shape[0]
    experimenter_reward_given = get_experimenter_reward_flags(augmented_trial_df)

    skip_mask = trial_features.make_skip_trial_mask(experimenter_reward_given=experimenter_reward_given, actions=actions)
    if skip_mask is None:
        skip_mask = np.zeros(n_trials, dtype=bool)
    valid_mask = ~skip_mask

    relative_omissions_index = np.full(n_trials, None, dtype=object)
    signed_omission_regressor = np.full(n_trials, None, dtype=object)
    relative_doubt_index = np.full(n_trials, None, dtype=object)
    relative_hazard_index = np.full(n_trials, None, dtype=object)
    perseveration_regressor = np.full(n_trials, None, dtype=object)

    rel_omission_valid = trial_features.relative_omissions_index(
        R_omissions=right_omissions_counterfactual[valid_mask],
        L_omissions=left_omissions_counterfactual[valid_mask],
    )
    signed_omission_valid = trial_features.signed_omission_regressor(
        loss_streak=loss_streak[valid_mask],
        lam=omission_lam,
        choice_side=actions[valid_mask],
    )
    rel_doubt_valid = trial_features.relative_doubt_index(
        R_omissions=right_omissions_counterfactual[valid_mask],
        L_omissions=left_omissions_counterfactual[valid_mask],
        lam=omission_lam,
    )
    rel_hazard_valid = trial_features.relative_hazard_index(
        relative_monotonic_cf_value=relative_monotonic_cf_value[valid_mask],
        lam=hazard_lam,
    )
    perseveration_valid = trial_features.perseveration_regressor(
        choices=actions[valid_mask],
        decay=perseveration_decay,
    )

    relative_omissions_index[valid_mask] = rel_omission_valid
    signed_omission_regressor[valid_mask] = signed_omission_valid
    relative_doubt_index[valid_mask] = rel_doubt_valid
    relative_hazard_index[valid_mask] = rel_hazard_valid
    perseveration_regressor[valid_mask] = perseveration_valid

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df["relative_omissions_index"] = relative_omissions_index
    augmented_trial_df["signed_omission_regressor"] = signed_omission_regressor
    augmented_trial_df["relative_doubt_index"] = relative_doubt_index
    augmented_trial_df["relative_hazard_index"] = relative_hazard_index
    augmented_trial_df["perseveration_regressor"] = perseveration_regressor
    return augmented_trial_df


def collect_model_value_features(
    augmented_trial_df: pd.DataFrame,
    params: TaskParams,
) -> pd.DataFrame:
    """Add model-derived relative-value features to an augmented trial table.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`, including
        `action`, `reward`, and optionally `experimenter_reward_given`.
    params : TaskParams
        Parameters used by Q-learning and HMM feature functions.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with model-value feature columns added.
    """
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    validate_trial_feature_inputs(augmented_trial_df)
    actions = augmented_trial_df['action'].values
    rewards = augmented_trial_df['reward'].values
    experimenter_reward_given = (
        augmented_trial_df[EXPERIMENTER_REWARD_GIVEN_COLUMN].values
        if EXPERIMENTER_REWARD_GIVEN_COLUMN in augmented_trial_df.columns
        else None
    )

    ql_rel_value = trial_features.qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        learning_rate=params.QL_learning_rate,
        n_actions=params.n_actions,
        experimenter_reward_given=experimenter_reward_given,
    )
    # Standard forgetting-Q: decay=.7 and default reward update rate (1 - decay).
    fql_rel_value = trial_features.forgetting_qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        decay=params.FQL_decay,
        n_actions=params.n_actions,
        experimenter_reward_given=experimenter_reward_given,
    )
    # Fast-learn forgetting-Q: same decay but explicit reward update rate.
    fql_rel_value_fast_learn = trial_features.forgetting_qlearning_relative_value(
        actions=actions,
        rewards=rewards,
        decay=params.FQL_decay,
        reward_update_rate=params.FQL_reward_update_rate_fast_learn,
        n_actions=params.n_actions,
        experimenter_reward_given=experimenter_reward_given,
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
        experimenter_reward_given=experimenter_reward_given,
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
        experimenter_reward_given=experimenter_reward_given,
    )

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df['Qlearning_rel_value'] = ql_rel_value
    augmented_trial_df['FQlearning_rel_value'] = fql_rel_value
    augmented_trial_df['FQlearning_rel_value_fast_learn'] = fql_rel_value_fast_learn
    augmented_trial_df['HMM_rel_value_logodds'] = hmm_rel_value_logodds
    augmented_trial_df['HMM_rel_value_logodds_decay'] = hmm_rel_value_logodds_decay
    return augmented_trial_df


def collect_residualized_trial_features(augmented_trial_df: pd.DataFrame) -> pd.DataFrame:
    """Add residualized value predictors to an augmented trial table.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `FQlearning_rel_value`, `HMM_rel_value_logodds_decay`,
        and `relative_hazard_index`.

    Returns
    -------
    pd.DataFrame
        Copy of `augmented_trial_df` with `HMM_decay_res` and
        `rel_hazard_res` columns added. Rows with missing/non-numeric source
        values are kept and receive `None` in residualized columns.
    """
    required_columns = [
        "FQlearning_rel_value",
        "HMM_rel_value_logodds_decay",
        "relative_hazard_index",
    ]
    missing_columns = [column for column in required_columns if column not in augmented_trial_df.columns]
    if missing_columns:
        raise ValueError(f"augmented_trial_df is missing required residualization columns: {missing_columns}")

    numeric_features = augmented_trial_df[required_columns].apply(pd.to_numeric, errors="coerce")
    n_trials = augmented_trial_df.shape[0]
    hmm_decay_res = np.full(n_trials, None, dtype=object)
    rel_hazard_res = np.full(n_trials, None, dtype=object)

    # Residualize overlapping value representations so trial GLMs can compare
    # HMM-unique and hazard-unique components without replacing the raw predictors.
    valid_hmm_rows = numeric_features[["FQlearning_rel_value", "HMM_rel_value_logodds_decay"]].notna().all(axis=1)
    if valid_hmm_rows.any():
        hmm_decay_res[valid_hmm_rows.to_numpy()] = residualization.residualize_values(
            target_values=numeric_features.loc[valid_hmm_rows, "HMM_rel_value_logodds_decay"],
            control_regressors=numeric_features.loc[valid_hmm_rows, ["FQlearning_rel_value"]],
        )

    residualized_feature_frame = numeric_features.copy()
    residualized_feature_frame["HMM_decay_res"] = pd.to_numeric(hmm_decay_res, errors="coerce")
    valid_hazard_rows = residualized_feature_frame[
        ["FQlearning_rel_value", "HMM_decay_res", "relative_hazard_index"]
    ].notna().all(axis=1)
    if valid_hazard_rows.any():
        rel_hazard_res[valid_hazard_rows.to_numpy()] = residualization.residualize_values(
            target_values=residualized_feature_frame.loc[valid_hazard_rows, "relative_hazard_index"],
            control_regressors=residualized_feature_frame.loc[
                valid_hazard_rows,
                ["FQlearning_rel_value", "HMM_decay_res"],
            ],
        )

    augmented_trial_df = augmented_trial_df.copy()
    augmented_trial_df["HMM_decay_res"] = hmm_decay_res
    augmented_trial_df["rel_hazard_res"] = rel_hazard_res
    return augmented_trial_df


def collect_trial_features(augmented_trial_df: pd.DataFrame, params: Optional[TaskParams] = None) -> tuple[pd.DataFrame, TaskParams]:
    """
    Add model-derived relative value features to an existing augmented trial dataframe.
    This function is intended for import/use from other files.
    """
    augmented_trial_df = normalize_experimenter_reward_column(augmented_trial_df)
    if params is None:
        params = TaskParams()

    augmented_trial_df = collect_model_value_features(augmented_trial_df, params=params)
    augmented_trial_df = collect_trial_index_features(
        augmented_trial_df,
        omission_lam=params.omission_lam,
        hazard_lam=params.hazard_lam,
        perseveration_decay=params.perseveration_decay,
    )
    augmented_trial_df = collect_residualized_trial_features(augmented_trial_df)
    return augmented_trial_df, params


def main():
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251223_latentInference/')
    # sess_id_full = 'CT014_2025-12-16_153200'
    # sess_id_full = 'CT014_2025-12-05_165240'
    sess_id_full = 'CT014_2025-12-23_163505'
    processed_data_path = session_data_home / 'processed'

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
