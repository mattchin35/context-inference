from pathlib import Path
import pandas as pd
import numpy as np
import re
import warnings
from collections import defaultdict
import src.behavior_analysis.decision_variable_counters as counters
from src.behavior_analysis.project_utils import is_present_value
from formulaic import model_matrix
import statsmodels.api as sm
from dataclasses import dataclass
from typing import Protocol

eps = np.finfo(float).eps


class Session(Protocol):
    multi_session_save_path: Path
    session_data_home: Path
    sess_id_full: str
    raw_behavior_folder: Path
    processed_data_path: Path
    figure_path: Path
    mouse: str
    date: str
    timestamp: str
    session_info_fname: str
    session_info: dict


@dataclass
class DecisionVariableState:
    """Mutable trial-history counters used to build decision variables."""
    consecutive_rewards: int = 0
    consecutive_omissions: int = 0
    consecutive_rewards_memory: int = 0
    consecutive_omissions_memory: int = 0
    negative_value: int = 0
    previous_trial_rewarded: bool = False
    left_value: int = 0
    right_value: int = 0
    left_omissions: int = 0
    right_omissions: int = 0
    left_value_cf: int = 0
    right_value_cf: int = 0
    left_omissions_cf: int = 0
    right_omissions_cf: int = 0


def _get_normalized_state_labels(trial_df: pd.DataFrame) -> np.ndarray:
    """Normalize real-session and simulated state labels to `left`/`right`.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table containing a `state` column. Real sessions use
        `left_patch`/`right_patch`; simulated runs use `left`/`right`.

    Returns
    -------
    np.ndarray
        Normalized state labels with shape `(n_trials,)` and values drawn from
        `{"left", "right"}` where possible. Unknown labels are preserved so
        downstream boolean masks simply evaluate to False.
    """
    normalized_states = trial_df["state"].astype(str).to_numpy(dtype=object)
    normalized_states[normalized_states == "left_patch"] = "left"
    normalized_states[normalized_states == "right_patch"] = "right"
    return normalized_states


def _get_uncued_block_mask(trial_df: pd.DataFrame) -> np.ndarray:
    """Return the uncued-block mask for real or simulated trial schemas.

    Adapter logic:
    - real sessions: uncued blocks have null-like `block_stimulus`
    - simulated runs: uncued blocks have `model_stimulus == -1`

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table containing either `block_stimulus` or `model_stimulus`.

    Returns
    -------
    np.ndarray
        Boolean mask with shape `(n_trials,)`, where True marks uncued trials.
    """
    if "block_stimulus" in trial_df.columns:
        stimulus = trial_df["block_stimulus"]
        return stimulus.isnull().to_numpy() | stimulus.astype(str).eq("None").to_numpy()

    if "model_stimulus" in trial_df.columns:
        stimulus = trial_df["model_stimulus"]
        return stimulus.isnull().to_numpy() | stimulus.astype(str).eq("-1").to_numpy()

    raise ValueError(
        "trial_df must contain either 'block_stimulus' (real sessions) or "
        "'model_stimulus' (simulated runs)."
    )


def _get_give_reward_array(trial_df: pd.DataFrame) -> np.ndarray:
    """Return experimenter-given reward flags, defaulting missing values to zero.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table that may or may not contain a `give_reward` column.

    Returns
    -------
    np.ndarray
        Array of shape `(n_trials,)`. Missing columns are treated as zeros,
        which is the correct behavior for simulated runs.
    """
    if "give_reward" not in trial_df.columns:
        return np.zeros(trial_df.shape[0], dtype=int)
    return np.copy(trial_df["give_reward"].to_numpy())


def _get_choice_latency(trial_df: pd.DataFrame) -> np.ndarray:
    """Return trialwise choice latencies or NaN when timing columns are absent."""
    if {"choice_time", "start_time"}.issubset(trial_df.columns):
        return (
            trial_df["choice_time"].to_numpy(dtype=float)
            - trial_df["start_time"].to_numpy(dtype=float)
        )
    return np.full(trial_df.shape[0], np.nan, dtype=float)


def get_block_types(trial_df: pd.DataFrame) -> np.array:
    """Assign block types for real or simulated runs.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table containing a `state` column and either:
        - `block_stimulus` for real sessions
        - `model_stimulus` for simulated runs

    Returns
    -------
    np.ndarray
        Block-type labels with shape `(n_trials,)`, values in
        `{"left_cued", "left_uncued", "right_cued", "right_uncued"}`.
    """
    normalized_states = _get_normalized_state_labels(trial_df)
    uncued_block = _get_uncued_block_mask(trial_df)
    cued_block = ~uncued_block
    left_cued_ix = (normalized_states == 'left') & cued_block
    left_uncued_ix = (normalized_states == 'left') & uncued_block
    right_cued_ix = (normalized_states == 'right') & cued_block
    right_uncued_ix = (normalized_states == 'right') & uncued_block

    block_types = np.zeros(trial_df.shape[0], dtype=object)
    block_types[left_cued_ix] = 'left_cued'
    block_types[right_cued_ix] = 'right_cued'
    block_types[left_uncued_ix] = 'left_uncued'
    block_types[right_uncued_ix] = 'right_uncued'
    return block_types


def percent_correct(augmented_trial_df: pd.DataFrame) -> dict:
    """Calculate the percentage of correct choices made by the agent. Calculate for
    left uncued, right uncued, left cued, and right cued."""
    assert 'block_type' in augmented_trial_df.keys(), "'block_type' key was not found in dataframe."
    assert 'correct' in augmented_trial_df.keys(), "'correct' key was not found in dataframe."

    left_cued_ix = augmented_trial_df['block_type'] == 'left_cued'
    left_uncued_ix = augmented_trial_df['block_type'] == 'left_uncued'
    right_cued_ix = augmented_trial_df['block_type'] == 'right_cued'
    right_uncued_ix = augmented_trial_df['block_type'] == 'right_uncued'

    if left_cued_ix.sum():
        left_cued_correct = np.sum(augmented_trial_df.loc[left_cued_ix, 'correct']) / np.sum(left_cued_ix)
    else:
        left_cued_correct = 'None'

    if left_uncued_ix.sum():
        left_uncued_correct = np.sum(augmented_trial_df.loc[left_uncued_ix, 'correct']) / np.sum(left_uncued_ix)
    else:
        left_uncued_correct = 'None'

    if right_cued_ix.sum():
        right_cued_correct = np.sum(augmented_trial_df.loc[right_cued_ix, 'correct']) / np.sum(right_cued_ix)
    else:
        right_cued_correct = 'None'

    if right_uncued_ix.sum():
        right_uncued_correct = np.sum(augmented_trial_df.loc[right_uncued_ix, 'correct']) / np.sum(right_uncued_ix)
    else:
        right_uncued_correct = 'None'

    overall = np.sum(augmented_trial_df['correct']) / augmented_trial_df.shape[0]
    return dict(left_cued_correct=left_cued_correct, left_uncued_correct=left_uncued_correct,
                right_cued_correct=right_cued_correct, right_uncued_correct=right_uncued_correct,
                overall_correct=overall)


def add_numeric_trials_to_correct(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with numeric trials-to-correct values.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `trials_to_correct`.

    Returns
    -------
    pd.DataFrame
        Copy with `trials_to_correct_numeric`; invalid values are `NaN`.
    """
    summary_df = block_performance.copy()
    summary_df["trials_to_correct_numeric"] = pd.to_numeric(
        summary_df["trials_to_correct"],
        errors="coerce",
    )
    return summary_df


def get_nonfinal_missing_trials_to_correct_block_ix(summary_df: pd.DataFrame) -> list[int]:
    """Return non-final block IDs with missing trials-to-correct values.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `trials_to_correct_numeric` and optionally `block_ix`.

    Returns
    -------
    list[int]
        Block IDs for missing `trials_to_correct_numeric` rows, excluding the
        final row.
    """
    missing_trials_to_correct = summary_df["trials_to_correct_numeric"].isna()
    nonfinal_missing = missing_trials_to_correct.copy()
    if nonfinal_missing.size:
        nonfinal_missing.iloc[-1] = False

    if "block_ix" in summary_df.columns:
        return summary_df.loc[nonfinal_missing, "block_ix"].astype(int).tolist()

    return summary_df.index[nonfinal_missing].astype(int).tolist()


def summarize_missing_trials_to_correct_blocks(nonfinal_block_ix: list[int]) -> dict:
    """Return warning metadata for non-final missing trials-to-correct blocks.

    Parameters
    ----------
    nonfinal_block_ix : list[int]
        Non-final block IDs with missing trials-to-correct values.

    Returns
    -------
    dict
        Warning metadata using the public session-summary key names.
    """
    warning_flag = len(nonfinal_block_ix) > 0
    if warning_flag:
        warning_message = (
            "Non-final invalid trials_to_correct entries detected; "
            f"excluding them from summaries. block_ix={nonfinal_block_ix}"
        )
        warnings.warn(warning_message, UserWarning, stacklevel=3)
        nonfinal_block_ix_str = str(nonfinal_block_ix)
    else:
        warning_message = "None"
        nonfinal_block_ix_str = "None"

    return dict(
        trials_to_correct_warning_flag=warning_flag,
        trials_to_correct_warning_message=warning_message,
        trials_to_correct_nonfinal_invalid_block_ix=nonfinal_block_ix_str,
    )


def mean_trials_to_correct_for_block_type(summary_df: pd.DataFrame, block_type: str) -> float | str:
    """Return mean trials-to-correct for one block type.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `block_type` and `trials_to_correct_numeric`.
    block_type : str
        Block type to summarize.

    Returns
    -------
    float or str
        Mean valid trials-to-correct value, or `"None"` when no valid values
        exist for `block_type`.
    """
    values = summary_df.loc[
        summary_df["block_type"] == block_type,
        "trials_to_correct_numeric",
    ].dropna()
    return values.mean() if not values.empty else "None"


def mean_trials_to_correct(summary_df: pd.DataFrame) -> float | str:
    """Return overall mean trials-to-correct across valid blocks.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Block summary dataframe with shape `(n_blocks, n_columns)`, including
        `trials_to_correct_numeric`.

    Returns
    -------
    float or str
        Mean valid trials-to-correct value, or `"None"` when no valid values
        exist.
    """
    valid_trials = summary_df["trials_to_correct_numeric"].dropna()
    return valid_trials.mean() if not valid_trials.empty else "None"


def summarize_trials_to_correct(block_performance: pd.DataFrame) -> dict:
    """Summarize blockwise trials-to-correct while tolerating invalid sentinels.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise summary dataframe with shape `(n_blocks, n_columns)`
        containing:
        - `block_type` : str, block category for each block
        - `trials_to_correct` : int-like or sentinel values such as `'None'`
        - `block_ix` : optional integer-like block identifier used for
          warning metadata

    Returns
    -------
    dict
        Mean trials-to-correct by block type and overall. Invalid
        `trials_to_correct` values are ignored for numeric summaries.
        Conditions with no valid numeric values return `'None'`.
        Non-final invalid values emit a warning and are reported via:
        - `trials_to_correct_warning_flag` : bool
        - `trials_to_correct_warning_message` : str or `'None'`
        - `trials_to_correct_nonfinal_invalid_block_ix` : str or `'None'`
    """
    summary_df = add_numeric_trials_to_correct(block_performance)
    nonfinal_missing_block_ix = get_nonfinal_missing_trials_to_correct_block_ix(summary_df)
    missing_summary = summarize_missing_trials_to_correct_blocks(nonfinal_missing_block_ix)
    return dict(
        left_cued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "left_cued"),
        left_uncued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "left_uncued"),
        right_cued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "right_cued"),
        right_uncued_trials_to_correct=mean_trials_to_correct_for_block_type(summary_df, "right_uncued"),
        overall_trials_to_correct=mean_trials_to_correct(summary_df),
        **missing_summary,
    )


def get_block_switches(trial_df: pd.DataFrame) -> tuple[int, int]:
    # handle any give_reward trials
    actions = np.copy(trial_df['action'].values)
    give_reward = _get_give_reward_array(trial_df)
    for i, (a, g) in enumerate(zip(actions, give_reward)):
        if g in [1, '1']:
            try:
                if i == 0:  # I'll need a better handling of index 0 in the future
                    actions[i] = actions[i + 1]
                    continue

                actions[i] = actions[i - 1]
            except IndexError:
                actions[i] = actions[i + 1]

    actions = actions.astype(int)
    n_switches = np.sum(np.abs(np.diff(actions)))
    if trial_df.shape[0] > 1:
        normalized_switches = n_switches / (trial_df.shape[0] - 1)  # must subtract 1 for n-1 opportunites to switch
    else:
        normalized_switches = 0
    return n_switches, normalized_switches


def make_augmented_trial_df(trial_df: pd.DataFrame) -> pd.DataFrame:
    augmented_trial_df = trial_df.copy(deep=True)
    block_types = get_block_types(trial_df)
    augmented_trial_df['block_type'] = block_types
    augmented_trial_df['time_to_choice'] = _get_choice_latency(trial_df)

    prev_action = np.zeros(trial_df.shape[0], dtype='object')
    prev_reward = np.zeros(trial_df.shape[0], dtype='object')
    prev_action[0] = 'None'
    prev_reward[0] = 'None'
    prev_action[1:] = trial_df['action'].to_numpy()[:-1]
    prev_reward[1:] = trial_df['reward'].to_numpy()[:-1]
    augmented_trial_df['prev_action'] = prev_action
    augmented_trial_df['prev_reward'] = prev_reward

    decision_vars = count_decision_variables(trial_df)
    augmented_trial_df = pd.concat([augmented_trial_df, decision_vars], axis=1)

    return augmented_trial_df


def summarize_block_performance(augmented_trial_df: pd.DataFrame, session_id: str) -> pd.DataFrame:
    """Summarize performance for each task block in one session.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns include
        `cur_block`, `block_type`, `correct`, `reward`, `time_to_choice`, and
        decision-variable columns from `make_augmented_trial_df`.
    session_id : str
        Session identifier copied into the `session_ID` output column.

    Returns
    -------
    pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
    """
    choice_latency = augmented_trial_df['time_to_choice'].to_numpy(dtype=float)
    blocks = np.unique(augmented_trial_df['cur_block'])
    block_performance = []

    for ix, b in enumerate(blocks):
        cur_block_ix = augmented_trial_df['cur_block'] == b
        cur_block_df = augmented_trial_df[cur_block_ix]

        if ix == 0:
            prev_n_correct = 0
            prev_n_rewarded = 0
            prev_consecutive_rewards = 0
            prev_consecutive_rewards_memory = 0
        else:
            prev_n_correct = performance['n_correct']
            prev_n_rewarded = performance['n_rewarded']
            prev_consecutive_rewards = cur_block_df['consecutive_rewards'].values[0]
            prev_consecutive_rewards_memory = cur_block_df['consecutive_rewards_memory'].values[0]

        performance = percent_correct(cur_block_df)
        n_switches, normalized_switches = get_block_switches(cur_block_df)
        correct_ix = np.nonzero(cur_block_df['correct'])[0]
        if correct_ix.size:
            trials_to_correct = correct_ix[0]
        else:
            trials_to_correct = 'None' #np.nan  # could occur on last block in session, or when the task switches to dark mode

        performance = dict(block_ix=b, block_type=cur_block_df['block_type'].values[0],
                           trials_to_correct=trials_to_correct,
                           prev_consecutive_rewards=prev_consecutive_rewards,
                           prev_consecutive_rewards_memory=prev_consecutive_rewards_memory,
                           prev_n_correct=prev_n_correct,
                           prev_n_rewarded=prev_n_rewarded,
                           n_switches=n_switches,
                           normalized_switches=normalized_switches,
                           confusion_flag=n_switches > 3,
                           n_correct=np.sum(cur_block_df['correct']),
                           percent_correct=performance['overall_correct'],
                           n_rewarded=np.sum(cur_block_df['reward']),
                           mean_choice_time=np.mean(choice_latency[cur_block_ix.to_numpy()]),
                           median_choice_time=np.median(choice_latency[cur_block_ix.to_numpy()]),
                           std_choice_time=np.std(choice_latency[cur_block_ix.to_numpy()]),
                           session_ID=session_id)

        block_performance.append(performance)

    block_performance = pd.DataFrame(block_performance)
    block_performance['block_ix'] = np.arange(len(block_performance))
    return block_performance


def summarize_session_performance(
    augmented_trial_df: pd.DataFrame,
    block_performance: pd.DataFrame,
    date: str,
) -> pd.DataFrame:
    """Summarize whole-session performance from trial and block tables.

    Parameters
    ----------
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`, including `block_type`
        and `correct`.
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`,
        including `trials_to_correct` and `block_type`.
    date : str
        Session date copied into the `date` output column.

    Returns
    -------
    pd.DataFrame
        One-row session performance table.
    """
    session_performance = percent_correct(augmented_trial_df)
    session_performance = session_performance | summarize_trials_to_correct(block_performance)
    session_performance['date'] = date
    return pd.DataFrame(session_performance, index=[0])


def analyze_session(trial_df: pd.DataFrame, mouse: str, date: str) -> tuple:
    """Build trial, block, and session performance summaries.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns
        include `state`, `action`, `reward`, `correct`, and `cur_block`.
    mouse : str
        Mouse or simulated-agent identifier copied into the block-level
        `session_ID` prefix.
    date : str
        Session date copied into the session summary and block-level
        `session_ID`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(session_performance, block_performance, augmented_trial_df)`, where
        `session_performance` has one row, `block_performance` has shape
        `(n_blocks, n_columns)`, and `augmented_trial_df` has shape
        `(n_trials, n_columns)`.
    """
    session_id = mouse + '_' + date
    augmented_trial_df = make_augmented_trial_df(trial_df)
    block_performance = summarize_block_performance(augmented_trial_df, session_id=session_id)
    session_performance = summarize_session_performance(augmented_trial_df, block_performance, date=date)

    return session_performance, block_performance, augmented_trial_df


def append_decision_variables(
    decision_variable_dict: dict[str, list],
    state: DecisionVariableState,
) -> None:
    """Append the current pre-trial decision-variable state.

    Parameters
    ----------
    decision_variable_dict : dict[str, list]
        Output accumulator with one list per decision-variable column.
    state : DecisionVariableState
        Current trial-history state before the current trial outcome is used.

    Returns
    -------
    None
        Mutates `decision_variable_dict` in place.
    """
    decision_variable_dict['negative_value'].append(state.negative_value)
    decision_variable_dict['consecutive_rewards_memory'].append(state.consecutive_rewards_memory)
    decision_variable_dict['consecutive_omissions_memory'].append(state.consecutive_omissions_memory)
    decision_variable_dict['consecutive_rewards'].append(state.consecutive_rewards)
    decision_variable_dict['consecutive_omissions'].append(state.consecutive_omissions)
    decision_variable_dict['left_value'].append(state.left_value)
    decision_variable_dict['right_value'].append(state.right_value)
    decision_variable_dict['relative_value'].append(state.left_value - state.right_value)
    decision_variable_dict['left_omissions'].append(state.left_omissions)
    decision_variable_dict['right_omissions'].append(state.right_omissions)
    decision_variable_dict['relative_omissions'].append(state.left_omissions - state.right_omissions)
    decision_variable_dict['left_cf_value'].append(state.left_value_cf)
    decision_variable_dict['right_cf_value'].append(state.right_value_cf)
    decision_variable_dict['relative_cf_value'].append(state.left_value_cf - state.right_value_cf)
    decision_variable_dict['left_cf_omissions'].append(state.left_omissions_cf)
    decision_variable_dict['right_cf_omissions'].append(state.right_omissions_cf)
    decision_variable_dict['relative_cf_omissions'].append(state.left_omissions_cf - state.right_omissions_cf)


def should_skip_decision_variable_update(give_reward: int | str, action: int | str) -> bool:
    """Return whether a trial should leave decision-variable state unchanged.

    Parameters
    ----------
    give_reward : int or str
        Experimenter-reward flag for one trial.
    action : int or str
        Animal action for one trial; `"None"` marks no animal choice.

    Returns
    -------
    bool
        True when the trial should not update history counters.
    """
    return give_reward in [1, '1'] or action == 'None'


def increment_reward_history_decision_vars(
    state: DecisionVariableState,
    reward: int,
) -> DecisionVariableState:
    """Update reward-history counters from one trial outcome.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after reward-history updates.
    """
    state.negative_value = counters.negative_value_counter(state.negative_value, reward)
    state.consecutive_rewards = counters.consecutive_reward_counter(state.consecutive_rewards, reward)
    state.consecutive_omissions = counters.consecutive_fail_counter(state.consecutive_omissions, reward)
    state.consecutive_rewards_memory = counters.consecutive_reward_renewal_counter(
        state.consecutive_rewards_memory,
        reward,
        state.previous_trial_rewarded,
    )
    state.consecutive_omissions_memory = counters.consecutive_fail_renewal_counter(
        state.consecutive_omissions_memory,
        reward,
        state.previous_trial_rewarded,
    )
    state.previous_trial_rewarded = reward > 0
    return state


def increment_choice_value_decision_vars(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update non-counterfactual side-specific counters.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after side-specific updates.
    """
    state.left_value, state.right_value = counters.sided_value_counter(
        state.left_value,
        state.right_value,
        action,
        reward,
        zero_min=True,
        counterfactual=False,
    )
    state.left_omissions, state.right_omissions = counters.sided_omissions_counter(
        state.left_omissions,
        state.right_omissions,
        action,
        reward,
        counterfactual=False,
    )
    return state


def increment_counterfactual_decision_vars(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update counterfactual side-specific counters.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after counterfactual updates.
    """
    state.left_value_cf, state.right_value_cf = counters.sided_value_counter(
        state.left_value_cf,
        state.right_value_cf,
        action,
        reward,
        zero_min=True,
        counterfactual=True,
    )
    state.left_omissions_cf, state.right_omissions_cf = counters.sided_omissions_counter(
        state.left_omissions_cf,
        state.right_omissions_cf,
        action,
        reward,
        counterfactual=True,
    )
    return state


def update_decision_variable_state(
    state: DecisionVariableState,
    action: int,
    reward: int,
) -> DecisionVariableState:
    """Update all decision-variable counters from one non-skipped trial.

    Parameters
    ----------
    state : DecisionVariableState
        Mutable decision-variable state before the current trial update.
    action : int
        Current trial action, encoded as 0 for right or 1 for left.
    reward : int
        Current trial reward, encoded as 0 or 1.

    Returns
    -------
    DecisionVariableState
        The same state object after all counter updates.
    """
    state = increment_reward_history_decision_vars(state, reward)
    state = increment_choice_value_decision_vars(state, action, reward)
    state = increment_counterfactual_decision_vars(state, action, reward)
    return state


def count_decision_variables(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Collect pre-trial decision variables for each trial.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial dataframe with shape `(n_trials, n_columns)`, including `action`,
        `reward`, and optionally `give_reward`.

    Returns
    -------
    pd.DataFrame
        Decision-variable dataframe with shape `(n_trials, n_decision_vars)`.
        Each row describes trial history before the current trial outcome.
    """
    state = DecisionVariableState()
    decision_variable_dict = defaultdict(list)
    give_reward_flags = _get_give_reward_array(trial_df)

    for i in range(trial_df.shape[0]):
        append_decision_variables(decision_variable_dict, state)

        action = trial_df.loc[i, 'action']
        if should_skip_decision_variable_update(give_reward_flags[i], action):
            continue

        state = update_decision_variable_state(
            state,
            action=action,
            reward=trial_df.loc[i, 'reward'],
        )

    return pd.DataFrame(decision_variable_dict)


def summarize_block_switches(block_performance: pd.DataFrame, min_counts=0) -> tuple:
    """This function is not used anywhere, I may delete it"""

    grouped_switches = block_performance.groupby(['consecutive_rewards'])['trials_to_correct']
    counts = grouped_switches.count()
    # ic(counts)

    ix = counts > min_counts
    mean = grouped_switches.mean()[ix]
    std = grouped_switches.std()[ix]
    sem = grouped_switches.sem()[ix]
    rewards = counts.index[ix]
    return rewards, mean, std, sem


def session_stats(dependent_var, independent_var) -> tuple[float | str, float | str, float | str, float | str]:
    """Fit a simple linear model for paired numeric session summary values.

    Parameters
    ----------
    dependent_var : array-like
        One-dimensional numeric values with shape `(n_observations,)`.
    independent_var : array-like
        One-dimensional numeric values with shape `(n_observations,)`, aligned
        row-wise to `dependent_var`.

    Returns
    -------
    tuple[float or str, float or str, float or str, float or str]
        `(slope, intercept, r_value, p_value)` for `dependent_var ~
        independent_var`. Returns string `"None"` for all outputs when fewer
        than two paired observations are available. Input filtering and numeric
        conversion remain the caller's responsibility.
    """
    stats_df = pd.DataFrame({
        'y': dependent_var,
        'a': independent_var,
    })
    if stats_df.shape[0] < 2:
        return "None", "None", "None", "None"

    y, X = model_matrix("y ~ a", stats_df)
    model = sm.OLS(y, X)
    results = model.fit()
    r_value, p_value = results.rsquared, results.pvalues['a']
    intercept, slope = results.params['Intercept'], results.params['a']
    return slope, intercept, r_value, p_value


def _get_valid_trials_to_correct_mask(
    trials_to_correct: pd.Series,
    prev_n_correct: pd.Series,
) -> pd.Series:
    """Identify block rows usable for trials-to-correct regression summaries.

    Parameters
    ----------
    trials_to_correct : pd.Series
        Blockwise trials-to-correct values with shape `(n_blocks,)`. Valid rows
        contain integer-like values; invalid rows use real missing values or
        string missing-value sentinels such as `"None"`.
    prev_n_correct : pd.Series
        Number of correct trials in the previous block, shape `(n_blocks,)`.
        The first block uses a real missing value or string missing-value
        sentinel when no previous block exists.

    Returns
    -------
    pd.Series
        Boolean mask with shape `(n_blocks,)`, aligned to the input index.
    """
    return is_present_value(trials_to_correct) & is_present_value(prev_n_correct)


def _add_regression_stats(
    session_performance: pd.DataFrame,
    column_prefix: str,
    dependent_var: pd.Series,
    independent_var: pd.Series,
) -> pd.DataFrame:
    """Add one simple-regression result to a one-row session summary.

    Parameters
    ----------
    session_performance : pd.DataFrame
        One-row session summary dataframe.
    column_prefix : str
        Prefix naming the independent variable in generated columns.
    dependent_var : pd.Series
        Numeric dependent values with shape `(n_observations,)`.
    independent_var : pd.Series
        Numeric independent values with shape `(n_observations,)`, aligned
        row-wise to `dependent_var`.

    Returns
    -------
    pd.DataFrame
        Copy of `session_performance` with `{column_prefix}_slope`,
        `{column_prefix}_intercept`, `{column_prefix}_r_value`, and
        `{column_prefix}_p_value` columns.
    """
    slope, intercept, r_value, p_value = session_stats(
        dependent_var=dependent_var,
        independent_var=independent_var,
    )
    session_performance[f"{column_prefix}_slope"] = slope
    session_performance[f"{column_prefix}_intercept"] = intercept
    session_performance[f"{column_prefix}_r_value"] = r_value
    session_performance[f"{column_prefix}_p_value"] = p_value
    return session_performance


def add_regression_stats_to_session_performance(
    session_performance: pd.DataFrame,
    trials_to_correct: pd.Series,
    prev_n_correct: pd.Series,
    prev_consecutive_rewards: pd.Series,
    n_blocks: int,
) -> pd.DataFrame:
    """Add block-learning regression summaries to a session summary.

    Parameters
    ----------
    session_performance : pd.DataFrame
        One-row session summary dataframe.
    trials_to_correct : pd.Series
        Blockwise trials-to-correct values with shape `(n_blocks,)`.
    prev_n_correct : pd.Series
        Previous-block correct-trial counts with shape `(n_blocks,)`.
    prev_consecutive_rewards : pd.Series
        Previous-block consecutive-reward counts with shape `(n_blocks,)`.
    n_blocks : int
        Number of analyzed blocks in the session.

    Returns
    -------
    pd.DataFrame
        Copy of `session_performance` with explicit regression-stat columns
        for `prev_consecutive_rewards` and `prev_n_correct`, plus `n_blocks`.
    """
    session_performance = session_performance.copy()
    valid_rows = _get_valid_trials_to_correct_mask(trials_to_correct, prev_n_correct)
    dependent_var = trials_to_correct[valid_rows].astype(int)

    session_performance = _add_regression_stats(
        session_performance=session_performance,
        column_prefix="prev_consecutive_rewards",
        dependent_var=dependent_var,
        independent_var=prev_consecutive_rewards[valid_rows].astype(int),
    )
    session_performance = _add_regression_stats(
        session_performance=session_performance,
        column_prefix="prev_n_correct",
        dependent_var=dependent_var,
        independent_var=prev_n_correct[valid_rows].astype(int),
    )
    session_performance["n_blocks"] = n_blocks
    return session_performance


def add_block_bias_columns(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Add legacy block-bias summaries to a block-performance table.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance dataframe with shape `(n_blocks, n_columns)`.
        Required columns are `trials_to_correct` and `prev_n_correct`.

    Returns
    -------
    pd.DataFrame
        Copy of `block_performance` with `bias_rl`, `bias_inf`,
        `bias_rl_flag`, `bias_inf_flag`, and `bias_full_flag` columns.
        Invalid block rows receive the string sentinel `"None"`.
    """
    block_performance = block_performance.copy()
    valid_rows = _get_valid_trials_to_correct_mask(
        block_performance["trials_to_correct"],
        block_performance["prev_n_correct"],
    )

    base_array = np.zeros(block_performance.shape[0], dtype=object)
    base_array[:] = 0
    base_array[~valid_rows] = "None"

    trials_to_correct = block_performance.loc[valid_rows, "trials_to_correct"].astype(int)
    prev_n_correct = block_performance.loc[valid_rows, "prev_n_correct"].astype(int)

    bias_rl_values = (
        (trials_to_correct - prev_n_correct)
        / (trials_to_correct + prev_n_correct + eps)
    )
    bias_rl = base_array.copy()
    bias_rl[valid_rows] = bias_rl_values.to_numpy()

    bias_inf_values = (trials_to_correct - 5) / (trials_to_correct + 5)
    bias_inf = base_array.copy()
    bias_inf[valid_rows] = bias_inf_values.to_numpy()

    bias_thresh = .2
    bias_rl_flag_values = bias_rl_values > bias_thresh
    bias_inf_flag_values = bias_inf_values > bias_thresh
    bias_full_flag_values = bias_rl_flag_values & bias_inf_flag_values

    bias_rl_flag = base_array.copy()
    bias_rl_flag[valid_rows] = bias_rl_flag_values.to_numpy()
    bias_inf_flag = base_array.copy()
    bias_inf_flag[valid_rows] = bias_inf_flag_values.to_numpy()
    bias_full_flag = base_array.copy()
    bias_full_flag[valid_rows] = bias_full_flag_values.to_numpy()

    block_performance["bias_rl"] = bias_rl
    block_performance["bias_inf"] = bias_inf
    block_performance["bias_rl_flag"] = bias_rl_flag
    block_performance["bias_inf_flag"] = bias_inf_flag
    block_performance["bias_full_flag"] = bias_full_flag
    return block_performance


def assert_saved_csv(path: Path) -> None:
    """Verify that a CSV save produced a non-empty file.

    Parameters
    ----------
    path : Path
        Expected CSV output path.

    Returns
    -------
    None
        Returns None when the file exists and has nonzero byte size.

    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    OSError
        If `path` exists but is empty.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected saved CSV was not found: {path}")
    if path.stat().st_size == 0:
        raise OSError(f"Expected saved CSV is empty: {path}")


def save_analysis(session_performance: pd.DataFrame, block_performance: pd.DataFrame, augmented_trial_df: pd.DataFrame,
                  sess_id: str, session_save_path: Path, multisession_save_path: Path=None) -> pd.DataFrame:
    """Save within-session outputs and optionally update multisession summaries.

    Parameters
    ----------
    session_performance : pd.DataFrame
        Single-row session summary dataframe.
    block_performance : pd.DataFrame
        Blockwise summary dataframe.
    augmented_trial_df : pd.DataFrame
        Trialwise augmented dataframe.
    sess_id : str
        Full session identifier.
    session_save_path : Path
        Directory where within-session CSV outputs are written.
    multisession_save_path : Path or None, default=None
        Multisession summary directory. If None, only within-session outputs
        are written and `session_performance` is returned unchanged.

    Returns
    -------
    pd.DataFrame
        Updated multisession dataframe if `multisession_save_path` is provided,
        otherwise the input `session_performance`.
    """
    assert session_save_path.exists(), "within-session data save path does not exist"

    match = re.search(r"(.+?)_(\d{4}-\d{2}-\d{2})_(\d{6})", sess_id)
    if match is None:
        raise ValueError(f"sess_id does not match expected pattern: {sess_id}")
    mouse, date, _time = match.groups()
    block_performance_path = session_save_path / (sess_id + '_block_performance.csv')
    augmented_trial_path = session_save_path / (sess_id + '_augmented_trials.csv')

    block_performance.to_csv(block_performance_path, index=False, na_rep='None')
    augmented_trial_df.to_csv(augmented_trial_path, index=False, na_rep='None')
    assert_saved_csv(block_performance_path)
    assert_saved_csv(augmented_trial_path)

    if multisession_save_path is None:
        return session_performance

    assert multisession_save_path.exists(), "between-session data save path does not exist"
    multisession_summary_path = multisession_save_path / (mouse + '_overall_performance.csv')
    if multisession_summary_path.exists():
        multisession_df = pd.read_csv(multisession_summary_path, na_filter=False)
        current_session_date = multisession_df['date'] == date
        multisession_df = multisession_df[~current_session_date]
        multisession_df = pd.concat([multisession_df, session_performance], axis=0)
    else:
        multisession_df = session_performance.copy()

    multisession_df.sort_values(by='date', inplace=True)
    multisession_df.reset_index(drop=True, inplace=True)
    multisession_df.to_csv(multisession_summary_path, index=False, na_rep='None')
    assert_saved_csv(multisession_summary_path)
    return multisession_df


def load_analysis(sess_id_full: str, session_data_folder: Path, multisession_data_folder: Path) -> tuple:
    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    mouse, date, timestamp = match.groups()
    # print(f"Mouse id: {mouse}")  # abc
    # print(f"Date: {date}")  # YYYY-MM-DD
    # print(f"Time: {timestamp}")  # HHMMSS

    block_performance = pd.read_csv(session_data_folder / (sess_id_full + '_block_performance.csv'), sep=',', na_filter=False)
    augmented_trial_df = pd.read_csv(session_data_folder / (sess_id_full + '_augmented_trials.csv'), sep=',', na_filter=False)
    multisession_df = pd.read_csv(multisession_data_folder / (mouse + '_overall_performance.csv'), sep=',', na_filter=False)
    return multisession_df, block_performance, augmented_trial_df


def run_analysis(trial_df: pd.DataFrame, session: Session):
    """Run single-session analysis with optional multisession persistence.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns match
        `analyze_session`.
    session : Session
        Session metadata. Required attributes are `mouse`, `date`,
        `sess_id_full`, `processed_data_path`, and `multi_session_save_path`.
        If `multi_session_save_path` is None, only within-session CSV outputs
        are saved.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        `(augmented_trial_df, block_performance, multisession_df)`. When
        multisession saving is disabled, the third dataframe is the one-row
        session summary instead of a loaded cross-session table.
    """
    session_performance, block_performance, augmented_trial_df = analyze_session(trial_df, mouse=session.mouse, date=session.date)
    session_performance = add_regression_stats_to_session_performance(
        session_performance=session_performance,
        trials_to_correct=block_performance["trials_to_correct"],
        prev_n_correct=block_performance["prev_n_correct"],
        prev_consecutive_rewards=block_performance["prev_consecutive_rewards"],
        n_blocks=block_performance.shape[0],
    )
    block_performance = add_block_bias_columns(block_performance)

    print(
        "prev_consecutive_rewards slope: {}, intercept: {}, r_value: {}, p_value: {}".format(
            session_performance.loc[0, "prev_consecutive_rewards_slope"],
            session_performance.loc[0, "prev_consecutive_rewards_intercept"],
            session_performance.loc[0, "prev_consecutive_rewards_r_value"],
            session_performance.loc[0, "prev_consecutive_rewards_p_value"],
        )
    )
    print(
        "prev_n_correct slope: {}, intercept: {}, r_value: {}, p_value: {}".format(
            session_performance.loc[0, "prev_n_correct_slope"],
            session_performance.loc[0, "prev_n_correct_intercept"],
            session_performance.loc[0, "prev_n_correct_r_value"],
            session_performance.loc[0, "prev_n_correct_p_value"],
        )
    )

    multisession_df = save_analysis(
        session_performance,
        block_performance,
        augmented_trial_df,
        sess_id=session.sess_id_full,
        session_save_path=session.processed_data_path,
        multisession_save_path=getattr(session, "multi_session_save_path", None),
    )
    return augmented_trial_df, block_performance, multisession_df


def main():
    """Analyze the data from a single mouse session"""

    data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = data_home / 'rpi' / sess_id_full
    processed_data_path = data_home / 'processed'
    figure_path = data_home / 'figures'
    multi_session_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')

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

    sess_id_abbreviated = mouse + '_' + date
    trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_trials.csv'), sep=',', na_filter=False)
    sess = type("SessionConfig", (), {})()
    sess.mouse = mouse
    sess.date = date
    sess.sess_id_full = sess_id_full
    sess.processed_data_path = processed_data_path
    sess.multi_session_save_path = multi_session_save_path
    augmented_trial_df, block_performance, multisession_df = run_analysis(trial_df, session=sess)


if __name__ == '__main__':
    main()
