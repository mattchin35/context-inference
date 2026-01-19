from pathlib import Path
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import src.behavior_analysis.decision_variable_counters as counters
# import src.behavior_analysis.session_analysis as session_analysis
from formulaic import model_matrix
import statsmodels.api as sm
# from icecream import ic
import statsmodels.formula.api as smf
import scipy as sp
from typing import Protocol


"""
Analyze behavior within a single session. Want trials to switch L/R/overall, % correct L/R/overall. Do for sessions
overall AND within each block.
"""

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


def get_block_types(trial_df: pd.DataFrame) -> np.array:
    # left_cued_ix = (trial_df['state'] == 'left_patch') & (~trial_df['block_stimulus'].isnull())  # left cued
    # left_uncued_ix = (trial_df['state'] == 'left_patch') & (trial_df['block_stimulus'].isnull())  # left uncued
    # right_cued_ix = (trial_df['state'] == 'right_patch') & (~trial_df['block_stimulus'].isnull())  # right cued
    # right_uncued_ix = (trial_df['state'] == 'right_patch') & (trial_df['block_stimulus'].isnull())  # right uncued

    uncued_block = trial_df['block_stimulus'] == 'None' # or trial_df['block_stimulus'].isnull()
    cued_block = ~uncued_block # or trial_df['block_stimulus'].isnull()
    left_cued_ix = (trial_df['state'] == 'left_patch') & cued_block  # left cued
    left_uncued_ix = (trial_df['state'] == 'left_patch') & uncued_block  # left uncued
    right_cued_ix = (trial_df['state'] == 'right_patch') & cued_block  # right cued
    right_uncued_ix = (trial_df['state'] == 'right_patch') & uncued_block  # right uncued

    block_types = np.zeros(trial_df.shape[0], dtype=object)
    block_types[left_cued_ix] = 'left_cued'
    block_types[right_cued_ix] = 'right_cued'
    block_types[left_uncued_ix] = 'left_uncued'
    block_types[right_uncued_ix] = 'right_uncued'
    return block_types


def percent_correct(augmented_trial_df: pd.DataFrame) -> dict:
    """Calculate the percentage of correct choices made by the agent. Calculate for
    left uncued, right uncued, left cued, and right cued."""
    # 1. collect left and right choices 2. categorize choices as cued or uncued 3. calculate percentages
    assert 'block_type' in augmented_trial_df.keys(), "'block_type' key was not found in dataframe."
    assert 'correct' in augmented_trial_df.keys(), "'correct' key was not found in dataframe."

    left_cued_ix = augmented_trial_df['block_type'] == 'left_cued'
    left_uncued_ix = augmented_trial_df['block_type'] == 'left_uncued'
    right_cued_ix = augmented_trial_df['block_type'] == 'right_cued'
    right_uncued_ix = augmented_trial_df['block_type'] == 'right_uncued'

    if left_cued_ix.sum():
        left_cued_correct = np.sum(augmented_trial_df.loc[left_cued_ix, 'correct']) / np.sum(left_cued_ix)
    else:
        # left_cued_correct = np.nan
        left_cued_correct = 'None'

    if left_uncued_ix.sum():
        left_uncued_correct = np.sum(augmented_trial_df.loc[left_uncued_ix, 'correct']) / np.sum(left_uncued_ix)
    else:
        # left_uncued_correct = np.nan
        left_uncued_correct = 'None'

    if right_cued_ix.sum():
        right_cued_correct = np.sum(augmented_trial_df.loc[right_cued_ix, 'correct']) / np.sum(right_cued_ix)
    else:
        # right_cued_correct = np.nan
        right_cued_correct = 'None'

    if right_uncued_ix.sum():
        right_uncued_correct = np.sum(augmented_trial_df.loc[right_uncued_ix, 'correct']) / np.sum(right_uncued_ix)
    else:
        # right_uncued_correct = np.nan
        right_uncued_correct = 'None'

    overall = np.sum(augmented_trial_df['correct']) / augmented_trial_df.shape[0]
    return dict(left_cued_correct=left_cued_correct, left_uncued_correct=left_uncued_correct,
                right_cued_correct=right_cued_correct, right_uncued_correct=right_uncued_correct,
                overall_correct=overall)


def summarize_trials_to_correct(block_performance: pd.DataFrame) -> dict:
    # 1. collect left and right choices 2. categorize choices as cued or uncued 3. calculate percentages
    left_cued_ix = block_performance['block_type'] == 'left_cued'
    left_uncued_ix = block_performance['block_type'] == 'left_uncued'
    right_cued_ix = block_performance['block_type'] == 'right_cued'
    right_uncued_ix = block_performance['block_type'] == 'right_uncued'

    # condition to handle - last block may have no correct choices (lack of engagement, didn't get it right before session end)
    # handle by removing this block entirely from overall summary as incomplete data
    if block_performance['trials_to_correct'].iloc[-1] == 'None':
        ix_valid = block_performance.shape[0] - 1
        block_performance = block_performance.iloc[:ix_valid]
        left_cued_ix = left_cued_ix.iloc[:ix_valid]
        left_uncued_ix = left_uncued_ix.iloc[:ix_valid]
        right_cued_ix = right_cued_ix.iloc[:ix_valid]
        right_uncued_ix = right_uncued_ix.iloc[:ix_valid]

    try:
        if np.sum(left_cued_ix) == 0:
            # left_cued_trials = np.nan
            left_cued_trials = 'None'
        else:
            left_cued_trials = np.mean(block_performance.loc[left_cued_ix, 'trials_to_correct'])

        if np.sum(left_uncued_ix) == 0:
            # left_uncued_trials = np.nan
            left_uncued_trials = 'None'
        else:
            left_uncued_trials = np.mean(block_performance.loc[left_uncued_ix, 'trials_to_correct'])

        if np.sum(right_cued_ix) == 0:
            # right_cued_trials = np.nan
            right_cued_trials = 'None'
        else:
            right_cued_trials = np.mean(block_performance.loc[right_cued_ix, 'trials_to_correct'])

        if np.sum(right_uncued_ix) == 0:
            # right_uncued_trials = np.nan
            right_uncued_trials = 'None'
        else:
            right_uncued_trials = np.mean(block_performance.loc[right_uncued_ix, 'trials_to_correct'])

    except Exception as e:
        print(e)

    overall = np.mean(block_performance['trials_to_correct'])
    return dict(left_cued_trials_to_correct=left_cued_trials, left_uncued_trials_to_correct=left_uncued_trials,
                right_cued_trials_to_correct=right_cued_trials, right_uncued_trials_to_correct=right_uncued_trials,
                overall_trials_to_correct=overall)


def get_block_switches(trial_df: pd.DataFrame) -> tuple[int, int]:
    # handle any give_reward trials
    actions = np.copy(trial_df['action'].values)
    rewards = np.copy(trial_df['reward'].values)
    give_reward = np.copy(trial_df['give_reward'].values)
    for i, (a, g) in enumerate(zip(actions, give_reward)):
        # if a in ['None', -1]:
        if g in [1, '1']:
            try:
                actions[i] = actions[i - 1]
            except IndexError:
                actions[i] = actions[i + 1]

    actions = actions.astype(int)
    n_switches = np.sum(np.abs(np.diff(actions)))
    normalized_switches = n_switches / (trial_df.shape[0] - 1)  # must subtract 1 for n-1 opportunites to switch
    return n_switches, normalized_switches


def analyze_session(trial_df: pd.DataFrame, mouse: str, date: str) -> tuple:
    """
    Get trial decision variables, use them to create/modify an augmented trial_df.
    Get block trials to switch and consecutive rewards in previous block, use to create/modify a block df.
    """
    sess_id = mouse + '_' + date

    # collect decision variables - need to add these to a df of augmented trials
    decision_vars = count_decision_variables(trial_df)
    augmented_trial_df = pd.concat([trial_df, decision_vars], axis=1)
    augmented_trial_df['time_to_choice'] = augmented_trial_df['choice_time'] - augmented_trial_df['start_time']

    # get block qualities: trials-to-switch, consecutive rewards, other augmentations. Add new variables here too
    blocks = np.unique(trial_df['cur_block'])
    block_performance = []
    block_types = get_block_types(trial_df)
    augmented_trial_df['block_type'] = block_types
    for ix, b in enumerate(blocks):
        cur_block_ix = augmented_trial_df['cur_block'] == b
        cur_block_df = augmented_trial_df[cur_block_ix]

        if ix == 0:
            # prev_n_correct = 'None'
            # prev_n_rewarded = 'None'
            # prev_consecutive_rewards = 'None'
            # prev_consecutive_rewards_memory = 'None'
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

        # performance = dict(block_ix=b, block_type=cur_block_df['block_type'].values[0], trials_to_correct=trials_to_correct,
        #                    consecutive_rewards=cur_block_df['consecutive_rewards'].values[0],
        #                    percent_correct=performance['overall_correct'],
        #                    session_ID=sess_id)
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
                           mean_choice_time=np.mean(cur_block_df['choice_time']-cur_block_df['start_time']),
                           median_choice_time=np.median(cur_block_df['choice_time']-cur_block_df['start_time']),
                           std_choice_time=np.std(cur_block_df['choice_time']-cur_block_df['start_time']),
                           session_ID=sess_id)

        block_performance.append(performance)

    block_performance = pd.DataFrame(block_performance)
    session_performance = percent_correct(augmented_trial_df)
    session_performance = session_performance | summarize_trials_to_correct(block_performance)
    session_performance['date'] = date
    session_performance = pd.DataFrame(session_performance, index=[0])

    prev_action = np.zeros(trial_df.shape[0], dtype='object')
    prev_reward = np.zeros(trial_df.shape[0], dtype='object')
    prev_action[0] = 'None'
    prev_reward[0] = 'None'
    prev_action[1:] = trial_df['action'].to_numpy()[:-1]
    prev_reward[1:] = trial_df['reward'].to_numpy()[:-1]
    augmented_trial_df['prev_action'] = prev_action
    augmented_trial_df['prev_reward'] = prev_reward

    return session_performance, block_performance, augmented_trial_df


def count_decision_variables(trial_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collect the decision variables for each trial; DVs reflect the trial history that influence choices,
    and so do not include rewards and actions from the current trial.

    Trials with experimenter-given rewards will not accumulate DV value.
    """
    consecutive_rewards = 0
    consecutive_failures = 0
    consecutive_rewards_memory = 0
    consecutive_failures_memory = 0
    negative_value = 0
    previous_trial_rewarded = False

    left_value = 0
    right_value = 0
    relative_value = 0
    left_omissions = 0
    right_omissions = 0

    n_trials = trial_df.shape[0]
    decision_variable_dict = defaultdict(list)

    # need 2 sets of counts
    # 1. the integrate-and-reset params from Cazettes 2023. Why is value negative again?
    # 2. the last-seen/"memory" version used, which resets when failures/rewards start anew but don't reset on switches
    for i in range(n_trials):
        decision_variable_dict['negative_value'].append(negative_value)
        decision_variable_dict['consecutive_rewards_memory'].append(consecutive_rewards_memory)
        decision_variable_dict['consecutive_failures_memory'].append(consecutive_failures_memory)
        decision_variable_dict['consecutive_rewards'].append(consecutive_rewards)
        decision_variable_dict['consecutive_failures'].append(consecutive_failures)
        
        # counterfactual decision vars
        decision_variable_dict['left_value'].append(left_value)
        decision_variable_dict['right_value'].append(right_value)
        decision_variable_dict['relative_value'].append(left_value-right_value)
        decision_variable_dict['left_nonneg_value'].append(np.amax([left_value, 0]))
        decision_variable_dict['right_nonneg_value'].append(np.amax([right_value,0]))
        decision_variable_dict['relative_nonneg_value'].append(np.amax([left_value, 0]) -
                                                               np.amax([right_value,0]))  # this can be negative, but the values it comes from cannot
        decision_variable_dict['left_omissions'].append(left_omissions)
        decision_variable_dict['right_omissions'].append(right_omissions)
        decision_variable_dict['relative_omissions'].append(left_omissions-right_omissions)

        # Don't update DVs for trials with experimenter-given rewards. These trials shouldn't be included in action
        # prediction models either
        if trial_df.loc[i, 'action'] == 'None':
            continue

        reward = trial_df.loc[i, 'reward']
        action = trial_df.loc[i, 'action']
        negative_value = counters.negative_value_counter(negative_value, reward)
        consecutive_rewards = counters.consecutive_reward_counter(consecutive_rewards, reward)
        consecutive_failures = counters.consecutive_fail_counter(consecutive_failures, reward)
        consecutive_rewards_memory = counters.consecutive_reward_renewal_counter(consecutive_rewards_memory, reward, previous_trial_rewarded)
        consecutive_failures_memory = counters.consecutive_fail_renewal_counter(consecutive_failures_memory, reward, previous_trial_rewarded)
        previous_trial_rewarded = reward > 0

        left_value, right_value = counters.counterfactual_value_counter(left_value, right_value, action, reward)
        left_omissions, right_omissions = counters.counterfactual_omissions_counter(left_omissions, right_omissions, action, reward)

    return pd.DataFrame(decision_variable_dict)


def summarize_block_switches(block_performance: pd.DataFrame, min_counts=0) -> tuple:
    grouped_switches = block_performance.groupby(['consecutive_rewards'])['trials_to_correct']
    counts = grouped_switches.count()
    # ic(counts)

    ix = counts > min_counts
    mean = grouped_switches.mean()[ix]
    std = grouped_switches.std()[ix]
    sem = grouped_switches.sem()[ix]
    rewards = counts.index[ix]
    return rewards, mean, std, sem


def session_stats(dependent_var, independent_var) -> tuple[float, float, float, float]:
    stats_df = pd.DataFrame({
        'y': dependent_var,
        'a': independent_var,
    })

    y, X = model_matrix("y ~ a", stats_df)
    model = sm.OLS(y, X)
    results = model.fit()
    r_value, p_value = results.rsquared, results.pvalues['a']
    intercept, slope = results.params['Intercept'], results.params['a']
    return slope, intercept, r_value, p_value


def save_analysis(session_performance: pd.DataFrame, block_performance: pd.DataFrame, augmented_trial_df: pd.DataFrame,
                  sess_id: str, session_save_path: Path, overall_save_path: Path=None):
    assert session_save_path.exists(), "within-session data save path does not exist"
    assert overall_save_path.exists(), "between-session data save path does not exist"

    mouse, date, time = sess_id.split('_')
    block_performance.to_csv(session_save_path / (sess_id + '_block_performance.csv'), index=False)
    augmented_trial_df.to_csv(session_save_path / (sess_id + '_augmented_trials.csv'), index=False)
    overall_fname = overall_save_path / (mouse + '_overall_performance.csv')
    if overall_fname.exists():
        overall_df = pd.read_csv(overall_fname, na_filter=False)
        ix = overall_df['date'] == date
        if ix.any():
            overall_df[ix] = session_performance  # doing it this way allows updates to existing data
            # overall_df.sort_values('date')  # double-check this when analyzing multiple sessions
        else:
            overall_df = pd.concat([overall_df, session_performance], axis=0)
        overall_df.sort_values(by='date', inplace=True)
        overall_df.to_csv(overall_fname, index=False)
    else:
        session_performance.to_csv(overall_fname, index=False)


def load_analysis(sess_id_full: str, session_data_folder: Path, multisession_data_folder: Path) -> tuple:
    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)
    mouse, date, timestamp = match.groups()
    print(f"Mouse id: {mouse}")  # abc
    print(f"Date: {date}")  # YYYY-MM-DD
    print(f"Time: {timestamp}")  # HHMMSS

    block_performance = pd.read_csv(session_data_folder / (sess_id_full + '_block_performance.csv'), sep=',', na_filter=False)
    augmented_trial_df = pd.read_csv(session_data_folder / (sess_id_full + '_augmented_trials.csv'), sep=',', na_filter=False)
    multisession_df = pd.read_csv(multisession_data_folder / (mouse + '_overall_performance.csv'), sep=',', na_filter=False)
    return multisession_df, block_performance, augmented_trial_df


# def run_analysis(trial_df: pd.DataFrame, sess_id_full: str, processed_data_path: Path,):
def run_analysis(trial_df: pd.DataFrame, session: Session):
    session_performance, block_performance, augmented_trial_df = analyze_session(trial_df, mouse=session.mouse, date=session.date)
    ix_valid = (block_performance['trials_to_correct'] != 'None') & (block_performance['prev_n_correct'] != 'None')
    slope, intercept, r_value, p_value = session_stats(
        dependent_var=block_performance['trials_to_correct'][ix_valid].astype(int),
        independent_var=block_performance['prev_consecutive_rewards'][ix_valid].astype(int))

    """
    TTS vs prev rewards: condition by confusion (nswitch > 3 or so) 
    TTS vs rewards by bias (TTS >> block rewards. Could quantify that - TTS / rewards (block, consec, or consec_memory) or TTS - rewards
    Condition by mean, median or std choice time
    """
    base_array = np.zeros_like(block_performance['trials_to_correct'])
    base_array[~ix_valid] = 'None'

    # using n_correct for biases as the maximum possible count for value comparison
    _bias_rl = ((block_performance['trials_to_correct'][ix_valid] - block_performance['prev_n_correct'][ix_valid]) /
                (block_performance['trials_to_correct'][ix_valid] + block_performance['prev_n_correct'][ix_valid]))
    bias_rl = base_array.copy()
    bias_rl[ix_valid] = _bias_rl.to_numpy()

    _bias_inf = (block_performance['trials_to_correct'][ix_valid] - 5) / (
                block_performance['trials_to_correct'][ix_valid] + 5)
    bias_inf = base_array.copy()
    bias_inf[ix_valid] = _bias_inf.to_numpy()

    bias_thresh = .2
    _bias_rl_flag = _bias_rl > bias_thresh
    _bias_inf_flag = _bias_inf > bias_thresh
    _bias_full_flag = _bias_rl_flag & _bias_inf_flag
    bias_rl_flag = base_array.copy()
    bias_rl_flag[ix_valid] = _bias_rl_flag.to_numpy()
    bias_inf_flag = base_array.copy()
    bias_inf_flag[ix_valid] = _bias_inf_flag.to_numpy()
    bias_full_flag = base_array.copy()
    bias_full_flag[ix_valid] = _bias_full_flag.to_numpy()

    block_performance['bias_rl'] = bias_rl
    block_performance['bias_inf'] = bias_inf
    block_performance['bias_rl_flag'] = bias_rl_flag
    block_performance['bias_inf_flag'] = bias_inf_flag
    block_performance['bias_full_flag'] = bias_full_flag

    # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
    session_performance['slope'] = slope
    session_performance['intercept'] = intercept
    session_performance['r_value'] = r_value
    session_performance['p_value'] = p_value
    session_performance['n_switches'] = block_performance.shape[0]
    # ic(slope, intercept, r_value, p_value)
    print('slope: {}, intercept: {}, r_value: {}, p_value: {}'.format(slope, intercept, r_value, p_value))
    # block_performance.to_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), index=False)
    # augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)

    save_analysis(session_performance, block_performance, augmented_trial_df,
                  sess_id=session.sess_id_full, session_save_path=session.processed_data_path,
                  overall_save_path=session.multi_session_save_path)

    multisession_performance = pd.read_csv(session.processed_data_path / (session.sess_id_full + '_.csv'), sep=',', na_filter=False)
    multisession_df, block_performance, augmented_trial_df = load_analysis(session.sess_id_full, session.processed_data_path,
                                                                           session.multi_session_save_path)


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
    session_performance, block_performance, augmented_trial_df = analyze_session(trial_df, mouse=mouse, date=date)
    ix_valid = (block_performance['trials_to_correct'] != 'None') & (block_performance['prev_n_correct'] != 'None')
    slope, intercept, r_value, p_value = session_stats(dependent_var=block_performance['trials_to_correct'][ix_valid].astype(int),
                                                    independent_var=block_performance['prev_consecutive_rewards'][ix_valid].astype(int))

    """
    TTS vs prev rewards: condition by confusion (nswitch > 3 or so) 
    TTS vs rewards by bias (TTS >> block rewards. Could quantify that - TTS / rewards (block, consec, or consec_memory) or TTS - rewards
    Condition by mean, median or std choice time
    """
    base_array = np.zeros_like(block_performance['trials_to_correct'])
    base_array[~ix_valid] = 'None'

    # using n_correct for biases as the maximum possible count for value comparison
    _bias_rl = ((block_performance['trials_to_correct'][ix_valid] - block_performance['prev_n_correct'][ix_valid]) /
               (block_performance['trials_to_correct'][ix_valid] + block_performance['prev_n_correct'][ix_valid]))
    bias_rl = base_array.copy()
    bias_rl[ix_valid] = _bias_rl.to_numpy()

    _bias_inf = (block_performance['trials_to_correct'][ix_valid] - 5) / (block_performance['trials_to_correct'][ix_valid] + 5)
    bias_inf = base_array.copy()
    bias_inf[ix_valid] = _bias_inf.to_numpy()

    bias_thresh = .2
    _bias_rl_flag = _bias_rl > bias_thresh
    _bias_inf_flag = _bias_inf > bias_thresh
    _bias_full_flag = _bias_rl_flag & _bias_inf_flag
    bias_rl_flag = base_array.copy()
    bias_rl_flag[ix_valid] = _bias_rl_flag.to_numpy()
    bias_inf_flag = base_array.copy()
    bias_inf_flag[ix_valid] = _bias_inf_flag.to_numpy()
    bias_full_flag = base_array.copy()
    bias_full_flag[ix_valid] = _bias_full_flag.to_numpy()

    block_performance['bias_rl'] = bias_rl
    block_performance['bias_inf'] = bias_inf
    block_performance['bias_rl_flag'] = bias_rl_flag
    block_performance['bias_inf_flag'] = bias_inf_flag
    block_performance['bias_full_flag'] = bias_full_flag

    # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
    session_performance['slope'] = slope
    session_performance['intercept'] = intercept
    session_performance['r_value'] = r_value
    session_performance['p_value'] = p_value
    session_performance['n_switches'] = block_performance.shape[0]
    # ic(slope, intercept, r_value, p_value)
    print('slope: {}, intercept: {}, r_value: {}, p_value: {}'.format(slope, intercept, r_value, p_value))
    # block_performance.to_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), index=False)
    # augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)

    save_analysis(session_performance, block_performance, augmented_trial_df,
                  sess_id=sess_id_full, session_save_path=processed_data_path, overall_save_path=multi_session_save_path)

    multisession_performance = pd.read_csv(processed_data_path / (sess_id_full + '_.csv'), sep=',', na_filter=False)
    multisession_df, block_performance, augmented_trial_df = load_analysis(sess_id_full, processed_data_path, multi_session_save_path)

    # TODO - allow combining new session analysis with old by replacing dates and sorting the matrix by date.


if __name__ == '__main__':
    main()
