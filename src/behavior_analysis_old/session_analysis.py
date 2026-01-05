import session_overview
import fileIO
from pathlib import Path
import pandas as pd
import numpy as np
import context_switch_analysis
from icecream import ic
from collections import defaultdict
import decision_variable_counters as counters
from formulaic import model_matrix
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf

"""Analyze behavior within a single session. Want trials to switch L/R/overall, % correct L/R/overall. Do for sessions
overall AND within each block."""


def percent_correct(event_df: pd.DataFrame) -> dict:
    """Calculate the percentage of correct choices made by the agent. Calculate for
    left uncued, right uncued, left cued, and right cued."""
    # 1. collect left and right choices 2. categorize choices as cued or uncued 3. calculate percentages
    left_cued_ix = (event_df['state'] == 'left_patch') & (~event_df['block_stimulus'].isnull())  # left cued
    left_uncued_ix = (event_df['state'] == 'left_patch') & (event_df['block_stimulus'].isnull())  # left uncued
    right_cued_ix = (event_df['state'] == 'right_patch') & (~event_df['block_stimulus'].isnull())  # right cued
    right_uncued_ix = (event_df['state'] == 'right_patch') & (event_df['block_stimulus'].isnull())  # right uncued

    if np.sum(left_cued_ix) == 0:
        left_cued_correct = np.nan
    else:
        left_cued_correct = np.sum(event_df.loc[left_cued_ix, 'correct']) / np.sum(left_cued_ix)

    if np.sum(left_uncued_ix) == 0:
        left_uncued_correct = np.nan
    else:
        left_uncued_correct = np.sum(event_df.loc[left_uncued_ix, 'correct']) / np.sum(left_uncued_ix)

    if np.sum(right_cued_ix) == 0:
        right_cued_correct = np.nan
    else:
        right_cued_correct = np.sum(event_df.loc[right_cued_ix, 'correct']) / np.sum(right_cued_ix)

    if np.sum(right_uncued_ix) == 0:
        right_uncued_correct = np.nan
    else:
        right_uncued_correct = np.sum(event_df.loc[right_uncued_ix, 'correct']) / np.sum(right_uncued_ix)

    overall = np.sum(event_df['correct']) / event_df.shape[0]
    return dict(left_cued_correct=left_cued_correct, left_uncued_correct=left_uncued_correct,
                right_cued_correct=right_cued_correct, right_uncued_correct=right_uncued_correct,
                overall_correct=overall)


def summarize_trials_to_correct(block_performance: pd.DataFrame) -> dict:
    # 1. collect left and right choices 2. categorize choices as cued or uncued 3. calculate percentages
    left_cued_ix = block_performance['block_type'] == 'left_cued'
    left_uncued_ix = block_performance['block_type'] == 'left_uncued'
    right_cued_ix = block_performance['block_type'] == 'right_cued'
    right_uncued_ix = block_performance['block_type'] == 'right_uncued'

    if np.sum(left_cued_ix) == 0:
        left_cued_trials = np.nan
    else:
        left_cued_trials = np.mean(block_performance.loc[left_cued_ix, 'trials_to_correct'])

    if np.sum(left_uncued_ix) == 0:
        left_uncued_trials = np.nan
    else:
        left_uncued_trials = np.mean(block_performance.loc[left_uncued_ix, 'trials_to_correct'])

    if np.sum(right_cued_ix) == 0:
        right_cued_trials = np.nan
    else:
        right_cued_trials = np.mean(block_performance.loc[right_cued_ix, 'trials_to_correct'])

    if np.sum(right_uncued_ix) == 0:
        right_uncued_trials = np.nan
    else:
        right_uncued_trials = np.mean(block_performance.loc[right_uncued_ix, 'trials_to_correct'])

    overall = np.mean(block_performance['trials_to_correct'])
    return dict(left_cued_trials_to_correct=left_cued_trials, left_uncued_trials_to_correct=left_uncued_trials,
                right_cued_trials_to_correct=right_cued_trials, right_uncued_trials_to_correct=right_uncued_trials,
                overall_trials_to_correct=overall)


def analyze_session(event_df: pd.DataFrame, mouse: str, date: str) -> tuple:
    sess_ID = mouse + '_' + date
    # event_df = session_overview.load_trial_df(sess_ID, data_path / mouse)
    decision_vars = count_decision_variables(event_df)
    choices_df = pd.concat([event_df, decision_vars], axis=1)

    blocks = np.unique(event_df['cur_block'])
    block_performance = []
    for b in blocks:
        block_ix = choices_df['cur_block'] == b
        block_df = choices_df.loc[block_ix, :]

        performance = percent_correct(block_df)
        for k, v in performance.items():
            if np.isnan(v):
                pass
            else:
                _block_type = k.split('_')
                block_type = _block_type[0] + '_' + _block_type[1]
                break

        correct_ix = np.nonzero(block_df['correct'])[0]
        if correct_ix.size > 0:
            trials_to_correct = correct_ix[0]
        else:
            trials_to_correct = np.nan

        performance = dict(block_ix=b, block_type=block_type, trials_to_correct=trials_to_correct,
                           consecutive_rewards=block_df['consecutive_rewards'].values[0],
                           percent_correct=performance['overall_correct'],
                           session_ID=sess_ID)
        block_performance.append(performance)

    block_performance = pd.DataFrame(block_performance)
    session_performance = percent_correct(event_df)
    session_performance = session_performance | summarize_trials_to_correct(block_performance)
    session_performance['date'] = date
    session_performance = pd.DataFrame(session_performance, index=[0])
    return session_performance, block_performance, choices_df


def count_decision_variables(event_df: pd.DataFrame) -> pd.DataFrame:
    """Collect the decision variables for each trial; DVs reflect the trial history that influence choices,
    and so do not include rewards and actions from the current trial."""
    consecutive_rewards = 0
    consecutive_failures = 0
    consecutive_rewards_memory = 0
    consecutive_failures_memory = 0
    negative_value = 0
    previous_trial_rewarded = False

    n_trials = event_df.shape[0]
    decision_variable_dict = defaultdict(list)

    # need 2 sets of counts
    # 1. the integrate-and-reset params from Cazettes
    # 2. the last-seen version used, which resets when failures/rewards start anew but don't reset on switches
    for i in range(n_trials):
        decision_variable_dict['negative_value'].append(negative_value)
        decision_variable_dict['consecutive_rewards_memory'].append(consecutive_rewards_memory)
        decision_variable_dict['consecutive_failures_memory'].append(consecutive_failures_memory)
        decision_variable_dict['consecutive_rewards'].append(consecutive_rewards)
        decision_variable_dict['consecutive_failures'].append(consecutive_failures)

        reward = event_df.loc[i, 'reward']
        negative_value = counters.negative_value_counter(negative_value, reward)
        consecutive_rewards = counters.consecutive_reward_counter(consecutive_rewards, reward)
        consecutive_failures = counters.consecutive_fail_counter(consecutive_failures, reward)
        consecutive_rewards_memory = counters.consecutive_reward_renewal_counter(consecutive_rewards_memory, reward, previous_trial_rewarded)
        consecutive_failures_memory = counters.consecutive_fail_renewal_counter(consecutive_failures_memory, reward, previous_trial_rewarded)
        previous_trial_rewarded = reward > 0

    return pd.DataFrame(decision_variable_dict)


def count_decision_variables_current(event_df: pd.DataFrame) -> pd.DataFrame:
    """Collect the decision variables for each trial; DVs reflect the trial history including
    rewards and actions from the current trial."""
    consecutive_rewards = 0
    consecutive_failures = 0
    consecutive_rewards_memory = 0
    consecutive_failures_memory = 0
    negative_value = 0
    previous_trial_rewarded = False

    n_trials = event_df.shape[0]
    decision_variable_dict = defaultdict(list)

    # need 2 sets of counts
    # 1. the integrate-and-reset params from Cazettes
    # 2. the last-seen version used, which resets when failures/rewards start anew but don't reset on switches
    for i in range(n_trials):
        reward = event_df.loc[i, 'reward']
        negative_value = counters.negative_value_counter(negative_value, reward)
        consecutive_rewards = counters.consecutive_reward_counter(consecutive_rewards, reward)
        consecutive_failures = counters.consecutive_fail_counter(consecutive_failures, reward)
        consecutive_rewards_memory = counters.consecutive_reward_renewal_counter(consecutive_rewards_memory, reward,
                                                                                 previous_trial_rewarded)
        consecutive_failures_memory = counters.consecutive_fail_renewal_counter(consecutive_failures_memory, reward,
                                                                                previous_trial_rewarded)

        decision_variable_dict['negative_value'].append(negative_value)
        decision_variable_dict['consecutive_rewards_memory'].append(consecutive_rewards_memory)
        decision_variable_dict['consecutive_failures_memory'].append(consecutive_failures_memory)
        decision_variable_dict['consecutive_rewards'].append(consecutive_rewards)
        decision_variable_dict['consecutive_failures'].append(consecutive_failures)
        previous_trial_rewarded = reward > 0

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


def save_analysis(session_performance: pd.DataFrame, block_performance: pd.DataFrame, choices_df: pd.DataFrame,
                  sess_id: str, session_save_path: Path,
                  # mouse: str, date: str, save_path: str=None):
                  overall_save_path: Path=None):
    """Save the block to block performance and the overall session performance. If a file already exists for overall
    session performance, append to or update it."""
    # if not save_path.exists():
    #     save_path.mkdir()

    # sess_ID = mouse + '_' + date
    mouse, date, time = sess_id.split('_')
    block_performance.to_csv(session_save_path / (sess_id + '_block_performance.csv'), index=False)
    choices_df.to_csv(session_save_path / (sess_id + '_choices.csv'), index=False)
    overall_fname = overall_save_path / (mouse + '_overall_performance.csv')
    if overall_fname.exists():
        overall_df = pd.read_csv(overall_fname)
        ix = overall_df['date'] == date
        if ix.any():
            overall_df[ix] = session_performance  # doing it this way allows updates to existing data
        else:
            overall_df = pd.concat([overall_df, session_performance], axis=0)
        overall_df.sort_values(by='date', inplace=True)
        overall_df.to_csv(overall_fname, index=False)
    else:
        session_performance.to_csv(overall_fname, index=False)


def load_analysis(mouse: str, date: str, data_path: Path) -> tuple:
    sess_ID = mouse + '_' + date
    block_performance = pd.read_csv(data_path / mouse / (sess_ID + '_block_performance.csv'))
    choices_df = pd.read_csv(data_path / mouse / (sess_ID + '_choices.csv'))
    overall_df = pd.read_csv(data_path / mouse / (mouse + '_overall_performance.csv'))
    return overall_df, block_performance, choices_df


def main():
    plot_path = Path('../../reports/figures')
    data_path = Path('../../data/processed')

    """Analyze the data from a single mouse session"""
    # date = ['2024-09-06']
    # mouse = ['HD005']

    date = ['2024-08-19', '2024-08-21', '2024-08-22', '2024-08-23', '2024-08-26', '2024-08-28', '2024-08-29', '2024-08-30', '2024-09-06',
            '2024-09-07', '2024-09-09', '2024-09-10']
    mouse = ['CT001'] * len(date)
    for m, d in zip(mouse, date):
        sess_id = m + '_' + d
        event_df = session_overview.load_trial_df(sess_id, data_path / m)
        session_performance, block_performance, choices_df = analyze_session(event_df, mouse=m, date=d)

        # rewards, mean, std, sem = summarize_block_switches(block_performance, min_counts=0)
        slope, intercept, r_value, p_value = session_stats(dependent_var=block_performance['trials_to_correct'],
                                                           independent_var=block_performance['consecutive_rewards'])
        session_performance['slope'] = slope
        session_performance['intercept'] = intercept
        session_performance['r_value'] = r_value
        session_performance['p_value'] = p_value
        session_performance['n_switches'] = block_performance.shape[0]

        save_analysis(session_performance, block_performance, choices_df, m, d, data_path)


if __name__ == '__main__':
    main()
