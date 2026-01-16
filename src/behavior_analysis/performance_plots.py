import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import session_analysis
from typing import Iterable
from icecream import ic
import re


block_types = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'dark period']
color_dict = {'right_cued': 'darkred', 'left_cued': 'darkblue', 'right_uncued': 'red', 'left_uncued': 'blue', 'dark period': 'black'}
state_dict = {0: 'right', 1: 'left'}


def plot_session_correct(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """For a single session, plot the blockwise percentage of correct choices made by the agent."""
    block_types = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'dark period']

    f, ax = plt.subplots()
    x = np.arange(len(block_performance))
    for b in block_types:
        ix = np.where(block_performance['block_type'] == b)[0]
        if len(ix) > 0:
            ax.plot(x[ix], block_performance['percent_correct'][ix], 'o', color=color_dict[b], label=b)

    plt.ylabel('Percent Correct')
    plt.xlabel('Block')
    plt.title('{} Block Performance'.format(sess_ID))
    plt.legend(fancybox=False)

    plt.tight_layout()
    save_path = plot_path / '{}_block_performance.png'.format(sess_ID)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def plot_multisession_correct(overall_df: pd.DataFrame, plot_path: Path, figure_id: str, use_dates: bool=True):
    block_types = ['right_cued_correct', 'left_cued_correct', 'right_uncued_correct', 'left_uncued_correct']
    color_dict = {'right_cued_correct': 'darkred', 'left_cued_correct': 'darkblue', 'right_uncued_correct': 'red', 'left_uncued_correct': 'blue'}
    overall_df = overall_df[~overall_df['date'].isna()]
    dates = overall_df['date'].unique()

    f, ax = plt.subplots()
    x = np.arange(overall_df.shape[0])
    for b in block_types:
        type_correct = overall_df[b].values
        if np.sum(np.isnan(type_correct)) == type_correct.size:  # will have to remove this later on...
            continue

        type_correct[np.isnan(type_correct)] = 0
        ax.plot(x, type_correct, color=color_dict[b], label=b)
        if use_dates:
            ax.set_xticks(x, dates)
            ax.tick_params(axis='x', labelrotation=60, labelsize=8)

    plt.ylabel('Percent Correct')
    plt.xlabel('Session')
    plt.title('{} Overall Performance'.format(figure_id))
    plt.legend(fancybox=False)

    plt.tight_layout()
    save_path = plot_path / '{}_overall_performance.png'.format(figure_id)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def plot_trials_to_correct_summary(block_performance: pd.DataFrame, plot_path: Path, figure_id: str):
    """plot the trials to correct across a set of collected blocks, which can be from multiple sessions
    (choose carefully!). To compare early vs late, you'll need to copy the code in F31_Apr2024."""

    f, ax = plt.subplots()
    rewards, mean, std, sem = session_analysis.summarize_block_switches(block_performance)
    plt.plot(mean.index, mean)
    plt.fill_between(mean.index, mean - sem,
                     mean + sem, alpha=.3, linewidth=0)

    plt.xlabel('Consecutive rewards')
    plt.ylabel('Trials to Correct')
    plt.title('Trials to Switch vs Rewards')
    plt.tight_layout()

    save_path = plot_path / '{}_trials-to-correct-summary.png'.format(figure_id)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def scatter_trials_to_correct(block_performance: pd.DataFrame, slope: float, intercept: float, plot_path: Path, figure_id: str):
    f1, ax1 = plt.subplots()
    x = np.arange(len(block_performance))
    ix_valid = (block_performance['trials_to_correct'] != 'None') & (block_performance['prev_n_correct'] != 'None')
    block_performance = block_performance[ix_valid]

    block_type = block_performance['block_type'].to_numpy()
    trials_to_correct = block_performance['trials_to_correct'].astype(int).to_numpy()
    prev_consecutive_rewards = block_performance['prev_consecutive_rewards'].astype(int).to_numpy()

    for b in block_types:
        ix = np.where(block_type == b)[0]
        if len(ix) > 0:
            ax1.plot(prev_consecutive_rewards[ix], trials_to_correct[ix],
                     'o', color=color_dict[b], label=b)

    x = np.array([0, np.amax(prev_consecutive_rewards)])
    ax1.plot(x, slope*x + intercept, 'k--')

    plt.ylabel('Trials to Correct')
    plt.xlabel('Consecutive Rewards')
    plt.title('{} Trials to Switch vs Rewards'.format(figure_id))
    plt.legend(fancybox=False)

    plt.tight_layout()
    save_path = plot_path / '{}_scatter_trials-to-correct.png'.format(figure_id)
    f1.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def plot_session_trials_to_correct(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """For a single session, plot the blockwise trials to the correct choice made by the agent."""
    f, ax = plt.subplots()
    x = np.arange(len(block_performance))
    for b in block_types:
        ix = np.where(block_performance['block_type'] == b)[0]
        if len(ix) > 0:
            ax.plot(x[ix], block_performance['trials_to_correct'][ix], 'o', color=color_dict[b], label=b)

    plt.ylabel('Trials to Correct')
    plt.xlabel('Block')
    plt.title('{} Blockwise Trials to Correct'.format(sess_ID))
    plt.legend(fancybox=False)

    plt.tight_layout()
    save_path = plot_path / '{}_block_trials_to_correct.png'.format(sess_ID)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def plot_session_nswitches(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """For a single session, plot the blockwise number of switches. Will approximate confusion."""
    f, ax = plt.subplots()
    x = np.arange(len(block_performance))
    for b in block_types:
        ix = np.where(block_performance['block_type'] == b)[0]
        if len(ix) > 0:
            ax.plot(x[ix], block_performance['n_switches'][ix], 'o', color=color_dict[b], label=b)

    plt.ylabel('Num Switches in block')
    plt.xlabel('Block')
    plt.title('{} Block Confusion'.format(sess_ID))
    plt.legend(fancybox=False)

    plt.tight_layout()
    save_path = plot_path / '{}_block_nswitches.png'.format(sess_ID)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def plot_multisession_trials_to_correct(overall_df: pd.DataFrame, plot_path: Path, figure_id: str, use_dates: bool=True):
    block_types = ['right_cued_trials_to_correct', 'left_cued_trials_to_correct',
                   'right_uncued_trials_to_correct', 'left_uncued_trials_to_correct']
    color_dict = {'right_cued_trials_to_correct': 'darkred', 'left_cued_trials_to_correct': 'darkblue',
                  'right_uncued_trials_to_correct': 'red', 'left_uncued_trials_to_correct': 'blue'}
    overall_df = overall_df[~overall_df['date'].isna()]
    dates = overall_df['date'].unique()

    f, ax = plt.subplots()
    x = np.arange(overall_df.shape[0])
    for b in block_types:
        type_ttc = overall_df[b].values
        if np.sum(np.isnan(type_ttc)) == type_ttc.size:  # will have to remove this later on...
            continue

        type_ttc[np.isnan(type_ttc)] = 0
        ax.plot(x, type_ttc, color=color_dict[b], label=b)
        if use_dates:
            ax.set_xticks(x, dates)
            ax.tick_params(axis='x', labelrotation=60, labelsize=8)

    plt.ylabel('Trials to Correct')
    plt.xlabel('Session')
    plt.title('{} Overall Trials to Correct'.format(figure_id))
    plt.legend(fancybox=False)

    plt.tight_layout()
    save_path = plot_path / '{}_overall_trials_to_correct.png'.format(figure_id)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    plt.close('all')


def plot_learning_curve(coefficients: np.ndarray, switches_per_session: np.ndarray, figure_id: str, plot_path: Path, dates: list[str]=None):
    f, ax = plt.subplots()
    plt.plot(np.arange(1, len(coefficients) + 1), coefficients, 'ko')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Regression Coefficient', fontsize=18)
    plt.title('{} Learning Curve'.format(figure_id))
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.tick_params(width=2)

    if dates is not None:
        ax.set_xticks(np.arange(1, len(dates) + 1))
        ax.set_xticklabels(dates, rotation=60, fontsize=8)

    plt.tight_layout()
    save_path = plot_path / '{}_learning-curve.png'.format(figure_id)
    f.savefig(save_path, format='png', dpi=300)
    # f.savefig(plot_path / '{}_learning-curve.svg'.format(save_name), format='svg')
    print('Saved as {}'.format(save_path))
    plt.close('all')


def main_single_session():
    multisession_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    processed_data_path = session_data_home / 'processed'
    session_figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
        sess_id_abbreviated = mouse + '_' + date
    else:
        print("Double-check the session name!")
        return

    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(sess_id_full,
                                                                                            session_data_folder=processed_data_path,
                                                                                            multisession_data_folder=multisession_save_path)
    overall_df = multisession_df[~multisession_df['date'].isna()]
    plot_session_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_trials_to_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_nswitches(block_performance, session_figure_path, sess_id_full)
    # plot_multisession_correct(overall_df, mouse_plot_path, figure_id=mouse)
    # plot_multisession_trials_to_correct(overall_df, mouse_plot_path, figure_id=mouse)

    slope = multisession_df.loc[multisession_df['date']==date, 'slope'].values[0]
    intercept = multisession_df.loc[multisession_df['date']==date, 'intercept'].values[0]
    scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
                              plot_path=session_figure_path, figure_id=sess_id_full)
    plot_learning_curve(multisession_df['slope'], multisession_df['n_switches'], figure_id=mouse, plot_path=multisession_save_path,
                        dates=multisession_df['date'].values)


def main_multiple_sessions():
    plot_path = Path('../../reports/figures')
    experiment_folder = Path('/home/matt/Documents/EXPERIMENTS/')
    # experiment_folder = Path('C:/Users/mattc/EinsteinMed Dropbox/Matthew Chin/phd_data/remotework/EXPERIMENTS/')
    raw_data_path = experiment_folder / 'raw_behavior_data'
    processed_data_path = experiment_folder / 'processed_data'
    # data_path = Path('../../data/processed')

    ### Plotting for multiple sessions ###
    mouse = ['CT010']
    date = ['2024-08-23']

    # date = ['2024-08-22', '2024-08-23', '2024-08-26', '2024-08-28', '2024-08-29',
    #         '2024-08-30', '2024-09-06', '2024-09-07']
    # mouse = ['CT005'] * len(date)

    # block_performance_list = []
    # for m, d in zip(mouse, date):
    #     sess_ID = m + '_' + d
    #     mouse_plot_path = plot_path / m / 'performance_plots'
    #     if not mouse_plot_path.exists():
    #         mouse_plot_path.mkdir()
    #
    #     overall_df, block_performance, choices_df = session_analysis.load_analysis(m, d, data_path)
    #     plot_session_correct(block_performance, mouse_plot_path, sess_ID)
    #     plot_session_trials_to_correct(block_performance, mouse_plot_path, sess_ID)
    #
    #     ix = overall_df['date'] == d
    #     if not ix.any():
    #         print('No data for {}'.format(d))
    #         continue
    #
    #     slope = overall_df.loc[overall_df['date'] == d, 'slope'].values[0]
    #     intercept = overall_df.loc[overall_df['date'] == d, 'intercept'].values[0]
    #     scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
    #                               plot_path=mouse_plot_path, figure_id=sess_ID)
    #
    #     block_performance_list.append(block_performance)

    # block_performance_combined = pd.concat(block_performance_list, axis=0, ignore_index=True)
    # plot_trials_to_correct_summary(block_performance_combined, mouse_plot_path, figure_id=mouse[0])

    overall_df = pd.read_csv(processed_data_path / mouse[0] / (mouse[0] + '_overall_performance.csv'))
    overall_df.sort_values(by='date', inplace=True)
    # plot_learning_curve(overall_df['slope'], overall_df['n_switches'], figure_id=mouse[0], plot_path=mouse_plot_path)
    plot_learning_curve(overall_df['slope'], overall_df['n_switches'], figure_id=mouse[0], plot_path=processed_data_path / mouse[0])


def main():
    main_single_session()
    # main_multiple_sessions()


if __name__ == '__main__':
    main()

