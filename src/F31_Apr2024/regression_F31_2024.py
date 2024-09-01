import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from formulaic import model_matrix
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle as pkl
from pathlib import Path
import src.behavior_analysis.context_switch_analysis as vdp
from typing import Any, List, Tuple, Union
from icecream import ic


def plot_trials_to_switch(df: pd.DataFrame, key: Any, save_name: str, plot_path: Path, max_ix: int=None):
    _, state_df = vdp.get_multisession_switches(df)
    rewards, mean, std, sem = vdp.consecutive_summary_measures(state_df, key=key, min_counts=0)
    if max_ix:
        mean = mean[:max_ix]
        sem = sem[:max_ix]

    f_2D, ax_2D = plt.subplots()
    plt.plot(mean.index, mean)
    plt.fill_between(mean.index, mean - sem,
                     mean + sem, alpha=.3, linewidth=0)

    plt.xlabel('Consecutive rewards')
    plt.ylabel(key)

    # plt.legend()
    plt.title('Trials to Switch vs Rewards'.format(key))
    plt.tight_layout()

    save_path = plot_path / '{}_trials-to-switch.png'.format(save_name)
    f_2D.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))


def plot_single_session_regression(df: pd.DataFrame, key: Any, save_name: str, plot_path: Path):
    f, ax = plt.subplots()
    _, state_df = vdp.get_multisession_switches(df)
    rewards, mean, std, sem = vdp.consecutive_summary_measures(state_df, key=key, min_counts=0)
    plt.plot(mean.index, mean)
    plt.fill_between(mean.index, mean - sem,
                     mean + sem, alpha=.3, linewidth=0)

    plt.xlabel('Consecutive rewards')
    plt.ylabel(key)
    plt.title('Trials to Switch vs Rewards'.format(key))
    plt.tight_layout()

    save_path = plot_path / '{}_trials-to-switch.png'.format(save_name)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))


def plot_multiple_session_regression(df_list: List[pd.DataFrame], df_labels: List[str], key: Any, save_name: str, plot_path: Path,
                                     max_ix: int=None):
    f, ax = plt.subplots()
    for i, df in enumerate(df_list):
        _, state_df = vdp.get_multisession_switches(df)
        rewards, mean, std, sem = vdp.consecutive_summary_measures(state_df, key=key, min_counts=0)
        if max_ix:
            mean = mean.loc[:max_ix]
            sem = sem.loc[:max_ix]

        plt.plot(mean.index, mean, label=df_labels[i], )
        plt.fill_between(mean.index, mean - sem,
                         mean + sem, alpha=.3, linewidth=0)

    plt.xlabel('Consecutive rewards', fontsize=18)
    plt.ylabel("Trials to Switch", fontsize=16)
    # ax.tick_params(axis='y', which='major', labelsize=12)
    # ax.yticks(fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(frameon=False, fontsize=14)
    # plt.title('Trials to Switch vs Rewards'.format(key))
    if max_ix:
        plt.xticks(np.arange(1, max_ix+1, 1))
        plt.xlim([1, max_ix])

    plt.tight_layout()
    save_path = plot_path / '{}_trials-to-switch.png'.format(save_name)
    f.savefig(save_path, format='png', dpi=300)
    save_path = plot_path / '{}_trials-to-switch.svg'.format(save_name)
    f.savefig(save_path, format='svg')
    print('Saved as {}'.format(save_path))


def visualize_switches(df_list: List[pd.DataFrame], df_labels: List[str]):
    # _, state_df = vdp.get_multisession_switches(df)
    # rewards, mean, std, sem = vdp.consecutive_summary_measures(state_df, key=key, min_counts=0)

    f, ax = plt.subplots()
    for i, df in enumerate(df_list):
        _, state_df = vdp.get_multisession_switches(df)
        plt.plot(state_df['consecutive_rewards'], state_df['trials_to_correct'], 'o',
                 label=df_labels[i], alpha=0.5)

        coef = vdp.session_stats(state_df, key='trials_to_correct')
        print(coef)

    plt.xlabel('Consecutive rewards', fontsize=18)
    plt.ylabel("Trials to Switch", fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(frameon=False, fontsize=14)
    plt.tight_layout()
    ax.legend()
    plt.show()


def plot_learning_curve(df: pd.DataFrame, save_name: str, plot_path: Path):
    sessions = np.unique(df['session_ID'])
    coefs, switches_per_session = vdp.collect_regression_coefficients(df, sessions)
    ic(coefs)

    f, ax = plt.subplots()
    plt.plot(np.arange(1, len(coefs)+1), coefs, 'k')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Regression Coefficient', fontsize=18)
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.tick_params(width=2)

    plt.tight_layout()
    f.savefig(plot_path / '{}_learning-curve.png'.format(save_name), format='png', dpi=300)
    # f.savefig(plot_path / '{}_learning-curve.svg'.format(save_name), format='svg')


def main_single_session():
    plot_path = Path('../../reports/figures/F31_Apr2024')
    data_path = Path('../../data/processed/Mitch_behavior')

    # load the mouse performance dataframe
    mouse = 'MF03'
    date = '2023-10-02'
    sess_ID = mouse + '_' + date
    p = data_path / (sess_ID + '_performance.pkl')
    with p.open('rb') as f:
        df = pkl.load(f)

    vdp.consecutive_summary_measures(df, key='trials_to_correct', save_name=sess_ID,
                              plot_path=plot_path, min_counts=0)


def main_multiple_sessions():
    plot_path = Path('../../reports/figures/F31_Apr2024')
    data_path = Path('../../data/processed/Mitch_behavior')

    # mice = ['MF03'] * 2
    # dates = ['2023-10-02', '2023-10-03']

    # mice = ['MF24'] * 7
    # dates = ['2023-10-04', '2023-10-05', '2023-10-06', '2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13']

    mice = ['MF24'] * 3
    dates = ['2023-10-04', '2023-10-05', '2023-10-06']
    early_sess_ids = [m + '_' + d for m, d in zip(mice, dates)]

    mice = ['MF24'] * 4
    dates = ['2023-10-10', '2023-10-11', '2023-10-12', '2023-10-13']
    late_sess_ids = [m + '_' + d for m, d in zip(mice, dates)]
    all_sess_ids = early_sess_ids + late_sess_ids

    early_df_list = []
    for sess in early_sess_ids:
        # sess_ID = sess[0] + '_' + sess[1]
        p = data_path / (sess + '_events.pkl')
        with p.open('rb') as f:
            df = pkl.load(f)
        early_df_list.append(df)
    early_df = pd.concat(early_df_list)

    late_df_list = []
    for sess in late_sess_ids:
        # sess_ID = sess[0] + '_' + sess[1]
        p = data_path / (sess + '_events.pkl')
        with p.open('rb') as f:
            df = pkl.load(f)
        late_df_list.append(df)
    late_df = pd.concat(late_df_list)
    full_df = pd.concat([early_df, late_df])

    # plot_trials_to_switch(df, key='trials_to_correct', save_name='MF24-early',
    #                           plot_path=plot_path)

    for id in all_sess_ids:
        _df = full_df[full_df['session_ID'] == id]
        plot_single_session_regression(_df, key='trials_to_correct', save_name=id + '_regression', plot_path=plot_path)

    # plot_multiple_session_regression(df_list=[early_df, late_df], df_labels=['Early','Late'], key='trials_to_correct',
    #                                  max_ix=5, save_name='MF24-early-vs-late', plot_path=plot_path)

    # plot_learning_curve(full_df, save_name='MF24-full', plot_path=plot_path)

    # visualize_switches(full_df, key='trials_to_correct', save_name='MF24-full', plot_path=plot_path)
    # visualize_switches(df_list=[early_df, late_df], df_labels=['Early','Late'])


if __name__ == '__main__':
    main_multiple_sessions()