import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# import controller
# import demo_tasks
from formulaic import model_matrix
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle as pkl
from pathlib import Path
import fileIO
import decision_variable_counters as counters


from typing import Tuple, List, Any, Iterable
from collections import defaultdict

# plt.style.use('dark_background')


def get_relative_value(df, max_rewards):
    ix_trial = df['cur_trial'] < max_rewards
    grouped_values = df.loc[ix_trial].groupby(['cur_trial'])
    mean_value = grouped_values.mean().loc[:, 'relative_value']
    std_value = grouped_values.std().loc[:, 'relative_value']
    sem_value = grouped_values.sem().loc[:, 'relative_value']
    return mean_value, std_value, sem_value


def value_accumulation_plot(plot_name='value_accumulation'):
    """This function was originally made to copy figure 1D from Vertechi 2020, comparing HMM and FQ-learning."""
    df_HMM = demo_tasks.load_collected_runs(agent_type='HMM', p_reward=.9, p_switch=.3, rewards_before_switch=6)
    df_FQL = demo_tasks.load_collected_runs(agent_type='F-Qlearning', p_reward=.9, p_switch=.3, rewards_before_switch=6)

    mean_HMM_value, std_HMM_value, sem_HMM_value = get_relative_value(df_HMM, max_rewards=6)
    mean_QL_value, std_QL_value, sem_QL_value = get_relative_value(df_FQL, max_rewards=6)

    f, ax = plt.subplots()
    X = np.arange(6) + 1
    ax.plot(X, mean_HMM_value, 'k', label='HMM')
    ax.plot(X, mean_QL_value, 'k--', label='FQ-learning')
    plt.xlabel('Number of rewards')
    plt.ylabel('Relative value')
    plt.ylim([0,1])
    plt.legend()
    plt.title('Relative value accumulation')
    plt.tight_layout()
    # plt.show()
    figure_format = 'png'
    plt.savefig('../figures/{}.{}'.format(plot_name, figure_format), format=figure_format, dpi=300)


def get_session_switch_ix(session_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    # df should be for one session, not multiple sessions concatenated
    actions = session_df['action'].values
    states = session_df['state'].values
    ix_action_switch = actions[1:] != actions[:-1]
    ix_state_switch = states[1:] != states[:-1]

    ix_action_switch = np.nonzero(ix_action_switch)[0] + 1
    ix_state_switch = np.nonzero(ix_state_switch)[0] + 1
    return ix_action_switch, ix_state_switch


def count_consecutive_events(df) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # df should only be for one session, not a bunch concatenated
    # TODO - integrate with analyze_session.py code, this only works for agents with no skipped blocks
    consecutive_rewards = 0
    consecutive_failures = 0
    consecutive_rewards_renewal = 0
    consecutive_failures_renewal = 0
    negative_value = 0
    previous_trial_rewarded = False

    ix_action_switch, ix_state_switch = get_session_switch_ix(df)
    # ix_state_switch = np.nonzero(ix_state_switch)[0]  # convert to indices
    action_switch_dict = defaultdict(list)
    state_switch_dict = defaultdict(list)
    # trials_to_correct_dict = defaultdict(list)
    # trials_to_value_dict = defaultdict(list)

    n_trials = df.shape[0]
    # count = np.arange(n_trials)
    # correct_value_sign = np.zeros(n_trials)
    # correct_value_sign[df['state'] == 0] = -1
    # correct_value_sign[df['state'] == 1] = 1
    # agent_value_sign = np.sign(df['relative_value'].values)
    # correct_value_ix = np.nonzero(agent_value_sign == correct_value_sign)[0]
    correct_trial_ix = np.nonzero(df['correct'].values)[0]

    # need 2 sets of counts
    # 1. the integrate-and-reset params from Cazettes
    # 2. the last-seen version used, which resets when failures/rewards start anew but don't reset on switches
    for i in range(n_trials-1):
        reward = df.loc[i, 'reward']
        negative_value = counters.negative_value_counter(negative_value, reward)
        consecutive_rewards_renewal = counters.consecutive_reward_renewal_counter(consecutive_rewards_renewal, reward, previous_trial_rewarded)
        consecutive_failures_renewal = counters.consecutive_fail_renewal_counter(consecutive_failures_renewal, reward, previous_trial_rewarded)

        consecutive_rewards = counters.consecutive_reward_counter(consecutive_rewards, reward)
        consecutive_failures = counters.consecutive_fail_counter(consecutive_failures, reward)

        # if ix_action_switch[i] and df.loc[i, 'correct']:
        if i in ix_action_switch and df.loc[i, 'correct']:
            action_switch_dict['trial_ix'].append(i)
            action_switch_dict['consecutive_rewards'].append(consecutive_rewards_renewal)
            action_switch_dict['consecutive_failures'].append(consecutive_failures_renewal)
            action_switch_dict['negative_value'].append(negative_value)
            action_switch_dict['action_switched_from'].append(df.loc[i, 'action'])
            action_switch_dict['state_switched_from'].append(df.loc[i, 'state'])

        future_switches = ix_state_switch[ix_state_switch > i]
        next_switch = np.append(future_switches, n_trials)[0]
        if i in ix_state_switch:
            # state_change_ix = i+1
            state_change_ix = i
            first_correct_trial_ix = (correct_trial_ix > state_change_ix) & (correct_trial_ix < next_switch)
            # first_correct_value_ix = (correct_value_ix > state_change_ix) & (correct_value_ix < next_switch)
            if first_correct_trial_ix.any():
                trials_to_correct = correct_trial_ix[first_correct_trial_ix][0] - i
                # trials_to_value = correct_value_ix[first_correct_value_ix][0] - i
            else:
                trials_to_correct = np.nan

            # if first_correct_trial_ix.any() or first_correct_value_ix.any():
            if first_correct_trial_ix.any():
                state_switch_dict['state_change_ix'].append(state_change_ix)
                # state_switch_dict['consecutive_rewards'].append(consecutive_rewards)
                state_switch_dict['consecutive_rewards'].append(consecutive_rewards_renewal)
                state_switch_dict['consecutive_failures'].append(consecutive_failures_renewal)
                state_switch_dict['negative_value'].append(negative_value)
                state_switch_dict['trials_to_correct'].append(trials_to_correct)
                # state_switch_dict['trials_to_value'].append(trials_to_value)

        previous_trial_rewarded = reward > 0

    action_switch_df = pd.DataFrame(action_switch_dict)
    state_switch_df = pd.DataFrame(state_switch_dict)
    return action_switch_df, state_switch_df


def get_demo_switches(agent_type: str, p_reward: float, p_switch: float, max_rewards: int) -> pd.DataFrame:
    df_list = []
    reward_list = np.arange(max_rewards) + 1
    for i in reward_list:
        df_agent = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch,
                                                  rewards_before_switch=i)
        # get a single session
        agent_ids = np.unique(df_agent['agent_ID'])
        for id in agent_ids:
            ix = df_agent['agent_ID'] == id  # probably change to sessionID for uniqueness
            switch_df = count_consecutive_events(df_agent.loc[ix])
            df_list.append(switch_df)

    full_switch_df = pd.concat(df_list)
    return full_switch_df


def get_session_switches(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    action_df_list = []
    state_df_list = []

    _action_switch_df, _state_switch_df = count_consecutive_events(df)
    action_df_list.append(_action_switch_df)
    state_df_list.append(_state_switch_df)

    action_switch_df = pd.concat(action_df_list)
    state_switch_df = pd.concat(state_df_list)
    return action_switch_df, state_switch_df


def get_multisession_switches(concatenated_sessions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    action_df_list = []
    state_df_list = []
    # regression_coefs = []

    # run a single session at a time
    sess_ids = np.unique(concatenated_sessions_df['session_ID'])
    for id in sess_ids:
        ix = concatenated_sessions_df['session_ID'] == id
        _action_switch_df, _state_switch_df = count_consecutive_events(concatenated_sessions_df.loc[ix])
        action_df_list.append(_action_switch_df)
        state_df_list.append(_state_switch_df)

        # coef = session_stats(_state_switch_df, key='trials_to_correct')
        # regression_coefs.append(coef)

    action_switch_df = pd.concat(action_df_list)
    state_switch_df = pd.concat(state_df_list)
    return action_switch_df, state_switch_df


def consecutive_summary_measures(df: pd.DataFrame, key: Any, min_counts=20) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grouped_switches = df.groupby(['consecutive_rewards'])[key]
    counts = grouped_switches.count()
    # print(counts)

    ix = counts > min_counts
    mean = grouped_switches.mean()[ix]
    std = grouped_switches.std()[ix]
    sem = grouped_switches.sem()[ix]
    rewards = counts.index[ix]
    return rewards, mean, std, sem


def fig_1E(plot_name='Vertechi2020_1E'):
    # make the mean-subtracted (1E) and non-subtracted (2D) versions
    # With real data, will have to mold it into compatibility with these fxns: basically that's the HMM_df and QL_df
    # The rest should work fine
    # HMM_df = get_demo_switches(agent_type='HMM', p_reward=.9, p_switch=.9, max_rewards=6)
    # QL_df = get_demo_switches(agent_type='F-Qlearning', p_reward=.9, p_switch=.9, max_rewards=6)

    HMM_action_df, HMM_state_df = get_session_switches(agent_type='HMM', p_reward=.9, p_switch=.3)
    FQL_action_df, FQL_state_df = get_session_switches(agent_type='F-Qlearning', p_reward=.9, p_switch=.3)
    # QL_action_df, QL_state_df = get_session_switches(agent_type='Qlearning', p_reward=.9, p_switch=.3)

    HMM_rewards, HMM_mean_failures, HMM_std_failures, HMM_sem_failures = count_consecutives(HMM_action_df, key='consecutive_failures')
    FQL_rewards, FQL_mean_failures, FQL_std_failures, FQL_sem_failures = count_consecutives(FQL_action_df, key='consecutive_failures')
    # QL_rewards, QL_mean_failures, QL_std_failures, QL_sem_failures = count_consecutives(QL_df)

    HMM_mean_subtracted_failures = HMM_mean_failures - HMM_mean_failures.mean()
    FQL_mean_subtracted_failures = FQL_mean_failures - FQL_mean_failures.mean()
    # QL_mean_subtracted_failures = QL_mean_failures - QL_mean_failures.mean()

    # Figure 1E
    f_1E, ax_1E = plt.subplots()
    plt.plot(HMM_rewards, HMM_mean_subtracted_failures, 'k', label='HMM')
    plt.plot(FQL_rewards, FQL_mean_subtracted_failures, 'k--', label='FQ-learning')
    # plt.plot(QL_rewards, QL_mean_subtracted_failures, 'k-', label='Q-learning')
    plt.xlabel('Consecutive rewards')
    plt.ylabel('Consecutive failures (mean-subtracted)')
    plt.title('Consecutive failures (mean-subtracted) vs rewards')
    plt.legend()
    plt.tight_layout()
    figure_format = 'png'
    # plt.show()
    f_1E.savefig('../figures/{}.{}'.format('Vertechi2020_1E', figure_format), format=figure_format, dpi=300)
    print('Saved Vertechi 2020 Figure 1E')

    f_2D, ax_2D = plt.subplots()
    plt.plot(HMM_mean_failures.index, HMM_mean_failures, 'k', label='HMM')
    plt.fill_between(HMM_mean_failures.index, HMM_mean_failures - HMM_sem_failures,
                     HMM_mean_failures + HMM_sem_failures,
                     color='k', alpha=.3, linewidth=0)

    plt.plot(FQL_mean_failures.index, FQL_mean_failures, 'm', label='FQ-learning')
    plt.fill_between(FQL_mean_failures.index, FQL_mean_failures - FQL_sem_failures,
                     FQL_mean_failures + FQL_sem_failures,
                     color='m', alpha=.3, linewidth=0)

    # plt.plot(QL_mean_failures.index, QL_mean_failures, 'r', label='Q-learning')
    # plt.fill_between(QL_mean_failures.index, QL_mean_failures - QL_sem_failures,
    #                  QL_mean_failures + QL_sem_failures,
    #                  color='r', alpha=.3, linewidth=0)
    plt.ylim([0, 10])
    plt.xlabel('Consecutive rewards')
    plt.ylabel('Consecutive failures')

    plt.legend()
    plt.title('Consecutive failures vs rewards')
    plt.tight_layout()

    f_2D.savefig('../figures/{}.{}'.format('Vertechi2020_2D', figure_format), format=figure_format, dpi=300)
    print('Saved Vertechi 2020 Figure 2D')

    HMM_rewards, HMM_mean_trials_to_corr, HMM_std_trials_to_corr, HMM_sem_trials_to_corr = count_consecutives(HMM_state_df,
                                                                                            key='trials_to_correct')
    FQL_rewards, FQL_mean_trials_to_corr, FQL_std_trials_to_corr, FQL_sem_trials_to_corr = count_consecutives(FQL_action_df,
                                                                                            key='trials_to_correct')

    HMM_mean = HMM_mean_trials_to_corr - HMM_mean_trials_to_corr.mean()
    FQL_mean = FQL_mean_failures - FQL_mean_failures.mean()


def session_stats(df: pd.DataFrame, key: Any, plot=False):
    stats_df = pd.DataFrame({
        'y': df[key],
        'a': df['consecutive_rewards'],
    })

    y, X = model_matrix("y ~ a", stats_df)
    model = sm.OLS(y, X)
    results = model.fit()
    r_value, p_value = results.rsquared, results.pvalues['a']
    intercept, slope = results.params['Intercept'], results.params['a']

    # print(results.summary())
    # print("Parameters", results.params)
    # print("R-squared", results.rsquared)
    # print("Std Errors", results.bse)

    # if plot:
    #     pred_ols = results.get_prediction()
    #     iv_l = pred_ols.summary_frame()["obs_ci_lower"]
    #     iv_u = pred_ols.summary_frame()["obs_ci_upper"]
    #
    #     f, ax = plt.subplots()
    #     ax.plot(df['consecutive_rewards'], df[key], "o", label="data")
    #     ax.plot(df['consecutive_rewards'], results.fittedvalues, "r--.", label="OLS")
    #     ax.plot(df['consecutive_rewards'], iv_u, "r--")
    #     ax.plot(df['consecutive_rewards'], iv_l, "r--")
    #     ax.legend(loc="best")
    #     plt.show()

    return slope


def compare_consecutive_rewards(p_reward: float, p_switch: float, key: Any, min_counts=20):
    # load the mouse
    mouse = 'MF03'
    date = '2023-10-03'
    sess_ID = mouse + '_' + date
    p = Path('../mouse_behavior') / (sess_ID + '_performance.pkl')
    with p.open('rb') as f:
        mouse_df = pkl.load(f)
    _, mouse_state_df = get_multisession_switches(mouse_df)

    agent_type = 'F-Qlearning'
    FQL_df = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch, n_agents=10)
    _, FQL_state_df = get_multisession_switches(FQL_df)

    agent_type = 'HMM'
    HMM_df = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch, n_agents=10)
    _, HMM_state_df = get_multisession_switches(HMM_df)

    HMM_rewards, HMM_mean, HMM_std, HMM_sem = consecutive_summary_measures(HMM_state_df, key=key, min_counts=min_counts)
    FQL_rewards, FQL_mean, FQL_std, FQL_sem = consecutive_summary_measures(FQL_state_df, key=key)
    mouse_rewards, mouse_mean, _, mouse_sem = consecutive_summary_measures(mouse_state_df, key=key)

    HMM_mean_subtracted = HMM_mean - HMM_mean.mean()
    FQL_mean_subtracted = FQL_mean - FQL_mean.mean()

    # Figure 1E
    f_1E, ax_1E = plt.subplots()
    plt.plot(HMM_rewards, HMM_mean_subtracted, 'k', label='HMM')
    plt.plot(FQL_rewards, FQL_mean_subtracted, 'k--', label='FQ-learning')
    # plt.plot(QL_rewards, QL_mean_subtracted_failures, 'k-', label='Q-learning')
    plt.xlabel('Consecutive rewards')
    plt.ylabel('{} (mean-subtracted)'.format(key))
    plt.title('Consecutive failures (mean-subtracted) vs rewards')
    plt.legend()
    plt.tight_layout()
    figure_format = 'png'
    # plt.show()
    f_1E.savefig('../figures/{}.{}'.format('Vertechi2020_1E', figure_format), format=figure_format, dpi=300)
    print('Saved Vertechi 2020 Figure 1E')

    f_2D, ax_2D = plt.subplots()
    plt.plot(HMM_mean.index, HMM_mean, 'w', label='HMM')
    plt.fill_between(HMM_mean.index, HMM_mean - HMM_sem,
                     HMM_mean + HMM_sem,
                     color='w', alpha=.3, linewidth=0)

    plt.plot(FQL_mean.index, FQL_mean, 'm', label='FQ-learning')
    plt.fill_between(FQL_mean.index, FQL_mean - FQL_sem,
                     FQL_mean + FQL_sem,
                     color='m', alpha=.3, linewidth=0)

    plt.plot(mouse_mean.index, mouse_mean, label='mouse')
    plt.fill_between(mouse_mean.index, mouse_mean - mouse_sem,
                     mouse_mean + mouse_sem, alpha=.3, linewidth=0)

    # plt.ylim([0, 8])
    # plt.xlim([0, 6])
    plt.xlabel('Consecutive rewards')
    plt.ylabel(key)

    plt.legend()
    plt.title('Trials to Correct vs Rewards'.format(key))
    plt.tight_layout()

    f_2D.savefig('../figures/{}.{}'.format('Vertechi2020_2D', figure_format), format=figure_format, dpi=300)
    print('Saved Vertechi 2020 Figure 2D')


def collect_regression_coefficients(df: pd.DataFrame, sess_IDs: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray]:
    regression_coefs = []
    n_switches = []
    for sess in sess_IDs:
        ix = df['session_ID'] == sess
        action_df, state_df = count_consecutive_events(df[ix])
        coef = session_stats(state_df, key='trials_to_correct')
        regression_coefs.append(coef)
        n_switches.append(state_df.shape[0])

    print('Number of switches:', n_switches)
    return np.array(regression_coefs), np.array(n_switches)


def compare_regression_coefficients(p_reward: float, p_switch: float):
    agent_type = 'F-Qlearning'
    df_agent = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch, n_agents=10)
    sessions = np.unique(df_agent['session_ID'])
    QL_coefs = collect_regression_coefficients(df_agent, sessions)

    agent_type = 'HMM'
    df_agent = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch, n_agents=10)
    sessions = np.unique(df_agent['session_ID'])
    HMM_coefs = collect_regression_coefficients(df_agent, sessions)

    markersize = 8
    f, ax = plt.subplots(figsize=(4, 4))
    plt.scatter(np.zeros_like(QL_coefs), QL_coefs, c='m', label='RL', s=markersize)
    plt.scatter(np.ones_like(HMM_coefs), HMM_coefs, c='w', label='Inference', s=markersize)
    plt.xticks([0, 1], ['RL', 'Inference'])
    plt.ylim([np.amin(HMM_coefs)-.05, 1])
    plt.xlim([-.1, 1.1])

    # plt.legend()
    plt.title('Regression Coefficients by Model')
    plt.tight_layout()

    # plt.show()
    figure_format = 'png'
    f.savefig('../figures/{}.{}'.format('Vertechi2020_2E', figure_format), format=figure_format, dpi=300)
    print('Saved Vertechi 2020 Figure 2E')


def main():
    p_reward = .9
    p_switch = .07

    # fig_1D()
    # fig_1E()
    # fig_2D()

    # HMM_action_df, HMM_state_df = get_session_switches(agent_type='HMM', p_reward=.9, p_switch=.3)
    # FQL_action_df, FQL_state_df = get_session_switches(agent_type='F-Qlearning', p_reward=.9, p_switch=.3)
    # compare_consecutive_rewards(HMM_df=HMM_action_df, FQL_df=FQL_action_df, key='consecutive_failures')
    # compare_consecutive_rewards(HMM_df=HMM_state_df, FQL_df=FQL_state_df, key='trials_to_correct', min_counts=10)
    # compare_consecutive_rewards(HMM_df=HMM_state_df, FQL_df=FQL_state_df, key='trials_to_value', min_counts=10)

    # compare_consecutive_rewards(p_reward=p_reward, p_switch=p_switch, key='trials_to_correct', min_counts=10)
    # compare_regression_coefficients(p_reward=p_reward, p_switch=p_switch)

    # mouse_consecutive_rewards(df, key='trials_to_correct', min_counts=10)


if __name__ == '__main__':
    main()