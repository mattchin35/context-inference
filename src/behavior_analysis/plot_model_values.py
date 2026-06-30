import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional
from pathlib import Path
import pickle as pkl
import re
import json

from src.behavior_analysis import ideal_observer
from src.visualization.agent_run_plot import plot_run_dataframe

plt.style.use('dark_background')

"""
Plot one or more model runs. 
Make functions to plot 1 run, and to show multiple runs on the same plot. Want to compare QL-FQL-RFLR-HMM-etc.
"""

RIGHT_IX = 0
LEFT_IX = 1
action_int_to_string = {0: 'Right', 1: 'Left'}
color_dict = {'right_cued': 'cyan', 'left_cued': 'darkgreen', 'right_uncued': 'plum', 'left_uncued': 'indigo',
              'right': 'cyan', 'left': 'darkgreen',
              0: 'cyan', 1: 'darkkhaki'}
mouse_history_colorblock_dict = {
    'right_cued': 'darkred',
    'right_uncued': 'darkred',
    'right': 'darkred',
    'left_cued': 'blue',
    'left_uncued': 'blue',
    'left': 'blue',
    'dark period': 'black',
}
state_dict = {0: 'right', 1: 'left'}


def _add_mouse_history_colorblocks(
    axes: list[plt.Axes] | np.ndarray,
    observer_run_df: pd.DataFrame,
    alpha: float = 0.12,
) -> None:
    """Add task-context spans to one or more mouse-history observer axes.

    Parameters
    ----------
    axes : list[matplotlib.axes.Axes] or np.ndarray
        Axes receiving background spans. Each axis is assumed to use trial
        index on the x-axis.
    observer_run_df : pd.DataFrame
        Observer run table with shape `(n_valid_trials, n_columns)`. Must
        contain a `state` column with task-context labels such as `left`,
        `right`, or `dark period`.
    alpha : float, default=0.12
        Transparency of the color spans in matplotlib alpha units.

    Returns
    -------
    None
        Mutates the supplied axes by adding background patches.
    """
    if "state" not in observer_run_df.columns or observer_run_df.empty:
        return

    states = observer_run_df["state"].astype(str).to_numpy()
    if states.size == 0:
        return

    state_change_ix = np.flatnonzero(states[:-1] != states[1:]) + 1
    state_starts = np.concatenate(([0], state_change_ix))
    state_ends = np.concatenate((state_change_ix, [states.size]))

    for ax in np.atleast_1d(axes):
        for start_ix, end_ix in zip(state_starts, state_ends):
            state_name = states[start_ix]
            color = mouse_history_colorblock_dict.get(state_name)
            if color is None:
                continue
            ax.axvspan(start_ix, end_ix, color=color, alpha=alpha, zorder=1)


def plot_multiple_runs(value_dfs: list[pd.DataFrame], plot_name: str, action_df: Optional[pd.DataFrame], ):
    """
    First plot all the values, then actions if they exist.
    """
    n_trials = 100
    value_dfs = [df[:n_trials] for df in value_dfs]
    base_df = value_dfs[0][0]
    states = base_df['state'].values

    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size - 1])))
    bin_types = states[bins]

    # plot model relative value and/or P(left)
    f, ax = plt.subplots(figsize=(8, 4))
    for (df, label) in value_dfs:
        action0_dist = df['action_dist'].values
        x = np.arange(action0_dist.size)
        # plt.plot(x, (1 - action0_dist) * 2 - 1, label='P(left)', c='coral', linestyle='--')
        # plt.plot(x, (1 - action0_dist) * 2 - 1, label='P(left)', linestyle='--')
        plt.plot(x, action0_dist * 2 - 1, label=label)
        # if 'relative_value' in df.keys():
        #     plt.plot(x, -df['relative_value'], label=label)
            # plt.plot(x, df['relative_value'], 'w', label='relative value')

    plt.plot(x, np.zeros_like(x), c='k', alpha=.3, linestyle=(0, (1, 1)))

    # Plot actions
    if action_df is not None:
        action_df = action_df[:n_trials]
        actions = action_df['action'].values
        rewards = action_df['reward'].values
        x = np.arange(actions.size)
        action_ix = actions * 2 - 1
        plt.scatter(x, action_ix, s=2, label='actions', c='ivory')
        rew_ix = rewards > 0
        plt.scatter(x[rew_ix], (rewards * action_ix)[rew_ix], s=10, c='ivory')  # , label='rewards')

    # FIGURE ADJUSTMENTS
    plt.yticks([-1, 1], ['Right', 'Left'])
    plt.xticks(bins)
    plt.xlim([0, x[-1]])
    plt.title('{}'.format('Mouse behavior comparison'), fontsize=14)

    # axis labels are provided by seaborn/pandas, but you can adjust them anyways to your liking
    plt.ylabel('Actions', fontsize=12)
    plt.xlabel('Trial', fontsize=12)

    ### COLOR BLOCKS ###
    unique_bins = np.unique(bin_types).tolist()
    for i in range(bins.size - 1):
        if bin_types[i] in unique_bins:
            plt.axvspan(bins[i], bins[i + 1], color=color_dict[bin_types[i]], alpha=.2, label=state_dict[bin_types[i]])
            unique_bins.remove(bin_types[i])
        else:
            plt.axvspan(bins[i], bins[i + 1], color=color_dict[bin_types[i]], alpha=.2)

    ax.legend(fancybox=False)  # , loc='lower right')
    f.tight_layout()

    figure_format = 'png'
    plt.show()
    plt.savefig('../figures/{}.{}'.format(plot_name, figure_format), format=figure_format, dpi=300)
    plt.close()
    print('saved figure as {}'.format(plot_name))


def plot_experiment(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                    relative_value: np.ndarray, p_left: np.ndarray,
                    fig_title: str, filename: str, figure_path: Path):
    """
    Plot a session's choices alongside the relative value and/or probability of a left choice.
    Provide the values/probabilities to plot as dicts with value, p_left, model_label (like HMM or RFLR), and color.
    Values and pLeft should match the length of the dataframe trials.
    """

    trial_ix = np.arange(actions.size)
    none_ix = actions == 'None'
    if none_ix.any():
        good_ix = ~none_ix
        actions = actions[good_ix].astype(int)
        trial_ix = trial_ix[good_ix]
        p_left = p_left[good_ix].astype(float)
        relative_value = relative_value[good_ix].astype(float)
        states = states[good_ix]
        rewards = rewards[good_ix]

    # get bins for task state color blocks
    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size-1])))
    bin_types = states[bins]

    action_str = [action_int_to_string[a] for a in actions]
    # x = np.arange(df['action'].size)
    # x = np.arange(actions.size)

    ### PLOT MODEL ACTIONS between -1 and 1 ###
    f, ax = plt.subplots()
    # sns.swarmplot(plot_df, x='trial', y='action', hue='model_type', s=4)  # no dodge/jitter plotting
    # sns.swarmplot(plot_df, x='trial', y='action', hue='model_type', s=4, dodge=True)
    # plt.plot(x[window:], avg_action[window:], 'k', label='average action')
    # plt.plot(x, (df['relative_value'] + 1)/2, 'k', label='relative value')

    markersize = 6
    action_ix = actions * 2 - 1
    plt.scatter(trial_ix, action_ix, s=2, label='actions', c='ivory')
    # plt.scatter(x, (1 - p_left) * 2 - 1, s=4, label='P(left)')
    # plt.plot(trial_ix, (1 - p_left) * 2 - 1, label='P(left)', c='coral', linestyle='--')
    plt.plot(trial_ix, p_left * 2 - 1, label='P(left)', c='coral', linestyle='--')

    rew_ix = rewards > 0
    plt.scatter(trial_ix[rew_ix], (rewards * action_ix)[rew_ix], s=6, c='ivory') #, label='rewards')
    plt.plot(trial_ix, relative_value, 'w', label='relative value')
    plt.plot(trial_ix, np.zeros_like(trial_ix), c='k', alpha=.3, linestyle=(0, (1, 1)))

    # FIGURE ADJUSTMENTS
    plt.yticks([-1,1],['Right', 'Left'])
    # plt.ylim([-.1,1.1])
    plt.xticks(bins,fontsize=8)
    plt.xlim([0, trial_ix[-1]])
    # task_str = 'pReward_{}_pSwitch_{}'.format(0,0)
    # model_str = 'pReward_{}_pSwitch_{}'.format(0,0)
    plt.title(fig_title, fontsize=14)

    # axis labels are provided by seaborn/pandas, but you can adjust them anyways to your liking
    plt.ylabel('Actions', fontsize=12)
    plt.xlabel('Trial', fontsize=12)
    # ax = plt.gca()

    ### COLOR BLOCKS ###
    unique_bins = np.unique(bin_types).tolist()
    for i in range(bins.size-1):
        if bin_types[i] in unique_bins:  # this first bit is just for the legend/label
            plt.axvspan(bins[i], bins[i+1], color=color_dict[bin_types[i]], alpha=.2, label=bin_types[i])
            unique_bins.remove(bin_types[i])
        else:
            plt.axvspan(bins[i], bins[i+1], color=color_dict[bin_types[i]], alpha=.2)

    ax.legend(fancybox=False)  # , loc='lower right')
    figure_format = 'png'
    f.tight_layout()
    # plt.show()
    plt.savefig(figure_path / '{}.{}'.format(filename, figure_format), format=figure_format, dpi=300)
    plt.close()
    print('saved figure as {}'.format(filename))


def plot_prior(df: pd.DataFrame):
    action_dist = df['action_dist'].values
    value_normalized = (df['relative_value'].values + 1) / 2
    try:
        prior = np.stack(df['prior'].values)
        prior_label = 'probability of state 1'

    except Exception as e:
        prior = np.stack(df['q_values'].values)
        value_normalized = (df['relative_value'].values + 1) / 2
        prior_label = 'Q-value of action 1'

    plt.plot(prior[:, 1], label=prior_label)
    plt.plot(value_normalized, label='relative value')
    plt.plot(1-action_dist, label='probability of action 1')
    plt.legend()

    # plt.title('Probability of state 1/right')
    # plt.imshow(prior)
    plt.show()


def plot_one_model():
    """
    Plot the relative value and probability of a left choice for a single model type over an existing behavior session.
    """
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    raw_behavior_folder = session_data_home / 'rpi' / sess_id_full
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

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

    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)

    p_reward = .8
    p_switch = .2

    stickiness = 1
    fql_decay = .7

    stickiness = 1
    w_reward_hist = 1.5
    w_decay = 1
    temp = .1

    agent_type = 'FQLearning'
    agent_param_str = 'stickiness={}_temp={}_decay={}'.format(stickiness, temp, fql_decay)
    model_name = 'FQ-learning'

    agent_type = 'HMM'
    agent_param_str = 'stickiness={}_temp={}_Psw={}'.format(stickiness, temp, p_switch)
    stickiness = .3
    state_transition_prob = 0.2,
    active_reward_probability =  0.8,

    agent_param_str = 'stickiness={}_Prew={}_Psw={}'.format(stickiness, active_reward_probability[0], state_transition_prob[0])
    agent_type = 'HMM'
    fname = sess_id_full + '_{}_{}'.format(agent_type, agent_param_str)
    plot_experiment(states=trial_df['block_type'].values, actions=trial_df['action'].values,
                    rewards=trial_df['reward'].values,
                    relative_value=trial_df['HMM_rel_value'].values,
                    p_left=trial_df['HMM_prob_left'].values,
                    fig_title=agent_type + '_plot', filename=fname, figure_path=figure_path)

    stickiness = .3
    w_reward_hist = .5
    w_decay = 2.6
    temp = .1
    agent_type = 'Logistic'  # RFLR, reduced-form-logistic-regression
    agent_param_str = 'stickiness={}_wRewardHist={}_wDecay={}'.format(stickiness, w_reward_hist, w_decay)
    plot_name = 'RFLR_plot'

    fname = sess_id_full + '_{}_{}'.format(agent_type, agent_param_str)
    # plot_name = '{}_plot'.format(fname)

    # sess_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(agent_type, p_reward, p_switch)
    # fname = sess_name + '_' + agent_param_str
    # extras_str = ''
    # if extras_str:
    #     fname += '_' + extras_str

    # actions = trial_df['action'].values
    # actions.astype(int)  # must deal with Nonetypes!!

    # p = Path('../saved_models') / (fname + '.pkl')
    plot_experiment(states=trial_df['block_type'].values, actions=trial_df['action'].values, rewards=trial_df['reward'].values,
                    relative_value=trial_df['RFLR_rel_value'].values, p_left=trial_df['RFLR_prob_left'].values,
                    fig_title=plot_name, filename=fname, figure_path=figure_path)

    stickiness = .1
    fql_decay = .7
    temp = .1
    agent_type = 'FQLearning'
    agent_param_str = 'stickiness={}_temp={}_decay={}'.format(stickiness, temp, fql_decay)
    plot_name = 'FQlearning_plot'
    fname = sess_id_full + '_{}_{}'.format(agent_type, agent_param_str)
    plot_experiment(states=trial_df['block_type'].values, actions=trial_df['action'].values,
                    rewards=trial_df['reward'].values,
                    relative_value=trial_df['FQlearning_rel_value'].values, p_left=trial_df['FQlearning_prob_left'].values,
                    fig_title=plot_name, filename=fname, figure_path=figure_path)

    stickiness = 0
    QL_learning_rate = .7
    temp = 0
    agent_param_str = 'stickiness={}_temp={}_lr={}'.format(stickiness, temp, QL_learning_rate)
    agent_type = 'QLearning'
    fname = sess_id_full + '_{}_{}'.format(agent_type, agent_param_str)
    plot_experiment(states=trial_df['block_type'].values, actions=trial_df['action'].values,
                    rewards=trial_df['reward'].values,
                    relative_value=trial_df['Qlearning_rel_value'].values,
                    p_left=trial_df['Qlearning_prob_left'].values,
                    fig_title=agent_type + '_plot', filename=fname, figure_path=figure_path)

    # df = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch, rewards_before_switch=6)
    # df = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch)
    # df = df[df['session_ID'] == 1]

    # plot_experiment(df, model_name=model_name, plot_name=plot_name)
    # plot_prior(df)


def plot_combined_trial_feature_values():
    """
    Plot QL/FQL/HMM (log-odds) relative-value features from augmented_trial_df on a single axis.
    Loads trial_feature_params.json written by gather_trial_features.
    """
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    trial_feature_params_path = processed_data_path / 'trial_feature_params.json'

    trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)
    with open(trial_feature_params_path, 'r') as f:
        params = json.load(f)

    required_cols = (
        'Qlearning_rel_value',
        'FQlearning_rel_value',
        'HMM_rel_value_logodds',
        'action',
        'reward',
    )
    missing_cols = [col for col in required_cols if col not in trial_df.columns]
    if missing_cols:
        raise ValueError(f"augmented_trial_df missing required columns: {missing_cols}")
    if 'block_type' not in trial_df.columns:
        raise ValueError("augmented_trial_df missing required column: ['block_type']")

    x = np.arange(trial_df.shape[0])
    ql = pd.to_numeric(trial_df['Qlearning_rel_value'], errors='coerce')
    fql = pd.to_numeric(trial_df['FQlearning_rel_value'], errors='coerce')
    hmm_logodds = pd.to_numeric(trial_df['HMM_rel_value_logodds'], errors='coerce')
    actions = pd.to_numeric(trial_df['action'], errors='coerce').to_numpy()
    rewards = pd.to_numeric(trial_df['reward'], errors='coerce').to_numpy()
    states = trial_df['block_type'].to_numpy()

    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size - 1])))
    bin_types = states[bins]

    f, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, ql, label='Qlearning_rel_value', linewidth=1.5)
    ax.plot(x, fql, label='FQlearning_rel_value', linewidth=1.5)
    ax.plot(x, hmm_logodds, label='HMM_rel_value_logodds', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=1, alpha=0.6, linestyle='--')

    valid_action_ix = ~np.isnan(actions)
    action_trial_ix = x[valid_action_ix]
    action_signed = actions[valid_action_ix] * 2 - 1
    ax.scatter(action_trial_ix, action_signed, s=4, c='ivory', label='actions')

    rewarded_ix = valid_action_ix & (rewards > 0)
    rewarded_trial_ix = x[rewarded_ix]
    rewarded_actions = actions[rewarded_ix] * 2 - 1
    ax.scatter(rewarded_trial_ix, rewarded_actions, s=10, c='white', label='rewards')

    def _state_color_label(state):
        if state in color_dict:
            return color_dict[state], state_dict.get(state, str(state))
        state_str = str(state).lower()
        if state_str in color_dict:
            return color_dict[state_str], state_dict.get(state_str, state_str)
        try:
            state_int = int(state)
            if state_int in color_dict:
                return color_dict[state_int], state_dict.get(state_int, str(state_int))
        except (TypeError, ValueError):
            pass
        return 'gray', str(state)

    unique_bins = []
    for i in range(bins.size - 1):
        color, label = _state_color_label(bin_types[i])
        if label not in unique_bins:
            ax.axvspan(bins[i], bins[i + 1], color=color, alpha=.15, label=label)
            unique_bins.append(label)
        else:
            ax.axvspan(bins[i], bins[i + 1], color=color, alpha=.15)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Relative value / action')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Right', '0', 'Left'])
    ax.set_xticks(bins)
    ax.set_title(
        f"{sess_id_full} combined model values\n"
        f"QL lr={params.get('QL_learning_rate')}, "
        f"FQL decay={params.get('FQL_decay')}, "
        f"HMM p_switch={params.get('state_transition_prob')}, "
        f"HMM p_rew={params.get('active_reward_probability')}"
    )
    ax.legend(fancybox=False)
    f.tight_layout()

    figure_format = 'png'
    plot_name = sess_id_full + '_combined_model_relative_values'
    plt.savefig(figure_path / '{}.{}'.format(plot_name, figure_format), format=figure_format, dpi=300)
    plt.close()
    print('saved figure as {}'.format(plot_name))


def plot_combined_trial_index_features():
    """
    Plot relative_doubt_index and perseveration_regressor from augmented_trial_df on a separate axis.
    """
    session_data_home = Path(
        '/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    trial_feature_params_path = processed_data_path / 'trial_feature_params.json'

    trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)
    with open(trial_feature_params_path, 'r') as f:
        params = json.load(f)

    required_cols = (
        'relative_doubt_index',
        'perseveration_regressor',
        'action',
        'reward',
    )
    missing_cols = [col for col in required_cols if col not in trial_df.columns]
    if missing_cols:
        raise ValueError(f"augmented_trial_df missing required columns: {missing_cols}")
    if 'block_type' not in trial_df.columns:
        raise ValueError("augmented_trial_df missing required column: ['block_type']")

    x = np.arange(trial_df.shape[0])
    relative_doubt = pd.to_numeric(trial_df['relative_doubt_index'], errors='coerce')
    perseveration = pd.to_numeric(trial_df['perseveration_regressor'], errors='coerce')
    actions = pd.to_numeric(trial_df['action'], errors='coerce').to_numpy()
    rewards = pd.to_numeric(trial_df['reward'], errors='coerce').to_numpy()
    states = trial_df['block_type'].to_numpy()

    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size - 1])))
    bin_types = states[bins]

    f, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, relative_doubt, label='relative_doubt_index', linewidth=1.5)
    ax.plot(x, perseveration, label='perseveration_regressor', linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=1, alpha=0.6, linestyle='--')

    valid_action_ix = ~np.isnan(actions)
    action_trial_ix = x[valid_action_ix]
    action_signed = actions[valid_action_ix] * 2 - 1
    ax.scatter(action_trial_ix, action_signed, s=4, c='ivory', label='actions')

    rewarded_ix = valid_action_ix & (rewards > 0)
    rewarded_trial_ix = x[rewarded_ix]
    rewarded_actions = actions[rewarded_ix] * 2 - 1
    ax.scatter(rewarded_trial_ix, rewarded_actions, s=10, c='white', label='rewards')

    def _state_color_label(state):
        if state in color_dict:
            return color_dict[state], state_dict.get(state, str(state))
        state_str = str(state).lower()
        if state_str in color_dict:
            return color_dict[state_str], state_dict.get(state_str, state_str)
        try:
            state_int = int(state)
            if state_int in color_dict:
                return color_dict[state_int], state_dict.get(state_int, str(state_int))
        except (TypeError, ValueError):
            pass
        return 'gray', str(state)

    unique_bins = []
    for i in range(bins.size - 1):
        color, label = _state_color_label(bin_types[i])
        if label not in unique_bins:
            ax.axvspan(bins[i], bins[i + 1], color=color, alpha=.15, label=label)
            unique_bins.append(label)
        else:
            ax.axvspan(bins[i], bins[i + 1], color=color, alpha=.15)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Index / action')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Right', '0', 'Left'])
    ax.set_xticks(bins)
    ax.set_title(
        f"{sess_id_full} trial index features\n"
        f"omission_lam={params.get('omission_lam')}, "
        f"perseveration_decay={params.get('perseveration_decay')}"
    )
    ax.legend(fancybox=False)
    f.tight_layout()

    figure_format = 'png'
    plot_name = sess_id_full + '_combined_trial_index_features'
    plt.savefig(figure_path / '{}.{}'.format(plot_name, figure_format), format=figure_format, dpi=300)
    plt.close()
    print('saved figure as {}'.format(plot_name))


def plot_user_defined_trial_columns(
    value_columns,
    plot_name='custom_trial_columns',
    plot_title='Custom trial feature plot',
    session_data_home=Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/'),
    sess_id_full='CT014_2025-12-16_153200',
):
    """
    Plot a user-defined set of numeric columns from augmented_trial_df on one axis,
    with action/reward overlays and task-state bin shading.
    """
    if not value_columns:
        raise ValueError("value_columns must contain at least one column name.")

    processed_data_path = session_data_home / 'processed'
    figure_path = session_data_home / 'figures'

    augmented_trial_df_path = processed_data_path / (sess_id_full + '_augmented_trials.csv')
    trial_df = pd.read_csv(augmented_trial_df_path, sep=',', na_filter=False)

    required_cols = list(value_columns) + ['action', 'reward', 'block_type']
    missing_cols = [col for col in required_cols if col not in trial_df.columns]
    if missing_cols:
        raise ValueError(f"augmented_trial_df missing required columns: {missing_cols}")

    x = np.arange(trial_df.shape[0])
    actions = pd.to_numeric(trial_df['action'], errors='coerce').to_numpy()
    rewards = pd.to_numeric(trial_df['reward'], errors='coerce').to_numpy()
    states = trial_df['block_type'].to_numpy()

    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size - 1])))
    bin_types = states[bins]

    f, ax = plt.subplots(figsize=(12, 5))
    for col in value_columns:
        y = pd.to_numeric(trial_df[col], errors='coerce')
        ax.plot(x, y, label=col, linewidth=1.5)
    ax.axhline(0, color='gray', linewidth=1, alpha=0.6, linestyle='--')

    valid_action_ix = ~np.isnan(actions)
    action_trial_ix = x[valid_action_ix]
    action_signed = actions[valid_action_ix] * 2 - 1
    ax.scatter(action_trial_ix, action_signed, s=4, c='ivory', label='actions')

    rewarded_ix = valid_action_ix & (rewards > 0)
    rewarded_trial_ix = x[rewarded_ix]
    rewarded_actions = actions[rewarded_ix] * 2 - 1
    ax.scatter(rewarded_trial_ix, rewarded_actions, s=10, c='white', label='rewards')

    def _state_color_label(state):
        if state in color_dict:
            return color_dict[state], state_dict.get(state, str(state))
        state_str = str(state).lower()
        if state_str in color_dict:
            return color_dict[state_str], state_dict.get(state_str, state_str)
        try:
            state_int = int(state)
            if state_int in color_dict:
                return color_dict[state_int], state_dict.get(state_int, str(state_int))
        except (TypeError, ValueError):
            pass
        return 'gray', str(state)

    unique_bins = []
    for i in range(bins.size - 1):
        color, label = _state_color_label(bin_types[i])
        if label not in unique_bins:
            ax.axvspan(bins[i], bins[i + 1], color=color, alpha=.15, label=label)
            unique_bins.append(label)
        else:
            ax.axvspan(bins[i], bins[i + 1], color=color, alpha=.15)

    ax.set_xlabel('Trial')
    ax.set_ylabel('Value / action')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Right', '0', 'Left'])
    ax.set_xticks(bins)
    ax.set_title(f"{sess_id_full} {plot_title}")
    ax.legend(fancybox=False)
    f.tight_layout()

    figure_format = 'png'
    output_name = sess_id_full + '_' + plot_name
    plt.savefig(figure_path / '{}.{}'.format(output_name, figure_format), format=figure_format, dpi=300)
    plt.close()
    print('saved figure as {}'.format(output_name))


def plot_mouse_history_ideal_observer_values(
    trial_df: pd.DataFrame,
    figure_path: Path,
    sess_id_full: str,
    params: ideal_observer.IdealObserverParams | None = None,
    theme: str = "light",
    value_columns: list[str] | None = None,
    show_action_probability: bool = False,
    show_legend: bool = False,
    xtick_interval: int | None = 50,
) -> tuple[pd.DataFrame, Path]:
    """Plot observer values computed from the mouse's actual trial history.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        those accepted by
        `ideal_observer.compute_mouse_history_observer_run`: `state`,
        `action`, `reward`, `p_active_rew`, and `p_switch`. Choice convention
        is `0` right and `1` left; rewards are in task reward units.
    figure_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_id_full : str
        Full session identifier used in the plot title and filename.
    params : ideal_observer.IdealObserverParams or None
        Observer parameters. Defaults match session-level ideal-observer
        summaries.
    theme : str, default="light"
        Plot theme forwarded to `plot_run_dataframe`.
    value_columns : list[str] or None
        Observer value columns to plot. Defaults to `agent_relative_value`
        only, while allowing HMM and doubt components to be requested manually.
    show_action_probability : bool, default=False
        If True, include pre-update P(left) scaled to the action axis.
    show_legend : bool, default=False
        If True, draw a legend.
    xtick_interval : int or None, default=50
        Tick interval in trials. If None, use context-change ticks.

    Returns
    -------
    tuple[pd.DataFrame, pathlib.Path]
        The observer run dataframe with shape `(n_valid_trials, n_columns)` and
        the saved PNG path.
    """
    observer_run_df = ideal_observer.compute_mouse_history_observer_run(
        trial_df,
        params=params,
    )
    if value_columns is None:
        value_columns = ["agent_relative_value"]
    save_path = figure_path / f"{sess_id_full}_mouse_history_ideal_observer_values.png"
    fig, _ = plot_run_dataframe(
        observer_run_df,
        title=f"{sess_id_full} mouse-history ideal observer",
        value_columns=value_columns,
        show_action_probability=show_action_probability,
        show_legend=show_legend,
        xtick_interval=xtick_interval,
        save_path=save_path,
        show=False,
        theme=theme,
    )
    if hasattr(fig, "axes"):
        _add_mouse_history_colorblocks(fig.axes, observer_run_df)
    if hasattr(fig, "savefig"):
        fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print("saved figure as {}".format(save_path))
    return observer_run_df, save_path


def plot_mouse_history_ideal_observer_value_on_ax(
    ax: plt.Axes,
    trial_df: pd.DataFrame,
    params: ideal_observer.IdealObserverParams | None = None,
    value_column: str = "agent_relative_value",
    show_choices: bool = True,
    show_rewards: bool = True,
    line_width: float = 1.0,
    choice_marker_size: float = 5.0,
    reward_marker_size: float = 12.0,
    axis_label_size: float = 8,
    tick_label_size: float = 7,
) -> pd.DataFrame:
    """Plot one mouse-history ideal-observer value trace on an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the value trace.
    trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`. Required columns are
        those accepted by
        `ideal_observer.compute_mouse_history_observer_run`: `state`,
        `action`, `reward`, `p_active_rew`, and `p_switch`. Choice convention
        is `0` right and `1` left; rewards are in task reward units.
    params : ideal_observer.IdealObserverParams or None, default=None
        Observer parameters. None uses the observer helper defaults.
    value_column : str, default="agent_relative_value"
        Column from the observer run dataframe to plot. Values are plotted on
        the signed action/value axis where negative favors right and positive
        favors left.
    show_choices : bool, default=True
        Whether to overlay small mouse choice markers. Actions use the signed
        convention right=-1 and left=+1.
    show_rewards : bool, default=True
        Whether to overlay larger rewarded-choice markers. Requires `reward`
        and `action` columns in the observer run dataframe.
    line_width : float, default=1.0
        Width of the observer value line.
    choice_marker_size : float, default=5.0
        Marker area in points squared for all valid choices.
    reward_marker_size : float, default=12.0
        Marker area in points squared for rewarded choices.
    axis_label_size : float, default=8
        Font size for axis labels and the title.
    tick_label_size : float, default=7
        Font size for x/y tick labels.

    Returns
    -------
    pd.DataFrame
        Observer run dataframe with shape `(n_valid_trials, n_columns)`.
    """
    observer_run_df = ideal_observer.compute_mouse_history_observer_run(
        trial_df,
        params=params,
    )
    if value_column not in observer_run_df.columns:
        raise ValueError(f"observer_run_df is missing requested value column: {value_column}")

    plot_df = observer_run_df.copy()
    plot_df[value_column] = pd.to_numeric(plot_df[value_column], errors="coerce")
    x = np.arange(plot_df.shape[0])
    ax.plot(x, plot_df[value_column].to_numpy(dtype=float), color="black", linewidth=line_width)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6, linestyle="--")

    if show_choices and "action" in plot_df.columns:
        actions = pd.to_numeric(plot_df["action"], errors="coerce")
        valid_action_ix = actions.notna().to_numpy()
        if valid_action_ix.any():
            signed_actions = actions.to_numpy(dtype=float) * 2 - 1
            ax.scatter(
                x[valid_action_ix],
                signed_actions[valid_action_ix],
                s=choice_marker_size,
                c="0.25",
                alpha=0.55,
                linewidths=0,
                label="choices",
                zorder=3,
            )

            if show_rewards and "reward" in plot_df.columns:
                rewards = pd.to_numeric(plot_df["reward"], errors="coerce").fillna(0).to_numpy(dtype=float)
                rewarded_ix = valid_action_ix & (rewards > 0)
                if rewarded_ix.any():
                    ax.scatter(
                        x[rewarded_ix],
                        signed_actions[rewarded_ix],
                        s=reward_marker_size,
                        facecolors="white",
                        edgecolors="black",
                        linewidths=0.5,
                        label="rewards",
                        zorder=4,
                    )

    if "state" in plot_df.columns and plot_df.shape[0] > 0:
        states = plot_df["state"].to_numpy()
        change_ix = states[:-1] != states[1:]
        state_changes = np.arange(1, states.size)[change_ix]
        bins = np.unique(np.concatenate(([0], state_changes, [states.size - 1])))
        bin_types = states[bins]
        unique_bins = []
        for i in range(bins.size - 1):
            state = bin_types[i]
            if state in color_dict:
                label = state_dict.get(state, str(state))
                if label not in unique_bins:
                    ax.axvspan(bins[i], bins[i + 1], color=color_dict[state], alpha=.15, label=label)
                    unique_bins.append(label)
                else:
                    ax.axvspan(bins[i], bins[i + 1], color=color_dict[state], alpha=.15)

    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["Right", "0", "Left"])
    ax.set_xlabel("Trial", fontsize=axis_label_size)
    ax.set_ylabel(value_column, fontsize=axis_label_size)
    ax.set_title("Mouse-history observer", fontsize=axis_label_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)
    return observer_run_df


def main():
    plot_combined_trial_feature_values()
    plot_combined_trial_index_features()
    plot_user_defined_trial_columns(['HMM_rel_value_logodds_decay',
                                     'FQlearning_rel_value'])
                                     # 'relative_doubt_index',
                                     # 'perseveration_regressor'],)


if __name__ == '__main__':
    main()
    # rnn_main()
