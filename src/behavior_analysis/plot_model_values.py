import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional
from pathlib import Path
import pickle as pkl
import re

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
state_dict = {0: 'right', 1: 'left'}


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
        plt.scatter(x, action_ix, s=4, label='actions', c='ivory')
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
    Values and pLeft should match the length of the dataframe trials. How should I handle skip/give_reward trials?
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
    plt.xticks(bins)
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


def multiplot_main():
    # load the mouse
    mouse = 'MF03'
    date = '2023-10-03'
    sess_ID = mouse + '-' + date
    p = Path('../mouse_behavior') / (sess_ID + '_performance.pkl')
    with p.open('rb') as f:
        mouse_df = pkl.load(f)

    alpha = .1
    temp = .2
    decay = .6
    agent_type = 'F-Qlearning'
    agent_param_str = '{}_alpha={}_temp={}_decay={}'.format(agent_type, alpha, temp, decay)
    fname = sess_ID + '_' + agent_param_str
    extras_str = 'comparison-model'
    if extras_str:
        fname += '_' + extras_str

    p = Path('../saved_models') / (fname + '.pkl')
    _, _, QL_df, _ = controller.load_experiment(p)


    alpha = 0
    temp = .2
    HMM_transition_prob = .1
    agent_type = 'HMM'
    agent_param_str = '{}_alpha={}_temp={}_model-Psw={}'.format(agent_type, alpha, temp, HMM_transition_prob)
    fname = sess_ID + '_' + agent_param_str
    extras_str = 'comparison-model'
    if extras_str:
        fname += '_' + extras_str

    p = Path('../saved_models') / (fname + '.pkl')
    _, _, HMM_df, _ = controller.load_experiment(p)

    alpha = .1
    beta = 1.5
    tau = 1
    agent_type = 'Logistic'  # RFLR, reduced-form-linear-regression
    agent_param_str = '{}_alpha={}_beta={}_tau={}'.format(agent_type, alpha, beta, tau)
    fname = sess_ID + '_' + agent_param_str
    extras_str = 'comparison-model'
    if extras_str:
        fname += '_' + extras_str

    p = Path('../saved_models') / (fname + '.pkl')
    _, _, LR_df, _ = controller.load_experiment(p)

    action_df = mouse_df
    value_dfs = [(QL_df, 'Q-learning'), (HMM_df, 'HMM'), (LR_df, 'Logistic')]

    # plot_multiple_runs(value_dfs, plot_name='mouse_comparison', action_df=action_df)


if __name__ == '__main__':
    plot_one_model()
    # main()
    # rnn_main()
    # multiplot_main()

