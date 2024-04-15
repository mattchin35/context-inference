import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import controller
from typing import List, Optional
from pathlib import Path
import pickle as pkl
import demo_tasks

plt.style.use('dark_background')


"""
Plot one or more model runs. 
Make functions to plot 1 run, and to show multiple runs on the same plot. Want to compare Q-FQ-stickyHMM-pureHMM
"""

RIGHT_IX = 0
LEFT_IX = 1
action_int_to_string = {0: 'Right', 1: 'Left'}
color_dict = {'right_cued': 'cyan', 'left_cued': 'darkgreen', 'right_uncued': 'plum', 'left_uncued': 'indigo',
              'right': 'cyan', 'left': 'darkgreen',
              0: 'cyan', 1: 'darkkhaki'}
state_dict = {0: 'right', 1: 'left'}


def moving_average(a: np.ndarray, window_size: int = 3):
    df = pd.DataFrame(a).rolling(window_size).mean()
    return df.values


def plot_multiple_runs(value_dfs: List[pd.DataFrame], plot_name: str, action_df: Optional[pd.DataFrame], ):
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

    f, ax = plt.subplots(figsize=(8, 4))

    # plot model relative value and/or P(left)
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


def plot_experiment(df: pd.DataFrame, model_name: str, plot_name: str):
    df = df[:100]
    states = df['state'].values
    actions = df['action'].values
    action0_dist = df['action_dist'].values
    rewards = df['reward'].values

    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size-1])))
    bin_types = states[bins]

    action_str = [action_int_to_string[a] for a in actions]
    x = np.arange(df['action'].size)

    ### PLOT MODEL ACTIONS between -1 and 1
    f0, ax0 = plt.subplots()
    # sns.swarmplot(plot_df, x='trial', y='action', hue='model_type',s=4)  # no dodge plotting
    # sns.swarmplot(plot_df, x='trial', y='action', hue='model_type', s=4, dodge=True)
    # plt.plot(x[window:], avg_action[window:], 'k', label='average action')
    # plt.plot(x, (df['relative_value'] + 1)/2, 'k', label='relative value')

    action_ix = actions * 2 - 1
    markersize = 6
    plt.scatter(x, action_ix, s=2, label='actions', c='ivory')
    # plt.scatter(x, (1 - action0_dist) * 2 - 1, s=4, label='P(left)')
    plt.plot(x, (1 - action0_dist) * 2 - 1, label='P(left)', c='coral', linestyle='--')

    rew_ix = rewards > 0
    plt.scatter(x[rew_ix], (rewards * action_ix)[rew_ix], s=6, c='ivory') #, label='rewards')

    if 'relative_value' in df.keys():
        plt.plot(x, df['relative_value'], 'w', label='relative value')
    plt.plot(x, np.zeros_like(x), c='k', alpha=.3, linestyle=(0, (1, 1)))

    # FIGURE ADJUSTMENTS
    plt.yticks([-1,1],['Right', 'Left'])
    # plt.ylim([-.1,1.1])
    plt.xticks(bins)
    plt.xlim([0, x[-1]])
    # task_str = 'pReward_{}_pSwitch_{}'.format(0,0)
    # model_str = 'pReward_{}_pSwitch_{}'.format(0,0)
    plt.title('{}'.format(model_name), fontsize=14)

    # axis labels are provided by seaborn/pandas, but you can adjust them anyways to your liking
    plt.ylabel('Actions', fontsize=12)
    plt.xlabel('Trial', fontsize=12)
    ax = plt.gca()

    ### COLOR BLOCKS ###
    unique_bins = np.unique(bin_types).tolist()
    for i in range(bins.size-1):
        if bin_types[i] in unique_bins:
            plt.axvspan(bins[i], bins[i+1], color=color_dict[bin_types[i]], alpha=.2, label=state_dict[bin_types[i]])
            unique_bins.remove(bin_types[i])
        else:
            plt.axvspan(bins[i], bins[i+1], color=color_dict[bin_types[i]], alpha=.2)

    # ax.legend(fancybox=False)  # , loc='lower right')
    figure_format = 'png'
    f0.tight_layout()
    # plt.show()
    plt.savefig('../figures/{}.{}'.format(plot_name, figure_format), format=figure_format, dpi=300)
    plt.close()
    print('saved figure as {}'.format(plot_name))


def plot_prior(df: pd.DataFrame):
    action_dist = df['action_dist'].values
    value_normalized = (df['relative_value'].values + 1) / 2
    try:
        prior = np.stack(df['prior'].values)
        prior_label = 'probability of state 1'

    except:
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


def main():
    p_reward = .9
    p_switch = .1
    alpha = .1
    beta = 1.5
    tau = 1
    temp = .1
    decay = .5
    HMM_transition_prob = .1

    agent_type = 'F-Qlearning'
    agent_param_str = 'alpha={}_temp={}_decay={}'.format(alpha, temp, decay)
    model_name = 'Q-learning'
    plot_name = 'QL_plot_comparison'

    # agent_type = 'HMM'
    # agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(alpha, temp, HMM_transition_prob)
    # model_name = 'Greedy HMM'
    # plot_name = 'HMM_plot_greedy'

    # agent_type = 'HMM_RFLR'
    # agent_param_str = 'alpha={}_temp={}_model-Psw={}'.format(alpha, temp, HMM_transition_prob)
    # model_name = 'HMM RFLR'
    # plot_name = 'HMM-RFLR_plot'

    # agent_type = 'Logistic'  # RFLR, reduced-form-linear-regression
    # agent_param_str = 'alpha={}_beta={}_tau={}'.format(alpha, beta, tau)
    # model_name = 'Logistic'
    # plot_name = 'RFLR_plot'

    # mouse = 'MF03'
    # date = '2023-10-03'
    # sess_ID = mouse + '-' + date
    # fname = sess_ID + '_{}_{}'.format(agent_type, agent_param_str)
    # plot_name = '{}_plot'.format(fname)

    sess_name = '{}_pReward_{}_pSwitch_{:.2f}'.format(agent_type, p_reward, p_switch)
    fname = sess_name + '_' + agent_param_str
    extras_str = 'comparison-model'
    if extras_str:
        fname += '_' + extras_str

    p = Path('../saved_models') / (fname + '.pkl')
    task, agent, performance_df, params = controller.load_experiment(p)
    plot_experiment(performance_df, model_name=model_name, plot_name=plot_name)

    # df = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch, rewards_before_switch=6)
    # df = demo_tasks.load_collected_runs(agent_type=agent_type, p_reward=p_reward, p_switch=p_switch)
    # df = df[df['session_ID'] == 1]

    # plot_experiment(df, model_name=model_name, plot_name=plot_name)
    # plot_prior(df)


def rnn_main():
    p_reward = .9
    p_switch = .1

    agent_type = 'RNN_policy_gradient'
    # sess_name = 'policy-RNN-eval'
    sess_name = '{}_pReward_{}_pSwitch_{}_eval'.format(agent_type, p_reward, p_switch)
    plot_name = '{}_plot'.format(sess_name)

    p = Path('../saved_models') / (sess_name + '.pkl')
    with p.open('rb') as f:
        data_dict = pkl.load(f)

    performance = data_dict['performance']
    name = '{}_pReward_{}_pSwitch_{}'.format(agent_type, p_reward, p_switch)
    plot_experiment(performance, sess_name, plot_name)


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

    plot_multiple_runs(value_dfs, plot_name='mouse_comparison', action_df=action_df)






if __name__ == '__main__':
    # main()
    # rnn_main()
    multiplot_main()

