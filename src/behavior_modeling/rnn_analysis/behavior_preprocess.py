from src.behavior_modeling import rnn_controller as rc
from src.behavior_modeling.fileIO import rnn_io
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import src.behavior_analysis.context_switch_analysis as vdp
import pandas as pd
import scipy as sp
import src.neural_similarity.inspect_abstraction_measures as iam
import src.neural_similarity.similarity_measures as sm
from sklearn import decomposition, svm
from icecream import ic
from typing import Tuple

data_dir = Path('../../../data/processed/rnn_experiments')
model_dir = Path('../../../saved_models/RNN')
plot_path = Path('../../../reports/figures/model_behavior')

states = ['right', 'left']
state_dict = {s: i for i, s in enumerate(states)}  # i.e. [0 right, 1 left]


def preprocess_rnn_trials(experiment: dict, sess_id: str):
    # agent_name = 'RNN_reinforce'
    # date = '2024-04-11'
    # p_reward = .9
    # p_switch = .1
    # fixed_blocks = 10
    #
    # # load_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name, p_reward, p_switch)
    # load_name = '{}_pReward_{}_fixedBlocks_{}'.format(agent_name, p_reward, fixed_blocks)
    # session_name = data_dir / date / (load_name + '_eval_dict.pkl')
    # # session_name = data_dir / date / (load_name + '_dict.pkl')
    # experiment = rnn_io.load_experiment(session_name)
    performance = experiment['performance']

    inputs = np.stack(experiment['rnn_dict']['inputs'], axis=0)
    Rstimulus = inputs[:, 2] > 0
    Lstimulus = inputs[:, 3] > 0
    allstimulus = Rstimulus | Lstimulus

    go_cue = inputs[:, 1] > 0
    go_cue = np.nonzero(go_cue)[0]
    go_cue = go_cue[go_cue > 0]  # only keep timesteps with activity before the first cue

    activity = np.squeeze(experiment['rnn_dict']['hidden_states'])
    go_cue_activity = activity[go_cue]

    # 'state', 'state_int', 'cur_trial', 'cur_trial_in_block', 'cur_block',
    # 'action', 'correct', 'reward', 'p_active_rew', 'p_inactive_rew',
    # 'stimulus', 'session_ID'

    # behavior dict at the time of each action/when the go cue is active
    behavior_dict = {
        'state': np.array(performance['states'])[go_cue],
        'state_int': np.array([state_dict[s] for s in performance['states']])[go_cue],
        'cur_trial': np.array(performance['cur_trial'])[go_cue],
        'cur_trial_in_block': np.array(performance['cur_trial_in_block'])[go_cue],
        'cur_block': np.array(performance['cur_block'])[go_cue],
        'action': np.array(performance['action'])[go_cue],
        'correct': np.array(performance['correct'])[go_cue],
        'reward': np.array(performance['reward'])[go_cue],
        'p_active_rew': np.array(performance['p_active_reward'])[go_cue],
        'p_inactive_rew': np.array(performance['p_inactive_reward'])[go_cue],
        'stimulus': allstimulus[go_cue],
        'session_ID': [sess_id] * go_cue.shape[0]
    }
    behavior_df = pd.DataFrame(behavior_dict)
    # behavior_dict['activity'] = go_cue_activity

    print("Performance: {}% correct".format(np.mean(behavior_df['correct'])))
    return behavior_df, go_cue_activity


def regress_behavior(df: pd.DataFrame, save_name: str):
    # summary plot
    # reg.plot_multiple_session_regression(df_list=[df], df_labels=['RNN'], key='trials_to_correct',
    #                                      save_name=save_name, plot_path=plot_path)

    # compare cued to uncued switches
    _, state_switch_df = vdp.get_multisession_switches(df)
    stim_active = df['stimulus'][state_switch_df['state_change_ix']]

    # run the cued switches
    cued_switch_df = state_switch_df[stim_active.to_numpy()]
    uncued_switch_df = state_switch_df[~stim_active.to_numpy()]

    rewards_cued, mean_cued, std_cued, sem_cued = vdp.consecutive_summary_measures(cued_switch_df, key='trials_to_correct', min_counts=0)
    rewards_uncued, mean_uncued, std_uncued, sem_uncued = vdp.consecutive_summary_measures(uncued_switch_df, key='trials_to_correct', min_counts=0)

    f, ax = plt.subplots()
    plt.plot(mean_cued.index, mean_cued, label='cued')
    plt.fill_between(mean_cued.index, mean_cued - sem_cued,
                     mean_cued + sem_cued, alpha=.3, linewidth=0)

    plt.plot(mean_uncued.index, mean_uncued, label='uncued')
    plt.fill_between(mean_uncued.index, mean_uncued - sem_uncued,
                     mean_uncued + sem_uncued, alpha=.3, linewidth=0)

    plt.xlabel('Consecutive rewards', fontsize=18)
    plt.ylabel("Trials to Switch", fontsize=16)
    # ax.tick_params(axis='y', which='major', labelsize=12)
    # ax.yticks(fontsize=18)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(frameon=False, fontsize=14)
    # plt.title('Trials to Switch vs Rewards'.format(key))
    max_ix = 10
    # max_ix = None
    if max_ix:
        plt.xticks(np.arange(1, max_ix + 1, 1))
        plt.xlim([1, max_ix])

    # plt.ylim([0, 5])

    plt.tight_layout()
    save_path = plot_path / '{}_trials-to-switch.png'.format(save_name)
    f.savefig(save_path, format='png', dpi=300)
    print('Saved as {}'.format(save_path))
    # save_path = plot_path / '{}_trials-to-switch.svg'.format(save_name)
    # f.savefig(save_path, format='svg')
    # print('Saved as {}'.format(save_path))


def separate_conditions(behavior_df: pd.DataFrame, neural_data: pd.DataFrame) -> pd.DataFrame:
    """
        need to sort each trial into a block type/condition
        then gather the activity for each block type into a df
        """

    block_ids = behavior_df['cur_block'].unique()
    Rcued, Runcued, Lcued, Luncued = [], [], [], []
    for id in block_ids:
        block_df = behavior_df[behavior_df['cur_block'] == id]
        block_stimulus = block_df['stimulus'].to_numpy()[0]
        block_state = block_df['state'].to_numpy()[0]
        if block_stimulus and block_state == 'right':  # right cued
            Rcued.append(neural_data[block_df.index])
        elif not block_stimulus and block_state == 'right':  # right uncued
            Runcued.append(neural_data[block_df.index])
        elif block_stimulus and block_state == 'left':  # left cued
            Lcued.append(neural_data[block_df.index])
        elif not block_stimulus and block_state == 'left':  # left uncued
            Luncued.append(neural_data[block_df.index])
        else:
            raise ValueError("Something went wrong")

        # ic(block_activity)

    conditions = ["Rcued", "Runcued", "Lcued", "Luncued"]
    condition_data = [Rcued, Runcued, Lcued, Luncued]
    condition_data = [np.concatenate(d, axis=0) for d in condition_data]
    df_list = [pd.DataFrame(d) for d in condition_data]
    [df.insert(0, "condition", [cond] * df.shape[0]) for df, cond in zip(df_list, conditions)]
    data = pd.concat(df_list, ignore_index=True)
    data_zscored = sp.stats.zscore(data.drop(columns="condition"), axis=0)
    data_zscored = pd.DataFrame(data_zscored, columns=data.columns[1:])
    data_zscored.insert(0, "condition", data["condition"])
    return data_zscored


def pca_analysis(data: pd.DataFrame, n_components=3, centers=None):
    # PCA. Data should already be normalized!!!
    # data = data.fillna(0)
    data_np = data.drop(columns="condition")
    conditions = data["condition"]

    pca = decomposition.PCA(n_components=n_components)
    pca.fit(data.drop(columns="condition"))
    proj = pca.transform(data.drop(columns="condition"))

    centers = iam.get_cluster_centers(data)
    center_proj = pca.transform(centers.drop(columns="condition"))

    f, ax = plt.subplots()
    if n_components == 2:
        pc_df = pd.DataFrame(proj, columns=["PC1", "PC2"])
        pc_df["condition"] = conditions
        sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="condition")

    elif n_components == 3:
        ax = plt.axes(projection='3d')
        Rcued = proj[data['condition'] == 'Rcued']
        Runcued = proj[data['condition'] == 'Runcued']
        Lcued = proj[data['condition'] == 'Lcued']
        Lunccued = proj[data['condition'] == 'Luncued']

        ax.scatter(Rcued[:, 0], Rcued[:, 1], Rcued[:, 2], label="Rcued")
        ax.scatter(Runcued[:, 0], Runcued[:, 1], Runcued[:, 2], label="Runcued")
        ax.scatter(Lcued[:, 0], Lcued[:, 1], Lcued[:, 2], label="Lcued")
        ax.scatter(Lunccued[:, 0], Lunccued[:, 1], Lunccued[:, 2], label="Luncued")
        plt.legend()

    f, ax = plt.subplots()
    ax = plt.axes(projection='3d')
    ax.scatter(center_proj[0, 0], center_proj[0, 1], center_proj[0, 2], label="Rcued", s=20)
    ax.scatter(center_proj[1, 0], center_proj[1, 1], center_proj[1, 2], label="Runcued", s=20)
    ax.scatter(center_proj[2, 0], center_proj[2, 1], center_proj[2, 2], label="Lcued", s=20)
    ax.scatter(center_proj[3, 0], center_proj[3, 1], center_proj[3, 2], label="Luncued", s=20)
    plt.legend(fancybox=False, frameon=False)

    plt.show()
    return pca


def separate_correct_trials(behavior_df: pd.DataFrame, neural_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    correct = behavior_df['correct']
    correct_df = behavior_df[correct]
    incorrect_df = behavior_df[~correct]
    correct_data = neural_data[correct]
    incorrect_data = neural_data[~correct]
    return correct_df, incorrect_df, correct_data, incorrect_data


def process_activity(data_zscored: pd.DataFrame, ccgp_mode='centers'):
    """
    need to sort each trial into a block type/condition
    then gather the activity for each block type into a df
    """
    assert ccgp_mode in ['centers', 'resampled']
    # test shattering_dimensionality
    dichotomy_names = ["context", "stimulus"]
    dichotomies = ((('Rcued', 'Runcued'), ('Lcued', 'Luncued')),
                   (('Rcued', 'Lcued'), ('Runcued', 'Luncued')))
    SD, sd_scores = sm.shattering_dimensionality(data=data_zscored, dichotomies=dichotomies)

    # test CCGP
    # context_split = [['R1', 'R2', 'R3'], ['L1', 'L2', 'L3']]
    # create dichotomies for context and stimulus
    context_split = [['Rcued', 'Runcued'], ['Lcued', 'Luncued']]
    stimulus_split = [['Rcued', 'Lcued'], ['Runcued', 'Luncued']]
    context_train, context_test = iam.condition_dichotomies(context_split[0], context_split[1])
    stimulus_train, stimulus_test = iam.condition_dichotomies(stimulus_split[0], stimulus_split[1])

    # ic(context_train, context_test)

    if ccgp_mode == 'centers':
        centers = iam.get_cluster_centers(data_zscored)

        ic('processing context CCGP')
        contextA_train, contextB_train = iam.split_conditions(context_train)
        contextA_test, contextB_test = iam.split_conditions(context_test)
        groups = dict(trainA=contextA_train, trainB=contextB_train, testA=contextA_test, testB=contextB_test)
        ccgp_centers, scores_centers = sm.CCGP(data=centers, dichotomy_groups=groups)

        ic('processing stimulus CCGP')
        stimulusA_train, stimulusB_train = iam.split_conditions(stimulus_train)
        stimulusA_test, stimulusB_test = iam.split_conditions(stimulus_test)
        groups = dict(trainA=stimulusA_train, trainB=stimulus_train, testA=stimulusA_test, testB=stimulusB_test)
        ccgp_centers, ccgp_scores_centers = sm.CCGP(data=centers, dichotomy_groups=groups)

    elif ccgp_mode == 'resampled':
        resampled = iam.resample_clusters(data_zscored, n_samples=10000)
        ic('processing context CCGP')
        contextA_train, contextB_train = iam.split_conditions(context_train)
        contextA_test, contextB_test = iam.split_conditions(context_test)
        groups = dict(trainA=contextA_train, trainB=contextB_train, testA=contextA_test, testB=contextB_test)
        ccgp_resampled, scores_resampled = sm.CCGP(data=resampled, dichotomy_groups=groups)

        ic('processing stimulus CCGP')
        stimulusA_train, stimulusB_train = iam.split_conditions(stimulus_train)
        stimulusA_test, stimulusB_test = iam.split_conditions(stimulus_test)
        groups = dict(trainA=stimulusA_train, trainB=stimulus_train, testA=stimulusA_test, testB=stimulusB_test)
        ccgp_resampled, ccgp_scores_resampled = sm.CCGP(data=resampled, dichotomy_groups=groups)

    # ic('processing context CCGP')
    # contextA_train, contextB_train = iam.split_conditions(context_train)
    # contextA_test, contextB_test = iam.split_conditions(context_test)
    # groups = dict(trainA=contextA_train, trainB=contextB_train, testA=contextA_test, testB=contextB_test)
    # # sm.CCGP(data=data_zscored, dichotomy_groups=groups)
    # ccgp_centers, scores_centers = sm.CCGP(data=centers, dichotomy_groups=groups)
    # # ccgp_resampled, scores_resampled = sm.CCGP(data=resampled, dichotomy_groups=groups)
    #
    # ic('processing stimulus CCGP')
    # stimulusA_train, stimulusB_train = iam.split_conditions(stimulus_train)
    # stimulusA_test, stimulusB_test = iam.split_conditions(stimulus_test)
    # groups = dict(trainA=stimulusA_train, trainB=stimulus_train, testA=stimulusA_test, testB=stimulusB_test)
    # # sm.CCGP(data=data_zscored, dichotomy_groups=groups)
    # ccgp_centers, scores_centers = sm.CCGP(data=centers, dichotomy_groups=groups)
    # # ccgp_resampled, scores_resampled = sm.CCGP(data=resampled, dichotomy_groups=groups)


def load_data():
    agent_name = 'RNN_reinforce'
    date = '2024-04-29'
    p_reward = .5
    p_switch = .1
    fixed_blocks = 10
    note = 'overtrain3'

    load_name = '{}_pReward_{}_pSwitch_{}'.format(agent_name, p_reward, p_switch)
    # load_name = '{}_pReward_{}_fixedBlocks_{}'.format(agent_name, p_reward, fixed_blocks)
    if note:
        load_name += '_{}'.format(note)

    session_name = data_dir / date / (load_name + '_eval_dict.pkl')
    # session_name = data_dir / date / (load_name + '_dict.pkl')
    experiment = rnn_io.load_experiment(session_name)
    return experiment, load_name


def main():
    if not plot_path.exists():
        plot_path.mkdir(parents=True)

    experiment, sess_id = load_data()
    behavior_df, activity = preprocess_rnn_trials(experiment, sess_id)

    regress_behavior(behavior_df, save_name=behavior_df['session_ID'].unique()[0])

    # zscored_data = separate_conditions(behavior_df, activity)
    # correct_behavior, incorrect_behavior, correct_data, incorrect_data = separate_correct_trials(behavior_df, zscored_data)

    # process_activity(correct_data, ccgp_mode='centers')
    # pca = pca_analysis(correct_data, n_components=3)


if __name__ == '__main__':
    main()
