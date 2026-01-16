"""
Blockwise LM-HMM using the code from Cazettes et al 2023 made by Lucca Mazzucato
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import re
from pathlib import Path
import ssm # note this should be the forked ssm repo above
from ssm.util import find_permutation
from ssm.plots import gradient_cmap, white_to_color_cmap
import src.state_space_modeling.utilplot as utilplot

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange"
    ]
colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

np.random.seed(0)

def mle_block_states(block_df: pd.DataFrame, figure_path: Path, sess_id: str, plot: bool=False):
    """Maximum likelihood estimation of block strategies/states."""
    ix_valid = (block_df['trials_to_correct'] != 'None') & (block_df['prev_n_correct'] != 'None')
    df = block_df[ix_valid]

    # consecutive_rewards = df['prev_consecutive_rewards'].to_numpy().reshape(-1, 1).astype(int)
    # prev_correct = df['prev_n_correct'].to_numpy().reshape(-1, 1).astype(int)
    prev_rewards = df['prev_n_rewarded'].to_numpy().reshape(-1, 1).astype(int)
    n_switches = df['n_switches'].to_numpy().reshape(-1, 1).astype(int)
    bias_flag = df['bias_full_flag'].to_numpy().reshape(-1, 1) == 'True'
    trials_to_correct = df['trials_to_correct'].to_numpy().reshape(-1, 1).astype(int)

    # predictors = prev_rewards
    predictors = np.concatenate([prev_rewards, n_switches, bias_flag], axis=1)
    # predictors = np.concatenate([prev_rewards, bias_flag], axis=1)
    # predictors = np.concatenate([prev_rewards, n_switches], axis=1)

    # num states - start with 2, Inf/RL, then do 3 (inf/rl/biased). Would like Inf(maybe lo and hi thresh)/RL/biased/disengaged/confused. I think this is
    # what cross-validation is gonna be for. less is better!
    num_states = 3
    obs_dim = 1
    input_dim = predictors.shape[1]

    model_dict = {}
    mle_hmm = ssm.HMM(num_states, obs_dim, M=input_dim, observations="input_driven_obs_gaussian", transitions="standard")
    N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    # fit_ll = mle_hmm.fit(obs, inputs=inpt, method="em", num_iters=N_iters, tolerance=10**-6)
    # fit_log_likelihood = mle_hmm.fit(trials_to_correct, inputs=prev_rewards, method="em", num_iters=N_iters, tolerance=10 ** -6)
    fit_log_likelihood = mle_hmm.fit(trials_to_correct, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['fit_log_likelihood'] = fit_log_likelihood

    # Plot the log probabilities of the true and fit models. Fit model final LL should be greater
    # than or equal to true LL.
    if plot:
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_log_likelihood, label="EM")
        # plt.plot([0, len(fit_ll)], (true_ll) * np.ones(2), ':k', label="True")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        # plt.xlim(0, len(fit_ll))
        plt.ylabel("Log Probability")
        plt.title("MLE EM fit of observed data")
        save_path = figure_path / '{}_mle_convergence.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    # mle_hmm.permute(find_permutation(true_states, most_likely_states))
    # most_likely_states = mle_hmm.most_likely_states(trials_to_correct, input=prev_rewards)
    most_likely_states = mle_hmm.most_likely_states(trials_to_correct, input=predictors)
    recovered_weights = mle_hmm.observations.Wks
    recovered_mus = mle_hmm.observations.mus

    # revisit this after fitting MAP
    weight_dict = {}
    weight_dict[0] = {}
    weight_dict[0]['weights'] = recovered_weights
    weight_dict[0]['mus'] = recovered_mus
    weight_dict[0]['label'] = 'mle'
    model_dict['weight_dict'] = weight_dict
    if plot:
        utilplot.plot_weights_comparison(weight_dict)
        plt.title("recovered weights")
        save_path = figure_path / '{}_mle_weights.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

        mle_transition_mat = np.exp(mle_hmm.transitions.log_Ps)
        utilplot.plot_trans_matrix(mle_transition_mat)
        plt.title("MLE transition matrix", fontsize=15)
        save_path = figure_path / '{}_mle_transition_mat.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

        # plt.subplot(1, 2, 2)
        # plt.title("MAP transition matrix", fontsize=15)
        # utilplot.plot_trans_matrix(map_transition_mat)
        # f, ax = utilplot.plt.subplots_adjust(0, 0, 1, 1)

    ### Get expected states ###
    # posterior_probs = mle_hmm.expected_states(data=trials_to_correct, input=prev_rewards)[0]
    posterior_probs = mle_hmm.expected_states(data=trials_to_correct, input=predictors)[0]
    if plot:
        # fig, ax = utilplot.plot_postprob_obs(posterior_probs, trials_to_correct, prev_rewards, mle_hmm, colors, cmap)
        fig, ax = utilplot.plot_postprob_obs(posterior_probs, trials_to_correct, predictors, mle_hmm, colors, cmap)
        plt.title("MLE HMM states")
        save_path = figure_path / '{}_mle_predicted_states.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['posterior_probs'] = posterior_probs
    model_dict['inferred_state_list'] = inferred_state_list
    model_dict['inferred_durations'] = inferred_durations
    model_dict['hmm_z'] = most_likely_states

    inferred_states = np.zeros(block_df.shape[0], dtype='object')
    inferred_states[ix_valid] = most_likely_states
    inferred_states[~ix_valid] = 'None'
    block_df['inferred_strategy'] = inferred_states

    if plot:
        ## Rearrange the lists of durations to be a nested list where
        ## the nth inner list is a list of durations for state n
        inferred_durations_stacked = []
        for s in range(num_states):
            inferred_durations_stacked.append(inferred_durations[inferred_state_list == s])

        fig = plt.figure(figsize=(8, 4))
        plt.hist(inferred_durations_stacked, label=['state ' + str(s) for s in range(num_states)], color=colors[:num_states])
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Inferred State Durations')
        # plt.show()

    mle_savename = sess_id + '_mle_statedict.pkl'
    with open(mle_savename, 'wb') as file:
        pkl.dump(model_dict, file)

    return model_dict, block_df


def map_block_states(block_df: pd.DataFrame, figure_path: Path, sess_id: str, plot: bool=False,
                     prior_sigma=1, prior_alpha=1):
    """Maximum a priori estimation of block strategies/states. Using a prior helps with small-data problems."""
    ix_valid = (block_df['trials_to_correct'] != 'None') & (block_df['prev_n_correct'] != 'None')
    df = block_df[ix_valid]

    consecutive_rewards = df['prev_consecutive_rewards'].to_numpy().reshape(-1, 1).astype(int)
    prev_rewards = df['prev_n_rewarded'].to_numpy().reshape(-1, 1).astype(int)
    #prev_correct = df['prev_n_correct'].to_numpy().reshape(-1, 1).astype(int)
    trials_to_correct = df['trials_to_correct'].to_numpy().reshape(-1, 1).astype(int)

    # num states - start with 2, Inf/RL, then do 3 (inf/rl/biased). Would like Inf(maybe lo and hi thresh)/RL/biased/disengaged/confused. I think this is
    # what cross-validation is gonna be for. less is better!
    # prior_sigma = 1
    # prior_alpha = 1
    num_states = 2
    obs_dim = 1
    input_dim = 1

    model_dict = {}
    map_hmm = ssm.HMM(num_states, obs_dim, M=input_dim,
                      observations="input_driven_obs_gaussian",
                      observation_kwargs=dict(prior_sigma=prior_sigma),
                      transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0))

    N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    # fit_ll = mle_hmm.fit(obs, inputs=inpt, method="em", num_iters=N_iters, tolerance=10**-6)
    fit_log_likelihood = map_hmm.fit(trials_to_correct, inputs=prev_rewards, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['fit_log_likelihood'] = fit_log_likelihood

    # Plot the log probabilities of the true and fit models. Fit model final LL should be greater
    # than or equal to true LL.
    if plot:
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_log_likelihood, label="EM")
        # plt.plot([0, len(fit_ll)], (true_ll) * np.ones(2), ':k', label="True")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        # plt.xlim(0, len(fit_ll))
        plt.ylabel("Log Probability")
        plt.title("MAP EM fit of observed data")
        save_path = figure_path / '{}_map_convergence.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = map_hmm.most_likely_states(trials_to_correct, input=prev_rewards)
    recovered_weights = map_hmm.observations.Wks
    recovered_mus = map_hmm.observations.mus

    # revisit this after fitting MAP
    weight_dict = {}
    weight_dict[0] = {}
    weight_dict[0]['weights'] = recovered_weights
    weight_dict[0]['mus'] = recovered_mus
    weight_dict[0]['label'] = 'mle'
    model_dict['weight_dict'] = weight_dict
    if plot:
        utilplot.plot_weights_comparison(weight_dict)
        plt.title("recovered weights")
        save_path = figure_path / '{}_map_weights.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

        map_transition_mat = np.exp(map_hmm.transitions.log_Ps)
        utilplot.plot_trans_matrix(map_transition_mat)
        plt.title("MAP transition matrix", fontsize=15)
        save_path = figure_path / '{}_map_transition_mat.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

        # plt.subplot(1, 2, 2)
        # plt.title("MAP transition matrix", fontsize=15)
        # utilplot.plot_trans_matrix(map_transition_mat)
        # f, ax = utilplot.plt.subplots_adjust(0, 0, 1, 1)

    ### Get expected states
    posterior_probs = map_hmm.expected_states(data=trials_to_correct, input=prev_rewards)[0]
    if plot:
        fig, ax = utilplot.plot_postprob_obs(posterior_probs, trials_to_correct, prev_rewards, map_hmm, colors, cmap)
        plt.title("MAP HMM states")
        save_path = figure_path / '{}_map_predicted_states.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['posterior_probs'] = posterior_probs
    model_dict['inferred_state_list'] = inferred_state_list
    model_dict['inferred_durations'] = inferred_durations
    model_dict['hmm_z'] = most_likely_states

    inferred_states = np.zeros(block_df.shape[0], dtype='object')
    inferred_states[ix_valid] = most_likely_states
    inferred_states[~ix_valid] = 'None'
    block_df['inferred_strategy'] = inferred_states

    if plot:
        ## Rearrange the lists of durations to be a nested list where
        ## the nth inner list is a list of durations for state n
        inferred_durations_stacked = []
        for s in range(num_states):
            inferred_durations_stacked.append(inferred_durations[inferred_state_list == s])

        fig = plt.figure(figsize=(8, 4))
        plt.hist(inferred_durations_stacked, label=['state ' + str(s) for s in range(num_states)], color=colors[:num_states])
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Inferred State Durations')
        # plt.show()

    mle_savename = sess_id + '_map_statedict.pkl'
    with open(mle_savename, 'wb') as file:
        pkl.dump(model_dict, file)

    return model_dict, block_df


def main():
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

    augmented_trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), sep=',',
                                     na_filter=False)
    block_performance = pd.read_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), sep=',',
                                    na_filter=False)

    mle_model_dict, block_performance = mle_block_states(block_performance, figure_path, sess_id_abbreviated, plot=True)
    # map_model_dict, block_performance = map_block_states(block_performance, figure_path, sess_id_abbreviated, plot=True)
    block_performance = declare_inferred_strategy(block_performance)
    augmented_trial_df = trials_inherit_strategy(block_performance, augmented_trial_df)

    # augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)
    # block_performance.to_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), index=False)
    # mle_savename = processed_data_path / (sess_id_full + '_mle_statedict.pkl')
    # with open(mle_savename, 'wb') as file:
    #     pkl.dump(mle_model_dict, file)


def declare_inferred_strategy(block_df: pd.DataFrame) -> pd.DataFrame:
    """
    Hardcode the mouse's strategy for a block. YOU MUST DO THIS BY INSPECTING THE BIAS BLOCKS AND THE
    INFERRED RL/INFERENCE BLOCKS. This is to prepare code for dataframe analysis, not an algorithm.
    Bias supercedes any HMM inference.
    """
    ix_valid = (block_df['trials_to_correct'] != 'None') & (block_df['prev_n_correct'] != 'None')
    df = block_df[ix_valid]

    strategy = np.zeros(df.shape[0], dtype='object')
    rl_ix = df['inferred_strategy'].to_numpy().astype(int) == 0
    inference_ix = df['inferred_strategy'].to_numpy().astype(int) == 1
    bias_ix = (df['bias_full_flag'] == 'True').to_numpy()
    strategy[inference_ix] = 'Inference'
    strategy[bias_ix] = 'Bias'
    strategy[rl_ix] = 'Qlearning'

    declared_strategy = np.zeros(block_df.shape[0], dtype='object')
    declared_strategy[ix_valid] = strategy
    declared_strategy[~ix_valid] = 'None'
    block_df['declared_strategy'] = declared_strategy
    return block_df


def trials_inherit_strategy(block_df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    block_ix = block_df['block_ix'].to_numpy()
    inherited_strategy = np.zeros(trial_df.shape[0], dtype='object')
    inferred_strategy = block_df['inferred_strategy'].to_numpy()
    for i in block_ix:
        tmp_ix = trial_df['cur_block'].to_numpy() == i
        inherited_strategy[tmp_ix] = inferred_strategy[i]
    trial_df['inherited_strategy'] = inherited_strategy
    return trial_df



if __name__ == '__main__':
    main()