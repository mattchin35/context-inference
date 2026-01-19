"""
Blockwise GLM-HMM using the code from Cazettes et al 2023 made by Lucca Mazzucato
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
from ssm.util import find_permutation

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


def plot_postprob_obs(posterior_probs: np.ndarray, observations: np.ndarray, inputs: np.ndarray,
                      hmm_fit, colors, cmap, session_lengths: np.ndarray=None) -> tuple[plt.Figure, plt.Axes]:

    fig, (a0, a1, a2) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 3]})

    # Plot the data and the smoothed data
    time_bins = len(inputs)
    obs_dim = len(observations[0])
    num_states = hmm_fit.transitions.log_Ps.shape[0]
    input_dim = len(inputs[0])
    for k in range(num_states):
        a0.plot(posterior_probs[:, k], label="State " + str(k + 1), lw=2, color=colors[k])

    a0.set_ylim((-0.05, 1.05))
    a0.set_yticks([0, 1], labels=[0, 1], fontsize=15)
    a0.set_ylabel("p(state)", fontsize=15)
    a0.set_xlim(0, time_bins-1)
    a0.set_xticks([])

    lim_input = 1.1 * abs(inputs).max()
    for d in range(input_dim):
        # a1.imshow(ind_state[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1, extent=(0, time_bins, -lim*obs_dim, (obs_dim)*lim), alpha=0.5)
        # a1.plot(inputs[:, d] - lim_input * d, '-c', label='inpt dim ' + str(d))
        a1.plot(inputs[:, d] - lim_input * d, label='inpt dim ' + str(d))
    # a1.plot(observations[:, d], label='obs' * (d == 0))

    a1.set_xticks([])
    a1.set_xlim(0, time_bins)
    a1.set_yticks(-np.arange(input_dim) * lim_input, ["$x_{{ {} }}$".format(n + 1) for n in range(input_dim)])
    a1.set_title('Input')
    a1.legend(fancybox=False, fontsize=10)

    lim = 2 * abs(observations).max()
    # state_detected = posterior_probs
    ind_state = np.argmax(posterior_probs, axis=1)
    indnot = np.all(posterior_probs < 0.8, axis=1)
    ind_state[indnot] = -1
    cmap.set_under('w')

    # Ey = hmm_fit.observations.mus[ind_state]
    # EW = hmm_fit.observations.Wks[ind_state]
    Ey = hmm_fit.observations.Wk[:,:,-1][ind_state]   # refactor code later to identify the last input as 1s for bias and last weights as the means
    EW = hmm_fit.observations.Wk[:,:,:-1][ind_state]
    for d in range(obs_dim):
        a2.imshow(ind_state[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=(0, time_bins, -lim * obs_dim, lim), alpha=0.5)
        a2.plot(observations[:, d] - lim * d, '-k', label='obs' * (d == 0))
        a2.plot(Ey[:, d] - lim * d, ':k', label='bias' * (d == 0))  # means
        a2.plot(EW[:, d] - lim * d, '--k', label='weight' * (d == 0))  # slopes
    a2.set_xlim(0, time_bins-1)
    # a2.set_yticks(-np.arange(obs_dim) * lim, ["$y_{{ {} }}$".format(n + 1) for n in range(obs_dim)])
    a2.set_yticks([])
    a2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    a2.set_xlabel("Context changes")
    a2.set_ylim(0, abs(observations).max())
    # a2.set_ylabel("inputs, obs")

    if session_lengths is not None:
        splits = np.cumsum(session_lengths)
        for i in range(splits.shape[0] - 1):
            a0.axvline(x=splits[i], color='k', linestyle='--')
            a1.axvline(x=splits[i], color='k', linestyle='--')
            a2.axvline(x=splits[i], color='k', linestyle='--')

    fig.tight_layout()
    return fig, (a0, a1, a2)


def plot_labeled_observations(states1: np.ndarray, posterior_probs2: np.ndarray, observations: np.ndarray, inputs: np.ndarray,
                      hmm_fit, colors, cmap, session_lengths: np.ndarray=None) -> tuple[plt.Figure, plt.Axes]:
    fig, (a0, a1, a2) = plt.subplots(3, 1)#, gridspec_kw={'height_ratios': [1, 1, 1]})
    lim = 2 * abs(observations).max()
    obs_dim = len(observations[0])
    time_bins = len(inputs)
    cmap.set_under('w')

    for d in range(obs_dim):
        a0.imshow(states1[None,:], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=(0, time_bins, -lim * obs_dim, lim), alpha=0.5)
        a0.plot(observations[:, d] - lim * d, '-k', label='obs' * (d == 0))
    a0.set_xlim(0, time_bins - 1)
    # a1.set_yticks(-np.arange(obs_dim) * lim, ["$y_{{ {} }}$".format(n + 1) for n in range(obs_dim)])
    a0.set_yticks([])
    a0.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    a0.set_xlabel("Context changes")
    a0.set_ylim(0, abs(observations).max())
    # a0.set_ylabel("inputs, obs")

    #############################################

    # state_detected = posterior_probs
    ind_state = np.argmax(posterior_probs2, axis=1)
    indnot = np.all(posterior_probs2 < 0.8, axis=1)
    ind_state[indnot] = -1
    cmap.set_under('w')

    Ey = hmm_fit.observations.Wk[:,:,-1][ind_state]   # refactor code later to identify the last input as 1s for bias and last weights as the means
    EW = hmm_fit.observations.Wk[:,:,:-1][ind_state]
    for d in range(obs_dim):
        a1.imshow(ind_state[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors) - 1,
                  extent=(0, time_bins, -lim * obs_dim, lim), alpha=0.5)
        a1.plot(observations[:, d] - lim * d, '-k', label='obs' * (d == 0))
        # a1.plot(Ey[:, d] - lim * d, ':k', label='bias' * (d == 0))  # means
        # a1.plot(EW[:, d] - lim * d, '--k', label='weight' * (d == 0))  # slopes
    a1.set_xlim(0, time_bins-1)
    # a1.set_yticks(-np.arange(obs_dim) * lim, ["$y_{{ {} }}$".format(n + 1) for n in range(obs_dim)])
    a1.set_yticks([])
    # a1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    a1.set_xlabel("Context changes")
    a1.set_ylim(0, abs(observations).max())
    # a1.set_ylabel("inputs, obs")

    if session_lengths is not None:
        splits = np.cumsum(session_lengths)
        for i in range(splits.shape[0] - 1):
            a0.axvline(x=splits[i], color='k', linestyle='--')
            a1.axvline(x=splits[i], color='k', linestyle='--')

    num_states = hmm_fit.transitions.log_Ps.shape[0]
    for k in range(num_states):
        a2.plot(posterior_probs2[:, k], label="State " + str(k + 1), lw=2, color=colors[k])

    a2.set_ylim((-0.05, 1.05))
    a2.set_yticks([0, 1], labels=[0, 1], fontsize=15)
    a2.set_ylabel("p(state)", fontsize=15)
    a2.set_xlim(0, time_bins - 1)
    a2.set_xticks([])

    fig.tight_layout()
    return fig, (a0, a1, a2)


def mle_trial_states(trial_df: pd.DataFrame, figure_path: Path, sess_id: str, plot: bool=False):
    """Maximum likelihood estimation of block strategies/states."""
    ix_valid = (trial_df['prev_action'] != 'None') & (trial_df['give_reward'] == 0)
    df = trial_df[ix_valid]

    correct = df['correct'].to_numpy().reshape(-1,1).astype(int)
    block_strategy = df['inherited_strategy'].to_numpy()
    block_strategy[block_strategy != 'None'] = block_strategy[block_strategy != 'None'].astype(int)
    block_strategy[block_strategy == 'None'] = np.amax(block_strategy[block_strategy != 'None']) + 1
    block_strategy = block_strategy.reshape(-1,1).astype(int) #+ 1
    block_bias = df['inherited_bias_flag'].to_numpy()
    block_bias[block_bias != 'True'] = False
    block_bias[block_bias == 'True'] = True
    block_bias = block_bias.reshape(-1, 1).astype(int)

    FQL_value = df['FQlearning_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    HMM_value = df['HMM_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    FQL_pLeft = df['FQlearning_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    HMM_pLeft= df['HMM_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    prev_action = df['prev_action'].to_numpy().reshape(-1, 1).astype(int)
    prev_reward = df['prev_action'].to_numpy().reshape(-1, 1).astype(int)
    bias = np.ones(df.shape[0]).reshape(-1, 1).astype(float)
    relative_value = df['relative_value'].to_numpy().reshape(-1, 1).astype(int)
    relative_omissions = df['relative_omissions'].to_numpy().reshape(-1, 1).astype(int)

    # predictors = np.concatenate([FQL_value, HMM_value, prev_action, prev_reward, bias], axis=1)
    # pred_labels = ['FQL_value', 'HMM_value', 'prev_action', 'prev_reward', 'bias']
    predictors = np.concatenate([FQL_pLeft, HMM_pLeft, block_strategy, correct, bias], axis=1)
    pred_labels = ['FQL_pLeft', 'HMM_pLeft','block_strategy', 'correct', 'bias']
    predictors = np.concatenate([FQL_pLeft, HMM_pLeft, prev_action, prev_reward, bias], axis=1)
    pred_labels = ['FQL_value', 'HMM_value', 'prev_action', 'prev_reward', 'bias']
    # predictors = np.concatenate([FQL_value, HMM_value, FQL_pLeft, HMM_pLeft,
    #                              prev_action, prev_reward, block_strategy, correct, bias], axis=1)
    # pred_labels = ['FQL_value', 'HMM_value', 'FQL_pLeft', 'HMM_pLeft',
    #                'prev_action', 'prev_reward', 'block_strategy', 'bias']
    predictors = np.concatenate([relative_value, relative_omissions, prev_action, prev_reward, bias],
                                axis=1)
    pred_labels = ['relative_value', 'relative_omissions', 'prev_action', 'prev_reward', 'bias']
    predictors = np.concatenate([relative_value, relative_omissions, prev_action, prev_reward,
                                 block_bias, block_strategy, bias], axis=1)
    pred_labels = ['relative_value', 'relative_omissions', 'prev_action', 'prev_reward',
                   'block_bias', 'block_strategy', 'bias']


    action = df['action'].to_numpy().reshape(-1, 1).astype(int)

    # num states - start with 2, Inf/RL, then do 3 (inf/rl/biased). Would like Inf(maybe lo and hi thresh)/RL/biased/disengaged/confused. I think this is
    # what cross-validation is gonna be for. less is better!
    num_states = 5
    obs_dim = 1
    input_dim = predictors.shape[1]
    num_categories = 2

    model_dict = {}
    mle_hmm = ssm.HMM(num_states, obs_dim, M=input_dim, observations="input_driven_obs",
                      observation_kwargs=dict(C=num_categories), transitions="standard")
    # mle_hmm = ssm.HMM(num_states, obs_dim, M=input_dim, observations="input_driven_obs", transitions="standard")
    N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    fit_log_likelihood = mle_hmm.fit(action, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
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
        save_path = figure_path / '{}_trial_mle_convergence.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = mle_hmm.most_likely_states(action, input=predictors)
    # recovered_weights = mle_hmm.observations.Wk
    recovered_weights = mle_hmm.observations.params
    # recovered_mus = mle_hmm.observations.mu

    # weight_dict = {}
    # weight_dict[0] = {}
    # weight_dict[0]['weights'] = recovered_weights
    # # weight_dict[0]['mus'] = recovered_mus
    # weight_dict[0]['label'] = 'mle'
    # model_dict['weight_dict'] = weight_dict
    if plot:
        f, ax = plt.subplots()
        for k in range(num_states):
            plt.plot(range(input_dim), recovered_weights[k][0], color=colors[k],
                     lw=1.5, linestyle='--')#, label='state {}'.format(k),)

        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        plt.xticks(np.arange(len(pred_labels)), pred_labels, fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        # plt.legend()
        plt.title("Weight recovery", fontsize=15)
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_weights.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

    #     utilplot.plot_weights_comparison(weight_dict)
    #     plt.title("recovered weights")
    #     save_path = figure_path / '{}_trial_mle_weights.png'.format(sess_id)
    #     plt.gcf().savefig(save_path, format='png', dpi=300)
    #
    #     mle_transition_mat = np.exp(mle_hmm.transitions.log_Ps)
    #     utilplot.plot_trans_matrix(mle_transition_mat)
    #     plt.title("MLE transition matrix", fontsize=15)
    #     save_path = figure_path / '{}_trial_mle_transition_mat.png'.format(sess_id)
    #     plt.gcf().savefig(save_path, format='png', dpi=300)

    ### Get expected states ###
    # Get expected states:
    posterior_probs = mle_hmm.expected_states(data=action, input=predictors)[0]
    if plot:
        fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
        # sess_id = 0  # session id; can choose any index between 0 and num_sess-1
        for k in range(num_states):
            plt.plot(posterior_probs[:,k], label="State " + str(k + 1), lw=2,
                     color=colors[k])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize=10)
        plt.xlabel("trial #", fontsize=15)
        plt.ylabel("p(state)", fontsize=15)
        # fig, ax = utilplot.plot_postprob_obs(posterior_probs, action, predictors, mle_hmm, colors, cmap)
        plt.title("MLE HMM states")
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_predicted_states.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

        f, ax = plot_postprob_obs(posterior_probs=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=mle_hmm, colors=colors, cmap=cmap)
        save_path = figure_path / '{}_trial_mle_utilplotSummary.png'.format(sess_id)
        f.savefig(save_path, format='png', dpi=300)


        f, ax = plot_labeled_observations(states1=block_strategy, posterior_probs2=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=mle_hmm, colors=colors, cmap=cmap)

        # f, ax = plt.subplots()
        # plt.plot(action)
        # plt.xlabel("trial #", fontsize=15)
        # plt.ylabel("action", fontsize=15)
        # plt.title("Mouse actions", fontsize=15)
        # plt.tight_layout()


    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['posterior_probs'] = posterior_probs
    model_dict['inferred_state_list'] = inferred_state_list
    model_dict['inferred_durations'] = inferred_durations
    model_dict['hmm_z'] = most_likely_states

    inferred_states = np.zeros(trial_df.shape[0], dtype='object')
    inferred_states[ix_valid] = most_likely_states
    inferred_states[~ix_valid] = 'None'
    trial_df['inferred_strategy'] = inferred_states

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
        plt.show()

    mle_savename = sess_id + '_trial_mle_statedict.pkl'
    with open(mle_savename, 'wb') as file:
        pkl.dump(model_dict, file)

    return model_dict, trial_df


def map_trial_states(trial_df: pd.DataFrame, figure_path: Path, sess_id: str, plot: bool=False,
                     prior_sigma=1, prior_alpha = 1, num_states=4):
    """Maximum likelihood estimation of block strategies/states."""
    ix_valid = (trial_df['prev_action'] != 'None') & (trial_df['give_reward'] == 0)
    df = trial_df[ix_valid]

    correct = df['correct'].to_numpy().reshape(-1,1).astype(int)
    block_strategy = df['inherited_strategy'].to_numpy()
    block_strategy[block_strategy != 'None'] = block_strategy[block_strategy != 'None'].astype(int)
    block_strategy[block_strategy == 'None'] = np.amax(block_strategy[block_strategy != 'None']) + 1
    block_strategy = block_strategy.reshape(-1,1).astype(int) #+ 1
    block_bias = df['inherited_bias_flag'].to_numpy()
    block_bias[block_bias != 'True'] = False
    block_bias[block_bias == 'True'] = True
    block_bias = block_bias.reshape(-1, 1).astype(int)

    FQL_value = df['FQlearning_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    HMM_value = df['HMM_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    FQL_pLeft = df['FQlearning_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    HMM_pLeft= df['HMM_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    prev_action = df['prev_action'].to_numpy().reshape(-1, 1).astype(int)
    prev_reward = df['prev_action'].to_numpy().reshape(-1, 1).astype(int)
    bias = np.ones(df.shape[0]).reshape(-1, 1).astype(float)
    relative_value = df['relative_value'].to_numpy().reshape(-1, 1).astype(int)
    relative_omissions = df['relative_omissions'].to_numpy().reshape(-1, 1).astype(int)

    # predictors = np.concatenate([FQL_value, HMM_value, prev_action, prev_reward, bias], axis=1)
    # pred_labels = ['FQL_value', 'HMM_value', 'prev_action', 'prev_reward', 'bias']
    predictors = np.concatenate([FQL_pLeft, HMM_pLeft, block_strategy, correct, bias], axis=1)
    pred_labels = ['FQL_pLeft', 'HMM_pLeft','block_strategy', 'correct', 'bias']
    predictors = np.concatenate([FQL_pLeft, HMM_pLeft, prev_action, prev_reward, bias], axis=1)
    pred_labels = ['FQL_value', 'HMM_value', 'prev_action', 'prev_reward', 'bias']
    # predictors = np.concatenate([FQL_value, HMM_value, FQL_pLeft, HMM_pLeft,
    #                              prev_action, prev_reward, block_strategy, correct, bias], axis=1)
    # pred_labels = ['FQL_value', 'HMM_value', 'FQL_pLeft', 'HMM_pLeft',
    #                'prev_action', 'prev_reward', 'block_strategy', 'bias']
    predictors = np.concatenate([relative_value, relative_omissions, prev_action, prev_reward, bias],
                                axis=1)
    pred_labels = ['relative_value', 'relative_omissions', 'prev_action', 'prev_reward', 'bias']
    predictors = np.concatenate([relative_value, relative_omissions, prev_action, prev_reward,
                                 block_bias, block_strategy, bias], axis=1)
    pred_labels = ['relative_value', 'relative_omissions', 'prev_action', 'prev_reward',
                   'block_bias', 'block_strategy', 'bias']


    action = df['action'].to_numpy().reshape(-1, 1).astype(int)

    # num states - start with 2, Inf/RL, then do 3 (inf/rl/biased). Would like Inf(maybe lo and hi thresh)/RL/biased/disengaged/confused. I think this is
    # what cross-validation is gonna be for. less is better!
    # num_states = 4
    obs_dim = 1
    num_categories = 2
    input_dim = predictors.shape[1]

    model_dict = {}
    map_hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                        observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
                        transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0))
    N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    fit_log_likelihood = map_hmm.fit(action, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
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
        save_path = figure_path / '{}_trial_map_convergence.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = map_hmm.most_likely_states(action, input=predictors)
    # recovered_weights = mle_hmm.observations.Wk
    recovered_weights = map_hmm.observations.params
    # recovered_mus = mle_hmm.observations.mu

    # weight_dict = {}
    # weight_dict[0] = {}
    # weight_dict[0]['weights'] = recovered_weights
    # # weight_dict[0]['mus'] = recovered_mus
    # weight_dict[0]['label'] = 'mle'
    # model_dict['weight_dict'] = weight_dict
    if plot:
        f, ax = plt.subplots()
        for k in range(num_states):
            plt.plot(range(input_dim), recovered_weights[k][0], color=colors[k],
                     lw=1.5, linestyle='--')#, label='state {}'.format(k),)

        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        plt.xticks(np.arange(len(pred_labels)), pred_labels, fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        # plt.legend()
        plt.title("Weight recovery", fontsize=15)
        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_weights.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

    #     utilplot.plot_weights_comparison(weight_dict)
    #     plt.title("recovered weights")
    #     save_path = figure_path / '{}_trial_mle_weights.png'.format(sess_id)
    #     plt.gcf().savefig(save_path, format='png', dpi=300)
    #
    #     mle_transition_mat = np.exp(mle_hmm.transitions.log_Ps)
    #     utilplot.plot_trans_matrix(mle_transition_mat)
    #     plt.title("MLE transition matrix", fontsize=15)
    #     save_path = figure_path / '{}_trial_mle_transition_mat.png'.format(sess_id)
    #     plt.gcf().savefig(save_path, format='png', dpi=300)

    ### Get expected states ###
    # Get expected states:
    posterior_probs = map_hmm.expected_states(data=action, input=predictors)[0]
    if plot:
        fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
        # sess_id = 0  # session id; can choose any index between 0 and num_sess-1
        for k in range(num_states):
            plt.plot(posterior_probs[:,k], label="State " + str(k + 1), lw=2,
                     color=colors[k])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize=10)
        plt.xlabel("trial #", fontsize=15)
        plt.ylabel("p(state)", fontsize=15)
        # fig, ax = utilplot.plot_postprob_obs(posterior_probs, action, predictors, mle_hmm, colors, cmap)
        plt.title("MAP HMM states")
        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_predicted_states.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

        f, ax = plot_postprob_obs(posterior_probs=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=map_hmm, colors=colors, cmap=cmap)
        save_path = figure_path / '{}_trial_map_utilplotSummary.png'.format(sess_id)
        f.savefig(save_path, format='png', dpi=300)


        f, ax = plot_labeled_observations(states1=block_strategy, posterior_probs2=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=map_hmm, colors=colors, cmap=cmap)

        # f, ax = plt.subplots()
        # plt.plot(action)
        # plt.xlabel("trial #", fontsize=15)
        # plt.ylabel("action", fontsize=15)
        # plt.title("Mouse actions", fontsize=15)
        # plt.tight_layout()


    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['posterior_probs'] = posterior_probs
    model_dict['inferred_state_list'] = inferred_state_list
    model_dict['inferred_durations'] = inferred_durations
    model_dict['hmm_z'] = most_likely_states

    inferred_states = np.zeros(trial_df.shape[0], dtype='object')
    inferred_states[ix_valid] = most_likely_states
    inferred_states[~ix_valid] = 'None'
    trial_df['inferred_strategy'] = inferred_states

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
        plt.show()

    mle_savename = sess_id + '_trial_map_statedict.pkl'
    with open(mle_savename, 'wb') as file:
        pkl.dump(model_dict, file)

    return model_dict, trial_df


# BIC model selection
def single_func(synthetic_data,synthetic_inpts):
        xval_hmm = ssm.HMM(num_states, obs_dim, M=input_dim,
                observations="input_driven_obs_gaussian", transitions="standard")
        hmm_lls = xval_hmm.fit(synthetic_data, inputs=synthetic_inpts, method="em", num_iters=N_iters, tolerance=TOL)
        out=xval_hmm.log_likelihood(synthetic_data,inputs=synthetic_inpts)
        return out


def run_aic():
    #Number of parameters for the model: (transition matrix) + (mean values for each state) + (covariance matrix for each state)
    time_bins=len(synthetic_data) # total number of licks
    BIC = np.zeros((max_states,nRunEM))
    AIC = np.zeros((max_states,nRunEM))
    for iS, num_states in enumerate(range(1,max_states+1)):
        K = (num_states+1)*(num_states-1) + num_states*(obs_dim*input_dim+2*obs_dim)
        results = Parallel(n_jobs=NumThread)(
            delayed(single_func)(synthetic_data, synthetic_inpts)
            for iRun in range(nRunEM))
        for iRun in range(nRunEM):
            BIC[iS,iRun] = K*np.log(time_bins) - 2*results[iRun]
            AIC[iS,iRun] = K*2 - 2*results[iRun]

    kindspline = 'quadratic'
    # Plot the xval loglik
    fig = plt.figure(figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')

    plt.subplot(1, 2, 1)

    # reshape variables for plot
    ll_training_plot = ll_training.reshape(max_states, nKfold * nRunEM)
    ll_heldout_plot = ll_heldout.reshape(max_states, nKfold * nRunEM)
    ll_training_map_plot = ll_training_map.reshape(max_states, nKfold * nRunEM)
    ll_heldout_map_plot = ll_heldout_map.reshape(max_states, nKfold * nRunEM)

    for iS, num_states in enumerate(range(1, max_states + 1)):
        plt.plot((iS + 1) * np.ones(nKfold * nRunEM), ll_training_plot[iS, :], color=colors[0], marker='o', lw=0)
        plt.plot((iS + 1) * np.ones(nKfold * nRunEM), ll_heldout_plot[iS, :], color=colors[1], marker='o', lw=0)
        plt.plot((iS + 1) * np.ones(nKfold * nRunEM), ll_training_map_plot[iS, :], color=colors[2], marker='o', lw=0)
        plt.plot((iS + 1) * np.ones(nKfold * nRunEM), ll_heldout_map_plot[iS, :], color=colors[3], marker='o', lw=0)

    x = range(1, max_states + 1)
    y = ll_training_plot.mean(axis=1);
    error = ll_training_plot.std(axis=1)
    plt.plot(x, y, label="training_MLE", color=colors[0])
    plt.fill_between(x, y - error, y + error, alpha=0.1)
    #
    y = ll_heldout_plot.mean(axis=1);
    error = ll_heldout_plot.std(axis=1)
    plt.plot(x, y, label="test_MLE", color=colors[1])
    plt.fill_between(x, y - error, y + error, alpha=0.1)
    #
    y = ll_training_map_plot.mean(axis=1);
    error = ll_training_map_plot.std(axis=1)
    plt.plot(x, y, label="training_MAP", color=colors[2])
    plt.fill_between(x, y - error, y + error, alpha=0.1)
    #
    y = ll_heldout_map_plot.mean(axis=1);
    error = ll_heldout_map_plot.std(axis=1)
    plt.plot(x, y, label="test_MAP", color=colors[3])
    plt.fill_between(x, y - error, y + error, alpha=0.1)
    plt.legend(loc="lower right")
    plt.xlabel("states")
    plt.xlim(0, max_states + 1)
    plt.ylabel("Log-Likelihood per trial")

    plt.subplot(1, 2, 2)

    x = range(1, max_states + 1);
    y = np.mean(BIC, 1);
    error = np.std(BIC, 1)
    plt.plot(x, y, label="BIC")
    plt.fill_between(x, y - error, y + error,
                     alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    y = np.mean(AIC, 1);
    error = np.std(AIC, 1)
    plt.plot(x, y, label="AIC")
    plt.xlabel("states")
    plt.xlim(0, max_states + 1)
    plt.ylabel("criterion")
    plt.legend(loc="lower left")

    savefile = 'HMM_model_sel_concAllBouts.pdf'
    plt.savefig(savefile, format="pdf", bbox_inches="tight")
    # plt.close(fig)
    plt.show()

    model_sel = {'ll_training': ll_training, 'll_heldout': ll_heldout, 'll_training_map': ll_training_map,
                 'll_heldout_map': ll_heldout_map, 'BIC': BIC}


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
    # model_dict, augmented_trial_df = mle_trial_states(augmented_trial_df, figure_path, sess_id_abbreviated, plot=True)
    model_dict, augmented_trial_df = map_trial_states(augmented_trial_df, figure_path, sess_id_abbreviated, plot=True)
    augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)


if __name__ == '__main__':
    main()