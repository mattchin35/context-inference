"""
Blockwise GLM-HMM using the code from Ashwood et al 2022
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import re
from pathlib import Path
import ssm
from joblib import Parallel, delayed
from typing import Protocol
from src.behavior_analysis.project_utils import is_present_value, is_zero_flag
from src.behavior_analysis.plotting_utils import (
    build_presentation_colors,
    build_state_colormap,
    get_state_colors,
)
from src.behavior_analysis import state_space_plotting

np.random.seed(0)


TRIAL_GLM_PREDICTOR_LABELS = {
    "FQlearning_rel_value": "FQlearning",
    # "HMM_rel_value_logodds": "HMM",
    "HMM_rel_value_logodds_decay": "HMM_decay",
    "relative_doubt_index": "doubt",
    "perseveration_regressor": "perseveration",
}
DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS = tuple(TRIAL_GLM_PREDICTOR_LABELS.keys())

# Binary GLM note for this project:
# - task actions are coded 0=right, 1=left
# - many engineered regressors are left-minus-right, so positive regressor
#   values indicate leftward evidence
# - the local `ssm` binary `input_driven_obs` implementation stores weights for
#   category 0 only, with category 1 as the zero-logit baseline
# Therefore, raw `ssm` weights describe right-choice logits. For analysis and
# weight plots in this file, we also store an interpreted left-choice view,
# which is simply the sign-flipped version of the raw binary weights.


class Session(Protocol):
    multi_session_save_path: Path
    session_data_home: Path
    sess_id_full: str
    sess_id_abbreviated: str
    raw_behavior_folder: Path
    processed_data_path: Path
    figure_path: Path
    mouse: str
    date: str
    timestamp: str
    session_info_fname: str
    session_info: dict


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
    a2.set_ylim(0-.05, abs(observations).max()+.05)
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
    a0.set_xlabel("Block context changes")
    a0.set_ylim(0-.05, abs(observations).max()+.05)
    # a0.set_ylabel("inputs, obs")

    #############################################

    # state_detected = posterior_probs
    ind_state = np.argmax(posterior_probs2, axis=1)
    indnot = np.all(posterior_probs2 < 0.8, axis=1)
    ind_state[indnot] = -1
    cmap.set_under('w')

    Ey = hmm_fit.observations.Wk[:,:,-1][ind_state]   # refactor code later to identify the last input as 1s for bias and last weights as the means
    EW = hmm_fit.observations.Wk[:,:,:-1][ind_state]
    # EW = hmm_fit.o9[:,:,:-1][ind_state]
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
    a1.set_xlabel("Trial context changes")
    a1.set_ylim(0-.05, abs(observations).max()+.05)
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


def prepare_trial_glm_hmm_data(
    trial_df: pd.DataFrame,
    require_inherited_strategy: bool = False,
    predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
) -> dict[str, np.ndarray | list[str]]:
    """Build session-wise GLM-HMM observations and predictors from trial data.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe where rows correspond to task trials. Required base
        columns are `prev_action`, `give_reward`, and `action`, plus each
        selected predictor column in `predictor_columns`. If
        `require_inherited_strategy=True`, `inherited_block_strategy` must also be
        present on valid rows. Real NaN values and string missing-value
        sentinels are invalid.
    require_inherited_strategy : bool, default=False
        Whether valid trials must also have a present `inherited_block_strategy`
        label. This is only needed for plotting and
        block-trial comparison logic in the MAP fitting path.
    predictor_columns : tuple[str, ...], default=DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS
        Tuple of non-bias predictor column names to include in the GLM-HMM
        input matrix. A bias column of ones is always appended as the final
        predictor so one-predictor GLM-HMM fits remain valid.

    Returns
    -------
    dict[str, np.ndarray | list[str]]
        Dictionary with:
        - `observations`: np.ndarray, shape (n_valid_trials, 1), binary action
          observations
        - `inputs`: np.ndarray, shape (n_valid_trials, n_predictors + 1), GLM
          regressors in the requested predictor order with a final `bias`
          column
        - `valid_mask`: np.ndarray, shape (n_trials,), boolean mask selecting
          valid trials
        - `predictor_labels`: list[str], regressor names in input-column order
    """
    if isinstance(predictor_columns, str):
        predictor_columns = (predictor_columns,)
    if len(predictor_columns) == 0:
        raise ValueError("predictor_columns must include at least one non-bias GLM predictor.")

    required_columns = {"prev_action", "give_reward", "action"}
    if require_inherited_strategy:
        required_columns.add("inherited_block_strategy")
    required_columns.update(predictor_columns)

    missing_columns = sorted(required_columns.difference(trial_df.columns))
    if missing_columns:
        missing_summary = ", ".join(missing_columns)
        raise ValueError(f"trial_df is missing required GLM-HMM columns: {missing_summary}")

    unknown_predictors = [col for col in predictor_columns if col not in TRIAL_GLM_PREDICTOR_LABELS]
    if unknown_predictors:
        unknown_summary = ", ".join(unknown_predictors)
        raise ValueError(f"Unknown GLM-HMM predictor columns requested: {unknown_summary}")

    valid_mask = is_present_value(trial_df["prev_action"]) & is_zero_flag(trial_df["give_reward"])
    if require_inherited_strategy:
        valid_mask = valid_mask & is_present_value(trial_df["inherited_block_strategy"])
    df = trial_df[valid_mask]

    observations = df["action"].to_numpy().reshape(-1, 1).astype(int)
    bias = np.ones((df.shape[0], 1), dtype=float)
    predictor_arrays = [
        df[predictor_name].to_numpy().reshape(-1, 1).astype(float)
        for predictor_name in predictor_columns
    ]
    inputs = np.concatenate([*predictor_arrays, bias], axis=1)

    return {
        "observations": observations,
        "inputs": inputs,
        "valid_mask": valid_mask.to_numpy(),
        "predictor_labels": [TRIAL_GLM_PREDICTOR_LABELS[col] for col in predictor_columns] + ["bias"],
    }


def normalize_binary_glm_weights(raw_weights: np.ndarray) -> np.ndarray:
    """Normalize binary GLM-HMM weights to shape (num_states, 1, num_predictors).

    Parameters
    ----------
    raw_weights : np.ndarray
        Raw `ssm` observation weights for a binary GLM-HMM. Expected logical
        shape is `(num_states, 1, num_predictors)` because binary
        `input_driven_obs` stores parameters for category 0 only.

    Returns
    -------
    np.ndarray
        Normalized raw weights with shape `(num_states, 1, num_predictors)`.
    """
    normalized = np.asarray(raw_weights, dtype=float)
    if normalized.ndim == 1:
        normalized = normalized[np.newaxis, np.newaxis, :]
    elif normalized.ndim == 2:
        normalized = normalized[:, np.newaxis, :]
    if normalized.ndim != 3 or normalized.shape[1] != 1:
        raise ValueError(
            "Binary GLM-HMM weights must have shape (num_states, 1, num_predictors)."
        )
    return normalized


def interpret_binary_glm_weights(raw_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return raw and interpreted binary GLM-HMM weights.

    Parameters
    ----------
    raw_weights : np.ndarray
        Raw `ssm` binary GLM weights for category 0. In this task category 0 is
        the right choice and category 1 is the left-choice baseline.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - normalized raw weights with shape `(num_states, 1, num_predictors)`
        - interpreted left-choice weights with the same shape, where positive
          values mean the regressor promotes left choices
    """
    normalized_raw = normalize_binary_glm_weights(raw_weights)
    interpreted_left_choice = -normalized_raw
    return normalized_raw, interpreted_left_choice


def build_block_comparison_colors(block_weight_dict: dict) -> tuple[list, object]:
    """Build semantic colors for trial plots that display block-model states.

    Parameters
    ----------
    block_weight_dict : dict
        Block-model weight dictionary containing `weights`. Expected weight
        shape is `(num_states, obs_dim, input_dim)`, but squeezed one-state or
        one-predictor arrays are accepted.

    Returns
    -------
    tuple[list, object]
        Semantic presentation colors and a matching state colormap.
    """
    block_weights = np.asarray(block_weight_dict["weights"], dtype=float)
    if block_weights.ndim == 0:
        primary_predictor_weights = block_weights.reshape(1)
    elif block_weights.ndim == 1:
        primary_predictor_weights = block_weights
    else:
        primary_predictor_weights = block_weights.reshape(block_weights.shape[0], -1)[:, 0]

    return build_presentation_colors(primary_predictor_weights)


def split_blocked_holdout_sequences(
    observations: np.ndarray,
    inputs: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Split one trial sequence into train segments and one held-out block.

    Parameters
    ----------
    observations : np.ndarray
        Binary action observations with shape (T, 1), where rows index valid
        trials within one session.
    inputs : np.ndarray
        Predictor matrix with shape (T, M), aligned row-wise to `observations`.
    test_indices : np.ndarray
        Integer indices of the contiguous held-out trial block, shape (K,).

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]
        - training observation sequences: list of arrays, each shape (T_i, 1)
        - training input sequences: list of arrays, each shape (T_i, M)
        - held-out observations: np.ndarray, shape (K, 1)
        - held-out inputs: np.ndarray, shape (K, M)
    """
    test_indices = np.sort(np.asarray(test_indices, dtype=int))
    train_mask = np.ones(observations.shape[0], dtype=bool)
    train_mask[test_indices] = False

    train_indices = np.flatnonzero(train_mask)
    train_observation_sequences: list[np.ndarray] = []
    train_input_sequences: list[np.ndarray] = []
    if train_indices.size > 0:
        split_points = np.where(np.diff(train_indices) != 1)[0] + 1
        contiguous_segments = np.split(train_indices, split_points)
        for segment_indices in contiguous_segments:
            train_observation_sequences.append(observations[segment_indices])
            train_input_sequences.append(inputs[segment_indices])

    return (
        train_observation_sequences,
        train_input_sequences,
        observations[test_indices],
        inputs[test_indices],
    )


def mle_trial_states(trial_df: pd.DataFrame, figure_path: Path, sess_id: str, plot: bool=False, model_dict=None,
                     num_states=2,
                     predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS):
    """Maximum likelihood estimation of block strategies/states."""
    prepared = prepare_trial_glm_hmm_data(trial_df, predictor_columns=predictor_columns)
    ix_valid = prepared["valid_mask"]
    df = trial_df[ix_valid]

    correct = df['correct'].to_numpy().reshape(-1,1).astype(int)
    block_strategy = df['inherited_block_strategy'].to_numpy()
    block_strategy[block_strategy != 'None'] = block_strategy[block_strategy != 'None'].astype(int)
    block_strategy[block_strategy == 'None'] = np.amax(block_strategy[block_strategy != 'None']) + 1
    block_strategy = block_strategy.reshape(-1,1).astype(int) #+ 1
    block_bias = df['inherited_block_bias'].to_numpy()
    block_bias[block_bias != 'True'] = False
    block_bias[block_bias == 'True'] = True
    block_bias = block_bias.reshape(-1, 1).astype(int)

    # FQL_value = df['FQlearning_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    # HMM_value = df['HMM_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    # FQL_pLeft = df['FQlearning_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    # HMM_pLeft= df['HMM_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    predictors = prepared["inputs"]
    pred_labels = prepared["predictor_labels"]
    action = prepared["observations"]

    # num states - start with 2, Inf/RL, then do 3 (inf/rl/biased). Would like Inf(maybe lo and hi thresh)/RL/biased/disengaged/confused. I think this is
    # what cross-validation is gonna be for. less is better!
    obs_dim = 1
    input_dim = predictors.shape[1]
    num_categories = 2
    state_colors = get_state_colors(num_states)
    state_cmap = build_state_colormap(state_colors)

    if model_dict is None:
        model_dict = {}
    model_dict['mle'] = {}
    weight_dict = {}

    mle_hmm = ssm.HMM(num_states, obs_dim, M=input_dim, observations="input_driven_obs",
                      observation_kwargs=dict(C=num_categories), transitions="standard")
    # mle_hmm = ssm.HMM(num_states, obs_dim, M=input_dim, observations="input_driven_obs", transitions="standard")
    N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    fit_log_likelihood = mle_hmm.fit(action, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['mle']['fit_log_likelihood'] = fit_log_likelihood

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
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_convergence.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = mle_hmm.most_likely_states(action, input=predictors)
    raw_recovered_weights, interpreted_weights = interpret_binary_glm_weights(
        mle_hmm.observations.params
    )

    # `raw_weights` preserve the native `ssm` category-0/right-choice logits.
    # `weights` are the binary-task interpretation used for analysis and plots:
    # positive weights mean the predictor promotes left choices.
    weight_dict['weights'] = interpreted_weights
    weight_dict['raw_weights'] = raw_recovered_weights
    weight_dict['weight_labels'] = pred_labels
    weight_dict['label'] = 'mle'
    weight_dict['weight_convention'] = (
        "weights are interpreted left-choice coefficients; raw_weights are "
        "binary ssm category 0 (right-choice) coefficients"
    )

    model_dict['mle']['weight_dict'] = weight_dict

    # weight_dict = {}
    # weight_dict[0] = {}
    # weight_dict[0]['weights'] = recovered_weights
    # # weight_dict[0]['mus'] = recovered_mus
    # weight_dict[0]['label'] = 'mle'
    # model_dict['weight_dict'] = weight_dict
    if plot:
        f, ax = plt.subplots()
        for k in range(num_states):
            plt.plot(range(input_dim), interpreted_weights[k][0], color=state_colors[k],
                     lw=1.5, linestyle='--')#, label='state {}'.format(k),)

        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        # plt.xticks(np.arange(len(pred_labels)), pred_labels, fontsize=12, rotation=45)
        plt.xticks(np.arange(len(pred_labels)), [lab.replace('_', ' ') for lab in pred_labels], fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.title("Model weights (interpreted left-choice coefficients)", fontsize=15)
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_weights.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

    ### Get expected states ###
    posterior_probs = mle_hmm.expected_states(data=action, input=predictors)[0]
    if plot:
        fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
        # sess_id = 0  # session id; can choose any index between 0 and num_sess-1
        for k in range(num_states):
            plt.plot(posterior_probs[:,k], label="State " + str(k + 1), lw=2,
                     color=state_colors[k])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize=10)
        plt.xlabel("trial #", fontsize=15)
        plt.ylabel("p(state)", fontsize=15)
        plt.title("MLE HMM states")
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_predicted_states.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

        f, ax = plot_postprob_obs(posterior_probs=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=mle_hmm, colors=state_colors, cmap=state_cmap)
        save_path = figure_path / '{}_trial_mle_state_summary.png'.format(sess_id)
        f.savefig(save_path, format='png', dpi=300)


        f, ax = plot_labeled_observations(states1=block_strategy, posterior_probs2=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=mle_hmm, colors=state_colors, cmap=state_cmap)

        # f, ax = plt.subplots()
        # plt.plot(action)
        # plt.xlabel("trial #", fontsize=15)
        # plt.ylabel("action", fontsize=15)
        # plt.title("Mouse actions", fontsize=15)
        # plt.tight_layout()

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['mle']['posterior_probs'] = posterior_probs
    model_dict['mle']['inferred_state_list'] = inferred_state_list
    model_dict['mle']['inferred_durations'] = inferred_durations
    model_dict['mle']['hmm_z'] = most_likely_states

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
        plt.hist(inferred_durations_stacked, label=['state ' + str(s) for s in range(num_states)], color=state_colors)
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Inferred State Durations')
        # plt.show()


    plt.close('all')
    return model_dict, trial_df


def map_trial_states(trial_df: pd.DataFrame, figure_path: Path, sess_id: str, plot: bool=False,
                     num_states=1, prior_sigma=1, prior_alpha=2, model_dict=None, block_dict=None,
                     predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS):
    """Maximum likelihood estimation of block strategies/states."""
    prepared = prepare_trial_glm_hmm_data(
        trial_df,
        require_inherited_strategy=True,
        predictor_columns=predictor_columns,
    )
    ix_valid = prepared["valid_mask"]
    df = trial_df[ix_valid]
    n_trials = df.shape[0]
    state_colors = get_state_colors(num_states)
    state_cmap = build_state_colormap(state_colors)

    correct = df['correct'].to_numpy().reshape(-1,1).astype(int)
    block_strategy = df['inherited_block_strategy'].to_numpy()
    # block_strategy[block_strategy != 'None'] = block_strategy[block_strategy != 'None'].astype(int)
    # block_strategy[block_strategy == 'None'] = np.amax(block_strategy[block_strategy != 'None']) + 1
    block_strategy = block_strategy.reshape(-1,1).astype(int) #+ 1
    block_bias = df['inherited_block_bias'].to_numpy()
    block_bias[block_bias != 'True'] = False
    block_bias[block_bias == 'True'] = True
    block_bias = block_bias.reshape(-1, 1).astype(int)

    # FQL_value = df['FQlearning_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    # HMM_value = df['HMM_rel_value'].to_numpy().reshape(-1, 1).astype(float)
    # FQL_pLeft = df['FQlearning_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    # HMM_pLeft= df['HMM_prob_left'].to_numpy().reshape(-1, 1).astype(float)
    predictors = prepared["inputs"]
    pred_labels = prepared["predictor_labels"]
    action = prepared["observations"]
    # observations = np.concatenate([action, block_strategy], axis=1)
    observations = action

    obs_dim = 1
    num_categories = 2
    input_dim = predictors.shape[1]

    if model_dict is None:
        model_dict = {}

    model_dict['map'] = {}
    weight_dict = {}

    map_hmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                        observation_kwargs=dict(C=num_categories, prior_sigma=prior_sigma),
                        transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0))
    N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    fit_log_likelihood = map_hmm.fit(action, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['map']['fit_log_likelihood'] = fit_log_likelihood

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
        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_convergence.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = map_hmm.most_likely_states(action, input=predictors)
    raw_recovered_weights, interpreted_weights = interpret_binary_glm_weights(
        map_hmm.observations.params
    )

    # `raw_weights` preserve the native `ssm` category-0/right-choice logits.
    # `weights` are the binary-task interpretation used for analysis and plots:
    # positive weights mean the predictor promotes left choices.
    weight_dict['weights'] = interpreted_weights
    weight_dict['raw_weights'] = raw_recovered_weights
    weight_dict['weight_labels'] = pred_labels
    weight_dict['label'] = 'map'
    weight_dict['weight_convention'] = (
        "weights are interpreted left-choice coefficients; raw_weights are "
        "binary ssm category 0 (right-choice) coefficients"
    )
    model_dict['map']['weight_dict'] = weight_dict

    # weight_dict = {}
    # weight_dict[0] = {}
    # weight_dict[0]['weights'] = recovered_weights
    # # weight_dict[0]['mus'] = recovered_mus
    # weight_dict[0]['label'] = 'mle'
    # model_dict['weight_dict'] = weight_dict
    if plot:
        f, ax = plt.subplots()
        for k in range(num_states):
            plt.plot(range(input_dim), interpreted_weights[k][0], color=state_colors[k],
                     lw=1.5, linestyle='--')#, label='state {}'.format(k),)

        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        # plt.xticks(np.arange(len(pred_labels)), pred_labels, fontsize=12, rotation=45)
        plt.xticks(np.arange(len(pred_labels)), [lab.replace('_', ' ') for lab in pred_labels], fontsize=12,
                   rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.title("Model weights (interpreted left-choice coefficients)", fontsize=15)

        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_weights.png'.format(sess_id)
        plt.gcf().savefig(save_path, format='png', dpi=300)

    ### Get expected states ###
    posterior_probs = map_hmm.expected_states(data=action, input=predictors)[0]
    if plot:
        fig = plt.figure(figsize=(5, 2.5), dpi=80, facecolor='w', edgecolor='k')
        # sess_id = 0  # session id; can choose any index between 0 and num_sess-1
        for k in range(num_states):
            plt.plot(posterior_probs[:,k], label="State " + str(k + 1), lw=2,
                     color=state_colors[k])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize=10)
        plt.xlabel("Trial", fontsize=15)
        plt.ylabel("p(Strategy)", fontsize=15)
        plt.xlim((0, n_trials))
        # plt.title("MAP HMM states")
        plt.title("HMM behavior probabilities")
        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_predicted_states.png'.format(sess_id)
        fig.savefig(save_path, format='png', dpi=300)

        f, ax = plot_postprob_obs(posterior_probs=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=map_hmm, colors=state_colors, cmap=state_cmap)
        save_path = figure_path / '{}_trial_map_state_summary.png'.format(sess_id)
        f.savefig(save_path, format='png', dpi=300)


        f, ax = plot_labeled_observations(states1=block_strategy, posterior_probs2=posterior_probs,observations=action, inputs=predictors,
                                  hmm_fit=map_hmm, colors=state_colors, cmap=state_cmap)

        ############################################
        if block_dict is not None:
            f, ax = plt.subplots(2,1)

            block_present_colors, block_present_cmap = build_block_comparison_colors(block_dict['weight_dict'])
            block_posterior_probs = block_dict['posterior_probs']
            lim = 2 * abs(observations).max()
            ind_state = np.argmax(block_posterior_probs, axis=1)
            indnot = np.all(block_posterior_probs < 0.55, axis=1)
            ind_state[indnot] = -1
            time_bins = len(predictors)

            ax[0].imshow(ind_state[None, :], aspect="auto", cmap=block_present_cmap,
                      vmin=0, vmax=len(block_present_colors) - 1,
                      extent=(0, time_bins, -lim * obs_dim, lim), alpha=0.5)
            ax[0].plot(observations[:, 0], '-k', label='obs')
            ax[0].set_ylim(-0.2, 1.2)
            ax[0].set_yticks([0, 1])
            ax[0].set_yticklabels(['Right', 'Left'], fontsize=12)
            ax[0].set_title('Trials labeled by block strategy')

            ind_state = np.argmax(posterior_probs, axis=1)
            indnot = np.all(posterior_probs < 0.55, axis=1)
            ind_state[indnot] = -1

            ax[1].imshow(ind_state[None, :], aspect="auto", cmap=state_cmap,
                      vmin=0, vmax=len(state_colors) - 1,
                      extent=(0, time_bins, -lim * obs_dim, lim), alpha=0.5)
            ax[1].plot(observations[:, 0], '-k', label='obs')
            ax[1].set_ylim(-0.2, 1.2)
            ax[1].set_yticks([0, 1])
            ax[1].set_yticklabels(['Right', 'Left'], fontsize=12)
            ax[1].set_title('Trials labeled by trial strategy')
            plt.tight_layout()

            save_path = figure_path / '{}_trialMapStateComparison.png'.format(sess_id)
            f.savefig(save_path, format='png', dpi=300)
            plt.show()

        ######################

        # f, ax = plt.subplots()
        # plt.plot(action)
        # plt.xlabel("trial #", fontsize=15)
        # plt.ylabel("action", fontsize=15)
        # plt.title("Mouse actions", fontsize=15)
        # plt.tight_layout()


    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['map']['posterior_probs'] = posterior_probs
    model_dict['map']['inferred_state_list'] = inferred_state_list
    model_dict['map']['inferred_durations'] = inferred_durations
    model_dict['map']['hmm_z'] = most_likely_states

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
        plt.hist(inferred_durations_stacked, label=['state ' + str(s) for s in range(num_states)], color=state_colors)
        plt.xlabel('Duration')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title('Histogram of Inferred State Durations')
        # plt.show()

    # if num_states > 1:
    #     if np.sum(~ix_valid) > 0:
    #         cur_strategy_slope = np.zeros(block_df.shape[0], dtype='object')
    #         cur_strategy_slope[ix_valid] = np.squeeze(recovered_weights)[:,0][inferred_states[ix_valid].astype(int)]
    #         cur_strategy_slope[~ix_valid] = 'None'
    #         block_df['cur_strategy_slope'] = cur_strategy_slope
    #     else:
    #         block_df['cur_strategy_slope'] = np.squeeze(recovered_weights)[:,0][inferred_states.astype(int)]
    # else:
    #     block_df['cur_strategy_slope'] = np.ones(block_df.shape[0]) * np.squeeze(recovered_weights)[0]


    return model_dict, trial_df


def save_trial_model_dict(model_dict: dict, session: Session) -> Path:
    """Save a fitted trial GLM-HMM model dictionary.

    Parameters
    ----------
    model_dict : dict
        Model output dictionary containing fitted trial-model results. Expected
        top-level keys are workflow-dependent, commonly `mle` and `map`.
    session : Session
        Session metadata with `processed_data_path` and `sess_id_full`.

    Returns
    -------
    Path
        Pickle path with filename `{sess_id_full}_trial_statedict.pkl`.
    """
    save_path = session.processed_data_path / f"{session.sess_id_full}_trial_statedict.pkl"
    with open(save_path, "wb") as file:
        pkl.dump(model_dict, file)
    return save_path


def run_trial_modeling(trial_df: pd.DataFrame, session: Session, num_states: int=2,
                       prior_alpha=1, prior_sigma=1,
                       predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS) -> pd.DataFrame:
    block_dict_fname = session.processed_data_path / (session.sess_id_full + '_block_statedict.pkl')
    with open(block_dict_fname, 'rb') as file:
        block_dict = pkl.load(file)

    mle_model_dict, _ = mle_trial_states(trial_df, session.figure_path, session.sess_id_abbreviated,
                                                         plot=True, num_states=num_states,
                                                         predictor_columns=predictor_columns)
    map_model_dict, augmented_trial_df = map_trial_states(trial_df, session.figure_path, session.sess_id_abbreviated,
                                                         plot=True, model_dict=mle_model_dict, num_states=num_states,
                                                          prior_alpha=prior_alpha, prior_sigma=prior_sigma,
                                                          block_dict=block_dict['map'],
                                                          predictor_columns=predictor_columns)
    save_trial_model_dict(map_model_dict, session)

    augmented_trial_df.to_csv(session.processed_data_path / (session.sess_id_full + '_augmented_trials.csv'), index=False)

    fig, ax = state_space_plotting.plot_trial_glm_hmm_weights(
        [map_model_dict['map']['weight_dict']],
    )
    save_path = session.figure_path / f"{session.sess_id_full}_trial_hmm_weights.png"
    fig.savefig(save_path, format='png', dpi=300)
    return augmented_trial_df


def build_input_driven_glm_hmm(
    num_states: int,
    obs_dim: int,
    input_dim: int,
    algorithm: str = 'MLE',
    prior_alpha: float = 1,
    prior_sigma: float = 1,
):
    """Build an input-driven GLM-HMM using either MLE or MAP settings.

    Parameters
    ----------
    num_states : int
        Number of hidden GLM-HMM states.
    obs_dim : int
        Observation dimensionality. For binary choice GLM-HMM this is 1.
    input_dim : int
        Number of regressors in the trial-level input matrix.
    algorithm : str, default='MLE'
        Estimation family. `MLE` uses standard transitions and `MAP` uses a
        sticky prior.
    prior_alpha : float, default=1
        Sticky-transition concentration parameter used for `MAP`.
    prior_sigma : float, default=1
        Observation prior scale used for `MAP`.

    Returns
    -------
    ssm.HMM
        Configured input-driven categorical-observation HMM.
    """
    algorithm = algorithm.upper()
    if algorithm == 'MLE':
        return ssm.HMM(
            num_states,
            obs_dim,
            M=input_dim,
            observations="input_driven_obs",
            observation_kwargs=dict(C=2),
            transitions="standard",
        )
    if algorithm == 'MAP':
        return ssm.HMM(
            num_states,
            obs_dim,
            M=input_dim,
            observations="input_driven_obs",
            observation_kwargs=dict(C=2, prior_sigma=prior_sigma),
            transitions="sticky",
            transition_kwargs=dict(alpha=prior_alpha, kappa=0),
        )
    raise ValueError(f"Algorithm must be 'MLE' or 'MAP', got {algorithm}")


def fit_glm_hmm_and_score_log_likelihood(observations: np.ndarray, inputs: np.ndarray, num_states: int,
                                         prior_sigma=1, prior_alpha=1, algorithm='MLE',
                                         n_iter: int = 1000, tol: float = 10**-4):
    """Fit one GLM-HMM restart and return its training log likelihood.

    Parameters
    ----------
    observations : np.ndarray
        Binary action observations with shape `(n_trials, 1)`.
    inputs : np.ndarray
        Predictor matrix with shape `(n_trials, n_predictors)`.
    num_states : int
        Number of hidden GLM-HMM states.
    prior_sigma : float, default=1
        Observation prior scale used for MAP fits.
    prior_alpha : float, default=1
        Sticky-transition concentration used for MAP fits.
    algorithm : str, default='MLE'
        Fitting mode, either 'MLE' or 'MAP'.
    n_iter : int, default=1000
        Maximum EM iterations.
    tol : float, default=1e-4
        EM convergence tolerance.

    Returns
    -------
    float
        Fitted model log likelihood on `observations`, in log-probability units.
    """
    assert algorithm in ['MAP', 'MLE'], "Algorithm must be MAP or MLE"

    obs_dim, input_dim = observations.shape[1], inputs.shape[1]
    hmm = build_input_driven_glm_hmm(
        num_states=num_states,
        obs_dim=obs_dim,
        input_dim=input_dim,
        algorithm=algorithm,
        prior_alpha=prior_alpha,
        prior_sigma=prior_sigma,
    )

    hmm_lls = hmm.fit(observations, inputs=inputs, method="em", num_iters=n_iter, tolerance=tol)
    return hmm.log_likelihood(observations, inputs=inputs)


def count_glm_hmm_parameters(num_states: int, input_dim: int, num_categories: int = 2) -> int:
    """Count free parameters for binary input-driven GLM-HMM information criteria.

    Parameters
    ----------
    num_states : int
        Number of hidden GLM-HMM states.
    input_dim : int
        Number of regressors in the input matrix. This includes the bias column
        if a bias regressor is used explicitly.
    num_categories : int, default=2
        Number of categorical observation outcomes.

    Returns
    -------
    int
        Total free-parameter count: transition rows, initial-state
        probabilities, and state-specific GLM weights.
    """
    n_transition_params = num_states * (num_states - 1)
    n_initial_params = num_states - 1
    n_observation_params = num_states * (num_categories - 1) * input_dim
    return n_transition_params + n_initial_params + n_observation_params


def calculate_information_criteria(observations: np.ndarray, inputs: np.ndarray, states: npt.NDArray[np.int64],
                                   nRunEM: int, n_jobs: int, algorithm='MLE', prior_alpha=1, prior_sigma=1, ):
    if algorithm.upper() != 'MLE':
        raise ValueError("GLM-HMM information criteria should only be run on MLE models.")

    n_timesteps = observations.shape[0]
    input_dim = inputs.shape[1]
    n_states = states.size
    num_categories = 2

    AIC = np.zeros((n_states, nRunEM))
    BIC = np.zeros((n_states, nRunEM))
    for iS, num_states in enumerate(states): #range(2, n + 1)):
        print("running {} state(s)".format(num_states))

        K = count_glm_hmm_parameters(num_states=num_states, input_dim=input_dim, num_categories=num_categories)
        delayed_calls = [
            delayed(fit_glm_hmm_and_score_log_likelihood)(
                observations,
                inputs,
                num_states,
                prior_sigma,
                prior_alpha,
                'MLE',
                n_iter=1000,
                tol=1e-4,
            )
            for iRun in range(nRunEM)
        ]
        results = Parallel(n_jobs=n_jobs)(delayed_calls)

        for iRun in range(nRunEM):
            AIC[iS, iRun] = K * 2 - 2 * results[iRun]
            BIC[iS, iRun] = K * np.log(n_timesteps) - 2 * results[iRun]

    return AIC, BIC


def plot_information_criteria(aic, bic, states, figure_path: Path, sess_id_full: str):
    """Plot GLM-HMM AIC/BIC scores by state count.

    Parameters
    ----------
    aic : np.ndarray
        AIC values with shape `(n_states, n_restarts)`.
    bic : np.ndarray
        BIC values with shape `(n_states, n_restarts)`.
    states : np.ndarray
        Candidate state counts with shape `(n_states,)`.
    figure_path : Path
        Directory where the plot is saved.
    sess_id_full : str
        Full session identifier used in the saved figure filename.

    Returns
    -------
    None
        Saves an AIC/BIC plot to `figure_path`.
    """
    f, ax = plt.subplots(facecolor='w', edgecolor='k')

    y = np.mean(bic, 1)
    error = np.std(bic, 1)
    bic_line = plt.plot(states, y, label="BIC")[0]
    bic_color = bic_line.get_color()
    plt.fill_between(states, y - error, y + error,
                     alpha=0.3, color=bic_color)

    y = np.mean(aic, 1)
    error = np.std(aic, 1)
    aic_line = plt.plot(states, y, label="AIC")[0]
    aic_color = aic_line.get_color()
    plt.fill_between(states, y - error, y + error,
                     alpha=0.3, color=aic_color)
    plt.xlim(np.amin(states)-.2, np.amax(states)+.2)
    plt.xlabel("Number of states")
    plt.xticks(states)
    plt.ylabel("criterion")
    plt.legend(loc="upper left", frameon=False)
    plt.title("BIC and AIC", fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    save_path = figure_path / '{}_trial_HMM_AIC_BIC.png'.format(sess_id_full)
    plt.tight_layout()
    plt.gcf().savefig(save_path, format='png', dpi=300)
    print(f"Saved AIC/BIC plot to {save_path}")
    plt.show()


def build_blocked_holdout_indices(n_timesteps: int, n_folds: int = 5) -> list[np.ndarray]:
    """Create contiguous within-session held-out trial blocks.

    Parameters
    ----------
    n_timesteps : int
        Number of valid trials in the session sequence.
    n_folds : int, default=5
        Number of contiguous held-out blocks.

    Returns
    -------
    list[np.ndarray]
        List of contiguous test-index arrays. Each array indexes one held-out
        trial block along the session time axis.
    """
    if n_folds < 2 or n_folds > n_timesteps:
        raise ValueError(f"n_folds must be in [2, {n_timesteps}], got {n_folds}")
    return [fold_indices.astype(int) for fold_indices in np.array_split(np.arange(n_timesteps), n_folds)]


def single_blocked_holdout_func(
    observations: np.ndarray,
    inputs: np.ndarray,
    num_states: int,
    test_indices_list: list[np.ndarray],
    algorithm: str = 'MLE',
    n_iter: int = 1000,
    tol: float = 1e-4,
    prior_alpha: float = 1,
    prior_sigma: float = 1,
):
    """Score one GLM-HMM state count on blocked within-session held-out splits.

    Parameters
    ----------
    observations : np.ndarray
        Binary action observations with shape (T, 1), where rows index valid
        session trials.
    inputs : np.ndarray
        Predictor matrix with shape (T, M), aligned row-wise to `observations`.
    num_states : int
        Number of hidden GLM-HMM states to fit.
    test_indices_list : list[np.ndarray]
        List of contiguous held-out block indices. Each entry is a 1D integer
        array indexing one held-out block within the session.
    algorithm : str, default='MLE'
        Estimation family passed to the HMM builder. `MLE` is the primary
        session-wise selector; `MAP` can be used for secondary held-out support.
    n_iter : int, default=1000
        Maximum EM iterations per fit.
    tol : float, default=1e-4
        EM convergence tolerance.
    prior_alpha : float, default=1
        Sticky-transition concentration for MAP fits.
    prior_sigma : float, default=1
        Observation prior scale for MAP fits.

    Returns
    -------
    np.ndarray
        Held-out log-likelihood per trial block, shape (n_folds,). Each score
        is normalized by the number of held-out observations in that block.
    """
    fold_log_likelihoods = np.zeros(len(test_indices_list))
    obs_dim, input_dim = observations.shape[1], inputs.shape[1]

    for i_fold, test_idx in enumerate(test_indices_list):
        train_observations, train_inputs, test_observations, test_inputs = split_blocked_holdout_sequences(
            observations=observations,
            inputs=inputs,
            test_indices=test_idx,
        )
        hmm = build_input_driven_glm_hmm(
            num_states=num_states,
            obs_dim=obs_dim,
            input_dim=input_dim,
            algorithm=algorithm,
            prior_alpha=prior_alpha,
            prior_sigma=prior_sigma,
        )
        hmm.fit(train_observations, inputs=train_inputs, method="em", num_iters=n_iter, tolerance=tol)
        fold_log_likelihoods[i_fold] = (
            hmm.log_likelihood(test_observations, inputs=test_inputs) / len(test_observations)
        )

    return fold_log_likelihoods


def calculate_blocked_holdout_scores(
    observations: np.ndarray,
    inputs: np.ndarray,
    states: npt.NDArray[np.int64],
    nRunEM: int = 5,
    n_folds: int = 5,
    n_jobs: int = 4,
    algorithm: str = 'MLE',
    prior_alpha: float = 1,
    prior_sigma: float = 1,
):
    """Compute blocked within-session held-out log-likelihoods across state counts.

    Parameters
    ----------
    observations : np.ndarray
        Observation array with shape (T, 1), where rows index valid session
        trials and values are binary actions.
    inputs : np.ndarray
        Predictor matrix with shape (T, M), aligned row-wise to `observations`.
    states : np.ndarray
        Candidate hidden-state counts, shape (n_states,).
    nRunEM : int, default=5
        Number of EM restarts per state count.
    n_folds : int, default=5
        Number of contiguous held-out blocks within the session.
    n_jobs : int, default=4
        Number of joblib workers used across EM restarts.
    algorithm : str, default='MLE'
        Estimation family passed to the HMM builder.
    prior_alpha : float, default=1
        Sticky-transition concentration for MAP fits.
    prior_sigma : float, default=1
        Observation prior scale for MAP fits.

    Returns
    -------
    np.ndarray
        Held-out log-likelihood per trial, shape
        (n_states, n_restarts, n_folds).
    """
    n_states = states.size
    test_indices_list = build_blocked_holdout_indices(observations.shape[0], n_folds=n_folds)
    cv_ll = np.zeros((n_states, nRunEM, n_folds))

    for iS, num_states in enumerate(states):
        print(f"evaluating blocked holdout for {num_states} state(s)")
        delayed_calls = [
            delayed(single_blocked_holdout_func)(
                observations,
                inputs,
                num_states,
                test_indices_list=test_indices_list,
                algorithm=algorithm,
                n_iter=1000,
                tol=1e-4,
                prior_alpha=prior_alpha,
                prior_sigma=prior_sigma,
            )
            for _ in range(nRunEM)
        ]
        run_results = Parallel(n_jobs=n_jobs)(delayed_calls)
        cv_ll[iS] = np.asarray(run_results)

    return cv_ll


def plot_cross_validation_scores(
    cv_log_likelihoods: np.ndarray,
    states: npt.NDArray[np.int64],
    figure_path: Path,
    sess_id_full: str,
):
    """Plot blocked within-session held-out log likelihood by state count.

    Parameters
    ----------
    cv_log_likelihoods : np.ndarray
        Held-out log likelihoods with shape `(n_states, n_restarts, n_folds)`.
    states : np.ndarray
        Candidate state counts with shape `(n_states,)`.
    figure_path : Path
        Directory where the plot is saved.
    sess_id_full : str
        Full session identifier used in the saved figure filename.

    Returns
    -------
    dict
        Dictionary containing `CV_log_likelihood` and `states`.
    """
    mean_ll = np.mean(cv_log_likelihoods, axis=(1, 2))
    sem_ll = np.std(cv_log_likelihoods, axis=(1, 2)) / np.sqrt(
        cv_log_likelihoods.shape[1] * cv_log_likelihoods.shape[2]
    )

    f, ax = plt.subplots(facecolor='w', edgecolor='k')
    cv_line = plt.plot(states, mean_ll, label="Blocked held-out LL")[0]
    cv_color = cv_line.get_color()
    plt.fill_between(states, mean_ll - sem_ll, mean_ll + sem_ll, alpha=0.3, color=cv_color)
    plt.xlabel("states")
    plt.xticks(states)
    plt.ylabel("held-out log likelihood per trial")
    plt.legend(loc="upper left", frameon=False)
    plt.title("Blocked held-out model selection", fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    save_path = figure_path / f'{sess_id_full}_trial_HMM_blocked_holdout_loglikelihood.png'
    plt.gcf().savefig(save_path, format='png', dpi=300)
    plt.show()

    return {'CV_log_likelihood': cv_log_likelihoods, 'states': states}


def run_cross_validation(trial_df: pd.DataFrame, session: Session, algorithm='MLE',
                         prior_alpha=1, prior_sigma=1,
                         min_states: int = 1, max_states: int = 5,
                         n_threads: int = 4, n_runs: int = 5, n_folds: int = 5,
                         predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS):
    """Run blocked within-session held-out scoring over hidden-state count.

    This is the secondary GLM-HMM model-selection path. The primary selector is
    MLE AIC/BIC fit independently within each session.
    """
    prepared = prepare_trial_glm_hmm_data(trial_df, predictor_columns=predictor_columns)
    observations = prepared["observations"]
    predictors = prepared["inputs"]

    states = np.arange(min_states, max_states + 1)
    cv_ll = calculate_blocked_holdout_scores(
        observations=observations,
        inputs=predictors,
        states=states,
        nRunEM=n_runs,
        n_folds=n_folds,
        n_jobs=n_threads,
        algorithm=algorithm,
        prior_alpha=prior_alpha,
        prior_sigma=prior_sigma,
    )
    return plot_cross_validation_scores(
        cv_ll,
        states,
        figure_path=session.figure_path,
        sess_id_full=session.sess_id_full,
    )


def run_information_criteria(trial_df: pd.DataFrame, session: Session, algorithm='MLE',
                             prior_alpha=1, prior_sigma=1,
                             predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
                             min_states: int = 1, max_states: int = 5,
                             n_threads: int = 4, n_runs: int = 10):
    """Run GLM-HMM AIC/BIC model selection for one session.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.
    session : Session
        Session metadata used by `plot_information_criteria`.
    algorithm : str, default='MLE'
        Fitting mode. Information criteria are currently restricted to 'MLE'.
    prior_alpha : float, default=1
        Transition prior concentration passed through to the scorer.
    prior_sigma : float, default=1
        Observation prior scale passed through to the scorer.
    predictor_columns : tuple[str, ...]
        Trial-level predictor columns used with an added bias column.
    min_states : int, default=1
        Smallest hidden-state count to evaluate.
    max_states : int, default=5
        Largest hidden-state count to evaluate, inclusive.
    n_threads : int, default=4
        Parallel jobs used across EM restarts.
    n_runs : int, default=10
        Number of EM restarts per state count.

    Returns
    -------
    dict
        Model-selection dictionary containing `AIC` and `BIC` arrays with shape
        `(n_state_counts, n_runs)`.
    """
    if algorithm.upper() != 'MLE':
        raise ValueError("GLM-HMM information criteria should only be run on MLE models.")

    prepared = prepare_trial_glm_hmm_data(trial_df, predictor_columns=predictor_columns)
    action = prepared["observations"]
    predictors = prepared["inputs"]

    states = np.arange(min_states,max_states+1)
    AIC, BIC = calculate_information_criteria(observations=action, inputs=predictors, states=states, algorithm='MLE',
                                              prior_alpha=prior_alpha, prior_sigma=prior_sigma,
                                              nRunEM=n_runs, n_jobs=n_threads)
    plot_information_criteria(
        AIC,
        BIC,
        states,
        figure_path=session.figure_path,
        sess_id_full=session.sess_id_full,
    )
    return {'AIC': AIC, 'BIC': BIC}


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
