"""
Blockwise LM-HMM using the code from Cazettes et al 2023 made by Lucca Mazzucato
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import re
from pathlib import Path
# from matplotlib.font_manager import weight_dict
from typing import Protocol
from joblib import Parallel, delayed
import ssm # note this should be the forked ssm repo
from src.behavior_analysis.project_utils import (
    is_present_value,
    spawn_child_seeds,
    temporary_numpy_seed,
)
from src.behavior_analysis.plotting_utils import build_presentation_colors, build_state_colormap, get_state_colors
from src.behavior_analysis import state_space_plotting


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


def make_valid_block_history_mask(block_df: pd.DataFrame) -> pd.Series:
    """Return blocks with current outcome and previous-block history.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with `trials_to_correct` and `prev_n_correct`
        columns. `trials_to_correct` marks whether the current block has a
        valid behavioral outcome. `prev_n_correct` is the canonical marker that
        previous-block history exists, even when a model uses another previous-
        block predictor such as `prev_n_rewarded`. Real NaN values and string
        missing-value sentinels are invalid.

    Returns
    -------
    pd.Series
        Boolean mask with shape `(n_blocks,)`, aligned to `block_df.index`.
    """
    has_current_block_outcome = is_present_value(block_df["trials_to_correct"])
    has_previous_block_history = is_present_value(block_df["prev_n_correct"])
    return has_current_block_outcome & has_previous_block_history


def prepare_block_lm_hmm_data(
    block_df: pd.DataFrame,
    predictor_columns: tuple[str, ...] = ("prev_n_rewarded",),
) -> dict[str, np.ndarray | list[str]]:
    """Build session-wise LM-HMM observations and predictors from block data.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe where rows correspond to task blocks.
        Required columns are `trials_to_correct`, `prev_n_correct`, and every
        column named in `predictor_columns`. `prev_n_correct` is used as the
        canonical previous-block-history marker even when it is not the chosen
        LM-HMM predictor.
    predictor_columns : tuple[str, ...], default=("prev_n_rewarded",)
        Predictor columns used as LM-HMM inputs. Each predictor is returned as
        one input dimension in the same order as provided here.

    Returns
    -------
    dict[str, np.ndarray | list[str]]
        Dictionary with:
        - `observations`: np.ndarray, shape (n_valid_blocks, 1), integer-valued
          `trials_to_correct`
        - `inputs`: np.ndarray, shape (n_valid_blocks, n_predictors), integer-
          valued predictor matrix
        - `valid_mask`: np.ndarray, shape (n_blocks,), boolean mask selecting
          rows with valid observations
        - `predictor_labels`: list[str], predictor names in input-column order
    """
    valid_mask = make_valid_block_history_mask(block_df)
    df = block_df[valid_mask]

    observations = df["trials_to_correct"].to_numpy().reshape(-1, 1).astype(int)
    predictor_arrays = []
    for predictor_name in predictor_columns:
        column = df[predictor_name].to_numpy().reshape(-1, 1)
        if predictor_name == "bias_full_flag":
            predictor_arrays.append((column == "True").astype(int))
        else:
            predictor_arrays.append(column.astype(int))

    inputs = np.concatenate(predictor_arrays, axis=1)
    return {
        "observations": observations,
        "inputs": inputs,
        "valid_mask": valid_mask.to_numpy(),
        "predictor_labels": list(predictor_columns),
    }


def normalize_lm_observation_parameters(
    recovered_weights: np.ndarray,
    recovered_mus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize LM-HMM observation parameters to stable array shapes.

    Parameters
    ----------
    recovered_weights : np.ndarray
        State-specific LM slopes returned by `ssm`, expected to describe
        `(num_states, obs_dim, input_dim)` but sometimes consumed downstream
        via shape-collapsing operations.
    recovered_mus : np.ndarray
        State-specific LM intercepts with expected shape `(num_states, obs_dim)`.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - normalized_weights: np.ndarray, shape `(num_states, obs_dim, input_dim)`
        - normalized_mus: np.ndarray, shape `(num_states, obs_dim)`
    """
    normalized_weights = np.asarray(recovered_weights, dtype=float)
    normalized_mus = np.asarray(recovered_mus, dtype=float)

    if normalized_weights.ndim == 1:
        normalized_weights = normalized_weights[np.newaxis, np.newaxis, :]
    elif normalized_weights.ndim == 2:
        normalized_weights = normalized_weights[:, np.newaxis, :]

    if normalized_mus.ndim == 0:
        normalized_mus = normalized_mus[np.newaxis, np.newaxis]
    elif normalized_mus.ndim == 1:
        normalized_mus = normalized_mus[:, np.newaxis]

    return normalized_weights, normalized_mus


def assign_inferred_states_to_blocks(
    block_df: pd.DataFrame,
    valid_mask: np.ndarray,
    most_likely_states: np.ndarray,
    column_name: str = "inferred_strategy",
) -> pd.DataFrame:
    """Assign inferred HMM states to valid block rows.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`.
    valid_mask : np.ndarray
        Boolean mask with shape `(n_blocks,)`, selecting rows used for HMM
        fitting.
    most_likely_states : np.ndarray
        Inferred state ids with shape `(n_valid_blocks,)`.
    column_name : str, default="inferred_strategy"
        Destination column for the assigned states.

    Returns
    -------
    pd.DataFrame
        `block_df` with `column_name` assigned. Invalid rows are set to string
        `"None"` for CSV compatibility.
    """
    inferred_states = np.zeros(block_df.shape[0], dtype='object')
    inferred_states[valid_mask] = most_likely_states
    inferred_states[~valid_mask] = 'None'
    block_df[column_name] = inferred_states
    return block_df


def add_cur_strategy_slope(
    block_df: pd.DataFrame,
    valid_mask: np.ndarray,
    inferred_states: np.ndarray,
    primary_predictor_weights: np.ndarray,
) -> pd.DataFrame:
    """Assign the current inferred strategy slope to each block.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`.
    valid_mask : np.ndarray
        Boolean mask with shape `(n_blocks,)`, selecting rows used for HMM
        fitting.
    inferred_states : np.ndarray
        Inferred state ids aligned to `block_df`, shape `(n_blocks,)`. Invalid
        rows may contain string `"None"`.
    primary_predictor_weights : np.ndarray
        State-specific primary predictor weights with shape `(num_states,)`.

    Returns
    -------
    pd.DataFrame
        `block_df` with `cur_strategy_slope` assigned. For multi-state models,
        invalid rows receive string `"None"` for CSV compatibility. For
        one-state models, every row receives the single state slope, matching
        historical behavior.
    """
    if primary_predictor_weights.size == 1:
        block_df['cur_strategy_slope'] = (
            np.ones(block_df.shape[0]) * primary_predictor_weights[0]
        )
        return block_df

    cur_strategy_slope = np.zeros(block_df.shape[0], dtype='object')
    cur_strategy_slope[valid_mask] = primary_predictor_weights[inferred_states[valid_mask].astype(int)]
    cur_strategy_slope[~valid_mask] = 'None'
    block_df['cur_strategy_slope'] = cur_strategy_slope
    return block_df


def split_blocked_holdout_sequences(
    observations: np.ndarray,
    inputs: np.ndarray,
    test_indices: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Split one session into disjoint training sequences and one held-out block.

    Parameters
    ----------
    observations : np.ndarray
        Observation array with shape (T, 1), where rows index consecutive blocks.
    inputs : np.ndarray
        Predictor array with shape (T, M), aligned row-wise to `observations`.
    test_indices : np.ndarray
        Integer indices of the contiguous held-out block, shape (K,).

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


def mle_block_states(block_df: pd.DataFrame, figure_path: Path, sess_id_tag: str, plot: bool=False, model_dict=None,
                     num_states=2, random_seed: int | None = None):
    """Fit MLE block LM-HMM states and assign them back to the block table.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        columns are defined by `prepare_block_lm_hmm_data`.
    figure_path : Path
        Directory where diagnostic figures are saved when `plot=True`.
    sess_id_tag : str
        Caller-provided session identifier tag used in saved figure filenames.
    plot : bool, default=False
        Whether to save diagnostic fit and state-summary figures.
    model_dict : dict or None, default=None
        Existing model dictionary to update. If None, a new dictionary is used.
    num_states : int, default=2
        Number of hidden LM-HMM states.
    random_seed : int or None, default=None
        Seed used for stochastic HMM construction and EM initialization. None
        preserves the current random behavior.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Updated model dictionary and `block_df` with inferred state columns.
    """
    prepared = prepare_block_lm_hmm_data(block_df)
    ix_valid = prepared["valid_mask"]
    trials_to_correct = prepared["observations"]
    predictors = prepared["inputs"]

    obs_dim = trials_to_correct.shape[1]
    input_dim = predictors.shape[1]
    state_colors = get_state_colors(num_states)
    state_cmap = build_state_colormap(state_colors)

    if model_dict is None:
        model_dict = {}
    model_dict['mle'] = {}
    model_dict['mle']['random_seed'] = random_seed
    weight_dict = {}

    with temporary_numpy_seed(random_seed):
        mle_hmm = build_input_driven_hmm(
            num_states=num_states,
            obs_dim=obs_dim,
            input_dim=input_dim,
            algorithm='MLE',
        )
        N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
        fit_log_likelihood = mle_hmm.fit(trials_to_correct, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['mle']['fit_log_likelihood'] = fit_log_likelihood
    if plot:
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_log_likelihood, label="EM")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.title("MLE EM fit of observed data")
        save_path = figure_path / '{}_mle_convergence.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = mle_hmm.most_likely_states(trials_to_correct, input=predictors)
    recovered_weights = mle_hmm.observations.Wks
    recovered_mus = mle_hmm.observations.mus
    normalized_weights, normalized_mus = normalize_lm_observation_parameters(
        recovered_weights=recovered_weights,
        recovered_mus=recovered_mus,
    )

    weight_dict['weights'] = normalized_weights
    weight_dict['mus'] = normalized_mus
    weight_dict['label'] = 'mle'

    model_dict['mle']['weight_dict'] = weight_dict
    if plot:
        mle_transition_mat = np.exp(mle_hmm.transitions.log_Ps)
        fig, ax = state_space_plotting.plot_transition_matrix(mle_transition_mat)
        ax.set_title("MLE transition matrix", fontsize=15)
        save_path = figure_path / '{}_mle_transition_mat.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

    posterior_probs = mle_hmm.expected_states(data=trials_to_correct, input=predictors)[0]
    if plot:
        fig, ax = state_space_plotting.plot_block_lm_hmm_state_summary(
            posterior_probs,
            trials_to_correct,
            predictors,
            mle_hmm,
            state_colors,
            state_cmap,
        )
        ax[0].set_title("MLE HMM states")
        save_path = figure_path / '{}_mle_predicted_states.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['mle']['posterior_probs'] = posterior_probs
    model_dict['mle']['inferred_state_list'] = inferred_state_list
    model_dict['mle']['inferred_durations'] = inferred_durations
    model_dict['mle']['hmm_z'] = most_likely_states
    model_dict['mle']['hmm'] = mle_hmm

    block_df = assign_inferred_states_to_blocks(block_df, ix_valid, most_likely_states)

    if plot:
        fig, ax = state_space_plotting.plot_state_duration_histogram(
            inferred_state_list=inferred_state_list,
            inferred_durations=inferred_durations,
            num_states=num_states,
            state_colors=state_colors,
        )
        save_path = figure_path / f'{sess_id_tag}_mle_state_durations.png'
        fig.savefig(save_path, format='png', dpi=300)

    return model_dict, block_df


def map_block_states(block_df: pd.DataFrame, figure_path: Path, sess_id_tag: str, plot: bool = False,
                     num_states=2, prior_sigma=1, prior_alpha=1, model_dict=None,
                     random_seed: int | None = None):
    """Fit MAP block LM-HMM states and assign them back to the block table.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        columns are defined by `prepare_block_lm_hmm_data`.
    figure_path : Path
        Directory where diagnostic figures are saved when `plot=True`.
    sess_id_tag : str
        Caller-provided session identifier tag used in saved figure filenames.
    plot : bool, default=False
        Whether to save diagnostic fit and state-summary figures.
    num_states : int, default=2
        Number of hidden LM-HMM states.
    prior_sigma : float, default=1
        Observation prior scale passed to the MAP HMM.
    prior_alpha : float, default=1
        Sticky-transition concentration passed to the MAP HMM.
    model_dict : dict or None, default=None
        Existing model dictionary to update. If None, a new dictionary is used.
    random_seed : int or None, default=None
        Seed used for stochastic HMM construction and EM initialization. None
        preserves the current random behavior.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Updated model dictionary and `block_df` with inferred state columns.
    """
    prepared = prepare_block_lm_hmm_data(block_df)
    ix_valid = prepared["valid_mask"]
    trials_to_correct = prepared["observations"]
    predictors = prepared["inputs"]
    pred_labels = prepared["predictor_labels"]

    obs_dim = trials_to_correct.shape[1]
    input_dim = predictors.shape[1]

    if model_dict is None:
        model_dict = {}

    model_dict['map'] = {}
    model_dict['map']['random_seed'] = random_seed
    weight_dict = {}

    with temporary_numpy_seed(random_seed):
        map_hmm = build_input_driven_hmm(
            num_states=num_states,
            obs_dim=obs_dim,
            input_dim=input_dim,
            algorithm='MAP',
            prior_alpha=prior_alpha,
            prior_sigma=prior_sigma,
        )

        # maximum number of EM iterations.
        # was 10k before, should probably stop by 1k. If it doesn't work by then it may not be realistic/result in overfitting
        N_iters = 1000

        fit_log_likelihood = map_hmm.fit(trials_to_correct, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['map']['fit_log_likelihood'] = fit_log_likelihood

    if plot:
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_log_likelihood, label="EM")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.title("MAP EM fit of observed data")
        save_path = figure_path / '{}_map_convergence.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

    most_likely_states = map_hmm.most_likely_states(trials_to_correct, input=predictors)
    recovered_weights = map_hmm.observations.Wks
    recovered_mus = map_hmm.observations.mus
    normalized_weights, normalized_mus = normalize_lm_observation_parameters(
        recovered_weights=recovered_weights,
        recovered_mus=recovered_mus,
    )

    weight_dict['weights'] = normalized_weights
    weight_dict['mus'] = normalized_mus
    weight_dict['label'] = 'map'
    weight_dict['weight_labels'] = pred_labels
    model_dict['map']['weight_dict'] = weight_dict
    if plot:
        map_transition_mat = np.exp(map_hmm.transitions.log_Ps)
        fig, ax = state_space_plotting.plot_transition_matrix(map_transition_mat)
        ax.set_title("MAP transition matrix", fontsize=15)
        save_path = figure_path / '{}_map_transition_mat.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

    state_weights = normalized_weights[:, 0, :]
    primary_predictor_weights = state_weights[:, 0]
    present_colors, present_cmap = build_presentation_colors(primary_predictor_weights)

    posterior_probs = map_hmm.expected_states(data=trials_to_correct, input=predictors)[0]
    if plot:
        fig, ax = state_space_plotting.plot_block_lm_hmm_presentation_summary(
            posterior_probs,
            trials_to_correct,
            predictors,
            map_hmm,
            present_colors,
            present_cmap,
            predictor_labels=pred_labels,
        )
        ax[0].set_title("MAP HMM states")
        save_path = figure_path / '{}_map_predicted_states.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['map']['posterior_probs'] = posterior_probs
    model_dict['map']['inferred_state_list'] = inferred_state_list
    model_dict['map']['inferred_durations'] = inferred_durations
    model_dict['map']['hmm_z'] = most_likely_states
    model_dict['map']['hmm'] = map_hmm

    block_df = assign_inferred_states_to_blocks(block_df, ix_valid, most_likely_states)
    inferred_states = block_df['inferred_strategy'].to_numpy()
    if plot:
        fig, ax = state_space_plotting.plot_state_duration_histogram(
            inferred_state_list=inferred_state_list,
            inferred_durations=inferred_durations,
            num_states=num_states,
            state_colors=present_colors,
        )
        save_path = figure_path / f'{sess_id_tag}_map_state_durations.png'
        fig.savefig(save_path, format='png', dpi=300)

    block_df = add_cur_strategy_slope(
        block_df,
        valid_mask=ix_valid,
        inferred_states=inferred_states,
        primary_predictor_weights=primary_predictor_weights,
    )

    return model_dict, block_df


def save_block_model_dict(model_dict: dict, processed_data_path: Path, sess_id_full: str) -> Path:
    """Save a fitted block LM-HMM model dictionary.

    Parameters
    ----------
    model_dict : dict
        Model output dictionary containing fitted block-model results. Expected
        top-level keys are workflow-dependent, commonly `mle` and `map`.
    processed_data_path : Path
        Directory where the block model pickle should be saved.
    sess_id_full : str
        Full session identifier used in the saved filename.

    Returns
    -------
    Path
        Pickle path with filename `{sess_id_full}_block_statedict.pkl`.
    """
    save_path = processed_data_path / f"{sess_id_full}_block_statedict.pkl"
    with open(save_path, "wb") as file:
        pkl.dump(model_dict, file)
    return save_path


def run_block_modeling(block_performance: pd.DataFrame, augmented_trial_df: pd.DataFrame, session: Session, num_states: int=2,
                       prior_alpha=1, prior_sigma=1, random_seed: int | None = None):
    mle_seed, map_seed = spawn_child_seeds(random_seed, 2)

    mle_model_dict, block_performance = mle_block_states(
        block_performance,
        figure_path=session.figure_path,
        sess_id_tag=session.sess_id_full,
        plot=True,
        num_states=num_states,
        random_seed=mle_seed,
    )
    map_model_dict, block_performance = map_block_states(
        block_performance,
        figure_path=session.figure_path,
        sess_id_tag=session.sess_id_full,
        plot=True,
        model_dict=mle_model_dict,
        num_states=num_states,
        prior_alpha=prior_alpha,
        prior_sigma=prior_sigma,
        random_seed=map_seed,
    )
    save_block_model_dict(
        map_model_dict,
        processed_data_path=session.processed_data_path,
        sess_id_full=session.sess_id_full,
    )

    block_performance = hardcode_block_strategy(block_performance)  # These are kept for inspection, they are NOT used in modeling or passed down to trials
    block_performance.to_csv(session.processed_data_path / (session.sess_id_full + '_block_performance.csv'), index=False)

    augmented_trial_df = trials_inherit_strategy(block_performance, augmented_trial_df)
    augmented_trial_df.to_csv(session.processed_data_path / (session.sess_id_full + '_augmented_trials.csv'), index=False)

    normalized_weights, _ = normalize_lm_observation_parameters(
        recovered_weights=map_model_dict['map']['weight_dict']['weights'],
        recovered_mus=map_model_dict['map']['weight_dict']['mus'],
    )
    primary_predictor_weights = normalized_weights[:, 0, 0]
    present_colors, present_cmap = build_presentation_colors(primary_predictor_weights)

    fig, ax = state_space_plotting.plot_block_lm_hmm_weight_comparison(
        [map_model_dict['mle']['weight_dict'], map_model_dict['map']['weight_dict']],
    )
    save_path = session.figure_path / f'{session.sess_id_full}_hmm_weight_comparison.png'
    fig.savefig(save_path, format='png', dpi=300)

    fig, ax = state_space_plotting.plot_block_lm_hmm_weights(
        map_model_dict['map']['weight_dict'],
        present_colors,
    )
    save_path = session.figure_path / f'{session.sess_id_full}_hmm_weights.png'
    fig.savefig(save_path, format='png', dpi=300)
    return block_performance, augmented_trial_df


def build_input_driven_hmm(num_states: int, obs_dim: int, input_dim: int,
                           algorithm: str = 'MLE', prior_alpha: float = 1, prior_sigma: float = 1):
    """Build an input-driven LM-HMM using either MLE or MAP settings."""
    algorithm = algorithm.upper()
    if algorithm == 'MLE':
        return ssm.HMM(num_states, obs_dim, M=input_dim,
                       observations="input_driven_obs_gaussian", transitions="standard")
    if algorithm == 'MAP':
        return ssm.HMM(num_states, obs_dim, M=input_dim,
                       observations="input_driven_obs_gaussian",
                       observation_kwargs=dict(prior_sigma=prior_sigma),
                       transitions="sticky", transition_kwargs=dict(alpha=prior_alpha, kappa=0))
    raise ValueError(f"algorithm must be 'MLE' or 'MAP', got {algorithm}")


def fit_lm_hmm_and_score_log_likelihood(observations: np.ndarray, inputs: np.ndarray, num_states: int,
                                         algorithm: str = 'MLE', n_iter: int = 1000, tol: float = 10**-4,
                                         prior_alpha=1, prior_sigma=1,
                                         random_seed: int | None = None):
    """Fit one LM-HMM restart and return its training log likelihood.

    Parameters
    ----------
    observations : np.ndarray
        Observation matrix with shape `(n_blocks, observation_dimensions)`.
    inputs : np.ndarray
        Predictor matrix with shape `(n_blocks, input_dimensions)`.
    num_states : int
        Number of hidden states to fit.
    algorithm : str, default='MLE'
        Fitting mode passed to `build_input_driven_hmm`; either 'MLE' or 'MAP'.
    n_iter : int, default=1000
        Maximum EM iterations.
    tol : float, default=1e-4
        EM convergence tolerance.
    prior_alpha : float, default=1
        Sticky-transition prior concentration for MAP fits.
    prior_sigma : float, default=1
        Observation prior scale for MAP fits.
    random_seed : int or None, default=None
        Seed used for stochastic HMM construction and EM initialization. None
        preserves the current random behavior.

    Returns
    -------
    float
        Fitted model log likelihood on `observations`, in log-probability units.
    """
    obs_dim, input_dim = observations.shape[1], inputs.shape[1]
    with temporary_numpy_seed(random_seed):
        hmm = build_input_driven_hmm(num_states=num_states, obs_dim=obs_dim, input_dim=input_dim,
                                     algorithm=algorithm, prior_alpha=prior_alpha, prior_sigma=prior_sigma)
        hmm.fit(observations, inputs=inputs, method="em", num_iters=n_iter, tolerance=tol)
    out = hmm.log_likelihood(observations, inputs=inputs)
    return out


def build_blocked_holdout_indices(n_timesteps: int, n_folds: int = 5) -> list[np.ndarray]:
    """Create contiguous within-session held-out blocks.

    Parameters
    ----------
    n_timesteps : int
        Number of valid LM-HMM blocks in the session.
    n_folds : int, default=5
        Number of contiguous held-out blocks.

    Returns
    -------
    list[np.ndarray]
        List of contiguous test-index arrays. Each array indexes one held-out
        block along the session time axis.
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
    random_seed: int | None = None,
):
    """Score one LM-HMM state count on blocked within-session held-out splits.

    Parameters
    ----------
    observations : np.ndarray
        Observation array with shape (T, 1), where T is the number of valid
        session blocks and the single column is `trials_to_correct`.
    inputs : np.ndarray
        Predictor array with shape (T, M), aligned row-wise to `observations`.
    num_states : int
        Number of hidden LM-HMM states to fit.
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
    random_seed : int or None, default=None
        Base seed used to derive one deterministic seed per held-out fold. None
        preserves the current random behavior.

    Returns
    -------
    np.ndarray
        Held-out log-likelihood per block, shape (n_folds,). Each score is
        normalized by the number of held-out observations in that block.
    """
    fold_log_likelihoods = np.zeros(len(test_indices_list))
    obs_dim, input_dim = observations.shape[1], inputs.shape[1]
    fold_random_seeds = spawn_child_seeds(random_seed, len(test_indices_list))

    for i_fold, (test_idx, fold_random_seed) in enumerate(zip(test_indices_list, fold_random_seeds)):
        train_observations, train_inputs, test_observations, test_inputs = split_blocked_holdout_sequences(
            observations=observations,
            inputs=inputs,
            test_indices=test_idx,
        )
        with temporary_numpy_seed(fold_random_seed):
            hmm = build_input_driven_hmm(
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


def calculate_blocked_holdout_scores(observations: np.ndarray, inputs: np.ndarray, states: np.ndarray,
                                     nRunEM: int = 5, n_folds: int = 5, n_jobs: int = 4,
                                     algorithm: str = 'MLE', prior_alpha: float = 1, prior_sigma: float = 1,
                                     random_seed: int | None = None,
                                     restart_random_seeds: list[int | None] | None = None):
    """Compute blocked within-session held-out log-likelihoods across state counts.

    Parameters
    ----------
    observations : np.ndarray
        Observation array with shape (T, 1), where rows index valid session
        blocks and values are `trials_to_correct`.
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
    random_seed : int or None, default=None
        Base seed used when `restart_random_seeds` is not provided.
    restart_random_seeds : list[int or None] or None, default=None
        Flat list with one seed per `(state, restart)` combination.

    Returns
    -------
    np.ndarray
        Held-out log-likelihood per observation, shape
        (n_states, n_restarts, n_folds).
    """
    n_states = states.size
    test_indices_list = build_blocked_holdout_indices(observations.shape[0], n_folds=n_folds)
    cv_ll = np.zeros((n_states, nRunEM, n_folds))
    if restart_random_seeds is None:
        restart_random_seeds = spawn_child_seeds(random_seed, n_states * nRunEM)
    if len(restart_random_seeds) != n_states * nRunEM:
        raise ValueError(
            f"Expected {n_states * nRunEM} restart seeds, got {len(restart_random_seeds)}"
        )
    restart_seed_grid = np.asarray(restart_random_seeds, dtype=object).reshape(n_states, nRunEM)

    for iS, num_states in enumerate(states):
        print(f"evaluating blocked holdout for {num_states} state(s)")
        delayed_calls = [
            delayed(single_blocked_holdout_func)(
                observations, inputs, num_states,
                test_indices_list=test_indices_list,
                algorithm=algorithm,
                n_iter=1000,
                tol=1e-4,
                prior_alpha=prior_alpha,
                prior_sigma=prior_sigma,
                random_seed=restart_seed_grid[iS, iRun],
            )
            for iRun in range(nRunEM)
        ]
        run_results = Parallel(n_jobs=n_jobs)(delayed_calls)
        cv_ll[iS] = np.asarray(run_results)

    return cv_ll


def plot_cross_validation_scores(
    cv_log_likelihoods: np.ndarray,
    states: np.ndarray,
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
    sem_ll = np.std(cv_log_likelihoods, axis=(1, 2)) / np.sqrt(cv_log_likelihoods.shape[1] * cv_log_likelihoods.shape[2])

    f, ax = plt.subplots(facecolor='w', edgecolor='k')
    cv_line = plt.plot(states, mean_ll, label="Cross-validated LL")[0]
    cv_color = cv_line.get_color()
    plt.fill_between(states, mean_ll - sem_ll, mean_ll + sem_ll, alpha=0.3, color=cv_color)
    plt.xlabel("states")
    plt.xticks(states)
    plt.ylabel("held-out log likelihood")
    plt.legend(loc="upper left", frameon=False)
    plt.title("Blocked held-out model selection", fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    save_path = figure_path / f'{sess_id_full}_block_HMM_CV_loglikelihood.png'
    plt.gcf().savefig(save_path, format='png', dpi=300)
    plt.show()

    return {'CV_log_likelihood': cv_log_likelihoods, 'states': states}


def run_cross_validation(block_performance: pd.DataFrame, session: Session, algorithm: str = 'MLE',
                         prior_alpha: float = 1, prior_sigma: float = 1,
                         min_states: int = 1, max_states: int = 5,
                         n_threads: int = 4, n_runs: int = 5, n_folds: int = 5,
                         random_seed: int | None = None):
    """Run blocked within-session held-out scoring over hidden-state count.

    This is the secondary LM-HMM model-selection path. The primary selector is
    MLE AIC/BIC fit independently within each session.
    """
    prepared = prepare_block_lm_hmm_data(block_performance)
    trials_to_correct = prepared["observations"]
    predictors = prepared["inputs"]

    states = np.arange(min_states, max_states + 1)
    restart_random_seeds = spawn_child_seeds(random_seed, states.size * n_runs)
    cv_ll = calculate_blocked_holdout_scores(observations=trials_to_correct, inputs=predictors, states=states,
                                             nRunEM=n_runs, n_folds=n_folds, n_jobs=n_threads,
                                             algorithm=algorithm, prior_alpha=prior_alpha, prior_sigma=prior_sigma,
                                             random_seed=random_seed,
                                             restart_random_seeds=restart_random_seeds)
    model_selection = plot_cross_validation_scores(
        cv_ll,
        states,
        figure_path=session.figure_path,
        sess_id_full=session.sess_id_full,
    )
    model_selection["random_seed"] = random_seed
    model_selection["restart_random_seeds"] = np.asarray(
        restart_random_seeds,
        dtype=object,
    ).reshape(states.size, n_runs)
    return model_selection


def run_information_criteria(block_performance: pd.DataFrame, session: Session, algorithm: str = 'MLE',
                             prior_alpha: float = 1, prior_sigma: float = 1,
                             min_states: int = 1, max_states: int = 5,
                             n_threads: int = 4, n_runs: int = 10,
                             random_seed: int | None = None):
    """Run LM-HMM AIC/BIC model selection for one session.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`.
    session : Session
        Session metadata providing `figure_path` and `sess_id_full`.
    algorithm : str, default='MLE'
        Fitting mode. Information criteria are currently restricted to 'MLE'.
    prior_alpha : float, default=1
        Transition prior concentration passed through to the scorer.
    prior_sigma : float, default=1
        Observation prior scale passed through to the scorer.
    min_states : int, default=1
        Smallest hidden-state count to evaluate.
    max_states : int, default=5
        Largest hidden-state count to evaluate, inclusive.
    n_threads : int, default=4
        Parallel jobs used across EM restarts.
    n_runs : int, default=10
        Number of EM restarts per state count.
    random_seed : int or None, default=None
        Base seed used to derive one deterministic seed per EM restart. None
        preserves the current random behavior.

    Returns
    -------
    dict
        Model-selection dictionary containing `AIC` and `BIC` arrays with shape
        `(n_state_counts, n_runs)`.
    """
    if algorithm.upper() != 'MLE':
        raise ValueError("LM-HMM information criteria should only be run on MLE models.")

    prepared = prepare_block_lm_hmm_data(block_performance)
    trials_to_correct = prepared["observations"]
    predictors = prepared["inputs"]

    states = np.arange(min_states,max_states+1)
    restart_random_seeds = spawn_child_seeds(random_seed, states.size * n_runs)
    AIC, BIC = calculate_information_criteria(observations=trials_to_correct, inputs=predictors, states=states,
                                              nRunEM=n_runs, n_jobs=n_threads, algorithm=algorithm,
                                              prior_alpha=prior_alpha, prior_sigma=prior_sigma,
                                              random_seed=random_seed,
                                              restart_random_seeds=restart_random_seeds)

    plot_information_criteria(
        AIC,
        BIC,
        states,
        figure_path=session.figure_path,
        sess_id_full=session.sess_id_full,
    )
    return {
        'AIC': AIC,
        'BIC': BIC,
        'random_seed': random_seed,
        'restart_random_seeds': np.asarray(restart_random_seeds, dtype=object).reshape(states.size, n_runs),
    }


def count_lm_hmm_parameters(num_states: int, obs_dim: int, input_dim: int) -> int:
    """Count free parameters for block LM-HMM information criteria.

    Parameters
    ----------
    num_states : int
        Number of hidden LM-HMM states.
    obs_dim : int
        Observation dimensionality.
    input_dim : int
        Predictor dimensionality.

    Returns
    -------
    int
        Total free-parameter count used in AIC/BIC calculations.
    """
    transition_params = (num_states + 1) * (num_states - 1)
    observation_params = num_states * (obs_dim * input_dim + 2 * obs_dim)
    return transition_params + observation_params


def calculate_information_criteria(observations: np.ndarray, inputs: np.ndarray, states: np.ndarray, nRunEM: int, n_jobs: int,
                                   algorithm: str = 'MLE', prior_alpha: float = 1, prior_sigma: float = 1,
                                   random_seed: int | None = None,
                                   restart_random_seeds: list[int | None] | None = None):
    if algorithm.upper() != 'MLE':
        raise ValueError("LM-HMM information criteria should only be run on MLE models.")

    obs_dim = observations.shape[1]
    n_timesteps = observations.shape[0]
    input_dim = inputs.shape[1]
    n_states = states.size

    AIC = np.zeros((n_states, nRunEM))
    BIC = np.zeros((n_states, nRunEM))
    if restart_random_seeds is None:
        restart_random_seeds = spawn_child_seeds(random_seed, n_states * nRunEM)
    if len(restart_random_seeds) != n_states * nRunEM:
        raise ValueError(
            f"Expected {n_states * nRunEM} restart seeds, got {len(restart_random_seeds)}"
        )
    restart_seed_grid = np.asarray(restart_random_seeds, dtype=object).reshape(n_states, nRunEM)

    for iS, num_states in enumerate(states):#range(2, n + 1)):
        print("running {} state(s)".format(num_states))

        K = count_lm_hmm_parameters(num_states=num_states, obs_dim=obs_dim, input_dim=input_dim)
        delayed_calls = [
            delayed(fit_lm_hmm_and_score_log_likelihood)(
                observations, inputs, num_states,
                algorithm=algorithm,
                n_iter=1000,
                tol=1e-4,
                prior_alpha=prior_alpha,
                prior_sigma=prior_sigma,
                random_seed=restart_seed_grid[iS, iRun],
            )
            for iRun in range(nRunEM)
        ]
        results = Parallel(n_jobs=n_jobs)(delayed_calls)

        for iRun in range(nRunEM):
            AIC[iS, iRun] = K * 2 - 2 * results[iRun]
            BIC[iS, iRun] = K * np.log(n_timesteps) - 2 * results[iRun]

    return AIC, BIC


def plot_information_criteria(aic, bic, states, figure_path: Path, sess_id_full: str):
    """Plot LM-HMM AIC/BIC scores by state count.

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
    plt.xlabel("states")
    plt.xticks(states)
    plt.ylabel("criterion")
    plt.legend(loc="upper left", frameon=False)
    plt.title("BIC and AIC", fontsize=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    save_path = figure_path / '{}_block_HMM_AIC_BIC.png'.format(sess_id_full)
    plt.tight_layout()
    plt.gcf().savefig(save_path, format='png', dpi=300)
    print("saved AIC/BIC plot to {}".format(save_path))


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
    else:
        print("Double-check the session name!")
        return

    augmented_trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), sep=',',
                                     na_filter=False)
    block_performance = pd.read_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), sep=',',
                                    na_filter=False)

    mle_model_dict, block_performance = mle_block_states(
        block_performance,
        figure_path=figure_path,
        sess_id_tag=sess_id_full,
        plot=True,
    )

    block_performance = hardcode_block_strategy(block_performance)
    block_performance.to_csv(processed_data_path / (sess_id_full + '_block_performance.csv'), index=False)

    augmented_trial_df = trials_inherit_strategy(block_performance, augmented_trial_df)
    augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)


def hardcode_block_strategy(block_df: pd.DataFrame, inference_cutoff=.5) -> pd.DataFrame:
    """Hardcode block strategy labels for inspection.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with `trials_to_correct`, `prev_n_correct`, and
        `cur_strategy_slope` columns. Labels are assigned only to blocks with a
        valid current outcome and previous-block history.
    inference_cutoff : float, default=.5
        Strategy-slope cutoff below which blocks are labeled `Inference`; values
        greater than or equal to the cutoff are labeled `Qlearning`.

    Returns
    -------
    pd.DataFrame
        Copy-like reference to `block_df` with `hardcoded_strategy` added.
        Hardcoded strategies are for inspection and comparison to HMM inferred
        strategies, not for trial-level inheritance or model fitting.
    """
    ix_valid = make_valid_block_history_mask(block_df)
    df = block_df[ix_valid]

    strategy = np.zeros(df.shape[0], dtype='object')
    rl_ix = df['cur_strategy_slope'].to_numpy() >= inference_cutoff
    inference_ix = df['cur_strategy_slope'].to_numpy() < inference_cutoff
    strategy[inference_ix] = 'Inference'
    strategy[rl_ix] = 'Qlearning'

    hardcoded_strategy = np.zeros(block_df.shape[0], dtype='object')
    hardcoded_strategy[ix_valid] = strategy
    hardcoded_strategy[~ix_valid] = 'None'
    block_df['hardcoded_strategy'] = hardcoded_strategy
    return block_df


def trials_inherit_strategy(block_df: pd.DataFrame, trial_df: pd.DataFrame) -> pd.DataFrame:
    """Inherit HMM block states and block bias down to the trial level.

    Parameters
    ----------
    block_df : pd.DataFrame
        Blockwise dataframe with `block_ix`, `inferred_strategy`, and
        `bias_full_flag` columns. `block_ix` values identify task blocks and do
        not need to match dataframe row positions.
    trial_df : pd.DataFrame
        Trialwise dataframe with `cur_block`, one block id per trial.

    Returns
    -------
    pd.DataFrame
        Copy of `trial_df` with `inherited_block_strategy` and
        `inherited_block_bias` columns. Trials with no matching block receive
        the string `"None"` in both inherited columns.
    """
    strategy_by_block = dict(zip(block_df["block_ix"], block_df["inferred_strategy"]))
    bias_by_block = dict(zip(block_df["block_ix"], block_df["bias_full_flag"]))

    trial_df["inherited_block_strategy"] = (
        trial_df["cur_block"].map(strategy_by_block).fillna("None")
    )
    trial_df["inherited_block_bias"] = (
        trial_df["cur_block"].map(bias_by_block).fillna("None")
    )
    return trial_df


if __name__ == '__main__':
    main()
