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
from src.behavior_analysis.project_utils import (
    is_present_value,
    is_zero_flag,
    spawn_child_seeds,
    temporary_numpy_seed,
)
from src.behavior_analysis.plotting_utils import (
    build_presentation_colors,
    build_state_colormap,
    get_state_colors,
)
from src.behavior_analysis import state_space_plotting


TRIAL_GLM_PREDICTOR_LABELS = {
    "FQlearning_rel_value": "FQlearning",
    # "HMM_rel_value_logodds": "HMM",
    "HMM_rel_value_logodds_decay": "HMM_decay",
    "relative_doubt_index": "doubt",
    "perseveration_regressor": "perseveration",
    "time_to_choice": "time_to_choice",
}
DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS = (
    "FQlearning_rel_value",
    "HMM_rel_value_logodds_decay",
    "relative_doubt_index",
    "perseveration_regressor",
)

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


def make_valid_trial_glm_hmm_mask(
    trial_df: pd.DataFrame,
    predictor_columns: tuple[str, ...],
    require_inherited_strategy: bool = False,
) -> pd.Series:
    """Return trial rows that can be cast into GLM-HMM arrays.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are `prev_action`, `give_reward`, `action`, and each selected
        predictor column. If `require_inherited_strategy=True`,
        `inherited_block_strategy` is also required.
    predictor_columns : tuple[str, ...]
        Non-bias predictor columns that will be converted to the GLM-HMM input
        matrix.
    require_inherited_strategy : bool, default=False
        Whether rows must have a present inherited block strategy.

    Returns
    -------
    pd.Series
        Boolean mask with shape `(n_trials,)`, aligned to `trial_df.index`.
    """
    valid_mask = (
        is_present_value(trial_df["prev_action"])
        & is_zero_flag(trial_df["give_reward"])
        & is_present_value(trial_df["action"])
    )
    for predictor_name in predictor_columns:
        valid_mask = valid_mask & is_present_value(trial_df[predictor_name])
    if require_inherited_strategy:
        valid_mask = valid_mask & is_present_value(trial_df["inherited_block_strategy"])
    return valid_mask


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

    valid_mask = make_valid_trial_glm_hmm_mask(
        trial_df,
        predictor_columns=predictor_columns,
        require_inherited_strategy=require_inherited_strategy,
    )
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


def assign_inferred_states_to_trials(
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray,
    most_likely_states: np.ndarray,
    column_name: str = "inferred_strategy",
) -> pd.DataFrame:
    """Assign inferred GLM-HMM states to valid trial rows.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.
    valid_mask : np.ndarray
        Boolean mask with shape `(n_trials,)`, selecting rows used for HMM
        fitting.
    most_likely_states : np.ndarray
        Inferred state ids with shape `(n_valid_trials,)`.
    column_name : str, default="inferred_strategy"
        Destination column for the assigned states.

    Returns
    -------
    pd.DataFrame
        `trial_df` with `column_name` assigned. Invalid rows are set to string
        `"None"` for CSV compatibility.
    """
    inferred_states = np.zeros(trial_df.shape[0], dtype='object')
    inferred_states[valid_mask] = most_likely_states
    inferred_states[~valid_mask] = 'None'
    trial_df[column_name] = inferred_states
    return trial_df


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


def get_session_boundary_markers(
    trial_df: pd.DataFrame,
    valid_mask: np.ndarray,
    session_column: str = "source_date",
) -> tuple[np.ndarray, np.ndarray]:
    """Return session-boundary positions and labels for filtered trial plots.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Multisession
        tables may include `session_column`, typically `source_date`.
    valid_mask : np.ndarray
        Boolean mask with shape `(n_trials,)` selecting rows included in the
        plotted GLM-HMM posterior arrays.
    session_column : str, default="source_date"
        Column containing session labels. Boundaries are labeled with the
        session value that starts after the boundary.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - boundary positions in filtered GLM-HMM trial coordinates, shape
          `(n_boundaries,)`
        - boundary labels, shape `(n_boundaries,)`
    """
    if session_column not in trial_df.columns:
        return np.array([], dtype=int), np.array([], dtype=object)

    valid_mask = np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape[0] != trial_df.shape[0]:
        raise ValueError("valid_mask must have one entry per trial_df row.")

    valid_session_labels = trial_df.loc[valid_mask, session_column].to_numpy(dtype=object)
    if valid_session_labels.size <= 1:
        return np.array([], dtype=int), np.array([], dtype=object)

    boundary_positions = np.flatnonzero(valid_session_labels[1:] != valid_session_labels[:-1]) + 1
    boundary_labels = valid_session_labels[boundary_positions]
    return boundary_positions.astype(int), boundary_labels


def mle_trial_states(trial_df: pd.DataFrame, figure_path: Path, sess_id_tag: str, plot: bool=False, model_dict=None,
                     num_states=2,
                     predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
                     random_seed: int | None = None,
                     state_plot_line_width: float | None = None,
                     state_plot_figsize: tuple[float, float] | None = None):
    """Fit MLE trial GLM-HMM states and assign them back to the trial table.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are defined by `prepare_trial_glm_hmm_data`; the current MLE
        diagnostics also require `inherited_block_strategy`.
    figure_path : Path
        Directory where diagnostic figures are saved when `plot=True`.
    sess_id_tag : str
        Caller-provided session identifier tag used in saved figure filenames.
    plot : bool, default=False
        Whether to save diagnostic fit and state-summary figures.
    model_dict : dict or None, default=None
        Existing model dictionary to update. If None, a new dictionary is used.
    num_states : int, default=2
        Number of hidden GLM-HMM states.
    predictor_columns : tuple[str, ...], default=DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS
        Non-bias GLM predictor columns used in the input matrix.
    random_seed : int or None, default=None
        Seed used for stochastic HMM construction and EM initialization. None
        preserves the current random behavior.
    state_plot_line_width : float or None, default=None
        Optional linewidth for trial state-summary traces. None preserves the
        existing plotting defaults.
    state_plot_figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches for trial state-summary
        figures. None preserves the existing plotting defaults.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Updated model dictionary and `trial_df` with inferred state columns.
    """
    prepared = prepare_trial_glm_hmm_data(
        trial_df,
        require_inherited_strategy=True,
        predictor_columns=predictor_columns,
    )
    ix_valid = prepared["valid_mask"]
    df = trial_df[ix_valid]
    session_boundary_positions, session_boundary_labels = get_session_boundary_markers(
        trial_df,
        valid_mask=ix_valid,
    )

    block_strategy = df['inherited_block_strategy'].to_numpy()
    block_strategy[block_strategy != 'None'] = block_strategy[block_strategy != 'None'].astype(int)
    block_strategy[block_strategy == 'None'] = np.amax(block_strategy[block_strategy != 'None']) + 1
    block_strategy = block_strategy.reshape(-1,1).astype(int) #+ 1

    predictors = prepared["inputs"]
    pred_labels = prepared["predictor_labels"]
    action = prepared["observations"]

    # num states - start with 2, Inf/RL, then do 3 (inf/rl/biased). Would like Inf(maybe lo and hi thresh)/RL/biased/disengaged/confused. I think this is
    # what cross-validation is gonna be for. less is better!
    obs_dim = action.shape[1]
    input_dim = predictors.shape[1]
    state_colors = get_state_colors(num_states)
    state_cmap = build_state_colormap(state_colors)

    if model_dict is None:
        model_dict = {}
    model_dict['mle'] = {}
    model_dict['mle']['random_seed'] = random_seed
    weight_dict = {}

    with temporary_numpy_seed(random_seed):
        mle_hmm = build_input_driven_glm_hmm(
            num_states=num_states,
            obs_dim=obs_dim,
            input_dim=input_dim,
            algorithm='MLE',
        )
        N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
        fit_log_likelihood = mle_hmm.fit(action, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['mle']['fit_log_likelihood'] = fit_log_likelihood

    if plot:
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_log_likelihood, label="EM")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.title("MLE EM fit of observed data")
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_convergence.png'.format(sess_id_tag)
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

    if plot:
        f, ax = plt.subplots()
        for k in range(num_states):
            plt.plot(range(input_dim), interpreted_weights[k][0], color=state_colors[k],
                     lw=1.5, linestyle='--')#, label='state {}'.format(k),)

        plt.yticks(fontsize=10)
        plt.ylabel("GLM weight", fontsize=15)
        plt.xlabel("covariate", fontsize=15)
        plt.xticks(np.arange(len(pred_labels)), [lab.replace('_', ' ') for lab in pred_labels], fontsize=12, rotation=45)
        plt.axhline(y=0, color="k", alpha=0.5, ls="--")
        plt.title("Model weights (interpreted left-choice coefficients)", fontsize=15)
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_weights.png'.format(sess_id_tag)
        plt.gcf().savefig(save_path, format='png', dpi=300)

    posterior_probs = mle_hmm.expected_states(data=action, input=predictors)[0]
    if plot:
        posterior_figsize = (5, 2.5) if state_plot_figsize is None else state_plot_figsize
        posterior_line_width = 2 if state_plot_line_width is None else state_plot_line_width
        fig = plt.figure(figsize=posterior_figsize, dpi=80, facecolor='w', edgecolor='k')
        for k in range(num_states):
            plt.plot(posterior_probs[:,k], label="State " + str(k + 1), lw=posterior_line_width,
                     color=state_colors[k])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize=10)
        plt.xlabel("trial #", fontsize=15)
        plt.ylabel("p(state)", fontsize=15)
        plt.title("MLE HMM states")
        state_space_plotting.add_session_boundary_markers(
            axes=plt.gca(),
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
        )
        plt.tight_layout()
        save_path = figure_path / '{}_trial_mle_predicted_states.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

        f, ax = state_space_plotting.plot_trial_glm_hmm_state_summary(
            posterior_probs=posterior_probs,
            observations=action,
            inputs=predictors,
            hmm_fit=mle_hmm,
            colors=state_colors,
            cmap=state_cmap,
            line_width=state_plot_line_width,
            figsize=state_plot_figsize,
            session_boundary_positions=session_boundary_positions,
            session_boundary_labels=session_boundary_labels,
        )
        save_path = figure_path / '{}_trial_mle_state_summary.png'.format(sess_id_tag)
        f.savefig(save_path, format='png', dpi=300)

        f, ax = state_space_plotting.plot_trial_glm_hmm_block_state_comparison(
            block_states=block_strategy,
            trial_posterior_probs=posterior_probs,
            observations=action,
            inputs=predictors,
            hmm_fit=mle_hmm,
            colors=state_colors,
            cmap=state_cmap,
            line_width=state_plot_line_width,
            figsize=state_plot_figsize,
            session_boundary_positions=session_boundary_positions,
            session_boundary_labels=session_boundary_labels,
        )
        save_path = figure_path / f'{sess_id_tag}_trial_mle_block_state_comparison.png'
        f.savefig(save_path, format='png', dpi=300)

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['mle']['posterior_probs'] = posterior_probs
    model_dict['mle']['inferred_state_list'] = inferred_state_list
    model_dict['mle']['inferred_durations'] = inferred_durations
    model_dict['mle']['hmm_z'] = most_likely_states

    trial_df = assign_inferred_states_to_trials(trial_df, ix_valid, most_likely_states)

    if plot:
        fig, ax = state_space_plotting.plot_state_duration_histogram(
            inferred_state_list=inferred_state_list,
            inferred_durations=inferred_durations,
            num_states=num_states,
            state_colors=state_colors,
        )
        save_path = figure_path / f'{sess_id_tag}_trial_mle_state_durations.png'
        fig.savefig(save_path, format='png', dpi=300)


    plt.close('all')
    return model_dict, trial_df


def map_trial_states(trial_df: pd.DataFrame, figure_path: Path, sess_id_tag: str, plot: bool=False,
                     num_states=1, prior_sigma=1, prior_alpha=2, model_dict=None, block_dict=None,
                     predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
                     random_seed: int | None = None,
                     state_plot_line_width: float | None = None,
                     state_plot_figsize: tuple[float, float] | None = None):
    """Fit MAP trial GLM-HMM states and assign them back to the trial table.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are defined by `prepare_trial_glm_hmm_data`.
    figure_path : Path
        Directory where diagnostic figures are saved when `plot=True`.
    sess_id_tag : str
        Caller-provided session identifier tag used in saved figure filenames.
    plot : bool, default=False
        Whether to save diagnostic fit and state-summary figures.
    num_states : int, default=1
        Number of hidden GLM-HMM states.
    prior_sigma : float, default=1
        Observation prior scale passed to the MAP HMM.
    prior_alpha : float, default=2
        Sticky-transition concentration passed to the MAP HMM.
    model_dict : dict or None, default=None
        Existing model dictionary to update. If None, a new dictionary is used.
    block_dict : dict or None, default=None
        Optional block-model dictionary used for block/trial comparison plots.
    predictor_columns : tuple[str, ...], default=DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS
        Non-bias GLM predictor columns used in the input matrix.
    random_seed : int or None, default=None
        Seed used for stochastic HMM construction and EM initialization. None
        preserves the current random behavior.
    state_plot_line_width : float or None, default=None
        Optional linewidth for trial state-summary traces. None preserves the
        existing plotting defaults.
    state_plot_figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches for trial state-summary
        figures. None preserves the existing plotting defaults.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Updated model dictionary and `trial_df` with inferred state columns.
    """
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
    session_boundary_positions, session_boundary_labels = get_session_boundary_markers(
        trial_df,
        valid_mask=ix_valid,
    )

    block_strategy = df['inherited_block_strategy'].to_numpy()
    block_strategy = block_strategy.reshape(-1,1).astype(int) #+ 1

    predictors = prepared["inputs"]
    pred_labels = prepared["predictor_labels"]
    action = prepared["observations"]
    observations = action

    obs_dim = action.shape[1]
    input_dim = predictors.shape[1]

    if model_dict is None:
        model_dict = {}

    model_dict['map'] = {}
    model_dict['map']['random_seed'] = random_seed
    weight_dict = {}

    with temporary_numpy_seed(random_seed):
        map_hmm = build_input_driven_glm_hmm(
            num_states=num_states,
            obs_dim=obs_dim,
            input_dim=input_dim,
            algorithm='MAP',
            prior_alpha=prior_alpha,
            prior_sigma=prior_sigma,
        )
        N_iters = 10000  # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
        fit_log_likelihood = map_hmm.fit(action, inputs=predictors, method="em", num_iters=N_iters, tolerance=10 ** -6)
    model_dict['map']['fit_log_likelihood'] = fit_log_likelihood

    if plot:
        fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(fit_log_likelihood, label="EM")
        plt.legend(loc="lower right")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log Probability")
        plt.title("MAP EM fit of observed data")
        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_convergence.png'.format(sess_id_tag)
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
        save_path = figure_path / '{}_trial_map_weights.png'.format(sess_id_tag)
        plt.gcf().savefig(save_path, format='png', dpi=300)

    posterior_probs = map_hmm.expected_states(data=action, input=predictors)[0]
    if plot:
        posterior_figsize = (5, 2.5) if state_plot_figsize is None else state_plot_figsize
        posterior_line_width = 2 if state_plot_line_width is None else state_plot_line_width
        fig = plt.figure(figsize=posterior_figsize, dpi=80, facecolor='w', edgecolor='k')
        # sess_id = 0  # session id; can choose any index between 0 and num_sess-1
        for k in range(num_states):
            plt.plot(posterior_probs[:,k], label="State " + str(k + 1), lw=posterior_line_width,
                     color=state_colors[k])
        plt.ylim((-0.01, 1.01))
        plt.yticks([0, 0.5, 1], fontsize=10)
        plt.xlabel("Trial", fontsize=15)
        plt.ylabel("p(Strategy)", fontsize=15)
        plt.xlim((0, n_trials))
        # plt.title("MAP HMM states")
        plt.title("HMM behavior probabilities")
        state_space_plotting.add_session_boundary_markers(
            axes=plt.gca(),
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
        )
        plt.tight_layout()
        save_path = figure_path / '{}_trial_map_predicted_states.png'.format(sess_id_tag)
        fig.savefig(save_path, format='png', dpi=300)

        f, ax = state_space_plotting.plot_trial_glm_hmm_state_summary(
            posterior_probs=posterior_probs,
            observations=action,
            inputs=predictors,
            hmm_fit=map_hmm,
            colors=state_colors,
            cmap=state_cmap,
            line_width=state_plot_line_width,
            figsize=state_plot_figsize,
            session_boundary_positions=session_boundary_positions,
            session_boundary_labels=session_boundary_labels,
        )
        save_path = figure_path / '{}_trial_map_state_summary.png'.format(sess_id_tag)
        f.savefig(save_path, format='png', dpi=300)

        f, ax = state_space_plotting.plot_trial_glm_hmm_block_state_comparison(
            block_states=block_strategy,
            trial_posterior_probs=posterior_probs,
            observations=action,
            inputs=predictors,
            hmm_fit=map_hmm,
            colors=state_colors,
            cmap=state_cmap,
            line_width=state_plot_line_width,
            figsize=state_plot_figsize,
            session_boundary_positions=session_boundary_positions,
            session_boundary_labels=session_boundary_labels,
        )
        save_path = figure_path / f'{sess_id_tag}_trial_map_block_state_comparison.png'
        f.savefig(save_path, format='png', dpi=300)

    if plot and block_dict is not None:
        figure_kwargs = {} if state_plot_figsize is None else {"figsize": state_plot_figsize}
        trace_line_width_kwargs = {} if state_plot_line_width is None else {"linewidth": state_plot_line_width}
        f, ax = plt.subplots(2,1, **figure_kwargs)

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
        ax[0].plot(observations[:, 0], '-k', label='obs', **trace_line_width_kwargs)
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
        ax[1].plot(observations[:, 0], '-k', label='obs', **trace_line_width_kwargs)
        ax[1].set_ylim(-0.2, 1.2)
        ax[1].set_yticks([0, 1])
        ax[1].set_yticklabels(['Right', 'Left'], fontsize=12)
        ax[1].set_title('Trials labeled by trial strategy')
        state_space_plotting.add_session_boundary_markers(
            axes=ax,
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
        )
        plt.tight_layout()

        save_path = figure_path / '{}_trialMapStateComparison.png'.format(sess_id_tag)
        f.savefig(save_path, format='png', dpi=300)
        # plt.show()

    inferred_state_list, inferred_durations = ssm.util.rle(most_likely_states)
    model_dict['map']['posterior_probs'] = posterior_probs
    model_dict['map']['inferred_state_list'] = inferred_state_list
    model_dict['map']['inferred_durations'] = inferred_durations
    model_dict['map']['hmm_z'] = most_likely_states

    trial_df = assign_inferred_states_to_trials(trial_df, ix_valid, most_likely_states)

    if plot:
        fig, ax = state_space_plotting.plot_state_duration_histogram(
            inferred_state_list=inferred_state_list,
            inferred_durations=inferred_durations,
            num_states=num_states,
            state_colors=state_colors,
        )
        save_path = figure_path / f'{sess_id_tag}_trial_map_state_durations.png'
        fig.savefig(save_path, format='png', dpi=300)

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
                       predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
                       random_seed: int | None = None,
                       state_plot_line_width: float | None = None,
                       state_plot_figsize: tuple[float, float] | None = None) -> pd.DataFrame:
    """Run MLE and MAP trial GLM-HMM modeling and save trial-level outputs.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. Required
        columns are defined by `prepare_trial_glm_hmm_data`.
    session : Session
        Session metadata with `processed_data_path`, `figure_path`, and
        `sess_id_full` attributes controlling output locations and filenames.
    num_states : int, default=2
        Number of hidden GLM-HMM states.
    prior_alpha : float, default=1
        Sticky-transition concentration passed to MAP fitting.
    prior_sigma : float, default=1
        Observation prior scale passed to MAP fitting.
    predictor_columns : tuple[str, ...], default=DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS
        Non-bias GLM predictor columns used in the input matrix.
    random_seed : int or None, default=None
        Base seed split into MLE and MAP child seeds. None preserves current
        stochastic behavior.
    state_plot_line_width : float or None, default=None
        Optional linewidth for trial state-summary traces. None preserves the
        existing plotting defaults.
    state_plot_figsize : tuple[float, float] or None, default=None
        Optional matplotlib figure size in inches for trial state-summary
        figures. None preserves the existing plotting defaults.

    Returns
    -------
    pd.DataFrame
        Updated trial table with inferred trial-state labels.
    """
    block_dict_fname = session.processed_data_path / (session.sess_id_full + '_block_statedict.pkl')
    with open(block_dict_fname, 'rb') as file:
        block_dict = pkl.load(file)

    mle_seed, map_seed = spawn_child_seeds(random_seed, 2)
    mle_model_dict, _ = mle_trial_states(
        trial_df,
        figure_path=session.figure_path,
        sess_id_tag=session.sess_id_full,
        plot=True,
        num_states=num_states,
        predictor_columns=predictor_columns,
        random_seed=mle_seed,
        state_plot_line_width=state_plot_line_width,
        state_plot_figsize=state_plot_figsize,
    )
    map_model_dict, augmented_trial_df = map_trial_states(
        trial_df,
        figure_path=session.figure_path,
        sess_id_tag=session.sess_id_full,
        plot=True,
        model_dict=mle_model_dict,
        num_states=num_states,
        prior_alpha=prior_alpha,
        prior_sigma=prior_sigma,
        block_dict=block_dict['map'],
        predictor_columns=predictor_columns,
        random_seed=map_seed,
        state_plot_line_width=state_plot_line_width,
        state_plot_figsize=state_plot_figsize,
    )
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
                                         n_iter: int = 1000, tol: float = 10**-4,
                                         random_seed: int | None = None):
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
    random_seed : int or None, default=None
        Seed used for stochastic HMM construction and EM initialization. None
        preserves the current random behavior.

    Returns
    -------
    float
        Fitted model log likelihood on `observations`, in log-probability units.
    """
    assert algorithm in ['MAP', 'MLE'], "Algorithm must be MAP or MLE"

    obs_dim, input_dim = observations.shape[1], inputs.shape[1]
    with temporary_numpy_seed(random_seed):
        hmm = build_input_driven_glm_hmm(
            num_states=num_states,
            obs_dim=obs_dim,
            input_dim=input_dim,
            algorithm=algorithm,
            prior_alpha=prior_alpha,
            prior_sigma=prior_sigma,
        )

        hmm.fit(observations, inputs=inputs, method="em", num_iters=n_iter, tolerance=tol)
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
                                   nRunEM: int, n_jobs: int, algorithm='MLE', prior_alpha=1, prior_sigma=1,
                                   random_seed: int | None = None,
                                   restart_random_seeds: list[int | None] | None = None):
    if algorithm.upper() != 'MLE':
        raise ValueError("GLM-HMM information criteria should only be run on MLE models.")

    n_timesteps = observations.shape[0]
    input_dim = inputs.shape[1]
    n_states = states.size
    num_categories = 2

    AIC = np.zeros((n_states, nRunEM))
    BIC = np.zeros((n_states, nRunEM))
    if restart_random_seeds is None:
        restart_random_seeds = spawn_child_seeds(random_seed, n_states * nRunEM)
    if len(restart_random_seeds) != n_states * nRunEM:
        raise ValueError(
            f"Expected {n_states * nRunEM} restart seeds, got {len(restart_random_seeds)}"
        )
    restart_seed_grid = np.asarray(restart_random_seeds, dtype=object).reshape(n_states, nRunEM)

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
    # plt.show()


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
    random_seed: int | None = None,
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
    random_seed : int or None, default=None
        Base seed used to derive one deterministic seed per held-out fold. None
        preserves the current random behavior.

    Returns
    -------
    np.ndarray
        Held-out log-likelihood per trial block, shape (n_folds,). Each score
        is normalized by the number of held-out observations in that block.
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
    random_seed: int | None = None,
    restart_random_seeds: list[int | None] | None = None,
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
    random_seed : int or None, default=None
        Base seed used when `restart_random_seeds` is not provided.
    restart_random_seeds : list[int or None] or None, default=None
        Flat list with one seed per `(state, restart)` combination.

    Returns
    -------
    np.ndarray
        Held-out log-likelihood per trial, shape
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
                observations,
                inputs,
                num_states,
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
    # plt.show()

    return {'CV_log_likelihood': cv_log_likelihoods, 'states': states}


def run_cross_validation(trial_df: pd.DataFrame, session: Session, algorithm='MLE',
                         prior_alpha=1, prior_sigma=1,
                         min_states: int = 1, max_states: int = 5,
                         n_threads: int = 4, n_runs: int = 5, n_folds: int = 5,
                         predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
                         random_seed: int | None = None):
    """Run blocked within-session held-out scoring over hidden-state count.

    This is the secondary GLM-HMM model-selection path. The primary selector is
    MLE AIC/BIC fit independently within each session.
    """
    prepared = prepare_trial_glm_hmm_data(trial_df, predictor_columns=predictor_columns)
    observations = prepared["observations"]
    predictors = prepared["inputs"]

    states = np.arange(min_states, max_states + 1)
    restart_random_seeds = spawn_child_seeds(random_seed, states.size * n_runs)
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
        random_seed=random_seed,
        restart_random_seeds=restart_random_seeds,
    )
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


def run_information_criteria(trial_df: pd.DataFrame, session: Session, algorithm='MLE',
                             prior_alpha=1, prior_sigma=1,
                             predictor_columns: tuple[str, ...] = DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS,
                             min_states: int = 1, max_states: int = 5,
                             n_threads: int = 4, n_runs: int = 10,
                             random_seed: int | None = None):
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
        raise ValueError("GLM-HMM information criteria should only be run on MLE models.")

    prepared = prepare_trial_glm_hmm_data(trial_df, predictor_columns=predictor_columns)
    action = prepared["observations"]
    predictors = prepared["inputs"]

    states = np.arange(min_states,max_states+1)
    restart_random_seeds = spawn_child_seeds(random_seed, states.size * n_runs)
    AIC, BIC = calculate_information_criteria(observations=action, inputs=predictors, states=states, algorithm='MLE',
                                              prior_alpha=prior_alpha, prior_sigma=prior_sigma,
                                              nRunEM=n_runs, n_jobs=n_threads,
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

    session_info_path = '{}_session_info.pkl'.format(sess_id_full)
    with open(raw_behavior_folder / session_info_path, 'rb') as f:
        session_info = pkl.load(f)

    augmented_trial_df = pd.read_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), sep=',',
                                     na_filter=False)
    # model_dict, augmented_trial_df = mle_trial_states(augmented_trial_df, figure_path, sess_id_tag=sess_id_full, plot=True)
    model_dict, augmented_trial_df = map_trial_states(
        augmented_trial_df,
        figure_path=figure_path,
        sess_id_tag=sess_id_full,
        plot=True,
    )
    augmented_trial_df.to_csv(processed_data_path / (sess_id_full + '_augmented_trials.csv'), index=False)


if __name__ == '__main__':
    main()
