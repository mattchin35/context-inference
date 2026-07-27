"""Dynamax Gaussian HMMs for exemplar-model residual state analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle as pkl
from typing import Sequence

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dynamax.hidden_markov_model import GaussianHMM
from dynamax.hidden_markov_model.inference import hmm_smoother
from scipy.special import softmax
from scipy.stats import norm

from src.behavior_analysis import block_exemplar_models


STATE_COLORS = [
    "#6b6b6b",
    "#00897b",
    "#8e24aa",
    "#7cb342",
    "#3949ab",
    "#c2185b",
]
RESIDUAL_TRACE_COLORS = [
    "#111111",
    "#d55e00",
    "#0072b2",
    "#009e73",
]


@dataclass
class DynamaxExemplarHMMResult:
    """Fitted Dynamax Gaussian HMM output for exemplar residual emissions.

    Attributes
    ----------
    model : GaussianHMM
        Dynamax model object with `num_states` hidden states and `emission_dim`
        residual-emission dimensions.
    params : object
        Fitted Dynamax parameter PyTree returned by `GaussianHMM.fit_em`.
    props : object
        Dynamax property PyTree returned by `GaussianHMM.initialize`.
    log_probs : np.ndarray
        EM marginal log probabilities with shape `(num_iters,)`.
    smoothed_probs : np.ndarray
        Posterior state probabilities with shape `(n_valid_blocks, num_states)`.
    argmax_states : np.ndarray
        State labels from `argmax(smoothed_probs, axis=1)`, shape
        `(n_valid_blocks,)`.
    """

    model: GaussianHMM
    params: object
    props: object
    log_probs: np.ndarray
    smoothed_probs: np.ndarray
    argmax_states: np.ndarray


@dataclass
class FixedExemplarHMMSmoothingResult:
    """Dynamax smoothing output for fixed exemplar likelihoods.

    Attributes
    ----------
    initial_distribution : np.ndarray
        Initial state probabilities, shape `(n_models,)`, unitless.
    transition_matrix : np.ndarray
        Sticky transition matrix, shape `(n_models, n_models)`, unitless
        transition probabilities.
    smoothed_probs : np.ndarray
        Posterior state probabilities with shape `(n_valid_blocks, n_models)`.
    smoothed_states : np.ndarray
        State labels from `argmax(smoothed_probs, axis=1)`, shape
        `(n_valid_blocks,)`.
    marginal_loglik : float
        Sequence marginal log likelihood returned by Dynamax.
    """

    initial_distribution: np.ndarray
    transition_matrix: np.ndarray
    smoothed_probs: np.ndarray
    smoothed_states: np.ndarray
    marginal_loglik: float


def default_exemplar_normalized_residual_columns() -> list[str]:
    """Return normalized residual columns for the default exemplar models.

    Parameters
    ----------
    None

    Returns
    -------
    list[str]
        Column names, length `n_default_exemplar_models`. Each column contains
        normalized trials-to-correct residuals in residual-SD units.
    """
    return [
        f"{spec.name}_exemplar_normalized_residual_TTS"
        for spec in block_exemplar_models.DEFAULT_BLOCK_EXEMPLAR_MODELS
    ]


def exemplar_raw_residual_column(model_name: str) -> str:
    """Return the raw residual column for one exemplar model.

    Parameters
    ----------
    model_name : str
        Exemplar model name from the saved parameter CSV.

    Returns
    -------
    str
        Column name containing raw trials-to-correct residuals, in trials.
    """
    return f"{model_name}_exemplar_residual_TTS"


def exemplar_normalized_residual_column(model_name: str) -> str:
    """Return the normalized residual column for one exemplar model.

    Parameters
    ----------
    model_name : str
        Exemplar model name from the saved parameter CSV.

    Returns
    -------
    str
        Column name containing normalized residuals in residual-SD units.
    """
    return f"{model_name}_exemplar_normalized_residual_TTS"


def require_columns(table: pd.DataFrame, required_columns: Sequence[str], context: str) -> None:
    """Raise a readable error when a dataframe lacks required columns.

    Parameters
    ----------
    table : pandas.DataFrame
        Table with shape `(n_rows, n_columns)`.
    required_columns : Sequence[str]
        Column names required by the caller.
    context : str
        Human-readable source description for the error message.

    Returns
    -------
    None
        Returns None when all required columns are present.
    """
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"{context} is missing required exemplar residual columns: {missing_columns}")


def prepare_exemplar_hmm_emissions(
    block_performance: pd.DataFrame,
    residual_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract valid exemplar residual emissions for a Gaussian HMM.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required residual
        columns contain normalized trials-to-correct residuals in residual-SD
        units. Rows may contain the project missing-value sentinel `"None"`.
    residual_columns : Sequence[str] or None, default=None
        Residual columns to use as emissions. None uses the normalized
        residual columns for `DEFAULT_BLOCK_EXEMPLAR_MODELS`.

    Returns
    -------
    tuple[pandas.DataFrame, np.ndarray]
        Valid block rows, shape `(n_valid_blocks, n_columns)`, and emission
        matrix, shape `(n_valid_blocks, n_residual_columns)`, in residual-SD
        units. Only rows with numeric values in every residual column are kept.

    Raises
    ------
    ValueError
        If required residual columns are absent or no rows have complete
        numeric residual emissions.
    """
    selected_columns = (
        list(residual_columns)
        if residual_columns is not None
        else default_exemplar_normalized_residual_columns()
    )
    require_columns(block_performance, selected_columns, "block_performance")

    residual_table = block_performance.loc[:, selected_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid_rows = residual_table.notna().all(axis=1)
    if not valid_rows.any():
        raise ValueError("No block rows contain complete numeric exemplar residual emissions.")

    valid_blocks = block_performance.loc[valid_rows].copy().reset_index(drop=True)
    emissions = residual_table.loc[valid_rows].to_numpy(dtype=float)
    return valid_blocks, emissions


def fit_dynamax_gaussian_hmm(
    emissions: np.ndarray,
    num_states: int = 4,
    num_iters: int = 100,
    random_seed: int = 0,
    transition_matrix_stickiness: float = 10.0,
) -> DynamaxExemplarHMMResult:
    """Fit a Dynamax Gaussian HMM to exemplar residual emissions.

    Parameters
    ----------
    emissions : np.ndarray
        Emission matrix with shape `(n_valid_blocks, emission_dim)`. Values are
        normalized exemplar residuals in residual-SD units.
    num_states : int, default=4
        Number of discrete hidden states in the Gaussian HMM.
    num_iters : int, default=100
        Number of EM iterations.
    random_seed : int, default=0
        JAX PRNG seed used for k-means initialization.
    transition_matrix_stickiness : float, default=10.0
        Dynamax transition-matrix stickiness concentration. Larger values
        encourage state persistence.

    Returns
    -------
    DynamaxExemplarHMMResult
        Fitted model, fitted parameters, EM log probabilities, smoothed state
        probabilities, and argmax state labels.

    Raises
    ------
    ValueError
        If `emissions` is not a finite 2D array or has fewer rows than states.
    """
    emission_array = np.asarray(emissions, dtype=float)
    if emission_array.ndim != 2:
        raise ValueError("emissions must have shape (n_valid_blocks, emission_dim).")
    if not np.isfinite(emission_array).all():
        raise ValueError("emissions must contain only finite numeric values.")
    if emission_array.shape[0] < int(num_states):
        raise ValueError(
            "Cannot fit exemplar Gaussian HMM with fewer valid blocks than states: "
            f"{emission_array.shape[0]} blocks, {num_states} states."
        )

    model = GaussianHMM(
        num_states=int(num_states),
        emission_dim=int(emission_array.shape[1]),
        transition_matrix_stickiness=float(transition_matrix_stickiness),
    )
    emissions_jax = jnp.asarray(emission_array)
    key = jr.PRNGKey(int(random_seed))
    params, props = model.initialize(key=key, method="kmeans", emissions=emissions_jax)
    params, log_probs = model.fit_em(
        params,
        props,
        emissions_jax,
        num_iters=int(num_iters),
        verbose=False,
    )
    posterior = model.smoother(params, emissions_jax)
    smoothed_probs = np.asarray(posterior.smoothed_probs, dtype=float)
    argmax_states = np.asarray(np.argmax(smoothed_probs, axis=1), dtype=int)
    return DynamaxExemplarHMMResult(
        model=model,
        params=params,
        props=props,
        log_probs=np.asarray(log_probs, dtype=float),
        smoothed_probs=smoothed_probs,
        argmax_states=argmax_states,
    )


def add_dynamax_state_outputs(
    valid_blocks: pd.DataFrame,
    smoothed_probs: np.ndarray,
) -> pd.DataFrame:
    """Add Dynamax state probabilities and argmax labels to valid block rows.

    Parameters
    ----------
    valid_blocks : pandas.DataFrame
        Valid block table with shape `(n_valid_blocks, n_columns)`. Rows should
        correspond one-to-one with `smoothed_probs`.
    smoothed_probs : np.ndarray
        Posterior state probabilities with shape `(n_valid_blocks, num_states)`.

    Returns
    -------
    pandas.DataFrame
        Copy of `valid_blocks` with `dynamax_state_argmax` and
        `dynamax_state_prob_<state_index>` columns. Probabilities are unitless.

    Raises
    ------
    ValueError
        If `smoothed_probs` does not have one row per valid block.
    """
    prob_array = np.asarray(smoothed_probs, dtype=float)
    if prob_array.ndim != 2:
        raise ValueError("smoothed_probs must have shape (n_valid_blocks, num_states).")
    if prob_array.shape[0] != valid_blocks.shape[0]:
        raise ValueError(
            "smoothed_probs row count must match valid_blocks row count: "
            f"{prob_array.shape[0]} probabilities, {valid_blocks.shape[0]} blocks."
        )

    output_df = valid_blocks.copy()
    output_df["dynamax_state_argmax"] = np.argmax(prob_array, axis=1).astype(int)
    for state_index in range(prob_array.shape[1]):
        output_df[f"dynamax_state_prob_{state_index}"] = prob_array[:, state_index]
    return output_df


def make_serializable_dynamax_hmm_payload(
    fit_result: DynamaxExemplarHMMResult,
    emissions: np.ndarray,
    residual_columns: Sequence[str],
    modeled_csv_path: Path,
    num_iters: int,
    random_seed: int,
    transition_matrix_stickiness: float,
) -> dict:
    """Build a pickle-safe payload from a fitted Dynamax Gaussian HMM.

    Parameters
    ----------
    fit_result : DynamaxExemplarHMMResult
        In-memory fitted HMM result. Dynamax model/property objects are not
        serialized because they may contain unpickleable TensorFlow Probability
        internals.
    emissions : np.ndarray
        Emission matrix with shape `(n_valid_blocks, emission_dim)` in
        residual-SD units.
    residual_columns : Sequence[str]
        Residual column names corresponding to the emission dimensions.
    modeled_csv_path : pathlib.Path
        Path to the saved valid-block CSV.
    num_iters : int
        Number of EM iterations used for fitting.
    random_seed : int
        JAX PRNG seed used for initialization.
    transition_matrix_stickiness : float
        Dynamax transition stickiness concentration used when constructing the
        model.

    Returns
    -------
    dict
        Pickle-safe dictionary containing NumPy arrays and scalar metadata.
        Array shapes include `(num_states,)` initial probabilities,
        `(num_states, num_states)` transition matrix, `(num_states,
        emission_dim)` emission means, and `(num_states, emission_dim,
        emission_dim)` emission covariances.
    """
    params = fit_result.params
    emission_array = np.asarray(emissions, dtype=float)
    return {
        "num_states": int(fit_result.smoothed_probs.shape[1]),
        "emission_dim": int(emission_array.shape[1]),
        "num_iters": int(num_iters),
        "random_seed": int(random_seed),
        "transition_matrix_stickiness": float(transition_matrix_stickiness),
        "residual_columns": list(residual_columns),
        "emissions": emission_array,
        "modeled_csv_path": str(modeled_csv_path),
        "initial_probs": np.asarray(params.initial.probs, dtype=float),
        "transition_matrix": np.asarray(params.transitions.transition_matrix, dtype=float),
        "emission_means": np.asarray(params.emissions.means, dtype=float),
        "emission_covariances": np.asarray(params.emissions.covs, dtype=float),
        "log_probs": np.asarray(fit_result.log_probs, dtype=float),
        "smoothed_probs": np.asarray(fit_result.smoothed_probs, dtype=float),
        "argmax_states": np.asarray(fit_result.argmax_states, dtype=int),
    }


def contiguous_state_runs(state_values: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous runs of identical inferred HMM states.

    Parameters
    ----------
    state_values : np.ndarray
        One-dimensional integer state labels with shape `(n_valid_blocks,)`.
        Values are discrete inferred HMM state ids.

    Returns
    -------
    list[tuple[int, int, int]]
        Contiguous state runs. Each tuple is `(state_id, start_index,
        end_index)`, where indices are valid-block ordinal positions and
        `end_index` is inclusive.
    """
    states = np.asarray(state_values, dtype=int)
    if states.ndim != 1:
        raise ValueError("state_values must have shape (n_valid_blocks,).")
    if states.shape[0] == 0:
        return []

    runs = []
    start_index = 0
    current_state = int(states[0])
    for row_index in range(1, states.shape[0]):
        row_state = int(states[row_index])
        if row_state != current_state:
            runs.append((current_state, start_index, row_index - 1))
            start_index = row_index
            current_state = row_state
    runs.append((current_state, start_index, states.shape[0] - 1))
    return runs


def add_state_spans_to_axis(ax: plt.Axes, state_values: np.ndarray, alpha: float = 0.14) -> None:
    """Add faint inferred-state backgrounds to one valid-block axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose x-coordinate is valid-block ordinal index.
    state_values : np.ndarray
        One-dimensional integer state labels with shape `(n_valid_blocks,)`.
        Values are discrete inferred HMM state ids.
    alpha : float, default=0.14
        Patch alpha for the background spans.

    Returns
    -------
    None
        Modifies `ax` in place.
    """
    for state_id, start_index, end_index in contiguous_state_runs(state_values):
        ax.axvspan(
            start_index - 0.5,
            end_index + 0.5,
            color=STATE_COLORS[state_id % len(STATE_COLORS)],
            alpha=alpha,
            linewidth=0,
        )


def plot_dynamax_exemplar_hmm_states(
    modeled_blocks: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    residual_columns: Sequence[str] | None = None,
) -> Path:
    """Plot Dynamax exemplar-HMM state labels and residual emissions.

    Parameters
    ----------
    modeled_blocks : pandas.DataFrame
        Valid block table with shape `(n_valid_blocks, n_columns)`. Required
        columns are `dynamax_state_argmax` and the selected residual columns.
    plot_path : pathlib.Path
        Directory where the PNG is saved.
    figure_id : str
        Filename prefix and plot title identifier.
    residual_columns : Sequence[str] or None, default=None
        Residual columns to plot. None uses the normalized residual columns for
        `DEFAULT_BLOCK_EXEMPLAR_MODELS`.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    selected_columns = (
        list(residual_columns)
        if residual_columns is not None
        else default_exemplar_normalized_residual_columns()
    )
    require_columns(modeled_blocks, ["dynamax_state_argmax", *selected_columns], "modeled_blocks")

    plot_df = modeled_blocks.copy().reset_index(drop=True)
    state_values = pd.to_numeric(plot_df["dynamax_state_argmax"], errors="raise").to_numpy(dtype=int)
    residual_table = plot_df.loc[:, selected_columns].apply(pd.to_numeric, errors="raise")
    x_values = np.arange(plot_df.shape[0])

    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2]},
    )
    for axis in axes:
        add_state_spans_to_axis(axis, state_values)

    axes[0].step(x_values, state_values, where="mid", color="black", linewidth=1.5)
    state_colors = [STATE_COLORS[state_id % len(STATE_COLORS)] for state_id in state_values]
    axes[0].scatter(x_values, state_values, color=state_colors, s=24, zorder=3)
    axes[0].set_yticks(np.arange(int(np.max(state_values)) + 1))
    axes[0].set_ylabel("state")
    axes[0].set_title(f"{figure_id} Dynamax Exemplar HMM")

    for column_index, column in enumerate(selected_columns):
        axes[1].plot(
            x_values,
            residual_table[column].to_numpy(dtype=float),
            linewidth=1.2,
            color=RESIDUAL_TRACE_COLORS[column_index % len(RESIDUAL_TRACE_COLORS)],
            label=column.replace("_exemplar_normalized_residual_TTS", ""),
        )
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Valid block index")
    axes[1].set_ylabel("normalized residual")
    axes[1].legend(fancybox=False, frameon=False, fontsize=8, ncol=2)
    for axis in axes:
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)

    save_path = plot_path / f"{figure_id}_dynamax_exemplar_hmm_states.png"
    fig.tight_layout()
    fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_dynamax_exemplar_hmm_for_block_csv(
    block_performance_csv: Path,
    output_path: Path,
    figure_path: Path,
    figure_id: str,
    num_states: int = 4,
    num_iters: int = 100,
    random_seed: int = 0,
    transition_matrix_stickiness: float = 10.0,
) -> pd.DataFrame:
    """Fit and save a single-session Dynamax exemplar Gaussian HMM.

    Parameters
    ----------
    block_performance_csv : pathlib.Path
        Input block-performance CSV. It must contain the default normalized
        exemplar residual columns. Values are normalized residuals in
        residual-SD units.
    output_path : pathlib.Path
        Directory where the modeled valid-block CSV and pickle are saved.
    figure_path : pathlib.Path
        Directory where the state/residual PNG is saved.
    figure_id : str
        Session identifier used in output filenames.
    num_states : int, default=4
        Number of Gaussian HMM states.
    num_iters : int, default=100
        Number of EM iterations.
    random_seed : int, default=0
        JAX PRNG seed for k-means initialization.
    transition_matrix_stickiness : float, default=10.0
        Dynamax transition-matrix stickiness concentration. Larger values
        encourage state persistence.

    Returns
    -------
    pandas.DataFrame
        Valid block table with `dynamax_state_*` columns, shape
        `(n_valid_blocks, n_columns)`.
    """
    block_performance = pd.read_csv(block_performance_csv, sep=",", na_filter=False)
    valid_blocks, emissions = prepare_exemplar_hmm_emissions(block_performance)
    fit_result = fit_dynamax_gaussian_hmm(
        emissions=emissions,
        num_states=num_states,
        num_iters=num_iters,
        random_seed=random_seed,
        transition_matrix_stickiness=transition_matrix_stickiness,
    )
    modeled_blocks = add_dynamax_state_outputs(
        valid_blocks,
        smoothed_probs=fit_result.smoothed_probs,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    modeled_csv_path = output_path / f"{figure_id}_dynamax_exemplar_hmm_block_performance.csv"
    modeled_blocks.to_csv(modeled_csv_path, index=False, na_rep=block_exemplar_models.MISSING_VALUE)

    pickle_path = output_path / f"{figure_id}_dynamax_exemplar_hmm.pkl"
    residual_columns = default_exemplar_normalized_residual_columns()
    payload = make_serializable_dynamax_hmm_payload(
        fit_result=fit_result,
        emissions=emissions,
        residual_columns=residual_columns,
        modeled_csv_path=modeled_csv_path,
        num_iters=num_iters,
        random_seed=random_seed,
        transition_matrix_stickiness=transition_matrix_stickiness,
    )
    with pickle_path.open("wb") as file:
        pkl.dump(payload, file)

    plot_dynamax_exemplar_hmm_states(
        modeled_blocks=modeled_blocks,
        plot_path=figure_path,
        figure_id=figure_id,
    )
    return modeled_blocks


def load_exemplar_model_parameters(parameter_csv: Path) -> pd.DataFrame:
    """Load saved exemplar model parameters for fixed likelihood scoring.

    Parameters
    ----------
    parameter_csv : pathlib.Path
        CSV with shape `(n_models, n_parameter_columns)`. Required columns are
        `name` and `residual_sd`; `residual_sd` is in trials.

    Returns
    -------
    pandas.DataFrame
        Parameter table preserving file order, with numeric `residual_sd`.

    Raises
    ------
    ValueError
        If required columns are absent, model names are missing/duplicated, or
        residual deviations are nonnumeric.
    """
    parameters = pd.read_csv(parameter_csv, sep=",", na_filter=False)
    require_columns(parameters, ["name", "residual_sd"], "exemplar parameter table")
    output_df = parameters.copy()
    output_df["name"] = output_df["name"].astype(str)
    if output_df["name"].isna().any() or output_df["name"].isin(["", "None"]).any():
        raise ValueError("exemplar parameter table contains missing model names.")
    if output_df["name"].duplicated().any():
        duplicated_names = output_df.loc[output_df["name"].duplicated(), "name"].tolist()
        raise ValueError(f"exemplar parameter table contains duplicated model names: {duplicated_names}")
    output_df["residual_sd"] = pd.to_numeric(output_df["residual_sd"], errors="raise")
    return output_df


def prepare_fixed_exemplar_likelihood_inputs(
    block_performance: pd.DataFrame,
    parameters: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Read fixed-exemplar residual matrices from saved block outputs.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. For each model in
        `parameters.name`, required columns are raw residuals in trials and
        normalized residuals in residual-SD units.
    parameters : pandas.DataFrame
        Exemplar parameter table with shape `(n_models, n_parameter_columns)`.
        Required columns are `name` and `residual_sd`.

    Returns
    -------
    tuple[pandas.DataFrame, np.ndarray, np.ndarray, list[str], np.ndarray]
        Valid block rows, raw residual matrix `(n_valid_blocks, n_models)` in
        trials, existing normalized residual matrix `(n_valid_blocks,
        n_models)` in residual-SD units, model names, and residual deviations
        `(n_models,)` in trials.
    """
    require_columns(parameters, ["name", "residual_sd"], "exemplar parameter table")
    model_names = parameters["name"].astype(str).tolist()
    sigmas = pd.to_numeric(parameters["residual_sd"], errors="raise").to_numpy(dtype=float)
    raw_columns = [exemplar_raw_residual_column(model_name) for model_name in model_names]
    normalized_columns = [
        exemplar_normalized_residual_column(model_name) for model_name in model_names
    ]
    require_columns(block_performance, [*raw_columns, *normalized_columns], "block_performance")

    raw_table = block_performance.loc[:, raw_columns].apply(pd.to_numeric, errors="coerce")
    normalized_table = block_performance.loc[:, normalized_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    valid_rows = raw_table.notna().all(axis=1) & normalized_table.notna().all(axis=1)
    if not valid_rows.any():
        raise ValueError("No block rows contain complete fixed-exemplar residual inputs.")

    valid_blocks = block_performance.loc[valid_rows].copy().reset_index(drop=True)
    raw_residuals = raw_table.loc[valid_rows].to_numpy(dtype=float)
    z_residuals = normalized_table.loc[valid_rows].to_numpy(dtype=float)
    return valid_blocks, raw_residuals, z_residuals, model_names, sigmas


def compute_fixed_exemplar_log_likelihoods(
    residuals: np.ndarray,
    sigmas: np.ndarray,
    sigma_floor: float = 1e-6,
) -> np.ndarray:
    """Compute fixed-exemplar Gaussian log likelihoods from raw residuals.

    Parameters
    ----------
    residuals : np.ndarray
        Raw residual matrix with shape `(n_valid_blocks, n_models)`, in trials.
    sigmas : np.ndarray
        Residual standard deviations with shape `(n_models,)`, in trials.
    sigma_floor : float, default=1e-6
        Minimum allowed sigma in trials.

    Returns
    -------
    np.ndarray
        Log likelihood matrix with shape `(n_valid_blocks, n_models)`.
    """
    residual_array = np.asarray(residuals, dtype=float)
    sigma_array = np.asarray(sigmas, dtype=float)
    if residual_array.ndim != 2:
        raise ValueError("residuals must have shape (n_valid_blocks, n_models).")
    if sigma_array.ndim != 1 or sigma_array.shape[0] != residual_array.shape[1]:
        raise ValueError("sigmas must have shape (n_models,) matching residuals.")
    if not np.isfinite(residual_array).all() or not np.isfinite(sigma_array).all():
        raise ValueError("residuals and sigmas must contain only finite numeric values.")
    sigmas_safe = np.maximum(sigma_array, float(sigma_floor))
    return norm.logpdf(residual_array, loc=0.0, scale=sigmas_safe[np.newaxis, :])


def independent_model_probabilities(log_likelihoods: np.ndarray) -> np.ndarray:
    """Convert independent block log likelihoods to row-wise model probabilities.

    Parameters
    ----------
    log_likelihoods : np.ndarray
        Log likelihood matrix with shape `(n_valid_blocks, n_models)`.

    Returns
    -------
    np.ndarray
        Softmax-normalized probabilities with shape `(n_valid_blocks,
        n_models)`.
    """
    log_likelihood_array = np.asarray(log_likelihoods, dtype=float)
    if log_likelihood_array.ndim != 2:
        raise ValueError("log_likelihoods must have shape (n_valid_blocks, n_models).")
    if not np.isfinite(log_likelihood_array).all():
        raise ValueError("log_likelihoods must contain only finite numeric values.")
    return np.asarray(softmax(log_likelihood_array, axis=1), dtype=float)


def build_sticky_transition_matrix(num_states: int, stay_prob: float = 0.9) -> np.ndarray:
    """Build a fixed sticky transition matrix over exemplar states.

    Parameters
    ----------
    num_states : int
        Number of exemplar states/models.
    stay_prob : float, default=0.9
        Diagonal transition probability.

    Returns
    -------
    np.ndarray
        Transition matrix with shape `(num_states, num_states)`.
    """
    state_count = int(num_states)
    if state_count < 1:
        raise ValueError("num_states must be at least 1.")
    if not 0 <= float(stay_prob) <= 1:
        raise ValueError("stay_prob must be between 0 and 1.")
    if state_count == 1:
        return np.ones((1, 1), dtype=float)

    transition_matrix = np.full(
        (state_count, state_count),
        (1.0 - float(stay_prob)) / (state_count - 1),
        dtype=float,
    )
    np.fill_diagonal(transition_matrix, float(stay_prob))
    return transition_matrix


def smooth_fixed_exemplar_likelihoods_with_dynamax(
    log_likelihoods: np.ndarray,
    stay_prob: float = 0.9,
) -> FixedExemplarHMMSmoothingResult:
    """Smooth fixed exemplar log likelihoods with Dynamax HMM inference.

    Parameters
    ----------
    log_likelihoods : np.ndarray
        Log likelihood matrix with shape `(n_valid_blocks, n_models)`.
    stay_prob : float, default=0.9
        Diagonal probability of the fixed transition matrix.

    Returns
    -------
    FixedExemplarHMMSmoothingResult
        Initial distribution, transition matrix, smoothed probabilities, and
        argmax smoothed state labels.
    """
    log_likelihood_array = np.asarray(log_likelihoods, dtype=float)
    if log_likelihood_array.ndim != 2:
        raise ValueError("log_likelihoods must have shape (n_valid_blocks, n_models).")
    if log_likelihood_array.shape[0] == 0:
        raise ValueError("log_likelihoods must contain at least one valid block.")
    if not np.isfinite(log_likelihood_array).all():
        raise ValueError("log_likelihoods must contain only finite numeric values.")

    num_states = log_likelihood_array.shape[1]
    initial_distribution = np.ones(num_states, dtype=float) / num_states
    transition_matrix = build_sticky_transition_matrix(num_states, stay_prob=stay_prob)
    posterior = hmm_smoother(
        jnp.asarray(initial_distribution),
        jnp.asarray(transition_matrix),
        jnp.asarray(log_likelihood_array),
    )
    smoothed_probs = np.asarray(posterior.smoothed_probs, dtype=float)
    return FixedExemplarHMMSmoothingResult(
        initial_distribution=initial_distribution,
        transition_matrix=transition_matrix,
        smoothed_probs=smoothed_probs,
        smoothed_states=np.argmax(smoothed_probs, axis=1).astype(int),
        marginal_loglik=float(np.asarray(posterior.marginal_loglik)),
    )


def add_fixed_exemplar_hmm_outputs(
    valid_blocks: pd.DataFrame,
    model_names: Sequence[str],
    z_residuals: np.ndarray,
    log_likelihoods: np.ndarray,
    independent_probs: np.ndarray,
    smoothed_probs: np.ndarray,
) -> pd.DataFrame:
    """Add fixed-exemplar likelihood and HMM-smoothed outputs to block rows.

    Parameters
    ----------
    valid_blocks : pandas.DataFrame
        Valid block table with shape `(n_valid_blocks, n_columns)`.
    model_names : Sequence[str]
        Exemplar names with length `n_models`.
    z_residuals : np.ndarray
        Existing normalized residual matrix, shape `(n_valid_blocks,
        n_models)`, in residual-SD units.
    log_likelihoods : np.ndarray
        Gaussian log likelihood matrix, shape `(n_valid_blocks, n_models)`.
    independent_probs : np.ndarray
        Row-wise softmax likelihood probabilities, shape `(n_valid_blocks,
        n_models)`.
    smoothed_probs : np.ndarray
        Dynamax smoothed probabilities, shape `(n_valid_blocks, n_models)`.

    Returns
    -------
    pandas.DataFrame
        Copy of `valid_blocks` with fixed-exemplar score, probability, state,
        and model-name columns.
    """
    names = list(model_names)
    matrices = [
        np.asarray(z_residuals, dtype=float),
        np.asarray(log_likelihoods, dtype=float),
        np.asarray(independent_probs, dtype=float),
        np.asarray(smoothed_probs, dtype=float),
    ]
    expected_shape = (valid_blocks.shape[0], len(names))
    for matrix in matrices:
        if matrix.shape != expected_shape:
            raise ValueError(
                "All fixed-exemplar matrices must have shape "
                f"{expected_shape}; received {matrix.shape}."
            )

    output_df = valid_blocks.copy()
    for model_index, model_name in enumerate(names):
        output_df[f"exemplar_z_residual_{model_name}"] = matrices[0][:, model_index]
        output_df[f"exemplar_log_likelihood_{model_name}"] = matrices[1][:, model_index]
        output_df[f"exemplar_independent_prob_{model_name}"] = matrices[2][:, model_index]
        output_df[f"exemplar_hmm_smoothed_prob_{model_name}"] = matrices[3][:, model_index]

    ll_states = np.argmax(matrices[1], axis=1).astype(int)
    smoothed_states = np.argmax(matrices[3], axis=1).astype(int)
    output_df["exemplar_ll_argmax_state"] = ll_states
    output_df["exemplar_ll_argmax_model"] = [names[state] for state in ll_states]
    output_df["exemplar_hmm_smoothed_state"] = smoothed_states
    output_df["exemplar_hmm_smoothed_model"] = [names[state] for state in smoothed_states]
    return output_df


def get_valid_session_boundary_markers(
    modeled_blocks: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Return session-boundary markers for valid-block diagnostic plots.

    Parameters
    ----------
    modeled_blocks : pandas.DataFrame
        Valid modeled block table with shape `(n_valid_blocks, n_columns)`.
        If present, `source_date` is used for readable labels. Otherwise,
        `source_session_id` is used. Rows are assumed to be in plotted valid
        block order.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Boundary positions in valid-block ordinal coordinates, shape
        `(n_boundaries,)`, and labels for the session starting at each
        boundary.
    """
    label_column = None
    for candidate_column in ("source_date", "source_session_id"):
        if candidate_column in modeled_blocks.columns:
            label_column = candidate_column
            break
    if label_column is None or modeled_blocks.empty:
        return np.asarray([], dtype=int), []

    labels = modeled_blocks[label_column].astype(str).to_numpy(dtype=object)
    boundary_positions = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundary_labels = [str(labels[position]) for position in boundary_positions]
    return boundary_positions.astype(int), boundary_labels


def add_session_boundary_markers_to_axes(
    axes: Sequence[plt.Axes],
    boundary_positions: np.ndarray,
    boundary_labels: Sequence[str],
    label_axis: plt.Axes,
) -> None:
    """Draw labeled session boundaries on fixed-exemplar diagnostic axes.

    Parameters
    ----------
    axes : Sequence[matplotlib.axes.Axes]
        Axes sharing valid-block x coordinates.
    boundary_positions : np.ndarray
        Boundary x positions in valid-block ordinal coordinates, shape
        `(n_boundaries,)`.
    boundary_labels : Sequence[str]
        Label for each boundary, usually the date of the session starting at
        that boundary.
    label_axis : matplotlib.axes.Axes
        Axis that receives below-axis text labels.

    Returns
    -------
    None
        Modifies axes in place.
    """
    positions = np.asarray(boundary_positions, dtype=int)
    labels = list(boundary_labels)
    if positions.shape[0] != len(labels):
        raise ValueError("boundary_positions and boundary_labels must have the same length.")

    for boundary_position in positions:
        for axis in axes:
            axis.axvline(
                x=boundary_position,
                color="gray",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                zorder=1,
            )
    for boundary_position, boundary_label in zip(positions, labels):
        label_axis.text(
            boundary_position,
            -0.32,
            str(boundary_label),
            transform=label_axis.get_xaxis_transform(),
            ha="center",
            va="top",
            rotation=90,
            fontsize=8,
            clip_on=False,
        )


def plot_fixed_exemplar_hmm_diagnostic(
    modeled_blocks: pd.DataFrame,
    model_names: Sequence[str],
    plot_path: Path,
    figure_id: str,
    session_boundary_positions: np.ndarray | None = None,
    session_boundary_labels: Sequence[str] | None = None,
) -> Path:
    """Plot fixed-exemplar residual scores and state assignments.

    Parameters
    ----------
    modeled_blocks : pandas.DataFrame
        Valid block table with shape `(n_valid_blocks, n_columns)`. Required
        columns include `exemplar_z_residual_<model>`,
        `exemplar_log_likelihood_<model>`, `exemplar_ll_argmax_state`, and
        `exemplar_hmm_smoothed_state`.
    model_names : Sequence[str]
        Exemplar model names in state-index order.
    plot_path : pathlib.Path
        Directory where the PNG is saved.
    figure_id : str
        Filename prefix and title identifier.
    session_boundary_positions : np.ndarray or None, default=None
        Optional session-boundary x positions in valid-block ordinal
        coordinates, shape `(n_boundaries,)`.
    session_boundary_labels : Sequence[str] or None, default=None
        Label for each boundary. Required when `session_boundary_positions` is
        provided.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    names = list(model_names)
    required_columns = [
        *[f"exemplar_z_residual_{name}" for name in names],
        *[f"exemplar_log_likelihood_{name}" for name in names],
        "exemplar_ll_argmax_state",
        "exemplar_hmm_smoothed_state",
    ]
    require_columns(modeled_blocks, required_columns, "modeled_blocks")

    plot_df = modeled_blocks.copy().reset_index(drop=True)
    x_values = np.arange(plot_df.shape[0])
    ll_states = pd.to_numeric(plot_df["exemplar_ll_argmax_state"], errors="raise").to_numpy(dtype=int)
    hmm_states = pd.to_numeric(plot_df["exemplar_hmm_smoothed_state"], errors="raise").to_numpy(dtype=int)

    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)

    for model_index, model_name in enumerate(names):
        line_color = RESIDUAL_TRACE_COLORS[model_index % len(RESIDUAL_TRACE_COLORS)]
        axes[0].plot(
            x_values,
            pd.to_numeric(plot_df[f"exemplar_z_residual_{model_name}"], errors="raise"),
            color=line_color,
            linewidth=1.2,
            label=model_name,
        )
        axes[1].plot(
            x_values,
            pd.to_numeric(plot_df[f"exemplar_log_likelihood_{model_name}"], errors="raise"),
            color=line_color,
            linewidth=1.2,
            label=model_name,
        )

    axes[0].axhline(0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("z residual")
    axes[0].set_title(f"{figure_id} Fixed Exemplar Likelihood HMM")
    axes[0].legend(fancybox=False, frameon=False, fontsize=8, ncol=2)
    axes[1].set_ylabel("log likelihood")
    axes[1].legend(fancybox=False, frameon=False, fontsize=8, ncol=2)

    for axis, states, ylabel in [
        (axes[2], ll_states, "LL argmax"),
        (axes[3], hmm_states, "HMM smooth"),
    ]:
        add_state_spans_to_axis(axis, states)
        colors = [STATE_COLORS[state % len(STATE_COLORS)] for state in states]
        axis.step(x_values, states, where="mid", color="black", linewidth=1.3)
        axis.scatter(x_values, states, color=colors, s=22, zorder=3)
        axis.set_yticks(np.arange(len(names)))
        axis.set_yticklabels(names)
        axis.set_ylabel(ylabel)

    axes[3].set_xlabel("Valid block index")
    for axis in axes:
        axis.spines["right"].set_visible(False)
        axis.spines["top"].set_visible(False)

    if session_boundary_positions is not None:
        if session_boundary_labels is None:
            raise ValueError("session_boundary_labels must be provided with boundary positions.")
        add_session_boundary_markers_to_axes(
            axes=axes,
            boundary_positions=session_boundary_positions,
            boundary_labels=session_boundary_labels,
            label_axis=axes[3],
        )

    save_path = plot_path / f"{figure_id}_exemplar_likelihood_hmm_diagnostic.png"
    fig.tight_layout()
    if session_boundary_positions is not None and np.asarray(session_boundary_positions).size > 0:
        fig.subplots_adjust(bottom=0.18)
    fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_fixed_exemplar_hmm_for_block_csv(
    block_performance_csv: Path,
    parameter_csv: Path,
    output_path: Path,
    figure_path: Path,
    figure_id: str,
    stay_prob: float = 0.9,
    plot_session_boundaries: bool = False,
) -> pd.DataFrame:
    """Run fixed-exemplar likelihood scoring and HMM smoothing for one session.

    Parameters
    ----------
    block_performance_csv : pathlib.Path
        Input block-performance CSV containing saved raw and normalized
        exemplar residual columns.
    parameter_csv : pathlib.Path
        Saved exemplar model parameter CSV containing `name` and `residual_sd`.
    output_path : pathlib.Path
        Directory where modeled CSV and pickle are saved.
    figure_path : pathlib.Path
        Directory where diagnostic PNG is saved.
    figure_id : str
        Session identifier used in output filenames.
    stay_prob : float, default=0.9
        Diagonal transition probability for HMM smoothing.
    plot_session_boundaries : bool, default=False
        If True, derive session-boundary markers from valid modeled rows using
        `source_date` or `source_session_id` and draw them on the diagnostic
        plot.

    Returns
    -------
    pandas.DataFrame
        Valid block table with fixed-exemplar score and state columns, shape
        `(n_valid_blocks, n_columns)`.
    """
    block_performance = pd.read_csv(block_performance_csv, sep=",", na_filter=False)
    parameters = load_exemplar_model_parameters(parameter_csv)
    valid_blocks, raw_residuals, z_residuals, model_names, sigmas = (
        prepare_fixed_exemplar_likelihood_inputs(block_performance, parameters)
    )
    log_likelihoods = compute_fixed_exemplar_log_likelihoods(raw_residuals, sigmas)
    independent_probs = independent_model_probabilities(log_likelihoods)
    smoothing_result = smooth_fixed_exemplar_likelihoods_with_dynamax(
        log_likelihoods,
        stay_prob=stay_prob,
    )
    modeled_blocks = add_fixed_exemplar_hmm_outputs(
        valid_blocks=valid_blocks,
        model_names=model_names,
        z_residuals=z_residuals,
        log_likelihoods=log_likelihoods,
        independent_probs=independent_probs,
        smoothed_probs=smoothing_result.smoothed_probs,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    modeled_csv_path = output_path / f"{figure_id}_exemplar_likelihood_hmm_block_performance.csv"
    modeled_blocks.to_csv(modeled_csv_path, index=False, na_rep=block_exemplar_models.MISSING_VALUE)

    pickle_path = output_path / f"{figure_id}_exemplar_likelihood_hmm.pkl"
    payload = {
        "model_names": model_names,
        "sigmas": np.asarray(sigmas, dtype=float),
        "raw_residuals": np.asarray(raw_residuals, dtype=float),
        "z_residuals": np.asarray(z_residuals, dtype=float),
        "log_likelihoods": np.asarray(log_likelihoods, dtype=float),
        "independent_probs": np.asarray(independent_probs, dtype=float),
        "stay_prob": float(stay_prob),
        "initial_distribution": smoothing_result.initial_distribution,
        "transition_matrix": smoothing_result.transition_matrix,
        "smoothed_probs": smoothing_result.smoothed_probs,
        "smoothed_states": smoothing_result.smoothed_states,
        "marginal_loglik": smoothing_result.marginal_loglik,
        "modeled_csv_path": str(modeled_csv_path),
    }
    with pickle_path.open("wb") as file:
        pkl.dump(payload, file)

    session_boundary_positions = None
    session_boundary_labels = None
    if plot_session_boundaries:
        session_boundary_positions, session_boundary_labels = get_valid_session_boundary_markers(
            modeled_blocks
        )
    plot_fixed_exemplar_hmm_diagnostic(
        modeled_blocks=modeled_blocks,
        model_names=model_names,
        plot_path=figure_path,
        figure_id=figure_id,
        session_boundary_positions=session_boundary_positions,
        session_boundary_labels=session_boundary_labels,
    )
    return modeled_blocks


def main() -> None:
    """Run a hardcoded first-pass fixed-exemplar likelihood HMM for one session.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Loads the hardcoded block-performance and exemplar-parameter CSVs,
        saves modeled valid blocks, saves a pickle with fixed-exemplar
        likelihood outputs, and writes a diagnostic PNG plot.
    """
    mouse = "CT024"
    date = "2026-05-29"
    timestamp = "143237"
    task_tag = "latent_inference"
    sess_id = f"{mouse}_{date}_{timestamp}"
    session_home = (
        Path("/home/matt/Documents/EXPERIMENTS/contextProjectData")
        / mouse
        / f"{mouse}_{date.replace('-', '')}_{task_tag}"
    )
    run_fixed_exemplar_hmm_for_block_csv(
        block_performance_csv=session_home / "processed" / f"{sess_id}_block_performance.csv",
        parameter_csv=session_home / "processed" / f"{sess_id}_block_exemplar_model_parameters.csv",
        output_path=session_home / "processed",
        figure_path=session_home / "figures",
        figure_id=sess_id,
        stay_prob=0.9,
    )


def main_multisession() -> None:
    """Run a hardcoded one-mouse multisession fixed-exemplar HMM.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Loads a hardcoded `{mouse}_multisession_block_performance.csv` and a
        manually selected saved exemplar-parameter CSV, saves fixed-exemplar
        HMM outputs into the mouse `cross_session_analysis` folder, and writes
        a diagnostic PNG with session-boundary labels.
    """
    mouse = "CT024"
    parameter_date = "2026-05-29"
    parameter_timestamp = "143237"
    task_tag = "latent_inference"
    figure_id = f"{mouse}_multisession"
    data_root = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData")
    cross_session_path = data_root / mouse / "cross_session_analysis"
    parameter_session_home = (
        data_root
        / mouse
        / f"{mouse}_{parameter_date.replace('-', '')}_{task_tag}"
    )
    parameter_session_id = f"{mouse}_{parameter_date}_{parameter_timestamp}"

    run_fixed_exemplar_hmm_for_block_csv(
        block_performance_csv=cross_session_path / f"{mouse}_multisession_block_performance.csv",
        parameter_csv=(
            parameter_session_home
            / "processed"
            / f"{parameter_session_id}_block_exemplar_model_parameters.csv"
        ),
        output_path=cross_session_path,
        figure_path=cross_session_path,
        figure_id=figure_id,
        stay_prob=0.9,
        plot_session_boundaries=True,
    )


if __name__ == "__main__":
    # main()
    main_multisession()
