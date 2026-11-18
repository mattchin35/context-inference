"""Cached population calculations and population-view selection helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import (
    population_pca_decoding,
    population_pca_switch_trajectories,
)
from src.neural_analysis.population import pca as population_pca


NEURAL_DISPLAY_SPIKE_RASTER = "Spike raster"

NEURAL_DISPLAY_LFP_SPECTROGRAM = "LFP spectrogram + trace"

NEURAL_DISPLAY_POPULATION_PCA = "Population PCA"

NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED = "Population PCA: concatenated trials"

NEURAL_DISPLAY_OPTIONS = [
    NEURAL_DISPLAY_SPIKE_RASTER,
    NEURAL_DISPLAY_LFP_SPECTROGRAM,
    NEURAL_DISPLAY_POPULATION_PCA,
    NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED,
]

PCA_BIN_SIZE_OPTIONS = [0.1, 0.05, 0.02]

DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT = 5

DEFAULT_PCA_VARIANCE_COMPONENT_COUNT = 50

DEFAULT_PCA_COMPONENT_COUNT = DEFAULT_PCA_TRAJECTORY_COMPONENT_COUNT

DEFAULT_CONCATENATED_PCA_VISIBLE_TRIAL_COUNT = 20

CONCATENATED_PCA_VISIBLE_TRIAL_OPTIONS = [10, 20, 40]

DEFAULT_CONCATENATED_PCA_COMPONENT_COUNT = 3

CONCATENATED_PCA_FIGURE_SIZE = (11.0, 5.0)

PCA_DECODING_DEFAULT_COMPONENT_COUNT = 5

PCA_DECODING_DEFAULT_CV_FOLDS = 5

PCA_DECODING_DEFAULT_PERMUTATIONS = 100

PCA_DECODING_DISPLAY_PERFORMANCE = "Decoding performance"

PCA_DECODING_DISPLAY_AVERAGE_PC = "Average PC by condition/target"

PCA_DECODING_DISPLAY_OPTIONS = [
    PCA_DECODING_DISPLAY_PERFORMANCE,
    PCA_DECODING_DISPLAY_AVERAGE_PC,
]

PCA_DECODING_SHOW_RAW_PC_SCORES_DEFAULT = False

def is_population_pca_display(neural_display: str) -> bool:
    """
    Return whether a trial-view neural display uses population PCA.

    Parameters
    ----------
    neural_display : str
        Display label selected from ``NEURAL_DISPLAY_OPTIONS``.

    Returns
    -------
    bool
        ``True`` for single-trial and concatenated population PCA displays;
        ``False`` for spike-raster displays.
    """

    return neural_display in {
        NEURAL_DISPLAY_POPULATION_PCA,
        NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED,
    }

def resolve_pca_decoding_component_minimum(display_option: str) -> int:
    """
    Resolve the minimum PCA component count for a PCA decoding display.

    Parameters
    ----------
    display_option : str
        Selected PCA decoding display mode. Expected values are entries in
        ``PCA_DECODING_DISPLAY_OPTIONS``.

    Returns
    -------
    int
        Minimum number of PCA components. The average PC score display requires
        two components because it plots mean PC1 against mean PC2; decoding
        performance can use one component.
    """

    if display_option == PCA_DECODING_DISPLAY_AVERAGE_PC:
        return 2
    return 1

def select_visible_concatenated_trial_indices(
    trial_indices: np.ndarray,
    page_index: int,
    visible_trial_count: int,
) -> tuple[np.ndarray, int]:
    """
    Select one bounded page of trial indices for concatenated PCA plotting.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Values are
        row indices into the trial table.
    page_index : int
        Zero-based requested page index. Values beyond the last page are
        clamped to the last available page.
    visible_trial_count : int
        Maximum number of trials shown in one concatenated figure.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(visible_trial_indices, page_count)``. ``visible_trial_indices`` has
        shape ``(n_visible_trials,)`` and preserves the input order.
        ``page_count`` is at least one for nonempty inputs and zero for empty
        inputs.
    """

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if int(visible_trial_count) <= 0:
        raise ValueError("visible_trial_count must be positive.")
    if normalized_trial_indices.size == 0:
        return np.array([], dtype=int), 0

    page_count = int(math.ceil(normalized_trial_indices.size / int(visible_trial_count)))
    clamped_page_index = min(max(int(page_index), 0), page_count - 1)
    start_index = clamped_page_index * int(visible_trial_count)
    end_index = start_index + int(visible_trial_count)
    return normalized_trial_indices[start_index:end_index], page_count

def select_concatenated_trial_viewport(
    trial_indices: np.ndarray,
    start_position: int,
    visible_trial_count: int,
) -> tuple[np.ndarray, int]:
    """
    Select one fixed-size chronological viewport of concatenated PCA trials.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial indices with shape ``(n_trials,)``. Values are
        row indices into the trial table.
    start_position : int
        Zero-based requested start position within ``trial_indices``.
    visible_trial_count : int
        Number of trials shown in the fixed viewport. If enough trials are
        available, oversized start positions are clamped to keep the viewport
        full.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(visible_trial_indices, clamped_start_position)``.
        ``visible_trial_indices`` has shape ``(n_visible_trials,)`` and
        preserves the input order. ``clamped_start_position`` is the zero-based
        position actually used after bounds checking.
    """

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if int(visible_trial_count) <= 0:
        raise ValueError("visible_trial_count must be positive.")
    if normalized_trial_indices.size == 0:
        return np.array([], dtype=int), 0

    visible_count = int(visible_trial_count)
    max_start_position = max(0, normalized_trial_indices.size - visible_count)
    clamped_start_position = min(max(int(start_position), 0), max_start_position)
    end_position = clamped_start_position + visible_count
    return normalized_trial_indices[clamped_start_position:end_position], clamped_start_position

def resolve_population_pca_unit_ids(
    selected_unit_metadata: pd.DataFrame,
    page_unit_ids: np.ndarray | None = None,
) -> np.ndarray:
    """
    Resolve unit ids used for population PCA in the trial-view webapp.

    Parameters
    ----------
    selected_unit_metadata : pd.DataFrame
        Filtered unit metadata table with one row per selected unit and a
        ``cluster_id`` column. Rows are the current region/channel/quality
        selection.
    page_unit_ids : np.ndarray | None, optional
        Paginated visible unit ids from raster mode. This input is accepted to
        make the contract explicit but is not used for PCA.

    Returns
    -------
    np.ndarray
        One-dimensional integer array with shape ``(n_selected_units,)``. These
        are all selected region units, not only a displayed page.
    """

    if "cluster_id" not in selected_unit_metadata.columns:
        raise ValueError("selected_unit_metadata is missing cluster_id column.")
    return selected_unit_metadata["cluster_id"].to_numpy(dtype=int)

def filter_trial_indices_for_valid_alignment(
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split selected trials by whether they have a finite alignment time.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and an ``alignment_event`` column in
        seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)``.
    alignment_event : str
        Trial time column used as time zero, such as ``"choice_time"`` or
        ``"start_time"``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(valid_trial_indices, invalid_trial_indices)``. Both arrays are
        one-dimensional integer arrays of trial row indices. Valid trials have
        finite alignment times in seconds.

    Raises
    ------
    ValueError
        If ``alignment_event`` is missing, no trial indices are supplied, or no
        selected trials have valid alignment times.
    """

    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")
    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if normalized_trial_indices.size == 0:
        raise ValueError("trial_indices must contain at least one trial.")
    alignment_times = pd.to_numeric(
        trial_df.loc[normalized_trial_indices, alignment_event],
        errors="coerce",
    ).to_numpy(dtype=float)
    valid_mask = np.isfinite(alignment_times)
    valid_trial_indices = normalized_trial_indices[valid_mask]
    invalid_trial_indices = normalized_trial_indices[~valid_mask]
    if valid_trial_indices.size == 0:
        raise ValueError(f"No selected trials have valid {alignment_event} alignment times.")
    return valid_trial_indices, invalid_trial_indices

@st.cache_data(show_spinner="Computing population PCA...")
def compute_population_pca_cached(
    session_key: str,
    aligned_spike_path: str,
    unit_ids: tuple[int, ...],
    trial_indices: tuple[int, ...],
    alignment_event: str,
    window_start_s: float,
    window_end_s: float,
    bin_size_s: float,
    normalization: str,
    trajectory_component_count: int,
    variance_component_count: int,
    _spike_group,
    _trial_df: pd.DataFrame,
):
    """
    Compute and cache population PCA for one webapp trial selection.

    Parameters
    ----------
    session_key : str
        Session identifier included in the Streamlit cache key.
    aligned_spike_path : str
        Aligned spike file path included in the cache key. Units: filesystem
        path.
    unit_ids : tuple[int, ...]
        Unit ids included as PCA features, shape ``(n_units,)``.
    trial_indices : tuple[int, ...]
        Filtered trial row indices used to fit PCA, shape ``(n_trials,)``.
    alignment_event : str
        Trial time column used as time zero.
    window_start_s, window_end_s : float
        Relative window bounds in seconds.
    bin_size_s : float
        PCA bin width in seconds.
    normalization : str
        PCA normalization mode.
    trajectory_component_count : int
        Requested number of leading PCs to display in the trial trajectory plot.
    variance_component_count : int
        Requested number of leading PCs to display in the cumulative explained
        variance plot.
    _spike_group
        Pynapple spike group keyed by unit id. Leading underscore excludes this
        potentially large object from Streamlit hashing.
    _trial_df : pd.DataFrame
        Trial table. Leading underscore excludes this dataframe from Streamlit
        hashing; ``session_key`` and ``trial_indices`` carry the cache identity.

    Returns
    -------
    tuple[np.ndarray, population_pca.PopulationPCAResult]
        ``(pca_time_s, pca_result)``. ``pca_time_s`` has shape ``(n_bins,)`` in
        seconds relative to alignment. ``pca_result.scores`` has shape
        ``(n_trials, n_bins, n_fit_components)``. ``n_fit_components`` is the
        larger requested display count capped inside ``fit_population_pca``.
    """

    trajectory_component_count = int(trajectory_component_count)
    variance_component_count = int(variance_component_count)
    if trajectory_component_count < 1 or variance_component_count < 1:
        raise ValueError("PCA component counts must be positive.")
    fit_component_count = max(trajectory_component_count, variance_component_count)
    rate_tensor_hz, pca_time_s = population_pca.build_trial_unit_rate_tensor(
        spike_group=_spike_group,
        unit_ids=np.asarray(unit_ids, dtype=int),
        trial_df=_trial_df,
        trial_indices=np.asarray(trial_indices, dtype=int),
        alignment_event=alignment_event,
        window=(float(window_start_s), float(window_end_s)),
        bin_size_s=float(bin_size_s),
    )
    pca_result = population_pca.fit_population_pca(
        rate_tensor_hz=rate_tensor_hz,
        n_components=fit_component_count,
        normalization=normalization,
    )
    return pca_time_s, pca_result

@st.cache_data(show_spinner="Computing PCA decoding...")
def compute_population_pca_decoding_cached(
    session_key: str,
    aligned_spike_path: str,
    unit_ids: tuple[int, ...],
    condition_names: tuple[str, ...],
    target: str,
    mode: str,
    n_components: int,
    cv: int,
    n_permutations: int,
    normalization: str,
    include_score_summary: bool,
    include_raw_score_points: bool,
    _spike_group,
    _trial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, int, pd.DataFrame, pd.DataFrame]:
    """
    Compute and cache choice-aligned PCA decoding for the webapp.

    Parameters
    ----------
    session_key : str
        Session/probe identifier included in the Streamlit cache key.
    aligned_spike_path : str
        Aligned spike file path included in the cache key. Units: filesystem
        path.
    unit_ids : tuple[int, ...]
        Unit ids used as raw PCA features, shape ``(n_units,)``.
    condition_names : tuple[str, ...]
        Base decoding condition names to evaluate.
    target : str
        Decode target, ``"state_int"`` or ``"action"``.
    mode : str
        PCA fitting mode from ``population_pca_decoding.PCA_DECODING_MODE_OPTIONS``.
    n_components : int
        Requested number of leading PCs.
    cv : int
        Number of cross-validation folds.
    n_permutations : int
        Number of label permutations for null scoring.
    normalization : str
        Exploratory PCA unit-normalization mode. Rigorous mode fits a scaler in
        each CV split and ignores this value.
    include_score_summary : bool
        If ``True``, also compute average PC1/PC2 score summaries in one shared
        exploratory PCA coordinate system.
    include_raw_score_points : bool
        If ``True``, also compute raw trial-level PC1/PC2 score points in the
        same shared exploratory PCA coordinate system. Rows are condition
        memberships, so overlapping base conditions can duplicate a trial index.
    _spike_group
        Pynapple spike group keyed by unit id. Leading underscore excludes this
        potentially large object from Streamlit hashing.
    _trial_df : pd.DataFrame
        Trial table. Leading underscore excludes this dataframe from Streamlit
        hashing; ``session_key`` and decoding settings carry cache identity.

    Returns
    -------
    tuple[pd.DataFrame, int, pd.DataFrame, pd.DataFrame]
        ``(results_df, n_pca_trials, score_summary_df, raw_score_df)``.
        ``results_df`` is long-form decoding performance. ``n_pca_trials`` is
        the number of base-condition trials used to fit/extract the PCA decoding
        rates. ``score_summary_df`` and ``raw_score_df`` are empty unless their
        corresponding include flags are true.
    """

    if mode not in population_pca_decoding.PCA_DECODING_MODE_OPTIONS:
        raise ValueError(f"Unknown PCA decoding mode {mode!r}.")
    if len(condition_names) == 0:
        raise ValueError("Select at least one PCA decoding base condition.")

    trial_indices = population_pca_decoding.select_pca_decoding_trial_indices(
        trial_df=_trial_df,
        condition_names=condition_names,
    )
    if trial_indices.size == 0:
        raise ValueError("No valid trials match the selected PCA decoding base conditions.")
    rate_tensor_hz, _bin_centers_s = population_pca_decoding.build_choice_aligned_rate_tensor(
        spike_group=_spike_group,
        unit_ids=np.asarray(unit_ids, dtype=int),
        trial_df=_trial_df,
        trial_indices=trial_indices,
        bin_size_s=population_pca_decoding.PCA_DECODING_BIN_SIZE_S,
    )
    score_summary_df = pd.DataFrame(columns=population_pca_decoding.PCA_SCORE_SUMMARY_COLUMNS)
    raw_score_df = pd.DataFrame(columns=population_pca_decoding.PCA_RAW_SCORE_COLUMNS)
    if include_score_summary or include_raw_score_points:
        _trial_bins, exploratory_pca_result = population_pca_decoding.build_exploratory_pca_decoder_trial_bins(
            rate_tensor_hz=rate_tensor_hz,
            trial_df=_trial_df,
            trial_indices=trial_indices,
            n_components=int(n_components),
            normalization=normalization,
        )
        if include_score_summary:
            score_summary_df = population_pca_decoding.summarize_pca_scores_by_condition_and_target(
                pca_scores=exploratory_pca_result.scores,
                trial_df=_trial_df,
                trial_indices=trial_indices,
                condition_names=condition_names,
                target=target,
            )
        if include_raw_score_points:
            raw_score_df = population_pca_decoding.extract_pca_score_points_by_condition_and_target(
                pca_scores=exploratory_pca_result.scores,
                trial_df=_trial_df,
                trial_indices=trial_indices,
                condition_names=condition_names,
                target=target,
            )
    if mode == population_pca_decoding.PCA_DECODING_MODE_EXPLORATORY:
        results_df = population_pca_decoding.run_exploratory_pca_choice_decoding(
            rate_tensor_hz=rate_tensor_hz,
            trial_df=_trial_df,
            trial_indices=trial_indices,
            n_components=int(n_components),
            condition_names=condition_names,
            target=target,
            cv=int(cv),
            n_permutations=int(n_permutations),
            random_state=42,
            normalization=normalization,
        )
    else:
        results_df = population_pca_decoding.run_rigorous_pca_choice_decoding(
            rate_tensor_hz=rate_tensor_hz,
            trial_df=_trial_df,
            trial_indices=trial_indices,
            n_components=int(n_components),
            condition_names=condition_names,
            target=target,
            cv=int(cv),
            n_permutations=int(n_permutations),
            random_state=42,
        )
    return results_df, int(trial_indices.size), score_summary_df, raw_score_df

@st.cache_data(show_spinner="Computing PCA switch trajectories...")
def compute_population_pca_switch_trajectories_cached(
    session_key: str,
    aligned_spike_path: str,
    unit_ids: tuple[int, ...],
    pre_switch_filter: str,
    normalization: str,
    _spike_group,
    _trial_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Fit choice-aligned PCA on all valid trials and extract switch trajectories.

    Parameters
    ----------
    session_key : str
        Session/probe identifier included in the Streamlit cache key.
    aligned_spike_path : str
        Aligned spike file path included in the cache key. Units: filesystem
        path.
    unit_ids : tuple[int, ...]
        Unit ids used as PCA features, shape ``(n_units,)``.
    pre_switch_filter : str
        Previous-trial outcome filter from
        ``population_pca_switch_trajectories.SWITCH_PRE_FILTER_OPTIONS``.
    normalization : str
        Unit normalization mode passed to ``fit_population_pca``.
    _spike_group
        Pynapple spike group keyed by unit id. The leading underscore excludes
        this potentially large object from Streamlit hashing.
    _trial_df : pd.DataFrame
        Full trial table with choice times in seconds and scalar action,
        correctness, and reward labels. The leading underscore excludes the
        dataframe from Streamlit hashing.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, int]
        ``(trajectory_df, event_count_df, n_pca_trials)``. ``trajectory_df``
        has four PC1/PC2 rows per plotted switch event. ``event_count_df`` has
        one row per supported correctness transition. ``n_pca_trials`` is the
        number of all-valid trials used for the PCA fit.
    """

    trial_indices = population_pca_switch_trajectories.select_valid_choice_trial_indices(
        _trial_df
    )
    if trial_indices.size == 0:
        raise ValueError("No valid trials have finite choice times and actions.")
    rate_tensor_hz, _bin_centers_s = population_pca_decoding.build_choice_aligned_rate_tensor(
        spike_group=_spike_group,
        unit_ids=np.asarray(unit_ids, dtype=int),
        trial_df=_trial_df,
        trial_indices=trial_indices,
        bin_size_s=population_pca_decoding.PCA_DECODING_BIN_SIZE_S,
    )
    pca_result = population_pca.fit_population_pca(
        rate_tensor_hz=rate_tensor_hz,
        n_components=2,
        normalization=normalization,
    )
    switch_events = population_pca_switch_trajectories.select_choice_switch_events(
        trial_df=_trial_df,
        pre_switch_filter=pre_switch_filter,
    )
    trajectory_df = population_pca_switch_trajectories.extract_switch_event_pca_trajectories(
        pca_scores=pca_result.scores,
        pca_trial_indices=trial_indices,
        switch_events=switch_events,
    )
    event_count_df = population_pca_switch_trajectories.summarize_switch_event_counts(
        switch_events
    )
    return trajectory_df, event_count_df, int(trial_indices.size)
