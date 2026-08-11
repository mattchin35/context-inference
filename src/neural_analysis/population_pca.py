from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pynapple as nap
from sklearn.decomposition import PCA


PCA_NORMALIZATION_ZSCORE = "Z-score units"
PCA_NORMALIZATION_MEAN_CENTER = "Mean-center only"
PCA_NORMALIZATION_OPTIONS = (PCA_NORMALIZATION_ZSCORE, PCA_NORMALIZATION_MEAN_CENTER)


@dataclass(frozen=True)
class PopulationPCAResult:
    """
    Population PCA output for trial-time firing-rate observations.

    Attributes
    ----------
    scores : np.ndarray
        PC scores with shape ``(n_trials, n_bins, n_components)``. Scores are
        in standardized firing-rate space for z-scored PCA and in centered-Hz
        space for mean-center-only PCA.
    explained_variance_ratio : np.ndarray
        Fraction of variance explained by each fit component, shape
        ``(n_components,)``.
    cumulative_explained_variance : np.ndarray
        Cumulative explained variance ratio, shape ``(n_components,)``.
    unit_mean_hz : np.ndarray
        Mean firing rate subtracted from each unit, shape ``(n_units,)`` in Hz.
    unit_scale_hz : np.ndarray
        Unit scale used for normalization, shape ``(n_units,)`` in Hz. Values
        are one for mean-center-only PCA and for zero-variance units.
    normalization : str
        Normalization mode used before PCA.
    """

    scores: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray
    unit_mean_hz: np.ndarray
    unit_scale_hz: np.ndarray
    normalization: str


def _make_bin_edges(window: tuple[float, float], bin_size_s: float) -> np.ndarray:
    """
    Build fixed-width bin edges for a relative trial window.

    Parameters
    ----------
    window : tuple[float, float]
        Relative window bounds in seconds as ``(start_s, end_s)``.
    bin_size_s : float
        Bin width in seconds.

    Returns
    -------
    np.ndarray
        One-dimensional bin edges with shape ``(n_bins + 1,)`` in seconds.
        Any fractional final bin is omitted.
    """

    if float(bin_size_s) <= 0:
        raise ValueError("bin_size_s must be positive.")
    if len(window) != 2 or float(window[0]) >= float(window[1]):
        raise ValueError("window must be a two-value tuple with start < end.")
    window_start = float(window[0])
    window_end = float(window[1])
    n_bins = int(np.floor((window_end - window_start) / float(bin_size_s) + 1e-12))
    if n_bins < 1:
        raise ValueError("window must contain at least one complete bin.")
    return window_start + np.arange(n_bins + 1, dtype=float) * float(bin_size_s)


def _get_spike_times_seconds(spike_group: nap.TsGroup, unit_id: int) -> np.ndarray:
    """
    Return spike times for one unit as a sorted one-dimensional array.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Times are in seconds.
    unit_id : int
        Unit id to extract.

    Returns
    -------
    np.ndarray
        One-dimensional spike-time array with shape ``(n_spikes,)`` in seconds.
    """

    if int(unit_id) not in spike_group:
        raise ValueError(f"Unit {unit_id} is not present in the spike group.")
    return np.asarray(spike_group[int(unit_id)].index.to_numpy(), dtype=float).reshape(-1)


def build_trial_unit_rate_tensor(
    spike_group: nap.TsGroup,
    unit_ids: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    alignment_event: str,
    window: tuple[float, float],
    bin_size_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin selected units into trial-aligned firing rates for population PCA.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike times are in
        seconds.
    unit_ids : np.ndarray
        One-dimensional unit ids with shape ``(n_units,)``. All units are
        included as PCA features.
    trial_df : pd.DataFrame
        Trial table with one row per trial and an ``alignment_event`` time
        column in seconds.
    trial_indices : np.ndarray
        One-dimensional trial row indices with shape ``(n_trials,)``.
    alignment_event : str
        Trial time column used as time zero, such as ``"choice_time"`` or
        ``"start_time"``.
    window : tuple[float, float]
        Relative window bounds in seconds as ``(start_s, end_s)``.
    bin_size_s : float
        Bin width in seconds. Counts are divided by this value to produce Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(rate_tensor_hz, bin_centers_s)``. ``rate_tensor_hz`` has shape
        ``(n_trials, n_bins, n_units)`` in Hz. ``bin_centers_s`` has shape
        ``(n_bins,)`` in seconds relative to ``alignment_event``.
    """

    if not isinstance(spike_group, nap.TsGroup):
        raise TypeError("spike_group must be a pynapple TsGroup.")
    if alignment_event not in trial_df.columns:
        raise ValueError(f"trial_df is missing alignment event column {alignment_event!r}.")

    normalized_unit_ids = np.asarray(unit_ids, dtype=int).reshape(-1)
    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if normalized_unit_ids.size == 0:
        raise ValueError("unit_ids must contain at least one unit.")
    if normalized_trial_indices.size == 0:
        raise ValueError("trial_indices must contain at least one trial.")

    bin_edges = _make_bin_edges(window=window, bin_size_s=bin_size_s)
    bin_centers_s = bin_edges[:-1] + np.diff(bin_edges) / 2.0
    rate_tensor_hz = np.zeros(
        (normalized_trial_indices.size, bin_centers_s.size, normalized_unit_ids.size),
        dtype=float,
    )
    unit_spike_times = {
        int(unit_id): _get_spike_times_seconds(spike_group, int(unit_id))
        for unit_id in normalized_unit_ids
    }

    for trial_position, trial_index in enumerate(normalized_trial_indices):
        alignment_value = pd.to_numeric(
            pd.Series([trial_df.loc[int(trial_index), alignment_event]]),
            errors="coerce",
        ).iloc[0]
        if pd.isna(alignment_value):
            continue
        alignment_time_s = float(alignment_value)
        for unit_position, unit_id in enumerate(normalized_unit_ids):
            relative_spikes_s = unit_spike_times[int(unit_id)] - alignment_time_s
            spike_counts, _ = np.histogram(relative_spikes_s, bins=bin_edges)
            rate_tensor_hz[trial_position, :, unit_position] = spike_counts.astype(float) / float(bin_size_s)
    return rate_tensor_hz, bin_centers_s


def prepare_pca_observation_matrix(
    rate_tensor_hz: np.ndarray,
    normalization: str = PCA_NORMALIZATION_ZSCORE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten trial-time rates and normalize units for PCA.

    Parameters
    ----------
    rate_tensor_hz : np.ndarray
        Firing-rate tensor with shape ``(n_trials, n_bins, n_units)`` in Hz.
    normalization : str, default=PCA_NORMALIZATION_ZSCORE
        Normalization mode. ``PCA_NORMALIZATION_ZSCORE`` subtracts each unit's
        mean and divides by its standard deviation. ``PCA_NORMALIZATION_MEAN_CENTER``
        subtracts each unit's mean only.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(observations, unit_mean_hz, unit_scale_hz)``. ``observations`` has
        shape ``(n_trials * n_bins, n_units)``. Means and scales have shape
        ``(n_units,)`` in Hz.
    """

    rates = np.asarray(rate_tensor_hz, dtype=float)
    if rates.ndim != 3:
        raise ValueError("rate_tensor_hz must have shape (n_trials, n_bins, n_units).")
    if not np.isfinite(rates).all():
        raise ValueError("rate_tensor_hz must contain only finite values.")
    if normalization not in PCA_NORMALIZATION_OPTIONS:
        raise ValueError(f"normalization must be one of {PCA_NORMALIZATION_OPTIONS}.")

    n_trials, n_bins, n_units = rates.shape
    observations = rates.reshape(n_trials * n_bins, n_units)
    unit_mean_hz = observations.mean(axis=0)
    centered_observations = observations - unit_mean_hz
    if normalization == PCA_NORMALIZATION_ZSCORE:
        unit_scale_hz = centered_observations.std(axis=0)
        unit_scale_hz = np.where(unit_scale_hz > 0, unit_scale_hz, 1.0)
    else:
        unit_scale_hz = np.ones(n_units, dtype=float)
    normalized_observations = centered_observations / unit_scale_hz
    return normalized_observations, unit_mean_hz, unit_scale_hz


def fit_population_pca(
    rate_tensor_hz: np.ndarray,
    n_components: int,
    normalization: str = PCA_NORMALIZATION_ZSCORE,
) -> PopulationPCAResult:
    """
    Fit PCA to trial-time population firing-rate observations.

    Parameters
    ----------
    rate_tensor_hz : np.ndarray
        Firing-rate tensor with shape ``(n_trials, n_bins, n_units)`` in Hz.
        Axes are trial, time bin, unit.
    n_components : int
        Requested number of principal components.
    normalization : str, default=PCA_NORMALIZATION_ZSCORE
        Unit normalization applied before PCA.

    Returns
    -------
    PopulationPCAResult
        PCA scores and explained-variance metadata. The number of returned
        components is capped at ``min(n_components, n_units, n_observations)``.
    """

    rates = np.asarray(rate_tensor_hz, dtype=float)
    if rates.ndim != 3:
        raise ValueError("rate_tensor_hz must have shape (n_trials, n_bins, n_units).")
    n_trials, n_bins, n_units = rates.shape
    n_observations = n_trials * n_bins
    if n_units < 2:
        raise ValueError("At least two units are required for population PCA.")
    if n_observations < 2:
        raise ValueError("At least two observations are required for population PCA.")
    if int(n_components) < 1:
        raise ValueError("n_components must be positive.")

    observations, unit_mean_hz, unit_scale_hz = prepare_pca_observation_matrix(
        rates,
        normalization=normalization,
    )
    n_components_fit = min(int(n_components), n_units, n_observations)
    pca_model = PCA(n_components=n_components_fit)
    flat_scores = pca_model.fit_transform(observations)
    scores = flat_scores.reshape(n_trials, n_bins, n_components_fit)
    explained_variance_ratio = np.asarray(pca_model.explained_variance_ratio_, dtype=float)
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    return PopulationPCAResult(
        scores=scores,
        explained_variance_ratio=explained_variance_ratio,
        cumulative_explained_variance=cumulative_explained_variance,
        unit_mean_hz=unit_mean_hz,
        unit_scale_hz=unit_scale_hz,
        normalization=normalization,
    )
