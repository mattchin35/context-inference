from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.neural_analysis import population_pca
from src.neural_analysis import spike_behavior_pynapple


PCA_DECODING_MODE_EXPLORATORY = "Exploratory"
PCA_DECODING_MODE_RIGOROUS = "Rigorous"
PCA_DECODING_MODE_OPTIONS = (PCA_DECODING_MODE_EXPLORATORY, PCA_DECODING_MODE_RIGOROUS)
PCA_DECODING_BASE_CONDITIONS = [
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
]
PCA_DECODING_WINDOWS = {
    "pre_choice": (-0.5, 0.0),
    "post_choice": (0.0, 0.5),
}
PCA_DECODING_BIN_SIZE_S = 0.5
PCA_DECODING_WINDOW = (-0.5, 0.5)
PCA_DECODING_DISPLAY_COLUMNS = [
    "condition",
    "window",
    "target",
    "mode",
    "status",
    "reason",
    "n_samples",
    "n_classes",
    "cv_score",
    "cv_pvalue",
    "permutation_score_mean",
    "permutation_score_std",
]
PCA_SCORE_SUMMARY_COLUMNS = [
    "condition",
    "window",
    "target",
    "target_value",
    "target_label",
    "pc1_mean",
    "pc2_mean",
    "n_trials",
]
PCA_RAW_SCORE_COLUMNS = [
    "condition",
    "window",
    "target",
    "target_value",
    "target_label",
    "trial_index",
    "pc1",
    "pc2",
]


def build_choice_aligned_rate_tensor(
    spike_group: nap.TsGroup,
    unit_ids: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    bin_size_s: float = PCA_DECODING_BIN_SIZE_S,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the fixed choice-aligned firing-rate tensor used by PCA decoding.

    Parameters
    ----------
    spike_group : nap.TsGroup
        Pynapple spike group keyed by integer cluster id. Spike times are in
        seconds.
    unit_ids : np.ndarray
        One-dimensional unit ids with shape ``(n_units,)``. These are the raw
        neural features before PCA.
    trial_df : pd.DataFrame
        Trial table with one row per trial and a numeric ``choice_time`` column
        in seconds.
    trial_indices : np.ndarray
        One-dimensional trial row positions with shape ``(n_trials,)``.
    bin_size_s : float, default=0.5
        Bin width in seconds. The initial decoding view uses ``0.5`` so each
        trial contributes one pre-choice and one post-choice sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(rate_tensor_hz, bin_centers_s)``. ``rate_tensor_hz`` has shape
        ``(n_trials, 2, n_units)`` in Hz for the fixed ``(-0.5, 0.5)`` window.
        ``bin_centers_s`` has shape ``(2,)`` in seconds relative to choice.
    """

    return population_pca.build_trial_unit_rate_tensor(
        spike_group=spike_group,
        unit_ids=unit_ids,
        trial_df=trial_df,
        trial_indices=trial_indices,
        alignment_event="choice_time",
        window=PCA_DECODING_WINDOW,
        bin_size_s=float(bin_size_s),
    )


def select_pca_decoding_trial_indices(
    trial_df: pd.DataFrame,
    condition_names: list[str] | tuple[str, ...] = tuple(PCA_DECODING_BASE_CONDITIONS),
) -> np.ndarray:
    """
    Select trials eligible for PCA fitting under the base decoding conditions.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Must contain the columns required
        by ``spike_behavior_pynapple.make_trial_type_masks`` and a numeric
        ``choice_time`` column in seconds.
    condition_names : list[str] | tuple[str, ...], default=base conditions
        Base condition masks to include in the PCA fit. Conditions are combined
        by logical OR. These are decoding-specific conditions, not the generic
        webapp trial filters.

    Returns
    -------
    np.ndarray
        One-dimensional integer trial positions with shape ``(n_selected,)``.
        Only trials in at least one requested base condition and with finite
        choice times are returned.
    """

    if "choice_time" not in trial_df.columns:
        raise ValueError("trial_df is missing required choice_time column.")
    trial_masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)
    finite_choice_mask = pd.to_numeric(trial_df["choice_time"], errors="coerce").notna()
    selected_mask = pd.Series(False, index=trial_df.index)
    for condition_name in condition_names:
        if condition_name not in trial_masks:
            raise ValueError(f"Unknown PCA decoding condition {condition_name!r}.")
        selected_mask = selected_mask | trial_masks[condition_name]
    selected_mask = selected_mask & finite_choice_mask
    return np.flatnonzero(np.asarray(selected_mask, dtype=bool))


def build_valid_base_condition_masks(
    trial_df: pd.DataFrame,
    condition_names: list[str] | tuple[str, ...],
) -> dict[str, pd.Series]:
    """
    Build base-condition masks restricted to finite choice-aligned trials.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial and a numeric ``choice_time`` column
        in seconds.
    condition_names : list[str] | tuple[str, ...]
        Names from ``PCA_DECODING_BASE_CONDITIONS`` to return.

    Returns
    -------
    dict[str, pd.Series]
        Mapping from condition name to a boolean mask indexed like
        ``trial_df``. Masks exclude trials with missing ``choice_time``.
    """

    if "choice_time" not in trial_df.columns:
        raise ValueError("trial_df is missing required choice_time column.")
    trial_masks = spike_behavior_pynapple.make_trial_type_masks(trial_df)
    finite_choice_mask = pd.to_numeric(trial_df["choice_time"], errors="coerce").notna()
    valid_masks: dict[str, pd.Series] = {}
    for condition_name in condition_names:
        if condition_name not in trial_masks:
            raise ValueError(f"Unknown PCA decoding condition {condition_name!r}.")
        valid_masks[condition_name] = trial_masks[condition_name] & finite_choice_mask
    return valid_masks


def build_exploratory_pca_decoder_trial_bins(
    rate_tensor_hz: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    n_components: int,
    normalization: str = population_pca.PCA_NORMALIZATION_ZSCORE,
) -> tuple[list[dict[str, Any]], population_pca.PopulationPCAResult]:
    """
    Convert choice-window PCA scores into the existing decoder-bin structure.

    Parameters
    ----------
    rate_tensor_hz : np.ndarray
        Choice-aligned firing-rate tensor with shape ``(n_selected_trials, 2,
        n_units)`` in Hz. Axes are selected trial, choice-window bin, unit.
    trial_df : pd.DataFrame
        Full trial table with one row per trial. Required columns are
        ``choice_time``, ``state_int``, and ``action``. Times are in seconds.
    trial_indices : np.ndarray
        One-dimensional full-table trial positions represented by
        ``rate_tensor_hz``, shape ``(n_selected_trials,)``.
    n_components : int
        Requested number of PCA components. The fit count is capped by the
        PCA implementation.
    normalization : str, default=PCA_NORMALIZATION_ZSCORE
        Unit normalization applied before exploratory PCA.

    Returns
    -------
    tuple[list[dict[str, Any]], population_pca.PopulationPCAResult]
        ``trial_bins`` has length ``len(trial_df)``. Entries corresponding to
        ``trial_indices`` contain PC scores under the legacy key
        ``"binned_spikes"`` with shape ``(n_fit_pcs, 2)``. Scores are in PCA
        units, not spike counts. ``pca_result`` contains the full PCA result.
    """

    _validate_decoder_trial_table(trial_df)
    normalized_trial_indices = _normalize_trial_indices(trial_indices, trial_df)
    rates = np.asarray(rate_tensor_hz, dtype=float)
    if rates.ndim != 3:
        raise ValueError("rate_tensor_hz must have shape (n_trials, 2, n_units).")
    if rates.shape[0] != normalized_trial_indices.size:
        raise ValueError("rate_tensor_hz trial axis must match trial_indices length.")
    if rates.shape[1] != 2:
        raise ValueError("rate_tensor_hz must contain exactly two choice-window bins.")

    pca_result = population_pca.fit_population_pca(
        rates,
        n_components=int(n_components),
        normalization=normalization,
    )
    trial_bins = _build_decoder_trial_bins_from_scores(
        pca_scores=pca_result.scores,
        trial_df=trial_df,
        trial_indices=normalized_trial_indices,
    )
    return trial_bins, pca_result


def run_exploratory_pca_choice_decoding(
    rate_tensor_hz: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    n_components: int,
    condition_names: list[str] | tuple[str, ...],
    target: str,
    cv: int = 5,
    n_permutations: int = 100,
    random_state: int = 42,
    normalization: str = population_pca.PCA_NORMALIZATION_ZSCORE,
) -> pd.DataFrame:
    """
    Decode trial variables from PCA scores fit once on selected trial windows.

    Parameters
    ----------
    rate_tensor_hz : np.ndarray
        Choice-aligned firing-rate tensor with shape ``(n_selected_trials, 2,
        n_units)`` in Hz.
    trial_df : pd.DataFrame
        Full trial table with base-condition columns, ``choice_time``,
        ``state_int``, and ``action``.
    trial_indices : np.ndarray
        Full-table trial positions represented in ``rate_tensor_hz``.
    n_components : int
        Requested number of leading PCs.
    condition_names : list[str] | tuple[str, ...]
        Base decoding conditions to evaluate.
    target : str
        Decode target. Supported values are ``"state_int"`` and ``"action"``.
    cv : int, default=5
        Number of cross-validation folds.
    n_permutations : int, default=100
        Number of label permutations for null scoring.
    random_state : int, default=42
        Random seed passed to stochastic decoder steps.
    normalization : str, default=PCA_NORMALIZATION_ZSCORE
        Unit normalization applied before exploratory PCA.

    Returns
    -------
    pd.DataFrame
        Long-form decoding results with one row per condition and pre/post
        choice window. The schema includes ``PCA_DECODING_DISPLAY_COLUMNS`` and
        PCA metadata columns.
    """

    _validate_decode_target(trial_df, target)
    trial_bins, pca_result = build_exploratory_pca_decoder_trial_bins(
        rate_tensor_hz=rate_tensor_hz,
        trial_df=trial_df,
        trial_indices=trial_indices,
        n_components=int(n_components),
        normalization=normalization,
    )
    condition_masks = build_valid_base_condition_masks(trial_df, condition_names)
    collected_bins = spike_behavior_pynapple.collect_condition_classifier_bins(
        region_trial_binned=trial_bins,
        trial_df=trial_df,
        trial_masks=condition_masks,
        condition_names=list(condition_names),
        windows=PCA_DECODING_WINDOWS,
        event="choice_time",
    )
    rows = _decode_collected_condition_bins(
        collected_bins=collected_bins,
        condition_names=condition_names,
        target=target,
        mode=PCA_DECODING_MODE_EXPLORATORY,
        n_components_requested=int(n_components),
        n_components_fit=int(pca_result.scores.shape[2]),
        cv=int(cv),
        n_permutations=int(n_permutations),
        random_state=int(random_state),
    )
    return pd.DataFrame(rows)


def run_rigorous_pca_choice_decoding(
    rate_tensor_hz: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    n_components: int,
    condition_names: list[str] | tuple[str, ...],
    target: str,
    cv: int = 5,
    n_permutations: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Decode trial variables with scaler/PCA fit inside each CV training split.

    Parameters
    ----------
    rate_tensor_hz : np.ndarray
        Choice-aligned firing-rate tensor with shape ``(n_selected_trials, 2,
        n_units)`` in Hz.
    trial_df : pd.DataFrame
        Full trial table with base-condition columns, ``choice_time``,
        ``state_int``, and ``action``.
    trial_indices : np.ndarray
        Full-table trial positions represented in ``rate_tensor_hz``.
    n_components : int
        Requested number of leading PCs. The effective count is capped so PCA
        can be fit inside the smallest CV training fold.
    condition_names : list[str] | tuple[str, ...]
        Base decoding conditions to evaluate.
    target : str
        Decode target. Supported values are ``"state_int"`` and ``"action"``.
    cv : int, default=5
        Number of cross-validation folds.
    n_permutations : int, default=100
        Number of label permutations for null scoring.
    random_state : int, default=42
        Random seed passed to stochastic decoder steps.

    Returns
    -------
    pd.DataFrame
        Long-form decoding results with the same display schema as
        ``run_exploratory_pca_choice_decoding``.
    """

    _validate_decoder_trial_table(trial_df)
    _validate_decode_target(trial_df, target)
    normalized_trial_indices = _normalize_trial_indices(trial_indices, trial_df)
    rates = np.asarray(rate_tensor_hz, dtype=float)
    if rates.ndim != 3:
        raise ValueError("rate_tensor_hz must have shape (n_trials, 2, n_units).")
    if rates.shape[0] != normalized_trial_indices.size:
        raise ValueError("rate_tensor_hz trial axis must match trial_indices length.")
    if rates.shape[1] != 2:
        raise ValueError("rate_tensor_hz must contain exactly two choice-window bins.")

    condition_masks = build_valid_base_condition_masks(trial_df, condition_names)
    rate_position_by_trial = {
        int(trial_index): trial_position
        for trial_position, trial_index in enumerate(normalized_trial_indices)
    }

    rows: list[dict[str, Any]] = []
    for condition_name in condition_names:
        condition_mask = np.asarray(condition_masks[condition_name], dtype=bool)
        condition_trial_indices = [
            int(trial_index)
            for trial_index in np.flatnonzero(condition_mask)
            if int(trial_index) in rate_position_by_trial
        ]
        for window_name, bin_index in (("pre_choice", 0), ("post_choice", 1)):
            label = f"{condition_name}_{window_name}"
            if not condition_trial_indices:
                rows.append(
                    _failed_result_row(
                        reason="no_selected_bins",
                        label=label,
                        condition=condition_name,
                        window=window_name,
                        target=target,
                        mode=PCA_DECODING_MODE_RIGOROUS,
                        n_components_requested=int(n_components),
                        n_components_fit=0,
                    )
                )
                continue

            rate_positions = np.asarray(
                [rate_position_by_trial[int(trial_index)] for trial_index in condition_trial_indices],
                dtype=int,
            )
            raw_feature_bins = rates[rate_positions, bin_index, :].T
            target_values = pd.to_numeric(
                trial_df.iloc[condition_trial_indices][target],
                errors="coerce",
            ).to_numpy(dtype=float)
            decode_result = cv_decodeability_score_with_pca_pipeline(
                binned_rates=raw_feature_bins,
                target_values=target_values,
                n_components=int(n_components),
                cv=int(cv),
                n_permutations=int(n_permutations),
                random_state=int(random_state),
                label=label,
            )
            rows.append(
                {
                    "condition": condition_name,
                    "window": window_name,
                    "target": target,
                    "mode": PCA_DECODING_MODE_RIGOROUS,
                    "n_pcs_requested": int(n_components),
                    **decode_result,
                }
            )
    return pd.DataFrame(rows)


def cv_decodeability_score_with_pca_pipeline(
    binned_rates: np.ndarray,
    target_values: np.ndarray,
    n_components: int,
    cv: int = 5,
    n_permutations: int = 100,
    random_state: int = 42,
    label: str = "",
) -> dict[str, Any]:
    """
    Compute rigorous PCA decodeability with PCA fit inside CV folds.

    Parameters
    ----------
    binned_rates : np.ndarray
        Raw firing-rate feature matrix with shape ``(n_units, n_samples)`` in
        Hz.
    target_values : np.ndarray
        One-dimensional target labels with shape ``(n_samples,)``.
    n_components : int
        Requested PCA component count.
    cv : int, default=5
        Number of stratified cross-validation folds.
    n_permutations : int, default=100
        Number of label permutations for null scoring.
    random_state : int, default=42
        Random seed for CV shuffling, permutation testing, and logistic
        regression.
    label : str, default=""
        Human-readable label for the decoding run.

    Returns
    -------
    dict[str, Any]
        Decodeability metrics or a standardized failed-result dictionary. The
        returned ``n_pcs_fit`` is the PCA component count actually used inside
        the pipeline.
    """

    filtered_rates, filtered_targets, summary = _prepare_feature_target_inputs(
        feature_bins=binned_rates,
        target_values=target_values,
    )
    result_base = {"label": label, **summary}
    if summary["n_samples"] == 0:
        return _failed_decode_result("no_valid_samples", n_pcs_fit=0, **result_base)
    if summary["n_classes"] < 2:
        return _failed_decode_result("insufficient_classes", n_pcs_fit=0, **result_base)

    class_counts = np.unique(filtered_targets, return_counts=True)[1]
    if summary["n_samples"] < int(cv) or int(class_counts.min()) < int(cv):
        return _failed_decode_result("insufficient_samples", cv=int(cv), n_pcs_fit=0, **result_base)

    splitter = StratifiedKFold(n_splits=int(cv), shuffle=True, random_state=int(random_state))
    min_train_size = min(
        train_indices.size
        for train_indices, _test_indices in splitter.split(filtered_rates.T, filtered_targets)
    )
    n_components_fit = min(int(n_components), filtered_rates.shape[0], int(min_train_size))
    if n_components_fit < 1:
        return _failed_decode_result("insufficient_samples", cv=int(cv), n_pcs_fit=0, **result_base)

    classifier = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=int(n_components_fit))),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    max_iter=10000,
                    random_state=int(random_state),
                ),
            ),
        ]
    )
    cv_score, permutation_scores, cv_pvalue = permutation_test_score(
        classifier,
        filtered_rates.T,
        filtered_targets,
        scoring="accuracy",
        cv=splitter,
        n_permutations=int(n_permutations),
        random_state=int(random_state),
    )
    return {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "n_pcs_fit": int(n_components_fit),
        "cv_score": float(cv_score),
        "cv_pvalue": float(cv_pvalue),
        "permutation_scores": np.asarray(permutation_scores, dtype=float),
        "permutation_score_mean": float(np.mean(permutation_scores)),
        "permutation_score_std": float(np.std(permutation_scores)),
    }


def summarize_pca_decoding_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the compact columns shown in the PCA decoding webapp view.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form PCA decoding result table. Must contain
        ``PCA_DECODING_DISPLAY_COLUMNS``.

    Returns
    -------
    pd.DataFrame
        Shallow copy with only display columns, preserving row order.
    """

    missing_columns = set(PCA_DECODING_DISPLAY_COLUMNS) - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"results_df is missing required columns: {sorted(missing_columns)}")
    return results_df.loc[:, PCA_DECODING_DISPLAY_COLUMNS].copy()


def plot_pca_decoding_pre_post_scores(
    results_df: pd.DataFrame,
    figure_size: tuple[float, float] = (7.0, 3.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot side-by-side pre/post choice PCA decoding accuracy by condition.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form PCA decoding result table with ``condition``, ``window``,
        ``status``, and ``cv_score`` columns. Only rows with ``status == "ok"``
        and finite ``cv_score`` are plotted.
    figure_size : tuple[float, float], default=(7.0, 3.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing grouped bars. X-axis groups are base
        conditions; pre/post bars are horizontally adjacent within each group.
    """

    required_columns = {"condition", "window", "status", "cv_score"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"results_df is missing required columns: {sorted(missing_columns)}")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    ok_rows = results_df.loc[results_df["status"] == "ok"].copy()
    ok_rows["cv_score"] = pd.to_numeric(ok_rows["cv_score"], errors="coerce")
    ok_rows = ok_rows.loc[ok_rows["cv_score"].notna()]

    figure, axis = plt.subplots(1, 1, figsize=(float(figure_size[0]), float(figure_size[1])))
    if ok_rows.empty:
        axis.set_ylabel("CV decoding accuracy")
        axis.set_ylim(0.0, 1.0)
        axis.set_title("PCA decoding before/after choice")
        figure.tight_layout()
        return figure, axis

    condition_order = [
        condition_name
        for condition_name in PCA_DECODING_BASE_CONDITIONS
        if condition_name in set(ok_rows["condition"])
    ]
    condition_order.extend(
        condition_name
        for condition_name in ok_rows["condition"].tolist()
        if condition_name not in condition_order
    )
    condition_order = list(dict.fromkeys(condition_order))
    window_offsets = {"pre_choice": -0.18, "post_choice": 0.18}
    window_colors = {"pre_choice": "tab:blue", "post_choice": "tab:orange"}
    bar_width = 0.32
    group_spacing = 1.35
    group_centers = np.arange(len(condition_order), dtype=float) * group_spacing

    for condition_position, condition_name in enumerate(condition_order):
        condition_rows = ok_rows.loc[ok_rows["condition"] == condition_name]
        for window_name in ("pre_choice", "post_choice"):
            window_rows = condition_rows.loc[condition_rows["window"] == window_name]
            if window_rows.empty:
                continue
            bar_x = group_centers[condition_position] + window_offsets[window_name]
            axis.bar(
                bar_x,
                float(window_rows.iloc[0]["cv_score"]),
                width=bar_width,
                color=window_colors[window_name],
                label=window_name,
            )

    handles, labels = axis.get_legend_handles_labels()
    unique_labels: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        unique_labels.setdefault(label, handle)
    if unique_labels:
        axis.legend(unique_labels.values(), unique_labels.keys(), loc="upper right", fontsize="small")
    axis.set_xticks(group_centers)
    axis.set_xticklabels(condition_order, rotation=25, ha="right")
    axis.set_ylabel("CV decoding accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("PCA decoding before/after choice")
    figure.tight_layout()
    return figure, axis


def summarize_pca_scores_by_condition_and_target(
    pca_scores: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    condition_names: list[str] | tuple[str, ...],
    target: str,
) -> pd.DataFrame:
    """
    Average PC1/PC2 scores by base condition, choice window, and target value.

    Parameters
    ----------
    pca_scores : np.ndarray
        Exploratory PCA score tensor with shape ``(n_selected_trials, 2,
        n_pcs)``. Axes are selected trial, pre/post choice bin, and PC. Scores
        must contain at least two PCs.
    trial_df : pd.DataFrame
        Full trial table with base-condition columns and the requested target
        column. ``choice_time`` values are in seconds.
    trial_indices : np.ndarray
        Full-table trial positions represented by the first axis of
        ``pca_scores``.
    condition_names : list[str] | tuple[str, ...]
        Base condition masks to summarize.
    target : str
        Grouping target. Supported values are ``"state_int"`` and ``"action"``.

    Returns
    -------
    pd.DataFrame
        One row per populated ``condition x window x target_value`` group with
        columns ``PCA_SCORE_SUMMARY_COLUMNS``. ``pc1_mean`` and ``pc2_mean`` are
        PCA score means in the shared exploratory PCA coordinate system.
    """

    _validate_decode_target(trial_df, target)
    scores = np.asarray(pca_scores, dtype=float)
    if scores.ndim != 3:
        raise ValueError("pca_scores must have shape (n_trials, 2, n_pcs).")
    if scores.shape[1] != 2:
        raise ValueError("pca_scores must contain exactly two choice-window bins.")
    if scores.shape[2] < 2:
        raise ValueError("pca_scores must contain at least two PCs.")

    normalized_trial_indices = _normalize_trial_indices(trial_indices, trial_df)
    if scores.shape[0] != normalized_trial_indices.size:
        raise ValueError("pca_scores trial axis must match trial_indices length.")

    condition_masks = build_valid_base_condition_masks(trial_df, condition_names)
    target_values = pd.to_numeric(trial_df[target], errors="coerce").to_numpy(dtype=float)
    score_position_by_trial = {
        int(trial_index): score_position
        for score_position, trial_index in enumerate(normalized_trial_indices)
    }

    rows: list[dict[str, Any]] = []
    for condition_name in condition_names:
        condition_trial_indices = [
            int(trial_index)
            for trial_index in np.flatnonzero(np.asarray(condition_masks[condition_name], dtype=bool))
            if int(trial_index) in score_position_by_trial
        ]
        if not condition_trial_indices:
            continue

        condition_targets = target_values[np.asarray(condition_trial_indices, dtype=int)]
        unique_target_values = [
            float(target_value)
            for target_value in np.unique(condition_targets[np.isfinite(condition_targets)])
        ]
        for target_value in unique_target_values:
            matching_trials = [
                trial_index
                for trial_index in condition_trial_indices
                if np.isfinite(target_values[trial_index]) and float(target_values[trial_index]) == target_value
            ]
            if not matching_trials:
                continue
            score_positions = np.asarray(
                [score_position_by_trial[int(trial_index)] for trial_index in matching_trials],
                dtype=int,
            )
            for window_name, window_index in (("pre_choice", 0), ("post_choice", 1)):
                group_scores = scores[score_positions, window_index, :2]
                rows.append(
                    {
                        "condition": condition_name,
                        "window": window_name,
                        "target": target,
                        "target_value": float(target_value),
                        "target_label": format_pca_score_target_label(target=target, target_value=target_value),
                        "pc1_mean": float(np.mean(group_scores[:, 0])),
                        "pc2_mean": float(np.mean(group_scores[:, 1])),
                        "n_trials": int(score_positions.size),
                    }
                )
    return pd.DataFrame(rows, columns=PCA_SCORE_SUMMARY_COLUMNS)


def extract_pca_score_points_by_condition_and_target(
    pca_scores: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
    condition_names: list[str] | tuple[str, ...],
    target: str,
) -> pd.DataFrame:
    """
    Extract raw PC1/PC2 score points by condition, choice window, and target value.

    Parameters
    ----------
    pca_scores : np.ndarray
        Exploratory PCA score tensor with shape ``(n_selected_trials, 2,
        n_pcs)``. Axes are selected trial, pre/post choice bin, and PC. Scores
        must contain at least two PCs.
    trial_df : pd.DataFrame
        Full trial table with base-condition columns and the requested target
        column. ``choice_time`` values are in seconds.
    trial_indices : np.ndarray
        Full-table trial positions represented by the first axis of
        ``pca_scores``.
    condition_names : list[str] | tuple[str, ...]
        Base condition masks to extract. Rows are condition memberships, not
        unique trials, so overlapping conditions can duplicate a trial index.
    target : str
        Grouping target. Supported values are ``"state_int"`` and ``"action"``.

    Returns
    -------
    pd.DataFrame
        One row per populated ``condition x trial x window`` membership with
        columns ``PCA_RAW_SCORE_COLUMNS``. ``pc1`` and ``pc2`` are PCA scores in
        the shared exploratory PCA coordinate system.
    """

    _validate_decode_target(trial_df, target)
    scores = np.asarray(pca_scores, dtype=float)
    if scores.ndim != 3:
        raise ValueError("pca_scores must have shape (n_trials, 2, n_pcs).")
    if scores.shape[1] != 2:
        raise ValueError("pca_scores must contain exactly two choice-window bins.")
    if scores.shape[2] < 2:
        raise ValueError("pca_scores must contain at least two PCs.")

    normalized_trial_indices = _normalize_trial_indices(trial_indices, trial_df)
    if scores.shape[0] != normalized_trial_indices.size:
        raise ValueError("pca_scores trial axis must match trial_indices length.")

    condition_masks = build_valid_base_condition_masks(trial_df, condition_names)
    target_values = pd.to_numeric(trial_df[target], errors="coerce").to_numpy(dtype=float)
    score_position_by_trial = {
        int(trial_index): score_position
        for score_position, trial_index in enumerate(normalized_trial_indices)
    }

    rows: list[dict[str, Any]] = []
    for condition_name in condition_names:
        condition_trial_indices = [
            int(trial_index)
            for trial_index in np.flatnonzero(np.asarray(condition_masks[condition_name], dtype=bool))
            if int(trial_index) in score_position_by_trial
        ]
        for trial_index in condition_trial_indices:
            target_value = target_values[trial_index]
            if not np.isfinite(target_value):
                continue
            score_position = score_position_by_trial[int(trial_index)]
            for window_name, window_index in (("pre_choice", 0), ("post_choice", 1)):
                rows.append(
                    {
                        "condition": condition_name,
                        "window": window_name,
                        "target": target,
                        "target_value": float(target_value),
                        "target_label": format_pca_score_target_label(target=target, target_value=target_value),
                        "trial_index": int(trial_index),
                        "pc1": float(scores[score_position, window_index, 0]),
                        "pc2": float(scores[score_position, window_index, 1]),
                    }
                )
    return pd.DataFrame(rows, columns=PCA_RAW_SCORE_COLUMNS)


def format_pca_score_target_label(target: str, target_value: float) -> str:
    """
    Format target values for PCA score summary displays.

    Parameters
    ----------
    target : str
        Target column name, either ``"state_int"`` or ``"action"``.
    target_value : float
        Numeric target value.

    Returns
    -------
    str
        Human-readable label for legends and tables.
    """

    if target == "state_int":
        return f"state {int(target_value)}"
    if target == "action":
        if int(target_value) == 0:
            return "right (0)"
        if int(target_value) == 1:
            return "left (1)"
        return f"action {int(target_value)}"
    raise ValueError("target must be 'state_int' or 'action'.")


def plot_average_pca_scores_by_condition_and_target(
    score_summary_df: pd.DataFrame,
    raw_score_df: pd.DataFrame | None = None,
    figure_size: tuple[float, float] = (8.0, 4.0),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot average PC1/PC2 score points by condition and target value.

    Parameters
    ----------
    score_summary_df : pd.DataFrame
        Output from ``summarize_pca_scores_by_condition_and_target`` with
        columns ``PCA_SCORE_SUMMARY_COLUMNS``.
    raw_score_df : pd.DataFrame | None, default=None
        Optional output from ``extract_pca_score_points_by_condition_and_target``.
        Rows are plotted as transparent condition-membership points beneath the
        group means. A trial can appear more than once if selected conditions
        overlap.
    figure_size : tuple[float, float], default=(8.0, 4.0)
        Matplotlib figure size as ``(width_inches, height_inches)``.

    Returns
    -------
    tuple[plt.Figure, np.ndarray]
        Figure and two axes. The first axis shows pre-choice group means; the
        second axis shows post-choice group means. X is mean PC1 and Y is mean
        PC2.
    """

    missing_columns = set(PCA_SCORE_SUMMARY_COLUMNS) - set(score_summary_df.columns)
    if missing_columns:
        raise ValueError(f"score_summary_df is missing required columns: {sorted(missing_columns)}")
    if raw_score_df is not None:
        missing_raw_columns = set(PCA_RAW_SCORE_COLUMNS) - set(raw_score_df.columns)
        if missing_raw_columns:
            raise ValueError(f"raw_score_df is missing required columns: {sorted(missing_raw_columns)}")
    if len(figure_size) != 2 or float(figure_size[0]) <= 0 or float(figure_size[1]) <= 0:
        raise ValueError("figure_size must be a two-value tuple of positive inches.")

    figure, axes = plt.subplots(
        1,
        2,
        sharex=True,
        sharey=True,
        figsize=(float(figure_size[0]), float(figure_size[1])),
    )
    axes = np.asarray(axes, dtype=object).reshape(-1)

    condition_values = score_summary_df["condition"].tolist()
    if raw_score_df is not None:
        condition_values.extend(raw_score_df["condition"].tolist())
    condition_order = [
        condition_name
        for condition_name in PCA_DECODING_BASE_CONDITIONS
        if condition_name in set(condition_values)
    ]
    condition_order.extend(
        condition_name
        for condition_name in condition_values
        if condition_name not in condition_order
    )
    condition_order = list(dict.fromkeys(condition_order))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    condition_colors = {
        condition_name: color_cycle[index % len(color_cycle)]
        for index, condition_name in enumerate(condition_order)
    }
    target_label_values = score_summary_df["target_label"].tolist()
    if raw_score_df is not None:
        target_label_values.extend(raw_score_df["target_label"].tolist())
    target_labels = list(dict.fromkeys(target_label_values))
    marker_cycle = ["o", "^", "s", "D", "P", "X"]
    target_markers = {
        target_label: marker_cycle[index % len(marker_cycle)]
        for index, target_label in enumerate(target_labels)
    }

    for axis, window_name, title in zip(
        axes,
        ("pre_choice", "post_choice"),
        ("Pre-choice (-0.5 to 0 s)", "Post-choice (0 to 0.5 s)"),
        strict=True,
    ):
        if raw_score_df is not None:
            raw_window_rows = raw_score_df.loc[raw_score_df["window"] == window_name]
            for _, row in raw_window_rows.iterrows():
                axis.scatter(
                    float(row["pc1"]),
                    float(row["pc2"]),
                    color=condition_colors[str(row["condition"])],
                    marker=target_markers[str(row["target_label"])],
                    s=18,
                    alpha=0.18,
                    edgecolor="none",
                    linewidth=0.0,
                )
        window_rows = score_summary_df.loc[score_summary_df["window"] == window_name]
        for _, row in window_rows.iterrows():
            axis.scatter(
                float(row["pc1_mean"]),
                float(row["pc2_mean"]),
                color=condition_colors[str(row["condition"])],
                marker=target_markers[str(row["target_label"])],
                s=70,
                edgecolor="black",
                linewidth=0.6,
                alpha=1.0,
            )
        axis.axhline(0.0, color="0.75", linewidth=0.8, zorder=0)
        axis.axvline(0.0, color="0.75", linewidth=0.8, zorder=0)
        axis.set_title(title)
        axis.set_xlabel("Mean PC1 score")
    axes[0].set_ylabel("Mean PC2 score")

    condition_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=condition_colors[condition_name],
            label=condition_name,
        )
        for condition_name in condition_order
    ]
    target_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=target_markers[target_label],
            linestyle="",
            color="black",
            markerfacecolor="white",
            label=target_label,
        )
        for target_label in target_labels
    ]
    if condition_handles:
        axes[1].legend(
            handles=[*condition_handles, *target_handles],
            loc="best",
            fontsize="small",
            title="Condition / target",
        )
    figure.tight_layout()
    return figure, axes


def _validate_decoder_trial_table(trial_df: pd.DataFrame) -> None:
    """
    Validate columns needed to adapt PCA scores to decoder bins.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial.

    Returns
    -------
    None
        Raises ``ValueError`` if required columns are missing.
    """

    required_columns = {"choice_time", "state_int", "action"}
    missing_columns = required_columns - set(trial_df.columns)
    if missing_columns:
        raise ValueError(f"trial_df is missing required columns: {sorted(missing_columns)}")


def _validate_decode_target(trial_df: pd.DataFrame, target: str) -> None:
    """
    Validate the requested decode target.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial.
    target : str
        Requested target column. Supported values are ``"state_int"`` and
        ``"action"``.

    Returns
    -------
    None
        Raises ``ValueError`` if the target is unsupported or missing.
    """

    if target not in {"state_int", "action"}:
        raise ValueError("target must be 'state_int' or 'action'.")
    if target not in trial_df.columns:
        raise ValueError(f"trial_df is missing requested target column {target!r}.")


def _normalize_trial_indices(trial_indices: np.ndarray, trial_df: pd.DataFrame) -> np.ndarray:
    """
    Normalize full-table trial positions for decoder-rate tensors.

    Parameters
    ----------
    trial_indices : np.ndarray
        One-dimensional trial row positions.
    trial_df : pd.DataFrame
        Trial table used for bounds checking.

    Returns
    -------
    np.ndarray
        One-dimensional integer trial positions.
    """

    normalized_trial_indices = np.asarray(trial_indices, dtype=int).reshape(-1)
    if normalized_trial_indices.size == 0:
        raise ValueError("trial_indices must contain at least one trial.")
    if normalized_trial_indices.min() < 0 or normalized_trial_indices.max() >= len(trial_df):
        raise ValueError("trial_indices must be row positions within trial_df.")
    return normalized_trial_indices


def _build_decoder_trial_bins_from_scores(
    pca_scores: np.ndarray,
    trial_df: pd.DataFrame,
    trial_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Build decoder-compatible trial dictionaries from PCA score tensors.

    Parameters
    ----------
    pca_scores : np.ndarray
        PCA score tensor with shape ``(n_selected_trials, 2, n_pcs)``.
    trial_df : pd.DataFrame
        Full trial table with one row per trial.
    trial_indices : np.ndarray
        Full-table row positions corresponding to the first axis of
        ``pca_scores``.

    Returns
    -------
    list[dict[str, Any]]
        One decoder-compatible dictionary per trial row. PCA score entries are
        stored as ``"binned_spikes"`` with shape ``(n_pcs, 2)`` for
        compatibility with existing decoding helpers.
    """

    scores = np.asarray(pca_scores, dtype=float)
    if scores.ndim != 3 or scores.shape[1] != 2:
        raise ValueError("pca_scores must have shape (n_trials, 2, n_pcs).")
    if scores.shape[0] != trial_indices.size:
        raise ValueError("pca_scores trial axis must match trial_indices length.")

    n_pcs = int(scores.shape[2])
    score_position_by_trial = {
        int(trial_index): trial_position
        for trial_position, trial_index in enumerate(trial_indices)
    }
    trial_bins: list[dict[str, Any]] = []
    for trial_position in range(len(trial_df)):
        trial_row = trial_df.iloc[trial_position]
        choice_time = pd.to_numeric(pd.Series([trial_row["choice_time"]]), errors="coerce").iloc[0]
        if pd.isna(choice_time):
            bin_edges = np.full(3, np.nan, dtype=float)
        else:
            bin_edges = float(choice_time) + np.array([-0.5, 0.0, 0.5], dtype=float)

        if trial_position in score_position_by_trial:
            selected_position = score_position_by_trial[trial_position]
            feature_bins = scores[selected_position, :, :].T
        else:
            feature_bins = np.zeros((n_pcs, 2), dtype=float)

        state_value = pd.to_numeric(pd.Series([trial_row["state_int"]]), errors="coerce").iloc[0]
        action_value = pd.to_numeric(pd.Series([trial_row["action"]]), errors="coerce").iloc[0]
        trial_bins.append(
            {
                "trial_ix": int(trial_position),
                "binned_spikes": feature_bins,
                "bin_edges": bin_edges,
                "bin_states": np.full(2, float(state_value) if not pd.isna(state_value) else np.nan, dtype=float),
                "bin_choices": np.full(2, float(action_value) if not pd.isna(action_value) else np.nan, dtype=float),
            }
        )
    return trial_bins


def _decode_collected_condition_bins(
    collected_bins: Mapping[tuple[str, str], dict[str, Any]],
    condition_names: list[str] | tuple[str, ...],
    target: str,
    mode: str,
    n_components_requested: int,
    n_components_fit: int,
    cv: int,
    n_permutations: int,
    random_state: int,
) -> list[dict[str, Any]]:
    """
    Decode collected condition/window feature matrices with existing helpers.

    Parameters
    ----------
    collected_bins : Mapping[tuple[str, str], dict[str, Any]]
        Output from ``collect_condition_classifier_bins``.
    condition_names : list[str] | tuple[str, ...]
        Ordered condition names.
    target : str
        Decode target, either ``"state_int"`` or ``"action"``.
    mode : str
        PCA decoding mode label.
    n_components_requested : int
        Requested PC count.
    n_components_fit : int
        Fit PC count.
    cv : int
        Cross-validation fold count.
    n_permutations : int
        Number of permutation-test shuffles.
    random_state : int
        Random seed.

    Returns
    -------
    list[dict[str, Any]]
        Long-form decoding result rows.
    """

    rows: list[dict[str, Any]] = []
    for condition_name in condition_names:
        for window_name in PCA_DECODING_WINDOWS:
            label = f"{condition_name}_{window_name}"
            entry = collected_bins[(condition_name, window_name)]
            if entry["status"] != "ok":
                rows.append(
                    _failed_result_row(
                        reason=str(entry.get("reason", "no_selected_bins")),
                        label=label,
                        condition=condition_name,
                        window=window_name,
                        target=target,
                        mode=mode,
                        n_components_requested=int(n_components_requested),
                        n_components_fit=int(n_components_fit),
                    )
                )
                continue

            target_values = entry["state_bins"] if target == "state_int" else entry["choice_bins"]
            decode_result = spike_behavior_pynapple.cv_decodeability_score(
                binned_spikes=entry["spike_bins"],
                target_values=target_values,
                cv=int(cv),
                n_permutations=int(n_permutations),
                random_state=int(random_state),
                label=label,
            )
            rows.append(
                {
                    "condition": condition_name,
                    "window": window_name,
                    "target": target,
                    "mode": mode,
                    "n_pcs_requested": int(n_components_requested),
                    "n_pcs_fit": int(n_components_fit),
                    **decode_result,
                }
            )
    return rows


def _prepare_feature_target_inputs(
    feature_bins: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Filter raw feature matrices and target labels for decoding.

    Parameters
    ----------
    feature_bins : np.ndarray
        Feature matrix with shape ``(n_features, n_samples)``.
    target_values : np.ndarray
        One-dimensional target labels with shape ``(n_samples,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, int]]
        ``(filtered_features, filtered_targets, summary)`` after dropping
        samples with missing targets.
    """

    filtered_targets = np.asarray(target_values, dtype=float)
    filtered_features = np.asarray(feature_bins, dtype=float)
    if filtered_features.ndim != 2:
        raise ValueError("feature_bins must be a two-dimensional array with shape (n_features, n_samples).")
    if filtered_targets.ndim != 1:
        raise ValueError("target_values must be a one-dimensional array.")
    if filtered_features.shape[1] != filtered_targets.shape[0]:
        raise ValueError("feature_bins and target_values must agree on sample count.")

    valid_target_mask = ~np.isnan(filtered_targets)
    filtered_features = filtered_features[:, valid_target_mask]
    filtered_targets = filtered_targets[valid_target_mask]
    summary = {
        "n_samples": int(filtered_targets.shape[0]),
        "n_classes": int(np.unique(filtered_targets).size),
    }
    return filtered_features, filtered_targets, summary


def _failed_decode_result(reason: str, label: str = "", **extra_fields: Any) -> dict[str, Any]:
    """
    Build a standardized failed PCA-decoding result.

    Parameters
    ----------
    reason : str
        Machine-readable failure reason.
    label : str, default=""
        Human-readable condition/window label.
    **extra_fields : Any
        Additional metadata fields to include.

    Returns
    -------
    dict[str, Any]
        Result dictionary with failed status and NaN performance metrics.
    """

    result = {
        "status": "failed",
        "reason": reason,
        "label": label,
        "n_samples": int(extra_fields.pop("n_samples", 0)),
        "n_classes": int(extra_fields.pop("n_classes", 0)),
        "cv_score": np.nan,
        "cv_pvalue": np.nan,
        "permutation_scores": np.array([], dtype=float),
        "permutation_score_mean": np.nan,
        "permutation_score_std": np.nan,
    }
    result.update(extra_fields)
    return result


def _failed_result_row(
    reason: str,
    label: str,
    condition: str,
    window: str,
    target: str,
    mode: str,
    n_components_requested: int,
    n_components_fit: int,
) -> dict[str, Any]:
    """
    Build a full long-form failed row for condition/window decoding.

    Parameters
    ----------
    reason : str
        Failure reason.
    label : str
        Human-readable condition/window label.
    condition : str
        Base condition name.
    window : str
        Window name, ``"pre_choice"`` or ``"post_choice"``.
    target : str
        Decode target.
    mode : str
        PCA decoding mode.
    n_components_requested : int
        Requested PC count.
    n_components_fit : int
        Effective fit PC count.

    Returns
    -------
    dict[str, Any]
        Long-form failed decoding row.
    """

    return {
        "condition": condition,
        "window": window,
        "target": target,
        "mode": mode,
        "n_pcs_requested": int(n_components_requested),
        "n_pcs_fit": int(n_components_fit),
        **_failed_decode_result(reason=reason, label=label),
    }
