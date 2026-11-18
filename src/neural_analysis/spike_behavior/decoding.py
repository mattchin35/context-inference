"""Seeded behavioral decoding from trial-aligned spike-count matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import permutation_test_score, train_test_split

from src.neural_analysis.spike_behavior.binning import make_classifier_bins
from src.neural_analysis.spike_behavior.trials import make_trial_type_masks

def get_decode_target(trial_df: pd.DataFrame, target: str = "state_int") -> np.ndarray:
    """
    Extract one supported decoding target from the trial table.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial table with one row per trial. Must contain the requested target column.
    target : str, optional
        Name of the decode target column. Supported values are ``"state_int"`` and ``"action"``.

    Returns
    -------
    np.ndarray
        One-dimensional float array with shape ``(n_trials,)`` containing the requested target values.
    """

    if target not in {"state_int", "action"}:
        raise ValueError("target must be 'state_int' or 'action'.")
    if target not in trial_df.columns:
        raise ValueError(f"trial_df is missing requested target column {target!r}.")
    return pd.to_numeric(trial_df[target], errors="coerce").to_numpy(dtype=float)

def _prepare_decode_inputs(
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Filter decode inputs and summarize class counts.

    Parameters
    ----------
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, int]]
        ``(filtered_spikes, filtered_targets, summary)`` where ``filtered_spikes`` retains shape
        ``(n_units, n_valid_samples)``, ``filtered_targets`` has shape ``(n_valid_samples,)``,
        and ``summary`` contains integer counts for ``n_samples`` and ``n_classes``.
    """

    filtered_targets = np.asarray(target_values, dtype=float)
    filtered_spikes = np.asarray(binned_spikes, dtype=float)
    if filtered_spikes.ndim != 2:
        raise ValueError("binned_spikes must be a two-dimensional array with shape (n_units, n_samples).")
    if filtered_targets.ndim != 1:
        raise ValueError("target_values must be a one-dimensional array.")
    if filtered_spikes.shape[1] != filtered_targets.shape[0]:
        raise ValueError("binned_spikes and target_values must agree on sample count.")

    valid_target_mask = ~np.isnan(filtered_targets)
    filtered_spikes = filtered_spikes[:, valid_target_mask]
    filtered_targets = filtered_targets[valid_target_mask]
    summary = {
        "n_samples": int(filtered_targets.shape[0]),
        "n_classes": int(np.unique(filtered_targets).size),
    }
    return filtered_spikes, filtered_targets, summary

def _failed_decode_result(reason: str, label: str = "", **extra_fields: Any) -> dict[str, Any]:
    """
    Build a standardized failed-decoding result dictionary.

    Parameters
    ----------
    reason : str
        Short machine-readable failure reason.
    label : str, optional
        Optional human-readable label for the attempted decode.
    **extra_fields : Any
        Additional key-value pairs to include in the result dictionary.

    Returns
    -------
    dict[str, Any]
        Result dictionary with ``status == "failed"`` and the supplied metadata.
    """

    result = {"status": "failed", "reason": reason, "label": label}
    result.update(extra_fields)
    return result

def cv_decodeability_score(
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
    cv: int = 5,
    n_permutations: int = 100,
    random_state: int = 42,
    label: str = "",
) -> dict[str, Any]:
    """
    Compute a cross-validated decodeability score with a permutation-test null.

    Parameters
    ----------
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``. Values are class labels.
    cv : int, optional
        Number of cross-validation folds.
    n_permutations : int, optional
        Number of label permutations used for ``permutation_test_score``.
    random_state : int, optional
        Random seed passed to the permutation test and classifier.
    label : str, optional
        Human-readable label describing the decodeability run.

    Returns
    -------
    dict[str, Any]
        Result dictionary containing decodeability metrics or an explicit failure reason.
    """

    filtered_spikes, filtered_targets, summary = _prepare_decode_inputs(binned_spikes, target_values)
    result_base = {"label": label, **summary}
    if summary["n_samples"] == 0:
        return _failed_decode_result("no_valid_samples", **result_base)
    if summary["n_classes"] < 2:
        return _failed_decode_result("insufficient_classes", **result_base)

    class_counts = np.unique(filtered_targets, return_counts=True)[1]
    if summary["n_samples"] < cv or int(class_counts.min()) < cv:
        return _failed_decode_result("insufficient_samples", cv=cv, **result_base)

    classifier = LogisticRegression(
        solver="saga",
        # penalty="l1",
        l1_ratio=.5,  # L1 penalty is l1_ratio=1; for L2 set to 0; for elastic net set to .5
        max_iter=10000,
        random_state=random_state,
    )
    cv_score, permutation_scores, cv_pvalue = permutation_test_score(
        classifier,
        filtered_spikes.T,
        filtered_targets,
        scoring="accuracy",
        cv=cv,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    return {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "cv_score": float(cv_score),
        "cv_pvalue": float(cv_pvalue),
        "permutation_scores": np.asarray(permutation_scores, dtype=float),
        "permutation_score_mean": float(np.mean(permutation_scores)),
        "permutation_score_std": float(np.std(permutation_scores)),
    }

def summarize_decoding_results(
    results_df: pd.DataFrame,
    value_columns: list[str],
) -> pd.DataFrame:
    """
    Select a compact, ordered subset of decoding-result columns for display.

    Parameters
    ----------
    results_df : pd.DataFrame
        Decoding-results table with at least ``condition``, ``window``, ``status``, and ``reason`` columns.
    value_columns : list[str]
        Ordered list of metric columns to include after the core display columns.

    Returns
    -------
    pd.DataFrame
        A shallow copy of ``results_df`` containing only the requested display columns in a stable order.
    """

    required_columns = {"condition", "window", "status", "reason"}
    missing_columns = required_columns - set(results_df.columns)
    if missing_columns:
        raise ValueError(f"results_df is missing required columns: {sorted(missing_columns)}")

    selected_columns = ["condition", "window", "status", "reason", *value_columns]
    for column_name in value_columns:
        if column_name not in results_df.columns:
            raise ValueError(f"results_df is missing requested value column {column_name!r}.")

    return results_df.loc[:, selected_columns].copy()

def _collect_base_condition_decode_inputs(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    target: str,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None]:
    """
    Extract base-condition classifier bins for repeated decode workflows.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table with mask columns, event times, and decode targets.
    target : str
        Decode target column name. Supported values are ``"state_int"`` and ``"action"``.

    Returns
    -------
    dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None]
        Mapping from ``(condition, window_name)`` to ``(spike_bins, target_values)``.
        Missing or empty selections map to ``None``.
    """

    get_decode_target(trial_df, target=target)
    trial_masks = make_trial_type_masks(trial_df)
    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }

    extracted_bins: dict[tuple[str, str], tuple[np.ndarray, np.ndarray] | None] = {}
    for condition in base_conditions:
        for window_name, bounds in windows.items():
            try:
                spike_bins, state_bins, choice_bins = make_classifier_bins(
                    region_trial_binned=region_trial_binned,
                    trial_df=trial_df,
                    trial_mask=trial_masks[condition],
                    event="choice_time",
                    bounds=bounds,
                )
            except ValueError:
                extracted_bins[(condition, window_name)] = None
                continue

            target_values = state_bins if target == "state_int" else choice_bins
            extracted_bins[(condition, window_name)] = (spike_bins, target_values)
    return extracted_bins

def train_single_decoder_with_shuffle_null(
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
    test_size: float = 0.2,
    n_shuffles: int = 1000,
    random_state: int = 42,
    label: str = "",
) -> tuple[LogisticRegression | None, dict[str, Any]]:
    """
    Train one classifier and compare its held-out score against shuffled-label null fits.

    Parameters
    ----------
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``. Values are class labels.
    test_size : float, optional
        Fraction of samples reserved for held-out evaluation.
    n_shuffles : int, optional
        Number of shuffled-label null fits on the training split.
    random_state : int, optional
        Random seed passed to the train-test split and primary classifier.
    label : str, optional
        Human-readable label describing the training run.

    Returns
    -------
    tuple[LogisticRegression | None, dict[str, Any]]
        Fitted classifier and a metrics dictionary, or ``(None, failed_result)`` if fitting is not possible.
    """

    filtered_spikes, filtered_targets, summary = _prepare_decode_inputs(binned_spikes, target_values)
    result_base = {"label": label, **summary}
    if summary["n_samples"] == 0:
        return None, _failed_decode_result("no_valid_samples", **result_base)
    if summary["n_classes"] < 2:
        return None, _failed_decode_result("insufficient_classes", **result_base)
    if summary["n_samples"] < 2:
        return None, _failed_decode_result("insufficient_samples", test_size=test_size, **result_base)

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            filtered_spikes.T,
            filtered_targets,
            test_size=test_size,
            random_state=random_state,
        )
    except ValueError:
        return None, _failed_decode_result("insufficient_samples", test_size=test_size, **result_base)

    if np.unique(y_train).size < 2:
        return None, _failed_decode_result("insufficient_classes_after_split", test_size=test_size, **result_base)

    classifier = LogisticRegression(
        solver="saga",
        # penalty="l1",
        l1_ratio=1,  # L1 penalty is l1_ratio=1; for L2 set to 0; for elastic net set to .5
        max_iter=10000,
        random_state=random_state,
    )
    classifier.fit(x_train, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(x_train))
    test_accuracy = accuracy_score(y_test, classifier.predict(x_test))

    shuffle_accuracies = []
    for shuffle_index in range(n_shuffles):
        shuffled_targets = np.array(y_train, copy=True)
        rng = np.random.default_rng(random_state + shuffle_index)
        rng.shuffle(shuffled_targets)
        if np.unique(shuffled_targets).size < 2:
            continue
        shuffle_classifier = LogisticRegression(max_iter=10000)
        shuffle_classifier.fit(x_train, shuffled_targets)
        shuffle_accuracies.append(accuracy_score(y_test, shuffle_classifier.predict(x_test)))

    shuffle_accuracy_array = np.asarray(shuffle_accuracies, dtype=float)
    shuffle_pvalue = float(np.mean(shuffle_accuracy_array >= test_accuracy)) if shuffle_accuracy_array.size else np.nan
    return classifier, {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "shuffle_pvalue": shuffle_pvalue,
        "shuffle_accuracy_mean": float(np.mean(shuffle_accuracy_array)) if shuffle_accuracy_array.size else np.nan,
        "shuffle_accuracy_std": float(np.std(shuffle_accuracy_array)) if shuffle_accuracy_array.size else np.nan,
        "n_shuffles": int(n_shuffles),
    }

def evaluate_decoder_on_condition(
    classifier: LogisticRegression | None,
    binned_spikes: np.ndarray,
    target_values: np.ndarray,
    label: str = "",
) -> dict[str, Any]:
    """
    Evaluate a fitted classifier on one condition-specific dataset.

    Parameters
    ----------
    classifier : LogisticRegression | None
        Fitted classifier from ``train_single_decoder_with_shuffle_null``.
    binned_spikes : np.ndarray
        Spike-count matrix with shape ``(n_units, n_samples)``.
    target_values : np.ndarray
        One-dimensional target array with shape ``(n_samples,)``. Values are class labels.
    label : str, optional
        Human-readable label for the evaluation set.

    Returns
    -------
    dict[str, Any]
        Evaluation result dictionary containing held-dataset accuracy or an explicit failure reason.
    """

    filtered_spikes, filtered_targets, summary = _prepare_decode_inputs(binned_spikes, target_values)
    result_base = {"label": label, **summary}
    if classifier is None:
        return _failed_decode_result("missing_classifier", **result_base)
    if summary["n_samples"] == 0:
        return _failed_decode_result("no_valid_samples", **result_base)
    if summary["n_classes"] < 2:
        return _failed_decode_result("insufficient_classes", **result_base)

    predictions = classifier.predict(filtered_spikes.T)
    return {
        "status": "ok",
        "reason": "",
        "label": label,
        "n_samples": summary["n_samples"],
        "n_classes": summary["n_classes"],
        "test_accuracy": float(accuracy_score(filtered_targets, predictions)),
    }

def run_base_condition_decoding(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    target: str = "state_int",
    cv: int = 5,
    n_permutations: int = 100,
    n_shuffles: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run the simple base-condition decoding workflow around choice time.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table containing mask columns, event times, and decode targets.
    target : str, optional
        Decode target column name. Supported values are ``"state_int"`` and ``"action"``.
    cv : int, optional
        Number of folds used for decodeability scoring.
    n_permutations : int, optional
        Number of permutations used for cross-validated decodeability scoring.
    n_shuffles : int, optional
        Number of shuffled-label null fits for the single trained decoder.
    random_state : int, optional
        Base random seed used for all stochastic decoding steps.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the fitted classifier, training metrics, decodeability results,
        and generalization results for base trial conditions and pre/post choice windows.
    """

    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = {
        "pre_choice": (-0.5, 0.0),
        "post_choice": (0.0, 0.5),
    }
    extracted_bin_inputs = _collect_base_condition_decode_inputs(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        target=target,
    )

    decodeability_rows: list[dict[str, Any]] = []
    for condition in base_conditions:
        for window_name in windows:
            condition_label = f"{condition}_{window_name}"
            extracted_bin_entry = extracted_bin_inputs[(condition, window_name)]
            if extracted_bin_entry is None:
                decodeability_rows.append(
                    {
                        "condition": condition,
                        "window": window_name,
                        "target": target,
                        **_failed_decode_result("no_selected_bins", label=condition_label),
                    }
                )
                continue

            spike_bins, condition_targets = extracted_bin_entry
            decodeability_result = cv_decodeability_score(
                binned_spikes=spike_bins,
                target_values=condition_targets,
                cv=cv,
                n_permutations=n_permutations,
                random_state=random_state,
                label=condition_label,
            )
            decodeability_rows.append(
                {
                    "condition": condition,
                    "window": window_name,
                    "target": target,
                    **decodeability_result,
                }
            )

    train_key = ("correct_rewarded", "post_choice")
    train_bins = extracted_bin_inputs.get(train_key)
    if train_bins is None:
        classifier = None
        training_result = _failed_decode_result(
            "missing_training_bins",
            label="correct_rewarded_post_choice",
            condition="correct_rewarded",
            window="post_choice",
            target=target,
        )
    else:
        classifier, training_metrics = train_single_decoder_with_shuffle_null(
            binned_spikes=train_bins[0],
            target_values=train_bins[1],
            n_shuffles=n_shuffles,
            random_state=random_state,
            label="correct_rewarded_post_choice",
        )
        training_result = {
            "condition": "correct_rewarded",
            "window": "post_choice",
            "target": target,
            **training_metrics,
        }

    generalization_rows: list[dict[str, Any]] = []
    for condition in base_conditions:
        for window_name in windows:
            if condition == "correct_rewarded" and window_name == "post_choice":
                generalization_rows.append(
                    {
                        "condition": condition,
                        "window": window_name,
                        "target": target,
                        **{key: value for key, value in training_result.items() if key not in {"shuffle_accuracy_mean", "shuffle_accuracy_std", "shuffle_pvalue", "n_shuffles"}},
                        "test_accuracy": training_result.get("test_accuracy", np.nan),
                    }
                )
                continue

            condition_bins = extracted_bin_inputs.get((condition, window_name))
            if condition_bins is None:
                evaluation_result = _failed_decode_result(
                    "no_selected_bins",
                    label=f"{condition}_{window_name}",
                )
            else:
                evaluation_result = evaluate_decoder_on_condition(
                    classifier=classifier,
                    binned_spikes=condition_bins[0],
                    target_values=condition_bins[1],
                    label=f"{condition}_{window_name}",
                )
            generalization_rows.append(
                {
                    "condition": condition,
                    "window": window_name,
                    "target": target,
                    **evaluation_result,
                }
            )

    return {
        "classifier": classifier,
        "training_result": training_result,
        "decodeability_results": pd.DataFrame(decodeability_rows),
        "generalization_results": pd.DataFrame(generalization_rows),
    }

def run_repeated_correct_rewarded_decoder(
    region_trial_binned: list[dict[str, Any]],
    trial_df: pd.DataFrame,
    target: str = "state_int",
    n_decoder_runs: int = 5,
    n_shuffles: int = 1000,
    random_state: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Train multiple correct-rewarded post-choice decoders and evaluate each across base conditions.

    Parameters
    ----------
    region_trial_binned : list[dict[str, Any]]
        Output from ``bin_region_trials`` with one dictionary per trial.
    trial_df : pd.DataFrame
        Trial table with mask columns, event times, and decode targets.
    target : str, optional
        Decode target column name. Supported values are ``"state_int"`` and ``"action"``.
    n_decoder_runs : int, optional
        Number of independently seeded decoder fits to run.
    n_shuffles : int, optional
        Number of shuffled-label null fits per decoder run.
    random_state : int, optional
        Base random seed. Each decoder run uses ``random_state + decoder_run``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with ``training_results`` and ``generalization_results`` tables.
    """

    base_conditions = ["correct_rewarded", "incorrect", "omission", "switch", "stay"]
    windows = ["pre_choice", "post_choice"]
    extracted_bin_inputs = _collect_base_condition_decode_inputs(
        region_trial_binned=region_trial_binned,
        trial_df=trial_df,
        target=target,
    )

    training_rows: list[dict[str, Any]] = []
    generalization_rows: list[dict[str, Any]] = []
    for decoder_run in range(int(n_decoder_runs)):
        run_seed = int(random_state) + decoder_run
        train_key = ("correct_rewarded", "post_choice")
        train_bins = extracted_bin_inputs.get(train_key)
        if train_bins is None:
            classifier = None
            training_result = _failed_decode_result(
                "missing_training_bins",
                label="correct_rewarded_post_choice",
                training_condition="correct_rewarded",
                training_window="post_choice",
            )
        else:
            classifier, training_metrics = train_single_decoder_with_shuffle_null(
                binned_spikes=train_bins[0],
                target_values=train_bins[1],
                n_shuffles=n_shuffles,
                random_state=run_seed,
                label="correct_rewarded_post_choice",
            )
            training_result = {
                "training_condition": "correct_rewarded",
                "training_window": "post_choice",
                **training_metrics,
            }

        training_rows.append(
            {
                "decoder_run": decoder_run,
                "target": target,
                **training_result,
            }
        )

        for condition in base_conditions:
            for window_name in windows:
                condition_bins = extracted_bin_inputs.get((condition, window_name))
                if condition == "correct_rewarded" and window_name == "post_choice":
                    evaluation_result = {
                        "status": training_result["status"],
                        "reason": training_result["reason"],
                        "label": f"{condition}_{window_name}",
                        "n_samples": training_result.get("n_samples", 0),
                        "n_classes": training_result.get("n_classes", 0),
                        "test_accuracy": training_result.get("test_accuracy", np.nan),
                    }
                elif condition_bins is None:
                    evaluation_result = _failed_decode_result(
                        "no_selected_bins",
                        label=f"{condition}_{window_name}",
                    )
                else:
                    evaluation_result = evaluate_decoder_on_condition(
                        classifier=classifier,
                        binned_spikes=condition_bins[0],
                        target_values=condition_bins[1],
                        label=f"{condition}_{window_name}",
                    )

                generalization_rows.append(
                    {
                        "decoder_run": decoder_run,
                        "target": target,
                        "condition": condition,
                        "window": window_name,
                        **evaluation_result,
                    }
                )

    return {
        "training_results": pd.DataFrame(training_rows),
        "generalization_results": pd.DataFrame(generalization_rows),
    }
