"""Established per-session decoding table and CSV publication contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.neural_analysis.spike_behavior.loading import SUPPORTED_ANALYSIS_REGIONS, Session

def normalize_region_name_for_filename(region_name: str) -> str:
    """
    Normalize a brain-region label for analysis filenames.

    Parameters
    ----------
    region_name : str
        Brain-region label. Supported values are ``"HPC"``, ``"V1"``, and
        ``"PFC"``, case-insensitive.

    Returns
    -------
    str
        Canonical region label for filenames.
    """

    normalized_region = str(region_name).strip().upper()
    if normalized_region not in SUPPORTED_ANALYSIS_REGIONS:
        raise ValueError(
            f"Unsupported region {region_name!r}. Expected one of {sorted(SUPPORTED_ANALYSIS_REGIONS)}."
        )
    return normalized_region

def make_region_analysis_filename(region_name: str, base_filename: str) -> str:
    """
    Prefix an analysis filename with a canonical brain-region label.

    Parameters
    ----------
    region_name : str
        Brain-region label used to namespace per-session outputs.
    base_filename : str
        Base analysis filename including extension.

    Returns
    -------
    str
        Region-prefixed filename, e.g. ``"HPC_state_decodability_analysis.csv"``.
    """

    return f"{normalize_region_name_for_filename(region_name)}_{base_filename}"

def build_state_decodability_session_table(
    session: Session,
    region_name: str,
    decodeability_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape state decodeability results into one row per condition with paired pre/post columns.

    Parameters
    ----------
    session : Session
        Session metadata providing the session identifier, mouse, and date strings.
    region_name : str
        Human-readable region label for the decoded units.
    decodeability_results : pd.DataFrame
        Long-form decodeability results from ``run_base_condition_decoding``. Required columns are
        ``condition``, ``window``, ``status``, ``reason``, ``n_samples``, ``n_classes``,
        ``cv_score``, ``cv_pvalue``, ``permutation_score_mean``, and ``permutation_score_std``.

    Returns
    -------
    pd.DataFrame
        Wide per-session state decodeability table with one row per condition and paired
        ``*_pre`` / ``*_post`` columns for the before- and after-choice bins.
    """

    required_columns = {
        "condition",
        "window",
        "status",
        "reason",
        "n_samples",
        "n_classes",
        "cv_score",
        "cv_pvalue",
        "permutation_score_mean",
        "permutation_score_std",
    }
    missing_columns = required_columns - set(decodeability_results.columns)
    if missing_columns:
        raise ValueError(f"decodeability_results is missing required columns: {sorted(missing_columns)}")

    row_order = list(dict.fromkeys(decodeability_results["condition"].tolist()))
    session_rows: list[dict[str, Any]] = []
    for condition_name in row_order:
        condition_rows = decodeability_results.loc[decodeability_results["condition"] == condition_name]
        pre_row = condition_rows.loc[condition_rows["window"] == "pre_choice"]
        post_row = condition_rows.loc[condition_rows["window"] == "post_choice"]
        if pre_row.shape[0] != 1 or post_row.shape[0] != 1:
            raise ValueError(f"Condition {condition_name!r} must have exactly one pre_choice and one post_choice row.")

        pre_result = pre_row.iloc[0]
        post_result = post_row.iloc[0]
        session_rows.append(
            {
                "session_id": session.sess_id_full,
                "mouse": session.mouse,
                "date": session.date,
                "region": region_name,
                "condition": condition_name,
                "status_pre": pre_result["status"],
                "reason_pre": pre_result["reason"],
                "n_trials_pre": pre_result["n_samples"],
                "n_classes_pre": pre_result["n_classes"],
                "cv_score_pre": pre_result["cv_score"],
                "p_value_pre": pre_result["cv_pvalue"],
                "score_mean_pre": pre_result["permutation_score_mean"],
                "score_std_pre": pre_result["permutation_score_std"],
                "status_post": post_result["status"],
                "reason_post": post_result["reason"],
                "n_trials_post": post_result["n_samples"],
                "n_classes_post": post_result["n_classes"],
                "cv_score_post": post_result["cv_score"],
                "p_value_post": post_result["cv_pvalue"],
                "score_mean_post": post_result["permutation_score_mean"],
                "score_std_post": post_result["permutation_score_std"],
            }
        )

    return pd.DataFrame(session_rows)

def save_state_decodability_session_csv(
    output_dir: Path | str,
    state_decodability_table: pd.DataFrame,
    region_name: str,
) -> Path:
    """
    Save the per-session state decodeability table to a region-specific CSV.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the CSV should be written.
    state_decodability_table : pd.DataFrame
        Wide per-session state decodeability table with one row per condition.
    region_name : str
        Brain-region label used to namespace the saved CSV filename.

    Returns
    -------
    Path
        Saved CSV path.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / make_region_analysis_filename(region_name, "state_decodability_analysis.csv")
    state_decodability_table.to_csv(csv_path, index=False)
    return csv_path

def build_correct_rewarded_decoding_performance_session_table(
    session: Session,
    region_name: str,
    training_results: pd.DataFrame,
    generalization_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reshape repeated decoder outputs into one row per decoder run with all condition results.

    Parameters
    ----------
    session : Session
        Session metadata providing the session identifier, mouse, and date strings.
    region_name : str
        Human-readable region label for the decoded units.
    training_results : pd.DataFrame
        One row per decoder run with training metrics.
    generalization_results : pd.DataFrame
        One row per ``decoder_run x condition x window`` with evaluation metrics.

    Returns
    -------
    pd.DataFrame
        Wide per-session decoder-performance table with one row per decoder run and paired
        before/after columns for each base condition.
    """

    required_training_columns = {
        "decoder_run",
        "training_condition",
        "training_window",
        "train_accuracy",
        "test_accuracy",
        "shuffle_pvalue",
        "shuffle_accuracy_mean",
        "shuffle_accuracy_std",
    }
    missing_training = required_training_columns - set(training_results.columns)
    if missing_training:
        raise ValueError(f"training_results is missing required columns: {sorted(missing_training)}")

    required_generalization_columns = {
        "decoder_run",
        "condition",
        "window",
        "status",
        "reason",
        "test_accuracy",
        "n_samples",
        "n_classes",
    }
    missing_generalization = required_generalization_columns - set(generalization_results.columns)
    if missing_generalization:
        raise ValueError(
            f"generalization_results is missing required columns: {sorted(missing_generalization)}"
        )

    condition_order = list(dict.fromkeys(generalization_results["condition"].tolist()))
    decoder_rows: list[dict[str, Any]] = []
    for _, training_row in training_results.sort_values("decoder_run").iterrows():
        decoder_run = int(training_row["decoder_run"])
        row = {
            "session_id": session.sess_id_full,
            "mouse": session.mouse,
            "date": session.date,
            "region": region_name,
            "decoder_run": decoder_run,
            "training_condition": training_row["training_condition"],
            "training_window": training_row["training_window"],
            "train_accuracy": training_row.get("train_accuracy", np.nan),
            "heldout_test_accuracy": training_row.get("test_accuracy", np.nan),
            "shuffle_pvalue": training_row.get("shuffle_pvalue", np.nan),
            "shuffle_accuracy_mean": training_row.get("shuffle_accuracy_mean", np.nan),
            "shuffle_accuracy_std": training_row.get("shuffle_accuracy_std", np.nan),
        }
        decoder_generalization = generalization_results.loc[
            generalization_results["decoder_run"] == decoder_run
        ]
        for condition_name in condition_order:
            for window_name, suffix in (("pre_choice", "pre"), ("post_choice", "post")):
                matching_rows = decoder_generalization.loc[
                    (decoder_generalization["condition"] == condition_name)
                    & (decoder_generalization["window"] == window_name)
                ]
                if matching_rows.shape[0] != 1:
                    raise ValueError(
                        f"Decoder run {decoder_run} condition {condition_name!r} must have exactly one {window_name} row."
                    )
                result_row = matching_rows.iloc[0]
                row[f"{condition_name}_status_{suffix}"] = result_row["status"]
                row[f"{condition_name}_reason_{suffix}"] = result_row["reason"]
                row[f"{condition_name}_test_accuracy_{suffix}"] = result_row["test_accuracy"]
                row[f"{condition_name}_n_trials_{suffix}"] = result_row["n_samples"]
                row[f"{condition_name}_n_classes_{suffix}"] = result_row["n_classes"]
        decoder_rows.append(row)

    return pd.DataFrame(decoder_rows)

def save_correct_rewarded_decoding_performance_session_csv(
    output_dir: Path | str,
    decoding_performance_table: pd.DataFrame,
    region_name: str,
) -> Path:
    """
    Save the repeated decoder-performance table to a region-specific CSV.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the CSV should be written.
    decoding_performance_table : pd.DataFrame
        Wide per-session decoder-performance table with one row per decoder run.
    region_name : str
        Brain-region label used to namespace the saved CSV filename.

    Returns
    -------
    Path
        Saved CSV path.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / make_region_analysis_filename(region_name, "correct_rewarded_decoding_performance.csv")
    decoding_performance_table.to_csv(csv_path, index=False)
    return csv_path
