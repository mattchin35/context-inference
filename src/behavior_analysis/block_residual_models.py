"""Regularized block-level residual models for trials-to-correct analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

MODEL_SPECS: tuple[tuple[str, float], ...] = (
    ("lasso", 1.0),
    ("elastic_net", 0.5),
)
LAMBDA_CHOICES: tuple[str, ...] = ("lambda.min", "lambda.1se")
INTERACTION_FORMULA = "rewards_x_side"
ADDITIVE_FORMULA = "rewards_plus_side"
INTERACTION_FEATURE_NAMES: tuple[str, ...] = (
    "prev_n_rewarded",
    "block_side_left",
    "prev_n_rewarded:block_side_left",
)
ADDITIVE_FEATURE_NAMES: tuple[str, ...] = (
    "prev_n_rewarded",
    "block_side_left",
)
COEFFICIENT_COLUMNS: tuple[str, ...] = (
    "coefficient_intercept",
    "coefficient_prev_n_rewarded",
    "coefficient_block_side_left",
    "coefficient_prev_n_rewarded_block_side_left",
)
SIDE_SPECIFIC_COEFFICIENT_COLUMNS: tuple[str, ...] = (
    "right_intercept",
    "left_intercept",
    "right_reward_slope",
    "left_reward_slope",
    "side_intercept_delta_left_minus_right",
    "side_reward_slope_delta_left_minus_right",
)
REQUIRED_COLUMNS: tuple[str, ...] = ("trials_to_correct", "prev_n_rewarded", "block_type")
MISSING_VALUE = "None"
SCALED_MAD_CONSTANT = 1.4826


@dataclass(frozen=True)
class BlockResidualFormulaSpec:
    """Design-matrix contract for one block residual model formula.

    Attributes
    ----------
    name : str
        Stable formula identifier used in summary rows and output filenames.
    feature_names : tuple[str, ...]
        Predictor names in the design matrix column order. Counts are in
        rewarded trials; side indicator is unitless.
    include_interaction : bool
        If True, include the reward-by-left-side interaction term.
    write_legacy_aliases : bool
        If True, also write the original formula-unspecified residual and
        session metric columns for backward compatibility.
    """

    name: str
    feature_names: tuple[str, ...]
    include_interaction: bool
    write_legacy_aliases: bool = False


FORMULA_SPECS: tuple[BlockResidualFormulaSpec, ...] = (
    BlockResidualFormulaSpec(
        name=INTERACTION_FORMULA,
        feature_names=INTERACTION_FEATURE_NAMES,
        include_interaction=True,
        write_legacy_aliases=True,
    ),
    BlockResidualFormulaSpec(
        name=ADDITIVE_FORMULA,
        feature_names=ADDITIVE_FEATURE_NAMES,
        include_interaction=False,
        write_legacy_aliases=False,
    ),
)


@dataclass(frozen=True)
class PreparedBlockResidualData:
    """Numeric design matrix for the block residual model.

    Attributes
    ----------
    x : numpy.ndarray
        Predictor matrix with shape `(n_valid_blocks, n_features)`. Columns are
        defined by `feature_names`. Counts are in rewarded trials; side
        indicator is unitless.
    y : numpy.ndarray
        Trials-to-correct values with shape `(n_valid_blocks,)`, in trials.
    valid_index : pandas.Index
        Original block dataframe index labels for rows included in `x` and `y`.
    feature_names : list[str]
        Names for the columns of `x`, preserving the matrix order.
    """

    x: np.ndarray
    y: np.ndarray
    valid_index: pd.Index
    feature_names: list[str]


def derive_block_side(block_type_values: Iterable[object]) -> pd.Series:
    """Map block type labels to left/right side labels.

    Parameters
    ----------
    block_type_values : iterable
        Block labels with shape `(n_blocks,)`. Values beginning with `"right"`
        are mapped to `"right"`; values beginning with `"left"` are mapped to
        `"left"`.

    Returns
    -------
    pandas.Series
        Side labels with shape `(n_blocks,)`. Unknown or missing labels are NA.
        Right is the reference side for downstream design-matrix construction.
    """
    block_types = pd.Series(block_type_values, dtype="object")
    normalized = block_types.astype(str).str.lower().str.strip()
    block_side = pd.Series(pd.NA, index=block_types.index, dtype="object")
    block_side.loc[normalized.str.startswith("right", na=False)] = "right"
    block_side.loc[normalized.str.startswith("left", na=False)] = "left"
    return block_side


def _formula_spec_by_name(model_formula: str) -> BlockResidualFormulaSpec:
    """Return the configured formula spec for a stable formula identifier."""
    for formula_spec in FORMULA_SPECS:
        if formula_spec.name == model_formula:
            return formula_spec
    raise ValueError(f"Unknown block residual model formula: {model_formula}")


def prepare_block_residual_model_data(
    block_performance: pd.DataFrame,
    model_formula: str = INTERACTION_FORMULA,
) -> PreparedBlockResidualData:
    """Build a right-reference design matrix for one session.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Block table with shape `(n_blocks, n_columns)`. Required columns are
        `trials_to_correct` in trials, `prev_n_rewarded` in rewarded trials,
        and `block_type` labels that begin with `left` or `right`.
    model_formula : str, default="rewards_x_side"
        Formula identifier. Supported values are `"rewards_x_side"` for
        `trials_to_correct ~ prev_n_rewarded * block_side` and
        `"rewards_plus_side"` for
        `trials_to_correct ~ prev_n_rewarded + block_side`.

    Returns
    -------
    PreparedBlockResidualData
        Valid rows and numeric design matrix for
        the requested formula, with right as the reference side.
    """
    formula_spec = _formula_spec_by_name(model_formula)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in block_performance.columns]
    if missing_columns:
        return PreparedBlockResidualData(
            x=np.empty((0, len(formula_spec.feature_names)), dtype=float),
            y=np.empty(0, dtype=float),
            valid_index=pd.Index([], dtype=block_performance.index.dtype),
            feature_names=list(formula_spec.feature_names),
        )

    trials_to_correct = pd.to_numeric(block_performance["trials_to_correct"], errors="coerce")
    previous_rewards = pd.to_numeric(block_performance["prev_n_rewarded"], errors="coerce")
    block_side = derive_block_side(block_performance["block_type"])
    valid_rows = trials_to_correct.notna() & previous_rewards.notna() & block_side.notna()

    valid_index = block_performance.index[valid_rows]
    previous_rewards_valid = previous_rewards.loc[valid_rows].to_numpy(dtype=float)
    side_left = (block_side.loc[valid_rows].to_numpy(dtype=object) == "left").astype(float)
    x_columns = [previous_rewards_valid, side_left]
    if formula_spec.include_interaction:
        x_columns.append(previous_rewards_valid * side_left)
    x = np.column_stack(x_columns)
    y = trials_to_correct.loc[valid_rows].to_numpy(dtype=float)
    return PreparedBlockResidualData(
        x=x,
        y=y,
        valid_index=valid_index,
        feature_names=list(formula_spec.feature_names),
    )


def compute_residual_spread_metrics(residuals: np.ndarray) -> dict[str, float | str]:
    """Compute spread summaries for model residuals.

    Parameters
    ----------
    residuals : numpy.ndarray
        Residual vector with shape `(n_valid_blocks,)`, in trials, defined as
        observed trials-to-correct minus fitted trials-to-correct.

    Returns
    -------
    dict[str, float | str]
        Raw MAD, scaled MAD using R's default 1.4826 multiplier, IQR, RMSE,
        and sample SD. Empty or all-missing vectors return the project missing
        sentinel.
    """
    residual_values = np.asarray(residuals, dtype=float)
    residual_values = residual_values[np.isfinite(residual_values)]
    if residual_values.size == 0:
        return _missing_residual_metrics()

    median_residual = float(np.median(residual_values))
    mad_raw = float(np.median(np.abs(residual_values - median_residual)))
    q1, q3 = np.percentile(residual_values, [25, 75])
    residual_sd: float | str = MISSING_VALUE
    if residual_values.size >= 2:
        residual_sd = float(np.std(residual_values, ddof=1))
    return {
        "residual_mad_raw": mad_raw,
        "residual_mad_scaled": float(mad_raw * SCALED_MAD_CONSTANT),
        "residual_iqr": float(q3 - q1),
        "residual_rmse": float(np.sqrt(np.mean(residual_values**2))),
        "residual_sd": residual_sd,
    }


def _missing_residual_metrics() -> dict[str, str]:
    """Return missing residual spread metrics."""
    return {
        "residual_mad_raw": MISSING_VALUE,
        "residual_mad_scaled": MISSING_VALUE,
        "residual_iqr": MISSING_VALUE,
        "residual_rmse": MISSING_VALUE,
        "residual_sd": MISSING_VALUE,
    }


def _summary_prefix(model_type: str, lambda_choice: str) -> str:
    """Return the stable output prefix for one model/lambda pair."""
    lambda_tag = "lambda_min" if lambda_choice == "lambda.min" else "lambda_1se"
    return f"{model_type}_{lambda_tag}"


def _formula_summary_prefix(model_formula: str, model_type: str, lambda_choice: str) -> str:
    """Return the stable output prefix for one formula/model/lambda pair."""
    lambda_tag = "lambda_min" if lambda_choice == "lambda.min" else "lambda_1se"
    return f"{model_type}_{model_formula}_{lambda_tag}"


def _residual_column(model_type: str, lambda_choice: str) -> str:
    """Return the block-level residual column for one model/lambda pair."""
    return f"{_summary_prefix(model_type, lambda_choice)}_residual_TTS"


def _formula_residual_column(model_formula: str, model_type: str, lambda_choice: str) -> str:
    """Return the formula-specific block residual column."""
    return f"{_formula_summary_prefix(model_formula, model_type, lambda_choice)}_residual_TTS"


def _metric_column(model_type: str, lambda_choice: str, metric_name: str) -> str:
    """Return the session-level metric column for one model/lambda pair."""
    return f"{_summary_prefix(model_type, lambda_choice)}_{metric_name}"


def _formula_metric_column(
    model_formula: str,
    model_type: str,
    lambda_choice: str,
    metric_name: str,
) -> str:
    """Return the formula-specific session-level metric column."""
    return f"{_formula_summary_prefix(model_formula, model_type, lambda_choice)}_{metric_name}"


def _empty_summary_row(
    model_formula: str,
    model_type: str,
    alpha: float,
    lambda_choice: str,
    n_valid_blocks: int,
    session_id: str | None,
    mouse: str | None,
    date: str | None,
) -> dict[str, object]:
    """Build one missing summary row for an unfitted model."""
    row: dict[str, object] = {
        "session_id": session_id or MISSING_VALUE,
        "mouse": mouse or MISSING_VALUE,
        "date": date or MISSING_VALUE,
        "model_formula": model_formula,
        "model_type": model_type,
        "lambda_choice": lambda_choice,
        "lambda_value": MISSING_VALUE,
        "alpha": alpha,
        "n_valid_blocks": int(n_valid_blocks),
    }
    row.update({column: MISSING_VALUE for column in COEFFICIENT_COLUMNS})
    row.update({column: MISSING_VALUE for column in SIDE_SPECIFIC_COEFFICIENT_COLUMNS})
    row.update(_missing_residual_metrics())
    return row


def _select_lambda_1se(alphas: np.ndarray, mse_path: np.ndarray) -> float:
    """Select the strongest alpha within one SE of the minimum CV error."""
    mean_mse = np.mean(mse_path, axis=1)
    se_mse = np.std(mse_path, axis=1, ddof=1) / np.sqrt(mse_path.shape[1])
    best_index = int(np.argmin(mean_mse))
    threshold = mean_mse[best_index] + se_mse[best_index]
    eligible = np.flatnonzero(mean_mse <= threshold)
    if eligible.size == 0:
        return float(alphas[best_index])
    strongest_index = eligible[np.argmax(alphas[eligible])]
    return float(alphas[strongest_index])


def _fit_selected_elastic_net(
    x_scaled: np.ndarray,
    y: np.ndarray,
    selected_alpha: float,
    l1_ratio: float,
    seed: int,
) -> ElasticNet:
    """Fit one selected regularized linear model on standardized predictors."""
    model = ElasticNet(
        alpha=selected_alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=100000,
        random_state=seed,
        selection="cyclic",
    )
    model.fit(x_scaled, y)
    return model


def _side_specific_coefficients(coefficient_values: dict[str, float]) -> dict[str, float]:
    """Derive side-specific intercepts and reward slopes from right-reference terms."""
    intercept = coefficient_values["coefficient_intercept"]
    reward_slope = coefficient_values["coefficient_prev_n_rewarded"]
    left_offset = coefficient_values["coefficient_block_side_left"]
    left_reward_delta = coefficient_values["coefficient_prev_n_rewarded_block_side_left"]
    return {
        "right_intercept": float(intercept),
        "left_intercept": float(intercept + left_offset),
        "right_reward_slope": float(reward_slope),
        "left_reward_slope": float(reward_slope + left_reward_delta),
        "side_intercept_delta_left_minus_right": float(left_offset),
        "side_reward_slope_delta_left_minus_right": float(left_reward_delta),
    }


def _original_scale_coefficients(
    model: ElasticNet,
    scaler: StandardScaler,
    feature_names: list[str],
) -> dict[str, float]:
    """Convert coefficients from standardized predictors to original units."""
    standardized_coef = np.asarray(model.coef_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    mean = np.asarray(scaler.mean_, dtype=float)
    original_coef = standardized_coef / scale
    original_intercept = float(model.intercept_ - np.sum(standardized_coef * mean / scale))
    coefficient_by_feature = {
        feature_name: float(coefficient)
        for feature_name, coefficient in zip(feature_names, original_coef, strict=True)
    }
    coefficient_values = {
        "coefficient_intercept": original_intercept,
        "coefficient_prev_n_rewarded": coefficient_by_feature.get("prev_n_rewarded", 0.0),
        "coefficient_block_side_left": coefficient_by_feature.get("block_side_left", 0.0),
        "coefficient_prev_n_rewarded_block_side_left": coefficient_by_feature.get(
            "prev_n_rewarded:block_side_left",
            0.0,
        ),
    }
    coefficient_values.update(_side_specific_coefficients(coefficient_values))
    return coefficient_values


def _fit_model_family(
    prepared: PreparedBlockResidualData,
    model_formula: str,
    model_type: str,
    alpha: float,
    seed: int,
    nfolds: int,
    session_id: str | None,
    mouse: str | None,
    date: str | None,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], dict[str, dict[str, object]]]:
    """Fit one regularization family and return residuals plus summaries."""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(prepared.x)
    cv = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    cv_model = ElasticNetCV(
        l1_ratio=alpha,
        fit_intercept=True,
        cv=cv,
        max_iter=100000,
        random_state=seed,
    )
    cv_model.fit(x_scaled, prepared.y)
    selected_alphas = {
        "lambda.min": float(cv_model.alpha_),
        "lambda.1se": _select_lambda_1se(
            np.asarray(cv_model.alphas_, dtype=float),
            np.asarray(cv_model.mse_path_, dtype=float),
        ),
    }

    residuals_by_lambda: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []
    metric_values: dict[str, dict[str, object]] = {}
    for lambda_choice in LAMBDA_CHOICES:
        selected_alpha = selected_alphas[lambda_choice]
        model = _fit_selected_elastic_net(
            x_scaled=x_scaled,
            y=prepared.y,
            selected_alpha=selected_alpha,
            l1_ratio=alpha,
            seed=seed,
        )
        predictions = model.predict(x_scaled)
        residuals = prepared.y - predictions
        residual_metrics = compute_residual_spread_metrics(residuals)
        row: dict[str, object] = {
            "session_id": session_id or MISSING_VALUE,
            "mouse": mouse or MISSING_VALUE,
            "date": date or MISSING_VALUE,
            "model_formula": model_formula,
            "model_type": model_type,
            "lambda_choice": lambda_choice,
            "lambda_value": selected_alpha,
            "alpha": alpha,
            "n_valid_blocks": int(prepared.y.shape[0]),
        }
        coefficient_values = _original_scale_coefficients(
            model,
            scaler,
            feature_names=prepared.feature_names,
        )
        row.update(coefficient_values)
        row.update(residual_metrics)
        summary_rows.append(row)
        residuals_by_lambda[lambda_choice] = residuals
        metric_values[lambda_choice] = residual_metrics
    return residuals_by_lambda, summary_rows, metric_values


def _empty_summary_dataframe(
    n_valid_blocks: int,
    session_id: str | None,
    mouse: str | None,
    date: str | None,
    formula_specs: tuple[BlockResidualFormulaSpec, ...] = FORMULA_SPECS,
) -> pd.DataFrame:
    """Return missing rows for skipped residual-model fitting."""
    rows = [
        _empty_summary_row(
            model_formula=formula_spec.name,
            model_type=model_type,
            alpha=alpha,
            lambda_choice=lambda_choice,
            n_valid_blocks=n_valid_blocks,
            session_id=session_id,
            mouse=mouse,
            date=date,
        )
        for formula_spec in formula_specs
        for model_type, alpha in MODEL_SPECS
        for lambda_choice in LAMBDA_CHOICES
    ]
    return pd.DataFrame(rows)


def add_block_residual_model_outputs(
    block_performance: pd.DataFrame,
    session_performance: pd.DataFrame,
    seed: int = 12345,
    min_valid_blocks: int = 5,
    session_id: str | None = None,
    mouse: str | None = None,
    date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add regularized residual-model outputs to block and session tables.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required fitting
        columns are `trials_to_correct`, `prev_n_rewarded`, and `block_type`.
    session_performance : pandas.DataFrame
        One-row session summary table with shape `(1, n_columns)`.
    seed : int, default=12345
        Random seed for shuffled cross-validation folds.
    min_valid_blocks : int, default=5
        Minimum valid block rows required for fitting. Smaller sessions receive
        missing residuals and missing coefficient/summary metrics.
    session_id, mouse, date : str or None
        Optional identifiers copied into the residual-model summary table.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Updated block table, updated one-row session table, and one residual
        model summary table with one row per model/lambda pair.
    """
    updated_blocks = block_performance.copy()
    updated_session = session_performance.copy()
    if updated_session.empty:
        updated_session = pd.DataFrame(index=[0])

    for formula_spec in FORMULA_SPECS:
        for model_type, _alpha in MODEL_SPECS:
            for lambda_choice in LAMBDA_CHOICES:
                initialized_residual = pd.Series(
                    np.full(updated_blocks.shape[0], MISSING_VALUE, dtype=object),
                    index=updated_blocks.index,
                    dtype=object,
                )
                updated_blocks[
                    _formula_residual_column(formula_spec.name, model_type, lambda_choice)
                ] = initialized_residual.copy()
                if formula_spec.write_legacy_aliases:
                    updated_blocks[_residual_column(model_type, lambda_choice)] = initialized_residual.copy()

                missing_metrics = _missing_residual_metrics()
                for metric_name, metric_value in missing_metrics.items():
                    updated_session[
                        _formula_metric_column(
                            formula_spec.name,
                            model_type,
                            lambda_choice,
                            metric_name,
                        )
                    ] = metric_value
                    if formula_spec.write_legacy_aliases:
                        updated_session[
                            _metric_column(model_type, lambda_choice, metric_name)
                        ] = metric_value

    summary_rows: list[dict[str, object]] = []
    for formula_spec in FORMULA_SPECS:
        prepared = prepare_block_residual_model_data(updated_blocks, model_formula=formula_spec.name)
        n_valid_blocks = int(prepared.y.shape[0])
        if n_valid_blocks < min_valid_blocks:
            summary_rows.extend(
                _empty_summary_dataframe(
                    n_valid_blocks=n_valid_blocks,
                    session_id=session_id,
                    mouse=mouse,
                    date=date,
                    formula_specs=(formula_spec,),
                ).to_dict(orient="records")
            )
            continue

        nfolds = min(10, n_valid_blocks)
        for model_type, alpha in MODEL_SPECS:
            residuals_by_lambda, model_summary_rows, metric_values = _fit_model_family(
                prepared=prepared,
                model_formula=formula_spec.name,
                model_type=model_type,
                alpha=alpha,
                seed=seed,
                nfolds=nfolds,
                session_id=session_id,
                mouse=mouse,
                date=date,
            )
            summary_rows.extend(model_summary_rows)
            for lambda_choice in LAMBDA_CHOICES:
                residual_values = residuals_by_lambda[lambda_choice].astype(float)
                formula_residual_column = _formula_residual_column(
                    formula_spec.name,
                    model_type,
                    lambda_choice,
                )
                updated_blocks.loc[prepared.valid_index, formula_residual_column] = residual_values
                if formula_spec.write_legacy_aliases:
                    updated_blocks.loc[
                        prepared.valid_index,
                        _residual_column(model_type, lambda_choice),
                    ] = residual_values
                for metric_name, metric_value in metric_values[lambda_choice].items():
                    updated_session[
                        _formula_metric_column(
                            formula_spec.name,
                            model_type,
                            lambda_choice,
                            metric_name,
                        )
                    ] = metric_value
                    if formula_spec.write_legacy_aliases:
                        updated_session[
                            _metric_column(model_type, lambda_choice, metric_name)
                        ] = metric_value

    return updated_blocks, updated_session, pd.DataFrame(summary_rows)


def save_block_residual_model_summary(
    summary_df: pd.DataFrame,
    session_save_path: Path,
    sess_id: str,
) -> Path:
    """Save one residual-model summary CSV for a session.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary table with shape `(4, n_columns)`, one row per model/lambda
        pair. Coefficients are in original predictor units and residual metrics
        are in trials.
    session_save_path : pathlib.Path
        Directory where the session's processed outputs are saved.
    sess_id : str
        Full session identifier used in the output file name.

    Returns
    -------
    pathlib.Path
        Path to `{sess_id}_block_residual_model_summary.csv`.
    """
    summary_path = session_save_path / f"{sess_id}_block_residual_model_summary.csv"
    summary_df.to_csv(summary_path, index=False, na_rep=MISSING_VALUE)
    return summary_path


def save_block_residual_model_summaries(
    summary_df: pd.DataFrame,
    session_save_path: Path,
    sess_id: str,
) -> dict[str, Path]:
    """Save combined and formula-specific residual-model summary CSVs.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary table with shape `(n_model_rows, n_columns)`. Must include
        `model_formula` when formula-specific CSVs should be written.
    session_save_path : pathlib.Path
        Directory where the session's processed outputs are saved.
    sess_id : str
        Full session identifier used in output file names.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from `"combined"` and each formula identifier to the saved CSV
        path. The combined path preserves the original filename convention.
    """
    saved_paths = {
        "combined": save_block_residual_model_summary(
            summary_df,
            session_save_path=session_save_path,
            sess_id=sess_id,
        )
    }
    if "model_formula" not in summary_df.columns:
        return saved_paths

    for formula_spec in FORMULA_SPECS:
        formula_rows = summary_df[summary_df["model_formula"] == formula_spec.name]
        formula_path = (
            session_save_path
            / f"{sess_id}_block_residual_model_summary_{formula_spec.name}.csv"
        )
        formula_rows.to_csv(formula_path, index=False, na_rep=MISSING_VALUE)
        saved_paths[formula_spec.name] = formula_path
    return saved_paths
