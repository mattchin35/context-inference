import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy.stats import pearsonr
from typing import Any


def residualize_values(
    target_values: ArrayLike,
    control_regressors: ArrayLike | pd.Series | pd.DataFrame,
    add_intercept: bool = True,
    return_model: bool = False,
) -> np.ndarray | tuple[np.ndarray, Any]:
    """
    Residualize one target vector with respect to control regressors using ordinary least squares.

    Parameters
    ----------
    target_values : array-like
        One-dimensional target values with shape ``(n_samples,)``.
    control_regressors : array-like, pd.Series, or pd.DataFrame
        Control regressors with shape ``(n_samples,)`` or ``(n_samples, n_control_regressors)``.
    add_intercept : bool, default=True
        If True, include an intercept column in the OLS design matrix.
    return_model : bool, default=False
        If True, also return the fitted statsmodels OLS result object.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, Any]
        Residualized target values with shape ``(n_samples,)``. If ``return_model=True``, also
        returns the fitted statsmodels OLS result object.
    """
    target_values = np.asarray(target_values)

    if isinstance(control_regressors, pd.Series):
        design_matrix = control_regressors.to_frame()
    elif np.ndim(control_regressors) == 1:
        design_matrix = np.asarray(control_regressors).reshape(-1, 1)
    else:
        design_matrix = control_regressors

    if add_intercept:
        design_matrix = sm.add_constant(design_matrix, has_constant='add')

    fitted_model = sm.OLS(target_values, design_matrix).fit()
    residualized_values = fitted_model.resid

    if return_model:
        return residualized_values, fitted_model
    else:
        return residualized_values


def residualize_dataframe_column(
    data: pd.DataFrame,
    target_column: str,
    control_columns: list[str],
) -> tuple[pd.Series, Any]:
    """
    Residualize one DataFrame column against others.

    Parameters
    ----------
    data : pd.DataFrame
        Input table with shape ``(n_samples, n_columns)``.
    target_column : str
        Name of the column to residualize.
    control_columns : list[str]
        Names of columns to regress out of ``target_column``.

    Returns
    -------
    tuple[pd.Series, Any]
        Residualized target column as a Series with shape ``(n_samples,)`` and the fitted
        statsmodels OLS result object.
    """
    design_matrix = sm.add_constant(data[control_columns], has_constant='add')
    target_values = data[target_column]

    fitted_model = sm.OLS(target_values, design_matrix).fit()
    residualized_series = pd.Series(
        fitted_model.resid,
        index=data.index,
        name=f"{target_column}_resid"
    )

    return residualized_series, fitted_model


def one_residualize_example(control_regressors, target_values):
    residualized_target, target_residualization_model = residualize_values(
        target_values,
        control_regressors,
        return_model=True,
    )
    print(target_residualization_model.summary())


def two_residualize_example():
    # A is left alone
    base_regressor = A

    # Remove A-like variance from B
    residualized_b_regressor = residualize_values(B, base_regressor)
    r, p = pearsonr(residualized_b_regressor, base_regressor)
    print(f"Correlation after residualization: r={r:.4f}, p={p:.3g}")

    # Remove A-like and B-like variance from C
    c_control_matrix = np.column_stack([base_regressor, residualized_b_regressor])
    residualized_c_regressor = residualize_values(C, c_control_matrix)
    residual_check_model = sm.OLS(
        residualized_c_regressor,
        sm.add_constant(np.column_stack([base_regressor, residualized_b_regressor])),
    ).fit()

    print(residual_check_model.rsquared)

    glm_regressor_table = pd.DataFrame(
        {
            "Q": base_regressor,
            "HMM_unique": residualized_b_regressor,
            "Hazard_unique": residualized_c_regressor,
            "Perseveration": perseveration,
            "Doubt": doubt,
        }
    )

    # DataFrame columns:
    # Q, HMM, Hazard, Perseveration, Doubt

    df["HMM_resid"], hmm_model = residualize_dataframe_column(
        df,
        target_column="HMM",
        control_columns=["Q"],
    )

    df["Hazard_resid"], hazard_model = residualize_dataframe_column(
        df,
        target_column="Hazard",
        control_columns=["Q", "HMM_resid"],
    )
