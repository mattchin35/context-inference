"""Project-wide small helpers shared by analysis modules."""

from collections.abc import Iterable

import pandas as pd


MISSING_VALUE = "None"
MISSING_STRINGS = {MISSING_VALUE, "", "nan", "NaN"}


def _as_series(values) -> pd.Series:
    """Normalize scalar or one-dimensional values to a Series.

    Parameters
    ----------
    values : scalar, sequence, or pd.Series
        Values to normalize. Sequence and Series inputs are one-dimensional
        with shape `(n_values,)`; scalar inputs are treated as shape `(1,)`.

    Returns
    -------
    pd.Series
        One-dimensional Series with shape `(n_values,)`. Series inputs preserve
        their index; non-Series inputs use a default RangeIndex.
    """
    if isinstance(values, pd.Series):
        return values
    if isinstance(values, str) or not isinstance(values, Iterable):
        return pd.Series([values])
    return pd.Series(values)


def is_present_value(values) -> pd.Series:
    """Return values that are not real missing values or string sentinels.

    Parameters
    ----------
    values : scalar, sequence, or pd.Series
        Values to evaluate. Sequence and Series inputs are one-dimensional with
        shape `(n_values,)`; scalar inputs are treated as shape `(1,)`.

    Returns
    -------
    pd.Series
        Boolean Series with shape `(n_values,)`. Series inputs preserve their
        index; non-Series inputs use a default RangeIndex.
    """
    value_series = _as_series(values)
    text_values = value_series.astype(str).str.strip()
    return value_series.notna() & ~text_values.isin(MISSING_STRINGS)


def is_zero_flag(values) -> pd.Series:
    """Return flags that are explicitly numeric zero.

    Parameters
    ----------
    values : scalar, sequence, or pd.Series
        Numeric flag values to evaluate. Sequence and Series inputs are one-
        dimensional with shape `(n_values,)`; scalar inputs are treated as shape
        `(1,)`. Present values must be numeric or numeric strings.

    Returns
    -------
    pd.Series
        Boolean Series with shape `(n_values,)`, True only where values equal
        numeric zero. Missing sentinels are returned as False. Series inputs
        preserve their index; non-Series inputs use a default RangeIndex.
    """
    value_series = _as_series(values)
    present_mask = is_present_value(value_series)
    present_values = value_series.loc[present_mask]
    numeric_values = pd.to_numeric(present_values, errors="coerce")

    invalid_values = present_values.loc[numeric_values.isna()]
    if not invalid_values.empty:
        invalid_summary = sorted(invalid_values.astype(str).unique())
        raise ValueError(f"Flag values must be numeric zero/nonzero, got: {invalid_summary}")

    zero_mask = pd.Series(False, index=value_series.index)
    zero_mask.loc[numeric_values.index] = numeric_values.eq(0)
    return zero_mask
