"""Project-wide small helpers shared by analysis modules."""

from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import pandas as pd

try:
    import autograd.numpy.random as autograd_random
except ImportError:  # pragma: no cover - autograd is expected in this project.
    autograd_random = None


MISSING_VALUE = "None"
MISSING_STRINGS = {MISSING_VALUE, "", "nan", "NaN"}


@contextmanager
def temporary_numpy_seed(seed: int | None):
    """Temporarily seed NumPy-style global random generators.

    Parameters
    ----------
    seed : int or None
        Integer seed used for NumPy and autograd NumPy random draws. If None,
        no RNG state is changed.

    Yields
    ------
    None
        Context block in which stochastic library initialization can run with
        deterministic global RNG state.
    """
    if seed is None:
        yield
        return

    numpy_state = np.random.get_state()
    autograd_state = None
    if autograd_random is not None and hasattr(autograd_random, "get_state"):
        autograd_state = autograd_random.get_state()

    np.random.seed(seed)
    if autograd_random is not None and hasattr(autograd_random, "seed"):
        autograd_random.seed(seed)

    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        if autograd_state is not None and hasattr(autograd_random, "set_state"):
            autograd_random.set_state(autograd_state)


def spawn_child_seeds(random_seed: int | None, n_children: int) -> list[int | None]:
    """Derive deterministic child seeds from one optional base seed.

    Parameters
    ----------
    random_seed : int or None
        Base seed for deterministic child-seed generation. If None, stochastic
        behavior is preserved by returning None for every child.
    n_children : int
        Number of child seeds to create.

    Returns
    -------
    list[int or None]
        One-dimensional list with shape `(n_children,)`. Integer seeds are
        deterministic and distinct for a fixed `random_seed`.
    """
    if n_children < 0:
        raise ValueError(f"n_children must be non-negative, got {n_children}")
    if random_seed is None:
        return [None] * n_children

    seed_sequence = np.random.SeedSequence(random_seed)
    child_sequences = seed_sequence.spawn(n_children)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in child_sequences]


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
