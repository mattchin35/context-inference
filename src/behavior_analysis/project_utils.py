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
LEGACY_GIVE_REWARD_COLUMN = "give_reward"
EXPERIMENTER_REWARD_GIVEN_COLUMN = "experimenter_reward_given"
NO_CHOICE_ACTION_LABELS = {"none", "no_choice", ""}


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


def make_no_choice_action_mask(actions) -> pd.Series:
    """Return actions that explicitly mark no animal choice.

    Parameters
    ----------
    actions : scalar, sequence, or pd.Series
        Trial action labels. Sequence and Series inputs have shape
        `(n_trials,)`; scalar inputs are treated as shape `(1,)`.

    Returns
    -------
    pd.Series
        Boolean Series with shape `(n_trials,)`. True marks old or new
        no-choice sentinels, including real missing values, string `"None"`,
        and string `"no_choice"`.
    """
    action_series = _as_series(actions)
    text_actions = action_series.astype(str).str.strip().str.lower()
    return action_series.isna() | text_actions.isin(NO_CHOICE_ACTION_LABELS)


def normalize_experimenter_reward_column(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Return a trial table using the explicit experimenter-reward flag name.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`. It may contain
        old `give_reward`, new `experimenter_reward_given`, both, or neither.

    Returns
    -------
    pd.DataFrame
        Copy of `trial_df` with at most one experimenter-reward flag column. Old
        `give_reward` is renamed to `experimenter_reward_given`; matching
        duplicate aliases keep only the new name.
    """
    has_legacy = LEGACY_GIVE_REWARD_COLUMN in trial_df.columns
    has_new = EXPERIMENTER_REWARD_GIVEN_COLUMN in trial_df.columns

    if not has_legacy and not has_new:
        return trial_df.copy()

    normalized = trial_df.copy()
    if has_legacy and not has_new:
        return normalized.rename(
            columns={LEGACY_GIVE_REWARD_COLUMN: EXPERIMENTER_REWARD_GIVEN_COLUMN}
        )

    if has_legacy and has_new:
        legacy_flags = _canonical_experimenter_reward_flags(normalized[LEGACY_GIVE_REWARD_COLUMN])
        new_flags = _canonical_experimenter_reward_flags(normalized[EXPERIMENTER_REWARD_GIVEN_COLUMN])
        if not legacy_flags.equals(new_flags):
            raise ValueError(
                "Trial dataframe has conflicting 'give_reward' and "
                "'experimenter_reward_given' columns."
            )
        normalized = normalized.drop(columns=[LEGACY_GIVE_REWARD_COLUMN])

    return normalized


def get_experimenter_reward_flags(
    trial_df: pd.DataFrame,
    default_zero: bool = True,
) -> np.ndarray:
    """Return experimenter-reward flags from a normalized or legacy trial table.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trialwise dataframe with shape `(n_trials, n_columns)`.
    default_zero : bool, default=True
        If True, missing experimenter-reward columns are treated as all-zero,
        which supports simulated runs with no manual rewards.

    Returns
    -------
    np.ndarray
        One-dimensional array with shape `(n_trials,)`.
    """
    normalized = normalize_experimenter_reward_column(trial_df)
    if EXPERIMENTER_REWARD_GIVEN_COLUMN not in normalized.columns:
        if default_zero:
            return np.zeros(normalized.shape[0], dtype=int)
        raise ValueError(
            "trial_df must contain 'experimenter_reward_given' or legacy 'give_reward'."
        )
    return np.copy(normalized[EXPERIMENTER_REWARD_GIVEN_COLUMN].to_numpy())


def _canonical_experimenter_reward_flags(values: pd.Series) -> pd.Series:
    """Normalize reward-flag aliases to comparable string values."""
    value_series = _as_series(values)
    canonical = pd.Series("missing", index=value_series.index, dtype=object)
    present_mask = is_present_value(value_series)
    present_values = value_series.loc[present_mask]

    zero_mask = is_zero_flag(present_values)
    canonical.loc[present_values.index[zero_mask.to_numpy()]] = "0"
    canonical.loc[present_values.index[~zero_mask.to_numpy()]] = "1"
    return canonical
