"""Pure chronological task labels and deterministic baseline block folds."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.behavior_analysis.project_utils import normalize_experimenter_reward_column

TARGET_NAMES = ("context", "choice", "outcome", "left_choice_switch", "right_choice_switch")
CLASS_NAMES = (
    ("right", "left"), ("right", "left"),
    ("left_reward", "right_reward", "left_no_reward", "right_no_reward"),
    ("not_left_switch", "left_switch"), ("not_right_switch", "right_switch"),
)
SOURCE_FIELDS = ("cur_trial", "cur_block", "start_time", "choice_time", "state_int",
                 "action", "reward", "experimenter_reward_given")


@dataclass(frozen=True)
class PopulationTargets:
    """Full-table integer labels/Boolean validity/string reasons, each (N,5).

    Target order is TARGET_NAMES, unknown values are -1. class_names declares
    probability-column order per target. All codes/counts are dimensionless.
    """

    values: np.ndarray
    valid: np.ndarray
    reasons: np.ndarray
    class_names: tuple[tuple[str, ...], ...] = CLASS_NAMES


def numeric_column(values: object) -> np.ndarray:
    """Convert a one-dimensional source column to float (N,), unknowns -> NaN.

    Numeric codes retain their units; text no-choice/missing values remain
    unknown. This conversion does not change or reorder the input column.
    """
    result = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return result


def build_population_targets(state: object, action: object, reward: object,
                             manual_flags: object) -> PopulationTargets:
    """Map four complete chronological columns (N,) to five dimensionless labels.

    State/action codes are right=0, left=1; reward is absent=0/present=1.
    manual_flags must be known zero for an animal choice. Successor validity
    uses only the next unfiltered animal action, irrespective of its context,
    reward, or neural coverage. Terminal switch targets are unavailable.
    """
    state, action, reward, manual = [numeric_column(values) for values in (state, action, reward, manual_flags)]
    n = len(state)
    if any(values.shape != (n,) for values in (action, reward, manual)):
        raise ValueError("Target source columns must share shape (N,)")
    values = np.full((n, 5), -1, dtype=np.int8)
    reasons = np.full((n, 5), "unknown_target", dtype="U64")
    animal = np.isin(action, [0, 1]) & (manual == 0)
    context_valid = np.isin(state, [0, 1])
    values[context_valid, 0] = state[context_valid].astype(np.int8)
    values[animal, 1] = action[animal].astype(np.int8)
    outcome_valid = animal & np.isin(reward, [0, 1])
    values[outcome_valid, 2] = ((1 - action[outcome_valid]) + 2 * (1 - reward[outcome_valid])).astype(np.int8)
    if n > 1:
        available = animal[:-1] & animal[1:]
        indices = np.flatnonzero(available)
        values[indices, 3] = ((action[indices] == 1) & (action[indices + 1] == 0)).astype(np.int8)
        values[indices, 4] = ((action[indices] == 0) & (action[indices + 1] == 1)).astype(np.int8)
    valid = values >= 0
    reasons[valid] = ""
    reasons[:, 3:][~valid[:, 3:]] = "unknown_successor_or_animal_choice"
    if n:
        reasons[-1, 3:] = "terminal_trial"
    return PopulationTargets(values, valid, reasons)


def assign_block_folds(block_ids: object, baseline: np.ndarray) -> tuple[np.ndarray, str]:
    """Map complete block column and Boolean baseline(N,) to fold IDs (N,).

    Use five deterministic GroupKFold splits on baseline rows before target or
    regional masking. Invalid/missing groups receive -1. Fewer than five groups
    returns all -1 and a reason. IDs and counts are dimensionless.
    """
    blocks = pd.Series(block_ids).reset_index(drop=True)
    baseline = np.asarray(baseline)
    if baseline.shape != (len(blocks),) or baseline.dtype != bool:
        raise ValueError("baseline must be Boolean (N,)")
    folds = np.full(len(blocks), -1, dtype=np.int8)
    usable = baseline & blocks.notna().to_numpy()
    if any(isinstance(value, (float, np.floating)) and not np.isfinite(value) for value in blocks[usable]):
        raise ValueError("Block identities must be finite")
    codes, unique = pd.factorize(blocks[usable], sort=False)
    if len(unique) < 5:
        return folds, "Five folds require at least five usable baseline blocks"
    positions = np.flatnonzero(usable)
    for fold, (_, test) in enumerate(GroupKFold(n_splits=5, shuffle=False).split(np.zeros((len(codes), 1)), groups=codes)):
        folds[positions[test]] = fold
    return folds, "Missing block identities" if np.any(baseline & ~usable) else ""


def validate_trial_tables(augmented: pd.DataFrame, source: pd.DataFrame) -> None:
    """Validate complete tables (N,C) on source fields, with UTC-second times.

    Correctness/model columns are ignored. Paired unknowns compare equal;
    action text codes compare literally so no-choice categories cannot mask
    disagreements. No sorting, joining, mutation, or row removal is performed.
    Returns None or raises ValueError for structural/used-field disagreement.
    """
    first = normalize_experimenter_reward_column(augmented)
    second = normalize_experimenter_reward_column(source)
    for table in (first, second):
        if not set(SOURCE_FIELDS) <= set(table):
            raise ValueError(f"Missing source fields: {set(SOURCE_FIELDS) - set(table)}")
        if table.cur_trial.isna().any() or table.cur_trial.duplicated().any():
            raise ValueError("Original trial IDs must be present and unique")
        times = numeric_column(table.start_time)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Trial start_time must increase chronologically")
    if len(first) != len(second):
        raise ValueError("Complete source tables have different lengths")
    for field in SOURCE_FIELDS:
        a, b = first[field].reset_index(drop=True), second[field].reset_index(drop=True)
        equal = (a.eq(b) | (a.isna() & b.isna())).fillna(False)
        if not equal.all():
            raise ValueError(f"Source tables disagree in {field}")
