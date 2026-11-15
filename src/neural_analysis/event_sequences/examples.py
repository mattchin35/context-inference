"""Pure behavioral episode selection and source-preserving display intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import selection


SequenceCondition = tuple[str, str, int | None]
FAMILIES = ("probe", "omission_probe", "ideal_transition", "lucky_omission_switch", "indecisive_transition")
CONDITIONS = selection.ADDITIONAL_CONDITIONS + tuple(
    ("ideal_transition", "left" if direction == "left_to_right" else "right", repeats)
    for direction, repeats in selection.TRANSITION_CONDITIONS
)


def find_behavior_examples(
    trial_df: pd.DataFrame, *, probe_return_outcome: str = "rewarded_or_omission",
) -> dict[SequenceCondition, np.ndarray]:
    """Map full chronological (N,C) processed trials to 32 condition arrays.

    Inputs follow selection.prepare_sequence_trials, plus cur_block/state_int.
    action/state are 0=right, 1=left; choice_time is UTC Unix seconds. The
    return policy affects both probe families, as documented by the selector.
    Return integer (E,S) original-row positions per family/initial-side/repeat
    key; S=3..7. Empty arrays retain (0,S). No rows are removed or reordered,
    inputs are not modified, and neural sources/coverage are never inspected.
    """
    matches = selection.select_additional_sequences(trial_df, probe_return_outcome=probe_return_outcome)
    for (direction, repeats), positions in selection.select_ideal_transitions(trial_df).items():
        side = "left" if direction == "left_to_right" else "right"
        matches[("ideal_transition", side, repeats)] = positions
    return matches


def extract_example_trials(
    trial_df: pd.DataFrame, episode_positions: np.ndarray, *, trials_before: int = 5, trials_after: int = 5,
) -> pd.DataFrame:
    """Return one (M,C') display interval, including invalid surrounding rows.

    trial_df is a full (N,C) raw processed table or saved normalized table with
    row_position/trial_id/index_label identities. episode_positions is a nonempty
    integer (S,) vector of consecutive zero-based source positions. Context
    counts are nonnegative integers in original rows, not elapsed seconds.
    Return a RangeIndexed copy with bool in_episode and original identities;
    attrs.context_truncated states whether requested context met a session edge.
    Units, labels, row order and source inputs are unchanged. Invalid positions
    or counts raise ValueError. No behavioral matching or file IO occurs.
    """
    positions = np.asarray(episode_positions)
    if (positions.ndim != 1 or not positions.size or not np.issubdtype(positions.dtype, np.integer)
            or positions[0] < 0 or positions[-1] >= len(trial_df) or np.any(np.diff(positions) != 1)):
        raise ValueError("episode_positions must be consecutive integer positions inside the full trial table.")
    for count in (trials_before, trials_after):
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("Context counts must be nonnegative integers.")
    identities = {"row_position", "trial_id", "index_label"}
    trials = trial_df if identities.issubset(trial_df.columns) else selection.prepare_sequence_trials(trial_df)
    start = max(0, int(positions[0]) - trials_before)
    stop = min(len(trials), int(positions[-1]) + trials_after + 1)
    rows = trials.iloc[start:stop].copy().reset_index(drop=True)
    rows["in_episode"] = rows.row_position.isin(positions)
    rows.attrs["context_truncated"] = bool(start != positions[0] - trials_before or stop != positions[-1] + trials_after + 1)
    return rows
