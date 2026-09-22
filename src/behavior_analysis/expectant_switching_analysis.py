"""Post hoc behavioral summaries for expectant-switching analyses.

These helpers describe observed or simulated sequences. True correct choice is
used only to label behavior after the fact; it never updates an exemplar.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RateSummary:
    """A binary-event count, opportunity count, and their ratio."""

    n_opportunities: int
    n_excursions: int
    rate: float


@dataclass(frozen=True)
class RecurrenceSummary:
    """Recurrences after eligible completed excursions in stable context."""

    n_eligible_completed_excursions: int
    n_recurrences: int
    fraction: float


def _validated_arrays(
    actions: np.ndarray,
    correct_choices: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned one-dimensional arrays after contract validation."""
    action_array = np.asarray(actions)
    correct_array = np.asarray(correct_choices)
    validity = np.asarray(valid_mask, dtype=bool)
    if action_array.ndim != 1 or correct_array.ndim != 1 or validity.ndim != 1:
        raise ValueError("inputs must be one-dimensional arrays.")
    if not (action_array.shape == correct_array.shape == validity.shape):
        raise ValueError("inputs must have matching shapes.")
    valid_actions = action_array[validity]
    valid_correct = correct_array[validity]
    if np.any(~np.isin(valid_actions, [0, 1])) or np.any(
        ~np.isin(valid_correct, [0, 1])
    ):
        raise ValueError("valid choices must be encoded 0=right or 1=left.")
    return action_array, correct_array, validity


def post_reward_excursion_rate(
    actions: np.ndarray,
    rewards: np.ndarray,
    valid_mask: np.ndarray,
) -> RateSummary:
    """Measure choice departures immediately after rewarded valid trials.

    Parameters
    ----------
    actions : ndarray, shape (n_trials,)
        Choices encoded ``0=right`` and ``1=left`` on valid rows.
    rewards : ndarray, shape (n_trials,)
        Reward amounts in task units; positive means rewarded.
    valid_mask : ndarray of bool, shape (n_trials,)
        Invalid rows are skipped and do not break valid-trial continuity.

    Returns
    -------
    RateSummary
        An opportunity is a rewarded valid trial with a later valid trial. An
        excursion occurs when that next valid choice differs. Rate is ``NaN``
        when there are no opportunities.
    """
    action_array = np.asarray(actions)
    reward_array = np.asarray(rewards, dtype=float)
    validity = np.asarray(valid_mask, dtype=bool)
    if action_array.ndim != 1 or reward_array.ndim != 1 or validity.ndim != 1:
        raise ValueError("actions, rewards, and valid_mask must be one-dimensional.")
    if not (action_array.shape == reward_array.shape == validity.shape):
        raise ValueError("actions, rewards, and valid_mask must have matching shapes.")
    valid_indices = np.flatnonzero(validity)
    if valid_indices.size:
        valid_actions = action_array[valid_indices]
        if np.any(~np.isin(valid_actions, [0, 1])):
            raise ValueError("valid actions must be encoded 0=right or 1=left.")

    opportunities = 0
    excursions = 0
    for current_index, next_index in zip(valid_indices[:-1], valid_indices[1:]):
        if reward_array[current_index] > 0:
            opportunities += 1
            excursions += int(action_array[next_index] != action_array[current_index])
    rate = excursions / opportunities if opportunities else float("nan")
    return RateSummary(opportunities, excursions, rate)


def _incorrect_runs(
    actions: np.ndarray,
    correct_choices: np.ndarray,
    valid_mask: np.ndarray,
) -> list[tuple[int, int, bool, int]]:
    """Return wrong-run positions in valid-trial coordinates.

    Each tuple is ``(start, stop_exclusive, returned, correct_side)``. A true
    correct-side change ends the current context episode and cannot count as a
    behavioral return.
    """
    action_array, correct_array, validity = _validated_arrays(
        actions, correct_choices, valid_mask
    )
    valid_actions = action_array[validity]
    valid_correct = correct_array[validity]
    runs: list[tuple[int, int, bool, int]] = []
    position = 0
    while position < valid_actions.size:
        if valid_actions[position] == valid_correct[position]:
            position += 1
            continue
        start = position
        correct_side = int(valid_correct[position])
        while (
            position < valid_actions.size
            and valid_correct[position] == correct_side
            and valid_actions[position] != valid_correct[position]
        ):
            position += 1
        returned = bool(
            position < valid_actions.size
            and valid_correct[position] == correct_side
            and valid_actions[position] == correct_side
        )
        runs.append((start, position, returned, correct_side))
    return runs


def incorrect_side_run_lengths(
    actions: np.ndarray,
    correct_choices: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return lengths of all incorrect-side runs in valid trials.

    Inputs are aligned one-dimensional arrays; choices use ``0=right`` and
    ``1=left``. The returned integer array has shape ``(n_runs,)`` and units of
    valid trials. Runs split at a true correct-side change.
    """
    return np.asarray(
        [stop - start for start, stop, _, _ in _incorrect_runs(actions, correct_choices, valid_mask)],
        dtype=int,
    )


def return_intervals(
    actions: np.ndarray,
    correct_choices: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Return departure-to-return intervals for completed excursions.

    The integer result has shape ``(n_completed_runs,)`` and units of valid
    trials. Each interval includes every incorrect trial and the first correct
    return trial. Runs interrupted by a true context change are incomplete.
    """
    return np.asarray(
        [stop - start + 1 for start, stop, returned, _ in _incorrect_runs(actions, correct_choices, valid_mask) if returned],
        dtype=int,
    )


def recurrence_without_context_change(
    actions: np.ndarray,
    correct_choices: np.ndarray,
    valid_mask: np.ndarray,
) -> RecurrenceSummary:
    """Measure repeated excursions within stable true-context episodes.

    A completed excursion is eligible when another incorrect run later starts
    before the true correct side changes. That later start is one recurrence.
    Counts are in events; the fraction is ``NaN`` if no completed excursion is
    eligible for recurrence.
    """
    runs = _incorrect_runs(actions, correct_choices, valid_mask)
    eligible = 0
    recurrences = 0
    for run_index, (_, stop, returned, correct_side) in enumerate(runs):
        if not returned:
            continue
        later_same_episode = any(
            later_start >= stop and later_side == correct_side
            for later_start, _, _, later_side in runs[run_index + 1 :]
        )
        if later_same_episode:
            eligible += 1
            recurrences += 1
    fraction = recurrences / eligible if eligible else float("nan")
    return RecurrenceSummary(eligible, recurrences, fraction)


def summarize_expectant_switching_by_reward_count(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize behavior within pre-trial expectancy-count strata.

    Parameters
    ----------
    frame : pandas.DataFrame, shape (n_trials, n_columns)
        Must contain ``action``, ``reward``, ``correct_choice``,
        ``expectancy_reward_count``, and ``valid``. Choices use ``0=right`` and
        ``1=left``; rewards use task reward units; counts use rewarded trials.

    Returns
    -------
    pandas.DataFrame
        One row per observed reward count, sorted ascending. Run and return
        statistics are calculated inside each stratum and therefore do not
        bridge count boundaries.
    """
    required = {
        "action",
        "reward",
        "correct_choice",
        "expectancy_reward_count",
        "valid",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    actions = frame["action"].to_numpy()
    rewards = frame["reward"].to_numpy(dtype=float)
    correct_choices = frame["correct_choice"].to_numpy()
    reward_counts = frame["expectancy_reward_count"].to_numpy()
    all_valid = frame["valid"].to_numpy(dtype=bool)
    valid_indices = np.flatnonzero(all_valid)

    transition_counts: dict[float, tuple[int, int]] = {}
    for current_index, next_index in zip(valid_indices[:-1], valid_indices[1:]):
        if rewards[current_index] <= 0 or np.isnan(reward_counts[current_index]):
            continue
        count = reward_counts[current_index]
        opportunities, excursions = transition_counts.get(count, (0, 0))
        transition_counts[count] = (
            opportunities + 1,
            excursions + int(actions[next_index] != actions[current_index]),
        )

    rows: list[dict[str, float | int]] = []
    for reward_count in sorted(frame["expectancy_reward_count"].dropna().unique()):
        in_count = reward_counts == reward_count
        valid = all_valid & in_count
        opportunities, excursions = transition_counts.get(reward_count, (0, 0))
        excursion_rate = excursions / opportunities if opportunities else float("nan")
        run_lengths = incorrect_side_run_lengths(
            actions, correct_choices, valid
        )
        intervals = return_intervals(
            actions, correct_choices, valid
        )
        recurrence = recurrence_without_context_change(
            actions, correct_choices, valid
        )
        rows.append(
            {
                "expectancy_reward_count": reward_count,
                "post_reward_opportunities": opportunities,
                "post_reward_excursions": excursions,
                "post_reward_excursion_rate": excursion_rate,
                "incorrect_run_count": int(run_lengths.size),
                "mean_incorrect_run_length": float(np.mean(run_lengths))
                if run_lengths.size
                else float("nan"),
                "completed_return_count": int(intervals.size),
                "mean_return_interval": float(np.mean(intervals))
                if intervals.size
                else float("nan"),
                "eligible_completed_excursions": (
                    recurrence.n_eligible_completed_excursions
                ),
                "recurrences": recurrence.n_recurrences,
                "recurrence_fraction": recurrence.fraction,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "expectancy_reward_count",
            "post_reward_opportunities",
            "post_reward_excursions",
            "post_reward_excursion_rate",
            "incorrect_run_count",
            "mean_incorrect_run_length",
            "completed_return_count",
            "mean_return_interval",
            "eligible_completed_excursions",
            "recurrences",
            "recurrence_fraction",
        ],
    )
