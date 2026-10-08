"""RED contracts for condition-resolved task-variable decoding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.spike_behavior.trials import make_trial_type_masks
from src.neural_analysis.task_decoding import conditions


CONDITION_ORDER = (
    "all",
    "correct_rewarded",
    "omission",
    "incorrect",
    "switch",
    "stay",
)


def _trial_table() -> pd.DataFrame:
    """Return one chronological condition fixture with shape ``(9, 4)``.

    Returns
    -------
    pandas.DataFrame
        Rows contain dimensionless binary outcomes/actions and the canonical
        experimenter-reward exclusion flag. The non-default index proves that
        output arrays follow row position rather than pandas labels.
    """
    return pd.DataFrame(
        {
            "experimenter_reward_given": [0, 0, 0, 0, 0, 1, 0, 0, 0],
            "correct": [1, 1, 0, 0, 1, 1, 1, 0, 1],
            "reward": [1, 0, 0, 0, 0, 1, 0, 0, 1],
            "action": [0, 0, 1, 1, 0, 1, np.nan, 1, 0],
        },
        index=[11, 13, 17, 19, 23, 29, 31, 37, 41],
    )


def test_condition_masks_exactly_reuse_shared_project_definitions() -> None:
    """Every named condition should equal the established shared trial mask."""
    trial_df = _trial_table()
    shared = make_trial_type_masks(trial_df)

    observed = conditions.build_condition_masks(trial_df, CONDITION_ORDER)

    assert tuple(observed) == CONDITION_ORDER
    np.testing.assert_array_equal(observed["all"], np.ones(len(trial_df), dtype=bool))
    for name in CONDITION_ORDER[1:]:
        np.testing.assert_array_equal(
            observed[name],
            shared[name].to_numpy(dtype=bool),
        )


def test_switch_and_stay_keep_the_existing_current_unrewarded_next_choice_rule() -> None:
    """Switch/stay should classify an unrewarded row from its immediate next row."""
    trial_df = _trial_table()

    observed = conditions.build_condition_masks(trial_df, ("switch", "stay"))

    np.testing.assert_array_equal(
        np.flatnonzero(observed["switch"]),
        np.array([1, 3], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(observed["stay"]),
        np.array([2], dtype=np.int64),
    )


def test_condition_masks_require_one_dimensional_canonical_boolean_outputs() -> None:
    """Unknown names and malformed shared masks must fail rather than filter silently."""
    trial_df = _trial_table()
    with pytest.raises(ValueError, match="condition"):
        conditions.build_condition_masks(trial_df, ("rewarded",))

    with pytest.raises(ValueError, match="condition|duplicate"):
        conditions.build_condition_masks(trial_df, ("all", "all"))


@pytest.mark.parametrize("missing_column", ("correct", "reward", "action"))
def test_nonpooled_conditions_require_all_shared_condition_sources(
    missing_column: str,
) -> None:
    """Condition filtering must report any absent source even for a target subset."""
    trial_df = _trial_table().drop(columns=[missing_column])

    with pytest.raises(ValueError, match=missing_column):
        conditions.build_condition_masks(trial_df, ("correct_rewarded",))


def test_all_condition_does_not_require_outcome_columns() -> None:
    """Legacy pooled-only configurations should preserve their earlier input schema."""
    trial_df = _trial_table().loc[:, ["experimenter_reward_given"]]

    observed = conditions.build_condition_masks(trial_df, ("all",))

    np.testing.assert_array_equal(observed["all"], np.ones(len(trial_df), dtype=bool))
