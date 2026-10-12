"""Contract tests for the deterministic behavior benchmark scenarios."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.behavior_modeling.benchmark_scenarios import (
    BehaviorScenario,
    build_default_scenarios,
    mirror_scenario,
    validate_scenario,
)


SCENARIO_IDS = (
    "AT-01",
    "AT-02",
    "AT-03",
    "AT-04",
    "AT-05",
    "AT-06",
    "AT-07",
    "PR-01",
    "PR-02",
    "PR-03",
    "PR-04",
    "EX-01",
    "EX-02",
    "EX-03",
    "EX-04",
    "RS-01",
    "RS-02",
    "RS-03",
    "CT-01",
    "CT-02",
    "CT-03A",
    "CT-03B",
)


# Each trial tuple is (action, reward, correct_side), using 0=right and 1=left.
EXPECTED_TRIALS = {
    "AT-01": [(1, 1.0, 1)],
    "AT-02": [(1, 1.0, 1), (1, 1.0, 1)],
    "AT-03": [(1, 1.0, 1), (0, 0.0, 1)],
    "AT-04": [(1, 0.0, 1), (1, 1.0, 1)],
    "AT-05": [(1, 0.0, 1), (0, 0.0, 1)],
    "AT-06": [(1, 0.0, 0), (0, 1.0, 0)],
    "AT-07": [(1, 0.0, 0), (1, 0.0, 0)],
    "PR-01": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (0, 0.0, 1),
        (0, 0.0, 1),
        (0, 0.0, 1),
        (1, 1.0, 1),
    ],
    "PR-02": [(1, 1.0, 1), (0, 0.0, 1), (1, 1.0, 1)],
    "PR-03": [(1, 1.0, 1), (1, 0.0, 1), (0, 0.0, 1)],
    "PR-04": [
        (1, 1.0, 1),
        (1, 0.0, 1),
        (1, 0.0, 1),
        (1, 0.0, 1),
        (0, 0.0, 1),
    ],
    "EX-01": [
        (1, 1.0, 1),
        (1, 0.0, 1),
        (1, 1.0, 1),
        (1, 0.0, 1),
        (1, 1.0, 1),
        (0, 0.0, 1),
    ],
    "EX-02": [(1, 1.0, 1)] * 6,
    "EX-03": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (0, 1.0, 0),
        (0, 1.0, 0),
        (0, 1.0, 0),
    ],
    "EX-04": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (0, 1.0, 0),
        (1, 0.0, 0),
        (0, 1.0, 0),
    ],
    "RS-01": [
        (0, 1.0, 0),
        (1, 0.0, 0),
        (0, 1.0, 0),
        (1, 0.0, 0),
        (0, 1.0, 0),
        (1, 0.0, 0),
    ],
    "RS-02": [
        (1, 1.0, 1),
        (0, 0.0, 1),
        (0, 0.0, 1),
        (0, 0.0, 1),
        (0, 0.0, 1),
        (1, 1.0, 1),
    ],
    "RS-03": [
        (1, 1.0, 1),
        (0, 0.0, 1),
        (1, 0.0, 1),
        (0, 0.0, 1),
        (1, 1.0, 1),
    ],
    "CT-01": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 0.0, 0),
        (0, 1.0, 0),
    ],
    "CT-02": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 0.0, 1),
        (0, 0.0, 1),
    ],
    "CT-03A": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (0, 0.0, 1),
    ],
    "CT-03B": [
        (1, 1.0, 1),
        (1, 1.0, 1),
        (1, 1.0, 1),
        (0, 1.0, 0),
    ],
}


EXPECTED_CHECKPOINTS = {
    "AT-01": [0],
    "AT-02": [1],
    "AT-03": [1],
    "AT-04": [1],
    "AT-05": [1],
    "AT-06": [1],
    "AT-07": [1],
    "PR-01": [3, 4, 5, 6],
    "PR-02": [1, 2],
    "PR-03": [2],
    "PR-04": [4],
    "EX-01": [5],
    "EX-02": [1, 2, 3, 4, 5],
    "EX-03": [3, 4],
    "EX-04": [3, 4, 5],
    "RS-01": [1, 2, 3, 4, 5],
    "RS-02": [1, 5],
    "RS-03": [1, 2, 3, 4],
    "CT-01": [4],
    "CT-02": [4],
    "CT-03A": [3],
    "CT-03B": [3],
}


@pytest.fixture
def scenarios() -> tuple[BehaviorScenario, ...]:
    """Return the complete immutable default scenario registry."""
    return build_default_scenarios()


def test_default_registry_has_stable_order_and_unique_ids(scenarios) -> None:
    """The default suite must expose exactly the 22 documented IDs in order."""
    actual_ids = tuple(scenario.scenario_id for scenario in scenarios)
    assert actual_ids == SCENARIO_IDS
    assert len(actual_ids) == len(set(actual_ids))


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_default_scenario_arrays_match_specification(scenarios, scenario_id) -> None:
    """Actions, outcomes, contexts, and checkpoints must match the spec."""
    scenario = next(item for item in scenarios if item.scenario_id == scenario_id)
    expected = np.asarray(EXPECTED_TRIALS[scenario_id], dtype=float)

    np.testing.assert_array_equal(scenario.action, expected[:, 0].astype(np.int8))
    np.testing.assert_array_equal(scenario.reward, expected[:, 1])
    np.testing.assert_array_equal(
        scenario.correct_side, expected[:, 2].astype(np.int8)
    )

    expected_checkpoint = np.zeros(expected.shape[0], dtype=bool)
    expected_checkpoint[EXPECTED_CHECKPOINTS[scenario_id]] = True
    np.testing.assert_array_equal(scenario.checkpoint, expected_checkpoint)


def test_scenario_arrays_are_aligned_typed_and_read_only(scenarios) -> None:
    """Frozen records must not expose mutable arrays or ambiguous dtypes."""
    expected_dtypes = {
        "action": np.dtype(np.int8),
        "reward": np.dtype(float),
        "valid": np.dtype(bool),
        "correct_side": np.dtype(np.int8),
        "correct": np.dtype(bool),
        "checkpoint": np.dtype(bool),
    }
    for scenario in scenarios:
        for field_name, expected_dtype in expected_dtypes.items():
            values = getattr(scenario, field_name)
            assert values.ndim == 1
            assert values.shape == scenario.action.shape
            assert values.dtype == expected_dtype
            assert not values.flags.writeable


def test_correctness_and_reward_are_consistent(scenarios) -> None:
    """Correctness is derived from action/context and reward is binary here."""
    for scenario in scenarios:
        np.testing.assert_array_equal(
            scenario.correct,
            (scenario.action == scenario.correct_side) & scenario.valid,
        )
        assert np.all(scenario.valid)
        assert set(np.unique(scenario.reward)).issubset({0.0, 1.0})


def test_mirror_swaps_sides_without_sharing_or_mutating_arrays(scenarios) -> None:
    """Mirroring must be symmetric and allocate independent immutable arrays."""
    original = next(item for item in scenarios if item.scenario_id == "EX-04")
    original_snapshots = {
        name: getattr(original, name).copy()
        for name in (
            "action",
            "reward",
            "valid",
            "correct_side",
            "correct",
            "checkpoint",
        )
    }

    mirrored = mirror_scenario(original)

    np.testing.assert_array_equal(mirrored.action, 1 - original.action)
    np.testing.assert_array_equal(mirrored.correct_side, 1 - original.correct_side)
    for field_name in ("reward", "valid", "correct", "checkpoint"):
        np.testing.assert_array_equal(
            getattr(mirrored, field_name), getattr(original, field_name)
        )
    for field_name, snapshot in original_snapshots.items():
        np.testing.assert_array_equal(getattr(original, field_name), snapshot)
        assert not np.shares_memory(
            getattr(original, field_name), getattr(mirrored, field_name)
        )


def test_validate_scenario_rejects_misalignment_and_inconsistent_metadata(
    scenarios,
) -> None:
    """Validation must reject shape, side, validity, and correctness errors."""
    base = scenarios[1]
    malformed_factories = (
        lambda: replace(base, reward=np.array([1.0])),
        lambda: replace(base, action=np.array([1, 2], dtype=np.int8)),
        lambda: replace(base, correct_side=np.array([1, -1], dtype=np.int8)),
        lambda: replace(base, reward=np.array([1.0, np.nan])),
        lambda: replace(base, correct=np.array([False, True])),
    )
    for build_malformed in malformed_factories:
        with pytest.raises(ValueError):
            build_malformed()


def test_invalid_rows_may_retain_alignment_without_valid_metadata() -> None:
    """An invalid row stays aligned and is excluded from correctness checks."""
    scenario = BehaviorScenario(
        scenario_id="invalid-row-test",
        title="Invalid row continuity",
        action=np.array([1, -1, 0], dtype=np.int8),
        reward=np.array([1.0, np.nan, 0.0]),
        valid=np.array([True, False, True]),
        correct_side=np.array([1, -1, 1], dtype=np.int8),
        correct=np.array([True, False, False]),
        checkpoint=np.array([False, False, True]),
    )

    validate_scenario(scenario)
    assert not scenario.action.flags.writeable
    assert not scenario.reward.flags.writeable
