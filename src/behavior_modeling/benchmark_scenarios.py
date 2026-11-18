"""Immutable prescribed behaviors for the deterministic model benchmark.

Choice and correct-side arrays use ``0=right`` and ``1=left``. Each row is one
aligned trial; model predictions for a row are made before that row's outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BehaviorScenario:
    """One independent prescribed behavior sample.

    Parameters
    ----------
    scenario_id, title : str
        Stable identifier and human-readable label.
    action, correct_side : ndarray, shape (n_trials,)
        Side codes with ``0=right`` and ``1=left`` on valid rows. Invalid rows
        may use a sentinel such as ``-1``.
    reward : ndarray, shape (n_trials,)
        Realized rewards in task reward units. Positive values are rewarded;
        zero or negative values are omissions. Invalid rows may be ``NaN``.
    valid, correct, checkpoint : ndarray of bool, shape (n_trials,)
        Aligned validity, retrospective correctness, and highlighted pre-trial
        prediction masks.

    Notes
    -----
    Arrays are defensively copied, normalized to stable dtypes, validated, and
    marked read-only during construction.
    """

    scenario_id: str
    title: str
    action: np.ndarray
    reward: np.ndarray
    valid: np.ndarray
    correct_side: np.ndarray
    correct: np.ndarray
    checkpoint: np.ndarray

    def __post_init__(self) -> None:
        """Normalize, validate, and freeze all aligned trial arrays."""
        array_dtypes = {
            "action": np.int8,
            "reward": float,
            "valid": bool,
            "correct_side": np.int8,
            "correct": bool,
            "checkpoint": bool,
        }
        for field_name, dtype in array_dtypes.items():
            values = np.array(getattr(self, field_name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, field_name, values)
        validate_scenario(self)


def build_default_scenarios() -> tuple[BehaviorScenario, ...]:
    """Return the 22 prescribed benchmark samples in stable display order.

    Returns
    -------
    tuple of BehaviorScenario
        Independent immutable samples. Every array has shape ``(n_trials,)``;
        rewards are in task reward units and sides use ``0=right, 1=left``.
    """
    return (
        _scenario("AT-01", "Initial state", [1], [1], [1.0], [0]),
        _scenario(
            "AT-02", "Reward followed by stay", [1, 1], [1, 1], [1, 1], [1]
        ),
        _scenario(
            "AT-03", "Reward followed by switch", [1, 0], [1, 1], [1, 0], [1]
        ),
        _scenario(
            "AT-04",
            "Correct omission followed by stay",
            [1, 1],
            [1, 1],
            [0, 1],
            [1],
        ),
        _scenario(
            "AT-05",
            "Correct omission followed by switch",
            [1, 0],
            [1, 1],
            [0, 0],
            [1],
        ),
        _scenario(
            "AT-06",
            "Incorrect omission followed by switch",
            [1, 0],
            [0, 0],
            [0, 1],
            [1],
        ),
        _scenario(
            "AT-07",
            "Incorrect omission followed by stay",
            [1, 1],
            [0, 0],
            [0, 0],
            [1],
        ),
        _scenario(
            "PR-01",
            "Three-reward probe and three-omission return",
            [1, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 1],
            [3, 4, 5, 6],
        ),
        _scenario(
            "PR-02",
            "One-reward probe and one-omission return",
            [1, 0, 1],
            [1, 1, 1],
            [1, 0, 1],
            [1, 2],
        ),
        _scenario(
            "PR-03",
            "Switch after a correct omission",
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
            [2],
        ),
        _scenario(
            "PR-04",
            "Accumulated correct omissions before switching",
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [4],
        ),
        _scenario(
            "EX-01",
            "Rewards separated by correct omissions",
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0],
            [5],
        ),
        _scenario(
            "EX-02",
            "High expectancy without switching",
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5],
        ),
        _scenario(
            "EX-03",
            "Rewarded switch followed by adoption",
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [3, 4],
        ),
        _scenario(
            "EX-04",
            "Rewarded switch followed by immediate rejection",
            [1, 1, 1, 0, 1, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 1],
            [3, 4, 5],
        ),
        _scenario(
            "RS-01",
            "Repeated reward-switch and omission-return loop",
            [0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0],
            [1, 2, 3, 4, 5],
        ),
        _scenario(
            "RS-02",
            "Persistent incorrect sampling before return",
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 5],
        ),
        _scenario(
            "RS-03",
            "Alternation without continuing reward triggers",
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 2, 3, 4],
        ),
        _scenario(
            "CT-01",
            "True reversal after negative evidence",
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 1],
            [4],
        ),
        _scenario(
            "CT-02",
            "Stochastic correct omission without reversal",
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [4],
        ),
        _scenario(
            "CT-03A",
            "Prospective switch without reversal",
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 0],
            [3],
        ),
        _scenario(
            "CT-03B",
            "Prospective switch after hidden reversal",
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
            [3],
        ),
    )


def mirror_scenario(scenario: BehaviorScenario) -> BehaviorScenario:
    """Return an independent left/right mirror of one scenario.

    Parameters
    ----------
    scenario : BehaviorScenario
        Source sample with arrays of shape ``(n_trials,)``.

    Returns
    -------
    BehaviorScenario
        New immutable sample with valid actions and correct sides exchanged.
        Rewards, validity, correctness, and checkpoints are unchanged.
    """
    validate_scenario(scenario)
    mirrored_action = np.where(scenario.valid, 1 - scenario.action, -1)
    mirrored_correct_side = np.where(
        scenario.valid, 1 - scenario.correct_side, -1
    )
    return BehaviorScenario(
        scenario_id=f"{scenario.scenario_id}-mirror",
        title=f"{scenario.title} (mirrored)",
        action=mirrored_action,
        reward=scenario.reward,
        valid=scenario.valid,
        correct_side=mirrored_correct_side,
        correct=scenario.correct,
        checkpoint=scenario.checkpoint,
    )


def validate_scenario(scenario: BehaviorScenario) -> None:
    """Validate one aligned scenario without changing it.

    Parameters
    ----------
    scenario : BehaviorScenario
        Candidate sample. Trial arrays must be one-dimensional and aligned;
        valid side values use ``0=right, 1=left`` and valid rewards must be
        finite task-reward amounts.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If identifiers, shapes, valid-row data, correctness, or checkpoint
        placement violate the scenario contract.
    """
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id:
        raise ValueError("scenario_id must be a nonempty string.")
    if not isinstance(scenario.title, str) or not scenario.title:
        raise ValueError("title must be a nonempty string.")

    field_names = (
        "action",
        "reward",
        "valid",
        "correct_side",
        "correct",
        "checkpoint",
    )
    arrays = {name: np.asarray(getattr(scenario, name)) for name in field_names}
    if any(values.ndim != 1 for values in arrays.values()):
        raise ValueError("All scenario trial arrays must be one-dimensional.")
    expected_shape = arrays["action"].shape
    if expected_shape[0] == 0:
        raise ValueError("A scenario must contain at least one trial row.")
    if any(values.shape != expected_shape for values in arrays.values()):
        raise ValueError("All scenario trial arrays must have matching shapes.")

    valid = arrays["valid"].astype(bool, copy=False)
    if np.any(~np.isin(arrays["action"][valid], (0, 1))):
        raise ValueError("Valid actions must be 0 (right) or 1 (left).")
    if np.any(~np.isin(arrays["correct_side"][valid], (0, 1))):
        raise ValueError("Valid correct sides must be 0 (right) or 1 (left).")
    if np.any(~np.isfinite(arrays["reward"][valid])):
        raise ValueError("Valid rewards must be finite task-reward amounts.")

    expected_correct = (
        arrays["action"] == arrays["correct_side"]
    ) & valid
    if not np.array_equal(arrays["correct"], expected_correct):
        raise ValueError("correct must equal action == correct_side on valid rows.")
    if np.any(arrays["checkpoint"] & ~valid):
        raise ValueError("A checkpoint must identify a valid prediction row.")


# ---------------------------------------------------------------------------
# Helpers


def _scenario(
    scenario_id: str,
    title: str,
    action: list[int],
    correct_side: list[int],
    reward: list[float],
    checkpoint_indices: list[int],
) -> BehaviorScenario:
    """Build one explicit valid-row scenario from aligned Python literals."""
    action_array = np.asarray(action, dtype=np.int8)
    side_array = np.asarray(correct_side, dtype=np.int8)
    reward_array = np.asarray(reward, dtype=float)
    valid = np.ones(action_array.shape, dtype=bool)
    checkpoints = np.zeros(action_array.shape, dtype=bool)
    checkpoints[np.asarray(checkpoint_indices, dtype=int)] = True
    return BehaviorScenario(
        scenario_id=scenario_id,
        title=title,
        action=action_array,
        reward=reward_array,
        valid=valid,
        correct_side=side_array,
        correct=action_array == side_array,
        checkpoint=checkpoints,
    )
