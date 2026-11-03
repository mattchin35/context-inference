"""Align source targets and one full-table block map to fixed timeline choices."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import is_present_value
from src.neural_analysis.interregional.preparation import build_block_fold_assignment
from src.neural_analysis.population.session_timeline import (
    SessionTimelineResult, prepare_timeline_choices,
)
from src.neural_analysis.task_decoding.conditions import build_condition_masks
from src.neural_analysis.task_decoding.config import (
    CONDITION_IDENTIFIERS, TARGET_DEFINITION_BY_IDENTIFIER, TaskDecodingConfig,
)
from src.neural_analysis.task_decoding.targets import build_target_table

from .config import TARGET_NAMES


@dataclass(frozen=True)
class PredictionInputs:
    """Aligned immutable-by-convention arrays; no file or estimator behavior.

    timeline contains independent regional scores (W_r,B,K_r) and maps into N
    shared choices. Targets/validity are float64/bool (N,T), in source units;
    baseline/correct_rewarded/fold_ids/block_values_json are (N,). Missing folds
    are -1. condition_masks is bool (N,C), retaining overlapping memberships.
    target_metadata preserves source/family/class labels; availability maps
    target names to an empty string or a reason. Time edges are relative seconds.
    """

    timeline: SessionTimelineResult
    target_names: tuple[str, ...]
    target_families: tuple[str, ...]
    target_values: np.ndarray
    target_valid: np.ndarray
    baseline_valid: np.ndarray
    correct_rewarded: np.ndarray
    fold_ids: np.ndarray
    block_values_json: np.ndarray
    condition_names: tuple[str, ...]
    condition_masks: np.ndarray
    training_bin_mask: np.ndarray
    bin_edges_s: np.ndarray
    target_metadata: dict
    target_availability: dict[str, str]


def training_bin_mask(bin_edges_s: np.ndarray) -> np.ndarray:
    """Return bool (B,) selecting complete [-1,+1) bins from (B+1,) second edges.

    Edges must be finite, increasing and contain both training boundaries.
    No bin is interpolated, averaged or reassigned to another choice.
    """
    edges = np.asarray(bin_edges_s, dtype=float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("Bin edges must be finite and strictly increasing")
    if not all(np.any(np.isclose(edges, bound, rtol=0, atol=1e-12)) for bound in (-1, 1)):
        raise ValueError("Bin edges must include exact -1 and +1 training boundaries")
    return (edges[:-1] >= -1 - 1e-12) & (edges[1:] <= 1 + 1e-12)


def prepare_prediction_inputs(
    trial_df: pd.DataFrame,
    timeline: SessionTimelineResult,
    base_config: TaskDecodingConfig,
) -> PredictionInputs:
    """Return targets/folds mapped from original trials to saved timeline choices.

    trial_df is the full chronological augmented (original trial,column) table;
    UTC columns use Unix seconds. timeline scores retain (regional window,bin,PC)
    order and units. base_config supplies existing target definitions; its
    target subset/alignment are locally replaced. Input objects are not changed.
    Reject mismatched choice identities using absolute UTC tolerance 1 us.
    """
    config = replace(base_config, target_names=TARGET_NAMES, alignment="choice_time")
    targets = build_target_table(
        trial_df, config, allow_missing_blocks=True, allow_empty_numerical_targets=True,
    )
    _, expected_choices = prepare_timeline_choices(trial_df)
    choices = timeline.choices
    if len(expected_choices) != len(choices):
        raise ValueError("Timeline choices do not match augmented-table identities")
    for column in ("row_position", "trial_id", "action"):
        if not np.array_equal(expected_choices[column].to_numpy(), choices[column].to_numpy()):
            raise ValueError(f"Timeline {column} does not match augmented-table identity")
    if not np.allclose(expected_choices.choice_time, choices.choice_time, rtol=0, atol=1e-6):
        raise ValueError("Timeline choice_time does not match augmented-table alignment")
    positions = choices.row_position.to_numpy(dtype=np.int64)
    grouping = trial_df[["cur_block"]].copy()
    grouping["cur_block"] = grouping.cur_block.where(is_present_value(grouping.cur_block))
    assignment = build_block_fold_assignment(grouping)
    all_folds = np.array([-1 if fold is None else fold for fold in assignment.fold_ids], dtype=np.int64)
    masks = build_condition_masks(trial_df, CONDITION_IDENTIFIERS)
    conditions = np.column_stack([masks[name][positions] for name in CONDITION_IDENTIFIERS])
    families = tuple(TARGET_DEFINITION_BY_IDENTIFIER[name].family for name in TARGET_NAMES)
    availability = {
        name: "target_unavailable_no_values" if family == "numerical" and not targets[name + "_valid"].any() else ""
        for name, family in zip(TARGET_NAMES, families, strict=True)
    }
    width = timeline.meta["bin_width_s"]
    from src.neural_analysis.population.session_timeline import choice_bin_edges

    edges = choice_bin_edges(width)
    for region in timeline.regions.values():
        mapping = region.choice_positions
        if (mapping.ndim != 1 or not np.issubdtype(mapping.dtype, np.integer)
                or np.any(mapping < 0) or np.any(mapping >= len(choices))
                or np.any(np.diff(mapping) <= 0)):
            raise ValueError("Regional choice mapping must be ordered unique choice positions")
        if (not np.array_equal(region.bin_edges_s, edges)
                or region.scores.ndim != 3 or region.scores.shape[:2] != (len(mapping), len(edges) - 1)
                or region.scores.shape[2] < 1 or not np.isfinite(region.scores).all()):
            raise ValueError("Regional score axes, finiteness or bin grid disagree")
    return PredictionInputs(
        timeline, TARGET_NAMES, families,
        targets.loc[:, TARGET_NAMES].to_numpy(dtype=float)[positions],
        targets.loc[:, [name + "_valid" for name in TARGET_NAMES]].to_numpy(dtype=bool)[positions],
        targets.baseline_valid.to_numpy(dtype=bool)[positions], conditions[:, 1].copy(),
        all_folds[positions], np.asarray(assignment.block_values_json, dtype=str)[positions],
        CONDITION_IDENTIFIERS, conditions, training_bin_mask(edges), edges,
        targets.attrs["target_metadata"], availability,
    )
