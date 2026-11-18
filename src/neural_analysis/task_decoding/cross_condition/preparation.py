"""Original-row behavioral alignment and explicit complementary transfer cells."""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from src.behavior_analysis.project_utils import is_present_value
from src.neural_analysis.population.session_timeline import prepare_timeline_choices
from src.neural_analysis.task_decoding.conditions import build_condition_masks
from src.neural_analysis.task_decoding.config import CONDITION_IDENTIFIERS, TaskDecodingConfig
from src.neural_analysis.task_decoding.session_prediction.preparation import training_bin_mask
from src.neural_analysis.task_decoding.targets import build_target_table, validate_augmented_trials

from .config import TARGET_NAMES, TRANSFERS, WINDOW_NAMES, TransferDefinition


@dataclass(frozen=True)
class TransferInputs:
    """Aligned behavioral arrays and cell definitions; no regional neural scores.

    N=ordinary saved choices, T=4 targets, S=18 splits, H=3 windows, B=bins,
    C=canonical conditions. choices is a copied (N,column) DataFrame retaining
    original row_position/trial_id and UTC Unix-second choice_time. target_values
    is float64 (N,T), target_valid bool (N,T); labels are dimensionless and retain
    loader validity even when common eligibility excludes a trial. baseline_valid
    is bool (N,), with dark context excluded; conflict_mask is separate bool (N,).
    condition_masks is bool (N,C), intersected with baseline and no-conflict
    eligibility. source_cells/test_cells are bool (N,S,2), last axis actual class
    0/1, with eligibility applied. window_bin_masks is bool (H,B), selected from
    bin_edges_s float (B+1,) in choice-relative seconds without resampling.
    Names, target_metadata and transfers retain label/role semantics. conflicts
    is a DataFrame auditing original row_position/trial_id and raw recorded
    state_int/action/correct plus reason; dark_state_count counts ordinary dark
    choices. All records are immutable by convention; no fold axis is present.
    """

    choices: pd.DataFrame
    target_names: tuple[str, ...]
    target_metadata: dict
    target_values: np.ndarray
    target_valid: np.ndarray
    baseline_valid: np.ndarray
    conflict_mask: np.ndarray
    condition_names: tuple[str, ...]
    condition_masks: np.ndarray
    source_cells: np.ndarray
    test_cells: np.ndarray
    window_names: tuple[str, ...]
    window_bin_masks: np.ndarray
    bin_edges_s: np.ndarray
    dark_state_count: int
    conflicts: pd.DataFrame
    transfers: tuple[TransferDefinition, ...]


def prepare_transfer_inputs(
    trial_df: pd.DataFrame,
    choices: pd.DataFrame,
    bin_edges_s: np.ndarray,
    base_config: TaskDecodingConfig,
) -> TransferInputs:
    """Align full chronological trials to saved choices and construct all transfer masks.

    trial_df is (J,column), original chronological rows with dimensionless
    labels/flags and UTC Unix-second timestamps; its index is not row identity.
    choices is (N,column) from saved PCA or prepare_timeline_choices, retaining
    row_position, trial_id, action and choice_time. bin_edges_s is finite float
    (B+1,) in choice-relative seconds, including exact -1/0/+1 boundaries.
    base_config supplies existing target contracts; selected targets/alignment
    are replaced locally and blocks ignored. Returns TransferInputs with copied
    choice coordinates and explicit N/T/S/H axes; inputs stay unchanged. Raises
    ValueError on malformed labels, missing ordinary context or misalignment.
    No neural data is loaded and no chronological rows are dropped before masks.
    """
    config = replace(base_config, target_names=TARGET_NAMES, alignment="choice_time")
    # Validate baseline fields before the choice helper consumes manual flags.
    validate_augmented_trials(trial_df, replace(config, target_names=("current_action",)), ignore_blocks=True)
    _, expected_choices = prepare_timeline_choices(trial_df)
    ordinary = np.zeros(len(trial_df), dtype=bool)
    ordinary[expected_choices.row_position.to_numpy(dtype=np.int64)] = True
    working = trial_df.copy(deep=True)
    if "state_int" not in working:
        raise ValueError("Augmented trial table is missing required state_int")
    # Excluded rows need no context; validate every other source using the loader.
    working["state_int"] = working.state_int.astype(object).where(ordinary, np.nan)
    targets = build_target_table(working, config, ignore_blocks=True)
    state = pd.to_numeric(working.state_int.where(is_present_value(working.state_int)), errors="raise").to_numpy(dtype=float)
    if np.any(ordinary & ~np.isin(state, [0, 1, 2])):
        raise ValueError("state_int must be present and binary, or known dark state 2, on ordinary choices")
    aligned_choices = _aligned_choices(expected_choices, choices)
    positions = expected_choices.row_position.to_numpy(dtype=np.int64)
    dark = ordinary & (state == 2)
    conflict, audit = _conflict_audit(trial_df, targets, state, ordinary)
    baseline = targets.baseline_valid.to_numpy(dtype=bool) & ~dark
    eligible = (baseline & ~conflict)[positions]
    values = targets.loc[:, TARGET_NAMES].to_numpy(dtype=np.float64)[positions]
    valid = targets.loc[:, [name + "_valid" for name in TARGET_NAMES]].to_numpy(dtype=bool)[positions]
    normalized = _normalized_condition_trials(
        targets.current_action.to_numpy(dtype=float),
        targets.current_action_is_correct.to_numpy(dtype=float),
        targets.current_reward.to_numpy(dtype=float),
        working.experimenter_reward_given,
    )
    canonical = build_condition_masks(normalized, CONDITION_IDENTIFIERS)
    conditions = {name: mask[positions] & eligible for name, mask in canonical.items()}
    cells = _named_cells(values, conditions, eligible)
    source, test = _transfer_cells(cells, values, valid)
    windows = np.vstack([training_bin_mask(bin_edges_s, training_window=name) for name in WINDOW_NAMES])
    return TransferInputs(
        aligned_choices, TARGET_NAMES, targets.attrs["target_metadata"], values, valid,
        baseline[positions], conflict[positions], CONDITION_IDENTIFIERS,
        np.column_stack([conditions[name] for name in CONDITION_IDENTIFIERS]), source, test,
        WINDOW_NAMES, windows, np.asarray(bin_edges_s, dtype=float).copy(), int(dark.sum()), audit, TRANSFERS,
    )


def _aligned_choices(expected: pd.DataFrame, supplied: pd.DataFrame) -> pd.DataFrame:
    """Return a copied (N,column) choice table matching expected original coordinates.

    Both DataFrames contain row_position/trial_id/action and UTC-second
    choice_time. Compare times at absolute 1 us, relative zero; indices have no
    identity meaning. Raise ValueError on mismatched columns, counts or values.
    """
    columns = ("row_position", "trial_id", "action", "choice_time")
    if not isinstance(supplied, pd.DataFrame) or not set(columns).issubset(supplied.columns):
        raise ValueError("Saved choices require row_position, trial_id, action and choice_time")
    if len(expected) != len(supplied):
        raise ValueError("Saved choices do not match augmented-table choice count")
    for column in columns[:-1]:
        if not np.array_equal(expected[column].to_numpy(), supplied[column].to_numpy()):
            raise ValueError(f"Saved {column} does not match augmented-table identity")
    if not np.allclose(expected.choice_time, supplied.choice_time, rtol=0, atol=1e-6):
        raise ValueError("Saved choice_time does not match augmented-table alignment")
    return supplied.copy(deep=True)


def _conflict_audit(
    trial_df: pd.DataFrame, targets: pd.DataFrame, state: np.ndarray, ordinary: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return bool conflict(J,) and raw original-row audit DataFrame for binary conflicts.

    trial_df/targets have J chronological rows; float state(J,) and bool
    ordinary(J,) preserve row axes. Labels/IDs are dimensionless. Only rows
    with all binary state/action/correct labels can conflict. Raw recorded
    values and original zero-based positions are copied without normalization.
    """
    action = targets.current_action.to_numpy(dtype=float)
    correct = targets.current_action_is_correct.to_numpy(dtype=float)
    complete = ordinary & np.isin(state, [0, 1]) & np.isfinite(action) & np.isfinite(correct)
    conflict = complete & (correct != (state == action))
    positions = np.flatnonzero(conflict)
    audit = trial_df.iloc[positions][["cur_trial", "state_int", "action", "correct"]].copy()
    audit = audit.rename(columns={"cur_trial": "trial_id"}).reset_index(drop=True)
    audit.insert(0, "row_position", positions)
    audit["reason"] = "recorded_correctness_conflicts_with_context_and_action"
    return conflict, audit


def _normalized_condition_trials(
    action: np.ndarray, correct: np.ndarray, reward: np.ndarray, manual_flags: pd.Series,
) -> pd.DataFrame:
    """Return numeric mask-building columns (J,4) after public target-loader validation.

    action/correct/reward are normalized float (J,) arrays; manual_flags is the
    original (J,) Series in the same chronological order. All values are
    dimensionless and project missing entries stay NaN. The public loader must
    already have validated present source values. Input arrays/Series stay
    unchanged; the returned DataFrame has a new positional index.
    """
    return pd.DataFrame({
        "action": action,
        "correct": correct,
        "reward": reward,
        "experimenter_reward_given": pd.to_numeric(
            manual_flags.where(is_present_value(manual_flags)), errors="raise",
        ).to_numpy(dtype=float),
    })


def _named_cells(
    values: np.ndarray, conditions: dict[str, np.ndarray], eligible: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return named bool cells(N,) from dimensionless labels(N,4) and canonical masks.

    conditions and eligible contain bool (N,) values after common filters.
    Correct combines canonical correct-rewarded and omission; unrewarded combines
    omission and actual-unrewarded incorrect. Actual reward 1 is independent of
    correctness, including otherwise eligible trials with missing correctness.
    Context outcome cells and complementary unions use recorded context and
    canonical outcomes, including rewarded or unknown-reward incorrect trials.
    """
    state, action, reward = values[:, 0], values[:, 1], values[:, 2]
    correct = conditions["correct_rewarded"] | conditions["omission"]
    incorrect = conditions["incorrect"]
    unrewarded = conditions["omission"] | (incorrect & (reward == 0))
    rewarded = eligible & (reward == 1)
    cells = {}
    for side, label in (("right", 0), ("left", 1)):
        cells[f"context_{side}_correct"] = (state == label) & correct
        cells[f"context_{side}_incorrect"] = (state == label) & incorrect
        cells[f"context_{side}_correct_rewarded"] = (state == label) & conditions["correct_rewarded"]
        cells[f"context_{side}_omission"] = (state == label) & conditions["omission"]
        cells[f"context_{side}_omission_or_incorrect"] = (state == label) & (conditions["omission"] | incorrect)
        cells[f"context_{side}_correct_rewarded_or_incorrect"] = (state == label) & (
            conditions["correct_rewarded"] | incorrect
        )
        cells[f"choice_{side}_correct"] = (action == label) & correct
        cells[f"choice_{side}_incorrect"] = (action == label) & incorrect
        cells[f"choice_{side}_rewarded"] = (action == label) & rewarded
        cells[f"choice_{side}_unrewarded"] = (action == label) & unrewarded
    return cells


def _transfer_cells(
    cells: dict[str, np.ndarray], values: np.ndarray, valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack named bool cells(N,) into source/test bool (N,S=18,2) actual-class axes.

    values/valid are float/bool (N,4) dimensionless targets. Verify each selected
    cell has the registry's actual label and source/test roles are disjoint.
    Empty cells stay empty; labels or memberships are never synthesized.
    """
    source = np.zeros((len(values), len(TRANSFERS), 2), dtype=bool)
    test = np.zeros_like(source)
    for split, definition in enumerate(TRANSFERS):
        target = TARGET_NAMES.index(definition.target_name)
        for output, names in ((source, definition.source_cells), (test, definition.test_cells)):
            for label, name in enumerate(names):
                selected = cells[name]
                if np.any(selected & (~valid[:, target] | (values[:, target] != label))):
                    raise ValueError(f"Transfer {definition.split_id} cell {name} has invalid class orientation")
                output[:, split, label] = selected
        if np.any(source[:, split].any(axis=1) & test[:, split].any(axis=1)):
            raise ValueError(f"Transfer {definition.split_id} source and test cells overlap")
    return source, test
