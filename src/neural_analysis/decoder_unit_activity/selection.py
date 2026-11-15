"""Pure signed-weight summaries and deterministic, exclusive display assignments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

GROUP_IDS = (
    "left_context", "right_context", "left_choice", "right_choice",
    "left_reward", "right_reward", "left_no_reward", "right_no_reward",
    "left_to_right_switch", "right_to_left_switch",
)
GROUP_LABELS = (
    "Left context", "Right context", "Left choice", "Right choice",
    "Left reward", "Right reward", "Left no reward", "Right no reward",
    "L->R switch", "R->L switch",
)
MODEL_NAMES = ("context", "choice", "outcome", "left_switch_vs_stay", "right_switch_vs_stay")
OUTCOME_CLASSES = ("left_reward", "right_reward", "left_no_reward", "right_no_reward")
SUMMARY_COLUMNS = ("group_id", "group_index", "unit_id", "unit_index", "mean_weight",
                   "available_folds", "positive_folds", "percentile", "candidate", "exclusion_reason")
ASSIGNMENT_COLUMNS = ("unit_id", "unit_index", "group_id", "group_index", "mean_weight",
    "percentile", "positive_folds", "runner_up_group", "runner_up_percentile", "tied",
    "displayed", "display_order")


def build_group_weights(cells: Mapping[str, dict], unit_ids: Sequence[str]) -> np.ndarray:
    """Restore saved direct-unit cells to float weights (5,10,U), without mutation.

    cells maps supported model IDs to saved cell dictionaries with explicit
    classes, five fold records, original IDs, masks(U,), scalers(U,) in Hz and
    coefficients(1,P)/(4,P). Weights are logit/relative-logit per training unit
    SD. Omitted units, unavailable folds and absent models produce NaN, not zero.
    Structural disagreement raises ValueError; file provenance is checked by IO.
    """
    ids = _validate_unit_ids(unit_ids)
    if set(cells) - set(MODEL_NAMES):
        raise ValueError("Unsupported decoder model in coefficient inputs")
    weights = np.full((5, len(GROUP_IDS), len(ids)), np.nan)
    for name, cell in cells.items():
        try:
            records = _validate_cell_metadata(name, cell, ids)
            for record in records:
                if not record["available"]:
                    continue
                fold = record["fold"]
                beta, keep = _fold_parameters(cell, fold, len(ids), 4 if name == "outcome" else 1)
                restored = np.full((len(beta), len(ids)), np.nan)
                restored[:, keep] = beta
                if name == "outcome":
                    # Class-relative logits are invariant to a shared coefficient shift.
                    contrast = (4 * restored - restored.sum(axis=0, keepdims=True)) / 3
                    weights[fold, 4:8] = contrast
                elif name in ("context", "choice"):
                    start = 0 if name == "context" else 2
                    weights[fold, start] = restored[0]
                    weights[fold, start+1] = -restored[0]
                else:
                    weights[fold, 8 if name.startswith("left") else 9] = restored[0]
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError(f"Malformed saved coefficients for {name}: {error}") from error
    return weights


def summarize_group_weights(weights: np.ndarray, group_ids: Sequence[str],
                            unit_ids: Sequence[str]) -> pd.DataFrame:
    """Summarize float weights(5,10,U) into a labeled 10*U-row DataFrame.

    IDs define the existing axes; means retain coefficient units. Percentiles,
    counts and masks are dimensionless. Five finite folds, positive mean and
    >=4 positive folds define candidates. Rank all five-fold-usable signed
    means with average ties before sign filtering. No input is modified.
    """
    ids = _validate_unit_ids(unit_ids)
    values = np.asarray(weights, dtype=float)
    if tuple(group_ids) != GROUP_IDS or values.shape != (5, 10, len(ids)) or np.isinf(values).any():
        raise ValueError("Expected weights(5,10,U), canonical group IDs and finite-or-NaN coefficients")
    available = np.isfinite(values).sum(axis=0)
    positive = (values > 0).sum(axis=0)
    means = np.where(available == 5, np.where(np.isfinite(values), values, 0).sum(axis=0) / 5, np.nan)
    rows = []
    for group, group_id in enumerate(GROUP_IDS):
        usable = available[group] == 5
        percentiles = np.full(len(ids), np.nan)
        if usable.any():
            percentiles[usable] = (rankdata(means[group, usable], method="average") - .5) / usable.sum()
        for unit, unit_id in enumerate(ids):
            reason = ("incomplete_fold_support" if not usable[unit] else
                      "nonpositive_mean" if means[group, unit] <= 0 else
                      "inconsistent_sign" if positive[group, unit] < 4 else "")
            rows.append((group_id, group, unit_id, unit, means[group, unit],
                int(available[group, unit]), int(positive[group, unit]), percentiles[unit], not reason, reason))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def assign_unit_groups(summary: pd.DataFrame, max_units_per_group: int = 10) -> pd.DataFrame:
    """Assign a complete long-form summary to one row per original unit, no mutation.

    summary is the 10*U-row table from summarize_group_weights. cap is a positive
    Python integer. Return U assignment rows with group=-1 for unassigned units,
    runner-up diagnostics, and display_order=-1 when hidden. Percentiles/counts
    are dimensionless; mean_weight retains input coefficient units. Assignment
    precedes cap, preserving membership as the cap changes; no backfilling.
    """
    if type(max_units_per_group) is not int or max_units_per_group < 1:
        raise ValueError("max_units_per_group must be a positive integer")
    if not set(SUMMARY_COLUMNS) <= set(summary):
        raise ValueError("Incomplete group summary columns")
    rows = []
    identities = summary[["unit_id", "unit_index"]].drop_duplicates().sort_values("unit_index")
    for identity in identities.itertuples(index=False):
        candidates = summary.loc[(summary.unit_index == identity.unit_index) & summary.candidate].sort_values(
            ["percentile", "positive_folds", "group_index"], ascending=[False, False, True])
        record = {"unit_id": identity.unit_id, "unit_index": identity.unit_index,
            "group_id": "", "group_index": -1, "mean_weight": np.nan, "percentile": np.nan,
            "positive_folds": 0, "runner_up_group": "", "runner_up_percentile": np.nan,
            "tied": False, "displayed": False, "display_order": -1}
        if not candidates.empty:
            best = candidates.iloc[0]
            for key in ("group_id", "group_index", "mean_weight", "percentile", "positive_folds"):
                record[key] = best[key]
            if len(candidates) > 1:
                second = candidates.iloc[1]
                record.update(runner_up_group=second.group_id, runner_up_percentile=second.percentile,
                              tied=bool(best.percentile == second.percentile))
        rows.append(record)
    assigned = pd.DataFrame(rows, columns=ASSIGNMENT_COLUMNS)
    order = 0
    for group in range(len(GROUP_IDS)):
        chosen = assigned.loc[assigned.group_index == group].sort_values(
            ["percentile", "unit_id"], ascending=[False, True]).head(max_units_per_group)
        assigned.loc[chosen.index, "displayed"] = True
        assigned.loc[chosen.index, "display_order"] = np.arange(order, order + len(chosen))
        order += len(chosen)
    return assigned


# Validation helpers keep saved-file assumptions out of the numerical rules.
def _validate_unit_ids(unit_ids: Sequence[str]) -> tuple[str, ...]:
    """Validate ordered unique string IDs(U,), no physical units; return immutable tuple."""
    ids = tuple(unit_ids)
    if any(not isinstance(unit, str) or not unit for unit in ids) or len(set(ids)) != len(ids):
        raise ValueError("Unit IDs must be unique nonempty strings")
    return ids


def _validate_cell_metadata(name: str, cell: dict, unit_ids: tuple[str, ...]) -> list[dict]:
    """Validate saved cell identities/classes/fold IDs; return five dimensionless records."""
    meta = cell["meta"]
    side = name.split("_")[0] if name.endswith("switch_vs_stay") else None
    expected = OUTCOME_CLASSES if name == "outcome" else ("stay", "switch") if side else ("right", "left")
    if (meta["target"] != name or tuple(meta["unit_ids"]) != unit_ids
            or tuple(meta["class_names"]) != expected or meta.get("conditioning_side") != side
            or not np.array_equal(cell["class_codes"], np.arange(len(expected)))):
        raise ValueError(f"Saved unit/class/side axes disagree for {name}")
    records = meta["representations"]["units"]["folds"]
    fold_ids = [record["fold"] for record in records]
    if (len(records) != 5 or any(type(fold) is not int for fold in fold_ids)
            or set(fold_ids) != set(range(5))
            or any(type(record["available"]) is not bool for record in records)):
        raise ValueError(f"Expected unique fold IDs 0..4 for {name}")
    return records


def _fold_parameters(cell: dict, fold: int, units: int, classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Return finite beta(classes,P) and retained bool(U,) from one available saved fold.

    Scaler means/SDs are Hz(U,); beta maps training unit SDs to logits. Missing
    or malformed available parameters raise ValueError; arrays are unchanged.
    """
    keep = np.asarray(cell[f"fold_{fold}_retained_mask"])
    means = np.asarray(cell[f"fold_{fold}_mean_hz"])
    scales = np.asarray(cell[f"fold_{fold}_scale_hz"])
    beta = np.asarray(cell[f"fold_{fold}_units_coefficients"], dtype=float)
    if (keep.shape != (units,) or keep.dtype != bool or means.shape != (units,) or scales.shape != (units,)
            or not np.isfinite(means).all() or not np.isfinite(scales).all()):
        raise ValueError("Invalid training-unit mask/scaler axes")
    if not keep.any() or beta.shape != (classes, int(keep.sum())) or not np.isfinite(beta).all() or (scales[keep] <= 0).any():
        raise ValueError("Invalid available coefficient shape/scale")
    return beta, keep
