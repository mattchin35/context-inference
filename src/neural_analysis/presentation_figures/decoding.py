"""Regional choice and outcome readouts with pooled whole-block validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.neural_analysis.population_decoding.features import (
    FoldPCA, UnitScaling, fit_fold_pca, fit_unit_scaling, transform_pc_activity,
    transform_unit_activity,
)
from src.neural_analysis.population_decoding.modeling import (
    ClassifierFit, fit_classifier, predict_probabilities,
)
from src.neural_analysis.population_decoding.targets import TARGET_NAMES, numeric_column

from .preparation import REGIONAL_WINDOW_S, REGIONS, PreparedChoices

WIDTHS_MS = (500, 100)
READOUTS = (
    ("context", "all"), ("choice", "all"), ("context", "correct_rewarded"),
    ("choice", "correct_rewarded"), ("outcome", "all"),
)
GROUPS = ("overall", "correct_rewarded", "omission", "incorrect")
MAX_REGIONAL_PCS = 30
PCS = tuple(range(MAX_REGIONAL_PCS + 1))
PROBABILITY_COLUMNS = ("p_0", "p_1", "p_2", "p_3")


@dataclass(frozen=True)
class Readout:
    """Training-only Hz scaling, optional fold PCA, and fixed logistic fits by PC count.

    A model key of zero denotes standardized units. Positive keys denote PC
    prefixes. `reason` describes failure of the all-unit fit; an absent positive
    key indicates an unsupported or failed PC fit.
    """

    scaling: UnitScaling | None
    pca: FoldPCA | None
    models: dict[int, ClassifierFit]
    reason: str = ""


def compute_decoding(prepared: PreparedChoices) -> dict[str, pd.DataFrame]:
    """Return Figures 1-3 tables from joint (choice, bin, unit) spike counts.

    Outputs include pooled scores (dimensionless fraction), all-unit per-bin
    probabilities and original trial IDs, class order, confusion and reward
    discrimination, fold/block support, transfer mapping, PC recovery and
    recorded trial-table delivery latencies (seconds).
    """
    tables = {"scores": [], "probabilities": [], "folds": [], "confusion": [],
              "reward_discrimination": [], "recovery": [], "transfer_mapping": []}
    for width_ms in WIDTHS_MS:
        for region in REGIONS:
            for target, regime in READOUTS:
                head = decode_region(prepared, region, target, regime, width_ms)
                for name, table in head.items():
                    tables[name].append(table)
            tables["transfer_mapping"].append(_transfer_mapping(prepared, region, width_ms))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items() if parts}
    result["units"] = pd.DataFrame(
        [(region, len(prepared.unit_ids.get(region, ()))) for region in REGIONS],
        columns=["region", "n_units"],
    )
    result["classes"] = class_table(prepared)
    result["delivery_latency"] = _delivery_latency(prepared)
    return result


def decode_region(prepared: PreparedChoices, region: str, target: str, regime: str,
                  width_ms: int) -> dict[str, pd.DataFrame]:
    """Fit one regional readout, returning fold, probability and pooled score tables.

    Prepared counts are spikes/bin on (window choice, bin, unit), indexed by
    original source row positions. Binary targets use [-1,1) s; outcome uses
    [0,1) s. The all-unit and PC models inherit identical test identities.
    """
    window = prepared.windows[REGIONAL_WINDOW_S, width_ms]
    positions = window.row_positions
    target_index = TARGET_NAMES.index(target)
    labels = prepared.targets.values[positions, target_index]
    eligible = prepared.targets.valid[positions, target_index]
    class_ids = tuple(range(len(prepared.targets.class_names[target_index])))
    bin_start = (len(window.bin_edges_s) - 1) // 2 if target == "outcome" else 0
    edges = window.bin_edges_s[bin_start:]
    probabilities = np.full((len(PCS), len(positions), len(edges) - 1, len(class_ids)), np.nan)
    choice_blocks = prepared.trials.cur_block.iloc[positions].to_numpy()
    train_allowed = eligible.copy()
    if regime == "correct_rewarded":
        train_allowed &= prepared.condition_masks[regime][positions]

    identity = dict(region=region, target=target, regime=regime, width_ms=width_ms)
    source_reason = _source_reason(prepared, region, len(positions), region in window.counts)
    counts = window.counts.get(region)
    unit_ids = prepared.unit_ids.get(region, ())
    fold_rows = []
    for split in window.splits:
        train = split.train_indices[train_allowed[split.train_indices]]
        test = split.test_indices[eligible[split.test_indices]]
        reason = source_reason
        if not reason and not len(test):
            reason = "no_eligible_test_choices"
        if not reason:
            fitted = fit_readout(counts[train, bin_start:] / (width_ms / 1000),
                                 labels[train], unit_ids, class_ids, PCS)
            reason = fitted.reason
            if fitted.models:
                projected = predict_readout(counts[test, bin_start:] / (width_ms / 1000),
                                            unit_ids, fitted)
                for n_pcs, values in projected.items():
                    probabilities[n_pcs, test] = values
        fold_rows.append(dict(
            **identity, fold_id=split.fold_id, n_train=len(train), n_test=len(test),
            n_purged=int(train_allowed[split.purged_indices].sum()), reason=reason,
            **{f"blocks_{code}": len(pd.unique(choice_blocks[train[labels[train] == code]]))
               if code in class_ids else 0 for code in range(4)},
        ))

    scores, recovery = _summarize_readout(
        prepared, identity, positions, labels, eligible, probabilities, edges, source_reason,
    )
    result = {
        "scores": scores, "recovery": recovery, "folds": pd.DataFrame(fold_rows),
        "probabilities": _probability_table(prepared, identity, positions[eligible], labels[eligible],
                                            probabilities[0, eligible], edges, bin_start),
    }
    if target == "outcome":
        result.update(_outcome_tables(identity, labels[eligible], probabilities[0, eligible], source_reason))
    return result


def fit_readout(training_rates_hz: np.ndarray, labels: np.ndarray, unit_ids: tuple[str, ...],
                classes: tuple[int, ...], dimensions: tuple[int, ...]) -> Readout:
    """Fit on (choice, supervised bin, unit) Hz and integer choice labels (choice,).

    Every choice contributes total dimensionless weight one. Unit IDs and class
    IDs follow the supplied orders. Dimensions contain zero for standardized
    units and/or positive PC prefixes. Return shared scaling, optional PCA and
    estimated logistic parameters.
    """
    if not len(labels):
        return Readout(None, None, {}, "no_eligible_training_choices")
    if set(np.unique(labels)) != set(classes):
        return Readout(None, None, {}, "training_class_missing")
    try:
        scaling = fit_unit_scaling(training_rates_hz, unit_ids)
    except ValueError as error:
        return Readout(None, None, {}, str(error))
    standardized = transform_unit_activity(training_rates_hz, unit_ids, scaling)
    flattened = standardized.reshape(-1, standardized.shape[-1])
    bins = training_rates_hz.shape[1]
    bin_labels = np.repeat(labels, bins)
    weights = np.full(len(bin_labels), 1 / bins)
    models = {}
    reason = ""
    if 0 in dimensions:
        fit = fit_classifier(flattened, bin_labels, weights, classes)
        if not fit.available:
            return Readout(scaling, None, {}, fit.reason)
        models[0] = fit
    pca = None
    if any(dimensions):
        try:
            pca = fit_fold_pca(flattened)
        except (ValueError, np.linalg.LinAlgError):
            return Readout(scaling, None, models, reason)
        pcs = transform_pc_activity(flattened, pca)
        for n_pcs in dimensions:
            if 0 < n_pcs <= pcs.shape[-1]:
                fit = fit_classifier(pcs[:, :n_pcs], bin_labels, weights, classes)
                if fit.available:
                    models[n_pcs] = fit
    return Readout(scaling, pca, models, reason)


def predict_readout(rates_hz: np.ndarray, unit_ids: tuple[str, ...],
                    fitted: Readout) -> dict[int, np.ndarray]:
    """Project (choice, bin, unit) Hz through saved features and classifier.

    Return ordered dimensionless probabilities (choice, bin, class) keyed by
    model dimension. No training or changes to the inputs occur.
    """
    if not fitted.models:
        return {}
    standardized = transform_unit_activity(rates_hz, unit_ids, fitted.scaling)
    flattened = standardized.reshape(-1, standardized.shape[-1])
    projected = transform_pc_activity(flattened, fitted.pca) if fitted.pca is not None else None
    return {
        n_pcs: predict_probabilities(
            flattened if n_pcs == 0 else projected[:, :n_pcs], model,
        )[0].reshape(*rates_hz.shape[:2], len(model.classes))
        for n_pcs, model in fitted.models.items()
    }


def score_choices(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    """Average (choice, bin, class) probabilities, then score integer choice labels.

    Return balanced accuracy (fraction), reason, eligible/predicted counts,
    (class,) eligible counts and (class, class) integer confusion counts.
    Missing predictions and absent true classes make the primary score NaN.
    """
    probabilities, labels = np.asarray(probabilities, float), np.asarray(labels)
    if (probabilities.ndim != 3 or not probabilities.shape[1]
            or labels.shape != probabilities.shape[:1] or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(labels, np.arange(probabilities.shape[-1])).all()):
        raise ValueError("Expected (choice, bin, class) probabilities and integer (choice,) labels")
    class_count = probabilities.shape[-1]
    complete = np.isfinite(probabilities).all(axis=(1, 2))
    counts = np.bincount(labels, minlength=class_count)
    confusion = np.zeros((class_count, class_count), dtype=int)
    accuracy = np.nan
    n_predicted = int(complete.sum())
    if not len(labels):
        reason = "no_eligible_choices"
    elif n_predicted < len(labels):
        reason = "missing_predictions"
    else:
        predictions = probabilities.mean(axis=1).argmax(axis=1)
        np.add.at(confusion, (labels, predictions), 1)
        reason = "evaluation_class_missing" if (counts == 0).any() else ""
        if not reason:
            accuracy = float(np.mean(confusion.diagonal() / counts))
    return dict(balanced_accuracy=accuracy, n_eligible=len(labels), n_predicted=n_predicted,
                class_counts=counts, confusion=confusion, reason=reason)


def sustained_recovery(all_unit_accuracy: float, pc_scores: dict[int, float]) -> int | None:
    """Return the first supported PC count with all supported successors within .05.

    Accuracies are dimensionless fractions. Input keys are positive PC counts;
    unavailable (NaN) values do not participate. Return None if unreached or
    there is no available all-unit reference.
    """
    if not np.isfinite(all_unit_accuracy):
        return None
    recovered = None
    for count, score in sorted(pc_scores.items(), reverse=True):
        if np.isfinite(score):
            if score < all_unit_accuracy - .05:
                break
            recovered = count
    return recovered


def class_table(prepared: PreparedChoices) -> pd.DataFrame:
    """Return ordered dimensionless class IDs and names for each regional target."""
    return pd.DataFrame(
        [(target, code, name) for target in ("context", "choice", "outcome")
         for code, name in enumerate(prepared.targets.class_names[TARGET_NAMES.index(target)])],
        columns=("target", "class_id", "class_name"),
    )


def _source_reason(prepared: PreparedChoices, region: str, n_choices: int, has_counts: bool) -> str:
    """Return a shared input failure reason for one region's unitless choice count."""
    reasons = getattr(prepared, "unavailable", {})
    if region in reasons:
        return reasons[region]
    if not n_choices:
        return reasons.get("regional_window", "no_joint_choice_windows")
    if not has_counts:
        return "joint_regional_counts_unavailable"
    return ""


def _score_fields(score: dict, source_reason: str = "") -> dict:
    """Flatten one score into unitless count and fraction columns for output."""
    summary = {name: score[name] for name in ("balanced_accuracy", "n_eligible", "n_predicted", "reason")}
    summary["reason"] = source_reason or summary["reason"]
    counts = score["class_counts"]
    summary.update({f"count_{code}": counts[code] if code < len(counts) else 0 for code in range(4)})
    return summary


def _summarize_readout(prepared: PreparedChoices, identity: dict, positions: np.ndarray,
                       labels: np.ndarray, eligible: np.ndarray, probabilities: np.ndarray,
                       edges: np.ndarray, source_reason: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize (PC, choice, bin, class) probabilities using relative-second edges.

    Return long score and recovery tables on matched held-out choices, with
    all-unit aggregate and bin scores plus supported aggregate PC variants.
    """
    outcome = identity["target"] == "outcome"
    groups = GROUPS[:1] if outcome else GROUPS
    scores, recoveries = [], []
    center_s = (edges[:-1] + edges[1:]) / 2
    periods = {"full": slice(None)}
    if not outcome:
        middle = len(center_s) // 2
        periods.update(pre=slice(0, middle), post=slice(middle, None))
    for group in groups:
        selected = eligible & prepared.condition_masks[group][positions]
        truth = labels[selected]
        all_units = probabilities[0, selected]
        full = None
        for period, bins in periods.items():
            score = _score_fields(score_choices(all_units[:, bins], truth), source_reason)
            scores.append(dict(**identity, group=group, n_pcs=0, period=period,
                               bin_index=-1, bin_center_s=np.nan, **score))
            if period == "full":
                full = score
        for index, center in enumerate(center_s):
            score = _score_fields(score_choices(all_units[:, index:index + 1], truth), source_reason)
            # Outcome uses the original regional-bin indices also stored with probabilities.
            bin_index = index + len(center_s) if outcome else index
            scores.append(dict(**identity, group=group, n_pcs=0, period="bin",
                               bin_index=bin_index, bin_center_s=center, **score))
        pc_scores = {}
        if np.isfinite(full["balanced_accuracy"]):
            for n_pcs in PCS[1:]:
                score = _score_fields(score_choices(probabilities[n_pcs, selected], truth))
                if not np.isfinite(score["balanced_accuracy"]):
                    continue
                pc_scores[n_pcs] = score["balanced_accuracy"]
                scores.append(dict(**identity, group=group, n_pcs=n_pcs, period="full",
                                   bin_index=-1, bin_center_s=np.nan, **score))
        all_accuracy = full["balanced_accuracy"]
        recovered = sustained_recovery(all_accuracy, pc_scores)
        reason = full["reason"] or ("no_supported_pc_counts" if not pc_scores else "")
        status = "unavailable" if reason else "unreached" if recovered is None else "reached"
        recoveries.append(dict(**identity, group=group, all_unit_accuracy=all_accuracy,
                               recovery_count=recovered, max_supported_pc=max(pc_scores, default=0),
                               requested_max_pc=MAX_REGIONAL_PCS,
                               status=status, reason=reason))
    return pd.DataFrame(scores), pd.DataFrame(recoveries).astype({"recovery_count": float})


def _probability_table(prepared: PreparedChoices, identity: dict, positions: np.ndarray,
                       labels: np.ndarray, probabilities: np.ndarray, edges: np.ndarray,
                       bin_start: int) -> pd.DataFrame:
    """Flatten all-unit probabilities (choice, bin, class), keeping original IDs.

    Edges are seconds relative to choice. Unsupported bins/class columns carry
    NaN, including the two unused class columns of binary readouts.
    """
    bins = len(edges) - 1
    count = len(positions) * bins
    data = {key: np.repeat(value, count) for key, value in identity.items()}
    data.update(
        row_position=np.repeat(positions, bins),
        trial_id=np.repeat(prepared.trials.cur_trial.iloc[positions].to_numpy(), bins),
        fold_id=np.repeat(prepared.fold_ids[positions], bins),
        bin_index=np.tile(np.arange(bin_start, bin_start + bins), len(positions)),
        bin_left_s=np.tile(edges[:-1], len(positions)),
        bin_right_s=np.tile(edges[1:], len(positions)),
        true_class=np.repeat(labels, bins),
    )
    for code, name in enumerate(PROBABILITY_COLUMNS):
        data[name] = probabilities[..., code].ravel() if code < probabilities.shape[-1] else np.full(count, np.nan)
    return pd.DataFrame(data)


def _outcome_tables(identity: dict, labels: np.ndarray, probabilities: np.ndarray,
                    source_reason: str) -> dict[str, pd.DataFrame]:
    """Return outcome confusion and conditional reward tables from (choice, bin, 4)."""
    header = {key: identity[key] for key in ("region", "width_ms")}
    score = score_choices(probabilities, labels)
    confusion = []
    for truth, row in enumerate(score["confusion"]):
        count = int(score["class_counts"][truth])
        reason = source_reason
        if not reason and score["n_predicted"] != score["n_eligible"]:
            reason = "missing_predictions"
        if not reason and not count:
            reason = "evaluation_class_missing"
        confusion.extend(dict(
            **header, true_class=truth, predicted_class=prediction, count=int(value),
            row_count=count, fraction=np.nan if reason else value / count, reason=reason,
        ) for prediction, value in enumerate(row))
    discrimination = []
    for side, rewarded, unrewarded in (("left", 0, 2), ("right", 1, 3)):
        included = np.isin(labels, [rewarded, unrewarded])
        compared = probabilities[included][..., [unrewarded, rewarded]]
        score = score_choices(compared, (labels[included] == rewarded).astype(int))
        discrimination.append(dict(
            **header, side=side, balanced_accuracy=score["balanced_accuracy"],
            n_rewarded=score["class_counts"][1], n_unrewarded=score["class_counts"][0],
            n_predicted=score["n_predicted"], reason=source_reason or score["reason"],
        ))
    return {"confusion": pd.DataFrame(confusion), "reward_discrimination": pd.DataFrame(discrimination)}


def _transfer_mapping(prepared: PreparedChoices, region: str, width_ms: int) -> pd.DataFrame:
    """Return per-fold context-by-choice counts on correct-rewarded training trials.

    Counts use only joint known labels. A same/reversed mapping also requires
    both classes and identical training identities for the two label sets.
    """
    window = prepared.windows[REGIONAL_WINDOW_S, width_ms]
    values = prepared.targets.values[:, :2]
    valid = prepared.targets.valid[:, :2]
    correct_rewarded = prepared.condition_masks["correct_rewarded"]
    rows = []
    for split in window.splits:
        train = window.row_positions[split.train_indices]
        train = train[correct_rewarded[train]]
        joint = train[valid[train].all(axis=1)]
        counts = np.zeros((2, 2), dtype=int)
        np.add.at(counts, tuple(values[joint].T), 1)
        mapping = "unavailable"
        if (np.array_equal(valid[train, 0], valid[train, 1])
                and (counts.sum(axis=0) > 0).all() and (counts.sum(axis=1) > 0).all()):
            mapping = "distinct"
            if counts[0, 1] == counts[1, 0] == 0:
                mapping = "same"
            elif counts[0, 0] == counts[1, 1] == 0:
                mapping = "reversed"
        rows.append(dict(region=region, width_ms=width_ms, fold_id=split.fold_id, mapping=mapping,
                         **{f"count_{a}{b}": counts[a, b] for a in range(2) for b in range(2)}))
    return pd.DataFrame(rows)


def _delivery_latency(prepared: PreparedChoices) -> pd.DataFrame:
    """Summarize ordinary rewarded trial-table choice-to-delivery seconds, if recorded."""
    trials = prepared.trials
    rewarded = np.isin(prepared.targets.values[:, 2], (0, 1)) & prepared.condition_masks["overall"]
    latencies = np.empty(0, dtype=float)
    if "reward_time" in trials and "choice_time" in trials:
        elapsed = numeric_column(trials.reward_time) - numeric_column(trials.choice_time)
        latencies = elapsed[rewarded & np.isfinite(elapsed) & (elapsed >= 0)]
    return pd.DataFrame([dict(n_rewarded=int(rewarded.sum()), n_recorded=len(latencies),
                              n_under_100ms=int(np.count_nonzero(latencies < .1)),
                              median_latency_s=float(np.median(latencies)) if len(latencies) else np.nan)])
