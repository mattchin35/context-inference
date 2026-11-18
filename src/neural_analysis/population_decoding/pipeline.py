"""Explicit population OOF models and matched switch-scope comparisons."""

from __future__ import annotations

import numpy as np
from copy import deepcopy
from src.neural_analysis.task_decoding.pooled.decoding import stack_training_window

from .configuration import REPRESENTATIONS
from .features import fit_unit_scaling, transform_unit_activity, fit_fold_pca, transform_pc_activity
from .modeling import LOGISTIC_PARAMETERS, fit_classifier, predict_probabilities
from .preparation import PreparedPopulation
from .scoring import score_predictions, conditional_outcome_scores
from .targets import MODEL_TARGET_NAMES, CONDITIONAL_SWITCH_TARGETS, build_model_target


def fit_target(prepared: PreparedPopulation, region_name: str, target_name: str) -> dict:
    """Fit one region/target's two representations across fixed held-out block folds.

    Input contains native Hz(W,B,U), complete targets(N,5) and fold IDs(N,).
    Return named arrays: original row_positions(W_target,), bin_indices(B_scope,),
    labels(W_target,), probabilities/log probabilities(W_target,B_scope,C) per
    representation, and fold-specific numeric parameters. meta carries ordered
    classes, pooled/per-fold metrics, trial counts, training masks and reasons.
    Inputs are unchanged. Missing required OOF rows remain NaN.
    """
    if target_name not in MODEL_TARGET_NAMES or region_name not in ("PFC", "HPC"):
        raise ValueError("Unknown population region or target")
    model_target = build_model_target(target_name, prepared.targets)
    classes = tuple(range(len(model_target.class_names)))
    region = prepared.regions.get(region_name)
    width = prepared.meta["bin_width_ms"] / 1000
    edges = region.bin_edges_s if region is not None else np.linspace(-1, 1, round(2 / width) + 1)
    bins = np.flatnonzero(edges[:-1] >= 0) if target_name == "outcome" else np.arange(len(edges) - 1)
    positions = np.empty(0, int)
    if region is not None:
        rows = region.row_positions
        positions = np.flatnonzero(prepared.baseline[rows] & model_target.valid[rows])
        rows = rows[positions]
    else:
        rows = np.empty(0, int)
    labels = model_target.labels[rows]
    folds = prepared.fold_ids[rows]
    shape = (len(rows), len(bins), len(classes))
    cell = {"row_positions": rows, "bin_indices": bins, "labels": labels,
            "fold_ids": folds, "class_codes": np.asarray(classes),
            "meta": {"region": region_name, "target": target_name, "class_names": list(model_target.class_names),
                "source_target": model_target.source_target, "conditioning_side": model_target.conditioning_side,
                "supervised_interval_s": [0 if target_name == "outcome" else -1, 1],
                "bin_width_ms": prepared.meta["bin_width_ms"],
                "estimator": {**LOGISTIC_PARAMETERS, "class_name": "sklearn.linear_model.LogisticRegression",
                    "family": "multinomial" if len(classes) == 4 else "binary",
                    "sample_weight": "1 / supervised_bins_per_trial"},
                "preprocessing": {"unit_scaling": "training-only Hz mean and population SD", "ddof": 0,
                    "pca_solver": "full", "whiten": False, "pc_score_scaling": False,
                    "rank_rule": "max(training_observations,variable_units)*float64_epsilon*largest_singular_value"},
                "representations": {}, "unit_ids": list(region.unit_ids) if region else [],
                "unavailable_reason": prepared.unavailable.get(region_name, "")}}
    for representation in REPRESENTATIONS:
        cell[f"{representation}_probabilities"] = np.full(shape, np.nan)
        cell[f"{representation}_log_probabilities"] = np.full(shape, np.nan)
        cell["meta"]["representations"][representation] = {"folds": []}
    if region is not None:
        activity = region.rates_hz[positions][:, bins, :]
        for fold in range(5):
            _fit_fold(cell, activity, region.unit_ids, rows, labels, folds, fold, classes)
    flat_labels = np.repeat(labels, len(bins))
    for representation in REPRESENTATIONS:
        summary = cell["meta"]["representations"][representation]
        probability = cell[f"{representation}_probabilities"].reshape(-1, len(classes))
        logp = cell[f"{representation}_log_probabilities"].reshape(-1, len(classes))
        summary["pooled"] = score_predictions(flat_labels, probability, classes,
            logp if np.isfinite(logp).all() else None)
        summary["trial_class_counts"] = np.bincount(labels, minlength=len(classes)).tolist()
        summary["eligible_trials"] = len(rows)
        if target_name == "outcome":
            summary["conditional_outcomes"] = conditional_outcome_scores(flat_labels, probability, logp)
    if not any(np.isfinite(cell[f"{name}_probabilities"]).any() for name in REPRESENTATIONS):
        cell["meta"]["unavailable_reason"] = cell["meta"]["unavailable_reason"] or "No available held-out fits"
    return cell


def build_switch_comparison(current_choices: np.ndarray, reference_cell: dict, new_cell: dict) -> dict:
    """Align full-row current choices(N,) and paired W,B,2 OOF cells; return comparison.

    Returns reference_window_indices(W_side,) plus meta containing three score
    scopes per representation and matched-fold diagnostics. Both cells must use
    identical bin geometry, region, and required same-side trial/fold/label maps.
    Missing predictions remain required; no finite-row intersection, renormalizing,
    refitting or input mutation occurs. Scores/probabilities are dimensionless.
    The runner checks run identities before calling this pure function.
    """
    reference, new = reference_cell, new_cell
    target = new["meta"]["target"]
    if target not in CONDITIONAL_SWITCH_TARGETS:
        raise ValueError("Comparison requires a conditional switch target")
    side = "left" if target == "left_switch_vs_stay" else "right"
    reference_target = f"{side}_choice_switch"
    if (reference["meta"]["target"] != reference_target or new["meta"].get("conditioning_side") != side
            or reference["meta"]["region"] != new["meta"]["region"]
            or new["meta"]["class_names"] != ["stay", "switch"]
            or reference["meta"]["bin_width_ms"] != new["meta"]["bin_width_ms"]
            or reference["meta"]["supervised_interval_s"] != new["meta"]["supervised_interval_s"]):
        raise ValueError("Incompatible switch model scopes")
    choices = np.asarray(current_choices)
    rows = np.asarray(reference["row_positions"])
    if (choices.ndim != 1 or rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer)
            or np.any((rows < 0) | (rows >= len(choices))) or np.any(np.diff(rows) <= 0)):
        raise ValueError("Invalid reference row positions")
    indices = np.flatnonzero(choices[rows] == (1 if side == "left" else 0))
    if (not np.array_equal(rows[indices], new["row_positions"])
            or not np.array_equal(reference["bin_indices"], new["bin_indices"])
            or not np.array_equal(reference["fold_ids"][indices], new["fold_ids"])
            or not np.array_equal(reference["labels"][indices], new["labels"])):
        raise ValueError("Comparison requires exactly matching rows, bins, folds and labels")
    bins = len(new["bin_indices"])
    expected_shape = (len(new["labels"]), bins, 2)
    if bins == 0:
        raise ValueError("Comparison requires supervised bins")
    meta = {"reference_target": reference_target, "conditioning_side": side,
            "class_names": ["stay", "switch"], "required_trials": len(indices),
            "difference_definition": "within_side minus reference_matched_side; lower log loss is better",
            "representations": {}}
    for rep in REPRESENTATIONS:
        if (new[f"{rep}_probabilities"].shape != expected_shape
                or new[f"{rep}_log_probabilities"].shape != expected_shape
                or reference[f"{rep}_probabilities"].shape != (len(rows), bins, 2)
                or reference[f"{rep}_log_probabilities"].shape != (len(rows), bins, 2)):
            raise ValueError("Comparison probability axes do not match row/bin maps")
        matched = _score_window_rows(reference, rep, indices)
        within = _score_window_rows(new, rep, np.arange(len(indices)))
        record = {"reference_all_trials": {"trial_count": len(rows),
                      "scores": deepcopy(reference["meta"]["representations"][rep]["pooled"])},
                  "reference_matched_side": {"trial_count": len(indices), "scores": matched},
                  "within_side": {"trial_count": len(indices), "scores": within},
                  "differences": _metric_differences(matched, within), "folds": []}
        for fold in range(5):
            selected = np.flatnonzero(new["fold_ids"] == fold)
            a = _score_window_rows(reference, rep, indices[selected])
            b = _score_window_rows(new, rep, selected)
            record["folds"].append({"fold": fold, "trial_count": len(selected),
                "reference_matched_side": a, "within_side": b, "differences": _metric_differences(a, b)})
        meta["representations"][rep] = record
    return {"reference_window_indices": indices, "meta": meta}


def _score_window_rows(cell: dict, representation: str, window_indices: np.ndarray) -> dict:
    """Score selected integer window indices(W,) from saved W,B,2 arrays; return metrics.

    Labels are repeated in trial-major/bin-major order. NaN OOF values remain
    present, making required-scope metrics unavailable rather than dropping rows.
    """
    bins = len(cell["bin_indices"])
    p = cell[f"{representation}_probabilities"][window_indices].reshape(-1, 2)
    lp = cell[f"{representation}_log_probabilities"][window_indices].reshape(-1, 2)
    return score_predictions(np.repeat(cell["labels"][window_indices], bins), p, (0, 1),
                             lp if np.isfinite(lp).all() else None)


def _metric_differences(reference: dict, new: dict) -> dict:
    """Return new-minus-reference scalar metric differences, or None when undefined."""
    differences = {}
    for name in ("balanced_accuracy", "log_loss", "accuracy", "roc_auc", "average_precision"):
        a, b = reference[name], new[name]
        if name in ("roc_auc", "average_precision"):
            a, b = a[1], b[1]
        differences[name] = float(b-a) if a is not None and b is not None else None
    return differences


def _fit_fold(cell: dict, activity: np.ndarray, unit_ids: tuple[str, ...], rows: np.ndarray,
              labels: np.ndarray, folds: np.ndarray, fold: int, classes: tuple[int, ...]) -> None:
    """Populate one fold in a new cell from Hz(T,B,U), labels/folds(T,), and original rows.

    Only the cell under construction is mutated. The saved scaler is shared by
    both readouts; training uses known fold IDs other than this held-out fold.
    Array parameters retain physical Hz units and original input unit identity.
    """
    train, test = (folds >= 0) & (folds != fold), folds == fold
    train_counts = np.bincount(labels[train], minlength=len(classes)).tolist()
    test_counts = np.bincount(labels[test], minlength=len(classes)).tolist()
    common = {"fold": fold, "training_trials": int(train.sum()), "test_trials": int(test.sum()),
              "training_class_counts": train_counts, "test_class_counts": test_counts}
    cell[f"fold_{fold}_training_rows"] = rows[train]
    cell[f"fold_{fold}_test_rows"] = rows[test]
    reason = ("empty_test_scope" if not test.any() else
              "training_class_missing" if not min(train_counts, default=0) else "")
    scaler = None
    if not reason:
        try:
            scaler = fit_unit_scaling(activity[train], unit_ids)
            ztrain = transform_unit_activity(activity[train], unit_ids, scaler)
            ztest = transform_unit_activity(activity[test], unit_ids, scaler)
            for name, values in (("mean_hz", scaler.mean_hz), ("scale_hz", scaler.scale_hz),
                                 ("retained_mask", scaler.retained_mask)):
                cell[f"fold_{fold}_{name}"] = values
            cell["meta"][f"fold_{fold}_omitted_unit_reasons"] = {
                unit: "constant_training_activity" for unit, keep in zip(unit_ids, scaler.retained_mask) if not keep}
        except (ValueError, FloatingPointError) as error:
            reason = str(error)
    for representation in REPRESENTATIONS:
        record = {**common, "available": False, "reason": reason}
        if not reason and scaler is not None:
            try:
                train_features, test_features = ztrain, ztest
                if representation == "pcs":
                    pca = fit_fold_pca(ztrain.reshape(-1, ztrain.shape[-1]))
                    train_features = transform_pc_activity(ztrain, pca)
                    test_features = transform_pc_activity(ztest, pca)
                    for name in ("components", "center", "singular_values", "explained_variance", "explained_variance_ratio"):
                        cell[f"fold_{fold}_pca_{name}"] = getattr(pca, name)
                    record["pc_rank"] = len(pca.components)
                    record["rank_tolerance"] = pca.rank_tolerance
                x, y, weights = stack_training_window(train_features, labels[train])
                fitted = fit_classifier(x, y, weights, classes)
                record.update(available=fitted.available, reason=fitted.reason, iterations=fitted.iterations)
                if fitted.available:
                    probability, logp = predict_probabilities(test_features.reshape(-1, test_features.shape[-1]), fitted)
                    cell[f"{representation}_probabilities"][test] = probability.reshape(test.sum(), activity.shape[1], len(classes))
                    cell[f"{representation}_log_probabilities"][test] = logp.reshape(test.sum(), activity.shape[1], len(classes))
                    cell[f"fold_{fold}_{representation}_coefficients"] = fitted.coefficients
                    cell[f"fold_{fold}_{representation}_intercepts"] = fitted.intercepts
                    record["scores"] = score_predictions(np.repeat(labels[test], activity.shape[1]), probability, classes, logp)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                record.update(available=False, reason=f"{representation}_fit_failure: {error}")
        cell["meta"]["representations"][representation]["folds"].append(record)
