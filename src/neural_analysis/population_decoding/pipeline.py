"""Explicit five-target OOF analysis; no global PCA or behavior-derived features."""

from __future__ import annotations

import numpy as np
from src.neural_analysis.task_decoding.pooled.decoding import stack_training_window

from .configuration import REPRESENTATIONS
from .features import fit_unit_scaling, transform_unit_activity, fit_fold_pca, transform_pc_activity
from .modeling import fit_classifier, predict_probabilities
from .preparation import PreparedPopulation
from .scoring import score_predictions, conditional_outcome_scores
from .targets import TARGET_NAMES, CLASS_NAMES


def fit_target(prepared: PreparedPopulation, region_name: str, target_name: str) -> dict:
    """Fit one region/target's two representations across fixed held-out block folds.

    Input contains native Hz(W,B,U), complete targets(N,5) and fold IDs(N,).
    Return named arrays: original row_positions(W_target,), bin_indices(B_scope,),
    labels(W_target,), probabilities/log probabilities(W_target,B_scope,C) per
    representation, and fold-specific numeric parameters. meta carries ordered
    classes, pooled/per-fold metrics, trial counts, training masks and reasons.
    Inputs are unchanged. Missing required OOF rows remain NaN.
    """
    if target_name not in TARGET_NAMES or region_name not in ("PFC", "HPC"):
        raise ValueError("Unknown population region or target")
    target = TARGET_NAMES.index(target_name)
    classes = tuple(range(len(CLASS_NAMES[target])))
    region = prepared.regions.get(region_name)
    width = prepared.meta["bin_width_ms"] / 1000
    edges = region.bin_edges_s if region is not None else np.linspace(-1, 1, round(2 / width) + 1)
    bins = np.flatnonzero(edges[:-1] >= 0) if target_name == "outcome" else np.arange(len(edges) - 1)
    positions = np.empty(0, int)
    if region is not None:
        rows = region.row_positions
        positions = np.flatnonzero(prepared.baseline[rows] & prepared.targets.valid[rows, target])
        rows = rows[positions]
    else:
        rows = np.empty(0, int)
    labels = prepared.targets.values[rows, target]
    folds = prepared.fold_ids[rows]
    shape = (len(rows), len(bins), len(classes))
    cell = {"row_positions": rows, "bin_indices": bins, "labels": labels,
            "fold_ids": folds, "class_codes": np.asarray(classes),
            "meta": {"region": region_name, "target": target_name, "class_names": list(CLASS_NAMES[target]),
                "supervised_interval_s": [0 if target_name == "outcome" else -1, 1],
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
