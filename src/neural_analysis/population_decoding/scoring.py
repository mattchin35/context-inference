"""Complete pooled OOF metrics, class support, and conditional outcome summaries."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support, roc_auc_score)


def score_predictions(labels: np.ndarray, probabilities: np.ndarray,
                      classes: tuple[int, ...], log_probabilities: np.ndarray | None = None) -> dict:
    """Score required labels(M,) and ordered probabilities(M,C); return JSON metrics.

    Missing prediction rows are NaN, never negative targets. Primary balanced
    accuracy/log loss require all required rows predicted; balanced accuracy
    also requires every test class. Per-class discrimination requires positive
    and negative examples. Optional log probabilities preserve extreme-logit
    loss accuracy. Counts describe bin observations, all dimensionless.
    """
    y, p = np.asarray(labels), np.asarray(probabilities, float)
    c = len(classes)
    if y.ndim != 1 or p.shape != (len(y), c) or not np.isin(y, classes).all():
        raise ValueError("Labels(M,) and probabilities(M,C) must match declared classes")
    valid = np.isfinite(p).all(axis=1)
    if np.any((p[valid] < 0) | (p[valid] > 1)) or not np.allclose(p[valid].sum(axis=1), 1):
        raise ValueError("Finite probability rows must be nonnegative and sum to one")
    counts = [int(np.sum(y == value)) for value in classes]
    result = {"required_rows": len(y), "predicted_rows": int(valid.sum()), "class_counts": counts,
              "prevalence": [count / len(y) if len(y) else None for count in counts],
              "accuracy": None, "balanced_accuracy": None, "log_loss": None,
              "confusion": np.zeros((c, c), int).tolist(), "precision": [None] * c,
              "recall": [None] * c, "roc_auc": [None] * c, "average_precision": [None] * c,
              "always_negative_accuracy": counts[0] / len(y) if len(y) and c == 2 else None,
              "metric_reasons": {}}
    if not len(y) or not valid.all():
        reason = "empty_scope" if not len(y) else "incomplete_required_oof_predictions"
        result["metric_reasons"] = {name: reason for name in ("accuracy", "balanced_accuracy", "log_loss", "discrimination")}
        return result
    predictions = ((p[:, 1] >= .5).astype(int) if c == 2 else p.argmax(axis=1))
    cm = confusion_matrix(y, predictions, labels=list(classes))
    precision, recall, _, _ = precision_recall_fscore_support(y, predictions, labels=list(classes), zero_division=0)
    result.update(accuracy=float(accuracy_score(y, predictions)), confusion=cm.tolist(),
                  precision=[float(value) if cm[:, i].sum() else None for i, value in enumerate(precision)],
                  recall=[float(value) if counts[i] else None for i, value in enumerate(recall)])
    if log_probabilities is None:
        result["log_loss"] = float(log_loss(y, p, labels=list(classes)))
    else:
        lp = np.asarray(log_probabilities, float)
        if lp.shape != p.shape or not np.isfinite(lp).all() or not np.allclose(np.exp(lp), p):
            raise ValueError("Log probabilities must match probability rows")
        result["log_loss"] = float(-lp[np.arange(len(y)), y.astype(int)].mean())
    if min(counts) > 0:
        result["balanced_accuracy"] = float(np.mean(recall))
    else:
        result["metric_reasons"]["balanced_accuracy"] = "missing_test_class"
    for i, value in enumerate(classes):
        binary = y == value
        if binary.any() and (~binary).any():
            result["roc_auc"][i] = float(roc_auc_score(binary, p[:, i]))
            result["average_precision"][i] = float(average_precision_score(binary, p[:, i]))
        else:
            result["metric_reasons"][f"class_{value}_discrimination"] = "missing_test_class"
        if not counts[i]:
            result["metric_reasons"][f"class_{value}_recall"] = "missing_test_class"
        if not cm[:, i].sum():
            result["metric_reasons"][f"class_{value}_precision"] = "no_predicted_class"
    return result


def conditional_outcome_scores(labels: np.ndarray, probabilities: np.ndarray,
                               log_probabilities: np.ndarray | None = None) -> dict:
    """Summarize same four-class fit within actual side; return left/right metrics.

    labels(M,) use outcome codes 0..3, probabilities(M,4) use fixed outcome order.
    Conditional binary classes are no_reward=0/reward=1. Supplied log(M,4) values
    allow stable normalization of underflowed probabilities. Undefined rows stay
    unavailable; this function fits no additional classifiers.
    """
    y, p = np.asarray(labels), np.asarray(probabilities, float)
    result = {}
    for side, rewarded, absent in (("left", 0, 2), ("right", 1, 3)):
        selected = np.isin(y, [rewarded, absent])
        pair = p[selected][:, [absent, rewarded]]
        conditional = np.full(pair.shape, np.nan)
        conditional_log = None
        if log_probabilities is not None:
            lp = np.asarray(log_probabilities)[selected][:, [absent, rewarded]]
            finite = np.isfinite(lp).all(axis=1)
            conditional_log = np.full(pair.shape, np.nan)
            conditional_log[finite] = lp[finite] - logsumexp(lp[finite], axis=1, keepdims=True)
            conditional[finite] = np.exp(conditional_log[finite])
        else:
            denominator = pair.sum(axis=1)
            good = np.isfinite(pair).all(axis=1) & (denominator > 0)
            conditional[good] = pair[good] / denominator[good, None]
        result[side] = score_predictions((y[selected] == rewarded).astype(int), conditional,
                                         (0, 1), conditional_log if np.isfinite(conditional).all() else None)
    return result
