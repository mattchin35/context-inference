"""Pooled OOF scoring, class support, and side-conditional reward summaries."""

import numpy as np
from sklearn.metrics import average_precision_score, log_loss

from src.neural_analysis.population_decoding.scoring import score_predictions, conditional_outcome_scores


def test_missing_test_class_keeps_logloss_but_not_balanced_accuracy():
    """No inputs; binary test class absence changes metrics, not prediction validity."""
    y = np.zeros(3, dtype=int)
    p = np.array([[.8, .2], [.7, .3], [.9, .1]])
    result = score_predictions(y, p, (0, 1))
    assert result["log_loss"] == log_loss(y, p, labels=[0, 1])
    assert result["balanced_accuracy"] is None
    assert result["metric_reasons"]["balanced_accuracy"]
    assert result["confusion"] == [[3, 0], [0, 0]]


def test_incomplete_required_oof_disables_primary_metrics():
    """No inputs; missing predictions remain visible and cannot count as a complete score."""
    p = np.array([[.9, .1], [np.nan, np.nan], [.1, .9]])
    result = score_predictions(np.array([0, 0, 1]), p, (0, 1))
    assert result["log_loss"] is None and result["balanced_accuracy"] is None
    assert result["predicted_rows"] == 2 and result["required_rows"] == 3


def test_pooled_accuracy_and_average_precision_definition():
    """No inputs; pooled scores use all rows and average precision's stepwise definition."""
    y = np.array([0, 1, 0, 1, 1, 0, 0])
    positive = np.array([.1, .9, .2, .8, .4, .7, .3])
    result = score_predictions(y, np.column_stack([1-positive, positive]), (0, 1))
    assert np.isclose(result["accuracy"], 5 / 7)
    assert np.isclose(result["average_precision"][1], average_precision_score(y, positive))
    assert result["class_counts"] == [4, 3]
    assert result["always_negative_accuracy"] == 4 / 7


def test_outcome_summary_cannot_mistake_side_for_reward_information():
    """No inputs; perfect side-only four-class signals remain chance for within-side reward."""
    y = np.tile([0, 1, 2, 3], 10)
    p = np.array([[.5, 0, .5, 0] if value in (0, 2) else [0, .5, 0, .5] for value in y])
    result = conditional_outcome_scores(y, p)
    assert result["left"]["roc_auc"][1] == .5
    assert result["right"]["roc_auc"][1] == .5
    assert result["left"]["class_counts"] == [10, 10]
