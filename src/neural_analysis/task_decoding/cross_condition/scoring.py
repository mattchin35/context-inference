"""Saved-only, equal-trial performance records for each directed transfer."""

import numpy as np

from src.neural_analysis.task_decoding.session_prediction.scoring import trial_metrics

from .config import WEIGHTING_NAMES
from .modeling import RegionTransferResult
from .preparation import TransferInputs


SCORING_VERSION = "cross-condition-summary-v1"


def summarize_transfers(
    inputs: TransferInputs,
    regions: dict[str, RegionTransferResult],
    unavailable_regions: dict[str, str],
) -> list[dict]:
    """Return separate H/G/S/region/role/outcome summary dicts from saved products.

    inputs retains dimensionless targets(N,4) and canonical condition masks(N,C).
    Each regional result has its own W->N map and saved mean probabilities
    (H,G,S,W). unavailable_regions maps missing regional names to reasons; its
    coverage is unknown. No neural scores, raw files, fitting or reweighting are
    needed. Return rows with target_name, split_id, window, weighting, region,
    role, population; eligible_trials before finite-output filtering; scored
    n_trials/class_0_trials/class_1_trials; accuracy/balanced_accuracy fractions
    or None; and reason. Unknown regional eligible_trials is None, scored count
    zero. Single-class reason leaves ordinary accuracy available. Training means
    are in-sample diagnostics; complementary_test stays distinct. Inputs are
    unchanged, overlapping populations/directions are never aggregated.
    """
    if set(regions) & set(unavailable_regions):
        raise ValueError("A region cannot be both available and unavailable")
    correct_rewarded = inputs.condition_masks[:, inputs.condition_names.index("correct_rewarded")]
    omission = inputs.condition_masks[:, inputs.condition_names.index("omission")]
    incorrect = inputs.condition_masks[:, inputs.condition_names.index("incorrect")]
    reward = inputs.target_values[:, inputs.target_names.index("current_reward")]
    rows = []
    for region in (*regions, *unavailable_regions):
        model = regions.get(region)
        for split, definition in enumerate(inputs.transfers):
            target = inputs.target_names.index(definition.target_name)
            populations = _population_masks(definition.target_name, correct_rewarded, omission, incorrect, reward)
            for window, window_name in enumerate(inputs.window_names):
                for weighting, weighting_name in enumerate(WEIGHTING_NAMES):
                    for role in ("training", "complementary_test"):
                        for population, population_mask in populations.items():
                            identity = {
                                "target_name": definition.target_name, "split_id": definition.split_id,
                                "window": window_name, "weighting": weighting_name, "region": region,
                                "role": role, "population": population,
                            }
                            if model is None:
                                metrics = _unavailable_metrics(None, unavailable_regions[region])
                            else:
                                positions = model.choice_positions
                                role_mask = model.source_mask[split] if role == "training" else model.test_mask[split]
                                eligible = role_mask & population_mask[positions]
                                reason = (str(model.fit_reason[window, weighting, split])
                                          if model.fit_status[window, weighting, split] != "ok" else "")
                                metrics = _population_metrics(
                                    inputs.target_values[positions, target],
                                    model.trial_probabilities[window, weighting, split], eligible, reason,
                                )
                            rows.append({**identity, **metrics})
    return rows


def _population_masks(
    target_name: str, correct_rewarded: np.ndarray, omission: np.ndarray,
    incorrect: np.ndarray, reward: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return ordered outcome bool masks(N,) from canonical masks and actual rewards(N,).

    target_name identifies one dimensionless binary target. Inputs are bool(N,)
    canonical memberships and float(N,) actual reward 0/1/NaN. Correct combines
    rewarded/omission; actual_rewarded uses reward 1, independent of correctness.
    Masks are role-independent; exact role/coverage is intersected afterward.
    """
    all_trials = np.ones(len(reward), dtype=bool)
    if target_name == "current_reward":
        incorrect_unrewarded = incorrect & (reward == 0)
        return {"all": all_trials, "actual_rewarded": reward == 1,
                "actual_unrewarded": omission | incorrect_unrewarded,
                "omission": omission, "incorrect_unrewarded": incorrect_unrewarded}
    return {"all": all_trials, "correct": correct_rewarded | omission,
            "correct_rewarded": correct_rewarded, "omission": omission, "incorrect": incorrect}


def _population_metrics(
    actual: np.ndarray, probabilities: np.ndarray, eligible: np.ndarray, fit_reason: str,
) -> dict:
    """Return trial metrics/counts from aligned float actual/probability and bool eligibility(W,).

    Labels/probabilities and accuracy fractions are dimensionless. fit_reason is
    empty for available models. Score finite trial means with equal trial weight,
    threshold >=.5, retaining eligible count separately. Failed fits score zero
    trials with their reason. Single-class accuracy survives undefined balanced
    accuracy; empty groups remain unavailable. No arrays are changed.
    """
    eligible_count = int(eligible.sum())
    if fit_reason:
        return _unavailable_metrics(eligible_count, fit_reason)
    selected = eligible & np.isfinite(probabilities) & np.isfinite(actual)
    labels = actual[selected]
    metrics = trial_metrics(labels, probabilities[selected], "categorical")
    reason = "empty_population" if not len(labels) else "single_class" if np.unique(labels).size < 2 else ""
    return {"eligible_trials": eligible_count, **metrics,
            "class_0_trials": int((labels == 0).sum()), "class_1_trials": int((labels == 1).sum()),
            "reason": reason}


def _unavailable_metrics(eligible_trials: int | None, reason: str) -> dict:
    """Return zero scored counts/None metrics, retaining known eligible count or unknown None.

    eligible_trials is an integer trial count or None for unknown regional
    coverage; reason is the saved failure string. Counts are dimensionless and
    class counts describe scored labels, not unknown eligible class coverage.
    """
    return {"eligible_trials": eligible_trials, "n_trials": 0, "class_0_trials": 0, "class_1_trials": 0,
            "accuracy": None, "balanced_accuracy": None, "reason": reason}
