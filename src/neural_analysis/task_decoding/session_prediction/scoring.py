"""Pure trial-mean performance summaries from saved readouts; never fit or load files."""

import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.neural_analysis.task_decoding.modeling import regression_r2

from .config import FOLD_COUNT
from .modeling import RegionPredictions
from .preparation import PredictionInputs


SCORING_VERSION = "trial-mean-performance-v1"
SCORING_POPULATIONS = ("all", "training_outcomes", "correct_rewarded", "omission", "incorrect")


def trial_metrics(actual: np.ndarray, predictions: np.ndarray, family: str) -> dict:
    """Score aligned finite float vectors (trial,) with equal weight per trial.

    Categorical values are 0=right/1=left and predictions are left probabilities;
    threshold the trial-mean probability at >=0.5. Return n_trials, accuracy and
    balanced_accuracy (fractions). Numerical values/predictions use native units;
    return n_trials, dimensionless r2 and native-unit rmse. Empty/undefined
    metrics are None; constant/singleton targets still have an RMSE. Raise on
    malformed vectors. No input changes or bin-level independence assumptions.
    """
    actual, predictions = np.asarray(actual), np.asarray(predictions)
    if (actual.ndim != 1 or predictions.shape != actual.shape
            or not np.isfinite(actual).all() or not np.isfinite(predictions).all()):
        raise ValueError("Metrics require aligned finite trial vectors")
    count = len(actual)
    if family == "categorical":
        if not np.isin(actual, [0, 1]).all() or np.any((predictions < 0) | (predictions > 1)):
            raise ValueError("Binary metrics require 0/1 labels and probabilities in [0,1]")
        classified = predictions >= 0.5
        return {"n_trials": count,
                "accuracy": float(accuracy_score(actual, classified)) if count else None,
                "balanced_accuracy": (float(balanced_accuracy_score(actual, classified))
                                      if np.unique(actual).size == 2 else None)}
    if family == "numerical":
        return {"n_trials": count, "r2": regression_r2(actual, predictions),
                "rmse": float(np.sqrt(np.mean((actual - predictions) ** 2))) if count else None}
    raise ValueError("Target family must be categorical or numerical")


def compute_prediction_performance(
    inputs: PredictionInputs, regions: dict[str, RegionPredictions], target_name: str,
) -> dict[str, list[dict]]:
    """Return small selected-target performance records for validated saved inputs/models.

    Inputs retain N-choice targets (N,T), regional scores (W,B,K), coefficients
    (T,5,K), and saved predictions (W,B,T). Average only [-1,+1) bins; sigmoid
    precedes averaging for binary models. Each held-out trial occurs once per
    region/population. Training sets overlap, so report each fit separately.

    Returned lists: folds (training/matching held-out), held_out (pooled by
    population), laterality (choice/state, left/right), gaps (left minus right).
    Rows retain region/population/counts, and fractions or native-unit metrics
    from trial_metrics. Missing side labels affect only that grouping. Scores
    span the full saved session, independent of viewport/PC display settings.
    No source files, models, or arrays are changed or fitted.
    """
    target = inputs.target_names.index(target_name)
    family = inputs.target_families[target]
    report = {name: [] for name in ("folds", "held_out", "laterality", "gaps")}
    for role, model in regions.items():
        source = inputs.timeline.regions[role]
        positions = source.choice_positions
        actual = inputs.target_values[positions, target]
        folds = inputs.fold_ids[positions]
        baseline = inputs.baseline_valid[positions] & inputs.target_valid[positions, target]
        assigned = folds >= 0
        fitted = np.zeros(len(positions), dtype=bool)
        fitted[assigned] = model.fit_status[target, folds[assigned]] == "ok"
        bin_predictions = model.predictions[:, inputs.training_bin_mask, target]
        predictions = bin_predictions.mean(axis=1)
        eligible = (baseline & assigned & fitted & (model.prediction_reason[:, target] == "")
                    & np.isfinite(bin_predictions).all(axis=1))
        populations = {"all": np.ones(len(positions), dtype=bool),
                       "training_outcomes": inputs.training_condition_mask[positions]}
        populations.update({name: inputs.condition_masks[positions, inputs.condition_names.index(name)]
                            for name in SCORING_POPULATIONS[2:]})
        for fold in range(FOLD_COUNT):
            train = baseline & model.training_choice_mask[target, fold, positions]
            if model.fit_status[target, fold] == "ok":
                predicted = _training_trial_means(source.scores, np.flatnonzero(train), inputs.training_bin_mask,
                                                  model.coefficients[target, fold], model.intercepts[target, fold], family)
                finite = np.isfinite(predicted)
                metrics = trial_metrics(actual[train][finite], predicted[finite], family)
            else:
                metrics = trial_metrics(np.array([]), np.array([]), family)
            common = {"region": role, "fold": fold, "population": "training_outcomes",
                      "fit_status": str(model.fit_status[target, fold]), "fit_reason": str(model.fit_reason[target, fold])}
            report["folds"].append({**common, "split": "training", **metrics})
            test = eligible & (folds == fold) & populations["training_outcomes"]
            report["folds"].append({**common, "split": "held_out", **trial_metrics(actual[test], predictions[test], family)})
        for population, members in populations.items():
            selected = eligible & members
            common = {"region": role, "population": population}
            report["held_out"].append({**common, **trial_metrics(actual[selected], predictions[selected], family)})
            for grouping, side_target in (("choice", "current_action"), ("state", "current_state")):
                side_index = inputs.target_names.index(side_target)
                side_values = inputs.target_values[positions, side_index]
                side_valid = inputs.target_valid[positions, side_index]
                sides = {}
                for side, value in (("left", 1), ("right", 0)):
                    mask = selected & side_valid & (side_values == value)
                    sides[side] = trial_metrics(actual[mask], predictions[mask], family)
                    report["laterality"].append({**common, "grouping": grouping, "side": side, **sides[side]})
                differences = {
                    name: (sides["left"][name] - sides["right"][name]
                           if sides["left"][name] is not None and sides["right"][name] is not None else None)
                    for name in sides["left"] if name != "n_trials"
                }
                report["gaps"].append({**common, "grouping": grouping,
                                       "left_n_trials": sides["left"]["n_trials"],
                                       "right_n_trials": sides["right"]["n_trials"], **differences})
    return report


# Training reconstruction keeps temporary storage to one bin's windows/PCs.
def _training_trial_means(
    scores: np.ndarray, windows: np.ndarray, bins: np.ndarray,
    coefficients: np.ndarray, intercept: float, family: str,
) -> np.ndarray:
    """Return (selected W,) trial means from W,B,K scores in saved PC/native units.

    windows is ordered integer (selected W,), bins bool (B,), beta float (K,)
    and intercept native/logit units. Average sigmoid outputs for categorical
    families, affine outputs otherwise. Nonfinite bins invalidate their trial.
    No full selected-window/bin/PC tensor is allocated or input changed.
    """
    total = np.zeros(len(windows))
    for bin_index in np.flatnonzero(bins):
        with np.errstate(over="ignore", invalid="ignore"):
            linear = scores[windows, bin_index, :] @ coefficients + intercept
            values = expit(linear) if family == "categorical" else linear
        values[~np.isfinite(linear)] = np.nan
        total += values
    return total / bins.sum()
