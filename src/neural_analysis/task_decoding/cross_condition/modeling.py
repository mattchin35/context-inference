"""Fixed-penalty all-source readouts on independent regional session-PC scores."""

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from src.neural_analysis.task_decoding.pooled.decoding import stack_training_window
from src.neural_analysis.task_decoding.session_prediction.modeling import fit_readout

from .config import TARGET_NAMES, TRANSFERS, WEIGHTING_NAMES, WINDOW_NAMES
from .preparation import TransferInputs


@dataclass(frozen=True)
class RegionTransferResult:
    """One region's parameters, eligible memberships and matched-window trial means.

    W=regional choices, K=original PCs, H=3 windows, G=2 weightings, S=18 splits.
    choice_positions is integer (W,), indexing shared behavioral choices;
    pc_ids is integer (K,), original one-based ranks. coefficients is float64
    (H,G,S,K) in log-odds per original PC-score unit; intercepts is float64
    (H,G,S) in log-odds. fit_status/fit_reason are strings (H,G,S).
    trial_probabilities is float64 (H,G,S,W), dimensionless mean P(class 1)
    over only the selected supervised bins. source_mask/test_mask are bool
    (S,W); source_trial_weights is float64 (G,S,W), dimensionless total trial
    weights, zero off source. eligible_cell_counts is integer (S,2,2), role
    source/test then actual class 0/1. Unavailable fits retain NaN parameters
    and probabilities with reasons, without erasing weights or eligible counts.
    No neural scores, estimators, per-bin outputs or fold axes are retained.
    """

    choice_positions: np.ndarray
    pc_ids: np.ndarray
    coefficients: np.ndarray
    intercepts: np.ndarray
    fit_status: np.ndarray
    fit_reason: np.ndarray
    trial_probabilities: np.ndarray
    source_mask: np.ndarray
    test_mask: np.ndarray
    source_trial_weights: np.ndarray
    eligible_cell_counts: np.ndarray


def training_trial_weights(source_cells: np.ndarray, weighting: str) -> np.ndarray:
    """Return float64 dimensionless trial weights(W,) for disjoint bool source cells(W,2).

    The last cell axis is actual class 0/1. equal_trial assigns each source
    trial total weight 1; equal_cell assigns N/(2*nk) in source cell k after
    eligibility/coverage. Both sum to N when both cells exist. Missing either
    cell gives all-zero equal_cell weights; equal_trial still describes present
    trials. Weights are zero off source. Raise ValueError for malformed masks,
    overlapping cells or unknown schemes. Inputs are unchanged.
    """
    cells = np.asarray(source_cells)
    if cells.ndim != 2 or cells.shape[1] != 2 or cells.dtype != bool:
        raise ValueError("Source cells must be Boolean with shape (W,2)")
    if np.any(cells[:, 0] & cells[:, 1]):
        raise ValueError("Source class cells must be disjoint")
    if weighting not in WEIGHTING_NAMES:
        raise ValueError(f"Unknown weighting scheme: {weighting}")
    counts = cells.sum(axis=0)
    weights = np.zeros(len(cells), dtype=np.float64)
    if weighting == "equal_trial":
        weights[cells.any(axis=1)] = 1.
    elif np.all(counts > 0):
        for label in (0, 1):
            weights[cells[:, label]] = counts.sum() / (2 * counts[label])
    return weights


def fit_region_transfers(
    scores: np.ndarray, choice_positions: np.ndarray, inputs: TransferInputs,
) -> RegionTransferResult:
    """Fit H/G/S binary readouts sequentially and retain only matched-window trial means.

    scores is finite float (W,B,K) in saved unwhitened PC units; integer
    choice_positions(W,) is ordered/unique and indexes N shared choices in
    prepared inputs. Inputs retain float/bool labels(N,4), source/test cells
    (N,S,2), common masks(N,) and supervised bin masks(H,B). Return
    RegionTransferResult with original PC identities/units and no score copies.
    Intersect common/label/coverage eligibility before normalizing trial weights;
    divide each trial's total weight across selected bins using the shared
    stacking helper. A numerical failure affects only that H/G/S model. Missing
    classes never borrow test trials. Raise on structural errors before fitting;
    input arrays stay unchanged. Unavailable regions are handled by the caller.
    """
    values, positions = np.asarray(scores, dtype=np.float64), np.asarray(choice_positions)
    _validate_region_inputs(values, positions, inputs)
    source_cells, test_cells = _regional_cells(inputs, positions)
    source_mask, test_mask = source_cells.any(axis=2), test_cells.any(axis=2)
    counts = np.stack([source_cells.sum(axis=1), test_cells.sum(axis=1)], axis=1)
    shape = (len(WINDOW_NAMES), len(WEIGHTING_NAMES), len(TRANSFERS))
    coefficients = np.full((*shape, values.shape[2]), np.nan)
    intercepts = np.full(shape, np.nan)
    statuses = np.full(shape, "unavailable", dtype="U11")
    reasons = np.full(shape, "empty_training", dtype="U160")
    probabilities = np.full((*shape, len(positions)), np.nan)
    weights = np.zeros((len(WEIGHTING_NAMES), len(TRANSFERS), len(positions)))
    for split, definition in enumerate(inputs.transfers):
        target = inputs.target_names.index(definition.target_name)
        actual = inputs.target_values[positions, target]
        selected = source_mask[split] | test_mask[split]
        for weighting, name in enumerate(WEIGHTING_NAMES):
            weights[weighting, split] = training_trial_weights(source_cells[split], name)
        if not np.all(counts[split, 0] > 0):
            reasons[:, :, split] = "single_class_training" if counts[split, 0].sum() else "empty_training"
            continue
        for window, bins in enumerate(inputs.window_bin_masks):
            for weighting in range(len(WEIGHTING_NAMES)):
                beta, intercept, reason = _fit_selected_rows(
                    values, actual, source_mask[split], bins, weights[weighting, split],
                )
                if reason:
                    reasons[window, weighting, split] = reason
                    continue
                trial_means, reason = _trial_mean_probabilities(values, selected, bins, beta, intercept)
                if reason:
                    reasons[window, weighting, split] = reason
                    continue
                coefficients[window, weighting, split] = beta
                intercepts[window, weighting, split] = intercept
                probabilities[window, weighting, split] = trial_means
                statuses[window, weighting, split] = "ok"
                reasons[window, weighting, split] = ""
    return RegionTransferResult(
        positions.copy(), np.arange(1, values.shape[2] + 1, dtype=np.int64),
        coefficients, intercepts, statuses, reasons, probabilities,
        source_mask, test_mask, weights, counts,
    )


def _fit_selected_rows(
    scores: np.ndarray, actual: np.ndarray, source: np.ndarray,
    bins: np.ndarray, trial_weights: np.ndarray,
) -> tuple[np.ndarray | None, float, str]:
    """Fit one selected design; return beta(K,), logit intercept and failure reason.

    scores is finite float(W,B,K) in PC units; actual is binary float(W,)
    with missing values allowed off source,
    source/bins bool(W,)/(B,), trial_weights float(W,) dimensionless. Stack only
    selected trials/bins in trial-major/bin-major order; row weights sum to the
    supplied total for each source trial. The temporary design dies on return.
    Unavailable fits follow fit_readout's (None,NaN,reason) convention.
    """
    tensor = scores[source][:, bins, :]
    matrix, labels, row_weights = stack_training_window(tensor, actual[source])
    row_weights *= np.repeat(trial_weights[source], int(bins.sum()))
    return fit_readout(matrix, labels, row_weights, "categorical")


def _trial_mean_probabilities(
    scores: np.ndarray, selected: np.ndarray, bins: np.ndarray,
    coefficients: np.ndarray, intercept: float,
) -> tuple[np.ndarray, str]:
    """Return float probability means(W,) and reason from original scores(W,B,K).

    selected/bins are bool(W,)/(B,), coefficients float(K,) log-odds/PC unit
    and intercept scalar log-odds. Apply expit before averaging selected bins,
    retaining NaN off selected trials. Compute one bin at a time. A nonfinite
    affine value or probability anywhere selected invalidates the whole model;
    return all NaN with nonfinite_prediction, without clipping or inversion.
    """
    output = np.full(len(scores), np.nan)
    total = np.zeros(int(selected.sum()))
    for time in np.flatnonzero(bins):
        with np.errstate(over="ignore", invalid="ignore"):
            linear = scores[selected, time, :] @ coefficients + intercept
            probability = expit(linear)
        if not np.isfinite(linear).all() or not np.isfinite(probability).all():
            return output, "nonfinite_prediction"
        total += probability
    output[selected] = total / bins.sum()
    return output, ""


def _regional_cells(inputs: TransferInputs, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return eligible bool source/test cells(S,W,2) mapped from prepared N-choice arrays.

    positions is integer(W,), original shared-choice indices. Intersect baseline,
    no-conflict and selected-target validity before using actual-class cells.
    Raise ValueError for class orientation or role overlap; never relabel trials.
    """
    source, test = inputs.source_cells[positions].transpose(1, 0, 2).copy(), inputs.test_cells[positions].transpose(1, 0, 2).copy()
    common = inputs.baseline_valid[positions] & ~inputs.conflict_mask[positions]
    for split, definition in enumerate(inputs.transfers):
        target = inputs.target_names.index(definition.target_name)
        eligible = common & inputs.target_valid[positions, target]
        actual = inputs.target_values[positions, target]
        for cells in (source, test):
            cells[split] &= eligible[:, None]
            for label in (0, 1):
                if np.any(cells[split, :, label] & (actual != label)):
                    raise ValueError("Transfer cells must retain actual class-0/class-1 orientation")
        if np.any(source[split].any(axis=1) & test[split].any(axis=1)):
            raise ValueError("Source and complementary test roles must be disjoint")
    return source, test


def _validate_region_inputs(scores: np.ndarray, positions: np.ndarray, inputs: TransferInputs) -> None:
    """Validate W,B,K scores, integer W->N mapping and N-choice/bin contracts; return None.

    Scores retain original PC units; labels and masks are dimensionless. Ordinary
    structural/label errors raise ValueError before any readout is fitted.
    """
    if scores.ndim != 3 or not all(scores.shape) or not np.isfinite(scores).all():
        raise ValueError("Scores must be finite with nonempty shape (W,B,K)")
    n = len(inputs.choices)
    if (positions.shape != (len(scores),) or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions < 0) or np.any(positions >= n) or np.any(np.diff(positions) <= 0)):
        raise ValueError("choice_positions must be ordered unique integer indices into N choices")
    if inputs.target_names != TARGET_NAMES or inputs.transfers != TRANSFERS or inputs.window_names != WINDOW_NAMES:
        raise ValueError("Prepared target, split and window registries must match this analysis version")
    if inputs.target_values.shape != (n, len(TARGET_NAMES)):
        raise ValueError("Targets must have shape (N,4)")
    for mask, shape in (
        (inputs.target_valid, (n, len(TARGET_NAMES))), (inputs.baseline_valid, (n,)),
        (inputs.conflict_mask, (n,)), (inputs.source_cells, (n, len(TRANSFERS), 2)),
        (inputs.test_cells, (n, len(TRANSFERS), 2)),
        (inputs.window_bin_masks, (len(WINDOW_NAMES), scores.shape[1])),
    ):
        if mask.shape != shape or mask.dtype != bool:
            raise ValueError("Prepared masks must be Boolean with aligned N/S/H/B axes")
    if not inputs.window_bin_masks.any(axis=1).all():
        raise ValueError("Every supervised window must select at least one bin")
    if not np.isin(inputs.target_values[inputs.target_valid], [0, 1]).all():
        raise ValueError("Valid categorical targets must be finite binary labels")
