"""Target-specific OOF scope, independent representations, and fold leakage checks."""

from dataclasses import replace
import numpy as np

from src.neural_analysis.population_decoding import pipeline


def test_joint_outcome_scope_and_scientific_signal(synthetic_population):
    """Input known-signal Hz population; four-class OOF uses only two post-choice bins."""
    cell = pipeline.fit_target(synthetic_population, "PFC", "outcome")
    assert cell["units_probabilities"].shape == (240, 2, 4)
    np.testing.assert_array_equal(cell["bin_indices"], [2, 3])
    for representation in ("units", "pcs"):
        summary = cell["meta"]["representations"][representation]
        assert summary["pooled"]["balanced_accuracy"] > .95
        assert summary["conditional_outcomes"]["left"]["roc_auc"][1] > .95
        assert len(summary["folds"]) == 5
    np.testing.assert_allclose(cell["units_probabilities"], cell["pcs_probabilities"], atol=2e-3)


def test_test_fold_activity_cannot_change_its_model_or_scaling(synthetic_population):
    """Input population; perturb held-out activity while preserving fold-zero parameters."""
    original = pipeline.fit_target(synthetic_population, "PFC", "choice")
    region = synthetic_population.regions["PFC"]
    changed = region.rates_hz.copy()
    changed[synthetic_population.fold_ids == 0] += 1000
    modified = replace(synthetic_population, regions={"PFC": replace(region, rates_hz=changed)})
    result = pipeline.fit_target(modified, "PFC", "choice")
    for key in ("fold_0_mean_hz", "fold_0_scale_hz", "fold_0_pca_components",
                "fold_0_units_coefficients", "fold_0_pcs_coefficients"):
        np.testing.assert_array_equal(original[key], result[key])


def test_pca_failure_does_not_erase_raw_readout(synthetic_population, monkeypatch):
    """Input population/patch; PCA numerical failure leaves complete raw-unit OOF scores."""
    def fail_pca(values):
        """Input standardized training matrix; raise simulated SVD failure."""
        raise np.linalg.LinAlgError("simulated SVD failure")
    monkeypatch.setattr(pipeline, "fit_fold_pca", fail_pca)
    cell = pipeline.fit_target(synthetic_population, "PFC", "context")
    assert cell["meta"]["representations"]["units"]["pooled"]["balanced_accuracy"] > .95
    assert cell["meta"]["representations"]["pcs"]["pooled"]["balanced_accuracy"] is None


def test_missing_region_and_terminal_successor_have_explicit_scope(synthetic_population):
    """Input population; absent region unavailable, switch window excludes terminal row only."""
    unavailable = pipeline.fit_target(synthetic_population, "HPC", "choice")
    assert unavailable["meta"]["unavailable_reason"]
    cell = pipeline.fit_target(synthetic_population, "PFC", "left_choice_switch")
    assert cell["units_probabilities"].shape == (239, 4, 2)
    assert 239 not in cell["row_positions"]
