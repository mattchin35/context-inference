"""Training-only unit standardization and unwhitened full-rank PCA."""

import numpy as np
import pytest
from sklearn.decomposition import PCA

from src.neural_analysis.population_decoding.features import (
    fit_unit_scaling, transform_unit_activity, fit_fold_pca, transform_pc_activity,
)


def test_scaling_uses_training_only_and_original_unit_order():
    """No inputs; training Hz means/SDs omit constants even if test units vary."""
    train = np.array([[2., 9., 0.], [4., 9., 0.], [8., 9., 0.]])
    fitted = fit_unit_scaling(train, ("a", "constant", "silent"))
    np.testing.assert_array_equal(fitted.retained_mask, [True, False, False])
    np.testing.assert_allclose(fitted.mean_hz, train.mean(axis=0))
    np.testing.assert_allclose(fitted.scale_hz, train.std(axis=0, ddof=0))
    z = transform_unit_activity(train, ("a", "constant", "silent"), fitted)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-14)
    np.testing.assert_allclose(z.std(axis=0), 1)
    test = np.array([[100., 123., 99.]])
    transform_unit_activity(test, ("a", "constant", "silent"), fitted)
    np.testing.assert_allclose(fitted.mean_hz, train.mean(axis=0))
    with pytest.raises(ValueError, match="order"):
        transform_unit_activity(test, ("silent", "a", "constant"), fitted)
    with pytest.raises(ValueError, match="variable"):
        fit_unit_scaling(np.ones((4, 2)), ("a", "b"))


def test_pca_matches_library_transform_and_keeps_all_nonzero_components():
    """No inputs; full-SVD PCA numeric parameters reproduce scores with no whitening."""
    rng = np.random.default_rng(7)
    independent = rng.normal(size=(28, 4))
    activity = np.column_stack([independent, independent[:, 0] + independent[:, 1]])
    fitted = fit_fold_pca(activity)
    assert fitted.components.shape == (4, 5)
    oracle = PCA(svd_solver="full", whiten=False).fit(activity)
    np.testing.assert_allclose(transform_pc_activity(activity, fitted), oracle.transform(activity)[:, :4])
    heldout = rng.normal(size=(5, 5)) * 100
    before = fitted.components.copy()
    transform_pc_activity(heldout, fitted)
    np.testing.assert_array_equal(fitted.components, before)


def test_six_variable_dimensions_have_no_leading_pc_cap():
    """No inputs; training matrix (80,6) retains six nonzero PCs."""
    data = np.random.default_rng(8).normal(size=(80, 6))
    assert fit_fold_pca(data).components.shape == (6, 6)
