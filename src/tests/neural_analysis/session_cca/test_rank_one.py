"""Regression coverage for statsmodels' one-column endogenous-data squeeze."""

from __future__ import annotations

import numpy as np
import pytest

from src.neural_analysis.population.cca import fit_paired_cca, transform_cca_activity


@pytest.mark.parametrize("pfc_units", [1, 2])
@pytest.mark.parametrize("hpc_slope", [-2.0, 2.0])
def test_single_hpc_direction_handles_both_regional_rank_cases(pfc_units, hpc_slope):
    """Nonnegative counts (2,8,1/2)/(2,8,1) give one positive canonical correlation.

    pfc_units is the integer input/rank count. hpc_slope is a signed count-space
    slope; CCA may reverse the HPC loading, but the paired scores agree. Returns None.
    """
    linear = np.arange(16, dtype=float).reshape(2, 8, 1)
    pfc = linear if pfc_units == 1 else np.concatenate((linear, linear**2), axis=2)
    hpc = linear * hpc_slope + 40
    pfc_ids = tuple(f"ProbeA:{index}" for index in range(pfc_units))
    hpc_ids = ("ProbeB:0",)

    fitted = fit_paired_cca(pfc, hpc, pfc_ids, hpc_ids)

    assert fitted.transforms["PFC"].rank == pfc_units
    assert fitted.transforms["HPC"].rank == 1
    np.testing.assert_allclose(fitted.correlations, [1], atol=1e-10, rtol=0)
    expected = (linear - linear.mean()) / linear.std(ddof=1)
    for region, counts, ids in (("PFC", pfc, pfc_ids), ("HPC", hpc, hpc_ids)):
        assert fitted.scores[region].shape == (2, 8, 1)
        np.testing.assert_allclose(fitted.scores[region], expected, atol=1e-9, rtol=0)
        np.testing.assert_allclose(
            transform_cca_activity(counts, ids, fitted.transforms[region]),
            fitted.scores[region], atol=1e-10, rtol=1e-8,
        )
    coefficients = fitted.transforms["PFC"].coefficients[:, 0]
    assert coefficients[np.argmax(np.abs(coefficients))] > 0
    assert np.sign(fitted.transforms["HPC"].coefficients[0, 0]) == np.sign(hpc_slope)


@pytest.mark.parametrize("correlation", [-0.6, 0.0, 0.6])
def test_two_rank_one_regions_preserve_imperfect_or_zero_correlation(correlation):
    """Float counts (4,8,1) retain a known signed scalar r; return None.

    correlation is dimensionless. Orthogonal cosine vectors fix the sample
    correlation exactly; positive shifts leave that centered relation unchanged.
    """
    positions = np.arange(32, dtype=float) + 0.5
    scale = np.sqrt(2 * 31 / 32)
    x = scale * np.cos(np.pi * positions / 32)
    noise = scale * np.cos(2 * np.pi * positions / 32)
    y = correlation * x + np.sqrt(1 - correlation**2) * noise

    fitted = fit_paired_cca(
        (x + 5).reshape(4, 8, 1),
        (y + 5).reshape(4, 8, 1),
        ("ProbeA:0",), ("ProbeB:0",),
    )

    np.testing.assert_allclose(fitted.correlations, [abs(correlation)], atol=1e-10, rtol=0)
    left = fitted.scores["PFC"].reshape(32)
    right = fitted.scores["HPC"].reshape(32)
    np.testing.assert_allclose(left, x, atol=1e-10, rtol=0)
    np.testing.assert_allclose(np.corrcoef(left, right)[0, 1], abs(correlation), atol=1e-10, rtol=0)
    np.testing.assert_allclose(right.std(ddof=1), 1, atol=1e-10, rtol=0)
