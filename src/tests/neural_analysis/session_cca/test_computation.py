"""Direct-unit session CCA contracts: matched count axes and fixed score units."""

from __future__ import annotations

import numpy as np
import pytest
from statsmodels.multivariate.cancorr import CanCorr

from src.neural_analysis.population.cca import (
    PairedCCAFit,
    fit_paired_cca,
    transform_cca_activity,
)


@pytest.fixture
def analytic_pair():
    """Return nonnegative float counts (8,8,3)/(8,8,4) and ordered unit IDs.

    The three exact finite-sample canonical correlations are .9, .45 and .1.
    Values are count-like analytic inputs, not rounded experimental counts.
    """
    return _analytic_counts((0.9, 0.45, 0.1))


def test_known_finite_sample_correlations_and_canonical_covariances(analytic_pair):
    """Matched analytic counts recover known r and sample-SD-one scores."""
    fitted = fit_paired_cca(*analytic_pair)

    np.testing.assert_allclose(fitted.correlations, [0.9, 0.45, 0.1], atol=1e-10)
    assert fitted.n_observations == 64
    assert set(fitted.transforms) == {"PFC", "HPC"}
    assert set(fitted.scores) == {"PFC", "HPC"}
    assert fitted.scores["PFC"].shape == (8, 8, 3)
    assert fitted.scores["HPC"].shape == (8, 8, 3)
    assert fitted.transforms["PFC"].rank == 3
    assert fitted.transforms["HPC"].rank == 4
    _assert_score_covariances(fitted)


def test_full_rank_spectrum_agrees_with_direct_unit_statsmodels(analytic_pair):
    """A separate direct-unit library call agrees with the rank-adapter fit."""
    pfc, hpc, _, _ = analytic_pair
    x = _standardized_columns(pfc)
    y = _standardized_columns(hpc)
    reference = CanCorr(endog=y, exog=x, missing="raise", hasconst=False)

    fitted = fit_paired_cca(*analytic_pair)

    np.testing.assert_allclose(fitted.correlations, reference.cancorr, atol=1e-10)
    reference_x = x @ reference.x_cancoef * np.sqrt(len(x) - 1)
    reference_y = y @ reference.y_cancoef * np.sqrt(len(y) - 1)
    for region, expected in (("PFC", reference_x), ("HPC", reference_y)):
        actual = fitted.scores[region].reshape(len(x), -1)
        agreement = np.abs(actual.T @ expected / (len(x) - 1))
        np.testing.assert_allclose(agreement, np.eye(3), atol=1e-9, rtol=0)


def test_correlations_are_sorted_and_display_count_does_not_cap_basis():
    """All 13 numerical pairs survive despite the viewer's ten-row default."""
    expected = np.linspace(0.05, 0.95, 13)
    pair = _analytic_counts(tuple(expected))

    fitted = fit_paired_cca(*pair)

    np.testing.assert_allclose(fitted.correlations, expected[::-1], atol=1e-10)
    assert fitted.scores["PFC"].shape == (8, 8, 13)
    assert fitted.scores["HPC"].shape == (8, 8, 13)
    _assert_score_covariances(fitted)


def test_pooled_unit_statistics_and_constant_omission_keep_input_axis(analytic_pair):
    """Zero/decimal constants are omitted once; saved means/SDs retain input order."""
    pfc, hpc, pfc_ids, hpc_ids = analytic_pair
    pfc = np.concatenate((pfc, np.zeros((8, 8, 1)), np.full((8, 8, 1), 7.3)), axis=2)
    pfc_ids = (*pfc_ids, "ProbeA:zero", "ProbeA:decimal")

    fitted = fit_paired_cca(pfc, hpc, pfc_ids, hpc_ids)

    transform = fitted.transforms["PFC"]
    assert transform.input_unit_ids == pfc_ids
    assert transform.retained_unit_ids == pfc_ids[:3]
    assert transform.omitted_unit_ids == pfc_ids[3:]
    assert transform.coefficients.shape == (3, 3)
    assert transform.count_mean.shape == (5,)
    assert transform.count_scale.shape == (5,)
    np.testing.assert_allclose(transform.count_mean, pfc.mean(axis=(0, 1)))
    np.testing.assert_allclose(transform.count_scale[:3], pfc[:, :, :3].std(axis=(0, 1), ddof=0))
    retained = np.array([0, 1, 2])
    expected = ((pfc[..., retained] - transform.count_mean[retained])
                / transform.count_scale[retained]) @ transform.coefficients - transform.score_offset
    np.testing.assert_allclose(fitted.scores["PFC"], expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(fitted.correlations, [0.9, 0.45, 0.1], atol=1e-10)


def test_projection_reuses_saved_affine_transform_without_local_scaling(analytic_pair):
    """New (2,3,U) counts project with training scales/offsets in canonical SD units."""
    fitted = fit_paired_cca(*analytic_pair)
    transform = fitted.transforms["PFC"]
    held_out = np.random.default_rng(7301).uniform(15, 30, size=(2, 3, 3)).astype(np.float32)
    before = held_out.copy()
    coefficients_before = transform.coefficients.copy()
    expected = ((held_out.astype(np.float64) - transform.count_mean)
                / transform.count_scale) @ transform.coefficients - transform.score_offset

    projected = transform_cca_activity(held_out, analytic_pair[2], transform)

    assert projected.shape == (2, 3, 3)
    assert projected.dtype == np.float64
    np.testing.assert_allclose(projected, expected, rtol=1e-8, atol=1e-10)
    assert not np.allclose(projected.mean(axis=(0, 1)), 0)
    assert not np.allclose(projected.reshape(-1, 3).std(axis=0, ddof=1), 1)
    np.testing.assert_array_equal(held_out, before)
    np.testing.assert_array_equal(transform.coefficients, coefficients_before)


@pytest.mark.parametrize("unit_ids", [("ProbeA:1", "ProbeA:0", "ProbeA:2"), ("other:0", "ProbeA:1", "ProbeA:2")])
def test_projection_rejects_changed_unit_order_or_identity(analytic_pair, unit_ids):
    """Equal array sizes do not make different unit axes interchangeable."""
    fitted = fit_paired_cca(*analytic_pair)

    with pytest.raises(ValueError):
        transform_cca_activity(analytic_pair[0], unit_ids, fitted.transforms["PFC"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.0])
def test_projection_rejects_invalid_counts(analytic_pair, value):
    """Projection never repairs missing/nonphysical count values."""
    fitted = fit_paired_cca(*analytic_pair)
    invalid = analytic_pair[0].copy()
    invalid[0, 0, 0] = value

    with pytest.raises(ValueError):
        transform_cca_activity(invalid, analytic_pair[2], fitted.transforms["PFC"])


def test_dependent_units_preserve_observable_basis_and_minimum_norm(analytic_pair):
    """Duplicated standardized units split coefficients without discarding unit IDs."""
    pfc, hpc, pfc_ids, hpc_ids = analytic_pair
    original = fit_paired_cca(*analytic_pair)
    pfc_duplicate = np.concatenate((pfc, pfc[:, :, :1] * 3 + 5), axis=2)
    hpc_duplicate = np.concatenate((hpc, hpc[:, :, :1] * 2 + 3), axis=2)
    duplicate_ids = ((*pfc_ids, "ProbeA:duplicate"), (*hpc_ids, "ProbeB:duplicate"))

    fitted = fit_paired_cca(pfc_duplicate, hpc_duplicate, *duplicate_ids)

    np.testing.assert_allclose(fitted.correlations, original.correlations, atol=1e-10)
    for region, ids, expected_rank in (("PFC", duplicate_ids[0], 3), ("HPC", duplicate_ids[1], 4)):
        transform = fitted.transforms[region]
        assert transform.input_unit_ids == ids
        assert transform.retained_unit_ids == ids
        assert transform.omitted_unit_ids == ()
        assert transform.rank == expected_rank
        np.testing.assert_allclose(transform.coefficients[0], transform.coefficients[-1], atol=1e-9, rtol=1e-8)
        old_scores = original.scores[region].reshape(64, -1)
        new_scores = fitted.scores[region].reshape(64, -1)
        np.testing.assert_allclose(np.abs(old_scores.T @ new_scores / 63), np.eye(3), atol=1e-9, rtol=0)
    _assert_score_covariances(fitted)


@pytest.mark.parametrize("perturbation, expected_rank", [(1e-15, 1), (1e-4, 2)])
def test_rank_rule_removes_only_numerical_null_directions(perturbation, expected_rank):
    """A tiny non-null direction is retained without a variance-coverage threshold."""
    basis = _centered_basis(64, 2)
    pfc = _positive_columns(np.column_stack((basis[:, 0], basis[:, 0] + perturbation * basis[:, 1]))).reshape(8, 8, 2)
    hpc = _positive_columns(basis).reshape(8, 8, 2)
    ids = (("ProbeA:0", "ProbeA:1"), ("ProbeB:0", "ProbeB:1"))

    fitted = fit_paired_cca(pfc, hpc, *ids)

    transform = fitted.transforms["PFC"]
    assert transform.rank == expected_rank
    assert transform.retained_unit_ids == ids[0]
    assert transform.omitted_unit_ids == ()
    expected_cutoff = np.finfo(np.float64).eps * 64 * transform.singular_values[0]
    assert transform.rank_cutoff == pytest.approx(expected_cutoff, rel=1e-12)
    assert fitted.correlations.shape == (expected_rank,)
    np.testing.assert_allclose(fitted.correlations, 1, atol=1e-8, rtol=0)
    _assert_score_covariances(fitted)


def test_small_unit_variance_is_not_a_unit_selection_cutoff(analytic_pair):
    """Positive activity scales far below one count do not become constant units."""
    pfc, hpc, pfc_ids, hpc_ids = analytic_pair
    pfc = pfc.copy()
    pfc[:, :, 1] *= 1e-12

    fitted = fit_paired_cca(pfc, hpc, pfc_ids, hpc_ids)

    assert fitted.transforms["PFC"].retained_unit_ids == pfc_ids
    assert fitted.transforms["PFC"].rank == 3
    np.testing.assert_allclose(fitted.correlations, [0.9, 0.45, 0.1], atol=1e-10)


@pytest.mark.parametrize("correlations", [(0.7, 0.7, 0.0), (0.0, 0.0, 0.0)])
def test_tied_and_zero_correlations_remain_valid_pairs(correlations):
    """Degenerate spectra retain all numerical pairs without prescribing arbitrary vectors."""
    fitted = fit_paired_cca(*_analytic_counts(correlations))

    np.testing.assert_allclose(fitted.correlations, correlations, atol=1e-10, rtol=0)
    assert fitted.scores["PFC"].shape[-1] == 3
    assert fitted.scores["HPC"].shape[-1] == 3
    _assert_score_covariances(fitted)


def test_tied_nonzero_pairs_preserve_the_known_shared_subspace():
    """The tied first two PFC pairs span the analytic shared plane, with no unique axes."""
    fitted = fit_paired_cca(*_analytic_counts((0.7, 0.7, 0.0)))
    shared = _centered_basis(64, 7)[:, :2]
    scores = fitted.scores["PFC"].reshape(64, 3)[:, :2]

    np.testing.assert_allclose(scores @ scores.T / 63, shared @ shared.T / 63, atol=1e-9, rtol=0)


def test_sign_convention_and_common_color_limit_use_saved_session_scores(analytic_pair):
    """The largest PFC coefficient sets joint sign; colors span both full score arrays."""
    fitted = fit_paired_cca(*analytic_pair)
    coefficients = fitted.transforms["PFC"].coefficients
    dominant = np.argmax(np.abs(coefficients), axis=0)

    assert np.all(coefficients[dominant, np.arange(coefficients.shape[1])] > 0)
    assert fitted.joint_signs.shape == (3,)
    assert np.isin(fitted.joint_signs, [-1, 1]).all()
    expected_limit = max(float(np.abs(scores).max()) for scores in fitted.scores.values())
    assert fitted.color_limit == expected_limit
    assert (fitted.correlations >= 0).all()


def test_rank_saturation_reports_structurally_forced_unit_correlations():
    """Two rank-five regions in seven centered observation dimensions share at least three axes."""
    basis = _centered_basis(8, 7)
    pfc = _positive_columns(basis[:, :5]).reshape(1, 8, 5)
    hpc = _positive_columns(basis[:, 2:7]).reshape(1, 8, 5)
    ids = (tuple(f"ProbeA:{j}" for j in range(5)), tuple(f"ProbeB:{j}" for j in range(5)))

    fitted = fit_paired_cca(pfc, hpc, *ids)

    assert fitted.transforms["PFC"].rank == 5
    assert fitted.transforms["HPC"].rank == 5
    assert fitted.diagnostics["forced_unit_correlations"] == 3
    np.testing.assert_allclose(fitted.correlations, [1, 1, 1, 0, 0], atol=1e-9, rtol=0)
    _assert_score_covariances(fitted)


def test_silent_window_is_retained_and_included_in_pooled_scaling():
    """Covered zero-count bins are observations, not missing windows or forced zero scores."""
    rng = np.random.default_rng(7302)
    pfc = rng.poisson(3, size=(4, 8, 3))
    hpc = rng.poisson(3, size=(4, 8, 2))
    pfc[0] = 0
    hpc[0] = 0
    ids = (("ProbeA:0", "ProbeA:1", "ProbeA:2"), ("ProbeB:0", "ProbeB:1"))

    fitted = fit_paired_cca(pfc, hpc, *ids)

    assert fitted.n_observations == 32
    for region, counts in (("PFC", pfc), ("HPC", hpc)):
        np.testing.assert_allclose(fitted.transforms[region].count_mean, counts.mean(axis=(0, 1)))
        assert fitted.scores[region].shape[:2] == (4, 8)
        assert np.isfinite(fitted.scores[region][0]).all()
        np.testing.assert_allclose(fitted.scores[region][0], np.repeat(fitted.scores[region][0, :1], 8, axis=0))
    _assert_score_covariances(fitted)


def test_one_variable_unit_can_define_one_pair():
    """A valid rank-one population is not rejected as insufficient dimensionality."""
    pfc = np.arange(16, dtype=np.float64).reshape(2, 8, 1)
    hpc = np.concatenate((pfc * 2 + 1, pfc**2 + 3), axis=2)

    fitted = fit_paired_cca(pfc, hpc, ("ProbeA:0",), ("ProbeB:0", "ProbeB:1"))

    assert fitted.scores["PFC"].shape == (2, 8, 1)
    np.testing.assert_allclose(fitted.correlations, [1], atol=1e-10)
    _assert_score_covariances(fitted)


@pytest.mark.parametrize("region", ["PFC", "HPC"])
def test_region_with_only_constant_units_is_unavailable(analytic_pair, region):
    """A missing variable regional subspace cannot become a fabricated CCA fit."""
    pfc, hpc, pfc_ids, hpc_ids = analytic_pair
    if region == "PFC":
        pfc = np.full_like(pfc, 3)
    else:
        hpc = np.full_like(hpc, 3)

    with pytest.raises(ValueError):
        fit_paired_cca(pfc, hpc, pfc_ids, hpc_ids)


@pytest.mark.parametrize("region", ["PFC", "HPC"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_fit_rejects_nonfinite_or_negative_counts(analytic_pair, region, value):
    """Both regional inputs must be finite nonnegative measured count activity."""
    pfc, hpc, pfc_ids, hpc_ids = analytic_pair
    pfc = pfc.copy()
    hpc = hpc.copy()
    invalid = pfc if region == "PFC" else hpc
    invalid[0, 0, 0] = value

    with pytest.raises(ValueError):
        fit_paired_cca(pfc, hpc, pfc_ids, hpc_ids)


@pytest.mark.parametrize("pfc_shape,hpc_shape", [
    ((0, 8, 3), (0, 8, 4)),
    ((8, 0, 3), (8, 0, 4)),
    ((8, 8, 0), (8, 8, 4)),
    ((64, 3), (64, 4)),
    ((8, 8, 3, 1), (8, 8, 4)),
    ((7, 8, 3), (8, 8, 4)),
    ((8, 7, 3), (8, 8, 4)),
])
def test_fit_rejects_empty_or_unmatched_window_bin_axes(pfc_shape, hpc_shape):
    """Invalid tensor geometry fails before any reshaping or unpaired fitting."""
    rng = np.random.default_rng(7303)
    pfc = rng.uniform(0, 3, size=pfc_shape)
    hpc = rng.uniform(0, 3, size=hpc_shape)

    with pytest.raises(ValueError):
        fit_paired_cca(pfc, hpc, ("ProbeA:0", "ProbeA:1", "ProbeA:2"),
                       ("ProbeB:0", "ProbeB:1", "ProbeB:2", "ProbeB:3"))


@pytest.mark.parametrize("pfc_ids,hpc_ids", [
    (("ProbeA:0", "ProbeA:1"), ("ProbeB:0", "ProbeB:1", "ProbeB:2", "ProbeB:3")),
    (("ProbeA:0", "ProbeA:0", "ProbeA:2"), ("ProbeB:0", "ProbeB:1", "ProbeB:2", "ProbeB:3")),
    (("ProbeA:0", "ProbeA:1", "ProbeA:2"), ("ProbeB:0", "ProbeB:1", "ProbeB:1", "ProbeB:3")),
])
def test_fit_rejects_mislabeled_or_duplicate_unit_axes(analytic_pair, pfc_ids, hpc_ids):
    """Unit IDs uniquely label every regional input column in its exact order."""
    with pytest.raises(ValueError):
        fit_paired_cca(analytic_pair[0], analytic_pair[1], pfc_ids, hpc_ids)


def test_fit_rejects_fewer_than_two_observations():
    """A single pooled bin has no defined sample variance for normalized CCA."""
    with pytest.raises(ValueError):
        fit_paired_cca(np.ones((1, 1, 2)), np.ones((1, 1, 1)),
                       ("ProbeA:0", "ProbeA:1"), ("ProbeB:0",))


def test_fit_does_not_mutate_input_arrays_and_saves_float64_scores(analytic_pair):
    """Library centering must not modify caller-owned count tensors."""
    pfc, hpc, _, _ = analytic_pair
    before = (pfc.copy(), hpc.copy())

    fitted = fit_paired_cca(*analytic_pair)

    np.testing.assert_array_equal(pfc, before[0])
    np.testing.assert_array_equal(hpc, before[1])
    for region in ("PFC", "HPC"):
        assert fitted.scores[region].dtype == np.float64
        assert fitted.transforms[region].coefficients.dtype == np.float64
        assert fitted.transforms[region].count_mean.dtype == np.float64
        assert fitted.transforms[region].count_scale.dtype == np.float64


# Analytic-fixture and assertion helpers; no production CCA implementation lives here.
def _analytic_counts(correlations: tuple[float, ...]):
    """Return (PFC,HPC,IDs,IDs) with (8,8,K)/(8,8,K+1) nonnegative activity.

    correlations supplies K dimensionless finite-sample Pearson correlations.
    Full invertible regional unit maps preserve their canonical spectrum.
    """
    values = np.asarray(correlations, dtype=float)
    pair_count = len(values)
    basis = _centered_basis(64, pair_count * 2 + 1)
    x = basis[:, :pair_count]
    y = np.column_stack((
        x * values + basis[:, pair_count:pair_count * 2] * np.sqrt(1 - values**2),
        basis[:, -1],
    ))
    x_map = np.diag(np.linspace(1, 2, pair_count))
    x_map += np.diag(np.full(pair_count - 1, 0.2), k=1)
    y_map = np.diag(np.linspace(0.8, 1.6, pair_count + 1))
    y_map += np.diag(np.full(pair_count, 0.1), k=-1)
    pfc = _positive_columns(x @ x_map).reshape(8, 8, pair_count)
    hpc = _positive_columns(y @ y_map).reshape(8, 8, pair_count + 1)
    return (pfc, hpc, tuple(f"ProbeA:{j}" for j in range(pair_count)),
            tuple(f"ProbeB:{j}" for j in range(pair_count + 1)))


def _centered_basis(observations: int, columns: int) -> np.ndarray:
    """Return (observations,columns) cosine columns with zero mean/sample covariance I.

    Counts are dimensionless, 0 < columns < observations; no random seed is needed
    for this deterministic finite-sample analytical reference.
    """
    positions = np.arange(observations, dtype=float)[:, None] + 0.5
    frequencies = np.arange(1, columns + 1, dtype=float)[None, :]
    return np.sqrt(2 * (observations - 1) / observations) * np.cos(
        np.pi * positions * frequencies / observations
    )


def _positive_columns(values: np.ndarray) -> np.ndarray:
    """Shift each (observations,units) analytic column above zero without rounding.

    Return a new float64 array with the same axes and unchanged centered signals.
    """
    return np.asarray(values, dtype=np.float64) - values.min(axis=0) + 1


def _standardized_columns(counts: np.ndarray) -> np.ndarray:
    """Pool (windows,bins,units) and return unit z-scores (observations,units), ddof=0."""
    flat = counts.reshape(-1, counts.shape[-1])
    return (flat - flat.mean(axis=0)) / flat.std(axis=0, ddof=0)


def _assert_score_covariances(fitted: PairedCCAFit) -> None:
    """Assert session canonical means/covariances and paired Pearson r; return None.

    fitted contains float score tensors (W,B,K) in canonical sample-SD units.
    """
    x = fitted.scores["PFC"].reshape(fitted.n_observations, -1)
    y = fitted.scores["HPC"].reshape(fitted.n_observations, -1)
    pair_count = len(fitted.correlations)
    for values in (x, y):
        np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-8, rtol=0)
        np.testing.assert_allclose(values.T @ values / (len(values) - 1), np.eye(pair_count), atol=1e-8, rtol=0)
    np.testing.assert_allclose(x.T @ y / (len(x) - 1), np.diag(fitted.correlations), atol=1e-8, rtol=0)
