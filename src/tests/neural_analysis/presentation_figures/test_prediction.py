"""Synthetic acceptance tests for fixed-penalty population prediction."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def prediction_inputs():
    """Return synthetic paired counts (choice, 100-ms bin, unit) and five folds."""
    rng = np.random.default_rng(12)
    shared = rng.poisson(2, (10, 40, 1))
    offsets = np.arange(10)[:, None, None]
    counts = {
        region: shared + offsets + rng.poisson(2, (10, 40, n_units))
        for region, n_units in (("PFC", 64), ("HPC", 66))
    }
    splits = tuple(
        SimpleNamespace(
            fold_id=fold,
            train_indices=np.delete(np.arange(10), [2 * fold, 2 * fold + 1, (2 * fold + 2) % 10]),
            test_indices=np.arange(2 * fold, 2 * fold + 2),
            purged_indices=np.array([(2 * fold + 2) % 10]),
        )
        for fold in range(5)
    )
    window = SimpleNamespace(counts=counts, splits=splits, row_positions=np.arange(10, 20),
                             bin_edges_s=np.linspace(-2, 2, 41))
    unit_ids = {region: tuple(f"{region}:{i}" for i in range(matrix.shape[2]))
                for region, matrix in counts.items()}
    return SimpleNamespace(windows={((-2., 2.), 100): window}, unit_ids=unit_ids)


@pytest.fixture(scope="module")
def prediction_result(prediction_inputs):
    """Return the numerical Figure 5 result mapping for the synthetic inputs."""
    from src.neural_analysis.presentation_figures.prediction import compute_prediction

    return compute_prediction(prediction_inputs, seed=24)


def test_prediction_draws_and_sweep_contract(prediction_inputs, prediction_result):
    """Check five shared disjoint draws, nested sizes, matched directions and folds."""
    result = prediction_result
    assert result["alpha"] == 0.1
    assert result["panel_d_pcs"] == 30
    assert result["seed"] == 24
    assert result["source_pool_size"] == 32
    assert set(result["curves"].direction) == {"PFC->PFC", "PFC->HPC", "HPC->HPC", "HPC->PFC"}
    assert set(result["curves"].query("panel == 'A'")["count"]) == {1, 2, 4, 8, 16, 32}
    assert set(result["curves"].query("panel == 'B'")["count"]) == {*range(1, 31), 32}
    assert set(result["curves"].query("panel == 'C'")["count"]) == set(range(1, 33))
    assert set(result["curves"].query("panel == 'D'")["count"]) == set(range(1, 31))
    assert result["curves"].n_draws.eq(5).all()
    assert len(result["populations"]) == 5 * 2 * 2 * 32
    for _, units in result["populations"].groupby(["draw", "region"]):
        assert units.unit_id.is_unique
        assert set(units.groupby("role").size()) == {32}
        for _, role in units.groupby("role"):
            assert tuple(sorted(role.pool_position)) == tuple(range(32))
    for saved, split in zip(result["folds"].itertuples(), prediction_inputs.windows[((-2., 2.), 100)].splits):
        assert saved.fold == split.fold_id
        assert tuple(saved.train_indices) == tuple(split.train_indices)
        assert tuple(saved.test_indices) == tuple(split.test_indices)
        assert tuple(saved.purged_indices) == tuple(split.purged_indices)
        assert saved.n_purged_choices == 1


def test_fold_score_matches_training_only_ridge_and_pc_prefix(prediction_inputs, prediction_result):
    """Check raw-scale fold SSE against independent ridge and unwhitened PCA calculations."""
    window = prediction_inputs.windows[((-2., 2.), 100)]
    fold = window.splits[0]
    units = prediction_result["populations"].query("draw == 0")
    indices = {}
    for region, role in (("PFC", "source"), ("HPC", "target")):
        names = units.query("region == @region and role == @role").sort_values("pool_position").unit_id
        indices[region] = [prediction_inputs.unit_ids[region].index(name) for name in names]
    x = window.counts["PFC"][:, :, indices["PFC"]]
    y = window.counts["HPC"][:, :, indices["HPC"]]
    x_train, x_test = (x[index].reshape(-1, 32) for index in (fold.train_indices, fold.test_indices))
    y_train, y_test = (y[index].reshape(-1, 32) for index in (fold.train_indices, fold.test_indices))
    mean_x, scale_x = x_train.mean(0), x_train.std(0)
    mean_y, scale_y = y_train.mean(0), y_train.std(0)
    x_train_z, x_test_z = (x_train - mean_x) / scale_x, (x_test - mean_x) / scale_x
    _, _, pc_vectors = np.linalg.svd(x_train_z, full_matrices=False)
    cases = (
        ("A", 2, x_train_z[:, :2], x_test_z[:, :2]),
        ("A", 32, x_train_z, x_test_z),
        ("B", 2, x_train_z @ pc_vectors[:2].T, x_test_z @ pc_vectors[:2].T),
        ("B", 30, x_train_z @ pc_vectors[:30].T, x_test_z @ pc_vectors[:30].T),
        ("B", 32, x_train_z @ pc_vectors.T, x_test_z @ pc_vectors.T),
    )
    scores = prediction_result["fold_scores"].query("draw == 0 and fold == 0 and direction == 'PFC->HPC'")
    for panel, count, train, test in cases:
        coefficient = np.linalg.solve(train.T @ train + len(train) * 0.1 * np.eye(count),
                                      train.T @ ((y_train - mean_y) / scale_y))
        expected = test @ coefficient * scale_y + mean_y
        actual = scores.query("panel == @panel and count == @count and model == 'ridge'")
        assert len(actual) == 1
        score_index = actual.score_index.item()
        np.testing.assert_allclose(prediction_result["model_sse"][score_index],
                                   ((y_test - expected) ** 2).sum(0))
        np.testing.assert_allclose(prediction_result["baseline_sse"][score_index],
                                   ((y_test - mean_y) ** 2).sum(0))
    draws = prediction_result["draw_scores"]
    for direction, points in draws.groupby("direction"):
        full_units = points.query("panel == 'A' and count == 32").sort_values("draw")
        full_pcs = points.query("panel == 'B' and count == 32").sort_values("draw")
        full_rank = points.query("panel == 'C' and count == 32 and model == 'rrr'").sort_values("draw")
        full_pc_rank = points.query("panel == 'D' and count == 30 and model == 'rrr'").sort_values("draw")
        thirty_pcs = points.query("panel == 'B' and count == 30").sort_values("draw")
        assert len(full_pc_rank) == len(thirty_pcs) == 5
        np.testing.assert_allclose(full_units.mean_r2, full_pcs.mean_r2, err_msg=direction)
        np.testing.assert_allclose(full_units.mean_r2, full_rank.mean_r2, err_msg=direction)
        np.testing.assert_allclose(thirty_pcs.mean_r2, full_pc_rank.mean_r2, err_msg=direction)


def test_draw_r2_pools_squared_errors_before_equal_neuron_average():
    """Check pooled R2, negative retention, undefined neurons and missing folds."""
    from src.neural_analysis.presentation_figures.prediction import _summarize_draw_scores

    rows, model_sse, baseline_sse = [], [], []
    for draw, folds in ((0, range(5)), (1, range(4))):
        for fold in folds:
            errors = ((2, 1), (18, 9), (0, 0)) if fold == 0 else (
                ((4, 9), (0, 0), (0, 0)) if fold == 1 else ((0, 0),) * 3
            )
            rows.append(dict(draw=draw, direction="PFC->PFC", panel="A", count=1, model="ridge",
                             fold=fold, score_index=len(rows), status="ok", reason=""))
            model, baseline = zip(*errors)
            model_sse.append(model)
            baseline_sse.append(baseline)
    summary = _summarize_draw_scores(pd.DataFrame(rows), np.array(model_sse),
                                     np.array(baseline_sse), (0, 1, 2, 3, 4)).set_index("draw")
    assert summary.at[0, "n_neurons"] == 2
    assert summary.at[0, "mean_r2"] == pytest.approx((0.4 - 1.0) / 2)
    assert summary.at[1, "n_folds"] == 4
    assert np.isnan(summary.at[1, "mean_r2"])


def test_rank_references_and_both_directions_use_the_same_draw_intersection():
    """Reproduce the three-of-five case with two references and another direction."""
    from src.neural_analysis.presentation_figures.prediction import _summarize_curves

    rows = []
    for direction in ("PFC->PFC", "HPC->PFC"):
        for draw in range(5):
            for panel, count, model in (("D", 30, "rrr"), ("A", 32, "ridge"), ("B", 30, "ridge")):
                value = 0.10 if draw < 3 else 0.20
                if panel == "D" and (draw == 4 or (direction == "HPC->PFC" and draw == 3)):
                    value = np.nan
                rows.append(dict(draw=draw, direction=direction, panel=panel, count=count,
                                 model=model, mean_r2=value, n_folds=5 if np.isfinite(value) else 4,
                                 n_neurons=32 if np.isfinite(value) else 0,
                                 status="ok" if np.isfinite(value) else "incomplete", reason=""))
    curves = _summarize_curves(pd.DataFrame(rows), source_pool_size=32)
    d = curves.query("panel == 'D' and target_region == 'PFC'")
    assert len(d) == 6
    assert set(d.model) == {"rrr", "ridge_30pc", "ridge_units"}
    assert set(d.draw_ids) == {(0, 1, 2)}
    assert d.n_draws.eq(3).all()
    np.testing.assert_allclose(d.mean_r2, 0.10)


def test_thirty_pc_panel_is_unavailable_below_thirty_source_units(prediction_inputs):
    """Check explicit panel D unavailability alongside supported smaller panels."""
    from src.neural_analysis.presentation_figures.prediction import compute_prediction

    window = prediction_inputs.windows[((-2., 2.), 100)]
    counts = {region: values[:, :, :8] for region, values in window.counts.items()}
    units = {region: ids[:8] for region, ids in prediction_inputs.unit_ids.items()}
    small_window = SimpleNamespace(**{**vars(window), "counts": counts})
    prepared = SimpleNamespace(windows={((-2., 2.), 100): small_window}, unit_ids=units)
    curves = compute_prediction(prepared)["curves"]
    d = curves.query("panel == 'D' and model == 'rrr'")
    assert set(d.direction) == set(curves.direction)
    assert d.n_draws.eq(0).all()
    assert d.mean_r2.isna().all()
    assert d.reason.str.contains("thirty|30", case=False).all()
    assert curves.query("panel == 'A'").n_draws.eq(5).all()
