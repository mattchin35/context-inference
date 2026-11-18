"""Scientific synthetic signals and full saved-session acceptance at both bin widths."""

import shutil

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.neural_analysis.communication_subspace.configuration import CommunicationConfig
from src.neural_analysis.communication_subspace.modeling import (
    fit_ridge_decomposition,
    inverse_transform_predictions,
    predict_rank,
    transform_counts,
)
from src.neural_analysis.communication_subspace.plotting import plot_rank_curves
from src.neural_analysis.communication_subspace.preparation import load_session_inputs
from src.neural_analysis.communication_subspace.results import load_result
from src.neural_analysis.communication_subspace.run_session import run_session
from src.neural_analysis.communication_subspace.scoring import score_target_predictions


ROLES = {"PFC": "A", "HPC": "B"}


def _held_out_r2(source: np.ndarray, target: np.ndarray, rank: int) -> float:
    """Return dimensionless held-out mean R2 for 240/80 dimensionless rows.

    source and target have shapes (320, 3), observation then neuron; first
    240 rows train, remaining 80 test. rank bounds the slope coefficient.
    """
    fitted = fit_ridge_decomposition(source[:240], target[:240], alpha=0.01)
    predicted = predict_rank(source[240:], fitted, rank)
    return score_target_predictions(target[240:], predicted).mean_r2


def test_known_low_high_rank_and_independent_null_signals():
    """No inputs; rank-1 signal is compact, rank-3 signal needs more dimensions."""
    rng = np.random.default_rng(20261116)
    source = rng.normal(size=(320, 3))
    low_rank_target = source @ np.array([[1.], [0.5], [-0.4]]) @ np.array([[1., 0.7, -0.8]])
    low_rank_target += 0.15 * rng.normal(size=(320, 3))
    high_rank_target = source @ np.diag([1.0, 0.9, 0.8])
    high_rank_target += 0.15 * rng.normal(size=(320, 3))
    independent_target = rng.normal(size=(320, 3))

    low_one = _held_out_r2(source, low_rank_target, rank=1)
    low_three = _held_out_r2(source, low_rank_target, rank=3)
    high_one = _held_out_r2(source, high_rank_target, rank=1)
    high_three = _held_out_r2(source, high_rank_target, rank=3)
    null_one = _held_out_r2(source, independent_target, rank=1)
    null_three = _held_out_r2(source, independent_target, rank=3)

    assert low_one > 0.85
    assert low_three - low_one < 0.03
    assert high_three > 0.85
    assert high_three - high_one > 0.30
    assert null_one < 0.10
    assert null_three < 0.10


def _verify_saved_predictions(result, prepared) -> int:
    """Replay all scored outer fits; return compared per-target score row count.

    result contains selected fit factors and fold rows. prepared has paired
    float64 counts (choice, bin, unit), spikes/bin. Replay uses only saved
    indices/transforms/decomposition, then compares SSE/SST in counts/bin
    squared and dimensionless R2 against every available target row.
    """
    compared = 0
    fold_scores = result.computation.fold_scores
    for fit in result.computation.fit_records:
        source_region, target_region = fit.direction.split("->")
        split = result.computation.splits[fit.fold]
        source_counts = prepared.counts[source_region][
            split.outer_test_indices][:, :, fit.source_indices]
        target_counts = prepared.counts[target_region][
            split.outer_test_indices][:, :, fit.target_indices]
        source_flat = source_counts.reshape(-1, len(fit.source_indices))
        target_flat = target_counts.reshape(-1, len(fit.target_indices))
        source_z = transform_counts(source_flat, fit.source_transform)
        rows = fold_scores.loc[
            (fold_scores.draw == fit.draw) & (fold_scores.direction == fit.direction) &
            (fold_scores.fold == fit.fold) & (fold_scores.alpha == fit.alpha) &
            (fold_scores.status.isin(("ok", "undefined_score")))
        ]
        for (model, rank), target_rows in rows.groupby(["model", "rank"], dropna=False):
            requested_rank = None if model == "ridge" else int(rank)
            predictions_z = predict_rank(source_z, fit.decomposition, requested_rank)
            predictions = inverse_transform_predictions(predictions_z, fit.target_transform)
            replayed = score_target_predictions(target_flat, predictions)
            for row in target_rows.itertuples():
                target_number = fit.target_unit_ids.index(row.target_id)
                np.testing.assert_allclose(row.sse, replayed.sse[target_number], atol=1e-9)
                np.testing.assert_allclose(row.sst, replayed.sst[target_number], atol=1e-9)
                np.testing.assert_allclose(row.r2, replayed.r2[target_number], atol=1e-9,
                                           equal_nan=True)
                compared += 1
    return compared


@pytest.mark.parametrize("bin_width_ms", [100, 500])
def test_real_on_disk_pipeline_reopens_and_replays_both_bin_widths(
    synthetic_communication_session, tmp_path, bin_width_ms,
):
    """Metadata session -> real PNG/NPZ -> source-free reopen at 100/500 ms."""
    config = CommunicationConfig(synthetic_communication_session, ROLES,
                                 bin_width_ms=bin_width_ms, seed=11)
    report = run_session(config)
    assert report.status == "prepared"
    assert report.result_path.is_file()
    png = report.run_directory / "rank_curves.png"
    assert png.is_file() and png.stat().st_size > 5_000
    image = plt.imread(png)
    assert image.ndim == 3 and image.shape[0] > 500 and image.shape[1] > 1000
    assert report.result_path.parent == synthetic_communication_session.parent / "communication_subspace" / "results"
    print(f"SYNTHETIC_SESSION={synthetic_communication_session}")
    print(f"SYNTHETIC_PNG_{bin_width_ms}={png}")

    loaded = load_result(report.result_path)
    prepared = load_session_inputs(config)
    assert loaded.computation.status == "success"
    assert len(loaded.computation.draws) == 5
    assert len(loaded.computation.splits) == 5
    assert set(loaded.computation.fold_scores.direction) == {
        "PFC->PFC", "PFC->HPC", "HPC->HPC", "HPC->PFC",
    }
    scored_rows = int(loaded.computation.fold_scores.status.isin(("ok", "undefined_score")).sum())
    assert scored_rows > 500
    assert _verify_saved_predictions(loaded, prepared) == scored_rows

    archive_only = tmp_path / "archive_only" / "communication_subspace" / "results"
    archive_only.mkdir(parents=True)
    copied_archive = archive_only / report.result_path.name
    shutil.copyfile(report.result_path, copied_archive)
    source_free = load_result(copied_archive)
    assert source_free.computation.status == loaded.computation.status
    figure = plot_rank_curves(source_free)
    try:
        assert len(figure.axes) == 2
    finally:
        plt.close(figure)
