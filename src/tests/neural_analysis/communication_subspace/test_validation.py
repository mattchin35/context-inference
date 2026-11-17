"""Nested validation, support masks, and fold/draw aggregation contracts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.communication_subspace.computation import compute_communication_subspace
from src.neural_analysis.communication_subspace.preparation import build_validation_splits
from src.neural_analysis.communication_subspace.scoring import (
    summarize_draw_scores,
    summarize_fold_scores,
)


DIRECTIONS = {"PFC->PFC", "PFC->HPC", "HPC->HPC", "HPC->PFC"}


def _synthetic_prepared(seed: int = 14) -> SimpleNamespace:
    """Return 40-choice, 8-bin, 4-unit/region positive counts/bin and metadata."""
    rng = np.random.default_rng(seed)
    choice_count, bin_count, unit_count = 40, 8, 4
    latent = rng.normal(size=(choice_count, bin_count, 2))
    counts = {}
    for region in ("PFC", "HPC"):
        mixing = rng.normal(size=(2, unit_count))
        rates = np.exp(1.8 + 0.22 * (latent @ mixing))
        counts[region] = rng.poisson(rates).astype(np.float64)
    choice_times = 1_700_000_000.0 + 10.0 * np.arange(choice_count)
    choices = pd.DataFrame({"choice_time": choice_times, "cur_trial": np.arange(choice_count)})
    return SimpleNamespace(
        session_id="synthetic-computation", time_origin_s=float(choice_times[0] - 4),
        probe_roles={"PFC": "A", "HPC": "B"}, choices=choices,
        selected_trials=choices.copy(), choice_positions=np.arange(choice_count),
        counts=counts,
        unit_ids={region: tuple(f"{region}:{index}" for index in range(unit_count))
                  for region in counts},
        bin_edges_s=np.linspace(-2., 2., bin_count + 1),
        coverage_s={region: (choice_times[0] - 5, choice_times[-1] + 5)
                    for region in counts},
        configuration={"bin_width_ms": 500, "seed": seed},
    )


def test_outer_test_perturbation_cannot_change_own_inner_tuning_or_outer_fit():
    """No inputs; a fold's held-out (choice,bin,unit) counts cannot alter its fit."""
    prepared = _synthetic_prepared()
    original = compute_communication_subspace(prepared, seed=3)
    folds = build_validation_splits(prepared.choices.choice_time.to_numpy())
    changed_counts = {region: values.copy() for region, values in prepared.counts.items()}
    for region in changed_counts:
        changed_counts[region][folds[0].outer_test_indices] += 1000.
    perturbed = SimpleNamespace(**{**vars(prepared), "counts": changed_counts})
    changed = compute_communication_subspace(perturbed, seed=3)

    original_inner = original.inner_scores.loc[original.inner_scores.fold == 0].reset_index(drop=True)
    changed_inner = changed.inner_scores.loc[changed.inner_scores.fold == 0].reset_index(drop=True)
    pd.testing.assert_frame_equal(original_inner, changed_inner)
    original_outer = original.fold_scores.loc[original.fold_scores.fold == 0]
    changed_outer = changed.fold_scores.loc[changed.fold_scores.fold == 0]
    pd.testing.assert_frame_equal(
        original_outer[["draw", "direction", "fold", "model", "rank", "alpha", "status"]]
        .reset_index(drop=True),
        changed_outer[["draw", "direction", "fold", "model", "rank", "alpha", "status"]]
        .reset_index(drop=True),
    )
    original_fits = {(fit.draw, fit.direction, fit.fold, fit.alpha): fit
                     for fit in original.fit_records if fit.fold == 0}
    changed_fits = {(fit.draw, fit.direction, fit.fold, fit.alpha): fit
                    for fit in changed.fit_records if fit.fold == 0}
    assert original_fits.keys() == changed_fits.keys()
    for key in original_fits:
        before, after = original_fits[key], changed_fits[key]
        np.testing.assert_array_equal(before.source_transform.means, after.source_transform.means)
        np.testing.assert_array_equal(before.target_transform.means, after.target_transform.means)
        np.testing.assert_array_equal(before.decomposition.ridge_coefficients,
                                      after.decomposition.ridge_coefficients)
        np.testing.assert_array_equal(before.decomposition.right_vectors,
                                      after.decomposition.right_vectors)


def test_shared_folds_four_directions_and_full_rank_ridge_agreement():
    """No inputs; all models share five outer folds and feasible rank 2 matches ridge."""
    result = compute_communication_subspace(_synthetic_prepared(), seed=9)
    assert set(result.fold_scores.direction) == DIRECTIONS
    assert set(result.fold_scores.draw) == set(range(5))
    assert set(result.fold_scores.fold) == set(range(5))
    assert len(result.splits) == 5
    assert len(result.draws) == 5
    for direction in DIRECTIONS:
        for draw in range(5):
            each = result.fold_scores.loc[
                (result.fold_scores.direction == direction) & (result.fold_scores.draw == draw)
            ]
            assert set(each.fold) == set(range(5))
            for fold in range(5):
                full_rank = each.loc[(each.fold == fold) & (each.model == "rrr") &
                                     (each["rank"] == 2)].sort_values("target_id")
                ridge = each.loc[(each.fold == fold) & (each.model == "ridge")]
                ridge = ridge.sort_values("target_id")
                assert len(full_rank) == len(ridge) == 2
                assert full_rank.status.eq("ok").all()
                assert ridge.status.eq("ok").all()
                np.testing.assert_allclose(full_rank.alpha.to_numpy(), ridge.alpha.to_numpy())
                np.testing.assert_allclose(full_rank.r2.to_numpy(), ridge.r2.to_numpy(), atol=1e-10)
    assert result.status == "success"


def test_inner_constant_unit_does_not_silently_relabel_unsupported_rank():
    """No inputs; train-constant source/target counts make rank 2 unavailable."""
    prepared = _synthetic_prepared()
    counts = {region: values.copy() for region, values in prepared.counts.items()}
    # The final outer fold is never used to fit itself; all other choices have
    # a constant PFC unit, while its held-out activity is allowed to vary.
    counts["PFC"][:32, :, 0] = 0.
    prepared.counts = counts
    result = compute_communication_subspace(prepared, seed=2)
    affected = result.fold_scores.loc[
        (result.fold_scores.fold == 4) &
        (result.fold_scores.direction == "PFC->PFC") &
        (result.fold_scores.model == "rrr") & (result.fold_scores["rank"] == 2)
    ]
    assert not affected.empty
    assert affected.status.eq("unsupported_rank").all()
    assert affected.r2.isna().all()
    assert affected.reason.str.len().gt(0).all()


def test_complete_fold_and_draw_summaries_keep_exact_support():
    """No inputs; target-level dimensionless R2 aggregates only complete folds/draws."""
    rows = []
    for draw in range(3):
        for fold in range(5):
            for model, rank in (("rrr", 1), ("ridge", None)):
                valid = not (draw == 2 and fold == 4 and model == "rrr")
                rows.append({
                    "draw": draw, "direction": "PFC->HPC", "fold": fold,
                    "model": model, "rank": rank, "target_id": "HPC:0",
                    "r2": float(draw + fold) if valid else np.nan,
                    "valid": valid, "status": "ok" if valid else "undefined_score",
                    "reason": "" if valid else "constant held-out target",
                })
    fold_rows = pd.DataFrame(rows)
    draw_scores = summarize_fold_scores(fold_rows, expected_folds=5)
    rank_rows = draw_scores.loc[draw_scores.model == "rrr"].sort_values("draw")
    assert rank_rows.n_folds.tolist() == [5, 5, 4]
    assert rank_rows.status.tolist() == ["ok", "ok", "incomplete"]
    assert rank_rows.mean_r2.iloc[:2].tolist() == [2., 3.]
    assert np.isnan(rank_rows.mean_r2.iloc[2])
    curve_scores = summarize_draw_scores(draw_scores, expected_draws=3)
    rank_curve = curve_scores.loc[curve_scores.model == "rrr"].iloc[0]
    ridge_curve = curve_scores.loc[curve_scores.model == "ridge"].iloc[0]
    assert rank_curve.n_draws == 2
    assert tuple(rank_curve.draw_ids) == (0, 1)
    assert rank_curve.status == "incomplete"
    assert rank_curve.mean_r2 == 2.5
    assert ridge_curve.n_draws == 3
    assert tuple(ridge_curve.draw_ids) == (0, 1, 2)
    assert ridge_curve.status == "ok"


def test_prepared_counts_and_axes_fail_before_fitting():
    """No inputs; malformed counts/bin or identity/edge axes raise ValueError."""
    prepared = _synthetic_prepared()
    prepared.counts["PFC"][0, 0, 0] = -1.
    with pytest.raises(ValueError, match="nonnegative|negative"):
        compute_communication_subspace(prepared)

    prepared = _synthetic_prepared()
    prepared.bin_edges_s = prepared.bin_edges_s[:-1]
    with pytest.raises(ValueError, match="bin|edge"):
        compute_communication_subspace(prepared)

    prepared = _synthetic_prepared()
    prepared.unit_ids["PFC"] = ("0",) + prepared.unit_ids["PFC"][1:]
    with pytest.raises(ValueError, match="unit|qualified|probe"):
        compute_communication_subspace(prepared)


def test_progress_reports_completed_tasks_total_and_message():
    """No inputs; 25 draw/fold completions provide monotone counts and text."""
    calls = []
    def record_progress(completed: int, total: int, message: str) -> None:
        """Store one (completed, total, message) notification; return None."""
        calls.append((completed, total, message))

    compute_communication_subspace(_synthetic_prepared(), seed=1, progress=record_progress)
    assert len(calls) == 25
    assert [call[0] for call in calls] == list(range(1, 26))
    assert all(call[1] == 25 for call in calls)
    assert all(isinstance(call[2], str) and call[2] for call in calls)
