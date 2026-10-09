"""Tests for descriptive in-sample linear and Poisson Granger magnitudes."""

from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.interregional import granger, pipeline, records
from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    PCAConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    TemporalConfig,
)
from src.neural_analysis.interregional.pca import fit_descriptive_regional_pcas
from src.neural_analysis.interregional.preparation import build_analysis_trial_masks
from src.neural_analysis.interregional.records import RegionalCountTensor


def _population(role: str, probe_id: str, cluster_ids: tuple[int, ...]):
    """Return one resolved population with stable qualified unit identities."""
    return ResolvedRegionalPopulation(
        role=role,
        probe_id=probe_id,
        channel_source="explicit",
        selected_channels=tuple(range(len(cluster_ids))),
        cluster_ids=cluster_ids,
        unit_ids=tuple(f"{probe_id}:{value}" for value in cluster_ids),
    )


def _prepared(
    *,
    analyses: tuple[str, ...] = ("linear_granger",),
    representations: tuple[str, ...] = ("units",),
    lag_bins: int = 1,
) -> pipeline.PreparedInterregionalSession:
    """Build a deterministic all-eligible-row fixture without CV folds."""
    rng = np.random.default_rng(811)
    n_trials, n_bins = 12, 7
    hpc = rng.poisson(2.5, size=(n_trials, n_bins, 2)).astype(np.int64)
    pfc = rng.poisson(1.5, size=(n_trials, n_bins, 2)).astype(np.int64)
    for bin_position in range(1, n_bins):
        pfc[:, bin_position, 0] += hpc[:, bin_position - 1, 0]
        hpc[:, bin_position, 1] += pfc[:, bin_position - 1, 1]
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": 0,
            "correct": 1,
            "reward": 1,
            "action": 1,
            "choice_time": np.arange(n_trials, dtype=float) * 10.0,
            "cur_block": [None, *np.repeat(np.arange(11), 1)],
        },
        index=np.arange(300, 300 + n_trials),
    )
    windows = AnalysisWindows(
        whole_start_s=-0.3, split_s=0.0, whole_stop_s=0.4
    )
    temporal = TemporalConfig(
        bin_size_s=0.1, lag_bins=lag_bins, order_bins=1
    )
    config = InterregionalAnalysisConfig(
        session_metadata_path="/data/session/neural_session.json",
        pfc_population=RegionalPopulationConfig(role="PFC", probe_id="pfc"),
        hpc_population=RegionalPopulationConfig(role="HPC", probe_id="hpc"),
        windows=windows,
        prediction_windows=("whole",),
        temporal=temporal,
        filters=FilterConfig(conditions=("all",)),
        pca=PCAConfig(pfc_components=2, hpc_components=2),
        representations=representations,
        analyses=analyses,
    )
    edges = windows.whole_start_s + np.arange(n_bins + 1) * temporal.bin_size_s
    edges[0], edges[-1] = windows.whole_start_s, windows.whole_stop_s
    rows = np.arange(n_trials, dtype=np.int64)
    labels = tuple(str(value) for value in trial_df.index)
    pfc_population = _population("PFC", "pfc", (1, 2))
    hpc_population = _population("HPC", "hpc", (3, 4))
    return pipeline.PreparedInterregionalSession(
        session_id="granger-synthetic",
        config=config,
        pfc_population=pfc_population,
        hpc_population=hpc_population,
        pfc_counts=RegionalCountTensor(
            pfc, rows, labels, edges, pfc_population.unit_ids
        ),
        hpc_counts=RegionalCountTensor(
            hpc, rows, labels, edges, hpc_population.unit_ids
        ),
        trial_masks=build_analysis_trial_masks(
            trial_df, alignment="choice_time", filters=config.filters
        ),
        fold_assignment=None,
    )


def test_linear_granger_matches_hand_calculated_log_sse_ratio() -> None:
    """Linear magnitude uses identical-row restricted and full residual sums."""
    observed = np.array([1.0, 2.0, 4.0])
    restricted = np.array([2.0, 2.0, 2.0])
    full = np.array([1.0, 2.0, 3.0])

    scores = granger.compute_linear_granger(observed, restricted, full)

    assert scores.sse_restricted == pytest.approx(5.0)
    assert scores.sse_full == pytest.approx(1.0)
    assert scores.linear_granger == pytest.approx(math.log(5.0))
    assert scores.diagnostic == ""


def test_linear_granger_rejects_zero_residual_and_substantive_negative_gain() -> None:
    """Undefined ratios and worse nested fits retain frozen reason codes."""
    observed = np.array([1.0, 2.0, 3.0])
    with pytest.raises(granger.GrangerUnavailable) as zero_error:
        granger.compute_linear_granger(observed, observed + 1.0, observed)
    assert zero_error.value.reason == "zero_granger_residual"

    with pytest.raises(granger.GrangerUnavailable) as nested_error:
        granger.compute_linear_granger(
            observed,
            np.array([1.0, 2.0, 3.1]),
            np.array([2.0, 3.0, 4.0]),
        )
    assert nested_error.value.reason == "nested_fit_inconsistency"


def test_nested_roundoff_is_clamped_and_diagnosed() -> None:
    """A scale-aware tiny negative improvement is zero, not unavailable."""
    improvement, diagnostic = granger.validate_nested_fit_improvement(
        1.0, 1.0 + 5e-13, larger_is_better=False
    )

    assert improvement == 0.0
    assert diagnostic == "nested_roundoff"


def test_poisson_granger_matches_log_likelihood_and_deviance_identity() -> None:
    """Poisson LR agrees with restricted-minus-full deviance on the same rows."""
    observed = np.array([0, 1, 2, 3])
    restricted = np.array([1.2, 1.2, 1.2, 1.2])
    full = np.array([0.4, 1.1, 2.1, 2.8])

    scores = granger.compute_poisson_granger(observed, restricted, full)
    hand_restricted_llf = sum(
        count * math.log(mean) - mean - math.lgamma(count + 1.0)
        for count, mean in zip(observed, restricted, strict=True)
    )
    hand_full_llf = sum(
        count * math.log(mean) - mean - math.lgamma(count + 1.0)
        for count, mean in zip(observed, full, strict=True)
    )
    hand_lr = 2.0 * (hand_full_llf - hand_restricted_llf)

    assert scores.llf_restricted == pytest.approx(hand_restricted_llf)
    assert scores.llf_full == pytest.approx(hand_full_llf)
    assert scores.likelihood_ratio == pytest.approx(hand_lr)
    assert scores.deviance_restricted - scores.deviance_full == pytest.approx(
        hand_lr
    )
    assert scores.mean_deviance_improvement == pytest.approx(hand_lr / 4.0)
    assert scores.diagnostic == ""


def test_poisson_granger_rejects_inconsistent_nested_predictions() -> None:
    """A substantively worse full Poisson fit is a fit-consistency failure."""
    observed = np.array([0, 1, 2, 3])
    with pytest.raises(granger.GrangerUnavailable) as caught:
        granger.compute_poisson_granger(
            observed,
            np.array([0.5, 1.0, 2.0, 3.0]),
            np.full(4, 20.0),
        )
    assert caught.value.reason == "nested_fit_inconsistency"


def test_granger_only_preparation_does_not_construct_folds(monkeypatch) -> None:
    """A missing-block Granger-only session remains valid without GroupKFold."""
    prepared_fixture = _prepared()
    config = prepared_fixture.config
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": [0, 0],
            "choice_time": [1.0, 2.0],
        }
    )
    masks = build_analysis_trial_masks(
        trial_df, alignment="choice_time", filters=config.filters
    )
    rows = np.flatnonzero(masks.scientific_eligible).astype(np.int64)
    edges = prepared_fixture.pfc_counts.bin_edges_s

    def fake_tensor(_spikes, population, *_args, **_kwargs):
        values = np.ones((2, len(edges) - 1, len(population.unit_ids)), dtype=np.int64)
        return RegionalCountTensor(
            values,
            rows,
            tuple(str(value) for value in trial_df.index),
            edges,
            population.unit_ids,
        )

    monkeypatch.setattr(pipeline, "build_regional_count_tensor", fake_tensor)
    monkeypatch.setattr(
        pipeline,
        "build_block_fold_assignment",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("folds constructed")
        ),
    )

    prepared = pipeline.prepare_interregional_session(
        "no-folds",
        object(),
        object(),
        prepared_fixture.pfc_population,
        prepared_fixture.hpc_population,
        trial_df,
        config,
    )

    assert prepared.fold_assignment is None
    assert prepared.trial_masks.scientific_eligible.tolist() == [True, True]


def test_cv_prepared_session_still_requires_fold_assignment() -> None:
    """Making folds optional for Granger must not weaken any CV contract."""
    prepared = _prepared()
    with pytest.raises(ValueError, match="fold assignment"):
        replace(
            prepared,
            config=replace(prepared.config, analyses=("ols_cv",)),
            fold_assignment=None,
        )


def test_linear_pipeline_uses_all_scientific_trials_and_frozen_tables() -> None:
    """Missing-block trials enter descriptive fits and no fold rows are created."""
    prepared = _prepared()

    scores, populations, pca_fits = pipeline.run_descriptive_granger(prepared)

    assert len(scores) == 4
    assert scores["evaluation_scope"].eq("in_sample").all()
    assert scores["model_family"].eq("ols").all()
    assert scores["n_trials"].eq(12).all()
    assert scores["n_rows"].eq(72).all()
    assert scores["linear_granger"].notna().all()
    assert set(populations["metric_name"]) == {"linear_granger"}
    assert populations["evaluation_scope"].eq("in_sample").all()
    assert pca_fits.empty
    records.validate_result_table("granger_scores", scores)
    records.validate_result_table("population_summaries", populations)


def test_linear_granger_uses_only_descriptive_pca_scope() -> None:
    """Session-wide PC Granger rejects a fold-local transform."""
    prepared = _prepared(representations=("pcs",))
    tensor_rows = prepared.pfc_counts.trial_rows
    condition_masks = {
        name: mask[tensor_rows]
        for name, mask in prepared.trial_masks.condition_masks.items()
    }
    descriptive = fit_descriptive_regional_pcas(
        pfc_activity=prepared.pfc_counts.counts,
        hpc_activity=prepared.hpc_counts.counts,
        pfc_unit_ids=prepared.pfc_counts.unit_ids,
        hpc_unit_ids=prepared.hpc_counts.unit_ids,
        scientific_trials=np.ones(len(tensor_rows), dtype=bool),
        condition_masks=condition_masks,
        requested_conditions=prepared.config.filters.conditions,
        requested_components=2,
    )
    fold_scoped = replace(descriptive, scope="fold", fold_id=0)

    with pytest.raises(ValueError, match="descriptive"):
        pipeline.run_descriptive_granger(
            prepared, descriptive_pcas=fold_scoped
        )

    scores, _, pca_fits = pipeline.run_descriptive_granger(
        prepared, descriptive_pcas=descriptive
    )
    assert scores["representation"].eq("pcs").all()
    assert pca_fits["scope"].eq("descriptive").all()
    assert pca_fits["fold_id"].isna().all()


def test_linear_and_poisson_granger_stages_are_runtime_independent(monkeypatch) -> None:
    """Either descriptive model family runs without invoking its CV counterpart."""
    linear_prepared = _prepared(analyses=("linear_granger",))
    original_poisson_fit = pipeline.fit_poisson_target
    monkeypatch.setattr(
        pipeline,
        "fit_poisson_target",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Poisson called")
        ),
    )
    linear_scores, _, _ = pipeline.run_descriptive_granger(linear_prepared)
    assert linear_scores["model_family"].eq("ols").all()
    monkeypatch.setattr(pipeline, "fit_poisson_target", original_poisson_fit)

    poisson_prepared = _prepared(analyses=("poisson_granger",))
    monkeypatch.setattr(
        pipeline,
        "fit_ols_targets",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OLS called")),
    )
    poisson_scores, _, _ = pipeline.run_descriptive_granger(poisson_prepared)
    assert poisson_scores["model_family"].eq("poisson").all()
    assert poisson_scores["likelihood_ratio"].notna().any()


def test_poisson_granger_reuses_each_cell_design_rank_across_targets(
    monkeypatch,
) -> None:
    """Each descriptive direction passes its two shared ranks to target fits."""
    prepared = _prepared(analyses=("poisson_granger",))
    original_diagnostics = pipeline._design_diagnostics
    original_fit = pipeline.fit_poisson_target
    diagnostic_calls = 0
    fitted_designs: list[tuple[int, int | None]] = []

    def count_diagnostics(design):
        """Count shared direction diagnostics while preserving their result."""
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        return original_diagnostics(design)

    def record_fit(design, count_response, **fit_kwargs):
        """Record the reused rank and delegate to the production target fit."""
        fitted_designs.append((design.shape[1], fit_kwargs.get("precomputed_rank")))
        return original_fit(design, count_response, **fit_kwargs)

    monkeypatch.setattr(pipeline, "_design_diagnostics", count_diagnostics)
    monkeypatch.setattr(pipeline, "fit_poisson_target", record_fit)

    pipeline.run_descriptive_granger(prepared)

    assert diagnostic_calls == 4
    assert len(fitted_designs) == 8
    assert all(rank == feature_count for feature_count, rank in fitted_designs)


def test_lag_restricted_pipeline_preserves_requested_history_gap(monkeypatch) -> None:
    """A lag above one is not silently filled with recent source bins."""
    prepared = _prepared(lag_bins=2)
    captured_full_designs: list[np.ndarray] = []
    original_fit = pipeline.fit_ols_targets

    def capture_fit(design, responses):
        if design.shape[1] > 3:
            captured_full_designs.append(np.array(design, copy=True))
        return original_fit(design, responses)

    monkeypatch.setattr(pipeline, "fit_ols_targets", capture_fit)
    pipeline.run_descriptive_granger(prepared)

    assert captured_full_designs
    first_full = captured_full_designs[0]
    assert first_full.shape[0] == 60
    assert first_full.shape[1] == 5


def test_granger_progress_records_exact_cell_array_sizes() -> None:
    """Each descriptive cell exposes enough array detail for memory diagnosis."""
    prepared = _prepared(analyses=("linear_granger", "poisson_granger"))
    events: list[dict[str, object]] = []

    pipeline.run_descriptive_granger(prepared, progress_callback=events.append)

    starts = [event for event in events if event["event"] == "analysis_cell_start"]
    assert len(starts) == 4
    assert {(event["model_family"], event["direction"]) for event in starts} == {
        ("ols", "HPC_to_PFC"),
        ("ols", "PFC_to_HPC"),
        ("poisson", "HPC_to_PFC"),
        ("poisson", "PFC_to_HPC"),
    }
    for event in starts:
        assert event["n_trials"] == 12
        assert event["n_rows"] == 72
        assert event["response_shape"] == [72, 2]
        assert event["response_bytes"] > 0
        assert event["restricted_design_shape"] == [72, 3]
        assert event["restricted_design_bytes"] > 0
        assert event["full_design_shape"] == [72, 5]
        assert event["full_design_bytes"] > 0
