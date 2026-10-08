"""Synthetic integration tests for the direct-unit inter-regional OLS path."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    PCAConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    TemporalConfig,
)
from src.neural_analysis.interregional.preparation import (
    build_analysis_trial_masks,
    build_block_fold_assignment,
)
from src.neural_analysis.interregional.records import RegionalCountTensor
from src.neural_analysis.interregional.pca import fit_descriptive_regional_pcas
from src.neural_analysis.interregional import pipeline
from src.neural_analysis.interregional.poisson import PoissonFitUnavailable


def _resolved(role: str, probe: str, cluster_id: int) -> ResolvedRegionalPopulation:
    """Return one single-unit resolved regional population."""
    return ResolvedRegionalPopulation(
        role=role,
        probe_id=probe,
        channel_source="explicit",
        selected_channels=(0,),
        cluster_ids=(cluster_id,),
        unit_ids=(f"{probe}:{cluster_id}",),
    )


def _multiunit_prepared(
    representations: tuple[str, ...],
) -> pipeline.PreparedInterregionalSession:
    """Build a deterministic two-region fixture with two units per region."""
    rng = np.random.default_rng(73)
    n_trials, n_bins = 50, 8
    hpc_counts = rng.poisson(3.0, size=(n_trials, n_bins, 2)).astype(np.int64)
    pfc_counts = rng.poisson(2.0, size=(n_trials, n_bins, 2)).astype(np.int64)
    for bin_position in range(1, n_bins):
        pfc_counts[:, bin_position, 0] += hpc_counts[:, bin_position - 1, 0]
        hpc_counts[:, bin_position, 1] += pfc_counts[:, bin_position - 1, 1]
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": 0,
            "correct": 1,
            "reward": 1,
            "action": 1,
            "choice_time": np.arange(n_trials, dtype=float) * 10.0,
            "cur_block": np.repeat(np.arange(10), 5),
        }
    )
    windows = AnalysisWindows(whole_start_s=-0.4, split_s=0.0, whole_stop_s=0.4)
    temporal = TemporalConfig(bin_size_s=0.1)
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
    )
    edges = windows.whole_start_s + np.arange(n_bins + 1) * temporal.bin_size_s
    edges[0], edges[-1] = windows.whole_start_s, windows.whole_stop_s
    rows = np.arange(n_trials, dtype=np.int64)
    labels = tuple(str(value) for value in trial_df.index)
    pfc_population = ResolvedRegionalPopulation(
        role="PFC",
        probe_id="pfc",
        channel_source="explicit",
        selected_channels=(0, 1),
        cluster_ids=(1, 2),
        unit_ids=("pfc:1", "pfc:2"),
    )
    hpc_population = ResolvedRegionalPopulation(
        role="HPC",
        probe_id="hpc",
        channel_source="explicit",
        selected_channels=(0, 1),
        cluster_ids=(3, 4),
        unit_ids=("hpc:3", "hpc:4"),
    )
    return pipeline.PreparedInterregionalSession(
        session_id="pc-synthetic",
        config=config,
        pfc_population=pfc_population,
        hpc_population=hpc_population,
        pfc_counts=RegionalCountTensor(
            pfc_counts, rows, labels, edges, pfc_population.unit_ids
        ),
        hpc_counts=RegionalCountTensor(
            hpc_counts, rows, labels, edges, hpc_population.unit_ids
        ),
        trial_masks=build_analysis_trial_masks(
            trial_df, alignment="choice_time", filters=config.filters
        ),
        fold_assignment=build_block_fold_assignment(trial_df),
    )


def test_bidirectional_unit_cv_detects_seeded_source_history_and_preserves_rows() -> None:
    """The deliberately coupled synthetic direction has stronger held-out increment."""
    rng = np.random.default_rng(17)
    n_trials, n_bins = 50, 8
    hpc_counts = rng.poisson(3.0, size=(n_trials, n_bins, 1)).astype(np.int64)
    pfc_counts = np.empty((n_trials, n_bins, 1), dtype=np.int64)
    pfc_counts[:, 0, 0] = rng.poisson(2.0, size=n_trials)
    for bin_position in range(1, n_bins):
        pfc_counts[:, bin_position, 0] = (
            2 * hpc_counts[:, bin_position - 1, 0]
            + rng.poisson(1.0, size=n_trials)
        )

    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": np.zeros(n_trials, dtype=int),
            "correct": np.ones(n_trials, dtype=int),
            "reward": np.ones(n_trials, dtype=int),
            "action": np.ones(n_trials, dtype=int),
            "choice_time": np.arange(n_trials, dtype=float) * 10.0,
            "cur_block": np.repeat(np.arange(10), 5),
        },
        index=np.arange(1000, 1000 + n_trials),
    )
    windows = AnalysisWindows(whole_start_s=-0.4, split_s=0.0, whole_stop_s=0.4)
    temporal = TemporalConfig(bin_size_s=0.1, lag_bins=1, order_bins=1)
    config = InterregionalAnalysisConfig(
        session_metadata_path="/data/session/neural_session.json",
        pfc_population=RegionalPopulationConfig(role="PFC", probe_id="pfc"),
        hpc_population=RegionalPopulationConfig(role="HPC", probe_id="hpc"),
        windows=windows,
        prediction_windows=("whole",),
        temporal=temporal,
        filters=FilterConfig(conditions=("all",)),
    )
    edges = windows.whole_start_s + np.arange(n_bins + 1, dtype=float) * temporal.bin_size_s
    edges[0], edges[-1] = windows.whole_start_s, windows.whole_stop_s
    trial_rows = np.arange(n_trials, dtype=np.int64)
    labels = tuple(str(value) for value in trial_df.index)
    pfc_tensor = RegionalCountTensor(
        pfc_counts, trial_rows, labels, edges, ("pfc:1",)
    )
    hpc_tensor = RegionalCountTensor(
        hpc_counts, trial_rows, labels, edges, ("hpc:2",)
    )
    prepared = pipeline.PreparedInterregionalSession(
        session_id="synthetic",
        config=config,
        pfc_population=_resolved("PFC", "pfc", 1),
        hpc_population=_resolved("HPC", "hpc", 2),
        pfc_counts=pfc_tensor,
        hpc_counts=hpc_tensor,
        trial_masks=build_analysis_trial_masks(
            trial_df, alignment="choice_time", filters=config.filters
        ),
        fold_assignment=build_block_fold_assignment(trial_df),
    )

    fold_scores, target_summaries, population_summaries = (
        pipeline.run_linear_cross_validation(prepared)
    )

    assert len(fold_scores) == 10
    assert set(fold_scores["fold_id"]) == set(range(5))
    assert fold_scores["status"].eq("ok").all()
    assert fold_scores["train_row_set_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert fold_scores["test_row_set_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    direction_means = target_summaries.loc[
        target_summaries["metric_name"].eq("delta_r2")
    ].set_index("direction")["mean_value"]
    assert direction_means["HPC_to_PFC"] > direction_means["PFC_to_HPC"]
    assert population_summaries["evaluation_scope"].eq("held_out_cv").all()


def test_full_rank_failure_preserves_independent_restricted_fit_status() -> None:
    """A full-only rank failure does not mislabel the estimable restricted design."""
    rng = np.random.default_rng(23)
    n_trials, n_bins = 25, 8
    shared_counts = rng.poisson(3.0, size=(n_trials, n_bins, 1)).astype(np.int64)
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": 0,
            "correct": 1,
            "reward": 1,
            "action": 1,
            "choice_time": np.arange(n_trials, dtype=float) * 10.0,
            "cur_block": np.repeat(np.arange(5), 5),
        }
    )
    windows = AnalysisWindows(whole_start_s=-0.4, split_s=0.0, whole_stop_s=0.4)
    temporal = TemporalConfig(bin_size_s=0.1)
    config = InterregionalAnalysisConfig(
        session_metadata_path="/data/session/neural_session.json",
        pfc_population=RegionalPopulationConfig(role="PFC", probe_id="pfc"),
        hpc_population=RegionalPopulationConfig(role="HPC", probe_id="hpc"),
        windows=windows,
        prediction_windows=("whole",),
        temporal=temporal,
        filters=FilterConfig(conditions=("all",)),
    )
    edges = windows.whole_start_s + np.arange(n_bins + 1) * temporal.bin_size_s
    edges[0], edges[-1] = windows.whole_start_s, windows.whole_stop_s
    rows = np.arange(n_trials, dtype=np.int64)
    labels = tuple(str(value) for value in trial_df.index)
    prepared = pipeline.PreparedInterregionalSession(
        session_id="rank-test",
        config=config,
        pfc_population=_resolved("PFC", "pfc", 1),
        hpc_population=_resolved("HPC", "hpc", 2),
        pfc_counts=RegionalCountTensor(
            shared_counts, rows, labels, edges, ("pfc:1",)
        ),
        hpc_counts=RegionalCountTensor(
            shared_counts.copy(), rows, labels, edges, ("hpc:2",)
        ),
        trial_masks=build_analysis_trial_masks(
            trial_df, alignment="choice_time", filters=config.filters
        ),
        fold_assignment=build_block_fold_assignment(trial_df),
    )

    fold_scores, _, _ = pipeline.run_linear_cross_validation(prepared)

    assert fold_scores["restricted_status"].eq("ok").all()
    assert fold_scores["restricted_reason"].eq("").all()
    assert fold_scores["full_status"].eq("fit_unavailable").all()
    assert fold_scores["full_reason"].eq("rank_deficient_full").all()
    assert fold_scores["status"].eq("fit_unavailable").all()


def test_pc_cv_is_bidirectional_and_leaves_unit_results_unchanged() -> None:
    """Adding fold-local PCs appends PC rows without perturbing direct-unit OLS."""
    units_only = _multiunit_prepared(("units",))
    units_and_pcs = _multiunit_prepared(("units", "pcs"))

    baseline_scores, baseline_targets, _ = pipeline.run_linear_cross_validation(
        units_only
    )
    combined_scores, combined_targets, _ = pipeline.run_linear_cross_validation(
        units_and_pcs
    )

    pd.testing.assert_frame_equal(
        baseline_scores,
        combined_scores.loc[combined_scores["representation"].eq("units")]
        .reset_index(drop=True)
        .astype(baseline_scores.dtypes.to_dict()),
    )
    pd.testing.assert_frame_equal(
        baseline_targets,
        combined_targets.loc[combined_targets["representation"].eq("units")]
        .reset_index(drop=True)
        .astype(baseline_targets.dtypes.to_dict()),
    )
    pc_scores = combined_scores.loc[combined_scores["representation"].eq("pcs")]
    assert set(pc_scores["direction"]) == {"HPC_to_PFC", "PFC_to_HPC"}
    assert set(pc_scores["target_id"]) == {
        "PFC:PC01",
        "PFC:PC02",
        "HPC:PC01",
        "HPC:PC02",
    }
    assert set(pc_scores["target_rank"].dropna().astype(int)) == {1, 2}
    assert len(pc_scores) == 20


def test_pc_cv_uses_the_same_fold_rows_as_unit_cv() -> None:
    """Representation changes do not change train/test observation identities."""
    prepared = _multiunit_prepared(("units", "pcs"))

    fold_scores, _, _ = pipeline.run_linear_cross_validation(prepared)

    identity_columns = [
        "direction",
        "condition",
        "window",
        "fold_id",
        "n_train_rows",
        "n_test_rows",
        "train_row_set_sha256",
        "test_row_set_sha256",
    ]
    units = (
        fold_scores.loc[fold_scores["representation"].eq("units"), identity_columns]
        .drop_duplicates()
        .sort_values(identity_columns[:4])
        .reset_index(drop=True)
    )
    pcs = (
        fold_scores.loc[fold_scores["representation"].eq("pcs"), identity_columns]
        .drop_duplicates()
        .sort_values(identity_columns[:4])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(units, pcs)


def test_descriptive_pca_transforms_are_rejected_by_cv() -> None:
    """An all-data PCA basis cannot be injected into held-out evaluation."""
    prepared = _multiunit_prepared(("pcs",))
    tensor_rows = prepared.pfc_counts.trial_rows
    conditions = {
        name: mask[tensor_rows]
        for name, mask in prepared.trial_masks.condition_masks.items()
    }
    descriptive = fit_descriptive_regional_pcas(
        pfc_activity=prepared.pfc_counts.counts,
        hpc_activity=prepared.hpc_counts.counts,
        pfc_unit_ids=prepared.pfc_counts.unit_ids,
        hpc_unit_ids=prepared.hpc_counts.unit_ids,
        scientific_trials=np.ones(len(tensor_rows), dtype=bool),
        condition_masks=conditions,
        requested_conditions=prepared.config.filters.conditions,
        requested_components=2,
    )

    with np.testing.assert_raises_regex(ValueError, "fold-scoped"):
        pipeline.run_linear_cross_validation(
            prepared, fold_pcas={fold_id: descriptive for fold_id in range(5)}
        )


def test_requested_but_unavailable_pc_rank_has_explicit_incomplete_rows() -> None:
    """A rank above the fold dimension remains visible rather than disappearing."""
    prepared = _multiunit_prepared(("pcs",))
    prepared = replace(
        prepared,
        config=replace(
            prepared.config, pca=PCAConfig(pfc_components=3, hpc_components=3)
        ),
    )

    fold_scores, target_summaries, _ = pipeline.run_linear_cross_validation(prepared)

    missing = fold_scores.loc[fold_scores["target_rank"].eq(3)]
    assert len(missing) == 10
    assert missing["status"].eq("fit_unavailable").all()
    assert missing["reason"].eq("pca_insufficient_components").all()
    missing_summaries = target_summaries.loc[target_summaries["target_rank"].eq(3)]
    assert missing_summaries["status"].eq("incomplete_folds").all()
    assert missing_summaries["valid_folds"].eq(0).all()


def test_fold_pca_is_fit_once_per_fold_and_emits_frozen_metadata(monkeypatch) -> None:
    """Conditions and windows reuse one regional transform pair for each fold."""
    prepared = _multiunit_prepared(("pcs",))
    prepared = replace(
        prepared,
        config=replace(
            prepared.config,
            prediction_windows=("before", "after"),
            filters=FilterConfig(conditions=("all", "switch")),
        ),
        trial_masks=replace(
            prepared.trial_masks,
            condition_masks={
                "all": prepared.trial_masks.condition_masks["all"],
                "switch": prepared.trial_masks.condition_masks["all"],
            },
        ),
    )
    original_fit = pipeline.fit_fold_regional_pcas
    fitted_fold_ids: list[int] = []

    def recording_fit(**kwargs):
        fitted_fold_ids.append(kwargs["fold_id"])
        return original_fit(**kwargs)

    monkeypatch.setattr(pipeline, "fit_fold_regional_pcas", recording_fit)

    fold_pcas, pca_fits = pipeline.fit_cross_validation_pcas(prepared)
    pipeline.run_linear_cross_validation(prepared, fold_pcas=fold_pcas)

    assert fitted_fold_ids == list(range(5))
    assert len(pca_fits) == 10
    assert pca_fits["scope"].eq("fold").all()
    assert set(pca_fits["region"]) == {"PFC", "HPC"}
    assert pca_fits["status"].eq("ok").all()
    assert pca_fits["n_training_observations"].gt(0).all()


def test_wp7_seeded_unit_and_pc_results_are_deterministic_with_explicit_gaps() -> None:
    """The complete standard model set is repeatable without hiding missing ranks."""
    prepared = _multiunit_prepared(("units", "pcs"))
    prepared = replace(
        prepared,
        config=replace(
            prepared.config,
            pca=PCAConfig(pfc_components=3, hpc_components=3),
        ),
    )

    def execute() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Fit fold-local transforms and return all standard OLS result tables."""
        fold_pcas, pca_fits = pipeline.fit_cross_validation_pcas(prepared)
        result_tables = pipeline.run_linear_cross_validation(
            prepared, fold_pcas=fold_pcas
        )
        return (*result_tables, pca_fits)

    first = execute()
    second = execute()

    for first_table, second_table in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(first_table, second_table)
    fold_scores, target_summaries, population_summaries, pca_fits = first
    assert {"units", "pcs"} == set(fold_scores["representation"])
    assert set(fold_scores["model_family"]) == {"ols"}
    unit_statuses = fold_scores.loc[
        fold_scores["representation"].eq("units"), "status"
    ]
    assert unit_statuses.eq("ok").all()
    unavailable_rank = fold_scores.loc[
        fold_scores["representation"].eq("pcs") & fold_scores["target_rank"].eq(3)
    ]
    assert len(unavailable_rank) == 10
    assert unavailable_rank["status"].eq("fit_unavailable").all()
    assert unavailable_rank["reason"].eq("pca_insufficient_components").all()
    assert target_summaries.loc[
        target_summaries["target_rank"].eq(3), "status"
    ].eq("incomplete_folds").all()
    assert not population_summaries.empty
    assert len(pca_fits) == 10


def _poisson_prepared() -> pipeline.PreparedInterregionalSession:
    """Return the multiunit fixture with matched OLS and Poisson CV requested."""
    prepared = _multiunit_prepared(("units",))
    return replace(
        prepared,
        config=replace(prepared.config, analyses=("ols_cv", "poisson_cv")),
    )


def test_poisson_cv_reuses_every_unit_ols_fold_and_history_identity() -> None:
    """Poisson targets use the exact OLS unit folds, rows, and design dimensions."""
    prepared = _poisson_prepared()

    ols_scores, _, _ = pipeline.run_linear_cross_validation(prepared)
    poisson_scores, _, _ = pipeline.run_poisson_cross_validation(prepared)

    identity = [
        "session_id",
        "direction",
        "condition",
        "window",
        "target_id",
        "fold_id",
        "evaluation_scope",
        "n_train_trials",
        "n_test_trials",
        "n_train_rows",
        "n_test_rows",
        "train_row_set_sha256",
        "test_row_set_sha256",
        "restricted_feature_count",
        "full_feature_count",
        "restricted_rank",
        "full_rank",
        "restricted_df_resid",
        "full_df_resid",
    ]
    pd.testing.assert_frame_equal(
        ols_scores.loc[:, identity].sort_values(identity[:6]).reset_index(drop=True),
        poisson_scores.loc[:, identity]
        .sort_values(identity[:6])
        .reset_index(drop=True),
    )
    assert poisson_scores["representation"].eq("units").all()
    assert poisson_scores["model_family"].eq("poisson").all()


def test_poisson_cv_local_fit_failure_does_not_abort_other_targets(monkeypatch) -> None:
    """One recognized target fit error remains local while later fits continue."""
    prepared = _poisson_prepared()
    original_fit = pipeline.fit_poisson_target
    call_count = 0

    def fail_once(design, count_response):
        """Fail the first target fit, then delegate every independent fit."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise PoissonFitUnavailable("poisson_nonconverged", "synthetic")
        return original_fit(design, count_response)

    monkeypatch.setattr(pipeline, "fit_poisson_target", fail_once)

    fold_scores, _, _ = pipeline.run_poisson_cross_validation(prepared)

    failures = fold_scores.loc[fold_scores["restricted_status"].ne("ok")]
    assert len(failures) == 1
    assert failures.iloc[0]["restricted_reason"] == "poisson_nonconverged_restricted"
    assert fold_scores["full_status"].eq("ok").all()
    assert call_count == 40


def test_poisson_cv_allows_unexpected_fit_errors_to_reach_run_boundary(monkeypatch) -> None:
    """Programming or API errors are not converted into scientific status rows."""
    prepared = _poisson_prepared()

    def unexpected_error(design, count_response):
        """Represent an unexpected bug below the orchestration boundary."""
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pipeline, "fit_poisson_target", unexpected_error)

    with np.testing.assert_raises_regex(RuntimeError, "unexpected"):
        pipeline.run_poisson_cross_validation(prepared)


def test_poisson_cv_summaries_require_complete_folds_and_exclude_raw_deviance() -> None:
    """Complete normalized metrics aggregate, while raw deviance stays fold-only."""
    prepared = _poisson_prepared()

    fold_scores, target_summaries, population_summaries = (
        pipeline.run_poisson_cross_validation(prepared)
    )

    assert fold_scores["status"].eq("ok").all()
    primary = target_summaries.loc[
        target_summaries["metric_name"].eq("delta_deviance_explained")
    ]
    assert not primary.empty
    assert primary["status"].eq("ok").all()
    assert primary["valid_folds"].eq(5).all()
    assert population_summaries.loc[
        population_summaries["metric_name"].eq("delta_deviance_explained"),
        "n_targets",
    ].gt(0).all()
    assert not {
        "deviance_restricted",
        "deviance_full",
        "null_deviance",
    } & set(target_summaries["metric_name"])


def test_poisson_fold_rows_keep_only_poisson_diagnostics_and_count_mse() -> None:
    """Poisson rows populate convergence/deviance/MSE fields but no OLS R-squared."""
    fold_scores, _, _ = pipeline.run_poisson_cross_validation(_poisson_prepared())

    assert fold_scores["restricted_converged"].eq(True).all()
    assert fold_scores["full_converged"].eq(True).all()
    assert fold_scores["restricted_iterations"].gt(0).all()
    assert fold_scores["full_iterations"].gt(0).all()
    assert fold_scores[["r2_restricted", "r2_full", "delta_r2"]].isna().all().all()
    assert fold_scores[
        [
            "mse_restricted",
            "mse_full",
            "deviance_restricted",
            "deviance_full",
            "null_deviance",
            "deviance_explained_restricted",
            "deviance_explained_full",
            "delta_deviance_explained",
        ]
    ].notna().all().all()


def test_progress_callbacks_preserve_ols_and_poisson_scientific_outputs() -> None:
    """Execution-only callbacks do not alter any seeded result-table value."""
    prepared = _poisson_prepared()
    baseline_ols = pipeline.run_linear_cross_validation(prepared)
    baseline_poisson = pipeline.run_poisson_cross_validation(prepared)
    linear_events: list[dict[str, object]] = []
    poisson_events: list[dict[str, object]] = []

    instrumented_ols = pipeline.run_linear_cross_validation(
        prepared, progress_callback=linear_events.append
    )
    instrumented_poisson = pipeline.run_poisson_cross_validation(
        prepared, progress_callback=poisson_events.append
    )

    for baseline, instrumented in zip(
        (*baseline_ols, *baseline_poisson),
        (*instrumented_ols, *instrumented_poisson),
        strict=True,
    ):
        pd.testing.assert_frame_equal(baseline, instrumented)
    assert linear_events
    assert poisson_events


def test_poisson_progress_identifies_cells_shapes_bytes_and_target_ranges() -> None:
    """Poisson progress has enough exact context to localize retained memory."""
    events: list[dict[str, object]] = []

    pipeline.run_poisson_cross_validation(
        _poisson_prepared(), progress_callback=events.append
    )

    starts = [event for event in events if event["event"] == "analysis_cell_start"]
    ends = [event for event in events if event["event"] == "analysis_cell_end"]
    targets = [
        event for event in events if event["event"] == "poisson_target_progress"
    ]
    assert len(starts) == 10
    assert len(ends) == 10
    first = starts[0]
    assert first | {
        "model_family": "poisson",
        "representation": "units",
        "direction": "HPC_to_PFC",
        "condition": "all",
        "window": "whole",
        "fold_id": 0,
        "cell_index": 1,
        "total_cells": 10,
        "total_targets": 2,
    } == first
    for shape_key, byte_key in (
        ("train_response_shape", "train_response_bytes"),
        ("test_response_shape", "test_response_bytes"),
        ("restricted_train_shape", "restricted_train_bytes"),
        ("full_train_shape", "full_train_bytes"),
        ("restricted_test_shape", "restricted_test_bytes"),
        ("full_test_shape", "full_test_bytes"),
    ):
        shape = first[shape_key]
        assert isinstance(shape, list) and len(shape) == 2
        assert first[byte_key] == int(np.prod(shape, dtype=np.int64)) * 8
    assert [
        event["completed_targets"]
        for event in targets
        if event["cell_index"] == 1
    ] == [1, 2]
    assert ends[0]["completed_targets"] == 2
    assert isinstance(ends[0]["unavailable_targets"], int)
    assert ends[0]["unavailable_targets"] >= 0


def test_poisson_target_progress_cadence_is_first_every_25_and_final() -> None:
    """The coarse target cadence avoids per-fit logging on large populations."""
    reported = [
        completed
        for completed in range(1, 52)
        if pipeline._target_progress_due(completed, 51)
    ]

    assert reported == [1, 25, 50, 51]
