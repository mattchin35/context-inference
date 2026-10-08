"""Synthetic integration tests for the direct-unit inter-regional OLS path."""

from __future__ import annotations

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
