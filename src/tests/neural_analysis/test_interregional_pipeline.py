"""Synthetic integration tests for the direct-unit inter-regional OLS path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    TemporalConfig,
)
from src.neural_analysis.interregional.preparation import (
    build_analysis_trial_masks,
    build_block_fold_assignment,
)
from src.neural_analysis.interregional.records import RegionalCountTensor
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
