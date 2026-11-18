"""Tests for trial-local inter-regional count and design preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    ResolvedRegionalPopulation,
    TemporalConfig,
)
from src.neural_analysis.interregional import preparation


def _resolved_population(
    role: str = "PFC",
    probe_id: str = "probe_a",
    cluster_ids: tuple[int, ...] = (7, 3),
) -> ResolvedRegionalPopulation:
    """Return a small explicit population in requested unit order."""
    return ResolvedRegionalPopulation(
        role=role,
        probe_id=probe_id,
        channel_source="explicit",
        selected_channels=(0,),
        cluster_ids=tuple(sorted(cluster_ids)),
        unit_ids=tuple(f"{probe_id}:{cluster_id}" for cluster_id in sorted(cluster_ids)),
    )


def test_pynapple_count_tensor_uses_half_open_bins_and_positional_rows() -> None:
    """Counts preserve trial/bin/unit axes and exclude the configured final edge."""
    trial_df = pd.DataFrame(
        {"choice_time": [10.0, 20.0]},
        index=pd.Index(["trial-a", "trial-b"], name="source_index"),
    )
    spike_group = nap.TsGroup(
        {
            3: nap.Ts(t=np.array([9.8, 9.9, 10.0, 10.1, 10.2, 19.85, 20.05])),
            7: nap.Ts(t=np.array([9.95, 10.15, 19.8, 19.9, 20.0, 20.1, 20.2])),
        }
    )
    population = _resolved_population(cluster_ids=(3, 7))

    tensor = preparation.build_regional_count_tensor(
        spike_group=spike_group,
        population=population,
        trial_df=trial_df,
        trial_rows=np.array([0, 1], dtype=np.int64),
        alignment="choice_time",
        windows=AnalysisWindows(whole_start_s=-0.2, split_s=0.0, whole_stop_s=0.2),
        temporal=TemporalConfig(bin_size_s=0.1),
    )

    assert tensor.counts.dtype == np.int64
    assert tensor.counts.shape == (2, 4, 2)
    assert tensor.trial_rows.tolist() == [0, 1]
    assert tensor.original_index_labels == ("trial-a", "trial-b")
    assert tensor.unit_ids == ("probe_a:3", "probe_a:7")
    np.testing.assert_array_equal(
        tensor.counts,
        np.array(
            [
                [[1, 0], [1, 1], [1, 0], [1, 1]],
                [[1, 1], [0, 1], [1, 1], [0, 1]],
            ],
            dtype=np.int64,
        ),
    )
    expected_edges = -0.2 + np.arange(5, dtype=np.float64) * 0.1
    expected_edges[0], expected_edges[-1] = -0.2, 0.2
    np.testing.assert_array_equal(tensor.bin_edges_s, expected_edges)


def test_count_tensor_rejects_nonascending_rows_and_invalid_alignment() -> None:
    """Prepared trial identity is ascending, positional, unique, and finite."""
    trial_df = pd.DataFrame({"choice_time": [1.0, np.nan]}, index=[10, 20])
    support = nap.IntervalSet(start=0.0, end=30.0)
    spike_group = nap.TsGroup({3: nap.Ts(t=np.array([1.0]), time_support=support)})
    population = _resolved_population(cluster_ids=(3,))
    kwargs = {
        "spike_group": spike_group,
        "population": population,
        "trial_df": trial_df,
        "alignment": "choice_time",
        "windows": AnalysisWindows(),
        "temporal": TemporalConfig(),
    }

    with pytest.raises(ValueError, match="ascending"):
        preparation.build_regional_count_tensor(
            trial_rows=np.array([1, 0], dtype=np.int64), **kwargs
        )
    with pytest.raises(ValueError, match="alignment"):
        preparation.build_regional_count_tensor(
            trial_rows=np.array([1], dtype=np.int64), **kwargs
        )


def test_regional_tensors_require_identical_trial_and_bin_axes() -> None:
    """PFC and HPC comparisons share byte-identical trial and time axes."""
    trial_df = pd.DataFrame({"choice_time": [10.0]})
    support = nap.IntervalSet(start=0.0, end=30.0)
    spike_group = nap.TsGroup({3: nap.Ts(t=np.array([10.0]), time_support=support)})
    population = _resolved_population(cluster_ids=(3,))
    tensor = preparation.build_regional_count_tensor(
        spike_group,
        population,
        trial_df,
        np.array([0], dtype=np.int64),
        "choice_time",
        AnalysisWindows(),
        TemporalConfig(),
    )
    preparation.validate_regional_tensor_axes(tensor, tensor)

    shifted = preparation.RegionalCountTensor(
        counts=tensor.counts,
        trial_rows=tensor.trial_rows,
        original_index_labels=tensor.original_index_labels,
        bin_edges_s=tensor.bin_edges_s + 0.1,
        unit_ids=tensor.unit_ids,
    )
    with pytest.raises(ValueError, match="bin"):
        preparation.validate_regional_tensor_axes(tensor, shifted)


def test_analysis_masks_separate_scientific_condition_and_cv_eligibility() -> None:
    """Missing blocks affect CV only; filters and exclusions affect scientific eligibility."""
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": [0, 0, 0, 1, 0, 0],
            "correct": [1, 1, 0, 1, 1, 1],
            "reward": [1, 0, 0, 1, 1, 1],
            "action": [1, 1, 1, 1, 0, 1],
            "state_int": [0, 0, 1, 0, 0, 0],
            "choice_time": [1.0, 2.0, 3.0, 4.0, np.nan, 6.0],
            "cur_block": [0, None, 1, 1, 2, 2],
        },
        index=[100, 200, 300, 400, 500, 600],
    )
    filters = FilterConfig(
        conditions=("all", "correct_rewarded", "incorrect"),
        choice="left",
        context="right",
        excluded_trial_rows=(5,),
    )

    masks = preparation.build_analysis_trial_masks(
        trial_df, alignment="choice_time", filters=filters
    )

    np.testing.assert_array_equal(
        masks.scientific_eligible,
        np.array([True, True, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks.condition_masks["correct_rewarded"],
        np.array([True, False, False, False, False, False]),
    )
    np.testing.assert_array_equal(
        masks.cv_eligible,
        np.array([True, False, False, False, False, False]),
    )
    assert masks.original_index_labels == ("100", "200", "300", "400", "500", "600")


def test_selected_choice_or_context_filter_requires_its_trial_column() -> None:
    """A selected side filter never silently produces an empty analysis."""
    trial_df = pd.DataFrame(
        {
            "experimenter_reward_given": [0],
            "correct": [1],
            "reward": [1],
            "choice_time": [1.0],
            "cur_block": [0],
        }
    )
    with pytest.raises(ValueError, match="action"):
        preparation.build_analysis_trial_masks(
            trial_df,
            alignment="choice_time",
            filters=FilterConfig(conditions=("all",), choice="left"),
        )
    with pytest.raises(ValueError, match="state_int"):
        preparation.build_analysis_trial_masks(
            trial_df,
            alignment="choice_time",
            filters=FilterConfig(conditions=("all",), context="right"),
        )


def test_block_folds_are_deterministic_grouped_and_preserve_missing_rows() -> None:
    """Normalized block identities produce one deterministic five-fold mapping."""
    blocks = pd.Series(
        [1, 1.0, "1", "1", 2.5, 2.5, "a", "a", 3, 3, None], dtype=object
    )
    trial_df = pd.DataFrame(
        {"cur_block": blocks.to_numpy()}, index=np.arange(10, 21)
    )

    first = preparation.build_block_fold_assignment(trial_df)
    second = preparation.build_block_fold_assignment(trial_df)

    assert first.fold_ids == second.fold_ids
    assert first.block_values_json[0] == "1"
    assert first.block_values_json[1] == "1"
    assert first.block_values_json[2] == '"1"'
    assert first.fold_ids[-1] is None
    for block_json in set(first.block_values_json[:-1]):
        block_folds = {
            fold
            for value, fold in zip(first.block_values_json, first.fold_ids, strict=True)
            if value == block_json
        }
        assert len(block_folds) == 1


def test_block_folds_reject_too_few_or_invalid_groups() -> None:
    """CV has no random fallback and accepts only canonical scalar block values."""
    with pytest.raises(ValueError, match="five"):
        preparation.build_block_fold_assignment(
            pd.DataFrame({"cur_block": [0, 0, 1, 1]})
        )
    with pytest.raises(ValueError, match="block"):
        preparation.build_block_fold_assignment(
            pd.DataFrame({"cur_block": [0, 1, 2, 3, True]})
        )


def test_window_selection_and_histories_preserve_whole_bin_identity() -> None:
    """Histories stay within trials and retain whole-window target-bin positions."""
    target = np.arange(2 * 4 * 2, dtype=float).reshape(2, 4, 2)
    source = np.arange(100, 100 + 2 * 4, dtype=float).reshape(2, 4, 1)
    before, before_positions = preparation.select_window_bins(
        target,
        "before",
        AnalysisWindows(whole_start_s=-0.2, split_s=0.0, whole_stop_s=0.2),
        TemporalConfig(bin_size_s=0.1),
    )
    assert before.shape == (2, 2, 2)
    np.testing.assert_array_equal(before_positions, np.array([0, 1], dtype=np.int64))

    histories = preparation.build_history_matrices(
        target_activity=target,
        source_activity=source,
        trial_rows=np.array([4, 9], dtype=np.int64),
        target_bin_positions=np.arange(4, dtype=np.int64),
        lag_bins=1,
        order_bins=2,
    )
    assert histories.responses.shape == (4, 2)
    np.testing.assert_array_equal(histories.row_trial, np.array([4, 4, 9, 9]))
    np.testing.assert_array_equal(histories.row_target_bin, np.array([2, 3, 2, 3]))
    np.testing.assert_array_equal(histories.target_history[0], np.r_[target[0, 1], target[0, 0]])
    np.testing.assert_array_equal(histories.source_history[0], np.r_[source[0, 1], source[0, 0]])


def test_default_history_row_counts_and_row_fingerprints() -> None:
    """Default histories yield 19/39 rows and fingerprints describe identity sets."""
    two_second = preparation.build_history_matrices(
        np.zeros((1, 20, 1)),
        np.zeros((1, 20, 2)),
        np.array([8], dtype=np.int64),
        np.arange(20, dtype=np.int64),
        lag_bins=1,
        order_bins=1,
    )
    four_second = preparation.build_history_matrices(
        np.zeros((1, 40, 1)),
        np.zeros((1, 40, 2)),
        np.array([8], dtype=np.int64),
        np.arange(40, dtype=np.int64),
        lag_bins=1,
        order_bins=1,
    )
    assert two_second.responses.shape[0] == 19
    assert four_second.responses.shape[0] == 39

    first = preparation.fingerprint_row_identities(
        np.array([2, 1], dtype=np.int64), np.array([3, 4], dtype=np.int64)
    )
    second = preparation.fingerprint_row_identities(
        np.array([1, 2], dtype=np.int64), np.array([4, 3], dtype=np.int64)
    )
    assert first == second
    with pytest.raises(ValueError, match="duplicate"):
        preparation.fingerprint_row_identities(
            np.array([1, 1], dtype=np.int64), np.array([4, 4], dtype=np.int64)
        )
