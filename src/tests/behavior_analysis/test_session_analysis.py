from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from types import ModuleType
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

formulaic_stub = ModuleType("formulaic")
formulaic_stub.model_matrix = lambda *_args, **_kwargs: None
sys.modules.setdefault("formulaic", formulaic_stub)
statsmodels_stub = ModuleType("statsmodels")
statsmodels_api_stub = ModuleType("statsmodels.api")
statsmodels_stub.api = statsmodels_api_stub
sys.modules.setdefault("statsmodels", statsmodels_stub)
sys.modules.setdefault("statsmodels.api", statsmodels_api_stub)

import src.behavior_analysis.session_analysis as session_analysis


def make_simulated_trial_df() -> pd.DataFrame:
    """Build a small switched-run trial table using the simulation schema."""
    return pd.DataFrame(
        {
            "state": ["right", "right", "left", "left", "right", "right"],
            "model_stimulus": [0, -1, 1, -1, 0, -1],
            "action": [0, 1, 1, 0, 0, 0],
            "reward": [1, 0, 1, 0, 1, 0],
            "correct": [1, 0, 1, 0, 1, 1],
            "cur_block": [0, 0, 1, 1, 2, 2],
            "cur_trial": [0, 1, 2, 3, 4, 5],
            "cur_trial_in_block": [0, 1, 0, 1, 0, 1],
        }
    )


def test_get_block_types_accepts_simulated_schema():
    """Simulated runs should map `model_stimulus` and `left/right` states."""
    block_types = session_analysis.get_block_types(make_simulated_trial_df())

    assert block_types.tolist() == [
        "right_cued",
        "right_uncued",
        "left_cued",
        "left_uncued",
        "right_cued",
        "right_uncued",
    ]


def test_count_decision_variables_defaults_give_reward_to_zero_when_missing():
    """Decision-variable counting should treat missing `give_reward` as all-zero."""
    decision_vars = session_analysis.count_decision_variables(make_simulated_trial_df())

    assert decision_vars.shape[0] == 6
    assert "relative_value" in decision_vars.columns
    assert decision_vars["relative_value"].iloc[0] == 0


def test_count_decision_variables_preserves_existing_output():
    """Decision variables should still describe trial history before each trial."""
    trial_df = pd.DataFrame(
        {
            "action": [0, 0, 1, 1, 0],
            "reward": [1, 0, 1, 0, 1],
            "give_reward": [0, 0, 0, 0, 1],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    expected_decision_vars = pd.DataFrame(
        {
            "negative_value": [0, -1, 0, -1, 0],
            "consecutive_rewards_memory": [0, 1, 1, 1, 1],
            "consecutive_omissions_memory": [0, 0, 1, 1, 1],
            "consecutive_rewards": [0, 1, 0, 1, 0],
            "consecutive_omissions": [0, 0, 1, 0, 1],
            "left_value": [0, 0, 0, 1, 0],
            "right_value": [0, 1, 0, 0, 0],
            "relative_value": [0, -1, 0, 1, 0],
            "left_omissions": [0, 0, 0, 0, 1],
            "right_omissions": [0, 0, 1, 1, 1],
            "relative_omissions": [0, 0, -1, -1, 0],
            "left_cf_value": [0, 0, 0, 1, 0],
            "right_cf_value": [0, 1, 0, 0, 0],
            "relative_cf_value": [0, -1, 0, 1, 0],
            "left_cf_omissions": [0, 0, 0, 0, 1],
            "right_cf_omissions": [0, 0, 1, 0, 0],
            "relative_cf_omissions": [0, 0, -1, 0, 1],
            "left_monotonic_cf_value": [0, 0, 0, 1, 1],
            "right_monotonic_cf_value": [0, 1, 1, 0, 0],
            "relative_monotonic_cf_value": [0, -1, -1, 1, 1],
        }
    )

    pd.testing.assert_frame_equal(decision_vars, expected_decision_vars)


def test_compute_agent_mouse_agreement_by_trial_uses_repeat_on_tie():
    """Agent agreement should use the ideal-observer repeat-on-tie rule."""
    trial_df = pd.DataFrame(
        {
            "action": [0, 1, 1, 0, 0],
            "reward": [0, 0, 0, 0, 0],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
            "agent_value": [0.0, 1.0, 0.0, -1.0, 0.0],
        }
    )

    agreement = session_analysis.compute_agent_mouse_agreement_by_trial(
        trial_df,
        value_column="agent_value",
    )

    assert agreement.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_compute_session_side_bias_metrics_uses_valid_behavioral_choices_only():
    """Session side-bias ratios should exclude no-choice and manual-reward rows."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "left", "right", "right", "left", "right", "left"],
            "action": [1, 0, 1, 1, "no_choice", 0, 1],
            "observer_value": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 1],
        }
    )

    metrics = session_analysis.compute_session_side_bias_metrics(trial_df)

    assert metrics["n_left_choices"] == 3
    assert metrics["n_right_choices"] == 2
    assert metrics["n_true_left_trials"] == 2
    assert metrics["n_true_right_trials"] == 3
    assert metrics["n_ideal_left_trials"] == 2
    assert metrics["n_ideal_right_trials"] == 3
    assert metrics["left_choice_per_true_left"] == 1.5
    assert metrics["right_choice_per_true_right"] == 2 / 3
    assert metrics["left_choice_per_ideal_left"] == 1.5
    assert metrics["right_choice_per_ideal_right"] == 2 / 3


def test_compute_session_side_bias_metrics_uses_ideal_observer_tie_convention():
    """Ideal-action denominators should repeat the previous greedy choice on ties."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "left", "left", "right"],
            "action": [0, 1, 1, 0],
            "observer_value": [0.0, 1.0, 0.0, -1.0],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    metrics = session_analysis.compute_session_side_bias_metrics(trial_df)

    assert metrics["n_left_choices"] == 2
    assert metrics["n_right_choices"] == 2
    assert metrics["n_ideal_left_trials"] == 2
    assert metrics["n_ideal_right_trials"] == 2
    assert metrics["left_choice_per_ideal_left"] == 1.0
    assert metrics["right_choice_per_ideal_right"] == 1.0


def test_compute_session_side_bias_metrics_returns_none_for_zero_denominator():
    """Ratios with no matching reference-side trials should use the string sentinel."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "left"],
            "action": [1, 1],
            "observer_value": [1.0, 1.0],
            "experimenter_reward_given": [0, 0],
        }
    )

    metrics = session_analysis.compute_session_side_bias_metrics(trial_df)

    assert metrics["n_true_right_trials"] == 0
    assert metrics["n_ideal_right_trials"] == 0
    assert metrics["right_choice_per_true_right"] == "None"
    assert metrics["right_choice_per_ideal_right"] == "None"


def test_compute_session_side_bias_metrics_adds_signed_bias_metrics():
    """Signed side-bias metrics should use valid behavioral choices only."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "right", "right", "left", "right", "left"],
            "action": [1, 1, 0, "no_choice", 0, 1],
            "observer_value": [1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 1],
        }
    )

    metrics = session_analysis.compute_session_side_bias_metrics(trial_df)

    assert metrics["bias_oracle"] == pytest.approx((2 - 1) / 4)
    assert metrics["bias_ideal"] == pytest.approx((2 - 2) / 4)
    assert metrics["raw_side_bias"] == pytest.approx((2 - 2) / 4)


def test_compute_session_side_bias_metrics_uses_matched_valid_ideal_trials_for_ideal_bias():
    """Ideal-relative bias should exclude rows without a valid ideal choice."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "left", "left"],
            "action": [1, 1, 0],
            "observer_value": [1.0, "None", -1.0],
            "experimenter_reward_given": [0, 0, 0],
        }
    )

    metrics = session_analysis.compute_session_side_bias_metrics(trial_df)

    assert metrics["bias_oracle"] == pytest.approx((2 - 3) / 3)
    assert metrics["bias_ideal"] == pytest.approx((1 - 1) / 2)


def test_compute_session_block_quality_metrics_summarizes_numeric_block_values():
    """Block summary metrics should ignore missing sentinels and use eligible rows."""
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": [0, 2, 7, "None"],
            "percent_correct_after_first_correct": [1.0, 0.5, "None", 0.75],
            "block_history_ideal_mouse_agreement": [0.9, 0.4, "None", 0.7],
        }
    )

    metrics = session_analysis.compute_session_block_quality_metrics(
        block_performance,
        tts_threshold=5,
        post_switch_correct_threshold=0.7,
        ideal_agreement_threshold=0.6,
    )

    assert metrics["median_TTS"] == pytest.approx(2.0)
    assert metrics["q3_TTS"] == pytest.approx(4.5)
    assert metrics["frac_blocks_TTS_gt_5"] == pytest.approx(1 / 3)
    assert metrics["median_post_switch_correct"] == pytest.approx(0.75)
    assert metrics["q1_post_switch_correct"] == pytest.approx(0.625)
    assert metrics["frac_blocks_post_switch_correct_lt_0p7"] == pytest.approx(1 / 3)
    assert metrics["median_ideal_agreement"] == pytest.approx(0.7)
    assert metrics["q1_ideal_agreement"] == pytest.approx(0.55)
    assert metrics["frac_blocks_ideal_agreement_lt_0p6"] == pytest.approx(1 / 3)


def test_compute_session_block_quality_metrics_returns_none_without_eligible_rows():
    """Block summary metrics should use the CSV sentinel when no eligible rows exist."""
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": ["None"],
            "percent_correct_after_first_correct": ["None"],
            "block_history_ideal_mouse_agreement": ["None"],
        }
    )

    metrics = session_analysis.compute_session_block_quality_metrics(block_performance)

    assert metrics["median_TTS"] == "None"
    assert metrics["q3_TTS"] == "None"
    assert metrics["frac_blocks_TTS_gt_5"] == "None"
    assert metrics["median_post_switch_correct"] == "None"
    assert metrics["q1_post_switch_correct"] == "None"
    assert metrics["frac_blocks_post_switch_correct_lt_0p7"] == "None"
    assert metrics["median_ideal_agreement"] == "None"
    assert metrics["q1_ideal_agreement"] == "None"
    assert metrics["frac_blocks_ideal_agreement_lt_0p6"] == "None"


def test_compute_post_switch_ideal_agreement_summary_includes_first_correct_choice():
    """Post-switch ideal agreement should include the first correct behavioral choice."""
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 1, 1],
            "action": [0, 1, 1, 1, 0],
            "reward": [0, 0, 0, 0, 0],
            "correct": [0, 1, 1, 1, 0],
            "observer_value": [-1.0, 1.0, 1.0, 1.0, 1.0],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["left_uncued", "left_uncued"],
        }
    )

    metrics = session_analysis.compute_post_switch_ideal_agreement_summary(
        augmented_trial_df,
        block_performance,
    )

    assert metrics["median_post_switch_ideal_agreement"] == pytest.approx(0.75)
    assert metrics["q1_post_switch_ideal_agreement"] == pytest.approx(0.625)


def test_summarize_session_performance_includes_new_session_summary_metrics(monkeypatch):
    """Whole-session summaries should expose the new scalar quality metrics."""
    monkeypatch.setattr(
        session_analysis.ideal_observer,
        "summarize_ideal_observer_behavior",
        lambda *_args, **_kwargs: {},
    )
    augmented_trial_df = pd.DataFrame(
        {
            "state": ["left", "right", "right"],
            "block_type": ["left_uncued", "right_uncued", "right_uncued"],
            "action": [1, 0, 1],
            "reward": [1, 1, 0],
            "correct": [1, 1, 0],
            "cur_block": [0, 1, 1],
            "observer_value": [1.0, -1.0, 1.0],
            "experimenter_reward_given": [0, 0, 0],
        }
    )
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["left_uncued", "right_uncued"],
            "trials_to_correct": [0, 0],
            "percent_correct_after_first_correct": [1.0, 0.5],
            "block_history_ideal_mouse_agreement": [1.0, 0.5],
        }
    )

    summary = session_analysis.summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date="2026-06-30",
        ideal_observer_n_replays=1,
    )
    row = summary.iloc[0]

    assert row["bias_oracle"] == pytest.approx((2 - 1) / 3)
    assert row["bias_ideal"] == pytest.approx((2 - 2) / 3)
    assert row["raw_side_bias"] == pytest.approx((2 - 1) / 3)
    assert row["median_TTS"] == 0
    assert row["frac_blocks_TTS_gt_5"] == 0
    assert row["median_post_switch_correct"] == pytest.approx(0.75)
    assert row["median_ideal_agreement"] == pytest.approx(0.75)
    assert "median_post_switch_ideal_agreement" in row.index


def test_compute_zero_trials_to_correct_metrics_excludes_never_correct_blocks():
    """Never-correct and dark-period blocks should not enter zero-TTC fractions."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3, 4],
            "block_type": ["right_uncued", "left_uncued", "left_cued", "right_cued", "dark period"],
            "trials_to_correct": [0, 0, "None", 3, 0],
        }
    )

    metrics = session_analysis.compute_zero_trials_to_correct_metrics(block_performance)

    assert metrics["left_zero_ttc_n_blocks"] == 1
    assert metrics["left_zero_ttc_n_zero"] == 1
    assert metrics["right_zero_ttc_n_blocks"] == 1
    assert metrics["right_zero_ttc_n_zero"] == 0
    assert metrics["overall_zero_ttc_n_blocks"] == 2
    assert metrics["overall_zero_ttc_n_zero"] == 1
    assert metrics["overall_zero_ttc_fraction"] == 0.5


def test_compute_zero_trials_to_correct_metrics_splits_left_right_overall():
    """Zero-TTC fractions should summarize left, right, and pooled block changes."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3, 4],
            "block_type": ["right_uncued", "left_uncued", "left_cued", "right_cued", "right_uncued"],
            "trials_to_correct": [4, 0, 2, 0, 5],
        }
    )

    metrics = session_analysis.compute_zero_trials_to_correct_metrics(block_performance)

    assert metrics["left_zero_ttc_fraction"] == 0.5
    assert metrics["right_zero_ttc_fraction"] == 0.5
    assert metrics["overall_zero_ttc_fraction"] == 0.5
    assert metrics["left_zero_ttc_n_blocks"] == 2
    assert metrics["right_zero_ttc_n_blocks"] == 2
    assert metrics["overall_zero_ttc_n_blocks"] == 4


def test_compute_zero_trials_to_correct_metrics_excludes_first_block_as_non_switch():
    """The first block should not contribute because it is not a block change."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["left_uncued", "left_uncued", "left_uncued"],
            "trials_to_correct": [0, 2, 4],
        }
    )

    metrics = session_analysis.compute_zero_trials_to_correct_metrics(block_performance)

    assert metrics["left_zero_ttc_n_blocks"] == 2
    assert metrics["left_zero_ttc_n_zero"] == 0
    assert metrics["left_zero_ttc_fraction"] == 0.0


def test_compute_agent_mouse_agreement_by_trial_excludes_non_behavioral_rows():
    """No-choice and experimenter-reward rows should not enter agreement."""
    trial_df = pd.DataFrame(
        {
            "action": [0, "no_choice", 1, 1],
            "reward": [0, 0, 1, 0],
            "experimenter_reward_given": [0, 0, 1, 0],
            "agent_value": [-1.0, 1.0, 1.0, -1.0],
        }
    )

    agreement = session_analysis.compute_agent_mouse_agreement_by_trial(
        trial_df,
        value_column="agent_value",
    )

    assert agreement.tolist() == [1.0, "None", "None", 0.0]


def test_add_block_agent_mouse_agreement_columns_summarizes_by_block():
    """Agent agreement columns should summarize valid trialwise agreement by block."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [10, 11],
            "block_type": ["right_cued", "left_cued"],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [10, 10, 11, 11],
            "action": [0, 1, 1, 0],
            "reward": [0, 0, 0, 0],
            "experimenter_reward_given": [0, 0, 0, 0],
            "agent_value": [-1.0, -1.0, 1.0, 1.0],
        }
    )

    updated = session_analysis.add_block_agent_mouse_agreement_columns(
        block_performance,
        augmented_trial_df,
        agent_value_columns={"agent_mouse_agreement": "agent_value"},
    )

    assert updated["agent_mouse_agreement"].tolist() == [0.5, 0.5]


def test_add_block_agent_mouse_agreement_columns_requires_missing_agent_columns():
    """Requested agent value columns should fail loudly when absent."""
    block_performance = pd.DataFrame({"block_ix": [0], "block_type": ["right_cued"]})
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0],
            "action": [0],
            "reward": [0],
            "experimenter_reward_given": [0],
        }
    )

    with pytest.raises(ValueError, match="missing_value"):
        session_analysis.add_block_agent_mouse_agreement_columns(
            block_performance,
            augmented_trial_df,
            agent_value_columns={"missing_agent_agreement": "missing_value"},
        )


def test_monotonic_counterfactual_value_counts_rewards_and_resets_opposite_side():
    """Monotonic CF values should count rewarded-side rewards and reset the other side."""
    trial_df = pd.DataFrame(
        {
            "action": [0, 0, 1, 1, 0],
            "reward": [1, 1, 1, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    assert decision_vars["left_monotonic_cf_value"].tolist() == [0, 0, 0, 1, 2]
    assert decision_vars["right_monotonic_cf_value"].tolist() == [0, 1, 2, 0, 0]
    assert decision_vars["relative_monotonic_cf_value"].tolist() == [0, -1, -2, 1, 2]


def test_monotonic_counterfactual_value_ignores_omissions():
    """Omissions should not decrement the monotonic CF reward counter."""
    trial_df = pd.DataFrame(
        {
            "action": [0, 0, 0, 1],
            "reward": [1, 0, 0, 0],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    assert decision_vars["right_monotonic_cf_value"].tolist() == [0, 1, 1, 1]
    assert decision_vars["left_monotonic_cf_value"].tolist() == [0, 0, 0, 0]
    assert decision_vars["relative_monotonic_cf_value"].tolist() == [0, -1, -1, -1]


def test_monotonic_counterfactual_value_skips_manual_reward_and_no_choice_trials():
    """Manual rewards and no-choice trials should not update monotonic CF values."""
    trial_df = pd.DataFrame(
        {
            "action": [0, "no_choice", 1],
            "reward": [1, 1, 1],
            "experimenter_reward_given": [0, 1, 0],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    assert decision_vars["relative_monotonic_cf_value"].tolist() == [0, -1, -1]


def test_count_decision_variables_parses_string_choice_actions():
    """CSV-loaded string actions should update integer-coded decision counters."""
    trial_df = pd.DataFrame(
        {
            "action": ["0", "0", "1"],
            "reward": [1, 1, 1],
            "experimenter_reward_given": [0, 0, 0],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    assert decision_vars["right_monotonic_cf_value"].tolist() == [0, 1, 2]
    assert decision_vars["left_monotonic_cf_value"].tolist() == [0, 0, 0]
    assert decision_vars["relative_monotonic_cf_value"].tolist() == [0, -1, -2]


def test_count_decision_variables_skips_no_choice_but_updates_string_choices():
    """A no-choice row should not prevent neighboring string choices from updating."""
    trial_df = pd.DataFrame(
        {
            "action": ["0", "no_choice", "1"],
            "reward": [1, 1, 1],
            "experimenter_reward_given": [0, 1, 0],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    assert decision_vars["right_monotonic_cf_value"].tolist() == [0, 1, 1]
    assert decision_vars["left_monotonic_cf_value"].tolist() == [0, 0, 0]
    assert decision_vars["relative_monotonic_cf_value"].tolist() == [0, -1, -1]
    assert decision_vars["relative_cf_omissions"].tolist() == [0, 0, 0]


def test_parse_decision_variable_update_values_rejects_invalid_action_after_skip():
    """Invalid actions should fail clearly if they reach the counter update parser."""
    with pytest.raises(ValueError, match="action must be 0 or 1"):
        session_analysis.parse_decision_variable_update_values(
            action="bad",
            reward=1,
        )


def test_count_decision_variables_uses_experimenter_reward_given_column():
    """New trial tables should skip manual rewards using the explicit flag name."""
    trial_df = pd.DataFrame(
        {
            "action": [0, "no_choice", 1],
            "reward": [1, 1, 1],
            "experimenter_reward_given": [0, 1, 0],
        }
    )

    decision_vars = session_analysis.count_decision_variables(trial_df)

    expected_without_manual_reward = session_analysis.count_decision_variables(
        pd.DataFrame(
            {
                "action": [0, 1],
                "reward": [1, 1],
                "experimenter_reward_given": [0, 0],
            }
        )
    )
    pd.testing.assert_series_equal(
        decision_vars.loc[[0, 2], "relative_value"].reset_index(drop=True),
        expected_without_manual_reward["relative_value"],
    )
    assert decision_vars.loc[1, "relative_value"] == decision_vars.loc[2, "relative_value"]


def test_append_decision_variables_records_pre_trial_state():
    """Appending should record state before the current trial updates counters."""
    state = session_analysis.DecisionVariableState(
        negative_value=2,
        consecutive_rewards_memory=3,
        consecutive_omissions_memory=4,
        consecutive_rewards=5,
        consecutive_omissions=6,
        left_value=7,
        right_value=2,
        left_omissions=1,
        right_omissions=8,
        left_value_cf=9,
        right_value_cf=3,
        left_omissions_cf=4,
        right_omissions_cf=10,
        left_monotonic_cf_value=11,
        right_monotonic_cf_value=5,
    )
    decision_variable_dict = defaultdict(list)

    session_analysis.append_decision_variables(decision_variable_dict, state)

    assert decision_variable_dict["negative_value"] == [2]
    assert decision_variable_dict["relative_value"] == [5]
    assert decision_variable_dict["relative_omissions"] == [-7]
    assert decision_variable_dict["relative_cf_value"] == [6]
    assert decision_variable_dict["relative_cf_omissions"] == [-6]
    assert decision_variable_dict["relative_monotonic_cf_value"] == [6]


def test_should_skip_decision_variable_update_skips_experimenter_reward_and_no_choice_action():
    """Manual rewards and no-choice trials should not update history counters."""
    assert session_analysis.should_skip_decision_variable_update(experimenter_reward_given=1, action=0)
    assert session_analysis.should_skip_decision_variable_update(experimenter_reward_given="1", action=0)
    assert session_analysis.should_skip_decision_variable_update(experimenter_reward_given=0, action="None")
    assert session_analysis.should_skip_decision_variable_update(experimenter_reward_given=0, action="no_choice")
    assert not session_analysis.should_skip_decision_variable_update(experimenter_reward_given=0, action=0)
    assert not session_analysis.should_skip_decision_variable_update(experimenter_reward_given="0", action=1)


def test_update_decision_variable_state_matches_counter_helpers():
    """A state update should match the direct decision-variable counter calls."""
    state = session_analysis.DecisionVariableState()

    updated_state = session_analysis.update_decision_variable_state(
        state,
        action=0,
        reward=1,
    )

    assert updated_state.negative_value == session_analysis.counters.negative_value_counter(0, 1)
    assert updated_state.consecutive_rewards == session_analysis.counters.consecutive_reward_counter(0, 1)
    assert updated_state.consecutive_omissions == session_analysis.counters.consecutive_fail_counter(0, 1)
    assert updated_state.consecutive_rewards_memory == session_analysis.counters.consecutive_reward_renewal_counter(0, 1, False)
    assert updated_state.consecutive_omissions_memory == session_analysis.counters.consecutive_fail_renewal_counter(0, 1, False)
    assert updated_state.previous_trial_rewarded is True
    assert (updated_state.left_value, updated_state.right_value) == (
        session_analysis.counters.sided_value_counter(0, 0, 0, 1, zero_min=True, counterfactual=False)
    )
    assert (updated_state.left_value_cf, updated_state.right_value_cf) == (
        session_analysis.counters.sided_value_counter(0, 0, 0, 1, zero_min=True, counterfactual=True)
    )
    assert (updated_state.left_monotonic_cf_value, updated_state.right_monotonic_cf_value) == (
        session_analysis.counters.sided_value_counter(
            0,
            0,
            0,
            1,
            zero_min=True,
            counterfactual=True,
            monotonic=True,
        )
    )


def test_get_block_switches_defaults_give_reward_to_zero_when_missing():
    """Block-switch counting should not require a mouse-only `give_reward` column."""
    n_switches, normalized_switches = session_analysis.get_block_switches(
        make_simulated_trial_df().iloc[:3]
    )

    assert n_switches == 1
    assert normalized_switches == 0.5


def test_get_block_switches_skips_no_choice_manual_reward_rows():
    """Manual reward rows should not be cast as choices or counted as switches."""
    trial_df = pd.DataFrame(
        {
            "action": [0, "no_choice", 1],
            "experimenter_reward_given": [0, 1, 0],
        }
    )

    n_switches, normalized_switches = session_analysis.get_block_switches(trial_df)

    assert n_switches == 1
    assert normalized_switches == 0.5


def test_analyze_session_handles_missing_choice_time_and_start_time():
    """Simulated runs without latency columns should yield NaN latency summaries."""
    _, block_performance, augmented_trial_df = session_analysis.analyze_session(
        make_simulated_trial_df(),
        mouse="sample_switch_mode_3",
        date="2026-03-25",
    )

    assert "time_to_choice" in augmented_trial_df.columns
    assert augmented_trial_df["time_to_choice"].isna().all()
    assert block_performance["mean_choice_time"].isna().all()
    assert block_performance["median_choice_time"].isna().all()
    assert block_performance["std_choice_time"].isna().all()


def test_analyze_session_skips_experimenter_reward_rows_in_behavioral_summaries():
    """Manual reward no-choice rows should not count as choices or latency rows."""
    trial_df = pd.DataFrame(
        {
            "state": ["right_patch", "right_patch", "left_patch"],
            "block_stimulus": ["None", "None", "None"],
            "action": [0, "no_choice", 1],
            "reward": [1, 1, 1],
            "correct": [1, 0, 1],
            "experimenter_reward_given": [0, 1, 0],
            "choice_time": [2.5, "None", 4.5],
            "start_time": [2.0, 3.0, 4.0],
            "cur_block": [0, 0, 1],
            "cur_trial": [0, 1, 2],
            "cur_trial_in_block": [0, 1, 0],
        }
    )

    _, block_performance, augmented_trial_df = session_analysis.analyze_session(
        trial_df,
        mouse="CT999",
        date="2026-05-27",
    )

    assert pd.isna(augmented_trial_df.loc[1, "time_to_choice"])
    assert block_performance.loc[0, "n_correct"] == 1
    assert block_performance.loc[0, "n_rewarded"] == 1
    assert block_performance.loc[0, "percent_correct"] == 1.0
    assert block_performance.loc[0, "mean_choice_time"] == 0.5


def test_percent_correct_accepts_string_encoded_correct_values():
    """Behavior summaries should handle CSV-style string 0/1 correctness."""
    augmented_trial_df = pd.DataFrame(
        {
            "block_type": ["right_cued", "right_cued", "left_uncued", "left_uncued"],
            "action": ["0", "1", "1", "0"],
            "correct": ["1", "0", "1", "0"],
            "experimenter_reward_given": ["0", "0", "0", "0"],
        }
    )

    performance = session_analysis.percent_correct(augmented_trial_df)

    assert performance["right_cued_correct"] == 0.5
    assert performance["left_uncued_correct"] == 0.5
    assert performance["overall_correct"] == 0.5


def test_summarize_block_performance_accepts_string_encoded_correct_values():
    """Block summaries should coerce string correctness before counting."""
    augmented_trial_df = session_analysis.make_augmented_trial_df(
        make_simulated_trial_df().astype({"correct": str})
    )

    block_performance = session_analysis.summarize_block_performance(
        augmented_trial_df,
        session_id="sample_switch_mode_3_2026-03-25",
    )

    assert block_performance["n_correct"].tolist() == [1.0, 1.0, 2.0]
    assert block_performance["percent_correct"].tolist() == [0.5, 0.5, 1.0]


def test_summarize_block_performance_sums_string_encoded_rewards_numerically():
    """Block reward counts should coerce CSV-style string rewards before summing."""
    augmented_trial_df = session_analysis.make_augmented_trial_df(
        make_simulated_trial_df().astype({"reward": str})
    )

    block_performance = session_analysis.summarize_block_performance(
        augmented_trial_df,
        session_id="sample_switch_mode_3_2026-03-25",
    )

    assert block_performance["n_rewarded"].tolist() == [1.0, 1.0, 1.0]
    assert block_performance["prev_n_rewarded"].tolist() == [0, 1.0, 1.0]


def test_summarize_block_performance_counts_explore_trials_by_block():
    """Block summaries should save the number of explore-tagged trials per block."""
    trial_df = make_simulated_trial_df()
    trial_df["explore_trial"] = [1, 0, True, "true", 0, "1"]
    augmented_trial_df = session_analysis.make_augmented_trial_df(trial_df)

    block_performance = session_analysis.summarize_block_performance(
        augmented_trial_df,
        session_id="sample_switch_mode_3_2026-03-25",
    )

    assert block_performance["n_explore_trials"].tolist() == [1, 2, 1]


def test_summarize_block_performance_saves_percent_correct_after_first_correct():
    """Post-first-correct accuracy should include the first correct behavioral trial."""
    trial_df = pd.DataFrame(
        {
            "state": ["right", "right", "right", "right", "right"],
            "model_stimulus": [0, 0, 0, 0, 0],
            "action": [1, 1, 0, 0, 1],
            "reward": [0, 0, 1, 1, 0],
            "correct": [0, 0, 1, 1, 0],
            "cur_block": [0, 0, 0, 0, 0],
            "cur_trial": [0, 1, 2, 3, 4],
            "cur_trial_in_block": [0, 1, 2, 3, 4],
        }
    )

    _, block_performance, _ = session_analysis.analyze_session(
        trial_df,
        mouse="CT999",
        date="2026-06-25",
    )

    assert block_performance.loc[0, "first_correct_trial_in_block"] == 2
    assert block_performance.loc[0, "n_trials_after_first_correct"] == 3
    assert block_performance.loc[0, "percent_correct_after_first_correct"] == pytest.approx(2 / 3)


def test_summarize_block_performance_marks_no_correct_post_first_correct_metrics_missing():
    """Blocks without a correct choice should keep post-first-correct metrics missing."""
    trial_df = pd.DataFrame(
        {
            "state": ["left", "left", "left"],
            "model_stimulus": [0, 0, 0],
            "action": [0, 0, 0],
            "reward": [0, 0, 0],
            "correct": [0, 0, 0],
            "cur_block": [0, 0, 0],
            "cur_trial": [0, 1, 2],
            "cur_trial_in_block": [0, 1, 2],
        }
    )

    _, block_performance, _ = session_analysis.analyze_session(
        trial_df,
        mouse="CT999",
        date="2026-06-25",
    )

    assert block_performance.loc[0, "first_correct_trial_in_block"] == "None"
    assert block_performance.loc[0, "n_trials_after_first_correct"] == "None"
    assert block_performance.loc[0, "percent_correct_after_first_correct"] == "None"


def test_summarize_block_performance_saves_block_history_ideal_mouse_agreement():
    """Block summaries should save mouse agreement with the greedy history-ideal policy."""
    trial_df = pd.DataFrame(
        {
            "state": ["right", "right", "right", "right"],
            "model_stimulus": [0, 0, 0, 0],
            "action": [1, 1, 1, 0],
            "reward": [0, 0, 0, 1],
            "correct": [0, 0, 0, 1],
            "cur_block": [0, 0, 0, 0],
            "cur_trial": [0, 1, 2, 3],
            "cur_trial_in_block": [0, 1, 2, 3],
            "HMM_rel_value_logodds_decay": [1.0, -1.0, 1.0, -1.0],
            "relative_doubt_index": [0.0, 0.0, 0.0, 0.0],
        }
    )

    _, block_performance, _ = session_analysis.analyze_session(
        trial_df,
        mouse="CT999",
        date="2026-06-26",
    )

    assert block_performance.loc[0, "block_history_ideal_mouse_agreement"] == pytest.approx(0.75)


def test_summarize_block_performance_marks_missing_history_ideal_agreement():
    """Blocks without valid behavioral choices should keep ideal agreement missing."""
    trial_df = pd.DataFrame(
        {
            "state": ["right", "right"],
            "model_stimulus": [0, 0],
            "action": ["no_choice", "no_choice"],
            "reward": [1, 1],
            "correct": [0, 0],
            "experimenter_reward_given": [1, 1],
            "cur_block": [0, 0],
            "cur_trial": [0, 1],
            "cur_trial_in_block": [0, 1],
            "HMM_rel_value_logodds_decay": [1.0, -1.0],
            "relative_doubt_index": [0.0, 0.0],
        }
    )

    _, block_performance, _ = session_analysis.analyze_session(
        trial_df,
        mouse="CT999",
        date="2026-06-26",
    )

    assert block_performance.loc[0, "block_history_ideal_mouse_agreement"] == "None"


def test_percent_correct_rejects_invalid_correct_for_behavioral_choice():
    """Malformed correctness values on animal-choice rows should fail clearly."""
    augmented_trial_df = pd.DataFrame(
        {
            "block_type": ["right_cued"],
            "action": [0],
            "correct": ["None"],
            "experimenter_reward_given": [0],
        }
    )

    with pytest.raises(ValueError, match="correct values must be numeric"):
        session_analysis.percent_correct(augmented_trial_df)


def test_summarize_block_performance_matches_analyze_session_output():
    """Block-performance helper should preserve analyze_session block output."""
    trial_df = make_simulated_trial_df()
    _, expected_block_performance, _ = session_analysis.analyze_session(
        trial_df,
        mouse="sample_switch_mode_3",
        date="2026-03-25",
    )
    augmented_trial_df = session_analysis.make_augmented_trial_df(trial_df)

    block_performance = session_analysis.summarize_block_performance(
        augmented_trial_df,
        session_id="sample_switch_mode_3_2026-03-25",
    )

    pd.testing.assert_frame_equal(block_performance, expected_block_performance)


def test_summarize_session_performance_matches_analyze_session_output():
    """Session-performance helper should preserve analyze_session session output."""
    trial_df = make_simulated_trial_df()
    expected_session_performance, expected_block_performance, augmented_trial_df = (
        session_analysis.analyze_session(
            trial_df,
            mouse="sample_switch_mode_3",
            date="2026-03-25",
        )
    )

    session_performance = session_analysis.summarize_session_performance(
        augmented_trial_df,
        expected_block_performance,
        date="2026-03-25",
    )

    pd.testing.assert_frame_equal(session_performance, expected_session_performance)


def test_summarize_session_performance_includes_oracle_behavior_metrics():
    """Session summaries should include oracle accuracy and reward collection fields."""
    trial_df = make_simulated_trial_df().assign(
        p_active_rew=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9],
        p_inactive_rew=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        p_switch=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    )
    _, block_performance, augmented_trial_df = session_analysis.analyze_session(
        trial_df,
        mouse="sample_switch_mode_3",
        date="2026-03-25",
    )

    session_performance = session_analysis.summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date="2026-03-25",
    )

    expected_columns = {
        "oracle_choice_accuracy",
        "actual_reward_collected",
        "oracle_expected_reward",
        "oracle_reward_fraction",
        "oracle_reward_difference",
    }
    assert expected_columns.issubset(session_performance.columns)
    assert session_performance.loc[0, "oracle_choice_accuracy"] == 4 / 6
    assert session_performance.loc[0, "actual_reward_collected"] == 3.0
    assert session_performance.loc[0, "oracle_expected_reward"] == 4.8
    assert session_performance.loc[0, "oracle_reward_fraction"] == 3.0 / 4.8
    assert np.isclose(session_performance.loc[0, "oracle_reward_difference"], -1.8)


def test_summarize_session_performance_includes_rewarded_switch_metric():
    """Session summaries should include rewarded next-trial switch probability."""
    trial_df = make_simulated_trial_df().assign(
        p_active_rew=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9],
        p_inactive_rew=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        p_switch=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    )
    _, block_performance, augmented_trial_df = session_analysis.analyze_session(
        trial_df,
        mouse="sample_switch_mode_3",
        date="2026-03-25",
    )

    session_performance = session_analysis.summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date="2026-03-25",
    )

    assert "rewarded_choice_switch_probability" in session_performance.columns
    assert "rewarded_choice_switch_n_trials" in session_performance.columns
    assert session_performance.loc[0, "rewarded_choice_switch_probability"] == 2 / 3
    assert session_performance.loc[0, "rewarded_choice_switch_n_trials"] == 3


def test_summarize_session_performance_includes_ideal_observer_metrics():
    """Session summaries should include history and fixed-replay ideal metrics."""
    trial_df = make_simulated_trial_df().assign(
        p_active_rew=[0.8, 0.8, 0.7, 0.7, 0.9, 0.9],
        p_inactive_rew=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        p_switch=[0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
    )
    _, block_performance, augmented_trial_df = session_analysis.analyze_session(
        trial_df,
        mouse="sample_switch_mode_3",
        date="2026-03-25",
        ideal_observer_n_replays=4,
        ideal_observer_seed=123,
    )

    session_performance = session_analysis.summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date="2026-03-25",
        ideal_observer_n_replays=4,
        ideal_observer_seed=123,
    )

    expected_columns = {
        "history_ideal_oracle_accuracy",
        "history_ideal_mouse_agreement",
        "history_ideal_expected_reward",
        "history_ideal_reward_fraction",
        "history_ideal_reward_difference",
        "fixed_replay_ideal_n_replays",
        "fixed_replay_ideal_reward_mean",
        "fixed_replay_ideal_reward_q1",
        "fixed_replay_ideal_reward_median",
        "fixed_replay_ideal_reward_q3",
        "fixed_replay_ideal_expected_reward_mean",
        "fixed_replay_ideal_oracle_accuracy_mean",
        "fixed_replay_ideal_oracle_accuracy_q1",
        "fixed_replay_ideal_oracle_accuracy_median",
        "fixed_replay_ideal_oracle_accuracy_q3",
    }
    assert expected_columns.issubset(session_performance.columns)
    assert session_performance.loc[0, "fixed_replay_ideal_n_replays"] == 4


def test_summarize_session_performance_includes_side_bias_metrics():
    """Session summaries should include whole-session side-bias ratios."""
    augmented_trial_df = pd.DataFrame(
        {
            "state": ["left", "left", "right", "right"],
            "action": [1, 0, 1, 0],
            "correct": [1, 0, 0, 1],
            "reward": [1, 0, 0, 1],
            "observer_value": [1.0, -1.0, 1.0, -1.0],
            "experimenter_reward_given": [0, 0, 0, 0],
            "block_type": ["left_uncued", "left_uncued", "right_uncued", "right_uncued"],
        }
    )
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": [0, 1],
            "block_type": ["left_uncued", "right_uncued"],
        }
    )

    session_performance = session_analysis.summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date="2026-03-25",
        ideal_observer_n_replays=1,
        ideal_observer_seed=123,
    )

    assert session_performance.loc[0, "n_left_choices"] == 2
    assert session_performance.loc[0, "n_right_choices"] == 2
    assert session_performance.loc[0, "left_choice_per_true_left"] == 1.0
    assert session_performance.loc[0, "right_choice_per_true_right"] == 1.0
    assert session_performance.loc[0, "left_choice_per_ideal_left"] == 1.0
    assert session_performance.loc[0, "right_choice_per_ideal_right"] == 1.0


def test_summarize_session_performance_includes_zero_trials_to_correct_metrics():
    """Session summaries should include immediate-correction block fractions."""
    augmented_trial_df = pd.DataFrame(
        {
            "state": ["left", "right"],
            "action": [1, 0],
            "correct": [1, 1],
            "reward": [1, 1],
            "observer_value": [1.0, -1.0],
            "experimenter_reward_given": [0, 0],
            "block_type": ["left_uncued", "right_uncued"],
        }
    )
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "trials_to_correct": [0, 0, 3, "None"],
            "block_type": ["left_uncued", "left_uncued", "right_uncued", "right_uncued"],
        }
    )

    session_performance = session_analysis.summarize_session_performance(
        augmented_trial_df,
        block_performance,
        date="2026-03-25",
        ideal_observer_n_replays=1,
        ideal_observer_seed=123,
    )

    assert session_performance.loc[0, "left_zero_ttc_fraction"] == 1.0
    assert session_performance.loc[0, "right_zero_ttc_fraction"] == 0.0
    assert session_performance.loc[0, "overall_zero_ttc_fraction"] == 0.5
    assert session_performance.loc[0, "overall_zero_ttc_n_blocks"] == 2
    assert session_performance.loc[0, "overall_zero_ttc_n_zero"] == 1


def test_session_stats_returns_none_stats_for_one_valid_row():
    """Regression stats are undefined with fewer than two paired observations."""
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        stats = session_analysis.session_stats(
            dependent_var=pd.Series([2]),
            independent_var=pd.Series([1]),
        )

    assert stats == ("None", "None", "None", "None")
    assert recorded_warnings == []


def test_session_stats_returns_none_stats_for_empty_inputs():
    """Regression stats are undefined with zero paired observations."""
    stats = session_analysis.session_stats(
        dependent_var=pd.Series([], dtype=float),
        independent_var=pd.Series([], dtype=float),
    )

    assert stats == ("None", "None", "None", "None")


def test_add_regression_stats_to_session_performance_uses_explicit_column_names():
    """Session regression summaries should name the regressor used for each stat."""
    trial_df = make_simulated_trial_df()
    session_performance, block_performance, _ = session_analysis.analyze_session(
        trial_df,
        mouse="sample_switch_mode_3",
        date="2026-03-25",
    )

    session_performance = session_analysis.add_regression_stats_to_session_performance(
        session_performance=session_performance,
        trials_to_correct=block_performance["trials_to_correct"],
        prev_n_correct=block_performance["prev_n_correct"],
        prev_consecutive_rewards=block_performance["prev_consecutive_rewards"],
        prev_n_rewarded=block_performance["prev_n_rewarded"],
        n_blocks=block_performance.shape[0],
    )

    expected_columns = {
        "prev_n_rewarded_slope",
        "prev_n_rewarded_intercept",
        "prev_n_rewarded_r_value",
        "prev_n_rewarded_p_value",
        "prev_consecutive_rewards_slope",
        "prev_consecutive_rewards_intercept",
        "prev_consecutive_rewards_r_value",
        "prev_consecutive_rewards_p_value",
        "prev_n_correct_slope",
        "prev_n_correct_intercept",
        "prev_n_correct_r_value",
        "prev_n_correct_p_value",
        "n_blocks",
    }
    removed_columns = {"slope", "intercept", "r_value", "p_value", "n_switches"}

    assert expected_columns.issubset(session_performance.columns)
    assert removed_columns.isdisjoint(session_performance.columns)
    assert session_performance.loc[0, "n_blocks"] == block_performance.shape[0]

    valid_rows = (
        (block_performance["trials_to_correct"] != "None")
        & (block_performance["prev_n_correct"] != "None")
    )
    expected_rewards_stats = session_analysis.session_stats(
        dependent_var=block_performance.loc[valid_rows, "trials_to_correct"].astype(int),
        independent_var=block_performance.loc[valid_rows, "prev_consecutive_rewards"].astype(int),
    )
    expected_prev_n_rewarded_stats = session_analysis.session_stats(
        dependent_var=block_performance.loc[valid_rows, "trials_to_correct"].astype(int),
        independent_var=block_performance.loc[valid_rows, "prev_n_rewarded"].astype(int),
    )
    expected_correct_stats = session_analysis.session_stats(
        dependent_var=block_performance.loc[valid_rows, "trials_to_correct"].astype(int),
        independent_var=block_performance.loc[valid_rows, "prev_n_correct"].astype(int),
    )

    actual_prev_n_rewarded_stats = tuple(
        session_performance.loc[
            0,
            [
                "prev_n_rewarded_slope",
                "prev_n_rewarded_intercept",
                "prev_n_rewarded_r_value",
                "prev_n_rewarded_p_value",
            ],
        ]
    )
    actual_rewards_stats = tuple(
        session_performance.loc[
            0,
            [
                "prev_consecutive_rewards_slope",
                "prev_consecutive_rewards_intercept",
                "prev_consecutive_rewards_r_value",
                "prev_consecutive_rewards_p_value",
            ],
        ]
    )
    actual_correct_stats = tuple(
        session_performance.loc[
            0,
            [
                "prev_n_correct_slope",
                "prev_n_correct_intercept",
                "prev_n_correct_r_value",
                "prev_n_correct_p_value",
            ],
        ]
    )

    np.testing.assert_allclose(
        np.asarray(actual_prev_n_rewarded_stats, dtype=float),
        np.asarray(expected_prev_n_rewarded_stats, dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual_rewards_stats, dtype=float),
        np.asarray(expected_rewards_stats, dtype=float),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        np.asarray(actual_correct_stats, dtype=float),
        np.asarray(expected_correct_stats, dtype=float),
        equal_nan=True,
    )


def test_add_regression_stats_to_session_performance_drops_non_numeric_prev_n_rewarded_rows(monkeypatch):
    """Stale nonnumeric reward-count rows should not crash regression summaries."""
    session_performance = pd.DataFrame({"date": ["2026-06-30"]})
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": [0, 2, 4],
            "prev_n_correct": [1, 2, 3],
            "prev_consecutive_rewards": [1, 2, 3],
            "prev_n_rewarded": [1, "11111111111111111111111111111111111111111111111111", 3],
        }
    )
    regression_inputs = []

    def fake_session_stats(dependent_var, independent_var):
        regression_inputs.append(
            (
                dependent_var.to_numpy(dtype=float).tolist(),
                independent_var.to_numpy(dtype=float).tolist(),
            )
        )
        return 0.0, 0.0, 0.0, 1.0

    monkeypatch.setattr(session_analysis, "session_stats", fake_session_stats)

    updated = session_analysis.add_regression_stats_to_session_performance(
        session_performance=session_performance,
        trials_to_correct=block_performance["trials_to_correct"],
        prev_n_correct=block_performance["prev_n_correct"],
        prev_consecutive_rewards=block_performance["prev_consecutive_rewards"],
        prev_n_rewarded=block_performance["prev_n_rewarded"],
        n_blocks=block_performance.shape[0],
    )

    assert updated.loc[0, "prev_n_rewarded_slope"] == 0.0
    assert regression_inputs[0] == ([0.0, 4.0], [1.0, 3.0])
    assert regression_inputs[1] == ([0.0, 2.0, 4.0], [1.0, 2.0, 3.0])
    assert regression_inputs[2] == ([0.0, 2.0, 4.0], [1.0, 2.0, 3.0])


def test_run_analysis_handles_string_encoded_rewards_in_prev_n_rewarded_regression(tmp_path, monkeypatch):
    """String reward inputs should not produce concatenated previous reward counts."""
    trial_df = make_simulated_trial_df().astype({"reward": str})
    session = SimpleNamespace(
        mouse="sample_switch_mode_3",
        date="2026-03-25",
        sess_id_full="sample_switch_mode_3_2026-03-25_120000_agent-sample_switch_mode_3",
        processed_data_path=tmp_path,
        multi_session_save_path=None,
    )
    monkeypatch.setattr(session_analysis, "session_stats", lambda *_args, **_kwargs: (0.0, 0.0, 0.0, 1.0))

    _, block_performance, returned_session_df = session_analysis.run_analysis(
        trial_df,
        session=session,
    )

    assert block_performance["prev_n_rewarded"].tolist() == [0, 1.0, 1.0]
    assert "prev_n_rewarded_slope" in returned_session_df.columns


def test_add_block_bias_columns_preserves_current_bias_calculation():
    """Block-bias helper should preserve the existing bias columns and sentinels."""
    _, block_performance, _ = session_analysis.analyze_session(
        make_simulated_trial_df(),
        mouse="sample_switch_mode_3",
        date="2026-03-25",
    )
    block_performance["trials_to_correct"] = block_performance["trials_to_correct"].astype(str)
    block_performance["prev_n_correct"] = block_performance["prev_n_correct"].astype(str)
    block_performance.loc[1, "trials_to_correct"] = "None"

    block_performance = session_analysis.add_block_bias_columns(block_performance)

    expected_columns = {
        "bias_rl",
        "bias_inf",
        "min_value_bias",
        "bias_rl_flag",
        "bias_inf_flag",
        "bias_full_flag",
    }
    assert expected_columns.issubset(block_performance.columns)

    valid_rows = (
        (block_performance["trials_to_correct"] != "None")
        & (block_performance["prev_n_correct"] != "None")
    )
    expected_bias_rl = (
        (
            block_performance.loc[valid_rows, "trials_to_correct"].astype(int)
            - block_performance.loc[valid_rows, "prev_n_correct"].astype(int)
        )
        / (
            block_performance.loc[valid_rows, "trials_to_correct"].astype(int)
            + block_performance.loc[valid_rows, "prev_n_correct"].astype(int)
            + session_analysis.eps
        )
    )
    expected_bias_inf = (
        (block_performance.loc[valid_rows, "trials_to_correct"].astype(int) - 5)
        / (block_performance.loc[valid_rows, "trials_to_correct"].astype(int) + 5)
    )

    pd.testing.assert_series_equal(
        block_performance.loc[valid_rows, "bias_rl"],
        expected_bias_rl,
        check_names=False,
        check_dtype=False,
    )
    pd.testing.assert_series_equal(
        block_performance.loc[valid_rows, "bias_inf"],
        expected_bias_inf,
        check_names=False,
        check_dtype=False,
    )
    expected_min_value_bias = pd.concat([expected_bias_rl, expected_bias_inf], axis=1).min(axis=1)
    pd.testing.assert_series_equal(
        block_performance.loc[valid_rows, "min_value_bias"],
        expected_min_value_bias,
        check_names=False,
        check_dtype=False,
    )
    assert block_performance.loc[~valid_rows, "bias_rl"].tolist() == ["None"]
    assert block_performance.loc[~valid_rows, "min_value_bias"].tolist() == ["None"]
    assert block_performance.loc[~valid_rows, "bias_full_flag"].tolist() == ["None"]


def test_add_block_bias_columns_saves_rl_status_columns():
    """RL status columns should be added without replacing legacy bias values."""
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": [0, 1, 2, "None"],
            "prev_n_correct": [0, 0, 0, 1],
        }
    )

    block_performance = session_analysis.add_block_bias_columns(block_performance)

    expected_columns = {
        "rl_effective_prev_n_correct",
        "rl_thresh",
        "rl_thresh_flag",
        "bias_rl_status_value",
        "rl_status",
    }
    assert expected_columns.issubset(block_performance.columns)
    assert block_performance["rl_effective_prev_n_correct"].tolist() == [1, 1, 1, "None"]
    assert block_performance["rl_thresh"].tolist() == [0.0, 0.0, 0.0, "None"]
    assert block_performance["rl_thresh_flag"].tolist() == [True, True, True, "None"]


def test_add_block_bias_columns_zero_previous_correct_gets_one_trial_grace_period():
    """Zero previous-correct blocks should be classified using an effective value of one."""
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": [1, 2],
            "prev_n_correct": [0, 0],
        }
    )

    block_performance = session_analysis.add_block_bias_columns(block_performance)

    assert block_performance["rl_status"].tolist() == ["valid_rl", "biased_rl"]
    assert block_performance["bias_rl_status_value"].iloc[0] == pytest.approx(0.0)
    assert block_performance["bias_rl_status_value"].iloc[1] == pytest.approx(1 / 3)


def test_add_block_bias_columns_marks_below_rl_threshold():
    """Blocks below the RL threshold should be distinct from valid or biased RL."""
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": [1, 2, 4],
            "prev_n_correct": [4, 4, 4],
        }
    )

    block_performance = session_analysis.add_block_bias_columns(block_performance)

    assert block_performance["rl_thresh"].tolist() == [2.0, 2.0, 2.0]
    assert block_performance["rl_thresh_flag"].tolist() == [False, True, True]
    assert block_performance["rl_status"].tolist() == [
        "below_rl_threshold",
        "valid_rl",
        "valid_rl",
    ]


def test_add_block_bias_columns_keeps_invalid_rl_status_rows_as_none():
    """Invalid trials-to-correct rows should receive None sentinels for RL status."""
    block_performance = pd.DataFrame(
        {
            "trials_to_correct": ["None"],
            "prev_n_correct": [0],
        }
    )

    block_performance = session_analysis.add_block_bias_columns(block_performance)

    assert block_performance.loc[0, "rl_effective_prev_n_correct"] == "None"
    assert block_performance.loc[0, "rl_thresh"] == "None"
    assert block_performance.loc[0, "rl_thresh_flag"] == "None"
    assert block_performance.loc[0, "bias_rl_status_value"] == "None"
    assert block_performance.loc[0, "rl_status"] == "None"


def test_run_analysis_can_skip_multisession_save_and_load(tmp_path):
    """Simulation analysis should save within-session outputs without multisession files."""
    session = SimpleNamespace(
        mouse="sample_switch_mode_3",
        date="2026-03-25",
        sess_id_full="sample_switch_mode_3_2026-03-25_153737_agent-sample_switch_mode_3",
        processed_data_path=tmp_path,
        multi_session_save_path=None,
    )

    augmented_trial_df, block_performance, returned_session_df = session_analysis.run_analysis(
        make_simulated_trial_df(),
        session=session,
    )

    assert isinstance(returned_session_df, pd.DataFrame)
    assert returned_session_df.shape[0] == 1
    assert (tmp_path / f"{session.sess_id_full}_block_performance.csv").exists()
    assert (tmp_path / f"{session.sess_id_full}_augmented_trials.csv").exists()
    assert not (tmp_path / f"{session.mouse}_overall_performance.csv").exists()
    assert block_performance.shape[0] == 3
    assert augmented_trial_df.shape[0] == 6
    assert "n_blocks" in returned_session_df.columns
    assert returned_session_df.loc[0, "n_blocks"] == block_performance.shape[0]
    assert {
        "prev_n_rewarded_slope",
        "prev_n_rewarded_intercept",
        "prev_n_rewarded_r_value",
        "prev_n_rewarded_p_value",
        "prev_consecutive_rewards_slope",
        "prev_consecutive_rewards_intercept",
        "prev_consecutive_rewards_r_value",
        "prev_consecutive_rewards_p_value",
        "prev_n_correct_slope",
        "prev_n_correct_intercept",
        "prev_n_correct_r_value",
        "prev_n_correct_p_value",
    }.issubset(returned_session_df.columns)
    assert {"slope", "intercept", "r_value", "p_value", "n_switches"}.isdisjoint(
        returned_session_df.columns
    )


def test_run_analysis_returns_multisession_df_without_reloading(tmp_path, monkeypatch):
    """Mouse-session analysis should save and return the updated multisession table."""
    processed_data_path = tmp_path / "processed"
    multisession_save_path = tmp_path / "cross_session"
    processed_data_path.mkdir()
    multisession_save_path.mkdir()
    session = SimpleNamespace(
        mouse="sample_switch_mode_3",
        date="2026-03-25",
        sess_id_full="sample_switch_mode_3_2026-03-25_153737_agent-sample_switch_mode_3",
        processed_data_path=processed_data_path,
        multi_session_save_path=multisession_save_path,
    )

    def fail_if_reloaded(*_args, **_kwargs):
        raise AssertionError("run_analysis should not reload CSVs after saving")

    monkeypatch.setattr(session_analysis, "load_analysis", fail_if_reloaded)

    augmented_trial_df, block_performance, multisession_df = session_analysis.run_analysis(
        make_simulated_trial_df(),
        session=session,
    )

    multisession_summary_path = multisession_save_path / (
        f"{session.mouse}_overall_performance.csv"
    )
    assert multisession_summary_path.exists()
    assert multisession_df.shape[0] == 1
    assert multisession_df.loc[0, "date"] == session.date
    assert block_performance.shape[0] == 3
    assert augmented_trial_df.shape[0] == 6


def test_save_analysis_replaces_existing_multisession_date(tmp_path):
    """Saving one session should update its multisession row without duplicates."""
    session_save_path = tmp_path / "processed"
    multisession_save_path = tmp_path / "cross_session"
    session_save_path.mkdir()
    multisession_save_path.mkdir()
    sess_id = "sample_switch_mode_3_2026-03-25_153737_agent-sample_switch_mode_3"
    existing_multisession_df = pd.DataFrame(
        {
            "date": ["2026-03-24", "2026-03-25"],
            "n_blocks": [2, 99],
            "prev_consecutive_rewards_slope": [0.1, 99.0],
        }
    )
    existing_multisession_df.to_csv(
        multisession_save_path / "sample_switch_mode_3_overall_performance.csv",
        index=False,
    )
    session_performance = pd.DataFrame(
        {
            "date": ["2026-03-25"],
            "n_blocks": [3],
            "prev_consecutive_rewards_slope": [0.25],
        }
    )

    multisession_df = session_analysis.save_analysis(
        session_performance=session_performance,
        block_performance=pd.DataFrame({"block_ix": [0, 1, 2]}),
        augmented_trial_df=pd.DataFrame({"cur_trial": [0, 1, 2]}),
        sess_id=sess_id,
        session_save_path=session_save_path,
        multisession_save_path=multisession_save_path,
    )

    assert multisession_df["date"].tolist() == ["2026-03-24", "2026-03-25"]
    assert multisession_df.loc[
        multisession_df["date"] == "2026-03-25",
        "n_blocks",
    ].tolist() == [3]
    assert multisession_df.loc[
        multisession_df["date"] == "2026-03-25",
        "prev_consecutive_rewards_slope",
    ].tolist() == [0.25]


def test_assert_saved_csv_accepts_nonempty_file_and_rejects_missing_or_empty(tmp_path):
    """CSV save checks should require a real non-empty file."""
    missing_path = tmp_path / "missing.csv"
    empty_path = tmp_path / "empty.csv"
    nonempty_path = tmp_path / "nonempty.csv"

    empty_path.touch()
    nonempty_path.write_text("header\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        session_analysis.assert_saved_csv(missing_path)
    with pytest.raises(OSError):
        session_analysis.assert_saved_csv(empty_path)
    assert session_analysis.assert_saved_csv(nonempty_path) is None


def test_add_numeric_trials_to_correct_coerces_missing_values():
    """Trials-to-correct conversion should preserve input data and mark missing values."""
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "left_uncued", "right_cued"],
            "trials_to_correct": [0, "None", 2],
        }
    )

    summary_df = session_analysis.add_numeric_trials_to_correct(block_performance)

    assert "trials_to_correct_numeric" not in block_performance.columns
    assert summary_df["trials_to_correct_numeric"].iloc[0] == 0
    assert np.isnan(summary_df["trials_to_correct_numeric"].iloc[1])
    assert summary_df["trials_to_correct_numeric"].iloc[2] == 2


def test_get_nonfinal_missing_trials_to_correct_block_ix_uses_block_ix():
    """Missing non-final blocks should report block_ix when the column exists."""
    summary_df = pd.DataFrame(
        {
            "block_ix": [10, 11, 12],
            "trials_to_correct_numeric": [0.0, np.nan, np.nan],
        }
    )

    block_ix = session_analysis.get_nonfinal_missing_trials_to_correct_block_ix(
        summary_df
    )

    assert block_ix == [11]


def test_get_nonfinal_missing_trials_to_correct_block_ix_uses_index_without_block_ix():
    """Missing non-final blocks should fall back to dataframe index."""
    summary_df = pd.DataFrame(
        {
            "trials_to_correct_numeric": [0.0, np.nan, np.nan],
        },
        index=[20, 21, 22],
    )

    block_ix = session_analysis.get_nonfinal_missing_trials_to_correct_block_ix(
        summary_df
    )

    assert block_ix == [21]


def test_mean_trials_to_correct_for_block_type_ignores_missing_values():
    """Block-type means should use only valid numeric trials-to-correct values."""
    summary_df = pd.DataFrame(
        {
            "block_type": ["right_cued", "right_cued", "left_uncued"],
            "trials_to_correct_numeric": [0.0, 2.0, np.nan],
        }
    )

    right_cued_mean = session_analysis.mean_trials_to_correct_for_block_type(
        summary_df,
        "right_cued",
    )
    left_uncued_mean = session_analysis.mean_trials_to_correct_for_block_type(
        summary_df,
        "left_uncued",
    )

    assert right_cued_mean == 1.0
    assert left_uncued_mean == "None"


def test_summarize_trials_to_correct_ignores_none_in_overall_mean():
    """Block-summary means should ignore string sentinels in `trials_to_correct`."""
    block_performance = pd.DataFrame(
        {
            "block_type": ["right_cued", "left_uncued", "right_cued"],
            "trials_to_correct": [0, "None", 2],
        }
    )

    summary = session_analysis.summarize_trials_to_correct(block_performance)

    assert summary["overall_trials_to_correct"] == 1.0
    assert summary["left_uncued_trials_to_correct"] == "None"
    assert summary["right_cued_trials_to_correct"] == 1.0
    assert summary["trials_to_correct_warning_flag"] is True
    assert summary["trials_to_correct_nonfinal_invalid_block_ix"] == "[1]"


def test_summarize_trials_to_correct_warns_for_nonfinal_none():
    """Non-final string sentinels should raise a warning and return metadata."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "left_uncued", "right_cued"],
            "trials_to_correct": [0, "None", 2],
        }
    )

    with pytest.warns(UserWarning, match="Non-final invalid trials_to_correct"):
        summary = session_analysis.summarize_trials_to_correct(block_performance)

    assert summary["trials_to_correct_warning_flag"] is True
    assert summary["trials_to_correct_nonfinal_invalid_block_ix"] == "[1]"
    assert "block_ix=[1]" in summary["trials_to_correct_warning_message"]


def test_summarize_trials_to_correct_allows_final_none_without_warning():
    """A terminal incomplete block should be ignored without a data-quality warning."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "block_type": ["right_cued", "left_uncued", "right_cued"],
            "trials_to_correct": [0, 2, "None"],
        }
    )

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        summary = session_analysis.summarize_trials_to_correct(block_performance)

    assert len(recorded_warnings) == 0
    assert summary["overall_trials_to_correct"] == 1.0
    assert summary["trials_to_correct_warning_flag"] is False
    assert summary["trials_to_correct_nonfinal_invalid_block_ix"] == "None"
    assert summary["trials_to_correct_warning_message"] == "None"


def test_run_analysis_handles_blocks_with_no_correct_trials(tmp_path):
    """Simulation analysis should not crash when one block never reaches correct."""
    trial_df = make_simulated_trial_df().copy()
    trial_df.loc[2:3, "correct"] = 0

    session = SimpleNamespace(
        mouse="sample_switch_mode_3",
        date="2026-03-26",
        sess_id_full="sample_switch_mode_3_2026-03-26_120000_agent-sample_switch_mode_3",
        processed_data_path=tmp_path,
        multi_session_save_path=None,
    )

    _, block_performance, returned_session_df = session_analysis.run_analysis(
        trial_df,
        session=session,
    )

    assert returned_session_df.shape[0] == 1
    assert "None" in block_performance["trials_to_correct"].tolist()
