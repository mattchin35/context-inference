import sys
from types import ModuleType

import numpy as np
import pandas as pd
import pytest


def install_statsmodels_stub() -> None:
    """Install a small `statsmodels.api` stub for residualization tests.

    Returns
    -------
    None
        Mutates `sys.modules` so `gather_trial_features` can import its
        residualization dependency in test environments without statsmodels.
    """
    if "statsmodels.api" in sys.modules:
        return

    statsmodels_stub = ModuleType("statsmodels")
    api_stub = ModuleType("statsmodels.api")

    def add_constant(design_matrix, has_constant="add"):
        matrix = np.asarray(design_matrix, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        return np.column_stack([np.ones(matrix.shape[0]), matrix])

    class OLS:
        def __init__(self, target_values, design_matrix):
            self.target_values = np.asarray(target_values, dtype=float)
            self.design_matrix = np.asarray(design_matrix, dtype=float)

        def fit(self):
            coefficients, *_ = np.linalg.lstsq(
                self.design_matrix,
                self.target_values,
                rcond=None,
            )
            fitted = self.design_matrix @ coefficients
            return type("OLSResult", (), {"resid": self.target_values - fitted})()

    api_stub.add_constant = add_constant
    api_stub.OLS = OLS
    statsmodels_stub.api = api_stub
    sys.modules["statsmodels"] = statsmodels_stub
    sys.modules["statsmodels.api"] = api_stub


install_statsmodels_stub()

from src.behavior_analysis import gather_trial_features as gtf


def make_feature_df() -> pd.DataFrame:
    """Build a minimal feature dataframe for observer/leave-stay tests.

    Returns
    -------
    pd.DataFrame
        Trialwise dataframe with shape `(5, n_columns)`. Value columns use the
        left-positive convention used by existing trial features.
    """
    return pd.DataFrame(
        {
            "state": [0, 0, 1, 1, 0],
            "state_int": [0, 0, 1, 1, 0],
            "cur_trial": [0, 1, 2, 3, 4],
            "cur_trial_in_block": [0, 1, 0, 1, 2],
            "cur_block": [0, 0, 1, 1, 1],
            "action": [1, 1, 1, 0, 0],
            "correct": [1, 1, 0, 1, 0],
            "reward": [0, 1, 0, 1, 0],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
            "session_ID": ["sess", "sess", "sess", "sess", "sess"],
            "block_type": ["A", "A", "B", "B", "B"],
            "time_to_choice": [0.5, 0.6, 0.7, 0.8, 0.9],
            "prev_action": ["None", 1, 0, 0, "no_choice"],
            "prev_reward": ["None", 1, 0, 1, 0],
            "inherited_block_strategy": ["None", 1, 1, 2, 2],
            "inherited_block_bias": ["None", "False", "False", "True", "True"],
            "Qlearning_rel_value": [0.1, 0.2, -0.3, 0.4, 0.5],
            "FQlearning_rel_value": [0.2, 0.4, -0.6, 0.8, 1.0],
            "FQlearning_rel_value_fast_learn": [0.3, 0.5, -0.7, 0.9, 1.1],
            "HMM_rel_value_logodds": [0.4, 0.6, -0.8, 1.0, 1.2],
            "HMM_rel_value_logodds_decay": [0.5, 0.7, -0.9, 0.1, 1.3],
            "relative_omissions_index": [0.05, 0.15, -0.25, 0.35, 0.45],
            "signed_omission_regressor": [0.06, 0.16, -0.26, 0.36, 0.46],
            "relative_doubt_index": [0.1, -0.2, 0.3, -0.4, 0.5],
            "relative_hazard_index": [0.25, -0.5, 0.75, -1.0, 1.25],
            "perseveration_regressor": [0.0, 0.2, -0.4, 0.6, 0.8],
            "HMM_decay_res": [0.11, 0.21, -0.31, 0.41, 0.51],
            "rel_hazard_res": [0.12, 0.22, -0.32, 0.42, 0.52],
        }
    )


def make_trial_type_flag_df() -> pd.DataFrame:
    """Build trial rows with block, switch, stay, and explore cases.

    Returns
    -------
    pd.DataFrame
        Trialwise dataframe with shape `(8, n_columns)`. Actions use the task
        convention `0=right`, `1=left`; `cur_block` is a unitless block id.
    """
    return pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 1, 1, 1, 2, 2],
            "action": [1, 1, 0, "no_choice", 0, 1, 0, 0],
            "prev_action": ["None", 1, 1, 0, 0, 0, 1, 0],
            "correct": [1, 1, 0, 0, 1, 1, 1, 1],
            "reward": [1, 1, 0, 0, 1, 1, 0, 1],
            "prev_reward": ["None", 1, 1, 0, 0, 1, 1, 0],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 1, 0],
        }
    )


def test_add_trial_type_flags_marks_switch_stay_block_entry_and_explore_trials():
    """Trial-type flags should mark primitive and intersection categories."""
    flagged_df = gtf.add_trial_type_flags(make_trial_type_flag_df())

    expected = {
        "prev_correct": [False, True, True, False, False, True, True, False],
        "block_entry_trial": [True, False, False, False, True, False, False, True],
        "switch_trial": [False, False, True, False, False, True, False, False],
        "stay_trial": [False, True, False, False, True, False, False, True],
        "first_switch_in_block": [False, False, True, False, False, True, False, False],
        "explore_trial": [False, False, True, False, False, True, False, False],
        "block_entry_explore_trial": [False, False, True, False, False, True, False, False],
    }
    for column_name, expected_values in expected.items():
        assert flagged_df[column_name].tolist() == expected_values


def test_add_trial_type_flags_accepts_csv_loaded_experimenter_reward_flags():
    """String zero reward flags from CSV should not invalidate every choice."""
    trial_df = make_trial_type_flag_df()
    trial_df["experimenter_reward_given"] = ["0", "0", "0", "0", "0", "0", "1", "0"]

    flagged_df = gtf.add_trial_type_flags(trial_df)

    expected = {
        "block_entry_trial": [True, False, False, False, True, False, False, True],
        "switch_trial": [False, False, True, False, False, True, False, False],
        "stay_trial": [False, True, False, False, True, False, False, True],
        "first_switch_in_block": [False, False, True, False, False, True, False, False],
        "explore_trial": [False, False, True, False, False, True, False, False],
    }
    for column_name, expected_values in expected.items():
        assert flagged_df[column_name].tolist() == expected_values


def test_add_explore_run_flags_marks_short_rewarded_leave_and_correct_return():
    """A brief rewarded-side leave followed by a correct return should be tagged."""
    trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 0, 0],
            "action": [1, 0, 0, 1, 1],
            "correct": [1, 0, 0, 1, 1],
            "reward": [1, 0, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    feature_df = gtf.add_explore_run_flags(trial_df, max_explore_run_length=5)

    assert feature_df["explore_run_start"].tolist() == [False, True, False, False, False]
    assert feature_df["explore_run_trial"].tolist() == [False, True, True, False, False]
    assert feature_df["explore_run_return"].tolist() == [False, False, False, True, False]
    assert feature_df["explore_run_id"].tolist() == ["None", 0, 0, 0, "None"]
    assert feature_df["explore_run_length"].tolist() == ["None", 2, 2, 2, "None"]


def test_add_explore_run_flags_rejects_runs_longer_than_user_limit():
    """Runs longer than max_explore_run_length should not be treated as explore runs."""
    trial_df = pd.DataFrame(
        {
            "cur_block": [0] * 8,
            "action": [1, 0, 0, 0, 0, 0, 0, 1],
            "correct": [1, 0, 0, 0, 0, 0, 0, 1],
            "reward": [1, 0, 0, 0, 0, 0, 0, 0],
            "experimenter_reward_given": [0] * 8,
        }
    )

    rejected_df = gtf.add_explore_run_flags(trial_df, max_explore_run_length=5)
    accepted_df = gtf.add_explore_run_flags(trial_df, max_explore_run_length=6)

    assert rejected_df["explore_run_trial"].sum() == 0
    assert rejected_df["explore_run_return"].sum() == 0
    assert accepted_df["explore_run_trial"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    assert accepted_df["explore_run_return"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]


def test_add_explore_run_flags_rejects_block_boundary_switches():
    """A switch at a block boundary should not start an exploratory run."""
    trial_df = pd.DataFrame(
        {
            "cur_block": [0, 1, 1],
            "action": [1, 0, 1],
            "correct": [1, 0, 1],
            "reward": [1, 0, 0],
            "experimenter_reward_given": [0, 0, 0],
        }
    )

    feature_df = gtf.add_explore_run_flags(trial_df, max_explore_run_length=5)

    assert feature_df["explore_run_start"].sum() == 0
    assert feature_df["explore_run_trial"].sum() == 0
    assert feature_df["explore_run_return"].sum() == 0


def test_add_explore_run_flags_ignores_no_choice_rows_inside_run():
    """No-choice rows should not increase run length or break a valid run."""
    trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 0, 0],
            "action": [1, 0, "no_choice", 0, 1],
            "correct": [1, 0, 0, 0, 1],
            "reward": [1, 0, 0, 0, 0],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    feature_df = gtf.add_explore_run_flags(trial_df, max_explore_run_length=2)

    assert feature_df["explore_run_trial"].tolist() == [False, True, False, True, False]
    assert feature_df["explore_run_return"].tolist() == [False, False, False, False, True]
    assert feature_df["explore_run_length"].tolist() == ["None", 2, "None", 2, 2]


def test_add_explore_run_flags_handles_nondefault_dataframe_index():
    """Run tagging should align to rows even when the dataframe index is not positional."""
    trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0],
            "action": [1, 0, 1],
            "correct": [1, 0, 1],
            "reward": [1, 0, 0],
            "experimenter_reward_given": [0, 0, 0],
        },
        index=[10, 20, 30],
    )

    feature_df = gtf.add_explore_run_flags(trial_df, max_explore_run_length=1)

    assert feature_df.index.tolist() == [10, 20, 30]
    assert feature_df["explore_run_start"].tolist() == [False, True, False]
    assert feature_df["explore_run_return"].tolist() == [False, False, True]


def test_add_explore_run_flags_rejects_invalid_run_length_setting():
    """The run-length threshold should be a positive trial count."""
    with pytest.raises(ValueError, match="max_explore_run_length"):
        gtf.add_explore_run_flags(make_trial_type_flag_df(), max_explore_run_length=0)


def test_leave_stay_writer_copies_trial_type_flags_unchanged():
    """Leave-stay CSV generation should preserve augmented-trial flags."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())
    feature_df = gtf.add_trial_type_flags(feature_df)

    side_df = gtf.make_prev_action_side_equivalent_trial_values(feature_df)

    flag_columns = [
        "prev_correct",
        "block_entry_trial",
        "switch_trial",
        "stay_trial",
        "first_switch_in_block",
        "explore_trial",
        "block_entry_explore_trial",
    ]
    for column_name in flag_columns:
        assert side_df[column_name].tolist() == feature_df[column_name].tolist()


def test_add_observer_value_feature_subtracts_relative_doubt():
    """Observer value should be HMM decay minus relative doubt."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    assert "observer_value" in feature_df.columns
    assert feature_df["observer_value"].tolist() == pytest.approx([0.4, 0.9, -1.2, 0.5, 0.8])


def test_add_observer_value_feature_requires_source_columns():
    """Missing observer source columns should fail with a clear error."""
    incomplete = make_feature_df().drop(columns=["relative_doubt_index"])

    with pytest.raises(ValueError, match="relative_doubt_index"):
        gtf.add_observer_value_feature(incomplete)


def test_make_prev_action_side_equivalent_trial_values_flips_right_reference_side():
    """Previous-right rows should invert left-positive values."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    side_df = gtf.make_prev_action_side_equivalent_trial_values(feature_df)

    assert side_df.loc[0, "observer_value_prev_action_side"] == "None"
    assert side_df.loc[1, "observer_value_prev_action_side"] == pytest.approx(0.9)
    assert side_df.loc[2, "observer_value_prev_action_side"] == pytest.approx(1.2)
    assert side_df.loc[3, "observer_value_prev_action_side"] == pytest.approx(-0.5)
    assert side_df.loc[4, "observer_value_prev_action_side"] == "None"
    assert side_df.loc[1, "FQlearning_rel_value_prev_action_side"] == pytest.approx(0.4)
    assert side_df.loc[2, "FQlearning_rel_value_prev_action_side"] == pytest.approx(0.6)
    assert side_df.loc[1, "relative_doubt_index_prev_action_side"] == pytest.approx(-0.2)
    assert side_df.loc[2, "relative_doubt_index_prev_action_side"] == pytest.approx(-0.3)


def test_make_prev_action_side_equivalent_trial_values_recodes_action_to_stay_leave():
    """Leave-stay action should preserve raw side choices separately."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    side_df = gtf.make_prev_action_side_equivalent_trial_values(feature_df)

    assert side_df["raw_action"].tolist() == [1, 1, 1, 0, 0]
    assert side_df["action"].tolist() == ["None", 1, 0, 1, "None"]
    assert side_df["prev_reward"].tolist() == ["None", 1, 0, 1, 0]


def test_make_prev_action_side_equivalent_trial_values_contains_requested_columns():
    """Leave-stay table should include all requested side-equivalent regressors."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    side_df = gtf.make_prev_action_side_equivalent_trial_values(feature_df)

    assert side_df.columns.tolist() == [
        "state",
        "state_int",
        "cur_trial",
        "cur_trial_in_block",
        "cur_block",
        "raw_action",
        "action",
        "correct",
        "reward",
        "experimenter_reward_given",
        "session_ID",
        "block_type",
        "time_to_choice",
        "prev_action",
        "prev_reward",
        "inherited_block_strategy",
        "inherited_block_bias",
        "Qlearning_rel_value_prev_action_side",
        "FQlearning_rel_value_prev_action_side",
        "FQlearning_rel_value_fast_learn_prev_action_side",
        "HMM_rel_value_logodds_prev_action_side",
        "HMM_rel_value_logodds_decay_prev_action_side",
        "relative_omissions_index_prev_action_side",
        "signed_omission_regressor_prev_action_side",
        "relative_doubt_index_prev_action_side",
        "relative_hazard_index_prev_action_side",
        "perseveration_regressor_prev_action_side",
        "observer_value_prev_action_side",
        "HMM_decay_res_prev_action_side",
        "rel_hazard_res_prev_action_side",
    ]


def test_make_prev_action_side_equivalent_trial_values_omits_missing_optional_predictors():
    """Leave-stay writer should tolerate optional side predictors being absent."""
    feature_df = gtf.add_observer_value_feature(
        make_feature_df().drop(columns=["HMM_decay_res", "rel_hazard_res"])
    )

    side_df = gtf.make_prev_action_side_equivalent_trial_values(feature_df)

    assert "HMM_decay_res_prev_action_side" not in side_df.columns
    assert "rel_hazard_res_prev_action_side" not in side_df.columns
    assert "FQlearning_rel_value_prev_action_side" in side_df.columns


def test_save_leave_stay_trial_values_writes_csv(tmp_path):
    """The leave-stay helper should save a non-empty side-equivalent CSV."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    side_df = gtf.save_leave_stay_trial_values(
        augmented_trial_df=feature_df,
        processed_data_path=tmp_path,
        sess_id_full="unit_session",
    )

    save_path = tmp_path / "unit_session_leave_stay_trial_values.csv"
    assert save_path.exists()
    assert save_path.stat().st_size > 0
    pd.testing.assert_frame_equal(pd.read_csv(save_path, na_filter=False), side_df)


def test_collect_and_save_trial_features_writes_leave_stay_csv(tmp_path):
    """Main feature save path should write augmented and leave-stay feature CSVs."""
    augmented_trial_df = pd.DataFrame(
        {
            "action": [0, 1, 0, 1],
            "prev_action": ["None", 0, 1, 0],
            "prev_reward": ["None", 1, 0, 0],
            "reward": [1, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0],
            "right_omissions": [0, 1, 2, 0],
            "left_omissions": [0, 0, 1, 2],
            "right_cf_omissions": [0, 1, 2, 0],
            "left_cf_omissions": [0, 0, 1, 2],
            "consecutive_omissions": [0, 1, 2, 0],
            "relative_monotonic_cf_value": [0, -1, -1, 2],
        }
    )

    feature_df, _params = gtf.collect_and_save_trial_features(
        augmented_trial_df=augmented_trial_df,
        processed_data_path=tmp_path,
        sess_id_full="unit_session",
    )

    assert "observer_value" in feature_df.columns
    assert "switch_trial" in feature_df.columns
    assert (tmp_path / "unit_session_augmented_trials.csv").exists()
    leave_stay_path = tmp_path / "unit_session_leave_stay_trial_values.csv"
    assert leave_stay_path.exists()
    saved_leave_stay = pd.read_csv(leave_stay_path, na_filter=False)
    assert "observer_value_prev_action_side" in saved_leave_stay.columns
    assert "raw_action" in saved_leave_stay.columns
    assert "prev_reward" in saved_leave_stay.columns
