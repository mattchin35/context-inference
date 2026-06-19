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
        Trialwise dataframe with shape `(4, 9)`. Value columns use the
        left-positive convention used by existing trial features.
    """
    return pd.DataFrame(
        {
            "cur_trial": [0, 1, 2, 3],
            "cur_block": [0, 0, 1, 1],
            "action": [1, 1, 0, 0],
            "prev_action": ["None", 1, 0, "no_choice"],
            "FQlearning_rel_value": [0.2, 0.4, -0.6, 0.8],
            "HMM_rel_value_logodds_decay": [0.5, 0.7, -0.9, 0.1],
            "relative_doubt_index": [0.1, -0.2, 0.3, -0.4],
            "relative_hazard_index": [0.25, -0.5, 0.75, -1.0],
            "perseveration_regressor": [0.0, 0.2, -0.4, 0.6],
        }
    )


def test_add_observer_value_feature_subtracts_relative_doubt():
    """Observer value should be HMM decay minus relative doubt."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    assert "observer_value" in feature_df.columns
    assert feature_df["observer_value"].tolist() == pytest.approx([0.4, 0.9, -1.2, 0.5])


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
    assert side_df.loc[3, "observer_value_prev_action_side"] == "None"
    assert side_df.loc[1, "FQlearning_rel_value_prev_action_side"] == pytest.approx(0.4)
    assert side_df.loc[2, "FQlearning_rel_value_prev_action_side"] == pytest.approx(0.6)
    assert side_df.loc[1, "relative_doubt_index_prev_action_side"] == pytest.approx(-0.2)
    assert side_df.loc[2, "relative_doubt_index_prev_action_side"] == pytest.approx(-0.3)


def test_make_prev_action_side_equivalent_trial_values_contains_requested_columns():
    """Leave-stay table should include all requested side-equivalent regressors."""
    feature_df = gtf.add_observer_value_feature(make_feature_df())

    side_df = gtf.make_prev_action_side_equivalent_trial_values(feature_df)

    assert side_df.columns.tolist() == [
        "cur_trial",
        "cur_block",
        "action",
        "prev_action",
        "FQlearning_rel_value_prev_action_side",
        "HMM_rel_value_logodds_decay_prev_action_side",
        "relative_doubt_index_prev_action_side",
        "relative_hazard_index_prev_action_side",
        "perseveration_regressor_prev_action_side",
        "observer_value_prev_action_side",
    ]


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
    assert (tmp_path / "unit_session_augmented_trials.csv").exists()
    leave_stay_path = tmp_path / "unit_session_leave_stay_trial_values.csv"
    assert leave_stay_path.exists()
    saved_leave_stay = pd.read_csv(leave_stay_path, na_filter=False)
    assert "observer_value_prev_action_side" in saved_leave_stay.columns
