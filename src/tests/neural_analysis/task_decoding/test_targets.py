"""Target-table validation and eligibility tests for task-variable decoding."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding import config as decoding_config
from src.neural_analysis.task_decoding import targets as decoding_targets


def make_config(tmp_path: Path, *, target_names=None, alignment: str = "choice_time"):
    """Load a minimal configuration for one target-table test.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory used as a session root.
    target_names : sequence[str] or None
        Optional selected target identifiers. ``None`` requests loader defaults.
    alignment : str, default="choice_time"
        Selected event-time source column.

    Returns
    -------
    object
        Public ``TaskDecodingConfig`` instance produced by the JSON loader.
    """
    session_root = tmp_path / "session"
    processed_path = session_root / "processed"
    processed_path.mkdir(parents=True)
    (session_root / "neural_session.json").write_text("{}\n", encoding="ascii")
    (processed_path / "augmented_trials.csv").write_text("cur_trial\n0\n", encoding="ascii")
    (processed_path / "trial_feature_params.json").write_text("{}\n", encoding="ascii")
    payload = {
        "session_metadata_path": "neural_session.json",
        "augmented_trial_path": "processed/augmented_trials.csv",
        "trial_feature_parameter_path": "processed/trial_feature_params.json",
        "pfc_region": {
            "region": "PFC",
            "probe_id": "probe-a",
            "channel_labels": ["good"],
            "require_inside_brain": True,
            "cluster_groups": ["good", "mua"],
        },
        "hpc_region": {
            "region": "HPC",
            "probe_id": "probe-b",
            "channel_labels": ["good"],
            "require_inside_brain": True,
            "cluster_groups": ["good", "mua"],
        },
        "alignment": alignment,
        "output_root": "analysis_runs",
    }
    if target_names is not None:
        payload["target_names"] = list(target_names)
    config_path = session_root / "task_decoding_config.json"
    config_path.write_text(json.dumps(payload), encoding="ascii")
    return decoding_config.load_task_decoding_config(config_path)


def make_augmented_trial_df() -> pd.DataFrame:
    """Build a complete six-row chronological augmented-trial fixture.

    Returns
    -------
    pd.DataFrame
        Table with shape ``(6, 19)``. Rows include one manual choice and one
        recognized no-choice row so shifted-target and baseline contracts can
        be tested without changing the source dataframe.
    """
    return pd.DataFrame(
        {
            "cur_trial": [0, 1, 2, 3, 4, 5],
            "cur_trial_in_block": [0, 1, 2, 0, 1, 2],
            "cur_block": [0, 0, 0, 1, 1, 1],
            "state_int": [0, 1, 2, 0, 1, 0],
            "action": [0, 1, 1, "no_choice", 0, 1],
            "correct": [1, 0, 1, 0, 1, 1],
            "reward": [1, 1, 0, 0, 1, 0],
            "experimenter_reward_given": [0, 1, 0, 0, 0, 0],
            "choice_time": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "start_time": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
            "consecutive_omissions": [0, 0, 0, 1, 0, 0],
            "consecutive_rewards": [0, 1, 1, 0, 1, 1],
            "rewards_in_block": [0, 1, 1, 0, 1, 1],
            "Qlearning_rel_value": [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
            "FQlearning_rel_value": [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
            "HMM_rel_value_logodds": [-0.8, -0.4, 0.0, 0.4, 0.8, 0.9],
            "HMM_rel_value_logodds_decay": [-0.7, -0.3, 0.0, 0.3, 0.7, 0.8],
            "relative_doubt_index": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "existing_note": ["a", "b", "c", "d", "e", "f"],
        },
        index=pd.Index([101, 103, 107, 109, 113, 127], name="source_row"),
    )


def build_target_table(tmp_path: Path, trial_df: pd.DataFrame, *, target_names=None):
    """Build one public target table with the requested target subset.

    Parameters
    ----------
    tmp_path : Path
        Temporary session root for configuration loading.
    trial_df : pd.DataFrame
        Chronological augmented table with shape ``(n_trials, n_columns)``.
    target_names : sequence[str] or None
        Optional target subset passed through the public configuration loader.

    Returns
    -------
    pd.DataFrame
        Public target table with one row per input trial.
    """
    config = make_config(tmp_path, target_names=target_names)
    return decoding_targets.build_target_table(trial_df, config)


def test_validate_augmented_trials_reports_all_missing_columns_together(tmp_path):
    """Schema validation should list every missing selected-target input in one error."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df().loc[:, ["cur_trial", "action"]]

    with pytest.raises(ValueError) as error_info:
        decoding_targets.validate_augmented_trials(trial_df, config)

    message = str(error_info.value)
    for column in ("cur_block", "choice_time", "state_int", "relative_doubt_index"):
        assert column in message


def test_selected_target_subset_requires_only_its_sources_and_selected_alignment(tmp_path):
    """Choice alignment should not require start time or unselected target columns."""
    config = make_config(tmp_path, target_names=["current_action"])
    trial_df = make_augmented_trial_df().loc[
        :, ["cur_trial", "cur_block", "action", "experimenter_reward_given", "choice_time"]
    ]

    decoding_targets.validate_augmented_trials(trial_df, config)

    start_config = make_config(
        tmp_path / "start",
        target_names=["current_action"],
        alignment="start_time",
    )
    with pytest.raises(ValueError, match="start_time"):
        decoding_targets.validate_augmented_trials(trial_df, start_config)


@pytest.mark.parametrize(
    "cur_trial",
    [
        [],
        [0, 1, np.nan, 3, 4, 5],
        [0, 1, "bad", 3, 4, 5],
        [0, 1, np.inf, 3, 4, 5],
        [0, 1.5, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6],
        [0, 1, 1, 3, 4, 5],
        [0, 1, 3, 4, 5, 6],
    ],
)
def test_validate_augmented_trials_rejects_invalid_cur_trial_sequence(tmp_path, cur_trial):
    """Trial identity must be the complete finite zero-based chronological sequence."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df()
    if not cur_trial:
        trial_df = trial_df.iloc[0:0].copy()
    else:
        trial_df["cur_trial"] = cur_trial

    with pytest.raises(ValueError, match="cur_trial|empty"):
        decoding_targets.validate_augmented_trials(trial_df, config)


@pytest.mark.parametrize("missing_block", [None, np.nan, "None", "", "nan"])
def test_validate_augmented_trials_rejects_missing_block_identity(tmp_path, missing_block):
    """Every row must have a project-present block identity for grouped decoding."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df()
    trial_df["cur_block"] = trial_df["cur_block"].astype(object)
    trial_df.loc[trial_df.index[2], "cur_block"] = missing_block

    with pytest.raises(ValueError, match="cur_block"):
        decoding_targets.validate_augmented_trials(trial_df, config)


def test_validate_augmented_trials_rejects_noncontiguous_block_reuse(tmp_path):
    """One block label must occupy one contiguous chronological row segment."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df()
    trial_df["cur_block"] = [0, 1, 0, 1, 1, 1]

    with pytest.raises(ValueError, match="contiguous|cur_block"):
        decoding_targets.validate_augmented_trials(trial_df, config)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("choice_time", "bad"),
        ("choice_time", np.inf),
        ("experimenter_reward_given", "bad"),
        ("experimenter_reward_given", np.inf),
        ("reward", "bad"),
        ("consecutive_rewards", "bad"),
        ("relative_doubt_index", np.inf),
    ],
)
def test_validate_augmented_trials_rejects_present_malformed_numeric_values(
    tmp_path,
    column,
    value,
):
    """Present required numeric values should never be silently coerced to missingness."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df()
    trial_df[column] = trial_df[column].astype(object)
    trial_df.loc[trial_df.index[0], column] = value

    with pytest.raises(ValueError, match=column):
        decoding_targets.validate_augmented_trials(trial_df, config)


def test_validate_augmented_trials_keeps_recognized_missing_values_as_row_level_missingness(tmp_path):
    """Project sentinels and no-choice actions should not become malformed numeric values."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df()
    trial_df["relative_doubt_index"] = trial_df["relative_doubt_index"].astype(object)
    trial_df.loc[trial_df.index[0], "relative_doubt_index"] = "None"

    decoding_targets.validate_augmented_trials(trial_df, config)


@pytest.mark.parametrize("column", ["choice_time", "relative_doubt_index"])
def test_selected_alignment_and_numerical_targets_require_one_finite_value(
    tmp_path,
    column,
):
    """A selected numeric source cannot be entirely unavailable in one session."""
    config = make_config(tmp_path, target_names=["relative_doubt"])
    trial_df = make_augmented_trial_df()
    trial_df[column] = "None"

    with pytest.raises(ValueError, match=column):
        decoding_targets.validate_augmented_trials(trial_df, config)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("action", 0.5),
        ("action", np.inf),
        ("action", "bad"),
        ("correct", 2),
        ("correct", -1),
        ("state_int", 3),
        ("state_int", -1),
    ],
)
def test_validate_augmented_trials_rejects_extra_present_categorical_classes(
    tmp_path,
    column,
    value,
):
    """Only documented 0/1 labels, plus dark state 2, may appear in categorical sources."""
    config = make_config(tmp_path)
    trial_df = make_augmented_trial_df()
    trial_df[column] = trial_df[column].astype(object)
    trial_df.loc[trial_df.index[0], column] = value

    with pytest.raises(ValueError, match=column):
        decoding_targets.validate_augmented_trials(trial_df, config)


def test_build_target_table_preserves_original_rows_identity_and_input(tmp_path):
    """Target construction should not mutate a non-default-index source dataframe.

    The fixture has shape ``(6, 19)`` and the output must retain all six
    original chronological rows even though some are baseline-ineligible.
    """
    trial_df = make_augmented_trial_df()
    original_trial_df = trial_df.copy(deep=True)

    target_table = build_target_table(tmp_path, trial_df)

    pd.testing.assert_frame_equal(trial_df, original_trial_df)
    assert target_table.shape[0] == trial_df.shape[0]
    assert target_table["row_position"].tolist() == list(range(trial_df.shape[0]))
    assert target_table["trial_id"].tolist() == trial_df["cur_trial"].tolist()
    assert target_table["block_id"].tolist() == trial_df["cur_block"].tolist()


def test_shifted_targets_are_pre_filter_and_do_not_bridge_invalid_adjacent_choices(tmp_path):
    """Shifted labels use adjacent original rows but reject manual/no-choice neighbors."""
    trial_df = make_augmented_trial_df()
    target_table = build_target_table(
        tmp_path,
        trial_df,
        target_names=["previous_action", "previous_action_was_rewarded", "next_action"],
    )

    assert target_table.loc[1, "previous_action_valid"]
    assert target_table.loc[1, "previous_action"] == 0
    assert target_table.loc[1, "previous_action_was_rewarded_valid"]
    assert target_table.loc[1, "previous_action_was_rewarded"] == 1
    assert not target_table.loc[2, "previous_action_valid"]
    assert not target_table.loc[2, "next_action_valid"]
    assert not target_table.loc[4, "previous_action_valid"]
    assert target_table.loc[4, "next_action_valid"]
    assert target_table.loc[4, "next_action"] == 1


def test_switch_stay_targets_match_adjacent_valid_action_sequence(tmp_path):
    """Switch/stay labels should use one only for an adjacent change of side."""
    trial_df = make_augmented_trial_df()
    trial_df.loc[trial_df.index[1], "experimenter_reward_given"] = 0
    trial_df.loc[trial_df.index[3], "action"] = 1
    target_table = build_target_table(
        tmp_path,
        trial_df,
        target_names=["current_choice_switch_stay", "next_choice_switch_stay"],
    )

    pd.testing.assert_series_equal(
        target_table["current_choice_switch_stay"],
        pd.Series([np.nan, 1, 0, 0, 1, 1], name="current_choice_switch_stay"),
    )
    pd.testing.assert_series_equal(
        target_table["next_choice_switch_stay"],
        pd.Series([1, 0, 0, 1, 1, np.nan], name="next_choice_switch_stay"),
    )


def test_dark_state_is_target_ineligible_without_invalidating_other_current_targets(tmp_path):
    """Known dark state 2 should not become a third class or suppress current action."""
    trial_df = make_augmented_trial_df()
    target_table = build_target_table(
        tmp_path,
        trial_df,
        target_names=["current_state", "current_action"],
    )

    assert not target_table.loc[2, "current_state_valid"]
    assert target_table.loc[2, "current_action_valid"]
    assert target_table.loc[2, "current_action"] == 1


def test_current_categorical_targets_copy_exact_binary_values_and_missingness(tmp_path):
    """Current binary targets should preserve coding and mark sentinels unavailable."""
    trial_df = make_augmented_trial_df()
    trial_df["state_int"] = trial_df["state_int"].astype(object)
    trial_df["correct"] = trial_df["correct"].astype(object)
    trial_df.loc[trial_df.index[0], "state_int"] = "None"
    trial_df.loc[trial_df.index[4], "correct"] = "None"
    target_table = build_target_table(
        tmp_path,
        trial_df,
        target_names=["current_state", "current_action", "current_action_is_correct"],
    )

    assert not target_table.loc[0, "current_state_valid"]
    assert target_table.loc[1, "current_state"] == 1
    assert target_table.loc[2, "current_action"] == 1
    assert not target_table.loc[3, "current_action_valid"]
    assert not target_table.loc[4, "current_action_is_correct_valid"]
    assert target_table.loc[5, "current_action_is_correct"] == 1


def test_target_specific_missingness_does_not_change_unrelated_masks(tmp_path):
    """One missing numerical target should not remove a separately valid categorical target."""
    trial_df = make_augmented_trial_df()
    trial_df["relative_doubt_index"] = trial_df["relative_doubt_index"].astype(object)
    trial_df.loc[trial_df.index[4], "relative_doubt_index"] = "None"
    target_table = build_target_table(
        tmp_path,
        trial_df,
        target_names=["current_action", "relative_doubt"],
    )

    assert target_table.loc[4, "current_action_valid"]
    assert not target_table.loc[4, "relative_doubt_valid"]


def test_baseline_mask_excludes_manual_no_choice_and_missing_alignment_rows(tmp_path):
    """Baseline eligibility should be independent of target-specific shifted-label rules."""
    trial_df = make_augmented_trial_df()
    trial_df["choice_time"] = trial_df["choice_time"].astype(object)
    trial_df.loc[trial_df.index[5], "choice_time"] = "None"
    target_table = build_target_table(tmp_path, trial_df, target_names=["current_action"])

    assert target_table["baseline_valid"].tolist() == [True, False, True, False, True, False]

    reasons = target_table["baseline_invalid_reason"].tolist()
    assert reasons[0] == ""
    assert "manual" in reasons[1] or "experimenter" in reasons[1]
    assert "choice" in reasons[3]
    assert "alignment" in reasons[5]


def test_target_metadata_persists_binary_positive_class_mappings(tmp_path):
    """Saved target-table metadata should retain labels and the integer-one positive class."""
    target_table = build_target_table(tmp_path, make_augmented_trial_df())
    metadata = target_table.attrs["target_metadata"]

    assert metadata["current_action"]["positive_class"] == 1
    assert metadata["current_action"]["positive_label"] == "left"
    assert metadata["previous_action_was_rewarded"]["positive_class"] == 1
    assert metadata["previous_action_was_rewarded"]["positive_label"] == "rewarded"
    assert metadata["current_choice_switch_stay"]["positive_class"] == 1
    assert metadata["current_choice_switch_stay"]["positive_label"] == "switch"

    expected_class_labels = {
        "current_state": {"0": "right", "1": "left"},
        "current_action": {"0": "right", "1": "left"},
        "current_action_is_correct": {"0": "incorrect", "1": "correct"},
        "previous_action": {"0": "right", "1": "left"},
        "previous_action_was_rewarded": {"0": "unrewarded", "1": "rewarded"},
        "next_action": {"0": "right", "1": "left"},
        "current_choice_switch_stay": {"0": "stay", "1": "switch"},
        "next_choice_switch_stay": {"0": "stay", "1": "switch"},
    }
    for target_name, class_labels in expected_class_labels.items():
        assert metadata[target_name]["class_labels"] == class_labels
    assert metadata["previous_action"]["derivation"] == "previous_valid_action"
    assert metadata["next_choice_switch_stay"]["derivation"] == (
        "next_vs_current_action"
    )


def test_numerical_targets_retain_exact_native_numeric_source_values(tmp_path):
    """Numerical targets should equal their stored source values without target scaling."""
    trial_df = make_augmented_trial_df()
    target_names = [
        "consecutive_omissions",
        "consecutive_rewards",
        "session_trial_index",
        "trial_index_in_block",
        "rewards_in_block",
        "qlearning_relative_value",
        "forgetting_q_relative_value",
        "hmm_signed_belief",
        "hmm_decay_signed_belief",
        "relative_doubt",
    ]
    target_table = build_target_table(tmp_path, trial_df, target_names=target_names)
    source_columns = {
        "consecutive_omissions": "consecutive_omissions",
        "consecutive_rewards": "consecutive_rewards",
        "session_trial_index": "cur_trial",
        "trial_index_in_block": "cur_trial_in_block",
        "rewards_in_block": "rewards_in_block",
        "qlearning_relative_value": "Qlearning_rel_value",
        "forgetting_q_relative_value": "FQlearning_rel_value",
        "hmm_signed_belief": "HMM_rel_value_logodds",
        "hmm_decay_signed_belief": "HMM_rel_value_logodds_decay",
        "relative_doubt": "relative_doubt_index",
    }

    for target_name, source_column in source_columns.items():
        np.testing.assert_array_equal(
            target_table[target_name].to_numpy(),
            pd.to_numeric(trial_df[source_column]).to_numpy(),
        )
