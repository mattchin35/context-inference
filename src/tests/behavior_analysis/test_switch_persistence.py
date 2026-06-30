from pathlib import Path

import pandas as pd
import pytest

from src.behavior_analysis import switch_persistence


def make_block_performance() -> pd.DataFrame:
    """Build minimal block summary rows for switch-persistence tests.

    Returns
    -------
    pd.DataFrame
        Blockwise table with shape `(4, 4)`. `block_ix` uses the same integer
        axis as `augmented_trial_df.cur_block`; counts are unitless trial
        counts from prior session analysis.
    """
    return pd.DataFrame(
        {
            "block_ix": [0, 1, 2, 3],
            "prev_consecutive_rewards": [0, 2, 1, 3],
            "prev_n_correct": [0, 2, 2, 1],
            "prev_n_rewarded": [0, 1, 2, 1],
        }
    )


def test_add_previous_block_omission_metrics_counts_correct_unrewarded_only():
    """Previous omissions should count correct unrewarded choices only."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 1, 1, 1, 2],
            "state": [0, 0, 0, 1, 1, 1, 0],
            "action": [0, 1, 0, 1, 1, 0, 0],
            "correct": [1, 0, 1, 1, 1, 0, 1],
            "reward": [1, 0, 0, 0, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 0],
        }
    )

    enriched = switch_persistence.add_previous_block_omission_metrics(
        block_performance,
        augmented_trial_df,
    )

    assert enriched["prev_n_omissions"].tolist() == [0, 1, 2, 0]
    assert enriched["prev_consecutive_omissions"].tolist() == [0, 1, 2, 0]


def test_previous_block_omission_metrics_ignore_none_correct_on_no_choice_rows():
    """No-choice rows with string missing correctness should not crash omission counts."""
    block_performance = make_block_performance().iloc[:2]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 1],
            "state": [0, 0, 0, 1],
            "action": [0, "no_choice", 0, 1],
            "correct": [1, "None", 1, 1],
            "reward": [1, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    enriched = switch_persistence.add_previous_block_omission_metrics(
        block_performance,
        augmented_trial_df,
    )

    assert enriched["prev_n_omissions"].tolist() == [0, 1]
    assert enriched["prev_consecutive_omissions"].tolist() == [0, 1]


def test_previous_block_omission_metrics_ignore_none_correct_on_manual_reward_rows():
    """Manual reward rows with missing correctness should not enter omission counts."""
    block_performance = make_block_performance().iloc[:2]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 0, 1],
            "state": [0, 0, 0, 1],
            "action": [0, 1, 0, 1],
            "correct": [1, "None", 1, 1],
            "reward": [1, 1, 0, 1],
            "experimenter_reward_given": [0, 1, 0, 0],
        }
    )

    enriched = switch_persistence.add_previous_block_omission_metrics(
        block_performance,
        augmented_trial_df,
    )

    assert enriched["prev_n_omissions"].tolist() == [0, 1]
    assert enriched["prev_consecutive_omissions"].tolist() == [0, 1]


def test_previous_block_omission_metrics_reject_none_correct_on_valid_choice_rows():
    """Malformed correctness on animal-choice rows should still fail clearly."""
    block_performance = make_block_performance().iloc[:2]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 1],
            "state": [0, 0, 1],
            "action": [0, 1, 1],
            "correct": [1, "None", 1],
            "reward": [1, 0, 1],
            "experimenter_reward_given": [0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="correct values must be numeric"):
        switch_persistence.add_previous_block_omission_metrics(
            block_performance,
            augmented_trial_df,
        )


def test_previous_block_omission_metrics_align_noncontiguous_trial_blocks():
    """Contiguous block rows should align to skipped raw trial block ids by order."""
    block_performance = pd.DataFrame({"block_ix": [0, 1, 2]})
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 2, 2, 5],
            "state": [0, 0, 1, 1, 0],
            "action": [0, 0, 1, 1, 0],
            "correct": [1, 1, 1, 1, 1],
            "reward": [1, 0, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    enriched = switch_persistence.add_previous_block_omission_metrics(
        block_performance,
        augmented_trial_df,
    )

    assert enriched["prev_n_omissions"].tolist() == [0, 1, 2]
    assert enriched["prev_consecutive_omissions"].tolist() == [0, 1, 2]


def test_compute_switch_persistence_aligns_noncontiguous_trial_blocks():
    """Switch persistence should use raw cur_block ids even when block_ix is recoded."""
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1, 2],
            "prev_consecutive_rewards": [0, 2, 1],
            "prev_n_correct": [0, 2, 2],
            "prev_n_rewarded": [0, 1, 2],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(7)),
            "cur_block": [0, 0, 2, 2, 2, 5, 5],
            "state": [1, 1, 0, 0, 0, 1, 1],
            "action": [1, 1, 1, 0, 0, 0, 1],
            "correct": [1, 1, 0, 1, 1, 0, 1],
            "reward": [1, 1, 0, 1, 1, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance,
        augmented_trial_df,
    )
    block_detail = detail[detail["block_ix"] == 1]

    assert block_detail["trial_cur_block"].tolist() == [2, 2, 2]
    assert block_detail["switch_direction"].tolist() == ["L_to_R"] * 3
    assert block_detail["choice_status"].tolist() == ["stay", "switch", "post_switch"]


def test_compute_switch_persistence_raises_when_block_ids_cannot_align():
    """A mismatch in block counts should fail with a block-alignment error."""
    block_performance = pd.DataFrame({"block_ix": [0, 1, 2]})
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 2, 2],
            "state": [1, 1, 0, 0],
            "action": [1, 1, 0, 0],
            "correct": [1, 1, 1, 1],
            "reward": [1, 1, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="Cannot align block_performance block_ix"):
        switch_persistence.compute_switch_persistence_trials(
            block_performance,
            augmented_trial_df,
        )


def test_compute_switch_persistence_l_to_r_stops_after_first_switch():
    """An L-to-R block should contribute stays until the first right choice."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(7)),
            "cur_block": [0, 0, 1, 1, 1, 1, 2],
            "state": [1, 1, 0, 0, 0, 0, 1],
            "action": [1, 1, 1, 1, 0, 0, 1],
            "correct": [1, 1, 0, 0, 1, 1, 1],
            "reward": [1, 1, 0, 0, 1, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance.iloc[:3],
        augmented_trial_df,
    )
    block_detail = detail[detail["block_ix"] == 1]

    assert block_detail["switch_direction"].tolist() == ["L_to_R"] * 4
    assert block_detail["choice_trial_after_switch"].tolist() == [1, 2, 3, 4]
    assert block_detail["choice_status"].tolist() == ["stay", "stay", "switch", "post_switch"]
    assert block_detail["include_trial_in_session_metric"].tolist() == [True, True, True, False]
    assert block_detail["stay"].tolist() == [True, True, False, False]


def test_compute_switch_persistence_r_to_l_direction():
    """An R-to-L block should treat right choices as stays."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 1, 1, 1],
            "state": [0, 0, 1, 1, 1],
            "action": [0, 0, 0, 1, 1],
            "correct": [1, 1, 0, 1, 1],
            "reward": [1, 1, 0, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance.iloc[:2],
        augmented_trial_df,
    )
    block_detail = detail[detail["block_ix"] == 1]

    assert block_detail["switch_direction"].tolist() == ["R_to_L"] * 3
    assert block_detail["choice_status"].tolist() == ["stay", "switch", "post_switch"]
    assert block_detail["stay"].tolist() == [True, False, False]


def test_no_choice_before_switch_flags_and_excludes_block_from_summary():
    """Blocks with no-choice before first switch are diagnostic-only."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 1, 1, 1],
            "state": [1, 1, 0, 0, 0],
            "action": [1, 1, "no_choice", 1, 0],
            "correct": [1, 1, 0, 0, 1],
            "reward": [1, 1, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance.iloc[:2],
        augmented_trial_df,
    )
    summary = switch_persistence.summarize_switch_persistence(detail)

    block_detail = detail[detail["block_ix"] == 1]
    assert block_detail["choice_status"].tolist() == ["no_choice", "stay", "switch"]
    assert block_detail["include_block_in_session_metric"].unique().tolist() == [False]
    assert block_detail["exclusion_reason"].unique().tolist() == ["no_choice_before_switch"]
    assert summary.empty


def test_experimenter_reward_before_switch_flags_and_excludes_block_from_summary():
    """Manual rewards before first switch should exclude a block from summaries."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 1, 1, 1],
            "state": [1, 1, 0, 0, 0],
            "action": [1, 1, 1, 1, 0],
            "correct": [1, 1, 0, 0, 1],
            "reward": [1, 1, 0, 0, 1],
            "experimenter_reward_given": [0, 0, 1, 0, 0],
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance.iloc[:2],
        augmented_trial_df,
    )
    summary = switch_persistence.summarize_switch_persistence(detail)

    block_detail = detail[detail["block_ix"] == 1]
    assert block_detail["include_block_in_session_metric"].unique().tolist() == [False]
    assert block_detail["exclusion_reason"].unique().tolist() == ["experimenter_reward_before_switch"]
    assert summary.empty


def test_no_switch_block_retained_but_excluded_from_summary():
    """Blocks without a new-side choice should stay available for diagnostics."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 1, 1, 1],
            "state": [1, 1, 0, 0, 0],
            "action": [1, 1, 1, 1, 1],
            "correct": [1, 1, 0, 0, 0],
            "reward": [1, 1, 0, 0, 0],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance.iloc[:2],
        augmented_trial_df,
    )
    summary = switch_persistence.summarize_switch_persistence(detail)

    block_detail = detail[detail["block_ix"] == 1]
    assert block_detail["choice_status"].tolist() == ["stay", "stay", "stay"]
    assert block_detail["include_block_in_session_metric"].unique().tolist() == [False]
    assert block_detail["exclusion_reason"].unique().tolist() == ["no_switch_before_block_end"]
    assert summary.empty


def test_summary_counts_decreasing_denominators_after_first_switch():
    """Later valid-choice indices should include only not-yet-switched blocks."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(11)),
            "cur_block": [0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3],
            "state": [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            "action": [1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
            "correct": [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
            "reward": [1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1],
            "experimenter_reward_given": [0] * 11,
        }
    )

    detail = switch_persistence.compute_switch_persistence_trials(
        block_performance,
        augmented_trial_df,
    )
    summary = switch_persistence.summarize_switch_persistence(detail)

    combined = summary[summary["switch_group"] == "combined"].sort_values(
        "choice_trial_after_switch"
    )
    assert combined["choice_trial_after_switch"].tolist() == [1, 2, 3, 4]
    assert combined["n_blocks"].tolist() == [3, 3, 1, 1]
    assert combined["n_stay"].tolist() == [3, 1, 1, 0]
    assert combined["proportion_stay"].tolist() == [1.0, 1 / 3, 1.0, 0.0]


def test_state_block_alignment_validation_raises_on_inconsistent_block_states():
    """Each block must have exactly one normalized left/right state."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0, 0, 1, 1],
            "state": [1, 0, 0, 0],
            "action": [1, 0, 1, 0],
            "correct": [1, 1, 0, 1],
            "reward": [1, 1, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="multiple context states"):
        switch_persistence.compute_switch_persistence_trials(
            block_performance.iloc[:2],
            augmented_trial_df,
        )


def test_save_switch_persistence_outputs_writes_detail_and_summary_csvs(tmp_path):
    """Switch-persistence saving should produce non-empty inspectable tables."""
    block_performance = make_block_performance()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 1, 1, 1],
            "state": [1, 1, 0, 0, 0],
            "action": [1, 1, 1, 0, 0],
            "correct": [1, 1, 0, 1, 1],
            "reward": [1, 1, 0, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    detail, summary = switch_persistence.save_switch_persistence_outputs(
        block_performance=block_performance.iloc[:2],
        augmented_trial_df=augmented_trial_df,
        processed_data_path=tmp_path,
        sess_id_full="CT999_2026-06-18_120000",
    )

    detail_path = tmp_path / "CT999_2026-06-18_120000_switch_persistence_trials.csv"
    summary_path = tmp_path / "CT999_2026-06-18_120000_switch_persistence_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()
    assert detail_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    pd.testing.assert_frame_equal(pd.read_csv(detail_path, na_filter=False), detail)
    pd.testing.assert_frame_equal(pd.read_csv(summary_path, na_filter=False), summary)


def test_compute_post_first_correct_accuracy_includes_first_correct_as_index_zero():
    """The first correct choice in each block should be the x=0 anchor."""
    block_performance = make_block_performance().iloc[:2]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(6)),
            "cur_block": [0, 0, 0, 1, 1, 1],
            "state": [1, 1, 1, 0, 0, 0],
            "action": [0, 1, 1, 1, 0, 0],
            "correct": [0, 1, 1, 0, 1, 1],
            "reward": [0, 1, 1, 0, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_post_first_correct_accuracy_trials(
        block_performance,
        augmented_trial_df,
    )

    included = detail[detail["include_trial_in_session_metric"]]
    assert included["block_ix"].tolist() == [0, 0, 1, 1]
    assert included["choice_trial_after_first_correct"].tolist() == [0, 1, 0, 1]
    assert included["correct"].tolist() == [True, True, True, True]
    assert included["correct_side"].tolist() == ["left", "left", "right", "right"]


def test_compute_post_first_correct_accuracy_stays_within_block():
    """Later x indices should not borrow choices from the following block."""
    block_performance = make_block_performance().iloc[:2]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 1, 1, 1],
            "state": [1, 1, 0, 0, 0],
            "action": [1, 1, 1, 0, 0],
            "correct": [0, 1, 0, 1, 1],
            "reward": [0, 1, 0, 1, 1],
            "experimenter_reward_given": [0, 0, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_post_first_correct_accuracy_trials(
        block_performance,
        augmented_trial_df,
    )
    included = detail[detail["include_trial_in_session_metric"]]

    assert included[included["block_ix"] == 0]["choice_trial_after_first_correct"].tolist() == [0]
    assert included[included["block_ix"] == 1]["choice_trial_after_first_correct"].tolist() == [0, 1]


def test_compute_post_first_correct_accuracy_excludes_no_choice_and_manual_reward_trials():
    """No-choice and manual-reward rows should not define or enter the accuracy curve."""
    block_performance = make_block_performance().iloc[:1]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(5)),
            "cur_block": [0, 0, 0, 0, 0],
            "state": [1, 1, 1, 1, 1],
            "action": ["no_choice", 1, 0, 1, 1],
            "correct": ["None", 1, 0, 1, 1],
            "reward": [0, 1, 0, 1, 1],
            "experimenter_reward_given": [0, 1, 0, 0, 0],
        }
    )

    detail = switch_persistence.compute_post_first_correct_accuracy_trials(
        block_performance,
        augmented_trial_df,
    )
    included = detail[detail["include_trial_in_session_metric"]]

    assert detail["choice_status"].tolist() == [
        "no_choice",
        "manual_reward",
        "pre_first_correct",
        "post_first_correct",
        "post_first_correct",
    ]
    assert included["cur_trial"].tolist() == [3, 4]
    assert included["choice_trial_after_first_correct"].tolist() == [0, 1]


def test_summarize_post_first_correct_accuracy_groups_combined_left_right():
    """Post-first-correct summaries should produce combined and side-specific curves."""
    detail = pd.DataFrame(
        {
            "correct_side": ["left", "left", "right", "right", "right"],
            "choice_trial_after_first_correct": [0, 1, 0, 1, 2],
            "correct": ["True", "False", "True", "True", "False"],
            "include_trial_in_session_metric": [True, True, True, True, True],
        }
    )

    summary = switch_persistence.summarize_post_first_correct_accuracy(detail)

    combined = summary[summary["correct_group"] == "combined"].sort_values(
        "choice_trial_after_first_correct"
    )
    left = summary[summary["correct_group"] == "left"].sort_values(
        "choice_trial_after_first_correct"
    )
    right = summary[summary["correct_group"] == "right"].sort_values(
        "choice_trial_after_first_correct"
    )
    assert combined["choice_trial_after_first_correct"].tolist() == [0, 1, 2]
    assert combined["n_blocks"].tolist() == [2, 2, 1]
    assert combined["n_correct"].tolist() == [2, 1, 0]
    assert combined["proportion_correct"].tolist() == [1.0, 0.5, 0.0]
    assert left["proportion_correct"].tolist() == [1.0, 0.0]
    assert right["proportion_correct"].tolist() == [1.0, 1.0, 0.0]


def test_save_post_first_correct_accuracy_outputs_writes_detail_and_summary_csvs(tmp_path):
    """Post-first-correct saving should produce inspectable detail and summary tables."""
    block_performance = make_block_performance().iloc[:1]
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": list(range(3)),
            "cur_block": [0, 0, 0],
            "state": [1, 1, 1],
            "action": [0, 1, 1],
            "correct": [0, 1, 1],
            "reward": [0, 1, 1],
            "experimenter_reward_given": [0, 0, 0],
        }
    )

    detail, summary = switch_persistence.save_post_first_correct_accuracy_outputs(
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        processed_data_path=tmp_path,
        sess_id_full="CT999_2026-06-18_120000",
    )

    detail_path = tmp_path / "CT999_2026-06-18_120000_post_first_correct_accuracy_trials.csv"
    summary_path = tmp_path / "CT999_2026-06-18_120000_post_first_correct_accuracy_summary.csv"
    assert detail_path.exists()
    assert summary_path.exists()
    assert detail_path.stat().st_size > 0
    assert summary_path.stat().st_size > 0
    saved_detail = pd.read_csv(detail_path, na_filter=False)
    assert saved_detail.shape == detail.shape
    assert saved_detail["choice_status"].tolist() == detail["choice_status"].tolist()
    assert saved_detail["choice_trial_after_first_correct"].tolist() == ["None", "0", "1"]
    pd.testing.assert_frame_equal(pd.read_csv(summary_path, na_filter=False), summary)
