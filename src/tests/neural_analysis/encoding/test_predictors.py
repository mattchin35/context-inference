"""Joint trial-fixed context/choice and choice-aligned outcome contracts."""

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.encoding.predictors import (
    build_joint_predictors, classify_choice_outcomes, encode_context,
    select_baseline_trials, validate_trial_tables, diagnose_context_events,
)


@pytest.fixture
def trials():
    """Return six ordinary outcome categories, irregular trial IDs, UTC seconds, and nontrivial index labels."""
    return pd.DataFrame({
        "cur_trial": [10, 12, 17, 19, 24, 30], "cur_block": [0, 1, 2, 3, 4, 5],
        "start_time": 100. + 10*np.arange(6), "choice_time": 104. + 10*np.arange(6),
        "state_int": [1, 0, 1, 0, 0, 1], "action": [1, 0, 1, 0, 1, 0],
        "reward": [1, 1, 0, 0, 0, 0], "correct": [1, 1, 1, 1, 0, 0],
        "experimenter_reward_given": np.zeros(6, dtype=int),
    }, index=[90, 5, 33, 8, 12, 1])


def test_classification_uses_chosen_side_and_distinguishes_all_six_outcomes(trials):
    """Input six source rows; return no data, assert six native outcome names and no mutation."""
    before = trials.copy(deep=True)
    codes, valid, reasons = classify_choice_outcomes(trials.action, trials.reward, trials.correct)
    assert codes.tolist() == ["left_reward", "right_reward", "left_omission", "right_omission", "left_incorrect", "right_incorrect"]
    assert valid.all() and not reasons.any()
    pd.testing.assert_frame_equal(trials, before)


@pytest.mark.parametrize("width", [0.5, 0.1, 0.05, 0.02])
@pytest.mark.parametrize("encoding", ["signed_context", "separate_context"])
def test_entire_one_second_halves_hold_trial_values_and_gate_only_outcomes(trials, width, encoding):
    """Input trials/seconds/native encoding; assert trial-major X(N,P) with exactly half its bins post-choice."""
    edges = np.linspace(-1, 1, round(2/width)+1)
    result = build_joint_predictors(trials, np.arange(6), edges, encoding)
    names = result["column_names"]
    x = result["predictors"].reshape(6, len(edges)-1, -1)
    post = edges[:-1] >= 0
    assert result["valid"].all() and x.shape[-1] == (8 if encoding == "signed_context" else 9)
    np.testing.assert_array_equal(x[:, :, names.index("signed_choice")], np.repeat([1,-1,1,-1,1,-1], len(post)).reshape(6,-1))
    if encoding == "signed_context":
        np.testing.assert_array_equal(x[:, :, 0], np.repeat([1,-1,1,-1,-1,1], len(post)).reshape(6,-1))
    else:
        np.testing.assert_array_equal(x[:, :, 0]+x[:, :, 1], np.ones((6, len(post))))
    for row, name in enumerate(["left_reward", "right_reward", "left_omission", "right_omission", "left_incorrect", "right_incorrect"]):
        outcomes = x[row, :, -6:]
        np.testing.assert_array_equal(outcomes[~post], np.zeros((sum(~post),6)))
        np.testing.assert_array_equal(outcomes[post].sum(axis=1), np.ones(sum(post)))
        np.testing.assert_array_equal(x[row,:,names.index(name)], post.astype(float))
        assert result["condition_counts"][name] == 1
    assert result["valid"].reshape(6,-1)[-1].all()  # Terminal trial needs no next state.


@pytest.mark.parametrize("field,value", [("reward", np.nan), ("reward", 3), ("correct", np.nan), ("correct", 9)])
def test_unknown_outcome_excludes_both_halves_not_an_all_zero_negative(trials, field, value):
    """Input source field/value; unknown unrewarded outcome masks the complete window with a reason."""
    trials.iloc[2, trials.columns.get_loc(field)] = value
    result = build_joint_predictors(trials, np.arange(6), np.array([-1,-.5,0,.5,1]), "signed_context")
    assert not result["valid"].reshape(6,4)[2].any()
    assert all(result["invalid_reason"].reshape(6,4)[2])


def test_recorded_reward_does_not_require_inferred_correctness(trials):
    """Known rewarded choice remains reward-classified when correctness is unavailable."""
    trials.iloc[0, trials.columns.get_loc("correct")] = np.nan
    codes, valid, _ = classify_choice_outcomes(trials.action, trials.reward, trials.correct)
    assert valid[0] and codes[0] == "left_reward"


@pytest.mark.parametrize("edges", [np.array([-2,-1,0,1,2]), np.array([-1,-.1,.1,1]), np.array([-1,0,0,1]), np.array([-1,0,.9])])
def test_joint_geometry_is_exact_and_cannot_straddle_choice(trials, edges):
    """Input malformed relative-second edges; reject a different or ambiguous scientific window."""
    with pytest.raises(ValueError, match="edge|window|zero|bin"):
        build_joint_predictors(trials, np.arange(6), edges, "signed_context")


@pytest.mark.parametrize("field,value,reason", [("experimenter_reward_given",1,"manual"), ("state_int",2,"dark"), ("action","no_choice","choice")])
def test_baseline_exclusion_precedes_predictor_eligibility(trials, field, value, reason):
    """Manual/dark/no-choice trials cannot become joint observations through outcome coding."""
    trials[field] = trials[field].astype(object)
    trials.iloc[0,trials.columns.get_loc(field)] = value
    baseline = select_baseline_trials(trials)
    assert not baseline.baseline_valid.iloc[0]
    assert reason in baseline.baseline_invalid_reason.iloc[0]
    result = build_joint_predictors(trials, np.arange(6), np.array([-1,-.5,0,.5,1]), "signed_context")
    assert not result["valid"][:4].any()


@pytest.mark.parametrize("field", ["action","reward","correct","state_int","cur_trial","choice_time","start_time"])
def test_complete_source_correspondence_includes_every_joint_field(trials, field):
    """Input source field name; mismatched source/augmented values cannot silently define a new design."""
    source = trials.copy(deep=True)
    source.iloc[0,source.columns.get_loc(field)] += 1
    with pytest.raises(ValueError, match=field):
        validate_trial_tables(trials, source)


def test_matching_tables_ignore_index_labels_and_retain_native_context_encoding(trials):
    """Index labels do not change original positions; separate context retains native 0/1 coding."""
    validate_trial_tables(trials, trials.reset_index(drop=True))
    x, valid, names = encode_context(np.array([0,1,2,np.nan]), "separate_context")
    np.testing.assert_array_equal(x[:2], [[0,1],[1,0]])
    assert valid.tolist() == [True,True,False,False] and np.isnan(x[2:]).all()
    assert names == ("left_context","right_context")


def test_context_event_diagnostics_remain_advisory(trials):
    """A mismatching logged event is diagnosed and never overwrites the fixed trial context."""
    result = build_joint_predictors(trials,np.arange(6),np.array([-1,-.5,0,.5,1]),"signed_context")
    messages = diagnose_context_events(trials,pd.DataFrame({"Time":[99.],"Event":["enter_dark_period"]}))
    assert messages and result["predictors"][0,0] == 1
