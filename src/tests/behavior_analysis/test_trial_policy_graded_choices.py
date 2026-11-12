"""Fixed-temperature logistic choices and their likelihood/provenance contracts."""

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, log_expit

from src.behavior_analysis import trial_policy_analysis as analysis
from src.behavior_analysis import trial_policy_models as models
from src.behavior_analysis import trial_policy_results as results
from src.behavior_analysis.gather_trial_features import TaskParams


TEMPERATURES = {"q_learning": .14, "forgetting_q_learning": .14,
                "bayesian_hmm": .40, "reward_decay_hmm": .40, "ideal_observer": .40}


@pytest.fixture
def trials():
    """Return chronological binary observations (T=6), action 0=Right/1=Left."""
    return pd.DataFrame({"cur_trial": [10, 12, 15, 18, 20, 25], "cur_block": [1, 1, 2, 2, 2, 3],
                         "action": [1, 0, 1, 0, 1, 0], "reward": [1, 0, 1, 0, 0, 1]},
                        index=[8, 4, 5, 9, 3, 2])


def compare(table, ids=None, temperatures=None):
    """Table[T,C]/ordered IDs/temperature overrides yield history, choices, result[T,K]."""
    candidates = models.resolve_trial_policy_candidates(models.DEFAULT_MODEL_IDS if ids is None else ids)
    choices = models.effective_trial_policy_choice_parameters(candidates, temperatures=temperatures)
    history = models.prepare_trial_policy_history(table, session_id="graded_session")
    scores, available = models.replay_trial_policy_candidates(history, candidates, TaskParams())
    comparison = analysis.compute_trial_policy_comparison(
        history, scores, available, tuple(candidate.model_id for candidate in candidates), choice_parameters=choices)
    return history, choices, comparison


def test_approved_five_logistic_rules_and_two_discrete_rules():
    """No inputs; validate the approved choice rules and temperatures in score units."""
    settings = models.effective_trial_policy_choice_parameters(models.resolve_trial_policy_candidates())
    for model_id, temperature in TEMPERATURES.items():
        assert settings[model_id] == {"rule": "logistic", "temperature": temperature}
    for model_id in ("win_stay_lose_switch", "immediate_persistence"):
        assert settings[model_id] == {"rule": "fixed_error", "opposite_action_error_prob": .1}


def test_mixed_choice_probabilities_and_bernoulli_likelihoods(trials):
    """Table[T,C] yields model-specific p_right and observed-action log likelihood[T,K]."""
    _, settings, comparison = compare(trials)
    for column, model_id in enumerate(comparison.model_ids):
        score = comparison.policy_scores[:, column]
        if model_id in TEMPERATURES:
            drive = score / TEMPERATURES[model_id]
            expected_right = expit(-drive)
            expected_ll = np.where(trials.action.to_numpy() == 1, log_expit(drive), log_expit(-drive))
        else:
            expected_right = np.where(score > 0, .1, np.where(score < 0, .9, .5))
            expected_ll = np.where(trials.action.to_numpy() == 0,
                                   np.log(expected_right), np.log1p(-expected_right))
        np.testing.assert_allclose(comparison.p_right[:, column], expected_right, atol=1e-15)
        np.testing.assert_allclose(comparison.log_likelihoods[:, column], expected_ll, atol=1e-15)
    assert settings["ideal_observer"]["temperature"] == .4
    q_column = list(comparison.model_ids).index("q_learning")
    assert comparison.p_right[1, q_column] == pytest.approx(expit(-.3 / .14))
    assert comparison.p_right[1, q_column] != .1


def test_logistic_zero_symmetry_monotonicity_no_cap_and_stable_extreme_likelihoods(trials):
    """Table fixture and synthetic scores[T,1] preserve exact ties and finite log evidence."""
    history = models.prepare_trial_policy_history(trials.assign(action=0), session_id="graded_session")
    scores = np.array([[-100.], [-.1], [0.], [.1], [1.], [100.]])
    settings = {"test": {"rule": "logistic", "temperature": .1}}
    comparison = analysis.compute_trial_policy_comparison(
        history, scores, np.ones_like(scores, bool), ("test",), choice_parameters=settings)
    assert comparison.p_right[2, 0] == .5 and comparison.model_indifferent[2, 0]
    assert np.all(np.diff(comparison.p_right[:, 0]) < 0)
    assert comparison.p_right[0, 0] > .9 and comparison.p_right[-1, 0] < .1
    assert comparison.p_right[1, 0] == pytest.approx(1 - comparison.p_right[3, 0])
    assert np.isfinite(comparison.log_likelihoods).all()
    assert comparison.log_likelihoods[-1, 0] == pytest.approx(-1000.)


def test_fql_decay_now_weakens_choice_probability_without_changing_history():
    """No inputs; a reward followed by omissions yields p_left approaching 0.5."""
    table = pd.DataFrame({"cur_trial": np.arange(23), "cur_block": 1,
                          "action": 1, "reward": [1] + [0] * 22})
    _, _, comparison = compare(table, ("forgetting_q_learning",))
    expected_scores = np.r_[0, .3 * .7 ** np.arange(22)]
    np.testing.assert_allclose(comparison.policy_scores[:, 0], expected_scores, atol=1e-15)
    assert 1 - comparison.p_right[1, 0] == pytest.approx(.895, abs=1e-4)
    assert 1 - comparison.p_right[11, 0] == pytest.approx(.5151, abs=1e-4)
    assert .5 < 1 - comparison.p_right[-1, 0] < .501


@pytest.mark.parametrize("value", [0, -.1, np.nan, np.inf, "invalid"])
def test_invalid_fixed_temperatures_raise(value):
    """Scalar temperature must be finite and positive, with candidate context."""
    with pytest.raises(ValueError, match="q_learning.*temperature|temperature.*q_learning"):
        models.effective_trial_policy_choice_parameters(
            models.resolve_trial_policy_candidates(), temperatures={"q_learning": value})


@pytest.mark.parametrize("model_id", ["unimplemented", "immediate_persistence", "win_stay_lose_switch"])
def test_temperature_overrides_only_target_graded_candidates(model_id):
    """String model ID rejects unknown or discrete-model temperature overrides."""
    with pytest.raises(ValueError, match=model_id):
        models.effective_trial_policy_choice_parameters(
            models.resolve_trial_policy_candidates(), temperatures={model_id: .2})


def test_partial_overrides_subsets_and_candidate_reordering_are_explicit(trials):
    """Table[T,C] and overrides keep defaults fixed and resolve by stable IDs."""
    history, settings, baseline = compare(trials)
    _, changed, overridden = compare(trials, temperatures={"q_learning": .3})
    assert changed["q_learning"]["temperature"] == .3
    for column, model_id in enumerate(baseline.model_ids):
        if model_id != "q_learning":
            np.testing.assert_array_equal(overridden.p_right[:, column], baseline.p_right[:, column])
    subset = models.resolve_trial_policy_candidates(("q_learning",))
    selected = models.effective_trial_policy_choice_parameters(subset, temperatures=TEMPERATURES)
    assert selected == {"q_learning": settings["q_learning"]}
    permuted = analysis.compute_trial_policy_comparison(
        history, baseline.policy_scores[:, ::-1], baseline.prediction_valid[:, ::-1],
        tuple(baseline.model_ids[::-1]), choice_parameters=settings)
    np.testing.assert_allclose(permuted.p_right[:, ::-1], baseline.p_right)
    expected = analysis.smooth_trial_policy_comparison(baseline)
    actual = analysis.smooth_trial_policy_comparison(permuted)
    np.testing.assert_allclose(actual.hmm_smoothed_probs[:, ::-1], expected.hmm_smoothed_probs, atol=2e-6)


def test_graded_exclusions_missing_reward_and_causality_keep_existing_replay_rules(trials):
    """Table[T,C] keeps row axes and score histories under exclusions and missing reward."""
    changed = trials.astype({"reward": float}).assign(experimenter_reward_given=[0, 1, 0, 0, 0, 0])
    changed.iloc[2, changed.columns.get_loc("reward")] = np.nan
    original = changed.copy(deep=True)
    history, _, result = compare(changed)
    assert not result.comparison_valid[1] and np.isnan(result.p_right[1]).all()
    assert result.comparison_valid[2] and not result.history_update_valid[2]
    candidates = models.resolve_trial_policy_candidates()
    expected_scores, _ = models.replay_trial_policy_candidates(history, candidates, TaskParams())
    np.testing.assert_array_equal(result.policy_scores, expected_scores)
    later = changed.copy()
    later.iloc[3:, later.columns.get_loc("action")] = 1 - later.iloc[3:].action
    _, _, altered = compare(later)
    np.testing.assert_array_equal(altered.p_right[:4], result.p_right[:4])
    pd.testing.assert_frame_equal(changed, original)


def test_graded_scores_settings_cache_identity_and_legacy_loading(trials, tmp_path):
    """Table/Path fixtures persist scores[T,K], effective rules and old archive compatibility."""
    history, settings, comparison = compare(trials)
    candidates = models.resolve_trial_policy_candidates()
    metadata = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test")
    assert metadata["analysis_version"] == "trial_policy_hmm_v5"
    assert metadata["candidate_choice_parameters"] == settings
    changed = results.build_trial_policy_metadata(history, candidates, TaskParams(), parameter_source="test",
                                                  choice_temperatures={"q_learning": .3})
    assert changed["signature"] != metadata["signature"]
    path = results.save_trial_policy_results(comparison, processed_data_path=tmp_path, metadata=metadata)
    loaded, saved = results.load_trial_policy_results(path)
    np.testing.assert_array_equal(loaded.policy_scores, comparison.policy_scores)
    assert saved["candidate_choice_parameters"] == settings
    attached = results.attach_trial_policy_outputs(trials, loaded, metadata=saved)
    np.testing.assert_array_equal(attached.trial_policy_policy_scores_q_learning,
                                  comparison.policy_scores[:, list(comparison.model_ids).index("q_learning")])
    legacy = tmp_path / "v4.npz"
    arrays = {field.name: getattr(comparison, field.name) for field in fields(comparison)
              if field.name not in ("status", "policy_scores") and getattr(comparison, field.name) is not None}
    np.savez_compressed(legacy, **arrays, meta={"analysis_version": "trial_policy_hmm_v4", "status": "ok"})
    older, _ = results.load_trial_policy_results(legacy)
    assert older.policy_scores is None and older.p_right.shape == comparison.p_right.shape
