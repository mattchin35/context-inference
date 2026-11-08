"""Synthetic single/joint workflow contracts for default trial-policy inference."""

from pathlib import Path
from types import SimpleNamespace
import json
import threading

import numpy as np
import pandas as pd
import pytest

from src.behavior_analysis.gather_trial_features import TaskParams


def source_session(root, identity="source", *, flag_column=None):
    """Path/ID/optional flag name yield saved-session metadata and binary T=6 inputs."""
    from src.main import SavedSessionAnalysis

    session = SimpleNamespace(sess_id_full=identity, sess_id_abbreviated=identity,
                              date="2026-01-01", mouse="synthetic", session_info={},
                              raw_behavior_folder=root / identity / "raw",
                              processed_data_path=root / identity / "processed",
                              figure_path=root / identity / "figures")
    session.processed_data_path.mkdir(parents=True)
    trials = pd.DataFrame({"cur_trial": [1, 2, 3, 4, 5, 6], "cur_block": [1, 1, 1, 2, 2, 2],
                           "action": [0, 1, 0, 1, 1, 0], "reward": [1, 0, 1, 1, 0, 0],
                           "fresh_feature": np.arange(6)})
    if flag_column:
        trials[flag_column] = [0, 0, 1, 0, 0, 0]
    blocks = pd.DataFrame({"block_ix": [0, 1], "trials_to_correct": [2., 4.],
                           "prev_n_rewarded": [3., 6.], "prev_rewards_session_centered": [-1.5, 1.5],
                           "block_side_code": [-1, 1]})
    return SavedSessionAnalysis(session, blocks, trials)


def test_default_single_runner_saves_aligned_outputs_and_reuses_only_arrays(tmp_path, monkeypatch):
    """A real runner preserves current features/block fields on numerical cache hits."""
    import src.main as main
    from src.behavior_analysis import trial_policy_models as models

    source = source_session(tmp_path)
    trials = source.augmented_trial_df.assign(inherited_block_model="fixed")
    attached, result, path = main.run_dynamax_trial_analysis(trials, source.session, feature_params=TaskParams(), plot=True)
    assert result.p_right.shape == (6, 3) and path.exists()
    assert attached["inherited_block_model"].tolist() == ["fixed"] * 6
    assert len(list(source.session.figure_path.glob("trial_policy_hmm/*/*.png"))) == 1
    before = path.read_bytes()

    def reject_replay(*args, **kwargs):
        """Arbitrary replay arguments are forbidden for a valid cache hit."""
        pytest.fail("matching numerical cache must not replay policies")

    monkeypatch.setattr(models, "replay_trial_policy_candidates", reject_replay)
    fresh = trials.assign(fresh_feature=99)
    current, reused, second_path = main.run_dynamax_trial_analysis(fresh, source.session, feature_params=TaskParams(), plot=False)
    assert path == second_path and path.read_bytes() == before
    assert current["fresh_feature"].tolist() == [99] * 6
    saved = pd.read_csv(source.session.processed_data_path / "source_augmented_trials.csv")
    assert saved["fresh_feature"].tolist() == [99] * 6
    assert len(list(path.parent.parent.glob("*/augmented_trials.csv"))) == 2
    np.testing.assert_equal(result.hmm_smoothed_probs, reused.hmm_smoothed_probs)


def test_smoothing_disabled_refresh_clears_saved_hmm_fields(tmp_path):
    """A rerun with smoothing off removes previous HMM columns and retains independent ones."""
    import src.main as main

    source = source_session(tmp_path)
    current, _, _ = main.run_dynamax_trial_analysis(source.augmented_trial_df, source.session,
                                                   feature_params=TaskParams(), plot=False)
    assert any(column.startswith("trial_policy_hmm_") for column in current)
    current, result, path = main.run_dynamax_trial_analysis(current, source.session,
                                                          feature_params=TaskParams(), smoothing=False, plot=False)
    assert result.hmm_smoothed_probs is None
    assert not any(column.startswith("trial_policy_hmm_") for column in current)
    assert "trial_policy_independent_model" in current


def test_validation_failure_records_parent_error_without_success_artifact(tmp_path):
    """Malformed inputs produce a contextual parent log before numerical signature exists."""
    import src.main as main

    source = source_session(tmp_path)
    bad = source.augmented_trial_df.assign(action=[0, .5, 0, 1, 1, 0])
    with pytest.raises(ValueError, match="source.*2.*action"):
        main.run_dynamax_trial_analysis(bad, source.session, feature_params=TaskParams(), plot=False)
    logs = list(source.session.processed_data_path.glob("trial_policy_hmm/runs/*/run.log"))
    assert len(logs) == 1 and "action" in logs[0].read_text()
    assert not list(source.session.processed_data_path.glob("trial_policy_hmm/*/*/*/results.npz"))


def test_joint_results_equal_separate_sources_with_mixed_schemas_and_parameters(tmp_path, monkeypatch):
    """Serial/threaded joint runs reset each history, preserve local IDs and source files."""
    import src.main as main
    from src.behavior_analysis import trial_policy_models as models
    from src.behavior_analysis import trial_policy_results as persistence

    sources = [source_session(tmp_path, "first"), source_session(tmp_path, "second", flag_column="give_reward")]
    parameters = {"first": TaskParams(FQL_decay=.6), "second": TaskParams(state_transition_prob=.1)}
    references, source_csvs = [], {}
    for source in sources:
        _, reference, _ = main.run_dynamax_trial_analysis(source.augmented_trial_df, source.session,
                                                         feature_params=parameters[source.session.sess_id_full], plot=False)
        references.append(reference)
        csv = source.session.processed_data_path / f"{source.session.sess_id_full}_augmented_trials.csv"
        source_csvs[csv] = csv.read_bytes()
    combined = main.concatenate_saved_sessions(sources)
    combined.augmented_trial_df["trial_policy_stale"] = 99
    joint = main.build_multisession_session("synthetic", tmp_path / "cross_session_analysis")
    calls = []
    replay = models.replay_trial_policy_candidates
    save = persistence.save_trial_policy_results

    def record_replay(*args, **kwargs):
        """Record numerical thread identity and delegate (T,K) policy replay."""
        calls.append(threading.current_thread().name)
        return replay(*args, **kwargs)

    def parent_save(*args, **kwargs):
        """Require artifact writes on the parent thread; delegate path-returning save."""
        assert threading.current_thread() is threading.main_thread()
        return save(*args, **kwargs)

    monkeypatch.setattr(models, "replay_trial_policy_candidates", record_replay)
    monkeypatch.setattr(persistence, "save_trial_policy_results", parent_save)
    serial = main.run_multisession_analysis(combined, joint, source_sessions=sources,
                                            trial_feature_params_by_session=parameters,
                                            trial_policy_n_jobs=1, rerun_trial_policy_analysis=True)
    assert len(serial) == 4 and serial[:2] == (None, None)
    serial_calls = calls.copy()
    calls.clear()
    parallel = main.run_multisession_analysis(combined, joint, source_sessions=sources,
                                              trial_feature_params_by_session=parameters,
                                              trial_policy_n_jobs=2, rerun_trial_policy_analysis=True)
    assert all(name == threading.main_thread().name for name in serial_calls)
    assert calls and all(name != threading.main_thread().name for name in calls)
    pd.testing.assert_frame_equal(serial[3], parallel[3])
    output = parallel[3]
    assert "trial_policy_stale" not in output
    assert "inherited_block_model" in output
    for source, reference in zip(sources, references):
        part = output.loc[output.source_session_id.eq(source.session.sess_id_full)]
        for column, model_id in enumerate(reference.model_ids):
            np.testing.assert_allclose(part[f"trial_policy_p_right_{model_id}"], reference.p_right[:, column], equal_nan=True)
            np.testing.assert_allclose(part[f"trial_policy_hmm_smoothed_probs_{model_id}"], reference.hmm_smoothed_probs[:, column])
        assert part["cur_trial"].tolist() == [1, 2, 3, 4, 5, 6]
        assert part["source_cur_block"].tolist() == [1, 1, 1, 2, 2, 2]
    for path, content in source_csvs.items():
        assert path.read_bytes() == content
    assert list(joint.figure_path.glob("*trial-policy*.png"))
    assert list(joint.figure_path.glob("trial_policy_hmm/*/metadata.json"))


def test_joint_failure_leaves_fresh_block_export_without_stale_trial_fields(tmp_path):
    """A bad source stops joint trial publication after the block export succeeds."""
    import src.main as main

    sources = [source_session(tmp_path, "first"), source_session(tmp_path, "bad")]
    sources[1].augmented_trial_df["reward"] = [1, "broken", 1, 0, 1, 0]
    combined = main.concatenate_saved_sessions(sources)
    combined.augmented_trial_df["trial_policy_stale"] = 99
    joint = main.build_multisession_session("synthetic", tmp_path / "joint")
    with pytest.raises(ValueError, match="bad.*2.*reward"):
        main.run_multisession_analysis(combined, joint, source_sessions=sources, trial_policy_n_jobs=1)
    current = pd.read_csv(joint.processed_data_path / f"{joint.sess_id_full}_augmented_trials.csv")
    assert "inherited_block_model" in current
    assert not any(column.startswith("trial_policy_") for column in current)
    assert not list(joint.figure_path.glob("*trial-policy*.png"))


def test_joint_disabled_and_missing_metadata_contract(tmp_path):
    """Block-only direct calls need no trial source metadata and clear stale fields."""
    import src.main as main

    sources = [source_session(tmp_path)]
    combined = main.concatenate_saved_sessions(sources)
    combined.augmented_trial_df["trial_policy_stale"] = 99
    joint = main.build_multisession_session("synthetic", tmp_path / "joint")
    output = main.run_multisession_analysis(combined, joint, run_trial_policy_analysis=False)[3]
    assert "trial_policy_stale" not in output
    assert not list(sources[0].session.processed_data_path.glob("trial_policy_hmm"))
    with pytest.raises(ValueError, match="source_sessions"):
        main.run_multisession_analysis(combined, joint)


def test_joint_saved_parameters_overrides_and_empty_sources(tmp_path):
    """Saved partial parameters default explicitly; complete overrides win; empty sources persist."""
    import src.main as main
    from src.behavior_analysis.trial_policy_results import load_trial_policy_results

    sources = [source_session(tmp_path, "first"), source_session(tmp_path, "empty")]
    sources[1].augmented_trial_df = sources[1].augmented_trial_df.iloc[:0]
    sources[1].block_performance = sources[1].block_performance.iloc[:0]
    parameter_file = sources[0].session.processed_data_path / "trial_feature_params.json"
    parameter_file.write_text(json.dumps({"FQL_decay": .9}), encoding="ascii")
    combined = main.concatenate_saved_sessions(sources)
    joint = main.build_multisession_session("synthetic", tmp_path / "joint")
    main.run_multisession_analysis(combined, joint, source_sessions=sources, trial_policy_n_jobs=1)
    first_artifact = sorted(sources[0].session.processed_data_path.glob("trial_policy_hmm/*/*/*/results.npz"))[-1]
    _, first_meta = load_trial_policy_results(first_artifact)
    assert first_meta["candidate_parameters"]["forgetting_q_learning"]["decay"] == .9
    assert first_meta["candidate_parameters"]["bayesian_hmm"]["state_transition_prob"] == .2
    assert "trial_feature_params.json" in first_meta["parameter_source"]
    empty_artifact = next(sources[1].session.processed_data_path.glob("trial_policy_hmm/*/*/*/results.npz"))
    empty_result, empty_meta = load_trial_policy_results(empty_artifact)
    assert empty_result.status == "empty" and empty_result.p_right.shape == (0, 3)
    before = set(sources[0].session.processed_data_path.glob("trial_policy_hmm/*/*/*/results.npz"))
    main.run_multisession_analysis(combined, joint, source_sessions=sources, trial_policy_n_jobs=1,
                                   trial_feature_params_by_session={"first": TaskParams(FQL_decay=.6)})
    after = set(sources[0].session.processed_data_path.glob("trial_policy_hmm/*/*/*/results.npz"))
    override_artifact, = after - before
    _, override_meta = load_trial_policy_results(override_artifact)
    assert override_meta["candidate_parameters"]["forgetting_q_learning"]["decay"] == .6


def test_malformed_saved_parameters_raise_with_source_context(tmp_path):
    """A damaged source JSON is an explicit workflow failure, without fallback defaults."""
    import src.main as main

    source = source_session(tmp_path, "bad_parameters")
    (source.session.processed_data_path / "trial_feature_params.json").write_text("{broken", encoding="ascii")
    combined = main.concatenate_saved_sessions([source])
    joint = main.build_multisession_session("synthetic", tmp_path / "joint")
    with pytest.raises(ValueError, match="bad_parameters.*parameter"):
        main.run_multisession_analysis(combined, joint, source_sessions=[source], trial_policy_n_jobs=1)


@pytest.mark.parametrize("enabled,failing", [(True, False), (False, False), (True, True)])
def test_full_single_workflow_save_order_default_disabled_and_failed(tmp_path, monkeypatch, enabled, failing):
    """The real trial runner follows block attachment and current exports never retain stale results."""
    import src.main as main

    source = source_session(tmp_path)
    trials = source.augmented_trial_df.assign(trial_policy_stale=99)
    summary = pd.DataFrame({"date": [source.session.date], "prev_n_rewarded_slope": [1.], "prev_n_rewarded_intercept": [0.]})
    monkeypatch.setattr(main, "load_or_preprocess_session", lambda **kwargs: (trials, trials, None))
    monkeypatch.setattr(main, "load_or_run_session_analysis", lambda **kwargs: (trials, source.block_performance, summary))
    monkeypatch.setattr(main, "ensure_session_side_bias_metrics", lambda **kwargs: summary)
    monkeypatch.setattr(main, "collect_configured_trial_features", lambda **kwargs: (trials, TaskParams()))
    for name in ("add_block_explore_counts_from_trials", "add_block_explore_run_counts_from_trials", "_add_single_session_metadata_columns"):
        monkeypatch.setattr(main, name, unchanged)
    for name in ("add_block_agent_mouse_agreement_columns", "add_session_block_collection_variables"):
        monkeypatch.setattr(main.session_analysis, name, unchanged)
    for name in ("plot_session", "save_and_plot_switch_persistence_for_session", "save_and_plot_post_first_correct_accuracy_for_session",
                 "plot_expectant_switching_for_session", "save_and_plot_expectant_model_selectivity_for_session", "plot_mouse_history_ideal_observer_for_session"):
        monkeypatch.setattr(main, name, no_output)
    for name in ("plot_session_correct", "plot_session_correct_after_first_correct", "plot_session_history_ideal_mouse_agreement",
                 "plot_session_trials_to_correct", "plot_session_nswitches", "plot_block_bias_quadrants", "plot_session_block_trials_to_correct_summary",
                 "plot_session_zero_trials_to_correct_fraction", "plot_session_side_bias_ratios", "plot_session_signed_side_bias", "scatter_trials_to_correct",
                 "plot_session_explore_trials", "plot_session_explore_runs", "plot_session_agent_mouse_agreement", "plot_single_session_summary_grid"):
        monkeypatch.setattr(main.performance_plots, name, no_output)
    if failing:
        trials["action"] = [0, .5, 0, 1, 1, 0]
    config = main.SingleSessionAnalysisConfig(choice_window_sizes=(), run_trial_policy_analysis=enabled)
    assert main.SingleSessionAnalysisConfig().run_trial_policy_analysis is True
    if failing:
        with pytest.raises(ValueError, match="action"):
            main.run_single_session_workflow(source.session, config)
    else:
        main.run_single_session_workflow(source.session, config)
    current = pd.read_csv(source.session.processed_data_path / "source_augmented_trials.csv")
    assert "inherited_block_model" in current and "fresh_feature" in current
    assert "trial_policy_stale" not in current
    assert ("trial_policy_comparison_valid" in current) is (enabled and not failing)


def unchanged(table, *args, **kwargs):
    """Return first DataFrame argument unchanged; other test orchestration inputs unused."""
    return table


def no_output(*args, **kwargs):
    """Accept arbitrary unrelated plotting arguments and return None."""
    return None
