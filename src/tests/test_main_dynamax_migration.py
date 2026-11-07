"""Synthetic integration contracts for the normal fixed-exemplar workflow."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def test_fresh_main_import_does_not_import_ssm():
    """A real fresh interpreter imports the normal entry point without ssm."""
    completed = subprocess.run(
        [sys.executable, "-c", "import sys; import src.main; assert 'ssm' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True,
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    assert completed.returncode == 0, completed.stderr


def test_normal_runner_reuses_model_only_and_keeps_current_trial_features(tmp_path):
    """Cached block states attach to fresh features with full-row alignment."""
    from src.main import run_dynamax_block_analysis

    session = SimpleNamespace(
        processed_data_path=tmp_path / "processed", figure_path=tmp_path / "figures",
        sess_id_full="synthetic", sess_id_abbreviated="synthetic",
    )
    blocks = pd.DataFrame({
        "block_ix": [3, 8, 11], "trials_to_correct": [2., np.nan, 8.],
        "prev_n_rewarded": [4., 3., 9.], "bias_full_flag": [False, True, False],
    }, index=[5, 9, 15])
    trials = pd.DataFrame({"cur_block": [3, 3, 8, 11], "fresh_feature": [1, 2, 3, 4]})
    first_blocks, first_trials, first_result, first_path = run_dynamax_block_analysis(
        blocks, trials, session, plot=False,
    )
    updated_trials = trials.assign(fresh_feature=[11, 12, 13, 14])
    second_blocks, second_trials, second_result, second_path = run_dynamax_block_analysis(
        blocks, updated_trials, session, plot=False,
    )
    assert first_path == second_path
    assert second_trials["fresh_feature"].tolist() == [11, 12, 13, 14]
    assert second_blocks.index.tolist() == [5, 9, 15]
    assert pd.isna(second_trials.loc[2, "inherited_block_strategy"])
    assert second_trials["inherited_block_bias"].tolist() == [False, False, True, False]
    np.testing.assert_allclose(
        first_result.smoothing_result.smoothed_probs,
        second_result.smoothing_result.smoothed_probs,
    )
    assert first_path.exists()
    assert len(list(first_path.parent.glob("*block*.csv"))) >= 1
    assert first_trials["inherited_block_model"].notna().sum() == 3


def test_multisession_normal_runner_matches_continuous_reference(tmp_path):
    """One joint inference preserves histories and ordered source identity."""
    from src.main import run_dynamax_block_analysis
    from src.behavior_analysis.exemplar_hmm_modeling import compute_fixed_exemplar_block_states

    session = SimpleNamespace(
        processed_data_path=tmp_path, figure_path=tmp_path,
        sess_id_full="synthetic_multisession", sess_id_abbreviated="synthetic",
    )
    blocks = pd.DataFrame({
        "block_ix": [0, 1, 2, 3], "source_session_id": ["a", "a", "b", "b"],
        "source_date": ["2026-01-01"] * 4,
        "trials_to_correct": [2., 2., 12., 12.], "prev_n_rewarded": [3., 8., 3., 8.],
    })
    trials = pd.DataFrame({"cur_block": [0, 1, 2, 3], "source_session_id": ["a", "a", "b", "b"]})
    modeled, inherited, result, _path = run_dynamax_block_analysis(
        blocks, trials, session, plot=False,
    )
    reference = compute_fixed_exemplar_block_states(modeled, result.parameters)
    np.testing.assert_allclose(result.smoothing_result.smoothed_probs, reference.smoothing_result.smoothed_probs)
    assert modeled["prev_n_rewarded"].tolist() == [3., 8., 3., 8.]
    assert inherited["source_session_id"].tolist() == ["a", "a", "b", "b"]


def test_concatenation_records_ordered_manifest_including_empty_sessions():
    """Zero-row sources remain part of the continuous sequence cache identity."""
    from src.main import SavedSessionAnalysis, concatenate_saved_sessions

    sessions = []
    for session_id in ["same_day_a", "same_day_b"]:
        session = SimpleNamespace(sess_id_full=session_id, mouse="synthetic", date="2026-01-01")
        blocks = pd.DataFrame(columns=[
            "block_ix", "prev_rewards_session_centered", "block_side_code", "prev_n_rewarded",
        ])
        trials = pd.DataFrame(columns=["cur_block"])
        sessions.append(SavedSessionAnalysis(session, blocks, trials))
    combined = concatenate_saved_sessions(sessions)
    assert [entry["session_id"] for entry in combined.source_session_manifest] == [
        "same_day_a", "same_day_b",
    ]
    assert [entry["n_blocks"] for entry in combined.source_session_manifest] == [0, 0]


def test_single_workflow_preserves_exports_before_upstream_regeneration(tmp_path, monkeypatch):
    """Pre-migration tables are archived before any preprocessing can overwrite."""
    import src.main as main

    original = tmp_path / "synthetic_block_performance.csv"
    original.write_text("block_ix,hardcoded_strategy\n3,legacy\n", encoding="ascii")
    session = SimpleNamespace(
        processed_data_path=tmp_path, sess_id_full="synthetic", session_info={},
        raw_behavior_folder=tmp_path,
    )

    def stop_before_preprocessing(**kwargs):
        """Stop after export preservation; inputs unused, no returned data."""
        raise RuntimeError("synthetic stop")

    monkeypatch.setattr(main, "load_or_preprocess_session", stop_before_preprocessing)
    with pytest.raises(RuntimeError, match="synthetic stop"):
        main.run_single_session_workflow(session, main.SingleSessionAnalysisConfig())
    archived = list((tmp_path / "exemplar_hmm" / "archives").rglob(original.name))
    assert len(archived) == 1
    assert archived[0].read_bytes() == original.read_bytes()


def test_full_single_workflow_runs_real_dynamax_on_current_features(tmp_path, monkeypatch):
    """Synthetic upstream data reaches real save/inherit/plot model integration."""
    import src.main as main

    blocks = pd.DataFrame({
        "block_ix": [2, 7, 9], "trials_to_correct": [2., np.nan, 8.],
        "prev_n_rewarded": [4., 3., 9.], "bias_full_flag": [False, True, False],
    })
    trials = pd.DataFrame({"cur_block": [2, 7, 9], "fresh_feature": [11, 12, 13]})
    summary = pd.DataFrame({"date": ["2026-01-01"], "prev_n_rewarded_slope": [1.],
                            "prev_n_rewarded_intercept": [0.]})
    session = SimpleNamespace(
        session_info={}, raw_behavior_folder=tmp_path, processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures", sess_id_full="synthetic_full",
        sess_id_abbreviated="synthetic", date="2026-01-01",
    )
    session.processed_data_path.mkdir()
    monkeypatch.setattr(main, "load_or_preprocess_session", lambda **kwargs: (trials, trials, None))
    monkeypatch.setattr(main, "load_or_run_session_analysis", lambda **kwargs: (trials, blocks, summary))
    monkeypatch.setattr(main, "ensure_session_side_bias_metrics", lambda **kwargs: summary)
    monkeypatch.setattr(main, "collect_configured_trial_features", lambda **kwargs: (trials, {}))
    monkeypatch.setattr(main, "plot_session", _no_output)
    for name in ["add_block_explore_counts_from_trials", "add_block_explore_run_counts_from_trials",
                 "_add_single_session_metadata_columns"]:
        monkeypatch.setattr(main, name, _unchanged_blocks)
    for name in ["add_block_agent_mouse_agreement_columns", "add_session_block_collection_variables"]:
        monkeypatch.setattr(main.session_analysis, name, _unchanged_blocks)
    for name in ["save_and_plot_switch_persistence_for_session", "save_and_plot_post_first_correct_accuracy_for_session",
                 "plot_expectant_switching_for_session", "save_and_plot_expectant_model_selectivity_for_session",
                 "plot_mouse_history_ideal_observer_for_session"]:
        monkeypatch.setattr(main, name, _no_output)
    for name in ["plot_session_correct", "plot_session_correct_after_first_correct",
                 "plot_session_history_ideal_mouse_agreement", "plot_session_trials_to_correct",
                 "plot_session_nswitches", "plot_block_bias_quadrants",
                 "plot_session_block_trials_to_correct_summary", "plot_session_zero_trials_to_correct_fraction",
                 "plot_session_side_bias_ratios", "plot_session_signed_side_bias", "scatter_trials_to_correct",
                 "plot_session_explore_trials", "plot_session_explore_runs", "plot_session_agent_mouse_agreement",
                 "plot_single_session_summary_grid"]:
        monkeypatch.setattr(main.performance_plots, name, _no_output)
    main.run_single_session_workflow(session, main.SingleSessionAnalysisConfig(choice_window_sizes=()))
    saved_trials = pd.read_csv(session.processed_data_path / "synthetic_full_augmented_trials.csv")
    assert saved_trials["fresh_feature"].tolist() == [11, 12, 13]
    assert saved_trials["inherited_block_model"].notna().sum() == 2
    assert list(session.figure_path.glob("*.png"))
    assert list((session.processed_data_path / "exemplar_hmm").rglob("results.npz"))


def _no_output(*args, **kwargs):
    """Accept arbitrary unrelated UI/summary inputs and return no data."""
    return None


def _unchanged_blocks(block_performance, *args, **kwargs):
    """Return the input N-row block table unchanged, in its original units."""
    return block_performance
