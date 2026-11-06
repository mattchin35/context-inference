"""Workflow tests ensuring windows supplement existing context-block outputs."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.tests.test_main_multisession import load_main_module


@pytest.fixture
def main_module():
    """Return main.py with the existing lightweight dependency stubs."""
    return load_main_module()


@pytest.mark.parametrize("sizes", [(10, 20), ()])
@pytest.mark.parametrize("cached_hmm", [False, True])
def test_single_session_keeps_block_outputs_and_hmm_inputs(
    main_module, tmp_path, monkeypatch, sizes, cached_hmm
):
    """Windows run after block saving/plotting and never alter HMM inputs."""
    main = main_module
    config = main.SingleSessionAnalysisConfig(
        choice_window_sizes=sizes, skip_block_hmm_if_existing=cached_hmm
    )
    trial = pd.DataFrame({"action": [0] * 23, "Qlearning_rel_value": [-1.0] * 23})
    blocks = pd.DataFrame({"block_ix": [0, 1], "qlearning_mouse_agreement": [0.5, 0.8]})
    summary = pd.DataFrame({"date": ["2026-11-06"], "prev_n_rewarded_slope": [1.0],
                            "prev_n_rewarded_intercept": [0.0]})
    session = SimpleNamespace(
        session_info={}, raw_behavior_folder=tmp_path, processed_data_path=tmp_path,
        figure_path=tmp_path, sess_id_full="session", sess_id_abbreviated="session",
        date="2026-11-06",
    )
    calls = []
    monkeypatch.setattr(main, "load_or_preprocess_session", lambda **kwargs: (trial, trial, None))
    monkeypatch.setattr(main, "plot_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "load_or_run_session_analysis", lambda **kwargs: (trial, blocks, summary))
    monkeypatch.setattr(main, "ensure_session_side_bias_metrics", lambda **kwargs: summary)
    monkeypatch.setattr(main, "collect_configured_trial_features", lambda **kwargs: (trial, {"alpha": 0.1}))
    for name in ["add_block_explore_counts_from_trials", "add_block_explore_run_counts_from_trials",
                 "_add_single_session_metadata_columns"]:
        monkeypatch.setattr(main, name, return_blocks)
    monkeypatch.setattr(main.session_analysis, "add_session_block_collection_variables", return_blocks, raising=False)

    def block_agreement(*args, **kwargs):
        """Record the original context-block computation; return its table."""
        calls.append("blocks")
        return blocks

    monkeypatch.setattr(main.session_analysis, "add_block_agent_mouse_agreement_columns", block_agreement, raising=False)
    for name in ["plot_session_correct", "plot_session_correct_after_first_correct",
                 "plot_session_history_ideal_mouse_agreement", "plot_session_trials_to_correct",
                 "plot_session_nswitches", "plot_block_bias_quadrants",
                 "plot_session_block_trials_to_correct_summary", "plot_session_zero_trials_to_correct_fraction",
                 "plot_session_side_bias_ratios", "plot_session_signed_side_bias",
                 "scatter_trials_to_correct", "plot_session_explore_trials", "plot_session_explore_runs",
                 "plot_single_session_summary_grid"]:
        monkeypatch.setattr(main.performance_plots, name, no_op, raising=False)
    monkeypatch.setattr(main.performance_plots, "plot_session_agent_mouse_agreement",
                        lambda *args, **kwargs: calls.append("block_plot"), raising=False)
    for name in ["save_and_plot_switch_persistence_for_session", "save_and_plot_post_first_correct_accuracy_for_session",
                 "plot_expectant_switching_for_session", "save_and_plot_expectant_model_selectivity_for_session",
                 "plot_mouse_history_ideal_observer_for_session"]:
        monkeypatch.setattr(main, name, no_op)

    def window_analysis(**kwargs):
        """Check windows see current features and a previously saved block CSV."""
        calls.append("windows")
        assert kwargs["augmented_trial_df"] is trial
        assert kwargs["window_sizes"] == sizes
        assert kwargs["feature_parameters"] == {"alpha": 0.1}
        saved = pd.read_csv(tmp_path / "session_block_performance.csv")
        pd.testing.assert_frame_equal(saved, blocks)

    monkeypatch.setattr(main, "save_and_plot_choice_window_agreement_for_session", window_analysis)
    monkeypatch.setattr(main, "missing_block_hmm_output_paths", lambda sess: [])

    def hmm(*args, **kwargs):
        """Require the same unmodified block/trial tables at either HMM branch."""
        calls.append("hmm")
        if args and isinstance(args[0], pd.DataFrame):
            pd.testing.assert_frame_equal(args[0], blocks)
            pd.testing.assert_frame_equal(args[1], trial)
        return blocks, trial

    monkeypatch.setattr(main.bssm, "run_block_modeling", hmm, raising=False)
    monkeypatch.setattr(main, "load_saved_block_hmm_outputs", hmm)
    main.run_single_session_workflow(session, config)
    assert calls == ["blocks", "block_plot", *(["windows"] if sizes else []), "hmm"]


def return_blocks(block_performance, *args, **kwargs):
    """Return the supplied block table unchanged; other orchestration inputs arbitrary."""
    return block_performance


def no_op(*args, **kwargs):
    """Stand in for unrelated plotting; arbitrary inputs, no return value."""
    return None


@pytest.mark.parametrize("sizes", [(10, 20), ()])
def test_multisession_agreement_only_runs_both_paths(main_module, monkeypatch, sizes):
    """Existing agreement runs first; windows run before the HMM-free return."""
    main = main_module
    calls = []
    monkeypatch.setattr(main, "resolve_multisession_dates", lambda **kwargs: ["date"])
    monkeypatch.setattr(main, "find_saved_session_by_date", lambda **kwargs: object())
    monkeypatch.setattr(main, "load_saved_session_analysis", lambda session: object())
    monkeypatch.setattr(main, "save_and_plot_multisession_agent_mouse_agreement",
                        lambda **kwargs: calls.append("blocks"))
    monkeypatch.setattr(main, "save_and_plot_multisession_choice_window_agreement",
                        lambda **kwargs: calls.append("windows"))
    monkeypatch.setattr(main, "concatenate_saved_sessions", fail_if_reached)
    main.main_multisession(agent_agreement_only=True, choice_window_sizes=sizes)
    assert calls == ["blocks", *(["windows"] if sizes else [])]


def fail_if_reached(*args, **kwargs):
    """Fail when agreement-only workflow reaches unrelated analysis."""
    pytest.fail("Unexpected full-analysis execution")


def test_single_session_windows_plot_in_figures_even_for_cached_results(main_module, tmp_path):
    """Figures use the session figure root; cached numerical outputs stay processed."""
    session = SimpleNamespace(
        sess_id_full="session", processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures",
    )
    trial = pd.DataFrame({"action": [0] * 23, "Qlearning_rel_value": [-1.0] * 23})
    _, first = main_module.save_and_plot_choice_window_agreement_for_session(trial, session)
    assert first["results"].is_relative_to(session.processed_data_path)
    assert first["plot"].is_relative_to(session.figure_path)
    assert not list(session.processed_data_path.rglob("*.png"))
    first["plot"].unlink()
    _, cached = main_module.save_and_plot_choice_window_agreement_for_session(trial, session)
    assert cached["results"] == first["results"]
    assert cached["plot"] == first["plot"]
    assert cached["plot"].exists()
    _, rerun = main_module.save_and_plot_choice_window_agreement_for_session(trial, session, rerun=True)
    assert rerun["plot"].is_relative_to(session.figure_path)
    assert rerun["plot"] != first["plot"]
    assert first["plot"].exists()


def test_multisession_collection_resets_windows_and_preserves_blocks(main_module, tmp_path):
    """Each 13-choice session yields one ten-choice window, not a joined window."""
    sessions = []
    for index in range(2):
        processed = tmp_path / str(index)
        processed.mkdir()
        session = SimpleNamespace(sess_id_full=f"session_{index}", date=f"date_{index}",
                                  processed_data_path=processed, figure_path=tmp_path / "figures" / str(index))
        trial = pd.DataFrame({"action": np.zeros(13), "Qlearning_rel_value": [-1.0] * 13})
        sessions.append(SimpleNamespace(session=session, augmented_trial_df=trial,
                                        block_performance=pd.DataFrame({"original": [index]})))
    originals = [saved.block_performance.copy() for saved in sessions]
    table, path = main_module.save_and_plot_multisession_choice_window_agreement(
        sessions, tmp_path, "mouse", window_sizes=(10, 20), n_jobs=1
    )
    assert table.groupby("session_id").size().tolist() == [1, 1]
    assert table.trial_start.tolist() == [0, 0]
    assert table.window_size.tolist() == [10, 10]
    assert path.exists()
    assert (path.parent / "run.log").exists()
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert len(metadata["figures"]) == 2
    for figure in metadata["figures"]:
        assert Path(figure).parent == tmp_path
        assert Path(figure).exists()
    assert len(list(tmp_path.glob("*_choice-window-10_agreement.png"))) == 1
    assert len(list(tmp_path.glob("*_choice-window-20_agreement.png"))) == 1
    summary = pd.read_csv(path.parent / "mouse_choice_window_summary.csv")
    assert {"agreement_q1", "agreement_q3", "n_scored_windows"}.issubset(summary.columns)
    for saved, original in zip(sessions, originals):
        pd.testing.assert_frame_equal(saved.block_performance, original)
        assert list(saved.session.figure_path.rglob("*_choice_window_agreement.png"))
        assert not list(saved.session.processed_data_path.rglob("*.png"))


def test_multisession_parallel_matches_serial_and_dry_run_writes_nothing(main_module, tmp_path):
    """Session workers agree with serial scoring; dry runs create no artifacts."""
    sessions = []
    for index in range(2):
        session = SimpleNamespace(sess_id_full=f"session_{index}", date=f"date_{index}",
                                  processed_data_path=tmp_path / str(index),
                                  figure_path=tmp_path / "figures" / str(index))
        trial = pd.DataFrame({"action": [0] * 13, "Qlearning_rel_value": [-1.0] * 13})
        sessions.append(SimpleNamespace(session=session, augmented_trial_df=trial))
    _, dry_path = main_module.save_and_plot_multisession_choice_window_agreement(
        sessions, tmp_path, "mouse", n_jobs=1, dry_run=True
    )
    assert dry_path is None
    assert list(tmp_path.iterdir()) == []
    serial, _ = main_module.save_and_plot_multisession_choice_window_agreement(
        sessions, tmp_path, "mouse", n_jobs=1
    )
    parallel, _ = main_module.save_and_plot_multisession_choice_window_agreement(
        sessions, tmp_path, "mouse", n_jobs=2, rerun=True
    )
    pd.testing.assert_frame_equal(serial, parallel)


def test_empty_multisession_collection_saves_placeholder_figures(main_module, tmp_path):
    """An empty selection produces readable agreement figures at both resolutions."""
    table, path = main_module.save_and_plot_multisession_choice_window_agreement(
        [], tmp_path, "empty", window_sizes=(10, 20), n_jobs=1,
    )
    assert table.empty
    assert path.exists()
    assert len(list(tmp_path.glob("*_choice-window-*_agreement.png"))) == 2


@pytest.mark.parametrize("collection_only", [True, False])
def test_multisession_collection_only_skips_windows_full_mode_includes_them(
    main_module, monkeypatch, collection_only
):
    """Windows respect the existing collection-only gate before full modeling."""
    main = main_module
    calls = []
    monkeypatch.setattr(main, "resolve_multisession_dates", lambda **kwargs: ["date"])
    monkeypatch.setattr(main, "find_saved_session_by_date", lambda **kwargs: object())
    monkeypatch.setattr(main, "load_saved_session_analysis",
                        lambda session: SimpleNamespace(session=SimpleNamespace()))
    monkeypatch.setattr(main, "concatenate_saved_sessions", lambda sessions: object())
    monkeypatch.setattr(main, "build_multisession_session", lambda **kwargs: object())
    monkeypatch.setattr(main, "run_or_collect_multisession_analysis", lambda **kwargs: calls.append("collection"))
    monkeypatch.setattr(main, "save_and_plot_multisession_choice_window_agreement",
                        lambda **kwargs: calls.append("windows"))

    def stop_before_unrelated_analysis(**kwargs):
        """Stop after the integration boundary; inputs unused, raises sentinel."""
        raise RuntimeError("full workflow reached")

    monkeypatch.setattr(main, "prepare_learning_curve_data", stop_before_unrelated_analysis)
    if collection_only:
        main.main_multisession()
        assert calls == ["collection"]
    else:
        with pytest.raises(RuntimeError, match="full workflow reached"):
            main.main_multisession(multisession_collection_only=False)
        assert calls == ["windows"]
