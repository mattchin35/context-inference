"""Synthetic integration contracts for the normal fixed-exemplar workflow."""

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd


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
