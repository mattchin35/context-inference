from pathlib import Path
from types import ModuleType
import sys

import numpy as np
import pandas as pd
import pytest

formulaic_stub = ModuleType("formulaic")
formulaic_stub.model_matrix = lambda *_args, **_kwargs: None
sys.modules.setdefault("formulaic", formulaic_stub)
statsmodels_stub = ModuleType("statsmodels")
statsmodels_api_stub = ModuleType("statsmodels.api")
statsmodels_stub.api = statsmodels_api_stub
sys.modules.setdefault("statsmodels", statsmodels_stub)
sys.modules.setdefault("statsmodels.api", statsmodels_api_stub)

from src.behavior_analysis import block_residual_models
from src.behavior_analysis import session_analysis


def make_block_model_df(n_blocks: int = 12) -> pd.DataFrame:
    """Build block rows with a known side/reward interaction signal."""
    block_types = ["right_uncued", "left_uncued"] * (n_blocks // 2)
    prev_rewards = np.arange(n_blocks) % 6
    side_is_left = np.array([block_type.startswith("left") for block_type in block_types])
    trials_to_correct = 2.0 + 0.4 * prev_rewards + 1.5 * side_is_left + 0.2 * prev_rewards * side_is_left
    return pd.DataFrame(
        {
            "block_ix": np.arange(n_blocks),
            "block_type": block_types,
            "prev_n_rewarded": prev_rewards,
            "trials_to_correct": trials_to_correct,
        }
    )


def test_derive_block_side_maps_cued_and_uncued_labels():
    """Block side should ignore cued/uncued suffixes and keep left/right only."""
    block_df = pd.DataFrame(
        {"block_type": ["right_uncued", "left_uncued", "right_cued", "left_cued"]}
    )

    block_side = block_residual_models.derive_block_side(block_df["block_type"])

    assert block_side.tolist() == ["right", "left", "right", "left"]


def test_prepare_block_residual_design_matrix_uses_right_reference():
    """The design matrix should match trials_to_correct ~ prev_n_rewarded * block_side."""
    block_df = pd.DataFrame(
        {
            "block_type": ["right_uncued", "left_uncued"],
            "prev_n_rewarded": [2, 3],
            "trials_to_correct": [1, 4],
        }
    )

    prepared = block_residual_models.prepare_block_residual_model_data(block_df)

    assert prepared.feature_names == [
        "prev_n_rewarded",
        "block_side_left",
        "prev_n_rewarded:block_side_left",
    ]
    np.testing.assert_allclose(prepared.x, np.array([[2.0, 0.0, 0.0], [3.0, 1.0, 3.0]]))
    np.testing.assert_allclose(prepared.y, np.array([1.0, 4.0]))
    assert prepared.valid_index.tolist() == [0, 1]


def test_residual_spread_metrics_include_raw_scaled_mad_iqr_and_rmse():
    """Residual summaries should include raw MAD, R-style scaled MAD, IQR, and RMSE."""
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    metrics = block_residual_models.compute_residual_spread_metrics(residuals)

    assert metrics["residual_mad_raw"] == pytest.approx(1.0)
    assert metrics["residual_mad_scaled"] == pytest.approx(1.4826)
    assert metrics["residual_iqr"] == pytest.approx(2.0)
    assert metrics["residual_rmse"] == pytest.approx(np.sqrt(2.0))


def test_fit_block_residual_models_adds_residual_columns_and_summary_rows():
    """Lasso and elastic-net fits should add one residual column per lambda choice."""
    block_df = make_block_model_df()
    session_performance = pd.DataFrame({"existing_metric": [1]})

    updated_blocks, updated_session, summary_df = block_residual_models.add_block_residual_model_outputs(
        block_df,
        session_performance,
        seed=123,
    )

    expected_residual_columns = {
        "lasso_lambda_min_residual_TTS",
        "lasso_lambda_1se_residual_TTS",
        "elastic_net_lambda_min_residual_TTS",
        "elastic_net_lambda_1se_residual_TTS",
    }
    assert expected_residual_columns.issubset(updated_blocks.columns)
    for column in expected_residual_columns:
        assert pd.to_numeric(updated_blocks[column], errors="coerce").notna().all()

    assert set(summary_df["model_type"]) == {"lasso", "elastic_net"}
    assert set(summary_df["lambda_choice"]) == {"lambda.min", "lambda.1se"}
    assert summary_df.shape[0] == 4
    for coefficient_column in [
        "coefficient_intercept",
        "coefficient_prev_n_rewarded",
        "coefficient_block_side_left",
        "coefficient_prev_n_rewarded_block_side_left",
    ]:
        assert coefficient_column in summary_df.columns

    for metric_prefix in ["lasso_lambda_min", "elastic_net_lambda_1se"]:
        assert f"{metric_prefix}_residual_rmse" in updated_session.columns
        assert pd.to_numeric(updated_session.loc[0, f"{metric_prefix}_residual_rmse"], errors="coerce") >= 0


def test_fit_block_residual_models_preserves_missing_rows_as_none():
    """Rows excluded from fitting should keep a clear missing residual sentinel."""
    block_df = make_block_model_df()
    block_df["trials_to_correct"] = block_df["trials_to_correct"].astype(object)
    block_df.loc[3, "trials_to_correct"] = "None"
    session_performance = pd.DataFrame({"existing_metric": [1]})

    updated_blocks, _updated_session, _summary_df = block_residual_models.add_block_residual_model_outputs(
        block_df,
        session_performance,
        seed=123,
    )

    assert updated_blocks.loc[3, "lasso_lambda_min_residual_TTS"] == "None"
    assert pd.to_numeric(updated_blocks.loc[0, "lasso_lambda_min_residual_TTS"], errors="coerce") == pytest.approx(
        float(updated_blocks.loc[0, "lasso_lambda_min_residual_TTS"])
    )


def test_fit_block_residual_models_skips_small_sessions():
    """Sessions with too few valid blocks should save missing metrics instead of fitting."""
    block_df = make_block_model_df(n_blocks=4)
    session_performance = pd.DataFrame({"existing_metric": [1]})

    updated_blocks, updated_session, summary_df = block_residual_models.add_block_residual_model_outputs(
        block_df,
        session_performance,
        seed=123,
        min_valid_blocks=5,
    )

    assert (updated_blocks["lasso_lambda_min_residual_TTS"] == "None").all()
    assert summary_df.shape[0] == 4
    assert summary_df["n_valid_blocks"].tolist() == [4, 4, 4, 4]
    assert (summary_df["coefficient_intercept"] == "None").all()
    assert updated_session.loc[0, "lasso_lambda_min_residual_rmse"] == "None"


def test_save_block_residual_model_summary_writes_single_summary_csv(tmp_path: Path):
    """The per-session residual-model summary should be saved as one CSV."""
    summary_df = pd.DataFrame(
        {
            "session_id": ["CT024_2026-06-09_143852"],
            "model_type": ["lasso"],
            "lambda_choice": ["lambda.min"],
        }
    )

    saved_path = block_residual_models.save_block_residual_model_summary(
        summary_df,
        session_save_path=tmp_path,
        sess_id="CT024_2026-06-09_143852",
    )

    assert saved_path == tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary.csv"
    assert saved_path.exists()


def test_save_analysis_adds_block_residual_outputs_and_summary_csv(tmp_path: Path):
    """Canonical save path should augment block, session, and residual-summary outputs."""
    session_performance = pd.DataFrame({"median_TTS": [2.0]})
    block_performance = make_block_model_df()
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [0],
            "action": [1],
            "reward": [1],
            "experimenter_reward_given": [0],
        }
    )
    multisession_path = tmp_path / "cross_session_analysis"

    multisession_df = session_analysis.save_analysis(
        session_performance=session_performance,
        block_performance=block_performance,
        augmented_trial_df=augmented_trial_df,
        sess_id="CT024_2026-06-09_143852",
        session_save_path=tmp_path,
        multisession_save_path=multisession_path,
    )

    saved_block = pd.read_csv(tmp_path / "CT024_2026-06-09_143852_block_performance.csv", na_filter=False)
    saved_summary = pd.read_csv(
        tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary.csv",
        na_filter=False,
    )
    saved_overall = pd.read_csv(multisession_path / "CT024_overall_performance.csv", na_filter=False)

    assert "lasso_lambda_min_residual_TTS" in saved_block.columns
    assert "elastic_net_lambda_1se_residual_rmse" in multisession_df.columns
    assert "elastic_net_lambda_1se_residual_rmse" in saved_overall.columns
    assert saved_summary.shape[0] == 4
    assert "coefficient_intercept" in saved_summary.columns
