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


def test_residual_spread_metrics_include_raw_scaled_mad_iqr_rmse_and_sd():
    """Residual summaries should include MAD, IQR, RMSE, and sample SD."""
    residuals = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    metrics = block_residual_models.compute_residual_spread_metrics(residuals)

    assert metrics["residual_mad_raw"] == pytest.approx(1.0)
    assert metrics["residual_mad_scaled"] == pytest.approx(1.4826)
    assert metrics["residual_iqr"] == pytest.approx(2.0)
    assert metrics["residual_rmse"] == pytest.approx(np.sqrt(2.0))
    assert metrics["residual_sd"] == pytest.approx(np.std(residuals, ddof=1))


def test_residual_spread_metrics_use_none_for_sd_with_too_few_values():
    """Sample SD should be unavailable when fewer than two residuals exist."""
    metrics = block_residual_models.compute_residual_spread_metrics(np.array([1.0]))

    assert metrics["residual_sd"] == "None"
    assert metrics["residual_mad_raw"] == pytest.approx(0.0)
    assert metrics["residual_iqr"] == pytest.approx(0.0)
    assert metrics["residual_rmse"] == pytest.approx(1.0)


def test_fit_block_residual_models_adds_residual_columns_and_summary_rows():
    """Fits should save interaction/additive residuals and interpretable side slopes."""
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
        "lasso_rewards_x_side_lambda_min_residual_TTS",
        "lasso_rewards_x_side_lambda_1se_residual_TTS",
        "elastic_net_rewards_x_side_lambda_min_residual_TTS",
        "elastic_net_rewards_x_side_lambda_1se_residual_TTS",
        "lasso_rewards_plus_side_lambda_min_residual_TTS",
        "lasso_rewards_plus_side_lambda_1se_residual_TTS",
        "elastic_net_rewards_plus_side_lambda_min_residual_TTS",
        "elastic_net_rewards_plus_side_lambda_1se_residual_TTS",
    }
    assert expected_residual_columns.issubset(updated_blocks.columns)
    for column in expected_residual_columns:
        assert pd.to_numeric(updated_blocks[column], errors="coerce").notna().all()

    assert set(summary_df["model_formula"]) == {"rewards_x_side", "rewards_plus_side"}
    assert set(summary_df["model_type"]) == {"lasso", "elastic_net"}
    assert set(summary_df["lambda_choice"]) == {"lambda.min", "lambda.1se"}
    assert summary_df.shape[0] == 8
    for coefficient_column in [
        "coefficient_intercept",
        "coefficient_prev_n_rewarded",
        "coefficient_block_side_left",
        "coefficient_prev_n_rewarded_block_side_left",
        "right_intercept",
        "left_intercept",
        "right_reward_slope",
        "left_reward_slope",
        "side_intercept_delta_left_minus_right",
        "side_reward_slope_delta_left_minus_right",
    ]:
        assert coefficient_column in summary_df.columns

    additive_rows = summary_df[summary_df["model_formula"] == "rewards_plus_side"]
    assert (additive_rows["coefficient_prev_n_rewarded_block_side_left"] == 0.0).all()
    np.testing.assert_allclose(
        additive_rows["right_reward_slope"].astype(float),
        additive_rows["left_reward_slope"].astype(float),
    )

    interaction_rows = summary_df[summary_df["model_formula"] == "rewards_x_side"]
    np.testing.assert_allclose(
        interaction_rows["left_reward_slope"].astype(float),
        interaction_rows["coefficient_prev_n_rewarded"].astype(float)
        + interaction_rows["coefficient_prev_n_rewarded_block_side_left"].astype(float),
    )

    for metric_prefix in ["lasso_lambda_min", "elastic_net_lambda_1se"]:
        assert f"{metric_prefix}_residual_rmse" in updated_session.columns
        assert pd.to_numeric(updated_session.loc[0, f"{metric_prefix}_residual_rmse"], errors="coerce") >= 0
        assert f"{metric_prefix}_residual_sd" in updated_session.columns
        assert pd.to_numeric(updated_session.loc[0, f"{metric_prefix}_residual_sd"], errors="coerce") >= 0
    for metric_prefix in [
        "lasso_rewards_plus_side_lambda_min",
        "elastic_net_rewards_x_side_lambda_1se",
    ]:
        assert f"{metric_prefix}_residual_rmse" in updated_session.columns
        assert pd.to_numeric(updated_session.loc[0, f"{metric_prefix}_residual_rmse"], errors="coerce") >= 0
        assert f"{metric_prefix}_residual_sd" in updated_session.columns
        assert pd.to_numeric(updated_session.loc[0, f"{metric_prefix}_residual_sd"], errors="coerce") >= 0


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
    assert summary_df.shape[0] == 8
    assert summary_df["n_valid_blocks"].tolist() == [4] * 8
    assert set(summary_df["model_formula"]) == {"rewards_x_side", "rewards_plus_side"}
    assert (summary_df["coefficient_intercept"] == "None").all()
    assert updated_session.loc[0, "lasso_lambda_min_residual_rmse"] == "None"
    assert updated_session.loc[0, "lasso_lambda_min_residual_sd"] == "None"
    assert updated_session.loc[0, "lasso_rewards_plus_side_lambda_min_residual_rmse"] == "None"
    assert updated_session.loc[0, "lasso_rewards_plus_side_lambda_min_residual_sd"] == "None"


def test_save_block_residual_model_summaries_writes_combined_and_formula_csvs(tmp_path: Path):
    """Per-session residual summaries should be saved as combined and formula-specific CSVs."""
    summary_df = pd.DataFrame(
        {
            "session_id": ["CT024_2026-06-09_143852", "CT024_2026-06-09_143852"],
            "model_formula": ["rewards_x_side", "rewards_plus_side"],
            "model_type": ["lasso", "lasso"],
            "lambda_choice": ["lambda.min", "lambda.min"],
        }
    )

    saved_paths = block_residual_models.save_block_residual_model_summaries(
        summary_df,
        session_save_path=tmp_path,
        sess_id="CT024_2026-06-09_143852",
    )

    assert saved_paths["combined"] == tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary.csv"
    assert saved_paths["rewards_x_side"] == (
        tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary_rewards_x_side.csv"
    )
    assert saved_paths["rewards_plus_side"] == (
        tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary_rewards_plus_side.csv"
    )
    for saved_path in saved_paths.values():
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
        tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary_rewards_x_side.csv",
        na_filter=False,
    )
    saved_additive_summary = pd.read_csv(
        tmp_path / "CT024_2026-06-09_143852_block_residual_model_summary_rewards_plus_side.csv",
        na_filter=False,
    )
    saved_overall = pd.read_csv(multisession_path / "CT024_overall_performance.csv", na_filter=False)

    assert "lasso_lambda_min_residual_TTS" in saved_block.columns
    assert "lasso_rewards_plus_side_lambda_min_residual_TTS" in saved_block.columns
    assert "elastic_net_lambda_1se_residual_rmse" in multisession_df.columns
    assert "elastic_net_lambda_1se_residual_sd" in multisession_df.columns
    assert "elastic_net_rewards_plus_side_lambda_1se_residual_rmse" in multisession_df.columns
    assert "elastic_net_rewards_plus_side_lambda_1se_residual_sd" in multisession_df.columns
    assert "elastic_net_lambda_1se_residual_rmse" in saved_overall.columns
    assert "elastic_net_lambda_1se_residual_sd" in saved_overall.columns
    assert "elastic_net_rewards_plus_side_lambda_1se_residual_rmse" in saved_overall.columns
    assert "elastic_net_rewards_plus_side_lambda_1se_residual_sd" in saved_overall.columns
    assert saved_summary.shape[0] == 4
    assert saved_additive_summary.shape[0] == 4
    assert (saved_summary["model_formula"] == "rewards_x_side").all()
    assert (saved_additive_summary["model_formula"] == "rewards_plus_side").all()
    assert "coefficient_intercept" in saved_summary.columns
    assert "left_reward_slope" in saved_summary.columns
    assert "residual_sd" in saved_summary.columns
    assert "residual_sd" in saved_additive_summary.columns
