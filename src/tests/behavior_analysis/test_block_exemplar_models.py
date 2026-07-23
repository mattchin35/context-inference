import json
from pathlib import Path

import pandas as pd
import pytest

from src.behavior_analysis import block_exemplar_models


def test_validate_exemplar_model_specs_rejects_duplicate_names():
    """Exemplar specs should have stable unique names for column prefixes."""
    specs = [
        block_exemplar_models.BlockExemplarModelSpec(
            name="same_name",
            intercept=1.0,
            reward_slope=0.2,
            residual_sd=1.0,
        ),
        block_exemplar_models.BlockExemplarModelSpec(
            name="same_name",
            intercept=2.0,
            reward_slope=0.3,
            residual_sd=1.0,
        ),
    ]

    with pytest.raises(ValueError, match="Duplicate exemplar model name"):
        block_exemplar_models.validate_exemplar_model_specs(specs)


@pytest.mark.parametrize("bad_name", ["", "has space", "has-dash"])
def test_validate_exemplar_model_specs_rejects_unsafe_names(bad_name: str):
    """Model names should be safe to use directly as CSV column prefixes."""
    specs = [
        block_exemplar_models.BlockExemplarModelSpec(
            name=bad_name,
            intercept=1.0,
            reward_slope=0.2,
            residual_sd=1.0,
        )
    ]

    with pytest.raises(ValueError, match="exemplar model name"):
        block_exemplar_models.validate_exemplar_model_specs(specs)


def test_validate_exemplar_model_specs_rejects_nonpositive_residual_sd():
    """Residual SD is a denominator and must be strictly positive."""
    specs = [
        block_exemplar_models.BlockExemplarModelSpec(
            name="valid_name",
            intercept=1.0,
            reward_slope=0.2,
            residual_sd=0.0,
        )
    ]

    with pytest.raises(ValueError, match="residual_sd"):
        block_exemplar_models.validate_exemplar_model_specs(specs)


def test_add_block_exemplar_model_outputs_adds_residual_columns_only():
    """Exemplar model outputs should include raw and normalized residuals only."""
    block_df = pd.DataFrame(
        {
            "trials_to_correct": [3.0, 5.0, "None"],
            "prev_n_rewarded": [2.0, 4.0, 2.0],
        }
    )
    specs = [
        block_exemplar_models.BlockExemplarModelSpec(
            name="example",
            intercept=1.0,
            reward_slope=0.5,
            residual_sd=2.0,
            description="Test exemplar.",
        )
    ]

    updated_blocks, parameters = block_exemplar_models.add_block_exemplar_model_outputs(
        block_df,
        specs=specs,
    )

    assert "example_exemplar_residual_TTS" in updated_blocks.columns
    assert "example_exemplar_normalized_residual_TTS" in updated_blocks.columns
    assert "example_exemplar_prediction_TTS" not in updated_blocks.columns
    assert "example_exemplar_absolute_residual_TTS" not in updated_blocks.columns
    assert updated_blocks["example_exemplar_residual_TTS"].tolist() == [1.0, 2.0, "None"]
    assert updated_blocks["example_exemplar_normalized_residual_TTS"].tolist() == [
        0.5,
        1.0,
        "None",
    ]
    assert parameters.to_dict(orient="records") == [
        {
            "name": "example",
            "intercept": 1.0,
            "reward_slope": 0.5,
            "residual_sd": 2.0,
            "description": "Test exemplar.",
        }
    ]


def test_save_block_exemplar_model_parameters_writes_csv_and_json(tmp_path: Path):
    """Exemplar parameters should be saved in inspectable tabular and JSON formats."""
    parameters = pd.DataFrame(
        {
            "name": ["example"],
            "intercept": [1.0],
            "reward_slope": [0.5],
            "residual_sd": [2.0],
            "description": ["Test exemplar."],
        }
    )

    saved_paths = block_exemplar_models.save_block_exemplar_model_parameters(
        parameters,
        session_save_path=tmp_path,
        sess_id="CT024_2026-06-09_143852",
    )

    assert saved_paths["csv"] == tmp_path / "CT024_2026-06-09_143852_block_exemplar_model_parameters.csv"
    assert saved_paths["json"] == tmp_path / "CT024_2026-06-09_143852_block_exemplar_model_parameters.json"
    assert pd.read_csv(saved_paths["csv"]).loc[0, "name"] == "example"
    with saved_paths["json"].open("r", encoding="utf-8") as file:
        json_payload = json.load(file)
    assert json_payload["models"][0]["name"] == "example"
