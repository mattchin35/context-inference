"""User-defined block exemplar models for trials-to-correct residuals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd

MISSING_VALUE = "None"
PARAMETER_COLUMNS = ("name", "intercept", "reward_slope", "residual_sd", "description")
SAFE_MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class BlockExemplarModelSpec:
    """User-defined linear block exemplar model.

    Parameters
    ----------
    name : str
        Stable identifier used as the output-column prefix. Names must start
        with a letter and contain only letters, numbers, and underscores.
    intercept : float
        Predicted trials-to-correct when `prev_n_rewarded == 0`, in trials.
    reward_slope : float
        Change in predicted trials-to-correct per previous rewarded trial.
    residual_sd : float
        Positive residual standard deviation, in trials, used to normalize
        residuals.
    description : str, default=""
        Human-readable notes saved with the parameter inspection outputs.
    """

    name: str
    intercept: float
    reward_slope: float
    residual_sd: float
    description: str = ""


DEFAULT_BLOCK_EXEMPLAR_MODELS: tuple[BlockExemplarModelSpec, ...] = (
    BlockExemplarModelSpec(
        name="fast_low_variance",
        intercept=1.0,
        reward_slope=0.0,
        residual_sd=1.0,
        description="Editable exemplar: fast switching with low residual variance.",
    ),
    BlockExemplarModelSpec(
        name="fast_reward_sensitive",
        intercept=1.0,
        reward_slope=0.25,
        residual_sd=1.5,
        description="Editable exemplar: fast baseline with reward-load sensitivity.",
    ),
    BlockExemplarModelSpec(
        name="slow_low_variance",
        intercept=4.0,
        reward_slope=0.0,
        residual_sd=1.0,
        description="Editable exemplar: slow switching with low residual variance.",
    ),
    BlockExemplarModelSpec(
        name="slow_reward_sensitive",
        intercept=2.0,
        reward_slope=0.5,
        residual_sd=2.0,
        description="Editable exemplar: slower switching with reward-load sensitivity.",
    ),
)


def validate_exemplar_model_specs(specs: Sequence[BlockExemplarModelSpec]) -> None:
    """Validate user-defined exemplar model specifications.

    Parameters
    ----------
    specs : Sequence[BlockExemplarModelSpec]
        Exemplar model specs. Each spec defines one linear model for
        `trials_to_correct ~ prev_n_rewarded + intercept`.

    Returns
    -------
    None
        Returns None when all specs are valid.

    Raises
    ------
    ValueError
        If names are unsafe or duplicated, numeric parameters are not finite,
        or `residual_sd` is not positive.
    """
    seen_names: set[str] = set()
    for spec in specs:
        if not SAFE_MODEL_NAME_PATTERN.fullmatch(spec.name):
            raise ValueError(
                "Invalid exemplar model name. Names must start with a letter "
                "and contain only letters, numbers, and underscores."
            )
        if spec.name in seen_names:
            raise ValueError(f"Duplicate exemplar model name: {spec.name}")
        seen_names.add(spec.name)

        numeric_values = {
            "intercept": spec.intercept,
            "reward_slope": spec.reward_slope,
            "residual_sd": spec.residual_sd,
        }
        for parameter_name, parameter_value in numeric_values.items():
            if not np.isfinite(float(parameter_value)):
                raise ValueError(f"{spec.name} has non-finite {parameter_name}.")
        if float(spec.residual_sd) <= 0:
            raise ValueError(f"{spec.name} residual_sd must be > 0.")


def exemplar_model_parameters_dataframe(
    specs: Sequence[BlockExemplarModelSpec],
) -> pd.DataFrame:
    """Convert exemplar specs to an inspection dataframe.

    Parameters
    ----------
    specs : Sequence[BlockExemplarModelSpec]
        Validated exemplar model specs.

    Returns
    -------
    pandas.DataFrame
        Table with shape `(n_models, 5)` and columns `name`, `intercept`,
        `reward_slope`, `residual_sd`, and `description`.
    """
    validate_exemplar_model_specs(specs)
    return pd.DataFrame([asdict(spec) for spec in specs], columns=PARAMETER_COLUMNS)


def _exemplar_residual_column(model_name: str) -> str:
    """Return the raw residual column name for one exemplar model."""
    return f"{model_name}_exemplar_residual_TTS"


def _exemplar_normalized_residual_column(model_name: str) -> str:
    """Return the normalized residual column name for one exemplar model."""
    return f"{model_name}_exemplar_normalized_residual_TTS"


def add_block_exemplar_model_outputs(
    block_performance: pd.DataFrame,
    specs: Sequence[BlockExemplarModelSpec] = DEFAULT_BLOCK_EXEMPLAR_MODELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add user-defined exemplar residuals to a block table.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required columns
        are `trials_to_correct`, in trials, and `prev_n_rewarded`, in rewarded
        trials from the previous block.
    specs : Sequence[BlockExemplarModelSpec], default=DEFAULT_BLOCK_EXEMPLAR_MODELS
        Exemplar model specs. Each model predicts
        `intercept + reward_slope * prev_n_rewarded`.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Updated block table and exemplar parameter table. For each model, the
        block table contains raw residuals in trials and normalized residuals
        in residual-SD units. Rows with missing required inputs use `"None"`.

    Raises
    ------
    ValueError
        If required input columns are missing or any model spec is invalid.
    """
    missing_columns = [
        column
        for column in ("trials_to_correct", "prev_n_rewarded")
        if column not in block_performance.columns
    ]
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    parameters = exemplar_model_parameters_dataframe(specs)
    output_df = block_performance.copy()
    trials_to_correct = pd.to_numeric(output_df["trials_to_correct"], errors="coerce")
    prev_n_rewarded = pd.to_numeric(output_df["prev_n_rewarded"], errors="coerce")
    valid_rows = trials_to_correct.notna() & prev_n_rewarded.notna()

    for spec in specs:
        residual_values = np.full(output_df.shape[0], MISSING_VALUE, dtype=object)
        normalized_residual_values = np.full(output_df.shape[0], MISSING_VALUE, dtype=object)
        if valid_rows.any():
            prediction = spec.intercept + spec.reward_slope * prev_n_rewarded.loc[valid_rows]
            residual = trials_to_correct.loc[valid_rows] - prediction
            residual_values[valid_rows.to_numpy()] = residual.to_numpy(dtype=float)
            normalized_residual_values[valid_rows.to_numpy()] = (
                residual / spec.residual_sd
            ).to_numpy(dtype=float)

        output_df[_exemplar_residual_column(spec.name)] = residual_values
        output_df[_exemplar_normalized_residual_column(spec.name)] = normalized_residual_values

    return output_df, parameters


def save_block_exemplar_model_parameters(
    parameters: pd.DataFrame,
    session_save_path: Path,
    sess_id: str,
) -> dict[str, Path]:
    """Save exemplar model parameters as CSV and JSON.

    Parameters
    ----------
    parameters : pandas.DataFrame
        Parameter table with shape `(n_models, 5)` and columns defined by
        `PARAMETER_COLUMNS`.
    session_save_path : pathlib.Path
        Directory where within-session outputs are saved.
    sess_id : str
        Full session identifier used in the output file names.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping with `csv` and `json` paths.
    """
    missing_columns = [column for column in PARAMETER_COLUMNS if column not in parameters.columns]
    if missing_columns:
        raise ValueError(f"exemplar parameter table is missing columns: {missing_columns}")

    csv_path = session_save_path / f"{sess_id}_block_exemplar_model_parameters.csv"
    json_path = session_save_path / f"{sess_id}_block_exemplar_model_parameters.json"
    ordered_parameters = parameters.loc[:, PARAMETER_COLUMNS]
    ordered_parameters.to_csv(csv_path, index=False, na_rep=MISSING_VALUE)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump({"models": ordered_parameters.to_dict(orient="records")}, file, indent=2)
        file.write("\n")
    return {"csv": csv_path, "json": json_path}
