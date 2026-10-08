"""Canonical trial-condition masks for task-variable decoding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.neural_analysis.spike_behavior.trials import make_trial_type_masks
from src.neural_analysis.task_decoding.config import CONDITION_IDENTIFIERS


def build_condition_masks(
    trial_df: pd.DataFrame,
    condition_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Build selected canonical masks on the original chronological table.

    Parameters
    ----------
    trial_df : pandas.DataFrame
        Chronological table with shape ``(trial, column)``. Any non-``all``
        condition requires dimensionless ``correct``, ``reward``, and
        ``action`` columns plus the project experimenter-reward flag accepted
        by :func:`make_trial_type_masks`.
    condition_names : sequence[str]
        Nonempty, duplicate-free sequence selected from
        :data:`CONDITION_IDENTIFIERS`. Order is preserved in the returned
        mapping; configuration loading supplies canonical order.

    Returns
    -------
    dict[str, numpy.ndarray]
        Ordered mapping whose values are independent Boolean arrays with shape
        ``(trial,)``. ``all`` is true for every row because target, baseline,
        alignment, and neural-coverage eligibility are intersected later.

    Raises
    ------
    ValueError
        If condition identifiers or required source columns are invalid.
    """
    if isinstance(condition_names, (str, bytes)):
        raise ValueError("condition_names must be a sequence of condition identifiers.")
    selected = tuple(condition_names)
    if not selected:
        raise ValueError("condition_names must not be empty.")
    if len(set(selected)) != len(selected):
        raise ValueError("condition_names must not contain duplicates.")
    unknown = [name for name in selected if name not in CONDITION_IDENTIFIERS]
    if unknown:
        raise ValueError(f"Unknown condition identifiers: {unknown}")

    shared_masks: dict[str, pd.Series] = {}
    if any(name != "all" for name in selected):
        shared_masks = make_trial_type_masks(trial_df)

    output: dict[str, np.ndarray] = {}
    for name in selected:
        if name == "all":
            output[name] = np.ones(len(trial_df), dtype=np.bool_)
            continue
        values = np.asarray(shared_masks[name], dtype=np.bool_)
        if values.shape != (len(trial_df),):
            raise ValueError(f"Shared condition mask {name!r} has an invalid shape.")
        output[name] = values.copy()
    return output
