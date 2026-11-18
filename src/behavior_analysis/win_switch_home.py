"""Prepare and store the Win-switch/Home E+D presentation diagnostic.

Saved Home and Ideal traces are consumed as supplied, without model fitting.
Trial axes remain chronological recorded rows; all scores are unitless and
left-positive. Storage contains only the columns needed for this diagnostic.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .choice_window_analysis import prepare_choice_window_agent_values


ANALYSIS_VERSION = "1.0.0"
HOME_COLUMNS = (
    "home_referenced_expectancy_doubt_value",
    "home_referenced_expectancy_doubt_expectancy_choice",
    "home_referenced_expectancy_doubt_doubt_choice",
    "home_referenced_expectancy_doubt_reward_count",
    "home_referenced_expectancy_doubt_expectancy_strength",
)
REQUIRED_COLUMNS = ("action", "reward", "cur_block", "observer_value", *HOME_COLUMNS)


def prepare_win_switch_home_data(trial_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return copied trials, block spans and preparation provenance.

    Input is a chronological (N, columns) DataFrame: actions 0=right/1=left,
    rewards in task reward units, contiguous context blocks, saved Home/Ideal
    signed scores. Optional WSLS and manual-reward flags follow the shared
    choice-window contract. No model parameters are inferred. Outputs retain
    the original trial index/order; spans have B rows in positional trial units
    and agreement fractions [0,1]. No input or filesystem mutation occurs.
    """
    from .performance_plots import prepare_win_switch_home_block_agreement_spans

    missing = sorted(set(REQUIRED_COLUMNS).difference(trial_df.columns))
    if missing:
        raise ValueError(f"Missing Win-switch/Home diagnostic columns: {missing}")
    prepared, provenance = prepare_choice_window_agent_values(trial_df)
    spans = prepare_win_switch_home_block_agreement_spans(prepared)
    return prepared, spans, provenance


def save_win_switch_home_data(
    path: Path, trials: pd.DataFrame, spans: pd.DataFrame, meta: dict,
) -> None:
    """Save plotting-only tables (N trial rows, B block rows) and JSON metadata.

    Path must not already exist. Scores/fractions retain their unitless units,
    and indices retain their labels. Named NPZ arrays include ``meta``. Object
    arrays support legacy missing sentinels; load only trusted project files.
    Returns None, writing one NPZ. Directory creation belongs to the caller.
    """
    optional = [c for c in ("state", "experimenter_reward_given", "manual_reward_given") if c in trials]
    columns = list(REQUIRED_COLUMNS) + ["win_switch_lose_stay_regressor"] + optional
    with Path(path).open("xb") as output:
        np.savez_compressed(
            output, trial_values=trials[columns].to_numpy(), trial_columns=np.array(columns),
            trial_index=trials.index.to_numpy(), block_values=spans.to_numpy(),
            block_columns=np.array(spans.columns), meta=np.array(json.dumps(meta)),
        )


def load_win_switch_home_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load a trusted project NPZ Path; return trial table, spans and metadata.

    Shapes are (N, saved columns) and (B, span columns). Original row labels,
    positional span coordinates and unitless signed scores/fractions are kept.
    No input file is modified. Pickled object arrays require trusted input.
    """
    with np.load(path, allow_pickle=True) as saved:
        trials = pd.DataFrame(saved["trial_values"], columns=saved["trial_columns"], index=saved["trial_index"])
        spans = pd.DataFrame(saved["block_values"], columns=saved["block_columns"])
        meta = json.loads(str(saved["meta"].item()))
    return trials, spans, meta
