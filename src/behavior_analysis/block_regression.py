"""Legacy-independent validity and sliding regression for block tables."""

import numpy as np
import pandas as pd

from .project_utils import is_present_value


def make_valid_block_history_mask(block_df: pd.DataFrame) -> pd.Series:
    """Return valid block rows from an `(n_blocks, n_columns)` table.

    `trials_to_correct` and canonical `prev_n_correct` must both be present.
    Returns a boolean Series of shape `(n_blocks,)` aligned to the input index.
    Counts retain their source units (trials/choices); no conversion occurs.
    """
    return is_present_value(block_df["trials_to_correct"]) & is_present_value(block_df["prev_n_correct"])


def compute_sliding_block_regression(
    block_df: pd.DataFrame, valid_mask: np.ndarray | pd.Series | None = None,
    predictor_column: str = "prev_n_rewarded", response_column: str = "trials_to_correct",
    window_size: int = 10, step_size: int = 5,
) -> pd.DataFrame:
    """Compute OLS slope/intercept on successive valid-row windows.

    `block_df` has shape `(n_blocks, n_columns)`; mask has `(n_blocks,)` and
    positions index the compact valid-block axis. Predictor units are rewards
    and response units are trials by default. Returns one row per complete
    window, with slope in response/predictor units and intercept in response
    units. Constant predictors have slope zero.
    """
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive.")
    columns = ["window_start_position", "window_end_position", "window_center_position",
               "window_start_block_ix", "window_end_block_ix", "n_blocks",
               f"{predictor_column}_weight", "window_intercept"]
    if valid_mask is None:
        mask = make_valid_block_history_mask(block_df).to_numpy(dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (len(block_df),):
            raise ValueError("valid_mask must have one entry per block_df row.")
    valid = block_df.loc[mask]
    if len(valid) < window_size:
        return pd.DataFrame(columns=columns)
    x_all = pd.to_numeric(valid[predictor_column], errors="raise").to_numpy(dtype=float)
    y_all = pd.to_numeric(valid[response_column], errors="raise").to_numpy(dtype=float)
    ids = valid["block_ix"].to_numpy() if "block_ix" in valid else valid.index.to_numpy()
    rows = []
    for start in range(0, len(valid) - window_size + 1, step_size):
        stop = start + window_size
        x, y = x_all[start:stop], y_all[start:stop]
        xc, yc = x - x.mean(), y - y.mean()
        slope = 0.0 if np.sum(xc ** 2) == 0 else float(np.sum(xc * yc) / np.sum(xc ** 2))
        rows.append({"window_start_position": start, "window_end_position": stop - 1,
                     "window_center_position": (start + stop - 1) / 2,
                     "window_start_block_ix": ids[start], "window_end_block_ix": ids[stop - 1],
                     "n_blocks": window_size, f"{predictor_column}_weight": slope,
                     "window_intercept": float(y.mean() - slope * x.mean())})
    return pd.DataFrame(rows, columns=columns)
