"""Plots for fixed-exemplar block state inference, independent of model fitting."""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _model_column(frame: pd.DataFrame, prefix: str, name: str) -> np.ndarray:
    """Return a numeric model vector from `(n_blocks, n_columns)` input.

    Source values are z residuals (residual-SD units), log likelihoods
    (natural-log units), or probabilities (unitless). Returns float array
    `(n_blocks,)`, retaining missing values as NaN.
    """
    key = f"{prefix}{name}"
    if key not in frame:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float)


def plot_exemplar_block_diagnostic(
    block_performance: pd.DataFrame, model_names: list[str] | tuple[str, ...],
    plot_path: Path, figure_id: str,
) -> Path:
    """Save five full-block diagnostic panels and return their timestamped path.

    Input is the full `(n_blocks, n_columns)` current/saved block table;
    model columns are in `model_names` order. X values are 0-based full table
    row ordinals, not compact valid positions. Residuals use residual-SD units,
    log likelihoods are natural-log units, and probabilities are unitless.
    Returns a PNG `Path`; missing model columns and invalid rows remain gaps.
    """
    names = list(model_names)
    x = np.arange(len(block_performance))
    panels = [
        ("z residual", "exemplar_z_residual_", "Residual (SD)"),
        ("log likelihood", "exemplar_log_likelihood_", "Log likelihood"),
        ("independent argmax", "", "Model state"),
        ("HMM-smoothed assignment", "", "Model state"),
        ("smoothed state probability", "", "Probability"),
    ]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f_UTC")
    destination = Path(plot_path)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{figure_id}_exemplar-block-diagnostic_{stamp}.png"
    with plt.rc_context({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
                         "xtick.color": "black", "ytick.color": "black"}):
        fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
        if names:
            for model_index, name in enumerate(names):
                color = plt.get_cmap("tab10")(model_index % 10)
                axes[0].plot(x, _model_column(block_performance, panels[0][1], name), marker=".", label=name, color=color)
                axes[1].plot(x, _model_column(block_performance, panels[1][1], name), marker=".", label=name, color=color)
        if "exemplar_ll_argmax_state" in block_performance:
            axes[2].step(x, pd.to_numeric(block_performance["exemplar_ll_argmax_state"], errors="coerce"), where="mid")
        if "exemplar_hmm_smoothed_state" in block_performance:
            axes[3].step(x, pd.to_numeric(block_performance["exemplar_hmm_smoothed_state"], errors="coerce"), where="mid")
        axes[4].set_ylim(0, 1)
        for model_index, name in enumerate(names):
            axes[4].plot(x, _model_column(block_performance, "exemplar_hmm_smoothed_prob_", name), alpha=0.8,
                         label=f"P({name})", color=plt.get_cmap("tab10")(model_index % 10), marker=".")
        # Mark source-session boundaries even when two sessions share a date.
        source = block_performance.get("source_session_id")
        if source is not None:
            ids = source.astype(str).to_numpy()
            starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1] if len(ids) else []
            for index in starts:
                if index:
                    for axis in axes:
                        axis.axvline(index - 0.5, color="0.4", linewidth=0.8, linestyle="--")
                axes[0].text(index, 0.98, str(ids[index]), transform=axes[0].get_xaxis_transform(),
                             rotation=90, va="top", ha="left", fontsize=8, color="black")
        for axis, (title, _, ylabel) in zip(axes, panels, strict=True):
            axis.set_title(title, fontsize=13, color="black")
            axis.set_ylabel(ylabel, fontsize=11)
            axis.tick_params(colors="black", labelsize=10)
            axis.set_xlim(-0.5, max(len(x) - 0.5, 0.5))
            if axis is axes[0]:
                values = [_model_column(block_performance, panels[0][1], name) for name in names]
            elif axis is axes[1]:
                values = [_model_column(block_performance, panels[1][1], name) for name in names]
            elif axis is axes[2]:
                values = [pd.to_numeric(block_performance.get("exemplar_ll_argmax_state", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)]
            elif axis is axes[3]:
                values = [pd.to_numeric(block_performance.get("exemplar_hmm_smoothed_state", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)]
            else:
                values = [_model_column(block_performance, "exemplar_hmm_smoothed_prob_", name) for name in names]
            has_finite_data = any(np.isfinite(value).any() for value in values)
            if not has_finite_data:
                axis.text(0.5, 0.5, "No valid modeled block data", transform=axis.transAxes, ha="center", color="black")
            if has_finite_data and (title.startswith("z ") or title.startswith("log ")):
                axis.legend(loc="best", fontsize=9)
        if names and axes[4].lines:
            axes[4].legend(loc="best", fontsize=9)
        axes[2].set_yticks(range(len(names)), names)
        axes[3].set_yticks(range(len(names)), names)
        for axis in axes[2:4]:
            axis.set_ylim(-0.5, max(len(names) - 0.5, 0.5))
        axes[4].set_yticks(np.linspace(0, 1, 5))
        axes[-1].set_xlabel("Full-block ordinal (row position; source sessions separated by dashed lines)", fontsize=12)
        fig.suptitle(f"{figure_id}: fixed-exemplar block diagnostics", fontsize=15, color="black")
        fig.text(0.01, 0.005, "Panels show full block rows; gaps indicate missing/invalid model outputs.", color="black")
        fig.tight_layout(rect=(0, 0.02, 1, 0.98))
        fig.savefig(path, dpi=150, facecolor="white", transparent=False)
        plt.close(fig)
    return path
