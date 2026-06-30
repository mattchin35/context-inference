import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
try:
    from . import session_analysis
except ImportError:
    from behavior_analysis import session_analysis
from typing import Callable, Iterable
import re


cmap = plt.cm.tab20
n_colors = cmap.N  # Number of discrete colors (10 for tab10)
# Access by integer index (0 to N-1)
all_colors = [cmap(i) for i in range(n_colors)]

block_types = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'dark period']
# color_dict = {'right_cued': 'darkred', 'left_cued': 'darkblue', 'right_uncued': 'red', 'left_uncued': 'blue', 'dark period': 'black'}
color_dict = {'right_cued': all_colors[1], 'left_cued': all_colors[3],
              'right_uncued': all_colors[0], 'left_uncued': all_colors[2], 'dark period': 'black'}
state_dict = {0: 'right', 1: 'left'}
DEFAULT_LEARNING_REGRESSOR = "prev_consecutive_rewards"
DEFAULT_TRIALS_TO_CORRECT_SCATTER_REGRESSOR = "prev_n_rewarded"
TRIALS_TO_CORRECT_SCATTER_REGRESSORS = {
    "prev_n_rewarded": "Rewards in Previous Block",
    "prev_consecutive_rewards": "Consecutive Rewards",
    "prev_n_correct": "Correct Choices in Previous Block",
}
DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS = {
    "QL": "qlearning_mouse_agreement",
    "FQL": "fql_mouse_agreement",
    "HMM": "hmm_logodds_mouse_agreement",
    "HMM decay": "hmm_logodds_decay_mouse_agreement",
    "Persev": "perseveration_mouse_agreement",
    "Doubt+P": "doubt_perseveration_mouse_agreement",
    "WSLS": "wsls_mouse_agreement",
    "Ideal": "observer_mouse_agreement",
}
AGENT_MOUSE_AGREEMENT_COLORS = {
    "QL": "#1f77b4",
    "FQL": "#d62728",
    "HMM": "#2ca02c",
    "HMM decay": "#9467bd",
    "Persev": "#ff7f0e",
    "Doubt+P": "#8c564b",
    "WSLS": "#17becf",
    "Ideal": "#111111",
}


def save_performance_figure(fig: plt.Figure, save_path: Path) -> None:
    """Save one performance plot with layout settings that preserve text.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure to save.
    save_path : Path
        PNG output path.

    Returns
    -------
    None
        Saves `fig` as a PNG and closes only that figure.
    """
    fig.tight_layout()
    fig.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
    print("Saved as {}".format(save_path))
    plt.close(fig)


def get_session_block_count_column(multisession_df: pd.DataFrame) -> str:
    """Return the session-level block-count column in a multisession summary.

    Parameters
    ----------
    multisession_df : pd.DataFrame
        Multisession summary table with shape `(n_sessions, n_columns)`.
        New files contain `n_blocks`; legacy files may contain `n_switches`.

    Returns
    -------
    str
        Name of the block-count column to use for plotting.
    """
    if "n_blocks" in multisession_df.columns:
        return "n_blocks"

    if "n_switches" in multisession_df.columns:
        # Legacy CSVs used `n_switches` for session block count; new outputs save `n_blocks`.
        return "n_switches"

    raise ValueError("multisession_df must contain 'n_blocks' or legacy 'n_switches'.")


def plot_block_hmm_state_feature_scatter(
    state_features_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    fit_type: str = "map",
    x_column: str = "bias",
    y_column: str = "prev_n_rewarded_weight",
    size_column: str = "state_block_count",
    marker_mode: str | None = None,
    use_block_count_marker_size: bool | None = True,
    fixed_marker_size: float = 60.0,
    fixed_marker_alpha: float = 0.75,
    min_marker_size: float = 30.0,
    max_marker_size: float = 250.0,
    min_marker_alpha: float = 0.25,
    max_marker_alpha: float = 0.9,
) -> Path:
    """Plot block-HMM state features across sessions for one fit type.

    Parameters
    ----------
    state_features_df : pd.DataFrame
        State-level summary table with shape `(n_states, n_columns)`.
        Required columns are `fit_type`, `x_column`, `y_column`, and
        `size_column`. Each row describes one HMM state from one session and
        fit type.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or dataset identifier used in the plot title and filename.
    fit_type : str, default="map"
        Fit label to plot from the `fit_type` column, usually `"map"` or
        `"mle"`.
    x_column : str, default="bias"
        Numeric state-feature column plotted on the x-axis.
    y_column : str, default="prev_n_rewarded_weight"
        Numeric state-feature column plotted on the y-axis.
    size_column : str, default="state_block_count"
        Numeric column controlling marker area. Units are blocks.
    marker_mode : str or None, default=None
        Marker visual encoding mode. Valid values are `"default"`,
        `"markersize"`, and `"alpha"`. If None, the legacy
        `use_block_count_marker_size` argument selects `"markersize"` or
        `"default"`.
    use_block_count_marker_size : bool or None, default=True
        Legacy marker-size toggle. Used only when `marker_mode` is None.
    fixed_marker_size : float, default=60.0
        Marker area in points squared for `"default"` and `"alpha"` modes.
    fixed_marker_alpha : float, default=0.75
        Marker opacity for `"default"` and `"markersize"` modes.
    min_marker_size : float, default=30.0
        Minimum marker area in points squared for `"markersize"` mode.
    max_marker_size : float, default=250.0
        Maximum marker area in points squared for `"markersize"` mode.
    min_marker_alpha : float, default=0.25
        Minimum marker opacity for `"alpha"` mode.
    max_marker_alpha : float, default=0.9
        Maximum marker opacity for `"alpha"` mode.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    if marker_mode is None:
        marker_mode = "markersize" if use_block_count_marker_size else "default"
    valid_marker_modes = {"default", "markersize", "alpha"}
    if marker_mode not in valid_marker_modes:
        valid_summary = ", ".join(sorted(valid_marker_modes))
        raise ValueError(
            f"marker_mode must be one of: {valid_summary}. "
            f"Received {marker_mode!r}."
        )

    required_columns = {"fit_type", x_column, y_column, size_column}
    missing_columns = sorted(required_columns.difference(state_features_df.columns))
    if missing_columns:
        raise ValueError(
            "state_features_df is missing required columns for block HMM "
            f"state-feature plotting: {missing_columns}"
        )

    plot_path.mkdir(parents=True, exist_ok=True)
    save_path = plot_path / f"{figure_id}_{fit_type}-block-hmm-state-feature-scatter.png"

    plot_df = state_features_df[state_features_df["fit_type"] == fit_type].copy()
    for column in [x_column, y_column, size_column]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_column, y_column, size_column])

    fig, ax = plt.subplots(figsize=(6, 5))
    if plot_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No valid {fit_type.upper()} state-feature rows",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        block_counts = plot_df[size_column].to_numpy(dtype=float)
        count_min = float(np.min(block_counts))
        count_max = float(np.max(block_counts))
        if marker_mode == "markersize" and count_max > count_min:
            marker_sizes = min_marker_size + (
                (block_counts - count_min)
                / (count_max - count_min)
                * (max_marker_size - min_marker_size)
            )
        elif marker_mode == "markersize":
            marker_sizes = np.full(block_counts.shape, (min_marker_size + max_marker_size) / 2)
        else:
            marker_sizes = fixed_marker_size

        marker_colors = np.zeros((plot_df.shape[0], 4), dtype=float)
        marker_colors[:, 3] = fixed_marker_alpha
        if marker_mode == "alpha" and count_max > count_min:
            marker_colors[:, 3] = min_marker_alpha + (
                (block_counts - count_min)
                / (count_max - count_min)
                * (max_marker_alpha - min_marker_alpha)
            )
        elif marker_mode == "alpha":
            marker_colors[:, 3] = (min_marker_alpha + max_marker_alpha) / 2

        scatter = ax.scatter(
            plot_df[x_column].to_numpy(dtype=float),
            plot_df[y_column].to_numpy(dtype=float),
            s=marker_sizes,
            c=marker_colors,
            edgecolors="white",
            linewidths=0.5,
        )
        if marker_mode == "markersize":
            legend_handles, legend_labels = scatter.legend_elements(
                prop="sizes",
                num=3,
                func=lambda size: count_min
                + (
                    (size - min_marker_size)
                    / (max_marker_size - min_marker_size)
                    * (count_max - count_min)
                )
                if count_max > count_min
                else count_min,
            )
            if legend_handles:
                ax.legend(
                    legend_handles,
                    legend_labels,
                    title="Blocks",
                    frameon=False,
                    loc="best",
                )

    ax.axhline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.8", linewidth=0.8, zorder=0)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f"{figure_id} {fit_type.upper()} Block HMM State Features")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    save_performance_figure(fig, save_path)
    return save_path


def plot_switch_persistence_summary(
    summary_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
) -> Path:
    """Plot post-switch persistence probability for one session.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Switch-persistence summary table with shape `(n_rows, n_columns)`.
        Required columns are `switch_group`, `choice_trial_after_switch`,
        `proportion_stay`, and `n_blocks`. `proportion_stay` is unitless and
        `n_blocks` is the number of switched blocks contributing at each
        choice index.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Session identifier used in the output filename and title.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    required_columns = {
        "switch_group",
        "choice_trial_after_switch",
        "proportion_stay",
        "n_blocks",
    }
    missing_columns = sorted(required_columns.difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_columns}")

    plot_path.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.copy()
    plot_df["choice_trial_after_switch"] = pd.to_numeric(
        plot_df["choice_trial_after_switch"],
        errors="raise",
    )
    plot_df["proportion_stay"] = pd.to_numeric(
        plot_df["proportion_stay"],
        errors="raise",
    )
    plot_df["n_blocks"] = pd.to_numeric(plot_df["n_blocks"], errors="raise").astype(int)

    group_styles = {
        "L_to_R": {"color": all_colors[0], "label": "L->R", "zorder": 2},
        "R_to_L": {"color": all_colors[2], "label": "R->L", "zorder": 2},
        "combined": {"color": "black", "label": "combined", "zorder": 3},
    }
    f, ax = plt.subplots(figsize=(7, 5))
    for group_name in ("L_to_R", "R_to_L", "combined"):
        group_df = plot_df[plot_df["switch_group"] == group_name].sort_values(
            "choice_trial_after_switch"
        )
        if group_df.empty:
            continue
        style = group_styles[group_name]
        ax.plot(
            group_df["choice_trial_after_switch"].to_numpy(),
            group_df["proportion_stay"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=style["color"],
            label=style["label"],
            zorder=style["zorder"],
        )
        for _, row in group_df.iterrows():
            ax.annotate(
                f"n={int(row['n_blocks'])}",
                xy=(row["choice_trial_after_switch"], row["proportion_stay"]),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=style["color"],
            )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Valid choice trial after context switch")
    ax.set_ylabel("Proportion stay with previous side")
    ax.set_title(f"{figure_id} Switch Persistence")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fancybox=False)
    save_path = plot_path / f"{figure_id}_switch_persistence.png"
    save_performance_figure(f, save_path)
    return save_path


def plot_post_first_correct_accuracy_summary(
    summary_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
) -> Path:
    """Plot correct-choice probability after the first correct choice in a block.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Post-first-correct summary table with shape `(n_rows, n_columns)`.
        Required columns are `correct_group`,
        `choice_trial_after_first_correct`, `proportion_correct`, and
        `n_blocks`. `proportion_correct` is unitless and `n_blocks` is the
        number of valid block-trials contributing at each choice index.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Session identifier used in the output filename and title.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    required_columns = {
        "correct_group",
        "choice_trial_after_first_correct",
        "proportion_correct",
        "n_blocks",
    }
    missing_columns = sorted(required_columns.difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_columns}")

    plot_path.mkdir(parents=True, exist_ok=True)
    plot_df = summary_df.copy()
    plot_df["choice_trial_after_first_correct"] = pd.to_numeric(
        plot_df["choice_trial_after_first_correct"],
        errors="raise",
    )
    plot_df["proportion_correct"] = pd.to_numeric(
        plot_df["proportion_correct"],
        errors="raise",
    )
    plot_df["n_blocks"] = pd.to_numeric(plot_df["n_blocks"], errors="raise").astype(int)

    group_styles = {
        "left": {"color": color_dict["left_uncued"], "label": "left", "zorder": 2},
        "right": {"color": color_dict["right_uncued"], "label": "right", "zorder": 2},
        "combined": {"color": "black", "label": "combined", "zorder": 3},
    }
    f, ax = plt.subplots(figsize=(7, 5))
    for group_name in ("left", "right", "combined"):
        group_df = plot_df[plot_df["correct_group"] == group_name].sort_values(
            "choice_trial_after_first_correct"
        )
        if group_df.empty:
            continue
        style = group_styles[group_name]
        ax.plot(
            group_df["choice_trial_after_first_correct"].to_numpy(),
            group_df["proportion_correct"].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            color=style["color"],
            label=style["label"],
            zorder=style["zorder"],
        )
        for _, row in group_df.iterrows():
            ax.annotate(
                f"n={int(row['n_blocks'])}",
                xy=(
                    row["choice_trial_after_first_correct"],
                    row["proportion_correct"],
                ),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=style["color"],
            )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Valid choice trial after first correct choice")
    ax.set_ylabel("Proportion correct")
    ax.set_title(f"{figure_id} Post-First-Correct Accuracy")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fancybox=False)
    save_path = plot_path / f"{figure_id}_post_first_correct_accuracy.png"
    save_performance_figure(f, save_path)
    return save_path


def plot_session_correct(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """For a single session, plot the blockwise percentage of correct choices made by the agent."""
    block_types = ['right_cued', 'left_cued', 'right_uncued', 'left_uncued', 'dark period']
    plot_df = block_performance.loc[:, ['block_type', 'percent_correct']].copy()
    plot_df['block_position'] = np.arange(len(block_performance))
    plot_df['percent_correct_numeric'] = pd.to_numeric(
        plot_df['percent_correct'],
        errors='coerce',
    )
    plot_df = plot_df.dropna(subset=['percent_correct_numeric'])

    f, ax = plt.subplots(figsize=(7, 5))
    for b in block_types:
        block_df = plot_df[plot_df['block_type'] == b]
        if not block_df.empty:
            ax.plot(
                block_df['block_position'].to_numpy(),
                block_df['percent_correct_numeric'].to_numpy(dtype=float),
                'o',
                color=color_dict[b],
                label=b,
            )

    plt.ylabel('Percent Correct')
    plt.xlabel('Block')
    plt.title('{} Block Performance'.format(sess_ID))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(fancybox=False)

    save_path = plot_path / '{}_block_performance.png'.format(sess_ID)
    save_performance_figure(f, save_path)


def plot_session_correct_after_first_correct(
    block_performance: pd.DataFrame,
    plot_path: Path,
    sess_ID: str,
) -> None:
    """Plot block accuracy after the first correct behavioral choice.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type` and
        `percent_correct_after_first_correct`. Values are fractions in the
        range 0-1 when a first correct choice exists; missing sentinels such
        as `"None"` mark blocks with no correct choice.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the plot title and output filename.

    Returns
    -------
    None
        Saves `{sess_ID}_block_correct_after_first_correct.png`.
    """
    required_columns = {"block_type", "percent_correct_after_first_correct"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.loc[
        :,
        ["block_type", "percent_correct_after_first_correct"],
    ].copy()
    if "block_ix" in block_performance.columns:
        plot_df["block_position"] = pd.to_numeric(
            block_performance["block_ix"],
            errors="coerce",
        )
        if plot_df["block_position"].isna().any():
            plot_df["block_position"] = np.arange(len(block_performance))
    else:
        plot_df["block_position"] = np.arange(len(block_performance))
    plot_df["percent_correct_numeric"] = pd.to_numeric(
        plot_df["percent_correct_after_first_correct"],
        errors="coerce",
    )

    no_correct_y = 1.08
    f, axes = plt.subplots(2, 1, figsize=(7, 8))
    timeline_ax, side_ax = axes
    for block_type in block_types:
        block_df = plot_df[plot_df["block_type"] == block_type]
        if block_df.empty:
            continue

        numeric_blocks = block_df.dropna(subset=["percent_correct_numeric"])
        if not numeric_blocks.empty:
            timeline_ax.plot(
                numeric_blocks["block_position"].to_numpy(dtype=float),
                numeric_blocks["percent_correct_numeric"].to_numpy(dtype=float),
                "o",
                color=color_dict[block_type],
                label=block_type.replace("_", " "),
            )

        no_correct_blocks = block_df[block_df["percent_correct_numeric"].isna()]
        if not no_correct_blocks.empty:
            timeline_ax.plot(
                no_correct_blocks["block_position"].to_numpy(dtype=float),
                np.full(no_correct_blocks.shape[0], no_correct_y),
                "^",
                color=color_dict[block_type],
                markerfacecolor="none",
                label=f"{block_type.replace('_', ' ')} no correct",
            )

    timeline_ax.axhline(no_correct_y, color="gray", linestyle=":", linewidth=1)
    timeline_ax.set_ylim(-0.05, 1.18)
    timeline_ax.set_ylabel("Percent Correct After First Correct")
    timeline_ax.set_xlabel("Block")
    timeline_ax.set_title(f"{sess_ID} Block Performance After First Correct")
    handles, _labels = timeline_ax.get_legend_handles_labels()
    if handles:
        timeline_ax.legend(fancybox=False, fontsize=8)

    _plot_side_metric_summary_on_ax(
        side_ax=side_ax,
        plot_df=plot_df,
        metric_column="percent_correct_numeric",
        y_label="Percent Correct After First Correct",
        missing_marker_y=no_correct_y,
        missing_label="no correct",
    )
    side_ax.set_ylim(-0.05, 1.18)

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_ID}_block_correct_after_first_correct.png"
    save_performance_figure(f, save_path)


def plot_session_history_ideal_mouse_agreement(
    block_performance: pd.DataFrame,
    plot_path: Path,
    sess_ID: str,
) -> None:
    """Plot blockwise agreement with the greedy mouse-history ideal observer.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type` and
        `block_history_ideal_mouse_agreement`. Agreement values are fractions
        in the range 0-1; missing sentinels such as `"None"` mark blocks with
        no valid ideal-observer comparisons.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the plot title and output filename.

    Returns
    -------
    None
        Saves `{sess_ID}_history_ideal_mouse_agreement.png`.
    """
    required_columns = {"block_type", "block_history_ideal_mouse_agreement"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.loc[
        :,
        ["block_type", "block_history_ideal_mouse_agreement"],
    ].copy()
    if "block_ix" in block_performance.columns:
        plot_df["block_position"] = pd.to_numeric(
            block_performance["block_ix"],
            errors="coerce",
        )
        if plot_df["block_position"].isna().any():
            plot_df["block_position"] = np.arange(len(block_performance))
    else:
        plot_df["block_position"] = np.arange(len(block_performance))
    plot_df["agreement_numeric"] = pd.to_numeric(
        plot_df["block_history_ideal_mouse_agreement"],
        errors="coerce",
    )

    missing_y = 1.08
    f, axes = plt.subplots(2, 1, figsize=(7, 8))
    timeline_ax, side_ax = axes
    for block_type in block_types:
        block_df = plot_df[plot_df["block_type"] == block_type]
        if block_df.empty:
            continue

        numeric_blocks = block_df.dropna(subset=["agreement_numeric"])
        if not numeric_blocks.empty:
            timeline_ax.plot(
                numeric_blocks["block_position"].to_numpy(dtype=float),
                numeric_blocks["agreement_numeric"].to_numpy(dtype=float),
                "o",
                color=color_dict[block_type],
                label=block_type.replace("_", " "),
            )

        missing_blocks = block_df[block_df["agreement_numeric"].isna()]
        if not missing_blocks.empty:
            timeline_ax.plot(
                missing_blocks["block_position"].to_numpy(dtype=float),
                np.full(missing_blocks.shape[0], missing_y),
                "^",
                color=color_dict[block_type],
                markerfacecolor="none",
                label=f"{block_type.replace('_', ' ')} no valid ideal",
            )

    timeline_ax.axhline(missing_y, color="gray", linestyle=":", linewidth=1)
    timeline_ax.set_ylim(-0.05, 1.18)
    timeline_ax.set_ylabel("Mouse / History Ideal Agreement")
    timeline_ax.set_xlabel("Block")
    timeline_ax.set_title(f"{sess_ID} History Ideal Agreement")
    handles, _labels = timeline_ax.get_legend_handles_labels()
    if handles:
        timeline_ax.legend(fancybox=False, fontsize=8)

    _plot_side_metric_summary_on_ax(
        side_ax=side_ax,
        plot_df=plot_df,
        metric_column="agreement_numeric",
        y_label="Mouse / History Ideal Agreement",
        missing_marker_y=missing_y,
        missing_label="no valid ideal",
    )
    side_ax.set_ylim(-0.05, 1.18)

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_ID}_history_ideal_mouse_agreement.png"
    save_performance_figure(f, save_path)


def plot_session_agent_mouse_agreement(
    block_performance: pd.DataFrame,
    plot_path: Path,
    sess_id_full: str,
    agreement_columns: dict[str, str] | None = None,
) -> Path:
    """Plot blockwise mouse-agent agreement for multiple agents.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required columns
        are the requested agreement columns. If `block_ix` is present, it is
        used as the x-axis; otherwise row position is used.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_id_full : str
        Full session identifier used in the plot title and output filename.
    agreement_columns : dict[str, str] or None, default=None
        Mapping from legend label to blockwise agreement column. None uses the
        default Q-learning, forgetting Q-learning, HMM log-odds, HMM log-odds
        with decay, and ideal-observer agreement columns.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{sess_id_full}_agent_mouse_agreement.png`.

    Raises
    ------
    ValueError
        If any requested agreement column is absent.
    """
    if agreement_columns is None:
        agreement_columns = DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS

    missing_columns = [
        column for column in agreement_columns.values()
        if column not in block_performance.columns
    ]
    if missing_columns:
        raise ValueError(f"block_performance is missing requested agreement columns: {missing_columns}")

    block_positions = _get_block_positions_for_plot(block_performance)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    timeline_ax, summary_ax = axes
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", all_colors)
    for agent_index, (label, column) in enumerate(agreement_columns.items()):
        agreement_values = pd.to_numeric(block_performance[column], errors="coerce")
        valid_rows = agreement_values.notna() & block_positions.notna()
        if not valid_rows.any():
            continue
        color = AGENT_MOUSE_AGREEMENT_COLORS.get(label, color_cycle[agent_index % len(color_cycle)])
        timeline_ax.plot(
            block_positions.loc[valid_rows].to_numpy(dtype=float),
            agreement_values.loc[valid_rows].to_numpy(dtype=float),
            marker="o",
            linewidth=2,
            markersize=5,
            color=color,
            label=label,
        )

    timeline_ax.set_ylim(-0.05, 1.05)
    timeline_ax.set_xlabel("Block")
    timeline_ax.set_ylabel("Agent agreement")
    timeline_ax.set_title(f"{sess_id_full} Mouse-Agent Agreement")
    timeline_ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    handles, _labels = timeline_ax.get_legend_handles_labels()
    if handles:
        timeline_ax.legend(fancybox=False, fontsize=8)

    _plot_agent_metric_summary_on_ax(
        summary_ax=summary_ax,
        block_performance=block_performance,
        agreement_columns=agreement_columns,
        y_label="Agent agreement",
        color_cycle=color_cycle,
    )

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_id_full}_agent_mouse_agreement.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_session_side_bias_ratios(
    session_summary: pd.DataFrame,
    plot_path: Path,
    sess_id_full: str,
) -> Path:
    """Plot whole-session side-bias ratios against true and ideal references.

    Parameters
    ----------
    session_summary : pd.DataFrame
        One-row session summary dataframe with shape `(1, n_columns)`.
        Required ratio columns are `left_choice_per_true_left`,
        `right_choice_per_true_right`, `left_choice_per_ideal_left`, and
        `right_choice_per_ideal_right`. Ratio values are unitless and are not
        bounded by 1; string `"None"` values are skipped.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_id_full : str
        Full session identifier used in the plot title and output filename.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{sess_id_full}_session_side_bias_ratios.png`.
    """
    required_columns = {
        "left_choice_per_true_left",
        "right_choice_per_true_right",
        "left_choice_per_ideal_left",
        "right_choice_per_ideal_right",
    }
    missing_columns = sorted(required_columns.difference(session_summary.columns))
    if missing_columns:
        raise ValueError(f"session_summary is missing required columns: {missing_columns}")
    if session_summary.empty:
        raise ValueError("session_summary must contain one session row.")

    row = session_summary.iloc[0]
    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    x_positions = {"true": 1.0, "ideal": 2.0}
    side_specs = {
        "left": {
            "offset": -0.15,
            "color": color_dict["left_uncued"],
            "columns": {
                "true": "left_choice_per_true_left",
                "ideal": "left_choice_per_ideal_left",
            },
        },
        "right": {
            "offset": 0.15,
            "color": color_dict["right_uncued"],
            "columns": {
                "true": "right_choice_per_true_right",
                "ideal": "right_choice_per_ideal_right",
            },
        },
    }

    for side, spec in side_specs.items():
        x_values = []
        y_values = []
        for reference, column in spec["columns"].items():
            numeric_value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
            if pd.isna(numeric_value):
                continue
            x_values.append(x_positions[reference] + spec["offset"])
            y_values.append(float(numeric_value))
        if not y_values:
            continue
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linestyle="none",
            markersize=8,
            color=spec["color"],
            label=side,
        )

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks([x_positions["true"], x_positions["ideal"]])
    ax.set_xticklabels(["True context", "Ideal agent"])
    ax.set_ylabel("Choice Count / Reference Count")
    ax.set_title(f"{sess_id_full} Session Side-Bias Ratios")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fancybox=False, frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_id_full}_session_side_bias_ratios.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_session_signed_side_bias(
    session_summary: pd.DataFrame,
    plot_path: Path,
    sess_id_full: str,
) -> Path:
    """Plot signed whole-session side-bias metrics.

    Parameters
    ----------
    session_summary : pd.DataFrame
        One-row session summary dataframe with shape `(1, n_columns)`.
        Required columns are `bias_oracle`, `bias_ideal`, and
        `raw_side_bias`. Bias values are unitless signed fractions where
        positive values indicate excess left choices.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_id_full : str
        Full session identifier used in the plot title and output filename.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{sess_id_full}_session_signed_side_bias.png`.
    """
    required_columns = ["bias_oracle", "bias_ideal", "raw_side_bias"]
    missing_columns = sorted(set(required_columns).difference(session_summary.columns))
    if missing_columns:
        raise ValueError(f"session_summary is missing required columns: {missing_columns}")
    if session_summary.empty:
        raise ValueError("session_summary must contain one session row.")

    row = session_summary.iloc[0]
    values = pd.to_numeric(row.loc[required_columns], errors="coerce")
    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(required_columns))
    ax.plot(
        x,
        values.to_numpy(dtype=float),
        "o-",
        color="black",
        linewidth=2,
        markersize=7,
        label="signed bias",
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["Oracle", "Ideal", "Raw"], rotation=20)
    ax.set_ylabel("Left bias")
    ax.set_title(f"{sess_id_full} Signed Side Bias")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)

    save_path = plot_path / f"{sess_id_full}_session_signed_side_bias.png"
    save_performance_figure(fig, save_path)
    return save_path


def _prepare_zero_trials_to_correct_plot_df(block_performance: pd.DataFrame) -> pd.DataFrame:
    """Return valid block-change rows with binary zero-TTC flags for plotting.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        columns are `block_type` and `trials_to_correct`. The first row is
        excluded because it is not a block change. Dark-period rows and rows
        with missing/non-numeric `trials_to_correct` are excluded.

    Returns
    -------
    pd.DataFrame
        Plotting dataframe with shape `(n_valid_block_changes, n_columns)`,
        including `rewarded_side` and `zero_trials_to_correct_flag`.
    """
    required_columns = {"block_type", "trials_to_correct"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.copy().reset_index(drop=True)
    plot_df = plot_df.iloc[1:].copy()
    plot_df["rewarded_side"] = plot_df["block_type"].map(_get_rewarded_side_for_plot)
    plot_df["trials_to_correct_numeric"] = pd.to_numeric(
        plot_df["trials_to_correct"],
        errors="coerce",
    )
    plot_df = plot_df[
        plot_df["rewarded_side"].isin(["left", "right"])
        & plot_df["trials_to_correct_numeric"].notna()
    ].copy()
    plot_df["zero_trials_to_correct_flag"] = (
        plot_df["trials_to_correct_numeric"].eq(0).astype(float)
    )
    return plot_df


def plot_session_zero_trials_to_correct_fraction(
    block_performance: pd.DataFrame,
    plot_path: Path,
    sess_id_full: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> Path:
    """Plot block-change zero-trials-to-correct flags and mean fractions.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise dataframe with shape `(n_blocks, n_columns)`. Required
        columns are `block_type` and `trials_to_correct`. The first block,
        dark periods, and never-correct/non-numeric `trials_to_correct` rows
        are excluded before plotting.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_id_full : str
        Full session identifier used in the plot title and output filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw 0/1 block markers.
    jitter_seed : int, default=0
        Seed for deterministic raw-marker jitter.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{sess_id_full}_zero_trials_to_correct_fraction.png`.
    """
    plot_df = _prepare_zero_trials_to_correct_plot_df(block_performance)
    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    group_styles = {
        "left": {"x": 0.0, "color": color_dict["left_uncued"], "seed_offset": 0},
        "right": {"x": 1.0, "color": color_dict["right_uncued"], "seed_offset": 1},
        "overall": {"x": 2.0, "color": "black", "seed_offset": 2},
    }
    group_data = {
        "left": plot_df[plot_df["rewarded_side"] == "left"],
        "right": plot_df[plot_df["rewarded_side"] == "right"],
        "overall": plot_df,
    }

    for group_name, group_df in group_data.items():
        if group_df.empty:
            continue
        style = group_styles[group_name]
        raw_values = group_df["zero_trials_to_correct_flag"].to_numpy(dtype=float)
        ax.plot(
            _jitter_x_coordinates(
                np.full(raw_values.shape[0], style["x"]),
                jitter_width=point_jitter,
                seed=jitter_seed + style["seed_offset"],
            ),
            raw_values,
            "o",
            color=style["color"],
            alpha=0.35,
            markersize=4,
            label=f"{group_name} raw blocks",
        )
        ax.plot(
            [style["x"]],
            [float(np.mean(raw_values))],
            "D",
            color=style["color"],
            markersize=8,
            label=f"{group_name} mean",
        )

    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["left", "right", "overall"])
    ax.set_xlabel("Block group")
    ax.set_ylabel("0 Trials-to-Correct Flag")
    ax.set_title(f"{sess_id_full} 0-Trials-to-Correct Block Changes")
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_id_full}_zero_trials_to_correct_fraction.png"
    save_performance_figure(fig, save_path)
    return save_path


def _get_block_positions_for_plot(block_performance: pd.DataFrame) -> pd.Series:
    """Return numeric block positions for plotting a blockwise metric.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Optional `block_ix`
        values are block identifiers; otherwise row index position is used.

    Returns
    -------
    pd.Series
        Numeric x-axis positions with shape `(n_blocks,)`.
    """
    if "block_ix" in block_performance.columns:
        block_positions = pd.to_numeric(block_performance["block_ix"], errors="coerce")
        if not block_positions.isna().any():
            return block_positions
    return pd.Series(np.arange(block_performance.shape[0]), index=block_performance.index)


def _plot_agent_metric_summary_on_ax(
    summary_ax: plt.Axes,
    block_performance: pd.DataFrame,
    agreement_columns: dict[str, str],
    y_label: str,
    color_cycle: list | None = None,
    rng_seed: int = 0,
) -> None:
    """Plot raw block values plus median and Q1-Q3 summaries by agent.

    Parameters
    ----------
    summary_ax : matplotlib.axes.Axes
        Axis modified in place.
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Requested
        agreement columns contain unitless fractions in [0, 1] or missing
        sentinels such as `"None"`.
    agreement_columns : dict[str, str]
        Mapping from short visible agent label to blockwise agreement column.
    y_label : str
        Y-axis label.
    color_cycle : list or None, default=None
        Plot colors. None uses Matplotlib's current color cycle.
    rng_seed : int, default=0
        Seed for deterministic horizontal jitter of raw block points.

    Returns
    -------
    None
        Mutates `summary_ax` in place.
    """
    if color_cycle is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", all_colors)
    rng = np.random.default_rng(rng_seed)
    x_positions = np.arange(len(agreement_columns), dtype=float)
    for agent_index, (label, column) in enumerate(agreement_columns.items()):
        values = pd.to_numeric(block_performance[column], errors="coerce").dropna()
        if values.empty:
            continue
        color = AGENT_MOUSE_AGREEMENT_COLORS.get(label, color_cycle[agent_index % len(color_cycle)])
        y_values = values.to_numpy(dtype=float)
        jittered_x = x_positions[agent_index] + rng.uniform(-0.08, 0.08, size=y_values.shape[0])
        summary_ax.scatter(
            jittered_x,
            y_values,
            s=22,
            alpha=0.65,
            color=color,
            label="_nolegend_",
        )
        quartiles = values.quantile([0.25, 0.5, 0.75])
        q1 = float(quartiles.loc[0.25])
        median = float(quartiles.loc[0.5])
        q3 = float(quartiles.loc[0.75])
        summary_ax.plot(
            [x_positions[agent_index], x_positions[agent_index]],
            [q1, q3],
            color=color,
            linewidth=2,
        )
        for cap_y in (q1, q3):
            summary_ax.plot(
                [x_positions[agent_index] - 0.08, x_positions[agent_index] + 0.08],
                [cap_y, cap_y],
                color=color,
                linewidth=2,
            )
        summary_ax.plot(
            [x_positions[agent_index] - 0.12, x_positions[agent_index] + 0.12],
            [median, median],
            color=color,
            linewidth=4,
            label=f"{label} median",
        )

    summary_ax.set_ylim(-0.05, 1.05)
    summary_ax.set_xlim(-0.5, len(agreement_columns) - 0.5)
    summary_ax.set_xticks(x_positions)
    summary_ax.set_xticklabels(list(agreement_columns.keys()), rotation=35, ha="right")
    summary_ax.set_xlabel("Agent")
    summary_ax.set_ylabel(y_label)
    summary_ax.set_title("Agent Summary")
    summary_ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)


def plot_multisession_correct(overall_df: pd.DataFrame, plot_path: Path, figure_id: str, use_dates: bool=True):
    block_types = ['right_cued_correct', 'left_cued_correct', 'right_uncued_correct', 'left_uncued_correct']
    color_dict = {'right_cued_correct': 'darkred', 'left_cued_correct': 'darkblue', 'right_uncued_correct': 'red', 'left_uncued_correct': 'blue'}
    overall_df = overall_df[~overall_df['date'].isna()]
    dates = overall_df['date'].unique()

    f, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(overall_df.shape[0])
    for b in block_types:
        type_correct = overall_df[b].values
        if np.sum(np.isnan(type_correct)) == type_correct.size:  # will have to remove this later on...
            continue

        type_correct[np.isnan(type_correct)] = 0
        ax.plot(x, type_correct, color=color_dict[b], label=b)
        if use_dates:
            ax.set_xticks(x, dates)
            ax.tick_params(axis='x', labelrotation=60, labelsize=8)

    plt.ylabel('Percent Correct')
    plt.xlabel('Session')
    plt.title('{} Overall Performance'.format(figure_id))
    plt.legend(fancybox=False)

    save_path = plot_path / '{}_overall_performance.png'.format(figure_id)
    save_performance_figure(f, save_path)


def plot_trials_to_correct_summary(block_performance: pd.DataFrame, plot_path: Path, figure_id: str):
    """plot the trials to correct across a set of collected blocks, which can be from multiple sessions
    (choose carefully!). To compare early vs late, you'll need to copy the code in F31_Apr2024."""

    f, ax = plt.subplots(figsize=(7, 5))
    rewards, mean, std, sem = session_analysis.summarize_block_switches(block_performance)
    plt.plot(mean.index, mean)
    plt.fill_between(mean.index, mean - sem,
                     mean + sem, alpha=.3, linewidth=0)

    plt.xlabel('Consecutive rewards')
    plt.ylabel('Trials to Correct')
    plt.title('Trials to Switch vs Rewards')
    save_path = plot_path / '{}_trials-to-correct-summary.png'.format(figure_id)
    save_performance_figure(f, save_path)


def _jitter_integer_scatter_points(
    x_values: np.ndarray,
    y_values: np.ndarray,
    jitter_width: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return display-only jittered coordinates for integer scatter data.

    Parameters
    ----------
    x_values : np.ndarray
        Original x-axis values, shape `(n_points,)`, in plot-axis units.
        Values are expected to be integer counts.
    y_values : np.ndarray
        Original y-axis values, shape `(n_points,)`, in plot-axis units.
        Values are expected to be integer trial counts.
    jitter_width : float
        Maximum absolute jitter applied independently to each axis, in
        plot-axis units. Use `0.0` to return exact coordinates.
    seed : int
        Seed for deterministic jitter generation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Jittered x and y display coordinates, each with shape `(n_points,)`,
        in the same plot-axis units as the inputs. The input arrays are not
        modified.
    """
    if jitter_width < 0:
        raise ValueError("jitter_width must be non-negative.")

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    if jitter_width == 0 or x_values.size == 0:
        return x_values.copy(), y_values.copy()

    rng = np.random.default_rng(seed)
    x_jitter = rng.uniform(-jitter_width, jitter_width, size=x_values.shape)
    y_jitter = rng.uniform(-jitter_width, jitter_width, size=y_values.shape)
    return x_values + x_jitter, y_values + y_jitter


def _jitter_x_coordinates(x_values: np.ndarray, jitter_width: float, seed: int) -> np.ndarray:
    """Return display-only horizontal jitter for categorical scatter points.

    Parameters
    ----------
    x_values : np.ndarray
        Original x-axis coordinates, shape `(n_points,)`, in plot-axis units.
    jitter_width : float
        Maximum absolute horizontal jitter in plot-axis units. Use `0.0` to
        return exact coordinates.
    seed : int
        Seed for deterministic jitter generation.

    Returns
    -------
    np.ndarray
        Jittered x coordinates with shape `(n_points,)`, in plot-axis units.
        The input array is not modified.
    """
    if jitter_width < 0:
        raise ValueError("jitter_width must be non-negative.")

    x_values = np.asarray(x_values, dtype=float)
    if jitter_width == 0 or x_values.size == 0:
        return x_values.copy()

    rng = np.random.default_rng(seed)
    return x_values + rng.uniform(-jitter_width, jitter_width, size=x_values.shape)


def _normalize_binary_flag_for_plot(flag_values: pd.Series, column_name: str) -> np.ndarray:
    """Return numeric plotting coordinates for saved binary flag values.

    Parameters
    ----------
    flag_values : pd.Series
        Binary flag values with shape `(n_blocks,)`. Values may be booleans,
        numeric 0/1, or CSV-loaded strings such as `"True"`, `"False"`, and
        `"None"`.
    column_name : str
        Source column name used in validation error messages.

    Returns
    -------
    np.ndarray
        Float array with shape `(n_blocks,)`. Valid false values are `0.0`,
        valid true values are `1.0`, and missing sentinels are `NaN`.
    """
    text_values = flag_values.astype(str).str.strip().str.lower()
    normalized = np.full(flag_values.shape[0], np.nan, dtype=float)

    false_mask = text_values.isin(["false", "0", "0.0"])
    true_mask = text_values.isin(["true", "1", "1.0"])
    missing_mask = flag_values.isna() | text_values.isin(["none", "nan", ""])
    invalid_mask = ~(false_mask | true_mask | missing_mask)

    if invalid_mask.any():
        invalid_values = sorted(flag_values.loc[invalid_mask].astype(str).unique())
        raise ValueError(f"{column_name} must contain binary flag values, got: {invalid_values}")

    normalized[false_mask.to_numpy()] = 0.0
    normalized[true_mask.to_numpy()] = 1.0
    return normalized


def plot_block_bias_quadrants(block_performance: pd.DataFrame, plot_path: Path, figure_id: str,
                              title: str | None = None, point_jitter: float = 0.08,
                              jitter_seed: int = 0) -> None:
    """Plot block-level RL and inference bias flags as a jittered quadrant plot.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type`, `bias_inf_flag`, and
        `bias_rl_flag`. Bias flag columns may contain booleans, numeric 0/1,
        or CSV-loaded strings such as `"True"`, `"False"`, and `"None"`.
    plot_path : Path
        Directory where the PNG figure is saved.
    figure_id : str
        Figure identifier used in the output filename.
    title : str or None, default=None
        Optional display title prefix. If None, `figure_id` is used.
    point_jitter : float, default=0.08
        Maximum absolute display-only jitter for scatter markers on both axes,
        in categorical plot-axis units. Original binary flags are not changed.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Saves `{figure_id}_block-bias-quadrants.png` and closes the figure.
    """
    required_columns = {"block_type", "bias_inf_flag", "bias_rl_flag"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    inference_bias = _normalize_binary_flag_for_plot(
        block_performance["bias_inf_flag"],
        "bias_inf_flag",
    )
    rl_bias = _normalize_binary_flag_for_plot(
        block_performance["bias_rl_flag"],
        "bias_rl_flag",
    )
    valid_rows = ~np.isnan(inference_bias) & ~np.isnan(rl_bias)

    plot_df = block_performance.loc[valid_rows, ["block_type"]].copy()
    plot_df["inference_bias"] = inference_bias[valid_rows]
    plot_df["rl_bias"] = rl_bias[valid_rows]
    jittered_inference_bias, jittered_rl_bias = _jitter_integer_scatter_points(
        plot_df["inference_bias"].to_numpy(dtype=float),
        plot_df["rl_bias"].to_numpy(dtype=float),
        jitter_width=point_jitter,
        seed=jitter_seed,
    )
    plot_df["jittered_inference_bias"] = jittered_inference_bias
    plot_df["jittered_rl_bias"] = jittered_rl_bias

    fig, ax = plt.subplots(figsize=(6, 5))
    any_points_plotted = False
    for block_type in block_types:
        block_type_rows = plot_df["block_type"] == block_type
        if block_type_rows.any():
            ax.plot(
                plot_df.loc[block_type_rows, "jittered_inference_bias"],
                plot_df.loc[block_type_rows, "jittered_rl_bias"],
                "o",
                color=color_dict[block_type],
                label=block_type.replace("_", " "),
            )
            any_points_plotted = True

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["inference unbiased", "inference biased"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["RL unbiased", "RL biased"])
    ax.set_xlabel("Inference bias flag")
    ax.set_ylabel("RL bias flag")
    if title is not None:
        ax.set_title(f"{title} Block Bias Quadrants".replace("_", " "))
    else:
        ax.set_title(f"{figure_id} Block Bias Quadrants".replace("_", " "))

    if any_points_plotted:
        ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_path = plot_path / f"{figure_id}_block-bias-quadrants.png"
    save_performance_figure(fig, save_path)


def scatter_trials_to_correct(
    block_performance: pd.DataFrame,
    slope: float,
    intercept: float,
    plot_path: Path,
    figure_id: str,
    title=None,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
    regressor_column: str = DEFAULT_TRIALS_TO_CORRECT_SCATTER_REGRESSOR,
):
    """Plot block trials-to-correct against a selected previous-block regressor.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table, shape `(n_blocks, n_columns)`. Required
        columns are `block_type`, `trials_to_correct`, `prev_n_correct`, and
        `regressor_column`. Count columns are interpreted as integer
        block/trial counts. Rows with missing `trials_to_correct` or
        `prev_n_correct` are excluded to match session regression summaries.
    slope : float
        Regression slope in trials-to-correct per selected regressor unit.
    intercept : float
        Regression intercept in trials-to-correct units.
    plot_path : Path
        Directory where the PNG figure is saved.
    figure_id : str
        Figure identifier used in the output filename.
    title : str or None, default=None
        Optional display title prefix. If None, `figure_id` is used.
    point_jitter : float, default=0.08
        Maximum absolute display-only jitter for scatter markers on both axes,
        in plot-axis units. Original data and the regression line remain
        unjittered.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.
    regressor_column : str, default="prev_n_rewarded"
        Block-performance column to use for both scatter x-values and the
        regression-line x-scale. Supported values are `"prev_n_rewarded"`,
        `"prev_consecutive_rewards"`, and `"prev_n_correct"`.

    Returns
    -------
    None
        Saves `{figure_id}_scatter_trials-to-correct_{regressor_column}.png`
        and closes the figure.
    """
    if regressor_column not in TRIALS_TO_CORRECT_SCATTER_REGRESSORS:
        supported = ", ".join(TRIALS_TO_CORRECT_SCATTER_REGRESSORS)
        raise ValueError(
            f"scatter_trials_to_correct regressor_column must be one of: {supported}"
        )

    required_columns = {"block_type", "trials_to_correct", "prev_n_correct", regressor_column}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(
            "scatter_trials_to_correct block_performance is missing required "
            f"columns: {missing_columns}"
        )

    try:
        slope = float(slope)
        intercept = float(intercept)
    except (TypeError, ValueError) as exc:
        raise ValueError("scatter_trials_to_correct slope and intercept must be numeric.") from exc

    f1, ax1 = plt.subplots(figsize=(7, 5))
    ix_valid = (block_performance['trials_to_correct'] != 'None') & (block_performance['prev_n_correct'] != 'None')
    block_performance = block_performance[ix_valid]
    if block_performance.empty:
        raise ValueError("scatter_trials_to_correct requires at least one valid block row.")

    block_type = block_performance['block_type'].to_numpy()
    trials_to_correct = block_performance['trials_to_correct'].astype(int).to_numpy()
    regressor_values = block_performance[regressor_column].astype(int).to_numpy()
    jittered_regressor, jittered_trials_to_correct = _jitter_integer_scatter_points(
        regressor_values,
        trials_to_correct,
        jitter_width=point_jitter,
        seed=jitter_seed,
    )

    # use this block for separate colors
    for b in block_types:
        ix = np.where(block_type == b)[0]
        if len(ix) > 0:
            ax1.plot(jittered_regressor[ix], jittered_trials_to_correct[ix],
                     'o', color=color_dict[b], label=b.replace('_', ' '))

    # use this block for one color
    # ax1.plot(regressor_values, trials_to_correct, 'o')

    x = np.array([0, np.amax(regressor_values)])
    ax1.plot(x, slope*x + intercept, 'k--')
    _annotate_regression_slope(ax1, slope)

    plt.ylabel('Trials to Correct')
    x_label = TRIALS_TO_CORRECT_SCATTER_REGRESSORS[regressor_column]
    plt.xlabel(x_label)
    if title is not None:
        plt.title(f'{title} Trials to Switch vs {x_label}'.replace('_', ' '))
    else:
        plt.title(f'{figure_id} Trials to Switch vs {x_label}'.replace('_', ' '))

    # plt.legend(fancybox=False)
    plt.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    save_path = plot_path / f'{figure_id}_scatter_trials-to-correct_{regressor_column}.png'
    save_performance_figure(f1, save_path)


def _annotate_regression_slope(ax: plt.Axes, slope: float) -> None:
    """Display a regression slope in the upper-left corner of an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis receiving the text annotation. Coordinates are interpreted in
        axis-relative units, independent of the plotted data range.
    slope : float
        Regression slope in the units used by the plotted regression line.

    Returns
    -------
    None
        Adds unboxed text to `ax` in place. The displayed slope is rounded to
        two decimal places.
    """
    ax.text(
        0.04,
        0.96,
        f"slope = {slope:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
    )


def _plot_summary_placeholder(ax: plt.Axes, message: str) -> None:
    """Draw a visibly empty panel with an explanatory message.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the placeholder text.
    message : str
        User-facing explanation for why the panel is absent.

    Returns
    -------
    None
        Mutates `ax` in place.
    """
    ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def _get_summary_grid_block_style(block_type: str) -> dict:
    """Return side-colored marker styling for summary-grid block panels.

    Parameters
    ----------
    block_type : str
        Block label such as `"right_cued"`, `"left_uncued"`, or
        `"dark period"`.

    Returns
    -------
    dict
        Matplotlib style fields. Colors follow the colorblock raster
        convention: right=darkred, left=blue, dark=black. All block points use
        filled circles for compact summary readability.
    """
    block_type_text = str(block_type)
    if block_type_text.startswith("right_"):
        color = "darkred"
    elif block_type_text.startswith("left_"):
        color = "blue"
    else:
        color = "black"

    return {
        "color": color,
        "marker": "o",
        "markerfacecolor": color,
    }


def _plot_trials_to_correct_scatter_on_ax(
    ax: plt.Axes,
    block_performance: pd.DataFrame,
    slope: float,
    intercept: float,
    regressor_column: str = DEFAULT_TRIALS_TO_CORRECT_SCATTER_REGRESSOR,
    title: str = "Session Regression",
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
    use_summary_side_colors: bool = False,
) -> plt.Axes:
    """Plot trials-to-correct regression on an existing summary-grid axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the scatter and regression line.
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required columns
        are `block_type`, `trials_to_correct`, `prev_n_correct`, and
        `regressor_column`. Count columns are in blocks/trials.
    slope : float
        Regression slope in trials-to-correct per regressor unit.
    intercept : float
        Regression intercept in trials.
    regressor_column : str, default="prev_n_rewarded"
        Block-performance column used for both x-values and regression line.
    title : str, default="Session Regression"
        Panel title.
    point_jitter : float, default=0.08
        Maximum absolute display-only jitter in axis units.
    jitter_seed : int, default=0
        Seed for deterministic jitter.
    use_summary_side_colors : bool, default=False
        If True, use colorblock-compatible left/right colors and marker style
        to preserve cued/uncued identity. False preserves the legacy block-type
        colors used by standalone plots.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    if regressor_column not in TRIALS_TO_CORRECT_SCATTER_REGRESSORS:
        supported = ", ".join(TRIALS_TO_CORRECT_SCATTER_REGRESSORS)
        raise ValueError(f"regressor_column must be one of: {supported}")

    required_columns = {"block_type", "trials_to_correct", "prev_n_correct", regressor_column}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    valid_rows = (
        is_present_block_value(block_performance["trials_to_correct"])
        & is_present_block_value(block_performance["prev_n_correct"])
        & is_present_block_value(block_performance[regressor_column])
    )
    plot_df = block_performance.loc[valid_rows].copy()
    if plot_df.empty:
        _plot_summary_placeholder(ax, "No valid regression blocks")
        return ax

    trials_to_correct = plot_df["trials_to_correct"].astype(int).to_numpy()
    regressor_values = plot_df[regressor_column].astype(int).to_numpy()
    jittered_regressor, jittered_trials_to_correct = _jitter_integer_scatter_points(
        regressor_values,
        trials_to_correct,
        jitter_width=point_jitter,
        seed=jitter_seed,
    )
    block_type = plot_df["block_type"].to_numpy()

    for b in block_types:
        ix = np.where(block_type == b)[0]
        if len(ix) > 0:
            if use_summary_side_colors:
                style = _get_summary_grid_block_style(b)
                marker = style["marker"]
                color = style["color"]
                markerfacecolor = style["markerfacecolor"]
            else:
                marker = "o"
                color = color_dict[b]
                markerfacecolor = color
            ax.plot(
                jittered_regressor[ix],
                jittered_trials_to_correct[ix],
                marker,
                color=color,
                markerfacecolor=markerfacecolor,
                label=b.replace("_", " "),
                markersize=3,
            )

    x = np.array([0, np.amax(regressor_values)])
    ax.plot(x, slope * x + intercept, "k--", linewidth=1)
    _annotate_regression_slope(ax, slope)
    ax.set_ylabel("Trials to Correct")
    ax.set_xlabel(TRIALS_TO_CORRECT_SCATTER_REGRESSORS[regressor_column])
    ax.set_title(title)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def is_present_block_value(values: pd.Series) -> pd.Series:
    """Return rows that are not missing or string sentinel block values.

    Parameters
    ----------
    values : pd.Series
        Blockwise column with shape `(n_blocks,)`.

    Returns
    -------
    pd.Series
        Boolean mask with shape `(n_blocks,)`, True for present values.
    """
    return values.notna() & (values.astype(str) != "None")


def _plot_block_metric_on_ax(
    ax: plt.Axes,
    block_performance: pd.DataFrame,
    value_column: str,
    y_label: str,
    title: str,
    missing_marker_y: float | None = None,
    use_summary_side_colors: bool = False,
) -> plt.Axes:
    """Plot one blockwise metric by block index on an existing axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the blockwise scatter.
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`. Required columns
        are `block_type` and `value_column`; optional `block_ix` provides the
        x-axis block IDs.
    value_column : str
        Numeric block metric to plot. Missing sentinels such as `"None"` are
        skipped unless `missing_marker_y` is provided.
    y_label : str
        Y-axis label including units where applicable.
    title : str
        Panel title.
    missing_marker_y : float or None, default=None
        Optional y-position for hollow markers representing missing metric
        values, such as blocks with no correct choice.
    use_summary_side_colors : bool, default=False
        If True, use colorblock-compatible left/right colors and marker style
        to preserve cued/uncued identity. False preserves the legacy block-type
        colors used by standalone plots.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    required_columns = {"block_type", value_column}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.loc[:, ["block_type", value_column]].copy()
    if "block_ix" in block_performance.columns:
        plot_df["block_position"] = pd.to_numeric(block_performance["block_ix"], errors="coerce")
        if plot_df["block_position"].isna().any():
            plot_df["block_position"] = np.arange(block_performance.shape[0])
    else:
        plot_df["block_position"] = np.arange(block_performance.shape[0])
    plot_df["metric_numeric"] = pd.to_numeric(plot_df[value_column], errors="coerce")

    for block_type in block_types:
        block_df = plot_df[plot_df["block_type"] == block_type]
        if block_df.empty:
            continue

        numeric_blocks = block_df.dropna(subset=["metric_numeric"])
        if not numeric_blocks.empty:
            if use_summary_side_colors:
                style = _get_summary_grid_block_style(block_type)
                marker = style["marker"]
                color = style["color"]
                markerfacecolor = style["markerfacecolor"]
            else:
                marker = "o"
                color = color_dict[block_type]
                markerfacecolor = color
            ax.plot(
                numeric_blocks["block_position"].to_numpy(dtype=float),
                numeric_blocks["metric_numeric"].to_numpy(dtype=float),
                marker,
                color=color,
                markerfacecolor=markerfacecolor,
                label=block_type.replace("_", " "),
                markersize=3,
            )

        missing_blocks = block_df[block_df["metric_numeric"].isna()]
        if missing_marker_y is not None and not missing_blocks.empty:
            missing_color = (
                _get_summary_grid_block_style(block_type)["color"]
                if use_summary_side_colors
                else color_dict[block_type]
            )
            ax.plot(
                missing_blocks["block_position"].to_numpy(dtype=float),
                np.full(missing_blocks.shape[0], missing_marker_y),
                "^",
                color=missing_color,
                markerfacecolor="none",
                markersize=4,
            )

    if missing_marker_y is not None:
        ax.axhline(missing_marker_y, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Block")
    ax.set_title(title)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def _load_block_model_dict_for_summary(
    processed_data_path: Path | None,
    figure_id: str,
    block_hmm_module=None,
):
    """Load a saved block HMM dictionary when a processed-data path is supplied.

    Parameters
    ----------
    processed_data_path : pathlib.Path or None
        Directory containing `{figure_id}_block_statedict.pkl`. None means no
        load is attempted.
    figure_id : str
        Full session identifier used in the saved model filename.
    block_hmm_module : module or None, default=None
        Optional dependency injection object with `load_block_model_dict`.

    Returns
    -------
    dict or None
        Loaded model dictionary, or None when it is unavailable.
    """
    if processed_data_path is None:
        return None
    if block_hmm_module is None:
        try:
            from . import block_state_space_modeling as block_hmm_module
        except ImportError:
            from behavior_analysis import block_state_space_modeling as block_hmm_module
    try:
        return block_hmm_module.load_block_model_dict(Path(processed_data_path), figure_id)
    except FileNotFoundError:
        return None


def _plot_hmm_summary_grid_panels(
    map_ax: plt.Axes,
    regression_ax: plt.Axes,
    block_performance: pd.DataFrame,
    block_model_dict: dict | None,
    block_hmm_module=None,
    hmm_observation_plotter: Callable | None = None,
    sliding_regression_plotter: Callable | None = None,
    sliding_regression_window_size: int = 10,
    sliding_regression_step_size: int = 5,
) -> bool:
    """Plot the HMM-dependent panels in the single-session summary grid.

    Parameters
    ----------
    map_ax : matplotlib.axes.Axes
        Axis for the MAP colored trials-to-switch panel.
    regression_ax : matplotlib.axes.Axes
        Axis for the sliding block regression panel.
    block_performance : pd.DataFrame
        Blockwise table with shape `(n_blocks, n_columns)`.
    block_model_dict : dict or None
        Saved block HMM dictionary. Required keys are `map.posterior_probs`,
        `map.hmm`, and `map.weight_dict`.
    block_hmm_module : module or None, default=None
        Optional dependency injection object with block-HMM data helpers.
    hmm_observation_plotter : Callable or None, default=None
        Optional axis-level HMM observation plotting function.
    sliding_regression_plotter : Callable or None, default=None
        Optional axis-level sliding-regression plotting function.
    sliding_regression_window_size : int, default=10
        Number of valid blocks per sliding-regression window.
    sliding_regression_step_size : int, default=5
        Number of valid blocks between successive windows.

    Returns
    -------
    bool
        True when both HMM-dependent panels were available; False when
        placeholders were drawn.
    """
    if block_model_dict is None or "map" not in block_model_dict:
        _plot_summary_placeholder(map_ax, "Block HMM not available")
        _plot_summary_placeholder(regression_ax, "Windowed regression not available")
        return False

    if block_hmm_module is None:
        try:
            from . import block_state_space_modeling as block_hmm_module
        except ImportError:
            from behavior_analysis import block_state_space_modeling as block_hmm_module
    if hmm_observation_plotter is None or sliding_regression_plotter is None:
        try:
            from . import state_space_plotting
        except ImportError:
            from behavior_analysis import state_space_plotting
        if hmm_observation_plotter is None:
            hmm_observation_plotter = state_space_plotting.plot_block_lm_hmm_presentation_observations
        if sliding_regression_plotter is None:
            sliding_regression_plotter = state_space_plotting.plot_sliding_block_regression

    try:
        map_dict = block_model_dict["map"]
        prepared = block_hmm_module.prepare_block_lm_hmm_data(
            block_performance,
            predictor_columns=("prev_n_rewarded",),
        )
        posterior_probs = np.asarray(map_dict["posterior_probs"])
        hmm_fit = map_dict["hmm"]
        weight_dict = map_dict["weight_dict"]
        normalized_weights, _ = block_hmm_module.normalize_lm_observation_parameters(
            recovered_weights=weight_dict["weights"],
            recovered_mus=weight_dict["mus"],
        )
        primary_predictor_weights = normalized_weights[:, 0, 0]
        present_colors, present_cmap = block_hmm_module.build_presentation_colors(primary_predictor_weights)
        hmm_observation_plotter(
            obs_ax=map_ax,
            posterior_probs=posterior_probs,
            observations=prepared["observations"],
            hmm_fit=hmm_fit,
            colors=present_colors,
            cmap=present_cmap,
            line_width=0.8,
            xlabel="Context changes",
        )
        map_ax.set_title("MAP HMM Trials to Switch")

        sliding_regression_df = block_hmm_module.compute_sliding_block_regression(
            block_df=block_performance,
            valid_mask=prepared["valid_mask"],
            predictor_column="prev_n_rewarded",
            response_column="trials_to_correct",
            window_size=sliding_regression_window_size,
            step_size=sliding_regression_step_size,
        )
        if sliding_regression_df.empty:
            _plot_summary_placeholder(regression_ax, "Not enough valid blocks for windowed regression")
        else:
            sliding_regression_plotter(
                regression_ax=regression_ax,
                sliding_regression_df=sliding_regression_df,
                line_width=0.8,
                axis_label_size=7,
                tick_label_size=6,
                legend_font_size=5,
            )
            regression_ax.set_title("Windowed Block Regression")
        return True
    except (KeyError, ValueError, AttributeError, TypeError) as exc:
        _plot_summary_placeholder(map_ax, "Block HMM not available")
        _plot_summary_placeholder(regression_ax, "Windowed regression not available")
        print(f"Block HMM summary panels could not be plotted: {exc}")
        return False


def plot_single_session_summary_grid(
    block_performance: pd.DataFrame,
    augmented_trial_df: pd.DataFrame,
    event_df: pd.DataFrame,
    session_info: dict,
    plot_path: Path,
    figure_id: str,
    scatter_slope: float,
    scatter_intercept: float,
    title: str | None = None,
    scatter_regressor: str = DEFAULT_TRIALS_TO_CORRECT_SCATTER_REGRESSOR,
    processed_data_path: Path | None = None,
    block_model_dict: dict | None = None,
    raster_axis_plotter: Callable | None = None,
    observer_axis_plotter: Callable | None = None,
    block_hmm_module=None,
    hmm_observation_plotter: Callable | None = None,
    sliding_regression_plotter: Callable | None = None,
    sliding_regression_window_size: int = 10,
    sliding_regression_step_size: int = 5,
) -> Path:
    """Plot a 3x3 single-session behavioral and block-HMM summary grid.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns include `block_type`, `trials_to_correct`,
        `prev_n_correct`, `scatter_regressor`, `n_switches`,
        `percent_correct_after_first_correct`, and
        `block_history_ideal_mouse_agreement`.
    augmented_trial_df : pd.DataFrame
        Trial table with shape `(n_trials, n_columns)`, passed to the
        mouse-history ideal-observer plotter. Trial times and values use the
        existing observer helper's conventions.
    event_df : pd.DataFrame
        Event table with shape `(n_events, n_columns)`, required columns
        `Event` and `Time`; times are in seconds.
    session_info : dict
        Session metadata passed to the colorblock raster helper.
    plot_path : pathlib.Path
        Directory where the PNG summary grid is saved.
    figure_id : str
        Full session identifier used in the output filename.
    scatter_slope : float
        Session regression slope for `scatter_regressor`, in trials per
        regressor unit.
    scatter_intercept : float
        Session regression intercept in trials.
    title : str or None, default=None
        Optional display title prefix. None uses `figure_id`.
    scatter_regressor : str, default="prev_n_rewarded"
        Regressor column used for panel 1a.
    processed_data_path : pathlib.Path or None, default=None
        Directory containing `{figure_id}_block_statedict.pkl`. Used only when
        `block_model_dict` is None.
    block_model_dict : dict or None, default=None
        Optional saved block HMM dictionary. If absent and not loadable, panels
        2c and 3c are drawn as missing.
    raster_axis_plotter : Callable or None, default=None
        Optional dependency injection function for panel 1b.
    observer_axis_plotter : Callable or None, default=None
        Optional dependency injection function for panel 1c.
    block_hmm_module : module or None, default=None
        Optional dependency injection object for block-HMM helper functions.
    hmm_observation_plotter : Callable or None, default=None
        Optional dependency injection function for panel 2c.
    sliding_regression_plotter : Callable or None, default=None
        Optional dependency injection function for panel 3c.
    sliding_regression_window_size : int, default=10
        Number of valid blocks per sliding-regression window.
    sliding_regression_step_size : int, default=5
        Number of valid blocks between successive windows.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{figure_id}_single-session-summary-grid.png`.
    """
    if raster_axis_plotter is None:
        try:
            from . import raster_plots
        except ImportError:
            from behavior_analysis import raster_plots
        raster_axis_plotter = raster_plots.plot_colorblock_raster_on_ax
    if observer_axis_plotter is None:
        try:
            from . import plot_model_values
        except ImportError:
            from behavior_analysis import plot_model_values
        observer_axis_plotter = plot_model_values.plot_mouse_history_ideal_observer_value_on_ax

    if block_model_dict is None:
        block_model_dict = _load_block_model_dict_for_summary(
            processed_data_path=processed_data_path,
            figure_id=figure_id,
            block_hmm_module=block_hmm_module,
        )

    figure_title = figure_id if title is None else title
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    _plot_trials_to_correct_scatter_on_ax(
        ax=axes[0, 0],
        block_performance=block_performance,
        slope=scatter_slope,
        intercept=scatter_intercept,
        regressor_column=scatter_regressor,
        title="Session Regression",
        use_summary_side_colors=True,
    )
    raster_axis_plotter(
        ax=axes[0, 1],
        event_df=event_df,
        session_info=session_info,
        plot_choices=False,
        axis_label_size=8,
        tick_label_size=6,
        legend_font_size=5,
    )
    axes[0, 1].set_title("Colorblock Lick Raster")
    observer_axis_plotter(
        ax=axes[0, 2],
        trial_df=augmented_trial_df,
        value_column="agent_relative_value",
        axis_label_size=8,
        tick_label_size=6,
    )
    axes[0, 2].set_title("Mouse-History Observer Value")

    _plot_block_metric_on_ax(
        ax=axes[1, 0],
        block_performance=block_performance,
        value_column="trials_to_correct",
        y_label="Trials to Correct",
        title="Trials to Correct",
        use_summary_side_colors=True,
    )
    _plot_block_metric_on_ax(
        ax=axes[1, 1],
        block_performance=block_performance,
        value_column="n_switches",
        y_label="Number of Switches",
        title="Switches in Block",
        use_summary_side_colors=True,
    )
    hmm_available = _plot_hmm_summary_grid_panels(
        map_ax=axes[1, 2],
        regression_ax=axes[2, 2],
        block_performance=block_performance,
        block_model_dict=block_model_dict,
        block_hmm_module=block_hmm_module,
        hmm_observation_plotter=hmm_observation_plotter,
        sliding_regression_plotter=sliding_regression_plotter,
        sliding_regression_window_size=sliding_regression_window_size,
        sliding_regression_step_size=sliding_regression_step_size,
    )
    if not hmm_available:
        print("Block HMM outputs not found; summary grid saved without HMM MAP and windowed regression panels.")

    _plot_block_metric_on_ax(
        ax=axes[2, 0],
        block_performance=block_performance,
        value_column="percent_correct_after_first_correct",
        y_label="Percent Correct",
        title="Correct After First Correct",
        missing_marker_y=1.08,
        use_summary_side_colors=True,
    )
    axes[2, 0].set_ylim(-0.05, 1.18)
    _plot_block_metric_on_ax(
        ax=axes[2, 1],
        block_performance=block_performance,
        value_column="block_history_ideal_mouse_agreement",
        y_label="Agreement",
        title="History-Ideal Agreement",
        missing_marker_y=1.08,
        use_summary_side_colors=True,
    )
    axes[2, 1].set_ylim(-0.05, 1.18)

    fig.suptitle(f"{figure_title} Single-Session Summary", fontsize=14)
    save_path = plot_path / f"{figure_id}_single-session-summary-grid.png"
    save_performance_figure(fig, save_path)
    return save_path


def _get_rewarded_side_for_plot(block_type: str) -> str | None:
    """Return the rewarded side for a block type used in performance plots.

    Parameters
    ----------
    block_type : str
        Block label such as `"left_cued"`, `"right_uncued"`, or
        `"dark period"`.

    Returns
    -------
    str or None
        `"left"` for left rewarded blocks, `"right"` for right rewarded
        blocks, and None for non-side blocks such as dark periods.
    """
    block_type_text = str(block_type)
    if block_type_text.startswith("left_"):
        return "left"
    if block_type_text.startswith("right_"):
        return "right"
    return None


def _plot_side_metric_summary_on_ax(
    side_ax: plt.Axes,
    plot_df: pd.DataFrame,
    metric_column: str,
    y_label: str,
    missing_marker_y: float | None = None,
    missing_label: str = "missing",
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Plot left/right raw block values with median and Q1-Q3 summaries.

    Parameters
    ----------
    side_ax : matplotlib.axes.Axes
        Axis that receives the side summary.
    plot_df : pd.DataFrame
        Blockwise plotting table with shape `(n_blocks, n_columns)`. Required
        columns are `block_type` and `metric_column`; rows are task blocks.
        `metric_column` values are in the units described by `y_label` and may
        include missing sentinels such as `"None"`.
    metric_column : str
        Numeric metric column to summarize by rewarded side.
    y_label : str
        Y-axis label including metric units where applicable.
    missing_marker_y : float or None, default=None
        Optional y-position for missing side-block values. None omits missing
        markers.
    missing_label : str, default="missing"
        Label for missing values when `missing_marker_y` is provided.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Mutates `side_ax` in place.
    """
    required_columns = {"block_type", metric_column}
    missing_columns = sorted(required_columns.difference(plot_df.columns))
    if missing_columns:
        raise ValueError(f"plot_df is missing required columns: {missing_columns}")

    side_plot_df = plot_df.copy()
    side_plot_df["rewarded_side"] = side_plot_df["block_type"].map(_get_rewarded_side_for_plot)
    side_plot_df["metric_numeric"] = pd.to_numeric(side_plot_df[metric_column], errors="coerce")

    side_styles = {
        "left": {"x": 0.0, "color": color_dict["left_uncued"], "seed_offset": 0},
        "right": {"x": 1.0, "color": color_dict["right_uncued"], "seed_offset": 1},
    }
    side_blocks = side_plot_df[side_plot_df["rewarded_side"].isin(["left", "right"])].copy()
    for side, style in side_styles.items():
        side_df = side_blocks[side_blocks["rewarded_side"] == side]
        if side_df.empty:
            continue

        numeric_side_blocks = side_df.dropna(subset=["metric_numeric"])
        if not numeric_side_blocks.empty:
            side_ax.plot(
                _jitter_x_coordinates(
                    np.full(numeric_side_blocks.shape[0], style["x"]),
                    jitter_width=point_jitter,
                    seed=jitter_seed + style["seed_offset"],
                ),
                numeric_side_blocks["metric_numeric"].to_numpy(dtype=float),
                "o",
                color=style["color"],
                alpha=0.35,
                markersize=4,
                label=f"{side} raw blocks",
            )

            quartiles = numeric_side_blocks["metric_numeric"].quantile([0.25, 0.5, 0.75])
            q1 = float(quartiles.loc[0.25])
            median = float(quartiles.loc[0.5])
            q3 = float(quartiles.loc[0.75])
            cap_half_width = 0.08
            side_ax.plot(
                [style["x"], style["x"]],
                [q1, q3],
                "-",
                linewidth=3,
                color=style["color"],
                label=f"{side} Q1-Q3",
            )
            for cap_y in (q1, q3):
                side_ax.plot(
                    [style["x"] - cap_half_width, style["x"] + cap_half_width],
                    [cap_y, cap_y],
                    "-",
                    linewidth=3,
                    color=style["color"],
                    label="_nolegend_",
                )
            side_ax.plot(
                [style["x"]],
                [median],
                "D",
                color=style["color"],
                markersize=7,
                label=f"{side} median",
            )

        missing_side_blocks = side_df[side_df["metric_numeric"].isna()]
        if missing_marker_y is not None and not missing_side_blocks.empty:
            side_ax.plot(
                _jitter_x_coordinates(
                    np.full(missing_side_blocks.shape[0], style["x"]),
                    jitter_width=point_jitter,
                    seed=jitter_seed + 100 + style["seed_offset"],
                ),
                np.full(missing_side_blocks.shape[0], missing_marker_y),
                "^",
                color=style["color"],
                markerfacecolor="none",
                alpha=0.85,
                label=f"{side} {missing_label}",
            )

    if missing_marker_y is not None:
        side_ax.axhline(missing_marker_y, color="gray", linestyle=":", linewidth=1)
        side_ax.text(
            1.15,
            missing_marker_y,
            missing_label.replace("_", " ").title(),
            va="center",
            ha="left",
            fontsize=9,
            color="gray",
        )

    side_ax.set_xlim(-0.4, 1.4)
    if missing_marker_y is not None:
        side_ax.set_ylim(bottom=0, top=missing_marker_y + max(0.1, 0.1 * abs(missing_marker_y)))
    side_ax.set_xticks([0, 1])
    side_ax.set_xticklabels(["left", "right"])
    side_ax.set_xlabel("Rewarded side")
    side_ax.set_ylabel(y_label)
    side_ax.set_title("Side Summary")
    side_handles, _ = side_ax.get_legend_handles_labels()
    if side_handles:
        side_ax.legend(frameon=False, fontsize=8)


def plot_session_block_trials_to_correct_summary(
    block_performance: pd.DataFrame,
    plot_path: Path,
    sess_ID: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Plot single-session block completion and side-specific trials-to-correct.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type` and `trials_to_correct`; optional
        `block_ix` gives the x-axis block index. `trials_to_correct` is in
        trials and may contain missing sentinels such as `"None"` for side
        blocks where no correct choice occurred.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the plot title and output filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for side-summary raw block markers,
        in categorical x-axis units. Timeline markers are not jittered.
    jitter_seed : int, default=0
        Seed for deterministic side-summary marker jitter.

    Returns
    -------
    None
        Saves `{sess_ID}_block-trials-to-correct-summary.png` and closes the
        figure.
    """
    required_columns = {"block_type", "trials_to_correct"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.copy()
    if "block_ix" in plot_df.columns:
        block_positions = pd.to_numeric(plot_df["block_ix"], errors="coerce")
        if block_positions.isna().any():
            block_positions = pd.Series(np.arange(plot_df.shape[0]), index=plot_df.index)
    else:
        block_positions = pd.Series(np.arange(plot_df.shape[0]), index=plot_df.index)

    plot_df["block_position"] = block_positions.to_numpy(dtype=float)
    plot_df["trials_to_correct_numeric"] = pd.to_numeric(
        plot_df["trials_to_correct"],
        errors="coerce",
    )
    plot_df["rewarded_side"] = plot_df["block_type"].map(_get_rewarded_side_for_plot)

    valid_ttc = plot_df["trials_to_correct_numeric"].dropna()
    no_correct_y = float(np.ceil(valid_ttc.max()) + 1) if not valid_ttc.empty else 1.0
    dark_y = no_correct_y + 1

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    timeline_ax, side_ax = axes

    for block_type in block_types:
        block_type_rows = plot_df["block_type"] == block_type
        block_df = plot_df[block_type_rows]
        if block_df.empty:
            continue

        label_text = block_type.replace("_", " ")
        if block_type == "dark period":
            timeline_ax.plot(
                block_df["block_position"].to_numpy(dtype=float),
                np.full(block_df.shape[0], dark_y),
                "s",
                color=color_dict[block_type],
                label="dark period",
            )
            continue

        numeric_blocks = block_df.dropna(subset=["trials_to_correct_numeric"])
        if not numeric_blocks.empty:
            timeline_ax.plot(
                numeric_blocks["block_position"].to_numpy(dtype=float),
                numeric_blocks["trials_to_correct_numeric"].to_numpy(dtype=float),
                "o",
                color=color_dict[block_type],
                label=f"{label_text} timeline",
            )

        no_correct_blocks = block_df[block_df["trials_to_correct_numeric"].isna()]
        if not no_correct_blocks.empty:
            timeline_ax.plot(
                no_correct_blocks["block_position"].to_numpy(dtype=float),
                np.full(no_correct_blocks.shape[0], no_correct_y),
                "^",
                color=color_dict[block_type],
                markerfacecolor="none",
                label=f"{label_text} no correct",
            )

    timeline_ax.axhline(no_correct_y, color="gray", linestyle=":", linewidth=1)
    timeline_ax.text(
        plot_df["block_position"].max() + 0.25,
        no_correct_y,
        "No correct",
        va="center",
        ha="left",
        fontsize=9,
        color="gray",
    )
    if (plot_df["block_type"] == "dark period").any():
        timeline_ax.axhline(dark_y, color="black", linestyle=":", linewidth=1)
        timeline_ax.text(
            plot_df["block_position"].max() + 0.25,
            dark_y,
            "Dark",
            va="center",
            ha="left",
            fontsize=9,
            color="black",
        )

    timeline_ax.set_ylabel("Trials to correct")
    timeline_ax.set_xlabel("Block")
    timeline_ax.set_title(f"{sess_ID} Block Quality Timeline")
    timeline_ax.set_xlim(plot_df["block_position"].min() - 0.5, plot_df["block_position"].max() + 1.8)
    timeline_ax.set_ylim(bottom=0, top=dark_y + 1)
    timeline_handles, _ = timeline_ax.get_legend_handles_labels()
    if timeline_handles:
        timeline_ax.legend(frameon=False, fontsize=8)

    _plot_side_metric_summary_on_ax(
        side_ax=side_ax,
        plot_df=plot_df,
        metric_column="trials_to_correct_numeric",
        y_label="Trials to correct",
        missing_marker_y=no_correct_y,
        missing_label="no correct",
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )
    side_ax.set_ylim(bottom=0, top=no_correct_y + 1)

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_ID}_block-trials-to-correct-summary.png"
    save_performance_figure(fig, save_path)


def plot_session_block_quality_summary(
    block_performance: pd.DataFrame,
    plot_path: Path,
    sess_ID: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Compatibility wrapper for the renamed trials-to-correct summary plot.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the output filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for side-summary raw markers.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Saves `{sess_ID}_block-trials-to-correct-summary.png`.
    """
    plot_session_block_trials_to_correct_summary(
        block_performance=block_performance,
        plot_path=plot_path,
        sess_ID=sess_ID,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )


def plot_session_trials_to_correct(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """For a single session, plot the blockwise trials to the correct choice made by the agent."""
    f, ax = plt.subplots(figsize=(7, 5))
    plot_df = block_performance.loc[:, ['block_type', 'trials_to_correct']].copy()
    plot_df['block_position'] = np.arange(len(block_performance))
    plot_df['trials_to_correct_numeric'] = pd.to_numeric(
        plot_df['trials_to_correct'],
        errors='coerce',
    )
    plot_df = plot_df.dropna(subset=['trials_to_correct_numeric'])

    for b in block_types:
        block_df = plot_df[plot_df['block_type'] == b]
        if not block_df.empty:
            ax.plot(
                block_df['block_position'].to_numpy(),
                block_df['trials_to_correct_numeric'].astype(int).to_numpy(),
                'o',
                color=color_dict[b],
                label=b.replace('_', ' '),
            )

    plt.ylabel('Trials to Correct')
    plt.xlabel('Block')
    plt.title('{} Blockwise Trials to Correct'.format(sess_ID))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        plt.legend(fancybox=False)

    save_path = plot_path / '{}_block_trials_to_correct.png'.format(sess_ID)
    save_performance_figure(f, save_path)


def plot_session_nswitches(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """Plot blockwise switch counts with a side-specific summary panel.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type` and `n_switches`; optional
        `block_ix` gives the x-axis block index. `n_switches` is a count per
        block.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the plot title and output filename.

    Returns
    -------
    None
        Saves `{sess_ID}_block_nswitches.png`.
    """
    required_columns = {"block_type", "n_switches"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.loc[:, ["block_type", "n_switches"]].copy()
    if "block_ix" in block_performance.columns:
        plot_df["block_position"] = pd.to_numeric(block_performance["block_ix"], errors="coerce")
        if plot_df["block_position"].isna().any():
            plot_df["block_position"] = np.arange(len(block_performance))
    else:
        plot_df["block_position"] = np.arange(len(block_performance))
    plot_df["n_switches_numeric"] = pd.to_numeric(plot_df["n_switches"], errors="coerce")

    f, axes = plt.subplots(2, 1, figsize=(7, 8))
    timeline_ax, side_ax = axes
    for b in block_types:
        block_df = plot_df[plot_df["block_type"] == b].dropna(subset=["n_switches_numeric"])
        if not block_df.empty:
            timeline_ax.plot(
                block_df["block_position"].to_numpy(dtype=float),
                block_df["n_switches_numeric"].to_numpy(dtype=float),
                'o',
                color=color_dict[b],
                label=b,
            )

    timeline_ax.set_ylabel('Num Switches in block')
    timeline_ax.set_xlabel('Block')
    timeline_ax.set_title('{} Block Confusion'.format(sess_ID))
    handles, _labels = timeline_ax.get_legend_handles_labels()
    if handles:
        timeline_ax.legend(fancybox=False)

    _plot_side_metric_summary_on_ax(
        side_ax=side_ax,
        plot_df=plot_df,
        metric_column="n_switches_numeric",
        y_label="Num Switches in block",
    )

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / '{}_block_nswitches.png'.format(sess_ID)
    save_performance_figure(f, save_path)


def plot_session_explore_trials(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """Plot blockwise rewarded-switch explore-trial counts.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type` and `n_explore_trials`; optional
        `block_ix` gives the x-axis block index. `n_explore_trials` is a count
        in trials per block.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the plot title and output filename.

    Returns
    -------
    None
        Saves `{sess_ID}_block_explore_trials.png`.
    """
    required_columns = {"block_type", "n_explore_trials"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.loc[:, ["block_type", "n_explore_trials"]].copy()
    if "block_ix" in block_performance.columns:
        plot_df["block_position"] = pd.to_numeric(block_performance["block_ix"], errors="coerce")
        if plot_df["block_position"].isna().any():
            plot_df["block_position"] = np.arange(len(block_performance))
    else:
        plot_df["block_position"] = np.arange(len(block_performance))
    plot_df["n_explore_trials_numeric"] = pd.to_numeric(
        plot_df["n_explore_trials"],
        errors="coerce",
    )

    f, axes = plt.subplots(2, 1, figsize=(7, 8))
    timeline_ax, side_ax = axes
    for b in block_types:
        if b == "dark period":
            continue

        block_df = plot_df[plot_df["block_type"] == b].dropna(subset=["n_explore_trials_numeric"])
        if not block_df.empty:
            timeline_ax.plot(
                block_df["block_position"].to_numpy(dtype=float),
                block_df["n_explore_trials_numeric"].to_numpy(dtype=float),
                "o",
                color=color_dict[b],
                label=b,
            )

    timeline_ax.set_ylabel("Explore Trials in Block")
    timeline_ax.set_xlabel("Block")
    timeline_ax.set_title(f"{sess_ID} Rewarded-Switch Explore Trials")
    handles, _labels = timeline_ax.get_legend_handles_labels()
    if handles:
        timeline_ax.legend(fancybox=False)

    _plot_side_metric_summary_on_ax(
        side_ax=side_ax,
        plot_df=plot_df,
        metric_column="n_explore_trials_numeric",
        y_label="Explore Trials in Block",
    )

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_ID}_block_explore_trials.png"
    save_performance_figure(f, save_path)


def plot_session_explore_runs(block_performance: pd.DataFrame, plot_path: Path, sess_ID: str):
    """Plot blockwise short exploratory leave-return run counts.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table with shape `(n_blocks, n_columns)`.
        Required columns are `block_type` and `n_explore_runs`; optional
        `block_ix` gives the x-axis block index. `n_explore_runs` is a count in
        runs per block.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    sess_ID : str
        Session identifier used in the plot title and output filename.

    Returns
    -------
    None
        Saves `{sess_ID}_block_explore_runs.png`.
    """
    required_columns = {"block_type", "n_explore_runs"}
    missing_columns = sorted(required_columns.difference(block_performance.columns))
    if missing_columns:
        raise ValueError(f"block_performance is missing required columns: {missing_columns}")

    plot_df = block_performance.loc[:, ["block_type", "n_explore_runs"]].copy()
    if "block_ix" in block_performance.columns:
        plot_df["block_position"] = pd.to_numeric(block_performance["block_ix"], errors="coerce")
        if plot_df["block_position"].isna().any():
            plot_df["block_position"] = np.arange(len(block_performance))
    else:
        plot_df["block_position"] = np.arange(len(block_performance))
    plot_df["n_explore_runs_numeric"] = pd.to_numeric(
        plot_df["n_explore_runs"],
        errors="coerce",
    )

    f, axes = plt.subplots(2, 1, figsize=(7, 8))
    timeline_ax, side_ax = axes
    for b in block_types:
        if b == "dark period":
            continue

        block_df = plot_df[plot_df["block_type"] == b].dropna(subset=["n_explore_runs_numeric"])
        if not block_df.empty:
            timeline_ax.plot(
                block_df["block_position"].to_numpy(dtype=float),
                block_df["n_explore_runs_numeric"].to_numpy(dtype=float),
                "o",
                color=color_dict[b],
                label=b,
            )

    timeline_ax.set_ylabel("Explore Runs in Block")
    timeline_ax.set_xlabel("Block")
    timeline_ax.set_title(f"{sess_ID} Short Explore Runs")
    handles, _labels = timeline_ax.get_legend_handles_labels()
    if handles:
        timeline_ax.legend(fancybox=False)

    _plot_side_metric_summary_on_ax(
        side_ax=side_ax,
        plot_df=plot_df,
        metric_column="n_explore_runs_numeric",
        y_label="Explore Runs in Block",
    )

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{sess_ID}_block_explore_runs.png"
    save_performance_figure(f, save_path)


def plot_multisession_trials_to_correct(overall_df: pd.DataFrame, plot_path: Path, figure_id: str,
                                        use_dates: bool=True):
    block_types = ['right_cued_trials_to_correct', 'left_cued_trials_to_correct',
                   'right_uncued_trials_to_correct', 'left_uncued_trials_to_correct']
    color_dict = {'right_cued_trials_to_correct': 'darkred', 'left_cued_trials_to_correct': 'darkblue',
                  'right_uncued_trials_to_correct': 'red', 'left_uncued_trials_to_correct': 'blue'}
    overall_df = overall_df[~overall_df['date'].isna()]
    dates = overall_df['date'].unique()

    f, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(overall_df.shape[0])
    for b in block_types:
        type_ttc = overall_df[b].values
        if np.sum(np.isnan(type_ttc)) == type_ttc.size:  # will have to remove this later on...
            continue

        type_ttc[np.isnan(type_ttc)] = 0
        ax.plot(x, type_ttc, color=color_dict[b], label=b)
        if use_dates:
            ax.set_xticks(x, dates)
            ax.tick_params(axis='x', labelrotation=60, labelsize=8)

    plt.ylabel('Trials to Correct')
    plt.xlabel('Session')
    plt.title('{} Overall Trials to Correct'.format(figure_id))
    plt.legend(fancybox=False)

    save_path = plot_path / '{}_overall_trials_to_correct.png'.format(figure_id)
    save_performance_figure(f, save_path)


def plot_learning_curve(coefficients: np.ndarray, switches_per_session: np.ndarray, figure_id: str, plot_path: Path, dates: list[str]=None):
    f, ax = plt.subplots(figsize=(8, 5))
    plt.plot(np.arange(1, len(coefficients) + 1), coefficients, 'ko-')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Day', fontsize=16)
    plt.ylabel('Regression Coefficient', fontsize=18)
    plt.title('{} Learning Curve'.format(figure_id))
    ax.tick_params(axis='y', which='major', labelsize=12)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_linewidth(2)
    # ax.spines['left'].set_linewidth(2)
    # ax.tick_params(width=2)

    if dates is not None:
        ax.set_xticks(np.arange(1, len(dates) + 1))
        ax.set_xticklabels(dates, rotation=60, fontsize=8)

    save_path = plot_path / '{}_learning-curve.png'.format(figure_id)
    save_performance_figure(f, save_path)


def plot_cross_mouse_learning_curve(
    cross_mouse_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    learning_regressor: str,
) -> Path:
    """Plot mouse-specific and group-mean learning curves.

    Parameters
    ----------
    cross_mouse_df : pd.DataFrame
        Long-form dataframe with shape `(n_mouse_sessions, n_columns)`.
        Required columns are `mouse`, `training_day`, and `slope`.
        `training_day` is a one-indexed session number within each mouse, and
        `slope` is the regression coefficient for `learning_regressor` in
        trials-to-switch per predictor unit.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Figure identifier used as the filename prefix and title label.
    learning_regressor : str
        Regression predictor prefix used in the plot title and filename.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    required_columns = {"mouse", "training_day", "slope"}
    missing_columns = sorted(required_columns.difference(cross_mouse_df.columns))
    if missing_columns:
        raise ValueError(f"cross_mouse_df is missing required columns: {missing_columns}")

    plot_df = cross_mouse_df.copy()
    plot_df["training_day"] = pd.to_numeric(plot_df["training_day"], errors="raise").astype(int)
    plot_df["slope"] = pd.to_numeric(plot_df["slope"], errors="raise")
    plot_df = plot_df.sort_values(["mouse", "training_day"])

    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    mouse_names = sorted(plot_df["mouse"].astype(str).unique())
    for mouse_index, mouse in enumerate(mouse_names):
        mouse_df = plot_df[plot_df["mouse"].astype(str) == mouse].sort_values("training_day")
        ax.plot(
            mouse_df["training_day"].to_numpy(dtype=int),
            mouse_df["slope"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            color=all_colors[mouse_index % len(all_colors)],
            alpha=0.75,
            label=mouse,
        )

    mean_df = (
        plot_df.groupby("training_day", sort=True)["slope"]
        .mean()
        .reset_index()
    )
    ax.plot(
        mean_df["training_day"].to_numpy(dtype=int),
        mean_df["slope"].to_numpy(dtype=float),
        marker="o",
        linewidth=3.0,
        color="black",
        label="group mean",
        zorder=5,
    )
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Training Day")
    ax.set_ylabel("Regression Coefficient")
    ax.set_title(f"{figure_id} {learning_regressor} Learning Curve")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fancybox=False, frameon=False)

    save_path = plot_path / f"{figure_id}_{learning_regressor}_learning_curve.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_cross_mouse_session_metric_curve(
    cross_mouse_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    metric_column: str,
    metric_label: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    """Plot mouse-specific and group-mean session metric curves.

    Parameters
    ----------
    cross_mouse_df : pd.DataFrame
        Long-form dataframe with shape `(n_mouse_sessions, n_columns)`.
        Required columns are `mouse`, `training_day`, and `metric_value`.
        `training_day` is a one-indexed session number within each mouse, and
        `metric_value` is the session-level metric in the units described by
        `metric_label`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Figure identifier used as the filename prefix and title label.
    metric_column : str
        Source metric column name from the overall-performance CSV. This is
        used in the output filename and title.
    metric_label : str
        Human-readable y-axis label describing the plotted metric and units.
    ylim : tuple[float, float] or None, default=None
        Optional y-axis limits in `metric_value` units. None lets matplotlib
        choose limits from the data.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    required_columns = {"mouse", "training_day", "metric_value"}
    missing_columns = sorted(required_columns.difference(cross_mouse_df.columns))
    if missing_columns:
        raise ValueError(f"cross_mouse_df is missing required columns: {missing_columns}")

    plot_df = cross_mouse_df.copy()
    plot_df["training_day"] = pd.to_numeric(plot_df["training_day"], errors="raise").astype(int)
    plot_df["metric_value"] = pd.to_numeric(plot_df["metric_value"], errors="raise")
    plot_df = plot_df.sort_values(["mouse", "training_day"])

    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    mouse_names = sorted(plot_df["mouse"].astype(str).unique())
    for mouse_index, mouse in enumerate(mouse_names):
        mouse_df = plot_df[plot_df["mouse"].astype(str) == mouse].sort_values("training_day")
        ax.plot(
            mouse_df["training_day"].to_numpy(dtype=int),
            mouse_df["metric_value"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            color=all_colors[mouse_index % len(all_colors)],
            alpha=0.75,
            label=mouse,
        )

    mean_df = (
        plot_df.groupby("training_day", sort=True)["metric_value"]
        .mean()
        .reset_index()
    )
    ax.plot(
        mean_df["training_day"].to_numpy(dtype=int),
        mean_df["metric_value"].to_numpy(dtype=float),
        marker="o",
        linewidth=3.0,
        color="black",
        label="group mean",
        zorder=5,
    )
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training Day")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{figure_id} {metric_column} Across Training")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fancybox=False, frameon=False)

    save_path = plot_path / f"{figure_id}_{metric_column}_session_metric_curve.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_trials_to_correct_session_summary(
    summary_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
) -> None:
    """Plot session-level trials-to-correct median and quartile range.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Session summary table with shape `(n_sessions, n_columns)`. Required
        columns are `date`, `trials_to_correct_q1`,
        `trials_to_correct_median`, and `trials_to_correct_q3`. Trial-count
        columns are counts in trials and are plotted after numeric coercion.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output filename.

    Returns
    -------
    None
        Saves `{figure_id}_trials-to-correct-session-summary.png`.
    """
    required_columns = [
        "date",
        "trials_to_correct_q1",
        "trials_to_correct_median",
        "trials_to_correct_q3",
    ]
    missing_columns = sorted(set(required_columns).difference(summary_df.columns))
    if missing_columns:
        missing_summary = ", ".join(missing_columns)
        raise ValueError(
            "summary_df is missing required columns for trials-to-correct plotting: "
            f"{missing_summary}"
        )

    plot_df = summary_df.loc[:, required_columns].copy()
    numeric_columns = [
        "trials_to_correct_q1",
        "trials_to_correct_median",
        "trials_to_correct_q3",
    ]
    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=numeric_columns)
    if plot_df.empty:
        raise ValueError("summary_df must contain at least one valid session to plot.")

    x = np.arange(1, plot_df.shape[0] + 1)
    f, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        x,
        plot_df["trials_to_correct_q1"].to_numpy(dtype=float),
        plot_df["trials_to_correct_q3"].to_numpy(dtype=float),
        alpha=0.3,
        linewidth=0,
        label="Q1-Q3",
    )
    ax.plot(
        x,
        plot_df["trials_to_correct_median"].to_numpy(dtype=float),
        "ko-",
        label="Median",
    )

    ax.set_xlabel("Day", fontsize=16)
    ax.set_ylabel("Trials to Correct", fontsize=18)
    ax.set_title("{} Trials to Correct Across Sessions".format(figure_id))
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["date"].to_numpy(), rotation=60, fontsize=8)
    ax.tick_params(axis="y", which="major", labelsize=12)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)

    save_path = plot_path / "{}_trials-to-correct-session-summary.png".format(figure_id)
    save_performance_figure(f, save_path)


def plot_correct_after_first_session_summary(
    summary_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
) -> Path:
    """Plot overall post-first-correct accuracy median and quartile range.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Cross-session post-first-correct summary with shape `(n_rows,
        n_columns)`. Required columns are `date`,
        `correct_after_first_group`, `percent_correct_after_first_q1`,
        `percent_correct_after_first_median`, and
        `percent_correct_after_first_q3`. Only rows with
        `correct_after_first_group == "overall"` are plotted. Values are
        unitless fractions.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output filename.

    Returns
    -------
    pathlib.Path
        Saved PNG path.
    """
    required_columns = [
        "date",
        "correct_after_first_group",
        "percent_correct_after_first_q1",
        "percent_correct_after_first_median",
        "percent_correct_after_first_q3",
    ]
    missing_columns = sorted(set(required_columns).difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_columns}")

    plot_df = summary_df.loc[
        summary_df["correct_after_first_group"] == "overall",
        required_columns,
    ].copy()
    numeric_columns = [
        "percent_correct_after_first_q1",
        "percent_correct_after_first_median",
        "percent_correct_after_first_q3",
    ]
    for column in numeric_columns:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=numeric_columns)
    if plot_df.empty:
        raise ValueError("summary_df must contain at least one valid overall session row.")

    x = np.arange(1, plot_df.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        x,
        plot_df["percent_correct_after_first_q1"].to_numpy(dtype=float),
        plot_df["percent_correct_after_first_q3"].to_numpy(dtype=float),
        alpha=0.3,
        linewidth=0,
        label="Q1-Q3",
    )
    ax.plot(
        x,
        plot_df["percent_correct_after_first_median"].to_numpy(dtype=float),
        "ko-",
        label="Median",
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Day", fontsize=16)
    ax.set_ylabel("Post-first-correct accuracy", fontsize=16)
    ax.set_title(f"{figure_id} Post-First-Correct Accuracy Across Sessions")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["date"].to_numpy(), rotation=60, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)

    save_path = plot_path / f"{figure_id}_correct-after-first-session-summary.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_session_summary_metric_family(
    overall_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    family_name: str,
    metric_specs: dict[str, str],
    y_label: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    """Plot related session summary metrics together across sessions.

    Parameters
    ----------
    overall_df : pd.DataFrame
        Overall-performance summary with shape `(n_sessions, n_columns)`.
        Required columns are `date` and every key in `metric_specs`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output filename.
    family_name : str
        Short name used in the output filename.
    metric_specs : dict[str, str]
        Mapping from overall-performance column name to legend label. Values
        are unitless unless specified by `y_label`.
    y_label : str
        Axis label describing the metric family and units.
    ylim : tuple[float, float] or None, default=None
        Optional y-axis limits in metric units.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{figure_id}_{family_name}-session-summary-metrics.png`.
    """
    required_columns = {"date", *metric_specs.keys()}
    missing_columns = sorted(required_columns.difference(overall_df.columns))
    if missing_columns:
        raise ValueError(f"overall_df is missing required columns: {missing_columns}")

    plot_df = overall_df.copy()
    for column in metric_specs:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.sort_values("date").reset_index(drop=True)
    x = np.arange(1, plot_df.shape[0] + 1)

    plot_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for column, label in metric_specs.items():
        valid_rows = plot_df[column].notna().to_numpy()
        if not valid_rows.any():
            continue
        ax.plot(
            x[valid_rows],
            plot_df.loc[valid_rows, column].to_numpy(dtype=float),
            "o-",
            linewidth=2,
            label=label,
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Session date")
    ax.set_ylabel(y_label)
    ax.set_title(f"{figure_id} {family_name.replace('_', ' ').title()} Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["date"].to_numpy(), rotation=60, fontsize=8)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{figure_id}_{family_name}-session-summary-metrics.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_side_trials_to_correct_quality_on_ax(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
    trial_display_cap: float = 25,
) -> plt.Axes:
    """Plot cross-session trials-to-correct medians, Q1-Q3, and raw blocks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the trials-to-correct summary.
    summary_df : pd.DataFrame
        Side-specific session summary with shape `(n_session_sides,
        n_columns)`. Required columns are `date`, `rewarded_side`,
        `completion_fraction`, `trials_to_correct_q1`,
        `trials_to_correct_median`, and `trials_to_correct_q3`. Trial-count
        columns are in trials.
    block_points_df : pd.DataFrame
        Raw side-block table with shape `(n_side_blocks, n_columns)`.
        Required columns are `date`, `rewarded_side`,
        `trials_to_correct_numeric`, and `no_correct_choice`.
        `trials_to_correct_numeric` is in trials; `no_correct_choice` marks
        side blocks where no numeric trials-to-correct value exists.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units. Summary lines are not jittered.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.
    trial_display_cap : float, default=25
        Maximum trials-to-correct value plotted to scale. Values above this
        threshold are plotted on a shared overflow row at
        `trial_display_cap + 1`, preserving low-end resolution.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    summary_required_columns = {
        "date",
        "rewarded_side",
        "completion_fraction",
        "trials_to_correct_q1",
        "trials_to_correct_median",
        "trials_to_correct_q3",
    }
    point_required_columns = {
        "date",
        "rewarded_side",
        "trials_to_correct_numeric",
        "no_correct_choice",
    }
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    missing_point_columns = sorted(point_required_columns.difference(block_points_df.columns))
    if missing_summary_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_summary_columns}")
    if missing_point_columns:
        raise ValueError(f"block_points_df is missing required columns: {missing_point_columns}")

    summary_plot_df = summary_df.copy()
    point_plot_df = block_points_df.copy()
    numeric_summary_columns = [
        "completion_fraction",
        "trials_to_correct_q1",
        "trials_to_correct_median",
        "trials_to_correct_q3",
    ]
    for column in numeric_summary_columns:
        summary_plot_df[column] = pd.to_numeric(summary_plot_df[column], errors="coerce")
    point_plot_df["trials_to_correct_numeric"] = pd.to_numeric(
        point_plot_df["trials_to_correct_numeric"],
        errors="coerce",
    )
    point_plot_df["no_correct_choice"] = (
        point_plot_df["no_correct_choice"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    )

    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one side/session row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    side_styles = {
        "overall": {"color": "black", "seed_offset": 2, "zorder": 3},
        "left": {"color": color_dict["left_uncued"], "seed_offset": 0, "zorder": 2},
        "right": {"color": color_dict["right_uncued"], "seed_offset": 1, "zorder": 2},
    }

    finite_trial_values = pd.concat(
        [
            summary_plot_df["trials_to_correct_q3"],
            point_plot_df["trials_to_correct_numeric"],
        ],
        axis=0,
    ).dropna()
    overflow_y = float(trial_display_cap + 1)
    has_overflow_values = bool((finite_trial_values > trial_display_cap).any())
    if has_overflow_values:
        no_correct_y = float(trial_display_cap + 2)
    else:
        no_correct_y = float(np.ceil(finite_trial_values.max()) + 1) if not finite_trial_values.empty else 1.0

    def _display_trial_values(values: pd.Series | np.ndarray) -> np.ndarray:
        """Map high trials-to-correct values to the overflow display row."""
        value_array = np.asarray(values, dtype=float)
        return np.where(value_array > trial_display_cap, overflow_y, value_array)

    for side, style in side_styles.items():
        side_summary = summary_plot_df[summary_plot_df["rewarded_side"] == side].copy()
        if side_summary.empty:
            continue

        side_summary["x_position"] = side_summary["date"].map(x_by_date)
        side_summary.sort_values("x_position", inplace=True)

        spread_rows = side_summary.dropna(
            subset=[
                "trials_to_correct_q1",
                "trials_to_correct_median",
                "trials_to_correct_q3",
            ]
        )
        if not spread_rows.empty:
            ax.fill_between(
                spread_rows["x_position"].to_numpy(dtype=float),
                _display_trial_values(spread_rows["trials_to_correct_q1"]),
                _display_trial_values(spread_rows["trials_to_correct_q3"]),
                alpha=0.2,
                linewidth=0,
                color=style["color"],
                label=f"{side} Q1-Q3",
                zorder=style["zorder"],
            )
            ax.plot(
                spread_rows["x_position"].to_numpy(dtype=float),
                _display_trial_values(spread_rows["trials_to_correct_median"]),
                "o-",
                color=style["color"],
                label=f"{side} median",
                zorder=style["zorder"],
            )

        side_points = point_plot_df[point_plot_df["rewarded_side"] == side].copy()
        if side_points.empty:
            continue

        side_points["x_position"] = side_points["date"].map(x_by_date)
        numeric_points = side_points.dropna(subset=["trials_to_correct_numeric"])
        if not numeric_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    numeric_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + style["seed_offset"],
                ),
                _display_trial_values(numeric_points["trials_to_correct_numeric"]),
                "o",
                color=style["color"],
                alpha=0.3,
                markersize=4,
                label=f"{side} raw blocks",
                zorder=style["zorder"],
            )

        no_correct_points = side_points[side_points["no_correct_choice"]]
        if not no_correct_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    no_correct_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + 100 + style["seed_offset"],
                ),
                np.full(no_correct_points.shape[0], no_correct_y),
                "^",
                color=style["color"],
                markerfacecolor="none",
                alpha=0.8,
                label=f"{side} no correct",
            )

    if has_overflow_values:
        ax.axhline(overflow_y, color="gray", linestyle=":", linewidth=1)
        ax.text(
            len(dates) + 0.25,
            overflow_y,
            f">{trial_display_cap:g} trials",
            va="center",
            ha="left",
            fontsize=9,
            color="gray",
        )
    ax.axhline(no_correct_y, color="gray", linestyle=":", linewidth=1)
    ax.text(
        len(dates) + 0.25,
        no_correct_y,
        "No correct",
        va="center",
        ha="left",
        fontsize=9,
        color="gray",
    )
    ax.set_ylabel("Trials to correct")
    ax.set_xlabel("Session date")
    ax.set_title(f"{figure_id} Trials to Correct Across Sessions")
    ax.set_xticks(np.arange(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=60, fontsize=8)
    ax.set_ylim(bottom=0, top=no_correct_y + 1)
    trial_handles, _ = ax.get_legend_handles_labels()
    if trial_handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_side_trials_to_correct_quality(
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
    trial_display_cap: float = 25,
) -> None:
    """Plot cross-session block completion and trials-to-correct by rewarded side.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Side-specific session summary with shape `(n_session_sides,
        n_columns)`. Required columns are `date`, `rewarded_side`,
        `completion_fraction`, `trials_to_correct_q1`,
        `trials_to_correct_median`, and `trials_to_correct_q3`. Trial-count
        columns are in trials.
    block_points_df : pd.DataFrame
        Raw side-block table with shape `(n_side_blocks, n_columns)`.
        Required columns are `date`, `rewarded_side`,
        `trials_to_correct_numeric`, and `no_correct_choice`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output
        filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.
    trial_display_cap : float, default=25
        Maximum trials-to-correct value plotted to scale. Larger values are
        drawn on a shared overflow row.

    Returns
    -------
    None
        Saves `{figure_id}_side-trials-to-correct-quality.png`.
    """
    required_columns = {"date", "rewarded_side", "completion_fraction"}
    missing_columns = sorted(required_columns.difference(summary_df.columns))
    if missing_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_columns}")

    summary_plot_df = summary_df.copy()
    summary_plot_df["completion_fraction"] = pd.to_numeric(
        summary_plot_df["completion_fraction"],
        errors="coerce",
    )
    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one side/session row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    side_styles = {
        "overall": {"color": "black"},
        "left": {"color": color_dict["left_uncued"]},
        "right": {"color": color_dict["right_uncued"]},
    }

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    completion_ax, trials_ax = axes

    for side, style in side_styles.items():
        side_summary = summary_plot_df[summary_plot_df["rewarded_side"] == side].copy()
        if side_summary.empty:
            continue

        side_summary["x_position"] = side_summary["date"].map(x_by_date)
        side_summary.sort_values("x_position", inplace=True)
        completion_ax.plot(
            side_summary["x_position"].to_numpy(dtype=float),
            side_summary["completion_fraction"].to_numpy(dtype=float),
            "o-",
            color=style["color"],
            label=f"{side} completion",
        )

    completion_ax.set_ylim(-0.05, 1.05)
    completion_ax.set_ylabel("Block completion fraction")
    completion_ax.set_title(f"{figure_id} Side-Specific Block Quality")
    completion_handles, _ = completion_ax.get_legend_handles_labels()
    if completion_handles:
        completion_ax.legend(frameon=False)

    plot_side_trials_to_correct_quality_on_ax(
        ax=trials_ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
        trial_display_cap=trial_display_cap,
    )

    for ax in axes:
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    save_path = plot_path / f"{figure_id}_side-trials-to-correct-quality.png"
    save_performance_figure(fig, save_path)


def plot_multisession_block_switches_quality_on_ax(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> plt.Axes:
    """Plot cross-session choice switches per block on one axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the switch summary.
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`, `switch_group`,
        `n_switches_q1`, `n_switches_median`, and `n_switches_q3`.
        Switch counts are raw within-block choice switches.
    block_points_df : pd.DataFrame
        Raw switch table with shape `(n_points, n_columns)`. Required columns
        are `date`, `switch_group`, and `n_switches`. Side blocks appear in
        both their side-specific group and the `"overall"` group.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units. Summary lines are not jittered.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    summary_required_columns = {
        "date",
        "switch_group",
        "n_switches_q1",
        "n_switches_median",
        "n_switches_q3",
    }
    point_required_columns = {"date", "switch_group", "n_switches"}
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    missing_point_columns = sorted(point_required_columns.difference(block_points_df.columns))
    if missing_summary_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_summary_columns}")
    if missing_point_columns:
        raise ValueError(f"block_points_df is missing required columns: {missing_point_columns}")

    summary_plot_df = summary_df.copy()
    point_plot_df = block_points_df.copy()
    numeric_summary_columns = ["n_switches_q1", "n_switches_median", "n_switches_q3"]
    for column in numeric_summary_columns:
        summary_plot_df[column] = pd.to_numeric(summary_plot_df[column], errors="coerce")
    point_plot_df["n_switches"] = pd.to_numeric(point_plot_df["n_switches"], errors="coerce")
    point_plot_df = point_plot_df.dropna(subset=["n_switches"])

    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one session/group row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    group_styles = {
        "overall": {"color": "black", "seed_offset": 0, "alpha": 0.14},
        "left": {"color": color_dict["left_uncued"], "seed_offset": 1, "alpha": 0.18},
        "right": {"color": color_dict["right_uncued"], "seed_offset": 2, "alpha": 0.18},
    }

    for switch_group, style in group_styles.items():
        group_summary = summary_plot_df[summary_plot_df["switch_group"] == switch_group].copy()
        if group_summary.empty:
            continue

        group_summary["x_position"] = group_summary["date"].map(x_by_date)
        group_summary.sort_values("x_position", inplace=True)
        spread_rows = group_summary.dropna(subset=numeric_summary_columns)
        if not spread_rows.empty:
            ax.fill_between(
                spread_rows["x_position"].to_numpy(dtype=float),
                spread_rows["n_switches_q1"].to_numpy(dtype=float),
                spread_rows["n_switches_q3"].to_numpy(dtype=float),
                alpha=style["alpha"],
                linewidth=0,
                color=style["color"],
                label=f"{switch_group} Q1-Q3",
            )
            ax.plot(
                spread_rows["x_position"].to_numpy(dtype=float),
                spread_rows["n_switches_median"].to_numpy(dtype=float),
                "o-",
                color=style["color"],
                label=f"{switch_group} median",
            )

        group_points = point_plot_df[point_plot_df["switch_group"] == switch_group].copy()
        if group_points.empty:
            continue

        group_points["x_position"] = group_points["date"].map(x_by_date)
        group_points = group_points.dropna(subset=["x_position"])
        if not group_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    group_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + style["seed_offset"],
                ),
                group_points["n_switches"].to_numpy(dtype=float),
                "o",
                color=style["color"],
                alpha=0.28 if switch_group == "overall" else 0.35,
                markersize=4,
                label=f"{switch_group} raw blocks",
            )

    ax.set_ylabel("Choice Switches per Block")
    ax.set_xlabel("Session date")
    ax.set_title(f"{figure_id} Choice Switches per Block Across Sessions")
    ax.set_xticks(np.arange(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=60, fontsize=8)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_multisession_block_switches_quality(
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Plot cross-session raw choice switches per block.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`, `switch_group`,
        `n_switches_q1`, `n_switches_median`, and `n_switches_q3`.
    block_points_df : pd.DataFrame
        Raw switch table with shape `(n_points, n_columns). Required columns
        are `date`, `switch_group`, and `n_switches`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output
        filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Saves `{figure_id}_block-switches-quality.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_multisession_block_switches_quality_on_ax(
        ax=ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )

    save_path = plot_path / f"{figure_id}_block-switches-quality.png"
    save_performance_figure(fig, save_path)


def plot_multisession_block_explore_quality_on_ax(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> plt.Axes:
    """Plot cross-session rewarded-switch explore trials per block on one axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the explore-trial summary.
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`, `explore_group`,
        `n_explore_trials_q1`, `n_explore_trials_median`, and
        `n_explore_trials_q3`. Counts are trials per block.
    block_points_df : pd.DataFrame
        Raw explore table with shape `(n_points, n_columns)`. Required columns
        are `date`, `explore_group`, and `n_explore_trials`. Side blocks appear
        in both their side-specific group and the `"overall"` group.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units. Summary lines are not jittered.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    summary_required_columns = {
        "date",
        "explore_group",
        "n_explore_trials_q1",
        "n_explore_trials_median",
        "n_explore_trials_q3",
    }
    point_required_columns = {"date", "explore_group", "n_explore_trials"}
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    missing_point_columns = sorted(point_required_columns.difference(block_points_df.columns))
    if missing_summary_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_summary_columns}")
    if missing_point_columns:
        raise ValueError(f"block_points_df is missing required columns: {missing_point_columns}")

    summary_plot_df = summary_df.copy()
    point_plot_df = block_points_df.copy()
    numeric_summary_columns = [
        "n_explore_trials_q1",
        "n_explore_trials_median",
        "n_explore_trials_q3",
    ]
    for column in numeric_summary_columns:
        summary_plot_df[column] = pd.to_numeric(summary_plot_df[column], errors="coerce")
    point_plot_df["n_explore_trials"] = pd.to_numeric(
        point_plot_df["n_explore_trials"],
        errors="coerce",
    )
    point_plot_df = point_plot_df.dropna(subset=["n_explore_trials"])

    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one session/group row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    group_styles = {
        "overall": {"color": "black", "seed_offset": 0, "alpha": 0.14},
        "left": {"color": color_dict["left_uncued"], "seed_offset": 1, "alpha": 0.18},
        "right": {"color": color_dict["right_uncued"], "seed_offset": 2, "alpha": 0.18},
    }

    for explore_group, style in group_styles.items():
        group_summary = summary_plot_df[summary_plot_df["explore_group"] == explore_group].copy()
        if group_summary.empty:
            continue

        group_summary["x_position"] = group_summary["date"].map(x_by_date)
        group_summary.sort_values("x_position", inplace=True)
        spread_rows = group_summary.dropna(subset=numeric_summary_columns)
        if not spread_rows.empty:
            ax.fill_between(
                spread_rows["x_position"].to_numpy(dtype=float),
                spread_rows["n_explore_trials_q1"].to_numpy(dtype=float),
                spread_rows["n_explore_trials_q3"].to_numpy(dtype=float),
                alpha=style["alpha"],
                linewidth=0,
                color=style["color"],
                label=f"{explore_group} Q1-Q3",
            )
            ax.plot(
                spread_rows["x_position"].to_numpy(dtype=float),
                spread_rows["n_explore_trials_median"].to_numpy(dtype=float),
                "o-",
                color=style["color"],
                label=f"{explore_group} median",
            )

        group_points = point_plot_df[point_plot_df["explore_group"] == explore_group].copy()
        if group_points.empty:
            continue

        group_points["x_position"] = group_points["date"].map(x_by_date)
        group_points = group_points.dropna(subset=["x_position"])
        if not group_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    group_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + style["seed_offset"],
                ),
                group_points["n_explore_trials"].to_numpy(dtype=float),
                "o",
                color=style["color"],
                alpha=0.28 if explore_group == "overall" else 0.35,
                markersize=4,
                label=f"{explore_group} raw blocks",
            )

    ax.set_ylabel("Explore Trials per Block")
    ax.set_xlabel("Session date")
    ax.set_title(f"{figure_id} Rewarded-Switch Explore Trials Across Sessions")
    ax.set_xticks(np.arange(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=60, fontsize=8)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_multisession_block_explore_quality(
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Plot cross-session rewarded-switch explore trials per block.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`, `explore_group`,
        `n_explore_trials_q1`, `n_explore_trials_median`, and
        `n_explore_trials_q3`.
    block_points_df : pd.DataFrame
        Raw explore table with shape `(n_points, n_columns)`. Required columns
        are `date`, `explore_group`, and `n_explore_trials`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output
        filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Saves `{figure_id}_block-explore-quality.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_multisession_block_explore_quality_on_ax(
        ax=ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )

    save_path = plot_path / f"{figure_id}_block-explore-quality.png"
    save_performance_figure(fig, save_path)


def plot_multisession_block_explore_run_quality_on_ax(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> plt.Axes:
    """Plot cross-session short explore runs per block on one axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the explore-run summary.
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`, `explore_group`,
        `n_explore_runs_q1`, `n_explore_runs_median`, and
        `n_explore_runs_q3`. Counts are runs per block.
    block_points_df : pd.DataFrame
        Raw explore-run table with shape `(n_points, n_columns)`. Required
        columns are `date`, `explore_group`, and `n_explore_runs`. Side blocks
        appear in both their side-specific group and the `"overall"` group.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units. Summary lines are not jittered.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    summary_required_columns = {
        "date",
        "explore_group",
        "n_explore_runs_q1",
        "n_explore_runs_median",
        "n_explore_runs_q3",
    }
    point_required_columns = {"date", "explore_group", "n_explore_runs"}
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    missing_point_columns = sorted(point_required_columns.difference(block_points_df.columns))
    if missing_summary_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_summary_columns}")
    if missing_point_columns:
        raise ValueError(f"block_points_df is missing required columns: {missing_point_columns}")

    summary_plot_df = summary_df.copy()
    point_plot_df = block_points_df.copy()
    numeric_summary_columns = [
        "n_explore_runs_q1",
        "n_explore_runs_median",
        "n_explore_runs_q3",
    ]
    for column in numeric_summary_columns:
        summary_plot_df[column] = pd.to_numeric(summary_plot_df[column], errors="coerce")
    point_plot_df["n_explore_runs"] = pd.to_numeric(
        point_plot_df["n_explore_runs"],
        errors="coerce",
    )
    point_plot_df = point_plot_df.dropna(subset=["n_explore_runs"])

    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one session/group row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    group_styles = {
        "overall": {"color": "black", "seed_offset": 0, "alpha": 0.14},
        "left": {"color": color_dict["left_uncued"], "seed_offset": 1, "alpha": 0.18},
        "right": {"color": color_dict["right_uncued"], "seed_offset": 2, "alpha": 0.18},
    }

    for explore_group, style in group_styles.items():
        group_summary = summary_plot_df[summary_plot_df["explore_group"] == explore_group].copy()
        if group_summary.empty:
            continue

        group_summary["x_position"] = group_summary["date"].map(x_by_date)
        group_summary.sort_values("x_position", inplace=True)
        spread_rows = group_summary.dropna(subset=numeric_summary_columns)
        if not spread_rows.empty:
            ax.fill_between(
                spread_rows["x_position"].to_numpy(dtype=float),
                spread_rows["n_explore_runs_q1"].to_numpy(dtype=float),
                spread_rows["n_explore_runs_q3"].to_numpy(dtype=float),
                alpha=style["alpha"],
                linewidth=0,
                color=style["color"],
                label=f"{explore_group} Q1-Q3",
            )
            ax.plot(
                spread_rows["x_position"].to_numpy(dtype=float),
                spread_rows["n_explore_runs_median"].to_numpy(dtype=float),
                "o-",
                color=style["color"],
                label=f"{explore_group} median",
            )

        group_points = point_plot_df[point_plot_df["explore_group"] == explore_group].copy()
        if group_points.empty:
            continue

        group_points["x_position"] = group_points["date"].map(x_by_date)
        group_points = group_points.dropna(subset=["x_position"])
        if not group_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    group_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + style["seed_offset"],
                ),
                group_points["n_explore_runs"].to_numpy(dtype=float),
                "o",
                color=style["color"],
                alpha=0.28 if explore_group == "overall" else 0.35,
                markersize=4,
                label=f"{explore_group} raw blocks",
            )

    ax.set_ylabel("Explore Runs per Block")
    ax.set_xlabel("Session date")
    ax.set_title(f"{figure_id} Short Explore Runs Across Sessions")
    ax.set_xticks(np.arange(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=60, fontsize=8)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_multisession_block_explore_run_quality(
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Plot cross-session short exploratory runs per block.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`, `explore_group`,
        `n_explore_runs_q1`, `n_explore_runs_median`, and `n_explore_runs_q3`.
    block_points_df : pd.DataFrame
        Raw explore-run table with shape `(n_points, n_columns)`. Required
        columns are `date`, `explore_group`, and `n_explore_runs`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output
        filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Saves `{figure_id}_block-explore-run-quality.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_multisession_block_explore_run_quality_on_ax(
        ax=ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )

    save_path = plot_path / f"{figure_id}_block-explore-run-quality.png"
    save_performance_figure(fig, save_path)


def plot_multisession_correct_after_first_quality_on_ax(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> plt.Axes:
    """Plot cross-session accuracy after first correct choice on one axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the accuracy summary.
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`,
        `correct_after_first_group`, `percent_correct_after_first_q1`,
        `percent_correct_after_first_median`, and
        `percent_correct_after_first_q3`. Percent-correct values are fractions
        from 0 to 1.
    block_points_df : pd.DataFrame
        Raw side-block table with shape `(n_points, n_columns)`. Required
        columns are `date`, `correct_after_first_group`,
        `percent_correct_after_first_numeric`, and `no_correct_choice`.
        `no_correct_choice` marks side blocks where no first correct choice
        was made.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units. Summary lines are not jittered.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    summary_required_columns = {
        "date",
        "correct_after_first_group",
        "percent_correct_after_first_q1",
        "percent_correct_after_first_median",
        "percent_correct_after_first_q3",
    }
    point_required_columns = {
        "date",
        "correct_after_first_group",
        "percent_correct_after_first_numeric",
        "no_correct_choice",
    }
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    missing_point_columns = sorted(point_required_columns.difference(block_points_df.columns))
    if missing_summary_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_summary_columns}")
    if missing_point_columns:
        raise ValueError(f"block_points_df is missing required columns: {missing_point_columns}")

    summary_plot_df = summary_df.copy()
    point_plot_df = block_points_df.copy()
    numeric_summary_columns = [
        "percent_correct_after_first_q1",
        "percent_correct_after_first_median",
        "percent_correct_after_first_q3",
    ]
    for column in numeric_summary_columns:
        summary_plot_df[column] = pd.to_numeric(summary_plot_df[column], errors="coerce")
    point_plot_df["percent_correct_after_first_numeric"] = pd.to_numeric(
        point_plot_df["percent_correct_after_first_numeric"],
        errors="coerce",
    )
    point_plot_df["no_correct_choice"] = (
        point_plot_df["no_correct_choice"].astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])
    )

    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one session/group row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    group_styles = {
        "overall": {"color": "black", "seed_offset": 0, "alpha": 0.14},
        "left": {"color": color_dict["left_uncued"], "seed_offset": 1, "alpha": 0.18},
        "right": {"color": color_dict["right_uncued"], "seed_offset": 2, "alpha": 0.18},
    }
    no_correct_y = 1.08

    for correct_after_first_group, style in group_styles.items():
        group_summary = summary_plot_df[
            summary_plot_df["correct_after_first_group"] == correct_after_first_group
        ].copy()
        if not group_summary.empty:
            group_summary["x_position"] = group_summary["date"].map(x_by_date)
            group_summary.sort_values("x_position", inplace=True)
            spread_rows = group_summary.dropna(subset=numeric_summary_columns)
            if not spread_rows.empty:
                ax.fill_between(
                    spread_rows["x_position"].to_numpy(dtype=float),
                    spread_rows["percent_correct_after_first_q1"].to_numpy(dtype=float),
                    spread_rows["percent_correct_after_first_q3"].to_numpy(dtype=float),
                    alpha=style["alpha"],
                    linewidth=0,
                    color=style["color"],
                    label=f"{correct_after_first_group} Q1-Q3",
                )
                ax.plot(
                    spread_rows["x_position"].to_numpy(dtype=float),
                    spread_rows["percent_correct_after_first_median"].to_numpy(dtype=float),
                    "o-",
                    color=style["color"],
                    label=f"{correct_after_first_group} median",
                )

        group_points = point_plot_df[
            point_plot_df["correct_after_first_group"] == correct_after_first_group
        ].copy()
        if group_points.empty:
            continue

        group_points["x_position"] = group_points["date"].map(x_by_date)
        group_points = group_points.dropna(subset=["x_position"])
        numeric_points = group_points.dropna(subset=["percent_correct_after_first_numeric"])
        if not numeric_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    numeric_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + style["seed_offset"],
                ),
                numeric_points["percent_correct_after_first_numeric"].to_numpy(dtype=float),
                "o",
                color=style["color"],
                alpha=0.28 if correct_after_first_group == "overall" else 0.35,
                markersize=4,
                label=f"{correct_after_first_group} raw blocks",
            )

        no_correct_points = group_points[group_points["no_correct_choice"]]
        if not no_correct_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    no_correct_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + 100 + style["seed_offset"],
                ),
                np.full(no_correct_points.shape[0], no_correct_y),
                "^",
                color=style["color"],
                markerfacecolor="none",
                alpha=0.8,
                label=f"{correct_after_first_group} no correct",
            )

    ax.axhline(no_correct_y, color="gray", linestyle=":", linewidth=1)
    ax.text(
        len(dates) + 0.25,
        no_correct_y,
        "No correct",
        va="center",
        ha="left",
        fontsize=9,
        color="gray",
    )
    ax.set_ylim(-0.05, 1.18)
    ax.set_ylabel("Percent Correct After First Correct")
    ax.set_xlabel("Session date")
    ax.set_title(f"{figure_id} Block Accuracy After First Correct Across Sessions")
    ax.set_xticks(np.arange(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=60, fontsize=8)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_multisession_correct_after_first_quality(
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> None:
    """Plot cross-session accuracy after the first correct choice in each block.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Session/group summary table with shape `(n_session_groups,
        n_columns)`. Required columns are `date`,
        `correct_after_first_group`, `percent_correct_after_first_q1`,
        `percent_correct_after_first_median`, and
        `percent_correct_after_first_q3`.
    block_points_df : pd.DataFrame
        Raw side-block table with shape `(n_points, n_columns)`. Required
        columns are `date`, `correct_after_first_group`,
        `percent_correct_after_first_numeric`, and `no_correct_choice`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the plot title and output
        filename.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    None
        Saves `{figure_id}_correct-after-first-quality.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_multisession_correct_after_first_quality_on_ax(
        ax=ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )

    save_path = plot_path / f"{figure_id}_correct-after-first-quality.png"
    save_performance_figure(fig, save_path)


def plot_multisession_agent_mouse_agreement_quality_on_ax(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> plt.Axes:
    """Plot cross-session mouse-agent agreement summaries on one axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis that receives the agent agreement summary.
    summary_df : pd.DataFrame
        Session/agent summary table with shape `(n_session_agents, n_columns)`.
        Required columns are `date`, `agent`, `agreement_q1`,
        `agreement_median`, and `agreement_q3`. Agreement values are fractions
        from 0 to 1.
    block_points_df : pd.DataFrame
        Raw block-agent table with shape `(n_block_agents, n_columns)`.
        Required columns are `date`, `agent`, and `agreement`.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers, in
        categorical x-axis units.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    summary_required_columns = {
        "date",
        "agent",
        "agreement_q1",
        "agreement_median",
        "agreement_q3",
    }
    point_required_columns = {"date", "agent", "agreement"}
    missing_summary_columns = sorted(summary_required_columns.difference(summary_df.columns))
    missing_point_columns = sorted(point_required_columns.difference(block_points_df.columns))
    if missing_summary_columns:
        raise ValueError(f"summary_df is missing required columns: {missing_summary_columns}")
    if missing_point_columns:
        raise ValueError(f"block_points_df is missing required columns: {missing_point_columns}")

    summary_plot_df = summary_df.copy()
    point_plot_df = block_points_df.copy()
    numeric_summary_columns = ["agreement_q1", "agreement_median", "agreement_q3"]
    for column in numeric_summary_columns:
        summary_plot_df[column] = pd.to_numeric(summary_plot_df[column], errors="coerce")
    point_plot_df["agreement"] = pd.to_numeric(point_plot_df["agreement"], errors="coerce")
    point_plot_df = point_plot_df.dropna(subset=["agreement"])

    if summary_plot_df.empty:
        raise ValueError("summary_df must contain at least one session/agent row to plot.")

    dates = pd.unique(summary_plot_df["date"])
    x_by_date = {date: index + 1 for index, date in enumerate(dates)}
    agent_order = [
        agent for agent in DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS
        if agent in summary_plot_df["agent"].values
    ]
    agent_order.extend(
        agent for agent in pd.unique(summary_plot_df["agent"])
        if agent not in agent_order
    )
    if not agent_order:
        raise ValueError("summary_df must contain at least one agent to plot.")

    offsets = np.linspace(-0.24, 0.24, num=max(len(agent_order), 1))
    offset_by_agent = dict(zip(agent_order, offsets))
    fallback_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", all_colors)

    for agent_index, agent in enumerate(agent_order):
        color = AGENT_MOUSE_AGREEMENT_COLORS.get(agent, fallback_colors[agent_index % len(fallback_colors)])
        offset = offset_by_agent[agent]
        agent_summary = summary_plot_df[summary_plot_df["agent"] == agent].copy()
        if not agent_summary.empty:
            agent_summary["x_position"] = agent_summary["date"].map(x_by_date) + offset
            agent_summary.sort_values("x_position", inplace=True)
            spread_rows = agent_summary.dropna(subset=numeric_summary_columns)
            if not spread_rows.empty:
                x_values = spread_rows["x_position"].to_numpy(dtype=float)
                q1_values = spread_rows["agreement_q1"].to_numpy(dtype=float)
                median_values = spread_rows["agreement_median"].to_numpy(dtype=float)
                q3_values = spread_rows["agreement_q3"].to_numpy(dtype=float)
                for x_value, q1_value, q3_value in zip(x_values, q1_values, q3_values):
                    ax.plot(
                        [x_value, x_value],
                        [q1_value, q3_value],
                        color=color,
                        linewidth=2,
                    )
                ax.plot(
                    x_values,
                    median_values,
                    "o-",
                    color=color,
                    linewidth=2,
                    markersize=5,
                    label=f"{agent} median",
                )

        agent_points = point_plot_df[point_plot_df["agent"] == agent].copy()
        if agent_points.empty:
            continue

        agent_points["x_position"] = agent_points["date"].map(x_by_date) + offset
        agent_points = agent_points.dropna(subset=["x_position"])
        if not agent_points.empty:
            ax.plot(
                _jitter_x_coordinates(
                    agent_points["x_position"].to_numpy(dtype=float),
                    jitter_width=point_jitter,
                    seed=jitter_seed + agent_index,
                ),
                agent_points["agreement"].to_numpy(dtype=float),
                "o",
                color=color,
                alpha=0.35,
                markersize=4,
                label="_nolegend_",
            )

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Mouse-Agent Agreement")
    ax.set_xlabel("Session date")
    ax.set_title(f"{figure_id} Mouse-Agent Agreement Across Sessions")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.set_xticks(np.arange(1, len(dates) + 1))
    ax.set_xticklabels(dates, rotation=60, fontsize=8)
    handles, _labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(frameon=False, fontsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return ax


def plot_multisession_agent_mouse_agreement_quality(
    summary_df: pd.DataFrame,
    block_points_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
) -> Path:
    """Plot standalone cross-session mouse-agent agreement quality.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Session/agent summary table with shape `(n_session_agents, n_columns)`.
    block_points_df : pd.DataFrame
        Raw block-agent table with shape `(n_block_agents, n_columns)`.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or subject identifier used in the output filename and title.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{figure_id}_agent-mouse-agreement-quality.png`.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_multisession_agent_mouse_agreement_quality_on_ax(
        ax=ax,
        summary_df=summary_df,
        block_points_df=block_points_df,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )

    save_path = plot_path / f"{figure_id}_agent-mouse-agreement-quality.png"
    save_performance_figure(fig, save_path)
    return save_path


def plot_multisession_oracle_behavior(
    overall_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    use_dates: bool = True,
) -> None:
    """Plot oracle behavior summaries across sessions.

    Parameters
    ----------
    overall_df : pd.DataFrame
        Multisession summary dataframe with shape `(n_sessions, n_columns)`.
        Required columns are `date`, `oracle_choice_accuracy`,
        `oracle_reward_fraction`, and `oracle_reward_difference`.
        Accuracy and fraction values are unitless; reward difference is in task
        reward units.
    plot_path : pathlib.Path
        Directory where PNG figures are saved.
    figure_id : str
        Mouse or subject identifier used in plot titles and output filenames.
    use_dates : bool, default=True
        If True, label equally spaced session ticks with the `date` column.

    Returns
    -------
    None
        Saves `{figure_id}_oracle-choice-accuracy.png` and
        `{figure_id}_oracle-reward-collection.png`.
    """
    required_columns = [
        "date",
        "oracle_choice_accuracy",
        "oracle_reward_fraction",
        "oracle_reward_difference",
    ]
    missing_columns = sorted(set(required_columns).difference(overall_df.columns))
    if missing_columns:
        missing_summary = ", ".join(missing_columns)
        raise ValueError(
            "overall_df is missing oracle behavior columns; rerun session analyses "
            f"to regenerate overall_performance. Missing columns: {missing_summary}"
        )

    plot_df = overall_df.loc[:, required_columns].copy()
    plot_df = plot_df[~plot_df["date"].isna()]
    plot_df.sort_values(by="date", inplace=True)
    for column in required_columns[1:]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    accuracy_df = plot_df.dropna(subset=["oracle_choice_accuracy"])
    if accuracy_df.empty:
        raise ValueError("overall_df must contain at least one valid oracle_choice_accuracy value.")

    x_accuracy = np.arange(1, accuracy_df.shape[0] + 1)
    f_accuracy, ax_accuracy = plt.subplots(figsize=(8, 5))
    ax_accuracy.plot(
        x_accuracy,
        accuracy_df["oracle_choice_accuracy"].to_numpy(dtype=float),
        "ko-",
    )
    ax_accuracy.set_xlabel("Day", fontsize=16)
    ax_accuracy.set_ylabel("Oracle Choice Accuracy", fontsize=18)
    ax_accuracy.set_ylim(0, 1.05)
    ax_accuracy.set_title("{} Oracle Choice Accuracy".format(figure_id))
    if use_dates:
        ax_accuracy.set_xticks(x_accuracy)
        ax_accuracy.set_xticklabels(accuracy_df["date"].to_numpy(), rotation=60, fontsize=8)
    ax_accuracy.tick_params(axis="y", which="major", labelsize=12)
    ax_accuracy.spines["right"].set_visible(False)
    ax_accuracy.spines["top"].set_visible(False)
    save_performance_figure(
        f_accuracy,
        plot_path / "{}_oracle-choice-accuracy.png".format(figure_id),
    )

    reward_df = plot_df.dropna(subset=["oracle_reward_fraction", "oracle_reward_difference"])
    if reward_df.empty:
        raise ValueError(
            "overall_df must contain at least one valid oracle reward fraction "
            "and reward difference value."
        )

    x_reward = np.arange(1, reward_df.shape[0] + 1)
    f_reward, reward_axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    reward_axes[0].plot(
        x_reward,
        reward_df["oracle_reward_fraction"].to_numpy(dtype=float),
        "ko-",
    )
    reward_axes[0].axhline(1, color="gray", linestyle="--")
    reward_axes[0].set_ylabel("Actual / Oracle Reward", fontsize=14)
    reward_axes[0].set_title("{} Oracle Reward Collection".format(figure_id))

    reward_axes[1].plot(
        x_reward,
        reward_df["oracle_reward_difference"].to_numpy(dtype=float),
        "ko-",
    )
    reward_axes[1].axhline(0, color="gray", linestyle="--")
    reward_axes[1].set_xlabel("Day", fontsize=16)
    reward_axes[1].set_ylabel("Actual - Oracle Reward", fontsize=14)
    if use_dates:
        reward_axes[1].set_xticks(x_reward)
        reward_axes[1].set_xticklabels(reward_df["date"].to_numpy(), rotation=60, fontsize=8)

    for ax in reward_axes:
        ax.tick_params(axis="y", which="major", labelsize=12)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    save_performance_figure(
        f_reward,
        plot_path / "{}_oracle-reward-collection.png".format(figure_id),
    )


def plot_multisession_ideal_observer_choices_on_ax(
    ax_choice: plt.Axes,
    overall_df: pd.DataFrame,
    figure_id: str,
    use_dates: bool = True,
) -> plt.Axes:
    """Plot ideal-observer choice summaries on one axis.

    Parameters
    ----------
    ax_choice : matplotlib.axes.Axes
        Axis that receives the choice summary.
    overall_df : pd.DataFrame
        Multisession summary dataframe with shape `(n_sessions, n_columns)`.
        Required columns include history-policy choice metrics and fixed-state
        replay reward/accuracy quartiles.
    figure_id : str
        Mouse or subject identifier used in the plot title.
    use_dates : bool, default=True
        If True, label equally spaced session ticks with the `date` column.

    Returns
    -------
    matplotlib.axes.Axes
        The same axis passed in.
    """
    required_columns = [
        "date",
        "history_ideal_oracle_accuracy",
        "history_ideal_mouse_agreement",
        "fixed_replay_ideal_oracle_accuracy_q1",
        "fixed_replay_ideal_oracle_accuracy_median",
        "fixed_replay_ideal_oracle_accuracy_q3",
    ]
    missing_columns = sorted(set(required_columns).difference(overall_df.columns))
    if missing_columns:
        missing_summary = ", ".join(missing_columns)
        raise ValueError(
            "overall_df is missing ideal-observer columns; rerun session analyses "
            f"to regenerate overall_performance. Missing columns: {missing_summary}"
        )

    plot_df = overall_df.loc[:, required_columns].copy()
    plot_df = plot_df[~plot_df["date"].isna()]
    plot_df.sort_values(by="date", inplace=True)
    for column in required_columns[1:]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    choice_columns = [
        "history_ideal_oracle_accuracy",
        "history_ideal_mouse_agreement",
        "fixed_replay_ideal_oracle_accuracy_q1",
        "fixed_replay_ideal_oracle_accuracy_median",
        "fixed_replay_ideal_oracle_accuracy_q3",
    ]
    choice_df = plot_df.dropna(subset=choice_columns)
    if choice_df.empty:
        raise ValueError("overall_df must contain at least one valid ideal-observer choice row.")

    x_choice = np.arange(1, choice_df.shape[0] + 1)
    ax_choice.fill_between(
        x_choice,
        choice_df["fixed_replay_ideal_oracle_accuracy_q1"].to_numpy(dtype=float),
        choice_df["fixed_replay_ideal_oracle_accuracy_q3"].to_numpy(dtype=float),
        alpha=0.25,
        linewidth=0,
        label="Fixed replay Q1-Q3",
    )
    ax_choice.plot(
        x_choice,
        choice_df["fixed_replay_ideal_oracle_accuracy_median"].to_numpy(dtype=float),
        "ko-",
        label="Fixed replay median",
    )
    ax_choice.plot(
        x_choice,
        choice_df["history_ideal_oracle_accuracy"].to_numpy(dtype=float),
        color="tab:blue",
        marker="o",
        label="History ideal vs oracle",
    )
    ax_choice.plot(
        x_choice,
        choice_df["history_ideal_mouse_agreement"].to_numpy(dtype=float),
        color="tab:orange",
        marker="o",
        label="History ideal vs mouse",
    )
    ax_choice.set_xlabel("Day", fontsize=16)
    ax_choice.set_ylabel("Choice Fraction", fontsize=18)
    ax_choice.set_ylim(0, 1.05)
    ax_choice.set_title("{} Ideal-Observer Choices".format(figure_id))
    if use_dates:
        ax_choice.set_xticks(x_choice)
        ax_choice.set_xticklabels(choice_df["date"].to_numpy(), rotation=60, fontsize=8)
    ax_choice.tick_params(axis="y", which="major", labelsize=12)
    ax_choice.spines["right"].set_visible(False)
    ax_choice.spines["top"].set_visible(False)
    ax_choice.legend(frameon=False)
    return ax_choice


def plot_multisession_ideal_observer_behavior(
    overall_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    use_dates: bool = True,
) -> None:
    """Plot ideal-observer summaries across sessions.

    Parameters
    ----------
    overall_df : pd.DataFrame
        Multisession summary dataframe with shape `(n_sessions, n_columns)`.
        Required columns include history-policy choice metrics and fixed-state
        replay reward/accuracy quartiles.
    plot_path : pathlib.Path
        Directory where PNG figures are saved.
    figure_id : str
        Mouse or subject identifier used in plot titles and output filenames.
    use_dates : bool, default=True
        If True, label equally spaced session ticks with the `date` column.

    Returns
    -------
    None
        Saves `{figure_id}_ideal-observer-choices.png` and
        `{figure_id}_ideal-observer-reward.png`.
    """
    required_columns = [
        "date",
        "actual_reward_collected",
        "history_ideal_oracle_accuracy",
        "history_ideal_mouse_agreement",
        "history_ideal_reward_fraction",
        "history_ideal_expected_reward",
        "fixed_replay_ideal_reward_q1",
        "fixed_replay_ideal_reward_median",
        "fixed_replay_ideal_reward_q3",
        "fixed_replay_ideal_oracle_accuracy_q1",
        "fixed_replay_ideal_oracle_accuracy_median",
        "fixed_replay_ideal_oracle_accuracy_q3",
    ]
    missing_columns = sorted(set(required_columns).difference(overall_df.columns))
    if missing_columns:
        missing_summary = ", ".join(missing_columns)
        raise ValueError(
            "overall_df is missing ideal-observer columns; rerun session analyses "
            f"to regenerate overall_performance. Missing columns: {missing_summary}"
        )

    f_choice, ax_choice = plt.subplots(figsize=(8, 5))
    plot_multisession_ideal_observer_choices_on_ax(
        ax_choice=ax_choice,
        overall_df=overall_df,
        figure_id=figure_id,
        use_dates=use_dates,
    )
    save_performance_figure(
        f_choice,
        plot_path / "{}_ideal-observer-choices.png".format(figure_id),
    )

    plot_df = overall_df.loc[:, required_columns].copy()
    plot_df = plot_df[~plot_df["date"].isna()]
    plot_df.sort_values(by="date", inplace=True)
    for column in required_columns[1:]:
        plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    reward_columns = [
        "actual_reward_collected",
        "history_ideal_expected_reward",
        "history_ideal_reward_fraction",
        "fixed_replay_ideal_reward_q1",
        "fixed_replay_ideal_reward_median",
        "fixed_replay_ideal_reward_q3",
    ]
    reward_df = plot_df.dropna(subset=reward_columns)
    if reward_df.empty:
        raise ValueError("overall_df must contain at least one valid ideal-observer reward row.")

    x_reward = np.arange(1, reward_df.shape[0] + 1)
    f_reward, reward_axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    reward_axes[0].fill_between(
        x_reward,
        reward_df["fixed_replay_ideal_reward_q1"].to_numpy(dtype=float),
        reward_df["fixed_replay_ideal_reward_q3"].to_numpy(dtype=float),
        alpha=0.25,
        linewidth=0,
        label="Fixed replay Q1-Q3",
    )
    reward_axes[0].plot(
        x_reward,
        reward_df["fixed_replay_ideal_reward_median"].to_numpy(dtype=float),
        "ko-",
        label="Fixed replay median",
    )
    reward_axes[0].plot(
        x_reward,
        reward_df["history_ideal_expected_reward"].to_numpy(dtype=float),
        color="tab:blue",
        marker="o",
        label="History ideal expected",
    )
    reward_axes[0].plot(
        x_reward,
        reward_df["actual_reward_collected"].to_numpy(dtype=float),
        color="tab:orange",
        marker="o",
        label="Mouse actual",
    )
    reward_axes[0].set_ylabel("Reward", fontsize=14)
    reward_axes[0].set_title("{} Ideal-Observer Reward".format(figure_id))
    reward_axes[0].legend(frameon=False)

    reward_axes[1].plot(
        x_reward,
        reward_df["history_ideal_reward_fraction"].to_numpy(dtype=float),
        "ko-",
    )
    reward_axes[1].axhline(1, color="gray", linestyle="--")
    reward_axes[1].set_xlabel("Day", fontsize=16)
    reward_axes[1].set_ylabel("Mouse / History Ideal", fontsize=14)
    if use_dates:
        reward_axes[1].set_xticks(x_reward)
        reward_axes[1].set_xticklabels(reward_df["date"].to_numpy(), rotation=60, fontsize=8)

    for ax in reward_axes:
        ax.tick_params(axis="y", which="major", labelsize=12)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    save_performance_figure(
        f_reward,
        plot_path / "{}_ideal-observer-reward.png".format(figure_id),
    )


def plot_multisession_summary_grid(
    block_performance: pd.DataFrame,
    side_trials_to_correct_summary: pd.DataFrame,
    side_trials_to_correct_block_points: pd.DataFrame,
    correct_after_first_summary: pd.DataFrame,
    correct_after_first_block_points: pd.DataFrame,
    block_switch_summary: pd.DataFrame,
    block_switch_block_points: pd.DataFrame,
    overall_df: pd.DataFrame,
    plot_path: Path,
    figure_id: str,
    block_model_session_id: str | None = None,
    processed_data_path: Path | None = None,
    block_model_dict: dict | None = None,
    block_hmm_module=None,
    hmm_observation_plotter: Callable | None = None,
    sliding_regression_plotter: Callable | None = None,
    ttc_axis_plotter: Callable | None = None,
    correct_after_first_axis_plotter: Callable | None = None,
    block_switch_axis_plotter: Callable | None = None,
    ideal_choices_axis_plotter: Callable | None = None,
    sliding_regression_window_size: int = 10,
    sliding_regression_step_size: int = 5,
    point_jitter: float = 0.08,
    jitter_seed: int = 0,
    trial_display_cap: float = 25,
) -> Path:
    """Plot a 2x3 multisession behavior and block-HMM summary grid.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Concatenated multisession block table with shape `(n_blocks,
        n_columns)`. Required columns for HMM panels are those accepted by
        `prepare_block_lm_hmm_data`, including `trials_to_correct`,
        `prev_n_correct`, and `prev_n_rewarded`.
    side_trials_to_correct_summary : pd.DataFrame
        Side-specific trials-to-correct summary with one row per session/side.
    side_trials_to_correct_block_points : pd.DataFrame
        Raw side-specific trials-to-correct block points.
    correct_after_first_summary : pd.DataFrame
        Percent-correct-after-first summary with one row per session/group.
    correct_after_first_block_points : pd.DataFrame
        Raw percent-correct-after-first block points.
    block_switch_summary : pd.DataFrame
        Choice-switch count summary with one row per session/group.
    block_switch_block_points : pd.DataFrame
        Raw choice-switch count block points.
    overall_df : pd.DataFrame
        Mouse-level overall-performance summary with ideal-observer columns.
    plot_path : pathlib.Path
        Directory where the PNG figure is saved.
    figure_id : str
        Mouse or dataset identifier used in the output filename.
    block_model_session_id : str or None, default=None
        Saved block-model session id. For multisession analyses this is
        typically `{figure_id}_multisession`. None uses `figure_id`.
    processed_data_path : pathlib.Path or None, default=None
        Directory containing `{block_model_session_id}_block_statedict.pkl`.
        Used only when `block_model_dict` is None.
    block_model_dict : dict or None, default=None
        Optional saved block HMM model dictionary. If unavailable, HMM panels
        are drawn as placeholders.
    block_hmm_module : module or None, default=None
        Optional dependency injection object for block-HMM helpers.
    hmm_observation_plotter : Callable or None, default=None
        Optional axis-level MAP trials-to-switch plotting function.
    sliding_regression_plotter : Callable or None, default=None
        Optional axis-level sliding-regression plotting function.
    ttc_axis_plotter : Callable or None, default=None
        Optional axis-level trials-to-correct plotter.
    correct_after_first_axis_plotter : Callable or None, default=None
        Optional axis-level percent-correct-after-first plotter.
    block_switch_axis_plotter : Callable or None, default=None
        Optional axis-level choice-switch plotter.
    ideal_choices_axis_plotter : Callable or None, default=None
        Optional axis-level ideal-observer choice plotter.
    sliding_regression_window_size : int, default=10
        Number of valid blocks per sliding-regression window.
    sliding_regression_step_size : int, default=5
        Number of valid blocks between successive windows.
    point_jitter : float, default=0.08
        Maximum absolute horizontal jitter for raw block markers in
        categorical x-axis units.
    jitter_seed : int, default=0
        Seed for deterministic marker jitter.
    trial_display_cap : float, default=25
        Maximum trials-to-correct value plotted to scale in the TTC panel.
        Larger values are drawn on a shared overflow row.

    Returns
    -------
    pathlib.Path
        Saved PNG path `{figure_id}_multisession-summary-grid.png`.
    """
    if ttc_axis_plotter is None:
        ttc_axis_plotter = plot_side_trials_to_correct_quality_on_ax
    if correct_after_first_axis_plotter is None:
        correct_after_first_axis_plotter = plot_multisession_correct_after_first_quality_on_ax
    if block_switch_axis_plotter is None:
        block_switch_axis_plotter = plot_multisession_block_switches_quality_on_ax
    if ideal_choices_axis_plotter is None:
        ideal_choices_axis_plotter = plot_multisession_ideal_observer_choices_on_ax

    if block_model_dict is None:
        model_session_id = figure_id if block_model_session_id is None else block_model_session_id
        block_model_dict = _load_block_model_dict_for_summary(
            processed_data_path=processed_data_path,
            figure_id=model_session_id,
            block_hmm_module=block_hmm_module,
        )

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    hmm_available = _plot_hmm_summary_grid_panels(
        map_ax=axes[0, 0],
        regression_ax=axes[1, 0],
        block_performance=block_performance,
        block_model_dict=block_model_dict,
        block_hmm_module=block_hmm_module,
        hmm_observation_plotter=hmm_observation_plotter,
        sliding_regression_plotter=sliding_regression_plotter,
        sliding_regression_window_size=sliding_regression_window_size,
        sliding_regression_step_size=sliding_regression_step_size,
    )
    if not hmm_available:
        print("Block HMM outputs not found; multisession summary grid saved without HMM MAP and windowed regression panels.")

    ttc_axis_plotter(
        ax=axes[0, 1],
        summary_df=side_trials_to_correct_summary,
        block_points_df=side_trials_to_correct_block_points,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
        trial_display_cap=trial_display_cap,
    )
    block_switch_axis_plotter(
        ax=axes[0, 2],
        summary_df=block_switch_summary,
        block_points_df=block_switch_block_points,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )
    correct_after_first_axis_plotter(
        ax=axes[1, 1],
        summary_df=correct_after_first_summary,
        block_points_df=correct_after_first_block_points,
        figure_id=figure_id,
        point_jitter=point_jitter,
        jitter_seed=jitter_seed,
    )
    ideal_choices_axis_plotter(
        axes[1, 2],
        overall_df=overall_df,
        figure_id=figure_id,
    )

    fig.suptitle(f"{figure_id} Multisession Summary", fontsize=14)
    save_path = plot_path / f"{figure_id}_multisession-summary-grid.png"
    save_performance_figure(fig, save_path)
    return save_path


def main_single_session():
    multisession_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    session_data_home = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/CT014_20251216_latentInference/')
    sess_id_full = 'CT014_2025-12-16_153200'
    processed_data_path = session_data_home / 'processed'
    session_figure_path = session_data_home / 'figures'

    pattern = r'(\w+)_([\d\-]+)_(\d+)'
    match = re.search(pattern, sess_id_full)

    if match:
        mouse, date, timestamp = match.groups()
        print(f"Mouse id: {mouse}")  # abc123
        print(f"Date: {date}")  # YYYY-MM-DD
        print(f"Time: {timestamp}")  # HHMMSS
        sess_id_abbreviated = mouse + '_' + date
    else:
        print("Double-check the session name!")
        return

    multisession_df, block_performance, augmented_trial_df = session_analysis.load_analysis(sess_id_full,
                                                                                            session_data_folder=processed_data_path,
                                                                                            multisession_data_folder=multisession_save_path)
    overall_df = multisession_df[~multisession_df['date'].isna()]
    plot_session_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_correct_after_first_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_history_ideal_mouse_agreement(block_performance, session_figure_path, sess_id_full)
    plot_session_trials_to_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_nswitches(block_performance, session_figure_path, sess_id_full)
    # plot_multisession_correct(overall_df, mouse_plot_path, figure_id=mouse)
    # plot_multisession_trials_to_correct(overall_df, mouse_plot_path, figure_id=mouse)

    scatter_regressor = DEFAULT_TRIALS_TO_CORRECT_SCATTER_REGRESSOR
    slope = multisession_df.loc[
        multisession_df['date'] == date,
        f'{scatter_regressor}_slope',
    ].values[0]
    intercept = multisession_df.loc[
        multisession_df['date'] == date,
        f'{scatter_regressor}_intercept',
    ].values[0]
    scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
                              plot_path=session_figure_path, figure_id=sess_id_full,
                              title=sess_id_abbreviated,
                              regressor_column=scatter_regressor)
    # block_count_col = get_session_block_count_column(multisession_df)
    # plot_learning_curve(multisession_df[f'{DEFAULT_LEARNING_REGRESSOR}_slope'],
    #                     multisession_df[block_count_col], figure_id=mouse,
    #                     plot_path=multisession_save_path, dates=multisession_df['date'].values)


def main_multiple_sessions():
    multisession_save_path = Path('/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014/cross_session_analysis')
    mouse = 'CT014'
    multisession_df = pd.read_csv(multisession_save_path / (mouse + '_overall_performance.csv'), sep=',', na_filter=False)
    multisession_df.sort_values(by='date', inplace=True)
    block_count_col = get_session_block_count_column(multisession_df)
    plot_learning_curve(multisession_df[f'{DEFAULT_LEARNING_REGRESSOR}_slope'], multisession_df[block_count_col], figure_id=mouse,
                        plot_path=multisession_save_path,
                        dates=multisession_df['date'].values)


def main():
    main_single_session()
    # main_multiple_sessions()


if __name__ == '__main__':
    main()
