import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
try:
    from . import session_analysis
except ImportError:
    from behavior_analysis import session_analysis
from typing import Iterable
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


def scatter_trials_to_correct(block_performance: pd.DataFrame, slope: float, intercept: float, plot_path: Path,
                              figure_id: str, title=None, point_jitter: float = 0.08,
                              jitter_seed: int = 0):
    """Plot block trials-to-correct against prior consecutive rewards.

    Parameters
    ----------
    block_performance : pd.DataFrame
        Blockwise performance table, shape `(n_blocks, n_columns)`. Required
        columns are `block_type`, `trials_to_correct`, `prev_n_correct`, and
        `prev_consecutive_rewards`. Count columns are interpreted as integer
        block/trial counts.
    slope : float
        Regression slope in trials-to-correct per prior consecutive reward.
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

    Returns
    -------
    None
        Saves `{figure_id}_scatter_trials-to-correct.png` and closes the
        figure.
    """
    try:
        slope = float(slope)
        intercept = float(intercept)
    except (TypeError, ValueError) as exc:
        raise ValueError("scatter_trials_to_correct slope and intercept must be numeric.") from exc

    f1, ax1 = plt.subplots(figsize=(7, 5))
    ix_valid = (block_performance['trials_to_correct'] != 'None') & (block_performance['prev_n_correct'] != 'None')
    block_performance = block_performance[ix_valid]

    block_type = block_performance['block_type'].to_numpy()
    trials_to_correct = block_performance['trials_to_correct'].astype(int).to_numpy()
    prev_consecutive_rewards = block_performance['prev_consecutive_rewards'].astype(int).to_numpy()
    jittered_rewards, jittered_trials_to_correct = _jitter_integer_scatter_points(
        prev_consecutive_rewards,
        trials_to_correct,
        jitter_width=point_jitter,
        seed=jitter_seed,
    )
    
    # use this block for separate colors 
    for b in block_types:
        ix = np.where(block_type == b)[0]
        if len(ix) > 0:
            ax1.plot(jittered_rewards[ix], jittered_trials_to_correct[ix],
                     'o', color=color_dict[b], label=b.replace('_', ' '))

    # use this block for one color
    # ax1.plot(prev_consecutive_rewards, trials_to_correct, 'o')

    x = np.array([0, np.amax(prev_consecutive_rewards)])
    ax1.plot(x, slope*x + intercept, 'k--')

    plt.ylabel('Trials to Correct')
    plt.xlabel('Consecutive Rewards')
    if title is not None:
        plt.title('{} Trials to Switch vs Rewards'.format(title).replace('_', ' '))
    else:
        plt.title('{} Trials to Switch vs Rewards'.format(figure_id).replace('_', ' '))

    # plt.legend(fancybox=False)
    plt.legend(frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    save_path = plot_path / '{}_scatter_trials-to-correct.png'.format(figure_id)
    save_performance_figure(f1, save_path)


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
    """For a single session, plot the blockwise number of switches. Will approximate confusion."""
    f, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(block_performance))
    for b in block_types:
        ix = np.where(block_performance['block_type'] == b)[0]
        if len(ix) > 0:
            ax.plot(x[ix], block_performance['n_switches'][ix], 'o', color=color_dict[b], label=b)

    plt.ylabel('Num Switches in block')
    plt.xlabel('Block')
    plt.title('{} Block Confusion'.format(sess_ID))
    plt.legend(fancybox=False)

    save_path = plot_path / '{}_block_nswitches.png'.format(sess_ID)
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
    f_choice, ax_choice = plt.subplots(figsize=(8, 5))
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
    save_performance_figure(
        f_choice,
        plot_path / "{}_ideal-observer-choices.png".format(figure_id),
    )

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
    plot_session_trials_to_correct(block_performance, session_figure_path, sess_id_full)
    plot_session_nswitches(block_performance, session_figure_path, sess_id_full)
    # plot_multisession_correct(overall_df, mouse_plot_path, figure_id=mouse)
    # plot_multisession_trials_to_correct(overall_df, mouse_plot_path, figure_id=mouse)

    slope = multisession_df.loc[
        multisession_df['date'] == date,
        f'{DEFAULT_LEARNING_REGRESSOR}_slope',
    ].values[0]
    intercept = multisession_df.loc[
        multisession_df['date'] == date,
        f'{DEFAULT_LEARNING_REGRESSOR}_intercept',
    ].values[0]
    scatter_trials_to_correct(block_performance, slope=slope, intercept=intercept,
                              plot_path=session_figure_path, figure_id=sess_id_full,
                              title=sess_id_abbreviated)
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
