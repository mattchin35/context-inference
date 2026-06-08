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


def scatter_trials_to_correct(block_performance: pd.DataFrame, slope: float, intercept: float, plot_path: Path,
                              figure_id: str, title=None):
    try:
        slope = float(slope)
        intercept = float(intercept)
    except (TypeError, ValueError) as exc:
        raise ValueError("scatter_trials_to_correct slope and intercept must be numeric.") from exc

    f1, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(block_performance))
    ix_valid = (block_performance['trials_to_correct'] != 'None') & (block_performance['prev_n_correct'] != 'None')
    block_performance = block_performance[ix_valid]

    block_type = block_performance['block_type'].to_numpy()
    trials_to_correct = block_performance['trials_to_correct'].astype(int).to_numpy()
    prev_consecutive_rewards = block_performance['prev_consecutive_rewards'].astype(int).to_numpy()

    # for b in block_types:
    #     ix = np.where(block_type == b)[0]
    #     if len(ix) > 0:
    #         ax1.plot(prev_consecutive_rewards[ix], trials_to_correct[ix],
    #                  'o', color=color_dict[b], label=b.replace('_', ' '))

    ax1.plot(prev_consecutive_rewards, trials_to_correct, 'o')

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
