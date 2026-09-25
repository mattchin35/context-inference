from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


COLOR_DICT = {
    'right_cued': 'cyan',
    'left_cued': 'darkgreen',
    'right_uncued': 'plum',
    'left_uncued': 'indigo',
    'right': 'cyan',
    'left': 'darkgreen',
    0: 'cyan',
    1: 'darkkhaki',
}
STATE_LABELS = {
    'right': 'right',
    'left': 'left',
    0: 'right',
    1: 'left',
}
VALUE_COLUMN_LABELS = {
    'agent_relative_value': 'Combined policy value',
    'agent_hmm_value': 'HMM value',
    'agent_doubt_value': 'Relative doubt value',
    'simple_probe_persistence_value': 'Probe + persistence',
    'expectancy_persistence_doubt_value': 'Expectancy + persistence + doubt',
}
VALUE_COLUMN_STYLES = {
    'agent_relative_value': {'color': 'white', 'linewidth': 1.8},
    'agent_hmm_value': {'color': 'deepskyblue', 'linewidth': 1.5},
    'agent_doubt_value': {'color': 'gold', 'linewidth': 1.5},
    'simple_probe_persistence_value': {'color': '#e377c2', 'linewidth': 1.6},
    'expectancy_persistence_doubt_value': {'color': '#4c78a8', 'linewidth': 1.8},
}
STRATEGY_COLORS = {
    'HMM': '#5DA5DA',
    'F-Qlearning': '#F17CB0',
    'HMM_reward_decay_relative_doubt': '#60BD68',
}
REQUIRED_COLUMNS = ('state', 'action', 'reward')
THEME_STYLES = {
    "light": {
        "figure_facecolor": "#ffffff",
        "axes_facecolor": "#ffffff",
        "text_color": "#111111",
        "spine_color": "#111111",
        "zero_line_color": "#7a7a7a",
        "combined_policy_color": "#3a3a3a",
        "hmm_value_color": "deepskyblue",
        "doubt_value_color": "goldenrod",
        "action_color": "#2f2f2f",
        "reward_color": "#4a4a4a",
        "reward_edgecolor": "#1f1f1f",
        "strategy_divider_color": "#4a4a4a",
        "legend_facecolor": "#ffffff",
        "legend_edgecolor": "#cccccc",
    },
    "dark": {
        "figure_facecolor": "#111111",
        "axes_facecolor": "#111111",
        "text_color": "#f5f5f5",
        "spine_color": "#f5f5f5",
        "zero_line_color": "#7a7a7a",
        "combined_policy_color": "#ffffff",
        "hmm_value_color": "deepskyblue",
        "doubt_value_color": "gold",
        "action_color": "ivory",
        "reward_color": "#ffffff",
        "reward_edgecolor": "#dcdcdc",
        "strategy_divider_color": "#f5f5f5",
        "legend_facecolor": "#111111",
        "legend_edgecolor": "#666666",
    },
}


def validate_run_plot_columns(run_df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in run_df.columns]
    if missing_cols:
        raise ValueError(f"run dataframe missing required columns: {missing_cols}")


def resolve_value_columns(
    run_df: pd.DataFrame,
    value_columns: Optional[Sequence[str]] = None,
) -> list[str]:
    if value_columns is None:
        candidate_columns = [
            'agent_relative_value',
            'agent_hmm_value',
            'agent_doubt_value',
        ]
        resolved = []
        for column in candidate_columns:
            if column not in run_df.columns:
                continue
            values = pd.to_numeric(run_df[column], errors='coerce')
            if values.notna().any():
                resolved.append(column)
    else:
        resolved = list(value_columns)
        missing_cols = [col for col in resolved if col not in run_df.columns]
        if missing_cols:
            raise ValueError(f"requested plot columns missing from dataframe: {missing_cols}")

    if not resolved:
        raise ValueError("no plottable value columns found in run dataframe.")
    return resolved


def _state_color_and_label(state) -> tuple[str, str]:
    if state in COLOR_DICT:
        return COLOR_DICT[state], STATE_LABELS.get(state, str(state))

    state_str = str(state).lower()
    if state_str in COLOR_DICT:
        return COLOR_DICT[state_str], STATE_LABELS.get(state_str, state_str)

    try:
        state_int = int(state)
    except (TypeError, ValueError):
        return 'gray', str(state)

    if state_int in COLOR_DICT:
        return COLOR_DICT[state_int], STATE_LABELS.get(state_int, str(state_int))
    return 'gray', str(state)


def _get_state_bins(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if states.size == 0:
        return np.array([0]), np.array([], dtype=object)
    change_ix = states[:-1] != states[1:]
    state_changes = np.arange(1, states.size)[change_ix]
    bins = np.unique(np.concatenate(([0], state_changes, [states.size - 1])))
    return bins, states[bins]


def _get_segment_bins(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.size == 0:
        return np.array([0]), np.array([], dtype=object)
    change_ix = labels[:-1] != labels[1:]
    change_points = np.arange(1, labels.size)[change_ix]
    bins = np.concatenate(([0], change_points, [labels.size]))
    segment_labels = labels[bins[:-1]]
    return bins, segment_labels


def _normalize_save_path(save_path) -> Path:
    path = Path(save_path)
    if path.suffix == '':
        path = path.with_suffix('.png')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_theme_styles(theme: str) -> dict[str, str]:
    if theme not in THEME_STYLES:
        raise ValueError(f"theme must be one of {tuple(THEME_STYLES)}")
    return THEME_STYLES[theme]


def plot_run_dataframe(
    run_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
    value_columns: Optional[Sequence[str]] = None,
    show_action_probability: bool = True,
    show_legend: bool = True,
    xtick_interval: Optional[int] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
    theme: str = "light",
    ax=None,
):
    """Plot trialwise agent values with mouse or agent actions.

    Parameters
    ----------
    run_df : pd.DataFrame
        Run dataframe with shape `(n_trials, n_columns)`. Required columns are
        `state`, `action`, and `reward`; value columns are selected by
        `value_columns`.
    title : str or None
        Optional axis title.
    value_columns : sequence of str or None
        Numeric columns to plot against trial index. Defaults to available
        agent value columns.
    show_action_probability : bool, default=True
        If True and `agent_action_dist` exists, plot pre-update P(left) scaled
        to the action axis.
    show_legend : bool, default=True
        If True, draw the legend. If False, all labels remain on artists but no
        legend box is shown.
    xtick_interval : int or None, default=None
        If provided, use ticks every `xtick_interval` trials. If None, ticks
        are placed at state-transition bins.
    save_path : pathlib.Path or None
        Optional PNG path. A missing suffix is treated as `.png`.
    show : bool, default=False
        If True, display the figure.
    theme : {"light", "dark"}, default="light"
        Figure color theme.
    ax : matplotlib axis or None
        Optional existing axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        Figure and axis containing the plot.
    """
    validate_run_plot_columns(run_df)
    if xtick_interval is not None:
        try:
            xtick_interval = int(xtick_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("xtick_interval must be a positive integer or None.") from exc
        if xtick_interval <= 0:
            raise ValueError("xtick_interval must be a positive integer or None.")
    resolved_value_columns = resolve_value_columns(run_df, value_columns=value_columns)
    theme_styles = resolve_theme_styles(theme)

    states = run_df['state'].to_numpy()
    actions = pd.to_numeric(run_df['action'], errors='coerce').to_numpy()
    rewards = pd.to_numeric(run_df['reward'], errors='coerce').to_numpy()
    x = np.arange(run_df.shape[0])

    bins, bin_types = _get_state_bins(states)
    created_figure = ax is None
    if created_figure:
        fig = Figure(figsize=(12, 5))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    fig.patch.set_facecolor(theme_styles["figure_facecolor"])
    ax.set_facecolor(theme_styles["axes_facecolor"])

    for column in resolved_value_columns:
        values = pd.to_numeric(run_df[column], errors='coerce').to_numpy()
        style = dict(VALUE_COLUMN_STYLES.get(column, {}))
        if column == "agent_relative_value":
            style["color"] = theme_styles["combined_policy_color"]
        elif column == "agent_hmm_value":
            style["color"] = theme_styles["hmm_value_color"]
        elif column == "agent_doubt_value":
            style["color"] = theme_styles["doubt_value_color"]
        ax.plot(x, values, label=VALUE_COLUMN_LABELS.get(column, column), **style)

    if show_action_probability and 'agent_action_dist' in run_df.columns:
        p_left = pd.to_numeric(run_df['agent_action_dist'], errors='coerce').to_numpy()
        if np.isfinite(p_left).any():
            ax.plot(
                x,
                p_left * 2 - 1,
                label='P(left)',
                color='coral',
                linestyle='--',
                linewidth=1.2,
                alpha=0.9,
            )

    valid_action_ix = np.isfinite(actions)
    signed_actions = actions[valid_action_ix] * 2 - 1
    ax.scatter(
        x[valid_action_ix],
        signed_actions,
        s=8,
        c=theme_styles["action_color"],
        label='actions',
        zorder=3,
    )

    rewarded_ix = valid_action_ix & (rewards > 0)
    ax.scatter(
        x[rewarded_ix],
        actions[rewarded_ix] * 2 - 1,
        s=22,
        c=theme_styles["reward_color"],
        edgecolors=theme_styles["reward_edgecolor"],
        linewidths=0.5,
        label='rewards',
        zorder=4,
    )

    ax.axhline(0, color=theme_styles["zero_line_color"], linewidth=1, alpha=0.6, linestyle='--')

    seen_labels = set()
    for i in range(max(bins.size - 1, 0)):
        color, label = _state_color_and_label(bin_types[i])
        span_label = label if label not in seen_labels else None
        ax.axvspan(bins[i], bins[i + 1], color=color, alpha=0.15, label=span_label)
        seen_labels.add(label)

    if 'active_strategy' in run_df.columns:
        strategy_labels = run_df['active_strategy'].astype(str).to_numpy()
        strategy_bins, strategy_names = _get_segment_bins(strategy_labels)
        y_min, y_max = ax.get_ylim()
        text_y = y_max - 0.06 * (y_max - y_min if y_max != y_min else 1.0)
        for i, strategy_name in enumerate(strategy_names):
            start = strategy_bins[i]
            end = strategy_bins[i + 1]
            if i > 0:
                ax.axvline(
                    start,
                    color=theme_styles["strategy_divider_color"],
                    linewidth=1.0,
                    linestyle=':',
                    alpha=0.9,
                )
            ax.text(
                (start + end - 1) / 2 if end > start else start,
                text_y,
                strategy_name,
                color=STRATEGY_COLORS.get(strategy_name, 'white'),
                fontsize=9,
                ha='center',
                va='top',
                alpha=0.95,
            )

    ax.set_xlabel('Trial')
    ax.set_ylabel('Model value / action')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Right', '0', 'Left'])
    ax.tick_params(colors=theme_styles["text_color"])
    ax.xaxis.label.set_color(theme_styles["text_color"])
    ax.yaxis.label.set_color(theme_styles["text_color"])
    ax.title.set_color(theme_styles["text_color"])
    for spine in ax.spines.values():
        spine.set_color(theme_styles["spine_color"])
    if xtick_interval is not None:
        tick_stop = int(x[-1]) if x.size > 0 else 0
        ax.set_xticks(np.arange(0, tick_stop + 1, xtick_interval))
        ax.set_xlim([0, tick_stop])
    elif bins.size > 0:
        ax.set_xticks(bins)
        ax.set_xlim([0, x[-1] if x.size > 0 else 0])
    if title is not None:
        ax.set_title(title)
    if show_legend:
        legend = ax.legend(fancybox=False)
        if legend is not None:
            legend.get_frame().set_facecolor(theme_styles["legend_facecolor"])
            legend.get_frame().set_edgecolor(theme_styles["legend_edgecolor"])
            for text in legend.get_texts():
                text.set_color(theme_styles["text_color"])
    fig.tight_layout()

    if save_path is not None:
        normalized_path = _normalize_save_path(save_path)
        fig.savefig(normalized_path, dpi=300)

    if show:
        fig.show()

    return fig, ax
