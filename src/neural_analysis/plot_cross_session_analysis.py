"""Compatibility facade retaining cross-session plots and direct dispatch."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.population.cross_session import (
    BASE_CONDITIONS,
    DEFAULT_DECODER_RUN_INDEX,
    SUPPORTED_ANALYSIS_REGIONS,
    normalize_region_name_for_filename,
    make_region_analysis_filename,
    make_analysis_filename_tag,
    make_analysis_title_suffix,
    find_session_analysis_csvs,
    load_state_decodability_sessions,
    load_state_decoder_performance_sessions,
    save_cross_session_tables,
    filter_table_by_date_selection,
    select_cross_session_date_range,
    make_output_filename,
)


DEFAULT_MOUSE_ROOT = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014")
DEFAULT_OUTPUT_DIR = DEFAULT_MOUSE_ROOT / "cross_session_analysis"
REGION_NAME = "V1"
DATE_SELECTION_START = "2025-12-16" #None
DATE_SELECTION_END = "2025-12-23" #None
DATE_SELECTION_INCLUDE = None
DATE_SELECTION_LABEL = "12-16 to 12-23"
DATE_SELECTION_FILENAME_TAG = "2025-12-16_to_2025-12-23"
BEFORE_AFTER_XLIM = (-0.2, 1.2)


def append_plot_title_suffix(axis: plt.Axes, title_suffix: str | None) -> None:
    """
    Append a date-selection label to an existing plot title.

    Parameters
    ----------
    axis : plt.Axes
        Matplotlib axis whose title should be updated.
    title_suffix : str | None
        Optional text appended on a new title line.

    Returns
    -------
    None
        The axis title is modified in place when ``title_suffix`` is supplied.
    """

    if title_suffix is None:
        return
    axis.set_title(f"{axis.get_title()}\n{title_suffix}")


def _format_condition_label(trial_condition: str) -> str:
    """Convert underscore-separated trial-condition names into plot-friendly labels."""
    return trial_condition.replace("_", " ")


def _format_session_date_label(session_row: pd.Series) -> str:
    """
    Format a session row into a date-only label for plot legends.

    Parameters
    ----------
    session_row : pd.Series
        One row containing at least a ``date`` field.

    Returns
    -------
    str
        Session date label for the legend.
    """

    if "date" not in session_row or pd.isna(session_row["date"]):
        raise ValueError("Session-mean decoder plots require a non-null 'date' column for legend labels.")
    return str(session_row["date"])


def _plot_paired_before_after(
    values_df: pd.DataFrame,
    trial_condition: str,
    before_column: str,
    after_column: str,
    ylabel: str,
    title_prefix: str,
    *,
    y_limits: tuple[float, float] = (0.0, 1.0),
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot paired before/after values across sessions in the legacy style.

    Parameters
    ----------
    values_df : pd.DataFrame
        Cross-session table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    before_column : str
        Column containing before-choice values.
    after_column : str
        Column containing after-choice values.
    ylabel : str
        Y-axis label for the metric being plotted.
    title_prefix : str
        Prefix used in the figure title before the condition label.
    y_limits : tuple[float, float], optional
        Lower and upper y-axis limits for the plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the paired before/after plot.
    """

    condition_df = values_df.loc[values_df["trial_condition"] == trial_condition].copy()
    if condition_df.empty:
        raise ValueError(f"No rows found for trial_condition={trial_condition!r}.")

    figure, axis = plt.subplots(figsize=(8, 6))
    before_values = condition_df[before_column].to_numpy(dtype=float)
    after_values = condition_df[after_column].to_numpy(dtype=float)

    for _, row in condition_df.iterrows():
        axis.plot(
            [0, 1],
            [float(row[before_column]), float(row[after_column])],
            color="gray",
            alpha=0.5,
        )

    axis.scatter(np.zeros(before_values.shape[0]), before_values, color="C0", alpha=0.5)
    axis.scatter(np.ones(after_values.shape[0]), after_values, color="C1", alpha=0.5)
    axis.plot(
        [0, 1],
        [float(np.nanmean(before_values)), float(np.nanmean(after_values))],
        color="k",
        linestyle="--",
        linewidth=2,
    )
    axis.set_xlim(*BEFORE_AFTER_XLIM)
    axis.set_ylim(*y_limits)
    axis.set_xticks([0, 1], ["Before\nchoice", "After\nchoice"])
    axis.set_ylabel(ylabel)
    axis.set_title(f"{title_prefix}, {_format_condition_label(trial_condition)} trials")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis


def plot_cross_session_decodability_scores(
    decodability_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cross-session before/after CV scores for one trial condition.

    Parameters
    ----------
    decodability_df : pd.DataFrame
        Cross-session decodability table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the CV-score plot.
    """

    return _plot_paired_before_after(
        decodability_df,
        trial_condition=trial_condition,
        before_column="cv_score_before",
        after_column="cv_score_after",
        ylabel="CV score",
        title_prefix="Decoder performance",
        show=show,
    )


def plot_cross_session_decodability_pvalues(
    decodability_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cross-session before/after decodability p-values for one trial condition.

    Parameters
    ----------
    decodability_df : pd.DataFrame
        Cross-session decodability table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the p-value plot.
    """

    return _plot_paired_before_after(
        decodability_df,
        trial_condition=trial_condition,
        before_column="p_value_before",
        after_column="p_value_after",
        ylabel="P value",
        title_prefix="Decoder performance",
        show=show,
    )


def plot_cross_session_decoder_accuracy(
    decoder_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot cross-session before/after decoder test accuracy for one trial condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session and condition.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the test-accuracy plot.
    """

    return _plot_paired_before_after(
        decoder_df,
        trial_condition=trial_condition,
        before_column="test_accuracy_before",
        after_column="test_accuracy_after",
        ylabel="Test accuracy",
        title_prefix="Decoder performance",
        show=show,
    )


def _compute_session_mean_decoder_table(
    decoder_df: pd.DataFrame,
    trial_condition: str,
) -> pd.DataFrame:
    """
    Compute per-session decoder means for one trial condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session, condition, and decoder run.
    trial_condition : str
        Condition name whose per-session decoder means should be computed.

    Returns
    -------
    pd.DataFrame
        One row per session with columns ``session``, ``date``, ``test_accuracy_before``,
        and ``test_accuracy_after`` representing the mean across decoder runs.
    """

    condition_df = decoder_df.loc[decoder_df["trial_condition"] == trial_condition].copy()
    if condition_df.empty:
        raise ValueError(f"No rows found for trial_condition={trial_condition!r}.")
    if "date" not in condition_df.columns:
        condition_df["date"] = condition_df["session"].astype(str)

    return (
        condition_df.groupby("session", as_index=False).agg(
            date=("date", "first"),
            test_accuracy_before=("test_accuracy_before", "mean"),
            test_accuracy_after=("test_accuracy_after", "mean"),
        )
        .sort_values("session")
    )


def plot_cross_session_decoder_superplot(
    decoder_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot all decoder runs, per-session means, and the mean of session means for one condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session, condition, and decoder run.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the mixed-model style decoder super-plot.
    """

    condition_df = decoder_df.loc[decoder_df["trial_condition"] == trial_condition].copy()
    if condition_df.empty:
        raise ValueError(f"No rows found for trial_condition={trial_condition!r}.")

    figure, axis = plt.subplots(figsize=(8, 6))
    session_names = sorted(condition_df["session"].unique().tolist())
    color_values = np.linspace(0.0, 1.0, max(len(session_names), 2))
    session_colors = {
        session_name: plt.cm.tab10(color_values[min(ix, 9)])
        for ix, session_name in enumerate(session_names)
    }

    for session_name in session_names:
        session_df = condition_df.loc[condition_df["session"] == session_name]
        color = session_colors[session_name]
        for _, decoder_row in session_df.iterrows():
            axis.plot(
                [0, 1],
                [
                    float(decoder_row["test_accuracy_before"]),
                    float(decoder_row["test_accuracy_after"]),
                ],
                color=color,
                alpha=0.25,
                linewidth=1.0,
            )

    session_mean_df = _compute_session_mean_decoder_table(decoder_df, trial_condition=trial_condition)
    for _, session_row in session_mean_df.iterrows():
        axis.plot(
            [0, 1],
            [
                float(session_row["test_accuracy_before"]),
                float(session_row["test_accuracy_after"]),
            ],
            color=session_colors[str(session_row["session"])],
            alpha=0.9,
            linewidth=2.5,
        )

    overall_mean_before = float(session_mean_df["test_accuracy_before"].mean())
    overall_mean_after = float(session_mean_df["test_accuracy_after"].mean())
    axis.plot(
        [0, 1],
        [overall_mean_before, overall_mean_after],
        color="k",
        linestyle="--",
        linewidth=2,
    )
    axis.set_xlim(*BEFORE_AFTER_XLIM)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0, 1], ["Before\nchoice", "After\nchoice"])
    axis.set_ylabel("Test accuracy")
    axis.set_title(f"Decoder performance, {_format_condition_label(trial_condition)} trials")
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis


def plot_cross_session_decoder_session_means(
    decoder_df: pd.DataFrame,
    trial_condition: str,
    *,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot per-session decoder means and the mean of session means for one condition.

    Parameters
    ----------
    decoder_df : pd.DataFrame
        Cross-session decoder table with one row per session, condition, and decoder run.
    trial_condition : str
        Condition name to plot.
    show : bool, optional
        Whether to call ``plt.show()``.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Figure and axis containing the session-mean decoder plot with a date legend.
    """

    session_mean_df = _compute_session_mean_decoder_table(decoder_df, trial_condition=trial_condition)

    figure, axis = plt.subplots(figsize=(8, 6))
    session_names = sorted(session_mean_df["session"].unique().tolist())
    color_values = np.linspace(0.0, 1.0, max(len(session_names), 2))
    session_colors = {
        session_name: plt.cm.tab10(color_values[min(ix, 9)])
        for ix, session_name in enumerate(session_names)
    }
    for _, session_row in session_mean_df.iterrows():
        axis.plot(
            [0, 1],
            [
                float(session_row["test_accuracy_before"]),
                float(session_row["test_accuracy_after"]),
            ],
            color=session_colors[str(session_row["session"])],
            alpha=0.9,
            linewidth=2.0,
            label=_format_session_date_label(session_row),
        )

    overall_mean_before = float(session_mean_df["test_accuracy_before"].mean())
    overall_mean_after = float(session_mean_df["test_accuracy_after"].mean())
    axis.plot(
        [0, 1],
        [overall_mean_before, overall_mean_after],
        color="k",
        linestyle="--",
        linewidth=2,
        label="Overall mean",
    )
    axis.set_xlim(*BEFORE_AFTER_XLIM)
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks([0, 1], ["Before\nchoice", "After\nchoice"])
    axis.set_ylabel("Test accuracy")
    axis.set_title(f"Decoder performance, {_format_condition_label(trial_condition)} trials")
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    figure.tight_layout()
    if show:
        plt.show()
    return figure, axis


def main() -> None:
    """
    Gather per-session neural-analysis CSVs, save cross-session CSVs, and make legacy-style plots.

    The default behavior searches under ``DEFAULT_MOUSE_ROOT`` for session ``processed`` folders
    containing per-session neural-analysis CSVs, writes the cross-session CSVs into
    ``DEFAULT_OUTPUT_DIR``, and saves one light-mode PNG plot per base condition for:
    CV score, CV p-value, and decoder accuracy using ``DEFAULT_DECODER_RUN_INDEX``.
    """

    session_csvs = find_session_analysis_csvs(DEFAULT_MOUSE_ROOT, region_name=REGION_NAME)
    if session_csvs.empty:
        raise FileNotFoundError(f"No per-session analysis CSVs found under {DEFAULT_MOUSE_ROOT}")

    decodability_df = load_state_decodability_sessions(
        session_csvs["decodability_csv_path"].tolist()
    )
    decoder_df = load_state_decoder_performance_sessions(
        session_csvs["decoder_performance_csv_path"].tolist(),
        decoder_run_index=DEFAULT_DECODER_RUN_INDEX,
    )
    decoder_all_runs_df = load_state_decoder_performance_sessions(
        session_csvs["decoder_performance_csv_path"].tolist(),
        decoder_run_index=None,
    )
    decodability_df, decoder_df, decoder_all_runs_df, selected_dates_df = select_cross_session_date_range(
        decodability_df,
        decoder_df,
        decoder_all_runs_df,
        start_date=DATE_SELECTION_START,
        end_date=DATE_SELECTION_END,
        include_dates=DATE_SELECTION_INCLUDE,
    )
    decodability_csv_path, decoder_csv_path = save_cross_session_tables(
        DEFAULT_OUTPUT_DIR,
        decodability_df,
        decoder_df,
        region_name=REGION_NAME,
    )
    selected_dates_csv_path = DEFAULT_OUTPUT_DIR / make_region_analysis_filename(
        REGION_NAME,
        "selected_cross_session_dates.csv",
    )
    selected_dates_df.to_csv(selected_dates_csv_path, index=False)
    plot_filename_tag = make_analysis_filename_tag(REGION_NAME, DATE_SELECTION_FILENAME_TAG)
    plot_title_suffix = make_analysis_title_suffix(REGION_NAME, DATE_SELECTION_LABEL)

    for trial_condition in BASE_CONDITIONS:
        score_figure, score_axis = plot_cross_session_decodability_scores(
            decodability_df,
            trial_condition=trial_condition,
            show=False,
        )
        append_plot_title_suffix(score_axis, plot_title_suffix)
        score_figure.savefig(
            DEFAULT_OUTPUT_DIR / make_output_filename(
                f"cv_decoder_summary_{trial_condition}",
                plot_filename_tag,
            ),
            dpi=300,
        )
        plt.close(score_figure)

        pvalue_figure, pvalue_axis = plot_cross_session_decodability_pvalues(
            decodability_df,
            trial_condition=trial_condition,
            show=False,
        )
        append_plot_title_suffix(pvalue_axis, plot_title_suffix)
        pvalue_figure.savefig(
            DEFAULT_OUTPUT_DIR / make_output_filename(
                f"cv_decoder_pvalue_{trial_condition}",
                plot_filename_tag,
            ),
            dpi=300,
        )
        plt.close(pvalue_figure)

        decoder_figure, decoder_axis = plot_cross_session_decoder_accuracy(
            decoder_df,
            trial_condition=trial_condition,
            show=False,
        )
        append_plot_title_suffix(decoder_axis, plot_title_suffix)
        decoder_figure.savefig(
            DEFAULT_OUTPUT_DIR / make_output_filename(
                f"shuffle_decoder_summary_{trial_condition}",
                plot_filename_tag,
            ),
            dpi=300,
        )
        plt.close(decoder_figure)

        decoder_superplot_figure, decoder_superplot_axis = plot_cross_session_decoder_superplot(
            decoder_all_runs_df,
            trial_condition=trial_condition,
            show=False,
        )
        append_plot_title_suffix(decoder_superplot_axis, plot_title_suffix)
        decoder_superplot_figure.savefig(
            DEFAULT_OUTPUT_DIR / make_output_filename(
                f"shuffle_decoder_superplot_{trial_condition}",
                plot_filename_tag,
            ),
            dpi=300,
        )
        plt.close(decoder_superplot_figure)

        decoder_session_mean_figure, decoder_session_mean_axis = plot_cross_session_decoder_session_means(
            decoder_all_runs_df,
            trial_condition=trial_condition,
            show=False,
        )
        append_plot_title_suffix(decoder_session_mean_axis, plot_title_suffix)
        decoder_session_mean_figure.savefig(
            DEFAULT_OUTPUT_DIR / make_output_filename(
                f"shuffle_decoder_session_mean_{trial_condition}",
                plot_filename_tag,
            ),
            dpi=300,
        )
        plt.close(decoder_session_mean_figure)

    print(f"Saved cross-session decodability CSV: {decodability_csv_path}")
    print(f"Saved cross-session decoder CSV: {decoder_csv_path}")
    print(f"Saved selected cross-session dates CSV: {selected_dates_csv_path}")


if __name__ == "__main__":
    main()
