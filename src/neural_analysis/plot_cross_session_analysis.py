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
from src.neural_analysis.population.plotting import (
    BEFORE_AFTER_XLIM,
    _compute_session_mean_decoder_table,
    _format_condition_label,
    _format_session_date_label,
    _plot_paired_before_after,
    append_plot_title_suffix,
    plot_cross_session_decodability_pvalues,
    plot_cross_session_decodability_scores,
    plot_cross_session_decoder_accuracy,
    plot_cross_session_decoder_session_means,
    plot_cross_session_decoder_superplot,
)


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
