from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MOUSE_ROOT = Path("/home/matt/Documents/EXPERIMENTS/contextProjectData/CT014")
DEFAULT_OUTPUT_DIR = DEFAULT_MOUSE_ROOT / "cross_session_analysis"
DEFAULT_DECODER_RUN_INDEX = 0
BASE_CONDITIONS = [
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
]


def find_session_analysis_csvs(mouse_root: Path | str) -> pd.DataFrame:
    """
    Find per-session analysis CSVs under a mouse data directory.

    Parameters
    ----------
    mouse_root : Path | str
        Root directory containing one or more session folders. Each session must have a
        ``processed`` directory with ``state_decodability_analysis.csv`` and
        ``correct_rewarded_decoding_performance.csv``.

    Returns
    -------
    pd.DataFrame
        One row per discovered session with columns:
        ``session_dir``, ``processed_dir``, ``decodability_csv_path``,
        and ``decoder_performance_csv_path``.
    """

    mouse_root = Path(mouse_root)
    decodability_paths = sorted(mouse_root.rglob("state_decodability_analysis.csv"))
    session_rows: list[dict[str, Path]] = []
    for decodability_path in decodability_paths:
        processed_dir = decodability_path.parent
        decoder_performance_path = processed_dir / "correct_rewarded_decoding_performance.csv"
        if not decoder_performance_path.exists():
            continue
        session_rows.append(
            {
                "session_dir": processed_dir.parent,
                "processed_dir": processed_dir,
                "decodability_csv_path": decodability_path,
                "decoder_performance_csv_path": decoder_performance_path,
            }
        )

    return pd.DataFrame(session_rows)


def load_state_decodability_sessions(session_csv_paths: list[Path] | tuple[Path, ...]) -> pd.DataFrame:
    """
    Load per-session state decodability CSVs into one cross-session table.

    Parameters
    ----------
    session_csv_paths : list[Path] | tuple[Path, ...]
        Paths to per-session ``state_decodability_analysis.csv`` files.

    Returns
    -------
    pd.DataFrame
        One row per ``session x trial_condition`` with columns:
        ``session``, ``mouse``, ``date``, ``region``, ``trial_condition``,
        ``cv_score_before``, ``cv_score_after``, ``p_value_before``, and ``p_value_after``.
    """

    rows: list[pd.DataFrame] = []
    for csv_path in session_csv_paths:
        session_df = pd.read_csv(csv_path)
        required_columns = {
            "session_id",
            "mouse",
            "date",
            "region",
            "condition",
            "cv_score_pre",
            "cv_score_post",
            "p_value_pre",
            "p_value_post",
        }
        missing_columns = required_columns - set(session_df.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

        rows.append(
            session_df.rename(
                columns={
                    "session_id": "session",
                    "condition": "trial_condition",
                    "cv_score_pre": "cv_score_before",
                    "cv_score_post": "cv_score_after",
                    "p_value_pre": "p_value_before",
                    "p_value_post": "p_value_after",
                }
            )[
                [
                    "session",
                    "mouse",
                    "date",
                    "region",
                    "trial_condition",
                    "cv_score_before",
                    "cv_score_after",
                    "p_value_before",
                    "p_value_after",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "session",
                "mouse",
                "date",
                "region",
                "trial_condition",
                "cv_score_before",
                "cv_score_after",
                "p_value_before",
                "p_value_after",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def load_state_decoder_performance_sessions(
    session_csv_paths: list[Path] | tuple[Path, ...],
    decoder_run_index: int = DEFAULT_DECODER_RUN_INDEX,
) -> pd.DataFrame:
    """
    Load one decoder run from each per-session repeated-decoder CSV.

    Parameters
    ----------
    session_csv_paths : list[Path] | tuple[Path, ...]
        Paths to per-session ``correct_rewarded_decoding_performance.csv`` files.
    decoder_run_index : int, optional
        Decoder-run index to extract from each session table.

    Returns
    -------
    pd.DataFrame
        One row per ``session x trial_condition`` with columns:
        ``session``, ``mouse``, ``date``, ``region``, ``trial_condition``,
        ``decoder_run_index``, ``test_accuracy_before``, and ``test_accuracy_after``.
    """

    rows: list[dict[str, object]] = []
    for csv_path in session_csv_paths:
        session_df = pd.read_csv(csv_path)
        required_columns = {"session_id", "mouse", "date", "region", "decoder_run"}
        missing_columns = required_columns - set(session_df.columns)
        if missing_columns:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing_columns)}")

        selected_rows = session_df.loc[session_df["decoder_run"] == int(decoder_run_index)]
        if selected_rows.shape[0] != 1:
            raise ValueError(
                f"{csv_path} must contain exactly one row for decoder_run == {decoder_run_index}."
            )
        selected_row = selected_rows.iloc[0]
        for condition_name in BASE_CONDITIONS:
            rows.append(
                {
                    "session": selected_row["session_id"],
                    "mouse": selected_row["mouse"],
                    "date": selected_row["date"],
                    "region": selected_row["region"],
                    "trial_condition": condition_name,
                    "decoder_run_index": int(decoder_run_index),
                    "test_accuracy_before": selected_row[f"{condition_name}_test_accuracy_pre"],
                    "test_accuracy_after": selected_row[f"{condition_name}_test_accuracy_post"],
                }
            )

    return pd.DataFrame(rows)


def save_cross_session_tables(
    output_dir: Path | str,
    decodability_df: pd.DataFrame,
    decoder_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Save the cross-session decodability and decoder-performance CSVs.

    Parameters
    ----------
    output_dir : Path | str
        Directory where the cross-session CSVs should be written.
    decodability_df : pd.DataFrame
        Cross-session state decodability table with one row per session and trial condition.
    decoder_df : pd.DataFrame
        Cross-session decoder-performance table with one row per session and trial condition.

    Returns
    -------
    tuple[Path, Path]
        Saved decodability CSV path and saved decoder-performance CSV path.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    decodability_path = output_dir / "state_decodability_analysis_cross_session.csv"
    decoder_path = output_dir / "correct_rewarded_state_decoding_performance_cross_session.csv"
    decodability_df.to_csv(decodability_path, index=False)
    decoder_df.to_csv(decoder_path, index=False)
    return decodability_path, decoder_path


def _format_condition_label(trial_condition: str) -> str:
    """Convert underscore-separated trial-condition names into plot-friendly labels."""
    return trial_condition.replace("_", " ")


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
    axis.set_xlim(-0.5, 1.5)
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


def main() -> None:
    """
    Gather per-session neural-analysis CSVs, save cross-session CSVs, and make legacy-style plots.

    The default behavior searches under ``DEFAULT_MOUSE_ROOT`` for session ``processed`` folders
    containing per-session neural-analysis CSVs, writes the cross-session CSVs into
    ``DEFAULT_OUTPUT_DIR``, and saves one light-mode PNG plot per base condition for:
    CV score, CV p-value, and decoder accuracy using ``DEFAULT_DECODER_RUN_INDEX``.
    """

    session_csvs = find_session_analysis_csvs(DEFAULT_MOUSE_ROOT)
    if session_csvs.empty:
        raise FileNotFoundError(f"No per-session analysis CSVs found under {DEFAULT_MOUSE_ROOT}")

    decodability_df = load_state_decodability_sessions(
        session_csvs["decodability_csv_path"].tolist()
    )
    decoder_df = load_state_decoder_performance_sessions(
        session_csvs["decoder_performance_csv_path"].tolist(),
        decoder_run_index=DEFAULT_DECODER_RUN_INDEX,
    )
    decodability_csv_path, decoder_csv_path = save_cross_session_tables(
        DEFAULT_OUTPUT_DIR,
        decodability_df,
        decoder_df,
    )

    for trial_condition in BASE_CONDITIONS:
        score_figure, _ = plot_cross_session_decodability_scores(
            decodability_df,
            trial_condition=trial_condition,
            show=False,
        )
        score_figure.savefig(
            DEFAULT_OUTPUT_DIR / f"cv_decoder_summary_{trial_condition}.png",
            dpi=300,
        )
        plt.close(score_figure)

        pvalue_figure, _ = plot_cross_session_decodability_pvalues(
            decodability_df,
            trial_condition=trial_condition,
            show=False,
        )
        pvalue_figure.savefig(
            DEFAULT_OUTPUT_DIR / f"cv_decoder_pvalue_{trial_condition}.png",
            dpi=300,
        )
        plt.close(pvalue_figure)

        decoder_figure, _ = plot_cross_session_decoder_accuracy(
            decoder_df,
            trial_condition=trial_condition,
            show=False,
        )
        decoder_figure.savefig(
            DEFAULT_OUTPUT_DIR / f"shuffle_decoder_summary_{trial_condition}.png",
            dpi=300,
        )
        plt.close(decoder_figure)

    print(f"Saved cross-session decodability CSV: {decodability_csv_path}")
    print(f"Saved cross-session decoder CSV: {decoder_csv_path}")


if __name__ == "__main__":
    main()
