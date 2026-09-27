from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.neural_analysis import plot_cross_session_analysis


def _write_session_csvs(
    session_root: Path,
    *,
    session_id: str,
    mouse: str,
    date: str,
    region: str = "HPC",
    filename_region: str | None = None,
    decoder_runs: tuple[int, ...] = (0, 1),
) -> tuple[Path, Path]:
    """Create synthetic per-session decodability and decoder CSVs in a processed folder."""
    processed_dir = session_root / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    decodability_df = pd.DataFrame(
        [
            {
                "session_id": session_id,
                "mouse": mouse,
                "date": date,
                "region": region,
                "condition": "correct_rewarded",
                "status_pre": "ok",
                "reason_pre": "",
                "n_trials_pre": 12,
                "n_classes_pre": 2,
                "cv_score_pre": 0.70,
                "p_value_pre": 0.02,
                "score_mean_pre": 0.51,
                "score_std_pre": 0.04,
                "status_post": "ok",
                "reason_post": "",
                "n_trials_post": 12,
                "n_classes_post": 2,
                "cv_score_post": 0.83,
                "p_value_post": 0.01,
                "score_mean_post": 0.52,
                "score_std_post": 0.05,
            },
            {
                "session_id": session_id,
                "mouse": mouse,
                "date": date,
                "region": region,
                "condition": "incorrect",
                "status_pre": "ok",
                "reason_pre": "",
                "n_trials_pre": 7,
                "n_classes_pre": 2,
                "cv_score_pre": 0.58,
                "p_value_pre": 0.20,
                "score_mean_pre": 0.50,
                "score_std_pre": 0.03,
                "status_post": "ok",
                "reason_post": "",
                "n_trials_post": 7,
                "n_classes_post": 2,
                "cv_score_post": 0.62,
                "p_value_post": 0.12,
                "score_mean_post": 0.49,
                "score_std_post": 0.03,
            },
            {
                "session_id": session_id,
                "mouse": mouse,
                "date": date,
                "region": region,
                "condition": "omission",
                "status_pre": "ok",
                "reason_pre": "",
                "n_trials_pre": 5,
                "n_classes_pre": 2,
                "cv_score_pre": 0.61,
                "p_value_pre": 0.10,
                "score_mean_pre": 0.50,
                "score_std_pre": 0.03,
                "status_post": "ok",
                "reason_post": "",
                "n_trials_post": 5,
                "n_classes_post": 2,
                "cv_score_post": 0.66,
                "p_value_post": 0.08,
                "score_mean_post": 0.51,
                "score_std_post": 0.04,
            },
            {
                "session_id": session_id,
                "mouse": mouse,
                "date": date,
                "region": region,
                "condition": "switch",
                "status_pre": "ok",
                "reason_pre": "",
                "n_trials_pre": 4,
                "n_classes_pre": 2,
                "cv_score_pre": 0.57,
                "p_value_pre": 0.18,
                "score_mean_pre": 0.49,
                "score_std_pre": 0.03,
                "status_post": "ok",
                "reason_post": "",
                "n_trials_post": 4,
                "n_classes_post": 2,
                "cv_score_post": 0.68,
                "p_value_post": 0.05,
                "score_mean_post": 0.50,
                "score_std_post": 0.04,
            },
            {
                "session_id": session_id,
                "mouse": mouse,
                "date": date,
                "region": region,
                "condition": "stay",
                "status_pre": "ok",
                "reason_pre": "",
                "n_trials_pre": 6,
                "n_classes_pre": 2,
                "cv_score_pre": 0.60,
                "p_value_pre": 0.15,
                "score_mean_pre": 0.50,
                "score_std_pre": 0.03,
                "status_post": "ok",
                "reason_post": "",
                "n_trials_post": 6,
                "n_classes_post": 2,
                "cv_score_post": 0.64,
                "p_value_post": 0.11,
                "score_mean_post": 0.50,
                "score_std_post": 0.04,
            },
        ]
    )
    filename_prefix = "" if filename_region is None else f"{filename_region}_"
    decodability_path = processed_dir / f"{filename_prefix}state_decodability_analysis.csv"
    decodability_df.to_csv(decodability_path, index=False)

    decoder_rows: list[dict[str, object]] = []
    for decoder_run in decoder_runs:
        decoder_rows.append(
            {
                "session_id": session_id,
                "mouse": mouse,
                "date": date,
                "region": region,
                "decoder_run": decoder_run,
                "training_condition": "correct_rewarded",
                "training_window": "post_choice",
                "train_accuracy": 0.90 + 0.01 * decoder_run,
                "heldout_test_accuracy": 0.82 + 0.01 * decoder_run,
                "shuffle_pvalue": 0.01,
                "shuffle_accuracy_mean": 0.50,
                "shuffle_accuracy_std": 0.05,
                "correct_rewarded_status_pre": "ok",
                "correct_rewarded_reason_pre": "",
                "correct_rewarded_test_accuracy_pre": 0.72 + 0.01 * decoder_run,
                "correct_rewarded_n_trials_pre": 12,
                "correct_rewarded_n_classes_pre": 2,
                "correct_rewarded_status_post": "ok",
                "correct_rewarded_reason_post": "",
                "correct_rewarded_test_accuracy_post": 0.84 + 0.01 * decoder_run,
                "correct_rewarded_n_trials_post": 12,
                "correct_rewarded_n_classes_post": 2,
                "incorrect_status_pre": "ok",
                "incorrect_reason_pre": "",
                "incorrect_test_accuracy_pre": 0.54 + 0.01 * decoder_run,
                "incorrect_n_trials_pre": 7,
                "incorrect_n_classes_pre": 2,
                "incorrect_status_post": "ok",
                "incorrect_reason_post": "",
                "incorrect_test_accuracy_post": 0.58 + 0.01 * decoder_run,
                "incorrect_n_trials_post": 7,
                "incorrect_n_classes_post": 2,
                "omission_status_pre": "ok",
                "omission_reason_pre": "",
                "omission_test_accuracy_pre": 0.56 + 0.01 * decoder_run,
                "omission_n_trials_pre": 5,
                "omission_n_classes_pre": 2,
                "omission_status_post": "ok",
                "omission_reason_post": "",
                "omission_test_accuracy_post": 0.60 + 0.01 * decoder_run,
                "omission_n_trials_post": 5,
                "omission_n_classes_post": 2,
                "switch_status_pre": "ok",
                "switch_reason_pre": "",
                "switch_test_accuracy_pre": 0.55 + 0.01 * decoder_run,
                "switch_n_trials_pre": 4,
                "switch_n_classes_pre": 2,
                "switch_status_post": "ok",
                "switch_reason_post": "",
                "switch_test_accuracy_post": 0.61 + 0.01 * decoder_run,
                "switch_n_trials_post": 4,
                "switch_n_classes_post": 2,
                "stay_status_pre": "ok",
                "stay_reason_pre": "",
                "stay_test_accuracy_pre": 0.57 + 0.01 * decoder_run,
                "stay_n_trials_pre": 6,
                "stay_n_classes_pre": 2,
                "stay_status_post": "ok",
                "stay_reason_post": "",
                "stay_test_accuracy_post": 0.59 + 0.01 * decoder_run,
                "stay_n_trials_post": 6,
                "stay_n_classes_post": 2,
            }
        )

    decoder_path = processed_dir / f"{filename_prefix}correct_rewarded_decoding_performance.csv"
    pd.DataFrame(decoder_rows).to_csv(decoder_path, index=False)
    return decodability_path, decoder_path


def test_find_session_analysis_csvs_finds_expected_processed_csvs(tmp_path: Path):
    _write_session_csvs(
        tmp_path / "session_a",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
    )
    _write_session_csvs(
        tmp_path / "session_b",
        session_id="CT014_2025-12-17_153200",
        mouse="CT014",
        date="2025-12-17",
    )

    session_csvs = plot_cross_session_analysis.find_session_analysis_csvs(tmp_path)

    assert session_csvs.shape[0] == 2
    assert {
        "session_dir",
        "decodability_csv_path",
        "decoder_performance_csv_path",
    }.issubset(session_csvs.columns)


def test_find_session_analysis_csvs_uses_region_specific_filenames(tmp_path: Path):
    _write_session_csvs(
        tmp_path / "session_hpc",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
        region="HPC",
        filename_region="HPC",
    )
    _write_session_csvs(
        tmp_path / "session_v1",
        session_id="CT014_2025-12-17_153200",
        mouse="CT014",
        date="2025-12-17",
        region="V1",
        filename_region="V1",
    )

    session_csvs = plot_cross_session_analysis.find_session_analysis_csvs(
        tmp_path,
        region_name="HPC",
    )

    assert session_csvs.shape[0] == 1
    assert session_csvs.iloc[0]["decodability_csv_path"].name == "HPC_state_decodability_analysis.csv"
    assert session_csvs.iloc[0]["decoder_performance_csv_path"].name == "HPC_correct_rewarded_decoding_performance.csv"


def test_load_state_decodability_sessions_reshapes_to_one_row_per_session_condition(tmp_path: Path):
    decodability_path, _ = _write_session_csvs(
        tmp_path / "session_a",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
    )

    cross_session_df = plot_cross_session_analysis.load_state_decodability_sessions([decodability_path])

    assert cross_session_df.shape[0] == 5
    assert {
        "session",
        "mouse",
        "date",
        "region",
        "trial_condition",
        "cv_score_before",
        "cv_score_after",
        "p_value_before",
        "p_value_after",
    }.issubset(cross_session_df.columns)
    correct_row = cross_session_df.loc[cross_session_df["trial_condition"] == "correct_rewarded"].iloc[0]
    assert correct_row["cv_score_before"] == pytest.approx(0.70)
    assert correct_row["cv_score_after"] == pytest.approx(0.83)


def test_load_state_decoder_performance_sessions_uses_requested_decoder_run(tmp_path: Path):
    _, decoder_path = _write_session_csvs(
        tmp_path / "session_a",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
        decoder_runs=(0, 1),
    )

    cross_session_df = plot_cross_session_analysis.load_state_decoder_performance_sessions(
        [decoder_path],
        decoder_run_index=0,
    )

    assert cross_session_df.shape[0] == 5
    assert {
        "session",
        "trial_condition",
        "decoder_run_index",
        "test_accuracy_before",
        "test_accuracy_after",
    }.issubset(cross_session_df.columns)
    assert set(cross_session_df["decoder_run_index"].tolist()) == {0}
    correct_row = cross_session_df.loc[cross_session_df["trial_condition"] == "correct_rewarded"].iloc[0]
    assert correct_row["test_accuracy_before"] == pytest.approx(0.72)
    assert correct_row["test_accuracy_after"] == pytest.approx(0.84)


def test_load_state_decoder_performance_sessions_loads_all_decoder_runs_when_requested(tmp_path: Path):
    _, decoder_path = _write_session_csvs(
        tmp_path / "session_a",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
        decoder_runs=(0, 1),
    )

    cross_session_df = plot_cross_session_analysis.load_state_decoder_performance_sessions(
        [decoder_path],
        decoder_run_index=None,
    )

    assert cross_session_df.shape[0] == 10
    assert set(cross_session_df["decoder_run_index"].tolist()) == {0, 1}


def test_save_cross_session_tables_writes_expected_filenames(tmp_path: Path):
    decodability_df = pd.DataFrame({"session": ["a"], "trial_condition": ["correct_rewarded"]})
    decoder_df = pd.DataFrame({"session": ["a"], "trial_condition": ["correct_rewarded"]})

    decodability_path, decoder_path = plot_cross_session_analysis.save_cross_session_tables(
        tmp_path,
        decodability_df,
        decoder_df,
    )

    assert decodability_path.name == "state_decodability_analysis_cross_session.csv"
    assert decoder_path.name == "correct_rewarded_state_decoding_performance_cross_session.csv"
    assert decodability_path.exists()
    assert decoder_path.exists()


def test_save_cross_session_tables_writes_region_specific_filenames(tmp_path: Path):
    decodability_df = pd.DataFrame({"session": ["a"], "trial_condition": ["correct_rewarded"]})
    decoder_df = pd.DataFrame({"session": ["a"], "trial_condition": ["correct_rewarded"]})

    decodability_path, decoder_path = plot_cross_session_analysis.save_cross_session_tables(
        tmp_path,
        decodability_df,
        decoder_df,
        region_name="V1",
    )

    assert decodability_path.name == "V1_state_decodability_analysis_cross_session.csv"
    assert decoder_path.name == "V1_correct_rewarded_state_decoding_performance_cross_session.csv"
    assert decodability_path.exists()
    assert decoder_path.exists()


def test_make_analysis_filename_tag_combines_region_and_date_tags():
    assert plot_cross_session_analysis.make_analysis_filename_tag("hpc", None) == "HPC"
    assert (
        plot_cross_session_analysis.make_analysis_filename_tag("hpc", "2025-12-02_to_2025-12-16")
        == "HPC_2025-12-02_to_2025-12-16"
    )


def test_make_region_analysis_filename_preserves_legacy_when_region_is_none():
    assert (
        plot_cross_session_analysis.make_region_analysis_filename(
            region_name=None,
            base_filename="state_decodability_analysis.csv",
        )
        == "state_decodability_analysis.csv"
    )
    assert (
        plot_cross_session_analysis.make_region_analysis_filename(
            region_name="pfc",
            base_filename="state_decodability_analysis.csv",
        )
        == "PFC_state_decodability_analysis.csv"
    )


def test_filter_table_by_date_selection_uses_inclusive_range():
    table_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b", "sess_c"],
            "date": ["2025-12-02", "2025-12-05", "2025-12-16"],
            "trial_condition": ["correct_rewarded", "correct_rewarded", "correct_rewarded"],
        }
    )

    filtered = plot_cross_session_analysis.filter_table_by_date_selection(
        table_df,
        start_date="2025-12-05",
        end_date="2025-12-16",
    )

    assert filtered["session"].tolist() == ["sess_b", "sess_c"]


def test_select_cross_session_date_range_rejects_missing_requested_date():
    decodability_df = pd.DataFrame(
        {
            "session": ["sess_a"],
            "date": ["2025-12-02"],
            "trial_condition": ["correct_rewarded"],
        }
    )
    decoder_df = decodability_df.copy()
    decoder_all_runs_df = decodability_df.copy()

    with pytest.raises(ValueError, match="2025-12-05"):
        plot_cross_session_analysis.select_cross_session_date_range(
            decodability_df,
            decoder_df,
            decoder_all_runs_df,
            include_dates=("2025-12-05",),
        )


def test_select_cross_session_date_range_filters_tables_and_summarizes_dates():
    decodability_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b", "sess_c"],
            "date": ["2025-12-02", "2025-12-05", "2025-12-16"],
            "trial_condition": ["correct_rewarded", "correct_rewarded", "correct_rewarded"],
        }
    )
    decoder_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b", "sess_c"],
            "date": ["2025-12-02", "2025-12-05", "2025-12-16"],
            "trial_condition": ["correct_rewarded", "correct_rewarded", "correct_rewarded"],
            "decoder_run_index": [0, 0, 0],
        }
    )
    decoder_all_runs_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_a", "sess_b", "sess_b", "sess_c", "sess_c"],
            "date": [
                "2025-12-02",
                "2025-12-02",
                "2025-12-05",
                "2025-12-05",
                "2025-12-16",
                "2025-12-16",
            ],
            "trial_condition": ["correct_rewarded"] * 6,
            "decoder_run_index": [0, 1, 0, 1, 0, 1],
        }
    )

    filtered_decodability, filtered_decoder, filtered_all_runs, selected_dates = (
        plot_cross_session_analysis.select_cross_session_date_range(
            decodability_df,
            decoder_df,
            decoder_all_runs_df,
            start_date="2025-12-05",
            end_date="2025-12-16",
        )
    )

    assert filtered_decodability["session"].tolist() == ["sess_b", "sess_c"]
    assert filtered_decoder["session"].tolist() == ["sess_b", "sess_c"]
    assert filtered_all_runs["session"].tolist() == ["sess_b", "sess_b", "sess_c", "sess_c"]
    assert selected_dates["date"].tolist() == ["2025-12-05", "2025-12-16"]
    assert selected_dates["session"].tolist() == ["sess_b", "sess_c"]
    assert selected_dates["n_decoder_all_run_rows"].tolist() == [2, 2]


def test_select_cross_session_date_range_rejects_duplicate_sessions_per_date():
    decodability_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b"],
            "date": ["2025-12-02", "2025-12-02"],
            "trial_condition": ["correct_rewarded", "correct_rewarded"],
        }
    )
    decoder_df = decodability_df.copy()
    decoder_all_runs_df = decodability_df.copy()

    with pytest.raises(ValueError, match="multiple sessions"):
        plot_cross_session_analysis.select_cross_session_date_range(
            decodability_df,
            decoder_df,
            decoder_all_runs_df,
            include_dates=("2025-12-02",),
        )


def test_make_output_filename_adds_optional_date_tag():
    assert (
        plot_cross_session_analysis.make_output_filename(
            "cv_decoder_summary_correct_rewarded",
            None,
        )
        == "cv_decoder_summary_correct_rewarded.png"
    )
    assert (
        plot_cross_session_analysis.make_output_filename(
            "cv_decoder_summary_correct_rewarded",
            "2025-12-02_to_2025-12-16",
        )
        == "cv_decoder_summary_correct_rewarded_2025-12-02_to_2025-12-16.png"
    )


def test_append_plot_title_suffix_appends_only_when_supplied():
    figure, axis = plt.subplots()
    axis.set_title("Base title")

    plot_cross_session_analysis.append_plot_title_suffix(axis, None)
    assert axis.get_title() == "Base title"

    plot_cross_session_analysis.append_plot_title_suffix(axis, "2025-12-02 to 2025-12-16")
    assert axis.get_title() == "Base title\n2025-12-02 to 2025-12-16"
    plt.close(figure)


def test_plot_cross_session_decodability_scores_returns_expected_line_count():
    cross_session_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b"],
            "trial_condition": ["correct_rewarded", "correct_rewarded"],
            "cv_score_before": [0.70, 0.75],
            "cv_score_after": [0.82, 0.86],
        }
    )

    figure, axis = plot_cross_session_analysis.plot_cross_session_decodability_scores(
        cross_session_df,
        trial_condition="correct_rewarded",
        show=False,
    )

    assert len(axis.lines) == 3
    assert axis.get_ylabel() == "CV score"
    assert axis.get_xlim() == pytest.approx((-0.2, 1.2))
    plt.close(figure)


def test_plot_cross_session_decodability_pvalues_returns_expected_line_count():
    cross_session_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b"],
            "trial_condition": ["switch", "switch"],
            "p_value_before": [0.12, 0.15],
            "p_value_after": [0.04, 0.06],
        }
    )

    figure, axis = plot_cross_session_analysis.plot_cross_session_decodability_pvalues(
        cross_session_df,
        trial_condition="switch",
        show=False,
    )

    assert len(axis.lines) == 3
    assert axis.get_ylabel() == "P value"
    assert axis.get_xlim() == pytest.approx((-0.2, 1.2))
    plt.close(figure)


def test_plot_cross_session_decoder_accuracy_returns_expected_line_count():
    cross_session_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_b"],
            "trial_condition": ["stay", "stay"],
            "decoder_run_index": [0, 0],
            "test_accuracy_before": [0.57, 0.60],
            "test_accuracy_after": [0.63, 0.67],
        }
    )

    figure, axis = plot_cross_session_analysis.plot_cross_session_decoder_accuracy(
        cross_session_df,
        trial_condition="stay",
        show=False,
    )

    assert len(axis.lines) == 3
    assert axis.get_ylabel() == "Test accuracy"
    assert axis.get_xlim() == pytest.approx((-0.2, 1.2))
    plt.close(figure)


def test_plot_cross_session_decoder_superplot_returns_expected_line_count():
    cross_session_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_a", "sess_b", "sess_b"],
            "trial_condition": ["stay", "stay", "stay", "stay"],
            "decoder_run_index": [0, 1, 0, 1],
            "test_accuracy_before": [0.57, 0.59, 0.60, 0.62],
            "test_accuracy_after": [0.63, 0.65, 0.67, 0.69],
        }
    )

    figure, axis = plot_cross_session_analysis.plot_cross_session_decoder_superplot(
        cross_session_df,
        trial_condition="stay",
        show=False,
    )

    assert len(axis.lines) == 7
    assert axis.get_ylabel() == "Test accuracy"
    assert axis.get_xlim() == pytest.approx((-0.2, 1.2))
    plt.close(figure)


def test_plot_cross_session_decoder_session_means_returns_expected_line_count():
    cross_session_df = pd.DataFrame(
        {
            "session": ["sess_a", "sess_a", "sess_b", "sess_b"],
            "date": ["2025-12-16", "2025-12-16", "2025-12-17", "2025-12-17"],
            "trial_condition": ["switch", "switch", "switch", "switch"],
            "decoder_run_index": [0, 1, 0, 1],
            "test_accuracy_before": [0.55, 0.57, 0.58, 0.60],
            "test_accuracy_after": [0.61, 0.63, 0.64, 0.66],
        }
    )

    figure, axis = plot_cross_session_analysis.plot_cross_session_decoder_session_means(
        cross_session_df,
        trial_condition="switch",
        show=False,
    )

    assert len(axis.lines) == 3
    assert axis.get_ylabel() == "Test accuracy"
    assert axis.get_xlim() == pytest.approx((-0.2, 1.2))
    legend = axis.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels == ["2025-12-16", "2025-12-17", "Overall mean"]
    plt.close(figure)


def test_main_collects_cross_session_csvs_and_saves_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_root = tmp_path / "mouse_data"
    _write_session_csvs(
        session_root / "session_a",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
        filename_region="HPC",
    )
    output_dir = tmp_path / "cross_session_analysis"

    monkeypatch.setattr(plot_cross_session_analysis, "DEFAULT_MOUSE_ROOT", session_root)
    monkeypatch.setattr(plot_cross_session_analysis, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(plot_cross_session_analysis, "REGION_NAME", "HPC")
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_START", None)
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_END", None)
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_INCLUDE", None)
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_LABEL", None)
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_FILENAME_TAG", None)

    plot_cross_session_analysis.main()

    assert (output_dir / "HPC_state_decodability_analysis_cross_session.csv").exists()
    assert (output_dir / "HPC_correct_rewarded_state_decoding_performance_cross_session.csv").exists()
    assert (output_dir / "HPC_selected_cross_session_dates.csv").exists()
    assert (output_dir / "cv_decoder_summary_correct_rewarded_HPC.png").exists()
    assert (output_dir / "cv_decoder_pvalue_switch_HPC.png").exists()
    assert (output_dir / "shuffle_decoder_summary_stay_HPC.png").exists()
    assert (output_dir / "shuffle_decoder_superplot_stay_HPC.png").exists()
    assert (output_dir / "shuffle_decoder_session_mean_switch_HPC.png").exists()


def test_main_applies_date_selection_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    session_root = tmp_path / "mouse_data"
    _write_session_csvs(
        session_root / "session_a",
        session_id="CT014_2025-12-02_151940",
        mouse="CT014",
        date="2025-12-02",
        filename_region="HPC",
    )
    _write_session_csvs(
        session_root / "session_b",
        session_id="CT014_2025-12-16_153200",
        mouse="CT014",
        date="2025-12-16",
        filename_region="HPC",
    )
    output_dir = tmp_path / "cross_session_analysis"

    monkeypatch.setattr(plot_cross_session_analysis, "DEFAULT_MOUSE_ROOT", session_root)
    monkeypatch.setattr(plot_cross_session_analysis, "DEFAULT_OUTPUT_DIR", output_dir)
    monkeypatch.setattr(plot_cross_session_analysis, "REGION_NAME", "HPC")
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_START", "2025-12-16")
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_END", "2025-12-16")
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_INCLUDE", None)
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_LABEL", "test date label")
    monkeypatch.setattr(plot_cross_session_analysis, "DATE_SELECTION_FILENAME_TAG", "test-date")

    plot_cross_session_analysis.main()

    decodability_df = pd.read_csv(output_dir / "HPC_state_decodability_analysis_cross_session.csv")
    selected_dates_df = pd.read_csv(output_dir / "HPC_selected_cross_session_dates.csv")
    assert sorted(decodability_df["date"].unique().tolist()) == ["2025-12-16"]
    assert selected_dates_df["date"].tolist() == ["2025-12-16"]
    assert (output_dir / "cv_decoder_summary_correct_rewarded_HPC_test-date.png").exists()
