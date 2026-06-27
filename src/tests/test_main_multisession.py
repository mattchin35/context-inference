import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def load_main_module():
    """Import `main.py` with lightweight stubs for heavy dependencies.

    Returns
    -------
    module
        Imported main module. Stubbed dependencies expose only the attributes
        needed by the multisession orchestration tests.
    """
    behavior_analysis_pkg = ModuleType("behavior_analysis")
    behavior_analysis_pkg.performance_plots = ModuleType("behavior_analysis.performance_plots")
    behavior_analysis_pkg.raster_plots = ModuleType("behavior_analysis.raster_plots")
    behavior_analysis_pkg.session_analysis = ModuleType("behavior_analysis.session_analysis")
    behavior_analysis_pkg.simulate_priors = ModuleType("behavior_analysis.simulate_priors")
    behavior_analysis_pkg.plot_model_values = ModuleType("behavior_analysis.plot_model_values")
    behavior_analysis_pkg.block_state_space_modeling = ModuleType(
        "behavior_analysis.block_state_space_modeling"
    )
    behavior_analysis_pkg.trial_state_space_modeling = ModuleType(
        "behavior_analysis.trial_state_space_modeling"
    )
    behavior_analysis_pkg.gather_trial_features = ModuleType(
        "behavior_analysis.gather_trial_features"
    )
    sys.modules["behavior_analysis"] = behavior_analysis_pkg
    sys.modules["behavior_analysis.performance_plots"] = behavior_analysis_pkg.performance_plots
    sys.modules["behavior_analysis.raster_plots"] = behavior_analysis_pkg.raster_plots
    sys.modules["behavior_analysis.session_analysis"] = behavior_analysis_pkg.session_analysis
    sys.modules["behavior_analysis.simulate_priors"] = behavior_analysis_pkg.simulate_priors
    sys.modules["behavior_analysis.plot_model_values"] = behavior_analysis_pkg.plot_model_values
    sys.modules[
        "behavior_analysis.block_state_space_modeling"
    ] = behavior_analysis_pkg.block_state_space_modeling
    sys.modules[
        "behavior_analysis.trial_state_space_modeling"
    ] = behavior_analysis_pkg.trial_state_space_modeling
    sys.modules[
        "behavior_analysis.gather_trial_features"
    ] = behavior_analysis_pkg.gather_trial_features

    mouse_pkg = ModuleType("mouse_behavior_preprocessing")
    mouse_pkg.process_behavior_log = ModuleType("mouse_behavior_preprocessing.process_behavior_log")
    sys.modules["mouse_behavior_preprocessing"] = mouse_pkg
    sys.modules["mouse_behavior_preprocessing.process_behavior_log"] = mouse_pkg.process_behavior_log

    seaborn_stub = ModuleType("seaborn")
    seaborn_stub.set_style = lambda *_args, **_kwargs: None
    seaborn_stub.set_context = lambda *_args, **_kwargs: None
    seaborn_stub.xkcd_palette = lambda colors: colors
    sys.modules["seaborn"] = seaborn_stub

    utilplot_stub = ModuleType("src.state_space_modeling.utilplot")
    sys.modules["src.state_space_modeling.utilplot"] = utilplot_stub
    state_space_modeling_pkg = ModuleType("src.state_space_modeling")
    state_space_modeling_pkg.utilplot = utilplot_stub
    sys.modules["src.state_space_modeling"] = state_space_modeling_pkg

    main_path = Path(__file__).resolve().parents[1] / "main.py"
    module_spec = importlib.util.spec_from_file_location("main_module_for_multisession_test", main_path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def write_saved_session(
    data_root: Path,
    mouse: str,
    date: str,
    timestamp: str,
    block_df: pd.DataFrame,
    trial_df: pd.DataFrame,
    directory_suffix: str = "_latentInference",
) -> Path:
    """Create saved CSVs matching the real CT014 directory layout.

    Parameters
    ----------
    data_root : pathlib.Path
        Mouse-level data directory. The session folder is created beneath it.
    mouse : str
        Mouse identifier used in the session id.
    date : str
        Session date formatted as `YYYY-MM-DD`.
    timestamp : str
        Session timestamp formatted as `HHMMSS`.
    block_df : pandas.DataFrame
        Block table with shape `(n_blocks, n_columns)`.
    trial_df : pandas.DataFrame
        Trial table with shape `(n_trials, n_columns)`.
    directory_suffix : str, default="_latentInference"
        Suffix appended to `mouse_YYYYMMDD` for the session folder name.

    Returns
    -------
    pathlib.Path
        Path to the created session data home directory.
    """
    date_compact = date.replace("-", "")
    sess_id_full = f"{mouse}_{date}_{timestamp}"
    session_data_home = data_root / f"{mouse}_{date_compact}{directory_suffix}"
    processed_path = session_data_home / "processed"
    figure_path = session_data_home / "figures"
    processed_path.mkdir(parents=True)
    figure_path.mkdir()
    block_df.to_csv(processed_path / f"{sess_id_full}_block_performance.csv", index=False)
    trial_df.to_csv(processed_path / f"{sess_id_full}_augmented_trials.csv", index=False)
    return session_data_home


def test_find_saved_session_by_date_resolves_one_saved_session(tmp_path):
    """Date resolution should find one saved session and fill Session metadata."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    write_saved_session(
        data_root=data_root,
        mouse="CT014",
        date="2025-12-05",
        timestamp="165240",
        block_df=pd.DataFrame({"block_ix": [0], "trials_to_correct": [1]}),
        trial_df=pd.DataFrame({"cur_block": [0], "action": [1]}),
    )

    session = main_module.find_saved_session_by_date(
        mouse="CT014",
        date="2025-12-05",
        session_data_root=data_root,
        multi_session_save_path=cross_session_path,
    )

    assert session.sess_id_full == "CT014_2025-12-05_165240"
    assert session.sess_id_abbreviated == "CT014_2025-12-05"
    assert session.processed_data_path.name == "processed"
    assert session.figure_path.name == "figures"
    assert session.multi_session_save_path == cross_session_path


def test_find_saved_session_by_date_errors_when_date_is_missing(tmp_path):
    """Missing saved CSVs for a requested date should raise a clear error."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="2025-12-05"):
        main_module.find_saved_session_by_date(
            mouse="CT014",
            date="2025-12-05",
            session_data_root=data_root,
            multi_session_save_path=cross_session_path,
        )


def test_concatenate_saved_sessions_offsets_block_and_trial_block_ids(tmp_path):
    """Continuous concat should offset block ids while preserving source ids."""
    main_module = load_main_module()
    first_session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    second_session = SimpleNamespace(sess_id_full="CT014_2025-12-16_153200", date="2025-12-16")
    first = main_module.SavedSessionAnalysis(
        session=first_session,
        block_performance=pd.DataFrame({"block_ix": [0, 1], "value": [10, 11]}),
        augmented_trial_df=pd.DataFrame(
            {"cur_trial": [0, 1, 2], "cur_block": [0, 1, 1], "action": [1, 0, 0]}
        ),
    )
    second = main_module.SavedSessionAnalysis(
        session=second_session,
        block_performance=pd.DataFrame({"block_ix": [0, 1, 2], "value": [20, 21, 22]}),
        augmented_trial_df=pd.DataFrame(
            {"cur_trial": [0, 1, 2, 3], "cur_block": [0, 0, 1, 2], "action": [0, 1, 1, 0]}
        ),
    )

    concatenated = main_module.concatenate_saved_sessions([first, second])

    assert concatenated.block_performance["block_ix"].tolist() == [0, 1, 2, 3, 4]
    assert concatenated.augmented_trial_df["cur_block"].tolist() == [0, 1, 1, 2, 2, 3, 4]
    assert concatenated.block_session_lengths.tolist() == [2, 3]
    assert concatenated.trial_session_lengths.tolist() == [3, 4]
    assert concatenated.block_performance["source_block_ix"].tolist() == [0, 1, 0, 1, 2]
    assert concatenated.augmented_trial_df["source_cur_block"].tolist() == [0, 1, 1, 0, 0, 1, 2]
    assert concatenated.block_performance["source_session_id"].tolist() == [
        "CT014_2025-12-05_165240",
        "CT014_2025-12-05_165240",
        "CT014_2025-12-16_153200",
        "CT014_2025-12-16_153200",
        "CT014_2025-12-16_153200",
    ]
    assert concatenated.augmented_trial_df["source_date"].tolist() == [
        "2025-12-05",
        "2025-12-05",
        "2025-12-05",
        "2025-12-16",
        "2025-12-16",
        "2025-12-16",
        "2025-12-16",
    ]


def test_concatenate_saved_sessions_uses_raw_trial_block_ids_as_source_blocks(tmp_path):
    """Saved block rows may be renumbered while trial rows keep raw block ids."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-23_163505", date="2025-12-23")
    saved = main_module.SavedSessionAnalysis(
        session=session,
        block_performance=pd.DataFrame({"block_ix": [0, 1, 2], "value": [10, 11, 12]}),
        augmented_trial_df=pd.DataFrame(
            {
                "cur_trial": [0, 1, 2, 3, 4],
                "cur_block": [4, 5, 5, 6, 6],
                "action": [1, 0, 1, 0, 1],
            }
        ),
    )

    concatenated = main_module.concatenate_saved_sessions([saved])

    assert concatenated.block_performance["source_block_ix"].tolist() == [4, 5, 6]
    assert concatenated.block_performance["block_ix"].tolist() == [0, 1, 2]
    assert concatenated.augmented_trial_df["source_cur_block"].tolist() == [4, 5, 5, 6, 6]
    assert concatenated.augmented_trial_df["cur_block"].tolist() == [0, 1, 1, 2, 2]


def test_concatenate_saved_sessions_rejects_block_count_mismatch(tmp_path):
    """Raw trial block count must match saved block rows before renumbering."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-23_163505", date="2025-12-23")
    saved = main_module.SavedSessionAnalysis(
        session=session,
        block_performance=pd.DataFrame({"block_ix": [0, 1], "value": [10, 11]}),
        augmented_trial_df=pd.DataFrame(
            {
                "cur_trial": [0, 1, 2],
                "cur_block": [4, 5, 6],
                "action": [1, 0, 1],
            }
        ),
    )

    with pytest.raises(ValueError, match="3 trial blocks but 2 block rows"):
        main_module.concatenate_saved_sessions([saved])


def test_prepare_learning_curve_data_drops_none_slopes_and_returns_numeric_arrays(tmp_path):
    """Learning-curve preparation should drop unavailable slopes before plotting."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "cross_session_analysis"
    cross_session_path.mkdir()
    multisession_df = pd.DataFrame(
        {
            "date": ["2025-12-16", "2025-12-05", "2025-12-23", "2025-12-08"],
            "prev_consecutive_rewards_slope": ["None", "0.1", "0.3", "0.2"],
            "n_blocks": ["None", "4", "6", "5"],
        }
    )
    multisession_df.to_csv(cross_session_path / "CT014_overall_performance.csv", index=False)

    coefficients, block_counts, dates = main_module.prepare_learning_curve_data(
        mouse="CT014",
        multi_session_save_path=cross_session_path,
    )

    assert coefficients.tolist() == [0.1, 0.2, 0.3]
    assert block_counts.tolist() == [4, 5, 6]
    assert dates.tolist() == ["2025-12-05", "2025-12-08", "2025-12-23"]
    assert coefficients.dtype.kind == "f"
    assert block_counts.dtype.kind in {"f", "i"}


def test_prepare_learning_curve_data_uses_requested_regressor(tmp_path):
    """Learning-curve preparation should read the requested regressor slope column."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "cross_session_analysis"
    cross_session_path.mkdir()
    multisession_df = pd.DataFrame(
        {
            "date": ["2025-12-05", "2025-12-08"],
            "prev_consecutive_rewards_slope": [9.9, 8.8],
            "prev_n_rewarded_slope": [0.25, 0.5],
            "n_blocks": [4, 5],
        }
    )
    multisession_df.to_csv(cross_session_path / "CT014_overall_performance.csv", index=False)

    coefficients, block_counts, dates = main_module.prepare_learning_curve_data(
        mouse="CT014",
        multi_session_save_path=cross_session_path,
        learning_regressor="prev_n_rewarded",
    )

    assert coefficients.tolist() == [0.25, 0.5]
    assert block_counts.tolist() == [4, 5]
    assert dates.tolist() == ["2025-12-05", "2025-12-08"]


def test_prepare_learning_curve_data_loudly_rejects_missing_requested_regressor(tmp_path):
    """Missing requested learning regressors should fail before plotting stale data."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "cross_session_analysis"
    cross_session_path.mkdir()
    multisession_df = pd.DataFrame(
        {
            "date": ["2025-12-05"],
            "prev_consecutive_rewards_slope": [0.1],
            "n_blocks": [4],
        }
    )
    multisession_df.to_csv(cross_session_path / "CT014_overall_performance.csv", index=False)

    with pytest.raises(ValueError, match="prev_n_rewarded_slope"):
        main_module.prepare_learning_curve_data(
            mouse="CT014",
            multi_session_save_path=cross_session_path,
            learning_regressor="prev_n_rewarded",
        )


def test_prepare_session_trials_to_correct_summary_computes_median_and_quartiles():
    """Session TTC summaries should pool block types and report Q1/median/Q3."""
    main_module = load_main_module()
    first_session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    second_session = SimpleNamespace(sess_id_full="CT014_2025-12-16_153200", date="2025-12-16")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=first_session,
            block_performance=pd.DataFrame(
                {
                    "block_type": ["right_cued", "left_uncued", "right_uncued", "left_cued"],
                    "trials_to_correct": [1, 3, 5, 7],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
        main_module.SavedSessionAnalysis(
            session=second_session,
            block_performance=pd.DataFrame(
                {
                    "block_type": ["right_cued", "left_uncued", "right_uncued"],
                    "trials_to_correct": [2, "None", 10],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
    ]

    summary_df = main_module.prepare_session_trials_to_correct_summary(saved_sessions)

    assert summary_df["date"].tolist() == ["2025-12-05", "2025-12-16"]
    assert summary_df["session_id"].tolist() == [
        "CT014_2025-12-05_165240",
        "CT014_2025-12-16_153200",
    ]
    assert summary_df["n_valid_blocks"].tolist() == [4, 2]
    assert summary_df["trials_to_correct_q1"].tolist() == [2.5, 4.0]
    assert summary_df["trials_to_correct_median"].tolist() == [4.0, 6.0]
    assert summary_df["trials_to_correct_q3"].tolist() == [5.5, 8.0]


def test_prepare_session_trials_to_correct_summary_skips_invalid_sessions_with_warning():
    """Sessions without numeric TTC values should be omitted with a user warning."""
    main_module = load_main_module()
    valid_session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    invalid_session = SimpleNamespace(sess_id_full="CT014_2025-12-16_153200", date="2025-12-16")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=valid_session,
            block_performance=pd.DataFrame({"trials_to_correct": [1, 3]}),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
        main_module.SavedSessionAnalysis(
            session=invalid_session,
            block_performance=pd.DataFrame({"trials_to_correct": ["None", np.nan]}),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
    ]

    with pytest.warns(UserWarning, match="CT014_2025-12-16_153200"):
        summary_df = main_module.prepare_session_trials_to_correct_summary(saved_sessions)

    assert summary_df["session_id"].tolist() == ["CT014_2025-12-05_165240"]
    assert summary_df["n_valid_blocks"].tolist() == [2]


def test_prepare_side_trials_to_correct_block_points_labels_side_and_no_correct_blocks():
    """Side-specific block points should preserve numeric TTC and no-correct blocks."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2, 3],
                    "block_type": ["right_cued", "left_uncued", "dark period", "left_cued"],
                    "trials_to_correct": [2, "None", 1, "6"],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        )
    ]

    points_df = main_module.prepare_side_trials_to_correct_block_points(saved_sessions)

    assert points_df["block_ix"].tolist() == [0, 1, 3]
    assert points_df["rewarded_side"].tolist() == ["right", "left", "left"]
    assert points_df["trials_to_correct_numeric"].tolist()[:1] == [2.0]
    assert np.isnan(points_df.loc[1, "trials_to_correct_numeric"])
    assert points_df["no_correct_choice"].tolist() == [False, True, False]


def test_prepare_side_trials_to_correct_summary_counts_completion_by_side():
    """Side summaries should count no-correct blocks separately from valid TTC values."""
    main_module = load_main_module()
    first_session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    second_session = SimpleNamespace(sess_id_full="CT014_2025-12-16_153200", date="2025-12-16")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=first_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2, 3, 4],
                    "block_type": ["left_cued", "left_uncued", "left_cued", "right_cued", "dark period"],
                    "trials_to_correct": [2, 6, "None", 1, "None"],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
        main_module.SavedSessionAnalysis(
            session=second_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1],
                    "block_type": ["right_uncued", "right_cued"],
                    "trials_to_correct": ["None", 5],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
    ]

    summary_df = main_module.prepare_side_trials_to_correct_summary(saved_sessions)

    left_first = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["rewarded_side"] == "left")
    ].iloc[0]
    right_second = summary_df[
        (summary_df["date"] == "2025-12-16") & (summary_df["rewarded_side"] == "right")
    ].iloc[0]

    assert left_first["n_blocks"] == 3
    assert left_first["n_valid_blocks"] == 2
    assert left_first["n_no_correct_blocks"] == 1
    assert left_first["completion_fraction"] == pytest.approx(2 / 3)
    assert left_first["trials_to_correct_q1"] == 3.0
    assert left_first["trials_to_correct_median"] == 4.0
    assert left_first["trials_to_correct_q3"] == 5.0
    assert right_second["n_blocks"] == 2
    assert right_second["n_valid_blocks"] == 1
    assert right_second["n_no_correct_blocks"] == 1
    assert right_second["completion_fraction"] == pytest.approx(0.5)


def test_prepare_side_trials_to_correct_summary_uses_nan_quartiles_without_valid_blocks():
    """A side with only no-correct blocks should keep completion counts and NaN quartiles."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1],
                    "block_type": ["left_cued", "left_uncued"],
                    "trials_to_correct": ["None", np.nan],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        )
    ]

    summary_df = main_module.prepare_side_trials_to_correct_summary(saved_sessions)

    assert summary_df["rewarded_side"].tolist() == ["left"]
    assert summary_df.loc[0, "n_blocks"] == 2
    assert summary_df.loc[0, "n_valid_blocks"] == 0
    assert summary_df.loc[0, "n_no_correct_blocks"] == 2
    assert summary_df.loc[0, "completion_fraction"] == 0
    assert np.isnan(summary_df.loc[0, "trials_to_correct_q1"])
    assert np.isnan(summary_df.loc[0, "trials_to_correct_median"])
    assert np.isnan(summary_df.loc[0, "trials_to_correct_q3"])


def test_prepare_block_switch_summary_calculates_overall_and_side_medians():
    """Switch summaries should use raw n_switches for overall and side groups."""
    main_module = load_main_module()
    first_session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    second_session = SimpleNamespace(sess_id_full="CT014_2025-12-16_153200", date="2025-12-16")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=first_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2, 3],
                    "block_type": ["left_cued", "left_uncued", "right_cued", "dark period"],
                    "n_switches": [1, 3, 2, 9],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
        main_module.SavedSessionAnalysis(
            session=second_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2],
                    "block_type": ["left_cued", "right_uncued", "right_cued"],
                    "n_switches": [5, 1, 7],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
    ]

    points_df = main_module.prepare_block_switch_block_points(saved_sessions)
    summary_df = main_module.prepare_block_switch_summary(saved_sessions)

    assert points_df["switch_group"].tolist() == [
        "overall",
        "left",
        "overall",
        "left",
        "overall",
        "right",
        "overall",
        "overall",
        "left",
        "overall",
        "right",
        "overall",
        "right",
    ]

    first_overall = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["switch_group"] == "overall")
    ].iloc[0]
    first_left = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["switch_group"] == "left")
    ].iloc[0]
    second_right = summary_df[
        (summary_df["date"] == "2025-12-16") & (summary_df["switch_group"] == "right")
    ].iloc[0]

    assert first_overall["n_blocks"] == 4
    assert first_overall["n_switches_median"] == 2.5
    assert first_overall["n_switches_q1"] == 1.75
    assert first_overall["n_switches_q3"] == 4.5
    assert first_left["n_blocks"] == 2
    assert first_left["n_switches_median"] == 2.0
    assert second_right["n_blocks"] == 2
    assert second_right["n_switches_median"] == 4.0


def test_prepare_block_explore_summary_calculates_overall_and_side_medians():
    """Explore summaries should use raw n_explore_trials for overall and side groups."""
    main_module = load_main_module()
    first_session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    second_session = SimpleNamespace(sess_id_full="CT014_2025-12-16_153200", date="2025-12-16")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=first_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2, 3],
                    "block_type": ["left_cued", "left_uncued", "right_cued", "dark period"],
                    "n_explore_trials": [1, 3, 2, 0],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
        main_module.SavedSessionAnalysis(
            session=second_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2],
                    "block_type": ["left_cued", "right_uncued", "right_cued"],
                    "n_explore_trials": [5, 1, 7],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
    ]

    points_df = main_module.prepare_block_explore_block_points(saved_sessions)
    summary_df = main_module.prepare_block_explore_summary(saved_sessions)

    assert points_df["explore_group"].tolist() == [
        "overall",
        "left",
        "overall",
        "left",
        "overall",
        "right",
        "overall",
        "left",
        "overall",
        "right",
        "overall",
        "right",
    ]

    first_overall = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["explore_group"] == "overall")
    ].iloc[0]
    first_left = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["explore_group"] == "left")
    ].iloc[0]
    second_right = summary_df[
        (summary_df["date"] == "2025-12-16") & (summary_df["explore_group"] == "right")
    ].iloc[0]

    assert first_overall["n_blocks"] == 3
    assert first_overall["n_explore_trials_median"] == 2.0
    assert first_overall["n_explore_trials_q1"] == 1.5
    assert first_overall["n_explore_trials_q3"] == 2.5
    assert first_left["n_blocks"] == 2
    assert first_left["n_explore_trials_median"] == 2.0
    assert second_right["n_blocks"] == 2
    assert second_right["n_explore_trials_median"] == 4.0


def test_prepare_block_explore_points_derives_counts_from_augmented_trials():
    """Saved sessions without block explore counts should use trial explore tags."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1],
                    "block_type": ["left_cued", "right_uncued"],
                }
            ),
            augmented_trial_df=pd.DataFrame(
                {
                    "cur_block": [10, 10, 11, 11, 11],
                    "explore_trial": [1, 0, "true", "0", 1],
                }
            ),
        )
    ]

    points_df = main_module.prepare_block_explore_block_points(saved_sessions)

    overall_points = points_df[points_df["explore_group"] == "overall"]
    assert overall_points["n_explore_trials"].tolist() == [1.0, 2.0]


def test_prepare_correct_after_first_summary_excludes_dark_periods_and_summarizes_groups():
    """Post-first-correct summaries should use only side blocks for all groups."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2, 3],
                    "block_type": ["left_cued", "left_uncued", "right_cued", "dark period"],
                    "percent_correct_after_first_correct": [0.5, 1.0, 0.25, 1.0],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        )
    ]

    points_df = main_module.prepare_correct_after_first_block_points(saved_sessions)
    summary_df = main_module.prepare_correct_after_first_summary(saved_sessions)

    assert points_df["correct_after_first_group"].tolist() == [
        "overall",
        "left",
        "overall",
        "left",
        "overall",
        "right",
    ]
    assert "dark period" not in points_df["block_type"].tolist()

    overall = summary_df[summary_df["correct_after_first_group"] == "overall"].iloc[0]
    left = summary_df[summary_df["correct_after_first_group"] == "left"].iloc[0]
    right = summary_df[summary_df["correct_after_first_group"] == "right"].iloc[0]

    assert overall["n_blocks"] == 3
    assert overall["n_valid_blocks"] == 3
    assert overall["percent_correct_after_first_median"] == 0.5
    assert overall["percent_correct_after_first_q1"] == 0.375
    assert overall["percent_correct_after_first_q3"] == 0.75
    assert left["n_blocks"] == 2
    assert left["percent_correct_after_first_median"] == 0.75
    assert right["n_blocks"] == 1
    assert right["percent_correct_after_first_median"] == 0.25


def test_prepare_correct_after_first_block_points_marks_no_correct_blocks():
    """Missing post-first-correct values should become no-correct marker rows."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1],
                    "block_type": ["left_cued", "right_uncued"],
                    "percent_correct_after_first_correct": ["None", np.nan],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        )
    ]

    points_df = main_module.prepare_correct_after_first_block_points(saved_sessions)
    summary_df = main_module.prepare_correct_after_first_summary(saved_sessions)

    assert points_df["no_correct_choice"].tolist() == [True, True, True, True]
    assert points_df["percent_correct_after_first_numeric"].isna().all()
    assert summary_df["n_valid_blocks"].tolist() == [0, 0, 0]
    assert summary_df["n_no_correct_blocks"].tolist() == [2, 1, 1]
    assert summary_df["percent_correct_after_first_median"].isna().all()


def test_load_saved_multisession_augmented_trials_validates_required_columns(tmp_path):
    """Saved multisession trial input should load only when HMM columns exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data directory containing the saved trial CSV.

    Returns
    -------
    None
        Asserts that a saved trial table with required GLM-HMM and inherited
        block columns is loaded with legacy reward flags normalized.
    """
    main_module = load_main_module()
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        sess_id_full="CT014_multisession",
    )
    trial_df = pd.DataFrame(
        {
            "prev_action": [0, 1],
            "give_reward": [0, 0],
            "action": [1, 0],
            "FQlearning_rel_value": [0.1, -0.2],
            "inherited_block_strategy": [0, 1],
            "inherited_block_bias": ["False", "True"],
        }
    )
    trial_df.to_csv(tmp_path / "CT014_multisession_augmented_trials.csv", index=False)

    loaded = main_module.load_saved_multisession_augmented_trials(
        session=session,
        predictor_columns=("FQlearning_rel_value",),
    )

    assert "experimenter_reward_given" in loaded.columns
    assert "give_reward" not in loaded.columns
    assert loaded["inherited_block_strategy"].tolist() == [0, 1]
    assert loaded["inherited_block_bias"].tolist() == [False, True]


def test_load_saved_multisession_augmented_trials_errors_when_inherited_columns_missing(tmp_path):
    """Saved multisession trial input should require inherited block columns.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data directory containing the incomplete trial CSV.

    Returns
    -------
    None
        Asserts that missing inherited block columns raise a clear ValueError.
    """
    main_module = load_main_module()
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        sess_id_full="CT014_multisession",
    )
    trial_df = pd.DataFrame(
        {
            "prev_action": [0],
            "give_reward": [0],
            "action": [1],
            "FQlearning_rel_value": [0.1],
        }
    )
    trial_df.to_csv(tmp_path / "CT014_multisession_augmented_trials.csv", index=False)

    with pytest.raises(ValueError, match="inherited_block_strategy"):
        main_module.load_saved_multisession_augmented_trials(
            session=session,
            predictor_columns=("FQlearning_rel_value",),
        )


def test_run_multisession_analysis_uses_saved_augmented_trials_when_requested(tmp_path, monkeypatch):
    """Saved-trial mode should skip block modeling and return the saved trial CSV.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces HMM functions with lightweight captures.

    Returns
    -------
    None
        Asserts that block modeling and pre-modeling save are skipped while the
        saved block-inherited trial table is loaded and returned.
    """
    main_module = load_main_module()
    synthetic_session = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
        figure_path=tmp_path,
    )
    concatenated = SimpleNamespace(
        block_performance=pd.DataFrame({"block_ix": [0]}),
        augmented_trial_df=pd.DataFrame({"action": [0]}),
    )
    saved_trial_df = pd.DataFrame(
        {
            "prev_action": [0, 1],
            "give_reward": [0, 0],
            "action": [1, 0],
            "FQlearning_rel_value": [0.1, -0.2],
            "inherited_block_strategy": [0, 1],
            "inherited_block_bias": ["False", "True"],
        }
    )
    saved_trial_df.to_csv(tmp_path / "CT014_multisession_augmented_trials.csv", index=False)
    monkeypatch.setattr(
        main_module,
        "save_concatenated_multisession_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("saved input should not be overwritten")),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.bssm,
        "run_block_modeling",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("block modeling should be skipped")),
        raising=False,
    )
    block_selection, trial_selection, modeled_block_df, modeled_trial_df = main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        trial_num_states=3,
        prior_alpha=4,
        prior_sigma=5,
        trial_random_seed=2001,
        trial_predictor_columns=("FQlearning_rel_value",),
        trial_input_source="saved_augmented_trials",
    )

    assert block_selection is None
    assert trial_selection is None
    pd.testing.assert_frame_equal(modeled_block_df, concatenated.block_performance)
    assert "experimenter_reward_given" in modeled_trial_df.columns
    assert "give_reward" not in modeled_trial_df.columns
    assert modeled_trial_df["inherited_block_strategy"].tolist() == [0, 1]


def test_run_multisession_analysis_rejects_unknown_trial_input_source(tmp_path):
    """Multisession trial input source should fail fast on unsupported values."""
    main_module = load_main_module()
    synthetic_session = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
        figure_path=tmp_path,
    )
    concatenated = SimpleNamespace(
        block_performance=pd.DataFrame({"block_ix": [0]}),
        augmented_trial_df=pd.DataFrame({"action": [0]}),
    )

    with pytest.raises(ValueError, match="trial_input_source"):
        main_module.run_multisession_analysis(
            concatenated=concatenated,
            session=synthetic_session,
            trial_predictor_columns=("FQlearning_rel_value",),
            trial_input_source="saved_trials",
        )


def test_run_multisession_analysis_forwards_block_plot_settings(tmp_path, monkeypatch):
    """Multisession analysis should forward block-state plotting settings."""
    main_module = load_main_module()
    synthetic_session = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
        figure_path=tmp_path,
    )
    concatenated = SimpleNamespace(
        block_performance=pd.DataFrame({"block_ix": [0, 1]}),
        augmented_trial_df=pd.DataFrame({"cur_block": [0, 1], "action": [1, 0]}),
        block_session_lengths=np.array([2]),
        trial_session_lengths=np.array([2]),
    )
    captured = {}

    def fake_run_block_modeling(
        block_performance,
        augmented_trial_df,
        session,
        num_states,
        prior_alpha,
        prior_sigma,
        random_seed,
        predicted_state_line_width=None,
        state_plot_figsize=None,
        block_secondary_trace=None,
        plot_sliding_regression=False,
        sliding_regression_window_size=10,
        sliding_regression_step_size=5,
    ):
        captured["predicted_state_line_width"] = predicted_state_line_width
        captured["state_plot_figsize"] = state_plot_figsize
        captured["block_secondary_trace"] = block_secondary_trace
        captured["plot_sliding_regression"] = plot_sliding_regression
        captured["sliding_regression_window_size"] = sliding_regression_window_size
        captured["sliding_regression_step_size"] = sliding_regression_step_size
        return (
            block_performance.assign(inferred_strategy=[0, 1]),
            augmented_trial_df.assign(inherited_block_strategy=[0, 1], inherited_block_bias=["False", "False"]),
        )

    monkeypatch.setattr(main_module.bssm, "run_block_modeling", fake_run_block_modeling, raising=False)
    monkeypatch.setattr(main_module.tssm, "run_trial_modeling", lambda trial_df, **_kwargs: trial_df, raising=False)
    monkeypatch.setattr(main_module, "save_concatenated_multisession_inputs", lambda *_args, **_kwargs: None)

    main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        trial_predictor_columns=("FQlearning_rel_value",),
        block_predicted_state_line_width=0.4,
        block_state_plot_figsize=(18, 6),
        block_secondary_trace="bias_rl",
        plot_sliding_regression=True,
        sliding_regression_window_size=10,
        sliding_regression_step_size=5,
    )

    assert captured == {
        "predicted_state_line_width": 0.4,
        "state_plot_figsize": (18, 6),
        "block_secondary_trace": "bias_rl",
        "plot_sliding_regression": True,
        "sliding_regression_window_size": 10,
        "sliding_regression_step_size": 5,
    }


def test_run_multisession_analysis_runs_block_modeling_on_concatenated_inputs(tmp_path, monkeypatch):
    """Multisession orchestration should run block modeling on concatenated data."""
    main_module = load_main_module()
    synthetic_session = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
        figure_path=tmp_path,
    )
    concatenated = SimpleNamespace(
        block_performance=pd.DataFrame({"block_ix": [0, 1]}),
        augmented_trial_df=pd.DataFrame({"cur_block": [0, 1], "action": [1, 0]}),
        block_session_lengths=np.array([2]),
        trial_session_lengths=np.array([2]),
    )
    calls = []
    save_calls = []
    trial_predictor_columns = (
        "FQlearning_rel_value",
        "HMM_rel_value_logodds_decay",
        "relative_doubt_index",
        "perseveration_regressor",
        "time_to_choice",
    )

    def fake_run_block_modeling(
        block_performance,
        augmented_trial_df,
        session,
        num_states,
        prior_alpha,
        prior_sigma,
        random_seed,
        predicted_state_line_width=None,
        state_plot_figsize=None,
        block_secondary_trace=None,
        plot_sliding_regression=False,
        sliding_regression_window_size=10,
        sliding_regression_step_size=5,
    ):
        calls.append(
            (
                "block",
                session.sess_id_full,
                num_states,
                prior_alpha,
                prior_sigma,
                random_seed,
                predicted_state_line_width,
                state_plot_figsize,
                block_secondary_trace,
                plot_sliding_regression,
                sliding_regression_window_size,
                sliding_regression_step_size,
            )
        )
        return (
            block_performance.assign(inferred_strategy=[0, 1]),
            augmented_trial_df.assign(inherited_block_strategy=[0, 1], inherited_block_bias=["False", "False"]),
        )

    monkeypatch.setattr(main_module.bssm, "run_block_modeling", fake_run_block_modeling, raising=False)
    monkeypatch.setattr(
        main_module,
        "save_concatenated_multisession_inputs",
        lambda *args, **_kwargs: save_calls.append(args),
        raising=False,
    )

    block_selection, trial_selection, modeled_block_df, modeled_trial_df = main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        block_num_states=2,
        trial_num_states=3,
        prior_alpha=4,
        prior_sigma=5,
        block_random_seed=1001,
        trial_random_seed=2001,
        trial_predictor_columns=trial_predictor_columns,
        block_predicted_state_line_width=0.5,
        block_state_plot_figsize=(18, 6),
        block_secondary_trace="bias_rl",
        plot_sliding_regression=True,
        sliding_regression_window_size=10,
        sliding_regression_step_size=5,
    )

    assert len(save_calls) == 1
    assert save_calls[0][0] is concatenated
    assert save_calls[0][1] is synthetic_session
    assert calls == [
        ("block", "CT014_multisession", 2, 4, 5, 1001, 0.5, (18, 6), "bias_rl", True, 10, 5),
    ]
    assert block_selection is None
    assert trial_selection is None
    assert modeled_block_df["inferred_strategy"].tolist() == [0, 1]
    assert modeled_trial_df["inherited_block_strategy"].tolist() == [0, 1]
