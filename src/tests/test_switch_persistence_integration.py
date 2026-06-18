import importlib
import sys
from types import ModuleType, SimpleNamespace

import matplotlib
import pandas as pd

matplotlib.use("Agg")


def install_session_analysis_import_stubs() -> None:
    """Provide lightweight import stubs for unused regression dependencies.

    Returns
    -------
    None
        Mutates `sys.modules` so `session_analysis` can be imported in tests
        that monkeypatch away the regression code paths.
    """
    formulaic_stub = ModuleType("formulaic")
    formulaic_stub.model_matrix = lambda *_args, **_kwargs: None
    sys.modules.setdefault("formulaic", formulaic_stub)

    statsmodels_stub = ModuleType("statsmodels")
    statsmodels_api_stub = ModuleType("statsmodels.api")
    statsmodels_stub.api = statsmodels_api_stub
    sys.modules.setdefault("statsmodels", statsmodels_stub)
    sys.modules.setdefault("statsmodels.api", statsmodels_api_stub)


def test_run_analysis_saves_switch_persistence_outputs(tmp_path, monkeypatch):
    """Single-session analysis should save switch-persistence CSVs with core outputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces heavyweight analysis helpers with deterministic tables.

    Returns
    -------
    None
        Asserts that `run_analysis` writes the two switch-persistence tables
        beside the existing block and augmented-trial CSVs.
    """
    install_session_analysis_import_stubs()
    session_analysis = importlib.import_module("src.behavior_analysis.session_analysis")
    session = SimpleNamespace(
        mouse="CT999",
        date="2026-06-18",
        sess_id_full="CT999_2026-06-18_120000",
        processed_data_path=tmp_path,
        multi_session_save_path=None,
    )
    session_performance = pd.DataFrame(
        {
            "overall_correct": [0.5],
            "date": ["2026-06-18"],
            "prev_consecutive_rewards_slope": [0.0],
            "prev_consecutive_rewards_intercept": [0.0],
            "prev_consecutive_rewards_r_value": [0.0],
            "prev_consecutive_rewards_p_value": [1.0],
            "prev_n_correct_slope": [0.0],
            "prev_n_correct_intercept": [0.0],
            "prev_n_correct_r_value": [0.0],
            "prev_n_correct_p_value": [1.0],
        }
    )
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["left_uncued", "right_uncued"],
            "trials_to_correct": [0, 1],
            "prev_n_correct": [0, 2],
            "prev_n_rewarded": [0, 2],
            "prev_consecutive_rewards": [0, 2],
            "n_correct": [2, 1],
            "n_rewarded": [2, 1],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_trial": [0, 1, 2, 3],
            "cur_block": [0, 0, 1, 1],
            "state": [1, 1, 0, 0],
            "action": [1, 1, 1, 0],
            "correct": [1, 1, 0, 1],
            "reward": [1, 1, 0, 1],
            "experimenter_reward_given": [0, 0, 0, 0],
        }
    )

    monkeypatch.setattr(
        session_analysis,
        "analyze_session",
        lambda *_args, **_kwargs: (
            session_performance.copy(),
            block_performance.copy(),
            augmented_trial_df.copy(),
        ),
    )
    monkeypatch.setattr(
        session_analysis,
        "add_regression_stats_to_session_performance",
        lambda session_performance, **_kwargs: session_performance,
    )
    monkeypatch.setattr(
        session_analysis,
        "add_block_bias_columns",
        lambda block_performance: block_performance,
    )

    session_analysis.run_analysis(
        pd.DataFrame({"placeholder": [1]}),
        session=session,
    )

    expected_paths = [
        tmp_path / "CT999_2026-06-18_120000_block_performance.csv",
        tmp_path / "CT999_2026-06-18_120000_augmented_trials.csv",
        tmp_path / "CT999_2026-06-18_120000_switch_persistence_trials.csv",
        tmp_path / "CT999_2026-06-18_120000_switch_persistence_summary.csv",
    ]
    for path in expected_paths:
        assert path.exists()
        assert path.stat().st_size > 0


def test_plot_switch_persistence_summary_saves_grouped_lines_with_block_counts(
    tmp_path,
):
    """Switch-persistence plots should include grouped traces and n-block labels.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary figure output directory.

    Returns
    -------
    None
        Asserts that the switch-persistence summary plot is saved using the
        expected single-session filename.
    """
    install_session_analysis_import_stubs()
    session_analysis = importlib.import_module("src.behavior_analysis.session_analysis")
    sys.modules["session_analysis"] = session_analysis
    performance_plots = importlib.import_module("src.behavior_analysis.performance_plots")
    summary_df = pd.DataFrame(
        {
            "switch_group": [
                "combined",
                "combined",
                "L_to_R",
                "L_to_R",
                "R_to_L",
                "R_to_L",
            ],
            "choice_trial_after_switch": [1, 2, 1, 2, 1, 2],
            "proportion_stay": [1.0, 0.5, 1.0, 0.25, 1.0, 0.75],
            "n_blocks": [4, 4, 2, 2, 2, 2],
        }
    )

    save_path = performance_plots.plot_switch_persistence_summary(
        summary_df=summary_df,
        plot_path=tmp_path,
        figure_id="CT999_2026-06-18_120000",
    )

    assert save_path == tmp_path / "CT999_2026-06-18_120000_switch_persistence.png"
    assert save_path.exists()
