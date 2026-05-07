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


def test_plot_mouse_learning_curve_uses_all_available_summary_dates(tmp_path, monkeypatch):
    """Learning curves should use every date in the overall CSV, not the model dates."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "cross_session_analysis"
    cross_session_path.mkdir()
    multisession_df = pd.DataFrame(
        {
            "date": ["2025-12-16", "2025-12-05", "2025-12-23"],
            "prev_consecutive_rewards_slope": [0.2, 0.1, 0.3],
            "n_blocks": [5, 4, 6],
        }
    )
    multisession_df.to_csv(cross_session_path / "CT014_overall_performance.csv", index=False)
    captured = {}

    def fake_plot_learning_curve(coefficients, switches_per_session, figure_id, plot_path, dates=None):
        captured["coefficients"] = list(coefficients)
        captured["switches_per_session"] = list(switches_per_session)
        captured["figure_id"] = figure_id
        captured["plot_path"] = plot_path
        captured["dates"] = list(dates)

    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_learning_curve",
        fake_plot_learning_curve,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "get_session_block_count_column",
        lambda df: "n_blocks",
        raising=False,
    )

    plotted_df = main_module.plot_mouse_learning_curve(
        mouse="CT014",
        multi_session_save_path=cross_session_path,
    )

    assert plotted_df["date"].tolist() == ["2025-12-05", "2025-12-16", "2025-12-23"]
    assert captured == {
        "coefficients": [0.1, 0.2, 0.3],
        "switches_per_session": [4, 5, 6],
        "figure_id": "CT014",
        "plot_path": cross_session_path,
        "dates": ["2025-12-05", "2025-12-16", "2025-12-23"],
    }


def test_run_multisession_analysis_calls_block_then_trial_modeling(tmp_path, monkeypatch):
    """Multisession orchestration should model blocks before trial states."""
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

    def fake_run_block_modeling(block_performance, augmented_trial_df, session, num_states, prior_alpha, prior_sigma):
        calls.append(("block", session.sess_id_full, num_states, prior_alpha, prior_sigma))
        return (
            block_performance.assign(inferred_strategy=[0, 1]),
            augmented_trial_df.assign(inherited_block_strategy=[0, 1], inherited_block_bias=["False", "False"]),
        )

    def fake_run_trial_modeling(trial_df, session, num_states, prior_alpha, prior_sigma):
        calls.append(("trial", session.sess_id_full, num_states, prior_alpha, prior_sigma))
        return trial_df.assign(inferred_strategy=[1, 0])

    monkeypatch.setattr(main_module.bssm, "run_block_modeling", fake_run_block_modeling, raising=False)
    monkeypatch.setattr(main_module.tssm, "run_trial_modeling", fake_run_trial_modeling, raising=False)
    monkeypatch.setattr(
        main_module,
        "save_concatenated_multisession_inputs",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    modeled_block_df, modeled_trial_df = main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        block_num_states=2,
        trial_num_states=3,
        prior_alpha=4,
        prior_sigma=5,
    )

    assert calls == [
        ("block", "CT014_multisession", 2, 4, 5),
        ("trial", "CT014_multisession", 3, 4, 5),
    ]
    assert modeled_block_df["inferred_strategy"].tolist() == [0, 1]
    assert modeled_trial_df["inferred_strategy"].tolist() == [1, 0]
