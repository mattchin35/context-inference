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
        block columns is loaded unchanged.
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

    assert loaded.shape == trial_df.shape
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


def test_main_multisession_calls_plot_learning_curve_with_cleaned_data(tmp_path, monkeypatch):
    """main_multisession should call the existing plotter with numeric data."""
    main_module = load_main_module()
    captured = {}
    coefficients = np.array([0.1, 0.3], dtype=float)
    block_counts = np.array([4, 6], dtype=int)
    dates = np.array(["2025-12-05", "2025-12-23"], dtype=object)

    monkeypatch.setattr(
        main_module,
        "prepare_learning_curve_data",
        lambda mouse, multi_session_save_path: (coefficients, block_counts, dates),
    )

    def fake_plot_learning_curve(plot_coefficients, switches_per_session, figure_id, plot_path, dates=None):
        captured["coefficients"] = plot_coefficients
        captured["switches_per_session"] = switches_per_session
        captured["figure_id"] = figure_id
        captured["plot_path"] = plot_path
        captured["dates"] = dates

    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_learning_curve",
        fake_plot_learning_curve,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "find_saved_session_by_date",
        lambda mouse, date, session_data_root, multi_session_save_path: SimpleNamespace(
            sess_id_full=f"{mouse}_{date}_000000",
            date=date,
            processed_data_path=tmp_path,
        ),
    )
    monkeypatch.setattr(
        main_module,
        "load_saved_session_analysis",
        lambda session: SimpleNamespace(session=session),
    )
    monkeypatch.setattr(
        main_module,
        "concatenate_saved_sessions",
        lambda saved_sessions: SimpleNamespace(),
    )
    monkeypatch.setattr(
        main_module,
        "build_multisession_session",
        lambda mouse, multi_session_save_path, sess_id_full: SimpleNamespace(sess_id_full=sess_id_full),
    )
    monkeypatch.setattr(main_module, "run_multisession_analysis", lambda **kwargs: None)

    main_module.main_multisession()

    np.testing.assert_array_equal(captured["coefficients"], coefficients)
    np.testing.assert_array_equal(captured["switches_per_session"], block_counts)
    np.testing.assert_array_equal(captured["dates"], dates)
    assert captured["figure_id"] == "CT014"


def test_run_multisession_analysis_uses_saved_augmented_trials_when_requested(tmp_path, monkeypatch):
    """Saved-trial mode should skip block modeling and use the saved trial CSV.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces HMM functions with lightweight captures.

    Returns
    -------
    None
        Asserts that block modeling and pre-modeling save are skipped, while
        trial modeling receives the saved block-inherited trial table.
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
    captured = {}

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
    monkeypatch.setattr(
        main_module.tssm,
        "run_information_criteria",
        lambda trial_df, **_kwargs: {"trial_ic": "saved"},
        raising=False,
    )

    def fake_run_trial_modeling(
        trial_df,
        session,
        num_states,
        prior_alpha,
        prior_sigma,
        predictor_columns,
        random_seed,
        state_plot_line_width=None,
        state_plot_figsize=None,
    ):
        captured["trial_df"] = trial_df.copy()
        captured["num_states"] = num_states
        captured["predictor_columns"] = predictor_columns
        return trial_df.assign(inferred_strategy=[1, 0])

    monkeypatch.setattr(main_module.tssm, "run_trial_modeling", fake_run_trial_modeling, raising=False)

    main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        trial_num_states=3,
        prior_alpha=4,
        prior_sigma=5,
        trial_random_seed=2001,
        trial_predictor_columns=("FQlearning_rel_value",),
        trial_input_source="saved_augmented_trials",
    )

    assert captured["num_states"] == 3
    assert captured["predictor_columns"] == ("FQlearning_rel_value",)
    assert captured["trial_df"]["inherited_block_strategy"].tolist() == [0, 1]


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


def test_run_multisession_analysis_forwards_trial_plot_settings(tmp_path, monkeypatch):
    """Multisession analysis should forward trial-state plotting settings.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces HMM functions with lightweight captures.

    Returns
    -------
    None
        Asserts that multisession-specific trial plotting settings reach
        `trial_state_space_modeling.run_trial_modeling`.
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
    captured = {}

    def fake_run_trial_modeling(
        trial_df,
        session,
        num_states,
        prior_alpha,
        prior_sigma,
        predictor_columns,
        random_seed,
        state_plot_line_width=None,
        state_plot_figsize=None,
    ):
        captured["state_plot_line_width"] = state_plot_line_width
        captured["state_plot_figsize"] = state_plot_figsize
        return trial_df

    monkeypatch.setattr(main_module.tssm, "run_trial_modeling", fake_run_trial_modeling, raising=False)

    main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        trial_predictor_columns=("FQlearning_rel_value",),
        trial_input_source="saved_augmented_trials",
        trial_state_plot_line_width=0.4,
        trial_state_plot_figsize=(12, 4),
    )

    assert captured == {
        "state_plot_line_width": 0.4,
        "state_plot_figsize": (12, 4),
    }


def test_run_multisession_analysis_runs_model_selection_before_modeling(tmp_path, monkeypatch):
    """Multisession orchestration should mirror the full `main_mouse` HMM flow."""
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
    trial_predictor_columns = (
        "FQlearning_rel_value",
        "HMM_rel_value_logodds_decay",
        "relative_doubt_index",
        "perseveration_regressor",
        "time_to_choice",
    )

    def fake_block_information_criteria(block_performance, session, algorithm, prior_alpha, prior_sigma, random_seed):
        calls.append(("block_ic", session.sess_id_full, algorithm, prior_alpha, prior_sigma, random_seed))
        return {"block_ic": True}

    def fake_trial_information_criteria(
        trial_df,
        session,
        algorithm,
        prior_alpha,
        prior_sigma,
        predictor_columns,
        random_seed,
    ):
        calls.append(
            (
                "trial_ic",
                session.sess_id_full,
                algorithm,
                prior_alpha,
                prior_sigma,
                predictor_columns,
                random_seed,
                "inherited_block_strategy" in trial_df.columns,
            )
        )
        return {"trial_ic": True}

    def fake_run_block_modeling(
        block_performance,
        augmented_trial_df,
        session,
        num_states,
        prior_alpha,
        prior_sigma,
        random_seed,
        predicted_state_line_width=None,
    ):
        calls.append(("block", session.sess_id_full, num_states, prior_alpha, prior_sigma, random_seed))
        return (
            block_performance.assign(inferred_strategy=[0, 1]),
            augmented_trial_df.assign(inherited_block_strategy=[0, 1], inherited_block_bias=["False", "False"]),
        )

    def fake_run_trial_modeling(
        trial_df,
        session,
        num_states,
        prior_alpha,
        prior_sigma,
        predictor_columns,
        random_seed,
        state_plot_line_width=None,
        state_plot_figsize=None,
    ):
        calls.append(
            (
                "trial",
                session.sess_id_full,
                num_states,
                prior_alpha,
                prior_sigma,
                predictor_columns,
                random_seed,
                "inherited_block_strategy" in trial_df.columns,
            )
        )
        return trial_df.assign(inferred_strategy=[1, 0])

    monkeypatch.setattr(main_module.bssm, "run_information_criteria", fake_block_information_criteria, raising=False)
    monkeypatch.setattr(main_module.bssm, "run_block_modeling", fake_run_block_modeling, raising=False)
    monkeypatch.setattr(main_module.tssm, "run_information_criteria", fake_trial_information_criteria, raising=False)
    monkeypatch.setattr(main_module.tssm, "run_trial_modeling", fake_run_trial_modeling, raising=False)
    monkeypatch.setattr(
        main_module,
        "save_concatenated_multisession_inputs",
        lambda *_args, **_kwargs: None,
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
    )

    assert calls == [
        ("block_ic", "CT014_multisession", "MLE", 4, 5, 1001),
        ("block", "CT014_multisession", 2, 4, 5, 1001),
        ("trial_ic", "CT014_multisession", "MLE", 4, 5, trial_predictor_columns, 2001, True),
        ("trial", "CT014_multisession", 3, 4, 5, trial_predictor_columns, 2001, True),
    ]
    assert block_selection == {"block_ic": True}
    assert trial_selection == {"trial_ic": True}
    assert modeled_block_df["inferred_strategy"].tolist() == [0, 1]
    assert modeled_trial_df["inferred_strategy"].tolist() == [1, 0]
