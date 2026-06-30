import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd


def load_main_module():
    """Import `main.py` with lightweight stubs for heavy dependencies.

    Returns
    -------
    module
        Imported main module. Stubbed dependencies expose only the attributes
        needed by the switch-persistence wiring test.
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
    module_spec = importlib.util.spec_from_file_location("main_module_for_switch_test", main_path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_save_and_plot_switch_persistence_for_session_delegates_to_analysis_and_plotters(
    tmp_path,
    monkeypatch,
):
    """Main should wire session metadata into switch-persistence save and plot calls.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the computation and plotting helpers with call recorders.

    Returns
    -------
    None
        Asserts that the main helper passes block/trial tables and session
        paths through to the behavior-analysis module and plotter.
    """
    main_module = load_main_module()
    block_performance = pd.DataFrame({"block_ix": [0, 1]})
    augmented_trial_df = pd.DataFrame({"cur_block": [0, 1], "action": [1, 0]})
    summary_df = pd.DataFrame(
        {
            "switch_group": ["combined"],
            "choice_trial_after_switch": [1],
            "proportion_stay": [0.5],
            "n_blocks": [2],
        }
    )
    detail_df = pd.DataFrame({"block_ix": [1], "choice_status": ["switch"]})
    session = SimpleNamespace(
        sess_id_full="CT999_2026-06-18_120000",
        processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures",
    )
    captured = {}

    def fake_save_switch_persistence_outputs(
        block_performance,
        augmented_trial_df,
        processed_data_path,
        sess_id_full,
    ):
        captured["block_performance"] = block_performance
        captured["augmented_trial_df"] = augmented_trial_df
        captured["processed_data_path"] = processed_data_path
        captured["sess_id_full"] = sess_id_full
        return detail_df, summary_df

    def fake_plot_switch_persistence_summary(summary_df, plot_path, figure_id):
        captured["summary_df"] = summary_df
        captured["plot_path"] = plot_path
        captured["figure_id"] = figure_id
        return plot_path / f"{figure_id}_switch_persistence.png"

    monkeypatch.setattr(
        main_module.switch_persistence,
        "save_switch_persistence_outputs",
        fake_save_switch_persistence_outputs,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_switch_persistence_summary",
        fake_plot_switch_persistence_summary,
        raising=False,
    )

    returned_detail, returned_summary, returned_plot_path = (
        main_module.save_and_plot_switch_persistence_for_session(
            block_performance=block_performance,
            augmented_trial_df=augmented_trial_df,
            session=session,
        )
    )

    assert captured["block_performance"] is block_performance
    assert captured["augmented_trial_df"] is augmented_trial_df
    assert captured["processed_data_path"] == session.processed_data_path
    assert captured["sess_id_full"] == session.sess_id_full
    assert captured["summary_df"] is summary_df
    assert captured["plot_path"] == session.figure_path
    assert captured["figure_id"] == session.sess_id_full
    assert returned_detail is detail_df
    assert returned_summary is summary_df
    assert returned_plot_path == session.figure_path / (
        "CT999_2026-06-18_120000_switch_persistence.png"
    )


def test_save_and_plot_post_first_correct_accuracy_for_session_delegates_to_analysis_and_plotters(
    tmp_path,
    monkeypatch,
):
    """Main should wire session metadata into post-first-correct save and plot calls.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary processed-data and figure directory.
    monkeypatch : pytest.MonkeyPatch
        Replaces the computation and plotting helpers with call recorders.

    Returns
    -------
    None
        Asserts that the main helper passes block/trial tables and session
        paths through to the behavior-analysis module and plotter.
    """
    main_module = load_main_module()
    block_performance = pd.DataFrame({"block_ix": [0, 1]})
    augmented_trial_df = pd.DataFrame({"cur_block": [0, 1], "action": [1, 0]})
    summary_df = pd.DataFrame(
        {
            "correct_group": ["combined"],
            "choice_trial_after_first_correct": [0],
            "proportion_correct": [1.0],
            "n_blocks": [2],
        }
    )
    detail_df = pd.DataFrame({"block_ix": [1], "choice_status": ["post_first_correct"]})
    session = SimpleNamespace(
        sess_id_full="CT999_2026-06-18_120000",
        processed_data_path=tmp_path / "processed",
        figure_path=tmp_path / "figures",
    )
    captured = {}

    def fake_save_post_first_correct_accuracy_outputs(
        block_performance,
        augmented_trial_df,
        processed_data_path,
        sess_id_full,
    ):
        captured["block_performance"] = block_performance
        captured["augmented_trial_df"] = augmented_trial_df
        captured["processed_data_path"] = processed_data_path
        captured["sess_id_full"] = sess_id_full
        return detail_df, summary_df

    def fake_plot_post_first_correct_accuracy_summary(summary_df, plot_path, figure_id):
        captured["summary_df"] = summary_df
        captured["plot_path"] = plot_path
        captured["figure_id"] = figure_id
        return plot_path / f"{figure_id}_post_first_correct_accuracy.png"

    monkeypatch.setattr(
        main_module.switch_persistence,
        "save_post_first_correct_accuracy_outputs",
        fake_save_post_first_correct_accuracy_outputs,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_post_first_correct_accuracy_summary",
        fake_plot_post_first_correct_accuracy_summary,
        raising=False,
    )

    returned_detail, returned_summary, returned_plot_path = (
        main_module.save_and_plot_post_first_correct_accuracy_for_session(
            block_performance=block_performance,
            augmented_trial_df=augmented_trial_df,
            session=session,
        )
    )

    assert captured["block_performance"] is block_performance
    assert captured["augmented_trial_df"] is augmented_trial_df
    assert captured["processed_data_path"] == session.processed_data_path
    assert captured["sess_id_full"] == session.sess_id_full
    assert captured["summary_df"] is summary_df
    assert captured["plot_path"] == session.figure_path
    assert captured["figure_id"] == session.sess_id_full
    assert returned_detail is detail_df
    assert returned_summary is summary_df
    assert returned_plot_path == session.figure_path / (
        "CT999_2026-06-18_120000_post_first_correct_accuracy.png"
    )
