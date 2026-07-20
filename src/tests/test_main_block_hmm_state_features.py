import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd


def load_main_module():
    """Import `main.py` with lightweight dependency stubs.

    Returns
    -------
    module
        Imported main module with heavy behavior-analysis dependencies
        replaced by lightweight module objects.
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
    behavior_analysis_pkg.trial_state_space_modeling.DEFAULT_TRIAL_GLM_PREDICTOR_COLUMNS = (
        "FQlearning_rel_value",
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
    module_spec = importlib.util.spec_from_file_location(
        "main_module_for_block_hmm_state_feature_test",
        main_path,
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_main_multisession_collects_block_hmm_state_features_for_session_list(
    tmp_path,
    monkeypatch,
):
    """Multisession flow should collect separately modeled block-HMM state CSVs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path used by monkeypatched session metadata.
    monkeypatch : pytest.MonkeyPatch
        Replaces all unrelated multisession work with lightweight call
        recorders.

    Returns
    -------
    None
        Asserts that `main_multisession` passes the resolved session list to
        the block-state feature collector.
    """
    main_module = load_main_module()
    sessions = [
        SimpleNamespace(
            sess_id_full="CT016_2026-05-11_124709",
            date="2026-05-11",
            processed_data_path=tmp_path / "first",
        ),
        SimpleNamespace(
            sess_id_full="CT016_2026-05-12_124709",
            date="2026-05-12",
            processed_data_path=tmp_path / "second",
        ),
    ]
    captured = {}

    monkeypatch.setattr(
        main_module,
        "prepare_learning_curve_data",
        lambda **_kwargs: ([], [], []),
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_learning_curve",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "load_overall_performance_summary",
        lambda mouse, multi_session_save_path: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_oracle_behavior",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_ideal_observer_behavior",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "find_saved_session_by_date",
        lambda mouse, date, session_data_root, multi_session_save_path: sessions[
            len(captured.setdefault("resolved_dates", []))
        ],
    )

    def fake_find_saved_session_by_date(mouse, date, session_data_root, multi_session_save_path):
        captured.setdefault("resolved_dates", []).append(date)
        return sessions[(len(captured["resolved_dates"]) - 1) % len(sessions)]

    monkeypatch.setattr(main_module, "find_saved_session_by_date", fake_find_saved_session_by_date)
    monkeypatch.setattr(
        main_module,
        "load_saved_session_analysis",
        lambda session: SimpleNamespace(session=session),
    )
    monkeypatch.setattr(
        main_module,
        "prepare_agent_mouse_agreement_summary",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_agent_mouse_agreement_block_points",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_agent_mouse_agreement_quality",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_session_trials_to_correct_summary",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_trials_to_correct_session_summary",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_side_trials_to_correct_summary",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_side_trials_to_correct_block_points",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_side_trials_to_correct_quality",
        lambda *_args, **kwargs: captured.setdefault(
            "side_ttc_trial_display_cap",
            kwargs.get("trial_display_cap"),
        ),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_correct_after_first_summary",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_correct_after_first_block_points",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_correct_after_first_quality",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_correct_after_first_session_summary",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_block_switch_summary",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_block_switch_block_points",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_block_switches_quality",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_block_explore_summary",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_block_explore_block_points",
        lambda saved_sessions: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_block_explore_quality",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_block_explore_run_summary",
        lambda saved_sessions, min_explore_run_length_to_count: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_block_explore_run_block_points",
        lambda saved_sessions, min_explore_run_length_to_count: pd.DataFrame(),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_block_explore_run_quality",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_session_summary_metric_family",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(main_module, "concatenate_saved_sessions", lambda saved_sessions: SimpleNamespace())
    monkeypatch.setattr(
        main_module,
        "build_multisession_session",
        lambda mouse, multi_session_save_path, sess_id_full: SimpleNamespace(sess_id_full=sess_id_full),
    )
    monkeypatch.setattr(
        main_module,
        "run_multisession_analysis",
        lambda **_kwargs: (None, None, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_summary_grid",
        lambda *_args, **kwargs: captured.setdefault(
            "summary_grid_trial_display_cap",
            kwargs.get("trial_display_cap"),
        ),
        raising=False,
    )

    def fake_collect_block_hmm_state_features_for_sessions(sessions, output_path, mouse):
        captured["sessions"] = list(sessions)
        captured["output_path"] = output_path
        captured["mouse"] = mouse
        return pd.DataFrame()

    monkeypatch.setattr(
        main_module.bssm,
        "collect_block_hmm_state_features_for_sessions",
        fake_collect_block_hmm_state_features_for_sessions,
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_block_hmm_state_feature_scatter",
        lambda *_args, **_kwargs: None,
        raising=False,
    )

    main_module.main_multisession(multisession_collection_only=False)

    assert captured["sessions"]
    assert all(session.sess_id_full.startswith("CT016_") for session in captured["sessions"])
    assert isinstance(captured["mouse"], str)
    assert captured["output_path"].name == "cross_session_analysis"
    assert captured["side_ttc_trial_display_cap"] == 25
    assert captured["summary_grid_trial_display_cap"] == 25
