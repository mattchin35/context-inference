from types import SimpleNamespace

import pandas as pd


from src.tests.test_main_multisession import load_main_module


def test_main_multisession_avoids_legacy_state_features_and_preserves_plot_settings(
    tmp_path,
    monkeypatch,
):
    """Normal full flow avoids legacy state CSVs and retains behavior plot settings.

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
        Asserts that normal full flow never collects legacy learned-model
        features and retains trials-to-correct display caps in its figures.
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
    monkeypatch.setattr(main_module, "resolve_multisession_dates", lambda **kwargs: ["2026-05-11", "2026-05-12"])
    monkeypatch.setattr(main_module, "collect_mouse_block_residual_model_summaries", lambda **kwargs: None)
    monkeypatch.setattr(main_module, "save_and_plot_multisession_choice_window_agreement", lambda **kwargs: None)
    monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *args, **kwargs: None)

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

    assert "sessions" not in captured
    assert captured["side_ttc_trial_display_cap"] == 25
    assert captured["summary_grid_trial_display_cap"] == 25
