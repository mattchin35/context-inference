import importlib.util
import pickle
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
    behavior_analysis_pkg.performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS = {
        "QL": "qlearning_mouse_agreement",
        "FQL": "fql_mouse_agreement",
        "HMM": "hmm_logodds_mouse_agreement",
        "HMM decay": "hmm_logodds_decay_mouse_agreement",
        "Persev": "perseveration_mouse_agreement",
        "Doubt+P": "doubt_perseveration_mouse_agreement",
        "WSLS": "wsls_mouse_agreement",
        "Ideal": "observer_mouse_agreement",
        "Probe+P": "simple_probe_persistence_mouse_agreement",
        "Expect+D+P": "expectancy_persistence_doubt_mouse_agreement",
    }
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


def test_main_uses_canonical_agent_agreement_plot_registry():
    """Single- and multisession workflows should share one agent registry."""
    main_module = load_main_module()

    assert main_module.AGENT_MOUSE_AGREEMENT_COLUMNS == (
        main_module.performance_plots.DEFAULT_AGENT_MOUSE_AGREEMENT_COLUMNS
    )
    assert "Probe+P" in main_module.AGENT_MOUSE_AGREEMENT_COLUMNS
    assert "Expect+D+P" in main_module.AGENT_MOUSE_AGREEMENT_COLUMNS


def test_collect_configured_trial_features_forwards_supplied_params(tmp_path, monkeypatch):
    """The session entrypoint should forward adjustable model parameters."""
    main_module = load_main_module()
    supplied_params = object()
    config = main_module.SingleSessionAnalysisConfig(
        trial_feature_params=supplied_params,
        max_explore_run_length=7,
    )
    session = SimpleNamespace(
        processed_data_path=tmp_path,
        sess_id_full="CT016_test",
    )
    captured = {}

    def fake_collect(frame, **kwargs):
        captured.update(kwargs)
        return frame.copy(), supplied_params

    monkeypatch.setattr(main_module.gtf, "collect_and_save_trial_features", fake_collect, raising=False)

    output, returned_params = main_module.collect_configured_trial_features(
        pd.DataFrame({"action": [1], "reward": [1]}),
        session,
        config,
    )

    assert output.shape[0] == 1
    assert returned_params is supplied_params
    assert captured["params"] is supplied_params
    assert captured["max_explore_run_length"] == 7


def test_save_and_plot_multisession_agent_agreement_uses_existing_plot(
    tmp_path,
    monkeypatch,
):
    """The lightweight alignment path should only summarize and plot agreement."""
    main_module = load_main_module()
    summary = pd.DataFrame({"agent": ["Probe+P"], "agreement_median": [0.6]})
    blocks = pd.DataFrame({"agent": ["Probe+P"], "agreement": [0.6]})
    calls = {}
    monkeypatch.setattr(
        main_module,
        "prepare_agent_mouse_agreement_summary",
        lambda _sessions: summary,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_agent_mouse_agreement_block_points",
        lambda _sessions: blocks,
    )

    def fake_plot(**kwargs):
        calls.update(kwargs)
        return tmp_path / "CT016_agent-mouse-agreement-quality.png"

    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_multisession_agent_mouse_agreement_quality",
        fake_plot,
        raising=False,
    )

    returned_summary, returned_blocks, plot_path = (
        main_module.save_and_plot_multisession_agent_mouse_agreement(
            saved_sessions=[object()],
            output_path=tmp_path,
            mouse="CT016",
            show_raw_blocks=False,
        )
    )

    pd.testing.assert_frame_equal(returned_summary, summary)
    pd.testing.assert_frame_equal(returned_blocks, blocks)
    assert (tmp_path / "CT016_agent_mouse_agreement_summary.csv").exists()
    assert (tmp_path / "CT016_agent_mouse_agreement_block_points.csv").exists()
    assert calls["show_raw_blocks"] is False
    assert plot_path.name == "CT016_agent-mouse-agreement-quality.png"


def test_expectant_session_outputs_call_value_and_component_plots(tmp_path, monkeypatch):
    """Each processed session should receive both complementary tuning plots."""
    main_module = load_main_module()
    session = SimpleNamespace(
        figure_path=tmp_path,
        processed_data_path=tmp_path,
        sess_id_full="CT016_test",
    )
    params = object()
    trial_df = pd.DataFrame({"action": [1], "reward": [1]})
    calls = []

    monkeypatch.setattr(
        main_module.plot_model_values,
        "plot_mouse_history_expectant_switching_values",
        lambda **_kwargs: (trial_df.copy(), tmp_path / "values.png"),
        raising=False,
    )
    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_expectant_switching_diagnostics",
        lambda *_args, **_kwargs: calls.append("diagnostic") or tmp_path / "diagnostic.png",
        raising=False,
    )

    outputs = main_module.plot_expectant_switching_for_session(
        augmented_trial_df=trial_df,
        session=session,
        params=params,
    )

    assert outputs["value_plot"] == tmp_path / "values.png"
    assert outputs["diagnostic_plot"] == tmp_path / "diagnostic.png"
    assert calls == ["diagnostic"]


def test_multisession_agent_agreement_only_returns_before_hmm_work(monkeypatch):
    """Agreement alignment should not require concatenation or HMM execution."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT016_test", date="2026-06-10")
    saved_session = object()
    calls = []
    monkeypatch.setattr(
        main_module,
        "resolve_multisession_dates",
        lambda **_kwargs: ["2026-06-10"],
    )
    monkeypatch.setattr(
        main_module,
        "find_saved_session_by_date",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(
        main_module,
        "load_saved_session_analysis",
        lambda _session: saved_session,
    )
    monkeypatch.setattr(
        main_module,
        "save_and_plot_multisession_agent_mouse_agreement",
        lambda **_kwargs: calls.append("agreement"),
    )
    monkeypatch.setattr(
        main_module,
        "concatenate_saved_sessions",
        lambda _sessions: (_ for _ in ()).throw(AssertionError("HMM path reached")),
    )

    result = main_module.main_multisession(agent_agreement_only=True)

    assert result is None
    assert calls == ["agreement"]


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


def write_raw_session(
    data_root: Path,
    mouse: str,
    date: str,
    timestamp: str,
    task_tag: str = "latent_inference",
) -> Path:
    """Create a raw session folder with one session-info pickle."""
    date_compact = date.replace("-", "")
    sess_id_full = f"{mouse}_{date}_{timestamp}"
    raw_path = data_root / f"{mouse}_{date_compact}_{task_tag}" / "rpi" / sess_id_full
    raw_path.mkdir(parents=True)
    with open(raw_path / f"{sess_id_full}_session_info.pkl", "wb") as file:
        pickle.dump({"session": sess_id_full}, file)
    return raw_path


def block_collection_columns(block_types: list[str], prev_rewards: list[object]) -> dict[str, list[object]]:
    """Return required block collection columns for multisession fixtures."""
    numeric_prev_rewards = pd.to_numeric(pd.Series(prev_rewards), errors="coerce")
    prev_reward_mean = numeric_prev_rewards.dropna().mean()
    centered_rewards = [
        float(value - prev_reward_mean) if not pd.isna(value) else "None"
        for value in numeric_prev_rewards
    ]
    side_codes = []
    for block_type in block_types:
        if str(block_type).startswith("right_"):
            side_codes.append(-0.5)
        elif str(block_type).startswith("left_"):
            side_codes.append(0.5)
        else:
            side_codes.append("None")
    return {
        "block_type": block_types,
        "prev_n_rewarded": prev_rewards,
        "prev_rewards_session_centered": centered_rewards,
        "block_side_code": side_codes,
    }


def test_find_raw_session_by_date_resolves_one_timestamp_with_exact_task_tag(tmp_path):
    """Raw date resolution should use the exact task-tagged session folder."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    write_raw_session(data_root, "CT014", "2025-12-05", "165240", task_tag="latent_inference")
    write_raw_session(data_root, "CT014", "2025-12-05", "111111", task_tag="other_task")

    session = main_module.find_raw_session_by_date(
        mouse="CT014",
        date="2025-12-05",
        session_data_root=data_root,
        task_tag="latent_inference",
        multi_session_save_path=cross_session_path,
    )

    assert session.sess_id_full == "CT014_2025-12-05_165240"
    assert session.sess_id_abbreviated == "CT014_2025-12-05"
    assert session.timestamp == "165240"
    assert session.raw_behavior_folder.name == "CT014_2025-12-05_165240"
    assert session.processed_data_path.name == "processed"
    assert session.figure_path.name == "figures"
    assert session.session_info == {"session": "CT014_2025-12-05_165240"}


def test_find_raw_session_by_date_errors_when_timestamp_is_missing(tmp_path):
    """Missing raw timestamp folders should raise with the requested date."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    (data_root / "CT014_20251205_latent_inference" / "rpi").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="2025-12-05"):
        main_module.find_raw_session_by_date(
            mouse="CT014",
            date="2025-12-05",
            session_data_root=data_root,
            task_tag="latent_inference",
            multi_session_save_path=cross_session_path,
        )


def test_find_raw_session_by_date_errors_when_multiple_timestamps_match(tmp_path):
    """Ambiguous raw timestamp folders should fail loudly with candidate paths."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    first_path = write_raw_session(data_root, "CT014", "2025-12-05", "165240")
    second_path = write_raw_session(data_root, "CT014", "2025-12-05", "171212")

    with pytest.raises(ValueError, match="2025-12-05") as exc_info:
        main_module.find_raw_session_by_date(
            mouse="CT014",
            date="2025-12-05",
            session_data_root=data_root,
            task_tag="latent_inference",
            multi_session_save_path=cross_session_path,
        )

    assert str(first_path) in str(exc_info.value)
    assert str(second_path) in str(exc_info.value)


def test_find_raw_session_by_date_resolves_explicit_timestamp_when_date_is_ambiguous(tmp_path):
    """Explicit timestamps should allow single-session runs on ambiguous dates."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    write_raw_session(data_root, "CT014", "2025-12-05", "165240")
    write_raw_session(data_root, "CT014", "2025-12-05", "171212")

    session = main_module.find_raw_session_by_date(
        mouse="CT014",
        date="2025-12-05",
        session_data_root=data_root,
        task_tag="latent_inference",
        multi_session_save_path=cross_session_path,
        behavior_timestamp="171212",
    )

    assert session.sess_id_full == "CT014_2025-12-05_171212"
    assert session.timestamp == "171212"
    assert session.raw_behavior_folder.name == "CT014_2025-12-05_171212"


def test_find_raw_session_by_date_errors_when_explicit_timestamp_is_missing(tmp_path):
    """Missing explicit timestamp folders should raise with the requested timestamp."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    write_raw_session(data_root, "CT014", "2025-12-05", "165240")

    with pytest.raises(FileNotFoundError, match="171212"):
        main_module.find_raw_session_by_date(
            mouse="CT014",
            date="2025-12-05",
            session_data_root=data_root,
            task_tag="latent_inference",
            multi_session_save_path=cross_session_path,
            behavior_timestamp="171212",
        )


def test_find_raw_session_dates_for_task_tag_returns_sorted_exact_matches(tmp_path):
    """Date discovery should scan direct mouse-root children for exact task folders."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    data_root.mkdir()
    (data_root / "CT014_20251216_latent_inference").mkdir()
    (data_root / "CT014_20251205_latent_inference").mkdir()
    (data_root / "CT014_20251204_other_task").mkdir()
    (data_root / "CT999_20251201_latent_inference").mkdir()

    dates = main_module.find_raw_session_dates_for_task_tag(
        mouse="CT014",
        session_data_root=data_root,
        task_tag="latent_inference",
    )

    assert dates == ["2025-12-05", "2025-12-16"]


def test_find_raw_session_dates_for_task_tag_ignores_non_matching_folder_names(tmp_path):
    """Unrelated root-level folders should not be treated as session dates."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    data_root.mkdir()
    (data_root / "cross_session_analysis").mkdir()
    (data_root / "CT014_20251205_latent_inference_backup").mkdir()
    (data_root / "CT014_2025-12-05_latent_inference").mkdir()
    (data_root / "CT014_20251205_latent_inference").mkdir()

    dates = main_module.find_raw_session_dates_for_task_tag(
        mouse="CT014",
        session_data_root=data_root,
        task_tag="latent_inference",
    )

    assert dates == ["2025-12-05"]


def test_find_raw_session_dates_for_task_tag_errors_when_no_matches(tmp_path):
    """Missing task-tagged session folders should raise with the task tag."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    data_root.mkdir()
    (data_root / "CT014_20251205_other_task").mkdir()

    with pytest.raises(FileNotFoundError, match="latent_inference"):
        main_module.find_raw_session_dates_for_task_tag(
            mouse="CT014",
            session_data_root=data_root,
            task_tag="latent_inference",
        )


def test_run_single_session_batch_resolves_dates_and_calls_runner(tmp_path):
    """Batch single-session analysis should call the runner once per resolved date."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    cross_session_path = data_root / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    write_raw_session(data_root, "CT014", "2025-12-05", "165240")
    write_raw_session(data_root, "CT014", "2025-12-16", "153200")
    captured = []

    def fake_runner(session, config):
        captured.append((session.sess_id_full, config.preprocess_raw_session))

    config = main_module.SingleSessionAnalysisConfig(preprocess_raw_session=False)

    main_module.run_single_session_batch(
        mouse="CT014",
        dates=["2025-12-05", "2025-12-16"],
        task_tag="latent_inference",
        session_data_root=data_root,
        multi_session_save_path=cross_session_path,
        config=config,
        runner=fake_runner,
    )

    assert captured == [
        ("CT014_2025-12-05_165240", False),
        ("CT014_2025-12-16_153200", False),
    ]


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


def test_resolve_multisession_dates_returns_explicit_dates_without_discovery(tmp_path):
    """Explicit multisession dates should be preserved when discovery is off."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    data_root.mkdir()

    dates = main_module.resolve_multisession_dates(
        mouse="CT014",
        session_data_root=data_root,
        task_tag="latent_inference",
        explicit_dates=["2025-12-16", "2025-12-05"],
        use_all_dates_for_task_tag=False,
        require_saved_outputs=True,
    )

    assert dates == ["2025-12-16", "2025-12-05"]


def test_resolve_multisession_dates_discovers_task_dates_and_validates_saved_outputs(tmp_path):
    """Discovered multisession dates should require matching saved outputs."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    write_saved_session(
        data_root=data_root,
        mouse="CT014",
        date="2025-12-16",
        timestamp="153200",
        block_df=pd.DataFrame({"block_ix": [0], "trials_to_correct": [1]}),
        trial_df=pd.DataFrame({"cur_block": [0], "action": [1]}),
        directory_suffix="_latent_inference",
    )
    write_saved_session(
        data_root=data_root,
        mouse="CT014",
        date="2025-12-05",
        timestamp="165240",
        block_df=pd.DataFrame({"block_ix": [0], "trials_to_correct": [2]}),
        trial_df=pd.DataFrame({"cur_block": [0], "action": [0]}),
        directory_suffix="_latent_inference",
    )

    dates = main_module.resolve_multisession_dates(
        mouse="CT014",
        session_data_root=data_root,
        task_tag="latent_inference",
        explicit_dates=["2025-12-01"],
        use_all_dates_for_task_tag=True,
        require_saved_outputs=True,
    )

    assert dates == ["2025-12-05", "2025-12-16"]


def test_resolve_multisession_dates_errors_when_discovered_date_lacks_saved_outputs(tmp_path):
    """Discovered raw dates without saved analysis outputs should fail loudly."""
    main_module = load_main_module()
    data_root = tmp_path / "CT014"
    (data_root / "CT014_20251205_latent_inference").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="2025-12-05"):
        main_module.resolve_multisession_dates(
            mouse="CT014",
            session_data_root=data_root,
            task_tag="latent_inference",
            explicit_dates=[],
            use_all_dates_for_task_tag=True,
            require_saved_outputs=True,
        )


def test_concatenate_saved_sessions_offsets_block_and_trial_block_ids(tmp_path):
    """Continuous concat should offset block ids while preserving source ids."""
    main_module = load_main_module()
    first_session = SimpleNamespace(
        mouse="CT014",
        sess_id_full="CT014_2025-12-05_165240",
        date="2025-12-05",
    )
    second_session = SimpleNamespace(
        mouse="CT014",
        sess_id_full="CT014_2025-12-16_153200",
        date="2025-12-16",
    )
    first = main_module.SavedSessionAnalysis(
        session=first_session,
        block_performance=pd.DataFrame(
            {
                "block_ix": [0, 1],
                "value": [10, 11],
                **block_collection_columns(
                    ["right_uncued", "left_uncued"],
                    [0, 2],
                ),
            }
        ),
        augmented_trial_df=pd.DataFrame(
            {"cur_trial": [0, 1, 2], "cur_block": [0, 1, 1], "action": [1, 0, 0]}
        ),
    )
    second = main_module.SavedSessionAnalysis(
        session=second_session,
        block_performance=pd.DataFrame(
            {
                "block_ix": [0, 1, 2],
                "value": [20, 21, 22],
                **block_collection_columns(
                    ["right_uncued", "right_uncued", "left_uncued"],
                    [4, 6, "None"],
                ),
            }
        ),
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
    assert concatenated.block_performance["mouse"].tolist() == ["CT014"] * 5
    assert concatenated.block_performance["source_mouse"].tolist() == ["CT014"] * 5
    assert concatenated.augmented_trial_df["mouse"].tolist() == ["CT014"] * 7
    assert concatenated.augmented_trial_df["source_mouse"].tolist() == ["CT014"] * 7
    assert concatenated.block_performance["prev_rewards_session_centered"].tolist() == [
        -1.0,
        1.0,
        -1.0,
        1.0,
        "None",
    ]
    assert concatenated.block_performance["prev_rewards_mouse_centered"].tolist() == [
        -3.0,
        -1.0,
        1.0,
        3.0,
        "None",
    ]
    assert concatenated.block_performance["block_side_code"].tolist() == [
        -0.5,
        0.5,
        -0.5,
        -0.5,
        0.5,
    ]


def test_concatenate_saved_sessions_uses_raw_trial_block_ids_as_source_blocks(tmp_path):
    """Saved block rows may be renumbered while trial rows keep raw block ids."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-23_163505", date="2025-12-23")
    saved = main_module.SavedSessionAnalysis(
        session=session,
        block_performance=pd.DataFrame(
            {
                "block_ix": [0, 1, 2],
                "value": [10, 11, 12],
                **block_collection_columns(
                    ["right_uncued", "left_uncued", "right_uncued"],
                    [0, 1, 2],
                ),
            }
        ),
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
        block_performance=pd.DataFrame(
            {
                "block_ix": [0, 1],
                "value": [10, 11],
                **block_collection_columns(
                    ["right_uncued", "left_uncued"],
                    [0, 1],
                ),
            }
        ),
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


def test_concatenate_saved_sessions_rejects_missing_collection_columns(tmp_path):
    """Multisession concat should require current single-session block variables."""
    main_module = load_main_module()
    session = SimpleNamespace(
        mouse="CT014",
        sess_id_full="CT014_2025-12-23_163505",
        date="2025-12-23",
    )
    saved = main_module.SavedSessionAnalysis(
        session=session,
        block_performance=pd.DataFrame(
            {
                "block_ix": [0, 1],
                "block_type": ["right_uncued", "left_uncued"],
                "prev_n_rewarded": [0, 1],
            }
        ),
        augmented_trial_df=pd.DataFrame({"cur_trial": [0, 1], "cur_block": [0, 1]}),
    )

    with pytest.raises(ValueError, match="prev_rewards_session_centered.*block_side_code"):
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


def test_discover_mice_with_overall_performance_finds_valid_mouse_folders(tmp_path):
    """Cross-mouse discovery should return mouse folders with matching summary CSVs."""
    main_module = load_main_module()
    for mouse in ("CT014", "CT016"):
        cross_session_path = tmp_path / mouse / "cross_session_analysis"
        cross_session_path.mkdir(parents=True)
        pd.DataFrame({"date": ["2025-12-05"]}).to_csv(
            cross_session_path / f"{mouse}_overall_performance.csv",
            index=False,
        )
    (tmp_path / "cross_mouse_analysis").mkdir()
    mismatched_path = tmp_path / "CT999" / "cross_session_analysis"
    mismatched_path.mkdir(parents=True)
    pd.DataFrame({"date": ["2025-12-05"]}).to_csv(
        mismatched_path / "wrong_name_overall_performance.csv",
        index=False,
    )

    mice = main_module.discover_mice_with_overall_performance(tmp_path)

    assert mice == ["CT014", "CT016"]


def test_load_cross_mouse_learning_curve_data_preserves_training_day_before_dropping_invalid_slopes(tmp_path):
    """Invalid slopes should create gaps instead of renumbering training days."""
    main_module = load_main_module()
    ct014_path = tmp_path / "CT014" / "cross_session_analysis"
    ct014_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-12-03", "2025-12-01", "2025-12-02"],
            "prev_n_rewarded_slope": [0.3, 0.1, "None"],
        }
    ).to_csv(ct014_path / "CT014_overall_performance.csv", index=False)
    ct016_path = tmp_path / "CT016" / "cross_session_analysis"
    ct016_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-02"],
            "prev_n_rewarded_slope": [0.2, 0.4],
        }
    ).to_csv(ct016_path / "CT016_overall_performance.csv", index=False)

    cross_mouse_df = main_module.load_cross_mouse_learning_curve_data(
        data_root=tmp_path,
        mice=["CT014", "CT016"],
        learning_regressor="prev_n_rewarded",
    )

    assert cross_mouse_df["mouse"].tolist() == ["CT014", "CT014", "CT016", "CT016"]
    assert cross_mouse_df["date"].tolist() == [
        "2025-12-01",
        "2025-12-03",
        "2025-12-01",
        "2025-12-02",
    ]
    assert cross_mouse_df["training_day"].tolist() == [1, 3, 1, 2]
    assert cross_mouse_df["slope"].tolist() == [0.1, 0.3, 0.2, 0.4]
    assert cross_mouse_df["learning_regressor"].unique().tolist() == ["prev_n_rewarded"]


def test_load_cross_mouse_learning_curve_data_fails_loudly_when_regressor_missing(tmp_path):
    """Cross-mouse loading should fail when a mouse lacks the requested slope column."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "CT014" / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-12-01"],
            "prev_consecutive_rewards_slope": [0.1],
        }
    ).to_csv(cross_session_path / "CT014_overall_performance.csv", index=False)

    with pytest.raises(ValueError, match="CT014.*prev_n_rewarded_slope"):
        main_module.load_cross_mouse_learning_curve_data(
            data_root=tmp_path,
            mice=["CT014"],
            learning_regressor="prev_n_rewarded",
        )


def test_load_cross_mouse_session_metric_data_preserves_training_day_before_dropping_invalid_rows(tmp_path):
    """Invalid metric rows should leave training-day gaps, matching learning curves."""
    main_module = load_main_module()
    ct014_path = tmp_path / "CT014" / "cross_session_analysis"
    ct014_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-12-03", "2025-12-01", "2025-12-02"],
            "median_TTS": [3, 1, "None"],
        }
    ).to_csv(ct014_path / "CT014_overall_performance.csv", index=False)
    ct016_path = tmp_path / "CT016" / "cross_session_analysis"
    ct016_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-12-01", "2025-12-02"],
            "median_TTS": [2, 4],
        }
    ).to_csv(ct016_path / "CT016_overall_performance.csv", index=False)

    cross_mouse_df = main_module.load_cross_mouse_session_metric_data(
        data_root=tmp_path,
        mice=["CT014", "CT016"],
        metric_column="median_TTS",
        metric_label="Median Trials to Correct",
    )

    assert cross_mouse_df["mouse"].tolist() == ["CT014", "CT014", "CT016", "CT016"]
    assert cross_mouse_df["date"].tolist() == [
        "2025-12-01",
        "2025-12-03",
        "2025-12-01",
        "2025-12-02",
    ]
    assert cross_mouse_df["training_day"].tolist() == [1, 3, 1, 2]
    assert cross_mouse_df["metric_value"].tolist() == [1, 3, 2, 4]
    assert cross_mouse_df["metric_column"].unique().tolist() == ["median_TTS"]
    assert cross_mouse_df["metric_label"].unique().tolist() == ["Median Trials to Correct"]


def test_load_cross_mouse_session_metric_data_fails_loudly_when_metric_missing(tmp_path):
    """Cross-mouse metric loading should fail when a mouse lacks the requested column."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "CT014" / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "date": ["2025-12-01"],
            "median_post_switch_correct": [0.75],
        }
    ).to_csv(cross_session_path / "CT014_overall_performance.csv", index=False)

    with pytest.raises(ValueError, match="CT014.*median_TTS"):
        main_module.load_cross_mouse_session_metric_data(
            data_root=tmp_path,
            mice=["CT014"],
            metric_column="median_TTS",
            metric_label="Median Trials to Correct",
        )


def test_load_cross_mouse_multisession_block_performance_fills_legacy_mouse_metadata(tmp_path):
    """Cross-mouse block loading should tag old multisession CSVs by selected mouse."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "CT014" / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "source_session_id": ["CT014_2025-12-05_165240", "CT014_2025-12-05_165240"],
            "source_date": ["2025-12-05", "2025-12-05"],
            "block_ix": [0, 1],
            "block_type": ["right_uncued", "left_uncued"],
            "prev_n_rewarded": [0, 2],
            "prev_rewards_session_centered": [-1.0, 1.0],
            "prev_rewards_mouse_centered": [-1.0, 1.0],
            "block_side_code": [-0.5, 0.5],
            "trials_to_correct": [1, 2],
        }
    ).to_csv(cross_session_path / "CT014_multisession_block_performance.csv", index=False)

    cross_mouse_df = main_module.load_cross_mouse_multisession_block_performance(
        data_root=tmp_path,
        mice=["CT014"],
    )

    assert cross_mouse_df["mouse"].tolist() == ["CT014", "CT014"]
    assert cross_mouse_df["source_mouse"].tolist() == ["CT014", "CT014"]
    assert cross_mouse_df["session_id"].tolist() == [
        "CT014_2025-12-05_165240",
        "CT014_2025-12-05_165240",
    ]
    assert cross_mouse_df["mouse_order"].tolist() == [0, 0]
    assert cross_mouse_df["prev_rewards_global_centered"].tolist() == [-1.0, 1.0]


def test_collect_cross_mouse_multisession_block_performance_writes_combined_csv(tmp_path):
    """Collector should save one concatenated multisession block CSV."""
    main_module = load_main_module()
    for mouse, ttc_value in (("CT014", 1), ("CT016", 2)):
        cross_session_path = tmp_path / mouse / "cross_session_analysis"
        cross_session_path.mkdir(parents=True)
        pd.DataFrame(
            {
                "mouse": [mouse],
                "source_mouse": [mouse],
                "source_session_id": [f"{mouse}_2025-12-05_165240"],
                "source_date": ["2025-12-05"],
                "block_ix": [0],
                "block_type": ["right_uncued"],
                "prev_n_rewarded": [ttc_value],
                "prev_rewards_session_centered": [0.0],
                "prev_rewards_mouse_centered": [0.0],
                "block_side_code": [-0.5],
                "trials_to_correct": [ttc_value],
            }
        ).to_csv(cross_session_path / f"{mouse}_multisession_block_performance.csv", index=False)

    saved_path = main_module.collect_cross_mouse_multisession_block_performance(
        data_root=tmp_path,
        output_path=tmp_path / "cross_mouse_analysis",
        mice=["CT014", "CT016"],
    )

    saved_df = pd.read_csv(saved_path, na_filter=False)
    assert saved_path == (
        tmp_path
        / "cross_mouse_analysis"
        / "cross_mouse_multisession_block_performance.csv"
    )
    assert saved_df["mouse"].tolist() == ["CT014", "CT016"]
    assert saved_df["trials_to_correct"].tolist() == [1, 2]
    assert saved_df["prev_rewards_global_centered"].tolist() == [-0.5, 0.5]


def test_load_cross_mouse_multisession_block_performance_requires_centered_columns(tmp_path):
    """Cross-mouse collection should fail until mouse multisession CSVs are rerun."""
    main_module = load_main_module()
    cross_session_path = tmp_path / "CT014" / "cross_session_analysis"
    cross_session_path.mkdir(parents=True)
    pd.DataFrame(
        {
            "source_session_id": ["CT014_2025-12-05_165240"],
            "source_date": ["2025-12-05"],
            "block_ix": [0],
            "block_type": ["right_uncued"],
            "prev_n_rewarded": [0],
        }
    ).to_csv(cross_session_path / "CT014_multisession_block_performance.csv", index=False)

    with pytest.raises(ValueError, match="prev_rewards_session_centered.*prev_rewards_mouse_centered.*block_side_code"):
        main_module.load_cross_mouse_multisession_block_performance(
            data_root=tmp_path,
            mice=["CT014"],
        )


def test_main_cross_mouse_multisession_block_performance_uses_cross_mouse_output(monkeypatch):
    """The entry point should write into the cross-mouse analysis folder."""
    main_module = load_main_module()
    calls = []

    def record_collect(**kwargs):
        calls.append(kwargs)
        return kwargs["output_path"] / "cross_mouse_multisession_block_performance.csv"

    monkeypatch.setattr(
        main_module,
        "collect_cross_mouse_multisession_block_performance",
        record_collect,
    )

    saved_path = main_module.main_cross_mouse_multisession_block_performance(
        data_root=Path("/tmp/contextProjectData"),
        mice=["CT014", "CT016"],
    )

    assert calls[0]["data_root"] == Path("/tmp/contextProjectData")
    assert calls[0]["output_path"] == Path("/tmp/contextProjectData/cross_mouse_analysis")
    assert calls[0]["mice"] == ["CT014", "CT016"]
    assert saved_path == Path(
        "/tmp/contextProjectData/cross_mouse_analysis/cross_mouse_multisession_block_performance.csv"
    )


def test_load_cross_mouse_multisession_block_performance_fails_for_missing_mouse_csv(tmp_path):
    """Selected mice without multisession block CSVs should fail loudly."""
    main_module = load_main_module()

    with pytest.raises(FileNotFoundError, match="CT014.*multisession block performance"):
        main_module.load_cross_mouse_multisession_block_performance(
            data_root=tmp_path,
            mice=["CT014"],
        )


def test_run_cross_mouse_session_metric_curves_writes_metric_csvs_and_plots(tmp_path, monkeypatch):
    """The cross-mouse metric runner should save one CSV and plot per requested metric."""
    main_module = load_main_module()
    for mouse, tts_values, accuracy_values in (
        ("CT014", [1, "None", 3], [0.5, 0.75, "None"]),
        ("CT016", [2, 4, 6], [0.6, 0.8, 0.9]),
    ):
        cross_session_path = tmp_path / mouse / "cross_session_analysis"
        cross_session_path.mkdir(parents=True)
        pd.DataFrame(
            {
                "date": ["2025-12-01", "2025-12-02", "2025-12-03"],
                "median_TTS": tts_values,
                "median_post_switch_correct": accuracy_values,
            }
        ).to_csv(cross_session_path / f"{mouse}_overall_performance.csv", index=False)

    plot_calls = []

    def record_plot(**kwargs):
        plot_calls.append(kwargs)
        return kwargs["plot_path"] / f"{kwargs['figure_id']}_{kwargs['metric_column']}_session_metric_curve.png"

    monkeypatch.setattr(
        main_module.performance_plots,
        "plot_cross_mouse_session_metric_curve",
        record_plot,
        raising=False,
    )

    main_module.run_cross_mouse_session_metric_curves(
        data_root=tmp_path,
        output_path=tmp_path / "cross_mouse_analysis",
        mice=["CT014", "CT016"],
        metric_specs={
            "median_TTS": {"label": "Median Trials to Correct"},
            "median_post_switch_correct": {
                "label": "Median Post-First-Correct Accuracy",
                "ylim": (0, 1),
            },
        },
    )

    assert len(plot_calls) == 2
    assert (tmp_path / "cross_mouse_analysis" / "cross_mouse_median_TTS_session_metric_curve_data.csv").exists()
    assert (
        tmp_path
        / "cross_mouse_analysis"
        / "cross_mouse_median_post_switch_correct_session_metric_curve_data.csv"
    ).exists()
    assert plot_calls[0]["metric_column"] == "median_TTS"
    assert plot_calls[0]["metric_label"] == "Median Trials to Correct"
    assert plot_calls[1]["metric_column"] == "median_post_switch_correct"
    assert plot_calls[1]["ylim"] == (0, 1)


def test_run_or_collect_multisession_collection_only_overwrites_inputs_and_skips_hmm(
    tmp_path,
    monkeypatch,
    capsys,
):
    """Collection-only mode should overwrite multisession CSVs without HMM calls."""
    main_module = load_main_module()
    concatenated = SimpleNamespace(
        block_performance=pd.DataFrame({"block_ix": [0], "prev_n_rewarded": [1]}),
        augmented_trial_df=pd.DataFrame({"cur_block": [0], "action": [1]}),
    )
    multisession = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
        figure_path=tmp_path,
    )
    calls = []

    def fail_hmm(**_kwargs):
        raise AssertionError("HMM should not run in collection-only mode")

    monkeypatch.setattr(main_module, "run_multisession_analysis", fail_hmm)

    result = main_module.run_or_collect_multisession_analysis(
        concatenated=concatenated,
        session=multisession,
        collection_only=True,
    )

    saved_block_df = pd.read_csv(tmp_path / "CT014_multisession_block_performance.csv")
    saved_trial_df = pd.read_csv(tmp_path / "CT014_multisession_augmented_trials.csv")
    pd.testing.assert_frame_equal(saved_block_df, concatenated.block_performance)
    pd.testing.assert_frame_equal(saved_trial_df, concatenated.augmented_trial_df)
    assert result is None
    assert "collection-only mode overwrote multisession block/trial CSVs" in capsys.readouterr().out
    assert calls == []


def test_run_or_collect_multisession_modeling_path_calls_hmm(monkeypatch):
    """Normal multisession mode should still call the HMM orchestration."""
    main_module = load_main_module()
    concatenated = SimpleNamespace()
    multisession = SimpleNamespace(sess_id_full="CT014_multisession")
    expected = (None, None, pd.DataFrame({"block_ix": [0]}), pd.DataFrame({"cur_block": [0]}))
    captured = {}

    def record_hmm(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(main_module, "run_multisession_analysis", record_hmm)

    result = main_module.run_or_collect_multisession_analysis(
        concatenated=concatenated,
        session=multisession,
        collection_only=False,
        block_num_states=3,
        trial_num_states=4,
        prior_alpha=1,
        prior_sigma=2,
    )

    assert result == expected
    assert captured["concatenated"] is concatenated
    assert captured["session"] is multisession
    assert captured["block_num_states"] == 3
    assert captured["trial_num_states"] == 4


def test_main_multisession_collection_only_saves_and_skips_plot_and_hmm_work(
    tmp_path,
    monkeypatch,
):
    """Collection-only main flow should save concatenated CSVs and return early."""
    main_module = load_main_module()
    session = SimpleNamespace(
        mouse="CT017",
        sess_id_full="CT017_2026-04-17_131606",
        date="2026-04-17",
    )
    concatenated = SimpleNamespace(
        block_performance=pd.DataFrame({"block_ix": [0], "prev_n_rewarded": [2]}),
        augmented_trial_df=pd.DataFrame({"cur_block": [0], "action": [1]}),
    )
    multisession = SimpleNamespace(
        sess_id_full="CT017_multisession",
        processed_data_path=tmp_path,
        figure_path=tmp_path,
    )

    monkeypatch.setattr(
        main_module,
        "resolve_multisession_dates",
        lambda **_kwargs: ["2026-04-17"],
    )
    monkeypatch.setattr(
        main_module,
        "find_saved_session_by_date",
        lambda **_kwargs: session,
    )
    monkeypatch.setattr(
        main_module,
        "load_saved_session_analysis",
        lambda loaded_session: SimpleNamespace(session=loaded_session),
    )
    monkeypatch.setattr(
        main_module,
        "concatenate_saved_sessions",
        lambda saved_sessions: concatenated,
    )
    monkeypatch.setattr(
        main_module,
        "build_multisession_session",
        lambda **_kwargs: multisession,
    )
    monkeypatch.setattr(
        main_module,
        "prepare_learning_curve_data",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("plot setup should be skipped")),
    )
    monkeypatch.setattr(
        main_module.bssm,
        "collect_block_hmm_state_features_for_sessions",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("HMM feature collection should be skipped")),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "run_multisession_analysis",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("HMM modeling should be skipped")),
    )

    main_module.main_multisession()

    saved_block_df = pd.read_csv(tmp_path / "CT017_multisession_block_performance.csv")
    saved_trial_df = pd.read_csv(tmp_path / "CT017_multisession_augmented_trials.csv")
    pd.testing.assert_frame_equal(saved_block_df, concatenated.block_performance)
    pd.testing.assert_frame_equal(saved_trial_df, concatenated.augmented_trial_df)


def test_main_cross_mouse_metrics_runs_learning_and_session_metrics(monkeypatch):
    """The user-facing cross-mouse entry point should run all cross-mouse metric plots."""
    main_module = load_main_module()
    calls = []

    monkeypatch.setattr(main_module, "discover_mice_with_overall_performance", lambda _root: ["CT014"])
    monkeypatch.setattr(
        main_module,
        "run_cross_mouse_learning_curve",
        lambda **kwargs: calls.append(("learning", kwargs)),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "run_cross_mouse_session_metric_curves",
        lambda **kwargs: calls.append(("session", kwargs)),
    )
    monkeypatch.setattr(
        main_module,
        "collect_cross_mouse_multisession_block_performance",
        lambda **kwargs: calls.append(("blocks", kwargs)),
    )
    monkeypatch.setattr(
        main_module,
        "collect_cross_mouse_block_residual_model_summaries",
        lambda **kwargs: calls.append(("residual_models", kwargs)),
    )

    main_module.main_cross_mouse_metrics(
        data_root=Path("/tmp/contextProjectData"),
        use_discovered_mice=True,
        mice=[],
    )

    assert [call[0] for call in calls] == ["learning", "session", "blocks", "residual_models"]
    assert "median_ideal_agreement" in calls[1][1]["metric_specs"]
    assert calls[2][1]["mice"] == ["CT014"]
    assert calls[3][1]["mice"] == ["CT014"]


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

    assert points_df["block_ix"].tolist() == [0, 0, 1, 1, 3, 3]
    assert points_df["rewarded_side"].tolist() == ["overall", "right", "overall", "left", "overall", "left"]
    assert points_df["trials_to_correct_numeric"].tolist()[:1] == [2.0]
    assert np.isnan(points_df.loc[2, "trials_to_correct_numeric"])
    assert points_df["no_correct_choice"].tolist() == [False, False, True, True, False, False]


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
    overall_first = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["rewarded_side"] == "overall")
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
    assert overall_first["n_blocks"] == 4
    assert overall_first["n_valid_blocks"] == 3
    assert overall_first["n_no_correct_blocks"] == 1
    assert overall_first["completion_fraction"] == pytest.approx(3 / 4)
    assert overall_first["trials_to_correct_median"] == 2.0
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

    assert summary_df["rewarded_side"].tolist() == ["overall", "left"]
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


def test_add_block_explore_counts_overwrites_stale_counts_from_augmented_trials():
    """Trial explore tags should take priority over stale saved block counts."""
    main_module = load_main_module()
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["left_cued", "right_uncued"],
            "n_explore_trials": [0, 0],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [5, 5, 6, 6, 6],
            "explore_trial": [True, False, "true", "0", 1],
        }
    )

    updated_block_performance = main_module.add_block_explore_counts_from_trials(
        block_performance,
        augmented_trial_df,
    )

    assert updated_block_performance["n_explore_trials"].tolist() == [1, 2]


def test_add_block_explore_run_counts_overwrites_stale_counts_from_augmented_trials():
    """Trial explore-run starts should take priority over stale saved block counts."""
    main_module = load_main_module()
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["left_cued", "right_uncued"],
            "n_explore_runs": [0, 0],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [5, 5, 6, 6, 6],
            "explore_run_start": [True, False, "true", "0", 1],
        }
    )

    updated_block_performance = main_module.add_block_explore_run_counts_from_trials(
        block_performance,
        augmented_trial_df,
    )

    assert updated_block_performance["n_explore_runs"].tolist() == [1, 2]


def test_add_block_explore_run_counts_filters_by_minimum_run_length():
    """Explore-run counts should optionally ignore one-trial runs."""
    main_module = load_main_module()
    block_performance = pd.DataFrame(
        {
            "block_ix": [0, 1],
            "block_type": ["left_cued", "right_uncued"],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [5, 5, 6, 6, 6],
            "explore_run_start": [True, True, True, False, True],
            "explore_run_length": [1, 2, 1, "None", 3],
        }
    )

    updated_block_performance = main_module.add_block_explore_run_counts_from_trials(
        block_performance,
        augmented_trial_df,
        min_explore_run_length_to_count=2,
    )

    assert updated_block_performance["n_explore_runs"].tolist() == [1, 1]


def test_add_block_explore_run_counts_requires_lengths_for_filtered_counts():
    """Filtering by run length should fail if run lengths were not saved."""
    main_module = load_main_module()
    block_performance = pd.DataFrame(
        {
            "block_ix": [0],
            "block_type": ["left_cued"],
        }
    )
    augmented_trial_df = pd.DataFrame(
        {
            "cur_block": [5, 5],
            "explore_run_start": [True, False],
        }
    )

    with pytest.raises(ValueError, match="explore_run_length"):
        main_module.add_block_explore_run_counts_from_trials(
            block_performance,
            augmented_trial_df,
            min_explore_run_length_to_count=2,
        )


def test_ensure_block_agent_mouse_agreement_columns_regenerates_missing_columns(monkeypatch):
    """Missing block agent-agreement columns should be regenerated from trial features."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_session = main_module.SavedSessionAnalysis(
        session=session,
        block_performance=pd.DataFrame(
            {
                "block_ix": [0, 1],
                "block_type": ["left_cued", "right_cued"],
            }
        ),
        augmented_trial_df=pd.DataFrame({"cur_block": [0, 1]}),
    )

    def fake_add_block_agent_mouse_agreement_columns(block_performance, augmented_trial_df):
        assert augmented_trial_df is saved_session.augmented_trial_df
        return block_performance.assign(
            qlearning_mouse_agreement=[0.5, 1.0],
            fql_mouse_agreement=[0.25, 0.75],
            hmm_logodds_mouse_agreement=[1.0, 0.0],
            hmm_logodds_decay_mouse_agreement=[0.0, 1.0],
            perseveration_mouse_agreement=[0.5, 0.25],
            doubt_perseveration_mouse_agreement=[0.75, 0.25],
            wsls_mouse_agreement=[1.0, 0.5],
            observer_mouse_agreement=[0.5, 0.5],
            simple_probe_persistence_mouse_agreement=[0.5, 0.75],
            expectancy_persistence_doubt_mouse_agreement=[0.75, 1.0],
        )

    monkeypatch.setattr(
        main_module.session_analysis,
        "add_block_agent_mouse_agreement_columns",
        fake_add_block_agent_mouse_agreement_columns,
        raising=False,
    )

    updated = main_module.ensure_block_agent_mouse_agreement_columns(saved_session)

    assert updated.block_performance["qlearning_mouse_agreement"].tolist() == [0.5, 1.0]
    assert updated.block_performance["wsls_mouse_agreement"].tolist() == [1.0, 0.5]
    assert updated.block_performance["observer_mouse_agreement"].tolist() == [0.5, 0.5]


def test_prepare_agent_mouse_agreement_block_points_long_format():
    """Agent agreement block points should use one row per valid block and agent."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1],
                    "block_type": ["left_cued", "right_uncued"],
                    "qlearning_mouse_agreement": [0.5, "None"],
                    "fql_mouse_agreement": [0.25, 0.75],
                    "hmm_logodds_mouse_agreement": [1.0, 0.0],
                    "hmm_logodds_decay_mouse_agreement": [0.0, 1.0],
                    "perseveration_mouse_agreement": [0.25, 0.5],
                    "doubt_perseveration_mouse_agreement": [0.75, 0.25],
                    "wsls_mouse_agreement": [1.0, "None"],
                    "observer_mouse_agreement": [0.5, 0.5],
                    "simple_probe_persistence_mouse_agreement": [0.5, 0.75],
                    "expectancy_persistence_doubt_mouse_agreement": [0.75, 1.0],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0, 1]}),
        )
    ]

    points_df = main_module.prepare_agent_mouse_agreement_block_points(saved_sessions)

    assert points_df.columns.tolist() == [
        "date",
        "session_id",
        "agent",
        "block_ix",
        "block_type",
        "agreement",
    ]
    ql_points = points_df[points_df["agent"] == "QL"]
    assert ql_points["agreement"].tolist() == [0.5]
    ideal_points = points_df[points_df["agent"] == "Ideal"]
    assert ideal_points["agreement"].tolist() == [0.5, 0.5]
    wsls_points = points_df[points_df["agent"] == "WSLS"]
    assert wsls_points["agreement"].tolist() == [1.0]


def test_prepare_agent_mouse_agreement_summary_calculates_session_agent_quartiles():
    """Agent agreement summaries should calculate Q1/median/Q3 per session and agent."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2],
                    "block_type": ["left_cued", "right_uncued", "left_uncued"],
                    "qlearning_mouse_agreement": [0.0, 0.5, 1.0],
                    "fql_mouse_agreement": [0.25, 0.75, 1.0],
                    "hmm_logodds_mouse_agreement": [1.0, 0.0, 0.5],
                    "hmm_logodds_decay_mouse_agreement": [0.0, 1.0, 0.5],
                    "perseveration_mouse_agreement": [0.25, 0.5, 0.75],
                    "doubt_perseveration_mouse_agreement": [0.0, 0.5, 1.0],
                    "wsls_mouse_agreement": [1.0, 0.5, 0.0],
                    "observer_mouse_agreement": [0.5, 0.5, 1.0],
                    "simple_probe_persistence_mouse_agreement": [0.25, 0.5, 0.75],
                    "expectancy_persistence_doubt_mouse_agreement": [0.0, 0.5, 1.0],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0, 1, 2]}),
        )
    ]

    summary_df = main_module.prepare_agent_mouse_agreement_summary(saved_sessions)
    ql_summary = summary_df[summary_df["agent"] == "QL"].iloc[0]

    assert ql_summary["n_blocks"] == 3
    assert ql_summary["agreement_q1"] == 0.25
    assert ql_summary["agreement_median"] == 0.5
    assert ql_summary["agreement_q3"] == 0.75
    assert {"Persev", "Doubt+P", "WSLS"}.issubset(set(summary_df["agent"]))


def test_prepare_block_explore_run_summary_calculates_overall_and_side_medians():
    """Explore-run summaries should use raw n_explore_runs for overall and side groups."""
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
                    "n_explore_runs": [1, 3, 2, 0],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
        main_module.SavedSessionAnalysis(
            session=second_session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1, 2],
                    "block_type": ["right_cued", "left_cued", "right_uncued"],
                    "n_explore_runs": [5, 1, 7],
                }
            ),
            augmented_trial_df=pd.DataFrame({"cur_block": [0]}),
        ),
    ]

    points_df = main_module.prepare_block_explore_run_block_points(saved_sessions)
    summary_df = main_module.prepare_block_explore_run_summary(saved_sessions)

    first_overall = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["explore_group"] == "overall")
    ].iloc[0]
    first_left = summary_df[
        (summary_df["date"] == "2025-12-05") & (summary_df["explore_group"] == "left")
    ].iloc[0]
    second_right = summary_df[
        (summary_df["date"] == "2025-12-16") & (summary_df["explore_group"] == "right")
    ].iloc[0]

    overall_points = points_df[points_df["explore_group"] == "overall"]
    assert overall_points["n_explore_runs"].tolist() == [1.0, 3.0, 2.0, 5.0, 1.0, 7.0]
    assert first_overall["n_blocks"] == 3
    assert first_overall["n_explore_runs_median"] == 2.0
    assert first_overall["n_explore_runs_q1"] == 1.5
    assert first_overall["n_explore_runs_q3"] == 2.5
    assert first_left["n_blocks"] == 2
    assert first_left["n_explore_runs_median"] == 2.0
    assert second_right["n_explore_runs_median"] == 6.0


def test_prepare_block_explore_run_summary_filters_by_minimum_run_length():
    """Explore-run summaries should apply the requested minimum run length."""
    main_module = load_main_module()
    session = SimpleNamespace(sess_id_full="CT014_2025-12-05_165240", date="2025-12-05")
    saved_sessions = [
        main_module.SavedSessionAnalysis(
            session=session,
            block_performance=pd.DataFrame(
                {
                    "block_ix": [0, 1],
                    "block_type": ["left_cued", "right_uncued"],
                    "n_explore_runs": [99, 99],
                }
            ),
            augmented_trial_df=pd.DataFrame(
                {
                    "cur_block": [5, 5, 6, 6],
                    "explore_run_start": [True, True, True, True],
                    "explore_run_length": [1, 2, 1, 3],
                }
            ),
        )
    ]

    points_df = main_module.prepare_block_explore_run_block_points(
        saved_sessions,
        min_explore_run_length_to_count=2,
    )
    summary_df = main_module.prepare_block_explore_run_summary(
        saved_sessions,
        min_explore_run_length_to_count=2,
    )

    overall_points = points_df[points_df["explore_group"] == "overall"]
    overall_summary = summary_df[summary_df["explore_group"] == "overall"].iloc[0]
    assert overall_points["n_explore_runs"].tolist() == [1.0, 1.0]
    assert overall_summary["n_explore_runs_median"] == 1.0


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


def test_block_hmm_outputs_exist_requires_all_expected_files(tmp_path):
    """Block HMM reuse should require the model, block table, and trial table."""
    main_module = load_main_module()
    session = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
    )
    output_paths = main_module.get_block_hmm_output_paths(session)

    assert main_module.block_hmm_outputs_exist(session) is False

    output_paths["model_dict"].write_bytes(b"placeholder")
    output_paths["block_performance"].write_text("block_ix\n0\n")
    assert main_module.block_hmm_outputs_exist(session) is False

    output_paths["augmented_trials"].write_text("cur_block\n0\n")
    assert main_module.block_hmm_outputs_exist(session) is True


def test_load_saved_block_hmm_outputs_loads_block_and_trial_csvs(tmp_path):
    """Saved block HMM reuse should load the modeled block and trial CSVs."""
    main_module = load_main_module()
    session = SimpleNamespace(
        sess_id_full="CT014_multisession",
        processed_data_path=tmp_path,
    )
    output_paths = main_module.get_block_hmm_output_paths(session)
    output_paths["model_dict"].write_bytes(b"placeholder")
    pd.DataFrame({"block_ix": [0, 1], "inferred_strategy": [1, 0]}).to_csv(
        output_paths["block_performance"],
        index=False,
    )
    pd.DataFrame({"cur_block": [0, 1], "inherited_block_strategy": [1, 0]}).to_csv(
        output_paths["augmented_trials"],
        index=False,
    )

    block_df, trial_df = main_module.load_saved_block_hmm_outputs(session)

    assert block_df["inferred_strategy"].tolist() == [1, 0]
    assert trial_df["inherited_block_strategy"].tolist() == [1, 0]


def test_run_multisession_analysis_skips_block_modeling_when_outputs_exist(tmp_path, monkeypatch):
    """Existing block HMM outputs should be reused when the skip flag is enabled."""
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
    output_paths = main_module.get_block_hmm_output_paths(synthetic_session)
    output_paths["model_dict"].write_bytes(b"placeholder")
    pd.DataFrame({"block_ix": [0, 1], "inferred_strategy": [1, 0]}).to_csv(
        output_paths["block_performance"],
        index=False,
    )
    pd.DataFrame({"cur_block": [0, 1], "inherited_block_strategy": [1, 0]}).to_csv(
        output_paths["augmented_trials"],
        index=False,
    )
    monkeypatch.setattr(
        main_module.bssm,
        "run_block_modeling",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("block modeling should be skipped")),
        raising=False,
    )
    monkeypatch.setattr(
        main_module,
        "save_concatenated_multisession_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("inputs should not be overwritten")),
        raising=False,
    )

    block_selection, trial_selection, modeled_block_df, modeled_trial_df = main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        trial_predictor_columns=("FQlearning_rel_value",),
        skip_block_hmm_if_existing=True,
    )

    assert block_selection is None
    assert trial_selection is None
    assert modeled_block_df["inferred_strategy"].tolist() == [1, 0]
    assert modeled_trial_df["inherited_block_strategy"].tolist() == [1, 0]


def test_run_multisession_analysis_runs_block_modeling_when_skip_outputs_missing(tmp_path, monkeypatch):
    """The skip flag should not suppress modeling when required outputs are absent."""
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
        calls.append(session.sess_id_full)
        return (
            block_performance.assign(inferred_strategy=[0, 1]),
            augmented_trial_df.assign(inherited_block_strategy=[0, 1]),
        )

    monkeypatch.setattr(main_module.bssm, "run_block_modeling", fake_run_block_modeling, raising=False)
    monkeypatch.setattr(
        main_module,
        "save_concatenated_multisession_inputs",
        lambda *args, **_kwargs: save_calls.append(args),
        raising=False,
    )

    _block_selection, _trial_selection, modeled_block_df, modeled_trial_df = main_module.run_multisession_analysis(
        concatenated=concatenated,
        session=synthetic_session,
        trial_predictor_columns=("FQlearning_rel_value",),
        skip_block_hmm_if_existing=True,
    )

    assert calls == ["CT014_multisession"]
    assert len(save_calls) == 1
    assert modeled_block_df["inferred_strategy"].tolist() == [0, 1]
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
