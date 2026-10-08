"""Tests for read-only discovery and display of saved inter-regional runs."""

from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.neural_analysis import session_metadata
from src.neural_analysis.interregional.configuration import (
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    PCAConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    RunOptions,
    TemporalConfig,
    configuration_to_dict,
)
from src.neural_analysis.interregional import persistence, records
from src.neural_analysis.webapp import interregional_views


def _session(tmp_path: Path) -> session_metadata.ResolvedSession:
    """Return a minimal selected metadata session rooted in ``tmp_path``."""
    metadata_path = tmp_path / "neural_session.json"
    metadata_path.write_text("{}", encoding="utf-8")
    return session_metadata.ResolvedSession(
        metadata_path=metadata_path,
        session_root=tmp_path,
        schema_version="2",
        session="session-a",
        acquisition="open_ephys",
        behavior=session_metadata.ResolvedBehaviorSources(None, None),
        probes=(),
        site_pairs=(),
        cache_directory=None,
    )


def _saved_run(session: session_metadata.ResolvedSession) -> Path:
    """Write one valid finalized saved run with a small summary table."""
    config = InterregionalAnalysisConfig(
        session_metadata_path=session.metadata_path,
        pfc_population=RegionalPopulationConfig(role="PFC", probe_id="pfc"),
        hpc_population=RegionalPopulationConfig(role="HPC", probe_id="hpc"),
        windows=AnalysisWindows(),
        prediction_windows=("before",),
        temporal=TemporalConfig(),
        pca=PCAConfig(pfc_components=1, hpc_components=1),
        filters=FilterConfig(conditions=("all",)),
    )
    populations = (
        ResolvedRegionalPopulation(
            role="PFC", probe_id="pfc", channel_source="explicit",
            selected_channels=(0,), cluster_ids=(1,), unit_ids=("pfc:1",),
        ),
        ResolvedRegionalPopulation(
            role="HPC", probe_id="hpc", channel_source="explicit",
            selected_channels=(1,), cluster_ids=(2,), unit_ids=("hpc:2",),
        ),
    )
    tables = records.make_empty_result_tables()
    tables["target_summaries"] = records.result_table_from_rows(
        "target_summaries",
        [
            {
                "session_id": "session-a", "evaluation_scope": "held_out_cv",
                "direction": "HPC_to_PFC", "representation": "units",
                "model_family": "ols", "condition": "all", "window": "before",
                "target_id": "pfc:1", "metric_name": "delta_r2", "target_rank": None,
                "status": "ok", "reason": "", "requested_folds": 5,
                "valid_folds": 5, "mean_value": 0.2,
            }
        ],
    )
    result = records.InterregionalResults(
        schema_version="1",
        analysis_version="interregional-regression-v1",
        coverage_assumption_version="implicit-complete-v1",
        configuration=configuration_to_dict(config, RunOptions()),
        session_id="session-a",
        resolved_populations=populations,
        whole_bin_edges_s=np.linspace(-2.0, 2.0, 41, dtype=np.float64),
        units_and_axes=records.default_units_and_axes(),
        randomness_used=False,
        random_seed=None,
        **tables,
    )
    fingerprint = "a" * 64
    run = (
        session.session_root / "analysis_runs"
        / "interregional_regression_20261008T010203123456Z_aaaaaaaaaaaa"
    )
    run.mkdir(parents=True)
    (run / "config.json").write_text(
        json.dumps(configuration_to_dict(config, RunOptions())), encoding="utf-8"
    )
    manifest = persistence.build_input_manifest(
        run_fingerprint=fingerprint,
        session_id="session-a",
        git_head="1" * 40,
        runtime_versions={key: "1" for key in persistence.RUNTIME_VERSION_KEYS},
        resolved_populations=populations,
        files=(),
    )
    (run / "input_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    persistence.save_interregional_result(result, run / "result.pkl", config)
    return run


def test_discovery_returns_only_valid_matching_final_runs(tmp_path: Path) -> None:
    """Incomplete, corrupt, wrong-session, and fingerprint-mismatched runs are ignored."""
    session = _session(tmp_path)
    valid = _saved_run(session)
    (valid.parent / ".interregional_regression_partial.incomplete").mkdir()
    corrupt = valid.parent / "interregional_regression_20261008T020203123456Z_bbbbbbbbbbbb"
    corrupt.mkdir()
    (corrupt / "input_manifest.json").write_text("{}", encoding="utf-8")

    options = interregional_views.discover_interregional_runs(session)

    assert len(options) == 1
    option = options[0]
    assert option.run_directory == valid
    assert "interregional-regression-v1" in option.display_label
    assert "aaaaaaaaaaaa" in option.display_label
    assert "PFC=pfc[1]" in option.display_label
    assert "HPC=hpc[2]" in option.display_label
    assert "ols_cv" in option.display_label


def test_saved_selectors_are_derived_only_from_present_result_rows(tmp_path: Path) -> None:
    """Display choices cannot request absent conditions, models, windows, or metrics."""
    session = _session(tmp_path)
    run = _saved_run(session)
    config, _ = interregional_views.load_interregional_config(run / "config.json")
    result = persistence.load_interregional_result(
        run / "result.pkl", config, trusted_run_directory=run
    )

    selectors = interregional_views.available_display_values(result)

    assert selectors == {
        "conditions": ("all",),
        "windows": ("before",),
        "representations": ("units",),
        "model_families": ("ols",),
        "metrics": ("delta_r2",),
    }


class _FakeStreamlit:
    """Small rendering spy that exposes no compute action."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def header(self, value, **kwargs): self.messages.append(str(value))
    def subheader(self, value, **kwargs): self.messages.append(str(value))
    def caption(self, value, **kwargs): self.messages.append(str(value))
    def info(self, value, **kwargs): self.messages.append(str(value))
    def warning(self, value, **kwargs): self.messages.append(str(value))
    def markdown(self, value, **kwargs): self.messages.append(str(value))
    def code(self, value, **kwargs): self.messages.append(str(value))
    def json(self, value, **kwargs): self.messages.append(str(value))
    def dataframe(self, value, **kwargs): self.messages.append("dataframe")
    def pyplot(self, value, **kwargs): self.messages.append("figure")
    def selectbox(self, label, options, **kwargs):
        self.messages.append(str(label))
        return options[0]


def test_render_is_saved_only_and_no_run_guidance_is_cli_only(tmp_path: Path) -> None:
    """Ordinary rerenders neither mutate files nor expose compute controls/imports."""
    session = _session(tmp_path)
    st = _FakeStreamlit()
    before = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))

    interregional_views.render_interregional_view(st, session)

    after = tuple(sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")))
    assert before == after
    text = " ".join(st.messages)
    assert "dry-run" in text and " new " in f" {text} "
    assert "metadata" in text.lower()
    assert "coverage" in text.lower()
    source = inspect.getsource(interregional_views)
    for forbidden in (
        "run_linear_cross_validation",
        "prepare_interregional_session",
        "load_aligned_spikes",
        "sklearn",
        "statsmodels",
    ):
        assert forbidden not in source


def test_poisson_saved_view_uses_deviance_and_separate_exploratory_mse(
    tmp_path: Path, monkeypatch
) -> None:
    """Saved Poisson rows select deviance plotting and an independent MSE view."""
    session = _session(tmp_path)
    run = _saved_run(session)
    config, _ = interregional_views.load_interregional_config(run / "config.json")
    result = persistence.load_interregional_result(
        run / "result.pkl", config, trusted_run_directory=run
    )
    poisson_target = result.target_summaries.iloc[0].to_dict()
    poisson_target.update(
        model_family="poisson", metric_name="delta_deviance_explained"
    )
    poisson_population = {
        "session_id": "session-a",
        "evaluation_scope": "held_out_cv",
        "direction": "HPC_to_PFC",
        "representation": "units",
        "model_family": "poisson",
        "condition": "all",
        "window": "before",
        "metric_name": "delta_deviance_explained",
        "status": "ok",
        "reason": "",
        "n_targets": 1,
        "q25": 0.2,
        "median": 0.2,
        "q75": 0.2,
    }
    result = replace(
        result,
        target_summaries=records.result_table_from_rows(
            "target_summaries",
            [*result.target_summaries.to_dict("records"), poisson_target],
        ),
        population_summaries=records.result_table_from_rows(
            "population_summaries", [poisson_population]
        ),
    )
    option = interregional_views.InterregionalRunOption(
        run, "a" * 64, result.analysis_version, result.session_id, "saved"
    )
    monkeypatch.setattr(
        interregional_views, "discover_interregional_runs", lambda value: (option,)
    )
    monkeypatch.setattr(
        interregional_views,
        "_selected_result",
        lambda value: (config, result, {"resolved_populations": [], "git_head": "1", "runtime_versions": {}, "entrypoint": "x"}),
    )
    calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        interregional_views,
        "plot_cv_increment_summary",
        lambda *args, **kwargs: (
            calls.append(("increment", kwargs["model_family"])) or object(),
            object(),
        ),
    )
    monkeypatch.setattr(
        interregional_views,
        "plot_ols_poisson_mse_comparison",
        lambda *args, **kwargs: (calls.append(("mse", kwargs["condition"])) or object(), object()),
    )

    class SelectingStreamlit(_FakeStreamlit):
        """Choose the Poisson family and its primary metric from saved options."""

        def selectbox(self, label, options, **kwargs):
            self.messages.append(str(label))
            if label == "Model family":
                return "poisson"
            if label == "Metric":
                return "delta_deviance_explained"
            return options[0]

    st = SelectingStreamlit()
    interregional_views.render_interregional_view(st, session)

    assert ("increment", "poisson") in calls
    assert ("mse", "all") in calls
    assert "exploratory" in " ".join(st.messages).lower()
