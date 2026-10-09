"""Tests for inter-regional single-session and batch composition roots."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

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
    load_interregional_config,
)
from src.neural_analysis.interregional import persistence, records
from src.neural_analysis.interregional import run_batch, run_session


def _config(metadata_path: Path, output_root: Path) -> InterregionalAnalysisConfig:
    """Return one compact units-only analysis configuration."""
    return InterregionalAnalysisConfig(
        session_metadata_path=metadata_path,
        pfc_population=RegionalPopulationConfig(
            role="PFC", probe_id="pfc", channel_source="explicit", selected_channels=(0,)
        ),
        hpc_population=RegionalPopulationConfig(
            role="HPC", probe_id="hpc", channel_source="explicit", selected_channels=(1,)
        ),
        windows=AnalysisWindows(),
        prediction_windows=("before",),
        temporal=TemporalConfig(),
        pca=PCAConfig(pfc_components=1, hpc_components=1),
        filters=FilterConfig(conditions=("all",)),
        representations=("units",),
        analyses=("ols_cv",),
    )


def _populations() -> tuple[ResolvedRegionalPopulation, ResolvedRegionalPopulation]:
    """Return one explicit selected unit per region."""
    return (
        ResolvedRegionalPopulation(
            role="PFC",
            probe_id="pfc",
            channel_source="explicit",
            selected_channels=(0,),
            cluster_ids=(1,),
            unit_ids=("pfc:1",),
        ),
        ResolvedRegionalPopulation(
            role="HPC",
            probe_id="hpc",
            channel_source="explicit",
            selected_channels=(1,),
            cluster_ids=(2,),
            unit_ids=("hpc:2",),
        ),
    )


def _result(config: InterregionalAnalysisConfig) -> records.InterregionalResults:
    """Return a valid deterministic pure result used at the runner seam."""
    return records.InterregionalResults(
        schema_version="1",
        analysis_version="interregional-regression-v1",
        coverage_assumption_version="implicit-complete-v1",
        configuration=configuration_to_dict(config, RunOptions()),
        session_id="session-a",
        resolved_populations=_populations(),
        whole_bin_edges_s=np.linspace(-2.0, 2.0, 41, dtype=np.float64),
        units_and_axes=records.default_units_and_axes(),
        randomness_used=False,
        random_seed=None,
        **records.make_empty_result_tables(),
    )


def _runtime_versions() -> dict[str, str]:
    """Return exact runtime keys without depending on installed version text."""
    return {
        "python": "3.12",
        "numpy": "2",
        "pandas": "2",
        "scipy": "1",
        "pynapple": "0",
        "scikit_learn": "1",
        "statsmodels": "0",
        "matplotlib": "3",
    }


def _plan(tmp_path: Path) -> run_session.SessionPlan:
    """Construct a fully resolved lightweight plan for orchestration tests."""
    metadata = tmp_path / "session" / "neural_session.json"
    metadata.parent.mkdir()
    metadata.write_text("{}", encoding="utf-8")
    trial_table = metadata.parent / "trials.csv"
    trial_table.write_text("x\n1\n", encoding="utf-8")
    output_root = tmp_path / "outputs"
    config = _config(metadata.resolve(), output_root.resolve())
    return run_session.SessionPlan(
        config_path=tmp_path / "config.json",
        config=config,
        run_options=RunOptions(output_root=output_root.resolve()),
        session_id="session-a",
        session_root=metadata.parent,
        output_root=output_root.resolve(),
        resolved_populations=_populations(),
        input_files={"session_metadata": metadata, "trial_table": trial_table},
        input_sizes={"session_metadata": 2, "trial_table": 4},
        runtime_versions=_runtime_versions(),
        git_head="1" * 40,
    )


def test_single_session_new_reuses_matching_run_and_rerun_is_immutable(
    tmp_path: Path, monkeypatch
) -> None:
    """Default new skips a matching fingerprint while rerun creates another run."""
    plan = _plan(tmp_path)
    monkeypatch.setattr(run_session, "plan_single_session", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        run_session,
        "compute_single_session",
        lambda value, **kwargs: _result(value.config),
    )

    first = run_session.run_single_session(plan.config_path, command="new")
    reused = run_session.run_single_session(plan.config_path, command="new")
    rerun = run_session.run_single_session(
        plan.config_path, command="new", rerun=True
    )

    assert first.status == "completed"
    assert reused.status == "reused"
    assert reused.run_path == first.run_path
    assert rerun.status == "completed"
    assert rerun.run_path != first.run_path
    assert rerun.run_fingerprint == first.run_fingerprint
    assert first.run_path is not None
    assert {
        "config.json",
        "input_manifest.json",
        "result.pkl",
        "run.log",
        "summary.md",
        "run_session.py",
        "run_batch.py",
        "resource_trace.jsonl",
        "resource_summary.json",
        "figures",
    } <= {path.name for path in first.run_path.iterdir()}
    summary = (first.run_path / "summary.md").read_text(encoding="utf-8")
    assert f"Output directory: `{first.run_path}`" in summary
    assert f'"output_root": "{plan.output_root}"' in summary
    assert ".incomplete" not in summary


def test_dry_run_reports_plan_without_hashing_or_creating_output(
    tmp_path: Path, monkeypatch
) -> None:
    """Dry-run stops after path/size/runtime validation and has no side effects."""
    plan = _plan(tmp_path)
    monkeypatch.setattr(run_session, "plan_single_session", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        run_session.persistence,
        "hash_input_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hashed")),
    )

    report = run_session.run_single_session(plan.config_path, command="dry-run")

    assert report.status == "planned"
    assert report.run_fingerprint is None
    assert report.run_path is None
    assert report.input_bytes == 6
    assert not plan.output_root.exists()


def test_failure_is_logged_per_session_without_finalizing(tmp_path: Path, monkeypatch) -> None:
    """A caught computation error leaves one incomplete directory and failure record."""
    plan = _plan(tmp_path)
    monkeypatch.setattr(run_session, "plan_single_session", lambda *args, **kwargs: plan)

    def fail(_plan: run_session.SessionPlan, **kwargs):
        raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(run_session, "compute_single_session", fail)

    report = run_session.run_single_session(plan.config_path, command="new")

    assert report.status == "failed"
    assert report.error == "RuntimeError: synthetic runner failure"
    incomplete = tuple(plan.output_root.glob("*.incomplete"))
    assert len(incomplete) == 1
    assert (incomplete[0] / "failure.json").is_file()
    trace_records = [
        json.loads(line)
        for line in (incomplete[0] / "resource_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert trace_records
    assert not (incomplete[0] / "resource_summary.json").exists()
    assert not tuple(plan.output_root.glob("interregional_regression_*"))


def test_batch_list_worker_count_and_dry_run_are_session_separated(
    tmp_path: Path, monkeypatch
) -> None:
    """Batch planning preserves config order and applies the documented worker minimum."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    config_list = tmp_path / "configs.txt"
    config_list.write_text("# sessions\nfirst.json\n\nsecond.json\n", encoding="utf-8")
    calls: list[Path] = []

    def plan(config_path: Path, **kwargs):
        calls.append(config_path)
        return run_session.SessionRunReport(
            session_id=config_path.stem,
            status="planned",
            run_path=None,
            run_fingerprint=None,
            input_bytes=10,
            error=None,
        )

    monkeypatch.setattr(run_batch, "run_single_session", plan)

    paths = run_batch.read_config_list(config_list)
    result = run_batch.run_batch(
        paths, command="dry-run", workers=8, cpu_count=4
    )

    assert paths == (first.resolve(), second.resolve())
    assert run_batch.choose_worker_count(2, requested=8, cpu_count=4) == 2
    assert result.worker_count == 2
    assert result.total_input_bytes == 20
    assert result.max_session_input_bytes == 10
    assert [report.session_id for report in result.reports] == ["first", "second"]
    assert calls == list(paths)


def test_cli_parsers_expose_only_documented_commands() -> None:
    """Session and batch parsers keep dry-run/new/rerun syntax synchronized."""
    session_args = run_session._parser().parse_args(
        ["new", "--config", "/tmp/config.json", "--rerun"]
    )
    batch_args = run_batch._parser().parse_args(
        ["dry-run", "--config-list", "/tmp/configs.txt", "--workers", "2"]
    )

    assert isinstance(session_args, argparse.Namespace)
    assert (session_args.command, session_args.rerun) == ("new", True)
    assert (batch_args.command, batch_args.workers) == ("dry-run", 2)


def _write_synthetic_session(
    tmp_path: Path,
    *,
    full_standard: bool = False,
    poisson: bool = False,
    granger: bool = False,
) -> Path:
    """Write one seeded metadata-v2 session and return its analysis config path."""
    session_root = tmp_path / "synthetic_session"
    session_root.mkdir()
    n_trials = 25
    choice_times = np.arange(n_trials, dtype=float) * 10.0 + 5.0
    trial_table = pd.DataFrame(
        {
            "experimenter_reward_given": 0,
            "correct": 1,
            "reward": 0 if full_standard else 1,
            "action": 1,
            "state_int": 1,
            "choice_time": choice_times,
            "cur_block": np.repeat(np.arange(5), 5),
        }
    )
    if granger:
        trial_table = trial_table.drop(columns="cur_block")
    trial_table.to_csv(session_root / "trials.csv", index=False)
    rng = np.random.default_rng(91)
    probes: dict[str, object] = {}
    probe_specs = (("pfc", 0, (1, 2)), ("hpc", 1, (3, 4)))
    for probe_id, channel, available_cluster_ids in probe_specs:
        cluster_ids = available_cluster_ids if full_standard else available_cluster_ids[:1]
        sorter = session_root / f"{probe_id}_sorter"
        sorter.mkdir()
        unit_times = [
            np.concatenate(
                [
                    trial_time + rng.uniform(-1.9, 1.9, size=35)
                    for trial_time in choice_times
                ]
            )
            for _cluster_id in cluster_ids
        ]
        spike_times = np.concatenate(unit_times)
        spike_clusters = np.concatenate(
            [
                np.full(times.size, cluster_id)
                for times, cluster_id in zip(unit_times, cluster_ids)
            ]
        )
        order = np.argsort(spike_times, kind="stable")
        spike_times = spike_times[order]
        spike_clusters = spike_clusters[order]
        np.save(sorter / "spike_clusters.npy", spike_clusters)
        pd.DataFrame(
            {
                "cluster_id": cluster_ids,
                "ch": [channel] * len(cluster_ids),
                "group": ["good"] * len(cluster_ids),
            }
        ).to_csv(sorter / "cluster_info.tsv", sep="\t", index=False)
        np.savez(session_root / f"{probe_id}_sync.npz", spike_utc_unix=spike_times)
        probes[probe_id] = {
            "lfp": "",
            "alignment": f"{probe_id}_sync.npz",
            "sorter": f"{probe_id}_sorter",
            "quality": f"{probe_id}_quality.csv",
            "sites": {},
            "unit_channels": [channel],
        }
    metadata_path = session_root / "neural_session.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "2",
                "session": "synthetic-session",
                "acquisition": "open_ephys",
                "behavior": {"trials": "trials.csv", "events": None},
                "probes": probes,
                "site_pairs": [],
                "cache": None,
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "analysis_outputs"
    config = _config(metadata_path.resolve(), output_root.resolve())
    if poisson:
        config = replace(config, analyses=("ols_cv", "poisson_cv"))
    if granger:
        config = replace(
            config,
            analyses=("linear_granger", "poisson_granger"),
            representations=("units",),
        )
    if full_standard:
        config = replace(
            config,
            prediction_windows=("before", "after", "whole"),
            pca=PCAConfig(pfc_components=3, hpc_components=3),
            filters=FilterConfig(conditions=("all", "stay")),
            representations=("units", "pcs"),
        )
    config_path = tmp_path / "interregional_config.json"
    config_path.write_text(
        json.dumps(
            configuration_to_dict(config, RunOptions(output_root=output_root.resolve()))
        ),
        encoding="utf-8",
    )
    return config_path


def test_public_single_session_composition_runs_synthetic_metadata_to_reload(
    tmp_path: Path, monkeypatch
) -> None:
    """The production entry point composes real loaders, fitting, save, and reload."""
    config_path = _write_synthetic_session(tmp_path)
    monkeypatch.setattr(run_session, "_git_identity", lambda root: ("1" * 40, ()))

    report = run_session.run_single_session(config_path, command="new")

    assert report.status == "completed", report.error
    assert report.run_path is not None
    config, _ = load_interregional_config(config_path)
    loaded = persistence.load_interregional_result(
        report.run_path / "result.pkl",
        config,
        trusted_run_directory=report.run_path,
    )
    assert loaded.session_id == "synthetic-session"
    assert not loaded.fold_scores.empty
    assert loaded.fold_scores["representation"].eq("units").all()
    assert loaded.pca_fits.empty
    run_log = (report.run_path / "run.log").read_text(encoding="utf-8")
    for stage in (
        "input_validation",
        "input_hashing",
        "preparation",
        "ols_cv",
        "persistence",
        "figures",
        "summary",
    ):
        assert f"stage={stage}" in run_log
    assert tuple((report.run_path / "figures").glob("*.png"))


def test_wp9_poisson_synthetic_run_round_trips_with_matched_ols_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """The public runner persists complete matched unit OLS and Poisson folds."""
    config_path = _write_synthetic_session(tmp_path, poisson=True)
    monkeypatch.setattr(run_session, "_git_identity", lambda root: ("1" * 40, ()))

    report = run_session.run_single_session(config_path, command="new")

    assert report.status == "completed", report.error
    assert report.run_path is not None
    config, _ = load_interregional_config(config_path)
    loaded = persistence.load_interregional_result(
        report.run_path / "result.pkl",
        config,
        trusted_run_directory=report.run_path,
    )
    assert set(loaded.fold_scores["model_family"]) == {"ols", "poisson"}
    poisson_rows = loaded.fold_scores.loc[
        loaded.fold_scores["model_family"].eq("poisson")
    ]
    assert poisson_rows["status"].eq("ok").all()
    assert poisson_rows["restricted_converged"].eq(True).all()
    assert poisson_rows["full_converged"].eq(True).all()
    identity = [
        "direction",
        "condition",
        "window",
        "target_id",
        "fold_id",
        "train_row_set_sha256",
        "test_row_set_sha256",
    ]
    ols_rows = loaded.fold_scores.loc[loaded.fold_scores["model_family"].eq("ols")]
    pd.testing.assert_frame_equal(
        ols_rows.loc[:, identity].reset_index(drop=True),
        poisson_rows.loc[:, identity].reset_index(drop=True),
    )
    run_log = (report.run_path / "run.log").read_text(encoding="utf-8")
    assert "stage=poisson_cv event=start" in run_log
    assert "stage=poisson_cv event=end" in run_log
    trace_records = [
        json.loads(line)
        for line in (report.run_path / "resource_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    stage_events = [
        (record["stage"], record["progress_event"])
        for record in trace_records
        if record.get("progress_event") in {"stage_start", "stage_end"}
    ]
    assert stage_events.index(("preparation", "stage_end")) < stage_events.index(
        ("ols_cv", "stage_start")
    )
    assert stage_events.index(("ols_cv", "stage_end")) < stage_events.index(
        ("poisson_cv", "stage_start")
    )
    assert stage_events.index(("poisson_cv", "stage_end")) < stage_events.index(
        ("result_assembly", "stage_start")
    )
    resource_summary = json.loads(
        (report.run_path / "resource_summary.json").read_text(encoding="utf-8")
    )
    assert resource_summary["completed"] is True
    summary = (report.run_path / "summary.md").read_text(encoding="utf-8")
    assert "Incremental CV deviance explained" in summary
    assert "Exploratory OLS/Poisson count-MSE comparison" in summary
    assert "not a formal test of model superiority" in summary
    figure_names = {path.name for path in (report.run_path / "figures").glob("*.png")}
    assert "cv_increment_units_poisson_before.png" in figure_names
    assert "cv_mse_ols_poisson_all_before.png" in figure_names


def test_wp10_granger_only_synthetic_run_needs_no_blocks_or_cv(
    tmp_path: Path, monkeypatch
) -> None:
    """The public runner persists independent in-sample Granger stages and figures."""
    config_path = _write_synthetic_session(tmp_path, granger=True)
    monkeypatch.setattr(run_session, "_git_identity", lambda root: ("1" * 40, ()))

    report = run_session.run_single_session(config_path, command="new")

    assert report.status == "completed", report.error
    assert report.run_path is not None
    config, _ = load_interregional_config(config_path)
    loaded = persistence.load_interregional_result(
        report.run_path / "result.pkl",
        config,
        trusted_run_directory=report.run_path,
    )
    assert loaded.fold_assignments.empty
    assert loaded.fold_scores.empty
    assert loaded.target_summaries.empty
    assert not loaded.granger_scores.empty
    assert set(loaded.granger_scores["model_family"]) == {"ols", "poisson"}
    assert loaded.granger_scores["evaluation_scope"].eq("in_sample").all()
    assert loaded.granger_scores["n_trials"].eq(25).all()
    assert loaded.population_summaries["evaluation_scope"].eq("in_sample").all()
    run_log = (report.run_path / "run.log").read_text(encoding="utf-8")
    assert "stage=linear_granger event=start" in run_log
    assert "stage=linear_granger event=end" in run_log
    assert "stage=poisson_granger event=start" in run_log
    assert "stage=poisson_granger event=end" in run_log
    assert "stage=ols_cv" not in run_log
    assert "stage=poisson_cv" not in run_log
    summary = (report.run_path / "summary.md").read_text(encoding="utf-8")
    assert "descriptive in-sample" in summary.lower()
    assert "not significance tests" in summary.lower()
    assert "Primary held-out results" not in summary
    assert "Goal: quantify descriptive in-sample" in summary
    figure_names = {path.name for path in (report.run_path / "figures").glob("*.png")}
    assert "granger_units_ols_before.png" in figure_names
    assert "granger_units_poisson_before.png" in figure_names


def test_wp7_full_standard_synthetic_run_round_trips_and_records_evidence(
    tmp_path: Path, monkeypatch, record_property
) -> None:
    """The seeded public workflow preserves results and emits bounded-run evidence."""
    config_path = _write_synthetic_session(tmp_path, full_standard=True)
    monkeypatch.setattr(run_session, "_git_identity", lambda root: ("1" * 40, ()))

    started = time.perf_counter()
    first_report = run_session.run_single_session(config_path, command="new")
    second_report = run_session.run_single_session(
        config_path, command="new", rerun=True
    )
    elapsed_seconds = time.perf_counter() - started

    assert first_report.status == "completed", first_report.error
    assert second_report.status == "completed", second_report.error
    assert first_report.run_path is not None
    assert second_report.run_path is not None
    assert first_report.run_fingerprint == second_report.run_fingerprint
    assert first_report.run_path != second_report.run_path
    config, _ = load_interregional_config(config_path)
    first = persistence.load_interregional_result(
        first_report.run_path / "result.pkl",
        config,
        trusted_run_directory=first_report.run_path,
    )
    second = persistence.load_interregional_result(
        second_report.run_path / "result.pkl",
        config,
        trusted_run_directory=second_report.run_path,
    )
    for table_name in records.RESULT_TABLE_DTYPES:
        pd.testing.assert_frame_equal(
            getattr(first, table_name), getattr(second, table_name)
        )
    np.testing.assert_array_equal(first.whole_bin_edges_s, second.whole_bin_edges_s)

    assert {"units", "pcs"} == set(first.fold_scores["representation"])
    assert set(first.fold_scores["model_family"]) == {"ols"}
    assert not first.pca_fits.empty
    unavailable_rank = first.fold_scores.loc[
        first.fold_scores["representation"].eq("pcs")
        & first.fold_scores["target_rank"].eq(3)
    ]
    assert len(unavailable_rank) == 60
    assert unavailable_rank["status"].eq("fit_unavailable").all()
    assert unavailable_rank["reason"].eq("pca_insufficient_components").all()
    assert first.fold_scores.loc[
        first.fold_scores["representation"].eq("units"), "status"
    ].eq("ok").all()
    assert first.granger_scores.empty
    poisson_columns = (
        "restricted_converged",
        "full_converged",
        "restricted_iterations",
        "full_iterations",
        "deviance_restricted",
        "deviance_full",
        "null_deviance",
        "deviance_explained_restricted",
        "deviance_explained_full",
        "delta_deviance_explained",
    )
    assert first.fold_scores.loc[:, poisson_columns].isna().all().all()
    assert first.fold_scores["train_row_set_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert first.fold_scores["test_row_set_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    figures = tuple((first_report.run_path / "figures").glob("*.png"))
    assert figures
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in figures)
    run_log = (first_report.run_path / "run.log").read_text(encoding="utf-8")
    assert "completed elapsed_seconds=" in run_log
    assert all(
        f"stage={stage} event=end elapsed_seconds=" in run_log
        for stage in (
            "input_validation",
            "input_hashing",
            "ols_cv",
            "persistence",
            "figures",
            "summary",
        )
    )
    output_bytes = sum(
        path.stat().st_size for path in first_report.run_path.rglob("*") if path.is_file()
    )
    evidence = {
        "seed": 91,
        "trials": 25,
        "whole_window_bins": 40,
        "pfc_units": 2,
        "hpc_units": 2,
        "requested_components_per_region": 3,
        "conditions": 2,
        "prediction_windows": 3,
        "fold_score_rows": len(first.fold_scores),
        "figure_count": len(figures),
        "first_run_output_bytes": output_bytes,
        "two_run_elapsed_seconds": elapsed_seconds,
    }
    record_property("wp7_synthetic_evidence", json.dumps(evidence, sort_keys=True))
    print(f"WP7_SYNTHETIC_EVIDENCE={json.dumps(evidence, sort_keys=True)}")


def test_new_refuses_dirty_code_before_hashing(tmp_path: Path, monkeypatch) -> None:
    """Git-unidentified source changes stop a new run before input hashing."""
    plan = replace(_plan(tmp_path), code_dirty_paths=("src/neural_analysis/new.py",))
    monkeypatch.setattr(run_session, "plan_single_session", lambda *args, **kwargs: plan)
    monkeypatch.setattr(
        run_session.persistence,
        "hash_input_files",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("hashed")),
    )

    with pytest.raises(ValueError, match="clean tracked code"):
        run_session.run_single_session(plan.config_path, command="new")
