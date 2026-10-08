"""Tests for inter-regional single-session and batch composition roots."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
)
from src.neural_analysis.interregional import records
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
    monkeypatch.setattr(run_session, "compute_single_session", lambda value: _result(value.config))

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
        "figures",
    } <= {path.name for path in first.run_path.iterdir()}


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

    def fail(_plan: run_session.SessionPlan):
        raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(run_session, "compute_single_session", fail)

    report = run_session.run_single_session(plan.config_path, command="new")

    assert report.status == "failed"
    assert report.error == "RuntimeError: synthetic runner failure"
    incomplete = tuple(plan.output_root.glob("*.incomplete"))
    assert len(incomplete) == 1
    assert (incomplete[0] / "failure.json").is_file()
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
