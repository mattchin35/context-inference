"""Single-session run lifecycle, preflight, and reuse behavior."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.communication_subspace.configuration import CommunicationConfig
from src.neural_analysis.communication_subspace.results import load_result
from src.neural_analysis.communication_subspace import run_session as runner


ROLES = {"PFC": "A", "HPC": "B"}


def test_dry_run_checks_population_and_splits_without_raw_spike_loading(
    synthetic_communication_session, monkeypatch,
):
    """Dry run validates good n and purge feasibility without fitting or writing."""
    config = CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=500)

    def forbidden(*args, **kwargs):
        """Flag an unexpected raw load or model fit during preflight."""
        raise AssertionError("dry run must not load spikes or compute")

    monkeypatch.setattr(runner, "load_session_inputs", forbidden)
    monkeypatch.setattr(runner, "compute_communication_subspace", forbidden)
    report = runner.run_session(config, dry_run=True)
    assert report.status == "dry-run"
    assert report.result_path is None
    assert report.run_directory is None
    assert report.configuration_id
    assert not (synthetic_communication_session.parent / "communication_subspace").exists()

    labels = synthetic_communication_session.parent / "A" / "cluster_info.tsv"
    table = pd.read_csv(labels, sep="\t")
    table.loc[3:7, "group"] = "noise"
    table.to_csv(labels, sep="\t", index=False)
    with pytest.raises(ValueError, match="unit|population|good"):
        runner.run_session(config, dry_run=True)


def test_dry_run_rejects_dense_overlapping_choices(synthetic_communication_session):
    """A shared split failure is discovered before raw-spike loading or output writes."""
    root = synthetic_communication_session.parent
    trials = pd.read_csv(root / "trials.csv")
    trials["choice_time"] = trials.choice_time.iloc[0] + 0.1 * np.arange(len(trials))
    trials.to_csv(root / "trials.csv", index=False)
    with pytest.raises(ValueError, match="purge|overlap|choice|train|fold"):
        runner.run_session(CommunicationConfig(synthetic_communication_session, ROLES), dry_run=True)
    assert not (root / "communication_subspace").exists()


def test_run_reuse_and_explicit_recompute_create_distinct_runs(
    synthetic_communication_session, monkeypatch,
):
    """Only explicit Run computes; reuse is read-only and rerun preserves prior files."""
    def save_test_figure(result, path):
        """Write a small PNG marker for runner output-path testing."""
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n")
        return Path(path)

    monkeypatch.setattr(runner, "save_rank_figure", save_test_figure)
    config = CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=500)
    first = runner.run_session(config)
    assert first.status == "prepared"
    assert first.result_path.is_file()
    assert first.run_directory.is_dir()
    assert first.result_path.parent == synthetic_communication_session.parent / "communication_subspace" / "results"
    assert first.run_directory.parent == synthetic_communication_session.parent / "communication_subspace" / "runs"
    for name in ("config.json", "run_session.py", "run.log", "summary.md", "rank_curves.png"):
        assert (first.run_directory / name).is_file()
    saved = load_result(first.result_path)
    assert saved.meta["configuration"]["configuration_id"] == first.configuration_id
    assert saved.meta["generator"] == "run_session"
    assert saved.meta["analysis_version"] == "communication-subspace-v1"
    assert saved.meta["seed"] == 0
    assert saved.computation.status in {"success", "partial", "unavailable"}
    assert "single-session" in (first.run_directory / "summary.md").read_text().lower()
    original_bytes = first.result_path.read_bytes()

    def forbidden_compute(*args, **kwargs):
        """Flag a fit attempted during compatible-result reuse."""
        raise AssertionError("reuse must not compute")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "compute_communication_subspace", forbidden_compute)
        reused = runner.run_session(config)
    assert reused.status == "reused"
    assert reused.result_path == first.result_path
    assert first.result_path.read_bytes() == original_bytes

    second = runner.run_session(config, rerun=True)
    assert second.status == "prepared"
    assert second.result_path != first.result_path
    assert second.run_directory != first.run_directory
    assert second.result_path.is_file()
    assert first.result_path.read_bytes() == original_bytes


def test_reject_repository_output_and_changed_source_identity(
    synthetic_communication_session, monkeypatch,
):
    """Runner must refuse Git-contained output and a source edit during fitting."""
    config = CommunicationConfig(synthetic_communication_session, ROLES, bin_width_ms=500)
    repository = Path(__file__).resolve().parents[4]
    original_inspect = runner.inspect_session_inputs

    def repository_root_inspection(settings):
        """Model a misconfigured session root inside the source repository."""
        inspection = original_inspect(settings)
        return replace(inspection, session=replace(inspection.session, session_root=repository))

    with monkeypatch.context() as patch:
        patch.setattr(runner, "inspect_session_inputs", repository_root_inspection)
        with pytest.raises(ValueError, match="repository|outside|output"):
            runner.run_session(config)

    original_compute = runner.compute_communication_subspace

    def mutate_source(prepared, **kwargs):
        """Change trial source after preparation but before result publication."""
        trials = synthetic_communication_session.parent / "trials.csv"
        trials.write_text(trials.read_text() + "\n", encoding="ascii")
        return original_compute(prepared, **kwargs)

    monkeypatch.setattr(runner, "compute_communication_subspace", mutate_source)
    with pytest.raises(ValueError, match="source|changed|identity|configuration"):
        runner.run_session(config)
