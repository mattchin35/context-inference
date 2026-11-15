"""Saved behavior preparation, reuse, batch and trusted-local archive contracts."""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

from src.tests.neural_analysis.event_sequences.example_fixtures import example_session, example_trials


def test_prepare_roundtrip_reuse_and_immutable_recompute(example_session):
    """Input: no-neural synthetic session. Check exact (N,C)/(E,S) saved data and provenance."""
    runner = importlib.import_module("src.neural_analysis.event_sequences.run_examples")
    io = importlib.import_module("src.neural_analysis.event_sequences.example_io")
    first = runner.prepare_behavior_examples(example_session.metadata_path)
    path = first["result_path"]
    before = path.read_bytes()
    packet = io.load_examples(path)
    assert len(packet["matches"][("probe", "left", 1)]) == 3
    assert packet["meta"]["parameters"]["probe_return_outcome"] == "rewarded_or_omission"
    assert packet["meta"]["random_seed"] is None
    assert packet["meta"]["units"]["choice_time"] == "UTC Unix seconds"
    assert path.is_relative_to(example_session.behavior.trial_table_file.parent / "behavior_examples")
    for name in ("run_examples.py", "run_examples_batch.py", "summary.md", "run.log"):
        assert (first["run_directory"] / name).is_file()
    with np.load(path, allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
        assert "trials" in archive.files
    reused = runner.prepare_behavior_examples(example_session.metadata_path)
    assert reused["status"] == "reused"
    assert reused["result_path"] == path
    fresh = runner.prepare_behavior_examples(example_session.metadata_path, rerun=True)
    assert fresh["result_path"] != path
    assert path.read_bytes() == before
    pd.testing.assert_frame_equal(io.load_examples(fresh["result_path"])["trials"], packet["trials"])
    with pytest.raises(FileExistsError):
        io.save_examples(path, packet)


def test_input_parameters_and_version_invalidate_reuse(example_session, monkeypatch):
    """Input: source/version/parameter changes. Each scientific change requires a new product."""
    runner = importlib.import_module("src.neural_analysis.event_sequences.run_examples")
    io = importlib.import_module("src.neural_analysis.event_sequences.example_io")
    first = runner.prepare_behavior_examples(example_session.metadata_path)
    strict = runner.prepare_behavior_examples(example_session.metadata_path, probe_return_outcome="rewarded")
    assert first["configuration_id"] != strict["configuration_id"]
    source = example_session.behavior.trial_table_file
    source.write_text(source.read_text() + "\n", encoding="ascii")
    changed = runner.prepare_behavior_examples(example_session.metadata_path)
    assert changed["configuration_id"] != first["configuration_id"]
    monkeypatch.setattr(io, "ANALYSIS_VERSION", io.ANALYSIS_VERSION + "-test")
    versioned = runner.prepare_behavior_examples(example_session.metadata_path)
    assert versioned["configuration_id"] != changed["configuration_id"]


def test_dry_run_has_no_outputs_and_batch_validates_requests(example_session):
    """Input: explicit metadata list. Dry run makes no files/processes and duplicates are invalid."""
    runner = importlib.import_module("src.neural_analysis.event_sequences.run_examples")
    batch = importlib.import_module("src.neural_analysis.event_sequences.run_examples_batch")
    report = runner.prepare_behavior_examples(example_session.metadata_path, dry_run=True)
    assert report["status"] == "dry-run"
    assert not (example_session.session_root / "analysis_runs").exists()
    assert len(batch.run_batch([example_session.metadata_path], dry_run=True)) == 1
    with pytest.raises(ValueError):
        batch.run_batch([example_session.metadata_path] * 2, dry_run=True)
    with pytest.raises(ValueError):
        batch.run_batch([example_session.metadata_path], workers=0)
    reports = batch.run_batch([example_session.metadata_path], workers=1)
    assert reports[0]["status"] == "prepared"


def test_export_records_exact_selection_and_preserves_prior_png(example_session):
    """Input: one saved match plus context counts. Export PNG/JSON with exact source identities."""
    runner = importlib.import_module("src.neural_analysis.event_sequences.run_examples")
    report = runner.prepare_behavior_examples(example_session.metadata_path)
    condition = ("probe", "left", 1)
    first = runner.export_behavior_example(report["result_path"], condition, 0, trials_before=0, trials_after=2)
    contents = first.read_bytes()
    assert contents.startswith(b"\x89PNG")
    metadata = json.loads(first.with_suffix(".json").read_text())
    assert metadata["episode_trial_ids"] == [501, 502, 503]
    assert metadata["trials_before"] == 0
    assert metadata["trials_after"] == 2
    assert metadata["condition"] == list(condition)
    second = runner.export_behavior_example(report["result_path"], condition, 0)
    assert second != first
    assert first.read_bytes() == contents
