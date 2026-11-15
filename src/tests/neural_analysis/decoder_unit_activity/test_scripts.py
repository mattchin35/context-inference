"""Single-session saved-only exports and session-level batch orchestration."""

import json
from pathlib import Path

import pytest

from src.neural_analysis.decoder_unit_activity.run_session import run_session, plot_selection, main
from src.neural_analysis.decoder_unit_activity.run_batch import read_requests, run_batch
from src.neural_analysis.population_decoding import pipeline, preparation


def forbid_analysis(*args, **kwargs):
    """Accept arbitrary calls; fail if a saved-result action tries raw preparation or fitting."""
    pytest.fail("Saved unit activity attempted neural analysis")


def test_dry_run_export_reuse_and_plot_from_selection(saved_decoder_run, monkeypatch):
    """Input complete run; export twice/replot without fitting, overwriting, or raw inputs."""
    monkeypatch.setattr(pipeline, "fit_target", forbid_analysis)
    monkeypatch.setattr(preparation, "prepare_session", forbid_analysis)
    root = saved_decoder_run.parent.parent
    before = set(root.rglob("*"))
    dry = run_session(saved_decoder_run, "PFC", command="dry-run", max_units_per_group=2)
    assert dry["status"] == "dry-run" and set(root.rglob("*")) == before
    report = run_session(saved_decoder_run, "PFC", max_units_per_group=2)
    assert report["status"] == "complete"
    directory = Path(report["run_directory"])
    assert all((directory / name).is_file() for name in ("activity.png", "unit_groups.csv", "summary.md", "run.log", "run_session.py", "run_batch.py"))
    repeated = run_session(saved_decoder_run, "PFC", max_units_per_group=2, scale="hz")
    assert repeated["selection_path"] == report["selection_path"]
    assert repeated["run_directory"] != report["run_directory"] and repeated["selection_reused"]
    different = run_session(saved_decoder_run, "PFC", max_units_per_group=1)
    assert different["selection_path"] != report["selection_path"]
    (root / "A/aligned.npz").unlink()
    plotted = plot_selection(report["selection_path"], scale="hz")
    assert plotted["status"] == "complete" and Path(plotted["figure_path"]).is_file()
    assert main(["dry-run", "--source-run", str(saved_decoder_run), "--region", "PFC"]) == 0


def test_request_schema_and_partial_batch(saved_decoder_run):
    """Input one session/two regions; a pending region does not cancel a valid saved export."""
    root = saved_decoder_run.parent.parent
    path = root / "unit_requests.json"
    path.write_text(json.dumps([{"source_run": str(saved_decoder_run.relative_to(root)), "regions": ["PFC", "HPC"]}]))
    requests = read_requests(path)
    assert requests[0]["source_run"] == saved_decoder_run
    dry = run_batch(path, command="dry-run", workers=1)
    assert dry["status"] == "dry-run" and dry["worker_count"] == 0
    (saved_decoder_run / "cells/HPC_left_switch_vs_stay.npz").unlink()
    dry_partial = run_batch(path, command="dry-run", workers=1)
    assert dry_partial["status"] == "partial" and dry_partial["worker_count"] == 0
    with pytest.raises(ValueError, match="cap|group"):
        run_batch(path, command="dry-run", max_units_per_group=0)
    report = run_batch(path, workers=1, max_units_per_group=1)
    assert report["status"] == "partial"
    assert [item["status"] for item in report["reports"]] == ["complete", "pending"]
    path.write_text(json.dumps([{"source_run": str(saved_decoder_run), "regions": ["PFC"]}]*2))
    with pytest.raises(ValueError, match="Duplicate"):
        read_requests(path)


@pytest.mark.parametrize("entry", [[], [{"source_run": "missing", "regions": []}],
    [{"source_run": "missing", "regions": ["PFC", "PFC"]}], [{"source_run": "missing", "regions": ["else"]}]])
def test_invalid_requests_fail_before_workers(tmp_path, entry):
    """Input malformed JSON requests; no workers or output writes are launched."""
    path = tmp_path / "requests.json"
    path.write_text(json.dumps(entry))
    with pytest.raises(ValueError):
        read_requests(path)
