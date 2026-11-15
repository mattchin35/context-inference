"""Read-only planning, immutable execution, and source-free interrupted-run resume."""

import json
from pathlib import Path
import pytest

from src.neural_analysis.population_decoding import pipeline, preparation, results
from src.neural_analysis.population_decoding.run_session import run_session, initialize_config, main


def forbid_compute(*args, **kwargs):
    """Reject any computational arguments; saved-only controls must not compute."""
    pytest.fail("Unexpected preparation or refitting")


def test_dry_run_writes_no_outputs_and_fits_nothing(population_config_path, monkeypatch):
    """Input session/patch; read-only plan leaves source tree untouched, no activity loading."""
    monkeypatch.setattr(preparation, "prepare_session", forbid_compute)
    monkeypatch.setattr(pipeline, "fit_target", forbid_compute)
    before = set(population_config_path.parent.rglob("*"))
    report = run_session(population_config_path, command="dry-run")
    assert report["status"] == "dry-run" and report["maximum_classifier_fits"] == 140
    assert before == set(population_config_path.parent.rglob("*"))


def test_new_reuse_source_free_resume_and_published_file_adoption(population_config_path, monkeypatch):
    """Input sources/patch; publish fourteen cells, reuse without fitting, recover stale progress."""
    report = run_session(population_config_path)
    run = Path(report["run_directory"])
    assert report["status"] == "complete" and report["completed_cells"] == 14
    assert len(list((run / "cells").glob("*.npz"))) == 14
    for name in ("run_session.py", "run_batch.py", "run.log", "summary.md", "performance.json"):
        assert (run / name).is_file()
    performance = json.loads((run / "performance.json").read_text())
    assert performance["elapsed_s"] > 0 and performance["peak_rss_bytes"] > 0
    monkeypatch.setattr(pipeline, "fit_target", forbid_compute)
    assert run_session(population_config_path)["status"] == "reused"
    original = {path.name: path.read_bytes() for path in (run / "cells").glob("*.npz")}
    (run / "progress.json").write_text(json.dumps({"status": "partial", "completed_cells": 0}))
    for path in (population_config_path.parent / "trials.csv", population_config_path.parent / "A/aligned.npz"):
        path.unlink()
    monkeypatch.setattr(preparation, "prepare_session", forbid_compute)
    resumed = run_session(command="resume", run_directory=run)
    assert resumed["status"] == "complete" and resumed["completed_cells"] == 14
    assert original == {path.name: path.read_bytes() for path in (run / "cells").glob("*.npz")}
    assert run_session(command="status", run_directory=run)["completed_cells"] == 14


def test_new_parameters_keep_old_outputs_and_rerun_uses_new_directory(population_config_path):
    """Input config; explicit rerun/new bin width keep previous immutable products."""
    first = run_session(population_config_path)
    second = run_session(population_config_path, rerun=True)
    assert first["run_directory"] != second["run_directory"]
    payload = json.loads(population_config_path.read_text())
    payload["bin_width_ms"] = 100
    population_config_path.write_text(json.dumps(payload))
    third = run_session(population_config_path)
    assert third["run_directory"] not in (first["run_directory"], second["run_directory"])


def test_initialization_and_cli_status(population_config_path, capsys):
    """Input metadata/config; initialization is non-overwriting and CLI prints JSON status."""
    metadata = population_config_path.parent / "neural_session.json"
    with pytest.raises(FileExistsError):
        initialize_config(metadata, probe_roles={"PFC": "A", "HPC": "B"})
    population_config_path.unlink()
    path = initialize_config(metadata, probe_roles={"PFC": "A", "HPC": "B"})
    assert path == population_config_path
    assert main(["dry-run", "--config", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"


def test_corrupt_existing_cell_is_not_overwritten(population_config_path):
    """Input run sources; corrupt published cell blocks resume and retains original bytes."""
    report = run_session(population_config_path)
    run = Path(report["run_directory"])
    cell = next((run / "cells").glob("*.npz"))
    cell.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        run_session(command="resume", run_directory=run)
    assert cell.read_bytes() == b"corrupt"


def test_unavailable_fold_reasons_are_recorded_in_run_log(population_config_path):
    """Input small class-limited session; log records the scientific fit-support reason."""
    report = run_session(population_config_path)
    text = (Path(report["run_directory"]) / "run.log").read_text()
    assert "training_class_missing" in text


def test_conditional_comparison_interruption_resumes_without_sources(population_config_path, monkeypatch):
    """Input synthetic session; interrupt before conditional publication, retain references and resume."""
    original_builder = pipeline.build_switch_comparison
    original_fit = pipeline.fit_target

    def interrupt(*args, **kwargs):
        """Accept comparison inputs and simulate interruption before atomic publication."""
        raise RuntimeError("interrupted comparison")

    monkeypatch.setattr(pipeline, "build_switch_comparison", interrupt)
    with pytest.raises(RuntimeError, match="interrupted comparison"):
        run_session(population_config_path)
    run = results.discover_runs(population_config_path.parent / "analysis_runs")[0]
    published = {path: path.read_bytes() for path in (run / "cells").glob("*.npz")}
    assert len(published) == 5
    assert not (run / "cells/PFC_left_switch_vs_stay.npz").exists()
    for path in (population_config_path.parent / "trials.csv", population_config_path.parent / "A/aligned.npz"):
        path.unlink()
    monkeypatch.setattr(preparation, "prepare_session", forbid_compute)
    monkeypatch.setattr(pipeline, "build_switch_comparison", original_builder)
    fitted = []

    def record_fit(prepared, region, target):
        """Accept prepared Hz/region/target; record model ID and return ordinary fitted cell."""
        fitted.append(f"{region}_{target}")
        return original_fit(prepared, region, target)

    monkeypatch.setattr(pipeline, "fit_target", record_fit)
    assert run_session(command="resume", run_directory=run)["completed_cells"] == 14
    assert len(fitted) == 9 and fitted[0] == "PFC_left_switch_vs_stay"
    assert all(path.read_bytes() == data for path, data in published.items())
