"""Offline session command contracts on small seeded, IRIG-backed source files."""

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from src.neural_analysis.encoding import pipeline, results, run_session
from src.neural_analysis.encoding.configuration import EncodingConfig
from src.neural_analysis.population import session_timeline_io


def forbid_loading(*args: object, **kwargs: object) -> None:
    """Fail on any attempted raw population load; arbitrary inputs, no outputs/units."""
    pytest.fail("This command must consume small metadata or immutable saved results")


def interrupt_before_cells(*args: object, **kwargs: object) -> None:
    """Raise a simulated interruption after preparation; arbitrary inputs, no outputs."""
    raise RuntimeError("synthetic interruption before cells")


def config_path(config: EncodingConfig) -> Path:
    """Input validated config; return its fixture JSON Path, with no physical units."""
    return config.session_root / "encoding.json"


def test_dry_run_reports_behavior_and_cells_without_raw_loading_or_writes(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input fixture config/patch; report eight baseline trials and 48 cells, return None."""
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_loading)
    report = run_session.run_encoding_session(config_path(encoding_session), command="dry-run")
    assert report["status"] == "dry-run"
    assert report["baseline_trials"] == 8
    assert report["planned_cells"] == 48
    assert report["neural_counts_pending"]
    assert not encoding_session.run_root.exists()
    assert not encoding_session.intermediate_root.exists()


def test_dry_run_reports_deferred_manual_alignment_without_spike_extrema(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input config/patch; missing IRIG member is an explicit regional blocker, return None."""
    path = encoding_session.session_root / "A/aligned.npz"
    np.savez(path, spike_utc_unix=np.array([1_700_000_000.0, 1_700_000_120.0]))
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_loading)
    report = run_session.run_encoding_session(config_path(encoding_session), command="dry-run")
    reason = report["input_blockers"]["PFC"].lower()
    assert "irig" in reason and ("manual" in reason or "deferred" in reason)
    assert not encoding_session.run_root.exists()


def test_new_reuses_completed_run_and_saves_session_local_audit_files(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input config/patch; run, saved products and reuse remain in the source session, return None."""
    first = run_session.run_encoding_session(config_path(encoding_session))
    directory = Path(first["run_directory"])
    assert first["status"] == "complete"
    assert directory.parent == encoding_session.run_root
    for name in ("config.json", "progress.json", "run.log", "summary.md", "run_session.py", "run_batch.py"):
        assert (directory / name).is_file()
    saved = results.load_encoding_run(directory)
    assert len(saved["cells"]) == 48
    preparation = encoding_session.session_root / saved["manifest"]["preparation_path"]
    assert preparation.is_relative_to(encoding_session.intermediate_root)
    assert preparation.is_file()
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_loading)
    again = run_session.run_encoding_session(config_path(encoding_session))
    assert again["status"] == "reused"
    assert again["run_directory"] == first["run_directory"]


def test_rerun_preserves_prior_products_in_a_new_timestamped_directory(
    encoding_session: EncodingConfig,
) -> None:
    """Input config; explicit rerun creates a distinct run, preserving original bytes, return None."""
    first = run_session.run_encoding_session(config_path(encoding_session))
    old_path = Path(first["run_directory"]) / "config.json"
    original = old_path.read_bytes()
    second = run_session.run_encoding_session(config_path(encoding_session), rerun=True)
    assert second["status"] == "complete"
    assert second["run_directory"] != first["run_directory"]
    assert old_path.read_bytes() == original


def test_alpha_change_reuses_preparation_without_loading_spikes(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input config/patch; changed dimensionless alpha changes model only, return None."""
    path = config_path(encoding_session)
    first = run_session.run_encoding_session(path)
    original = results.load_run_manifest(Path(first["run_directory"]))
    payload = json.loads(path.read_text())
    payload["alpha"] = 3.0
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_loading)
    second = run_session.run_encoding_session(path)
    changed = results.load_run_manifest(Path(second["run_directory"]))
    assert second["status"] == "complete"
    assert original["identities"]["preparation_id"] == changed["identities"]["preparation_id"]
    assert original["identities"]["model_id"] != changed["identities"]["model_id"]
    assert original["preparation_path"] == changed["preparation_path"]


def test_partial_requires_explicit_resume_and_uses_saved_inputs(
    encoding_session: EncodingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input config/patch; resume published preparation despite absent live files, return None."""
    actual_pipeline = pipeline.run_prepared_encoding
    monkeypatch.setattr(pipeline, "run_prepared_encoding", interrupt_before_cells)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run_session.run_encoding_session(config_path(encoding_session))
    directories = list(encoding_session.run_root.glob("context_encoding_*"))
    assert len(directories) == 1
    directory = directories[0]
    report = run_session.run_encoding_session(config_path(encoding_session))
    assert report["status"] == "partial"
    assert Path(report["run_directory"]) == directory
    monkeypatch.setattr(pipeline, "run_prepared_encoding", actual_pipeline)
    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbid_loading)
    encoding_session.augmented_trials.unlink()
    encoding_session.feature_parameters.unlink()
    config_path(encoding_session).unlink()
    status = run_session.run_encoding_session(command="status", run_directory=directory)
    assert status["status"] != "complete"
    resumed = run_session.run_encoding_session(command="resume", run_directory=directory)
    assert resumed["status"] == "complete"
    assert len(results.load_encoding_run(directory)["cells"]) == 48


def test_main_parses_dry_run_and_status_errors(
    encoding_session: EncodingConfig, capsys: pytest.CaptureFixture,
) -> None:
    """Input config/captured terminal; public CLI prints JSON and useful errors, return None."""
    assert run_session.main(["dry-run", "--config", str(config_path(encoding_session))]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "dry-run"
    assert run_session.main(["status", "--run-directory", str(encoding_session.run_root / "absent")]) == 1
    assert capsys.readouterr().err


def test_cli_limits_nested_blas_before_importing_numerical_libraries() -> None:
    """No inputs; fresh UV Python process inspects module imports/thread settings, return None."""
    script = (
        "import os, sys; import src.neural_analysis.encoding.run_session; "
        "import src.neural_analysis.encoding.run_batch; "
        "assert 'numpy' not in sys.modules; "
        "assert all(os.environ[k] == '1' for k in "
        "('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'))"
    )
    completed = subprocess.run(["uv", "run", "--frozen", "--no-sync", "python", "-c", script],
                               capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
