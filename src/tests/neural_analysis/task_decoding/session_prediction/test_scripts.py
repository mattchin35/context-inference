"""Real synthetic preparation through the shared runner and CLI boundaries."""

import json

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding.session_prediction import run_session, run_batch
from src.neural_analysis.task_decoding.session_prediction.results import load_prediction_result
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as timeline_session_fixture,
)

synthetic_timeline_session = timeline_session_fixture


@pytest.fixture
def runnable_session(synthetic_timeline_session):
    """Return schema-2 metadata for 18 choices/6 blocks and seeded two-probe spikes.

    Existing UTC spike fixtures are stretched to 240 seconds; choices are ten
    seconds apart, leaving nonoverlapping [-2,+2) windows. Target units stay native.
    """
    session = synthetic_timeline_session
    root = session.session_root
    rows = np.arange(18)
    actions = (rows // 3) % 2
    origin = 1_700_000_000.0
    trials = pd.DataFrame({
        "cur_trial": rows, "cur_block": rows // 3, "action": actions,
        "state_int": actions, "correct": np.ones(18),
        "reward": np.where(rows % 4 == 0, 0, 1), "experimenter_reward_given": np.zeros(18),
        "start_time": origin + 1 + rows * 10, "choice_time": origin + 4 + rows * 10,
        "Qlearning_rel_value": np.linspace(-1, 1, 18),
        "FQlearning_rel_value": np.linspace(-2, 2, 18),
        "HMM_rel_value_logodds": np.linspace(-3, 3, 18),
        "HMM_rel_value_logodds_decay": np.linspace(-4, 4, 18),
        "relative_doubt_index": np.linspace(0, 1, 18),
    })
    trials.to_csv(root / "trials.csv", index=False)
    trials.to_csv(root / "augmented.csv", index=False)
    (root / "feature_params.json").write_text("{}")
    for probe in ("A", "B"):
        path = root / probe / "aligned.npz"
        with np.load(path) as archive:
            spikes = origin + (archive["spike_utc_unix"] - origin) * 6
            coverage = origin + (archive["irig_utc_unix"] - origin) * 6
        np.savez(path, spike_utc_unix=spikes, irig_utc_unix=coverage)
    config = {
        "session_metadata_path": "neural_session.json", "augmented_trial_path": "augmented.csv",
        "trial_feature_parameter_path": "feature_params.json",
        "pfc_region": {"region": "PFC", "probe_id": "A", "channel_labels": ["good"],
                       "require_inside_brain": True, "cluster_groups": ["good", "mua"]},
        "hpc_region": {"region": "HPC", "probe_id": "B", "channel_labels": ["good"],
                       "require_inside_brain": True, "cluster_groups": ["good", "mua"]},
    }
    (root / "task_decoding_config.json").write_text(json.dumps(config))
    return session.metadata_path


def test_dry_run_writes_and_fits_nothing(runnable_session, monkeypatch):
    """Small-source inspection reports missing PCA, grids/counts/bytes without preparation."""
    root = runnable_session.parent

    def forbidden(*args, **kwargs):
        """Forbidden numerical seam: no return; fail if dry-run reaches it."""
        pytest.fail("Dry-run cannot load spikes or fit")

    monkeypatch.setattr(run_session, "run_session_timeline", forbidden)
    monkeypatch.setattr(run_session, "fit_region_prediction_models", forbidden)
    report = run_session.prepare_session_predictions(runnable_session, probe_roles={"PFC": "A", "HPC": "B"}, dry_run=True)
    assert report.status == "dry-run"
    assert report.result_path is None
    assert report.details["source_pca_missing"]
    assert report.details["max_model_count"] == 70
    assert report.details["prediction_bytes_upper_bound"] > 0
    assert report.details["source_selection"]["PFC"]["unit_count"] > 0
    assert "unit_ids" not in report.details["source_selection"]["PFC"]
    assert not (root / "analysis_runs").exists()


def test_prepare_reuse_rerun_and_source_snapshots(runnable_session, monkeypatch):
    """A real seeded pipeline fits once, reuses by identity and preserves old reruns."""
    roles = {"PFC": "A", "HPC": "B"}
    first = run_session.prepare_session_predictions(runnable_session, probe_roles=roles)
    loaded = load_prediction_result(first.result_path)
    assert first.status == "prepared"
    assert set(loaded.regions) == {"PFC", "HPC"}
    assert all(model.predictions.shape[1:] == (8, 7) for model in loaded.regions.values())
    directory = first.result_path.parent
    for name in ("run_session.py", "run_batch.py", "summary.md", "run.log", "config.json"):
        assert (directory / name).is_file()
    assert (directory / "source_snapshots/modeling.py").is_file()
    assert "Block-held-out" in (directory / "summary.md").read_text()
    original_fit = run_session.fit_region_prediction_models

    def forbidden(*args, **kwargs):
        """Reuse seam: any fitting would be a failure; no return."""
        pytest.fail("Matching preparation must be reused")

    monkeypatch.setattr(run_session, "fit_region_prediction_models", forbidden)
    second = run_session.prepare_session_predictions(runnable_session, probe_roles=roles)
    assert second.status == "reused" and second.result_path == first.result_path
    monkeypatch.setattr(run_session, "fit_region_prediction_models", original_fit)
    third = run_session.prepare_session_predictions(runnable_session, probe_roles=roles, rerun=True)
    assert third.result_path != first.result_path
    assert load_prediction_result(third.result_path).meta["source_pca"] == loaded.meta["source_pca"]


def test_controls_and_explicit_bin_probe_overrides(runnable_session):
    """100-ms choices use their matching PCA; an explicit one-region override is honored."""
    controls = runnable_session.parent / "session_prediction_config.json"
    controls.write_text(json.dumps({"bin_width_ms": 100}))
    report = run_session.prepare_session_predictions(runnable_session, probe_roles={"HPC": "B"})
    loaded = load_prediction_result(report.result_path)
    assert set(loaded.regions) == {"HPC"}
    assert loaded.regions["HPC"].predictions.shape[1] == 40
    assert loaded.inputs.training_bin_mask.sum() == 20
    assert loaded.meta["effective_config"]["bin_width_ms"] == 100


def test_version2_without_roles_and_missing_base_config_are_actionable(runnable_session):
    """Do not silently label an unassigned schema-2 probe as an anatomical region."""
    with pytest.raises(ValueError, match="probe|role"):
        run_session.prepare_session_predictions(runnable_session, dry_run=True)
    (runnable_session.parent / "task_decoding_config.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        run_session.prepare_session_predictions(runnable_session, probe_roles={"PFC": "A"}, dry_run=True)


def test_cli_and_batch_dry_run_use_shared_runner(runnable_session, capsys):
    """Documented metadata CLI and session-list batch stay thin and never submit jobs."""
    assert run_session.main(["--session-metadata", str(runnable_session), "--pfc-probe", "A", "--dry-run"]) == 0
    assert "dry-run" in capsys.readouterr().out
    listing = runnable_session.parent / "sessions.txt"
    listing.write_text("neural_session.json\n")
    assert run_batch.main(["--session-list", str(listing), "--pfc-probe", "A", "--dry-run"]) == 0
    assert not (runnable_session.parent / "analysis_runs").exists()
