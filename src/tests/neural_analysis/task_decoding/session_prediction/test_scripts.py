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


@pytest.mark.parametrize("mode, target_count", [("correct_rewarded_generalization", 7), ("mixed_outcome_decoding", 8)])
@pytest.mark.parametrize("window, expected_training_bins", [("full", 4), ("before", 2), ("after", 2)])
def test_prepare_reuse_rerun_and_source_snapshots(
    runnable_session, monkeypatch, mode, target_count, window, expected_training_bins,
):
    """Each mode/window fits once, reuses its matching identity and preserves reruns."""
    roles = {"PFC": "A", "HPC": "B"}
    first = run_session.prepare_session_predictions(
        runnable_session, probe_roles=roles, analysis_mode=mode, training_window=window,
    )
    loaded = load_prediction_result(first.result_path)
    assert first.status == "prepared"
    assert set(loaded.regions) == {"PFC", "HPC"}
    assert all(model.predictions.shape[1:] == (8, target_count) for model in loaded.regions.values())
    assert first.details["analysis_mode"] == mode
    assert first.details["max_model_count"] == target_count * 2 * 5
    assert first.details["prediction_bytes_upper_bound"] == 18 * 8 * target_count * 8 * 2
    assert loaded.inputs.training_window == window
    assert loaded.inputs.training_bin_mask.sum() == expected_training_bins
    assert loaded.meta["fit_count"] <= target_count * 2 * 5
    directory = first.result_path.parent
    for name in ("run_session.py", "run_batch.py", "summary.md", "run.log", "config.json"):
        assert (directory / name).is_file()
    assert (directory / "source_snapshots/modeling.py").is_file()
    assert "Block-held-out" in (directory / "summary.md").read_text()
    assert mode in (directory / "summary.md").read_text()
    summary_text = (directory / "summary.md").read_text()
    assert f"({window})" in summary_text
    assert mode in (directory / "run.log").read_text()
    if mode == "mixed_outcome_decoding":
        assert "relative to average loss" in (directory / "summary.md").read_text()
        assert "class 1 is rewarded" in summary_text
        assert "eventual recorded reward" in summary_text
        assert "post-choice, not necessarily post-delivery" in summary_text
        assert "not subjective expectation" in summary_text
    original_fit = run_session.fit_region_prediction_models

    def forbidden(*args, **kwargs):
        """Reuse seam: any fitting would be a failure; no return."""
        pytest.fail("Matching preparation must be reused")

    monkeypatch.setattr(run_session, "fit_region_prediction_models", forbidden)
    second = run_session.prepare_session_predictions(
        runnable_session, probe_roles=roles, analysis_mode=mode, training_window=window,
    )
    assert second.status == "reused" and second.result_path == first.result_path
    monkeypatch.setattr(run_session, "fit_region_prediction_models", original_fit)
    third = run_session.prepare_session_predictions(
        runnable_session, probe_roles=roles, analysis_mode=mode, training_window=window, rerun=True,
    )
    assert third.result_path != first.result_path
    assert load_prediction_result(third.result_path).meta["source_pca"] == loaded.meta["source_pca"]


def test_controls_and_explicit_bin_probe_overrides(runnable_session):
    """100-ms choices use their matching PCA; an explicit one-region override is honored."""
    controls = runnable_session.parent / "session_prediction_config.json"
    controls.write_text(json.dumps({"bin_width_ms": 100, "training_window": "before"}))
    report = run_session.prepare_session_predictions(runnable_session, probe_roles={"HPC": "B"})
    loaded = load_prediction_result(report.result_path)
    assert set(loaded.regions) == {"HPC"}
    assert loaded.regions["HPC"].predictions.shape[1] == 40
    assert loaded.inputs.training_bin_mask.sum() == 10
    assert loaded.inputs.training_window == "before"
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
    assert run_session.main([
        "--session-metadata", str(runnable_session), "--pfc-probe", "A",
        "--training-window", "after", "--dry-run",
    ]) == 0
    assert "dry-run" in capsys.readouterr().out
    listing = runnable_session.parent / "sessions.txt"
    listing.write_text("neural_session.json\n")
    assert run_batch.main([
        "--session-list", str(listing), "--pfc-probe", "A",
        "--training-window", "before", "--dry-run",
    ]) == 0
    assert not (runnable_session.parent / "analysis_runs").exists()


@pytest.mark.parametrize("width", [500, 100])
def test_mixed_controls_train_eight_targets_and_reuse_exact_pca(runnable_session, width):
    """A separate mixed product reuses source PCA and never replaces the original run."""
    roles = {"PFC": "A", "HPC": "B"}
    reference = run_session.prepare_session_predictions(runnable_session, probe_roles=roles, bin_width_ms=width)
    original_bytes = reference.result_path.read_bytes()
    source = load_prediction_result(reference.result_path).meta["source_pca"]
    controls = runnable_session.parent / "session_prediction_config.json"
    controls.write_text(json.dumps({"analysis_mode": "mixed_outcome_decoding", "bin_width_ms": width}))
    mixed = run_session.prepare_session_predictions(runnable_session, probe_roles=roles)
    loaded = load_prediction_result(mixed.result_path)
    assert loaded.inputs.target_names == ("current_state", "current_action", "current_reward",
                                         "qlearning_relative_value", "forgetting_q_relative_value",
                                         "hmm_signed_belief", "hmm_decay_signed_belief", "relative_doubt")
    assert loaded.meta["source_pca"] == source
    assert mixed.identity != reference.identity
    assert reference.result_path.read_bytes() == original_bytes
    assert loaded.meta["fit_count"] <= 80
    assert mixed.details["max_model_count"] == 80
    explicit = run_session.prepare_session_predictions(
        runnable_session, probe_roles=roles, analysis_mode="correct_rewarded_generalization",
    )
    assert explicit.status == "reused" and explicit.result_path == reference.result_path


def test_mixed_cli_batch_forwarding_and_dry_run(runnable_session, capsys):
    """Both entry points forward canonical mode overrides without fitting or writing."""
    assert run_session.main([
        "--session-metadata", str(runnable_session), "--pfc-probe", "A",
        "--analysis-mode", "mixed_outcome_decoding", "--dry-run",
    ]) == 0
    output = capsys.readouterr().out
    assert '"max_model_count": 40' in output
    assert "mixed_outcome_decoding" in output
    reports = run_batch.run_prediction_batch(
        (runnable_session,), probe_roles={"PFC": "A"},
        analysis_mode="mixed_outcome_decoding", dry_run=True,
    )
    assert reports[0].details["analysis_mode"] == "mixed_outcome_decoding"
    listing = runnable_session.parent / "mixed_sessions.txt"
    listing.write_text("neural_session.json\n")
    assert run_batch.main([
        "--session-list", str(listing), "--pfc-probe", "A", "--analysis-mode",
        "mixed_outcome_decoding", "--dry-run",
    ]) == 0
    assert not (runnable_session.parent / "analysis_runs").exists()


def test_training_window_control_precedence_and_dry_run_contract(runnable_session):
    """CLI overrides controls; dry-run reports matched choice-relative bounds and masks."""
    root = runnable_session.parent
    controls = root / "session_prediction_config.json"
    controls.write_text(json.dumps({"training_window": "before"}))
    controlled = run_session.prepare_session_predictions(
        runnable_session, probe_roles={"PFC": "A"}, dry_run=True,
    )
    assert controlled.details["training_window"] == "before"
    assert controlled.details["training_window_s"] == [-1.0, 0.0]
    assert controlled.details["training_bins"] == 2
    assert controlled.details["scoring_window_s"] == [-1.0, 0.0]
    assert controlled.details["training_conditions"] == ["correct_rewarded"]
    assert controlled.details["candidate_outcome_counts"]
    assert controlled.details["max_model_count"] == 35

    overridden = run_session.prepare_session_predictions(
        runnable_session, probe_roles={"PFC": "A"}, training_window="after", dry_run=True,
    )
    assert overridden.details["training_window"] == "after"
    assert overridden.details["training_window_s"] == [0.0, 1.0]
    assert overridden.details["scoring_window_s"] == [0.0, 1.0]


def test_session_and_batch_cli_forward_explicit_window(runnable_session, monkeypatch, capsys):
    """Both CLI layers forward a selected half-window into the common runner."""
    calls = []

    def capture(*args, **kwargs):
        """Capture Path positional input and keyword controls; return an empty dry-run report.

        Inputs are metadata Paths plus JSON-compatible scalar/None controls;
        output is PredictionRunReport with no result path and no numerical arrays.
        """
        calls.append(kwargs)
        return run_session.PredictionRunReport("dry-run", None, "identity", 0.0, {})

    monkeypatch.setattr(run_session, "prepare_session_predictions", capture)
    monkeypatch.setattr(run_batch, "prepare_session_predictions", capture)
    assert run_session.main([
        "--session-metadata", str(runnable_session), "--training-window", "after", "--dry-run",
    ]) == 0
    listing = runnable_session.parent / "sessions.txt"
    listing.write_text("neural_session.json\n")
    assert run_batch.main([
        "--session-list", str(listing), "--training-window", "before", "--dry-run",
    ]) == 0
    assert calls[0]["training_window"] == "after"
    assert calls[1]["training_window"] == "before"
