"""Session-level publication and reuse are independent of experimental data."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.neural_analysis.presentation_figures import run_session as runner


def test_session_dry_run_and_compatible_run_reuse(synthetic_presentation_session, monkeypatch):
    """A metadata Path produces one complete immutable run per scientific identity."""
    metadata = synthetic_presentation_session
    computed = []

    def synthetic_stages(prepared, seed):
        """Record a fit and yield four named synthetic stage payloads."""
        computed.append(seed)
        for stage in ("regional", "replays", "prediction", "movement"):
            yield stage, {"trial_ids": np.array([101, 102])}

    monkeypatch.setattr(runner, "compute_stages", synthetic_stages)
    run_root = metadata.parent / "analysis_runs"
    preview = runner.run_session(metadata, dry_run=True)
    assert preview["status"] == "dry-run"
    assert preview["good_units"] == {"PFC": 8, "HPC": 8}
    assert computed == [] and not run_root.exists()

    first = runner.run_session(metadata, seed=3)
    manifest = Path(first["manifest"])
    saved = json.loads(manifest.read_text())
    assert set(saved["archives"]) == {"regional", "replays", "prediction", "movement"}
    assert all((manifest.parent / file).exists() for file in saved["archives"].values())
    assert {"run_session.py", "run_batch.py", "summary.md", "run.log"} <= {
        file.name for file in manifest.parent.iterdir()
    }
    assert runner.run_session(metadata, seed=3)["manifest"] == str(manifest)
    other = runner.run_session(metadata, seed=7)
    assert other["manifest"] != str(manifest)
    assert computed == [3, 7]
    # A changed source must invalidate otherwise identical scientific settings.
    source = metadata.parent / "events.csv"
    with source.open("a") as stream:
        stream.write("\n")
    refreshed = runner.run_session(metadata, seed=3)
    assert refreshed["manifest"] != str(manifest) and computed == [3, 7, 3]
    with np.load(manifest.parent / "regional.npz", allow_pickle=True) as archive:
        assert archive["meta"].item()["producer"].endswith("decoding.compute_decoding")


def test_reuse_reviewed_replays_and_movement(synthetic_presentation_session, monkeypatch):
    """Revising the PC analyses reuses unchanged stage fields from the named session."""
    from src.neural_analysis.presentation_figures import decoding, movement, prediction, replays

    metadata = synthetic_presentation_session
    marker = {"trial_ids": np.array([11, 12])}
    for module, name in ((decoding, "compute_decoding"), (movement, "compute_movement"),
                         (prediction, "compute_prediction"), (replays, "compute_replays")):
        monkeypatch.setattr(module, name, lambda *args, **kwargs: marker)
    old = Path(runner.run_session(metadata)["manifest"])
    def unexpected_fit(*args, **kwargs):
        """Fail if either unchanged stage is recomputed."""
        pytest.fail("Unexpected replay or movement refit")
    monkeypatch.setattr(movement, "compute_movement", unexpected_fit)
    monkeypatch.setattr(replays, "compute_replays", unexpected_fit)
    new = Path(runner.run_session(metadata, rerun=True, reuse_fixed_from=old)["manifest"])
    for stage in ("replays", "movement"):
        with np.load(new.parent / (stage + ".npz"), allow_pickle=True) as archive:
            np.testing.assert_equal(archive["trial_ids"], marker["trial_ids"])
            header = archive["meta"].item()
            assert header["reused_from_manifest"] == str(old)
            assert header["parameters"]["panel_d_pcs"] == 30
    with (metadata.parent / "events.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source|incompatible"):
        runner.run_session(metadata, reuse_fixed_from=old)
