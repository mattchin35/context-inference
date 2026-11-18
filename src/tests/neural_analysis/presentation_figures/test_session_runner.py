"""Session-level publication and reuse are independent of experimental data."""

import json
from pathlib import Path

import numpy as np

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
