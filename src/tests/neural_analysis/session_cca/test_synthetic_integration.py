"""Seeded spike-to-view acceptance and real spawned-session execution."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population.cca import transform_cca_activity
from src.neural_analysis.population import run_session_cca as runner
from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata


ROLES = {"PFC": "A", "HPC": "B"}


@pytest.fixture
def shared_private_session(tmp_path):
    """Return metadata for seeded shared/private counts, constants and unequal coverage.

    Eighteen choices are 14 seconds apart. Spikes lie at 10-ms bin centers in
    real UTC seconds, with dependent cloned units and a silent good unit in
    each probe. PFC excludes the first choice, HPC the last; 16 windows pair.
    No experimental data or expected population-correlation value is assumed.
    """
    root = tmp_path / "shared_private"
    root.mkdir()
    origin = 1_700_000_000.0
    choice_elapsed = 4.123 + 14 * np.arange(18)
    choices = origin + choice_elapsed
    trials = pd.DataFrame({
        "cur_trial": np.arange(101, 119), "start_time": choices - 4.123,
        "choice_time": choices, "led_on_time": choices - 2.5,
        "action": np.arange(18) % 2, "reward": np.arange(18) % 3 == 0,
        "experimenter_reward_given": np.zeros(18, dtype=int),
    })
    trials.to_csv(root / "trials.csv", index=False)
    events = pd.DataFrame({
        "Time": origin + np.array([0, 2, 6, 80, 160, 230]),
        "Event": ["trial_start", "LED_on", "LED_off", "enter_left_patch",
                  "enter_right_patch", "enter_dark_period"],
        "Note": [""] * 6,
    })
    events.to_csv(root / "events.csv", index=False)
    rng = np.random.default_rng(20261113)
    shared = rng.poisson(.18, (18, 400))
    pfc_private = rng.poisson(.12, (18, 400, 2))
    hpc_private = rng.poisson(.12, (18, 400, 3))
    silent = np.zeros_like(shared)
    counts = {
        "A": np.stack([shared, shared + pfc_private[..., 0],
                       pfc_private[..., 1], shared, silent], axis=-1),
        "B": np.stack([shared + hpc_private[..., 0], hpc_private[..., 1],
                       hpc_private[..., 2], shared + hpc_private[..., 0],
                       silent, shared], axis=-1),
    }
    probes = {}
    centers = choices[:, None] - 2 + (np.arange(400) + .5) * .01
    for probe_id, activity in counts.items():
        directory = root / probe_id
        directory.mkdir()
        unit_count = activity.shape[-1]
        times, clusters = [], []
        for unit in range(unit_count):
            unit_times = np.repeat(centers.ravel(), activity[..., unit].ravel())
            # Sorter populations include every unit; this gap spike keeps the
            # task-silent unit represented without changing any fitted bin.
            unit_times = np.r_[unit_times, origin + 12]
            times.extend(unit_times)
            clusters.extend([unit] * len(unit_times))
        order = np.argsort(times, kind="stable")
        np.save(directory / "spike_clusters.npy", np.asarray(clusters)[order])
        coverage = [origin + 10, origin + 246] if probe_id == "A" else [origin, origin + 235]
        np.savez(directory / "aligned.npz", spike_utc_unix=np.asarray(times)[order],
                 irig_utc_unix=np.asarray(coverage))
        pd.DataFrame({"cluster_id": np.arange(unit_count), "ch": np.arange(unit_count),
                      "group": ["good"] * unit_count}).to_csv(
                          directory / "cluster_info.tsv", sep="\t", index=False)
        pd.DataFrame({"channel_id": [f"CH{i}" for i in range(unit_count)],
                      "label": ["good"] * unit_count, "inside_brain": [True] * unit_count,
                      "x_um": np.zeros(unit_count), "y_um": 20 * np.arange(unit_count)}).to_csv(
                          directory / "channel_quality.csv", index=False)
        probes[probe_id] = {"lfp": "", "alignment": f"{probe_id}/aligned.npz",
                            "sorter": probe_id, "quality": f"{probe_id}/channel_quality.csv",
                            "sites": {}}
    metadata = {"schema_version": "2", "session": "synthetic-shared-private",
                "acquisition": "open_ephys", "behavior": {"trials": "trials.csv", "events": "events.csv"},
                "probes": probes, "site_pairs": []}
    path = root / "neural_session.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(path), path)


@pytest.mark.parametrize("width,bins", [(.5, 8), (.1, 40), (.05, 80), (.02, 200)])
def test_seeded_pipeline_replay_and_display_preserve_all_fitted_observations(shared_private_session, width, bins):
    """Prepare/load/replay/full+bounded plotting/reuse/rerun obey one paired basis."""
    from src.neural_analysis.population.cca_plotting import plot_session_cca

    session = shared_private_session
    report = runner.run_session_cca(session.metadata_path, ROLES, width)
    result = cca_io.load_cca_result(report.result_path)
    inputs = cca_io.load_cca_inputs(session, ROLES, width)
    assert len(result.choices) == 18
    np.testing.assert_array_equal(result.choice_positions, np.arange(1, 17))
    assert result.fit.n_observations == 16 * bins
    assert len(result.fit.correlations) == 3
    for region in ROLES:
        transform = result.fit.transforms[region]
        assert transform.omitted_unit_ids == (f"{ROLES[region]}:4",)
        projected = transform_cca_activity(inputs.counts[region], inputs.unit_ids[region], transform)
        np.testing.assert_allclose(projected, result.fit.scores[region], rtol=1e-8, atol=1e-10)
        flat = projected.reshape(-1, 3)
        np.testing.assert_allclose(flat.mean(0), 0, rtol=0, atol=1e-8)
        np.testing.assert_allclose(flat.T @ flat / (len(flat) - 1), np.eye(3), rtol=0, atol=1e-8)
    x, y = [result.fit.scores[region].reshape(-1, 3) for region in ROLES]
    np.testing.assert_allclose(x.T @ y / (len(x) - 1), np.diag(result.fit.correlations), rtol=0, atol=1e-8)
    before = report.result_path.read_bytes()
    for loaded in (None, (10.0, 70.0)):
        figure = plot_session_cca(result, loaded_range_s=loaded, viewport_s=(10, 70))
        heatmaps = [trace for trace in figure.data if trace.type == "heatmap"]
        assert len(heatmaps) == 2
        assert heatmaps[0].z.shape == heatmaps[1].z.shape
        finite_columns = np.isfinite(heatmaps[0].z[0]).sum()
        assert finite_columns == (16 if loaded is None else 4) * bins
        assert "canonical" in figure.to_json().lower()
    reused = runner.run_session_cca(session.metadata_path, ROLES, width)
    assert reused.status == "reused" and reused.result_path == report.result_path
    again = runner.run_session_cca(session.metadata_path, ROLES, width, rerun=True)
    assert again.result_path != report.result_path
    assert report.result_path.read_bytes() == before
    for region in ROLES:
        np.testing.assert_array_equal(cca_io.load_cca_result(again.result_path).fit.scores[region],
                                      result.fit.scores[region])


def test_actual_spawn_isolates_failure_and_completes_other_session(shared_private_session):
    """Real batch CLI spawn processes isolate failure and preserve input report order.

    Run in a fresh interpreter because Streamlit AppTest replaces __main__
    with a temporary script that cannot be imported by multiprocessing spawn.
    """
    session = shared_private_session
    invalid = session.metadata_path.parent / "invalid.json"
    # The CLI requires existing list entries; malformed JSON fails inside a worker.
    invalid.write_text("{invalid json}", encoding="ascii")
    session_list = session.session_root / "batch_sessions.txt"
    session_list.write_text(f"{invalid}\n{session.metadata_path}\n", encoding="ascii")
    process = subprocess.run(
        [sys.executable, "-m", "src.neural_analysis.population.run_session_cca_batch",
         "--session-list", str(session_list), "--pfc-probe", "A", "--hpc-probe", "B", "--workers", "2"],
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert process.returncode == 1, process.stderr
    lines = process.stdout.splitlines()
    assert len(lines) == 2, (process.stdout, process.stderr)
    assert lines[0].startswith(f"{invalid}: failed ")
    assert lines[1].startswith(f"{session.metadata_path}: prepared ")
    loaded = cca_io.load_cca_result(Path(lines[1].split(": prepared ", 1)[1]))
    assert loaded.fit.scores["PFC"].shape == (16, 8, 3)
