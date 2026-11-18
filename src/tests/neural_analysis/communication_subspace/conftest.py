"""Small on-disk session with known spike and behavioral identities."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_communication_session(tmp_path):
    """Return metadata Path for 45 choices, 8 good units/region, UTC-second spikes.

    The choice axis is chronological and spaced by 10 s. Each probe also has
    MUA, noise, and unlabeled clusters, which must not enter the good-unit axis.
    Both probes deliberately reuse cluster numbers to test qualified identities.
    """
    root = tmp_path / "communication"
    root.mkdir()
    origin = 1_700_000_000.0
    choices = origin + 10.0 + 10.0 * np.arange(45)
    trials = pd.DataFrame({
        "cur_trial": np.arange(100, 145),
        "start_time": choices - 4.0,
        "choice_time": choices,
        "action": np.arange(45) % 2,
        "reward": np.arange(45) % 3 == 0,
        "experimenter_reward_given": np.zeros(45, dtype=int),
        "context_label": ["left" if index % 2 else "right" for index in range(45)],
        "outcome_note": [f"outcome-{index}" for index in range(45)],
    })
    trials.to_csv(root / "trials.csv", index=False)
    probes = {}
    for probe_id in ("A", "B"):
        directory = root / probe_id
        directory.mkdir()
        times = []
        clusters = []
        for unit in range(11):
            unit_times = choices + (unit % 4) * 0.5 - 2.0
            unit_times = np.r_[unit_times, origin + 5.0 + unit * 0.01]
            times.extend(unit_times)
            clusters.extend([unit] * len(unit_times))
        order = np.argsort(times, kind="stable")
        np.save(directory / "spike_clusters.npy", np.asarray(clusters)[order])
        np.savez(directory / "aligned.npz", spike_utc_unix=np.asarray(times)[order],
                 irig_utc_unix=np.asarray([origin, origin + 470.0]))
        pd.DataFrame({
            "cluster_id": np.arange(11),
            "ch": np.arange(11),
            "group": ["good"] * 8 + ["mua", "noise", ""],
            "KSLabel": ["good"] * 11,
        }).to_csv(directory / "cluster_info.tsv", sep="\t", index=False)
        pd.DataFrame({
            "channel_id": [f"CH{index}" for index in range(11)],
            "label": ["good"] * 11,
            "inside_brain": [True] * 11,
            "x_um": np.zeros(11),
            "y_um": 20 * np.arange(11),
        }).to_csv(directory / "channel_quality.csv", index=False)
        probes[probe_id] = {
            "lfp": "", "alignment": f"{probe_id}/aligned.npz",
            "sorter": probe_id, "quality": f"{probe_id}/channel_quality.csv",
            "sites": {},
        }
    metadata = {
        "schema_version": "2", "session": "synthetic-communication",
        "acquisition": "open_ephys",
        "behavior": {"trials": "trials.csv"},
        "probes": probes, "site_pairs": [],
    }
    path = root / "neural_session.json"
    path.write_text(json.dumps(metadata), encoding="ascii")
    return path
