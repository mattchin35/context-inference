"""Small seeded source sessions shared by context-encoding scientific tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.encoding.configuration import EncodingConfig, load_encoding_config


@pytest.fixture
def encoding_session(tmp_path: Path) -> EncodingConfig:
    """Create a seeded two-probe session and return its validated EncodingConfig.

    Input tmp_path is an empty pytest directory. Source tables have 12 trials,
    irregular IDs and UTC-second timestamps. Aligned spikes are (S,) UTC seconds
    and cluster assignments (S,); IRIG arrays define independent trusted probe
    bounds. No files outside the temporary session are touched. Seed=20261103.
    """
    root = tmp_path / "encoding-session"
    root.mkdir()
    (root / "processed").mkdir()
    origin = 1_700_000_000.0
    trials = pd.DataFrame({
        "cur_trial": [10, 12, 17, 19, 24, 30, 31, 39, 40, 43, 51, 60],
        "cur_block": np.arange(12),
        "start_time": origin + np.arange(12) * 10.0,
        "choice_time": origin + np.arange(12) * 10.0 + 4.0,
        "state_int": [0, 0, 1, 0, 2, 1, 0, 1, 0, 1, 0, 1],
        "action": [0, 0, 1, "no_choice", 0, 1, 0, 1, 0, 1, 0, 1],
        "experimenter_reward_given": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "reward": [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    trials.to_csv(root / "trials.csv", index=False)
    trials.to_csv(root / "processed/augmented.csv", index=False)
    (root / "processed/trial_feature_params.json").write_text(
        json.dumps({"random_seed": 20261103}), encoding="ascii")
    pd.DataFrame({"Time": origin + np.array([0.0, 14.2, 24.2, 34.2, 44.2, 54.2]),
        "Event": ["enter_right_patch", "enter_left_patch", "enter_right_patch",
                  "enter_dark_period", "enter_left_patch", "enter_right_patch"],
    }).to_csv(root / "events.csv", index=False)
    probes = {}
    rng = np.random.default_rng(20261103)
    for probe, lower in (("A", 0.0), ("B", 62.0)):
        directory = root / probe
        directory.mkdir()
        groups = ["good", "mua", "good", "noise"]
        pd.DataFrame({"cluster_id": [7, 8, 9, 10], "ch": [0, 1, 2, 0],
            "group": groups, "KSLabel": ["mua", "good", "good", "good"],
        }).to_csv(directory / "cluster_info.tsv", sep="\t", index=False)
        pd.DataFrame({"channel_id": ["CH0", "CH1", "CH2"],
            "label": ["good", "good", "bad"], "inside_brain": [True, True, True],
            "x_um": [0.0, 0.0, 0.0], "y_um": [0.0, 20.0, 40.0],
        }).to_csv(directory / "channel_quality.csv", index=False)
        spikes = origin + rng.uniform(0.0, 120.0, size=(4, 600))
        assignments = np.repeat([7, 8, 9, 10], 600)
        order = np.argsort(spikes.ravel())
        np.save(directory / "spike_clusters.npy", assignments[order])
        np.savez(directory / "aligned.npz", spike_utc_unix=spikes.ravel()[order],
                 irig_utc_unix=origin + np.arange(lower, 121.0))
        probes[probe] = {"lfp": "", "alignment": f"{probe}/aligned.npz",
            "sorter": probe, "quality": f"{probe}/channel_quality.csv", "sites": {probe: 0}}
    metadata = {"schema_version": "2", "session": "synthetic-context-encoding",
        "acquisition": "open_ephys", "behavior": {"trials": "trials.csv", "events": "events.csv"},
        "probes": probes, "site_pairs": []}
    (root / "neural_session.json").write_text(json.dumps(metadata), encoding="ascii")
    config_path = root / "encoding.json"
    config_path.write_text(json.dumps({"session_metadata": "neural_session.json",
        "augmented_trials": "processed/augmented.csv",
        "feature_parameters": "processed/trial_feature_params.json",
        "probe_roles": {"PFC": "A", "HPC": "B"}}), encoding="ascii")
    return load_encoding_config(config_path)
