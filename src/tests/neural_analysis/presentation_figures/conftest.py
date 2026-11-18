"""Shared on-disk, two-probe synthetic session for presentation-figure tests."""

import json

import numpy as np
import pandas as pd
import pytest

from src.tests.neural_analysis.communication_subspace.conftest import (
    synthetic_communication_session,
)


@pytest.fixture
def synthetic_presentation_session(synthetic_communication_session):
    """Return a v3 metadata Path for 45 chronological trial rows and UTC events.

    There are nine five-row blocks, eight good units per probe, and one
    ordinary choice with two-second but without four-second joint coverage.
    """
    path = synthetic_communication_session
    trials_path = path.parent / "trials.csv"
    trials = pd.read_csv(trials_path)
    trials["cur_block"] = 20 + np.arange(len(trials)) // 5
    trials["state_int"] = trials.cur_block % 2
    trials["correct"] = (trials.action == trials.state_int).astype(float)
    trials["reward"] = trials.reward.astype(float)
    trials.loc[0, ["correct", "reward"]] = [1, 0]
    trials.loc[1, "experimenter_reward_given"] = 1
    trials.loc[2, "state_int"] = np.nan
    trials.loc[3, "reward"] = np.nan
    trials.loc[4, "correct"] = np.nan
    trials.loc[5, ["correct", "reward"]] = [0, 1]
    trials.to_csv(trials_path, index=False)

    pd.DataFrame({
        "Time": [trials.start_time.iloc[0], trials.choice_time.iloc[5] + 0.05],
        "Event": ["left_context", "rewA"],
        "Note": ["", "synthetic delivery"],
    }).to_csv(path.parent / "events.csv", index=False)
    metadata = json.loads(path.read_text())
    metadata["schema_version"] = "3"
    metadata["behavior"]["events"] = "events.csv"
    metadata["analysis"] = {
        "probe_roles": {"pfc": "B", "hpc": "A", "ppc": "A"},
        "ppc": {
            "prerequisite_cache": "processed/source",
            "preview_cache": "processed/preview",
            "final_cache": "processed/final",
            "preview_shuffles": 100, "final_shuffles": 1000,
        },
    }
    path.write_text(json.dumps(metadata), encoding="ascii")

    alignment_path = path.parent / "B" / "aligned.npz"
    with np.load(alignment_path, allow_pickle=False) as archive:
        spikes = archive["spike_utc_unix"]
    np.savez(alignment_path, spike_utc_unix=spikes,
             irig_utc_unix=[trials.choice_time.iloc[0] - 1.5, 1_700_000_470.0])
    return path
