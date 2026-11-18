"""Synthetic session inputs shared by preparation and saved-product tests."""

import json

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
from src.tests.neural_analysis.test_session_timeline_io import (
    synthetic_timeline_session as base_session_fixture,
)


synthetic_timeline_session = base_session_fixture


@pytest.fixture
def event_session(synthetic_timeline_session):
    """Input: resolved synthetic session. Return augmented session with (9,C) trials.

    UTC choices are >4 seconds apart. First/last windows lack full coverage;
    an ideal left-to-right x1 episode occupies original positions [1,2,3].
    Manual-label/channel conflicts and deterministic spikes come from the
    existing two-probe fixture; no experimental data or sources are accessed.
    """
    session = synthetic_timeline_session
    origin = 1_700_000_000.0
    choices = origin + np.array([-1.5, 4.123, 9.123, 14.123, 19.123, 24.123, 29.123, 34.123, 39.123])
    trials = pd.DataFrame({
        "cur_trial": np.arange(101, 110), "start_time": choices - 3,
        "choice_time": choices, "action": [1, 1, 1, 0, 0, 1, 1, 0, 1],
        "correct": [1, 1, 0, 1, 1, 0, 1, 1, 1],
        "reward": [1, 1, 0, 1, 0, 0, 1, 1, 0],
        "experimenter_reward_given": np.zeros(9, dtype=int),
        "cur_block": [0, 0, 1, 1, 1, 2, 2, 2, 2],
        "state_int": [1, 1, 0, 0, 0, 1, 1, 1, 1],
    })
    trials.to_csv(session.behavior.trial_table_file, index=False)
    metadata = json.loads(session.metadata_path.read_text())
    metadata["schema_version"] = "3"
    metadata["analysis"] = {
        "probe_roles": {"pfc": "A", "hpc": "B", "ppc": "B"},
        "ppc": {
            "prerequisite_cache": "processed/event-test-prerequisite",
            "preview_cache": "processed/event-test-preview",
            "final_cache": "processed/event-test-final",
            "preview_shuffles": 100, "final_shuffles": 1000,
        },
    }
    session.metadata_path.write_text(json.dumps(metadata))
    return resolve_session_metadata(load_session_metadata(session.metadata_path), session.metadata_path)
