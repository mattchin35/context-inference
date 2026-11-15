"""Deterministic behavior fixtures; choices use UTC seconds and 0=right/1=left."""

import json

import numpy as np
import pandas as pd
import pytest

from src.tests.neural_analysis.event_sequences.test_additional_selection import _episode


@pytest.fixture
def example_trials() -> pd.DataFrame:
    """Return full (N,C) synthetic trials with all five families and duplicate labels.

    Separate episodes use distinct blocks and a non-choice separator. Trial IDs
    are consecutive integers; choice times are UTC Unix seconds. No RNG is used.
    """
    parts = []
    for index, key in enumerate([
        ("probe", "left", 1), ("probe", "left", 1),
        ("probe", "right", 2), ("omission_probe", "left", 1),
        ("lucky_omission_switch", "left", None),
        ("indecisive_transition", "right", 1),
    ]):
        episode = _episode(key).copy()
        episode["cur_block"] += index * 100
        if key[0] in ("probe", "omission_probe"):
            episode.iloc[-1, episode.columns.get_loc("reward")] = 0
        separator = episode.iloc[[-1]].copy()
        separator["action"] = np.nan
        separator["state_int"] = 2
        parts.extend([episode, separator])
    trials = pd.concat(parts)
    trials["cur_trial"] = np.arange(501, 501 + len(trials))
    trials["choice_time"] = 1_700_000_000.0 + np.arange(len(trials)) * 5
    return trials


@pytest.fixture
def example_session(tmp_path, example_trials) -> object:
    """Return resolved metadata for a synthetic session containing no neural files.

    Source trials are (N,C), in UTC seconds. All outputs remain in pytest's
    temporary session directory; source behavior has no event-table dependency.
    """
    from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata

    root = tmp_path / "behavior-session"
    root.mkdir()
    example_trials.to_csv(root / "trials.csv", index=False)
    metadata_path = root / "neural_session.json"
    metadata_path.write_text(json.dumps({
        "schema_version": "2", "session": "synthetic-behavior", "acquisition": "open_ephys",
        "behavior": {"trials": "trials.csv"}, "probes": {}, "site_pairs": [],
    }), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
