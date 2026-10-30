"""Synthetic metadata, selection, and persistence contracts for the timeline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import (
    load_session_metadata,
    resolve_session_metadata,
)


@pytest.fixture
def synthetic_timeline_session(tmp_path):
    """Return a resolved two-probe synthetic session; all source timestamps are UTC seconds."""
    root = tmp_path / "synthetic"
    root.mkdir()
    origin = 1_700_000_000.0
    trials = pd.DataFrame(
        {
            "cur_trial": [101, 102, 103],
            "start_time": origin + np.array([0.0, 14.0, 28.0]),
            "choice_time": origin + np.array([4.123, 18.123, 32.123]),
            "action": [1, 0, 1],
            "reward": [1, 0, 0],
            "experimenter_reward_given": [0, 0, 0],
        }
    )
    trials.to_csv(root / "trials.csv", index=False)
    pd.DataFrame(
        {
            "Time": origin + np.array([0.0, 3.0, 5.0, 17.0, 20.0, 99.0]),
            "Event": [
                "trial_start",
                "LED_on",
                "LED_off",
                "left_entry",
                "right_entry",
                "correct_choice_left_patch",
            ],
            "Note": [""] * 6,
        }
    ).to_csv(root / "events.csv", index=False)
    probes = {}
    rng = np.random.default_rng(20261030)
    for probe_id in ("A", "B"):
        folder = root / probe_id
        folder.mkdir()
        rows = pd.DataFrame(
            {
                "cluster_id": [7, 8, 9, 10, 11],
                "ch": [0, 1, 2, 3, 4],
                "group": ["good", "mua", "good", "good", "noise"],
                "KSLabel": ["mua", "good", "good", "good", "good"],
            }
        )
        rows.to_csv(folder / "cluster_info.tsv", sep="\t", index=False)
        times = []
        clusters = []
        for cluster in rows.cluster_id:
            spikes = origin + np.sort(rng.uniform(0.0, 40.0, size=160))
            times.extend(spikes)
            clusters.extend([cluster] * len(spikes))
        order = np.argsort(times)
        np.save(folder / "spike_clusters.npy", np.asarray(clusters)[order])
        np.savez(
            folder / "aligned.npz",
            spike_utc_unix=np.asarray(times)[order],
            irig_utc_unix=origin + np.arange(41),
        )
        pd.DataFrame(
            {
                "channel_id": ["CH0", "CH1", "CH2", "CH3", "CH4"],
                "label": ["good", "good", "bad", "good", "good"],
                "inside_brain": [True, True, True, False, True],
                "x_um": [0.0] * 5,
                "y_um": [0.0, 20.0, 40.0, 60.0, 80.0],
            }
        ).to_csv(folder / "channel_quality.csv", index=False)
        probes[probe_id] = {
            "lfp": "",
            "alignment": f"{probe_id}/aligned.npz",
            "sorter": probe_id,
            "quality": f"{probe_id}/channel_quality.csv",
            "sites": {f"{probe_id}-site1": 0, f"{probe_id}-site2": 1},
        }
    metadata = {
        "schema_version": "2",
        "session": "synthetic-timeline",
        "acquisition": "open_ephys",
        "behavior": {"trials": "trials.csv", "events": "events.csv"},
        "probes": probes,
        "site_pairs": [],
    }
    path = root / "neural_session.json"
    path.write_text(json.dumps(metadata), encoding="ascii")
    return resolve_session_metadata(load_session_metadata(path), path)


def test_population_reuses_probe_quality_rules_not_individual_sites(
    synthetic_timeline_session,
):
    """One probe includes manual good/mua on all good inside-brain channels."""
    population = timeline_io.load_timeline_population(synthetic_timeline_session, "A")
    assert population.unit_ids == ("A:7", "A:8")
    assert population.cluster_ids.tolist() == [7, 8]
    assert population.coverage_s == (1_700_000_000.0, 1_700_000_040.0)


def test_metadata_channel_restrictions_are_intersected(synthetic_timeline_session):
    """unit_channels narrows the existing quality selection without new anatomical rules."""
    path = synthetic_timeline_session.metadata_path
    metadata = json.loads(path.read_text())
    metadata["probes"]["A"]["unit_channels"] = [1, 2, 3]
    path.write_text(json.dumps(metadata))
    session = resolve_session_metadata(load_session_metadata(path), path)
    assert timeline_io.load_timeline_population(session, "A").unit_ids == ("A:8",)


def test_behavior_does_not_synthesize_choice_from_final_raw_event(
    synthetic_timeline_session,
):
    """Only completed processed choices get IDs/windows, despite a later choice event."""
    origin, choices, events = timeline_io.load_timeline_behavior(
        synthetic_timeline_session
    )
    assert origin == 1_700_000_000.0
    assert choices.trial_id.tolist() == [101, 102, 103]
    assert events.elapsed_s.max() == 99.0


def test_optional_events_can_be_absent(synthetic_timeline_session):
    """Missing optional events leave choices available without manufacturing events."""
    synthetic_timeline_session.behavior.event_table_file.unlink()
    _, choices, events = timeline_io.load_timeline_behavior(synthetic_timeline_session)
    assert len(choices) == 3
    assert events.empty


def test_source_identity_changes_when_quality_file_changes(synthetic_timeline_session):
    """File content/mtime changes invalidate analysis reuse, while display settings are absent."""
    session = synthetic_timeline_session
    first = timeline_io.timeline_configuration(session, {"PFC": "A", "HPC": "B"}, 0.5)
    quality = session.session_root / "A/channel_quality.csv"
    quality.write_text(quality.read_text() + "\n")
    second = timeline_io.timeline_configuration(session, {"PFC": "A", "HPC": "B"}, 0.5)
    assert first != second
    assert "viewport" not in first


def test_result_round_trip_keeps_basis_units_counts_and_origin(
    synthetic_timeline_session, tmp_path
):
    """Named-array persistence reconstructs exact scientific arrays without refitting."""
    origin, choices, events = timeline_io.load_timeline_behavior(
        synthetic_timeline_session
    )
    rng = np.random.default_rng(6)
    regional = timeline.fit_timeline_activity(
        rng.poisson(3, size=(3, 8, 2)), ("A:7", "A:8"), np.arange(3), "PFC", "A", 0.5
    )
    result = timeline.SessionTimelineResult(
        "synthetic-timeline",
        origin,
        choices,
        events,
        {"PFC": regional},
        {"HPC": "missing probe"},
        {"bin_width_s": 0.5},
    )
    path = tmp_path / "results.npz"
    timeline_io.save_timeline_result(path, result)
    loaded = timeline_io.load_timeline_result(path)
    assert loaded.time_origin_s == origin
    assert loaded.unavailable == result.unavailable
    pd.testing.assert_frame_equal(loaded.choices, choices)
    np.testing.assert_array_equal(loaded.choices.reward, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(loaded.regions["PFC"].scores, regional.scores)
    np.testing.assert_array_equal(
        loaded.regions["PFC"].transform.components, regional.transform.components
    )
    np.testing.assert_array_equal(
        loaded.regions["PFC"].window_total_variance, regional.window_total_variance
    )
    assert loaded.regions["PFC"].transform.retained_unit_ids == ("A:7", "A:8")
    with np.load(path, allow_pickle=True) as archive:
        assert isinstance(archive["meta"].item(), dict)
        assert "pfc_scores" in archive.files
    with pytest.raises(FileExistsError):
        timeline_io.save_timeline_result(path, result)
