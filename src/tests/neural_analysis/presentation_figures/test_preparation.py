"""Acceptance tests for live-metadata inputs, shared windows and block folds."""

import numpy as np

from src.neural_analysis.population import session_timeline_io
from src.neural_analysis.presentation_figures.preparation import (
    make_block_splits,
    preflight,
    prepare_choices,
)


def test_metadata_entry_loads_good_units_once_for_all_joint_windows(
    synthetic_presentation_session, monkeypatch,
):
    """Metadata roles drive paired counts (choice, bin, unit) at all widths."""
    calls = []
    original = session_timeline_io.load_timeline_population

    def record_load(session, probe, **kwargs):
        """Record scalar probe/policy arguments and return the real population."""
        calls.append((probe, kwargs))
        return original(session, probe, **kwargs)

    monkeypatch.setattr(session_timeline_io, "load_timeline_population", record_load)
    prepared = prepare_choices(synthetic_presentation_session)

    policy = {"population_quality": "good", "missing_label_policy": "exclude"}
    assert sorted(calls) == [("A", policy), ("B", policy)]
    assert prepared.probe_roles == {"PFC": "B", "HPC": "A"}
    assert prepared.coverage_s["PFC"] == (1_700_000_008.5, 1_700_000_470.0)
    assert prepared.coverage_s["HPC"] == (1_700_000_000.0, 1_700_000_470.0)
    assert set(prepared.windows) == {((-1., 1.), 100), ((-1., 1.), 500), ((-2., 2.), 100)}
    for (bounds, width_ms), window in prepared.windows.items():
        expected_rows = np.delete(np.arange(45), [1] if bounds[0] == -1 else [0, 1])
        np.testing.assert_array_equal(window.row_positions, expected_rows)
        n_bins = int((bounds[1] - bounds[0]) * 1000 / width_ms)
        np.testing.assert_allclose(window.bin_edges_s, np.linspace(*bounds, n_bins + 1))
        for region, probe in prepared.probe_roles.items():
            assert prepared.unit_ids[region] == tuple(f"{probe}:{unit}" for unit in range(8))
            assert window.counts[region].shape == (len(expected_rows), n_bins, 8)
            np.testing.assert_allclose(window.counts[region][1].sum(axis=0),
                                       [0, 0, 1, 1] * 2 if n_bins < 40 else np.ones(8))
        assert len(window.splits) == 5
        for split in window.splits:
            assert np.all(prepared.fold_ids[expected_rows[split.test_indices]] == split.fold_id)
            assert not np.any(prepared.fold_ids[expected_rows[split.train_indices]] == split.fold_id)


def test_full_chronology_and_independent_label_masks(synthetic_presentation_session):
    """Unknown context/reward/correctness leaves other recorded targets usable."""
    prepared = prepare_choices(synthetic_presentation_session)
    np.testing.assert_array_equal(prepared.trials.cur_trial, np.arange(100, 145))
    np.testing.assert_array_equal(prepared.trials.row_position, np.arange(45))
    np.testing.assert_array_equal(prepared.trials.cur_block, 20 + np.arange(45) // 5)
    np.testing.assert_array_equal(prepared.choices.row_position, np.delete(np.arange(45), 1))
    assert prepared.events.Event.tolist() == ["left_context", "rewA"]
    assert prepared.events.Note.iloc[1] == "synthetic delivery"
    np.testing.assert_allclose(prepared.events.Time - prepared.time_origin_s, prepared.events.elapsed_s)
    assert not prepared.baseline[1]
    assert prepared.baseline[2:5].all()
    assert not prepared.targets.valid[2, 0]
    assert prepared.targets.valid[2, 1]
    assert not prepared.targets.valid[3, 2]
    assert prepared.condition_masks["omission"][0]
    assert prepared.condition_masks["overall"][4]
    assert not prepared.condition_masks["correct_rewarded"][4]
    assert prepared.condition_masks["incorrect"][5]
    assert prepared.meta["trial_audit"]["manual_rewards"] == 1
    assert prepared.meta["trial_audit"]["incorrect_rewarded"] >= 1
    assert prepared.meta["trial_audit"]["unknown_context"] == 1
    assert prepared.meta["trial_audit"]["unknown_reward"] == 1
    for block in prepared.trials.cur_block.unique():
        assert len(np.unique(prepared.fold_ids[prepared.trials.cur_block == block])) == 1


def test_preflight_is_read_only_and_checks_actual_spike_files(
    synthetic_presentation_session, monkeypatch,
):
    """Source inspection reads no raw spike arrays and reports missing sorter inputs."""
    def forbidden(*args, **kwargs):
        """Reject any arguments if called because no spike read is expected."""
        raise AssertionError("preflight attempted raw spike loading")

    monkeypatch.setattr(session_timeline_io, "load_timeline_population", forbidden)
    monkeypatch.setattr(session_timeline_io.loading, "load_aligned_spikes", forbidden)
    monkeypatch.setattr(session_timeline_io.loading, "load_sorter_metadata", forbidden)
    root = synthetic_presentation_session.parent
    before = {path: path.stat().st_mtime_ns for path in root.rglob("*")}
    inspected = preflight(synthetic_presentation_session)
    assert inspected.probe_roles == {"PFC": "B", "HPC": "A"}
    assert not inspected.unavailable
    assert {path: path.stat().st_mtime_ns for path in root.rglob("*")} == before

    (root / "A" / "spike_clusters.npy").unlink()
    inspected = preflight(synthetic_presentation_session)
    assert "spike_clusters" in inspected.unavailable["HPC"]


def test_folds_purge_strictly_overlapping_whole_choices():
    """Purging leaves touching half-open windows and returns five fold records."""
    times = np.array([0., 1., 2., 4., 9., 14., 19.])
    folds = np.array([0, 1, 2, 1, 3, 4, -1])
    splits = make_block_splits(times, folds, window_s=(-1., 1.))
    first = splits[0]
    assert len(splits) == 5
    np.testing.assert_array_equal(first.test_indices, [0])
    np.testing.assert_array_equal(first.purged_indices, [1])
    np.testing.assert_array_equal(first.train_indices, [2, 3, 4, 5])
    np.testing.assert_array_equal(splits[1].test_indices, [1, 3])
