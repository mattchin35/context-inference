"""Shared on-disk, two-probe synthetic session for presentation-figure tests."""

import json
from pathlib import Path

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


@pytest.fixture
def synthetic_presentation_roster(tmp_path, synthetic_presentation_session):
    """Return four metadata Paths with full switch, delivery, and speed coverage.

    The first three sessions support thirty source PCs with matched good pools.
    The fourth has a smaller pool. All activity and treadmill values are
    generated from seeded models of the prescribed behavior.
    """
    metadata = json.loads(synthetic_presentation_session.read_text())
    roster = []
    for session_number, day in enumerate((3, 4, 10, 28)):
        session_id = f"SYNTH_2026-08-{day:02d}_120000"
        root = tmp_path / session_id
        root.mkdir()
        path = root / "neural_session.json"
        metadata["session"] = session_id
        metadata["behavior"]["treadmill"] = "treadmill.txt"
        path.write_text(json.dumps(metadata), encoding="ascii")

        origin_s = 1_700_000_000.0 + session_number * 86400
        trials = _acceptance_trials(origin_s)
        trials.to_csv(root / "trials.csv", index=False)
        _write_acceptance_events(root, trials)
        rng = np.random.default_rng(759 + session_number)
        bins = np.arange(40)
        trial_in_block = np.arange(len(trials)) % 8
        speed = (6 + 3 * trial_in_block[:, None] + 3 * np.sin(bins / 4)[None, :]
                 + rng.uniform(0, 2, size=(len(trials), len(bins))))
        _write_acceptance_speed(path, trials.choice_time.to_numpy(), speed, origin_s)
        for probe_id, large_count in (("A", 64), ("B", 62)):
            n_units = large_count if session_number < 3 else 8
            _write_acceptance_spikes(root / probe_id, trials, speed, origin_s, n_units, rng)
        roster.append(path)
    return tuple(roster)


def _acceptance_trials(origin_s: float) -> pd.DataFrame:
    """Return 80 choice rows across ten blocks with UTC-second times and behavioral flags."""
    choices = []
    for block in range(10):
        context = block % 2
        # Block 1 begins with a new-side omission; block 2 begins with old-side repeats.
        first = {
            1: [(context, 0), (context, 1), (1 - context, 0)],
            2: [(1 - context, 0), (1 - context, 0), (context, 1)],
        }.get(block, [(context, 1), (context, 0), (1 - context, 0)])
        rest = [(context, 1), (1 - context, 0), (context, 0), (context, 1), (context, 1)]
        choices.extend((block, context, action, reward) for action, reward in first + rest)
    trials = pd.DataFrame(choices, columns=["cur_block", "state_int", "action", "reward"])
    trials["cur_trial"] = np.arange(1001, 1001 + len(trials))
    trials["choice_time"] = origin_s + 10 + 6 * np.arange(len(trials))
    trials["start_time"] = trials.choice_time - 1.25
    trials["reward_time"] = (trials.choice_time + 0.055).where(trials.reward.eq(1))
    trials["correct"] = trials.action.eq(trials.state_int).astype(int)
    trials["experimenter_reward_given"] = 0
    return trials


def _write_acceptance_events(root: Path, trials: pd.DataFrame) -> None:
    """Write UTC-second context entries and matching ephys-rig pump delivery events."""
    events = []
    for trial in trials.iloc[::8].itertuples():
        side = "left" if trial.state_int else "right"
        events.append((trial.start_time - 0.2, f"enter_{side}_patch"))
    for trial in trials.loc[trials.reward.eq(1)].itertuples():
        pump = 2 if trial.action else 3
        events.append((trial.reward_time, f"pump{pump} reward_amount: 3"))
    pd.DataFrame(sorted(events), columns=["Time", "Event"]).to_csv(root / "events.csv", index=False)


def _write_acceptance_spikes(
    probe_root: Path, trials: pd.DataFrame, speed: np.ndarray, origin_s: float,
    n_units: int, rng: np.random.Generator,
) -> None:
    """Write clustered UTC spikes from (choice, 40-bin) shared speed and seeded task tuning."""
    probe_root.mkdir()
    speed_signal = 0.16 * speed[:, :, None]
    epochs = np.arange(speed.shape[1])
    unit_groups = np.arange(n_units) % 6
    choice_tuning = trials.action.to_numpy()[:, None] == (unit_groups % 2)
    context_tuning = trials.state_int.to_numpy()[:, None] == (unit_groups % 2)
    reward_tuning = trials.reward.to_numpy()[:, None] == (unit_groups % 2)
    rates_per_bin = (0.45 + speed_signal + 4 * (choice_tuning & (unit_groups < 2))[:, None, :]
                     + 4 * (context_tuning & np.isin(unit_groups, [2, 3]))[:, None, :]
                     + 4 * (reward_tuning & (unit_groups > 3))[:, None, :] * (epochs >= 20)[None, :, None])
    counts = rng.poisson(rates_per_bin)
    trial_ids, bins, units = np.nonzero(counts)
    repeats = counts[trial_ids, bins, units]
    bin_starts_s = trials.choice_time.to_numpy()[trial_ids] - 2 + bins / 10
    cluster_ids = np.repeat(units, repeats)
    times_s = np.repeat(bin_starts_s, repeats) + rng.uniform(0.005, 0.095, len(cluster_ids))
    order = np.argsort(times_s, kind="stable")
    np.save(probe_root / "spike_clusters.npy", cluster_ids[order])
    np.savez(probe_root / "aligned.npz", spike_utc_unix=times_s[order],
             irig_utc_unix=[origin_s, trials.choice_time.iloc[-1] + 10])
    pd.DataFrame(dict(cluster_id=np.arange(n_units), ch=np.arange(n_units), group="good",
                      KSLabel="good")).to_csv(probe_root / "cluster_info.tsv", sep="\t", index=False)
    pd.DataFrame(dict(channel_id=[f"CH{channel}" for channel in range(n_units)], label="good",
                      inside_brain=True, x_um=0, y_um=20 * np.arange(n_units))).to_csv(
        probe_root / "channel_quality.csv", index=False)


def _write_acceptance_speed(
    metadata_path: Path, choices_s: np.ndarray, speed_mm_s: np.ndarray, origin_s: float,
) -> None:
    """Cache the (choice, 40-bin) speed magnitude as a real signed-speed archive."""
    from src.neural_analysis.session_metadata import load_session_metadata, resolve_session_metadata
    from src.neural_analysis.session_speed import TreadmillSpeed, save_speed_result, speed_configuration

    root = metadata_path.parent
    (root / "treadmill.txt").write_text("Synthetic aligned treadmill input\n", encoding="ascii")
    resolved = resolve_session_metadata(load_session_metadata(metadata_path), metadata_path)
    configuration = speed_configuration(resolved)
    speed_directory = root / "analysis_runs" / f"treadmill_speed_fixture_{configuration['configuration_id']}"
    speed_directory.mkdir(parents=True)
    bin_starts_s = choices_s[:, None] - 2 + np.arange(40) / 10
    times_s = np.r_[origin_s, bin_starts_s.ravel(), choices_s[-1] + 10]
    signed = speed_mm_s * np.where(np.arange(40) % 3 == 0, -1, 1)
    aligned = TreadmillSpeed(utc_s=times_s, speed_mm_s=np.r_[0., signed.ravel(), 0.],
                             source_rows=np.arange(len(times_s)), frame_clock_s=np.array([]),
                             frame_utc_s=np.array([]), meta=configuration)
    save_speed_result(speed_directory / "results.npz", aligned)
