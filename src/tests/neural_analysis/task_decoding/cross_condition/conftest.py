"""Independent small behavioral/control fixtures; no neural or experimental data."""

import json

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population.session_timeline import prepare_timeline_choices
from src.neural_analysis.task_decoding.config import load_task_decoding_config
from src.neural_analysis.task_decoding.cross_condition.preparation import prepare_transfer_inputs


@pytest.fixture
def transfer_session(tmp_path):
    """Return (metadata Path, base config) in a temporary version-3 session.

    JSON locators are relative paths; behavior/feature inputs contain no neural
    arrays. Region configuration uses existing task-decoding prerequisites.
    """
    root = tmp_path / "session"
    processed = root / "processed"
    processed.mkdir(parents=True)
    metadata = root / "neural_session.json"
    payload = {
        "schema_version": "3", "session": "Synthetic_2030-01-02", "acquisition": "open_ephys",
        "behavior": {"trials": "behavior/trials.csv", "events": "behavior/events.csv"},
        "probes": {
            "ProbeA": {"lfp": "ephys/a/lfp.dat", "alignment": "ephys/a/alignment.npz",
                       "sorter": "ephys/a/kilosort4", "quality": "ephys/a/quality.csv", "sites": {"PFC": 5}},
            "ProbeB": {"lfp": "ephys/b/lfp.dat", "alignment": "ephys/b/alignment.npz",
                       "sorter": "ephys/b/kilosort4", "quality": "ephys/b/quality.csv", "sites": {"HPC": 12}},
        },
        "site_pairs": [["PFC", "HPC"]],
        "analysis": {
            "probe_roles": {"pfc": "ProbeA", "hpc": "ProbeB", "ppc": "ProbeB"},
            "ppc": {"prerequisite_cache": "processed/ppc_source", "preview_cache": "processed/ppc_preview",
                    "final_cache": "processed/ppc_final", "preview_shuffles": 100, "final_shuffles": 1000},
        },
    }
    metadata.write_text(json.dumps(payload), encoding="ascii")
    (processed / "augmented_trials.csv").write_text("cur_trial\n0\n", encoding="ascii")
    (processed / "trial_feature_params.json").write_text("{}\n", encoding="ascii")
    base_payload = {
        "session_metadata_path": "neural_session.json",
        "augmented_trial_path": "processed/augmented_trials.csv",
        "trial_feature_parameter_path": "processed/trial_feature_params.json",
        "pfc_region": {"region": "PFC", "probe_id": "ProbeA", "channel_labels": ["good"],
                       "require_inside_brain": True, "cluster_groups": ["good", "mua"]},
        "hpc_region": {"region": "HPC", "probe_id": "ProbeB", "channel_labels": ["good"],
                       "require_inside_brain": True, "cluster_groups": ["good", "mua"]},
        "alignment": "start_time", "target_names": ["previous_action"], "output_root": "analysis_runs",
    }
    base_path = root / "task_decoding_config.json"
    base_path.write_text(json.dumps(base_payload), encoding="ascii")
    return metadata, load_task_decoding_config(base_path)


@pytest.fixture
def transfer_trials():
    """Return chronological J=20 rows with nondefault indices and Unix-second times.

    Rows 0..7 span correct reward/omission and incorrect unrewarded/rewarded
    choices on both sides. Rows 8..11 have missing outcomes, 12 is dark,
    13..14 conflict, 15..18 fail baseline, and 19 repeats left-correct-rewarded.
    All labels/flags are dimensionless; missing values remain explicit NaN.
    """
    state = [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 2, 1, 0, np.nan, np.nan, np.nan, np.nan, 1]
    action = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, np.nan, 0, 1, 1]
    correct = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0, np.nan, np.nan, 0, 0, 1, 1, 0, 1, 1, 1]
    reward = [1, 0, 1, 0, 0, 0, 1, 1, np.nan, np.nan, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1]
    times = 1_700_000_000.0 + 10 * np.arange(20)
    trials = pd.DataFrame({
        "cur_trial": np.arange(20), "cur_block": np.repeat(np.arange(5), 4),
        "state_int": state, "action": action, "correct": correct, "reward": reward,
        "experimenter_reward_given": [0] * 15 + [1, 0, 0, np.nan, 0],
        "choice_time": times, "start_time": times - 3,
    }, index=pd.Index(101 + 3 * np.arange(20), name="original_index"))
    trials.loc[trials.index[17], "choice_time"] = np.nan
    return trials


@pytest.fixture
def transfer_data(transfer_session, transfer_trials):
    """Return (trials(J,C), choices(N=16,C), edges(B+1,), base config).

    Choice identities retain original row positions/IDs; edges span [-2,+2]
    seconds at 500 ms. No regional score arrays are created or loaded.
    """
    _, choices = prepare_timeline_choices(transfer_trials)
    return transfer_trials, choices, np.linspace(-2, 2, 9), transfer_session[1]


@pytest.fixture
def regional_transfer_data(transfer_data):
    """Return prepared N=16 inputs and independent regional (scores, positions) tuples.

    PFC covers all N choices with float64 scores(W=16,B=8,K=3); HPC omits
    original trial IDs 0,6,19 and has independent scores(W=13,B=8,K=2).
    Maps index shared choices, not original rows. Scores retain unequal original
    PC scales/offsets; labels are dimensionless. All randomness uses seed 109.
    """
    inputs = prepare_transfer_inputs(*transfer_data)
    generator = np.random.default_rng(109)
    pfc_positions = np.arange(len(inputs.choices))
    hpc_positions = np.flatnonzero(~inputs.choices.trial_id.isin([0, 6, 19]).to_numpy())
    pfc_scores = generator.normal(size=(len(pfc_positions), 8, 3)) * [0.2, 3.0, 10.0] + [2.0, -1.0, 6.0]
    hpc_scores = generator.normal(size=(len(hpc_positions), 8, 2)) * [1.7, 0.3] + [1.0, -2.0]
    return inputs, {"PFC": (pfc_scores, pfc_positions), "HPC": (hpc_scores, hpc_positions)}
