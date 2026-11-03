"""Seeded chronological behavior and independent regional coverage fixtures."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.population.session_timeline import (
    SessionTimelineResult, fit_timeline_activity, prepare_timeline_choices,
)
from src.tests.neural_analysis.task_decoding.test_targets import make_config


@pytest.fixture
def prediction_inputs(tmp_path):
    """Return (base config, trials(N=18,C), timeline) with UTC times and 6 blocks.

    Trial 1 is manual, trial 2 is a nonchoice. Regional (W,B,K) scores have
    independent window coverage, B=8, K=3; all randomness uses seed 72.
    """
    config = make_config(tmp_path)
    rows = np.arange(18)
    actions = (rows // 3) % 2
    choice_times = 1_700_000_000.0 + rows * 10
    trials = pd.DataFrame({
        "cur_trial": rows, "cur_block": rows // 3,
        "start_time": choice_times - 3, "choice_time": choice_times,
        "action": actions.astype(float), "state_int": actions,
        "correct": np.ones(18), "reward": np.where(rows % 4 == 0, 0, 1),
        "experimenter_reward_given": np.zeros(18),
        "Qlearning_rel_value": np.linspace(-1, 1, 18),
        "FQlearning_rel_value": np.linspace(-2, 2, 18),
        "HMM_rel_value_logodds": np.linspace(-3, 3, 18),
        "HMM_rel_value_logodds_decay": np.linspace(-4, 4, 18),
        "relative_doubt_index": np.linspace(0, 1, 18),
    }, index=rows + 100)
    trials.loc[101, "experimenter_reward_given"] = 1
    trials.loc[102, "action"] = np.nan
    origin, choices = prepare_timeline_choices(trials)
    generator = np.random.default_rng(72)
    regions = {}
    for role, positions in (("PFC", np.arange(len(choices) - 1)),
                            ("HPC", np.arange(2, len(choices)))):
        counts = generator.poisson(3, size=(len(positions), 8, 3))
        regions[role] = fit_timeline_activity(
            counts, (role + ":1", role + ":2", role + ":3"),
            positions, role, role, 0.5,
        )
    timeline = SessionTimelineResult(
        "synthetic", origin, choices,
        pd.DataFrame(columns=["Time", "Event", "Note", "elapsed_s"]), regions, {},
        {"configuration_id": "fixture", "bin_width_s": 0.5},
    )
    return replace(config, target_names=("current_action",)), trials, timeline
