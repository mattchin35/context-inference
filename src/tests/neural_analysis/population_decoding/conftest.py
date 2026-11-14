"""Seeded source-session fixtures; all experimental-like files are temporary."""

import json
import sys
import numpy as np
import pandas as pd
import pytest

from src.tests.neural_analysis.encoding.conftest import encoding_session

_PYTEST_MAIN = sys.modules["__main__"]


@pytest.fixture(autouse=True)
def restore_process_main_module():
    """Restore collection-time pytest entry point during each test; return no data.

    Earlier unrelated AppTest modules can leak their generated __main__ before
    these tests start. Spawn workers must see pytest, not that browser script.
    Preserve the incoming module on teardown to keep this fix local to our tests.
    """
    incoming = sys.modules["__main__"]
    sys.modules["__main__"] = _PYTEST_MAIN
    yield
    sys.modules["__main__"] = incoming


@pytest.fixture
def population_config_path(encoding_session):
    """Input synthetic encoding config; return new decoder JSON Path, no fitting."""
    path = encoding_session.session_root / "population_decoding.json"
    path.write_text(json.dumps({
        "analysis_version": "population-task-decoding-v1",
        "session_metadata": "neural_session.json",
        "augmented_trials": "processed/augmented.csv",
        "probe_roles": {"PFC": "A", "HPC": "B"},
    }), encoding="ascii")
    return path


@pytest.fixture
def synthetic_population():
    """Return seeded prepared (240 trials,4 bins,8 units) Hz data with known signals.

    Unit axes carry context, choice, four post-choice outcomes and two switch
    indicators; seed=81. Twenty behavioral blocks contain mixed task values.
    No experimental sources or output files are used.
    """
    from src.neural_analysis.population.session_timeline import SessionTimelineResult, prepare_timeline_choices
    from src.neural_analysis.encoding.preparation import build_bin_rows
    from src.neural_analysis.population_decoding.preparation import PreparedPopulation, PreparedPopulationRegion
    from src.neural_analysis.population_decoding.targets import build_population_targets, assign_block_folds

    rng = np.random.default_rng(81)
    n = 240
    action, state, reward = [rng.integers(0, 2, n) for _ in range(3)]
    trials = pd.DataFrame({"cur_trial": 10 + 3*np.arange(n), "cur_block": np.repeat(np.arange(20), 12),
        "start_time": 1700000000 + 5*np.arange(n), "choice_time": 1700000002 + 5*np.arange(n),
        "action": action, "state_int": state, "reward": reward, "experimenter_reward_given": 0})
    targets = build_population_targets(state, action, reward, np.zeros(n))
    baseline = np.ones(n, bool)
    folds, _ = assign_block_folds(trials.cur_block, baseline)
    origin, choices = prepare_timeline_choices(trials)
    behavior = SessionTimelineResult("synthetic-population", origin, choices,
        pd.DataFrame(columns=["Time", "Event", "elapsed_s"]), {}, {}, {})
    rates = np.full((n, 4, 8), 20.) + rng.normal(0, 1, (n, 4, 8))
    rates[:, :, 0] += 25 * state[:, None]
    rates[:, :, 1] += 25 * action[:, None]
    for c in range(4):
        rates[:, 2:, 2+c] += 30 * (targets.values[:, 2] == c)[:, None]
    rates[:, :, 6] += 30 * (targets.values[:, 3] == 1)[:, None]
    rates[:, :, 7] += 30 * (targets.values[:, 4] == 1)[:, None]
    edges = np.linspace(-1, 1, 5)
    rows = build_bin_rows(trials, choices, np.arange(n), edges, origin, folds)
    region = PreparedPopulationRegion("PFC", "A", rates, tuple(f"A:{i}" for i in range(8)),
                                      np.arange(n), edges, rows, {})
    return PreparedPopulation("synthetic-population", trials, behavior, baseline, targets, folds,
        {"PFC": region}, {"HPC": "No probe role configured"}, {"bin_width_ms": 500, "random_seed": 81})
