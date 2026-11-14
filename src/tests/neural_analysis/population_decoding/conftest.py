"""Seeded source-session fixtures; all experimental-like files are temporary."""

import json
import pytest

from src.tests.neural_analysis.encoding.conftest import (
    encoding_session,
    restore_process_main_module,
)


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
