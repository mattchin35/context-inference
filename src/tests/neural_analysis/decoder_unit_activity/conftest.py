"""Reuse seeded source fixtures; decoder fitting occurs only in synthetic test setup."""

from pathlib import Path

import pytest

from src.tests.neural_analysis.population_decoding.conftest import (
    encoding_session, population_config_path, synthetic_population,
    legacy_population_run, restore_process_main_module,
)


@pytest.fixture
def saved_decoder_run(population_config_path):
    """Return a complete synthetic source-run Path; all files stay in pytest's project-local temp tree."""
    from src.neural_analysis.population_decoding.run_session import run_session
    return Path(run_session(population_config_path)["run_directory"])
