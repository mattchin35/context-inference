"""Saved CCA records cannot silently change session labels or event clocks."""

from dataclasses import replace

import pytest

from src.neural_analysis.population.session_cca_io import save_cca_result
from src.tests.neural_analysis.session_cca.test_results import (
    cca_result as cca_result,
    synthetic_timeline_session as synthetic_timeline_session,
)


@pytest.mark.parametrize("field", ["session_id", "probe_roles"])
def test_record_identity_must_match_its_scientific_configuration(cca_result, tmp_path, field):
    """A conflicting record label cannot retain the original configuration ID.

    cca_result is a valid paired record; field names a unitless identity field.
    tmp_path is pytest scratch storage. Return None after asserting rejection.
    """
    value = "another-session" if field == "session_id" else {"PFC": "B", "HPC": "A"}
    inconsistent = replace(cca_result, **{field: value})
    path = tmp_path / "results.npz"

    with pytest.raises(ValueError):
        save_cca_result(path, inconsistent)
    assert not path.exists()


@pytest.mark.parametrize("problem", ["missing_elapsed", "shifted_elapsed"])
def test_nonempty_events_retain_the_shared_utc_and_elapsed_clock(cca_result, tmp_path, problem):
    """Nonempty source-event tables need consistent UTC/elapsed-second columns.

    problem selects a missing column or a 17-second clock error in fixture data.
    Original event labels/IDs gain no new eligibility filter. Return None.
    """
    events = cca_result.events.copy()
    if problem == "missing_elapsed":
        events = events.drop(columns="elapsed_s")
    else:
        events["elapsed_s"] += 17.0
    path = tmp_path / "results.npz"

    with pytest.raises(ValueError):
        save_cca_result(path, replace(cca_result, events=events))
    assert not path.exists()
