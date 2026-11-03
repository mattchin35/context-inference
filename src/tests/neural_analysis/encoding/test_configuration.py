"""Portable configuration contracts for the PC-first context encoding analysis."""

import json
from pathlib import Path

import pytest

from src.neural_analysis.encoding.configuration import load_encoding_config


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    """Return a Path to small JSON inputs in a synthetic session; no array units.

    Input tmp_path is pytest's temporary directory. The returned scalar Path
    names a configuration beside valid v3 metadata; neural files do not exist.
    """
    session = tmp_path / "session"
    session.mkdir()
    (session / "processed").mkdir()
    metadata = {
        "schema_version": "3",
        "session": "synthetic-session",
        "acquisition": "open_ephys",
        "behavior": {"trials": "behavior/trials.csv", "events": "behavior/events.csv"},
        "probes": {
            "ProbeA": {"lfp": "a/lfp.dat", "alignment": "a/alignment.npz",
                       "sorter": "a/sorter", "quality": "a/quality.csv", "sites": {"PFC": 0}},
            "ProbeB": {"lfp": "b/lfp.dat", "alignment": "b/alignment.npz",
                       "sorter": "b/sorter", "quality": "b/quality.csv", "sites": {"HPC": 0}},
        },
        "site_pairs": [["PFC", "HPC"]],
        "analysis": {
            "probe_roles": {"pfc": "ProbeA", "hpc": "ProbeB", "ppc": "ProbeB"},
            "ppc": {"prerequisite_cache": "processed/source", "preview_cache": "processed/preview",
                    "final_cache": "processed/final", "preview_shuffles": 10, "final_shuffles": 20},
        },
    }
    (session / "neural_session.json").write_text(json.dumps(metadata), encoding="ascii")
    (session / "processed/augmented.csv").write_text("cur_trial\n10\n", encoding="ascii")
    (session / "processed/trial_feature_params.json").write_text("{}", encoding="ascii")
    path = session / "context_encoding.json"
    path.write_text(json.dumps({
        "session_metadata": "neural_session.json",
        "augmented_trials": "processed/augmented.csv",
        "feature_parameters": "processed/trial_feature_params.json",
    }), encoding="ascii")
    return path


def update_config(path: Path, updates: dict[str, object]) -> None:
    """Update scalar JSON fields at Path with a mapping; return None, no units."""
    payload = json.loads(path.read_text(encoding="ascii"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="ascii")


def test_defaults_are_the_approved_scientific_contract(config_file: Path) -> None:
    """Given a scalar config Path, assert dimensionless controls; return None."""
    config = load_encoding_config(config_file)

    assert config.bin_width_ms == 500
    assert config.windows == ("before", "after", "whole")
    assert config.timing_modes == ("choice_updated", "trial_window")
    assert config.encodings == ("signed_context", "separate_context")
    assert config.estimators == ("ridge", "ols")
    assert config.alpha == 1.0
    assert config.penalty_modes == ("same_numeric",)
    assert config.probe_roles == {"PFC": "ProbeA", "HPC": "ProbeB"}
    assert config.timeline_result is None
    assert config.alignment == "choice_time"
    assert config.fold_count == 5
    assert config.window_s == (-2.0, 2.0)


def test_paths_resolve_relative_to_config_without_loading_spikes(config_file: Path) -> None:
    """Input scalar Path resolves file/output Paths; no arrays or physical units."""
    config = load_encoding_config(config_file)
    session = config_file.parent

    assert config.session_metadata == session / "neural_session.json"
    assert config.augmented_trials == session / "processed/augmented.csv"
    assert config.feature_parameters == session / "processed/trial_feature_params.json"
    assert config.session_root == session
    assert config.intermediate_root == session / "processed/context_encoding"
    assert config.run_root == session / "analysis_runs"
    assert not config.intermediate_root.exists()
    assert not config.run_root.exists()


@pytest.mark.parametrize("width", [500, 100, 50, 20])
def test_supported_bin_widths_and_selected_settings(config_file: Path, width: int) -> None:
    """Input width is integer milliseconds and config Path; return None."""
    update_config(config_file, {"bin_width_ms": width, "windows": ["after"],
                  "timing_modes": ["trial_window"], "encodings": ["separate_context"],
                  "estimators": ["ols"], "alpha": 2.5,
                  "penalty_modes": ["same_numeric", "matched_context"]})
    config = load_encoding_config(config_file)

    assert config.bin_width_ms == width
    assert config.windows == ("after",)
    assert config.timing_modes == ("trial_window",)
    assert config.encodings == ("separate_context",)
    assert config.estimators == ("ols",)
    assert config.alpha == 2.5
    assert config.penalty_modes == ("same_numeric", "matched_context")


@pytest.mark.parametrize("field,value", [
    ("bin_width_ms", 200), ("bin_width_ms", True), ("bin_width_ms", "500"),
    ("bin_width_ms", 500.0), ("alpha", 0), ("alpha", -1), ("alpha", True),
    ("alpha", float("nan")), ("alpha", float("inf")), ("alpha", "1"),
    ("windows", []), ("windows", ["before", "before"]), ("windows", "whole"),
    ("windows", ["trial_total"]), ("timing_modes", ["reward_updated"]),
    ("encodings", ["current_action"]), ("estimators", ["poisson"]),
    ("penalty_modes", ["automatic"]), ("penalty_modes", []),
    ("fold_count", 3), ("alignment", "start_time"), ("pc_count", 10),
    ("predictors", ["FQlearning_rel_value"]), ("output_root", "elsewhere"),
    ("unknown", 1),
])
def test_invalid_or_unapproved_controls_fail_before_neural_loading(
    config_file: Path, field: str, value: object,
) -> None:
    """Input Path/string/scalar-or-list field values; assert ValueError, return None."""
    update_config(config_file, {field: value})
    with pytest.raises(ValueError, match=field):
        load_encoding_config(config_file)


@pytest.mark.parametrize("field", ["session_metadata", "augmented_trials", "feature_parameters"])
def test_required_paths_must_name_existing_files(config_file: Path, field: str) -> None:
    """Input config Path and field name; missing scalar Path raises, return None."""
    update_config(config_file, {field: "absent-file"})
    with pytest.raises(ValueError, match=field):
        load_encoding_config(config_file)


def test_optional_timeline_path_is_resolved(config_file: Path) -> None:
    """Input config Path resolves a saved NPZ Path without opening it; return None."""
    saved = config_file.parent / "processed/timeline.npz"
    saved.write_bytes(b"not read during configuration")
    update_config(config_file, {"timeline_result": "processed/timeline.npz"})
    assert load_encoding_config(config_file).timeline_result == saved


@pytest.mark.parametrize("text", ["[]", "null", '"text"', "{"])
def test_feature_parameters_must_be_a_json_object(config_file: Path, text: str) -> None:
    """Input config Path and JSON text; require provenance object, return None."""
    (config_file.parent / "processed/trial_feature_params.json").write_text(text, encoding="ascii")
    with pytest.raises(ValueError, match="feature_parameters"):
        load_encoding_config(config_file)


def test_v2_metadata_requires_explicit_roles(config_file: Path) -> None:
    """Input config Path; no inferred anatomy from v2 probe sites, return None."""
    path = config_file.parent / "neural_session.json"
    metadata = json.loads(path.read_text(encoding="ascii"))
    metadata["schema_version"] = "2"
    del metadata["analysis"]
    metadata["cache"] = "processed/cache"
    path.write_text(json.dumps(metadata), encoding="ascii")
    with pytest.raises(ValueError, match="probe_roles"):
        load_encoding_config(config_file)

    update_config(config_file, {"probe_roles": {"PFC": "ProbeA", "HPC": "ProbeB"}})
    assert load_encoding_config(config_file).probe_roles == {"PFC": "ProbeA", "HPC": "ProbeB"}


@pytest.mark.parametrize("roles", [
    {}, {"PFC": "ProbeA", "HPC": "ProbeA"},
    {"PFC": "missing", "HPC": "ProbeB"}, {"pfc": "ProbeA", "hpc": "ProbeB"}, [],
])
def test_probe_roles_are_explicit_distinct_metadata_probes(config_file: Path, roles: object) -> None:
    """Input Path and role mapping/object; malformed anatomy fails, return None."""
    update_config(config_file, {"probe_roles": roles})
    with pytest.raises(ValueError, match="probe_roles"):
        load_encoding_config(config_file)


def test_explicit_partial_roles_allow_independent_regional_analysis(config_file: Path) -> None:
    """Input config Path; one region mapping is valid and unitless, return None."""
    update_config(config_file, {"probe_roles": {"PFC": "ProbeA"}})
    assert load_encoding_config(config_file).probe_roles == {"PFC": "ProbeA"}


def test_configuration_does_not_expose_mutable_default_sequences(config_file: Path) -> None:
    """Input config Path; frozen controls resist accidental reassignment, return None."""
    config = load_encoding_config(config_file)
    with pytest.raises((AttributeError, TypeError)):
        config.alpha = 9.0
