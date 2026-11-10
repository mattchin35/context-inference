"""Small controls and fixed transfer registry contracts."""

import json

import pytest

from src.neural_analysis.task_decoding.cross_condition import config as transfer_config
from src.neural_analysis.task_decoding.cross_condition.config import (
    TARGET_NAMES, TRANSFERS, load_cross_condition_config,
)


def test_default_controls_use_version_three_roles_and_existing_base(transfer_session):
    """Metadata Path alone returns contained Paths, 500 ms and PFC/HPC role mapping."""
    metadata, base = transfer_session
    controls = load_cross_condition_config(metadata)
    assert controls.session_metadata_path == metadata.resolve()
    assert controls.base_config_path == metadata.parent / "task_decoding_config.json"
    assert controls.bin_width_ms == 500
    assert controls.probe_roles == {"PFC": "ProbeA", "HPC": "ProbeB"}
    assert base.target_names == ("previous_action",)


@pytest.mark.parametrize("width", [500, 100, 50, 20])
def test_contained_controls_and_explicit_width_override(transfer_session, width):
    """Contained config Path resolves base relative to its parent; integer ms override wins."""
    metadata, _ = transfer_session
    directory = metadata.parent / "controls"
    directory.mkdir()
    controls_path = directory / "custom.json"
    controls_path.write_text(json.dumps({"base_config_path": "../task_decoding_config.json", "bin_width_ms": 100}), encoding="ascii")
    controls = load_cross_condition_config(metadata, config_path=controls_path, bin_width_ms=width)
    assert controls.base_config_path == metadata.parent / "task_decoding_config.json"
    assert controls.bin_width_ms == width
    default_path = metadata.parent / "cross_condition_config.json"
    default_path.write_text(json.dumps({"bin_width_ms": width}), encoding="ascii")
    assert load_cross_condition_config(metadata).bin_width_ms == width


@pytest.mark.parametrize("width", [True, 100.0, "100", 0, 200, None])
def test_invalid_file_widths_raise(transfer_session, width):
    """JSON bin width must be a supported exact integer in milliseconds."""
    metadata, _ = transfer_session
    path = metadata.parent / "cross_condition_config.json"
    path.write_text(json.dumps({"bin_width_ms": width}), encoding="ascii")
    with pytest.raises(ValueError, match="bin_width"):
        load_cross_condition_config(metadata)


@pytest.mark.parametrize("width", [True, 100.0, "100", 0, 200])
def test_invalid_explicit_widths_raise(transfer_session, width):
    """Explicit width arguments obey the same exact-integer millisecond contract."""
    with pytest.raises(ValueError, match="bin_width"):
        load_cross_condition_config(transfer_session[0], bin_width_ms=width)


@pytest.mark.parametrize("payload", [{"training_fraction": 0.8}, [], {"base_config_path": ""}])
def test_malformed_or_unknown_controls_raise(transfer_session, payload):
    """Only base_config_path and bin_width_ms belong in the small controls object."""
    metadata, _ = transfer_session
    path = metadata.parent / "cross_condition_config.json"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError):
        load_cross_condition_config(metadata)


def test_invalid_json_and_missing_explicit_controls_raise(transfer_session):
    """Malformed existing controls fail; an absent explicit path is not a default."""
    metadata, _ = transfer_session
    with pytest.raises(FileNotFoundError):
        load_cross_condition_config(metadata, config_path=metadata.parent / "absent.json")
    path = metadata.parent / "cross_condition_config.json"
    path.write_text("{", encoding="ascii")
    with pytest.raises(ValueError):
        load_cross_condition_config(metadata)


@pytest.mark.parametrize("locator", ["../outside.json", "/tmp/outside.json"])
def test_base_locator_cannot_escape_session(transfer_session, locator):
    """String relative/absolute locators outside the selected session are rejected."""
    metadata, _ = transfer_session
    path = metadata.parent / "cross_condition_config.json"
    path.write_text(json.dumps({"base_config_path": locator}), encoding="ascii")
    with pytest.raises(ValueError):
        load_cross_condition_config(metadata)


def test_controls_path_and_base_metadata_must_belong_to_selected_session(transfer_session, tmp_path):
    """Reject outside controls and a contained base that points at another metadata file."""
    metadata, _ = transfer_session
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="ascii")
    with pytest.raises(ValueError):
        load_cross_condition_config(metadata, config_path=outside)
    other = metadata.parent / "other"
    other.mkdir()
    (other / "neural_session.json").write_text(metadata.read_text(), encoding="ascii")
    base_path = metadata.parent / "task_decoding_config.json"
    payload = json.loads(base_path.read_text())
    payload["session_metadata_path"] = "other/neural_session.json"
    base_path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="metadata|session"):
        load_cross_condition_config(metadata)


@pytest.mark.parametrize("version", ["1", "2", "4"])
def test_only_version_three_metadata_is_accepted(transfer_session, version):
    """Other metadata schema strings fail rather than guessing regional analysis roles."""
    metadata, _ = transfer_session
    payload = json.loads(metadata.read_text())
    payload["schema_version"] = version
    if version == "2":
        del payload["analysis"]
    metadata.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="version|schema|3"):
        load_cross_condition_config(metadata)


def test_version_three_requires_valid_roles(transfer_session):
    """Missing role metadata must not fall back to base-config probe guessing."""
    metadata, _ = transfer_session
    payload = json.loads(metadata.read_text())
    del payload["analysis"]
    metadata.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError):
        load_cross_condition_config(metadata)


def test_fixed_registry_preserves_original_targets_and_appends_two_context_models():
    """Preserve T=4 and the first sixteen target assignments; append only two class-ordered Context models."""
    assert TARGET_NAMES == ("current_state", "current_action", "current_reward", "current_action_is_correct")
    assert len(TRANSFERS) == len({item.split_id for item in TRANSFERS}) == 18
    assert tuple(item.target_name for item in TRANSFERS[:16]) == tuple(name for name in TARGET_NAMES for _ in range(4))
    assert tuple((item.split_id, item.target_name) for item in TRANSFERS[16:]) == (
        ("context_correct_rewarded_to_other_outcomes", "current_state"),
        ("context_omission_to_other_outcomes", "current_state"),
    )
    for item in TRANSFERS:
        assert len(item.source_cells) == len(item.test_cells) == 2
        assert len(set(item.source_cells + item.test_cells)) == 4


def test_context_outcome_mapping_reuses_existing_incorrect_model_once():
    """Three ordered outcome models reference eighteen unique fits; incorrect-trained remains the original split."""
    expected = {
        "context_correct_rewarded_to_other_outcomes": ("correct_rewarded", ("omission", "incorrect")),
        "context_omission_to_other_outcomes": ("omission", ("correct_rewarded", "incorrect")),
        "context_incorrect_to_correct": ("incorrect", ("correct_rewarded", "omission")),
    }
    mapping = transfer_config.CONTEXT_OUTCOME_MODELS
    assert tuple(mapping) == tuple(expected)
    assert dict(mapping) == expected
    by_id = {item.split_id: item for item in TRANSFERS}
    assert all(by_id[split_id].target_name == "current_state" for split_id in mapping)
    assert sum(item.split_id == "context_incorrect_to_correct" for item in TRANSFERS) == 1
    assert TRANSFERS[3].split_id == "context_incorrect_to_correct"
    assert transfer_config.CONTEXT_CONDITION_SPLIT_IDS == (
        "context_correct_to_incorrect", "context_left_choice_to_right_choice",
        "context_right_choice_to_left_choice", "context_incorrect_to_correct",
    )
    assert set(mapping) & set(transfer_config.CONTEXT_CONDITION_SPLIT_IDS) == {"context_incorrect_to_correct"}


def test_expanded_registry_requires_v2_analysis_and_schema_two():
    """The expanded S=18 bundle has its own version; metadata input remains independently version 3."""
    assert transfer_config.ANALYSIS_VERSION == "cross-condition-generalization-v2"
    assert transfer_config.SCHEMA_VERSION == 2
