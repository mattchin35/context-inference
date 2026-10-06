"""Configuration-contract tests for task-variable decoding."""

import json
from pathlib import Path
import shutil

import pytest

from src.neural_analysis.task_decoding import config as decoding_config


CANONICAL_TARGETS = (
    "current_state",
    "current_action",
    "current_action_is_correct",
    "previous_action",
    "previous_action_was_rewarded",
    "next_action",
    "current_choice_switch_stay",
    "next_choice_switch_stay",
    "consecutive_omissions",
    "consecutive_rewards",
    "session_trial_index",
    "trial_index_in_block",
    "rewards_in_block",
    "qlearning_relative_value",
    "forgetting_q_relative_value",
    "hmm_signed_belief",
    "hmm_decay_signed_belief",
    "relative_doubt",
)


def make_config_payload() -> dict[str, object]:
    """Build the minimal portable task-decoding JSON configuration payload.

    Returns
    -------
    dict[str, object]
        JSON-serializable configuration using only documented dataclass-style
        scientific field names and the one execution-only ``output_root``.
    """
    return {
        "session_metadata_path": "neural_session.json",
        "augmented_trial_path": "processed/augmented_trials.csv",
        "trial_feature_parameter_path": "processed/trial_feature_params.json",
        "pfc_region": {
            "region": "PFC",
            "probe_id": "probe-a",
            "channel_labels": ["good"],
            "require_inside_brain": True,
            "cluster_groups": ["good", "mua"],
        },
        "hpc_region": {
            "region": "HPC",
            "probe_id": "probe-b",
            "channel_labels": ["good"],
            "require_inside_brain": True,
            "cluster_groups": ["good", "mua"],
        },
        "output_root": "analysis_runs",
    }


def write_config_file(
    tmp_path: Path,
    payload: dict[str, object] | None = None,
) -> tuple[Path, Path]:
    """Create a small session tree and write one portable JSON configuration.

    Parameters
    ----------
    tmp_path : Path
        Temporary test directory.
    payload : dict[str, object] or None
        Optional JSON payload. ``None`` uses :func:`make_config_payload`.

    Returns
    -------
    tuple[Path, Path]
        ``(config_path, session_root)``. The session root contains the three
        small declared input files required for path-resolution tests.
    """
    session_root = tmp_path / "session"
    processed_path = session_root / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)
    (session_root / "neural_session.json").write_text("{}\n", encoding="ascii")
    (processed_path / "augmented_trials.csv").write_text("cur_trial\n0\n", encoding="ascii")
    (processed_path / "trial_feature_params.json").write_text("{}\n", encoding="ascii")
    config_path = session_root / "task_decoding_config.json"
    config_path.write_text(
        json.dumps(payload if payload is not None else make_config_payload()),
        encoding="ascii",
    )
    return config_path, session_root


def load_config(tmp_path: Path, payload: dict[str, object] | None = None):
    """Load one test configuration through the public configuration entry point.

    Parameters
    ----------
    tmp_path : Path
        Temporary test directory.
    payload : dict[str, object] or None
        Optional JSON payload passed to :func:`write_config_file`.

    Returns
    -------
    tuple[object, Path, Path]
        ``(config, config_path, session_root)`` using the public loader.
    """
    config_path, session_root = write_config_file(tmp_path, payload)
    return decoding_config.load_task_decoding_config(config_path), config_path, session_root


def test_defaults_freeze_revision_five_target_order_and_fold_counts(tmp_path):
    """Defaults should expose the complete revision-5 target order and legal folds."""
    config, _, _ = load_config(tmp_path)

    assert decoding_config.TARGET_IDENTIFIERS == CANONICAL_TARGETS
    assert tuple(config.target_names) == CANONICAL_TARGETS
    assert config.alignment == "choice_time"
    assert config.bin_width_ms == 100
    assert config.pfc_pc_count == 10
    assert config.hpc_pc_count == 10
    assert config.regularization_mode == "fixed"
    assert config.outer_fold_count == 5
    assert config.inner_fold_count == 3


@pytest.mark.parametrize(
    "feature_parameter_text",
    ["[]\n", "null\n", '"text"\n', "{\n"],
)
def test_feature_parameter_file_must_decode_to_a_json_object(tmp_path, feature_parameter_text):
    """Feature parameters are provenance input and must use an object JSON document."""
    config_path, session_root = write_config_file(tmp_path)
    (session_root / "processed/trial_feature_params.json").write_text(
        feature_parameter_text,
        encoding="ascii",
    )

    with pytest.raises(ValueError, match="feature|JSON|object"):
        decoding_config.load_task_decoding_config(config_path)


def test_region_config_requires_distinct_recognized_regions_and_probes(tmp_path):
    """PFC/HPC records should reject unknown regions and shared probe identifiers."""
    payload = make_config_payload()
    payload["hpc_region"] = {**payload["hpc_region"], "probe_id": "probe-a"}
    with pytest.raises(ValueError, match="probe"):
        load_config(tmp_path, payload)

    payload = make_config_payload()
    payload["pfc_region"] = {**payload["pfc_region"], "region": "cortex"}
    with pytest.raises(ValueError, match="PFC|HPC|region"):
        load_config(tmp_path, payload)


@pytest.mark.parametrize(
    "channel_ids",
    [
        [3, 0, 7],
        [],
    ],
)
def test_region_channel_ids_accept_json_integers_and_sort_them(tmp_path, channel_ids):
    """Explicit channel IDs should retain only sorted duplicate-free JSON integers."""
    payload = make_config_payload()
    payload["pfc_region"] = {
        **payload["pfc_region"],
        "channel_ids": channel_ids,
    }

    config, _, _ = load_config(tmp_path, payload)

    assert tuple(config.pfc_region.channel_ids) == tuple(sorted(channel_ids))


@pytest.mark.parametrize(
    "channel_ids",
    [
        [1, 1],
        [-1],
        [True],
        [1.0],
        ["1"],
    ],
)
def test_region_channel_ids_reject_lossy_or_ambiguous_values(tmp_path, channel_ids):
    """Channel restrictions should fail before a downstream integer coercion can occur."""
    payload = make_config_payload()
    payload["pfc_region"] = {
        **payload["pfc_region"],
        "channel_ids": channel_ids,
    }

    with pytest.raises(ValueError, match="channel"):
        load_config(tmp_path, payload)


@pytest.mark.parametrize("bin_width_ms", [500, 100, 50, 20])
def test_supported_bin_widths_and_three_outer_folds_are_accepted(
    tmp_path,
    bin_width_ms,
):
    """Every frozen bin width and the explicit three-fold mode should load unchanged."""
    payload = make_config_payload()
    payload["bin_width_ms"] = bin_width_ms
    payload["outer_fold_count"] = 3

    config, _, _ = load_config(tmp_path, payload)

    assert config.bin_width_ms == bin_width_ms
    assert config.outer_fold_count == 3


def test_valid_trusted_bounds_are_finite_ordered_and_recorded_portably(tmp_path):
    """Manual-alignment bounds should retain configured probe IDs and UTC seconds."""
    payload = make_config_payload()
    payload["trusted_utc_bounds"] = {
        "probe-b": [200.0, 300.0],
        "probe-a": [100.0, 400.0],
    }

    config, _, _ = load_config(tmp_path, payload)
    scientific_payload = decoding_config.scientific_config_payload(config)

    assert config.trusted_utc_bounds == {
        "probe-a": (100.0, 400.0),
        "probe-b": (200.0, 300.0),
    }
    assert scientific_payload["trusted_utc_bounds"] == {
        "probe-a": [100.0, 400.0],
        "probe-b": [200.0, 300.0],
    }


EXPECTED_TARGET_METADATA = (
    ("current_state", "categorical", "Current state", "state_int"),
    ("current_action", "categorical", "Current action", "action"),
    ("current_action_is_correct", "categorical", "Current action is correct", "correct"),
    ("previous_action", "categorical", "Previous action", "action"),
    ("previous_action_was_rewarded", "categorical", "Previous action was rewarded", "reward"),
    ("next_action", "categorical", "Next action", "action"),
    ("current_choice_switch_stay", "categorical", "Current choice switch/stay", "action"),
    ("next_choice_switch_stay", "categorical", "Next choice switch/stay", "action"),
    ("consecutive_omissions", "numerical", "Consecutive omissions", "consecutive_omissions"),
    ("consecutive_rewards", "numerical", "Consecutive rewards", "consecutive_rewards"),
    ("session_trial_index", "numerical", "Session trial index", "cur_trial"),
    ("trial_index_in_block", "numerical", "Trial index in block", "cur_trial_in_block"),
    ("rewards_in_block", "numerical", "Rewards in block", "rewards_in_block"),
    ("qlearning_relative_value", "numerical", "Q-learning relative value", "Qlearning_rel_value"),
    ("forgetting_q_relative_value", "numerical", "Forgetting-Q relative value", "FQlearning_rel_value"),
    ("hmm_signed_belief", "numerical", "HMM signed belief", "HMM_rel_value_logodds"),
    ("hmm_decay_signed_belief", "numerical", "HMM-decay signed belief", "HMM_rel_value_logodds_decay"),
    ("relative_doubt", "numerical", "Relative doubt", "relative_doubt_index"),
)


def test_target_definitions_freeze_each_identifier_metadata_and_order():
    """The exported target definitions should make every scientific mapping inspectable."""
    observed = tuple(
        (
            definition.identifier,
            definition.family,
            definition.display_label,
            definition.source_column,
        )
        for definition in decoding_config.TARGET_DEFINITIONS
    )

    assert observed == EXPECTED_TARGET_METADATA


def test_target_subset_normalizes_to_canonical_order(tmp_path):
    """Equivalent target subsets should produce one stable scientific target axis."""
    payload = make_config_payload()
    payload["target_names"] = ["relative_doubt", "current_action", "rewards_in_block"]

    config, _, _ = load_config(tmp_path, payload)

    assert tuple(config.target_names) == (
        "current_action",
        "rewards_in_block",
        "relative_doubt",
    )


@pytest.mark.parametrize(
    "target_names",
    [
        [],
        ["current_action", "current_action"],
        ["Current action"],
        ["action"],
        ["CURRENT_ACTION"],
        ["unknown_target"],
    ],
)
def test_target_names_reject_aliases_duplicates_and_unknown_values(tmp_path, target_names):
    """Target JSON names should be case-sensitive canonical identifiers only."""
    payload = make_config_payload()
    payload["target_names"] = target_names

    with pytest.raises(ValueError, match="target"):
        load_config(tmp_path, payload)


@pytest.mark.parametrize("outer_fold_count", [True, False, 4, 6, 3.0, 5.5, "5"])
def test_outer_fold_count_accepts_only_integer_three_or_five(tmp_path, outer_fold_count):
    """Outer grouped-fold counts should reject coercible non-contract values."""
    payload = make_config_payload()
    payload["outer_fold_count"] = outer_fold_count

    with pytest.raises(ValueError, match="outer_fold_count"):
        load_config(tmp_path, payload)


@pytest.mark.parametrize("inner_fold_count", [True, False, 2, 4, 3.0, 3.5, "3"])
def test_inner_fold_count_accepts_only_integer_three(tmp_path, inner_fold_count):
    """Inner grouped-fold counts should reject coercible non-contract values."""
    payload = make_config_payload()
    payload["inner_fold_count"] = inner_fold_count

    with pytest.raises(ValueError, match="inner_fold_count"):
        load_config(tmp_path, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("alignment", "stimulus_time"),
        ("bin_width_ms", 75),
        ("pfc_pc_count", 0),
        ("hpc_pc_count", -1),
        ("trusted_utc_bounds", {"probe-a": [1.0, 1.0]}),
        ("trusted_utc_bounds", {"probe-a": [2.0, 1.0]}),
        ("trusted_utc_bounds", {"probe-a": [1.0, float("inf")]}),
        ("trusted_utc_bounds", {"probe-a": [True, 2.0]}),
        ("trusted_utc_bounds", {"probe-a": ["bad", 2.0]}),
        ("trusted_utc_bounds", {"unknown-probe": [1.0, 2.0]}),
        ("window_start_s", -1.0),
        ("seed", 1),
        ("coefficient_tolerance", 1e-6),
        ("tuning_grid", [{"C": 1.0}]),
    ],
)
def test_config_rejects_invalid_or_frozen_scientific_knobs(tmp_path, field, value):
    """Validation should reject unsupported values and initial JSON override knobs."""
    payload = make_config_payload()
    payload[field] = value

    with pytest.raises(ValueError, match="alignment|bin|pc|bound|window|seed|tolerance|tuning|unknown"):
        load_config(tmp_path, payload)


@pytest.mark.parametrize("field", ["pfc_pc_count", "hpc_pc_count"])
@pytest.mark.parametrize("value", [True, 1.0, "10", 0, -1])
def test_pc_counts_require_positive_json_integers(tmp_path, field, value):
    """Requested component counts should reject booleans and coercible values."""
    payload = make_config_payload()
    payload[field] = value

    with pytest.raises(ValueError, match="pc|count"):
        load_config(tmp_path, payload)


def test_config_resolves_paths_against_config_parent_and_keeps_portable_payload(tmp_path):
    """Relative JSON paths should resolve within one session while payload paths stay portable."""
    config, config_path, session_root = load_config(tmp_path)
    scientific_payload = decoding_config.scientific_config_payload(config)

    assert config.session_metadata_path == session_root / "neural_session.json"
    assert config.augmented_trial_path == session_root / "processed/augmented_trials.csv"
    assert config.trial_feature_parameter_path == session_root / "processed/trial_feature_params.json"
    assert config.output_root == session_root / "analysis_runs"
    assert config_path.parent == session_root
    assert json.dumps(scientific_payload)
    assert str(session_root) not in json.dumps(scientific_payload)
    assert "output_root" not in scientific_payload
    assert scientific_payload["analysis_version"] == decoding_config.ANALYSIS_VERSION


def test_moving_the_complete_session_tree_preserves_scientific_payload(tmp_path):
    """Portable scientific identity should not contain the execution-host root prefix."""
    config, config_path, session_root = load_config(tmp_path / "original")
    original_payload = decoding_config.scientific_config_payload(config)
    moved_root = tmp_path / "moved" / session_root.name
    shutil.copytree(session_root, moved_root)

    moved_config = decoding_config.load_task_decoding_config(
        moved_root / config_path.name
    )

    assert decoding_config.scientific_config_payload(moved_config) == original_payload


def test_metadata_path_must_name_the_canonical_session_file(tmp_path):
    """The metadata-parent session root should come from ``neural_session.json`` only."""
    payload = make_config_payload()
    payload["session_metadata_path"] = "metadata.json"
    config_path, session_root = write_config_file(tmp_path, payload)
    (session_root / "metadata.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(ValueError, match="neural_session.json|metadata"):
        decoding_config.load_task_decoding_config(config_path)


def test_config_outside_session_uses_session_default_output_root(tmp_path):
    """An external config should still default output to the metadata session root."""
    session_root = tmp_path / "session"
    processed_path = session_root / "processed"
    processed_path.mkdir(parents=True)
    (session_root / "neural_session.json").write_text("{}\n", encoding="ascii")
    (processed_path / "augmented_trials.csv").write_text("cur_trial\n0\n", encoding="ascii")
    (processed_path / "trial_feature_params.json").write_text("{}\n", encoding="ascii")
    config_directory = tmp_path / "configuration"
    config_directory.mkdir()
    payload = make_config_payload()
    payload["session_metadata_path"] = "../session/neural_session.json"
    payload["augmented_trial_path"] = "../session/processed/augmented_trials.csv"
    payload["trial_feature_parameter_path"] = (
        "../session/processed/trial_feature_params.json"
    )
    payload.pop("output_root")
    config_path = config_directory / "task_decoding_config.json"
    config_path.write_text(json.dumps(payload), encoding="ascii")

    config = decoding_config.load_task_decoding_config(config_path)

    assert config.output_root == session_root / "analysis_runs"


def test_validated_config_does_not_expose_mutable_scientific_bounds(tmp_path):
    """A frozen configuration should not permit in-place scientific mutation."""
    payload = make_config_payload()
    payload["trusted_utc_bounds"] = {"probe-a": [1.0, 2.0]}
    config, _, _ = load_config(tmp_path, payload)

    with pytest.raises(TypeError):
        config.trusted_utc_bounds["probe-a"] = (3.0, 4.0)


@pytest.mark.parametrize(
    ("field", "value", "error_match"),
    [
        ("session_metadata_path", "../outside/neural_session.json", "session|path"),
        ("augmented_trial_path", "../outside/augmented.csv", "session|path"),
        ("trial_feature_parameter_path", "../outside/params.json", "session|path"),
        ("output_root", ".", "output"),
        ("output_root", "neural_session.json", "output"),
        ("output_root", "processed", "output"),
        ("output_root", "task_decoding_config.json", "output"),
    ],
)
def test_config_rejects_paths_outside_or_overlapping_session_inputs(
    tmp_path,
    field,
    value,
    error_match,
):
    """Required inputs and the dedicated output root should not overlap or escape a session."""
    payload = make_config_payload()
    payload[field] = value

    with pytest.raises(ValueError, match=error_match):
        load_config(tmp_path, payload)


def test_scientific_payload_excludes_execution_fields_and_records_frozen_controls(tmp_path):
    """Scientific identity should exclude execution choices but retain frozen code controls."""
    config, _, _ = load_config(tmp_path)
    payload = decoding_config.scientific_config_payload(config)
    serialized = json.dumps(payload, sort_keys=True)

    expected_scientific_fields = {
        "session_metadata_path",
        "augmented_trial_path",
        "trial_feature_parameter_path",
        "pfc_region",
        "hpc_region",
        "alignment",
        "bin_width_ms",
        "pfc_pc_count",
        "hpc_pc_count",
        "target_names",
        "regularization_mode",
        "outer_fold_count",
        "inner_fold_count",
        "trusted_utc_bounds",
    }
    assert expected_scientific_fields.issubset(payload)

    for execution_key in (
        "output_root",
        "workers",
        "execution_mode",
        "scheduler_resources",
        "display_selection",
    ):
        assert execution_key not in serialized
    assert '"seed": 0' in serialized
    assert "-2.0" in serialized
    assert "2.0" in serialized
    assert "1e-08" in serialized
    assert "LogisticRegression" in serialized
    assert "ElasticNet" in serialized
    assert "PCA" in serialized

    estimator_controls = payload["frozen_controls"]["estimators"]
    assert estimator_controls["LogisticRegression"]["random_state"] == 0
    assert "penalty" not in estimator_controls["LogisticRegression"]
    assert estimator_controls["PCA"]["random_state"] == 0
