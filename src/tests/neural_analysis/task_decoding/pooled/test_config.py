"""Configuration tests for pooled-window task-variable decoding."""

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from src.neural_analysis.task_decoding.pooled import config as pooled_config


CANONICAL_WINDOWS = ("pre_choice", "post_choice", "full")


def make_base_config_payload() -> dict[str, object]:
    """Return a minimal valid base task-decoding configuration mapping.

    Returns
    -------
    dict[str, object]
        JSON-safe base configuration using session-relative input paths.
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
        "target_names": ["current_action", "session_trial_index"],
        "condition_names": ["stay", "all"],
        "output_root": "analysis_runs",
    }


def write_configuration_tree(
    tmp_path: Path,
    *,
    base_payload: dict[str, object] | None = None,
    pooled_payload: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    """Write a portable base configuration and pooled overlay.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty test directory in which one session tree is created.
    base_payload, pooled_payload : dict[str, object] or None
        Optional JSON mappings. Defaults create valid fixed-mode choice-aligned
        configurations.

    Returns
    -------
    tuple[pathlib.Path, pathlib.Path, pathlib.Path]
        Pooled overlay path, base configuration path, and session root.
    """
    session_root = tmp_path / "session"
    processed = session_root / "processed"
    processed.mkdir(parents=True)
    (session_root / "neural_session.json").write_text("{}\n", encoding="ascii")
    (processed / "augmented_trials.csv").write_text("cur_trial\n0\n", encoding="ascii")
    (processed / "trial_feature_params.json").write_text("{}\n", encoding="ascii")

    base_path = session_root / "task_decoding_config.json"
    base_path.write_text(
        json.dumps(base_payload if base_payload is not None else make_base_config_payload()),
        encoding="ascii",
    )
    overlay = (
        {"base_config_path": base_path.name}
        if pooled_payload is None
        else pooled_payload
    )
    pooled_path = session_root / "pooled_task_decoding_config.json"
    pooled_path.write_text(json.dumps(overlay), encoding="ascii")
    return pooled_path, base_path, session_root


def canonical_edges(bin_width_ms: int = 100) -> np.ndarray:
    """Return the canonical choice-relative edge axis in seconds.

    Parameters
    ----------
    bin_width_ms : int, default=100
        Supported base-task bin width in milliseconds.

    Returns
    -------
    numpy.ndarray
        Float64 edge vector from -2 through +2 seconds, inclusive.
    """
    bin_count = 4000 // bin_width_ms
    return np.linspace(-2.0, 2.0, bin_count + 1, dtype=np.float64)


def test_overlay_defaults_and_canonicalizes_windows_and_conditions(tmp_path):
    """Defaults and unordered subsets should resolve to canonical coordinates."""
    pooled_path, base_path, _session_root = write_configuration_tree(tmp_path / "default")
    default = pooled_config.load_pooled_decoding_config(pooled_path)
    default_windows = pooled_config.resolve_training_windows(default, canonical_edges())

    assert default.base_config_path == base_path.resolve()
    assert default.training_window_names == ("pre_choice",)
    assert default.condition_names == ("all", "correct_rewarded", "stay")
    assert tuple(window.name for window in default_windows) == ("pre_choice",)
    assert (default_windows[0].start_s, default_windows[0].stop_s) == (-0.5, 0.0)
    np.testing.assert_array_equal(default_windows[0].bin_indices, np.arange(15, 20))

    pooled_path, _base_path, _session_root = write_configuration_tree(
        tmp_path / "ordered",
        pooled_payload={
            "base_config_path": "task_decoding_config.json",
            "training_window_names": ["full", "post_choice", "pre_choice"],
        },
    )
    ordered = pooled_config.load_pooled_decoding_config(pooled_path)

    assert ordered.training_window_names == CANONICAL_WINDOWS


@pytest.mark.parametrize(
    "pooled_payload",
    [
        {},
        {"base_config_path": "task_decoding_config.json", "unexpected": True},
        {"base_config_path": "task_decoding_config.json", "training_window_names": []},
        {
            "base_config_path": "task_decoding_config.json",
            "training_window_names": ["pre_choice", "pre_choice"],
        },
        {
            "base_config_path": "task_decoding_config.json",
            "training_window_names": ["choice"],
        },
    ],
)
def test_overlay_rejects_missing_unknown_empty_duplicate_or_unknown_fields(
    tmp_path,
    pooled_payload,
):
    """The overlay should expose exactly the two reviewed configuration fields."""
    pooled_path, _base_path, _session_root = write_configuration_tree(
        tmp_path,
        pooled_payload=pooled_payload,
    )

    with pytest.raises(ValueError, match="base_config_path|field|window|duplicate|unknown"):
        pooled_config.load_pooled_decoding_config(pooled_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("alignment", "start_time"), ("regularization_mode", "tuned")],
)
def test_overlay_rejects_unsupported_base_modes(tmp_path, field, value):
    """Pooled revision 1 should accept only choice-aligned fixed-mode bases."""
    base_payload = make_base_config_payload()
    base_payload[field] = value
    pooled_path, _base_path, _session_root = write_configuration_tree(
        tmp_path,
        base_payload=base_payload,
    )

    with pytest.raises(ValueError, match="choice_time|fixed|alignment|regularization"):
        pooled_config.load_pooled_decoding_config(pooled_path)


def test_window_resolution_rejects_malformed_or_misaligned_edges(tmp_path):
    """Window bounds must coincide with a valid ascending edge axis."""
    pooled_path, _base_path, _session_root = write_configuration_tree(tmp_path)
    config = pooled_config.load_pooled_decoding_config(pooled_path)

    with pytest.raises(ValueError, match="edge|boundary|align"):
        pooled_config.resolve_training_windows(config, canonical_edges() + 0.01)
    with pytest.raises(ValueError, match="edge|ascending|finite"):
        pooled_config.resolve_training_windows(config, np.array([-2.0, -1.0, -1.0, 2.0]))


def test_scientific_payload_is_path_and_format_independent(tmp_path):
    """Only resolved scientific content should determine pooled identity."""
    first_overlay, _first_base, first_root = write_configuration_tree(
        tmp_path / "first",
        pooled_payload={
            "base_config_path": "task_decoding_config.json",
            "training_window_names": ["full", "pre_choice"],
        },
    )
    second_root = tmp_path / "second" / "moved_session"
    shutil.copytree(first_root, second_root)
    second_overlay = second_root / "renamed_overlay.json"
    second_overlay.write_text(
        '{\n  "training_window_names": ["full", "pre_choice"],\n'
        '  "base_config_path": "task_decoding_config.json"\n}\n',
        encoding="ascii",
    )

    first_config = pooled_config.load_pooled_decoding_config(first_overlay)
    second_config = pooled_config.load_pooled_decoding_config(second_overlay)
    first_windows = pooled_config.resolve_training_windows(first_config, canonical_edges())
    second_windows = pooled_config.resolve_training_windows(second_config, canonical_edges())
    first_payload = pooled_config.scientific_pooled_config_payload(
        first_config,
        first_windows,
    )
    second_payload = pooled_config.scientific_pooled_config_payload(
        second_config,
        second_windows,
    )

    assert first_payload == second_payload
    assert first_payload["analysis_version"] == "pooled-task-variable-decoding-v1"
    assert first_payload["reference_condition"] == "correct_rewarded"
    assert first_payload["condition_names"] == ["all", "correct_rewarded", "stay"]
    assert [item["name"] for item in first_payload["training_windows"]] == [
        "pre_choice",
        "full",
    ]
    assert "base_config_path" not in first_payload
    assert str(first_root) not in json.dumps(first_payload)


def write_prepared_run(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Create the small immutable sidecars consumed by the prepared loader.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory containing the source configuration tree.

    Returns
    -------
    tuple[pathlib.Path, dict[str, object]]
        Prepared run directory and canonical pooled scientific payload.
    """
    pooled_path, base_path, session_root = write_configuration_tree(
        tmp_path,
        pooled_payload={
            "base_config_path": "task_decoding_config.json",
            "training_window_names": ["post_choice", "pre_choice"],
        },
    )
    config = pooled_config.load_pooled_decoding_config(pooled_path)
    windows = pooled_config.resolve_training_windows(config, canonical_edges())
    payload = pooled_config.scientific_pooled_config_payload(config, windows)
    run_directory = session_root / "analysis_runs" / "pooled_run"
    run_directory.mkdir(parents=True)
    (run_directory / "config.json").write_text(json.dumps(payload), encoding="ascii")
    (run_directory / "execution.json").write_text(
        json.dumps({"session_root": str(session_root.resolve())}),
        encoding="ascii",
    )
    (run_directory / "pooled_config.input.json").write_bytes(pooled_path.read_bytes())
    (run_directory / "base_config.input.json").write_bytes(base_path.read_bytes())
    return run_directory, payload


def test_prepared_loader_reconstructs_without_parsing_audit_copies(tmp_path):
    """Resume configuration should come only from canonical and execution sidecars."""
    run_directory, expected_payload = write_prepared_run(tmp_path)
    (run_directory / "pooled_config.input.json").write_text("not JSON", encoding="ascii")
    (run_directory / "base_config.input.json").write_text("also not JSON", encoding="ascii")

    config, windows = pooled_config.load_prepared_pooled_decoding_config(run_directory)

    assert config.base_config.output_root == run_directory.parent.resolve()
    assert config.base_config_path == (run_directory / "base_config.input.json").resolve()
    assert config.base_config.session_root == (tmp_path / "session").resolve()
    assert pooled_config.scientific_pooled_config_payload(config, windows) == expected_payload


def test_prepared_loader_rejects_missing_root_path_escape_and_payload_drift(tmp_path):
    """Prepared reconstruction should fail before any neural-array loading."""
    run_directory, original_payload = write_prepared_run(tmp_path)

    (run_directory / "execution.json").write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="session_root"):
        pooled_config.load_prepared_pooled_decoding_config(run_directory)

    (run_directory / "execution.json").write_text(
        json.dumps({"session_root": str((tmp_path / "session").resolve())}),
        encoding="ascii",
    )
    escaped_payload = json.loads(json.dumps(original_payload))
    escaped_payload["base_config"]["augmented_trial_path"] = "../outside.csv"
    (run_directory / "config.json").write_text(json.dumps(escaped_payload), encoding="ascii")
    with pytest.raises(ValueError, match="escape|contained|relative"):
        pooled_config.load_prepared_pooled_decoding_config(run_directory)

    drifted_payload = json.loads(json.dumps(original_payload))
    drifted_payload["base_config"]["frozen_controls"]["seed"] = 99
    (run_directory / "config.json").write_text(json.dumps(drifted_payload), encoding="ascii")
    with pytest.raises(ValueError, match="payload|frozen|canonical|drift"):
        pooled_config.load_prepared_pooled_decoding_config(run_directory)
