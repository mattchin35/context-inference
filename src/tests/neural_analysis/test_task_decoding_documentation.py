"""WP8 contracts for task-decoding documentation and portable commands.

These tests exercise only repository documentation, small temporary metadata,
and bounded planning/preparation seams. They never fit a decoder or load sorter
spike arrays.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.neural_analysis.task_decoding import config as decoding_config
from src.neural_analysis.task_decoding import pipeline, run_session
from src.neural_analysis.webapp import task_decoding_views
from src.tests.neural_analysis.task_decoding.test_pipeline import (
    expected_follow_up_commands,
    fail_if_called,
    prepare_with_clean_identity,
    write_session_inputs,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCIENTIST_README = REPOSITORY_ROOT / "src" / "neural_analysis" / "README.md"
PACKAGE_DIRECTORY = REPOSITORY_ROOT / "src" / "neural_analysis" / "task_decoding"
PACKAGE_README = PACKAGE_DIRECTORY / "README.md"
EXAMPLE_CONFIG = (
    REPOSITORY_ROOT
    / "docs"
    / "examples"
    / "neural_analysis"
    / "task_decoding_config.json"
)
SESSION_MODULE = "src.neural_analysis.task_decoding.run_session"
BATCH_MODULE = "src.neural_analysis.task_decoding.run_batch"


def _read_required_text(path: Path) -> str:
    """Read one required UTF-8 documentation artifact.

    Parameters
    ----------
    path : pathlib.Path
        Repository documentation or example path with no physical units.

    Returns
    -------
    str
        Complete UTF-8 file content.
    """
    assert path.is_file(), f"Missing required WP8 artifact: {path}"
    return path.read_text(encoding="utf-8")


def _write_example_inputs(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Copy the portable example beside its three small declared inputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-owned directory receiving one synthetic session tree.
    payload : dict[str, object]
        JSON example whose path fields are session-relative and unitless.

    Returns
    -------
    pathlib.Path
        Copied configuration accepted by the production configuration loader.
    """
    session_root = tmp_path / "session"
    session_root.mkdir()
    path_contents = {
        "session_metadata_path": "{}\n",
        "augmented_trial_path": "cur_trial\n0\n",
        "trial_feature_parameter_path": "{}\n",
    }
    for field_name, content in path_contents.items():
        declared_path = session_root / str(payload[field_name])
        declared_path.parent.mkdir(parents=True, exist_ok=True)
        declared_path.write_text(content, encoding="ascii")
    config_path = session_root / "task_decoding_config.json"
    config_path.write_text(json.dumps(payload), encoding="ascii")
    return config_path


def test_portable_example_exposes_every_config_field_and_loads(tmp_path):
    """The reusable example is explicit, portable, and production-loadable."""
    example_text = _read_required_text(EXAMPLE_CONFIG)
    assert "CT026" not in example_text
    payload = json.loads(example_text)
    assert set(payload) == decoding_config._ALLOWED_TOP_LEVEL_FIELDS
    for field_name in (
        "session_metadata_path",
        "augmented_trial_path",
        "trial_feature_parameter_path",
        "output_root",
    ):
        assert not Path(payload[field_name]).is_absolute()

    config = decoding_config.load_task_decoding_config(
        _write_example_inputs(tmp_path, payload)
    )

    assert config.session_metadata_path.name == "neural_session.json"
    assert config.output_root.is_relative_to(config.session_root)
    assert config.pfc_region.probe_id != config.hpc_region.probe_id


def test_maintainer_readme_owns_every_package_file_and_external_entrypoint():
    """Maintainers can locate every package file and adjacent user entrypoint."""
    readme = _read_required_text(PACKAGE_README)
    for module_path in PACKAGE_DIRECTORY.glob("*.py"):
        assert module_path.name in readme
    assert "psth_webapp.py" in readme
    assert "webapp/task_decoding_views.py" in readme
    assert "docs/examples/neural_analysis/task_decoding_config.json" in readme
    assert "WP11" in readme and "WP13" in readme


def test_scientist_quickstart_documents_all_current_cli_routes():
    """The scientist guide gives copyable single-session and batch workflows."""
    readme = " ".join(_read_required_text(SCIENTIST_README).split())
    required_fragments = (
        f"python -m {SESSION_MODULE} dry-run",
        f"python -m {SESSION_MODULE} new",
        f"python -m {SESSION_MODULE} resume",
        f"python -m {SESSION_MODULE} status",
        f"python -m {BATCH_MODULE} dry-run",
        f"python -m {BATCH_MODULE} new",
        "--detach",
        "--rerun",
        "--resource-run-directory",
        "Task-variable decoding results",
    )
    for fragment in required_fragments:
        assert fragment in readme


def test_cli_help_succeeds_and_lists_documented_public_modes():
    """Both standard-library entry modules advertise their stable public modes."""
    expected_modes = {
        SESSION_MODULE: {"dry-run", "new", "resume", "status"},
        BATCH_MODULE: {"dry-run", "new"},
    }
    for module_name, modes in expected_modes.items():
        completed = subprocess.run(
            [sys.executable, "-m", module_name, "--help"],
            cwd=REPOSITORY_ROOT,
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert modes <= set(completed.stdout.replace("{", " ").replace("}", " ").replace(",", " ").split())


def test_documented_dry_run_uses_small_inputs_without_loading_spikes(
    monkeypatch,
    tmp_path,
    capsys,
):
    """The documented dry-run command plans a real fixture without spike loads."""
    paths = write_session_inputs(tmp_path)
    original_load = np.load

    def forbid_sorter_spike_load(path, *args, **kwargs):
        """Allow alignment archives but reject sorter arrays during planning."""
        if Path(path).name in {"spike_clusters.npy", "spike_times.npy"}:
            raise AssertionError("documented dry-run loaded sorter spike arrays")
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(pipeline.activity, "load_region_activity", fail_if_called)
    monkeypatch.setattr(pipeline.activity, "build_session_rate_tensors", fail_if_called)
    monkeypatch.setattr(pipeline.modeling, "decode_target", fail_if_called)
    monkeypatch.setattr(np, "load", forbid_sorter_spike_load)

    assert run_session.main(["dry-run", "--config", str(paths["config"])]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["session_id"] == "synthetic-session"
    assert report["tensor_allocation_bytes"] > 0


def test_preparation_writes_exact_follow_up_commands_before_execution(
    monkeypatch,
    tmp_path,
):
    """Resume and status instructions exist before a potentially long stage starts."""
    paths = write_session_inputs(tmp_path)
    run_directory = prepare_with_clean_identity(monkeypatch, paths, mode="foreground")
    expected_resume, expected_status = expected_follow_up_commands(run_directory)
    entered_execution = False

    def inspect_prepared_commands(directory: Path) -> None:
        """Represent a long stage while checking its instructions already exist."""
        nonlocal entered_execution
        entered_execution = True
        assert Path(directory) == run_directory
        assert (run_directory / "resume_command.txt").read_text(
            encoding="utf-8"
        ) == expected_resume
        assert (run_directory / "status_command.txt").read_text(
            encoding="utf-8"
        ) == expected_status

    monkeypatch.setattr(pipeline, "run_prepared_task_decoding", inspect_prepared_commands)
    assert run_session.main(["resume", "--run-directory", str(run_directory)]) == 0
    assert entered_execution


def test_quickstart_paths_match_implementation_constants():
    """User-facing output locations stay synchronized with runtime constants."""
    readme = _read_required_text(SCIENTIST_README)
    for fragment in (
        pipeline._RUN_PREFIX,
        pipeline._STATE_FILE,
        pipeline._RESULT_FILE,
        task_decoding_views.DEFAULT_TASK_DECODING_RESULTS_ROOT,
        "checkpoints/",
        "figures/",
        "run.log",
        "console.log",
    ):
        assert fragment in readme


def test_quickstart_defines_safe_unattended_and_backfill_workflows():
    """The guide makes unattended ownership and narrow CSV migration explicit."""
    readme = " ".join(_read_required_text(SCIENTIST_README).lower().split())
    for fragment in (
        "close the terminal",
        "close the codex task",
        "one-shot",
        "timestamped backup",
        "sole missing column",
        "rewards_in_block",
        "normal behavior regeneration",
        "do not hand-edit",
        "do not rerun unrelated models",
        "one worker",
        "--resource-run-directory",
        "not discovered automatically",
    ):
        assert fragment in readme
