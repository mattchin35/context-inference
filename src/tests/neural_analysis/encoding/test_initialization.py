"""Small-input config creation, preservation, and CLI contracts; no neural work."""

import json
from pathlib import Path
import subprocess

import pytest

from src.neural_analysis.encoding import initialization, run_session
from src.neural_analysis.encoding.configuration import load_encoding_config
from .test_configuration import config_file as config_file_fixture

config_file = config_file_fixture


@pytest.fixture
def initialization_session(config_file):
    """Return v3 metadata Path and task locator JSON with intentionally invalid unused settings."""
    root = config_file.parent
    payload = json.loads(config_file.read_text())
    task = {"augmented_trial_path": payload["augmented_trials"],
            "trial_feature_parameter_path": payload["feature_parameters"],
            "pfc_pc_count": "unused invalid decoder setting"}
    (root / "task_decoding_config.json").write_text(json.dumps(task))
    return root / "neural_session.json"


def test_create_config_from_metadata_and_task_locators(initialization_session, monkeypatch):
    """Generated paths are session-relative and scientific defaults match the reviewed example."""
    metadata = initialization_session
    original = {path: path.read_bytes() for path in metadata.parent.rglob("*") if path.is_file()}
    monkeypatch.chdir(metadata.parent.parent)
    report = initialization.initialize_encoding_config(metadata)
    destination = metadata.parent / "context_encoding_config.json"
    assert report == {"status": "created", "config_path": str(destination)}
    payload = json.loads(destination.read_text())
    example = Path(__file__).parents[4] / "docs/examples/neural_analysis/context_encoding_config.json"
    expected = json.loads(example.read_text())
    expected["augmented_trials"] = "processed/augmented.csv"
    assert payload == expected
    config = load_encoding_config(destination)
    assert config.probe_roles == {"PFC": "ProbeA", "HPC": "ProbeB"}
    assert config.penalty_modes == ("same_numeric", "matched_context")
    assert not config.run_root.exists() and not config.intermediate_root.exists()
    assert all(path.read_bytes() == content for path, content in original.items())


@pytest.mark.parametrize("width", [500, 100, 50, 20])
def test_explicit_inputs_and_bin_width_override_task_defaults(initialization_session, width):
    """Relative overrides resolve from the session; malformed unused task JSON is ignored."""
    root = initialization_session.parent
    (root / "task_decoding_config.json").write_text("malformed unused JSON")
    (root / "processed/other.csv").write_text("cur_trial\n1\n")
    (root / "processed/other.json").write_text("{}")
    report = initialization.initialize_encoding_config(
        initialization_session, augmented_trials="processed/other.csv",
        feature_parameters=root / "processed/other.json", bin_width_ms=width,
    )
    config = load_encoding_config(report["config_path"])
    assert config.bin_width_ms == width
    assert config.augmented_trials == root / "processed/other.csv"
    assert config.feature_parameters == root / "processed/other.json"
    payload = json.loads(Path(report["config_path"]).read_text())
    assert payload["feature_parameters"] == "processed/other.json"


def test_one_explicit_input_uses_task_locator_for_the_other(initialization_session):
    """Overrides are independent; named input paths need not be guessed from directory contents."""
    root = initialization_session.parent
    (root / "processed/another.csv").write_text("cur_trial\n1\n")
    report = initialization.initialize_encoding_config(initialization_session, augmented_trials="processed/another.csv")
    config = load_encoding_config(report["config_path"])
    assert config.augmented_trials == root / "processed/another.csv"
    assert config.feature_parameters == root / "processed/trial_feature_params.json"


def test_existing_config_preserved_even_when_sources_are_unavailable(initialization_session):
    """Repeated init never validates, rewrites or repairs a pre-existing user config."""
    first = initialization.initialize_encoding_config(initialization_session)
    path = Path(first["config_path"])
    path.write_bytes(b"user-edited config, deliberately not valid JSON\n")
    initialization_session.unlink()
    (path.parent / "task_decoding_config.json").unlink()
    second = initialization.initialize_encoding_config(initialization_session)
    assert second == {"status": "preserved", "config_path": str(path)}
    assert path.read_bytes() == b"user-edited config, deliberately not valid JSON\n"


@pytest.mark.parametrize("defect, message", [
    ("no_task", "--augmented-trials"), ("bad_task", "task_decoding_config.json"),
    ("missing_locator", "--feature-parameters"), ("missing_file", "--augmented-trials"),
    ("bad_feature_json", "feature"), ("bad_metadata", "metadata"),
    ("invalid_width", "bin_width_ms"), ("invalid_roles", "probe_roles"),
])
def test_invalid_inputs_leave_no_partial_config(initialization_session, defect, message):
    """Ordinary missing/invalid inputs fail before any destination or analysis directory is written."""
    root = initialization_session.parent
    task = root / "task_decoding_config.json"
    kwargs = {}
    if defect == "no_task":
        task.unlink()
    elif defect == "bad_task":
        task.write_text("{")
    elif defect == "missing_locator":
        task.write_text(json.dumps({"augmented_trial_path": "processed/augmented.csv"}))
    elif defect == "missing_file":
        kwargs["augmented_trials"] = "processed/absent.csv"
    elif defect == "bad_feature_json":
        (root / "processed/trial_feature_params.json").write_text("[]")
    elif defect == "bad_metadata":
        initialization_session.write_text("{")
    elif defect == "invalid_width":
        kwargs["bin_width_ms"] = True
    else:
        kwargs["probe_roles"] = {"PFC": "missing"}
    with pytest.raises(ValueError, match=message):
        initialization.initialize_encoding_config(initialization_session, **kwargs)
    assert not (root / "context_encoding_config.json").exists()
    assert not (root / "analysis_runs").exists()
    assert not (root / "processed/context_encoding").exists()


def test_v2_metadata_requires_roles_and_allows_single_region(initialization_session):
    """Version-2 inputs need explicit anatomical roles; a one-region mapping remains valid."""
    payload = json.loads(initialization_session.read_text())
    payload["schema_version"] = "2"
    payload.pop("analysis")
    initialization_session.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="--pfc-probe|--hpc-probe"):
        initialization.initialize_encoding_config(initialization_session)
    report = initialization.initialize_encoding_config(initialization_session, probe_roles={"PFC": "ProbeA"})
    assert load_encoding_config(report["config_path"]).probe_roles == {"PFC": "ProbeA"}


def test_init_cli_reports_created_and_preserved_without_running_analysis(initialization_session, monkeypatch, capsys):
    """CLI initialization bypasses the scientific runner and reports a JSON path/status."""
    def forbidden(*args, **kwargs):
        """Any analysis invocation is a test failure; arbitrary arguments, no return."""
        pytest.fail("init-config cannot execute an encoding analysis")

    monkeypatch.setattr(run_session, "run_encoding_session", forbidden)
    args = ["init-config", "--session-metadata", str(initialization_session),
            "--bin-width-ms", "100", "--hpc-probe", "ProbeA", "--pfc-probe", "ProbeB"]
    assert run_session.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "created"
    config = load_encoding_config(report["config_path"])
    assert config.probe_roles == {"PFC": "ProbeB", "HPC": "ProbeA"}
    assert config.bin_width_ms == 100
    assert run_session.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "preserved"


def test_init_cli_explicit_paths_and_actionable_errors(initialization_session, capsys):
    """No task config is required when both named inputs are supplied through the CLI."""
    root = initialization_session.parent
    (root / "task_decoding_config.json").unlink()
    args = ["init-config", "--session-metadata", str(initialization_session)]
    assert run_session.main(args) == 1
    assert "--augmented-trials" in capsys.readouterr().err
    assert run_session.main(args + ["--augmented-trials", "processed/augmented.csv",
                                   "--feature-parameters", "processed/trial_feature_params.json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "created"


def test_init_subprocess_does_not_import_numerical_pipeline(initialization_session):
    """Fresh CLI initializes using metadata/JSON only, without NumPy or scientific pipeline imports."""
    script = (
        "import sys; from src.neural_analysis.encoding.run_session import main; "
        "assert main(['init-config','--session-metadata',sys.argv[1]]) == 0; "
        "assert 'numpy' not in sys.modules; "
        "assert 'src.neural_analysis.encoding.pipeline' not in sys.modules"
    )
    completed = subprocess.run(
        ["uv", "run", "--frozen", "--no-sync", "--offline", "python", "-c", script, str(initialization_session)],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
