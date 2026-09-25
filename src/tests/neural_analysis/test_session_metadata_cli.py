"""Command-line contracts for creating and validating session metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.neural_analysis.session_metadata import load_session_metadata
from src.neural_analysis.session_metadata_cli import main


def test_create_writes_one_deterministic_structurally_valid_skeleton(
    tmp_path: Path,
) -> None:
    """Create writes only neural_session.json and does not scan the session tree."""
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "unrelated-recording.bin").write_bytes(b"do not inspect")

    assert main(["create", "--session-root", str(first_root)]) == 0
    assert main(["create", "--session-root", str(second_root)]) == 0

    first_path = first_root / "neural_session.json"
    second_path = second_root / "neural_session.json"
    assert first_path.read_bytes() == second_path.read_bytes()
    assert sorted(path.name for path in first_root.iterdir()) == [
        "neural_session.json",
        "unrelated-recording.bin",
    ]
    metadata = load_session_metadata(first_path)
    assert metadata.schema_version == "2"
    assert metadata.session == ""
    assert metadata.acquisition == "open_ephys"
    assert metadata.probes == ()


def test_create_refuses_to_replace_existing_metadata(tmp_path: Path) -> None:
    """Create preserves an existing user-edited metadata document."""
    metadata_path = tmp_path / "neural_session.json"
    metadata_path.write_text("user content\n", encoding="ascii")

    assert main(["create", "--session-root", str(tmp_path)]) != 0
    assert metadata_path.read_text(encoding="ascii") == "user content\n"


def test_create_requires_an_existing_session_directory(tmp_path: Path) -> None:
    """Create does not invent a missing session directory hierarchy."""
    missing_root = tmp_path / "missing"

    assert main(["create", "--session-root", str(missing_root)]) != 0
    assert not missing_root.exists()


def test_validate_without_action_reports_all_four_categories(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """General validation reports availability without requiring all actions."""
    assert main(["create", "--session-root", str(tmp_path)]) == 0
    capsys.readouterr()

    return_code = main(["validate", "--metadata", str(tmp_path / "neural_session.json")])
    output = capsys.readouterr().out

    assert return_code == 0
    assert "webapp:" in output
    assert "power:" in output
    assert "synchrony:" in output
    assert "spike-phase:" in output
    assert "unavailable" in output


def test_validate_requested_action_returns_nonzero_when_inputs_are_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An explicitly requested unavailable action fails with named missing inputs."""
    assert main(["create", "--session-root", str(tmp_path)]) == 0
    capsys.readouterr()

    return_code = main(
        [
            "validate",
            "--metadata",
            str(tmp_path / "neural_session.json"),
            "--action",
            "spike-phase",
        ]
    )
    output = capsys.readouterr().out

    assert return_code != 0
    assert "spike-phase: unavailable" in output
    assert "probes" in output


def test_validate_malformed_metadata_returns_nonzero_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Malformed JSON produces one concise user-facing error."""
    metadata_path = tmp_path / "neural_session.json"
    metadata_path.write_text("{broken\n", encoding="ascii")

    return_code = main(["validate", "--metadata", str(metadata_path)])
    captured = capsys.readouterr()

    assert return_code != 0
    assert "error:" in captured.err.lower()
    assert "traceback" not in captured.err.lower()


def test_create_skeleton_is_compact_and_keeps_optional_fields_visible(tmp_path: Path) -> None:
    """The editable skeleton contains only the probe-centered version-2 fields."""
    assert main(["create", "--session-root", str(tmp_path)]) == 0

    payload = json.loads((tmp_path / "neural_session.json").read_text(encoding="ascii"))

    assert payload == {
        "schema_version": "2",
        "session": "",
        "acquisition": "open_ephys",
        "behavior": {"trials": "", "events": None},
        "probes": {},
        "site_pairs": [],
        "cache": None,
    }
