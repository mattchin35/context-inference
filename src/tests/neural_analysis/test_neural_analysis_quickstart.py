"""Usability contracts for the neural-analysis README and example metadata."""

from __future__ import annotations

from pathlib import Path
import shutil

from src.neural_analysis import session_metadata_cli
from src.neural_analysis.lfp_spike_phase_launcher import parse_launcher_command
from src.neural_analysis.psth_webapp import parse_webapp_arguments
from src.neural_analysis.session_metadata import load_session_metadata


_ROOT = Path(__file__).resolve().parents[3]
_README = _ROOT / "src/neural_analysis/README.md"
_EXAMPLES = _ROOT / "docs/examples/neural_analysis"


def test_three_examples_cover_open_ephys_spikeglx_and_arbitrary_names() -> None:
    """Examples are valid compact files with one acquisition family per session."""
    expected = {
        "open_ephys_session.json": ("CT026-example", {"open_ephys"}),
        "spikeglx_session.json": ("CT014-example", {"spikeglx"}),
        "differently_named_session.json": ("Mouse-Z-example", {"open_ephys"}),
    }

    assert {path.name for path in _EXAMPLES.glob("*.json")} == set(expected)
    for name, (session_id, acquisition_families) in expected.items():
        metadata = load_session_metadata(_EXAMPLES / name)
        assert metadata.session == session_id
        assert {probe.acquisition_family for probe in metadata.probes} == (
            acquisition_families
        )
        assert all(site.probe_id in {probe.probe_id for probe in metadata.probes} for site in metadata.sites)


def test_quickstart_documents_required_user_workflow() -> None:
    """A reader can find every supported entry point without reading source code."""
    text = _README.read_text(encoding="ascii")

    for phrase in (
        "Create and validate session metadata",
        "Launch the webapp",
        "Run Spike-phase/PPC computation",
        "Local preview",
        "Slurm",
        "Resume and recover",
        "Where outputs go",
        "Python API",
        "docs/examples/neural_analysis/open_ephys_session.json",
        "docs/examples/neural_analysis/spikeglx_session.json",
        "docs/examples/neural_analysis/differently_named_session.json",
    ):
        assert phrase in text


def test_documented_commands_are_owned_by_existing_parsers() -> None:
    """Copyable argument shapes parse without executing analysis or submitting jobs."""
    create = session_metadata_cli._parser().parse_args(
        ["create", "--session-root", "/data/session"]
    )
    validate = session_metadata_cli._parser().parse_args(
        ["validate", "--metadata", "/data/session/neural_session.json"]
    )
    webapp = parse_webapp_arguments(
        ["--session-metadata", "/data/session/neural_session.json"]
    )
    new_arguments = [
        "new",
        "--session-metadata",
        "/data/session/neural_session.json",
        "--probe",
        "probe-main",
        "--cache-directory",
        "/data/session/processed/lfp-summary-cache",
        "--shuffles",
        "100",
        "--workers",
        "8",
    ]
    preview = parse_launcher_command([*new_arguments, "--dry-run"])
    local = parse_launcher_command(new_arguments)
    resume = parse_launcher_command(
        ["resume", "--run-directory", "/data/session/analysis_runs/exact-run"]
    )
    recover = parse_launcher_command(
        ["recover-report", "--run-directory", "/data/session/analysis_runs/exact-run"]
    )
    rerender = parse_launcher_command(
        ["rerender-report", "--run-directory", "/data/session/analysis_runs/exact-run"]
    )

    assert create.command == "create"
    assert validate.command == "validate"
    assert webapp.session_metadata == Path("/data/session/neural_session.json")
    assert preview.dry_run is True
    assert local.dry_run is False
    assert {resume.mode, recover.mode, rerender.mode} == {
        "resume",
        "recover-report",
        "rerender-report",
    }


def test_each_example_can_be_copied_and_validated_as_an_incomplete_template(
    tmp_path: Path,
) -> None:
    """The documented copy-and-edit start succeeds without requiring real files."""
    for example in sorted(_EXAMPLES.glob("*.json")):
        session_root = tmp_path / example.stem
        session_root.mkdir()
        metadata_path = session_root / "neural_session.json"
        shutil.copyfile(example, metadata_path)

        assert session_metadata_cli.main(
            ["validate", "--metadata", str(metadata_path)]
        ) == 0
