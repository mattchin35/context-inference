"""Preparation-only contracts for a new single-session neural analysis."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from src.neural_analysis.lfp_spike_phase_launcher import parse_launcher_command
from src.neural_analysis.single_session_setup import (
    _parser as setup_parser,
    build_analysis_commands,
    build_metadata_analysis_commands,
    inspect_session_readiness,
    main as setup_main,
    scaffold_session_analysis,
)
from src.neural_analysis.task_decoding.pooled.run_session import _parser as pooled_parser
from src.neural_analysis.task_decoding.run_session import _parser as task_parser


def _write_complete_metadata(session_root: Path, cache: Path) -> None:
    """Write small placeholder inputs with the production metadata layout."""
    processed = session_root / "processed"
    processed.mkdir(exist_ok=True)
    (processed / "augmented_trials.csv").write_text("choice_time\n1.0\n", encoding="ascii")
    (processed / "trial_feature_params.json").write_text("{}\n", encoding="ascii")
    cache.mkdir()
    probes = {}
    for probe_id, site_id in (("ProbePFC", "PFC1"), ("ProbeHPC", "HPC1")):
        probe_root = session_root / probe_id
        sorter = probe_root / "sorter"
        sorter.mkdir(parents=True)
        for filename in ("spike_times.npy", "spike_clusters.npy"):
            (sorter / filename).write_bytes(b"placeholder")
        (sorter / "cluster_info.tsv").write_text("cluster_id\tgroup\n", encoding="ascii")
        (probe_root / "lfp.dat").write_bytes(b"placeholder")
        (probe_root / "lfp_preprocessing.json").write_text("{}\n", encoding="ascii")
        (probe_root / "alignment.npz").write_bytes(b"placeholder")
        (probe_root / "channel_quality.csv").write_text(
            "channel,quality\n0,good\n", encoding="ascii"
        )
        probes[probe_id] = {
            "lfp": f"{probe_id}/lfp.dat",
            "alignment": f"{probe_id}/alignment.npz",
            "sorter": f"{probe_id}/sorter",
            "quality": f"{probe_id}/channel_quality.csv",
            "sites": {site_id: 0},
            "unit_channels": [0],
        }
    metadata = {
        "schema_version": "2",
        "session": "synthetic-session",
        "acquisition": "open_ephys",
        "behavior": {"trials": "processed/augmented_trials.csv", "events": None},
        "probes": probes,
        "site_pairs": [["PFC1", "HPC1"]],
        "cache": str(cache.relative_to(session_root)),
    }
    (session_root / "neural_session.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="ascii"
    )


def _write_version_three_metadata(session_root: Path) -> Path:
    """Write portable metadata and return its canonical control-file path.

    Parameters
    ----------
    session_root : pathlib.Path
        Existing synthetic session directory.

    Returns
    -------
    pathlib.Path
        Version-3 metadata path containing scalar relative path identities only.
    """
    payload = {
        "schema_version": "3",
        "session": "synthetic-session",
        "acquisition": "open_ephys",
        "behavior": {"trials": "processed/trials.csv", "events": None},
        "probes": {
            "ProbeA": {
                "lfp": "ephys/A/lfp.dat",
                "alignment": "ephys/A/alignment.npz",
                "sorter": "ephys/A/sorter",
                "quality": "ephys/A/channel_quality.csv",
                "sites": {"PFC1": 0},
            },
            "ProbeB": {
                "lfp": "ephys/B/lfp.dat",
                "alignment": "ephys/B/alignment.npz",
                "sorter": "ephys/B/sorter",
                "quality": "ephys/B/channel_quality.csv",
                "sites": {"HPC1": 0},
            },
        },
        "site_pairs": [["PFC1", "HPC1"]],
        "analysis": {
            "probe_roles": {"pfc": "ProbeA", "hpc": "ProbeB", "ppc": "ProbeB"},
            "ppc": {
                "prerequisite_cache": "processed/ppc_source",
                "preview_cache": "processed/ppc_preview",
                "final_cache": "processed/ppc_final",
                "preview_shuffles": 100,
                "final_shuffles": 1000,
            },
        },
    }
    metadata_path = session_root / "neural_session.json"
    metadata_path.write_text(json.dumps(payload) + "\n", encoding="ascii")
    return metadata_path


def _upgrade_complete_metadata_to_version_three(metadata_path: Path) -> None:
    """Add portable analysis intent to one complete version-2 fixture.

    Parameters
    ----------
    metadata_path : pathlib.Path
        Existing synthetic metadata file whose probe inputs are present.

    Returns
    -------
    None
        The scalar JSON control file is replaced; scientific inputs are unchanged.
    """
    payload = json.loads(metadata_path.read_text(encoding="ascii"))
    payload["schema_version"] = "3"
    payload.pop("cache")
    payload["analysis"] = {
        "probe_roles": {"pfc": "ProbePFC", "hpc": "ProbeHPC", "ppc": "ProbeHPC"},
        "ppc": {
            "prerequisite_cache": "processed/ppc_source",
            "preview_cache": "processed/ppc_preview",
            "final_cache": "processed/ppc_final",
            "preview_shuffles": 100,
            "final_shuffles": 1000,
        },
    }
    metadata_path.write_text(json.dumps(payload) + "\n", encoding="ascii")


def test_scaffold_creates_editable_files_without_overwriting_metadata(tmp_path: Path) -> None:
    """Scaffolding preserves metadata and writes the three missing analysis configs."""
    metadata_path = tmp_path / "neural_session.json"
    metadata_path.write_text('{"user": "content"}\n', encoding="ascii")

    result = scaffold_session_analysis(
        tmp_path,
        pfc_probe_id="ProbePFC",
        hpc_probe_id="ProbeHPC",
        training_window_names=("pre_choice", "post_choice", "full"),
    )

    assert metadata_path.read_text(encoding="ascii") == '{"user": "content"}\n'
    assert {path.name for path in result.created} == {
        "task_decoding_config.json",
        "pooled_task_decoding_config.json",
        "interregional_regression_config.json",
    }
    assert result.preserved == (metadata_path.resolve(),)
    pooled = json.loads((tmp_path / "pooled_task_decoding_config.json").read_text())
    assert pooled == {
        "base_config_path": "task_decoding_config.json",
        "training_window_names": ["pre_choice", "post_choice", "full"],
    }
    interregional = json.loads(
        (tmp_path / "interregional_regression_config.json").read_text()
    )
    assert interregional["session_metadata_path"] == "neural_session.json"
    assert interregional["run"]["output_root"] == "analysis_runs"


def test_scaffold_second_call_preserves_every_existing_file(tmp_path: Path) -> None:
    """Repeated setup is idempotent and never replaces user edits."""
    first = scaffold_session_analysis(
        tmp_path, pfc_probe_id="ProbePFC", hpc_probe_id="ProbeHPC"
    )
    task_path = tmp_path / "task_decoding_config.json"
    task_path.write_text("user edit\n", encoding="ascii")

    second = scaffold_session_analysis(
        tmp_path, pfc_probe_id="OtherPFC", hpc_probe_id="OtherHPC"
    )

    assert len(first.created) == 4
    assert second.created == ()
    assert len(second.preserved) == 4
    assert task_path.read_text(encoding="ascii") == "user edit\n"


def test_complete_session_is_ready_for_all_four_analysis_families(tmp_path: Path) -> None:
    """Small valid sidecars make every requested preparation target ready."""
    cache = tmp_path / "processed" / "ppc-cache"
    scaffold_session_analysis(
        tmp_path, pfc_probe_id="ProbePFC", hpc_probe_id="ProbeHPC"
    )
    _write_complete_metadata(tmp_path, cache)

    report = inspect_session_readiness(
        tmp_path,
        pfc_probe_id="ProbePFC",
        hpc_probe_id="ProbeHPC",
        ppc_probe_id="ProbeHPC",
    )

    assert report.ready is True
    assert tuple(report.analyses) == (
        "task_decoding",
        "condition_generalization",
        "interregional_regression",
        "spike_phase_ppc",
    )
    assert all(item.ready for item in report.analyses.values())


def test_missing_ppc_lfp_input_does_not_block_non_ppc_analyses(tmp_path: Path) -> None:
    """Analysis-specific blockers remain isolated in the readiness report."""
    cache = tmp_path / "processed" / "ppc-cache"
    scaffold_session_analysis(
        tmp_path, pfc_probe_id="ProbePFC", hpc_probe_id="ProbeHPC"
    )
    _write_complete_metadata(tmp_path, cache)
    (tmp_path / "ProbeHPC" / "lfp.dat").unlink()

    report = inspect_session_readiness(
        tmp_path,
        pfc_probe_id="ProbePFC",
        hpc_probe_id="ProbeHPC",
        ppc_probe_id="ProbeHPC",
    )

    assert report.analyses["task_decoding"].ready
    assert report.analyses["condition_generalization"].ready
    assert report.analyses["interregional_regression"].ready
    assert not report.analyses["spike_phase_ppc"].ready
    assert any("lfp" in blocker for blocker in report.analyses["spike_phase_ppc"].blockers)


def test_configured_probe_roles_must_match_requested_roles(tmp_path: Path) -> None:
    """Readiness rejects anatomy arguments that disagree with configuration files."""
    cache = tmp_path / "processed" / "ppc-cache"
    scaffold_session_analysis(
        tmp_path, pfc_probe_id="ProbePFC", hpc_probe_id="ProbeHPC"
    )
    _write_complete_metadata(tmp_path, cache)

    report = inspect_session_readiness(
        tmp_path,
        pfc_probe_id="ProbeHPC",
        hpc_probe_id="ProbePFC",
        ppc_probe_id="ProbeHPC",
    )

    assert not report.analyses["task_decoding"].ready
    assert not report.analyses["condition_generalization"].ready
    assert not report.analyses["interregional_regression"].ready
    assert all(
        any("probe role" in blocker for blocker in report.analyses[name].blockers)
        for name in (
            "task_decoding",
            "condition_generalization",
            "interregional_regression",
        )
    )


def test_commands_are_copyable_and_owned_by_existing_parsers(tmp_path: Path) -> None:
    """Generated commands adopt once and keep preview/final caches independent."""
    cache = tmp_path / "processed" / "ppc cache"
    scaffold_session_analysis(
        tmp_path, pfc_probe_id="ProbePFC", hpc_probe_id="ProbeHPC"
    )
    _write_complete_metadata(tmp_path, cache)

    commands = build_analysis_commands(
        tmp_path,
        ppc_probe_id="ProbeHPC",
        ppc_cache_directory=cache,
    )

    task_tokens = shlex.split(commands["task_decoding"].local_preflight)
    pooled_tokens = shlex.split(commands["condition_generalization"].local_preflight)
    adoption_tokens = shlex.split(commands["spike_phase_ppc"].preparation)
    ppc_preflight_tokens = shlex.split(commands["spike_phase_ppc"].local_preflight)
    ppc_preview_tokens = shlex.split(commands["spike_phase_ppc"].local_preview)
    ppc_final_tokens = shlex.split(commands["spike_phase_ppc"].local_final)
    assert task_parser().parse_args(task_tokens[5:]).command == "dry-run"
    assert pooled_parser().parse_args(pooled_tokens[5:]).command == "dry-run"
    assert adoption_tokens[3:5] == [
        "-m",
        "src.neural_analysis.lfp_summary_cache_adoption",
    ]
    assert adoption_tokens[-2:] == ["--source-cache", str(cache.resolve())]
    preflight = parse_launcher_command(ppc_preflight_tokens[5:])
    preview = parse_launcher_command(ppc_preview_tokens[5:])
    assert preflight.dry_run is True
    assert preflight.shuffle_count == 100
    assert preview.shuffle_count == 100
    final = parse_launcher_command(ppc_final_tokens[5:])
    assert final.shuffle_count == 1000
    assert final.final_run is True
    assert preflight.cache_directory == tmp_path / "processed" / "ppc_prerequisites_preview"
    assert preview.cache_directory == preflight.cache_directory
    assert final.cache_directory == tmp_path / "processed" / "ppc_prerequisites_final"
    assert preview.cache_directory != final.cache_directory
    assert commands["spike_phase_ppc"].cluster_preview.startswith(
        "sbatch src/shell_scripts/hpc_ppc.sh new"
    )
    cluster_preview_tokens = shlex.split(commands["spike_phase_ppc"].cluster_preview)
    cluster_final_tokens = shlex.split(commands["spike_phase_ppc"].cluster_final)
    assert cluster_preview_tokens[cluster_preview_tokens.index("--cache-directory") + 1] == str(
        preview.cache_directory
    )
    assert cluster_final_tokens[cluster_final_tokens.index("--cache-directory") + 1] == str(
        final.cache_directory
    )


def test_setup_commands_accept_only_session_metadata_for_stable_identity() -> None:
    """Normal setup derives the session root, probe roles, and caches from v3.

    Returns
    -------
    None
        Parsed paths are scalar control-file identities; no numerical arrays,
        axes, or physical units are involved.
    """
    metadata_path = Path("/data/session/neural_session.json")
    parser = setup_parser()

    for command in ("init", "check", "prepare"):
        arguments = parser.parse_args(
            [command, "--session-metadata", str(metadata_path)]
        )
        assert arguments.command == command
        assert arguments.session_metadata == metadata_path
        assert not hasattr(arguments, "session_root")
        assert not hasattr(arguments, "pfc_probe")
        assert not hasattr(arguments, "hpc_probe")
        assert not hasattr(arguments, "ppc_probe")
        assert not hasattr(arguments, "ppc_cache_directory")


def test_readiness_only_reads_control_files_and_never_creates_outputs(tmp_path: Path) -> None:
    """Inspection does not create analysis directories, caches, or run receipts."""
    scaffold_session_analysis(
        tmp_path, pfc_probe_id="ProbePFC", hpc_probe_id="ProbeHPC"
    )
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    report = inspect_session_readiness(
        tmp_path,
        pfc_probe_id="ProbePFC",
        hpc_probe_id="ProbeHPC",
        ppc_probe_id="ProbeHPC",
    )
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert not report.ready
    assert before == after
    assert not (tmp_path / "analysis_runs").exists()


def test_metadata_commands_include_simple_slurm_preparation(tmp_path: Path) -> None:
    """The metadata workflow exposes one wrapper-owned cluster preparation call.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed session root containing scalar metadata only.

    Returns
    -------
    None
        The assertion covers command text; no scheduler job or analysis runs.
    """
    metadata_path = _write_version_three_metadata(tmp_path)

    commands = build_metadata_analysis_commands(metadata_path)["spike_phase_ppc"]

    assert commands.cluster_preparation == shlex.join(
        (
            "sbatch",
            "src/shell_scripts/hpc_lfp_prepare.sh",
            "--session-metadata",
            str(metadata_path.resolve()),
        )
    )
    assert commands.preparation == shlex.join(
        (
            "uv",
            "run",
            "python",
            "-m",
            "src.neural_analysis.single_session_setup",
            "prepare",
            "--session-metadata",
            str(metadata_path.resolve()),
        )
    )


def test_blocked_check_prints_labeled_instructions_for_every_analysis(
    tmp_path: Path,
    capsys: object,
) -> None:
    """Missing inputs retain all eventual commands without creating outputs.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed session containing only version-3 scalar metadata.
    capsys : pytest capture fixture
        Captures human-readable stdout and stderr without shell execution.

    Returns
    -------
    None
        No scheduler or numerical runner is invoked by this test.
    """
    metadata_path = _write_version_three_metadata(tmp_path)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    exit_code = setup_main(("check", "--session-metadata", str(metadata_path)))
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))

    assert exit_code == 2
    assert before == after
    assert "task_decoding: BLOCKED" in output
    assert "condition_generalization: BLOCKED" in output
    assert "interregional_regression: BLOCKED" in output
    assert "spike_phase_ppc: BLOCKED" in output
    assert "Command instructions:" in output
    assert "task_decoding:" in output
    assert "condition_generalization:" in output
    assert "interregional_regression:" in output
    assert "spike_phase_ppc:" in output
    assert "local dry-run:" in output
    assert "cluster submission:" in output
    assert "workstation preparation:" in output
    assert "cluster preparation:" in output
    assert "preview dry-run:" in output
    assert "src/shell_scripts/hpc_lfp_prepare.sh" in output
    assert "src/shell_scripts/task_variable_decoding_slurm.sh" in output
    assert "src/shell_scripts/pooled_task_variable_decoding_slurm.sh" in output
    assert "src/shell_scripts/interregional_regression_slurm.sh" in output
    assert not (tmp_path / "analysis_runs").exists()


def test_ready_check_prints_all_commands_and_returns_success(
    tmp_path: Path,
    capsys: object,
) -> None:
    """A complete session reports every local and cluster execution route.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-managed complete synthetic session.
    capsys : pytest capture fixture
        Captures deterministic command guidance.

    Returns
    -------
    None
        Assertions concern text and status only; no analysis is executed.
    """
    scaffold_session_analysis(
        tmp_path,
        pfc_probe_id="ProbePFC",
        hpc_probe_id="ProbeHPC",
    )
    preview_cache = tmp_path / "processed" / "ppc_preview"
    _write_complete_metadata(tmp_path, preview_cache)
    _upgrade_complete_metadata_to_version_three(tmp_path / "neural_session.json")

    exit_code = setup_main(
        ("check", "--session-metadata", str(tmp_path / "neural_session.json"))
    )
    output = capsys.readouterr().out  # type: ignore[attr-defined]

    assert exit_code == 0
    assert output.count("local dry-run:") == 3
    assert output.count("cluster submission:") == 3
    assert output.count("workstation preparation:") == 1
    assert output.count("cluster preparation:") == 1
    assert output.count("preview dry-run:") == 1
    assert output.count("cluster preview:") == 1
    assert output.count("cluster final:") == 1


def test_single_session_guide_documents_every_cluster_job() -> None:
    """The quickstart includes dry-run and submission guidance for all families."""
    guide = (
        Path(__file__).resolve().parents[2]
        / "neural_analysis"
        / "SINGLE_SESSION_SETUP.md"
    ).read_text(encoding="ascii")

    for heading in (
        "Task-variable decoding",
        "Condition generalization",
        "Interregional regression",
        "PPC preparation, preview, and final",
    ):
        assert heading in guide
    for command in (
        "src.neural_analysis.task_decoding.run_session dry-run",
        "src/shell_scripts/task_variable_decoding_slurm.sh",
        "src.neural_analysis.task_decoding.pooled.run_session dry-run",
        "src/shell_scripts/pooled_task_variable_decoding_slurm.sh",
        "src.neural_analysis.interregional.run_session dry-run",
        "src/shell_scripts/interregional_regression_slurm.sh",
        "src/shell_scripts/hpc_lfp_prepare.sh",
        "src/shell_scripts/hpc_ppc.sh preview",
        "src/shell_scripts/hpc_ppc.sh final",
    ):
        assert command in guide
    assert "does not run task-variable decoding" in guide


def test_single_session_guide_documents_non_destructive_copy_back() -> None:
    """The quickstart copies all session outputs without deleting local data."""
    repository_root = Path(__file__).resolve().parents[3]
    guide = (
        repository_root / "src" / "neural_analysis" / "SINGLE_SESSION_SETUP.md"
    ).read_text(encoding="ascii")
    readme = (
        repository_root / "src" / "neural_analysis" / "README.md"
    ).read_text(encoding="ascii")

    section = guide.split("## 8. Copy cluster outputs back to the workstation", 1)[1]
    assert 'LOCAL_SESSION="/home/matt/Documents/EXPERIMENTS/contextProjectData/' in section
    assert 'REMOTE_SESSION="mchin1@mchin1.hpc.einsteinmed.edu:' in section
    assert "rsync -a --dry-run --itemize-changes --relative" in section
    assert "rsync -a --info=progress2 --relative" in section
    assert '"$REMOTE_SESSION/./analysis_runs/"' in section
    assert '"$REMOTE_SESSION/./processed/ppc_prerequisites_preview/"' in section
    assert '"$REMOTE_SESSION/./processed/ppc_prerequisites_final/"' in section
    assert '"$REMOTE_SESSION/./processed/ppc_prerequisites_adoption.json"' in section
    assert "lfp_summary_work" in section
    assert "intentionally excluded" in section
    assert "--delete" not in section
    assert "Task-variable decoding" in guide
    assert "PPC preparation, preview, and final" in guide
    assert "SINGLE_SESSION_SETUP.md#8-copy-cluster-outputs-back-to-the-workstation" in readme
