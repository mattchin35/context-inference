"""Preparation-only contracts for a new single-session neural analysis."""

from __future__ import annotations

import json
from pathlib import Path
import shlex

from src.neural_analysis.lfp_spike_phase_launcher import parse_launcher_command
from src.neural_analysis.single_session_setup import (
    build_analysis_commands,
    inspect_session_readiness,
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
    """Generated local commands retain exact existing command interfaces."""
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
    ppc_preview_tokens = shlex.split(commands["spike_phase_ppc"].local_preview)
    ppc_final_tokens = shlex.split(commands["spike_phase_ppc"].local_final)
    assert task_parser().parse_args(task_tokens[5:]).command == "dry-run"
    assert pooled_parser().parse_args(pooled_tokens[5:]).command == "dry-run"
    assert parse_launcher_command(ppc_preview_tokens[5:]).shuffle_count == 100
    final = parse_launcher_command(ppc_final_tokens[5:])
    assert final.shuffle_count == 1000
    assert final.final_run is True
    assert "--dry-run" in commands["spike_phase_ppc"].local_preflight
    assert commands["spike_phase_ppc"].cluster_preview.startswith(
        "sbatch src/shell_scripts/hpc_ppc.sh new"
    )


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
