"""WP8 documentation contracts for portable pooled cluster operation."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
SCIENTIST_README = REPOSITORY_ROOT / "src" / "neural_analysis" / "README.md"
IMPLEMENTATION_PLAN = (
    REPOSITORY_ROOT / "docs" / "task_variable_condition_generalization_implementation_plan.md"
)


def test_pooled_cluster_workflow_documents_safe_sync_and_exact_commands() -> None:
    """The operator guide exposes portable sync, array, status, and return steps."""
    text = " ".join(
        (
            SCIENTIST_README.read_text(encoding="utf-8")
            + IMPLEMENTATION_PLAN.read_text(encoding="utf-8")
        ).split()
    )
    for fragment in (
        "pooled_task_variable_decoding_slurm.sh submit-new",
        "pooled_task_variable_decoding_slurm.sh submit-array",
        "--partition=normal",
        "--mem=3G",
        "--time=24:00:00",
        "/home/matt/Documents/EXPERIMENTS/contextProjectData",
        "/gs/gsfs0/users/mchin1/contextProjectData",
        "--exclude='/analysis_runs/'",
        ".incoming-<run_id>",
        "rsync -a --info=progress2",
    ):
        assert fragment in text
    assert "rsync --delete" not in text

