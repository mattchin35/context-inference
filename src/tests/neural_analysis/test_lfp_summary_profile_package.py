"""Canonical ownership contracts for LFP-summary profiling support."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


MODULES = {
    "ppc_profile": (
        "src.neural_analysis.lfp_summary_ppc_profile",
        (
            "PPCProfileWorkload",
            "PPCProfileResult",
            "PPCProductionProfileResult",
            "GroupedPPCProfileMetadata",
            "GroupedPPCProfileResult",
            "RepresentativePPCProfileJob",
            "representative_serial_ppc_workloads",
            "select_representative_ppc_profile_job",
            "profile_serial_ppc_workload",
            "profile_production_ppc_job",
            "summarize_grouped_ppc_profile_metadata",
            "profile_grouped_ppc_component",
        ),
    ),
    "ct026_profile_adapter": (
        "src.neural_analysis.lfp_summary_ct026_profile_adapter",
        (
            "CT026PPCProfileSlice",
            "CT026PPCProfileAdapterDependencies",
            "run_ct026_ppc_profile_adapter",
            "make_production_ct026_profile_dependencies",
            "slice_ct026_profile_job",
            "run_isolated_ct026_profile_job",
            "recover_ct026_profile_work",
            "production_git_fingerprint",
        ),
    ),
    "ct026_profile_locks": (
        "src.neural_analysis.lfp_summary_ct026_profile_locks",
        ("acquire_ct026_profile_run_lock",),
    ),
    "ct026_profile_runner": (
        "src.neural_analysis.lfp_summary_ct026_profile_runner",
        (
            "PhaseProfilePreparation",
            "CT026PPCProfileRunResult",
            "run_ct026_ppc_profile",
        ),
    ),
}


@pytest.mark.parametrize(("canonical_name", "legacy_contract"), MODULES.items())
def test_profile_modules_have_one_canonical_owner(
    canonical_name: str,
    legacy_contract: tuple[str, tuple[str, ...]],
) -> None:
    """Preserve old imports as aliases of their like-named canonical owner."""
    legacy_name, public_names = legacy_contract
    canonical_module_name = f"src.neural_analysis.lfp_summary.{canonical_name}"
    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_module_name)

    assert legacy is canonical
    for name in public_names:
        value = getattr(canonical, name)
        assert getattr(legacy, name) is value
        assert value.__module__ == canonical_module_name


def test_ct026_adapter_uses_canonical_profile_dependencies() -> None:
    """Keep CT026 explicit while removing back-imports to root compatibility paths."""
    adapter = importlib.import_module(
        "src.neural_analysis.lfp_summary.ct026_profile_adapter"
    )
    profile = importlib.import_module("src.neural_analysis.lfp_summary.ppc_profile")
    locks = importlib.import_module("src.neural_analysis.lfp_summary.ct026_profile_locks")
    runner = importlib.import_module("src.neural_analysis.lfp_summary.ct026_profile_runner")

    assert adapter.RepresentativePPCProfileJob is profile.RepresentativePPCProfileJob
    assert adapter.profile_grouped_ppc_component is profile.profile_grouped_ppc_component
    assert adapter.profile_production_ppc_job is profile.profile_production_ppc_job
    assert adapter.select_representative_ppc_profile_job is profile.select_representative_ppc_profile_job
    assert adapter.acquire_ct026_profile_run_lock is locks.acquire_ct026_profile_run_lock
    assert adapter.run_ct026_ppc_profile is runner.run_ct026_ppc_profile

    tree = ast.parse(Path(adapter.__file__).read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        name.startswith("src.neural_analysis.lfp_summary_") for name in imports
    )
