"""Canonical PPC planning and execution ownership contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType


_PLANNING_PUBLIC = (
    "PPCJobPlan",
    "PPCAllocationEstimate",
    "PPCComponentPlan",
    "estimate_grouped_ppc_allocation",
    "plan_grouped_ppc_component",
)
_EXECUTION_PUBLIC = (
    "PPCExecutionResult",
    "PPCComponentExecutionResult",
    "execute_ppc_blocks",
    "execute_grouped_ppc_component",
)
_SPAWN_OBJECTS = (
    "_PhaseWorkDescriptor",
    "_GroupedParallelBlockTask",
    "_GroupedParallelBlockResult",
    "_ParallelBlockTask",
    "_ParallelBlockResult",
    "_initialize_grouped_parallel_worker",
    "_compute_grouped_parallel_block",
    "_compute_parallel_block",
)


def _imported_module_names(module: ModuleType) -> set[str]:
    """Return absolute names used by one module's import statements."""

    module_path = Path(module.__file__ or "")
    syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    return imported_names


def test_ppc_planning_and_execution_modules_are_importable() -> None:
    """The single reviewed planner/executor boundary has direct imports."""

    planning = importlib.import_module("src.neural_analysis.lfp_summary.ppc_planning")
    execution = importlib.import_module("src.neural_analysis.lfp_summary.ppc_execution")

    assert planning.__name__ == "src.neural_analysis.lfp_summary.ppc_planning"
    assert execution.__name__ == "src.neural_analysis.lfp_summary.ppc_execution"


def test_ppc_public_objects_have_one_canonical_owner() -> None:
    """Plans belong to planning while results and executors belong to execution."""

    planning = importlib.import_module("src.neural_analysis.lfp_summary.ppc_planning")
    execution = importlib.import_module("src.neural_analysis.lfp_summary.ppc_execution")

    for symbol_name in _PLANNING_PUBLIC:
        assert getattr(planning, symbol_name).__module__ == planning.__name__
        assert getattr(execution, symbol_name) is getattr(planning, symbol_name)
    for symbol_name in _EXECUTION_PUBLIC:
        assert getattr(execution, symbol_name).__module__ == execution.__name__


def test_legacy_ppc_runtime_aliases_canonical_execution() -> None:
    """The old module keeps exact state and monkeypatch reach-through."""

    legacy = importlib.import_module("src.neural_analysis.lfp_summary_ppc_runtime")
    execution = importlib.import_module("src.neural_analysis.lfp_summary.ppc_execution")

    assert legacy is execution


def test_planning_has_no_execution_or_checkpoint_dependency() -> None:
    """Pure planning remains independent of processes, locks, and work caches."""

    planning = importlib.import_module("src.neural_analysis.lfp_summary.ppc_planning")
    imported_names = _imported_module_names(planning)

    assert "src.neural_analysis.lfp_summary.ppc_execution" not in imported_names
    assert "src.neural_analysis.lfp_summary.work_cache" not in imported_names
    assert "multiprocessing" not in imported_names
    assert "concurrent.futures" not in imported_names


def test_execution_imports_the_canonical_planning_owner() -> None:
    """Execution consumes immutable plans without copying planning logic."""

    execution = importlib.import_module("src.neural_analysis.lfp_summary.ppc_execution")

    assert (
        "src.neural_analysis.lfp_summary.ppc_planning"
        in _imported_module_names(execution)
    )


def test_spawn_workers_and_records_are_top_level_execution_objects() -> None:
    """Process-spawn targets retain canonical importable module identities."""

    execution = importlib.import_module("src.neural_analysis.lfp_summary.ppc_execution")

    for symbol_name in _SPAWN_OBJECTS:
        value = getattr(execution, symbol_name)
        assert value.__module__ == execution.__name__
        assert "<locals>" not in value.__qualname__


def test_grouped_and_legacy_executors_both_remain_available() -> None:
    """R4C organizes both generations without deleting either implementation."""

    execution = importlib.import_module("src.neural_analysis.lfp_summary.ppc_execution")

    assert callable(execution.execute_grouped_ppc_component)
    assert callable(execution.execute_ppc_blocks)
