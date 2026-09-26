"""Ownership contracts for the LFP-summary launcher runtime split."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path


ROOT_MODULE = "src.neural_analysis.lfp_spike_phase_launcher"
RUNTIME_MODULE = "src.neural_analysis.lfp_summary.launcher_runtime"


def test_launcher_runtime_has_canonical_internal_owner() -> None:
    """Require one launcher-specific owner for the established helper groups."""
    root = importlib.import_module(ROOT_MODULE)
    runtime = importlib.import_module(RUNTIME_MODULE)

    for name in (
        "LauncherCommand",
        "RepositoryState",
        "LauncherDependencies",
        "LauncherRunResult",
        "make_production_launcher_dependencies",
        "_prepare_new_run",
        "_prepare_resume",
        "_prepare_report_recovery",
        "_validate_new_cache_preflight",
        "_validate_retained_work_root",
        "_execute_locked_run",
        "_cleanup_and_complete",
        "_persist_state",
        "_launcher_lock",
        "_LinuxProcessTreeSampler",
        "_launcher_signal_handlers",
    ):
        canonical = getattr(runtime, name)
        assert getattr(root, name) is canonical
        assert canonical.__module__ == RUNTIME_MODULE


def test_root_keeps_command_parsing_dispatch_and_direct_execution() -> None:
    """Keep the documented command's public control flow at its root module."""
    root = importlib.import_module(ROOT_MODULE)
    runtime = importlib.import_module(RUNTIME_MODULE)

    for name in ("parse_launcher_command", "run_launcher", "main"):
        root_callable = getattr(root, name)
        assert root_callable.__module__ == ROOT_MODULE
        assert not hasattr(runtime, name)


def test_canonical_launcher_runtime_does_not_import_root_entrypoint() -> None:
    """Prevent the internal owner from depending back on the command facade."""
    runtime = importlib.import_module(RUNTIME_MODULE)
    source_path = Path(runtime.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )

    assert ROOT_MODULE not in imported_modules
