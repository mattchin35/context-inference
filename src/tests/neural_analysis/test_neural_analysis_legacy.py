"""Structural contracts for retained exploratory neural-analysis modules."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


NEURAL_ROOT = Path(__file__).parents[2] / "neural_analysis"
LEGACY_ROOT = NEURAL_ROOT / "legacy"
LEGACY_MODULES = (
    "behavior_pynap",
    "spike_behavior_analysis",
    "spike_behavior_binning",
    "modified_sinc_smoother",
    "plot_single_session_analysis",
)


def test_legacy_package_is_non_eager() -> None:
    """Require a package marker that cannot trigger exploratory module I/O."""
    package_path = LEGACY_ROOT / "__init__.py"
    tree = ast.parse(package_path.read_text(encoding="utf-8"))

    executable_nodes = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    assert executable_nodes == []


@pytest.mark.parametrize("module_name", LEGACY_MODULES)
def test_legacy_module_has_thin_root_compatibility_entry(
    module_name: str,
) -> None:
    """Move source under ``legacy`` while preserving its old module path."""
    legacy_path = LEGACY_ROOT / f"{module_name}.py"
    root_path = NEURAL_ROOT / f"{module_name}.py"

    assert legacy_path.is_file()
    assert root_path.is_file()

    wrapper_source = root_path.read_text(encoding="utf-8")
    wrapper_tree = ast.parse(wrapper_source)
    assert len(wrapper_source.splitlines()) <= 20
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in wrapper_tree.body
    )
    assert f"src.neural_analysis.legacy.{module_name}" in wrapper_source
    assert "sys.modules[__name__]" in wrapper_source
    assert "runpy.run_module" in wrapper_source


def test_legacy_modules_keep_their_established_source_shapes() -> None:
    """Retain the exploratory callables and intentionally empty placeholder."""
    expected_definitions = {
        "behavior_pynap": {
            "_validate_event_df",
            "_ts_from_events",
            "prepare_trial_starts",
            "prepare_led_on",
            "prepare_led_off",
            "prepare_choices",
            "prepare_lick_times",
            "prepare_reward_deliveries",
            "build_behavior_pynapple",
            "main",
        },
        "spike_behavior_analysis": set(),
        "spike_behavior_binning": {
            "bin_spikes_to_trial",
            "bin_licks_to_trial",
            "decode_from_spikes",
            "shuffle_decode_only",
            "cv_decode_only",
            "plot_trial",
            "bin_spikes",
            "make_classifier_bins",
            "decode_with_cv",
        },
        "modified_sinc_smoother": {
            "cube",
            "fit_weighted",
            "extend_data",
            "edge_weights",
            "edge_weights1",
            "window_ms",
            "corr_coeffs_ms",
            "corr_coeffs_ms1",
            "kernel_ms",
            "kernel_ms1",
            "smooth_ms",
            "smooth_ms1",
        },
        "plot_single_session_analysis": set(),
    }

    for module_name, expected_names in expected_definitions.items():
        tree = ast.parse(
            (LEGACY_ROOT / f"{module_name}.py").read_text(encoding="utf-8")
        )
        actual_names = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert actual_names == expected_names
