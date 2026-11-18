"""Canonical domain-plotting ownership and compatibility contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_PLOTTING_SURFACES = (
    (
        "src.neural_analysis.lfp.plotting",
        (
            (
                "src.neural_analysis.lfp_phase_clustering",
                ("plot_phase_clustering",),
            ),
            (
                "src.neural_analysis.unit_spike_plotting",
                (
                    "break_wrapped_phase_trace",
                    "plot_trial_lfp_spectrogram_and_behavior",
                    "plot_trial_lfp_relative_phase_and_behavior",
                    "plot_trial_lfp_phase_analysis_and_behavior",
                ),
            ),
        ),
    ),
    (
        "src.neural_analysis.spike_behavior.plotting",
        (
            (
                "src.neural_analysis.spike_behavior_pynapple",
                ("plot_trial_raster",),
            ),
            (
                "src.neural_analysis.psth_behavior",
                (
                    "plot_lick_peth",
                    "plot_single_unit_spike_peth",
                    "save_figure_with_message",
                    "plot_peth",
                ),
            ),
            (
                "src.neural_analysis.unit_spike_plotting",
                (
                    "filter_trials_for_unit_plot",
                    "paginate_trial_indices",
                    "split_trial_indices_by_action",
                    "extract_relative_unit_spikes",
                    "compute_psth_hz",
                    "compute_binned_firing_rates_hz",
                    "plot_unit_binned_rate_trial_traces",
                    "plot_unit_binned_rate_mean_sd",
                    "compute_trial_event_offsets",
                    "paginate_unit_ids",
                    "get_trial_alignment_time",
                    "extract_relative_events_for_trial",
                    "extract_relative_spikes_for_units",
                    "compute_population_psth_hz",
                    "plot_trial_behavior_and_spike_raster",
                    "draw_single_trial_behavior_axis",
                    "plot_unit_raster_and_psth",
                    "plot_unit_left_right_choice_comparison",
                    "save_unit_plot_figure",
                ),
            ),
        ),
    ),
    (
        "src.neural_analysis.spike_lfp.plotting",
        (
            (
                "src.neural_analysis.unit_spike_plotting",
                (
                    "plot_spike_lfp_phase_locking",
                    "plot_trial_spike_lfp_hilbert_phase_and_behavior",
                ),
            ),
        ),
    ),
    (
        "src.neural_analysis.population.plotting",
        (
            (
                "src.neural_analysis.unit_spike_plotting",
                (
                    "plot_trial_behavior_and_population_pca",
                    "build_concatenated_trial_time_axis",
                    "plot_concatenated_trial_behavior_and_population_pca",
                    "plot_pca_cumulative_explained_variance",
                ),
            ),
            (
                "src.neural_analysis.population_pca_decoding",
                (
                    "plot_pca_decoding_pre_post_scores",
                    "plot_average_pca_scores_by_condition_and_target",
                ),
            ),
            (
                "src.neural_analysis.population_pca_switch_trajectories",
                ("plot_switch_event_pca_trajectories",),
            ),
            (
                "src.neural_analysis.plot_cross_session_analysis",
                (
                    "append_plot_title_suffix",
                    "plot_cross_session_decodability_scores",
                    "plot_cross_session_decodability_pvalues",
                    "plot_cross_session_decoder_accuracy",
                    "plot_cross_session_decoder_superplot",
                    "plot_cross_session_decoder_session_means",
                ),
            ),
        ),
    ),
)

_LEGACY_DIRECT_ENTRY_MODULES = (
    "src.neural_analysis.spike_behavior_pynapple",
    "src.neural_analysis.psth_behavior",
    "src.neural_analysis.plot_cross_session_analysis",
)


def _imported_module_names(module: ModuleType) -> set[str]:
    """Return absolute module names used by one plotting module."""

    module_path = Path(module.__file__ or "")
    syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_names: set[str] = set()
    for node in ast.walk(syntax_tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_names.add(node.module)
        elif isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
    return imported_names


def _has_direct_main_dispatch(module: ModuleType) -> bool:
    """Return whether a module retains an ``if __name__`` call to ``main``."""

    module_path = Path(module.__file__ or "")
    syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in syntax_tree.body:
        if not isinstance(node, ast.If):
            continue
        if not (
            isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            continue
        if any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "main"
            for child in ast.walk(node)
        ):
            return True
    return False


@pytest.mark.parametrize(
    "canonical_module_name,_legacy_surfaces",
    _PLOTTING_SURFACES,
)
def test_canonical_domain_plotting_modules_are_importable(
    canonical_module_name: str,
    _legacy_surfaces: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """Each approved domain plotting owner must be directly importable."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "canonical_module_name,legacy_surfaces",
    _PLOTTING_SURFACES,
)
def test_legacy_plotting_paths_forward_exact_canonical_objects(
    canonical_module_name: str,
    legacy_surfaces: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """Every established plotting import must expose its canonical callable."""

    canonical_module = importlib.import_module(canonical_module_name)
    for legacy_module_name, function_names in legacy_surfaces:
        legacy_module = importlib.import_module(legacy_module_name)
        for function_name in function_names:
            function = getattr(canonical_module, function_name)
            assert getattr(legacy_module, function_name) is function
            assert function.__module__ == canonical_module_name


@pytest.mark.parametrize(
    "canonical_module_name,_legacy_surfaces",
    _PLOTTING_SURFACES,
)
def test_domain_plotting_modules_do_not_depend_on_ui_workflows_or_legacy_mixtures(
    canonical_module_name: str,
    _legacy_surfaces: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    """Reusable plots may depend on science domains but not UI or workflows."""

    canonical_module = importlib.import_module(canonical_module_name)
    imported_names = _imported_module_names(canonical_module)

    assert imported_names.isdisjoint(
        {
            "src.neural_analysis.lfp_phase_clustering",
            "src.neural_analysis.spike_behavior_pynapple",
            "src.neural_analysis.psth_behavior",
            "src.neural_analysis.unit_spike_plotting",
            "src.neural_analysis.population_pca_decoding",
            "src.neural_analysis.population_pca_switch_trajectories",
            "src.neural_analysis.plot_cross_session_analysis",
            "src.neural_analysis.psth_webapp",
        }
    )
    assert not any(
        imported_name == "src.neural_analysis.lfp_summary"
        or imported_name.startswith("src.neural_analysis.lfp_summary_")
        for imported_name in imported_names
    )
    assert not hasattr(canonical_module, "main")


@pytest.mark.parametrize("legacy_module_name", _LEGACY_DIRECT_ENTRY_MODULES)
def test_example_entry_points_remain_at_legacy_paths(legacy_module_name: str) -> None:
    """R3 moves plots without relocating existing example execution policy."""

    legacy_module = importlib.import_module(legacy_module_name)

    assert legacy_module.main.__module__ == legacy_module_name
    assert _has_direct_main_dispatch(legacy_module)


def test_workflow_specific_summary_plotting_is_not_absorbed_by_domain_modules() -> None:
    """Production-summary figures remain owned by their workflow module in R3."""

    summary_plotting = importlib.import_module("src.neural_analysis.lfp_summary_plotting")

    assert summary_plotting.plot_condition_psd.__module__ == summary_plotting.__name__
    for canonical_module_name, _legacy_surfaces in _PLOTTING_SURFACES:
        canonical_module = importlib.import_module(canonical_module_name)
        assert not hasattr(canonical_module, "plot_condition_psd")
