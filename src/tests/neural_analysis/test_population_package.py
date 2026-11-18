"""Canonical population-analysis ownership and compatibility contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_MODULE_SURFACES = (
    (
        "src.neural_analysis.population_pca",
        "src.neural_analysis.population.pca",
        (
            "PopulationPCAResult",
            "PopulationPCAProfile",
            "build_trial_unit_rate_tensor",
            "build_trial_unit_rate_tensor_numpy",
            "build_trial_unit_rate_tensor_pynapple",
            "prepare_pca_observation_matrix",
            "fit_population_pca",
            "profile_population_pca_pipeline",
        ),
    ),
    (
        "src.neural_analysis.population_pca_decoding",
        "src.neural_analysis.population.decoding",
        (
            "build_choice_aligned_rate_tensor",
            "select_pca_decoding_trial_indices",
            "build_valid_base_condition_masks",
            "build_exploratory_pca_decoder_trial_bins",
            "run_exploratory_pca_choice_decoding",
            "run_rigorous_pca_choice_decoding",
            "cv_decodeability_score_with_pca_pipeline",
            "summarize_pca_decoding_results",
            "summarize_pca_scores_by_condition_and_target",
            "extract_pca_score_points_by_condition_and_target",
            "format_pca_score_target_label",
        ),
    ),
    (
        "src.neural_analysis.population_pca_switch_trajectories",
        "src.neural_analysis.population.switch_trajectories",
        (
            "select_valid_choice_trial_indices",
            "select_choice_switch_events",
            "extract_switch_event_pca_trajectories",
            "summarize_switch_event_counts",
        ),
    ),
    (
        "src.neural_analysis.plot_cross_session_analysis",
        "src.neural_analysis.population.cross_session",
        (
            "normalize_region_name_for_filename",
            "make_region_analysis_filename",
            "make_analysis_filename_tag",
            "make_analysis_title_suffix",
            "find_session_analysis_csvs",
            "load_state_decodability_sessions",
            "load_state_decoder_performance_sessions",
            "save_cross_session_tables",
            "filter_table_by_date_selection",
            "select_cross_session_date_range",
            "make_output_filename",
        ),
    ),
)

_CANONICAL_MODULE_NAMES = tuple(canonical for _, canonical, _ in _MODULE_SURFACES)

_CONSTANT_SURFACES = (
    (
        "src.neural_analysis.population_pca",
        "src.neural_analysis.population.pca",
        (
            "PCA_NORMALIZATION_ZSCORE",
            "PCA_NORMALIZATION_MEAN_CENTER",
            "PCA_NORMALIZATION_OPTIONS",
            "PCA_BINNING_METHOD_NUMPY",
            "PCA_BINNING_METHOD_PYNAPPLE",
            "PCA_BINNING_METHOD_OPTIONS",
        ),
    ),
    (
        "src.neural_analysis.population_pca_decoding",
        "src.neural_analysis.population.decoding",
        (
            "PCA_DECODING_MODE_EXPLORATORY",
            "PCA_DECODING_MODE_RIGOROUS",
            "PCA_DECODING_MODE_OPTIONS",
            "PCA_DECODING_BASE_CONDITIONS",
            "PCA_DECODING_WINDOWS",
            "PCA_DECODING_BIN_SIZE_S",
            "PCA_DECODING_WINDOW",
            "PCA_DECODING_DISPLAY_COLUMNS",
            "PCA_SCORE_SUMMARY_COLUMNS",
            "PCA_RAW_SCORE_COLUMNS",
        ),
    ),
    (
        "src.neural_analysis.population_pca_switch_trajectories",
        "src.neural_analysis.population.switch_trajectories",
        (
            "SWITCH_PRE_FILTER_ALL",
            "SWITCH_PRE_FILTER_CORRECT_REWARDED",
            "SWITCH_PRE_FILTER_OMISSION",
            "SWITCH_PRE_FILTER_OPTIONS",
            "SWITCH_TYPE_ORDER",
            "SWITCH_TYPE_TITLES",
            "SWITCH_DIRECTION_ORDER",
            "SWITCH_DIRECTION_TITLES",
            "SWITCH_EVENT_COLUMNS",
            "SWITCH_TRAJECTORY_COLUMNS",
        ),
    ),
    (
        "src.neural_analysis.plot_cross_session_analysis",
        "src.neural_analysis.population.cross_session",
        ("SUPPORTED_ANALYSIS_REGIONS",),
    ),
)

_PLOTTING_FUNCTIONS = {
    "src.neural_analysis.population_pca_decoding": (
        "plot_pca_decoding_pre_post_scores",
        "plot_average_pca_scores_by_condition_and_target",
    ),
    "src.neural_analysis.population_pca_switch_trajectories": (
        "plot_switch_event_pca_trajectories",
    ),
    "src.neural_analysis.plot_cross_session_analysis": (
        "append_plot_title_suffix",
        "plot_cross_session_decodability_scores",
        "plot_cross_session_decodability_pvalues",
        "plot_cross_session_decoder_accuracy",
        "plot_cross_session_decoder_superplot",
        "plot_cross_session_decoder_session_means",
    ),
    "src.neural_analysis.unit_spike_plotting": (
        "plot_trial_behavior_and_population_pca",
        "plot_concatenated_trial_behavior_and_population_pca",
        "plot_pca_cumulative_explained_variance",
    ),
}


def _imported_module_names(module: ModuleType) -> set[str]:
    """Return absolute module names used by one canonical source module."""

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


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_population_modules_are_importable(canonical_module_name: str) -> None:
    """Each approved population-analysis owner must be directly importable."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, moved_symbols",
    _MODULE_SURFACES,
)
def test_legacy_population_paths_forward_exact_moved_objects(
    legacy_module_name: str,
    canonical_module_name: str,
    moved_symbols: tuple[str, ...],
) -> None:
    """Legacy modules must expose the exact objects owned by canonical modules."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    for symbol_name in moved_symbols:
        symbol = getattr(canonical_module, symbol_name)
        assert getattr(legacy_module, symbol_name) is symbol
        assert symbol.__module__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, constant_names",
    _CONSTANT_SURFACES,
)
def test_legacy_population_paths_share_canonical_scientific_and_table_constants(
    legacy_module_name: str,
    canonical_module_name: str,
    constant_names: tuple[str, ...],
) -> None:
    """Scientific choices and saved-table schemas must retain one exact owner."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    for constant_name in constant_names:
        assert getattr(legacy_module, constant_name) is getattr(
            canonical_module,
            constant_name,
        )


def test_pca_legacy_module_alias_preserves_internal_monkeypatch_seams() -> None:
    """PCA profiling patches through the old path must reach canonical globals."""

    legacy_pca = importlib.import_module("src.neural_analysis.population_pca")
    canonical_pca = importlib.import_module("src.neural_analysis.population.pca")

    assert legacy_pca is canonical_pca


def test_population_callers_keep_canonical_and_mixed_legacy_module_seams() -> None:
    """The webapp uses canonical PCA while mixed plotting facades stay intact."""

    canonical_pca = importlib.import_module("src.neural_analysis.population.pca")
    canonical_decoding = importlib.import_module("src.neural_analysis.population.decoding")
    legacy_decoding = importlib.import_module("src.neural_analysis.population_pca_decoding")
    legacy_switch = importlib.import_module(
        "src.neural_analysis.population_pca_switch_trajectories"
    )
    webapp = importlib.import_module("src.neural_analysis.psth_webapp")

    assert canonical_decoding.population_pca is canonical_pca
    assert webapp.population_pca is canonical_pca
    assert webapp.population_pca_decoding is legacy_decoding
    assert webapp.population_pca_switch_trajectories is legacy_switch


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_population_modules_do_not_import_legacy_workflow_or_ui_modules(
    canonical_module_name: str,
) -> None:
    """Reusable population modules must not depend on legacy mixtures or UI."""

    canonical_module = importlib.import_module(canonical_module_name)
    imported_names = _imported_module_names(canonical_module)

    assert "matplotlib.pyplot" not in imported_names
    assert imported_names.isdisjoint(
        {
            "src.neural_analysis.population_pca",
            "src.neural_analysis.population_pca_decoding",
            "src.neural_analysis.population_pca_switch_trajectories",
            "src.neural_analysis.plot_cross_session_analysis",
            "src.neural_analysis.psth_webapp",
            "src.neural_analysis.unit_spike_plotting",
        }
    )
    assert not any(
        imported_name == "src.neural_analysis.lfp_summary"
        or imported_name.startswith("src.neural_analysis.lfp_summary_")
        for imported_name in imported_names
    )


def test_population_plotting_has_one_owner_and_cross_session_dispatch_remains_legacy() -> None:
    """R3 owns population plots without moving cross-session execution policy."""

    canonical_plotting = importlib.import_module("src.neural_analysis.population.plotting")
    for legacy_module_name, function_names in _PLOTTING_FUNCTIONS.items():
        legacy_module = importlib.import_module(legacy_module_name)
        for function_name in function_names:
            assert getattr(legacy_module, function_name) is getattr(
                canonical_plotting,
                function_name,
            )

    for canonical_module_name in _CANONICAL_MODULE_NAMES:
        canonical_module = importlib.import_module(canonical_module_name)
        assert not hasattr(canonical_module, "main")
        assert not any(name.startswith("plot_") for name in vars(canonical_module))
        assert not hasattr(canonical_module, "append_plot_title_suffix")

    legacy_cross_session = importlib.import_module(
        "src.neural_analysis.plot_cross_session_analysis"
    )
    assert legacy_cross_session.main.__module__ == legacy_cross_session.__name__
    assert _has_direct_main_dispatch(legacy_cross_session)
