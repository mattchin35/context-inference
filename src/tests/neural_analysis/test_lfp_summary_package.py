"""Canonical LFP-summary foundation and compatibility contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_MODULE_SURFACES = (
    (
        "src.neural_analysis.lfp_summary_models",
        "src.neural_analysis.lfp_summary.models",
        (
            "LFPSiteConfig",
            "UnitPopulationConfig",
            "TrialFilterConfig",
            "PhaseAnalysisConfig",
            "PPCAnalysisConfig",
            "PPCExecutionConfig",
            "LFPSummaryConfig",
            "ProgressEvent",
            "default_lfp_summary_config",
            "canonical_config_json",
            "lfp_summary_config_from_json",
            "validate_lfp_summary_config",
            "source_value_semantics",
            "component_fingerprint",
            "fingerprint_source_files",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_session",
        "src.neural_analysis.lfp_summary.session",
        (
            "LFPSummarySessionRequest",
            "build_active_unit_population",
            "build_metadata_spike_phase_config",
            "build_lfp_summary_config",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_preparation",
        "src.neural_analysis.lfp_summary.preparation",
        (
            "PreparedTrials",
            "PreparedSiteTraces",
            "TrialRelativeSpikeTrains",
            "build_prepared_trials",
            "build_common_event_grid",
            "interpolate_complex_coefficients_to_common_grid",
            "prepare_site_trial_traces",
            "load_site_trial_traces",
            "preflight_open_ephys_site_metadata",
            "validate_open_ephys_aligned_sync_path",
            "build_trial_relative_spike_trains",
            "validate_progress_events",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_pipeline",
        "src.neural_analysis.lfp_summary.pipeline",
        (
            "PipelineDependencies",
            "PPCWorkCleanupTarget",
            "ComponentPayload",
            "ComponentRunResult",
            "ComputeAllResult",
            "compute_power_component",
            "compute_synchrony_component",
            "compute_spike_phase_component",
            "compute_all_components",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_payloads",
        "src.neural_analysis.lfp_summary.payloads",
        ("ArrayContract", "build_component_payload", "validate_component_payload"),
    ),
    (
        "src.neural_analysis.lfp_summary_io",
        "src.neural_analysis.lfp_summary.cache",
        (
            "ComponentStatus",
            "load_or_initialize_manifest",
            "validate_component_npz_headers",
            "load_component_arrays",
            "write_component_transaction",
            "assess_component_status",
            "rebind_power_synchrony_manifest",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_work_cache",
        "src.neural_analysis.lfp_summary.work_cache",
        (
            "PreparedPhaseCache",
            "PPCCheckpoint",
            "recover_stale_lock",
            "write_prepared_phase_cache",
            "load_prepared_phase_cache",
            "write_ppc_checkpoint",
            "load_valid_ppc_checkpoint",
            "load_valid_ppc_checkpoints",
            "cleanup_ppc_run",
        ),
    ),
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


@pytest.mark.parametrize(
    "_legacy_module_name, canonical_module_name, _public_symbols",
    _MODULE_SURFACES,
)
def test_canonical_lfp_summary_foundations_are_importable(
    _legacy_module_name: str,
    canonical_module_name: str,
    _public_symbols: tuple[str, ...],
) -> None:
    """Each approved workflow foundation has one directly importable owner."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, public_symbols",
    _MODULE_SURFACES,
)
def test_legacy_foundation_modules_alias_the_canonical_module(
    legacy_module_name: str,
    canonical_module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Old paths preserve module state, monkeypatch seams, and public objects."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    assert legacy_module is canonical_module
    for symbol_name in public_symbols:
        symbol = getattr(canonical_module, symbol_name)
        assert getattr(legacy_module, symbol_name) is symbol
        assert symbol.__module__ == canonical_module_name


@pytest.mark.parametrize(
    "_legacy_module_name, canonical_module_name, _public_symbols",
    _MODULE_SURFACES,
)
def test_canonical_foundations_do_not_back_import_legacy_paths(
    _legacy_module_name: str,
    canonical_module_name: str,
    _public_symbols: tuple[str, ...],
) -> None:
    """Canonical foundations depend on canonical workflow owners only."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert not any(
        name.startswith("src.neural_analysis.lfp_summary_")
        for name in _imported_module_names(canonical_module)
    )


def test_canonical_models_preserve_config_json_and_component_fingerprints() -> None:
    """Moving config owners must not alter existing persisted identities."""

    models = importlib.import_module("src.neural_analysis.lfp_summary.models")
    lfp_config = importlib.import_module("src.neural_analysis.lfp.config")

    assert models.AnalysisWindowConfig is lfp_config.AnalysisWindowConfig
    assert models.FrequencyBandConfig is lfp_config.FrequencyBandConfig
    assert models.PowerAnalysisConfig is lfp_config.PowerAnalysisConfig

    config = models.default_lfp_summary_config()
    encoded = models.canonical_config_json(config)
    restored = models.lfp_summary_config_from_json(encoded)

    assert models.canonical_config_json(restored) == encoded
    for component_name in ("power", "synchrony", "spike_phase"):
        assert models.component_fingerprint(
            component_name, restored
        ) == models.component_fingerprint(component_name, config)


def test_canonical_cache_preserves_historical_manifest_generator(tmp_path: Path) -> None:
    """The module move must not rewrite the persisted producer identity."""

    cache = importlib.import_module("src.neural_analysis.lfp_summary.cache")
    models = importlib.import_module("src.neural_analysis.lfp_summary.models")

    manifest = cache.load_or_initialize_manifest(
        tmp_path,
        models.default_lfp_summary_config(),
    )

    assert manifest["generator"] == {"module": "src.neural_analysis.lfp_summary_io"}


def test_canonical_session_avoids_derived_version_one_metadata_views() -> None:
    """The version-2 adapter reads probes and cache fields without v1 reconstruction."""

    session_module = importlib.import_module("src.neural_analysis.lfp_summary.session")
    module_path = Path(session_module.__file__ or "")
    syntax_tree = ast.parse(module_path.read_text(encoding="utf-8"))
    accessed_attributes = {
        node.attr for node in ast.walk(syntax_tree) if isinstance(node, ast.Attribute)
    }

    assert accessed_attributes.isdisjoint(
        {"populations", "channel_groups", "lfp_summary_cache_directory"}
    )
