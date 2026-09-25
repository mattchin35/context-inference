"""Canonical Spike-LFP ownership, compatibility, and caller contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from types import ModuleType

import pytest


_MODULE_SURFACES = (
    (
        "src.neural_analysis.spike_lfp_hilbert_phase",
        "src.neural_analysis.spike_lfp.hilbert",
        (
            "HilbertPhaseTrace",
            "SpikeHilbertPhaseSamples",
            "SingleTrialSpikeLFPHilbertResult",
            "compute_hilbert_phase_trace",
            "sample_hilbert_phase_at_spikes",
            "compute_single_trial_spike_lfp_hilbert",
            "save_single_trial_spike_lfp_hilbert_result",
        ),
    ),
    (
        "src.neural_analysis.spike_lfp_phase_locking",
        "src.neural_analysis.spike_lfp.phase_locking",
        (
            "SpikePhaseSamples",
            "SpikePhaseLockingResult",
            "PhaseRateSparsityAssessment",
            "assess_phase_rate_sparsity",
            "merge_trial_windows",
            "sample_wavelet_phase_at_spikes",
            "bin_phase_spike_counts",
            "compute_phase_occupancy",
            "compute_phase_firing_rate_hz",
            "compute_frequency_phase_metrics",
            "compute_trial_aligned_spike_phase_locking",
            "save_spike_lfp_phase_locking_result",
        ),
    ),
    (
        "src.neural_analysis.spike_lfp_summary",
        "src.neural_analysis.spike_lfp.ppc",
        (
            "ObservedPPCResult",
            "BandPPCSummary",
            "PopulationPPCSummary",
            "RepresentativePhaseHistograms",
            "PPCExemplarSelection",
            "PermutationNullSummary",
            "TrialShufflePPCResult",
            "EdgeSufficientStatistics",
            "SignificantPrevalenceSummary",
            "build_probe_qualified_unit_ids",
            "compute_observed_ppc",
            "compute_band_ppc_means",
            "build_population_ppc_summary",
            "build_representative_phase_histograms",
            "select_ppc_exemplars",
            "generate_trial_derangement_schedule",
            "compute_edge_sufficient_statistics",
            "reduce_scheduled_shuffle_block",
            "compute_trial_shuffle_ppc",
            "summarize_permutation_null",
            "permutation_null_summary_to_arrays",
            "permutation_null_summary_from_arrays",
            "adjust_ppc_pvalues_bh",
            "build_shuffle_run_metadata",
            "compute_significant_prevalence",
        ),
    ),
    (
        "src.neural_analysis.lfp_summary_ppc_kernel",
        "src.neural_analysis.spike_lfp.ppc_kernel",
        (
            "SourceTrialSpikeGeometry",
            "SegmentedEdgeStatistics",
            "ObservedTrialSegmentedPPCStatistics",
            "ObservedSegmentedPPCStatistics",
            "ObservedSegmentedPPCMetrics",
            "KernelAllocationEstimate",
            "build_source_trial_spike_geometry",
            "compute_segmented_edge_statistics",
            "compute_observed_trial_segmented_ppc_statistics",
            "compute_selected_observed_trial_segmented_ppc_statistics",
            "aggregate_observed_trial_segmented_ppc_statistics",
            "compute_observed_segmented_ppc_statistics",
            "compose_observed_segmented_ppc_metrics",
            "reduce_segmented_schedule_to_ppc",
            "estimate_segmented_kernel_allocation",
        ),
    ),
)

_CANONICAL_MODULE_NAMES = tuple(canonical for _, canonical, _ in _MODULE_SURFACES)

_PPC_RUNTIME_KERNEL_SYMBOLS = (
    "aggregate_observed_trial_segmented_ppc_statistics",
    "build_source_trial_spike_geometry",
    "compose_observed_segmented_ppc_metrics",
    "compute_observed_trial_segmented_ppc_statistics",
    "compute_selected_observed_trial_segmented_ppc_statistics",
    "compute_segmented_edge_statistics",
    "estimate_segmented_kernel_allocation",
    "reduce_segmented_schedule_to_ppc",
)


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


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_spike_lfp_modules_are_importable(canonical_module_name: str) -> None:
    """Each approved numerical Spike-LFP owner must be directly importable."""

    canonical_module = importlib.import_module(canonical_module_name)

    assert canonical_module.__name__ == canonical_module_name


@pytest.mark.parametrize(
    "legacy_module_name, canonical_module_name, public_symbols",
    _MODULE_SURFACES,
)
def test_legacy_spike_lfp_paths_alias_canonical_module_state(
    legacy_module_name: str,
    canonical_module_name: str,
    public_symbols: tuple[str, ...],
) -> None:
    """Old imports and monkeypatches must reach the exact canonical module state."""

    legacy_module = importlib.import_module(legacy_module_name)
    canonical_module = importlib.import_module(canonical_module_name)

    assert legacy_module is canonical_module
    for symbol_name in public_symbols:
        symbol = getattr(canonical_module, symbol_name)
        assert getattr(legacy_module, symbol_name) is symbol
        assert symbol.__module__ == canonical_module_name


def test_scientific_constants_and_persistence_versions_are_unchanged() -> None:
    """Canonical moves must retain established bands, thresholds, and versions."""

    hilbert = importlib.import_module("src.neural_analysis.spike_lfp.hilbert")
    phase_locking = importlib.import_module("src.neural_analysis.spike_lfp.phase_locking")
    ppc = importlib.import_module("src.neural_analysis.spike_lfp.ppc")

    assert hilbert.ANALYSIS_VERSION == "0.1.0"
    assert hilbert.DEFAULT_PHASE_BAND_HZ == (6.0, 10.0)
    assert hilbert.DEFAULT_FILTER_PADDING_S == 1.0
    assert phase_locking.ANALYSIS_VERSION == "0.2.0"
    assert ppc.THETA_FREQUENCIES_HZ == (6.0, 8.0, 10.0)
    assert ppc.GAMMA_FREQUENCIES_HZ == tuple(
        float(value) for value in range(30, 81, 2) if value not in {58, 60, 62}
    )
    assert ppc.MINIMUM_COMPUTABLE_SPIKES == 2
    assert ppc.MINIMUM_RELIABLE_SPIKES == 50


def test_canonical_spike_lfp_modules_use_canonical_scientific_dependencies() -> None:
    """Moved calculations must depend on the narrow LFP and Spike-LFP owners."""

    lfp_loading = importlib.import_module("src.neural_analysis.lfp.loading")
    lfp_phase = importlib.import_module("src.neural_analysis.lfp.phase")
    hilbert = importlib.import_module("src.neural_analysis.spike_lfp.hilbert")
    phase_locking = importlib.import_module("src.neural_analysis.spike_lfp.phase_locking")
    ppc = importlib.import_module("src.neural_analysis.spike_lfp.ppc")

    assert hilbert.lfp_loading is lfp_loading
    assert phase_locking.lfp_phase_clustering is lfp_phase
    assert ppc.spike_lfp_phase_locking is phase_locking


def test_ppc_cross_runtime_private_helpers_have_one_canonical_owner() -> None:
    """Tested schedule helpers shared with the runtime remain exact objects."""

    legacy_ppc = importlib.import_module("src.neural_analysis.spike_lfp_summary")
    canonical_ppc = importlib.import_module("src.neural_analysis.spike_lfp.ppc")
    ppc_runtime = importlib.import_module("src.neural_analysis.lfp_summary_ppc_runtime")

    for helper_name in (
        "_derangement_schedule",
        "_scheduled_trial_edges",
        "_compute_scheduled_shuffle_draws",
    ):
        helper = getattr(canonical_ppc, helper_name)
        assert getattr(legacy_ppc, helper_name) is helper
        assert helper.__module__ == canonical_ppc.__name__

    assert ppc_runtime._compute_scheduled_shuffle_draws is (
        canonical_ppc._compute_scheduled_shuffle_draws
    )
    assert ppc_runtime.generate_trial_derangement_schedule is (
        canonical_ppc.generate_trial_derangement_schedule
    )


def test_workflow_and_webapp_callers_reference_canonical_module_state() -> None:
    """Existing caller-local names must preserve patches while using new owners."""

    hilbert = importlib.import_module("src.neural_analysis.spike_lfp.hilbert")
    phase_locking = importlib.import_module("src.neural_analysis.spike_lfp.phase_locking")
    ppc = importlib.import_module("src.neural_analysis.spike_lfp.ppc")
    ppc_kernel = importlib.import_module("src.neural_analysis.spike_lfp.ppc_kernel")
    summary_runtime = importlib.import_module("src.neural_analysis.lfp_summary_runtime")
    ppc_runtime = importlib.import_module("src.neural_analysis.lfp_summary_ppc_runtime")
    webapp = importlib.import_module("src.neural_analysis.psth_webapp")

    assert summary_runtime.spike_lfp_hilbert_phase is hilbert
    assert summary_runtime.spike_lfp_summary is ppc
    assert ppc_runtime.spike_lfp_summary is ppc
    assert webapp.spike_lfp_hilbert_phase is hilbert
    assert webapp.spike_lfp_phase_locking is phase_locking
    for symbol_name in _PPC_RUNTIME_KERNEL_SYMBOLS:
        assert getattr(ppc_runtime, symbol_name) is getattr(ppc_kernel, symbol_name)


@pytest.mark.parametrize("canonical_module_name", _CANONICAL_MODULE_NAMES)
def test_canonical_spike_lfp_modules_do_not_import_workflows_ui_or_legacy_owners(
    canonical_module_name: str,
) -> None:
    """Reusable numerical modules must not depend on workflow, UI, or old owners."""

    canonical_module = importlib.import_module(canonical_module_name)
    imported_names = _imported_module_names(canonical_module)

    assert "matplotlib.pyplot" not in imported_names
    assert imported_names.isdisjoint(
        {
            "src.neural_analysis.spike_lfp_hilbert_phase",
            "src.neural_analysis.spike_lfp_phase_locking",
            "src.neural_analysis.spike_lfp_summary",
            "src.neural_analysis.lfp_summary_ppc_kernel",
            "src.neural_analysis.psth_webapp",
        }
    )
    assert not any(
        imported_name == "src.neural_analysis.lfp_summary"
        or imported_name.startswith("src.neural_analysis.lfp_summary_")
        for imported_name in imported_names
    )


def test_spike_lfp_plotting_remains_at_the_legacy_plot_owner() -> None:
    """R2C moves calculations while R3 retains the current plotting owner."""

    unit_plotting = importlib.import_module("src.neural_analysis.unit_spike_plotting")
    for plot_name in (
        "plot_spike_lfp_phase_locking",
        "plot_trial_spike_lfp_hilbert_phase_and_behavior",
    ):
        assert getattr(unit_plotting, plot_name).__module__ == unit_plotting.__name__

    for canonical_module_name in _CANONICAL_MODULE_NAMES:
        canonical_module = importlib.import_module(canonical_module_name)
        assert not hasattr(canonical_module, "plot_spike_lfp_phase_locking")
        assert not hasattr(
            canonical_module,
            "plot_trial_spike_lfp_hilbert_phase_and_behavior",
        )

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.neural_analysis.spike_lfp.plotting")
