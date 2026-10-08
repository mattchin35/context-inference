"""Inter-regional neural regression configuration and result contracts."""

from .configuration import (
    ANALYSIS_VERSION,
    CONFIG_SCHEMA_VERSION,
    COVERAGE_ASSUMPTION_VERSION,
    RESULT_SCHEMA_VERSION,
    AnalysisWindows,
    FilterConfig,
    InterregionalAnalysisConfig,
    PCAConfig,
    RegionalPopulationConfig,
    ResolvedRegionalPopulation,
    RunOptions,
    TemporalConfig,
    load_interregional_config,
)
from .records import FIT_STATUS_VALUES, UNAVAILABILITY_REASONS, InterregionalResults
from .persistence import load_interregional_result, save_interregional_result
from .pipeline import (
    PreparedInterregionalSession,
    prepare_interregional_session,
    run_linear_cross_validation,
)


__all__ = [
    "ANALYSIS_VERSION",
    "CONFIG_SCHEMA_VERSION",
    "COVERAGE_ASSUMPTION_VERSION",
    "FIT_STATUS_VALUES",
    "RESULT_SCHEMA_VERSION",
    "UNAVAILABILITY_REASONS",
    "AnalysisWindows",
    "FilterConfig",
    "InterregionalAnalysisConfig",
    "InterregionalResults",
    "PCAConfig",
    "PreparedInterregionalSession",
    "RegionalPopulationConfig",
    "ResolvedRegionalPopulation",
    "RunOptions",
    "TemporalConfig",
    "load_interregional_config",
    "load_interregional_result",
    "prepare_interregional_session",
    "run_linear_cross_validation",
    "save_interregional_result",
]
