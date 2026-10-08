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
    "RegionalPopulationConfig",
    "ResolvedRegionalPopulation",
    "RunOptions",
    "TemporalConfig",
    "load_interregional_config",
]
