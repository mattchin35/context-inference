"""Immutable configuration contracts for inter-regional neural regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import math
from numbers import Integral
import os
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "1"
RESULT_SCHEMA_VERSION = "1"
ANALYSIS_VERSION = "interregional-regression-v1"
COVERAGE_ASSUMPTION_VERSION = "implicit-complete-v1"
N_CV_FOLDS = 5
CV_GROUP_COLUMN = "cur_block"
ALLOWED_BIN_SIZES_S = (0.5, 0.1, 0.05, 0.02)
CANONICAL_CONDITIONS = (
    "all",
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
    "omission_switch",
    "omission_stay",
    "incorrect_switch",
    "incorrect_stay",
)
DEFAULT_CONDITIONS = (
    "all",
    "correct_rewarded",
    "incorrect",
    "omission",
    "switch",
    "stay",
)

_ALLOWED_CHANNEL_SOURCES = ("metadata_quality", "explicit")
_ALLOWED_UNIT_QUALITY_COLUMNS = ("group", "KSLabel")
_ALLOWED_ALIGNMENTS = ("choice_time", "start_time")
_ALLOWED_SIDES = ("all", "left", "right")
_CANONICAL_WINDOWS = ("before", "after", "whole")
_CANONICAL_REPRESENTATIONS = ("units", "pcs")
_CANONICAL_ANALYSES = ("ols_cv", "poisson_cv", "linear_granger", "poisson_granger")


def _require_nonempty_string(value: object, name: str) -> str:
    """Return one stripped identifier or raise a configuration error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value.strip()


def _normalize_integer_tuple(
    values: Sequence[object], name: str, *, allow_empty: bool
) -> tuple[int, ...]:
    """Return sorted unique nonnegative integers from one one-dimensional sequence."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of nonnegative integers.")
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must contain nonnegative integers.")
        normalized.append(int(value))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates.")
    result = tuple(sorted(normalized))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _normalize_label_tuple(
    values: Sequence[object], name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    """Return sorted unique nonempty string labels."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings.")
    labels = tuple(_require_nonempty_string(value, name) for value in values)
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must not contain duplicates.")
    result = tuple(sorted(labels))
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty.")
    return result


def _normalize_ordered_selection(
    values: Sequence[object], allowed: tuple[str, ...], name: str
) -> tuple[str, ...]:
    """Validate a unique nonempty selection and return canonical order."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence.")
    selected = tuple(values)
    if not selected:
        raise ValueError(f"{name} must not be empty.")
    if any(not isinstance(value, str) or value not in allowed for value in selected):
        raise ValueError(f"{name} contains an unsupported value.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"{name} must not contain duplicates.")
    return tuple(value for value in allowed if value in selected)


def _positive_integer(value: object, name: str) -> int:
    """Return a positive non-Boolean integer."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _finite_float(value: object, name: str) -> float:
    """Return a finite floating-point value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _normalized_absolute_path(value: object, name: str) -> Path:
    """Return an absolute lexically normalized path without touching the filesystem."""
    if not isinstance(value, (str, os.PathLike)) or isinstance(value, bytes):
        raise ValueError(f"{name} must be an absolute normalized path.")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute normalized path.")
    normalized = Path(os.path.normpath(os.fspath(path)))
    if path != normalized:
        raise ValueError(f"{name} must be an absolute normalized path.")
    return normalized


@dataclass(frozen=True)
class RegionalPopulationConfig:
    """Unresolved selection for one anatomical role.

    Parameters are identifiers or zero-based saved-channel indices and have no
    physical units. This record deliberately contains no cluster identifiers.
    """

    role: str
    probe_id: str
    channel_source: str = "metadata_quality"
    selected_channels: tuple[int, ...] = ()
    require_inside_brain: bool = True
    channel_quality_labels: tuple[str, ...] = ("good",)
    unit_quality_column: str = "group"
    unit_quality_labels: tuple[str, ...] = ("good", "mua")

    def __post_init__(self) -> None:
        """Normalize immutable selection values and validate their contract."""
        if self.role not in {"PFC", "HPC"}:
            raise ValueError("role must be 'PFC' or 'HPC'.")
        object.__setattr__(self, "probe_id", _require_nonempty_string(self.probe_id, "probe_id"))
        if self.channel_source not in _ALLOWED_CHANNEL_SOURCES:
            raise ValueError("channel_source must be 'metadata_quality' or 'explicit'.")
        channels = _normalize_integer_tuple(
            self.selected_channels,
            "selected_channels",
            allow_empty=self.channel_source == "metadata_quality",
        )
        if self.channel_source == "metadata_quality" and channels:
            raise ValueError("selected_channels must be empty for metadata_quality selection.")
        object.__setattr__(self, "selected_channels", channels)
        if not isinstance(self.require_inside_brain, bool):
            raise ValueError("require_inside_brain must be Boolean.")
        object.__setattr__(
            self,
            "channel_quality_labels",
            _normalize_label_tuple(self.channel_quality_labels, "channel_quality_labels"),
        )
        if self.unit_quality_column not in _ALLOWED_UNIT_QUALITY_COLUMNS:
            raise ValueError("unit_quality_column must be 'group' or 'KSLabel'.")
        object.__setattr__(
            self,
            "unit_quality_labels",
            _normalize_label_tuple(self.unit_quality_labels, "unit_quality_labels"),
        )


@dataclass(frozen=True)
class ResolvedRegionalPopulation:
    """Resolved channels and qualified units for one anatomical role.

    ``selected_channels`` and ``cluster_ids`` are zero-based/integer identifiers,
    not physical measurements. ``unit_ids`` has the same length and order as
    ``cluster_ids``.
    """

    role: str
    probe_id: str
    channel_source: str = "metadata_quality"
    selected_channels: tuple[int, ...] = ()
    require_inside_brain: bool = True
    channel_quality_labels: tuple[str, ...] = ("good",)
    unit_quality_column: str = "group"
    unit_quality_labels: tuple[str, ...] = ("good", "mua")
    cluster_ids: tuple[int, ...] = ()
    unit_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate resolved ordering and one-to-one cluster/unit identity."""
        if self.role not in {"PFC", "HPC"}:
            raise ValueError("role must be 'PFC' or 'HPC'.")
        object.__setattr__(self, "probe_id", _require_nonempty_string(self.probe_id, "probe_id"))
        if self.channel_source not in _ALLOWED_CHANNEL_SOURCES:
            raise ValueError("channel_source must be 'metadata_quality' or 'explicit'.")
        object.__setattr__(
            self,
            "selected_channels",
            _normalize_integer_tuple(
                self.selected_channels, "selected_channels", allow_empty=False
            ),
        )
        if not isinstance(self.require_inside_brain, bool):
            raise ValueError("require_inside_brain must be Boolean.")
        object.__setattr__(
            self,
            "channel_quality_labels",
            _normalize_label_tuple(self.channel_quality_labels, "channel_quality_labels"),
        )
        if self.unit_quality_column not in _ALLOWED_UNIT_QUALITY_COLUMNS:
            raise ValueError("unit_quality_column must be 'group' or 'KSLabel'.")
        object.__setattr__(
            self,
            "unit_quality_labels",
            _normalize_label_tuple(self.unit_quality_labels, "unit_quality_labels"),
        )
        cluster_ids = _normalize_integer_tuple(
            self.cluster_ids, "cluster_ids", allow_empty=False
        )
        if tuple(self.cluster_ids) != cluster_ids:
            raise ValueError("cluster_ids must be in ascending order.")
        unit_ids = tuple(_require_nonempty_string(value, "unit_ids") for value in self.unit_ids)
        if len(unit_ids) != len(cluster_ids):
            raise ValueError("unit_ids and cluster_ids must have equal lengths.")
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("unit_ids must be unique.")
        object.__setattr__(self, "cluster_ids", cluster_ids)
        object.__setattr__(self, "unit_ids", unit_ids)


def validate_resolved_populations(
    pfc_population: ResolvedRegionalPopulation,
    hpc_population: ResolvedRegionalPopulation,
) -> None:
    """Validate two nonempty, role-correct, disjoint resolved populations.

    Parameters
    ----------
    pfc_population, hpc_population : ResolvedRegionalPopulation
        Resolved selections with ordered qualified unit identifiers.

    Returns
    -------
    None
        The function returns only after validating the pair.
    """
    if pfc_population.role != "PFC" or hpc_population.role != "HPC":
        raise ValueError("Resolved populations must be supplied in PFC, HPC role order.")
    overlap = set(pfc_population.unit_ids) & set(hpc_population.unit_ids)
    if overlap:
        raise ValueError("Resolved PFC and HPC unit identities must be disjoint.")


@dataclass(frozen=True)
class AnalysisWindows:
    """Relative analysis-window boundaries in seconds from alignment time."""

    whole_start_s: float = -2.0
    split_s: float = 0.0
    whole_stop_s: float = 2.0

    def __post_init__(self) -> None:
        """Require finite ordered bounds with the split exactly at zero seconds."""
        start = _finite_float(self.whole_start_s, "whole_start_s")
        split = _finite_float(self.split_s, "split_s")
        stop = _finite_float(self.whole_stop_s, "whole_stop_s")
        if split != 0.0:
            raise ValueError("split_s must be exactly 0.0 seconds.")
        if not start < split < stop:
            raise ValueError(
                "Window boundaries must satisfy whole_start_s < split_s < whole_stop_s."
            )
        object.__setattr__(self, "whole_start_s", start)
        object.__setattr__(self, "split_s", split)
        object.__setattr__(self, "whole_stop_s", stop)


@dataclass(frozen=True)
class TemporalConfig:
    """Binning and history settings; bin width is in seconds."""

    bin_size_s: float = 0.1
    lag_bins: int = 1
    order_bins: int = 1

    def __post_init__(self) -> None:
        """Validate the frozen bin sizes and positive history counts."""
        bin_size = _finite_float(self.bin_size_s, "bin_size_s")
        if bin_size not in ALLOWED_BIN_SIZES_S:
            raise ValueError(f"bin_size_s must be one of {ALLOWED_BIN_SIZES_S}.")
        object.__setattr__(self, "bin_size_s", bin_size)
        object.__setattr__(self, "lag_bins", _positive_integer(self.lag_bins, "lag_bins"))
        object.__setattr__(self, "order_bins", _positive_integer(self.order_bins, "order_bins"))


@dataclass(frozen=True)
class PCAConfig:
    """Requested positive component counts for each regional PCA."""

    pfc_components: int = 10
    hpc_components: int = 10

    def __post_init__(self) -> None:
        """Validate requested component counts."""
        object.__setattr__(
            self, "pfc_components", _positive_integer(self.pfc_components, "pfc_components")
        )
        object.__setattr__(
            self, "hpc_components", _positive_integer(self.hpc_components, "hpc_components")
        )


@dataclass(frozen=True)
class FilterConfig:
    """Condition, side, context, and zero-based trial-row selections."""

    conditions: tuple[str, ...] = DEFAULT_CONDITIONS
    choice: str = "all"
    context: str = "all"
    excluded_trial_rows: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize condition order and excluded row positions."""
        object.__setattr__(
            self,
            "conditions",
            _normalize_ordered_selection(self.conditions, CANONICAL_CONDITIONS, "conditions"),
        )
        if self.choice not in _ALLOWED_SIDES:
            raise ValueError("choice must be 'all', 'left', or 'right'.")
        if self.context not in _ALLOWED_SIDES:
            raise ValueError("context must be 'all', 'left', or 'right'.")
        object.__setattr__(
            self,
            "excluded_trial_rows",
            _normalize_integer_tuple(
                self.excluded_trial_rows, "excluded_trial_rows", allow_empty=True
            ),
        )


def _validate_window_bin_geometry(windows: AnalysisWindows, temporal: TemporalConfig) -> None:
    """Require before, after, and whole durations to contain integral bins."""
    durations = (
        windows.split_s - windows.whole_start_s,
        windows.whole_stop_s - windows.split_s,
        windows.whole_stop_s - windows.whole_start_s,
    )
    for duration in durations:
        quotient = duration / temporal.bin_size_s
        if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Window durations must be integer multiples of bin_size_s.")


@dataclass(frozen=True)
class InterregionalAnalysisConfig:
    """Complete immutable scientific configuration for one session.

    ``session_metadata_path`` is an absolute normalized filesystem path.
    Window and bin values use seconds; counts and row selections are unitless.
    """

    session_metadata_path: Path | str
    pfc_population: RegionalPopulationConfig
    hpc_population: RegionalPopulationConfig
    config_schema_version: str = CONFIG_SCHEMA_VERSION
    analysis_version: str = ANALYSIS_VERSION
    coverage_assumption_version: str = COVERAGE_ASSUMPTION_VERSION
    alignment: str = "choice_time"
    windows: AnalysisWindows = field(default_factory=AnalysisWindows)
    prediction_windows: tuple[str, ...] = _CANONICAL_WINDOWS
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    representations: tuple[str, ...] = ("units",)
    analyses: tuple[str, ...] = ("ols_cv",)

    def __post_init__(self) -> None:
        """Validate versions, roles, selections, dependencies, and bin geometry."""
        expected_versions = {
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "coverage_assumption_version": COVERAGE_ASSUMPTION_VERSION,
        }
        for field_name, expected in expected_versions.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} must equal {expected!r}.")
        object.__setattr__(
            self,
            "session_metadata_path",
            _normalized_absolute_path(self.session_metadata_path, "session_metadata_path"),
        )
        if self.pfc_population.role != "PFC" or self.hpc_population.role != "HPC":
            raise ValueError("Population configurations must have PFC and HPC roles.")
        if self.alignment not in _ALLOWED_ALIGNMENTS:
            raise ValueError("alignment must be 'choice_time' or 'start_time'.")
        object.__setattr__(
            self,
            "prediction_windows",
            _normalize_ordered_selection(
                self.prediction_windows, _CANONICAL_WINDOWS, "prediction_windows"
            ),
        )
        object.__setattr__(
            self,
            "representations",
            _normalize_ordered_selection(
                self.representations, _CANONICAL_REPRESENTATIONS, "representations"
            ),
        )
        analyses = _normalize_ordered_selection(
            self.analyses, _CANONICAL_ANALYSES, "analyses"
        )
        object.__setattr__(self, "analyses", analyses)
        if "poisson_cv" in analyses and "ols_cv" not in analyses:
            raise ValueError("poisson_cv requires ols_cv for matched held-out comparison.")
        poisson_requested = bool({"poisson_cv", "poisson_granger"} & set(analyses))
        if poisson_requested and "units" not in self.representations:
            raise ValueError("Poisson analyses require the units representation.")
        _validate_window_bin_geometry(self.windows, self.temporal)


@dataclass(frozen=True)
class RunOptions:
    """Non-scientific output and execution controls.

    ``output_root`` is an absolute normalized path when supplied. ``batch_workers``
    is a positive process count without physical units.
    """

    output_root: Path | str | None = None
    rerun: bool = False
    batch_workers: int | None = None

    def __post_init__(self) -> None:
        """Validate non-scientific execution controls."""
        if self.output_root is not None:
            object.__setattr__(
                self,
                "output_root",
                _normalized_absolute_path(self.output_root, "output_root"),
            )
        if not isinstance(self.rerun, bool):
            raise ValueError("rerun must be Boolean.")
        if self.batch_workers is not None:
            object.__setattr__(
                self,
                "batch_workers",
                _positive_integer(self.batch_workers, "batch_workers"),
            )


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    """Return a string-keyed mapping used by the JSON parser."""
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], name: str
) -> None:
    """Reject missing or unknown fields in one JSON object."""
    unknown = set(value) - expected
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(unknown)}")
    missing = expected - set(value)
    if missing:
        raise ValueError(f"Missing {name} fields: {sorted(missing)}")


def _parse_population(value: object, role: str) -> RegionalPopulationConfig:
    """Parse one exact population JSON object."""
    payload = _require_mapping(value, f"populations.{role}")
    expected = {
        "probe_id",
        "channel_source",
        "selected_channels",
        "require_inside_brain",
        "channel_quality_labels",
        "unit_quality_column",
        "unit_quality_labels",
    }
    _require_exact_keys(payload, expected, f"populations.{role}")
    return RegionalPopulationConfig(role=role, **payload)


def _parse_payload(payload: Mapping[str, Any]) -> tuple[InterregionalAnalysisConfig, RunOptions]:
    """Parse an already-decoded portable JSON mapping."""
    top_fields = {
        "config_schema_version",
        "analysis_version",
        "coverage_assumption_version",
        "session_metadata_path",
        "populations",
        "alignment",
        "windows",
        "prediction_windows",
        "temporal",
        "filters",
        "pca",
        "representations",
        "analyses",
        "run",
    }
    _require_exact_keys(payload, top_fields, "configuration")

    populations = _require_mapping(payload["populations"], "populations")
    _require_exact_keys(populations, {"PFC", "HPC"}, "populations")
    windows = _require_mapping(payload["windows"], "windows")
    _require_exact_keys(windows, {"whole_start_s", "split_s", "whole_stop_s"}, "windows")
    temporal = _require_mapping(payload["temporal"], "temporal")
    _require_exact_keys(temporal, {"bin_size_s", "lag_bins", "order_bins"}, "temporal")
    filters = _require_mapping(payload["filters"], "filters")
    _require_exact_keys(
        filters, {"conditions", "choice", "context", "excluded_trial_rows"}, "filters"
    )
    pca = _require_mapping(payload["pca"], "pca")
    _require_exact_keys(pca, {"pfc_components", "hpc_components"}, "pca")
    run = _require_mapping(payload["run"], "run")
    _require_exact_keys(run, {"output_root"}, "run")

    config = InterregionalAnalysisConfig(
        config_schema_version=payload["config_schema_version"],
        analysis_version=payload["analysis_version"],
        coverage_assumption_version=payload["coverage_assumption_version"],
        session_metadata_path=payload["session_metadata_path"],
        pfc_population=_parse_population(populations["PFC"], "PFC"),
        hpc_population=_parse_population(populations["HPC"], "HPC"),
        alignment=payload["alignment"],
        windows=AnalysisWindows(**windows),
        prediction_windows=payload["prediction_windows"],
        temporal=TemporalConfig(**temporal),
        filters=FilterConfig(
            conditions=filters["conditions"],
            choice=filters["choice"],
            context=filters["context"],
            excluded_trial_rows=filters["excluded_trial_rows"],
        ),
        pca=PCAConfig(**pca),
        representations=payload["representations"],
        analyses=payload["analyses"],
    )
    return config, RunOptions(output_root=run["output_root"])


def load_interregional_config(
    path: Path | str,
) -> tuple[InterregionalAnalysisConfig, RunOptions]:
    """Load one exact portable JSON configuration without loading session data.

    Parameters
    ----------
    path : pathlib.Path or str
        JSON file path. The path itself may be relative to the caller; paths
        stored inside the JSON must already be absolute and normalized.

    Returns
    -------
    tuple[InterregionalAnalysisConfig, RunOptions]
        Validated immutable scientific settings and separate execution options.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If decoding or any configuration contract fails.
    """
    config_path = Path(path)
    try:
        decoded = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not decode inter-regional configuration {config_path}.") from error
    payload = _require_mapping(decoded, "configuration")
    return _parse_payload(payload)


def _population_to_dict(population: RegionalPopulationConfig) -> dict[str, object]:
    """Return one JSON-compatible unresolved population mapping."""
    return {
        "probe_id": population.probe_id,
        "channel_source": population.channel_source,
        "selected_channels": list(population.selected_channels),
        "require_inside_brain": population.require_inside_brain,
        "channel_quality_labels": list(population.channel_quality_labels),
        "unit_quality_column": population.unit_quality_column,
        "unit_quality_labels": list(population.unit_quality_labels),
    }


def configuration_to_dict(
    config: InterregionalAnalysisConfig, run_options: RunOptions
) -> dict[str, object]:
    """Return the sole canonical JSON-compatible configuration mapping.

    Parameters
    ----------
    config : InterregionalAnalysisConfig
        Validated scientific configuration; seconds and zero-based indices are
        preserved without conversion.
    run_options : RunOptions
        Separate non-scientific execution settings. CLI-only fields are omitted.

    Returns
    -------
    dict[str, object]
        JSON-compatible mapping in the documented portable format.
    """
    return {
        "config_schema_version": config.config_schema_version,
        "analysis_version": config.analysis_version,
        "coverage_assumption_version": config.coverage_assumption_version,
        "session_metadata_path": str(config.session_metadata_path),
        "populations": {
            "PFC": _population_to_dict(config.pfc_population),
            "HPC": _population_to_dict(config.hpc_population),
        },
        "alignment": config.alignment,
        "windows": {
            "whole_start_s": config.windows.whole_start_s,
            "split_s": config.windows.split_s,
            "whole_stop_s": config.windows.whole_stop_s,
        },
        "prediction_windows": list(config.prediction_windows),
        "temporal": {
            "bin_size_s": config.temporal.bin_size_s,
            "lag_bins": config.temporal.lag_bins,
            "order_bins": config.temporal.order_bins,
        },
        "filters": {
            "conditions": list(config.filters.conditions),
            "choice": config.filters.choice,
            "context": config.filters.context,
            "excluded_trial_rows": list(config.filters.excluded_trial_rows),
        },
        "pca": {
            "pfc_components": config.pca.pfc_components,
            "hpc_components": config.pca.hpc_components,
        },
        "representations": list(config.representations),
        "analyses": list(config.analyses),
        "run": {
            "output_root": None
            if run_options.output_root is None
            else str(run_options.output_root)
        },
    }
