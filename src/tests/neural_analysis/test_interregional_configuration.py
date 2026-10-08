"""Tests for immutable inter-regional regression configuration contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from src.neural_analysis.interregional import configuration


def _documented_payload(metadata_path: Path) -> dict[str, object]:
    """Return the portable JSON example with a caller-owned absolute path."""
    return {
        "config_schema_version": "1",
        "analysis_version": "interregional-regression-v1",
        "coverage_assumption_version": "implicit-complete-v1",
        "session_metadata_path": str(metadata_path),
        "populations": {
            "PFC": {
                "probe_id": "probe_a",
                "channel_source": "metadata_quality",
                "selected_channels": [],
                "require_inside_brain": True,
                "channel_quality_labels": ["good"],
                "unit_quality_column": "group",
                "unit_quality_labels": ["good", "mua"],
            },
            "HPC": {
                "probe_id": "probe_b",
                "channel_source": "metadata_quality",
                "selected_channels": [],
                "require_inside_brain": True,
                "channel_quality_labels": ["good"],
                "unit_quality_column": "group",
                "unit_quality_labels": ["good", "mua"],
            },
        },
        "alignment": "choice_time",
        "windows": {"whole_start_s": -2.0, "split_s": 0.0, "whole_stop_s": 2.0},
        "prediction_windows": ["before", "after", "whole"],
        "temporal": {"bin_size_s": 0.1, "lag_bins": 1, "order_bins": 1},
        "filters": {
            "conditions": [
                "all",
                "correct_rewarded",
                "incorrect",
                "omission",
                "switch",
                "stay",
            ],
            "choice": "all",
            "context": "all",
            "excluded_trial_rows": [],
        },
        "pca": {"pfc_components": 10, "hpc_components": 10},
        "representations": ["units"],
        "analyses": ["ols_cv"],
        "run": {"output_root": None},
    }


def _write_config(tmp_path: Path, payload: dict[str, object]) -> Path:
    """Write one test configuration and return its absolute path."""
    path = tmp_path / "interregional.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load(tmp_path: Path, payload: dict[str, object] | None = None):
    """Load one portable configuration through the production API."""
    metadata_path = (tmp_path / "session" / "neural_session.json").resolve()
    selected_payload = _documented_payload(metadata_path) if payload is None else payload
    return configuration.load_interregional_config(_write_config(tmp_path, selected_payload))


def test_documented_json_loads_to_frozen_records_and_round_trips(tmp_path: Path) -> None:
    """The documented format is the sole portable configuration representation."""
    config, run_options = _load(tmp_path)

    assert config.config_schema_version == configuration.CONFIG_SCHEMA_VERSION
    assert config.analysis_version == configuration.ANALYSIS_VERSION
    assert config.coverage_assumption_version == configuration.COVERAGE_ASSUMPTION_VERSION
    assert config.pfc_population.role == "PFC"
    assert config.hpc_population.role == "HPC"
    assert config.prediction_windows == ("before", "after", "whole")
    assert config.filters.conditions == configuration.DEFAULT_CONDITIONS
    assert run_options.output_root is None
    assert run_options.rerun is False
    assert run_options.batch_workers is None
    assert configuration.configuration_to_dict(config, run_options) == _documented_payload(
        config.session_metadata_path
    )

    with pytest.raises(FrozenInstanceError):
        config.alignment = "start_time"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("level", "field"),
    [
        ("top", "typo"),
        ("population", "cluster_ids"),
        ("windows", "units"),
        ("temporal", "smooth"),
        ("filters", "rewarded"),
        ("pca", "whiten"),
        ("run", "seed"),
    ],
)
def test_unknown_json_fields_are_rejected_at_every_level(
    tmp_path: Path, level: str, field: str
) -> None:
    """Typos cannot silently alter a scientific analysis."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    if level == "top":
        payload[field] = True
    elif level == "population":
        payload["populations"]["PFC"][field] = []  # type: ignore[index]
    else:
        payload[level][field] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="Unknown"):
        _load(tmp_path, payload)


@pytest.mark.parametrize("bin_size_s", [0.2, 0.0, -0.1, float("nan")])
def test_temporal_configuration_rejects_unsupported_bin_sizes(
    tmp_path: Path, bin_size_s: float
) -> None:
    """Only the frozen bin widths are accepted, in seconds."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["temporal"]["bin_size_s"] = bin_size_s  # type: ignore[index]

    with pytest.raises(ValueError, match="bin_size_s"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(("field", "value"), [("lag_bins", 0), ("order_bins", 1.5)])
def test_temporal_configuration_requires_positive_integer_lag_and_order(
    tmp_path: Path, field: str, value: object
) -> None:
    """Lag and autoregressive order are positive counts of bins."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["temporal"][field] = value  # type: ignore[index]

    with pytest.raises(ValueError, match=field):
        _load(tmp_path, payload)


def test_window_geometry_must_partition_at_zero_and_fit_the_bin_width(tmp_path: Path) -> None:
    """Before, after, and whole durations must share exact integer bin counts."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["windows"]["split_s"] = 0.01  # type: ignore[index]
    with pytest.raises(ValueError, match="split_s"):
        _load(tmp_path, payload)

    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["windows"]["whole_start_s"] = -2.03  # type: ignore[index]
    with pytest.raises(ValueError, match="bin_size_s"):
        _load(tmp_path, payload)


def test_population_selection_is_normalized_and_validated() -> None:
    """Selections use sorted unique channels/labels and explicit-source rules."""
    population = configuration.RegionalPopulationConfig(
        role="PFC",
        probe_id="probe_a",
        channel_source="explicit",
        selected_channels=(4, 2),
        channel_quality_labels=("good",),
        unit_quality_labels=("mua", "good"),
    )
    assert population.selected_channels == (2, 4)
    assert population.unit_quality_labels == ("good", "mua")

    with pytest.raises(ValueError, match="selected_channels"):
        configuration.RegionalPopulationConfig(
            role="PFC", probe_id="probe_a", channel_source="explicit"
        )
    with pytest.raises(ValueError, match="must be empty"):
        configuration.RegionalPopulationConfig(
            role="PFC",
            probe_id="probe_a",
            channel_source="metadata_quality",
            selected_channels=(1,),
        )


def test_resolved_populations_require_ordered_disjoint_qualified_units() -> None:
    """Resolved cluster order and cross-region unit identity are explicit."""
    pfc = configuration.ResolvedRegionalPopulation(
        role="PFC",
        probe_id="probe_a",
        channel_source="explicit",
        selected_channels=(1,),
        cluster_ids=(2, 9),
        unit_ids=("probe_a:2", "probe_a:9"),
    )
    hpc = configuration.ResolvedRegionalPopulation(
        role="HPC",
        probe_id="probe_b",
        channel_source="explicit",
        selected_channels=(3,),
        cluster_ids=(4,),
        unit_ids=("probe_b:4",),
    )
    configuration.validate_resolved_populations(pfc, hpc)

    with pytest.raises(ValueError, match="disjoint"):
        configuration.validate_resolved_populations(
            pfc,
            configuration.ResolvedRegionalPopulation(
                role="HPC",
                probe_id="probe_b",
                channel_source="explicit",
                selected_channels=(3,),
                cluster_ids=(4,),
                unit_ids=("probe_a:2",),
            ),
        )


def test_conditions_are_unique_canonical_and_exclude_legacy_rewarded(tmp_path: Path) -> None:
    """Requested conditions normalize to canonical order without aliases."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["filters"]["conditions"] = ["stay", "all", "switch"]  # type: ignore[index]
    config, _ = _load(tmp_path, payload)
    assert config.filters.conditions == ("all", "switch", "stay")

    for invalid in (["all", "all"], ["rewarded"], []):
        payload["filters"]["conditions"] = invalid  # type: ignore[index]
        with pytest.raises(ValueError, match="conditions"):
            _load(tmp_path, payload)


def test_analysis_dependencies_and_canonical_order_are_enforced(tmp_path: Path) -> None:
    """Poisson CV has its matched OLS/unit dependencies; Granger remains independent."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["representations"] = ["pcs", "units"]
    payload["analyses"] = ["linear_granger", "ols_cv"]
    config, _ = _load(tmp_path, payload)
    assert config.representations == ("units", "pcs")
    assert config.analyses == ("ols_cv", "linear_granger")

    payload["analyses"] = ["poisson_cv"]
    with pytest.raises(ValueError, match="ols_cv"):
        _load(tmp_path, payload)

    payload["analyses"] = ["poisson_granger"]
    payload["representations"] = ["pcs"]
    with pytest.raises(ValueError, match="units"):
        _load(tmp_path, payload)

    payload["analyses"] = ["linear_granger"]
    config, _ = _load(tmp_path, payload)
    assert config.analyses == ("linear_granger",)


def test_versions_and_execution_paths_are_strict(tmp_path: Path) -> None:
    """Version mismatches and nonnormalized relative execution paths are rejected."""
    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["analysis_version"] = "future"
    with pytest.raises(ValueError, match="analysis_version"):
        _load(tmp_path, payload)

    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["session_metadata_path"] = "relative/neural_session.json"
    with pytest.raises(ValueError, match="absolute"):
        _load(tmp_path, payload)

    payload = _documented_payload((tmp_path / "neural_session.json").resolve())
    payload["run"]["output_root"] = "relative/results"  # type: ignore[index]
    with pytest.raises(ValueError, match="absolute"):
        _load(tmp_path, payload)
