"""Command-line runner for the prescribed behavior model benchmark report."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

from src.behavior_analysis.behavior_model_benchmark import (
    build_score_table,
    build_trial_output_table,
    run_benchmark,
)
from src.behavior_analysis.behavior_model_benchmark_plots import (
    plot_combined_sample_agreement,
    plot_model_behavior_atlas,
    save_benchmark_figure,
)
from src.behavior_modeling.benchmark_models import (
    ORACLE_MODEL_ID,
    BenchmarkModelSpec,
    build_observable_models,
    canonical_model_parameters,
)
from src.behavior_modeling.benchmark_scenarios import (
    BehaviorScenario,
    build_default_scenarios,
)


ORACLE_DISPLAY_LABEL = "Oracle correct side (ceiling)"
OVERRIDE_FIELDS = {
    "simple_probe_persistence": {
        "persistence_weight",
        "probe_weight",
    },
    "expectancy_persistence_doubt": {
        "expectancy_threshold",
        "expectancy_scale",
        "doubt_lambda",
        "persistence_weight",
        "expectancy_weight",
        "doubt_weight",
    },
    "probe_persistence_doubt": {
        "persistence_weight",
        "probe_weight",
        "doubt_weight",
        "doubt_lambda",
    },
    "direct_expectancy_switch_doubt": {
        "expectancy_threshold",
        "expectancy_scale",
        "minimum_switch_probability",
        "switch_probability_span",
        "omission_persistence_weight",
        "doubt_weight",
        "doubt_lambda",
    },
    "global_reward_expectancy_doubt": {
        "reward_decay",
        "expectancy_threshold",
        "expectancy_scale",
        "minimum_switch_probability",
        "switch_probability_span",
        "omission_persistence_weight",
        "doubt_weight",
        "doubt_lambda",
    },
    "home_referenced_expectancy_doubt": {
        "expectancy_threshold",
        "expectancy_scale",
        "expectancy_weight",
        "doubt_weight",
        "doubt_lambda",
    },
}
NONNEGATIVE_FIELDS = {
    "expectancy_threshold",
    "persistence_weight",
    "probe_weight",
    "expectancy_weight",
    "doubt_weight",
    "omission_persistence_weight",
}
POSITIVE_FIELDS = {"expectancy_scale", "doubt_lambda"}
PROBABILITY_FIELDS = {
    "reward_decay",
    "minimum_switch_probability",
    "switch_probability_span",
}


def resolve_benchmark_parameters(
    overrides: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, float | int | str]]:
    """Validate partial override values and merge them with canonical defaults.

    Parameters
    ----------
    overrides : mapping or None
        Partial mapping whose accepted model IDs and numeric fields are listed
        in the benchmark specification. Values are fixed across all samples.

    Returns
    -------
    dict
        Complete independent parameter dictionaries for all 15 model IDs.

    Raises
    ------
    TypeError or ValueError
        If structure, fields, numeric types, finiteness, or documented ranges
        are invalid.
    """
    resolved = canonical_model_parameters()
    if overrides is None:
        return resolved
    if not isinstance(overrides, Mapping):
        raise TypeError("Parameter overrides must be a JSON object.")
    unknown_models = set(overrides).difference(OVERRIDE_FIELDS)
    if unknown_models:
        raise ValueError(
            f"Unknown or non-overrideable model IDs: {sorted(unknown_models)}"
        )

    for model_id, supplied_fields in overrides.items():
        if not isinstance(supplied_fields, Mapping):
            raise TypeError(f"Overrides for {model_id!r} must be an object.")
        unknown_fields = set(supplied_fields).difference(OVERRIDE_FIELDS[model_id])
        if unknown_fields:
            raise ValueError(
                f"Unknown override fields for {model_id!r}: "
                f"{sorted(unknown_fields)}"
            )
        for field_name, raw_value in supplied_fields.items():
            value = _validated_numeric_override(field_name, raw_value)
            resolved[model_id][field_name] = value

    for model_id in (
        "direct_expectancy_switch_doubt",
        "global_reward_expectancy_doubt",
    ):
        values = resolved[model_id]
        probability_sum = float(values["minimum_switch_probability"]) + float(
            values["switch_probability_span"]
        )
        if probability_sum > 1.0:
            raise ValueError(
                f"{model_id} minimum switch probability plus span must not "
                "exceed one."
            )
    # Construction validates the existing exemplar dataclasses as a final
    # defense and confirms that the resolved record is complete.
    build_observable_models(resolved)
    return resolved


def select_scenarios(
    scenarios: Sequence[BehaviorScenario],
    requested_ids: Sequence[str] | None,
) -> tuple[BehaviorScenario, ...]:
    """Filter scenarios by ID while retaining canonical registry order."""
    scenario_tuple = tuple(scenarios)
    if not requested_ids:
        return scenario_tuple
    requested = set(requested_ids)
    known = {scenario.scenario_id for scenario in scenario_tuple}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"Unknown scenario IDs: {unknown}")
    return tuple(
        scenario for scenario in scenario_tuple if scenario.scenario_id in requested
    )


def select_models(
    models: Sequence[BenchmarkModelSpec],
    requested_ids: Sequence[str] | None,
) -> tuple[tuple[BenchmarkModelSpec, ...], bool]:
    """Filter ordinary models in registry order and select the separate oracle."""
    model_tuple = tuple(models)
    if not requested_ids:
        return model_tuple, True
    requested = set(requested_ids)
    known = {model.model_id for model in model_tuple} | {ORACLE_MODEL_ID}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"Unknown model IDs: {unknown}")
    selected = tuple(model for model in model_tuple if model.model_id in requested)
    return selected, ORACLE_MODEL_ID in requested


def create_run_directory(
    output_root: Path,
    *,
    repository_root: Path,
    timestamp: datetime | None = None,
) -> Path:
    """Create one exclusive timestamped report directory outside the repository.

    Parameters
    ----------
    output_root : pathlib.Path
        Requested parent directory for generated human-readable artifacts.
    repository_root : pathlib.Path
        Source repository path that must never contain generated reports.
    timestamp : datetime or None
        Time used in the run name. ``None`` uses the current UTC time; supplied
        values must be timezone-aware.

    Returns
    -------
    pathlib.Path
        Newly created resolved run directory.
    """
    repository = Path(repository_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve(strict=False)
    if output == repository or output.is_relative_to(repository):
        raise ValueError("Benchmark output root must resolve outside the repository.")
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    if output == repository or output.is_relative_to(repository):
        raise ValueError("Benchmark output root must resolve outside the repository.")

    run_time = timestamp or datetime.now(timezone.utc)
    if run_time.tzinfo is None:
        raise ValueError("Run timestamps must be timezone-aware.")
    utc_time = run_time.astimezone(timezone.utc)
    run_name = utc_time.strftime("behavior_model_benchmark_%Y%m%dT%H%M%S%fZ")
    run_directory = output / run_name
    run_directory.mkdir(exist_ok=False)
    return run_directory.resolve()


def run_report(
    output_root: Path,
    *,
    model_ids: Sequence[str] | None = None,
    scenario_ids: Sequence[str] | None = None,
    parameter_overrides: Mapping[str, Mapping[str, Any]] | None = None,
    repository_root: Path | None = None,
    timestamp: datetime | None = None,
) -> Path:
    """Compute and save one selected deterministic benchmark report.

    Parameters
    ----------
    output_root : pathlib.Path
        Parent directory outside the repository.
    model_ids, scenario_ids : sequence of str or None
        Optional filters. Registry order is retained; ``None`` selects all.
    parameter_overrides : mapping or None
        Validated uniformly applied fixed parameter overrides.
    repository_root : pathlib.Path or None
        Repository used for containment and version metadata. ``None`` derives
        it from this module's location.
    timestamp : datetime or None
        Optional timezone-aware UTC naming time, primarily for deterministic
        tests. Normal runs use the current time.

    Returns
    -------
    pathlib.Path
        New report directory containing plots, tables, parameters, summary,
        and log. No session-shaped files are created.
    """
    start_time = time.perf_counter()
    repository = (
        Path(repository_root).resolve()
        if repository_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved_parameters = resolve_benchmark_parameters(parameter_overrides)
    all_scenarios = build_default_scenarios()
    selected_scenarios = select_scenarios(all_scenarios, scenario_ids)
    all_models = build_observable_models(resolved_parameters)
    selected_models, include_oracle = select_models(all_models, model_ids)
    if not selected_models and not include_oracle:
        raise ValueError("At least one model must be selected.")

    run_directory = create_run_directory(
        output_root,
        repository_root=repository,
        timestamp=timestamp,
    )
    result = run_benchmark(
        selected_scenarios,
        selected_models,
        include_oracle=include_oracle,
    )
    trial_table = build_trial_output_table(result, selected_scenarios)
    score_table = build_score_table(result)
    trial_table.to_csv(run_directory / "trialwise_model_outputs.csv", index=False)
    score_table.to_csv(run_directory / "scenario_model_scores.csv", index=False)
    _write_json(run_directory / "model_parameters.json", resolved_parameters)

    selected_metadata = [
        (model.model_id, model.display_label, model.parameters)
        for model in selected_models
    ]
    if include_oracle:
        selected_metadata.append(
            (
                ORACLE_MODEL_ID,
                ORACLE_DISPLAY_LABEL,
                resolved_parameters[ORACLE_MODEL_ID],
            )
        )
    model_labels = {
        model_id: display_label
        for model_id, display_label, _ in selected_metadata
    }
    diagnostics_root = run_directory / "diagnostics"
    for model_id, display_label, parameters in selected_metadata:
        model_directory = diagnostics_root / model_id
        model_directory.mkdir(parents=True, exist_ok=False)
        figures = plot_model_behavior_atlas(
            selected_scenarios,
            result.replays,
            model_id,
            display_label,
            parameters,
        )
        for page_number, figure in enumerate(figures, start=1):
            save_benchmark_figure(
                figure,
                model_directory / f"page_{page_number:02d}.png",
            )

    agreement_figure = plot_combined_sample_agreement(score_table, model_labels)
    save_benchmark_figure(
        agreement_figure,
        run_directory / "combined_sample_agreement.png",
    )
    code_version, dirty_status = _git_metadata(repository)
    elapsed_seconds = time.perf_counter() - start_time
    _write_summary(
        run_directory / "summary.md",
        selected_scenarios,
        selected_metadata,
        resolved_parameters,
        code_version,
        dirty_status,
    )
    _write_log(
        run_directory / "run.log",
        selected_scenarios,
        selected_metadata,
        elapsed_seconds,
    )
    return run_directory


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments, run one report, and return a status code."""
    parser = argparse.ArgumentParser(
        description="Run prescribed behaviors through fixed behavior models."
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model", action="append", dest="model_ids")
    parser.add_argument("--scenario", action="append", dest="scenario_ids")
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args(argv)
    overrides = _load_json_overrides(arguments.config)
    run_directory = run_report(
        arguments.output_root,
        model_ids=arguments.model_ids,
        scenario_ids=arguments.scenario_ids,
        parameter_overrides=overrides,
    )
    print(run_directory)
    return 0


# ---------------------------------------------------------------------------
# Helpers


def _validated_numeric_override(field_name: str, raw_value: Any) -> float:
    """Validate one JSON numeric value against its field-specific range."""
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise TypeError(f"{field_name} must be a number, not {type(raw_value).__name__}.")
    value = float(raw_value)
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite.")
    if field_name in NONNEGATIVE_FIELDS and value < 0:
        raise ValueError(f"{field_name} must be nonnegative.")
    if field_name in POSITIVE_FIELDS and value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    if field_name in PROBABILITY_FIELDS and not 0 <= value <= 1:
        raise ValueError(f"{field_name} must lie in [0, 1].")
    return value


def _load_json_overrides(path: Path | None) -> Mapping[str, Mapping[str, Any]] | None:
    """Load one optional UTF-8 JSON override object without mutating it."""
    if path is None:
        return None
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise TypeError("The configuration document must contain a JSON object.")
    return document


def _write_json(path: Path, value: object) -> None:
    """Write deterministic indented UTF-8 JSON beneath a prepared run path."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _git_metadata(repository: Path) -> tuple[str, str]:
    """Return commit identifier and dirty status without modifying the worktree."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return revision, "dirty" if status else "clean"


def _write_summary(
    path: Path,
    scenarios: Sequence[BehaviorScenario],
    model_metadata: Sequence[
        tuple[str, str, Mapping[str, float | int | str]]
    ],
    resolved_parameters: Mapping[str, Mapping[str, float | int | str]],
    code_version: str,
    dirty_status: str,
) -> None:
    """Write the human-readable goal, selection, scoring, and metadata summary."""
    scenario_text = ", ".join(item.scenario_id for item in scenarios)
    model_text = ", ".join(item[0] for item in model_metadata)
    text = f"""# Behavior model benchmark report

## Goal

Compare fixed behavior models against prescribed switching samples without
trialwise model selection or per-sample parameter fitting.

## Included samples and models

- Samples: {scenario_text}
- Models: {model_text}

## Scores

The report retains aligned choice probabilities, mean negative log probability,
switch/stay Brier score, and sample-as-block greedy agreement. Exact greedy ties
contribute 0.5. The NLL floor is 1e-12 and does not alter stored probabilities.

## Reproducibility

- Code version: `{code_version}`
- Worktree status at run time: `{dirty_status}`
- Resolved parameter groups: {len(resolved_parameters)}
"""
    path.write_text(text, encoding="utf-8")


def _write_log(
    path: Path,
    scenarios: Sequence[BehaviorScenario],
    model_metadata: Sequence[
        tuple[str, str, Mapping[str, float | int | str]]
    ],
    elapsed_seconds: float,
) -> None:
    """Write concise runtime, selection, and completion information."""
    lines = [
        "status=completed",
        f"elapsed_seconds={elapsed_seconds:.6f}",
        "scenario_ids=" + ",".join(item.scenario_id for item in scenarios),
        "model_ids=" + ",".join(item[0] for item in model_metadata),
        "warnings=none",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
