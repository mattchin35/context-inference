"""Tests for benchmark configuration and the path-safe report runner."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np
import pytest

from src.behavior_analysis.run_behavior_model_benchmark import (
    create_run_directory,
    resolve_benchmark_parameters,
    run_report,
    select_models,
    select_scenarios,
)
from src.behavior_modeling.benchmark_models import (
    ORACLE_MODEL_ID,
    ObservableHistory,
    build_observable_models,
)
from src.behavior_modeling.benchmark_scenarios import build_default_scenarios


def test_filters_preserve_registry_order_and_reject_unknown_ids() -> None:
    """CLI filters select named entries without adopting argument order."""
    scenarios = build_default_scenarios()
    selected_scenarios = select_scenarios(scenarios, ("CT-03B", "AT-01"))
    assert [item.scenario_id for item in selected_scenarios] == ["AT-01", "CT-03B"]

    models = build_observable_models()
    selected_models, include_oracle = select_models(
        models,
        ("forgetting_q_learning", ORACLE_MODEL_ID, "chance"),
    )
    assert [item.model_id for item in selected_models] == [
        "chance",
        "forgetting_q_learning",
    ]
    assert include_oracle

    with pytest.raises(ValueError, match="Unknown scenario"):
        select_scenarios(scenarios, ("not-a-scenario",))
    with pytest.raises(ValueError, match="Unknown model"):
        select_models(models, ("not-a-model",))


def test_partial_overrides_merge_with_defaults_and_affect_replay() -> None:
    """One resolved parameter set is complete and used by model construction."""
    resolved = resolve_benchmark_parameters(
        {"probe_persistence_doubt": {"probe_weight": 3.0}}
    )

    assert resolved["probe_persistence_doubt"]["probe_weight"] == 3.0
    assert resolved["probe_persistence_doubt"]["persistence_weight"] == 1.0
    assert resolved["history_ideal_observer"]["reward_decay_lambda"] == 0.1
    assert ORACLE_MODEL_ID in resolved

    model = next(
        item
        for item in build_observable_models(resolved)
        if item.model_id == "probe_persistence_doubt"
    )
    scenario = next(
        item for item in build_default_scenarios() if item.scenario_id == "PR-02"
    )
    replay = model.replay(
        ObservableHistory(scenario.action, scenario.reward, scenario.valid)
    )
    assert replay.p_switch[1] > 0.97


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown": {"weight": 1.0}},
        {"chance": {"weight": 1.0}},
        {"probe_persistence_doubt": {"unknown": 1.0}},
        {"probe_persistence_doubt": {"probe_weight": True}},
        {"probe_persistence_doubt": {"probe_weight": np.inf}},
        {"probe_persistence_doubt": {"probe_weight": -1.0}},
        {"direct_expectancy_switch_doubt": {"expectancy_scale": 0.0}},
        {"global_reward_expectancy_doubt": {"reward_decay": 1.1}},
        {
            "direct_expectancy_switch_doubt": {
                "minimum_switch_probability": 0.7,
                "switch_probability_span": 0.4,
            }
        },
    ],
)
def test_invalid_overrides_are_rejected(overrides) -> None:
    """Unknown, boolean, nonfinite, negative, and incoherent values fail early."""
    with pytest.raises((TypeError, ValueError)):
        resolve_benchmark_parameters(overrides)


def test_run_directory_resolves_containment_and_refuses_overwrite(tmp_path) -> None:
    """Resolved aliases cannot write inside source and run paths are exclusive."""
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    timestamp = datetime(2026, 10, 12, 15, 30, 45, 123456, tzinfo=timezone.utc)

    with pytest.raises(ValueError, match="outside"):
        create_run_directory(
            repository_root / "reports",
            repository_root=repository_root,
            timestamp=timestamp,
        )

    alias = tmp_path / "repository-alias"
    alias.symlink_to(repository_root, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        create_run_directory(
            alias / "reports",
            repository_root=repository_root,
            timestamp=timestamp,
        )

    output_root = tmp_path / "external-output"
    run_directory = create_run_directory(
        output_root,
        repository_root=repository_root,
        timestamp=timestamp,
    )
    assert run_directory.parent == output_root.resolve()
    assert run_directory.name == "behavior_model_benchmark_20261012T153045123456Z"
    with pytest.raises(FileExistsError):
        create_run_directory(
            output_root,
            repository_root=repository_root,
            timestamp=timestamp,
        )


def test_smoke_report_creates_exact_documented_artifacts(tmp_path) -> None:
    """A minimal run writes only benchmark artifacts beneath its new directory."""
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    output_root = tmp_path / "output"
    timestamp = datetime(2026, 10, 12, 16, 0, 0, tzinfo=timezone.utc)

    run_directory = run_report(
        output_root,
        model_ids=("chance",),
        scenario_ids=("AT-01",),
        parameter_overrides=None,
        repository_root=repository_root,
        timestamp=timestamp,
    )

    relative_files = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
    }
    assert relative_files == {
        "combined_sample_agreement.png",
        "diagnostics/chance/page_01.png",
        "model_parameters.json",
        "run.log",
        "scenario_model_scores.csv",
        "summary.md",
        "trialwise_model_outputs.csv",
    }
    assert all(path.is_relative_to(run_directory) for path in run_directory.rglob("*"))

    with (run_directory / "model_parameters.json").open(encoding="utf-8") as handle:
        saved_parameters = json.load(handle)
    assert set(saved_parameters) == {
        *[model.model_id for model in build_observable_models()],
        ORACLE_MODEL_ID,
    }
    summary = (run_directory / "summary.md").read_text(encoding="utf-8")
    log = (run_directory / "run.log").read_text(encoding="utf-8")
    assert "AT-01" in summary and "chance" in summary
    assert "greedy agreement" in summary.lower()
    assert "completed" in log.lower()
