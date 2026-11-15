"""Real spawned synthetic-session workers and duplicate-session protection."""

import shutil
import pytest

from src.neural_analysis.population_decoding.run_batch import run_batch, choose_worker_count


def test_worker_limits_and_duplicate_sessions(population_config_path):
    """Input one session config; cap workers and reject duplicate roots before writes."""
    assert choose_worker_count(3, None, cpu_count=2) == 2
    assert choose_worker_count(3, 10, cpu_count=2) == 2
    with pytest.raises(ValueError):
        run_batch((population_config_path, population_config_path), command="dry-run")


def test_two_synthetic_sessions_execute_in_spawned_workers(population_config_path, tmp_path):
    """Input small synthetic sources; run two independent sessions in two spawned processes."""
    second_root = tmp_path / "second-session"
    shutil.copytree(population_config_path.parent, second_root)
    second = second_root / population_config_path.name
    report = run_batch((population_config_path, second), workers=2)
    assert report["status"] == "complete", report
    assert report["worker_count"] == 2
    assert all(item["completed_cells"] == 14 for item in report["reports"])
