import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def run_import_smoke(import_stmt: str) -> subprocess.CompletedProcess[str]:
    script = f"{import_stmt}\nprint('import-ok')\n"
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_controller_imports_without_sys_path_hacks():
    result = run_import_smoke("from src.behavior_modeling import controller")

    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout


def test_visualization_wrappers_import_without_sys_path_hacks():
    result = run_import_smoke(
        "from src.visualization.agent_run_plot import plot_run_dataframe\n"
        "from src.behavior_modeling.visualize_behavior.agent_run_plot import resolve_value_columns"
    )

    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout
