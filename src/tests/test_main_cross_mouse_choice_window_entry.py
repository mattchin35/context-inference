"""Quick-edit entry wiring delegates to a stub, never to experimental analysis."""

import ast
from pathlib import Path
from unittest.mock import Mock

import pytest


NAME = "main_cross_mouse_choice_window_agreement"
MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"


def wrapper_namespace(overrides=None):
    """Compile only the wrapper AST with optional setting expressions and a stub runner.

    overrides maps setting names to Python expression strings, simulating edits
    in the opening settings block. Return the function namespace; no filesystem
    operations or top-level main execution occur. Values/paths have no physical units.
    """
    tree = ast.parse(MAIN_PATH.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == NAME)
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and target.id in (overrides or {}):
                statement.value = ast.parse(overrides[target.id], mode="eval").body
    namespace = {"Path": Path, "run_cross_mouse_choice_window_agreement_curves": Mock(return_value=(None, Path("/synthetic/exports")))}
    isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(isolated, str(MAIN_PATH), "exec"), namespace)
    return namespace


def test_quick_entry_defaults_and_printed_output(capsys):
    """No-argument wrapper selects the approved cohort, both sizes and automatic mode."""
    namespace = wrapper_namespace()
    namespace[NAME]()
    runner = namespace["run_cross_mouse_choice_window_agreement_curves"]
    runner.assert_called_once()
    settings = runner.call_args.kwargs
    assert settings["mice"] == ["CT016", "CT017", "CT019", "CT020", "CT021", "CT022", "CT023", "CT024", "CT025"]
    assert settings["window_sizes"] == (10, 20)
    assert settings["collection_runs"] is None
    assert settings["output_path"] == settings["data_root"] / "cross_mouse_analysis"
    assert "/synthetic/exports" in capsys.readouterr().out


@pytest.mark.parametrize("explicit", [False, True])
def test_opening_settings_control_runner_and_selection_mode(explicit):
    """Editing root/mice/size/run mapping at the function opening changes the delegated call."""
    overrides = {"data_root": 'Path("/synthetic/data")', "mice": '["example_mouse"]',
                 "window_sizes": "(20,)", "figure_id": '"quick_preview"'}
    if explicit:
        overrides["collection_runs"] = '{"chosen_mouse": Path("/synthetic/chosen_run")}'
    namespace = wrapper_namespace(overrides)
    namespace[NAME]()
    settings = namespace["run_cross_mouse_choice_window_agreement_curves"].call_args.kwargs
    assert settings["data_root"] == Path("/synthetic/data")
    assert settings["output_path"] == Path("/synthetic/data/cross_mouse_analysis")
    assert settings["window_sizes"] == (20,) and settings["figure_id"] == "quick_preview"
    if explicit:
        assert settings["mice"] is None
        assert settings["collection_runs"] == {"chosen_mouse": Path("/synthetic/chosen_run")}
    else:
        assert settings["mice"] == ["example_mouse"] and settings["collection_runs"] is None


def test_bottom_block_has_only_the_new_entry_call():
    """The script's active workflow is selected by a single no-argument call; imports do not call it."""
    tree = ast.parse(MAIN_PATH.read_text())
    guard = next(node for node in tree.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "__name__ == '__main__'")
    assert len(guard.body) == 1
    call = guard.body[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    assert call.func.id == NAME and not call.args and not call.keywords
    for module_name, expected in (("__main__", 1), ("imported_main", 0)):
        entry = Mock()
        namespace = {"__name__": module_name, NAME: entry}
        isolated = ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[]))
        exec(compile(isolated, str(MAIN_PATH), "exec"), namespace)
        assert entry.call_count == expected
