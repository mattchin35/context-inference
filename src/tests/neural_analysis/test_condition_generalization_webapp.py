"""Saved-result webapp contracts for pooled condition generalization."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from src.neural_analysis.task_decoding.pooled import results
from src.tests.neural_analysis.task_decoding.pooled.test_results import (
    make_result_template,
)


PACKAGE_ROOT = Path(__file__).parents[2] / "neural_analysis" / "webapp"


def _write_json(path: Path, value: object) -> None:
    """Write one small ASCII JSON test artifact.

    Parameters
    ----------
    path : pathlib.Path
        Destination inside a pytest-owned temporary directory.
    value : object
        JSON-compatible test payload.

    Returns
    -------
    None
        Writes exactly one fixture artifact.
    """
    path.write_text(json.dumps(value), encoding="ascii")


def _write_completed_run(
    run_directory: Path,
    *,
    original_session_root: str = "/cluster/copied/session",
) -> None:
    """Write one structurally valid completed pooled run fixture.

    Parameters
    ----------
    run_directory : pathlib.Path
        Direct child to populate with manifests and one valid ``results.npz``.
    original_session_root : str
        Execution-time provenance path. It may differ from the current test
        session to exercise portable cluster-to-workstation result copies.

    Returns
    -------
    None
        Creates only pytest-owned saved-result files.
    """
    arrays, metadata = make_result_template()
    results.save_pooled_run(run_directory, arrays=arrays, metadata=metadata)
    _write_json(
        run_directory / "run_state.json",
        {"lifecycle": "complete", "final_results_published": True},
    )
    _write_json(
        run_directory / "execution.json",
        {
            "run_fingerprint": metadata["provenance"]["run_fingerprint"],
            "session_root": original_session_root,
        },
    )
    _write_json(
        run_directory / "config.json",
        {
            "analysis_version": metadata["analysis_version"],
            "base_config": {"target_names": arrays["target_names"].tolist()},
            "condition_names": arrays["condition_names"].tolist(),
            "training_windows": [
                {
                    "name": str(name),
                    "start_s": float(start),
                    "stop_s": float(stop),
                }
                for name, start, stop in zip(
                    arrays["training_window_names"],
                    arrays["training_window_start_s"],
                    arrays["training_window_stop_s"],
                    strict=True,
                )
            ],
        },
    )


def test_condition_generalization_view_has_one_saved_result_only_owner() -> None:
    """The view owns presentation but imports no pooled computation pipeline."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    for object_name in (
        "resolve_condition_generalization_results_root",
        "discover_condition_generalization_runs",
        "load_selected_condition_generalization_run",
        "render_condition_generalization_view",
    ):
        assert getattr(view, object_name).__module__ == view.__name__

    source = (PACKAGE_ROOT / "condition_generalization_views.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "src.neural_analysis.task_decoding.pooled.results" in imported_modules
    assert "src.neural_analysis.task_decoding.pooled.plotting" in imported_modules
    assert not any(
        name.endswith(("pooled.pipeline", "pooled.decoding", "task_decoding.activity"))
        for name in imported_modules
    )
    assert "streamlit" not in source


def test_results_root_is_session_relative_contained_and_not_symlinked(
    tmp_path: Path,
) -> None:
    """Saved-result discovery cannot escape through traversal or a linked root."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    session_root = tmp_path / "session"
    session_root.mkdir()
    expected = session_root / "analysis_runs"
    assert view.resolve_condition_generalization_results_root(
        session_root, "analysis_runs"
    ) == expected.resolve()

    for locator in ("../outside", str(tmp_path / "absolute"), "."):
        with pytest.raises(ValueError, match="session|relative|child"):
            view.resolve_condition_generalization_results_root(session_root, locator)

    outside = tmp_path / "outside"
    outside.mkdir()
    expected.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        view.resolve_condition_generalization_results_root(
            session_root, "analysis_runs"
        )


def test_discovery_lists_only_direct_complete_portable_pooled_runs(
    tmp_path: Path,
) -> None:
    """Discovery is bounded, manifest-led, and permits copied cluster results."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    session_root = tmp_path / "workstation-session"
    results_root = session_root / "analysis_runs"
    results_root.mkdir(parents=True)
    valid = results_root / "pooled_task_variable_decoding_20261012T045944Z"
    _write_completed_run(valid, original_session_root="/cluster/session")

    hidden = results_root / ".pooled_task_variable_decoding_transfer"
    _write_completed_run(hidden)
    wrong_prefix = results_root / "ordinary_task_decoding_20261012"
    _write_completed_run(wrong_prefix)
    incomplete = results_root / "pooled_task_variable_decoding_incomplete"
    incomplete.mkdir()
    _write_json(
        incomplete / "run_state.json",
        {"lifecycle": "running", "final_results_published": False},
    )
    malformed = results_root / "pooled_task_variable_decoding_malformed"
    malformed.mkdir()
    (malformed / "run_state.json").write_text("not-json", encoding="ascii")
    nested = results_root / "group" / "pooled_task_variable_decoding_nested"
    _write_completed_run(nested)

    discovered = view.discover_condition_generalization_runs(session_root)

    assert [option.run_directory for option in discovered] == [valid.resolve()]
    assert discovered[0].run_fingerprint == "run-fingerprint"
    assert discovered[0].original_session_root == "/cluster/session"


def test_discovery_rejects_linked_run_and_required_member(tmp_path: Path) -> None:
    """Neither a linked run directory nor linked manifest can enter the selector."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    session_root = tmp_path / "session"
    results_root = session_root / "analysis_runs"
    results_root.mkdir(parents=True)
    outside = tmp_path / "outside" / "pooled_task_variable_decoding_external"
    _write_completed_run(outside)
    (results_root / outside.name).symlink_to(outside, target_is_directory=True)

    linked_member = results_root / "pooled_task_variable_decoding_linked_member"
    _write_completed_run(linked_member)
    (linked_member / "execution.json").unlink()
    (linked_member / "execution.json").symlink_to(outside / "execution.json")

    assert view.discover_condition_generalization_runs(session_root) == ()


def test_selected_run_is_reloaded_with_its_expected_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selection delegates strict NPZ validation with immutable run identity."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    run_directory = tmp_path / "pooled_task_variable_decoding_complete"
    _write_completed_run(run_directory)
    option = view.ConditionGeneralizationRunOption(
        run_directory=run_directory,
        run_fingerprint="expected-fingerprint",
        analysis_version="version",
        original_session_root="/cluster/session",
        display_label="label",
    )
    sentinel = {"arrays": object()}
    calls: list[tuple[Path, str | None]] = []

    def fake_load(
        directory: Path,
        *,
        expected_run_fingerprint: str | None = None,
    ) -> dict[str, object]:
        calls.append((directory, expected_run_fingerprint))
        return sentinel

    monkeypatch.setattr(view, "load_pooled_run", fake_load)

    assert view.load_selected_condition_generalization_run(option) is sentinel
    assert calls == [(run_directory, "expected-fingerprint")]


def test_empty_view_explains_the_offline_pooled_command(tmp_path: Path) -> None:
    """No-run state identifies the automatic root and metadata workflow command."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    session_root = tmp_path / "session"
    session_root.mkdir()
    messages: list[str] = []

    class EmptyStreamlit:
        """Record only controls reachable in the empty saved-result state."""

        def subheader(self, message: str) -> None:
            messages.append(message)

        def info(self, message: str) -> None:
            messages.append(message)

        def code(self, message: str) -> None:
            messages.append(message)

        def error(self, message: str) -> None:
            raise AssertionError(message)

    view.render_condition_generalization_view(
        EmptyStreamlit(), SimpleNamespace(session_root=session_root)
    )

    combined = " ".join(messages)
    assert "No completed condition-generalization runs" in combined
    assert str(session_root / "analysis_runs") in combined
    assert "pooled.run_session" in combined
    assert "pooled_task_decoding_config.json" in combined


def test_selectors_render_one_saved_figure_and_close_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Display controls map to the existing plotter without recomputation."""
    from src.neural_analysis.webapp import condition_generalization_views as view

    session_root = tmp_path / "session"
    run_directory = (
        session_root
        / "analysis_runs"
        / "pooled_task_variable_decoding_20261012T045944Z"
    )
    _write_completed_run(run_directory)
    calls: list[tuple[str, object]] = []
    figure_numbers: list[int] = []

    class SavedStreamlit:
        """Select transfer output and record the saved-only presentation."""

        def subheader(self, message: str) -> None:
            calls.append(("subheader", message))

        def selectbox(self, label: str, *, options: list[str]) -> str:
            calls.append(("selectbox", (label, tuple(options))))
            selected = {
                "Target": "current_action",
                "Result type": "Reference transfer",
            }
            return selected.get(label, options[0])

        def caption(self, message: str) -> None:
            calls.append(("caption", message))

        def json(self, value: object, **_kwargs: object) -> None:
            calls.append(("json", value))

        def pyplot(self, figure: plt.Figure) -> None:
            figure_numbers.append(figure.number)
            calls.append(("figure", figure))

        def info(self, message: str) -> None:
            raise AssertionError(message)

        def code(self, message: str) -> None:
            raise AssertionError(message)

        def error(self, message: str) -> None:
            raise AssertionError(message)

    def fake_plot(
        saved_result: object,
        *,
        figure_kind: str,
        window_name: str,
        target_name: str,
    ) -> plt.Figure:
        calls.append(
            (
                "plot",
                (saved_result, figure_kind, window_name, target_name),
            )
        )
        return plt.figure()

    monkeypatch.setattr(view, "plot_pooled_timecourses", fake_plot)
    view.render_condition_generalization_view(
        SavedStreamlit(), SimpleNamespace(session_root=session_root)
    )

    plot_calls = [value for name, value in calls if name == "plot"]
    assert len(plot_calls) == 1
    saved_result, figure_kind, window_name, target_name = plot_calls[0]
    assert saved_result["run_fingerprint"] == "run-fingerprint"
    assert (figure_kind, window_name, target_name) == (
        "transfer",
        "pre_choice",
        "current_action",
    )
    assert len(figure_numbers) == 1
    assert not plt.fignum_exists(figure_numbers[0])


def test_condition_generalization_is_available_without_raw_sources(
    tmp_path: Path,
) -> None:
    """The saved view remains available when spike and behavior files are absent."""
    from dataclasses import replace

    from src.neural_analysis.webapp import session_inputs
    from src.tests.neural_analysis.test_lfp_summary_session import (
        _write_resolved_session,
    )

    session = _write_resolved_session(tmp_path)
    session = replace(
        session,
        behavior=replace(session.behavior, trial_table_file=None),
        probes=tuple(
            replace(probe, sorter_directory=None, alignment_file=None)
            for probe in session.probes
        ),
    )

    availability = session_inputs.metadata_view_availability(session)

    assert (
        session_inputs.PLOT_VIEW_CONDITION_GENERALIZATION
        in session_inputs.PLOT_VIEW_OPTIONS
    )
    assert availability[
        session_inputs.PLOT_VIEW_CONDITION_GENERALIZATION
    ].available is True


def test_metadata_route_dispatches_condition_generalization_before_raw_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The condition-generalization selection returns from the early saved route."""
    from src.neural_analysis.webapp import app, session_inputs
    from src.tests.neural_analysis.test_lfp_summary_session import (
        _write_resolved_session,
    )

    session = _write_resolved_session(tmp_path)
    calls: list[tuple[str, object]] = []

    class Sidebar:
        """Select only the condition-generalization result route."""

        def caption(self, message: str) -> None:
            calls.append(("caption", message))

        def selectbox(self, label: str, **_kwargs: object) -> str:
            assert label == "Plot view"
            return session_inputs.PLOT_VIEW_CONDITION_GENERALIZATION

    fake_streamlit = SimpleNamespace(
        sidebar=Sidebar(),
        warning=lambda message: calls.append(("warning", message)),
    )
    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(
        app,
        "render_condition_generalization_view",
        lambda st_module, selected: calls.append(("view", (st_module, selected))),
    )

    assert app._start_metadata_webapp(session) is None
    assert [name for name, _value in calls].count("view") == 1


def test_condition_generalization_saved_view_is_documented() -> None:
    """User documentation names the saved view and its no-recompute behavior."""
    readme = (PACKAGE_ROOT.parent / "README.md").read_text(encoding="utf-8")
    task_readme = (
        PACKAGE_ROOT.parent / "task_decoding" / "README.md"
    ).read_text(encoding="utf-8")

    for text in (readme, task_readme):
        assert "Condition generalization results" in text
        assert "does not recompute" in text
