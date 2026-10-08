"""Ownership and compatibility contracts for the neural-analysis webapp package."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from src.neural_analysis import psth_webapp as legacy_webapp
from src.tests.neural_analysis.task_decoding.test_results import (
    save_run_fixture,
    write_input_fixture,
)


PACKAGE_ROOT = Path(__file__).parents[2] / "neural_analysis" / "webapp"


def _canonical_module(module_name: str):
    """Import one canonical webapp module by its short, unitless module name."""
    return importlib.import_module(f"src.neural_analysis.webapp.{module_name}")


def _assert_exact_owner(module_name: str, object_names: tuple[str, ...]) -> None:
    """Require legacy exports to be the exact objects defined by one canonical owner."""
    canonical = _canonical_module(module_name)
    for object_name in object_names:
        canonical_object = getattr(canonical, object_name)
        assert getattr(legacy_webapp, object_name) is canonical_object
        assert canonical_object.__module__ == canonical.__name__


def test_root_webapp_import_is_the_canonical_app_module() -> None:
    """The documented root import remains one live module with the canonical app owner."""
    app = _canonical_module("app")

    assert legacy_webapp is app
    assert app.parse_webapp_arguments.__module__ == app.__name__
    assert app.main.__module__ == app.__name__


def test_session_input_helpers_have_one_canonical_owner() -> None:
    """Version-2 session resolution and channel selection live in session_inputs."""
    _assert_exact_owner(
        "session_inputs",
        (
            "MetadataLFPSiteInputs",
            "MetadataViewAvailability",
            "load_webapp_session",
            "metadata_lfp_site_inputs",
            "metadata_view_availability",
            "resolve_region_channels_for_source",
        ),
    )


def test_compatibility_only_population_adapter_is_not_made_canonical() -> None:
    """The obsolete population-shaped adapter stays only at the compatibility entry point."""
    session_inputs = _canonical_module("session_inputs")
    app = _canonical_module("app")

    assert not hasattr(session_inputs, "MetadataPopulationInputs")
    assert not hasattr(session_inputs, "metadata_population_inputs")
    assert app.MetadataPopulationInputs is legacy_webapp.MetadataPopulationInputs
    assert app.metadata_population_inputs is legacy_webapp.metadata_population_inputs


def test_cached_loading_helpers_have_one_canonical_owner() -> None:
    """File loading and Streamlit-cached source access live in data_loading."""
    _assert_exact_owner(
        "data_loading",
        (
            "OpenEphysCacheToken",
            "build_open_ephys_cache_token",
            "load_viewer_data_cached",
            "load_phase_clustering_session_cached",
            "load_metadata_behavior_session_cached",
            "load_metadata_viewer_data_cached",
            "load_channel_quality_cached",
            "load_cluster_info_cached",
            "load_summary_cluster_metadata",
            "load_summary_channel_metadata",
            "decode_lfp_sync_cached",
            "load_trial_lfp_trace_cached",
            "load_trial_lfp_trace_for_format",
            "load_trial_lfp_trace_for_format_with_sample_rate",
        ),
    )


def test_lfp_view_helpers_have_one_canonical_owner() -> None:
    """LFP computations, controls, and rendered views live together in lfp_views."""
    _assert_exact_owner(
        "lfp_views",
        (
            "compute_trial_lfp_spectrogram_cached",
            "compute_shared_lfp_power_limits_cached",
            "compute_lfp_phase_site_cached",
            "compute_single_trial_relative_phase_cached",
            "render_single_trial_relative_phase_view",
            "render_lfp_phase_clustering_view",
            "build_lfp_dropdown_options",
            "resolve_lfp_view_sites",
            "resolve_lfp_filter_band",
        ),
    )


def test_metadata_lfp_sites_bypass_the_legacy_two_probe_bridge(tmp_path: Path) -> None:
    """Arbitrary probe/site identities reach LFP views without PFC/HPC translation."""
    lfp_views = _canonical_module("lfp_views")
    session_inputs = _canonical_module("session_inputs")
    front = session_inputs.MetadataLFPSiteInputs(
        site_id="cortex-contact",
        display_label="Cortex contact",
        probe_id="front-probe",
        acquisition_family="open_ephys",
        saved_channel_index=11,
        lfp_file=tmp_path / "front.lfp",
        synchronization_file=tmp_path / "front_sync.npz",
    )
    rear = session_inputs.MetadataLFPSiteInputs(
        site_id="depth-contact",
        display_label="Depth contact",
        probe_id="rear-probe",
        acquisition_family="open_ephys",
        saved_channel_index=23,
        lfp_file=tmp_path / "rear.lfp",
        synchronization_file=tmp_path / "rear_sync.npz",
    )

    resolved = lfp_views.resolve_lfp_view_sites(
        metadata_sites=(front, rear),
        hpc_v1_lfp_path="legacy-hpc",
        pfc_lfp_path="legacy-pfc",
        hpc_v1_aligned_spike_path="legacy-hpc-sync",
        pfc_aligned_spike_path="legacy-pfc-sync",
    )

    assert resolved == (front, rear)
    app_source = (PACKAGE_ROOT / "app.py").read_text(encoding="utf-8")
    assert "first_probe" not in app_source
    assert "second_probe" not in app_source


def test_lfp_site_default_preserves_manual_labels_and_matches_metadata_probe_ids(
    tmp_path: Path,
) -> None:
    """Site defaults keep legacy PFC selection while metadata uses exact probe ids."""
    lfp_views = _canonical_module("lfp_views")
    spike_lfp_views = _canonical_module("spike_lfp_views")
    session_inputs = _canonical_module("session_inputs")
    legacy_sites = lfp_views.resolve_lfp_view_sites(
        hpc_v1_lfp_path="hpc.lf.bin",
        pfc_lfp_path="pfc.lf.bin",
    )
    metadata_sites = (
        session_inputs.MetadataLFPSiteInputs(
            "front-site", "Front", "front-probe", "open_ephys", 1,
            tmp_path / "front.lfp", tmp_path / "front_sync.npz",
        ),
        session_inputs.MetadataLFPSiteInputs(
            "rear-site", "Rear", "rear-probe", "open_ephys", 2,
            tmp_path / "rear.lfp", tmp_path / "rear_sync.npz",
        ),
    )

    assert spike_lfp_views._default_lfp_site_index(legacy_sites, "HPC/V1") == 0
    assert spike_lfp_views._default_lfp_site_index(legacy_sites, "PFC") == 1
    assert spike_lfp_views._default_lfp_site_index(metadata_sites, "rear-probe") == 1


def test_spike_lfp_view_helpers_have_one_canonical_owner() -> None:
    """Spike-LFP cached calculations and rendered views share one direct owner."""
    _assert_exact_owner(
        "spike_lfp_views",
        (
            "compute_spike_lfp_phase_locking_cached",
            "compute_single_trial_spike_lfp_hilbert_cached",
            "render_spike_lfp_phase_locking_view",
            "render_single_trial_spike_lfp_hilbert_view",
        ),
    )


def test_population_view_helpers_have_one_canonical_owner() -> None:
    """Population display selection and cached PCA calculations live together."""
    _assert_exact_owner(
        "population_views",
        (
            "is_population_pca_display",
            "resolve_pca_decoding_component_minimum",
            "select_visible_concatenated_trial_indices",
            "select_concatenated_trial_viewport",
            "resolve_population_pca_unit_ids",
            "filter_trial_indices_for_valid_alignment",
            "compute_population_pca_cached",
            "compute_population_pca_decoding_cached",
            "compute_population_pca_switch_trajectories_cached",
        ),
    )


def test_unit_and_summary_helpers_have_direct_canonical_owners() -> None:
    """Unit presentation and summary composition are not retained in the app module."""
    _assert_exact_owner(
        "unit_views",
        (
            "build_trial_view_plot_save_path",
            "format_metadata_row_for_display",
        ),
    )
    _assert_exact_owner(
        "summary_view",
        (
            "render_metadata_lfp_summary_view",
            "render_lfp_summary_view",
        ),
    )


def test_webapp_modules_do_not_import_the_legacy_root_module() -> None:
    """Canonical modules form a shallow package without a cycle through psth_webapp."""
    required_modules = {
        "app.py",
        "data_loading.py",
        "lfp_views.py",
        "population_views.py",
        "session_inputs.py",
        "spike_lfp_views.py",
        "summary_view.py",
        "unit_views.py",
    }
    actual_modules = {path.name for path in PACKAGE_ROOT.glob("*.py")}
    assert required_modules | {"__init__.py"} <= actual_modules

    for path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "src.neural_analysis.psth_webapp" not in imported_modules


def test_root_module_keeps_direct_streamlit_dispatch() -> None:
    """Running the documented root module still delegates argv to the canonical main."""
    source = (PACKAGE_ROOT.parent / "psth_webapp.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(isinstance(child, ast.Name) and child.id == "__name__" for child in ast.walk(node.test))
        for node in tree.body
    )


def _mark_complete(run_directory: Path) -> None:
    """Write the small lifecycle record required for completed-run discovery.

    Parameters
    ----------
    run_directory : pathlib.Path
        Synthetic saved-result directory.

    Returns
    -------
    None
        Writes only a pytest-owned ``run_state.json`` file.
    """
    (run_directory / "run_state.json").write_text(
        json.dumps({"lifecycle": "complete", "final_results_published": True}),
        encoding="ascii",
    )


def test_task_decoding_view_has_one_saved_result_only_owner() -> None:
    """Discovery, rendering, and coefficient presentation share one canonical owner."""
    task_view = _canonical_module("task_decoding_views")

    for object_name in (
        "resolve_task_decoding_results_root",
        "discover_task_decoding_runs",
        "render_task_decoding_view",
    ):
        assert getattr(task_view, object_name).__module__ == task_view.__name__
    source = (PACKAGE_ROOT / "task_decoding_views.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "src.neural_analysis.task_decoding.results" in imported_modules
    assert not any(
        name.endswith(
            ("task_decoding.modeling", "task_decoding.pipeline", "task_decoding.activity")
        )
        for name in imported_modules
    )


def test_task_decoding_results_root_is_session_relative_and_contained(tmp_path: Path) -> None:
    """The display locator cannot traverse or resolve outside the selected session."""
    task_view = _canonical_module("task_decoding_views")
    session_root = tmp_path / "session"
    session_root.mkdir()

    assert task_view.resolve_task_decoding_results_root(session_root, "analysis_runs") == (
        session_root / "analysis_runs"
    ).resolve()
    assert task_view.resolve_task_decoding_results_root(session_root, "reports/decoding") == (
        session_root / "reports" / "decoding"
    ).resolve()
    for locator in ("../outside", str(tmp_path / "absolute"), "."):
        with pytest.raises(ValueError, match="session|relative|root"):
            task_view.resolve_task_decoding_results_root(session_root, locator)


def test_discovery_lists_only_direct_valid_completed_children(tmp_path: Path) -> None:
    """Hidden, nested, incomplete, and corrupt runs never enter the selector."""
    task_view = _canonical_module("task_decoding_views")
    inputs = write_input_fixture(tmp_path)
    results_root = inputs["session_root"] / "analysis_runs"
    valid = results_root / "20261007_120000_task_decoding"
    save_run_fixture(valid, inputs, regularization_mode="fixed")
    _mark_complete(valid)
    hidden = results_root / ".incoming-transfer"
    save_run_fixture(hidden, inputs, regularization_mode="fixed")
    _mark_complete(hidden)
    incomplete = results_root / "incomplete"
    incomplete.mkdir()
    (incomplete / "run_state.json").write_text(
        json.dumps({"lifecycle": "running", "final_results_published": False}),
        encoding="ascii",
    )
    corrupt = results_root / "corrupt"
    corrupt.mkdir()
    _mark_complete(corrupt)
    nested = results_root / "group" / "nested-complete"
    save_run_fixture(nested, inputs, regularization_mode="fixed")
    _mark_complete(nested)

    discovered = task_view.discover_task_decoding_runs(inputs["session_root"])

    assert [item.run_directory for item in discovered] == [valid.resolve()]
    assert discovered[0].run_fingerprint == "run-fingerprint-test"


def test_discovery_rejects_a_direct_child_symlink_resolving_outside_session(
    tmp_path: Path,
) -> None:
    """A lexical direct child cannot bypass session containment through a symlink."""
    task_view = _canonical_module("task_decoding_views")
    inputs = write_input_fixture(tmp_path / "inside")
    outside_inputs = write_input_fixture(tmp_path / "outside")
    outside_run = outside_inputs["session_root"] / "external-complete"
    save_run_fixture(outside_run, outside_inputs, regularization_mode="fixed")
    _mark_complete(outside_run)
    results_root = inputs["session_root"] / "analysis_runs"
    results_root.mkdir()
    (results_root / "linked-complete").symlink_to(outside_run, target_is_directory=True)

    assert task_view.discover_task_decoding_runs(inputs["session_root"]) == ()


def test_task_decoding_view_explains_the_no_saved_run_state(tmp_path: Path) -> None:
    """An empty contained results root gives offline-run guidance without raw loading."""
    task_view = _canonical_module("task_decoding_views")
    session_root = tmp_path / "session"
    session_root.mkdir()
    messages: list[str] = []

    class EmptyViewStreamlit:
        """Expose only controls reachable before empty-run discovery returns."""

        def subheader(self, message: str) -> None:
            messages.append(message)

        def text_input(self, _label: str, *, value: str) -> str:
            return value

        def info(self, message: str) -> None:
            messages.append(message)

        def error(self, message: str) -> None:
            raise AssertionError(message)

    task_view.render_task_decoding_view(
        EmptyViewStreamlit(),
        SimpleNamespace(session_root=session_root),
    )

    combined = " ".join(messages)
    assert "No completed task-variable decoding runs" in combined
    assert str(session_root / "analysis_runs") in combined
    assert "run_session" in combined
    assert "offline" in combined.lower()


def test_task_decoding_selectors_only_render_the_completed_saved_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Display changes call saved plotting/table paths without any fitting entry point."""
    task_view = _canonical_module("task_decoding_views")
    inputs = write_input_fixture(tmp_path)
    run_directory = inputs["session_root"] / "analysis_runs" / "complete"
    save_run_fixture(run_directory, inputs, regularization_mode="fixed")
    _mark_complete(run_directory)
    calls: list[tuple[str, object]] = []

    class SavedViewStreamlit:
        """Choose deterministic nondefault display controls and record outputs."""

        def subheader(self, message: str) -> None:
            calls.append(("subheader", message))

        def text_input(self, _label: str, *, value: str) -> str:
            return value

        def selectbox(self, label: str, *, options: list[str]) -> str:
            calls.append(("selectbox", (label, tuple(options))))
            selected = {
                "Categorical metric": "auc",
                "Condition": "all",
                "Region": "PFC",
                "Representation": "units",
                "Target": "current_action",
            }
            if label == "Time bin":
                return options[1]
            return selected.get(label, options[0])

        def caption(self, message: str) -> None:
            calls.append(("caption", message))

        def json(self, value: object, **_kwargs: object) -> None:
            calls.append(("json", value))

        def pyplot(self, figure: plt.Figure) -> None:
            calls.append(("figure", figure))

        def dataframe(self, table: pd.DataFrame, **_kwargs: object) -> None:
            calls.append(("table", table.copy()))

        def info(self, message: str) -> None:
            raise AssertionError(message)

        def error(self, message: str) -> None:
            raise AssertionError(message)

    def fake_heatmap(
        _saved_run: object,
        *,
        family: str,
        metric: str,
        condition: str,
    ) -> plt.Figure:
        calls.append(("heatmap", (condition, family, metric)))
        return plt.figure()

    def fake_summary(
        _saved_run: object,
        *,
        target: str,
        region: str,
        time_bin_index: int,
        condition: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        calls.append(
            ("coefficient-summary", (condition, target, region, time_bin_index))
        )
        return pd.DataFrame({"feature_id": ["probe-pfc:11"]}), pd.DataFrame(
            {"outer_fold": [0]}
        )

    def fake_coefficients(
        _saved_run: object,
        *,
        target: str,
        region: str,
        time_bin_index: int,
        condition: str,
    ) -> plt.Figure:
        calls.append(("coefficients", (condition, target, region, time_bin_index)))
        return plt.figure()

    monkeypatch.setattr(task_view, "plot_decoding_heatmap", fake_heatmap)
    monkeypatch.setattr(task_view, "summarize_unit_coefficients", fake_summary)
    monkeypatch.setattr(task_view, "plot_unit_coefficients", fake_coefficients)

    task_view.render_task_decoding_view(
        SavedViewStreamlit(),
        SimpleNamespace(session_root=inputs["session_root"]),
    )

    assert ("selectbox", ("Condition", ("all",))) in calls
    assert ("heatmap", ("all", "categorical", "auc")) in calls
    assert ("heatmap", ("all", "numerical", "r2")) in calls
    assert ("coefficient-summary", ("all", "current_action", "PFC", 1)) in calls
    assert ("coefficients", ("all", "current_action", "PFC", 1)) in calls
    assert sum(name == "json" for name, _value in calls) == 1
    assert sum(name == "table" for name, _value in calls) == 3


def test_metadata_keeps_saved_result_view_available_without_raw_sources(tmp_path: Path) -> None:
    """Saved inspection remains selectable when live spike and behavior inputs disappear."""
    from dataclasses import replace
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session_inputs = _canonical_module("session_inputs")
    session = _write_resolved_session(tmp_path)
    session = replace(
        session,
        behavior=replace(
            session.behavior,
            trial_table_file=None,
        ),
        probes=tuple(
            replace(probe, sorter_directory=None, alignment_file=None)
            for probe in session.probes
        ),
    )

    availability = session_inputs.metadata_view_availability(session)

    assert session_inputs.PLOT_VIEW_TASK_DECODING in session_inputs.PLOT_VIEW_OPTIONS
    assert availability[session_inputs.PLOT_VIEW_TASK_DECODING].available is True


def test_metadata_route_renders_saved_results_before_population_or_raw_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Selecting the saved view returns before population controls and raw loaders."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    app = _canonical_module("app")
    session_inputs = _canonical_module("session_inputs")
    session = _write_resolved_session(tmp_path)
    calls: list[tuple[str, object]] = []

    class Sidebar:
        """Minimal metadata sidebar selecting only the WP7 saved-results route."""

        def button(self, *_args: object, **_kwargs: object) -> bool:
            return False

        def caption(self, message: str) -> None:
            calls.append(("caption", message))

        def selectbox(self, label: str, **_kwargs: object) -> str:
            assert label == "Plot view"
            return session_inputs.PLOT_VIEW_TASK_DECODING

    fake_streamlit = SimpleNamespace(
        sidebar=Sidebar(),
        set_page_config=lambda **kwargs: calls.append(("page", kwargs)),
        title=lambda value: calls.append(("title", value)),
        cache_resource=SimpleNamespace(clear=lambda: None),
        cache_data=SimpleNamespace(clear=lambda: None),
        rerun=lambda: None,
        warning=lambda message: calls.append(("warning", message)),
    )

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("saved-results routing must precede raw/population loading")

    monkeypatch.setattr(app, "st", fake_streamlit)
    monkeypatch.setattr(app, "parse_webapp_arguments", lambda _argv: SimpleNamespace(
        session_metadata=tmp_path / "neural_session.json"
    ))
    monkeypatch.setattr(app, "load_webapp_session", lambda _path: session)
    monkeypatch.setattr(
        app,
        "render_task_decoding_view",
        lambda st_module, selected: calls.append(("task-view", (st_module, selected))),
    )
    monkeypatch.setattr(app, "_metadata_population_controls", forbidden)
    monkeypatch.setattr(app, "load_metadata_viewer_data_cached", forbidden)

    app.main(())

    assert [name for name, _value in calls].count("task-view") == 1
