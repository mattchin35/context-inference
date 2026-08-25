"""Pure-helper and fake-UI contracts for the cached LFP summary webapp route."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.neural_analysis import lfp_summary_webapp
from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    UnitPopulationConfig,
    default_lfp_summary_config,
)


class FakeStreamlit:
    """Record minimal Streamlit output calls without rendering a browser UI."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []
        self.figures: list[plt.Figure] = []

    def info(self, message: str) -> None:
        """Record one informational message."""

        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        """Record one warning message."""

        self.messages.append(("warning", message))

    def error(self, message: str) -> None:
        """Record one error message."""

        self.messages.append(("error", message))

    def pyplot(self, figure: plt.Figure) -> None:
        """Record a displayed Matplotlib figure."""

        self.figures.append(figure)


class FakeSidebar:
    """Return selected control values and record every visible input label."""

    def __init__(self) -> None:
        self.labels: list[str] = []

    def header(self, label: str) -> None:
        """Record a sidebar header."""

        self.labels.append(label)

    def selectbox(self, label: str, *, options: tuple[str, ...]) -> str:
        """Return deterministic nondefault filter/alignment selections."""

        self.labels.append(label)
        selected = {
            "Summary action": "power",
            "Summary view": "power",
            "Choice filter": "left",
            "Context filter": "right",
            "Alignment event": "start_time",
        }
        return selected.get(label, options[0])

    def checkbox(self, label: str, *, value: bool) -> bool:
        """Disable notch and stale inspection while recording the label."""

        self.labels.append(label)
        return False if label == "Apply 60 Hz notch" else value

    def number_input(self, label: str, *, value: float | int, **kwargs: object) -> float | int:
        """Return deterministic advanced values while recording the label."""

        self.labels.append(label)
        selected: dict[str, float | int] = {
            "Phase output rate (Hz)": 400.0,
            "Bootstrap count": 1000,
            "Random seed": 19,
        }
        return selected.get(label, value)

    def text_input(self, label: str, *, value: str) -> str:
        """Return two excluded trial rows while recording the label."""

        self.labels.append(label)
        return "2, 5" if label == "Excluded trial rows" else value

    def button(self, label: str) -> bool:
        """Trigger the synchronous compute action."""

        self.labels.append(label)
        return True


class InteractiveFakeStreamlit(FakeStreamlit):
    """Expose the sidebar protocol used by the full summary route."""

    def __init__(self) -> None:
        super().__init__()
        self.sidebar = FakeSidebar()


def _summary_sites(root: Path) -> tuple[LFPSiteConfig, ...]:
    """Build explicit CT026-style site inputs for configuration assembly tests."""

    return (
        LFPSiteConfig(
            "PFC",
            "PFC",
            "spikeglx",
            root / "pfc.lf.bin",
            root / "pfc.sync.npz",
            "ProbeA",
            5,
            "uV",
            2500.0,
        ),
        LFPSiteConfig(
            "HPC1",
            "HPC1",
            "spikeglx",
            root / "hpc1.lf.bin",
            root / "hpc1.sync.npz",
            "ProbeB",
            222,
            "uV",
            2500.0,
        ),
        LFPSiteConfig(
            "HPC2",
            "HPC2",
            "spikeglx",
            root / "hpc2.lf.bin",
            root / "hpc2.sync.npz",
            "ProbeB",
            14,
            "uV",
            2500.0,
        ),
    )


def _fake_dependencies(
    calls: list[str],
    manifests: list[dict[str, object]],
    *,
    failed: bool = False,
) -> lfp_summary_webapp.SummaryWebDependencies:
    """Build injected compute/cache/plot seams with no raw data loading."""

    def component_result(component: str) -> lfp_summary_webapp.SummaryActionResult:
        calls.append(f"compute_{component}")
        if failed:
            return lfp_summary_webapp.SummaryActionResult(
                component,
                "failed",
                None,
                "write failed",
            )
        return lfp_summary_webapp.SummaryActionResult(component, "complete", None, None)

    def load_manifest(cache_directory: Path) -> dict[str, object]:
        calls.append("load_manifest")
        manifest = {"components": {"power": {}, "synchrony": {}, "spike_phase": {}}}
        manifests.append(manifest)
        return manifest

    def load_component(
        cache_directory: Path,
        manifest: dict[str, object],
        component: str,
    ) -> dict[str, np.ndarray]:
        calls.append(f"load_{component}")
        return {"component_code": np.array([len(component)], dtype=np.int64)}

    def plot_view(
        view: str,
        arrays: dict[str, dict[str, np.ndarray]],
        config: object,
    ) -> plt.Figure:
        """Record a cache-only plot request with its active configuration."""
        del config
        calls.append(f"plot_{view}")
        figure, _axis = plt.subplots()
        return figure

    return lfp_summary_webapp.SummaryWebDependencies(
        compute_power=lambda config: component_result("power"),
        compute_synchrony=lambda config: component_result("synchrony"),
        compute_spike_phase=lambda config: component_result("spike_phase"),
        compute_all=lambda config: component_result("all"),
        load_manifest=load_manifest,
        load_component=load_component,
        plot_view=plot_view,
    )


def test_summary_defaults_match_approved_ct026_channels_and_analysis_settings() -> None:
    """The UI defaults should reproduce the approved first-session configuration."""

    defaults = lfp_summary_webapp.default_summary_ui_values()

    assert defaults.session_id == "CT026_2026-08-01_130853"
    assert defaults.site_channels == {"PFC": 5, "HPC1": 222, "HPC2": 14}
    assert defaults.alignment_event == "choice_time"
    assert defaults.output_rate_hz == 500.0
    assert defaults.notch_enabled
    assert defaults.bootstrap_count == 1000
    assert defaults.choice_filter == "all"
    assert defaults.context_filter == "all"


def test_assemble_summary_config_maps_active_inputs_and_advanced_controls(tmp_path: Path) -> None:
    """Config assembly preserves active paths, identities, filters, and advanced values."""

    population = UnitPopulationConfig(
        "good units",
        "ProbeA",
        tmp_path / "sorter",
        tmp_path / "aligned.npz",
        (1, 2),
        (("group", "good"),),
        ("ProbeA:11", "ProbeA:12"),
    )
    config = lfp_summary_webapp.assemble_summary_config(
        session_id="CT026_2026-08-01_130853",
        session_path=tmp_path,
        output_directory=tmp_path / "processed" / "lfp_summary_cache",
        sites=_summary_sites(tmp_path),
        site_pairs=(("PFC", "HPC1"), ("PFC", "HPC2"), ("HPC1", "HPC2")),
        unit_population=population,
        choice_filter="left",
        context_filter="right",
        excluded_trial_indices=(2, 5),
        alignment_event="start_time",
        notch_enabled=False,
        output_rate_hz=400.0,
        bootstrap_count=1000,
        random_seed=19,
    )

    assert config.session_id == "CT026_2026-08-01_130853"
    assert config.output_directory == tmp_path / "processed" / "lfp_summary_cache"
    assert tuple(site.saved_channel_index for site in config.sites) == (5, 222, 14)
    assert config.unit_population is population
    assert config.trial_filter.choice == "left"
    assert config.trial_filter.context == "right"
    assert config.trial_filter.excluded_trial_indices == (2, 5)
    assert config.analysis_windows.alignment_event == "start_time"
    assert not config.power.notch_enabled
    assert not config.phase.notch_enabled
    assert config.phase.output_rate_hz == 400.0
    assert config.phase.bootstrap_count == 1000
    assert config.phase.seed == 19
    assert config.ppc.seed == 19
    assert config.random_seed == 19


def test_component_actions_call_only_the_selected_injected_entrypoint() -> None:
    """Separate controls must dispatch Power, Synchrony, Spike phase, or All explicitly."""

    config = default_lfp_summary_config()
    for action, expected in (
        ("power", "compute_power"),
        ("synchrony", "compute_synchrony"),
        ("spike_phase", "compute_spike_phase"),
        ("all", "compute_all"),
    ):
        calls: list[str] = []
        dependencies = _fake_dependencies(calls, [])

        result = lfp_summary_webapp.run_summary_action(action, config, dependencies)

        assert result.state == "complete"
        assert calls == [expected, "load_manifest"]


def test_component_status_messages_cover_every_state_and_expose_stale_differences() -> None:
    """Cache state output must not hide stale reasons behind a generic status label."""

    statuses = {
        name: lfp_summary_webapp.component_status_message(
            ComponentStatus(name, ("changed notch",))
        )
        for name in ("missing", "compatible", "stale", "running", "failed")
    }

    for name, message in statuses.items():
        assert name in message.lower()
    assert "changed notch" in statuses["stale"]


def test_stale_results_require_explicit_override_before_loading_or_plotting() -> None:
    """A stale cached component must remain blocked until the user opts in explicitly."""

    stale = ComponentStatus("stale", ("configuration fingerprint differs",))

    with pytest.raises(ValueError, match="stale"):
        lfp_summary_webapp.require_renderable_component(stale, allow_stale=False)
    lfp_summary_webapp.require_renderable_component(stale, allow_stale=True)


def test_success_reloads_cache_while_failure_retains_prior_result() -> None:
    """Successful actions reload the saved cache; failures keep the previous valid object."""

    config = default_lfp_summary_config()
    prior_manifest = {"components": {"power": {"state": "complete"}}}
    success_calls: list[str] = []
    success = lfp_summary_webapp.run_summary_action(
        "power",
        config,
        _fake_dependencies(success_calls, []),
        previous_manifest=prior_manifest,
    )
    failure_calls: list[str] = []
    failure = lfp_summary_webapp.run_summary_action(
        "power",
        config,
        _fake_dependencies(failure_calls, [], failed=True),
        previous_manifest=prior_manifest,
    )

    assert success.manifest is not prior_manifest
    assert success_calls == ["compute_power", "load_manifest"]
    assert failure.manifest is prior_manifest
    assert failure.error == "write failed"
    assert failure_calls == ["compute_power"]


def test_summary_view_loads_only_the_component_required_by_its_selector() -> None:
    """Selectors should not load unrelated Power, Synchrony, or Spike-phase NPZ files."""

    config = default_lfp_summary_config()
    manifest = {"components": {"power": {}, "synchrony": {}, "spike_phase": {}}}
    for view, required_component in (
        ("power", "power"),
        ("synchrony", "synchrony"),
        ("spike_phase", "spike_phase"),
    ):
        calls: list[str] = []
        dependencies = _fake_dependencies(calls, [])

        arrays = lfp_summary_webapp.load_summary_view_arrays(
            view,
            config,
            manifest,
            dependencies,
        )

        assert set(arrays) == {required_component}
        assert calls == [f"load_{required_component}"]


def test_count_and_instability_messages_remain_visible_in_rendered_summary() -> None:
    """Trial, unit, spike counts and low-count instability must reach the UI output."""

    streamlit = FakeStreamlit()
    lfp_summary_webapp.render_summary_counts(
        streamlit,
        trial_count=7,
        unit_count=3,
        spike_count=41,
        unstable=True,
    )

    text = " ".join(message for _kind, message in streamlit.messages)
    assert "7" in text
    assert "3" in text
    assert "41" in text
    assert "unstable" in text.lower()


def test_summary_figure_renderer_closes_matplotlib_figure_after_streamlit_display() -> None:
    """Rendered cached-summary figures must not accumulate open Matplotlib handles."""

    streamlit = FakeStreamlit()
    figure, _axis = plt.subplots()
    figure_number = figure.number

    lfp_summary_webapp.render_summary_figure(streamlit, figure)

    assert streamlit.figures == [figure]
    assert not plt.fignum_exists(figure_number)


def test_full_summary_route_maps_visible_controls_into_computation_config(
    tmp_path: Path,
) -> None:
    """The route must not silently replace visible filters with hard-coded defaults."""

    streamlit = InteractiveFakeStreamlit()
    received_configs = []
    dependencies = _fake_dependencies([], [])
    dependencies = replace(
        dependencies,
        compute_power=lambda config: (
            received_configs.append(config)
            or lfp_summary_webapp.SummaryActionResult("power", "complete", None, None)
        ),
    )

    lfp_summary_webapp.render_lfp_summary_view(
        streamlit,
        session_id="CT026_2026-08-01_130853",
        session_path=tmp_path,
        output_directory=tmp_path / "processed" / "lfp_summary_cache",
        sites=_summary_sites(tmp_path),
        site_pairs=(("PFC", "HPC1"),),
        dependencies=dependencies,
    )

    config = received_configs[0]
    assert config.trial_filter.choice == "left"
    assert config.trial_filter.context == "right"
    assert config.trial_filter.excluded_trial_indices == (2, 5)
    assert config.analysis_windows.alignment_event == "start_time"
    assert not config.power.notch_enabled
    assert not config.phase.notch_enabled
    assert config.phase.output_rate_hz == 400.0
    assert config.phase.seed == 19 and config.ppc.seed == 19
    assert config.trial_table_path == (
        tmp_path
        / "processed"
        / "CT026_2026-08-01_130853_augmented_trials.csv"
    )
    assert {
        "Choice filter",
        "Context filter",
        "Excluded trial rows",
        "Alignment event",
        "Apply 60 Hz notch",
        "Phase output rate (Hz)",
        "Bootstrap count",
        "Random seed",
    }.issubset(streamlit.sidebar.labels)


def test_production_power_action_delegates_to_atomic_runtime_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production Power dispatch must use the pipeline and atomic runtime I/O seams.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces production bridge construction and pipeline computation without
        opening LFP files or writing a cache.
    """
    config = default_lfp_summary_config()
    calls: list[object] = []
    sentinel_dependencies = object()

    def fake_runtime_dependencies(**kwargs: object) -> object:
        """Record the runtime factory call and return an opaque dependency object."""
        calls.append(kwargs)
        return sentinel_dependencies

    def fake_compute(active_config: object, dependencies: object) -> object:
        """Record pipeline inputs and return a successful pipeline-like result."""
        calls.append((active_config, dependencies))
        return type(
            "Result",
            (),
            {"component": "power", "state": "complete", "error": None},
        )()

    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_runtime,
        "make_power_pipeline_dependencies",
        fake_runtime_dependencies,
    )
    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_pipeline,
        "compute_power_component",
        fake_compute,
    )

    result = lfp_summary_webapp.compute_production_power(config)

    assert result.component == "power"
    assert result.state == "complete"
    assert calls[-1] == (config, sentinel_dependencies)
    assert callable(calls[0]["trial_table_loader"])


def test_production_dependencies_load_cached_power_without_preparing_raw_lfp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached Power selection should read only ``power.npz`` instead of raw LFP.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fails if raw Power preparation is reached and records cache-array loads.
    """
    config = default_lfp_summary_config()
    calls: list[str] = []

    def fail_prepare(*args: object, **kwargs: object) -> object:
        """Fail if a cached view attempts to reopen native LFP input."""
        del args, kwargs
        raise AssertionError("cached Power view reopened raw LFP")

    def fake_load(
        path: Path,
        manifest: dict[str, object],
        component: str,
    ) -> dict[str, np.ndarray]:
        """Record the cache component request and return one safe numeric array."""
        del path, manifest
        calls.append(component)
        return {"trial_indices": np.array([0], dtype=np.int64)}

    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_runtime,
        "prepare_power_run",
        fail_prepare,
    )
    monkeypatch.setattr(lfp_summary_webapp, "load_component_arrays", fake_load)
    dependencies = lfp_summary_webapp.make_production_summary_dependencies()

    arrays = lfp_summary_webapp.load_summary_view_arrays(
        "power",
        config,
        {"components": {"power": {}}},
        dependencies,
    )

    assert calls == ["power"]
    assert arrays["power"]["trial_indices"].tolist() == [0]


def test_production_power_plot_delegates_cache_arrays_to_plotting_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The web boundary should adapt cached axes, not implement Matplotlib views.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the pure plotting function and records its cache-derived inputs.
    """
    config = replace(
        default_lfp_summary_config(),
        trial_filter=lfp_summary_webapp.TrialFilterConfig(choice="left"),
    )
    normalized_psd = np.arange(24, dtype=float).reshape(1, 2, 3, 4)
    arrays = {
        "frequency_hz": np.array((0.0, 40.0, 100.0, 102.0)),
        "normalized_psd_session_db": normalized_psd,
        "condition_names": np.array(("correct_rewarded", "omission")),
        "condition_membership": np.array(((True, False), (True, True))),
        "filter_membership": np.array((True, False)),
        "condition_effective_trial_count": np.array(((1,), (1,))),
        "site_ids": np.array(("PFC",)),
        "site_voltage_units": np.array(("uV",)),
        "epoch_names": np.array(("whole", "before", "after")),
    }
    captured: dict[str, object] = {}
    expected_figure, _axis = plt.subplots()

    def fake_plot(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture the plotting adapter's positional and keyword arguments."""
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected_figure, {}

    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_plotting,
        "plot_condition_psd",
        fake_plot,
    )
    dependencies = lfp_summary_webapp.make_production_summary_dependencies()

    figure = dependencies.plot_view("power", {"power": arrays}, config)

    assert figure is expected_figure
    args = captured["args"]
    assert isinstance(args, tuple)
    assert np.array_equal(args[0], np.array((0.0, 40.0, 100.0)))
    condition_psd = args[1]
    assert isinstance(condition_psd, np.ndarray)
    assert np.array_equal(condition_psd[0, 0], normalized_psd[0, 0, 0, :3])
    assert np.isnan(condition_psd[0, 1]).all()
    assert np.isnan(condition_psd[1, 0]).all()
    assert np.isnan(condition_psd[1, 1]).all()
    context = args[-1]
    assert context.session_id == config.session_id
    assert context.alignment_event == config.analysis_windows.alignment_event
    assert context.source_voltage_unit == "uV"


def test_cached_power_default_plot_uses_only_the_base_five_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The compact cached Power view must not silently render all nine groups.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Captures pure plotting input without raw LFP loading.
    """
    group_names = np.array(
        (
            "correct_rewarded",
            "omission",
            "incorrect",
            "switch",
            "stay",
            "omission_switch",
            "omission_stay",
            "incorrect_switch",
            "incorrect_stay",
        )
    )
    arrays = {
        "frequency_hz": np.array((8.0,)),
        "normalized_psd_session_db": np.ones((1, 1, 3, 1), dtype=float),
        "condition_names": group_names,
        "condition_membership": np.ones((1, group_names.size), dtype=bool),
        "filter_membership": np.array((True,)),
        "condition_effective_trial_count": np.ones(
            (group_names.size, 1),
            dtype=np.int64,
        ),
        "site_ids": np.array(("PFC",)),
        "site_voltage_units": np.array(("uV",)),
        "epoch_names": np.array(("whole", "before", "after")),
    }
    captured: dict[str, object] = {}

    def fake_plot(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture cache-derived condition names and return an unsaved figure."""
        captured["condition_names"] = args[2]
        return plt.figure(), {}

    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_plotting,
        "plot_condition_psd",
        fake_plot,
    )
    dependencies = lfp_summary_webapp.make_production_summary_dependencies()

    dependencies.plot_view("power", {"power": arrays}, default_lfp_summary_config())

    assert captured["condition_names"] == (
        "correct_rewarded",
        "omission",
        "incorrect",
        "switch",
        "stay",
    )


def test_full_route_assesses_fingerprint_compatibility_before_cached_render(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A manifest ``complete`` flag alone must not bypass stale-config checks.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces compatibility assessment with a visible stale difference.
    tmp_path : pathlib.Path
        Temporary active session root used by configuration assembly.
    """
    streamlit = InteractiveFakeStreamlit()
    streamlit.sidebar.button = lambda _label: False
    calls: list[str] = []
    dependencies = _fake_dependencies(calls, [])
    dependencies = replace(
        dependencies,
        load_manifest=lambda _path: {
            "components": {"power": {"state": "complete"}}
        },
    )

    def fake_assess(*args: object, **kwargs: object) -> ComponentStatus:
        """Return one stale fingerprint difference without touching cache files."""
        del args, kwargs
        return ComponentStatus("stale", ("trial_filter.choice: all -> left",))

    monkeypatch.setattr(
        lfp_summary_webapp,
        "assess_component_status",
        fake_assess,
    )

    lfp_summary_webapp.render_lfp_summary_view(
        streamlit,
        session_id="CT026_2026-08-01_130853",
        session_path=tmp_path,
        output_directory=tmp_path / "processed" / "lfp_summary_cache",
        sites=_summary_sites(tmp_path),
        site_pairs=(("PFC", "HPC1"),),
        dependencies=dependencies,
    )

    messages = " ".join(message for _kind, message in streamlit.messages)
    assert "stale" in messages
    assert "trial_filter.choice" in messages
    assert "load_power" not in calls


@pytest.mark.parametrize("action", ("synchrony", "spike_phase", "all"))
def test_production_dependencies_report_unavailable_nonpower_actions(action: str) -> None:
    """Unavailable Synchrony/Spike actions must fail explicitly without fabricated results.

    Parameters
    ----------
    action : str
        Unsupported production action selected by pytest parametrization.
    """
    config = default_lfp_summary_config()
    dependencies = lfp_summary_webapp.make_production_summary_dependencies()

    result = lfp_summary_webapp.run_summary_action(action, config, dependencies)

    assert result.state == "failed"
    assert result.manifest is None
    assert result.error is not None
    assert "unavailable" in result.error.lower()
