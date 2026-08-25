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

    def plot_view(view: str, arrays: dict[str, dict[str, np.ndarray]]) -> plt.Figure:
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
