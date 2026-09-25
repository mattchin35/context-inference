from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_phase_clustering, psth_webapp
from src.neural_analysis import lfp_summary_webapp
from src.neural_analysis.lfp_summary_models import component_fingerprint


def _population_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return small Probe metadata tables for active-population selection tests.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Cluster metadata has one categorical cluster id, zero-based channel, and
        quality group per row. Channel metadata has zero-based channel labels
        and boolean inside-brain flags. Neither table contains spike times,
        LFP values, physical units, or session paths.
    """

    clusters = pd.DataFrame(
        {
            "cluster_id": [7, 8, 9, 10],
            "ch": [1, 1, 2, 3],
            "group": ["good", "mua", "noise", "good"],
        }
    )
    channels = pd.DataFrame(
        {
            "channel": [1, 2, 3],
            "channel_quality": ["good", "good", "good"],
            "inside_brain": [True, False, True],
        }
    )
    return clusters, channels


@pytest.mark.parametrize("probe_label", ("ProbeA", "ProbeB"))
def test_active_summary_population_uses_one_probe_with_good_mua_good_inside_brain_defaults(
    tmp_path: Path,
    probe_label: str,
) -> None:
    """Each active selector choice creates one probe-qualified, never-combined population.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root used only for distinct sorter and aligned-spike identities.
    probe_label : str
        Exact active population selector value, either ProbeA or ProbeB.
    """

    clusters, channels = _population_metadata()
    sorter_path = tmp_path / probe_label / "kilosort4"
    aligned_spike_path = tmp_path / probe_label / "aligned_spikes.npz"

    population = lfp_summary_webapp.build_active_summary_population(
        probe_label=probe_label,
        sorter_path=sorter_path,
        aligned_spike_path=aligned_spike_path,
        cluster_metadata=clusters,
        channel_metadata=channels,
    )

    assert population.probe_label == probe_label
    assert population.sorter_path == sorter_path
    assert population.aligned_spike_path == aligned_spike_path
    assert population.selected_channels == (1, 3)
    assert population.stable_unit_ids == (f"{probe_label}:7", f"{probe_label}:8", f"{probe_label}:10")
    assert population.quality_settings == (
        ("channel_quality", "good"),
        ("inside_brain", "true"),
        ("unit_quality", "good,mua"),
    )
    assert all(unit_id.startswith(f"{probe_label}:") for unit_id in population.stable_unit_ids)


def test_active_summary_population_rejects_empty_probe_identity(
    tmp_path: Path,
) -> None:
    """The summary selector requires one nonempty metadata-defined probe ID."""

    clusters, channels = _population_metadata()

    with pytest.raises(ValueError, match="nonempty"):
        lfp_summary_webapp.build_active_summary_population(
            probe_label="",
            sorter_path=tmp_path / "sorter",
            aligned_spike_path=tmp_path / "aligned.npz",
            cluster_metadata=clusters,
            channel_metadata=channels,
        )


def test_switching_active_probe_changes_spike_phase_fingerprint_without_combining_paths(
    tmp_path: Path,
) -> None:
    """Probe choice must remain part of the Spike-phase identity and source provenance.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root supplying deliberately distinct ProbeA and ProbeB paths.
    """

    clusters, channels = _population_metadata()
    populations = {
        probe: lfp_summary_webapp.build_active_summary_population(
            probe_label=probe,
            sorter_path=tmp_path / probe / "kilosort4",
            aligned_spike_path=tmp_path / probe / "aligned_spikes.npz",
            cluster_metadata=clusters,
            channel_metadata=channels,
        )
        for probe in ("ProbeA", "ProbeB")
    }
    base = lfp_summary_webapp.default_lfp_summary_config()

    fingerprint_a = component_fingerprint(
        "spike_phase",
        replace(base, unit_population=populations["ProbeA"]),
    )
    fingerprint_b = component_fingerprint(
        "spike_phase",
        replace(base, unit_population=populations["ProbeB"]),
    )

    assert fingerprint_a != fingerprint_b
    assert populations["ProbeA"].sorter_path != populations["ProbeB"].sorter_path
    assert populations["ProbeA"].aligned_spike_path != populations["ProbeB"].aligned_spike_path


def test_webapp_keeps_existing_routes_and_exposes_lfp_summary_route():
    """The new cached-summary entry must coexist with every established neural route."""

    expected_routes = {
        psth_webapp.PLOT_VIEW_UNIT_RASTER,
        psth_webapp.PLOT_VIEW_TRIAL_SPIKES,
        psth_webapp.PLOT_VIEW_PCA_DECODING,
        psth_webapp.PLOT_VIEW_PCA_SWITCH_TRAJECTORIES,
        psth_webapp.PLOT_VIEW_LFP_PHASE_CLUSTERING,
        psth_webapp.PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE,
        psth_webapp.PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING,
        psth_webapp.PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT,
        psth_webapp.PLOT_VIEW_LFP_SUMMARY,
    }

    assert expected_routes.issubset(psth_webapp.PLOT_VIEW_OPTIONS)
    assert callable(psth_webapp.render_lfp_summary_view)


def test_webapp_argument_parser_accepts_one_explicit_session_metadata_path(tmp_path: Path) -> None:
    """The Streamlit script argument is explicit and has no CT-specific default."""
    metadata_path = tmp_path / "neural_session.json"

    arguments = psth_webapp.parse_webapp_arguments(
        ["--session-metadata", str(metadata_path)]
    )

    assert arguments.session_metadata == metadata_path


def test_metadata_summary_route_forwards_arbitrary_probe_sources_lazily(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Metadata-defined probe IDs reach the existing summary UI without array loading."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)
    received: dict[str, object] = {}

    def fake_build(active_session: object, request: object) -> object:
        """Return a config marker while recording the exact session request."""
        received["session"] = active_session
        received["request"] = request
        return SimpleNamespace(
            session_id=session.session_id,
            session_path=session.session_root,
            output_directory=tmp_path / "cache",
            sites=("site-config",),
            site_pairs=session.site_pairs,
            unit_population=None,
            trial_table_path=session.behavior.trial_table_file,
        )

    def fake_render(*args: object, **kwargs: object) -> None:
        """Record child inputs without touching source or cache arrays."""
        received["streamlit"] = args[0]
        received.update(kwargs)

    monkeypatch.setattr(psth_webapp, "build_lfp_summary_config", fake_build)
    monkeypatch.setattr(psth_webapp.lfp_summary_webapp, "render_lfp_summary_view", fake_render)
    sentinel_streamlit = object()

    psth_webapp.render_metadata_lfp_summary_view(
        sentinel_streamlit,
        session,
        dependencies=object(),
    )

    assert received["streamlit"] is sentinel_streamlit
    assert received["sorter_paths"] == {
        "front-probe": (tmp_path / "ephys/front/kilosort4").resolve(),
        "rear-probe": (tmp_path / "ephys/rear/kilosort4").resolve(),
    }
    assert received["aligned_spike_paths"] == {
        "front-probe": (tmp_path / "ephys/front_sync.npz").resolve(),
        "rear-probe": (tmp_path / "ephys/rear_sync.npz").resolve(),
    }
    assert received["trial_table_path"] == (tmp_path / "behavior/trials.csv").resolve()


def test_active_summary_population_accepts_metadata_probe_ids() -> None:
    """Population construction has no ProbeA/ProbeB identity restriction."""
    cluster_metadata = pd.DataFrame(
        {"cluster_id": [4], "ch": [7], "group": ["good"]}
    )
    channel_metadata = pd.DataFrame(
        {"channel": [7], "channel_quality": ["good"], "inside_brain": [True]}
    )

    population = lfp_summary_webapp.build_active_summary_population(
        probe_label="rear-probe",
        sorter_path=Path("/session/rear/kilosort4"),
        aligned_spike_path=Path("/session/rear_sync.npz"),
        cluster_metadata=cluster_metadata,
        channel_metadata=channel_metadata,
    )

    assert population.probe_label == "rear-probe"
    assert population.stable_unit_ids == ("rear-probe:4",)


def test_load_webapp_session_resolves_metadata_without_array_loading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Webapp startup reads one metadata document but no scientific arrays."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    expected = _write_resolved_session(tmp_path)
    metadata_path = tmp_path / "neural_session.json"

    def forbidden_load(*args: object, **kwargs: object) -> None:
        raise AssertionError("webapp startup must not load a scientific array")

    monkeypatch.setattr(np, "load", forbidden_load)

    session = psth_webapp.load_webapp_session(metadata_path)

    assert session == expected


def test_metadata_population_inputs_keep_hardware_anatomy_and_paths_separate(
    tmp_path: Path,
) -> None:
    """A population selects its probe, anatomical channels, and exact sources."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)

    selection = psth_webapp.metadata_population_inputs(session, "rear-probe")

    assert selection.population_id == "rear-probe"
    assert selection.population_label == "rear-probe"
    assert selection.probe_id == "rear-probe"
    assert selection.probe_label == "rear-probe"
    assert selection.channel_group_label == "rear-probe"
    assert selection.channel_indices == (7, 8, 9)
    assert selection.sorter_directory == (tmp_path / "ephys/rear/kilosort4").resolve()
    assert selection.aligned_spike_file == (tmp_path / "ephys/rear_sync.npz").resolve()
    assert selection.lfp_file == (tmp_path / "ephys/rear/lfp.dat").resolve()


class _MetadataPopulationSidebar:
    """Record the restored metadata-driven scientific population controls."""

    def __init__(self, probe_id: str, channel_source: str) -> None:
        self.probe_id = probe_id
        self.channel_source = channel_source
        self.control_labels: list[str] = []

    def header(self, label: str) -> None:
        """Record one sidebar section label."""
        self.control_labels.append(label)

    def selectbox(self, label: str, *, options: object, **_: object) -> object:
        """Select the requested probe and channel-source mode."""
        del options
        self.control_labels.append(label)
        if label == "Probe / region":
            return self.probe_id
        if label == "Channel source":
            return self.channel_source
        raise AssertionError(f"unexpected selectbox: {label}")

    def multiselect(self, label: str, *, default: object, **_: object) -> object:
        """Keep the default quality-label selection."""
        self.control_labels.append(label)
        return default

    def checkbox(self, label: str, *, value: bool, **_: object) -> bool:
        """Keep the established inside-brain default."""
        self.control_labels.append(label)
        return value

    def text_area(self, label: str, *, value: str, **_: object) -> str:
        """Keep the metadata-derived editable manual channel text."""
        self.control_labels.append(label)
        return value

    def caption(self, _: str) -> None:
        """Accept explanatory captions."""

    def warning(self, _: str) -> None:
        """Accept quality-consistency warnings."""


def test_metadata_population_controls_restore_quality_dropdowns_and_default_all_good_channels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Omitted unit_channels keeps the old quality controls and selects good channels."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)
    sidebar = _MetadataPopulationSidebar(
        "front-probe", psth_webapp.CHANNEL_SOURCE_CHANNEL_QUALITY
    )
    monkeypatch.setattr(psth_webapp, "st", SimpleNamespace(sidebar=sidebar))
    monkeypatch.setattr(
        psth_webapp,
        "load_channel_quality_cached",
        lambda _path: pd.DataFrame(
            {
                "ch": [0, 7, 8, 9],
                "label": ["good", "good", "bad", "good"],
                "is_good": [True, True, False, True],
                "inside_brain": [True, True, True, True],
            }
        ),
    )

    selection = psth_webapp._metadata_population_controls(session)

    assert selection[0] == "front-probe"
    assert selection[1] == "front-probe"
    assert selection[2] == str((tmp_path / "ephys/front/kilosort4").resolve())
    assert selection[3] == str((tmp_path / "ephys/front_sync.npz").resolve())
    assert selection[4] == str((tmp_path / "ephys/front/lfp.dat").resolve())
    assert selection[5].tolist() == [0, 7, 9]
    assert selection[6] == psth_webapp.CHANNEL_SOURCE_CHANNEL_QUALITY
    assert {
        "Region and Units",
        "Probe / region",
        "Channel source",
        "Channel labels",
        "Inside brain only",
    }.issubset(sidebar.control_labels)


def test_metadata_population_controls_apply_optional_channels_without_hiding_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """unit_channels restricts the default quality result but does not replace its UI."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)
    sidebar = _MetadataPopulationSidebar(
        "rear-probe", psth_webapp.CHANNEL_SOURCE_CHANNEL_QUALITY
    )
    monkeypatch.setattr(psth_webapp, "st", SimpleNamespace(sidebar=sidebar))
    monkeypatch.setattr(
        psth_webapp,
        "load_channel_quality_cached",
        lambda _path: pd.DataFrame(
            {
                "ch": [7, 8, 9],
                "label": ["good", "bad", "good"],
                "is_good": [True, False, True],
                "inside_brain": [True, True, True],
            }
        ),
    )

    selection = psth_webapp._metadata_population_controls(session)

    assert selection[0] == "rear-probe"
    assert selection[5].tolist() == [7, 9]
    assert "Channel labels" in sidebar.control_labels
    assert "Inside brain only" in sidebar.control_labels


def test_metadata_lfp_site_inputs_preserve_user_order_and_saved_channels(tmp_path: Path) -> None:
    """LFP controls come from site records rather than CT-specific presets."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)

    sites = psth_webapp.metadata_lfp_site_inputs(session)

    assert [(site.site_id, site.display_label) for site in sites] == [
        ("PFC", "PFC"),
        ("HPC", "HPC"),
    ]
    assert [site.probe_id for site in sites] == ["front-probe", "rear-probe"]
    assert [site.saved_channel_index for site in sites] == [0, 7]
    assert [site.acquisition_family for site in sites] == ["open_ephys", "open_ephys"]


def test_metadata_view_availability_disables_only_views_missing_their_sources(
    tmp_path: Path,
) -> None:
    """Missing spike inputs do not disable LFP-only views or cached inspection."""
    from dataclasses import replace
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)
    rear = replace(
        session.probes[1],
        sorter_directory=None,
    )
    session = replace(session, probes=(session.probes[0], rear))

    availability = psth_webapp.metadata_view_availability(session)

    assert availability[psth_webapp.PLOT_VIEW_LFP_PHASE_CLUSTERING].available is True
    assert availability[psth_webapp.PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE].available is True
    assert availability[psth_webapp.PLOT_VIEW_LFP_SUMMARY].available is True
    assert availability[psth_webapp.PLOT_VIEW_UNIT_RASTER].available is False
    assert "rear-probe" in availability[psth_webapp.PLOT_VIEW_UNIT_RASTER].reason
    assert availability[psth_webapp.PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING].available is False


def test_main_metadata_launch_reaches_summary_route_without_legacy_path_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The documented Streamlit argument launches the metadata summary route directly."""
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)
    calls: list[object] = []

    class Sidebar:
        """Minimal sidebar selecting the cached-summary view."""

        def button(self, *_: object, **__: object) -> bool:
            return False

        def selectbox(self, label: str, **_: object) -> str:
            assert label == "Plot view"
            return psth_webapp.PLOT_VIEW_LFP_SUMMARY

        def caption(self, message: str) -> None:
            calls.append(message)

    class Streamlit:
        """Minimal metadata-startup host with no legacy text inputs."""

        sidebar = Sidebar()

        def set_page_config(self, **_: object) -> None:
            pass

        def title(self, _: str) -> None:
            pass

    monkeypatch.setattr(psth_webapp, "st", Streamlit())
    monkeypatch.setattr(psth_webapp, "load_webapp_session", lambda _: session)
    monkeypatch.setattr(
        psth_webapp,
        "render_metadata_lfp_summary_view",
        lambda streamlit, active_session: calls.append((streamlit, active_session)),
    )
    monkeypatch.setattr(
        psth_webapp,
        "_initialize_path_input_state",
        lambda: (_ for _ in ()).throw(AssertionError("legacy path controls were initialized")),
    )

    psth_webapp.main(["--session-metadata", str(tmp_path / "neural_session.json")])

    assert any(isinstance(call, tuple) and call[1] is session for call in calls)


def test_main_metadata_launch_stops_before_unavailable_view_loaders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unavailable metadata view explains its missing source before data loading."""
    from dataclasses import replace
    from src.tests.neural_analysis.test_lfp_summary_session import _write_resolved_session

    session = _write_resolved_session(tmp_path)
    session = replace(
        session,
        probes=(session.probes[0], replace(session.probes[1], sorter_directory=None)),
    )
    warnings: list[str] = []

    class Sidebar:
        """Minimal sidebar selecting a spike-only view."""

        def button(self, *_: object, **__: object) -> bool:
            return False

        def selectbox(self, label: str, **_: object) -> str:
            assert label == "Plot view"
            return psth_webapp.PLOT_VIEW_UNIT_RASTER

        def caption(self, _: str) -> None:
            pass

    class Streamlit:
        """Minimal host recording the useful unavailability message."""

        sidebar = Sidebar()

        def set_page_config(self, **_: object) -> None:
            pass

        def title(self, _: str) -> None:
            pass

        def warning(self, message: str) -> None:
            warnings.append(message)

    monkeypatch.setattr(psth_webapp, "st", Streamlit())
    monkeypatch.setattr(psth_webapp, "load_webapp_session", lambda _: session)
    monkeypatch.setattr(
        psth_webapp,
        "load_viewer_data_cached",
        lambda **_: (_ for _ in ()).throw(AssertionError("viewer data loaded")),
    )

    psth_webapp.main(["--session-metadata", str(tmp_path / "neural_session.json")])

    assert warnings and "rear-probe" in warnings[0]


def test_real_summary_route_builds_usable_production_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The live route must supply production Power dependencies when none are injected.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records the production factory and route delegation without Streamlit UI.
    tmp_path : pathlib.Path
        Temporary active session directory passed through unchanged.
    """
    sentinel_dependencies = object()
    received: dict[str, object] = {}

    def fake_factory() -> object:
        """Return a usable opaque dependency marker for route delegation."""
        return sentinel_dependencies

    def fake_render(*args: object, **kwargs: object) -> None:
        """Record summary-view delegation keyword arguments."""
        del args
        received.update(kwargs)

    monkeypatch.setattr(
        psth_webapp.lfp_summary_webapp,
        "make_production_summary_dependencies",
        fake_factory,
    )
    monkeypatch.setattr(psth_webapp.lfp_summary_webapp, "render_lfp_summary_view", fake_render)

    psth_webapp.render_lfp_summary_view(
        str(tmp_path),
        "CT026_2026-08-01_130853",
        "hpc.lf.bin",
        "pfc.lf.bin",
        "hpc.sync.npz",
        "pfc.sync.npz",
    )

    assert received["dependencies"] is sentinel_dependencies
    assert received["session_path"] == tmp_path


def test_summary_route_forwards_sorter_and_aligned_paths_for_both_probes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The parent route must preserve paths and lazy metadata callbacks.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the child renderer and lazy loaders so the test can distinguish
        callback forwarding from eager metadata I/O.
    tmp_path : pathlib.Path
        Temporary root used to construct distinct ProbeA and ProbeB source paths.
    """

    received: dict[str, object] = {}
    loader_calls: list[Path] = []

    def fake_render(*args: object, **kwargs: object) -> None:
        """Record child-route inputs without loading metadata or cache files."""

        del args
        received.update(kwargs)

    def fail_cluster_loader(sorter_path: Path) -> pd.DataFrame:
        """Fail if parent delegation eagerly opens selected sorter metadata."""

        loader_calls.append(sorter_path)
        raise AssertionError("parent delegation eagerly loaded cluster metadata")

    def fail_channel_loader(sorter_path: Path) -> pd.DataFrame:
        """Fail if parent delegation eagerly opens selected channel metadata."""

        loader_calls.append(sorter_path)
        raise AssertionError("parent delegation eagerly loaded channel metadata")

    monkeypatch.setattr(
        psth_webapp.lfp_summary_webapp,
        "make_production_summary_dependencies",
        lambda: object(),
    )
    monkeypatch.setattr(psth_webapp.lfp_summary_webapp, "render_lfp_summary_view", fake_render)
    monkeypatch.setattr(
        psth_webapp,
        "load_summary_cluster_metadata",
        fail_cluster_loader,
        raising=False,
    )
    monkeypatch.setattr(
        psth_webapp,
        "load_summary_channel_metadata",
        fail_channel_loader,
        raising=False,
    )
    hpc_sorter = tmp_path / "probe_b" / "kilosort4"
    pfc_sorter = tmp_path / "probe_a" / "kilosort4"
    hpc_aligned = tmp_path / "probe_b" / "aligned_spikes.npz"
    pfc_aligned = tmp_path / "probe_a" / "aligned_spikes.npz"

    psth_webapp.render_lfp_summary_view(
        str(tmp_path),
        "synthetic-session",
        str(tmp_path / "probe_b.lf.bin"),
        str(tmp_path / "probe_a.lf.bin"),
        str(hpc_aligned),
        str(pfc_aligned),
        hpc_v1_sorter_output_path=str(hpc_sorter),
        pfc_sorter_output_path=str(pfc_sorter),
    )

    assert received["sorter_paths"] == {"ProbeA": pfc_sorter, "ProbeB": hpc_sorter}
    assert received["aligned_spike_paths"] == {"ProbeA": pfc_aligned, "ProbeB": hpc_aligned}
    assert received["cluster_metadata_loader"] is fail_cluster_loader
    assert received["channel_metadata_loader"] is fail_channel_loader
    assert loader_calls == []


def test_summary_metadata_helpers_use_selected_sorter_and_existing_cached_readers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Lazy summary metadata helpers read only one selected probe's cached tables.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces cached table readers and probe-directory inference. No sorter
        arrays, aligned spikes, raw LFP, or external filesystem are opened.
    tmp_path : pathlib.Path
        Temporary root used only to construct one explicit ``kilosort4`` path
        and its distinct Probe-derived parent directory.

    Returns
    -------
    None
        Requires the cluster helper to delegate the exact selected sorter path
        to a cluster-info-only cached reader, and the channel helper to resolve
        the same sorter path through ``infer_probe_derived_dir`` before calling
        the existing cached channel-quality loader for that Probe parent.
    """

    sorter_path = tmp_path / "ProbeB" / "kilosort4"
    probe_parent = sorter_path.parent
    cluster_calls: list[str] = []
    infer_calls: list[tuple[Path | str | None, Path | str | None]] = []
    channel_calls: list[str] = []
    expected_clusters = pd.DataFrame({"cluster_id": (7,), "ch": (1,), "group": ("good",)})
    expected_channels = pd.DataFrame({"channel": (1,), "channel_quality": ("good",), "inside_brain": (True,)})

    def load_cluster_info_cached(sorter_directory: str) -> pd.DataFrame:
        """Record the selected sorter directory and return only cluster metadata."""

        cluster_calls.append(sorter_directory)
        return expected_clusters

    def infer_probe_derived_dir(
        sorter_output_path: Path | str | None,
        lfp_path: Path | str | None,
    ) -> Path:
        """Record source identities and return the selected Probe parent."""

        infer_calls.append((sorter_output_path, lfp_path))
        return probe_parent

    def load_channel_quality_cached(probe_directory: str) -> pd.DataFrame:
        """Record the resolved Probe parent and return normalized channel metadata."""

        channel_calls.append(probe_directory)
        return expected_channels

    monkeypatch.setattr(psth_webapp, "load_cluster_info_cached", load_cluster_info_cached, raising=False)
    monkeypatch.setattr(
        psth_webapp.unit_spike_loading,
        "infer_probe_derived_dir",
        infer_probe_derived_dir,
    )
    monkeypatch.setattr(
        psth_webapp,
        "load_channel_quality_cached",
        load_channel_quality_cached,
    )

    clusters = psth_webapp.load_summary_cluster_metadata(sorter_path)
    channels = psth_webapp.load_summary_channel_metadata(sorter_path)

    assert clusters is expected_clusters
    assert channels is expected_channels
    assert cluster_calls == [str(sorter_path)]
    assert infer_calls == [(sorter_path, None)]
    assert channel_calls == [str(probe_parent)]


def test_derived_open_ephys_lfp_paths_use_aligned_sync_adapters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Derived ``lfp.dat`` inputs must not be mislabeled as SpikeGLX binaries.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records route delegation without opening the native recordings.
    tmp_path : pathlib.Path
        Temporary session root used to construct explicit probe paths.
    """
    received: dict[str, object] = {}

    def fake_render(*args: object, **kwargs: object) -> None:
        """Record summary-view delegation keyword arguments."""
        del args
        received.update(kwargs)

    monkeypatch.setattr(
        psth_webapp.lfp_summary_webapp,
        "make_production_summary_dependencies",
        lambda: object(),
    )
    monkeypatch.setattr(
        psth_webapp.lfp_summary_webapp,
        "render_lfp_summary_view",
        fake_render,
    )
    probe_a = tmp_path / "ProbeA" / "lfp.dat"
    probe_b = tmp_path / "ProbeB" / "lfp.dat"
    probe_a_sync = tmp_path / "probeA_sync.npz"
    probe_b_sync = tmp_path / "probeB_sync.npz"

    psth_webapp.render_lfp_summary_view(
        str(tmp_path),
        "CT026_2026-08-01_130853",
        str(probe_b),
        str(probe_a),
        str(probe_b_sync),
        str(probe_a_sync),
    )

    sites = received["sites"]
    assert isinstance(sites, tuple)
    assert [site.acquisition_format for site in sites] == [
        "open_ephys",
        "open_ephys",
        "open_ephys",
    ]
    assert [site.aligned_sync_path for site in sites] == [
        probe_a_sync,
        probe_b_sync,
        probe_b_sync,
    ]


def test_build_lfp_dropdown_options_uses_only_explicit_probe_paths():
    """LFP dropdown choices should come directly from user-entered probe path fields."""
    options = psth_webapp.build_lfp_dropdown_options(
        hpc_v1_lfp_path="/data/hpc_v1/run0_g0_t0.imec1.lf.bin",
        pfc_lfp_path="/data/pfc/run0_g0_t0.imec0.lf.bin",
    )

    assert list(options.keys()) == ["HPC/V1 LFP", "PFC LFP"]
    assert options["HPC/V1 LFP"] == "/data/hpc_v1/run0_g0_t0.imec1.lf.bin"
    assert options["PFC LFP"] == "/data/pfc/run0_g0_t0.imec0.lf.bin"


def test_webapp_exposes_spike_lfp_phase_locking_defaults_and_cache_contract():
    """Spike-LFP locking should be a dedicated pooled-trial view with explicit defaults."""

    assert psth_webapp.PLOT_VIEW_SPIKE_LFP_PHASE_LOCKING in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.SPIKE_LFP_PHASE_DEFAULT_WINDOW == (-0.5, 0.5)
    assert psth_webapp.SPIKE_LFP_PHASE_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.SPIKE_LFP_PHASE_MAX_FREQUENCY_HZ == 100.0
    assert psth_webapp.SPIKE_LFP_PHASE_FREQUENCY_COUNT == 50
    assert psth_webapp.SPIKE_LFP_PHASE_DEFAULT_POLAR_FREQUENCY_HZ == 8.0
    assert psth_webapp.SPIKE_LFP_PHASE_BIN_COUNT == 24
    assert psth_webapp.SPIKE_LFP_PHASE_AMPLITUDE_MASK_OPTIONS == ("Off", "Absolute magnitude")
    assert psth_webapp.SPIKE_LFP_PHASE_CACHE_MAX_ENTRIES == 6
    parameters = inspect.signature(psth_webapp.compute_spike_lfp_phase_locking_cached).parameters
    for required_parameter in (
        "unit_id",
        "unit_spike_times_s",
        "trial_indices",
        "event_times_s",
        "window_start_s",
        "window_end_s",
        "lfp_path",
        "saved_channel_index",
        "absolute_amplitude_threshold",
        "phase_bin_count",
    ):
        assert required_parameter in parameters


def test_webapp_exposes_single_trial_spike_lfp_hilbert_view_and_cache_contract():
    """The single-trial Hilbert viewer must be a separate source-rate cached path."""

    assert psth_webapp.PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT == "Single-trial spike-LFP phase"
    assert psth_webapp.PLOT_VIEW_SINGLE_TRIAL_SPIKE_LFP_HILBERT in psth_webapp.PLOT_VIEW_OPTIONS
    assert callable(psth_webapp.render_single_trial_spike_lfp_hilbert_view)
    parameters = inspect.signature(psth_webapp.compute_single_trial_spike_lfp_hilbert_cached).parameters
    for required_parameter in (
        "lfp_path",
        "lfp_mtime_ns",
        "saved_channel_index",
        "aligned_sync_mtime_ns",
        "unit_id",
        "unit_spike_times_s",
        "trial_index",
        "event_time_s",
        "window_start_s",
        "window_end_s",
        "band_low_hz",
        "band_high_hz",
        "filter_padding_s",
    ):
        assert required_parameter in parameters


def _write_open_ephys_cache_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write tiny selected-source identities with complete physical-uV sidecar metadata."""
    lfp_path = tmp_path / "lfp.dat"
    sync_path = tmp_path / "probe_sync.npz"
    sidecar_path = tmp_path / "lfp_preprocessing.json"
    lfp_path.write_bytes(b"large binary identity by stat")
    sync_path.write_bytes(b"aligned sync identity by stat")
    sidecar_path.write_text(
        json.dumps(
            {
                "output_binary": "lfp.dat",
                "sampling_frequency_hz": 2500.0,
                "num_channels": 1,
                "num_segments": 1,
                "num_samples_by_segment": [1],
                "dtype": "float32",
                "binary_layout": "time_major_channel_interleaved",
                "channel_ids_in_binary_order": ["CH0"],
                "lfp_binary_scaling": {
                    "data_units": "unscaled_binary_values",
                    "has_scaleable_traces": True,
                    "channel_ids": ["CH0"],
                    "gain_to_uV_by_channel": [0.195],
                    "offset_to_uV_by_channel": [0.0],
                    "physical_unit_by_channel": ["uV"],
                    "export_scale_factor": 1.0,
                    "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
                },
            },
            separators=(",", ":"),
        ),
        encoding="ascii",
    )
    return lfp_path, sync_path, sidecar_path


def _load_open_ephys_trace_from_public_cache(lfp_path: Path, sync_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Call the stable public trace helper without passing any implementation token."""
    return psth_webapp.load_trial_lfp_trace_cached(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path),
        saved_channel_index=0,
        alignment_time_s=1.0,
        window_start_s=-0.1,
        window_end_s=0.1,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        filter_low_hz=None,
        filter_high_hz=None,
        filter_padding_s=0.0,
        aligned_sync_npz_path=str(sync_path),
    )


def test_open_ephys_cache_token_is_hashable_and_hashes_same_length_sidecar_edits(
    tmp_path: Path,
) -> None:
    """A content digest makes the full selected Open Ephys source identity cache-visible."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    original_stat = sidecar_path.stat()
    original_digest = sha256(sidecar_path.read_bytes()).hexdigest()

    before = psth_webapp.build_open_ephys_cache_token(lfp_path, sync_path)
    unchanged = psth_webapp.build_open_ephys_cache_token(lfp_path, sync_path)
    changed_semantics = replace(
        before,
        source_value_semantics="open_ephys_affine_uV_v2",
    )
    changed_lfp_path = replace(before, lfp_path=str(lfp_path.with_name("other_lfp.dat")))
    changed_lfp_mtime = replace(before, lfp_mtime_ns=before.lfp_mtime_ns + 1)
    changed_sync_path = replace(before, aligned_sync_path=str(sync_path.with_name("other_sync.npz")))
    changed_sync_mtime = replace(before, aligned_sync_mtime_ns=before.aligned_sync_mtime_ns + 1)
    metadata = json.loads(sidecar_path.read_text(encoding="ascii"))
    metadata["lfp_binary_scaling"]["gain_to_uV_by_channel"] = [0.196]
    sidecar_path.write_text(json.dumps(metadata, separators=(",", ":")), encoding="ascii")
    os.utime(sidecar_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    after = psth_webapp.build_open_ephys_cache_token(lfp_path, sync_path)

    assert before == unchanged
    assert hash(before) == hash(unchanged)
    assert {before: "selected source"}[unchanged] == "selected source"
    assert len({before, unchanged}) == 1
    assert before.lfp_path == str(lfp_path.resolve())
    assert before.aligned_sync_path == str(sync_path.resolve())
    assert before.lfp_preprocessing_path == str(sidecar_path.resolve())
    assert before.lfp_size_bytes == lfp_path.stat().st_size
    assert before.lfp_mtime_ns == lfp_path.stat().st_mtime_ns
    assert before.aligned_sync_size_bytes == sync_path.stat().st_size
    assert before.aligned_sync_mtime_ns == sync_path.stat().st_mtime_ns
    assert before.source_value_semantics == "open_ephys_affine_uV_v1"
    assert changed_semantics != before
    for changed_identity in (
        changed_lfp_path,
        changed_lfp_mtime,
        changed_sync_path,
        changed_sync_mtime,
    ):
        assert changed_identity != before
    assert before.lfp_preprocessing_size_bytes == after.lfp_preprocessing_size_bytes
    assert before.lfp_preprocessing_mtime_ns == after.lfp_preprocessing_mtime_ns
    assert before.lfp_preprocessing_sha256 == original_digest
    assert before != after
    assert before.lfp_preprocessing_sha256 != after.lfp_preprocessing_sha256
    assert after.lfp_preprocessing_sha256 == sha256(sidecar_path.read_bytes()).hexdigest()

    lfp_path.write_bytes(lfp_path.read_bytes() + b"x")
    changed_lfp = psth_webapp.build_open_ephys_cache_token(lfp_path, sync_path)
    sync_path.write_bytes(sync_path.read_bytes() + b"y")
    changed_sync = psth_webapp.build_open_ephys_cache_token(lfp_path, sync_path)

    assert changed_lfp != after
    assert changed_lfp.lfp_size_bytes == after.lfp_size_bytes + 1
    assert changed_sync != changed_lfp
    assert changed_sync.aligned_sync_size_bytes == changed_lfp.aligned_sync_size_bytes + 1


def _assert_open_ephys_semantics_token_recomputes(
    monkeypatch: pytest.MonkeyPatch,
    sources: tuple[tuple[Path, Path], ...],
    clear_cache: Callable[[], None],
    invoke: Callable[[], object],
    computation_calls: list[object],
    calls_per_cache_miss: int,
) -> None:
    """Prove the real wrapper keys recomputation on a semantics-only token change.

    The public callers intentionally omit the optional token. The wrapper must
    build it at the selected-source boundary before entering its keyed cache;
    changing only the injected semantics field therefore produces one new
    cache miss while all paths, sizes, mtimes, and sidecar digest stay fixed.
    """
    active_tokens: dict[str, object] = {}
    for lfp_path, sync_path in sources:
        token = psth_webapp.build_open_ephys_cache_token(lfp_path, sync_path)
        active_tokens[str(lfp_path.resolve())] = token
    changed_path = str(sources[0][0].resolve())
    changed_token = replace(
        active_tokens[changed_path],
        source_value_semantics="open_ephys_affine_uV_v2",
    )
    assert changed_token != active_tokens[changed_path]

    def selected_source_builder(*args: object, **kwargs: object) -> object:
        """Return a fixed token for each selected source without rehashing during the test."""
        raw_path = args[0] if args else kwargs["lfp_path"]
        return active_tokens[str(Path(raw_path).resolve())]

    with monkeypatch.context() as scoped:
        scoped.setattr(
            psth_webapp,
            "build_open_ephys_cache_token",
            selected_source_builder,
        )
        clear_cache()
        start_count = len(computation_calls)
        invoke()
        invoke()
        assert len(computation_calls) == start_count + calls_per_cache_miss
        active_tokens[changed_path] = changed_token
        invoke()
        assert len(computation_calls) == start_count + 2 * calls_per_cache_miss


def test_public_open_ephys_trace_cache_derives_identity_before_keying_and_invalidates_sidecar_edits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Old callers omit tokens while a keyed inner cache still sees full sidecar identity."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    calls: list[str] = []

    def fake_loader(**_: object) -> tuple[np.ndarray, np.ndarray]:
        """Return one inexpensive physical-uV trace while recording a true cache miss."""
        calls.append("load")
        return np.array([-0.1, 0.0]), np.array([1.95, 3.9])

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format", fake_loader)
    original_stat = sidecar_path.stat()

    _assert_open_ephys_semantics_token_recomputes(
        monkeypatch,
        ((lfp_path, sync_path),),
        psth_webapp.load_trial_lfp_trace_cached.clear,
        lambda: _load_open_ephys_trace_from_public_cache(lfp_path, sync_path),
        calls,
        1,
    )
    calls.clear()
    psth_webapp.load_trial_lfp_trace_cached.clear()

    first = _load_open_ephys_trace_from_public_cache(lfp_path, sync_path)
    second = _load_open_ephys_trace_from_public_cache(lfp_path, sync_path)
    metadata = json.loads(sidecar_path.read_text(encoding="ascii"))
    metadata["lfp_binary_scaling"]["gain_to_uV_by_channel"] = [0.196]
    sidecar_path.write_text(json.dumps(metadata, separators=(",", ":")), encoding="ascii")
    os.utime(sidecar_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    third = _load_open_ephys_trace_from_public_cache(lfp_path, sync_path)

    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(second[1], third[1])
    assert calls == ["load", "load"]


def _rewrite_same_length_open_ephys_sidecar(sidecar_path: Path) -> None:
    """Change one scale value while preserving the sidecar's byte count and mtime."""
    before_stat = sidecar_path.stat()
    metadata = json.loads(sidecar_path.read_text(encoding="ascii"))
    metadata["lfp_binary_scaling"]["gain_to_uV_by_channel"] = [0.196]
    encoded = json.dumps(metadata, separators=(",", ":"))
    assert len(encoded.encode("ascii")) == before_stat.st_size
    sidecar_path.write_text(encoded, encoding="ascii")
    os.utime(sidecar_path, ns=(before_stat.st_atime_ns, before_stat.st_mtime_ns))


def _open_ephys_spectrogram_kwargs(lfp_path: Path, sync_path: Path) -> dict[str, object]:
    """Return one inexpensive public Open Ephys spectrogram-cache request."""
    return {
        "lfp_format": psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        "lfp_path": str(lfp_path),
        "saved_channel_index": 0,
        "alignment_time_s": 100.0,
        "visible_window_start_s": -0.1,
        "visible_window_end_s": 0.1,
        "digital_word": 0,
        "irig_line": 6,
        "bit_period_s": 1.0,
        "utc_offset_hours": 0.0,
        "frequencies_hz": (8.0,),
        "gaussian_width": 1.5,
        "window_length": 1.0,
        "precision": 8,
        "norm": "l1",
        "target_sample_rate_hz": 500.0,
        "notch_60_hz": False,
        "notch_quality_factor": 30.0,
        "aligned_sync_npz_path": str(sync_path),
    }


def test_open_ephys_spectrogram_cache_reuses_then_recomputes_after_same_length_sidecar_edit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The public omitted-token spectrogram cache keys its real source-rate loader by sidecar content."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    load_calls: list[dict[str, object]] = []
    morlet_inputs: list[np.ndarray] = []

    def fake_load_with_rate(**kwargs: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Return a physical-uV source trace at the actual spectrogram reader boundary."""
        load_calls.append(kwargs)
        return np.array([-0.2, -0.1, 0.0, 0.1, 0.2]), np.array([1.95, 3.9, 1.95, 0.0, -1.95]), 10.0

    def fake_morlet(**kwargs: object) -> SimpleNamespace:
        """Capture physical values and return a tiny retained power result without numerical cost."""
        values = np.asarray(kwargs["lfp_values"])
        morlet_inputs.append(values)
        return SimpleNamespace(
            time_s=np.array([-0.1, 0.0]),
            frequencies_hz=np.array([8.0]),
            lfp_values=values[1:3],
            log_power_db=np.zeros((2, 1)),
        )

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format_with_sample_rate", fake_load_with_rate)
    monkeypatch.setattr(psth_webapp.lfp_spectrogram, "compute_morlet_log_power", fake_morlet)
    psth_webapp.compute_trial_lfp_spectrogram_cached.clear()
    kwargs = _open_ephys_spectrogram_kwargs(lfp_path, sync_path)
    _assert_open_ephys_semantics_token_recomputes(
        monkeypatch,
        ((lfp_path, sync_path),),
        psth_webapp.compute_trial_lfp_spectrogram_cached.clear,
        lambda: psth_webapp.compute_trial_lfp_spectrogram_cached(**kwargs),
        load_calls,
        1,
    )
    load_calls.clear()
    morlet_inputs.clear()
    psth_webapp.compute_trial_lfp_spectrogram_cached.clear()
    first = psth_webapp.compute_trial_lfp_spectrogram_cached(**kwargs)
    second = psth_webapp.compute_trial_lfp_spectrogram_cached(**kwargs)
    _rewrite_same_length_open_ephys_sidecar(sidecar_path)
    third = psth_webapp.compute_trial_lfp_spectrogram_cached(**kwargs)

    assert len(load_calls) == 2
    assert len(morlet_inputs) == 2
    np.testing.assert_array_equal(morlet_inputs[0], np.array([1.95, 3.9, 1.95, 0.0, -1.95]))
    np.testing.assert_array_equal(first.time_s, second.time_s)
    np.testing.assert_array_equal(second.frequencies_hz, third.frequencies_hz)


def test_open_ephys_shared_power_cache_reuses_one_nested_token_and_recomputes_after_sidecar_edit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Shared limits retain one selected-source identity across nested trial spectrograms."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    nested_calls: list[dict[str, object]] = []

    def fake_spectrogram(**kwargs: object) -> SimpleNamespace:
        """Record each actual nested cache computation without a Morlet transform."""
        nested_calls.append(kwargs)
        return SimpleNamespace(log_power_db=np.zeros((1, 1)))

    monkeypatch.setattr(psth_webapp, "compute_trial_lfp_spectrogram_cached", fake_spectrogram)
    psth_webapp.compute_shared_lfp_power_limits_cached.clear()
    kwargs = {
        "reference_alignment_times_s": (100.0, 101.0),
        **_open_ephys_spectrogram_kwargs(lfp_path, sync_path),
        "lower_percentile": 5.0,
        "upper_percentile": 95.0,
    }
    kwargs.pop("alignment_time_s")
    _assert_open_ephys_semantics_token_recomputes(
        monkeypatch,
        ((lfp_path, sync_path),),
        psth_webapp.compute_shared_lfp_power_limits_cached.clear,
        lambda: psth_webapp.compute_shared_lfp_power_limits_cached(**kwargs),
        nested_calls,
        2,
    )
    nested_calls.clear()
    psth_webapp.compute_shared_lfp_power_limits_cached.clear()
    psth_webapp.compute_shared_lfp_power_limits_cached(**kwargs)
    psth_webapp.compute_shared_lfp_power_limits_cached(**kwargs)
    _rewrite_same_length_open_ephys_sidecar(sidecar_path)
    psth_webapp.compute_shared_lfp_power_limits_cached(**kwargs)

    assert len(nested_calls) == 4
    first_token = nested_calls[0]["open_ephys_cache_token"]
    assert nested_calls[1]["open_ephys_cache_token"] is first_token
    assert nested_calls[2]["open_ephys_cache_token"] != first_token
    assert nested_calls[3]["open_ephys_cache_token"] is nested_calls[2]["open_ephys_cache_token"]


def test_open_ephys_continuous_phase_cache_reuses_then_recomputes_after_sidecar_edit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Continuous phase caching consumes physical uV and cannot retain an identity-free OE entry."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    block_values: list[np.ndarray] = []

    def fake_load_with_rate(**_: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Return one source-rate physical-uV block on its existing relative grid."""
        return np.array([-0.1, 0.0, 0.1]), np.array([1.95, 3.9, 1.95]), 10.0

    def fake_phase_tensor(**kwargs: object) -> SimpleNamespace:
        """Exercise the public continuous block loader while avoiding a Morlet transform."""
        absolute_time_s, values_uv, sample_rate_hz = kwargs["block_loader"](100.0, 100.2)
        assert sample_rate_hz == 10.0
        np.testing.assert_allclose(absolute_time_s, np.array([100.0, 100.1, 100.2]), rtol=0.0, atol=1e-12)
        block_values.append(values_uv)
        return SimpleNamespace(phase_tensor=np.ones((1, 1, 1, 1), dtype=np.complex64))

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format_with_sample_rate", fake_load_with_rate)
    monkeypatch.setattr(psth_webapp.lfp_phase_clustering, "compute_site_phase_trial_tensor", fake_phase_tensor)
    psth_webapp.compute_lfp_phase_site_cached.clear()
    kwargs = {
        "lfp_format": psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        "lfp_path": str(lfp_path),
        "lfp_file_mtime_ns": lfp_path.stat().st_mtime_ns,
        "saved_channel_index": 0,
        "event_times_s": (100.0,),
        "trial_indices": (7,),
        "window_start_s": -0.1,
        "window_end_s": 0.1,
        "digital_word": 0,
        "irig_line": 6,
        "bit_period_s": 1.0,
        "utc_offset_hours": 0.0,
        "frequencies_hz": (8.0,),
        "gaussian_width": 1.5,
        "window_length": 1.0,
        "precision": 8,
        "norm": "l1",
        "output_sample_rate_hz": 500.0,
        "notch_60_hz": False,
        "notch_quality_factor": 30.0,
        "minimum_relative_magnitude": 1e-12,
        "maximum_core_duration_s": 10.0,
        "aligned_sync_npz_path": str(sync_path),
    }
    _assert_open_ephys_semantics_token_recomputes(
        monkeypatch,
        ((lfp_path, sync_path),),
        psth_webapp.compute_lfp_phase_site_cached.clear,
        lambda: psth_webapp.compute_lfp_phase_site_cached(**kwargs),
        block_values,
        1,
    )
    block_values.clear()
    psth_webapp.compute_lfp_phase_site_cached.clear()
    psth_webapp.compute_lfp_phase_site_cached(**kwargs)
    psth_webapp.compute_lfp_phase_site_cached(**kwargs)
    _rewrite_same_length_open_ephys_sidecar(sidecar_path)
    psth_webapp.compute_lfp_phase_site_cached(**kwargs)

    assert len(block_values) == 2
    np.testing.assert_array_equal(block_values[0], np.array([1.95, 3.9, 1.95]))


def test_open_ephys_relative_phase_cache_reuses_then_recomputes_after_sidecar_edit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Relative-phase caching propagates both selected OE source identities through its two loaders."""
    site_a_root = tmp_path / "site_a"
    site_b_root = tmp_path / "site_b"
    site_a_root.mkdir()
    site_b_root.mkdir()
    lfp_path_a, sync_path_a, sidecar_path_a = _write_open_ephys_cache_sources(site_a_root)
    lfp_path_b, sync_path_b, _ = _write_open_ephys_cache_sources(site_b_root)
    load_values: list[np.ndarray] = []

    def fake_load_with_rate(**_: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Return a physical-uV trace for either selected source site."""
        values_uv = np.array([1.95, 3.9, 1.95])
        load_values.append(values_uv)
        return np.array([-0.1, 0.0, 0.1]), values_uv, 10.0

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format_with_sample_rate", fake_load_with_rate)
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "compute_wavelet_coefficients",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "compute_single_trial_relative_phase",
        lambda **_: SimpleNamespace(relative_time_s=np.array([-0.1, 0.0, 0.1])),
    )
    psth_webapp.compute_single_trial_relative_phase_cached.clear()
    kwargs = {
        "lfp_format": psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        "lfp_path_a": str(lfp_path_a),
        "lfp_mtime_ns_a": lfp_path_a.stat().st_mtime_ns,
        "channel_a": 0,
        "site_a_label": "PFC",
        "aligned_sync_path_a": str(sync_path_a),
        "aligned_sync_mtime_ns_a": sync_path_a.stat().st_mtime_ns,
        "lfp_path_b": str(lfp_path_b),
        "lfp_mtime_ns_b": lfp_path_b.stat().st_mtime_ns,
        "channel_b": 0,
        "site_b_label": "HPC",
        "aligned_sync_path_b": str(sync_path_b),
        "aligned_sync_mtime_ns_b": sync_path_b.stat().st_mtime_ns,
        "trial_index": 7,
        "event_time_s": 100.0,
        "window_start_s": -0.1,
        "window_end_s": 0.1,
        "digital_word": 0,
        "irig_line": 6,
        "bit_period_s": 1.0,
        "utc_offset_hours": 0.0,
        "frequencies_hz": (8.0,),
        "gaussian_width": 1.5,
        "wavelet_window_length": 1.0,
        "precision": 8,
        "norm": "l1",
        "output_sample_rate_hz": 500.0,
        "notch_60_hz": False,
        "notch_quality_factor": 30.0,
        "minimum_relative_magnitude": 1e-12,
        "plv_window_cycles": 3.0,
        "plv_min_window_s": None,
        "plv_max_window_s": None,
    }
    _assert_open_ephys_semantics_token_recomputes(
        monkeypatch,
        ((lfp_path_a, sync_path_a), (lfp_path_b, sync_path_b)),
        psth_webapp.compute_single_trial_relative_phase_cached.clear,
        lambda: psth_webapp.compute_single_trial_relative_phase_cached(**kwargs),
        load_values,
        2,
    )
    load_values.clear()
    psth_webapp.compute_single_trial_relative_phase_cached.clear()
    psth_webapp.compute_single_trial_relative_phase_cached(**kwargs)
    psth_webapp.compute_single_trial_relative_phase_cached(**kwargs)
    _rewrite_same_length_open_ephys_sidecar(sidecar_path_a)
    psth_webapp.compute_single_trial_relative_phase_cached(**kwargs)

    assert len(load_values) == 4
    assert all(np.array_equal(values, np.array([1.95, 3.9, 1.95])) for values in load_values)


@pytest.mark.parametrize("cached_name", ("phase_locking", "hilbert"))
def test_open_ephys_spike_phase_caches_reuse_then_recompute_after_sidecar_edit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    cached_name: str,
) -> None:
    """Phase-locking and Hilbert views retain physical uV and invalidate omitted-token OE calls."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    loaded_values: list[np.ndarray] = []

    def fake_load_with_rate(**_: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Return one physical-uV source trace without reading a real recording."""
        values_uv = np.array([1.95, 3.9, 1.95])
        loaded_values.append(values_uv)
        return np.array([-0.1, 0.0, 0.1]), values_uv, 10.0

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format_with_sample_rate", fake_load_with_rate)
    common = {
        "lfp_format": psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        "lfp_path": str(lfp_path),
        "lfp_mtime_ns": lfp_path.stat().st_mtime_ns,
        "saved_channel_index": 0,
        "lfp_site_label": "PFC",
        "aligned_sync_path": str(sync_path),
        "aligned_sync_mtime_ns": sync_path.stat().st_mtime_ns,
        "unit_id": 7,
        "unit_spike_times_s": (100.0,),
        "digital_word": 0,
        "irig_line": 6,
        "bit_period_s": 1.0,
        "utc_offset_hours": 0.0,
    }
    if cached_name == "phase_locking":
        def fake_phase_locking(**kwargs: object) -> SimpleNamespace:
            """Exercise the phase-locking block loader without wavelet numerics."""
            _time_s, values_uv, _rate_hz = kwargs["block_loader"](100.0, 100.2)
            np.testing.assert_array_equal(values_uv, np.array([1.95, 3.9, 1.95]))
            return SimpleNamespace()

        monkeypatch.setattr(
            psth_webapp.spike_lfp_phase_locking,
            "compute_trial_aligned_spike_phase_locking",
            fake_phase_locking,
        )
        cached = psth_webapp.compute_spike_lfp_phase_locking_cached
        kwargs = {
            **common,
            "trial_indices": (7,),
            "event_times_s": (100.0,),
            "window_start_s": -0.1,
            "window_end_s": 0.1,
            "frequencies_hz": (8.0,),
            "gaussian_width": 1.5,
            "wavelet_window_length": 1.0,
            "precision": 8,
            "norm": "l1",
            "target_sample_rate_hz": 500.0,
            "notch_60_hz": False,
            "notch_quality_factor": 30.0,
            "minimum_relative_magnitude": 1e-12,
            "absolute_amplitude_threshold": 0.0,
            "maximum_core_duration_s": 10.0,
            "phase_bin_count": 12,
        }
    else:
        def fake_hilbert(**kwargs: object) -> SimpleNamespace:
            """Capture raw physical units before the Hilbert computation seam."""
            np.testing.assert_array_equal(kwargs["padded_raw_lfp"], np.array([1.95, 3.9, 1.95]))
            return SimpleNamespace()

        monkeypatch.setattr(
            psth_webapp.spike_lfp_hilbert_phase,
            "compute_single_trial_spike_lfp_hilbert",
            fake_hilbert,
        )
        cached = psth_webapp.compute_single_trial_spike_lfp_hilbert_cached
        kwargs = {
            **common,
            "trial_index": 7,
            "event_time_s": 100.0,
            "window_start_s": -0.1,
            "window_end_s": 0.1,
            "band_low_hz": 6.0,
            "band_high_hz": 10.0,
            "filter_padding_s": 0.0,
            "minimum_envelope": 0.0,
        }
    cached.clear()
    _assert_open_ephys_semantics_token_recomputes(
        monkeypatch,
        ((lfp_path, sync_path),),
        cached.clear,
        lambda: cached(**kwargs),
        loaded_values,
        1,
    )
    loaded_values.clear()
    cached.clear()
    cached(**kwargs)
    cached(**kwargs)
    _rewrite_same_length_open_ephys_sidecar(sidecar_path)
    cached(**kwargs)

    assert len(loaded_values) == 2


def test_spikeglx_trace_cache_keeps_its_existing_value_and_identity_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Open Ephys token must not alter the established SpikeGLX trace cache or uV values."""
    load_calls: list[dict[str, object]] = []

    def fake_loader(**kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        """Return the unchanged physical-uV SpikeGLX trace on its normal grid."""
        load_calls.append(kwargs)
        return np.array([-0.1, 0.0]), np.array([10.0, 20.0])

    def forbidden_open_ephys_token(*_: object, **__: object) -> object:
        """Fail if a SpikeGLX route accidentally builds Open Ephys provenance."""
        raise AssertionError("SpikeGLX must not build an Open Ephys cache token")

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format", fake_loader)
    monkeypatch.setattr(
        psth_webapp,
        "build_open_ephys_cache_token",
        forbidden_open_ephys_token,
        raising=False,
    )
    psth_webapp.load_trial_lfp_trace_cached.clear()
    kwargs = {
        "lfp_format": psth_webapp.LFP_FORMAT_SPIKEGLX,
        "lfp_path": "pfc.lf.bin",
        "saved_channel_index": 2,
        "alignment_time_s": 100.0,
        "window_start_s": -0.1,
        "window_end_s": 0.1,
        "digital_word": 0,
        "irig_line": 6,
        "bit_period_s": 1.0,
        "utc_offset_hours": 0.0,
        "filter_low_hz": None,
        "filter_high_hz": None,
        "filter_padding_s": 0.0,
        "aligned_sync_npz_path": None,
    }
    first = psth_webapp.load_trial_lfp_trace_cached(**kwargs)
    second = psth_webapp.load_trial_lfp_trace_cached(**kwargs)

    assert load_calls == [kwargs]
    np.testing.assert_array_equal(first[1], np.array([10.0, 20.0]))
    np.testing.assert_array_equal(second[1], first[1])


class _HilbertSaveSidebar:
    """Minimal deterministic sidebar for one saved single-trial Hilbert exploratory route."""

    def __init__(self, lfp_format: str) -> None:
        self.lfp_format = lfp_format

    def header(self, _: str) -> None:
        """Accept the fixed exploratory-view header."""

    def selectbox(self, label: str, *, options: list[object] | tuple[object, ...], **_: object) -> object:
        """Select one valid unit/trial/probe source without changing numerical defaults."""
        if label == "LFP format":
            return self.lfp_format
        if label == "LFP probe":
            return psth_webapp.LFP_DROPDOWN_LABEL_PFC
        return options[0]

    def number_input(self, _: str, *, value: float | int, **__: object) -> float | int:
        """Keep documented window, channel, and timing defaults."""
        return value

    def caption(self, _: str) -> None:
        """Accept status text without rendering it."""


class _HilbertSaveStreamlit:
    """Small context-manager Streamlit seam that deterministically presses the save control."""

    def __init__(self, lfp_format: str) -> None:
        self.sidebar = _HilbertSaveSidebar(lfp_format)

    def __enter__(self) -> "_HilbertSaveStreamlit":
        """Provide a no-op Streamlit column context."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the no-op Streamlit column context."""

    def columns(self, _: list[int]) -> tuple["_HilbertSaveStreamlit", "_HilbertSaveStreamlit"]:
        """Return two context-managed columns without a browser runtime."""
        return self, self

    def error(self, message: str) -> None:
        """Raise unexpected UI errors as test failures."""
        raise AssertionError(message)

    def stop(self) -> None:
        """Raise unexpected Streamlit stops as test failures."""
        raise AssertionError("unexpected Streamlit stop")

    def warning(self, message: str) -> None:
        """Raise unexpected empty-selection warnings as test failures."""
        raise AssertionError(message)

    def subheader(self, _: str) -> None:
        """Accept fixed display text."""

    def write(self, _: object) -> None:
        """Accept metadata display values."""

    def caption(self, _: str) -> None:
        """Accept explanatory display text."""

    def pyplot(self, _: object, **__: object) -> None:
        """Accept the synthetic figure."""

    def button(self, label: str) -> bool:
        """Press only the output-save button under test."""
        return label == "Save single-trial spike-LFP phase result"

    def success(self, _: str) -> None:
        """Accept save confirmation text."""


class _ExploratorySaveSidebar(_HilbertSaveSidebar):
    """Reuse one deterministic control seam for the non-Hilbert saved exploratory views."""

    def __init__(self, lfp_format: str, save_label: str) -> None:
        super().__init__(lfp_format)
        self.save_label = save_label

    def selectbox(self, label: str, *, options: list[object] | tuple[object, ...], **_: object) -> object:
        """Choose valid distinct relative-phase sources and one PFC source elsewhere."""
        selected = {
            "LFP format": self.lfp_format,
            "Site A probe": psth_webapp.LFP_DROPDOWN_LABEL_HPC_V1,
            "Site B probe": psth_webapp.LFP_DROPDOWN_LABEL_PFC,
            "Site 1 probe": psth_webapp.LFP_DROPDOWN_LABEL_PFC,
            "LFP probe": psth_webapp.LFP_DROPDOWN_LABEL_PFC,
        }
        return selected.get(label, options[0])

    def checkbox(self, _: str, *, value: bool, **__: object) -> bool:
        """Retain existing Boolean analysis defaults without enabling optional branches."""
        return value

    def slider(self, _: str, *, value: float | int, **__: object) -> float | int:
        """Retain numeric masking defaults without calculating a browser widget."""
        return value


class _ExploratorySaveStreamlit(_HilbertSaveStreamlit):
    """Context-compatible Streamlit fake that presses one named exploratory save action."""

    def __init__(self, lfp_format: str, save_label: str) -> None:
        super().__init__(lfp_format)
        self.sidebar = _ExploratorySaveSidebar(lfp_format, save_label)
        self.save_label = save_label

    def button(self, label: str) -> bool:
        """Press precisely the save control owned by the public render boundary under test."""
        return label == self.save_label

    def selectbox(self, label: str, *, options: list[object] | tuple[object, ...], **kwargs: object) -> object:
        """Serve the one top-level result selector used by the phase-clustering route."""
        return self.sidebar.selectbox(label, options=options, **kwargs)


def _write_two_open_ephys_cache_sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    """Return two distinct tiny source/sync/sidecar identities for pair-view save contracts."""
    first_root = tmp_path / "site_a"
    second_root = tmp_path / "site_b"
    first_root.mkdir()
    second_root.mkdir()
    first_lfp, first_sync, first_sidecar = _write_open_ephys_cache_sources(first_root)
    second_lfp, second_sync, second_sidecar = _write_open_ephys_cache_sources(second_root)
    return first_lfp, first_sync, first_sidecar, second_lfp, second_sync, second_sidecar


def _one_valid_phase_trial() -> pd.DataFrame:
    """Return one selected trial satisfying the established phase-view condition mask contract."""
    return pd.DataFrame(
        {
            "choice_time": [100.0],
            "experimenter_reward_given": [0],
            "correct": [1],
            "reward": [1],
            "action": [1],
        }
    )


def _assert_exploratory_open_ephys_provenance(
    metadata: dict[str, object],
    lfp_format: str,
    sidecars: dict[Path, Path],
) -> None:
    """Require OE-only scalar or per-source sidecar provenance from a public save route."""
    semantics_key = "source_value_semantics"
    digest_key = "lfp_preprocessing_sha256"
    digests_by_source_key = "lfp_preprocessing_sha256_by_source"
    if lfp_format == psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED:
        assert metadata[semantics_key] == "open_ephys_affine_uV_v1"
        if len(sidecars) == 1:
            sidecar = next(iter(sidecars.values()))
            assert metadata[digest_key] == sha256(sidecar.read_bytes()).hexdigest()
            assert digests_by_source_key not in metadata
        else:
            assert metadata[digests_by_source_key] == {
                str(source.resolve()): sha256(sidecar.read_bytes()).hexdigest()
                for source, sidecar in sidecars.items()
            }
            assert digest_key not in metadata
    else:
        assert semantics_key not in metadata
        assert digest_key not in metadata
        assert digests_by_source_key not in metadata


@pytest.mark.parametrize(
    "lfp_format",
    (psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED, psth_webapp.LFP_FORMAT_SPIKEGLX),
)
def test_saved_relative_phase_exploration_records_physical_units_and_open_ephys_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfp_format: str,
) -> None:
    """The public pair-view save records its two physical source traces and OE sidecars."""
    lfp_a, sync_a, sidecar_a, lfp_b, sync_b, sidecar_b = _write_two_open_ephys_cache_sources(tmp_path)
    streamlit = _ExploratorySaveStreamlit(lfp_format, "Save single-trial phase-analysis result")
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        frequencies_hz=np.array([8.0]),
        support_relative_time_s=np.array([-0.1, 0.0]),
        support_relative_phase_complex=np.ones((1, 2), dtype=np.complex64),
    )

    monkeypatch.setattr(psth_webapp, "st", streamlit)
    monkeypatch.setattr(psth_webapp.unit_spike_plotting, "filter_trials_for_unit_plot", lambda **_: np.array([0]))
    monkeypatch.setattr(psth_webapp, "compute_single_trial_relative_phase_cached", lambda **_: result)
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "make_relative_phase_display_mask",
        lambda *_args, **_kwargs: np.ones((1, 2), dtype=bool),
    )
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "make_relative_phase_support_mask",
        lambda *_args, **_kwargs: np.ones((1, 2), dtype=bool),
    )
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "compute_within_trial_plv",
        lambda **_: SimpleNamespace(),
    )
    monkeypatch.setattr(psth_webapp.spike_behavior_pynapple, "build_lick_time_dict", lambda _: {})

    def fake_plot(**kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture the physical trace label passed by the real relative-phase renderer."""
        captured["lfp_y_label"] = kwargs["lfp_y_label"]
        return plt.figure(), {}

    def fake_save(*_: object, metadata: dict[str, object], **__: object) -> None:
        """Capture metadata assembled by the public save action without writing a result file."""
        captured["metadata"] = metadata

    monkeypatch.setattr(psth_webapp.unit_spike_plotting, "plot_trial_lfp_phase_analysis_and_behavior", fake_plot)
    monkeypatch.setattr(psth_webapp.lfp_phase_clustering, "save_single_trial_phase_analysis_result", fake_save)

    psth_webapp.render_single_trial_relative_phase_view(
        trial_df=_one_valid_phase_trial(),
        event_df=pd.DataFrame(),
        session=SimpleNamespace(session_data_home=tmp_path, sess_id_full="synthetic-session"),
        hpc_v1_lfp_path=str(lfp_a),
        pfc_lfp_path=str(lfp_b),
        hpc_v1_aligned_spike_path=str(sync_a),
        pfc_aligned_spike_path=str(sync_b),
    )

    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert captured["lfp_y_label"] == "LFP (uV)"
    assert metadata["phase_units"] == "radians, A minus B"
    if lfp_format == psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED:
        assert metadata["source_lfp_units"] == "uV"
    else:
        assert "source_lfp_units" not in metadata
    _assert_exploratory_open_ephys_provenance(
        metadata,
        lfp_format,
        {lfp_a: sidecar_a, lfp_b: sidecar_b},
    )


@pytest.mark.parametrize(
    "lfp_format",
    (psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED, psth_webapp.LFP_FORMAT_SPIKEGLX),
)
def test_saved_phase_clustering_exploration_records_open_ephys_provenance_without_raw_reopen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfp_format: str,
) -> None:
    """The public phase-clustering save records a selected OE source without a second source read."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    streamlit = _ExploratorySaveStreamlit(lfp_format, "Save phase-clustering results")
    captured: list[dict[str, object]] = []
    combined = SimpleNamespace(
        trial_indices=np.array([0], dtype=np.int64),
        phase=np.ones((1, 1, 1, 2), dtype=np.complex64),
        valid=np.ones((1, 1, 1, 2), dtype=bool),
        relative_time_s=np.array([-0.1, 0.0]),
        excluded_trial_indices=np.array([], dtype=np.int64),
    )
    all_site_result = SimpleNamespace(
        values=np.ones((1, 1, 2), dtype=float),
        effective_trial_count=np.ones((1, 1, 2), dtype=np.int64),
        n_trials=1,
    )
    source_reopens: list[str] = []

    monkeypatch.setattr(psth_webapp, "st", streamlit)
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "make_phase_condition_masks",
        lambda _: {**{name: np.array([True]) for name in psth_webapp.CONDITION_OPTIONS}, "all": np.array([True])},
    )
    monkeypatch.setattr(psth_webapp, "compute_lfp_phase_site_cached", lambda **_: object())
    monkeypatch.setattr(psth_webapp.lfp_phase_clustering, "combine_phase_trial_tensors", lambda _: combined)
    monkeypatch.setattr(psth_webapp.lfp_phase_clustering, "select_tensor_trial_mask", lambda *_: np.array([True]))
    monkeypatch.setattr(psth_webapp.lfp_phase_clustering, "compute_itpc", lambda *_args, **_kwargs: all_site_result)
    monkeypatch.setattr(psth_webapp.lfp_phase_clustering, "plot_phase_clustering", lambda *_args, **_kwargs: (plt.figure(), object()))
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "save_phase_clustering_result",
        lambda *_args, metadata, **_: captured.append(metadata),
    )

    def forbidden_reopen(*_: object, **__: object) -> object:
        """Reject any source-reader use after the cached phase object has been supplied."""
        source_reopens.append("read")
        raise AssertionError("phase-clustering save reopened raw LFP")

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format_with_sample_rate", forbidden_reopen)

    psth_webapp.render_lfp_phase_clustering_view(
        trial_df=_one_valid_phase_trial(),
        session=SimpleNamespace(session_data_home=tmp_path, sess_id_full="synthetic-session"),
        hpc_v1_lfp_path=str(lfp_path),
        pfc_lfp_path=str(lfp_path),
        hpc_v1_aligned_spike_path=str(sync_path),
        pfc_aligned_spike_path=str(sync_path),
    )

    assert source_reopens == []
    assert len(captured) == 1
    assert captured[0]["value_units"] == "dimensionless phase consistency"
    _assert_exploratory_open_ephys_provenance(captured[0], lfp_format, {lfp_path: sidecar_path})


@pytest.mark.parametrize(
    "lfp_format",
    (psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED, psth_webapp.LFP_FORMAT_SPIKEGLX),
)
def test_saved_spike_phase_locking_exploration_records_open_ephys_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfp_format: str,
) -> None:
    """The public phase-locking save records OE semantics while preserving SpikeGLX metadata."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    streamlit = _ExploratorySaveStreamlit(lfp_format, "Save spike-LFP phase-locking result")
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        spike_times_s=np.array([100.0]),
        spike_phase_vectors=np.ones((1, 1), dtype=np.complex64),
        spike_phase_valid=np.ones((1, 1), dtype=bool),
        amplitude_at_spikes=np.ones((1, 1), dtype=float),
        phase_bin_edges_rad=np.array([-np.pi, np.pi]),
        phase_spike_counts=np.ones((1, 1), dtype=np.int64),
        phase_occupancy_s=np.ones((1, 1), dtype=float),
        phase_firing_rate_hz=np.ones((1, 1), dtype=float),
    )
    sparsity = SimpleNamespace(
        is_sparse=False,
        valid_spike_count=1,
        mean_spikes_per_bin=1.0,
        minimum_recommended_spikes=1,
        zero_occupancy_bin_count=0,
        phase_bin_count=1,
        minimum_mean_spikes_per_bin=1.0,
    )

    monkeypatch.setattr(psth_webapp, "st", streamlit)
    monkeypatch.setattr(psth_webapp.unit_spike_plotting, "filter_trials_for_unit_plot", lambda **_: np.array([0]))
    monkeypatch.setattr(psth_webapp.unit_spike_loading, "get_unit_spike_times", lambda *_: np.array([100.0]))
    monkeypatch.setattr(psth_webapp, "compute_spike_lfp_phase_locking_cached", lambda **_: result)
    monkeypatch.setattr(psth_webapp.unit_spike_plotting, "plot_spike_lfp_phase_locking", lambda **_: (plt.figure(), {}))
    monkeypatch.setattr(psth_webapp.spike_lfp_phase_locking, "assess_phase_rate_sparsity", lambda **_: sparsity)
    monkeypatch.setattr(
        psth_webapp.spike_lfp_phase_locking,
        "save_spike_lfp_phase_locking_result",
        lambda *, metadata, **_: captured.setdefault("metadata", metadata),
    )

    psth_webapp.render_spike_lfp_phase_locking_view(
        trial_df=pd.DataFrame({"choice_time": [100.0]}),
        session=SimpleNamespace(session_data_home=tmp_path, sess_id_full="synthetic-session"),
        spike_group=object(),
        unit_ids=np.array([7]),
        active_probe_label=psth_webapp.LFP_DROPDOWN_LABEL_PFC,
        hpc_v1_lfp_path=str(lfp_path),
        pfc_lfp_path=str(lfp_path),
        hpc_v1_aligned_spike_path=str(sync_path),
        pfc_aligned_spike_path=str(sync_path),
    )

    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["generator"] == "src.neural_analysis.spike_lfp_phase_locking"
    assert metadata["analysis_version"] == "0.2.0"
    assert metadata["phase_units"] == "radians"
    if lfp_format == psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED:
        assert metadata["amplitude_units"] == "uV"
    else:
        assert metadata["amplitude_units"] == "source-dependent wavelet magnitude"
    _assert_exploratory_open_ephys_provenance(metadata, lfp_format, {lfp_path: sidecar_path})


class _SpectrogramMainSidebar:
    """Deterministic controls for the public trial-view spectrogram route."""

    def __init__(self, root: Path, lfp_path: Path, sync_path: Path, lfp_format: str) -> None:
        self.root = root
        self.lfp_path = lfp_path
        self.sync_path = sync_path
        self.lfp_format = lfp_format

    def button(self, _: str) -> bool:
        """Do not clear cache state during the isolated route test."""
        return False

    def header(self, _: str) -> None:
        """Accept fixed sidebar headings."""

    def text_input(self, label: str, *, value: str = "", **_: object) -> str:
        """Return only temporary selected-source paths for the public route."""
        if label == "Session data home":
            return str(self.root)
        if "LFP path" in label:
            return str(self.lfp_path)
        if "aligned spike path" in label:
            return str(self.sync_path)
        if label == "Open Ephys aligned sync path":
            return str(self.sync_path)
        return value

    def selectbox(self, label: str, *, options: list[object] | tuple[object, ...], **_: object) -> object:
        """Select the public LFP-spectrogram route while retaining valid defaults elsewhere."""
        selected = {
            "Plot view": psth_webapp.PLOT_VIEW_TRIAL_SPIKES,
            "Channel source": psth_webapp.CHANNEL_SOURCE_MANUAL,
            "Neural display": psth_webapp.NEURAL_DISPLAY_LFP_SPECTROGRAM,
            "LFP format": self.lfp_format,
            "Active LFP file": psth_webapp.LFP_DROPDOWN_LABEL_PFC,
        }
        return selected.get(label, options[0])

    def number_input(self, _: str, *, value: float | int, **__: object) -> float | int:
        """Retain numeric defaults for windows, frequencies, and selected channel."""
        return value

    def checkbox(self, _: str, *, value: bool, **__: object) -> bool:
        """Retain documented Boolean display/filter defaults."""
        return value

    def text_area(self, _: str, *, value: str, **__: object) -> str:
        """Return a syntactically valid manual saved-channel selection."""
        return "0" if not value else value

    def caption(self, _: str) -> None:
        """Accept noninteractive status text."""


class _SpectrogramMainStreamlit(_HilbertSaveStreamlit):
    """No-browser Streamlit seam for the ordinary public spectrogram display route."""

    def __init__(self, root: Path, lfp_path: Path, sync_path: Path, lfp_format: str) -> None:
        super().__init__(lfp_format)
        self.sidebar = _SpectrogramMainSidebar(root, lfp_path, sync_path, lfp_format)

    def set_page_config(self, **_: object) -> None:
        """Accept the public app page configuration."""

    def title(self, _: str) -> None:
        """Accept the public app title."""

    def dataframe(self, *_: object, **__: object) -> None:
        """Accept a compact metadata table without rendering it."""


@pytest.mark.parametrize(
    "lfp_format",
    (psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED, psth_webapp.LFP_FORMAT_SPIKEGLX),
)
def test_public_spectrogram_route_labels_open_ephys_and_spikeglx_power_relative_to_uv_squared(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfp_format: str,
) -> None:
    """The live exploratory spectrogram passes physical-uV and relative-uV-squared labels to its plotter."""
    lfp_path, sync_path, _ = _write_open_ephys_cache_sources(tmp_path)
    lfp_path.with_suffix(".meta").write_text("", encoding="ascii")
    captured: dict[str, object] = {}
    streamlit = _SpectrogramMainStreamlit(tmp_path, lfp_path, sync_path, lfp_format)
    trial_df = pd.DataFrame(
        {
            "choice_time": [100.0],
            "experimenter_reward_given": [0],
            "correct": [1],
            "reward": [1],
            "action": [1],
        }
    )
    viewer_data = {
        "session": SimpleNamespace(session_data_home=tmp_path, sess_id_full="synthetic-session"),
        "event_df": pd.DataFrame(),
        "trial_df": trial_df,
        "cluster_info": pd.DataFrame({"cluster_id": [7]}),
        "spike_group": object(),
    }
    spectrogram_result = SimpleNamespace(
        time_s=np.array([-0.1, 0.0]),
        frequencies_hz=np.array([8.0]),
        log_power_db=np.zeros((2, 1)),
        lfp_values=np.array([1.95, 3.9]),
    )

    monkeypatch.setattr(psth_webapp, "st", streamlit)
    monkeypatch.setattr(psth_webapp, "_initialize_path_input_state", lambda: None)
    monkeypatch.setattr(psth_webapp, "_render_path_browser", lambda: None)
    monkeypatch.setattr(psth_webapp.unit_spike_loading, "get_probe_label_for_region", lambda *_args, **_kwargs: "PFC")
    monkeypatch.setattr(psth_webapp.unit_spike_loading, "infer_probe_derived_dir", lambda **_: None)
    monkeypatch.setattr(psth_webapp, "resolve_region_channels_for_source", lambda **_: (np.array([0]), "channel 0"))
    monkeypatch.setattr(psth_webapp, "load_viewer_data_cached", lambda **_: viewer_data)
    monkeypatch.setattr(psth_webapp, "_select_unit_metadata", lambda *_: pd.DataFrame({"cluster_id": [7]}))
    monkeypatch.setattr(psth_webapp.unit_spike_plotting, "filter_trials_for_unit_plot", lambda *_args, **_kwargs: np.array([0]))
    monkeypatch.setattr(psth_webapp.unit_spike_plotting, "get_trial_alignment_time", lambda **_: 100.0)
    monkeypatch.setattr(psth_webapp.spike_behavior_pynapple, "build_lick_time_dict", lambda _: {})
    monkeypatch.setattr(psth_webapp, "compute_trial_lfp_spectrogram_cached", lambda **_: spectrogram_result)
    monkeypatch.setattr(psth_webapp, "compute_shared_lfp_power_limits_cached", lambda **_: ((-1.0, 1.0), 1))

    def fake_plot(**kwargs: object) -> tuple[object, tuple[object, object, object]]:
        """Capture the exact LFP and power labels delivered by the public route."""
        captured["lfp_y_label"] = kwargs["lfp_y_label"]
        captured["power_unit_label"] = kwargs["power_unit_label"]
        figure, axes = plt.subplots(3, 1)
        return figure, tuple(axes)

    monkeypatch.setattr(
        psth_webapp.unit_spike_plotting,
        "plot_trial_lfp_spectrogram_and_behavior",
        fake_plot,
    )

    psth_webapp.main()

    assert captured == {
        "lfp_y_label": "LFP (uV)",
        "power_unit_label": "dB re 1 uV^2",
    }


@pytest.mark.parametrize(
    "lfp_format",
    (psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED, psth_webapp.LFP_FORMAT_SPIKEGLX),
)
def test_saved_hilbert_exploration_uses_truthful_units_and_open_ephys_only_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    lfp_format: str,
) -> None:
    """Saved exploratory Hilbert output labels uV values and records only OE scaling provenance."""
    lfp_path, sync_path, sidecar_path = _write_open_ephys_cache_sources(tmp_path)
    streamlit = _HilbertSaveStreamlit(lfp_format)
    captured: dict[str, object] = {}

    result = SimpleNamespace(
        spike_times_relative_s=np.array([0.0]),
        spike_phase_valid=np.array([True]),
        source_sample_rate_hz=2500.0,
    )

    monkeypatch.setattr(psth_webapp, "st", streamlit)
    monkeypatch.setattr(
        psth_webapp.unit_spike_plotting,
        "filter_trials_for_unit_plot",
        lambda **_: np.array([0]),
    )
    monkeypatch.setattr(psth_webapp.unit_spike_loading, "get_unit_spike_times", lambda *_: np.array([100.0]))
    monkeypatch.setattr(psth_webapp.spike_behavior_pynapple, "build_lick_time_dict", lambda _: {})
    monkeypatch.setattr(psth_webapp, "compute_single_trial_spike_lfp_hilbert_cached", lambda **_: result)

    def fake_plot(**kwargs: object) -> tuple[object, dict[str, object]]:
        """Capture the raw/filtered voltage label passed to the real plotting boundary."""
        captured["lfp_y_label"] = kwargs["lfp_y_label"]
        return plt.figure(), {}

    def fake_save(*, metadata: dict[str, object], **_: object) -> None:
        """Capture only the saved JSON-compatible exploratory provenance mapping."""
        captured["metadata"] = metadata

    monkeypatch.setattr(
        psth_webapp.unit_spike_plotting,
        "plot_trial_spike_lfp_hilbert_phase_and_behavior",
        fake_plot,
    )
    monkeypatch.setattr(
        psth_webapp.spike_lfp_hilbert_phase,
        "save_single_trial_spike_lfp_hilbert_result",
        fake_save,
    )
    session = SimpleNamespace(session_data_home=tmp_path, sess_id_full="synthetic-session")
    trial_df = pd.DataFrame({"choice_time": [100.0]})

    psth_webapp.render_single_trial_spike_lfp_hilbert_view(
        trial_df=trial_df,
        event_df=pd.DataFrame(),
        session=session,
        spike_group=object(),
        unit_ids=np.array([7]),
        active_probe_label=psth_webapp.LFP_DROPDOWN_LABEL_PFC,
        hpc_v1_lfp_path=str(lfp_path),
        pfc_lfp_path=str(lfp_path),
        hpc_v1_aligned_spike_path=str(sync_path),
        pfc_aligned_spike_path=str(sync_path),
    )

    metadata = captured["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["generator"] == "src.neural_analysis.spike_lfp_hilbert_phase"
    assert metadata["analysis_version"] == "0.1.0"
    assert captured["lfp_y_label"] == "LFP (uV)"
    assert metadata["raw_lfp_units"] == "uV"
    assert metadata["filtered_lfp_units"] == "uV"
    if lfp_format == psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED:
        assert metadata["source_value_semantics"] == "open_ephys_affine_uV_v1"
        assert metadata["lfp_preprocessing_sha256"] == sha256(sidecar_path.read_bytes()).hexdigest()
    else:
        assert "source_value_semantics" not in metadata
        assert "lfp_preprocessing_sha256" not in metadata


def test_open_ephys_nested_cache_paths_construct_one_source_token_and_reuse_it_without_numerics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Nested power and phase views reuse selected-source identities instead of rehashing raw inputs."""
    lfp_path, sync_path, _ = _write_open_ephys_cache_sources(tmp_path)
    second_root = tmp_path / "second_site"
    second_root.mkdir()
    lfp_path_b, sync_path_b, _ = _write_open_ephys_cache_sources(second_root)
    build_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    real_builder = getattr(psth_webapp, "build_open_ephys_cache_token", None)

    def counted_builder(*args: object, **kwargs: object) -> object:
        """Count source-boundary identity construction while retaining a hashable token."""
        build_calls.append((args, kwargs))
        if real_builder is not None:
            return real_builder(*args, **kwargs)
        return (tuple(map(str, args)), tuple(sorted((str(key), str(value)) for key, value in kwargs.items())))

    monkeypatch.setattr(
        psth_webapp,
        "build_open_ephys_cache_token",
        counted_builder,
        raising=False,
    )
    spectrogram_calls: list[dict[str, object]] = []

    def fake_spectrogram(**kwargs: object) -> SimpleNamespace:
        """Return a tiny power array while preserving the nested call boundary."""
        spectrogram_calls.append(kwargs)
        return SimpleNamespace(log_power_db=np.zeros((1, 1)))

    monkeypatch.setattr(psth_webapp, "compute_trial_lfp_spectrogram_cached", fake_spectrogram)
    psth_webapp.compute_shared_lfp_power_limits_cached(
        reference_alignment_times_s=(100.0, 101.0),
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path),
        saved_channel_index=0,
        visible_window_start_s=-0.1,
        visible_window_end_s=0.1,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        frequencies_hz=(8.0,),
        gaussian_width=1.5,
        window_length=1.0,
        precision=8,
        norm="l1",
        target_sample_rate_hz=500.0,
        notch_60_hz=False,
        notch_quality_factor=30.0,
        lower_percentile=5.0,
        upper_percentile=95.0,
        aligned_sync_npz_path=str(sync_path),
    )
    assert len(build_calls) == 1
    assert len(spectrogram_calls) == 2

    def fake_phase_tensor(**_: object) -> object:
        """Avoid a Morlet transform while preserving the cache wrapper's source boundary."""
        return object()

    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "compute_site_phase_trial_tensor",
        fake_phase_tensor,
    )
    psth_webapp.compute_lfp_phase_site_cached(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path),
        lfp_file_mtime_ns=lfp_path.stat().st_mtime_ns,
        saved_channel_index=0,
        event_times_s=(100.0,),
        trial_indices=(0,),
        window_start_s=-0.1,
        window_end_s=0.1,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        frequencies_hz=(8.0,),
        gaussian_width=1.5,
        window_length=1.0,
        precision=8,
        norm="l1",
        output_sample_rate_hz=500.0,
        notch_60_hz=False,
        notch_quality_factor=30.0,
        minimum_relative_magnitude=1e-12,
        maximum_core_duration_s=10.0,
        aligned_sync_npz_path=str(sync_path),
    )
    assert len(build_calls) == 2

    def fake_load_with_rate(**_: object) -> tuple[np.ndarray, np.ndarray, float]:
        """Return a tiny source-rate uV signal for relative phase and Hilbert wrappers."""
        return np.array([-0.1, 0.0, 0.1]), np.array([1.0, 2.0, 3.0]), 10.0

    monkeypatch.setattr(
        psth_webapp,
        "load_trial_lfp_trace_for_format_with_sample_rate",
        fake_load_with_rate,
    )
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "compute_wavelet_coefficients",
        lambda **_: object(),
    )
    monkeypatch.setattr(
        psth_webapp.lfp_phase_clustering,
        "compute_single_trial_relative_phase",
        lambda **_: object(),
    )
    psth_webapp.compute_single_trial_relative_phase_cached(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path_a=str(lfp_path), lfp_mtime_ns_a=lfp_path.stat().st_mtime_ns,
        channel_a=0, site_a_label="PFC", aligned_sync_path_a=str(sync_path),
        aligned_sync_mtime_ns_a=sync_path.stat().st_mtime_ns,
        lfp_path_b=str(lfp_path_b), lfp_mtime_ns_b=lfp_path_b.stat().st_mtime_ns,
        channel_b=0, site_b_label="HPC", aligned_sync_path_b=str(sync_path_b),
        aligned_sync_mtime_ns_b=sync_path_b.stat().st_mtime_ns,
        trial_index=0, event_time_s=100.0, window_start_s=-0.1, window_end_s=0.1,
        digital_word=0, irig_line=6, bit_period_s=1.0, utc_offset_hours=0.0,
        frequencies_hz=(8.0,), gaussian_width=1.5, wavelet_window_length=1.0,
        precision=8, norm="l1", output_sample_rate_hz=500.0, notch_60_hz=False,
        notch_quality_factor=30.0, minimum_relative_magnitude=1e-12,
        plv_window_cycles=3.0, plv_min_window_s=None, plv_max_window_s=None,
    )
    assert len(build_calls) == 4

    monkeypatch.setattr(
        psth_webapp.spike_lfp_phase_locking,
        "compute_trial_aligned_spike_phase_locking",
        lambda **_: object(),
    )
    psth_webapp.compute_spike_lfp_phase_locking_cached(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path), lfp_mtime_ns=lfp_path.stat().st_mtime_ns,
        saved_channel_index=0, lfp_site_label="PFC", aligned_sync_path=str(sync_path),
        aligned_sync_mtime_ns=sync_path.stat().st_mtime_ns, unit_id=1,
        unit_spike_times_s=(100.0,), trial_indices=(0,), event_times_s=(100.0,),
        window_start_s=-0.1, window_end_s=0.1, digital_word=0, irig_line=6,
        bit_period_s=1.0, utc_offset_hours=0.0, frequencies_hz=(8.0,),
        gaussian_width=1.5, wavelet_window_length=1.0, precision=8, norm="l1",
        target_sample_rate_hz=500.0, notch_60_hz=False, notch_quality_factor=30.0,
        minimum_relative_magnitude=1e-12, absolute_amplitude_threshold=0.0,
        maximum_core_duration_s=10.0, phase_bin_count=12,
    )
    assert len(build_calls) == 5

    monkeypatch.setattr(
        psth_webapp.spike_lfp_hilbert_phase,
        "compute_single_trial_spike_lfp_hilbert",
        lambda **_: object(),
    )
    psth_webapp.compute_single_trial_spike_lfp_hilbert_cached(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path), lfp_mtime_ns=lfp_path.stat().st_mtime_ns,
        saved_channel_index=0, lfp_site_label="PFC", aligned_sync_path=str(sync_path),
        aligned_sync_mtime_ns=sync_path.stat().st_mtime_ns, unit_id=1,
        unit_spike_times_s=(100.0,), trial_index=0, event_time_s=100.0,
        window_start_s=-0.1, window_end_s=0.1, band_low_hz=6.0, band_high_hz=10.0,
        filter_padding_s=0.0, digital_word=0, irig_line=6, bit_period_s=1.0,
        utc_offset_hours=0.0, minimum_envelope=0.0,
    )
    assert len(build_calls) == 6


def test_single_trial_spike_lfp_hilbert_cache_loads_one_padded_source_rate_trace(monkeypatch):
    """The single-trial cache should load exactly one padded raw LFP interval."""

    load_calls = []

    def fake_loader(**kwargs):
        """Record the requested bounds and return a deterministic source-rate segment."""

        load_calls.append(kwargs)
        time_s = np.arange(kwargs["window_start_s"], kwargs["window_end_s"], 0.01)
        return time_s, np.sin(2.0 * np.pi * 8.0 * time_s), 100.0

    monkeypatch.setattr(psth_webapp, "load_trial_lfp_trace_for_format_with_sample_rate", fake_loader)
    psth_webapp.compute_single_trial_spike_lfp_hilbert_cached.clear()
    result = psth_webapp.compute_single_trial_spike_lfp_hilbert_cached(
        lfp_format=psth_webapp.LFP_FORMAT_SPIKEGLX,
        lfp_path="pfc.lf.bin",
        lfp_mtime_ns=1,
        saved_channel_index=4,
        lfp_site_label="PFC channel 4",
        aligned_sync_path=None,
        aligned_sync_mtime_ns=-1,
        unit_id=12,
        unit_spike_times_s=(99.8, 100.2),
        trial_index=3,
        event_time_s=100.0,
        window_start_s=-1.0,
        window_end_s=2.0,
        band_low_hz=6.0,
        band_high_hz=10.0,
        filter_padding_s=1.0,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        minimum_envelope=0.0,
    )

    assert len(load_calls) == 1
    assert load_calls[0]["window_start_s"] == -2.0
    assert load_calls[0]["window_end_s"] == 3.0
    assert result.source_sample_rate_hz == 100.0


def test_spike_lfp_phase_locking_cache_phase_bin_count_is_explicitly_cacheable():
    """The phase-bin setting must participate in the cached computation contract."""
    parameter = inspect.signature(psth_webapp.compute_spike_lfp_phase_locking_cached).parameters[
        "phase_bin_count"
    ]

    assert parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD


def test_build_lfp_dropdown_options_preserves_empty_paths():
    """Empty LFP fields should stay selectable and be handled by later validation."""
    options = psth_webapp.build_lfp_dropdown_options(
        hpc_v1_lfp_path="",
        pfc_lfp_path="/data/pfc/run0_g0_t0.imec0.lf.bin",
    )

    assert options["HPC/V1 LFP"] == ""
    assert options["PFC LFP"] == "/data/pfc/run0_g0_t0.imec0.lf.bin"


def test_resolve_lfp_filter_band_maps_display_labels_to_cutoffs():
    """Displayed LFP filter labels should resolve to the intended frequency bands."""
    assert psth_webapp.resolve_lfp_filter_band("Default") is None
    assert psth_webapp.resolve_lfp_filter_band("Theta (5-10 Hz)") == (5.0, 10.0)
    assert psth_webapp.resolve_lfp_filter_band("Gamma (50-70 Hz)") == (50.0, 70.0)


def test_build_trial_view_plot_save_path_includes_plot_settings(tmp_path: Path):
    """Trial-view saved plots should be distinguished by the settings that define them."""
    save_path = psth_webapp.build_trial_view_plot_save_path(
        figure_path=tmp_path,
        session_id="CT014_2025-12-23_163505",
        region_name="HPC",
        trial_index=12,
        condition="correct_rewarded",
        action_label="left (1)",
        alignment_event="choice_time",
        window=(-2.0, 1.5),
        unit_page_index=3,
        population_psth_unit_scope="All selected units",
        population_psth_bin_size=0.05,
        lfp_label="HPC/V1 LFP, saved channel 12, Theta (5-10 Hz)",
        lfp_utc_offset_hours=1,
    )

    assert save_path.parent == tmp_path / "unit_spike_viewer"
    assert save_path.suffix == ".png"
    filename = save_path.name
    for expected_token in (
        "CT014_2025-12-23_163505",
        "HPC",
        "trial12",
        "correct_rewarded",
        "left-1",
        "choice_time",
        "window-2.0-to-1.5s",
        "unitpage3",
        "All-selected-units",
        "popbin0.05s",
        "HPC-V1-LFP-saved-channel-12-Theta-5-10-Hz",
        "lfputc1h",
    ):
        assert expected_token in filename


def test_format_metadata_row_for_display_keeps_mixed_values_arrow_safe():
    """Mixed numeric/string unit metadata should be shown as strings for Streamlit."""
    metadata_row = pd.Series(
        {
            "cluster_id": np.int64(12),
            "depth": 250.5,
            "group": "mua",
            "missing": np.nan,
        }
    )

    display_df = psth_webapp.format_metadata_row_for_display(metadata_row)

    assert display_df.index.tolist() == ["cluster_id", "depth", "group", "missing"]
    assert display_df.columns.tolist() == ["value"]
    assert display_df["value"].tolist() == ["12", "250.5", "mua", ""]
    assert all(isinstance(value, str) for value in display_df["value"].tolist())


def test_is_population_pca_display_includes_concatenated_mode():
    """Both PCA display modes should route through PCA computation and plotting."""
    assert psth_webapp.is_population_pca_display(psth_webapp.NEURAL_DISPLAY_POPULATION_PCA)
    assert psth_webapp.is_population_pca_display(psth_webapp.NEURAL_DISPLAY_POPULATION_PCA_CONCATENATED)
    assert not psth_webapp.is_population_pca_display(psth_webapp.NEURAL_DISPLAY_SPIKE_RASTER)


def test_webapp_exposes_single_trial_lfp_spectrogram_defaults():
    """The dedicated LFP view should expose the approved scientific defaults."""
    assert psth_webapp.NEURAL_DISPLAY_LFP_SPECTROGRAM in psth_webapp.NEURAL_DISPLAY_OPTIONS
    assert psth_webapp.LFP_SPECTROGRAM_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.LFP_SPECTROGRAM_MAX_FREQUENCY_HZ == 80.0
    assert psth_webapp.LFP_SPECTROGRAM_FREQUENCY_COUNT == 40
    assert psth_webapp.LFP_SPECTROGRAM_REFERENCE_TRIAL_COUNT == 24
    assert psth_webapp.LFP_SPECTROGRAM_COLOR_PERCENTILES == (2.0, 98.0)
    assert psth_webapp.LFP_SPECTROGRAM_NOTCH_DEFAULT is False


def test_webapp_exposes_separate_lfp_phase_clustering_view_defaults():
    """ITPC/ISPC should use a dedicated view with the approved initial settings."""
    assert psth_webapp.PLOT_VIEW_LFP_PHASE_CLUSTERING in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.LFP_PHASE_CLUSTERING_ANALYSIS_OPTIONS == ("ITPC", "ISPC")
    assert psth_webapp.LFP_PHASE_CLUSTERING_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.LFP_PHASE_CLUSTERING_MAX_FREQUENCY_HZ == 100.0
    assert psth_webapp.LFP_PHASE_CLUSTERING_FREQUENCY_COUNT == 50
    assert psth_webapp.LFP_PHASE_CLUSTERING_OUTPUT_SAMPLE_RATE_HZ == 500.0
    assert psth_webapp.LFP_PHASE_CLUSTERING_DEFAULT_WINDOW == (-1.0, 2.0)


def test_webapp_exposes_single_trial_relative_phase_view_defaults():
    """Relative phase should be a lightweight top-level single-trial view."""
    assert psth_webapp.PLOT_VIEW_SINGLE_TRIAL_RELATIVE_PHASE in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.RELATIVE_PHASE_DEFAULT_WINDOW == (-1.0, 2.0)
    assert psth_webapp.RELATIVE_PHASE_MIN_FREQUENCY_HZ == 2.0
    assert psth_webapp.RELATIVE_PHASE_MAX_FREQUENCY_HZ == 100.0
    assert psth_webapp.RELATIVE_PHASE_FREQUENCY_COUNT == 50
    assert psth_webapp.RELATIVE_PHASE_OUTPUT_SAMPLE_RATE_HZ == 500.0
    assert psth_webapp.RELATIVE_PHASE_AMPLITUDE_MASK_OPTIONS == (
        "Off",
        "Per-frequency percentile",
        "Absolute magnitude",
    )
    assert psth_webapp.RELATIVE_PHASE_DISPLAY_OPTIONS == (
        "Phase difference",
        "Within-trial PLV",
        "Phase + PLV",
    )
    assert psth_webapp.RELATIVE_PHASE_DEFAULT_DISPLAY == "Within-trial PLV"
    assert psth_webapp.WITHIN_TRIAL_PLV_DEFAULT_WINDOW_CYCLES == 3.0
    assert psth_webapp.WITHIN_TRIAL_PLV_DEFAULT_MIN_VALID_FRACTION == 0.8
    assert psth_webapp.WITHIN_TRIAL_PLV_MIN_WINDOW_DEFAULT_ENABLED is False
    assert psth_webapp.WITHIN_TRIAL_PLV_MAX_WINDOW_DEFAULT_ENABLED is False
    assert psth_webapp.RELATIVE_PHASE_CACHE_MAX_ENTRIES == 12


def test_single_trial_relative_phase_cache_loads_only_two_padded_trial_segments(monkeypatch):
    """The lightweight viewer must not load or construct a full-session trial tensor."""
    load_calls = []

    def fake_load_trial_lfp_trace_for_format_with_sample_rate(**kwargs):
        """Record requested relative bounds and return one synthetic source-rate segment."""
        load_calls.append((kwargs["lfp_path"], kwargs["window_start_s"], kwargs["window_end_s"]))
        relative_time_s = np.arange(kwargs["window_start_s"], kwargs["window_end_s"], 0.002)
        phase_offset = 0.4 if kwargs["saved_channel_index"] == 1 else 0.0
        values = np.sin(2.0 * np.pi * 10.0 * relative_time_s + phase_offset)
        return relative_time_s, values, 500.0

    def fake_compute_wavelet_coefficients(time_s, lfp_values, sample_rate_hz, frequencies_hz, **_kwargs):
        """Return deterministic coefficients on the loaded absolute sample grid."""
        times = np.asarray(time_s, dtype=float)
        frequencies = np.asarray(frequencies_hz, dtype=float)
        coefficients = np.stack(
            [np.exp(1j * 2.0 * np.pi * frequency * times) for frequency in frequencies]
        ).astype(np.complex64)
        return lfp_phase_clustering.WaveletCoefficientResult(
            time_s=times,
            frequencies_hz=frequencies,
            coefficients=coefficients,
            sample_rate_hz=float(sample_rate_hz),
            source_time_s=times,
            source_lfp_values=np.asarray(lfp_values),
            source_sample_rate_hz=float(sample_rate_hz),
        )

    monkeypatch.setattr(
        psth_webapp,
        "load_trial_lfp_trace_for_format_with_sample_rate",
        fake_load_trial_lfp_trace_for_format_with_sample_rate,
    )
    monkeypatch.setattr(
        lfp_phase_clustering,
        "compute_wavelet_coefficients",
        fake_compute_wavelet_coefficients,
    )
    psth_webapp.compute_single_trial_relative_phase_cached.clear()

    result = psth_webapp.compute_single_trial_relative_phase_cached(
        lfp_format=psth_webapp.LFP_FORMAT_SPIKEGLX,
        lfp_path_a="a.lf.bin",
        lfp_mtime_ns_a=1,
        channel_a=1,
        site_a_label="A",
        aligned_sync_path_a=None,
        aligned_sync_mtime_ns_a=-1,
        lfp_path_b="b.lf.bin",
        lfp_mtime_ns_b=2,
        channel_b=2,
        site_b_label="B",
        aligned_sync_path_b=None,
        aligned_sync_mtime_ns_b=-1,
        trial_index=3,
        event_time_s=100.0,
        window_start_s=-1.0,
        window_end_s=2.0,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        frequencies_hz=(2.0, 10.0),
        gaussian_width=1.5,
        wavelet_window_length=1.0,
        precision=16,
        norm="l1",
        output_sample_rate_hz=500.0,
        notch_60_hz=False,
        notch_quality_factor=30.0,
        minimum_relative_magnitude=1e-12,
        plv_window_cycles=3.0,
        plv_min_window_s=None,
        plv_max_window_s=None,
    )

    assert result.phase_angle_rad.shape == (2, 1500)
    assert load_calls == [("a.lf.bin", -5.75, 6.75), ("b.lf.bin", -5.75, 6.75)]
    assert result.support_relative_time_s[0] == -1.75
    assert np.isclose(result.support_relative_time_s[-1], 2.748)


def test_pca_decoding_has_separate_plot_view_from_trial_filters():
    """PCA decoding should be a separate performance-only view, not a trial display mode."""
    assert psth_webapp.PLOT_VIEW_PCA_DECODING in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.PLOT_VIEW_PCA_DECODING != "Trial spikes/licks/choices"
    assert psth_webapp.PCA_DECODING_DEFAULT_COMPONENT_COUNT == 5
    assert psth_webapp.PCA_DECODING_DISPLAY_PERFORMANCE in psth_webapp.PCA_DECODING_DISPLAY_OPTIONS
    assert psth_webapp.PCA_DECODING_DISPLAY_AVERAGE_PC in psth_webapp.PCA_DECODING_DISPLAY_OPTIONS
    assert psth_webapp.PCA_DECODING_SHOW_RAW_PC_SCORES_DEFAULT is False


def test_pca_switch_trajectories_have_a_separate_plot_view():
    """Choice-switch trajectories should not reuse decoding or trial-view controls."""
    assert psth_webapp.PLOT_VIEW_PCA_SWITCH_TRAJECTORIES in psth_webapp.PLOT_VIEW_OPTIONS
    assert psth_webapp.PLOT_VIEW_PCA_SWITCH_TRAJECTORIES != psth_webapp.PLOT_VIEW_PCA_DECODING


def test_resolve_pca_decoding_component_minimum_requires_two_for_pc_score_plot():
    """Average PC score plots need PC1 and PC2 while performance decoding can use one PC."""
    assert (
        psth_webapp.resolve_pca_decoding_component_minimum(
            psth_webapp.PCA_DECODING_DISPLAY_AVERAGE_PC
        )
        == 2
    )
    assert (
        psth_webapp.resolve_pca_decoding_component_minimum(
            psth_webapp.PCA_DECODING_DISPLAY_PERFORMANCE
        )
        == 1
    )


def test_select_visible_concatenated_trial_indices_limits_page_size():
    """Concatenated PCA should display a bounded page of selected trials."""
    trial_indices = np.arange(25, dtype=int)

    visible_trial_indices, page_count = psth_webapp.select_visible_concatenated_trial_indices(
        trial_indices=trial_indices,
        page_index=1,
        visible_trial_count=10,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(10, 20, dtype=int))
    assert page_count == 3


def test_select_visible_concatenated_trial_indices_clamps_oversized_page():
    """Out-of-range pages should resolve to the last available visible trial page."""
    trial_indices = np.arange(23, dtype=int)

    visible_trial_indices, page_count = psth_webapp.select_visible_concatenated_trial_indices(
        trial_indices=trial_indices,
        page_index=999,
        visible_trial_count=10,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(20, 23, dtype=int))
    assert page_count == 3


def test_select_concatenated_trial_viewport_uses_start_position():
    """The fixed concatenated PCA viewport should slide through the selected trial list."""
    trial_indices = np.arange(30, dtype=int)

    visible_trial_indices, clamped_start_position = psth_webapp.select_concatenated_trial_viewport(
        trial_indices=trial_indices,
        start_position=12,
        visible_trial_count=8,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(12, 20, dtype=int))
    assert clamped_start_position == 12


def test_select_concatenated_trial_viewport_clamps_to_last_full_window():
    """Oversized start positions should keep the viewport filled when enough trials exist."""
    trial_indices = np.arange(30, dtype=int)

    visible_trial_indices, clamped_start_position = psth_webapp.select_concatenated_trial_viewport(
        trial_indices=trial_indices,
        start_position=999,
        visible_trial_count=8,
    )

    np.testing.assert_array_equal(visible_trial_indices, np.arange(22, 30, dtype=int))
    assert clamped_start_position == 22


def test_concatenated_pca_defaults_are_compact_for_fixed_viewport():
    """The default concatenated PCA view should fit as a compact on-screen viewport."""
    assert psth_webapp.DEFAULT_CONCATENATED_PCA_VISIBLE_TRIAL_COUNT == 20
    assert psth_webapp.DEFAULT_CONCATENATED_PCA_COMPONENT_COUNT == 3
    assert psth_webapp.CONCATENATED_PCA_FIGURE_SIZE == (11.0, 5.0)


def test_webapp_has_matplotlib_cleanup_dependency():
    """The Streamlit app should keep the pyplot handle used for figure cleanup."""
    assert callable(psth_webapp.plt.close)


def test_webapp_dataframe_width_argument_avoids_deprecated_streamlit_api():
    """Metadata tables should use Streamlit's current width argument."""
    webapp_source = Path(psth_webapp.__file__).read_text()

    assert "use_container_width" not in webapp_source
    assert 'width="stretch"' in webapp_source
