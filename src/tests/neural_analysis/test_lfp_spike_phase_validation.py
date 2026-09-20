"""RED contracts for the immutable CT026 100-shuffle Spike-phase preview."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_spike_phase_validation
from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_models import (
    UnitPopulationConfig,
    component_fingerprint,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult
from src.neural_analysis.lfp_spike_phase_validation import (
    build_ct026_default_active_population,
    build_ct026_spike_phase_preview_config,
    make_production_spike_phase_preview_dependencies,
    render_cached_spike_phase_preview_validation,
    run_spike_phase_preview_validation,
)


def _population() -> UnitPopulationConfig:
    """Return a fixed active population with stable probe-qualified identities.

    Returns
    -------
    UnitPopulationConfig
        One ProbeA population with three selected channels and all three unit
        identities retained in ascending cluster order.
    """
    return UnitPopulationConfig(
        label="active",
        probe_label="ProbeA",
        sorter_path=Path("sorter"),
        aligned_spike_path=Path("aligned_spikes.npz"),
        selected_channels=(2, 7, 11),
        quality_settings=(("quality_labels", "good,mua"),),
        stable_unit_ids=("ProbeA:2", "ProbeA:7", "ProbeA:11"),
    )


def test_ct026_preview_config_uses_100_shuffles_without_changing_other_components(
    tmp_path: Path,
) -> None:
    """Preview must retain the active population and stale only Spike phase.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary CT026-like session root; no raw files are opened.
    """
    preview = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())
    final = replace(preview, ppc=replace(preview.ppc, shuffle_count=1000))

    assert preview.ppc.shuffle_count == 100
    assert preview.unit_population == _population()
    assert component_fingerprint("power", preview) == component_fingerprint("power", final)
    assert component_fingerprint("synchrony", preview) == component_fingerprint(
        "synchrony", final
    )
    assert component_fingerprint("spike_phase", preview) != component_fingerprint(
        "spike_phase", final
    )


def test_failed_preview_creates_no_immutable_report_directory(tmp_path: Path) -> None:
    """A Spike-phase failure must not write a partial report or recompute LFP.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary report parent and CT026-like session root.
    """
    calls: list[str] = []

    def fail_spike(config: object, dependencies: object) -> ComponentRunResult:
        """Record the isolated Spike-phase operation and return a pre-commit failure."""
        calls.append("compute_spike_phase")
        return ComponentRunResult("spike_phase", "failed", "prepare_spike", "failure", None)

    dependencies = SimpleNamespace(
        pipeline_dependencies=object(),
        compute_spike_phase_component=fail_spike,
        now_utc=lambda: "2026-08-26T17-00-00Z",
    )
    config = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())

    with pytest.raises(RuntimeError, match="failure"):
        run_spike_phase_preview_validation(config, tmp_path / "runs", dependencies)

    assert calls == ["compute_spike_phase"]
    assert not (tmp_path / "runs").exists()


def test_default_population_uses_qualified_probeb_good_inside_brain_units(
    tmp_path: Path,
) -> None:
    """The CT026 default must select sorted ProbeB good/mua units only.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary CT026-like root; injected loaders prevent raw-data access.
    """
    def load_clusters(path: Path) -> pd.DataFrame:
        """Return deliberately unordered cluster metadata for one ProbeB sorter."""
        assert path.name == "kilosort4"
        return pd.DataFrame(
            {"cluster_id": [9, 2, 5, 7], "ch": [12, 7, 11, 11],
             "group": ["noise", "good", "mua", "good"]}
        )

    def load_channels(path: Path) -> pd.DataFrame:
        """Return real CT026 column aliases with one outside-brain channel."""
        assert path.name == "kilosort4"
        return pd.DataFrame(
            {
                "channel_id": ["CH7", "CH11", "CH12"],
                "label": ["good", "good", "good"],
                "inside_brain": [True, True, False],
            }
        )

    population = build_ct026_default_active_population(
        tmp_path / "CT026", load_clusters, load_channels
    )

    assert population.probe_label == "ProbeB"
    assert population.selected_channels == (7, 11)
    assert population.stable_unit_ids == ("ProbeB:2", "ProbeB:5", "ProbeB:7")
    assert dict(population.quality_settings) == {
        "channel_quality": "good",
        "inside_brain": "true",
        "unit_quality": "good,mua",
    }


@pytest.mark.parametrize("probe_label", ("ProbeA", "ProbeB"))
def test_explicit_ct026_population_supports_one_probe_with_identical_defaults(
    tmp_path: Path,
    probe_label: str,
) -> None:
    """Probe choice changes paths/ids only, never quality-selection defaults."""
    clusters = pd.DataFrame(
        {
            "cluster_id": [9, 2, 5],
            "ch": [12, 7, 11],
            "group": ["noise", "good", "mua"],
        }
    )
    channels = pd.DataFrame(
        {
            "channel_id": ["CH7", "CH11", "CH12"],
            "label": ["good", "good", "good"],
            "inside_brain": [True, True, False],
        }
    )

    population = lfp_spike_phase_validation.build_ct026_active_population(
        tmp_path / "CT026",
        probe_label,
        lambda _: clusters,
        lambda _: channels,
    )

    assert population.probe_label == probe_label
    assert population.sorter_path.as_posix().endswith(f"{probe_label}/kilosort4")
    assert population.aligned_spike_path.name == f"probe{probe_label[-1]}_sync.npz"
    assert population.selected_channels == (7, 11)
    assert population.stable_unit_ids == (f"{probe_label}:2", f"{probe_label}:5")
    assert dict(population.quality_settings) == {
        "channel_quality": "good",
        "inside_brain": "true",
        "unit_quality": "good,mua",
    }

    with pytest.raises(ValueError, match="ProbeA or ProbeB"):
        lfp_spike_phase_validation.build_ct026_active_population(
            tmp_path / "CT026",
            "combined",
            lambda _: clusters,
            lambda _: channels,
        )


def _cache_arrays() -> dict[str, np.ndarray]:
    """Return a complete three-site Spike cache for bounded report tests.

    Returns
    -------
    dict[str, numpy.ndarray]
        Cache arrays with trial=4, unit=2, site=3, condition=9, epoch=3,
        band=2, and frequency=2. Spike counts are samples and PPC values are
        dimensionless; illustrative trial indices are trial-table row ids.
    """
    condition_names = (
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
    metric_shape = (2, 9, 3, 3, 2)
    cell_shape = (9, 3, 3, 2)
    trial_indices = np.array((3, 5, 7, 11), dtype=np.int64)
    offsets = np.array(
        ((0, 1, 2, 3, 4), (4, 5, 6, 7, 8)),
        dtype=np.int64,
    )
    return {
        "trial_indices": trial_indices,
        "unit_ids": np.array(("ProbeB:2", "ProbeB:9")),
        "population_ids": np.array(("CT026 ProbeB active",)),
        "site_ids": np.array(("PFC", "HPC1", "HPC2")),
        "site_voltage_units": np.array(("uV", "uV", "uV")),
        "condition_names": np.array(condition_names),
        "condition_membership": np.ones((4, 9), dtype=bool),
        "filter_membership": np.ones(4, dtype=bool),
        "epoch_names": np.array(("whole", "before", "after")),
        "band_names": np.array(("theta", "gamma")),
        "frequency_hz": np.array((8.0, 40.0)),
        "relative_time_s": np.array((-0.1, 0.0, 0.1)),
        "phase_bin_edges_rad": np.linspace(-np.pi, np.pi, 5),
        "ppc": np.broadcast_to(
            np.array((0.2, 0.4)),
            metric_shape,
        ).copy(),
        "preferred_phase_rad": np.full(metric_shape, 0.25),
        "computable": np.ones(metric_shape, dtype=bool),
        "reliable": np.ones(metric_shape, dtype=bool),
        "spike_count": np.full(metric_shape, 51, dtype=np.int64),
        "null_eligible": np.ones(metric_shape, dtype=bool),
        "significant": np.zeros(metric_shape, dtype=bool),
        "ppc_band_mean": np.stack(
            (
                np.full((9, 3, 3, 2), 0.2),
                np.full((9, 3, 3, 2), 0.8),
            )
        ),
        "representative_phase_hist_count": np.ones(
            (2, 9, 3, 3, 2, 4),
            dtype=np.int64,
        ),
        "relative_spike_times_s": np.array(
            (-0.05, -0.04, -0.03, -0.02, 0.02, 0.03, 0.04, 0.05)
        ),
        "relative_spike_time_offsets": offsets,
        "source_trace": np.ones((3, 4, 3)),
        "band_filtered_trace": np.ones((3, 4, 2, 3)),
        "hilbert_phase_rad": np.zeros((3, 4, 2, 3)),
        "selected_low_unit_ids": np.full(cell_shape, "ProbeB:2", dtype="<U64"),
        "selected_high_unit_ids": np.full(cell_shape, "ProbeB:9", dtype="<U64"),
        "illustrative_low_trial_indices": np.full(cell_shape, 3, dtype=np.int64),
        "illustrative_high_trial_indices": np.full(cell_shape, 11, dtype=np.int64),
        "illustrative_trial_indices": np.full(cell_shape, 11, dtype=np.int64),
    }


def _successful_dependencies(calls: list[str]) -> SimpleNamespace:
    """Return Spike-only cache/report seams whose plots receive cached arrays.

    Parameters
    ----------
    calls : list[str]
        Ordered side-effect labels used to prove compute and plotting isolation.
    """
    def compute(*_: object) -> ComponentRunResult:
        """Record Spike-only computation and return an atomic completed result."""
        calls.append("compute_spike_phase")
        return ComponentRunResult("spike_phase", "complete", "complete_spike_phase", None, {})

    def plot(kind: str):
        """Build one cache-only public PPC plot seam for the requested kind."""
        def render(*_: object, **__: object) -> tuple[object, dict[str, object]]:
            """Record plot use and return an opaque object accepted by save_png."""
            calls.append(kind)
            return object(), {}
        return render

    def save_png(_: object, path: Path) -> None:
        """Create a PNG marker only after a cache-only plot seam is called."""
        path.write_bytes(b"\x89PNG\r\n\x1a\nreport-test")

    return SimpleNamespace(
        pipeline_dependencies=object(),
        compute_spike_phase_component=compute,
        load_manifest=lambda *_: {"components": {"spike_phase": {}}},
        assess_component_status=lambda *_: ComponentStatus("compatible", ()),
        load_spike_phase_arrays=lambda *_: _cache_arrays(),
        now_utc=lambda: "2026-08-26T18-00-00Z",
        monotonic_seconds=iter((10.0, 13.5, 14.0, 14.75)).__next__,
        peak_memory_bytes=lambda: 4096,
        plot_unit_ppc_map=plot("unit_map"),
        plot_population_ppc_maps=plot("population_map"),
        plot_ppc_band_summary=plot("band_summary"),
        plot_ppc_exemplar=plot("exemplar"),
        plot_ppc_exemplar_pair=plot("exemplar_pair"),
        save_png=save_png,
        close_figure=lambda _: None,
        source_identifiers=lambda _: {"source": "fake"},
    )


def test_successful_preview_and_cached_render_write_public_ppc_report(tmp_path: Path) -> None:
    """Successful and render-only paths must use cache-only public PPC plots.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary CT026-like root, cache path, and immutable report parent.
    """
    config = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())
    calls: list[str] = []
    result = run_spike_phase_preview_validation(
        config, tmp_path / "runs", _successful_dependencies(calls)
    )

    assert calls[0] == "compute_spike_phase"
    assert {"unit_map", "population_map", "band_summary", "exemplar_pair"}.issubset(
        calls
    )
    assert all(path.is_file() for path in result.png_paths)
    assert result.report["wall_time_s"] == 3.5
    assert result.report["peak_memory_bytes"] == 4096
    assert result.report["trial_count"] == 4
    assert result.report["unit_count"] == 2
    assert result.report["spike_count"] == 8
    assert result.report["reliable_cell_count"] == 324
    assert result.report["null_eligible_cell_count"] == 324

    cached_calls: list[str] = []
    cached = render_cached_spike_phase_preview_validation(
        config, tmp_path / "cached", _successful_dependencies(cached_calls),
        spike_phase_wall_time_s=3.5, spike_phase_peak_memory_bytes=4096,
    )

    assert "compute_spike_phase" not in cached_calls
    assert cached.run_directory.is_dir()


def test_detailed_report_publishes_bounded_plan_and_then_exposes_cleanup(
    tmp_path: Path,
) -> None:
    """The 27-figure report must validate before launcher-owned cleanup is exposed."""
    config = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())
    config = replace(
        config,
        ppc_execution=replace(config.ppc_execution, worker_count=8),
    )
    calls: list[str] = []
    cleanup_calls: list[str] = []
    dependencies = _successful_dependencies(calls)

    def compute(*_: object) -> ComponentRunResult:
        """Return a committed component plus its exact still-deferred cleanup."""
        calls.append("compute_spike_phase")
        return ComponentRunResult(
            "spike_phase",
            "complete",
            "complete_spike_phase",
            None,
            {},
            deferred_cleanup=lambda: cleanup_calls.append("cleanup"),
        )

    dependencies.compute_spike_phase_component = compute
    benchmark = lfp_spike_phase_validation.SpikePhaseFilterBenchmark(
        source_label="accepted representative filter",
        wall_time_s=12.0,
        final_cache_size_bytes=100,
        intermediate_cache_size_bytes=None,
    )
    measurements = lfp_spike_phase_validation.SpikePhaseReportMeasurements(
        phase_preparation_seconds=None,
        ppc_planning_seconds=0.0,
        grouped_execution_seconds=2.0,
        component_total_seconds=None,
        requested_worker_count=8,
        planned_worker_count=8,
        active_worker_count=4,
        prepared_phase_cache_state="warm",
        peak_process_rss_bytes=1024,
        peak_process_tree_rss_bytes=0,
        peak_process_tree_pss_bytes=None,
        memory_provenance="measured process tree",
        final_component_size_bytes=100,
        final_cache_size_bytes=200,
        intermediate_work_size_bytes=None,
        warnings=("synthetic warning",),
        exclusions=("synthetic exclusion",),
        benchmark=benchmark,
    )

    result = run_spike_phase_preview_validation(
        config,
        tmp_path / "runs",
        dependencies,
        report_measurements=measurements,
    )

    assert calls.count("unit_map") == 6
    assert calls.count("population_map") == 6
    assert calls.count("band_summary") == 3
    assert calls.count("exemplar_pair") == 12
    assert len(result.png_paths) == 27
    assert cleanup_calls == []
    assert callable(result.deferred_cleanup)
    assert result.run_directory.is_dir()
    report = json.loads(result.report_path.read_text(encoding="ascii"))
    assert report["schema_version"] == "spike_phase_preview_report.v1"
    assert report["workers"] == {"requested": 8, "planned": 8, "active": 4}
    assert report["timing_seconds"]["phase_preparation"] is None
    assert report["timing_seconds"]["ppc_planning"] == 0.0
    assert report["memory_bytes"]["process_tree_rss"] == 0
    assert report["memory_bytes"]["process_tree_pss"] is None
    assert report["sizes_bytes"]["intermediate_work"] is None
    assert report["prepared_phase_cache_state"] == "warm"
    assert report["warnings"] == ["synthetic warning"]
    assert report["exclusions"] == ["synthetic exclusion"]
    projection = report["nine_filter_projection"]
    assert projection["benchmark_source"] == "accepted representative filter"
    assert projection["projected_wall_time_s"] == 108.0
    assert projection["projected_final_cache_size_bytes"] == 900
    assert projection["projected_intermediate_cache_size_bytes"] is None
    assert result.log_path.is_file()
    assert result.source_identifiers_path.is_file()
    assert "unavailable" in result.summary_path.read_text(encoding="ascii")
    result.deferred_cleanup()
    assert cleanup_calls == ["cleanup"]


def test_general_cached_report_supports_separate_1000_shuffle_final_run(
    tmp_path: Path,
) -> None:
    """The launcher report uses actual shuffles without changing preview schema."""
    config = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())
    config = replace(config, ppc=replace(config.ppc, shuffle_count=1000))
    calls: list[str] = []

    result = lfp_spike_phase_validation.render_cached_spike_phase_report(
        config,
        tmp_path / "runs",
        _successful_dependencies(calls),
        run_kind="final",
        component_wall_time_s=12.0,
        component_peak_memory_bytes=4096,
    )

    report = json.loads(result.report_path.read_text(encoding="ascii"))
    assert report["schema_version"] == "spike_phase_report.v1"
    assert report["run_kind"] == "final"
    assert report["shuffle_count"] == 1000
    assert "compute_spike_phase" not in calls


def test_real_dimension_population_maps_publish_inside_cache_backed_report(
    tmp_path: Path,
) -> None:
    """A 273-unit, nine-condition, 50-frequency report must render atomically."""
    config = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())
    arrays = _cache_arrays()
    unit_count = 273
    frequency_count = 50
    metric_shape = (unit_count, 9, 3, 3, frequency_count)
    arrays.update(
        {
            "unit_ids": np.array(
                tuple(f"ProbeB:{cluster_id}" for cluster_id in range(unit_count))
            ),
            "frequency_hz": np.arange(2.0, 102.0, 2.0),
            "ppc": np.full(metric_shape, 0.1),
            "computable": np.ones(metric_shape, dtype=bool),
            "reliable": np.ones(metric_shape, dtype=bool),
            "spike_count": np.full(metric_shape, 51, dtype=np.int64),
            "null_eligible": np.ones(metric_shape, dtype=bool),
            "significant": np.zeros(metric_shape, dtype=bool),
        }
    )
    calls: list[str] = []
    dependencies = _successful_dependencies(calls)
    dependencies.load_spike_phase_arrays = lambda *_: arrays
    dependencies.plot_population_ppc_maps = (
        lfp_spike_phase_validation._plot_cached_population_map
    )

    def save_png(figure: object, path: Path) -> None:
        """Save real population figures and marker files for injected stubs."""
        if hasattr(figure, "savefig"):
            figure.savefig(path)
        else:
            path.write_bytes(b"\x89PNG\r\n\x1a\nreport-test")

    dependencies.save_png = save_png
    dependencies.close_figure = lambda figure: (
        plt.close(figure) if hasattr(figure, "savefig") else None
    )

    result = render_cached_spike_phase_preview_validation(
        config,
        tmp_path / "runs",
        dependencies,
        spike_phase_wall_time_s=12.0,
        spike_phase_peak_memory_bytes=4096,
    )

    assert result.report["unit_count"] == 273
    assert calls.count("population_map") == 0
    assert len(result.png_paths) == 27
    assert all(path.is_file() for path in result.png_paths)


@pytest.mark.parametrize(
    "failure_seam",
    (
        "_render_cached_pngs",
        "_write_report_artifacts",
        "_validate_staged_report",
        "_publish_staged_report",
    ),
)
def test_report_failures_publish_nothing_and_never_expose_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_seam: str,
) -> None:
    """Serialization, validation, and publication failures retain PPC work."""
    config = build_ct026_spike_phase_preview_config(tmp_path / "CT026", _population())
    calls: list[str] = []
    cleanup_calls: list[str] = []
    dependencies = _successful_dependencies(calls)

    def compute(*_: object) -> ComponentRunResult:
        """Return a cleanup-capable completed component for failure injection."""
        return ComponentRunResult(
            "spike_phase",
            "complete",
            "complete_spike_phase",
            None,
            {},
            deferred_cleanup=lambda: cleanup_calls.append("cleanup"),
        )

    def fail(*_: object, **__: object) -> object:
        """Raise at one report-transaction seam before cleanup can escape."""
        raise OSError(f"injected {failure_seam} failure")

    dependencies.compute_spike_phase_component = compute
    monkeypatch.setattr(lfp_spike_phase_validation, failure_seam, fail)

    with pytest.raises(OSError, match=failure_seam):
        run_spike_phase_preview_validation(
            config,
            tmp_path / "runs",
            dependencies,
        )

    final = (
        tmp_path
        / "runs"
        / (
            f"{config.session_id}_lfp_spike_phase_preview_validation_"
            "2026-08-26T18-00-00Z"
        )
    )
    assert not final.exists()
    assert cleanup_calls == []


def test_production_preview_dependencies_bind_real_spike_cache_boundary() -> None:
    """The production factory must expose Spike compute, cache, plot, and clocks."""
    dependencies = make_production_spike_phase_preview_dependencies()

    assert callable(dependencies.compute_spike_phase_component)
    assert callable(dependencies.load_spike_phase_arrays)
    assert callable(dependencies.plot_unit_ppc_map)
    assert callable(dependencies.plot_population_ppc_maps)
    assert callable(dependencies.plot_ppc_band_summary)
    assert callable(dependencies.plot_ppc_exemplar)
    assert callable(dependencies.plot_ppc_exemplar_pair)
    assert callable(dependencies.monotonic_seconds)
    assert callable(dependencies.peak_memory_bytes)
