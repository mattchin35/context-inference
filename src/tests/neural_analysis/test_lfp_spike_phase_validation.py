"""RED contracts for the immutable CT026 100-shuffle Spike-phase preview."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_models import (
    UnitPopulationConfig,
    component_fingerprint,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult
from src.neural_analysis.lfp_spike_phase_validation import (
    build_ct026_default_active_population,
    build_ct026_spike_phase_preview_config,
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
        """Return channel labels with one outside-brain and one bad channel."""
        assert path.name == "kilosort4"
        return pd.DataFrame(
            {"channel": [7, 11, 12], "channel_quality": ["good", "good", "good"],
             "inside_brain": [True, True, False]}
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


def _cache_arrays() -> dict[str, np.ndarray]:
    """Return a minimal cache-shaped Spike-phase result for report-only tests.

    Returns
    -------
    dict[str, numpy.ndarray]
        Cache arrays with trial=2, unit=1, site=1, condition=1, epoch=1, and
        frequency=2. Spike counts are samples and PPC values are dimensionless.
    """
    shape = (1, 1, 1, 1, 2)
    return {
        "trial_indices": np.array((3, 9), dtype=np.int64),
        "unit_ids": np.array(("ProbeB:2",)),
        "site_ids": np.array(("PFC",)),
        "condition_names": np.array(("correct_rewarded",)),
        "epoch_names": np.array(("whole",)),
        "band_names": np.array(("theta",)),
        "frequency_hz": np.array((8.0, 40.0)),
        "ppc": np.full(shape, 0.2),
        "computable": np.ones(shape, dtype=bool),
        "reliable": np.array([[[[[True, False]]]]]),
        "spike_count": np.array([[[[[51, 40]]]]]),
        "null_eligible": np.array([[[[[True, False]]]]]),
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
        path.write_bytes(b"png")

    return SimpleNamespace(
        pipeline_dependencies=object(),
        compute_spike_phase_component=compute,
        load_manifest=lambda *_: {"components": {"spike_phase": {}}},
        assess_component_status=lambda *_: ComponentStatus("compatible", ()),
        load_spike_phase_arrays=lambda *_: _cache_arrays(),
        now_utc=lambda: "2026-08-26T18-00-00Z",
        monotonic_seconds=iter((10.0, 13.5)).__next__,
        peak_memory_bytes=lambda: 4096,
        plot_unit_ppc_map=plot("unit_map"),
        plot_population_ppc_maps=plot("population_map"),
        plot_ppc_band_summary=plot("band_summary"),
        plot_ppc_exemplar=plot("exemplar"),
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
    assert {"unit_map", "population_map", "band_summary", "exemplar"}.issubset(calls)
    assert all(path.is_file() for path in result.png_paths)
    assert result.report["wall_time_s"] == 3.5
    assert result.report["peak_memory_bytes"] == 4096
    assert result.report["trial_count"] == 2
    assert result.report["unit_count"] == 1
    assert result.report["spike_count"] == 91
    assert result.report["reliable_cell_count"] == 1
    assert result.report["null_eligible_cell_count"] == 1

    cached_calls: list[str] = []
    cached = render_cached_spike_phase_preview_validation(
        config, tmp_path / "cached", _successful_dependencies(cached_calls),
        spike_phase_wall_time_s=3.5, spike_phase_peak_memory_bytes=4096,
    )

    assert "compute_spike_phase" not in cached_calls
    assert cached.run_directory.is_dir()
