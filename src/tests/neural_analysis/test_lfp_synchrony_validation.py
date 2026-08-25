"""RED contracts for immutable cache-backed CT026 Synchrony validation reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult
from src.neural_analysis.lfp_synchrony_validation import (
    SynchronyValidationDependencies,
    build_ct026_synchrony_config,
    render_cached_synchrony_validation,
    run_synchrony_validation,
)


@dataclass
class _FakeFigure:
    """In-memory figure stand-in with no data values or physical units."""

    closed: bool = False


def _synchrony_arrays() -> dict[str, np.ndarray]:
    """Return small complete Synchrony cache arrays with documented axes.

    Returns
    -------
    dict[str, numpy.ndarray]
        Cache-shaped arrays: trial=2, site=2, pair=1, condition=9, frequency=2,
        epoch=3, band=2, and time=4. Source traces are uV and phases radians.
    """
    condition_count = 9
    return {
        "trial_indices": np.array((0, 1), dtype=np.int64),
        "site_ids": np.array(("PFC", "HPC1")),
        "site_voltage_units": np.array(("uV", "uV")),
        "condition_names": np.array((
            "correct_rewarded", "omission", "incorrect", "switch", "stay",
            "omission_switch", "omission_stay", "incorrect_switch",
            "incorrect_stay",
        )),
        "condition_membership": np.ones((2, condition_count), dtype=bool),
        "filter_membership": np.array((True, True)),
        "frequency_hz": np.array((8.0, 40.0)),
        "epoch_names": np.array(("whole", "before", "after")),
        "band_names": np.array(("theta", "gamma")),
        "relative_time_s": np.array((-2.0, -1.0, 0.0, 1.0)),
        "site_valid": np.ones((2, 2), dtype=bool),
        "pair_valid": np.ones((1, 2), dtype=bool),
        "site_exclusion_count": np.array((0, 0), dtype=np.int64),
        "pair_exclusion_count": np.array((0,), dtype=np.int64),
        "pair_site_a_ids": np.array(("PFC",)),
        "pair_site_b_ids": np.array(("HPC1",)),
        "itpc": np.ones((condition_count, 2, 2, 4)),
        "itpc_effective_trial_count": np.full((condition_count, 2, 2, 4), 2),
        "ispc": np.ones((condition_count, 1, 2, 4)),
        "ispc_phase_offset_rad": np.zeros((condition_count, 1, 2, 4)),
        "ispc_effective_trial_count": np.full((condition_count, 1, 2, 4), 2),
        "itpc_band_mean": np.ones((condition_count, 2, 3, 2)),
        "itpc_ci_low": np.full((condition_count, 2, 3, 2), 0.8),
        "itpc_ci_high": np.ones((condition_count, 2, 3, 2)),
        "itpc_unstable": np.ones((condition_count, 2, 3, 2), dtype=bool),
        "ispc_band_mean": np.ones((condition_count, 1, 3, 2)),
        "ispc_ci_low": np.full((condition_count, 1, 3, 2), 0.8),
        "ispc_ci_high": np.ones((condition_count, 1, 3, 2)),
        "ispc_unstable": np.ones((condition_count, 1, 3, 2), dtype=bool),
        "plv_by_frequency": np.ones((2, 1, 3, 2)),
        "plv_phase_offset_rad": np.zeros((2, 1, 3, 2)),
        "plv_valid_sample_count": np.full((2, 1, 3, 2), 1000),
        "plv_valid_sample_fraction": np.ones((2, 1, 3, 2)),
        "plv_computable": np.ones((2, 1, 3, 2), dtype=bool),
        "plv_band_mean": np.ones((2, 1, 3, 2)),
        "source_trace": np.zeros((2, 2, 4)),
        "band_filtered_trace": np.zeros((2, 2, 2, 4)),
        "hilbert_phase_rad": np.zeros((2, 2, 2, 4)),
    }


def _dependencies(calls: list[str]) -> SynchronyValidationDependencies:
    """Build fake Synchrony-only compute, cache, plotting, and report seams.

    Parameters
    ----------
    calls : list[str]
        Chronological operation labels without numerical values or units.

    Returns
    -------
    SynchronyValidationDependencies
        Dependencies that render cache-only fake figures and never invoke Power
        or Spike phase operations.
    """
    arrays = _synchrony_arrays()

    def compute_synchrony(config: object, dependencies: object) -> ComponentRunResult:
        """Record the Synchrony pipeline action and return one completed result."""
        calls.append("compute_synchrony")
        return ComponentRunResult("synchrony", "complete", "complete_synchrony", None, {})

    def load_manifest(directory: Path, config: object) -> dict[str, object]:
        """Return a complete Synchrony manifest without accessing raw recordings."""
        calls.append("load_manifest")
        return {"components": {"synchrony": {"file_name": "synchrony.npz"}}}

    def assess(*_: object) -> ComponentStatus:
        """Report compatible status for the synthetic Synchrony component."""
        return ComponentStatus("compatible", ())

    def load_arrays(*_: object) -> dict[str, np.ndarray]:
        """Return cache arrays, never a wavelet tensor or raw-file handle."""
        calls.append("load_synchrony")
        return arrays

    def plot_map(*_: object, **__: object) -> tuple[_FakeFigure, dict[str, object]]:
        """Return an unsaved fake map figure with no numerical conversion."""
        calls.append("map")
        return _FakeFigure(), {}

    def plot_summary(*_: object, **__: object) -> tuple[_FakeFigure, dict[str, object]]:
        """Return an unsaved fake summary figure with count display inputs."""
        calls.append("summary")
        return _FakeFigure(), {}

    def plot_distribution(*_: object, **__: object) -> tuple[_FakeFigure, dict[str, object]]:
        """Return an unsaved fake PLV distribution with coverage inputs."""
        calls.append("distribution")
        return _FakeFigure(), {}

    def plot_exemplar(*_: object, **__: object) -> tuple[_FakeFigure, dict[str, object]]:
        """Return an unsaved fake cached-trace PLV exemplar figure."""
        calls.append("exemplar")
        return _FakeFigure(), {}

    def save_png(figure: _FakeFigure, path: Path) -> None:
        """Create an empty PNG marker at the report path and record the action."""
        calls.append("save")
        path.write_bytes(b"png")

    def close_figure(figure: _FakeFigure) -> None:
        """Mark one fake figure closed after its report artifact is saved."""
        figure.closed = True

    return SynchronyValidationDependencies(
        pipeline_dependencies=object(),
        compute_synchrony_component=compute_synchrony,
        load_manifest=load_manifest,
        assess_component_status=assess,
        load_synchrony_arrays=load_arrays,
        plot_phase_map=plot_map,
        plot_phase_band_summary=plot_summary,
        plot_plv_distribution=plot_distribution,
        plot_plv_exemplar=plot_exemplar,
        save_png=save_png,
        close_figure=close_figure,
        now_utc=lambda: "2026-08-25T17-00-00Z",
        monotonic_seconds=lambda: 3.0,
        peak_memory_bytes=lambda: 4096,
        source_identifiers=lambda _: {"runtime": "synthetic"},
    )


def test_synchrony_validation_runs_only_synchrony_and_writes_cache_backed_report(
    tmp_path: Path,
) -> None:
    """Validation must compute Synchrony once then render maps, counts, and exemplars.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary report parent and disposable component-cache path.
    """
    config = build_ct026_synchrony_config(tmp_path / "CT026")
    calls: list[str] = []

    result = run_synchrony_validation(config, tmp_path / "runs", _dependencies(calls))

    assert calls[:3] == ["compute_synchrony", "load_manifest", "load_synchrony"]
    assert "compute_power" not in calls
    assert "compute_spike_phase" not in calls
    assert result.component == "synchrony"
    assert result.run_directory.is_dir()
    assert result.summary_path.is_file()
    assert result.log_path.is_file()
    assert result.manifest_snapshot_path.is_file()
    assert result.configuration_snapshot_path.is_file()
    names = {path.name for path in result.png_paths}
    assert "PFC_correct_rewarded_itpc_map.png" in names
    assert "PFC_HPC1_correct_rewarded_ispc_map.png" in names
    assert "PFC_theta_before_itpc_band_summary.png" in names
    assert "PFC_HPC1_theta_before_ispc_band_summary.png" in names
    assert "PFC_HPC1_correct_rewarded_theta_plv_distribution.png" in names
    assert "PFC_HPC1_correct_rewarded_theta_low_plv_exemplar.png" in names
    assert "PFC_HPC1_correct_rewarded_theta_high_plv_exemplar.png" in names
    assert {"map", "summary", "distribution", "exemplar"}.issubset(calls)


def test_synchrony_validation_aborts_before_report_when_component_fails(tmp_path: Path) -> None:
    """A failed Synchrony result must preserve the prior cache and create no report.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary report parent and generic cache path that remain unmodified.
    """
    config = build_ct026_synchrony_config(tmp_path / "CT026")
    calls: list[str] = []
    dependencies = _dependencies(calls)

    def failed_compute(config: object, dependencies: object) -> ComponentRunResult:
        """Return a failed Synchrony result without writing component files."""
        calls.append("compute_synchrony")
        return ComponentRunResult("synchrony", "failed", "payload_synchrony", "failure", None)

    dependencies = SynchronyValidationDependencies(
        **{**dependencies.__dict__, "compute_synchrony_component": failed_compute}
    )
    with pytest.raises(RuntimeError, match="failure"):
        run_synchrony_validation(config, tmp_path / "runs", dependencies)

    assert calls == ["compute_synchrony"]
    assert not (tmp_path / "runs").exists()


def test_synchrony_validation_does_not_publish_partial_report_on_plot_failure(
    tmp_path: Path,
) -> None:
    """A failed cache-only render must leave no visible immutable run directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary generic cache and report roots.
    """
    config = build_ct026_synchrony_config(tmp_path / "CT026")
    calls: list[str] = []
    dependencies = _dependencies(calls)

    def failed_save(figure: _FakeFigure, path: Path) -> None:
        """Raise before publishing the first cache-backed PNG."""
        del figure, path
        raise OSError("render failed")

    dependencies = SynchronyValidationDependencies(
        **{**dependencies.__dict__, "save_png": failed_save}
    )
    run_parent = tmp_path / "runs"
    with pytest.raises(OSError, match="render failed"):
        run_synchrony_validation(config, run_parent, dependencies)

    assert not run_parent.exists() or not tuple(run_parent.iterdir())


def test_cached_synchrony_validation_renders_without_recomputing(
    tmp_path: Path,
) -> None:
    """A compatible cache may be rerendered with preserved performance metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary generic cache and immutable report roots.
    """
    config = build_ct026_synchrony_config(tmp_path / "CT026")
    calls: list[str] = []

    result = render_cached_synchrony_validation(
        config,
        tmp_path / "runs",
        _dependencies(calls),
        synchrony_wall_time_s=600.5,
        synchrony_peak_memory_bytes=8192,
    )

    assert calls[:2] == ["load_manifest", "load_synchrony"]
    assert "compute_synchrony" not in calls
    assert result.wall_time_s == pytest.approx(600.5)
    assert result.peak_memory_bytes == 8192
    assert result.report["peak_memory_available"] is True
    assert result.run_directory.is_dir()


def test_cached_synchrony_validation_marks_zero_peak_memory_unavailable(
    tmp_path: Path,
) -> None:
    """A zero memory sentinel must not be reported as a measured zero-byte peak."""
    config = build_ct026_synchrony_config(tmp_path / "CT026")

    result = render_cached_synchrony_validation(
        config,
        tmp_path / "runs",
        _dependencies([]),
        synchrony_wall_time_s=600.5,
        synchrony_peak_memory_bytes=0,
    )

    assert result.report["peak_memory_available"] is False
    assert any("peak memory unavailable" in warning for warning in result.report["warnings"])
    assert "Peak memory: unavailable" in result.summary_path.read_text(encoding="ascii")
