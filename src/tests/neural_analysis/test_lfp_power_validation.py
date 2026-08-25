"""RED contracts for the production-facing CT026 Power validation runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.neural_analysis.lfp_power_validation import (
    PowerValidationDependencies,
    build_ct026_power_config,
    render_cached_power_validation,
    run_power_validation,
)
from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult


_SESSION_ID = "CT026_2026-08-01_130853"


@dataclass
class _FakeFigure:
    """In-memory plotting stand-in that records PNG paths and close calls.

    Attributes
    ----------
    saved_paths : list[pathlib.Path]
        PNG output paths passed to ``savefig``. Paths are filesystem locations;
        figures contain no experimental units.
    closed : bool
        True after the injected close seam is called.
    """

    saved_paths: list[Path]
    closed: bool = False


def _power_arrays() -> dict[str, np.ndarray]:
    """Return minimal cache-backed Power arrays for three sites and nine conditions.

    Returns
    -------
    dict[str, numpy.ndarray]
        Numeric/Unicode arrays with Power cache axes: site=3, trial=2, epoch=3,
        frequency=4, band=2, condition=9, and time=2. PSD values are dB; source
        trace values are uV; identity arrays are fixed-width Unicode.
    """
    return {
        "site_ids": np.array(("PFC", "HPC1", "HPC2")),
        "condition_names": np.array(
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
        ),
        "epoch_names": np.array(("whole", "before", "after")),
        "band_names": np.array(("theta", "gamma")),
        "frequency_hz": np.array((6.0, 8.0, 40.0, 120.0)),
        "normalized_psd_session_db": np.zeros((3, 2, 3, 4), dtype=float),
        "band_power_session_db": np.zeros((3, 2, 3, 2), dtype=float),
        "condition_membership": np.array(
            (
                (True, False, False, True, False, False, False, False, False),
                (False, True, False, False, True, False, True, False, False),
            )
        ),
        "filter_membership": np.array((True, False)),
        "condition_effective_trial_count": np.array(
            ((1, 1, 1),) + ((0, 0, 0),) * 8
        ),
        "condition_unstable": np.ones((9, 3), dtype=bool),
        "objective_valid": np.ones((3, 2), dtype=bool),
        "site_valid": np.ones((3, 2), dtype=bool),
        "user_excluded": np.array((False, False)),
        "exclusion_reason_code": np.full((3, 2), "", dtype="<U1"),
        "presession_reference_available": np.array((True, False, True)),
    }


def _completed_power_result() -> ComponentRunResult:
    """Return a successful Power-only pipeline outcome without array payloads.

    Returns
    -------
    ComponentRunResult
        Completed ``power`` result with no error or numerical data axes. Cache
        arrays are supplied independently by the injected loader.
    """
    return ComponentRunResult("power", "complete", "complete_power", None, {})


def _failed_power_result() -> ComponentRunResult:
    """Return a failed Power-only pipeline outcome for abort-path validation.

    Returns
    -------
    ComponentRunResult
        Failed ``power`` result with a concise error and no cache-manifest axes.
    """
    return ComponentRunResult(
        "power", "failed", "payload_power", "synthetic failure", None
    )


def _validation_dependencies(
    calls: list[str],
    *,
    power_result: ComponentRunResult,
) -> PowerValidationDependencies:
    """Build explicit fake runtime, cache, plotting, clock, and memory seams.

    Parameters
    ----------
    calls : list[str]
        Mutable chronological event log. Entries are categorical labels with no
        physical units.
    power_result : ComponentRunResult
        Injected result from the Power-only pipeline call; it contains no raw
        traces and no cache arrays.

    Returns
    -------
    PowerValidationDependencies
        Dependency bundle whose loaders return cache-backed axes from
        ``_power_arrays``; plotting functions return fake figures and never
        open data files. Clock values are seconds and peak memory is bytes.
    """
    arrays = _power_arrays()

    def compute_power(config: object, dependencies: object) -> ComponentRunResult:
        """Record the Power-only pipeline call and return its injected outcome.

        Parameters
        ----------
        config : object
            Production configuration; this fake does not inspect its units.
        dependencies : object
            Numerical pipeline dependencies passed through unchanged.

        Returns
        -------
        ComponentRunResult
            Preset completed or failed Power-only result without raw data arrays.
        """
        calls.append("compute_power")
        return power_result

    def load_manifest(cache_directory: Path, config: object) -> dict[str, object]:
        """Return a compatible Power manifest without filesystem reads.

        Parameters
        ----------
        cache_directory : pathlib.Path
            Disposable generic cache location; it has filesystem-path units.
        config : object
            Active configuration; no numerical axes are consumed by this fake.

        Returns
        -------
        dict[str, object]
            Manifest mapping with a complete Power component and no raw arrays.
        """
        calls.append("load_manifest")
        return {"components": {"power": {"file_name": "power.npz"}}}

    def assess_cache(
        cache_directory: Path,
        component: str,
        config: object,
        manifest: dict[str, object],
    ) -> ComponentStatus:
        """Return compatible status for the disposable Power cache.

        Parameters
        ----------
        cache_directory : pathlib.Path
            Generic cache path with filesystem-coordinate units.
        component : str
            Component identity; Power is the only accepted runner target.
        config : object
            Active configuration with unchanged numerical contracts.
        manifest : dict[str, object]
            Cache metadata without numerical array axes.

        Returns
        -------
        ComponentStatus
            Categorical ``"compatible"`` cache state with no differences.
        """
        assert component == "power"
        calls.append("assess_power")
        return ComponentStatus("compatible")

    def load_power(
        cache_directory: Path, manifest: dict[str, object]
    ) -> dict[str, np.ndarray]:
        """Return validated Power arrays from the fake disposable cache.

        Parameters
        ----------
        cache_directory : pathlib.Path
            Generic cache directory with filesystem-coordinate units.
        manifest : dict[str, object]
            Power metadata with no numerical array axes.

        Returns
        -------
        dict[str, numpy.ndarray]
            Cached Power arrays described by ``_power_arrays`` with Hz, dB, uV,
            and categorical axes preserved.
        """
        calls.append("load_power")
        return arrays

    def plot_psd(**kwargs: Any) -> tuple[_FakeFigure, dict[str, object]]:
        """Return a fake cached-PSD figure after recording cache-only plotting.

        Parameters
        ----------
        **kwargs : Any
            Cached-array plotting arguments. No raw LFP loader is accepted.

        Returns
        -------
        tuple[_FakeFigure, dict[str, object]]
            Unsaved fake figure and empty categorical axes mapping.
        """
        values = kwargs["condition_trial_psd_db"]
        assert isinstance(values, np.ndarray)
        assert np.max(kwargs["frequency_hz"]) <= 100.0
        assert np.isnan(values[:, 1]).all()
        assert kwargs["condition_names"] in (
            ("correct_rewarded", "omission", "incorrect", "switch", "stay"),
            (
                "omission_switch",
                "omission_stay",
                "incorrect_switch",
                "incorrect_stay",
            ),
        )
        calls.append("plot_psd")
        return _FakeFigure([]), {}

    def plot_band(**kwargs: Any) -> tuple[_FakeFigure, dict[str, object]]:
        """Return a fake cached-band figure after recording cache-only plotting.

        Parameters
        ----------
        **kwargs : Any
            Cached-array plotting arguments. No raw LFP loader is accepted.

        Returns
        -------
        tuple[_FakeFigure, dict[str, object]]
            Unsaved fake figure and empty categorical axes mapping.
        """
        values = kwargs["condition_trial_band_power_db"]
        assert isinstance(values, np.ndarray)
        assert np.isnan(values[:, 1]).all()
        assert kwargs["condition_names"] in (
            ("correct_rewarded", "omission", "incorrect", "switch", "stay"),
            (
                "omission_switch",
                "omission_stay",
                "incorrect_switch",
                "incorrect_stay",
            ),
        )
        calls.append("plot_band")
        return _FakeFigure([]), {}

    def save_png(figure: _FakeFigure, path: Path) -> None:
        """Record one report PNG path without writing an image.

        Parameters
        ----------
        figure : _FakeFigure
            Unsaved fake plot with no numerical data axes.
        path : pathlib.Path
            Timestamped PNG report path with filesystem-coordinate units.

        Returns
        -------
        None
            The path is retained on the figure and in the categorical call log.
        """
        figure.saved_paths.append(path)
        calls.append("save_png")

    def close_figure(figure: _FakeFigure) -> None:
        """Mark one fake figure closed after its PNG save.

        Parameters
        ----------
        figure : _FakeFigure
            Fake plot with no numerical data axes.

        Returns
        -------
        None
            The figure close state changes to true without filesystem effects.
        """
        figure.closed = True
        calls.append("close_figure")

    elapsed_seconds = iter((0.0, 12.5))

    def now_utc() -> str:
        """Return one deterministic filesystem-safe UTC timestamp."""
        return "2026-08-25T01-02-03Z"

    def monotonic_seconds() -> float:
        """Return start then stop monotonic seconds for a 12.5-second run."""
        return next(elapsed_seconds)

    def peak_memory_bytes() -> int:
        """Return deterministic process peak memory in bytes."""
        return 4096

    def source_identifiers(config: object) -> dict[str, str]:
        """Return a deterministic source identifier without inspecting Git."""
        del config
        return {"pipeline": "synthetic-pipeline-id"}

    return PowerValidationDependencies(
        pipeline_dependencies=object(),
        compute_power_component=compute_power,
        load_manifest=load_manifest,
        assess_component_status=assess_cache,
        load_power_arrays=load_power,
        plot_condition_psd=plot_psd,
        plot_band_power_summary=plot_band,
        save_png=save_png,
        close_figure=close_figure,
        now_utc=now_utc,
        monotonic_seconds=monotonic_seconds,
        peak_memory_bytes=peak_memory_bytes,
        source_identifiers=source_identifiers,
    )


def test_build_ct026_power_config_uses_exact_open_ephys_paths_channels_and_cache(
    tmp_path: Path,
) -> None:
    """CT026 configuration must preserve fixed session, site, cache, and filter contracts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Synthetic CT026 session root; no raw files are created or read.
    """
    config = build_ct026_power_config(tmp_path)

    derived = tmp_path / "ephys" / "derived"
    aligned = tmp_path / "ephys" / "aligned" / "aligned_open_ephys"
    assert config.session_id == _SESSION_ID
    assert (
        config.trial_table_path
        == tmp_path / "processed" / f"{_SESSION_ID}_augmented_trials.csv"
    )
    assert config.output_directory == tmp_path / "processed" / "lfp_summary_cache"
    assert config.trial_filter.choice == "all" and config.trial_filter.context == "all"
    assert config.random_seed == 0
    assert [(site.stable_id, site.saved_channel_index) for site in config.sites] == [
        ("PFC", 5),
        ("HPC1", 222),
        ("HPC2", 14),
    ]
    assert all(site.acquisition_format == "open_ephys" for site in config.sites)
    assert (
        config.sites[0].lfp_path
        == derived / "Record_Node_101_Neuropix-PXI-110.ProbeA/lfp.dat"
    )
    assert (
        config.sites[1].lfp_path
        == derived / "Record_Node_101_Neuropix-PXI-110.ProbeB/lfp.dat"
    )
    assert (
        config.sites[2].lfp_path
        == derived / "Record_Node_101_Neuropix-PXI-110.ProbeB/lfp.dat"
    )
    assert config.sites[0].aligned_sync_path == aligned / "probeA_sync.npz"
    assert all(
        site.aligned_sync_path == aligned / "probeB_sync.npz"
        for site in config.sites[1:]
    )


def test_power_validation_writes_immutable_report_and_cached_site_pngs(
    tmp_path: Path,
) -> None:
    """Power validation must render cache-only PNGs and report all required runtime metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary session and analysis-run parent directories; no external files
        or raw recordings are accessed.
    """
    config = build_ct026_power_config(tmp_path / "session")
    calls: list[str] = []
    result = run_power_validation(
        config,
        tmp_path / "analysis-runs",
        _validation_dependencies(calls, power_result=_completed_power_result()),
    )

    assert result.component == "power"
    assert result.run_directory.parent == tmp_path / "analysis-runs"
    assert _SESSION_ID in result.run_directory.name
    assert result.run_directory.name.endswith("2026-08-25T01-02-03Z")
    assert result.manifest_snapshot_path.name == "manifest.json"
    assert result.configuration_snapshot_path.name == "configuration.json"
    assert result.summary_path.name == "run_summary.md"
    assert result.log_path.name == "run.log"
    assert result.source_identifiers_path.name == "source_identifiers.json"
    assert len(result.png_paths) == 12
    assert all(
        path.suffix == ".png" and path.parent == result.run_directory
        for path in result.png_paths
    )
    assert result.wall_time_s == pytest.approx(12.5)
    assert result.peak_memory_bytes == 4096
    assert result.cache_size_bytes >= 0 and result.component_size_bytes >= 0
    assert result.trial_count == 1
    assert result.exclusion_count == 0
    assert "warnings" in result.report and "nine_filter_projection" in result.report
    assert result.report["total_trial_count"] == 2
    assert result.report["filtered_trial_count"] == 1
    assert result.report["condition_effective_trial_count"]["PFC"][
        "correct_rewarded"
    ] == 1
    assert result.report["condition_effective_trial_count"]["PFC"]["omission"] == 0
    assert result.report["site_valid_trial_count"] == {
        "PFC": 1,
        "HPC1": 1,
        "HPC2": 1,
    }
    assert any(
        "presession" in warning.lower() for warning in result.report["warnings"]
    )
    summary_text = result.summary_path.read_text(encoding="ascii")
    assert "Goal" in summary_text and _SESSION_ID in summary_text
    assert calls.count("compute_power") == 1
    assert (
        "load_power" in calls
        and calls.count("plot_psd") == 6
        and calls.count("plot_band") == 6
    )
    assert calls.count("save_png") == 12 and calls.count("close_figure") == 12
    assert {path.name for path in result.png_paths} == {
        f"{site}_{group}_{kind}.png"
        for site in ("PFC", "HPC1", "HPC2")
        for group in ("base", "subdivisions")
        for kind in ("condition_psd", "band_power")
    }


def test_cached_power_validation_renders_without_recomputing_power(
    tmp_path: Path,
) -> None:
    """A revised report must reuse a compatible cache and measured run metrics.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary cache and report roots; no raw recordings are accessed.
    """
    config = build_ct026_power_config(tmp_path / "session")
    calls: list[str] = []

    result = render_cached_power_validation(
        config,
        tmp_path / "analysis-runs",
        _validation_dependencies(calls, power_result=_completed_power_result()),
        power_wall_time_s=12.5,
        power_peak_memory_bytes=4096,
    )

    assert "compute_power" not in calls
    assert "load_power" in calls
    assert result.report["power_recomputed"] is False
    assert result.wall_time_s == 12.5
    assert result.peak_memory_bytes == 4096
    assert len(result.png_paths) == 12


def test_power_validation_refuses_to_overwrite_same_timestamped_run(
    tmp_path: Path,
) -> None:
    """A repeated timestamp must stop before computation and preserve prior reports.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary session and analysis-run parents with filesystem-path units.
    """
    config = build_ct026_power_config(tmp_path / "session")
    calls: list[str] = []
    first = run_power_validation(
        config,
        tmp_path / "analysis-runs",
        _validation_dependencies(calls, power_result=_completed_power_result()),
    )
    original_summary = first.summary_path.read_bytes()
    calls.clear()

    with pytest.raises(FileExistsError):
        run_power_validation(
            config,
            tmp_path / "analysis-runs",
            _validation_dependencies(calls, power_result=_completed_power_result()),
        )

    assert calls == []
    assert first.summary_path.read_bytes() == original_summary


def test_power_validation_aborts_before_cache_load_when_power_component_fails(
    tmp_path: Path,
) -> None:
    """Failed Power computation must not load cache, render plots, or write a report.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary session and run-parent directories with filesystem-path units.
    """
    calls: list[str] = []
    config = build_ct026_power_config(tmp_path / "session")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        run_power_validation(
            config,
            tmp_path / "analysis-runs",
            _validation_dependencies(calls, power_result=_failed_power_result()),
        )

    assert calls == ["compute_power"]
    assert not (tmp_path / "analysis-runs").exists()
