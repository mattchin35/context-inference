"""RED contracts for the immutable CT026 100-shuffle Spike-phase preview."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.neural_analysis.lfp_summary_models import (
    UnitPopulationConfig,
    component_fingerprint,
)
from src.neural_analysis.lfp_summary_pipeline import ComponentRunResult
from src.neural_analysis.lfp_spike_phase_validation import (
    build_ct026_spike_phase_preview_config,
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
