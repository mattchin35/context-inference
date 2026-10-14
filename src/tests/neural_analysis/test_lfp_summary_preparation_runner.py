"""Contracts for metadata-driven Power/Synchrony/phase preparation."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from src.neural_analysis.lfp_summary.models import default_lfp_summary_config
from src.neural_analysis.lfp_summary.pipeline import ComponentRunResult


def _metadata_path(tmp_path: Path) -> Path:
    """Write one version-3 control file and return its canonical path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest directory used only for scalar control-file fixtures.

    Returns
    -------
    pathlib.Path
        Canonical metadata path. No numerical arrays or physical-unit-bearing
        samples are written.
    """
    session = tmp_path / "session"
    session.mkdir()
    payload = {
        "schema_version": "3",
        "session": "synthetic-session",
        "acquisition": "open_ephys",
        "behavior": {"trials": "processed/trials.csv", "events": None},
        "probes": {
            "ProbeA": {
                "lfp": "ephys/A/lfp.dat",
                "alignment": "ephys/A/alignment.npz",
                "sorter": "ephys/A/sorter",
                "quality": "ephys/A/channel_quality.csv",
                "sites": {"PFC1": 0},
            },
            "ProbeB": {
                "lfp": "ephys/B/lfp.dat",
                "alignment": "ephys/B/alignment.npz",
                "sorter": "ephys/B/sorter",
                "quality": "ephys/B/channel_quality.csv",
                "sites": {"HPC1": 0},
            },
        },
        "site_pairs": [["PFC1", "HPC1"]],
        "analysis": {
            "probe_roles": {"pfc": "ProbeA", "hpc": "ProbeB", "ppc": "ProbeB"},
            "ppc": {
                "prerequisite_cache": "processed/ppc_source",
                "preview_cache": "processed/ppc_preview",
                "final_cache": "processed/ppc_final",
                "preview_shuffles": 100,
                "final_shuffles": 1000,
            },
        },
    }
    path = session / "neural_session.json"
    path.write_text(json.dumps(payload) + "\n", encoding="ascii")
    return path


def _config(session: Path, cache: Path, shuffles: int) -> object:
    """Return a scalar immutable configuration for injected runner seams.

    Parameters are filesystem identities and a dimensionless shuffle count.
    The returned configuration contains no loaded numerical arrays.
    """
    base = default_lfp_summary_config()
    return replace(
        base,
        session_id="synthetic-session",
        session_path=session,
        output_directory=cache,
        ppc=replace(base.ppc, shuffle_count=shuffles),
    )


def _complete(component: str) -> ComponentRunResult:
    """Return one successful metadata-only component result.

    Parameters
    ----------
    component : str
        Categorical component identity without physical units.

    Returns
    -------
    ComponentRunResult
        Successful result containing no numerical payload arrays.
    """
    return ComponentRunResult(component, "complete", f"complete_{component}", None, {})


def test_prepare_computes_missing_power_once_and_reuses_one_phase_for_synchrony(
    tmp_path: Path,
) -> None:
    """Cold preparation uses existing component seams with one phase object.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest directory used for control files and preparation receipts.

    Returns
    -------
    None
        Call ordering and object identity prove no duplicate numerical path or
        second phase transformation is introduced.
    """
    from src.neural_analysis.lfp_summary.preparation_runner import (
        PreparationDependencies,
        PreparationRequest,
        prepare_session_lfp,
    )

    metadata_path = _metadata_path(tmp_path)
    session = metadata_path.parent.resolve()
    calls: list[str] = []
    states = {"power": "missing", "synchrony": "missing"}
    phase = SimpleNamespace(prepared_phase_cache_state="cold")

    def load_config(
        _metadata_path: Path,
        _probe: str,
        cache: Path,
        shuffles: int,
        _workers: int,
    ) -> object:
        """Return a scalar config for one requested cache/profile."""
        return _config(session, cache, shuffles)

    def component_status(config: object, component: str) -> str:
        """Return mutable scalar component state for the source cache."""
        if config.output_directory.name != "ppc_source":
            return "missing"
        return states[component]

    def compute_power(config: object) -> ComponentRunResult:
        """Record one production-power dispatch without numerical work."""
        calls.append(f"power:{config.output_directory.name}")
        states["power"] = "compatible"
        return _complete("power")

    def prepare_phase(config: object) -> object:
        """Return one opaque cold phase object for exact identity assertions."""
        calls.append(f"phase:{config.output_directory.name}")
        return phase

    def compute_synchrony(config: object, prepared: object) -> ComponentRunResult:
        """Require the exact prepared object and record one Synchrony dispatch."""
        assert prepared is phase
        calls.append(f"synchrony:{config.output_directory.name}")
        states["synchrony"] = "compatible"
        return _complete("synchrony")

    def adopt(*, source: Path, preview: Path, final: Path, **_: object) -> None:
        """Record metadata-only publication of the independent target pair."""
        calls.append(f"adopt:{source.name}:{preview.name}:{final.name}")

    dependencies = PreparationDependencies(
        load_config=load_config,
        component_status=component_status,
        compute_power=compute_power,
        prepare_phase=prepare_phase,
        compute_synchrony=compute_synchrony,
        adopt_cache_pair=adopt,
        repository_state=lambda: ("a" * 40, True),
        now_utc=lambda: "2030-01-02T03-04-05Z",
    )

    result = prepare_session_lfp(
        PreparationRequest(metadata_path, ("prepare", str(metadata_path))),
        dependencies,
    )

    assert result.status == "complete"
    assert result.prepared_phase_state == "cold"
    assert calls == [
        "power:ppc_source",
        "phase:ppc_source",
        "synchrony:ppc_source",
        "adopt:ppc_source:ppc_preview:ppc_final",
    ]
    receipt = json.loads(result.receipt_path.read_text(encoding="ascii"))
    assert receipt["stages"] == {
        "power": "computed",
        "prepared_phase": "computed",
        "synchrony": "computed",
        "cache_pair": "adopted",
    }
    assert receipt["git_commit"] == "a" * 40


def test_prepare_is_idempotent_for_complete_targets_and_warm_phase(
    tmp_path: Path,
) -> None:
    """A fully prepared session performs no component computation or copying.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest directory used for control files and a new scalar receipt.

    Returns
    -------
    None
        Compatible target caches and warm phase state must be reported as
        reused while numerical and adoption seams remain untouched.
    """
    from src.neural_analysis.lfp_summary.preparation_runner import (
        PreparationDependencies,
        PreparationRequest,
        prepare_session_lfp,
    )

    metadata_path = _metadata_path(tmp_path)
    session = metadata_path.parent.resolve()
    processed = session / "processed"
    for name in ("ppc_preview", "ppc_final"):
        (processed / name).mkdir(parents=True, exist_ok=True)
    calls: list[str] = []

    def load_config(
        _metadata_path: Path,
        _probe: str,
        cache: Path,
        shuffles: int,
        _workers: int,
    ) -> object:
        """Return one scalar configuration for the selected target path."""
        return _config(session, cache, shuffles)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        """Fail if a completed preparation dispatches numerical/copy work."""
        raise AssertionError("completed preparation must be reused")

    def prepare_phase(config: object) -> object:
        """Record the required warm-cache validation and return its state."""
        calls.append(f"phase:{config.output_directory.name}")
        return SimpleNamespace(prepared_phase_cache_state="warm")

    dependencies = PreparationDependencies(
        load_config=load_config,
        component_status=lambda config, _component: (
            "compatible"
            if config.output_directory.name in {"ppc_preview", "ppc_final"}
            else "missing"
        ),
        compute_power=forbidden,
        prepare_phase=prepare_phase,
        compute_synchrony=forbidden,
        adopt_cache_pair=forbidden,
        repository_state=lambda: ("b" * 40, True),
        now_utc=lambda: "2030-01-02T03-04-06Z",
    )

    result = prepare_session_lfp(
        PreparationRequest(metadata_path, ("prepare", str(metadata_path))),
        dependencies,
    )

    assert result.status == "complete"
    assert result.prepared_phase_state == "warm"
    assert calls == ["phase:ppc_preview"]
    receipt = json.loads(result.receipt_path.read_text(encoding="ascii"))
    assert receipt["stages"] == {
        "power": "reused",
        "prepared_phase": "reused",
        "synchrony": "reused",
        "cache_pair": "reused",
    }


def test_prepare_refuses_partial_target_pair_without_overwrite(tmp_path: Path) -> None:
    """One existing target blocks preparation before any numerical dispatch.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest directory used for scalar filesystem fixtures.

    Returns
    -------
    None
        The runner raises without calling compute, phase, Synchrony, or adoption
        seams and leaves the existing target untouched.
    """
    import pytest

    from src.neural_analysis.lfp_summary.preparation_runner import (
        PreparationDependencies,
        PreparationRequest,
        prepare_session_lfp,
    )

    metadata_path = _metadata_path(tmp_path)
    session = metadata_path.parent.resolve()
    preview = session / "processed" / "ppc_preview"
    preview.mkdir(parents=True)
    marker = preview / "owned.txt"
    marker.write_text("preserve\n", encoding="ascii")

    def load_config(
        _metadata_path: Path,
        _probe: str,
        cache: Path,
        shuffles: int,
        _workers: int,
    ) -> object:
        """Return one scalar configuration for the selected cache path."""
        return _config(session, cache, shuffles)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        """Fail if unsafe partial-state validation reaches later work."""
        raise AssertionError("partial targets must fail before work")

    dependencies = PreparationDependencies(
        load_config=load_config,
        component_status=lambda _config, _component: "missing",
        compute_power=forbidden,
        prepare_phase=forbidden,
        compute_synchrony=forbidden,
        adopt_cache_pair=forbidden,
        repository_state=lambda: ("c" * 40, True),
        now_utc=lambda: "2030-01-02T03-04-07Z",
    )

    with pytest.raises(ValueError, match="partial|preview|final"):
        prepare_session_lfp(
            PreparationRequest(metadata_path, ("prepare", str(metadata_path))),
            dependencies,
        )

    assert marker.read_text(encoding="ascii") == "preserve\n"


def test_prepare_adopts_complete_rsynced_source_before_cluster_phase(
    tmp_path: Path,
) -> None:
    """A transferred source uses relocation validation, not active-path status.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty pytest directory used for scalar cache-member and receipt fixtures.

    Returns
    -------
    None
        Adoption precedes one cluster-local phase preparation; Power/Synchrony
        computation and source status classification are never dispatched.
    """
    from src.neural_analysis.lfp_summary.preparation_runner import (
        PreparationDependencies,
        PreparationRequest,
        prepare_session_lfp,
    )

    metadata_path = _metadata_path(tmp_path)
    session = metadata_path.parent.resolve()
    source = session / "processed" / "ppc_source"
    source.mkdir(parents=True)
    for name in ("manifest.json", "power.npz", "synchrony.npz"):
        (source / name).write_bytes(b"transferred")
    calls: list[str] = []

    def load_config(
        _metadata_path: Path,
        _probe: str,
        cache: Path,
        shuffles: int,
        _workers: int,
    ) -> object:
        """Return one scalar target configuration after adoption."""
        return _config(session, cache, shuffles)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        """Fail if transferred source bytes enter a numerical/status path."""
        raise AssertionError("rsynced source must be handled by adoption")

    def adopt(*, preview: Path, final: Path, **_: object) -> None:
        """Model successful metadata/header relocation to the target pair."""
        calls.append("adopt")
        preview.mkdir()
        final.mkdir()

    def prepare_phase(config: object) -> object:
        """Record one cluster-local phase preparation against preview config."""
        calls.append(f"phase:{config.output_directory.name}")
        return SimpleNamespace(prepared_phase_cache_state="cold")

    dependencies = PreparationDependencies(
        load_config=load_config,
        component_status=forbidden,
        compute_power=forbidden,
        prepare_phase=prepare_phase,
        compute_synchrony=forbidden,
        adopt_cache_pair=adopt,
        repository_state=lambda: ("d" * 40, True),
        now_utc=lambda: "2030-01-02T03-04-08Z",
    )

    result = prepare_session_lfp(
        PreparationRequest(metadata_path, ("prepare", str(metadata_path))),
        dependencies,
    )

    assert calls == ["adopt", "phase:ppc_preview"]
    receipt = json.loads(result.receipt_path.read_text(encoding="ascii"))
    assert receipt["stages"] == {
        "power": "reused",
        "prepared_phase": "computed",
        "synchrony": "reused",
        "cache_pair": "adopted",
    }
