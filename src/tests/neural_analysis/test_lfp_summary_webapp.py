"""Pure-helper and fake-UI contracts for the cached LFP summary webapp route."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_summary_webapp
from src.neural_analysis.lfp_summary_io import ComponentStatus
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    UnitPopulationConfig,
    canonical_config_json,
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


def _snapshot_receipt_digest(file_entries: list[dict[str, object]]) -> str:
    """Return the ordered transfer identity for synthetic final-cache files.

    Parameters
    ----------
    file_entries : list[dict[str, object]]
        Receipt entries ordered by filename. Each entry has an ASCII filename,
        integer ``size_bytes``, and lowercase hexadecimal ``sha256`` digest;
        it contains no numerical arrays or physical units.

    Returns
    -------
    str
        Lowercase SHA-256 of the newline-delimited filename, size, and per-file
        digest records, including a final newline. This is an identity checksum,
        not a scientific value.
    """

    lines = [
        f"{entry['filename']}\t{entry['size_bytes']}\t{entry['sha256']}"
        for entry in file_entries
    ]
    return sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def _write_snapshot_fixture(
    root: Path,
    *,
    component_states: dict[str, str] | None = None,
    component_population_labels: dict[str, str] | None = None,
    component_configuration_snapshots: dict[str, dict[str, object]] | None = None,
    receipt_transform: Callable[[dict[str, object]], None] | None = None,
) -> Path:
    """Create a tiny immutable-cache topology for receipt-only UI contracts.

    Parameters
    ----------
    root : pathlib.Path
        Writable temporary-test root. The helper creates exactly one
        ``cache_snapshot`` child and never reads session, CT026, or network data.
    component_states : dict[str, str] | None
        Optional committed manifest state for each scientific component. Omitted
        states are ``"complete"``. These are categorical state strings, not
        numerical results.
    component_population_labels : dict[str, str] | None
        Optional stable selected-population label for a component. Labels are
        categorical provenance values such as ``"ProbeB"`` and have no units.
    component_configuration_snapshots : dict[str, dict[str, object]] | None
        Optional complete JSON-compatible saved ``LFPSummaryConfig`` mappings
        for selected components. They retain source paths, seconds, Hz, and
        source-voltage-unit provenance without containing raw arrays.
    receipt_transform : callable or None
        Optional in-place mutation of the JSON-compatible receipt after its
        correct file identity has been recorded. It receives and returns no
        arrays, axes, units, or filesystem paths.

    Returns
    -------
    pathlib.Path
        Exact synthetic ``cache_snapshot`` directory containing an ASCII
        manifest, three tiny NPZ component files, and an identity receipt.
    """

    snapshot = root / "cache_snapshot"
    snapshot.mkdir()
    states = component_states or {}
    population_labels = component_population_labels or {}
    configuration_snapshots = component_configuration_snapshots or {}
    manifest = {
        "schema_version": "1",
        "session_id": "synthetic-session",
        "components": {
            component: {
                "state": states.get(component, "complete"),
                **(
                    {"configuration_snapshot": configuration_snapshots[component]}
                    if component in configuration_snapshots
                    else (
                        {
                        "configuration_snapshot": {
                            "unit_population": {
                                "label": f"{population_labels[component]} active",
                                "probe_label": population_labels[component],
                                "sorter_path": (
                                    f"/cluster/final/{population_labels[component]}/kilosort4"
                                ),
                                "aligned_spike_path": (
                                    f"/cluster/final/{population_labels[component]}/aligned_spikes.npz"
                                ),
                                "selected_channels": [1],
                                "quality_settings": [
                                    ["channel_quality", "good"],
                                    ["inside_brain", "true"],
                                    ["unit_quality", "good,mua"],
                                ],
                                "stable_unit_ids": [f"{population_labels[component]}:7"],
                            }
                        }
                        }
                        if component in population_labels
                        else {}
                    )
                ),
            }
            for component in lfp_summary_webapp.SUMMARY_COMPONENTS
        },
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for component in lfp_summary_webapp.SUMMARY_COMPONENTS:
        np.savez(
            snapshot / f"{component}.npz",
            marker=np.array([len(component)], dtype=np.int64),
        )
    _write_snapshot_receipt(snapshot, receipt_transform=receipt_transform)
    return snapshot


def _write_snapshot_receipt(
    snapshot: Path,
    *,
    receipt_transform: Callable[[dict[str, object]], None] | None = None,
) -> None:
    """Write a synthetic receipt from the current exact snapshot file identities.

    Parameters
    ----------
    snapshot : pathlib.Path
        Existing synthetic ``cache_snapshot`` directory containing exactly the
        manifest and three scientific NPZ component files. It has no raw data
        arrays beyond the small component marker arrays.
    receipt_transform : callable or None
        Optional in-place mutation of JSON-compatible receipt metadata after
        its correct identity is calculated. It receives no numerical arrays,
        physical units, or external paths.

    Returns
    -------
    None
        Replaces only ``cache_snapshot_identity.json`` using current filename,
        byte-size, and SHA-256 records. It does not modify scientific files.
    """

    file_entries: list[dict[str, object]] = []
    for filename in ("manifest.json", "power.npz", "synchrony.npz", "spike_phase.npz"):
        path = snapshot / filename
        file_entries.append(
            {
                "filename": filename,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()).hexdigest(),
            }
        )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "source_cluster_directory": "/cluster/final/cache",
        "copied_at_utc": "2026-09-22T00:00:00Z",
        "files": file_entries,
        "aggregate": {
            "algorithm": "sha256",
            "canonical_record_format": "filename\\tsize_bytes\\tsha256\\n",
            "order": [entry["filename"] for entry in file_entries],
            "sha256": _snapshot_receipt_digest(file_entries),
        },
    }
    if receipt_transform is not None:
        receipt_transform(receipt)
    (snapshot / "cache_snapshot_identity.json").write_text(
        json.dumps(receipt),
        encoding="ascii",
    )


def _change_snapshot_component_identity(snapshot: Path, component: str) -> None:
    """Change one synthetic component and manifest identity at a fixed path.

    Parameters
    ----------
    snapshot : pathlib.Path
        Existing synthetic final snapshot whose four committed files may be
        rewritten only within this temporary test fixture.
    component : str
        Selected scientific component filename stem. It is categorical and must
        be one of the approved summary components; marker arrays are shape
        ``(1,)`` and dimensionless.

    Returns
    -------
    None
        Updates the selected component marker, adds a manifest-only synthetic
        revision, and writes a coherent new receipt. No session or network file
        is read.
    """

    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    components = manifest["components"]
    assert isinstance(components, dict)
    entry = components[component]
    assert isinstance(entry, dict)
    entry["synthetic_snapshot_revision"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    np.savez(snapshot / f"{component}.npz", marker=np.array([99], dtype=np.int64))
    _write_snapshot_receipt(snapshot)


def _set_receipt_schema_version_to_text(receipt: dict[str, object]) -> None:
    """Replace a receipt's integer schema version with text for rejection tests.

    Parameters
    ----------
    receipt : dict[str, object]
        JSON-compatible transfer receipt without numerical arrays or units.

    Returns
    -------
    None
        Mutates only the categorical ``schema_version`` field in-place.
    """

    receipt["schema_version"] = "1"


def _remove_receipt_final_newline_format(receipt: dict[str, object]) -> None:
    """Corrupt only the receipt's required canonical aggregate record format.

    Parameters
    ----------
    receipt : dict[str, object]
        JSON-compatible transfer receipt without numerical arrays or units.

    Returns
    -------
    None
        Mutates only aggregate metadata, retaining all files and per-file hashes.
    """

    aggregate = receipt["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["canonical_record_format"] = "filename\\tsize_bytes\\tsha256"


def _reverse_first_receipt_files(receipt: dict[str, object]) -> None:
    """Change the required manifest-first aggregate order for rejection tests.

    Parameters
    ----------
    receipt : dict[str, object]
        JSON-compatible transfer receipt without numerical arrays or units.

    Returns
    -------
    None
        Mutates only the categorical aggregate order metadata in-place.
    """

    aggregate = receipt["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["order"] = [
        "power.npz",
        "manifest.json",
        "synchrony.npz",
        "spike_phase.npz",
    ]


def _synthetic_launcher_command(action: str, config: object) -> str:
    """Return one synthetic launcher command for action-dispatch contract tests.

    Parameters
    ----------
    action : str
        Categorical requested summary action with no numerical data.
    config : object
        Opaque immutable configuration passed through without inspection.

    Returns
    -------
    str
        ASCII launcher handoff text; it represents no executed shell command.
    """

    del config
    return f"launcher {action}"


def _synthetic_slurm_command(action: str, config: object) -> str:
    """Return one synthetic Slurm handoff string without submitting a job.

    Parameters
    ----------
    action : str
        Categorical requested summary action with no numerical data.
    config : object
        Opaque immutable configuration passed through without inspection.

    Returns
    -------
    str
        ASCII ``sbatch`` handoff text; it is never executed by a test.
    """

    del config
    return f"sbatch {action}"


def _full_saved_component_configuration(root: Path, probe_label: str = "ProbeB") -> dict[str, object]:
    """Return one complete JSON saved configuration for immutable-plot tests.

    Parameters
    ----------
    root : pathlib.Path
        Temporary local root used only for synthetic source/output path values.
        It contains no experimental recording and all saved time/frequency units
        are inherited from ``default_lfp_summary_config``.
    probe_label : str
        Exact ``"ProbeA"`` or ``"ProbeB"`` population label stored with the
        snapshot component.

    Returns
    -------
    dict[str, object]
        Complete JSON-compatible ``LFPSummaryConfig`` mapping retaining session,
        window, notch, band, source-unit, and selected-population provenance.
    """

    population = UnitPopulationConfig(
        f"{probe_label} saved active",
        probe_label,
        root / "cluster_sorter",
        root / "cluster_aligned_spikes.npz",
        (1,),
        (("channel_quality", "good"), ("inside_brain", "true"), ("unit_quality", "good,mua")),
        (f"{probe_label}:7",),
    )
    configuration = replace(
        default_lfp_summary_config(),
        session_id="cluster-produced-session",
        session_path=root / "cluster_session",
        output_directory=root / "cluster_cache",
        sites=_summary_sites(root),
        unit_population=population,
    )
    return json.loads(canonical_config_json(configuration))


def _rewrite_snapshot_manifest(
    snapshot: Path,
    update: Callable[[dict[str, object]], None],
) -> None:
    """Apply one JSON-only manifest mutation and refresh its synthetic receipt.

    Parameters
    ----------
    snapshot : pathlib.Path
        Existing temporary synthetic final snapshot. Only its manifest and
        receipt are rewritten; component marker arrays and external data are
        untouched.
    update : callable
        In-place mutator accepting the decoded JSON manifest mapping. It receives
        metadata only, not numerical arrays, physical units, or source files.

    Returns
    -------
    None
        Leaves a receipt whose ordered component identities match the modified
        temporary manifest.
    """

    manifest_path = snapshot / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    update(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _write_snapshot_receipt(snapshot)


def _small_power_snapshot_arrays() -> dict[str, np.ndarray]:
    """Return small cached Power arrays with ``(site, trial, epoch, frequency)`` axes.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dimensionless dB PSD/band arrays, Hz frequency coordinates, categorical
        condition/site/epoch/band axes, and trial-count metadata. No raw LFP
        trace is present.
    """

    return {
        "frequency_hz": np.array((8.0, 40.0)),
        "normalized_psd_session_db": np.arange(24, dtype=float).reshape(2, 2, 3, 2),
        "band_power_session_db": np.arange(24, dtype=float).reshape(2, 2, 3, 2),
        "condition_names": np.array(("all", "left")),
        "condition_membership": np.array(((True, False), (True, True))),
        "filter_membership": np.array((True, True)),
        "condition_effective_trial_count": np.array(((2, 2), (1, 1)), dtype=np.int64),
        "site_ids": np.array(("PFC", "HPC1")),
        "site_voltage_units": np.array(("uV", "uV")),
        "epoch_names": np.array(("whole", "before", "after")),
        "band_names": np.array(("theta", "gamma")),
    }


def _small_synchrony_snapshot_arrays() -> dict[str, np.ndarray]:
    """Return small committed Synchrony arrays using their saved public axes.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dimensionless ITPC/ISPC/PLV arrays with condition/site-or-pair/epoch/band
        axes, Hz/seconds coordinates, trial/sample counts, and tiny cached trace
        provenance. The arrays contain no raw recording access seam.
    """

    return {
        "trial_indices": np.array((10, 11), dtype=np.int64),
        "site_ids": np.array(("PFC", "HPC1")),
        "site_voltage_units": np.array(("uV", "uV")),
        "condition_names": np.array(("all", "left")),
        "condition_membership": np.array(((True, False), (True, True))),
        "filter_membership": np.array((True, True)),
        "frequency_hz": np.array((8.0, 40.0)),
        "epoch_names": np.array(("whole", "before", "after")),
        "band_names": np.array(("theta", "gamma")),
        "relative_time_s": np.array((-0.1, 0.1)),
        "site_valid": np.array(((True, True), (True, True))),
        "pair_valid": np.array(((True, True),)),
        "pair_site_a_ids": np.array(("PFC",)),
        "pair_site_b_ids": np.array(("HPC1",)),
        "itpc": np.full((2, 2, 2, 2), 0.11),
        "itpc_effective_trial_count": np.full((2, 2, 2, 2), 2, dtype=np.int64),
        "ispc": np.full((2, 1, 2, 2), 0.22),
        "ispc_effective_trial_count": np.full((2, 1, 2, 2), 2, dtype=np.int64),
        "itpc_band_mean": np.full((2, 2, 3, 2), 0.33),
        "itpc_ci_low": np.full((2, 2, 3, 2), 0.30),
        "itpc_ci_high": np.full((2, 2, 3, 2), 0.36),
        "itpc_unstable": np.zeros((2, 2, 3, 2), dtype=bool),
        "ispc_band_mean": np.full((2, 1, 3, 2), 0.44),
        "ispc_ci_low": np.full((2, 1, 3, 2), 0.40),
        "ispc_ci_high": np.full((2, 1, 3, 2), 0.48),
        "ispc_unstable": np.zeros((2, 1, 3, 2), dtype=bool),
        "plv_by_frequency": np.full((2, 1, 3, 2), 0.55),
        "plv_valid_sample_count": np.full((2, 1, 3, 2), 7, dtype=np.int64),
        "plv_valid_sample_fraction": np.full((2, 1, 3, 2), 0.75),
        "plv_computable": np.ones((2, 1, 3, 2), dtype=bool),
        "plv_band_mean": np.full((2, 1, 3, 2), 0.55),
        "source_trace": np.ones((2, 2, 2), dtype=float),
        "band_filtered_trace": np.ones((2, 2, 2, 2), dtype=float),
        "hilbert_phase_rad": np.zeros((2, 2, 2, 2), dtype=float),
    }


def _small_spike_snapshot_arrays() -> dict[str, np.ndarray]:
    """Return small committed Spike arrays with ``(unit, condition, site, epoch, frequency)`` axes.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dimensionless PPC statistics, boolean eligibility/reliability/FDR masks,
        count arrays, Hz/seconds coordinates, and cached exemplar metadata. No
        raw spikes, source recordings, or numerical computation is performed.
    """

    metric_shape = (2, 2, 2, 3, 2)
    band_shape = (2, 2, 2, 3, 2)
    arrays: dict[str, np.ndarray] = {
        "unit_ids": np.array(("ProbeB:7", "ProbeB:8")),
        "trial_indices": np.array((10, 11), dtype=np.int64),
        "site_ids": np.array(("PFC", "HPC1")),
        "site_voltage_units": np.array(("uV", "uV")),
        "condition_names": np.array(("all", "left")),
        "condition_membership": np.array(((True, False), (True, True))),
        "filter_membership": np.array((True, True)),
        "epoch_names": np.array(("whole", "before", "after")),
        "band_names": np.array(("theta", "gamma")),
        "frequency_hz": np.array((8.0, 40.0)),
        "relative_time_s": np.array((-0.1, 0.1)),
        "phase_bin_edges_rad": np.array((-np.pi, 0.0, np.pi)),
        "relative_spike_times_s": np.array((-0.05, 0.03, -0.02, 0.04)),
        "relative_spike_time_offsets": np.array(((0, 1, 2), (2, 3, 4)), dtype=np.int64),
        "ppc": np.full(metric_shape, 0.2),
        "computable": np.ones(metric_shape, dtype=bool),
        "reliable": np.ones(metric_shape, dtype=bool),
        "null_eligible": np.ones(metric_shape, dtype=bool),
        "significant": np.zeros(metric_shape, dtype=bool),
        "spike_count": np.full(metric_shape, 7, dtype=np.int64),
        "eligible_trial_count": np.full(metric_shape, 2, dtype=np.int64),
        "preferred_phase_rad": np.zeros(metric_shape),
        "representative_phase_hist_count": np.ones(metric_shape + (2,), dtype=np.int64),
        "ppc_band_mean": np.full(band_shape, 0.2),
        "source_trace": np.ones((2, 2, 2), dtype=float),
        "band_filtered_trace": np.ones((2, 2, 2, 2), dtype=float),
        "hilbert_phase_rad": np.zeros((2, 2, 2, 2), dtype=float),
        "selected_low_unit_ids": np.full((2, 2, 3, 2), "ProbeB:7"),
        "selected_high_unit_ids": np.full((2, 2, 3, 2), "ProbeB:8"),
        "illustrative_trial_indices": np.full((2, 2, 3, 2), 11, dtype=np.int64),
        "illustrative_low_trial_indices": np.full((2, 2, 3, 2), 10, dtype=np.int64),
        "illustrative_high_trial_indices": np.full((2, 2, 3, 2), 11, dtype=np.int64),
    }
    return arrays


class RouteSidebar:
    """Small deterministic sidebar with explicit source, probe, and action values.

    The stored controls are categorical UI labels, text paths, booleans, and
    scalar values only; this fake never stores arrays or opens files.
    ``snapshot_text`` is returned only for a text input whose label clearly
    requests a snapshot path/directory, preserving the route's explicit-input
    contract.
    """

    def __init__(
        self,
        *,
        source_mode: str,
        snapshot_text: str,
        action: str = "power",
        view: str = "synchrony",
    ) -> None:
        """Store deterministic categorical controls with no I/O side effects."""

        self.source_mode = source_mode
        self.snapshot_text = snapshot_text
        self.action = action
        self.view = view
        self.labels: list[str] = []
        self.button_labels: list[str] = []

    def header(self, label: str) -> None:
        """Record a visible categorical sidebar heading."""

        self.labels.append(label)

    def selectbox(self, label: str, *, options: tuple[str, ...], **kwargs: object) -> str:
        """Return a requested source/probe/action selector without filesystem access."""

        del kwargs
        self.labels.append(label)
        option_set = set(options)
        if option_set == {"snapshot", "live"}:
            return self.source_mode
        if option_set == {"ProbeA", "ProbeB"}:
            return "ProbeB"
        if "view" in label.lower() and self.view in option_set:
            return self.view
        if self.action in option_set:
            return self.action
        return options[0]

    def radio(self, label: str, *, options: tuple[str, ...], **kwargs: object) -> str:
        """Mirror selectbox for source-mode implementations using radio controls."""

        return self.selectbox(label, options=options, **kwargs)

    def text_input(self, label: str, *, value: str, **kwargs: object) -> str:
        """Return the explicit snapshot text only for a snapshot path control."""

        del kwargs
        self.labels.append(label)
        lowered = label.lower()
        if "snapshot" in lowered and ("path" in lowered or "directory" in lowered):
            return self.snapshot_text
        return value

    def checkbox(self, label: str, *, value: bool, **kwargs: object) -> bool:
        """Return existing boolean defaults without adding a compute action."""

        del kwargs
        self.labels.append(label)
        return value

    def number_input(self, label: str, *, value: float | int, **kwargs: object) -> float | int:
        """Return existing scalar defaults without altering saved units."""

        del kwargs
        self.labels.append(label)
        return value

    def button(self, label: str) -> bool:
        """Click only the action control for explicit live-mode route tests."""

        self.labels.append(label)
        self.button_labels.append(label)
        return self.source_mode == "live" and ("compute" in label.lower() or "run" in label.lower())


class RouteFakeStreamlit(FakeStreamlit):
    """Fake Streamlit route host retaining session state between deterministic rerenders."""

    def __init__(
        self,
        *,
        source_mode: str,
        snapshot_text: str,
        action: str = "power",
        view: str = "synchrony",
    ) -> None:
        """Create a no-I/O output host with persistent JSON-like session state."""

        super().__init__()
        self.sidebar = RouteSidebar(
            source_mode=source_mode,
            snapshot_text=snapshot_text,
            action=action,
            view=view,
        )
        self.session_state: dict[str, object] = {}

    def caption(self, message: str) -> None:
        """Record one noncomputational provenance/status caption."""

        self.messages.append(("caption", message))


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


def test_snapshot_receipt_accepts_only_the_exact_final_files_and_transfer_identity(
    tmp_path: Path,
) -> None:
    """A valid snapshot exposes provenance without replacing its cluster paths.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary root used for a four-file final-cache copy plus receipt.
    """

    snapshot = _write_snapshot_fixture(tmp_path)

    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)

    assert inspection.state == "valid"
    assert inspection.snapshot_directory == snapshot.resolve()
    assert inspection.source_cluster_directory == "/cluster/final/cache"
    assert inspection.component_paths == {
        component: snapshot / f"{component}.npz"
        for component in lfp_summary_webapp.SUMMARY_COMPONENTS
    }


@pytest.mark.parametrize(
    "receipt_transform",
    (
        _set_receipt_schema_version_to_text,
        _remove_receipt_final_newline_format,
        _reverse_first_receipt_files,
    ),
)
def test_snapshot_receipt_requires_integer_schema_and_exact_canonical_aggregate(
    tmp_path: Path,
    receipt_transform: Callable[[dict[str, object]], None],
) -> None:
    """Receipt schema and ordered-digest format are fixed transfer contracts.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary root used for one synthetic final snapshot receipt.
    receipt_transform : callable
        In-place categorical receipt mutation that preserves files while making
        the receipt schema or aggregate format invalid. It accepts and returns
        JSON-compatible metadata only, with no numerical arrays or units.
    """

    snapshot = _write_snapshot_fixture(tmp_path, receipt_transform=receipt_transform)

    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)

    assert inspection.state == "malformed"
    assert not inspection.is_scientific_result


@pytest.mark.parametrize(
    ("fixture_kind", "expected_state"),
    (
        ("missing", "missing"),
        ("malformed", "malformed"),
        ("tampered", "tampered"),
        ("incomplete", "incomplete"),
    ),
)
def test_snapshot_receipt_reports_noncommitted_states_without_live_fallback(
    tmp_path: Path,
    fixture_kind: str,
    expected_state: str,
) -> None:
    """Missing or invalid final-cache copies must remain noncomputational states.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary root used only for a tiny synthetic snapshot fixture.
    fixture_kind : str
        Categorical fixture mutation selecting a missing, malformed, tampered,
        or incomplete transfer state. It has no numerical units.
    expected_state : str
        Required visible categorical snapshot status for that mutation.
    """

    if fixture_kind == "missing":
        snapshot = tmp_path / "missing-cache_snapshot"
    else:
        snapshot = _write_snapshot_fixture(tmp_path)
        if fixture_kind == "malformed":
            (snapshot / "cache_snapshot_identity.json").write_text("{", encoding="ascii")
        elif fixture_kind == "tampered":
            (snapshot / "power.npz").write_bytes(b"tampered-after-receipt")
        else:
            (snapshot / "spike_phase.npz").unlink()

    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)

    assert inspection.state == expected_state
    assert not inspection.is_scientific_result


def test_snapshot_rejects_prepared_and_checkpoint_artifacts_as_scientific_components(
    tmp_path: Path,
) -> None:
    """Execution-only artifacts cannot be promoted into cache-inspection views.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary root used for a small synthetic cache and work artifact.
    """

    snapshot = _write_snapshot_fixture(
        tmp_path,
        component_states={"spike_phase": "prepared"},
    )
    np.savez(snapshot / "spike_phase_checkpoint.npz", marker=np.array([1]))

    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)

    assert inspection.state == "incomplete"
    assert not inspection.is_scientific_result
    assert "prepared" in inspection.message.lower() or "checkpoint" in inspection.message.lower()


def test_snapshot_source_is_default_blank_then_retains_only_the_explicit_path(
    tmp_path: Path,
) -> None:
    """Snapshot selection never discovers a run or substitutes a CT026 path.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root containing one explicitly supplied synthetic snapshot.
    """

    session_state: dict[str, object] = {}
    blank_source = lfp_summary_webapp.resolve_summary_source(
        source_mode="snapshot",
        entered_snapshot_directory="",
        session_state=session_state,
    )
    snapshot = _write_snapshot_fixture(tmp_path)
    entered_source = lfp_summary_webapp.resolve_summary_source(
        source_mode="snapshot",
        entered_snapshot_directory=str(snapshot),
        session_state=session_state,
    )
    rerendered_source = lfp_summary_webapp.resolve_summary_source(
        source_mode="snapshot",
        entered_snapshot_directory=str(snapshot),
        session_state=session_state,
    )
    cleared_source = lfp_summary_webapp.resolve_summary_source(
        source_mode="snapshot",
        entered_snapshot_directory="",
        session_state=session_state,
    )

    assert blank_source.mode == "snapshot"
    assert blank_source.snapshot_directory is None
    assert not blank_source.can_compute
    assert entered_source.snapshot_directory == snapshot.resolve()
    assert rerendered_source.snapshot_directory == snapshot.resolve()
    assert rerendered_source.mode == "snapshot"
    assert "latest" not in rerendered_source.message.lower()
    assert cleared_source.snapshot_directory is None
    assert not cleared_source.can_compute


def test_blank_or_invalid_snapshot_source_cannot_dispatch_live_or_compute_actions(
    tmp_path: Path,
) -> None:
    """Snapshot mode has no silent live-cache or numerical-action escape hatch.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary root used to form an invalid explicit snapshot path.
    """

    calls: list[str] = []
    source = lfp_summary_webapp.resolve_summary_source(
        source_mode="snapshot",
        entered_snapshot_directory=str(tmp_path / "absent"),
        session_state={},
    )

    result = lfp_summary_webapp.dispatch_summary_action(
        "power",
        source=source,
        config=default_lfp_summary_config(),
        dependencies=_fake_dependencies(calls, []),
        launcher_command_builder=_synthetic_launcher_command,
    )

    assert result.state == "blocked"
    assert calls == []
    assert "snapshot" in (result.error or "").lower()


def test_valid_snapshot_mode_blocks_actions_and_preserves_every_snapshot_file(
    tmp_path: Path,
) -> None:
    """Read-only snapshot inspection cannot dispatch live computation or write files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root containing one exact synthetic final snapshot.
    """

    snapshot = _write_snapshot_fixture(tmp_path)
    before = {
        path.name: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
        for path in snapshot.iterdir()
    }
    calls: list[str] = []
    source = lfp_summary_webapp.resolve_summary_source(
        source_mode="snapshot",
        entered_snapshot_directory=str(snapshot),
        session_state={},
    )

    result = lfp_summary_webapp.dispatch_summary_action(
        "synchrony",
        source=source,
        config=default_lfp_summary_config(),
        dependencies=_fake_dependencies(calls, []),
        launcher_command_builder=_synthetic_launcher_command,
    )

    after = {
        path.name: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
        for path in snapshot.iterdir()
    }
    assert result.state == "blocked"
    assert calls == []
    assert after == before


def test_snapshot_component_cache_opens_only_the_selected_component_once_per_identity(
    tmp_path: Path,
) -> None:
    """Rerenders cache one selected final component without touching unrelated files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root containing a receipt-validated synthetic final snapshot.
    """

    snapshot = _write_snapshot_fixture(tmp_path)
    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)
    cache = lfp_summary_webapp.SnapshotComponentCache()
    opened: list[str] = []

    def load_component(path: Path, manifest: dict[str, object], component: str) -> dict[str, np.ndarray]:
        """Record one component-path open and return a one-element synthetic array.

        Parameters are the exact selected NPZ path, JSON-compatible manifest, and
        categorical component name. The return contains a dimensionless
        one-dimensional ``marker`` array with shape ``(1,)``.
        """

        del path, manifest
        opened.append(component)
        return {"marker": np.array([len(component)], dtype=np.int64)}

    first = cache.load(inspection, "synchrony", load_component)
    second = cache.load(inspection, "synchrony", load_component)
    _change_snapshot_component_identity(snapshot, "synchrony")
    changed_inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)
    third = cache.load(changed_inspection, "synchrony", load_component)

    assert first is second
    assert changed_inspection.snapshot_directory == inspection.snapshot_directory
    assert third is not first
    assert opened == ["synchrony", "synchrony"]


@pytest.mark.parametrize("component", ("synchrony", "spike_phase"))
def test_snapshot_plotting_is_cache_only_and_closes_each_figure(
    tmp_path: Path,
    component: str,
) -> None:
    """Synchrony and Spike views must plot one cached component and close figures.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root containing a receipt-validated tiny snapshot.
    component : str
        Selected cached scientific component; no raw LFP, spike, or phase input
        is supplied to the view.
    """

    snapshot = _write_snapshot_fixture(tmp_path)
    streamlit = FakeStreamlit()
    opened: list[str] = []
    created_figure_numbers: list[int] = []
    before = {
        path.name: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
        for path in snapshot.iterdir()
    }

    def load_component(path: Path, manifest: dict[str, object], selected: str) -> dict[str, np.ndarray]:
        """Record the selected NPZ identity and return a synthetic marker array."""

        del path, manifest
        opened.append(selected)
        return {"marker": np.array([1], dtype=np.int64)}

    def plot_component(selected: str, arrays: dict[str, np.ndarray]) -> plt.Figure:
        """Build one unsaved figure from a selected dimensionless marker array."""

        assert selected == component
        assert arrays["marker"].shape == (1,)
        figure, _axis = plt.subplots()
        created_figure_numbers.append(figure.number)
        return figure

    lfp_summary_webapp.render_snapshot_component_view(
        streamlit,
        snapshot_directory=snapshot,
        component=component,
        component_cache=lfp_summary_webapp.SnapshotComponentCache(),
        load_component=load_component,
        plot_component=plot_component,
    )

    assert opened == [component]
    assert streamlit.figures
    assert not plt.fignum_exists(created_figure_numbers[0])
    after = {
        path.name: (path.stat().st_size, sha256(path.read_bytes()).hexdigest())
        for path in snapshot.iterdir()
    }
    assert after == before


def test_production_snapshot_plot_adapters_select_cached_synchrony_and_spike_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production adapters forward interactive cached slices to established plotters.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces existing pure plotting functions to capture only their selected
        cached arrays; no raw source, numerical preparation, or cache write occurs.
    """

    captured: dict[str, tuple[object, ...]] = {}

    def fake_phase_map(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture one Synchrony cached slice and return an unsaved figure.

        Parameters are the established plotting-module arguments: dimensionless
        metric/count arrays, Hz/seconds coordinates, categorical labels, and
        immutable plot context. No input arrays are transformed.
        """

        del kwargs
        captured["synchrony"] = args
        return plt.figure(), {}

    def fake_unit_ppc_map(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture one Spike cached slice and return an unsaved figure.

        Parameters are the established plotting-module arrays with unit and Hz
        axes plus categorical selection labels and plot context. No numerical
        computation or file I/O occurs.
        """

        del kwargs
        captured["spike_phase"] = args
        return plt.figure(), {}

    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_plotting,
        "plot_phase_map",
        fake_phase_map,
    )
    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_plotting,
        "plot_unit_ppc_map",
        fake_unit_ppc_map,
    )
    selection = lfp_summary_webapp.SnapshotPlotSelection(
        view="unit_ppc_map",
        site_id="HPC1",
        condition_name="left",
        epoch_name="after",
        band_name="gamma",
    )
    synchrony_arrays = {
        "condition_names": np.array(("all", "left")),
        "site_ids": np.array(("PFC", "HPC1")),
        "frequency_hz": np.array((8.0,)),
        "relative_time_s": np.array((0.0,)),
        "condition_membership": np.array(((True, False), (True, True))),
        "filter_membership": np.array((True, True)),
        "itpc": np.array([[[[1.0]], [[2.0]]], [[[3.0]], [[7.0]]]]),
        "itpc_effective_trial_count": np.array([[[[1]], [[1]]], [[[2]], [[2]]]]),
    }
    spike_band_ppc = np.full((1, 2, 2, 2, 2), np.nan)
    spike_band_ppc[0, 1, 1, 1, 1] = 29.0
    spike_ppc = np.full((1, 2, 2, 2, 1), np.nan)
    spike_ppc[0, 1, 1, 1, 0] = 11.0
    spike_arrays = {
        "unit_ids": np.array(("ProbeB:7",)),
        "condition_names": np.array(("all", "left")),
        "site_ids": np.array(("PFC", "HPC1")),
        "epoch_names": np.array(("before", "after")),
        "band_names": np.array(("theta", "gamma")),
        "frequency_hz": np.array((8.0,)),
        "ppc": spike_ppc,
        "computable": np.ones((1, 2, 2, 2, 1), dtype=bool),
        "reliable": np.ones((1, 2, 2, 2, 1), dtype=bool),
        "spike_count": np.ones((1, 2, 2, 2, 1), dtype=np.int64),
        "ppc_band_mean": spike_band_ppc,
    }

    spike_slice = lfp_summary_webapp.select_spike_snapshot_slice(
        spike_arrays,
        selection,
    )

    synchrony_figure = lfp_summary_webapp.plot_cached_snapshot_component(
        "synchrony",
        synchrony_arrays,
        default_lfp_summary_config(),
        selection,
    )
    spike_figure = lfp_summary_webapp.plot_cached_snapshot_component(
        "spike_phase",
        spike_arrays,
        default_lfp_summary_config(),
        selection,
    )

    assert np.array_equal(captured["synchrony"][0], np.array([[7.0]]))
    assert captured["synchrony"][5:7] == ("HPC1", "left")
    assert np.array_equal(captured["spike_phase"][0], np.array([[11.0]]))
    assert captured["spike_phase"][5:8] == (("ProbeB:7",), "left", "HPC1")
    assert captured["spike_phase"][8] == "after"
    assert spike_slice.view == "unit_ppc_map"
    assert spike_slice.condition_index == 1
    assert spike_slice.site_index == 1
    assert spike_slice.epoch_index == 1
    assert spike_slice.band_index == 1
    plt.close(synchrony_figure)
    plt.close(spike_figure)


def test_probe_a_selection_reports_final_probe_b_spike_cache_mismatch(
    tmp_path: Path,
) -> None:
    """A ProbeA view cannot relabel or plot a committed ProbeB Spike component.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root containing a synthetic final Spike component labeled ProbeB.
    """

    snapshot = _write_snapshot_fixture(
        tmp_path,
        component_population_labels={"spike_phase": "ProbeB"},
    )
    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)
    probe_a = UnitPopulationConfig(
        "ProbeA active",
        "ProbeA",
        tmp_path / "probe_a_sorter",
        tmp_path / "probe_a_aligned.npz",
        (1,),
        (("channel_quality", "good"), ("inside_brain", "true"), ("unit_quality", "good,mua")),
        ("ProbeA:11",),
    )

    status = lfp_summary_webapp.snapshot_component_population_status(
        inspection,
        "spike_phase",
        probe_a,
    )

    assert status.state == "population_mismatch"
    assert "ProbeB" in status.message
    assert not status.can_plot


def test_progress_rendering_and_launcher_handoff_do_not_compute_numerics() -> None:
    """Progress is display-only while Spike phase and Compute All remain launcher handoffs."""

    streamlit = FakeStreamlit()
    compute_calls: list[str] = []
    event = lfp_summary_webapp.ProgressEvent(
        "spike_phase",
        "shuffle",
        7,
        10,
        "running shuffled nulls",
        elapsed_seconds=12.0,
        eta_seconds=5.0,
    )

    lfp_summary_webapp.render_progress_event(streamlit, event)
    config = replace(
        default_lfp_summary_config(),
        unit_population=UnitPopulationConfig(
            "ProbeA active",
            "ProbeA",
            Path("/explicit/probe-a/kilosort4"),
            Path("/explicit/probe-a/aligned_spikes.npz"),
            (1,),
            (
                ("channel_quality", "good"),
                ("inside_brain", "true"),
                ("unit_quality", "good,mua"),
            ),
            ("ProbeA:7",),
        ),
    )
    new_run_commands = lfp_summary_webapp.build_launcher_handoff_commands(
        config,
        action="spike_phase",
    )
    assert set(new_run_commands) == {"local_new", "slurm_new"}
    assert all(isinstance(command, str) and command for command in new_run_commands.values())
    resume_run_directory = Path("/explicit/saved/spike-phase-run")
    commands = lfp_summary_webapp.build_launcher_handoff_commands(
        config,
        action="spike_phase",
        resume_run_directory=resume_run_directory,
    )
    assert set(commands) == {"local_new", "slurm_new", "local_resume", "slurm_resume"}
    assert all(isinstance(command, str) and command for command in commands.values())
    assert str(resume_run_directory) in commands["local_resume"]
    assert str(resume_run_directory) in commands["slurm_resume"]
    assert "latest" not in commands["local_resume"].lower()
    assert "latest" not in commands["slurm_resume"].lower()
    for action in ("spike_phase", "all"):
        handoff = lfp_summary_webapp.dispatch_summary_action(
            action,
            source=lfp_summary_webapp.resolve_summary_source(
                source_mode="live",
                entered_snapshot_directory="",
                session_state={},
            ),
            config=default_lfp_summary_config(),
            dependencies=_fake_dependencies(compute_calls, []),
            launcher_command_builder=_synthetic_slurm_command,
        )
        assert handoff.state == "handoff"
        assert handoff.launcher_command == f"sbatch {action}"

    assert compute_calls == []
    progress_text = " ".join(message for _kind, message in streamlit.messages)
    assert "7" in progress_text and "10" in progress_text and "ETA" in progress_text


def test_nonempty_absolute_amplitude_threshold_is_rejected_before_any_action_dispatch() -> None:
    """Deferred absolute masking must not appear applied or reach a compute callback."""

    calls: list[str] = []
    config = replace(
        default_lfp_summary_config(),
        phase=replace(
            default_lfp_summary_config().phase,
            absolute_amplitude_thresholds=(("PFC", 12.5),),
        ),
    )
    source = lfp_summary_webapp.resolve_summary_source(
        source_mode="live",
        entered_snapshot_directory="",
        session_state={},
    )

    result = lfp_summary_webapp.dispatch_summary_action(
        "power",
        source=source,
        config=config,
        dependencies=_fake_dependencies(calls, []),
        launcher_command_builder=_synthetic_launcher_command,
    )

    assert result.state == "blocked"
    assert "absolute amplitude" in (result.error or "").lower()
    assert calls == []


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


def test_live_power_and_synchrony_use_only_the_composed_runtime_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Live bounded actions share WP10's composed dependency factory, not ad hoc seams.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the composed runtime factory and two pipeline entry points without
        reading data, preparing phase, or writing a cache.
    """

    calls: list[str] = []
    composed_dependencies = object()

    def fake_composed_factory(**kwargs: object) -> object:
        """Record composed-boundary construction and return an opaque bundle."""

        assert callable(kwargs["trial_table_loader"])
        calls.append("compose")
        return composed_dependencies

    def fake_component(component: str) -> Callable[[object, object], object]:
        """Return a fake pipeline entry point that records one component identity."""

        def compute(config: object, dependencies: object) -> object:
            """Record exact composed inputs and return a successful result-like object."""

            assert config == default_lfp_summary_config()
            assert dependencies is composed_dependencies
            calls.append(component)
            return type(
                "Result",
                (),
                {"component": component, "state": "complete", "error": None},
            )()

        return compute

    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_runtime,
        "make_lfp_summary_pipeline_dependencies",
        fake_composed_factory,
    )
    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_pipeline,
        "compute_power_component",
        fake_component("power"),
    )
    monkeypatch.setattr(
        lfp_summary_webapp.lfp_summary_pipeline,
        "compute_synchrony_component",
        fake_component("synchrony"),
    )

    dependencies = lfp_summary_webapp.make_production_summary_dependencies()
    power = dependencies.compute_power(default_lfp_summary_config())
    synchrony = dependencies.compute_synchrony(default_lfp_summary_config())

    assert power.state == "complete"
    assert synchrony.state == "complete"
    assert calls == ["compose", "power", "synchrony"]


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
        "condition_names": np.array(
            ("correct_rewarded", "omission", "incorrect", "switch", "stay")
        ),
        "condition_membership": np.array(
            (
                (True, False, False, False, False),
                (True, True, False, False, False),
            )
        ),
        "filter_membership": np.array((True, False)),
        "condition_effective_trial_count": np.array(
            ((1,), (1,), (0,), (0,), (0,))
        ),
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


def test_snapshot_component_cache_reuses_inspection_identities_without_rehash_or_reopen(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A retained inspection must make identical component rerenders memory-only.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the private SHA-256 helper after receipt validation. A cache hit
        must not call it or reopen the tiny selected NPZ.
    tmp_path : pathlib.Path
        Temporary root containing only a synthetic final snapshot and no raw
        LFP, spike, network, or CT026 data.
    """

    inspection = lfp_summary_webapp.validate_cache_snapshot(_write_snapshot_fixture(tmp_path))
    cache = lfp_summary_webapp.SnapshotComponentCache()
    opened: list[Path] = []

    def load_component(path: Path, manifest: dict[str, object], component: str) -> dict[str, np.ndarray]:
        """Open one selected synthetic NPZ and return its dimensionless marker."""

        del manifest, component
        opened.append(path)
        with np.load(path) as cached:
            return {"marker": cached["marker"].copy()}

    def fail_rehash(path: Path) -> str:
        """Fail when a post-inspection rerender attempts a file-content hash."""

        raise AssertionError(f"component cache rehashed {path}")

    monkeypatch.setattr(lfp_summary_webapp, "_sha256_file", fail_rehash)

    first = cache.load(inspection, "synchrony", load_component)
    second = cache.load(inspection, "synchrony", load_component)

    assert first is second
    assert opened == [inspection.component_paths["synchrony"]]  # type: ignore[index]


def test_production_live_loader_passes_the_exact_selected_component_npz_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Live cache loading must give the IO boundary ``output_directory/component.npz``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the production array loader and records its filesystem input
        without opening a real cache or source recording.
    tmp_path : pathlib.Path
        Temporary output root used only to form the exact expected NPZ path.
    """

    config = replace(default_lfp_summary_config(), output_directory=tmp_path / "cache")
    paths: list[Path] = []

    def load_component_arrays(path: Path, manifest: dict[str, object], component: str) -> dict[str, np.ndarray]:
        """Record the selected component file and return one marker array."""

        del manifest, component
        paths.append(path)
        return {"marker": np.array((1,), dtype=np.int64)}

    monkeypatch.setattr(lfp_summary_webapp, "load_component_arrays", load_component_arrays)
    dependencies = lfp_summary_webapp.make_production_summary_dependencies()

    arrays = lfp_summary_webapp.load_summary_view_arrays(
        "synchrony",
        config,
        {"components": {"synchrony": {}}},
        dependencies,
    )

    assert paths == [config.output_directory / "synchrony.npz"]
    assert arrays["synchrony"]["marker"].tolist() == [1]


def test_spike_snapshot_population_provenance_fails_closed_but_nonspike_components_remain_irrelevant(
    tmp_path: Path,
) -> None:
    """Only a complete valid Spike population provenance can authorize Spike plots.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root for two receipt-coherent synthetic snapshots. Manifest
        mutations contain categorical provenance only, not scientific arrays.
    """

    population = UnitPopulationConfig(
        "ProbeB active",
        "ProbeB",
        tmp_path / "sorter",
        tmp_path / "aligned.npz",
        (1,),
        (("channel_quality", "good"), ("inside_brain", "true"), ("unit_quality", "good,mua")),
        ("ProbeB:7",),
    )
    missing_root = tmp_path / "missing"
    malformed_root = tmp_path / "malformed"
    missing_root.mkdir()
    malformed_root.mkdir()
    missing_snapshot = _write_snapshot_fixture(missing_root)
    malformed_snapshot = _write_snapshot_fixture(malformed_root)

    def make_probe_malformed(manifest: dict[str, object]) -> None:
        """Replace only Spike population provenance with an invalid probe label."""

        components = manifest["components"]
        assert isinstance(components, dict)
        spike = components["spike_phase"]
        assert isinstance(spike, dict)
        spike["configuration_snapshot"] = {"unit_population": {"probe_label": "ProbeZ"}}

    _rewrite_snapshot_manifest(malformed_snapshot, make_probe_malformed)
    missing_inspection = lfp_summary_webapp.validate_cache_snapshot(missing_snapshot)
    malformed_inspection = lfp_summary_webapp.validate_cache_snapshot(malformed_snapshot)

    for inspection in (missing_inspection, malformed_inspection):
        spike = lfp_summary_webapp.snapshot_component_population_status(
            inspection,
            "spike_phase",
            population,
        )
        assert not spike.can_plot
        assert "population" in spike.message.lower() or "provenance" in spike.message.lower()
    for component in ("power", "synchrony"):
        status = lfp_summary_webapp.snapshot_component_population_status(
            missing_inspection,
            component,
            population,
        )
        assert status.can_plot


def test_active_population_ignores_nonnumeric_channel_ids_and_keeps_exact_quality_selection(
    tmp_path: Path,
) -> None:
    """One probe keeps only good/MUA units on good inside-brain numeric channels.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root used only for explicit sorter/aligned path identities.
        Metadata frames are synthetic categorical labels and zero-based channels.
    """

    clusters = pd.DataFrame(
        {
            "cluster_id": (9, 2, 3, 4),
            "ch": (1, 2, 3, 2),
            "group": ("mua", "good", "noise", "good"),
        }
    )
    channels = pd.DataFrame(
        {
            "channel_id": ("CH001", "not-a-channel", "CH003", "CH002"),
            "label": ("GOOD", "good", "bad", "good"),
            "inside_brain": (True, True, True, False),
        }
    )

    population = lfp_summary_webapp.build_active_summary_population(
        probe_label="ProbeB",
        sorter_path=tmp_path / "explicit_sorter",
        aligned_spike_path=tmp_path / "explicit_aligned.npz",
        cluster_metadata=clusters,
        channel_metadata=channels,
    )

    assert population.selected_channels == (1,)
    assert population.stable_unit_ids == ("ProbeB:9",)
    assert population.sorter_path == tmp_path / "explicit_sorter"
    assert population.aligned_spike_path == tmp_path / "explicit_aligned.npz"


def test_snapshot_rejects_symlinked_final_entries_even_when_bytes_match(tmp_path: Path) -> None:
    """Receipt-valid bytes must still be regular immediate snapshot children.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root holding a small copied component outside the entered
        snapshot and a symlink with matching bytes. No source/session data is
        accessed.
    """

    snapshot = _write_snapshot_fixture(tmp_path)
    component = snapshot / "power.npz"
    copied_component = tmp_path / "outside_power.npz"
    copied_component.write_bytes(component.read_bytes())
    component.unlink()
    component.symlink_to(copied_component)

    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)

    assert not inspection.is_scientific_result
    assert inspection.state in {"malformed", "incomplete"}
    assert "symlink" in inspection.message.lower()


def test_snapshot_plot_uses_selected_saved_configuration_and_cluster_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Immutable snapshot figures must not inherit labels from local live configuration.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces the established ITPC plotter and captures only selected cached
        arrays and the immutable plot context.
    tmp_path : pathlib.Path
        Temporary root used for one complete saved configuration and tiny
        Synchrony arrays; no CT026 or raw source file is read.
    """

    saved = _full_saved_component_configuration(tmp_path)
    snapshot = _write_snapshot_fixture(
        tmp_path,
        component_configuration_snapshots={"synchrony": saved},
    )
    inspection = lfp_summary_webapp.validate_cache_snapshot(snapshot)
    captured: dict[str, object] = {}

    def plot_phase_map(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Capture one cache-only ITPC plot request and return an unsaved figure."""

        del kwargs
        captured["context"] = args[-1]
        return plt.figure(), {}

    monkeypatch.setattr(lfp_summary_webapp.lfp_summary_plotting, "plot_phase_map", plot_phase_map)
    figure = lfp_summary_webapp.plot_cached_snapshot_component(
        "synchrony",
        _small_synchrony_snapshot_arrays(),
        replace(default_lfp_summary_config(), session_id="local-live-drift"),
        lfp_summary_webapp.SnapshotPlotSelection(
            view="itpc_map",
            site_id="HPC1",
            condition_name="left",
            epoch_name="after",
            band_name="gamma",
        ),
        inspection=inspection,
    )

    context = captured["context"]
    assert context.session_id == "cluster-produced-session"
    assert context.alignment_event == saved["analysis_windows"]["alignment_event"]
    assert inspection.source_cluster_directory in context.reference_description
    plt.close(figure)


@pytest.mark.parametrize(
    ("component", "view", "plotter_name", "site_or_pair", "required_selectors"),
    (
        ("power", "condition_psd", "plot_condition_psd", "HPC1", ("HPC1", "after")),
        ("power", "band_power_summary", "plot_band_power_summary", "HPC1", ("HPC1",)),
        ("synchrony", "itpc_map", "plot_phase_map", "HPC1", ("HPC1", "left")),
        ("synchrony", "ispc_map", "plot_phase_map", "PFC-HPC1", ("PFC-HPC1", "left")),
        ("synchrony", "itpc_band_summary", "plot_phase_band_summary", "HPC1", ("HPC1", "left", "after", "gamma")),
        ("synchrony", "ispc_band_summary", "plot_phase_band_summary", "PFC-HPC1", ("PFC-HPC1", "left", "after", "gamma")),
        ("synchrony", "plv_distribution", "plot_plv_distribution", "PFC-HPC1", ("PFC-HPC1", "left", "gamma")),
        ("synchrony", "plv_exemplar", "plot_plv_exemplar", "PFC-HPC1", ("PFC-HPC1",)),
        ("spike_phase", "unit_ppc_map", "plot_unit_ppc_map", "HPC1", ("HPC1", "left", "after")),
        ("spike_phase", "population_ppc_maps", "plot_population_ppc_maps", "HPC1", ("HPC1", "after")),
        ("spike_phase", "ppc_band_summary", "plot_ppc_band_summary", "HPC1", ("HPC1",)),
        ("spike_phase", "ppc_exemplar_pair", "plot_ppc_exemplar_pair", "HPC1", ("HPC1", "left", "after", "gamma")),
    ),
)
def test_each_authoritative_snapshot_view_delegates_its_selected_cached_axes(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
    view: str,
    plotter_name: str,
    site_or_pair: str,
    required_selectors: tuple[str, ...],
) -> None:
    """Every approved cached view must select arrays then delegate to its plotter.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Replaces all existing pure plotters. The selected plotter returns one
        unsaved figure; any different plotter call is an adapter-routing error.
    component, view, plotter_name, site_or_pair : str
        Parametrized committed component, cached-view family, existing plotting
        function name, and relevant site or site-pair selector. They are
        categorical labels with no numerical units.
    required_selectors : tuple[str, ...]
        Exact categorical selectors relevant to this view family. Omitted
        selectors must not be inferred from the view-name text.
    """

    arrays = {
        "power": _small_power_snapshot_arrays,
        "synchrony": _small_synchrony_snapshot_arrays,
        "spike_phase": _small_spike_snapshot_arrays,
    }[component]()
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def selected_plotter(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Record the delegated cache-only arguments and return one blank figure."""

        calls.append((plotter_name, args, kwargs))
        return plt.figure(), {}

    def unexpected_plotter(*args: object, **kwargs: object) -> tuple[plt.Figure, dict[str, object]]:
        """Fail when an adapter routes a cached view to the wrong plot family."""

        del args, kwargs
        raise AssertionError("cached view delegated to an unrelated plotting function")

    for candidate in (
        "plot_condition_psd",
        "plot_band_power_summary",
        "plot_phase_map",
        "plot_phase_band_summary",
        "plot_plv_distribution",
        "plot_plv_exemplar",
        "plot_unit_ppc_map",
        "plot_population_ppc_maps",
        "plot_ppc_band_summary",
        "plot_ppc_exemplar_pair",
    ):
        monkeypatch.setattr(
            lfp_summary_webapp.lfp_summary_plotting,
            candidate,
            selected_plotter if candidate == plotter_name else unexpected_plotter,
        )
    selection = lfp_summary_webapp.SnapshotPlotSelection(
        view=view,
        site_id=site_or_pair,
        condition_name="left",
        epoch_name="after",
        band_name="gamma",
    )

    figure = lfp_summary_webapp.plot_cached_snapshot_component(
        component,
        arrays,
        default_lfp_summary_config(),
        selection,
    )

    assert [call[0] for call in calls] == [plotter_name]
    delegated_text = repr(calls[0][1]) + repr(calls[0][2])
    for selector in required_selectors:
        assert selector in delegated_text
    plt.close(figure)


@pytest.mark.parametrize("snapshot_text", ("", "{invalid}"))
def test_child_renderer_defaults_to_noncomputational_snapshot_and_defers_population_metadata(
    tmp_path: Path,
    snapshot_text: str,
) -> None:
    """Blank/invalid snapshot input must not load metadata, cache, raw data, or compute.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty local root used only to construct explicit missing paths and probe
        path identities. It contains no snapshot, CT026, raw LFP, or network data.
    snapshot_text : str
        Blank widget text or the ``"{invalid}"`` placeholder replaced with one
        explicit absent local directory. It is never treated as a live fallback.
    """

    entered = "" if not snapshot_text else str(tmp_path / "absent_snapshot")
    streamlit = RouteFakeStreamlit(source_mode="snapshot", snapshot_text=entered)
    calls: list[str] = []
    metadata_paths: list[Path] = []

    def cluster_metadata_loader(sorter_path: Path) -> pd.DataFrame:
        """Fail if a blank/invalid snapshot triggers selected-probe metadata I/O."""

        metadata_paths.append(sorter_path)
        raise AssertionError("blank or invalid snapshot loaded cluster metadata")

    def channel_metadata_loader(sorter_path: Path) -> pd.DataFrame:
        """Fail if a blank/invalid snapshot triggers selected-probe channel I/O."""

        metadata_paths.append(sorter_path)
        raise AssertionError("blank or invalid snapshot loaded channel metadata")

    lfp_summary_webapp.render_lfp_summary_view(
        streamlit,
        session_id="synthetic-session",
        session_path=tmp_path,
        output_directory=tmp_path / "live_cache",
        sites=_summary_sites(tmp_path),
        site_pairs=(("PFC", "HPC1"),),
        sorter_paths={"ProbeA": tmp_path / "sorter_a", "ProbeB": tmp_path / "sorter_b"},
        aligned_spike_paths={"ProbeA": tmp_path / "aligned_a.npz", "ProbeB": tmp_path / "aligned_b.npz"},
        cluster_metadata_loader=cluster_metadata_loader,
        channel_metadata_loader=channel_metadata_loader,
        dependencies=_fake_dependencies(calls, []),
    )

    assert metadata_paths == []
    assert calls == []
    assert streamlit.sidebar.button_labels == []
    message_text = " ".join(message for _kind, message in streamlit.messages).lower()
    assert "snapshot" in message_text


def test_child_renderer_reuses_valid_snapshot_inspection_and_selected_probe_cache_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An unchanged snapshot rerender must reuse inspection, load ProbeB once, and display provenance.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Wraps validation and replaces only the cache-only plotting adapter. No
        raw source, numerical pipeline, or component write seam is enabled.
    tmp_path : pathlib.Path
        Temporary root for one receipt-valid final snapshot, explicit ProbeA/B
        paths, and tiny cached Spike arrays with count/eligibility metadata.
    """

    saved = _full_saved_component_configuration(tmp_path)
    snapshot = _write_snapshot_fixture(
        tmp_path,
        component_configuration_snapshots={"spike_phase": saved},
    )

    def add_spike_warning(manifest: dict[str, object]) -> None:
        """Add one selected-component warning without changing cache arrays."""

        components = manifest["components"]
        assert isinstance(components, dict)
        spike = components["spike_phase"]
        assert isinstance(spike, dict)
        spike["warnings"] = ["synthetic instability warning"]

    _rewrite_snapshot_manifest(snapshot, add_spike_warning)
    streamlit = RouteFakeStreamlit(
        source_mode="snapshot",
        snapshot_text=str(snapshot),
        view="spike_phase",
    )
    dependency_calls: list[str] = []
    metadata_paths: list[Path] = []
    plot_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    validation_calls: list[Path | str] = []
    original_validate = lfp_summary_webapp.validate_cache_snapshot

    def validate_snapshot(path: Path | str) -> lfp_summary_webapp.SnapshotInspection:
        """Record validation entry while preserving receipt validation behavior."""

        validation_calls.append(path)
        return original_validate(path)

    def cluster_metadata_loader(sorter_path: Path) -> pd.DataFrame:
        """Return selected-ProbeB cluster metadata and record its exact sorter path."""

        metadata_paths.append(sorter_path)
        return pd.DataFrame({"cluster_id": (7,), "ch": (1,), "group": ("good",)})

    def channel_metadata_loader(sorter_path: Path) -> pd.DataFrame:
        """Return selected-ProbeB good inside-brain metadata for one channel."""

        metadata_paths.append(sorter_path)
        return pd.DataFrame({"channel": (1,), "channel_quality": ("good",), "inside_brain": (True,)})

    def plot_cached(*args: object, **kwargs: object) -> plt.Figure:
        """Record one cache-only Spike adapter request and return an unsaved figure."""

        plot_calls.append((args, kwargs))
        return plt.figure()

    def load_component(path: Path, manifest: dict[str, object], component: str) -> dict[str, np.ndarray]:
        """Return only selected tiny Spike arrays without opening a raw source file."""

        del path, manifest
        dependency_calls.append(f"load_{component}")
        return _small_spike_snapshot_arrays()

    dependencies = replace(_fake_dependencies(dependency_calls, []), load_component=load_component)
    monkeypatch.setattr(lfp_summary_webapp, "validate_cache_snapshot", validate_snapshot)
    monkeypatch.setattr(lfp_summary_webapp, "plot_cached_snapshot_component", plot_cached)
    route_kwargs = {
        "session_id": "synthetic-session",
        "session_path": tmp_path,
        "output_directory": tmp_path / "live_cache",
        "sites": _summary_sites(tmp_path),
        "site_pairs": (("PFC", "HPC1"),),
        "sorter_paths": {"ProbeA": tmp_path / "sorter_a", "ProbeB": tmp_path / "sorter_b"},
        "aligned_spike_paths": {"ProbeA": tmp_path / "aligned_a.npz", "ProbeB": tmp_path / "aligned_b.npz"},
        "cluster_metadata_loader": cluster_metadata_loader,
        "channel_metadata_loader": channel_metadata_loader,
        "dependencies": dependencies,
    }

    lfp_summary_webapp.render_lfp_summary_view(streamlit, **route_kwargs)
    lfp_summary_webapp.render_lfp_summary_view(streamlit, **route_kwargs)
    _change_snapshot_component_identity(snapshot, "spike_phase")
    lfp_summary_webapp.render_lfp_summary_view(streamlit, **route_kwargs)

    assert validation_calls == [snapshot, snapshot]
    assert metadata_paths == [tmp_path / "sorter_b", tmp_path / "sorter_b"]
    assert dependency_calls == ["load_spike_phase", "load_spike_phase"]
    assert len(plot_calls) == 3
    display_text = " ".join(message for _kind, message in streamlit.messages)
    assert "/cluster/final/cache" in display_text
    assert "synthetic instability warning" in display_text
    assert "ProbeB" in display_text
    assert "7" in display_text
    assert "eligible" in display_text.lower()
    assert "unstable" in display_text.lower()


@pytest.mark.parametrize(
    ("action", "expected_compute"),
    (("power", "compute_power"), ("synchrony", "compute_synchrony"), ("spike_phase", None), ("all", None)),
)
def test_child_renderer_live_mode_limits_compute_to_power_synchrony_and_handoffs_spike_actions(
    tmp_path: Path,
    action: str,
    expected_compute: str | None,
) -> None:
    """Explicit live mode exposes bounded actions or launcher text, never synchronous Spike work.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary root used only for explicit selected-probe paths and a live
        cache path. No data file is created or opened.
    action : str
        Requested categorical Power, Synchrony, Spike-phase, or Compute-All UI
        action.
    expected_compute : str or None
        The sole bounded injected compute call expected for Power/Synchrony;
        ``None`` requires a launcher-only result for Spike/All.
    """

    streamlit = RouteFakeStreamlit(source_mode="live", snapshot_text="", action=action, view="power")
    calls: list[str] = []

    def cluster_metadata_loader(sorter_path: Path) -> pd.DataFrame:
        """Return one selected-probe unit without touching an unselected sorter."""

        assert sorter_path == tmp_path / "sorter_b"
        return pd.DataFrame({"cluster_id": (7,), "ch": (1,), "group": ("good",)})

    def channel_metadata_loader(sorter_path: Path) -> pd.DataFrame:
        """Return one selected-probe good inside-brain channel."""

        assert sorter_path == tmp_path / "sorter_b"
        return pd.DataFrame({"channel": (1,), "channel_quality": ("good",), "inside_brain": (True,)})

    lfp_summary_webapp.render_lfp_summary_view(
        streamlit,
        session_id="synthetic-session",
        session_path=tmp_path,
        output_directory=tmp_path / "live_cache",
        sites=_summary_sites(tmp_path),
        site_pairs=(("PFC", "HPC1"),),
        sorter_paths={"ProbeA": tmp_path / "sorter_a", "ProbeB": tmp_path / "sorter_b"},
        aligned_spike_paths={"ProbeA": tmp_path / "aligned_a.npz", "ProbeB": tmp_path / "aligned_b.npz"},
        cluster_metadata_loader=cluster_metadata_loader,
        channel_metadata_loader=channel_metadata_loader,
        dependencies=_fake_dependencies(calls, []),
    )

    compute_calls = [call for call in calls if call.startswith("compute_")]
    if expected_compute is None:
        assert compute_calls == []
        display_text = " ".join(message for _kind, message in streamlit.messages).lower()
        assert "launcher" in display_text or "sbatch" in display_text or "uv run" in display_text
    else:
        assert compute_calls == [expected_compute]
