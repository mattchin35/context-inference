"""Streamlit-facing controls for cached LFP summary inspection.

Numerical preparation and computation remain injected pipeline dependencies.
This module only assembles immutable configuration, dispatches requested actions,
loads cached component arrays, and renders already-created Matplotlib figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from shlex import quote
import stat
from typing import Callable, Mapping, MutableMapping, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis import (
    lfp_summary_pipeline,
    lfp_summary_plotting,
    lfp_summary_runtime,
)
from src.neural_analysis.lfp_summary_io import (
    ComponentStatus,
    assess_component_status,
    load_component_arrays,
)
from src.neural_analysis.lfp_summary_models import (
    LFPSiteConfig,
    LFPSummaryConfig,
    TrialFilterConfig,
    UnitPopulationConfig,
    ProgressEvent,
    default_lfp_summary_config,
    lfp_summary_config_from_json,
    validate_lfp_summary_config,
)


SUMMARY_COMPONENTS = ("power", "synchrony", "spike_phase")
SUMMARY_ACTIONS = (*SUMMARY_COMPONENTS, "all")
SUMMARY_VIEWS = SUMMARY_COMPONENTS


class StreamlitOutput(Protocol):
    """Small Streamlit output protocol used by renderer helpers."""

    def info(self, message: str) -> None:
        """Display informational text."""

    def warning(self, message: str) -> None:
        """Display warning text."""

    def error(self, message: str) -> None:
        """Display error text."""

    def pyplot(self, figure: plt.Figure) -> None:
        """Display a Matplotlib figure."""


@dataclass(frozen=True)
class SummaryUIValues:
    """Approved summary-control defaults without raw recordings or UI state.

    Site channels are zero-based saved-channel indices. Frequencies are Hz,
    output rate is samples/second, and bootstrap count is resample count.
    """

    session_id: str
    site_channels: dict[str, int]
    alignment_event: str
    output_rate_hz: float
    notch_enabled: bool
    bootstrap_count: int
    choice_filter: str
    context_filter: str


@dataclass(frozen=True)
class SummaryActionResult:
    """UI-level compute outcome with optional current cache manifest.

    ``component`` is one summary component or ``"all"``. ``state`` is
    ``"complete"``, ``"failed"``, ``"blocked"``, or ``"handoff"``.
    Blocked results are explicit snapshot/threshold gates; handoff results carry
    copyable launcher text and have no manifest. The manifest contains
    JSON-compatible cache metadata only; numerical arrays remain in component
    NPZ files.
    """

    component: str
    state: str
    manifest: dict[str, object] | None
    error: str | None
    launcher_command: str | None = None


@dataclass(frozen=True)
class SnapshotInspection:
    """Read-only local snapshot validation state and immutable provenance.

    ``manifest`` is JSON metadata and ``component_paths`` point only inside the
    entered directory. ``file_digests`` are the receipt-validated SHA-256
    identities for the manifest and final component files. ``stat_identity``
    is the cheap ``(name, device, inode, size, mtime_ns)`` identity for every
    immediate final entry, used only to decide whether a retained inspection
    is still current. Component arrays retain their cached axes and units; this
    object contains no numerical arrays. ``state`` is one of ``valid``,
    ``missing``, ``malformed``, ``tampered``, or ``incomplete``.
    """

    state: str
    message: str
    snapshot_directory: Path | None
    manifest: dict[str, object] | None = None
    component_paths: dict[str, Path] | None = None
    source_cluster_directory: str | None = None
    file_digests: dict[str, str] | None = None
    stat_identity: tuple[tuple[str, int, int, int, int], ...] | None = None

    @property
    def is_scientific_result(self) -> bool:
        """Return whether all final snapshot metadata is committed and valid."""

        return self.state == "valid"


@dataclass(frozen=True)
class SummarySource:
    """Selected source-mode state without implicit cache discovery or compute.

    ``mode`` is ``snapshot`` or ``live``. ``snapshot_directory`` is the exact
    resolved user input or ``None``; no session-directory inference occurs.
    ``can_compute`` is true only for explicit live mode and has no units.
    """

    mode: str
    snapshot_directory: Path | None
    inspection: SnapshotInspection | None
    can_compute: bool
    message: str


@dataclass(frozen=True)
class SnapshotPopulationStatus:
    """Compatibility of one selected population with a cached component."""

    state: str
    message: str
    can_plot: bool


@dataclass(frozen=True)
class SnapshotPlotSelection:
    """Categorical cached-view selectors; no source data or numerical units."""

    view: str
    site_id: str
    condition_name: str
    epoch_name: str
    band_name: str


@dataclass(frozen=True)
class SpikeSnapshotSlice:
    """Selected named-axis positions in cached Spike arrays.

    Indices refer to ``(condition, site, epoch, band)`` categorical axes in
    the component manifest; no numerical array data is copied or transformed.
    """

    view: str
    condition_index: int
    site_index: int
    epoch_index: int
    band_index: int


ComputeAction = Callable[[LFPSummaryConfig], SummaryActionResult]
LoadManifest = Callable[[Path], dict[str, object]]
LoadComponent = Callable[[Path, dict[str, object], str], dict[str, np.ndarray]]
PlotView = Callable[[str, dict[str, dict[str, np.ndarray]], LFPSummaryConfig], plt.Figure]


@dataclass(frozen=True)
class SummaryWebDependencies:
    """Injected pipeline/cache/plot seams used by the summary webapp.

    Compute functions synchronously run the corresponding injected pipeline
    entry point. Cache loaders receive an exact component NPZ ``Path`` plus a
    JSON manifest and component name, then return validated numeric arrays with
    manifest-defined axes and units. ``plot_view`` consumes only those arrays
    plus immutable configuration metadata and returns an unsaved Matplotlib
    figure.
    """

    compute_power: ComputeAction
    compute_synchrony: ComputeAction
    compute_spike_phase: ComputeAction
    compute_all: ComputeAction
    load_manifest: LoadManifest
    load_component: LoadComponent
    plot_view: PlotView


_SNAPSHOT_FILENAMES = (
    "manifest.json",
    "power.npz",
    "synchrony.npz",
    "spike_phase.npz",
    "cache_snapshot_identity.json",
)
_SNAPSHOT_COMPONENT_FILENAMES = {
    "power": "power.npz",
    "synchrony": "synchrony.npz",
    "spike_phase": "spike_phase.npz",
}
_SNAPSHOT_RECORD_FORMAT = "filename\\tsize_bytes\\tsha256\\n"


def _sha256_file(path: Path) -> str:
    """Return a file's lowercase SHA-256 identity using bounded memory.

    Parameters
    ----------
    path : pathlib.Path
        Existing local file. Bytes are opaque cache or JSON content and are not
        interpreted as arrays, axes, or physical units.

    Returns
    -------
    str
        Lowercase 64-character SHA-256 digest of the exact file bytes.
    """

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_failure(state: str, message: str, directory: Path | None) -> SnapshotInspection:
    """Create one non-scientific snapshot result without filesystem access.

    Parameters
    ----------
    state : str
        Categorical snapshot status with no numerical value or units.
    message : str
        Human-readable validation status text.
    directory : pathlib.Path or None
        Explicit entered directory retained as provenance; it is not resolved
        or searched by this helper.

    Returns
    -------
    SnapshotInspection
        Non-scientific inspection record with no manifest, component paths, or
        numerical arrays.
    """

    return SnapshotInspection(state, message, directory)


def _snapshot_stat_identity(
    directory: Path,
) -> tuple[tuple[str, int, int, int, int], ...] | None:
    """Return cheap identities for exactly the final immediate snapshot files.

    Parameters
    ----------
    directory : pathlib.Path
        Explicit snapshot directory. Only its named immediate children are
        inspected with ``lstat``; no component bytes or source files are read.

    Returns
    -------
    tuple[tuple[str, int, int, int, int], ...] or None
        Ordered ``(filename, device, inode, size_bytes, mtime_ns)`` identities,
        or ``None`` when an entry is missing, symlinked, non-regular, or cannot
        be inspected. Values have no scientific units.
    """

    identities: list[tuple[str, int, int, int, int]] = []
    try:
        for name in _SNAPSHOT_FILENAMES:
            entry = directory / name
            entry_stat = entry.lstat()
            if entry.is_symlink() or not stat.S_ISREG(entry_stat.st_mode):
                return None
            identities.append(
                (name, entry_stat.st_dev, entry_stat.st_ino, entry_stat.st_size, entry_stat.st_mtime_ns)
            )
    except OSError:
        return None
    return tuple(identities)


def validate_cache_snapshot(snapshot_directory: Path | str) -> SnapshotInspection:
    """Validate one exact immutable local final-cache snapshot directory.

    Parameters
    ----------
    snapshot_directory : pathlib.Path or str
        Explicit local ``cache_snapshot`` directory. Only this directory and
        its five named immediate children are read; no session, latest-run, or
        network path is searched. Files contain JSON or opaque NPZ bytes whose
        numerical axes and units remain unchanged.

    Returns
    -------
    SnapshotInspection
        Valid inspection retains the manifest, final component paths, and
        original cluster provenance. Invalid states are reported rather than
        raising so the UI remains noncomputational.
    """

    directory = Path(snapshot_directory)
    if not directory.is_dir():
        return _snapshot_failure("missing", "Snapshot directory is missing.", directory)
    try:
        names = {path.name for path in directory.iterdir()}
    except OSError as error:
        return _snapshot_failure("missing", f"Snapshot directory is unreadable: {error}", directory)
    expected_names = set(_SNAPSHOT_FILENAMES)
    if not set(_SNAPSHOT_FILENAMES).issubset(names):
        return _snapshot_failure("incomplete", "Snapshot lacks a required final file.", directory)
    if names != expected_names:
        return _snapshot_failure("incomplete", "Snapshot contains checkpoint or nonfinal work artifacts.", directory)
    paths = {name: directory / name for name in _SNAPSHOT_FILENAMES}
    stat_identity = _snapshot_stat_identity(directory)
    if stat_identity is None:
        return _snapshot_failure(
            "incomplete",
            "Snapshot final entries must be non-symlink regular files.",
            directory,
        )
    try:
        receipt = json.loads(paths["cache_snapshot_identity.json"].read_text(encoding="ascii"))
        manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        return _snapshot_failure("malformed", f"Snapshot JSON is malformed: {error}", directory)
    if not isinstance(receipt, dict) or not isinstance(manifest, dict):
        return _snapshot_failure("malformed", "Snapshot JSON must contain mappings.", directory)
    if receipt.get("schema_version") != 1 or not isinstance(receipt.get("source_cluster_directory"), str) or not isinstance(receipt.get("copied_at_utc"), str):
        return _snapshot_failure("malformed", "Snapshot receipt schema is invalid.", directory)
    if manifest.get("schema_version") != "1" or not isinstance(manifest.get("components"), dict):
        return _snapshot_failure("malformed", "Snapshot manifest schema is invalid.", directory)
    files = receipt.get("files")
    aggregate = receipt.get("aggregate")
    expected_order = list(_SNAPSHOT_FILENAMES[:-1])
    if not isinstance(files, list) or not isinstance(aggregate, dict):
        return _snapshot_failure("malformed", "Snapshot receipt lacks identity records.", directory)
    if (
        aggregate.get("algorithm") != "sha256"
        or aggregate.get("canonical_record_format") != _SNAPSHOT_RECORD_FORMAT
        or aggregate.get("order") != expected_order
        or not isinstance(aggregate.get("sha256"), str)
        or len(files) != len(expected_order)
    ):
        return _snapshot_failure("malformed", "Snapshot aggregate format is invalid.", directory)
    records: list[str] = []
    file_digests: dict[str, str] = {}
    for expected_name, entry in zip(expected_order, files, strict=True):
        if not isinstance(entry, dict):
            return _snapshot_failure("malformed", "Snapshot file record is invalid.", directory)
        filename = entry.get("filename")
        size_bytes = entry.get("size_bytes")
        expected_digest = entry.get("sha256")
        if (
            filename != expected_name
            or not isinstance(size_bytes, int)
            or size_bytes < 0
            or not isinstance(expected_digest, str)
            or len(expected_digest) != 64
        ):
            return _snapshot_failure("malformed", "Snapshot file record is invalid.", directory)
        path = paths[expected_name]
        try:
            actual_digest = _sha256_file(path)
            actual_size = path.stat().st_size
        except OSError as error:
            return _snapshot_failure("missing", f"Snapshot file is unreadable: {error}", directory)
        if actual_size != size_bytes or actual_digest != expected_digest:
            return _snapshot_failure("tampered", f"Snapshot identity differs for {expected_name}.", directory)
        records.append(f"{filename}\t{size_bytes}\t{expected_digest}\n")
        file_digests[expected_name] = expected_digest
    if sha256("".join(records).encode("ascii")).hexdigest() != aggregate["sha256"]:
        return _snapshot_failure("tampered", "Snapshot aggregate identity differs.", directory)
    components = manifest["components"]
    assert isinstance(components, dict)
    for component in SUMMARY_COMPONENTS:
        entry = components.get(component)
        if not isinstance(entry, dict) or entry.get("state") != "complete":
            return _snapshot_failure("incomplete", f"Snapshot {component} is not committed.", directory)
    component_paths = {
        component: paths[filename]
        for component, filename in _SNAPSHOT_COMPONENT_FILENAMES.items()
    }
    return SnapshotInspection(
        "valid",
        "Valid committed snapshot.",
        directory.resolve(),
        manifest,
        component_paths,
        receipt["source_cluster_directory"],
        file_digests,
        stat_identity,
    )


def resolve_summary_source(
    *,
    source_mode: str,
    entered_snapshot_directory: str,
    session_state: MutableMapping[str, object],
) -> SummarySource:
    """Resolve explicit snapshot or live mode without fallback or discovery.

    Parameters
    ----------
    source_mode : str
        Exact ``"snapshot"`` or ``"live"`` mode selected by the UI.
    entered_snapshot_directory : str
        Current text-widget value. Blank explicitly clears the stored path; an
        uncleared rerender supplies the same nonblank value.
    session_state : mutable mapping
        Streamlit-like UI state storing only the current explicit path string;
        it has no numerical arrays, axes, or units.

    Returns
    -------
    SummarySource
        Source selection with a clear noncomputational blank/invalid snapshot
        state. Live mode is the sole compute-capable source.
    """

    if source_mode not in {"snapshot", "live"}:
        raise ValueError("source_mode must be snapshot or live")
    if source_mode == "live":
        return SummarySource("live", None, None, True, "Live cache mode.")
    text = entered_snapshot_directory.strip()
    state_key = "lfp_summary_snapshot_directory"
    if not text:
        session_state.pop(state_key, None)
        return SummarySource("snapshot", None, None, False, "Snapshot path is blank.")
    session_state[state_key] = text
    inspection = validate_cache_snapshot(Path(text))
    directory = inspection.snapshot_directory
    return SummarySource("snapshot", directory, inspection, False, inspection.message)


class SnapshotComponentCache:
    """One-process cache of selected final component arrays keyed by identity."""

    def __init__(self) -> None:
        """Initialize an empty cache; no file or numerical array is opened.

        Returns
        -------
        None
            Creates only an in-process dictionary keyed by categorical file
            identities. Stored arrays retain loader-defined axes and units.
        """

        self._entries: dict[tuple[str, str, str, str], dict[str, np.ndarray]] = {}

    def load(
        self,
        inspection: SnapshotInspection,
        component: str,
        loader: LoadComponent,
    ) -> dict[str, np.ndarray]:
        """Load one selected component once per manifest/file identity.

        Parameters
        ----------
        inspection : SnapshotInspection
            Valid read-only snapshot metadata and paths.
        component : str
            One selected scientific component. Returned arrays preserve the
            component's manifest-defined shapes, axes, units, and NaNs.
        loader : callable
            Existing cache loader accepting selected NPZ path, manifest, and
            component name; it validates stored array schema without compute.

        Returns
        -------
        dict[str, numpy.ndarray]
            Exact loader result retained only in process memory for this identity.
        """

        if (
            not inspection.is_scientific_result
            or inspection.manifest is None
            or inspection.component_paths is None
            or inspection.snapshot_directory is None
            or inspection.file_digests is None
        ):
            raise ValueError("snapshot is not a valid committed inspection source")
        if component not in SUMMARY_COMPONENTS:
            raise ValueError(f"Unknown summary component: {component!r}")
        component_path = inspection.component_paths[component]
        manifest_digest = inspection.file_digests.get("manifest.json")
        component_digest = inspection.file_digests.get(_SNAPSHOT_COMPONENT_FILENAMES[component])
        if not isinstance(manifest_digest, str) or not isinstance(component_digest, str):
            raise ValueError("snapshot inspection lacks retained file identities")
        key = (str(inspection.snapshot_directory), component, manifest_digest, component_digest)
        if key not in self._entries:
            self._entries[key] = loader(component_path, inspection.manifest, component)
        return self._entries[key]


def build_active_summary_population(
    *,
    probe_label: str,
    sorter_path: Path,
    aligned_spike_path: Path,
    cluster_metadata: object,
    channel_metadata: object,
) -> UnitPopulationConfig:
    """Build exactly one active ProbeA or ProbeB population from supplied paths.

    Parameters
    ----------
    probe_label : str
        Exact categorical ``"ProbeA"`` or ``"ProbeB"`` selection.
    sorter_path, aligned_spike_path : pathlib.Path
        Explicit page-supplied source identities. Paths are preserved verbatim
        and are not inferred from a session or replaced with CT026 defaults.
    cluster_metadata, channel_metadata : pandas.DataFrame-like objects
        Cluster table has ``cluster_id``, ``ch``, and ``group`` columns; channel
        table has either ``channel``/``channel_quality`` or ``channel_id``/``label``
        plus ``inside_brain``. Channel indices are zero based and categorical;
        no spike-time arrays or physical units are transformed.

    Returns
    -------
    UnitPopulationConfig
        Good/MUA clusters on good inside-brain channels, ordered by cluster id,
        with stable ``probe:cluster`` identifiers.
    """

    if probe_label not in {"ProbeA", "ProbeB"}:
        raise ValueError("probe_label must be ProbeA or ProbeB")
    required_clusters = {"cluster_id", "ch", "group"}
    if not hasattr(cluster_metadata, "columns") or not hasattr(channel_metadata, "columns"):
        raise ValueError("population metadata must be tabular")
    if not required_clusters.issubset(cluster_metadata.columns) or "inside_brain" not in channel_metadata.columns:
        raise ValueError("population metadata lacks required quality columns")
    if {"channel", "channel_quality"}.issubset(channel_metadata.columns):
        channel_numbers = channel_metadata["channel"]
        channel_labels = channel_metadata["channel_quality"]
    elif {"channel_id", "label"}.issubset(channel_metadata.columns):
        channel_numbers = channel_metadata["channel_id"].astype(str).str.replace("CH", "", regex=False)
        channel_labels = channel_metadata["label"]
    else:
        raise ValueError("channel metadata lacks channel labels")
    # Coercion follows the metadata loader's established table semantics: a
    # descriptive/non-numeric channel label is not a selected hardware channel.
    numeric_channels = pd.to_numeric(channel_numbers, errors="coerce")
    good_channels = channel_metadata.loc[
        channel_labels.astype(str).str.lower().eq("good")
        & channel_metadata["inside_brain"].astype(bool)
        & numeric_channels.notna()
    ]
    selected_channels = tuple(
        sorted({int(numeric_channels[index]) for index in good_channels.index})
    )
    selected = cluster_metadata.loc[
        cluster_metadata["ch"].isin(selected_channels)
        & cluster_metadata["group"].astype(str).str.lower().isin(("good", "mua"))
    ].sort_values("cluster_id")
    stable_unit_ids = tuple(f"{probe_label}:{int(value)}" for value in selected["cluster_id"])
    return UnitPopulationConfig(
        f"{probe_label} active",
        probe_label,
        Path(sorter_path),
        Path(aligned_spike_path),
        selected_channels,
        (
            ("channel_quality", "good"),
            ("inside_brain", "true"),
            ("unit_quality", "good,mua"),
        ),
        stable_unit_ids,
    )


def snapshot_component_population_status(
    inspection: SnapshotInspection,
    component: str,
    population: UnitPopulationConfig,
) -> SnapshotPopulationStatus:
    """Compare selected population identity with a committed snapshot component.

    Parameters
    ----------
    inspection : SnapshotInspection
        Valid snapshot manifest and provenance; it contains no component arrays.
    component : str
        Cached scientific component identity.
    population : UnitPopulationConfig
        Current one-probe selection with categorical channel and unit identities.

    Returns
    -------
    SnapshotPopulationStatus
        Plot permission for matching/irrelevant components or an explicit
        population mismatch without relabeling cached units.
    """

    if not inspection.is_scientific_result or inspection.manifest is None:
        return SnapshotPopulationStatus("unavailable", "Snapshot is not valid.", False)
    components = inspection.manifest.get("components")
    if not isinstance(components, Mapping):
        return SnapshotPopulationStatus("unavailable", "Snapshot manifest is invalid.", False)
    entry = components.get(component)
    if not isinstance(entry, Mapping):
        return SnapshotPopulationStatus("unavailable", "Snapshot component is unavailable.", False)
    configuration = entry.get("configuration_snapshot")
    if component != "spike_phase":
        return SnapshotPopulationStatus("compatible", "Population is not required for this component.", True)
    cached_population = configuration.get("unit_population") if isinstance(configuration, Mapping) else None
    cached_probe = cached_population.get("probe_label") if isinstance(cached_population, Mapping) else None
    if cached_probe not in {"ProbeA", "ProbeB"}:
        return SnapshotPopulationStatus(
            "population_unavailable",
            "Spike snapshot population provenance is missing or malformed.",
            False,
        )
    if cached_probe != population.probe_label:
        return SnapshotPopulationStatus(
            "population_mismatch",
            f"Spike phase is committed for {cached_probe}, not {population.probe_label}.",
            False,
        )
    return SnapshotPopulationStatus(
        "compatible",
        f"Population is compatible: {population.probe_label}.",
        True,
    )


def dispatch_summary_action(
    action: str,
    *,
    source: SummarySource,
    config: LFPSummaryConfig,
    dependencies: SummaryWebDependencies,
    launcher_command_builder: Callable[[str, LFPSummaryConfig], str],
) -> SummaryActionResult:
    """Gate live actions and return launcher-only Spike/All handoffs.

    Parameters
    ----------
    action : str
        Summary action identity. Power/Synchrony are bounded live actions;
        Spike phase and All are launcher handoffs only.
    source : SummarySource
        Explicit snapshot/live state. Snapshot mode is strictly read-only.
    config : LFPSummaryConfig
        Immutable configuration whose phase threshold units are source-voltage
        units and whose cached numerical axes remain unchanged.
    dependencies : SummaryWebDependencies
        Existing live compute/cache seams used only after live action gating.
    launcher_command_builder : callable
        Returns copyable command text and never executes a process.

    Returns
    -------
    SummaryActionResult
        Blocked, launcher-handoff, or bounded live-action result with no hidden
        fallback from snapshot mode to live computation.
    """

    if config.phase.absolute_amplitude_thresholds:
        return SummaryActionResult(action, "blocked", None, "Absolute amplitude thresholds are unsupported.")
    if source.mode != "live":
        return SummaryActionResult(action, "blocked", None, "Snapshot mode cannot dispatch computation.")
    if action in {"spike_phase", "all"}:
        return SummaryActionResult(action, "handoff", None, None, launcher_command_builder(action, config))
    if action not in {"power", "synchrony"}:
        return SummaryActionResult(action, "blocked", None, "Unknown summary action.")
    return run_summary_action(action, config, dependencies)


def build_launcher_handoff_commands(
    config: LFPSummaryConfig,
    *,
    action: str,
    resume_run_directory: Path | None = None,
) -> dict[str, str]:
    """Build copyable local/Slurm Spike launcher commands without executing them.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable config with exactly one ProbeA/ProbeB population, a PPC shuffle
        count, and worker count. No source arrays are opened.
    action : str
        ``"spike_phase"`` or ``"all"`` handoff category; both use the reviewed
        Spike launcher because Streamlit never runs the long numerical callback.
    resume_run_directory : pathlib.Path or None
        Exact saved launcher run directory. ``None`` deliberately omits resume
        commands rather than discovering or fabricating a latest run.

    Returns
    -------
    dict[str, str]
        Nonempty ``local_new`` and ``slurm_new`` command strings; explicit
        resume input additionally supplies ``local_resume`` and ``slurm_resume``.
    """

    if action not in {"spike_phase", "all"}:
        raise ValueError("launcher handoff is only available for spike_phase or all")
    population = config.unit_population
    if population is None or population.probe_label not in {"ProbeA", "ProbeB"}:
        raise ValueError("launcher new run requires one ProbeA or ProbeB population")
    command = (
        "python -m src.neural_analysis.lfp_spike_phase_launcher new "
        f"--session-path {quote(str(config.session_path))} "
        f"--probe {quote(population.probe_label)} "
        f"--shuffles {config.ppc.shuffle_count} "
        f"--workers {config.ppc_execution.worker_count}"
    )
    if config.ppc.shuffle_count == 1000:
        command += " --final-run"
    commands = {
        "local_new": f"uv run {command}",
        "slurm_new": f"sbatch src/shell_scripts/hpc_ppc.sh {command.removeprefix('python -m src.neural_analysis.lfp_spike_phase_launcher ')}",
    }
    if resume_run_directory is not None:
        resume = f"resume --run-directory {quote(str(Path(resume_run_directory)))}"
        commands["local_resume"] = (
            "uv run python -m src.neural_analysis.lfp_spike_phase_launcher " + resume
        )
        commands["slurm_resume"] = f"sbatch src/shell_scripts/hpc_ppc.sh {resume}"
    return commands


def render_progress_event(streamlit: StreamlitOutput, event: ProgressEvent) -> None:
    """Display saved scalar progress metadata without compute or cache access.

    Parameters
    ----------
    streamlit : StreamlitOutput
        Output seam receiving text only.
    event : ProgressEvent
        Scalar component/stage/count/time metadata. Counts are work units and
        elapsed/ETA values are seconds; no numerical result arrays are present.

    Returns
    -------
    None
        Emits one informational text message and does not start/restart actions.
    """

    total = "?" if event.total_count is None else str(event.total_count)
    timing = f"; elapsed {event.elapsed_seconds:g} s" if event.elapsed_seconds is not None else ""
    if event.eta_seconds is not None:
        timing += f"; ETA {event.eta_seconds:g} s"
    streamlit.info(f"{event.component} {event.stage}: {event.completed_count}/{total}; {event.message}{timing}")


def _snapshot_plot_context(
    config: LFPSummaryConfig,
    source_cluster_directory: str | None = None,
) -> lfp_summary_plotting.PlotContext:
    """Create cache-plot provenance from immutable configuration metadata only.

    Parameters
    ----------
    config : LFPSummaryConfig
        Immutable labels, epoch bounds in seconds, source voltage units, and
        notch metadata. No cached or raw numerical array is opened.

    Returns
    -------
    lfp_summary_plotting.PlotContext
        Plot labels retaining seconds, Hz, and source-voltage-unit provenance.
    """

    gamma = next(band for band in config.phase.bands if band.name == "gamma")
    return lfp_summary_plotting.PlotContext(
        session_id=config.session_id,
        alignment_event=config.analysis_windows.alignment_event,
        epoch_bounds_s={
            "whole": (config.analysis_windows.whole_start_s, config.analysis_windows.whole_stop_s),
            "before": (config.analysis_windows.before_start_s, config.analysis_windows.before_stop_s),
            "after": (config.analysis_windows.after_start_s, config.analysis_windows.after_stop_s),
        },
        notch_enabled=config.phase.notch_enabled,
        gamma_exclusion_hz=(
            gamma.excluded_intervals_hz[0] if gamma.excluded_intervals_hz else None
        ),
        reference_description=(
            "Cached snapshot provenance is retained in the manifest."
            if source_cluster_directory is None
            else f"Cached snapshot copied from {source_cluster_directory}."
        ),
        source_voltage_unit=config.sites[0].voltage_unit,
    )


def _saved_snapshot_config(
    inspection: SnapshotInspection | None,
    component: str,
    fallback: LFPSummaryConfig,
) -> tuple[LFPSummaryConfig, str | None]:
    """Return one component's complete saved configuration for cache labels.

    Parameters
    ----------
    inspection : SnapshotInspection or None
        Receipt-validated manifest provenance. ``None`` preserves the legacy
        adapter path using ``fallback`` only.
    component : str
        Final cached component whose configuration snapshot is required.
    fallback : LFPSummaryConfig
        Existing live configuration used solely by old callers without a
        snapshot inspection; it does not open files or transform arrays.

    Returns
    -------
    tuple[LFPSummaryConfig, str or None]
        Deserialized saved config and the retained cluster-directory provenance.
    """

    if inspection is None:
        return fallback, None
    if not inspection.is_scientific_result or not isinstance(inspection.manifest, Mapping):
        raise ValueError("snapshot plot requires a valid inspection")
    components = inspection.manifest.get("components")
    entry = components.get(component) if isinstance(components, Mapping) else None
    saved = entry.get("configuration_snapshot") if isinstance(entry, Mapping) else None
    if not isinstance(saved, Mapping):
        raise ValueError("snapshot component lacks a saved configuration")
    try:
        return lfp_summary_config_from_json(json.dumps(dict(saved))), inspection.source_cluster_directory
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot component saved configuration is malformed") from error


def _pair_labels(arrays: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Return ordered ``site_a-site_b`` labels from cached pair axes.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Synchrony cache mapping with one-dimensional categorical pair-site axes.

    Returns
    -------
    tuple[str, ...]
        Pair labels with no array value, shape, or unit transformation.
    """

    first = arrays.get("pair_site_a_ids")
    second = arrays.get("pair_site_b_ids")
    if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray) or first.shape != second.shape or first.ndim != 1:
        raise ValueError("cached Synchrony pair axes are invalid")
    return tuple(f"{a}-{b}" for a, b in zip(first.astype(str), second.astype(str), strict=True))


def _condition_trials(
    arrays: Mapping[str, np.ndarray], condition_index: int, validity: np.ndarray | None = None
) -> np.ndarray:
    """Return cached filter/condition rows optionally intersected with validity.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Cached condition membership ``(trial, condition)`` and filter ``(trial,)``.
    condition_index : int
        Selected categorical condition position.
    validity : numpy.ndarray or None
        Optional boolean ``(trial,)`` saved site/pair validity mask.

    Returns
    -------
    numpy.ndarray
        Boolean ``(trial,)`` selection mask; no numerical signal values change.
    """

    membership = np.asarray(arrays["condition_membership"], dtype=bool)
    filtered = np.asarray(arrays["filter_membership"], dtype=bool)
    if membership.ndim != 2 or filtered.shape != (membership.shape[0],):
        raise ValueError("cached condition membership axes are invalid")
    selected = membership[:, condition_index] & filtered
    if validity is not None:
        valid = np.asarray(validity, dtype=bool)
        if valid.shape != selected.shape:
            raise ValueError("cached validity mask axes are invalid")
        selected &= valid
    return selected


def _named_index(names: np.ndarray, selected: str, name: str) -> int:
    """Return one selected categorical named-axis position without data changes.

    Parameters
    ----------
    names : numpy.ndarray
        One-dimensional fixed-Unicode/string categorical axis labels with no
        physical units.
    selected, name : str
        Requested categorical label and its manifest array name.

    Returns
    -------
    int
        Unique zero-based selected axis position; no array values, shapes, or
        units are transformed.
    """

    if names.ndim != 1:
        raise ValueError(f"cached {name} axis must be one dimensional")
    matches = np.flatnonzero(names.astype(str) == selected)
    if matches.size != 1:
        raise ValueError(f"cached {name} lacks selected value {selected!r}")
    return int(matches[0])


def select_spike_snapshot_slice(
    arrays: Mapping[str, np.ndarray],
    selection: SnapshotPlotSelection,
) -> SpikeSnapshotSlice:
    """Validate and select categorical Spike component axes without recomputing.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Cached Spike arrays. ``ppc`` has axes ``(unit, condition, site, epoch,
        frequency)`` and ``ppc_band_mean`` has ``(unit, condition, site, epoch,
        band)``; PPC is dimensionless and frequencies are Hz.
    selection : SnapshotPlotSelection
        Categorical view/site/condition/epoch/band values.

    Returns
    -------
    SpikeSnapshotSlice
        Integer named-axis positions with no copied numerical data or changed
        units.
    """

    required = {"condition_names", "site_ids", "epoch_names", "band_names", "ppc", "ppc_band_mean"}
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"cached Spike arrays lack required array: {sorted(missing)[0]}")
    condition_index = _named_index(arrays["condition_names"], selection.condition_name, "condition_names")
    site_index = _named_index(arrays["site_ids"], selection.site_id, "site_ids")
    epoch_index = _named_index(arrays["epoch_names"], selection.epoch_name, "epoch_names")
    band_index = _named_index(arrays["band_names"], selection.band_name, "band_names")
    ppc = arrays["ppc"]
    band_ppc = arrays["ppc_band_mean"]
    if ppc.ndim != 5 or band_ppc.ndim != 5:
        raise ValueError("cached Spike PPC arrays have incompatible axes")
    if ppc.shape[:4] != band_ppc.shape[:4]:
        raise ValueError("cached Spike PPC axes disagree")
    if ppc.shape[1:4] != (
        arrays["condition_names"].size,
        arrays["site_ids"].size,
        arrays["epoch_names"].size,
    ) or band_ppc.shape[4] != arrays["band_names"].size:
        raise ValueError("cached Spike named axes disagree")
    return SpikeSnapshotSlice(selection.view, condition_index, site_index, epoch_index, band_index)


def _require_cached_arrays(arrays: Mapping[str, np.ndarray], names: set[str], component: str) -> None:
    """Require named arrays before a cache-only component adapter indexes them.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        One selected decoded NPZ component with saved shapes, axes, and units.
    names : set[str]
        Required stored array names; this helper does not alter their contents.
    component : str
        Categorical cache component name used only in an error message.

    Returns
    -------
    None
        Raises ``ValueError`` when a required saved array is absent.
    """

    missing = names - set(arrays)
    if missing:
        raise ValueError(f"cached {component} arrays lack required array: {sorted(missing)[0]}")


def _cached_axis_values(arrays: Mapping[str, np.ndarray], name: str) -> tuple[str, ...]:
    """Return one nonempty saved categorical axis without transforming cache values.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Selected component arrays with named categorical axes.
    name : str
        Saved one-dimensional axis key, such as ``"condition_names"``.

    Returns
    -------
    tuple[str, ...]
        Ordered string labels copied from the saved axis. Labels have no units.
    """

    values = np.asarray(arrays.get(name, ()))
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"cached {name} must be a nonempty one-dimensional axis")
    return tuple(str(value) for value in values)


def _retained_band_frequency_indices(
    frequency_hz: np.ndarray,
    config: LFPSummaryConfig,
    band_name: str,
) -> np.ndarray:
    """Select saved Hz bins within one band and outside its open exclusions.

    Parameters
    ----------
    frequency_hz : numpy.ndarray
        Finite one-dimensional saved frequency coordinates in Hz.
    config : LFPSummaryConfig
        Immutable saved phase-band bounds and open excluded intervals in Hz.
    band_name : str
        Categorical saved band label.

    Returns
    -------
    numpy.ndarray
        Nonempty integer positions into ``frequency_hz``. The outer bounds are
        inclusive and each configured exclusion interval is open.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    if frequency.ndim != 1 or not np.all(np.isfinite(frequency)):
        raise ValueError("cached frequency_hz must be a finite one-dimensional axis")
    band = next((item for item in config.phase.bands if item.name == band_name), None)
    if band is None:
        raise ValueError(f"saved phase configuration lacks band {band_name!r}")
    retained = (frequency >= band.lower_hz) & (frequency <= band.upper_hz)
    for low_hz, high_hz in band.excluded_intervals_hz:
        retained &= ~((frequency > low_hz) & (frequency < high_hz))
    indices = np.flatnonzero(retained)
    if not indices.size:
        raise ValueError(f"cached frequency axis has no retained {band_name} bins")
    return indices


def _plot_cached_power_component(
    arrays: Mapping[str, np.ndarray],
    selection: SnapshotPlotSelection,
    context: lfp_summary_plotting.PlotContext,
) -> plt.Figure:
    """Delegate one saved Power view using only selected categorical cache axes.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Power arrays with PSD axes ``(site, trial, epoch, frequency)`` in dB
        and band-power axes ``(site, trial, epoch, band)`` in dB.
    selection : SnapshotPlotSelection
        Saved site/condition/epoch labels; the band-summary view displays its
        complete saved epoch and band axes.
    context : lfp_summary_plotting.PlotContext
        Immutable saved provenance and physical-unit labels.

    Returns
    -------
    matplotlib.figure.Figure
        Unsaved cache-only Power figure delegated to the plotting module.
    """

    _require_cached_arrays(
        arrays,
        {
            "frequency_hz", "normalized_psd_session_db", "band_power_session_db",
            "condition_names", "condition_membership", "filter_membership",
            "condition_effective_trial_count", "site_ids", "epoch_names", "band_names",
        },
        "Power",
    )
    condition_index = _named_index(arrays["condition_names"], selection.condition_name, "condition_names")
    site_index = _named_index(arrays["site_ids"], selection.site_id, "site_ids")
    epoch_index = _named_index(arrays["epoch_names"], selection.epoch_name, "epoch_names")
    selected_trials = _condition_trials(arrays, condition_index)
    condition_names = _cached_axis_values(arrays, "condition_names")
    epoch_names = _cached_axis_values(arrays, "epoch_names")
    band_names = _cached_axis_values(arrays, "band_names")
    if selection.view == "condition_psd":
        psd = np.asarray(arrays["normalized_psd_session_db"])
        if psd.ndim != 4:
            raise ValueError("cached Power PSD axes are invalid")
        figure, _ = lfp_summary_plotting.plot_condition_psd(
            np.asarray(arrays["frequency_hz"]),
            psd[site_index, selected_trials, epoch_index, :][np.newaxis, :, :],
            (selection.condition_name,),
            np.array((np.count_nonzero(selected_trials),), dtype=np.int64),
            selection.site_id,
            selection.epoch_name,
            "session-normalized dB",
            context,
        )
        return figure
    if selection.view == "band_power_summary":
        values = np.asarray(arrays["band_power_session_db"])
        if values.ndim != 4:
            raise ValueError("cached Power band axes are invalid")
        selected_values = np.full((1, values.shape[1], values.shape[2], values.shape[3]), np.nan)
        selected_values[0, selected_trials] = values[site_index, selected_trials]
        counts = np.asarray(arrays["condition_effective_trial_count"])[condition_index, site_index]
        figure, _ = lfp_summary_plotting.plot_band_power_summary(
            selected_values,
            (condition_names[condition_index],),
            epoch_names,
            band_names,
            np.full((1, len(epoch_names)), int(counts), dtype=np.int64),
            selection.site_id,
            "session-normalized dB",
            context,
        )
        return figure
    raise ValueError(f"unknown cached Power view: {selection.view!r}")


def _synchrony_entity(
    arrays: Mapping[str, np.ndarray],
    selection: SnapshotPlotSelection,
) -> tuple[int, np.ndarray, str, str, str]:
    """Resolve one saved Synchrony site or pair and its display validity mask.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Synchrony arrays with site/pair labels and validity axes ``(entity, trial)``.
    selection : SnapshotPlotSelection
        Selected site or ``site_a-site_b`` pair plus view family.

    Returns
    -------
    tuple[int, numpy.ndarray, str, str, str]
        Entity index, boolean validity mask, metric-array prefix, metric label,
        and selected saved entity label. The mask has shape ``(trial,)``.
    """

    pair_views = {"ispc_map", "ispc_band_summary", "plv_distribution", "plv_exemplar"}
    trial_count = np.asarray(arrays["filter_membership"]).size
    if selection.view not in pair_views:
        index = _named_index(arrays["site_ids"], selection.site_id, "site_ids")
        validity = np.asarray(
            arrays.get("site_valid", np.ones((len(arrays["site_ids"]), trial_count), dtype=bool)),
            dtype=bool,
        )[index]
        return index, validity, "itpc", "ITPC", selection.site_id
    labels = _pair_labels(arrays)
    if selection.site_id not in labels:
        raise ValueError(f"cached pair labels lack selected value {selection.site_id!r}")
    index = labels.index(selection.site_id)
    validity = np.asarray(
        arrays.get("pair_valid", np.ones((len(labels), trial_count), dtype=bool)),
        dtype=bool,
    )[index]
    return index, validity, "ispc", "ISPC", selection.site_id


def _plot_cached_synchrony_component(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    selection: SnapshotPlotSelection,
    context: lfp_summary_plotting.PlotContext,
) -> plt.Figure:
    """Delegate one saved Synchrony view without recomputing phase statistics.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Synchrony arrays with Hz/time coordinates and documented saved trial,
        site/pair, epoch, and band axes.
    config : LFPSummaryConfig
        Saved band bounds/exclusions used only to select already-cached bins.
    selection : SnapshotPlotSelection
        Categorical saved view/site-or-pair/condition/epoch/band selection.
    context : lfp_summary_plotting.PlotContext
        Immutable saved plot provenance.

    Returns
    -------
    matplotlib.figure.Figure
        Unsaved cache-only Synchrony figure delegated to the plotting module.
    """

    _require_cached_arrays(
        arrays,
        {"condition_names", "site_ids", "frequency_hz", "relative_time_s", "condition_membership", "filter_membership"},
        "Synchrony",
    )
    condition_index = _named_index(arrays["condition_names"], selection.condition_name, "condition_names")
    entity_index, validity, metric_prefix, metric_name, entity_label = _synchrony_entity(arrays, selection)
    selected_trials = _condition_trials(arrays, condition_index, validity)
    if selection.view not in {"itpc_band_summary", "ispc_band_summary", "plv_distribution", "plv_exemplar"}:
        metric = np.asarray(arrays[metric_prefix])
        count = np.asarray(arrays[f"{metric_prefix}_effective_trial_count"])
        figure, _ = lfp_summary_plotting.plot_phase_map(
            metric[condition_index, entity_index],
            count[condition_index, entity_index],
            np.asarray(arrays["frequency_hz"]),
            np.asarray(arrays["relative_time_s"]),
            metric_name,
            entity_label,
            selection.condition_name,
            context,
            total_displayed_trial_count=int(np.count_nonzero(selected_trials)),
        )
        return figure
    if selection.view in {"itpc_band_summary", "ispc_band_summary"}:
        epoch_index = _named_index(arrays["epoch_names"], selection.epoch_name, "epoch_names")
        band_index = _named_index(arrays["band_names"], selection.band_name, "band_names")
        figure, _ = lfp_summary_plotting.plot_phase_band_summary(
            np.array((np.asarray(arrays[f"{metric_prefix}_band_mean"])[condition_index, entity_index, epoch_index, band_index],)),
            np.array((np.asarray(arrays[f"{metric_prefix}_ci_low"])[condition_index, entity_index, epoch_index, band_index],)),
            np.array((np.asarray(arrays[f"{metric_prefix}_ci_high"])[condition_index, entity_index, epoch_index, band_index],)),
            np.array((np.count_nonzero(selected_trials),), dtype=np.int64),
            (f"{entity_label} {selection.condition_name}",),
            selection.band_name,
            selection.epoch_name,
            metric_name,
            context,
        )
        return figure
    if selection.view not in {"plv_distribution", "plv_exemplar"}:
        raise ValueError(f"unknown cached Synchrony view: {selection.view!r}")
    band_index = _named_index(arrays["band_names"], selection.band_name, "band_names")
    retained_frequency = _retained_band_frequency_indices(
        np.asarray(arrays["frequency_hz"]), config, selection.band_name
    )
    band_plv = np.asarray(arrays["plv_band_mean"])[selected_trials, entity_index, :, band_index]
    sample_counts = np.asarray(arrays["plv_valid_sample_count"])[selected_trials, entity_index, :, :]
    sample_fractions = np.asarray(arrays["plv_valid_sample_fraction"])[selected_trials, entity_index, :, :]
    conservative_counts = np.min(sample_counts[..., retained_frequency], axis=-1)
    conservative_fractions = np.min(sample_fractions[..., retained_frequency], axis=-1)
    if selection.view == "plv_distribution":
        epoch_names = _cached_axis_values(arrays, "epoch_names")
        figure, _ = lfp_summary_plotting.plot_plv_distribution(
            band_plv,
            conservative_counts,
            conservative_fractions,
            epoch_names,
            entity_label,
            selection.band_name,
            selection.condition_name,
            context,
        )
        return figure
    epoch_index = _named_index(arrays["epoch_names"], selection.epoch_name, "epoch_names")
    candidate_rows = np.flatnonzero(selected_trials)
    if not candidate_rows.size:
        raise ValueError("selected PLV cache has no valid trial")
    local_values = np.asarray(arrays["plv_band_mean"])[candidate_rows, entity_index, epoch_index, band_index]
    local_index = int(np.nanargmax(local_values))
    trial_row = int(candidate_rows[local_index])
    trial_id = int(np.asarray(arrays["trial_indices"])[trial_row])
    site_a, site_b = entity_label.split("-", 1)
    site_labels = _cached_axis_values(arrays, "site_ids")
    site_rows = (site_labels.index(site_a), site_labels.index(site_b))
    figure, _ = lfp_summary_plotting.plot_plv_exemplar(
        np.asarray(arrays["relative_time_s"]),
        np.asarray(arrays["source_trace"])[list(site_rows), trial_row],
        np.asarray(arrays["band_filtered_trace"])[list(site_rows), trial_row, band_index],
        np.asarray(arrays["hilbert_phase_rad"])[list(site_rows), trial_row, band_index],
        (site_a, site_b),
        entity_label,
        trial_id,
        "high",
        float(np.nanmean(local_values)),
        float(local_values[local_index]),
        context,
    )
    return figure


def _plot_population_ppc_maps(
    arrays: Mapping[str, np.ndarray],
    selected: SpikeSnapshotSlice,
    condition_names: tuple[str, ...],
    context: lfp_summary_plotting.PlotContext,
    site_label: str,
    epoch_name: str,
) -> plt.Figure:
    """Plot saved population PPC using distinct reliable and eligible masks.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Spike PPC/mask arrays with axes ``(unit, condition, site, epoch, frequency)``.
    selected : SpikeSnapshotSlice
        Selected site and epoch positions; population maps retain all conditions.
    condition_names : tuple[str, ...]
        Saved labels for the population-map condition axis.
    context : lfp_summary_plotting.PlotContext
        Immutable saved provenance.
    site_label, epoch_name : str
        Selected categorical labels used only for figure text.

    Returns
    -------
    matplotlib.figure.Figure
        Unsaved reliable-median and eligible-FDR cache-only figure.
    """

    ppc = np.asarray(arrays["ppc"])[:, :, selected.site_index, selected.epoch_index, :]
    reliable = np.asarray(arrays["reliable"], dtype=bool)[:, :, selected.site_index, selected.epoch_index, :]
    eligible = np.asarray(arrays["null_eligible"], dtype=bool)[:, :, selected.site_index, selected.epoch_index, :]
    significant = np.asarray(arrays["significant"], dtype=bool)[:, :, selected.site_index, selected.epoch_index, :]
    median = np.full(ppc.shape[1:], np.nan)
    fraction = np.full(ppc.shape[1:], np.nan)
    eligible_count = np.sum(eligible, axis=0, dtype=np.int64)
    for condition_index in range(ppc.shape[1]):
        for frequency_index in range(ppc.shape[2]):
            reliable_values = ppc[:, condition_index, frequency_index][reliable[:, condition_index, frequency_index]]
            if reliable_values.size:
                median[condition_index, frequency_index] = np.nanmedian(reliable_values)
            eligible_values = significant[:, condition_index, frequency_index][eligible[:, condition_index, frequency_index]]
            if eligible_values.size:
                fraction[condition_index, frequency_index] = np.mean(eligible_values)
    figure, _ = lfp_summary_plotting.plot_population_ppc_maps(
        median,
        fraction,
        eligible_count,
        np.full_like(eligible_count, ppc.shape[0]),
        condition_names,
        np.asarray(arrays["frequency_hz"]),
        site_label,
        epoch_name,
        context,
    )
    return figure


def _spike_band_reliability(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    site_index: int,
    band_names: tuple[str, ...],
) -> np.ndarray:
    """Reduce saved frequency reliability to all-retained-bin band reliability.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Spike reliability mask with axes ``(unit, condition, site, epoch, frequency)``.
    config : LFPSummaryConfig
        Saved phase-band bounds and open exclusions in Hz.
    site_index : int
        Selected saved site-axis position.
    band_names : tuple[str, ...]
        Ordered saved output band labels.

    Returns
    -------
    numpy.ndarray
        Boolean shape ``(unit, condition, epoch, band)``. A true value means
        every retained frequency in that saved band is reliable.
    """

    reliability = np.asarray(arrays["reliable"], dtype=bool)[:, :, site_index, :, :]
    output = np.empty(reliability.shape[:3] + (len(band_names),), dtype=bool)
    for band_index, band_name in enumerate(band_names):
        frequency_indices = _retained_band_frequency_indices(
            np.asarray(arrays["frequency_hz"]), config, band_name
        )
        output[..., band_index] = np.all(reliability[..., frequency_indices], axis=-1)
    return output


def _spike_exemplar_panel(
    arrays: Mapping[str, np.ndarray],
    selected: SpikeSnapshotSlice,
    unit_ids: tuple[str, ...],
    percentile: str,
) -> lfp_summary_plotting.PPCExemplarPanel:
    """Build one saved low/high Spike panel without selecting a new trial.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Cached Spike arrays with saved unit/trial exemplar identities and packed
        relative spike times in seconds.
    selected : SpikeSnapshotSlice
        Selected condition/site/epoch/band positions.
    unit_ids : tuple[str, ...]
        Saved probe-qualified unit axis labels.
    percentile : str
        Exact ``"low"`` or ``"high"`` saved exemplar identity.

    Returns
    -------
    lfp_summary_plotting.PPCExemplarPanel
        One pooled unit metric and its separately saved illustrative trial.
    """

    if percentile not in {"low", "high"}:
        raise ValueError("Spike exemplar percentile must be low or high")
    index = (selected.condition_index, selected.site_index, selected.epoch_index, selected.band_index)
    unit_name = str(np.asarray(arrays[f"selected_{percentile}_unit_ids"])[index])
    if unit_name not in unit_ids:
        raise ValueError("cached Spike exemplar unit is unavailable")
    unit_index = unit_ids.index(unit_name)
    trial_id = int(np.asarray(arrays[f"illustrative_{percentile}_trial_indices"])[index])
    trial_rows = np.flatnonzero(np.asarray(arrays["trial_indices"]) == trial_id)
    if trial_rows.size != 1:
        raise ValueError("cached Spike exemplar trial is unavailable")
    trial_row = int(trial_rows[0])
    offsets = np.asarray(arrays["relative_spike_time_offsets"])[unit_index]
    times = np.asarray(arrays["relative_spike_times_s"])[offsets[trial_row]:offsets[trial_row + 1]]
    label = "5th percentile" if percentile == "low" else "95th percentile"
    return lfp_summary_plotting.PPCExemplarPanel(
        unit_name,
        label,
        np.asarray(arrays["ppc"])[unit_index, selected.condition_index, selected.site_index, selected.epoch_index],
        np.asarray(arrays["preferred_phase_rad"])[unit_index, selected.condition_index, selected.site_index, selected.epoch_index],
        np.asarray(arrays["representative_phase_hist_count"])[unit_index, selected.condition_index, selected.site_index, selected.epoch_index, selected.band_index],
        trial_id,
        np.asarray(arrays["source_trace"])[selected.site_index, trial_row],
        np.asarray(arrays["band_filtered_trace"])[selected.site_index, trial_row, selected.band_index],
        np.asarray(arrays["hilbert_phase_rad"])[selected.site_index, trial_row, selected.band_index],
        times,
    )


def _plot_cached_spike_component(
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    selection: SnapshotPlotSelection,
    context: lfp_summary_plotting.PlotContext,
) -> plt.Figure:
    """Delegate one saved Spike view while preserving all saved identities.

    Parameters
    ----------
    arrays : mapping[str, numpy.ndarray]
        Spike arrays with documented unit/condition/site/epoch/frequency axes
        and cached exemplar paths; PPC is dimensionless and frequency is Hz.
    config : LFPSummaryConfig
        Saved phase-band metadata used only for retained bins/display labels.
    selection : SnapshotPlotSelection
        Categorical saved view/site/condition/epoch/band selection.
    context : lfp_summary_plotting.PlotContext
        Immutable saved plot provenance.

    Returns
    -------
    matplotlib.figure.Figure
        Unsaved cache-only Spike figure delegated to the plotting module.
    """

    _require_cached_arrays(arrays, {"unit_ids", "frequency_hz", "ppc", "computable", "reliable", "spike_count"}, "Spike")
    selected = select_spike_snapshot_slice(arrays, selection)
    unit_ids = _cached_axis_values(arrays, "unit_ids")
    condition_names = _cached_axis_values(arrays, "condition_names")
    epoch_names = _cached_axis_values(arrays, "epoch_names")
    band_names = _cached_axis_values(arrays, "band_names")
    ppc = np.asarray(arrays["ppc"])[:, selected.condition_index, selected.site_index, selected.epoch_index, :]
    computable = np.asarray(arrays["computable"])[:, selected.condition_index, selected.site_index, selected.epoch_index, :]
    reliable = np.asarray(arrays["reliable"])[:, selected.condition_index, selected.site_index, selected.epoch_index, :]
    spike_count = np.asarray(arrays["spike_count"])[:, selected.condition_index, selected.site_index, selected.epoch_index, :]
    if selection.view == "unit_ppc_map":
        figure, _ = lfp_summary_plotting.plot_unit_ppc_map(
            ppc, computable, reliable, spike_count, np.asarray(arrays["frequency_hz"]), unit_ids,
            selection.condition_name, selection.site_id, selection.epoch_name, context,
        )
        return figure
    if selection.view == "population_ppc_maps":
        _require_cached_arrays(arrays, {"null_eligible", "significant"}, "Spike")
        return _plot_population_ppc_maps(
            arrays, selected, condition_names, context, selection.site_id, selection.epoch_name
        )
    if selection.view == "ppc_band_summary":
        _require_cached_arrays(arrays, {"ppc_band_mean"}, "Spike")
        band_values = np.asarray(arrays["ppc_band_mean"])[:, :, selected.site_index, :, :]
        reliable_band = _spike_band_reliability(arrays, config, selected.site_index, band_names)
        figure, _ = lfp_summary_plotting.plot_ppc_band_summary(
            np.moveaxis(band_values, 0, 1),
            np.moveaxis(reliable_band, 0, 1),
            condition_names,
            epoch_names,
            band_names,
            len(unit_ids),
            selection.site_id,
            context,
        )
        return figure
    if selection.view == "ppc_exemplar_pair":
        _require_cached_arrays(
            arrays,
            {
                "phase_bin_edges_rad", "relative_time_s", "selected_low_unit_ids",
                "selected_high_unit_ids", "illustrative_low_trial_indices",
                "illustrative_high_trial_indices", "relative_spike_time_offsets",
                "relative_spike_times_s", "preferred_phase_rad",
                "representative_phase_hist_count", "source_trace", "band_filtered_trace",
                "hilbert_phase_rad", "trial_indices",
            },
            "Spike",
        )
        representative_frequency_hz = 8.0 if selection.band_name == "theta" else 40.0
        figure, _ = lfp_summary_plotting.plot_ppc_exemplar_pair(
            frequency_hz=np.asarray(arrays["frequency_hz"]),
            phase_bin_edges_rad=np.asarray(arrays["phase_bin_edges_rad"]),
            relative_time_s=np.asarray(arrays["relative_time_s"]),
            low=_spike_exemplar_panel(arrays, selected, unit_ids, "low"),
            high=_spike_exemplar_panel(arrays, selected, unit_ids, "high"),
            condition_name=selection.condition_name,
            site_label=selection.site_id,
            epoch_name=selection.epoch_name,
            band_name=selection.band_name,
            representative_frequency_hz=representative_frequency_hz,
            context=context,
        )
        return figure
    raise ValueError(f"unknown cached Spike view: {selection.view!r}")


def plot_cached_snapshot_component(
    component: str,
    arrays: Mapping[str, np.ndarray],
    config: LFPSummaryConfig,
    selection: SnapshotPlotSelection,
    *,
    inspection: SnapshotInspection | None = None,
) -> plt.Figure:
    """Delegate every approved cached view to the existing pure plotters.

    Parameters
    ----------
    component : str
        ``"power"``, ``"synchrony"``, or ``"spike_phase"`` cache identity.
    arrays : mapping[str, numpy.ndarray]
        Validated selected component arrays with their stored axes/units. This
        function opens no source recording and applies no numerical transform.
    config : LFPSummaryConfig
        Immutable provenance and time/frequency units for labels only.
    selection : SnapshotPlotSelection
        Categorical cached axes selecting one plot slice.
    inspection : SnapshotInspection or None
        Optional validated snapshot used to deserialize the selected component's
        saved configuration and retained cluster provenance. No component bytes
        are opened here.

    Returns
    -------
    matplotlib.figure.Figure
        Unsaved cache-only figure delegated to ``lfp_summary_plotting``.
    """

    saved_config, cluster_directory = _saved_snapshot_config(inspection, component, config)
    context = _snapshot_plot_context(saved_config, cluster_directory)
    if component == "power":
        return _plot_cached_power_component(arrays, selection, context)
    if component == "synchrony":
        return _plot_cached_synchrony_component(arrays, saved_config, selection, context)
    if component == "spike_phase":
        return _plot_cached_spike_component(arrays, saved_config, selection, context)
    raise ValueError(f"snapshot plotting is unavailable for {component!r}")


def render_snapshot_component_view(
    streamlit: StreamlitOutput,
    *,
    snapshot_directory: Path,
    component: str,
    component_cache: SnapshotComponentCache,
    load_component: LoadComponent,
    plot_component: Callable[[str, dict[str, np.ndarray]], plt.Figure],
) -> None:
    """Render one selected read-only snapshot component with no compute seam.

    Parameters
    ----------
    streamlit : StreamlitOutput
        Display seam for already-created figures.
    snapshot_directory : pathlib.Path
        Exact user-entered local snapshot directory.
    component : str
        Selected final component; only its NPZ arrays are loaded.
    component_cache : SnapshotComponentCache
        In-process selected-component cache keyed by validated identity.
    load_component : callable
        Existing schema-validating component loader; it receives no raw input.
    plot_component : callable
        Cache-array-to-figure adapter with no numerical computation or writes.

    Returns
    -------
    None
        Displays and closes one figure. Invalid snapshots are reported without
        opening components, writing files, or falling back to live mode.
    """

    inspection = validate_cache_snapshot(snapshot_directory)
    if not inspection.is_scientific_result:
        streamlit.error(inspection.message)
        return
    try:
        arrays = component_cache.load(inspection, component, load_component)
        figure = plot_component(component, arrays)
    except (OSError, ValueError) as error:
        streamlit.error(str(error))
        return
    render_summary_figure(streamlit, figure)


def default_summary_ui_values() -> SummaryUIValues:
    """Return approved CT026 controls for the first LFP summary inspection.

    Returns
    -------
    SummaryUIValues
        CT026 session identifier, PFC/HPC saved channels, choice alignment,
        500 Hz phase grid, enabled 60 Hz notch, and 1,000 bootstraps.
    """

    return SummaryUIValues(
        session_id="CT026_2026-08-01_130853",
        site_channels={"PFC": 5, "HPC1": 222, "HPC2": 14},
        alignment_event="choice_time",
        output_rate_hz=500.0,
        notch_enabled=True,
        bootstrap_count=1000,
        choice_filter="all",
        context_filter="all",
    )


def parse_excluded_trial_rows(excluded_rows_text: str) -> tuple[int, ...]:
    """Parse comma-separated nonnegative trial-table row positions.

    Parameters
    ----------
    excluded_rows_text : str
        Comma-separated decimal integer row positions, such as ``"2, 5"``.
        Blank text returns no exclusions. Row positions are categorical table
        indices, not times, physical units, or one-based trial numbers.

    Returns
    -------
    tuple[int, ...]
        Unique nonnegative zero-based trial rows in the entered order.

    Raises
    ------
    ValueError
        If text contains an empty token, a noninteger, a negative row, or a
        duplicate row. No input array, time axis, or units are transformed.
    """

    if not isinstance(excluded_rows_text, str):
        raise ValueError("Excluded trial rows must be comma-separated text.")
    if not excluded_rows_text.strip():
        return ()
    rows: list[int] = []
    for token in excluded_rows_text.split(","):
        value = token.strip()
        if not value:
            raise ValueError("Excluded trial rows cannot contain empty entries.")
        try:
            row = int(value)
        except ValueError as error:
            raise ValueError("Excluded trial rows must be decimal integers.") from error
        if str(row) != value or row < 0:
            raise ValueError("Excluded trial rows must be nonnegative integers.")
        if row in rows:
            raise ValueError("Excluded trial rows must be unique.")
        rows.append(row)
    return tuple(rows)


def assemble_summary_config(
    *,
    session_id: str,
    session_path: Path,
    output_directory: Path,
    sites: tuple[LFPSiteConfig, ...],
    site_pairs: tuple[tuple[str, str], ...],
    unit_population: UnitPopulationConfig | None,
    choice_filter: str,
    context_filter: str,
    excluded_trial_indices: tuple[int, ...],
    alignment_event: str,
    notch_enabled: bool,
    bootstrap_count: int,
    random_seed: int,
    output_rate_hz: float = 500.0,
) -> LFPSummaryConfig:
    """Assemble and validate one immutable config from active summary controls.

    Paths identify the active session and disposable cache directory. Sites use
    explicit saved channels and source voltage units. The random seed propagates
    to phase bootstraps, PPC inference, and the top-level configuration.

    Returns
    -------
    LFPSummaryConfig
        Validated configuration; times are seconds, frequencies are Hz, and
        no raw traces, wavelets, or Streamlit objects are retained.
    """

    base = default_lfp_summary_config()
    phase = replace(
        base.phase,
        output_rate_hz=float(output_rate_hz),
        notch_enabled=bool(notch_enabled),
        bootstrap_count=int(bootstrap_count),
        seed=int(random_seed),
    )
    power = replace(base.power, notch_enabled=bool(notch_enabled))
    ppc = replace(base.ppc, seed=int(random_seed))
    config = replace(
        base,
        session_id=str(session_id),
        session_path=Path(session_path),
        output_directory=Path(output_directory),
        sites=tuple(sites),
        site_pairs=tuple(site_pairs),
        unit_population=unit_population,
        trial_filter=TrialFilterConfig(
            choice=str(choice_filter),
            context=str(context_filter),
            excluded_trial_indices=tuple(int(index) for index in excluded_trial_indices),
        ),
        analysis_windows=replace(base.analysis_windows, alignment_event=str(alignment_event)),
        power=power,
        phase=phase,
        ppc=ppc,
        trial_table_path=(
            Path(session_path) / "processed" / f"{session_id}_augmented_trials.csv"
        ),
        random_seed=int(random_seed),
    )
    validate_lfp_summary_config(config)
    return config


def compute_production_power(config: LFPSummaryConfig) -> SummaryActionResult:
    """Run the real atomic Power pipeline for one configured active session.

    Parameters
    ----------
    config : LFPSummaryConfig
        Validated active session configuration. Its trial CSV, source sites,
        output directory, axes, and physical units are passed unchanged to the
        runtime and pipeline layers.

    Returns
    -------
    SummaryActionResult
        Completed Power result with its committed manifest, or a failed result
        identifying the pipeline stage without replacing a previous cache.
    """
    dependencies = lfp_summary_runtime.make_power_pipeline_dependencies(
        trial_table_loader=lfp_summary_runtime.load_configured_trial_table,
    )
    result = lfp_summary_pipeline.compute_power_component(config, dependencies)
    return SummaryActionResult(
        result.component,
        result.state,
        getattr(result, "manifest", None),
        result.error,
    )


def make_production_summary_dependencies() -> SummaryWebDependencies:
    """Create live composed Power/Synchrony and cache-only plotting dependencies.

    Returns
    -------
    SummaryWebDependencies
        One WP10 composed runtime bundle is reused by live Power and Synchrony.
        Cached views load NPZ arrays only. Spike phase and Compute All remain
        unavailable synchronous actions because the UI exposes launcher handoffs.
    """
    composed = lfp_summary_runtime.make_lfp_summary_pipeline_dependencies(
        trial_table_loader=lfp_summary_runtime.load_configured_trial_table,
    )

    def composed_action(component: str) -> ComputeAction:
        """Return one live action using the shared composed pipeline bundle."""

        pipeline_action = {
            "power": lfp_summary_pipeline.compute_power_component,
            "synchrony": lfp_summary_pipeline.compute_synchrony_component,
        }[component]

        def action(config: LFPSummaryConfig) -> SummaryActionResult:
            """Run one bounded component through the shared composed dependency bundle."""

            result = pipeline_action(config, composed)
            return SummaryActionResult(
                result.component,
                result.state,
                getattr(result, "manifest", None),
                (
                    result.error
                    if result.state == "complete"
                    else f"{component} production action is unavailable: {result.error}"
                ),
            )

        return action

    def unavailable(component: str) -> ComputeAction:
        """Return one action reporting an unavailable production component."""
        def action(_: LFPSummaryConfig) -> SummaryActionResult:
            """Return failure metadata without preparing data or writing a cache."""
            return SummaryActionResult(
                component,
                "failed",
                None,
                f"{component} production action is unavailable",
            )

        return action

    def load_manifest(cache_directory: Path) -> dict[str, object]:
        """Load active cache metadata using the configuration bound by view callers."""
        manifest_path = cache_directory / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("no readable LFP summary manifest")
        try:
            import json

            decoded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("no readable LFP summary manifest") from error
        if not isinstance(decoded, dict):
            raise ValueError("no readable LFP summary manifest")
        return decoded

    def load_selected_component(
        output_directory: Path,
        manifest: dict[str, object],
        component: str,
    ) -> dict[str, np.ndarray]:
        """Load one live component from its exact final ``component.npz`` path.

        Parameters
        ----------
        output_directory : pathlib.Path
            Configured cache directory containing final component NPZ files.
        manifest : dict[str, object]
            Current cache metadata; numerical arrays retain its saved axes.
        component : str
            One summary component whose file is ``output_directory/component.npz``.

        Returns
        -------
        dict[str, numpy.ndarray]
            Loader-validated cached arrays with unchanged shapes and units.
        """

        return load_component_arrays(output_directory / f"{component}.npz", manifest, component)

    def plot_view(
        view: str,
        arrays: dict[str, dict[str, np.ndarray]],
        config: LFPSummaryConfig,
    ) -> plt.Figure:
        """Render a cached component view without raw-LFP preparation.

        Parameters
        ----------
        view : str
            Cached component selected by the webapp.
        arrays : dict[str, dict[str, numpy.ndarray]]
            Validated component NPZ arrays already loaded by the cache seam.
        config : LFPSummaryConfig
            Immutable metadata for labels, exact windows, and preprocessing
            provenance; it is not used to open raw recordings.

        Returns
        -------
        matplotlib.figure.Figure
            Unsaved figure built entirely from cache arrays and configuration.
        """
        if view == "power":
            return _plot_cached_power(arrays["power"], config)
        raise ValueError(f"{view} live cached plotting requires a snapshot selection")

    return SummaryWebDependencies(
        compute_power=composed_action("power"),
        compute_synchrony=composed_action("synchrony"),
        compute_spike_phase=unavailable("spike_phase"),
        compute_all=unavailable("all"),
        load_manifest=load_manifest,
        load_component=load_selected_component,
        plot_view=plot_view,
    )


def _plot_cached_power(
    arrays: dict[str, np.ndarray],
    config: LFPSummaryConfig,
) -> plt.Figure:
    """Adapt validated Power cache axes to the shared condition-PSD plotter.

    Parameters
    ----------
    arrays : dict[str, numpy.ndarray]
        Validated ``power.npz`` arrays. ``frequency_hz`` has Hz coordinates,
        condition membership has ``(trial, condition)`` axes, and normalized
        PSD has ``(site, trial, epoch, frequency)`` dB axes. No source paths,
        traces, or recordings are opened here.
    config : LFPSummaryConfig
        Immutable session, half-open window, notch, band, and source-unit
        metadata used only to label the cache-backed figure.

    Returns
    -------
    matplotlib.figure.Figure
        Unsaved condition-resolved whole-window PSD figure.
    """
    required_names = {
        "frequency_hz",
        "normalized_psd_session_db",
        "condition_names",
        "condition_membership",
        "filter_membership",
        "condition_effective_trial_count",
        "site_ids",
        "site_voltage_units",
        "epoch_names",
    }
    missing_names = sorted(required_names - set(arrays))
    if missing_names:
        raise ValueError(f"cached Power arrays lack required array: {missing_names[0]}")
    frequency_hz = arrays["frequency_hz"]
    normalized_psd_db = arrays["normalized_psd_session_db"]
    condition_names = arrays["condition_names"]
    membership = arrays["condition_membership"]
    filter_membership = arrays["filter_membership"]
    effective_count = arrays["condition_effective_trial_count"]
    site_ids = arrays["site_ids"]
    voltage_units = arrays["site_voltage_units"]
    epoch_names = arrays["epoch_names"]
    if (
        frequency_hz.ndim != 1
        or normalized_psd_db.ndim != 4
        or condition_names.ndim != 1
        or membership.ndim != 2
        or filter_membership.ndim != 1
        or effective_count.ndim != 2
        or site_ids.ndim != 1
        or voltage_units.ndim != 1
        or epoch_names.ndim != 1
    ):
        raise ValueError("cached Power arrays have incompatible axes")
    site_count, trial_count, epoch_count, frequency_count = normalized_psd_db.shape
    cached_condition_names = tuple(str(name) for name in condition_names)
    try:
        base_condition_indices = np.array(
            [
                cached_condition_names.index(name)
                for name in lfp_summary_plotting.POWER_BASE_CONDITIONS
            ],
            dtype=np.int64,
        )
    except ValueError as error:
        raise ValueError(
            "cached Power condition_names lack a required base condition"
        ) from error
    condition_count = condition_names.size
    if (
        frequency_hz.size != frequency_count
        or membership.shape != (trial_count, condition_count)
        or filter_membership.shape != (trial_count,)
        or effective_count.shape != (condition_count, site_count)
        or site_ids.size != site_count
        or voltage_units.size != site_count
        or epoch_names.size != epoch_count
    ):
        raise ValueError("cached Power arrays have inconsistent named-axis lengths")
    whole_matches = np.flatnonzero(epoch_names == "whole")
    if whole_matches.size != 1:
        raise ValueError("cached Power epoch_names must include exactly one whole epoch")
    whole_epoch_index = int(whole_matches[0])
    display_frequency = frequency_hz <= 100.0
    if not np.any(display_frequency):
        raise ValueError("cached Power frequency grid does not include 0-100 Hz")
    condition_psd_db = np.full(
        (
            base_condition_indices.size,
            trial_count,
            int(np.count_nonzero(display_frequency)),
        ),
        np.nan,
        dtype=normalized_psd_db.dtype,
    )
    if membership.dtype != np.dtype(bool) or filter_membership.dtype != np.dtype(bool):
        raise ValueError("cached Power condition and filter membership must be boolean")
    for output_index, condition_index in enumerate(base_condition_indices):
        selected_trials = membership[:, condition_index] & filter_membership
        condition_psd_db[output_index, selected_trials] = normalized_psd_db[
            0,
            selected_trials,
            whole_epoch_index,
        ][:, display_frequency]
    gamma_band = next(band for band in config.power.bands if band.name == "gamma")
    gamma_exclusion_hz = (
        gamma_band.excluded_intervals_hz[0]
        if gamma_band.excluded_intervals_hz
        else None
    )
    context = lfp_summary_plotting.PlotContext(
        session_id=config.session_id,
        alignment_event=config.analysis_windows.alignment_event,
        epoch_bounds_s={
            "whole": (
                config.analysis_windows.whole_start_s,
                config.analysis_windows.whole_stop_s,
            ),
            "before": (
                config.analysis_windows.before_start_s,
                config.analysis_windows.before_stop_s,
            ),
            "after": (
                config.analysis_windows.after_start_s,
                config.analysis_windows.after_stop_s,
            ),
        },
        notch_enabled=config.power.notch_enabled,
        gamma_exclusion_hz=gamma_exclusion_hz,
        reference_description=(
            "External/session reference and preprocessing provenance are cached "
            "in manifest metadata; inspect warnings before interpretation."
        ),
        source_voltage_unit=str(voltage_units[0]),
    )
    figure, _axes = lfp_summary_plotting.plot_condition_psd(
        frequency_hz[display_frequency],
        condition_psd_db,
        lfp_summary_plotting.POWER_BASE_CONDITIONS,
        effective_count[base_condition_indices, 0],
        str(site_ids[0]),
        "whole",
        "session-normalized dB",
        context,
    )
    return figure


def component_status_message(status: ComponentStatus) -> str:
    """Format a visible concise message for one cache component state.

    Parameters
    ----------
    status : ComponentStatus
        Missing, compatible, stale, running, or failed state plus differences.

    Returns
    -------
    str
        Human-readable state text. Stale differences are included verbatim.
    """

    details = "; ".join(status.differences)
    message = f"{status.status}: {details}" if details else status.status
    return message


def require_renderable_component(status: ComponentStatus, allow_stale: bool) -> None:
    """Reject noncurrent cached components unless stale inspection is explicit.

    Parameters
    ----------
    status : ComponentStatus
        Cache compatibility status for the selected component.
    allow_stale : bool
        Explicit user override permitting stale inspection with a warning.

    Returns
    -------
    None
        Raises ``ValueError`` for missing, running, failed, or non-overridden
        stale results; no arrays or physical units are transformed.
    """

    if status.status == "compatible" or (status.status == "stale" and allow_stale):
        return
    raise ValueError(component_status_message(status))


def run_summary_action(
    action: str,
    config: LFPSummaryConfig,
    dependencies: SummaryWebDependencies,
    *,
    previous_manifest: dict[str, object] | None = None,
) -> SummaryActionResult:
    """Dispatch one selected synchronous pipeline action and reload on success.

    Parameters
    ----------
    action : str
        One of power, synchrony, spike_phase, or all.
    config : LFPSummaryConfig
        Validated immutable active configuration.
    dependencies : SummaryWebDependencies
        Injected compute and cache-load seams; numerical work remains external.
    previous_manifest : dict | None
        Previously validated cache metadata retained after a failed action.

    Returns
    -------
    SummaryActionResult
        Success contains a freshly loaded manifest. Failure retains exactly the
        previous manifest object and does not replace a valid cached result.
    """

    validate_lfp_summary_config(config)
    actions = {
        "power": dependencies.compute_power,
        "synchrony": dependencies.compute_synchrony,
        "spike_phase": dependencies.compute_spike_phase,
        "all": dependencies.compute_all,
    }
    if action not in actions:
        raise ValueError(f"Unknown summary action: {action!r}")
    result = actions[action](config)
    if result.state != "complete":
        return SummaryActionResult(result.component, result.state, previous_manifest, result.error)
    manifest = dependencies.load_manifest(config.output_directory)
    return SummaryActionResult(result.component, result.state, manifest, result.error)


def load_summary_view_arrays(
    view: str,
    config: LFPSummaryConfig,
    manifest: dict[str, object],
    dependencies: SummaryWebDependencies,
) -> dict[str, dict[str, np.ndarray]]:
    """Load only the selected live ``output_directory/component.npz`` file.

    Parameters
    ----------
    view : str
        ``power``, ``synchrony``, or ``spike_phase`` component identity.
    config : LFPSummaryConfig
        Validated live configuration whose ``output_directory`` identifies the
        parent of the selected final NPZ file.
    manifest : dict[str, object]
        Current JSON cache metadata defining the returned array axes and units.
    dependencies : SummaryWebDependencies
        Loader seam receiving the exact component NPZ path, manifest, and name.

    Returns
    -------
    dict[str, dict[str, numpy.ndarray]]
        One named component mapping. Arrays preserve cache-defined shapes,
        physical units, and NaN meaning; no raw LFP/wavelet data is opened.
    """

    validate_lfp_summary_config(config)
    if view not in SUMMARY_VIEWS:
        raise ValueError(f"Unknown summary view: {view!r}")
    return {view: dependencies.load_component(config.output_directory, manifest, view)}


def render_summary_counts(
    streamlit: StreamlitOutput,
    *,
    trial_count: int,
    unit_count: int,
    spike_count: int,
    unstable: bool,
) -> None:
    """Display visible selected counts and an optional low-count warning.

    Counts are nonnegative categorical observations: trials, units, and valid
    spikes. They have no physical units. The instability flag is an inspection
    warning and does not alter cached numerical values.
    """

    streamlit.info(
        f"Trials: {int(trial_count)}; units: {int(unit_count)}; spikes: {int(spike_count)}"
    )
    if unstable:
        streamlit.warning("Unstable low trial count: inspect this result cautiously.")


def render_summary_figure(streamlit: StreamlitOutput, figure: plt.Figure) -> None:
    """Display one cached-summary figure and always close its Matplotlib handle.

    The figure is already populated from cache arrays with component-defined
    axes and units. This function neither saves nor transforms plotted data.
    """

    try:
        streamlit.pyplot(figure)
    finally:
        plt.close(figure)


def _retained_snapshot_inspection(
    session_state: MutableMapping[str, object],
    entered_path: str,
) -> SnapshotInspection | None:
    """Reuse a valid inspection only while its explicit path/stat identity matches.

    Parameters
    ----------
    session_state : mutable mapping
        Streamlit session state holding only prior read-only inspection metadata.
    entered_path : str
        Current explicit snapshot widget text. Blank clears retained state and
        never falls back to a previous directory.

    Returns
    -------
    SnapshotInspection or None
        A valid retained inspection or a newly validated inspection. ``None``
        represents the explicit blank state; no component bytes are opened.
    """

    text = entered_path.strip()
    if not text:
        for key in ("lfp_summary_snapshot_inspection", "lfp_summary_snapshot_path"):
            session_state.pop(key, None)
        return None
    requested = Path(text)
    try:
        resolved = requested.resolve()
    except OSError:
        resolved = requested
    retained = session_state.get("lfp_summary_snapshot_inspection")
    if (
        isinstance(retained, SnapshotInspection)
        and retained.is_scientific_result
        and retained.snapshot_directory == resolved
        and retained.stat_identity == _snapshot_stat_identity(resolved)
    ):
        return retained
    inspection = validate_cache_snapshot(requested)
    session_state["lfp_summary_snapshot_path"] = text
    if inspection.is_scientific_result:
        session_state["lfp_summary_snapshot_inspection"] = inspection
    else:
        session_state.pop("lfp_summary_snapshot_inspection", None)
    return inspection


def _selected_population(
    *,
    session_state: MutableMapping[str, object],
    probe_label: str,
    sorter_paths: Mapping[str, Path],
    aligned_spike_paths: Mapping[str, Path],
    cluster_metadata_loader: Callable[[Path], object],
    channel_metadata_loader: Callable[[Path], object],
) -> UnitPopulationConfig:
    """Load and retain metadata for exactly the currently selected probe.

    Parameters
    ----------
    session_state : mutable mapping
        Streamlit state used only to retain one immutable selected population.
    probe_label : str
        Exact ``ProbeA`` or ``ProbeB`` UI choice.
    sorter_paths, aligned_spike_paths : mapping[str, pathlib.Path]
        Explicit parent-route source identities for both probes. Only the
        selected mapping entry is handed to lazy metadata loaders.
    cluster_metadata_loader, channel_metadata_loader : callable
        Lazy selected-sorter seams returning table-like categorical metadata.

    Returns
    -------
    UnitPopulationConfig
        One good/MUA, good/inside-brain selected-probe population. No other
        probe metadata, spike arrays, or recordings are opened.
    """

    sorter_path = sorter_paths.get(probe_label)
    aligned_path = aligned_spike_paths.get(probe_label)
    if sorter_path is None or aligned_path is None:
        raise ValueError(f"missing explicit paths for {probe_label}")
    key = (probe_label, Path(sorter_path), Path(aligned_path))
    retained_key = session_state.get("lfp_summary_population_key")
    retained = session_state.get("lfp_summary_population")
    if retained_key == key and isinstance(retained, UnitPopulationConfig):
        return retained
    population = build_active_summary_population(
        probe_label=probe_label,
        sorter_path=Path(sorter_path),
        aligned_spike_path=Path(aligned_path),
        cluster_metadata=cluster_metadata_loader(Path(sorter_path)),
        channel_metadata=channel_metadata_loader(Path(sorter_path)),
    )
    session_state["lfp_summary_population_key"] = key
    session_state["lfp_summary_population"] = population
    return population


def _snapshot_view_options(component: str) -> tuple[str, ...]:
    """Return the approved cache-only views for one scientific component.

    Parameters
    ----------
    component : str
        ``"power"``, ``"synchrony"``, or ``"spike_phase"`` cache identity.

    Returns
    -------
    tuple[str, ...]
        Ordered categorical view labels. They have no numerical units and do
        not cause cache loading or computation.
    """

    views = {
        "power": ("condition_psd", "band_power_summary"),
        "synchrony": (
            "itpc_map", "ispc_map", "itpc_band_summary", "ispc_band_summary",
            "plv_distribution", "plv_exemplar",
        ),
        "spike_phase": (
            "unit_ppc_map", "population_ppc_maps", "ppc_band_summary", "ppc_exemplar_pair",
        ),
    }
    try:
        return views[component]
    except KeyError as error:
        raise ValueError(f"unknown snapshot component: {component!r}") from error


def _snapshot_selection_controls(
    sidebar: object,
    component: str,
    arrays: Mapping[str, np.ndarray],
) -> SnapshotPlotSelection:
    """Collect only the saved categorical controls relevant to one cached view.

    Parameters
    ----------
    sidebar : object
        Streamlit-compatible sidebar exposing ``selectbox`` only; it receives
        labels and categorical cached axes, never numerical array values.
    component : str
        Selected scientific cache component.
    arrays : mapping[str, numpy.ndarray]
        Already-loaded selected component with saved named categorical axes.

    Returns
    -------
    SnapshotPlotSelection
        One complete immutable selection. Fields irrelevant to the chosen view
        use the first saved axis value without presenting an extra UI control.
    """

    view = sidebar.selectbox("Cached view", options=_snapshot_view_options(component))
    site_ids = _cached_axis_values(arrays, "site_ids")
    condition_names = _cached_axis_values(arrays, "condition_names")
    epoch_names = _cached_axis_values(arrays, "epoch_names")
    band_names = _cached_axis_values(arrays, "band_names")
    site_id = site_ids[0]
    condition_name = condition_names[0]
    epoch_name = epoch_names[0]
    band_name = band_names[0]
    pair_views = {"ispc_map", "ispc_band_summary", "plv_distribution", "plv_exemplar"}
    if component == "power":
        site_id = sidebar.selectbox("Site", options=site_ids)
        if view == "condition_psd":
            condition_name = sidebar.selectbox("Condition", options=condition_names)
            epoch_name = sidebar.selectbox("Epoch", options=epoch_names)
        return SnapshotPlotSelection(view, site_id, condition_name, epoch_name, band_name)
    if component == "synchrony":
        if view in pair_views:
            site_id = sidebar.selectbox("Site pair", options=_pair_labels(arrays))
        else:
            site_id = sidebar.selectbox("Site", options=site_ids)
        condition_name = sidebar.selectbox("Condition", options=condition_names)
        if view in {"itpc_band_summary", "ispc_band_summary", "plv_exemplar"}:
            epoch_name = sidebar.selectbox("Epoch", options=epoch_names)
        if view in {"itpc_band_summary", "ispc_band_summary", "plv_distribution", "plv_exemplar"}:
            band_name = sidebar.selectbox("Band", options=band_names)
        return SnapshotPlotSelection(view, site_id, condition_name, epoch_name, band_name)
    if component == "spike_phase":
        site_id = sidebar.selectbox("Site", options=site_ids)
        if view in {"unit_ppc_map", "ppc_exemplar_pair"}:
            condition_name = sidebar.selectbox("Condition", options=condition_names)
        if view in {"unit_ppc_map", "population_ppc_maps", "ppc_exemplar_pair"}:
            epoch_name = sidebar.selectbox("Epoch", options=epoch_names)
        if view == "ppc_exemplar_pair":
            band_name = sidebar.selectbox("Band", options=band_names)
        return SnapshotPlotSelection(view, site_id, condition_name, epoch_name, band_name)
    raise ValueError(f"unknown snapshot component: {component!r}")


def _snapshot_component_counts(
    streamlit: StreamlitOutput,
    component: str,
    arrays: Mapping[str, np.ndarray],
    selection: SnapshotPlotSelection,
) -> None:
    """Display saved counts/eligibility warnings without recalculating metrics.

    Parameters
    ----------
    streamlit : StreamlitOutput
        Text-only display seam.
    component : str
        Selected cache component identity.
    arrays : mapping[str, numpy.ndarray]
        Already-loaded cache arrays whose counts/masks keep their documented
        axes; no numerical reduction changes a scientific result.
    selection : SnapshotPlotSelection
        Selected saved categorical axes. Counts are limited to this exact Spike
        condition/site/epoch slice, not the full component.

    Returns
    -------
    None
        Emits display-only count and eligibility text.
    """

    if component != "spike_phase":
        return
    _require_cached_arrays(
        arrays,
        {"filter_membership", "condition_membership", "unit_ids", "spike_count", "null_eligible", "reliable"},
        "Spike",
    )
    selected = select_spike_snapshot_slice(arrays, selection)
    selected_trials = _condition_trials(arrays, selected.condition_index)
    spike_counts = np.asarray(arrays["spike_count"])[
        :, selected.condition_index, selected.site_index, selected.epoch_index, :
    ]
    eligible = np.asarray(arrays["null_eligible"], dtype=bool)[
        :, selected.condition_index, selected.site_index, selected.epoch_index, :
    ]
    reliable = np.asarray(arrays["reliable"], dtype=bool)[
        :, selected.condition_index, selected.site_index, selected.epoch_index, :
    ]
    streamlit.info(
        "Trials: "
        f"{int(np.count_nonzero(selected_trials))}; units: {np.asarray(arrays['unit_ids']).size}; "
        f"spikes: {int(np.nanmin(spike_counts))}-{int(np.nanmax(spike_counts))}"
    )
    streamlit.info(f"Eligible cached unit-frequency entries: {int(np.count_nonzero(eligible))}.")
    streamlit.info(f"Reliable cached unit-frequency entries: {int(np.count_nonzero(reliable))}.")
    if np.any(~reliable):
        streamlit.warning("Unstable saved Spike reliability entries: inspect this result cautiously.")


def _render_source_enabled_summary_view(
    streamlit: object,
    *,
    session_id: str,
    session_path: Path,
    output_directory: Path,
    sites: tuple[LFPSiteConfig, ...],
    site_pairs: tuple[tuple[str, str], ...],
    sorter_paths: Mapping[str, Path],
    aligned_spike_paths: Mapping[str, Path],
    cluster_metadata_loader: Callable[[Path], object],
    channel_metadata_loader: Callable[[Path], object],
    dependencies: SummaryWebDependencies,
) -> None:
    """Render the explicit snapshot/live route while keeping work outside Streamlit.

    Parameters
    ----------
    streamlit : object
        Streamlit-compatible host with sidebar, session-state, and text/figure
        display methods. It is never passed to numerical pipeline code.
    session_id, session_path, output_directory : str, pathlib.Path, pathlib.Path
        Explicit live configuration provenance.
    sites, site_pairs : tuple
        Saved LFP site and pair definitions retaining their existing channels,
        source voltage units, and categorical identities.
    sorter_paths, aligned_spike_paths : mapping[str, pathlib.Path]
        Parent-route supplied ProbeA/ProbeB paths. Metadata is lazy and selected
        probe only.
    cluster_metadata_loader, channel_metadata_loader : callable
        Selected-sorter metadata seams; they are not called for blank/invalid
        snapshots.
    dependencies : SummaryWebDependencies
        Injected composed live actions and cache loaders/plotting seams.

    Returns
    -------
    None
        Renders a read-only snapshot or explicit bounded live controls; it does
        not write snapshots or invoke synchronous Spike/All computation.
    """

    sidebar = streamlit.sidebar
    sidebar.header("Cached LFP Summary")
    source_mode = sidebar.selectbox("Summary source", options=("snapshot", "live"))
    probe_label = sidebar.selectbox("Active population", options=("ProbeA", "ProbeB"))
    if source_mode == "snapshot":
        entered_path = sidebar.text_input("Snapshot directory", value="")
        inspection = _retained_snapshot_inspection(streamlit.session_state, entered_path)
        if inspection is None:
            streamlit.info("Snapshot path is blank.")
            return
        if not inspection.is_scientific_result:
            streamlit.error(inspection.message)
            return
        streamlit.caption(f"Snapshot source cluster directory: {inspection.source_cluster_directory}")
        component = sidebar.selectbox("Summary view", options=SUMMARY_VIEWS)
        try:
            population = _selected_population(
                session_state=streamlit.session_state, probe_label=probe_label,
                sorter_paths=sorter_paths, aligned_spike_paths=aligned_spike_paths,
                cluster_metadata_loader=cluster_metadata_loader,
                channel_metadata_loader=channel_metadata_loader,
            )
            status = snapshot_component_population_status(inspection, component, population)
            streamlit.info(status.message)
            if not status.can_plot:
                return
            cache = streamlit.session_state.get("lfp_summary_snapshot_component_cache")
            if not isinstance(cache, SnapshotComponentCache):
                cache = SnapshotComponentCache()
                streamlit.session_state["lfp_summary_snapshot_component_cache"] = cache
            arrays = cache.load(inspection, component, dependencies.load_component)
            selection = _snapshot_selection_controls(sidebar, component, arrays)
            entry = inspection.manifest["components"][component]  # type: ignore[index]
            if isinstance(entry, Mapping):
                warnings = tuple(str(warning) for warning in entry.get("warnings", ()))  # type: ignore[union-attr]
                for warning in warnings:
                    streamlit.warning(str(warning))
                if warnings:
                    streamlit.warning("Unstable status is reported by saved component warnings.")
            _snapshot_component_counts(streamlit, component, arrays, selection)
            saved_config, _ = _saved_snapshot_config(inspection, component, default_lfp_summary_config())
            figure = plot_cached_snapshot_component(component, arrays, saved_config, selection, inspection=inspection)
        except (KeyError, OSError, ValueError) as error:
            streamlit.error(str(error))
            return
        render_summary_figure(streamlit, figure)
        return

    # Live mode is explicit: selected-probe metadata is now needed to construct
    # the one population passed through the composed Power/Synchrony boundary.
    try:
        population = _selected_population(
            session_state=streamlit.session_state, probe_label=probe_label,
            sorter_paths=sorter_paths, aligned_spike_paths=aligned_spike_paths,
            cluster_metadata_loader=cluster_metadata_loader,
            channel_metadata_loader=channel_metadata_loader,
        )
        config = assemble_summary_config(
            session_id=session_id, session_path=session_path, output_directory=output_directory,
            sites=sites, site_pairs=site_pairs, unit_population=population,
            choice_filter="all", context_filter="all", excluded_trial_indices=(),
            alignment_event="choice_time", notch_enabled=True, bootstrap_count=1000,
            random_seed=0,
        )
    except ValueError as error:
        streamlit.error(str(error))
        return
    action = sidebar.selectbox("Summary action", options=SUMMARY_ACTIONS)
    if action in {"spike_phase", "all"}:
        resume_text = sidebar.text_input("Resume run directory (optional)", value="")
        resume_directory = Path(resume_text.strip()) if resume_text.strip() else None
        try:
            commands = build_launcher_handoff_commands(
                config,
                action=action,
                resume_run_directory=resume_directory,
            )
        except ValueError as error:
            streamlit.error(str(error))
            return
        command_text = "\n".join(commands.values())
        streamlit.info(f"Launcher handoff:\n{command_text}")
        return
    if not sidebar.button("Run summary action"):
        return
    outcome = dispatch_summary_action(
        action, source=SummarySource("live", None, None, True, "Live cache mode."),
        config=config, dependencies=dependencies,
        launcher_command_builder=lambda chosen, active: "\n".join(
            build_launcher_handoff_commands(active, action=chosen).values()
        ),
    )
    if outcome.launcher_command:
        streamlit.info(f"Launcher handoff:\n{outcome.launcher_command}")
    elif outcome.state != "complete":
        streamlit.error(outcome.error or "Live summary action failed.")


def render_lfp_summary_view(
    streamlit: object,
    *,
    session_id: str,
    session_path: Path,
    output_directory: Path,
    sites: tuple[LFPSiteConfig, ...],
    site_pairs: tuple[tuple[str, str], ...],
    unit_population: UnitPopulationConfig | None = None,
    dependencies: SummaryWebDependencies | None = None,
    sorter_paths: Mapping[str, Path] | None = None,
    aligned_spike_paths: Mapping[str, Path] | None = None,
    cluster_metadata_loader: Callable[[Path], object] | None = None,
    channel_metadata_loader: Callable[[Path], object] | None = None,
) -> None:
    """Render legacy controls or the explicit snapshot/live child route.

    Parameters
    ----------
    streamlit : object
        Streamlit-compatible UI host. It receives display-only cached figures.
    session_id, session_path, output_directory : str, pathlib.Path, pathlib.Path
        Explicit active-session provenance and live cache location.
    sites, site_pairs : tuple
        Existing saved site/pair definitions retaining channel axes and units.
    unit_population : UnitPopulationConfig or None
        Legacy-route active population. It is superseded by selected-probe
        construction when the additive path mappings are supplied.
    dependencies : SummaryWebDependencies or None
        Existing composed live/cache seams; absent dependencies are displayed
        as an integration error rather than fabricated.
    sorter_paths, aligned_spike_paths : mapping[str, pathlib.Path] or None
        Additive parent-route ProbeA/ProbeB identities. When provided together,
        default snapshot mode defers metadata until a valid snapshot or explicit
        live mode selects one probe.
    cluster_metadata_loader, channel_metadata_loader : callable or None
        Lazy selected-sorter metadata seams used only by the additive route.

    Returns
    -------
    None
        The legacy path keeps its public behavior. The additive route displays
        cache-only snapshots or bounded Power/Synchrony live actions, with
        Spike/All represented only by copyable launcher handoff text.
    """

    if dependencies is None:
        streamlit.error("LFP summary compute dependencies are not wired yet.")
        return
    if sorter_paths is not None or aligned_spike_paths is not None:
        if (
            sorter_paths is None
            or aligned_spike_paths is None
            or cluster_metadata_loader is None
            or channel_metadata_loader is None
        ):
            streamlit.error("Selected-probe paths and lazy metadata loaders are required.")
            return
        _render_source_enabled_summary_view(
            streamlit,
            session_id=session_id,
            session_path=session_path,
            output_directory=output_directory,
            sites=sites,
            site_pairs=site_pairs,
            sorter_paths=sorter_paths,
            aligned_spike_paths=aligned_spike_paths,
            cluster_metadata_loader=cluster_metadata_loader,
            channel_metadata_loader=channel_metadata_loader,
            dependencies=dependencies,
        )
        return
    defaults = default_summary_ui_values()
    sidebar = streamlit.sidebar
    sidebar.header("Cached LFP Summary")
    action = sidebar.selectbox("Summary action", options=SUMMARY_ACTIONS)
    view = sidebar.selectbox("Summary view", options=SUMMARY_VIEWS)
    choice_filter = sidebar.selectbox("Choice filter", options=("all", "left", "right"))
    context_filter = sidebar.selectbox("Context filter", options=("all", "left", "right"))
    excluded_rows_text = sidebar.text_input("Excluded trial rows", value="")
    alignment_event = sidebar.selectbox(
        "Alignment event",
        options=("choice_time", "start_time"),
    )
    notch_enabled = sidebar.checkbox("Apply 60 Hz notch", value=defaults.notch_enabled)
    output_rate_hz = float(
        sidebar.number_input(
            "Phase output rate (Hz)",
            value=defaults.output_rate_hz,
            min_value=1.0,
        )
    )
    bootstrap_count = int(
        sidebar.number_input(
            "Bootstrap count",
            value=defaults.bootstrap_count,
            min_value=1,
            step=1,
        )
    )
    random_seed = int(
        sidebar.number_input(
            "Random seed",
            value=0,
            min_value=0,
            step=1,
        )
    )
    allow_stale = sidebar.checkbox("Inspect stale results", value=False)
    try:
        excluded_trial_indices = parse_excluded_trial_rows(excluded_rows_text)
        config = assemble_summary_config(
            session_id=session_id,
            session_path=session_path,
            output_directory=output_directory,
            sites=sites,
            site_pairs=site_pairs,
            unit_population=unit_population,
            choice_filter=choice_filter,
            context_filter=context_filter,
            excluded_trial_indices=excluded_trial_indices,
            alignment_event=alignment_event,
            notch_enabled=notch_enabled,
            bootstrap_count=bootstrap_count,
            random_seed=random_seed,
            output_rate_hz=output_rate_hz,
        )
    except ValueError as error:
        streamlit.error(str(error))
        return
    previous_manifest: dict[str, object] | None = None
    if sidebar.button("Compute / recompute"):
        outcome = run_summary_action(
            action,
            config,
            dependencies,
            previous_manifest=previous_manifest,
        )
        previous_manifest = outcome.manifest
        if outcome.state != "complete":
            streamlit.error(outcome.error or f"{outcome.component} computation failed.")
            return
    if previous_manifest is None:
        try:
            previous_manifest = dependencies.load_manifest(config.output_directory)
        except (OSError, ValueError) as error:
            streamlit.error(f"No readable LFP summary cache: {error}")
            return
    status = assess_component_status(
        config.output_directory,
        view,
        config,
        previous_manifest,
    )
    streamlit.info(component_status_message(status))
    try:
        require_renderable_component(status, allow_stale)
        arrays = load_summary_view_arrays(view, config, previous_manifest, dependencies)
        figure = dependencies.plot_view(view, arrays, config)
    except (OSError, ValueError) as error:
        streamlit.error(str(error))
        return
    render_summary_figure(streamlit, figure)
