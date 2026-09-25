"""Immutable cache-snapshot validation, selection, and plotting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
from typing import Callable, Mapping, MutableMapping

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp_summary import plotting as lfp_summary_plotting
from src.neural_analysis.lfp_summary.models import (
    LFPSummaryConfig,
    PPCExecutionConfig,
    UnitPopulationConfig,
    default_lfp_summary_config,
    lfp_summary_config_from_json,
)


SUMMARY_COMPONENTS = ("power", "synchrony", "spike_phase")
LoadComponent = Callable[[Path, dict[str, object], str], dict[str, np.ndarray]]

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
    """Return one component's saved configuration for cache-only labels.

    Parameters
    ----------
    inspection : SnapshotInspection or None
        Receipt-validated manifest provenance. ``None`` preserves the legacy
        adapter path using ``fallback`` only.
    component : str
        Final cached component whose configuration snapshot is required.
    fallback : LFPSummaryConfig
        Existing live configuration used solely by old callers without a
        snapshot inspection; it does not fill or override snapshot metadata.

    Returns
    -------
    tuple[LFPSummaryConfig, str or None]
        Deserialized saved config and the retained cluster-directory provenance.
        For an otherwise complete legacy snapshot, only an absent
        execution-only ``ppc_execution`` mapping receives the current
        ``PPCExecutionConfig`` defaults. Every present saved field is retained,
        and any other missing or malformed field remains fail-closed.
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
    saved_configuration = dict(saved)
    if "ppc_execution" not in saved_configuration:
        saved_configuration["ppc_execution"] = asdict(PPCExecutionConfig())
    try:
        return (
            lfp_summary_config_from_json(json.dumps(saved_configuration)),
            inspection.source_cluster_directory,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("snapshot component saved configuration is malformed") from error


def snapshot_component_source_value_semantics(
    inspection: SnapshotInspection,
    component: str,
) -> dict[str, str]:
    """Return receipt-validated per-site source semantics for cache-only inspection.

    Parameters
    ----------
    inspection : SnapshotInspection
        A valid committed snapshot inspection. Invalid or unvalidated input is
        rejected before any historical fallback is considered.
    component : str
        Selected final component whose saved configuration/provenance is shown.

    Returns
    -------
    dict[str, str]
        Receipt-validated persisted per-site semantics with the exact saved
        Open Ephys site-id keys. Historical entries with an absent semantics
        field map only saved Open Ephys site ids to ``"legacy-unscaled"``;
        all-SpikeGLX snapshots remain an empty mapping.

    Raises
    ------
    ValueError
        If the inspection or saved configuration is invalid, a present mapping
        has missing, extra, non-string, empty, or non-current adapter-version
        values, or an all-SpikeGLX snapshot stores a nonempty mapping.
    """
    if not inspection.is_scientific_result or not isinstance(inspection.manifest, Mapping):
        raise ValueError("snapshot semantics require a valid inspection")
    components = inspection.manifest.get("components")
    entry = components.get(component) if isinstance(components, Mapping) else None
    if not isinstance(entry, Mapping):
        raise ValueError("snapshot component provenance is unavailable")
    saved_config, _ = _saved_snapshot_config(inspection, component, default_lfp_summary_config())
    expected_site_ids = tuple(
        site.stable_id
        for site in saved_config.sites
        if site.acquisition_format == "open_ephys"
    )
    if "source_value_semantics" in entry:
        persisted = entry["source_value_semantics"]
        if not isinstance(persisted, Mapping):
            raise ValueError("snapshot source value semantics is malformed")
        if set(persisted) != set(expected_site_ids):
            raise ValueError("snapshot source value semantics site ids do not match saved Open Ephys sites")
        if not expected_site_ids:
            return {}
        values = tuple(persisted[site_id] for site_id in expected_site_ids)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("snapshot source value semantics values must be nonempty strings")
        if any(value != lfp_loading.OPEN_EPHYS_AFFINE_UV_SEMANTICS for value in values):
            raise ValueError(
                "snapshot source value semantics must equal the current Open Ephys affine-uV version"
            )
        return {site_id: persisted[site_id] for site_id in expected_site_ids}
    return {
        site.stable_id: "legacy-unscaled"
        for site in saved_config.sites
        if site.acquisition_format == "open_ephys"
    }


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
    """Select saved Hz bins within one band and outside inclusive exclusions.

    Parameters
    ----------
    frequency_hz : numpy.ndarray
        Finite one-dimensional saved frequency coordinates in Hz.
    config : LFPSummaryConfig
        Immutable saved phase-band bounds and inclusively excluded intervals in
        Hz, matching the established report adapters.
    band_name : str
        Categorical saved band label.

    Returns
    -------
    numpy.ndarray
        Nonempty integer positions into ``frequency_hz``. The outer bounds are
        inclusive and each configured exclusion interval removes both endpoints.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    if frequency.ndim != 1 or not np.all(np.isfinite(frequency)):
        raise ValueError("cached frequency_hz must be a finite one-dimensional axis")
    band = next((item for item in config.phase.bands if item.name == band_name), None)
    if band is None:
        raise ValueError(f"saved phase configuration lacks band {band_name!r}")
    retained = (frequency >= band.lower_hz) & (frequency <= band.upper_hz)
    for low_hz, high_hz in band.excluded_intervals_hz:
        retained &= ~((frequency >= low_hz) & (frequency <= high_hz))
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
    site_index = _named_index(arrays["site_ids"], selection.site_id, "site_ids")
    condition_names = _cached_axis_values(arrays, "condition_names")
    epoch_names = _cached_axis_values(arrays, "epoch_names")
    band_names = _cached_axis_values(arrays, "band_names")
    if selection.view == "condition_psd":
        condition_index = _named_index(arrays["condition_names"], selection.condition_name, "condition_names")
        epoch_index = _named_index(arrays["epoch_names"], selection.epoch_name, "epoch_names")
        selected_trials = _condition_trials(arrays, condition_index)
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
        membership = np.asarray(arrays["condition_membership"], dtype=bool)
        filtered = np.asarray(arrays["filter_membership"], dtype=bool)
        if membership.shape != (values.shape[1], len(condition_names)) or filtered.shape != (values.shape[1],):
            raise ValueError("cached Power condition membership axes are invalid")
        selected_values = np.full(
            (len(condition_names), values.shape[1], values.shape[2], values.shape[3]),
            np.nan,
        )
        for saved_condition_index in range(len(condition_names)):
            rows = membership[:, saved_condition_index] & filtered
            selected_values[saved_condition_index, rows] = values[site_index, rows]
        counts = np.asarray(arrays["condition_effective_trial_count"])
        if counts.shape != (len(condition_names), values.shape[0]):
            raise ValueError("cached Power effective-count axes are invalid")
        epoch_counts = np.repeat(counts[:, site_index, np.newaxis], len(epoch_names), axis=1)
        figure, _ = lfp_summary_plotting.plot_band_power_summary(
            selected_values,
            condition_names,
            epoch_names,
            band_names,
            epoch_counts,
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
            observed_estimates=np.array(
                (
                    np.asarray(arrays[f"{metric_prefix}_band_mean"])[
                        condition_index,
                        entity_index,
                        epoch_index,
                        band_index,
                    ],
                )
            ),
            bootstrap_quantiles=np.array(
                (
                    (
                        np.asarray(arrays[f"{metric_prefix}_ci_low"])[
                            condition_index,
                            entity_index,
                            epoch_index,
                            band_index,
                        ],
                    ),
                    (
                        np.asarray(arrays[f"{metric_prefix}_bootstrap_q25"])[
                            condition_index,
                            entity_index,
                            epoch_index,
                            band_index,
                        ],
                    ),
                    (
                        np.asarray(arrays[f"{metric_prefix}_bootstrap_median"])[
                            condition_index,
                            entity_index,
                            epoch_index,
                            band_index,
                        ],
                    ),
                    (
                        np.asarray(arrays[f"{metric_prefix}_bootstrap_q75"])[
                            condition_index,
                            entity_index,
                            epoch_index,
                            band_index,
                        ],
                    ),
                    (
                        np.asarray(arrays[f"{metric_prefix}_ci_high"])[
                            condition_index,
                            entity_index,
                            epoch_index,
                            band_index,
                        ],
                    ),
                )
            ),
            selected_trial_counts=np.array(
                (
                    np.asarray(arrays[f"{metric_prefix}_band_trial_count"])[
                        condition_index,
                        entity_index,
                        epoch_index,
                        band_index,
                    ],
                ),
                dtype=np.int64,
            ),
            labels=(selection.condition_name,),
            band_name=selection.band_name,
            epoch_name=selection.epoch_name,
            metric_name=f"{metric_name} {entity_label}",
            bootstrap_count=config.phase.bootstrap_count,
            context=context,
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
