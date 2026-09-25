"""Streamlit-facing controls for cached LFP summary inspection.

Numerical preparation and computation remain injected pipeline dependencies.
This module only assembles immutable configuration, dispatches requested actions,
loads cached component arrays, and renders already-created Matplotlib figures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from shlex import quote
import stat
from typing import Callable, Mapping, MutableMapping, Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp_summary import pipeline as lfp_summary_pipeline
from src.neural_analysis.lfp_summary import plotting as lfp_summary_plotting
from src.neural_analysis.lfp_summary import runtime as lfp_summary_runtime
from src.neural_analysis.lfp_summary.cache import (
    ComponentStatus,
    assess_component_status,
    load_component_arrays,
)
from src.neural_analysis.lfp_summary.models import (
    LFPSiteConfig,
    LFPSummaryConfig,
    PPCExecutionConfig,
    TrialFilterConfig,
    UnitPopulationConfig,
    ProgressEvent,
    default_lfp_summary_config,
    lfp_summary_config_from_json,
    validate_lfp_summary_config,
)
from src.neural_analysis.lfp_summary.snapshot import (
    SnapshotComponentCache,
    SnapshotInspection,
    SnapshotPlotSelection,
    SnapshotPopulationStatus,
    SpikeSnapshotSlice,
    SummarySource,
    _cached_axis_values,
    _condition_trials,
    _named_index,
    _pair_labels,
    _plot_cached_power_component,
    _plot_cached_spike_component,
    _plot_cached_synchrony_component,
    _plot_population_ppc_maps,
    _require_cached_arrays,
    _retained_band_frequency_indices,
    _saved_snapshot_config,
    _sha256_file,
    _snapshot_failure,
    _snapshot_plot_context,
    _snapshot_stat_identity,
    _spike_band_reliability,
    _spike_exemplar_panel,
    _synchrony_entity,
    plot_cached_snapshot_component,
    resolve_summary_source,
    select_spike_snapshot_slice,
    snapshot_component_population_status,
    snapshot_component_source_value_semantics,
    validate_cache_snapshot,
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


def build_active_summary_population(
    *,
    probe_label: str,
    sorter_path: Path,
    aligned_spike_path: Path,
    cluster_metadata: object,
    channel_metadata: object,
) -> UnitPopulationConfig:
    """Build exactly one active metadata-selected population from supplied paths.

    Parameters
    ----------
    probe_label : str
        Nonempty stable hardware/probe identifier from session metadata.
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

    if not isinstance(probe_label, str) or not probe_label.strip():
        raise ValueError("probe_label must be a nonempty string")
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
    trial_table_path: Path | None = None,
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
            Path(trial_table_path)
            if trial_table_path is not None
            else Path(session_path) / "processed" / f"{session_id}_augmented_trials.csv"
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
        """Load one live directory component or one exact snapshot NPZ path.

        Parameters
        ----------
        output_directory : pathlib.Path
            Either a configured live cache directory containing final component
            NPZ files, or an exact selected final ``.npz`` path supplied by a
            receipt-validated snapshot inspection. Filesystem path units are
            preserved; the path is neither resolved nor rewritten.
        manifest : dict[str, object]
            Current cache metadata; numerical arrays retain its saved axes.
        component : str
            One summary component. A directory input selects
            ``output_directory/component.npz``. An NPZ input must have exactly
            that filename and is passed through unchanged.

        Returns
        -------
        dict[str, numpy.ndarray]
            Loader-validated cached arrays with unchanged shapes and units.

        Raises
        ------
        ValueError
            If an explicit NPZ path does not have the expected selected
            component filename.
        """

        expected_filename = f"{component}.npz"
        if output_directory.suffix == ".npz":
            if output_directory.name != expected_filename:
                raise ValueError(
                    f"Selected component path must be {expected_filename}: {output_directory}"
                )
            component_path = output_directory
        else:
            component_path = output_directory / expected_filename
        return load_component_arrays(component_path, manifest, component)

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
    trial_table_path: Path | None,
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
        Parent-route supplied stable probe IDs and paths. Metadata is lazy and
        selected-probe only.
    trial_table_path : pathlib.Path or None
        Explicit metadata-defined trial CSV. ``None`` retains the legacy
        session-ID-derived table path.
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
    probe_label = sidebar.selectbox("Active population", options=tuple(sorter_paths))
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
            streamlit.caption(
                f"Source value semantics: {snapshot_component_source_value_semantics(inspection, component)}"
            )
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
            trial_table_path=trial_table_path,
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
    trial_table_path: Path | None = None,
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
    trial_table_path : pathlib.Path or None
        Explicit trial CSV for metadata-driven live configuration. ``None``
        retains the legacy derived path.

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
            trial_table_path=trial_table_path,
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
