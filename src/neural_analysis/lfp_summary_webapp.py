"""Streamlit-facing controls for cached LFP summary inspection.

Numerical preparation and computation remain injected pipeline dependencies.
This module only assembles immutable configuration, dispatches requested actions,
loads cached component arrays, and renders already-created Matplotlib figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Protocol

import matplotlib.pyplot as plt
import numpy as np

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
    default_lfp_summary_config,
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
    ``"complete"`` or ``"failed"``. The manifest contains JSON-compatible
    cache metadata only; numerical arrays remain in component NPZ files.
    """

    component: str
    state: str
    manifest: dict[str, object] | None
    error: str | None


ComputeAction = Callable[[LFPSummaryConfig], SummaryActionResult]
LoadManifest = Callable[[Path], dict[str, object]]
LoadComponent = Callable[[Path, dict[str, object], str], dict[str, np.ndarray]]
PlotView = Callable[[str, dict[str, dict[str, np.ndarray]], LFPSummaryConfig], plt.Figure]


@dataclass(frozen=True)
class SummaryWebDependencies:
    """Injected pipeline/cache/plot seams used by the summary webapp.

    Compute functions synchronously run the corresponding injected pipeline
    entry point. Cache loaders return validated numeric arrays with their axes
    and units defined by the manifest. ``plot_view`` consumes only those arrays
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
    """Create live Power compute/cache dependencies and explicit unavailable actions.

    Returns
    -------
    SummaryWebDependencies
        Power delegates to the production atomic pipeline. Cached views load
        NPZ arrays only. Synchrony, Spike-phase, and Compute All report their
        unavailable production status until their runtime bridges are supplied.
    """
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
        if view != "power":
            raise ValueError(f"{view} cached plotting is unavailable")
        return _plot_cached_power(arrays["power"], config)

    return SummaryWebDependencies(
        compute_power=compute_production_power,
        compute_synchrony=unavailable("synchrony"),
        compute_spike_phase=unavailable("spike_phase"),
        compute_all=unavailable("all"),
        load_manifest=load_manifest,
        load_component=load_component_arrays,
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
    effective_count = arrays["condition_effective_trial_count"]
    site_ids = arrays["site_ids"]
    voltage_units = arrays["site_voltage_units"]
    epoch_names = arrays["epoch_names"]
    if (
        frequency_hz.ndim != 1
        or normalized_psd_db.ndim != 4
        or condition_names.ndim != 1
        or membership.ndim != 2
        or effective_count.ndim != 2
        or site_ids.ndim != 1
        or voltage_units.ndim != 1
        or epoch_names.ndim != 1
    ):
        raise ValueError("cached Power arrays have incompatible axes")
    site_count, trial_count, epoch_count, frequency_count = normalized_psd_db.shape
    condition_count = condition_names.size
    if (
        frequency_hz.size != frequency_count
        or membership.shape != (trial_count, condition_count)
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
    condition_psd_db = np.full(
        (condition_count, trial_count, frequency_count),
        np.nan,
        dtype=normalized_psd_db.dtype,
    )
    for condition_index in range(condition_count):
        selected_trials = membership[:, condition_index]
        if selected_trials.dtype != np.dtype(bool):
            raise ValueError("cached Power condition_membership must be boolean")
        condition_psd_db[condition_index, selected_trials] = normalized_psd_db[
            0,
            selected_trials,
            whole_epoch_index,
        ]
    gamma_band = next(band for band in config.power.bands if band.name == "gamma")
    gamma_exclusion_hz = gamma_band.excluded_intervals_hz[0]
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
        frequency_hz,
        condition_psd_db,
        tuple(str(name) for name in condition_names),
        effective_count[:, 0],
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
    """Load only the component NPZ file required for one selected summary view.

    ``view`` selects power, synchrony, or spike_phase. Returned arrays preserve
    their cache-defined axes and physical units; no raw LFP or wavelet data is
    opened or reconstructed by this helper.
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
) -> None:
    """Render synchronous cache controls using active session inputs and seams.

    This boundary presents action and view selectors for an active session. It
    does not prepare raw data or compute numerics. If production payload seams
    are unavailable, it reports that explicit integration gap instead of
    fabricating a second computational or cache path.
    """

    if dependencies is None:
        streamlit.error("LFP summary compute dependencies are not wired yet.")
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
