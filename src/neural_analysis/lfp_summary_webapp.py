"""Streamlit-facing controls for cached LFP summary inspection.

Numerical preparation and computation remain injected pipeline dependencies.
This module only assembles immutable configuration, dispatches requested actions,
loads cached component arrays, and renders already-created Matplotlib figures.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping, Protocol

import matplotlib.pyplot as plt
import numpy as np

from src.neural_analysis.lfp_summary_io import ComponentStatus
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
PlotView = Callable[[str, dict[str, dict[str, np.ndarray]]], plt.Figure]


@dataclass(frozen=True)
class SummaryWebDependencies:
    """Injected pipeline/cache/plot seams used by the summary webapp.

    Compute functions synchronously run the corresponding injected pipeline
    entry point. Cache loaders return validated numeric arrays with their axes
    and units defined by the manifest. ``plot_view`` consumes only those arrays
    and returns an unsaved Matplotlib figure.
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
        random_seed=int(random_seed),
    )
    validate_lfp_summary_config(config)
    return config


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
    component_entry = previous_manifest.get("components", {}).get(view, {})
    state = str(component_entry.get("state", "missing"))
    status = ComponentStatus(state)
    streamlit.info(component_status_message(status))
    try:
        require_renderable_component(status, allow_stale)
        arrays = load_summary_view_arrays(view, config, previous_manifest, dependencies)
        figure = dependencies.plot_view(view, arrays)
    except (OSError, ValueError) as error:
        streamlit.error(str(error))
        return
    render_summary_figure(streamlit, figure)
