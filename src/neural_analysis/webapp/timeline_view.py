"""Session timeline controls over saved, fixed population PCA products."""

from __future__ import annotations

import hashlib
from time import perf_counter

import streamlit as st

from src.neural_analysis.population.run_session_timeline import (
    find_saved_timeline,
    resolve_timeline_probe_roles,
    run_session_timeline,
)
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.session_timeline import (
    SessionTimelineResult,
    summarize_session_variance,
)
from src.neural_analysis.population.timeline_plotting import (
    EVENT_LABELS,
    EVENT_TRACKS,
    available_event_tracks,
    context_intervals,
    plot_session_timeline,
    session_time_bounds,
)
from src.neural_analysis.session_metadata import ResolvedSession


def render_session_timeline_view(st_module: object, session: ResolvedSession) -> None:
    """Render preparation, navigation, and signed-score tracks for one metadata session.

    st_module is the Streamlit interface; session contains resolved source paths.
    Computation occurs only on an explicit Prepare/Recompute click. Other
    interactions load matching saved products and construct a figure without PCA.
    """
    st_module.subheader("Session timeline and PCA")
    scope = (
        "timeline_"
        + hashlib.sha256(str(session.metadata_path).encode()).hexdigest()[:10]
    )
    roles, bin_width_s = _population_controls(st_module, session, scope)
    try:
        config = timeline_io.timeline_configuration(session, roles, bin_width_s)
    except (OSError, ValueError) as error:
        st_module.error(str(error))
        return
    saved = find_saved_timeline(session.session_root, config["configuration_id"])
    prepare = st_module.sidebar.button("Prepare", key=f"{scope}_prepare")
    recompute = st_module.sidebar.button(
        "Recompute", key=f"{scope}_recompute", disabled=saved is None
    )
    if prepare or recompute:
        try:
            with st_module.spinner("Preparing session-wide PCA..."):
                report = run_session_timeline(
                    session.metadata_path, roles, bin_width_s, rerun=recompute
                )
            saved = report.result_path
        except (OSError, ValueError) as error:
            st_module.error(f"Could not prepare timeline: {error}")
            return
    started = perf_counter()
    try:
        if saved is not None:
            info = saved.stat()
            result = load_timeline_result_cached(
                str(saved), info.st_size, info.st_mtime_ns
            )
            st_module.caption(f"Prepared result: {saved}")
        else:
            origin, choices, events = load_timeline_behavior_cached(
                str(session.metadata_path), config["configuration_id"]
            )
            unavailable = {
                role: "Not prepared" if role in roles else "No probe assigned"
                for role in ("PFC", "HPC")
            }
            result = SessionTimelineResult(
                session.session_id, origin, choices, events, {}, unavailable, config
            )
            st_module.info(
                "Behavior is available now. Select probes and click Prepare to compute and save their session PCs."
            )
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load timeline: {error}")
        return
    load_elapsed = perf_counter() - started
    state_key = f"{scope}_{config['configuration_id']}"
    visible, revision = _navigation_controls(st_module, result, state_key)
    pc_ranges, limits = _pc_controls(st_module, result, state_key)
    context_available = not context_intervals(
        result.events, session_time_bounds(result)[1]
    ).empty
    show_context = st_module.sidebar.checkbox(
        "Show active context",
        value=context_available,
        disabled=not context_available,
        key=f"{state_key}_context",
    )
    if not context_available:
        st_module.caption("Active context unavailable: no logged context-entry events.")
    elif show_context:
        st_module.caption(
            "Choice background: blue = left active; red = right active; gray = dark period. "
            "Unshaded time has unknown context."
        )
    available = available_event_tracks(result.events)
    selected_tracks = tuple(
        track
        for track in EVENT_TRACKS
        if st_module.sidebar.checkbox(
            EVENT_LABELS[track],
            value=track in available,
            disabled=track not in available,
            key=f"{state_key}_event_{track}",
        )
    )
    missing_tracks = [
        EVENT_LABELS[track] for track in EVENT_TRACKS if track not in available
    ]
    if missing_tracks:
        st_module.caption(
            "No optional events available for: " + ", ".join(missing_tracks)
        )
    bounded = st_module.sidebar.checkbox(
        "Load only visible neural interval", value=False, key=f"{state_key}_bounded"
    )
    loaded = None
    if bounded:
        low, high = session_time_bounds(result)
        duration = visible[1] - visible[0]
        loaded = (max(low, visible[0] - duration), min(high, visible[1] + duration))
        st_module.caption(
            f"Neural data loaded for {loaded[0]:.3f}-{loaded[1]:.3f} s. "
            "Use Window start or Jump to load another interval; pan within the loaded interval."
        )
    started = perf_counter()
    figure = plot_session_timeline(
        result,
        visible,
        pc_ranges,
        selected_tracks,
        loaded,
        limits,
        revision,
        show_context=show_context,
    )
    construction_elapsed = perf_counter() - started
    st_module.plotly_chart(
        figure,
        key=f"{scope}_chart",
        width="stretch",
        theme=None,
        config={"scrollZoom": False, "displaylogo": False},
    )
    st_module.caption(
        "Completed choices and unsmoothed activity within [-2, +2) s windows. "
        "Blank intervals contain no displayed neural measurements. "
        "Drag horizontally to pan; the mouse wheel scrolls the page. "
        "Display-control changes restore the Window start/duration viewport."
    )
    _render_variance_footer(st_module, result)
    with st_module.expander("Display timing"):
        st_module.write(
            f"Result/behavior load: {load_elapsed:.3f} s; figure construction: {construction_elapsed:.3f} s."
        )
        st_module.caption(
            "These timings exclude browser rendering and initial source loading/PCA preparation."
        )


def resolve_time_range(
    start_s: float, duration_s: float, bounds_s: tuple[float, float]
) -> tuple[float, float]:
    """Clamp an explicit viewport to elapsed-second session bounds without compressing time.

    Inputs are seconds and duration_s must be positive. Return (start,end);
    a requested duration longer than the session shows the full extent.
    """
    low, high = bounds_s
    if duration_s <= 0 or high <= low:
        raise ValueError("Duration and session extent must be positive.")
    duration = min(duration_s, high - low)
    start = min(max(start_s, low), high - duration)
    return float(start), float(start + duration)


def default_pc_range(component_count: int) -> tuple[int, int]:
    """Return inclusive one-based first-ten ranks for a positive available PC count."""
    if component_count < 1:
        raise ValueError("A PC display requires at least one component.")
    return 1, min(10, component_count)


@st.cache_data(show_spinner=False, max_entries=2)
def load_timeline_result_cached(
    path: str, size_bytes: int, mtime_ns: int
) -> SessionTimelineResult:
    """Load one saved NPZ, keyed by path/bytes/mtime rather than UI display settings.

    Numerical shapes/units remain those in SessionTimelineResult. Identity
    arguments bind file changes before Streamlit can return a cached result.
    """
    return timeline_io.load_timeline_result(path)


@st.cache_data(show_spinner=False, max_entries=2)
def load_timeline_behavior_cached(metadata_path: str, configuration_id: str) -> tuple:
    """Return origin/choices/events, keyed by actual source identity, without neural arrays."""
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    return timeline_io.load_timeline_behavior(load_webapp_session(metadata_path))


# UI helpers keep widget and navigation state outside numerical functions.
def _population_controls(
    st_module: object, session: ResolvedSession, scope: str
) -> tuple[dict[str, str], float]:
    """Return role/probe keys and bin width in seconds from metadata-based controls."""
    defaults = resolve_timeline_probe_roles(session, None)
    options = ["Not assigned"] + [probe.probe_id for probe in session.probes]
    roles = {}
    for role in ("PFC", "HPC"):
        default = defaults.get(role, "Not assigned")
        selected = st_module.sidebar.selectbox(
            f"{role} probe",
            options=options,
            index=options.index(default),
            key=f"{scope}_{role}_probe",
        )
        if selected != "Not assigned":
            roles[role] = selected
    width_ms = st_module.sidebar.selectbox(
        "Timeline bin width (ms)", options=[500, 100, 50, 20], key=f"{scope}_bin_ms"
    )
    st_module.sidebar.caption(
        "One population per probe: Phy good/mua; good inside-brain channels; metadata channel restrictions."
    )
    return roles, width_ms / 1000.0


def _navigation_controls(
    st_module: object, result: SessionTimelineResult, key: str
) -> tuple[tuple[float, float], int]:
    """Return explicit elapsed viewport and a revision changed only by navigation controls."""
    bounds = session_time_bounds(result)
    duration = st_module.sidebar.number_input(
        "Visible duration (s)",
        min_value=0.1,
        value=60.0,
        step=10.0,
        key=f"{key}_duration",
    )
    jump = st_module.sidebar.selectbox(
        "Jump to trial",
        options=[None] + result.choices.trial_id.tolist(),
        format_func=_format_trial,
        key=f"{key}_trial",
    )
    do_jump = st_module.sidebar.button("Jump", key=f"{key}_jump", disabled=jump is None)
    state = st_module.session_state
    start_key = f"{key}_start"
    initial = (
        float(result.choices.elapsed_s.iloc[0] - 2.0)
        if not result.choices.empty
        else bounds[0]
    )
    start = state.get(start_key, initial)
    if do_jump:
        start = (
            float(
                result.choices.loc[result.choices.trial_id == jump, "elapsed_s"].iloc[0]
            )
            - duration / 2
        )
    start, end = resolve_time_range(start, duration, bounds)
    state[start_key] = start
    max_start = max(bounds[0], bounds[1] - min(duration, bounds[1] - bounds[0]))
    start = st_module.sidebar.number_input(
        "Window start (s)",
        min_value=float(bounds[0]),
        max_value=float(max_start),
        step=1.0,
        format="%.3f",
        key=start_key,
        disabled=max_start == bounds[0],
    )
    visible = resolve_time_range(start, duration, bounds)
    previous_key = f"{key}_previous_view"
    revision_key = f"{key}_revision"
    if state.get(previous_key) != visible:
        state[revision_key] = state.get(revision_key, 0) + 1
        state[previous_key] = visible
    return visible, state.get(revision_key, 0)


def _pc_controls(
    st_module: object, result: SessionTimelineResult, key: str
) -> tuple[dict, dict]:
    """Return display-only inclusive rank ranges and optional symmetric score limits."""
    ranges = {}
    limits = {}
    for role, regional in result.regions.items():
        count = regional.scores.shape[2]
        if count == 1:
            ranges[role] = (1, 1)
            st_module.sidebar.caption(f"{role}: PC1 only")
        else:
            ranges[role] = st_module.sidebar.slider(
                f"{role} PC ranks",
                min_value=1,
                max_value=count,
                value=default_pc_range(count),
                key=f"{key}_{role}_pcs",
            )
        manual = st_module.sidebar.checkbox(
            f"{role} manual color range", value=False, key=f"{key}_{role}_manual"
        )
        if manual:
            limit = st_module.sidebar.number_input(
                f"{role} symmetric score limit",
                min_value=1e-12,
                value=regional.color_limit,
                key=f"{key}_{role}_limit",
            )
            limits[role] = limit
            st_module.caption(
                f"{role} manual display limits: {-limit:g} to {limit:g}; scores beyond these limits saturate."
            )
    return ranges, limits


def _render_variance_footer(st_module: object, result: SessionTimelineResult) -> None:
    """Render global leading-PC variance counts and population eligibility, never viewport metrics."""
    width = result.meta.get("bin_width_s", 0.5) * 1000
    st_module.markdown(
        f"**Variance coverage - all eligible +/-2 s choice windows, {width:g}-ms bins**"
    )
    for role in ("PFC", "HPC"):
        if role not in result.regions:
            st_module.write(
                f"{role}: unavailable - {result.unavailable.get(role, 'No data')}"
            )
            continue
        regional = result.regions[role]
        counts = summarize_session_variance(
            regional.transform.explained_variance, regional.total_session_variance
        )
        labels = [
            f"{int(q * 100)}% = {count} PCs"
            if count is not None
            else f"{int(q * 100)}% not reached by retained basis"
            for q, count in counts.items()
        ]
        st_module.write(f"{role} ({regional.probe_id}): " + " | ".join(labels))
        st_module.caption(
            f"{len(regional.transform.retained_unit_ids)} usable units; "
            f"{len(regional.transform.omitted_unit_ids)} constant units omitted; "
            f"{len(regional.choice_positions)}/{len(result.choices)} complete choice windows. "
            f"Default color limits +/-{regional.color_limit:g}; no score rescaling."
        )
    st_module.caption(
        "Separate session-wide regional bases; no whitening or post-PCA scaling. "
        "PC signs and identities apply to this configuration; PCs describe pooled choice-window activity."
    )


def _format_trial(value: object) -> str:
    """Format an original scalar trial identity, with None as the navigation placeholder."""
    return "Select trial..." if value is None else str(value)
