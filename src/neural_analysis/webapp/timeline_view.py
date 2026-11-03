"""Session timeline controls over saved, fixed population PCA products."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis import session_speed
from src.mouse_behavior_preprocessing.treadmill_speed import TreadmillSpeed
from src.neural_analysis.population.run_session_timeline import (
    find_saved_timeline,
    resolve_timeline_probe_roles,
    run_session_timeline,
)
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.run_window_variance import (
    find_saved_window_variance,
    run_window_variance,
)
from src.neural_analysis.population.window_variance import SessionWindowVariance
from src.neural_analysis.population.window_variance_io import load_window_variance
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
from src.neural_analysis.webapp.timeline_chart import render_timeline_chart


def render_session_timeline_view(st_module: object, session: ResolvedSession) -> None:
    """Render preparation, navigation, and signed-score tracks for one metadata session.

    st_module is the Streamlit interface; session contains resolved source paths.
    PCA occurs only on an explicit Prepare/Recompute click. Other interactions
    load saved products; optional per-PC z-scoring affects the display only.
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
            if "reward" not in result.choices:
                _, current_choices, _ = load_timeline_behavior_cached(
                    str(session.metadata_path), config["configuration_id"]
                )
                # Legacy archives lack reward flags; verify identity before attaching them.
                identity = ["row_position", "trial_id", "choice_time", "action"]
                if not result.choices[identity].equals(current_choices[identity]):
                    raise ValueError(
                        "Saved choices do not match the current trial table for reward labels."
                    )
                result = replace(
                    result,
                    choices=result.choices.assign(
                        reward=current_choices.reward.to_numpy()
                    ),
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
    summary = _window_summary_controls(st_module, saved, state_key)
    speed = _speed_controls(st_module, session, scope)
    visible, revision = _navigation_controls(st_module, result, state_key)
    zscore_pcs = st_module.sidebar.checkbox(
        "Z-score each PC",
        value=False,
        disabled=not result.regions,
        key=f"{state_key}_zscore_pcs",
        help="Use each PC's mean and population SD across all prepared choice-window bins in this session.",
    )
    pc_ranges, limits = _pc_controls(st_module, result, state_key, zscore_pcs)
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
        "Load only visible neural interval",
        value=bin_width_s == 0.02,
        key=f"{state_key}_bounded",
        help="Defaults on at 20 ms because full-session heatmap payloads can exceed browser transport limits. Window counts still use the whole session.",
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
        zscore_pcs=zscore_pcs,
        window_variance=summary.regions if summary is not None else None,
        speed=speed,
    )
    construction_elapsed = perf_counter() - started
    render_timeline_chart(figure, key=f"{scope}_chart")
    st_module.caption(
        "Completed choices and unsmoothed activity within [-2, +2) s windows. "
        "Choices: large filled = rewarded; small open = unrewarded; gray crosses = unknown reward. "
        "Blank intervals contain no displayed neural measurements. "
        "Drag horizontally to pan; the mouse wheel scrolls the page. "
        "PC and display changes preserve the current time range; "
        "Window start/duration and Jump request a new range."
    )
    _render_variance_footer(st_module, result, zscore_pcs)
    if summary is not None:
        _render_window_readout(
            st_module, summary, st_module.session_state.get(f"{state_key}_trial")
        )
    with st_module.expander("Display timing"):
        st_module.write(
            f"Result/behavior load: {load_elapsed:.3f} s; figure construction: {construction_elapsed:.3f} s."
        )
        st_module.caption(
            "These timings exclude browser rendering and initial source loading/PCA preparation."
        )


def render_timeline_navigation(
    st_module: object, result: SessionTimelineResult, key: str,
) -> tuple[tuple[float, float], int]:
    """Render existing elapsed-time/jump controls for a saved timeline result.

    Inputs are Streamlit's widget interface, a saved result with UTC origin and
    elapsed-second choice identities, and a widget-key string. Return the
    elapsed-second (start,end) viewport and integer navigation revision. No
    sources or neural arrays are loaded or fitted by these controls.
    """
    return _navigation_controls(st_module, result, key)


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
    """Return UTC-second origin, choices with reward flags, and source events.

    metadata_path identifies a session; configuration_id binds its current
    source files. Choices retain reward=1/0/NaN (rewarded/unrewarded/unknown).
    The cached tuple contains no neural arrays.
    """
    from src.neural_analysis.webapp.session_inputs import load_webapp_session

    return timeline_io.load_timeline_behavior(load_webapp_session(metadata_path))


# UI helpers keep widget and navigation state outside numerical functions.
@st.cache_data(show_spinner=False, max_entries=4)
def load_speed_result_cached(
    path: str, size_bytes: int, mtime_ns: int
) -> TreadmillSpeed:
    """Load signed-mm/s/UTC arrays once per exact NPZ path/stat; no raw reads or decoding."""
    return session_speed.load_speed_result(path)


def _speed_controls(
    st_module: object, session: ResolvedSession, scope: str
) -> TreadmillSpeed | None:
    """Return saved speed only when shown; decode solely on explicit prepare clicks.

    session supplies the optional log path. scope keys speed controls independently
    of neural probes/bins. A fixed status slot prevents chart remounts on toggles.
    Missing, stale, or invalid speed affects only this optional row.
    """
    status = st_module.empty()
    saved = None
    error = None
    try:
        config = session_speed.speed_configuration(session)
        saved = session_speed.find_saved_speed(
            session.session_root, config["configuration_id"]
        )
    except (OSError, ValueError) as failure:
        error = str(failure)
    prepare = st_module.sidebar.button(
        "Prepare speed", key=f"{scope}_prepare_speed", disabled=error is not None
    )
    rerun = st_module.sidebar.button(
        "Recompute speed", key=f"{scope}_recompute_speed", disabled=saved is None
    )
    show_key = f"{scope}_show_speed"
    if prepare or rerun:
        try:
            with st_module.spinner("Preparing IRIG-aligned signed speed..."):
                saved = session_speed.run_session_speed(
                    session.metadata_path, rerun=rerun
                ).result_path
            st_module.session_state[show_key] = True
        except (OSError, ValueError) as failure:
            error = f"Could not prepare speed: {failure}"
    if saved is None:
        st_module.session_state[show_key] = False
    show = st_module.sidebar.checkbox(
        "Show speed", value=saved is not None, disabled=saved is None, key=show_key
    )
    if error is not None:
        status.caption(f"Speed unavailable: {error}")
        return None
    if saved is None:
        status.caption(
            "Speed not prepared for this source/version. Click Prepare speed; neural preparation is independent."
        )
        return None
    status.caption(
        "Speed: signed logged mm/s, aligned by Teensy timing and IRIG; each value is held until the next update. Speed between choices is measured behavior."
    )
    if not show:
        return None
    try:
        stat = saved.stat()
        return load_speed_result_cached(str(saved), stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, KeyError) as failure:
        status.caption(f"Speed unavailable: {failure}")
        return None


@st.cache_data(show_spinner=False, max_entries=4)
def load_window_summary_cached(
    path: str, size_bytes: int, mtime_ns: int
) -> SessionWindowVariance:
    """Load a saved summary keyed by exact path/stat; no computation or raw loading.

    Return the documented session summary with unchanged window/PC axes/units.
    """
    return load_window_variance(path)


def _window_summary_controls(
    st_module: object, stage1_path: Path | None, key: str
) -> SessionWindowVariance | None:
    """Prepare only on explicit clicks; return saved summary when its display is enabled.

    stage1_path is the exact loaded archive (None if unprepared); key scopes UI
    state. Missing/stale summaries do not trigger automatic analysis or writes.
    """
    saved = find_saved_window_variance(stage1_path) if stage1_path is not None else None
    prepare = st_module.sidebar.button(
        "Prepare window summary",
        disabled=stage1_path is None,
        key=f"{key}_prepare_summary",
    )
    rerun = st_module.sidebar.button(
        "Recompute window summary", disabled=saved is None, key=f"{key}_rerun_summary"
    )
    try:
        if prepare or rerun:
            with st_module.spinner("Summarizing saved fixed-axis window variance..."):
                saved = run_window_variance(stage1_path, rerun=rerun).result_path
        if saved is None:
            st_module.caption(
                "Prepare window summary to add fixed-axis 95% counts from the saved session PCs."
            )
            return None
        enabled = st_module.sidebar.checkbox(
            "Show fixed-axis window counts", value=True, key=f"{key}_show_summary"
        )
        # Keep the chart's Streamlit position stable when metric rows are hidden.
        st_module.caption(
            "Fixed-axis counts use original session scores and all retained axes. "
            "They measure variation around each window's mean, not sustained mean displacement or local PCA dimensionality."
        )
        if not enabled:
            return None
        info = saved.stat()
        summary = load_window_summary_cached(str(saved), info.st_size, info.st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load window summary: {error}")
        return None
    return summary


def _render_window_readout(
    st_module: object, summary: SessionWindowVariance, trial_id: object
) -> None:
    """Display one original trial's sorted fixed-PC contributions from saved arrays.

    trial_id is the existing Jump selector value or None. Fractions are unitless;
    total variance is the sum of standardized-unit sample variances (ddof=1).
    Missing regional windows and invalid counts are explicitly described.
    """
    st_module.markdown("**Selected choice - fixed-session-axis contributions**")
    if trial_id is None:
        st_module.caption(
            "Select a trial with Jump to trial to inspect its contributions; Jump also centers the timeline."
        )
        return
    positions = np.flatnonzero(summary.choices.trial_id.to_numpy() == trial_id)
    if not len(positions):
        return
    for role, region in summary.regions.items():
        matches = np.flatnonzero(region.choice_positions == positions[0])
        if not len(matches):
            st_module.write(f"{role}: no complete neural window for Trial {trial_id}.")
            continue
        index = int(matches[0])
        count = region.counts95[index]
        status = region.status[index]
        label = f"{int(count)} PCs" if np.isfinite(count) else "unavailable"
        st_module.markdown(f"**Trial {trial_id} - {role}: {label} ({status})**")
        st_module.caption(
            f"Total standardized-unit sample variance: {region.total_variance[index]:.6g}; "
            f"{len(region.bin_edges_s) - 1} bins; {len(region.unit_ids)} units; "
            f"{'complete' if region.basis_complete else 'restricted'} basis "
            f"({region.pc_ids.shape[1]}/{region.full_pc_count} session axes)."
        )
        if status == "not_reached":
            st_module.write("95% not reached by retained basis.")
        elif status == "restricted":
            st_module.write(
                "Restricted count; the exact minimum over all session axes is unavailable."
            )
        elif status != "ok":
            continue
        stop = int(count) if np.isfinite(count) else region.pc_ids.shape[1]
        st_module.dataframe(
            pd.DataFrame(
                {
                    "Session PC": region.pc_ids[index, :stop],
                    "Window variance fraction": region.fractions[index, :stop],
                    "Cumulative fraction": region.cumulative[index, :stop],
                }
            ),
            hide_index=True,
        )


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
    previous_key = f"{key}_previous_navigation"
    revision_key = f"{key}_revision"
    navigation = (float(start), float(duration))
    if do_jump or state.get(previous_key) != navigation:
        state[revision_key] = state.get(revision_key, 0) + 1
        state[previous_key] = navigation
    return visible, state.get(revision_key, 0)


def _pc_controls(
    st_module: object, result: SessionTimelineResult, key: str, zscore_pcs: bool = False
) -> tuple[dict, dict]:
    """Return region/rank pairs and symmetric limits in raw-score or per-PC SD units.

    result contains saved (window,bin,PC) scores; key scopes widget state.
    zscore_pcs selects PC-SD units and keeps manual limits separate from raw
    score limits. Neither returned dictionary changes the saved analysis.
    """
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
            label = (
                f"{role} symmetric z-score limit (SD)"
                if zscore_pcs
                else f"{role} symmetric score limit"
            )
            mode = "zscore" if zscore_pcs else "raw"
            limit = st_module.sidebar.number_input(
                label,
                min_value=1e-12,
                value=3.0 if zscore_pcs else regional.color_limit,
                key=f"{key}_{role}_{mode}_limit",
            )
            limits[role] = limit
            units = " PC SD" if zscore_pcs else ""
            st_module.caption(
                f"{role} manual display limits: {-limit:g} to {limit:g}{units}; values beyond these limits saturate."
            )
    return ranges, limits


def _render_variance_footer(
    st_module: object, result: SessionTimelineResult, zscore_pcs: bool = False
) -> None:
    """Render original PCA variance counts, with accurate display-scaling descriptions.

    result is the saved session analysis; zscore_pcs describes displayed PC-SD
    units only. Counts always use the original full-session variance spectrum.
    """
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
        scaling = (
            "Heatmap values are per-PC session z-scores (PC SD)."
            if zscore_pcs
            else f"Default color limits +/-{regional.color_limit:g}; no score rescaling."
        )
        st_module.caption(
            f"{len(regional.transform.retained_unit_ids)} usable units; "
            f"{len(regional.transform.omitted_unit_ids)} constant units omitted; "
            f"{len(regional.choice_positions)}/{len(result.choices)} complete choice windows. "
            + scaling
        )
    st_module.caption(
        "Separate session-wide regional bases; saved PCA scores are unwhitened. "
        "PC signs and identities apply to this configuration; PCs describe pooled choice-window activity."
    )


def _format_trial(value: object) -> str:
    """Format an original scalar trial identity, with None as the navigation placeholder."""
    return "Select trial..." if value is None else str(value)
