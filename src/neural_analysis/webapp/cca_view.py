"""Explicit CCA preparation and source-independent saved-session viewing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from pickle import UnpicklingError
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis.population import session_cca_io as cca_io
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.population.cca_plotting import plot_session_cca
from src.neural_analysis.population.run_session_cca import run_session_cca
from src.neural_analysis.population.run_session_timeline import resolve_timeline_probe_roles
from src.neural_analysis.population.session_timeline import choice_bin_edges
from src.neural_analysis.population.timeline_plotting import (
    EVENT_LABELS, EVENT_TRACKS, available_event_tracks, context_intervals, session_time_bounds,
)
from src.neural_analysis.session_metadata import ResolvedSession, resolve_probe_sources
from src.neural_analysis.webapp.timeline_chart import render_timeline_chart
from src.neural_analysis.webapp.timeline_view import render_timeline_navigation


def render_session_cca_view(st_module: object, session: ResolvedSession) -> None:
    """Render one metadata session's archived CCA or behavior-only navigation.

    st_module is Streamlit's interface; session supplies resolved paths, not
    loaded arrays. Saved records retain (W,B,K) canonical-SD scores and UTC/
    elapsed-second behavior. Discovery/loading precedes preparation controls;
    only explicit Prepare/Recompute clicks inspect current scientific inputs
    and fit. Return None. Display controls do not modify saved files or scores.
    """
    st_module.subheader("Session timeline and CCA")
    scope = "cca_" + hashlib.sha256(str(session.metadata_path).encode()).hexdigest()[:10]
    # Variable controls/status stay in one block so the chart keeps its browser identity.
    with st_module.container(key=f"{scope}_controls"):
        headers = cca_io.discover_cca_results(session.session_root)
        saved, result = _saved_result_controls(st_module, headers, scope, session.session_root)
        roles, bin_width_s = _preparation_controls(st_module, session, scope, saved is not None)
        if result is None:
            try:
                origin, choices, events = timeline_io.load_timeline_behavior(session)
            except (OSError, ValueError) as error:
                st_module.error(f"No saved CCA or usable behavior: {error}")
                return
            empty = np.array([], dtype=int)
            result = cca_io.SessionCCAResult(
                session.session_id, origin, choices, events, empty,
                {"PFC": empty, "HPC": empty}, choice_bin_edges(bin_width_s), roles,
                None, "Not prepared; assign distinct probes and click Prepare",
                {"bin_width_s": bin_width_s},
            )
            st_module.info("Behavior is available. Assign PFC/HPC probes and click Prepare for paired CCA.")
        else:
            saved_ms = int(round(result.meta["bin_width_s"] * 1000))
            st_module.caption(f"Saved CCA: {saved}; {saved_ms} ms. Preparation settings apply only to the next explicit action.")
        navigation_key = f"{scope}_{result.session_id}_{result.time_origin_s}_navigation"
        viewport, revision = render_timeline_navigation(st_module, result, navigation_key)
        pair_range, pair, traces, limit = _pair_controls(st_module, result, scope)
        context_available = not context_intervals(result.events, session_time_bounds(result)[1]).empty
        show_context = st_module.sidebar.checkbox(
            "Show active context", value=context_available, disabled=not context_available,
            key=f"{scope}_context",
        )
        available = available_event_tracks(result.events)
        tracks = tuple(track for track in EVENT_TRACKS if st_module.sidebar.checkbox(
            EVENT_LABELS[track], value=track in available, disabled=track not in available,
            key=f"{scope}_event_{track}",
        ))
        # Fine-bin defaults follow the selected archive, independently of preparation settings.
        saved_ms = int(round(result.meta["bin_width_s"] * 1000))
        bounded = st_module.sidebar.checkbox(
            "Load only visible neural interval", value=saved_ms == 20,
            key=f"{scope}_bounded_{saved_ms}",
            help="Load the viewport plus neighboring context. Saved fitting and scales use all paired windows.",
        )
        loaded = None
        if bounded:
            low, high = session_time_bounds(result)
            duration = viewport[1] - viewport[0]
            loaded = (max(low, viewport[0] - duration), min(high, viewport[1] + duration))
            st_module.caption(f"Neural payload: {loaded[0]:.3f}-{loaded[1]:.3f} s. Use Window start or Jump to load elsewhere.")
        figure = plot_session_cca(
            result, pair_range=pair_range, selected_pair=pair, show_traces=traces,
            event_tracks=tracks, color_limit=limit, loaded_range_s=loaded,
            viewport_s=viewport, viewport_revision=revision, show_context=show_context,
        )
    render_timeline_chart(figure, key=f"{scope}_chart")
    st_module.caption(
        "In-sample canonical correlations; task activity retained and pooled bins temporally dependent. "
        "Scores use session sample SD. Blank intervals have no displayed neural measurements. "
        "Choices: large filled = rewarded; small open = unrewarded; gray cross = unknown. "
        "Display changes preserve time zoom; Window start/duration and Jump request a new range."
    )
    _result_details(st_module, result)


@st.cache_data(show_spinner=False, max_entries=2)
def load_cca_result_cached(path: str, size_bytes: int, mtime_ns: int) -> cca_io.SessionCCAResult:
    """Load a saved NPZ keyed by path, file size in bytes and modification nanoseconds.

    Return SessionCCAResult with unchanged (W,B,K) canonical-SD scores and
    archived UTC/elapsed behavior; no raw source files are accessed. File-stat
    arguments invalidate Streamlit's cache if the named archive changes.
    """
    return cca_io.load_cca_result(path)


# UI helpers keep preparation state separate from the selected saved analysis.
def _saved_result_controls(st_module: object, headers: tuple, scope: str, session_root: Path) -> tuple:
    """Return (Path|None, SessionCCAResult|None) from metadata-only saved headers.

    headers is the ordered CCAResultInfo tuple; scope keys widgets for one
    session; session_root locates saved analysis directories. Selectbox values
    are absolute archive strings or None. Labels
    show saved session/probes/bin/completion, with no current-source reads.
    Ordinary corrupt selections report errors and try the same saved identity.
    """
    options = [str(header.path) for header in headers] + [None]
    labels = {None: "Behavior only"}
    for header in headers:
        config = header.configuration
        roles = config["probe_roles"]
        labels[str(header.path)] = (f"{config['session_id']} | PFC {roles['PFC']} / HPC {roles['HPC']} | "
                                    f"{int(round(config['bin_width_s'] * 1000))} ms | {header.completed_utc}")
    state = st_module.session_state
    key = f"{scope}_saved"
    pending = state.pop(f"{scope}_pending_saved", None)
    if pending in options and pending is not None:
        state[key] = pending
    if key in state and state[key] not in options:
        del state[key]
    selected = st_module.sidebar.selectbox(
        "Saved CCA result", options=options, format_func=labels.__getitem__, key=key,
    )
    if selected is None:
        return None, None
    path = Path(selected)
    try:
        stat = path.stat()
        return path, load_cca_result_cached(str(path), stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, KeyError, EOFError, BadZipFile, UnpicklingError) as error:
        st_module.error(f"Could not load selected CCA: {error}")
    header = next(header for header in headers if str(header.path) == selected)
    fallback = cca_io.load_latest_cca_result(session_root, header.configuration["configuration_id"])
    if fallback is not None:
        st_module.warning(f"Showing compatible saved result: {fallback[0].path}")
        return fallback[0].path, fallback[1]
    return None, None


def _preparation_controls(st_module: object, session: ResolvedSession, scope: str, saved: bool) -> tuple[dict, float]:
    """Return explicit PFC/HPC selections and bin seconds; compute only on button clicks.

    session supplies metadata probe IDs/defaults. scope keys preparation widgets
    independently of saved selection; saved indicates a selected existing fit.
    Return (role mapping, bin width seconds). Successful actions select the new
    immutable archive on rerun; ordinary errors keep the current saved figure.
    """
    defaults = resolve_timeline_probe_roles(session, None)
    options = ["Not assigned"] + [probe.probe_id for probe in session.probes]
    roles = {}
    for region in ("PFC", "HPC"):
        selected = st_module.sidebar.selectbox(
            f"{region} probe", options=options, index=options.index(defaults.get(region, "Not assigned")),
            key=f"{scope}_{region}_probe",
        )
        if selected != "Not assigned":
            roles[region] = selected
    width_ms = st_module.sidebar.selectbox("CCA bin width (ms)", [500, 100, 50, 20], key=f"{scope}_bin_ms")
    unavailable = _preparation_unavailable_reason(session, roles)
    prepare = st_module.sidebar.button("Prepare", key=f"{scope}_prepare", disabled=unavailable is not None)
    recompute = st_module.sidebar.button("Recompute", key=f"{scope}_recompute", disabled=unavailable is not None or not saved)
    st_module.sidebar.caption("Phy good/mua units on good inside-brain channels, respecting metadata channel restrictions.")
    if unavailable is not None:
        st_module.caption(f"Preparation unavailable: {unavailable}")
    if prepare or recompute:
        try:
            with st_module.spinner("Preparing paired session-wide CCA..."):
                report = run_session_cca(session.metadata_path, roles, width_ms / 1000, rerun=recompute)
            st_module.session_state[f"{scope}_pending_saved"] = str(report.result_path)
            st_module.rerun()
        except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            st_module.error(f"Could not prepare CCA: {error}")
    return roles, width_ms / 1000


def _preparation_unavailable_reason(session: ResolvedSession, roles: dict[str, str]) -> str | None:
    """Return a short missing-prerequisite reason or None, using path checks only.

    session supplies resolved source Paths; roles explicitly assigns PFC/HPC.
    Require distinct probes, trials, aligned spikes, assignments and unit/channel
    quality tables. Optional events are not required. No file is opened and no
    scientific eligibility/configuration is calculated by this UI check.
    """
    if len(roles) != 2 or roles["PFC"] == roles["HPC"]:
        return "select two distinct PFC/HPC probes."
    required = [("trials", session.behavior.trial_table_file)]
    for region, probe_id in roles.items():
        probe = resolve_probe_sources(session, probe_id)
        required.extend([
            (f"{region} alignment", probe.alignment_file),
            (f"{region} spike assignments", probe.spike_clusters_file),
            (f"{region} cluster table", probe.cluster_info_file),
            (f"{region} channel quality", probe.channel_quality_file),
        ])
    missing = [label for label, path in required if path is None or not path.is_file()]
    return "missing " + ", ".join(missing) + "." if missing else None


def _pair_controls(st_module: object, result: cca_io.SessionCCAResult, scope: str) -> tuple:
    """Return (inclusive ranks, one-based trace pair, show flag, optional SD color limit).

    result has K saved pairs or fit=None; scope keeps display state per session.
    Scores remain in their saved canonical-SD units. No numerical work occurs.
    """
    if result.fit is None:
        return None, 1, False, None
    count = len(result.fit.correlations)
    state = st_module.session_state
    ranks_key, pair_key = f"{scope}_ranks", f"{scope}_pair"
    # Changing K changes Streamlit widget identity; restore clamped choices before creation.
    if state.get(f"{scope}_pair_count", count) != count:
        previous = state.get(ranks_key, (1, min(10, count)))
        state[ranks_key] = tuple(min(count, max(1, rank)) for rank in previous)
        state[pair_key] = min(count, max(1, state.get(pair_key, 1)))
    state[f"{scope}_pair_count"] = count
    if count == 1:
        ranks = (1, 1)
        st_module.sidebar.caption("Canonical pair ranks: CC1 only")
    else:
        ranks = st_module.sidebar.slider(
            "Canonical pair ranks", min_value=1, max_value=count,
            value=(1, min(10, count)), key=ranks_key,
        )
    pair = st_module.sidebar.selectbox("Trace pair", options=list(range(1, count + 1)), key=pair_key)
    traces = st_module.sidebar.checkbox("Show canonical traces", value=True, key=f"{scope}_traces")
    manual = st_module.sidebar.checkbox("Manual color range", value=False, key=f"{scope}_manual")
    limit = None
    if manual:
        limit = st_module.sidebar.number_input(
            "Symmetric canonical score limit (SD)", min_value=1e-12,
            value=result.fit.color_limit, key=f"{scope}_limit",
        )
        st_module.caption(f"Color range: +/-{limit:g} canonical SD; larger magnitudes saturate.")
    return ranks, pair, traces, limit


def _result_details(st_module: object, result: cca_io.SessionCCAResult) -> None:
    """Render saved unit/rank/window counts, full rho table and numerical diagnostics.

    result retains transformed-unit identities and dimensionless correlations;
    coverage counts index the original behavior table. Return None; no new
    scientific statistic or current-source eligibility is computed.
    """
    if result.fit is None:
        return
    st_module.write(f"{len(result.choice_positions)} paired windows from {len(result.choices)} choices; "
                    f"{result.fit.n_observations} pooled bins; {len(result.fit.correlations)} canonical pairs.")
    rows = []
    for region, transform in result.fit.transforms.items():
        rows.append({"Region": region, "Probe": result.probe_roles[region],
                     "Selected units": len(transform.input_unit_ids), "Nonconstant units": len(transform.retained_unit_ids),
                     "Numerical rank": transform.rank, "Eligible windows": len(result.coverage_positions[region])})
    st_module.dataframe(pd.DataFrame(rows), hide_index=True)
    with st_module.expander("All canonical correlations"):
        st_module.dataframe(pd.DataFrame({"Canonical pair": np.arange(1, len(result.fit.correlations) + 1),
                                         "In-sample rho": result.fit.correlations}), hide_index=True)
    with st_module.expander("Numerical diagnostics and provenance"):
        st_module.json(result.fit.diagnostics)
        st_module.json(result.meta)
