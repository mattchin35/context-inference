"""Saved-first Streamlit views for single-session event-sequence heatmaps."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path

import streamlit as st

from src.neural_analysis.event_sequences import io as event_io
from src.neural_analysis.event_sequences import plotting as event_plotting
from src.neural_analysis.event_sequences.run_session import (
    EventSequenceRequest,
    prepare_event_sequences,
)
from src.neural_analysis.phase_synchrony.results import parse_unit_id
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import ResolvedSession, resolve_probe_sources


CENTRAL_VIEW = "central"
TRANSITION_VIEW = "transition"
EVENT_SEQUENCE_CACHE_MAX_ENTRIES = 2
NEW_BIN_WIDTHS_S = (0.5, 0.1, 0.05, 0.02)


def render_event_sequence_view(
    st_module: object,
    session: ResolvedSession,
    *,
    kind: str = CENTRAL_VIEW,
) -> None:
    """Render one saved-first central-trial or transition heatmap view.

    Parameters
    ----------
    st_module : Streamlit-like object
        The Streamlit module or test double used for widgets and output.
    session : ResolvedSession
        One metadata-resolved session; paths are session-relative identities.
    kind : str, keyword-only
        ``central`` displays six central-trial conditions; ``transition``
        displays one direction/repeat motif. Motif rows are chronological and
        neural activity retains its saved units (Hz, PC scores, or SD units).

    Returns
    -------
    None
        Render controls, availability messages, and a Plotly figure. Source
        Units automatically prepare on selection and reuse saved results.
        PC preparation remains an explicit Prepare/Recompute action.
    """
    if kind not in (CENTRAL_VIEW, TRANSITION_VIEW):
        raise ValueError("kind must be 'central' or 'transition'.")

    title = "Event sequences: central trials" if kind == CENTRAL_VIEW else "Event sequences: ideal transitions"
    st_module.subheader(title)
    scope = hashlib.sha256(str(session.metadata_path).encode("utf-8")).hexdigest()[:12]
    products = event_io.discover_event_sequences(session.session_root)
    selected_product: dict | None = None
    request: EventSequenceRequest | None = None

    default_kind = products[0]["meta"]["source_kind"] if products else "unit"
    source_kind = st_module.selectbox(
        "Neural source", options=("unit", "pc"), index=("unit", "pc").index(default_kind),
        format_func=_source_kind_label, key=f"event_source_kind_{scope}",
    )
    if source_kind == "unit":
        unit_products = [product for product in products if product["meta"]["source_kind"] == "unit"]
        request = _render_unit_controls(st_module, session, scope, unit_products)
        if request is not None:
            selected_product = _resolve_unit_product(st_module, session, scope, request, unit_products)
    else:
        pc_products = [product for product in products if product["meta"]["source_kind"] == "pc"]
        request, selected_product = _render_pc_source(st_module, session, scope, pc_products)

    selectors = _render_display_selectors(st_module, scope, kind)
    action = None
    if request is not None:
        if selected_product is None and source_kind == "pc":
            if st_module.button("Prepare", key=f"event_prepare_{scope}"):
                action = "prepare"
        elif selected_product is not None and st_module.button("Recompute", key=f"event_recompute_{scope}"):
            action = "recompute"

    if action is not None:
        try:
            with st_module.spinner("Preparing event-sequence source..."):
                report = prepare_event_sequences(request, rerun=action == "recompute")
            result_path = report.get("result_path")
            products = event_io.discover_event_sequences(session.session_root)
            selected_product = _find_product_by_path(products, result_path)
            if selected_product is None:
                selected_product = _find_matching_product(products, session, request)
        except (OSError, ValueError) as error:
            st_module.error(f"Could not prepare event-sequence source: {error}")

    # Streamlit renders widgets before running their action. After the first
    # successful preparation, expose Recompute immediately on that same page.
    if action == "prepare" and selected_product is not None:
        if st_module.button("Recompute", key=f"event_recompute_{scope}"):
            try:
                with st_module.spinner("Preparing a fresh event-sequence run..."):
                    report = prepare_event_sequences(request, rerun=True)
                selected_product = _find_product_by_path(
                    event_io.discover_event_sequences(session.session_root),
                    report.get("result_path"),
                )
            except (OSError, ValueError) as error:
                st_module.error(f"Could not prepare event-sequence source: {error}")

    if selected_product is None:
        return
    if not selected_product["available"]:
        st_module.warning(selected_product["unavailable_reason"])
        return

    try:
        packet = _load_saved_product(selected_product)
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load saved event-sequence source: {error}")
        return

    rank = _selected_pc_rank(st_module, scope, packet)
    raw_source = event_plotting.prepare_display_source(packet, pc_rank=rank)
    zscore = _selected_zscore(st_module, scope, raw_source)
    display_source = (
        event_plotting.prepare_display_source(packet, pc_rank=rank, zscore=True)
        if zscore else raw_source
    )
    if kind == CENTRAL_VIEW:
        view = event_plotting.build_central_view(
            packet,
            display_source,
            central_outcome=selectors["central_outcome"],
            side=selectors["side"],
        )
    else:
        transition_reason = packet["meta"].get("transition_unavailable_reason", "")
        if transition_reason:
            st_module.info(transition_reason)
        view = event_plotting.build_transition_view(
            packet,
            display_source,
            direction=selectors["direction"],
            repeats=selectors["repeats"],
        )
    st_module.plotly_chart(
        event_plotting.plotly_figure(view),
        width="stretch",
        key=f"event_chart_{scope}_{kind}",
    )
    st_module.caption(view["caption"])

    if st_module.button("Export PNG", key=f"event_export_{scope}_{kind}"):
        try:
            export_path = _next_png_path(selected_product["result_path"], kind)
            event_plotting.export_png(view, export_path)
            st_module.success(f"PNG saved to {export_path}")
        except (OSError, ValueError) as error:
            st_module.error(f"Could not export event-sequence PNG: {error}")


def _render_unit_controls(
    st_module: object, session: ResolvedSession, scope: str, products: list[dict],
) -> EventSequenceRequest | None:
    """Input: UI/session/widget scope/saved unit headers. Return selected unit request or None.

    Probe/qualified unit IDs are unitless; bin width is seconds. Current units
    require explicit manual-good labels on allowed channels. Missing quality
    files permit saved-unit browsing only. A new probe has no implicit unit
    selection until the user chooses one; saved selections can load immediately.
    """
    unit_ids = _manual_good_unit_ids(st_module, session, scope)
    columns = st_module.columns(3)
    probe_ids = [probe.probe_id for probe in session.probes]
    if not probe_ids:
        st_module.warning("No metadata probes are available.")
        return None
    latest = products[0]["meta"]["parameters"] if products else {}
    probe_key = f"event_unit_probe_{scope}"
    previous_probe = st_module.session_state.get(probe_key, latest.get("probe_id"))
    probe = columns[0].selectbox(
        "Probe", options=probe_ids,
        index=probe_ids.index(previous_probe) if previous_probe in probe_ids else 0, key=probe_key,
    )
    width_key = f"event_unit_width_{scope}"
    previous_width = st_module.session_state.get(width_key, latest.get("bin_width_s", 0.5))
    width = columns[2].selectbox(
        "Bin width", options=NEW_BIN_WIDTHS_S,
        index=NEW_BIN_WIDTHS_S.index(previous_width) if previous_width in NEW_BIN_WIDTHS_S else 0,
        format_func=_format_bin_width, key=width_key,
    )
    sources = resolve_probe_sources(session, probe)
    metadata_ready = all(path is not None and path.is_file()
                         for path in (sources.channel_quality_file, sources.cluster_info_file))
    if metadata_ready:
        options = [unit for unit in unit_ids if parse_unit_id(unit)[0] == probe]
    else:
        options = sorted({product["meta"]["parameters"]["unit_id"] for product in products
                          if product["meta"]["parameters"]["probe_id"] == probe}, key=parse_unit_id)
        if options:
            st_module.info("Current quality metadata is unavailable; showing saved units.")
    if metadata_ready and options:
        st_module.caption(f"{len(options)} manual-good units on {probe}. New selections calculate automatically.")
    default_unit = next((product["meta"]["parameters"]["unit_id"] for product in products
                         if product["meta"]["parameters"]["probe_id"] == probe
                         and product["meta"]["parameters"]["bin_width_s"] == width
                         and product["meta"]["parameters"]["unit_id"] in options), None)
    unit_key = f"event_unit_id_{scope}_{probe}"
    selected = st_module.session_state.get(unit_key, default_unit)
    if unit_key in st_module.session_state and selected not in options:
        st_module.session_state[unit_key] = None
    unit = columns[1].selectbox(
        "Unit", options=options, index=options.index(selected) if selected in options else None,
        placeholder="Choose a unit", key=unit_key,
    )
    if not options:
        st_module.warning("No eligible manual-good units or saved units are available for this probe.")
    elif unit is None:
        st_module.info("Choose a unit to load its saved activity or calculate it automatically.")
    return None if unit is None else EventSequenceRequest(session.metadata_path, "unit", unit_id=unit, bin_width_s=width)


def _resolve_unit_product(
    st_module: object, session: ResolvedSession, scope: str,
    request: EventSequenceRequest, products: list[dict],
) -> dict | None:
    """Input: UI/session/scope/unit request/saved headers. Return reusable or newly prepared product.

    Compare declared file identities (paths/bytes/mtime) without reading signal
    arrays. A matching saved product works offline. Missing or changed current
    products prepare once using the existing pipeline/cache. Failed attempts
    wait for explicit Retry or changed source/selection, never motif reruns.
    """
    product = _find_matching_product(products, session, request)
    probe = resolve_probe_sources(session, parse_unit_id(request.unit_id)[0])
    paths = (session.behavior.trial_table_file, probe.alignment_file, probe.spike_clusters_file,
             probe.cluster_info_file, probe.channel_quality_file)
    identities = [event_io.source_file_identity(path) if path is not None else {"missing": True}
                  for path in paths]
    inputs_ready = all(not identity.get("missing", False) for identity in identities)
    if product is not None and (not inputs_ready or product["meta"]["configuration"]["sources"] == identities):
        return product
    if not inputs_ready:
        st_module.warning("This unit has no saved result for these bins, and its current preparation inputs are unavailable.")
        return None
    failure_key = f"event_unit_failure_{scope}"
    failure = st_module.session_state.get(failure_key)
    token = (request, identities)
    retry_shown = failure is not None and failure[0] == token
    if retry_shown:
        if not st_module.button("Retry unit", key=f"event_unit_retry_{scope}"):
            st_module.error(failure[1])
            return None
    try:
        with st_module.spinner(f"Preparing {request.unit_id}..."):
            report = prepare_event_sequences(request)
        product = _find_product_by_path(event_io.discover_event_sequences(session.session_root), report["result_path"])
        if product is None:
            raise ValueError("Unit preparation did not publish a readable saved result.")
        st_module.session_state.pop(failure_key, None)
        return product
    except (OSError, ValueError) as error:
        message = f"Could not prepare unit {request.unit_id}: {error}"
        st_module.session_state[failure_key] = (token, message)
        st_module.error(message)
        if not retry_shown:
            st_module.button("Retry unit", key=f"event_unit_retry_{scope}")
        return None


def _render_pc_source(
    st_module: object, session: ResolvedSession, scope: str, products: list[dict],
) -> tuple[EventSequenceRequest | None, dict | None]:
    """Input: UI/session/scope/saved PC headers. Return PC request/product, with new preparation always explicit."""
    configure = not products or st_module.checkbox("Configure a new source", key=f"event_configure_{scope}")
    if configure:
        request = _render_pc_configuration(st_module, session, scope)
        product = None if request is None else _find_matching_product(products, session, request)
        if product is None and request is not None:
            st_module.info("This PC source is not prepared yet. Click Prepare to create it.")
        elif product is not None:
            st_module.caption("Saved run with these source settings: " + _product_label(product))
        return request, product
    labels = [_product_label(product) for product in products]
    label = st_module.selectbox("Saved source configuration", options=labels, key=f"event_saved_source_{scope}")
    product = products[labels.index(label)]
    return _request_from_product(session, product), product


def _render_pc_configuration(st_module: object, session: ResolvedSession, scope: str) -> EventSequenceRequest | None:
    """Input: Streamlit/session/scope. Return selected regional PC request; bin width is seconds."""
    bin_width_s = st_module.selectbox(
        "Preparation bin width",
        options=NEW_BIN_WIDTHS_S,
        format_func=_format_bin_width,
        key=f"event_bin_width_{scope}",
    )
    population_columns = st_module.columns(3)
    region = population_columns[0].selectbox(
        "PC region", options=("PFC", "HPC"), key=f"event_region_{scope}"
    )
    quality = population_columns[1].selectbox(
        "PC population",
        options=("good_mua", "good"),
        key=f"event_population_{scope}",
    )
    default_probe = _metadata_probe_for_region(session, region)
    override_options = ("Use metadata role", *[probe.probe_id for probe in session.probes])
    override = population_columns[2].selectbox(
        "PC probe override",
        options=override_options,
        index=0,
        key=f"event_probe_override_{scope}",
    )
    probe_id = None if override == "Use metadata role" else override
    if default_probe is None and probe_id is None:
        st_module.info("No metadata probe role is configured; choose a probe override to prepare this region.")
        return None
    return EventSequenceRequest(
        session.metadata_path,
        "pc",
        region=region,
        probe_id=probe_id,
        bin_width_s=bin_width_s,
        population_quality=quality,
    )


def _manual_good_unit_ids(
    st_module: object, session: ResolvedSession, scope: str
) -> tuple[str, ...]:
    """Input: Streamlit state, metadata session and widget scope; return cached manual-good PROBE:CLUSTER IDs.

    The two-entry per-app snapshot cache keys the small quality inputs and
    metadata by their current size/mtime. It stores identities only, never
    spike arrays, and refreshes after any tracked source change.
    """
    cache_key = f"event_good_unit_ids_{scope}"
    signature = _quality_metadata_signature(session)
    snapshots = st_module.session_state.get(cache_key, {})
    if signature in snapshots:
        return snapshots[signature]
    unit_ids = []
    for probe in session.probes:
        if any(path is None or not path.is_file()
               for path in (probe.channel_quality_file, probe.cluster_info_file)):
            continue
        try:
            _, _, rows = timeline_io.select_timeline_population(
                session,
                probe.probe_id,
                population_quality="good",
                missing_label_policy="exclude",
            )
        except (OSError, ValueError):
            continue
        unit_ids.extend(f"{probe.probe_id}:{cluster_id}" for cluster_id in rows.cluster_id)
    result = tuple(unit_ids)
    snapshots[signature] = result
    while len(snapshots) > 2:
        snapshots.pop(next(iter(snapshots)))
    st_module.session_state[cache_key] = snapshots
    return result


def _quality_metadata_signature(session: ResolvedSession) -> tuple:
    """Input: resolved metadata session; return path/size/mtime identity for small unit-quality inputs."""
    paths = [session.metadata_path]
    for probe in session.probes:
        paths.extend((probe.channel_quality_file, probe.cluster_info_file))
    identity = []
    for path in paths:
        if path is None:
            identity.append(("", -1, -1))
            continue
        try:
            stat = Path(path).stat()
        except OSError:
            identity.append((str(path), -1, -1))
        else:
            identity.append((str(path), stat.st_size, stat.st_mtime_ns))
    return tuple(identity)


def _metadata_probe_for_region(session: ResolvedSession, region: str) -> str | None:
    """Input: resolved session and PFC/HPC role; return its metadata probe ID or None."""
    if session.analysis is None:
        return None
    return getattr(session.analysis.probe_roles, region.lower())


def _source_kind_label(source_kind: str) -> str:
    """Input: unit/pc source-kind string; return readable configuration label."""
    return "Single unit" if source_kind == "unit" else "Regional PCs"


def _format_bin_width(bin_width_s: float) -> str:
    """Input: bin width in seconds; return integer-millisecond display label."""
    return f"{int(round(bin_width_s * 1000))} ms"


def _find_matching_product(products: list[dict], session: ResolvedSession, request: EventSequenceRequest) -> dict | None:
    """Input: discovered saved runs/session/request; return newest exact control match without reading current quality files."""
    requested_probe = _request_probe_id(session, request)
    for product in products:
        parameters = product["meta"]["parameters"]
        if (
            parameters.get("source_kind") == request.source_kind
            and parameters.get("unit_id") == request.unit_id
            and parameters.get("region") == request.region
            and parameters.get("probe_id") == requested_probe
            and parameters.get("bin_width_s") == request.bin_width_s
            and parameters.get("population_quality") == request.population_quality
        ):
            return product
    return None


def _request_probe_id(session: ResolvedSession, request: EventSequenceRequest) -> str:
    """Input: session and validated request; return the resolved probe identity string."""
    if request.source_kind == "unit":
        return parse_unit_id(request.unit_id)[0]
    return request.probe_id or _metadata_probe_for_region(session, request.region) or ""


def _request_from_product(session: ResolvedSession, product: dict) -> EventSequenceRequest:
    """Input: resolved session and saved product; return an equivalent immutable preparation request."""
    parameters = product["meta"]["parameters"]
    if parameters["source_kind"] == "unit":
        return EventSequenceRequest(
            session.metadata_path,
            "unit",
            unit_id=parameters["unit_id"],
            bin_width_s=parameters["bin_width_s"],
        )
    return EventSequenceRequest(
        session.metadata_path,
        "pc",
        region=parameters["region"],
        probe_id=parameters["probe_id"],
        bin_width_s=parameters["bin_width_s"],
        population_quality=parameters["population_quality"],
    )


def _find_product_by_path(products: list[dict], result_path: Path | None) -> dict | None:
    """Input: discovered products and optional saved Path; return its matching run or None."""
    if result_path is None:
        return None
    expected = Path(result_path).resolve()
    return next(
        (product for product in products if Path(product["result_path"]).resolve() == expected),
        None,
    )


def _product_label(product: dict) -> str:
    """Input: discovered saved product; return readable source/probe/quality/bin/run label."""
    meta = product["meta"]
    parameters = meta["parameters"]
    width_ms = int(round(parameters["bin_width_s"] * 1000))
    run_name = Path(product["result_path"]).parent.name.removeprefix(
        "event_sequence_heatmaps_"
    )
    run_tag = "_".join(run_name.split("_")[:3])
    return (
        f"{meta['source_label']} | probe {parameters['probe_id']} | "
        f"{parameters['population_quality']} | {width_ms} ms | {run_tag}"
    )


def _render_display_selectors(st_module: object, scope: str, kind: str) -> dict:
    """Input: Streamlit/session scope/view kind; return live selectors with outcome/side or transition direction/repeats."""
    if kind == CENTRAL_VIEW:
        display_columns = st_module.columns(2)
        return {
            "central_outcome": display_columns[0].selectbox(
                "Central outcome",
                options=("correct_rewarded", "omission"),
                key=f"event_outcome_{scope}",
            ),
            "side": display_columns[1].selectbox(
                "Central side", options=("left", "right"), key=f"event_side_{scope}"
            ),
        }
    display_columns = st_module.columns(2)
    return {
        "direction": display_columns[0].selectbox(
            "Transition direction",
            options=("left_to_right", "right_to_left"),
            key=f"event_direction_{scope}",
        ),
        "repeats": display_columns[1].slider(
            "Incorrect repeats", min_value=1, max_value=4, value=1, step=1,
            key=f"event_repeats_{scope}",
        ),
    }


def _selected_pc_rank(st_module: object, scope: str, packet: dict) -> int:
    """Input: Streamlit/session scope/loaded packet; return valid one-based PC rank or fixed unit rank 1."""
    if packet["meta"]["source_kind"] != "pc":
        return 1
    region = packet["meta"]["pca_reference"]["region"]
    component_count = packet["timeline_result"].regions[region].scores.shape[2]
    key = f"event_pc_rank_{scope}"
    previous = int(st_module.session_state.get(key, 1))
    rank = min(max(previous, 1), component_count)
    if rank != previous:
        st_module.session_state[key] = rank
    if component_count == 1:
        st_module.caption("PC rank: 1 (only one component is available).")
        return 1
    return st_module.slider(
        "PC rank", min_value=1, max_value=component_count, value=rank, step=1, key=key
    )


def _selected_zscore(st_module: object, scope: str, source: dict) -> bool:
    """Input: Streamlit/session scope/raw prepared source; return z-score flag, disabled for zero-SD data."""
    disabled = source["constant_activity"] or bool(source["unavailable_reason"])
    key = f"event_zscore_{scope}"
    if disabled and st_module.session_state.get(key, False):
        st_module.session_state[key] = False
    checked = st_module.checkbox("Z-score", value=False, key=key, disabled=disabled)
    if disabled:
        reason = source["unavailable_reason"] or "Session activity has zero SD."
        st_module.caption(f"Z-scoring unavailable: {reason}")
    return bool(checked)


def _load_saved_product(product: dict) -> dict:
    """Input: discovered NPZ product; return its validated packet through a two-entry source-stat cache."""
    path = Path(product["result_path"]).resolve(strict=True)
    event_stat = path.stat()
    meta = product["meta"]
    reference = meta.get("pca_reference")
    if reference is None:
        pca_path, pca_size, pca_mtime_ns = "", -1, -1
    else:
        session_root = Path(product["result_path"]).resolve().parents[2]
        referenced = (session_root / reference["path"]).resolve()
        if referenced.is_file():
            pca_stat = referenced.stat()
            pca_path, pca_size, pca_mtime_ns = str(referenced), pca_stat.st_size, pca_stat.st_mtime_ns
        else:
            pca_path, pca_size, pca_mtime_ns = str(referenced), -1, -1
    return _load_saved_product_cached(
        str(path), event_stat.st_size, event_stat.st_mtime_ns,
        pca_path, pca_size, pca_mtime_ns,
    )


@st.cache_resource(show_spinner=False, max_entries=EVENT_SEQUENCE_CACHE_MAX_ENTRIES)
def _load_saved_product_cached(
    result_path: str,
    result_size: int,
    result_mtime_ns: int,
    pca_path: str,
    pca_size: int,
    pca_mtime_ns: int,
) -> dict:
    """Input: NPZ/PCA paths and current byte-size/mtime-ns stats; return packet arrays/tables with saved axes and units."""
    # All scalar identity arguments participate in Streamlit's cache key; io validates the exact PCA content hash.
    _ = (result_size, result_mtime_ns, pca_path, pca_size, pca_mtime_ns)
    return event_io.load_event_sequences(Path(result_path))


def _next_png_path(result_path: Path, kind: str) -> Path:
    """Input: selected results.npz Path and view kind; return a new timestamped PNG path under its figures directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    return Path(result_path).parent / "figures" / f"event_sequences_{kind}_{timestamp}.png"
