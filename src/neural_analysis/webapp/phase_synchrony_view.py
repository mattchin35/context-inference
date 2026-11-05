"""Manual-refresh timeline over independent, completed choice-window products."""

from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zipfile import BadZipFile

import streamlit as st

from src.neural_analysis.lfp.config import FrequencyBandConfig
from src.neural_analysis.phase_synchrony import preparation, results, plotting
from src.neural_analysis.population.session_timeline import SessionTimelineResult
from src.neural_analysis.population.session_timeline_io import load_timeline_behavior
from src.neural_analysis.population.timeline_plotting import available_event_tracks
from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.webapp.timeline_chart import render_timeline_chart
from src.neural_analysis.webapp.timeline_view import render_timeline_navigation


def discover_view_products(
    session: ResolvedSession, config: results.PhaseSynchronyConfig,
    *, headers: tuple[results.ProductInfo, ...] | None = None,
) -> tuple[results.SavedProduct, ...]:
    """Load newest compatible independent saved metrics without opening raw arrays.

    session supplies resolved metadata; config selects categorical sites/pairs/
    units/intervals and Hz/second settings. Optional headers are metadata-only
    discovery results in completion/run/path order. Return loaded SavedProduct
    tuples with original trial/frequency axes and units. Match actual source
    stats and family-relevant settings; unreadable newest arrays fall back to
    older compatible products. No prerequisites are prepared and nothing writes.
    """
    headers = results.discover_product_metadata(session.session_root) if headers is None else headers
    sites, pairs = preparation.select_metadata_inputs(session, site_ids=config.sites, pairs=config.pairs)
    requested = _component_keys(config, [site.site_id for site in sites], pairs)
    return _load_matching_products(session, config, headers, requested)


def load_inspection_products(
    session: ResolvedSession, config: results.PhaseSynchronyConfig,
    products: tuple[results.SavedProduct, ...], *, site_id: str, unit_id: str | None = None,
    headers: tuple[results.ProductInfo, ...] | None = None,
) -> tuple[results.SavedProduct, ...]:
    """Load existing selected-site waveform/phase prerequisites for explicit inspection.

    Inputs retain original UTC trial axes and calibrated uV/unit phase/counts.
    Return matching saved metrics plus available waveform/phase products for
    the requested intervals; never load native continuous recordings or spikes.
    When inspecting a unit PPC, phase must identify that PPC's actual prepared
    phase, rather than an arbitrary newer preparation with matching settings.
    Missing inspection inputs remain absent for the plotting unavailable labels.
    """
    headers = results.discover_product_metadata(session.session_root) if headers is None else headers
    selected = tuple(product for product in products
                     if product.meta["identity"]["family"] != "spike-ppc"
                     or (product.record.site_id == site_id and unit_id is not None and product.record.unit_id == unit_id))
    spike = [product for product in selected if product.meta["identity"]["family"] == "spike-ppc"]
    families = ("waveforms",) if spike else ("waveforms", "phase")
    requested = {(family, (site_id,), interval) for family in families for interval in config.intervals}
    prepared = _load_matching_products(session, config, headers, requested)
    if spike:
        required = {reference for product in spike for reference in product.meta["identity"]["preparation_ids"]}
        # The phase identity contains its actual self-reference; match before
        # loading large phase arrays, then validate the typed product on load.
        phase_headers = tuple(header for header in headers if header.meta["identity"]["family"] == "phase"
                              and required.intersection(header.meta["identity"]["preparation_ids"]))
        phase_request = {("phase", (site_id,), interval) for interval in config.intervals}
        prepared += _load_matching_products(session, config, phase_headers, phase_request)
    return selected + prepared


def newer_attempt_statuses(
    session_root: Path, config: results.PhaseSynchronyConfig, products: tuple[results.SavedProduct, ...],
) -> list[dict]:
    """Return newer failed/skipped/unfinished attempts separately from displayed completed sources.

    Root is a session directory; only each component's relevant scientific
    settings are compared. Ordering uses the recorded UTC run ID and completion
    instant, never filesystem mtime. Returned dictionaries identify categorical
    stage/selection/interval/status/error, attempted run and displayed source run.
    Only JSON logs/configurations are read; no arrays or analysis are executed.
    """
    displayed = {(product.meta["identity"]["family"], tuple(product.meta["identity"]["selection"]),
                  product.meta["identity"]["interval"]): product for product in products}
    requested = _component_keys(config, config.sites, config.pairs) | set(displayed)
    attempts = []
    for directory in (Path(session_root) / "analysis_runs").glob("session_timeline_phase_synchrony_*"):
        try:
            run_config = _read_config(directory / "config.json")
            path = directory / "measurements.json"
            measurements = json.loads(path.read_text()) if path.is_file() else {}
            run_id = measurements.get("run_id", directory.name.removeprefix("session_timeline_phase_synchrony_"))
            attempted = datetime.strptime(run_id[:22], "%Y%m%d_%H%M%S_%f").replace(tzinfo=timezone.utc)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        attempts.append((attempted, run_id, str(directory), run_config, measurements))
    statuses = []
    for attempted, run_id, _, run_config, measurements in sorted(attempts, key=lambda item: item[:3], reverse=True):
        for family, selection, interval in sorted(requested):
            product = displayed.get((family, selection, interval))
            completed = None if product is None else datetime.fromisoformat(product.meta["completed_utc"].replace("Z", "+00:00"))
            if (completed is not None and attempted <= completed) or not _attempt_requested(run_config, family, selection, interval):
                continue
            relevant = results.product_identity(family, selection, interval, config, source_ids={})["parameters"]
            attempted_parameters = results.product_identity(family, selection, interval, run_config, source_ids={})["parameters"]
            if relevant != attempted_parameters:
                continue
            matching = [stage for stage in measurements.get("stages", []) if stage.get("stage") == family
                        and tuple(stage.get("selection", ())) == selection and stage.get("interval") == interval]
            stage = matching[-1] if matching else {"status": "unfinished"}
            if stage["status"] in {"failed", "skipped", "unfinished"}:
                statuses.append({"stage": family, "selection": selection, "interval": interval,
                                 "status": stage["status"], "error": stage.get("error", ""),
                                 "run_id": run_id, "displayed_run_id": None if product is None else product.meta["run_id"]})
    return statuses


def render_phase_synchrony_view(st_module: object, session: ResolvedSession) -> None:
    """Render saved metrics, manual discovery Refresh and original-time navigation.

    st_module is Streamlit's widget interface; session is resolved metadata.
    Behavior retains original row/ID and UTC seconds. Numerical products load
    only from completed NPZs, with explicit inspection loading saved prerequisites.
    No button launches computation, workers, background jobs or source writes.
    A stable chart key and time-navigation revision preserve browser panning.
    """
    st_module.subheader("Session timeline: LFP power and synchrony")
    scope = "phase_timeline_" + hashlib.sha256(str(session.metadata_path).encode()).hexdigest()[:10]
    with st_module.container(key=f"{scope}_controls"):
        refresh = st_module.sidebar.button("Refresh results", key=f"{scope}_refresh")
        state = st_module.session_state
        header_key = f"{scope}_headers"
        if refresh or header_key not in state:
            state[header_key] = results.discover_product_metadata(session.session_root)
        headers = state[header_key]
        configs = _completed_configs(headers)
        settings = st_module.sidebar.selectbox("Saved analysis settings", options=list(range(len(configs))),
                                               format_func=lambda index: _settings_label(configs[index]), key=f"{scope}_settings")
        base = configs[settings]
        interval = st_module.sidebar.selectbox("Centered interval", ("primary", "secondary"), key=f"{scope}_interval")
        metric = st_module.sidebar.radio("LFP phase metric", ("plv", "ppc"), format_func=str.upper, key=f"{scope}_metric")
        site_ids = [site.site_id for site in session.sites]
        sites = st_module.sidebar.multiselect("LFP sites", site_ids, default=site_ids, key=f"{scope}_sites")
        _, available_pairs = preparation.select_metadata_inputs(session, site_ids=tuple(sites))
        pairs = st_module.sidebar.multiselect("LFP pairs", available_pairs, default=available_pairs,
                                              format_func=lambda pair: " / ".join(pair), key=f"{scope}_pairs")
        unit_options = list(dict.fromkeys(header.meta["identity"]["selection"][0] for header in headers
                                         if header.meta["identity"]["family"] == "spike-ppc"))
        units = st_module.sidebar.multiselect("Saved units", unit_options, default=unit_options[:1], key=f"{scope}_units")
        preferred_site = next((header.meta["identity"]["selection"][1] for header in headers
                               if header.meta["identity"]["family"] == "spike-ppc"
                               and units and header.meta["identity"]["selection"][0] == units[0]), None)
        lfp_site = st_module.sidebar.selectbox("Spike-PPC LFP site", site_ids,
                                               index=site_ids.index(preferred_site) if preferred_site in site_ids else 0,
                                               key=f"{scope}_lfp_site") if site_ids else None
        families = ("power", "coherence", "lfp-phase") + (("spike-ppc",) if units and lfp_site else ())
        config = replace(base, components=families, sites=tuple(sites), pairs=tuple(pairs), unit_ids=tuple(units),
                         lfp_site=lfp_site, intervals=(interval,))
        started = perf_counter()
        try:
            origin, choices, events = load_timeline_behavior(session)
            behavior = SessionTimelineResult(session.session_id, origin, choices, events, {}, {}, {})
            products = discover_view_products(session, config, headers=headers)
            # Offline empty selections mean metadata defaults; empty display
            # controls mean hide, so apply the explicit widget selections here.
            products = tuple(product for product in products if _display_selected(product, sites, pairs))
        except (OSError, ValueError, KeyError) as error:
            st_module.error(f"Could not load saved timeline: {error}")
            return
        load_s = perf_counter() - started
        event_options = available_event_tracks(events)
        tracks = st_module.sidebar.multiselect("Behavior events", event_options, default=[], key=f"{scope}_events")
        context = st_module.sidebar.checkbox("Show active context", value=True, key=f"{scope}_context")
        visible, revision = render_timeline_navigation(st_module, behavior, f"{scope}_navigation")
        heatmaps = [(product.meta["identity"]["family"], tuple(product.meta["identity"]["selection"]))
                    for product in products if product.meta["identity"]["family"] != "power"]
        heatmap = st_module.sidebar.selectbox("Frequency heatmap", [None] + heatmaps,
                                              format_func=lambda value: "Hidden" if value is None else f"{value[0]}: {' / '.join(value[1])}",
                                              key=f"{scope}_heatmap")
        completed_keys = {(product.meta["identity"]["family"], tuple(product.meta["identity"]["selection"]), interval)
                          for product in products}
        missing = sorted(_component_keys(config, sites, pairs) - completed_keys)
        if missing:
            st_module.info("Not prepared / unavailable: " + "; ".join(f"{family} {' / '.join(selection)} ({window})"
                                                                      for family, selection, window in missing)
                           + ". Prepare in a separate terminal, then Refresh results.")
        st_module.caption(_settings_label(config) + "; gamma excludes 58-62 Hz. No significance tests; no causal interpretation.")
        started = perf_counter()
        figure = plotting.plot_synchrony_timeline(behavior, products, interval=interval, phase_metric=metric,
                                                  time_range_s=visible, viewport_revision=revision,
                                                  event_tracks=tuple(tracks), show_context=context, heatmap_selection=heatmap)
        construction_s = perf_counter() - started
        started = perf_counter()
        payload_bytes = len(figure.to_json().encode("utf-8"))
        serialization_s = perf_counter() - started
    render_timeline_chart(figure, key=f"{scope}_chart")
    st_module.caption(figure.layout.meta["caption"] + " Drag horizontally to pan; use the time controls or Jump to navigate explicitly.")
    _render_saved_details(st_module, session, config, products, load_s, construction_s, serialization_s, payload_bytes)
    _render_inspection(st_module, session, behavior, config, products, headers, scope)


# Metadata/settings and saved-array helpers. These never prepare scientific data.
def _component_keys(config: results.PhaseSynchronyConfig, sites: tuple | list, pairs: tuple | list) -> set:
    """Return explicit categorical (family,selection,interval) keys; empty display lists remain empty."""
    requested = set()
    for interval in config.intervals:
        for family in config.components:
            selections = [(site,) for site in sites] if family == "power" else list(pairs)
            if family == "spike-ppc":
                selections = [(unit, config.lfp_site) for unit in config.unit_ids]
            requested.update((family, tuple(selection), interval) for selection in selections)
    return requested


def _load_matching_products(session: ResolvedSession, config: results.PhaseSynchronyConfig,
                            headers: tuple, requested: set) -> tuple[results.SavedProduct, ...]:
    """Load one completed record per requested categorical key, matching relevant settings/source stats."""
    loaded = {}
    for header in headers:
        identity = header.meta["identity"]
        key = identity["family"], tuple(identity["selection"]), identity["interval"]
        if key not in requested or key in loaded:
            continue
        try:
            sources = preparation.product_source_ids(session, key[0], key[1])
            expected = results.product_identity(key[0], key[1], key[2], config, source_ids=sources,
                                                 preparation_ids=tuple(identity["preparation_ids"]))
            if expected != identity:
                continue
            info = header.path.stat()
            loaded[key] = _load_product_cached(str(header.path), info.st_size, info.st_mtime_ns)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            continue
    return tuple(loaded.values())


def _display_selected(product: results.SavedProduct, sites: list[str], pairs: list[tuple]) -> bool:
    """Return whether one saved metric belongs to explicit display site/pair selections; no array changes."""
    identity = product.meta["identity"]
    family, selection = identity["family"], tuple(identity["selection"])
    if family == "power":
        return selection[0] in sites
    if family in {"coherence", "lfp-phase"}:
        return selection in pairs
    return True


@st.cache_data(show_spinner=False, max_entries=16)
def _load_product_cached(path: str, size_bytes: int, mtime_ns: int) -> results.SavedProduct:
    """Load immutable saved arrays keyed by path/bytes/mtime; preserve their documented units and axes."""
    return results.load_product(path)


def _read_config(path: Path) -> results.PhaseSynchronyConfig:
    """Read one run's JSON settings into the existing typed config (seconds/Hz); ignore run-only keys."""
    payload = json.loads(path.read_text())
    values = {field.name: payload[field.name] for field in fields(results.PhaseSynchronyConfig) if field.name in payload}
    for name in ("components", "intervals", "sites", "unit_ids", "phase_frequencies_hz"):
        if name in values:
            values[name] = tuple(values[name])
    if "pairs" in values:
        values["pairs"] = tuple(tuple(pair) for pair in values["pairs"])
    if "bands" in values:
        values["bands"] = tuple(FrequencyBandConfig(band["name"], band["lower_hz"], band["upper_hz"],
                                                   tuple(tuple(bounds) for bounds in band["excluded_intervals_hz"]))
                                for band in values["bands"])
    return results.PhaseSynchronyConfig(**values)


def _settings_key(config: results.PhaseSynchronyConfig) -> str:
    """Return a deterministic JSON settings key excluding requested/display categorical selections."""
    values = asdict(config)
    for name in ("components", "sites", "pairs", "unit_ids", "lfp_site", "intervals"):
        values.pop(name)
    return json.dumps(values, sort_keys=True)


def _settings_label(config: results.PhaseSynchronyConfig) -> str:
    """Describe actual saved notch, Welch and phase settings in readable physical units."""
    return (f"Notch {'60 Hz Q=' + str(config.notch_quality_factor) if config.notch_enabled else 'off'}; "
            f"Welch {config.welch_segment_s:g} s / {100 * config.welch_overlap_fraction:g}% overlap; "
            f"phase {config.phase_rate_hz:g} Hz, {len(config.phase_frequencies_hz)} frequencies; "
            f"phase thresholds {config.numerical_amplitude_threshold:g}/{config.absolute_amplitude_threshold:g}")


def _completed_configs(headers: tuple) -> tuple[results.PhaseSynchronyConfig, ...]:
    """Return unique scientific settings from completed metric runs, newest completion first; defaults if none."""
    configs = {}
    for header in headers:
        if header.meta["identity"]["family"] not in plotting.METRIC_FAMILIES:
            continue
        try:
            config = _read_config(header.path.parents[2] / "config.json")
            configs.setdefault(_settings_key(config), config)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return tuple(configs.values()) or (results.PhaseSynchronyConfig(),)


def _attempt_requested(config: results.PhaseSynchronyConfig, family: str, selection: tuple, interval: str) -> bool:
    """Check whether a recorded run requested an existing component key; no source/data reads."""
    if family not in config.components or interval not in config.intervals:
        return False
    if family == "spike-ppc":
        return selection[0] in config.unit_ids and selection[1] == config.lfp_site
    if config.sites and not set(selection).issubset(config.sites):
        return False
    return family == "power" or not config.pairs or selection in config.pairs


def _render_saved_details(st_module: object, session: ResolvedSession, config: results.PhaseSynchronyConfig,
                          products: tuple, load_s: float, construction_s: float,
                          serialization_s: float, payload_bytes: int) -> None:
    """Render completed source run/time separately from later attempt statuses and exclusive second timings."""
    with st_module.expander("Saved sources and timing"):
        st_module.json(asdict(config))
        rows = [{"family": product.meta["identity"]["family"], "selection": " / ".join(product.meta["identity"]["selection"]),
                 "interval": product.meta["identity"]["interval"], "source_run": product.meta["run_id"],
                 "completed_utc": product.meta["completed_utc"], "path": str(product.path)} for product in products]
        st_module.dataframe(rows)
        attempts = newer_attempt_statuses(session.session_root, config, products)
        if attempts:
            st_module.write("Newer attempts (displayed completed sources remain selected):")
            st_module.dataframe(attempts)
        st_module.write(f"Saved/behavior load: {load_s:.4f} s; figure construction: {construction_s:.4f} s.")
        st_module.caption(f"JSON payload: {payload_bytes} bytes; measured serialization: {serialization_s:.4f} s. "
                          "The shared chart renderer serializes once more for submission.")
        st_module.caption("These are Python timings, excluding browser rendering. Manual Refresh rediscovers completed products.")
        for directory in dict.fromkeys(product.path.parents[2] for product in products):
            try:
                measurements = json.loads((directory / "measurements.json").read_text())
                st_module.write({"source_run": measurements["run_id"], **measurements["timings_s"]})
                st_module.dataframe([{key: stage.get(key) for key in ("stage", "selection", "interval", "status", "elapsed_s", "cache_state")}
                                     for stage in measurements.get("stages", [])])
            except (OSError, ValueError, KeyError, TypeError):
                continue


def _render_inspection(st_module: object, session: ResolvedSession, behavior: SessionTimelineResult,
                       config: results.PhaseSynchronyConfig, products: tuple, headers: tuple, scope: str) -> None:
    """Offer explicit saved-only original-row/site/unit/frequency inspection; all coordinates keep seconds/Hz."""
    if behavior.choices.empty or not st_module.checkbox("Inspect a choice", key=f"{scope}_inspect"):
        return
    rows = behavior.choices.set_index("row_position")
    row = st_module.selectbox("Inspect original row", rows.index.tolist(),
                              format_func=lambda position: f"Trial {rows.loc[position, 'trial_id']} (row {position})", key=f"{scope}_inspect_row")
    site_ids = [site.site_id for site in session.sites]
    if not site_ids:
        st_module.info("Inspection unavailable: no metadata LFP site.")
        return
    site = st_module.selectbox("Inspection LFP site", site_ids, key=f"{scope}_inspect_site")
    units = list(config.unit_ids)
    unit = st_module.selectbox("Inspection unit", [None] + units, key=f"{scope}_inspect_unit")
    frequency = st_module.selectbox("Phase frequency (Hz)", config.phase_frequencies_hz,
                                    index=config.phase_frequencies_hz.index(8.0) if 8.0 in config.phase_frequencies_hz else 0,
                                    key=f"{scope}_inspect_frequency")
    selected = load_inspection_products(session, config, products, site_id=site, unit_id=unit, headers=headers)
    figure = plotting.plot_choice_inspection(behavior, selected, row_position=row, frequency_hz=frequency,
                                             interval=config.intervals[0], site_id=site, unit_id=unit)
    st_module.plotly_chart(figure, key=f"{scope}_inspection_chart", width="stretch")
    st_module.caption(figure.layout.meta["caption"])
