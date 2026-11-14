"""Inspect immutable context-encoding products without live sources or fitting."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis.encoding import results
from src.neural_analysis.encoding.configuration import (
    ENCODINGS,
    ESTIMATORS,
    PENALTY_MODES,
)
from src.neural_analysis.population.timeline_plotting import (
    available_event_tracks,
    session_time_bounds,
)
from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.webapp.timeline_chart import render_timeline_chart
from src.neural_analysis.webapp.timeline_view import render_timeline_navigation


def render_encoding_view(st_module: object, session: ResolvedSession) -> None:
    """Render controls and a coupled figure from one session's saved encoding runs.

    st_module is Streamlit's widget interface; session supplies only the metadata
    identity and session-root Path. Scores remain (window,bin,original PC) in
    unwhitened PC units. All time controls use elapsed seconds. This function
    reads saved preparation/one selected cell, never live CSVs, spikes, or fits.
    Returns None; optional PNG export also consumes only the saved products.
    """
    from src.neural_analysis.encoding.plotting import (
        contrast_values,
        order_responses,
        plot_encoding_timeline,
    )

    st_module.subheader("Single unit selectivity")
    scope = (
        "encoding_"
        + hashlib.sha256(str(session.metadata_path).encode()).hexdigest()[:10]
    )
    runs = results.discover_encoding_runs(session.session_root / "analysis_runs")
    options = [
        str(path)
        for path in runs
        if (path / "config.json").is_file() and (path / "progress.json").is_file()
    ]
    if not options:
        st_module.info(
            "No saved single unit selectivity runs are available for this session."
        )
        return
    directory = Path(
        st_module.sidebar.selectbox(
            "Saved encoding run",
            options=options,
            format_func=_run_label,
            key=f"{scope}_run",
        )
    )
    key = scope + "_" + hashlib.sha256(str(directory).encode()).hexdigest()[:10]
    try:
        manifest, progress, entries = _read_completed_metadata(directory)
        if progress["status"] != "complete":
            st_module.warning(
                "Partial run: completed model cells remain available for inspection."
            )
        if not entries:
            st_module.info("This run has no completed model cells to display yet.")
            return
        entry, penalty_mode = _configuration_controls(st_module, entries, key)
        path = Path(manifest["session_root"]) / manifest["preparation_path"]
        info = path.stat()
        prepared = load_preparation_cached(str(path), info.st_size, info.st_mtime_ns)
        if (
            prepared["meta"]["preparation_id"]
            != manifest["identities"]["preparation_id"]
        ):
            raise ValueError("Saved preparation identity differs from this run")
        cell_path = directory / entry["path"]
        info = cell_path.stat()
        cell = load_cell_cached(str(cell_path), info.st_size, info.st_mtime_ns)
        _check_cell_metadata(cell["meta"], entry["id"], manifest)
    except (OSError, ValueError, KeyError) as error:
        st_module.error(
            f"Could not read saved single unit selectivity results: {error}"
        )
        return

    region = cell["meta"]["settings"]["region"]
    meta = cell["meta"]
    if not meta["available"]:
        st_module.warning(meta["unavailable_reason"])
        return
    ids = cell["response_ids"].tolist()
    representation = meta["settings"]["representation"]
    row_key = f"{key}_{region}_{representation}"
    heatmap_scale = "native"
    zscore_limit = 3.0
    if representation == "units":
        heatmap_scale = st_module.sidebar.selectbox(
            "Heatmap scale",
            ["native", "within_unit_zscore"],
            format_func=_control_label,
            key=f"{row_key}_heatmap_scale",
        )
        if heatmap_scale == "within_unit_zscore":
            zscore_limit = float(
                st_module.sidebar.number_input(
                    "Heatmap limit (SD)",
                    min_value=0.1,
                    value=3.0,
                    step=0.5,
                    key=f"{row_key}_zscore_limit",
                )
            )
    coefficient_scale = st_module.sidebar.selectbox(
        "Coefficient scale",
        ["raw", "standardized"],
        format_func=_control_label,
        key=f"{key}_scale",
    )
    ordering = st_module.sidebar.selectbox(
        "Order responses",
        ["original", "coefficient_strength"],
        format_func=_control_label,
        key=f"{key}_ordering",
    )
    sorting_predictor = st_module.sidebar.selectbox(
        "Sorting predictor",
        [
            *cell["column_names"].tolist(),
            "left_reward_minus_omission",
            "right_reward_minus_omission",
        ],
        key=f"{key}_{cell['meta']['settings']['encoding']}_predictor",
    )
    first = int(
        st_module.sidebar.number_input(
            "First ordered row",
            min_value=1,
            max_value=len(ids),
            value=1,
            step=1,
            key=f"{row_key}_first",
        )
    )
    count = int(
        st_module.sidebar.number_input(
            "Displayed rows",
            min_value=1,
            max_value=len(ids),
            value=min(10, len(ids)),
            step=1,
            key=f"{row_key}_count",
        )
    )
    selected = st_module.sidebar.selectbox(
        "Selected response", ids, key=f"{row_key}_selected"
    )
    prediction_scope = st_module.sidebar.selectbox(
        "Prediction scope",
        ["held_out", "reference"],
        index=0,
        format_func=_control_label,
        key=f"{key}_{entry['id']}_prediction_scope",
    )
    visible, revision = render_timeline_navigation(st_module, prepared["behavior"], key)
    available = available_event_tracks(prepared["behavior"].events)
    event_tracks = tuple(
        st_module.sidebar.multiselect(
            "Event tracks", options=available, default=available, key=f"{key}_events"
        )
    )
    show_context = st_module.sidebar.checkbox(
        "Show logged context", value=True, key=f"{key}_context"
    )
    bounded = st_module.sidebar.checkbox(
        "Load only visible neural interval",
        value=cell["meta"]["bin_width_ms"] == 20,
        key=f"{key}_bounded",
    )
    loaded = session_time_bounds(prepared["behavior"])
    if bounded:
        low, high = session_time_bounds(prepared["behavior"])
        duration = visible[1] - visible[0]
        loaded = (max(low, visible[0] - duration), min(high, visible[1] + duration))
        st_module.caption(
            f"Neural interval loaded: {loaded[0]:.3f}-{loaded[1]:.3f} s. "
            "Use the time controls or Jump to load another interval."
        )
    figure = plot_encoding_timeline(
        prepared,
        cell,
        coefficient_scale=coefficient_scale,
        heatmap_scale=heatmap_scale,
        zscore_limit=zscore_limit,
        ordering=ordering,
        sorting_predictor=sorting_predictor,
        row_start=first - 1,
        row_count=count,
        selected_response=selected,
        prediction_scope=prediction_scope,
        time_range_s=visible,
        loaded_range_s=loaded,
        event_tracks=event_tracks,
        show_context=show_context,
        viewport_revision=revision,
        inspection_penalty_mode=penalty_mode,
    )
    render_timeline_chart(figure, key=f"{scope}_chart")
    penalty = (
        "unpenalized OLS"
        if meta["configured_alpha"] is None
        else (
            f"alpha={meta['configured_alpha']:g}; feature penalties={meta['feature_penalties']}"
        )
    )
    st_module.caption(
        f"{region}; {representation}; {meta['response_units']}; [-1,+1) s; {meta['bin_width_ms']} ms bins; {penalty}. "
        f"{meta['row_count']} eligible bins from {meta['trial_count']} trials."
    )
    if representation == "pcs":
        st_module.caption(
            "PC regression is conditional on fixed descriptive session PCA, which includes test-block neural activity."
        )
    st_module.caption(
        "Coefficient scaling changes coefficients and their strength ordering. "
        "Reward-minus-omission contrasts always use raw response units. Positive outcome slopes alone do not establish reward selectivity."
    )
    if heatmap_scale == "within_unit_zscore":
        st_module.caption(
            "Each unit's heatmap uses its own mean and population SD over all saved [-1,+1) bins, "
            "excluding gaps. Constant units display zero. Panning preserves normalization; "
            f"colors saturate outside +/-{zscore_limit:g} SD. Prediction traces remain Hz."
        )
    if not meta["cv_available"]:
        st_module.warning(f"Primary five-fold scores unavailable: {meta['cv_reason']}")
    ordered = order_responses(
        cell,
        coefficient_scale=coefficient_scale,
        ordering=ordering,
        sorting_predictor=sorting_predictor,
    )[first - 1 : first - 1 + count]
    slopes = np.array(
        cell["reference_slopes"]
        if coefficient_scale == "raw"
        else cell["standardized_slopes"],
        copy=True,
    )
    slopes[~cell["predictor_supported"]] = np.nan
    table = pd.DataFrame(
        slopes[:, ordered].T,
        columns=cell["column_names"].tolist(),
        index=cell["response_ids"][ordered],
    )
    contrasts = contrast_values(cell)
    for index, name in enumerate(
        ("left_reward_minus_omission", "right_reward_minus_omission")
    ):
        table[f"{name} ({meta['response_units']})"] = contrasts[index, ordered]
        if not cell["contrast_valid"][index]:
            st_module.caption(f"{name} unavailable: {cell['contrast_reasons'][index]}")
    table["Held-out MSE"] = cell["mse"][ordered]
    table["Held-out R-squared"] = cell["r2"][ordered]
    st_module.dataframe(table)
    selected_index = ids.index(selected)
    st_module.caption(
        f"{selected} reference intercept ({meta['response_units']}): "
        f"{_metric_label(cell['reference_intercept'][selected_index])}; "
        f"held-out MSE: {_metric_label(cell['mse'][selected_index])}; "
        f"held-out R-squared: {_metric_label(cell['r2'][selected_index])}."
    )
    if st_module.button(
        f"Export PNGs for {visible[0]:.3f}-{visible[1]:.3f} s", key=f"{key}_export"
    ):
        from src.neural_analysis.encoding.plotting import save_inspection_figures

        stamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d_%H%M%S_%f")
        destination = directory / "figures" / f"inspection_{stamp}"
        paths = save_inspection_figures(
            destination,
            prepared,
            cell,
            time_range_s=visible,
            selected_response=selected,
            coefficient_scale=coefficient_scale,
            heatmap_scale=heatmap_scale,
            zscore_limit=zscore_limit,
            prediction_scope=prediction_scope,
            inspection_penalty_mode=penalty_mode,
        )
        for output in paths:
            st_module.download_button(
                output.name,
                data=output.read_bytes(),
                file_name=output.name,
                mime="image/png",
            )
    with st_module.expander("Saved fit details"):
        st_module.json(meta)


# Saved-file helpers. Cache activity independently from selected prediction cells.
@st.cache_data(show_spinner=False, max_entries=2)
def load_preparation_cached(path: str, size_bytes: int, mtime_ns: int) -> dict:
    """Input NPZ path/byte count/mtime cache key; return shared native-axis saved preparation."""
    return results.load_preparation(path)


@st.cache_data(show_spinner=False, max_entries=2)
def load_cell_cached(path: str, size_bytes: int, mtime_ns: int) -> dict:
    """Input NPZ path/byte count/mtime cache key; return one cell's native coefficient/prediction arrays."""
    return results.load_cell(path)


def _read_completed_metadata(directory: Path) -> tuple[dict, dict, list[dict]]:
    """Input run Path; return manifest/progress/metadata entries without reading activity/prediction arrays."""
    manifest = results.load_run_manifest(directory)
    progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    entries = []
    for cell_id, relative in progress["completed_cells"].items():
        with np.load(directory / relative, allow_pickle=True) as archive:
            meta = archive["meta"].item()
        _check_cell_metadata(meta, cell_id, manifest)
        entries.append({"id": cell_id, "path": relative, "meta": meta})
    return manifest, progress, entries


def _check_cell_metadata(meta: dict, cell_id: str, manifest: dict) -> None:
    """Input saved cell metadata/ID/manifest; enforce model/preparation identity, return None, no units."""
    if not isinstance(meta, dict) or meta.get("cell_id") != cell_id:
        raise ValueError("Saved cell identity differs from the run index")
    if any(
        meta.get(key) != manifest["identities"][key]
        for key in ("preparation_id", "model_id")
    ):
        raise ValueError("Saved cell model/preparation identity differs from this run")


def _configuration_controls(
    st_module: object, entries: list[dict], key: str
) -> tuple[dict, str | None]:
    """Input widget interface/completed-cell metadata list/key; return selected cell and penalty alias.

    Options expose only saved combinations, including partial runs. Scientific
    controls are selectors here; they never launch preparation or fitting.
    """
    remaining = entries
    preferred_order = {
        "region": ("PFC", "HPC"),
        "representation": ("units", "pcs"),
        "encoding": ENCODINGS,
        "estimator": ESTIMATORS,
    }
    for label, field in (
        ("Region", "region"),
        ("Response representation", "representation"),
        ("Context encoding", "encoding"),
        ("Estimator", "estimator"),
    ):
        present = {entry["meta"]["settings"][field] for entry in remaining}
        options = [value for value in preferred_order[field] if value in present]
        chosen = st_module.sidebar.selectbox(
            label, options, format_func=_control_label, key=f"{key}_{field}"
        )
        remaining = [
            entry for entry in remaining if entry["meta"]["settings"][field] == chosen
        ]
    if remaining[0]["meta"]["settings"]["estimator"] == "ols":
        st_module.sidebar.selectbox(
            "Penalty comparison",
            ["unpenalized"],
            format_func=_control_label,
            key=f"{key}_penalty",
        )
        return remaining[0], None
    available_modes = {
        mode for entry in remaining for mode in entry["meta"]["penalty_aliases"]
    }
    modes = [mode for mode in PENALTY_MODES if mode in available_modes]
    mode = st_module.sidebar.selectbox(
        "Penalty comparison", modes, format_func=_control_label, key=f"{key}_penalty"
    )
    return next(
        entry for entry in remaining if mode in entry["meta"]["penalty_aliases"]
    ), mode


def _run_label(path: str) -> str:
    """Input saved-directory path string; return its human-visible basename, no physical units."""
    return Path(path).name


def _control_label(identifier: str) -> str:
    """Input saved categorical identifier; return a readable selector label, no physical units."""
    return {
        "native": "Hz",
        "within_unit_zscore": "Within-unit z-score",
        "raw": "Raw coefficients",
        "standardized": "Standardized coefficients",
        "original": "Original response order",
        "coefficient_strength": "Absolute coefficient or contrast strength",
        "units": "Units (Hz)",
        "pcs": "PCs (fixed descriptive basis)",
        "held_out": "Held-out predictions",
        "reference": "Reference-fit predictions",
        "signed_context": "Signed context",
        "separate_context": "Separate context indicators",
        "ridge": "Ridge",
        "ols": "OLS",
        "same_numeric": "Same numeric penalty",
        "matched_context": "Matched context shrinkage",
        "unpenalized": "Unpenalized",
    }.get(identifier, identifier)


def _metric_label(value: float) -> str:
    """Input scalar score/intercept in documented native units; return numeric/unavailable display text."""
    return f"{value:.6g}" if np.isfinite(value) else "unavailable"
