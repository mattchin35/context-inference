"""Saved-only decoder-weight unit groups inside the existing population view."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.neural_analysis.decoder_unit_activity import results
from src.neural_analysis.decoder_unit_activity.plotting import plot_unit_activity, viewport_unit_activity_data
from src.neural_analysis.decoder_unit_activity.run_session import activity_for_selection, run_session
from src.neural_analysis.decoder_unit_activity.selection import GROUP_LABELS
from src.neural_analysis.population.timeline_plotting import available_event_tracks, session_time_bounds
from src.neural_analysis.population_decoding.preparation import PreparedPopulation
from .timeline_chart import render_timeline_chart
from .timeline_view import render_timeline_navigation


def render_decoder_unit_activity_view(st_module: object, directory: Path, region: str,
                                      key: str, prepared: PreparedPopulation) -> None:
    """Render one saved run/region using shared preparation, cached summaries and native activity.

    directory/key identify saved source/UI scope. prepared retains W,B,U Hz and
    original choices. Selectors do no raw loading/fits/writes. Explicit export
    persists/reloads the selection and creates an immutable PNG run. Return None.
    """
    signature = source_signature(directory, prepared)
    loaded = cached_decoder_inputs(str(directory), region, signature, _prepared=prepared)
    loaded["prepared"] = prepared
    if loaded["status"] != "ready":
        st_module.info(loaded["reason"])
        return
    cap = int(st_module.sidebar.number_input("Units per group", min_value=1, value=10, step=1, key=key+"_unit_cap"))
    scale = st_module.sidebar.radio("Activity scale", ("zscore", "hz"), format_func=scale_label, key=key+"_unit_scale")
    cache_key = (str(directory), region, signature)
    selection = cached_unit_selection(cache_key, cap, _inputs=loaded)
    display = results.displayed_units(selection["assignments"])
    st_module.caption(f"{region}: {len(display)} unique units; native {prepared.meta['bin_width_ms']} ms bins; "
        "actual choice-centered [-1,+1) s activity. Five usable folds and >=4 positive weights; "
        "highest relative category rank, one group per unit.")
    if len(loaded["cells"]) < 5:
        st_module.caption("Legacy source: eight context/choice/outcome groups; within-side switch groups are unavailable.")
    st_module.dataframe(group_counts(selection), hide_index=True)
    st_module.dataframe(display, hide_index=True)
    st_module.caption("Source population scores use required held-out bins; trial and bin support are labeled separately.")
    st_module.dataframe(diagnostic_table(loaded["diagnostics"]["models"]), hide_index=True)
    st_module.caption("Outcome weights may reflect chosen side. Within-actual-side reward/no-reward population diagnostics:")
    st_module.dataframe(diagnostic_table(loaded["diagnostics"]["outcomes"]), hide_index=True)
    with st_module.expander("Other category weights, per-fold consistency, and assignment ties"):
        table = selection["summary"].copy()
        for fold in range(5):
            table[f"fold_{fold}_weight"] = selection["arrays"]["weights"][fold].reshape(-1)
        st_module.dataframe(table, hide_index=True)
        st_module.dataframe(selection["assignments"], hide_index=True)
        st_module.json(selection["meta"]["fit_reasons"])
    if display.empty:
        st_module.info("No units satisfy the five-fold/sign-consistency rule; no weaker units were substituted.")
        return
    time_range, revision = render_timeline_navigation(st_module, prepared.behavior, key)
    tracks = tuple(st_module.sidebar.multiselect("Event overlays", available_event_tracks(prepared.behavior.events), key=key+"_unit_events"))
    full_data = cached_activity_data(cache_key, cap, scale, _selection=selection, _prepared=prepared)
    bounds = session_time_bounds(prepared.behavior)
    span = time_range[1]-time_range[0]
    loaded_range = (max(bounds[0], time_range[0]-span), min(bounds[1], time_range[1]+span))
    data = viewport_unit_activity_data(full_data, loaded_range)
    figure = plot_unit_activity(data, prepared.behavior, display, tracks, revision, time_range)
    render_timeline_chart(figure, key=key+"_unit_chart")
    st_module.caption("Groups and same-session activity are descriptive. Outcome weights were fitted only [0,+1) s; "
        "the plot includes measured pre-choice firing. Sublanes preserve overlapping windows.")
    if st_module.button("Export unit activity PNG", key=key+"_unit_export"):
        report = run_session(directory, region, max_units_per_group=cap, scale=scale,
                             time_range_s=time_range, event_tracks=tracks, inputs=loaded)
        st_module.success(f"Saved {report['figure_path']}")


def source_signature(directory: Path, prepared: PreparedPopulation) -> tuple:
    """Return run/preparation/cell file-stat and derived-code cache key, without raw reads or writes."""
    from src.neural_analysis.population_decoding.results import load_manifest
    manifest = load_manifest(directory)
    preparation_path = Path(manifest["preparation_path"])
    if not preparation_path.is_absolute():
        preparation_path = directory / preparation_path
    paths = [directory / "config.json", preparation_path, *sorted((directory / "cells").glob("*.npz"))]
    stamps = tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    return (prepared.meta["preparation_id"], results.code_identity(), stamps)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_decoder_inputs(directory: str, region: str, signature: tuple,
                           _prepared: PreparedPopulation) -> dict:
    """Cache small saved coefficient inputs by source stats; exclude shared preparation from cached value."""
    loaded = results.load_decoder_inputs(directory, region, prepared=_prepared)
    loaded.pop("prepared", None)
    return loaded


@st.cache_data(show_spinner=False, max_entries=4)
def cached_unit_selection(signature: tuple, cap: int, _inputs: dict) -> dict:
    """Cache weights/membership/statistics for one source/cap; no activity copy in the returned selection."""
    return results.build_unit_selection(_inputs, cap)


@st.cache_data(show_spinner=False, max_entries=2)
def cached_activity_data(signature: tuple, cap: int, scale: str,
                          _selection: dict, _prepared: PreparedPopulation) -> dict:
    """Cache full selected native activity by source/cap/scale; panning slices this mapping only."""
    return activity_for_selection(_selection, _prepared, scale=scale)


def group_counts(selection: dict) -> pd.DataFrame:
    """Return ten rows of dimensionless candidate/assigned/displayed counts and group support reasons."""
    rows = []
    summary, assignments = selection["summary"], selection["assignments"]
    for group, label in enumerate(GROUP_LABELS):
        candidates = summary.loc[summary.group_index == group]
        assigned = assignments.loc[assignments.group_index == group]
        rows.append({"Group": label, "Candidates": int(candidates.candidate.sum()), "Assigned": len(assigned),
            "Displayed": int(assigned.displayed.sum()), "Reason":
                "No five-fold coefficient support" if not (candidates.available_folds == 5).any() else
                "No consistent positive candidates" if not candidates.candidate.any() else ""})
    return pd.DataFrame(rows)


def diagnostic_table(records: list[dict]) -> pd.DataFrame:
    """Format score/support rows for UI only, preserving numeric counts and JSON-safe nested diagnostics."""
    table = pd.DataFrame(records)
    for column in table:
        if table[column].map(is_nested_value).any():
            table[column] = table[column].map(format_nested_value)
    return table


def is_nested_value(value: object) -> bool:
    """Return whether a scalar UI cell is a list/dict; no physical units or mutation."""
    return isinstance(value, (list, dict))


def format_nested_value(value: object) -> object:
    """Return JSON text for nested diagnostic cell, leaving scalar score/count units unchanged."""
    return json.dumps(value) if is_nested_value(value) else value


def scale_label(value: str) -> str:
    """Format zscore/hz option string for users; return label, no numerical conversion."""
    return {"zscore": "Z-score", "hz": "Hz"}[value]
