"""Independent saved-results inspector for population task decoding."""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
import pandas as pd
import streamlit as st

from src.neural_analysis.population_decoding import results
from src.neural_analysis.population_decoding.targets import TARGET_NAMES
from src.neural_analysis.population_decoding.plotting import plot_decoding_timeline, export_png
from src.neural_analysis.population.timeline_plotting import available_event_tracks
from src.neural_analysis.webapp.timeline_view import render_timeline_navigation
from src.neural_analysis.webapp.timeline_chart import render_timeline_chart


def render_population_decoding_view(st_module: object, session: object) -> None:
    """Render saved population probabilities/metrics for a resolved metadata session.

    session supplies only session_root/metadata_path. File-stat-keyed caches load
    one preparation and selected paired cell. All times are elapsed seconds;
    probabilities are dimensionless. No sources, preparation, or fitting are
    invoked. Optional PNG export also consumes saved data only. Return None.
    """
    st_module.subheader("Population task decoding")
    scope = "population_decoding_" + hashlib.sha256(str(session.metadata_path).encode()).hexdigest()[:10]
    runs = results.discover_runs(session.session_root / "analysis_runs")
    if not runs:
        st_module.info("No saved population task decoding runs are available for this session.")
        st_module.code("uv run python -m src.neural_analysis.population_decoding.run_session new "
                       f'--config "{session.session_root / "population_decoding.json"}"', language="bash")
        return
    directory = Path(st_module.sidebar.selectbox("Saved population run", [str(path) for path in runs], key=scope+"_run"))
    key = scope + "_" + hashlib.sha256(str(directory).encode()).hexdigest()[:10]
    region = st_module.sidebar.selectbox("Region", ["PFC", "HPC"], key=key+"_region")
    representation = st_module.sidebar.selectbox("Population features", ["units", "pcs"], key=key+"_features")
    target = st_module.sidebar.selectbox("Target", list(TARGET_NAMES), key=key+"_target")
    try:
        manifest = results.load_manifest(directory)
        path = directory / "cells" / f"{region}_{target}.npz"
        if not path.exists():
            st_module.info("This target has not completed yet; completed targets remain available.")
            return
        prep_path = Path(manifest["preparation_path"])
        stat = prep_path.stat()
        prepared = load_preparation_cached(str(prep_path), stat.st_size, stat.st_mtime_ns)
        stat = path.stat()
        cell = load_cell_cached(str(path), stat.st_size, stat.st_mtime_ns)
        if (cell["meta"]["model_id"] != manifest["identities"]["model_id"]
                or prepared.meta["preparation_id"] != manifest["identities"]["preparation_id"]):
            raise ValueError("Saved preparation/model identities differ from the run")
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load saved population decoding: {error}")
        return
    if cell["meta"]["unavailable_reason"]:
        st_module.warning(cell["meta"]["unavailable_reason"])
    summary = cell["meta"]["representations"][representation]
    score = summary["pooled"]
    st_module.caption(f"{summary['eligible_trials']} eligible trials; choice-relative "
        f"{cell['meta']['supervised_interval_s']} s; training-fold scaling/PCA; pooled block-held-out scores.")
    st_module.dataframe(pd.DataFrame([{"Balanced accuracy": score["balanced_accuracy"],
        "Log loss": score["log_loss"], "Accuracy": score["accuracy"],
        "Required bins": score["required_rows"], "Predicted bins": score["predicted_rows"]}]), hide_index=True)
    st_module.dataframe(pd.DataFrame({"Class": cell["meta"]["class_names"],
        "Trials": summary["trial_class_counts"], "Bins": score["class_counts"], "Prevalence": score["prevalence"],
        "Precision": score["precision"], "Recall": score["recall"], "ROC-AUC": score["roc_auc"],
        "Average precision": score["average_precision"]}), hide_index=True)
    st_module.caption("Confusion counts: actual classes in rows, predicted classes in columns.")
    st_module.dataframe(pd.DataFrame(score["confusion"], index=cell["meta"]["class_names"], columns=cell["meta"]["class_names"]))
    if target.endswith("choice_switch"):
        st_module.caption(f"Always-negative accuracy: {score['always_negative_accuracy']}. "
                          "The full window includes post-choice activity; this is an offline next-choice label readout.")
    if target == "outcome":
        st_module.caption("Four-class decoding can reflect chosen side. Reward/no-reward discrimination within actual side:")
        st_module.json(summary["conditional_outcomes"])
    if score["metric_reasons"]:
        st_module.json({"Unavailable metrics": score["metric_reasons"]})
    with st_module.expander("Fold support and fit diagnostics"):
        st_module.json(summary["folds"])
    time_range, revision = render_timeline_navigation(st_module, prepared.behavior, key)
    tracks = tuple(st_module.sidebar.multiselect("Event overlays", available_event_tracks(prepared.behavior.events), key=key+"_events"))
    figure = plot_decoding_timeline(prepared, cell, representation, time_range, tracks, revision)
    render_timeline_chart(figure, key=key+"_chart")
    if st_module.button("Export probability PNG", key=key+"_export"):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = directory / "figures" / f"{region}_{target}_{representation}_{stamp}.png"
        export_png(output, prepared, cell, representation, time_range)
        st_module.success(f"Saved {output}")


@st.cache_data(show_spinner=False, max_entries=2)
def load_preparation_cached(path: str, size_bytes: int, mtime_ns: int) -> object:
    """Load saved scalar NPZ path, keyed by file bytes/mtime; return native-Hz preparation."""
    return results.load_preparation(Path(path))


@st.cache_data(show_spinner=False, max_entries=4)
def load_cell_cached(path: str, size_bytes: int, mtime_ns: int) -> dict:
    """Load saved scalar cell path, keyed by file bytes/mtime; return probabilities/metadata."""
    return results.load_cell(Path(path))
