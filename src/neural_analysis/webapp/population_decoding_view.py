"""Independent saved-results inspector for population task decoding."""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis.population_decoding import results
from src.neural_analysis.population_decoding.targets import CONDITIONAL_SWITCH_TARGETS
from src.neural_analysis.population_decoding.plotting import plot_decoding_timeline, export_png
from src.neural_analysis.population.timeline_plotting import available_event_tracks
from src.neural_analysis.webapp.timeline_view import render_timeline_navigation
from src.neural_analysis.webapp.timeline_chart import render_timeline_chart
from src.neural_analysis.session_metadata import ResolvedSession

if TYPE_CHECKING:
    from src.neural_analysis.population_decoding.preparation import PreparedPopulation


def render_population_decoding_view(st_module: object, session: ResolvedSession) -> None:
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
    try:
        manifest = results.load_manifest(directory)
        targets = results.model_targets_for_run(manifest)
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not read population run metadata: {error}")
        return
    region = st_module.sidebar.selectbox("Region", ["PFC", "HPC"], key=key+"_region")
    display_mode = st_module.sidebar.selectbox("Display", ["Probability timelines", "Decoder-weight unit activity"],
                                               key=key+"_display")
    if display_mode == "Decoder-weight unit activity":
        from .decoder_unit_activity_view import render_decoder_unit_activity_view
        try:
            prep_path = Path(manifest["preparation_path"])
            if not prep_path.is_absolute():
                prep_path = directory / prep_path
            stat = prep_path.stat()
            prepared = load_preparation_cached(str(prep_path), stat.st_size, stat.st_mtime_ns)
            render_decoder_unit_activity_view(st_module, directory, region, key, prepared)
        except (OSError, ValueError, KeyError) as error:
            st_module.error(f"Could not load saved decoder unit activity: {error}")
        return
    representation = st_module.sidebar.selectbox("Population features", ["units", "pcs"], key=key+"_features")
    target = st_module.sidebar.selectbox("Target", list(targets), format_func=_target_label, key=key+"_target")
    try:
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
                          "This offline readout includes current-port and post-choice information; "
                          "it does not isolate switching information from the current choice.")
    if target in CONDITIONAL_SWITCH_TARGETS:
        st_module.caption(f"Current {cell['meta']['conditioning_side']} choices only; always-stay accuracy: "
            f"{score['always_negative_accuracy']}. This offline target also decodes the next chosen port.")
    _show_switch_comparison(st_module, directory, manifest, cell, representation, targets)
    if target == "outcome":
        st_module.caption("Four-class decoding can reflect chosen side. Reward/no-reward discrimination within actual side:")
        side_rows = []
        for side, rewarded, absent in (("left", 0, 2), ("right", 1, 3)):
            conditional = summary["conditional_outcomes"][side]
            side_rows.append({"Side": side.capitalize(),
                "Reward trials": int(np.sum(cell["labels"] == rewarded)),
                "No-reward trials": int(np.sum(cell["labels"] == absent)),
                "Balanced accuracy": conditional["balanced_accuracy"], "Log loss": conditional["log_loss"],
                "Reward ROC-AUC": conditional["roc_auc"][1],
                "Reward average precision": conditional["average_precision"][1]})
        st_module.dataframe(pd.DataFrame(side_rows), hide_index=True)
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


def _target_label(name: str) -> str:
    """Map a model ID string to a readable dropdown label; return string, no data units."""
    if name in CONDITIONAL_SWITCH_TARGETS:
        return f"Switch vs stay: current {'left' if name == 'left_switch_vs_stay' else 'right'}"
    return name


def _show_switch_comparison(st_module: object, directory: Path, manifest: dict, cell: dict,
                            representation: str, declared_targets: tuple[str, ...]) -> None:
    """Render three saved score scopes for selected switch cell; return None, no rescoring.

    Inputs are run Path/manifest, a saved paired cell, units/pcs ID and declared
    model IDs. A reference selection may load its published companion by file
    stat. Metrics are dimensionless; trial/bin counts retain their saved scope.
    """
    comparison = cell["meta"].get("switch_comparison")
    target = cell["meta"]["target"]
    if comparison is None and target in ("left_choice_switch", "right_choice_switch"):
        side = "left" if target == "left_choice_switch" else "right"
        companion = f"{side}_switch_vs_stay"
        if companion not in declared_targets:
            st_module.caption("This v1 run has only the original targets. An extended run adds switch-versus-stay comparisons.")
            return
        path = directory / "cells" / f"{cell['meta']['region']}_{companion}.npz"
        if not path.exists():
            st_module.info("Within-side comparison pending; the existing directional result remains available.")
            return
        try:
            stat = path.stat()
            paired = load_cell_cached(str(path), stat.st_size, stat.st_mtime_ns)
            if paired["meta"]["model_id"] != manifest["identities"]["model_id"]:
                raise ValueError("Comparison companion belongs to a different run")
            comparison = paired["meta"]["switch_comparison"]
        except (OSError, ValueError, KeyError) as error:
            st_module.error(f"Could not load within-side comparison: {error}")
            return
    if comparison is None:
        return
    side = f"Current {comparison['conditioning_side']}"
    saved = comparison["representations"][representation]
    rows = []
    for name, training, evaluation in (("reference_all_trials", "All eligible", "All eligible"),
                                     ("reference_matched_side", "All eligible", side),
                                     ("within_side", side, side)):
        result = saved[name]
        scores = result["scores"]
        rows.append({"Fit scope": training, "Score scope": evaluation, "Scoring trials": result["trial_count"],
            "Required bins": scores["required_rows"], "Predicted bins": scores["predicted_rows"],
            "Switch fraction": scores["prevalence"][1], "Balanced accuracy": scores["balanced_accuracy"],
            "Log loss": scores["log_loss"], "Switch ROC-AUC": scores["roc_auc"][1],
            "Switch average precision": scores["average_precision"][1]})
    st_module.caption("Switch comparison: rows 2 and 3 score exactly the same held-out trials/bins. "
                      "Row 2 rescores the original directional decoder; row 3 is fitted within the current side.")
    st_module.dataframe(pd.DataFrame(rows), hide_index=True)
    with st_module.expander("Matched comparison availability and fold diagnostics"):
        st_module.json(saved)


@st.cache_data(show_spinner=False, max_entries=2)
def load_preparation_cached(path: str, size_bytes: int, mtime_ns: int) -> PreparedPopulation:
    """Load saved scalar NPZ path, keyed by file bytes/mtime; return native-Hz preparation."""
    return results.load_preparation(Path(path))


@st.cache_data(show_spinner=False, max_entries=4)
def load_cell_cached(path: str, size_bytes: int, mtime_ns: int) -> dict:
    """Load saved scalar cell path, keyed by file bytes/mtime; return probabilities/metadata."""
    return results.load_cell(Path(path))
