"""Summary-only inspection and explicit preparation of saved cross-condition transfers."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.task_decoding.config import SUPPORTED_BIN_WIDTH_MS
from src.neural_analysis.task_decoding.cross_condition.config import WEIGHTING_NAMES
from src.neural_analysis.task_decoding.cross_condition.results import (
    CrossConditionResult, discover_transfer_runs, load_transfer_result,
)
from src.neural_analysis.task_decoding.cross_condition.run_session import prepare_cross_condition_generalization
from src.neural_analysis.task_decoding.cross_condition.scoring import SCORING_VERSION, summarize_transfers
from src.neural_analysis.task_decoding.session_prediction.config import training_window_bounds


TARGET_LABELS = {"current_state": "Context", "current_action": "Choice", "current_reward": "Actual reward",
                 "current_action_is_correct": "Value (correctness-based)"}
DIRECTION_LABELS = {
    "current_state": ("Correct to incorrect", "Left choice to right choice", "Right choice to left choice", "Incorrect to correct"),
    "current_action": ("Correct to incorrect", "Left context to right context", "Right context to left context", "Incorrect to correct"),
    "current_reward": ("Left choice to right choice", "Left rewarded / right unrewarded", "Right rewarded / left unrewarded", "Right choice to left choice"),
    "current_action_is_correct": ("Left choice to right choice", "Left correct / right incorrect", "Right correct / left incorrect", "Right choice to left choice"),
}
POPULATION_LABELS = {"all": "All eligible", "correct": "Correct combined", "correct_rewarded": "Correct-rewarded",
                     "omission": "Correct-omission", "incorrect": "Incorrect", "actual_rewarded": "Actual rewarded",
                     "actual_unrewarded": "Actual unrewarded combined", "incorrect_unrewarded": "Incorrect-unrewarded"}
ROLE_LABELS = {"complementary_test": "Complementary test", "training": "Training (in-sample)"}
WEIGHTING_LABELS = {"equal_trial": "Equal trial", "equal_cell": "Equal cell"}


def render_cross_condition_view(st_module: object, session: ResolvedSession) -> None:
    """Render a saved all-direction summary and explicit full-bundle Prepare/Recompute controls.

    st_module is Streamlit or a compatible renderer; session supplies metadata
    and a session-local result root. Inspection reads only small saved JSON/NPZ
    products. Inputs keep N-choice labels and independent W/K regional axes;
    probabilities/accuracy are dimensionless, displayed as percentages. Only an
    explicit preparation button invokes the metadata runner. Return None; no
    raw/PCA loads, fits or writes occur on selector changes.
    """
    st_module.subheader("Cross-condition generalization")
    scope = f"cross_condition:{session.metadata_path}"
    selection_key = scope + ":run"
    options = discover_transfer_runs(session.session_root)
    st_module.caption("Prepare creates the complete bundle: sixteen directions, three supervised windows and both weightings, "
                      "up to 192 models across PFC/HPC. Compatible saved PCA is reused; any missing PCA may be prepared.")
    width = st_module.selectbox("Preparation bin width (ms)", options=SUPPORTED_BIN_WIDTH_MS, key=scope + ":prepare_width")
    prepare = st_module.button("Prepare cross-condition results", key=scope + ":prepare")
    recompute = st_module.button("Recompute cross-condition results", disabled=not options, key=scope + ":recompute")
    if prepare or recompute:
        try:
            with st_module.spinner("Preparing all cross-condition transfers..."):
                report = prepare_cross_condition_generalization(session.metadata_path, bin_width_ms=int(width), rerun=recompute)
            if report.result_path is None:
                raise ValueError("Preparation returned no saved result")
            options = discover_transfer_runs(session.session_root)
            st_module.session_state[selection_key] = str(report.result_path)
        except (OSError, ValueError) as error:
            st_module.error(f"Could not prepare cross-condition results: {error}")
    if not options:
        st_module.info("No saved cross-condition results are available. Use Prepare cross-condition results to create the full bundle.")
        return
    by_path = {str(option.result_path): option for option in options}
    if st_module.session_state.get(selection_key) not in by_path:
        st_module.session_state[selection_key] = next(iter(by_path))
    labels = {path: option.display_label for path, option in by_path.items()}
    selected = st_module.selectbox("Saved run", options=list(by_path), format_func=labels.get, key=selection_key)
    path = Path(selected)
    try:
        stats = _product_stats(path)
        packet = _load_saved_cached(str(path), *stats, SCORING_VERSION)
        report = _summaries_cached(str(path), *stats, SCORING_VERSION, _packet=packet)
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load the selected saved run: {error}")
        return
    saved_width = packet.meta["effective_config"]["bin_width_ms"]
    st_module.caption(f"Saved session {packet.meta['session_id']}: {path}; {saved_width} ms. "
                      "Preparation controls do not reinterpret saved settings.")
    target = st_module.selectbox("Target", options=packet.inputs.target_names, format_func=TARGET_LABELS.get, key=scope + ":target")
    window = st_module.selectbox("Supervised window", options=packet.inputs.window_names, format_func=_window_label, key=scope + ":window")
    weighting = st_module.selectbox("Weighting", options=WEIGHTING_NAMES, format_func=WEIGHTING_LABELS.get, key=scope + ":weighting")
    role = st_module.selectbox("Evaluation role", options=tuple(ROLE_LABELS), format_func=ROLE_LABELS.get, key=scope + ":role")
    populations = ("all", "actual_rewarded", "actual_unrewarded", "omission", "incorrect_unrewarded") if target == "current_reward" else (
        "all", "correct", "correct_rewarded", "omission", "incorrect")
    outcome_key = scope + ":outcome"
    if st_module.session_state.get(outcome_key) not in populations:
        st_module.session_state[outcome_key] = "all"
    population = st_module.selectbox("Outcome", options=populations, format_func=POPULATION_LABELS.get, key=outcome_key)
    _render_captions(st_module, packet, target, window)
    definitions = [item for item in packet.inputs.transfers if item.target_name == target]
    directions = {item.split_id: label for item, label in zip(definitions, DIRECTION_LABELS[target], strict=True)}
    selected_rows = [row for row in report if row["target_name"] == target and row["window"] == window
                     and row["weighting"] == weighting and row["role"] == role and row["population"] == population]
    st_module.dataframe(_performance_table(selected_rows, directions), hide_index=True)
    with st_module.expander("Fit details"):
        st_module.caption("Source/test counts describe eligible data before a fit failure; scored counts require available finite trial means. "
                          "Cell names and split IDs retain the exact directed definitions. Outcome selection does not fit another model.")
        st_module.dataframe(_fit_table(packet, target, window, weighting, directions), hide_index=True)
        st_module.caption(f"Dark-state exclusions: {packet.inputs.dark_state_count}. "
                          f"Excluded conflicting trial IDs: {packet.inputs.conflicts.trial_id.tolist()}.")
        if not packet.inputs.conflicts.empty:
            st_module.dataframe(packet.inputs.conflicts, hide_index=True)
    for region, reason in packet.unavailable_regions.items():
        st_module.warning(f"{region} unavailable: {reason}")


def _product_stats(path: Path) -> tuple[int, int, int, int]:
    """Return result/manifest byte-size and mtime-ns cache keys from one saved result Path; no source checks."""
    result, manifest = path.stat(), (path.parent / "config.json").stat()
    return result.st_size, result.st_mtime_ns, manifest.st_size, manifest.st_mtime_ns


@st.cache_data(show_spinner=False, max_entries=8)
def _load_saved_cached(
    path: str, size: int, mtime_ns: int, manifest_size: int, manifest_mtime_ns: int, scoring_version: str,
) -> CrossConditionResult:
    """Cache a validated N/W/H/G/S packet by path, both file stats and scoring-version string; retain original units."""
    return load_transfer_result(path)


@st.cache_data(show_spinner=False, max_entries=8)
def _summaries_cached(
    path: str, size: int, mtime_ns: int, manifest_size: int, manifest_mtime_ns: int, scoring_version: str,
    *, _packet: CrossConditionResult,
) -> list[dict]:
    """Cache all small trial-summary rows by saved-path/stat/version keys; exclude the packet itself from hashing.

    _packet retains validated N/W/H/G/S arrays. Return dimensionless fraction
    metrics/counts for all selectors; no target/window/weight/outcome selector
    can trigger another computation while the product/version keys are fixed.
    """
    return summarize_transfers(_packet.inputs, _packet.regions, _packet.unavailable_regions)


def _window_label(name: str) -> str:
    """Return a readable Full/Before/After label with exact half-open choice-relative second bounds."""
    start, stop = training_window_bounds(name)
    stop_label = f"+{stop:g}" if stop > 0 else f"{stop:g}"
    return f"{name.title()} [{start:g},{stop_label}) s"


def _render_captions(st_module: object, packet: CrossConditionResult, target: str, window: str) -> None:
    """Explain saved binary orientation, matched second interval and scientific limits without recomputation.

    packet holds saved target metadata and N/W/H/G/S arrays; target/window are
    stable selected identifiers. Class meanings come from saved labels, not
    inferred reward or current preparation controls. Return None, UI text only.
    """
    metadata = packet.inputs.target_metadata[target]
    suffix = " context" if target == "current_state" else " choice" if target == "current_action" else ""
    positive = metadata["positive_label"] + suffix
    negative = metadata["class_labels"]["0"] + suffix
    st_module.caption(f"{TARGET_LABELS[target]} - class 1: {positive}; class 0: {negative}. "
                      f"Trial outputs are mean P({positive}); the positive threshold is >= 0.5.")
    st_module.caption(f"Matched supervised training/scoring: {_window_label(window)}. "
                      "Bin probabilities are averaged once per trial before classification; no decoder outputs outside this interval are used.")
    st_module.caption("Training performance is in-sample. Complementary test trials are excluded from fitting; "
                      "each scored trial has equal evaluation weight under either fitted weighting scheme.")
    st_module.caption("Fixed session-wide PCA and unit scaling include test-condition neural activity. "
                      "This offline condition transfer is not temporal or block-held-out validation.")
    st_module.caption("Single-class populations retain ordinary accuracy; balanced accuracy requires both actual classes. "
                      "Empty or unavailable metrics are distinct from zero performance. Outcome groups may overlap and are not summed.")
    if target == "current_reward":
        st_module.caption("Actual reward uses recorded reward, including rewarded incorrect choices. Before predicts the eventual outcome; "
                          "After is post-choice, not necessarily post-delivery. These are not subjective expectation measurements.")
    elif target == "current_action_is_correct":
        st_module.caption("Value (correctness-based) is recorded correctness. Correct omissions remain class 1 here and class 0 for Actual reward; "
                          "this probability is not a Q-value, expected reward magnitude or subjective value.")


def _performance_table(rows: list[dict], directions: dict[str, str]) -> pd.DataFrame:
    """Return direction/region rows with counts, reasons and percentage accuracy, preserving 0 and missing None.

    rows contain saved-summary scalar metrics in fractions and dimensionless
    trial counts. directions maps exact split IDs to readable directed labels.
    No population/direction/region aggregation or class inversion occurs.
    """
    return pd.DataFrame([{
        "Direction": directions[row["split_id"]], "Region": row["region"],
        "Eligible trials": row["eligible_trials"], "Scored trials": row["n_trials"],
        "Class 0": row["class_0_trials"], "Class 1": row["class_1_trials"],
        "Accuracy (%)": None if row["accuracy"] is None else 100 * row["accuracy"],
        "Balanced accuracy (%)": None if row["balanced_accuracy"] is None else 100 * row["balanced_accuracy"],
        "Reason": row["reason"],
    } for row in rows])


def _fit_table(
    packet: CrossConditionResult, target: str, window: str, weighting: str, directions: dict[str, str],
) -> pd.DataFrame:
    """Return all four directed regional fit records and eligible actual-class counts, independent of outcome selection.

    packet has independent regional W/K and H/G/S parameter axes. Select one
    target/window/weighting; count axes are source/test, actual class 0/1.
    Unknown regional coverage stays None rather than fabricated zero counts.
    Exact split IDs and cell names support auditing the visible descriptions.
    """
    window_index = packet.inputs.window_names.index(window)
    weighting_index = WEIGHTING_NAMES.index(weighting)
    rows = []
    for split, definition in enumerate(packet.inputs.transfers):
        if definition.target_name != target:
            continue
        for region in ("PFC", "HPC"):
            model = packet.regions.get(region)
            if model is None:
                status, reason = "unavailable", packet.unavailable_regions[region]
                counts = [None, None, None, None]
            else:
                status = str(model.fit_status[window_index, weighting_index, split])
                reason = str(model.fit_reason[window_index, weighting_index, split])
                counts = model.eligible_cell_counts[split].reshape(-1).tolist()
            rows.append({
                "Direction": directions[definition.split_id], "Split ID": definition.split_id, "Region": region,
                "Fit status": status, "Fit reason": reason,
                "Source class 0": counts[0], "Source class 1": counts[1], "Test class 0": counts[2], "Test class 1": counts[3],
                "Source class 0 cell": definition.source_cells[0], "Source class 1 cell": definition.source_cells[1],
                "Test class 0 cell": definition.test_cells[0], "Test class 1 cell": definition.test_cells[1],
            })
    return pd.DataFrame(rows)
