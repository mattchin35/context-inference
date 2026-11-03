"""Explicit preparation controls and saved-only prediction display selections."""

from dataclasses import dataclass
import json
from pathlib import Path

import streamlit as st

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.task_decoding.session_prediction.config import ANALYSIS_MODES, GENERALIZATION_MODE, MIXED_MODE
from src.neural_analysis.task_decoding.session_prediction.plotting import (
    PredictionDisplay, analysis_mode_label, prediction_target_label, training_population_label,
)
from src.neural_analysis.task_decoding.session_prediction.results import (
    SessionPredictions, discover_prediction_runs, load_prediction_result,
)
from src.neural_analysis.task_decoding.session_prediction.run_session import prepare_session_predictions


@dataclass(frozen=True)
class PredictionViewState:
    """Selected mode/status and optional saved anchor/overlay; no loading behavior.

    anchor retains saved N/W/B/T/K arrays through SessionPredictions, while
    display is None when predictions are hidden or the selected mode is unprepared.
    Only the anchor's path is retained in Streamlit session state.
    """

    analysis_mode: str
    prepared: bool
    anchor: SessionPredictions | None
    display: PredictionDisplay | None


def render_prediction_controls(
    st_module: object, session: ResolvedSession, scope: str,
    probe_roles: dict[str, str], bin_width_ms: int,
) -> PredictionViewState:
    """Return selected mode, saved background anchor and optional prediction overlay.

    session supplies metadata/session root; probe_roles and integer millisecond
    width are explicit preparation selections. Discovery/loading uses only run
    JSON/NPZ and the pinned PCA, without parsing base config or raw inputs.
    """
    options = tuple(option for option in discover_prediction_runs(session.session_root)
                    if option.session_id == session.session_id)
    st_module.sidebar.markdown("**Condition predictions**")
    mode_key = f"{scope}_prediction_analysis"
    if mode_key not in st_module.session_state:
        available = {option.analysis_mode for option in options}
        st_module.session_state[mode_key] = (MIXED_MODE if available and GENERALIZATION_MODE not in available
                                            else GENERALIZATION_MODE)
    analysis_mode = st_module.sidebar.selectbox("Analysis", options=ANALYSIS_MODES,
                                               format_func=analysis_mode_label, key=mode_key)
    options = tuple(option for option in options if option.analysis_mode == analysis_mode)
    mode_scope = f"{scope}_{analysis_mode}"
    prepare = st_module.sidebar.button("Prepare predictions", key=f"{scope}_prepare_predictions")
    recompute = st_module.sidebar.button("Recompute predictions", disabled=not options,
                                        key=f"{scope}_recompute_predictions")
    selection_key = f"{mode_scope}_prediction_run"
    show_key = f"{mode_scope}_show_predictions"
    anchor_key = f"{scope}_prediction_anchor_path"
    if prepare or recompute:
        try:
            with st_module.spinner("Preparing block-held-out session predictions..."):
                report = prepare_session_predictions(session.metadata_path, probe_roles=probe_roles,
                                                       bin_width_ms=bin_width_ms, analysis_mode=analysis_mode,
                                                       rerun=recompute)
            options = tuple(option for option in discover_prediction_runs(session.session_root)
                            if option.session_id == session.session_id and option.analysis_mode == analysis_mode)
            st_module.session_state[selection_key] = str(report.result_path)
            st_module.session_state[show_key] = True
        except (OSError, ValueError) as error:
            st_module.error(f"Could not prepare predictions: {error}")
    by_path = {str(option.result_path): option for option in options if option.session_id == session.session_id}
    if show_key not in st_module.session_state:
        st_module.session_state[show_key] = bool(by_path)
    show = st_module.sidebar.checkbox("Show predictions", disabled=not by_path, key=show_key)
    if not by_path:
        st_module.caption(f"{analysis_mode_label(analysis_mode)} predictions are not prepared. "
                          "Prepare predictions uses the selected probes/bin width and the session's task_decoding_config.json.")
        anchor = _retained_anchor(st_module, session, anchor_key)
        return PredictionViewState(analysis_mode, False, anchor, None)
    if st_module.session_state.get(selection_key) not in by_path:
        st_module.session_state[selection_key] = next(iter(by_path))
    labels = {path: option.display_label.replace(option.analysis_mode, analysis_mode_label(option.analysis_mode), 1)
              for path, option in by_path.items()}
    selected = st_module.sidebar.selectbox("Saved predictions", options=list(by_path), key=selection_key,
                                          format_func=labels.get)
    try:
        result = load_saved_prediction_cached_for_path(Path(selected), session.session_root)
        if result.inputs.timeline.session_id != session.session_id or result.inputs.analysis_mode != analysis_mode:
            raise ValueError("Saved prediction session/mode differs from the selection")
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load saved predictions: {error}")
        return PredictionViewState(analysis_mode, False, _retained_anchor(st_module, session, anchor_key), None)
    st_module.session_state[anchor_key] = selected
    inputs = result.inputs
    st_module.caption(f"Saved {analysis_mode_label(inputs.analysis_mode)}: {selected}; "
                      f"{result.meta['effective_config']['bin_width_ms']} ms. "
                      "Preparation probe/bin controls do not change this saved basis.")
    if not show:
        return PredictionViewState(analysis_mode, True, result, None)
    target_labels = {name: prediction_target_label(inputs.analysis_mode, name, inputs.target_metadata[name]["display_label"])
                     for name in inputs.target_names}
    target = st_module.sidebar.selectbox("Prediction target", options=list(inputs.target_names),
                                         format_func=target_labels.get, key=f"{mode_scope}_prediction_target")
    mode = st_module.sidebar.selectbox("Prediction PC view", options=["hidden", "raw", "weighted"],
                                       key=f"{mode_scope}_prediction_pc_mode")
    order = st_module.sidebar.selectbox("Prediction PC order", options=["original", "contribution"],
                                        disabled=mode == "hidden", key=f"{mode_scope}_prediction_pc_order")
    maximum = max((len(model.pc_ids) for model in result.regions.values()), default=1)
    count_key = f"{mode_scope}_{result.inputs.timeline.meta['configuration_id']}_prediction_pc_count"
    count = int(st_module.sidebar.number_input("Prediction visible PC count", min_value=1, max_value=maximum,
                                               value=min(10, maximum), step=1,
                                               disabled=mode == "hidden" or order == "original",
                                               help="Contribution order shows the top N PCs; original order uses regional rank ranges.",
                                               key=count_key))
    st_module.caption("Block-held-out prediction using a fixed session-wide PCA basis. "
                      "Supervised training: [-1,+1) s; prediction: [-2,+2) s. "
                      "PCA/unit scaling include held-out neural data. This is an offline readout. "
                      "State and action readouts can coincide on correct-rewarded training labels.")
    if inputs.analysis_mode == MIXED_MODE:
        st_module.caption("Mixed-outcome training includes correct-rewarded, correct-omission and incorrect trials. "
                          "State decoding targets recorded task state; action decoding targets choice. "
                          "Mixed training changes outcome composition and sample size. At fixed C, more trials reduce "
                          "regularization relative to average loss; differences do not isolate outcome composition.")
    unavailable = inputs.target_availability.get(target, "")
    if unavailable:
        st_module.info(f"{target} unavailable: {unavailable}. Other targets remain available.")
    _render_model_details(st_module, result, target)
    return PredictionViewState(analysis_mode, True, result, PredictionDisplay(result, target, mode, order, count))


def load_saved_prediction_cached_for_path(path: Path, session_root: Path) -> SessionPredictions:
    """Return validated saved arrays, cached by result and source PCA path/stat identities.

    Paths are absolute session-contained product paths. Only small JSON and
    file stats are read on rerender; exact PCA bytes are hashed on each cache miss.
    No raw behavior configuration, spike source or behavioral table is opened.
    """
    meta = json.loads((path.parent / "config.json").read_text(encoding="ascii"))
    source = session_root / meta["source_pca"]["path"]
    result_stat, source_stat = path.stat(), source.stat()
    return _load_prediction_cached(str(path), str(session_root), result_stat.st_size,
                                   result_stat.st_mtime_ns, source_stat.st_size, source_stat.st_mtime_ns)


# Rendering/cache helpers; scientific arrays and units are unchanged.
def _render_model_details(st_module: object, result: SessionPredictions, target: str) -> None:
    """Render per-fold saved counts/parameters for a selected target identifier.

    result supplies choice masks (T,F,N), native target values (N,T), and
    canonical memberships (N,C). Derive factual counts without persisting them.
    Return None; no fits, files, or source arrays are changed.
    """
    inputs = result.inputs
    target_index = inputs.target_names.index(target)
    with st_module.expander("Prediction model details"):
        st_module.write(f"{analysis_mode_label(inputs.analysis_mode)}; {training_population_label(inputs.analysis_mode)}")
        for role, model in result.regions.items():
            st_module.write(f"{role}: {len(model.pc_ids)} predictor PCs; displaying fewer rows does not change the model.")
            rows = []
            for fold in range(5):
                mask = model.training_choice_mask[target_index, fold]
                row = {"held_out_fold": fold, "status": str(model.fit_status[target_index, fold]),
                 "reason": str(model.fit_reason[target_index, fold]),
                 "training_trials": int(model.training_trial_count[target_index, fold]),
                 "training_bins": int(model.training_row_count[target_index, fold]),
                 "intercept": float(model.intercepts[target_index, fold]),
                 **{name: int((mask & inputs.condition_masks[:, inputs.condition_names.index(name)]).sum())
                    for name in ("correct_rewarded", "omission", "incorrect")}}
                if inputs.target_families[target_index] == "categorical":
                    row.update({"right_class_0": int((mask & (inputs.target_values[:, target_index] == 0)).sum()),
                                "left_class_1": int((mask & (inputs.target_values[:, target_index] == 1)).sum())})
                rows.append(row)
            st_module.dataframe(rows, hide_index=True)


def _retained_anchor(st_module: object, session: ResolvedSession, anchor_key: str) -> SessionPredictions | None:
    """Load the last validated saved path for this session, or return None with an error."""
    path = st_module.session_state.get(anchor_key)
    if path is None:
        return None
    try:
        result = load_saved_prediction_cached_for_path(Path(path), session.session_root)
        if result.inputs.timeline.session_id != session.session_id:
            raise ValueError("Retained saved background belongs to another session")
        return result
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load saved background: {error}")
        return None


@st.cache_data(show_spinner=False, max_entries=2)
def _load_prediction_cached(
    path: str, root: str, size: int, mtime_ns: int, source_size: int, source_mtime_ns: int,
) -> SessionPredictions:
    """Cache loaded N/W/B/T/F/K products by exact path/stat keys; return original units."""
    return load_prediction_result(path, session_root=root)
