"""Explicit preparation controls and saved-only prediction display selections."""

import json
from pathlib import Path

import streamlit as st

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.task_decoding.session_prediction.plotting import PredictionDisplay
from src.neural_analysis.task_decoding.session_prediction.results import (
    SessionPredictions, discover_prediction_runs, load_prediction_result,
)
from src.neural_analysis.task_decoding.session_prediction.run_session import prepare_session_predictions


def render_prediction_controls(
    st_module: object, session: ResolvedSession, scope: str,
    probe_roles: dict[str, str], bin_width_ms: int,
) -> PredictionDisplay | None:
    """Return selected saved display or None; fitting occurs only on Prepare/Recompute.

    session supplies metadata/session root; probe_roles and integer millisecond
    width are explicit preparation selections. Discovery/loading uses only run
    JSON/NPZ and the pinned PCA, without parsing base config or raw inputs.
    """
    options = discover_prediction_runs(session.session_root)
    st_module.sidebar.markdown("**Condition predictions**")
    prepare = st_module.sidebar.button("Prepare predictions", key=f"{scope}_prepare_predictions")
    recompute = st_module.sidebar.button("Recompute predictions", disabled=not options,
                                        key=f"{scope}_recompute_predictions")
    selection_key = f"{scope}_prediction_run"
    show_key = f"{scope}_show_predictions"
    if prepare or recompute:
        try:
            with st_module.spinner("Preparing block-held-out session predictions..."):
                report = prepare_session_predictions(session.metadata_path, probe_roles=probe_roles,
                                                       bin_width_ms=bin_width_ms, rerun=recompute)
            options = discover_prediction_runs(session.session_root)
            st_module.session_state[selection_key] = str(report.result_path)
            st_module.session_state[show_key] = True
        except (OSError, ValueError) as error:
            st_module.error(f"Could not prepare predictions: {error}")
    by_path = {str(option.result_path): option for option in options if option.session_id == session.session_id}
    if show_key not in st_module.session_state:
        st_module.session_state[show_key] = bool(by_path)
    show = st_module.sidebar.checkbox("Show predictions", disabled=not by_path, key=show_key)
    if not by_path:
        st_module.caption("Predictions are not prepared. Prepare predictions uses the selected probes/bin width and the session's task_decoding_config.json.")
        return None
    selected = st_module.sidebar.selectbox("Saved predictions", options=list(by_path), key=selection_key,
                                          format_func=_prediction_run_label)
    if not show:
        return None
    try:
        result = load_saved_prediction_cached_for_path(Path(selected), session.session_root)
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load saved predictions: {error}")
        return None
    inputs = result.inputs
    target = st_module.sidebar.selectbox("Prediction target", options=list(inputs.target_names),
                                         key=f"{scope}_prediction_target")
    mode = st_module.sidebar.selectbox("Prediction PC view", options=["hidden", "raw", "weighted"],
                                       key=f"{scope}_prediction_pc_mode")
    order = st_module.sidebar.selectbox("Prediction PC order", options=["original", "contribution"],
                                        disabled=mode == "hidden", key=f"{scope}_prediction_pc_order")
    maximum = max((len(model.pc_ids) for model in result.regions.values()), default=1)
    count = int(st_module.sidebar.number_input("Prediction visible PC count", min_value=1, max_value=maximum,
                                               value=min(10, maximum), step=1,
                                               disabled=mode == "hidden" or order == "original",
                                               help="Contribution order shows the top N PCs; original order uses regional rank ranges.",
                                               key=f"{scope}_prediction_pc_count"))
    st_module.caption(f"Saved predictions: {selected}; {result.meta['effective_config']['bin_width_ms']} ms. "
                      "Preparation probe/bin controls do not change this saved basis.")
    st_module.caption("Block-held-out prediction using a fixed session-wide PCA basis. "
                      "Supervised training: [-1,+1) s; prediction: [-2,+2) s. "
                      "PCA/unit scaling include held-out neural data. This is an offline readout. "
                      "State and action readouts can coincide on correct-rewarded training labels.")
    unavailable = inputs.target_availability.get(target, "")
    if unavailable:
        st_module.info(f"{target} unavailable: {unavailable}. Other targets remain available.")
    target_index = inputs.target_names.index(target)
    with st_module.expander("Prediction model details"):
        for role, model in result.regions.items():
            st_module.write(f"{role}: {len(model.pc_ids)} predictor PCs; displaying fewer rows does not change the model.")
            st_module.dataframe([
                {"held_out_fold": fold, "status": str(model.fit_status[target_index, fold]),
                 "reason": str(model.fit_reason[target_index, fold]),
                 "training_trials": int(model.training_trial_count[target_index, fold]),
                 "training_bins": int(model.training_row_count[target_index, fold]),
                 "intercept": float(model.intercepts[target_index, fold])}
                for fold in range(5)
            ], hide_index=True)
    return PredictionDisplay(result, target, mode, order, count)


def _prediction_run_label(path: str) -> str:
    """Return the saved run directory name for an absolute result-path widget value."""
    return Path(path).parent.name


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


@st.cache_data(show_spinner=False, max_entries=2)
def _load_prediction_cached(
    path: str, root: str, size: int, mtime_ns: int, source_size: int, source_mtime_ns: int,
) -> SessionPredictions:
    """Cache loaded N/W/B/T/F/K products by exact path/stat keys; return original units."""
    return load_prediction_result(path, session_root=root)
