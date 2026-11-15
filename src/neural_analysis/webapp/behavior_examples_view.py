"""Single-session choice-pattern browsing backed by saved behavioral results."""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from src.neural_analysis.event_sequences import example_io, example_plotting, examples, selection
from src.neural_analysis.event_sequences.run_examples import (
    ANALYSIS_SOURCE_FILES, export_behavior_example, prepare_behavior_examples,
)
from src.neural_analysis.session_metadata import ResolvedSession


def render_behavior_examples_view(st_module: object, session: ResolvedSession) -> None:
    """Render saved choices, navigation and explicit PNG export for one session.

    st_module is Streamlit or a compatible UI object; session supplies resolved
    identity/metadata and trial-CSV paths. Source action/context conventions are
    0=right/1=left, choice_time stays UTC Unix seconds. Return None after
    rendering the selected (S,) match plus source-row context. Matching is
    prepared/reused once per source/code identity; navigation only loads cached
    behavior results. No neural data, neural metadata inventory or fitting is
    requested. Filesystem/schema failures are reported as actionable UI errors.
    """
    st_module.subheader("Behavior examples")
    scope = hashlib.sha256(str(session.metadata_path).encode("utf-8")).hexdigest()[:12]
    try:
        source = session.behavior.trial_table_file
        if source is None:
            raise ValueError("Session metadata needs a behavior trial CSV.")
        code_directory = Path(examples.__file__).parent
        token = (_file_token(source), _file_token(session.metadata_path), example_io.ANALYSIS_VERSION,
                 selection.ADDITIONAL_SELECTOR_VERSION,
                 tuple(_file_token(code_directory / name) for name in ANALYSIS_SOURCE_FILES))
        report_key, token_key = f"behavior_report_{scope}", f"behavior_token_{scope}"
        recompute = st_module.button("Recompute behavioral matches", key=f"behavior_recompute_{scope}")
        if recompute or st_module.session_state.get(token_key) != token:
            with st_module.spinner("Finding behavioral examples..."):
                report = prepare_behavior_examples(session.metadata_path, rerun=recompute)
            st_module.session_state[report_key] = report
            st_module.session_state[token_key] = token
        report = st_module.session_state[report_key]
        result_path = Path(report["result_path"])
        packet = _load_saved_examples(str(result_path), _file_token(result_path))
    except (OSError, ValueError, KeyError) as error:
        st_module.error(f"Could not load behavioral examples: {error}")
        return

    family = st_module.selectbox("Pattern", options=examples.FAMILIES, format_func=_family_label, key=f"behavior_family_{scope}")
    side = st_module.selectbox("Initial side", options=("left", "right"), key=f"behavior_side_{scope}")
    repeats = None
    if family != "lucky_omission_switch":
        repeats = st_module.slider("Incorrect repeats", min_value=1, max_value=3 if family == "indecisive_transition" else 4,
                                   value=1, key=f"behavior_repeats_{scope}_{family}")
    context_columns = st_module.columns(2)
    before = int(context_columns[0].number_input("Trials before", min_value=0, value=5, step=1, key=f"behavior_before_{scope}"))
    after = int(context_columns[1].number_input("Trials after", min_value=0, value=5, step=1, key=f"behavior_after_{scope}"))
    condition = (family, side, repeats)
    positions = packet["matches"][condition]
    count = len(positions)
    st_module.caption(f"{count} matching episodes; chronological order. Probe and omission-probe returns may be rewarded or omissions.")
    if not count:
        st_module.info("No matching episodes for these settings. Change the pattern, initial side, or repeat count.")
        return

    index_key = f"behavior_example_{scope}_{family}_{side}_{repeats}"
    if st_module.session_state.get(index_key, 0) >= count:
        st_module.session_state[index_key] = 0
    current = st_module.session_state.get(index_key, 0)
    controls = st_module.columns((1, 3, 1))
    controls[0].button("Previous", disabled=current == 0, key=f"behavior_previous_{scope}",
                       on_click=_step_example, args=(st_module.session_state, index_key, -1, count))
    index = controls[1].selectbox("Example", options=range(count), format_func=_example_label, key=index_key)
    controls[2].button("Next", disabled=current == count - 1, key=f"behavior_next_{scope}",
                       on_click=_step_example, args=(st_module.session_state, index_key, 1, count))
    view = example_plotting.build_example_view(
        packet["trials"], positions[index], session_id=session.session_id, condition=condition,
        trials_before=before, trials_after=after,
        probe_return_outcome=packet["meta"]["parameters"]["probe_return_outcome"],
    )
    st_module.plotly_chart(example_plotting.plotly_figure(view), width="stretch", key=f"behavior_chart_{scope}")
    st_module.caption(view["caption"])
    if st_module.button("Export PNG", key=f"behavior_export_{scope}"):
        try:
            output = export_behavior_example(result_path, condition, int(index), trials_before=before, trials_after=after)
            st_module.success(f"PNG and selection metadata saved to {output}")
        except (OSError, ValueError) as error:
            st_module.error(f"Could not export behavioral example: {error}")


@st.cache_resource(show_spinner=False, max_entries=2)
def _load_saved_examples(path: str, file_token: tuple[int, int]) -> dict:
    """Map trusted NPZ path and (size bytes, mtime ns) cache identity to a saved packet.

    Return {trials:(N,C) DataFrame, matches:integer (E,S) arrays, meta:dict};
    original UTC seconds/action conventions are unchanged. Cache is read-only
    by convention; drawing always copies the selected source rows.
    """
    return example_io.load_examples(path)


def _file_token(path: Path) -> tuple[int, int]:
    """Map an existing file Path to (size bytes, modification time ns) without reading content."""
    details = path.stat()
    return details.st_size, details.st_mtime_ns


def _family_label(family: str) -> str:
    """Map a unitless family slug to its human-readable widget label."""
    return family.replace("_", " ").capitalize()


def _example_label(index: int) -> str:
    """Map zero-based integer episode index to a one-based displayed number string."""
    return str(index + 1)


def _step_example(state: object, key: str, delta: int, count: int) -> None:
    """Update widget state before rerender; key/delta/count are unitless navigation values.

    state is a mutable Streamlit session-state mapping. Clamp integer index
    to [0,count-1], mutate that UI value only, and return None.
    """
    state[key] = min(count - 1, max(0, state.get(key, 0) + delta))
