"""Explicit communication-subspace controls and source-independent results."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

from src.neural_analysis.communication_subspace.configuration import CommunicationConfig
from src.neural_analysis.communication_subspace import preparation, results, run_session
from src.neural_analysis.communication_subspace.plotting import plot_rank_curves
from src.neural_analysis.population.run_session_timeline import resolve_timeline_probe_roles
from src.neural_analysis.session_metadata import ResolvedSession


def render_communication_subspace_view(st_module: object, session: ResolvedSession) -> None:
    """Render saved curves, paired-input readiness, and explicit run controls.

    st_module is a Streamlit-compatible interface and session contains resolved
    metadata/path records only. The function inspects small metadata sources
    for readiness, never loads spikes or fits unless Run/Recompute is clicked.
    Saved results load without raw sources and retain their archived caption.
    Return None; displayed scores have dimensionless held-out R2 units.
    """
    st_module.subheader("Communication subspace")
    scope = "communication_" + hashlib.sha256(str(session.metadata_path).encode()).hexdigest()[:10]
    saved_headers = results.discover_results(session.session_root)
    selected_info, selected_result = _saved_result_controls(st_module, saved_headers, scope)
    role_defaults = _role_defaults(session, selected_result)
    roles, width_ms, seed = _preparation_controls(st_module, session, scope, role_defaults)

    inspection = None
    readiness_error = None
    readiness_config = None
    if len(roles) == 2 and len(set(roles.values())) == 2:
        readiness_config = CommunicationConfig(session.metadata_path, roles, width_ms, seed)
        try:
            inspection = preparation.inspect_session_inputs(readiness_config)
            draws = preparation.draw_populations(inspection.unit_ids, seed=seed)
            times = inspection.choices.iloc[inspection.choice_positions].choice_time.to_numpy(dtype=float)
            preparation.build_validation_splits(times)
            _show_readiness(st_module, inspection, draws)
        except (OSError, ValueError, KeyError) as error:
            readiness_error = str(error)
    else:
        readiness_error = "Assign distinct PFC and HPC probes to check readiness."

    if readiness_error:
        st_module.warning(f"Run unavailable: {readiness_error}")

    valid = inspection is not None and readiness_error is None
    if selected_result is not None:
        configuration = selected_result.meta.get("configuration", selected_info.configuration)
        st_module.caption(
            f"Saved result: {selected_info.completed_utc} | "
            f"{configuration.get('bin_width_ms', '?')} ms | seed {configuration.get('seed', '?')} | "
            f"PFC {selected_result.probe_roles.get('PFC', '?')} / "
            f"HPC {selected_result.probe_roles.get('HPC', '?')}. "
            "Preparation controls apply only to the next explicit action."
        )
        figure = plot_rank_curves(selected_result)
        try:
            st_module.pyplot(figure, clear_figure=True)
        finally:
            plt.close(figure)
        _show_result_details(st_module, selected_result)

    run_col, recompute_col = st_module.columns(2)
    run_clicked = run_col.button("Run", key=f"{scope}_run", disabled=not valid)
    recompute_clicked = recompute_col.button(
        "Recompute", key=f"{scope}_recompute", disabled=not valid or selected_result is None,
    )
    if run_clicked or recompute_clicked:
        assert readiness_config is not None
        progress = st_module.progress(0.0, text="Starting communication-subspace analysis")

        def update_progress(completed: int, total: int, message: str) -> None:
            """Update Streamlit progress for integer completed/total draw-fold tasks."""
            progress.progress(completed / total if total else 0.0, text=message)

        try:
            report = run_session.run_session(
                readiness_config, rerun=bool(recompute_clicked), progress=update_progress,
            )
            st_module.info(f"Run status: {report.status}; configuration {report.configuration_id}.")
            if report.result_path is not None:
                st_module.session_state[f"{scope}_pending_result"] = str(report.result_path)
                st_module.rerun()
        except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            st_module.error(f"Communication-subspace run failed: {error}")


@st.cache_data(show_spinner=False, max_entries=4)
def _load_result_cached(path: str, size_bytes: int, mtime_ns: int):
    """Load a saved result keyed by local archive path, size bytes, and mtime ns.

    path identifies a project-generated NPZ. Return CommunicationResult with
    unchanged choice/bin/unit axes and dimensionless score tables; raw inputs
    are not accessed. File metadata invalidates the cache after archive changes.
    """
    return results.load_result(path)


def _saved_result_controls(st_module: object, headers: tuple, scope: str):
    """Return selected ResultInfo and loaded result, or (None, None).

    headers contains newest-first ResultInfo records; scope identifies widget
    state. Selection is a saved archive path, with None meaning no result.
    Loading uses only that archive and its filesystem stat, never raw session
    sources. Corrupt results report an error and leave no substituted result.
    """
    labels = {None: "No saved result"}
    options = []
    for header in headers:
        path = str(header.path)
        config = header.configuration
        roles = config.get("probe_roles", {})
        labels[path] = (
            f"{header.completed_utc} | {config.get('bin_width_ms', '?')} ms | "
            f"PFC {roles.get('PFC', '?')} / HPC {roles.get('HPC', '?')}"
        )
        options.append(path)
    options.append(None)
    key = f"{scope}_saved"
    pending = st_module.session_state.pop(f"{scope}_pending_result", None)
    if pending in options and pending is not None:
        st_module.session_state[key] = pending
    if key in st_module.session_state and st_module.session_state[key] not in options:
        del st_module.session_state[key]
    selected = st_module.selectbox(
        "Saved communication-subspace result", options=options,
        index=0 if headers else len(options) - 1,
        format_func=labels.__getitem__, key=key,
    )
    if selected is None:
        return None, None
    info = next(header for header in headers if str(header.path) == selected)
    try:
        stat = info.path.stat()
        return info, _load_result_cached(str(info.path), stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, KeyError, EOFError) as error:
        st_module.error(f"Could not load saved communication-subspace result: {error}")
        return info, None


def _role_defaults(session: ResolvedSession, result) -> dict[str, str]:
    """Return metadata-declared roles, or saved roles, without guessing probes.

    session has resolved metadata; result may be a saved CommunicationResult.
    Return only known PFC/HPC assignments. Version-2 metadata with no roles and
    no selected archive returns an empty mapping, requiring explicit choices.
    """
    defaults = resolve_timeline_probe_roles(session, None)
    if defaults:
        return defaults
    if result is not None and set(result.probe_roles) == {"PFC", "HPC"}:
        return dict(result.probe_roles)
    return {}


def _preparation_controls(st_module: object, session: ResolvedSession, scope: str,
                          role_defaults: dict[str, str]) -> tuple[dict[str, str], int, int]:
    """Return explicit role mapping, bin width milliseconds, and nonnegative seed.

    session supplies declared probe IDs; role_defaults contains only trusted
    metadata or archive assignments. Unassigned probes remain unset. Width is
    one of 100/500 ms and seed is dimensionless; widgets do no loading/fitting.
    """
    options = ["Not assigned"] + [probe.probe_id for probe in session.probes]
    roles = {}
    for region in ("PFC", "HPC"):
        default = role_defaults.get(region)
        index = options.index(default) if default in options else 0
        selected = st_module.selectbox(
            f"{region} probe", options=options, index=index, key=f"{scope}_{region}_probe",
        )
        if selected != "Not assigned":
            roles[region] = selected
    width_ms = st_module.selectbox(
        "Bin width (ms)", options=[100, 500], index=0, key=f"{scope}_bin_width_ms",
    )
    seed = int(st_module.number_input(
        "Random seed", min_value=0, step=1, value=0, key=f"{scope}_seed",
    ))
    st_module.caption(
        "Fixed choice window: [-2, +2) s; five matched population draws; five "
        "chronological whole-choice folds with overlap purging; symmetric count matching."
    )
    return roles, width_ms, seed


def _show_readiness(st_module: object, inspection, draws: tuple) -> None:
    """Display metadata-only unit, choice, and coverage readiness diagnostics.

    inspection is SessionInspection with qualified regional unit IDs, common
    choice positions and UTC-second coverage bounds; draws contains seeded
    matched populations. Values are counts or seconds and no neural arrays are
    opened. Return None.
    """
    common_choices = len(inspection.choice_positions)
    n = len(draws[0].source_indices["PFC"])
    st_module.success("PFC/HPC inputs are ready for an explicit run.")
    st_module.dataframe({
        "Region": ["PFC", "HPC"],
        "Good units": [len(inspection.unit_ids[region]) for region in ("PFC", "HPC")],
        "Matched units per source/target": [n, n],
        "Common eligible choices": [common_choices, common_choices],
        "Trusted coverage (UTC s)": [inspection.coverage_s[region] for region in ("PFC", "HPC")],
    }, hide_index=True, use_container_width=True)


def _show_result_details(st_module: object, result) -> None:
    """Display saved score counts and diagnostic rows without recomputation.

    result is a loaded CommunicationResult; fold/draw/curve tables retain
    dimensionless R2 and explicit validity/status fields. Return None and do
    not inspect source files or fit models.
    """
    st_module.caption(
        f"Status: {result.computation.status}; {len(result.choice_positions)} eligible choices; "
        f"{len(result.computation.draws)} population draws; {len(result.computation.splits)} outer folds."
    )
    st_module.dataframe(result.computation.curve_scores, hide_index=True, use_container_width=True)
    if not result.computation.fold_scores.empty:
        st_module.dataframe(result.computation.fold_scores, hide_index=True, use_container_width=True)
    with st_module.expander("Saved model and validation diagnostics"):
        st_module.dataframe(_population_diagnostics(result), hide_index=True, use_container_width=True)
        st_module.dataframe(_split_diagnostics(result), hide_index=True, use_container_width=True)
        st_module.dataframe(_fit_diagnostics(result), hide_index=True, use_container_width=True)


def _population_diagnostics(result) -> pd.DataFrame:
    """Build a table of saved source/target unit identities for each draw.

    result has regional qualified unit-ID tuples and population draw indices
    shaped (n,). Return one row per draw and region with actual source/target
    IDs and matched count; no spike activity is opened or selected here.
    """
    rows = []
    for draw_number, draw in enumerate(result.computation.draws):
        for region in ("PFC", "HPC"):
            source = [result.unit_ids[region][int(index)] for index in draw.source_indices[region]]
            target = [result.unit_ids[region][int(index)] for index in draw.target_indices[region]]
            rows.append({
                "draw": draw_number, "region": region,
                "matched_units_per_population": len(source),
                "source_unit_ids": ", ".join(source),
                "target_unit_ids": ", ".join(target),
            })
    return pd.DataFrame(rows, columns=[
        "draw", "region", "matched_units_per_population", "source_unit_ids", "target_unit_ids",
    ])


def _split_diagnostics(result) -> pd.DataFrame:
    """Build saved whole-choice split and purge counts for every outer fold.

    result.computation.splits contains choice-axis integer indices. Return
    counts only (dimensionless choice observations), including outer and inner
    training, validation, test, and overlap-purged choices; no split is rebuilt.
    """
    rows = []
    for fold_number, split in enumerate(result.computation.splits):
        rows.append({
            "fold": fold_number,
            "outer_train_choices": len(split.outer_train_indices),
            "outer_test_choices": len(split.outer_test_indices),
            "outer_purged_choices": len(split.outer_purged_indices),
            "inner_train_choices": len(split.inner_train_indices),
            "inner_validation_choices": len(split.inner_validation_indices),
            "inner_purged_choices": len(split.inner_purged_indices),
        })
    return pd.DataFrame(rows, columns=[
        "fold", "outer_train_choices", "outer_test_choices", "outer_purged_choices",
        "inner_train_choices", "inner_validation_choices", "inner_purged_choices",
    ])


def _fit_diagnostics(result) -> pd.DataFrame:
    """Build saved fold fit details including alpha, variable units, and rank.

    result.computation.fit_records contains outer-fold transforms and compact
    ridge/RRR factors. Return one row per draw/direction/fold with selected
    alpha, actual matched IDs, variable source/target counts, and maximum
    supported coefficient rank. Counts and rank are dimensionless; alpha is
    dimensionless under the normalized objective.
    """
    rows = []
    for fit in result.computation.fit_records:
        rows.append({
            "draw": fit.draw, "direction": fit.direction, "fold": fit.fold,
            "alpha": fit.alpha,
            "source_unit_ids": ", ".join(fit.source_unit_ids),
            "target_unit_ids": ", ".join(fit.target_unit_ids),
            "effective_source_units": int(np.asarray(fit.source_transform.variable_mask).sum()),
            "effective_target_units": int(np.asarray(fit.target_transform.variable_mask).sum()),
            "max_rank": fit.decomposition.max_rank,
        })
    return pd.DataFrame(rows, columns=[
        "draw", "direction", "fold", "alpha", "source_unit_ids", "target_unit_ids",
        "effective_source_units", "effective_target_units", "max_rank",
    ])
