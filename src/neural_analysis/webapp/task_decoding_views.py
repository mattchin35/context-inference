"""Read-only Streamlit presentation of completed task-decoding saved runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.task_decoding.plotting import (
    plot_decoding_heatmap,
    plot_unit_coefficients,
    summarize_unit_coefficients,
)
from src.neural_analysis.task_decoding.results import (
    condition_labels,
    load_task_decoding_run,
    select_condition_arrays,
)


DEFAULT_TASK_DECODING_RESULTS_ROOT = "analysis_runs"


@dataclass(frozen=True)
class TaskDecodingRunOption:
    """One validated direct-child run available to the saved-results selector.

    Attributes
    ----------
    run_directory : pathlib.Path
        Absolute contained completed-run directory.
    run_fingerprint : str
        Exact saved scientific/input/code identity.
    analysis_version : str
        Saved task-decoding analysis version.
    session_id : str
        Saved canonical session identifier.
    display_label : str
        Concise selector label derived from immutable saved metadata.
    """

    run_directory: Path
    run_fingerprint: str
    analysis_version: str
    session_id: str
    display_label: str


def resolve_task_decoding_results_root(
    session_root: Path | str,
    locator: str,
) -> Path:
    """Resolve a nonempty session-relative results root contained by the session.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Selected metadata session root.
    locator : str
        Relative directory locator, normally ``"analysis_runs"``. It changes
        discovery only and is never interpreted as a scientific setting.

    Returns
    -------
    pathlib.Path
        Absolute normalized contained result-root path. The directory need not
        exist; missing roots produce an empty discovery result.
    """
    root = Path(session_root).resolve()
    text = str(locator).strip()
    if not text:
        raise ValueError("Results root must be a nonempty session-relative path.")
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError("Results root must be relative to the selected session.")
    candidate = (root / relative).resolve()
    try:
        contained = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Results root must remain inside the selected session.") from error
    if contained == Path("."):
        raise ValueError("Results root must name a child directory of the session root.")
    return candidate


def _read_complete_state(run_directory: Path) -> bool:
    """Return whether one run has the canonical complete publication state.

    Parameters
    ----------
    run_directory : pathlib.Path
        Direct child of a contained results root.

    Returns
    -------
    bool
        True only for a JSON object with lifecycle ``complete`` and
        ``final_results_published`` exactly true.
    """
    try:
        state = json.loads(
            (run_directory / "run_state.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(state, dict)
        and state.get("lifecycle") == "complete"
        and state.get("final_results_published") is True
    )


def discover_task_decoding_runs(
    session_root: Path | str,
    results_root_locator: str = DEFAULT_TASK_DECODING_RESULTS_ROOT,
) -> tuple[TaskDecodingRunOption, ...]:
    """List valid completed direct-child runs without scanning the session tree.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Selected metadata session root.
    results_root_locator : str, default="analysis_runs"
        Contained session-relative directory whose immediate children are
        inspected. Hidden transfer directories are ignored.

    Returns
    -------
    tuple[TaskDecodingRunOption, ...]
        Valid completed runs ordered newest-name first. Each candidate is
        validated with the saved-result loader and discarded before the next
        candidate, so discovery retains no result arrays in memory.
    """
    results_root = resolve_task_decoding_results_root(session_root, results_root_locator)
    if not results_root.exists():
        return ()
    if not results_root.is_dir():
        raise ValueError("Task-decoding results root is not a directory.")
    options = []
    for child in sorted(results_root.iterdir(), key=lambda path: path.name, reverse=True):
        if child.name.startswith(".") or not child.is_dir():
            continue
        resolved_child = child.resolve()
        if resolved_child.parent != results_root or not _read_complete_state(resolved_child):
            continue
        try:
            loaded = load_task_decoding_run(resolved_child)
            meta = loaded["meta"]
            provenance = meta["provenance"]
            fingerprint = str(loaded["run_fingerprint"])
            analysis_version = str(meta["analysis_version"])
            session_id = str(provenance["session_id"])
        except (KeyError, OSError, TypeError, ValueError):
            continue
        options.append(
            TaskDecodingRunOption(
                run_directory=resolved_child,
                run_fingerprint=fingerprint,
                analysis_version=analysis_version,
                session_id=session_id,
                display_label=f"{child.name} | {analysis_version} | {fingerprint[:12]}",
            )
        )
    return tuple(options)


def _fold_detail_table(
    saved_run: Mapping[str, object],
    *,
    target: str,
    region: str,
    representation: str,
    metric: str,
    time_bin_index: int,
    condition: str = "all",
) -> pd.DataFrame:
    """Build one selected-cell outer-fold score/count/failure table.

    Parameters
    ----------
    saved_run : mapping[str, object]
        Valid loader-shaped saved run.
    target, region, representation, metric : str
        Exact labels on the saved target, region, representation, and metric
        axes. Scores are dimensionless.
    time_bin_index : int
        Zero-based saved time-bin position.
    condition : str, default="all"
        Exact saved condition whose fold rows are displayed.

    Returns
    -------
    pandas.DataFrame
        One row per outer fold with score, fit status/reason, train/test counts,
        class counts where applicable, and requested/effective feature counts.
    """
    arrays = select_condition_arrays(saved_run, condition)

    def position(axis: str, label: str) -> int:
        """Return an exact saved label position for the local table builder.

        Parameters
        ----------
        axis : str
            Name of a one-dimensional saved Unicode label array.
        label : str
            Exact label to locate.

        Returns
        -------
        int
            Zero-based axis position. Duplicate or absent labels are rejected.
        """
        matches = np.flatnonzero(arrays[axis] == label)
        if matches.size != 1:
            raise ValueError(f"Unknown saved label: {label}")
        return int(matches[0])

    target_index = position("target_labels", target)
    region_index = position("region_labels", region)
    representation_index = position("representation_labels", representation)
    metric_index = position("metric_labels", metric)
    rows = []
    for fold_position, fold_label in enumerate(arrays["fold_labels"].tolist()):
        fit_index = (
            target_index,
            region_index,
            representation_index,
            time_bin_index,
            fold_position,
        )
        score_index = (
            target_index,
            region_index,
            representation_index,
            metric_index,
            time_bin_index,
            fold_position,
        )
        train_classes = arrays["train_class_counts"][fit_index].tolist()
        test_classes = arrays["test_class_counts"][fit_index].tolist()
        rows.append(
            {
                "outer_fold": int(fold_label),
                "score": float(arrays["fold_scores"][score_index]),
                "fit_status": str(arrays["fit_status"][fit_index]),
                "failure_reason": str(arrays["fit_reason_codes"][fit_index]),
                "train_trial_count": int(arrays["train_counts"][fit_index]),
                "test_trial_count": int(arrays["test_counts"][fit_index]),
                "train_class_0/1": str(tuple(int(value) for value in train_classes)),
                "test_class_0/1": str(tuple(int(value) for value in test_classes)),
                "requested_features": int(arrays["requested_feature_counts"][fit_index]),
                "effective_features": int(arrays["effective_feature_counts"][fit_index]),
            }
        )
    return pd.DataFrame(rows)


def _render_run_identity(st_module: object, option: TaskDecodingRunOption) -> None:
    """Render immutable saved-run identity and nearby human-readable artifacts.

    Parameters
    ----------
    st_module : object
        Streamlit-compatible rendering object.
    option : TaskDecodingRunOption
        Selected completed saved run.

    Returns
    -------
    None
        Writes only UI text; no files or scientific arrays are changed.
    """
    st_module.caption(f"Run directory: {option.run_directory}")
    st_module.caption(f"Run fingerprint: {option.run_fingerprint}")
    st_module.caption(
        f"Session: {option.session_id}; analysis version: {option.analysis_version}"
    )
    summary = option.run_directory / "summary.md"
    figures = option.run_directory / "figures"
    st_module.caption(f"Summary: {summary}")
    st_module.caption(f"Default PNG directory: {figures}")


def render_task_decoding_view(st_module: object, session: ResolvedSession) -> None:
    """Render one metadata-session saved-results view without neural computation.

    Parameters
    ----------
    st_module : object
        Imported Streamlit module or a compatible test double. Controls are
        display-only and never invoke target construction, fitting, or tuning.
    session : ResolvedSession
        Metadata-resolved session exposing an absolute ``session_root`` path.

    Returns
    -------
    None
        Renders completed saved arrays, settings, heatmaps, fold coverage, and
        optional direct-unit coefficient summaries. Raw spike/behavior sources
        are never opened.
    """
    st_module.subheader("Task-variable decoding results")
    locator = st_module.text_input(
        "Results root (relative to session)",
        value=DEFAULT_TASK_DECODING_RESULTS_ROOT,
    )
    try:
        results_root = resolve_task_decoding_results_root(session.session_root, locator)
        options = discover_task_decoding_runs(session.session_root, locator)
    except (OSError, ValueError) as error:
        st_module.error(f"Invalid task-decoding results root: {error}")
        return
    if not options:
        st_module.info(
            "No completed task-variable decoding runs were found in "
            f"{results_root}. Run the analysis offline with `uv run python -m "
            "src.neural_analysis.task_decoding.run_session new <config.json>`, "
            "then return to this saved-results view."
        )
        return
    labels = [option.display_label for option in options]
    selected_label = st_module.selectbox("Completed run", options=labels)
    option = options[labels.index(selected_label)]
    try:
        saved_run = load_task_decoding_run(
            option.run_directory,
            expected_run_fingerprint=option.run_fingerprint,
        )
    except (OSError, ValueError) as error:
        st_module.error(f"Selected saved run is no longer valid: {error}")
        return
    _render_run_identity(st_module, option)
    st_module.caption("All controls below re-render saved arrays; none recompute a decoder.")
    st_module.json(saved_run["scientific_config"], expanded=False)
    arrays = saved_run["arrays"]
    condition = st_module.selectbox(
        "Condition",
        options=list(condition_labels(saved_run)),
    )
    families = set(arrays["target_families"].tolist())
    categorical_metric = "balanced_accuracy"
    if "categorical" in families:
        categorical_metric = st_module.selectbox(
            "Categorical metric",
            options=["balanced_accuracy", "auc"],
        )
        figure = plot_decoding_heatmap(
            saved_run,
            family="categorical",
            metric=categorical_metric,
            condition=condition,
        )
        try:
            st_module.pyplot(figure)
        finally:
            plt.close(figure)
    if "numerical" in families:
        figure = plot_decoding_heatmap(
            saved_run,
            family="numerical",
            metric="r2",
            condition=condition,
        )
        try:
            st_module.pyplot(figure)
        finally:
            plt.close(figure)
    region = st_module.selectbox("Region", options=arrays["region_labels"].tolist())
    representation = st_module.selectbox(
        "Representation",
        options=arrays["representation_labels"].tolist(),
    )
    target = st_module.selectbox("Target", options=arrays["target_labels"].tolist())
    time_options = [
        f"{index}: {float(center):g} s"
        for index, center in enumerate(arrays["time_bin_centers_s"].tolist())
    ]
    selected_time = st_module.selectbox("Time bin", options=time_options)
    time_bin_index = time_options.index(selected_time)
    target_index = int(np.flatnonzero(arrays["target_labels"] == target)[0])
    family = str(arrays["target_families"][target_index])
    metric = categorical_metric if family == "categorical" else "r2"
    fold_table = _fold_detail_table(
        saved_run,
        target=target,
        region=region,
        representation=representation,
        metric=metric,
        time_bin_index=time_bin_index,
        condition=condition,
    )
    st_module.subheader("Selected-cell outer-fold details")
    st_module.dataframe(fold_table, use_container_width=True, hide_index=True)
    if representation == "units":
        feature_table, coefficient_fold_table = summarize_unit_coefficients(
            saved_run,
            target=target,
            region=region,
            time_bin_index=time_bin_index,
            condition=condition,
        )
        coefficient_figure = plot_unit_coefficients(
            saved_run,
            target=target,
            region=region,
            time_bin_index=time_bin_index,
            condition=condition,
        )
        try:
            st_module.pyplot(coefficient_figure)
        finally:
            plt.close(coefficient_figure)
        st_module.subheader("Direct-unit coefficient summary")
        st_module.dataframe(feature_table, use_container_width=True, hide_index=True)
        st_module.dataframe(
            coefficient_fold_table,
            use_container_width=True,
            hide_index=True,
        )
