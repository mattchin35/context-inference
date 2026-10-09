"""Read-only Streamlit presentation of finalized inter-regional runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.interregional.configuration import (
    ANALYSIS_VERSION,
    InterregionalAnalysisConfig,
    load_interregional_config,
)
from src.neural_analysis.interregional.persistence import (
    ENTRYPOINT,
    MANIFEST_SCHEMA_VERSION,
    discover_finalized_runs,
    load_interregional_result,
)
from src.neural_analysis.interregional.plotting import (
    plot_absolute_cv_scores,
    plot_cv_increment_summary,
    plot_granger_summary,
    plot_ols_poisson_mse_comparison,
)
from src.neural_analysis.interregional.records import InterregionalResults


DEFAULT_RESULTS_ROOT = "analysis_runs"


@dataclass(frozen=True)
class InterregionalRunOption:
    """Validated finalized run metadata retained by the saved-run selector."""

    run_directory: Path
    run_fingerprint: str
    analysis_version: str
    session_id: str
    display_label: str


def _read_manifest(path: Path) -> dict[str, object]:
    """Read one exact supported input manifest without opening result arrays."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Saved input manifest is unreadable.") from error
    expected = {
        "manifest_schema_version",
        "run_fingerprint",
        "session_id",
        "git_head",
        "entrypoint",
        "runtime_versions",
        "resolved_populations",
        "files",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Saved input manifest has an unsupported schema.")
    if value["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Saved input manifest version is unsupported.")
    if value["entrypoint"] != ENTRYPOINT:
        raise ValueError("Saved input manifest entry point is unsupported.")
    return value


def _population_label(result: InterregionalResults, role: str) -> str:
    """Return a concise probe and ordered-cluster selector label."""
    population = next(value for value in result.resolved_populations if value.role == role)
    clusters = ",".join(str(value) for value in population.cluster_ids)
    return f"{role}={population.probe_id}[{clusters}]"


def discover_interregional_runs(
    session: ResolvedSession,
) -> tuple[InterregionalRunOption, ...]:
    """Discover validated finalized runs directly below one metadata session.

    Parameters
    ----------
    session : ResolvedSession
        Selected metadata session. Only ``<session_root>/analysis_runs`` is
        inspected and no scientific source array is loaded.

    Returns
    -------
    tuple[InterregionalRunOption, ...]
        Newest-name-first valid options. Incomplete, corrupt, mismatched, and
        unsupported candidates are silently excluded from display.
    """
    root = (session.session_root / DEFAULT_RESULTS_ROOT).resolve()
    options: list[InterregionalRunOption] = []
    for run_directory in reversed(discover_finalized_runs(root)):
        try:
            config, _ = load_interregional_config(run_directory / "config.json")
            manifest = _read_manifest(run_directory / "input_manifest.json")
            fingerprint = str(manifest["run_fingerprint"])
            if not run_directory.name.endswith(f"_{fingerprint[:12]}"):
                continue
            if manifest["session_id"] != session.session_id:
                continue
            if Path(config.session_metadata_path).resolve() != session.metadata_path.resolve():
                continue
            result = load_interregional_result(
                run_directory / "result.pkl",
                config,
                trusted_run_directory=run_directory,
            )
            if result.session_id != session.session_id or result.analysis_version != ANALYSIS_VERSION:
                continue
            stages = ",".join(config.analyses)
            display = (
                f"{run_directory.name} | {result.analysis_version} | {fingerprint[:12]} | "
                f"{_population_label(result, 'PFC')} | {_population_label(result, 'HPC')} | "
                f"{stages}"
            )
        except (KeyError, OSError, StopIteration, TypeError, ValueError):
            continue
        options.append(
            InterregionalRunOption(
                run_directory=run_directory,
                run_fingerprint=fingerprint,
                analysis_version=result.analysis_version,
                session_id=result.session_id,
                display_label=display,
            )
        )
    return tuple(options)


def _ordered_present(values) -> tuple[str, ...]:
    """Return nonempty saved string labels in stable first-seen order."""
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))


def available_display_values(result: InterregionalResults) -> dict[str, tuple[str, ...]]:
    """Return display selectors derived only from rows present in a saved result."""
    cv = result.target_summaries
    granger = result.granger_scores
    scopes = []
    conditions = []
    windows = []
    representations = []
    families = []
    metrics = []
    if not cv.empty:
        scopes.extend(cv["evaluation_scope"])
        conditions.extend(cv["condition"])
        windows.extend(cv["window"])
        representations.extend(cv["representation"])
        families.extend(cv["model_family"])
        metrics.extend(cv["metric_name"])
    if not granger.empty:
        scopes.extend(granger["evaluation_scope"])
        conditions.extend(granger["condition"])
        windows.extend(granger["window"])
        representations.extend(granger["representation"])
        families.extend(granger["model_family"])
        if granger["model_family"].eq("ols").any():
            metrics.append("linear_granger")
        if granger["model_family"].eq("poisson").any():
            metrics.append("mean_deviance_improvement")
    return {
        "evaluation_scopes": _ordered_present(scopes),
        "conditions": _ordered_present(conditions),
        "windows": _ordered_present(windows),
        "representations": _ordered_present(representations),
        "model_families": _ordered_present(families),
        "metrics": _ordered_present(metrics),
    }


def _scope_display_values(
    result: InterregionalResults, scope: str
) -> dict[str, tuple[str, ...]]:
    """Return compatible selectors for one saved evaluation scope."""
    if scope == "held_out_cv":
        rows = result.target_summaries
        metrics = _ordered_present(rows["metric_name"])
    elif scope == "in_sample":
        rows = result.granger_scores
        metric_values = []
        if rows["model_family"].eq("ols").any():
            metric_values.append("linear_granger")
        if rows["model_family"].eq("poisson").any():
            metric_values.append("mean_deviance_improvement")
        metrics = tuple(metric_values)
    else:
        raise ValueError("Unknown saved evaluation scope.")
    return {
        "conditions": _ordered_present(rows["condition"]),
        "windows": _ordered_present(rows["window"]),
        "representations": _ordered_present(rows["representation"]),
        "model_families": _ordered_present(rows["model_family"]),
        "metrics": metrics,
    }


def _render_no_run_guidance(st_module: object, session: ResolvedSession) -> None:
    """Render exact offline commands without offering an in-app action."""
    config_path = session.session_root / "interregional_regression_config.json"
    prefix = "uv run python -m src.neural_analysis.interregional.run_session"
    st_module.info(
        "No validated completed inter-regional run was found for this metadata session. "
        "Legacy manual loading is unsupported; create results explicitly from the CLI."
    )
    st_module.code(f"{prefix} dry-run --config {config_path}")
    st_module.code(f"{prefix} new --config {config_path}")
    st_module.caption(
        "This view is saved-only. Loaded spike coverage is assumed complete under the "
        "saved coverage declaration."
    )


def _selected_result(
    option: InterregionalRunOption,
) -> tuple[InterregionalAnalysisConfig, InterregionalResults, dict[str, object]]:
    """Reload one selected option through the validated trusted loader."""
    config, _ = load_interregional_config(option.run_directory / "config.json")
    result = load_interregional_result(
        option.run_directory / "result.pkl",
        config,
        trusted_run_directory=option.run_directory,
    )
    manifest = _read_manifest(option.run_directory / "input_manifest.json")
    return config, result, manifest


def render_interregional_view(st_module: object, session: ResolvedSession) -> None:
    """Render one selected saved run without creating or changing filesystem paths."""
    st_module.header("Inter-regional neural regression")
    options = discover_interregional_runs(session)
    if not options:
        _render_no_run_guidance(st_module, session)
        return
    by_label = {option.display_label: option for option in options}
    selected_label = st_module.selectbox("Completed saved run", tuple(by_label))
    option = by_label[selected_label]
    config, result, manifest = _selected_result(option)
    st_module.caption(f"Run directory: {option.run_directory}")
    st_module.caption(f"Run fingerprint: {option.run_fingerprint}")
    st_module.caption(
        f"Coverage assumption: {result.coverage_assumption_version}; loaded spike coverage "
        "is assumed complete. Predictive results do not establish causality."
    )
    st_module.subheader("Saved identity and scientific configuration")
    st_module.json(
        {
            "configuration": result.configuration,
            "resolved_populations": manifest["resolved_populations"],
            "git_head": manifest["git_head"],
            "runtime_versions": manifest["runtime_versions"],
            "entrypoint": manifest["entrypoint"],
        }
    )
    selectors = available_display_values(result)
    if not selectors["evaluation_scopes"]:
        st_module.warning("The saved run has no target-level display rows.")
        st_module.dataframe(result.fold_scores)
        return
    evaluation_scope = st_module.selectbox(
        "Evaluation scope", selectors["evaluation_scopes"]
    )
    scoped = _scope_display_values(result, evaluation_scope)
    condition = st_module.selectbox("Condition", scoped["conditions"])
    scope_rows = (
        result.target_summaries
        if evaluation_scope == "held_out_cv"
        else result.granger_scores
    )
    condition_rows = scope_rows.loc[scope_rows["condition"].eq(condition)]
    window = st_module.selectbox(
        "Window", _ordered_present(condition_rows["window"])
    )
    window_rows = condition_rows.loc[condition_rows["window"].eq(window)]
    representation = st_module.selectbox(
        "Representation", _ordered_present(window_rows["representation"])
    )
    representation_rows = window_rows.loc[
        window_rows["representation"].eq(representation)
    ]
    available_families = _ordered_present(representation_rows["model_family"])
    model_family = st_module.selectbox("Model family", available_families)
    family_rows = representation_rows.loc[
        representation_rows["model_family"].eq(model_family)
    ]
    if evaluation_scope == "held_out_cv":
        compatible_metrics = _ordered_present(family_rows["metric_name"])
    elif model_family == "ols":
        compatible_metrics = ("linear_granger",)
    else:
        compatible_metrics = ("mean_deviance_improvement",)
    metric = st_module.selectbox("Metric", compatible_metrics)
    if evaluation_scope == "in_sample":
        selected_granger = result.granger_scores.loc[
            result.granger_scores["condition"].eq(condition)
            & result.granger_scores["window"].eq(window)
            & result.granger_scores["representation"].eq(representation)
            & result.granger_scores["model_family"].eq(model_family)
        ]
        st_module.subheader("Descriptive in-sample Granger scores")
        st_module.caption(
            "These magnitudes are descriptive, not significance tests and not causal "
            "evidence. No model is computed in this saved-only view."
        )
        st_module.dataframe(selected_granger)
        figure, _ = plot_granger_summary(
            result.granger_scores,
            result.population_summaries,
            representation=representation,
            model_family=model_family,
            window=window,
            coverage_assumption_version=result.coverage_assumption_version,
        )
        st_module.pyplot(figure)
        st_module.subheader("PCA fits")
        st_module.dataframe(result.pca_fits)
        return
    selected_targets = result.target_summaries.loc[
        result.target_summaries["condition"].eq(condition)
        & result.target_summaries["window"].eq(window)
        & result.target_summaries["representation"].eq(representation)
        & result.target_summaries["model_family"].eq(model_family)
        & result.target_summaries["metric_name"].eq(metric)
    ]
    st_module.subheader("Target summaries")
    st_module.dataframe(selected_targets)
    unavailable = result.fold_scores.loc[result.fold_scores["status"].ne("ok")]
    if not unavailable.empty:
        st_module.subheader("Unavailable fold rows")
        st_module.dataframe(unavailable)
    if metric in {"delta_r2", "delta_deviance_explained"} and not result.population_summaries.empty:
        figure, _ = plot_cv_increment_summary(
            result.target_summaries,
            result.population_summaries,
            representation=representation,
            model_family=model_family,
            window=window,
            coverage_assumption_version=result.coverage_assumption_version,
        )
        st_module.pyplot(figure)
    if model_family == "ols" and {"r2_restricted", "r2_full"} <= set(
        compatible_metrics
    ):
        figure, _ = plot_absolute_cv_scores(
            result.target_summaries,
            representation=representation,
            model_family=model_family,
            condition=condition,
            window=window,
            coverage_assumption_version=result.coverage_assumption_version,
        )
        st_module.pyplot(figure)
    if model_family == "poisson" and {"ols", "poisson"} <= set(
        available_families
    ):
        st_module.caption(
            "Exploratory OLS/Poisson held-out count-MSE comparison; this is not a "
            "formal test of model superiority."
        )
        figure, _ = plot_ols_poisson_mse_comparison(
            result.fold_scores,
            condition=condition,
            window=window,
            coverage_assumption_version=result.coverage_assumption_version,
        )
        st_module.pyplot(figure)
    st_module.subheader("PCA fits")
    st_module.dataframe(result.pca_fits)
