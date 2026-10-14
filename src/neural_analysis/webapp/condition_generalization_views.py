"""Read-only Streamlit presentation of completed condition-generalization runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.task_decoding.pooled.config import ANALYSIS_VERSION
from src.neural_analysis.task_decoding.pooled.plotting import plot_pooled_timecourses
from src.neural_analysis.task_decoding.pooled.results import load_pooled_run


DEFAULT_CONDITION_GENERALIZATION_RESULTS_ROOT = "analysis_runs"
RUN_DIRECTORY_PREFIX = "pooled_task_variable_decoding_"
FIGURE_KIND_BY_LABEL = {
    "Native temporal stability": "native",
    "Reference transfer": "transfer",
    "Native - transfer gap": "gap",
}


@dataclass(frozen=True)
class ConditionGeneralizationRunOption:
    """One completed pooled run available to the saved-results selector.

    Attributes
    ----------
    run_directory : pathlib.Path
        Absolute, session-contained completed-run directory.
    run_fingerprint : str
        Exact saved scientific/input/code identity.
    analysis_version : str
        Supported pooled analysis version recorded in ``config.json``.
    original_session_root : str
        Execution-time session path retained as provenance. It is not used as
        portable identity because cluster results may be copied elsewhere.
    display_label : str
        Concise selector label derived from immutable saved metadata.
    """

    run_directory: Path
    run_fingerprint: str
    analysis_version: str
    original_session_root: str
    display_label: str


def _path_traverses_symlink(root: Path, relative: Path) -> bool:
    """Return whether an existing component below ``root`` is a symlink.

    Parameters
    ----------
    root : pathlib.Path
        Resolved selected session directory.
    relative : pathlib.Path
        Relative candidate path below ``root``.

    Returns
    -------
    bool
        True when any existing path component is a symbolic link.
    """
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            return True
    return False


def resolve_condition_generalization_results_root(
    session_root: Path | str,
    locator: str = DEFAULT_CONDITION_GENERALIZATION_RESULTS_ROOT,
) -> Path:
    """Resolve one non-symlinked session-relative saved-results root.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Selected metadata session directory. Machine-level aliases above this
        directory are canonicalized before containment checks.
    locator : str, default="analysis_runs"
        Nonempty relative child directory. The webapp uses the default without
        exposing it as a scientific or display setting.

    Returns
    -------
    pathlib.Path
        Absolute normalized path. The directory may be absent, which represents
        an empty saved-results view.
    """
    root = Path(session_root).resolve()
    text = str(locator).strip()
    if not text:
        raise ValueError("Results root must be a nonempty session-relative path.")
    relative = Path(text)
    if relative.is_absolute():
        raise ValueError("Results root must be relative to the selected session.")
    if _path_traverses_symlink(root, relative):
        raise ValueError("Results root must not traverse a symbolic link.")
    candidate = (root / relative).resolve()
    try:
        contained = candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Results root must remain inside the selected session.") from error
    if contained == Path("."):
        raise ValueError("Results root must name a child directory of the session.")
    return candidate


def _read_json_object(path: Path, label: str) -> dict[str, object]:
    """Read one required nonsymlink JSON object.

    Parameters
    ----------
    path : pathlib.Path
        Expected regular JSON file inside one candidate run.
    label : str
        Human-readable artifact name used in validation errors.

    Returns
    -------
    dict[str, object]
        Parsed JSON object without interpreting scientific arrays.
    """
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular nonsymlink file.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable.") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return value


def _completed_run_option(run_directory: Path) -> ConditionGeneralizationRunOption:
    """Validate lightweight manifests for one direct-child run candidate.

    Parameters
    ----------
    run_directory : pathlib.Path
        Real direct child of the selected session's result root.

    Returns
    -------
    ConditionGeneralizationRunOption
        Selector metadata. The potentially multi-megabyte NPZ remains unopened.
    """
    state = _read_json_object(run_directory / "run_state.json", "run state")
    if (
        state.get("lifecycle") != "complete"
        or state.get("final_results_published") is not True
    ):
        raise ValueError("Run is not completely published.")
    execution = _read_json_object(run_directory / "execution.json", "execution record")
    config = _read_json_object(run_directory / "config.json", "scientific config")
    result_path = run_directory / "results.npz"
    if result_path.is_symlink() or not result_path.is_file():
        raise ValueError("Pooled result must be a regular nonsymlink file.")

    fingerprint = execution.get("run_fingerprint")
    original_session_root = execution.get("session_root")
    analysis_version = config.get("analysis_version")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Execution record has no run fingerprint.")
    if not isinstance(original_session_root, str) or not original_session_root:
        raise ValueError("Execution record has no original session root.")
    if analysis_version != ANALYSIS_VERSION:
        raise ValueError("Condition-generalization analysis version is unsupported.")
    return ConditionGeneralizationRunOption(
        run_directory=run_directory,
        run_fingerprint=fingerprint,
        analysis_version=analysis_version,
        original_session_root=original_session_root,
        display_label=(
            f"{run_directory.name} | {analysis_version} | {fingerprint[:12]}"
        ),
    )


def discover_condition_generalization_runs(
    session_root: Path | str,
    results_root_locator: str = DEFAULT_CONDITION_GENERALIZATION_RESULTS_ROOT,
) -> tuple[ConditionGeneralizationRunOption, ...]:
    """Discover completed direct-child pooled runs using only small manifests.

    Parameters
    ----------
    session_root : pathlib.Path or str
        Selected metadata session directory.
    results_root_locator : str, default="analysis_runs"
        Session-relative result parent. Only immediate children are inspected.

    Returns
    -------
    tuple[ConditionGeneralizationRunOption, ...]
        Valid supported runs ordered newest-name first. Original absolute
        session paths are retained only as provenance so copied cluster results
        remain portable.
    """
    results_root = resolve_condition_generalization_results_root(
        session_root, results_root_locator
    )
    if not results_root.exists():
        return ()
    if not results_root.is_dir():
        raise ValueError("Condition-generalization results root is not a directory.")

    options: list[ConditionGeneralizationRunOption] = []
    children = sorted(results_root.iterdir(), key=lambda path: path.name, reverse=True)
    for child in children:
        if (
            child.name.startswith(".")
            or not child.name.startswith(RUN_DIRECTORY_PREFIX)
            or child.is_symlink()
            or not child.is_dir()
        ):
            continue
        resolved_child = child.resolve()
        if resolved_child.parent != results_root:
            continue
        try:
            options.append(_completed_run_option(resolved_child))
        except (OSError, TypeError, ValueError):
            continue
    return tuple(options)


def load_selected_condition_generalization_run(
    option: ConditionGeneralizationRunOption,
) -> dict[str, object]:
    """Strictly load one selected saved run with its expected fingerprint.

    Parameters
    ----------
    option : ConditionGeneralizationRunOption
        Validated lightweight discovery record.

    Returns
    -------
    dict[str, object]
        Pooled loader mapping containing copied arrays, metadata, and identity.
    """
    return load_pooled_run(
        option.run_directory,
        expected_run_fingerprint=option.run_fingerprint,
    )


def _render_no_run_guidance(
    st_module: object,
    session_root: Path,
    results_root: Path,
) -> None:
    """Render saved-only guidance for a session with no completed pooled run.

    Parameters
    ----------
    st_module : object
        Streamlit-compatible rendering object.
    session_root, results_root : pathlib.Path
        Selected session and its automatic saved-result directory.

    Returns
    -------
    None
        Writes UI text only.
    """
    config_path = session_root / "pooled_task_decoding_config.json"
    prefix = "uv run --frozen python -m src.neural_analysis.task_decoding.pooled.run_session"
    st_module.info(
        "No completed condition-generalization runs were found in "
        f"{results_root}. Run the pooled analysis offline, then return to this view."
    )
    st_module.code(f"{prefix} dry-run --config {config_path}")
    st_module.code(f"{prefix} new --config {config_path}")


def _render_run_identity(
    st_module: object,
    option: ConditionGeneralizationRunOption,
) -> None:
    """Render immutable local identity and relocatable execution provenance.

    Parameters
    ----------
    st_module : object
        Streamlit-compatible rendering object.
    option : ConditionGeneralizationRunOption
        Selected completed pooled run.

    Returns
    -------
    None
        Writes UI captions only.
    """
    st_module.caption(f"Run directory: {option.run_directory}")
    st_module.caption(f"Run fingerprint: {option.run_fingerprint}")
    st_module.caption(f"Analysis version: {option.analysis_version}")
    st_module.caption(
        "Original execution session (provenance only): "
        f"{option.original_session_root}"
    )
    st_module.caption(f"Summary: {option.run_directory / 'summary.md'}")
    st_module.caption(f"Saved PNG directory: {option.run_directory / 'figures'}")


def render_condition_generalization_view(
    st_module: object,
    session: ResolvedSession,
) -> None:
    """Render one completed pooled run without opening raw neural sources.

    Parameters
    ----------
    st_module : object
        Imported Streamlit module or compatible test double. Every control is
        display-only and cannot construct targets, fit models, or write files.
    session : ResolvedSession
        Metadata-resolved session exposing an absolute ``session_root`` path.

    Returns
    -------
    None
        Renders one selected native, transfer, or gap figure from validated
        saved arrays. Time values remain seconds relative to choice and scores
        retain their saved dimensionless metric conventions.
    """
    st_module.subheader("Condition generalization results")
    try:
        results_root = resolve_condition_generalization_results_root(
            session.session_root
        )
        options = discover_condition_generalization_runs(session.session_root)
    except (OSError, ValueError) as error:
        st_module.error(f"Invalid condition-generalization results root: {error}")
        return
    if not options:
        _render_no_run_guidance(
            st_module,
            Path(session.session_root).resolve(),
            results_root,
        )
        return

    labels = [option.display_label for option in options]
    selected_label = st_module.selectbox("Completed run", options=labels)
    option = options[labels.index(selected_label)]
    try:
        saved_run = load_selected_condition_generalization_run(option)
        scientific_config = _read_json_object(
            option.run_directory / "config.json", "scientific config"
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        st_module.error(f"Selected saved run is no longer valid: {error}")
        return

    _render_run_identity(st_module, option)
    st_module.caption(
        "All controls below re-render validated saved arrays; this view does not recompute."
    )
    st_module.json(scientific_config, expanded=False)
    arrays = saved_run["arrays"]
    window_names = arrays["training_window_names"].tolist()
    target_names = arrays["target_names"].tolist()
    selected_window = st_module.selectbox("Training window", options=window_names)
    selected_target = st_module.selectbox("Target", options=target_names)
    result_label = st_module.selectbox(
        "Result type", options=list(FIGURE_KIND_BY_LABEL)
    )
    figure = plot_pooled_timecourses(
        saved_run,
        figure_kind=FIGURE_KIND_BY_LABEL[result_label],
        window_name=selected_window,
        target_name=selected_target,
    )
    try:
        st_module.pyplot(figure)
    finally:
        plt.close(figure)
