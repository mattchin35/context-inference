"""Persistence, cache identity, and table exports for trial-policy results.

Numerical arrays retain `(T,)` trial and `(T,K)` ordered-candidate axes. All
choice probabilities/residuals are dimensionless; likelihoods are natural logs.
"""

from collections.abc import Mapping
from dataclasses import fields
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from .trial_policy_models import (
    effective_trial_policy_choice_parameters, effective_trial_policy_parameters, prepare_trial_policy_history,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from .trial_policy_analysis import TrialPolicyResult
    from .trial_policy_models import PreparedTrialHistory, TrialPolicyCandidate
    from .gather_trial_features import TaskParams

ANALYSIS_VERSION = "trial_policy_hmm_v5"
ANALYSIS_SOURCE_FILES = (
    "trial_policy_models.py", "trial_policy_analysis.py", "trial_policy_results.py",
    "trial_features.py", "project_utils.py", "gather_trial_features.py",
    "exemplar_hmm_modeling.py", "main.py", "../behavior_modeling/counterfactual_doubt.py",
)


def build_trial_policy_metadata(history: "PreparedTrialHistory", candidates: "Sequence[TrialPolicyCandidate]",
                                feature_params: "TaskParams", *, parameter_source: str,
                                error_prob=0.10, stay_prob=0.90, smoothing=True,
                                probability_tie_atol=1e-6,
                                choice_temperatures: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Build JSON-compatible provenance and deterministic cache identity.

    history has T retained source-session rows; candidates is ordered K
    adapters; feature_params contains effective TaskParams. Probabilities and
    transition settings are dimensionless; returned metadata is plain data.
    choice_temperatures optionally overrides fixed graded-model temperatures
    by ID, in model-score units. No parameters are fitted or session-normalized.
    """
    from .trial_policy_analysis import trial_policy_transition_matrix

    error_prob = float(error_prob)
    if not np.isfinite(error_prob) or not 0 < error_prob < 0.5 or not 0 < 1 - error_prob < 1:
        raise ValueError("error probability and its complement must be representable strictly inside (0, 0.5)")
    stay_prob = float(stay_prob)
    if smoothing:
        trial_policy_transition_matrix(len(candidates), stay_prob)
    if not np.isfinite(probability_tie_atol) or probability_tie_atol < 0:
        raise ValueError("probability_tie_atol must be finite and nonnegative")
    effective = effective_trial_policy_parameters(candidates, feature_params)
    choice_parameters = effective_trial_policy_choice_parameters(
        candidates, temperatures=choice_temperatures, error_prob=error_prob)
    packages = {}
    for package in ("numpy", "pandas", "scipy", "jax", "dynamax"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    metadata = {
        "generator": "trial_policy_results.save_trial_policy_results",
        "analysis_version": ANALYSIS_VERSION,
        "session_id": history.session_id,
        "session_ids": [history.session_id],
        "parameters": {"opposite_action_error_prob": float(error_prob),
                       "stay_prob": float(stay_prob), "smoothing": bool(smoothing),
                       "probability_tie_atol": float(probability_tie_atol),
                       "candidate_ids": [c.model_id for c in candidates]},
        "candidate_parameters": effective,
        "candidate_choice_parameters": choice_parameters,
        "choice_temperature_source": "approved fixed defaults" if choice_temperatures is None
                                     else "explicit overrides plus approved fixed defaults",
        "choice_probability_definition": "logistic P(Left)=expit(score/temperature), no cap or lapse; discrete policies use fixed opposite-action error",
        "parameter_source": str(parameter_source),
        "input_digest": history.metadata["input_digest"],
        "source_flag_columns": list(history.metadata["source_flag_columns"]),
        "initialization": history.metadata["initialization"],
        "analyzed_trial_range": history.metadata["retained_trial_range"],
        "trial_outcome_counts": dict(history.metadata["trial_outcome_counts"]),
        "random_seed": None,
        "package_versions": packages,
        "analysis_source_digest": _analysis_source_digest(),
        "units": {"probabilities": "unitless", "log_likelihoods": "natural log probability",
                  "policy_scores": "dimensionless; Q-learning/FQL use reward-indicator scale",
                  "choice_temperatures": "same units as each model's signed policy score",
                  "omission_counts": "trials", "omission_lam": "inverse trials"},
        "axis_conventions": {"arrays": "retained trial rows by ordered candidate IDs",
                             "actions": "0=Right, 1=Left", "states": "zero-based candidate indices"},
        "eligibility_rules": "common valid non-manual Left/Right animal choice; absent flag column assumes animal control",
        "history_update_rule": "manual/no-choice/unknown-flag rows skip updates; missing rewards freeze outcome-based policies; choice-only persistence remembers valid choices",
        "history_update_valid_definition": "usable observed choice/outcome pair; choice-only update eligibility is comparison_valid",
        "candidate_history_update_rules": {candidate.model_id: candidate.history_update_rule for candidate in candidates},
        "tie_rule": "absolute probability tolerance; preserve every maximum; no arbitrary tie break",
        "score_indifference_rule": "exact_zero",
        "candidate_indifference_rules": {candidate.model_id: candidate.indifference_rule for candidate in candidates},
        "transition_probability_floor": 1e-6,
        "transition_roundoff_atol": 1e-12,
        "jax_probability_dtype": _jax_probability_dtype(),
    }
    signature_metadata = {key: value for key, value in metadata.items()
                          if key not in ("generator",)}
    metadata["signature"] = sha256(json.dumps(signature_metadata, sort_keys=True,
                                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return metadata


def find_trial_policy_cache(processed_data_path: str | Path, signature: str) -> Path | None:
    """Find newest complete artifact for a signature.

    processed_data_path is the directory containing session outputs; signature
    is a hexadecimal identity string. Returns an NPZ Path or None; array axes
    and units are not transformed.
    """
    root = Path(processed_data_path) / "trial_policy_hmm" / ANALYSIS_VERSION / str(signature)
    matches = [path for path in root.glob("*/results.npz") if (path.parent / "complete.json").is_file()]
    return max(matches, key=lambda path: path.parent.name) if matches else None


def save_trial_policy_results(result: "TrialPolicyResult", *, processed_data_path: str | Path,
                              metadata: dict[str, Any]) -> Path:
    """Persist named arrays beneath a new immutable timestamped run directory.

    result contains T retained rows by K candidates and metadata is plain data.
    Returns a new NPZ Path; it never reuses or overwrites an existing artifact.
    """
    # The workflow performs cache lookup before deciding whether to compute.
    # Calling this writer is an explicit new numerical run and must be immutable.
    root = Path(processed_data_path) / "trial_policy_hmm" / ANALYSIS_VERSION / metadata["signature"]
    directory = root / _timestamp()
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "results.npz"
    arrays = {}
    for field in fields(result):
        value = getattr(result, field.name)
        if value is None or field.name == "status":
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            array = array.astype(str)
        arrays[field.name] = array
    saved_metadata = dict(metadata, status=result.status, numerical_artifact=str(path), created_at=directory.name)
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays, meta=saved_metadata)
    (directory / "complete.json").write_text(json.dumps({"numerical_artifact": str(path)}), encoding="utf-8")
    return path


def load_trial_policy_results(path: str | Path) -> tuple["TrialPolicyResult", dict[str, Any]]:
    """Load a trusted named NPZ into TrialPolicyResult and plain metadata.

    Path points to one analysis artifact. Named arrays retain original axes;
    optional HMM fields absent in unsmoothed archives restore as None.
    """
    from .trial_policy_analysis import TrialPolicyResult
    with np.load(path, allow_pickle=True) as saved:
        metadata = saved["meta"].item()
        values = {field.name: saved[field.name].copy() if field.name in saved else None
                  for field in fields(TrialPolicyResult) if field.name != "status"}
    values["session_id"] = str(values["session_id"].item())
    values["status"] = metadata["status"]
    return TrialPolicyResult(**values), metadata


def attach_trial_policy_outputs(trial_df: pd.DataFrame, result: "TrialPolicyResult", *,
                                metadata: dict[str, Any]) -> pd.DataFrame:
    """Return current table with checked trial-policy columns, preserving index.

    trial_df is current `(T,C)` source data. Its current normalized history and
    flag schema must match metadata input digest; result arrays align by row.
    Nullable state/tie sentinels are exposed as pandas nullable values.
    """
    history = prepare_trial_policy_history(trial_df, session_id=result.session_id)
    if history.metadata["input_digest"] != metadata.get("input_digest"):
        raise ValueError("Current trial history identity digest/schema does not match saved result")
    if len(trial_df) != len(result.row_position) or not np.array_equal(result.row_position, np.arange(len(trial_df))):
        raise ValueError("Current table row positions do not match saved result identity")
    output = clear_trial_policy_outputs(trial_df)
    direct = ("comparison_valid", "history_update_valid", "comparison_exclusion_reason",
              "history_update_exclusion_reason", "independent_state", "independent_tie",
              "hmm_smoothed_state", "hmm_smoothed_tie")
    for name in direct:
        values = getattr(result, name, None)
        if values is None:
            continue
        column = f"trial_policy_{name}"
        series = pd.Series(values, index=output.index)
        if name.endswith("_state"):
            series = pd.array(np.where(np.asarray(values) < 0, pd.NA, values), dtype="Int64")
        elif name.endswith("_tie"):
            series = pd.array(np.where(np.asarray(values) < 0, pd.NA, values), dtype="boolean")
        output[column] = series
    for name in ("policy_scores", "prediction_valid", "model_indifferent", "greedy_choice", "p_right", "log_likelihoods",
                 "choice_residuals", "independent_probs", "independent_best_model_mask",
                 "hmm_smoothed_probs", "hmm_smoothed_best_model_mask"):
        values = getattr(result, name, None)
        if values is None:
            continue
        for index, model_id in enumerate(result.model_ids):
            column_values = values[:, index]
            if name == "greedy_choice":
                column_values = pd.array(np.where(column_values < 0, pd.NA, column_values), dtype="Int8")
            output[f"trial_policy_{name}_{model_id}"] = column_values
    for name in ("independent_state", "hmm_smoothed_state"):
        if name == "hmm_smoothed_state" and result.hmm_smoothed_state is None:
            continue
        states = getattr(result, name)
        labels = pd.array([str(result.model_ids[value]) if value >= 0 else pd.NA for value in states], dtype="string")
        output[f"trial_policy_{name.removesuffix('_state')}_model"] = labels
    return output


def clear_trial_policy_outputs(table: pd.DataFrame) -> pd.DataFrame:
    """Copy a `(T,C)` table while dropping all `trial_policy_` owned columns.

    The returned DataFrame keeps row order, index, and every non-owned column.
    No units or array shapes change.
    """
    columns = [name for name in table.columns if str(name).startswith("trial_policy_")]
    return table.drop(columns=columns).copy()


def trial_policy_warnings(history: "PreparedTrialHistory") -> list[str]:
    """Format separate diagnostics for invalid attempts and missing observations.

    history has aligned T-row arrays and original trial IDs; returns a short
    list of strings with at most five IDs per condition. Data remain unchanged.
    """
    messages = []
    no_choice = np.isnan(history.actions)
    invalid_switch = np.array(["invalid_switch" in reason.split(";")
                               for reason in history.comparison_exclusion_reason], dtype=bool)
    for label, mask, action in (
        ("invalid switch", invalid_switch,
         "excluded these attempts from scoring and candidate history updates"),
        ("unclassified no-choice attempt", no_choice & ~invalid_switch & (history.manual_reward_flags != 1),
         "excluded these attempts; the log does not establish a switch or manual-reward outcome"),
        ("missing reward", history.comparison_valid & ~np.isfinite(history.rewards),
         "scored available choice predictions; outcome-based policies skipped history updates; immediate persistence, when selected, retained the observed choice"),
        ("unknown manual-reward flag", np.isnan(history.manual_reward_flags) & ~no_choice,
         "excluded these rows from scoring and candidate history updates"),
    ):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        ids = [str(history.trial_id[index]) for index in indices[:5]]
        preview = ", ".join(ids) + (", ..." if len(indices) > 5 else "")
        messages.append(f"Session {history.session_id}: {label} on {len(indices)} trial(s) ({preview}); {action}.")
    return messages


def write_trial_policy_export(trial_df: pd.DataFrame, *, export_directory: str | Path,
                              numerical_path: str | Path | None, metadata: dict[str, Any],
                              current_csv_path: str | Path | None = None) -> Path:
    """Write current augmented CSV and run records referencing immutable arrays.

    trial_df is the current attached `(T,C)` snapshot. Numerical NPZ is never
    copied or modified; None links the joint source manifest without claiming
    one joint numerical file. Optional current CSV is archived before replacement.
    Returns the snapshot Path named trial_policy_export.csv, distinct from the
    conventional session input. No table axes or physical units are transformed.
    """
    directory = Path(export_directory)
    directory.mkdir(parents=True, exist_ok=False)
    if current_csv_path is not None:
        _archive_current_csv(Path(current_csv_path))
        trial_df.to_csv(current_csv_path, index=False, na_rep="")
    exported = directory / "trial_policy_export.csv"
    trial_df.to_csv(exported, index=False, na_rep="")
    record = dict(metadata, export_artifact=str(exported))
    if numerical_path is None:
        record.pop("numerical_artifact", None)
    else:
        record["numerical_artifact"] = str(numerical_path)
    _write_record(directory, record)
    return exported


def write_trial_policy_run_record(directory: str | Path, metadata: dict[str, Any]) -> None:
    """Write plain run metadata/log/summary, including early errors.

    directory is a run-record destination and metadata is a JSON-compatible
    dictionary that need not contain signature/numerical-artifact keys.
    Returns None and does not create numerical arrays or a completion marker.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _write_record(directory, metadata)


def _analysis_source_digest() -> str:
    """Hash every required computation/results source and the main source."""
    digest = sha256()
    behavior_root = Path(__file__).resolve().parent
    source_root = behavior_root.parent
    for name in ANALYSIS_SOURCE_FILES:
        path = source_root / name if name == "main.py" else behavior_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required trial-policy source is missing: {path}")
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _jax_probability_dtype() -> str:
    """Return the actual default JAX array dtype used for a scalar probability."""
    import jax.numpy as jnp
    return str(jnp.asarray([0.5]).dtype)


def _timestamp() -> str:
    """Return a UTC filesystem-safe unique timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f_UTC")


def _archive_current_csv(path: Path) -> None:
    """Preserve CSV Path bytes once with an .archived.csv suffix; return None.

    Existing legacy-named archives also count as preserved and remain unchanged.
    The CSV contents, column axes, and units are not transformed.
    """
    if not path.exists():
        return
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    directory = path.parent / "trial_policy_hmm" / "archives" / digest
    target = directory / f"{path.stem}.archived.csv"
    if target.exists() or (directory / path.name).exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as output:
        output.write(content)


def _write_record(directory: Path, metadata: dict) -> None:
    """Write plain JSON, log and concise summary plus available source snapshots."""
    serialized = json.dumps(metadata, indent=2, default=_json_default)
    (directory / "metadata.json").write_text(serialized, encoding="utf-8")
    (directory / "run.log").write_text(serialized, encoding="utf-8")
    sessions = metadata.get("session_ids", [metadata.get("session_id", "unknown")])
    (directory / "summary.md").write_text(
        "# Trial-policy comparison\n\n"
        "Goal: compare fixed causal choice policies with independent likelihoods and optional retrospective HMM smoothing.\n\n"
        f"Included sessions: {json.dumps(sessions)}. Status: {metadata.get('status', 'recorded')}.\n\n"
        f"Trial outcomes: {json.dumps(metadata.get('trial_outcome_counts', {}))}.\n\n"
        "Main and batch orchestration reside in main.py. Numerical arrays remain in their referenced session artifact.\n",
        encoding="utf-8")
    main = Path(__file__).resolve().parents[1] / "main.py"
    if main.is_file():
        shutil.copy2(main, directory / "main.py")
        (directory / "batch.py").write_text("# Batch source snapshot: main.py contains both orchestration paths.\n" +
                                               main.read_text(encoding="utf-8"), encoding="utf-8")


def _json_default(value):
    """Convert NumPy scalar metadata to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize metadata value {type(value).__name__}")
