"""Session-local named-array persistence tied to an exact source PCA archive."""

from dataclasses import dataclass, fields, replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from scipy.special import expit

from src.neural_analysis.population.session_timeline_io import load_timeline_result
from src.neural_analysis.task_decoding.config import CONDITION_IDENTIFIERS

from .config import ANALYSIS_VERSION, FOLD_COUNT, LOGISTIC_C, RIDGE_ALPHA, SCHEMA_VERSION, TARGET_NAMES
from .modeling import RegionPredictions
from .preparation import PredictionInputs


SHARED_ARRAYS = (
    "target_values", "target_valid", "baseline_valid", "correct_rewarded",
    "fold_ids", "block_values_json", "condition_masks", "training_bin_mask", "bin_edges_s",
)


@dataclass(frozen=True)
class SessionPredictions:
    """Loaded inputs, regional model records and dictionary provenance.

    Array axes/units are those of PredictionInputs and RegionPredictions. The
    referenced timeline supplies original scores/events; no raw source is read.
    """

    inputs: PredictionInputs
    regions: dict[str, RegionPredictions]
    meta: dict


@dataclass(frozen=True)
class PredictionRunOption:
    """Small discovery record: absolute result path, identity, session and label."""

    result_path: Path
    identity: str
    session_id: str
    display_label: str


def build_prediction_metadata(
    inputs: PredictionInputs, session_root: Path | str, source_pca: Path | str,
    target_sources: dict[str, Path], effective_config: dict,
) -> dict:
    """Return portable input/code/config identity plus units, model and runtime provenance.

    Paths name existing session-contained files. target_sources names the small
    augmented-table/behavior-feature inputs; effective_config records selected
    probes and integer millisecond width. Scores/labels keep their native units.
    Hash source bytes, not absolute paths or unrelated metadata/display files.
    """
    root = Path(session_root).resolve()
    repository = Path(__file__).resolve().parents[4]
    source_files = (
        "task_decoding/session_prediction/config.py",
        "task_decoding/session_prediction/preparation.py",
        "task_decoding/session_prediction/modeling.py",
        "task_decoding/session_prediction/results.py",
        "task_decoding/session_prediction/run_session.py",
        "task_decoding/targets.py", "task_decoding/config.py", "task_decoding/conditions.py",
        "task_decoding/pooled/decoding.py", "interregional/preparation.py", "spike_behavior/trials.py",
    )
    code = {
        name: _file_hash(repository / "src/neural_analysis" / name)
        for name in source_files
    }
    code["behavior_analysis/project_utils.py"] = _file_hash(repository / "src/behavior_analysis/project_utils.py")
    config = dict(effective_config)
    config.update({"ridge_alpha": RIDGE_ALPHA, "logistic_C": LOGISTIC_C,
                   "training_window_s": [-1.0, 1.0], "prediction_window_s": [-2.0, 2.0],
                   "fold_count": FOLD_COUNT, "target_names": list(TARGET_NAMES)})
    pca = file_identity(source_pca, root)
    pca["configuration_id"] = inputs.timeline.meta["configuration_id"]
    identity_payload = {
        "analysis_version": ANALYSIS_VERSION, "schema_version": SCHEMA_VERSION,
        "session_id": inputs.timeline.session_id, "source_pca": pca,
        "target_sources": {name: file_identity(path, root) for name, path in target_sources.items()},
        "effective_config": config, "scientific_sources": code,
        "library_versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")},
    }
    encoded = json.dumps(identity_payload, sort_keys=True, separators=(",", ":"))
    return {
        **identity_payload, "identity": hashlib.sha256(encoded.encode()).hexdigest()[:20],
        "generator": "src.neural_analysis.task_decoding.session_prediction.run_session",
        "time_origin_s": inputs.timeline.time_origin_s,
        "target_metadata": inputs.target_metadata,
        "target_availability": inputs.target_availability,
        "pca_scope": "fixed session-wide PCA and unit scaling include held-out neural data",
        "row_weighting": "1/training_bin_count; each training trial has total weight one",
        "estimators": {"ridge": {"solver": "svd", "fit_intercept": True, "alpha": RIDGE_ALPHA},
                       "logistic": {"solver": "lbfgs", "l1_ratio": 0.0, "fit_intercept": True,
                                    "C": LOGISTIC_C, "tol": 1e-4, "max_iter": 5000, "class_weight": None}},
        "random_seed": None, "randomness_used": False,
        "contribution_strength": "abs(beta)*population_SD(pooled training PC); equal mean across fitted folds",
        "units": {"time": "seconds", "absolute_time": "UTC Unix seconds",
                  "scores": "unwhitened standardized-unit scores",
                  "categorical_prediction": "P(class 1 = left)",
                  "categorical_contribution": "log-odds", "numerical": "native source target units"},
        "axes": {"predictions": ["regional_window", "choice_bin", "target"],
                 "coefficients": ["target", "held_out_fold", "original_pc"],
                 "condition_masks": ["choice", "condition"]},
        "regions": {role: {"status": "available", "probe_id": region.probe_id,
                           "feature_count": region.scores.shape[2]}
                    for role, region in inputs.timeline.regions.items()}
                   | {role: {"status": "unavailable", "reason": reason}
                      for role, reason in inputs.timeline.unavailable.items()},
    }


def file_identity(path: Path | str, session_root: Path | str) -> dict:
    """Return relative path, byte size and SHA-256 for one existing contained file."""
    source = Path(path).resolve(strict=True)
    root = Path(session_root).resolve()
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("Prediction inputs must be files inside the metadata session")
    return {"path": source.relative_to(root).as_posix(), "size": source.stat().st_size,
            "sha256": _file_hash(source)}


def save_prediction_result(
    path: Path | str, inputs: PredictionInputs,
    regions: dict[str, RegionPredictions], meta: dict,
) -> Path:
    """Validate and publish a NEW named-array NPZ with dictionary-valued meta.

    inputs/regions retain their documented N/W/B/T/F/K axes and units. Write
    small config.json before atomic NPZ publication; a missing NPZ is incomplete.
    Caller writes summary/log/script snapshots before publication. Never overwrite.
    """
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    validate_prediction_result(inputs, regions)
    arrays = {name: getattr(inputs, name) for name in SHARED_ARRAYS}
    arrays.update({"choices": inputs.timeline.choices.to_records(index=False),
                   "target_names": np.asarray(inputs.target_names),
                   "target_families": np.asarray(inputs.target_families),
                   "condition_names": np.asarray(inputs.condition_names),
                   "meta": np.asarray(meta, dtype=object)})
    for role, model in regions.items():
        arrays[role.lower() + "_choice_positions"] = inputs.timeline.regions[role].choice_positions
        arrays.update({role.lower() + "_" + field.name: getattr(model, field.name)
                       for field in fields(RegionPredictions)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    (destination.parent / "config.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_prediction_result(path: Path | str, *, session_root: Path | str | None = None) -> SessionPredictions:
    """Load trusted project results and their exact PCA, without raw source access.

    path is a completed session/analysis_runs/run/results.npz; an explicit
    session_root supports relocated copies. Verify PCA bytes before loading it.
    Return original input/model axes and units; validate saved scientific arrays.
    """
    path = Path(path)
    root = Path(session_root).resolve() if session_root is not None else path.resolve().parents[2]
    with np.load(path, allow_pickle=True) as archive:
        meta = archive["meta"].item()
        if meta.get("analysis_version") != ANALYSIS_VERSION or meta.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported session prediction result version")
        source = meta["source_pca"]
        source_path = root / source["path"]
        identity = file_identity(source_path, root)
        if any(identity[name] != source[name] for name in ("path", "size", "sha256")):
            raise ValueError("Referenced PCA archive bytes differ from the saved prediction source")
        timeline = load_timeline_result(source_path)
        if (timeline.session_id != meta["session_id"] or timeline.time_origin_s != meta["time_origin_s"]
                or timeline.meta["configuration_id"] != source["configuration_id"]):
            raise ValueError("Referenced timeline identity does not match predictions")
        choices = pd.DataFrame.from_records(archive["choices"])
        identity_columns = ["row_position", "trial_id", "choice_time", "action", "elapsed_s"]
        if not choices[identity_columns].equals(timeline.choices[identity_columns]):
            raise ValueError("Saved prediction choices do not match source PCA choices")
        # Prediction choices may add optional rewards to an older PCA table.
        timeline = replace(timeline, choices=choices)
        inputs = PredictionInputs(
            timeline=timeline, target_names=tuple(archive["target_names"]),
            target_families=tuple(archive["target_families"]),
            condition_names=tuple(archive["condition_names"]),
            target_metadata=meta["target_metadata"], target_availability=meta["target_availability"],
            **{name: archive[name] for name in SHARED_ARRAYS},
        )
        regions = {}
        for role, regional in timeline.regions.items():
            prefix = role.lower() + "_"
            if not np.array_equal(archive[prefix + "choice_positions"], regional.choice_positions):
                raise ValueError("Regional prediction choice map differs from PCA source")
            regions[role] = RegionPredictions(**{field.name: archive[prefix + field.name]
                                                 for field in fields(RegionPredictions)})
    validate_prediction_result(inputs, regions)
    return SessionPredictions(inputs, regions, meta)


def discover_prediction_runs(session_root: Path | str) -> tuple[PredictionRunOption, ...]:
    """Read only small JSON metadata for published direct-child prediction runs.

    Return newest-first options; missing/invalid/partial directories are skipped.
    Raw sources and NPZ arrays are never opened by discovery.
    """
    options = []
    for path in sorted((Path(session_root) / "analysis_runs").glob("condition_session_prediction_*/config.json"), reverse=True):
        result = path.parent / "results.npz"
        if not result.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="ascii"))
            if meta["analysis_version"] != ANALYSIS_VERSION or meta["schema_version"] != SCHEMA_VERSION:
                continue
            label = f"{path.parent.name} ({meta['effective_config']['bin_width_ms']} ms)"
            options.append(PredictionRunOption(result, meta["identity"], meta["session_id"], label))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return tuple(options)


def find_saved_prediction(session_root: Path | str, identity: str) -> Path | None:
    """Return newest validated matching prediction path, or None; never refit or write."""
    for option in discover_prediction_runs(session_root):
        if option.identity != identity:
            continue
        try:
            load_prediction_result(option.result_path, session_root=session_root)
            return option.result_path
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            continue
    return None


def validate_prediction_result(inputs: PredictionInputs, regions: dict[str, RegionPredictions]) -> None:
    """Validate saved N/W/B/T/F/K shapes, units-related conventions and block exclusion.

    Inputs and region records follow their documented contracts. Return None or
    raise ValueError before publication/display. No array is changed or refitted.
    """
    n, t, b = len(inputs.timeline.choices), len(TARGET_NAMES), len(inputs.bin_edges_s) - 1
    if inputs.target_names != TARGET_NAMES or inputs.condition_names != CONDITION_IDENTIFIERS:
        raise ValueError("Noncanonical target or condition order")
    _shape(inputs.target_values, (n, t), "target values")
    for array, shape in ((inputs.target_valid, (n, t)), (inputs.baseline_valid, (n,)),
                         (inputs.correct_rewarded, (n,)), (inputs.condition_masks, (n, 6)),
                         (inputs.training_bin_mask, (b,))):
        _shape(array, shape, "shared mask", dtype_kind="b")
    if (not np.isfinite(inputs.target_values[inputs.target_valid]).all()
            or not np.isnan(inputs.target_values[~inputs.target_valid]).all()
            or not np.array_equal(inputs.correct_rewarded, inputs.condition_masks[:, 1])):
        raise ValueError("Target validity/condition mask conventions disagree")
    _shape(inputs.fold_ids, (n,), "fold IDs", dtype_kind="iu")
    _shape(inputs.block_values_json, (n,), "block identities")
    if not np.isin(inputs.fold_ids, [-1, 0, 1, 2, 3, 4]).all():
        raise ValueError("Fold IDs must be -1..4")
    for block in np.unique(inputs.block_values_json):
        folds = inputs.fold_ids[inputs.block_values_json == block]
        if len(np.unique(folds)) != 1 or (block == "null" and np.any(folds != -1)):
            raise ValueError("A block is split between folds or missing blocks have predictions")
    if set(regions) != set(inputs.timeline.regions):
        raise ValueError("Prediction regions must match available source PCA regions")
    for role, model in regions.items():
        _validate_region(inputs, role, model)


# Validation helpers keep file loading separate from scientific array checks.
def _validate_region(inputs: PredictionInputs, role: str, model: RegionPredictions) -> None:
    """Check one W,B,T result against N choices, K original PCs and five model folds."""
    source = inputs.timeline.regions[role]
    w, b, k = source.scores.shape
    n, t = inputs.target_values.shape
    expected = {
        "pc_ids": (k,), "predictions": (w, b, t), "prediction_reason": (w, t),
        "coefficients": (t, 5, k), "intercepts": (t, 5), "fit_status": (t, 5),
        "fit_reason": (t, 5), "training_choice_mask": (t, 5, n),
        "training_trial_count": (t, 5), "training_row_count": (t, 5),
        "training_pc_sd": (t, 5, k), "contribution_strength": (t, 5, k),
        "mean_contribution_strength": (t, k), "contribution_pc_order": (t, k),
        "contributing_fold_count": (t,), "contribution_color_limit": (t,),
    }
    for name, shape in expected.items():
        _shape(getattr(model, name), shape, name)
    if not np.array_equal(model.pc_ids, np.arange(1, k + 1)):
        raise ValueError("Model PCs must retain full original canonical rank order")
    if (model.training_choice_mask.dtype != bool or not np.isin(model.fit_status, ["ok", "unavailable"]).all()
            or not np.isfinite(model.contribution_color_limit).all() or np.any(model.contribution_color_limit <= 0)):
        raise ValueError("Invalid model status, mask or contribution limit")
    fitted = model.fit_status == "ok"
    for array in (model.coefficients, model.intercepts, model.training_pc_sd, model.contribution_strength):
        if not np.isfinite(array[fitted]).all() or not np.isnan(array[~fitted]).all():
            raise ValueError("Model parameter availability conventions disagree")
    if np.any((model.fit_reason == "") != fitted):
        raise ValueError("Fit reasons do not match model availability")
    coverage = np.zeros(n, dtype=bool)
    coverage[source.choice_positions] = True
    for target, family in enumerate(inputs.target_families):
        for fold in range(5):
            mask = model.training_choice_mask[target, fold]
            expected_mask = (coverage & inputs.baseline_valid & inputs.correct_rewarded & inputs.target_valid[:, target]
                             & (inputs.fold_ids >= 0) & (inputs.fold_ids != fold))
            if not np.array_equal(mask, expected_mask):
                raise ValueError("Training mask disagrees with coverage, eligibility or held-out fold")
            if (model.training_trial_count[target, fold] != mask.sum()
                    or model.training_row_count[target, fold] != mask.sum() * inputs.training_bin_mask.sum()):
                raise ValueError("Training counts disagree with saved masks")
        available = model.prediction_reason[:, target] == ""
        predicted = model.predictions[:, :, target]
        if not np.isfinite(predicted[available]).all() or not np.isnan(predicted[~available]).all():
            raise ValueError("Prediction availability and finite/NaN values disagree")
        if family == "categorical" and np.any((predicted[available] < 0) | (predicted[available] > 1)):
            raise ValueError("Binary probabilities must be within [0,1]")
        for fold in range(5):
            selected = inputs.fold_ids[source.choice_positions] == fold
            if not fitted[target, fold] and available[selected].any():
                raise ValueError("An unavailable model has destination predictions")
            selected &= available
            if selected.any():
                linear = source.scores[selected] @ model.coefficients[target, fold] + model.intercepts[target, fold]
                expected_values = expit(linear) if family == "categorical" else linear
                if not np.allclose(predicted[selected], expected_values, rtol=1e-10, atol=1e-10):
                    raise ValueError("Saved predictions do not reconstruct from their held-out models")
        if available[inputs.fold_ids[source.choice_positions] < 0].any():
            raise ValueError("Missing-block windows must not have predictions")
        count = fitted[target].sum()
        if model.contributing_fold_count[target] != count:
            raise ValueError("Contribution fold count disagrees with fitted models")
        mean = model.contribution_strength[target, fitted[target]].mean(axis=0) if count else np.full(k, np.nan)
        if not np.allclose(model.mean_contribution_strength[target], mean, equal_nan=True):
            raise ValueError("Contribution means disagree with fitted folds")
        order = np.lexsort((model.pc_ids, -mean)) + 1 if count else model.pc_ids
        if not np.array_equal(model.contribution_pc_order[target], order):
            raise ValueError("Contribution order disagrees with strengths/original rank ties")


def _shape(array: np.ndarray, shape: tuple[int, ...], name: str, dtype_kind: str | None = None) -> None:
    """Check an array against its dimensionless axis lengths and optional NumPy kind."""
    if array.shape != shape or (dtype_kind is not None and array.dtype.kind not in dtype_kind):
        raise ValueError(f"Invalid {name} shape/dtype: expected {shape}")


def _file_hash(path: Path) -> str:
    """Return SHA-256 hex for an existing file, reading bounded 1-MiB byte chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
