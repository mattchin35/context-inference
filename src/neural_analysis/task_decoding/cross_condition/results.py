"""Portable, self-contained transfer bundles and metadata-only source identities."""

from dataclasses import asdict, dataclass, fields
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import subprocess
import tempfile
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from src.neural_analysis.population.session_timeline import choice_bin_edges, trial_identities_match
from src.neural_analysis.task_decoding.config import CONDITION_IDENTIFIERS, SUPPORTED_BIN_WIDTH_MS, TARGET_DEFINITION_BY_IDENTIFIER
from src.neural_analysis.task_decoding.session_prediction.config import LOGISTIC_C, training_window_bounds
from src.neural_analysis.task_decoding.session_prediction.preparation import training_bin_mask
from src.neural_analysis.task_decoding.session_prediction.results import file_identity
from src.neural_analysis.task_decoding.targets import _CATEGORICAL_CLASS_LABELS

from .config import ANALYSIS_VERSION, SCHEMA_VERSION, TARGET_NAMES, TRANSFERS, WEIGHTING_NAMES, WINDOW_NAMES
from .modeling import RegionTransferResult, _regional_cells, training_trial_weights
from .preparation import TransferInputs, _named_cells, _transfer_cells


SHARED_ARRAYS = (
    "target_values", "target_valid", "baseline_valid", "conflict_mask", "condition_masks",
    "source_cells", "test_cells", "window_bin_masks", "bin_edges_s",
)
ROLE_NAMES = ("training", "complementary_test")
AXES = {
    "target_values": ["choice", "target"], "condition_masks": ["choice", "condition"],
    "source_cells": ["choice", "split", "actual_class"], "test_cells": ["choice", "split", "actual_class"],
    "window_bin_masks": ["supervised_window", "choice_bin"],
    "coefficients": ["supervised_window", "weighting", "split", "original_pc"],
    "intercepts": ["supervised_window", "weighting", "split"],
    "trial_probabilities": ["supervised_window", "weighting", "split", "regional_choice"],
    "source_mask": ["split", "regional_choice"], "test_mask": ["split", "regional_choice"],
    "source_trial_weights": ["weighting", "split", "regional_choice"],
    "eligible_cell_counts": ["split", "role", "actual_class"],
}
UNITS = {
    "time": "seconds", "absolute_time": "UTC Unix seconds",
    "scores": "unwhitened standardized-unit scores", "coefficients": "log-odds per PC-score unit",
    "intercepts": "log-odds", "trial_probabilities": "P(class 1)",
    "target_values": "dimensionless binary labels", "source_trial_weights": "dimensionless",
}
SCIENTIFIC_FILES = (
    "src/neural_analysis/task_decoding/cross_condition/config.py",
    "src/neural_analysis/task_decoding/cross_condition/preparation.py",
    "src/neural_analysis/task_decoding/cross_condition/modeling.py",
    "src/neural_analysis/task_decoding/session_prediction/config.py",
    "src/neural_analysis/task_decoding/session_prediction/preparation.py",
    "src/neural_analysis/task_decoding/session_prediction/modeling.py",
    "src/neural_analysis/task_decoding/targets.py", "src/neural_analysis/task_decoding/config.py",
    "src/neural_analysis/task_decoding/conditions.py", "src/neural_analysis/task_decoding/pooled/decoding.py",
    "src/neural_analysis/spike_behavior/trials.py", "src/neural_analysis/population/session_timeline.py",
    "src/behavior_analysis/project_utils.py",
)


@dataclass(frozen=True)
class CrossConditionResult:
    """Saved behavioral inputs, independent regional models, availability reasons and provenance.

    inputs/regions retain TransferInputs/RegionTransferResult N/W/B/K/H/G/S
    axes and units. unavailable_regions maps regional names to reasons without
    fabricated neural axes. meta contains portable source references, fitting
    identity and execution provenance. No external files are needed to inspect.
    """

    inputs: TransferInputs
    regions: dict[str, RegionTransferResult]
    unavailable_regions: dict[str, str]
    meta: dict


@dataclass(frozen=True)
class TransferRunOption:
    """Small discovery record: absolute result Path, identity/session/label strings and integer bin-width ms."""

    result_path: Path
    identity: str
    session_id: str
    display_label: str
    bin_width_ms: int


def build_transfer_metadata(
    inputs: TransferInputs, session_root: Path | str, source_pcas: dict[str, Path],
    target_sources: dict[str, Path], effective_config: dict, *, session_id: str,
) -> dict:
    """Return portable fitting identity and provenance using only small metadata and file hashes.

    inputs has documented N/T/S/H/B behavioral axes. session_root contains all
    selected source_pcas(region->Path), required target_sources augmented_trials/
    base_config/feature_parameters, and neural_session.json. effective_config
    supplies integer bin_width_ms and PFC/HPC probe roles; fixed science is added.
    session_id is the metadata's stable identifier. Each distinct source file is
    hashed once; PCA reads access only dictionary meta/configuration_id, not
    scores. Scientific code/library/source/config identity excludes runtime,
    Git receipt, raw metadata edits, timestamps and presentation/scoring code.
    Return a JSON-safe dictionary; caller fills runtime/RSS after computation.
    No input array changes or output writes occur.
    """
    config = _scientific_config(inputs, effective_config)
    _validate_inputs(inputs, config["bin_width_ms"])
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("session_id must be a nonempty metadata identifier")
    if not {"augmented_trials", "base_config", "feature_parameters"}.issubset(target_sources):
        raise ValueError("target_sources require augmented_trials, base_config and feature_parameters")
    if not set(source_pcas).issubset(config["probe_roles"]):
        raise ValueError("PCA source regions must have configured probe roles")
    root = Path(session_root).resolve()
    metadata_path = root / "neural_session.json"
    paths = {Path(path).resolve() for path in (*source_pcas.values(), *target_sources.values(), metadata_path)}
    identities = {path: file_identity(path, root) for path in paths}
    pca_identities = {}
    for path in sorted({Path(path).resolve() for path in source_pcas.values()}):
        with np.load(path, allow_pickle=True) as archive:
            pca_meta = archive["meta"].item()
        configuration_id = pca_meta.get("configuration_id") if isinstance(pca_meta, dict) else None
        if not isinstance(configuration_id, str) or not configuration_id:
            raise ValueError("Selected PCA source must have a configuration_id")
        info = {**identities[path], "configuration_id": configuration_id}
        pca_identities[info["path"]] = info
    repository = Path(__file__).resolve().parents[4]
    payload = {
        "analysis_version": ANALYSIS_VERSION, "schema_version": SCHEMA_VERSION, "session_id": session_id,
        "source_pcas": pca_identities,
        "region_sources": {role: identities[Path(path).resolve()]["path"] for role, path in source_pcas.items()},
        "target_sources": {name: identities[Path(path).resolve()] for name, path in target_sources.items()},
        "effective_config": config,
        "scientific_sources": {name: file_identity(repository / name, repository) for name in SCIENTIFIC_FILES},
        "library_versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas")},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        **payload, "identity": hashlib.sha256(encoded.encode()).hexdigest()[:20],
        "generator": "src.neural_analysis.task_decoding.cross_condition.run_session",
        "session_metadata": identities[metadata_path.resolve()], "target_metadata": inputs.target_metadata,
        "randomness_used": False, "random_seed": None, "runtime_s": None, "peak_process_rss_bytes": None,
        "axes": {name: list(axes) for name, axes in AXES.items()}, "units": UNITS.copy(), **_git_provenance(repository),
        "pca_scope": "fixed session-wide PCA and scaling include complementary-test neural activity",
    }


def save_transfer_result(path: Path | str, result: CrossConditionResult) -> Path:
    """Validate and atomically publish a new named-array NPZ and matching config.json.

    result retains its documented N/W/B/K/H/G/S axes and physical units. Only
    prepared behavior, parameters, trial means and provenance are stored, with
    dictionary-valued meta and raw-type-preserving structured table arrays.
    Caller writes other run documents before publication. Never overwrite an
    existing result or manifest; interrupted writes leave no final NPZ or temporary arrays.
    Return the destination Path without changing the supplied result.
    """
    destination = Path(path)
    manifest_path = destination.parent / "config.json"
    if destination.exists():
        raise FileExistsError(destination)
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    if destination.suffix != ".npz":
        raise ValueError("Transfer results must be an .npz file")
    _validate_result(result)
    meta = {**result.meta, "unavailable_regions": dict(result.unavailable_regions)}
    arrays = {name: getattr(result.inputs, name) for name in SHARED_ARRAYS}
    arrays.update(_registry_arrays())
    arrays.update({"choices": result.inputs.choices.to_records(index=False),
                   "conflicts": result.inputs.conflicts.to_records(index=False),
                   "dark_state_count": np.asarray(result.inputs.dark_state_count),
                   "region_names": np.asarray(tuple(result.regions), dtype=str), "meta": np.asarray(meta, dtype=object)})
    for role, model in result.regions.items():
        arrays.update({role.lower() + "_" + field.name: getattr(model, field.name) for field in fields(model)})
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="ascii")
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


def load_transfer_result(path: Path | str) -> CrossConditionResult:
    """Load/validate a completed trusted-project bundle using only its NPZ and sibling manifest.

    path is a final result .npz Path/string. Return original behavioral/model
    axes, units, raw conflict/ID types and provenance. Never open/hash original
    data, base config, PCA or current code, fit models, or rewrite the run.
    Reject unsupported schema, missing arrays and normal contradictory saved
    contracts with ValueError explaining the selected product's failure.
    """
    path = Path(path)
    try:
        manifest = json.loads((path.parent / "config.json").read_text(encoding="ascii"))
        _validate_metadata(manifest)
        with np.load(path, allow_pickle=True) as archive:
            meta = archive["meta"].item()
            _validate_metadata(meta)
            if manifest["identity"] != meta["identity"]:
                raise ValueError("Manifest/config identity differs from the transfer archive identity")
            for name, expected in _registry_arrays().items():
                if not np.array_equal(archive[name], expected):
                    raise ValueError(f"Saved {name} contradicts the fixed transfer registry")
            dark_count = archive["dark_state_count"]
            _shape(dark_count, (), "dark_state_count", "iu")
            inputs = TransferInputs(
                choices=pd.DataFrame.from_records(archive["choices"]), target_names=TARGET_NAMES,
                target_metadata=meta["target_metadata"], condition_names=CONDITION_IDENTIFIERS,
                window_names=WINDOW_NAMES, dark_state_count=int(dark_count.item()),
                conflicts=pd.DataFrame.from_records(archive["conflicts"]), transfers=TRANSFERS,
                **{name: archive[name] for name in SHARED_ARRAYS},
            )
            names = tuple(archive["region_names"].tolist())
            if len(set(names)) != len(names):
                raise ValueError("Duplicate saved regional names")
            regions = {role: RegionTransferResult(**{
                field.name: archive[role.lower() + "_" + field.name] for field in fields(RegionTransferResult)
            }) for role in names}
        result = CrossConditionResult(inputs, regions, meta["unavailable_regions"], meta)
        _validate_result(result)
        return result
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError,
            EOFError, BadZipFile, pickle.UnpicklingError) as error:
        raise ValueError(f"Invalid cross-condition result {path}: {error}") from error


def discover_transfer_runs(session_root: Path | str) -> tuple[TransferRunOption, ...]:
    """Return newest-first run options from small JSON manifests and final-NPZ existence only.

    session_root is a metadata-session Path/string. Skip incomplete/unsupported
    runs; do not read any NPZ arrays or original sources. Paths are absolute,
    bin widths integer milliseconds, and labels identify the saved settings.
    """
    options = []
    root = Path(session_root).resolve()
    for manifest in sorted((root / "analysis_runs").glob("cross_condition_generalization_*/config.json"), reverse=True):
        path = manifest.parent / "results.npz"
        if not path.is_file():
            continue
        try:
            meta = json.loads(manifest.read_text(encoding="ascii"))
            _validate_metadata(meta)
            width = meta["effective_config"]["bin_width_ms"]
            options.append(TransferRunOption(path, meta["identity"], meta["session_id"],
                                             f"{width} ms - {manifest.parent.name}", width))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return tuple(options)


def find_saved_transfer(session_root: Path | str, identity: str) -> Path | None:
    """Return newest validated exact-identity result Path, or None; never load original sources or fit."""
    for option in discover_transfer_runs(session_root):
        if option.identity != identity:
            continue
        try:
            load_transfer_result(option.result_path)
            return option.result_path
        except ValueError:
            continue
    return None


# Metadata and array checks use only saved contracts; no reconstruction from PCA.
def _scientific_config(inputs: TransferInputs, effective_config: dict) -> dict:
    """Return canonical JSON-safe fitting configuration from prepared axes and small controls.

    inputs supplies fixed target/split/window identities and second edges;
    effective_config supplies bin-width ms and regional probe-ID strings.
    Raise ValueError for contradictions; no input dictionary is changed.
    """
    if inputs.target_names != TARGET_NAMES or inputs.transfers != TRANSFERS or inputs.window_names != WINDOW_NAMES:
        raise ValueError("Prepared targets, transfers and windows must match this analysis version")
    if "prediction_window_s" in effective_config:
        raise ValueError("Decoder intervals are matched supervised windows; use pca_source_window_s for PCA coverage")
    width = effective_config.get("bin_width_ms")
    if type(width) is not int or width not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError("bin_width_ms must be a supported exact integer")
    roles = effective_config.get("probe_roles")
    if not isinstance(roles, dict) or set(roles) != {"PFC", "HPC"} or not all(isinstance(value, str) and value for value in roles.values()):
        raise ValueError("Effective probe_roles must identify PFC and HPC")
    fixed = _fixed_scientific_config()
    for name, value in fixed.items():
        if name in effective_config and effective_config[name] != value:
            raise ValueError(f"Effective configuration {name} contradicts the fixed analysis")
    return {**effective_config, **fixed}


def _fixed_scientific_config() -> dict:
    """Return JSON-safe fixed target/split/window/weight/estimator definitions, with bounds in seconds."""
    return {
        "target_names": list(TARGET_NAMES), "window_names": list(WINDOW_NAMES),
        "weighting_names": list(WEIGHTING_NAMES), "transfers": _transfer_definitions(),
        "window_bounds_s": {name: list(training_window_bounds(name)) for name in WINDOW_NAMES},
        "pca_source_window_s": [-2., 2.], "source_test_policy": "all_source_to_complementary_conditions",
        "conflict_policy": "exclude_recorded_correctness_conflicts", "ignore_blocks": True,
        "weight_normalization": "equal_trial=1; equal_cell=N/(2*cell_count); divide total across selected bins",
        "logistic": {"C": LOGISTIC_C, "l1_ratio": 0., "solver": "lbfgs", "fit_intercept": True,
                     "class_weight": None, "tol": 1e-4, "max_iter": 5000},
    }


def _transfer_definitions() -> list[dict]:
    """Return JSON-safe fixed S=16 split definitions with cell names ordered by actual class 0/1."""
    return [{**asdict(item), "source_cells": list(item.source_cells), "test_cells": list(item.test_cells)} for item in TRANSFERS]


def _registry_arrays() -> dict[str, np.ndarray]:
    """Return named string arrays for T/C/H/G/S/role identities, with cell-name arrays(S,2)."""
    return {"target_names": np.asarray(TARGET_NAMES), "condition_names": np.asarray(CONDITION_IDENTIFIERS),
            "window_names": np.asarray(WINDOW_NAMES), "weighting_names": np.asarray(WEIGHTING_NAMES),
            "split_ids": np.asarray([item.split_id for item in TRANSFERS]),
            "split_target_names": np.asarray([item.target_name for item in TRANSFERS]),
            "source_cell_names": np.asarray([item.source_cells for item in TRANSFERS]),
            "test_cell_names": np.asarray([item.test_cells for item in TRANSFERS]), "role_names": np.asarray(ROLE_NAMES)}


def _git_provenance(repository: Path) -> dict:
    """Return revision string/status lines from a repository Path; no file edits or fitting identity changes."""
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repository, capture_output=True, text=True, check=False)
    except OSError:
        return {"code_revision": "unknown", "code_status": []}
    return {"code_revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
            "code_status": status.stdout.splitlines() if status.returncode == 0 else []}


def _validate_metadata(meta: dict) -> None:
    """Validate version/labels/axes/units and portable provenance dictionaries without opening referenced files."""
    if not isinstance(meta, dict) or (meta.get("analysis_version"), meta.get("schema_version")) != (ANALYSIS_VERSION, SCHEMA_VERSION):
        raise ValueError("Unsupported cross-condition result analysis/schema version")
    if not isinstance(meta.get("identity"), str) or not meta["identity"]:
        raise ValueError("Missing fitting identity")
    if not isinstance(meta.get("session_id"), str) or not meta["session_id"]:
        raise ValueError("Missing session identifier")
    config = meta["effective_config"]
    if "prediction_window_s" in config:
        raise ValueError("Saved decoder intervals must distinguish supervised windows from PCA source coverage")
    for name, expected in _fixed_scientific_config().items():
        if config.get(name) != expected:
            raise ValueError(f"Saved scientific configuration {name} disagrees with this analysis version")
    width = config["bin_width_ms"]
    if type(width) is not int or width not in SUPPORTED_BIN_WIDTH_MS:
        raise ValueError("Invalid saved bin width")
    if set(config["probe_roles"]) != {"PFC", "HPC"}:
        raise ValueError("Saved probe roles must identify PFC and HPC")
    if any(meta["axes"].get(name) != value for name, value in AXES.items()):
        raise ValueError("Saved axis names disagree with transfer array contracts")
    if any(meta["units"].get(name) != value for name, value in UNITS.items()):
        raise ValueError("Saved units disagree with original-PC readout contracts")
    if meta.get("randomness_used") is not False or meta.get("random_seed") is not None:
        raise ValueError("Transfer fitting must record deterministic, nonrandom execution")
    for source in meta["source_pcas"].values():
        _validate_file_reference(source)
        if not isinstance(source.get("configuration_id"), str) or not source["configuration_id"]:
            raise ValueError("Missing PCA configuration identity")
    for path, source in meta["source_pcas"].items():
        if path != source["path"]:
            raise ValueError("PCA source key does not match its portable path")
    if any(path not in meta["source_pcas"] for path in meta["region_sources"].values()):
        raise ValueError("Unknown source path in regional source map")
    if not set(meta["region_sources"]).issubset(config["probe_roles"]):
        raise ValueError("Unknown regional source role")
    if not {"augmented_trials", "base_config", "feature_parameters"}.issubset(meta["target_sources"]):
        raise ValueError("Missing required behavioral/config source identities")
    for source in (*meta["target_sources"].values(), meta["session_metadata"], *meta["scientific_sources"].values()):
        _validate_file_reference(source)
    for name in TARGET_NAMES:
        definition, entry = TARGET_DEFINITION_BY_IDENTIFIER[name], meta["target_metadata"][name]
        if (entry["family"] != "categorical" or entry["source_column"] != definition.source_column
                or entry["positive_class"] != 1 or entry["positive_label"] != definition.positive_label
                or entry["class_labels"] != _CATEGORICAL_CLASS_LABELS[name]):
            raise ValueError(f"Saved target metadata has incorrect binary orientation for {name}")


def _validate_file_reference(source: dict) -> None:
    """Check one portable path/byte-size/SHA-256 record structurally, without filesystem access."""
    path = source["path"]
    if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
        raise ValueError("Source references must use contained relative paths")
    if type(source["size"]) is not int or source["size"] < 0 or not isinstance(source["sha256"], str) or len(source["sha256"]) != 64:
        raise ValueError("Source reference requires byte size and whole-file SHA-256")


def _validate_result(result: CrossConditionResult) -> None:
    """Validate a complete N/W/B/K/H/G/S packet using only saved labels, memberships and model arrays."""
    _validate_metadata(result.meta)
    _validate_inputs(result.inputs, result.meta["effective_config"]["bin_width_ms"])
    if result.inputs.target_metadata != result.meta["target_metadata"]:
        raise ValueError("Prepared and saved target metadata disagree")
    available, unavailable = set(result.regions), set(result.unavailable_regions)
    if available & unavailable or available | unavailable != set(result.meta["effective_config"]["probe_roles"]):
        raise ValueError("Every configured region must be available or have one unavailable reason")
    if not all(isinstance(reason, str) and reason for reason in result.unavailable_regions.values()):
        raise ValueError("Unavailable regions require explicit reasons")
    references = set(result.meta["region_sources"])
    if not available.issubset(references) or not references.issubset(available | unavailable):
        raise ValueError("Regional source references contradict available/unavailable roles")
    for model in result.regions.values():
        _validate_region(result.inputs, model)


def _shape(array: np.ndarray, shape: tuple[int, ...], name: str, kind: str) -> None:
    """Require exact array axes/dtype kind (float64 for kind f); raise ValueError or return None."""
    if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype.kind not in kind:
        raise ValueError(f"Invalid {name}: expected shape {shape} and dtype kind {kind}")
    if kind == "f" and array.dtype != np.float64:
        raise ValueError(f"Invalid {name}: expected float64 to preserve the saved numerical contract")


def _validate_inputs(inputs: TransferInputs, width_ms: int) -> None:
    """Validate saved N-choice behavior, bool masks, current-trial cells and exact second grid; return None."""
    if inputs.target_names != TARGET_NAMES or inputs.transfers != TRANSFERS or inputs.window_names != WINDOW_NAMES or inputs.condition_names != CONDITION_IDENTIFIERS:
        raise ValueError("Prepared registry names/order disagree with this analysis")
    n, bins = len(inputs.choices), len(inputs.bin_edges_s) - 1
    _shape(inputs.target_values, (n, 4), "target labels", "f")
    for name, shape in (("target_valid", (n, 4)), ("baseline_valid", (n,)), ("conflict_mask", (n,)),
                        ("condition_masks", (n, len(CONDITION_IDENTIFIERS))), ("source_cells", (n, 16, 2)),
                        ("test_cells", (n, 16, 2)), ("window_bin_masks", (3, bins))):
        _shape(getattr(inputs, name), shape, name, "b")
    _shape(inputs.bin_edges_s, (bins + 1,), "bin edges", "f")
    if not np.array_equal(inputs.bin_edges_s, choice_bin_edges(width_ms / 1000)):
        raise ValueError("Saved grid contradicts selected bin width and full PCA window")
    expected_windows = np.vstack([training_bin_mask(inputs.bin_edges_s, training_window=name) for name in WINDOW_NAMES])
    if not np.array_equal(inputs.window_bin_masks, expected_windows):
        raise ValueError("Saved supervised bin masks contradict exact window bounds")
    if not np.isin(inputs.target_values[inputs.target_valid], [0, 1]).all() or not np.isnan(inputs.target_values[~inputs.target_valid]).all():
        raise ValueError("Binary target values and label-validity masks disagree")
    _validate_choices(inputs.choices, inputs.target_values[:, 1], inputs.target_valid[:, 1])
    if (type(inputs.dark_state_count) is not int or inputs.dark_state_count < 0
            or inputs.dark_state_count != int((~inputs.baseline_valid).sum())
            or not np.array_equal(inputs.baseline_valid, inputs.target_valid[:, 0])):
        raise ValueError("Common dark-state eligibility/count disagrees with ordinary-choice context validity")
    _validate_conflicts(inputs.choices, inputs.target_values, inputs.target_valid, inputs.conflict_mask, inputs.conflicts)
    common = inputs.baseline_valid & ~inputs.conflict_mask
    correct, reward = inputs.target_values[:, 3], inputs.target_values[:, 2]
    expected_conditions = {"all": common, "correct_rewarded": common & (correct == 1) & (reward == 1),
                           "omission": common & (correct == 1) & (reward == 0), "incorrect": common & (correct == 0)}
    conditions = dict(zip(inputs.condition_names, inputs.condition_masks.T, strict=True))
    for name, expected in expected_conditions.items():
        if not np.array_equal(conditions[name], expected):
            raise ValueError(f"Saved {name} current-trial condition mask disagrees with labels/common eligibility")
    if np.any(inputs.condition_masks & ~common[:, None]):
        raise ValueError("Condition masks include common-ineligible choices")
    # Switch/stay can depend on excluded intervening original rows: retain them,
    # never reconstruct a compressed chronology from the saved ordinary choices.
    expected_source, expected_test = _transfer_cells(_named_cells(inputs.target_values, conditions, common), inputs.target_values, inputs.target_valid)
    if not np.array_equal(inputs.source_cells, expected_source) or not np.array_equal(inputs.test_cells, expected_test):
        raise ValueError("Saved source/test cells disagree with current-trial definitions or labels")


def _validate_choices(choices: pd.DataFrame, action: np.ndarray, action_valid: np.ndarray) -> None:
    """Check original N-row choice identities/times against binary action(N,) without changing raw ID types."""
    if not {"row_position", "trial_id", "choice_time", "action"}.issubset(choices.columns):
        raise ValueError("Saved choice coordinates are incomplete")
    positions = choices.row_position.to_numpy()
    _shape(positions, (len(choices),), "original row positions", "iu")
    if np.any(positions < 0) or np.any(np.diff(positions) <= 0) or not trial_identities_match(choices.trial_id, positions):
        raise ValueError("Saved original row/trial identities must be ordered and consistent")
    times = choices.choice_time.to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("Saved UTC choice times must be finite and chronological")
    if not action_valid.all() or not np.array_equal(choices.action.to_numpy(), action):
        raise ValueError("Saved choice actions disagree with target labels")


def _validate_conflicts(
    choices: pd.DataFrame, values: np.ndarray, valid: np.ndarray,
    conflict: np.ndarray, audit: pd.DataFrame,
) -> None:
    """Validate bool conflict(N,) and raw audit rows against saved labels(N,4)/original coordinates(N,column)."""
    complete = valid[:, 0] & valid[:, 1] & valid[:, 3]
    expected = complete & (values[:, 3] != (values[:, 0] == values[:, 1]))
    if not np.array_equal(conflict, expected):
        raise ValueError("Conflict exclusion mask disagrees with recorded binary labels")
    if not {"row_position", "trial_id", "state_int", "action", "correct", "reason"}.issubset(audit.columns):
        raise ValueError("Conflict audit columns are incomplete")
    if not np.array_equal(audit.row_position.to_numpy(), choices.row_position.to_numpy()[conflict]) or not trial_identities_match(audit.trial_id, choices.trial_id.to_numpy()[conflict]):
        raise ValueError("Conflict audit original row/trial identities disagree with exclusion mask")
    for name, target in (("state_int", 0), ("action", 1), ("correct", 3)):
        if not np.array_equal(pd.to_numeric(audit[name], errors="raise").to_numpy(dtype=float), values[conflict, target]):
            raise ValueError("Conflict audit recorded values disagree with saved labels")
    if any(not isinstance(value, str) or not value for value in audit.reason):
        raise ValueError("Conflict audit requires nonempty exclusion reasons")


def _validate_region(inputs: TransferInputs, model: RegionTransferResult) -> None:
    """Check regional W/K/H/G/S models, exact eligible cells/weights and NaN/finite availability conventions."""
    if (not isinstance(model.choice_positions, np.ndarray) or model.choice_positions.ndim != 1
            or not isinstance(model.pc_ids, np.ndarray) or model.pc_ids.ndim != 1):
        raise ValueError("Regional choice positions and PC IDs must have one-dimensional axes")
    n, w, k = len(inputs.choices), len(model.choice_positions), len(model.pc_ids)
    _shape(model.choice_positions, (w,), "regional choice positions", "iu")
    if not w or np.any(model.choice_positions < 0) or np.any(model.choice_positions >= n) or np.any(np.diff(model.choice_positions) <= 0):
        raise ValueError("Regional maps must be ordered unique shared-choice indices")
    _shape(model.pc_ids, (k,), "original PC IDs", "iu")
    if not k or not np.array_equal(model.pc_ids, np.arange(1, k + 1)):
        raise ValueError("Original one-based PC identities must remain complete and ordered")
    shape = (3, 2, 16)
    for name, axes, kind in (("coefficients", (*shape, k), "f"), ("intercepts", shape, "f"),
                             ("fit_status", shape, "U"), ("fit_reason", shape, "U"),
                             ("trial_probabilities", (*shape, w), "f"), ("source_mask", (16, w), "b"),
                             ("test_mask", (16, w), "b"), ("source_trial_weights", (2, 16, w), "f"),
                             ("eligible_cell_counts", (16, 2, 2), "iu")):
        _shape(getattr(model, name), axes, name, kind)
    source, test = _regional_cells(inputs, model.choice_positions)
    if not np.array_equal(model.source_mask, source.any(axis=2)) or not np.array_equal(model.test_mask, test.any(axis=2)):
        raise ValueError("Regional source/test masks disagree with exact eligibility and coverage")
    counts = np.stack([source.sum(axis=1), test.sum(axis=1)], axis=1)
    if not np.array_equal(model.eligible_cell_counts, counts):
        raise ValueError("Regional eligible cell counts disagree with source/test membership")
    for split in range(16):
        for weighting, scheme in enumerate(WEIGHTING_NAMES):
            expected = training_trial_weights(source[split], scheme)
            if not np.allclose(model.source_trial_weights[weighting, split], expected, rtol=1e-12, atol=1e-12):
                raise ValueError("Regional training weights disagree with post-coverage cell normalization")
        for window in range(3):
            for weighting in range(2):
                index = window, weighting, split
                beta, intercept, probability = model.coefficients[index], model.intercepts[index], model.trial_probabilities[index]
                available = model.fit_status[index] == "ok"
                if model.fit_status[index] not in ("ok", "unavailable") or bool(model.fit_reason[index]) == available:
                    raise ValueError("Fit status/reason availability conventions disagree")
                if not available:
                    if not np.isnan(beta).all() or not np.isnan(intercept) or not np.isnan(probability).all():
                        raise ValueError("Unavailable models must have NaN parameters and trial probabilities")
                    continue
                if not np.all(counts[split, 0] > 0) or not np.isfinite(beta).all() or not np.isfinite(intercept):
                    raise ValueError("Available models require two source classes and finite affine parameters")
                selected = model.source_mask[split] | model.test_mask[split]
                if (not np.isfinite(probability[selected]).all() or not np.isnan(probability[~selected]).all()
                        or np.any((probability[selected] < 0) | (probability[selected] > 1))):
                    raise ValueError("Saved trial probabilities must be finite [0,1] exactly on eligible source/test choices")
