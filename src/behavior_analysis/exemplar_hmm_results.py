"""Versioned fixed-exemplar arrays and immutable exports, independent of plots."""

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from . import block_exemplar_models
from . import exemplar_hmm_modeling as modeling


ANALYSIS_VERSION = "fixed_exemplar_hmm_v1"
ANALYSIS_SOURCE_FILES = (
    "block_exemplar_models.py", "exemplar_hmm_modeling.py", "exemplar_hmm_results.py",
)
NUMERICAL_ARRAY_NAMES = (
    "valid_mask", "valid_positions", "sigmas", "raw_residuals", "z_residuals",
    "log_likelihoods", "independent_probs", "initial_distribution",
    "transition_matrix", "smoothed_probs", "smoothed_states", "marginal_loglik",
)
LEGACY_BLOCK_COLUMNS = ("cur_strategy_slope", "hardcoded_strategy", "inferred_strategy")
NATIVE_OUTPUT_PREFIXES = (
    "exemplar_z_residual_", "exemplar_log_likelihood_",
    "exemplar_independent_prob_", "exemplar_hmm_smoothed_", "exemplar_ll_argmax_",
    "dynamax_state_",
)


def preserve_current_exports(processed_data_path: Path, session_id: str) -> None:
    """Archive conventional CSV bytes once per distinct content before replacing.

    Parameters
    ----------
    processed_data_path : pathlib.Path
        Session/joint processed directory containing zero or more conventional
        `<session_id>_block_performance.csv` and `_augmented_trials.csv` files.
    session_id : str
        Full session or synthetic joint identifier used in both filenames.

    Returns
    -------
    None
        Existing CSVs remain unchanged. Copies live beneath
        `exemplar_hmm/archives/<sha256>/<original_stem>.archived.csv`. Identical
        bytes are not copied again, including legacy archives with the original
        filename. Old model pickles and source sessions are untouched.
        No array, axis or physical-unit transformations occur.
    """
    root = Path(processed_data_path)
    for suffix in ("block_performance.csv", "augmented_trials.csv"):
        source = root / f"{session_id}_{suffix}"
        if not source.exists():
            continue
        content = source.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        archive_directory = root / "exemplar_hmm" / "archives" / digest
        destination = archive_directory / f"{source.stem}.archived.csv"
        existing = destination if destination.exists() else archive_directory / source.name
        if existing.exists():
            if existing.read_bytes() != content:
                raise ValueError(f"Archived export content does not match its digest: {existing}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as archived:
            archived.write(content)


def save_or_load_exemplar_hmm_results(
    block_performance: pd.DataFrame,
    trial_df: pd.DataFrame,
    processed_data_path: Path,
    session_id: str,
    specs: Sequence[block_exemplar_models.BlockExemplarModelSpec] = block_exemplar_models.DEFAULT_BLOCK_EXEMPLAR_MODELS,
    stay_prob: float = 0.9,
    rerun: bool = False,
    source_manifest: Sequence[dict] | None = None,
) -> tuple[modeling.FixedExemplarBlockStateResult, pd.DataFrame, Path]:
    """Reuse numerical arrays while exporting current blocks and trial features.

    Parameters
    ----------
    block_performance : pandas.DataFrame
        Ordered `(n_blocks, n_block_columns)` table with explicit `block_ix`,
        `trials_to_correct` (trials), and `prev_n_rewarded` (rewarded trials).
        Saved exemplar residuals are regenerated from these raw inputs and the
        exact supplied specs. Legacy derived labels are removed from exports.
    trial_df : pandas.DataFrame
        Current `(n_trials, n_trial_columns)` table with `cur_block` matching
        `block_ix`. If either table has `source_session_id`, both must have it.
        Current trial features are preserved; inherited model columns refresh.
    processed_data_path : pathlib.Path
        Destination processed directory for this session or synthetic joint
        session. Joint inference never writes to its source session folders.
    session_id : str
        Full session or joint identifier, used in provenance/current filenames.
    specs : Sequence[BlockExemplarModelSpec]
        Ordered fixed exemplars, in trials/rewarded-trial units. State indices
        follow this order. At least one valid spec is required.
    stay_prob : float, default=0.9
        Unitless diagonal transition probability in [0, 1].
    rerun : bool, default=False
        Force a new immutable numerical run even when matching arrays exist.
    source_manifest : Sequence[dict] or None
        Ordered plain-data source records, including sources with zero blocks.
        Any manifest change invalidates the joint cache. None derives full
        session IDs in block order; same-date sessions remain distinct.

    Returns
    -------
    tuple[FixedExemplarBlockStateResult, pandas.DataFrame, pathlib.Path]
        Full current block result, current inherited trial table, and immutable
        NPZ path. Numerical arrays use valid-block rows by ordered model columns;
        original table order/index survives. Every call saves fresh immutable
        CSV snapshots and provenance. Cache hits return the existing NPZ without
        rerunning inference, with a new timestamped export record. Conventional
        CSVs refresh only after archival. No plots or opaque models are saved.
    """
    started = perf_counter()
    specs = tuple(specs)
    if not specs:
        raise ValueError("specs must contain at least one exemplar model.")
    block_exemplar_models.validate_exemplar_model_specs(specs)
    modeling.build_sticky_transition_matrix(len(specs), stay_prob)
    modeling.require_columns(block_performance, ["block_ix"], "block_performance")
    modeling.require_columns(trial_df, ["cur_block"], "trial_df")

    # Archive before stripping derived columns or replacing conventional files.
    preserve_current_exports(processed_data_path, session_id)
    clean_blocks = _remove_obsolete_block_outputs(block_performance)
    prepared_blocks, parameters = block_exemplar_models.add_block_exemplar_model_outputs(clean_blocks, specs)
    metadata = _make_metadata(prepared_blocks, trial_df, session_id, specs, stay_prob, source_manifest)
    signature = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
    metadata["signature"] = signature
    cache_root = Path(processed_data_path) / "exemplar_hmm" / ANALYSIS_VERSION / signature
    previous = [] if rerun else sorted(cache_root.glob("*/results.npz"))
    # Only complete runs are reusable; failed exports retain their diagnostics.
    previous = [path for path in previous if (path.parent / "complete.json").exists()]
    numerical_path = previous[-1] if previous else None
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_%f_UTC")
    run_directory = cache_root / timestamp
    run_directory.mkdir(parents=True, exist_ok=False)
    metadata.update(
        created_at=timestamp, cache_hit=bool(previous), warnings=[], errors=[],
        export_record_directory=str(run_directory),
    )
    recorded_warnings = []
    try:
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            if numerical_path is None:
                result = modeling.compute_fixed_exemplar_block_states(prepared_blocks, parameters, stay_prob)
                numerical_path = run_directory / "results.npz"
            else:
                result = _restore_result(numerical_path, prepared_blocks, parameters, stay_prob)
            inherited_trials = modeling.inherit_fixed_exemplar_block_states(result.block_performance, trial_df)
        metadata["warnings"] = [str(record.message) for record in recorded_warnings]
        metadata["n_valid_blocks"] = int(result.valid_mask.sum())
        metadata["status"] = "ok" if result.valid_positions.size else "no_valid_blocks"
        metadata["runtime_seconds"] = perf_counter() - started
        metadata["numerical_artifact"] = str(numerical_path)
        result.block_performance.to_csv(run_directory / "block_performance.csv", index=False, na_rep="None")
        inherited_trials.to_csv(run_directory / "augmented_trials.csv", index=False, na_rep="None")
        _write_run_record(run_directory, metadata)
        if not previous:
            _save_numerical_arrays(numerical_path, result, metadata)
        result.block_performance.to_csv(
            Path(processed_data_path) / f"{session_id}_block_performance.csv", index=False, na_rep="None",
        )
        inherited_trials.to_csv(
            Path(processed_data_path) / f"{session_id}_augmented_trials.csv", index=False, na_rep="None",
        )
        (run_directory / "complete.json").write_text(
            json.dumps({"numerical_artifact": str(numerical_path)}), encoding="utf-8",
        )
    except Exception as error:
        metadata["runtime_seconds"] = perf_counter() - started
        metadata["warnings"] = [str(record.message) for record in recorded_warnings]
        metadata["errors"] = [f"{type(error).__name__}: {error}"]
        _write_run_record(run_directory, metadata)
        raise
    return result, inherited_trials, numerical_path


# Private helpers: provenance, current-table restoration, and serialization.
def _remove_obsolete_block_outputs(blocks: pd.DataFrame) -> pd.DataFrame:
    """Return raw/current block columns after removing stale modeling outputs.

    Input is a `(n_blocks, n_columns)` DataFrame with trial/reward counts and
    metadata. Returns a copy on the same row/index axis. Legacy state/slope,
    native state outputs and all old exemplar residuals are removed so the
    supplied specs can regenerate residuals in trials/residual-SD units.
    """
    obsolete = [
        column for column in blocks
        if column in LEGACY_BLOCK_COLUMNS or column.startswith(NATIVE_OUTPUT_PREFIXES)
        or column.endswith(("_exemplar_residual_TTS", "_exemplar_normalized_residual_TTS"))
    ]
    return blocks.drop(columns=obsolete).copy()


def _make_metadata(
    blocks: pd.DataFrame, trials: pd.DataFrame, session_id: str,
    specs: Sequence[block_exemplar_models.BlockExemplarModelSpec], stay_prob: float,
    source_manifest: Sequence[dict] | None,
) -> dict:
    """Build stable numerical identity and plain provenance before runtime fields.

    Inputs are full ordered block/trial DataFrames `(N, C)`/`(T, D)`, full ID,
    ordered fixed specs (trial units), unitless stay probability and optional
    ordered plain source records. Returns a JSON-compatible dictionary with
    input/source/code digests and units/axis contracts. Trial features and bias
    are excluded from numerical identity because inheritance is rebuilt fresh.
    """
    manifest = _ordered_source_manifest(blocks, trials, session_id, source_manifest)
    identity_columns = [column for column in blocks if column.startswith("source_")]
    input_columns = ["block_ix", "trials_to_correct", "prev_n_rewarded", *identity_columns]
    package_versions = {}
    for package in ("numpy", "pandas", "scipy", "jax", "dynamax"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "unavailable"
    return {
        "generator": "exemplar_hmm_results.save_or_load_exemplar_hmm_results",
        "analysis_version": ANALYSIS_VERSION, "session_id": str(session_id),
        "scope": "multisession" if source_manifest is not None or "source_session_id" in blocks else "single_session",
        "parameters": {"specs": [asdict(spec) for spec in specs], "stay_prob": float(stay_prob)},
        "model_names": [spec.name for spec in specs], "random_seed": None,
        "input_digest": _table_digest(blocks[input_columns]),
        "source_row_digest": _table_digest(blocks[["block_ix", *identity_columns]]),
        "input_shape": [len(blocks), len(input_columns)],
        "source_manifest": manifest, "analysis_source_digest": _analysis_source_digest(),
        "package_versions": package_versions,
        "units": {"raw_residuals": "trials", "sigmas": "trials", "z_residuals": "residual SD",
                  "prev_n_rewarded": "rewarded trials", "probabilities": "unitless"},
        "axis_conventions": {
            "tables": "original row order/index; explicit block_ix -> cur_block identity",
            "numerical_arrays": "valid-block positions by ordered exemplar/model index",
            "states": "zero-based indices into model_names; invalid rows have missing labels",
            "valid_positions": "zero-based positions into the complete block table",
            "sequence": "one continuous valid-block sequence across all ordered sources",
        },
    }


def _ordered_source_manifest(
    blocks: pd.DataFrame, trials: pd.DataFrame, session_id: str,
    manifest: Sequence[dict] | None,
) -> list[dict]:
    """Return plain ordered source metadata, including explicit zero-row sources.

    Inputs are full block/trial DataFrames `(N,C)`/`(T,D)`, a fallback session ID
    and an optional ordered sequence of JSON-compatible dictionaries. Returns a
    copied list, with no array reshaping/unit conversion. None derives unique
    full source IDs in first block occurrence order and their block/trial counts.
    """
    if manifest is not None:
        return json.loads(json.dumps(list(manifest)))
    if "source_session_id" not in blocks:
        return [{"session_id": str(session_id), "n_blocks": len(blocks), "n_trials": len(trials)}]
    records = []
    for source in pd.unique(blocks["source_session_id"]):
        matching_blocks = blocks["source_session_id"].eq(source)
        n_trials = int(trials["source_session_id"].eq(source).sum()) if "source_session_id" in trials else 0
        records.append({"session_id": str(source), "n_blocks": int(matching_blocks.sum()), "n_trials": n_trials})
    return records


def _table_digest(table: pd.DataFrame) -> str:
    """Hash a `(N,C)` table's ordered schema, index labels and scalar values.

    Input is a DataFrame in any units; returns a SHA-256 hexadecimal string.
    No data or units are changed. Exact schema and index changes invalidate the
    cache, even when numerical values could otherwise be treated as equivalent.
    """
    digest = hashlib.sha256()
    schema = {"columns": [(str(column), str(dtype)) for column, dtype in table.dtypes.items()],
              "index_names": [str(name) for name in table.index.names]}
    digest.update(json.dumps(schema, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(table.astype(str), index=True).to_numpy().tobytes())
    return digest.hexdigest()


def _analysis_source_digest() -> str:
    """Return SHA-256 of relevant project sources, including uncommitted edits.

    No inputs, arrays or units. Returns a hexadecimal string; project numerical
    and storage modules plus main/batch orchestration determine source identity.
    """
    digest = hashlib.sha256()
    sources = [Path(__file__).with_name(name) for name in ANALYSIS_SOURCE_FILES]
    sources.append(Path(__file__).resolve().parents[1] / "main.py")
    for source in sources:
        digest.update(source.name.encode("ascii"))
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _restore_result(
    path: Path, blocks: pd.DataFrame, parameters: pd.DataFrame, stay_prob: float,
) -> modeling.FixedExemplarBlockStateResult:
    """Attach saved numerical arrays to current blocks without invoking inference.

    Inputs are a trusted matching project NPZ Path (meta uses object storage),
    current `(N,C)` prepared block table, current `(K,P)` ordered parameters with
    SDs in trials, and unitless stay probability. Returns a data-only full-row
    result using valid arrays `(V,K)` and original current index/order. No saved
    feature table is loaded; the caller has already validated cache identity.
    """
    with np.load(path, allow_pickle=True) as saved:
        arrays = {name: saved[name].copy() for name in NUMERICAL_ARRAY_NAMES}
        metadata = saved["meta"].item()
    names = metadata["model_names"]
    valid_outputs = modeling.add_fixed_exemplar_hmm_outputs(
        pd.DataFrame(index=arrays["valid_positions"]), names, arrays["z_residuals"],
        arrays["log_likelihoods"], arrays["independent_probs"], arrays["smoothed_probs"],
    ).reindex(np.arange(len(blocks)))
    output_df = blocks.copy()
    for column in valid_outputs:
        values = valid_outputs[column]
        if column in ("exemplar_ll_argmax_state", "exemplar_hmm_smoothed_state"):
            values = values.astype("Int64")
        elif column in ("exemplar_ll_argmax_model", "exemplar_hmm_smoothed_model"):
            values = values.astype(object)
        output_df[column] = values.array
    output_df["inferred_strategy"] = output_df["exemplar_hmm_smoothed_state"].copy()
    smoothing = modeling.FixedExemplarHMMSmoothingResult(
        initial_distribution=arrays["initial_distribution"], transition_matrix=arrays["transition_matrix"],
        smoothed_probs=arrays["smoothed_probs"], smoothed_states=arrays["smoothed_states"],
        marginal_loglik=float(arrays["marginal_loglik"]),
    )
    return modeling.FixedExemplarBlockStateResult(
        block_performance=output_df, valid_mask=arrays["valid_mask"], valid_positions=arrays["valid_positions"],
        model_names=names, parameters=parameters.copy(), sigmas=arrays["sigmas"],
        raw_residuals=arrays["raw_residuals"], z_residuals=arrays["z_residuals"],
        log_likelihoods=arrays["log_likelihoods"], independent_probs=arrays["independent_probs"],
        smoothing_result=smoothing, stay_prob=float(stay_prob),
    )


def _save_numerical_arrays(path: Path, result: modeling.FixedExemplarBlockStateResult, metadata: dict) -> None:
    """Save plain named arrays and a `meta` dictionary into a new trusted NPZ.

    Inputs are a new Path, data-only result with valid matrices `(V,K)` in
    trials/residual-SD/probability units, and plain provenance dict. Returns None;
    no models or DataFrames are serialized and existing artifacts are untouched.
    """
    arrays = {name: getattr(result, name) for name in NUMERICAL_ARRAY_NAMES[:7]}
    smoothing = result.smoothing_result
    arrays.update({name: getattr(smoothing, name) for name in NUMERICAL_ARRAY_NAMES[7:]})
    with path.open("xb") as output:
        np.savez_compressed(output, **arrays, meta=metadata)


def _write_run_record(directory: Path, metadata: dict) -> None:
    """Save provenance, logs and source/main/batch snapshots for one new run.

    Inputs are a new timestamped output Path and a plain metadata dictionary
    containing session/source IDs, parameters, timings (seconds), warnings and
    errors. Returns None; writes JSON/Markdown/log and source copies without
    modifying numerical axes, units, current exports or prior run directories.
    """
    serialized = json.dumps(metadata, indent=2)
    (directory / "metadata.json").write_text(serialized, encoding="utf-8")
    (directory / "run.log").write_text(serialized, encoding="utf-8")
    source_root = Path(__file__).resolve().parents[1]
    shutil.copy2(source_root / "main.py", directory / "main.py")
    for name in ANALYSIS_SOURCE_FILES:
        shutil.copy2(Path(__file__).with_name(name), directory / name)
    # Main.py owns both session and batch orchestration, so snapshot the actual
    # same source under each role rather than inventing a separate batch runner.
    (directory / "batch.py").write_text(
        "# Batch source snapshot: main.py contains both orchestration paths.\n"
        + (source_root / "main.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    summary = (
        "# Fixed-exemplar block HMM\n\n"
        "Goal: infer ordered exemplar strategies from trials-to-correct residuals "
        "using a fixed sticky transition matrix and Dynamax smoothing.\n\n"
        f"Output session: {metadata['session_id']}. Scope: {metadata['scope']}.\n\n"
        f"Included ordered source sessions: {json.dumps(metadata['source_manifest'])}.\n\n"
        "Main and batch scripts: main.py and batch.py snapshot the same actual "
        "source, since both orchestration paths reside in main.py. Computation/storage sources "
        "are included alongside these snapshots.\n\n"
        f"Numerical cache hit: {metadata['cache_hit']}. "
        f"Runtime: {metadata['runtime_seconds']:.6f} seconds.\n\n"
        f"Observation status: {metadata.get('status', 'error')}. "
        f"Valid blocks: {metadata.get('n_valid_blocks', 'unavailable')}.\n\n"
        "Arrays use valid-block rows by ordered exemplar columns. Full tables retain "
        "invalid blocks with missing states. Inference is continuous across sources; "
        "within-session history features are preserved. No statistical tests were performed.\n"
    )
    (directory / "summary.md").write_text(summary, encoding="utf-8")
