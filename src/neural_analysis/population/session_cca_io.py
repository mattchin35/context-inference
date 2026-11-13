"""Paired choice-window CCA inputs and metadata-only scientific configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from pickle import UnpicklingError
import tempfile
import warnings
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from src.neural_analysis.population import cca
from src.neural_analysis.population import session_timeline as timeline
from src.neural_analysis.population import session_timeline_io as timeline_io
from src.neural_analysis.session_metadata import ResolvedSession
from src.neural_analysis.task_decoding.activity import inspect_probe_coverage


ANALYSIS_VERSION = "session-timeline-cca-v1"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PairedCCAInputs:
    """One session's shared behavior and complete paired regional count windows.

    choices retains all N choices, original row/trial IDs, UTC choice_time and
    elapsed_s. time_origin_s is first-trial UTC Unix seconds; events retains
    source event rows plus elapsed_s. choice_positions is integer (W,) indexing
    choices, while coverage_positions preserves regional eligible positions.
    bin_edges_s is (B+1,) choice-relative seconds; counts maps PFC/HPC to float64
    (W,B,U_region) unsmoothed counts per bin, with qualified ordered unit_ids.
    probe_roles/configuration retain metadata IDs and the scientific identity.
    """

    session_id: str
    time_origin_s: float
    choices: pd.DataFrame
    events: pd.DataFrame
    choice_positions: np.ndarray
    coverage_positions: dict[str, np.ndarray]
    bin_edges_s: np.ndarray
    probe_roles: dict[str, str]
    unit_ids: dict[str, tuple[str, ...]]
    counts: dict[str, np.ndarray]
    configuration: dict


@dataclass(frozen=True)
class SessionCCAResult:
    """Saved paired fit and behavior, or an in-memory unavailable display state.

    choices/events retain source columns and UTC/elapsed-second clocks.
    choice_positions (W,) and regional coverage_positions index choices;
    bin_edges_s (B+1,) is choice-relative seconds. probe_roles maps PFC/HPC
    to metadata IDs. fit contains (W,B,K) canonical-SD scores, or is None only
    for UI states explained by unavailable_reason. meta is the scientific and
    provenance dictionary. Completed files always contain a valid paired fit.
    """

    session_id: str
    time_origin_s: float
    choices: pd.DataFrame
    events: pd.DataFrame
    choice_positions: np.ndarray
    coverage_positions: dict[str, np.ndarray]
    bin_edges_s: np.ndarray
    probe_roles: dict[str, str]
    fit: cca.PairedCCAFit | None
    unavailable_reason: str | None
    meta: dict


@dataclass(frozen=True)
class CCAResultInfo:
    """Small archive header with no count/score arrays.

    path is the NPZ path, configuration is its saved metadata dictionary,
    completed_utc is an ISO timestamp string, and run_id is a unitless label.
    Discovery validates header support but array validity is checked on load.
    """

    path: Path
    configuration: dict
    completed_utc: str
    run_id: str


def cca_configuration(
    session: ResolvedSession,
    probe_roles: dict[str, str],
    bin_width_s: float,
    *,
    population_quality: str = "good_mua",
    missing_label_policy: str = "mua",
) -> dict:
    """Build a JSON-compatible CCA configuration without reading raw spike arrays.

    session supplies resolved source paths; probe_roles explicitly maps both
    PFC/HPC to distinct configured probe IDs. bin_width_s is .5/.1/.05/.02 s.
    Quality policies use the existing timeline selector's good_mua/good and
    mua/exclude conventions. Read only quality metadata, source stats and IRIG
    coverage. Return settings/source identities plus configuration_id; source
    times stay UTC seconds. Missing required inputs and invalid roles raise
    ValueError or the existing loader's ordinary source error.
    """
    roles = _validated_roles(session, probe_roles)
    timeline.choice_bin_edges(bin_width_s)
    sources = {
        "trials": _source_identity(session.behavior.trial_table_file),
        "events": _source_identity(session.behavior.event_table_file, required=False),
    }
    selection = {}
    acquisition = {}
    for region, probe_id in roles.items():
        source, channels, rows = timeline_io.select_timeline_population(
            session, probe_id, population_quality=population_quality,
            missing_label_policy=missing_label_policy,
        )
        sources[region] = {
            "alignment": _source_identity(source.alignment_file),
            "spike_clusters": _source_identity(source.spike_clusters_file),
            "cluster_info": _source_identity(source.cluster_info_file),
            "channel_quality": _source_identity(source.channel_quality_file),
        }
        coverage = inspect_probe_coverage(source)
        selection[region] = {
            "unit_ids": [f"{probe_id}:{cluster}" for cluster in rows.cluster_id],
            "channels": channels.tolist(),
            "coverage_s": [coverage.start_s, coverage.end_s],
        }
        acquisition[region] = source.acquisition_family
    configuration = {
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "numerical_version": cca.NUMERICAL_VERSION,
        "numerical_parameters": dict(cca.PARAMETERS),
        "session_id": session.session_id,
        "probe_roles": roles,
        "acquisition_families": acquisition,
        "bin_width_s": float(bin_width_s),
        "window_s": list(timeline.WINDOW_S),
        "population_quality": population_quality,
        "missing_label_policy": missing_label_policy,
        "window_pool": "complete_in_both_regions",
        "lag_s": 0.0,
        "task_activity": "retained",
        "score_normalization": "session_sample_sd_one",
        "sign_convention": "largest_absolute_pfc_coefficient_positive",
        "selection": selection,
        "sources": sources,
    }
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False)
    configuration["configuration_id"] = hashlib.sha256(serialized.encode("ascii")).hexdigest()[:16]
    return configuration


def load_cca_inputs(
    session: ResolvedSession,
    probe_roles: dict[str, str],
    bin_width_s: float,
    *,
    population_quality: str = "good_mua",
    missing_label_policy: str = "mua",
) -> PairedCCAInputs:
    """Load behavior and count tensors for the common complete regional windows.

    Arguments follow cca_configuration: explicit PFC/HPC probe IDs, seconds,
    and scalar quality policies. Return PairedCCAInputs with all behavior and
    counts (W,B,U_region), in counts per bin, indexed by common chronological
    choice positions. All temporal bin edges are choice-relative seconds;
    absolute choice/spike times remain UTC seconds. No LFP, PCA or CCA fit is
    performed, and no outputs are written. Missing common coverage/populations
    or inconsistent loaded axes raise ValueError or an ordinary source error.
    """
    configuration = cca_configuration(
        session, probe_roles, bin_width_s,
        population_quality=population_quality, missing_label_policy=missing_label_policy,
    )
    origin, choices, events = timeline_io.load_timeline_behavior(session)
    times = choices.choice_time.to_numpy(dtype=np.float64)
    coverage_positions = {
        region: timeline.select_complete_choice_windows(
            times, tuple(configuration["selection"][region]["coverage_s"])
        )
        for region in ("PFC", "HPC")
    }
    common = coverage_positions["PFC"][
        np.isin(coverage_positions["PFC"], coverage_positions["HPC"])
    ]
    if not len(common):
        raise ValueError("No choices have complete paired PFC/HPC neural windows.")

    edges = timeline.choice_bin_edges(bin_width_s)
    counts = {}
    unit_ids = {}
    for region, probe_id in configuration["probe_roles"].items():
        population = timeline_io.load_timeline_population(
            session, probe_id, population_quality=population_quality,
            missing_label_policy=missing_label_policy,
        )
        expected = configuration["selection"][region]
        if (population.unit_ids != tuple(expected["unit_ids"])
                or list(population.channels) != expected["channels"]
                or not np.array_equal(population.coverage_s, expected["coverage_s"])):
            raise ValueError(f"{region} loaded population differs from its source configuration.")
        actual_ids = tuple(f"{probe_id}:{cluster}" for cluster in population.cluster_ids)
        if population.unit_ids != actual_ids:
            raise ValueError(f"{region} unit IDs do not match the spike-group cluster axis.")
        values, regional_edges = timeline.bin_choice_counts(
            population.spike_group, population.cluster_ids, times[common], bin_width_s,
        )
        values = np.asarray(values, dtype=np.float64)
        expected_shape = (len(common), len(edges) - 1, len(population.unit_ids))
        if values.shape != expected_shape or not np.array_equal(regional_edges, edges):
            raise ValueError(f"{region} binned activity has inconsistent choice/bin/unit axes.")
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{region} binned counts must be finite and nonnegative.")
        counts[region] = values
        unit_ids[region] = population.unit_ids
    return PairedCCAInputs(
        session.session_id, origin, choices, events, common, coverage_positions,
        edges, dict(configuration["probe_roles"]), unit_ids, counts, configuration,
    )


def save_cca_result(output_path: Path | str, result: SessionCCAResult) -> Path:
    """Save a new trusted named-array NPZ, preserving scientific shapes/units.

    output_path must end in .npz and must not exist. result must hold a valid
    paired fit with the Section 3 axes; unavailable UI records are rejected.
    Return the saved Path. Metadata enrichment does not mutate result.meta;
    a temporary sibling prevents discovery of a partly written result.
    """
    path = Path(output_path)
    if path.suffix != ".npz":
        raise ValueError("CCA result path must end in .npz.")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    _validate_result(result)
    meta = _saved_metadata(result)
    arrays = {
        "choices": result.choices.to_records(index=False),
        "events": result.events.to_records(index=False),
        "choice_positions": result.choice_positions,
        "bin_edges_s": result.bin_edges_s,
        "correlations": result.fit.correlations,
        "joint_signs": result.fit.joint_signs,
        "meta": np.asarray(meta, dtype=object),
    }
    for region, transform in result.fit.transforms.items():
        prefix = region.lower() + "_"
        arrays[prefix + "scores"] = result.fit.scores[region]
        arrays[prefix + "coverage_positions"] = result.coverage_positions[region]
        for field in ("count_mean", "count_scale", "coefficients", "score_offset", "singular_values"):
            arrays[prefix + field] = getattr(transform, field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def load_cca_result(path: Path | str) -> SessionCCAResult:
    """Load a trusted CCA NPZ using only stored arrays/metadata, never raw sources.

    Return SessionCCAResult with original table fields, unit IDs, float64
    transforms/scores, and UTC/elapsed-second axes. Structural/finite/version
    inconsistencies raise ValueError; ordinary file errors remain OSError.
    Object loading is used for the project's dictionary/table metadata format.
    """
    try:
        arrays = _read_arrays(Path(path))
        meta = _validated_metadata(arrays["meta"].item())
        transforms = {}
        scores = {}
        coverage = {}
        for region in ("PFC", "HPC"):
            info = meta["regions"][region]
            prefix = region.lower() + "_"
            transforms[region] = cca.RegionalCCATransform(
                _unit_ids(info["input_unit_ids"]), _unit_ids(info["retained_unit_ids"]),
                _unit_ids(info["omitted_unit_ids"], allow_empty=True),
                arrays[prefix + "count_mean"], arrays[prefix + "count_scale"],
                arrays[prefix + "coefficients"], arrays[prefix + "score_offset"],
                arrays[prefix + "singular_values"], info["rank"], info["rank_cutoff"],
            )
            scores[region] = arrays[prefix + "scores"]
            coverage[region] = arrays[prefix + "coverage_positions"]
        fit = cca.PairedCCAFit(
            transforms, scores, arrays["correlations"], meta["n_observations"],
            meta["color_limit"], meta["diagnostics"], arrays["joint_signs"],
        )
        result = SessionCCAResult(
            meta["session_id"], meta["time_origin_s"],
            _restored_table(arrays["choices"], meta["table_dtypes"]["choices"]),
            _restored_table(arrays["events"], meta["table_dtypes"]["events"]),
            arrays["choice_positions"], coverage, arrays["bin_edges_s"],
            meta["probe_roles"], fit, None, meta,
        )
        _validate_result(result)
        return result
    except (KeyError, EOFError, BadZipFile, UnpicklingError) as error:
        raise ValueError(f"Malformed CCA result {path}: {error}") from error


def discover_cca_results(
    session_root: Path | str, configuration_id: str | None = None,
) -> tuple[CCAResultInfo, ...]:
    """Return supported metadata-only headers, newest completion/run/path first.

    session_root is a source or copied session directory. Optional identity
    filters stored scientific metadata; no current raw files or score members
    are read. Return a tuple of CCAResultInfo; incomplete/corrupt headers are
    skipped. Ordering uses UTC completion, then run ID and relative path.
    """
    root = Path(session_root)
    headers = []
    for path in (root / "analysis_runs").glob("session_timeline_cca_*/results.npz"):
        try:
            meta = _validated_metadata(_read_arrays(path, ("meta",))["meta"].item())
            if configuration_id is not None and meta["configuration_id"] != configuration_id:
                continue
            headers.append(CCAResultInfo(path, meta, meta["completed_utc"], meta["run_id"]))
        except (OSError, ValueError, KeyError, EOFError, BadZipFile, UnpicklingError):
            continue
    headers.sort(key=_header_order, reverse=True)
    return tuple(headers)


def load_latest_cca_result(
    session_root: Path | str, configuration_id: str,
) -> tuple[CCAResultInfo, SessionCCAResult] | None:
    """Load the newest valid result for one stored identity, warning on fallback.

    Inputs are a session path and scientific configuration label. Return its
    metadata header and source-independent saved record, or None if none load.
    Invalid candidates warn without hiding older compatible completed results.
    """
    for header in discover_cca_results(session_root, configuration_id):
        try:
            return header, load_cca_result(header.path)
        except (OSError, ValueError, EOFError, BadZipFile, UnpicklingError) as error:
            warnings.warn(f"Could not load CCA result {header.path}; trying an older result: {error}",
                          RuntimeWarning, stacklevel=2)
    return None


# Small source/role helpers; numerical spike reads stay in the existing loaders.
def _validated_roles(session: ResolvedSession, probe_roles: dict[str, str]) -> dict[str, str]:
    """Return PFC/HPC probe IDs in stable order, requiring distinct configured strings.

    session contains metadata only; probe_roles is a two-entry role mapping.
    IDs are unitless labels. Invalid/missing/unknown assignments raise ValueError.
    """
    if not isinstance(probe_roles, Mapping) or set(probe_roles) != {"PFC", "HPC"}:
        raise ValueError("CCA requires explicit PFC and HPC probe roles.")
    available = {probe.probe_id for probe in session.probes}
    roles = {region: probe_roles[region] for region in ("PFC", "HPC")}
    if any(not isinstance(probe, str) or probe not in available for probe in roles.values()):
        raise ValueError("CCA probe roles must name configured metadata probes.")
    if len(set(roles.values())) != 2:
        raise ValueError("PFC and HPC must use distinct probes.")
    return roles


def _source_identity(path: Path | None, *, required: bool = True) -> dict:
    """Return path/byte/mtime_ns identity or an explicit optional missing marker.

    path is one source file (or None for an omitted optional source). Returned
    paths are absolute strings, size is bytes, and mtime_ns is filesystem Unix
    nanoseconds. Required missing files raise ValueError; no arrays are read.
    """
    resolved = Path(path).resolve() if path is not None else None
    if resolved is None or not resolved.is_file():
        if required:
            raise ValueError(f"Missing required CCA source file: {resolved}")
        return {"path": str(resolved) if resolved is not None else None, "missing": True}
    stat = resolved.stat()
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _saved_metadata(result: SessionCCAResult) -> dict:
    """Return enriched metadata without modifying the caller's dictionary or arrays.

    result is a structurally validated paired record. The returned dictionary
    records UTC completion, dimensionless identities/ranks, units and axes.
    """
    now = datetime.now(timezone.utc)
    meta = dict(result.meta)
    meta.update({
        "analysis_version": ANALYSIS_VERSION,
        "schema_version": SCHEMA_VERSION,
        "numerical_version": cca.NUMERICAL_VERSION,
        "generator": "src.neural_analysis.population.session_cca_io.save_cca_result",
        "random_seed": None,
        "randomness_used": False,
        "session_id": result.session_id,
        "time_origin_s": result.time_origin_s,
        "probe_roles": dict(result.probe_roles),
        "n_observations": result.fit.n_observations,
        "color_limit": result.fit.color_limit,
        "diagnostics": result.fit.diagnostics,
        "table_dtypes": {
            "choices": {column: str(dtype) for column, dtype in result.choices.dtypes.items()},
            "events": {column: str(dtype) for column, dtype in result.events.dtypes.items()},
        },
        "units": {"absolute_time": "UTC Unix seconds", "relative_time": "seconds",
                  "activity": "spike counts per bin", "scores": "canonical session SD",
                  "correlations": "dimensionless Pearson r"},
        "axes": {"scores": ["choice_window", "time_bin", "canonical_pair"],
                 "coefficients": ["retained_unit", "canonical_pair"]},
        "regions": {},
    })
    meta.setdefault("completed_utc", now.isoformat().replace("+00:00", "Z"))
    meta.setdefault("run_id", now.strftime("%Y%m%d_%H%M%S_%f"))
    for region, transform in result.fit.transforms.items():
        meta["regions"][region] = {
            "input_unit_ids": list(transform.input_unit_ids),
            "retained_unit_ids": list(transform.retained_unit_ids),
            "omitted_unit_ids": list(transform.omitted_unit_ids),
            "rank": transform.rank, "rank_cutoff": transform.rank_cutoff,
        }
    return _validated_metadata(meta)


def _validated_metadata(meta: object) -> dict:
    """Validate a metadata-only header; return its dictionary without source reads.

    meta must contain the supported analysis/schema/numerical versions, UTC
    completion, unitless labels, and declared score/unit-axis conventions.
    Array shape validation belongs to _validate_result after selected loading.
    """
    if not isinstance(meta, dict):
        raise ValueError("CCA meta must be a dictionary.")
    for name, expected in (("analysis_version", ANALYSIS_VERSION), ("schema_version", SCHEMA_VERSION),
                           ("numerical_version", cca.NUMERICAL_VERSION)):
        if meta.get(name) != expected:
            raise ValueError(f"Unsupported CCA {name}: {meta.get(name)!r}")
    for name in ("configuration_id", "session_id", "run_id", "generator"):
        if not isinstance(meta.get(name), str) or not meta[name]:
            raise ValueError(f"CCA metadata needs a nonempty {name}.")
    _completion_time(meta.get("completed_utc"))
    if not isinstance(meta.get("regions"), dict) or set(meta["regions"]) != {"PFC", "HPC"}:
        raise ValueError("CCA metadata must describe both regions.")
    if any(not isinstance(info, dict) for info in meta["regions"].values()):
        raise ValueError("CCA regional metadata must contain transform dictionaries.")
    if (not isinstance(meta.get("table_dtypes"), dict)
            or set(meta["table_dtypes"]) != {"choices", "events"}):
        raise ValueError("CCA metadata must preserve original table dtypes.")
    expected_axes = {"scores": ["choice_window", "time_bin", "canonical_pair"],
                     "coefficients": ["retained_unit", "canonical_pair"]}
    if meta.get("axes") != expected_axes:
        raise ValueError("Unsupported CCA score/coefficient axis conventions.")
    units = meta.get("units", {})
    if (not isinstance(units, dict) or units.get("scores") != "canonical session SD"
            or units.get("absolute_time") != "UTC Unix seconds" or units.get("relative_time") != "seconds"):
        raise ValueError("Unsupported CCA score/time units.")
    return meta


def _validate_result(result: SessionCCAResult) -> None:
    """Check paired-record structure, finite arrays and IDs; return None on success.

    result preserves (choice,bin,pair) score axes, full input-unit preprocessing
    and UTC/elapsed-second tables. No raw input or full covariance matrices are
    loaded/recomputed. Mixed/missing trial IDs are preserved as behavior data.
    """
    fit = result.fit
    if fit is None or result.unavailable_reason is not None:
        raise ValueError("Only valid paired CCA fits may be published or loaded.")
    if not isinstance(result.meta, dict):
        raise ValueError("CCA scientific metadata must be a dictionary.")
    for name, expected in (("analysis_version", ANALYSIS_VERSION), ("schema_version", SCHEMA_VERSION),
                           ("numerical_version", cca.NUMERICAL_VERSION)):
        if name in result.meta and result.meta[name] != expected:
            raise ValueError(f"Unsupported CCA {name}.")
    if not isinstance(result.session_id, str) or not result.session_id:
        raise ValueError("CCA result requires a session identity.")
    _finite_scalar(result.time_origin_s, "time origin")
    _saved_roles(result.probe_roles)
    for field, value in (("session_id", result.session_id), ("probe_roles", result.probe_roles)):
        if field in result.meta and result.meta[field] != value:
            raise ValueError(f"CCA record {field} conflicts with its scientific configuration.")
    choices = result.choices
    required = {"row_position", "trial_id", "choice_time", "elapsed_s", "action"}
    if not isinstance(choices, pd.DataFrame) or not required.issubset(choices) or choices.empty:
        raise ValueError("CCA choices need original row/ID, UTC/elapsed time and action fields.")
    _positions(choices.row_position.to_numpy(), "source row positions")
    for field in ("choice_time", "elapsed_s"):
        values = pd.to_numeric(choices[field], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"CCA choices have invalid {field}.")
    if not np.allclose(choices.choice_time.to_numpy(dtype=float) - result.time_origin_s,
                       choices.elapsed_s.to_numpy(dtype=float), rtol=0, atol=1e-6):
        raise ValueError("CCA choice UTC and elapsed times disagree.")
    if np.any(np.diff(choices.choice_time.to_numpy(dtype=float)) < 0) or not choices.action.isin([0, 1]).all():
        raise ValueError("CCA choices must retain chronological left/right actions.")
    if not isinstance(result.events, pd.DataFrame):
        raise ValueError("CCA events must be a table.")
    if not result.events.empty:
        if not {"Time", "Event", "elapsed_s"}.issubset(result.events):
            raise ValueError("CCA events need Time, Event and elapsed_s columns.")
        absolute = pd.to_numeric(result.events.Time, errors="coerce").to_numpy(dtype=float)
        elapsed = pd.to_numeric(result.events.elapsed_s, errors="coerce").to_numpy(dtype=float)
        if (not np.isfinite(absolute).all() or not np.isfinite(elapsed).all()
                or not np.allclose(absolute - result.time_origin_s, elapsed, rtol=0, atol=1e-6)):
            raise ValueError("CCA event UTC and elapsed clocks must be finite and consistent.")
    common = _positions(result.choice_positions, "common choice positions", len(choices))
    if not len(common) or set(result.coverage_positions) != {"PFC", "HPC"}:
        raise ValueError("CCA requires nonempty common and regional choice axes.")
    for region in ("PFC", "HPC"):
        _positions(result.coverage_positions[region], f"{region} coverage positions", len(choices))
    intersection = np.intersect1d(result.coverage_positions["PFC"], result.coverage_positions["HPC"])
    if not np.array_equal(common, intersection):
        raise ValueError("CCA common positions must equal the regional coverage intersection.")
    try:
        expected_edges = timeline.choice_bin_edges(result.meta["bin_width_s"])
    except KeyError as error:
        raise ValueError("CCA metadata needs its scientific bin width.") from error
    edges = _finite_array(result.bin_edges_s, expected_edges.shape, "bin edges")
    if not np.array_equal(edges, expected_edges):
        raise ValueError("CCA bin edges disagree with the declared choice window/bin width.")
    if set(fit.transforms) != {"PFC", "HPC"} or set(fit.scores) != {"PFC", "HPC"}:
        raise ValueError("CCA fit must contain both regional transforms and score arrays.")
    correlations = np.asarray(fit.correlations)
    if correlations.ndim != 1 or not len(correlations):
        raise ValueError("CCA correlations must be a nonempty pair vector.")
    pairs = len(correlations)
    _finite_array(correlations, (pairs,), "correlations")
    bound = cca.PARAMETERS["correlation_bound_atol"]
    if np.any(correlations < -bound) or np.any(correlations > 1 + bound) or np.any(np.diff(correlations) > bound):
        raise ValueError("CCA correlations must be descending and within [0,1].")
    signs = np.asarray(fit.joint_signs)
    if signs.shape != (pairs,) or not np.isin(signs, [-1, 1]).all():
        raise ValueError("CCA joint signs must label all pairs with +/-1.")
    observations = len(common) * (len(edges) - 1)
    if fit.n_observations != observations or not isinstance(fit.diagnostics, dict):
        raise ValueError("CCA observation count or diagnostics are inconsistent.")
    for region in ("PFC", "HPC"):
        _finite_array(fit.scores[region], (len(common), len(edges) - 1, pairs), f"{region} scores")
        _validate_transform(fit.transforms[region], pairs, observations, result.meta, region)
    if pairs != min(transform.rank for transform in fit.transforms.values()):
        raise ValueError("CCA saved pair count must retain both regions' full available basis.")
    limit = _finite_scalar(fit.color_limit, "common color limit")
    if limit <= 0 or limit != max(float(np.abs(values).max()) for values in fit.scores.values()):
        raise ValueError("CCA common color limit disagrees with the full saved score range.")


def _validate_transform(
    transform: cca.RegionalCCATransform, pairs: int, observations: int, meta: dict, region: str,
) -> None:
    """Validate input/retained-unit maps and float64 arrays; return None.

    transform uses U original units, P nonconstants and K=pairs canonical axes;
    means/SDs are counts, coefficients are canonical SD per standardized unit.
    observations is the paired bin count. No source or sample covariance reads.
    """
    original = _unit_ids(transform.input_unit_ids)
    retained = _unit_ids(transform.retained_unit_ids)
    omitted = _unit_ids(transform.omitted_unit_ids, allow_empty=True)
    if set(retained).intersection(omitted) or set(original) != set(retained).union(omitted):
        raise ValueError(f"{region} retained/omitted IDs do not partition the input units.")
    if retained != tuple(unit for unit in original if unit in retained):
        raise ValueError(f"{region} retained unit order differs from its input order.")
    positions = [original.index(unit) for unit in retained]
    means = _finite_array(transform.count_mean, (len(original),), f"{region} count means")
    scales = _finite_array(transform.count_scale, (len(original),), f"{region} count scales")
    if np.any(means < 0) or np.any(scales < 0) or np.any(scales[positions] <= 0):
        raise ValueError(f"{region} count scaling is invalid.")
    omitted_positions = [original.index(unit) for unit in omitted]
    if np.any(scales[omitted_positions] != 0):
        raise ValueError(f"{region} omitted constant units must have zero scale.")
    _finite_array(transform.coefficients, (len(retained), pairs), f"{region} coefficients")
    _finite_array(transform.score_offset, (pairs,), f"{region} score offset")
    singular = _finite_array(transform.singular_values, (min(observations, len(retained)),),
                             f"{region} singular values")
    cutoff = _finite_scalar(transform.rank_cutoff, f"{region} rank cutoff")
    if (cutoff < 0 or np.any(singular < 0) or np.any(np.diff(singular) > 0)
            or not isinstance(transform.rank, (int, np.integer))
            or transform.rank < 1 or transform.rank != int(np.sum(singular > cutoff))):
        raise ValueError(f"{region} numerical rank diagnostics are inconsistent.")
    if transform.rank > min(observations - 1, len(retained)):
        raise ValueError(f"{region} numerical rank exceeds the centered observation space.")
    selection = meta.get("selection", {}).get(region, {})
    if "unit_ids" in selection and original != tuple(selection["unit_ids"]):
        raise ValueError(f"{region} transform units differ from the scientific selection.")


def _positions(values: np.ndarray, label: str, upper: int | None = None) -> np.ndarray:
    """Return a strictly increasing nonnegative integer index vector, without reshaping.

    values has shape (N,) and no physical units; upper optionally bounds every
    position. Original row positions can be nonconsecutive. Invalid axes fail.
    """
    positions = np.asarray(values)
    if (positions.ndim != 1 or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions < 0) or np.any(positions[1:] <= positions[:-1])
            or (upper is not None and np.any(positions >= upper))):
        raise ValueError(f"Invalid CCA {label}.")
    return positions


def _finite_array(values: np.ndarray, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Return an unchanged real finite float64 array of the specified physical axes.

    shape declares the caller's unit/time/score convention; no coercion, padding
    or reshape repairs malformed saved data. Invalid arrays raise ValueError.
    """
    array = np.asarray(values)
    if array.shape != shape or array.dtype != np.dtype("float64") or not np.isfinite(array).all():
        raise ValueError(f"Invalid CCA {label}: expected finite float64 shape {shape}.")
    return array


def _finite_scalar(value: object, label: str) -> float:
    """Return a finite numeric scalar in its caller-declared units, or raise ValueError."""
    if not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise ValueError(f"Invalid CCA {label}.")
    return float(value)


def _unit_ids(values: object, *, allow_empty: bool = False) -> tuple[str, ...]:
    """Return unique ordered string unit IDs from metadata sequences; IDs are unitless."""
    if not isinstance(values, (tuple, list)):
        raise ValueError("CCA unit IDs must be a string sequence.")
    ids = tuple(values)
    if ((not ids and not allow_empty) or any(not isinstance(unit, str) or not unit for unit in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("CCA unit IDs must be nonempty unique strings.")
    return ids


def _saved_roles(roles: object) -> None:
    """Validate two distinct saved probe labels without consulting current metadata; return None."""
    if (not isinstance(roles, dict) or set(roles) != {"PFC", "HPC"}
            or any(not isinstance(value, str) or not value for value in roles.values())
            or len(set(roles.values())) != 2):
        raise ValueError("CCA saved probe roles must identify two distinct populations.")


def _completion_time(value: object) -> datetime:
    """Parse an aware ISO completion string and return its UTC instant; timestamps have seconds."""
    if not isinstance(value, str):
        raise ValueError("CCA completion time must be an aware ISO string.")
    try:
        completed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Invalid CCA completion time.") from error
    if completed.tzinfo is None:
        raise ValueError("CCA completion time must include a timezone.")
    return completed.astimezone(timezone.utc)


def _header_order(header: CCAResultInfo) -> tuple[datetime, str, str]:
    """Return UTC/run/path ordering; all headers share a root so path order is relative-equivalent."""
    return _completion_time(header.completed_utc), header.run_id, str(header.path)


def _read_arrays(path: Path, members: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
    """Read chosen NPZ members, or all when None, and close the file before returning.

    path names a trusted project archive. Return arrays unchanged in stored
    units/axes; metadata-only callers request ("meta",) and never read scores.
    """
    archive = np.load(path, allow_pickle=True)
    if not hasattr(archive, "files"):
        raise ValueError("CCA result must be an NPZ archive.")
    with archive:
        return {member: archive[member] for member in (archive.files if members is None else members)}


def _restored_table(records: np.ndarray, dtypes: dict[str, str]) -> pd.DataFrame:
    """Rebuild structured table columns with their saved dtype before pandas inference.

    records has one row per original choice/event and named fields; dtypes maps
    those field names to pandas dtype strings. Return a DataFrame preserving
    object missing values and string/numeric conventions, with no unit changes.
    """
    names = records.dtype.names
    if records.ndim != 1 or names is None or not isinstance(dtypes, dict) or set(names) != set(dtypes):
        raise ValueError("CCA table fields and dtype metadata disagree.")
    try:
        return pd.DataFrame({name: pd.Series(records[name], dtype=dtypes[name]) for name in names})
    except (TypeError, ValueError) as error:
        raise ValueError("Unsupported or inconsistent CCA table dtype.") from error
