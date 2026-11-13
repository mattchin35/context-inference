"""Paired choice-window CCA inputs and metadata-only scientific configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

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
