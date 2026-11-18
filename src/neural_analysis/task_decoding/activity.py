"""Load configured neural populations and construct matched trial-rate tensors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.neural_analysis.population import pca as population_pca
from src.neural_analysis.session_metadata import ResolvedProbeSources
from src.neural_analysis.spike_behavior import loading as spike_loading
from src.neural_analysis.task_decoding.config import RegionConfig


@dataclass(frozen=True)
class ProbeCoverage:
    """Trusted aligned-recording bounds for one probe.

    Parameters
    ----------
    source : str
        ``"irig_utc_unix"`` for mechanical IRIG coverage or
        ``"trusted_config"`` for manual trusted bounds.
    start_s, end_s : float
        Inclusive UTC Unix recording bounds in seconds. Manual bounds have
        ``start_s < end_s``; a one-sample IRIG array may produce equal bounds.
    """

    source: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class RegionActivity:
    """Loaded selected units and their trusted coverage for one configured region.

    Parameters
    ----------
    region : str
        Canonical region display token such as ``"PFC"`` or ``"HPC"``.
    probe_id : str
        Explicit metadata probe identifier supplying these units.
    selected_channel_ids : np.ndarray
        Sorted one-dimensional zero-based saved-channel IDs with shape
        ``(n_channels,)``.
    selection_rules : dict[str, object]
        Persisted channel and cluster-selection inputs used for this population.
    cluster_metadata : pd.DataFrame
        Sorted selected sorter rows, one row per cluster.
    unit_metadata : pd.DataFrame
        Unit-axis table with one row per selected unit. It preserves stable
        ``unit_id``, region, probe ID, cluster ID, channel, and normalized
        quality columns in tensor-axis order.
    spike_group : object
        Pynapple ``TsGroup`` keyed by integer selected cluster IDs; timestamps
        are UTC Unix seconds.
    coverage : ProbeCoverage
        Trusted aligned-recording interval in UTC Unix seconds.
    source : ResolvedProbeSources
        Existing resolved source metadata used to load this population.
    """

    region: str
    probe_id: str
    selected_channel_ids: np.ndarray
    selection_rules: dict[str, object]
    cluster_metadata: pd.DataFrame
    unit_metadata: pd.DataFrame
    spike_group: object
    coverage: ProbeCoverage
    source: ResolvedProbeSources


@dataclass(frozen=True)
class SessionRateTensors:
    """Matched PFC and HPC trial-rate tensors for one session.

    Parameters
    ----------
    trial_row_indices : np.ndarray
        One-dimensional original zero-based target-table row positions with
        shape ``(n_trials,)``.
    trial_ids : np.ndarray
        One-dimensional stable trial identities with shape ``(n_trials,)``.
    pfc_rate_tensor_hz, hpc_rate_tensor_hz : np.ndarray
        Float64 arrays with shape ``(n_trials, n_time_bins, n_units)``. Values
        are unsmoothed firing rates in Hz.
    pfc_bin_centers_s, hpc_bin_centers_s : np.ndarray
        One-dimensional event-relative bin centers with shape ``(n_time_bins,)``
        in seconds.
    rate_units : str
        Literal unit label ``"Hz"``.
    pfc_numpy_reference_hz, hpc_numpy_reference_hz : np.ndarray | None
        Optional NumPy-reference tensors with the regional tensor shapes in Hz.
    """

    trial_row_indices: np.ndarray
    trial_ids: np.ndarray
    pfc_rate_tensor_hz: np.ndarray
    hpc_rate_tensor_hz: np.ndarray
    pfc_bin_centers_s: np.ndarray
    hpc_bin_centers_s: np.ndarray
    rate_units: str
    pfc_numpy_reference_hz: np.ndarray | None = None
    hpc_numpy_reference_hz: np.ndarray | None = None


@dataclass(frozen=True)
class ActivityDryRunReport:
    """Pre-binning activity dimensions and source-file sizes without neural arrays.

    Parameters
    ----------
    tensor_allocation_bytes : int
        Exact bytes for simultaneous float64 PFC and HPC rate tensors only.
    source_file_sizes_bytes : dict[Path, int]
        Individual source-file byte sizes. These are reported separately and do
        not contribute to ``tensor_allocation_bytes``.
    trial_row_indices : np.ndarray
        One-dimensional zero-based full-table row positions with bilateral
        full-window coverage, shape ``(tensor_trial,)``.
    tensor_trial_count : int
        Number of target-independent bilateral full-window trial rows.
    time_bin_count : int
        Number of event-relative time bins per trial.
    pfc_unit_count, hpc_unit_count : int
        Stable selected unit counts on the two regional tensor axes.
    """

    tensor_allocation_bytes: int
    source_file_sizes_bytes: dict[Path, int]
    trial_row_indices: np.ndarray
    tensor_trial_count: int
    time_bin_count: int
    pfc_unit_count: int
    hpc_unit_count: int


def _require_source_path(
    path: Path | None,
    description: str,
    *,
    expect_directory: bool,
) -> Path:
    """Validate a required resolved source path.

    Parameters
    ----------
    path : Path or None
        Optional resolved filesystem location.
    description : str
        Human-readable source label used in an error.
    expect_directory : bool
        Whether the declared source must be a directory rather than a file.

    Returns
    -------
    Path
        Existing source path.

    Raises
    ------
    ValueError
        If the metadata path is absent.
    FileNotFoundError
        If the declared path is missing or has the wrong file-system type.
    """
    if path is None:
        raise ValueError(f"Missing required {description} path in probe metadata.")
    normalized_path = Path(path)
    if not normalized_path.exists():
        raise FileNotFoundError(f"Missing {description}: {normalized_path}")
    if expect_directory and not normalized_path.is_dir():
        raise FileNotFoundError(f"Expected {description} directory: {normalized_path}")
    if not expect_directory and not normalized_path.is_file():
        raise FileNotFoundError(f"Expected {description} file: {normalized_path}")
    return normalized_path


def _resolve_probe_source(
    region_config: RegionConfig,
    probe_sources: Sequence[ResolvedProbeSources],
) -> ResolvedProbeSources:
    """Select exactly one explicit configured probe source.

    Parameters
    ----------
    region_config : RegionConfig
        Region rule naming the required explicit ``probe_id``.
    probe_sources : Sequence[ResolvedProbeSources]
        Resolved session sources, one record per available probe.

    Returns
    -------
    ResolvedProbeSources
        The sole source whose probe ID matches ``region_config.probe_id``.

    Raises
    ------
    ValueError
        If the configured probe is missing or appears more than once.
    """
    matches = [source for source in probe_sources if source.probe_id == region_config.probe_id]
    if len(matches) != 1:
        raise ValueError(
            f"Configured probe {region_config.probe_id!r} must match exactly one resolved probe; "
            f"found {len(matches)}."
        )
    return matches[0]


def _selected_region_metadata(
    region_config: RegionConfig,
    probe_sources: Sequence[ResolvedProbeSources],
) -> tuple[ResolvedProbeSources, np.ndarray, dict[str, object], pd.DataFrame]:
    """Resolve channel and cluster metadata without reading neural spike arrays.

    Parameters
    ----------
    region_config : RegionConfig
        Frozen channel and unit-selection settings for one region.
    probe_sources : Sequence[ResolvedProbeSources]
        Available resolved metadata sources.

    Returns
    -------
    tuple
        ``(source, channel_ids, selection_rules, cluster_metadata)`` where
        channel IDs are sorted zero-based integers and metadata is sorted by
        cluster ID.

    Raises
    ------
    ValueError
        If selection yields no channels or units, or selected cluster IDs are
        duplicated.
    """
    source = _resolve_probe_source(region_config, probe_sources)
    channel_quality_path = _require_source_path(
        source.channel_quality_file,
        "channel-quality file",
        expect_directory=False,
    )
    sorter_directory = _require_source_path(
        source.sorter_directory,
        "sorter directory",
        expect_directory=True,
    )
    channel_quality = spike_loading.load_channel_quality(channel_quality_path)
    selected_channel_ids = spike_loading.select_channels_from_quality(
        channel_quality,
        require_inside_brain=region_config.require_inside_brain,
        labels=region_config.channel_labels,
    )
    if source.unit_channels is not None:
        selected_channel_ids = np.intersect1d(
            selected_channel_ids,
            np.asarray(source.unit_channels, dtype=int),
        )
    if region_config.channel_ids:
        selected_channel_ids = np.intersect1d(
            selected_channel_ids,
            np.asarray(region_config.channel_ids, dtype=int),
        )
    selected_channel_ids = np.asarray(selected_channel_ids, dtype=int)
    if selected_channel_ids.size == 0:
        raise ValueError(f"Region {region_config.region} selected zero channels.")

    cluster_info_path = _require_source_path(
        source.cluster_info_file or sorter_directory / "cluster_info.tsv",
        "cluster metadata",
        expect_directory=False,
    )
    cluster_info = pd.read_csv(cluster_info_path, sep="\t")
    cluster_metadata = spike_loading.filter_cluster_metadata(
        cluster_info,
        selected_channel_ids,
        quality_labels=region_config.cluster_groups,
    )
    if cluster_metadata["cluster_id"].duplicated().any():
        raise ValueError("Selected cluster metadata contains duplicate cluster IDs.")
    if cluster_metadata.empty:
        raise ValueError(f"Region {region_config.region} selected zero units.")

    selection_rules: dict[str, object] = {
        "channel_labels": list(region_config.channel_labels),
        "require_inside_brain": region_config.require_inside_brain,
        "metadata_unit_channels": (
            None if source.unit_channels is None else list(source.unit_channels)
        ),
        "config_channel_ids": list(region_config.channel_ids),
        "cluster_groups": list(region_config.cluster_groups),
    }
    return source, selected_channel_ids, selection_rules, cluster_metadata


def _validate_trusted_bounds(trusted_utc_bounds: tuple[float, float] | None) -> tuple[float, float]:
    """Validate manual UTC coverage bounds.

    Parameters
    ----------
    trusted_utc_bounds : tuple[float, float] or None
        Candidate ``(start_s, end_s)`` UTC Unix bounds in seconds.

    Returns
    -------
    tuple[float, float]
        Finite ordered UTC Unix bounds in seconds.

    Raises
    ------
    ValueError
        If bounds are absent, malformed, non-finite, or unordered.
    """
    if trusted_utc_bounds is None or len(trusted_utc_bounds) != 2:
        raise ValueError("Manual alignment coverage requires ordered finite trusted bounds.")
    try:
        start_s = float(trusted_utc_bounds[0])
        end_s = float(trusted_utc_bounds[1])
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Trusted coverage bounds must be finite numeric values.") from error
    if not np.isfinite(start_s) or not np.isfinite(end_s) or start_s >= end_s:
        raise ValueError("Trusted coverage bounds must be finite and have start < end.")
    return start_s, end_s


def inspect_probe_coverage(
    probe_source: ResolvedProbeSources,
    trusted_utc_bounds: tuple[float, float] | None = None,
    *,
    dry_run: bool = False,
) -> ProbeCoverage:
    """Inspect trusted coverage without loading aligned spike timestamps.

    Parameters
    ----------
    probe_source : ResolvedProbeSources
        Existing resolved probe record with an aligned ``.npz`` archive.
    trusted_utc_bounds : tuple[float, float] or None, default=None
        Manual-alignment UTC Unix ``(start_s, end_s)`` bounds in seconds. They
        are required only when no IRIG array is present.
    dry_run : bool, default=False
        Retained explicit validation mode. Both modes inspect NPZ member names;
        neither reads ``spike_utc_unix`` values.

    Returns
    -------
    ProbeCoverage
        Mechanical IRIG or explicit manual coverage bounds in seconds.

    Raises
    ------
    ValueError
        If the spike member is absent, IRIG is malformed, or manual bounds are
        invalid. Configured bounds are forbidden when IRIG is present.
    """
    del dry_run
    alignment_path = _require_source_path(
        probe_source.alignment_file,
        "aligned spike archive",
        expect_directory=False,
    )
    with np.load(alignment_path, allow_pickle=False) as archive:
        if "spike_utc_unix" not in archive.files:
            raise ValueError(f"Aligned archive {alignment_path} is missing spike_utc_unix.")
        if "irig_utc_unix" not in archive.files:
            start_s, end_s = _validate_trusted_bounds(trusted_utc_bounds)
            return ProbeCoverage("trusted_config", start_s, end_s)
        irig_utc_unix = np.asarray(archive["irig_utc_unix"], dtype=float)

    if irig_utc_unix.ndim != 1 or irig_utc_unix.size == 0 or not np.isfinite(irig_utc_unix).all():
        raise ValueError("irig_utc_unix must be a nonempty one-dimensional finite array.")
    if trusted_utc_bounds is not None:
        raise ValueError("Configured trusted bounds are invalid when irig_utc_unix is present.")
    return ProbeCoverage(
        "irig_utc_unix",
        float(np.min(irig_utc_unix)),
        float(np.max(irig_utc_unix)),
    )


def load_region_activity(
    region_config: RegionConfig,
    probe_sources: Sequence[ResolvedProbeSources],
    *,
    trusted_utc_bounds: tuple[float, float] | None = None,
) -> RegionActivity:
    """Load one explicitly configured probe population and its trusted coverage.

    Parameters
    ----------
    region_config : RegionConfig
        Frozen settings for one PFC or HPC region and its explicit probe ID.
    probe_sources : Sequence[ResolvedProbeSources]
        Resolved session probe sources. Exactly one must match the configured ID.
    trusted_utc_bounds : tuple[float, float] or None, default=None
        Manual aligned-recording UTC Unix bounds in seconds for an archive
        without ``irig_utc_unix``.

    Returns
    -------
    RegionActivity
        Selected channels, sorted stable unit metadata, Pynapple spike group,
        and trusted coverage. Unit tensor-axis order follows cluster ID order.

    Raises
    ------
    ValueError
        If required selection, cluster integrity, coverage, or selected-spike
        finite-value contracts are violated.
    """
    source, selected_channel_ids, selection_rules, cluster_metadata = _selected_region_metadata(
        region_config,
        probe_sources,
    )
    sorter_directory = _require_source_path(
        source.sorter_directory,
        "sorter directory",
        expect_directory=True,
    )
    alignment_path = _require_source_path(
        source.alignment_file,
        "aligned spike archive",
        expect_directory=False,
    )
    spike_clusters, _cluster_info = spike_loading.load_sorter_metadata(sorter_directory)
    spike_times_s = spike_loading.load_aligned_spikes(alignment_path)
    spike_loading.validate_aligned_spike_inputs(spike_times_s, spike_clusters)

    cluster_ids = cluster_metadata["cluster_id"].to_numpy(dtype=int)
    assigned_cluster_ids = set(np.asarray(spike_clusters, dtype=int).tolist())
    absent_cluster_ids = [
        cluster_id for cluster_id in cluster_ids if int(cluster_id) not in assigned_cluster_ids
    ]
    if absent_cluster_ids:
        raise ValueError(
            f"selected cluster IDs are absent from spike assignments: {absent_cluster_ids}"
        )
    selected_spike_mask = np.isin(spike_clusters, cluster_ids)
    selected_spike_times_s = np.asarray(spike_times_s[selected_spike_mask], dtype=float)
    selected_spike_clusters = np.asarray(spike_clusters[selected_spike_mask], dtype=int)
    if not np.isfinite(selected_spike_times_s).all():
        raise ValueError("Selected aligned spike timestamps must be finite.")
    spike_group = spike_loading.build_spike_tsgroup(
        selected_spike_times_s,
        selected_spike_clusters,
        cluster_ids=cluster_ids,
    )
    unit_metadata = pd.DataFrame(
        {
            "unit_id": [f"{source.probe_id}:{cluster_id}" for cluster_id in cluster_ids],
            "region": region_config.region,
            "probe_id": source.probe_id,
            "cluster_id": cluster_ids,
            "channel": cluster_metadata["ch"].to_numpy(dtype=int),
            "quality_label": cluster_metadata["quality_label"].to_numpy(dtype=str),
        }
    )
    coverage = inspect_probe_coverage(source, trusted_utc_bounds)
    return RegionActivity(
        region=region_config.region,
        probe_id=source.probe_id,
        selected_channel_ids=selected_channel_ids,
        selection_rules=selection_rules,
        cluster_metadata=cluster_metadata,
        unit_metadata=unit_metadata,
        spike_group=spike_group,
        coverage=coverage,
        source=source,
    )


def _working_trial_table(target_table: pd.DataFrame) -> pd.DataFrame:
    """Copy one target table onto a positional RangeIndex for tensor builders.

    Parameters
    ----------
    target_table : pd.DataFrame
        Chronological target table with one row per original trial.

    Returns
    -------
    pd.DataFrame
        Copy with ``RangeIndex(0, n_trials)`` and unchanged columns/values.

    Raises
    ------
    ValueError
        If baseline or original-row identity columns are missing.
    """
    required_columns = {"baseline_valid", "row_position"}
    missing_columns = required_columns - set(target_table.columns)
    if missing_columns:
        raise ValueError(f"Target table is missing required columns: {sorted(missing_columns)}")
    working_table = target_table.reset_index(drop=True).copy()
    row_positions = pd.to_numeric(working_table["row_position"], errors="coerce").to_numpy(
        dtype=float,
    )
    expected_positions = np.arange(len(working_table), dtype=float)
    if (
        not np.isfinite(row_positions).all()
        or not np.array_equal(row_positions, expected_positions)
    ):
        raise ValueError("row_position must be the zero-based position of every input row.")
    return working_table


def _common_trial_positions(
    working_table: pd.DataFrame,
    pfc_coverage: ProbeCoverage,
    hpc_coverage: ProbeCoverage,
    alignment_event: str,
    window_s: tuple[float, float],
) -> np.ndarray:
    """Find baseline-valid rows with finite bilateral full-window coverage.

    Parameters
    ----------
    working_table : pd.DataFrame
        Range-indexed chronological target table.
    pfc_coverage, hpc_coverage : ProbeCoverage
        Trusted UTC Unix bounds in seconds for both required probes.
    alignment_event : str
        Target-table timestamp column in UTC Unix seconds.
    window_s : tuple[float, float]
        Event-relative ``(start_s, end_s)`` coverage window in seconds.

    Returns
    -------
    np.ndarray
        Ascending RangeIndex positions with shape ``(n_eligible_trials,)``.
    """
    if alignment_event not in working_table.columns:
        raise ValueError(f"Target table is missing alignment column {alignment_event!r}.")
    if len(window_s) != 2 or not np.isfinite(window_s).all() or window_s[0] >= window_s[1]:
        raise ValueError("window_s must contain finite start < end seconds.")
    alignment_values = pd.to_numeric(
        working_table[alignment_event],
        errors="coerce",
    ).to_numpy(dtype=float)
    baseline_valid = working_table["baseline_valid"].to_numpy(dtype=bool)
    finite_alignment = np.isfinite(alignment_values)
    start_values = alignment_values + float(window_s[0])
    end_values = alignment_values + float(window_s[1])
    coverage_mask = (
        (start_values >= pfc_coverage.start_s)
        & (end_values <= pfc_coverage.end_s)
        & (start_values >= hpc_coverage.start_s)
        & (end_values <= hpc_coverage.end_s)
    )
    return np.flatnonzero(baseline_valid & finite_alignment & coverage_mask).astype(int)


def build_session_rate_tensors(
    target_table: pd.DataFrame,
    pfc_activity: RegionActivity,
    hpc_activity: RegionActivity,
    *,
    alignment_event: str,
    window_s: tuple[float, float],
    bin_width_s: float,
    verify_numpy_reference: bool = False,
) -> SessionRateTensors:
    """Build common PFC and HPC event-aligned firing-rate tensors.

    Parameters
    ----------
    target_table : pd.DataFrame
        Chronological target table with one row per original trial, a Boolean
        ``baseline_valid`` column, zero-based ``row_position``, one stable
        ``trial_id`` or ``cur_trial`` identity, and the named alignment column
        in UTC Unix seconds.
    pfc_activity, hpc_activity : RegionActivity
        Fully loaded required regional populations with trusted coverage.
    alignment_event : str
        Timestamp column used as time zero.
    window_s : tuple[float, float]
        Event-relative half-open binning window in seconds.
    bin_width_s : float
        Positive fixed bin width in seconds; output values are Hz.
    verify_numpy_reference : bool, default=False
        If true, additionally build NumPy-reference tensors for equality tests.

    Returns
    -------
    SessionRateTensors
        Common-row regional float64 tensors with shape
        ``(n_eligible_trials, n_time_bins, n_region_units)`` in Hz.

    Raises
    ------
    ValueError
        If no row is jointly eligible, identities are missing, or regional axes
        disagree.
    """
    working_table = _working_trial_table(target_table)
    trial_positions = _common_trial_positions(
        working_table,
        pfc_activity.coverage,
        hpc_activity.coverage,
        alignment_event,
        window_s,
    )
    if trial_positions.size == 0:
        raise ValueError("No baseline-valid trials have bilateral full-window coverage.")
    if "trial_id" in working_table.columns:
        trial_ids = working_table.loc[trial_positions, "trial_id"].to_numpy(copy=True)
    elif "cur_trial" in working_table.columns:
        trial_ids = working_table.loc[trial_positions, "cur_trial"].to_numpy(copy=True)
    else:
        raise ValueError("Target table requires trial_id or cur_trial for tensor-row identity.")

    pfc_cluster_ids = pfc_activity.unit_metadata["cluster_id"].to_numpy(dtype=int)
    hpc_cluster_ids = hpc_activity.unit_metadata["cluster_id"].to_numpy(dtype=int)
    pfc_tensor_hz, pfc_centers_s = population_pca.build_trial_unit_rate_tensor(
        pfc_activity.spike_group,
        pfc_cluster_ids,
        working_table,
        trial_positions,
        alignment_event,
        window_s,
        bin_width_s,
    )
    hpc_tensor_hz, hpc_centers_s = population_pca.build_trial_unit_rate_tensor(
        hpc_activity.spike_group,
        hpc_cluster_ids,
        working_table,
        trial_positions,
        alignment_event,
        window_s,
        bin_width_s,
    )
    if not np.array_equal(pfc_centers_s, hpc_centers_s):
        raise ValueError("PFC and HPC bin-center axes differ.")
    pfc_reference_hz = None
    hpc_reference_hz = None
    if verify_numpy_reference:
        pfc_reference_hz, pfc_reference_centers_s = (
            population_pca.build_trial_unit_rate_tensor_numpy(
                pfc_activity.spike_group,
                pfc_cluster_ids,
                working_table,
                trial_positions,
                alignment_event,
                window_s,
                bin_width_s,
            )
        )
        hpc_reference_hz, hpc_reference_centers_s = (
            population_pca.build_trial_unit_rate_tensor_numpy(
                hpc_activity.spike_group,
                hpc_cluster_ids,
                working_table,
                trial_positions,
                alignment_event,
                window_s,
                bin_width_s,
            )
        )
        if not np.array_equal(pfc_centers_s, pfc_reference_centers_s) or not np.array_equal(
            hpc_centers_s,
            hpc_reference_centers_s,
        ):
            raise ValueError("Pynapple and NumPy bin-center axes differ.")
    return SessionRateTensors(
        trial_row_indices=working_table.loc[trial_positions, "row_position"].to_numpy(dtype=int),
        trial_ids=trial_ids,
        pfc_rate_tensor_hz=np.asarray(pfc_tensor_hz, dtype=np.float64),
        hpc_rate_tensor_hz=np.asarray(hpc_tensor_hz, dtype=np.float64),
        pfc_bin_centers_s=np.asarray(pfc_centers_s, dtype=float),
        hpc_bin_centers_s=np.asarray(hpc_centers_s, dtype=float),
        rate_units="Hz",
        pfc_numpy_reference_hz=(
            None if pfc_reference_hz is None else np.asarray(pfc_reference_hz, dtype=np.float64)
        ),
        hpc_numpy_reference_hz=(
            None if hpc_reference_hz is None else np.asarray(hpc_reference_hz, dtype=np.float64)
        ),
    )


def project_target_mask_to_tensor_rows(
    trial_row_indices: np.ndarray,
    target_mask: np.ndarray,
) -> np.ndarray:
    """Project one original-row target mask onto common tensor-row order.

    Parameters
    ----------
    trial_row_indices : np.ndarray
        One-dimensional zero-based original target-table positions with shape
        ``(n_tensor_trials,)``.
    target_mask : np.ndarray
        One-dimensional Boolean-like original-row mask with shape
        ``(n_original_trials,)``.

    Returns
    -------
    np.ndarray
        Boolean mask with shape ``(n_tensor_trials,)`` in tensor-row order.

    Raises
    ------
    ValueError
        If inputs are not one-dimensional or positions are out of range.
    """
    normalized_indices = np.asarray(trial_row_indices, dtype=int)
    normalized_mask = np.asarray(target_mask, dtype=bool)
    if normalized_indices.ndim != 1 or normalized_mask.ndim != 1:
        raise ValueError("trial_row_indices and target_mask must be one-dimensional.")
    if (normalized_indices < 0).any() or (normalized_indices >= normalized_mask.size).any():
        raise ValueError("trial_row_indices contain positions outside target_mask.")
    return normalized_mask[normalized_indices]


def estimate_rate_tensor_bytes(
    *,
    n_tensor_trials: int,
    n_time_bins: int,
    n_pfc_units: int,
    n_hpc_units: int,
) -> int:
    """Compute exact allocation bytes for simultaneous regional float64 tensors.

    Parameters
    ----------
    n_tensor_trials, n_time_bins, n_pfc_units, n_hpc_units : int
        Nonnegative tensor dimensions. The regional unit counts remain
        separate, then are summed only for the two tensor allocations.

    Returns
    -------
    int
        ``n_trials * n_bins * (n_pfc_units + n_hpc_units) * 8`` bytes.

    Raises
    ------
    ValueError
        If any dimension is negative or non-integral.
    """
    dimensions = tuple(
        _require_integer(value, "Tensor dimension", minimum=0)
        for value in (n_tensor_trials, n_time_bins, n_pfc_units, n_hpc_units)
    )
    return dimensions[0] * dimensions[1] * (dimensions[2] + dimensions[3]) * 8


def _require_integer(value: object, description: str, *, minimum: int) -> int:
    """Validate one integer scalar without leaking conversion exceptions.

    Parameters
    ----------
    value : object
        Candidate scalar value to validate.
    description : str
        Human-readable value name used in validation errors.
    minimum : int
        Inclusive lower bound for the validated value.

    Returns
    -------
    int
        Validated integer scalar.

    Raises
    ------
    ValueError
        If the value is boolean, malformed, non-integral, or below ``minimum``.
    """
    if isinstance(value, bool):
        raise ValueError(f"{description} must be an integer at least {minimum}.")
    try:
        normalized_value = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{description} must be an integer at least {minimum}.") from error
    try:
        is_exact_integer = bool(normalized_value == value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be an integer at least {minimum}.") from error
    if not is_exact_integer or normalized_value < minimum:
        raise ValueError(f"{description} must be an integer at least {minimum}.")
    return normalized_value


def select_memory_budget_bytes(
    *,
    local_mem_available_bytes: int | None = None,
    slurm_memory_limit_bytes: int | None = None,
    login_requested_memory_bytes: int | None = None,
) -> int:
    """Select the applicable conservative memory budget.

    Parameters
    ----------
    local_mem_available_bytes : int or None, default=None
        Locally observed available memory in bytes.
    slurm_memory_limit_bytes : int or None, default=None
        Active Slurm allocation or cgroup memory limit in bytes.
    login_requested_memory_bytes : int or None, default=None
        Approved login-side requested memory limit in bytes.

    Returns
    -------
    int
        Smallest supplied positive applicable budget in bytes.

    Raises
    ------
    ValueError
        If every budget is unknown or any supplied value is nonpositive.
    """
    supplied_budgets = [
        budget
        for budget in (
            local_mem_available_bytes,
            slurm_memory_limit_bytes,
            login_requested_memory_bytes,
        )
        if budget is not None
    ]
    if not supplied_budgets:
        raise ValueError("No applicable memory budget is available.")
    validated_budgets = [
        _require_integer(budget, "Memory budget", minimum=1)
        for budget in supplied_budgets
    ]
    return min(validated_budgets)


def validate_tensor_memory_budget(*, tensor_bytes: int, memory_budget_bytes: int) -> None:
    """Enforce the pre-benchmark 50-percent exact-tensor allocation guard.

    Parameters
    ----------
    tensor_bytes : int
        Exact regional float64 tensor allocation in bytes.
    memory_budget_bytes : int
        Applicable available or approved memory budget in bytes.

    Returns
    -------
    None
        Returns normally only when tensors occupy at most half the budget.

    Raises
    ------
    ValueError
        If values are invalid or tensor allocation exceeds 50 percent.
    """
    normalized_tensor_bytes = _require_integer(tensor_bytes, "Tensor bytes", minimum=0)
    normalized_memory_budget = _require_integer(memory_budget_bytes, "Memory budget", minimum=1)
    if normalized_tensor_bytes * 2 > normalized_memory_budget:
        raise ValueError("Tensor allocation exceeds the 50% memory-budget guard.")


def _n_time_bins(window_s: tuple[float, float], bin_width_s: float) -> int:
    """Validate complete fixed-width binning and return the number of bins.

    Parameters
    ----------
    window_s : tuple[float, float]
        Event-relative ``(start_s, end_s)`` window in seconds.
    bin_width_s : float
        Positive bin width in seconds.

    Returns
    -------
    int
        Number of complete bins spanning the window.
    """
    try:
        if len(window_s) != 2:
            raise ValueError
        window_start = float(window_s[0])
        window_end = float(window_s[1])
        normalized_bin_width_s = float(bin_width_s)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("window_s and bin_width_s must be finite numeric seconds.") from error
    if not np.isfinite(window_start) or not np.isfinite(window_end) or window_start >= window_end:
        raise ValueError("window_s must contain finite start < end seconds.")
    if not np.isfinite(normalized_bin_width_s) or normalized_bin_width_s <= 0:
        raise ValueError("bin_width_s must be a positive finite value.")
    n_bins = int(np.floor((window_end - window_start) / normalized_bin_width_s + 1e-12))
    if n_bins < 1:
        raise ValueError("window_s must contain at least one complete bin.")
    return n_bins


def _source_paths(source: ResolvedProbeSources) -> tuple[Path, ...]:
    """Return required reportable source files for one resolved probe.

    Parameters
    ----------
    source : ResolvedProbeSources
        One resolved probe source record.

    Returns
    -------
    tuple[Path, ...]
        Existing aligned archive, sorter cluster array, cluster metadata, and
        channel-quality metadata files.
    """
    sorter_directory = _require_source_path(
        source.sorter_directory,
        "sorter directory",
        expect_directory=True,
    )
    return (
        _require_source_path(
            source.alignment_file,
            "aligned spike archive",
            expect_directory=False,
        ),
        _require_source_path(
            source.spike_clusters_file or sorter_directory / "spike_clusters.npy",
            "spike clusters",
            expect_directory=False,
        ),
        _require_source_path(
            source.cluster_info_file or sorter_directory / "cluster_info.tsv",
            "cluster metadata",
            expect_directory=False,
        ),
        _require_source_path(
            source.channel_quality_file,
            "channel-quality file",
            expect_directory=False,
        ),
    )


def inspect_activity_dry_run(
    target_table: pd.DataFrame,
    pfc_region_config: RegionConfig,
    hpc_region_config: RegionConfig,
    probe_sources: Sequence[ResolvedProbeSources],
    *,
    alignment_event: str,
    window_s: tuple[float, float],
    bin_width_s: float,
    pfc_trusted_utc_bounds: tuple[float, float] | None = None,
    hpc_trusted_utc_bounds: tuple[float, float] | None = None,
) -> ActivityDryRunReport:
    """Inspect metadata, IRIG coverage, dimensions, and source sizes without spikes.

    Parameters
    ----------
    target_table : pd.DataFrame
        Chronological target table with baseline/row identity/alignment columns.
    pfc_region_config, hpc_region_config : RegionConfig
        Explicit PFC and HPC selection settings.
    probe_sources : Sequence[ResolvedProbeSources]
        Existing resolved session probe sources.
    alignment_event : str
        UTC Unix target-table timestamp column used as time zero.
    window_s : tuple[float, float]
        Event-relative coverage/binning window in seconds.
    bin_width_s : float
        Fixed bin width in seconds.
    pfc_trusted_utc_bounds, hpc_trusted_utc_bounds : tuple[float, float] or None
        Manual trusted bounds used only for the matching no-IRIG probe.

    Returns
    -------
    ActivityDryRunReport
        Exact tensor dimensions/allocation bytes and separate source-file byte
        sizes. Trial and unit counts are dimensionless; allocation and source
        sizes are bytes.

    Raises
    ------
    ValueError
        If metadata selection, coverage, or joint eligibility is invalid.
    """
    pfc_source, _pfc_channels, _pfc_rules, pfc_clusters = _selected_region_metadata(
        pfc_region_config,
        probe_sources,
    )
    hpc_source, _hpc_channels, _hpc_rules, hpc_clusters = _selected_region_metadata(
        hpc_region_config,
        probe_sources,
    )
    pfc_coverage = inspect_probe_coverage(
        pfc_source,
        pfc_trusted_utc_bounds,
        dry_run=True,
    )
    hpc_coverage = inspect_probe_coverage(
        hpc_source,
        hpc_trusted_utc_bounds,
        dry_run=True,
    )
    working_table = _working_trial_table(target_table)
    trial_positions = _common_trial_positions(
        working_table,
        pfc_coverage,
        hpc_coverage,
        alignment_event,
        window_s,
    )
    if trial_positions.size == 0:
        raise ValueError("No baseline-valid trials have bilateral full-window coverage.")
    time_bin_count = _n_time_bins(window_s, bin_width_s)
    tensor_allocation_bytes = estimate_rate_tensor_bytes(
        n_tensor_trials=trial_positions.size,
        n_time_bins=time_bin_count,
        n_pfc_units=len(pfc_clusters),
        n_hpc_units=len(hpc_clusters),
    )
    source_paths = _source_paths(pfc_source) + _source_paths(hpc_source)
    source_file_sizes_bytes = {path: path.stat().st_size for path in source_paths}
    return ActivityDryRunReport(
        tensor_allocation_bytes=tensor_allocation_bytes,
        source_file_sizes_bytes=source_file_sizes_bytes,
        trial_row_indices=np.array(trial_positions, dtype=np.int64, copy=True),
        tensor_trial_count=int(trial_positions.size),
        time_bin_count=time_bin_count,
        pfc_unit_count=int(len(pfc_clusters)),
        hpc_unit_count=int(len(hpc_clusters)),
    )
