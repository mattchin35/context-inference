"""Streamlit-cached loading and source-identity seams for neural views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.neural_analysis.lfp import loading as lfp_loading
from src.neural_analysis.lfp.loading import (
    OPEN_EPHYS_AFFINE_UV_SEMANTICS,
    sha256_file_content,
)
from src.neural_analysis.spike_behavior import loading as spike_behavior_pynapple
from src.neural_analysis.spike_behavior import loading as unit_spike_loading
from src.neural_analysis.webapp.session_inputs import (
    LFP_FORMAT_OPEN_EPHYS_DERIVED,
    LFP_FORMAT_SPIKEGLX,
)


@dataclass(frozen=True)
class OpenEphysCacheToken:
    """Hashable selected-source identity for Open Ephys raw-value cache entries.

    Attributes
    ----------
    lfp_path, aligned_sync_path, lfp_preprocessing_path : str
        Resolved filesystem paths for one selected derived LFP binary, its
        aligned synchronization NPZ, and the sibling preprocessing sidecar.
        Paths have no numerical array axis or physical unit.
    lfp_size_bytes, aligned_sync_size_bytes, lfp_preprocessing_size_bytes : int
        Source byte counts at token construction time. Units: bytes.
    lfp_mtime_ns, aligned_sync_mtime_ns, lfp_preprocessing_mtime_ns : int
        Source modification timestamps. Units: nanoseconds.
    lfp_preprocessing_sha256 : str
        Hex digest of sidecar bytes. It has no physical unit.
    source_value_semantics : str
        Code-owned affine-value contract applied by the reader; it preserves
        physical trace units as uV and has no array axis.
    """

    lfp_path: str
    lfp_size_bytes: int
    lfp_mtime_ns: int
    aligned_sync_path: str
    aligned_sync_size_bytes: int
    aligned_sync_mtime_ns: int
    lfp_preprocessing_path: str
    lfp_preprocessing_size_bytes: int
    lfp_preprocessing_mtime_ns: int
    lfp_preprocessing_sha256: str
    source_value_semantics: str = OPEN_EPHYS_AFFINE_UV_SEMANTICS

def build_open_ephys_cache_token(
    lfp_path: Path | str,
    aligned_sync_path: Path | str,
) -> OpenEphysCacheToken:
    """Build one complete cache identity at an Open Ephys source boundary.

    Parameters
    ----------
    lfp_path, aligned_sync_path : pathlib.Path or str
        Selected LFP binary and its aligned synchronization NPZ paths.

    Returns
    -------
    OpenEphysCacheToken
        Frozen identity containing resolved source/sync/sidecar path, byte,
        mtime, digest, and affine-uV semantics fields. Missing inputs raise
        before a cached numerical body can run.

    Raises
    ------
    OSError
        If the selected LFP, aligned sync, or preprocessing sidecar cannot be
        statted or streamed. No numerical LFP samples are loaded.
    """
    lfp_file = Path(lfp_path).resolve()
    sync_file = Path(aligned_sync_path).resolve()
    sidecar = lfp_file.parent / "lfp_preprocessing.json"
    lfp_stat = lfp_file.stat()
    sync_stat = sync_file.stat()
    sidecar_stat = sidecar.stat()
    return OpenEphysCacheToken(
        lfp_path=str(lfp_file),
        lfp_size_bytes=lfp_stat.st_size,
        lfp_mtime_ns=lfp_stat.st_mtime_ns,
        aligned_sync_path=str(sync_file),
        aligned_sync_size_bytes=sync_stat.st_size,
        aligned_sync_mtime_ns=sync_stat.st_mtime_ns,
        lfp_preprocessing_path=str(sidecar.resolve()),
        lfp_preprocessing_size_bytes=sidecar_stat.st_size,
        lfp_preprocessing_mtime_ns=sidecar_stat.st_mtime_ns,
        lfp_preprocessing_sha256=sha256_file_content(sidecar),
    )

def _validate_supplied_open_ephys_cache_token(
    token: OpenEphysCacheToken,
    lfp_path: str,
    aligned_sync_npz_path: str,
) -> OpenEphysCacheToken:
    """Validate that a supplied cache token belongs to the selected source paths.

    Parameters
    ----------
    token : OpenEphysCacheToken
        Frozen token for one physical-uV Open Ephys source. It contains only
        source identity metadata, never sample arrays.
    lfp_path, aligned_sync_npz_path : str
        Requested LFP binary and aligned sync paths. Units: filesystem paths;
        neither input has an array shape or physical value unit.

    Returns
    -------
    OpenEphysCacheToken
        The unchanged supplied instance. It is not rehashed or rebuilt, so a
        caller-selected semantics-only variant remains a distinct cache key.

    Raises
    ------
    ValueError
        If the token is not the expected frozen type or any resolved LFP,
        sync, or sidecar path is bound to a different selected source.
    """
    if not isinstance(token, OpenEphysCacheToken):
        raise ValueError("Open Ephys cache token must be an OpenEphysCacheToken")
    resolved_lfp = str(Path(lfp_path).resolve())
    resolved_sync = str(Path(aligned_sync_npz_path).resolve())
    resolved_sidecar = str((Path(resolved_lfp).parent / "lfp_preprocessing.json").resolve())
    if token.lfp_path != resolved_lfp:
        raise ValueError("Open Ephys cache token is not bound to the requested LFP path")
    if token.aligned_sync_path != resolved_sync:
        raise ValueError("Open Ephys cache token is not bound to the requested aligned sync path")
    if token.lfp_preprocessing_path != resolved_sidecar:
        raise ValueError("Open Ephys cache token is not bound to the requested preprocessing sidecar")
    return token

def _open_ephys_cache_token_or_none(
    lfp_format: str,
    lfp_path: str,
    aligned_sync_npz_path: str | None,
    open_ephys_cache_token: OpenEphysCacheToken | None,
) -> OpenEphysCacheToken | None:
    """Reuse or build one Open Ephys token before an inner cache is keyed.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label selecting SpikeGLX or derived Open Ephys.
    lfp_path : str
        Requested LFP binary path. Units: filesystem path; no array axis.
    aligned_sync_npz_path : str or None
        Requested Open Ephys aligned synchronization path, or ``None`` for
        SpikeGLX. Units: filesystem path.
    open_ephys_cache_token : OpenEphysCacheToken or None
        Optional frozen identity for physical-uV Open Ephys values. It contains
        no sample array and does not alter source trace shape or units.

    Returns
    -------
    OpenEphysCacheToken or None
        A complete token for Open Ephys, or ``None`` for unchanged SpikeGLX
        compatibility. A valid supplied instance is returned unchanged.

    Raises
    ------
    ValueError
        If Open Ephys lacks an aligned sync path or a supplied token is not
        bound to the requested resolved LFP, sync, and sidecar paths.
    OSError
        If an omitted-token Open Ephys request cannot build its source identity.
    """
    if lfp_format != LFP_FORMAT_OPEN_EPHYS_DERIVED:
        return None
    if aligned_sync_npz_path is None or not str(aligned_sync_npz_path).strip():
        raise ValueError("Open Ephys derived LFP requires an aligned sync .npz path.")
    if open_ephys_cache_token is not None:
        return _validate_supplied_open_ephys_cache_token(
            open_ephys_cache_token,
            lfp_path,
            aligned_sync_npz_path,
        )
    return build_open_ephys_cache_token(lfp_path, aligned_sync_npz_path)

def _open_ephys_exploratory_provenance(
    lfp_format: str,
    source_tokens: tuple[OpenEphysCacheToken, ...],
) -> dict[str, object]:
    """Return saved Open Ephys provenance from the exact compute-bound tokens.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label. SpikeGLX selects an empty provenance mapping.
    source_tokens : tuple[OpenEphysCacheToken, ...]
        Complete tokens constructed at the actual selected-source boundary and
        passed to cached computation. They contain no numerical array data;
        their source traces are physical uV by the token semantics contract.

    Returns
    -------
    dict[str, object]
        JSON-compatible semantics copied from the supplied token instances plus
        either one SHA-256 sidecar digest or a resolved-LFP-path keyed digest
        mapping. No sidecar is reopened, read, or hashed by this helper.

    Raises
    ------
    ValueError
        If Open Ephys provenance lacks a compute-bound token, contains an
        invalid token, or combines token instances with different semantics.
    """
    if lfp_format != LFP_FORMAT_OPEN_EPHYS_DERIVED:
        return {}
    if not source_tokens:
        raise ValueError("Open Ephys exploratory provenance requires compute-bound cache tokens")
    if any(not isinstance(token, OpenEphysCacheToken) for token in source_tokens):
        raise ValueError("Open Ephys exploratory provenance requires OpenEphysCacheToken instances")
    unique_tokens = tuple(dict.fromkeys(source_tokens))
    semantics_versions = {token.source_value_semantics for token in unique_tokens}
    if len(semantics_versions) != 1:
        raise ValueError("Open Ephys exploratory provenance cannot mix source value semantics")
    source_value_semantics = next(iter(semantics_versions))
    if not isinstance(source_value_semantics, str) or not source_value_semantics.strip():
        raise ValueError("Open Ephys exploratory provenance requires a nonempty source value semantics")
    digests = {
        token.lfp_path: token.lfp_preprocessing_sha256
        for token in unique_tokens
    }
    provenance: dict[str, object] = {"source_value_semantics": source_value_semantics}
    if len(digests) == 1:
        provenance["lfp_preprocessing_sha256"] = next(iter(digests.values()))
    else:
        provenance["lfp_preprocessing_sha256_by_source"] = digests
    return provenance

@st.cache_resource(show_spinner="Loading session data...")
def load_viewer_data_cached(
    session_data_home: str,
    sess_id_full: str,
    active_probe_label: str,
    sorter_output_path: str | None,
    aligned_spike_path: str | None,
    lfp_path: str | None,
):
    """
    Load one session for Streamlit with cache persistence until app restart.

    Parameters
    ----------
    session_data_home : str
        Root directory for one session.
    sess_id_full : str
        Session id formatted as ``mouse_YYYY-MM-DD_hhmmss``.
    active_probe_label : str
        Active probe label, such as ``"HPC/V1"`` or ``"PFC"``.
    sorter_output_path : str | None
        Active probe sorter output directory path.
    aligned_spike_path : str | None
        Active probe aligned spike ``.npz`` path.
    lfp_path : str | None
        Active probe LFP ``.lf.bin`` path. Not required for spike-only plots.

    Returns
    -------
    dict
        Loaded viewer data from ``unit_spike_loading.load_viewer_data_for_probe``.
    """

    probe_paths = unit_spike_loading.ProbeDataPaths(
        label=active_probe_label,
        sorter_output_path=sorter_output_path,
        aligned_spike_path=aligned_spike_path,
        lfp_path=lfp_path,
    )
    return unit_spike_loading.load_viewer_data_for_probe(
        session_data_home=Path(session_data_home),
        sess_id_full=sess_id_full,
        probe_paths=probe_paths,
    )

@st.cache_resource(show_spinner="Loading behavior data...")
def load_phase_clustering_session_cached(
    session_data_home: str,
    sess_id_full: str,
) -> tuple[spike_behavior_pynapple.Session, pd.DataFrame, pd.DataFrame]:
    """
    Load session metadata and trials without requiring sorted spike data.

    Parameters
    ----------
    session_data_home : str
        Root directory for one experimental session.
    sess_id_full : str
        Session identifier formatted as ``mouse_YYYY-MM-DD_hhmmss``.

    Returns
    -------
    tuple[spike_behavior_pynapple.Session, pd.DataFrame, pd.DataFrame]
        Session metadata, event table with shape ``(n_events, n_event_columns)``,
        and trial table with shape ``(n_trials, n_trial_columns)``. Event and
        trial times are in absolute seconds.
    """

    session_home = Path(session_data_home)
    mouse, date, timestamp = spike_behavior_pynapple.parse_session_id(sess_id_full)
    session = spike_behavior_pynapple.Session(
        session_data_home=session_home,
        sess_id_full=sess_id_full,
        sess_id_abbreviated=f"{mouse}_{date}",
        raw_behavior_folder=session_home / "rpi" / sess_id_full,
        processed_data_path=session_home / "processed",
        figure_path=session_home / "figures",
        mouse=mouse,
        date=date,
        timestamp=timestamp,
    )
    event_df, trial_df = spike_behavior_pynapple.load_session_tables(session)
    return session, event_df, trial_df

@st.cache_resource(show_spinner="Loading metadata-defined behavior data...")
def load_metadata_behavior_session_cached(
    session_root: str,
    subject_id: str,
    session_id: str,
    session_date: str | None,
    behavior_directory: str,
    trial_table_file: str,
    event_table_file: str | None,
) -> tuple[spike_behavior_pynapple.Session, pd.DataFrame, pd.DataFrame]:
    """Load explicit metadata behavior tables for one webapp session.

    Parameters
    ----------
    session_root, behavior_directory, trial_table_file, event_table_file : str or None
        Resolved filesystem paths. CSV time columns retain their existing
        absolute-second conventions; no resampling or unit conversion occurs.
    subject_id, session_id, session_date : str or None
        Display/saved identity labels from metadata.

    Returns
    -------
    tuple[Session, pandas.DataFrame, pandas.DataFrame]
        Session record, event rows, and trial rows. Missing optional event data
        produces an empty table; the required trial table is read exactly once.
    """
    root = Path(session_root)
    date = session_date or "unknown-date"
    session = spike_behavior_pynapple.Session(
        session_data_home=root,
        sess_id_full=session_id,
        sess_id_abbreviated=f"{subject_id}_{date}",
        raw_behavior_folder=Path(behavior_directory),
        processed_data_path=Path(trial_table_file).parent,
        figure_path=root / "figures",
        mouse=subject_id,
        date=date,
        timestamp="",
    )
    trial_df = pd.read_csv(trial_table_file)
    event_df = pd.read_csv(event_table_file) if event_table_file else pd.DataFrame()
    return session, event_df, trial_df

@st.cache_resource(show_spinner="Loading metadata-defined probe data...")
def load_metadata_viewer_data_cached(
    session_root: str,
    subject_id: str,
    session_id: str,
    session_date: str | None,
    behavior_directory: str,
    trial_table_file: str,
    event_table_file: str | None,
    probe_id: str,
    sorter_directory: str,
    aligned_spike_file: str,
    lfp_file: str | None,
) -> dict[str, object]:
    """Load explicit metadata behavior and one selected probe for existing views.

    Sorter cluster assignments and aligned spike times are one-dimensional
    arrays aligned by spike index; times are absolute seconds. The function is
    called only after the user selects a spike-dependent view.
    """
    session, event_df, trial_df = load_metadata_behavior_session_cached(
        session_root,
        subject_id,
        session_id,
        session_date,
        behavior_directory,
        trial_table_file,
        event_table_file,
    )
    sorter_path = Path(sorter_directory)
    aligned_path = Path(aligned_spike_file)
    spike_clusters, cluster_info = spike_behavior_pynapple.load_sorter_metadata(sorter_path)
    aligned_spike_times = spike_behavior_pynapple.load_aligned_spikes(aligned_path)
    spike_behavior_pynapple.validate_aligned_spike_inputs(
        aligned_spike_times,
        spike_clusters,
    )
    spike_group = spike_behavior_pynapple.build_spike_tsgroup(
        spike_times=aligned_spike_times,
        spike_clusters=spike_clusters,
    )
    session.session_info = {
        "active_probe_label": probe_id,
        "aligned_spike_path": aligned_spike_file,
        "lfp_path": lfp_file,
    }
    return {
        "session": session,
        "event_df": event_df,
        "trial_df": trial_df,
        "cluster_info": cluster_info,
        "spike_group": spike_group,
        "aligned_spike_times": aligned_spike_times,
    }

@st.cache_data(show_spinner="Loading channel quality...")
def load_channel_quality_cached(channel_quality_path: str) -> pd.DataFrame:
    """
    Load normalized channel-quality metadata with Streamlit caching.

    Parameters
    ----------
    channel_quality_path : str
        Path to ``channel_quality.csv``, ``channel_quality.json``, or a probe
        directory containing one of those files. Units: filesystem path.

    Returns
    -------
    pd.DataFrame
        Normalized channel-quality dataframe from
        ``unit_spike_loading.load_channel_quality``. Rows are channels; ``ch``
        is a zero-based channel index.
    """

    return unit_spike_loading.load_channel_quality(Path(channel_quality_path))

@st.cache_data(show_spinner="Loading sorter cluster metadata...")
def load_cluster_info_cached(sorter_directory: str) -> pd.DataFrame:
    """Load only one sorter's curation table with Streamlit data caching.

    Parameters
    ----------
    sorter_directory : str
        Explicit ``kilosort4`` output directory. Units: filesystem path; the
        helper opens only ``cluster_info.tsv`` below this directory.

    Returns
    -------
    pd.DataFrame
        Sorter ``cluster_info.tsv`` table with shape
        ``(n_clusters, n_metadata_columns)``. Rows are cluster identities and
        columns retain the sorter-provided categorical/numerical metadata. No
        spike-time, cluster-assignment, aligned-spike, or LFP array is loaded.
    """

    return pd.read_csv(Path(sorter_directory) / "cluster_info.tsv", sep="\t")

def load_summary_cluster_metadata(sorter_path: Path) -> pd.DataFrame:
    """Lazily load curation metadata for one explicit summary population.

    Parameters
    ----------
    sorter_path : pathlib.Path
        Selected ProbeA or ProbeB ``kilosort4`` directory. Units: filesystem
        path. It is passed unchanged to the cluster-info-only cached reader.

    Returns
    -------
    pd.DataFrame
        ``cluster_info.tsv`` metadata with shape
        ``(n_clusters, n_metadata_columns)``. No raw spike or LFP data is
        opened, and no table axes or units are transformed.
    """

    return load_cluster_info_cached(str(sorter_path))

def load_summary_channel_metadata(sorter_path: Path) -> pd.DataFrame:
    """Lazily load normalized channel quality for one explicit summary probe.

    Parameters
    ----------
    sorter_path : pathlib.Path
        Selected ProbeA or ProbeB ``kilosort4`` directory. Units: filesystem
        path. It is used only to infer the containing probe-derived directory.

    Returns
    -------
    pd.DataFrame
        Normalized channel-quality table with shape ``(n_channels, 7)`` and
        zero-based ``ch`` rows; spatial coordinates, when present, are in
        micrometers. The existing cached loader preserves its normalization.

    Raises
    ------
    ValueError
        If the supplied sorter path cannot identify a probe-derived directory.
    """

    probe_directory = unit_spike_loading.infer_probe_derived_dir(sorter_path, None)
    if probe_directory is None:
        raise ValueError(f"Could not infer a probe-derived directory from sorter path: {sorter_path}")
    return load_channel_quality_cached(str(probe_directory))

@st.cache_resource(show_spinner="Decoding LFP sync...")
def decode_lfp_sync_cached(
    lfp_path: str,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
):
    """
    Decode and cache LFP IRIG sync for one file/settings combination.

    Parameters
    ----------
    lfp_path : str
        SpikeGLX ``.lf.bin`` file path.
    digital_word : int
        Digital word index.
    irig_line : int
        IRIG-H digital line. Current IMEC default is line 6.
    bit_period_s : float
        IRIG-H bit period in seconds.
    utc_offset_hours : float
        Constant UTC offset in hours.

    Returns
    -------
    tuple
        ``(lfp_irig_df, sample_rate_hz)`` from ``lfp_loading.decode_lfp_sync``.
    """

    return lfp_loading.decode_lfp_sync(
        lfp_path=Path(lfp_path),
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
    )

@st.cache_data(show_spinner="Loading LFP trace...")
def _load_trial_lfp_trace_cached(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    filter_low_hz: float | None,
    filter_high_hz: float | None,
    filter_padding_s: float,
    aligned_sync_npz_path: str | None = None,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
):
    """
    Load and cache one trial-aligned LFP channel window.

    Parameters
    ----------
    lfp_format : str
        LFP file format. Supported values are ``LFP_FORMAT_SPIKEGLX`` and
        ``LFP_FORMAT_OPEN_EPHYS_DERIVED``.
    lfp_path : str
        LFP binary file path. Units: filesystem path.
    saved_channel_index : int
        Zero-based saved channel index into the binary rows.
    alignment_time_s : float
        Absolute trial alignment time in seconds.
    window_start_s, window_end_s : float
        Relative window bounds in seconds.
    digital_word : int
        Digital word index for sync decoding.
    irig_line : int
        IRIG-H digital line.
    bit_period_s : float
        IRIG-H bit period in seconds.
    utc_offset_hours : float
        Constant UTC offset in hours.
    filter_low_hz, filter_high_hz : float | None
        Optional bandpass cutoff frequencies in Hz. Both values must be
        provided to filter; ``None`` for either value keeps the trace
        unfiltered.
    filter_padding_s : float
        Seconds added to both sides of the requested window before filtering.
    aligned_sync_npz_path : str | None, optional
        Open Ephys probe sync ``.npz`` path. Required for
        ``LFP_FORMAT_OPEN_EPHYS_DERIVED`` and ignored for SpikeGLX.
    open_ephys_cache_token : OpenEphysCacheToken or None, optional
        Complete hashable Open Ephys source identity used only in this inner
        cache key. It has no sample axis or voltage values; the public wrapper
        builds or validates it before this decorated function executes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_values)`` for plotting. Time is in seconds
        relative to alignment.
    """

    del open_ephys_cache_token
    return load_trial_lfp_trace_for_format(
        lfp_format=lfp_format,
        lfp_path=lfp_path,
        saved_channel_index=int(saved_channel_index),
        alignment_time_s=float(alignment_time_s),
        window_start_s=float(window_start_s),
        window_end_s=float(window_end_s),
        digital_word=int(digital_word),
        irig_line=int(irig_line),
        bit_period_s=float(bit_period_s),
        utc_offset_hours=float(utc_offset_hours),
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
        filter_padding_s=float(filter_padding_s),
        aligned_sync_npz_path=aligned_sync_npz_path,
    )

def load_trial_lfp_trace_cached(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    filter_low_hz: float | None,
    filter_high_hz: float | None,
    filter_padding_s: float,
    aligned_sync_npz_path: str | None = None,
    *,
    open_ephys_cache_token: OpenEphysCacheToken | None = None,
):
    """Load one trial-aligned LFP trace through a source-identity cache.

    Parameters
    ----------
    lfp_format : str
        Acquisition-format label. Open Ephys selects the physical-uV affine
        reader; SpikeGLX retains its existing source behavior.
    lfp_path : str
        Selected LFP binary filesystem path with no array axis or physical
        unit.
    aligned_sync_npz_path : str or None
        Open Ephys aligned sync ``.npz`` filesystem path, required for that
        format and unused for SpikeGLX.
    saved_channel_index : int
        Zero-based saved-channel index on the binary channel axis.
    alignment_time_s, window_start_s, window_end_s, bit_period_s : float
        Absolute alignment time and event-relative window bounds in seconds;
        ``bit_period_s`` is the SpikeGLX IRIG period in seconds.
    digital_word, irig_line : int
        SpikeGLX synchronization selectors, ignored for Open Ephys.
    utc_offset_hours : float
        Constant synchronization offset in hours.
    filter_low_hz, filter_high_hz : float or None
        Optional bandpass cutoffs in Hz. Both are required to filter.
    filter_padding_s : float
        Padding on each side of the requested interval in seconds.
    open_ephys_cache_token : OpenEphysCacheToken or None, keyword-only
        Optional complete source identity. A supplied token is path-validated
        and reused unchanged; an omitted token is built before the keyed inner
        cache. It never contains sample values.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(relative_time_s, lfp_values)`` one-dimensional arrays with shape
        ``(sample,)``. Time is seconds relative to alignment. Open Ephys
        values are physical uV after one reader-side affine conversion.

    Raises
    ------
    ValueError
        If format/synchronization/window/filter settings are invalid or an
        Open Ephys token is absent, malformed, or bound to another source.
    OSError
        If the selected production source cannot be read.
    """
    token = _open_ephys_cache_token_or_none(
        lfp_format, lfp_path, aligned_sync_npz_path, open_ephys_cache_token
    )
    return _load_trial_lfp_trace_cached(
        lfp_format, lfp_path, saved_channel_index, alignment_time_s,
        window_start_s, window_end_s, digital_word, irig_line, bit_period_s,
        utc_offset_hours, filter_low_hz, filter_high_hz, filter_padding_s,
        aligned_sync_npz_path, token,
    )

load_trial_lfp_trace_cached.clear = _load_trial_lfp_trace_cached.clear

def load_trial_lfp_trace_for_format(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    filter_low_hz: float | None,
    filter_high_hz: float | None,
    filter_padding_s: float,
    aligned_sync_npz_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load one LFP trace using the selected acquisition format.

    Parameters
    ----------
    lfp_format : str
        LFP file format. Supported values are ``LFP_FORMAT_SPIKEGLX`` and
        ``LFP_FORMAT_OPEN_EPHYS_DERIVED``.
    lfp_path : str
        LFP binary file path. SpikeGLX expects ``.lf.bin`` with a sibling
        ``.meta`` file; Open Ephys expects derived ``lfp.dat`` with a sibling
        ``lfp_preprocessing.json``. Units: filesystem path.
    saved_channel_index : int
        Zero-based saved channel index. Units: channel index.
    alignment_time_s : float
        Absolute trial alignment timestamp in UTC Unix seconds.
    window_start_s, window_end_s : float
        Relative window bounds in seconds around ``alignment_time_s``.
    digital_word : int
        SpikeGLX digital word index. Ignored for Open Ephys derived LFP.
    irig_line : int
        SpikeGLX IRIG-H digital line. Ignored for Open Ephys derived LFP.
    bit_period_s : float
        SpikeGLX IRIG-H bit period in seconds. Ignored for Open Ephys derived
        LFP.
    utc_offset_hours : float
        Constant UTC offset in hours applied to decoded sync anchors.
    filter_low_hz, filter_high_hz : float | None
        Optional bandpass cutoff frequencies in Hz. Both values must be
        provided to filter; ``None`` for either value keeps the trace
        unfiltered.
    filter_padding_s : float
        Seconds added to both sides of the requested window before filtering.
    aligned_sync_npz_path : str | None, optional
        Open Ephys probe sync ``.npz`` path containing IRIG anchors. Required
        for Open Ephys derived LFP and ignored for SpikeGLX.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(relative_time_s, lfp_values)``. Both arrays have shape
        ``(n_samples,)``. Time is in seconds relative to alignment; LFP units
        are physical uV for both formats. Open Ephys values are converted once
        from canonical float32 binary storage before optional filtering.
    """

    frequency_band_hz = (
        (float(filter_low_hz), float(filter_high_hz))
        if filter_low_hz is not None and filter_high_hz is not None
        else None
    )
    if lfp_format == LFP_FORMAT_SPIKEGLX:
        lfp_irig_df, sample_rate_hz = lfp_loading.decode_lfp_sync(
            lfp_path=Path(lfp_path),
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
        )
        return lfp_loading.load_trial_lfp_trace(
            lfp_path=Path(lfp_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=float(sample_rate_hz),
            frequency_band_hz=frequency_band_hz,
            filter_padding_s=float(filter_padding_s),
        )
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        if aligned_sync_npz_path is None or str(aligned_sync_npz_path).strip() == "":
            raise ValueError("Open Ephys derived LFP requires an aligned sync .npz path.")
        return lfp_loading.load_open_ephys_trial_lfp_trace(
            lfp_path=Path(lfp_path),
            aligned_sync_npz_path=Path(aligned_sync_npz_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            utc_offset_hours=float(utc_offset_hours),
            frequency_band_hz=frequency_band_hz,
            filter_padding_s=float(filter_padding_s),
        )
    raise ValueError(f"Unsupported LFP format: {lfp_format!r}")

def load_trial_lfp_trace_for_format_with_sample_rate(
    lfp_format: str,
    lfp_path: str,
    saved_channel_index: int,
    alignment_time_s: float,
    window_start_s: float,
    window_end_s: float,
    digital_word: int,
    irig_line: int,
    bit_period_s: float,
    utc_offset_hours: float,
    aligned_sync_npz_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Load an unfiltered LFP window and sample rate for spectrogram analysis.

    Parameters
    ----------
    lfp_format : str
        LFP format from ``LFP_FORMAT_OPTIONS``.
    lfp_path : str
        SpikeGLX ``.lf.bin`` or derived Open Ephys ``lfp.dat`` path.
    saved_channel_index : int
        Zero-based saved-channel index.
    alignment_time_s : float
        Absolute trial alignment timestamp in UTC Unix seconds.
    window_start_s, window_end_s : float
        Relative LFP bounds in seconds around ``alignment_time_s``.
    digital_word : int
        SpikeGLX sync digital word. Ignored for Open Ephys.
    irig_line : int
        SpikeGLX IRIG line. Ignored for Open Ephys.
    bit_period_s : float
        SpikeGLX IRIG bit period in seconds. Ignored for Open Ephys.
    utc_offset_hours : float
        Constant offset applied to sync timestamps in hours.
    aligned_sync_npz_path : str | None, optional
        Open Ephys aligned sync ``.npz`` path; required for that format.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Relative times and LFP values with shape ``(n_samples,)``, followed by
        sample rate in Hz. Values are physical uV for both formats; Open Ephys
        values are converted once from canonical float32 binary storage.
    """

    if lfp_format == LFP_FORMAT_SPIKEGLX:
        lfp_irig_df, sample_rate_hz = decode_lfp_sync_cached(
            lfp_path=str(lfp_path),
            digital_word=int(digital_word),
            irig_line=int(irig_line),
            bit_period_s=float(bit_period_s),
            utc_offset_hours=float(utc_offset_hours),
        )
        return lfp_loading.load_trial_lfp_trace_with_sample_rate(
            lfp_path=Path(lfp_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            lfp_irig_df=lfp_irig_df,
            sample_rate_hz=float(sample_rate_hz),
        )
    if lfp_format == LFP_FORMAT_OPEN_EPHYS_DERIVED:
        if aligned_sync_npz_path is None or str(aligned_sync_npz_path).strip() == "":
            raise ValueError("Open Ephys derived LFP requires an aligned sync .npz path.")
        metadata = lfp_loading.load_open_ephys_lfp_metadata(Path(lfp_path))
        sample_rate_hz = float(metadata["sampling_frequency_hz"])
        relative_time_s, lfp_values = lfp_loading.load_open_ephys_trial_lfp_trace(
            lfp_path=Path(lfp_path),
            aligned_sync_npz_path=Path(aligned_sync_npz_path),
            saved_channel_index=int(saved_channel_index),
            alignment_time_s=float(alignment_time_s),
            window=(float(window_start_s), float(window_end_s)),
            utc_offset_hours=float(utc_offset_hours),
        )
        return relative_time_s, lfp_values, sample_rate_hz
    raise ValueError(f"Unsupported LFP format: {lfp_format!r}")
