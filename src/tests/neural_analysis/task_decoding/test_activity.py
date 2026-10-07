"""Synthetic activity-loading, coverage, and rate-tensor contract tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis.task_decoding import activity
from src.neural_analysis.task_decoding.config import RegionConfig
from src.neural_analysis.session_metadata import ResolvedProbeSources


def make_region_config(
    region: str,
    probe_id: str,
    *,
    channel_ids: tuple[int, ...] = (),
) -> RegionConfig:
    """Build one immutable synthetic PFC/HPC region selection.

    Parameters
    ----------
    region : str
        Canonical ``"PFC"`` or ``"HPC"`` display token.
    probe_id : str
        Explicit metadata probe identifier.
    channel_ids : tuple[int, ...], default=()
        Additional zero-based channel restriction.

    Returns
    -------
    RegionConfig
        Region rule with good/inside-brain channels and good/MUA units.
    """
    return RegionConfig(
        region=region,
        probe_id=probe_id,
        channel_labels=("good",),
        require_inside_brain=True,
        cluster_groups=("good", "mua"),
        channel_ids=channel_ids,
    )


def write_probe_fixture(
    tmp_path: Path,
    probe_id: str,
    *,
    include_irig: bool = True,
    spike_times_s: np.ndarray | None = None,
    spike_clusters: np.ndarray | None = None,
    unit_channels: tuple[int, ...] | None = None,
) -> ResolvedProbeSources:
    """Create one small sorter/channel/alignment probe fixture.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory containing synthetic session inputs.
    probe_id : str
        Explicit probe identifier retained in output stable-unit IDs.
    include_irig : bool, default=True
        Whether the aligned archive includes finite ``irig_utc_unix`` coverage.
    spike_times_s : np.ndarray or None
        Optional one-dimensional aligned timestamps in UTC seconds. Default is
        four finite spikes spanning 8.5--16.5 seconds.
    spike_clusters : np.ndarray or None
        Optional one-dimensional sorter cluster IDs aligned to spike times.
    unit_channels : tuple[int, ...] or None
        Optional probe-metadata channel restriction.

    Returns
    -------
    ResolvedProbeSources
        Existing metadata source record identifying this fixture's sorter
        directory, aligned archive, and channel-quality file.
    """
    probe_path = tmp_path / probe_id
    sorter_path = probe_path / "kilosort4"
    sorter_path.mkdir(parents=True)
    channel_quality_path = probe_path / "channel_quality.csv"
    pd.DataFrame(
        {
            "channel_id": [0, 1, 2],
            "label": ["good", "good", "good"],
            "inside_brain": [True, True, False],
            "x_um": [0.0, 20.0, 40.0],
            "y_um": [0.0, 0.0, 0.0],
        }
    ).to_csv(channel_quality_path, index=False)
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [11, 5, 17, 23],
            "ch": [1, 0, 2, 1],
            "group": ["mua", "good", "good", "noise"],
        }
    )
    cluster_info.to_csv(sorter_path / "cluster_info.tsv", sep="\t", index=False)
    if spike_times_s is None:
        spike_times_s = np.array([8.5, 10.2, 12.6, 16.5], dtype=float)
    if spike_clusters is None:
        spike_clusters = np.array([5, 11, 5, 11], dtype=int)
    np.save(sorter_path / "spike_clusters.npy", spike_clusters)
    aligned_path = probe_path / "aligned_spikes.npz"
    aligned_arrays = {"spike_utc_unix": np.asarray(spike_times_s, dtype=float)}
    if include_irig:
        aligned_arrays["irig_utc_unix"] = np.array([8.0, 17.0], dtype=float)
    np.savez(aligned_path, **aligned_arrays)
    return ResolvedProbeSources(
        probe_id=probe_id,
        acquisition_family="synthetic",
        lfp_file=None,
        lfp_metadata_file=None,
        alignment_file=aligned_path,
        sorter_directory=sorter_path,
        channel_quality_file=channel_quality_path,
        spike_times_file=None,
        spike_clusters_file=sorter_path / "spike_clusters.npy",
        cluster_info_file=sorter_path / "cluster_info.tsv",
        sites=(),
        unit_channels=unit_channels,
    )


def make_target_table() -> pd.DataFrame:
    """Build a non-default-index full target table for tensor-row mapping tests.

    Returns
    -------
    pd.DataFrame
        Six chronological rows with zero-based ``row_position``, stable
        ``trial_id``, baseline masks/reasons, and choice timestamps in seconds.
    """
    return pd.DataFrame(
        {
            "row_position": [0, 1, 2, 3, 4, 5],
            "trial_id": [0, 1, 2, 3, 4, 5],
            "baseline_valid": [True, False, True, False, True, True],
            "baseline_invalid_reason": ["", "manual_reward", "", "no_animal_choice", "", ""],
            "choice_time": [10.0, 11.0, 12.0, 13.0, 14.0, np.nan],
            "current_action_valid": [True, False, True, False, True, True],
            "relative_doubt_valid": [True, True, False, True, True, True],
        },
        index=pd.Index([101, 103, 107, 109, 113, 127], name="source_index"),
    )


def test_load_region_activity_uses_explicit_probe_and_intersects_all_channel_rules(tmp_path):
    """Explicit probing should intersect quality, brain, metadata, and config rules."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc", unit_channels=(0, 1))
    unrelated_inputs = write_probe_fixture(tmp_path, "probe-hpc")
    region_config = make_region_config("PFC", "probe-pfc", channel_ids=(1,))

    result = activity.load_region_activity(region_config, (unrelated_inputs, probe_inputs))

    assert result.region == "PFC"
    assert result.probe_id == "probe-pfc"
    np.testing.assert_array_equal(result.selected_channel_ids, np.array([1]))
    assert result.selection_rules == {
        "channel_labels": ["good"],
        "require_inside_brain": True,
        "metadata_unit_channels": [0, 1],
        "config_channel_ids": [1],
        "cluster_groups": ["good", "mua"],
    }
    with pytest.raises(ValueError, match="probe|unknown|missing"):
        activity.load_region_activity(
            make_region_config("PFC", "unconfigured-probe"),
            (unrelated_inputs, probe_inputs),
        )


def test_load_region_activity_applies_channel_quality_labels_independently(tmp_path):
    """A nonmatching channel label should remove that channel without other restrictions."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    channel_quality = pd.read_csv(probe_inputs.channel_quality_file)
    channel_quality.loc[channel_quality["channel_id"] == 0, "label"] = "noise"
    channel_quality.to_csv(probe_inputs.channel_quality_file, index=False)

    result = activity.load_region_activity(
        make_region_config("PFC", "probe-pfc"),
        (probe_inputs,),
    )

    np.testing.assert_array_equal(result.selected_channel_ids, np.array([1]))


def test_load_region_activity_applies_inside_brain_requirement_independently(tmp_path):
    """An outside-brain channel should be removed even when its quality label is good."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    channel_quality = pd.read_csv(probe_inputs.channel_quality_file)
    channel_quality.loc[channel_quality["channel_id"] == 0, "inside_brain"] = False
    channel_quality.to_csv(probe_inputs.channel_quality_file, index=False)

    result = activity.load_region_activity(
        make_region_config("PFC", "probe-pfc"),
        (probe_inputs,),
    )

    np.testing.assert_array_equal(result.selected_channel_ids, np.array([1]))


def test_load_region_activity_reuses_cluster_filter_with_configured_groups_and_stable_order(
    tmp_path,
    monkeypatch,
):
    """Unit selection should call the existing cluster filter and retain its sorted metadata."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    region_config = make_region_config("PFC", "probe-pfc")
    calls: list[tuple[np.ndarray, tuple[str, ...]]] = []
    original_filter = activity.spike_loading.filter_cluster_metadata

    def record_filter(cluster_info, region_channels, quality_labels, **kwargs):
        """Record arguments while delegating to the project cluster filter."""
        calls.append((np.asarray(region_channels), tuple(quality_labels)))
        return original_filter(cluster_info, region_channels, quality_labels, **kwargs)

    monkeypatch.setattr(activity.spike_loading, "filter_cluster_metadata", record_filter)

    result = activity.load_region_activity(region_config, (probe_inputs,))

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0], np.array([0, 1]))
    assert calls[0][1] == ("good", "mua")
    assert result.cluster_metadata["cluster_id"].tolist() == [5, 11]
    assert result.cluster_metadata["quality_label"].tolist() == ["good", "mua"]
    assert result.unit_metadata["unit_id"].tolist() == ["probe-pfc:5", "probe-pfc:11"]
    assert result.unit_metadata["cluster_id"].tolist() == [5, 11]
    assert result.unit_metadata["channel"].tolist() == [0, 1]


@pytest.mark.parametrize(
    ("spike_times_s", "spike_clusters", "error_match"),
    [
        (np.array([8.5, 10.2]), np.array([5]), "same length"),
        (np.array([8.5, np.nan]), np.array([5, 11]), "finite"),
        (np.array([8.5, 10.2]), np.array([5, 99]), "selected cluster"),
    ],
)
def test_load_region_activity_rejects_spike_cluster_integrity_errors(
    tmp_path,
    spike_times_s,
    spike_clusters,
    error_match,
):
    """Selected spike arrays must align, be finite, and reference selected clusters only."""
    probe_inputs = write_probe_fixture(
        tmp_path,
        "probe-pfc",
        spike_times_s=spike_times_s,
        spike_clusters=spike_clusters,
    )

    with pytest.raises(ValueError, match=error_match):
        activity.load_region_activity(make_region_config("PFC", "probe-pfc"), (probe_inputs,))


def test_load_region_activity_rejects_duplicate_selected_cluster_ids(tmp_path):
    """Selected metadata must have one stable row for every sorter cluster ID."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    cluster_info_path = probe_inputs.sorter_directory / "cluster_info.tsv"
    cluster_info = pd.read_csv(cluster_info_path, sep="\t")
    pd.concat([cluster_info, cluster_info.iloc[[1]]], ignore_index=True).to_csv(
        cluster_info_path,
        sep="\t",
        index=False,
    )

    with pytest.raises(ValueError, match="duplicate|cluster"):
        activity.load_region_activity(make_region_config("PFC", "probe-pfc"), (probe_inputs,))


def test_irig_coverage_is_mechanical_and_manual_bounds_are_not_an_ignored_override(tmp_path):
    """IRIG archives should derive coverage from finite one-dimensional IRIG values only."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc", include_irig=True)

    coverage = activity.inspect_probe_coverage(probe_inputs)

    assert coverage.source == "irig_utc_unix"
    assert coverage.start_s == 8.0
    assert coverage.end_s == 17.0
    with pytest.raises(ValueError, match="trusted|IRIG|irig"):
        activity.inspect_probe_coverage(probe_inputs, trusted_utc_bounds=(1.0, 2.0))


@pytest.mark.parametrize(
    "irig_values",
    [np.array([], dtype=float), np.array([[8.0, 17.0]]), np.array([8.0, np.inf])],
)
def test_irig_coverage_rejects_empty_multidimensional_or_nonfinite_values(tmp_path, irig_values):
    """Present IRIG coverage must be one-dimensional, nonempty, and finite."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc", include_irig=False)
    np.savez(
        probe_inputs.alignment_file,
        spike_utc_unix=np.array([9.0, 10.0]),
        irig_utc_unix=irig_values,
    )

    with pytest.raises(ValueError, match="irig_utc_unix"):
        activity.inspect_probe_coverage(probe_inputs)


def test_manual_coverage_requires_ordered_finite_bounds_and_dry_validation_avoids_spikes(
    tmp_path,
    monkeypatch,
):
    """Manual archives require trusted bounds and dry validation must not open spike data."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc", include_irig=False)
    opened_members: list[str] = []
    original_load = np.load

    class RecordingArchive:
        """Proxy recording archive member reads without materializing spike timestamps."""

        def __init__(self, archive):
            self._archive = archive
            self.files = archive.files

        def __getitem__(self, key):
            """Record one requested member and delegate to the real archive."""
            opened_members.append(key)
            return self._archive[key]

        def __enter__(self):
            """Return this archive proxy from the context manager."""
            return self

        def __exit__(self, *_args):
            """Close the wrapped archive after inspection."""
            self._archive.close()

    def recording_load(*args, **kwargs):
        """Wrap aligned NPZ loads while retaining normal NumPy behavior."""
        return RecordingArchive(original_load(*args, **kwargs))

    monkeypatch.setattr(activity.np, "load", recording_load)

    coverage = activity.inspect_probe_coverage(
        probe_inputs,
        trusted_utc_bounds=(8.0, 17.0),
        dry_run=True,
    )

    assert coverage.source == "trusted_config"
    assert coverage.start_s == 8.0
    assert coverage.end_s == 17.0
    assert "spike_utc_unix" not in opened_members
    with pytest.raises(ValueError, match="bounds"):
        activity.inspect_probe_coverage(probe_inputs)
    with pytest.raises(ValueError, match="bounds"):
        activity.inspect_probe_coverage(probe_inputs, trusted_utc_bounds=(17.0, 8.0))
    for invalid_bounds in ((np.nan, 17.0), (8.0, np.inf)):
        with pytest.raises(ValueError, match="finite|bounds"):
            activity.inspect_probe_coverage(probe_inputs, trusted_utc_bounds=invalid_bounds)


def test_dry_coverage_validation_requires_spike_member_name_without_loading_member(
    tmp_path,
    monkeypatch,
):
    """Dry validation should inspect archive names and reject a missing spike member early."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc", include_irig=False)
    np.savez(probe_inputs.alignment_file, irig_utc_unix=np.array([8.0, 17.0]))
    original_load = np.load

    def fail_if_member_read(*args, **kwargs):
        """Return a proxy whose member access proves dry validation stays name-only."""
        archive = original_load(*args, **kwargs)

        class NameOnlyArchive:
            """NPZ proxy exposing names but rejecting all array-member reads."""

            files = archive.files

            def __getitem__(self, key):
                """Fail if a dry validator attempts to load any NPZ member."""
                raise AssertionError(f"dry validation read unexpected member {key}")

            def __enter__(self):
                """Return the name-only archive proxy."""
                return self

            def __exit__(self, *_args):
                """Close the original archive."""
                archive.close()

        return NameOnlyArchive()

    monkeypatch.setattr(activity.np, "load", fail_if_member_read)

    with pytest.raises(ValueError, match="spike_utc_unix"):
        activity.inspect_probe_coverage(probe_inputs, dry_run=True)


def test_build_session_rate_tensors_uses_common_full_window_rows_and_hz_axes(tmp_path):
    """Both regions should use one full-window-covered row set and identical time axes."""
    pfc_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    hpc_inputs = write_probe_fixture(tmp_path, "probe-hpc")
    probe_sources = (pfc_inputs, hpc_inputs)
    pfc_activity = activity.load_region_activity(
        make_region_config("PFC", "probe-pfc"),
        probe_sources,
    )
    hpc_activity = activity.load_region_activity(
        make_region_config("HPC", "probe-hpc"),
        probe_sources,
    )
    target_table = make_target_table()

    result = activity.build_session_rate_tensors(
        target_table,
        pfc_activity,
        hpc_activity,
        alignment_event="choice_time",
        window_s=(-1.0, 1.0),
        bin_width_s=0.5,
    )

    np.testing.assert_array_equal(result.trial_row_indices, np.array([0, 2, 4]))
    np.testing.assert_array_equal(result.trial_ids, np.array([0, 2, 4]))
    assert result.pfc_rate_tensor_hz.shape == (3, 4, 2)
    assert result.hpc_rate_tensor_hz.shape == (3, 4, 2)
    np.testing.assert_allclose(result.pfc_bin_centers_s, result.hpc_bin_centers_s)
    assert result.rate_units == "Hz"


def test_rate_tensors_match_existing_numpy_binning_and_do_not_zero_fill_excluded_rows(tmp_path):
    """Pynapple tensors should agree with the existing NumPy reference on retained rows only."""
    pfc_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    hpc_inputs = write_probe_fixture(tmp_path, "probe-hpc")
    probe_sources = (pfc_inputs, hpc_inputs)
    pfc_activity = activity.load_region_activity(
        make_region_config("PFC", "probe-pfc"),
        probe_sources,
    )
    hpc_activity = activity.load_region_activity(
        make_region_config("HPC", "probe-hpc"),
        probe_sources,
    )
    result = activity.build_session_rate_tensors(
        make_target_table(),
        pfc_activity,
        hpc_activity,
        alignment_event="choice_time",
        window_s=(-1.0, 1.0),
        bin_width_s=0.5,
        verify_numpy_reference=True,
    )

    np.testing.assert_allclose(result.pfc_rate_tensor_hz, result.pfc_numpy_reference_hz)
    assert result.pfc_rate_tensor_hz.shape[0] == result.trial_row_indices.size
    assert 1 not in result.trial_row_indices
    assert 3 not in result.trial_row_indices
    assert 5 not in result.trial_row_indices


def test_one_probe_coverage_truncation_removes_a_row_from_both_region_tensors(tmp_path):
    """A full-window miss on HPC should remove that original row from PFC and HPC alike."""
    pfc_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    hpc_inputs = write_probe_fixture(tmp_path, "probe-hpc")
    np.savez(
        hpc_inputs.alignment_file,
        spike_utc_unix=np.array([8.5, 10.2, 12.6, 16.5]),
        irig_utc_unix=np.array([8.0, 13.5]),
    )
    probe_sources = (pfc_inputs, hpc_inputs)
    pfc_activity = activity.load_region_activity(
        make_region_config("PFC", "probe-pfc"),
        probe_sources,
    )
    hpc_activity = activity.load_region_activity(
        make_region_config("HPC", "probe-hpc"),
        probe_sources,
    )

    result = activity.build_session_rate_tensors(
        make_target_table(),
        pfc_activity,
        hpc_activity,
        alignment_event="choice_time",
        window_s=(-1.0, 1.0),
        bin_width_s=0.5,
    )

    np.testing.assert_array_equal(result.trial_row_indices, np.array([0, 2]))
    assert result.pfc_rate_tensor_hz.shape[0] == 2
    assert result.hpc_rate_tensor_hz.shape[0] == 2


def test_projecting_target_masks_through_common_tensor_rows_preserves_order():
    """Target masks should project into the shared PFC/HPC/combined tensor-row order."""
    projected = activity.project_target_mask_to_tensor_rows(
        np.array([0, 2, 4]),
        np.array([True, False, True, False, True, True]),
    )

    np.testing.assert_array_equal(projected, np.array([True, True, True]))


def test_tensor_memory_estimate_and_budget_guard_use_float64_tensor_bytes_only():
    """Memory preflight should count two regional float64 tensors, not source-file bytes."""
    tensor_bytes = activity.estimate_rate_tensor_bytes(
        n_tensor_trials=3,
        n_time_bins=4,
        n_pfc_units=2,
        n_hpc_units=3,
    )

    assert tensor_bytes == 3 * 4 * (2 + 3) * 8
    assert activity.select_memory_budget_bytes(local_mem_available_bytes=1_000) == 1_000
    assert activity.select_memory_budget_bytes(
        local_mem_available_bytes=1_000,
        slurm_memory_limit_bytes=800,
    ) == 800
    assert activity.select_memory_budget_bytes(
        local_mem_available_bytes=1_000,
        login_requested_memory_bytes=900,
    ) == 900
    with pytest.raises(ValueError, match="memory|budget"):
        activity.select_memory_budget_bytes(
            local_mem_available_bytes=None,
            slurm_memory_limit_bytes=None,
            login_requested_memory_bytes=None,
        )
    activity.validate_tensor_memory_budget(tensor_bytes=400, memory_budget_bytes=800)
    with pytest.raises(ValueError, match="50%|memory"):
        activity.validate_tensor_memory_budget(tensor_bytes=401, memory_budget_bytes=800)


def test_dry_activity_inspection_reports_source_sizes_separately_from_tensor_bytes(
    tmp_path,
    monkeypatch,
):
    """Dry inspection should report each source size without adding it to tensor allocation."""
    pfc_inputs = write_probe_fixture(tmp_path, "probe-pfc")
    hpc_inputs = write_probe_fixture(tmp_path, "probe-hpc")
    probe_sources = (pfc_inputs, hpc_inputs)
    original_load = np.load
    opened_members: list[str] = []

    class IrigOnlyArchive:
        """Proxy allowing dry IRIG reads while failing unexpected neural-array access."""

        def __init__(self, archive):
            self._archive = archive
            self.files = archive.files

        def __getitem__(self, key):
            """Record IRIG reads and reject loading the large spike-time member."""
            opened_members.append(key)
            if key == "spike_utc_unix":
                raise AssertionError("dry activity inspection read spike_utc_unix")
            return self._archive[key]

        def __enter__(self):
            """Return this archive proxy for ``with`` use."""
            return self

        def __exit__(self, *_args):
            """Close the wrapped archive after dry inspection."""
            self._archive.close()

    def dry_run_load(file_path, *args, **kwargs):
        """Reject sorter cluster-array loading while proxying aligned archives."""
        if Path(file_path).name == "spike_clusters.npy":
            raise AssertionError("dry activity inspection loaded spike_clusters.npy")
        loaded = original_load(file_path, *args, **kwargs)
        if Path(file_path).suffix == ".npz":
            return IrigOnlyArchive(loaded)
        return loaded

    monkeypatch.setattr(activity.np, "load", dry_run_load)
    report = activity.inspect_activity_dry_run(
        make_target_table(),
        make_region_config("PFC", "probe-pfc"),
        make_region_config("HPC", "probe-hpc"),
        probe_sources,
        alignment_event="choice_time",
        window_s=(-1.0, 1.0),
        bin_width_s=0.5,
    )

    expected_sizes = {
        source_path: source_path.stat().st_size
        for source_path in (
            pfc_inputs.alignment_file,
            pfc_inputs.sorter_directory / "spike_clusters.npy",
            pfc_inputs.sorter_directory / "cluster_info.tsv",
            pfc_inputs.channel_quality_file,
            hpc_inputs.alignment_file,
            hpc_inputs.sorter_directory / "spike_clusters.npy",
            hpc_inputs.sorter_directory / "cluster_info.tsv",
            hpc_inputs.channel_quality_file,
        )
    }
    assert report.tensor_allocation_bytes == 3 * 4 * (2 + 2) * 8
    assert report.source_file_sizes_bytes == expected_sizes
    assert report.tensor_trial_count == 3
    assert report.time_bin_count == 4
    assert report.pfc_unit_count == 2
    assert report.hpc_unit_count == 2
    assert "spike_utc_unix" not in opened_members


def test_zero_selected_channels_or_units_are_preflight_errors(tmp_path):
    """Configured regions with no selected channels or units must fail before tensor allocation."""
    probe_inputs = write_probe_fixture(tmp_path, "probe-pfc", unit_channels=(2,))

    with pytest.raises(ValueError, match="channel|zero"):
        activity.load_region_activity(make_region_config("PFC", "probe-pfc"), (probe_inputs,))

    probe_inputs = write_probe_fixture(tmp_path / "units", "probe-pfc")
    cluster_info_path = probe_inputs.sorter_directory / "cluster_info.tsv"
    pd.read_csv(cluster_info_path, sep="\t").assign(group="noise").to_csv(
        cluster_info_path,
        sep="\t",
        index=False,
    )
    with pytest.raises(ValueError, match="unit|zero"):
        activity.load_region_activity(make_region_config("PFC", "probe-pfc"), (probe_inputs,))
