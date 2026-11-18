from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import unit_spike_loading


def test_list_path_browser_entries_lists_sorted_directories_and_filtered_files(tmp_path):
    """The webapp path browser should expose sorted folders and selected file suffixes."""
    (tmp_path / "b_dir").mkdir()
    (tmp_path / "a_dir").mkdir()
    (tmp_path / "z.txt").write_text("not selected")
    (tmp_path / "b.npz").write_text("selected")
    (tmp_path / "a.npz").write_text("selected")

    entries = unit_spike_loading.list_path_browser_entries(
        tmp_path,
        file_suffix=".npz",
    )

    assert [path.name for path in entries["directories"]] == ["a_dir", "b_dir"]
    assert [path.name for path in entries["files"]] == ["a.npz", "b.npz"]


def test_list_path_browser_entries_rejects_missing_path(tmp_path):
    """Missing browser directories should fail clearly before UI rendering."""
    with pytest.raises(FileNotFoundError):
        unit_spike_loading.list_path_browser_entries(tmp_path / "missing")


def test_parse_channel_list_accepts_common_separators():
    """Editable channel text should support comma, whitespace, and newline separators."""
    parsed_channels = unit_spike_loading.parse_channel_list("1, 2\n3 4")

    np.testing.assert_array_equal(parsed_channels, np.array([1, 2, 3, 4], dtype=int))


def test_parse_channel_list_rejects_non_integer_tokens():
    """Channel text should fail clearly before it can silently change unit selection."""
    with pytest.raises(ValueError, match="integer"):
        unit_spike_loading.parse_channel_list("1, two, 3")


def test_ct014_region_channel_presets_include_nonempty_hpc_v1_and_pfc():
    """CT014 presets should provide useful defaults without being treated as universal."""
    presets = unit_spike_loading.get_ct014_region_channel_presets()

    assert {"HPC", "V1", "PFC", "Custom"}.issubset(presets)
    assert presets["HPC"].ndim == 1
    assert presets["HPC"].size > 0
    assert presets["V1"].ndim == 1
    assert presets["V1"].size > 0
    assert presets["PFC"].ndim == 1
    assert presets["PFC"].dtype.kind in {"i", "u"}
    assert presets["PFC"].size > 0
    np.testing.assert_array_equal(presets["Custom"], np.array([], dtype=int))


def test_ct014_region_channel_presets_have_no_duplicate_channels():
    """Hardcoded CT014 region presets should fail tests if duplicate channels reappear."""
    presets = unit_spike_loading.get_ct014_region_channel_presets()

    for region_name in ("HPC", "V1", "PFC"):
        region_channels = presets[region_name]
        assert np.unique(region_channels).size == region_channels.size, region_name


def test_ct014_region_channel_presets_use_zero_indexed_unit_channels():
    """Webapp unit presets should not include non-unit saved rows such as LFP sync channels."""
    presets = unit_spike_loading.get_ct014_region_channel_presets()

    for region_name in ("HPC", "V1", "PFC"):
        region_channels = presets[region_name]
        assert region_channels.min() >= 0, region_name
        assert region_channels.max() <= 383, region_name


def test_get_probe_label_for_region_maps_hpc_and_v1_to_hpc_v1():
    """HPC and V1 should share the same probe data source for current sessions."""
    assert unit_spike_loading.get_probe_label_for_region("HPC") == "HPC/V1"
    assert unit_spike_loading.get_probe_label_for_region("V1") == "HPC/V1"


def test_get_probe_label_for_region_maps_pfc_to_pfc():
    """PFC should route to the PFC probe data source."""
    assert unit_spike_loading.get_probe_label_for_region("PFC") == "PFC"


def test_get_probe_label_for_region_requires_custom_probe_for_custom():
    """Custom channel selections need explicit probe routing."""
    with pytest.raises(ValueError, match="Custom"):
        unit_spike_loading.get_probe_label_for_region("Custom")


def test_get_probe_label_for_region_uses_explicit_custom_probe():
    """Custom channels should use the user-selected probe source."""
    assert unit_spike_loading.get_probe_label_for_region("Custom", custom_probe_label="PFC") == "PFC"


def test_get_probe_label_for_region_rejects_unknown_region():
    """Unknown region names should fail rather than silently selecting the wrong probe."""
    with pytest.raises(ValueError, match="Unsupported"):
        unit_spike_loading.get_probe_label_for_region("M2")


def test_validate_probe_paths_for_spike_plot_allows_missing_lfp(tmp_path: Path):
    """Spike-only plots should not require an LFP file."""
    sorter_path = tmp_path / "sorter"
    sorter_path.mkdir()
    aligned_spike_path = tmp_path / "aligned_spikes.npz"
    aligned_spike_path.write_bytes(b"placeholder")
    probe_paths = unit_spike_loading.ProbeDataPaths(
        label="PFC",
        sorter_output_path=sorter_path,
        aligned_spike_path=aligned_spike_path,
        lfp_path=None,
    )

    unit_spike_loading.validate_probe_paths_for_spike_plot(probe_paths)


def test_validate_probe_paths_for_spike_plot_rejects_missing_sorter(tmp_path: Path):
    """The active probe must have a sorter directory for spike plots."""
    aligned_spike_path = tmp_path / "aligned_spikes.npz"
    aligned_spike_path.write_bytes(b"placeholder")
    probe_paths = unit_spike_loading.ProbeDataPaths(
        label="PFC",
        sorter_output_path=None,
        aligned_spike_path=aligned_spike_path,
        lfp_path=None,
    )

    with pytest.raises(ValueError, match="sorter"):
        unit_spike_loading.validate_probe_paths_for_spike_plot(probe_paths)


def test_validate_probe_paths_for_spike_plot_rejects_missing_aligned_spikes(tmp_path: Path):
    """The active probe must have aligned spike times for spike plots."""
    sorter_path = tmp_path / "sorter"
    sorter_path.mkdir()
    probe_paths = unit_spike_loading.ProbeDataPaths(
        label="PFC",
        sorter_output_path=sorter_path,
        aligned_spike_path=None,
        lfp_path=None,
    )

    with pytest.raises(ValueError, match="aligned spike"):
        unit_spike_loading.validate_probe_paths_for_spike_plot(probe_paths)


def test_load_viewer_data_for_probe_uses_selected_probe_paths(monkeypatch, tmp_path: Path):
    """Probe-aware loading should route sorter and aligned-spike reads to the active probe only."""
    sorter_path = tmp_path / "pfc_sorter"
    sorter_path.mkdir()
    aligned_spike_path = tmp_path / "pfc_sync.npz"
    aligned_spike_path.write_bytes(b"placeholder")
    probe_paths = unit_spike_loading.ProbeDataPaths(
        label="PFC",
        sorter_output_path=sorter_path,
        aligned_spike_path=aligned_spike_path,
        lfp_path=None,
    )
    calls = {}

    def fake_load_session_tables(session):
        calls["session"] = session
        return pd.DataFrame({"event": [1]}), pd.DataFrame({"choice_time": [1.0]})

    def fake_load_sorter_metadata(sorter_output_path):
        calls["sorter_output_path"] = sorter_output_path
        return (
            np.array([10, 11], dtype=int),
            pd.DataFrame({"cluster_id": [10, 11], "ch": [1, 2], "group": ["good", "mua"]}),
        )

    def fake_load_aligned_spikes(path):
        calls["aligned_spike_path"] = path
        return np.array([0.1, 0.2], dtype=float)

    def fake_validate_aligned_spike_inputs(aligned_spike_times, spike_clusters):
        calls["validated_shapes"] = (aligned_spike_times.shape, spike_clusters.shape)

    def fake_build_spike_tsgroup(spike_times, spike_clusters):
        calls["spike_times"] = spike_times
        calls["spike_clusters"] = spike_clusters
        return {"spike_group": True}

    monkeypatch.setattr(unit_spike_loading.sbp, "load_session_tables", fake_load_session_tables)
    monkeypatch.setattr(unit_spike_loading.sbp, "load_sorter_metadata", fake_load_sorter_metadata)
    monkeypatch.setattr(unit_spike_loading.sbp, "load_aligned_spikes", fake_load_aligned_spikes)
    monkeypatch.setattr(unit_spike_loading.sbp, "validate_aligned_spike_inputs", fake_validate_aligned_spike_inputs)
    monkeypatch.setattr(unit_spike_loading.sbp, "build_spike_tsgroup", fake_build_spike_tsgroup)

    viewer_data = unit_spike_loading.load_viewer_data_for_probe(
        session_data_home=tmp_path,
        sess_id_full="CT014_2025-12-23_163505",
        probe_paths=probe_paths,
    )

    assert calls["sorter_output_path"] == sorter_path
    assert calls["aligned_spike_path"] == aligned_spike_path
    assert calls["validated_shapes"] == ((2,), (2,))
    assert viewer_data["active_probe"] == probe_paths
    assert viewer_data["spike_group"] == {"spike_group": True}


def test_normalize_quality_labels_maps_real_missing_values_to_default_group():
    """Real missing group labels from cluster_info.tsv should become selectable strings."""
    quality_labels = unit_spike_loading.normalize_quality_labels(
        pd.Series([np.nan, "noise", "good"])
    )

    assert quality_labels.tolist() == ["mua", "noise", "good"]


def test_normalize_quality_labels_maps_string_missing_sentinels_to_default_group():
    """String missing labels should not survive into the Streamlit sorting options."""
    quality_labels = unit_spike_loading.normalize_quality_labels(
        pd.Series(["nan", "", "None", "  NaN  "])
    )

    assert quality_labels.tolist() == ["mua", "mua", "mua", "mua"]


def test_filter_cluster_metadata_uses_channels_and_quality_labels():
    """Unit metadata filtering should keep selected channels and requested quality labels."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11, 12, 13],
            "ch": [100, 101, 102, 103],
            "group": ["good", "mua", "noise", "nan"],
        }
    )

    filtered = unit_spike_loading.filter_cluster_metadata(
        cluster_info,
        region_channels=np.array([100, 101, 102, 103], dtype=int),
        quality_labels=("good", "mua"),
    )

    assert filtered["cluster_id"].tolist() == [10, 11, 13]
    assert filtered["quality_label"].tolist() == ["good", "mua", "mua"]


def test_filter_cluster_metadata_can_include_noise_when_requested():
    """The webapp should allow quality filtering without permanently excluding noise labels."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11],
            "ch": [100, 101],
            "group": ["mua", "noise"],
        }
    )

    filtered = unit_spike_loading.filter_cluster_metadata(
        cluster_info,
        region_channels=np.array([100, 101], dtype=int),
        quality_labels=("noise",),
    )

    assert filtered["cluster_id"].tolist() == [11]


def test_filter_cluster_metadata_includes_real_missing_groups_as_default_group():
    """Rows with real NaN manual labels should be treated as the default group."""
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [10, 11],
            "ch": [100, 101],
            "group": [np.nan, "noise"],
        }
    )

    filtered = unit_spike_loading.filter_cluster_metadata(
        cluster_info,
        region_channels=np.array([100, 101], dtype=int),
        quality_labels=("mua",),
    )

    assert filtered["cluster_id"].tolist() == [10]
    assert filtered["quality_label"].tolist() == ["mua"]
