from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import psth_webapp
from src.neural_analysis import unit_spike_loading


def _write_channel_quality_csv(path: Path) -> None:
    """Write a tiny channel-quality CSV fixture.

    Args:
        path: Destination CSV path.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "channel_id": ["CH0", "CH1", "CH2", "CH3"],
            "label": ["noise", "good", "good", "out"],
            "is_good": [False, True, True, False],
            "inside_brain": [True, True, False, False],
            "x_um": [0.0, 32.0, 0.0, 32.0],
            "y_um": [0.0, 0.0, 15.0, 15.0],
        }
    ).to_csv(path, index=False)


def _write_channel_quality_json(path: Path) -> None:
    """Write a tiny channel-quality JSON fixture without an is_good field.

    Args:
        path: Destination JSON path.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "test",
        "channels": [
            {"channel_id": "CH0", "label": "noise", "inside_brain": True, "x_um": 0.0, "y_um": 0.0},
            {"channel_id": "CH1", "label": "good", "inside_brain": True, "x_um": 32.0, "y_um": 0.0},
            {"channel_id": "CH2", "label": "good", "inside_brain": False, "x_um": 0.0, "y_um": 15.0},
        ],
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_load_channel_quality_csv_from_probe_directory_normalizes_schema(tmp_path: Path):
    probe_dir = tmp_path / "Record_Node_101_Neuropix-PXI-110.ProbeA"
    _write_channel_quality_csv(probe_dir / "channel_quality.csv")

    channel_quality = unit_spike_loading.load_channel_quality(probe_dir)

    assert channel_quality["ch"].tolist() == [0, 1, 2, 3]
    assert channel_quality["label"].tolist() == ["noise", "good", "good", "out"]
    assert channel_quality["is_good"].tolist() == [False, True, True, False]
    assert channel_quality["inside_brain"].tolist() == [True, True, False, False]
    np.testing.assert_allclose(channel_quality["x_um"].to_numpy(dtype=float), np.array([0.0, 32.0, 0.0, 32.0]))


def test_load_channel_quality_json_infers_is_good_from_label(tmp_path: Path):
    channel_quality_path = tmp_path / "channel_quality.json"
    _write_channel_quality_json(channel_quality_path)

    channel_quality = unit_spike_loading.load_channel_quality(channel_quality_path)

    assert channel_quality["ch"].tolist() == [0, 1, 2]
    assert channel_quality["is_good"].tolist() == [False, True, True]


def test_load_channel_quality_rejects_missing_required_columns(tmp_path: Path):
    channel_quality_path = tmp_path / "channel_quality.csv"
    pd.DataFrame({"channel_id": ["CH0"], "label": ["good"]}).to_csv(channel_quality_path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        unit_spike_loading.load_channel_quality(channel_quality_path)


def test_load_channel_quality_rejects_duplicate_channels(tmp_path: Path):
    channel_quality_path = tmp_path / "channel_quality.csv"
    pd.DataFrame(
        {
            "channel_id": ["CH1", "CH1"],
            "label": ["good", "noise"],
            "inside_brain": [True, True],
            "x_um": [0.0, 0.0],
            "y_um": [0.0, 0.0],
        }
    ).to_csv(channel_quality_path, index=False)

    with pytest.raises(ValueError, match="Duplicate"):
        unit_spike_loading.load_channel_quality(channel_quality_path)


def test_select_channels_from_quality_defaults_to_good_inside_brain(tmp_path: Path):
    probe_dir = tmp_path / "probe"
    _write_channel_quality_csv(probe_dir / "channel_quality.csv")
    channel_quality = unit_spike_loading.load_channel_quality(probe_dir)

    selected_channels = unit_spike_loading.select_channels_from_quality(channel_quality)

    np.testing.assert_array_equal(selected_channels, np.array([1], dtype=int))


def test_select_channels_from_quality_can_use_label_filter_without_inside_brain(tmp_path: Path):
    probe_dir = tmp_path / "probe"
    _write_channel_quality_csv(probe_dir / "channel_quality.csv")
    channel_quality = unit_spike_loading.load_channel_quality(probe_dir)

    selected_channels = unit_spike_loading.select_channels_from_quality(
        channel_quality,
        require_inside_brain=False,
        labels=("good",),
    )

    np.testing.assert_array_equal(selected_channels, np.array([1, 2], dtype=int))


def test_infer_probe_derived_dir_prefers_kilosort_parent(tmp_path: Path):
    probe_dir = tmp_path / "Record_Node_101_Neuropix-PXI-110.ProbeA"
    sorter_path = probe_dir / "kilosort4"
    lfp_path = probe_dir / "lfp.dat"

    inferred_dir = unit_spike_loading.infer_probe_derived_dir(
        sorter_output_path=sorter_path,
        lfp_path=lfp_path,
    )

    assert inferred_dir == probe_dir


def test_infer_probe_derived_dir_uses_lfp_parent_when_sorter_missing(tmp_path: Path):
    probe_dir = tmp_path / "Record_Node_101_Neuropix-PXI-110.ProbeA"
    lfp_path = probe_dir / "lfp.dat"

    inferred_dir = unit_spike_loading.infer_probe_derived_dir(
        sorter_output_path=None,
        lfp_path=lfp_path,
    )

    assert inferred_dir == probe_dir


def test_webapp_channel_source_resolver_preserves_manual_channels():
    channel_quality = pd.DataFrame(
        {
            "ch": [1],
            "channel_id": ["CH1"],
            "label": ["good"],
            "is_good": [True],
            "inside_brain": [True],
            "x_um": [0.0],
            "y_um": [0.0],
        }
    )

    channels, summary = psth_webapp.resolve_region_channels_for_source(
        channel_source=psth_webapp.CHANNEL_SOURCE_MANUAL,
        manual_channel_text="5, 6",
        channel_quality=channel_quality,
        require_inside_brain=True,
        channel_quality_labels=("good",),
    )

    np.testing.assert_array_equal(channels, np.array([5, 6], dtype=int))
    assert "Manual" in summary


def test_webapp_channel_source_resolver_uses_channel_quality_filters(tmp_path: Path):
    probe_dir = tmp_path / "probe"
    _write_channel_quality_csv(probe_dir / "channel_quality.csv")
    channel_quality = unit_spike_loading.load_channel_quality(probe_dir)

    channels, summary = psth_webapp.resolve_region_channels_for_source(
        channel_source=psth_webapp.CHANNEL_SOURCE_CHANNEL_QUALITY,
        manual_channel_text="5, 6",
        channel_quality=channel_quality,
        require_inside_brain=True,
        channel_quality_labels=("good",),
    )

    np.testing.assert_array_equal(channels, np.array([1], dtype=int))
    assert "1 / 4" in summary
