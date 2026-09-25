from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.neural_analysis import lfp_loading
from src.neural_analysis import psth_webapp
from src.neural_analysis import spike_behavior_pynapple
from src.neural_analysis import unit_spike_loading


def _write_open_ephys_lfp(
    lfp_path: Path,
    lfp_matrix: np.ndarray,
    sample_rate_hz: float = 2_500.0,
    *,
    scaling: dict[str, object] | None = None,
    metadata_overrides: dict[str, object] | None = None,
) -> None:
    """Write a tiny derived Open Ephys LFP file and metadata JSON.

    Args:
        lfp_path: Destination binary path. The sibling metadata file is named
            ``lfp_preprocessing.json``.
        lfp_matrix: Time-major LFP array with shape
        ``(n_samples, n_channels)`` in stored Open Ephys binary values.
        sample_rate_hz: LFP sample rate in Hz.
        scaling: Optional replacement for the observed nested
            ``lfp_binary_scaling`` object. The default records identity affine
            conversion to physical microvolts for each saved channel.
        metadata_overrides: Optional top-level metadata replacements for one
            malformed-sidecar test.

    Returns:
        None.
    """
    lfp_path.parent.mkdir(parents=True, exist_ok=True)
    lfp_matrix.astype(np.float32).tofile(lfp_path)
    n_channels = int(lfp_matrix.shape[1])
    observed_scaling = {
        "data_units": "unscaled_binary_values",
        "has_scaleable_traces": True,
        "channel_ids": [f"CH{index}" for index in range(n_channels)],
        "gain_to_uV_by_channel": [1.0] * n_channels,
        "offset_to_uV_by_channel": [0.0] * n_channels,
        "physical_unit_by_channel": ["uV"] * n_channels,
        "export_scale_factor": 1.0,
        "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
    }
    metadata: dict[str, object] = {
        "output_binary": lfp_path.name,
        "sampling_frequency_hz": float(sample_rate_hz),
        "num_channels": n_channels,
        "num_segments": 1,
        "num_samples_by_segment": [int(lfp_matrix.shape[0])],
        "dtype": "float32",
        "binary_layout": "time_major_channel_interleaved",
        "channel_ids_in_binary_order": [f"CH{index}" for index in range(n_channels)],
        "lfp_binary_scaling": observed_scaling if scaling is None else scaling,
    }
    if metadata_overrides is not None:
        metadata.update(metadata_overrides)
    (lfp_path.parent / "lfp_preprocessing.json").write_text(json.dumps(metadata) + "\n")


def _write_aligned_sync_npz(sync_path: Path) -> None:
    """Write a minimal aligned Open Ephys sync ``.npz`` for tests.

    Args:
        sync_path: Destination ``.npz`` path.

    Returns:
        None.
    """
    sync_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "generator": "sync_open_ephys_kilosort_spikes_to_utc",
        "continuous_start_sample_ix": 1_000,
        "units": {
            "irig_sample_ix": "Open Ephys global samples",
            "utc_unix": "seconds",
        },
    }
    np.savez(
        sync_path,
        irig_sample_ix=np.array([1_000, 31_000, 61_000], dtype=np.int64),
        irig_utc_unix=np.array([100.0, 101.0, 102.0], dtype=float),
        meta=np.asarray(meta, dtype=object),
    )


def test_load_sorter_metadata_requires_phy_cluster_info(tmp_path: Path):
    sorter_dir = tmp_path / "kilosort4"
    sorter_dir.mkdir()
    np.save(sorter_dir / "spike_clusters.npy", np.array([1, 2, 1], dtype=np.int64))

    with pytest.raises(FileNotFoundError, match="manual spike curation in Phy"):
        spike_behavior_pynapple.load_sorter_metadata(sorter_dir)


def test_filter_cluster_metadata_can_use_kslabel_quality_column():
    cluster_info = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "ch": [10, 10, 10],
            "group": [np.nan, "noise", np.nan],
            "KSLabel": ["good", "mua", "noise"],
        }
    )

    selected = unit_spike_loading.filter_cluster_metadata(
        cluster_info=cluster_info,
        region_channels=[10],
        quality_labels=("good", "mua"),
        quality_column="KSLabel",
    )

    assert selected["cluster_id"].tolist() == [1, 2]
    assert selected["quality_label"].tolist() == ["good", "mua"]


def test_load_open_ephys_lfp_metadata_reads_json_contract(tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(lfp_path, np.zeros((5, 3), dtype=np.float32), sample_rate_hz=1_250.0)

    metadata = lfp_loading.load_open_ephys_lfp_metadata(lfp_path)

    assert metadata["sampling_frequency_hz"] == 1_250.0
    assert metadata["num_channels"] == 3
    assert metadata["num_samples"] == 5
    assert metadata["dtype"] == "float32"
    assert metadata["binary_layout"] == "time_major_channel_interleaved"
    assert metadata["lfp_binary_scaling"]["gain_to_uV_by_channel"] == [1.0, 1.0, 1.0]
    assert metadata["lfp_binary_scaling"]["physical_unit_by_channel"] == ["uV", "uV", "uV"]


def test_read_open_ephys_lfp_channel_window_reads_time_major_float32(tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    lfp_matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    _write_open_ephys_lfp(lfp_path, lfp_matrix, sample_rate_hz=2_500.0)

    lfp_values, sample_rate_hz = lfp_loading.read_open_ephys_lfp_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=2,
        start_sample=1,
        stop_sample=4,
    )

    np.testing.assert_allclose(lfp_values, np.array([6.0, 10.0, 14.0], dtype=float))
    assert sample_rate_hz == 2_500.0


@pytest.mark.parametrize(
    ("remove_nested_scaling", "error_field"),
    (
        (None, "lfp_binary_scaling"),
        ("data_units", "data_units"),
        ("has_scaleable_traces", "has_scaleable_traces"),
        ("channel_ids", "channel_ids"),
        ("gain_to_uV_by_channel", "gain_to_uV_by_channel"),
        ("offset_to_uV_by_channel", "offset_to_uV_by_channel"),
        ("physical_unit_by_channel", "physical_unit_by_channel"),
        ("export_scale_factor", "export_scale_factor"),
        ("conversion", "conversion"),
    ),
)
def test_open_ephys_metadata_requires_nested_scaling_and_each_affine_vector(
    tmp_path: Path,
    remove_nested_scaling: str | None,
    error_field: str,
) -> None:
    """The nested scaling object and its channelwise affine fields are mandatory."""
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(lfp_path, np.ones((2, 1), dtype=np.float32))
    metadata_path = lfp_path.parent / "lfp_preprocessing.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if remove_nested_scaling is None:
        metadata.pop("lfp_binary_scaling")
    else:
        metadata["lfp_binary_scaling"].pop(remove_nested_scaling)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=error_field):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


def test_read_open_ephys_lfp_channel_window_applies_selected_affine_scaling_once(tmp_path: Path):
    """One copied channel window receives only its selected gain and offset in uV."""
    lfp_path = tmp_path / "lfp.dat"
    lfp_matrix = np.array(
        [[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]],
        dtype=np.float32,
    )
    _write_open_ephys_lfp(
        lfp_path,
        lfp_matrix,
        sample_rate_hz=1_250.0,
        scaling={
            "data_units": "unscaled_binary_values",
            "has_scaleable_traces": True,
            "channel_ids": ["CH0", "CH1", "CH2"],
            "gain_to_uV_by_channel": [0.1, 2.5, 99.0],
            "offset_to_uV_by_channel": [-1.0, 7.0, 13.0],
            "physical_unit_by_channel": ["uV", "uV", "uV"],
            "export_scale_factor": 1.0,
            "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
        },
        metadata_overrides={"channel_ids_in_binary_order": ["CH0", "CH1", "CH2"]},
    )

    values, sample_rate_hz = lfp_loading.read_open_ephys_lfp_channel_window(
        lfp_path=lfp_path,
        saved_channel_index=1,
        start_sample=0,
        stop_sample=3,
    )

    np.testing.assert_array_equal(values, np.array([32.0, 57.0, 82.0]))
    assert values.shape == (3,)
    assert values.dtype == float
    assert sample_rate_hz == 1_250.0


def test_open_ephys_channel_reader_materializes_only_the_requested_channel_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The binary adapter indexes one `(time window, channel)` slice, never a full matrix."""
    lfp_path = tmp_path / "lfp.dat"
    stored = np.arange(30, dtype=np.float32).reshape(10, 3)
    _write_open_ephys_lfp(lfp_path, stored)
    requested: list[tuple[object, object]] = []

    class WindowOnlyMemmap:
        """Minimal read seam that rejects any non-window binary access."""

        def __getitem__(self, index: tuple[object, object]) -> np.ndarray:
            """Record and serve exactly one channel-window key."""
            requested.append(index)
            assert index == (slice(3, 6), 1)
            return stored[index]

    def fake_memmap(*_: object, **__: object) -> WindowOnlyMemmap:
        """Return a window-only seam in place of a filesystem memmap."""
        return WindowOnlyMemmap()

    monkeypatch.setattr(lfp_loading.np, "memmap", fake_memmap)
    values, _ = lfp_loading.read_open_ephys_lfp_channel_window(lfp_path, 1, 3, 6)

    np.testing.assert_array_equal(values, stored[3:6, 1])
    assert requested == [(slice(3, 6), 1)]


def test_open_ephys_channel_reader_applies_both_affine_operations_in_place_on_one_copied_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The requested copied float window is the sole affine output buffer.

    The one copied float window may come from ``astype(..., copy=True)`` or
    ``np.asarray``/``np.array``. Both ``values *= gain``/``values += offset``
    and the equivalent ufunc ``out=values`` form satisfy this contract. A
    separate expression result such as ``values * gain + offset`` cannot
    satisfy the recorded output buffer identities.
    """
    original_asarray = np.asarray
    original_array = np.array
    copied_buffers: list[np.ndarray] = []
    operations: list[tuple[str, bool]] = []
    stored = np.arange(30, dtype=np.float32).reshape(10, 3)

    class TrackingArray(np.ndarray):
        """Track affine ufunc output storage without imposing one call syntax."""

        def astype(
            self,
            dtype: object,
            order: str = "K",
            casting: str = "unsafe",
            subok: bool = True,
            copy: bool = True,
        ) -> np.ndarray:
            """Record an allowed one-window ``astype(..., copy=True)`` allocation."""
            converted = super().astype(
                dtype,
                order=order,
                casting=casting,
                subok=subok,
                copy=copy,
            )
            if not copy:
                return converted
            copied = converted.reshape(-1).view(TrackingArray)
            copied_buffers.append(copied)
            return copied

        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: str,
            *inputs: object,
            **kwargs: object,
        ) -> object:
            """Record whether each supported affine ufunc writes back to this buffer."""
            if method != "__call__" or ufunc not in {np.multiply, np.add}:
                return NotImplemented
            output = kwargs.get("out")
            output_tuple = output if isinstance(output, tuple) else ()
            same_buffer = bool(output_tuple) and output_tuple[0] is not None and np.shares_memory(
                output_tuple[0], copied_buffers[0]
            )
            operations.append((ufunc.__name__, same_buffer))
            raw_inputs = tuple(
                value.view(np.ndarray) if isinstance(value, TrackingArray) else value
                for value in inputs
            )
            if output_tuple:
                kwargs["out"] = tuple(
                    value.view(np.ndarray) if isinstance(value, TrackingArray) else value
                    for value in output_tuple
                )
            result = getattr(ufunc, method)(*raw_inputs, **kwargs)
            return output_tuple[0] if output_tuple else result

    def tracking_float_copy(
        values: object,
        dtype: object | None = None,
        *_: object,
        **__: object,
    ) -> TrackingArray:
        """Provide a one-dimensional copied float buffer visible to affine ufuncs."""
        copied = original_array(values, dtype=dtype, copy=True).reshape(-1).view(TrackingArray)
        copied_buffers.append(copied)
        return copied

    class WindowOnlyMemmap:
        """Serve only the selected source slice without materializing the binary matrix."""

        def __getitem__(self, index: tuple[object, object]) -> np.ndarray:
            """Return the requested stored channel slice."""
            assert index == (slice(3, 6), 1)
            return stored[index].view(TrackingArray)

    metadata = {
        "sampling_frequency_hz": 1_250.0,
        "num_channels": 3,
        "num_samples": 10,
        "dtype": "float32",
        "binary_layout": "time_major_channel_interleaved",
        "lfp_binary_scaling": {
            "gain_to_uV_by_channel": [1.0, 2.5, 1.0],
            "offset_to_uV_by_channel": [0.0, 7.0, 0.0],
            "physical_unit_by_channel": ["uV", "uV", "uV"],
        },
    }
    with monkeypatch.context() as scoped:
        scoped.setattr(lfp_loading, "load_open_ephys_lfp_metadata", lambda _: metadata)
        scoped.setattr(lfp_loading.np, "memmap", lambda *_args, **_kwargs: WindowOnlyMemmap())
        scoped.setattr(lfp_loading.np, "asarray", tracking_float_copy)
        scoped.setattr(lfp_loading.np, "array", tracking_float_copy)
        values, sample_rate_hz = lfp_loading.read_open_ephys_lfp_channel_window(
            Path("lfp.dat"), 1, 3, 6
        )

    assert len(copied_buffers) == 1
    assert copied_buffers[0].shape == (3,)
    assert np.shares_memory(values, copied_buffers[0])
    assert operations == [("multiply", True), ("add", True)]
    np.testing.assert_array_equal(
        values.view(np.ndarray), np.array([32.0, 39.5, 47.0])
    )
    assert sample_rate_hz == 1_250.0


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("data_units", "uV"),
        ("data_units", None),
        ("has_scaleable_traces", False),
        ("has_scaleable_traces", "true"),
        ("export_scale_factor", 2.0),
        ("export_scale_factor", "1.0"),
        ("export_scale_factor", True),
        ("conversion", "trace_uV = trace_value * gain_to_uV"),
    ),
)
def test_open_ephys_metadata_rejects_wrong_observed_scaling_declarations(
    tmp_path: Path,
    field_name: str,
    invalid_value: object,
) -> None:
    """Only the documented CT026 scaling declarations are accepted."""
    lfp_path = tmp_path / "lfp.dat"
    scaling = {
        "data_units": "unscaled_binary_values",
        "has_scaleable_traces": True,
        "channel_ids": ["CH0"],
        "gain_to_uV_by_channel": [0.195],
        "offset_to_uV_by_channel": [0.0],
        "physical_unit_by_channel": ["uV"],
        "export_scale_factor": 1.0,
        "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
    }
    scaling[field_name] = invalid_value
    _write_open_ephys_lfp(lfp_path, np.ones((2, 1), dtype=np.float32), scaling=scaling)

    with pytest.raises(ValueError, match=field_name):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


@pytest.mark.parametrize(
    ("gain", "offset", "expected_valid"),
    (
        (float("nan"), 0.0, False),
        (float("inf"), 0.0, False),
        (0.0, 0.0, False),
        (-0.195, 0.0, False),
        (0.195, float("nan"), False),
        (0.195, float("inf"), False),
        ("0.195", 0.0, False),
        (True, 0.0, False),
        (0.195, "0.0", False),
        (0.195, False, False),
        (0.195, 0.0, True),
        (0.195, 1.5, True),
        (0.195, -1.5, True),
    ),
)
def test_open_ephys_metadata_validates_channel_affine_values(
    tmp_path: Path,
    gain: float,
    offset: float,
    expected_valid: bool,
) -> None:
    """Gains are finite positive values; finite signed offsets remain physical."""
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(
        lfp_path,
        np.ones((2, 1), dtype=np.float32),
        scaling={
            "data_units": "unscaled_binary_values",
            "has_scaleable_traces": True,
            "channel_ids": ["CH0"],
            "gain_to_uV_by_channel": [gain],
            "offset_to_uV_by_channel": [offset],
            "physical_unit_by_channel": ["uV"],
            "export_scale_factor": 1.0,
            "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
        },
    )

    if expected_valid:
        metadata = lfp_loading.load_open_ephys_lfp_metadata(lfp_path)
        assert metadata["lfp_binary_scaling"]["gain_to_uV_by_channel"] == [gain]
    else:
        with pytest.raises(ValueError, match="gain|offset"):
            lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


@pytest.mark.parametrize(
    ("scaling_updates", "metadata_overrides", "error_field"),
    (
        ({"channel_ids": ["CH0", "CH1"]}, {"channel_ids_in_binary_order": ["CH0", "CH2"]}, "channel_ids"),
        ({"gain_to_uV_by_channel": [0.195]}, {}, "gain_to_uV_by_channel"),
        ({"offset_to_uV_by_channel": [0.0]}, {}, "offset_to_uV_by_channel"),
        ({"physical_unit_by_channel": ["uV"]}, {}, "physical_unit_by_channel"),
        ({"physical_unit_by_channel": ["uV", "mV"]}, {}, "physical_unit_by_channel"),
        ({"physical_unit_by_channel": ["uV", "volts"]}, {}, "physical_unit_by_channel"),
        ({"physical_unit_by_channel": ["mV", "mV"]}, {}, "physical_unit_by_channel"),
        ({"physical_unit_by_channel": ["", ""]}, {}, "physical_unit_by_channel"),
        ({"physical_unit_by_channel": [None, None]}, {}, "physical_unit_by_channel"),
        ({}, {"num_channels": 3}, "num_channels"),
    ),
)
def test_open_ephys_metadata_rejects_channel_order_count_and_unit_disagreements(
    tmp_path: Path,
    scaling_updates: dict[str, object],
    metadata_overrides: dict[str, object],
    error_field: str,
) -> None:
    """Every affine vector is aligned to the exact saved-channel order in physical uV."""
    lfp_path = tmp_path / "lfp.dat"
    scaling = {
        "data_units": "unscaled_binary_values",
        "has_scaleable_traces": True,
        "channel_ids": ["CH0", "CH1"],
        "gain_to_uV_by_channel": [0.195, 0.196],
        "offset_to_uV_by_channel": [0.0, 0.0],
        "physical_unit_by_channel": ["uV", "uV"],
        "export_scale_factor": 1.0,
        "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
    }
    scaling.update(scaling_updates)
    _write_open_ephys_lfp(
        lfp_path,
        np.ones((2, 2), dtype=np.float32),
        scaling=scaling,
        metadata_overrides={"channel_ids_in_binary_order": ["CH0", "CH1"], **metadata_overrides},
    )

    with pytest.raises(ValueError, match=error_field):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


@pytest.mark.parametrize(
    "declared_dtype",
    ("int16", "float64", ">f4", "float16", "f4", "<f4", "=f4"),
)
def test_open_ephys_metadata_rejects_all_noncanonical_dtype_declarations(
    tmp_path: Path,
    declared_dtype: str,
) -> None:
    """The production adapter accepts only the observed native-endian float32 storage."""
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(
        lfp_path,
        np.ones((2, 1), dtype=np.float32),
        metadata_overrides={"dtype": declared_dtype},
    )

    with pytest.raises(ValueError, match="dtype"):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


@pytest.mark.parametrize(
    ("metadata_overrides", "error_field"),
    (
        ({"sampling_frequency_hz": None}, "sampling_frequency_hz"),
        ({"sampling_frequency_hz": "2500"}, "sampling_frequency_hz"),
        ({"sampling_frequency_hz": float("nan")}, "sampling_frequency_hz"),
        ({"sampling_frequency_hz": float("inf")}, "sampling_frequency_hz"),
        ({"sampling_frequency_hz": 0.0}, "sampling_frequency_hz"),
        ({"sampling_frequency_hz": -2500.0}, "sampling_frequency_hz"),
        ({"sampling_frequency_hz": True}, "sampling_frequency_hz"),
        ({"num_channels": None}, "num_channels"),
        ({"num_channels": "1"}, "num_channels"),
        ({"num_channels": 0}, "num_channels"),
        ({"num_channels": -1}, "num_channels"),
        ({"num_channels": True}, "num_channels"),
        ({"num_segments": None}, "num_segments"),
        ({"num_segments": "1"}, "num_segments"),
        ({"num_segments": 0}, "num_segments"),
        ({"num_segments": -1}, "num_segments"),
        ({"num_segments": True}, "num_segments"),
        ({"num_samples_by_segment": []}, "num_samples_by_segment"),
        ({"num_samples_by_segment": [None]}, "num_samples_by_segment"),
        ({"num_samples_by_segment": ["2"]}, "num_samples_by_segment"),
        ({"num_samples_by_segment": [0]}, "num_samples_by_segment"),
        ({"num_samples_by_segment": [-2]}, "num_samples_by_segment"),
        ({"num_samples_by_segment": [1.5]}, "num_samples_by_segment"),
        ({"num_samples_by_segment": [True]}, "num_samples_by_segment"),
    ),
)
def test_open_ephys_metadata_rejects_malformed_or_nonpositive_primitive_ranges(
    tmp_path: Path,
    metadata_overrides: dict[str, object],
    error_field: str,
) -> None:
    """Sampling, channel, segment, and sample-count primitives fail closed before binary reads."""
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(
        lfp_path,
        np.ones((2, 1), dtype=np.float32),
        metadata_overrides=metadata_overrides,
    )

    with pytest.raises(ValueError, match=error_field):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


@pytest.mark.parametrize(
    "metadata_overrides",
    (
        {"output_binary": "other.dat"},
        {"num_segments": 2},
        {"num_samples_by_segment": [2, 2]},
        {"binary_layout": "channel_major"},
    ),
)
def test_open_ephys_metadata_rejects_binary_name_and_segment_disagreements(
    tmp_path: Path,
    metadata_overrides: dict[str, object],
) -> None:
    """The sidecar describes exactly one selected binary and one sample-count segment."""
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(
        lfp_path,
        np.ones((2, 1), dtype=np.float32),
        metadata_overrides=metadata_overrides,
    )

    with pytest.raises(ValueError, match="output_binary|num_segments|num_samples_by_segment"):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


@pytest.mark.parametrize(
    "missing_key",
    (
        "output_binary",
        "sampling_frequency_hz",
        "num_channels",
        "num_segments",
        "num_samples_by_segment",
        "dtype",
        "binary_layout",
        "channel_ids_in_binary_order",
    ),
)
def test_open_ephys_metadata_requires_top_level_binary_identity_keys(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """A physical scaling declaration is unusable without its binary identity fields."""
    lfp_path = tmp_path / "lfp.dat"
    _write_open_ephys_lfp(lfp_path, np.ones((2, 1), dtype=np.float32))
    metadata_path = lfp_path.parent / "lfp_preprocessing.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop(missing_key)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=missing_key):
        lfp_loading.load_open_ephys_lfp_metadata(lfp_path)


def test_build_open_ephys_lfp_irig_df_converts_global_ap_samples_to_lfp_samples(tmp_path: Path):
    sync_path = tmp_path / "probeA_sync.npz"
    _write_aligned_sync_npz(sync_path)

    lfp_irig_df = lfp_loading.build_open_ephys_lfp_irig_df(
        aligned_sync_npz_path=sync_path,
        lfp_sample_rate_hz=2_500.0,
        continuous_sample_rate_hz=30_000.0,
    )

    np.testing.assert_allclose(lfp_irig_df["sample_ix"].to_numpy(dtype=float), np.array([0.0, 2500.0, 5000.0]))
    np.testing.assert_allclose(lfp_irig_df["utc_unix"].to_numpy(dtype=float), np.array([100.0, 101.0, 102.0]))


def test_load_open_ephys_trial_lfp_trace_uses_aligned_sync_npz(tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    sync_path = tmp_path / "probeA_sync.npz"
    lfp_matrix = np.arange(60, dtype=np.float32).reshape(20, 3)
    _write_open_ephys_lfp(lfp_path, lfp_matrix, sample_rate_hz=10.0)
    _write_aligned_sync_npz(sync_path)

    relative_time_s, lfp_values = lfp_loading.load_open_ephys_trial_lfp_trace(
        lfp_path=lfp_path,
        aligned_sync_npz_path=sync_path,
        saved_channel_index=1,
        alignment_time_s=101.0,
        window=(-0.2, 0.3),
        continuous_sample_rate_hz=30_000.0,
    )

    np.testing.assert_allclose(relative_time_s, np.array([-0.2, -0.1, 0.0, 0.1, 0.2]))
    np.testing.assert_allclose(lfp_values, lfp_matrix[8:13, 1].astype(float))


def test_open_ephys_trial_loader_scales_once_before_optional_filtering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Event-relative time support is unchanged while the raw trace becomes physical uV."""
    lfp_path = tmp_path / "lfp.dat"
    sync_path = tmp_path / "probeA_sync.npz"
    lfp_matrix = np.arange(20, dtype=np.float32).reshape(20, 1)
    _write_open_ephys_lfp(
        lfp_path,
        lfp_matrix,
        sample_rate_hz=10.0,
        scaling={
            "data_units": "unscaled_binary_values",
            "has_scaleable_traces": True,
            "channel_ids": ["CH0"],
            "gain_to_uV_by_channel": [2.0],
            "offset_to_uV_by_channel": [3.0],
            "physical_unit_by_channel": ["uV"],
            "export_scale_factor": 1.0,
            "conversion": "trace_uV = trace_value * gain_to_uV + offset_to_uV",
        },
        metadata_overrides={"channel_ids_in_binary_order": ["CH0"]},
    )
    _write_aligned_sync_npz(sync_path)
    filtered_inputs: list[np.ndarray] = []

    def fake_filter(
        relative_time_s: np.ndarray,
        lfp_uv: np.ndarray,
        sample_rate_hz: float,
        frequency_band_hz: tuple[float, float] | None,
    ) -> np.ndarray:
        """Prove the bandpass stage receives the once-converted physical trace."""
        assert sample_rate_hz == 10.0
        assert frequency_band_hz == (1.0, 4.0)
        assert relative_time_s.shape == lfp_uv.shape
        filtered_inputs.append(lfp_uv.copy())
        return lfp_uv

    monkeypatch.setattr(lfp_loading, "filter_lfp_trace", fake_filter)

    relative_time_s, lfp_uv = lfp_loading.load_open_ephys_trial_lfp_trace(
        lfp_path=lfp_path,
        aligned_sync_npz_path=sync_path,
        saved_channel_index=0,
        alignment_time_s=101.0,
        window=(-0.2, 0.3),
        continuous_sample_rate_hz=30_000.0,
        frequency_band_hz=(1.0, 4.0),
        filter_padding_s=0.0,
    )

    np.testing.assert_array_equal(relative_time_s, np.array([-0.2, -0.1, 0.0, 0.1, 0.2]))
    np.testing.assert_array_equal(lfp_uv, np.array([19.0, 21.0, 23.0, 25.0, 27.0]))
    assert len(filtered_inputs) == 1
    np.testing.assert_array_equal(filtered_inputs[0], lfp_uv)


def test_webapp_open_ephys_lfp_route_does_not_require_spikeglx_meta(monkeypatch, tmp_path: Path):
    lfp_path = tmp_path / "lfp.dat"
    sync_path = tmp_path / "probeA_sync.npz"
    seen = {}

    def fake_load_open_ephys_trial_lfp_trace(**kwargs):
        seen.update(kwargs)
        return np.array([0.0], dtype=float), np.array([1.0], dtype=float)

    monkeypatch.setattr(
        psth_webapp.lfp_loading,
        "load_open_ephys_trial_lfp_trace",
        fake_load_open_ephys_trial_lfp_trace,
    )

    relative_time_s, lfp_values = psth_webapp.load_trial_lfp_trace_for_format(
        lfp_format=psth_webapp.LFP_FORMAT_OPEN_EPHYS_DERIVED,
        lfp_path=str(lfp_path),
        saved_channel_index=3,
        alignment_time_s=101.0,
        window_start_s=-0.5,
        window_end_s=0.5,
        digital_word=0,
        irig_line=6,
        bit_period_s=1.0,
        utc_offset_hours=0.0,
        filter_low_hz=None,
        filter_high_hz=None,
        filter_padding_s=1.0,
        aligned_sync_npz_path=str(sync_path),
    )

    np.testing.assert_allclose(relative_time_s, np.array([0.0]))
    np.testing.assert_allclose(lfp_values, np.array([1.0]))
    assert seen["lfp_path"] == lfp_path
    assert seen["aligned_sync_npz_path"] == sync_path
    assert seen["saved_channel_index"] == 3
