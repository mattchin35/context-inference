from __future__ import annotations

from pathlib import Path

import numpy as np

from src.neural_analysis import spikeglx_sync_io


def test_read_digital_line_passes_path_objects_and_returns_one_dimensional_signal(
    monkeypatch,
    tmp_path: Path,
):
    binary_file = tmp_path / "run0_g0_t0.nidq.bin"
    binary_file.touch()
    seen: dict[str, object] = {}

    def fake_read_meta(bin_full_path: Path) -> dict[str, str]:
        seen["read_meta_type"] = type(bin_full_path)
        seen["read_meta_path"] = bin_full_path
        return {
            "nSavedChans": "1",
            "fileSizeBytes": "20",
            "typeThis": "nidq",
            "niSampRate": "10",
        }

    def fake_make_memmap_raw(bin_full_path: Path, meta: dict[str, str]) -> np.ndarray:
        seen["memmap_type"] = type(bin_full_path)
        seen["memmap_path"] = bin_full_path
        return np.zeros((1, 10), dtype=np.int16)

    def fake_extract_digital(
        raw_data: np.ndarray,
        first_sample: int,
        last_sample: int,
        digital_word: int,
        digital_lines: list[int],
        meta: dict[str, str],
    ) -> np.ndarray:
        return np.zeros((1, 10), dtype=bool)

    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "readMeta", fake_read_meta)
    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "SampRate", lambda meta: 10.0)
    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "makeMemMapRaw", fake_make_memmap_raw)
    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "ExtractDigital", fake_extract_digital)

    signal, sample_rate_hz = spikeglx_sync_io.read_digital_line(
        binary_file=binary_file,
        digital_word=0,
        digital_line=0,
    )

    assert signal.shape == (10,)
    assert sample_rate_hz == 10.0
    assert issubclass(seen["read_meta_type"], Path)
    assert issubclass(seen["memmap_type"], Path)
    assert seen["read_meta_path"] == binary_file
    assert seen["memmap_path"] == binary_file


def test_read_digital_lines_returns_expected_shape_and_sample_rate(monkeypatch, tmp_path: Path):
    binary_file = tmp_path / "run0_g0_t0.nidq.bin"
    binary_file.touch()

    monkeypatch.setattr(
        spikeglx_sync_io.readSGLX,
        "readMeta",
        lambda _: {
            "nSavedChans": "1",
            "fileSizeBytes": "20",
            "typeThis": "nidq",
            "niSampRate": "10",
        },
    )
    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "SampRate", lambda meta: 10.0)
    monkeypatch.setattr(
        spikeglx_sync_io.readSGLX,
        "makeMemMapRaw",
        lambda bin_full_path, meta: np.zeros((1, 10), dtype=np.int16),
    )
    monkeypatch.setattr(
        spikeglx_sync_io.readSGLX,
        "ExtractDigital",
        lambda raw_data, first_sample, last_sample, digital_word, digital_lines, meta: np.asarray(
            [[0, 1, 0, 1], [1, 1, 0, 0]],
            dtype=bool,
        ),
    )

    digital_lines, sample_rate_hz = spikeglx_sync_io.read_digital_lines(
        binary_file=binary_file,
        digital_word=0,
        digital_lines=[0, 1],
    )

    assert digital_lines.shape == (2, 4)
    assert sample_rate_hz == 10.0


def test_read_digital_line_matches_corresponding_row_of_multi_line_read(monkeypatch, tmp_path: Path):
    binary_file = tmp_path / "run0_g0_t0.nidq.bin"
    binary_file.touch()

    monkeypatch.setattr(
        spikeglx_sync_io.readSGLX,
        "readMeta",
        lambda _: {
            "nSavedChans": "1",
            "fileSizeBytes": "20",
            "typeThis": "nidq",
            "niSampRate": "10",
        },
    )
    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "SampRate", lambda meta: 10.0)
    monkeypatch.setattr(
        spikeglx_sync_io.readSGLX,
        "makeMemMapRaw",
        lambda bin_full_path, meta: np.zeros((1, 10), dtype=np.int16),
    )

    def fake_extract_digital(raw_data, first_sample, last_sample, digital_word, digital_lines, meta):
        if digital_lines == [0, 1]:
            return np.asarray([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=bool)
        if digital_lines == [1]:
            return np.asarray([[1, 0, 1, 0]], dtype=bool)
        raise AssertionError(f"Unexpected digital_lines: {digital_lines}")

    monkeypatch.setattr(spikeglx_sync_io.readSGLX, "ExtractDigital", fake_extract_digital)

    multi_line_signal, _ = spikeglx_sync_io.read_digital_lines(
        binary_file=binary_file,
        digital_word=0,
        digital_lines=[0, 1],
    )
    single_line_signal, _ = spikeglx_sync_io.read_digital_line(
        binary_file=binary_file,
        digital_word=0,
        digital_line=1,
    )

    assert np.array_equal(single_line_signal, multi_line_signal[1])
