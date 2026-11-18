"""Lightweight SpikeGLX digital I/O helpers for synchronization workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import src.external_tools.readSGLX as readSGLX


def read_digital_lines(
    binary_file: Path | str,
    digital_word: int,
    digital_lines: list[int],
) -> tuple[np.ndarray, float]:
    """Read one or more SpikeGLX digital lines from a binary file.

    Args:
        binary_file: Path to a SpikeGLX ``.bin`` file.
        digital_word: Digital word index used by ``ExtractDigital``.
        digital_lines: Line indices within ``digital_word``. The returned array
            follows this order along axis 0.

    Returns:
        tuple[np.ndarray, float]:
            - Digital signal array with shape ``(n_lines, n_samples)`` for
              multiple lines or ``(n_samples,)`` for one line, stored as bool.
            - Sampling rate in Hz.
    """
    binary_file = Path(binary_file)
    meta = readSGLX.readMeta(binary_file)
    sample_rate_hz = float(readSGLX.SampRate(meta))
    n_channels = int(meta["nSavedChans"])
    n_samples = int(int(meta["fileSizeBytes"]) / (2 * n_channels))
    first_sample = 0
    last_sample = n_samples - 1

    raw_data = readSGLX.makeMemMapRaw(binary_file, meta)
    digital_array = readSGLX.ExtractDigital(
        raw_data,
        first_sample,
        last_sample,
        digital_word,
        digital_lines,
        meta,
    )
    return np.asarray(np.squeeze(digital_array), dtype=bool), sample_rate_hz


def read_digital_line(
    binary_file: Path | str,
    digital_word: int,
    digital_line: int,
) -> tuple[np.ndarray, float]:
    """Read a single SpikeGLX digital line from a binary file.

    Args:
        binary_file: Path to a SpikeGLX ``.bin`` file.
        digital_word: Digital word index used by ``ExtractDigital``.
        digital_line: Line index within ``digital_word``.

    Returns:
        tuple[np.ndarray, float]:
            - One-dimensional digital signal with shape ``(n_samples,)`` stored
              as bool.
            - Sampling rate in Hz.
    """
    digital_signal, sample_rate_hz = read_digital_lines(
        binary_file=binary_file,
        digital_word=digital_word,
        digital_lines=[digital_line],
    )
    return np.asarray(digital_signal, dtype=bool).reshape(-1), sample_rate_hz
