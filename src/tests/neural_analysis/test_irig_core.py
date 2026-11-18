from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from src.irig_tools import irig_core


def _pulse_widths_from_bits(bits: np.ndarray) -> np.ndarray:
    width_lookup = {
        False: 0.2,
        True: 0.5,
        "P": 0.8,
    }
    return np.asarray([width_lookup[bit] for bit in bits], dtype=float)


def test_neurokairos_frame_round_trip_uses_utc_unix_seconds():
    unix_seconds = datetime(2026, 7, 7, 16, 10, 0, tzinfo=timezone.utc).timestamp()

    frame_bits = irig_core.encode_irig_frame(
        unix_seconds,
        irig_format="neurokairos",
        chrony_stratum=2,
        root_dispersion_s=0.0007,
    )
    frame_fields = irig_core.decode_irig_frame(frame_bits, irig_format="neurokairos", century_base=2000)

    assert frame_fields["decoded_utc_seconds"] == unix_seconds
    assert frame_fields["decoded_datetime_utc"] == "2026-07-07T16:10:00+00:00"
    assert frame_fields["decoded_stratum"] == "stratum_2"
    assert frame_fields["decoded_dispersion_bucket"] == "0.5-1 ms"


def test_standard_decisecond_frame_round_trip_preserves_deciseconds():
    unix_seconds = datetime(2026, 7, 7, 16, 10, 1, tzinfo=timezone.utc).timestamp() + 0.4

    frame_bits = irig_core.encode_irig_frame(unix_seconds, irig_format="standard_decisecond")
    frame_fields = irig_core.decode_irig_frame(
        frame_bits,
        irig_format="standard_decisecond",
        century_base=2000,
    )

    assert np.isclose(frame_fields["decoded_utc_seconds"], unix_seconds)
    assert frame_fields["deciseconds"] == 4


def test_frame_finding_skips_partial_initial_frame():
    frame = np.asarray(irig_core.encode_irig_frame(0.0), dtype=object)
    bits = np.concatenate([frame[21:], frame, frame])

    spans = irig_core.find_irig_frame_spans(bits)

    assert spans[:2] == [(39, 99), (99, 159)]


def test_decode_pulses_retains_invalid_pulses_and_warns():
    first_unix = datetime(2026, 7, 7, 16, 10, 0, tzinfo=timezone.utc).timestamp()
    first_frame = np.asarray(irig_core.encode_irig_frame(first_unix), dtype=object)
    second_frame = np.asarray(irig_core.encode_irig_frame(first_unix + 60), dtype=object)
    valid_bits = np.concatenate([first_frame, second_frame])
    pulse_widths = _pulse_widths_from_bits(valid_bits)
    pulse_widths = np.insert(pulse_widths, 60, 0.01)

    with pytest.warns(UserWarning, match="Invalid IRIG pulses"):
        decoded = irig_core.decode_irig_pulses(
            pulse_width_s=pulse_widths,
            source_onset_time_s=np.arange(pulse_widths.size, dtype=float),
            source_onset_ix=np.arange(pulse_widths.size, dtype=np.int64),
            irig_format="neurokairos",
            century_base=2000,
        )

    assert decoded.invalid_pulse_ix.tolist() == [60]
    assert decoded.pulse_df.shape[0] == pulse_widths.size
    assert decoded.pulse_df.loc[60, "irig_bit"] is None
    assert np.isnan(decoded.pulse_df.loc[60, "utc_unix"])
    assert decoded.frame_df["frame_start_pulse_ix"].tolist() == [0, 61]


def test_decode_pulses_preserves_current_one_second_utc_assignment():
    first_unix = datetime(2026, 7, 7, 16, 10, 0, tzinfo=timezone.utc).timestamp()
    bits = np.concatenate(
        [
            np.asarray(irig_core.encode_irig_frame(first_unix), dtype=object),
            np.asarray(irig_core.encode_irig_frame(first_unix + 60), dtype=object),
        ]
    )

    decoded = irig_core.decode_irig_pulses(
        pulse_width_s=_pulse_widths_from_bits(bits),
        source_onset_time_s=np.arange(bits.size, dtype=float),
        source_onset_ix=np.arange(bits.size, dtype=np.int64),
        irig_format="neurokairos",
        century_base=2000,
    )

    assert np.allclose(
        decoded.pulse_df["utc_unix"].to_numpy(dtype=float),
        first_unix + np.arange(bits.size, dtype=float),
    )
