from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.irig_tools import decode_irig_logic_csv
from src.irig_tools import irig_core
from src.irig_tools import irig_serial_io


def test_logic_csv_module_decodes_core_neurokairos_frame():
    first_unix = datetime(2026, 7, 7, 16, 10, 0, tzinfo=timezone.utc).timestamp()
    bits = np.concatenate(
        [
            np.asarray(irig_core.encode_irig_frame(first_unix, irig_format="neurokairos"), dtype=object),
            np.asarray(irig_core.encode_irig_frame(first_unix + 60, irig_format="neurokairos"), dtype=object),
        ]
    )
    pulse_width_lookup = {False: 0.2, True: 0.5, "P": 0.8}

    pulse_df = pd.DataFrame(
        {
            "rising_edge_ix": np.arange(bits.size, dtype=int),
            "rising_edge_utc_seconds": first_unix + np.arange(bits.size, dtype=float),
            "rising_edge_local_datetime": [""] * bits.size,
            "pulse_width_s": [pulse_width_lookup[bit] for bit in bits],
        }
    )

    frame_df = decode_irig_logic_csv.build_frame_debug_table(pulse_df, century_base=2000)

    assert frame_df.shape[0] == 2
    assert frame_df.loc[0, "decoded_utc_seconds"] == first_unix


def test_irig_serial_io_parses_coolterm_text_and_builds_transition_table():
    text = "\n".join(
        [
            "2026-07-07 12:06:54.000\tsyncPinState:HIGH;elapsed:0",
            "2026-07-07 12:06:54.200\tsyncPinState:LOW;elapsed:200000",
            "2026-07-07 12:06:55.000\tsyncPinState:HIGH;elapsed:800000",
            "2026-07-07 12:06:55.500\tsyncPinState:LOW;elapsed:500000",
        ]
    )

    parsed_df = irig_serial_io.parse_coolterm_serial_text(text)
    transition_df = irig_serial_io.treadmill_log_to_transition_df(parsed_df)

    assert parsed_df["time_value"].tolist() == [0.0, 0.2, 0.8, 0.5]
    assert transition_df["pin_state"].tolist() == [1, 0, 1, 0]
    assert np.allclose(transition_df["utc_seconds"].diff().dropna().to_numpy(), [0.2, 0.8, 0.5])
