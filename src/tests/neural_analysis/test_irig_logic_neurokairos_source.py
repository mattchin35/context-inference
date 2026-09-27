from __future__ import annotations

from pathlib import Path


IRIG_LOGIC_SOURCE = Path("src/irig_tools/irig_logic_neurokairos.c")


def test_neurokairos_logic_source_does_not_reference_deciseconds():
    """NeuroKairos frame bits 42-48 should document clock-quality metadata, not deciseconds."""
    source_text = IRIG_LOGIC_SOURCE.read_text()

    assert "decisecond" not in source_text.lower()


def test_neurokairos_logic_source_has_explicit_marker_and_metadata_bits():
    """Frame construction should make marker, reserved, and metadata bit positions reviewable."""
    source_text = IRIG_LOGIC_SOURCE.read_text()

    for marker_ix in (0, 9, 19, 29, 39, 49, 59):
        assert f"frame[{marker_ix}] = IRIG_P" in source_text

    for reserved_ix in (5, 14, 18, 24, 27, 28, 34, 42, 45, 54):
        assert f"frame[{reserved_ix}] = IRIG_ZERO" in source_text

    assert "frame[43] = (stratum_enc & 1)" in source_text
    assert "frame[44] = (stratum_enc & 2)" in source_text
    assert "frame[46] = (dispersion_enc & 1)" in source_text
    assert "frame[47] = (dispersion_enc & 2)" in source_text
    assert "frame[48] = (dispersion_enc & 4)" in source_text
