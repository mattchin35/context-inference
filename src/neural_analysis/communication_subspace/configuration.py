"""Fixed scientific controls for a communication-subspace session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


ANALYSIS_VERSION = "communication-subspace-v1"
WINDOW_S = (-2.0, 2.0)
BIN_WIDTHS_MS = (100, 500)
DRAW_COUNT = 5
OUTER_FOLD_COUNT = 5


@dataclass(frozen=True)
class CommunicationConfig:
    """Session path and scalar controls, without loading data.

    session_metadata is a metadata JSON Path. probe_roles maps PFC/HPC to
    distinct declared probe IDs, or None to use metadata defaults. bin_width_ms
    is 100 or 500 milliseconds; seed is a nonnegative integer for independent
    population permutations. Windows are fixed at [-2,+2) seconds.
    """

    session_metadata: Path
    probe_roles: Mapping[str, str] | None = None
    bin_width_ms: int = 100
    seed: int = 0

    @property
    def bin_width_s(self) -> float:
        """Return the bin width as a scalar in seconds, with no data loading."""
        return self.bin_width_ms / 1000.0


def validate_configuration(config: CommunicationConfig) -> None:
    """Validate scalar controls; return None or raise ValueError.

    config contains a metadata Path, optional PFC/HPC probe mapping, a
    millisecond bin width, and an integer seed. This function performs no I/O;
    source path existence and declared probe IDs are checked by inspection.
    """
    if not isinstance(config, CommunicationConfig):
        raise ValueError("Expected CommunicationConfig.")
    if not isinstance(config.session_metadata, (Path, str)):
        raise ValueError("session_metadata must be a path.")
    if type(config.bin_width_ms) is not int or config.bin_width_ms not in BIN_WIDTHS_MS:
        raise ValueError(f"bin_width_ms must be one of {BIN_WIDTHS_MS}.")
    if type(config.seed) is not int or config.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    roles = config.probe_roles
    if roles is not None:
        if (not isinstance(roles, Mapping) or set(roles) != {"PFC", "HPC"}
                or any(not isinstance(probe, str) or not probe for probe in roles.values())
                or len(set(roles.values())) != 2):
            raise ValueError("probe_roles must map PFC and HPC to distinct probe IDs.")
