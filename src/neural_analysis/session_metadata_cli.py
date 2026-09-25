"""Create and validate the editable metadata for one neural session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from src.neural_analysis.session_metadata import (
    CANONICAL_FILENAME,
    SUPPORTED_ACTIONS,
    load_session_metadata,
    resolve_session_metadata,
    validate_session_for_action,
)


def _skeleton() -> dict[str, object]:
    """Return the deterministic, intentionally incomplete version-2 skeleton."""
    return {
        "schema_version": "2",
        "session": "",
        "acquisition": "open_ephys",
        "behavior": {"trials": "", "events": None},
        "probes": {},
        "site_pairs": [],
        "cache": None,
    }


def _parser() -> argparse.ArgumentParser:
    """Return the two-command argument parser for metadata users."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create_parser = commands.add_parser("create", help="write an editable metadata skeleton")
    create_parser.add_argument("--session-root", type=Path, required=True)
    validate_parser = commands.add_parser("validate", help="validate one metadata file")
    validate_parser.add_argument("--metadata", type=Path, required=True)
    validate_parser.add_argument("--action", choices=SUPPORTED_ACTIONS)
    return parser


def _create(session_root: Path) -> int:
    """Write one skeleton below an existing session directory; return a process code."""
    if not session_root.is_dir():
        raise ValueError(f"session root is not an existing directory: {session_root}")
    metadata_path = session_root / CANONICAL_FILENAME
    encoded = json.dumps(_skeleton(), indent=2, sort_keys=True) + "\n"
    try:
        with metadata_path.open("x", encoding="ascii", newline="\n") as metadata_file:
            metadata_file.write(encoded)
    except FileExistsError as error:
        raise ValueError(f"metadata already exists: {metadata_path}") from error
    print(f"created {metadata_path}")
    return 0


def _validate(metadata_path: Path, action: str | None) -> int:
    """Print action availability for one metadata file; return a process code."""
    metadata = load_session_metadata(metadata_path)
    session = resolve_session_metadata(metadata, metadata_path)
    actions = (action,) if action is not None else SUPPORTED_ACTIONS
    results = tuple(validate_session_for_action(session, item) for item in actions)
    for result in results:
        state = "available" if result.available else "unavailable"
        print(f"{result.action}: {state}")
        if result.missing_inputs:
            print(f"  missing: {', '.join(result.missing_inputs)}")
    if action is not None and not results[0].available:
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the metadata CLI and return ``0`` for success or ``2`` for user errors.

    Parameters
    ----------
    argv : sequence of str or None
        Arguments after the module name. ``None`` reads ``sys.argv`` through
        ``argparse``.

    Returns
    -------
    int
        Process exit code; no scientific arrays or analysis outputs are written.
    """
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            return _create(arguments.session_root)
        return _validate(arguments.metadata, arguments.action)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
