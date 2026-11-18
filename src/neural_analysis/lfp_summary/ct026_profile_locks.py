"""Exclusive, recoverable run locks for work-only CT026 PPC profiling."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Callable, Iterator, Mapping


_LOCK_NAME = ".ct026-profile.lock"


@contextmanager
def acquire_ct026_profile_run_lock(
    run_directory: Path,
    identity: Mapping[str, str],
    *,
    pid: int,
    hostname: str,
    process_exists: Callable[[int], bool],
) -> Iterator[None]:
    """Acquire one exclusive CT026 profile-run lock or recover a safe stale lock.

    Parameters
    ----------
    run_directory : pathlib.Path
        Existing direct CT026 profile-run directory. The lock is stored only as
        ``.ct026-profile.lock`` inside this directory.
    identity : Mapping[str, str]
        Exact nonempty config, source, and Git identities for this run. These
        are categorical values with no physical units.
    pid : int
        Positive process identifier written to the ownership record.
    hostname : str
        Nonempty host identity written to the record.
    process_exists : callable
        Receives a PID and reports whether it is live on ``hostname``.

    Yields
    ------
    None
        The caller owns the lock until context exit. A stale lock is removed
        only after it is an exact-identity, same-host, dead-PID JSON record.

    Raises
    ------
    RuntimeError
        If a live, foreign-host, malformed, or otherwise unsafe lock exists.
    ValueError
        If the requested identity or a stale local record has incompatible
        identity fields.
    """
    directory = Path(run_directory)
    expected_identity = _validated_identity(identity)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("run_directory must be an existing nonsymlink directory")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ValueError("pid must be a positive integer")
    if not isinstance(hostname, str) or not hostname:
        raise ValueError("hostname must be nonempty")
    if not callable(process_exists):
        raise ValueError("process_exists must be callable")

    lock_path = directory / _LOCK_NAME
    record = {"hostname": hostname, "identity": expected_identity, "pid": pid}
    payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    try:
        descriptor = lock_path.open("xb")
    except FileExistsError:
        _recover_or_reject_existing_lock(lock_path, expected_identity, hostname, process_exists)
        try:
            descriptor = lock_path.open("xb")
        except FileExistsError as error:
            raise RuntimeError("CT026 profile run lock changed during recovery") from error
    with descriptor:
        descriptor.write(payload)
        descriptor.flush()
    try:
        yield
    finally:
        if lock_path.is_file() and lock_path.read_bytes() == payload:
            lock_path.unlink()


def _validated_identity(identity: Mapping[str, str]) -> dict[str, str]:
    """Copy the exact three nonempty categorical run identities."""
    if not isinstance(identity, Mapping) or set(identity) != {
        "config_fingerprint", "source_fingerprint", "git_fingerprint",
    }:
        raise ValueError("identity must contain exact config, source, and Git fingerprints")
    copied = dict(identity)
    if any(not isinstance(value, str) or not value for value in copied.values()):
        raise ValueError("identity fingerprints must be nonempty strings")
    return copied


def _recover_or_reject_existing_lock(
    lock_path: Path,
    expected_identity: Mapping[str, str],
    hostname: str,
    process_exists: Callable[[int], bool],
) -> None:
    """Remove only one validated dead local ownership record."""
    try:
        record = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("existing CT026 profile lock is not safely recoverable") from error
    if not isinstance(record, dict) or set(record) != {"hostname", "identity", "pid"}:
        raise RuntimeError("existing CT026 profile lock is not safely recoverable")
    if record["identity"] != dict(expected_identity):
        raise ValueError("CT026 profile lock identity does not match requested run")
    if record["hostname"] != hostname:
        raise RuntimeError("existing CT026 profile lock belongs to a foreign host")
    owner_pid = record["pid"]
    if isinstance(owner_pid, bool) or not isinstance(owner_pid, int) or owner_pid <= 0:
        raise RuntimeError("existing CT026 profile lock is not safely recoverable")
    if process_exists(owner_pid):
        raise RuntimeError("existing CT026 profile lock belongs to a live process")
    lock_path.unlink()
