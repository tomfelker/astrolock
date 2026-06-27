"""
Session directories and file naming for AstroLock Seeker.

Captured data lives under ``sessions/<ts>/`` with every file timestamp-prefixed for
age-based purging (see astrolock_seeker.md). Config is separate and not timestamped.

File naming: ``<ts>_<role>.<kind>`` where kind is ``ser`` (pixels), ``frames.jsonl``
(per-frame metadata / commit-point spine), ``detections.jsonl``, or ``state.jsonl``.
"""

import datetime
import os


def utc_stamp(dt=None):
    """Return a UTC timestamp like '20260624T210312Z' (ISO-8601 basic, second resolution)."""
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.strftime('%Y%m%dT%H%M%SZ')


def segment_stamp(dt=None):
    """
    Millisecond-resolution UTC stamp for naming capture segments, e.g. '20260624T210312123Z'.
    No underscore (so '<stamp>_<role>' role-parsing still works) and fixed width (so segment
    files sort chronologically). Used per .ser segment when a cam rolls over.
    """
    if dt is None:
        dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.strftime('%Y%m%dT%H%M%S') + f"{dt.microsecond // 1000:03d}Z"


def utc_now_iso():
    """Return a UTC wall-clock time like '2026-06-24T21:03:12.213Z'."""
    dt = datetime.datetime.now(datetime.timezone.utc)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.') + f"{dt.microsecond // 1000:03d}Z"


def new_session_dir(base='sessions', ts=None):
    """Create and return a fresh session directory path."""
    ts = ts or utc_stamp()
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path, ts


def ser_name(ts, role):
    return f"{ts}_{role}.ser"


def frames_name(ts, role):
    return f"{ts}_{role}.frames.jsonl"


def detections_name(ts, role):
    return f"{ts}_{role}.detections.jsonl"


def state_name(ts):
    return f"{ts}_state.jsonl"
