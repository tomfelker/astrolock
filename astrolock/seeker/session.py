"""
Session directories and file naming for AstroLock Seeker.

Captured data lives under ``sessions/<ts>/`` with every file timestamp-prefixed for
age-based purging (see astrolock_seeker.md). Config is separate and not timestamped.

File naming: ``<ts>_<role>.<kind>`` where kind is ``ser`` (pixels), ``frames.jsonl``
(per-frame metadata / commit-point spine), ``detections.jsonl``, or ``state.jsonl``.
"""

import datetime
import os
import threading
import time


# Deliberate poison offset on the shared timeline. The epoch of perf_counter is arbitrary
# (~boot, magnitude ~1e14 ns), so a stamp that BYPASSES mono_ns() would only be subtly wrong.
# Shifting our timeline by ~95 years makes any mix loudly absurd: a raw perf_counter_ns stamp
# reads ~95 years in the past, and a UTC time_ns mixup (~1.8e18 until 2065) is off by decades.
# Still comfortably inside int64 (9.2e18) for the '<q' record fields.
_EPOCH_OFFSET_NS = 3_000_000_000 * 1_000_000_000


def parent_lifeline():
    """
    Orphan prevention (the recorder's 'NOBODY EVER KILLS US' pattern, generalized): a
    parent spawns its children with stdin=PIPE and never writes to it. The OS closes that
    pipe when the parent dies -- cleanly, by crash, or by hard kill alike -- so a child
    that watches its stdin for EOF can never be orphaned. Returns a threading.Event that
    is set on EOF; long-running loops check it beside their usual stop conditions.

    ARMS ONLY WHEN STDIN IS AN ACTUAL PIPE: an interactive console, a redirected file, or
    NUL never trigger it (the event just stays unset forever), so standalone runs behave
    exactly as before. The backend itself calls this too -- a test harness or supervisor
    that spawns the backend with a pipe takes the WHOLE tree down even if it crashes
    without cleanup (a crashed harness abandoning a LIVE backend was the one orphan case
    the child lifelines could not cover: nothing had died).
    """
    import stat
    import sys
    dead = threading.Event()
    try:
        if not stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode):
            return dead                        # console/file/NUL stdin: never armed
    except (OSError, ValueError, AttributeError):
        return dead                            # no usable stdin at all: never armed

    def _watch():
        try:
            while sys.stdin.readline():        # nobody writes; a line is ignored, EOF = parent gone
                pass
        except Exception:
            pass                               # stdin torn down counts as "parent gone" too
        dead.set()

    threading.Thread(target=_watch, daemon=True).start()
    return dead


def mono_ns():
    """THE clock for every cross-process stamp: frame capture times, detection records,
    focuser position reports, state blobs, sim-time anchors, mount angle times.
    perf_counter_ns (QPC on Windows, CLOCK_MONOTONIC on Linux -- one machine-wide timeline,
    valid to subtract across processes) plus a deliberate ~95-year offset so a stamp from any
    OTHER source (bare time.*_ns()) is instantly, unmissably wrong instead of subtly late.
    Data that leaves a process is stamped with this and nothing else; process-local
    durations/pacing may use time.perf_counter()."""
    return time.perf_counter_ns() + _EPOCH_OFFSET_NS


def mono_s():
    """mono_ns() in float seconds -- for control code whose 'now' meets stamped capture times
    (servo horizons, pose-history interpolation). Same poisoned timeline, so mixing in a raw
    time.perf_counter() 'now' is ~95 years off and fails immediately, not subtly."""
    return mono_ns() * 1e-9


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


def sim_mount_name(ts):
    """Ground-truth trajectory of the sim mount (piecewise-linear anchors), for the sim camera to
    read -- distinct from <ts>_state.jsonl, which is the backend's *estimate* of the mount."""
    return f"{ts}_sim_mount.jsonl"
