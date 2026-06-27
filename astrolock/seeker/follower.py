"""
Follower: tail a camera's .ser via its .frames.jsonl spine.

The frames sidecar is the authoritative stream: line N (0-based) describes frame N, and a
committed line means that frame's pixels are already flushed. We additionally validate
cross-file ordering -- trust a sidecar line only if the .ser is physically long enough to
contain that frame (see astrolock_seeker.md, "Cross-file ordering"). This also handles a
torn final frame from an abrupt kill.

Rollover/successor chaining across multiple .ser segments is not needed for the hello-world
(one segment per session); locating the current segment is by newest timestamp.
"""

import glob
import os

from astrolock.seeker import ser as ser_mod
from astrolock.seeker import sidecar


class SerFollower:
    """
    Follow one role's capture within a session directory.

    Resolves the current ``<ts>_<role>.ser`` + ``<ts>_<role>.frames.jsonl`` pair (newest by
    name) lazily, so it can be started before the cam has created the files.
    """

    def __init__(self, session_dir, role):
        self.session_dir = str(session_dir)
        self.role = role
        self._ser_path = None
        self._frames_path = None
        self._reader = None

    def _resolve(self):
        """Find/refresh the newest .ser for this role; (re)open the reader if needed."""
        matches = sorted(glob.glob(os.path.join(self.session_dir, f"*_{self.role}.ser")))
        if not matches:
            return False
        newest = matches[-1]
        if newest != self._ser_path:
            if self._reader is not None:
                self._reader.close()
            self._ser_path = newest
            self._frames_path = newest[:-len('.ser')] + '.frames.jsonl'
            self._reader = None  # opened on demand below
        if self._reader is None:
            try:
                self._reader = ser_mod.SerReader(self._ser_path)
            except (ValueError, FileNotFoundError):
                self._reader = None  # header not fully written yet
                return False
        return True

    def committed_count(self):
        """
        Number of frames safe to read: min(committed sidecar lines, frames physically on
        disk in the .ser). The min is the cross-file ordering guard.
        """
        if not self._resolve():
            return 0
        sidecar_lines = sidecar.count_complete_lines(self._frames_path)
        on_disk = self._reader.frames_on_disk()
        return min(sidecar_lines, on_disk)

    def latest_index(self):
        return self.committed_count() - 1

    def read_frame(self, index, to_float=False):
        if not self._resolve():
            raise IndexError("no capture available yet")
        return self._reader.read_frame(index, to_float=to_float)

    def read_latest(self, to_float=False):
        """Return (index, frame) for the newest committed frame, or None if none yet."""
        idx = self.latest_index()
        if idx < 0:
            return None
        return idx, self.read_frame(idx, to_float=to_float)

    @property
    def header(self):
        return self._reader.header if self._reader is not None else None

    @property
    def ser_path(self):
        """Path of the currently-followed .ser (None until a frame exists)."""
        self._resolve()
        return self._ser_path

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
