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

import collections

from astrolock.seeker import framestream, sidecar


# A frame index is only meaningful against the .ser segment it was taken from: index N in one segment
# is a different frame than index N in the next. So a frame is named by (segment path, index) together,
# never a bare int -- read_frame refuses a ref whose segment we've already rolled past.
FrameRef = collections.namedtuple('FrameRef', ['ser_path', 'index'])


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
        """Find/refresh the newest segment for this role -- discovered BY SIDECAR (a shm segment
        has no .ser file; its identity path is virtual) -- and (re)open the reader if needed."""
        matches = framestream.sidecar_glob(self.session_dir, self.role)
        if not matches:
            return False
        newest = matches[-1]
        if newest != self._frames_path:
            if self._reader is not None:
                self._reader.close()
            self._frames_path = newest
            self._ser_path = framestream.ser_path_of(newest)
            self._reader = None  # opened on demand below
        if self._reader is None:
            try:
                self._reader = framestream.open_reader(self._ser_path)
            except (ValueError, FileNotFoundError):
                self._reader = None  # header not fully written yet / shm segment gone
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

    def latest_ref(self):
        """FrameRef of the newest committed frame (or None), without reading its pixels."""
        n = self.committed_count()
        return FrameRef(self._ser_path, n - 1) if n > 0 else None

    def read_frame(self, ref, to_float=False):
        """Read the frame named by a FrameRef. Refuses (IndexError) a ref for a segment we've rolled
        past -- an index is only valid against the file it was taken from, so we never silently read
        the wrong frame out of a fresher segment."""
        if not self._resolve():
            raise IndexError("no capture available yet")
        if ref.ser_path != self._ser_path:
            raise IndexError(f"frame {ref.index} is from {ref.ser_path}; current segment is {self._ser_path}")
        return self._reader.read_frame(ref.index, to_float=to_float)

    def read_latest(self, to_float=False):
        """
        Return (FrameRef, frame) for the newest committed frame, or None if none yet.

        Resolves the current segment exactly once and reads the count + frame from that same
        reader, so a rollover between resolving and reading can't ask a freshly-created (empty)
        segment for a stale index (the previous version resolved twice and could race at the
        300-frame rollover boundary). The returned ref carries the segment its index belongs to.
        """
        if not self._resolve():
            return None
        n = min(sidecar.count_complete_lines(self._frames_path), self._reader.frames_on_disk())
        if n <= 0:
            return None
        return FrameRef(self._ser_path, n - 1), self._reader.read_frame(n - 1, to_float=to_float)

    @property
    def header(self):
        return self._reader.header if self._reader is not None else None

    @property
    def ser_path(self):
        """Identity of the currently-followed segment ('<stem>.ser'; virtual for a shm segment)."""
        self._resolve()
        return self._ser_path

    @property
    def frames_path(self):
        """Path of the currently-followed frames sidecar (the on-disk, growing artifact)."""
        self._resolve()
        return self._frames_path

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
