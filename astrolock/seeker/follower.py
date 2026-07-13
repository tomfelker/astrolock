"""
Follower: the frame-ref convenience wrapper over framestream.StreamFollower.

A frame is named by (stream ident, ABSOLUTE index) together -- indices never wrap or restart
within a ring, so a FrameRef is globally unambiguous. The ident keeps the historical
'<stem>.ser' string shape (virtual): consumers use it only as an identity.

view(ref) is the zero-copy read (validate-on-exit, raises framestream.Lapped if the writer
reused the slot mid-use); read_frame(ref) is the copying convenience.
"""

import collections
from contextlib import contextmanager

from astrolock.seeker import framestream

FrameRef = collections.namedtuple('FrameRef', ['ser_path', 'index'])


def _seg_id(ident):
    return ident + '.ser'


class SerFollower:
    """Follow one stream within a session directory, chasing the newest frame."""

    def __init__(self, session_dir, role):
        self.session_dir = str(session_dir)
        self.role = role
        self._fo = framestream.StreamFollower(session_dir, role)

    def _reader(self, ser_path):
        for r in self._fo._readers:
            if _seg_id(r.ident) == ser_path:
                return r
        raise IndexError(f'{ser_path} is not an attached ring (reconfigured away?)')

    def committed_count(self):
        self._fo.poll()
        return self._fo.committed()

    def latest_ref(self):
        """FrameRef of the newest committed frame (or None), without touching its pixels."""
        self._fo.poll()
        got = self._fo.latest()
        return FrameRef(_seg_id(got[0].ident), got[1]) if got else None

    @contextmanager
    def view(self, ref):
        """ZERO-COPY view of the frame named by ref: yields (record, payload). Raises
        framestream.Lapped on exit if the writer reused the slot mid-use (discard the output),
        IndexError if the ring is gone / the frame not committed."""
        with self._reader(ref.ser_path).view(ref.index) as (rec, payload):
            yield rec, payload

    def read_frame(self, ref):
        """Copying read of the frame named by ref."""
        return self._reader(ref.ser_path).read(ref.index)

    def read_latest(self):
        """(FrameRef, frame copy) for the newest committed frame, or None if none yet."""
        self._fo.poll()
        got = self._fo.latest()
        if not got:
            return None
        r, i = got
        return FrameRef(_seg_id(r.ident), i), r.read(i)

    @property
    def header(self):
        return self._fo.header

    @property
    def meta(self):
        return self._fo.meta

    @property
    def ser_path(self):
        """Identity of the live ring ('<stem>.ser', virtual)."""
        ident = self._fo.ident
        return _seg_id(ident) if ident else None

    def poll(self):
        self._fo.poll()

    def committed(self):
        """Committed count of the live ring WITHOUT polling (pure memory read; wake probes)."""
        return self._fo.committed()

    def close(self):
        self._fo.close()
