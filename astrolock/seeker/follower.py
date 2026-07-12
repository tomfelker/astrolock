"""
Follower: the skip-to-newest consumer of a framestream (v2 chains).

Wraps framestream.StreamFollower with the frame-ref API the GUI/backend grew up with. A frame
is named by (segment id, index) together -- an index is only meaningful against its segment --
and read_frame refuses a ref for a segment we've released. The segment id remains the historical
'<stem>.ser' string (virtual: v2 segments store pixels in shm or a .dat heap); consumers use it
only as an identity and to derive sibling sidecar names (detections).
"""

import collections

from astrolock.seeker import framestream

FrameRef = collections.namedtuple('FrameRef', ['ser_path', 'index'])


def _seg_id(seg):
    return seg.frames_path[:-len('.frames.jsonl')] + '.ser'


class SerFollower:
    """Follow one stream within a session directory, always chasing the newest frame."""

    def __init__(self, session_dir, role):
        self.session_dir = str(session_dir)
        self.role = role
        self._fo = framestream.StreamFollower(session_dir, role)

    def _newest(self):
        self._fo.poll()
        return self._fo.segs[-1] if self._fo.segs else None

    def committed_count(self):
        seg = self._newest()
        return seg.committed() if seg is not None else 0

    def latest_ref(self):
        """FrameRef of the newest committed frame (or None), without reading its pixels."""
        self._fo.poll()
        got = self._fo.latest()
        return FrameRef(_seg_id(got[0]), got[1]) if got else None

    def read_frame(self, ref):
        """Read the frame named by a FrameRef. Refuses (IndexError) a ref for a segment we've
        rolled past and released -- never silently the wrong frame from a fresher segment."""
        for seg in self._fo.segs:
            if _seg_id(seg) == ref.ser_path:
                return seg.read(ref.index)
        raise IndexError(f"frame {ref.index} is from a released segment {ref.ser_path}")

    def read_latest(self):
        """(FrameRef, frame) for the newest committed frame, or None if none yet."""
        self._fo.poll()
        got = self._fo.latest()
        if not got:
            return None
        seg, i = got
        return FrameRef(_seg_id(seg), i), seg.read(i)

    @property
    def header(self):
        seg = self._fo.segs[-1] if self._fo.segs else None
        return seg.header if seg is not None else None

    @property
    def ser_path(self):
        """Identity of the currently-followed segment ('<stem>.ser', virtual)."""
        seg = self._newest()
        return _seg_id(seg) if seg is not None else None

    @property
    def frames_path(self):
        """The newest segment's sidecar (the on-disk artifact that grows per frame)."""
        self._fo.poll()
        return self._fo.frames_path

    def close(self):
        self._fo.close()
