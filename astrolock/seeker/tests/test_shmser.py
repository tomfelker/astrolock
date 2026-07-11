"""
Shared-memory .ser segments: the marker file redirects a plain SerReader into the section,
frames round-trip, commit semantics match the file contract. Runs as pytest or directly:

    python -m astrolock.seeker.tests.test_shmser
"""

import os

import numpy as np

from astrolock.seeker import ser, shmser
from astrolock.seeker.tests._util import fresh_dir


def test_shmser_roundtrip_via_serreader():
    out = fresh_dir('shmser')
    path = os.path.join(out, '20990101T000000000Z_guide.ser')
    w = shmser.ShmSerWriter(path, 32, 24, color_id=ser.ColorId.MONO,
                            pixel_depth_per_plane=16, cap=8)
    assert os.path.getsize(path) == ser.HEADER_SIZE      # only the marker touches the disk

    # A plain SerReader on the marker transparently reads the section.
    r = ser.SerReader(path)
    assert r.frames_on_disk() == 0
    for i in range(5):
        w.write_frame(np.full((24, 32), 100 + i, np.uint16))
        assert r.frames_on_disk() == i + 1               # publish-after-bytes commit point
    f3 = r.read_frame(3)
    assert f3.shape == (24, 32) and int(f3[0, 0]) == 103
    assert not f3.flags.writeable                        # SerReader parity: read-only
    try:
        r.read_frame(5)
        raise AssertionError("uncommitted frame should raise")
    except IndexError:
        pass

    # Capacity is a hard segment boundary -- the cam rolls before hitting it.
    for i in range(5, 8):
        w.write_frame(np.full((24, 32), 100 + i, np.uint16))
    try:
        w.write_frame(np.zeros((24, 32), np.uint16))
        raise AssertionError("beyond cap should raise")
    except ValueError:
        pass

    # Finalize patches the marker like SerWriter patches the file; an attached reader keeps
    # the section alive and keeps reading after the writer is gone.
    w.close()
    assert int(np.frombuffer(open(path, 'rb').read(), np.int32,
                             count=1, offset=ser.FRAME_COUNT_OFFSET)[0]) == 8
    assert int(r.read_frame(7)[0, 0]) == 107
    r.close()

    # With the last handle closed the section evaporates: a late reader gets ValueError
    # (the same class as an unreadable/half-written file, which followers already handle).
    try:
        ser.SerReader(path)
        raise AssertionError("dead section should raise")
    except ValueError:
        pass


if __name__ == '__main__':
    test_shmser_roundtrip_via_serreader()
    print("test_shmser: OK")
