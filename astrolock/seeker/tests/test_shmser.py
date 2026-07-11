"""
Shared-memory .ser segments: the marker file redirects a plain SerReader into the section,
frames round-trip, commit semantics match the file contract. Runs as pytest or directly:

    python -m astrolock.seeker.tests.test_shmser
"""

import os

import numpy as np

from astrolock.seeker import framestream, ser, shmser
from astrolock.seeker.tests._util import fresh_dir


def test_shmser_roundtrip_via_open_reader():
    out = fresh_dir('shmser')
    path = os.path.join(out, '20990101T000000000Z_guide.ser')
    w = shmser.ShmSerWriter(path, 32, 24, color_id=ser.ColorId.MONO,
                            pixel_depth_per_plane=16, cap=8)
    assert not os.path.exists(path)                      # NOTHING lands on disk

    # (No sidecar at this low level -- attach the section directly. framestream.open_reader
    # resolves via the sidecar's 'store' field; see test_framestream_write_and_retention.)
    r = shmser.ShmSerReader(path)
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

    # Finalize patches the region header like SerWriter patches the file; an attached reader
    # keeps the section alive and keeps reading after the writer is gone.
    w.close()
    assert r.finalized()
    assert int(r.read_frame(7)[0, 0]) == 107
    r.close()

    # With the last handle closed the section evaporates: a late reader gets ValueError
    # (the same class as an unreadable/half-written file, which followers already handle).
    try:
        shmser.ShmSerReader(path)
        raise AssertionError("dead section should raise")
    except (ValueError, FileNotFoundError):
        pass


def test_framestream_write_and_retention():
    out = fresh_dir('framestream')
    st = framestream.FrameStream(out, 'guide')
    st.open_segment('20990101T000001000Z', 16, 12, shm=True, cap=4)
    first_ser = st.ser_path
    for i in range(3):
        st.write(np.full((12, 16), i, np.uint16), index=i)
    # Roll: the previous section must SURVIVE the roll (retained by the writer) so a reader
    # that hadn't attached yet doesn't lose the race.
    st.open_segment('20990101T000002000Z', 16, 12, shm=True, cap=4)
    r = framestream.open_reader(first_ser)               # attach AFTER the roll
    assert r.frames_on_disk() == 3 and r.finalized()
    assert int(r.read_frame(2)[0, 0]) == 2
    r.close()
    st.write(np.zeros((12, 16), np.uint16), index=0)
    # Discovery: sidecars are the globbable spine, records self-describe the store.
    assert len(framestream.sidecar_glob(out, 'guide')) == 2
    from astrolock.seeker import sidecar as sc
    recs = sc.read_complete_lines(framestream.sidecar_glob(out, 'guide')[0])
    assert recs and all(rec['store'] == 'shm' for rec in recs)
    st.close()


if __name__ == '__main__':
    test_shmser_roundtrip_via_open_reader()
    test_framestream_write_and_retention()
    print("test_shmser: OK")
