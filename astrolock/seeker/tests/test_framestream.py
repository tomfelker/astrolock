"""
framestream v2: chained sidecars + head files + frame-heap stores (shm and file).

    python -m astrolock.seeker.tests.test_framestream
"""

import os

import numpy as np

from astrolock.seeker import framestream, ser
from astrolock.seeker.tests._util import fresh_dir


def _frame(v):
    return np.full((12, 16), v, np.uint16)


def test_stream_chain_shm():
    out = fresh_dir('fs2')
    st = framestream.FrameStream(out, 'guide')
    st.open_segment('20990101T000001000Z', 16, 12, pixel_depth=16, shm=True, cap=4)
    fo = framestream.StreamFollower(out, 'guide', keep_all=True)
    for i in range(3):
        st.write(_frame(i), index=i)
    fo.poll()
    assert [s.committed() for s in fo.segs] == [3]
    seg, k = fo.latest()
    assert k == 2 and int(seg.read(2)[0, 0]) == 2 and not seg.read(2).flags.writeable
    assert seg.header.image_width == 16 and not seg.finalized()

    # Roll: new section, old sidecar chains 'next'; NOTHING but sidecars on disk.
    st.open_segment('20990101T000002000Z', 16, 12, shm=True, cap=4)
    st.write(_frame(10), index=0)
    fo.poll()
    assert len(fo.segs) == 2
    old, new = fo.segs
    assert old.finalized() and old.next_sidecar() and not old.stream_ended()
    assert int(old.read(1)[0, 0]) == 1                    # retained across the roll
    assert int(new.read(0)[0, 0]) == 10
    assert not any(p.endswith(('.ser', '.dat')) for p in os.listdir(out))

    # Clean end: 'ended' in sidecar AND head; followers self-finish.
    st.close()
    fo.poll()
    assert fo.ended() and fo.segs[-1].stream_ended()

    # A late follower attaches while WE still hold the sections (reader = the buffer)...
    fo2 = framestream.StreamFollower(out, 'guide', keep_all=True)
    fo2.poll()
    assert sum(s.committed() for s in fo2.segs) == 4 and fo2.ended()
    fo2.close()
    fo.close()

    # ...and once every handle is gone, a re-attach counts the loss instead of crashing.
    fo3 = framestream.StreamFollower(out, 'guide', keep_all=True)
    fo3.poll()
    assert fo3.segs == [] and fo3.lost == 4 and fo3.ended()
    fo3.close()


def test_stream_file_store():
    out = fresh_dir('fs2f')
    st = framestream.FrameStream(out, 'main')
    st.open_segment('20990101T000001000Z', 16, 12, shm=False, cap=4)
    for i in range(2):
        st.write(_frame(100 + i), index=i)
    fo = framestream.StreamFollower(out, 'main', keep_all=True)
    fo.poll()
    seg = fo.segs[0]
    assert seg.meta['store'] == 'file' and int(seg.read(1)[0, 0]) == 101
    assert seg.recs[1]['off'] % 4096 == 0                 # 4K-aligned heap offsets
    st.close()
    fo.poll()
    assert fo.ended()
    fo.close()


if __name__ == '__main__':
    test_stream_chain_shm()
    test_stream_file_store()
    print("test_framestream: OK")
