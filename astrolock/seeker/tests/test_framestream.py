"""
framestream v3: binary in-section records, head-file chains, shm + file stores.

    python -m astrolock.seeker.tests.test_framestream
"""

import math
import os

import numpy as np

from astrolock.seeker import framestream, ser
from astrolock.seeker.tests._util import fresh_dir


def _frame(v):
    return np.full((12, 16), v, np.uint16)


def test_stream_chain_shm():
    out = fresh_dir('fs3')
    st = framestream.FrameStream(out, 'guide')
    st.open_segment('20990101T000001000Z', 16, 12, pixel_depth=16, shm=True, cap=4,
                    meta={'bin': [2, 2]})
    fo = framestream.StreamFollower(out, 'guide', keep_all=True)
    for i in range(3):
        st.write(_frame(i), t_mono_ns=1000 + i, src_index=i)
    fo.poll()
    seg = fo.segs[0]
    assert seg.committed() == 3 and not seg.finalized()
    assert seg.meta['bin'] == [2, 2] and seg.header.image_width == 16
    r2 = seg.record(2)
    assert r2['t_mono_ns'] == 1002 and r2['src_index'] == 2 and r2['off'] % 4096 == 0
    f2 = seg.read(2)
    assert int(f2[0, 0]) == 2 and not f2.flags.writeable

    # Roll: fresh section; the old one is FINALIZED (header flag), retained across the roll.
    st.open_segment('20990101T000002000Z', 16, 12, shm=True, cap=4)
    st.write(_frame(10), src_index=0)
    fo.poll()
    old, new = fo.segs
    assert old.finalized() and not old.stream_ended()
    assert int(old.read(1)[0, 0]) == 1 and int(new.read(0)[0, 0]) == 10
    assert not any(p.endswith(('.ser', '.dat', '.frames.jsonl')) for p in os.listdir(out))

    # Clean end: STREAM_ENDED flag + ended head event; followers self-finish.
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
    assert fo3.segs == [] and fo3.lost == 2 and fo3.ended()
    fo3.close()


def test_extras_and_file_store():
    out = fresh_dir('fs3f')
    st = framestream.FrameStream(out, 'main_focus',
                                 extras=('<3f', ['peak', 'strehl', 'dx']))
    st.open_segment('20990101T000001000Z', 16, 12, shm=False, cap=4)
    st.write(_frame(100), src_index=7, flags=1, extras=(0.5, float('nan'), -2.0))
    fo = framestream.StreamFollower(out, 'main_focus', keep_all=True)
    fo.poll()
    seg = fo.segs[0]
    assert seg.meta['extras'] == ['peak', 'strehl', 'dx']
    r = seg.record(0)
    assert abs(r['peak'] - 0.5) < 1e-6 and math.isnan(r['strehl']) and r['flags'] == 1
    assert int(seg.read(0)[0, 0]) == 100

    # File store: a reader opened EARLY keeps seeing new commits (no stale mapping).
    st.write(_frame(101), src_index=8, extras=(1.0, 2.0, 3.0))
    assert seg.committed() == 2 and int(seg.read(1)[0, 0]) == 101
    st.close()
    fo.poll()
    assert fo.ended() and seg.stream_ended()
    fo.close()


if __name__ == '__main__':
    test_stream_chain_shm()
    test_extras_and_file_store()
    print("test_framestream: OK")
