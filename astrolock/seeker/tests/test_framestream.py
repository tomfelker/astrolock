"""
framestream ring: absolute indices, zero-copy views, lap detection, reconfigure, file store.

    python -m astrolock.seeker.tests.test_framestream
"""

import math
import os

import numpy as np

from astrolock.seeker import framestream
from astrolock.seeker.tests._util import fresh_dir


def _frame(v, shape=(12, 16)):
    return np.full(shape, v, np.uint16)


def test_ring_shm():
    out = fresh_dir('fsr')
    st = framestream.FrameStream(out, 'guide')
    st.configure(16, 12, pixel_depth=16, shm=True, frames=4, meta={'bin': [2, 2]})
    fo = framestream.StreamFollower(out, 'guide')
    for i in range(3):
        st.write(_frame(i), t_mono_ns=1000 + i, src_index=i)
    fo.poll()
    rd = fo.current
    assert rd.committed() == 3 and not rd.finalized()
    assert rd.meta['bin'] == [2, 2] and rd.header.image_width == 16
    r2 = rd.record(2)
    assert r2['t_mono_ns'] == 1002 and r2['src_index'] == 2 and r2['off'] % 4096 == 0
    f2 = rd.read(2)
    assert int(f2[0, 0]) == 2 and not f2.flags.writeable
    with rd.view(1) as (rec, v):                      # zero-copy: a view into the section
        assert rec['src_index'] == 1 and int(v[0, 0]) == 1 and not v.flags.writeable
    del rec, v                                        # a view must not outlive its with-block

    # Lap: cap 4, write through frame 6 -- frames 0..2 are gone, explicitly.
    for i in range(3, 7):
        st.write(_frame(i), t_mono_ns=1000 + i, src_index=i)
    assert rd.committed() == 7 and rd.first_available() == 3
    try:
        rd.read(2)
        assert False, "reading a lapped frame must raise"
    except framestream.Lapped:
        pass
    assert int(rd.read(3)[0, 0]) == 3 and int(rd.read(6)[0, 0]) == 6

    # A fresh sequential consumer skips the lapped frames and COUNTS them.
    fo2 = framestream.StreamFollower(out, 'guide')
    fo2.poll()
    got = [(i, int(r.read(i)[0, 0])) for r, i in fo2.drain()]
    assert [i for i, _ in got] == [3, 4, 5, 6] and all(i == v for i, v in got)
    assert fo2.lost == 3

    # Reconfigure (geometry change) = a NEW ring; the old one is finalized + immutable, and
    # drain() crosses the boundary in order.
    st.configure(8, 6, pixel_depth=16, shm=True, frames=4)
    st.write(_frame(50, (6, 8)), t_mono_ns=2000, src_index=0)
    fo2.poll()
    assert fo2.current.header.image_width == 8
    more = [(r.header.image_width, i) for r, i in fo2.drain()]
    assert more == [(8, 0)]
    fo.poll()
    tail = [int(r.read(i)[0, 0]) for r, i in fo.drain()]  # fo never drained: 3..6 then the new ring
    assert tail == [3, 4, 5, 6, 50] and fo.lost == 3

    # Clean end: STREAM_ENDED + ended head event; followers self-finish.
    st.close()
    fo.poll()
    assert fo.ended() and fo.current.stream_ended()
    assert not any(p.endswith(('.ser', '.dat', '.frames.jsonl')) for p in os.listdir(out))
    fo.close()
    fo2.close()

    # Once every handle is gone, a re-attach counts the loss instead of crashing.
    fo3 = framestream.StreamFollower(out, 'guide')
    fo3.poll()
    assert fo3.current is None and fo3.lost == 2 and fo3.ended()
    fo3.close()

    # A producer RELAUNCH (Connect / source switch appends to the same head) reopens the
    # stream: 'ended' closed the previous run, not the stream forever.
    st2 = framestream.FrameStream(out, 'guide')
    st2.configure(16, 12, pixel_depth=16, shm=True, frames=4)
    st2.write(_frame(7), t_mono_ns=3000, src_index=0)
    fo4 = framestream.StreamFollower(out, 'guide')
    fo4.poll()
    assert not fo4.ended(), "a new ring after 'ended' must reopen the stream"
    assert [int(r.read(i)[0, 0]) for r, i in fo4.drain()] == [7]
    st2.close()
    fo4.poll()
    assert fo4.ended()
    fo4.close()


def test_extras_raw_and_file_store():
    out = fresh_dir('fsrf')
    st = framestream.FrameStream(out, 'main_focus',
                                 extras=('<3f', ['peak', 'strehl', 'dx']))
    st.configure(16, 12, shm=False, frames=4)
    st.write(_frame(100), src_index=7, flags=1, extras=(0.5, float('nan'), -2.0))
    fo = framestream.StreamFollower(out, 'main_focus')
    fo.poll()
    rd = fo.current
    assert rd.meta['extras'] == ['peak', 'strehl', 'dx']
    r = rd.record(0)
    assert abs(r['peak'] - 0.5) < 1e-6 and math.isnan(r['strehl']) and r['flags'] == 1
    assert int(rd.read(0)[0, 0]) == 100

    # File store: a reader opened EARLY keeps seeing new commits (no stale mapping).
    st.write(_frame(101), src_index=8, extras=(1.0, 2.0, 3.0))
    assert rd.committed() == 2 and int(rd.read(1)[0, 0]) == 101
    st.close()
    fo.poll()
    assert fo.ended() and rd.stream_ended()
    fo.close()

    # Raw payloads: variable-size blobs in fixed slots, bytes out.
    rt = framestream.FrameStream(out, 'guide_det')
    rt.configure(1 << 10, 1, pixel_depth=8, shm=True, frames=8, raw=True)
    rt.write(np.frombuffer(b'{"blobs":[]}', np.uint8), src_index=0)
    rt.write(np.frombuffer(b'{"blobs":[1,2,3]}', np.uint8), src_index=1)
    fo2 = framestream.StreamFollower(out, 'guide_det')
    fo2.poll()
    assert bytes(fo2.current.read(1)) == b'{"blobs":[1,2,3]}'
    assert bytes(fo2.current.read(0)) == b'{"blobs":[]}'
    rt.close()
    fo2.close()


def test_skip_to_latest():
    """A click-started consumer (the recorder) begins at NOW: prior frames are history --
    neither drained nor counted as lost."""
    out = fresh_dir('fskip')
    st = framestream.FrameStream(out, 'guide')
    st.configure(16, 12, pixel_depth=16, shm=True, frames=8)
    for i in range(5):
        st.write(_frame(i), t_mono_ns=1000 + i, src_index=i)
    fo = framestream.StreamFollower(out, 'guide')
    fo.skip_to_latest()
    assert list(fo.drain()) == [] and fo.lost == 0     # history: not backlog, not loss
    st.write(_frame(5), t_mono_ns=1005, src_index=5)
    st.write(_frame(6), t_mono_ns=1006, src_index=6)
    fo.poll()
    got = [(i, int(r.read(i)[0, 0])) for r, i in fo.drain()]
    assert got == [(5, 5), (6, 6)] and fo.lost == 0
    st.close()
    fo.close()


if __name__ == '__main__':
    test_ring_shm()
    test_extras_raw_and_file_store()
    test_skip_to_latest()
    print("test_framestream: OK")
