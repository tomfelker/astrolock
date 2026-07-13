"""
cam (synthetic) -> follower end to end. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_cam_follower
"""

import os

import numpy as np

from astrolock.seeker import cam, framestream, sidecar
from astrolock.seeker.follower import SerFollower
from astrolock.seeker.tests._util import fresh_dir


def test_cam_to_follower():
    out = fresh_dir('cam')

    cam.main(['--role', 'guide', '--out-dir', out, '--width', '320', '--height', '240',
              '--fps', '120', '--frame-limit', '20'])

    f = SerFollower(out, 'guide')
    assert f.committed_count() == 20

    res = f.read_latest()
    assert res is not None
    ref, frame = res
    assert ref.index == 19                         # ABSOLUTE index (never restarts)
    assert ref.ser_path.endswith('.ser')           # ring identity (ident + '.ser', virtual)
    assert frame.shape == (240, 320) and frame.dtype == np.uint16
    assert frame.max() > 50000, f"synthetic blob should be bright, got {frame.max()}"
    with f.view(ref) as (rec, v):                  # zero-copy path agrees with the copy path
        assert rec['src_index'] == 19 and v.shape == (240, 320)
    del rec, v                                     # a view must not outlive its with-block

    # Records are binary structs inside the ring; the head file announces the ring + ended.
    fo = framestream.StreamFollower(out, 'guide')
    fo.poll()
    rd = fo.current
    assert rd.meta['width'] == 320 and rd.committed() == 20
    assert rd.finalized() and rd.stream_ended()
    r = rd.record(19)
    assert r['t_mono_ns'] > 0 and r['t_utc_ns'] > 0 and r['src_index'] == 19
    assert r['off'] % 4096 == 0
    head = sidecar.read_complete_lines(os.path.join(out, 'guide.stream.jsonl'))
    assert head[-1] == {'event': 'ended'} and any('data' in e for e in head)
    fo.close()
    f.close()


if __name__ == '__main__':
    test_cam_to_follower()
    print("test_cam_follower: OK")
