"""
cam (synthetic) -> follower end to end. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_cam_follower
"""

import glob
import os

import numpy as np

from astrolock.seeker import cam, sidecar
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
    assert ref.index == 19
    assert ref.ser_path.endswith('_guide.ser')     # the index carries the segment it indexes into
    assert frame.shape == (240, 320) and frame.dtype == np.uint16
    assert frame.max() > 50000, f"synthetic blob should be bright, got {frame.max()}"

    frames_path = glob.glob(os.path.join(out, '*_guide.frames.jsonl'))[0]
    recs = sidecar.read_complete_lines(frames_path)
    start, frames = recs[0], [r for r in recs[1:] if 'off' in r]   # line 0 = stream-start metadata;
    assert start['type'] == 'start' and start['store'] == 'file' and start['width'] == 320
    assert recs[-1].get('type') == 'ended'             # ...last line = the clean end-of-stream record
    assert len(frames) == 20
    assert 't_mono_ns' in frames[0] and 't_utc' in frames[0] and 'off' in frames[0]

    # Only frame records (off/len) count as frames -- other line types never do.
    with open(frames_path, 'a', encoding='utf-8') as fp:
        fp.write('{"t_mono_ns":1,"t_utc":"x"}\n')
    assert f.committed_count() == 20, "non-frame lines must not count"
    f.close()


if __name__ == '__main__':
    test_cam_to_follower()
    print("test_cam_follower: OK")
