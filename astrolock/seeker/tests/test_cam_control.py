"""
Cam control channel: live setting changes via the control file -- toggling `important`,
forcing a rollover by lowering frame-limit, and stopping. Runs cam.main on a thread while
the test appends control lines. Run directly:

    python -m astrolock.seeker.tests.test_cam_control
"""

import glob
import os
import threading
import time

from astrolock.seeker import cam, sidecar
from astrolock.seeker.tests._util import fresh_dir


def _append(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(obj + '\n')


def test_control_file_drives_cam():
    out = fresh_dir('camctl')
    cf = os.path.join(out, 'control.jsonl')
    open(cf, 'w').close()

    # 10 frames/file rolls every ~0.17 s at 60 fps -> a handful of segments.
    t = threading.Thread(target=cam.main, args=([
        '--role', 'guide', '--out-dir', out, '--width', '64', '--height', '48', '--fps', '60',
        '--file-limit', '-1', '--frame-limit', '10', '--control-file', cf,
    ],), daemon=True)
    t.start()

    time.sleep(0.2)
    _append(cf, '{"fps": 60}')               # a live setting change partway through
    time.sleep(0.3)
    _append(cf, '{"stop": true}')            # finish + exit
    t.join(timeout=5.0)
    assert not t.is_alive(), "cam should have stopped"

    head = sidecar.read_complete_lines(os.path.join(out, 'guide.stream.jsonl'))
    segs = [e for e in head if e.get('event') == 'segment']
    assert len(segs) >= 2, f"frame-limit 10 should have rolled to several segments (got {len(segs)})"
    assert all('data' in e for e in segs)           # file store here (shm on the rig)
    assert head[-1] == {'event': 'ended'}
    # No 'important' concept anymore: recording is the recorder process's job.


if __name__ == '__main__':
    test_control_file_drives_cam()
    print("test_cam_control: OK")
