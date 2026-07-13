"""
Cam control channel: live setting changes via the control file and stopping. Runs cam.main on
a thread while the test appends control lines. Run directly:

    python -m astrolock.seeker.tests.test_cam_control
"""

import os
import threading
import time

from astrolock.seeker import cam, framestream, sidecar
from astrolock.seeker.tests._util import fresh_dir


def _append(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(obj + '\n')


def test_control_file_drives_cam():
    out = fresh_dir('camctl')
    cf = os.path.join(out, 'control.jsonl')
    open(cf, 'w').close()

    t = threading.Thread(target=cam.main, args=([
        '--role', 'guide', '--out-dir', out, '--width', '64', '--height', '48', '--fps', '60',
        '--frame-limit', '-1', '--control-file', cf,
    ],), daemon=True)
    t.start()

    time.sleep(0.2)
    _append(cf, '{"fps": 120}')              # a live setting change partway through
    time.sleep(0.3)
    _append(cf, '{"stop": true}')            # finish + exit
    t.join(timeout=5.0)
    assert not t.is_alive(), "cam should have stopped"

    # ONE ring for the whole run (no rolls -- settings changes don't split the stream), then a
    # clean ended. File store here (shm on the rig).
    head = sidecar.read_complete_lines(os.path.join(out, 'guide.stream.jsonl'))
    rings = [e for e in head if e.get('event') == 'ring']
    assert len(rings) == 1 and 'data' in rings[0], rings
    assert head[-1] == {'event': 'ended'}
    fo = framestream.StreamFollower(out, 'guide')
    fo.poll()
    assert fo.ended() and fo.committed() > 10       # ~0.5s at 60-120 fps
    fo.close()


if __name__ == '__main__':
    test_control_file_drives_cam()
    print("test_cam_control: OK")
