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
        '--file-limit', '-1', '--frame-limit', '10', '--important', '0', '--control-file', cf,
    ],), daemon=True)
    t.start()

    time.sleep(0.2)
    _append(cf, '{"important": 1}')          # start "recording" partway through
    time.sleep(0.3)
    _append(cf, '{"stop": true}')            # finish + exit
    t.join(timeout=5.0)
    assert not t.is_alive(), "cam should have stopped"

    sers = sorted(glob.glob(os.path.join(out, '*_guide.ser')))
    assert len(sers) >= 2, f"frame-limit 10 should have rolled to several files (got {len(sers)})"

    recs = [r for sp in sers
            for r in sidecar.read_complete_lines(sp[:-len('.ser')] + '.frames.jsonl')]
    assert any(r['important'] for r in recs), "some frames should be important"
    assert any(not r['important'] for r in recs), "some (pre-record) frames should not be"


if __name__ == '__main__':
    test_control_file_drives_cam()
    print("test_cam_control: OK")
