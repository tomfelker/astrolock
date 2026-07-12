"""
focus_sweep end to end: a fake focus stream (HFD = known hyperbola of position) + a fake
actuator answering position requests, sweep runs for real and must recover the vertex.

    python -m astrolock.seeker.tests.test_focus_sweep
"""

import json
import math
import sys
import threading
import time
import types

import numpy as np

from astrolock.seeker import focus_sweep, framestream
from astrolock.seeker import session as session_mod
from astrolock.seeker.tests._util import fresh_dir

P0, PK0, SLOPE = 3.7, 0.8, 0.9

def _peak_of(pos):                                       # 1/peak exactly quadratic in position
    return PK0 / (1.0 + (SLOPE * (pos - P0)) ** 2)


def test_fit_vcurve():
    pts = [(p, _peak_of(p)) for p in np.linspace(0, 8, 40)]
    p0, pk0, _ = focus_sweep.fit_vcurve(pts)
    assert abs(p0 - P0) < 1e-6 and abs(pk0 - PK0) < 1e-6
    dip = [(0.0, 0.8), (1.0, 0.2), (2.0, 0.8)]           # peak MINIMUM in range: no vertex
    assert focus_sweep.fit_vcurve(dip) is None


def test_focus_sweep():
    out = fresh_dir('sweep')
    sys.stdin = types.SimpleNamespace(readline=lambda: time.sleep(3600))  # park the abort watcher

    th = threading.Thread(target=focus_sweep.main, daemon=True,
                          args=(['--session', out, '--role', 'main',
                                 '--start', '0', '--end', '8', '--steps', '5',
                                 '--frames-per-step', '6', '--settle-s', '0',
                                 '--poll', '0.005'],))
    th.start()

    focus = framestream.FrameStream(out, 'main_focus',
                                    extras=('<8f', ['peak', 'peak_frame', 'hfd', 'strehl',
                                                    'dx', 'dy', 'com_rad_x', 'com_rad_y']))
    focuser = framestream.FrameStream(out, 'main_focuser')
    fo_sweep = framestream.StreamFollower(out, 'main_sweep', keep_all=True)
    frame = np.zeros((8, 8), np.uint16)
    pos = None
    state = {}
    deadline = time.time() + 30
    while th.is_alive() and time.time() < deadline:
        fo_sweep.poll()
        for seg in fo_sweep.segs:                        # latest state blob wins
            i = getattr(seg, '_used', 0)
            while i < seg.committed():
                state = json.loads(bytes(seg.read(i)).decode('utf-8'))
                i += 1
            seg._used = i
        if state.get('awaiting') == 'position' and state.get('want_pos') != pos:
            pos = state['want_pos']                      # "moved": echo the request back
            if focuser._seg is None or focuser.full:
                focuser.open_segment(session_mod.segment_stamp(), 1 << 12, 1, pixel_depth=8,
                                     cap=256, raw=True)
            focuser.write(np.frombuffer(json.dumps({'pos': pos}).encode('utf-8'), np.uint8),
                          t_mono_ns=time.monotonic_ns())
        if pos is not None:                              # the "focus process": known peak curve
            if focus._seg is None or focus.full:
                focus.open_segment(session_mod.segment_stamp(), 8, 8, cap=64)
            focus.write(frame, t_mono_ns=time.monotonic_ns(),
                        extras=(0.5, _peak_of(pos), 20.0, 0.5, 0.0, 0.0, math.nan, math.nan))
        time.sleep(0.005)
    th.join(10)
    assert not th.is_alive(), "sweep never finished"
    fo_sweep.poll()                                      # the final 'done' blob lands at exit
    for seg in fo_sweep.segs:
        i = getattr(seg, '_used', 0)
        while i < seg.committed():
            state = json.loads(bytes(seg.read(i)).decode('utf-8'))
            i += 1
        seg._used = i
    assert state.get('done') and not state.get('aborted'), state
    assert abs(state['p0'] - P0) < 0.05, state
    assert abs(state['peak0'] - PK0) < 0.05, state
    assert len(state['points']) == 5 * 6 and state['bracketed'], state
    assert state['sat_frac'] == 0.0
    focus.close()
    focuser.close()
    fo_sweep.close()


if __name__ == '__main__':
    test_fit_vcurve()
    test_focus_sweep()
    print("test_focus_sweep: OK")
