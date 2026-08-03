"""
focus_sweep end to end: a fake focus stream (stack Strehl = known 1/quadratic of position), a
fake actuator answering position requests, and a fake BACKEND writing the per-bucket stack
resets (control_focus_main.jsonl) when the sweep enters 'frames' -- the sweep runs for real
and must recover the vertex from one Strehl per bucket.

    python -m astrolock.seeker.tests.test_focus_sweep
"""

import json
import os
import sys
import threading
import time
import types

import numpy as np

from astrolock.seeker import focus_sweep, framestream
from astrolock.seeker import session as session_mod
from astrolock.seeker.sidecar import JsonlWriter
from astrolock.seeker.tests._util import fresh_dir

P0, S0, SLOPE = 3.7, 0.8, 0.9

EXTRAS = ('<16f', ['stack_peak', 'stack_strehl', 'stack_hfd', 'stack_n', 'ctl_seq', 'clip_px',
                   'ellipse_1', 'ellipse_2', 'skew_x', 'skew_y',
                   'instant_ellipse_1', 'instant_ellipse_2',
                   'instant_skew_x', 'instant_skew_y', 'skew_rad_x', 'skew_rad_y'])


def _strehl_of(pos):                                     # 1/strehl exactly quadratic in position
    return S0 / (1.0 + (SLOPE * (pos - P0)) ** 2)


def test_sweep_order():
    assert focus_sweep.sweep_order(5) == [0, 4, 1, 3, 2]     # extremes first, alternating inward
    assert focus_sweep.sweep_order(4) == [0, 3, 1, 2]
    assert sorted(focus_sweep.sweep_order(9)) == list(range(9))


def test_fit_vcurve():
    pts = [(p, _strehl_of(p)) for p in np.linspace(0, 8, 9)]
    p0, s0, _ = focus_sweep.fit_vcurve(pts)
    assert abs(p0 - P0) < 1e-6 and abs(s0 - S0) < 1e-6
    dip = [(0.0, 0.8), (1.0, 0.2), (2.0, 0.8)]           # strehl MINIMUM in range: no vertex
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

    focus = framestream.FrameStream(out, 'main_focus', extras=EXTRAS)
    focus.configure(8, 8, frames=64)
    focuser = framestream.FrameStream(out, 'main_focuser')
    focuser.configure(1 << 12, 1, pixel_depth=8, frames=256, raw=True)
    ctl = JsonlWriter(os.path.join(out, 'control_focus_main.jsonl'))
    fo_sweep = framestream.StreamFollower(out, 'main_sweep')
    frame = np.zeros((8, 8), np.uint16)
    pos = None
    seq = 0                          # the fake backend's control seq / the fake stack's ctl_seq
    n = 0                            # the fake stack's frame count
    reset_step = None
    state = {}
    visited = []
    deadline = time.time() + 30
    while th.is_alive() and time.time() < deadline:
        fo_sweep.poll()
        for rd, i in fo_sweep.drain():                   # latest state blob wins
            state = json.loads(bytes(rd.read(i)).decode('utf-8'))
        if state.get('awaiting') == 'position' and state.get('want_pos') != pos:
            pos = state['want_pos']                      # "moved": echo the request back
            visited.append(pos)
            focuser.write(np.frombuffer(json.dumps({'pos': pos}).encode('utf-8'), np.uint8),
                          t_mono_ns=session_mod.mono_ns())
        if (state.get('awaiting') == 'frames' and state.get('step') != reset_step):
            reset_step = state['step']                   # fake BACKEND: bucket -> stack reset
            seq += 1
            n = 0
            ctl.append({'reset': 1, 'average': 1, 'seq': seq})
        if pos is not None:                              # fake focus process: known Strehl curve
            n += 1
            focus.write(frame, t_mono_ns=session_mod.mono_ns(),
                        extras=(0.5, _strehl_of(pos), 20.0, float(n), float(seq), 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        time.sleep(0.005)
    th.join(10)
    assert not th.is_alive(), "sweep never finished"
    fo_sweep.poll()                                      # the final 'done' blob lands at exit
    for rd, i in fo_sweep.drain():
        state = json.loads(bytes(rd.read(i)).decode('utf-8'))
    assert state.get('done') and not state.get('aborted'), state
    assert abs(state['p0'] - P0) < 0.05, state
    assert abs(state['strehl0'] - S0) < 0.05, state
    assert len(state['points']) == 5 and state['bracketed'], state
    assert state['clip_frac'] == 0.0
    assert visited == [0.0, 8.0, 2.0, 6.0, 4.0], visited     # alternating sides, extremes first
    focus.close()
    focuser.close()
    ctl.close()
    fo_sweep.close()


if __name__ == '__main__':
    test_sweep_order()
    test_fit_vcurve()
    test_focus_sweep()
    print("test_focus_sweep: OK")
