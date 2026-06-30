"""
Closed-loop unit tests for the pixel tracker, with a simple pixel-domain plant model.

The mount is an integrator (rate -> angle), and image position relates to (source - boresight):
    d(px)/dt = (v_sky_az  - rate_az ) / rad_per_px
    d(py)/dt = -(v_sky_alt - rate_alt) / rad_per_px
The controller clamps its own output to max_rate, so we feed the returned rate straight back.

Covers: (1) in-band tracking converges with zero steady-state lag and correct signs; (2) the
dead-zoned derivative term brakes a hard acquisition slew so the target stays gated and image
speed stays bounded; (3) the dead zone does NOT throttle a locked fast mover; (4) lost fallback.
"""

import math

from astrolock.seeker.controller import PixelTracker

RADPP = 0.00025          # rad/px (sim optics: 2um pitch / 8mm focal)
CX, CY = 960.0, 540.0


def run_plant(dt, steps, start_off=0.0, v_saz=0.0, v_salt=0.0, kd=1.0, thresh=120.0):
    trk = PixelTracker(CX, CY, RADPP, ki=0.5, damping=1.3, kd=kd, max_track_px_s=thresh,
                       gate_px=80.0, max_rate_rad_s=math.radians(8.0))
    px, py = CX + start_off, CY
    trk.start(px, py, 0.0)
    t = 0.0
    raz = ralt = 0.0
    prev = (px, py)
    peak_after = 0.0
    lost = False
    for k in range(steps):
        t += dt
        px += (v_saz - raz) / RADPP * dt
        py += -(v_salt - ralt) / RADPP * dt
        speed = math.hypot(px - prev[0], py - prev[1]) / dt
        if k >= 2:                                   # ignore the first (unbraked) frame
            peak_after = max(peak_after, speed)
        prev = (px, py)
        raz, ralt, status, _ = trk.update([{'px': [px, py]}], True, t)
        if status == 'lost':
            lost = True
    return dict(final_err=max(abs(px - CX), abs(py - CY)), peak_after=peak_after,
                lost=lost, final_rate=math.hypot(raz, ralt))


def run():
    # 1. In-band tracking: converges to centre, integral settles to the true sky rate.
    truth = math.hypot(0.02, 0.012)
    r = run_plant(0.05, 600, start_off=100.0, v_saz=0.02, v_salt=-0.012)
    assert not r['lost'] and r['final_err'] < 3.0, r
    assert abs(r['final_rate'] - truth) < 0.05 * truth, r
    print(f"test_controller: in-band converge residual {r['final_err']:.2f}px, "
          f"rate {r['final_rate']:.4f} vs {truth:.4f}: OK")

    # 2. Dead-zone D brakes a hard acquisition slew: far static target still converges without
    #    being lost, and sustained image speed is lower than the un-braked case. (Exact braking
    #    is a tunable -- we only assert it helps and doesn't break tracking.)
    fast = run_plant(0.1, 300, start_off=300.0, kd=0.0)     # no braking
    slow = run_plant(0.1, 300, start_off=300.0, kd=1.0)     # braking
    assert not slow['lost'] and slow['final_err'] < 5.0, slow
    assert slow['peak_after'] < fast['peak_after'], (slow['peak_after'], fast['peak_after'])
    print(f"test_controller: acquisition braking sustained {slow['peak_after']:.0f}px/s "
          f"vs un-braked {fast['peak_after']:.0f}px/s, residual {slow['final_err']:.2f}px: OK")

    # 3. Dead zone does NOT throttle a locked fast mover: it's matched at a high mount rate.
    r = run_plant(0.1, 400, start_off=0.0, v_saz=0.10, v_salt=0.0)
    thr_equiv = 120.0 * RADPP                                # dead-zone speed as a rate (rad/s)
    assert not r['lost'] and r['final_err'] < 5.0, r
    assert r['final_rate'] > 2 * thr_equiv, (r['final_rate'], thr_equiv)
    assert abs(r['final_rate'] - 0.10) < 0.1 * 0.10, r['final_rate']
    print(f"test_controller: fast mover locked at {r['final_rate']:.4f} rad/s "
          f"(>> dead-zone {thr_equiv:.4f}): OK")

    # 3b. Diagnostics self-check: clean tune is quiet; an over-strong kii or too-low framerate warn.
    trk = PixelTracker(CX, CY, RADPP, ki=0.3, damping=1.3, kd=1.0, kii=0.0)
    info, warn = trk.diagnostics()                               # no arg -> checks at nominal rate
    assert any('nominal' in ln for ln in info) and not warn, (info, warn)
    _, warn = PixelTracker(CX, CY, RADPP, ki=0.3, damping=1.3, kii=0.6).diagnostics()
    assert any('kii' in w for w in warn), warn                   # kii past kp*ki -> flagged
    _, warn = trk.diagnostics(frame_dt=1 / 3.0)                  # measured-rate override
    assert any('fps' in w for w in warn), warn                   # 3 fps -> framerate margin flagged
    print("test_controller: diagnostics (quiet when good, warns on aggressive kii / low fps): OK")

    # 4. Lost fallback: no matching blobs -> 'lost' and zero rate.
    trk = PixelTracker(CX, CY, RADPP)
    trk.start(CX, CY, 0.0)
    t = 0.0
    status = 'track'
    for _ in range(int((trk.lost_s + 0.5) / 0.05)):
        t += 0.05
        raz, ralt, status, _ = trk.update([], True, t)
    assert status == 'lost' and raz == 0.0 and ralt == 0.0, status
    print("test_controller: lost fallback: OK")

    # 5. Coast vs stop on loss: lock a constant-velocity target until settled, then stop feeding.
    #    Settled lock -> 'coast' at the last (nonzero) rate; with coast disabled -> 'lost' at zero.
    def run_then_lose(lock_drift):
        trk = PixelTracker(CX, CY, RADPP, ki=0.5, damping=1.3, gate_px=120.0, lost_s=0.5,
                           lock_max_drift_rate=lock_drift, lock_min_time=0.3, max_rate_rad_s=math.radians(20))
        px, py, t, raz = CX, CY, 0.0, 0.0
        trk.start(px, py, t)
        for _ in range(60):                       # lock + settle on a steady sky-rate target
            t += 0.05
            px += (0.02 - raz) / RADPP * 0.05     # plant: image moves by (sky_rate - mount)/scale
            raz, ralt, status, _ = trk.update([{'px': [px, py]}], True, t)
        coast_rates = []
        for _ in range(15):                       # stop feeding -> lose the target
            t += 0.05
            raz, ralt, status, _ = trk.update([], True, t)
            if status == 'coast':
                coast_rates.append(raz)
        return status, raz, coast_rates
    status, raz, coast = run_then_lose(0.5)       # coast enabled
    assert status == 'coast' and len(coast) >= 2, (status, len(coast))
    assert abs(coast[0]) > 1e-4 and max(coast) - min(coast) < 1e-9, coast   # nonzero, held constant
    status, raz, coast = run_then_lose(0.0)       # coast disabled -> stop
    assert status == 'lost' and raz == 0.0 and coast == [], (status, raz)
    print("test_controller: coast holds a constant nonzero rate on settled loss, stops when disabled: OK")

    # 6. Low-framerate gain derate: below nominal the proportional rate for a fixed error is scaled
    #    down; disabling derate restores the full gain.
    def p_rate(dt, derate):
        trk = PixelTracker(CX, CY, RADPP, ki=0.5, damping=1.3, kd=0.0, nominal_rate_hz=10.0,
                           derate=derate, lock_max_drift_rate=0.0, gate_px=200.0,
                           max_rate_rad_s=math.radians(60))
        trk.start(CX, CY, 0.0)
        t = 0.0
        for _ in range(20):                       # warm dt_ema to dt, on target (no error, no integral)
            t += dt
            trk.update([{'px': [CX, CY]}], True, t)
        t += dt
        raz, _, _, _ = trk.update([{'px': [CX + 60, CY]}], True, t)   # one off-target frame
        return abs(raz)
    fast = p_rate(0.1, True)                       # 10 fps = nominal -> full gain
    slow = p_rate(0.5, True)                       # 2 fps -> derated (~1/5)
    nod = p_rate(0.5, False)                       # 2 fps, derate off -> full gain
    assert slow < 0.5 * fast, (slow, fast)
    assert nod > 1.8 * slow, (nod, slow)
    print(f"test_controller: low-framerate derate (rate {fast:.4f}->{slow:.4f}, off={nod:.4f}): OK")


if __name__ == '__main__':
    run()
