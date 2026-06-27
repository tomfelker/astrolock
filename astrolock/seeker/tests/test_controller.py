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
    trk = PixelTracker(CX, CY, RADPP, kp=1.5, ki=0.5, kd=kd, max_track_px_s=thresh,
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


if __name__ == '__main__':
    run()
