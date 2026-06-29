"""
Pixel-space tracking controller.

Closes the loop in guide-camera pixels (no world model, per the design): given the stream of
detected blobs, follow one locked target and emit mount axis rates that drive it to the
boresight (image centre for now -- calibration comes later).

Two parts:
  - association: each new detection, snap to the nearest blob to the predicted target
    position (gated); between detections, coast on the estimated image velocity. Drop to
    'lost' if nothing matches for a while.
  - control: PI on the position error plus dead-zoned derivative braking. The mount is itself
    an integrator (rate -> angle), so PI makes the loop a double integrator -> zero steady-state
    lag chasing a constant-velocity target: the integral settles to exactly the target's sky
    rate while the error goes to zero. The D term opposes only the image speed *above* a
    threshold (vx,vy beyond +/-v_thresh), so it slows the hard acquisition slew (keeping the
    target inside the gate and bounding motion blur) but is inactive once locked -- where the
    target is held still in-frame, so the mount may still slew fast to follow a quick mover.
    (Raw image-velocity feedforward is deliberately not added as in-band rate: the in-band
    image velocity is the residual rate error and feeding it back oscillates. The smoothed
    velocity feeds the dead-zoned brake, the prediction, and the association gate.)

    An optional second integral term (kii, off by default) integrates the position error twice.
    Plain PI drives the steady-state lag to zero for a constant-*velocity* target but leaves a
    residual lag proportional to acceleration -- which is what makes the cam lag then lead a
    satellite as it accelerates overhead. The second integral removes that constant-acceleration
    lag too. Keep it weak: stability (Routh-Hurwitz on s^3 + kp s^2 + ki s + kii) needs
    kii < kp*ki. It clamps and freezes-on-lost exactly like the first integral.

Sign/scale convention (gnomonic sim, camera upright, az->+x, alt->-y):
    rate_az  = +rad_per_px * (kp * ex + ix + kd * excess(vx))
    rate_alt = -rad_per_px * (kp * ey + iy + kd * excess(vy))
where (ex,ey) = target - centre, (ix,iy) = integral of position error (px/s), (vx,vy) = image
velocity (px/s), and excess() is the signed amount beyond the +/-v_thresh dead zone. If a real
rig is mirrored/rotated, flip via the sign args (that, and a proper scale, is calibration).
"""

import math


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _excess(v, thr):
    """Signed amount by which v exceeds the +/-thr dead zone (0 inside the zone)."""
    if v > thr:
        return v - thr
    if v < -thr:
        return v + thr
    return 0.0


class PixelTracker:
    def __init__(self, cx, cy, rad_per_px, ki=0.3, damping=1.3, kd=1.0, kii=0.0,
                 nominal_rate_hz=10.0, gate_px=80.0,
                 lost_s=1.5, vel_smoothing=0.1, max_track_px_s=120.0, max_rate_rad_s=math.radians(8.0),
                 sign_az=1.0, sign_alt=-1.0):
        self.cx, self.cy = cx, cy
        self.rad_per_px = rad_per_px
        self.ki, self.kd, self.kii = ki, kd, kii
        # The framerate the gains are characterized at: the diagnostics check the tune at this rate,
        # and (later, optionally) the effective gains derate below it / buff above it.
        self.nominal_rate_hz = nominal_rate_hz
        # For the loop (mount = integrator, PI control) the position error obeys
        # e'' + kp e' + ki e = 0, so kp = 2*sqrt(ki) is critically damped. `damping` >= 1 pushes
        # it slightly over-damped for margin against the real system's lags (mount update rate,
        # frame latency). I is kept modest for the same reason -- too much integral oscillates.
        self.kp = damping * 2.0 * math.sqrt(ki)
        self.gate_px = gate_px
        self.lost_s = lost_s
        self.vel_smoothing = vel_smoothing        # 0 = trust each new velocity fully; higher = smoother
        self.v_thresh = max_track_px_s           # dead-zone: brake image speed above this
        self.max_rate = max_rate_rad_s           # mount rate clamp
        self.i_clamp = max_rate_rad_s / rad_per_px   # integral alone can't exceed max motor rate
        # The second integral is fed by the first: integ holds ki*(int e), so (int int e) = integ/ki,
        # and the kii term accumulates kii*(integ/ki)*dt -> exactly kii*(int int e). Precompute the
        # per-step factor; if ki is 0 there's no first integral to build on, so disable kii.
        self.kii_step = (kii / ki) if ki > 0 else 0.0
        self.sign_az, self.sign_alt = sign_az, sign_alt
        self.active = False

    def diagnostics(self, frame_dt=None):
        """Bandwidth + stability summary for the current gains, as (info_lines, warnings) -- lists
        of strings the caller prints. Pure (no I/O). The sample-rate check uses frame_dt (s) if
        given, else the nominal rate the gains are characterized at. Warnings flag tunes likely to
        ring or go unstable, so a bad --track-* set is caught at lock time instead of mid-pass."""
        nominal = frame_dt is None
        if nominal and self.nominal_rate_hz > 0:
            frame_dt = 1.0 / self.nominal_rate_hz
        wn = math.sqrt(self.ki) if self.ki > 0 else 0.0            # loop natural frequency (rad/s)
        zeta = (self.kp / (2.0 * wn)) if wn > 0 else float('inf')   # damping ratio (= `damping` arg)
        settle = (4.0 / (zeta * wn)) if (wn > 0 and zeta > 0) else float('inf')   # ~2% settling time
        info = [f"loop: bandwidth {wn:.3f} rad/s ({wn / (2 * math.pi):.3f} Hz), "
                f"damping {zeta:.2f}, settling ~{settle:.1f} s"]
        warn = []
        if zeta < 0.7:
            warn.append(f"damping {zeta:.2f} < 0.7: underdamped, expect overshoot/ringing")

        # Second integral (kii): Routh-Hurwitz on s^3 + kp s^2 + ki s + kii needs kii < kp*ki.
        if self.kii > 0:
            limit = self.kp * self.ki
            frac = self.kii / limit if limit > 0 else float('inf')
            info.append(f"kii {self.kii:.4g} = {frac * 100:.0f}% of the kp*ki={limit:.4g} stability limit")
            if frac >= 1.0:
                warn.append(f"kii {self.kii:.4g} >= kp*ki {limit:.4g}: UNSTABLE second integral "
                            f"(set --track-kii below {limit:.3g})")
            elif frac > 0.5:
                warn.append(f"kii is {frac * 100:.0f}% of its kp*ki limit: little margin, expect slow ringing")

        # Sample-rate margin: kp*dt is ~the fraction of the position error the proportional term
        # commands per frame. Near 1 the loop tries to null the error in a single frame, so any
        # unmodeled lag (detection, mount accel) overshoots and rings. On this rig oscillation set in
        # around kp*dt~0.5, so aim well below. (Heuristic -- refine once cross-process lag is measured.)
        if frame_dt and frame_dt > 0:
            kpt = self.kp * frame_dt
            tag = ' nominal' if nominal else ''
            info.append(f"at {1.0 / frame_dt:.1f} fps{tag}: kp*dt={kpt:.2f} (aim <0.25; oscillation seen ~0.5)")
            if kpt > 0.5:
                warn.append(f"kp*dt={kpt:.2f} at {1.0 / frame_dt:.1f} fps{tag}: oscillation likely -- "
                            f"raise framerate or lower the gains")
            elif kpt > 0.25:
                warn.append(f"kp*dt={kpt:.2f} at {1.0 / frame_dt:.1f} fps{tag}: thin margin at this framerate")
        return info, warn

    def start(self, px, py, now):
        """Lock onto a target at (px, py)."""
        self.active = True
        self.meas = [float(px), float(py)]      # last measured position
        self.vel = [0.0, 0.0]                    # smoothed image velocity (px/s): predict + brake
        self._vel_raw = [0.0, 0.0]                # raw EMA (biased toward 0 early)
        self._vel_w = 0.0                          # EMA of 1's (0 -> 1): warm-start bias correction
        self.integ = [0.0, 0.0]                  # integral of position error (px/s)
        self.integ2 = [0.0, 0.0]                 # second integral (kii term; 0 unless kii > 0)
        self.meas_t = now
        self.good_t = now                        # last successful association
        self.last_t = now

    def stop(self):
        self.active = False

    def estimate(self, now):
        """Predicted target position at `now` (measurement + coasted velocity)."""
        dt = now - self.meas_t
        return self.meas[0] + self.vel[0] * dt, self.meas[1] + self.vel[1] * dt

    def update(self, blobs, new_data, now):
        """
        Advance the tracker and return (rate_az, rate_alt, status, target_px).
          blobs:    current detection blobs (each with 'px':[x,y]); used only if new_data
          new_data: True when these blobs are from a frame not seen before
          status:   'track' (associated/coasting) or 'lost'
        """
        ex_px, ey_px = self.estimate(now)        # predicted position

        if new_data and blobs:
            best = min(blobs, key=lambda b: _dist(b['px'][0], b['px'][1], ex_px, ey_px))
            if _dist(best['px'][0], best['px'][1], ex_px, ey_px) <= self.gate_px:
                mx, my = float(best['px'][0]), float(best['px'][1])
                dt = now - self.meas_t
                if dt > 1e-3:
                    inst = ((mx - self.meas[0]) / dt, (my - self.meas[1]) / dt)
                    # Per-frame EMA (the velocity noise is per detection -- tracker centroid
                    # error -- not per unit time), warm-started by dividing by an EMA of 1s that
                    # rises 0->1 with the same weight (bias correction: the first estimate is the
                    # raw measurement, no warm-up lag). inst itself uses the real dt, so the speed
                    # is correct; only the noise smoothing is per-frame.
                    a = 1.0 - self.vel_smoothing          # weight of the new sample
                    self._vel_raw = [self._vel_raw[0] * (1 - a) + inst[0] * a,
                                     self._vel_raw[1] * (1 - a) + inst[1] * a]
                    self._vel_w = self._vel_w * (1 - a) + a
                    self.vel = [self._vel_raw[0] / self._vel_w, self._vel_raw[1] / self._vel_w]
                self.meas = [mx, my]
                self.meas_t = now
                self.good_t = now
                ex_px, ey_px = mx, my

        dt = now - self.last_t
        self.last_t = now
        status = 'lost' if (now - self.good_t) > self.lost_s else 'track'

        ex = ex_px - self.cx
        ey = ey_px - self.cy
        if status == 'lost':                     # freeze the integrator; stop the mount
            return 0.0, 0.0, status, (ex_px, ey_px)

        # Derivative braking with a dead zone: oppose only the image speed *above* v_thresh.
        # During acquisition the target races across the frame (image speed >> v_thresh) so
        # this slows the slew, keeping the target inside the association gate and limiting
        # motion blur. Once locked, the target is held still in-frame (image speed ~0), so the
        # term is inactive -- the mount can still slew fast to follow a quick mover.
        evx = _excess(self.vel[0], self.v_thresh)     # brake on smoothed speed (image is noisy)
        evy = _excess(self.vel[1], self.v_thresh)

        # Integrate, clamped so the integral term alone commands at most the max motor rate.
        c = self.i_clamp
        self.integ[0] = max(-c, min(c, self.integ[0] + self.ki * ex * dt))
        self.integ[1] = max(-c, min(c, self.integ[1] + self.ki * ey * dt))
        # Second integral (kii): integrate the first integral again, same clamp. kii_step is 0
        # unless kii > 0, so integ2 stays 0 and this is a no-op for the default PI+D loop.
        self.integ2[0] = max(-c, min(c, self.integ2[0] + self.kii_step * self.integ[0] * dt))
        self.integ2[1] = max(-c, min(c, self.integ2[1] + self.kii_step * self.integ[1] * dt))

        m = self.max_rate
        rate_az = max(-m, min(m, self.sign_az * self.rad_per_px
                                 * (self.kp * ex + self.integ[0] + self.integ2[0] + self.kd * evx)))
        rate_alt = max(-m, min(m, self.sign_alt * self.rad_per_px
                                  * (self.kp * ey + self.integ[1] + self.integ2[1] + self.kd * evy)))
        return rate_az, rate_alt, status, (ex_px, ey_px)
