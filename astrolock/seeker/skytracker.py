"""
SkyTracker: sky-space closed-loop tracker (Layers A + B), an alternative to the pixel-space
PixelTracker. Select it with ``--track-mode sky``; PixelTracker (``pixel``) is the default and is
untouched.

Three parts:

  - Reconstruction (the bridge): a detection pixel + the mount pose interpolated to the frame's
    capture time + the plate scale => an absolute sky *direction*. Blind: the camera is assumed
    roughly upright (sign_az/sign_alt) and small-FoV; an unknown roll is a small cross-track bias
    the loop absorbs. This is the only approximation, and it is the same one the pixel loop makes.

  - Layer A (TargetModel): a blind model of the target's motion in sky directions, fed the
    reconstructed directions. Default EmaAngularVelModel (constant-angular-velocity). It answers "where will
    the target point at time t" -- nothing about the mount.

  - Layer B (MountServo): commands the mount to *intercept* the model's future prediction. It solves
    for the earliest arrival time the mount can physically reach, taking the command latency and the
    motor rate limit into account, and commands exactly the rate that gets there. In the easy case
    this degenerates to feedforward (the target's own axis rate) plus a position pull whose stiffness
    is 1/min_intercept; during acquisition and near the pole it becomes a full-rate lead-pursuit to
    the soonest reachable point. cos(alt) never appears explicitly: near the zenith the target's az
    *coordinate* sweeps fast, so the required az rate exceeds the limit and the solver just bumps the
    arrival time -- and, because a direction has two alt-az poses ((az, alt) and (az+pi, pi-alt)),
    it will tip altitude *over the top* rather than whip az 180 degrees, whichever is gentler.

Contract with the backend: ``owns_ik = True`` -- update() returns final axis rates (az, alt), so the
backend does NOT apply its own cos(alt) compensation for this tracker.

Not yet enforced (TODO): the motion-blur cap -- limiting the residual image-space speed at the
intercept so the target doesn't streak. During normal pursuit slewing toward the target only lowers
its image speed, so it is usually a non-issue; it bites only near the pole, where the fix is to add a
second feasibility term (residual speed at t_a under budget) alongside the reachability term below.
"""

import collections
import math

import torch

from astrolock.seeker import geometry as geo
from astrolock.seeker.target_model import EmaAngularVelModel


class SkyTracker:
    owns_ik = True     # our update() returns final axis rates; backend must not re-apply cos(alt)

    def __init__(self, cx, cy, rad_per_px, max_rate_rad_s,
                 model=None, min_intercept_s=0.3, command_latency_s=0.15, max_horizon_s=8.0,
                 horizon_step_s=0.1, gate_px=80.0, lost_s=1.5, lock_min_time=1.0,
                 sign_az=1.0, sign_alt=-1.0):
        self.cx, self.cy = cx, cy
        self.rad_per_px = rad_per_px
        self.max_rate = max_rate_rad_s
        self.model = model if model is not None else EmaAngularVelModel()
        self.min_intercept = min_intercept_s          # arrival-time floor; also sets P ~ 1/this
        self.latency = command_latency_s              # assumed delay before a command takes effect
        self.max_horizon = max_horizon_s              # beyond this with no intercept -> uncatchable
        self.horizon_step = horizon_step_s
        self.gate_px = gate_px
        self.lost_s = lost_s
        self.lock_min_time = lock_min_time
        self.sign_az, self.sign_alt = sign_az, sign_alt
        self.active = False
        # Piecewise-linear history of mount measurements (t, az, alt) to look up the pose at a *past*
        # frame time by interpolation. The latest measured rate is kept only to extrapolate *forward*
        # past the last measurement (the servo's command-latency lookahead) -- never backward.
        self._hist = collections.deque(maxlen=256)
        self._mount_rate = (0.0, 0.0)

    def diagnostics(self):
        """(info_lines, warnings) printed by the backend at lock time, mirroring PixelTracker."""
        info = [f"sky: model {type(self.model).__name__}, min-intercept {self.min_intercept:.2f}s "
                f"(position stiffness ~{1.0 / self.min_intercept:.1f}/s), latency {self.latency:.2f}s, "
                f"horizon {self.max_horizon:.1f}s"]
        return info, []

    # ---- reconstruction: pixel <-> absolute sky direction, given the mount pose at that instant ----

    def push_mount(self, st):
        """Record a mount measurement into the history. Call it as often as the mount is polled
        (finer than frame rate is fine); duplicates/out-of-order samples are ignored."""
        t = st['t_mono_ns'] * 1e-9
        self._mount_rate = (st['rate_az_rad_s'], st['rate_alt_rad_s'])
        if self._hist and t <= self._hist[-1][0]:
            return
        self._hist.append((t, st['az_rad'], st['alt_rad']))

    def _pose_at(self, t):
        """Mount (az, alt) at time ``t`` (seconds) from the measurement history.

        For a ``t`` within the history (a past frame time) we interpolate the bracketing measurements
        -- the piecewise-linear pose the mount actually followed, correct through rate changes. Only
        for ``t`` beyond the last measurement do we extrapolate, at the last measured rate.
        """
        hist = self._hist
        if not hist:
            return (0.0, 0.0)
        t_last, az_last, alt_last = hist[-1]
        if t >= t_last:                                   # future: extrapolate at the last measured rate
            dt = t - t_last
            return (az_last + self._mount_rate[0] * dt, alt_last + self._mount_rate[1] * dt)
        newer = None
        for s in reversed(hist):                          # walk back to the bracketing segment
            if newer is not None and s[0] <= t:
                frac = (t - s[0]) / (newer[0] - s[0]) if newer[0] > s[0] else 0.0
                return (s[1] + geo.wrap_pi(newer[1] - s[1]) * frac,
                        s[2] + geo.wrap_pi(newer[2] - s[2]) * frac)
            newer = s
        return (hist[0][1], hist[0][2])                   # older than the whole history: clamp

    def _pixel_to_dir(self, px, py, az, alt):
        """Absolute sky direction of a detection at (px, py) when the boresight is at (az, alt).

        Uses the mount's rotation matrix (columns = camera axes), so it stays correct as the mount
        tips past the pole -- no cos(alt), no 180-deg azimuth flip.
        """
        R = geo.mount_matrix(az, alt)                       # columns: forward, side(+az), up(+alt)
        ox = (px - self.cx) * self.rad_per_px * self.sign_az
        oy = (py - self.cy) * self.rad_per_px * self.sign_alt
        cam = torch.tensor([1.0, ox, oy], dtype=torch.float64)   # forward + ox*side + oy*up, in cam coords
        return geo.normalize(R @ cam)

    def _dir_to_pixel(self, direction, az, alt):
        """Inverse of _pixel_to_dir: where a sky direction lands in a frame with boresight (az, alt)."""
        R = geo.mount_matrix(az, alt)
        cam = R.transpose(0, 1) @ direction                 # world->cam: [forward.d, side.d, up.d]
        fd = float(cam[0])
        scale = 1.0 / fd if fd > 1e-6 else 1e6              # target behind the camera -> off-screen
        px = self.cx + float(cam[1]) * scale / (self.rad_per_px * self.sign_az)
        py = self.cy + float(cam[2]) * scale / (self.rad_per_px * self.sign_alt)
        return px, py

    # ---- lifecycle ----

    def start(self, px, py, obs_time, st):
        """Acquire: seed the model with one reconstructed direction from the initial pick."""
        self.active = True
        self.push_mount(st)
        az, alt = self._pose_at(obs_time)
        self.model.ingest(obs_time, self._pixel_to_dir(px, py, az, alt))
        self.good_t = obs_time                        # last successful association (start counts)
        self.settled_since = None                     # first association after a gap
        self.settled = False                          # associated for >= lock_min_time
        self.last_rate = (0.0, 0.0)

    def stop(self):
        self.active = False

    # ---- per-frame update ----

    def update(self, st, blobs, new_data, obs_time, now):
        """Advance and return (rate_az, rate_alt, status, target_px).

        ``st`` is a fresh mount.get_state(); ``obs_time`` is the frame's capture time (for
        reconstruction/ingest); ``now`` is the current time (the servo predicts into now + latency).
        status is 'track', 'coast' (settled lock lost -- keep intercepting the extrapolation), or
        'lost' (unsettled lock lost -- stop).
        """
        self.push_mount(st)
        if new_data and blobs:
            oaz, oalt = self._pose_at(obs_time)                # boresight when the frame was captured
            pred = self.model.predict(obs_time)                 # where we think the target was then
            epx, epy = self._dir_to_pixel(pred, oaz, oalt)      # ... projected into that frame
            best = min(blobs, key=lambda b: math.hypot(b['px'][0] - epx, b['px'][1] - epy))
            if math.hypot(best['px'][0] - epx, best['px'][1] - epy) <= self.gate_px:
                self.model.ingest(obs_time,
                                  self._pixel_to_dir(best['px'][0], best['px'][1], oaz, oalt))
                self.good_t = obs_time
                if self.settled_since is None:
                    self.settled_since = obs_time
                self.settled = (obs_time - self.settled_since) >= self.lock_min_time
            else:
                self.settled_since, self.settled = None, False

        # Publish the predicted target in the *current* frame (drives the ROI + GUI marker).
        caz, calt = self._pose_at(now)
        tpx = self._dir_to_pixel(self.model.predict(now), caz, calt)

        if (now - self.good_t) > self.lost_s:                   # lost the target
            if self.settled:                                    # PTO: keep intercepting the model
                raz, ralt = self._servo(now)
                self.last_rate = (raz, ralt)
                return raz, ralt, 'coast', tpx
            return 0.0, 0.0, 'lost', tpx                        # RTLS: never settled -> stop

        raz, ralt = self._servo(now)
        self.last_rate = (raz, ralt)
        return raz, ralt, 'track', tpx

    # ---- Layer B: minimum-time intercept ----

    def _servo(self, now):
        """Earliest-feasible min-time intercept of the model's prediction, in raw axis space.

        Evaluate a whole horizon of candidate arrival times at once (batched predict). For each, the
        two equivalent mount poses of the predicted direction give the required constant az/alt rates
        to close the gap from where the mount will be when the command lands. Take the soonest arrival
        at which some pose is reachable within the motor limit, preferring the gentler pose; if none
        is reachable within the horizon, command the least-infeasible pose clamped to the limit.
        """
        m = self.max_rate
        t0 = now + self.latency                                 # when this command takes effect
        az0, alt0 = self._pose_at(t0)                           # where the mount will be by then

        n = max(1, int(self.max_horizon / self.horizon_step))
        dt = self.min_intercept + self.horizon_step * torch.arange(n, dtype=torch.float64)  # (n,)
        dirs = self.model.predict(t0 + dt)                      # (n, 3) -- batched over the horizon
        paz, palt = geo.dir_to_azalt(dirs)                      # (n,), (n,)

        # Two alt-az poses per direction: (az, alt) and (az+pi, pi-alt). The second is the "tip over
        # the top" branch that avoids whipping az ~180 deg through the zenith.
        az_b = torch.stack([paz, paz + math.pi])                # (2, n)
        alt_b = torch.stack([palt, math.pi - palt])             # (2, n)
        raz = geo.wrap_pi(az_b - az0) / dt                       # (2, n) constant rate to intercept
        ralt = geo.wrap_pi(alt_b - alt0) / dt
        feas = (raz.abs() <= m) & (ralt.abs() <= m)             # (2, n)

        col = torch.nonzero(feas.any(dim=0))
        if len(col) > 0:
            j = int(col[0])                                     # soonest reachable arrival time
            cost = torch.maximum(raz[:, j].abs(), ralt[:, j].abs())
            cost = torch.where(feas[:, j], cost, torch.full_like(cost, float('inf')))
            b = int(torch.argmin(cost))                         # gentler of the reachable poses
            return float(raz[b, j]), float(ralt[b, j])

        # Uncatchable within the horizon: least-infeasible pose, clamped to the motor limit.
        over = torch.maximum(raz.abs(), ralt.abs()) - m
        flat = int(torch.argmin(over))
        b, j = divmod(flat, n)
        return (max(-m, min(m, float(raz[b, j]))), max(-m, min(m, float(ralt[b, j]))))
