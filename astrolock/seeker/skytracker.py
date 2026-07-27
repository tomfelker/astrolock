"""
SkyTracker: the sky-space closed-loop tracker (Layers A + B).

Three parts:

  - Reconstruction (the bridge): a detection pixel + the mount pose interpolated to the frame's
    capture time + the plate scale => an absolute sky *direction*. Blind: the camera is assumed
    roughly upright (sign_az/sign_alt) and small-FoV; an unknown roll is a small cross-track bias
    the loop absorbs. This is the only approximation.

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

Contract with the backend: update() returns *final* axis rates (az, alt) -- SkyTracker owns the
alt-az inverse kinematics (the cos(alt) az-rate scaling and the pole tip-over), so the backend just
forwards the rates to the mount.

Not yet enforced (TODO): the motion-blur cap -- limiting the residual image-space speed at the
intercept so the target doesn't streak. During normal pursuit slewing toward the target only lowers
its image speed, so it is usually a non-issue; it bites only near the pole, where the fix is to add a
second feasibility term (residual speed at t_a under budget) alongside the reachability term below.
"""

import math

import torch

from astrolock.seeker import geometry as geo
from astrolock.seeker.target_model import EmaAngularVelModel


class SkyVectorPid:
    """PID trim on the measured centering error, entirely in sky-vector space.

    The error is the rotation vector carrying the hold-point direction onto the measured target
    direction (their cross product; magnitude ~ the separation angle in radians). kp/ki/kd act on
    that 3-vector, and the output is an angular-velocity vector (rad/s) -- the tracker converts it
    to axis rates at the current pose. The integral's magnitude is clamped to integral_limit_rad_s
    (the windup limit). Complements the feedforward servo: the servo chases the model's prediction,
    this drives down whatever persistent offset survives it (rate execution error, model lag),
    without either one knowing which it was.
    """

    def __init__(self, kp=0.0, ki=0.5, kd=0.0, integral_limit_rad_s=math.radians(0.02)):
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.integral_limit = float(integral_limit_rad_s)
        self.integral = torch.zeros(3, dtype=torch.float64)    # angular velocity (rad/s), sky space
        self.output = torch.zeros(3, dtype=torch.float64)      # kp*e + integral + kd*de/dt
        self._last_error = None                                # (t, error_vector) for dt and the D term

    def ingest(self, t, hold_dir, measured_dir):
        """One PID update, clocked by detections: dt is the time since the previous update."""
        error = torch.linalg.cross(hold_dir, measured_dir)     # rotation hold -> target (small angle)
        derivative = torch.zeros(3, dtype=torch.float64)
        if self._last_error is not None:
            t_prev, error_prev = self._last_error
            dt = t - t_prev
            if dt > 0.0:
                self.integral = self.integral + error * (self.ki * dt)
                norm = float(torch.linalg.norm(self.integral))
                if norm > self.integral_limit:                 # windup limit: clamp the magnitude
                    self.integral = self.integral * (self.integral_limit / norm)
                derivative = (error - error_prev) / dt
        self._last_error = (t, error)
        self.output = error * self.kp + self.integral + derivative * self.kd

    def reset(self):
        """Zero everything (the tracker calls this whenever the trim isn't running --
        disabled or not settled)."""
        self.integral = torch.zeros(3, dtype=torch.float64)
        self.output = torch.zeros(3, dtype=torch.float64)
        self._last_error = None

    def authority_used(self):
        """Fraction of the windup limit the integral currently holds, 0..1."""
        if self.integral_limit <= 0.0:
            return 0.0
        return float(torch.linalg.norm(self.integral)) / self.integral_limit


class SkyTracker:
    # PID settle dwell: the servo must intercept at the min horizon for this many consecutive
    # min-horizons (= error-decay time constants) before the trim integrates -- lets the
    # approach transient die out instead of winding the integral.
    SETTLE_HORIZONS = 5.0

    def __init__(self, cx, cy, rad_per_px, max_rate_rad_s,
                 model=None, min_intercept_s=0.3, command_latency_s=0.15, max_horizon_s=8.0,
                 horizon_step_s=0.1, gate_px=80.0, lost_s=1.5, lock_min_time=1.0,
                 sign_az=1.0, sign_alt=-1.0, feedforward_enabled=True, pid_enabled=True,
                 pid=None):
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
        self.feedforward_enabled = feedforward_enabled    # off: the intercept servo contributes no rates
        self.pid_enabled = pid_enabled                    # off: the PID trim is zeroed and contributes nothing
        self.pid = pid if pid is not None else SkyVectorPid()
        self.intercept_at_min_horizon = False   # servo's last solve: soonest candidate was feasible
        self._at_min_horizon_since = None       # when that first held, for the settle dwell
        self._pid_pending = None                # (t, hold_dir, measured_dir) awaiting the settled check
        self.pid_engaged = False                # trim currently integrating + contributing (for status)
        self.active = False
        self.single_target = False        # feed reports exactly the target (main extended cam) -> no association
        self.last_meas_px = None          # last ingested detection pixel (current role's frame), for the GUI
        # Piecewise-linear history of mount measurements (t, az, alt) to look up the pose at a *past*
        # frame time by interpolation. The latest measured rate is kept only to extrapolate *forward*
        # past the last measurement (the servo's command-latency lookahead) -- never backward.
        self._pose = geo.PoseHistory(maxlen=256)

    def target_speed_rad_s(self):
        """Apparent SKY angular speed of the model's prediction (rad/s), for status display.
        Computed numerically from predict(), so it reads the same thing for any model -- an
        orbit model's internal ang_vel is about the EARTH'S centre, not the sky."""
        t = getattr(self.model, 't', None)
        if t is None:
            return 0.0
        d0 = self.model.predict(t)
        if d0 is None:
            return 0.0
        dt = 0.2
        d1 = self.model.predict(t + dt)
        s = torch.linalg.norm(torch.linalg.cross(d0, d1))
        return float(torch.atan2(s, torch.dot(d0, d1))) / dt

    def diagnostics(self):
        """(info_lines, warnings) printed by the backend at lock time."""
        info = [f"sky: model {type(self.model).__name__}, min-intercept {self.min_intercept:.2f}s "
                f"(position stiffness ~{1.0 / self.min_intercept:.1f}/s), latency {self.latency:.2f}s, "
                f"horizon {self.max_horizon:.1f}s"]
        return info, []

    # ---- reconstruction: pixel <-> absolute sky direction, given the mount pose at that instant ----

    def push_mount(self, st):
        """Record a mount measurement into the pose history (geo.PoseHistory -- shared with the
        GUI overlay). Call it as often as the mount is polled (finer than frame rate is fine)."""
        self._pose.push(st['t_mono_ns'] * 1e-9, st['az_rad'], st['alt_rad'],
                        st['rate_az_rad_s'], st['rate_alt_rad_s'])

    def _pose_at(self, t):
        """Mount (az, alt) at time ``t`` (seconds): interpolated within the history,
        rate-extrapolated beyond it (see geo.PoseHistory)."""
        return self._pose.pose_at(t)

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

    def predict_pixel_in(self, t, cx, cy, rad_per_px, sign_az, sign_alt):
        """Where the model's predicted target lands in ANOTHER camera's frame (its centre + plate scale)
        at time t -- so the backend can keep a fallback camera's ROI on the target while a different
        camera drives, making a handoff back clean. Returns (px, py), or None before the first fix."""
        direction = self.model.predict(t)
        if direction is None:
            return None
        R = geo.mount_matrix(*self._pose_at(t))
        cam = R.transpose(0, 1) @ direction
        fd = float(cam[0])
        scale = 1.0 / fd if fd > 1e-6 else 1e6
        return (cx + float(cam[1]) * scale / (rad_per_px * sign_az),
                cy + float(cam[2]) * scale / (rad_per_px * sign_alt))

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
        self.last_meas_px = (px, py)

    def stop(self):
        self.active = False

    def switch_role(self, cx, cy, rad_per_px, sign_az, sign_alt, single_target=False):
        """Hand the sky-space model to a different camera (role handoff). Because Layer A models the
        target in absolute sky *directions*, this is just swapping the reconstruction geometry -- optical
        centre (cx, cy), plate scale, axis signs -- so the new camera's detection pixels reconstruct to
        the right directions and its centre becomes the hold point. The model (and its angular-velocity
        estimate) is untouched; the new camera's detections simply feed in through the normal update path.
        With a decent boresight the direction the model already holds and the new camera's detection agree
        to within the boresight accuracy, so there's no jump worth hacking around.

        ``single_target`` marks a feed that reports exactly the target (the main extended detector) so
        update() ingests it directly, no association/gate."""
        self.cx, self.cy = cx, cy
        self.rad_per_px = rad_per_px
        self.sign_az, self.sign_alt = sign_az, sign_alt
        self.single_target = single_target

    # ---- per-frame update ----

    def update(self, st, blobs, new_data, obs_time, now, driving=True):
        """Advance and return (rate_az, rate_alt, status, target_px).

        ``st`` is a fresh mount.get_state(); ``obs_time`` is the frame's capture time (for
        reconstruction/ingest); ``now`` is the current time (the servo predicts into now + latency);
        ``driving`` says whether the backend is actually forwarding our rates to the mount
        (False in watch-only mode and during the post-lock hold -- the PID must not integrate
        an error it isn't being allowed to correct).
        status is 'track', 'coast' (settled lock lost -- keep intercepting the extrapolation), or
        'lost' (unsettled lock lost -- stop).
        """
        self.push_mount(st)
        if new_data and blobs:
            oaz, oalt = self._pose_at(obs_time)                # boresight when the frame was captured
            if self.single_target:
                # The feed already reports exactly the target (the extended detector's own present/
                # compactness made the call), so there's nothing to associate -- just ingest it.
                best, hit = blobs[0], True
            else:
                # Multi-blob feed (e.g. the guide's DoH stars): pick the blob nearest our prediction and
                # gate it, so a brighter neighbour can't steal the lock.
                pred = self.model.predict(obs_time)             # where we think the target was then
                epx, epy = self._dir_to_pixel(pred, oaz, oalt)  # ... projected into that frame
                best = min(blobs, key=lambda b: math.hypot(b['px'][0] - epx, b['px'][1] - epy))
                hit = math.hypot(best['px'][0] - epx, best['px'][1] - epy) <= self.gate_px
            if hit:
                measured_dir = self._pixel_to_dir(best['px'][0], best['px'][1], oaz, oalt)
                self.model.ingest(obs_time, measured_dir)
                # PID trim error: hold-point direction vs the MEASURED target direction, both at
                # the frame's capture time via the SAME interpolated pose -- ground truth as the
                # detector sees it, no model. Held until _rates(), where THIS tick's servo solve
                # decides whether we're settled enough to ingest it.
                self._pid_pending = (obs_time,
                                     self._pixel_to_dir(self.cx, self.cy, oaz, oalt), measured_dir)
                self.last_meas_px = (best['px'][0], best['px'][1])
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
                raz, ralt = self._rates(now, driving)
                self.last_rate = (raz, ralt)
                return raz, ralt, 'coast', tpx
            return 0.0, 0.0, 'lost', tpx                        # RTLS: never settled -> stop

        raz, ralt = self._rates(now, driving)
        self.last_rate = (raz, ralt)
        return raz, ralt, 'track', tpx

    def _rates(self, now, driving=True):
        """Final commanded axis rates: the feedforward intercept servo plus the PID trim
        (a disabled contributor adds zero).

        Settled = the servo is off, or its solve has been intercepting at the MIN horizon
        continuously for SETTLE_HORIZONS min-horizons. At the min horizon the position error
        decays exponentially with the min horizon as its time constant, so the dwell waits
        out ~5 decay constants of approach transient before the integral is allowed to see
        any error. Whenever the PID is disabled or we are NOT settled -- the horizon had to
        expand (pole tip-over, catch-up slew, near gimbal lock), nothing was reachable, the
        dwell hasn't elapsed, or ``driving`` is False (our rates aren't reaching the mount:
        watch-only, post-lock hold) -- the PID is zeroed and contributes nothing: a wound-up
        integral parks a self-inflicted offset (clamp x 1/stiffness) that only unwinds at
        1/(ki * min_horizon) per e-fold, and the near-pole axis-rate conversion blows up
        exactly where the horizon expands. While running, the PID updates on each new
        detection and its angular-velocity output is re-converted to axis rates every tick
        at the pose where the command will land (cos(alt) drifts between updates)."""
        raz, ralt = self._servo(now) if self.feedforward_enabled else (0.0, 0.0)
        if not driving or (self.feedforward_enabled and not self.intercept_at_min_horizon):
            # Our rates aren't reaching the mount (watch-only, post-lock hold) or the horizon
            # expanded: not settled, and the dwell restarts from scratch.
            self._at_min_horizon_since = None
            settled = False
        elif not self.feedforward_enabled:
            settled = True                                # PID alone drives; no horizon to consult
        else:
            if self._at_min_horizon_since is None:
                self._at_min_horizon_since = now
            settled = (now - self._at_min_horizon_since) >= self.SETTLE_HORIZONS * self.min_intercept
        self.pid_engaged = self.pid_enabled and settled
        if not self.pid_engaged:
            self._pid_pending = None
            self.pid.reset()                              # disabled or unsettled: zero everything
            return raz, ralt
        if self._pid_pending is not None:
            t, hold_dir, measured_dir = self._pid_pending
            self._pid_pending = None
            self.pid.ingest(t, hold_dir, measured_dir)
        trim_az, trim_alt = self._trim_axis_rates(now + self.latency)
        return raz + trim_az, ralt + trim_alt

    def _trim_axis_rates(self, t):
        """The PID's sky-space angular-velocity output as axis rates at the pose expected when
        the command lands (t = now + latency). The output vector only changes at detection
        updates, but this conversion runs every tick because cos(alt) drifts in between.

        The demanded boresight velocity is output x forward; the az axis moves the boresight
        along `side` at cos(alt) per radian of axis angle (d forward/d az = cos(alt) * side),
        the alt axis along `up` at 1:1 -- signs stay correct through a pole tip-over because
        cos(alt) goes negative with the pose."""
        az, alt = self._pose_at(t)
        R = geo.mount_matrix(az, alt)
        velocity = torch.linalg.cross(self.pid.output, R[:, 0])
        cos_alt = math.cos(alt)
        if abs(cos_alt) < 1e-3:                        # at the pole az can't act; don't divide by ~0
            cos_alt = math.copysign(1e-3, cos_alt if cos_alt != 0.0 else 1.0)
        return float(torch.dot(velocity, R[:, 1])) / cos_alt, float(torch.dot(velocity, R[:, 2]))

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
            self.intercept_at_min_horizon = (j == 0)            # easy regime: no horizon expansion
            cost = torch.maximum(raz[:, j].abs(), ralt[:, j].abs())
            cost = torch.where(feas[:, j], cost, torch.full_like(cost, float('inf')))
            b = int(torch.argmin(cost))                         # gentler of the reachable poses
            return float(raz[b, j]), float(ralt[b, j])

        # Uncatchable within the horizon: least-infeasible pose, clamped to the motor limit.
        self.intercept_at_min_horizon = False
        over = torch.maximum(raz.abs(), ralt.abs()) - m
        flat = int(torch.argmin(over))
        b, j = divmod(flat, n)
        return (max(-m, min(m, float(raz[b, j]))), max(-m, min(m, float(ralt[b, j]))))
