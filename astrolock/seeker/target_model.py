"""
TargetModel: a blind, measurement-fed model of a target's motion across the sky ("Layer A").

Feed it observations -- a sky direction (unit 3-vector) at a time -- with ingest(); ask where the
target will point at any future time(s) with predict(). It is coordinate-free (directions, not
az/alt), so it has no zenith singularity, and it knows nothing about the mount, the camera, command
latency, or motor limits -- that is Layer B's job (see skytracker.MountServo). It is deliberately
"blind": no TLE, no precise alignment, only what the detections + mount pose reconstruct.

The trivial model here is EmaAngularVelModel: an EMA of the target's angular-velocity *vector*, i.e.
"assume it keeps sweeping across the sky the way it has been" -- a constant-angular-velocity great
circle. That is the short-arc / zero-range-rate limit of richer models we can drop in later behind the same
ingest/predict face (great circle at an assumed altitude, a fitted straight 3D line that also
captures perspective foreshortening, a fitted orbital state vector). Layer B never changes.

Interface note: predict() takes a float or a tensor of times and returns a (3,) or (N, 3) direction,
so the intercept solver can evaluate a whole horizon of candidate arrival times in one batched call.
"""

import math

import torch

from astrolock.seeker import geometry as geo


class TargetModel:
    def ingest(self, t, direction):
        """Add an observation: a direction (3-vector, need not be unit) seen at time ``t`` seconds."""
        raise NotImplementedError

    def predict(self, t):
        """Predicted unit direction at time(s) ``t`` (float -> (3,), tensor[N] -> (N, 3)).
        Returns None before the first observation."""
        raise NotImplementedError


def _position_blend_weight(dt, position_smoothing_s, norm):
    """Warmup-corrected EMA weight for blending a new position measurement into the anchor.

    Returns (effective_weight, updated_norm). The raw time-correct weight is
    1 - exp(-dt/tau); ``norm`` accumulates total weight so early samples aren't biased
    toward the (meaningless) initial anchor: the first blended sample gets weight 1
    (snap), and the effective weight relaxes to the raw value as norm -> 1.
    """
    if position_smoothing_s <= 0.0:
        return 1.0, 1.0                                    # smoothing off: snap (current behavior)
    weight = 1.0 - math.exp(-dt / position_smoothing_s)
    norm = norm * (1.0 - weight) + weight
    return weight / norm, norm


class GreatCircleModel(TargetModel):
    """Constant-ALTITUDE model: the target rides a great circle on a sphere of radius
    R_earth + altitude about the Earth's centre -- a blind circular-orbit LEO pass (no TLE).

    Each observed direction is projected onto that sphere (ray-sphere intersection from the
    observer); the state is the position ON the sphere plus an EMA'd angular velocity about the
    EARTH'S centre, and predict() rotates the position and converts back to an observer-relative
    direction. Where the constant-SKY-rate model fails hardest -- the zenith speed-up and the
    horizon slow-down of a pass are pure perspective, not target dynamics -- this geometry
    produces them for free: the orbital rate really is ~constant, so the EMA has nothing to chase.

    Frame: the tracker's local topocentric frame (+z = zenith), observer at the origin, so the
    Earth's centre sits at (0, 0, -R_earth). Earth rotation and site curvature over a pass are
    below the boresight error budget -- deliberately ignored (blind model, not an ephemeris).
    """

    def __init__(self, smoothing_s=0.5, position_smoothing_s=0.0, altitude_m=250e3,
                 earth_radius_m=6371e3):
        self.smoothing_s = float(smoothing_s)                 # EMA time constant for the rate estimate (s)
        self.position_smoothing_s = float(position_smoothing_s)   # anchor blend time constant (0 = snap)
        self.re = float(earth_radius_m)
        self.r = float(earth_radius_m) + float(altitude_m)    # sphere radius about the Earth's centre
        self._c = torch.tensor([0.0, 0.0, self.re], dtype=torch.float64)   # observer - Earth centre
        self.t = None
        self.u = None                                          # unit position about the Earth's centre
        self.ang_vel = torch.zeros(3, dtype=torch.float64)     # rad/s about the Earth's centre
        self._measured_u = None                                # last RAW measurement (rate estimation)
        self._position_weight_norm = 0.0                       # warmup-EMA weight accumulator

    def _project(self, d):
        """Observer-relative unit direction -> unit position on the sphere (about Earth centre).
        |s*d + (0,0,re)| = r has one positive root for any pointing (r > re)."""
        dz = float(d[2])
        s = -self.re * dz + math.sqrt((self.re * dz) ** 2 + self.r ** 2 - self.re ** 2)
        return geo.normalize(s * d + self._c)

    def _dir_of(self, u):
        """Unit position(s) [..., 3] on the sphere -> observer-relative unit direction(s)."""
        return geo.normalize(self.r * u - self._c)

    def ingest(self, t, direction):
        d = geo.normalize(torch.as_tensor(direction, dtype=torch.float64))
        u = self._project(d)
        t = float(t)
        if self.u is None:
            self.u, self.t, self._measured_u = u, t, u
            return
        dt = t - self.t
        if dt <= 0.0:                                          # out-of-order / repeated frame: re-anchor
            self.u, self.t, self._measured_u = u, t, u
            return
        # Rate estimate from RAW consecutive measurements, not the blended anchor, so position
        # smoothing can't leak a position correction into the angular-velocity estimate.
        cross = torch.linalg.cross(self._measured_u, u)
        s = torch.linalg.norm(cross)
        c = torch.dot(self._measured_u, u)
        angle = torch.atan2(s, c)
        if float(s) > 1e-9:
            ang_vel_inst = cross / s * (angle / dt)            # axis * (angle/dt), about Earth centre
        else:
            ang_vel_inst = torch.zeros(3, dtype=torch.float64)
        weight = 1.0 - math.exp(-dt / self.smoothing_s) if self.smoothing_s > 0.0 else 1.0
        self.ang_vel = self.ang_vel * (1.0 - weight) + ang_vel_inst * weight
        self._measured_u = u
        # Anchor: blend the measurement with the model's own extrapolation (warmup-EMA style --
        # the first sample snaps, then the weight relaxes to 1 - exp(-dt/tau)). 0 = always snap.
        blend, self._position_weight_norm = _position_blend_weight(
            dt, self.position_smoothing_s, self._position_weight_norm)
        if blend >= 1.0:
            self.u = u
        else:
            rate = torch.linalg.norm(self.ang_vel)
            extrapolated = (geo.normalize(geo.rodrigues(self.u, self.ang_vel / rate, rate * dt))
                            if float(rate) > 1e-12 else self.u)
            self.u = geo.normalize(extrapolated * (1.0 - blend) + u * blend)
        self.t = t

    def predict(self, t):
        if self.u is None:
            return None
        t = torch.as_tensor(t, dtype=torch.float64)
        dt = t - self.t
        rate = torch.linalg.norm(self.ang_vel)
        if float(rate) < 1e-12:                                # not moving: constant position
            if dt.dim() == 0:
                return self._dir_of(self.u)
            return self._dir_of(self.u.expand(*dt.shape, 3))
        axis = self.ang_vel / rate
        return self._dir_of(geo.normalize(geo.rodrigues(self.u, axis, rate * dt)))


class EmaAngularVelModel(TargetModel):
    """Constant-angular-velocity model with an EMA on the estimated angular velocity.

    Between two directions the instantaneous angular velocity is the rotation carrying the old
    direction to the new one (axis = old x new, magnitude = angle / dt); we EMA that vector with a
    time-correct weight so the smoothing is frame-rate independent, and extrapolate by rotating the
    last direction about it (Rodrigues). Two observations are enough to predict motion; one holds still.
    """

    def __init__(self, smoothing_s=0.5, position_smoothing_s=0.0):
        self.smoothing_s = float(smoothing_s)                 # EMA time constant for the rate estimate (s)
        self.position_smoothing_s = float(position_smoothing_s)   # anchor blend time constant (0 = snap)
        self.t = None                                          # time of the current direction anchor
        self.dir = None                                        # unit direction at self.t  (torch (3,))
        self.ang_vel = torch.zeros(3, dtype=torch.float64)     # angular velocity (rad/s): axis * rate
        self._measured_dir = None                              # last RAW measurement (rate estimation)
        self._position_weight_norm = 0.0                       # warmup-EMA weight accumulator

    def ingest(self, t, direction):
        d = geo.normalize(torch.as_tensor(direction, dtype=torch.float64))
        t = float(t)
        if self.dir is None:
            self.dir, self.t, self._measured_dir = d, t, d
            return
        dt = t - self.t
        if dt <= 0.0:                                          # out-of-order / repeated frame: re-anchor
            self.dir, self.t, self._measured_dir = d, t, d
            return
        # Instantaneous angular velocity from the last RAW measurement to this one (not the blended
        # anchor, so position smoothing can't leak a position correction into the rate estimate).
        cross = torch.linalg.cross(self._measured_dir, d)
        s = torch.linalg.norm(cross)
        c = torch.dot(self._measured_dir, d)
        angle = torch.atan2(s, c)                              # unsigned angle between the two dirs
        if float(s) > 1e-9:
            ang_vel_inst = cross / s * (angle / dt)            # axis * (angle/dt)
        else:
            ang_vel_inst = torch.zeros(3, dtype=torch.float64)
        weight = 1.0 - math.exp(-dt / self.smoothing_s) if self.smoothing_s > 0.0 else 1.0
        self.ang_vel = self.ang_vel * (1.0 - weight) + ang_vel_inst * weight
        self._measured_dir = d
        # Anchor: blend the measurement with the model's own extrapolation (warmup-EMA style --
        # the first sample snaps, then the weight relaxes to 1 - exp(-dt/tau)). 0 = always snap.
        blend, self._position_weight_norm = _position_blend_weight(
            dt, self.position_smoothing_s, self._position_weight_norm)
        if blend >= 1.0:
            self.dir = d
        else:
            rate = torch.linalg.norm(self.ang_vel)
            extrapolated = (geo.normalize(geo.rodrigues(self.dir, self.ang_vel / rate, rate * dt))
                            if float(rate) > 1e-12 else self.dir)
            self.dir = geo.normalize(extrapolated * (1.0 - blend) + d * blend)
        self.t = t

    def predict(self, t):
        if self.dir is None:
            return None
        t = torch.as_tensor(t, dtype=torch.float64)
        dt = t - self.t
        rate = torch.linalg.norm(self.ang_vel)
        if float(rate) < 1e-12:                                # not moving: constant direction
            if dt.dim() == 0:
                return self.dir.clone()
            return self.dir.expand(*dt.shape, 3).clone()
        axis = self.ang_vel / rate
        return geo.normalize(geo.rodrigues(self.dir, axis, rate * dt))
