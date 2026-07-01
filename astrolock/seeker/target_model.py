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


class EmaAngularVelModel(TargetModel):
    """Constant-angular-velocity model with an EMA on the estimated angular velocity.

    Between two directions the instantaneous angular velocity is the rotation carrying the old
    direction to the new one (axis = old x new, magnitude = angle / dt); we EMA that vector with a
    time-correct weight so the smoothing is frame-rate independent, and extrapolate by rotating the
    last direction about it (Rodrigues). Two observations are enough to predict motion; one holds still.
    """

    def __init__(self, smoothing_s=0.5):
        self.smoothing_s = float(smoothing_s)                 # EMA time constant for the rate estimate (s)
        self.t = None                                          # time of the current direction anchor
        self.dir = None                                        # unit direction at self.t  (torch (3,))
        self.ang_vel = torch.zeros(3, dtype=torch.float64)     # angular velocity (rad/s): axis * rate

    def ingest(self, t, direction):
        d = geo.normalize(torch.as_tensor(direction, dtype=torch.float64))
        t = float(t)
        if self.dir is None:
            self.dir, self.t = d, t
            return
        dt = t - self.t
        if dt <= 0.0:                                          # out-of-order / repeated frame: re-anchor
            self.dir, self.t = d, t
            return
        # Instantaneous angular velocity from the last direction to this one.
        cross = torch.linalg.cross(self.dir, d)
        s = torch.linalg.norm(cross)
        c = torch.dot(self.dir, d)
        angle = torch.atan2(s, c)                              # unsigned angle between the two dirs
        if float(s) > 1e-9:
            ang_vel_inst = cross / s * (angle / dt)            # axis * (angle/dt)
        else:
            ang_vel_inst = torch.zeros(3, dtype=torch.float64)
        weight = 1.0 - math.exp(-dt / self.smoothing_s) if self.smoothing_s > 0.0 else 1.0
        self.ang_vel = self.ang_vel * (1.0 - weight) + ang_vel_inst * weight
        self.dir, self.t = d, t

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
