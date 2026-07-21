"""
Small pure-pytorch geometry helpers for the sky-space tracker (Layer A + reconstruction).

Everything works on *direction* vectors -- unit 3-vectors in the local topocentric frame with the
same convention as the old model.util: az measured from +x toward +y, alt from the xy-plane toward
+z, so d = (cos az cos alt, sin az cos alt, sin alt). Working in directions keeps Layer A
coordinate-free (no zenith singularity); the az/alt <-> dir conversions here are the only place
that knows the mount is alt-az.

Torch, not numpy, by house rule -- and it pays off: predict() can batch over a whole vector of
candidate times at once (the intercept solver leans on this), and a future fitted/orbital model can
autodiff, with no change to the interface.
"""

import collections
import math

import torch

_TWO_PI = 2.0 * math.pi


def azalt_to_dir(az, alt):
    """(az, alt) in radians -> unit direction. Scalars or broadcastable tensors; returns [..., 3]."""
    az = torch.as_tensor(az, dtype=torch.float64)
    alt = torch.as_tensor(alt, dtype=torch.float64)
    ca = torch.cos(alt)
    return torch.stack([torch.cos(az) * ca, torch.sin(az) * ca, torch.sin(alt)], dim=-1)


def dir_to_azalt(d):
    """Unit direction [..., 3] -> (az, alt) tensors [...], each in radians (alt in [-pi/2, pi/2])."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    az = torch.atan2(y, x)
    alt = torch.atan2(z, torch.hypot(x, y))
    return az, alt


def normalize(v, eps=1e-12):
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    return v / torch.clamp(n, min=eps)


def mount_matrix(az, alt):
    """Orientation of an alt-az mount at (az, alt) as a rotation matrix whose columns are the camera
    axes in world coordinates: forward (boresight), side (toward +az), up (toward +alt).

    This is the composition of the yaw and pitch rotations, R = Rz(az) @ pitch(alt); we build the
    columns directly. Singularity-free in the representation -- pitch can pass 90 deg (the mount tips
    over the pole) and R stays smooth, so pixel<->sky reconstruction through it is just a matrix
    multiply, no cos(alt) / no 180-deg azimuth flip / no zenith fallback. (Only decomposing a
    direction *back* to (az, alt) gimbal-locks at the pole, which is the servo's concern, not this.)
    """
    az = torch.as_tensor(az, dtype=torch.float64)
    alt = torch.as_tensor(alt, dtype=torch.float64)
    ca, sa = torch.cos(az), torch.sin(az)
    cl, sl = torch.cos(alt), torch.sin(alt)
    z = torch.zeros_like(ca)
    forward = torch.stack([ca * cl, sa * cl, sl])
    side = torch.stack([-sa, ca, z])
    up = torch.stack([-ca * sl, -sa * sl, cl])
    return torch.stack([forward, side, up], dim=1)      # columns = camera axes in world coords


def rodrigues(v, axis, angle):
    """Rotate vector ``v`` (3,) about unit ``axis`` (3,) by ``angle`` (radians).

    ``angle`` may be a tensor of shape [...] to produce [..., 3] -- i.e. rotate ``v`` to many future
    times in one call. ``axis`` is assumed unit-length.
    """
    angle = torch.as_tensor(angle, dtype=v.dtype)
    kxv = torch.linalg.cross(axis, v)       # (3,)
    kdv = torch.dot(axis, v)                 # scalar
    c = torch.cos(angle).unsqueeze(-1)       # [..., 1]
    s = torch.sin(angle).unsqueeze(-1)
    return v * c + kxv * s + (axis * kdv) * (1.0 - c)


def wrap_pi(a):
    """Shortest signed representative in (-pi, pi]. Works on floats or tensors."""
    return (a + math.pi) % _TWO_PI - math.pi


class PoseHistory:
    """Mount pose measurements -> the pose at an arbitrary instant. One problem, one solver:
    the tracker asks where the mount pointed when a (possibly several-frames-old) detection
    was captured, and the GUI overlay asks where it pointed at the displayed frame's capture
    stamp. Extracted from SkyTracker so both use the SAME battle-tested lookup.

    Within the history: piecewise-linear interpolation of the bracketing measurements
    (wrap-aware) -- the pose the mount actually followed, correct through rate changes.
    Beyond the newest measurement: extrapolate at the last measured rate -- clamping to a
    stale pose instead would displace a consumer by slew_rate x staleness the moment its
    query outruns the feed (a one-frame glyph jump in the GUI). Before the oldest: clamp.
    Duplicate / out-of-order pushes are ignored (their rates still update). Times are
    SECONDS on the shared mono timeline (mono_s); angles/rates are radians (house rule).
    """

    def __init__(self, maxlen=256):
        self._hist = collections.deque(maxlen=maxlen)
        self._rate = (0.0, 0.0)

    def __len__(self):
        return len(self._hist)

    def push(self, t_s, az_rad, alt_rad, rate_az_rad_s=0.0, rate_alt_rad_s=0.0):
        """Record a measurement. Call as often as the mount is polled (finer than frame
        rate is fine)."""
        self._rate = (rate_az_rad_s, rate_alt_rad_s)
        if self._hist and t_s <= self._hist[-1][0]:
            return
        self._hist.append((t_s, az_rad, alt_rad))

    def pose_at(self, t_s):
        """Mount (az_rad, alt_rad) at time ``t_s`` (seconds)."""
        hist = self._hist
        if not hist:
            return (0.0, 0.0)
        t_last, az_last, alt_last = hist[-1]
        if t_s >= t_last:                                 # future: extrapolate at the last rate
            dt = t_s - t_last
            return (az_last + self._rate[0] * dt, alt_last + self._rate[1] * dt)
        newer = None
        for s in reversed(hist):                          # walk back to the bracketing segment
            if newer is not None and s[0] <= t_s:
                frac = (t_s - s[0]) / (newer[0] - s[0]) if newer[0] > s[0] else 0.0
                return (s[1] + wrap_pi(newer[1] - s[1]) * frac,
                        s[2] + wrap_pi(newer[2] - s[2]) * frac)
            newer = s
        return (hist[0][1], hist[0][2])                   # older than the whole history: clamp
