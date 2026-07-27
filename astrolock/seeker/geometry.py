"""
Small pure-pytorch geometry helpers for the sky-space tracker (Layer A + reconstruction).

Everything works on *direction* vectors -- unit 3-vectors in the local topocentric frame with the
same convention as the old model.util: az measured from +x toward +y, alt from the xy-plane toward
+z, so d = (cos az cos alt, sin az cos alt, sin alt). Working in directions keeps Layer A
coordinate-free (no zenith singularity); the az/alt <-> dir conversions here are the only place
that knows the mount is alt-az.

THE convention (standardize on this everywhere new):
- Sky vector space: x = north, y = east, z = up (azimuth 0 = north, increasing eastward; the
  formulas above with mount az plugged in).
- Rotation matrices are ``parent_from_child``: columns are the child frame's axes expressed in
  the parent frame, vectors are columns, so d_parent = R @ d_child and chains read right to
  left (sky_from_camera = sky_from_mount @ mount_from_camera).
- mount_matrix(az, alt) is sky_from_camera for an ideally-aligned mount: columns = forward,
  side (toward +az), up (toward +alt).
- Caveat: skysim's renderer/projector historically uses (east, north, up) -- same angles, x/y
  swapped. Adapt at that boundary (a basis permute) rather than mixing frames.

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


def rotation_matrix(axis, angle):
    """Rotation matrix (3, 3) about unit ``axis`` (3,) by ``angle`` radians (right-handed).
    The matrix form of rodrigues(): R = I + sin K + (1 - cos) K^2 with K the cross-product
    matrix of ``axis``."""
    axis = torch.as_tensor(axis, dtype=torch.float64)
    angle = torch.as_tensor(angle, dtype=torch.float64)
    zero = torch.zeros((), dtype=torch.float64)
    K = torch.stack([torch.stack([zero, -axis[2], axis[1]]),
                     torch.stack([axis[2], zero, -axis[0]]),
                     torch.stack([-axis[1], axis[0], zero])])
    return (torch.eye(3, dtype=torch.float64) + torch.sin(angle) * K
            + (1.0 - torch.cos(angle)) * (K @ K))


def boresight_rotation(x_rad, y_rad, roll_rad=0.0):
    """guide_from_main: how the main camera sits in the guide camera's frame, from the three
    GUI angles -- yaw x_rad toward guide image right (about camera up), pitch y_rad toward
    image down (about the yawed side axis), roll_rad about the resulting boresight. Elementary
    rotations composed, mount_matrix style (yaw @ pitch @ roll); camera frame = (forward,
    side, up); d_guide = R @ d_main. All three angles use the IMAGE convention (x right,
    y down, roll clockwise as seen in the guide view -- hence the sign flip on roll: screen
    clockwise is a negative rotation about the into-scene forward axis)."""
    return (rotation_matrix(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), x_rad)
            @ rotation_matrix(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), y_rad)
            @ rotation_matrix(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), -roll_rad))


def matrix_from_euler(yaw_rad, pitch_rad, roll_rad):
    """Rotation from yaw @ pitch @ roll about the frame's z / y / x axes (the same composition
    shape as mount_matrix and boresight_rotation). Radians in, like everything internal."""
    return (rotation_matrix(torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64), yaw_rad)
            @ rotation_matrix(torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64), pitch_rad)
            @ rotation_matrix(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), roll_rad))


def euler_from_matrix(R):
    """(yaw, pitch, roll) radians inverting matrix_from_euler. Pitch lands in [-pi/2, pi/2];
    exactly at the poles the yaw/roll split is degenerate (any consistent pair is returned)."""
    pitch = torch.asin(torch.clamp(-R[2, 0], -1.0, 1.0))
    yaw = torch.atan2(R[1, 0], R[0, 0])
    roll = torch.atan2(R[2, 1], R[2, 2])
    return float(yaw), float(pitch), float(roll)


def perspective_matrix(f_px_x, f_px_y, cx, cy):
    """Pinhole projection as a MATRIX: camera-frame direction (forward, side, up) -> homogeneous
    image pixel (x right, y down), pixel_h = K @ d, pixel = pixel_h[:2] / pixel_h[2]. Its inverse
    unprojects a pixel back to an (unnormalized) camera-frame direction. This pair -- unproject,
    rotate between camera spaces, reproject -- is THE way to move pixels between cameras; never
    hand-rolled per-axis trig."""
    return torch.tensor([[cx, f_px_x, 0.0],
                         [cy, 0.0, -f_px_y],
                         [1.0, 0.0, 0.0]], dtype=torch.float64)


def project_pixel(K, d):
    """Project camera-frame direction ``d`` (3,) through perspective matrix ``K`` -> (px, py)
    floats. The caller checks d is in front (positive forward component) if it can be behind."""
    h = K @ torch.as_tensor(d, dtype=torch.float64)
    return float(h[0] / h[2]), float(h[1] / h[2])


def orthonormalized(R):
    """Nearest rotation matrix to ``R`` (SVD projection). Re-normalizes a chain of composed
    rotations so float drift can't accumulate shear/scale."""
    U, _, Vt = torch.linalg.svd(R)
    return U @ Vt


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
