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
