"""
Coarse point-cloud body models for the satellites we can resolve through a long-focal-length cam.

Each model is an (N, 3) array of points in the object's body frame, metres:
    x = ram (velocity / forward),  y = port (cross-track, the usual "wingspan"),  z = nadir (Earth-ward).
Just enough points to read as a shape (and to streak with motion) -- tweak freely. skysim places these
in the target's attitude and projects them, so it resolves into a shape instead of a dot.

The attitude is the LVLH "guess" skysim applies (ram forward, nadir down). That's accurate for the
LVLH-held stations (ISS, Tiangong); for Hubble (inertial pointing) and Starship (whatever) it's just
a plausible default so the shape grows/shrinks correctly -- exact attitude isn't modelled.

points_for_name() picks a model from the TLE object name; unknown objects fall back to a single point.
"""

import numpy as np


def _line(axis, lo, hi, n, cx=0.0, cy=0.0, cz=0.0):
    """n points along one axis from lo..hi, offset to (cx,cy,cz)."""
    t = np.linspace(lo, hi, n)
    z = np.zeros_like(t)
    if axis == 'x':
        return np.stack([cx + t, cy + z, cz + z], -1)
    if axis == 'y':
        return np.stack([cx + z, cy + t, cz + z], -1)
    return np.stack([cx + z, cy + z, cz + t], -1)


def _panel(cx, cy, cz, du, dv, nu, nv, plane='xy'):
    """A filled rectangular panel: nu x nv points centred at (cx,cy,cz), half-extents du,dv."""
    u, v = np.meshgrid(np.linspace(-du, du, nu), np.linspace(-dv, dv, nv))
    u, v, o = u.ravel(), v.ravel(), np.zeros(nu * nv)
    if plane == 'xy':
        return np.stack([cx + u, cy + v, cz + o], -1)
    if plane == 'xz':
        return np.stack([cx + u, cy + o, cz + v], -1)
    return np.stack([cx + o, cy + u, cz + v], -1)        # yz


def iss_points():
    """ISS (~109 x 73 m): long port-starboard truss, module stack along flight, four big solar wings."""
    parts = [_line('y', -54, 54, 13),                    # main truss
             _line('x', -22, 30, 7, cz=6.0)]             # pressurised modules, nadir-side
    for ys in (-46.0, 46.0):                             # four solar wings, fore & aft at the ends
        for xs in (-22.0, 22.0):
            parts.append(_panel(xs, ys, 0.0, 16, 9, 4, 3))
    return np.concatenate(parts).astype(np.float32)


def tiangong_points():
    """Tiangong / CSS (~37 m): T-shape -- Tianhe core along flight, two lab modules across, solar wings."""
    parts = [_line('x', -9, 9, 6),                       # Tianhe core (along flight)
             _line('y', -10, 10, 7, cx=8.0)]             # Wentian / Mengtian labs (across, at the node)
    for ys in (-20.0, 20.0):                             # two large solar wings on the lab ends
        parts.append(_panel(8.0, ys, 0.0, 7, 7, 3, 3))
    return np.concatenate(parts).astype(np.float32)


def hubble_points():
    """Hubble (~13 m): the optical tube with two small solar panels mid-body."""
    parts = [_line('x', -6.5, 6.5, 8)]                   # the tube
    for ys in (-4.0, 4.0):                               # two side solar panels
        parts.append(_panel(0.0, ys, 0.0, 3, 2, 3, 2))
    return np.concatenate(parts).astype(np.float32)


def starship_points():
    """Starship (~50 x 9 m): a cylinder of rings with a nose taper, no solar panels."""
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    rings = []
    for x in np.linspace(-25, 25, 9):
        r = 4.5 if x < 18 else 2.0                       # taper toward the nose (+x)
        rings.append(np.stack([np.full_like(ang, x), r * np.cos(ang), r * np.sin(ang)], -1))
    return np.concatenate(rings).astype(np.float32)


_MODELS = (
    (('ISS', 'ZARYA'), iss_points),
    (('TIANHE', 'TIANGONG', 'CSS'), tiangong_points),
    (('HUBBLE', 'HST'), hubble_points),
    (('STARSHIP',), starship_points),
)


def points_for_name(name):
    """Pick a body model by TLE object name (case-insensitive substring); unknown -> a single point
    (renders as a dot, the old behaviour)."""
    n = (name or '').upper()
    for keys, fn in _MODELS:
        if any(k in n for k in keys):
            return fn()
    return np.zeros((1, 3), dtype=np.float32)
