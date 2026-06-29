"""
Bayer helpers for AstroLock Seeker.

We keep the raw mosaic in the .ser (no interpolated / made-up data) and decode it on the
read side. The natural decode is the **4-plane split**: each 2x2 Bayer cell becomes four
co-located planes at half width / half height. This is lossless (every output value is a
real sensor reading), cheap, and the same representation used in focus.py / tensorez. From
the planes you can make a half-res RGB image for display, or sum them for a sensitive
half-res mono for detection.

Plane positions within a 2x2 cell: ul ur / ll lr.
"""

import torch

from astrolock.seeker import ser


def is_bayer(color_id):
    return int(ser.ColorId.BAYER_RGGB) <= int(color_id) <= int(ser.ColorId.BAYER_BGGR)


def split_planes(mosaic):
    """Return (ul, ur, ll, lr), each (H//2, W//2), from a raw Bayer mosaic (H, W).
    Slicing-only, so it works on a torch tensor or numpy array; the rest of this module is torch."""
    even = mosaic[0::2]
    odd = mosaic[1::2]
    return even[:, 0::2], even[:, 1::2], odd[:, 0::2], odd[:, 1::2]


# For each ColorId, which of (ul, ur, ll, lr) the R / two-G / B sites are.
# index 0=ul 1=ur 2=ll 3=lr
_RGGB = dict(r=0, g=(1, 2), b=3)
_GRBG = dict(r=1, g=(0, 3), b=2)
_GBRG = dict(r=2, g=(0, 3), b=1)
_BGGR = dict(r=3, g=(1, 2), b=0)
_LAYOUT = {
    int(ser.ColorId.BAYER_RGGB): _RGGB,
    int(ser.ColorId.BAYER_GRBG): _GRBG,
    int(ser.ColorId.BAYER_GBRG): _GBRG,
    int(ser.ColorId.BAYER_BGGR): _BGGR,
}


def rgb_plane_indices(color_id):
    """(r, (g0, g1), b) -- which of the 4 split planes hold the R / two-G / B sites."""
    layout = _LAYOUT.get(int(color_id))
    if layout is None:
        raise ValueError(f"not a Bayer color_id: {color_id}")
    return layout['r'], layout['g'], layout['b']


def debayer_to_rgb(mosaic, color_id):
    """
    Decode a raw Bayer mosaic to a half-resolution float32 RGB image (H//2, W//2, 3).
    The two green sites are averaged. No interpolation -- this is the 4-plane split recombined.
    """
    layout = _LAYOUT.get(int(color_id))
    if layout is None:
        raise ValueError(f"not a Bayer color_id: {color_id}")
    planes = [p.float() for p in split_planes(mosaic)]
    r = planes[layout['r']]
    g = (planes[layout['g'][0]] + planes[layout['g'][1]]) * 0.5
    b = planes[layout['b']]
    return torch.stack([r, g, b], dim=-1)


def to_mono_sum(mosaic):
    """Half-res mono = sum of the four Bayer planes (sensitive; good for detection)."""
    ul, ur, ll, lr = (p.float() for p in split_planes(mosaic))
    return ul + ur + ll + lr
