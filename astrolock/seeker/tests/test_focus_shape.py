"""Synthetic ground-truth tests for focus.py's shape estimators: the toroidal find and
the Gaussian-weighted moment fit (centroid / ellipse / skew).

The assertions check the properties the consumers actually rely on -- sub-pixel find under
a heavy background pedestal, exact ellipse orientation, pedestal invariance, the
symmetric-image skew null, skew sign + monotonicity (the collimation servo's needs), and
donut (defocused SCT) behaviour -- NOT absolute shape magnitudes, which are attenuated by
the measuring window by design (the GUI exaggeration knobs absorb that).

    python -m astrolock.seeker.tests.test_focus_shape
"""

import math

import torch

from astrolock.seeker.focus import _toroidal_com, _weighted_shape


def gauss2(h, w, cx, cy, sxx, syy, sxy, flux):
    ys = torch.arange(h, dtype=torch.float32)[:, None]
    xs = torch.arange(w, dtype=torch.float32)[None, :]
    det = sxx * syy - sxy * sxy
    a, b, c = syy / det, sxx / det, -sxy / det
    dx, dy = xs - cx, ys - cy
    g = torch.exp(-0.5 * (a * dx * dx + b * dy * dy + 2 * c * dx * dy))
    return flux * g / g.sum()


def test_toroidal_find_under_pedestal():
    torch.manual_seed(0)
    img = gauss2(64, 64, 40.3, 22.7, 9, 9, 0, 1000.0) + 50.0 + torch.randn(64, 64) * 0.5
    x, y = _toroidal_com(img)
    assert abs(x - 40.3) < 0.5 and abs(y - 22.7) < 0.5, (x, y)


def test_ellipse_orientation_and_null_skew():
    torch.manual_seed(0)
    theta = math.radians(30)
    s1, s2 = 16.0, 4.0
    sxx = s1 * math.cos(theta) ** 2 + s2 * math.sin(theta) ** 2
    syy = s1 * math.sin(theta) ** 2 + s2 * math.cos(theta) ** 2
    sxy = (s1 - s2) * math.sin(theta) * math.cos(theta)
    star = gauss2(64, 64, 31.5, 31.5, sxx, syy, sxy, 2000.0) + 20.0 + torch.randn(64, 64) * 0.2
    sh = _weighted_shape(star, 3.0)
    measured_theta = math.degrees(0.5 * math.atan2(sh['e2'], sh['e1']))
    assert abs(measured_theta - 30.0) < 3.0, measured_theta
    assert math.hypot(sh['e1'], sh['e2']) > 0.2          # attenuated vs raw 0.6, but loud
    assert math.hypot(sh['sx'], sh['sy']) < 0.05         # symmetric image -> skew null


def test_pedestal_invariance():
    a = _weighted_shape(gauss2(64, 64, 31.5, 31.5, 9, 4, 0, 2000.0), 3.0)
    b = _weighted_shape(gauss2(64, 64, 31.5, 31.5, 9, 4, 0, 2000.0) + 500.0, 3.0)
    assert abs(a['e1'] - b['e1']) < 0.01 and abs(a['e2'] - b['e2']) < 0.01


def test_skew_sign_and_monotonicity():
    """The collimation servo needs skew's SIGN (which way) and monotonicity (more flare ->
    more skew); absolute magnitude is display-gained."""
    def skew_x(flare_flux, side):
        blob = (gauss2(64, 64, 31.5, 31.5, 4, 4, 0, 1000.0)
                + gauss2(64, 64, 31.5 + side * 2.0, 31.5, 9, 9, 0, flare_flux) + 20.0)
        s = _weighted_shape(blob, 3.0)
        assert abs(s['sy']) < 0.05, s                    # no cross-axis leak
        return s['sx']

    weak, strong = skew_x(200.0, +1), skew_x(400.0, +1)
    assert weak > 0.02 and strong > weak, (weak, strong)         # sign + monotone
    assert skew_x(400.0, -1) < -0.02                             # mirrored flare flips sign


def test_donut_centre_and_astigmatic_donut():
    ys = torch.arange(64, dtype=torch.float32)[:, None]
    xs = torch.arange(64, dtype=torch.float32)[None, :]
    r = torch.sqrt((xs - 28.0) ** 2 + (ys - 36.0) ** 2)
    donut = torch.exp(-0.5 * ((r - 10.0) / 2.0) ** 2) * 100.0 + 30.0
    x, y = _toroidal_com(donut)                          # the CENTRE, not a rim point
    assert abs(x - 28.0) < 0.5 and abs(y - 36.0) < 0.5, (x, y)
    r_ell = torch.sqrt(((xs - 31.5) / 1.3) ** 2 + ((ys - 31.5) * 1.3) ** 2)
    donut_e = torch.exp(-0.5 * ((r_ell - 10.0) / 2.0) ** 2) * 100.0 + 30.0
    s = _weighted_shape(donut_e, 6.0)                    # x-elongated -> theta ~ 0
    theta = math.degrees(0.5 * math.atan2(s['e2'], s['e1']))
    assert math.hypot(s['e1'], s['e2']) > 0.1 and abs(theta) < 5.0, (s, theta)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print(f"  {name} OK")
    print("test_focus_shape: all tests passed")


if __name__ == '__main__':
    main()
