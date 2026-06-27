"""
Sky simulator sanity: a star the encoder points at lands at the boresight, and other
bright in-frame stars land where the projection predicts.

Requires Skyfield + torch, and a one-time network download (de421 ephemeris + Hipparcos
catalog) cached under data/skyfield_cache. Run directly:

    python -m astrolock.seeker.tests.test_skysim
"""

import math

import numpy as np
import torch

from astrolock.seeker.skysim import SkySim, SkySimConfig


def test_star_projects_to_boresight():
    cfg = SkySimConfig()                       # level tripod, no offsets, image-center boresight
    sim = SkySim(cfg)

    t = sim._sf_time(0.0)
    alt, az, mag = sim.sources_altaz(t)        # torch tensors
    up = alt > math.radians(40)
    assert bool(up.any()), "no stars above 40 deg at the configured epoch/location"
    masked = torch.where(up, mag, torch.full_like(mag, float('inf')))
    i = int(torch.argmin(masked))              # brightest star well above the horizon

    frame = sim.render(0.0, float(az[i]), float(alt[i]), exposure_s=0.2, substeps=1)
    cx, cy = cfg.width // 2, cfg.height // 2

    win = frame[cy - 8:cy + 9, cx - 8:cx + 9]
    assert int(win.max()) > 1000, f"no bright peak at boresight: {int(win.max())}"
    ly, lx = np.unravel_index(int(win.argmax()), win.shape)
    assert abs(lx - 8) <= 3 and abs(ly - 8) <= 3, f"peak off-center at ({lx},{ly})"

    # An off-center bright star should appear at its projected pixel.
    b, A, L = sim.boresight_basis(float(az[i]), float(alt[i]))
    px, py, vis = sim.project(alt, az, b, A, L)
    px, py, vis, mag_n = px.numpy(), py.numpy(), vis.numpy(), mag.numpy()
    inframe = (vis & (px > 30) & (px < cfg.width - 30)
               & (py > 30) & (py < cfg.height - 30) & (mag_n < 4.0))
    inframe[i] = False
    js = np.where(inframe)[0]
    if len(js):
        j = int(js[mag_n[js].argmin()])
        x, y = int(round(px[j])), int(round(py[j]))
        patch = frame[y - 5:y + 6, x - 5:x + 6]
        assert int(patch.max()) > 300, f"no flux at projected star: mag {mag_n[j]:.1f}, {int(patch.max())}"


if __name__ == '__main__':
    test_star_projects_to_boresight()
    print("test_skysim: OK")
