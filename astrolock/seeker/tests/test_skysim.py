"""
Sky renderer + ephemeris sanity (no network / no Skyfield -- propagation lives in sky_sim now):

  - a source direction at the boresight lands at image centre; an off-axis source lands at its
    gnomonic-projected pixel.
  - the ephemeris lerps a target's direction between anchors.

    python -m astrolock.seeker.tests.test_skysim
"""

import math
import os
import tempfile

import numpy as np
import torch

from astrolock.seeker.skysim import SkySim, SkySimConfig
from astrolock.seeker.ephemeris import SkyEphemeris, anchor_record
from astrolock.seeker.sidecar import JsonlWriter


def test_source_projects_to_boresight():
    cfg = SkySimConfig(width=512, height=512)      # level tripod, no offsets, image-center boresight
    sim = SkySim(cfg)
    enc_az, enc_alt = math.radians(30.0), math.radians(50.0)
    b, A, L = sim.boresight_basis(enc_az, enc_alt)     # (3,) each
    cx, cy = cfg.width // 2, cfg.height // 2

    # a source exactly at the boresight -> image centre
    dirs = b.view(1, 1, 3).contiguous()
    frame = sim.render(enc_az, enc_alt, 0.0, 0.0, dirs, torch.tensor([2.0]), exposure_s=0.2, substeps=1)
    win = frame[cy - 8:cy + 9, cx - 8:cx + 9]
    assert int(win.max()) > 1000, f"no bright peak at boresight: {int(win.max())}"
    ly, lx = np.unravel_index(int(win.argmax()), win.shape)
    assert abs(lx - 8) <= 2 and abs(ly - 8) <= 2, f"boresight peak off-center at ({lx},{ly})"

    # a source 0.5 deg toward image-right (A) -> at cx + f_px*tan(0.5 deg), same row
    off = math.radians(0.5)
    d2 = (b * math.cos(off) + A * math.sin(off)).view(1, 1, 3).contiguous()
    frame2 = sim.render(enc_az, enc_alt, 0.0, 0.0, d2, torch.tensor([2.0]), exposure_s=0.2, substeps=1)
    py, px = np.unravel_index(int(frame2.argmax()), frame2.shape)
    exp_x = cx + sim.f_px * math.tan(off)
    assert abs(px - exp_x) <= 3, f"off-axis peak at x={px}, expected {exp_x:.1f}"
    assert abs(py - cy) <= 3, f"off-axis peak drifted in y: {py}"


def test_ephemeris_lerp():
    d = tempfile.mkdtemp()
    path = os.path.join(d, 'eph.jsonl')
    w = JsonlWriter(path)
    w.append(anchor_record('a', 1.0, [1_000_000_000, 3_000_000_000], [[1, 0, 0], [0, 1, 0]]))
    w.close()
    e = SkyEphemeris(path)
    e.update()
    dirs, _ = e.dirs_at(torch.tensor([2_000_000_000], dtype=torch.int64))   # midway
    v = dirs[0, 0].tolist()
    assert abs(v[0] - 0.70711) < 0.01 and abs(v[1] - 0.70711) < 0.01 and abs(v[2]) < 0.01, v


if __name__ == '__main__':
    test_source_projects_to_boresight()
    test_ephemeris_lerp()
    print("test_skysim: OK")
