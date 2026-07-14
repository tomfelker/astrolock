"""
Target models: the GreatCircle (constant-altitude) model on a synthetic LEO pass.

    python -m astrolock.seeker.tests.test_target_model
"""

import math

import torch

from astrolock.seeker import geometry as geo
from astrolock.seeker.target_model import EmaAngularVelModel, GreatCircleModel

RE = 6371e3
ALT = 250e3
R = RE + ALT
OMEGA = 7754.0 / R                       # circular-orbit speed at 250 km -> rad/s about Earth centre


def _truth_dir(t):
    """A pass straight through the zenith: u(0) = z-hat, rotating about y-hat."""
    u0 = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    u = geo.rodrigues(u0, axis, torch.as_tensor(OMEGA * t, dtype=torch.float64))
    return geo.normalize(R * u - torch.tensor([0.0, 0.0, RE], dtype=torch.float64))


def _angle(a, b):
    return float(torch.rad2deg(torch.atan2(torch.linalg.norm(torch.linalg.cross(a, b)),
                                           torch.dot(a, b))))


def test_projection_roundtrip():
    m = GreatCircleModel(altitude_m=ALT, earth_radius_m=RE)
    for t in (-30.0, -5.0, 0.0, 3.0, 20.0):
        d = _truth_dir(t)
        assert _angle(m._dir_of(m._project(d)), d) < 1e-9


def test_great_circle_pass():
    """Ingest an approach to the zenith; predict onward. The orbital rate is constant, so the
    model should nail it -- and beat the constant-SKY-rate EMA, whose sky rate is changing
    fastest right there (the zenith speed-up is perspective, not dynamics)."""
    gc = GreatCircleModel(smoothing_s=0.5, altitude_m=ALT, earth_radius_m=RE)
    ema = EmaAngularVelModel(smoothing_s=0.5)
    for k in range(80):                                  # -8s .. 0s at 10 Hz, culminating at t=0
        t = -8.0 + 0.1 * k
        d = _truth_dir(t)
        gc.ingest(t, d)
        ema.ingest(t, d)
    t_pred = 1.9                                          # ~2s ahead, just past culmination
    truth = _truth_dir(t_pred)
    gc_err = _angle(gc.predict(t_pred), truth)
    ema_err = _angle(ema.predict(t_pred), truth)
    assert gc_err < 0.05, f"great-circle predict-ahead error {gc_err:.3f} deg"
    assert gc_err < ema_err / 10, f"gc {gc_err:.3f} vs ema {ema_err:.3f} deg"

    # Batched predict (the servo's horizon sweep) agrees with scalar predict.
    ts = torch.tensor([0.5, 1.0, 1.9], dtype=torch.float64)
    batch = gc.predict(ts)
    assert batch.shape == (3, 3)
    assert _angle(batch[2], gc.predict(1.9)) < 1e-9


if __name__ == '__main__':
    test_projection_roundtrip()
    test_great_circle_pass()
    print("test_target_model: OK")
