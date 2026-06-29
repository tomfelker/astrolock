"""
detect on a synthetic capture: the moving blob should be found, tracked, and flagged
moving. Runs as a pytest test or directly:

    python -m astrolock.seeker.tests.test_detect
"""

import glob
import os

import numpy as np

from astrolock.seeker import cam, detect, ser, sidecar
from astrolock.seeker.tests._util import fresh_dir


def _add_gaussian(img, cx, cy, peak, sigma):
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    img += peak * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)))


def test_detect_tracks_moving_blob():
    out = fresh_dir('detect')
    # 1 s at 30 fps so the synthetic blob actually moves between frames
    cam.main(['--role', 'guide', '--out-dir', out, '--width', '320', '--height', '240',
              '--fps', '30', '--frame-limit', '30'])

    # Offline: detect runs to completion (the .ser header is finalized) and exits.
    detect.main(['--session', out, '--role', 'guide', '--moving-frac', '0.1'])

    recs = sidecar.read_complete_lines(glob.glob(os.path.join(out, '*_guide.detections.jsonl'))[0])
    assert len(recs) == 30, len(recs)
    assert all(r['blobs'] for r in recs), "every frame should find at least one blob"

    top = [r['blobs'][0] for r in recs]            # brightest blob per frame (greedy order)
    assert all(b['score'] > 0.4 for b in top), "the synthetic blob is bright"

    xs = [b['px'][0] for b in top]
    assert max(xs) - min(xs) > 20, f"blob should sweep across x, got range {max(xs) - min(xs)}"

    moving = [b['moving'] for b in top[1:]]         # first frame has no prev
    assert all(m is not None for m in moving)
    assert sum(bool(m) for m in moving) >= 15, f"most frames should read as moving: {sum(bool(m) for m in moving)}"


def test_detect_rejects_extended_clutter():
    # Wide-FOV scene: a big bright "rooftop" slab plus one faint-ish point source.
    work = np.zeros((200, 200), np.float32)
    work[20:120, 20:120] = 40000.0                 # extended bright clutter
    _add_gaussian(work, 160, 50, 60000.0, 2.0)     # a pointlike source at (x=160, y=50)

    bp = detect.band_pass(work, 12)
    scale = ser.container_max(16)
    blobs = detect.detect_blobs(
        bp, work, None, threshold_rel=0.3, max_candidates=16, suppress_radius=6,
        min_blob_px=2, max_size_px=6.0, psf_px=5.0, moving_frac=0.5, scale=scale)

    assert blobs, "should detect the point source"
    bx, by = blobs[0]['px']                         # brightest band-pass peak
    assert abs(bx - 160) <= 3 and abs(by - 50) <= 3, (bx, by)
    assert blobs[0]['pointlike'] > 0.7
    assert blobs[0]['size_px'] < 6

    # The flat interior of the slab must not be detected as a (huge) blob.
    for b in blobs:
        cx, cy = b['px']
        assert not (40 < cx < 100 and 40 < cy < 100), f"spurious blob in slab interior at {b['px']}"


def test_roundness_rejects_streak():
    # A thin bright streak (wire/edge ridge) plus a round point source.
    work = np.zeros((200, 200), np.float32)
    work[100:102, 60:100] = 60000.0                # 2 px tall, 40 px wide -> elongated
    _add_gaussian(work, 150, 150, 60000.0, 2.0)    # round point

    bp = detect.band_pass(work, 12)
    scale = ser.container_max(16)
    common = dict(threshold_rel=0.3, max_candidates=16, suppress_radius=6,
                  min_blob_px=2, max_size_px=0.0, psf_px=5.0, moving_frac=0.5, scale=scale)

    def near_streak(b):
        return 60 <= b['px'][0] <= 100 and 95 <= b['px'][1] <= 105

    def near_point(b):
        return abs(b['px'][0] - 150) <= 3 and abs(b['px'][1] - 150) <= 3

    # No cut: both are detected, but the streak pieces read as low roundness, the point high.
    allb = detect.detect_blobs(bp, work, None, **common)
    pts = [b for b in allb if near_point(b)]
    strk = [b for b in allb if near_streak(b)]
    assert pts and pts[0]['roundness'] > 0.6, pts
    assert strk and all(b['roundness'] < 0.4 for b in strk), strk

    # With the roundness cut: the streak is gone, the point survives.
    cut = detect.detect_blobs(bp, work, None, min_roundness=0.5, **common)
    assert any(near_point(b) for b in cut)
    assert not any(near_streak(b) for b in cut), "streak should be rejected by roundness"


def test_doh_detects_blobs_rejects_edge():
    # Determinant-of-Hessian surface: peaks on round blobs, ~0 along an edge (one curvature
    # vanishes), so a long thin ridge must not register the way it would in a band-pass.
    work = np.zeros((200, 200), np.float32)
    _add_gaussian(work, 60, 60, 60000.0, 2.0)          # round blob A
    _add_gaussian(work, 140, 120, 50000.0, 2.5)        # round blob B
    work[150:152, 30:170] = 60000.0                    # a long thin horizontal edge/ridge

    doh = detect.det_of_hessian(work, sigma=3.0)
    # Direct property: the middle of the edge is ~flat in DoH compared to a blob center.
    assert float(doh[60, 60]) > 0.0
    assert abs(float(doh[150, 100])) < 0.1 * float(doh[60, 60]), \
        f"edge middle DoH {float(doh[150, 100]):.3g} not << blob {float(doh[60, 60]):.3g}"

    scale = ser.container_max(16)
    # A relative floor (the synthetic background is perfectly flat, so the MAD-sigma SNR cut would
    # collapse; real data has noise). The edge middle is ~3 orders below a blob, so it's rejected.
    blobs = detect.detect_blobs(
        doh, work, None, threshold_rel=0.05, max_candidates=16, suppress_radius=6,
        min_blob_px=1, max_size_px=0.0, psf_px=4.0, snr=8.0, moving_frac=0.5, scale=scale)

    def near(b, x, y):
        return abs(b['px'][0] - x) <= 3 and abs(b['px'][1] - y) <= 3

    assert any(near(b, 60, 60) for b in blobs), f"missed blob A: {[b['px'] for b in blobs]}"
    assert any(near(b, 140, 120) for b in blobs), f"missed blob B: {[b['px'] for b in blobs]}"
    # No detection along the middle stretch of the edge (its ends/corners may legitimately respond).
    for b in blobs:
        assert not (50 <= b['px'][0] <= 150 and 148 <= b['px'][1] <= 154), \
            f"DoH picked the edge middle at {b['px']}"


def test_doh_surface_selectable():
    # The detection_surface dispatcher returns the DoH map for detector='doh'.
    work = np.zeros((64, 64), np.float32)
    _add_gaussian(work, 32, 32, 50000.0, 2.0)
    bp = detect.detection_surface(work, detector='bandpass', bg_radius=12, psf_px=4.0, doh_sigma=0.0)
    dh = detect.detection_surface(work, detector='doh', bg_radius=12, psf_px=4.0, doh_sigma=0.0)
    import torch
    assert torch.is_tensor(bp) and torch.is_tensor(dh)
    assert int(dh.reshape(-1).argmax()) == 32 * 64 + 32     # DoH peaks at the blob center


def test_tile_density_keeps_distant_target():
    # A dense cluster of bright blobs (a "tree") in one corner plus one dimmer lone star far away.
    # With a small global budget and no tiling the cluster eats it and the star is missed; the
    # per-tile density cap leaves room and the star survives.
    work = np.zeros((200, 200), np.float32)
    for i in range(5):
        for j in range(5):
            _add_gaussian(work, 15 + i * 12, 15 + j * 12, 60000.0, 2.0)  # cluster in the top-left
    _add_gaussian(work, 175, 175, 30000.0, 2.0)                          # lone (dimmer) star

    surf = detect.detection_surface(work, detector='doh', bg_radius=12, psf_px=3.0, doh_sigma=2.0)
    scale = ser.container_max(16)
    common = dict(threshold_rel=0.02, suppress_radius=4, min_blob_px=1, max_size_px=0.0,
                  psf_px=3.0, snr=6.0, moving_frac=0.5, scale=scale)

    def near_star(b):
        return abs(b['px'][0] - 175) <= 4 and abs(b['px'][1] - 175) <= 4

    no_tile = detect.detect_blobs(surf, work, None, max_candidates=8, **common)
    assert not any(near_star(b) for b in no_tile), "without tiling the cluster should starve the star"

    tiled = detect.detect_blobs(surf, work, None, max_candidates=8, tile_grid=4, per_tile=1, **common)
    assert any(near_star(b) for b in tiled), "the density cap should preserve the distant star"


if __name__ == '__main__':
    test_detect_tracks_moving_blob()
    test_detect_rejects_extended_clutter()
    test_roundness_rejects_streak()
    test_doh_detects_blobs_rejects_edge()
    test_doh_surface_selectable()
    test_tile_density_keeps_distant_target()
    print("test_detect: OK")
