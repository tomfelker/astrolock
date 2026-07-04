"""
astrolock_seeker_detect_probe: an offline harness for developing/tuning detection on a .ser recording.

The live pipeline (detect.py) is already a pure file->file filter, but it needs a session dir + sidecars
and it writes JSONL, not something you can look at. This tool points straight at a bare .ser and gives you
the two views you actually need while tuning a detector on real sky:

  movermap  -- max of consecutive-frame differences over a frame range. Static terrain and slowly-drifting
               stars stay dim; anything that MOVES fast (a satellite) draws a bright track. Best first look
               to find where/when a target is, and at what angle vs the star drift.

  detect    -- run the real detect.py detection surface + blob finder (same code the tracker uses) on each
               frame, overlay the reported blobs on a contrast-stretched render, and print a per-frame
               detection summary. Pass the same --detector/--snr/--bg-radius/... flags as detect.py; whatever
               you tune here transfers verbatim to the live tracker.

Renders go to --out (PNG). Everything is CPU torch + PIL, no dpg/sockets, so it runs anywhere.

    python -m astrolock.seeker.detect_probe movermap FILE.ser --out out/
    python -m astrolock.seeker.detect_probe detect  FILE.ser --frames 40:70 --detector doh --snr 6 --out out/
"""

import argparse
import os

import numpy as np
import torch

from astrolock.seeker import ser as ser_mod
from astrolock.seeker import detect as det


def _load_png(path, arr):
    from PIL import Image
    Image.fromarray(arr).save(path)


def _stretch(a, lo=40.0, hi=99.7):
    a = a.astype(np.float32)
    l, h = np.percentile(a, lo), np.percentile(a, hi)
    return np.clip((a - l) / max(1e-6, h - l), 0.0, 1.0)


def _to_gray_u8(a, lo=40.0, hi=99.7):
    return (_stretch(a, lo, hi) * 255).astype(np.uint8)


def _parse_frames(spec, n):
    """'a:b' | 'a:b:step' | 'a' | '' -> a list of frame indices within [0, n)."""
    if not spec:
        return list(range(n))
    p = spec.split(':')
    a = int(p[0]) if p[0] else 0
    b = int(p[1]) if len(p) > 1 and p[1] else n
    s = int(p[2]) if len(p) > 2 and p[2] else 1
    return [i for i in range(a, min(b, n), s)]


def cmd_movermap(args):
    r = ser_mod.SerReader(args.ser)
    n = r.frames_on_disk()
    frames = _parse_frames(args.frames, n)
    H, W = r.header.image_height, r.header.image_width
    print(f"[probe] {os.path.basename(args.ser)}: {W}x{H} depth {r.header.pixel_depth_per_plane} "
          f"frames {n}; mover-map over {len(frames)} frames")

    def blurred(i):
        f = torch.from_numpy(np.asarray(r.read_frame(i)).astype(np.float32))
        return det.box_blur(f, args.blur) if args.blur > 0 else f

    prev = blurred(frames[0])
    dmax = torch.zeros(H, W)
    activity = []
    for i in frames[1:]:
        cur = blurred(i)
        d = (cur - prev).clamp(min=0)
        dmax = torch.maximum(dmax, d)
        activity.append((i, float(d.max())))
        prev = cur
    os.makedirs(args.out, exist_ok=True)
    _load_png(os.path.join(args.out, 'movermap.png'), _to_gray_u8(dmax.numpy(), 40, 99.8))
    act = np.array([a[1] for a in activity]) if activity else np.array([0.0])
    med = np.median(act); mad = 1.4826 * np.median(np.abs(act - med)) + 1e-6
    top = sorted(activity, key=lambda t: -t[1])[:args.top]
    print(f"[probe] wrote movermap.png. most-active frames (fast movers), idx: sigma-over-median:")
    for i, v in top:
        print(f"          frame {i:4d}   {(v - med) / mad:5.1f} sigma")


def cmd_detect(args):
    r = ser_mod.SerReader(args.ser)
    n = r.frames_on_disk()
    frames = _parse_frames(args.frames, n)
    scale = det.full_scale(r.header.color_id, r.header.pixel_depth_per_plane)
    os.makedirs(args.out, exist_ok=True)
    print(f"[probe] detect on {len(frames)} frames  detector={args.detector} snr={args.snr} "
          f"bg-radius={args.bg_radius} min-roundness={args.min_roundness} max-size-px={args.max_size_px}"
          + (f" decay={args.surprise_decay}" if args.detector == 'surprise' else ""))
    # 'surprise' is a stateful temporal detector: it must see frames in order from the start of the
    # range, and needs ~1/alpha frames of warm-up before its per-pixel variance settles.
    surprise = (det.SurpriseModel(args.surprise_alpha_mean, args.surprise_alpha_var,
                                  args.surprise_decay, args.surprise_var_floor_frac)
                if args.detector == 'surprise' else None)
    prev_bp = None
    total = 0
    for i in frames:
        frame = np.asarray(r.read_frame(i))
        work = det.work_image(frame, r.header.color_id)
        if surprise is not None:
            surf = surprise.update(work)
        else:
            surf = det.detection_surface(work, detector=args.detector, bg_radius=args.bg_radius,
                                         psf_px=args.psf_px, doh_sigma=args.doh_sigma)
        blobs = det.detect_blobs(
            surf, work, (None if surprise is not None else prev_bp),
            threshold_rel=args.threshold, max_candidates=args.max_candidates,
            suppress_radius=args.suppress_radius, min_blob_px=args.min_blob_px,
            max_size_px=args.max_size_px, psf_px=args.psf_px, snr=args.snr,
            min_roundness=args.min_roundness, moving_frac=args.moving_frac, scale=scale,
            tile_grid=args.tile_grid, per_tile=args.per_tile)
        prev_bp = surf
        total += len(blobs)
        if args.save_frames:                          # for surprise, draw on the trail surface (sat visible there)
            base = surf.numpy() if surprise is not None else work.numpy()
            _save_overlay(os.path.join(args.out, f'det_{i:04d}.png'), base, blobs)
        if blobs or args.verbose:
            desc = ', '.join(f"({b['px'][0]:.0f},{b['px'][1]:.0f}) sz{b.get('size_px', 0):.0f} "
                             f"rnd{b.get('roundness', 0):.2f}" for b in blobs[:6])
            print(f"  frame {i:4d}: {len(blobs):2d} blobs  {desc}")
    print(f"[probe] {total} detections across {len(frames)} frames "
          f"({total / max(1, len(frames)):.1f}/frame)")


def _save_overlay(path, work, blobs):
    from PIL import Image, ImageDraw
    g = _to_gray_u8(work, 40, 99.7)
    im = Image.fromarray(g).convert('RGB')
    d = ImageDraw.Draw(im)
    for b in blobs:
        x, y = b['px']
        rad = max(6.0, 0.5 * b.get('size_px', 8.0))
        d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(0, 255, 0))
    im.save(path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Offline detection probe/harness for a .ser recording")
    sub = p.add_subparsers(dest='cmd', required=True)

    pm = sub.add_parser('movermap', help="max consecutive-frame difference -> a map of fast movers")
    pm.add_argument('ser')
    pm.add_argument('--frames', default='', help="frame range a:b[:step] (default all)")
    pm.add_argument('--blur', type=int, default=1, help="pre-diff box-blur radius (noise suppression)")
    pm.add_argument('--top', type=int, default=12, help="how many most-active frames to list")
    pm.add_argument('--out', default='probe_out')
    pm.set_defaults(func=cmd_movermap)

    pd = sub.add_parser('detect', help="run the real detect.py surface+blobs; overlay + summarize")
    pd.add_argument('ser')
    pd.add_argument('--frames', default='', help="frame range a:b[:step] (default all)")
    pd.add_argument('--out', default='probe_out')
    pd.add_argument('--save-frames', action='store_true', help="write a det_<i>.png overlay per frame")
    pd.add_argument('--verbose', action='store_true', help="print frames even with 0 blobs")
    # detector knobs -- mirror detect.py so tuning transfers verbatim
    pd.add_argument('--detector', default='doh', choices=['bandpass', 'doh', 'surprise'])
    pd.add_argument('--surprise-alpha-mean', type=float, default=0.15)
    pd.add_argument('--surprise-alpha-var', type=float, default=0.08)
    pd.add_argument('--surprise-decay', type=float, default=0.85)
    pd.add_argument('--surprise-var-floor-frac', type=float, default=0.5)
    pd.add_argument('--snr', type=float, default=6.0)
    pd.add_argument('--threshold', type=float, default=0.0)
    pd.add_argument('--bg-radius', type=int, default=12)
    pd.add_argument('--doh-sigma', type=float, default=0.0)
    pd.add_argument('--psf-px', type=float, default=3.0)
    pd.add_argument('--max-candidates', type=int, default=16)
    pd.add_argument('--tile-grid', type=int, default=8)
    pd.add_argument('--per-tile', type=int, default=2)
    pd.add_argument('--suppress-radius', type=int, default=6)
    pd.add_argument('--min-blob-px', type=int, default=2)
    pd.add_argument('--max-size-px', type=float, default=0.0)
    pd.add_argument('--min-roundness', type=float, default=0.0)
    pd.add_argument('--moving-frac', type=float, default=0.5)
    pd.set_defaults(func=cmd_detect)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
