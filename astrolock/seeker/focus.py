"""
astrolock_seeker_focus: star-crop EMA + focus/collimation metrics -- a pure image->(ser+json) filter.

Follows one camera's .ser (live or finalized, exactly like detect) and that role's
<seg>_<role>.detections.jsonl (to know where the target is), and for each frame:
  - crops a search ROI around the target (the detector's strongest blob; else the frame's
    global brightest pixel, so it locks a star even before any detection),
  - finds the peak in that ROI and re-crops a small star window centred there,
  - EMA-stacks those star crops into a "lucky" average star.
This is the core of the standalone focus.py, lifted into the seeker pipeline.

It writes two files per segment (both discovered by SerFollower / the sidecar helpers, so the
GUI can PIP the star image and graph the metrics -- same naming convention as the _debug streams):
  - <seg>_<role>_focus.ser          : the EMA star image (mono 16-bit; auto-normalized so the
                                      shape is always visible -- the quantitative peak is a metric)
  - <seg>_<role>_focus.frames.jsonl : the commit spine AND the per-frame metrics --
                                      peak      (focus quality: brightness of the EMA's peak, 0..1),
                                      focus_x/y (per-axis sharpness: peak of the col/row-sum profile),
                                      com       ([dx, dy] px: CoM offset of the EMA = collimation),
                                      present   (True = locked a real detection this frame, not the
                                                 global-max fallback).

No mount, no sky model, no sockets -- files in, files out, so it runs identically on a live capture
or a recording (where you'd tune it offline).

    python -m astrolock.seeker.focus --session sessions/<ts> --role main
"""

import argparse
import glob
import os
import time

import torch

from astrolock.seeker import bayer, ser as ser_mod, skysim
from astrolock.seeker.detect import (resolve_device, work_image, full_scale,
                                     _segments, _committed, _segment_ready, _frame_count)
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def _effective_crop(header, crop):
    """The star crop actually used: the requested size, clamped odd to fit the (maybe Bayer-halved)
    analysis image, so the EMA and the output .ser always agree on dimensions even for tiny frames."""
    d = 2 if bayer.is_bayer(header.color_id) else 1
    h, w = header.image_height // d, header.image_width // d
    c = min(crop, h if h % 2 else h - 1, w if w % 2 else w - 1)
    return max(1, c)


def _crop(work, cx, cy, size):
    """Crop a size x size window centred at (cx, cy), clamped to stay on the image. Returns
    (crop, x0, y0); size is forced odd so the centre pixel is well-defined."""
    H, W = work.shape
    size = min(size, H if H % 2 else H - 1, W if W % 2 else W - 1)   # fit, keep odd
    half = size // 2
    x = min(max(int(round(cx)), half), W - half - 1)
    y = min(max(int(round(cy)), half), H - half - 1)
    x0, y0 = x - half, y - half
    return work[y0:y0 + size, x0:x0 + size], x0, y0


def _com_offset(img):
    """CoM of a 2-D non-negative image relative to its geometric centre, px (dx right, dy down).
    The window is centred + odd, so a symmetric background pedestal cancels (as in focus.py)."""
    H, W = img.shape
    dev = img.device
    tot = float(img.sum())
    if tot <= 0:
        return 0.0, 0.0
    ys = torch.arange(H, dtype=torch.float32, device=dev) - (H - 1) / 2.0
    xs = torch.arange(W, dtype=torch.float32, device=dev) - (W - 1) / 2.0
    dy = float((img.sum(dim=1) * ys).sum()) / tot
    dx = float((img.sum(dim=0) * xs).sum()) / tot
    return dx, dy


def _normalized_peak(crop):
    """Subtract the sky pedestal (mean of the crop's border pixels), scale so the crop sums to 1, and
    return the peak -- i.e. the fraction of the star's energy in its brightest pixel. Strehl = this for
    the measured star / this for the ideal diffraction PSF. 0 if there's no positive net signal."""
    edge = torch.cat([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    net = crop - edge.mean()
    total = float(net.sum())
    if total <= 0:
        return 0.0
    return float(net.max()) / total


class FocusEma:
    """The running star average + the metrics read off it. update(work, target) each frame returns
    (ema_crop, metrics). ``scale`` (full_scale) normalizes brightness to 0..1 so metrics are
    comparable across cameras / bit depths."""

    def __init__(self, crop, search, alpha, scale, peak_decay=0.9):
        self.crop, self.search, self.alpha, self.scale = crop, search, alpha, scale
        self.peak_decay = peak_decay
        self.ema = None
        self.peak_meter = None      # per-pixel decaying peak-hold of the RAW star crop, for saturation

    def update(self, work, target):
        # 1) search ROI around the target; 2) its brightest pixel; 3) star crop centred there.
        tx, ty = target
        region, rx0, ry0 = _crop(work, tx, ty, self.search)
        pk = int(torch.argmax(region))
        py, px = pk // region.shape[1], pk % region.shape[1]
        star, _, _ = _crop(work, rx0 + px, ry0 + py, self.crop)
        if self.ema is None or self.ema.shape != star.shape:
            self.ema = star.clone()
            self.peak_meter = star.clone()
        else:
            self.ema = torch.lerp(self.ema, star, self.alpha)          # EMA of the star crop
            # Decaying per-pixel peak-hold of the RAW crop: catches a saturating core even between the
            # EMA's slow settle, and holds the warning a moment after it stops clipping.
            self.peak_meter = torch.maximum(self.peak_meter * self.peak_decay, star)
        ema = self.ema
        dx, dy = _com_offset(ema)                                      # collimation: CoM offset (px)
        # peak = brightest pixel of the EMA (0..1 full-scale): the focus-quality proxy, which rises as
        # focus tightens. (Per-axis x/y sharpness was dropped -- astigmatism can lie on a diagonal, so it
        # wants the full second-moment matrix Mxx/Myy/Mxy, not marginals; not worth it unless collimating.)
        metrics = {
            'peak': round(float(ema.max()) / self.scale, 6),
            'com': [round(dx, 3), round(dy, 3)],
        }
        sat = self.peak_meter > (0.9 * self.scale)                    # near full well -> peak unreliable
        return ema, metrics, sat


def _ema_frame_u16(ema, scale, blank=None):
    """The EMA star crop as a uint16 mono image for the .ser, at ABSOLUTE brightness (NOT range-stretched)
    so the user can judge exposure by how bright it looks. ``blank`` (a saturation mask) zeroes those
    pixels -- applied on alternate frames, saturated cores FLASH as a 'peak measurement can't be trusted'
    warning."""
    img = (ema / scale).clamp(0, 1)
    if blank is not None:
        img = torch.where(blank, torch.zeros_like(img), img)
    return (img * 65535.0).round().cpu().numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker focus / collimation filter")
    p.add_argument('--session', required=True, help="session directory to follow")
    p.add_argument('--role', default='main', help="camera role to focus on (tails its .ser + detections)")
    p.add_argument('--follow', action='store_true',
                   help="live mode: track the newest segment and never exit (default: offline, process "
                        "each segment in order then exit)")
    p.add_argument('--crop', type=int, default=63, help="star crop size (px, forced odd); the EMA image")
    p.add_argument('--search', type=int, default=128,
                   help="search ROI around the target to find the peak in (px)")
    p.add_argument('--alpha', type=float, default=0.05, help="EMA rate for the star crop (bigger = faster)")
    # Optics -> the ideal diffraction PSF for the Strehl ratio. Strehl is emitted only when aperture > 0.
    p.add_argument('--aperture-mm', type=float, default=0.0,
                   help="objective aperture (mm); >0 enables the Strehl-ratio metric (measured vs ideal Airy peak)")
    p.add_argument('--focal-mm', type=float, default=0.0, help="effective focal length (mm), for the ideal PSF")
    p.add_argument('--pixel-um', type=float, default=0.0,
                   help="frame pixel pitch (um) at the cam's output binning; scaled to the work image internally")
    p.add_argument('--wavelength-nm', type=float, default=550.0, help="wavelength (nm) for the ideal Airy PSF")
    p.add_argument('--obstruction', type=float, default=0.0,
                   help="central obstruction (secondary/aperture diameter ratio) for the ideal PSF (0 = clear)")
    p.add_argument('--vanes', type=int, default=0, help="spider-vane count for the ideal PSF (Newtonian)")
    p.add_argument('--vane-width', type=float, default=0.0, help="spider-vane width / aperture diameter")
    p.add_argument('--poll', type=float, default=0.02, help="seconds between polls when caught up (live)")
    p.add_argument('--stop-file', default=None, help="stop cleanly when this file appears")
    p.add_argument('--device', default='auto',
                   help="torch device: 'auto' (default) = cuda if present else cpu, or force 'cpu'/'cuda'")
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    print(f"[focus:{args.role}] compute device: {device}", flush=True)

    # Wait for the first ready segment (header + first frame committed), like detect.
    while True:
        if args.stop_file and os.path.exists(args.stop_file):
            return
        ready = [s for s in _segments(args.session, args.role) if _segment_ready(s)]
        if ready:
            break
        time.sleep(args.poll)

    cur = ready[-1] if args.follow else ready[0]

    def open_segment(ser_path):
        reader = ser_mod.SerReader(ser_path)
        stem = ser_path[:-len('.ser')]
        crop = _effective_crop(reader.header, args.crop)    # fits the analysis image; writer + EMA agree
        # Output stream: <seg>_<role>_focus.ser + its .frames.jsonl spine (which also carries metrics).
        writer = ser_mod.SerWriter(stem + '_focus.ser', crop, crop,
                                   color_id=ser_mod.ColorId.MONO, pixel_depth_per_plane=16)
        spine = JsonlWriter(stem + '_focus.frames.jsonl')
        det = JsonlTailer(stem + '.detections.jsonl')       # where the target is (this role's detector)
        ema = FocusEma(crop, args.search, args.alpha, scale=None)
        return reader, writer, spine, det, ema

    reader, writer, spine, det, ema = open_segment(cur)
    latest_blobs = []
    next_index = 0
    scale = None
    total = 0
    strehl_ref = None            # ideal (diffraction-limited) normalized peak; computed once, kept across segments
    strehl_done = False          # False until we've decided whether Strehl is available (needs the plate scale)
    rad_per_px = None            # radians per work px (for the pixel-scale-free CoM offset); None if unknown

    def close_segment():
        reader.close(); writer.close(); spine.close(); det.close()

    def process(i):
        nonlocal scale, total, strehl_ref, strehl_done, rad_per_px
        frame = reader.read_frame(i)
        cid = reader.header.color_id
        if scale is None:
            scale = full_scale(cid, reader.header.pixel_depth_per_plane)
            ema.scale = scale
        work = work_image(frame, cid, device=device)
        coord_scale = reader.header.image_width / work.shape[1]     # frame px per (maybe half-res) work px
        # Strehl reference: the ideal PSF's normalized peak. With a known aperture it's the diffraction
        # Airy at the WORK-image plate scale (work px are coarser than sensor px -- e.g. 2x for a Bayer
        # mono-sum -- hence pixel_um * coord_scale). With the aperture UNKNOWN we assume the best a lens
        # could do is put all the energy in one pixel (a centred delta) -> ideal peak 1.0, so Strehl still
        # reports, just reading low unless the lens is genuinely sharp. Computed once we know coord_scale.
        if not strehl_done:
            strehl_done = True
            if args.focal_mm > 0 and args.pixel_um > 0:   # radians per WORK px, for a pixel-scale-free CoM
                rad_per_px = (args.pixel_um * coord_scale * 1e-3) / args.focal_mm
            if args.aperture_mm > 0 and args.focal_mm > 0 and args.pixel_um > 0:
                r_null = skysim.airy_r_null_px(args.focal_mm, args.aperture_mm, args.wavelength_nm,
                                               args.pixel_um * coord_scale)
                ideal = skysim.aperture_psf(ema.crop, r_null, obstruction=args.obstruction,
                                            vanes=args.vanes, vane_width_frac=args.vane_width, device=device)
            else:
                ideal = torch.zeros((ema.crop, ema.crop), device=device)   # perfect point source
                ideal[ema.crop // 2, ema.crop // 2] = 1.0
            strehl_ref = _normalized_peak(ideal)
        # Target in WORK px: the detector's strongest blob (its px are frame coords), else the frame's
        # global brightest pixel -- so we lock a star even before/without any detection.
        present = bool(latest_blobs)
        if present:
            fx, fy = latest_blobs[0]['px']
            target = (fx / coord_scale, fy / coord_scale)
        else:
            pk = int(torch.argmax(work))
            target = (pk % work.shape[1], pk // work.shape[1])
        star_ema, metrics, sat = ema.update(work, target)
        if strehl_ref:                                 # measured normalized peak vs the ideal (0..~1)
            metrics['strehl'] = round(_normalized_peak(star_ema) / strehl_ref, 4)
        if rad_per_px is not None:                     # CoM offset in radians (pixel-scale-free) -> screw guide
            metrics['com_rad'] = [round(metrics['com'][0] * rad_per_px, 9),
                                  round(metrics['com'][1] * rad_per_px, 9)]
        t = time.perf_counter_ns()
        even = (total % 2 == 0)                        # blank saturated cores on alternate frames -> flashing
        writer.write_frame(_ema_frame_u16(star_ema, scale, sat if even else None))
        spine.append({'index': i, 't_mono_ns': t, 'present': present, **metrics})
        total += 1

    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                break
            for rec in det.poll():                    # refresh the target location
                latest_blobs = rec.get('blobs', [])

            avail = _committed(reader, cur)
            if args.follow:
                if avail - 1 >= next_index:            # live: skip to the most recent frame
                    next_index = avail - 1
                    process(next_index)
                    next_index += 1
            else:
                while next_index < avail:              # offline: every frame in order
                    process(next_index)
                    next_index += 1
                    avail = _committed(reader, cur)

            newer = [s for s in _segments(args.session, args.role) if s > cur and _segment_ready(s)]
            if newer:
                close_segment()
                cur = newer[-1] if args.follow else newer[0]
                reader, writer, spine, det, ema = open_segment(cur)
                latest_blobs = []
                next_index = 0
                scale = None
                continue

            if _frame_count(cur) != ser_mod.SENTINEL_FRAME_COUNT and next_index >= _committed(reader, cur):
                if not args.follow:
                    break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        close_segment()
        print(f"[focus:{args.role}] processed {total} frames", flush=True)


if __name__ == '__main__':
    main()
