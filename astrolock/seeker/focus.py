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
import math
import os
import time

import torch

from astrolock.seeker import bayer, ser as ser_mod, skysim
from astrolock.seeker.detect import (resolve_device, work_image, full_scale,
                                     )
from astrolock.seeker import framestream
from astrolock.seeker import session as session_mod
from astrolock.seeker.sidecar import JsonlTailer


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


def _find_blur(img, sigma):
    """Gaussian low-pass for FINDING the star (matched filter for the best-focus PSF): a single
    hot pixel averages down by ~the kernel norm while a real star keeps most of its amplitude,
    so a dim star always beats a hot pixel at the argmax. Stacking and the peak metric use the
    RAW pixels -- only localization looks through this."""
    from astrolock.seeker.detect import gaussian_deriv_kernels, _conv1d_axis
    if sigma <= 0:
        return img
    r = min(max(1, int(3.0 * sigma + 0.5)), (min(img.shape) - 1) // 2)
    if r < 1:
        return img
    g, _, _ = gaussian_deriv_kernels(sigma, r, device=img.device)
    return _conv1d_axis(_conv1d_axis(img, g, 1), g, 0)


def _hfd(crop):
    """Half-flux diameter (px) of a star crop: the flux-weighted mean radius x2 about the CoM,
    after subtracting the border-pixel sky pedestal. THE autofocus metric: it measures spread,
    so it keeps working when the core saturates (where 'peak' pins at 1.0 and goes blind), and
    a focuser sweep of it is a clean V-curve whose minimum is best focus."""
    edge = torch.cat([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    # Subtract sky at edge mean + 2 sigma: with a plain mean, clamped noise residue spread over
    # the whole crop swamps the star and HFD reads ~crop-size regardless of focus.
    net = (crop - (edge.mean() + 2.0 * edge.std())).clamp(min=0)
    tot = float(net.sum())
    if tot <= 0:
        return 0.0
    H, W = crop.shape
    dev = crop.device
    ys = torch.arange(H, dtype=torch.float32, device=dev)
    xs = torch.arange(W, dtype=torch.float32, device=dev)
    cy = float((net.sum(dim=1) * ys).sum()) / tot
    cx = float((net.sum(dim=0) * xs).sum()) / tot
    r = torch.sqrt((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2)
    return 2.0 * float((net * r).sum()) / tot


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

    def __init__(self, crop, search, alpha, scale, peak_decay=0.9, find_sigma=2.0):
        self.crop, self.search, self.alpha, self.scale = crop, search, alpha, scale
        self.peak_decay = peak_decay
        self.find_sigma = find_sigma   # PSF-matched blur for LOCALIZATION only (hot-pixel immunity)
        self.ema = None
        self.peak_meter = None      # per-pixel decaying peak-hold of the RAW star crop, for saturation

    def update(self, work, target):
        # 1) search ROI around the target; 2) the MATCHED-FILTERED region's brightest pixel (a
        # hot pixel can't win); 3) star crop centred there, from the RAW pixels.
        tx, ty = target
        region, rx0, ry0 = _crop(work, tx, ty, self.search)
        pk = int(torch.argmax(_find_blur(region, self.find_sigma)))
        py, px = pk // region.shape[1], pk % region.shape[1]
        star, _, _ = _crop(work, rx0 + px, ry0 + py, self.crop)
        # NEVER clear a live EMA (user 2026-07-13): a finder hiccup blending through the average
        # beats restarting it -- the crop is target-locked, so a wrong frame fades in ~1/alpha.
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
            'peak_frame': round(float(star.max()) / self.scale, 6),   # per-frame; blinds when saturated
            'hfd': round(_hfd(star), 3),                              # per-frame; saturation-immune
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
    p.add_argument('--session', default=None, help="session directory to follow")
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
    p.add_argument('--shm-ser', action='store_true',
                   help="write the star stream to shared-memory segments instead of disk (matches the cams)")
    p.add_argument('--find-blur-px', type=float, default=0.0,
                   help="star-FINDING low-pass sigma (px): the matched filter that stops a hot "
                        "pixel from stealing the lock at dim exposures. 0 = auto (from the "
                        "optics' diffraction size when known, else 2.0). Stacking uses raw pixels")
    p.add_argument('--shm-frames', type=int, default=256,
                   help="shm star segments: frames per segment (the star crop is tiny)")
    p.add_argument('--device', default='auto',
                   help="torch device: 'auto' (default) = cuda if present else cpu, or force 'cpu'/'cuda'")
    p.add_argument('--ser', default=None,
                   help="OFFLINE: analyze a recorded .ser directly (no session/backend) -- run the "
                        "same find/EMA/Strehl pipeline over its frames and emit per-frame metrics. "
                        "For tuning against saved focus sweeps, and later the autofocus fit")
    p.add_argument('--stride', type=int, default=1, help="offline: analyze every Nth frame")
    p.add_argument('--limit', type=int, default=0, help="offline: stop after N analyzed frames (0 = all)")
    p.add_argument('--metrics-out', default=None, help="offline: write per-frame metrics JSONL here")
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    print(f"[focus:{args.role}] compute device: {device}", flush=True)
    if args.ser:
        return analyze_ser(args, device)

    # Follow the cam stream chain; the star OUTPUT is one stream for the whole run, rolling
    # its own segments (fresh on every input-segment switch, and when a shm segment fills).
    fo = framestream.StreamFollower(args.session, args.role, keep_all=True)

    def find_sigma0():
        """Star-finding blur sigma: explicit --find-blur-px, else 2.0 until the optics-derived
        value takes over (set in process() once the plate scale is known)."""
        return args.find_blur_px if args.find_blur_px > 0 else 2.0
    det_fo = framestream.StreamFollower(args.session, f'{args.role}_det')   # latest target blobs
    # Star stream: per-frame focus metrics ride as binary record extras (strehl None -> NaN;
    # 'present' is record flags bit 0). The GUI reads them straight from the section.
    out = framestream.FrameStream(args.session, f'{args.role}_focus',
                                  extras=('<8f', ['peak', 'peak_frame', 'hfd', 'strehl',
                                                  'dx', 'dy', 'com_rad_x', 'com_rad_y']))
    cur = None                                              # current input SegmentReader
    ema = None

    def switch_to(seg):
        nonlocal cur, ema, scale, strehl_done
        cur = seg
        crop = _effective_crop(seg.header, args.crop)       # fits the analysis image; writer + EMA agree
        out.open_segment(session_mod.segment_stamp(), crop, crop,
                         color_id=ser_mod.ColorId.MONO, pixel_depth=16,
                         shm=args.shm_ser, cap=args.shm_frames)
        # The EMA SURVIVES segment rolls (they're just the stream's paging, ~every few seconds);
        # it only rebuilds when the geometry actually changes -- never cleared otherwise.
        if ema is None or ema.crop != crop:
            fs = ema.find_sigma if ema is not None else find_sigma0()
            ema = FocusEma(crop, args.search, args.alpha, scale=None, find_sigma=fs)
            strehl_done = False                             # the ideal PSF is crop-sized: rebuild
        scale = None

    latest_blobs = []
    latest_tracking = False        # blobs are only trusted when they answer a tracker lock
    next_index = 0
    scale = None
    total = 0
    strehl_ref = None            # ideal (diffraction-limited) normalized peak; computed once, kept across segments
    strehl_done = False          # False until we've decided whether Strehl is available (needs the plate scale)
    rad_per_px = None            # radians per work px (for the pixel-scale-free CoM offset); None if unknown

    def close_all():
        fo.close()
        det_fo.close()
        out.close()

    def process(i):
        nonlocal scale, total, strehl_ref, strehl_done, rad_per_px
        frame = cur.read(i)
        cid = cur.header.color_id
        if scale is None:
            scale = full_scale(cid, cur.header.pixel_depth_per_plane)
            ema.scale = scale
        work = work_image(frame, cid, device=device)
        coord_scale = cur.header.image_width / work.shape[1]        # frame px per (maybe half-res) work px
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
                if args.find_blur_px <= 0:             # auto: match the finder to the Airy core
                    ema.find_sigma = max(1.0, 0.35 * r_null)   # gaussian sigma ~ the Airy core width
                ideal = skysim.aperture_psf(ema.crop, r_null, obstruction=args.obstruction,
                                            vanes=args.vanes, vane_width_frac=args.vane_width, device=device)
            else:
                ideal = torch.zeros((ema.crop, ema.crop), device=device)   # perfect point source
                ideal[ema.crop // 2, ema.crop // 2] = 1.0
            strehl_ref = _normalized_peak(ideal)
        # Target in WORK px. A detection steers us ONLY when it answers a tracker lock (the
        # detector's ROI/tracking phase): acquisition blobs jiggle and the extended detector is
        # for resolved targets, not stars. Untracked = the frame's matched-filtered brightest
        # pixel, full frame -- the brightest star is a steadier choice than any detector guess.
        present = bool(latest_blobs) and latest_tracking
        if present:                                # strongest blob (detect emits in tile order)
            fx, fy = max(latest_blobs, key=lambda b: b.get('score', 0.0))['px']
            target = (fx / coord_scale, fy / coord_scale)
        else:
            pk = int(torch.argmax(_find_blur(work, ema.find_sigma)))   # hot pixels can't win
            target = (pk % work.shape[1], pk // work.shape[1])
        star_ema, metrics, sat = ema.update(work, target)
        # Extras schema ('<6f'): strehl NaN when the aperture is unknown; com_rad NaN when the
        # plate scale is (consumers turn NaN back into None/absent).
        strehl = (_normalized_peak(star_ema) / strehl_ref) if strehl_ref else float('nan')
        crx = metrics['com'][0] * rad_per_px if rad_per_px is not None else float('nan')
        cry = metrics['com'][1] * rad_per_px if rad_per_px is not None else float('nan')
        even = (total % 2 == 0)                        # blank saturated cores on alternate frames -> flashing
        out.write(_ema_frame_u16(star_ema, scale, sat if even else None),
                  t_mono_ns=time.perf_counter_ns(), src_index=i, flags=1 if present else 0,
                  extras=(metrics['peak'], metrics['peak_frame'], metrics['hfd'], strehl,
                          metrics['com'][0], metrics['com'][1], crx, cry))
        total += 1

    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                break
            det_fo.poll()                             # refresh the target location (latest record)
            got_det = det_fo.latest()
            if got_det is not None:
                dseg, di = got_det
                if getattr(dseg, '_used', -1) != di:
                    dseg._used = di
                    import json as _json
                    d = _json.loads(bytes(dseg.read(di)).decode('utf-8'))
                    latest_blobs = d.get('blobs', [])
                    latest_tracking = bool(d.get('tracking'))
            fo.poll()
            worked = False
            seg = fo.segs[0] if fo.segs else None      # SEQUENTIAL: exactly one EMA sample per
            if seg is not None:                        # frame, in order, stalls notwithstanding
                if seg is not cur:
                    switch_to(seg)
                    latest_blobs = []
                    latest_tracking = False
                    next_index = 0
                n = min(seg.committed(), next_index + 8)   # catch up in bounded bites (stay live)
                while next_index < n:
                    process(next_index)
                    next_index += 1
                    worked = True
                if (seg.finalized() and next_index >= seg.committed()
                        and (len(fo.segs) > 1 or seg.stream_ended())):
                    done = seg.stream_ended()
                    fo.release(seg)
                    cur = None
                    if done:
                        break
                    continue
            if not worked:
                if fo.ended() and seg is None:
                    break                              # cam stream ended cleanly; we're done
                time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        close_all()
        print(f"[focus:{args.role}] processed {total} frames", flush=True)


def analyze_ser(args, device):
    """Offline sweep analysis: the live pipeline's exact find/EMA/Strehl math over a recorded
    .ser. Emits per-frame metrics (peak_frame is what an autofocus sweep fits); the summary
    reports the best-focus frame and how stable the star lock was (hot-pixel jumps show up as
    target teleports)."""
    import json as _json
    r = ser_mod.SerReader(args.ser)
    cid = r.header.color_id
    n = r.frames_on_disk()
    crop = _effective_crop(r.header, args.crop)
    scale = full_scale(cid, r.header.pixel_depth_per_plane)
    ema = FocusEma(crop, args.search, args.alpha, scale=scale,
                   find_sigma=(args.find_blur_px if args.find_blur_px > 0 else 2.0))
    strehl_ref = None
    rows = []
    outf = open(args.metrics_out, 'w', encoding='utf-8') if args.metrics_out else None
    last_t = None
    idxs = range(0, n, max(1, args.stride))
    if args.limit:
        idxs = list(idxs)[:args.limit]
    for k, i in enumerate(idxs):
        work = work_image(r.read_frame(i), cid, device=device)
        if strehl_ref is None:
            coord_scale = r.header.image_width / work.shape[1]
            if args.aperture_mm > 0 and args.focal_mm > 0 and args.pixel_um > 0:
                r_null = skysim.airy_r_null_px(args.focal_mm, args.aperture_mm,
                                               args.wavelength_nm, args.pixel_um * coord_scale)
                if args.find_blur_px <= 0:
                    ema.find_sigma = max(1.0, 0.35 * r_null)
                ideal = skysim.aperture_psf(ema.crop, r_null, obstruction=args.obstruction,
                                            vanes=args.vanes, vane_width_frac=args.vane_width,
                                            device=device)
            else:
                ideal = torch.zeros((ema.crop, ema.crop), device=device)
                ideal[ema.crop // 2, ema.crop // 2] = 1.0
            strehl_ref = _normalized_peak(ideal)
            print(f"[focus] {n} frames, crop {crop}, find_sigma {ema.find_sigma:.2f}px, "
                  f"stride {args.stride}", flush=True)
        pk = int(torch.argmax(_find_blur(work, ema.find_sigma)))
        target = (pk % work.shape[1], pk // work.shape[1])
        star_ema, metrics, sat = ema.update(work, target)
        row = {'i': i, 'tx': target[0], 'ty': target[1],
               'peak': metrics['peak'], 'peak_frame': metrics['peak_frame'],
               'hfd': metrics['hfd'],
               'strehl': round(_normalized_peak(star_ema) / strehl_ref, 4) if strehl_ref else None,
               'sat_px': int(sat.sum())}
        rows.append(row)
        if outf:
            outf.write(_json.dumps(row) + chr(10))
        if k % 200 == 0:
            print(f"  frame {i}/{n}  peak_frame {row['peak_frame']:.4f}", flush=True)
    if outf:
        outf.close()
    r.close()
    # Summary: best focus + lock stability (a teleporting target = the finder changed its mind).
    import math as _math
    best = max(rows, key=lambda x: x['peak_frame'])
    besth = min((x for x in rows if x['hfd'] > 0), key=lambda x: x['hfd'], default=None)
    if besth:
        print(f"[focus] min HFD {besth['hfd']:.2f}px at frame {besth['i']} "
              f"(saturation-immune best focus)", flush=True)
    jumps = sum(1 for a, b in zip(rows, rows[1:])
                if _math.hypot(b['tx'] - a['tx'], b['ty'] - a['ty']) > args.search / 2)
    satn = sum(1 for x in rows if x['sat_px'])
    print(f"[focus] best peak_frame {best['peak_frame']:.4f} at frame {best['i']} "
          f"(strehl there {best['strehl']}); target jumps >{args.search // 2}px: {jumps}; "
          f"frames with saturated px: {satn}/{len(rows)}", flush=True)


if __name__ == '__main__':
    main()
