"""
astrolock_seeker_focus: star stacking + stack-quality metrics -- a pure image->(ser+json) filter.

Follows one camera's stream (live or finalized, exactly like detect), and for each frame:
  - finds the star: argmax of a small separable-Gaussian matched filter (just big enough that
    a dead pixel can't out-shine a real star), over a centered ROI (--roi) or the whole work
    image. NOT a raw-flux CoM -- over a full frame the sky's total flux and its vignette
    gradient swamp a flux-weighted mean (2026-08-02 offline run), while an argmax doesn't
    care how much sky surrounds the star.
  - crops (2^k - 1) around that INTEGER peak -- perfectly centered with no resampling. That
    one crop is both the display frame and the stacking input.
  - stacks it: hot-start EMA (display; resettable via the control file), or a pure running
    mean (sweep buckets).

Quality is ONLY ever evaluated on the stacked image, never on individual frames (2026-08-02
field data: every reasonable metric reads the same story off the stack; per-frame values are
seeing-speckle draws). The quality procedure: subtract the edge-pixel mean, normalize by the
remaining sum, read the peak; Strehl = that / the ideal optics PSF's normalized peak. HFD of
the stack rides along as the clip-immune companion metric.

Control: 'control_focus_<role>.jsonl' in the session dir; {'reset': 1, 'average': 0|1,
'seq': n} restarts the stack in the given mode. The applied seq echoes in the 'ctl_seq'
extra so the sweep can gate 'this stack is MY bucket'.

It writes the star stream (both halves discovered by SerFollower, same naming as _debug):
  - <seg>_<role>_focus.ser          : [stack | this frame] side by side, mono 16-bit at
                                      ABSOLUTE brightness, TRUE values (clip marking is the
                                      GUI shader's rainbow, per view)
  - <seg>_<role>_focus.frames.jsonl : commit spine; per-frame extras carry the STACK metrics
                                      (stack_peak/stack_strehl/stack_hfd/stack_n/ctl_seq/
                                      clip_px) + shape fits (astigmatism ellipse, coma skew)

No mount, no sky model, no sockets -- files in, files out, so it runs identically on a live
capture or a recording (where you'd tune it offline).

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
    """The star crop actually used: the largest (2^k - 1) size that fits both the request and
    the (maybe Bayer-halved) analysis image. Power-of-two-minus-one = an exact center pixel at
    integer coordinates, so the CoM-rounded crop is perfectly centered with no resampling."""
    d = 2 if bayer.is_bayer(header.color_id) else 1
    c = min(crop, header.image_height // d, header.image_width // d)
    p = 1
    while p * 2 + 1 <= c:
        p = p * 2 + 1
    return p


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


def _toroidal_com(window):
    """Sub-pixel star position within ``window``: per-axis CIRCULAR mean of the raw pixel
    values, treating the window as a torus (the circmean detector's trick). A uniform
    background contributes zero net circular mean -- background immunity with no background
    estimate and no thresholding -- and a star near the window edge wraps instead of
    truncating into a bias. Returns (x, y) in window px."""
    height, width = window.shape
    device = window.device

    def axis_pos(marginal, n):
        theta = torch.arange(n, dtype=torch.float32, device=device) * (2.0 * math.pi / n)
        c = float((marginal * torch.cos(theta)).sum())
        s = float((marginal * torch.sin(theta)).sum())
        return (math.atan2(s, c) % (2.0 * math.pi)) * n / (2.0 * math.pi)

    return axis_pos(window.sum(dim=0), width), axis_pos(window.sum(dim=1), height)


def _weighted_shape(img, sigma_seed, passes=2):
    """
    Closed-form Gaussian-weighted shape fit of the star in ``img`` (the moment-matching /
    'adaptive moments' estimator -- the one-shot solution GD would converge to for this
    model family). Per pass, about the current centroid with window sigma_w:

      background = median of the ring the window ignores (r > 3 sigma_w) -- "measure sky
                   where the model says there's no star"; residuals KEEP their sign (no
                   clamp, no k-sigma threshold: far pixels are weightless, not zeroed);
      centroid   = weighted first moment (updated each pass);
      ellipse    = central second moments -> (e1, e2) = ((Mxx-Myy)/T, 2Mxy/T), T = Mxx+Myy;
      skew       = r^2-weighted displacement, s = sum(w j d |d|^2) / sum(w j |d|^2), px --
                   "where the halo sits relative to the centroid"; zero iff symmetric,
                   under ANY symmetric window (the collimation null is window-independent);
      sigma_w    = adapted to the measured size, clamped to [1 px, min(H, W)/4].

    Weighted sizes are NOT corrected for the window (only ratios and nulls are consumed).
    Starts from the image centre; ``sigma_seed`` seeds the window width.
    Returns dict with cx, cy (px), e1, e2, sx, sy, sigma_w.
    """
    height, width = img.shape
    device = img.device
    ys = torch.arange(height, dtype=torch.float32, device=device)[:, None]
    xs = torch.arange(width, dtype=torch.float32, device=device)[None, :]
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    sigma_w = min(max(1.0, float(sigma_seed)), min(height, width) / 4.0)
    e1 = e2 = sx = sy = 0.0
    for _ in range(max(1, passes)):
        r2 = (xs - cx) ** 2 + (ys - cy) ** 2
        w = torch.exp(-r2 / (2.0 * sigma_w * sigma_w))
        ring = img[r2 > (3.0 * sigma_w) ** 2]
        background = float(ring.median()) if ring.numel() else float(img.median())
        j = (img - background) * w
        total = float(j.sum())
        if total <= 0:
            break                                # no net weighted flux: keep the last shape
        cx = float((j * xs).sum()) / total
        cy = float((j * ys).sum()) / total
        dx = xs - cx
        dy = ys - cy
        mxx = float((j * dx * dx).sum()) / total
        myy = float((j * dy * dy).sum()) / total
        mxy = float((j * dx * dy).sum()) / total
        t = mxx + myy
        if t <= 0:
            break
        e1, e2 = (mxx - myy) / t, 2.0 * mxy / t
        d2 = dx * dx + dy * dy
        halo = float((j * d2).sum())
        if halo > 0:
            sx = float((j * dx * d2).sum()) / halo
            sy = float((j * dy * d2).sum()) / halo
        # Adapt the window to the star's TRUE size by deconvolving the window from the
        # measured (attenuated) size: 1/true^2 = 1/measured^2 - 1/window^2. Feeding the
        # attenuated size straight back shrinks the window every pass (fixed point zero).
        cap = min(height, width) / 4.0
        inverse = 1.0 / max(t / 2.0, 1e-9) - 1.0 / (sigma_w * sigma_w)
        true_sq = (1.0 / inverse) if inverse > 1e-9 else cap * cap
        sigma_w = min(max(math.sqrt(true_sq), 1.0), cap)
    return {'cx': cx, 'cy': cy, 'e1': e1, 'e2': e2, 'sx': sx, 'sy': sy, 'sigma_w': sigma_w}


def _find_blur(img, sigma):
    """Gaussian low-pass for FINDING the star (separable, cheap): a single hot pixel averages
    down by ~the kernel norm while a real star keeps most of its amplitude, so a dim star
    always beats a hot pixel at the argmax. Works full-frame -- unlike a raw-flux CoM, the
    argmax doesn't care how much sky surrounds the star. Stacking and every metric use the
    RAW pixels; only localization looks through this."""
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
    """The running star stack + the metrics read off it. Finding is the argmax of a small
    separable-Gaussian matched filter (hot-pixel-immune, full-frame-capable) -- an INTEGER
    pixel, so the (2^k - 1) crop is perfectly centered on it with no resampling. Two stack
    modes: hot-start EMA (display; reset() on star loss / a confirmed new focus), or a pure
    running mean for sweep buckets -- mean += (frame - mean)/n, NOT a tiny-alpha lerp (which
    numerically stalls in float32 once the mean is established). Quality is ONLY ever read
    off the stack; individual frames are seeing-speckle draws (2026-08-02 field data)."""

    def __init__(self, crop, alpha, scale, roi=0, find_sigma=2.0):
        self.crop, self.alpha, self.scale, self.roi = crop, alpha, scale, roi
        self.find_sigma = find_sigma  # finder blur sigma; also seeds the shape fits (optics-set)
        self.stack = None
        self.average = False        # True: pure running mean (sweep bucket); False: EMA
        self.n = 0                  # frames since the last reset (the mean's divisor)

    def reset(self, average):
        """Restart the stack: star lost, a confirmed new focus position, or a sweep bucket."""
        self.average = bool(average)
        self.stack = None
        self.n = 0

    def update(self, work):
        # Find: matched-filtered argmax over the (optionally bounded) region. The blur is just
        # big enough that a dead/hot pixel can't out-shine a real star; the argmax is already
        # integer, so the power-of-two-minus-one crop centers on it exactly, no resampling.
        H, W = work.shape
        if self.roi and self.roi < min(H, W):
            region, x0, y0 = _crop(work, W / 2.0, H / 2.0, self.roi)
        else:
            region, x0, y0 = work, 0, 0
        pk = int(torch.argmax(_find_blur(region, self.find_sigma)))
        px, py = x0 + pk % region.shape[1], y0 + pk // region.shape[1]
        star, _, _ = _crop(work, px, py, self.crop)
        self.n += 1
        if self.stack is None or self.stack.shape != star.shape:
            self.stack = star.clone()
            self.n = 1
        elif self.average:
            self.stack += (star - self.stack) / self.n                 # exact running mean
        else:
            self.stack = torch.lerp(self.stack, star, self.alpha)      # hot-start EMA
        stack = self.stack
        # Stack quality: edge-mean pedestal off, sum normalizes, peak on top.
        edge = torch.cat([stack[0], stack[-1], stack[:, 0], stack[:, -1]])
        net = stack - edge.mean()
        total = float(net.sum())
        norm_peak = float(net.max()) / total if total > 0 else 0.0
        shape = _weighted_shape(stack, self.find_sigma)                # stack shape
        shape_instant = _weighted_shape(star, self.find_sigma)         # same fit, this frame
        metrics = {
            'stack_peak': round(float(stack.max()) / self.scale, 6),  # raw stack top (exposure aid)
            'norm_peak': norm_peak,                                   # Strehl numerator (of the stack)
            'stack_hfd': round(_hfd(stack), 3),                       # clip-immune focus metric
            'stack_n': self.n,
            'clip_px': int((star >= 0.98 * self.scale).sum()),        # THIS frame's clipped pixels
            'ellipse': [round(shape['e1'], 4), round(shape['e2'], 4)],           # astigmatism
            'skew': [round(shape['sx'], 3), round(shape['sy'], 3)],              # coma -> screws
            'instant_ellipse': [round(shape_instant['e1'], 4), round(shape_instant['e2'], 4)],
            'instant_skew': [round(shape_instant['sx'], 3), round(shape_instant['sy'], 3)],
        }
        return stack, star, metrics


def _ema_frame_u16(ema, scale):
    """A star crop as a uint16 mono image for the .ser, at ABSOLUTE brightness (NOT
    range-stretched) so the user can judge exposure by how bright it looks. TRUE values,
    nothing baked in -- clip marking is the GUI shader's job (the clip rainbow), so clipped
    pixels must arrive here still at full scale."""
    img = (ema / scale).clamp(0, 1)
    return (img * 65535.0).round().cpu().numpy()


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker focus / collimation filter")
    p.add_argument('--session', default=None, help="session directory to follow")
    p.add_argument('--role', default='main', help="camera role to focus on (tails its .ser + detections)")
    p.add_argument('--follow', action='store_true',
                   help="live mode: track the newest segment and never exit (default: offline, process "
                        "each segment in order then exit)")
    p.add_argument('--crop', type=int, default=127,
                   help="star crop size (px; snapped down to 2^k - 1 so the center is exact)")
    p.add_argument('--roi', type=int, default=0,
                   help="star-FINDING region: a centered square this many work px across "
                        "(0 = the whole work image). The toroidal CoM of this region is the star; "
                        "set the camera ROI (or this) so the star dominates it")
    p.add_argument('--alpha', type=float, default=0.05, help="EMA rate for the star stack (bigger = faster)")
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

    # Follow the cam stream; the star OUTPUT ring reconfigures only when the input ring does
    # (a geometry change -- the only roll left anywhere).
    fo = framestream.StreamFollower(args.session, args.role)
    # Stack control: {'reset': 1, 'average': 0|1, 'seq': n} restarts the stack in the given
    # mode. Written by the backend (GUI reset button; sweep bucket boundaries). The applied
    # seq echoes in the 'ctl_seq' extra so the sweep can gate on 'this stack is MY bucket'.
    ctl = JsonlTailer(os.path.join(args.session, f'control_focus_{args.role}.jsonl'))
    ctl_seq = 0
    ctl_average = False
    # Star stream: per-frame STACK metrics ride as binary record extras (strehl None -> NaN).
    out = framestream.FrameStream(args.session, f'{args.role}_focus',
                                  extras=('<16f', ['stack_peak', 'stack_strehl', 'stack_hfd',
                                                   'stack_n', 'ctl_seq', 'clip_px',
                                                   'ellipse_1', 'ellipse_2', 'skew_x', 'skew_y',
                                                   'instant_ellipse_1', 'instant_ellipse_2',
                                                   'instant_skew_x', 'instant_skew_y',
                                                   'skew_rad_x', 'skew_rad_y']))
    cur = None                                              # current input RingReader
    ema = None

    def switch_to(rd):
        nonlocal cur, ema, scale, strehl_done
        cur = rd
        crop = _effective_crop(rd.header, args.crop)        # fits the analysis image; writer + EMA agree
        out.configure(crop * 2, crop,                       # [stack | this frame's star]
                      color_id=ser_mod.ColorId.MONO, pixel_depth=16,
                      shm=args.shm_ser, frames=args.shm_frames,
                      # src_seg + each record's ABSOLUTE src_index fully names the source frame.
                      meta={'src_seg': rd.ident})
        # The stack survives everything except an actual geometry change (and explicit resets).
        if ema is None or ema.crop != crop:
            fs = ema.find_sigma if ema is not None else 2.0
            ema = FocusEma(crop, args.alpha, scale=None, roi=args.roi, find_sigma=fs)
            ema.average = ctl_average
            strehl_done = False                             # the ideal PSF is crop-sized: rebuild
        scale = None

    last_done = None               # (cam ring ident, index) of the last frame processed
    scale = None
    total = 0
    strehl_ref = None            # ideal (diffraction-limited) normalized peak; computed once, kept across segments
    strehl_done = False          # False until we've decided whether Strehl is available (needs the plate scale)
    rad_per_px = None            # radians per work px (for the pixel-scale-free CoM offset); None if unknown

    def close_all():
        fo.close()
        out.close()

    def process(i):
        nonlocal scale, total, strehl_ref, strehl_done, rad_per_px
        # Record + pixels read together, UP FRONT: processing can be slow (GPU warmup), and the
        # slot may be lapped before we finish -- a late record re-read would raise mid-write.
        rec_in = cur.record(i)
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
                ema.find_sigma = max(1.0, 0.35 * r_null)    # match the finder to the Airy core
                ideal = skysim.aperture_psf(ema.crop, r_null, obstruction=args.obstruction,
                                            vanes=args.vanes, vane_width_frac=args.vane_width, device=device)
            else:
                ideal = torch.zeros((ema.crop, ema.crop), device=device)   # perfect point source
                ideal[ema.crop // 2, ema.crop // 2] = 1.0
            strehl_ref = _normalized_peak(ideal)
        stack, star_now, metrics = ema.update(work)
        # Extras schema ('<16f'): strehl NaN when the aperture is unknown; skew_rad NaN when the
        # plate scale is (consumers turn NaN back into None/absent). Strehl = the STACK's
        # normalized peak over the ideal's -- quality of the average, not average of qualities.
        strehl = (metrics['norm_peak'] / strehl_ref) if strehl_ref else float('nan')
        skew_rad_x = metrics['skew'][0] * rad_per_px if rad_per_px is not None else float('nan')
        skew_rad_y = metrics['skew'][1] * rad_per_px if rad_per_px is not None else float('nan')
        # Side-by-side: the stack (left) next to the RAW star crop this instant (right) -- the
        # right half is the direct check that the finder is on the star at all.
        import numpy as _np
        pair = _np.concatenate([_ema_frame_u16(stack, scale),
                                _ema_frame_u16(star_now, scale)], axis=1)
        # Stamp with the SOURCE frame's capture time (not our processing time): the metrics
        # describe the light at capture, and consumers (graph x-axis, the sweep's settle gate)
        # should be clocked off that, immune to our own lag. The cam ALWAYS stamps; a zero here
        # is a broken producer, and papering over it with "now" would hide that.
        src_t = rec_in['t_mono_ns']
        if not src_t:
            raise ValueError(f"frame {i} of {cur.ident} has no capture stamp")
        out.write(pair,
                  t_mono_ns=src_t, src_index=i,
                  extras=(metrics['stack_peak'], strehl, metrics['stack_hfd'],
                          float(metrics['stack_n']), float(ctl_seq), float(metrics['clip_px']),
                          metrics['ellipse'][0], metrics['ellipse'][1],
                          metrics['skew'][0], metrics['skew'][1],
                          metrics['instant_ellipse'][0], metrics['instant_ellipse'][1],
                          metrics['instant_skew'][0], metrics['instant_skew'][1],
                          skew_rad_x, skew_rad_y))
        total += 1

    parent_dead = session_mod.parent_lifeline()   # backend gone (however it died) -> stop
    try:
        while True:
            if parent_dead.is_set() or (args.stop_file and os.path.exists(args.stop_file)):
                break
            for cmd in ctl.poll():                    # stack control (backend-written)
                if cmd.get('reset'):
                    ctl_average = bool(cmd.get('average'))
                    ctl_seq = int(cmd.get('seq', ctl_seq + 1))
                    if ema is not None:
                        ema.reset(ctl_average)
                    print(f"[focus:{args.role}] stack reset "
                          f"({'average' if ctl_average else 'ema'}, seq {ctl_seq})", flush=True)
            fo.poll()
            worked = False
            # LATEST-ONLY (like the detectors): always process the newest committed frame and
            # skip whatever piled up behind it -- a focus readout is only useful live, so
            # latency beats completeness. (Skipping stretches the stack's effective time
            # constant when we're slow; that's the right trade for a focusing aid.)
            got = fo.latest()
            if got is not None:
                rd, i = got
                if rd is not cur:
                    switch_to(rd)
                if last_done != (rd.ident, i):
                    last_done = (rd.ident, i)
                    try:
                        process(i)
                        worked = True
                    except framestream.Lapped:
                        pass
            if not worked:
                if fo.ended():
                    break                              # cam stream ended cleanly; we're done
                time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        close_all()
        print(f"[focus:{args.role}] processed {total} frames", flush=True)


def analyze_ser(args, device):
    """Offline analysis: the live pipeline's exact find/stack/Strehl math over a recorded .ser.
    Emits per-frame STACK metrics; the summary reports the best-focus frame and the clipped
    fraction (the thing that quietly ruins Strehl on real nights)."""
    import json as _json
    r = ser_mod.SerReader(args.ser)
    cid = r.header.color_id
    n = r.frames_on_disk()
    crop = _effective_crop(r.header, args.crop)
    scale = full_scale(cid, r.header.pixel_depth_per_plane)
    ema = FocusEma(crop, args.alpha, scale=scale, roi=args.roi)
    strehl_ref = None
    rows = []
    outf = open(args.metrics_out, 'w', encoding='utf-8') if args.metrics_out else None
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
                ema.find_sigma = max(1.0, 0.35 * r_null)
                ideal = skysim.aperture_psf(ema.crop, r_null, obstruction=args.obstruction,
                                            vanes=args.vanes, vane_width_frac=args.vane_width,
                                            device=device)
            else:
                ideal = torch.zeros((ema.crop, ema.crop), device=device)
                ideal[ema.crop // 2, ema.crop // 2] = 1.0
            strehl_ref = _normalized_peak(ideal)
            print(f"[focus] {n} frames, crop {crop}, stride {args.stride}", flush=True)
        _stack, _star_now, metrics = ema.update(work)
        row = {'i': i,
               'stack_peak': metrics['stack_peak'],
               'stack_strehl': round(metrics['norm_peak'] / strehl_ref, 5) if strehl_ref else None,
               'stack_hfd': metrics['stack_hfd'],
               'clip_px': metrics['clip_px']}
        rows.append(row)
        if outf:
            outf.write(_json.dumps(row) + chr(10))
        if k % 200 == 0:
            print(f"  frame {i}/{n}  strehl {row['stack_strehl']} hfd {row['stack_hfd']:.1f}",
                  flush=True)
    if outf:
        outf.close()
    r.close()
    best = max(rows, key=lambda x: x['stack_strehl'] or 0.0)
    clipped = sum(1 for x in rows if x['clip_px'])
    print(f"[focus] best stack Strehl {best['stack_strehl']} at frame {best['i']} "
          f"(hfd there {best['stack_hfd']:.1f}px); frames with clipped px: "
          f"{clipped}/{len(rows)}", flush=True)


if __name__ == '__main__':
    main()
