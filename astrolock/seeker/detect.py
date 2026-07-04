"""
astrolock_seeker_detect: a pure image->json filter.

Follows one camera's .ser (live or finalized), finds bright/moving blobs per frame, and
writes a <ts>_<role>.detections.jsonl sidecar. No mount, no sky model, no sockets -- just
files in, files out, so it runs identically on a live capture or last week's recording
(the primary place we develop/tune detection offline).

It is deliberately dumb about intent: it reports every bright spot it sees and flags which
ones changed since the previous frame ("moving"). Temporal association, gating to the
expected region, and target lock are the backend's job.

Math is torch (consistent with the rest of seeker, and differentiable if we want it). The
image-op functions accept numpy or torch and return torch.

    python -m astrolock.seeker.detect --session sessions/<ts> --role guide
"""

import argparse
import glob
import math
import os
import time

import torch

from astrolock.seeker import bayer, ser as ser_mod, sidecar
from astrolock.seeker.sidecar import JsonlWriter

_DEVICE = torch.device('cpu')        # switch to 'cuda' once a CUDA torch is installed

# The hot per-frame surfaces are torch.compile'd into single fused kernels where a C++ compiler is
# present (a real win on the live detector -- DoH ~85->66ms, surprise ~17->8ms at 1080p); suppress_errors
# falls back to eager otherwise, so it's seamless -- no flag. (TORCHDYNAMO_DISABLE=1 forces eager.)
# Compiled lazily + cached, and only ever applied to fixed-shape *full-frame* inputs -- never the
# variable-size track ROI -- so each compiles exactly once.
_compiled = {}


def _compiled_fn(fn):
    c = _compiled.get(fn)
    if c is None:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
        c = _compiled[fn] = torch.compile(fn)
    return c


def work_image(frame, color_id, device=None):
    """The grayscale image we analyze: Bayer -> sensitive half-res mono sum; else as-is. All torch
    (device-parameterized). torch has no uint16, so cast the raw frame to int32 at this ingest."""
    mosaic = torch.from_numpy(frame.astype('int32')).to(device or _DEVICE)
    if bayer.is_bayer(color_id):
        return bayer.to_mono_sum(mosaic)
    return mosaic.float()


def _running_mean(a, r, axis):
    """Separable 1-D box mean of radius r with edge padding (torch)."""
    k = 2 * r + 1
    if axis == 0:
        ap = torch.cat([a[:1].expand(r, -1), a, a[-1:].expand(r, -1)], dim=0)
    else:
        ap = torch.cat([a[:, :1].expand(-1, r), a, a[:, -1:].expand(-1, r)], dim=1)
    cs = torch.cumsum(ap, dim=axis)
    cs = torch.cat([torch.zeros_like(cs.narrow(axis, 0, 1)), cs], dim=axis)
    n = a.shape[axis]
    return (cs.narrow(axis, k, n) - cs.narrow(axis, 0, n)) / k


def box_blur(img, r):
    """2-D box blur (separable), radius r."""
    img = torch.as_tensor(img, dtype=torch.float32)
    if r < 1:
        return img
    return _running_mean(_running_mean(img, r, 0), r, 1)


def band_pass(work, bg_radius):
    """
    Point-source band-pass: subtract a local background (large blur). Point sources survive
    as sharp peaks; large bright AREAS are ~flat in their interior so they cancel, and smooth
    sky gradients vanish. This is what makes detection pick pointlike things, not rooftops.

    Left unclipped (can be negative) so the background is a roughly zero-mean noise field we
    can estimate a robust sigma from for SNR thresholding.
    """
    work = torch.as_tensor(work, dtype=torch.float32)
    return work - box_blur(work, bg_radius)


def gaussian_deriv_kernels(sigma, radius):
    """1-D Gaussian and its 1st/2nd derivatives, sampled on [-radius, radius] (torch)."""
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    g1 = -(x / sigma ** 2) * g                           # d/dx of the (normalized) Gaussian
    g2 = ((x ** 2 - sigma ** 2) / sigma ** 4) * g         # d^2/dx^2
    return g, g1, g2


def _conv1d_axis(img, k, axis):
    """Convolve 2-D ``img`` with 1-D kernel ``k`` along ``axis`` (0=y, 1=x), reflect-padded."""
    pad = k.numel() // 2
    x = img[None, None]
    if axis == 1:
        x = torch.nn.functional.pad(x, (pad, pad, 0, 0), mode='reflect')
        return torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1))[0, 0]
    x = torch.nn.functional.pad(x, (0, 0, pad, pad), mode='reflect')
    return torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1))[0, 0]


def det_of_hessian(work, sigma):
    """
    Scale-normalized determinant of the Hessian at scale ``sigma`` (px), as the detection
    surface (an alternative to band_pass). Computed via separable Gaussian-derivative
    convolutions: Lxx = g'' * g, Lyy = g * g'', Lxy = g' * g', then DoH = Lxx*Lyy - Lxy^2.

    Peaks on round blobs ~sigma in size; an edge/line gives ~0 (one principal curvature
    vanishes, so the determinant collapses regardless of orientation -- the -Lxy^2 term cancels
    Lxx*Lyy for a diagonal edge) and a saddle goes negative. So it discriminates star/target
    blobs from the door-frame and wire edges that fool a plain band-pass.

    We return sqrt(max(DoH, 0)) -- the geometric mean of the two principal curvatures,
    sqrt(lambda1*lambda2). The raw determinant scales as contrast^2 (each Hessian term is linear
    in amplitude), which crushes faint stars toward the noise floor; the square root is *linear*
    in contrast, restoring faint-source sensitivity comparable to a matched filter, while still
    vanishing on edges (one curvature ~0 -> product ~0) so the edge rejection survives. It also
    broadens the otherwise razor-sharp peaks, which helps the min-blob-px size cut. Saddles
    (negative determinant) clamp to 0. The Gaussian's sigma is both the blob scale and the noise
    low-pass. See astrolock_seeker.md.
    """
    work = torch.as_tensor(work, dtype=torch.float32)
    h, w = work.shape
    radius = max(1, min(int(4.0 * sigma + 0.5), (min(h, w) - 1) // 2))
    g, g1, g2 = gaussian_deriv_kernels(sigma, radius)
    lxx = _conv1d_axis(_conv1d_axis(work, g2, 1), g, 0)
    lyy = _conv1d_axis(_conv1d_axis(work, g, 1), g2, 0)
    lxy = _conv1d_axis(_conv1d_axis(work, g1, 1), g1, 0)
    doh = (lxx * lyy - lxy * lxy) * (sigma ** 4)          # gamma-normalized, comparable across scales
    return torch.sqrt(torch.clamp(doh, min=0.0))          # linearize in contrast; keeps edge rejection


def detection_surface(work, *, detector, bg_radius, psf_px, doh_sigma, compiled=False):
    """The 2-D map detect_blobs picks peaks from: band-pass (default) or determinant-of-Hessian.
    (The stateful 'surprise' detector is produced by SurpriseModel, not here.) Pass compiled=True on the
    full-frame acquisition path to fuse the DoH surface via torch.compile -- NOT from the track ROI,
    whose window size varies (that would recompile every frame)."""
    if detector == 'doh':
        sigma = doh_sigma if doh_sigma > 0 else psf_px
        return (_compiled_fn(det_of_hessian) if compiled else det_of_hessian)(work, sigma)
    return band_pass(work, bg_radius)


class SurpriseModel:
    """Per-pixel temporal detector for faint fast movers (satellite trails) that a single-frame blob
    detector misses -- because a trail spreads its light along many pixels (so each is far dimmer than
    a star) and a determinant-of-Hessian surface actively suppresses lines. Two per-pixel EMAs plus a
    decaying peak-hold, all stateful (call update(work) once per frame, in order):

      surprise:  z = max(0, (x - mean) / sqrt(var + floor)), where mean/var are the pixel's own EMAs.
                 How surprising this frame's value is given the pixel's history. Static terrain never
                 deviates; a twinkling/drifting STAR has high variance so its wiggles aren't surprising;
                 a dark sky pixel suddenly lit by a mover spikes. The mean's time constant also
                 separates a fast mover (a 1-frame spike) from slow star drift (tracked + absorbed by
                 the mean). This alone removes stars + terrain -- the surprise map is nearly black.

      trail:     trail = max(decay * trail, z) -- a decaying peak-hold of the surprise. A mover paints
                 its recent path as a bright comet: each pixel keeps the FULL spike height as the object
                 passes, fading over ~1/(1-decay) frames, so a per-frame few-sigma spike becomes a
                 spatially extended, connected feature that detect_blobs can catch (same block-max /
                 density-cap path as the other detectors), while isolated noise just decays. `decay`
                 trades latency for sensitivity: ~0 = single-frame, ->1 = long integration for the
                 faintest trails.

    var_floor_frac ties the noise floor to the typical per-pixel temporal variance (its median across
    the frame), so no absolute-DN tuning is needed. Needs ~1/alpha frames of warm-up before the
    variance estimate settles. The per-pixel elementwise step (_surprise_step) is torch.compile'd into
    one fused kernel (see _compiled_fn). See astrolock_seeker.md."""

    def __init__(self, alpha_mean=0.15, alpha_var=0.08, decay=0.85, var_floor_frac=0.5):
        self.a_m, self.a_v, self.decay, self.vff = alpha_mean, alpha_var, decay, var_floor_frac
        self.mu = self.var = self.trail = None

    def update(self, work):
        work = torch.as_tensor(work, dtype=torch.float32)
        if self.mu is None:                                   # first frame: seed, emit nothing
            self.mu = work.clone()
            self.var = torch.full_like(work, float(work.var()) + 1.0)
            self.trail = torch.zeros_like(work)
            return self.trail
        floor = self.vff * torch.median(self.var) + 1e-6      # 0-d tensor (dynamic; no per-frame recompile)
        self.trail, self.mu, self.var = _compiled_fn(_surprise_step)(
            work, self.mu, self.var, self.trail, floor, self.a_m, self.a_v, self.decay)
        return self.trail


def _surprise_step(work, mu, var, trail, floor, a_m, a_v, decay):
    """One SurpriseModel step, elementwise -> one fused kernel: surprise z-score + decaying peak-hold
    trail + EMA mean/variance updates. Returns (trail, mu, var)."""
    d = work - mu
    z = (d / torch.sqrt(var + floor)).clamp(min=0.0)
    return torch.maximum(decay * trail, z), mu + a_m * d, (1.0 - a_v) * var + a_v * d * d


def detect_blobs(bp, work, prev_bp, *, threshold_rel, max_candidates, suppress_radius,
                 min_blob_px, max_size_px, psf_px, snr=0.0, min_roundness=0.0, moving_frac, scale,
                 tile_grid=0, per_tile=0):
    """
    Peak detection on the detection surface ``bp`` (band-pass or determinant-of-Hessian), fully
    vectorized in torch (device-agnostic; no Python per-pixel loop -- only the final
    <= max_candidates results cross back to Python as dicts). ``work`` is the original grayscale
    (absolute brightness ``score``) and ``prev_bp`` the previous surface (the "moving" flag).

    Candidates are the max of each ``2*suppress_radius+1`` tile (a strided max-pool, already
    ~r-spaced). For all candidates at once we compute a sub-pixel centre, size, ``pointlike``
    (1 = PSF-sized, →0 = extended) and ``roundness`` (1 = circular, →0 = line/edge, from the
    second-moment eigenvalues), then cut: peaks must clear ``snr`` sigma over the surface
    background (robust MAD) and/or a ``threshold_rel`` floor, and pass ``min_blob_px`` /
    ``max_size_px`` / ``min_roundness``.

    Density cap: with ``tile_grid`` > 0 the frame is split into ~``tile_grid`` tiles across and at
    most ``per_tile`` blobs are kept per tile (strongest first) -- so a dense bright region (a
    foliage blob-field) can't eat the whole ``max_candidates`` budget and starve real targets.
    """
    bp = torch.as_tensor(bp, dtype=torch.float32)
    work = torch.as_tensor(work, dtype=torch.float32)
    blobs = []
    m = float(bp.max())
    if m <= 0:
        return blobs
    h, w = bp.shape

    # Absolute SNR threshold from a robust background sigma (MAD), with optional relative floor.
    flat = bp.reshape(-1)
    med = torch.median(flat)
    sigma = 1.4826 * float(torch.median(torch.abs(flat - med))) + 1e-6
    thresh = max(snr * sigma, threshold_rel * m)

    diff = None
    if prev_bp is not None:
        prev_bp = torch.as_tensor(prev_bp, dtype=torch.float32)
        if prev_bp.shape == bp.shape:
            diff = bp - prev_bp

    # Candidates: one max per (2r+1) tile via a strided max-pool (return_indices gives locations);
    # already ~r-spaced, so no extra NMS. A blob straddling a tile boundary may give two
    # near-coincident candidates -- harmless (they centroid to ~the same point).
    dev = bp.device
    F = torch.nn.functional
    r = max(1, suppress_radius)
    t = 2 * r + 1
    vals, idx = F.max_pool2d(bp[None, None], kernel_size=t, stride=t, ceil_mode=True, return_indices=True)
    vals, idx = vals.reshape(-1), idx.reshape(-1)
    sel = vals >= thresh
    if not bool(sel.any()):
        return blobs
    vals, idx = vals[sel], idx[sel]
    order = torch.argsort(vals, descending=True)            # strongest first
    vals, idx = vals[order], idx[order]
    cy, cx = idx // w, idx % w                               # (K,) peak-pixel coords
    K = cy.numel()

    # Batched window around every candidate at once; out-of-bounds pixels masked to 0.
    off = torch.arange(-r, r + 1, device=dev)
    yraw = cy[:, None, None] + off[None, :, None]
    xraw = cx[:, None, None] + off[None, None, :]
    valid = (yraw >= 0) & (yraw < h) & (xraw >= 0) & (xraw < w)        # (K, t, t) by broadcast
    yy = yraw.clamp(0, h - 1).expand(K, t, t)
    xx = xraw.clamp(0, w - 1).expand(K, t, t)
    win = torch.where(valid, bp[yy, xx], torch.zeros((), device=dev))  # (K, t, t)
    peak = vals[:, None, None]

    n_above = (win >= 0.5 * peak).sum(dim=(1, 2))                      # (K,)
    wsub = (win - 0.5 * peak).clamp(min=0.0)
    tot = wsub.sum(dim=(1, 2)).clamp(min=1e-6)
    cxf = (xx * wsub).sum(dim=(1, 2)) / tot                            # sub-pixel centroid
    cyf = (yy * wsub).sum(dim=(1, 2)) / tot
    dxw, dyw = xx - cxf[:, None, None], yy - cyf[:, None, None]
    ixx = (wsub * dxw * dxw).sum(dim=(1, 2)) / tot                     # second moments -> roundness
    iyy = (wsub * dyw * dyw).sum(dim=(1, 2)) / tot
    ixy = (wsub * dxw * dyw).sum(dim=(1, 2)) / tot
    tr = ixx + iyy
    s = torch.sqrt(torch.clamp((tr / 2) ** 2 - (ixx * iyy - ixy * ixy), min=0.0))
    l1, l2 = tr / 2 + s, tr / 2 - s
    roundness = torch.where(l1 > 1e-6, l2 / l1.clamp(min=1e-6), torch.ones_like(l1))
    size_px = torch.sqrt(n_above.float() / math.pi) * 2.0
    pointlike = torch.clamp(psf_px / size_px.clamp(min=psf_px), max=1.0)

    keep = n_above >= min_blob_px                                      # cuts (vectorized)
    if max_size_px:
        keep &= size_px <= max_size_px
    if min_roundness:
        keep &= roundness >= min_roundness

    # Density cap: <= per_tile surviving blobs per coarse grid tile (value order preserved).
    if tile_grid > 0 and per_tile > 0:
        tpx = math.ceil(w / tile_grid)
        ncols = math.ceil(w / tpx)
        s_pos = torch.nonzero(keep, as_tuple=False).squeeze(1)        # survivors, value order
        if s_pos.numel() > 0:
            tid = (cy[s_pos] // tpx) * ncols + (cx[s_pos] // tpx)
            g = torch.argsort(tid, stable=True)
            tid_s = tid[g]
            M = tid.numel()
            ar = torch.arange(M, device=dev)
            newg = torch.ones(M, dtype=torch.bool, device=dev)
            if M > 1:
                newg[1:] = tid_s[1:] != tid_s[:-1]
            gstart = torch.cummax(torch.where(newg, ar, torch.zeros_like(ar)), dim=0).values
            rank = torch.empty(M, dtype=torch.long, device=dev)
            rank[g] = ar - gstart                                     # rank within tile (value order)
            keep[s_pos[rank >= per_tile]] = False

    # Global cap, then bring just the <= max_candidates survivors back to Python as dicts.
    final = torch.nonzero(keep, as_tuple=False).squeeze(1)[:max_candidates]
    if final.numel() == 0:
        return blobs
    px, py = cxf[final].tolist(), cyf[final].tolist()
    sz, pt, rd = size_px[final].tolist(), pointlike[final].tolist(), roundness[final].tolist()
    sc = (work[cy[final], cx[final]] / scale).tolist()
    mv = (diff[cy[final], cx[final]] > moving_frac * vals[final]).tolist() if diff is not None else None
    return [{
        'id': i,
        'px': [round(px[i], 2), round(py[i], 2)],            # [x, y] in the work image
        'score': round(sc[i], 4),                            # absolute brightness 0..1
        'size_px': round(sz[i], 1),
        'pointlike': round(pt[i], 3),
        'roundness': round(rd[i], 3),
        'moving': (bool(mv[i]) if mv is not None else None),
    } for i in range(final.numel())]


def _segments(session, role):
    return sorted(glob.glob(os.path.join(session, f'*_{role}.ser')))


def _committed(reader, ser_path):
    """Frames safe to read in a segment: min(committed sidecar lines, frames on disk)."""
    lines = sidecar.count_complete_lines(ser_path[:-len('.ser')] + '.frames.jsonl')
    return min(lines, reader.frames_on_disk())


def _segment_ready(ser_path):
    """A just-created segment's .ser exists a beat before the cam has flushed its header + first
    frame. Gate on the commit point -- the sidecar's first complete line, which the cam appends only
    after those bytes are on disk -- so we never open a header-less .ser."""
    return sidecar.count_complete_lines(ser_path[:-len('.ser')] + '.frames.jsonl') >= 1


def _frame_count(ser_path):
    with open(ser_path, 'rb') as f:
        return ser_mod.unpack_header(f.read(ser_mod.HEADER_SIZE)).frame_count


def detect_roi_peak(work, roi, coord_scale, *, detector, bg_radius, psf_px, doh_sigma, snr):
    """Track-mode detection: find the single locked target in a small ROI around the predicted
    position -- far cheaper than a full-frame multi-blob pass, and no merging/gating needed.

    ``roi`` is [cx, cy, size] in *frame* px (from the backend's predicted target); ``work`` is the
    (possibly half-res) analysis image with ``coord_scale`` frame px per work px. Returns one blob in
    frame coords -- the centroid of the strongest detection-surface peak, biased toward the predicted
    centre so a brighter nearby star can't steal the lock -- or [] if nothing target-like is in the
    window (target lost / drifted out), which the tracker then treats as a miss.
    """
    work = torch.as_tensor(work, dtype=torch.float32)
    H, W = work.shape
    cx, cy, size = roi
    half = (size / coord_scale) / 2.0
    ecx, ecy = cx / coord_scale, cy / coord_scale            # expected centre, work coords
    x0, x1 = max(0, int(ecx - half)), min(W, int(ecx + half) + 1)
    y0, y1 = max(0, int(ecy - half)), min(H, int(ecy + half) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []                                            # ROI off the frame
    sub = work[y0:y1, x0:x1]
    bp = detection_surface(sub, detector=detector, bg_radius=bg_radius, psf_px=psf_px, doh_sigma=doh_sigma)
    m = float(bp.max())
    if m <= 0:
        return []
    h, w = bp.shape
    dev = bp.device
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=dev),
                            torch.arange(w, dtype=torch.float32, device=dev), indexing='ij')
    sig_b = max(2.0 * psf_px, half / 2.0)                     # gentle pull toward the predicted centre
    weight = torch.exp(-(((xx - (ecx - x0)) ** 2 + (yy - (ecy - y0)) ** 2)) / (2.0 * sig_b ** 2))
    pidx = int(torch.argmax(bp * weight))                    # localize: strongest centre-biased peak
    py, px = pidx // w, pidx % w
    # Found-test on *raw brightness*: the target is a genuinely bright source, while a DoH peak on
    # noise sits at background level (DoH of noise is heavy-tailed, so an SNR cut on DoH itself isn't
    # reliable). Require the peak's work value to clear snr sigma over the ROI's robust background.
    wflat = sub.reshape(-1)
    wmed = torch.median(wflat)
    wsig = 1.4826 * float(torch.median(torch.abs(wflat - wmed))) + 1e-6
    if float(sub[py, px]) - float(wmed) < snr * wsig:
        return []                                            # nothing target-like near the prediction
    cr = max(1, int(round(psf_px)))                          # sub-pixel centroid in a +/-psf window
    wy0, wy1 = max(0, py - cr), min(h, py + cr + 1)
    wx0, wx1 = max(0, px - cr), min(w, px + cr + 1)
    patch = bp[wy0:wy1, wx0:wx1].clamp(min=0)
    s = float(patch.sum())
    if s > 0:
        pyy, pxx = torch.meshgrid(torch.arange(wy0, wy1, dtype=torch.float32, device=dev),
                                  torch.arange(wx0, wx1, dtype=torch.float32, device=dev), indexing='ij')
        cpx, cpy = float((patch * pxx).sum()) / s, float((patch * pyy).sum()) / s
    else:
        cpx, cpy = float(px), float(py)
    return [{'px': [(x0 + cpx) * coord_scale, (y0 + cpy) * coord_scale],
             'moving': True, 'size_px': psf_px * coord_scale, 'score': float(bp[py, px] / m)}]


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker blob detector")
    p.add_argument('--session', required=True, help="session directory to follow")
    p.add_argument('--role', default='guide')
    p.add_argument('--follow', action='store_true',
                   help="live mode: track the newest segment, roll across segments, never exit "
                        "on finalize (default: offline, process segments in order then exit)")
    p.add_argument('--snr', type=float, default=6.0,
                   help="detect peaks this many sigma above the band-passed background")
    p.add_argument('--threshold', type=float, default=0.0,
                   help="optional relative floor: fraction of the brightest band-passed pixel (0 = off)")
    p.add_argument('--detector', default='doh', choices=['bandpass', 'doh', 'surprise'],
                   help="detection surface: 'doh' (default) = determinant of the Hessian "
                        "(Gaussian-derivative blob detector; rejects edges/lines by construction), "
                        "'bandpass' (the older local-background subtraction), or 'surprise' = per-pixel "
                        "temporal surprise + decaying peak-hold (finds faint fast movers / satellite "
                        "trails that the single-frame surfaces miss; see SurpriseModel)")
    p.add_argument('--surprise-alpha-mean', type=float, default=0.15,
                   help="surprise: EMA rate for the per-pixel mean (bigger = adapts faster to drift)")
    p.add_argument('--surprise-alpha-var', type=float, default=0.08,
                   help="surprise: EMA rate for the per-pixel variance")
    p.add_argument('--surprise-decay', type=float, default=0.85,
                   help="surprise: trail peak-hold decay per frame (~0 = single-frame; ->1 integrates "
                        "longer for the faintest trails, at more latency)")
    p.add_argument('--surprise-var-floor-frac', type=float, default=0.5,
                   help="surprise: noise floor as this fraction of the median per-pixel variance")
    p.add_argument('--debug-ser', action='store_true',
                   help="also write the detection surface as <seg>_<role>_debug.ser (+ .frames.jsonl), a "
                        "normalized greyscale movie of exactly what the detector 'sees' -- follow it in "
                        "the GUI or any SER viewer to tune (esp. the surprise trail).")
    p.add_argument('--bg-radius', type=int, default=12,
                   help="bandpass: local-background blur radius (px); larger = pass bigger features")
    p.add_argument('--doh-sigma', type=float, default=0.0,
                   help="doh: Gaussian scale in px (0 = use --psf-px); the blob size it responds to")
    p.add_argument('--max-candidates', type=int, default=16)
    p.add_argument('--tile-grid', type=int, default=8,
                   help="density cap: split the frame into ~this many tiles across and keep at most "
                        "--per-tile blobs per tile, so a dense bright region can't eat the whole "
                        "budget (0 = off, report globally strongest only)")
    p.add_argument('--per-tile', type=int, default=2, help="density cap: max blobs kept per tile")
    p.add_argument('--suppress-radius', type=int, default=6, help="non-max-suppression radius (px)")
    p.add_argument('--min-blob-px', type=int, default=2, help="ignore peaks smaller than this")
    p.add_argument('--max-size-px', type=float, default=0.0,
                   help="reject blobs fatter than this (0 = keep all; rejects extended clutter)")
    p.add_argument('--psf-px', type=float, default=3.0, help="reference point-source size for the pointlike score")
    p.add_argument('--min-roundness', type=float, default=0.0,
                   help="reject blobs below this roundness 0..1 (0 = keep all; rejects edges/streaks)")
    p.add_argument('--moving-frac', type=float, default=0.5,
                   help="frame-diff at the peak must exceed this fraction of the peak to be 'moving'")
    p.add_argument('--poll', type=float, default=0.02, help="seconds between polls when caught up (live)")
    p.add_argument('--stop-file', default=None, help="stop when this file appears")
    p.add_argument('--device', default='cpu', help="torch device for detection (cpu / cuda)")
    args = p.parse_args(argv)
    device = torch.device(args.device)

    # Wait for the first *ready* segment (header + first frame committed via the sidecar), not merely
    # for the .ser to exist -- the cam creates the file a beat before it writes the header.
    while True:
        if args.stop_file and os.path.exists(args.stop_file):
            return
        ready = [s for s in _segments(args.session, args.role) if _segment_ready(s)]
        if ready:
            break
        time.sleep(args.poll)

    # Live tracks the newest segment; offline starts at the oldest and processes in order.
    cur = ready[-1] if args.follow else ready[0]

    def open_segment(ser_path):
        reader = ser_mod.SerReader(ser_path)
        writer = JsonlWriter(ser_path[:-len('.ser')] + '.detections.jsonl')
        print(f"[detect:{args.role}] {os.path.basename(ser_path)}", flush=True)
        return reader, writer

    def new_surprise():                                # per-pixel temporal detector (stateful), or None
        if args.detector != 'surprise':
            return None
        return SurpriseModel(args.surprise_alpha_mean, args.surprise_alpha_var,
                             args.surprise_decay, args.surprise_var_floor_frac)

    reader, writer = open_segment(cur)
    prev = None
    surprise = new_surprise()
    dbg = {'writer': None, 'sidecar': None}            # debug detection-surface .ser (lazy, per segment)
    next_index = 0
    scale = None
    total = 0

    def close_debug():
        if dbg['writer'] is not None:
            dbg['writer'].close(); dbg['sidecar'].close()
            dbg['writer'] = dbg['sidecar'] = None

    # Track-mode: tail the backend state for a predicted ROI around the target. When present (this
    # role is being tracked), detect just that small window with a single-peak/centroid pass instead
    # of the whole frame -- far higher framerate, which is when we most need it (the catch-up slew).
    state = {'tailer': None, 'roi': None}

    def poll_state():
        if state['tailer'] is None:
            sf = sorted(glob.glob(os.path.join(args.session, '*_state.jsonl')))
            if sf:
                state['tailer'] = sidecar.JsonlTailer(sf[-1])
        if state['tailer'] is not None:
            for rec in state['tailer'].poll():
                state['roi'] = rec.get('track_roi') if rec.get('track_role') == args.role else None

    def process(i):
        nonlocal prev, scale, total
        frame = reader.read_frame(i)
        cid = reader.header.color_id
        if scale is None:
            scale = full_scale(cid, reader.header.pixel_depth_per_plane)
        work = work_image(frame, cid, device=device)
        coord_scale = reader.header.image_width / work.shape[1]    # frame px per (maybe half-res) work px
        # Keep the temporal model current every frame (even in track mode) so its state never goes stale.
        trail = surprise.update(work) if surprise is not None else None
        if state['roi'] is not None:                               # track mode: single peak in the ROI
            # ROI-peak needs a single-frame surface; 'surprise' is full-frame temporal, so fall back.
            roi_detector = 'bandpass' if args.detector == 'surprise' else args.detector
            blobs = detect_roi_peak(work, state['roi'], coord_scale, detector=roi_detector,
                                    bg_radius=args.bg_radius, psf_px=args.psf_px,
                                    doh_sigma=args.doh_sigma, snr=args.snr)   # already frame coords
            prev = None                                            # frame-diff not used in ROI mode
            debug_surface = trail                                  # in ROI mode the only surface we have
        else:                                                      # acquisition: full-frame multi-blob
            bp = trail if surprise is not None else detection_surface(
                work, detector=args.detector, bg_radius=args.bg_radius,
                psf_px=args.psf_px, doh_sigma=args.doh_sigma, compiled=True)   # fixed full-frame shape
            debug_surface = bp
            blobs = detect_blobs(
                bp, work, (None if surprise is not None else prev),
                threshold_rel=args.threshold, max_candidates=args.max_candidates,
                suppress_radius=args.suppress_radius, min_blob_px=args.min_blob_px,
                max_size_px=args.max_size_px, psf_px=args.psf_px,
                snr=args.snr, min_roundness=args.min_roundness,
                moving_frac=args.moving_frac, scale=scale,
                tile_grid=args.tile_grid, per_tile=args.per_tile)
            # Report blobs in the frame's image space. We may analyse a downsampled grid (Bayer ->
            # half-res mono sum), so scale coords back up; consumers then need no idea how we work.
            if coord_scale != 1:
                for b in blobs:
                    b['px'] = [b['px'][0] * coord_scale, b['px'][1] * coord_scale]
                    if 'size_px' in b:
                        b['size_px'] = b['size_px'] * coord_scale
            prev = bp
        writer.append({'index': i, 't_mono_ns': time.perf_counter_ns(), 'blobs': blobs})
        # Debug movie of the detection surface: a parallel .ser + commit spine the GUI can follow.
        if args.debug_ser and debug_surface is not None:
            if dbg['writer'] is None:
                dpath = cur[:-len('.ser')] + '_debug.ser'
                dh, dw = debug_surface.shape
                dbg['writer'] = ser_mod.SerWriter(dpath, dw, dh, color_id=ser_mod.ColorId.MONO,
                                                  pixel_depth_per_plane=16)
                dbg['sidecar'] = JsonlWriter(dpath[:-len('.ser')] + '.frames.jsonl')
            dbg['writer'].write_frame(debug_frame_u16(debug_surface))
            dbg['sidecar'].append({'t_mono_ns': time.perf_counter_ns(), 'index': i})
        total += 1

    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                break

            poll_state()                         # refresh the predicted track ROI (if any)
            avail = _committed(reader, cur)
            if args.follow:
                # Live: never build a backlog -- skip straight to the most recent frame.
                if avail - 1 >= next_index:
                    next_index = avail - 1
                    process(next_index)
                    next_index += 1
            else:
                while next_index < avail:        # offline: process every frame in order
                    process(next_index)
                    next_index += 1
                    avail = _committed(reader, cur)

            # Caught up on the current segment. Roll to a newer one -- but only once it's *ready*
            # (header + first frame committed), else we'd race the cam and open a header-less .ser.
            newer = [s for s in _segments(args.session, args.role) if s > cur and _segment_ready(s)]
            if newer:
                reader.close()
                writer.close()
                cur = newer[-1] if args.follow else newer[0]   # live: jump to newest
                close_debug()                          # finalize this segment's debug .ser
                reader, writer = open_segment(cur)
                prev = None
                surprise = new_surprise()              # fresh temporal state for the new segment
                next_index = 0
                scale = None
                continue

            # No newer segment: live waits; offline exits once this one is finalized.
            if _frame_count(cur) != ser_mod.SENTINEL_FRAME_COUNT and next_index >= _committed(reader, cur):
                if not args.follow:
                    break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        writer.close()
        close_debug()
        print(f"[detect:{args.role}] processed {total} frames", flush=True)


def full_scale(color_id, pixel_depth):
    """Max possible value of work_image, for an absolute 0..1 brightness score."""
    base = ser_mod.container_max(pixel_depth)
    return base * (4 if bayer.is_bayer(color_id) else 1)


def debug_frame_u16(surf):
    """Normalize a detection surface (float, possibly negative) to a uint16 greyscale image for the
    debug .ser: clamp to >=0, scale so the 99.5th percentile hits full white (robust to outliers).
    16-bit (not 8) -- the surface is *linear*, and the viewer applies gamma, which would band an 8-bit
    linear image badly."""
    a = torch.as_tensor(surf, dtype=torch.float32).clamp(min=0)
    flat = a.reshape(-1)
    sub = flat[:: max(1, flat.numel() // 100000)]                # subsample for a cheap robust high
    hi = float(torch.quantile(sub, 0.995)) if sub.numel() else 1.0
    img = (a / max(hi, 1e-6)).clamp(0, 1) * 65535.0
    return img.round().cpu().numpy()                             # SerWriter casts to uint16


if __name__ == '__main__':
    main()
