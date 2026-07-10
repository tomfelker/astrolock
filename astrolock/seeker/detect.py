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

_DEVICE = torch.device('cpu')        # fallback for direct library calls; the CLI picks via --device


def resolve_device(name):
    """Map a --device string to a torch.device. 'auto' (the default) -> cuda when present, else cpu."""
    if name in (None, '', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)

# The hot per-frame surfaces are torch.compile'd into single fused kernels where dynamo is supported and a
# C++ compiler is present (a real win on the live detector -- DoH ~85->66ms, surprise ~17->8ms at 1080p).
# is_dynamo_supported() gates the build (older torch RAISES on Windows, before suppress_errors could help),
# and suppress_errors degrades runtime graph breaks to eager -- so it's seamless. (TORCHDYNAMO_DISABLE=1 forces eager.)
# Compiled lazily + cached, and only ever applied to fixed-shape *full-frame* inputs -- never the
# variable-size track ROI -- so each compiles exactly once.
_compiled = {}


def _compiled_fn(fn):
    c = _compiled.get(fn)
    if c is None:
        c = fn                                         # eager fallback (default if compile is unavailable)
        try:
            import logging
            import torch._dynamo
            # is_dynamo_supported() is the boolean sibling of the check that RAISES "Windows not yet
            # supported" on older torch -- gate on it so torch.compile() can't hard-crash the process at
            # build time (a plain try/except is only the backstop for torch too old to even have it).
            if torch._dynamo.is_dynamo_supported():
                torch._dynamo.config.suppress_errors = True    # runtime graph breaks -> eager, quietly...
                logging.getLogger("torch._dynamo").setLevel(logging.ERROR)   # ...(e.g. no Triton on Win+CUDA)
                c = torch.compile(fn)
        except Exception:
            c = fn
        _compiled[fn] = c
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


def gaussian_deriv_kernels(sigma, radius, device=None):
    """1-D Gaussian and its 1st/2nd derivatives, sampled on [-radius, radius] (torch). Built on
    ``device`` so the conv kernels land where the image is (else a CUDA image + CPU kernel mismatch)."""
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
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
    g, g1, g2 = gaussian_deriv_kernels(sigma, radius, device=work.device)
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

      blur:      optional spatial Gaussian low-pass (sigma `blur_px`) on the SIGNED surprisal before
                 the clamp -- the matched filter for an extended mover. A trail lights several
                 adjacent pixels at once, so the blur integrates them, while independent per-pixel
                 noise (zero-mean pre-clamp, so it cancels rather than piling into a pedestal)
                 averages down by ~the kernel norm. The result is divided by that norm, so it stays
                 in true sigma units: a 3-sigma-per-pixel trail reads ~several sigma, single-pixel
                 twinkle reads ~1. The glasses-off trick, quantified.

      trail:     trail = max(decay * trail, z) -- a decaying peak-hold of the surprise. A mover paints
                 its recent path as a bright comet: each pixel keeps the FULL spike height as the object
                 passes, fading over ~1/(1-decay) frames, so a per-frame few-sigma spike becomes a
                 spatially extended, connected feature that detect_blobs can catch (same block-max /
                 density-cap path as the other detectors), while isolated noise just decays. `decay`
                 trades latency for sensitivity: ~0 = single-frame, ->1 = long integration for the
                 faintest trails.

    var_floor_frac ties the noise floor to the typical per-pixel temporal variance (its median across
    the frame), so no absolute-DN tuning is needed. Needs ~1/alpha frames of warm-up before the
    variance estimate settles. The elementwise pieces are torch.compile'd into fused kernels (see
    _compiled_fn); the blur is two separable 1-D convs. See astrolock_seeker.md."""

    def __init__(self, alpha_mean=0.15, alpha_var=0.08, decay=0.85, var_floor_frac=0.5, blur_px=1.5):
        self.a_m, self.a_v, self.decay, self.vff = alpha_mean, alpha_var, decay, var_floor_frac
        self.blur_px = blur_px
        self._g = self._g_norm = None                         # blur kernel (built lazily on the work device)
        self.mu = self.var = self.trail = self.z = None       # z = this frame's surprise map (sigma units)
        self.zs = None                                        # ...and its SIGNED twin (dark movers dip below 0)
        self._mu_f = None                                     # matched-filtered mu (lazy, per frame)

    def _kernel(self, device):
        if self._g is None:
            radius = max(1, int(4.0 * self.blur_px + 0.5))
            self._g, _, _ = gaussian_deriv_kernels(self.blur_px, radius, device=device)
            self._g_norm = float((self._g ** 2).sum())
        return self._g

    def _blur_unit_noise(self, zs):
        """Separable Gaussian blur, renormalized so unit-variance input noise stays unit-variance:
        for kernel k (per axis), independent noise std scales by sqrt(sum k^2) per axis, so divide
        the 2-D result by sum(k^2) (computed from the actual truncated kernel, so it's exact)."""
        g = self._kernel(zs.device)
        return _conv1d_axis(_conv1d_axis(zs, g, 1), g, 0) / self._g_norm

    def mu_filtered(self):
        """The running mean through the same matched filter (plain flux-preserving blur; no noise
        renorm -- the tile pass thresholds relative to each tile's own stats, so only the RATIO
        matters). The detection surface for FIXED targets; read scores/display off the raw mu.
        Lazy per frame: computed on first call after an update, cached until the next."""
        if self.blur_px <= 0 or self.mu is None:
            return self.mu
        if self._mu_f is None:
            g = self._kernel(self.mu.device)
            self._mu_f = _conv1d_axis(_conv1d_axis(self.mu, g, 1), g, 0)
        return self._mu_f

    def update(self, work):
        work = torch.as_tensor(work, dtype=torch.float32)
        if self.mu is None:                                   # first frame: seed, emit nothing
            self.mu = work.clone()
            self.var = torch.full_like(work, float(work.var()) + 1.0)
            self.trail = torch.zeros_like(work)
            self.z = torch.zeros_like(work)
            self.zs = torch.zeros_like(work)
            return self.trail
        floor = self.vff * torch.median(self.var) + 1e-6      # 0-d tensor (dynamic; no per-frame recompile)
        zs, self.mu, self.var = _compiled_fn(_surprise_step)(
            work, self.mu, self.var, floor, self.a_m, self.a_v)
        if self.blur_px > 0:
            zs = self._blur_unit_noise(zs)
        self.zs = zs
        self._mu_f = None                                     # mu moved -> refilter on next mu_filtered()
        self.z, self.trail = _compiled_fn(_trail_step)(zs, self.trail, self.decay)
        return self.trail


def _surprise_step(work, mu, var, floor, a_m, a_v):
    """SurpriseModel EMA step, elementwise -> one fused kernel: SIGNED surprise z-score + EMA
    mean/variance updates. Returns (z_signed, mu, var). (Signed, so the optional spatial blur sees
    zero-mean noise; the clamp happens in _trail_step after it.)"""
    d = work - mu
    return d / torch.sqrt(var + floor), mu + a_m * d, (1.0 - a_v) * var + a_v * d * d


def _trail_step(zs, trail, decay):
    """Clamp the (possibly blurred) signed surprisal + decaying peak-hold, fused. Returns (z, trail)."""
    z = zs.clamp(min=0.0)
    return z, torch.maximum(decay * trail, z)


def tile_targets(mu, zs, work, *, tile_size, fixed_nsigma, moving_nsigma, mask_px, scale,
                 mu_score=None):
    """
    Per-tile extremum detector: the acquisition pass for the surprise detector.

    The frame is cut into tile_size^2 tiles and each tile may emit up to 4 targets:

      FIXED bright/dark:  the max / min pixel of ``mu`` (the per-pixel temporal running mean -- a
                          deeply denoised sky, so stars stand far above a tile's spatial noise),
                          kept if it sits >= fixed_nsigma spatial stdevs from the tile's mean.
      MOVING bright/dark: the max / min pixel of ``zs`` (the SIGNED, matched-filtered surprisal;
                          a dark mover dips negative), kept if >= moving_nsigma stdevs from the
                          tile's spatial mean -- computed with all fixed extrema masked out,
                          because a twinkling star IS temporally surprising. The mask dilates
                          every tile's fixed extrema (kept or not: the tile's brightest star is
                          the twinkle risk even below threshold) by mask_px, and it's GLOBAL, so
                          a star's halo can't leak into a neighbouring tile's mover slot.

    One argmax/argmin + mean/var per tile per surface, all batched (a reshape to (tiles, S*S) and
    per-tile reductions) -- embarrassingly parallel, no global sort, no NMS, no shape cuts. False
    positives are cheap by design: the user (or the tracker's own gating) picks among candidates,
    and a tile can never emit more than 4. Thresholds are per-tile ("N stdevs above THIS tile's
    background"), so vignetting / sky gradients / light pollution self-normalize. Edge tiles are
    replicate-padded, slightly overweighting edge pixels in the stats -- harmless.

    Returns blob dicts in WORK coords (caller scales to frame coords): 'moving' True/False,
    'dark' True on the minima, 'nsigma' = stdevs from the tile mean, 'score' = absolute
    brightness of that pixel (from ``mu_score`` -- the raw running mean, when ``mu`` itself is
    a matched-filtered surface -- for fixed; the live frame for movers).
    """
    F = torch.nn.functional
    mu = torch.as_tensor(mu, dtype=torch.float32)
    zs = torch.as_tensor(zs, dtype=torch.float32)
    work = torch.as_tensor(work, dtype=torch.float32)
    mu_score = mu if mu_score is None else torch.as_tensor(mu_score, dtype=torch.float32)
    H, W = mu.shape
    S = int(tile_size)
    ny, nx = (H + S - 1) // S, (W + S - 1) // S
    ph, pw = ny * S - H, nx * S - W
    T = ny * nx
    dev = mu.device

    def pad(img):
        return F.pad(img[None, None], (0, pw, 0, ph), mode='replicate')[0, 0] if (ph or pw) else img

    def tiles(img):
        return img.reshape(ny, S, nx, S).permute(0, 2, 1, 3).reshape(T, S * S)

    # ---- fixed pass: extrema of the running mean vs the tile's spatial stats -------------------
    tm = tiles(pad(mu))
    fmean = tm.mean(dim=1)
    fstd = tm.std(dim=1) + 1e-6
    fmax, imax = tm.max(dim=1)
    fmin, imin = tm.min(dim=1)
    f_hi = (fmax - fmean) / fstd
    f_lo = (fmean - fmin) / fstd

    # ---- mask around every fixed extremum: separable square dilation on the padded canvas ------
    tid = torch.arange(T, device=dev)
    tid2 = tid.repeat(2)
    ii2 = torch.cat([imax, imin])
    my = (tid2 // nx) * S + ii2 // S
    mx = (tid2 % nx) * S + ii2 % S
    mask = torch.zeros(ny * S, nx * S, dtype=torch.float32, device=dev)
    mask[my, mx] = 1.0
    r, k = int(mask_px), 2 * int(mask_px) + 1
    mask = F.max_pool2d(mask[None, None], (k, 1), stride=1, padding=(r, 0))
    mask = F.max_pool2d(mask, (1, k), stride=1, padding=(0, r))[0, 0] > 0

    # ---- moving pass: extrema of the signed surprisal, fixed extrema excluded ------------------
    tz = tiles(pad(zs))
    valid = tiles(mask.logical_not())
    cnt = valid.sum(dim=1).clamp(min=1)
    zmean = (tz * valid).sum(dim=1) / cnt
    zstd = ((((tz - zmean[:, None]) * valid) ** 2).sum(dim=1) / cnt).sqrt() + 1e-6
    ninf = torch.tensor(float('-inf'), device=dev)
    zmax, jmax = torch.where(valid, tz, ninf).max(dim=1)
    zmin, jmin = torch.where(valid, tz, -ninf).min(dim=1)
    m_hi = (zmax - zmean) / zstd
    m_lo = (zmean - zmin) / zstd

    out = []

    def emit(keep, idx, sig, moving, dark, surf):
        sel = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if sel.numel() == 0:
            return
        ii = idx[sel]
        # tile-local -> image coords; clamp maps a replicate-padded pick back onto the real edge px
        y = ((tid[sel] // nx) * S + ii // S).clamp(max=H - 1)
        x = ((tid[sel] % nx) * S + ii % S).clamp(max=W - 1)
        sc = surf[y, x] / scale
        for xx, yy, ss, gg in zip(x.tolist(), y.tolist(), sc.tolist(), sig[sel].tolist()):
            out.append({'px': [float(xx), float(yy)], 'score': round(ss, 4),
                        'nsigma': round(gg, 1), 'moving': moving,
                        **({'dark': True} if dark else {})})

    emit(f_hi >= fixed_nsigma, imax, f_hi, False, False, mu_score)
    emit(f_lo >= fixed_nsigma, imin, f_lo, False, True, mu_score)
    emit(m_hi >= moving_nsigma, jmax, m_hi, True, False, work)
    emit(m_lo >= moving_nsigma, jmin, m_lo, True, True, work)
    for i, b in enumerate(out):
        b['id'] = i
    return out


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

    n_above = (win >= 0.5 * peak).sum(dim=(1, 2))                      # size proxy: area above half-max
    # Background-subtracted intensity-weighted centroid over the WHOLE window. Subtract the window mean
    # (removes any DC pedestal -- e.g. the DoH surface's positive floor -- that would otherwise drag the
    # centroid toward the window centre; the star is a few of ~t^2 pixels so it barely inflates the mean),
    # then clamp and take the true first moment. (The old (win - 0.5*peak) core threshold discarded
    # everything but the peak pixel for a tight/undersampled star, collapsing onto the integer peak.)
    bg = win.mean(dim=(1, 2), keepdim=True)
    wsub = (win - bg).clamp(min=0.0)
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
        'px': [px[i], py[i]],                                # [x, y] in the work image (full sub-pixel)
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


def matched_roi_peak(work, roi, coord_scale, *, blur_px, psf_px, scale, pull):
    """Tracking-phase detector ('matched'): stateless, current-frame only.

    While locked we want to HOLD track, so there is no found/lost gate: crop the predicted ROI,
    matched-filter it (Gaussian ~ the PSF), standardize against the window's own mean/std (z~,
    spatial sigma units), then pick the EXTREMAL pixel of |z~| - pull*distance. The cone penalty
    is linear and in sigma units: a candidate d px from the prediction wins iff it's pull*d
    sigmas more surprising -- equivalently a Laplacian (heavy-tailed) prior on prediction error,
    forgiving of the big misses a catch-up slew produces where a multiplicative Gaussian would
    wall off. Extremal (|z~|) tracks dark targets too (daytime planes / silhouettes); a dip
    carries 'dark': True. Deliberately no temporal state -- per-pixel history only adds lag when
    the mount is moving the whole field.

    Always answers (the only empty return is an ROI fully off the frame). Confidence, never a
    gate: 'nsigma' = |z~| at the pick, and 'conf' = nsigma - E[max of the window under NO
    target]. That null max is sqrt(2 ln(2 N_eff)) with N_eff = pixels / (2.355*blur_px)^2 --
    extreme value theory over the window's independent resolution elements (the matched filter
    correlates neighbours, so resels not pixels; the 2x is the two-sided extremal test). So
    conf ~ 0 honestly reads "answering from pure noise". TODO: replace/validate the Gumbel
    approximation with a one-time Monte-Carlo E[max] (blur white-noise windows of this size
    with the real kernel and average the maxima) -- exact for our filter, cheap at startup."""
    work = torch.as_tensor(work, dtype=torch.float32)
    H, W = work.shape
    cx, cy, size = roi
    half = (size / coord_scale) / 2.0
    ecx, ecy = cx / coord_scale, cy / coord_scale            # expected centre, work coords
    x0, x1 = max(0, int(ecx - half)), min(W, int(ecx + half) + 1)
    y0, y1 = max(0, int(ecy - half)), min(H, int(ecy + half) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []                                            # ROI entirely off the frame
    sub = work[y0:y1, x0:x1]
    if blur_px > 0:                                          # matched filter (reflect pad needs r < dim)
        r = min(max(1, int(4.0 * blur_px + 0.5)), (min(sub.shape) - 1) // 2)
        g, _, _ = gaussian_deriv_kernels(blur_px, r, device=sub.device)
        sub = _conv1d_axis(_conv1d_axis(sub, g, 1), g, 0)
    zt = (sub - sub.mean()) / (sub.std() + 1e-6)             # window-standardized, spatial sigma units
    h, w = zt.shape
    dev = zt.device
    yy, xx = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=dev),
                            torch.arange(w, dtype=torch.float32, device=dev), indexing='ij')
    dist = torch.sqrt((xx - (ecx - x0)) ** 2 + (yy - (ecy - y0)) ** 2)
    pidx = int(torch.argmax(zt.abs() - pull * dist))
    py, px = pidx // w, pidx % w
    sgn = 1.0 if float(zt[py, px]) >= 0.0 else -1.0
    nsig = abs(float(zt[py, px]))
    resel = max(1.0, (2.355 * blur_px) ** 2)                 # pixels per independent resolution element
    n_eff = max(2.0, (h * w) / resel)
    conf = nsig - math.sqrt(2.0 * math.log(2.0 * n_eff))    # excess over the empty-window expected max
    cr = max(1, int(round(psf_px + 2.0 * blur_px)))          # centroid window covers the FILTERED blob
    wy0, wy1 = max(0, py - cr), min(h, py + cr + 1)
    wx0, wx1 = max(0, px - cr), min(w, px + cr + 1)
    patch = (zt[wy0:wy1, wx0:wx1] * sgn).clamp(min=0)        # flip a dark target positive for the centroid
    s = float(patch.sum())
    if s > 0:
        pyy, pxx = torch.meshgrid(torch.arange(wy0, wy1, dtype=torch.float32, device=dev),
                                  torch.arange(wx0, wx1, dtype=torch.float32, device=dev), indexing='ij')
        cpx, cpy = float((patch * pxx).sum()) / s, float((patch * pyy).sum()) / s
    else:
        cpx, cpy = float(px), float(py)
    out = {'px': [(x0 + cpx) * coord_scale, (y0 + cpy) * coord_scale], 'moving': True,
           'size_px': psf_px * coord_scale, 'nsigma': round(nsig, 1), 'conf': round(conf, 1),
           'score': round(float(work[y0 + py, x0 + px]) / scale, 4)}
    if sgn < 0:
        out['dark'] = True
    return [out]


def marginal_target(work, *, nsigma=4.0, density_thresh=0.7):
    """Symmetric single-extended-target detector (satellites, planes, rockets, daytime silhouettes).

    Assumes ONE bright-or-dark compact target on a flat background. Project the frame onto each axis
    (mean across rows -> the x marginal; mean across cols -> the y marginal), and on each 1-D marginal:
    threshold two-tailed at ``|value - mean| > nsigma*stdev`` -- two-tailed so a dark daytime silhouette
    registers as well as a bright satellite -- then measure how CLUSTERED the above-threshold columns are.
    On an empty frame they scatter across the axis; on a target they clump. ``density`` is that
    compactness (~1 = clumped target, ~0.5 = uniform noise). ``present`` := density over its threshold in
    BOTH axes. All coords in WORK px."""
    work = torch.as_tensor(work, dtype=torch.float32)

    def axis(profile):
        length = profile.numel()
        position = torch.arange(length, dtype=torch.float32, device=profile.device)
        background = profile.mean()
        noise = profile.std()
        is_target = torch.abs(profile - background) > nsigma * noise    # two-tailed: bright OR dark stands out
        mass = int(is_target.sum())                                     # how many columns cleared the threshold
        if mass < 1:
            return {'present': False, 'center': length / 2.0, 'lo': 0.0, 'hi': length - 1.0,
                    'density': 0.0, 'mass': 0}
        hit = is_target.float()
        center = float((hit * position).sum() / mass)                  # centre of the above-threshold columns
        avg_distance = float((hit * (position - center).abs()).sum() / mass)
        # The average distance from the centre runs from `packed` (all mass in a solid run around the
        # centre) up to `spread` = (length-1)/2 (mass split to the two extreme columns 0 and length-1).
        # density = where avg_distance lands in that range: 1 at the packed end, 0 at the spread end.
        # Both ends are the EXACT discrete values, so a solid clump (and a single point) come out to
        # exactly 1 and the fully-split case to exactly 0. `packed` is the discrete average distance of a
        # solid run of `mass` columns -- the sum of |i-centre| over consecutive integers is floor(mass^2/4).
        # Uniform noise (avg distance ~length/4, about halfway) -> ~0.5. Pixel-size invariant.
        packed, spread = (mass * mass // 4) / mass, (length - 1) / 2.0
        density = (spread - avg_distance) / (spread - packed)
        edges = position[is_target]
        present = density >= density_thresh
        return {'present': present, 'center': center, 'lo': float(edges.min()), 'hi': float(edges.max()),
                'density': density, 'mass': mass}

    x = axis(work.mean(dim=0))          # x marginal: mean across rows
    y = axis(work.mean(dim=1))          # y marginal: mean across cols
    present = x['present'] and y['present']
    peak = 0.0
    if present:
        window = work[int(y['lo']):int(y['hi']) + 1, int(x['lo']):int(x['hi']) + 1]
        peak = float(window.max()) if window.numel() else float(work.max())
    return {'present': present,
            'com': [x['center'], y['center']],
            'bbox': [x['lo'], y['lo'], x['hi'] - x['lo'], y['hi'] - y['lo']],
            'density': [x['density'], y['density']],
            'mass': [x['mass'], y['mass']],
            'peak': peak}


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
    p.add_argument('--detector', default='doh', choices=['bandpass', 'doh', 'surprise', 'extended'],
                   help="detection surface: 'doh' (default) = determinant of the Hessian "
                        "(Gaussian-derivative blob detector; rejects edges/lines by construction), "
                        "'bandpass' (the older local-background subtraction), 'surprise' = per-pixel "
                        "temporal surprise + decaying peak-hold (finds faint fast movers / satellite "
                        "trails that the single-frame surfaces miss; see SurpriseModel), or 'extended' = "
                        "the symmetric single-bright-or-dark extended-target detector (marginal projection "
                        "+ compactness; for narrow-field main-cam tracking; see marginal_target)")
    p.add_argument('--surprise-alpha-mean', type=float, default=0.15,
                   help="surprise: EMA rate for the per-pixel mean (bigger = adapts faster to drift)")
    p.add_argument('--surprise-alpha-var', type=float, default=0.08,
                   help="surprise: EMA rate for the per-pixel variance")
    p.add_argument('--surprise-decay', type=float, default=0.85,
                   help="surprise: trail peak-hold decay per frame (~0 = single-frame; ->1 integrates "
                        "longer for the faintest trails, at more latency)")
    p.add_argument('--surprise-var-floor-frac', type=float, default=0.5,
                   help="surprise: noise floor as this fraction of the median per-pixel variance")
    p.add_argument('--surprise-blur-px', type=float, default=1.5,
                   help="surprise: spatial Gaussian low-pass sigma (px) on the surprisal map -- "
                        "integrates a trail's adjacent pixels while averaging independent per-pixel "
                        "noise down (renormalized, so the map stays in sigma units). 0 = off")
    p.add_argument('--tile-size', type=int, default=128,
                   help="surprise: tile edge (px) for the per-tile extremum pass; each tile emits at "
                        "most 4 targets (fixed bright/dark from the running mean, moving bright/dark "
                        "from the surprisal)")
    p.add_argument('--tile-fixed-nsigma', type=float, default=8.0,
                   help="surprise: keep a tile's fixed (running-mean) extremum only if it's this many "
                        "spatial stdevs from the tile mean")
    p.add_argument('--tile-moving-nsigma', type=float, default=5.0,
                   help="surprise: keep a tile's moving (surprisal) extremum only if it's this many "
                        "spatial stdevs from the tile mean (fixed extrema masked out)")
    p.add_argument('--tile-mask-px', type=int, default=16,
                   help="surprise: mask radius (px) around every tile's fixed extrema when hunting "
                        "movers, so a twinkling star can't claim the mover slot")
    p.add_argument('--track-detector', default='peak', choices=['peak', 'matched'],
                   help="the TRACKING-phase detector (answers 'where is the locked target?' inside the "
                        "backend's predicted ROI, one answer per frame; independent of --detector): "
                        "'peak' = the original detection-surface peak with a brightness found-test, "
                        "'matched' = stateless matched filter + centre pull + argmax on the current "
                        "frame, no found/lost gate (confidence reported, never used to refuse)")
    p.add_argument('--track-blur-px', type=float, default=1.5,
                   help="matched track detector: matched-filter Gaussian sigma (px), ~the PSF width")
    p.add_argument('--track-pull', type=float, default=0.5,
                   help="matched track detector: cone pull strength (sigma per work px) -- a candidate "
                        "d px from the prediction must be pull*d sigmas more surprising to win the "
                        "argmax. Bigger = harder for a bright field star to steal the lock, but less "
                        "tolerant of big prediction misses during a catch-up slew")
    p.add_argument('--ext-nsigma', type=float, default=3.0,
                   help="extended: two-tailed |value-mean| threshold on the row/col marginals, in sigmas "
                        "(catches bright AND dark targets; lower = includes dimmer target extremities)")
    p.add_argument('--ext-density', type=float, default=0.7,
                   help="extended: min compactness to call a target present (~1 = tightly clumped target, "
                        "~0.5 = uniform noise), in both axes")
    p.add_argument('--debug-ser', action='store_true',
                   help="also write the detection surface as <seg>_<role>_debug.ser (+ .frames.jsonl), a "
                        "normalized greyscale movie of exactly what the detector 'sees' -- follow it in "
                        "the GUI or any SER viewer to tune. For 'surprise' this is the raw per-frame "
                        "surprisal (fixed scale: white = 10 sigma), not the peak-hold trail.")
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
    p.add_argument('--device', default='auto',
                   help="torch device for detection: 'auto' (default) = cuda if present else cpu, "
                        "or force 'cpu' / 'cuda'")
    args = p.parse_args(argv)
    device = resolve_device(args.device)
    print(f"[detect:{args.role}] compute device: {device}", flush=True)

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
        return reader, writer

    def new_surprise():                                # per-pixel temporal detector (stateful), or None
        if args.detector != 'surprise':
            return None
        return SurpriseModel(args.surprise_alpha_mean, args.surprise_alpha_var,
                             args.surprise_decay, args.surprise_var_floor_frac,
                             blur_px=args.surprise_blur_px)

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
                state['roi'] = (rec.get('track_roi') or {}).get(args.role)   # per-role: {role: [cx,cy,size]}

    def process(i):
        nonlocal prev, scale, total
        t_start = time.perf_counter()                              # whole-frame cost -> proc_ms in the record
        frame = reader.read_frame(i)
        cid = reader.header.color_id
        if scale is None:
            scale = full_scale(cid, reader.header.pixel_depth_per_plane)
        work = work_image(frame, cid, device=device)
        coord_scale = reader.header.image_width / work.shape[1]    # frame px per (maybe half-res) work px
        # Keep the temporal model current every frame (even in track mode) so its state never goes stale.
        trail = surprise.update(work) if surprise is not None else None
        extra, status = {}, ''                                     # extra record fields + a freeform status line

        def run_extended():
            """The 'detection' phase (main-cam style): full-frame single-large-target,
            present-or-not (marginal projection + compactness). Returns frame coords."""
            ext = marginal_target(work, nsigma=args.ext_nsigma, density_thresh=args.ext_density)
            cs = coord_scale
            com_f = [ext['com'][0] * cs, ext['com'][1] * cs]
            bbox_f = [v * cs for v in ext['bbox']]
            bl = ([{'px': com_f, 'bbox': bbox_f, 'moving': True,
                    'score': round(ext['peak'] / (scale or 1.0), 4)}] if ext['present'] else [])
            ex = {'ext': {'present': ext['present'], 'com': com_f, 'bbox': bbox_f,
                          'density': [round(d, 3) for d in ext['density']],
                          'mass': [round(m, 1) for m in ext['mass']]}}
            st = (f"{'Target' if ext['present'] else 'No Target'} "
                  f"(Density X: {ext['density'][0]:.2f}, Y: {ext['density'][1]:.2f})")
            return bl, ex, st

        # ---- phase dispatch ------------------------------------------------------------------
        # acquisition: no lock -> full-frame, many candidates (the user picks one).
        # tracking:    the backend broadcasts a predicted ROI for this role -> one answer,
        #              found by the per-phase --track-detector (independent of --detector).
        # detection:   'extended' -- the main-cam single-large-target present-or-not test;
        #              runs full-frame in every phase (the backend never sends it an ROI).
        if args.detector == 'extended':
            blobs, extra, status = run_extended()
            debug_surface, debug_hi = work, None                   # debug-ser shows the analysed frame
            prev = None
        elif state['roi'] is not None:                             # tracking: one answer near the prediction
            if args.track_detector == 'matched':                   # stateless current-frame matched filter
                blobs = matched_roi_peak(work, state['roi'], coord_scale,
                                         blur_px=args.track_blur_px, psf_px=args.psf_px,
                                         scale=scale, pull=args.track_pull)
            else:                                                  # 'peak': single-frame surface peak
                roi_det = args.detector if args.detector in ('bandpass', 'doh') else 'bandpass'
                blobs = detect_roi_peak(work, state['roi'], coord_scale, detector=roi_det,
                                        bg_radius=args.bg_radius, psf_px=args.psf_px,
                                        doh_sigma=args.doh_sigma, snr=args.snr)   # already frame coords
            prev = None                                            # frame-diff not used in ROI mode
            if surprise is not None:                               # keep showing the surprisal if it's running
                debug_surface, debug_hi = surprise.z, 10.0         # sigma units: white = 10 sigma
            else:
                debug_surface, debug_hi = work, None
            status = "Locked On" if blobs else "Searching"
        else:                                                      # acquisition: full-frame multi-blob
            if args.detector == 'surprise':
                # Tile-extremum pass: fixed targets from the running mean, movers from the signed
                # matched-filtered surprisal (fixed extrema masked out). Debug shows the RAW
                # per-frame surprisal (sigma units).
                blobs = tile_targets(surprise.mu_filtered(), surprise.zs, work,
                                     tile_size=args.tile_size, fixed_nsigma=args.tile_fixed_nsigma,
                                     moving_nsigma=args.tile_moving_nsigma,
                                     mask_px=args.tile_mask_px, scale=scale,
                                     mu_score=surprise.mu)
                debug_surface, debug_hi = surprise.z, 10.0        # sigma units: white = 10 sigma
                nf = sum(1 for b in blobs if not b['moving'])
                status = f"{nf} fixed + {len(blobs) - nf} moving"
            else:
                bp = detection_surface(
                    work, detector=args.detector, bg_radius=args.bg_radius,
                    psf_px=args.psf_px, doh_sigma=args.doh_sigma, compiled=True)   # fixed full-frame shape
                debug_surface, debug_hi = bp, None                 # arbitrary units: autoscale
                blobs = detect_blobs(
                    bp, work, prev,
                    threshold_rel=args.threshold, max_candidates=args.max_candidates,
                    suppress_radius=args.suppress_radius, min_blob_px=args.min_blob_px,
                    max_size_px=args.max_size_px, psf_px=args.psf_px,
                    snr=args.snr, min_roundness=args.min_roundness,
                    moving_frac=args.moving_frac, scale=scale,
                    tile_grid=args.tile_grid, per_tile=args.per_tile)
                prev = bp
                status = f"Acquired {len(blobs)} target{'' if len(blobs) == 1 else 's'}"
            # Report blobs in the frame's image space. We may analyse a downsampled grid (Bayer ->
            # half-res mono sum), so scale coords back up; consumers then need no idea how we work.
            if coord_scale != 1:
                for b in blobs:
                    b['px'] = [b['px'][0] * coord_scale, b['px'][1] * coord_scale]
                    if 'size_px' in b:
                        b['size_px'] = b['size_px'] * coord_scale
        writer.append({'index': i, 't_mono_ns': time.perf_counter_ns(),
                       'proc_ms': round((time.perf_counter() - t_start) * 1e3, 1),
                       'blobs': blobs, 'status': status, **extra})
        # Debug movie of the detection surface: a parallel .ser + commit spine the GUI can follow.
        if args.debug_ser and debug_surface is not None:
            if dbg['writer'] is None:
                dpath = cur[:-len('.ser')] + '_debug.ser'
                dh, dw = debug_surface.shape
                dbg['writer'] = ser_mod.SerWriter(dpath, dw, dh, color_id=ser_mod.ColorId.MONO,
                                                  pixel_depth_per_plane=16)
                dbg['sidecar'] = JsonlWriter(dpath[:-len('.ser')] + '.frames.jsonl')
            # A surprisal surface is in true sigma units -> FIXED scale (debug_hi, white = 10 sigma)
            # so standout is judgeable across frames; other surfaces autoscale per frame (hi=None).
            dbg['writer'].write_frame(debug_frame_u16(debug_surface, hi=debug_hi))
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


def debug_frame_u16(surf, hi=None):
    """Normalize a detection surface (float, possibly negative) to a uint16 greyscale image for the
    debug .ser: clamp to >=0, scale so `hi` hits full white. hi=None -> per-frame autoscale to the
    99.5th percentile (robust to outliers; for surfaces with arbitrary units). Pass a fixed `hi` for
    a surface with meaningful units (the surprisal z map, in sigmas) so brightness is comparable
    across frames. 16-bit (not 8) -- the surface is *linear*, and the viewer applies gamma, which
    would band an 8-bit linear image badly."""
    a = torch.as_tensor(surf, dtype=torch.float32).clamp(min=0)
    if hi is None:
        flat = a.reshape(-1)
        sub = flat[:: max(1, flat.numel() // 100000)]            # subsample for a cheap robust high
        hi = float(torch.quantile(sub, 0.995)) if sub.numel() else 1.0
    img = (a / max(hi, 1e-6)).clamp(0, 1) * 65535.0
    return img.round().cpu().numpy()                             # SerWriter casts to uint16


if __name__ == '__main__':
    main()
