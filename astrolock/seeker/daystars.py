"""
Daytime-star detector: pull sidereal-rate drifters out of daytime sky captures.

The camera sits still, so a star is a faint point gliding at a constant (slow) velocity
under a huge photon-noise background. We exploit exactly that model:

1. **Stack** the raw frames into ``--stacks`` (default 32) mean-stacked slices -> a
   (T, H, W) cube. Stacking buys shot-noise averaging; at sidereal rates (<~2 px per
   ~2 s stack on the 25 mm guide lens) the intra-stack smear stays under the PSF.
2. Subtract the **per-pixel temporal mean** of the cube: static scene, sky gradient,
   fixed-pattern noise and hot pixels vanish entirely (a drifter loses only the sliver
   of mean it contributed to each pixel it crossed).
3. Cut the cube into ``chunk``^3 voxel blocks (full T x chunk x chunk, overlapping by
   ``stride``), subtract each time-slice's spatial mean (kills global flicker: clouds,
   haze), and taper with a separable Hann window on all edges.
4. FFT each block. A constant-velocity point source is a **plane through the origin**
   of the 3D spectrum (normal ~ (1, v)). Take the squared magnitude and inverse-FFT:
   that's the **autocorrelation**, where each plane collapses to a **line through the
   origin** with direction (1, vy, vx).
5. Integrate the autocorrelation along the line for every candidate velocity on a grid
   (|v| in [vmin, vmax] px/stack -- nonzero to skip anything static, capped near the
   fastest sidereal drift). The integral, normalized by the block's total energy
   A(0,0,0), is the match score for "a point source moving at v lives here".
6. Score significance is a robust z (median/MAD across all blocks, per velocity, since
   line length varies with |v|). Overlapping hits are merged by greedy NMS.

``--inject x,y,vx,vy,peak`` drops a synthetic Gaussian drifter into the *real* cube
before detection -- a calibrated end-to-end sensitivity probe (find the peak amplitude
where z clears ~5 and you know the magnitude limit of the whole pipeline).

Localizing a detection *within* its block (from the FFT phase) is a follow-up; for now
positions are block centers (+-chunk/2).

    python -m astrolock.seeker.daystars recordings/foo.ser --arcsec-per-px 16.5
    python -m astrolock.seeker.daystars recordings/foo.ser --inject 1920,1080,1.0,0.5,0.002
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as Fnn

from . import ser as ser_mod


# ---------------------------------------------------------------- stacking

def stack_ser(path, n_stacks, cache_dir=None):
    """
    Mean-stack the SER into a (n_stacks, H, W) float32 cube, container-normalized to 0..1.
    Cached as .npy (+ .json meta) in ``cache_dir`` (default: <ser dir>/daystars_cache) --
    the 20+ GB read only happens once per (file, n_stacks).
    Returns (cube, meta) where meta carries frames/stack and the per-stack duration.
    """
    cdir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(path)), 'daystars_cache')
    os.makedirs(cdir, exist_ok=True)
    stem = os.path.join(cdir, f'{os.path.basename(path)}.stack{n_stacks}')
    if os.path.exists(stem + '.npy') and os.path.exists(stem + '.json'):
        with open(stem + '.json') as f:
            meta = json.load(f)
        return np.load(stem + '.npy'), meta

    with ser_mod.SerReader(path) as r:
        n = r.frames_on_disk()
        if n < n_stacks:
            raise SystemExit(f"{path}: only {n} frames, need >= {n_stacks}")
        h, w = r.header.image_height, r.header.image_width
        if r.num_channels != 1:
            raise SystemExit(f"{path}: expected mono frames")
        per = n // n_stacks                      # trailing remainder frames are dropped
        scale = 1.0 / (per * ser_mod.container_max(r.header.pixel_depth_per_plane))
        cube = np.zeros((n_stacks, h, w), np.float32)
        t0 = time.monotonic()
        for s in range(n_stacks):
            acc = cube[s]
            for i in range(s * per, (s + 1) * per):
                acc += r.read_frame(i)
            acc *= scale
            print(f"\r  stacking {s + 1}/{n_stacks} "
                  f"({(s + 1) * per / (time.monotonic() - t0):.0f} fps)", end='', flush=True)
        print()

        duration = None
        if r.finalized() and n > 1:              # timestamp trailer -> true capture duration
            with open(path, 'rb') as f:
                f.seek(ser_mod.HEADER_SIZE + n * r.bytes_per_frame)
                raw = f.read(8 * n)
            if len(raw) == 8 * n:
                ts = np.frombuffer(raw, dtype='<i8')
                duration = float((ts[-1] - ts[0]) / 1e7)

    stack_dt = duration * per / (n - 1) if duration else None    # seconds spanned per stack
    meta = {'n_frames': n, 'frames_per_stack': per, 'duration_s': duration,
            'stack_dt_s': stack_dt, 'width': w, 'height': h}
    np.save(stem + '.npy', cube)
    with open(stem + '.json', 'w') as f:
        json.dump(meta, f)
    return cube, meta


def inject_star(cube, x, y, vx, vy, peak, sigma=1.3):
    """Add a Gaussian drifter (peak amplitude in normalized units) at (x, y) on stack 0,
    moving (vx, vy) px/stack. In-place; a ground-truth probe for the detector."""
    t_dim, h, w = cube.shape
    rad = int(math.ceil(4 * sigma))
    for t in range(t_dim):
        cx, cy = x + vx * t, y + vy * t
        x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad + 1)
        y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cube[t, y0:y1, x0:x1] += peak * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)


def transit_maps(cube, window):
    """
    Excluded-window transit statistics, per pixel (user-designed, 2026-07-17):
    compare each pixel's whole-capture mean/std against the mean/std computed with a
    sliding time window EXCLUDED. If a star crossed the pixel during the excluded
    window, the outside stats lose the bright bump: mean drops, std drops.

    Returns (dmean, dstd): per-pixel MAX over window positions of
      dmean = mean_full - mean_outside      (bump flux that lived inside the window)
      dstd  = std_full  - std_outside       (bump's contribution to the variance)
    Prefix sums make every window O(1); window slides one stack at a time.
    """
    t_dim = cube.shape[0]
    c64 = cube.astype(np.float64)
    s1 = np.concatenate([np.zeros((1,) + cube.shape[1:]), np.cumsum(c64, axis=0)])
    s2 = np.concatenate([np.zeros((1,) + cube.shape[1:]),
                         np.cumsum(c64 * c64, axis=0)])
    n_out = t_dim - window
    mean_full = s1[t_dim] / t_dim
    var_full = s2[t_dim] / t_dim - mean_full ** 2
    std_full = np.sqrt(np.clip(var_full, 0, None))

    dmean = np.full(cube.shape[1:], -np.inf, np.float64)
    dstd = np.full(cube.shape[1:], -np.inf, np.float64)
    for t0 in range(0, t_dim - window + 1):
        win1 = s1[t0 + window] - s1[t0]
        win2 = s2[t0 + window] - s2[t0]
        mean_out = (s1[t_dim] - win1) / n_out
        var_out = (s2[t_dim] - win2) / n_out - mean_out ** 2
        np.maximum(dmean, mean_full - mean_out, out=dmean)
        np.maximum(dstd, std_full - np.sqrt(np.clip(var_out, 0, None)), out=dstd)
    return dmean.astype(np.float32), dstd.astype(np.float32)


def blip_maps(cube, sigmas, hp_mult=4.0, sp_hp=0.0, device=None, keep_cube=False):
    """
    Per-pixel temporal blip response (user-designed), FFT edition: subtract each
    pixel's temporal mean, Hann-taper the series (makes the circular FFT convolution
    seam zero-to-zero -- no last-to-first step), then for each ``sigma`` in the width
    ladder multiply the time spectrum by a full-length Gaussian smoother and, when
    ``hp_mult`` > 0, a complementary high-pass 1 - G(hp_mult*sigma) that nulls
    variation slower than ~hp_mult*sigma stacks (clouds). ``sp_hp`` > 0 additionally
    subtracts a sigma-``sp_hp``-px spatial blur from every filtered slice BEFORE the
    max over time (the max is the one nonlinear step, so the pedestal must go first
    or cloudy moments steal pixels' maxima). Returns {sigma: (H, W) ndarray}.

    Processes in row bands so the (T, H, W) intermediates fit on the GPU at 128+
    stacks.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t_dim, h, w = cube.shape
    margin = int(math.ceil(4 * sp_hp)) if sp_hp > 0 else 0
    band = max(64, int(6e8 / (t_dim * w * 4)))
    hann = torch.hann_window(t_dim, periodic=False, device=device)[:, None, None]
    f = torch.fft.rfftfreq(t_dim, device=device)[:, None, None]    # cycles/stack
    if sp_hp > 0:
        tt = torch.arange(-margin, margin + 1, dtype=torch.float32, device=device)
        blur_k = torch.exp(-tt * tt / (2 * sp_hp * sp_hp))
        blur_k = blur_k / blur_k.sum()

    out = {s: np.empty((h, w), np.float32) for s in sigmas}
    cube_out = np.empty_like(cube) if keep_cube else None
    for r0 in range(0, h, band):
        r1 = min(h, r0 + band)
        a, b = max(0, r0 - margin), min(h, r1 + margin)
        x = torch.from_numpy(np.ascontiguousarray(cube[:, a:b, :])).to(device)
        x = x - x.mean(dim=0, keepdim=True)
        x = x * hann
        spec = torch.fft.rfft(x, dim=0)
        del x
        for sigma in sigmas:
            shape_f = torch.exp(-2 * math.pi ** 2 * sigma ** 2 * f * f)
            if hp_mult > 0:
                hp = hp_mult * sigma
                shape_f = shape_f * (1 - torch.exp(-2 * math.pi ** 2 * hp * hp * f * f))
            resp = torch.fft.irfft(spec * shape_f, n=t_dim, dim=0)
            if sp_hp > 0:                       # separable Gaussian, per time slice
                r4 = resp.unsqueeze(1)          # (T, 1, rows, W)
                blur = Fnn.conv2d(r4, blur_k.view(1, 1, -1, 1), padding=(margin, 0))
                blur = Fnn.conv2d(blur, blur_k.view(1, 1, 1, -1), padding=(0, margin))
                resp = resp - blur.squeeze(1)
            if keep_cube:
                cube_out[:, r0:r1] = resp[:, r0 - a:r0 - a + (r1 - r0)].cpu().numpy()
            m = resp.max(dim=0).values
            out[sigma][r0:r1] = m[r0 - a:r0 - a + (r1 - r0)].cpu().numpy()
    return (out, cube_out) if keep_cube else out


# ---------------------------------------------------------------- detection

def velocity_grid(vmax, vstep, vmin):
    """(K, 2) array of (vy, vx) px/stack candidates with vmin <= max(|vy|,|vx|) <= vmax.
    The infinity-norm floor drops the static axis (fixed-pattern residue lives at v=0)."""
    vals = np.arange(-vmax, vmax + 1e-9, vstep)
    vy, vx = np.meshgrid(vals, vals, indexing='ij')
    v = np.stack([vy.ravel(), vx.ravel()], axis=1)
    keep = np.maximum(np.abs(v[:, 0]), np.abs(v[:, 1])) >= vmin
    return v[keep].astype(np.float32)


def detect(cube, chunk=32, stride=16, vmax=2.0, vstep=0.25, vmin=0.4,
           batch=1024, device=None, dc='slice'):
    """
    Run the FFT/autocorrelation velocity matched filter over every chunk position.
    ``cube``: (T, H, W) float32 ndarray (T is the full time depth of every block).
    Returns dict with chunk grid coords, per-chunk scores (N, K), velocity grid (K, 2).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t_dim, h, w = cube.shape
    c = chunk

    cube_t = torch.from_numpy(cube)
    cube_t = cube_t - cube_t.mean(dim=0, keepdim=True)      # static scene/FPN -> gone

    ys = list(range(0, h - c + 1, stride))
    xs = list(range(0, w - c + 1, stride))
    positions = [(y, x) for y in ys for x in xs]

    # Separable Hann taper on every edge of the block.
    wt = torch.hann_window(t_dim, periodic=False)
    ws = torch.hann_window(c, periodic=False)
    w3 = (wt[:, None, None] * ws[None, :, None] * ws[None, None, :]).to(device)

    # Candidate velocities and the line-sample grid for grid_sample: points
    # (tau, vy*tau, vx*tau) for tau = 1..t_dim/2-1, valid while the spatial lag stays
    # inside the un-wrapped +-(c/2-1) range (beyond that the FFT autocorr aliases).
    v = torch.from_numpy(velocity_grid(vmax, vstep, vmin))              # (K, 2) [vy, vx]
    taus = torch.arange(1, t_dim // 2, dtype=torch.float32)             # (Nt,)
    k, nt = v.shape[0], taus.shape[0]
    lag_y = v[:, 0:1] * taus[None, :]                                   # (K, Nt)
    lag_x = v[:, 1:2] * taus[None, :]
    valid = (lag_y.abs() <= c // 2 - 1) & (lag_x.abs() <= c // 2 - 1)   # (K, Nt)
    # normalized grid coords, align_corners=True: index i -> 2*i/(S-1) - 1; origin sits
    # at index S//2 after fftshift (torch.roll).
    def norm(idx, size):
        return idx * 2.0 / (size - 1) - 1.0
    gz = norm(t_dim // 2 + taus, t_dim).expand(k, nt)                   # tau axis
    gy = norm(c // 2 + lag_y, c)
    gx = norm(c // 2 + lag_x, c)
    grid = torch.stack([gx, gy, gz], dim=-1).view(1, k, nt, 1, 3).to(device)
    valid = valid.to(device)
    n_valid = valid.sum(dim=1).clamp_min(1)                             # (K,)

    scores = torch.empty(len(positions), k, dtype=torch.float32)
    t0 = time.monotonic()
    for b0 in range(0, len(positions), batch):
        pos = positions[b0:b0 + batch]
        blocks = torch.stack([cube_t[:, y:y + c, x:x + c] for (y, x) in pos]).to(device)
        if dc == 'slice':
            # per-slice DC: zeroes the kx=ky=0 flicker column (haze brightening the
            # whole block boosts every velocity line equally if left in)
            blocks = blocks - blocks.mean(dim=(2, 3), keepdim=True)
        else:                                                     # one scalar per block
            blocks = blocks - blocks.mean(dim=(1, 2, 3), keepdim=True)
        blocks = blocks * w3
        spec = torch.fft.rfftn(blocks, dim=(1, 2, 3))
        power = spec.real ** 2 + spec.imag ** 2
        ac = torch.fft.irfftn(power, s=(t_dim, c, c), dim=(1, 2, 3))
        a0 = ac[:, 0, 0, 0].clamp_min(1e-20)                      # total windowed energy
        ac = torch.roll(ac, shifts=(t_dim // 2, c // 2, c // 2), dims=(1, 2, 3))
        samp = Fnn.grid_sample(ac.unsqueeze(1), grid.expand(len(pos), -1, -1, -1, -1),
                               mode='bilinear', padding_mode='zeros', align_corners=True)
        line = (samp[:, 0, :, :, 0] * valid).sum(dim=2) / n_valid       # (B, K)
        scores[b0:b0 + len(pos)] = (line / a0[:, None]).cpu()
        done = b0 + len(pos)
        print(f"\r  matched filter {done}/{len(positions)} blocks "
              f"({done / (time.monotonic() - t0):.0f}/s)", end='', flush=True)
    print()
    return {'ys': ys, 'xs': xs, 'positions': positions, 'chunk': c,
            'scores': scores.numpy(), 'v': v.numpy()}


def detect_coherent(cube, vmax=2.0, vstep=0.25, vmin=0.4, psf_sigma=1.3,
                    hp_sigma=16.0, border=80, device=None, dc='slice'):
    """
    Coherent synthetic tracking, full frame: for each candidate velocity, phase-shift
    every stack by -v*t in the Fourier domain and sum -- a star aligned with v adds up
    linearly in *amplitude* (the autocorrelation route is quadratic, so this digs ~3x
    fainter). The Gaussian PSF matched filter and a high-pass that suppresses cloud
    bands (sigma ``hp_sigma`` px) are folded into the same spectrum multiply, so each
    velocity costs 32 complex multiply-adds + one inverse FFT.

    Returns dict: 'v' (K,2), per-velocity robust 'sigma' (K,), and 'cands': for each
    velocity the top pixels (y, x, z) outside ``border`` (FFT shifts wrap; the border
    hides the wrapped smear).
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    t_dim, h, w = cube.shape

    x = torch.from_numpy(cube).to(device)
    x = x - x.mean(dim=0, keepdim=True)                 # static scene/FPN
    if dc == 'slice':
        x = x - x.mean(dim=(1, 2), keepdim=True)        # global flicker
    spec = torch.fft.rfft2(x)                           # (T, H, W//2+1) complex64
    del x

    fy = torch.fft.fftfreq(h, device=device)[:, None]   # cycles/px
    fx = torch.fft.rfftfreq(w, device=device)[None, :]
    f2 = fx * fx + fy * fy
    # PSF matched filter x high-pass (unsharp): G_psf * (1 - G_hp), both Gaussian.
    shape_f = (torch.exp(-2 * math.pi ** 2 * psf_sigma ** 2 * f2)
               * (1 - torch.exp(-2 * math.pi ** 2 * hp_sigma ** 2 * f2)))
    spec = spec * shape_f                               # filter once, before the v loop

    v = velocity_grid(vmax, vstep, vmin)
    ts = torch.arange(t_dim, dtype=torch.float32, device=device) - (t_dim - 1) / 2
    out = {'v': v, 'sigma': np.zeros(len(v), np.float32), 'cands': []}
    t0 = time.monotonic()
    for k, (vy, vx) in enumerate(v):
        # Shift slice t by -v*t (Fourier shift theorem: x(n - s) <-> X(f) e^{-2pi i f s}
        # with s = -v*t) so a source moving at +v stacks coherently; positions land at
        # the source's mid-capture location (ts is centered).
        py = torch.exp(2j * math.pi * fy[None] * (float(vy) * ts)[:, None, None])
        px = torch.exp(2j * math.pi * fx[None] * (float(vx) * ts)[:, None, None])
        acc = (spec * py * px).sum(dim=0)
        img = torch.fft.irfft2(acc, s=(h, w)) / t_dim
        core = img[border:-border, border:-border]
        med = core.median()
        sigma = (core - med).abs().median() * 1.4826
        zmap = (core - med) / sigma.clamp_min(1e-12)
        topv, topi = zmap.flatten().topk(20)
        yy = topi // core.shape[1] + border
        xx = topi % core.shape[1] + border
        out['sigma'][k] = float(sigma)
        out['cands'].append(torch.stack(
            [yy.float(), xx.float(), topv]).T.cpu().numpy())
        print(f"\r  coherent {k + 1}/{len(v)} velocities "
              f"({(k + 1) / (time.monotonic() - t0):.1f}/s)", end='', flush=True)
    print()
    return out


def coherent_detections(res, n_top=15, nms_px=48):
    """Merge per-velocity candidate pixels into a single NMS'd list, best z first."""
    rows = []
    for k, cand in enumerate(res['cands']):
        vy, vx = res['v'][k]
        for y, x, z in cand:
            rows.append((float(z), float(x), float(y), float(vx), float(vy)))
    rows.sort(reverse=True)
    out = []
    for z, x, y, vx, vy in rows:
        if len(out) >= n_top:
            break
        if any((y - d['y']) ** 2 + (x - d['x']) ** 2 < nms_px ** 2 for d in out):
            continue
        out.append({'x': int(x), 'y': int(y), 'z': z, 'vx': vx, 'vy': vy})
    return out


def zscore(scores):
    """Robust per-velocity z across blocks: each velocity's line length (and hence noise
    floor) differs, so normalize each column by its own median/MAD."""
    med = np.median(scores, axis=0, keepdims=True)
    mad = np.median(np.abs(scores - med), axis=0, keepdims=True)
    return (scores - med) / (1.4826 * mad + 1e-20)


def velocity_consensus(res, z, z_floor=3.5, n_top=5):
    """Per-velocity tally of blocks above ``z_floor``. Real stars across the field share
    (nearly) one sidereal drift vector, so a genuine star field spikes one velocity bin
    far beyond the ~Poisson count that noise gives every bin -- a detection statistic
    that digs below the single-block threshold. Returns [(count, max_z, vy, vx), ...]."""
    counts = (z > z_floor).sum(axis=0)
    zmax = z.max(axis=0)
    order = np.argsort(counts)[::-1][:n_top]
    return [(int(counts[k]), float(zmax[k]), float(res['v'][k][0]), float(res['v'][k][1]))
            for k in order]


def top_detections(res, z, n_top=20, nms_px=None):
    """Greedy NMS over block centers: overlapping blocks see the same star; keep the
    strongest, drop neighbors within ``nms_px`` (default: one block). Best first."""
    nms_px = nms_px or res['chunk']
    zmax = z.max(axis=1)
    vbest = z.argmax(axis=1)
    order = np.argsort(zmax)[::-1]
    half = res['chunk'] // 2
    out = []
    for i in order:
        if len(out) >= n_top:
            break
        y, x = res['positions'][i]
        cy, cx = y + half, x + half
        if any((cy - d['y']) ** 2 + (cx - d['x']) ** 2 < nms_px ** 2 for d in out):
            continue
        vy, vx = res['v'][vbest[i]]
        out.append({'x': cx, 'y': cy, 'z': float(zmax[i]),
                    'vx': float(vx), 'vy': float(vy)})
    return out


# ---------------------------------------------------------------- reporting

def save_png(path, cube_mean, res, z, dets, z_floor=5.0):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_axes = 2 if res is not None else 1
    fig, axes = plt.subplots(n_axes, 1, figsize=(14, 7.5 * n_axes), squeeze=False)
    axes = axes[:, 0]
    lo, hi = np.percentile(cube_mean, [1, 99.5])
    axes[0].imshow(cube_mean, cmap='gray', vmin=lo, vmax=hi)
    axes[0].set_title('temporal mean (stretched)')
    for d in dets:
        good = d['z'] >= z_floor
        color = 'lime' if good else 'orange'
        axes[0].add_patch(plt.Circle((d['x'], d['y']), 40, fill=False, color=color))
        axes[0].annotate(f"z={d['z']:.1f}", (d['x'] + 45, d['y']), color=color, fontsize=8)
        axes[0].arrow(d['x'], d['y'], d['vx'] * 30, d['vy'] * 30,
                      color=color, head_width=10)
    if res is not None:
        # Velocity-coded block map: hue = direction of each block's best-matching
        # velocity, saturation = its speed, brightness = its z. A real star field is
        # many bright dots of ONE color (shared sidereal drift); noise is dim random
        # confetti; clouds are a bright single-color region.
        import matplotlib.colors as mcolors
        ny, nx = len(res['ys']), len(res['xs'])
        zmax = z.max(axis=1)
        vbest = res['v'][z.argmax(axis=1)]                    # (N, 2) [vy, vx]
        vspan = np.abs(res['v']).max()
        hue = (np.arctan2(vbest[:, 0], vbest[:, 1]) / (2 * np.pi)) % 1.0
        sat = np.clip(np.hypot(vbest[:, 0], vbest[:, 1]) / vspan, 0, 1)
        val = np.clip((zmax - 2.0) / 6.0, 0, 1)               # z<=2 black, z>=8 full
        rgb = mcolors.hsv_to_rgb(np.stack([hue, sat, val], -1).reshape(ny, nx, 3))
        ext = (res['xs'][0], res['xs'][-1] + res['chunk'],
               res['ys'][-1] + res['chunk'], res['ys'][0])
        axes[1].imshow(rgb, extent=ext, aspect='equal')
        axes[1].set_title('block best velocity: hue=direction, sat=speed, brightness=z '
                          '(z 2..8); wheel = direction legend')
        # hue-wheel legend (same angle convention: image +y is down)
        n = 101
        yy, xx = np.mgrid[-1:1:n * 1j, -1:1:n * 1j]
        rr = np.hypot(yy, xx)
        wheel = mcolors.hsv_to_rgb(np.dstack(
            [(np.arctan2(yy, xx) / (2 * np.pi)) % 1.0, np.clip(rr, 0, 1),
             (rr <= 1.0).astype(float)]))
        axw = fig.add_axes([0.015, 0.03, 0.10, 0.10])
        axw.imshow(wheel)
        axw.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="FFT velocity matched filter for daytime stars")
    p.add_argument('ser', help=".ser capture of (nominally empty) sky, static camera")
    p.add_argument('--stacks', type=int, default=32, help="time depth T of the cube")
    p.add_argument('--chunk', type=int, default=64,
                   help="spatial block size (px); a sidereal track spans ~56 px over a "
                        "60 s capture, so 64 holds the whole track (32 only ever sees 1/3)")
    p.add_argument('--stride', type=int, default=None,
                   help="block overlap stride (px, default chunk/2)")
    p.add_argument('--vmax', type=float, default=2.0, help="max |v| px/stack")
    p.add_argument('--vstep', type=float, default=0.15, help="velocity grid step px/stack")
    p.add_argument('--vmin', type=float, default=0.15,
                   help="min max(|vy|,|vx|) px/stack (excludes the static axis; keep low "
                        "enough for near-pole drift, which crawls at <0.3 px/stack)")
    p.add_argument('--batch', type=int, default=0,
                   help="blocks per GPU batch (0 = auto by chunk size)")
    p.add_argument('--top', type=int, default=15, help="detections to report")
    p.add_argument('--arcsec-per-px', type=float, default=0.0,
                   help="plate scale, for arcsec/s in the report (0 = omit)")
    p.add_argument('--cache-dir', default=None, help="stack cache dir")
    p.add_argument('--png', default=None,
                   help="output figure (default: <cache>/<ser>.daystars.png)")
    p.add_argument('--inject', action='append', default=[],
                   metavar='x,y,vx,vy,peak',
                   help="add a synthetic drifter (px, px/stack, normalized peak); repeatable")
    p.add_argument('--inject-sigma', type=float, default=1.3, help="injected PSF sigma px")
    p.add_argument('--blip', action='store_true',
                   help="per-pixel temporal Gaussian blip response (FFT, Hann-tapered), "
                        "max over time -> PNG panel per width")
    p.add_argument('--blip-sigmas', default='1,2,4,8',
                   help="blip: comma list of Gaussian widths in stacks (transit widths)")
    p.add_argument('--blip-hp', type=float, default=4.0,
                   help="blip: high-pass at this multiple of each sigma (0 = off)")
    p.add_argument('--blip-sphp', type=float, default=12.0,
                   help="blip: spatial high-pass sigma in px, applied per filtered "
                        "slice before the max (0 = off)")
    p.add_argument('--transit', action='store_true',
                   help="excluded-window per-pixel transit maps (mean/std drop); "
                        "writes the two images and exits")
    p.add_argument('--transit-window', type=int, default=8,
                   help="excluded-window length in stacks")
    p.add_argument('--coherent', action='store_true',
                   help="full-frame coherent synthetic tracking instead of the "
                        "block-FFT autocorrelation (linear in amplitude: ~3x fainter)")
    p.add_argument('--hp-sigma', type=float, default=16.0,
                   help="coherent mode: high-pass sigma px (suppresses cloud bands)")
    p.add_argument('--pre', choices=('mean', 'whiten', 'blip'), default='mean',
                   help="preprocessing: 'mean' = subtract per-pixel temporal mean; "
                        "'whiten' = also divide by the per-pixel temporal std; "
                        "'blip' = feed the blip-filtered cube (first --blip-sigmas "
                        "width, temporal band-pass + spatial HP) to the detector")
    p.add_argument('--dc', choices=('slice', 'scalar'), default='slice',
                   help="mean removal: 'slice' = per time slice (zeroes the flicker "
                        "DC column), 'scalar' = one mean per block/cube")
    args = p.parse_args(argv)

    cube, meta = stack_ser(args.ser, args.stacks, args.cache_dir)
    dt = meta.get('stack_dt_s')
    print(f"cube {cube.shape[0]}x{cube.shape[1]}x{cube.shape[2]}, "
          f"{meta['frames_per_stack']} frames/stack"
          + (f", {dt:.2f} s/stack" if dt else ""))

    for spec in args.inject:
        x, y, vx, vy, peak = (float(s) for s in spec.split(','))
        inject_star(cube, x, y, vx, vy, peak, args.inject_sigma)
        print(f"injected drifter at ({x:.0f},{y:.0f}) v=({vx},{vy}) px/stack peak={peak}")

    if args.pre == 'blip':
        sigma0 = float(args.blip_sigmas.split(',')[0])
        cube_view = cube.mean(axis=0)          # keep the raw mean for the figure
        _, cube = blip_maps(cube, [sigma0], args.blip_hp, args.blip_sphp,
                            keep_cube=True)
        print(f"blip-filtered cube (sigma {sigma0:g}, hp {args.blip_hp:g}x, "
              f"spatial hp {args.blip_sphp:g}px)")
    elif args.pre == 'whiten':
        # Per-pixel z against the pixel's OWN whole-capture statistics (the offline
        # version of the tracker's realtime EMA surprise): subtract the actual temporal
        # mean, divide by the actual temporal std. Flattens the vignette's noise
        # gradient so no region dominates the detection statistic.
        cube_view = cube.mean(axis=0)          # keep the raw mean for the figure
        std = cube.std(axis=0, keepdims=True)
        cube = (cube - cube.mean(axis=0, keepdims=True)) / (std + 1e-10)
        print("whitened (per-pixel whole-capture mean/std)")
    else:
        cube_view = None

    png = args.png or os.path.join(
        args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.ser)),
                                       'daystars_cache'),
        os.path.basename(args.ser) + '.daystars.png')

    if args.blip:
        sigmas = [float(s) for s in args.blip_sigmas.split(',')]
        maps = blip_maps(cube, sigmas, args.blip_hp, args.blip_sphp)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        n = len(sigmas)
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(22, 6.5 * rows), squeeze=False)
        for i, sigma in enumerate(sigmas):
            ax = axes[i // 2][i % 2]
            img = maps[sigma]
            lo, hi = np.percentile(img, [50, 99.9])
            ax.imshow(img, cmap='inferno', vmin=lo, vmax=hi)
            hp = f', hp={args.blip_hp:g}x' if args.blip_hp else ''
            ax.set_title(f'sigma={sigma:g} stacks{hp} (p50..p99.9)')
        for i in range(n, rows * 2):
            axes[i // 2][i % 2].axis('off')
        fig.suptitle('max-over-time blip response (FFT, Hann-tapered)')
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"wrote {png}")
        return

    if args.transit:
        dmean, dstd = transit_maps(cube, args.transit_window)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(14, 15))
        for ax, img, title in [(axes[0], dmean, 'mean drop when window excluded'),
                               (axes[1], dstd, 'std drop when window excluded')]:
            lo, hi = np.percentile(img, [50, 99.9])
            im = ax.imshow(img, cmap='inferno', vmin=lo, vmax=hi)
            ax.set_title(f'{title} (w={args.transit_window} stacks; stretch p50..p99.9)')
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"wrote {png}")
        return

    if args.coherent:
        res = detect_coherent(cube, args.vmax, args.vstep, args.vmin,
                              args.inject_sigma, args.hp_sigma, dc=args.dc)
        dets = coherent_detections(res, args.top)
        print(f"\ntop {len(dets)} (coherent; positions are mid-capture):")
        for d in dets:
            line = (f"  ({d['x']:4d},{d['y']:4d})  z={d['z']:6.1f}  "
                    f"v=({d['vx']:+.2f},{d['vy']:+.2f}) px/stack")
            if dt:
                pxs = math.hypot(d['vx'], d['vy']) / dt
                line += f"  |v|={pxs:.2f} px/s"
                if args.arcsec_per_px:
                    line += f" = {pxs * args.arcsec_per_px:.1f} arcsec/s"
            print(line)
        save_png(png, cube_view if cube_view is not None else cube.mean(axis=0),
                 None, None, dets)
        print(f"\nwrote {png}")
        return

    stride = args.stride or args.chunk // 2
    batch = args.batch or max(64, 1024 * 32 * 32 // (args.chunk * args.chunk))
    res = detect(cube, args.chunk, stride, args.vmax, args.vstep, args.vmin, batch,
                 dc=args.dc)
    z = zscore(res['scores'])
    dets = top_detections(res, z, args.top)

    print(f"\ntop {len(dets)} (block centers, +-{res['chunk'] // 2} px):")
    for d in dets:
        line = (f"  ({d['x']:4d},{d['y']:4d})  z={d['z']:6.1f}  "
                f"v=({d['vx']:+.2f},{d['vy']:+.2f}) px/stack")
        if dt:
            pxs = math.hypot(d['vx'], d['vy']) / dt
            line += f"  |v|={pxs:.2f} px/s"
            if args.arcsec_per_px:
                line += f" = {pxs * args.arcsec_per_px:.1f} arcsec/s"
        print(line)

    print("\nvelocity consensus (blocks with z>3.5 per velocity; noise ~ flat, "
          "a star field spikes one bin):")
    for count, zmx, vy, vx in velocity_consensus(res, z):
        line = f"  v=({vx:+.2f},{vy:+.2f}) px/stack  blocks={count:4d}  max z={zmx:5.1f}"
        if dt and args.arcsec_per_px:
            line += f"  |v|={math.hypot(vx, vy) / dt * args.arcsec_per_px:.1f} arcsec/s"
        print(line)

    save_png(png, cube_view if cube_view is not None else cube.mean(axis=0),
             res, z, dets)
    print(f"\nwrote {png}")


if __name__ == '__main__':
    main()
