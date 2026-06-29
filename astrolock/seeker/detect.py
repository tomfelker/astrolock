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


def work_image(frame, color_id):
    """The grayscale image we analyze: Bayer -> sensitive half-res mono sum; else as-is."""
    if bayer.is_bayer(color_id):
        return torch.as_tensor(bayer.to_mono_sum(frame), dtype=torch.float32)
    # frame is a read-only buffer view from the SER reader; astype gives a writable copy.
    return torch.from_numpy(frame.astype('float32'))


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


def detection_surface(work, *, detector, bg_radius, psf_px, doh_sigma):
    """The 2-D map detect_blobs picks peaks from: band-pass (default) or determinant-of-Hessian."""
    if detector == 'doh':
        sigma = doh_sigma if doh_sigma > 0 else psf_px
        return det_of_hessian(work, sigma)
    return band_pass(work, bg_radius)


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


def _frame_count(ser_path):
    with open(ser_path, 'rb') as f:
        return ser_mod.unpack_header(f.read(ser_mod.HEADER_SIZE)).frame_count


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
    p.add_argument('--detector', default='doh', choices=['bandpass', 'doh'],
                   help="detection surface: 'doh' (default) = determinant of the Hessian "
                        "(Gaussian-derivative blob detector; rejects edges/lines by construction), "
                        "or 'bandpass' (the older local-background subtraction)")
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
    args = p.parse_args(argv)

    # Wait for the first segment to exist.
    while not _segments(args.session, args.role):
        if args.stop_file and os.path.exists(args.stop_file):
            return
        time.sleep(args.poll)

    # Live tracks the newest segment; offline starts at the oldest and processes in order.
    segs = _segments(args.session, args.role)
    cur = segs[-1] if args.follow else segs[0]

    def open_segment(ser_path):
        reader = ser_mod.SerReader(ser_path)
        writer = JsonlWriter(ser_path[:-len('.ser')] + '.detections.jsonl')
        print(f"[detect:{args.role}] {os.path.basename(ser_path)}", flush=True)
        return reader, writer

    reader, writer = open_segment(cur)
    prev = None
    next_index = 0
    scale = None
    total = 0

    def process(i):
        nonlocal prev, scale, total
        frame = reader.read_frame(i)
        cid = reader.header.color_id
        if scale is None:
            scale = full_scale(cid, reader.header.pixel_depth_per_plane)
        work = work_image(frame, cid)
        bp = detection_surface(work, detector=args.detector, bg_radius=args.bg_radius,
                               psf_px=args.psf_px, doh_sigma=args.doh_sigma)
        blobs = detect_blobs(
            bp, work, prev,
            threshold_rel=args.threshold, max_candidates=args.max_candidates,
            suppress_radius=args.suppress_radius, min_blob_px=args.min_blob_px,
            max_size_px=args.max_size_px, psf_px=args.psf_px,
            snr=args.snr, min_roundness=args.min_roundness,
            moving_frac=args.moving_frac, scale=scale,
            tile_grid=args.tile_grid, per_tile=args.per_tile)
        # Report blobs in the frame's image space. We may analyse a downsampled grid (Bayer ->
        # half-res mono sum), so scale coords back up; consumers then need no idea how we work.
        coord_scale = reader.header.image_width / work.shape[1]
        if coord_scale != 1:
            for b in blobs:
                b['px'] = [b['px'][0] * coord_scale, b['px'][1] * coord_scale]
                if 'size_px' in b:
                    b['size_px'] = b['size_px'] * coord_scale
        writer.append({'index': i, 't_mono_ns': time.perf_counter_ns(), 'blobs': blobs})
        prev = bp
        total += 1

    try:
        while True:
            if args.stop_file and os.path.exists(args.stop_file):
                break

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

            # Caught up on the current segment. Roll if a newer one exists.
            newer = [s for s in _segments(args.session, args.role) if s > cur]
            if newer:
                reader.close()
                writer.close()
                cur = newer[-1] if args.follow else newer[0]   # live: jump to newest
                reader, writer = open_segment(cur)
                prev = None
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
        print(f"[detect:{args.role}] processed {total} frames", flush=True)


def full_scale(color_id, pixel_depth):
    """Max possible value of work_image, for an absolute 0..1 brightness score."""
    base = ser_mod.container_max(pixel_depth)
    return base * (4 if bayer.is_bayer(color_id) else 1)


if __name__ == '__main__':
    main()
