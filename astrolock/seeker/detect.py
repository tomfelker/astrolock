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


def detect_blobs(bp, work, prev_bp, *, threshold_rel, max_candidates, suppress_radius,
                 min_blob_px, max_size_px, psf_px, snr=0.0, min_roundness=0.0, moving_frac, scale):
    """
    Greedy peak detection on the band-passed image ``bp`` (pointlike features), with non-max
    suppression. ``work`` is the original grayscale (for the absolute brightness score) and
    ``prev_bp`` is the previous band-passed frame (for the "moving" flag).

    Each blob gets a sub-pixel center, a size, a brightness ``score``, a ``pointlike`` score
    (1 = PSF-sized, →0 = extended), and a ``roundness`` (1 = circular, →0 = a line/edge, from
    the ratio of the second-moment eigenvalues). Peaks must clear ``snr`` sigma above the
    band-passed background (robust MAD); an optional ``threshold_rel`` relative floor and the
    ``max_size_px`` / ``min_roundness`` cuts reject extended clutter and edges/streaks.
    """
    bp = torch.as_tensor(bp, dtype=torch.float32)
    work = torch.as_tensor(work, dtype=torch.float32)
    blobs = []
    m = float(bp.max())
    if m <= 0:
        return blobs

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

    scratch = bp.clone()
    h, w = bp.shape
    r = suppress_radius
    found = 0
    for _ in range(max_candidates * 4):     # allow extra tries since some get rejected
        if found >= max_candidates:
            break
        flat_idx = int(torch.argmax(scratch))
        y, x = divmod(flat_idx, w)
        peak = float(scratch[y, x])
        if peak < thresh:
            break
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        win = bp[y0:y1, x0:x1]
        n_above = int((win >= 0.5 * peak).sum())
        scratch[y0:y1, x0:x1] = 0.0          # suppress this neighborhood regardless
        if n_above < min_blob_px:
            continue

        ys, xs = torch.meshgrid(torch.arange(y0, y1, dtype=torch.float32),
                                torch.arange(x0, x1, dtype=torch.float32), indexing='ij')
        wsub = torch.clamp(win - 0.5 * peak, min=0.0)
        tot = float(wsub.sum()) or 1.0
        cx = float((xs * wsub).sum()) / tot
        cy = float((ys * wsub).sum()) / tot

        # Second-moment eigenvalues -> roundness (minor/major axis variance ratio).
        dx, dy = xs - cx, ys - cy
        ixx = float((wsub * dx * dx).sum()) / tot
        iyy = float((wsub * dy * dy).sum()) / tot
        ixy = float((wsub * dx * dy).sum()) / tot
        tr = ixx + iyy
        s = math.sqrt(max(0.0, (tr / 2.0) ** 2 - (ixx * iyy - ixy * ixy)))
        l1, l2 = tr / 2.0 + s, tr / 2.0 - s
        roundness = (l2 / l1) if l1 > 1e-6 else 1.0

        size_px = math.sqrt(n_above / math.pi) * 2.0
        pointlike = min(1.0, psf_px / max(size_px, psf_px))
        if max_size_px and size_px > max_size_px:
            continue                          # too extended (rooftop)
        if min_roundness and roundness < min_roundness:
            continue                          # too elongated (edge / streak / wire)

        moving = bool(diff[y, x] > moving_frac * peak) if diff is not None else None
        blobs.append({
            'id': found,
            'px': [round(cx, 2), round(cy, 2)],      # [x, y] in the work image
            'score': round(float(work[y, x]) / scale, 4),  # absolute brightness 0..1
            'size_px': round(size_px, 1),
            'pointlike': round(pointlike, 3),
            'roundness': round(roundness, 3),
            'moving': moving,
        })
        found += 1
    return blobs


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
    p.add_argument('--bg-radius', type=int, default=12,
                   help="local-background blur radius (px); larger = pass bigger features")
    p.add_argument('--max-candidates', type=int, default=16)
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
        bp = band_pass(work, args.bg_radius)
        blobs = detect_blobs(
            bp, work, prev,
            threshold_rel=args.threshold, max_candidates=args.max_candidates,
            suppress_radius=args.suppress_radius, min_blob_px=args.min_blob_px,
            max_size_px=args.max_size_px, psf_px=args.psf_px,
            snr=args.snr, min_roundness=args.min_roundness,
            moving_frac=args.moving_frac, scale=scale)
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
