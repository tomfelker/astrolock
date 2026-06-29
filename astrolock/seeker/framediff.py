"""
Quick proof-of-concept: write a new SER whose frames are the magnitude of the difference
between successive frames of an input SER. With a static mount this cancels everything fixed
(stars, door frame, blown-out tree) and leaves only what moved -- handy for spotting dim
movers (satellites/aircraft) the single-frame detector misses.

    python -m astrolock.seeker.framediff in.ser out.ser [--scale S] [--ema A] [--max-frames N]

Default: output frame i = |frame[i+1] - frame[i]| * scale (pairwise). With --ema A (0<A<1) it
instead subtracts a running EMA *background* (`bg += A*(frame-bg)`): output = |frame - bg|. The
EMA background is a low-noise average of the static scene, so a slow/dim mover keeps its full
signal (no self-cancellation) against only single-frame noise -- much better than pairwise for
slow movers. Same dimensions / color id / depth as the input (so the GUI can replay it).
"""

import argparse
import time

import torch

from astrolock.seeker import ser as ser_mod
from astrolock.seeker.sidecar import JsonlWriter


def main(argv=None):
    p = argparse.ArgumentParser(description="SER successive-frame difference magnitude")
    p.add_argument('input')
    p.add_argument('output')
    p.add_argument('--scale', type=float, default=1.0, help="amplify the difference (dim movers)")
    p.add_argument('--ema', type=float, default=0.0,
                   help="if >0, subtract an EMA background with this decay instead of pairwise diff")
    p.add_argument('--max-frames', type=int, default=0, help="limit input frames read (0 = all)")
    p.add_argument('--device', default='cpu', help="torch device (cpu / cuda)")
    args = p.parse_args(argv)
    device = torch.device(args.device)

    reader = ser_mod.SerReader(args.input)
    h = reader.header
    n = reader.frames_on_disk()
    if args.max_frames:
        n = min(n, args.max_frames)
    if n < 2:
        raise SystemExit(f"need >= 2 frames, have {n}")

    writer = ser_mod.SerWriter(args.output, h.image_width, h.image_height,
                               color_id=h.color_id, pixel_depth_per_plane=h.pixel_depth_per_plane)
    sidecar = JsonlWriter(args.output[:-len('.ser')] + '.frames.jsonl')   # so the GUI can replay it
    hi = 65535.0

    def load(i):                                          # int32 ingest (torch has no uint16) -> float
        return torch.from_numpy(reader.read_frame(i).astype('int32')).to(device).float()

    ref = load(0)                                         # previous frame, or the EMA background
    for i in range(1, n):
        cur = load(i)
        diff = ((cur - ref).abs() * args.scale).clamp(0, hi).to(torch.int32)
        writer.write_frame(diff.cpu().numpy().astype('uint16'))
        sidecar.append({'t_mono_ns': time.perf_counter_ns(), 'important': True})
        if args.ema > 0.0:
            ref += args.ema * (cur - ref)                # EMA background (low-noise static scene)
        else:
            ref = cur                                     # pairwise: just the previous frame
        if i % 50 == 0:
            print(f"  {i}/{n}", flush=True)
    writer.close()
    sidecar.close()
    reader.close()
    print(f"wrote {n - 1} diff frames -> {args.output}", flush=True)


if __name__ == '__main__':
    main()
