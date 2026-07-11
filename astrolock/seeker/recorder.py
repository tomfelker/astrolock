"""
astrolock_seeker_recorder: read a live shm stream, archive it to disk. That's all.

The cam is a pure sensor->shm streamer (see shmser/framestream); recording is THIS separate
process, started/stopped by the backend's 'record' command. It follows one role's stream like
any other consumer, but reads EVERY frame (sequentially, unlike detect's skip-to-newest) and
writes a plain .ser at whatever pace the drive allows. The RAM write-behind buffer is the shm
architecture itself: the sections we still hold handles on ARE the backlog (kernel-refcounted),
so a drive stall just deepens our lag instead of dropping frames -- the ASICap trick, for free.
If the lag exceeds --max-lag-frames we skip forward and say how much we dropped.

Output goes OUTSIDE the session's live-IPC namespace (default recordings/), with a friendly
name, and no sidecar: nothing live ever reads it. Posterity needs only the pixels + the SER
timestamp trailer, which we stamp with each frame's CAPTURE time (from the live sidecar), so
the trailer records true cadence even when the disk writes late. Never deleted by anything.

    python -m astrolock.seeker.recorder --session sessions/<ts> --role main
"""

import argparse
import datetime as dt
import os
import time

from astrolock.seeker import framestream, ser as ser_mod, sidecar
from astrolock.seeker.sidecar import JsonlTailer


def _ready(ser_path):
    """A segment is joinable once its sidecar has a committed line (before that its store --
    disk or shm -- is unknown; a cam settings-change roll creates the sidecar a beat early)."""
    return sidecar.count_complete_lines(framestream.frames_path_of(ser_path)) >= 1


def _parse_utc(s):
    try:
        return dt.datetime.fromisoformat(s.replace('Z', '+00:00')) if s else None
    except ValueError:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker recorder (shm stream -> disk .ser)")
    p.add_argument('--session', required=True, help="session directory of the live stream")
    p.add_argument('--role', default='main', help="stream to record (guide/main)")
    p.add_argument('--out-dir', default='recordings', help="archive directory (never cleaned up)")
    p.add_argument('--stop-file', default=None, help="finish draining + finalize when this appears")
    p.add_argument('--max-lag-segments', type=int, default=4,
                   help="max SEGMENTS of backlog held open (held shm sections are the RAM "
                        "write-behind buffer; ~cap*frame_size RAM each). Exceeded -> drop the "
                        "oldest held segment (skip forward) with logged accounting")
    p.add_argument('--back-segments', type=int, default=1,
                   help="start this many segments BEFORE the newest (retroactive record: the "
                        "recent past is still alive in retained shm sections)")
    p.add_argument('--poll', type=float, default=0.02)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Start a little in the past: retained sections mean the frames just before the click still exist.
    segs = [s for s in framestream.segment_paths(args.session, args.role) if _ready(s)]
    while not segs:
        if args.stop_file and os.path.exists(args.stop_file):
            return
        time.sleep(args.poll)
        segs = [s for s in framestream.segment_paths(args.session, args.role) if _ready(s)]
    start_seg = segs[max(0, len(segs) - 1 - max(0, args.back_segments))]

    out = None                    # SerWriter, (re)opened on geometry change
    out_geom = None
    out_path = None
    written = skipped = 0

    # ATTACH EAGERLY, drain in order. Holding a segment's reader is what keeps its section
    # alive after the cam rolls on -- the held queue IS the write-behind buffer, so discovery
    # must attach the moment a segment is ready, not when we get around to draining it.
    seen = set()
    q = []                        # held segments, oldest first: dicts of path/reader/tail/recs/i

    def lines_of(pth):
        return sidecar.count_complete_lines(framestream.frames_path_of(pth))

    def discover():
        nonlocal skipped
        for pth in framestream.segment_paths(args.session, args.role):
            if pth < start_seg or pth in seen or not _ready(pth):
                continue
            seen.add(pth)
            try:
                q.append({'path': pth, 'r': framestream.open_reader(pth),
                          'tail': JsonlTailer(framestream.frames_path_of(pth)), 'recs': [], 'i': 0})
            except (ValueError, FileNotFoundError):       # section already gone before we attached
                lost = lines_of(pth)
                skipped += lost
                print(f"[rec:{args.role}] segment gone before attach; skipped {lost} "
                      f"({skipped} total)", flush=True)

    def drop(seg, reason):
        nonlocal skipped
        lost = max(0, lines_of(seg['path']) - seg['i'])
        skipped += lost
        seg['r'].close(); seg['tail'].close()
        if lost:                                          # dropping a fully-drained segment is routine
            print(f"[rec:{args.role}] {reason}; skipped {lost} ({skipped} total)", flush=True)

    stopping = False
    try:
        while True:
            if not stopping and args.stop_file and os.path.exists(args.stop_file):
                stopping = True   # drain what's already HELD, then exit -- and stop discovering:
            if not stopping:      # against a live stream we'd otherwise never catch up and never exit
                discover()
            while len(q) > max(1, args.max_lag_segments):  # lag bound: skip forward, keep the newest
                drop(q.pop(0), f"behind by >{args.max_lag_segments} segments")
            if not q:
                if stopping:
                    break
                time.sleep(args.poll)
                continue
            seg = q[0]
            seg['recs'].extend(seg['tail'].poll())
            avail = min(len(seg['recs']), seg['r'].frames_on_disk())
            wrote = False
            while seg['i'] < avail:
                try:
                    frame = seg['r'].read_frame(seg['i'])
                except (IndexError, ValueError):
                    break                                  # not readable (shouldn't happen: we hold it)
                t_utc = _parse_utc(seg['recs'][seg['i']].get('t_utc')) if seg['i'] < len(seg['recs']) else None
                if out is None or out_geom != frame.shape:
                    if out is not None:
                        out.close()
                    out_geom = frame.shape
                    # Every output file is named for its FIRST frame's capture time (a geometry
                    # change -- ROI/binning relaunch -- just starts the next file the same way).
                    t0 = t_utc or dt.datetime.now(dt.timezone.utc)
                    stamp = t0.strftime('%Y%m%dT%H%M%S') + f"{t0.microsecond // 1000:03d}Z"
                    out_path = os.path.join(args.out_dir, f"{stamp}_{args.role}.ser")
                    print(f"[rec:{args.role}] -> {out_path}", flush=True)
                    out = ser_mod.SerWriter(out_path, frame.shape[1], frame.shape[0],
                                            color_id=seg['r'].header.color_id,
                                            pixel_depth_per_plane=seg['r'].header.pixel_depth_per_plane)
                out.write_frame(frame, t_utc=t_utc)        # trailer gets CAPTURE time, not write time
                written += 1
                seg['i'] += 1
                wrote = True
            if seg['i'] >= avail and seg['r'].finalized() and seg['i'] >= seg['r'].frames_on_disk():
                if len(q) > 1 or stopping or _ready_newer(args, seg['path']):
                    seg['r'].close(); seg['tail'].close()  # fully drained -> release the section
                    q.pop(0)
                    continue
            if seg['i'] >= avail and stopping and len(q) == 1:
                break                                      # drained everything committed so far
            if not wrote:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not None:
            out.close()
        for seg in q:
            seg['r'].close(); seg['tail'].close()
        print(f"[rec:{args.role}] done: {written} frames written, {skipped} skipped -> {out_path}",
              flush=True)


def _ready_newer(args, cur):
    return any(s > cur and _ready(s) for s in framestream.segment_paths(args.session, args.role))


if __name__ == '__main__':
    main()
