"""
astrolock_seeker_recorder: read a live stream, archive it to disk. That's all.

The cam is a pure sensor->shm streamer (see framestream); recording is THIS separate process,
started by the backend's record command. Recording starts AT THE CLICK -- the ring's history
is deliberately not archived (chasing a deep backlog risks never catching up to real time).
From there it drains every frame in order and writes plain .ser archives at whatever pace the
drive allows, straight from the shared memory (the file write() reads the ring slot in place
-- zero copies). The ring is the write-behind buffer: a drive stall deepens our lag instead of
dropping frames, up to the ring's whole capacity; beyond that the writer laps us and the loss
is counted explicitly (and a frame overwritten mid-write is detected and un-written).

NOBODY EVER KILLS US. Stop signals:
  * the stream's ended flag (clean shutdown): drain everything committed, finalize, exit;
  * a line -- or EOF -- on STDIN (the backend holds our pipe: 'stop recording' is a line, and a
    crashed backend closes the pipe): one final drain pass of what's already committed,
    finalize, exit.

Output goes OUTSIDE the session's live-IPC namespace (default recordings/), one .ser per
geometry, each named for its FIRST frame's capture time; the SER trailer carries per-frame
CAPTURE times (from the live records), so true cadence survives late writes. Never deleted by
anything.

    python -m astrolock.seeker.recorder --session sessions/<ts> --role main
"""

import argparse
import datetime as dt
import os
import sys
import threading
import time

from astrolock.seeker import framestream, ser as ser_mod


def _utc_of(rec):
    ns = rec.get('t_utc_ns') or 0
    return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc) if ns else None


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker recorder (live stream -> disk .ser)")
    p.add_argument('--session', required=True, help="session directory of the live stream")
    p.add_argument('--role', default='main', help="stream to record (guide/main)")
    p.add_argument('--out-dir', default='recordings', help="archive directory (never cleaned up)")
    p.add_argument('--poll', type=float, default=0.02)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    stop = threading.Event()

    def _stdin_watch():                       # a line = stop recording; EOF = backend gone: stop too
        try:
            sys.stdin.readline()
        except Exception:
            pass
        stop.set()
    threading.Thread(target=_stdin_watch, daemon=True).start()

    fo = framestream.StreamFollower(args.session, args.role)
    fo.skip_to_latest()                       # record from the CLICK; history is not backlog
    base_lost = fo.lost                       # anything before this moment isn't "skipped recording"

    out = None
    out_geom = None
    out_path = None
    written = skipped = 0

    def skipped_now():
        return skipped + max(0, fo.lost - base_lost)

    def archive_one(rd, i):
        """Write frame i straight from the ring; True if it landed, False if lapped."""
        nonlocal out, out_geom, out_path, written, skipped
        wrote = False
        try:
            with rd.view(i) as (rec, frame):           # ZERO-COPY: write() reads the slot in place
                if out is None or out_geom != frame.shape:
                    if out is not None:
                        out.close()
                    out_geom = frame.shape
                    # Every output file is named for its FIRST frame's capture time (a
                    # geometry change just starts the next file the same way).
                    t0 = _utc_of(rec) or dt.datetime.now(dt.timezone.utc)
                    stamp = t0.strftime('%Y%m%dT%H%M%S') + f"{t0.microsecond // 1000:03d}Z"
                    out_path = os.path.join(args.out_dir, f"{stamp}_{args.role}.ser")
                    print(f"[rec:{args.role}] -> {out_path}", flush=True)
                    out = ser_mod.SerWriter(out_path, frame.shape[1], frame.shape[0],
                                            color_id=rd.header.color_id,
                                            pixel_depth_per_plane=rd.header.pixel_depth_per_plane)
                out.write_frame(frame, t_utc=_utc_of(rec))   # trailer gets CAPTURE time
                wrote = True
        except framestream.Lapped:
            if wrote:
                out.truncate_last_frame()              # the slot was reused mid-write: torn bytes
            skipped += 1
            return False
        written += 1
        return True

    try:
        while True:
            fo.poll()
            worked = False
            for rd, i in fo.drain(limit=256):
                worked = archive_one(rd, i) or worked
            if stop.is_set():
                for rd, i in fo.drain():               # freeze: flush what's already committed
                    archive_one(rd, i)
                break
            if not worked and fo.ended():
                break
            if not worked:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not None:
            out.close()
        fo.close()
        print(f"[rec:{args.role}] done: {written} frames written, {skipped_now()} skipped"
              + (f" -> {out_path}" if out_path else " (nothing arrived)"), flush=True)


if __name__ == '__main__':
    main()
