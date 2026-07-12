"""
astrolock_seeker_recorder: read a live stream, archive it to disk. That's all.

The cam is a pure sensor->shm streamer (see framestream); recording is THIS separate process,
started by the backend's record command. It follows the stream chain like any consumer, reads
EVERY frame in order, and writes plain .ser archives at whatever pace the drive allows. The RAM
write-behind buffer is the architecture itself: segments we hold open ARE the backlog
(kernel-refcounted sections), so a drive stall deepens our lag instead of dropping frames --
the ASICap trick, for free -- bounded by --max-lag-segments (skip forward, with accounting).
Recording starts as far back as sections are still alive (~the previous segment), so the
moments before the click make it in.

NOBODY EVER KILLS US, and stopping never drops queued frames. Stop signals:
  * the stream's 'ended' record (clean shutdown): drain everything committed, finalize, exit;
  * a line -- or EOF -- on STDIN (the backend holds our pipe: 'stop recording' is a line, and a
    crashed backend closes the pipe): FREEZE the queue (stop tailing the head/sidecars), flush
    every frame already committed in the segments we hold, finalize, exit.

Output goes OUTSIDE the session's live-IPC namespace (default recordings/), one .ser per
geometry, each named for its FIRST frame's capture time; no sidecar -- the SER trailer carries
per-frame CAPTURE times (from the live records), so true cadence survives late writes. Never
deleted by anything.

    python -m astrolock.seeker.recorder --session sessions/<ts> --role main
"""

import argparse
import datetime as dt
import os
import sys
import threading
import time

from astrolock.seeker import framestream, ser as ser_mod


def _parse_utc(s):
    try:
        return dt.datetime.fromisoformat(s.replace('Z', '+00:00')) if s else None
    except ValueError:
        return None


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker recorder (live stream -> disk .ser)")
    p.add_argument('--session', required=True, help="session directory of the live stream")
    p.add_argument('--role', default='main', help="stream to record (guide/main)")
    p.add_argument('--out-dir', default='recordings', help="archive directory (never cleaned up)")
    p.add_argument('--max-lag-segments', type=int, default=4,
                   help="max segments of backlog held open (each ~cap x frame-size of RAM); "
                        "exceeded -> drop the oldest (skip forward) with logged accounting")
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

    fo = framestream.StreamFollower(args.session, args.role, keep_all=True)
    fo.poll()
    base_lost = fo.lost                       # dead-before-we-existed history isn't "skipped recording"

    out = None
    out_geom = None
    out_path = None
    written = skipped = 0

    def skipped_now():
        return skipped + max(0, fo.lost - base_lost)

    try:
        while True:
            if not stop.is_set():
                fo.poll()                     # stop = FREEZE the queue: no new segments or commits
            while len(fo.segs) > max(1, args.max_lag_segments):   # lag bound: keep the newest
                seg = fo.segs[0]
                lost = max(0, seg.committed() - getattr(seg, '_rec_drained', 0))
                skipped += lost
                fo.release(seg)
                print(f"[rec:{args.role}] behind by >{args.max_lag_segments} segments; "
                      f"skipped {lost} ({skipped_now()} total)", flush=True)
            worked = False
            seg = fo.segs[0] if fo.segs else None
            if seg is not None:
                drained = getattr(seg, '_rec_drained', 0)
                while drained < seg.committed():
                    frame = seg.read(drained)
                    rec = seg.recs[drained]
                    t_utc = _parse_utc(rec.get('t_utc'))
                    if out is None or out_geom != frame.shape:
                        if out is not None:
                            out.close()
                        out_geom = frame.shape
                        # Every output file is named for its FIRST frame's capture time (a
                        # geometry change just starts the next file the same way).
                        t0 = t_utc or dt.datetime.now(dt.timezone.utc)
                        stamp = t0.strftime('%Y%m%dT%H%M%S') + f"{t0.microsecond // 1000:03d}Z"
                        out_path = os.path.join(args.out_dir, f"{stamp}_{args.role}.ser")
                        print(f"[rec:{args.role}] -> {out_path}", flush=True)
                        out = ser_mod.SerWriter(out_path, frame.shape[1], frame.shape[0],
                                                color_id=seg.header.color_id,
                                                pixel_depth_per_plane=seg.header.pixel_depth_per_plane)
                    out.write_frame(frame, t_utc=t_utc)   # trailer gets CAPTURE time, not write time
                    written += 1
                    drained += 1
                    worked = True
                seg._rec_drained = drained
                if drained >= seg.committed() and (seg.finalized() or stop.is_set()):
                    # Segment fully flushed (or frozen at its final committed frame): move on.
                    stream_done = seg.stream_ended()
                    fo.release(seg)
                    if stream_done:
                        break                             # clean end of stream: our exit signal
                    continue
            elif stop.is_set() or fo.ended():
                break                                     # queue empty + stop/ended: all flushed
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
