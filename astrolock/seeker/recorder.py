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
  * the stream's ended flag (the cam stopped or relaunched): drain everything committed,
    finalize, exit. Continuity across a cam relaunch is the backend's reconciliation loop's
    job -- it spawns a fresh recorder against the successor stream;
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
import json
import os
import sys
import threading
import time

from astrolock.seeker import bayer, framestream, ser as ser_mod


def _utc_of(rec):
    ns = rec.get('t_utc_ns') or 0
    return dt.datetime.fromtimestamp(ns / 1e9, dt.timezone.utc) if ns else None


def _iso(t):
    return t.strftime('%Y-%m-%dT%H:%M:%S.') + f"{t.microsecond // 1000:03d}Z" if t else 'unknown'


def _colour_format(color_id, depth):
    """The sidecar's 'Colour Format' in ASICap's vocabulary: RAW8/RAW16 for a Bayer mosaic,
    MONO8/MONO16 otherwise."""
    bits = 8 if depth <= 8 else 16
    return f"{'RAW' if bayer.is_bayer(color_id) else 'MONO'}{bits}"


class Sidecar:
    """ASICap-compatible <recording>.ser.txt metadata sidecar (SharpCap/FireCapture write the
    same style): '[camera]' then 'Key = Value' lines. The static block lands at file OPEN (a
    crash still leaves the settings); 'Change = ...' lines are appended as live control changes
    arrive; EndCapture/FrameCount close it out. One sidecar per output .ser."""

    def __init__(self, ser_path, meta, color_id, depth, w, h, start_utc):
        self.path = ser_path + '.txt'
        meta = dict(meta)
        camera = meta.pop('Camera', 'unknown')
        static = {
            'Bin': meta.pop('Bin', None),
            'Capture Area Size': f"{w} * {h}",
            'Colour Format': _colour_format(color_id, depth),
            'Exposure': meta.pop('Exposure', None),
            'Gain': meta.pop('Gain', None),
            'Output Format': '*.SER',
            'Raw Format': 'ON',
            'StartCapture': _iso(start_utc),
            'Timestamp Frames': 'ON',              # we always write the SER timestamp trailer
            'TimeZone': f"{dt.datetime.now().astimezone().utcoffset().total_seconds() / 3600:g}",
        }
        if bayer.is_bayer(color_id):
            static['Debayer Type'] = ser_mod.ColorId(color_id).name.replace('BAYER_', '')
        static.update(meta)                        # the backend's remaining faithful keys
        self._f = open(self.path, 'w', encoding='ascii', errors='replace', newline='\r\n')
        self._f.write(f"[{camera}]\n")
        for k in sorted(static):
            if static[k] is not None:
                self._f.write(f"{k} = {static[k]}\n")
        self._f.flush()

    def change(self, utc, frame_index, text):
        """A live control change, stamped with capture time + the frame it lands before."""
        self._f.write(f"Change = {_iso(utc)} frame {frame_index}: {text}\n")
        self._f.flush()

    def close(self, end_utc, frame_count, dropped):
        self._f.write(f"EndCapture = {_iso(end_utc)}\n")
        self._f.write(f"FrameCount = {frame_count}\n")
        if dropped:
            self._f.write(f"Frames Dropped = {dropped}\n")
        self._f.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker recorder (live stream -> disk .ser)")
    p.add_argument('--session', required=True, help="session directory of the live stream")
    p.add_argument('--role', default='main', help="stream to record (guide/main)")
    p.add_argument('--name', default=None,
                   help="filename suffix: <timestamp>_<name>.ser (default: the role)")
    p.add_argument('--out-dir', default='recordings', help="archive directory (never cleaned up)")
    # SER header identity strings (each stored as 40 ASCII bytes), composed by the backend:
    # Observer = recording software, Instrument = camera + capture settings,
    # Telescope = configured optics.
    p.add_argument('--observer', default='')
    p.add_argument('--instrument', default='')
    p.add_argument('--telescope', default='')
    p.add_argument('--sidecar-meta', default='',
                   help="JSON dict of static ASICap-style sidecar fields (from the backend); "
                        "empty = no .ser.txt sidecars")
    p.add_argument('--resume', action='store_true',
                   help="a reconcile RELAUNCH (predecessor died mid-recording): start from the "
                        "ring's oldest available frame instead of 'now', so the ring bridges "
                        "the handover with no gap")
    p.add_argument('--poll', type=float, default=0.02)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    stop = threading.Event()
    change_queue = []                         # sidecar change events from the backend (stdin JSON)
    change_lock = threading.Lock()

    def _stdin_watch():
        """The backend holds our pipe. JSON lines are sidecar change events; any OTHER line --
        or EOF (backend gone) -- means stop recording."""
        try:
            for line in sys.stdin:
                line = line.strip()
                if line.startswith('{'):
                    try:
                        with change_lock:
                            change_queue.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"[rec:{args.role}] bad stdin event ignored: {e}", flush=True)
                    continue
                break                          # non-JSON line = explicit stop
        except Exception:
            pass
        stop.set()
    threading.Thread(target=_stdin_watch, daemon=True).start()

    fo = framestream.StreamFollower(args.session, args.role)
    fo.poll()
    if not args.resume:
        fo.skip_to_latest()                   # record from the CLICK; history is not backlog
    base_lost = fo.lost                       # anything before this moment isn't "skipped recording"

    out = None
    out_geom = None
    out_path = None
    written = skipped = thinned = 0
    first_t_ns = last_t_ns = None                 # capture stamps of the first/last WRITTEN frame
    phase = 0                                     # decimation phase for the watermark backoff
    sidecar_meta = json.loads(args.sidecar_meta) if args.sidecar_meta else None
    sidecar = None                                # the CURRENT file's .ser.txt (one per .ser)
    side_count0 = 0                               # `written` when the current file opened
    side_dropped0 = 0
    side_last_utc = None                          # capture UTC of the current file's last frame

    def sidecar_finish():
        nonlocal sidecar
        if sidecar is not None:
            sidecar.close(side_last_utc, written - side_count0,
                          (skipped_now() + thinned) - side_dropped0)
            sidecar = None

    def skipped_now():
        return skipped + max(0, fo.lost - base_lost)

    def keep_stride(lag_frac):
        """How aggressively to thin as the drain cursor falls behind the write head. Writing the
        oldest available frame is the one MOST likely to be lapped mid-write, so as our backlog
        fills the ring we deliberately drop evenly-spaced frames to hold a safety margin ahead of
        the writer -- graceful degradation instead of hugging the trailing edge and losing whole
        chunks to laps. (Disk steady-but-slower-than-USB is exactly this case.)"""
        if lag_frac < 0.80:
            return 1                              # plenty of runway: keep every frame
        if lag_frac < 0.90:
            return 2                              # keep every other
        if lag_frac < 0.95:
            return 4
        return 8                                  # deep in the buffer: thin hard to escape the write head

    def archive_one(rd, i):
        """Write frame i straight from the ring; True if it landed, False if lapped."""
        nonlocal out, out_geom, out_path, written, skipped, first_t_ns, last_t_ns
        nonlocal sidecar, side_count0, side_dropped0, side_last_utc
        wrote = False
        try:
            with rd.view(i) as (rec, frame):           # ZERO-COPY: write() reads the slot in place
                # A new file on ANY format change -- shape, mosaic, or bit depth (a bit-depth
                # switch keeps the shape while changing the sample size; appending those frames
                # to the old file would corrupt it).
                geom = (frame.shape, rd.header.color_id, rd.header.pixel_depth_per_plane)
                if out is None or out_geom != geom:
                    if out is not None:
                        out.close()
                    sidecar_finish()
                    out_geom = geom
                    # Every output file is named for its FIRST frame's capture time (a
                    # geometry change just starts the next file the same way).
                    t0 = _utc_of(rec) or dt.datetime.now(dt.timezone.utc)
                    stamp = t0.strftime('%Y%m%dT%H%M%S') + f"{t0.microsecond // 1000:03d}Z"
                    out_path = os.path.join(args.out_dir, f"{stamp}_{args.name or args.role}.ser")
                    print(f"[rec:{args.role}] -> {out_path} "
                          f"[{args.instrument} | {args.telescope}]", flush=True)
                    out = ser_mod.SerWriter(out_path, frame.shape[1], frame.shape[0],
                                            color_id=rd.header.color_id,
                                            pixel_depth_per_plane=rd.header.pixel_depth_per_plane,
                                            observer=args.observer, instrument=args.instrument,
                                            telescope=args.telescope)
                    if sidecar_meta is not None:
                        sidecar = Sidecar(out_path, sidecar_meta, rd.header.color_id,
                                          rd.header.pixel_depth_per_plane,
                                          frame.shape[1], frame.shape[0], t0)
                        side_count0 = written
                        side_dropped0 = skipped_now() + thinned
                        side_last_utc = t0
                u = _utc_of(rec)
                out.write_frame(frame, t_utc=u)        # trailer gets CAPTURE time
                if u is not None:
                    side_last_utc = u
                t_ns = int(rec['t_mono_ns'])
                wrote = True
        except framestream.Lapped:
            if wrote:
                out.truncate_last_frame()              # the slot was reused mid-write: torn bytes
            skipped += 1
            return False
        if first_t_ns is None:
            first_t_ns = t_ns
        last_t_ns = t_ns
        written += 1
        return True

    def _archive_or_thin(rd, i):
        """Apply the watermark backoff, then archive. Deliberately-thinned frames count separately
        from lapped ones. `rd.cap` is the ring's slot count; committed()-i is how far behind we are."""
        nonlocal phase, thinned
        cap = getattr(rd, 'cap', 0) or 1
        lag_frac = max(0, rd.committed() - i) / cap
        stride = keep_stride(lag_frac)
        phase += 1
        if stride > 1 and phase % stride != 0:
            thinned += 1
            return False
        return archive_one(rd, i)

    def _fps(n):
        dur = (last_t_ns - first_t_ns) / 1e9 if (first_t_ns and last_t_ns and last_t_ns > first_t_ns) else 0.0
        return (n - 1) / dur if dur > 0 and n > 1 else 0.0

    try:
        while True:
            fo.poll()
            worked = False
            for rd, i in fo.drain(limit=256):
                worked = _archive_or_thin(rd, i) or worked
            with change_lock:                          # live control changes -> sidecar lines
                events, change_queue[:] = list(change_queue), []
            for ev in events:
                if sidecar is not None and ev.get('sidecar_change'):
                    sidecar.change(side_last_utc, written - side_count0, ev['sidecar_change'])
            if stop.is_set():
                for rd, i in fo.drain():               # freeze: flush what's already committed
                    _archive_or_thin(rd, i)
                break
            if not worked and fo.ended():
                # The stream ended (cam stopped or relaunched). Finalize and exit -- simple,
                # testable lifecycle. Continuity across a cam RELAUNCH is the BACKEND'S job:
                # its reconciliation loop sees a wanted-but-dead recorder and spawns a fresh
                # one against the successor stream (the missed-ISS-pass bug was that nothing
                # ever did).
                break
            if not worked:
                time.sleep(args.poll)
    except KeyboardInterrupt:
        pass
    finally:
        if out is not None:
            out.close()
        sidecar_finish()
        fo.close()
        dropped = skipped_now() + thinned
        avg_fps = _fps(written)                                     # actual written frame rate
        ideal_fps = _fps(written + dropped)                        # what it'd have been with zero drops
        print(f"[rec:{args.role}] done: {written} frames written @ {avg_fps:.1f} fps"
              f" ({dropped} dropped: {skipped_now()} lapped + {thinned} thinned; "
              f"~{ideal_fps:.1f} fps if none dropped)"
              + (f" -> {out_path}" if out_path else " (nothing arrived)"), flush=True)


if __name__ == '__main__':
    main()
