"""
astrolock_seeker_focus_sweep: actuator-agnostic focus sweep -- request positions, fit the V-curve.

The sweep doesn't know who moves the focuser. It publishes its state -- most importantly the
current position REQUEST ("focus to x please") -- on the '{role}_sweep' stream (raw JSON blobs,
latest wins), and watches '{role}_focuser' for position REPORTS ({'pos': x}). Whoever can move
the knob plays actuator: an electronic-focuser process reporting real positions, or the GUI
showing the human "Set focuser to x" with an OK button that echoes the request back. Same sweep
either way.

Focus quality comes from the running focus process's per-frame records ('{role}_focus'
extras, capture-timestamped). After a position is reported we skip a settle window (hands just
shook the tube / motor still ringing), then collect the next N frames at that position. Every
collected frame goes into ONE least-squares fit at the end. KISS (for now): the metric is
peak_frame -- peak scales ~1/HFD^2, so 1/peak is quadratic in position just like hfd^2 was --
fit 1/peak = a p^2 + b p + c linearly, then best focus p0 = -b/2a with expected peak
1/(c - b^2/4a). No iteration. Saturated frames (peak pinned at full scale) carry no focus
information and are EXCLUDED from the fit; a high excluded fraction = reduce exposure.

The backend holds our stdin: a line or EOF = abort (publish the aborted state, end the stream,
exit). We also exit if the focus stream ends under us.

    python -m astrolock.seeker.focus_sweep --session sessions/<ts> --role main \
        --start 1 --end 9 --steps 9
"""

import argparse
import json
import math
import sys
import threading
import time

import numpy as np

from astrolock.seeker import framestream
from astrolock.seeker import session as session_mod

SAT_PEAK = 0.98          # peak_frame at/above this = clipped core: blind, excluded from the fit


def fit_vcurve(points):
    """Least squares over ALL (pos, peak_frame) frames: 1/peak = a p^2 + b p + c (peak scales
    ~1/spread^2, so its reciprocal is the same quadratic hfd^2 was). Returns (p0, peak0, coeffs)
    or None when the data has no interior best (a <= 0: focus wasn't bracketed, or noise)."""
    p = np.array([q[0] for q in points], dtype=np.float64)
    y = 1.0 / np.array([q[1] for q in points], dtype=np.float64)
    A = np.stack([p * p, p, np.ones_like(p)], axis=1)
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    if a <= 0:
        return None
    p0 = -b / (2 * a)
    ymin = c - b * b / (4 * a)
    return p0, (1.0 / ymin if ymin > 0 else 0.0), (a, b, c)


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker focus sweep (actuator-agnostic)")
    p.add_argument('--session', required=True)
    p.add_argument('--role', default='main', help="role whose focus stream feeds the sweep")
    p.add_argument('--start', type=float, required=True, help="first focuser position")
    p.add_argument('--end', type=float, required=True, help="last focuser position")
    p.add_argument('--steps', type=int, default=9, help="number of positions (>= 3 to fit)")
    p.add_argument('--frames-per-step', type=int, default=40,
                   help="frames collected at each position")
    p.add_argument('--settle-s', type=float, default=1.0,
                   help="dead time after a position report before frames count (vibration)")
    p.add_argument('--pos-tol', type=float, default=0.0,
                   help="accept a reported position within this of the request "
                        "(0 = auto: a quarter of the step spacing)")
    p.add_argument('--poll', type=float, default=0.02)
    args = p.parse_args(argv)

    steps = max(3, args.steps)
    positions = list(np.linspace(args.start, args.end, steps))
    spacing = abs(args.end - args.start) / max(1, steps - 1)
    tol = args.pos_tol if args.pos_tol > 0 else max(spacing / 4.0, 1e-9)

    stop = threading.Event()

    def _stdin_watch():                        # backend holds our pipe: line = abort, EOF = abort
        try:
            sys.stdin.readline()
        except Exception:
            pass
        stop.set()
    threading.Thread(target=_stdin_watch, daemon=True).start()

    out = framestream.FrameStream(args.session, f'{args.role}_sweep')
    fo_hfd = framestream.StreamFollower(args.session, f'{args.role}_focus', keep_all=True)
    fo_pos = framestream.StreamFollower(args.session, f'{args.role}_focuser', keep_all=True)

    state = {'of': steps, 'need': args.frames_per_step, 'done': False}

    def publish(**kw):
        state.update(kw)
        if out._seg is None or out.full:
            out.open_segment(session_mod.segment_stamp(), 1 << 14, 1, pixel_depth=8,
                             cap=256, raw=True)
        payload = json.dumps(state, separators=(',', ':')).encode('utf-8')
        out.write(np.frombuffer(payload, np.uint8), t_mono_ns=time.monotonic_ns())

    def next_position_report(after_ns, want):
        """Newest position report at ~want stamped after the request, else None."""
        fo_pos.poll()
        hit = None
        for seg in fo_pos.segs:
            i = getattr(seg, '_used', 0)
            while i < seg.committed():
                rec = seg.record(i)
                blob = json.loads(bytes(seg.read(i)).decode('utf-8'))
                i += 1
                t = rec['t_mono_ns'] or time.monotonic_ns()
                if t >= after_ns and abs(float(blob.get('pos', math.nan)) - want) <= tol:
                    hit = t
            seg._used = i
        while len(fo_pos.segs) > 1 and getattr(fo_pos.segs[0], '_used', 0) \
                >= fo_pos.segs[0].committed() and fo_pos.segs[0].finalized():
            fo_pos.release(fo_pos.segs[0])
        return hit

    def drain_metrics(since_ns, sink, limit):
        """Feed per-frame peak_frame records captured after since_ns into sink, up to limit."""
        fo_hfd.poll()
        while fo_hfd.segs and len(sink) < limit:
            seg = fo_hfd.segs[0]
            i = getattr(seg, '_used', 0)
            while i < seg.committed() and len(sink) < limit:
                rec = seg.record(i)
                i += 1
                pf = rec.get('peak_frame')
                if rec['t_mono_ns'] < since_ns or pf is None or math.isnan(pf) or pf <= 0:
                    continue
                sink.append(pf)
            seg._used = i
            if seg.finalized() and i >= seg.committed() and len(fo_hfd.segs) > 1:
                fo_hfd.release(seg)
                continue
            break

    points = []                                # (pos, peak_frame) for every unsaturated frame
    sat = total = 0
    aborted = False
    try:
        for k, pos in enumerate(positions):
            want = round(float(pos), 6)
            publish(step=k + 1, want_pos=want, awaiting='position', collected=0)
            print(f"[sweep:{args.role}] step {k + 1}/{steps}: focus to {want}", flush=True)
            req_ns = time.monotonic_ns()
            t_ok = None
            while t_ok is None and not stop.is_set() and not fo_hfd.ended():
                t_ok = next_position_report(req_ns, want)
                if t_ok is None:
                    time.sleep(args.poll)
            if t_ok is None:
                break
            publish(awaiting='frames')
            window = []
            since = t_ok + int(args.settle_s * 1e9)
            while len(window) < args.frames_per_step and not stop.is_set():
                n0 = len(window)
                drain_metrics(since, window, args.frames_per_step)
                if fo_hfd.ended() and not fo_hfd.segs:
                    break
                if len(window) != n0:
                    publish(collected=len(window))
                else:
                    time.sleep(args.poll)
            if stop.is_set():
                break
            points.extend((want, pf) for pf in window if pf < SAT_PEAK)  # saturated = no info
            sat += sum(1 for pf in window if pf >= SAT_PEAK)
            total += len(window)
        aborted = stop.is_set() or total == 0
    finally:
        result = {'done': True, 'aborted': aborted,
                  'points': [[p_, round(v, 4)] for p_, v in points],
                  'sat_frac': round(sat / total, 3) if total else 0.0}
        fit = fit_vcurve(points) if len(points) >= 3 and not aborted else None
        if fit is not None:
            p0, peak0, _ = fit
            result.update(p0=round(p0, 4), peak0=round(peak0, 4),
                          bracketed=bool(min(positions) <= p0 <= max(positions)))
            print(f"[sweep:{args.role}] best focus p0={p0:.4g} (peak {peak0:.2f}, "
                  f"{total} frames, sat {result['sat_frac']:.0%})", flush=True)
        elif not aborted:
            result['error'] = 'no minimum found (focus not bracketed, or data too noisy)'
            print(f"[sweep:{args.role}] fit failed: {result['error']}", flush=True)
        else:
            print(f"[sweep:{args.role}] aborted", flush=True)
        publish(**result)
        out.close()
        fo_hfd.close()
        fo_pos.close()


if __name__ == '__main__':
    main()
