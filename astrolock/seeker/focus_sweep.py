"""
astrolock_seeker_focus_sweep: actuator-agnostic focus sweep -- request positions, fit the V-curve.

The sweep doesn't know who moves the focuser. It publishes its state -- most importantly the
current position REQUEST ("focus to x please") -- on the '{role}_sweep' stream (raw JSON blobs,
latest wins), and watches '{role}_focuser' for position REPORTS ({'pos': x}). Whoever can move
the knob plays actuator: the backend's EAF driver, or the GUI showing the human "Set focuser
to x" with an OK button that echoes the request back. Same sweep either way.

Each position (bucket) produces ONE stacked image and ONE quality figure of merit: after the
move + settle we publish awaiting='frames', the BACKEND answers by resetting the focus
process's stack into pure-average mode (control_focus_<role>.jsonl), and we read the focus
stream's extras until a post-reset record (ctl_seq advanced past the pre-bucket value) reports
stack_n >= --frames-per-step. That record's stack_strehl is the bucket's value -- quality of
the average, never averages of per-frame quality (individual frames are seeing-speckle draws).

Buckets visit positions ALTERNATING SIDES, extremes first (-R, +R, next in, ...), so slow
seeing/transparency drift cancels out of the fit by symmetry instead of biasing the vertex --
the same instinct as a human going way off one side, then the other, then splitting the
difference.

Fit: defocus adds width in quadrature, so 1/strehl = a p^2 + b p + c; best focus p0 = -b/2a
with expected strehl 1/(c - b^2/4a). Clipped frames degrade the stack Strehl gracefully (the
full-region denominator keeps the sign correct), so no points are excluded -- but the clipped
fraction rides along per bucket and the result warns when it's high.

The backend holds our stdin: a line or EOF = abort (publish the aborted state, end the stream,
exit). We also exit if the focus stream ends under us.

    python -m astrolock.seeker.focus_sweep --session sessions/<ts> --role main \
        --start -1000 --end 1000 --steps 9 --frames-per-step 50
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


def sweep_order(steps):
    """Bucket visit order: extremes first, alternating sides, walking inward. Adjacent-in-time
    buckets sit on opposite sides at similar leverage, so a linear drift in seeing or
    transparency largely cancels out of the fit."""
    lo, hi = 0, steps - 1
    order = []
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo, hi = lo + 1, hi - 1
    return order


def fit_vcurve(points):
    """Least squares over the bucket (pos, strehl) pairs: 1/strehl = a p^2 + b p + c (defocus
    adds width in quadrature; strehl ~ 1/width^2). Returns (p0, strehl0, coeffs) or None when
    the data has no interior best (a <= 0: focus wasn't bracketed, or noise)."""
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
    p.add_argument('--frames-per-step', type=int, default=50,
                   help="frames stacked per position (the backend derives this from the "
                        "seconds-of-exposure knob and the camera's exposure time)")
    p.add_argument('--settle-s', type=float, default=1.0,
                   help="dead time after a position report before the stack starts (vibration)")
    p.add_argument('--pos-tol', type=float, default=0.0,
                   help="accept a reported position within this of the request "
                        "(0 = auto: a quarter of the step spacing)")
    p.add_argument('--poll', type=float, default=0.02)
    args = p.parse_args(argv)

    steps = max(3, args.steps)
    frames = max(1, args.frames_per_step)
    grid = list(np.linspace(args.start, args.end, steps))
    positions = [grid[k] for k in sweep_order(steps)]
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
    fo_focus = framestream.StreamFollower(args.session, f'{args.role}_focus')
    fo_pos = framestream.StreamFollower(args.session, f'{args.role}_focuser')

    seen_focus = False       # a focus record has arrived during THIS run

    def focus_gone():
        """The focus stream is dead -- but only once we've seen it ALIVE in this run. At sweep
        start the newest head event may be a PREVIOUS focus process's 'ended' (the backend
        relaunches focus for a crop change, and the new one configures its ring only on its
        first camera frame) -- sticky-ended then would abort us instantly for no reason."""
        return seen_focus and fo_focus.ended()

    state = {'of': steps, 'need': frames, 'done': False}

    def publish(**kw):
        state.update(kw)
        if not out.configured:
            out.configure(1 << 14, 1, pixel_depth=8, frames=256, raw=True)
        payload = json.dumps(state, separators=(',', ':')).encode('utf-8')
        out.write(np.frombuffer(payload, np.uint8), t_mono_ns=session_mod.mono_ns())

    def next_position_report(after_ns, want):
        """Newest position report at ~want stamped after the request, else None."""
        fo_pos.poll()
        hit = None
        for rd, i in fo_pos.drain():
            try:
                rec = rd.record(i)
                blob = json.loads(bytes(rd.read(i)).decode('utf-8'))
            except framestream.Lapped:
                continue
            # Deliberate grace, not failure-eating: an external actuator that doesn't stamp
            # its report gets its ARRIVAL time -- only the report/collect ordering matters.
            t = rec['t_mono_ns'] or session_mod.mono_ns()
            if t >= after_ns and abs(float(blob.get('pos', math.nan)) - want) <= tol:
                hit = t
        return hit

    def latest_ctl_seq():
        """The newest focus record's applied control seq (the pre-bucket baseline)."""
        nonlocal seen_focus
        fo_focus.poll()
        got = fo_focus.latest()
        if got is None:
            return -1.0
        seen_focus = True
        try:
            return float(got[0].record(got[1])['ctl_seq'])
        except framestream.Lapped:
            return -1.0

    points = []                                # (pos, stack_strehl, clip_frac) per bucket
    clip_frames = total_frames = 0
    aborted = False

    def pts_out():
        return [[p_, round(s_, 5), round(cf_, 3)] for p_, s_, cf_ in points]

    try:
        for k, pos in enumerate(positions):
            want = round(float(pos), 6)
            publish(step=k + 1, want_pos=want, awaiting='position', collected=0,
                    points=pts_out())
            print(f"[sweep:{args.role}] step {k + 1}/{steps}: focus to {want}", flush=True)
            req_ns = session_mod.mono_ns()
            t_ok = None
            while t_ok is None and not stop.is_set() and not focus_gone():
                fo_focus.poll()                          # keeps seen_focus/ended fresh
                if fo_focus.latest() is not None:
                    seen_focus = True
                t_ok = next_position_report(req_ns, want)
                if t_ok is None:
                    time.sleep(args.poll)
            if t_ok is None:
                break
            end_settle = time.monotonic() + args.settle_s
            while time.monotonic() < end_settle and not stop.is_set():
                time.sleep(args.poll)
            if stop.is_set():
                break
            seq0 = latest_ctl_seq()
            # 'frames' is the backend's cue to reset the focus stack into pure-average mode;
            # the applied reset shows up as ctl_seq advancing past seq0.
            publish(awaiting='frames')
            strehl = None
            bucket_clip = bucket_n = 0
            while strehl is None and not stop.is_set() and not focus_gone():
                fo_focus.poll()
                advanced = False
                for rd, i in fo_focus.drain():
                    try:
                        rec = rd.record(i)
                    except framestream.Lapped:
                        continue
                    seen_focus = True
                    if float(rec['ctl_seq']) <= seq0:
                        continue                     # pre-reset stragglers
                    bucket_n += 1
                    if rec['clip_px'] > 0:
                        bucket_clip += 1
                    if int(rec['stack_n']) != state.get('collected'):
                        publish(collected=int(rec['stack_n']))
                        advanced = True
                    if int(rec['stack_n']) >= frames:
                        strehl = float(rec['stack_strehl'])
                        break
                if strehl is None and not advanced:
                    time.sleep(args.poll)
            if strehl is None:
                break
            cf = bucket_clip / bucket_n if bucket_n else 0.0
            clip_frames += bucket_clip
            total_frames += bucket_n
            points.append((want, strehl, cf))
            print(f"[sweep:{args.role}] bucket {k + 1}: pos {want:g} strehl {strehl:.4f}"
                  + (f" (CLIP {cf:.0%})" if cf else ""), flush=True)
            publish(points=pts_out())
        aborted = stop.is_set() or not points
    finally:
        result = {'done': True, 'aborted': aborted, 'points': pts_out(),
                  'clip_frac': round(clip_frames / total_frames, 3) if total_frames else 0.0}
        usable = [q for q in points if q[1] > 0]
        fit = fit_vcurve(usable) if len(usable) >= 3 and not aborted else None
        if fit is not None:
            p0, strehl0, _ = fit
            result.update(p0=round(p0, 4), strehl0=round(strehl0, 4),
                          bracketed=bool(min(grid) <= p0 <= max(grid)))
            warn = (f"; CLIP {result['clip_frac']:.0%} of frames -- reduce exposure"
                    if result['clip_frac'] > 0.2 else "")
            print(f"[sweep:{args.role}] best focus p0={p0:.4g} (strehl {strehl0:.3f}, "
                  f"{len(points)} buckets{warn})", flush=True)
        elif not aborted:
            result['error'] = 'no best-focus vertex found (focus not bracketed, or data too noisy)'
            print(f"[sweep:{args.role}] fit failed: {result['error']}", flush=True)
        else:
            print(f"[sweep:{args.role}] aborted", flush=True)
        publish(**result)
        out.close()
        fo_focus.close()
        fo_pos.close()


if __name__ == '__main__':
    main()
