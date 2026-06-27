"""
astrolock_seeker_backend: the orchestrator + (manual) control loop.

Creates a session, launches a cam + detector per role (and optionally the GUI), then runs a
~20 Hz control loop that:
  - maintains an *encoder-angle estimate* (integrating commanded rates), published to
    <ts>_state.jsonl with health -- the backend's belief about the mount, which the sim
    camera follows (and a real mount driver will later replace);
  - accepts live GUI commands over a non-blocking socket (slew/stop/record/capture).

Cameras are driven via per-role control JSONL files (control_<role>.jsonl): record toggles
their `important` flag; capture on/off (re)launches or stops a cam. On exit, any .ser whose
sidecar has no important frame is deleted; if nothing important remains, the session goes.

    python -m astrolock.seeker.backend --roles guide --source sky
"""

import argparse
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

from astrolock.seeker import control
from astrolock.seeker import session as session_mod
from astrolock.seeker import sidecar
from astrolock.seeker.follower import SerFollower
from astrolock.seeker.sidecar import JsonlWriter


def _spawn(module, args):
    """Launch `python -m <module> <args>` using the same interpreter, in the repo cwd."""
    return subprocess.Popen([sys.executable, '-m', module, *args])


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker backend / orchestrator")
    p.add_argument('--roles', default='guide', help="comma-separated camera roles to launch")
    p.add_argument('--source', default='synthetic', choices=['synthetic', 'zwo', 'sky'])
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--segment-frames', type=int, default=300,
                   help="roll cams to a new file every N frames (old non-important ones are deleted)")
    p.add_argument('--auto', dest='auto', action='store_true', default=True,
                   help="zwo: auto-exposure + auto-gain (default on)")
    p.add_argument('--no-auto', dest='auto', action='store_false', help="zwo: fixed exposure/gain")
    p.add_argument('--start-az-deg', type=float, default=299.3, help="initial encoder az estimate")
    p.add_argument('--start-alt-deg', type=float, default=62.3, help="initial encoder alt estimate")
    p.add_argument('--max-rate-deg-s', type=float, default=8.0, help="clamp on commanded slew rate")
    p.add_argument('--sky-rate-az', type=float, default=0.0, help="sky source: scripted az slew (deg/s)")
    p.add_argument('--sky-rate-alt', type=float, default=0.0, help="sky source: scripted alt slew (deg/s)")
    p.add_argument('--sky-substeps', type=int, default=6, help="sky source: substeps per exposure")
    p.add_argument('--sky-exposure-s', type=float, default=0.1, help="sky source: simulated exposure (s)")
    p.add_argument('--sky-focal-mm', type=float, default=8.0, help="sky source: lens focal length (mm)")
    p.add_argument('--wb-r', type=float, default=1.24, help="GUI display-only WB gain for red (1 = none)")
    p.add_argument('--wb-b', type=float, default=1.98, help="GUI display-only WB gain for blue (1 = none)")
    p.add_argument('--gui', dest='gui', action='store_true', default=True)
    p.add_argument('--no-gui', dest='gui', action='store_false')
    p.add_argument('--duration', type=float, default=0.0, help="stop after N seconds (0 = until Ctrl-C)")
    p.add_argument('--keep', action='store_true', help="keep all captured files on exit")
    args = p.parse_args(argv)

    roles = [r.strip() for r in args.roles.split(',') if r.strip()]
    session_dir, ts = session_mod.new_session_dir()
    stop_file = os.path.join(session_dir, 'stop')          # for detect (cams use control files)

    cmd_server = control.CommandServer()
    with open(os.path.join(session_dir, f"{ts}_backend.json"), 'w') as f:
        json.dump({'command_host': cmd_server.host, 'command_port': cmd_server.port}, f)
    state_writer = JsonlWriter(os.path.join(session_dir, session_mod.state_name(ts)))
    print(f"[backend] session {session_dir} roles={roles} source={args.source} "
          f"cmd_port={cmd_server.port}", flush=True)

    sky_args = (['--sky-rate-az', str(args.sky_rate_az), '--sky-rate-alt', str(args.sky_rate_alt),
                 '--sky-substeps', str(args.sky_substeps), '--sky-exposure-s', str(args.sky_exposure_s),
                 '--sky-focal-mm', str(args.sky_focal_mm), '--sky-follow-state']
                if args.source == 'sky' else [])

    # Mount-estimate state (rad, rad/s) and record state.
    enc = [math.radians(args.start_az_deg), math.radians(args.start_alt_deg)]
    rate = [0.0, 0.0]
    estop = False
    recording = False
    max_rate = math.radians(args.max_rate_deg_s)

    sources = {role: args.source for role in roles}      # switchable live (sim <-> real)
    launch_seq = {role: 0 for role in roles}
    cam_procs = {}
    control_writers = {}

    def control_path(role):
        return os.path.join(session_dir, f"control_{role}_{launch_seq[role]}.jsonl")

    def launch_cam(role):
        cf = control_path(role)                    # unique per launch -> no clobber/replay
        if role in control_writers:
            control_writers[role].close()
        control_writers[role] = JsonlWriter(cf)
        cam_procs[role] = _spawn('astrolock.seeker.cam', [
            '--role', role, '--out-dir', session_dir, '--source', sources[role],
            '--width', str(args.width), '--height', str(args.height), '--fps', str(args.fps),
            '--frame-limit', str(args.segment_frames), '--file-limit', '-1',
            '--important', '1' if recording else '0', '--control-file', cf,
            *(['--auto'] if args.auto else []), *sky_args,
        ])
        control_writers[role].append({'important': 1 if recording else 0})

    detect_procs = {}

    def launch_detect(role):
        detect_procs[role] = _spawn('astrolock.seeker.detect',
                                    ['--session', session_dir, '--role', role, '--follow',
                                     '--stop-file', stop_file])

    for role in roles:
        launch_cam(role)
        launch_detect(role)
    gui_proc = _spawn('astrolock.seeker.gui',
                      ['--session', session_dir, '--wb-r', str(args.wb_r), '--wb-b', str(args.wb_b)]) \
        if args.gui else None

    followers = {role: SerFollower(session_dir, role) for role in roles}

    def control_write(role, obj):
        if role in control_writers:
            control_writers[role].append(obj)

    def restart_cam(role, stop_first):
        if stop_first:
            control_write(role, {'stop': True})    # old cam finalizes its file + exits
        launch_seq[role] += 1
        launch_cam(role)                           # new cam (new source / fresh segment)
        if role not in detect_procs or detect_procs[role].poll() is not None:
            launch_detect(role)                    # safety: detect normally rolls on its own

    def delete_old_segments():
        """Rolling cleanup: drop non-important segments older than the two newest per role."""
        if args.keep:
            return
        for role in roles:
            segs = sorted(glob.glob(os.path.join(session_dir, f'*_{role}.ser')))
            for sp in segs[:-2]:
                stem = sp[:-len('.ser')]
                important = any(r.get('important')
                                for r in sidecar.read_complete_lines(stem + '.frames.jsonl'))
                if not important:
                    for ext in ('.ser', '.frames.jsonl', '.detections.jsonl'):
                        try:
                            os.remove(stem + ext)
                        except OSError:
                            pass

    def apply_command(cmd):
        nonlocal estop, recording
        t = cmd.get('type')
        if t == 'set_rate':
            rate[0] = max(-max_rate, min(max_rate, math.radians(cmd.get('az', 0.0))))
            rate[1] = max(-max_rate, min(max_rate, math.radians(cmd.get('alt', 0.0))))
            estop = False
        elif t == 'stop':
            rate[0] = rate[1] = 0.0
        elif t == 'estop':
            rate[0] = rate[1] = 0.0
            estop = True
        elif t == 'record':
            recording = bool(cmd.get('on', False))
            # Record: whole pass in one important file (stop rolling). Stop: resume rolling,
            # which finalizes the (now over-length) pass file and starts a fresh throwaway.
            for role in roles:
                control_write(role, {'important': 1 if recording else 0,
                                     'frame_limit': -1 if recording else args.segment_frames})
        elif t == 'capture':
            role = cmd.get('role')
            if role in roles:
                if cmd.get('on', True):                 # (re)start a stopped camera + its detector
                    if role not in cam_procs or cam_procs[role].poll() is not None:
                        restart_cam(role, stop_first=False)
                else:
                    control_write(role, {'stop': True})  # cam finalizes its file and exits
        elif t == 'set_source':
            role = cmd.get('role')
            src = cmd.get('source')
            if role in roles and src in ('synthetic', 'zwo', 'sky'):
                sources[role] = src
                restart_cam(role, stop_first=True)       # swap sim <-> real live

    start = time.perf_counter()
    last = start
    last_health = start
    last_cleanup = start
    clean = False
    try:
        while True:
            time.sleep(0.05)                      # ~20 Hz control loop
            now = time.perf_counter()
            dt = now - last
            last = now

            for cmd in cmd_server.drain():
                apply_command(cmd)

            enc[0] = (enc[0] + rate[0] * dt) % (2 * math.pi)
            enc[1] = max(-math.pi / 2, min(math.pi / 2, enc[1] + rate[1] * dt))
            mode = 'estop' if estop else ('slew' if (rate[0] or rate[1]) else 'idle')

            state_writer.append({
                't_mono_ns': time.perf_counter_ns(),
                't_utc': session_mod.utc_now_iso(),
                'mode': mode,
                'enc_az_deg': round(math.degrees(enc[0]), 4),
                'enc_alt_deg': round(math.degrees(enc[1]), 4),
                'rate_az_deg_s': round(math.degrees(rate[0]), 4),
                'rate_alt_deg_s': round(math.degrees(rate[1]), 4),
                'recording': recording,
                'sources': dict(sources),
                'capturing': {role: (role in cam_procs and cam_procs[role].poll() is None)
                              for role in roles},
                'cameras': {role: {'frames': followers[role].committed_count()} for role in roles},
            })

            if now - last_health >= 1.0:
                last_health = now
                health = ', '.join(f"{role}={followers[role].committed_count()}f" for role in roles)
                print(f"[backend] {mode}{' REC' if recording else ''} "
                      f"az={math.degrees(enc[0]):.2f} alt={math.degrees(enc[1]):.2f} | {health}", flush=True)

            if now - last_cleanup >= 2.0:
                last_cleanup = now
                delete_old_segments()

            if gui_proc is not None and gui_proc.poll() is not None:
                print("[backend] gui exited; stopping", flush=True)
                break
            if args.duration and (now - start) >= args.duration:
                print("[backend] duration reached; stopping", flush=True)
                break
        clean = True
    except KeyboardInterrupt:
        print("[backend] interrupted; stopping", flush=True)
        clean = True
    finally:
        for role in roles:                         # tell cams to finalize + exit
            control_write(role, {'stop': True})
            control_writers[role].close()
        open(stop_file, 'w').close()               # tell detectors to exit
        if gui_proc is not None and gui_proc.poll() is None:
            gui_proc.terminate()
        cmd_server.close()
        state_writer.close()
        for fo in followers.values():
            fo.close()

        deadline = time.perf_counter() + 3.0
        for pr in list(cam_procs.values()) + list(detect_procs.values()) + ([gui_proc] if gui_proc else []):
            try:
                pr.wait(timeout=max(0.0, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired:
                pr.terminate()

        _cleanup(session_dir, keep=args.keep, clean=clean)
        print("[backend] done", flush=True)


def _cleanup(session_dir, keep, clean):
    """Delete .ser segments with no important frames; remove the session if nothing remains."""
    if keep:
        print(f"[backend] kept session {session_dir} (--keep)", flush=True)
        return
    kept = 0
    for ser_path in glob.glob(os.path.join(session_dir, '*.ser')):
        stem = ser_path[:-len('.ser')]
        important = any(r.get('important') for r in sidecar.read_complete_lines(stem + '.frames.jsonl'))
        if important:
            kept += 1
        else:
            for ext in ('.ser', '.frames.jsonl', '.detections.jsonl'):
                try:
                    os.remove(stem + ext)
                except OSError:
                    pass
    if kept == 0 and clean:
        shutil.rmtree(session_dir, ignore_errors=True)
        print(f"[backend] removed session {session_dir} (nothing important)", flush=True)
    else:
        print(f"[backend] kept session {session_dir} ({kept} important segment(s))", flush=True)


if __name__ == '__main__':
    main()
