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
from astrolock.seeker import mount as mount_mod
from astrolock.seeker import session as session_mod
from astrolock.seeker import sidecar
from astrolock.seeker.controller import PixelTracker
from astrolock.seeker.follower import SerFollower
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def _spawn(module, args):
    """Launch `python -m <module> <args>` using the same interpreter, in the repo cwd."""
    return subprocess.Popen([sys.executable, '-m', module, *args])


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker backend / orchestrator")
    p.add_argument('--roles', default='guide', help="comma-separated camera roles to launch")
    p.add_argument('--source', default='sky', choices=['synthetic', 'zwo', 'sky'],
                   help="default 'sky' runs the baked-in ISS test pass; 'synthetic' needs no deps")
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--segment-frames', type=int, default=300,
                   help="roll cams to a new file every N frames (old non-important ones are deleted)")
    p.add_argument('--auto', dest='auto', action='store_true', default=True,
                   help="zwo: auto-exposure + auto-gain (default on)")
    p.add_argument('--no-auto', dest='auto', action='store_false', help="zwo: fixed exposure/gain")
    p.add_argument('--start-az-deg', type=float, default=228.0,
                   help="sim mount initial az -- roughly (not exactly) at the ISS test target, so "
                        "acquisition is exercised")
    p.add_argument('--start-alt-deg', type=float, default=19.5, help="sim mount initial alt")
    p.add_argument('--max-rate-deg-s', type=float, default=3.0,
                   help="max slew rate -- matches the real CPC (~3 deg/s, battery-dependent)")
    p.add_argument('--mount', default='sim', choices=['sim', 'celestron'],
                   help="sim integrates commanded rates; celestron drives the real NexStar mount")
    p.add_argument('--mount-url', default=None, help="celestron: e.g. celestron_nexstar_hc:COM3")
    # simulated site (the sim mount's GPS) + dynamics + global sim time scale
    p.add_argument('--lat', type=float, default=37.51089, help="sim site latitude (deg)")
    p.add_argument('--lon', type=float, default=-122.2719388888889, help="sim site longitude (deg)")
    p.add_argument('--elev', type=float, default=60.0, help="sim site elevation (m)")
    p.add_argument('--epoch', default='2026-07-06T05:22:00Z',
                   help="sim start UTC -- the sim mount's GPS time and the sky-sim epoch "
                        "(default: the ISS test pass over San Carlos, just rising)")
    p.add_argument('--mount-accel-deg-s2', type=float, default=20.0, help="sim mount acceleration limit (deg/s^2)")
    p.add_argument('--mount-update-hz', type=float, default=10.0, help="sim mount update rate (Hz)")
    # auto-tracking (pixel-space closed loop)
    p.add_argument('--sky-pixel-um', type=float, default=2.0, help="guide sensor pixel pitch (um)")
    p.add_argument('--arcsec-per-px', type=float, default=0.0,
                   help="guide plate scale override (0 = derive from --sky-pixel-um / --sky-focal-mm)")
    p.add_argument('--track-ki', type=float, default=0.3,
                   help="tracker integral gain (carries the slew rate); kept modest to avoid oscillation")
    p.add_argument('--track-damping', type=float, default=1.3,
                   help="P is derived for critical damping (kp=2*sqrt(ki)); >1 over-damps for lag margin")
    p.add_argument('--track-kd', type=float, default=1.0,
                   help="tracker derivative braking gain (on image speed above --track-max-px-s)")
    p.add_argument('--track-max-px-s', type=float, default=120.0,
                   help="image-speed dead zone (px/s): brake the slew above this during acquisition")
    p.add_argument('--track-vel-smoothing', type=float, default=0.1,
                   help="velocity-estimate smoothing per frame (0 = none/trust new; higher = smoother)")
    p.add_argument('--track-gate-px', type=float, default=80.0, help="max px to associate a blob to the target")
    p.add_argument('--track-lost-s', type=float, default=1.5, help="give up tracking after this long unmatched")
    p.add_argument('--track-sign-az', type=float, default=1.0, help="flip if az moves the image the wrong way")
    p.add_argument('--track-sign-alt', type=float, default=-1.0, help="flip if alt moves the image the wrong way")
    p.add_argument('--sky-tle-file', default='data/iss_25544.tle',
                   help="sky source: satellite TLE file (default: the ISS)")
    p.add_argument('--sky-target-mag', type=float, default=-4.0, help="sky source: satellite magnitude")
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

    max_rate = math.radians(args.max_rate_deg_s)
    site = {'lat_deg': args.lat, 'lon_deg': args.lon, 'elev_m': args.elev, 'epoch_utc': args.epoch}
    mount = mount_mod.make_mount(
        args.mount, az0_rad=math.radians(args.start_az_deg), alt0_rad=math.radians(args.start_alt_deg),
        site=site, max_rate_rad_s=max_rate, accel_rad_s2=math.radians(args.mount_accel_deg_s2),
        update_hz=args.mount_update_hz, url=args.mount_url)
    msite = mount.get_site()        # GPS/site comes from the mount; it drives the sky-sim camera

    sky_args = []
    if args.source == 'sky':
        sky_args = ['--sky-rate-az', str(args.sky_rate_az), '--sky-rate-alt', str(args.sky_rate_alt),
                    '--sky-substeps', str(args.sky_substeps), '--sky-exposure-s', str(args.sky_exposure_s),
                    '--sky-focal-mm', str(args.sky_focal_mm), '--sky-pixel-um', str(args.sky_pixel_um),
                    '--sky-lat', str(msite['lat_deg']), '--sky-lon', str(msite['lon_deg']),
                    '--sky-elev', str(msite['elev_m']), '--sky-epoch', str(msite['epoch_utc']),
                    '--sky-follow-state']
        if args.sky_tle_file:
            sky_args += ['--sky-tle-file', args.sky_tle_file, '--sky-target-mag', str(args.sky_target_mag)]
    estop = False
    recording = False

    # Auto-tracking state (pixel-space closed loop).
    tracker = None
    tracking = False
    track_role = None
    track_target = None
    track_seen_index = -1
    rad_per_px = (math.radians(args.arcsec_per_px / 3600.0) if args.arcsec_per_px > 0
                  else args.sky_pixel_um * 1e-3 / args.sky_focal_mm)

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

    det_tailers = {role: None for role in roles}     # follow each role's detections across rollover
    det_ser = {role: None for role in roles}
    latest_blobs = {role: [] for role in roles}
    latest_det_index = {role: -1 for role in roles}

    def update_detections():
        for role in roles:
            f = followers[role]
            f.committed_count()                      # refresh which segment is newest
            sp = f.ser_path
            if not sp:
                continue
            if det_ser[role] != sp:                  # rolled to a new segment -> re-point tailer
                if det_tailers[role] is not None:
                    det_tailers[role].close()
                det_tailers[role] = JsonlTailer(sp[:-len('.ser')] + '.detections.jsonl')
                det_ser[role] = sp
            for rec in det_tailers[role].poll():
                latest_blobs[role] = rec.get('blobs', [])
                latest_det_index[role] = rec.get('index', latest_det_index[role])

    def frame_binning(role):
        """Sensor pixels per frame pixel for a role (from the cam's frame sidecar; default 1)."""
        sp = followers[role].ser_path
        if not sp:
            return 1.0
        for r in sidecar.read_complete_lines(sp[:-len('.ser')] + '.frames.jsonl'):
            if 'bin' in r:
                return float(r['bin'][0])
        return 1.0

    def frame_time_s(role, index):
        """Capture time (seconds) of frame `index` for a role, from the cam's frame sidecar.
        The whole control loop is clocked off these so PID dt is the true inter-frame interval
        (the cam's monotonic clock), not the backend's polling jitter."""
        sp = followers[role].ser_path
        if not sp or index < 0:
            return None
        lines = sidecar.read_complete_lines(sp[:-len('.ser')] + '.frames.jsonl')
        if index < len(lines) and 't_mono_ns' in lines[index]:
            return lines[index]['t_mono_ns'] * 1e-9
        return None

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
        nonlocal estop, recording, tracking, track_role, tracker, track_seen_index
        t = cmd.get('type')
        if t == 'set_rate':
            tracking = False                          # manual slew overrides auto-track
            mount.set_rates(math.radians(cmd.get('az', 0.0)), math.radians(cmd.get('alt', 0.0)))
            estop = False
        elif t == 'stop':
            tracking = False
            mount.set_rates(0.0, 0.0)
        elif t == 'estop':
            tracking = False
            mount.set_rates(0.0, 0.0)
            estop = True
        elif t == 'track':                            # lock the pixel-space loop onto a target
            role = cmd.get('role', roles[0])
            px = cmd.get('px')
            if role in roles and px and followers[role].header is not None:
                hdr = followers[role].header
                # Blobs are in frame image space; hold the target at the frame centre. rad_per_px
                # is per *sensor* pixel, so scale by the cam's binning to get rad per frame pixel.
                rpp = rad_per_px * frame_binning(role)
                ft = frame_time_s(role, latest_det_index[role])    # clock off the frame, not wall time
                if ft is not None:
                    tracker = PixelTracker(hdr.image_width / 2.0, hdr.image_height / 2.0, rpp,
                                           ki=args.track_ki, damping=args.track_damping, kd=args.track_kd,
                                           gate_px=args.track_gate_px, lost_s=args.track_lost_s,
                                           vel_smoothing=args.track_vel_smoothing,
                                           max_track_px_s=args.track_max_px_s, max_rate_rad_s=max_rate,
                                           sign_az=args.track_sign_az, sign_alt=args.track_sign_alt)
                    tracker.start(float(px[0]), float(px[1]), ft)
                    track_seen_index = latest_det_index[role]
                    tracking = True
                    track_role = role
                    estop = False
        elif t == 'untrack':
            tracking = False
            mount.set_rates(0.0, 0.0)
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
    last_health = start
    last_cleanup = start
    clean = False
    try:
        while True:
            time.sleep(0.05)                      # ~20 Hz control loop
            now = time.perf_counter()

            for cmd in cmd_server.drain():
                apply_command(cmd)
            update_detections()

            track_status = None
            if tracking and tracker is not None and not estop:
                role = track_role
                if latest_det_index[role] != track_seen_index:   # act once per new frame...
                    ft = frame_time_s(role, latest_det_index[role])
                    if ft is not None:                            # ...clocked by its capture time
                        track_seen_index = latest_det_index[role]
                        raz, ralt, track_status, tpx = tracker.update(latest_blobs[role], True, ft)
                        mount.set_rates(raz, ralt)
                        track_target = list(tpx) if track_status == 'track' else None
                        if track_status == 'lost':
                            tracking = False

            st = mount.get_state()
            moving = abs(st['rate_az_rad_s']) > 1e-9 or abs(st['rate_alt_rad_s']) > 1e-9
            if tracking:
                mode = 'track'
            elif track_status == 'lost':
                mode = 'lost'
            elif estop:
                mode = 'estop'
            elif moving:
                mode = 'slew'
            else:
                mode = 'idle'

            state_writer.append({
                't_mono_ns': time.perf_counter_ns(),
                't_utc': session_mod.utc_now_iso(),
                'mode': mode,
                'enc_az_deg': round(math.degrees(st['az_rad']), 4),
                'enc_alt_deg': round(math.degrees(st['alt_rad']), 4),
                'rate_az_deg_s': round(math.degrees(st['rate_az_rad_s']), 4),
                'rate_alt_deg_s': round(math.degrees(st['rate_alt_rad_s']), 4),
                'recording': recording,
                'tracking': tracking,
                'track_role': track_role if tracking else None,
                'target_px': track_target if tracking else None,
                'sources': dict(sources),
                'capturing': {role: (role in cam_procs and cam_procs[role].poll() is None)
                              for role in roles},
                'cameras': {role: {'frames': followers[role].committed_count()} for role in roles},
            })

            if now - last_health >= 1.0:
                last_health = now
                health = ', '.join(f"{role}={followers[role].committed_count()}f" for role in roles)
                print(f"[backend] {mode}{' REC' if recording else ''} "
                      f"az={math.degrees(st['az_rad']):.2f} alt={math.degrees(st['alt_rad']):.2f} | {health}",
                      flush=True)

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
        mount.close()
        for fo in followers.values():
            fo.close()
        for dt_ in det_tailers.values():
            if dt_ is not None:
                dt_.close()

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
