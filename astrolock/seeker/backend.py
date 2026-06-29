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

Defaults are our rig: both the guide (ASI678MC + 8mm) and main (ASI678MC + C11) sim cameras
on the ISS sky pass, with the GUI. Use --sim-downscale N for faster/smaller views.

    python -m astrolock.seeker.backend                       # the two-camera rig, defaults
    python -m astrolock.seeker.backend --sim-downscale 4     # same, 1/4 render res (faster)
    python -m astrolock.seeker.backend --roles guide --source synthetic   # one cam, no deps
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
from astrolock.seeker import optics
from astrolock.seeker.follower import SerFollower
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def _spawn(module, args):
    """Launch `python -m <module> <args>` using the same interpreter, in the repo cwd."""
    return subprocess.Popen([sys.executable, '-m', module, *args])


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker backend / orchestrator")
    p.add_argument('--roles', default='guide,main', help="comma-separated camera roles to launch")
    p.add_argument('--source', default='sky', choices=['synthetic', 'zwo', 'sky', 'playback'],
                   help="default 'sky' runs the baked-in ISS test pass; 'synthetic' needs no deps; "
                        "'playback' replays a recorded .ser through the live pipeline")
    p.add_argument('--playback-ser', default=None, help="playback source: the .ser file to replay")
    p.add_argument('--playback-speed', type=float, default=1.0, help="playback source: speed multiplier")
    p.add_argument('--playback-loop', action='store_true', help="playback source: loop instead of stopping")
    p.add_argument('--detect-roles', default='guide',
                   help="comma-separated roles to run blob detection on (default: guide only -- the "
                        "main cam is narrow-field and will get centroid tracking later, not blob detect)")
    p.add_argument('--detector', default='doh', choices=['bandpass', 'doh'],
                   help="detection surface: 'doh' (default) = determinant of the Hessian, or 'bandpass'")
    p.add_argument('--doh-sigma', type=float, default=0.0, help="doh detector: Gaussian scale px (0 = psf default)")
    p.add_argument('--snr', type=float, default=6.0, help="detect: report peaks this many sigma above background")
    p.add_argument('--max-candidates', type=int, default=16, help="detect: max blobs reported per frame")
    p.add_argument('--min-blob-px', type=int, default=2, help="detect: ignore peaks smaller than this")
    p.add_argument('--tile-grid', type=int, default=8,
                   help="detect density cap: tiles across the frame (0 = off); keeps targets from "
                        "a dense bright region from starving the rest of the frame")
    p.add_argument('--per-tile', type=int, default=2, help="detect density cap: max blobs per tile")
    p.add_argument('--device', default='cpu', help="torch device for detect + gui (cpu / cuda)")
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--sim-downscale', type=int, default=1,
                   help="render sim cameras at 1/N resolution, same FoV (fewer pixels = faster + "
                        "smaller windows); e.g. 4. Only affects DB-optic sim cams.")
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
    # Per-role plate scale from the vendored optics DB (names from `python -m astrolock.seeker.optics`).
    p.add_argument('--guide-sensor', default='ZWO ASI678MC', help="guide camera sensor name in the optics DB")
    p.add_argument('--guide-optic', default='8mm CS f/1.4', help="guide optic name in the optics DB")
    p.add_argument('--guide-reducer', default=None, help="guide reducer/barlow name (optional)")
    p.add_argument('--main-sensor', default='ZWO ASI678MC', help="main camera sensor name in the optics DB")
    p.add_argument('--main-optic', default='Celestron C11 f/10', help="main optic name in the optics DB")
    p.add_argument('--main-reducer', default=None, help="main reducer/barlow name (optional)")
    p.add_argument('--track-ki', type=float, default=0.3,
                   help="tracker integral gain (carries the slew rate); kept modest to avoid oscillation")
    p.add_argument('--track-kii', type=float, default=0.0,
                   help="tracker second-integral gain (0 = off): removes the residual lag against a "
                        "constant-acceleration target (satellite overhead). Keep weak -- needs "
                        "kii < kp*ki for stability (kp*ki ~ 0.43 at the default ki/damping)")
    p.add_argument('--track-nominal-rate', type=float, default=10.0,
                   help="framerate (Hz) the track gains are characterized at: used for the lock-time "
                        "stability self-check (and, later, optional gain derate below / buff above it)")
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
    p.add_argument('--track-zenith-zone-deg', type=float, default=3.0,
                   help="within this angle of the zenith, zero the az slew (chasing the az singularity "
                        "is futile); altitude tips over and we re-acquire once the target leaves the zone")
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
    detect_roles = {r.strip() for r in args.detect_roles.split(',') if r.strip()} & set(roles)
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
                    '--sky-lat', str(msite['lat_deg']), '--sky-lon', str(msite['lon_deg']),
                    '--sky-elev', str(msite['elev_m']), '--sky-epoch', str(msite['epoch_utc']),
                    '--sky-follow-state']
        if args.sky_tle_file:
            sky_args += ['--sky-tle-file', args.sky_tle_file, '--sky-target-mag', str(args.sky_target_mag)]
    playback_args = (['--playback-ser', args.playback_ser, '--playback-speed', str(args.playback_speed)]
                     + (['--playback-loop'] if args.playback_loop else [])
                     if args.source == 'playback' and args.playback_ser else [])
    estop = False
    recording = False
    gui_quit = False        # set when the GUI tells us it's closing (faster than watching it exit)

    # Auto-tracking state (pixel-space closed loop).
    tracker = None
    tracking = False
    track_role = None
    track_target = None
    track_seen_index = -1
    rad_per_px = (math.radians(args.arcsec_per_px / 3600.0) if args.arcsec_per_px > 0
                  else args.sky_pixel_um * 1e-3 / args.sky_focal_mm)
    # Per-role plate scale from the optics DB if a sensor+optic is named for that role; else the
    # sky-derived/override rad_per_px above. Print the resolved FoV so it's easy to sanity-check.
    _sensors, _optics, _reducers = optics.load_db()
    ds = max(1, args.sim_downscale)            # render sim cams at 1/ds resolution (same FoV)
    rad_per_px_by_role = {}
    render_by_role = {}        # role -> (res_x, res_y, pixel_um, focal_mm) for the sim sky cam
    fov_by_role = {}           # role -> (fov_x_deg, fov_y_deg) -> GUI nesting overlays
    for role in roles:
        rad_per_px_by_role[role] = rad_per_px
        render_by_role[role] = (args.width, args.height, args.sky_pixel_um, args.sky_focal_mm)
        sname, oname = getattr(args, f'{role}_sensor', None), getattr(args, f'{role}_optic', None)
        rname = getattr(args, f'{role}_reducer', None)
        try:
            if sname and oname:
                s, o = _sensors[sname], _optics[oname]
                mult = _reducers[rname] if rname else 1.0
                feff = o.focal_length_mm * mult
                # Downscale: 1/ds the pixels per axis, ds x the effective pixel pitch -> the FoV is
                # unchanged (chip size constant) but each rendered pixel spans ds x the sky. The
                # plate scale the tracker uses is per *rendered* pixel, so it scales by ds too.
                rx, ry = max(1, s.res_x // ds), max(1, s.res_y // ds)
                pum = s.pixel_um * ds
                rad_per_px_by_role[role] = optics.rad_per_px(pum, feff)
                render_by_role[role] = (rx, ry, pum, feff)
                fx, fy = optics.fov_deg(s, feff)      # physical FoV (full sensor; ds-invariant)
                fov_by_role[role] = (fx, fy)
                print(f"[backend] {role}: {sname} + {oname}{f' x{mult}' if mult != 1.0 else ''} -> "
                      f"{fx:.3f}x{fy:.3f} deg, {optics.arcsec_per_px(pum, feff):.3f} arcsec/px, "
                      f"render {rx}x{ry}{f' (downscale {ds}x)' if ds > 1 else ''}", flush=True)
        except KeyError as e:
            print(f"[backend] {role}: unknown optics {e}; using fallback plate scale", flush=True)
    zenith_zone_cos = math.sin(math.radians(args.track_zenith_zone_deg))   # |cos(alt)| below this = zone

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
        rx, ry, pum, fmm = render_by_role[role]       # per-role render size + sky optics (from the DB)
        per_role_sky = []
        if sources[role] == 'sky':
            per_role_sky = ['--sky-focal-mm', str(fmm), '--sky-pixel-um', str(pum)]
            if role in fov_by_role:                   # DB optics named -> render at the true sensor
                per_role_sky += ['--sky-width', str(rx), '--sky-height', str(ry)]   # res, so FoV matches
        cam_procs[role] = _spawn('astrolock.seeker.cam', [
            '--role', role, '--out-dir', session_dir, '--source', sources[role],
            '--width', str(rx), '--height', str(ry), '--fps', str(args.fps),
            '--frame-limit', str(args.segment_frames), '--file-limit', '-1',
            '--important', '1' if recording else '0', '--control-file', cf,
            *(['--auto'] if args.auto else []), *sky_args, *per_role_sky, *playback_args,
        ])
        control_writers[role].append({'important': 1 if recording else 0})

    detect_procs = {}

    def launch_detect(role):
        detect_procs[role] = _spawn('astrolock.seeker.detect',
                                    ['--session', session_dir, '--role', role, '--follow',
                                     '--stop-file', stop_file,
                                     '--detector', args.detector, '--doh-sigma', str(args.doh_sigma),
                                     '--snr', str(args.snr), '--max-candidates', str(args.max_candidates),
                                     '--min-blob-px', str(args.min_blob_px),
                                     '--tile-grid', str(args.tile_grid), '--per-tile', str(args.per_tile),
                                     '--device', args.device])

    # Pre-warm the skyfield ephemeris/star cache once, serially. Two sky cams starting together
    # otherwise race to download de421.bsp / hipparcos into the shared cache and one loses the
    # rename (WinError 5) -- which is what crashed both sim cams in a fresh worktree. Best-effort:
    # if it fails, the cams fall back to their own (racy) download exactly as before.
    if any(s == 'sky' for s in sources.values()):
        try:
            from astrolock.seeker import skysim
            skysim.ensure_cache()
        except Exception as e:
            print(f"[backend] ephemeris pre-warm skipped: {e}", flush=True)

    print(f"[backend] detect roles: {sorted(detect_roles) or 'none'}", flush=True)
    for role in roles:
        launch_cam(role)
        if role in detect_roles:
            launch_detect(role)
    gui_proc = _spawn('astrolock.seeker.gui',
                      ['--session', session_dir, '--wb-r', str(args.wb_r), '--wb-b', str(args.wb_b),
                       '--device', args.device]) \
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
        if role in detect_roles and (role not in detect_procs or detect_procs[role].poll() is not None):
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
        nonlocal estop, recording, tracking, track_role, tracker, track_seen_index, gui_quit
        t = cmd.get('type')
        if t == 'shutdown':                           # GUI is closing -> stop the whole session
            gui_quit = True
            return
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
            if role not in detect_roles:              # no detector on this role -> nothing to track
                print(f"[backend] ignoring track on {role!r} (no detector; see --detect-roles)", flush=True)
            elif role in roles and px and followers[role].header is not None:
                hdr = followers[role].header
                # Blobs are in frame image space; hold the target at the frame centre. rad_per_px
                # is per *sensor* pixel (from this role's optics), so scale by the cam's binning.
                rpp = rad_per_px_by_role.get(role, rad_per_px) * frame_binning(role)
                ft = frame_time_s(role, latest_det_index[role])    # clock off the frame, not wall time
                if ft is not None:
                    tracker = PixelTracker(hdr.image_width / 2.0, hdr.image_height / 2.0, rpp,
                                           ki=args.track_ki, damping=args.track_damping, kd=args.track_kd,
                                           kii=args.track_kii, nominal_rate_hz=args.track_nominal_rate,
                                           gate_px=args.track_gate_px, lost_s=args.track_lost_s,
                                           vel_smoothing=args.track_vel_smoothing,
                                           max_track_px_s=args.track_max_px_s, max_rate_rad_s=max_rate,
                                           sign_az=args.track_sign_az, sign_alt=args.track_sign_alt)
                    tracker.start(float(px[0]), float(px[1]), ft)
                    track_seen_index = latest_det_index[role]
                    tracking = True
                    track_role = role
                    estop = False
                    # Stability/bandwidth self-check at lock time, characterized at the nominal rate.
                    info, warns = tracker.diagnostics()
                    for ln in info:
                        print(f"[backend] track {role}: {ln}", flush=True)
                    for w in warns:
                        print(f"[backend] WARNING (track {role}): {w}", flush=True)
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

            st = mount.get_state()
            track_status = None
            if tracking and tracker is not None and not estop:
                role = track_role
                if latest_det_index[role] != track_seen_index:   # act once per new frame...
                    ft = frame_time_s(role, latest_det_index[role])
                    if ft is not None:                            # ...clocked by its capture time
                        track_seen_index = latest_det_index[role]
                        raz, ralt, track_status, tpx = tracker.update(latest_blobs[role], True, ft)
                        # Gimbal compensation: the image's response to an azimuth slew scales
                        # with cos(alt) -- it shrinks toward the zenith and reverses past it. So
                        # divide the az command by cos(alt) to keep the az loop gain constant (and
                        # correctly signed above 90 deg). Inside the zenith zone, just zero az:
                        # chasing the singularity is futile and would fling the mount around;
                        # altitude tips over and we re-acquire once the target leaves the zone.
                        ca = math.cos(st['alt_rad'])
                        raz = 0.0 if abs(ca) < zenith_zone_cos else raz / ca
                        mount.set_rates(raz, ralt)
                        track_target = list(tpx) if track_status == 'track' else None
                        if track_status == 'lost':
                            tracking = False

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
                'enc_t_mono_ns': st.get('t_mono_ns'),   # when the angles were measured (for extrapolation)
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
                'optics': {r: {'fov_x_deg': round(fv[0], 4), 'fov_y_deg': round(fv[1], 4)}
                           for r, fv in fov_by_role.items()},
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

            if gui_quit:
                print("[backend] gui requested shutdown; stopping", flush=True)
                break
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

        # Make sure every child is fully dead before cleanup: a still-terminating cam/detect holds
        # its .ser / sidecars open, and on Windows os.remove/rmtree fail on open files -- so a slow
        # (4K) detect that misses the graceful window used to leave whole sessions behind on exit.
        _reap(list(cam_procs.values()) + list(detect_procs.values()) + [gui_proc])

        _cleanup(session_dir, keep=args.keep, clean=clean)
        print("[backend] done", flush=True)


def _reap(procs, graceful_s=4.0, kill_s=3.0):
    """Wait for child processes to exit, escalating wait -> terminate -> kill, and *waiting after
    each escalation*. The wait is the point: a process that's been signalled is not dead yet, and on
    Windows its open .ser / sidecar handles block os.remove/rmtree until it actually exits. A Ctrl-C
    during the wait means "stop waiting, just kill" -- so we don't dump a traceback from inside it."""
    procs = [p for p in procs if p is not None]
    try:
        deadline = time.perf_counter() + graceful_s
        for p in procs:                          # phase 1: let them stop on their own (stop files)
            try:
                p.wait(timeout=max(0.0, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired:
                pass
        alive = [p for p in procs if p.poll() is None]
        for p in alive:                          # phase 2: ask the stragglers to terminate...
            p.terminate()
        deadline = time.perf_counter() + kill_s
        for p in alive:                          # ...then wait for the handles to be released
            try:
                p.wait(timeout=max(0.0, deadline - time.perf_counter()))
            except subprocess.TimeoutExpired:
                try:
                    p.kill()                     # phase 3: force, and reap
                    p.wait(timeout=1.0)
                except Exception:
                    pass
    except KeyboardInterrupt:
        for p in procs:                          # impatient second Ctrl-C: don't wait, just kill
            try:
                p.kill()
            except Exception:
                pass


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
