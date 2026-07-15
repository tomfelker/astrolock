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

Defaults are our rig: both the guide (ASI678MC + 8mm) and main (ASI678MC + CPC 1100) sim cameras
on the ISS sky pass, with the GUI. Bin a sim camera (--<role>-bin N) for faster/smaller views --
a sim downscale and NxN binning render identically.

    python -m astrolock.seeker.backend                       # the two-camera rig, defaults
    python -m astrolock.seeker.backend --guide-bin 4         # guide at 1/4 render res (faster)
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

from astrolock.seeker import cam as cam_mod
from astrolock.seeker import control
from astrolock.seeker import framestream
from astrolock.seeker import mount as mount_mod
from astrolock.seeker import session as session_mod
from astrolock.seeker import sidecar
from astrolock.seeker import optics
from astrolock.seeker.follower import SerFollower
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def _spawn(module, args):
    """Launch `python -m <module> <args>` using the same interpreter, in the repo cwd."""
    return subprocess.Popen([sys.executable, '-m', module, *args])


# Bortle dark-sky class -> approximate zenith sky surface brightness (V mag/arcsec^2). Higher = darker.
_BORTLE_SKY_MAG = {1: 21.9, 2: 21.8, 3: 21.5, 4: 21.0, 5: 20.4, 6: 19.5, 7: 18.9, 8: 18.4, 9: 18.0}


def bortle_to_sky_mag(zone):
    """Sky surface brightness (mag/arcsec^2) for a Bortle class 1..9 (clamped)."""
    return _BORTLE_SKY_MAG[int(max(1, min(9, round(zone))))]


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker backend / orchestrator")
    p.add_argument('--roles', default='guide,main', help="comma-separated camera roles to launch")
    p.add_argument('--source', default='sky', choices=['synthetic', 'zwo', 'sky', 'playback'],
                   help="default 'sky' runs the baked-in ISS test pass; 'synthetic' needs no deps; "
                        "'playback' replays a recorded .ser through the live pipeline")
    p.add_argument('--playback-ser', default=None, help="playback source: the .ser file to replay")
    p.add_argument('--playback-speed', type=float, default=1.0, help="playback source: speed multiplier")
    p.add_argument('--playback-loop', action='store_true', help="playback source: loop instead of stopping")
    p.add_argument('--playback-fps', type=float, default=10.0,
                   help="playback source: frame-rate cap (0 = unlimited); the only pacing for a "
                        "recording with no timestamp sidecar")
    p.add_argument('--detect-roles', default='guide,main',
                   help="comma-separated roles to run detection on. Default both: the guide runs the "
                        "multi-blob detector (--detector) for acquisition, the main runs its "
                        "single-extended-target detector (--main-detector) for the handoff.")
    p.add_argument('--detector', default='doh',
                   choices=['bandpass', 'doh', 'surprise', 'extended', 'circmean'],
                   help="detection surface: 'doh' (default) = determinant of the Hessian, 'bandpass', "
                        "'surprise' (per-pixel temporal surprise + decaying peak-hold; finds faint fast "
                        "movers / satellite trails), 'extended' (symmetric single-extended-target "
                        "marginal+compactness detector for the main cam), or 'circmean' (toroidal-COM "
                        "single-target detector: band-pass + circular-mean Rayleigh test; robust to "
                        "hollow/dim targets + hot-pixel fields). Per-role override: --<role>-detector")
    p.add_argument('--doh-sigma', type=float, default=0.0, help="doh detector: Gaussian scale px (0 = psf default)")
    p.add_argument('--surprise-decay', type=float, default=0.85,
                   help="surprise detector: trail peak-hold decay (->1 integrates longer for fainter trails)")
    p.add_argument('--surprise-blur-px', type=float, default=1.5,
                   help="surprise detector: spatial low-pass sigma (px) on the surprisal map (0 = off)")
    p.add_argument('--tile-size', type=int, default=128,
                   help="surprise detector: tile edge (px) for the per-tile extremum pass")
    p.add_argument('--tile-fixed-nsigma', type=float, default=8.0,
                   help="surprise detector: stdevs from the tile mean to keep a fixed (running-mean) extremum")
    p.add_argument('--tile-moving-nsigma', type=float, default=5.0,
                   help="surprise detector: stdevs from the tile mean to keep a moving (surprisal) extremum")
    p.add_argument('--tile-mask-px', type=int, default=16,
                   help="surprise detector: mask radius (px) around fixed extrema when hunting movers")
    p.add_argument('--track-detector', default='peak', choices=['peak', 'matched'],
                   help="tracking-phase detector (single answer inside the predicted ROI; independent "
                        "of the acquisition --detector): 'peak' = the original surface peak with a "
                        "brightness found-test, 'matched' = stateless matched filter + centre pull + "
                        "argmax, no found/lost gate. Per-role: --<role>-track-detector")
    p.add_argument('--guide-track-detector', default=None, choices=['peak', 'matched'],
                   help="override --track-detector for the guide role")
    p.add_argument('--main-track-detector', default=None, choices=['peak', 'matched'],
                   help="override --track-detector for the main role")
    p.add_argument('--track-blur-px', type=float, default=1.5,
                   help="matched track detector: matched-filter Gaussian sigma (px), ~the PSF width")
    p.add_argument('--track-pull', type=float, default=0.5,
                   help="matched track detector: cone pull (sigma per work px of distance from the "
                        "predicted position)")
    p.add_argument('--debug-detect-ser', action='store_true',
                   help="have each detector also write <seg>_<role>_debug.ser -- a movie of its detection "
                        "surface, followable in the GUI/any SER viewer for tuning")
    p.add_argument('--snr', type=float, default=6.0, help="detect: report peaks this many sigma above background")
    p.add_argument('--max-candidates', type=int, default=16, help="detect: max blobs reported per frame")
    p.add_argument('--min-blob-px', type=int, default=2, help="detect: ignore peaks smaller than this")
    p.add_argument('--tile-grid', type=int, default=8,
                   help="detect density cap: tiles across the frame (0 = off); keeps targets from "
                        "a dense bright region from starving the rest of the frame")
    p.add_argument('--per-tile', type=int, default=2, help="detect density cap: max blobs per tile")
    # Extended detector ('extended') knobs -- the symmetric single-target marginal+compactness detector.
    p.add_argument('--ext-nsigma', type=float, default=3.0,
                   help="extended detector: two-tailed |value-mean| marginal threshold in sigmas "
                        "(catches bright AND dark targets; lower = includes dimmer target extremities)")
    p.add_argument('--ext-density', type=float, default=0.7,
                   help="extended detector: min compactness to call a target present (~1 = tightly "
                        "clumped, ~0.5 = uniform noise), in both axes")
    # Circmean detector ('circmean') knobs -- the toroidal-COM single-target detector (main-cam).
    p.add_argument('--circ-down', type=int, default=4,
                   help="circmean detector: extra downscale before detection (denoise + speed)")
    p.add_argument('--circ-bg-radius', type=int, default=64,
                   help="circmean detector: box-blur background radius in DOWNSCALED px (kills vignette)")
    p.add_argument('--circ-nsigma', type=float, default=5.0,
                   help="circmean detector: two-tailed residual threshold in robust (MAD) sigmas (~5)")
    p.add_argument('--circ-z', type=float, default=15.0,
                   help="circmean detector: present when the Rayleigh min(Zx,Zy)=n*R^2 clears this "
                        "(noise p ~ e^-Z/axis; real targets Z=60..8000, black sky <=4)")
    # Focus/collimation process (one optional, toggled live from the GUI Focus tab).
    p.add_argument('--focus-crop', type=int, default=127,
                   help="focus: star crop size (px, forced odd) -- the EMA star image "
                        "(big enough for a defocused donut, not just the in-focus core)")
    p.add_argument('--focus-search', type=int, default=256,
                   help="focus: search ROI around the target to find the peak in (px)")
    p.add_argument('--focus-alpha', type=float, default=0.05,
                   help="focus: EMA rate for the star crop (bigger = faster to settle, noisier)")
    p.add_argument('--device', default='auto',
                   help="torch device for detect + gui: 'auto' (default) = cuda if present else cpu, "
                        "or force 'cpu' / 'cuda' (passed through to both child processes)")
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=0.0,
                   help="cam-loop max framerate cap (0 = unlimited, the default -- a real camera "
                        "self-paces by exposure + readout + USB; this is just an optional throttle, "
                        "settable live per role in the GUI). Sim/playback have their own pacing too.")
    p.add_argument('--bit-depth', type=int, default=16, choices=[8, 16],
                   help="capture container depth: 8 = RAW8 (half the USB bytes/frame -> higher fps), "
                        "16 = full precision (12-bit ASI / 12-bit sim). Per-role, relaunch on change.")
    p.add_argument('--sim-downscale', type=int, default=1,
                   help="DEPRECATED: folded into binning (a sim downscale is an NxN bin). "
                        "Prefer --<role>-bin; this multiplies into every role's bin.")
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
    # guide/main plate scale, optics, and binning
    p.add_argument('--sky-pixel-um', type=float, default=2.0, help="guide sensor pixel pitch (um)")
    p.add_argument('--arcsec-per-px', type=float, default=0.0,
                   help="guide plate scale override (0 = derive from --sky-pixel-um / --sky-focal-mm)")
    # Per-role plate scale from the vendored optics DB (names from `python -m astrolock.seeker.optics`).
    p.add_argument('--guide-sensor', default='ZWO ASI678MC', help="guide camera sensor name in the optics DB")
    p.add_argument('--guide-optic', default='8mm CS f/1.4', help="guide optic name in the optics DB")
    p.add_argument('--guide-reducer', default=None, help="guide reducer/barlow name (optional)")
    p.add_argument('--main-sensor', default='ZWO ASI678MC', help="main camera sensor name in the optics DB")
    p.add_argument('--main-optic', default='Celestron CPC 1100', help="main optic name in the optics DB")
    p.add_argument('--main-reducer', default=None, help="main reducer/barlow name (optional)")
    # Per-role detector override (default: --detector). Typical: main-cam extended-target tracking.
    p.add_argument('--guide-detector', default=None,
                   choices=['bandpass', 'doh', 'surprise', 'extended', 'circmean'],
                   help="override --detector for the guide role (default: --detector)")
    p.add_argument('--main-detector', default='extended',
                   choices=['bandpass', 'doh', 'surprise', 'extended', 'circmean'],
                   help="detector for the main role (default 'extended': the narrow-field single-target "
                        "detector that drives the guide->main handoff; 'circmean' is the newer, more "
                        "robust single-target detector)")
    p.add_argument('--guide-bin', type=int, default=2,
                   help="guide camera NxN binning (default 2x2): combine NxN pixels. A color cam "
                        "binned 2x2 reads out mono at ~the debayered resolution -- the natural guide "
                        "format, and it saves USB/SSD/memory bandwidth. Sim renders mono at 1/N res.")
    p.add_argument('--main-bin', type=int, default=1,
                   help="main camera NxN binning (default 1 = native full resolution)")
    # Tracking: sky-space closed loop (TargetModel + min-time-intercept servo). A blind, measurement-
    # driven model of the target's motion; the servo commands the mount to intercept its prediction.
    p.add_argument('--track-rate-smoothing-s', type=float, default=1.0,
                   help="EMA time constant for the target angular-velocity estimate "
                        "(bigger = smoother but laggier rate; smaller = snappier but noisier). "
                        "Live-adjustable in the GUI Tracking panel")
    p.add_argument('--track-model', default='greatcircle', choices=['ema', 'greatcircle'],
                   help="target motion model at lock: 'ema' = constant angular velocity across "
                        "the SKY (right for anything far); 'greatcircle' = constant-altitude "
                        "great circle about the Earth's centre (LEO passes: the zenith speed-up "
                        "is perspective the geometry produces for free). GUI: Tracking > Model")
    p.add_argument('--track-alt-km', type=float, default=415.0,
                   help="greatcircle model: assumed target altitude (km)")
    p.add_argument('--track-min-intercept-s', type=float, default=1.0,
                   help="min intercept time; also sets the position-correction stiffness (P ~ 1/this)")
    p.add_argument('--track-command-latency-s', type=float, default=0.0,
                   help="assumed delay before a rate command takes effect (s). ~0 for a direct mount; "
                        "only worth setting for a slow serial link (e.g. 9600-baud NexStar)")
    p.add_argument('--track-max-horizon-s', type=float, default=8.0,
                   help="declare the target uncatchable if no intercept is reachable within this long")
    p.add_argument('--track-gate-px', type=float, default=80.0, help="max px to associate a blob to the target")
    p.add_argument('--track-lost-s', type=float, default=1.5, help="give up tracking after this long unmatched")
    p.add_argument('--track-lock-min-time', type=float, default=1.0,
                   help="coast-on-loss: how long (s) a lock must hold before losing it coasts (keep "
                        "slewing the extrapolation to re-acquire) rather than stops")
    p.add_argument('--track-roi-size', type=int, default=128,
                   help="while tracking, publish a square ROI (this many frame px) around the predicted "
                        "target so detect can work just that window instead of the whole frame. 0 = full-frame.")
    p.add_argument('--track-delay-s', type=float, default=0.3,
                   help="after a new lock, hold the mount still for this long while the tracker learns "
                        "the target's angular velocity -- a better estimate before the catch-up slew "
                        "means a better chance of re-acquiring if the slew loses it. Live-adjustable "
                        "in the GUI Tracking panel")
    p.add_argument('--track-sign-az', type=float, default=1.0, help="flip if az moves the image the wrong way")
    p.add_argument('--track-sign-alt', type=float, default=-1.0, help="flip if alt moves the image the wrong way")
    # Camera handoff: guide acquires, then promote the sky-space loop to a finer role (main) running the
    # extended detector once it stably sees the target -- closing the loop in main px self-corrects the
    # guide->main boresight and rides an extended target's steadier centroid. Demote back on loss.
    p.add_argument('--handoff', dest='handoff', action='store_true', default=True,
                   help="promote tracking to the main cam once its extended detector locks (default on)")
    p.add_argument('--no-handoff', dest='handoff', action='store_false',
                   help="disable main-cam handoff -- track on the acquisition (guide) cam only")
    p.add_argument('--handoff-promote-s', type=float, default=1.5,
                   help="seconds the fine cam must continuously see the target before promoting to it")
    p.add_argument('--handoff-demote-s', type=float, default=0.4,
                   help="demote back to the guide when the evidence accumulator falls to this (s)")
    p.add_argument('--handoff-fall-rate', type=float, default=3.0,
                   help="how many seconds of accumulated evidence are shed per second the fine cam has no "
                        "target (>1 = demote faster than promote, so a brief flicker won't drop the lock)")
    p.add_argument('--boresight-x-mrad', type=float, default=0.0,
                   help="main-vs-guide boresight X offset (mrad, guide image right): where the main cam "
                        "points relative to the guide, so a guide-tracked target is centred in the main "
                        "cam. Tune live in the GUI (Boresight).")
    p.add_argument('--boresight-y-mrad', type=float, default=0.0,
                   help="boresight Y offset (mrad, guide image down)")
    p.add_argument('--track-debug', action='store_true',
                   help="print per-frame commanded vs measured axis rates and target offset")
    p.add_argument('--sky-tle-file', default='data/iss_25544.tle',
                   help="sky source: satellite TLE file (default: the ISS)")
    p.add_argument('--sky-target-mag', type=float, default=-4.0, help="sky source: satellite magnitude")
    p.add_argument('--sky-rate-az', type=float, default=0.0, help="sky source: scripted az slew (deg/s)")
    p.add_argument('--sky-rate-alt', type=float, default=0.0, help="sky source: scripted alt slew (deg/s)")
    p.add_argument('--sky-substeps', type=int, default=6, help="sky source: substeps per exposure")
    p.add_argument('--sky-exposure-s', type=float, default=0.1, help="sky source: simulated exposure (s)")
    p.add_argument('--sim-cam-noop', action='store_true',
                   help="sky source: skip the frame simulation (canned noise frame) but keep all the "
                        "pacing, so the cams write at full commanded speed -- reproduces the real rig's "
                        "sustained-write disk pressure. Pair with --fps 999 --sky-exposure-s 0.002")
    p.add_argument('--sim-cam-bandwidth-limit', type=float, default=400.0,
                   help="sky source: model of the camera data link (MB/s; default ~USB3, capping "
                        "full-res frames at ~24 fps like the real hardware). 0 = unlimited")
    p.add_argument('--shm-ser', dest='shm_ser', action='store_true', default=True,
                   help="non-recording cam segments live in shared memory (default): only a "
                        "178-byte marker .ser lands on disk, so idle streaming stops grinding the "
                        "SSD. Recording segments always write real files")
    p.add_argument('--no-shm-ser', dest='shm_ser', action='store_false',
                   help="write every segment to disk (the old behavior)")
    p.add_argument('--shm-frames', type=int, default=128,
                   help="shm segments: frames per segment (committed RAM; rolls at this cadence)")
    p.add_argument('--sky-focal-mm', type=float, default=8.0, help="sky source: lens focal length (mm)")
    p.add_argument('--sky-psf-wavelength-nm', type=float, default=550.0,
                   help="sky source: wavelength for the physically-sized Airy-disc PSF (nm)")
    p.add_argument('--sky-psf-sigma-px', type=float, default=2.0,
                   help="sky source: per-camera residual-blur Gaussian (lens softness/defocus) on top of the "
                        "Airy disc, px. Default 2.0 softens the sub-pixel guide PSF so tracking isn't jumpy; "
                        "pass 0 for the sharp physical-PSF worst case. Live-adjustable per camera in the GUI "
                        "(the sky camera's Defocus control)")
    p.add_argument('--sky-seeing-r0-m', type=float, default=0.0,
                   help="sky source: atmospheric Fried parameter r0 (m); >0 adds a seeing blur to all sim cams")
    p.add_argument('--sky-bortle', type=int, default=4,
                   help="sky source: Bortle dark-sky class 1..9 (sky brightness); 1 = pristine, 9 = inner city. "
                        "With a physical sensor (QE/full-well in the DB) this sets the sim sky background. "
                        "Live-adjustable in the GUI Simulation panel")
    p.add_argument('--wb-r', type=float, default=1.24, help="GUI display-only WB gain for red (1 = none)")
    p.add_argument('--wb-b', type=float, default=1.98, help="GUI display-only WB gain for blue (1 = none)")
    p.add_argument('--gui', dest='gui', action='store_true', default=True)
    p.add_argument('--no-gui', dest='gui', action='store_false')
    p.add_argument('--duration', type=float, default=0.0, help="stop after N seconds (0 = until Ctrl-C)")
    p.add_argument('--keep', action='store_true', help="keep all captured files on exit")
    p.add_argument('--recordings-dir', default='recordings',
                   help="archive directory for the recorder processes (never cleaned up)")
    p.add_argument('--list-optics', action='store_true',
                   help="list the sensors / optics / reducers in the DB (the valid names for "
                        "--{role}-sensor / --{role}-optic / --{role}-reducer) and exit")
    args = p.parse_args(argv)

    if args.list_optics:
        optics.list_db()
        return

    roles = [r.strip() for r in args.roles.split(',') if r.strip()]
    detect_roles = {r.strip() for r in args.detect_roles.split(',') if r.strip()} & set(roles)
    session_dir, ts = session_mod.new_session_dir()
    stop_file = os.path.join(session_dir, 'stop')          # for detect (cams use control files)

    cmd_server = control.CommandServer()
    # State is a latest-wins shm slot, not a file: the ~20 Hz append used to ride the control
    # loop, and a volume-wide write throttle (drive breather) could stall the loop that has to
    # keep processing estop. backend.json advertises the slot; cams/detect get it by argument.
    state_slot = framestream.LatestSlot(create=True)
    with open(os.path.join(session_dir, f"{ts}_backend.json"), 'w') as f:
        json.dump({'command_host': cmd_server.host, 'command_port': cmd_server.port,
                   'state_shm': state_slot.name}, f)
    print(f"[backend] session {session_dir} roles={roles} source={args.source} "
          f"cmd_port={cmd_server.port}", flush=True)

    max_rate = math.radians(args.max_rate_deg_s)
    site = {'lat_deg': args.lat, 'lon_deg': args.lon, 'elev_m': args.elev, 'epoch_utc': args.epoch}
    # The sim mount writes its ground-truth trajectory here for the sim camera to follow (truth, not
    # the backend's estimate). Harmless/ignored for the real mount.
    sim_mount_path = os.path.join(session_dir, session_mod.sim_mount_name(ts))
    mount = mount_mod.make_mount(
        args.mount, az0_rad=math.radians(args.start_az_deg), alt0_rad=math.radians(args.start_alt_deg),
        site=site, max_rate_rad_s=max_rate, accel_rad_s2=math.radians(args.mount_accel_deg_s2),
        update_hz=args.mount_update_hz, url=args.mount_url, sidecar_path=sim_mount_path)
    msite = mount.get_site()        # GPS/site comes from the mount; it drives the sky-sim camera
    # Mount lifecycle (GUI connect/disconnect). The site is captured here from the startup mount and
    # feeds the sky-sim once; switching mounts later doesn't re-derive it (fine -- the sky sim is the
    # sim workflow, and a real mount's site is a TODO fallback anyway).
    mount_desired_url = 'sim' if args.mount == 'sim' else (args.mount_url or 'sim')
    mount_connected = True                                  # the CLI-selected mount is live at startup
    mounts_available = mount_mod.available_mount_urls()     # detected Celestron COM ports (GUI chooser)

    sky_args = []
    almanac_path = os.path.join(session_dir, f"{ts}_almanac.jsonl")       # sky_sim publishes here
    if args.source == 'sky':
        # Sky *positions* come from the shared sky_sim almanac (one propagator, one system clock) -- not
        # from each cam, which used to drift apart. The follow flag (which mount trajectory to render
        # from) is added per-launch by sky_follow_flag(), so switching mounts in the GUI re-points them.
        sky_args = ['--sky-rate-az', str(args.sky_rate_az), '--sky-rate-alt', str(args.sky_rate_alt),
                    '--sky-substeps', str(args.sky_substeps), '--sky-exposure-s', str(args.sky_exposure_s),
                    '--sky-almanac', almanac_path,
                    '--sim-cam-bandwidth-limit', str(args.sim_cam_bandwidth_limit)] \
            + (['--sim-cam-noop'] if args.sim_cam_noop else [])

    def sky_follow_flag():
        """Which mount trajectory a sky-sim cam renders from, chosen per-launch from the CURRENT mount:
        a SimMount publishes an exact ground-truth sidecar (--sky-follow-mount); a real or disconnected
        mount has none, so follow the backend's published encoder estimate (--sky-follow-state). Evaluated
        at each (re)launch so connecting the real mount in the GUI re-points the sky cams -- otherwise a
        run started on the sim keeps following the now-stale sim sidecar and the sky freezes."""
        return '--sky-follow-mount' if isinstance(mount, mount_mod.SimMount) else '--sky-follow-state'
    # Per-role playback source (.ser to replay + loop), so a role can be switched to playback live from
    # the GUI. Seeded from the launch args; launch_cam builds the per-role --playback-* flags.
    playback_by_role = {role: {'ser': args.playback_ser, 'loop': bool(args.playback_loop)} for role in roles}
    estop = False
    recording = {role: False for role in roles}   # per-role: a recorder process is running
    rec_procs = {}                                # role -> recorder subprocess

    def _recorder_alive(role):
        pr = rec_procs.get(role)
        return pr is not None and pr.poll() is None
    gui_quit = False        # set when the GUI tells us it's closing (faster than watching it exit)

    # Auto-tracking state (sky-space closed loop).
    tracker = None
    tracking = False
    coasting = False          # lost a settled lock -> holding the last rate, still re-acquiring
    # Follow = drive the mount to hold a lock. PERSISTENT user intent (not per-lock): settable any
    # time from the GUI, so tracking can be engaged watch-only (lock + estimate, mount still).
    follow_enabled = True
    track_delay_s = max(0.0, args.track_delay_s)   # hold the mount this long after a new lock (learn v first)
    track_started_t = 0.0     # perf_counter at the current lock's start (for the delay)
    track_role = None         # the role whose detections currently feed the sky-space model (guide or main)
    track_target = None       # model-predicted target px in the source frame (the smooth blue pipper)
    track_detect_px = None    # raw detection px of the locked target in the source frame (the green marker)
    track_seen_det = None          # (seg ident, index) of the last detection acted on -- NEVER a
                                   # bare index: indices restart per segment; alone they can't name a frame
    track_center = None       # (cx, cy) true frame centre at lock time, for live boresight re-offset
    # Camera handoff (guide acquires -> promote to the narrow main cam once it stably sees the target,
    # closing the loop directly in main px so the guide->main boresight error self-corrects; demote back
    # on loss). A time-based leaky integrator (framerate-independent) with asymmetric rise/fall + Schmitt
    # hysteresis, fed by the fine cam's extended-detector 'present'. track_primary is the fallback role.
    track_primary = None
    handoff_fine = None       # the finer role to promote to (e.g. 'main'), or None if none is eligible
    handoff_conf = 0.0        # seconds-of-evidence accumulator, clamped to [0, --handoff-promote-s]
    track_pref = 'auto'       # GUI radio: 'guide' (pin), 'main' (prefer), or 'auto' (hysteresis handoff)
    # Boresight: where the main cam points relative to the guide, as an image-frame angular offset
    # (x right, y down), radians. Offsets the guide tracker's hold point so the target lands in the main.
    boresight = {'x': args.boresight_x_mrad * 1e-3, 'y': args.boresight_y_mrad * 1e-3}
    rad_per_px = (math.radians(args.arcsec_per_px / 3600.0) if args.arcsec_per_px > 0
                  else args.sky_pixel_um * 1e-3 / args.sky_focal_mm)
    # Per-role plate scale from the optics DB if a sensor+optic is named for that role; else the
    # sky-derived/override rad_per_px above. Print the resolved FoV so it's easy to sanity-check.
    _sensors, _optics, _reducers = optics.load_db()
    ds = max(1, args.sim_downscale)            # DEPRECATED: folded into binning (a sim downscale *is* a bin)
    if ds > 1:
        print(f"[backend] note: --sim-downscale {ds} is deprecated; folding it into binning. A sim "
              f"downscale and NxN binning render identically -- prefer --<role>-bin.", flush=True)
    rad_per_px_by_role = {}
    render_by_role = {}        # role -> (res_x, res_y, pixel_um, focal_mm) for the sim sky cam
    fov_by_role = {}           # role -> (fov_x_deg, fov_y_deg) -> GUI nesting overlays
    bin_by_role = {}           # role -> physical NxN bin (recorded in frame metadata; scales plate scale)
    fps_by_role = {role: args.fps for role in roles}    # role -> cam-loop max-fps cap (0 = unlimited); live
    depth_by_role = {role: args.bit_depth for role in roles}   # role -> capture bit depth (8 or 16); relaunch
    roi_by_role = {role: None for role in roles}        # role -> None (full) or (out_w, out_h) centered readout
                                                        # window in binned/output px (GUI picks square pow2)
    full_res_by_role = {}      # role -> (res_x, res_y) native px of the un-cropped sensor (for ROI sizing)
    roi_window_by_role = {}    # role -> [x0, y0, w, h] in native sensor px (provenance -> frame metadata)
    aperture_by_role = {role: 0.0 for role in roles}    # role -> objective aperture mm (sim Airy-disc PSF)
    obstruction_by_role = {role: 0.0 for role in roles} # role -> central-obstruction diameter ratio (obstructed PSF)
    vanes_by_role = {role: 0 for role in roles}         # role -> spider-vane count (Newtonian diffraction spikes)
    vane_width_by_role = {role: 0.0 for role in roles}  # role -> vane width / aperture diameter
    qe_by_role = {role: 0.0 for role in roles}          # role -> sensor peak QE (sim physical-flux model)
    full_well_by_role = {role: 0.0 for role in roles}   # role -> sensor full-well capacity (e-)
    read_noise_by_role = {role: 0.0 for role in roles}  # role -> sensor read noise (e-)
    optics_sel = {role: [None, None, None] for role in roles}   # role -> [sensor, optic, reducer] in effect

    def _geom(role, s, feff, aperture_mm=0.0, obstruction=0.0, vanes=0, vane_width_frac=0.0):
        aperture_by_role[role] = aperture_mm            # objective aperture -> physically-sized diffraction PSF
        obstruction_by_role[role] = obstruction         # + central obstruction / spider -> obstructed/vaned PSF
        vanes_by_role[role] = vanes
        vane_width_by_role[role] = vane_width_frac
        qe_by_role[role] = s.qe                         # sensor photometry -> physical exposure model
        full_well_by_role[role] = s.full_well_e
        read_noise_by_role[role] = s.read_noise_e
        """The one place geometry is computed: render size / plate scale / FoV / readout window for a
        role from its binning (bin_by_role) and ROI window (roi_by_role), given DB sensor `s`
        at effective focal length `feff`. Startup and every live bin/ROI/optics change call this.

        Binning shrinks the rendered frame 1/b per axis at b x the pixel pitch (same FoV), reported as
        frame_binning so the tracker's rad_per_px * bin recovers the true scale -- exactly a real binned
        camera. ROI is a *centered crop* of the sensor: same plate scale, fewer pixels, smaller FoV (the
        boresight stays at frame centre, so the tracker needs no offset). Returns the rendered (rx, ry)."""
        b = max(1, bin_by_role.get(role, 1))
        roi = roi_by_role.get(role)                       # (out_w, out_h) in binned px, or None = full sensor
        if roi:
            rx, ry = max(1, min(int(roi[0]), s.res_x // b)), max(1, min(int(roi[1]), s.res_y // b))
        else:
            rx, ry = max(1, s.res_x // b), max(1, s.res_y // b)                        # rendered (binned)
        roi_w, roi_h = rx * b, ry * b                                                  # centered native window
        x0, y0 = (s.res_x - roi_w) // 2, (s.res_y - roi_h) // 2
        rad_per_px_by_role[role] = optics.rad_per_px(s.pixel_um, feff)                 # per native px
        render_by_role[role] = (rx, ry, s.pixel_um * b, feff)
        roi_window_by_role[role] = [x0, y0, roi_w, roi_h]
        full_res_by_role[role] = (s.res_x, s.res_y)                                    # for ROI-size choices
        pmm = s.pixel_um / 1000.0
        fov_by_role[role] = (math.degrees(2 * math.atan(roi_w * pmm / (2 * feff))),
                             math.degrees(2 * math.atan(roi_h * pmm / (2 * feff))))
        return rx, ry

    for role in roles:
        b = max(1, getattr(args, f'{role}_bin', 1)) * ds        # deprecated downscale folds straight in
        bin_by_role[role] = b
        rad_per_px_by_role[role] = rad_per_px
        render_by_role[role] = (max(1, args.width // b), max(1, args.height // b),
                                args.sky_pixel_um * b, args.sky_focal_mm)
        full_res_by_role[role] = (args.width, args.height)     # sim's native res (no DB sensor picked yet)
        sname, oname = getattr(args, f'{role}_sensor', None), getattr(args, f'{role}_optic', None)
        rname = getattr(args, f'{role}_reducer', None)
        try:
            if sname and oname:
                s, o = _sensors[sname], _optics[oname]
                mult = _reducers[rname] if rname else 1.0
                feff = o.focal_length_mm * mult
                optics_sel[role] = [sname, oname, rname]
                rx, ry = _geom(role, s, feff, o.aperture_mm, o.obstruction_frac, o.spider_vanes, o.vane_width_frac)
                fx, fy = fov_by_role[role]
                extra = f" (bin {b}x{b})" if b > 1 else ""
                print(f"[backend] {role}: {sname} + {oname}{f' x{mult}' if mult != 1.0 else ''} -> "
                      f"{fx:.3f}x{fy:.3f} deg, {optics.arcsec_per_px(s.pixel_um * b, feff):.3f} arcsec/px, "
                      f"render {rx}x{ry}{extra}", flush=True)
                # Binning notes (only where the DB actually says color vs mono -- no guessing):
                #  - a color cam feeding detection but left unbinned wastes bandwidth (detection bins
                #    it to mono anyway);
                #  - binning a mono cam is a real resolution cut (a color 2x2 ~= its debayered res, but
                #    mono has no Bayer cell to fold in).
                color = s.is_color
                if color is True and b == 1 and role in detect_roles:
                    print(f"[backend] note: {role} is color ({s.bayer}) and unbinned but feeds detection; "
                          f"it'll be binned to mono for detection anyway -- --{role}-bin 2 halves bandwidth",
                          flush=True)
                elif color is False and b > 1:
                    print(f"[backend] note: {role} is mono ({sname}) binned {b}x{b} -- a true 1/{b} "
                          f"resolution cut (no Bayer cell to fold in); fine if you're not resolution-limited",
                          flush=True)
        except KeyError as e:
            print(f"[backend] {role}: unknown optics {e}; using fallback plate scale", flush=True)

    def resolve_optics(role, sname, oname, rname):
        """Re-resolve a role's plate scale + sim render size + FoV + readout window from the optics DB
        (via _geom, the shared geometry) and record the selection. True if a known sensor+optic named."""
        optics_sel[role] = [sname or None, oname or None, rname or None]
        if not (sname and oname and sname in _sensors and oname in _optics):
            return False
        o = _optics[oname]
        mult = _reducers.get(rname, 1.0) if rname else 1.0
        _geom(role, _sensors[sname], o.focal_length_mm * mult, o.aperture_mm,
              o.obstruction_frac, o.spider_vanes, o.vane_width_frac)
        return True

    def recompute_render(role):
        """Recompute a role's render geometry after a bin/ROI change. Uses the DB optics if named,
        else falls back to binning the configured frame (ROI is DB-optics-only for now)."""
        if not resolve_optics(role, *optics_sel[role]):
            b = max(1, bin_by_role.get(role, 1))
            render_by_role[role] = (max(1, args.width // b), max(1, args.height // b),
                                    args.sky_pixel_um * b, args.sky_focal_mm)
            roi_window_by_role.pop(role, None)
            aperture_by_role[role] = obstruction_by_role[role] = vane_width_by_role[role] = 0.0
            vanes_by_role[role] = 0
            qe_by_role[role] = full_well_by_role[role] = read_noise_by_role[role] = 0.0

    sources = {role: args.source for role in roles}      # switchable live (sim <-> real)
    # Per-role sim residual-blur sigma (px): lens softness/defocus atop the physical Airy PSF -- a
    # per-camera optical property (each sim cam is a different lens), so it lives here, not globally.
    # None = auto (sharp physical PSF); the GUI exposes it as the sky cam's 'Defocus' control.
    psf_sigma_by_role = {role: args.sky_psf_sigma_px for role in roles}
    camera_url = {role: None for role in roles}           # per-role selected ZWO camera URL (None = default)
    cams_available = cam_mod.zwo_camera_urls()            # detected ZWO cameras (model-qualified URLs), [] if none
    cam_caps = {role: None for role in roles}             # role -> {'source','controls':[...]} from caps_<role>.json
    caps_mtime = {}                                       # role -> last-read mtime of its caps file
    cam_control_vals = {role: {} for role in roles}       # role -> {control name: value}; live user changes
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
            per_role_sky = ['--sky-focal-mm', str(fmm), '--sky-pixel-um', str(pum),
                            '--sky-aperture-mm', str(aperture_by_role.get(role, 0.0)),
                            '--sky-central-obstruction', str(obstruction_by_role.get(role, 0.0)),
                            '--sky-spider-vanes', str(vanes_by_role.get(role, 0)),
                            '--sky-vane-width-frac', str(vane_width_by_role.get(role, 0.0)),
                            '--sky-qe', str(qe_by_role.get(role, 0.0)),
                            '--sky-full-well-e', str(full_well_by_role.get(role, 0.0)),
                            '--sky-read-noise-e', str(read_noise_by_role.get(role) or 2.0),
                            '--sky-sky-mag', str(bortle_to_sky_mag(args.sky_bortle)),
                            '--sky-psf-wavelength-nm', str(args.sky_psf_wavelength_nm),
                            '--sky-seeing-r0-m', str(args.sky_seeing_r0_m)]
            if psf_sigma_by_role.get(role) is not None:   # else the cam auto-picks (0 if aperture known, else 1.3)
                per_role_sky += ['--sky-psf-sigma-px', str(psf_sigma_by_role[role])]
            if role in fov_by_role:                   # DB optics named -> render at the true sensor
                per_role_sky += ['--sky-width', str(rx), '--sky-height', str(ry)]   # res, so FoV matches
            if role in roi_window_by_role:            # centered readout window (sensor px) -> frame metadata
                per_role_sky += ['--roi', ','.join(str(v) for v in roi_window_by_role[role])]
            per_role_sky += [sky_follow_flag()]       # follow the current mount (re-evaluated each launch)
        if sources[role] == 'zwo' and roi_by_role.get(role) and role in roi_window_by_role:
            per_role_sky += ['--roi', ','.join(str(v) for v in roi_window_by_role[role])]   # crop in hardware
        # Open the cam already at the user's live exposure/gain, so the very FIRST frame is correct --
        # no default-brightness flash before the post-launch control re-apply catches up. (zwo sets
        # these before start_video_capture in _open_zwo; the sim's re-apply is instant, so it's left
        # to the {'controls'} write below.)
        cv = cam_control_vals.get(role) or {}
        init_args = []
        if sources[role] == 'zwo':
            if cv.get('exposure') is not None:
                init_args += ['--exposure-us', str(max(1, int(round(float(cv['exposure']) * 1000))))]
            if cv.get('gain') is not None:
                init_args += ['--gain', str(int(round(float(cv['gain']))))]
        cam_sel = ['--camera-url', camera_url[role]] if (sources[role] == 'zwo' and camera_url[role]) else []
        pb = playback_by_role.get(role) or {}
        playback_args = (['--playback-ser', pb['ser'], '--playback-speed', str(args.playback_speed),
                          '--playback-fps', str(args.playback_fps)]
                         + (['--playback-loop'] if pb.get('loop') else [])
                         if sources[role] == 'playback' and pb.get('ser') else [])
        cam_procs[role] = _spawn('astrolock.seeker.cam', [
            '--role', role, '--out-dir', session_dir, '--source', sources[role],
            '--width', str(rx), '--height', str(ry), '--fps', str(fps_by_role.get(role, args.fps)),
            '--bit-depth', str(depth_by_role.get(role, args.bit_depth)),
            '--state-shm', state_slot.name,        # sky follow-state pose (file-free)
            '--device', args.device,               # sky-sim render device (zwo/synthetic ignore it)
            '--bin', str(bin_by_role[role]),       # physical NxN bin (sim: metadata; zwo: hardware)
            *(['--shm-ser', '--shm-frames', str(args.shm_frames)] if args.shm_ser else []),
            '--control-file', cf,
            *cam_sel, *init_args, *(['--auto'] if args.auto else []), *sky_args, *per_role_sky, *playback_args,
        ])
        if cam_control_vals.get(role):                 # re-apply live control values across a relaunch
            control_writers[role].append({'controls': dict(cam_control_vals[role])})

    detect_procs = {}

    def launch_detect(role):
        det = getattr(args, f'{role}_detector', None) or args.detector   # per-role override
        tdet = getattr(args, f'{role}_track_detector', None) or args.track_detector
        detect_procs[role] = _spawn('astrolock.seeker.detect',
                                    ['--session', session_dir, '--role', role, '--follow',
                                     '--stop-file', stop_file,
                                     '--detector', det, '--track-detector', tdet,
                                     '--state-shm', state_slot.name,
                                     *(['--shm-ser', '--shm-frames', str(min(64, args.shm_frames))]
                                       if args.shm_ser else []),
                                     '--track-blur-px', str(args.track_blur_px),
                                     '--track-pull', str(args.track_pull),
                                     '--doh-sigma', str(args.doh_sigma),
                                     '--surprise-decay', str(args.surprise_decay),
                                     '--surprise-blur-px', str(args.surprise_blur_px),
                                     '--tile-size', str(args.tile_size),
                                     '--tile-fixed-nsigma', str(args.tile_fixed_nsigma),
                                     '--tile-moving-nsigma', str(args.tile_moving_nsigma),
                                     '--tile-mask-px', str(args.tile_mask_px),
                                     '--ext-nsigma', str(args.ext_nsigma),
                                     '--ext-density', str(args.ext_density),
                                     '--circ-down', str(args.circ_down),
                                     '--circ-bg-radius', str(args.circ_bg_radius),
                                     '--circ-nsigma', str(args.circ_nsigma),
                                     '--circ-z', str(args.circ_z),
                                     '--snr', str(args.snr), '--max-candidates', str(args.max_candidates),
                                     '--min-blob-px', str(args.min_blob_px),
                                     '--tile-grid', str(args.tile_grid), '--per-tile', str(args.per_tile),
                                     '--device', args.device]
                                    + (['--debug-ser'] if args.debug_detect_ser else []))

    # Focus/collimation: one optional focus process, toggled from the GUI Focus tab (Guide or Main).
    # It follows a role's cam .ser + detections and writes <seg>_<role>_focus.ser (the EMA star image,
    # PIP-able) + _focus.frames.jsonl (peak / focus_x/y / com metrics for the graph). Only one runs at a
    # time; a role switch stops the old one. focus_stop is its private stop flag (the session stop file
    # is for the always-on cams/detectors); we remove it before each launch and touch it to stop.
    focus_proc = None
    focus_role = None
    focus_stop = os.path.join(session_dir, 'focus_stop')

    def stop_focus():
        nonlocal focus_proc, focus_role
        if focus_proc is not None and focus_proc.poll() is None:
            open(focus_stop, 'w').close()          # ask it to finish cleanly; it polls the flag
            _reap([focus_proc])
        try:
            os.remove(focus_stop)                  # clear the flag so the next launch isn't pre-stopped
        except OSError:
            pass
        focus_proc, focus_role = None, None

    def launch_focus(role):
        nonlocal focus_proc, focus_role
        stop_focus()                               # only one focus at a time
        _rx, _ry, pum, fmm = render_by_role[role]  # frame pixel pitch + focal -> the ideal PSF for Strehl
        focus_proc = _spawn('astrolock.seeker.focus',
                            ['--session', session_dir, '--role', role, '--follow',
                             '--stop-file', focus_stop, '--device', args.device,
                             *(['--shm-ser'] if args.shm_ser else []),
                             '--crop', str(args.focus_crop), '--search', str(args.focus_search),
                             '--alpha', str(args.focus_alpha),
                             # Optics for the Strehl reference (aperture 0 = unknown -> delta ideal).
                             '--aperture-mm', str(aperture_by_role.get(role, 0.0)),
                             '--focal-mm', str(fmm), '--pixel-um', str(pum),
                             '--wavelength-nm', str(args.sky_psf_wavelength_nm),
                             '--obstruction', str(obstruction_by_role.get(role, 0.0)),
                             '--vanes', str(vanes_by_role.get(role, 0)),
                             '--vane-width', str(vane_width_by_role.get(role, 0.0))])
        focus_role = role
        print(f"[backend] focus started on {role}", flush=True)

    # Focus sweep: an optional actuator-agnostic sweep process (focus_sweep.py). It watches the
    # focus stream's per-frame HFD, publishes position requests on {role}_sweep, and whoever can
    # move the focuser (GUI human, or a future electronic-focuser process) answers on
    # {role}_focuser. We hold its stdin: closing the pipe = abort (recorder pattern).
    sweep_proc = None
    sweep_role = None

    def stop_sweep():
        nonlocal sweep_proc, sweep_role
        if sweep_proc is not None and sweep_proc.poll() is None:
            try:
                sweep_proc.stdin.close()           # abort: publish final state, end stream, exit
            except OSError:
                pass
            _reap([sweep_proc])
        sweep_proc, sweep_role = None, None

    def launch_sweep(role, cmd):
        nonlocal sweep_proc, sweep_role
        stop_sweep()
        if focus_role != role or focus_proc is None or focus_proc.poll() is not None:
            launch_focus(role)                     # the sweep feeds on the focus stream's HFD
        sweep_proc = subprocess.Popen(
            [sys.executable, '-m', 'astrolock.seeker.focus_sweep',
             '--session', session_dir, '--role', role,
             '--start', str(float(cmd['start'])), '--end', str(float(cmd['end'])),
             '--steps', str(int(cmd.get('steps', 9))),
             '--frames-per-step', str(int(cmd.get('frames', 40))),
             '--settle-s', str(float(cmd.get('settle', 1.0)))],
            stdin=subprocess.PIPE)
        sweep_role = role
        print(f"[backend] focus sweep started on {role}: "
              f"{cmd['start']} .. {cmd['end']} x{cmd.get('steps', 9)}", flush=True)

    # Pre-warm the skyfield ephemeris/star cache once, serially. Two sky cams starting together
    # otherwise race to download de421.bsp / hipparcos into the shared cache and one loses the
    # rename (WinError 5) -- which is what crashed both sim cams in a fresh worktree. Best-effort:
    # if it fails, the cams fall back to their own (racy) download exactly as before.
    sky_sim_proc = None
    if any(s == 'sky' for s in sources.values()):
        try:
            from astrolock.seeker import skysim
            skysim.ensure_cache()
        except Exception as e:
            print(f"[backend] ephemeris pre-warm skipped: {e}", flush=True)
        # One sky_sim process propagates stars + satellite and publishes their directions on the
        # shared system clock, so every camera reads identical positions (fixes the two-cam drift).
        ss_args = ['--out', almanac_path, '--lat', str(msite['lat_deg']),
                   '--lon', str(msite['lon_deg']), '--elev', str(msite['elev_m']),
                   '--epoch', str(msite['epoch_utc']), '--stop-file', stop_file]
        if args.sky_tle_file:
            ss_args += ['--tle-file', args.sky_tle_file, '--target-mag', str(args.sky_target_mag)]
        sky_sim_proc = _spawn('astrolock.seeker.sky_sim', ss_args)
        print(f"[backend] sky_sim -> {almanac_path}", flush=True)

    print(f"[backend] detect roles: {sorted(detect_roles) or 'none'}", flush=True)
    for role in roles:
        launch_cam(role)
        if role in detect_roles:
            launch_detect(role)
    gui_proc = _spawn('astrolock.seeker.gui_imgui',
                      ['--session', session_dir, '--wb-r', str(args.wb_r), '--wb-b', str(args.wb_b),
                       '--device', args.device]) \
        if args.gui else None

    followers = {role: SerFollower(session_dir, role) for role in roles}

    det_fos = {role: framestream.StreamFollower(session_dir, f'{role}_det')
               for role in roles}                    # detections stream (raw JSON payloads)
    latest_blobs = {role: [] for role in roles}
    # A detection names its frame as (cam segment ident, index) + carries the frame's OWN capture
    # time. Never look a bare index up in "the current segment": the detector lags the cam, so
    # near a roll that reads the WRONG segment's record (a phantom-motion timestamp under slew).
    latest_det_key = {role: None for role in roles}      # (seg, index): frame identity
    latest_det_t_s = {role: None for role in roles}      # that frame's capture time (s, mono clock)
    latest_ext = {role: None for role in roles}          # role -> extended-detector metrics (present/...)

    def update_detections():
        for role in roles:
            fo_ = det_fos[role]
            fo_.poll()
            for rd, i in fo_.drain():                # every new detection record, in order
                try:
                    rec = json.loads(bytes(rd.read(i)).decode('utf-8'))
                except framestream.Lapped:
                    continue                         # ancient record overwritten; fo_.lost counts it
                latest_blobs[role] = rec.get('blobs', [])
                latest_det_key[role] = (rec.get('seg'), rec.get('index'))
                t_ns = rec.get('t_mono_ns')
                latest_det_t_s[role] = t_ns * 1e-9 if t_ns else None
                latest_ext[role] = rec.get('ext')    # extended: {'present', 'com', ...} or None

    def frame_binning(role):
        """Sensor pixels per frame pixel for a role, from the live segment's header meta.
        (v3 killed the frames.jsonl sidecar this used to read -- the dead lookup silently
        returned 1.0, halving the reconstruction plate scale for a 2x2-binned cam: pixel
        motion then cancelled only HALF the mount motion, so a fixed target appeared to flee
        at half the slew rate and every lock overshot.)"""
        b = (followers[role].meta or {}).get('bin')
        return float(b[0]) if b else 1.0

    def _detector_for(role):
        return getattr(args, f'{role}_detector', None) or args.detector

    # The full-frame single-target detectors: present/absent + one COM, no per-target ROI. Both drive
    # the guide->main handoff and never take a tracking ROI (the backend runs them full-frame always).
    SINGLE_TARGET_DETECTORS = ('extended', 'circmean')

    def _single_target(role):
        return _detector_for(role) in SINGLE_TARGET_DETECTORS

    def track_geom(role):
        """Reconstruction geometry for tracking on `role`: (cx, cy, rad_per_px, sign_az, sign_alt). The
        optical centre is the frame centre, except the guide holds the target at the boresight pixel (so
        it lands in the main cam); the main cam holds at its own centre and self-corrects the boresight."""
        hdr = followers[role].header
        rpp = rad_per_px_by_role.get(role, rad_per_px) * frame_binning(role)
        cx, cy = hdr.image_width / 2.0, hdr.image_height / 2.0
        if role == 'guide':
            cx += boresight['x'] / rpp
            cy += boresight['y'] / rpp
        return cx, cy, rpp, args.track_sign_az, args.track_sign_alt

    def _fine_present():
        """True if the fine (main) cam's extended detector currently sees a target."""
        ext = latest_ext.get(handoff_fine)
        return bool(ext and ext.get('present'))

    def control_write(role, obj):
        if role in control_writers:
            control_writers[role].append(obj)

    def read_caps(role):
        """Pick up a role's camera controls from caps_<role>.json (the cam rewrites it each launch)."""
        path = os.path.join(session_dir, f'caps_{role}.json')
        try:
            m = os.path.getmtime(path)
        except OSError:
            return
        if caps_mtime.get(role) == m:
            return
        try:
            cam_caps[role] = json.load(open(path, encoding='utf-8'))
            caps_mtime[role] = m
        except (OSError, ValueError):
            pass

    def _roi_choices(role):
        """ROI options as rendered (binned) heights: 'Full', then powers of two from the largest that
        fits the sensor's binned height down to 64px. Recomputed live (depends on binning)."""
        b = max(1, bin_by_role.get(role, 1))
        binned_h = max(1, full_res_by_role.get(role, (0, 0))[1] // b)
        sizes, n = [], 64
        while n <= binned_h:
            sizes.append(n)
            n *= 2
        return ['Full'] + [str(s) for s in reversed(sizes)]

    def _roi_value(role):
        """Current ROI as one of _roi_choices: 'Full', else the output height in px (square in the GUI)."""
        roi = roi_by_role.get(role)
        return 'Full' if not roi else str(int(roi[1]))

    def geometry_caps(role):
        """Backend-owned geometry controls (binning, ROI). Unlike exposure/gain -- which the cam owns,
        having the device open -- the backend owns render/readout geometry, so it publishes these.
        Relaunch-tier (they change frame geometry), presented as choices so one pick = one relaunch.
        Binning is real on both sky (render downscale) and zwo (hardware NxN bin -- a color cam binned
        >=2 reads out mono); ROI applies to both -- the sim render *is* the crop, and zwo crops in
        hardware (SetROIFormat, output px snapped to width%8 / height%2 / even start)."""
        src = sources.get(role)
        caps = []
        if src in ('sky', 'zwo', 'synthetic'):
            caps.append({'name': 'bin', 'label': 'Binning', 'kind': 'choice', 'choices': ['1', '2', '3', '4'],
                         'value': str(bin_by_role.get(role, 1)), 'live': False})
            # Max-FPS cap: the cam loop's optional throttle (0 = unlimited). Live -- applied via the
            # control stream's {'fps': ...} channel, no relaunch.
            caps.append({'name': 'fps', 'label': 'Max FPS', 'kind': 'number', 'unit': 'fps',
                         'scale': 'linear', 'min': 0.0, 'max': 1000.0,
                         'value': float(fps_by_role.get(role, args.fps)), 'live': True})
        if src in ('sky', 'zwo'):                          # capture bit depth (8 = RAW8 fast; 16 = full)
            caps.append({'name': 'depth', 'label': 'Bit Depth', 'kind': 'choice', 'choices': ['8', '16'],
                         'value': str(depth_by_role.get(role, args.bit_depth)), 'live': False})
        if src in ('sky', 'zwo'):                          # sim crops the render; zwo crops in hardware
            caps.append({'name': 'roi', 'label': 'ROI', 'kind': 'choice', 'choices': _roi_choices(role),
                         'value': _roi_value(role), 'live': False})
        if src == 'sky':                                   # sim-only: lens softness/defocus atop the Airy PSF
            sig = psf_sigma_by_role.get(role)              # None (auto) shows as 0 -> "sharp physical PSF"
            caps.append({'name': 'defocus', 'label': 'Defocus', 'kind': 'number', 'unit': 'px',
                         'scale': 'linear', 'min': 0.0, 'max': 8.0,
                         'value': 0.0 if sig is None else float(sig), 'live': False})
        return caps

    def published_caps(role):
        """The role's caps with each control's value overridden by the live user setting (so the GUI
        shows what's actually in effect, not the launch default), plus backend geometry controls."""
        caps = cam_caps.get(role)
        if not caps:
            return None
        vals = cam_control_vals.get(role, {})
        ctrls = [{**c, 'value': vals.get(c['name'], c.get('value'))} for c in caps.get('controls', [])]
        return {'source': caps.get('source'), 'controls': ctrls + geometry_caps(role)}

    def is_connected(role):
        """True if this role's cam process is currently running (its capture is live)."""
        return role in cam_procs and cam_procs[role].poll() is None

    def restart_cam(role, stop_first):
        if stop_first:
            control_write(role, {'stop': True})    # old cam finalizes its file + exits
        launch_seq[role] += 1
        launch_cam(role)                           # new cam (new source / fresh segment)
        if role in detect_roles and (role not in detect_procs or detect_procs[role].poll() is not None):
            launch_detect(role)                    # safety: detect normally rolls on its own

    def delete_old_segments():
        """Rolling GC of live-session leftovers. A ring stream leaves nothing per frame -- the
        only per-stream disk artifacts are .dat rings (file-store sessions), and only ones
        orphaned by a reconfigure are worth sweeping. Head files are tiny and append-only;
        recordings live in --recordings-dir and are never touched."""
        if args.keep:
            return
        streams = [n for role in roles for n in (role, f'{role}_debug', f'{role}_focus')]
        for name in streams:
            dats = sorted(glob.glob(os.path.join(session_dir, f'{name}_*.dat')),
                          key=os.path.getmtime)
            for sp in dats[:-1]:                  # keep the live ring; reconfigures are rare
                try:
                    os.remove(sp)
                except OSError:
                    pass

    def connect_mount(url):
        """Replace the current mount driver with a fresh one for `url` ('sim' or a celestron URL),
        continuing from the current pose. On failure, keep the current mount."""
        nonlocal mount, mount_connected, mount_desired_url, tracking, coasting
        mount_desired_url = url
        st = mount.get_state()
        try:
            if url == 'sim':
                new = mount_mod.make_mount(
                    'sim', az0_rad=st['az_rad'], alt0_rad=st['alt_rad'], site=site, max_rate_rad_s=max_rate,
                    accel_rad_s2=math.radians(args.mount_accel_deg_s2), update_hz=args.mount_update_hz,
                    sidecar_path=sim_mount_path)
            else:
                new = mount_mod.make_mount('celestron', az0_rad=st['az_rad'], alt0_rad=st['alt_rad'],
                                           site=site, max_rate_rad_s=max_rate, url=url)
        except Exception as e:
            print(f"[backend] mount connect failed ({url}): {e}", flush=True)
            return
        old, mount = mount, new
        mount_connected = True
        tracking = coasting = False                        # a fresh mount -> drop any active track
        try:
            old.close()
        except Exception:
            pass
        for r in roles:                                    # re-point sky-sim cams at the new mount
            if sources.get(r) == 'sky':
                restart_cam(r, stop_first=True)
        print(f"[backend] mount connected: {url}", flush=True)

    def disconnect_mount():
        """Stop + close the current mount and leave a NullMount (holds pose, ignores commands)."""
        nonlocal mount, mount_connected, tracking, coasting
        st = mount.get_state()
        old, mount = mount, mount_mod.NullMount(st['az_rad'], st['alt_rad'], site=mount.get_site())
        mount_connected = False
        tracking = coasting = False
        try:
            old.close()
        except Exception:
            pass
        for r in roles:                                    # sky-sim cams now follow the frozen state pose
            if sources.get(r) == 'sky':
                restart_cam(r, stop_first=True)
        print("[backend] mount disconnected", flush=True)

    def apply_command(cmd):
        nonlocal estop, recording, tracking, coasting, track_role, tracker, track_seen_det, gui_quit
        nonlocal track_center, mount_desired_url, follow_enabled, track_primary, handoff_fine, handoff_conf
        nonlocal track_pref, focus_proc, focus_role, track_delay_s, track_started_t
        t = cmd.get('type')
        if t == 'shutdown':                           # GUI is closing -> stop the whole session
            gui_quit = True
            return
        if t == 'set_rate':
            tracking = coasting = False               # manual slew overrides auto-track (and coast)
            mount.set_rates(math.radians(cmd.get('az', 0.0)), math.radians(cmd.get('alt', 0.0)))
            estop = False
        elif t == 'stop':                             # legacy: halt mount + tracking (kept for compat)
            tracking = coasting = False
            mount.set_rates(0.0, 0.0)
        elif t == 'follow':                           # Follow on/off: PERSISTENT -- drive the mount to hold
            follow_enabled = bool(cmd.get('on', True))    # a lock, or watch-only (tracker keeps estimating,
            if not follow_enabled:                        # the mount holds still). Settable before a lock.
                mount.set_rates(0.0, 0.0)
        elif t == 'estop':
            tracking = coasting = False               # e-stop halts a coast too
            mount.set_rates(0.0, 0.0)
            estop = True
        elif t == 'track':                            # lock the sky-space tracker onto a target
            role = cmd.get('role', roles[0])
            px = cmd.get('px')
            if role not in detect_roles:              # no detector on this role -> nothing to track
                print(f"[backend] ignoring track on {role!r} (no detector; see --detect-roles)", flush=True)
            elif role in roles and px and followers[role].header is not None:
                hdr = followers[role].header
                # Blobs are in frame image space; hold the target at the frame centre. rad_per_px
                # is per *sensor* pixel (from this role's optics), so scale by the cam's binning.
                rpp = rad_per_px_by_role.get(role, rad_per_px) * frame_binning(role)
                ft = latest_det_t_s[role]                 # the detection frame's OWN capture time
                if ft is None:                            # never eat a user's lock click silently
                    print(f"[backend] ignoring track on {role}: no detection with a capture "
                          f"time yet", flush=True)
                if ft is not None:
                    from astrolock.seeker.skytracker import SkyTracker
                    from astrolock.seeker.target_model import (EmaAngularVelModel,
                                                               GreatCircleModel)
                    model = (GreatCircleModel(smoothing_s=args.track_rate_smoothing_s,
                                              altitude_m=args.track_alt_km * 1e3)
                             if args.track_model == 'greatcircle' else
                             EmaAngularVelModel(smoothing_s=args.track_rate_smoothing_s))
                    cx0, cy0 = hdr.image_width / 2.0, hdr.image_height / 2.0
                    track_center = (cx0, cy0)                 # true optical centre (for live re-offset)
                    if role == 'guide':                       # aim so the target lands in the main cam:
                        cx0 += boresight['x'] / rpp           # hold it at the main's boresight pixel, not
                        cy0 += boresight['y'] / rpp           # the guide centre (used for recon + setpoint)
                    tracker = SkyTracker(cx0, cy0, rpp,
                                         max_rate_rad_s=max_rate,
                                         model=model,
                                         min_intercept_s=args.track_min_intercept_s,
                                         command_latency_s=args.track_command_latency_s,
                                         max_horizon_s=args.track_max_horizon_s,
                                         gate_px=args.track_gate_px, lost_s=args.track_lost_s,
                                         lock_min_time=args.track_lock_min_time,
                                         sign_az=args.track_sign_az, sign_alt=args.track_sign_alt)
                    tracker.start(float(px[0]), float(px[1]), ft, mount.get_state())
                    track_seen_det = latest_det_key[role]
                    tracking = True
                    coasting = False
                    track_started_t = session_mod.mono_s()    # the follow delay counts from here
                    if not follow_enabled or track_delay_s > 0:
                        mount.set_rates(0.0, 0.0)             # watch-only / learn-first: hold still
                    track_role = role
                    # Handoff setup: this role is the fallback ('primary'); if a *finer* role runs the
                    # extended detector, watch it to promote once it stably sees the target.
                    track_primary = role
                    handoff_conf = 0.0
                    handoff_fine = next((r for r in roles if r != role and r in detect_roles
                                         and _single_target(r)), None) if args.handoff else None
                    estop = False
                    print(f"[backend] acquired target on {role} at "
                          f"({float(px[0]):.0f},{float(px[1]):.0f})px"
                          + (f"; will promote to {handoff_fine} when it locks" if handoff_fine else ""),
                          flush=True)
                    info, warns = tracker.diagnostics()      # tracker self-report at lock time
                    for ln in info:
                        print(f"[backend] track {role}: {ln}", flush=True)
                    for w in warns:
                        print(f"[backend] WARNING (track {role}): {w}", flush=True)
        elif t == 'set_boresight':
            boresight['x'] = float(cmd.get('x_mrad', 0.0)) * 1e-3
            boresight['y'] = float(cmd.get('y_mrad', 0.0)) * 1e-3
            if tracker is not None and track_center is not None and track_role == 'guide':
                rpp = rad_per_px_by_role.get(track_role, rad_per_px) * frame_binning(track_role)
                tracker.cx = track_center[0] + boresight['x'] / rpp   # re-aim the live hold point
                tracker.cy = track_center[1] + boresight['y'] / rpp
            print(f"[backend] boresight -> ({boresight['x'] * 1e3:.3f}, {boresight['y'] * 1e3:.3f}) mrad",
                  flush=True)
        elif t == 'set_track_delay':                  # hold the mount this long after a new lock
            track_delay_s = max(0.0, float(cmd.get('value', 0.0)))
            print(f"[backend] track delay = {track_delay_s:.1f}s", flush=True)
        elif t == 'set_track_smoothing':              # live tracker-tuning: EMA time constant (s)
            v = max(0.0, float(cmd.get('value', args.track_rate_smoothing_s)))
            args.track_rate_smoothing_s = v           # remembered for the next lock...
            if tracker is not None and hasattr(tracker.model, 'smoothing_s'):
                tracker.model.smoothing_s = v         # ...and applied to the running lock right now
            print(f"[backend] track rate smoothing = {v:.2f}s", flush=True)
        elif t == 'set_track_model':                  # target motion model for the NEXT lock
            m = cmd.get('model')
            if m in ('ema', 'greatcircle'):
                args.track_model = m
            if 'alt_km' in cmd:
                args.track_alt_km = max(1.0, float(cmd['alt_km']))
            print(f"[backend] track model = {args.track_model}"
                  + (f" (alt {args.track_alt_km:g} km)" if args.track_model == 'greatcircle'
                     else ""), flush=True)
        elif t == 'set_track_pref':                   # which cam drives tracking: 'guide' / 'main' / 'auto'
            p = cmd.get('pref')
            if p in ('guide', 'main', 'auto'):
                track_pref = p
                print(f"[backend] track source preference = {track_pref}", flush=True)
        elif t == 'set_sky_render':                   # global sim-truth atmosphere: seeing r0 + sky brightness
            if 'r0_m' in cmd:
                args.sky_seeing_r0_m = max(0.0, float(cmd['r0_m']))
            if 'bortle' in cmd:
                args.sky_bortle = int(max(1, min(9, round(float(cmd['bortle'])))))
            print(f"[backend] sky render: seeing r0 = {args.sky_seeing_r0_m}m, Bortle {args.sky_bortle} "
                  f"({bortle_to_sky_mag(args.sky_bortle):.1f} mag/arcsec^2)", flush=True)
            for r in roles:                            # rebuild each live sky cam (PSF + sky rate set at launch)
                if sources.get(r) == 'sky' and is_connected(r):
                    restart_cam(r, stop_first=True)
        elif t == 'untrack':                          # "Unlock": drop the lock entirely + halt the mount
            tracking = coasting = False
            handoff_fine, handoff_conf = None, 0.0    # drop any pending main-cam handoff
            mount.set_rates(0.0, 0.0)
        elif t == 'focus':                            # GUI Focus tab: start/stop the focus process on a role
            role = cmd.get('role')
            if 'alpha' in cmd:                        # star-crop EMA smoothing (relaunch-tier)
                args.focus_alpha = max(1e-4, min(1.0, float(cmd['alpha'])))
            if cmd.get('on', True):
                if role in roles:
                    launch_focus(role)
                else:
                    print(f"[backend] ignoring focus on {role!r} (unknown role)", flush=True)
            else:
                stop_focus()
                print("[backend] focus stopped", flush=True)
        elif t == 'sweep':                            # GUI Focus tab: start/abort a focus sweep
            role = cmd.get('role')
            if cmd.get('on', True):
                if role in roles:
                    launch_sweep(role, cmd)
                else:
                    print(f"[backend] ignoring sweep on {role!r} (unknown role)", flush=True)
            else:
                stop_sweep()
                print("[backend] focus sweep aborted", flush=True)
        elif t == 'record':
            # Recording is a separate PROCESS per role (astrolock.seeker.recorder): it tails the
            # role's shm stream and archives every frame to recordings/ at drive pace -- the cam
            # never touches the disk, and the held shm sections are the write-behind buffer. The
            # cam process isn't told anything; there is no 'important' concept anymore.
            on = bool(cmd.get('on', False))
            r = cmd.get('role')
            targets = [r] if r in roles else roles    # a named role, else all (back-compat)
            for role in targets:
                if on and not _recorder_alive(role):
                    # We hold the recorder's stdin: 'stop' is a line on it, and if WE die the
                    # pipe closes and the recorder freezes+flushes+exits on its own. Nobody is
                    # ever killed; it also self-exits at the stream's 'ended' record.
                    rec_procs[role] = subprocess.Popen(
                        [sys.executable, '-m', 'astrolock.seeker.recorder',
                         '--session', session_dir, '--role', role,
                         '--out-dir', args.recordings_dir],
                        stdin=subprocess.PIPE)
                elif not on and _recorder_alive(role):
                    try:
                        rec_procs[role].stdin.close()     # freeze + flush + exit
                    except OSError:
                        pass
                recording[role] = on
            print(f"[backend] recording {'ON' if on else 'off'} for {', '.join(targets)}", flush=True)
        elif t == 'capture':
            role = cmd.get('role')
            if role in roles:
                if cmd.get('on', True):                 # (re)start a stopped camera + its detector
                    if role not in cam_procs or cam_procs[role].poll() is not None:
                        restart_cam(role, stop_first=False)
                        print(f"[backend] capture started on {role}", flush=True)
                else:
                    control_write(role, {'stop': True})  # cam finalizes its file and exits
                    print(f"[backend] capture stopped on {role}", flush=True)
        elif t == 'set_source':
            role = cmd.get('role')
            src = cmd.get('source')
            if role in roles and src in ('synthetic', 'zwo', 'sky', 'playback'):
                sources[role] = src
                # Don't jump the gun: a driver change never auto-connects. Stop any running cam and
                # wait for an explicit Connect, so the user can pick which camera first -- two roles
                # both left on '(auto)' zwo would otherwise open one camera twice and wedge the USB bus.
                if is_connected(role):
                    control_write(role, {'stop': True})
                print(f"[backend] {role} source -> {src} (disconnected; press Connect)", flush=True)
        elif t == 'set_optics':
            role = cmd.get('role')
            if role in roles:
                ok = resolve_optics(role, cmd.get('sensor'), cmd.get('optic'), cmd.get('reducer'))
                print(f"[backend] {role} optics -> {optics_sel[role]} ({'ok' if ok else 'fallback'})", flush=True)
                if ok and sources[role] == 'sky' and is_connected(role):   # sim FoV changed -> relaunch live
                    restart_cam(role, stop_first=True)
                if focus_proc is not None and focus_proc.poll() is None and role == focus_role:
                    launch_focus(role)                 # new optics -> rebuild the Strehl reference PSF
        elif t == 'set_camera':
            role = cmd.get('role')
            if role in roles:
                camera_url[role] = cmd.get('url') or None
                print(f"[backend] {role} camera -> {camera_url[role]}", flush=True)
                if sources[role] == 'zwo' and is_connected(role):   # live swap only if already connected
                    restart_cam(role, stop_first=True)              # else it's picked up on Connect
        elif t == 'set_playback':
            role = cmd.get('role')
            if role in roles:
                pb = playback_by_role.setdefault(role, {'ser': None, 'loop': True})
                if 'ser' in cmd:
                    pb['ser'] = cmd.get('ser') or None
                if 'loop' in cmd:
                    pb['loop'] = bool(cmd.get('loop'))
                print(f"[backend] {role} playback -> {pb['ser']} loop={pb['loop']}", flush=True)
                if sources[role] == 'playback' and is_connected(role):
                    restart_cam(role, stop_first=True)
        elif t == 'rescan_cameras':
            cams_available[:] = cam_mod.zwo_camera_urls()
            print(f"[backend] cameras: {cams_available}", flush=True)
        elif t == 'set_mount':
            url = cmd.get('url')
            if url:
                mount_desired_url = url
                if mount_connected:            # don't jump the gun: reselect -> disconnect, then Connect
                    disconnect_mount()
                print(f"[backend] mount selected: {url} (press Connect)", flush=True)
        elif t == 'mount_connect':
            if cmd.get('on', True):
                connect_mount(mount_desired_url)
            else:
                disconnect_mount()
        elif t == 'rescan_mounts':
            mounts_available[:] = mount_mod.available_mount_urls()
            print(f"[backend] mounts: {mounts_available}", flush=True)
        elif t == 'set_cam_control':
            role, name, value = cmd.get('role'), cmd.get('name'), cmd.get('value')
            if role in roles and name == 'bin':                        # geometry: recompute render (+ relaunch)
                bin_by_role[role] = max(1, int(round(float(value))))
                recompute_render(role)
                if is_connected(role):                                 # live relaunch; else applied on Connect
                    restart_cam(role, stop_first=True)
                print(f"[backend] {role} bin = {bin_by_role[role]}", flush=True)
            elif role in roles and name == 'roi':
                if str(value).lower() == 'full':
                    roi_by_role[role] = None
                else:
                    n = max(64, int(float(str(value))))     # GUI picks a square; the model allows any (w, h)
                    roi_by_role[role] = (n, n)
                recompute_render(role)
                if is_connected(role):
                    restart_cam(role, stop_first=True)
                print(f"[backend] {role} roi = {_roi_value(role)}", flush=True)
            elif role in roles and name == 'defocus':      # sim residual blur (px); rebuilds the PSF -> relaunch
                b = float(value)
                psf_sigma_by_role[role] = b if b > 0.0 else None   # 0 -> auto (sharp physical PSF)
                if is_connected(role):
                    restart_cam(role, stop_first=True)
                print(f"[backend] {role} defocus = {psf_sigma_by_role[role]} px", flush=True)
            elif role in roles and name == 'fps':          # cam-loop max-fps cap: live, its own channel
                fps_by_role[role] = max(0.0, float(value))
                control_write(role, {'fps': fps_by_role[role]})    # cam handles cmd['fps'] directly
                print(f"[backend] {role} max fps = {fps_by_role[role] or 'unlimited'}", flush=True)
            elif role in roles and name == 'depth':        # capture bit depth: changes the format -> relaunch
                depth_by_role[role] = 8 if int(float(str(value))) == 8 else 16
                if is_connected(role):
                    restart_cam(role, stop_first=True)
                print(f"[backend] {role} bit depth = {depth_by_role[role]}", flush=True)
            elif role in roles and name is not None:
                cam_control_vals[role][name] = value
                live = next((c.get('live', True) for c in (cam_caps.get(role) or {}).get('controls', [])
                             if c['name'] == name), True)
                if live:
                    control_write(role, {'controls': {name: value}})   # apply on the running cam
                else:
                    restart_cam(role, stop_first=True)                 # format change -> relaunch (later)
                print(f"[backend] {role} control {name} = {value} ({'live' if live else 'relaunch'})", flush=True)

    start = session_mod.mono_s()
    last_health = start
    last_cleanup = start
    handoff_last = start          # for the handoff integrator's real-time dt (framerate-independent)
    clean = False
    try:
        while True:
            time.sleep(0.05)                      # ~20 Hz control loop
            now = session_mod.mono_s()            # SAME timeline as every t_mono_ns stamp: this
                                                  # 'now' meets frame times in the tracker/servo

            for cmd in cmd_server.drain():
                apply_command(cmd)
            update_detections()
            for role in roles:
                read_caps(role)                        # pick up each cam's live-control descriptors

            st = mount.get_state()
            if tracking and tracker is not None:
                tracker.push_mount(st)        # build the mount-pose history at the full loop rate
            track_status = None
            if tracking and tracker is not None and not estop:
                role = track_role
                if latest_det_key[role] != track_seen_det:       # act once per new frame...
                    ft = latest_det_t_s[role]
                    if ft is not None:                            # ...clocked by its capture time
                        track_seen_det = latest_det_key[role]
                        # SkyTracker returns final axis rates -- it owns the alt-az IK and its own pole
                        # handling (tips altitude over the top), given the mount-pose history + now.
                        raz, ralt, track_status, tpx = tracker.update(
                            st, latest_blobs[role], True, ft, now)
                        if args.track_debug:
                            print(f"[track] cmd az {math.degrees(raz):+6.2f} alt {math.degrees(ralt):+6.2f} | "
                                  f"meas az {math.degrees(st['rate_az_rad_s']):+6.2f} "
                                  f"alt {math.degrees(st['rate_alt_rad_s']):+6.2f} deg/s | "
                                  f"off ({tpx[0] - tracker.cx:+5.0f},{tpx[1] - tracker.cy:+5.0f})px | "
                                  f"alt {math.degrees(st['alt_rad']):5.1f} | {track_status}", flush=True)
                        # Drive only when following is on AND the post-lock learn delay has elapsed
                        # ("Stop Moving" / watch-only / delay all hold the mount but keep tracking).
                        if follow_enabled and (now - track_started_t) >= track_delay_s:
                            mount.set_rates(raz, ralt)
                        # On 'track' and 'coast' the target estimate keeps moving, so keep publishing it
                        # (the ROI follows, so detect keeps searching and can re-acquire during coast).
                        track_target = list(tpx) if track_status in ('track', 'coast') else None
                        # Raw detection (green marker) only while we have a fresh one; None on coast/lost.
                        track_detect_px = (list(tracker.last_meas_px)
                                           if track_status == 'track' and tracker.last_meas_px else None)
                        if track_status == 'lost':
                            tracking, coasting = False, False
                            print(f"[backend] lost target on {role}", flush=True)
                        elif track_status == 'coast' and not coasting:
                            coasting = True
                            print(f"[backend] coasting on {role} (settled lock lost; holding last rate "
                                  f"to re-acquire -- e-stop to halt)", flush=True)
                        elif track_status == 'track':
                            coasting = False

            # ---- camera handoff: choose which cam feeds the model, honouring the GUI preference ---------
            # Leaky integrator of the fine cam's 'present' signal, in real seconds (framerate-independent):
            # rise while present, fall faster (--handoff-fall-rate) while absent. Desired source per pref:
            #   'guide' -> always the guide; 'main' -> the main whenever it has seen the target recently;
            #   'auto'  -> Schmitt-triggered (promote after sustained presence, demote on loss). A switch is
            # just swapping which role's detections feed the sky-space model (switch_role = a geometry swap);
            # the model's velocity carries across untouched.
            if tracking and tracker is not None and handoff_fine is not None and not estop:
                dt = now - handoff_last
                handoff_conf += dt * (1.0 if _fine_present() else -args.handoff_fall_rate)
                handoff_conf = max(0.0, min(args.handoff_promote_s, handoff_conf))
                if track_pref == 'guide':
                    want = track_primary
                elif track_pref == 'main':
                    want = handoff_fine if handoff_conf >= args.handoff_demote_s else track_primary
                elif track_role == track_primary:                    # 'auto', currently on guide
                    want = handoff_fine if handoff_conf >= args.handoff_promote_s else track_primary
                else:                                                # 'auto', currently on main
                    want = track_primary if handoff_conf <= args.handoff_demote_s else handoff_fine
                if want != track_role and followers[want].header is not None:
                    tracker.switch_role(*track_geom(want), single_target=_single_target(want))
                    track_role, track_seen_det = want, latest_det_key[want]
                    print(f"[backend] handoff: now tracking on {want}", flush=True)
            handoff_last = now

            moving = abs(st['rate_az_rad_s']) > 1e-9 or abs(st['rate_alt_rad_s']) > 1e-9
            if tracking:
                mode = 'coast' if coasting else 'track'
            elif track_status == 'lost':
                mode = 'lost'
            elif estop:
                mode = 'estop'
            elif moving:
                mode = 'slew'
            else:
                mode = 'idle'

            # ROIs to hand the detectors: the active source's window around the predicted target, PLUS
            # the fallback (guide) role's, projected into its own frame -- so the guide keeps ROI-tracking
            # the target while the main cam drives, and a demotion is a clean handoff, not a re-acquire.
            track_rois = {}
            if tracking and tracker is not None and args.track_roi_size > 0:
                # The active source's ROI -- but only if its detector actually uses one (the main cam's
                # extended/circmean single-target detectors are full-frame, so no ROI for them).
                if track_target and not _single_target(track_role):
                    track_rois[track_role] = [round(track_target[0]), round(track_target[1]), args.track_roi_size]
                # The fallback (guide) keeps ROI-tracking too, projected into its own frame.
                fb = track_primary
                if (fb and fb != track_role and fb in detect_roles and not _single_target(fb)
                        and followers[fb].header is not None):
                    hdr = followers[fb].header
                    grpp = rad_per_px_by_role.get(fb, rad_per_px) * frame_binning(fb)
                    pp = tracker.predict_pixel_in(now, hdr.image_width / 2.0, hdr.image_height / 2.0,
                                                  grpp, args.track_sign_az, args.track_sign_alt)
                    if pp is not None:
                        track_rois[fb] = [round(pp[0]), round(pp[1]), args.track_roi_size]

            # Freeform status line for the GUI: what the backend is doing right now.
            delay_left = (max(0.0, track_delay_s - (now - track_started_t)) if tracking else 0.0)
            if tracking:
                omega = math.degrees(tracker.target_speed_rad_s()) if tracker is not None else 0.0
                hold = (" (watch-only: follow off)" if not follow_enabled
                        else (f" (mount holds {delay_left:.0f}s more)" if delay_left > 0 else ""))
                backend_status = (f"{'following' if (follow_enabled and delay_left <= 0) else 'tracking'} "
                                  f"{track_role}, target {omega:.2f} deg/s"
                                  + (" (coasting)" if coasting else "") + hold)
            elif estop:
                backend_status = "e-stopped"
            else:
                backend_status = {'lost': 'target lost', 'slew': 'slewing'}.get(mode, 'idle')

            state_slot.write({
                't_mono_ns': session_mod.mono_ns(),
                't_utc': session_mod.utc_now_iso(),
                'mode': mode,
                'status': backend_status,               # freeform backend status line for the GUI
                'detect_roles': sorted(detect_roles),   # which roles have a detector (for the GUI status)
                'enc_t_mono_ns': st.get('t_mono_ns'),   # when the angles were measured (for extrapolation)
                'enc_az_deg': round(math.degrees(st['az_rad']), 4),
                'enc_alt_deg': round(math.degrees(st['alt_rad']), 4),
                'rate_az_deg_s': round(math.degrees(st['rate_az_rad_s']), 4),
                'rate_alt_deg_s': round(math.degrees(st['rate_alt_rad_s']), 4),
                'recording': {r: _recorder_alive(r) for r in roles},   # recorder process alive
                'tracking': tracking,                   # "locked": the tracker has a target
                'following': bool(tracking and follow_enabled and delay_left <= 0),   # actively slewing
                'follow_enabled': follow_enabled,       # the persistent user intent (GUI checkbox)
                'track_delay_s': track_delay_s,         # post-lock mount-hold time (learn velocity first)
                'track_delay_left': round(delay_left, 1),
                'track_role': track_role if tracking else None,
                'track_pref': track_pref,               # GUI radio: guide / main / auto
                'track_model': args.track_model,        # target model at the next lock (+ alt)
                'track_alt_km': args.track_alt_km,
                'target_px': track_target if tracking else None,        # model prediction (blue pipper)
                'detect_px': track_detect_px if tracking else None,     # raw detection (green marker)
                # Handoff telemetry: which finer role we may promote to, the evidence accumulator, and
                # whether we're currently locked on it (track_role == handoff_fine).
                'handoff': ({'fine': handoff_fine, 'primary': track_primary,
                             'conf': round(handoff_conf, 2), 'on_fine': track_role == handoff_fine}
                            if tracking and handoff_fine else None),
                # Focus process: which role it's running on (the GUI follows <role>_focus + its metrics).
                'focus': {'running': focus_proc is not None and focus_proc.poll() is None,
                          'role': focus_role},
                # Focus sweep process (the GUI follows <role>_sweep for its prompts/result).
                'sweep': {'running': sweep_proc is not None and sweep_proc.poll() is None,
                          'role': sweep_role},
                # ROI (cx, cy, size px) around the predicted target, for detect to clamp its work to.
                'track_roi': track_rois or None,        # {role: [cx, cy, size]} -- active source + guide fallback
                'sources': dict(sources),
                'playback': {r: dict(playback_by_role[r]) for r in roles},   # per-role .ser + loop

                'cameras_available': list(cams_available),   # detected ZWO URLs, for the GUI chooser
                'mounts_available': list(mounts_available),   # detected mount URLs, for the GUI chooser
                'mount_url': mount_desired_url,               # selected mount ('sim' or a celestron URL)
                'mount_connected': mount_connected,           # is a real/sim mount driving (vs disconnected)
                'camera': dict(camera_url),                   # per-role selected camera URL (or null)
                'camera_caps': {r: published_caps(r) for r in roles},   # live camera controls for the GUI
                'debug_ser': bool(args.debug_detect_ser),      # detectors are writing <role>_debug.ser streams
                'capturing': {role: (role in cam_procs and cam_procs[role].poll() is None)
                              for role in roles},
                'cameras': {role: {'frames': followers[role].committed_count()} for role in roles},
                'optics': {r: {'fov_x_deg': round(fv[0], 4), 'fov_y_deg': round(fv[1], 4)}
                           for r, fv in fov_by_role.items()},
                'optics_sel': {r: list(sel) for r, sel in optics_sel.items()},   # for the GUI Optics tab
                'boresight_mrad': [round(boresight['x'] * 1e3, 4), round(boresight['y'] * 1e3, 4)],
            })

            # Per-second health line -- commented out so stdout carries only rare events. Uncomment
            # for live debugging.
            # if now - last_health >= 1.0:
            #     last_health = now
            #     health = ', '.join(f"{role}={followers[role].committed_count()}f" for role in roles)
            #     print(f"[backend] {mode}{' REC' if recording else ''} "
            #           f"az={math.degrees(st['az_rad']):.2f} alt={math.degrees(st['alt_rad']):.2f} | {health}",
            #           flush=True)

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
        open(focus_stop, 'w').close()              # and the focus process, if one is running
        if sweep_proc is not None and sweep_proc.poll() is None:
            try:
                sweep_proc.stdin.close()           # abort a running sweep (it exits promptly)
            except OSError:
                pass
        # Recorders need no signal: the cams' 'ended' records end their streams and they drain
        # + finalize + exit on their own (reaped below, patiently -- drain time is legitimate).
        if gui_proc is not None and gui_proc.poll() is None:
            gui_proc.terminate()
        cmd_server.close()
        state_slot.close()
        mount.close()
        for fo in followers.values():
            fo.close()
        for fo_ in det_fos.values():
            fo_.close()

        # Make sure every child is fully dead before cleanup: a still-terminating cam/detect holds
        # its .ser / sidecars open, and on Windows os.remove/rmtree fail on open files -- so a slow
        # (4K) detect that misses the graceful window used to leave whole sessions behind on exit.
        _reap(list(cam_procs.values()) + list(detect_procs.values())
              + [focus_proc, sweep_proc, gui_proc, sky_sim_proc])
        _reap(list(rec_procs.values()), graceful_s=300.0)   # a draining recorder is NOT hung

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
    """Remove the live-IPC session dir: it holds only ephemera (sidecars/detections/state) --
    recordings are archived elsewhere by the recorder and are never touched."""
    if keep:
        print(f"[backend] kept session {session_dir} (--keep)", flush=True)
    elif clean:
        shutil.rmtree(session_dir, ignore_errors=True)
        print(f"[backend] removed session {session_dir}", flush=True)


if __name__ == '__main__':
    main()
