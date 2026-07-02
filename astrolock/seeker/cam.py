"""
astrolock_seeker_cam: capture one camera continuously to <ts>_<role>.ser plus a
<ts>_<role>.frames.jsonl per-frame sidecar (the commit point).

Sources:
  - synthetic (default): a moving bright blob on a faint noisy background. Needs no
    hardware, so the whole pipeline is exercisable on any machine.
  - zwo: a ZWO ASI camera via the zwoasi library (only if installed + present).

Runs standalone (point it at an --out-dir) or is launched by the backend. Stops cleanly on
Ctrl-C or when --stop-file appears (the backend's graceful-stop mechanism), patching the
SER header's frame count on the way out.

    python -m astrolock.seeker.cam --role guide --out-dir sessions/<ts> --fps 15
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from astrolock.seeker import control as control_mod
from astrolock.seeker import ser as ser_mod
from astrolock.seeker import session as session_mod
from astrolock.seeker import sidecar
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def make_synthetic_frame(width, height, t, max_val=65535):
    """A faint-noise background with one bright Gaussian blob moving in a Lissajous path (torch;
    uint16 numpy only at the SER-writer boundary)."""
    yy, xx = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                            torch.arange(width, dtype=torch.float32), indexing='ij')
    amp_x, amp_y = width * 0.30, height * 0.30
    cx = width * 0.5 + amp_x * math.sin(t * 0.7)
    cy = height * 0.5 + amp_y * math.sin(t * 0.9 + 1.0)
    sigma = max(2.0, min(width, height) * 0.01)
    blob = torch.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    bg = torch.rand((height, width)) * 0.02
    img = torch.clamp(bg + 0.95 * blob, 0.0, 1.0)
    return (img * max_val).to(torch.int32).numpy().astype(np.uint16)


def _zwo_module():
    """Import + init the zwoasi library (idempotent), with friendly errors."""
    try:
        import zwoasi
    except ImportError as e:
        raise RuntimeError("zwoasi is not installed (pip install zwoasi)") from e
    lib = os.getenv('ZWO_ASI_LIB') or 'C:/Program Files/ASIStudio/ASICamera2.dll'
    try:
        zwoasi.init(lib)
    except Exception as e:
        # init() raises if called a second time in a process; tolerate that only.
        if 'already' not in str(e).lower():
            raise RuntimeError(
                f"could not load the ASI SDK from {lib!r}; set ZWO_ASI_LIB to ASICamera2.dll"
            ) from e
    return zwoasi


def list_zwo_cameras():
    """Return [(index, name, properties_dict), ...] for attached ZWO cameras."""
    z = _zwo_module()
    out = []
    for i, name in enumerate(z.list_cameras()):
        cam = z.Camera(i)
        try:
            out.append((i, name, cam.get_camera_property()))
        finally:
            cam.close()
    return out


# ASI BayerPattern enum (RG/BG/GR/GB) -> our SER ColorId for the raw mosaic.
_ASI_BAYER_TO_COLOR_ID = {
    0: ser_mod.ColorId.BAYER_RGGB,
    1: ser_mod.ColorId.BAYER_BGGR,
    2: ser_mod.ColorId.BAYER_GRBG,
    3: ser_mod.ColorId.BAYER_GBRG,
}


def _open_zwo(camera_index, exposure_us, gain, force_mono=False,
              auto=False, auto_max_exp_ms=200, auto_max_gain=400, auto_target=100,
              neutral_wb=True, bin=1):
    """
    Open a ZWO camera for RAW16 full-frame video capture.
    Returns (capture, width, height, color_id, get_settings).
      capture()      -> uint16 ndarray or None (timeout)
      get_settings() -> str describing current exposure/gain (or None)
    For a color camera we record the raw Bayer mosaic and tag the SER with its Bayer
    ColorId (so the GUI/detector can debayer); pixel data stays single-channel.

    With auto=True the camera runs auto-exposure + auto-gain (until we have a controls UI),
    bounded by auto_max_exp_ms / auto_max_gain and aiming at auto_target brightness (0-255).
    """
    z = _zwo_module()
    names = z.list_cameras()
    if not names:
        raise RuntimeError("no ZWO cameras found (plugged in? ASI SDK installed?)")

    cam = z.Camera(camera_index)
    info = cam.get_camera_property()
    is_color = bool(info.get('IsColorCam', False))

    def _set(ctrl, value, is_auto=False):
        try:
            cam.set_control_value(ctrl, value, auto=is_auto)
        except Exception as e:
            print(f"[cam] could not set control {ctrl}: {e}", flush=True)

    # NxN binning. A color ASI keeps the Bayer mosaic when it bins (it only sums same-color wells),
    # so to get a true gray frame we ask the SDK to merge across the Bayer cell -- ASI_MONO_BIN. That
    # eats the first power-of-two of the requested bin (one Bayer cell -> one mono pixel). It only
    # works in *software* bin mode (ASI_HARDWARE_BIN=0; confirmed host-side by a short-exposure fps
    # test -- mono bin doesn't save USB, hardware Bayer bin does). We still use the API, not a
    # hand-rolled sum, so we'd get the win for free if ZWO ever moves it pre-USB. Falls back to
    # half-res Bayer if the camera lacks MonoBin.
    ctrls = cam.get_controls()
    mono_ok = is_color and not force_mono and bin >= 2 and 'MonoBin' in ctrls
    if mono_ok:
        if 'HardwareBin' in ctrls:
            _set(z.ASI_HARDWARE_BIN, 0)        # software bin: required for the cross-color merge
        _set(z.ASI_MONO_BIN, 1)                # merge the Bayer cell -> mono
    cam.set_roi(bins=bin, image_type=z.ASI_IMG_RAW16)   # full frame, 16-bit raw, NxN binned
    width, height, bins, img_type = cam.get_roi_format()

    if auto:
        _set(z.ASI_EXPOSURE, exposure_us, is_auto=True)
        _set(z.ASI_GAIN, gain, is_auto=True)
        _set(z.ASI_AUTO_MAX_EXP, auto_max_exp_ms)       # milliseconds
        _set(z.ASI_AUTO_MAX_GAIN, auto_max_gain)
        _set(z.ASI_AUTO_MAX_BRIGHTNESS, auto_target)    # target brightness, 0-255
    else:
        _set(z.ASI_EXPOSURE, exposure_us)
        _set(z.ASI_GAIN, gain)

    # ZWO bakes white balance into the RAW16 mosaic on color cams (R,B get a digital gain;
    # G is the unity reference). WB=50 is unity on the [1,99] range, so neutral WB gives
    # pristine raw -- all planes clean 12-bit-left-shifted, no WB in the data. We want this
    # for real captures (the main cam feeds tensorez, which expects genuine raw Bayer).
    if neutral_wb and is_color:
        _set(z.ASI_WB_R, 50)
        _set(z.ASI_WB_B, 50)

    cam.start_video_capture()

    if is_color and not force_mono and not mono_ok:      # Bayer: full res, or half-res if binned w/o MonoBin
        color_id = _ASI_BAYER_TO_COLOR_ID.get(int(info.get('BayerPattern', 0)), ser_mod.ColorId.BAYER_RGGB)
    else:
        color_id = ser_mod.ColorId.MONO                  # mono cam, force_mono, or MonoBin merged to gray
    # Record the camera's true ADC precision in the SER; pixels are still stored in a
    # 16-bit container (RAW16, 12-bit value left-shifted by 4; see ser.container_max).
    bit_depth = int(info.get('BitDepth', 16))
    print(f"[cam] ZWO '{info.get('Name', '?')}' {width}x{height} RAW16 {bit_depth}-bit "
          f"{'auto-exposure' if auto else f'exposure={exposure_us}us gain={gain}'} "
          f"WB={'neutral' if (neutral_wb and is_color) else 'camera'} "
          f"({color_id.name} mosaic)", flush=True)

    timeout_ms = max(1000, (auto_max_exp_ms if auto else exposure_us // 1000) + 2000)

    def capture():
        try:
            f = cam.capture_video_frame(timeout=timeout_ms)
        except z.ZWO_IOError:
            return None  # timeout; caller skips this iteration
        a = np.asarray(f, dtype=np.uint16)
        if a.ndim == 1:
            a = a.reshape((height, width))
        return a

    def get_settings():
        try:
            exp, exp_auto = cam.get_control_value(z.ASI_EXPOSURE)
            g, g_auto = cam.get_control_value(z.ASI_GAIN)
            return f"exp {exp}us{'*' if exp_auto else ''} gain {g}{'*' if g_auto else ''}"
        except Exception:
            return None

    # Sensor->frame mapping for this capture (constant); the backend uses it to map detection
    # pixels back to sensor angles. We capture full-frame, so roi origin is (0,0).
    meta = {'bin': [bins, bins], 'roi': [0, 0, width, height]}
    return capture, width, height, color_id, bit_depth, get_settings, meta


def _open_sky(args, state_path=None, mount_path=None):
    """
    Open the sky simulator as a frame source. Returns (capture, width, height, color_id,
    pixel_depth, None). capture() renders the next frame, advancing sim time at the configured
    fps. Encoder pose comes from the backend's <ts>_state.jsonl when --sky-follow-state is set
    (the manual closed loop); otherwise from a scripted pose/slew that defaults to auto-pointing
    a bright star.
    """
    import math as _math
    import torch
    from astrolock.seeker.skysim import SkySim, SkySimConfig
    from astrolock.seeker.ephemeris import SkyEphemeris

    cfg = SkySimConfig(width=args.sky_width, height=args.sky_height,
                       focal_length_mm=args.sky_focal_mm, pixel_pitch_um=args.sky_pixel_um)
    sim = SkySim(cfg)                                   # render-only; propagation lives in sky_sim.py
    ephem = SkyEphemeris(args.sky_ephemeris)            # shared, system-clock-timed source directions
    fov_x = _math.degrees(2 * _math.atan(cfg.width * cfg.pixel_pitch_um * 1e-3 / (2 * cfg.focal_length_mm)))

    # Fallback pose only for scripted (non-follow) runs; the mount drives it in closed loop.
    az0 = _math.radians(args.sky_az_deg) if args.sky_az_deg is not None else 0.0
    alt0 = _math.radians(args.sky_alt_deg) if args.sky_alt_deg is not None else _math.radians(45.0)
    rate_az, rate_alt = _math.radians(args.sky_rate_az), _math.radians(args.sky_rate_alt)
    print(f"[cam] sky sim {cfg.width}x{cfg.height} FoV {fov_x:.1f}deg ephemeris={args.sky_ephemeris} "
          f"exp={args.sky_exposure_s}s substeps={args.sky_substeps}", flush=True)

    # Prefer the sim mount's ground-truth trajectory (piecewise-linear, exact) over the backend's
    # reconstructed estimate. The mount sidecar uses 'az_deg'/'t_mono_ns'; the legacy state file
    # uses 'enc_az_deg'/'enc_t_mono_ns'. With ground truth each anchor holds until the next, so we
    # extrapolate with no upper cap (a constant-rate segment can be long with no new anchor); the
    # estimate path keeps the old 0.2 s cap as a guard against a stalled backend.
    follow_mount = getattr(args, 'sky_follow_mount', False) and mount_path is not None
    follow_state = args.sky_follow_state and state_path is not None
    tailer = JsonlTailer(mount_path if follow_mount else state_path) if (follow_mount or follow_state) else None
    ka, kl = ('az_deg', 'alt_deg') if follow_mount else ('enc_az_deg', 'enc_alt_deg')
    kt = 't_mono_ns' if follow_mount else 'enc_t_mono_ns'
    ahead_cap = 5.0 if follow_mount else 0.2
    pose = {'az': az0, 'alt': alt0, 'raz': rate_az, 'ralt': rate_alt, 'enc_t': None}
    exp, S = args.sky_exposure_s, args.sky_substeps
    fr = (torch.arange(S, dtype=torch.float64) + 0.5) / S      # (S,) substep mid-fractions
    start_ns = time.perf_counter_ns()

    def capture():
        # One shared system clock (perf_counter_ns / QPC) times everything -- the exposure substeps,
        # the mount-pose extrapolation, and the frame stamp -- so both cameras place a fast satellite
        # at the same world instant (no per-process epoch drift).
        now_ns = time.perf_counter_ns()
        if tailer is not None:
            for rec in tailer.poll():            # latest mount trajectory anchor wins
                pose['az'] = _math.radians(rec.get(ka, _math.degrees(pose['az'])))
                pose['alt'] = _math.radians(rec.get(kl, _math.degrees(pose['alt'])))
                pose['raz'] = _math.radians(rec.get('rate_az_deg_s', 0.0))
                pose['ralt'] = _math.radians(rec.get('rate_alt_deg_s', 0.0))
                pose['enc_t'] = rec.get(kt)
            ahead = 0.0
            if pose['enc_t']:                    # extrapolate the anchor pose to now (mount->cam latency)
                ahead = min(ahead_cap, max(0.0, now_ns * 1e-9 - pose['enc_t'] * 1e-9))
            az = pose['az'] + pose['raz'] * ahead
            alt = pose['alt'] + pose['ralt'] * ahead
        else:
            elapsed = (now_ns - start_ns) * 1e-9
            az, alt = az0 + rate_az * elapsed, alt0 + rate_alt * elapsed
        # Source directions at each exposure substep, looked up on the shared clock. Stars are ~static
        # across the substeps; the satellite points move -- both interpolated from the same ephemeris.
        sub_t = now_ns + (fr * exp * 1e9).to(torch.int64)    # keep now_ns exact (int64, not float64)
        ephem.update()
        dirs, mags = ephem.dirs_at(sub_t)
        frame = sim.render(az, alt, pose['raz'], pose['ralt'], dirs, mags, exposure_s=exp, substeps=S)
        return frame, int(now_ns + 0.5 * exp * 1e9)           # stamp at the exposure midpoint

    meta = {'bin': [args.bin, args.bin], 'roi': [0, 0, cfg.width, cfg.height]}
    return capture, cfg.width, cfg.height, ser_mod.ColorId.MONO, 12, None, meta


def _open_playback(args):
    """
    Replay an existing .ser as if it were a live camera, paced by its frame timestamps (x
    --playback-speed). Loops at the end. Lets the whole live pipeline (detect, gui, tracking)
    run on a recording -- the easy way to review a capture with detections overlaid.
    """
    src = ser_mod.SerReader(args.playback_ser)
    h = src.header
    n = src.frames_on_disk()
    if n < 1:
        raise RuntimeError(f"no frames in {args.playback_ser}")
    recs = (sidecar.read_complete_lines(args.playback_ser[:-len('.ser')] + '.frames.jsonl')
            if os.path.exists(args.playback_ser[:-len('.ser')] + '.frames.jsonl') else [])
    times = [r.get('t_mono_ns') for r in recs]
    meta = {'bin': [1, 1], 'roi': [0, 0, h.image_width, h.image_height]}
    for r in recs:
        if 'bin' in r:
            meta = {'bin': r['bin'], 'roi': r.get('roi', meta['roi'])}
            break
    print(f"[cam] playback {os.path.basename(args.playback_ser)} {h.image_width}x{h.image_height} "
          f"{n} frames x{args.playback_speed}", flush=True)

    st = {'i': 0, 'wall0': None, 't0': None}

    def capture():
        if st['i'] >= n:
            if not args.playback_loop:
                return None                       # one-shot: signal end of stream
            st['i'], st['wall0'] = 0, None        # loop back to the start
        i = st['i']
        frame = src.read_frame(i)
        if times and i < len(times) and times[i] is not None:   # pace to the recorded cadence
            if st['wall0'] is None:
                st['wall0'], st['t0'] = time.perf_counter(), times[i]
            delay = (st['wall0'] + (times[i] - st['t0']) * 1e-9 / max(1e-6, args.playback_speed)
                     - time.perf_counter())
            if delay > 0:
                time.sleep(delay)
        st['i'] = i + 1
        return frame

    return (capture, h.image_width, h.image_height, ser_mod.ColorId(h.color_id),
            h.pixel_depth_per_plane, None, meta)


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker camera capture")
    p.add_argument('--role', default='guide', help="camera role / file basename (e.g. guide, main)")
    p.add_argument('--out-dir', default=None, help="session dir to write into (default: a new sessions/<ts>)")
    p.add_argument('--source', default='synthetic', choices=['synthetic', 'zwo', 'sky', 'playback'])
    p.add_argument('--playback-ser', default=None, help="playback: the .ser file to replay")
    p.add_argument('--playback-speed', type=float, default=1.0, help="playback: speed multiplier")
    p.add_argument('--playback-loop', action='store_true', help="playback: loop instead of stopping at the end")
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--bin', type=int, default=1,
                   help="NxN binning. sim/synthetic: --width/--height are already the binned size; this "
                        "just records bin=[N,N] in the frame metadata. zwo: sets hardware binning "
                        "(a color cam binned >1 reads out mono).")
    p.add_argument('--fps', type=float, default=15.0)
    p.add_argument('--frame-limit', type=int, default=-1,
                   help="frames for the current file before rolling over (-1 = unlimited)")
    p.add_argument('--file-limit', type=int, default=1,
                   help="how many (more) files to capture; exit when 0 (-1 = unlimited)")
    p.add_argument('--important', type=int, default=1,
                   help="1 marks frames important (kept); 0 = not recording (auto-deletable)")
    p.add_argument('--control-file', default=None,
                   help="JSONL of live setting updates to merge (or '-' for stdin)")
    p.add_argument('--exposure-us', type=int, default=2000, help="zwo only (manual exposure)")
    p.add_argument('--gain', type=int, default=190, help="zwo only (manual gain)")
    p.add_argument('--auto', action='store_true', help="zwo: enable auto-exposure + auto-gain")
    p.add_argument('--auto-max-exp-ms', type=int, default=200, help="auto: max exposure (ms)")
    p.add_argument('--auto-max-gain', type=int, default=400, help="auto: max gain")
    p.add_argument('--auto-target', type=int, default=100, help="auto: target brightness (0-255)")
    # sky simulator (--source sky): the camera only renders point sources it reads from the shared
    # sky_sim ephemeris. It has no notion of stars vs satellites, nor of epoch/site/TLE -- sky_sim
    # owns all propagation. These args are just this camera's optics + pose + exposure.
    p.add_argument('--sky-width', type=int, default=1920, help="sky: sensor width (px)")
    p.add_argument('--sky-height', type=int, default=1080, help="sky: sensor height (px)")
    p.add_argument('--sky-focal-mm', type=float, default=8.0, help="sky: lens focal length (mm); FoV = w*pitch/focal")
    p.add_argument('--sky-pixel-um', type=float, default=2.0, help="sky: sensor pixel pitch (um)")
    p.add_argument('--sky-az-deg', type=float, default=None, help="sky: fallback encoder az for scripted (non-follow) runs")
    p.add_argument('--sky-alt-deg', type=float, default=None, help="sky: fallback encoder alt")
    p.add_argument('--sky-rate-az', type=float, default=0.0, help="sky: scripted az slew (deg/s) for streaks")
    p.add_argument('--sky-rate-alt', type=float, default=0.0, help="sky: scripted alt slew (deg/s)")
    p.add_argument('--sky-exposure-s', type=float, default=0.1, help="sky: simulated exposure (s)")
    p.add_argument('--sky-substeps', type=int, default=6, help="sky: substeps per exposure (streak smoothness)")
    p.add_argument('--sky-ephemeris', default=None,
                   help="sky: shared source-direction ephemeris (JSONL) published by sky_sim")
    p.add_argument('--sky-follow-state', action='store_true',
                   help="sky: render from the backend's encoder estimate in <ts>_state.jsonl")
    p.add_argument('--sky-follow-mount', action='store_true',
                   help="sky: render from the sim mount's ground-truth trajectory in <ts>_sim_mount.jsonl "
                        "(piecewise-linear; preferred over --sky-follow-state for the sim mount)")
    p.add_argument('--camera-index', type=int, default=0, help="zwo camera index")
    p.add_argument('--camera-wb', action='store_true',
                   help="zwo: keep the camera's white balance (default: neutral WB for pristine raw)")
    p.add_argument('--mono', action='store_true', help="store raw mosaic as MONO (no Bayer tag)")
    p.add_argument('--list-cameras', action='store_true', help="list ZWO cameras and exit")
    args = p.parse_args(argv)

    if args.list_cameras:
        cams = list_zwo_cameras()
        if not cams:
            print("no ZWO cameras found")
        for i, name, info in cams:
            print(f"  [{i}] {name}  {info.get('MaxWidth')}x{info.get('MaxHeight')}  "
                  f"{'color' if info.get('IsColorCam') else 'mono'}  bayer={info.get('BayerPattern')}")
        return

    if args.out_dir is None:
        out_dir, ts = session_mod.new_session_dir()
    else:
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        ts = os.path.basename(os.path.normpath(out_dir))

    capture = None
    get_settings = None
    width, height = args.width, args.height
    color_id = ser_mod.ColorId.MONO
    pixel_depth = 16  # synthetic frames are full-range 16-bit
    frame_meta = None
    if args.source == 'zwo':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta = _open_zwo(
            args.camera_index, args.exposure_us, args.gain, force_mono=args.mono,
            auto=args.auto, auto_max_exp_ms=args.auto_max_exp_ms,
            auto_max_gain=args.auto_max_gain, auto_target=args.auto_target,
            neutral_wb=not args.camera_wb, bin=args.bin)
    elif args.source == 'sky':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta = _open_sky(
            args, state_path=os.path.join(out_dir, session_mod.state_name(ts)),
            mount_path=os.path.join(out_dir, session_mod.sim_mount_name(ts)))
    elif args.source == 'playback':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta = _open_playback(args)
    if frame_meta is None:                                  # synthetic: rendered at the binned size
        frame_meta = {'bin': [args.bin, args.bin], 'roi': [0, 0, width, height]}

    control = control_mod.ControlReader(args.control_file) if args.control_file else None
    cfg = {'frame_limit': args.frame_limit, 'file_limit': args.file_limit,
           'important': args.important, 'fps': args.fps}

    print(f"[cam:{args.role}] {args.source} {width}x{height} {color_id.name} {pixel_depth}-bit "
          f"frame_limit={cfg['frame_limit']} file_limit={cfg['file_limit']} "
          f"important={cfg['important']} control={args.control_file} -> {out_dir}", flush=True)

    start = time.perf_counter()
    total = 0
    last_status = start
    last_status_n = 0
    stop = False
    try:
        while not stop and cfg['file_limit'] != 0:
            seg_ts = session_mod.segment_stamp()
            ser_path = os.path.join(out_dir, session_mod.ser_name(seg_ts, args.role))
            frames_path = os.path.join(out_dir, session_mod.frames_name(seg_ts, args.role))
            writer = ser_mod.SerWriter(ser_path, width, height,
                                       color_id=color_id, pixel_depth_per_plane=pixel_depth)
            sidecar = JsonlWriter(frames_path)
            frames_in_file = 0
            rolled = False
            try:
                while True:
                    if control is not None:
                        for cmd in control.drain():
                            if cmd.get('stop'):
                                stop = True
                            for k in ('frame_limit', 'file_limit', 'important', 'fps'):
                                if k in cmd:
                                    cfg[k] = cmd[k]
                    if stop or cfg['file_limit'] == 0:    # {stop} or shutdown -> finalize + exit
                        stop = True
                        break

                    loop_start = time.perf_counter()
                    cap_t_ns = None                            # a source may supply the true frame time
                    if capture is not None:
                        frame = capture()
                        if isinstance(frame, tuple):           # (frame, t_mono_ns) -- e.g. sim exposure midpoint
                            frame, cap_t_ns = frame
                        if frame is None:
                            if args.source == 'playback' and not args.playback_loop:
                                print(f"[cam:{args.role}] playback complete", flush=True)
                                stop = True
                                break
                            print(f"[cam:{args.role}] capture timeout, skipping", flush=True)
                            continue
                    else:
                        frame = make_synthetic_frame(width, height, loop_start - start)

                    writer.write_frame(frame)                 # pixels flushed
                    sidecar.append({                           # then commit-point line
                        't_mono_ns': cap_t_ns if cap_t_ns is not None else time.perf_counter_ns(),
                        't_utc': session_mod.utc_now_iso(),
                        'important': bool(cfg['important']),
                        **frame_meta,                          # bin + roi (sensor->frame mapping)
                    })
                    frames_in_file += 1
                    total += 1

                    # Per-second status -- commented out to keep stdout for rare events. Uncomment
                    # for live debugging (fps / peak / exposure-gain).
                    # if loop_start - last_status >= 1.0:
                    #     fps = (total - last_status_n) / (loop_start - last_status)
                    #     extra = f"  {get_settings()}" if get_settings else ""
                    #     print(f"[cam:{args.role}] {total} frames, {fps:.1f} fps, "
                    #           f"peak {int(frame.max())}, important={cfg['important']}{extra}", flush=True)
                    #     last_status, last_status_n = loop_start, total

                    if cfg['frame_limit'] != -1 and frames_in_file >= cfg['frame_limit']:
                        rolled = True
                        break

                    period = 1.0 / cfg['fps'] if cfg['fps'] > 0 else 0.0
                    if period:
                        sleep = period - (time.perf_counter() - loop_start)
                        if sleep > 0:
                            time.sleep(sleep)
            finally:
                writer.close()
                sidecar.close()
            print(f"[cam:{args.role}] segment {os.path.basename(ser_path)}: {frames_in_file} frames", flush=True)
            if rolled and not stop and cfg['file_limit'] != -1:
                cfg['file_limit'] -= 1                 # consumed one of our file budget
    except KeyboardInterrupt:
        print(f"[cam:{args.role}] interrupted", flush=True)
    finally:
        if control is not None:
            control.close()
        print(f"[cam:{args.role}] done, {total} frames total", flush=True)


if __name__ == '__main__':
    main()
