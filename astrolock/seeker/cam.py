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
import os
import time

import numpy as np

from astrolock.seeker import control as control_mod
from astrolock.seeker import ser as ser_mod
from astrolock.seeker import session as session_mod
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def make_synthetic_frame(width, height, t, max_val=65535):
    """A faint-noise background with one bright Gaussian blob moving in a Lissajous path."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    amp_x, amp_y = width * 0.30, height * 0.30
    cx = width * 0.5 + amp_x * np.sin(t * 0.7)
    cy = height * 0.5 + amp_y * np.sin(t * 0.9 + 1.0)
    sigma = max(2.0, min(width, height) * 0.01)
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2)))
    bg = np.random.default_rng().random((height, width), dtype=np.float32) * 0.02
    img = np.clip(bg + 0.95 * blob, 0.0, 1.0)
    return (img * max_val).astype(np.uint16)


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
              neutral_wb=True):
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
    cam.set_roi(image_type=z.ASI_IMG_RAW16)          # full frame, 16-bit raw
    width, height, bins, img_type = cam.get_roi_format()

    def _set(ctrl, value, is_auto=False):
        try:
            cam.set_control_value(ctrl, value, auto=is_auto)
        except Exception as e:
            print(f"[cam] could not set control {ctrl}: {e}", flush=True)

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
    is_color = bool(info.get('IsColorCam', False))
    if neutral_wb and is_color:
        _set(z.ASI_WB_R, 50)
        _set(z.ASI_WB_B, 50)

    cam.start_video_capture()

    if is_color and not force_mono:
        color_id = _ASI_BAYER_TO_COLOR_ID.get(int(info.get('BayerPattern', 0)), ser_mod.ColorId.BAYER_RGGB)
    else:
        color_id = ser_mod.ColorId.MONO
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


def _open_sky(args, state_path=None):
    """
    Open the sky simulator as a frame source. Returns (capture, width, height, color_id,
    pixel_depth, None). capture() renders the next frame, advancing sim time at the configured
    fps. Encoder pose comes from the backend's <ts>_state.jsonl when --sky-follow-state is set
    (the manual closed loop); otherwise from a scripted pose/slew that defaults to auto-pointing
    a bright star.
    """
    import math as _math
    from astrolock.seeker.skysim import SkySim, SkySimConfig

    cfg = SkySimConfig(width=args.sky_width, height=args.sky_height,
                       focal_length_mm=args.sky_focal_mm, pixel_pitch_um=args.sky_pixel_um)
    if args.sky_epoch:
        cfg.epoch_utc = args.sky_epoch
    if args.sky_lat is not None:
        cfg.lat_deg = args.sky_lat
    if args.sky_lon is not None:
        cfg.lon_deg = args.sky_lon
    if args.sky_elev is not None:
        cfg.elev_m = args.sky_elev
    if args.sky_tle_file:
        lines = [ln.strip() for ln in open(args.sky_tle_file) if ln.strip()]
        name, l1, l2 = (lines[0], lines[1], lines[2]) if len(lines) >= 3 else ('TARGET', lines[0], lines[1])
        cfg.target_tle = (l1, l2, name)
        cfg.target_mag = args.sky_target_mag
    sim = SkySim(cfg)
    fov_x = _math.degrees(2 * _math.atan(cfg.width * cfg.pixel_pitch_um * 1e-3 / (2 * cfg.focal_length_mm)))

    if args.sky_az_deg is None or args.sky_alt_deg is None:
        import torch
        alt, az, mag = sim.sources_altaz(sim._sf_time(0.0))
        if sim.satellite is not None:
            az0, alt0 = float(az[-1]), float(alt[-1])      # point at the satellite target (last source)
        else:
            i = int(torch.argmin(mag))                     # just the brightest source (no alt limit)
            az0, alt0 = float(az[i]), float(alt[i])
    else:
        az0, alt0 = _math.radians(args.sky_az_deg), _math.radians(args.sky_alt_deg)
    rate_az, rate_alt = _math.radians(args.sky_rate_az), _math.radians(args.sky_rate_alt)
    tgt = f" target={cfg.target_tle[2]}(mag {cfg.target_mag})" if cfg.target_tle else ""
    print(f"[cam] sky sim {cfg.width}x{cfg.height} FoV {fov_x:.1f}deg epoch {cfg.epoch_utc} "
          f"point=({_math.degrees(az0):.2f},{_math.degrees(alt0):.2f})deg "
          f"slew=({args.sky_rate_az},{args.sky_rate_alt})deg/s exp={args.sky_exposure_s}s{tgt}", flush=True)

    follow = args.sky_follow_state and state_path is not None
    tailer = JsonlTailer(state_path) if follow else None
    pose = {'az': az0, 'alt': alt0, 'raz': rate_az, 'ralt': rate_alt}
    t0 = time.perf_counter()

    def capture():
        # Sim time is wall-clock from t0: if rendering lags, that's just a lower effective
        # framerate (we render the latest mount state), exactly like a slow real camera.
        t = time.perf_counter() - t0
        if tailer is not None:
            for rec in tailer.poll():            # latest backend encoder estimate wins
                pose['az'] = _math.radians(rec.get('enc_az_deg', _math.degrees(pose['az'])))
                pose['alt'] = _math.radians(rec.get('enc_alt_deg', _math.degrees(pose['alt'])))
                pose['raz'] = _math.radians(rec.get('rate_az_deg_s', 0.0))
                pose['ralt'] = _math.radians(rec.get('rate_alt_deg_s', 0.0))
            az, alt = pose['az'], pose['alt']
        else:
            az, alt = az0 + rate_az * t, alt0 + rate_alt * t
        return sim.render(t, az, alt, pose['raz'], pose['ralt'],
                          exposure_s=args.sky_exposure_s, substeps=args.sky_substeps)

    meta = {'bin': [1, 1], 'roi': [0, 0, cfg.width, cfg.height]}
    return capture, cfg.width, cfg.height, ser_mod.ColorId.MONO, 12, None, meta


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker camera capture")
    p.add_argument('--role', default='guide', help="camera role / file basename (e.g. guide, main)")
    p.add_argument('--out-dir', default=None, help="session dir to write into (default: a new sessions/<ts>)")
    p.add_argument('--source', default='synthetic', choices=['synthetic', 'zwo', 'sky'])
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
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
    # sky simulator (--source sky); defaults bake in the ISS test pass over San Carlos
    p.add_argument('--sky-epoch', default='2026-07-06T05:22:00Z', help="sky: UTC epoch ISO")
    p.add_argument('--sky-lat', type=float, default=None, help="sky: observer latitude (deg)")
    p.add_argument('--sky-lon', type=float, default=None, help="sky: observer longitude (deg)")
    p.add_argument('--sky-elev', type=float, default=None, help="sky: observer elevation (m)")
    p.add_argument('--sky-width', type=int, default=1920, help="sky: sensor width (px)")
    p.add_argument('--sky-height', type=int, default=1080, help="sky: sensor height (px)")
    p.add_argument('--sky-focal-mm', type=float, default=8.0, help="sky: lens focal length (mm); FoV = w*pitch/focal")
    p.add_argument('--sky-pixel-um', type=float, default=2.0, help="sky: sensor pixel pitch (um)")
    p.add_argument('--sky-az-deg', type=float, default=None, help="sky: encoder az (default: auto-point a bright star)")
    p.add_argument('--sky-alt-deg', type=float, default=None, help="sky: encoder alt")
    p.add_argument('--sky-rate-az', type=float, default=0.0, help="sky: scripted az slew (deg/s) for streaks")
    p.add_argument('--sky-rate-alt', type=float, default=0.0, help="sky: scripted alt slew (deg/s)")
    p.add_argument('--sky-exposure-s', type=float, default=0.1, help="sky: simulated exposure (s)")
    p.add_argument('--sky-substeps', type=int, default=6, help="sky: substeps per exposure (streak smoothness)")
    p.add_argument('--sky-tle-file', default='data/iss_25544.tle',
                   help="sky: TLE file (2 or 3 lines) for a satellite target (default: the ISS)")
    p.add_argument('--sky-target-mag', type=float, default=-4.0, help="sky: satellite target magnitude")
    p.add_argument('--sky-follow-state', action='store_true',
                   help="sky: render from the backend's encoder estimate in <ts>_state.jsonl")
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
            neutral_wb=not args.camera_wb)
    elif args.source == 'sky':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta = _open_sky(
            args, state_path=os.path.join(out_dir, session_mod.state_name(ts)))
    if frame_meta is None:                                  # synthetic: full frame, no binning
        frame_meta = {'bin': [1, 1], 'roi': [0, 0, width, height]}

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
                    if capture is not None:
                        frame = capture()
                        if frame is None:
                            print(f"[cam:{args.role}] capture timeout, skipping", flush=True)
                            continue
                    else:
                        frame = make_synthetic_frame(width, height, loop_start - start)

                    writer.write_frame(frame)                 # pixels flushed
                    sidecar.append({                           # then commit-point line
                        't_mono_ns': time.perf_counter_ns(),
                        't_utc': session_mod.utc_now_iso(),
                        'important': bool(cfg['important']),
                        **frame_meta,                          # bin + roi (sensor->frame mapping)
                    })
                    frames_in_file += 1
                    total += 1

                    if loop_start - last_status >= 1.0:
                        fps = (total - last_status_n) / (loop_start - last_status)
                        extra = f"  {get_settings()}" if get_settings else ""
                        print(f"[cam:{args.role}] {total} frames, {fps:.1f} fps, "
                              f"peak {int(frame.max())}, important={cfg['important']}{extra}", flush=True)
                        last_status, last_status_n = loop_start, total

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
