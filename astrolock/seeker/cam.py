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
import json
import math
import os
import time

import numpy as np
import torch

from astrolock.seeker import control as control_mod
from astrolock.seeker import ser as ser_mod
from astrolock.seeker import framestream
from astrolock.seeker import session as session_mod
from astrolock.seeker import sidecar
from astrolock.seeker.sidecar import JsonlWriter, JsonlTailer


def resolve_device(name):
    """Map a --device string to a torch.device. 'auto' (the default) -> cuda when present, else cpu."""
    if name in (None, '', 'auto'):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def set_priority(level):
    """Raise this process's scheduling priority. The capture loop has a hard deadline: a real camera
    free-runs, so if we're descheduled and call capture_video_frame() late, its buffer overflows and
    the CAMERA drops frames at the source -- lost forever, and visible as a lower, jittery fps rather
    than a clean stall. 'realtime' is deliberately not offered: it can starve the OS (input included).
    A failure here is a warning, not fatal -- capture still works, just at normal priority."""
    if level in (None, '', 'normal'):
        return
    if os.name == 'nt':
        import ctypes
        classes = {'above': 0x00008000, 'high': 0x00000080}     # ABOVE_NORMAL / HIGH_PRIORITY_CLASS
        k32 = ctypes.windll.kernel32
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), classes[level]):
            print(f"[cam] could not set priority {level!r}: {ctypes.WinError()}", flush=True)
            return
    else:
        try:
            os.nice({'above': -5, 'high': -10}[level])          # needs privileges; warn if denied
        except PermissionError as e:
            print(f"[cam] could not set priority {level!r}: {e}", flush=True)
            return
    print(f"[cam] process priority: {level}", flush=True)


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


_zwo = None


def _zwo_module():
    """Import zwoasi and load its SDK library from the path we choose (ZWO_ASI_LIB or the ASIStudio
    default). zwoasi resolves the DLL at import via find_library('ASICamera2'), which on Windows searches
    only PATH -- so we make the SDK dir findable there first (appended, so it never shadows a system
    copy), and the import loads it cleanly. We also add_dll_directory(sdk_dir) (covers the WinAPI
    LoadLibrary search + the DLL's dependencies) and, if the import still didn't load it, init() from the
    explicit path -- which then raises loudly on a genuinely bad path. Cached, so the setup runs once."""
    global _zwo
    if _zwo is not None:
        return _zwo
    lib = os.getenv('ZWO_ASI_LIB') or 'C:/Program Files/ASIStudio/ASICamera2.dll'
    sdk_dir = os.path.dirname(lib)
    if os.path.isdir(sdk_dir):
        if sdk_dir not in os.environ.get('PATH', '').split(os.pathsep):
            os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + sdk_dir   # find_library reads PATH
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(sdk_dir)                # WinAPI LoadLibrary search + the DLL's deps
    import zwoasi
    if zwoasi.zwolib is None:                            # import didn't load it -> load from the exact path
        zwoasi.init(lib)
    _zwo = zwoasi
    return _zwo


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


def _zwo_available():
    """True if the zwoasi *module* is installed -- checked WITHOUT importing it, because importing
    triggers zwoasi's load-the-DLL-on-import side effect, which must wait until _zwo_module() has put
    the SDK dir on the search path. A missing/broken DLL is deliberately NOT hidden here; it surfaces
    loudly when we actually init/use the library, not as a silent 'no cameras'."""
    import importlib.util
    return importlib.util.find_spec('zwoasi') is not None


def list_zwo_camera_names():
    """ZWO camera model names in enumeration order. Safe to call while a camera is in use -- it only
    *lists* (never opens one). Returns [] ONLY when the zwoasi module isn't installed (sim-only rig);
    a missing/broken SDK DLL or a driver error propagates loudly instead of masquerading as 'no cameras'.
    None attached is not an error -- list_cameras() just returns []."""
    if not _zwo_available():
        return []
    return list(_zwo_module().list_cameras())


def zwo_camera_urls(names=None):
    """Model-qualified URLs for the attached ZWO cameras: 'zwo:<model>', with '#<k>' appended when
    several of the same model are present so two identical cams stay distinguishable, e.g.
    'zwo:ZWO ASI678MC#0'. Keying on the model (not the USB endpoint) keeps per-camera settings stable."""
    names = list_zwo_camera_names() if names is None else names
    seen, urls = {}, []
    for nm in names:
        if names.count(nm) > 1:
            k = seen.get(nm, 0)
            seen[nm] = k + 1
            urls.append(f"zwo:{nm}#{k}")
        else:
            urls.append(f"zwo:{nm}")
    return urls


def _resolve_zwo_index(camera_url, names):
    """Map a 'zwo:<model>[#k]' URL (or a bare integer) to a list_cameras() index, else None."""
    if not camera_url or not camera_url.startswith('zwo:'):
        return None
    spec = camera_url[len('zwo:'):]
    inst = 0
    if '#' in spec:
        spec, _, k = spec.rpartition('#')
        inst = int(k) if k.isdigit() else 0
    matches = [i for i, nm in enumerate(names) if nm == spec]
    if matches:
        return matches[min(inst, len(matches) - 1)]
    return int(spec) if spec.isdigit() and int(spec) < len(names) else None


# ASI BayerPattern enum (RG/BG/GR/GB) -> our SER ColorId for the raw mosaic.
_ASI_BAYER_TO_COLOR_ID = {
    0: ser_mod.ColorId.BAYER_RGGB,
    1: ser_mod.ColorId.BAYER_BGGR,
    2: ser_mod.ColorId.BAYER_GRBG,
    3: ser_mod.ColorId.BAYER_GBRG,
}


def _open_zwo(camera_index, exposure_us, gain, force_mono=False,
              auto=False, auto_max_exp_ms=200, auto_max_gain=400, auto_target=100,
              neutral_wb=True, bin=1, camera_url=None, roi=None, capture_bits=16):
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

    idx = _resolve_zwo_index(camera_url, names)          # a model URL wins; else fall back to the index
    if idx is None:
        if camera_url:
            print(f"[cam] camera-url {camera_url!r} not found among {names}; using index {camera_index}",
                  flush=True)
        idx = camera_index
    cam = z.Camera(idx)
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
    # RAW8 halves the bytes/frame over USB (a big framerate win); RAW16 keeps the full ADC precision
    # (12 bits left-justified into 16). Either way the value stays LINEAR -- gamma is pinned at 50.
    img_t = z.ASI_IMG_RAW8 if int(capture_bits) == 8 else z.ASI_IMG_RAW16
    cap_dtype = np.uint8 if int(capture_bits) == 8 else np.uint16
    roi_win = None
    if roi:
        try:
            roi_win = [int(v) for v in str(roi).split(',')]   # backend's native centered window [x0,y0,w,h]
        except ValueError:
            # The ROI comes from OUR backend -- malformed means a bug, not user input to shrug off.
            print(f"[cam] malformed --roi {roi!r}; using full frame", flush=True)
            roi_win = None
    if roi_win:
        _x0, _y0, _w, _h = roi_win
        out_w = max(8, (_w // bin) // 8 * 8)            # ASI ROI is in output px: width % 8, height % 2,
        out_h = max(2, (_h // bin) // 2 * 2)            # start even to keep the Bayer phase on color cams
        cam.set_roi(start_x=(_x0 // bin) & ~1, start_y=(_y0 // bin) & ~1,
                    width=out_w, height=out_h, bins=bin, image_type=img_t)
    else:
        cam.set_roi(bins=bin, image_type=img_t)         # full frame, NxN binned, RAW8 or RAW16
    width, height, bins, img_type = cam.get_roi_format()

    # This application always wants a fixed, user-specified exposure/gain -- never the camera's auto
    # mode (a moving satellite would fight an auto loop). Set both with auto explicitly OFF, so a camera
    # that came up in auto (or a stale setting) is forced to manual and uses exactly the values we ask.
    _set(z.ASI_EXPOSURE, exposure_us, is_auto=False)
    _set(z.ASI_GAIN, gain, is_auto=False)

    # ZWO bakes white balance into the RAW16 mosaic on color cams (R,B get a digital gain;
    # G is the unity reference). WB=50 is unity on the [1,99] range, so neutral WB gives
    # pristine raw -- all planes clean 12-bit-left-shifted, no WB in the data. We want this
    # for real captures (the main cam feeds tensorez, which expects genuine raw Bayer).
    if neutral_wb and is_color:
        _set(z.ASI_WB_R, 50)
        _set(z.ASI_WB_B, 50)

    # Neutral gamma (50 = unity on ASI's [0,100] range): keep the pixels LINEAR. Everything downstream
    # (detector, focus, tensorez) assumes linear counts; we apply display gamma only at the GUI tonemap.
    if 'Gamma' in ctrls:
        _set(z.ASI_GAMMA, 50)

    # High-speed readout OFF by default -- a deterministic starting point rather than whatever the
    # camera powered up in. On the 678/USB3 it doesn't raise RAW8 fps anyway (already bandwidth-bound)
    # and trades a little read noise / bit depth for the faster clock. Toggle it live in the GUI.
    if 'HighSpeedMode' in ctrls:
        _set(z.ASI_HIGH_SPEED_MODE, 0)

    cam.start_video_capture()

    if is_color and not force_mono and not mono_ok:      # Bayer: full res, or half-res if binned w/o MonoBin
        color_id = _ASI_BAYER_TO_COLOR_ID.get(int(info.get('BayerPattern', 0)), ser_mod.ColorId.BAYER_RGGB)
    else:
        color_id = ser_mod.ColorId.MONO                  # mono cam, force_mono, or MonoBin merged to gray
    # Recorded SER precision: RAW8 stores a true 8-bit container; RAW16 stores the camera's ADC
    # precision left-justified in 16 bits (12-bit value << 4; see ser.container_max).
    adc_bits = int(info.get('BitDepth', 16))
    pixel_depth = 8 if int(capture_bits) == 8 else adc_bits
    print(f"[cam] ZWO '{info.get('Name', '?')}' {width}x{height} "
          f"{'RAW8' if int(capture_bits) == 8 else f'RAW16 {adc_bits}-bit'} "
          f"{'auto-exposure' if auto else f'exposure={exposure_us}us gain={gain}'} "
          f"WB={'neutral' if (neutral_wb and is_color) else 'camera'} "
          f"({color_id.name} mosaic)", flush=True)

    timeout_ms = max(1000, (auto_max_exp_ms if auto else exposure_us // 1000) + 2000)

    def capture():
        try:
            f = cam.capture_video_frame(timeout=timeout_ms)
        except z.ZWO_IOError:
            return None  # timeout; caller skips this iteration
        a = np.asarray(f, dtype=cap_dtype)
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

    # Live-settable controls the GUI renders, with the device's own ranges (from get_controls()).
    def _rng(nm, dlo, dhi):
        c = ctrls.get(nm, {})
        return c.get('MinValue', dlo), c.get('MaxValue', dhi)
    exp_lo, exp_hi = _rng('Exposure', 32, 2_000_000)             # microseconds
    gain_lo, gain_hi = _rng('Gain', 0, 570)
    caps = [
        {'name': 'exposure', 'label': 'Exposure', 'kind': 'number', 'unit': 'ms', 'scale': 'log',
         'min': exp_lo / 1000.0, 'max': exp_hi / 1000.0, 'value': exposure_us / 1000.0, 'live': True},
        {'name': 'gain', 'label': 'Gain', 'kind': 'number', 'unit': '', 'scale': 'linear',
         'min': float(gain_lo), 'max': float(gain_hi), 'value': float(gain), 'live': True},
    ]

    def _cur(nm, ctrl_id):
        """Current value of a control, or None if this camera doesn't have it."""
        if nm not in ctrls:
            return None
        try:
            return cam.get_control_value(ctrl_id)[0]
        except Exception:
            return None

    # USB bandwidth throttle (BANDWIDTHOVERLOAD, %): raising it lets the camera push more over USB;
    # the SDK's default is conservative. The main framerate lever for full-frame captures.
    bw_cur = _cur('BandWidth', z.ASI_BANDWIDTHOVERLOAD)
    if bw_cur is not None:
        bw_lo, bw_hi = _rng('BandWidth', 40, 100)
        caps.append({'name': 'bandwidth', 'label': 'USB Bandwidth', 'kind': 'number', 'unit': '%',
                     'scale': 'linear', 'min': float(bw_lo), 'max': float(bw_hi),
                     'value': float(bw_cur), 'live': True})
    # High-speed readout mode: faster ADC/readout clock (on many models tied to reduced bit depth /
    # a bit more read noise). Exposed to experiment with the speed/quality trade.
    hs_cur = _cur('HighSpeedMode', z.ASI_HIGH_SPEED_MODE)
    if hs_cur is not None:
        caps.append({'name': 'highspeed', 'label': 'High Speed Mode', 'kind': 'bool',
                     'value': bool(hs_cur), 'live': True})

    def set_control(name, value):
        if name == 'exposure':
            us = max(1, int(round(value * 1000)))
            _set(z.ASI_EXPOSURE, us)
            return us / 1000.0
        if name == 'gain':
            g = int(round(value))
            _set(z.ASI_GAIN, g)
            return float(g)
        if name == 'bandwidth':
            v = int(round(value))
            _set(z.ASI_BANDWIDTHOVERLOAD, v)
            return float(v)
        if name == 'highspeed':
            on = 1 if value else 0
            _set(z.ASI_HIGH_SPEED_MODE, on)
            return bool(on)
        return None

    # Full control snapshot to the log on connect -- the fastest way to diagnose exposure/gamma/
    # bit-depth surprises (is gamma really 50? what does high-speed mode change? is anything on auto?).
    print(f"[cam] ZWO '{info.get('Name', '?')}' controls @ connect "
          f"(img_type={'RAW8' if int(capture_bits) == 8 else 'RAW16'}):", flush=True)
    for nm in sorted(ctrls):
        c = ctrls[nm]
        try:
            val, is_auto = cam.get_control_value(c['ControlType'])
        except Exception:
            val, is_auto = '?', False
        span = (f"[{c.get('MinValue')}..{c.get('MaxValue')}]"
                if c.get('IsWritable', True) else "(read-only)")
        print(f"    {nm:22s} = {val}{'  (auto)' if is_auto else ''}   {span}", flush=True)

    controls = {'source': 'zwo', 'controls': caps, 'set': set_control}
    # Sensor->frame mapping for this capture (constant); the backend uses it to map detection
    # pixels back to sensor angles. We capture full-frame, so roi origin is (0,0).
    meta = {'bin': [bins, bins], 'roi': [0, 0, width, height]}
    return capture, width, height, color_id, pixel_depth, get_settings, meta, controls


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
    from astrolock.seeker.almanac import SkyAlmanac

    adc_bits = 8 if int(getattr(args, 'bit_depth', 16)) == 8 else 12   # 8-bit fast mode, else native 12-bit
    cfg = SkySimConfig(width=args.sky_width, height=args.sky_height,
                       focal_length_mm=args.sky_focal_mm, pixel_pitch_um=args.sky_pixel_um,
                       aperture_mm=args.sky_aperture_mm, psf_wavelength_nm=args.sky_psf_wavelength_nm,
                       central_obstruction=args.sky_central_obstruction, spider_vanes=args.sky_spider_vanes,
                       vane_width_frac=args.sky_vane_width_frac,
                       qe=args.sky_qe, full_well_e=args.sky_full_well_e, read_noise_e=args.sky_read_noise_e,
                       sky_mag_arcsec2=args.sky_sky_mag, adc_bits=adc_bits,
                       psf_sigma_px=args.sky_psf_sigma_px, seeing_r0_m=args.sky_seeing_r0_m)

    # Model of the camera's data link (USB3): frames can't be DELIVERED faster than the link
    # carries their bytes, whatever the exposure allows -- the real rig's effective fps ceiling
    # (a 16.6MB full-res frame over ~400MB/s = ~24fps, matching the observed hardware).
    frame_bytes = cfg.width * cfg.height * (1 if adc_bits <= 8 else 2)   # container bytes/frame (8- or 16-bit)
    bw_interval = (frame_bytes / (args.sim_cam_bandwidth_limit * 1e6)
                   if args.sim_cam_bandwidth_limit > 0 else 0.0)
    _bw = {'last': None}

    def _bw_wait():
        if bw_interval:
            if _bw['last'] is not None:
                d = _bw['last'] + bw_interval - time.perf_counter()
                if d > 0:
                    time.sleep(d)
            _bw['last'] = time.perf_counter()

    _live = {'exp': args.sky_exposure_s, 'gain_cb': float(args.sky_gain_cb)}   # exposure (s) + gain (cB), live

    if args.sim_cam_noop:
        # I/O-stress mode: keep ALL the pacing (exposure floors fps; the bandwidth model caps it;
        # --fps still applies in the main loop) but skip the expensive render -- one canned noise
        # frame, delivered at full commanded speed. Reproduces the real rig's sustained-write
        # pressure on the disk without the sim's GPU cost throttling it.
        import numpy as _np
        _hi = 250 if adc_bits <= 8 else 400
        noop_frame = _np.random.default_rng(0).integers(
            150, _hi, size=(cfg.height, cfg.width), dtype=_np.uint8 if adc_bits <= 8 else _np.uint16)
        print(f"[cam] sky sim NOOP {cfg.width}x{cfg.height} ({frame_bytes / 1e6:.1f} MB/frame, "
              f"link {args.sim_cam_bandwidth_limit:g} MB/s -> <= "
              f"{(1.0 / bw_interval) if bw_interval else float('inf'):.1f} fps)", flush=True)

        def capture_noop():
            now_ns = session_mod.mono_ns()
            exp = _live['exp']
            _bw_wait()
            return noop_frame, int(now_ns + 0.5 * exp * 1e9), now_ns + int(exp * 1e9)

        caps = [{'name': 'exposure', 'label': 'Exposure', 'kind': 'number', 'unit': 'ms', 'scale': 'log',
                 'min': 0.01, 'max': 2000.0, 'value': _live['exp'] * 1000.0, 'live': True},
                {'name': 'gain', 'label': 'Gain', 'kind': 'number', 'unit': 'cB', 'scale': 'linear',
                 'min': 0.0, 'max': 600.0, 'value': _live['gain_cb'], 'live': True}]

        def set_control_noop(name, value):
            if name == 'exposure':
                _live['exp'] = max(1e-5, value / 1000.0)
                return _live['exp'] * 1000.0
            if name == 'gain':                              # accepted for parity; nothing to brighten
                _live['gain_cb'] = max(0.0, value)
                return _live['gain_cb']
            return None
        controls = {'source': 'sky', 'controls': caps, 'set': set_control_noop}
        meta = {'bin': [args.bin, args.bin], 'roi': [0, 0, cfg.width, cfg.height]}
        return capture_noop, cfg.width, cfg.height, ser_mod.ColorId.MONO, adc_bits, None, meta, controls

    device = resolve_device(getattr(args, 'device', 'auto'))
    sim = SkySim(cfg, device=device)                   # render-only; propagation lives in sky_sim.py
    almanac = SkyAlmanac(args.sky_almanac)              # shared, system-clock-timed source directions
    fov_x = _math.degrees(2 * _math.atan(cfg.width * cfg.pixel_pitch_um * 1e-3 / (2 * cfg.focal_length_mm)))

    # Fallback pose only for scripted (non-follow) runs; the mount drives it in closed loop.
    az0 = _math.radians(args.sky_az_deg) if args.sky_az_deg is not None else 0.0
    alt0 = _math.radians(args.sky_alt_deg) if args.sky_alt_deg is not None else _math.radians(45.0)
    rate_az, rate_alt = _math.radians(args.sky_rate_az), _math.radians(args.sky_rate_alt)
    print(f"[cam] sky sim {cfg.width}x{cfg.height} FoV {fov_x:.1f}deg almanac={args.sky_almanac} "
          f"exp={args.sky_exposure_s}s substeps={args.sky_substeps} device={device}", flush=True)

    # Prefer the sim mount's ground-truth trajectory (piecewise-linear, exact) over the backend's
    # reconstructed estimate. The mount sidecar uses 'az_deg'/'t_mono_ns'; the legacy state file
    # uses 'enc_az_deg'/'enc_t_mono_ns'. With ground truth each anchor holds until the next, so we
    # extrapolate with no upper cap (a constant-rate segment can be long with no new anchor); the
    # estimate path keeps the old 0.2 s cap as a guard against a stalled backend.
    follow_mount = getattr(args, 'sky_follow_mount', False) and mount_path is not None
    follow_state = args.sky_follow_state and getattr(args, 'state_shm', None)
    tailer = JsonlTailer(mount_path) if follow_mount else None
    slot = None
    if follow_state and not follow_mount:                    # backend pose via the shm state slot
        try:
            slot = framestream.LatestSlot(name=args.state_shm)
        except ValueError:
            slot = None                                      # standalone run: scripted pose below
    ka, kl = ('az_deg', 'alt_deg') if follow_mount else ('enc_az_deg', 'enc_alt_deg')
    kt = 't_mono_ns' if follow_mount else 'enc_t_mono_ns'
    ahead_cap = 5.0 if follow_mount else 0.2
    pose = {'az': az0, 'alt': alt0, 'raz': rate_az, 'ralt': rate_alt, 'enc_t': None}
    sim.gain_mult = 10.0 ** (_live['gain_cb'] / 200.0)         # centibels -> linear signal multiplier
    S = args.sky_substeps
    fr = (torch.arange(S, dtype=torch.float64) + 0.5) / S      # (S,) substep mid-fractions
    start_ns = session_mod.mono_ns()

    def capture():
        # One shared system clock (perf_counter_ns / QPC) times everything -- the exposure substeps,
        # the mount-pose extrapolation, and the frame stamp -- so both cameras place a fast satellite
        # at the same world instant (no per-process epoch drift).
        now_ns = session_mod.mono_ns()
        if slot is not None or tailer is not None:
            if slot is not None:                 # state slot: latest-wins, pure memory read
                got = slot.read()
                recs = (got[1],) if got else ()
            else:
                recs = tailer.poll()
            for rec in recs:                     # latest mount trajectory anchor wins
                pose['az'] = _math.radians(rec.get(ka, _math.degrees(pose['az'])))
                pose['alt'] = _math.radians(rec.get(kl, _math.degrees(pose['alt'])))
                pose['raz'] = _math.radians(rec.get('rate_az_deg_s', 0.0))
                pose['ralt'] = _math.radians(rec.get('rate_alt_deg_s', 0.0))
                pose['enc_t'] = rec.get(kt)
            # Extrapolate the HELD pose every frame (not just on anchor ticks): between anchors
            # the mount keeps moving at the anchor rate, and a fresh-record-only pose froze the
            # rendered sky under a slewing mount (tracker chased its own motion to the rate cap).
            ahead = 0.0
            if pose['enc_t']:                    # anchor pose -> now (mount->cam latency)
                ahead = min(ahead_cap, max(0.0, now_ns * 1e-9 - pose['enc_t'] * 1e-9))
            az = pose['az'] + pose['raz'] * ahead
            alt = pose['alt'] + pose['ralt'] * ahead
        else:
            elapsed = (now_ns - start_ns) * 1e-9
            az, alt = az0 + rate_az * elapsed, alt0 + rate_alt * elapsed
        # Source directions at each exposure substep, looked up on the shared clock. Stars are ~static
        # across the substeps; the satellite points move -- both interpolated from the same almanac.
        exp = _live['exp']
        sub_t = now_ns + (fr * exp * 1e9).to(torch.int64)    # keep now_ns exact (int64, not float64)
        almanac.update()
        dirs, mags = almanac.dirs_at(sub_t)
        frame = sim.render(az, alt, pose['raz'], pose['ralt'], dirs, mags, exposure_s=exp, substeps=S)
        _bw_wait()                              # the data link caps delivery rate, whatever the exposure
        # (frame, stamp, available-at): the sim renders in ~zero wall-clock, but a real camera can't
        # deliver a frame until the exposure ends. Stamp at the exposure midpoint (best time to
        # associate the averaged light with), and tell the loop not to *commit* the frame until the
        # exposure-end wall-clock -- so consumers see the same latency, and a long exposure floors fps.
        return frame, int(now_ns + 0.5 * exp * 1e9), now_ns + int(exp * 1e9)

    caps = [{'name': 'exposure', 'label': 'Exposure', 'kind': 'number', 'unit': 'ms', 'scale': 'log',
             'min': 0.01, 'max': 2000.0, 'value': _live['exp'] * 1000.0, 'live': True},
            {'name': 'gain', 'label': 'Gain', 'kind': 'number', 'unit': 'cB', 'scale': 'linear',
             'min': 0.0, 'max': 600.0, 'value': _live['gain_cb'], 'live': True}]

    def set_control(name, value):
        if name == 'exposure':
            _live['exp'] = max(1e-5, value / 1000.0)     # down to ~0.01 ms; the sim can render any exposure
            return _live['exp'] * 1000.0
        if name == 'gain':
            _live['gain_cb'] = max(0.0, value)
            sim.gain_mult = 10.0 ** (_live['gain_cb'] / 200.0)   # +60 cB (=6 dB) doubles the signal
            return _live['gain_cb']
        return None
    controls = {'source': 'sky', 'controls': caps, 'set': set_control}
    meta = {'bin': [args.bin, args.bin], 'roi': [0, 0, cfg.width, cfg.height]}
    return capture, cfg.width, cfg.height, ser_mod.ColorId.MONO, adc_bits, None, meta, controls


def _open_playback(args):
    """
    Replay an existing .ser as if it were a live camera, paced by its frame timestamps (x
    --playback-speed) and capped at --playback-fps (the only pacing when the recording has no
    timestamp sidecar -- otherwise it free-runs and outpaces the detector). Loops at the end.
    Lets the whole live pipeline (detect, gui, tracking) run on a recording -- the easy way to
    review a capture with detections overlaid.
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
    interval = 1.0 / args.playback_fps if args.playback_fps > 0 else 0.0
    cap_note = f" fps<={args.playback_fps:g}" if interval else ""
    print(f"[cam] playback {os.path.basename(args.playback_ser)} {h.image_width}x{h.image_height} "
          f"{n} frames x{args.playback_speed}{cap_note}", flush=True)

    st = {'i': 0, 'wall0': None, 't0': None, 'last': None}

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
        if interval:                              # frame-rate cap, on top of any timestamp pacing
            if st['last'] is not None:
                delay = st['last'] + interval - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            st['last'] = time.perf_counter()
        st['i'] = i + 1
        return frame

    return (capture, h.image_width, h.image_height, ser_mod.ColorId(h.color_id),
            h.pixel_depth_per_plane, None, meta, None)     # controls: none yet (loop/speed/file -> later)


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker camera capture")
    p.add_argument('--role', default='guide', help="camera role / file basename (e.g. guide, main)")
    p.add_argument('--out-dir', default=None, help="session dir to write into (default: a new sessions/<ts>)")
    p.add_argument('--source', default='synthetic', choices=['synthetic', 'zwo', 'sky', 'playback'])
    p.add_argument('--playback-ser', default=None, help="playback: the .ser file to replay")
    p.add_argument('--playback-speed', type=float, default=1.0, help="playback: speed multiplier")
    p.add_argument('--playback-loop', action='store_true', help="playback: loop instead of stopping at the end")
    p.add_argument('--playback-fps', type=float, default=10.0,
                   help="playback: frame-rate cap (0 = unlimited). The only pacing for a recording "
                        "with no timestamp sidecar, which would otherwise replay as fast as the "
                        "disk allows and swamp the detector")
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--bin', type=int, default=1,
                   help="NxN binning. sim/synthetic: --width/--height are already the binned size; this "
                        "just records bin=[N,N] in the frame metadata. zwo: sets hardware binning "
                        "(a color cam binned >1 reads out mono).")
    p.add_argument('--roi', default=None,
                   help="centered readout window 'x0,y0,w,h' in native sensor px, recorded in the frame "
                        "metadata (the render is already cropped to it via --width/--height).")
    p.add_argument('--fps', type=float, default=15.0)
    p.add_argument('--device', default='auto',
                   help="torch device for the sky-sim render: 'auto' (default) = cuda if present else "
                        "cpu, or force 'cpu' / 'cuda'. (zwo / synthetic / playback ignore it.)")
    p.add_argument('--frame-limit', type=int, default=-1,
                   help="frames for the current file before rolling over (-1 = unlimited)")
    p.add_argument('--shm-ser', action='store_true',
                   help="write non-important segments to a shared-memory section instead of disk "
                        "(only a 178-byte marker .ser lands on disk; readers redirect transparently). "
                        "Live pipelines only -- the section dies with its processes, so offline "
                        "workflows need real files. Recording ('important') always writes to disk.")
    p.add_argument('--shm-frames', type=int, default=128,
                   help="shm segments: frames per segment (committed RAM = frames x frame size; "
                        "rolls at this even if --frame-limit is longer)")
    p.add_argument('--file-limit', type=int, default=1,
                   help="how many (more) files to capture; exit when 0 (-1 = unlimited)")

    p.add_argument('--control-file', default=None,
                   help="JSONL of live setting updates to merge (or '-' for stdin)")
    p.add_argument('--exposure-us', type=int, default=2000, help="zwo only (manual exposure)")
    p.add_argument('--gain', type=int, default=190, help="zwo only (manual gain)")
    p.add_argument('--auto', action='store_true', help="zwo: enable auto-exposure + auto-gain")
    p.add_argument('--auto-max-exp-ms', type=int, default=200, help="auto: max exposure (ms)")
    p.add_argument('--auto-max-gain', type=int, default=400, help="auto: max gain")
    p.add_argument('--auto-target', type=int, default=100, help="auto: target brightness (0-255)")
    # sky simulator (--source sky): the camera only renders point sources it reads from the shared
    # sky_sim almanac. It has no notion of stars vs satellites, nor of epoch/site/TLE -- sky_sim
    # owns all propagation. These args are just this camera's optics + pose + exposure.
    p.add_argument('--sky-width', type=int, default=1920, help="sky: sensor width (px)")
    p.add_argument('--sky-height', type=int, default=1080, help="sky: sensor height (px)")
    p.add_argument('--sky-focal-mm', type=float, default=8.0, help="sky: lens focal length (mm); FoV = w*pitch/focal")
    p.add_argument('--sky-pixel-um', type=float, default=2.0, help="sky: sensor pixel pitch (um)")
    p.add_argument('--sky-aperture-mm', type=float, default=0.0,
                   help="sky: objective aperture (mm); >0 renders a physically-sized Airy-disc PSF")
    p.add_argument('--sky-central-obstruction', type=float, default=0.0,
                   help="sky: central obstruction as a LINEAR (diameter) ratio secondary/aperture; "
                        ">0 renders the obstructed (annular) diffraction PSF instead of a clear Airy")
    p.add_argument('--sky-spider-vanes', type=int, default=0,
                   help="sky: number of spider vanes (Newtonian secondary support); adds diffraction spikes")
    p.add_argument('--sky-vane-width-frac', type=float, default=0.0,
                   help="sky: spider-vane width as a fraction of the aperture diameter")
    p.add_argument('--sky-qe', type=float, default=0.0,
                   help="sky: sensor peak quantum efficiency (0..1); >0 enables the physical flux model "
                        "(aperture area x QE x throughput x mag) instead of a fixed flux")
    p.add_argument('--sky-full-well-e', type=float, default=0.0,
                   help="sky: sensor full-well capacity (electrons); the 12-bit ADC saturates here")
    p.add_argument('--sky-read-noise-e', type=float, default=2.0, help="sky: sensor read noise (electrons RMS)")
    p.add_argument('--sky-sky-mag', type=float, default=21.0,
                   help="sky: sky surface brightness (mag/arcsec^2), from the Bortle zone (darker = higher)")
    p.add_argument('--sky-psf-wavelength-nm', type=float, default=550.0,
                   help="sky: wavelength for the Airy disc (nm)")
    p.add_argument('--sky-psf-sigma-px', type=float, default=None,
                   help="sky: residual-blur Gaussian (defocus/aberration) on top of the Airy disc, px. "
                        "Default auto: 0 when the aperture is known (pure diffraction), else 1.3.")
    p.add_argument('--sky-seeing-r0-m', type=float, default=0.0,
                   help="sky: atmospheric Fried parameter r0 (m); >0 adds a seeing blur (FWHM ~0.98*lambda/r0)")
    p.add_argument('--sky-az-deg', type=float, default=None, help="sky: fallback encoder az for scripted (non-follow) runs")
    p.add_argument('--sky-alt-deg', type=float, default=None, help="sky: fallback encoder alt")
    p.add_argument('--sky-rate-az', type=float, default=0.0, help="sky: scripted az slew (deg/s) for streaks")
    p.add_argument('--sky-rate-alt', type=float, default=0.0, help="sky: scripted alt slew (deg/s)")
    p.add_argument('--sky-exposure-s', type=float, default=0.1, help="sky: simulated exposure (s)")
    p.add_argument('--sky-gain-cb', type=float, default=0.0,
                   help="sky: sensor gain in centibels (ZWO-style, 0.1 dB units; +60 cB doubles signal). "
                        "Amplifies electrons->ADU so you can brighten without a longer exposure (keeps fps)")
    p.add_argument('--sky-substeps', type=int, default=6, help="sky: substeps per exposure (streak smoothness)")
    p.add_argument('--sim-cam-noop', action='store_true',
                   help="sky: skip the actual frame simulation (one canned noise frame) but keep ALL the "
                        "pacing -- exposure, --fps, and the bandwidth model -- so the cam writes at full "
                        "commanded speed. For reproducing sustained-write disk-pressure issues")
    p.add_argument('--sim-cam-bandwidth-limit', type=float, default=400.0,
                   help="sky: model the camera's data link (MB/s; default ~USB3): a frame can't be "
                        "delivered faster than the link carries its bytes, so full-res frames cap at "
                        "~24 fps like the real hardware. 0 = unlimited")
    p.add_argument('--sky-almanac', default=None,
                   help="sky: shared source-direction almanac (JSONL) published by sky_sim")
    p.add_argument('--state-shm', default=None,
                   help="backend state slot (shm name) for --sky-follow-state pose")
    p.add_argument('--sky-follow-state', action='store_true',
                   help="sky: render from the backend's encoder estimate in <ts>_state.jsonl")
    p.add_argument('--sky-follow-mount', action='store_true',
                   help="sky: render from the sim mount's ground-truth trajectory in <ts>_sim_mount.jsonl "
                        "(piecewise-linear; preferred over --sky-follow-state for the sim mount)")
    p.add_argument('--camera-index', type=int, default=0, help="zwo camera index (fallback)")
    p.add_argument('--camera-url', default=None,
                   help="zwo camera by model: 'zwo:<model>[#k]' (see zwo_camera_urls); wins over --camera-index")
    p.add_argument('--camera-wb', action='store_true',
                   help="zwo: keep the camera's white balance (default: neutral WB for pristine raw)")
    p.add_argument('--bit-depth', type=int, default=16, choices=[8, 16],
                   help="capture container depth: 8 = RAW8 (half the USB bytes/frame -> higher fps; "
                        "sim renders an 8-bit ADC), 16 = full precision (RAW16 = 12-bit for the ASI; "
                        "sim renders its native 12-bit). Pixels stay LINEAR either way.")
    p.add_argument('--mono', action='store_true', help="store raw mosaic as MONO (no Bayer tag)")
    p.add_argument('--list-cameras', action='store_true', help="list ZWO cameras and exit")
    p.add_argument('--priority', default='normal', choices=['normal', 'above', 'high'],
                   help="scheduling priority for this capture process. A real camera free-runs: if we "
                        "call capture late because we were descheduled, the camera's buffer overflows "
                        "and IT drops frames. Raise this when the fps sits below the sensor's rated "
                        "rate with jittery frame intervals. ('realtime' is not offered -- it can starve "
                        "the OS.)")
    p.add_argument('--benchmark', action='store_true',
                   help="measure sustained framerate then exit: run the normal capture->framestream "
                        "loop (frames still written -- nobody reads them) with the --fps cap forced "
                        "off, for --benchmark-seconds after a warm-up, then print the delivered fps + "
                        "throughput for the current geometry/bit-depth. Compare against the frame-rate "
                        "table in the camera's manual. Add --shm-ser to take the disk out of the loop.")
    p.add_argument('--benchmark-seconds', type=float, default=10.0,
                   help="benchmark: timed window in seconds (default 10)")
    p.add_argument('--benchmark-warmup', type=float, default=1.0,
                   help="benchmark: warm-up seconds discarded before timing (the first frames after "
                        "start_video_capture can straggle while the USB pipeline fills)")
    args = p.parse_args(argv)

    set_priority(args.priority)          # before we open the camera and start free-running

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
    controls = None                                        # {'source','controls':[caps],'set':fn} or None
    if args.source == 'zwo':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta, controls = _open_zwo(
            args.camera_index, args.exposure_us, args.gain, force_mono=args.mono,
            auto=args.auto, auto_max_exp_ms=args.auto_max_exp_ms,
            auto_max_gain=args.auto_max_gain, auto_target=args.auto_target,
            neutral_wb=not args.camera_wb, bin=args.bin, camera_url=args.camera_url, roi=args.roi,
            capture_bits=args.bit_depth)
    elif args.source == 'sky':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta, controls = _open_sky(
            args, state_path=os.path.join(out_dir, session_mod.state_name(ts)),
            mount_path=os.path.join(out_dir, session_mod.sim_mount_name(ts)))
    elif args.source == 'playback':
        capture, width, height, color_id, pixel_depth, get_settings, frame_meta, controls = _open_playback(args)
    if frame_meta is None:                                  # synthetic: rendered at the binned size
        frame_meta = {'bin': [args.bin, args.bin], 'roi': [0, 0, width, height]}
    if args.roi:                                            # backend's centered readout window (sensor px)
        try:
            frame_meta['roi'] = [int(v) for v in args.roi.split(',')]
        except ValueError:
            print(f"[cam:{args.role}] malformed --roi {args.roi!r}; meta keeps full frame", flush=True)

    # Publish the camera's live controls (name/kind/range/value) so the backend + GUI can render them.
    caps_path = os.path.join(out_dir, f'caps_{args.role}.json')
    with open(caps_path, 'w', encoding='utf-8') as _cf:
        json.dump({'source': args.source, 'controls': (controls['controls'] if controls else [])}, _cf)
    applied = {c['name']: c.get('value') for c in controls['controls']} if controls else {}   # live values in effect

    control = control_mod.ControlReader(args.control_file) if args.control_file else None
    cfg = {'frame_limit': args.frame_limit, 'file_limit': args.file_limit,
           'fps': args.fps}

    print(f"[cam:{args.role}] {args.source} {width}x{height} {color_id.name} {pixel_depth}-bit "
          f"frame_limit={cfg['frame_limit']} file_limit={cfg['file_limit']} "
          f"control={args.control_file} -> {out_dir}", flush=True)

    stream = framestream.FrameStream(out_dir, args.role)
    # ONE ring for the whole run (geometry is fixed per cam process; a resolution/bin/ROI
    # change relaunches the cam). --shm-frames sizes the ring = the stream's entire
    # write-behind budget; there are no rolls, no per-segment anything, no VM syscalls after
    # this line. Live settings (exposure/gain) just apply -- they no longer split the stream.
    stream.configure(width, height, color_id=color_id, pixel_depth=pixel_depth,
                     shm=bool(args.shm_ser), frames=args.shm_frames if args.shm_ser else 64,
                     meta=frame_meta)         # bin + roi (sensor->frame mapping)
    start = time.perf_counter()
    total = 0
    last_status = start
    last_status_n = 0
    stop = False
    # Benchmark: run the real capture->write->commit loop (frames land in the ring/disk, unread)
    # with the fps cap forced off, and time a fixed window after a warm-up. bench['n0']/['t0'] mark
    # the window start; ['end'] the wall-clock to stop. The frame count over that window is the fps.
    bench = None
    if args.benchmark:
        cfg['fps'] = 0.0                                  # uncapped: measure the source's own ceiling
        bench = {'warm_until': start + max(0.0, args.benchmark_warmup),
                 't0': None, 'n0': None, 'end': None}
        print(f"[cam:{args.role}] BENCHMARK {args.source} {width}x{height} bin={args.bin} "
              f"{'RAW8' if int(args.bit_depth) == 8 else 'RAW16'} {color_id.name} {pixel_depth}-bit"
              + (f"  [{get_settings()}]" if get_settings else "")
              + f"  (warmup {args.benchmark_warmup:g}s, window {args.benchmark_seconds:g}s, "
              f"{'shm' if args.shm_ser else 'disk'})", flush=True)
    try:
        while True:
            if control is not None:
                for cmd in control.drain():
                    if cmd.get('stop'):
                        stop = True
                    for k in ('frame_limit', 'fps'):
                        if k in cmd:
                            cfg[k] = cmd[k]
                    if 'controls' in cmd and controls is not None:   # live camera controls
                        for _n, _v in cmd['controls'].items():
                            got = controls['set'](_n, _v)
                            applied[_n] = got if got is not None else _v
            if stop:                                  # {stop} or shutdown -> finalize + exit
                break
            if cfg['frame_limit'] != -1 and total >= cfg['frame_limit']:
                break                                 # tests/tools: a total-frame budget

            loop_start = time.perf_counter()
            cap_t_ns = None                            # a source may supply the true frame time
            avail_ns = None                            # ...and a wall-clock before which not to commit
            if capture is not None:
                frame = capture()
                if isinstance(frame, tuple):           # (frame, t_mono_ns[, available_at_ns])
                    frame, cap_t_ns, *rest = frame     # e.g. sim: exposure midpoint + exposure-end
                    avail_ns = rest[0] if rest else None
                if frame is None:
                    if args.source == 'playback' and not args.playback_loop:
                        print(f"[cam:{args.role}] playback complete", flush=True)
                        break
                    print(f"[cam:{args.role}] capture timeout, skipping", flush=True)
                    continue
            else:
                frame = make_synthetic_frame(width, height, loop_start - start)

            stream.write_pixels(frame)                # pixels staged (claim precedes the touch)
            if avail_ns is not None:                   # hold the commit until the exposure really ends
                wait = (avail_ns - session_mod.mono_ns()) * 1e-9   # avail_ns is on the mono clock
                if wait > 0:
                    time.sleep(wait)
            stream.commit(                             # the commit: one shm store
                t_mono_ns=cap_t_ns if cap_t_ns is not None else session_mod.mono_ns(),
                t_utc_ns=time.time_ns(),
                src_index=total)
            total += 1

            if bench is not None:                          # warm up, then time a fixed window
                now = time.perf_counter()
                if bench['t0'] is None:
                    if now >= bench['warm_until']:
                        bench['t0'], bench['n0'] = now, total
                        bench['end'] = now + max(0.1, args.benchmark_seconds)
                elif now >= bench['end']:
                    elapsed = now - bench['t0']
                    frames = total - bench['n0']
                    fps = frames / elapsed if elapsed > 0 else 0.0
                    bpp = 1 if int(args.bit_depth) == 8 else 2
                    print(f"[cam:{args.role}] BENCHMARK result: {frames} frames in {elapsed:.2f}s "
                          f"= {fps:.1f} fps ({fps * width * height / 1e6:.1f} Mpx/s, "
                          f"{fps * width * height * bpp / 1e6:.0f} MB/s over USB)", flush=True)
                    break

            period = 1.0 / cfg['fps'] if cfg['fps'] > 0 else 0.0
            if period:
                sleep = period - (time.perf_counter() - loop_start)
                if sleep > 0:
                    time.sleep(sleep)
    except KeyboardInterrupt:
        print(f"[cam:{args.role}] interrupted", flush=True)
    finally:
        stream.close()                                # finalize the open segment + release retained shm
        if control is not None:
            control.close()
        print(f"[cam:{args.role}] done, {total} frames total", flush=True)


if __name__ == '__main__':
    main()
