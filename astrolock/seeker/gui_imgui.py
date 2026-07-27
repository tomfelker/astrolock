"""
astrolock_seeker_gui: Dear ImGui (imgui-bundle) + moderngl viewer.

A fixed, self-tiling layout (not free-floating windows): a large 'big' pane top-left showing a
selected stream, a docked settings/telemetry panel on the right (drag-resizable), and a strip of
PIP panes along the bottom. Roles (guide, main) are decoupled from slots. Each pane letterboxes
its stream (preserve aspect, centred) at a power-of-two scale; zoom in further (crops, with edge
indicators). Everything reflows on viewport resize.

What this buys over the retired Dear PyGui front end it replaced:
  * The raw u16/u8 SER mosaic goes straight to the GPU (double-buffered R16UI textures); a
    fragment shader does the Bayer 4-plane stack + WB + gamma into an RGBA8 offscreen target
    that ImGui samples at any zoom. The ~30ms/frame CPU tonemap + 32MB float upload is gone.
  * The event loop blocks in glfw.wait_events_timeout(); a watcher thread posts
    glfw.post_empty_event() when a sidecar the GUI draws from grows. Portable (no Win32
    PostThreadMessage), and idle cost is a ~4Hz heartbeat frame.
  * Immediate mode end to end: no widget-value reconciliation -- panel state lives in plain
    dicts the panel reads/writes every frame.

    python -m astrolock.seeker.gui_imgui --session sessions/<ts>

Requires `imgui-bundle`, `moderngl`, `glfw` (pip install imgui-bundle moderngl glfw).
"""

import argparse
import ctypes
import datetime
import glob
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from astrolock.seeker import bayer, control, ser
from astrolock.seeker import geometry as geo
from astrolock.seeker import optics as optics_db
from astrolock.seeker import settings as settings_store
from astrolock.seeker import framestream
from astrolock.seeker import session as session_mod
from astrolock.seeker import skysim
from astrolock.seeker.almanac import SkyAlmanac
from astrolock.seeker.follower import FrameRef, SerFollower
from astrolock.seeker.sidecar import JsonlTailer


def _newest(session_dir, suffix):
    matches = sorted(glob.glob(os.path.join(session_dir, '*' + suffix)))
    return matches[-1] if matches else None


class _Meter:
    """Rolling events-per-second: hit() as events happen, sample(now) each loop; `rate` refreshes ~1/s."""
    def __init__(self):
        self._n = 0
        self._t = None
        self.rate = 0.0
        self.spin = 0        # advances once per real update (n>0) -> a per-meter spinner that FREEZES when

    def hit(self, n=1):      # the thing stops producing, even while `rate` still reads its last (stale) value
        self._n += n
        if n:
            self.spin += 1

    def sample(self, now):
        if self._t is None:
            self._t = now
        elif now - self._t >= 1.0:
            self.rate = self._n / (now - self._t)
            self._n = 0
            self._t = now


def _color_name(cid):
    try:
        return ser.ColorId(int(cid)).name
    except ValueError:
        return str(cid)


ROLES = ('guide', 'main')      # the two fixed roles: a wide guide cam that points a narrow main cam.
                               # Either may be absent/unconfigured; we don't add roles dynamically.

PANEL_W = 380                    # default right-panel width (logical px, pre-DPI)
PANEL_MIN_W = 220
ZOOM_MULTS = (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)   # multiplier over the auto power-of-two fit (1 = fit-to-pane;
                                            # <1 zooms OUT -- a tiny focus crop auto-fits at ~16x, where
                                            # '-' used to be pinned at the floor and felt dead)
# Detector choices for the Detection tab (must match backend/detect.py argparse choices).
_DETECTORS = ['bandpass', 'doh', 'surprise', 'extended', 'circmean']
_TRACK_DETECTORS = ['peak', 'matched']
MAXPIP = 4                       # pool of PIP panes along the bottom; each shows a stream not in the big pane


def _default_settings(key=None):
    # The sky overlay defaults ON for the guide (wide FoV -- it's how you find the pass) and
    # OFF for the main (narrow FoV, already on target; keeps its pane clean and cheap).
    main = (key == 'main')
    return {'zoom': 1, 'reticles': True, 'histogram': False, 'wait_for_detector': True,
            'show_stars': not main, 'show_target_names': not main, 'show_star_names': False}


def _fmt_time_delta(delta_ns):
    """'13d 20:07:03.2 ahead of system time' -- the backend-time vs system-clock offset. Never
    exactly zero: mono_ns is the perf counter, which drifts against (and ignores changes to)
    the system clock."""
    s = abs(delta_ns) * 1e-9
    d, s = divmod(s, 86400.0)
    h, s = divmod(s, 3600.0)
    m, s = divmod(s, 60.0)
    return ((f"{int(d)}d " if d else "") + f"{int(h):02d}:{int(m):02d}:{s:04.1f} "
            + ('ahead of' if delta_ns >= 0 else 'behind') + " system time")


def _floor_pow2(x):
    """Largest power of two <= x (x may be < 1), clamped to a sane display range."""
    if x <= 0:
        return 1.0
    return min(16.0, max(1.0 / 16, 2.0 ** math.floor(math.log2(x))))


def _open_folder(path):
    """Open a folder in the OS file browser (Explorer / Finder / xdg-open)."""
    try:
        os.makedirs(path, exist_ok=True)
        if sys.platform == 'win32':
            os.startfile(path)                         # noqa: only exists on Windows
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception as e:
        print(f"[gui] could not open {path}: {e}", flush=True)


# --- 2D slew pad -------------------------------------------------------------------------
SLEW_MAX = 4.0                     # deg/s at the pad edge
SLEW_RMIN = 0.02                   # deg/s at the dead-zone edge (the log-scale floor)
SLEW_DEAD = 0.07                   # centre dead-zone as a fraction of the half-size (= zero rate)
SLEW_GRID = (0.1, 0.5, 1.0, 2.0)   # deg/s gridlines


def _u_to_rate(u):
    """Pad position u in [-1,1] -> axis rate (deg/s): log-scaled, with a centre dead-zone at zero."""
    au = min(1.0, abs(u))
    if au <= SLEW_DEAD:
        return 0.0
    frac = (au - SLEW_DEAD) / (1.0 - SLEW_DEAD)
    return math.copysign(SLEW_RMIN * (SLEW_MAX / SLEW_RMIN) ** frac, u)


def _rate_to_u(r):
    """Axis rate (deg/s) -> pad position u in [-1,1] (inverse of _u_to_rate; for gridlines + readout)."""
    ar = abs(r)
    if ar <= SLEW_RMIN:
        return 0.0
    frac = min(1.0, math.log(ar / SLEW_RMIN) / math.log(SLEW_MAX / SLEW_RMIN))
    return math.copysign(SLEW_DEAD + (1.0 - SLEW_DEAD) * frac, r)


def _waker(glfw_mod, wake, stop):
    """Poke the (blocked-in-wait_events) GUI thread awake ONLY when something it draws changed.
    Three signals, all self-gating -- a parked, not-tracking, not-capturing rig emits none:
      * shm PROBES -- frame counts read straight out of the followed segments' headers (pure
        memory reads, no syscalls: v3 frames commit in RAM);
      * files -- head files (segment/ended events) + detection sidecars that grew or appeared;
      * state file -- only a *meaningful* field change wakes us. The backend rewrites state at
        ~20 Hz even when parked, so we diff with the volatile fields stripped.
    glfw.post_empty_event() is documented thread-safe, and portable."""
    sizes = {}
    last_counts = None
    last_state = None
    while not stop.is_set():
        dirty = False
        counts = []
        for seg in (wake.get('probes') or ()):
            try:
                counts.append(seg.committed())
            except (AttributeError, ValueError, IndexError):   # closed/repolled mid-probe: benign race
                counts.append(-1)
        counts = tuple(counts)
        if counts != last_counts:
            last_counts = counts
            dirty = True
        paths = wake.get('paths') or ()
        for pth in paths:                         # unseen path (sizes.get -> None) or a grown one = new data
            try:
                sz = os.path.getsize(pth)
            except OSError:
                continue
            if sizes.get(pth) != sz:
                sizes[pth] = sz
                dirty = True
        for stale in [k for k in sizes if k not in paths]:
            del sizes[stale]                      # forget rolled-over segments so the dict stays bounded
        slot = wake.get('state_slot')             # backend state: latest-wins shm slot (memory read)
        if slot is not None:
            try:
                got = slot.read()
            except (AttributeError, ValueError):  # closed mid-probe: benign race
                got = None
            if got:
                rec = dict(got[1])
                for k in ('t_mono_ns', 't_utc', 'enc_t_mono_ns'):
                    rec.pop(k, None)              # volatile fields the GUI doesn't draw
                if rec != last_state:             # only a MEANINGFUL change wakes us (else the
                    last_state = rec              # ~20 Hz heartbeat would never let the GUI idle)
                    dirty = True
        if dirty:
            glfw_mod.post_empty_event()
        stop.wait(0.02)                           # cheap: a few os.stat + one small read per tick


# --- GPU display path ---------------------------------------------------------------------
# One fullscreen-triangle pass per NEW camera frame: raw integer mosaic in, tonemapped RGBA8 out.
# ImGui then samples the RGBA8 target (bilinear) at whatever pane size/zoom -- scaling is free.
_VS = """#version 330 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

# Bayer: output pixel (x,y) <- the 2x2 cell at (2x,2y): R and B from their sites, G = mean of the two
# G sites -- the same lossless 4-plane stack as bayer.py, on the GPU.
# WB gains are display-only (data stays pristine); full container range maps to [0,1], no auto-stretch.
_FS = """#version 330 core
uniform usampler2D mosaic;
uniform int mono;                       // 1 = pass-through gray, 0 = Bayer stack
uniform ivec2 off_r;                    // R / G / G / B site offsets within the 2x2 cell
uniform ivec2 off_g0;
uniform ivec2 off_g1;
uniform ivec2 off_b;
uniform vec2 wb;                        // display-only R,B gains
uniform float inv_white;                // 1 / container max (255 or 65535)
out vec4 frag;
// The ONLY nonlinearity in the whole pipeline: capture + histogram stay linear; here at the very
// last step we encode linear light to sRGB for the display (IEC 61966-2-1: a linear toe below
// 0.0031308, a ~2.4 gamma above). No adjustable gamma -- the data is neutral end to end.
vec3 lin_to_srgb(vec3 c) {
    vec3 lo = c * 12.92;
    vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;
    return mix(lo, hi, step(vec3(0.0031308), c));
}
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 rgb;
    if (mono == 1) {
        rgb = vec3(float(texelFetch(mosaic, p, 0).r));
    } else {
        ivec2 c = p * 2;
        float r = float(texelFetch(mosaic, c + off_r, 0).r) * wb.x;
        float g = 0.5 * (float(texelFetch(mosaic, c + off_g0, 0).r)
                       + float(texelFetch(mosaic, c + off_g1, 0).r));
        float b = float(texelFetch(mosaic, c + off_b, 0).r) * wb.y;
        rgb = vec3(r, g, b);
    }
    rgb = clamp(rgb * inv_white, 0.0, 1.0);
    frag = vec4(lin_to_srgb(rgb), 1.0);
}
"""


# Magnified-view blit: imgui 1.92's GL3 backend binds its OWN (bilinear) GL sampler object
# while rendering, and a bound sampler OVERRIDES texture-level filters -- so the color
# target's (LINEAR, NEAREST) filter no longer gives crisp pixels when a pane magnifies.
# Instead we nearest-resample the pane's visible crop into a pane-sized target with our own
# sampler and let imgui draw THAT at 1:1, where its bilinear sampler is a no-op.
_VS_BLIT = """#version 330 core
uniform vec4 src;            // visible crop of the source texture: (u0, v0, du, dv)
out vec2 uv;
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
    uv = src.xy + p * src.zw;
}
"""

_FS_BLIT = """#version 330 core
uniform sampler2D img;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(img, uv); }
"""


class _StreamGL:
    """GPU resources for one stream: a double-buffered raw integer texture (so a write never stalls
    on a frame the GPU is still reading) + a matching pair of RGBA8 render targets. upload() runs the
    debayer/WB/gamma pass and leaves `tex_id` pointing at the freshly rendered target."""

    def __init__(self, ctx, prog, vao):
        self.ctx, self.prog, self.vao = ctx, prog, vao
        self.raw = self.color = self.fbo = None
        self.key = None                          # (w, h, dtype, color_id) the textures were built for
        self.flip = 0
        self.tex_id = None                       # GL texture name of the latest rendered RGBA8 target

    def _rebuild(self, w, h, dtype, color_id):
        self.release()
        gl_dtype = 'u2' if dtype.itemsize == 2 else 'u1'
        self.raw = [self.ctx.texture((w, h), 1, dtype=gl_dtype, alignment=1) for _ in range(2)]
        for t in self.raw:
            t.filter = (self.ctx.NEAREST, self.ctx.NEAREST)   # integer textures must not linear-sample
        nb = 2 if bayer.is_bayer(color_id) else 1
        ow, oh = w // nb, h // nb
        self.color = [self.ctx.texture((ow, oh), 4, dtype='f1') for _ in range(2)]
        for t in self.color:
            # Smooth when zoomed OUT (minify), but crisp NEAREST pixels when zoomed IN (magnify) -- so a
            # deep zoom shows the actual sensor pixels instead of a bilinear smear.
            t.filter = (self.ctx.LINEAR, self.ctx.NEAREST)
        self.fbo = [self.ctx.framebuffer(color_attachments=[t]) for t in self.color]
        self.key = (w, h, dtype, int(color_id))
        self.out_size = (ow, oh)

    def upload(self, frame, color_id, wb):
        """Raw mosaic/mono frame (numpy u16/u8) -> the tonemapped RGBA8 target. Returns (w, h) of the
        display texture (half-res for Bayer: the 4-plane stack). Linear in, sRGB out (no gamma knob)."""
        h, w = frame.shape[0], frame.shape[1]
        if self.key != (w, h, frame.dtype, int(color_id)):
            self._rebuild(w, h, frame.dtype, color_id)
        self.flip ^= 1
        i = self.flip
        self.tex_obj = self.color[i]             # moderngl handle for the nearest-blit pass
        self.raw[i].write(np.ascontiguousarray(frame))
        prog = self.prog
        white = int(np.iinfo(frame.dtype).max)
        prog['mosaic'].value = 0
        prog['inv_white'].value = 1.0 / white
        if bayer.is_bayer(color_id):
            ri, (g0, g1), bi = bayer.rgb_plane_indices(color_id)
            prog['mono'].value = 0
            for name, idx in (('off_r', ri), ('off_g0', g0), ('off_g1', g1), ('off_b', bi)):
                prog[name].value = (idx & 1, idx >> 1)        # plane index -> (x, y) within the 2x2 cell
            prog['wb'].value = (wb[0], wb[1])
        else:
            prog['mono'].value = 1
        self.fbo[i].use()
        self.raw[i].use(0)
        self.vao.render(mode=self.ctx.TRIANGLES, vertices=3)
        self.tex_id = self.color[i].glo
        return self.out_size

    def release(self):
        for group in (self.fbo, self.color, self.raw):
            for t in (group or ()):
                t.release()
        self.raw = self.color = self.fbo = None


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker GUI viewer (imgui-bundle)")
    p.add_argument('--session', required=True, help="session directory to view")
    p.add_argument('--roles', default=None,
                   help="playback override: view a subset of an old session (default: guide,main)")
    p.add_argument('--wb-r', type=float, default=1.24, help="display-only WB gain for red")
    p.add_argument('--wb-b', type=float, default=1.98, help="display-only WB gain for blue")
    p.add_argument('--slew-rate', type=float, default=3.0, help="(unused: the slew pad is log-scaled)")
    p.add_argument('--ui-scale', type=float, default=0.0,
                   help="UI/DPI scale factor (0 = auto-detect from the OS; e.g. 1.5 for a 150%% display)")
    p.add_argument('--device', default='auto',
                   help="(accepted for CLI compatibility; this GUI tonemaps on the GPU via GL, no torch)")
    p.add_argument('--priority', default='below', choices=['below', 'normal'],
                   help="GUI process priority: 'below' yields to the capture/track pipeline under "
                        "load (tier 4 of the backend's --sched ladder); 'normal' opts out")
    args = p.parse_args(argv)
    wb = (args.wb_r, args.wb_b)

    if args.priority == 'below':                          # frames matter more than widgets
        try:
            import psutil
            from astrolock.seeker.cam import prio_value
            psutil.Process().nice(prio_value('below'))
        except Exception as e:
            print(f"[gui] could not lower priority: {e}", flush=True)

    import glfw
    import moderngl
    from imgui_bundle import imgui
    from imgui_bundle import portable_file_dialogs as pfd

    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)          # harmless on Windows, needed on mac
    glfw.window_hint(glfw.SCALE_TO_MONITOR, glfw.TRUE)               # size the window by the monitor DPI
    glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)                      # field use: all the pixels, always
    window = glfw.create_window(1400, 900, "AstroLock Seeker", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw.create_window() failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ui_scale = args.ui_scale
    if ui_scale <= 0:
        ui_scale = glfw.get_window_content_scale(window)[0] or 1.0

    def S(v):
        """Scale a pixel dimension by the display's DPI factor, rounded to an int."""
        return int(round(v * ui_scale))

    ctx = moderngl.create_context()
    prog = ctx.program(vertex_shader=_VS, fragment_shader=_FS)
    vao = ctx.vertex_array(prog, [])             # fullscreen triangle from gl_VertexID; no buffers
    prog_blit = ctx.program(vertex_shader=_VS_BLIT, fragment_shader=_FS_BLIT)
    vao_blit = ctx.vertex_array(prog_blit, [])
    samp_nearest = ctx.sampler(filter=(moderngl.NEAREST, moderngl.NEAREST))
    blit_slots = {}          # slot name -> {'size', 'tex', 'fbo'}: per-pane magnified-view target

    imgui.create_context()
    io = imgui.get_io()
    io.set_ini_filename("")                      # a fixed tiled layout -- nothing worth persisting to imgui.ini
    imgui.style_colors_dark()
    imgui.get_style().font_scale_dpi = ui_scale  # 1.92 dynamic fonts: re-rasterize crisply at the DPI
    ui_font = None
    for fp in ("C:/Windows/Fonts/segoeui.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(fp):
            ui_font = io.fonts.add_font_from_file_ttf(fp, 16.0)
            break
    mono_font = None                             # monospace, so the aligned readouts actually line up
    for fp in ("C:/Windows/Fonts/consola.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        if os.path.exists(fp):
            mono_font = io.fonts.add_font_from_file_ttf(fp, 13.0)
            break
    imgui.backends.glfw_init_for_opengl(ctypes.cast(window, ctypes.c_void_p).value, True)
    imgui.backends.opengl3_init("#version 330")

    ENTER = imgui.InputTextFlags_.enter_returns_true
    OPEN = imgui.TreeNodeFlags_.default_open
    PANE_WF = imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
    ROOT_WF = (imgui.WindowFlags_.no_title_bar | imgui.WindowFlags_.no_resize
               | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_collapse
               | imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse
               | imgui.WindowFlags_.no_bring_to_front_on_focus | imgui.WindowFlags_.no_nav_focus
               | imgui.WindowFlags_.no_saved_settings)

    def C(t):
        """(r,g,b[,a]) 0-255 tuple -> packed ImU32 draw-list colour."""
        return imgui.IM_COL32(int(t[0]), int(t[1]), int(t[2]), int(t[3]) if len(t) > 3 else 255)

    def C4(r, g, b, a=255):
        """0-255 -> ImVec4 float colour (for text_colored / style pushes)."""
        return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)

    def _tip(text):
        """Hover tooltip on the just-submitted item (wrapped, small delay)."""
        if imgui.is_item_hovered(imgui.HoveredFlags_.for_tooltip):
            imgui.begin_tooltip()
            imgui.push_text_wrap_pos(S(280))
            imgui.text_unformatted(text)
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()

    # The two fixed roles (guide, main). --roles stays only as an optional playback override.
    roles = ([r.strip() for r in args.roles.split(',') if r.strip()] if args.roles else list(ROLES))
    followers = {}
    cams = {}                 # stream -> live camera data (GL textures + frames + detections); lazily created
    perf = {'gui': _Meter(), 'mount': _Meter(), 'focus': _Meter(), 'frame': 0,
            'cam': {r: _Meter() for r in roles},                  # per-role frames *produced* / s
            'det': {r: _Meter() for r in roles},                  # per-role detector frames processed / s
            'skip': {r: _Meter() for r in roles},                 # per-role frames the detector SKIPPED / s
            'idx': {},                                            # role -> last committed frame index seen
            'ms': {}}                                             # name -> EMA of a timed section (ms)

    def _prof(name, ms):                                          # exponential moving average of a timing (ms)
        m = perf['ms']
        m[name] = ms if name not in m else m[name] * 0.9 + ms * 0.1

    view_settings = {}        # stream -> display prefs {zoom, reticles, histogram}; persists across cams
    cam_ctrl_val = {}         # (role, control name) -> current value; the GUI owns it once a control is shown
    layout = {'panel_open': True, 'pip_open': True, 'pip_debug': False, 'panel_w': S(PANEL_W),
              'pip_h': S(200), 'big_role': roles[0] if roles else ROLES[0], 'big_stream': None,
              'pip_map': {}, 'pip_slots': []}
    boresight_ui = {'x': 0.0, 'y': 0.0, 'roll': 0.0, 'step': 0.1}   # mrad; the Boresight panel's state
    align_ui = {'step': 0.1}                              # deg; manual mount alignment (Alignment panel)
    # Checklist attestations: boxes the user must tick by hand each run (the software can't
    # verify them). Deliberately NOT persisted -- re-confirm every session.
    checklist_ui = {'guide_optics': False, 'main_optics': False, 'focus': False,
                    'boresight': False}
    site_ui = {'lat': 0.0, 'lon': 0.0, 'elev': 0.0}       # observer location (Location panel)
    # Sky/navigation overlay state: the almanac reader attaches once its feed file exists.
    nav = {'alm': None}

    # Pose at the DISPLAYED frame's capture stamp, for the sky overlay -- the SAME solver the
    # tracker uses to place old detections on the sky (interpolate within the history,
    # rate-extrapolate beyond it; clamping instead caused one-frame glyph jumps while slewing).
    pose_hist = geo.PoseHistory(maxlen=256)

    def _poll_navigation():
        """Attach to the session's navigation feed once it exists, then tail it."""
        if nav['alm'] is None:
            hits = glob.glob(os.path.join(args.session, '*_navigation.jsonl'))
            if not hits:
                return
            nav['alm'] = SkyAlmanac(max(hits, key=os.path.getmtime))
        nav['alm'].update()
    track_ui = {'smoothing': 1.0, 'pref': 'auto', 'auto_switch': True,   # Tracking panel state; backend defaults
                'delay': 0.0,                                            # post-lock mount-hold (s)
                'latency': 0.0,                                          # assumed command latency (s)
                'rate_scale': 1.0,                                       # commanded-rate trim multiplier
                'model': 'ema', 'alt_km': 250.0,                         # target model (next lock)
                'feedforward': True,                                     # intercept servo drives
                'position_smoothing': 0.0,                               # model anchor blend constant (s)
                'min_horizon': 1.0, 'max_horizon': 8.0,                  # intercept time bounds (s)
                'pid_on': True,                                          # PID trim on the measured error
                'pid_kp': 0.0, 'pid_ki': 0.5, 'pid_kd': 0.0,             # gains (sky-vector space)
                'pid_integral_limit': 0.02}                              # windup limit (deg/s)
    sim_ui = {'r0': 0.0, 'bortle': 4}                     # Simulation panel: seeing r0 (m) + Bortle sky class
    focus_ui = {'role': roles[-1] if roles else 'main', 'want': False, 'fo': None,
                't0': None, 'com_mult': 10.0,             # skew-arrow exaggeration on the star view
                'shape_gain': 5.0,                        # astigmatism-ellipse ellipticity exaggeration
                'screw_phase': 0.0,                       # deg: rotation of the 3 collimation screws in the image
                'rad_per_turn': 100e-6,                   # empirical: skew radians removed per screw turn
                'invert_x': False, 'invert_y': False,     # flip screw-turn direction per axis (image parity)
                'alpha': 0.05,                            # star-crop EMA smoothing (relaunch-tier)
                'skew_rad': (0.0, 0.0),                   # latest skew (radians) for the screw dial
                'shape_ema': None,                        # latest (e1, e2, skew_x, skew_y) of the EMA image
                'shape_instant': None,                    # same for the instantaneous crop
                'series': {k: [] for k in ('t', 'peak', 'peakf', 'hfd', 'strehl')}}
    sweep_ui = {'start': 0.0, 'end': 9.0, 'step': 1.0, 'frames': 40,  # focus sweep (human actuator)
                'role': None, 'fo': None, 'state': None,              # prompt/result stream tail
                'confirmed': None,                                    # last position we OK'd
                'writer': None, 'writer_role': None}                  # our position-report stream
    FOCUS_MAX = 600                                       # rolling window of metric points kept in the graph

    # Immediate-mode UI state: text-input buffers + selections the panel reads/writes each frame.
    ui = {'txt': {'bore_x': '0', 'bore_y': '0', 'bore_roll': '0',
                  'track_delay': f"{track_ui['delay']:g}",
                  'sim_r0': f"{sim_ui['r0']:g}", 'track_smooth': f"{track_ui['smoothing']:g}",
                  'track_latency': f"{track_ui['latency']:g}",
                  'track_rate_scale': f"{track_ui['rate_scale']:g}",
                  'track_alt': f"{track_ui['alt_km']:g}",
                  'track_position_smoothing': f"{track_ui['position_smoothing']:g}",
                  'track_min_horizon': f"{track_ui['min_horizon']:g}",
                  'track_max_horizon': f"{track_ui['max_horizon']:g}",
                  'track_pid_kp': f"{track_ui['pid_kp']:g}",
                  'track_pid_ki': f"{track_ui['pid_ki']:g}",
                  'track_pid_kd': f"{track_ui['pid_kd']:g}",
                  'track_pid_integral_limit': f"{track_ui['pid_integral_limit']:g}",
                  'focus_alpha': f"{focus_ui['alpha']:g}", 'focus_com_mult': f"{focus_ui['com_mult']:g}",
                  'focus_shape_gain': f"{focus_ui['shape_gain']:g}",
                  'sweep_start': f"{sweep_ui['start']:g}", 'sweep_end': f"{sweep_ui['end']:g}",
                  'sweep_step': f"{sweep_ui['step']:g}",
                  'settings_name': ''},
          'src': {},                                      # role -> unified source dropdown value
          'pb_loop': {r: True for r in roles}, 'pb_dlg': None,     # (role, pfd.open_file) in flight
          'mount_sel': 'sim',
          'opt': {r: {'sensor': '', 'optic': '', 'reducer': '(none)'} for r in roles},
          'settings_sel': '', 'settings_items': settings_store.list_settings()}

    ctrl = {'client': None, 'tailer': None, 'state': None, 'last_rate': None}

    def _send(obj):
        if ctrl['client'] is not None:
            ctrl['client'].send(obj)

    def _shutdown():
        """Tell the backend we're closing (it stops as soon as it drains this), then drop the process
        immediately -- os._exit skips interpreter/atexit teardown that could otherwise keep us (and so
        the backend) alive after the window closes."""
        try:
            _send({'type': 'shutdown'})
        except Exception:
            pass
        try:
            if ctrl['client'] is not None:
                ctrl['client'].close()
        except Exception:
            pass
        sys.stdout.flush()
        os._exit(0)

    # ---- slots (which stream shows where) --------------------------------------------------
    def _slot_stream(name):
        """The stream a slot DISPLAYS. The big pane shows `big_stream` when set (the focus star, or a
        stream promoted with ^), else its role; each PIP shows one entry of the live pip map."""
        if name == 'big':
            return layout.get('big_stream') or layout['big_role']
        return (layout.get('pip_map') or {}).get(name) or layout['big_role']

    def _slot_role(name):
        """The underlying CAMERA role a slot acts on (target-pick, zoom key): the shown stream with any
        _focus/_debug suffix stripped."""
        s = _slot_stream(name)
        for suf in ('_focus', '_debug'):
            if s.endswith(suf):
                return s[:-len(suf)]
        return s

    def _promote(stream):
        """'^' on a PIP: make its stream the big pane. A plain camera role becomes a normal big-role view
        (so Swap / auto-switch / persistence keep working); a derived stream (focus/debug) rides big_stream."""
        if stream in roles:
            layout['big_role'], layout['big_stream'] = stream, None
        else:
            layout['big_stream'] = stream

    def _toggle_dbg():
        layout['pip_debug'] = not layout['pip_debug']
        # The detectors only WRITE the <role>_debug stream when launched with --debug-ser --
        # ask the backend to relaunch them with it (this used to silently show an empty pane
        # unless the backend happened to start with --debug-detect-ser).
        _send({'type': 'set_debug_ser', 'on': layout['pip_debug']})
        if layout['pip_debug']:
            layout['pip_open'] = True                    # no point showing the debug surface with the pip hidden

    def _pip_streams():
        """Ordered streams to show as PIPs: every connected camera, plus the focus star (if running) and
        the big pane's detector debug surface (if Dbg on), MINUS whatever the big pane already shows."""
        cap = (ctrl['state'] or {}).get('capturing') or {}
        streams = [r for r in roles if cap.get(r)]
        if focus_ui['want']:
            fs = focus_ui['role'] + '_focus'
            if fs not in streams:
                streams.append(fs)
        if layout.get('pip_debug'):
            dbg = layout['big_role'] + '_debug'
            if dbg not in streams:
                streams.append(dbg)
        big = _slot_stream('big')
        return [s for s in streams if s != big][:MAXPIP]

    def _active_slots():
        return ['big'] + ((layout.get('pip_slots') or []) if layout['pip_open'] else [])

    def _zoom_step(stream, delta):
        s = view_settings.setdefault(stream, _default_settings(stream))
        i = ZOOM_MULTS.index(s['zoom']) if s['zoom'] in ZOOM_MULTS else ZOOM_MULTS.index(1)
        s['zoom'] = ZOOM_MULTS[max(0, min(len(ZOOM_MULTS) - 1, i + delta))]

    # ---- per-stream camera data (GL textures + frames + detections) --------------------------
    det_fos = {}                                     # role -> detections stream follower

    def _det_records(role):
        """Drain new detection records (raw JSON payloads) for a role, in order."""
        fo_ = det_fos.get(role)
        if fo_ is None:
            fo_ = det_fos[role] = framestream.StreamFollower(args.session, f'{role}_det')
        fo_.poll()
        out_ = []
        for rd, i in fo_.drain():
            try:
                out_.append(json.loads(bytes(rd.read(i)).decode('utf-8')))
            except framestream.Lapped:
                continue
        return out_

    def update_cam(stream):
        """Advance a stream's follower: upload a new frame to the GPU (debayer/tonemap pass), poll
        detections, refresh the histogram. When a detector is producing records for this stream we
        display the exact frame it last processed (not the newest), so the boxes sit on the object.
        Returns True if a new frame was uploaded."""
        f = followers.get(stream) or followers.setdefault(stream, SerFollower(args.session, stream))
        ref = f.latest_ref()
        if ref is None or f.header is None:
            return False
        seg, idx = ref.ser_path, ref.index               # (ring ident, ABSOLUTE index)
        if stream in perf['cam']:                        # production rate = newest-index delta *within* a segment
            pseg, pidx = perf['idx'].get(stream, (seg, idx))
            perf['cam'][stream].hit(idx - pidx if (pseg == seg and idx >= pidx) else 0)
            perf['idx'][stream] = (seg, idx)
        cam = cams.get(stream)
        if cam is None:
            fh0, fw0 = f.header.image_height, f.header.image_width
            nb = 2 if bayer.is_bayer(f.header.color_id) else 1
            w, h = fw0 // nb, fh0 // nb
            cam = cams[stream] = dict(gl=_StreamGL(ctx, prog, vao), w=w, h=h, fw=fw0, fh=fh0,
                                      ox=w / fw0, oy=h / fh0, color_id=f.header.color_id,
                                      blobs=[], ext=None, status=None, det_idx=-1, last_idx=-1,
                                      hist=None, ser_path=seg)
        # segment rollover / source switch: det_idx/last_idx are indices into cam['ser_path'],
        # so reset them -- index N in the old segment is a different frame than in the new.
        if seg != cam['ser_path']:
            cam['ser_path'] = seg
            cam['last_idx'] = cam['det_idx'] = -1
            cam['blobs'], cam['ext'], cam['status'] = [], None, None   # old stream's detections are stale
        for rec in (_det_records(stream) if stream in roles else ()):
            # Detections index the cam stream they were made on; a record for a DIFFERENT stream (an
            # in-flight leftover from the previous camera/geometry) must not paint boxes on this one.
            same_seg = (rec.get('seg', '') + '.ser') == cam['ser_path']
            if same_seg:
                cam['blobs'] = rec.get('blobs', [])
                cam['ext'] = rec.get('ext')                     # extended/circmean metrics (or None)
                cam['status'] = rec.get('status')               # detector's freeform status line (or None)
            new_idx = rec.get('index', cam['det_idx']) if same_seg else cam['det_idx']
            if stream in perf['det']:                    # a detection record = one frame the detector ran;
                perf['det'][stream].hit()                # an index gap = frames it skipped to keep up
                if 0 <= cam['det_idx'] < new_idx:
                    perf['skip'][stream].hit(new_idx - cam['det_idx'] - 1)
                if rec.get('proc_ms') is not None:       # the detector's own whole-frame cost
                    _prof(f'det:{stream}', float(rec['proc_ms']))
            cam['det_idx'] = new_idx
        # Pick the frame to show, as an index into `seg`. With "wait for detector" on (default), show the
        # frame the detector last processed (clamped, never ahead) so its boxes match the pixels; with it
        # off, show the newest frame for minimum latency (the detection overlay then lags a frame or two).
        wait_det = view_settings.get(stream, {}).get('wait_for_detector', True)
        show_idx = cam['det_idx'] if (wait_det and 0 <= cam['det_idx'] <= idx) else idx
        if show_idx == cam['last_idx']:
            return False
        hist = None
        try:
            # ZERO-COPY: the GL upload (and the histogram subsample) read the ring slot in
            # place; the view's exit validates the slot wasn't reused mid-use.
            with f.view(FrameRef(seg, show_idx)) as (_rec, disp):
                fh, fw = disp.shape[0], disp.shape[1]
                cam['t_ns'] = int(_rec['t_mono_ns'] or 0)    # capture stamp of the DISPLAYED frame
                _t = time.perf_counter()
                w, h = cam['gl'].upload(disp, f.header.color_id, wb)
                _prof('upload', (time.perf_counter() - _t) * 1e3)
                # Histogram inset (off by default): counts of the RAW LINEAR pixel values -- no gamma,
                # no stretch, so every ADC level lands in its own place (an 8-bit scene fills its bins
                # evenly instead of combing). WB-less subsample -- close enough to WYSIWYG.
                if view_settings.get(stream, {}).get('histogram'):
                    white = float(np.iinfo(disp.dtype).max)
                    samp = (disp[::8, ::8].astype(np.float32) / white).clip(0.0, 1.0)
                    counts, _ = np.histogram(samp, bins=64, range=(0.0, 1.0))
                    m = counts.max()
                    hist = np.sqrt(counts / m) if m > 0 else None
        except (IndexError, ValueError, framestream.Lapped):
            return False                             # reconfigure race / lapped -- retry next tick
        if (w, h) != (cam['w'], cam['h']):               # frame size changed (source/optics switch)
            cam.update(w=w, h=h, fw=fw, fh=fh, ox=w / fw, oy=h / fh, color_id=f.header.color_id)
        cam['last_idx'] = show_idx
        cam['hist'] = hist
        return True

    # ---- pane drawing (letterboxed image + overlays into the pane child's draw list) ---------
    PIPPER_R = S(15)      # pipper (HUD aiming pip) circle radius -- NOT picture-in-picture; a view's
                          # centre crosshairs stop here so a centred pipper joins them

    # Overlay colour scheme: RED = tracker inputs / info; GREEN = tracker products; BLUE = model output.
    # Amber flags a coasting model (no fresh detection).
    COL_INPUT = (255, 70, 70, 160)          # red
    COL_MODEL = (90, 160, 255, 220)         # blue
    COL_COAST = (255, 180, 40, 200)         # amber
    COL_STATIC = (255, 200, 40, 200)        # yellow: a non-moving detection (potential target)
    _SCREW_COL = (230, 185, 70, 235)        # collimation-screw guide colour (amber)
    COL_NAV_STAR = (255, 215, 90, 220)      # gold: where the sky model says a star is
    COL_NAV_SAT = (255, 140, 220, 235)      # magenta: the satellite's pass track + position

    # The 10 outline vertices of the star glyph, as (dx, dy) offsets from its centre. Fixed for
    # the session (S() scales by the startup UI scale), so precompute once instead of 20 trig
    # calls per star per frame -- the overlay draws ~100 of these each redraw.
    _STAR_OFF = [((S(8) if k % 2 == 0 else S(3)) * math.cos(-math.pi / 2.0 + k * math.pi / 5.0),
                  (S(8) if k % 2 == 0 else S(3)) * math.sin(-math.pi / 2.0 + k * math.pi / 5.0))
                 for k in range(10)]

    def _star_glyph(dl, sx, sy, col=COL_NAV_STAR):
        """Five-pointed star outline centred at screen (sx, sy), fixed UI size (not zoomed).
        THE overlay glyph: a star means 'drawn from our alignment + sim time' -- as distinct
        from a box (a detection from the camera) and the pipper circle (a modelled target)."""
        pts = [imgui.ImVec2(sx + dx, sy + dy) for dx, dy in _STAR_OFF]
        dl.add_polyline(pts, C(col), 1.5, int(imgui.ImDrawFlags_.closed))

    def _green(active, a=210):
        """Detection green: bright when it comes from the active tracking source, dim otherwise."""
        return (70, 230, 100, a) if active else (46, 122, 74, min(a, 150))

    def _text(dl, pos, txt, size, color):
        dl.add_text(imgui.get_font(), float(size), pos, C(color), txt)

    def _draw_pipper(dl, A, tx, ty, col, box):
        """Circle at the target, clamped to stay inside the visible camera view `box` (x0,y0,x1,y1 local),
        + 4 short lines from the circle pointing toward the *true* target centre."""
        x0, y0, x1, y1 = box
        m = PIPPER_R + S(2)
        ccx = min(max(tx, x0 + m), max(x0 + m, x1 - m))   # inner max guards a view narrower than 2m
        ccy = min(max(ty, y0 + m), max(y0 + m, y1 - m))
        dl.add_circle(A(ccx, ccy), PIPPER_R, C(col), 0, 1.0)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sx, sy = ccx + dx * PIPPER_R, ccy + dy * PIPPER_R      # a point on the circle
            vx, vy = tx - sx, ty - sy                         # toward the true target centre
            n = math.hypot(vx, vy)
            if n < 1e-6:
                continue
            dl.add_line(A(sx, sy), A(sx + vx / n * PIPPER_R * 0.5, sy + vy / n * PIPPER_R * 0.5), C(col), 1.0)

    def _draw_turn_arc(dl, A, cx, cy, r, turns):
        """A curved arrow around a screw: sweep ∝ turns (clamped to ±1 turn for the arc), arrowhead at the
        end showing CW (tighten, +) / CCW (loosen, −)."""
        sweep = max(-1.0, min(1.0, turns)) * 2.0 * math.pi
        if abs(sweep) < 0.08:
            return
        a0 = -math.pi / 2.0                               # start at the top of the screw
        steps = max(4, int(abs(sweep) / (math.pi / 12)))
        pts = [A(cx + r * math.cos(a0 + sweep * i / steps), cy + r * math.sin(a0 + sweep * i / steps))
               for i in range(steps + 1)]
        dl.add_polyline(pts, C(_SCREW_COL), float(S(2)), 0)
        ae = a0 + sweep                                   # arrowhead at the sweep end, along the tangent
        ex, ey = cx + r * math.cos(ae), cy + r * math.sin(ae)
        tx, ty = -math.sin(ae) * (1 if sweep > 0 else -1), math.cos(ae) * (1 if sweep > 0 else -1)  # tangent
        px, py = -ty, tx                                  # perpendicular
        h = S(5)
        dl.add_triangle_filled(A(ex + tx * h, ey + ty * h),
                               A(ex - tx * h * 0.4 + px * h * 0.7, ey - ty * h * 0.4 + py * h * 0.7),
                               A(ex - tx * h * 0.4 - px * h * 0.7, ey - ty * h * 0.4 - py * h * 0.7),
                               C(_SCREW_COL))

    def _draw_screw_dial(dl, A, cx, cy, R, offset, phase_deg, rad_per_turn):
        """Collimation-screw guide: 3 SCT secondary screws at 120° (rotated by phase_deg to match the
        physical scope), each with a turn arrow to null the CoM offset. `offset` is the collimation error
        in radians (pixel-scale-free). Screw turn = min-norm solution t_k = −(2/3)(û_k·offset)/sens."""
        ox, oy = offset
        sens = rad_per_turn if abs(rad_per_turn) > 1e-12 else 1e-12
        dl.add_circle(A(cx, cy), R, C((90, 96, 110, 130)), 0, 1.0)
        for k in range(3):
            ang = math.radians(phase_deg + 120.0 * k)     # image frame (x right, y down == screen)
            ux, uy = math.cos(ang), math.sin(ang)
            sx, sy = cx + R * ux, cy + R * uy
            t = -(2.0 / 3.0) * (ux * ox + uy * oy) / sens     # turns to null the offset via this screw
            dl.add_circle(A(sx, sy), S(9), C((160, 165, 182, 230)), 0, float(S(2)))
            _text(dl, A(sx - S(3), sy - S(7)), str(k + 1), S(13), (190, 195, 210, 235))
            _draw_turn_arc(dl, A, sx, sy, S(15), t)
            _text(dl, A(sx - S(11), sy + S(13)), f"{t:+.2f}", S(12), _SCREW_COL)

    def _draw_sweep_curve(dl, A, gx0, gy1, W, stt):
        """Sweep peak-vs-position curve on the STAR PANE, stacked under the focus graph: every
        collected frame as a dot (red = saturated, excluded from the fit), the fitted best-focus
        position as a vertical line. Returns the vertical space consumed (0 = nothing drawn)."""
        pts = stt.get('points') or []
        if len(pts) < 3:
            return 0
        H = S(80)
        gy0 = gy1 - H
        xs = [p_ for p_, _h in pts]
        ys = [h_ for _p, h_ in pts]
        pl, ph = min(xs), max(xs)
        if ph <= pl:                                       # first step: one position so far
            pl, ph = pl - 0.5, ph + 0.5
        hh = (max(ys) * 1.05) or 1.0

        def PX(p_):
            return gx0 + S(4) + (W - S(8)) * (p_ - pl) / (ph - pl)

        def PY(h_):
            return gy1 - S(4) - (H - S(8)) * min(h_, hh) / hh
        dl.add_rect_filled(A(gx0 - S(4), gy0 - S(4)), A(gx0 + W + S(4), gy1 + S(4)), C((0, 0, 0, 150)))
        for p_, h_ in pts:
            sat_ = h_ >= 0.98                              # clipped: excluded from the fit
            dl.add_circle_filled(A(PX(p_), PY(h_)), S(1.5),
                                 C((235, 100, 100, 200) if sat_ else (120, 220, 255, 180)))
        p0 = stt.get('p0')
        if p0 is not None and pl <= p0 <= ph:
            dl.add_line(A(PX(p0), gy0 + S(2)), A(PX(p0), gy1 - S(2)), C((255, 190, 90, 220)), 1.2)
        dl.add_rect(A(gx0, gy0), A(gx0 + W, gy1), C((150, 155, 172, 170)), 0.0, 1.0)
        lbl = f"sweep p0 {p0:g}" if p0 is not None else "sweep"
        _text(dl, A(gx0, gy0 - S(16)), lbl, S(12), (255, 190, 90, 235))
        return H + S(26)

    def _draw_placeholder(dl, A, SW, SH, lines):
        """Centred multi-line placeholder text (a str is treated as one line)."""
        if isinstance(lines, str):
            lines = [lines]
        cy = SH / 2.0 - S(11) * (len(lines) - 1)          # vertically centre the block
        for i, ln in enumerate(lines):
            _text(dl, A(SW / 2.0 - S(4) * len(ln), cy + i * S(22)), ln, S(18), (150, 155, 170, 255))

    def _bore_guide_from_main():
        """The guide_from_main boresight rotation from the backend's status (identity until it
        arrives). All cross-camera pixel math goes through this matrix + the cameras'
        perspective matrices -- unproject, rotate, reproject -- never per-axis trig."""
        R9 = (ctrl['state'] or {}).get('boresight_rotation') or (1.0, 0.0, 0.0,
                                                                 0.0, 1.0, 0.0,
                                                                 0.0, 0.0, 1.0)
        return torch.tensor([R9[0:3], R9[3:6], R9[6:9]], dtype=torch.float64)

    def _cam_perspective(w, h, fv, cx=None, cy=None):
        """Perspective matrix for a camera view (w, h) px spanning fv's FoV -- real frames and
        display-scaled views alike (only the pixel unit changes)."""
        fx = (w / 2.0) / math.tan(math.radians(fv['fov_x_deg'] / 2.0))
        fy = (h / 2.0) / math.tan(math.radians(fv['fov_y_deg'] / 2.0))
        return geo.perspective_matrix(fx, fy, w / 2.0 if cx is None else cx,
                                      h / 2.0 if cy is None else cy)

    def _pane_geom(cam, stream, SW, SH):
        """(scale, offx, offy) of the letterboxed image in a pane -- shared by draw + click mapping.

        Zoomed IN on the wider cam (guide), the view centres on the boresight-shifted
        main-cam FoV centre instead of the frame centre, so a deep guide zoom previews
        what the main cam should be seeing."""
        w, h = cam['w'], cam['h']
        zoom = view_settings.setdefault(stream, _default_settings(stream))['zoom']
        scale = _floor_pow2(min(SW / w, SH / h) * 0.95) * zoom
        offx, offy = (SW - w * scale) / 2.0, (SH - h * scale) / 2.0
        if zoom > 1:
            base = stream
            for suf in ('_focus', '_debug'):
                if base.endswith(suf):
                    base = base[:-len(suf)]
            stt = ctrl['state'] or {}
            optx = stt.get('optics') or {}
            me = optx.get(base)
            inner = None                       # a narrower co-aligned cam nested in this view
            if me:
                for r2, fv2 in optx.items():
                    if r2 != base and fv2['fov_x_deg'] < me['fov_x_deg'] \
                            and fv2['fov_y_deg'] < me['fov_y_deg']:
                        inner = fv2
                        break
            if inner is not None:
                mcx, mcy = geo.project_pixel(_cam_perspective(w, h, me),   # main-FoV centre,
                                             _bore_guide_from_main()[:, 0])   # frame px
                offx, offy = SW / 2.0 - mcx * scale, SH / 2.0 - mcy * scale
        return scale, offx, offy

    def draw_slot(name, X0, Y0, SW, SH, dl):
        """Draw the slot's assigned stream letterboxed + centred, with overlays, at the pane's size.
        All geometry is computed in pane-local coords and mapped to screen space through A()."""
        def A(x, y):
            return (X0 + x, Y0 + y)

        role = _slot_stream(name)                        # display stream (may be the <role>_debug surface)
        cam = cams.get(role)
        # Placeholder instead of a stale texture when there's no LIVE feed: either disconnected, or
        # mid-(re)connect -- the old cam's ring has ended and the new one hasn't produced a frame yet
        # (this is what caused "the previous camera flashes for a second when you connect a new one").
        capturing = bool((ctrl['state'] or {}).get('capturing', {}).get(role))
        fol = followers.get(role)
        live = bool(fol is not None and fol.committed_count() > 0 and not fol.ended())
        if role in roles and not (capturing and live):
            if capturing:                                  # connected, but the feed hasn't started yet
                _draw_placeholder(dl, A, SW, SH, [f"{role.capitalize()} Camera", "Connecting…", ""])
                return
            desired = ui['src'].get(role)                  # the dropdown pick (what Connect will use)
            who = ('the simulator' if desired == 'sky'
                   else 'playback' if desired == 'playback'
                   else desired or 'a camera')
            _draw_placeholder(dl, A, SW, SH, [f"{role.capitalize()} Camera", "No Data",
                                              f"Click to connect to {who}"])
            return
        if cam is None or cam['gl'].tex_id is None:
            _draw_placeholder(dl, A, SW, SH, [f"{str(role).capitalize()} Camera", "No Data"])
            return
        w, h = cam['w'], cam['h']
        sset = view_settings.setdefault(role, _default_settings(role))
        scale, offx, offy = _pane_geom(cam, role, SW, SH)
        dw, dh = w * scale, h * scale
        cx, cy = offx + dw / 2.0, offy + dh / 2.0   # FRAME centre (≠ pane centre when zoom
                                                    # re-centres on the main-cam FoV)

        def T(fx, fy):                          # frame (detect) px -> pane-local px
            return offx + fx * cam['ox'] * scale, offy + fy * cam['oy'] * scale

        img_ref = imgui.ImTextureRef(cam['gl'].tex_id)
        ip0, ip1 = A(offx, offy), A(offx + dw, offy + dh)
        if scale > 1.0:
            # Magnifying: route the visible crop through our NEAREST resample pass (see
            # _VS_BLIT) -- drawing the texture magnified through imgui is always bilinear.
            x0, y0 = int(round(max(0.0, offx))), int(round(max(0.0, offy)))
            x1 = int(round(min(float(SW), offx + dw)))
            y1 = int(round(min(float(SH), offy + dh)))
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            bs = blit_slots.setdefault(name, {'size': None, 'tex': None, 'fbo': None})
            if bs['size'] != (bw, bh):
                if bs['fbo'] is not None:
                    bs['fbo'].release()
                    bs['tex'].release()
                bs['tex'] = ctx.texture((bw, bh), 4, dtype='f1')
                bs['fbo'] = ctx.framebuffer(color_attachments=[bs['tex']])
                bs['size'] = (bw, bh)
            prog_blit['src'].value = ((x0 - offx) / dw, (y0 - offy) / dh,
                                      (x1 - x0) / dw, (y1 - y0) / dh)
            bs['fbo'].use()
            samp_nearest.use(0)
            cam['gl'].tex_obj.use(0)
            vao_blit.render(mode=ctx.TRIANGLES, vertices=3)
            samp_nearest.clear(0)
            img_ref = imgui.ImTextureRef(bs['tex'].glo)
            ip0, ip1 = A(x0, y0), A(x1, y1)
        dl.add_image(img_ref, ip0, ip1)

        # Tracking state shared by the overlays below.
        stt = ctrl['state'] or {}
        tracking = bool(stt.get('tracking'))
        active_src = stt.get('track_role') if tracking else None   # the cam currently feeding the model
        this_active = (active_src is None) or (role == active_src)  # bright green unless a *different* cam drives

        # Detection boxes = tracker products: green (moving) / yellow (static) potential targets.
        a = 210 if cam['det_idx'] >= cam['last_idx'] else 70
        for b in cam['blobs']:
            X, Y = T(b['px'][0], b['px'][1])
            half = max(S(4), b.get('size_px', 4) * cam['ox'] * scale) + S(3)
            col = _green(this_active, a) if b.get('moving') else (*COL_STATIC[:3], a)
            dl.add_rect(A(X - half, Y - half), A(X + half, Y + half), C(col), 0.0, 1.0)

        # Single-target-detector overlay: draw whatever the record reports -- an AABB when the
        # detector publishes a meaningful extent, a circle at the CoM -- plus the readout.
        ext = cam.get('ext')
        if ext is not None:
            present = bool(ext.get('present'))
            gcol = _green(this_active) if present else (120, 128, 138, 130)
            bb, com = ext.get('bbox'), ext.get('com')
            if present and bb:
                bx0, by0 = T(bb[0], bb[1])
                bx1, by1 = T(bb[0] + bb[2], bb[1] + bb[3])
                dl.add_rect(A(bx0, by0), A(bx1, by1), C(gcol), 0.0, 1.5)
            if present and com:
                mx_, my_ = T(com[0], com[1])
                dl.add_circle(A(mx_, my_), S(6), C(gcol), 0, 1.5)
            if ext.get('z') is not None:                        # circmean: Rayleigh Z + surviving-pixel count
                readout = f"Z {ext['z']:.0f}  n {ext.get('n', 0)}"
            else:                                               # extended: per-axis compactness
                d = ext.get('density') or [0.0, 0.0]
                readout = f"density {d[0]:.2f} {d[1]:.2f}"
            _text(dl, A(S(8), SH - S(38)),
                  f"{'TARGET' if present else 'no target'}  {readout}", S(13), gcol)

        # --- Reticles = tracker inputs/info: crosshairs + main-cam FoV rect (red; pinhole tan-ratio) ---
        RED = COL_INPUT
        il, ir, it, ib = offx, offx + dw, offy, offy + dh   # image edges (not the letterbox bars)
        if sset['reticles'] and role.endswith('_focus'):
            # Focus view is [EMA | instantaneous] side by side: one full crosshair PER HALF
            # (the plain single reticle would sit uselessly on the seam between them).
            for k in (0.25, 0.75):
                hx = offx + dw * k
                dl.add_line(A(offx + dw * (k - 0.25), cy), A(hx - PIPPER_R, cy), C(RED), 1.0)
                dl.add_line(A(offx + dw * (k + 0.25), cy), A(hx + PIPPER_R, cy), C(RED), 1.0)
                dl.add_line(A(hx, it), A(hx, cy - PIPPER_R), C(RED), 1.0)
                dl.add_line(A(hx, ib), A(hx, cy + PIPPER_R), C(RED), 1.0)
        elif sset['reticles']:
            optx = stt.get('optics', {})
            me = optx.get(role)
            inner = None                          # a narrower co-aligned cam nested in this view (main in guide)
            if me:
                for r2, fv2 in optx.items():
                    if r2 != role and fv2['fov_x_deg'] < me['fov_x_deg'] and fv2['fov_y_deg'] < me['fov_y_deg']:
                        inner = fv2
                        break
            if inner is not None:
                # Main-cam FoV: project the main frame's four CORNERS through guide_from_main
                # into this view -- unproject, rotate, reproject. With a boresight roll the
                # outline is a rotated quad, not an axis-aligned rect.
                R = _bore_guide_from_main()
                Kd = _cam_perspective(dw, dh, me, cx=cx, cy=cy)   # this (guide) view, display px
                Km = _cam_perspective(2.0, 2.0, inner)            # main, a nominal 2x2-px frame

                def _main_px_to_view(qx, qy):
                    return geo.project_pixel(Kd, R @ torch.linalg.solve(
                        Km, torch.tensor([qx, qy, 1.0], dtype=torch.float64)))

                def _centred_view(qx, qy):
                    """The same mapping with an identity boresight: where main-frame pixel
                    (qx, qy) would sit if the quad were perfectly centred."""
                    return geo.project_pixel(Kd, torch.linalg.solve(
                        Km, torch.tensor([qx, qy, 1.0], dtype=torch.float64)))

                quad = [_main_px_to_view(qx, qy)
                        for qx, qy in ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))]
                # Centre crosshairs: from each image edge inward, stopping exactly where the
                # FoV stubs' TIPS sit when the quad is centred -- crosshair and stub meet iff
                # the boresight is centred, so an offset shows as a visible break.
                dl.add_line(A(il, cy), A(max(il, _centred_view(-1.0, 1.0)[0]), cy), C(RED), 1.0)
                dl.add_line(A(ir, cy), A(min(ir, _centred_view(3.0, 1.0)[0]), cy), C(RED), 1.0)
                dl.add_line(A(cx, it), A(cx, max(it, _centred_view(1.0, -1.0)[1])), C(RED), 1.0)
                dl.add_line(A(cx, ib), A(cx, min(ib, _centred_view(1.0, 3.0)[1])), C(RED), 1.0)
                # Main-cam quad + a stub off each side's midpoint. Stubs are MAIN-frame pixel
                # segments (side midpoint, straight out of frame by half the frame size) mapped
                # through the same unproject-rotate-reproject pipeline as the corners -- no
                # screen-space geometry, so they stay correct under any roll. They line up with
                # the centre crosshairs iff the quad is centred, so a boresight offset shows as
                # a visible break (and a roll as a visible tilt).
                for i in range(4):
                    (x0q, y0q), (x1q, y1q) = quad[i], quad[(i + 1) % 4]
                    dl.add_line(A(x0q, y0q), A(x1q, y1q), C(RED), 1.0)
                for seg in (((1.0, 0.0), (1.0, -1.0)), ((2.0, 1.0), (3.0, 1.0)),
                            ((1.0, 2.0), (1.0, 3.0)), ((0.0, 1.0), (-1.0, 1.0))):
                    (x0q, y0q), (x1q, y1q) = (_main_px_to_view(*p) for p in seg)
                    dl.add_line(A(x0q, y0q), A(x1q, y1q), C(RED), 1.0)
            else:
                # Narrowest cam (main): crosshairs from each image edge to the pipper radius, so a
                # centred target's pipper circle connects them.
                dl.add_line(A(il, cy), A(cx - PIPPER_R, cy), C(RED), 1.0)
                dl.add_line(A(ir, cy), A(cx + PIPPER_R, cy), C(RED), 1.0)
                dl.add_line(A(cx, it), A(cx, cy - PIPPER_R), C(RED), 1.0)
                dl.add_line(A(cx, ib), A(cx, cy + PIPPER_R), C(RED), 1.0)

        # --- Sky overlay (navigation feed): five-pointed stars wherever the sky model puts
        # things (stars, Sun/Moon/planets, the satellite), at the encoder pose corrected by the
        # manual Alignment matrix -- nudge until they sit on the real sky. The star
        # glyph MEANS 'drawn from alignment + sim time'; the satellite's upcoming pass adds a
        # line, and bodies with a real angular size get their disc at true scale. Projection =
        # skysim.project_dirs, the SAME pinhole math the sim renders with.
        if (sset.get('show_stars', True) and role in roles and nav.get('alm') is not None
                and stt.get('enc_az_deg') is not None):
            fv_nav = (stt.get('optics') or {}).get(role)
            nav_ids = nav['alm'].ids
            nav_names = nav['alm'].names
            nav_radius = nav['alm'].angular_radius_rad
            if fv_nav and nav_ids:
                # Everything evaluates AT THE DISPLAYED FRAME'S capture time: almanac query
                # and pose alike, so the overlay can't lag the pixels it sits on.
                t0 = int(cam.get('t_ns') or session_mod.mono_ns())
                if len(pose_hist):
                    az0, alt0 = pose_hist.pose_at(t0 * 1e-9)
                else:                                     # no samples yet: latest state pose
                    az0 = math.radians(stt.get('enc_az_deg') or 0.0)
                    alt0 = math.radians(stt.get('enc_alt_deg') or 0.0)
                align9 = stt.get('align_matrix')          # sky_from_mount rotation (flat 3x3)
                az_nav, alt_nav = az0, alt0
                f_px_nav = (cam['w'] / 2.0) / math.tan(math.radians(fv_nav['fov_x_deg'] / 2.0))
                cx_nav, cy_nav = cam['w'] / 2.0, cam['h'] / 2.0
                # Two DIFFERENT kinds of thing, drawn differently:
                #  - stars / Sun / Moon / planets: look up NOW (t0) in each one's fix list to get
                #    its current direction -> one glyph. (The satellite is in here too: its 'now'
                #    glyph.)  One batched lerp over every target at the single time t0.
                #  - the satellite pass line: the almanac ALREADY holds it as a list of (time,
                #    dir) fixes -- the fixes ARE the polyline, so just draw them, no lookup and no
                #    resampling. Past fixes are already evicted at the query floor (~t0), so this
                #    is now -> horizon. Not worth the complexity to also trim points behind us.
                now_dirs, _mags = nav['alm'].dirs_at(torch.tensor([t0], dtype=torch.int64))
                now_dirs = now_dirs[:, 0, :]                    # one current direction per target
                track_index = nav_ids.index('sat:track') if 'sat:track' in nav_ids else -1
                _track_times, track_dirs = nav['alm'].fixes('sat:track')   # raw pass fixes (points, 3)
                # Project the glyphs (one per target) and the raw track fixes in ONE call, then split.
                n_targets = len(nav_ids)
                all_dirs = (now_dirs if track_dirs is None
                            else torch.cat([now_dirs, track_dirs], 0))
                if align9:
                    # Apply the model rotation to the world dirs: dirs @ A applies A^T per row,
                    # which projected at the raw encoder pose equals viewing through the aligned
                    # orientation A @ M_enc. The backend's matrix is in geometry's (N,E,U) world
                    # frame; the projector works in (E,N,U) -- same angles, x/y swapped -- so
                    # permute the basis first.
                    p = (1, 0, 2)
                    all_dirs = all_dirs @ torch.tensor(
                        [[align9[p[i] * 3 + p[j]] for j in range(3)] for i in range(3)],
                        dtype=all_dirs.dtype)
                px, py, ok = skysim.project_dirs(all_dirs, az_nav, alt_nav,
                                                 f_px_nav, cx_nav, cy_nav)
                px, py, ok = px.tolist(), py.tolist(), ok.tolist()

                def to_pane(x, y):                # frame coords -> pane: +0.5 because a frame
                    # coordinate is a PIXEL CENTRE (skysim's splat convention) and pixel i
                    # displays spanning [i, i+1) -- at high zoom the half pixel is visible.
                    return A(offx + (x + 0.5) * scale, offy + (y + 0.5) * scale)

                show_target_names = sset.get('show_target_names', True)
                show_star_names = sset.get('show_star_names', False)

                def _nav_label(sx, sy, target_index, col):
                    name = nav_names[target_index]
                    if name.startswith('star:'):  # no proper name published: fall back to HIP
                        name = 'HIP ' + name[5:]
                    _text(dl, imgui.ImVec2(sx + S(9), sy - S(16)), name, S(13), (*col[:3], 210))

                for target_index in range(n_targets):   # glyph at each target's current direction
                    if target_index == track_index or not ok[target_index]:
                        continue
                    x0, y0 = px[target_index], py[target_index]
                    if not (-64 <= x0 <= cam['w'] + 64 and -64 <= y0 <= cam['h'] + 64):
                        continue
                    sx, sy = to_pane(x0, y0)
                    _star_glyph(dl, sx, sy)
                    if nav_radius[target_index]:  # sun/moon/planets: the TRUE disc, at scale
                        r_px = f_px_nav * nav_radius[target_index] * scale
                        if r_px >= 2.0:
                            dl.add_circle(imgui.ImVec2(sx, sy), r_px, C(COL_NAV_STAR), 0, 1.5)
                    is_star = nav_ids[target_index].startswith('star:')
                    if (show_star_names if is_star else show_target_names):
                        _nav_label(sx, sy, target_index, COL_NAV_STAR)
                if track_index >= 0 and track_dirs is not None:
                    # The track fixes were projected right after the glyphs, so slice them back out.
                    track_px, track_py, track_visible = (px[n_targets:], py[n_targets:],
                                                         ok[n_targets:])
                    for point in range(len(track_px) - 1):   # raw pass fixes, drawn segment by segment
                        if track_visible[point] and track_visible[point + 1]:
                            dl.add_line(to_pane(track_px[point], track_py[point]),
                                        to_pane(track_px[point + 1], track_py[point + 1]),
                                        C((*COL_NAV_SAT[:3], 150)), 1.5)
                    if ok[track_index]:           # 'now' glyph at the satellite's current direction
                        sx, sy = to_pane(px[track_index], py[track_index])
                        _star_glyph(dl, sx, sy, COL_NAV_SAT)
                        if show_target_names:
                            _nav_label(sx, sy, track_index, COL_NAV_SAT)

        # Clamp box = the visible camera view (image ∩ pane): letterboxed image zoomed out, the pane zoomed in.
        box = (max(0.0, il), max(0.0, it), min(float(SW), ir), min(float(SH), ib))

        # BLUE = model prediction: drawn in the active source cam directly, and mapped via the pinhole
        # tan-ratio into any other cam. Amber while coasting.
        mpip = None
        if tracking and stt.get('target_px'):
            if active_src == role:
                mpip = T(stt['target_px'][0], stt['target_px'][1])
            elif active_src and active_src != role:
                optx = stt.get('optics') or {}
                src, sf, mf = cams.get(active_src), optx.get(active_src), optx.get(role)
                if src is not None and sf and mf:
                    gtx, gty = stt['target_px'][0] * src['ox'], stt['target_px'][1] * src['oy']
                    # Unproject from the source camera, rotate between the camera spaces
                    # (guide_from_main; transposed when the source is the wider/guide side),
                    # reproject through the destination camera. Matrices only -- exact across
                    # the whole frame, not just on the boresight axis.
                    R = _bore_guide_from_main()
                    d = torch.linalg.solve(_cam_perspective(src['w'], src['h'], sf),
                                           torch.tensor([gtx, gty, 1.0], dtype=torch.float64))
                    d = (R.T if sf['fov_x_deg'] >= mf['fov_x_deg'] else R) @ d
                    mtx, mty = geo.project_pixel(_cam_perspective(cam['w'], cam['h'], mf), d)
                    mpip = (offx + mtx * scale, offy + mty * scale)
        if mpip is not None:
            _draw_pipper(dl, A, mpip[0], mpip[1],
                         COL_COAST if stt.get('mode') == 'coast' else COL_MODEL, box)

        # GREEN = the raw detection this instant: a bright circle at where the active source's detector
        # actually put the locked target -- only in that source cam.
        if tracking and active_src == role and stt.get('detect_px'):
            dcx, dcy = T(stt['detect_px'][0], stt['detect_px'][1])
            dcx = min(max(dcx, box[0] + PIPPER_R), max(box[0] + PIPPER_R, box[2] - PIPPER_R))
            dcy = min(max(dcy, box[1] + PIPPER_R), max(box[1] + PIPPER_R, box[3] - PIPPER_R))
            dl.add_circle(A(dcx, dcy), PIPPER_R * 0.72, C(_green(True)), 0, 1.6)

        # Tracker ROI = an input we hand the detector: red; solid for the active source, dim for a fallback.
        roi = (stt.get('track_roi') or {}).get(role)
        if roi:
            rcx, rcy, rsz = roi
            x0, y0 = T(rcx - rsz / 2.0, rcy - rsz / 2.0)
            x1, y1 = T(rcx + rsz / 2.0, rcy + rsz / 2.0)
            col = COL_INPUT if role == active_src else (*COL_INPUT[:3], 80)
            dl.add_rect(A(x0, y0), A(x1, y1), C(col), 0.0, 1.0)

        # Focus overlays (focus stream only): focus-quality graph lower-right, collimation-screw dial
        # above it, and the collimation trail (last ~10 EMA CoM offsets, exaggerated) on the star.
        if role.endswith('_focus') and role == focus_ui['role'] + '_focus':
            sc = focus_ui['series']
            GW, GH, mgn = min(S(220), max(S(90), SW - S(20))), S(80), S(10)
            gx1, gy1 = SW - mgn, SH - mgn
            gx0 = gx1 - GW
            # Sweep curve takes the bottom slot; the focus graph (and the dial above it) stack up.
            gy1 -= _draw_sweep_curve(dl, A, gx0, gy1, GW, sweep_ui['state'] or {})
            gy0 = gy1 - GH
            strehls = sc['strehl']
            use_str = bool(strehls) and strehls[-1] is not None
            series = [v if v is not None else 0.0 for v in strehls] if use_str else sc['peak']
            if len(series) >= 2:
                # Both series draw min..max autoscaled: a focus graph is about the TREND, and a
                # fixed 0..1 axis glued a 0.03-Strehl line to the border (read as "no graph").
                # The label carries the absolute numbers.
                lo, hi = min(series), max(series)
                span = (hi - lo) or 1.0
                label = f"Strehl {strehls[-1]:.2f}" if use_str else f"focus peak {sc['peak'][-1]:.3f}"
                dl.add_rect_filled(A(gx0 - S(4), gy0 - S(4)), A(gx1 + S(4), gy1 + S(4)), C((0, 0, 0, 150)))
                n = len(series)
                pts = [A(gx0 + GW * i / (n - 1), gy1 - GH * max(0.0, min(1.0, (v - lo) / span)))
                       for i, v in enumerate(series)]
                dl.add_polyline(pts, C((120, 220, 255, 235)), 1.4, 0)
                dl.add_rect(A(gx0, gy0), A(gx1, gy1), C((150, 155, 172, 170)), 0.0, 1.0)
                _text(dl, A(gx0, gy0 - S(16)), label, S(12), (120, 220, 255, 235))
            Rd = S(38)
            ox_, oy_ = focus_ui['skew_rad']
            off = (-ox_ if focus_ui['invert_x'] else ox_, -oy_ if focus_ui['invert_y'] else oy_)
            dcy2 = max(Rd + S(14), gy0 - Rd - S(42))       # above the graph, clear of its top label
            _draw_screw_dial(dl, A, (gx0 + gx1) / 2.0, dcy2, Rd,
                             off, focus_ui['screw_phase'], focus_ui['rad_per_turn'])
            # Shape glyphs, one per half ([EMA | instantaneous], mono crop so ox == 1), same
            # math on both, just different target images:
            #  - astigmatism ELLIPSE: centred on the crosshair, mean radius = the crosshair
            #    gap, orientation exact, ellipticity exaggerated by Shape x; a circle = round.
            #  - coma SKEW ARROW: from the crosshair to the halo's centroid offset, in image
            #    px exaggerated by Skew x -- collimation drives this to zero.
            mult = focus_ui['com_mult']
            gain = focus_ui['shape_gain']
            for half_cx, shape in ((cam['fw'] / 4.0, focus_ui['shape_ema']),
                                   (cam['fw'] * 3.0 / 4.0, focus_ui['shape_instant'])):
                if shape is None:
                    continue
                e1_, e2_, skx_, sky_ = shape
                ctr_fx, ctr_fy = half_cx, cam['fh'] / 2.0
                emag = math.hypot(e1_, e2_)
                theta = 0.5 * math.atan2(e2_, e1_) if emag > 1e-9 else 0.0
                ex = min(0.95, gain * emag)
                a_ax = PIPPER_R * math.sqrt(1.0 + ex)
                b_ax = PIPPER_R * math.sqrt(1.0 - ex)
                ct, st_ = math.cos(theta), math.sin(theta)
                cgx, cgy = T(ctr_fx, ctr_fy)
                pts = []
                for k in range(32):
                    ang = 2.0 * math.pi * k / 32
                    px_ = a_ax * math.cos(ang)
                    py_ = b_ax * math.sin(ang)
                    pts.append(imgui.ImVec2(*A(cgx + px_ * ct - py_ * st_,
                                               cgy + px_ * st_ + py_ * ct)))
                dl.add_polyline(pts, C(COL_INPUT), 1.6, int(imgui.ImDrawFlags_.closed))
                tipx, tipy = T(ctr_fx + skx_ * mult, ctr_fy + sky_ * mult)
                dl.add_line(A(cgx, cgy), A(tipx, tipy), C(COL_INPUT), 1.6)
                dl.add_circle_filled(A(tipx, tipy), S(3), C(COL_INPUT))

        # Cut-off indicators: when zoomed past fit the image overflows -> arrows on the cropped edges.
        if dw > SW + 1:
            for ex, sx in ((-1, S(12)), (1, SW - S(12))):
                dl.add_triangle_filled(A(sx, cy - S(9)), A(sx, cy + S(9)), A(sx + ex * S(11), cy),
                                       C((255, 170, 40, 170)))
                dl.add_triangle(A(sx, cy - S(9)), A(sx, cy + S(9)), A(sx + ex * S(11), cy),
                                C((255, 170, 40, 220)))
        if dh > SH + 1:
            for ey, sy in ((-1, S(12)), (1, SH - S(12))):
                dl.add_triangle_filled(A(cx - S(9), sy), A(cx + S(9), sy), A(cx, sy + ey * S(11)),
                                       C((255, 170, 40, 170)))
                dl.add_triangle(A(cx - S(9), sy), A(cx + S(9), sy), A(cx, sy + ey * S(11)),
                                C((255, 170, 40, 220)))

        # Histogram inset (toggle), bottom-right, fixed UI size (not zoomed) -- judge exposure/clipping.
        if sset['histogram'] and cam.get('hist') is not None:
            bars = cam['hist']
            HW, HH, mgn = min(S(180), max(S(60), SW - S(20))), S(70), S(10)
            hx1, hy1 = SW - mgn, SH - mgn
            hx0, hy0 = hx1 - HW, hy1 - HH
            dl.add_rect_filled(A(hx0 - S(4), hy0 - S(4)), A(hx1 + S(4), hy1 + S(4)), C((0, 0, 0, 150)))
            bw = HW / len(bars)
            for i, hgt in enumerate(bars):
                bx0 = hx0 + i * bw
                dl.add_rect_filled(A(bx0, hy1 - float(hgt) * HH), A(bx0 + bw, hy1), C((205, 215, 235, 230)))
            dl.add_rect(A(hx0, hy0), A(hx1, hy1), C((180, 180, 180, 200)), 0.0, 1.0)

        # Status line (bottom-left) + a blinking red warning while tracking if this cam is disconnected
        # (not capturing) or running-but-not-recording.
        st_now = ctrl['state'] or {}
        capturing = bool(st_now.get('capturing', {}).get(role))
        recording = bool((st_now.get('recording') or {}).get(role)) and capturing
        srclabel, fps_str = '', ''
        if role in roles:                                # which source/camera + its production framerate
            src = (st_now.get('sources') or {}).get(role)
            camurl = (st_now.get('camera') or {}).get(role)
            srclabel = (camurl.replace('zwo:', '') if (src == 'zwo' and camurl) else (src or '?')) + "  "
            m = perf['cam'].get(role)
            fps_str = f"{m.rate:.0f} fps  " if m else ''
        status = (f"{role}  {srclabel}f{cam['last_idx']}  {fps_str}{_color_name(cam['color_id'])}  "
                  f"blobs {len(cam['blobs'])}  zoom {round(scale * 100)}%" + ("  REC" if recording else ""))
        _text(dl, A(S(8), SH - S(20)), status, S(13), (200, 205, 220, 230))
        if role in roles and st_now.get('tracking') and int(time.perf_counter() * 1.5) % 2 == 0:
            msg = "NOT CONNECTED" if not capturing else ("NOT RECORDING" if not recording else None)
            if msg:
                _text(dl, A(cx - len(msg) * S(40) * 0.30, cy + S(20)), msg, S(40), (255, 40, 40, 255))

    # ---- pane clicks -------------------------------------------------------------------------
    def _toggle_connect(role):
        """Connect/Disconnect: fully start or stop this role's cam process, driven off the backend's
        actual capture state (telemetry) so the button is a plain toggle. Connect first applies the
        desired source (+ camera) chosen in the dropdown, then starts capture -- so the feed only
        switches when you press Connect, never on a mere dropdown pick."""
        on = bool(((ctrl['state'] or {}).get('capturing') or {}).get(role))
        if not on:
            _apply_desired_source(role)                    # push desired source/camera, THEN capture on
        _send({'type': 'capture', 'role': role, 'on': not on})

    def _pane_click(name, X0, Y0, SW, SH):
        """A click in a pane body locks the nearest blob (or the bare point) and tracks. Derived
        streams (focus star / debug) are not pickers; a disconnected pane connects instead."""
        role = _slot_role(name)
        if _slot_stream(name) != role:
            return
        if role in roles and not bool((ctrl['state'] or {}).get('capturing', {}).get(role)):
            _toggle_connect(role)                          # disconnected pane body -> "Click to connect"
            return
        cam = cams.get(role)
        if cam is None:
            return
        if (ctrl['state'] or {}).get('tracking'):          # already locked -> a pane click doesn't re-target
            return                                          # (press Esc to unlock first, then pick a new one)
        mx, my = io.mouse_pos.x, io.mouse_pos.y
        scale, offx, offy = _pane_geom(cam, role, SW, SH)
        fx = ((mx - X0) - offx) / scale / cam['ox']         # pane screen -> texture px -> frame (detect) px
        fy = ((my - Y0) - offy) / scale / cam['oy']
        best, bd = None, 1e18
        for b in cam['blobs']:
            dx, dy = b['px'][0] - fx, b['px'][1] - fy
            d = dx * dx + dy * dy
            if d < bd:
                bd, best = d, b
        px = best['px'] if (best is not None and bd <= 40 * 40) else [fx, fy]
        _send({'type': 'track', 'role': role, 'px': [float(px[0]), float(px[1])]})

    def _toolbar_defs(name):
        """(label, action, checked) for a pane's top-left controls: checked=None renders a
        button, a bool renders a checkbox showing that state. Actions resolve the slot's
        stream at call time, so they follow a promote/swap."""
        if name == 'big':
            return [('Show Settings', lambda: layout.__setitem__('panel_open', not layout['panel_open']),
                     layout['panel_open']),
                    ('Show PIPs', lambda: layout.__setitem__('pip_open', not layout['pip_open']),
                     layout['pip_open']),
                    ('Request Debug', _toggle_dbg, layout['pip_debug']),   # detector surface pip
                    ('  -  ', lambda: _zoom_step(_slot_stream('big'), -1), None),
                    ('  +  ', lambda: _zoom_step(_slot_stream('big'), +1), None)]
        return [('  ^  ', lambda n=name: _promote(_slot_stream(n)), None),
                ('  -  ', lambda n=name: _zoom_step(_slot_stream(n), -1), None),
                ('  +  ', lambda n=name: _zoom_step(_slot_stream(n), +1), None)]

    def _pane(name, x, y, w, h):
        """One camera pane: a child window with a full-size click catcher, the stream + overlays
        drawn into its draw list, and real toolbar buttons on top."""
        imgui.set_cursor_screen_pos((x, y))
        imgui.push_style_color(imgui.Col_.child_bg, C4(30, 32, 38))   # dim letterbox bars
        imgui.begin_child(f"slot_{name}", (max(1.0, w), max(1.0, h)), 0, PANE_WF)
        X0, Y0 = imgui.get_window_pos().x, imgui.get_window_pos().y
        dl = imgui.get_window_draw_list()
        imgui.set_cursor_pos((0, 0))
        # The pane-wide click catcher must allow overlap, or the toolbar buttons submitted after it
        # (visually on top) never win hover -- ImGui gives the FIRST hover claimant priority otherwise.
        imgui.set_next_item_allow_overlap()
        imgui.invisible_button(f"##pane_{name}", (max(1.0, w), max(1.0, h)),
                               imgui.ButtonFlags_.allow_overlap)
        clicked = imgui.is_item_clicked(0)
        draw_slot(name, X0, Y0, w, h, dl)
        imgui.set_cursor_pos((S(6), S(6)))                # toolbar: real buttons, over the image
        for label, action, checked in _toolbar_defs(name):
            if checked is None:
                if imgui.button(f"{label}##tb_{name}"):
                    action()
            else:
                ch, _v = imgui.checkbox(f"{label}##tb_{name}", checked)
                if ch:
                    action()
            imgui.same_line()
        imgui.end_child()
        imgui.pop_style_color()
        if clicked:
            _pane_click(name, X0, Y0, w, h)

    # ---- optics DB gear pickers ----------------------------------------------------------------
    _SENS, _OPT, _RED = optics_db.load_db()
    _GEAR = {'sensor': sorted(_SENS), 'optic': sorted(_OPT), 'reducer': ['(none)'] + sorted(_RED)}
    owned = {'sensor': set(), 'optic': set(), 'reducer': set()}
    _DIV = '-' * 14                               # a (non-selectable) divider row in the combos

    def _gear_items(kind):
        own = [n for n in _GEAR[kind] if n in owned[kind]]
        rest = [n for n in _GEAR[kind] if n not in owned[kind]]
        return (own + [_DIV] + rest) if own else rest

    def _send_optics(role):
        sel = ui['opt'][role]
        sen, opt, red = sel['sensor'], sel['optic'], sel['reducer']
        if _DIV in (sen, opt, red):
            return
        _send({'type': 'set_optics', 'role': role, 'sensor': sen or None, 'optic': opt or None,
               'reducer': None if red in (None, '', '(none)') else red})

    def _automatch_optics(role, url):
        """When a ZWO camera whose model is a known DB sensor is picked, point this role's Optics
        sensor at it (so the plate scale matches the actual chip)."""
        if not (url and url.startswith('zwo:')):
            return
        model = url[len('zwo:'):].rsplit('#', 1)[0]
        if model in _SENS:
            ui['opt'][role]['sensor'] = model
            _send_optics(role)

    def _apply_desired_source(role):
        """Push the GUI's DESIRED source (+ camera, from ui['src']) to the backend. Called from
        Connect ONLY -- so picking in the dropdown never disturbs a running cam; only Connect
        switches the feed. The backend's set_source stops any running cam and waits, then the
        following capture-on relaunches it on the chosen source."""
        val = ui['src'].get(role)
        if val == 'sky':
            _send({'type': 'set_source', 'role': role, 'source': 'sky'})
        elif val == 'playback':
            _send({'type': 'set_source', 'role': role, 'source': 'playback'})
        elif val:                                          # a ZWO camera URL -> source zwo + that camera
            _send({'type': 'set_source', 'role': role, 'source': 'zwo'})
            _send({'type': 'set_camera', 'role': role, 'url': None if val == '(auto)' else val})

    # Unified source dropdown: one list of "where this pane's frames come from" -- each detected ZWO
    # camera (by model), 'sky', or 'playback'. Picking a camera sets source=zwo + that camera at once.
    def _source_items(st):
        return ['sky', 'playback'] + list((st or {}).get('cameras_available') or [])

    def _source_value(st, role):
        """The dropdown value that reflects the backend's (source, camera) for this role, or None."""
        src = ((st or {}).get('sources') or {}).get(role)
        cam = ((st or {}).get('camera') or {}).get(role)
        if src == 'sky':
            return 'sky'
        if src == 'playback':
            return 'playback'
        if src == 'zwo':
            return cam if (cam and cam != '(auto)') else None
        return None

    def _on_source_pick(role, val):
        """A dropdown pick only chooses the DESIRED source (ui['src'], set by the caller). It does
        NOT connect, disconnect, or switch anything -- press Connect to apply. Picking a camera
        auto-matches its optics as a convenience (harmless while disconnected)."""
        if val and val.startswith('zwo:'):
            _automatch_optics(role, val)

    # ---- caps-driven camera controls (exposure/gain/...) ---------------------------------------
    # The cam publishes each control's kind/range/value; we render a "[<<][<] value [>][>>]" stepper
    # per number control and push changes live. The GUI owns the value once shown; cam_ctrl_val (and
    # the input buffers) are re-seeded only when the *set* of controls changes (source switch).
    def _fmt_ctrl(v):
        return f"{v:.4g}"

    def _set_cam_ctrl(role, desc, value):
        value = min(desc['max'], max(desc['min'], float(value)))
        cam_ctrl_val[(role, desc['name'])] = value
        ui['txt'][f"cinp_{role}_{desc['name']}"] = _fmt_ctrl(value)
        _send({'type': 'set_cam_control', 'role': role, 'name': desc['name'], 'value': value})

    def _num_step(role, desc, kind):
        cur = cam_ctrl_val.get((role, desc['name']), desc.get('value', 0.0))
        if desc.get('scale') == 'log':
            cur = max(cur, 1e-9)                        # multiplicative: never stuck at 0
            cur *= {'ld': 0.5, 'sd': 2 ** -0.25, 'su': 2 ** 0.25, 'lu': 2.0}[kind]
        else:
            span = max(1e-9, desc['max'] - desc['min'])
            cur += {'ld': -span / 10, 'sd': -span / 50, 'su': span / 50, 'lu': span / 10}[kind]
        _set_cam_ctrl(role, desc, cur)

    def _seed_cam_controls(role, caps):
        """Take the backend's current control values as ours (on a control-set change only)."""
        for desc in (caps or {}).get('controls', []):
            cam_ctrl_val[(role, desc['name'])] = desc.get('value', 0.0)
            if desc.get('kind') == 'number':               # choice values are strings -- no text buffer
                ui['txt'][f"cinp_{role}_{desc['name']}"] = _fmt_ctrl(desc.get('value', 0.0))

    def _panel_cam_controls(role, caps):
        for desc in (caps or {}).get('controls', []):
            nm = desc['name']
            if desc.get('kind') == 'number':
                imgui.text(f"{desc.get('label', nm)}:")
                imgui.same_line()
                for lbl, k in (('<<', 'ld'), ('<', 'sd')):
                    if imgui.button(f"{lbl}##c_{role}_{nm}_{k}", (S(24), 0)):
                        _num_step(role, desc, k)
                    imgui.same_line()
                tid = f"cinp_{role}_{nm}"
                buf = ui['txt'].setdefault(tid, _fmt_ctrl(desc.get('value', 0.0)))
                imgui.set_next_item_width(S(62))
                ch, buf = imgui.input_text('##' + tid, buf, ENTER)
                ui['txt'][tid] = buf
                if ch:
                    try:
                        v = float(buf)
                    except ValueError:
                        v = cam_ctrl_val.get((role, nm), desc.get('value', 0.0))
                    _set_cam_ctrl(role, desc, v)
                for lbl, k in (('>', 'su'), ('>>', 'lu')):
                    imgui.same_line()
                    if imgui.button(f"{lbl}##c_{role}_{nm}_{k}", (S(24), 0)):
                        _num_step(role, desc, k)
                if desc.get('unit'):
                    imgui.same_line()
                    imgui.text_colored(C4(140, 145, 160), desc['unit'])
                if nm == 'gain' and (caps or {}).get('source') == 'zwo':   # HCG threshold from the sensor DB
                    sen = _SENS.get((ui['opt'].get(role) or {}).get('sensor', ''))
                    hcg = sen.hcg_gain if sen else 0
                    if hcg > 0:
                        on = cam_ctrl_val.get((role, nm), desc.get('value', 0.0)) >= hcg
                        imgui.same_line()
                        imgui.text_colored(C4(120, 210, 130) if on else C4(150, 155, 170),
                                           f"HCG@{hcg}{' on' if on else ''}")
                        _tip(f"High Conversion Gain engages at gain {hcg} on this sensor; read noise "
                             f"drops sharply there. Gain >= {hcg} runs in low-noise HCG mode.")
            elif desc.get('kind') == 'choice':         # relaunch-tier (binning/ROI): each pick relaunches
                imgui.text(f"{desc.get('label', nm)}:")
                imgui.same_line()
                items = [str(c) for c in desc.get('choices', [])]
                cur = str(cam_ctrl_val.get((role, nm), desc.get('value', '')))
                idx = items.index(cur) if cur in items else -1
                imgui.set_next_item_width(S(72))
                ch, nidx = imgui.combo(f"##c_{role}_{nm}", idx, items)
                if ch and 0 <= nidx < len(items):
                    cam_ctrl_val[(role, nm)] = items[nidx]
                    _send({'type': 'set_cam_control', 'role': role, 'name': nm, 'value': items[nidx]})
            elif desc.get('kind') == 'bool':               # a plain on/off toggle (e.g. High Speed Mode)
                cur = bool(cam_ctrl_val.get((role, nm), desc.get('value', False)))
                ch, v = imgui.checkbox(f"{desc.get('label', nm)}##c_{role}_{nm}", cur)
                if ch:
                    cam_ctrl_val[(role, nm)] = v
                    _send({'type': 'set_cam_control', 'role': role, 'name': nm, 'value': v})
            # (file kind -> later slices)

    # ---- settings persistence -------------------------------------------------------------------
    def gather_settings():
        return {
            'version': 1,
            'layout': {k: layout[k] for k in ('panel_w', 'pip_h', 'panel_open', 'pip_open', 'pip_debug', 'big_role')},
            'display': {role: dict(view_settings.get(role, _default_settings(role))) for role in roles},
            'optics': {
                'owned': {k: sorted(v) for k, v in owned.items()},
                'selection': {role: [ui['opt'][role][k] for k in ('sensor', 'optic', 'reducer')]
                              for role in roles},
            },
            'cameras': {role: ui['src'].get(role) for role in roles if ui['src'].get(role)},
            'boresight': [boresight_ui['x'], boresight_ui['y'], boresight_ui['roll']],
            # The alignment matrix lives on the backend; persist its latest published value.
            'alignment': {'matrix': (ctrl['state'] or {}).get('align_matrix')
                                    or [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
            # Location persists; TIME deliberately does not (a stale offset next session would be
            # worse than starting at the system clock).
            'location': {'lat_deg': site_ui['lat'], 'lon_deg': site_ui['lon'],
                         'elev_m': site_ui['elev']},
            'tracking': {'smoothing': track_ui['smoothing'], 'latency': track_ui['latency'],
                         'rate_scale': track_ui['rate_scale'], 'pref': track_ui['pref'],
                         'auto_switch': track_ui['auto_switch'], 'delay': track_ui['delay'],
                         'model': track_ui['model'], 'alt_km': track_ui['alt_km'],
                         'feedforward': track_ui['feedforward'],
                         'position_smoothing': track_ui['position_smoothing'],
                         'min_horizon': track_ui['min_horizon'],
                         'max_horizon': track_ui['max_horizon'],
                         'pid_on': track_ui['pid_on'], 'pid_kp': track_ui['pid_kp'],
                         'pid_ki': track_ui['pid_ki'], 'pid_kd': track_ui['pid_kd'],
                         'pid_integral_limit': track_ui['pid_integral_limit'],
                         'follow': bool((ctrl['state'] or {}).get('follow_enabled', True))},
            'sim': {'r0': sim_ui['r0'], 'bortle': sim_ui['bortle']},   # per-cam defocus rides with the cam caps
            'focus': {'com_mult': focus_ui['com_mult'], 'shape_gain': focus_ui['shape_gain'],
                      'screw_phase': focus_ui['screw_phase'],
                      'rad_per_turn': focus_ui['rad_per_turn'], 'alpha': focus_ui['alpha'],
                      'invert_x': focus_ui['invert_x'], 'invert_y': focus_ui['invert_y']},  # collimation calib
        }

    def apply_settings(data):
        for k, v in (data.get('layout') or {}).items():
            if k in layout:
                layout[k] = v
        for role, s in (data.get('display') or {}).items():
            vs = view_settings.setdefault(role, _default_settings(role))
            for k in ('zoom', 'reticles', 'histogram', 'wait_for_detector', 'show_stars',
                      'show_target_names', 'show_star_names'):
                if k in s:
                    vs[k] = s[k]
        opt = data.get('optics') or {}
        for k, names in (opt.get('owned') or {}).items():
            if k in owned:
                owned[k] = set(names)
        for role, sel in (opt.get('selection') or {}).items():
            if not (sel and role in ui['opt']):
                continue
            for k, v in zip(('sensor', 'optic', 'reducer'), sel):
                if v:
                    ui['opt'][role][k] = v
            _send_optics(role)                          # push the loaded optics to the backend
        for role, val in (data.get('cameras') or {}).items():
            if val and role in roles:
                if val in _source_items(ctrl['state']):
                    ui['src'][role] = val
                _on_source_pick(role, val)              # push the loaded source (+ camera) to the backend
        b = data.get('boresight')
        if b and len(b) >= 2:
            _bore_set(b[0], b[1], b[2] if len(b) > 2 else 0.0)   # update buffers + push to the backend
            ctrl['bore_init'] = True                    # ...and don't let the state-init clobber it
        al = data.get('alignment') or {}
        if al.get('matrix') and len(al['matrix']) == 9:   # push the saved alignment to the backend
            _send({'type': 'set_alignment', 'matrix': [float(v) for v in al['matrix']]})
        loc = data.get('location') or {}
        if 'lat_deg' in loc:
            _site_set(loc['lat_deg'], loc.get('lon_deg', 0.0), loc.get('elev_m', 0.0))
            ctrl['site_init'] = True                    # ...and don't let the state-init clobber it
        trk = data.get('tracking') or {}
        if 'smoothing' in trk:
            track_ui['smoothing'] = max(0.0, float(trk['smoothing']))
            ui['txt']['track_smooth'] = f"{track_ui['smoothing']:g}"
            _send({'type': 'set_track_smoothing', 'value': track_ui['smoothing']})
        if 'latency' in trk:
            track_ui['latency'] = max(-5.0, min(5.0, float(trk['latency'])))
            ui['txt']['track_latency'] = f"{track_ui['latency']:g}"
            _send({'type': 'set_track_latency', 'value': track_ui['latency']})
        if 'rate_scale' in trk:
            track_ui['rate_scale'] = max(0.5, min(2.0, float(trk['rate_scale'])))
            ui['txt']['track_rate_scale'] = f"{track_ui['rate_scale']:g}"
            _send({'type': 'set_track_rate_scale', 'value': track_ui['rate_scale']})
        if trk.get('pref') in ('guide', 'main', 'auto'):
            _set_track_pref(trk['pref'])                # push to the backend + reflect on the radio
        if 'auto_switch' in trk:
            track_ui['auto_switch'] = bool(trk['auto_switch'])
        if 'delay' in trk:
            track_ui['delay'] = max(0.0, float(trk['delay']))
            ui['txt']['track_delay'] = f"{track_ui['delay']:g}"
            _send({'type': 'set_track_delay', 'value': track_ui['delay']})
            ctrl['delay_init'] = True                   # don't let the state-init clobber the loaded value
        if trk.get('model') in ('ema', 'greatcircle') or 'alt_km' in trk:
            track_ui['model'] = trk.get('model', track_ui['model'])
            track_ui['alt_km'] = max(1.0, float(trk.get('alt_km', track_ui['alt_km'])))
            ui['txt']['track_alt'] = f"{track_ui['alt_km']:g}"
            _send({'type': 'set_track_model', 'model': track_ui['model'],
                   'alt_km': track_ui['alt_km']})
        if 'feedforward' in trk:
            track_ui['feedforward'] = bool(trk['feedforward'])
            _send({'type': 'set_track_feedforward', 'on': track_ui['feedforward']})
        if 'position_smoothing' in trk:
            track_ui['position_smoothing'] = max(0.0, float(trk['position_smoothing']))
            ui['txt']['track_position_smoothing'] = f"{track_ui['position_smoothing']:g}"
            _send({'type': 'set_track_position_smoothing',
                   'value': track_ui['position_smoothing']})
        if 'min_horizon' in trk or 'max_horizon' in trk:
            track_ui['min_horizon'] = max(0.1, min(10.0, float(trk.get('min_horizon',
                                                                       track_ui['min_horizon']))))
            track_ui['max_horizon'] = max(1.0, min(60.0, float(trk.get('max_horizon',
                                                                       track_ui['max_horizon']))))
            track_ui['max_horizon'] = max(track_ui['max_horizon'], track_ui['min_horizon'])
            ui['txt']['track_min_horizon'] = f"{track_ui['min_horizon']:g}"
            ui['txt']['track_max_horizon'] = f"{track_ui['max_horizon']:g}"
            _send({'type': 'set_track_horizons', 'min_s': track_ui['min_horizon'],
                   'max_s': track_ui['max_horizon']})
        if any(k in trk for k in ('pid_on', 'pid_kp', 'pid_ki', 'pid_kd', 'pid_integral_limit')):
            track_ui['pid_on'] = bool(trk.get('pid_on', track_ui['pid_on']))
            for k in ('pid_kp', 'pid_ki', 'pid_kd', 'pid_integral_limit'):
                track_ui[k] = float(trk.get(k, track_ui[k]))
                if k == 'pid_integral_limit':
                    track_ui[k] = max(0.0, track_ui[k])
                ui['txt'][f'track_{k}'] = f"{track_ui[k]:g}"
            _send({'type': 'set_track_pid', 'on': track_ui['pid_on'],
                   'kp': track_ui['pid_kp'], 'ki': track_ui['pid_ki'], 'kd': track_ui['pid_kd'],
                   'integral_limit_deg_s': track_ui['pid_integral_limit']})
        if 'follow' in trk:
            _send({'type': 'follow', 'on': bool(trk['follow'])})
        sim = data.get('sim') or {}
        if 'r0' in sim:
            sim_ui['r0'] = max(0.0, float(sim['r0']))
            ui['txt']['sim_r0'] = f"{sim_ui['r0']:g}"
            _send({'type': 'set_sky_render', 'r0_m': sim_ui['r0']})
        if 'bortle' in sim:
            sim_ui['bortle'] = int(max(1, min(9, sim['bortle'])))
            _send({'type': 'set_sky_render', 'bortle': sim_ui['bortle']})
        foc = data.get('focus') or {}                   # collimation-screw calibration + trail exaggeration
        if 'com_mult' in foc:
            focus_ui['com_mult'] = float(foc['com_mult'])
            ui['txt']['focus_com_mult'] = f"{focus_ui['com_mult']:g}"
        if 'shape_gain' in foc:
            focus_ui['shape_gain'] = float(foc['shape_gain'])
            ui['txt']['focus_shape_gain'] = f"{focus_ui['shape_gain']:g}"
        if 'screw_phase' in foc:
            focus_ui['screw_phase'] = float(foc['screw_phase'])
        if 'rad_per_turn' in foc:
            focus_ui['rad_per_turn'] = float(foc['rad_per_turn'])
        if 'alpha' in foc:
            focus_ui['alpha'] = float(foc['alpha'])
            ui['txt']['focus_alpha'] = f"{focus_ui['alpha']:g}"
        for _ax in ('x', 'y'):
            if f'invert_{_ax}' in foc:
                focus_ui[f'invert_{_ax}'] = bool(foc[f'invert_{_ax}'])

    def _settings_refresh(select=None):
        ui['settings_items'] = settings_store.list_settings()
        if select is not None:
            ui['settings_sel'] = select

    # ---- Boresight / Tracking / Simulation setters ----------------------------------------------
    def _bore_send():
        _send({'type': 'set_boresight', 'x_mrad': boresight_ui['x'], 'y_mrad': boresight_ui['y'],
               'roll_mrad': boresight_ui['roll']})


    def _site_set(lat_deg, lon_deg, elev_m, send=True):
        site_ui['lat'], site_ui['lon'] = float(lat_deg), float(lon_deg)
        site_ui['elev'] = float(elev_m)
        ui['txt']['site_lat'] = f"{site_ui['lat']:.8g}"       # 1e-5 deg ~ 1 m: keep full precision
        ui['txt']['site_lon'] = f"{site_ui['lon']:.8g}"
        ui['txt']['site_elev'] = f"{site_ui['elev']:g}"
        if send:
            _send({'type': 'set_site', 'lat_deg': site_ui['lat'], 'lon_deg': site_ui['lon'],
                   'elev_m': site_ui['elev']})

    def _bore_set(x, y, roll, send=True):
        boresight_ui['x'], boresight_ui['y'], boresight_ui['roll'] = float(x), float(y), float(roll)
        ui['txt']['bore_x'] = f"{boresight_ui['x']:.4g}"
        ui['txt']['bore_y'] = f"{boresight_ui['y']:.4g}"
        ui['txt']['bore_roll'] = f"{boresight_ui['roll']:.4g}"
        if send:
            _bore_send()

    def _euler_nudge_widget(wid, ui_state, unit, noun, on_nudge, on_reset, reset_tip):
        """The 'accurately set some euler angles' widget shared by Boresight and Alignment:
        a step-size combo, a 3x3 grid of screen-direction nudges, and a roll pair. Grid labels
        say where `noun` MOVES on screen; on_nudge(dx, dy, droll) receives screen-convention
        deltas (x right, y down, roll clockwise), already scaled by the step."""
        imgui.text("step:")
        imgui.same_line()
        steps = ['0.01', '0.1', '1', '10']
        cur = f"{ui_state['step']:g}"
        sidx = steps.index(cur) if cur in steps else 1
        imgui.set_next_item_width(S(64))
        ch, nidx = imgui.combo(f"##{wid}_step", sidx, steps)
        if ch:
            ui_state['step'] = float(steps[nidx])
        imgui.same_line()
        imgui.text_colored(C4(140, 145, 160), unit)
        _tip("Sensitivity for the nudge buttons below.")
        s_ = ui_state['step']
        for rowdefs in ((('NW', -1, -1), ('N', 0, -1), ('NE', +1, -1)),
                        (('W', -1, 0), ('0', 0, 0), ('E', +1, 0)),
                        (('SW', -1, +1), ('S', 0, +1), ('SE', +1, +1))):
            for i, (lbl, dx, dy) in enumerate(rowdefs):
                if i:
                    imgui.same_line()
                if imgui.button(f"{lbl}##{wid}_grid", (S(30), 0)):
                    if (dx, dy) == (0, 0):
                        on_reset()
                    else:
                        on_nudge(dx * s_, dy * s_, 0.0)
                _tip(reset_tip if (dx, dy) == (0, 0) else
                     f"Move {noun} {lbl} by {s_:g} {unit}.")
        imgui.text("roll:")
        imgui.same_line()
        for lbl, sign in (('  <  ', -1.0), ('  >  ', +1.0)):
            if sign > 0:
                imgui.same_line()
            if imgui.button(f"{lbl}##{wid}_roll{sign}"):
                on_nudge(0.0, 0.0, sign * s_)
            _tip(f"Rotate {noun} {'clockwise' if sign > 0 else 'counterclockwise'} "
                 f"by {s_:g} {unit}.")

    def _set_track_pref(val):
        track_ui['pref'] = val
        _send({'type': 'set_track_pref', 'pref': val})

    _TRACK_PREFS = (('guide', 'Guide'), ('main', 'Main'), ('auto', 'Auto'))

    # ---- Focus tab: start/stop the focus process + tail its metrics into the graphs --------------
    def _focus_reset_series():
        for k in focus_ui['series']:
            focus_ui['series'][k] = []
        focus_ui['t0'] = None
        if focus_ui['fo'] is not None:
            focus_ui['fo'].close()
        focus_ui['fo'] = None

    def _focus_start(role):
        """Point the big pane at this cam's star stream and (re)launch the focus process on it."""
        _focus_reset_series()
        focus_ui['role'], focus_ui['want'] = role, True
        layout['big_stream'] = role + '_focus'            # big pane shows the star; the cams drop to PIPs
        layout['big_role'] = role
        _send({'type': 'focus', 'on': True, 'role': role, 'alpha': focus_ui['alpha']})

    def _focus_stop():
        focus_ui['want'] = False
        if (layout.get('big_stream') or '').endswith('_focus'):
            layout['big_stream'] = None                   # drop the star view (unless it was moved to a PIP)
        _send({'type': 'focus', 'on': False})

    def _set_focus_role(role):
        if role == focus_ui['role'] and focus_ui['want']:
            return
        if focus_ui['want']:
            _focus_start(role)                            # live-switch the running focus to the other cam
        else:
            focus_ui['role'] = role                       # just remember the selection until Start

    # ---- Focus sweep: the GUI plays actuator (see astrolock.seeker.focus_sweep) ------------------
    # The sweep process publishes position REQUESTS on {role}_sweep; we render the prompt and the
    # human's OK publishes a position REPORT on {role}_focuser. A hardware focuser process would
    # answer the same requests -- the sweep can't tell the difference.
    def _sweep_start():
        role = focus_ui['role']
        sweep_ui['role'], sweep_ui['state'], sweep_ui['confirmed'] = role, None, None
        if sweep_ui['fo'] is not None:
            sweep_ui['fo'].close()
        sweep_ui['fo'] = framestream.StreamFollower(args.session, f'{role}_sweep')
        if not focus_ui['want']:
            _focus_start(role)                            # the sweep feeds on the focus stream
        sweep_ui['start'] = _flt(ui['txt']['sweep_start'], sweep_ui['start'])   # parsed only here
        sweep_ui['end'] = _flt(ui['txt']['sweep_end'], sweep_ui['end'])
        sweep_ui['step'] = abs(_flt(ui['txt']['sweep_step'], sweep_ui['step']))
        span = abs(sweep_ui['end'] - sweep_ui['start'])
        step = sweep_ui['step'] or 1.0                     # UI takes a step SIZE; the process wants a count
        steps = max(3, min(201, int(round(span / step)) + 1))
        _send({'type': 'sweep', 'on': True, 'role': role, 'start': sweep_ui['start'],
               'end': sweep_ui['end'], 'steps': steps, 'frames': sweep_ui['frames']})

    def _sweep_abort():
        _send({'type': 'sweep', 'on': False})

    def _sweep_ok(pos):
        """Human actuator: 'the focuser is at pos now'."""
        role = sweep_ui['role'] or focus_ui['role']
        w = sweep_ui['writer']
        if w is None or sweep_ui['writer_role'] != role:
            if w is not None:
                w.close()
            w = sweep_ui['writer'] = framestream.FrameStream(args.session, f'{role}_focuser')
            sweep_ui['writer_role'] = role
        if not w.configured:
            w.configure(1 << 12, 1, pixel_depth=8, frames=256, raw=True)
        payload = json.dumps({'pos': pos}, separators=(',', ':')).encode('utf-8')
        w.write(np.frombuffer(payload, np.uint8), t_mono_ns=session_mod.mono_ns())
        sweep_ui['confirmed'] = pos

    def _poll_sweep():
        """Tail the sweep's state blobs (latest wins) into sweep_ui['state']."""
        fo_ = sweep_ui['fo']
        if fo_ is None:
            return
        fo_.poll()
        got = fo_.latest()                                # latest wins; never lappable
        if got is not None and sweep_ui.get('_seen') != (got[0].ident, got[1]):
            sweep_ui['_seen'] = (got[0].ident, got[1])
            sweep_ui['state'] = json.loads(bytes(got[0].read(got[1])).decode('utf-8'))

    def _poll_focus_metrics():
        """Read the star stream's binary record extras (peak/strehl/com...) into the rolling
        series feeding the on-pane focus graph + collimation trail. Sequential across the
        segment chain; NaN extras decode back to None/absent (unknown aperture/plate scale)."""
        if not focus_ui['want']:
            return
        if focus_ui['fo'] is None:
            focus_ui['fo'] = framestream.StreamFollower(args.session, f"{focus_ui['role']}_focus")
        fo_ = focus_ui['fo']
        fo_.poll()
        s = focus_ui['series']
        for rd, i in fo_.drain():
            try:
                rec = rd.record(i)
            except framestream.Lapped:
                continue
            t = rec['t_mono_ns']
            if focus_ui['t0'] is None:
                focus_ui['t0'] = t * 1e-9
            s['t'].append(round(t * 1e-9 - focus_ui['t0'], 3))
            s['peak'].append(rec['peak'])
            s['peakf'].append(rec.get('peak_frame', rec['peak']))
            h = rec.get('hfd')
            s['hfd'].append(None if h is None or math.isnan(h) else h)
            s['strehl'].append(None if math.isnan(rec['strehl']) else rec['strehl'])
            # Latest shape per half image: (e1, e2, skew_x, skew_y); absent on old recordings.
            if rec.get('ellipse_1') is not None:
                focus_ui['shape_ema'] = (rec['ellipse_1'], rec['ellipse_2'],
                                         rec['skew_x'], rec['skew_y'])
                focus_ui['shape_instant'] = (rec['instant_ellipse_1'], rec['instant_ellipse_2'],
                                             rec['instant_skew_x'], rec['instant_skew_y'])
            srx = rec.get('skew_rad_x')                    # pixel-scale-free skew -> screw dial
            if srx is not None and not math.isnan(srx):
                focus_ui['skew_rad'] = (srx, rec['skew_rad_y'])
            perf['focus'].hit()                            # one record = one focus frame produced
        for k in s:                                       # keep only the last FOCUS_MAX points
            if len(s[k]) > FOCUS_MAX:
                del s[k][:len(s[k]) - FOCUS_MAX]

    # ---- status text ------------------------------------------------------------------------------
    _READOUT = C4(205, 210, 222)                          # one consistent tone for the numeric readouts

    def _perf_text():
        # One spinner PER meter, in place of the colon -- it advances only when that meter gets a real
        # frame, so a stalled source freezes its spinner even though its (stale) fps number stays high.
        def row(meter, label):
            return f"{label:>14} {'|/-\\'[meter.spin % 4]} {meter.rate:5.1f} fps"
        st = ctrl['state'] or {}
        det_roles = st.get('detect_roles') or []
        lines = [row(perf['gui'], 'GUI')]                 # aligned rate block
        for r in roles:
            lines.append(row(perf['cam'][r], r.capitalize() + ' Camera'))
        lines.append(row(perf['mount'], 'Mount'))
        for r in det_roles:
            if r in perf['det']:
                done, skip = perf['det'][r].rate, perf['skip'][r].rate
                pct = 100.0 * skip / (done + skip) if (done + skip) > 0 else 0.0
                ms = perf['ms'].get(f'det:{r}')
                lines.append(row(perf['det'][r], r.capitalize() + ' Detector') + f"  skip {pct:3.0f}%"
                             + (f" {ms:5.1f} ms" if ms is not None else ""))
        if focus_ui['want']:
            lines.append(row(perf['focus'], 'Focus'))
        # Freeform status lines below -- their own thing, so NOT aligned to the fps columns.
        for r in det_roles:
            s = (cams.get(r) or {}).get('status')
            if s:
                lines.append(f"{r.capitalize()} Detector: {s}")
        if st.get('status'):
            lines.append(f"Backend: {st['status']}")
        # Profiler: EMA ms of the hot GUI sections (per prepared frame). 'upload' = raw write + the
        # GPU tonemap pass submit; 'draw' = all panes' ImGui overlay building this frame.
        ms = perf['ms']
        if ms:
            g = ms.get
            lines.append(f"upload {g('upload', 0):4.1f}  draw {g('draw', 0):4.1f} ms")
        return "\n".join(lines)

    # ---- telemetry-driven per-frame state (no widgets touched here) --------------------------------
    def update_control():
        perf['frame'] += 1
        perf['gui'].hit()
        now = time.perf_counter()
        for _m in (perf['gui'], perf['mount'], perf['focus'], *perf['cam'].values(),
                   *perf['det'].values(), *perf['skip'].values()):
            _m.sample(now)

        # Connect to the backend command socket once its port file appears.
        if ctrl['client'] is None:
            bj = _newest(args.session, '_backend.json')
            if bj:
                try:
                    info = json.load(open(bj))
                    ctrl['client'] = control.CommandClient(info['command_host'], info['command_port'])
                    ctrl['state_shm'] = info.get('state_shm')
                except (OSError, ValueError, KeyError):
                    ctrl['client'] = None
        if ctrl.get('slot') is None and ctrl.get('state_shm'):
            try:
                ctrl['slot'] = framestream.LatestSlot(name=ctrl['state_shm'])
            except ValueError:
                ctrl['slot'] = None                       # backend gone; retry after reconnect
                ctrl['state_shm'] = None
        if ctrl.get('slot') is not None:
            got = ctrl['slot'].read()                     # latest-wins, pure memory read
            if got and got[0] != ctrl.get('state_v'):
                perf['mount'].hit(got[0] - (ctrl.get('state_v') or 0))   # meter true update count
                ctrl['state_v'] = got[0]
                ctrl['state'] = got[1]
                tp = got[1].get('enc_t_mono_ns')          # pose history: lets the sky overlay use
                if tp:                                    # the pose AT the displayed frame's time
                    pose_hist.push(int(tp) * 1e-9,
                                   math.radians(float(got[1].get('enc_az_deg', 0.0))),
                                   math.radians(float(got[1].get('enc_alt_deg', 0.0))),
                                   math.radians(float(got[1].get('rate_az_deg_s', 0.0))),
                                   math.radians(float(got[1].get('rate_alt_deg_s', 0.0))))
        st = ctrl['state']
        _poll_focus_metrics()                           # tail focus metrics into the graphs
        _poll_sweep()                                   # tail sweep prompts/result (if one ran)
        _poll_navigation()                              # tail the sparse sky feed (overlay)
        # Esc drops the current lock -- same as the Unlock button, but reachable on a laptop
        # in the dark without aiming at a pane. Not while a text field owns the keyboard
        # (there Esc means 'cancel this edit').
        if ((st or {}).get('tracking') and imgui.is_key_pressed(imgui.Key.escape)
                and not imgui.get_io().want_text_input):
            _send({'type': 'untrack'})
        # Auto-switch the main pane to follow the active tracking source (as if its ^ button were pressed):
        # only when the *previously* active cam is the one currently in the big pane, so manual choices hold.
        tr = (st or {}).get('track_role')
        prev = ctrl.get('prev_track_role')
        if (tr and prev and tr != prev and track_ui['auto_switch']
                and layout['big_role'] == prev and tr in roles):
            layout['big_role'] = tr
        elif prev and not tr and track_ui['auto_switch'] and 'guide' in roles:
            layout['big_role'] = 'guide'    # lock ended: back to the wide view to pick a new target
        ctrl['prev_track_role'] = tr

        src_init = ctrl.setdefault('src_init', set())
        for role in roles:
            if st is None:
                continue
            # One-time init of the unified source dropdown from the backend's actual (source, camera).
            uval = _source_value(st, role)
            if role not in src_init and uval:
                ui['src'][role] = uval
                src_init.add(role)
            # Playback loop checkbox: one-time init from state.
            pb = ((st or {}).get('playback') or {}).get(role) or {}
            if role not in ctrl.setdefault('pbloop_init', set()) and pb:
                ui['pb_loop'][role] = bool(pb.get('loop', True))
                ctrl['pbloop_init'].add(role)
            # Recording has NO logic here: policy (manual / record-while-tracking) lives in
            # the backend, which reconciles it against reality every tick. The checkboxes in
            # _panel_camera render the backend's flags and send plain toggles.

        # One-time init of the Optics tab dropdowns from the backend's current selection.
        opt_sel = (st or {}).get('optics_sel') or {}
        opt_init = ctrl.setdefault('opt_init', set())
        for role in roles:
            if role in opt_init or role not in opt_sel:
                continue
            sen, opt_, red = (list(opt_sel[role]) + [None, None, None])[:3]
            ui['opt'][role] = {'sensor': sen or '', 'optic': opt_ or '', 'reducer': red or '(none)'}
            opt_init.add(role)

        # Mount chooser: init the selection once from the backend.
        mount_items = ['sim'] + [u for u in ((st or {}).get('mounts_available') or []) if u != 'sim']
        if st is not None and 'mount_init' not in ctrl and st.get('mount_url'):
            ui['mount_sel'] = st['mount_url'] if st['mount_url'] in mount_items else 'sim'
            ctrl['mount_init'] = True

        # One-time init of the track-delay field from the backend's value (settings load may override).
        if 'delay_init' not in ctrl and st and st.get('track_delay_s') is not None:
            track_ui['delay'] = float(st['track_delay_s'])
            ui['txt']['track_delay'] = f"{track_ui['delay']:g}"
            ctrl['delay_init'] = True

        # One-time init of the boresight editor from the backend's value (settings load may override).
        if 'bore_init' not in ctrl and st and st.get('boresight_mrad') is not None:
            bx, by, br = (list(st['boresight_mrad']) + [0.0, 0.0, 0.0])[:3]
            _bore_set(bx, by, br, send=False)          # reflect the backend's value; don't echo it back
            ctrl['bore_init'] = True

        if 'site_init' not in ctrl and st and st.get('site') is not None:
            s = st['site']
            _site_set(s.get('lat_deg', 0.0), s.get('lon_deg', 0.0), s.get('elev_m', 0.0),
                      send=False)                          # reflect the backend; don't echo it back
            ctrl['site_init'] = True

        # Restored layout had Dbg on but the detectors aren't writing debug streams: sync once.
        if 'dbg_init' not in ctrl and st is not None:
            ctrl['dbg_init'] = True
            if layout.get('pip_debug') and not st.get('debug_ser'):
                _send({'type': 'set_debug_ser', 'on': True})

        # Caps-driven camera controls: re-seed values whenever the *set* of controls changes.
        caps_st = (st or {}).get('camera_caps') or {}
        ctrl_sig = ctrl.setdefault('ctrl_sig', {})
        for role in roles:
            caps = caps_st.get(role)
            sig = (caps.get('source'), tuple(c['name'] for c in caps.get('controls', []))) if caps else None
            if sig != ctrl_sig.get(role):
                _seed_cam_controls(role, caps)
                ctrl_sig[role] = sig

        # An async native file picker (playback .ser) that came back: push the choice.
        if ui['pb_dlg'] is not None and ui['pb_dlg'][1].ready():
            role, dlg = ui['pb_dlg']
            res = dlg.result()
            if res:
                _send({'type': 'set_playback', 'role': role, 'ser': res[0]})
            ui['pb_dlg'] = None

    # ---- the right settings/telemetry panel (immediate mode, rebuilt every frame) -------------------
    def _mono_text(txt):
        imgui.push_font(mono_font, 13.0)
        imgui.text_unformatted(txt)
        imgui.pop_font()

    def _input_commit(tid, width, commit):
        """A 'type then Enter' text field: the buffer lives in ui['txt']; on Enter, commit(text) parses,
        applies, and returns the canonical string to show."""
        buf = ui['txt'].setdefault(tid, '')
        imgui.set_next_item_width(width)
        ch, buf = imgui.input_text('##' + tid, buf, ENTER)
        ui['txt'][tid] = buf
        if ch:
            ui['txt'][tid] = commit(buf)

    def _flt(text, fallback):
        try:
            return float(text)
        except (ValueError, TypeError):
            return fallback

    def _ctext(col4, txt):
        """Coloured text, printf-safe (imgui.text_colored treats its string as a format -- a stray
        '%' in a filename would garble it)."""
        imgui.push_style_color(imgui.Col_.text, col4)
        imgui.text_unformatted(txt)
        imgui.pop_style_color()

    def _grey(txt):
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(C4(120, 125, 140), txt)
        imgui.pop_text_wrap_pos()

    def _panel_camera(role):
        """One camera's connection/capture/display settings (immediate mode)."""
        st = ctrl['state']
        sset = view_settings.setdefault(role, _default_settings(role))
        imgui.text("camera:")
        _tip("Where this pane's frames come from: a detected ZWO camera (by model), 'sky' (ISS "
             "simulator), or 'playback' (replay a .ser). Press Rescan after plugging a camera in.")
        imgui.same_line()
        items = _source_items(st)
        cur = ui['src'].get(role)
        idx = items.index(cur) if cur in items else -1
        imgui.set_next_item_width(-1)
        ch, nidx = imgui.combo(f"##src_{role}", idx, items)
        if ch and 0 <= nidx < len(items):
            ui['src'][role] = items[nidx]
            _on_source_pick(role, items[nidx])
        # Connect/Disconnect sits right under the dropdown it applies -- picking a source only sets the
        # desired; Connect is what actually starts it on that source.
        cap_on = bool(((st or {}).get('capturing') or {}).get(role))
        if imgui.button(("Disconnect" if cap_on else "Connect") + f"##conn_{role}"):
            _toggle_connect(role)
        _tip("Start/stop this camera's capture on the source picked above. Nothing connects until you "
             "press this -- so two roles don't both grab '(auto)' and wedge the USB bus.")
        # Playback source: a .ser to replay + a loop toggle; shown only when source == playback.
        if ui['src'].get(role) == 'playback':
            if imgui.button(f"File...##pb_{role}"):
                ui['pb_dlg'] = (role, pfd.open_file("Choose a .ser to replay", os.getcwd(),
                                                    ["SER video", "*.ser", "All files", "*"]))
            _tip("Choose a recorded .ser to replay through the pipeline.")
            imgui.same_line()
            pb = ((st or {}).get('playback') or {}).get(role) or {}
            _ctext(C4(150, 155, 170), os.path.basename(pb['ser']) if pb.get('ser') else "(none)")
            ch, v = imgui.checkbox(f"Loop##pb_{role}", ui['pb_loop'][role])
            if ch:
                ui['pb_loop'][role] = v
                _send({'type': 'set_playback', 'role': role, 'loop': v})
            _tip("Loop the recording instead of stopping at the end.")
        _panel_cam_controls(role, ((st or {}).get('camera_caps') or {}).get(role))
        # Recording checkboxes render the BACKEND'S policy flags and send plain toggles; the
        # backend computes desire (manual or auto-and-tracking) and keeps a recorder process
        # matching it every tick -- so a dead recorder is its problem to fix, never ours to
        # have silently latched over (the missed-ISS-pass bug).
        ch, v = imgui.checkbox(f"Record now##rec_{role}",
                               bool(((st or {}).get('record_manual') or {}).get(role)))
        if ch:
            _send({'type': 'record', 'role': role, 'manual': v})
        _tip("Archive this camera's frames: a recorder process tails the shared-memory ring and "
             "writes every frame to a .ser in the recordings dir at drive pace (the camera never "
             "touches the disk). Uncheck to finalize the file.")
        ch, v = imgui.checkbox(f"Auto record##autorec_{role}",
                               bool(((st or {}).get('record_auto') or {}).get(role)))
        if ch:
            _send({'type': 'record', 'role': role, 'auto': v})
        _tip("Record this camera automatically whenever tracking is engaged (same recorder as "
             "Record now; the file finalizes when the lock drops). Turns itself on when you "
             "switch this role to a real camera.")
        ch, v = imgui.checkbox(f"Reticles##ret_{role}", sset['reticles'])
        if ch:
            sset['reticles'] = v
        _tip("Show the centre crosshairs + the main-cam FoV box on this camera's pane.")
        ch, v = imgui.checkbox(f"Histogram##hist_{role}", sset['histogram'])
        if ch:
            sset['histogram'] = v
        _tip("Show a luminance histogram inset on this camera's pane (judge exposure/clipping).")
        ch, v = imgui.checkbox(f"Show stars##stars_{role}", sset.get('show_stars', True))
        if ch:
            sset['show_stars'] = v
        _tip("Overlay the sky model on this pane: five-pointed stars for everything it places "
             "(stars, Sun/Moon/planets, the satellite + its pass line), at the encoder pose "
             "corrected by the Alignment matrix.")
        ch, v = imgui.checkbox(f"Show target names##tnames_{role}",
                               sset.get('show_target_names', True))
        if ch:
            sset['show_target_names'] = v
        _tip("Label the overlay's non-star objects -- the satellite, Sun, Moon, planets.")
        ch, v = imgui.checkbox(f"Show star names##snames_{role}",
                               sset.get('show_star_names', False))
        if ch:
            sset['show_star_names'] = v
        _tip("Label the overlay stars with their proper names (Vega, ...; HIP ids when a star "
             "has no name).")
        ch, v = imgui.checkbox(f"Wait for detector##waitdet_{role}", sset.get('wait_for_detector', True))
        if ch:
            sset['wait_for_detector'] = v
        _tip("On (default): show each frame only once the detector has processed it, so the detection "
             "boxes sit exactly on the pixels. Off: show the newest frame immediately (lower latency), "
             "and let the boxes lag a frame or two behind.")

    def _panel():
        st = ctrl['state'] or {}
        if imgui.tree_node_ex("Status", OPEN):
            _mono_text(_perf_text())
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Mount", OPEN):
            if imgui.button("Rescan Mounts"):
                _send({'type': 'rescan_mounts'})
            _tip("Re-enumerate mounts (detected Celestron COM ports) after plugging one in.")
            imgui.text("Mount:")
            _tip("Which mount to drive: 'sim', or a detected Celestron on a COM port.")
            imgui.same_line()
            mount_items = ['sim'] + [u for u in (st.get('mounts_available') or []) if u != 'sim']
            midx = mount_items.index(ui['mount_sel']) if ui['mount_sel'] in mount_items else 0
            imgui.set_next_item_width(-1)
            ch, nidx = imgui.combo("##mount_combo", midx, mount_items)
            if ch and 0 <= nidx < len(mount_items):
                ui['mount_sel'] = mount_items[nidx]
                _send({'type': 'set_mount', 'url': mount_items[nidx]})
            if imgui.button("Disconnect from Mount" if st.get('mount_connected')
                            else "Connect to Mount"):
                _send({'type': 'mount_connect', 'on': not bool(st.get('mount_connected'))})
            _tip("Connect to / disconnect from the selected mount. Disconnected = the backend holds the "
                 "last pose and nothing moves.")
            if st:
                _mono_text(f"Az: {st.get('enc_az_deg', 0.0):7.3f}  Alt: {st.get('enc_alt_deg', 0.0):7.3f} deg\n"
                           f"    {st.get('rate_az_deg_s', 0.0):+7.3f}       {st.get('rate_alt_deg_s', 0.0):+7.3f} deg/s")
            else:
                _mono_text("Az:     ---  Alt:     --- deg\n        ---          --- deg/s")
            # 2D slew pad: a log-scaled az/alt rate plane. Drag = drive the mount (momentary override
            # of tracking); the green circle shows the current rate from telemetry.
            imgui.text_colored(C4(160, 170, 190), "Slew")
            _tip(f"Drag the pad to drive the mount (log scale, max {SLEW_MAX:g} deg/s; Az = right, "
                 f"Alt = up). Centre = stop. Momentarily overrides tracking, resumes on release.")
            _P = S(200)
            imgui.invisible_button("##slew_pad", (_P, _P))
            rmin = imgui.get_item_rect_min()
            dl = imgui.get_window_draw_list()
            _c, _H = _P / 2.0, _P / 2.0

            def PA(x, y):
                return (rmin.x + x, rmin.y + y)
            dl.add_rect_filled(PA(1, 1), PA(_P - 1, _P - 1), C((18, 20, 26, 220)))
            dl.add_rect(PA(1, 1), PA(_P - 1, _P - 1), C((80, 86, 100, 220)), 0.0, 1.0)
            for _g in SLEW_GRID:
                _d = _rate_to_u(_g) * _H
                for _sgn in (-1, 1):
                    dl.add_line(PA(_c + _sgn * _d, 2), PA(_c + _sgn * _d, _P - 2), C((80, 86, 100, 110)), 1.0)
                    dl.add_line(PA(2, _c + _sgn * _d), PA(_P - 2, _c + _sgn * _d), C((80, 86, 100, 110)), 1.0)
            dl.add_line(PA(_c, 2), PA(_c, _P - 2), C((150, 156, 172, 220)), 1.5)  # az = 0
            dl.add_line(PA(2, _c), PA(_P - 2, _c), C((150, 156, 172, 220)), 1.5)  # alt = 0
            if imgui.is_item_active():                     # drag latches while held, even outside the pad
                if not ctrl.get('slew_active'):            # drag start -> remember a track to resume on release
                    ctrl['slew_active'] = True
                    ctrl['slew_resume'] = (st.get('track_role'), st.get('target_px')) \
                        if st.get('tracking') else None
                az = _u_to_rate((io.mouse_pos.x - (rmin.x + _c)) / max(1.0, _H))
                alt = _u_to_rate(-(io.mouse_pos.y - (rmin.y + _c)) / max(1.0, _H))   # screen y down -> alt up
                if ctrl['client'] is not None and (az, alt) != ctrl['last_rate']:
                    ctrl['client'].send({'type': 'set_rate', 'az': az, 'alt': alt})
                    ctrl['last_rate'] = (az, alt)
            elif ctrl.get('slew_active'):                  # released -> stop, then resume any prior track
                ctrl['slew_active'] = False
                if ctrl['client'] is not None:
                    ctrl['client'].send({'type': 'stop'})
                    ctrl['last_rate'] = (0.0, 0.0)
                    res = ctrl.get('slew_resume')
                    if res and res[0] and res[1]:
                        ctrl['client'].send({'type': 'track', 'role': res[0],
                                             'px': [float(res[1][0]), float(res[1][1])]})
                ctrl['slew_resume'] = None
            # Current-rate circle on the pad, from telemetry.
            if st:
                cxp = _c + _rate_to_u(st.get('rate_az_deg_s', 0.0)) * _H
                cyp = _c - _rate_to_u(st.get('rate_alt_deg_s', 0.0)) * _H
                dl.add_circle(PA(cxp, cyp), S(6), C((70, 230, 100, 235)), 0, 2.0)
            if imgui.button("Stop Moving", (S(200), 0)):
                _send({'type': 'follow', 'on': False})
            _tip("Halt mount motion now. Keeps the lock -- the tracker keeps following the target in "
                 "software; only Unlock (Tracking panel) drops the lock.")
            if imgui.button("Resume Following", (S(200), 0)):
                _send({'type': 'follow', 'on': True})
            _tip("Resume slewing the mount to hold the locked target (undo Stop Moving).")
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Time", OPEN):
            off_ns = st.get('utc_offset_ns')
            if off_ns is None:
                _ctext(C4(150, 155, 170), "(waiting for backend)")
            else:
                utc_ns = off_ns + session_mod.mono_ns()    # ticks every frame; jumps on skips
                dt_utc = datetime.datetime.fromtimestamp(utc_ns * 1e-9, tz=datetime.timezone.utc)
                _mono_text(f"UTC    {dt_utc.strftime('%Y-%m-%d %H:%M:%S')}Z\n"
                           f"Local  {dt_utc.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                           f"({_fmt_time_delta(utc_ns - time.time_ns())})")
                _tip("The backend's current time: what the sky model (star overlay, satellite "
                     "pass) is computed for. All processes share one monotonic clock; this is "
                     "its offset to UTC, owned by the backend.")
            if imgui.button("Set from System Time"):
                _send({'type': 'set_time', 'source': 'system'})
            _tip("Set the current time from this computer's clock.")
            imgui.same_line()
            if imgui.button("Set from Mount GPS##time_gps"):
                _send({'type': 'set_time', 'source': 'gps'})
            _tip("Ask the mount's GPS for the time (Celestron: read through the hand controller; "
                 "sim mount: its example-pass clock). Applies when the read completes.")
            for i, (label, secs) in enumerate((('<d', -86_400), ('<h', -3_600), ('<m', -60),
                                               ('<<s', -10), ('<s', -1), ('s>', 1), ('s>>', 10),
                                               ('m>', 60), ('h>', 3_600), ('d>', 86_400))):
                if i:
                    imgui.same_line()
                if imgui.button(f"{label}##time_skip"):
                    _send({'type': 'set_time', 'adjust_ns': secs * 1_000_000_000})
                unit = {86_400: 'one day', 3_600: 'one hour', 60: 'one minute',
                        10: '10 seconds', 1: 'one second'}[abs(secs)]
                _tip(f"Skip the current time {'back' if secs < 0 else 'forward'} {unit}.")
            if st.get('gps_status'):
                _ctext(C4(150, 155, 170), st['gps_status'])
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Location", OPEN):
            for label, key, field in (("Latitude:", 'site_lat', 'lat'),
                                      ("Longitude:", 'site_lon', 'lon'),
                                      ("Elevation:", 'site_elev', 'elev')):
                imgui.text(label)
                imgui.same_line(S(76))

                def _commit_site(txt, _f=field):
                    v = dict(site_ui)
                    v[_f] = _flt(txt, site_ui[_f])
                    _site_set(v['lat'], v['lon'], v['elev'])
                    return ui['txt'][key]
                _input_commit(key, S(100), _commit_site)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "m" if field == 'elev' else "deg")
            _tip("Observer site, decimal degrees (north / east positive) + metres.")
            if imgui.button("Set from Mount GPS##site_gps"):
                _send({'type': 'set_site', 'source': 'gps'})
            _tip("Ask the mount's GPS for the site (Celestron: read through the hand controller; "
                 "sim mount: its example-pass site). Applies when the read completes.")
            if st.get('gps_status'):
                _ctext(C4(150, 155, 170), st['gps_status'])
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Simulation", OPEN):         # sim connect + sim-truth render knobs
            if imgui.button("Set Time and Location to Example Pass"):
                _send({'type': 'set_example_pass'})
            _tip("Jump the Time and Location tabs to the launch-configured example pass "
                 "(default: the Friday 2026-07-24 82-deg ISS pass over San Carlos, just rising) -- "
                 "the sky model re-propagates there.")
            if imgui.button("Connect Sim Mount and Cameras"):
                _send({'type': 'connect_sim_cams'})
            _tip("Connect the sim mount (if no mount is connected) and sim cameras wherever "
                 "nothing else is already connected. Does not touch time or location.")
            imgui.text("Seeing r0:")
            _tip("Atmospheric seeing as the Fried parameter r0 (m): bigger = steadier air. 0 = off; "
                 "~0.05 poor, ~0.2 excellent. One sky, so it blurs all sim cameras (FWHM ~ 0.98*lambda/r0). "
                 "Per-camera lens softness is the Defocus knob in each camera's settings.")
            imgui.same_line()

            def _commit_r0(txt):
                sim_ui['r0'] = max(0.0, _flt(txt, sim_ui['r0']))
                _send({'type': 'set_sky_render', 'r0_m': sim_ui['r0']})
                return f"{sim_ui['r0']:g}"
            _input_commit('sim_r0', S(56), _commit_r0)
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "m")
            imgui.text("Bortle:")
            _tip("Bortle dark-sky class 1..9 (sky brightness): 1 = pristine, 9 = inner city. Sets the sim "
                 "sky background per pixel (needs a physical sensor -- QE / full-well in the optics DB). "
                 "Darken the sky so a bright star's overexposed diffraction spikes clear the noise.")
            imgui.same_line()
            imgui.set_next_item_width(S(80))
            ch, v = imgui.input_int("##sim_bortle", sim_ui['bortle'], 1, 1, ENTER)
            if ch:
                sim_ui['bortle'] = int(max(1, min(9, v)))
                _send({'type': 'set_sky_render', 'bortle': sim_ui['bortle']})
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Cameras", OPEN):
            if imgui.button("Rescan Cameras"):
                _send({'type': 'rescan_cameras'})
            _tip("Re-enumerate attached ZWO cameras (after plugging one in).")
            for role in roles:
                if role != roles[0]:
                    imgui.separator()              # keep the two cameras' settings visually apart
                if imgui.tree_node_ex(f"{role.capitalize()} Camera", OPEN):
                    _panel_camera(role)
                    imgui.tree_pop()
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Optics"):
            opt_fov = st.get('optics') or {}
            for role in roles:
                if role != roles[0]:
                    imgui.separator()              # same rhythm as the Cameras section
                if not imgui.tree_node_ex(f"{role.capitalize()} Optics", OPEN):
                    continue
                for kind, has_owned in (('sensor', True), ('optic', True), ('reducer', False)):
                    imgui.text(f"{kind}:")
                    _tip(f"{role.capitalize()} {kind}. Owned gear is pinned to the top of the list.")
                    imgui.same_line()
                    items = _gear_items(kind)
                    cur = ui['opt'][role][kind]
                    idx = items.index(cur) if cur in items else -1
                    imgui.set_next_item_width(-S(30) if has_owned else -1)
                    ch, nidx = imgui.combo(f"##opt_{role}_{kind}", idx, items)
                    if ch and 0 <= nidx < len(items) and items[nidx] != _DIV:
                        ui['opt'][role][kind] = items[nidx]
                        _send_optics(role)
                    if has_owned:
                        imgui.same_line()
                        own_now = cur in owned[kind]
                        ch, v = imgui.checkbox(f"##own_{role}_{kind}", own_now)
                        if ch and cur and cur != _DIV:
                            (owned[kind].add if v else owned[kind].discard)(cur)
                        _tip("I own this — pin it to the top of the list (in every dropdown).")
                # Derived numbers, below the pickers: FoV from the backend (it knows the live
                # render geometry) + plate scale computed from the picked gear.
                fv = opt_fov.get(role)
                if fv:
                    imgui.text_colored(C4(150, 155, 170),
                                       f"FoV {fv['fov_x_deg']:.2f} × {fv['fov_y_deg']:.2f}°")
                sel = ui['opt'][role]
                if sel.get('sensor') in _SENS and sel.get('optic') in _OPT:
                    cfg = optics_db.configuration(_SENS[sel['sensor']], _OPT[sel['optic']],
                                                  _RED.get(sel.get('reducer'), 1.0))
                    imgui.text_colored(C4(150, 155, 170),
                                       f"{cfg['arcsec_per_px']:.2f} arcsec/px @ "
                                       f"f = {cfg['effective_focal_mm']:.0f} mm")
                    _tip("Plate scale at 1×1 binning (sensor pixel pitch / effective focal length); "
                         "hardware binning multiplies arcsec/px by the bin factor.")
                imgui.tree_pop()
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Boresight"):
            imgui.text("X:")
            imgui.same_line()

            def _commit_bx(txt):
                _bore_set(_flt(txt, boresight_ui['x']), boresight_ui['y'], boresight_ui['roll'])
                return ui['txt']['bore_x']
            _input_commit('bore_x', S(56), _commit_bx)
            imgui.same_line()
            imgui.text("Y:")
            imgui.same_line()

            def _commit_by(txt):
                _bore_set(boresight_ui['x'], _flt(txt, boresight_ui['y']), boresight_ui['roll'])
                return ui['txt']['bore_y']
            _input_commit('bore_y', S(56), _commit_by)
            imgui.same_line()
            imgui.text("Roll:")
            imgui.same_line()

            def _commit_broll(txt):
                _bore_set(boresight_ui['x'], boresight_ui['y'], _flt(txt, boresight_ui['roll']))
                return ui['txt']['bore_roll']
            _input_commit('bore_roll', S(56), _commit_broll)
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "mrad")
            _euler_nudge_widget(
                'bore', boresight_ui, 'mrad', "the main-cam FoV marker",
                on_nudge=lambda dx, dy, dr: _bore_set(boresight_ui['x'] + dx,
                                                      boresight_ui['y'] + dy,
                                                      boresight_ui['roll'] + dr),
                on_reset=lambda: _bore_set(0.0, 0.0, 0.0),
                reset_tip="Reset the boresight (X, Y, and Roll) to 0.")
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Alignment"):
            _tip("Manual mount alignment: one rotation matrix correcting the encoder-derived "
                 "pointing (azimuth zero, tripod tilt, roll). Nudge until the gold star overlay "
                 "sits on the real stars, then the satellite pass line is where the satellite "
                 "will actually fly. (No plate solving yet -- this is the manual version.)")
            # Euler readout/editor of the matrix (yaw about sky up, pitch about east, roll
            # about north -- geo.matrix_from_euler's convention). Refreshed from the backend
            # whenever the matrix changes (nudges included); committing a value re-composes
            # the matrix and sends it whole.
            a9 = tuple((st or {}).get('align_matrix') or ())
            if len(a9) == 9 and a9 != ctrl.get('align_euler_cache'):
                ctrl['align_euler_cache'] = a9
                yaw_rad, pitch_rad, roll_rad = geo.euler_from_matrix(
                    torch.tensor([a9[0:3], a9[3:6], a9[6:9]], dtype=torch.float64))
                ui['txt']['align_yaw'] = f"{math.degrees(yaw_rad):.4g}"
                ui['txt']['align_pitch'] = f"{math.degrees(pitch_rad):.4g}"
                ui['txt']['align_roll'] = f"{math.degrees(roll_rad):.4g}"

            def _commit_align_euler(txt, key):
                vals = [_flt(txt if k == key else ui['txt'].get(k, '0'), 0.0)
                        for k in ('align_yaw', 'align_pitch', 'align_roll')]
                matrix = geo.matrix_from_euler(*(math.radians(v) for v in vals))
                _send({'type': 'set_alignment',
                       'matrix': [float(v) for v in matrix.reshape(-1)]})
                return f"{_flt(txt, 0.0):.4g}"

            for label, key in (('Yaw:', 'align_yaw'), ('Pitch:', 'align_pitch'),
                               ('Roll:', 'align_roll')):
                if key != 'align_yaw':
                    imgui.same_line()
                imgui.text(label)
                imgui.same_line()
                _input_commit(key, S(56), lambda t, k=key: _commit_align_euler(t, k))
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "deg")
            # The nudges are camera-local rotations evaluated on the modeled orientation
            # (screen deltas: overlay up = camera pitched down about its side axis, overlay
            # right = camera yawed left about its up axis, roll clockwise = +roll about the
            # boresight), so a button's on-screen effect is the same no matter how much
            # alignment is already dialed in.
            _euler_nudge_widget(
                'align', align_ui, 'deg', "the star overlay",
                on_nudge=lambda dx, dy, dr: _send({'type': 'align_nudge', 'pitch_deg': -dy,
                                                   'yaw_deg': -dx, 'roll_deg': dr}),
                on_reset=lambda: _send({'type': 'align_nudge', 'reset': True}),
                reset_tip="Reset the alignment matrix to identity.")
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Focus"):          # focus + collimation assist (see astrolock.seeker.focus)
            cap = st.get('capturing') or {}
            imgui.text("Camera:")
            _tip("Which camera to focus/collimate. Switching while running moves the helper to that cam. "
                 "A camera that isn't connected is greyed out.")
            for r in roles:
                imgui.same_line()
                sel = focus_ui['role'] == r
                imgui.begin_disabled(not bool(cap.get(r)))
                if imgui.button(('* ' if sel else '  ') + r.capitalize() + f"##focus_role_{r}", (S(56), 0)):
                    _set_focus_role(r)
                imgui.end_disabled()
            if imgui.button('Stop Focus' if focus_ui['want'] else 'Start Focus'):
                _focus_stop() if focus_ui['want'] else _focus_start(focus_ui['role'])
            _tip("Start/stop the focus helper. It locks the brightest star, averages it (EMA), and shows "
                 "that lucky star in the main pane; the collimation trail is drawn on it and the graph "
                 "below tracks focus quality. Slowly sweep focus and maximize the peaks. Saturated cores "
                 "FLASH (their peak reading can't be trusted -- reduce exposure/gain).")
            imgui.text("Smoothing α:")
            _tip("Star-crop EMA rate (0..1): lower = a steadier averaged star (rides out seeing + tracking "
                 "jitter) but slower to settle; higher = snappier but noisier. Changing it restarts the "
                 "focus helper, so the average resets.")
            imgui.same_line()

            def _commit_alpha(txt):
                focus_ui['alpha'] = max(0.001, min(1.0, _flt(txt, focus_ui['alpha'])))
                if focus_ui['want']:                       # relaunch the running focus with the new smoothing
                    _send({'type': 'focus', 'on': True, 'role': focus_ui['role'], 'alpha': focus_ui['alpha']})
                return f"{focus_ui['alpha']:g}"
            _input_commit('focus_alpha', S(48), _commit_alpha)
            imgui.text("Skew ×:")
            _tip("Exaggeration for the skew (coma) arrows drawn on the star view -- higher magnifies "
                 "smaller miscollimation. The arrow points from each crosshair to where the halo's "
                 "light sits relative to the star's centroid.")
            imgui.same_line()

            def _commit_cm(txt):
                focus_ui['com_mult'] = max(1.0, _flt(txt, focus_ui['com_mult']))
                return f"{focus_ui['com_mult']:g}"
            _input_commit('focus_com_mult', S(48), _commit_cm)
            imgui.text("Shape ×:")
            _tip("Exaggeration for the astigmatism ellipses drawn on the star view: the drawn "
                 "ellipticity is this many times the measured one (orientation exact, mean radius "
                 "pinned to the crosshair gap). 1 = true shape.")
            imgui.same_line()

            def _commit_shape_gain(txt):
                focus_ui['shape_gain'] = max(1.0, _flt(txt, focus_ui['shape_gain']))
                return f"{focus_ui['shape_gain']:g}"
            _input_commit('focus_shape_gain', S(48), _commit_shape_gain)
            imgui.text("Screw phase:")               # SCT collimation-screw guide (dial on the star pane)
            _tip("Rotational orientation of the 3 secondary collimation screws in the camera image -- dial "
                 "it so the numbered screws on the star pane match how you physically see them on the "
                 "corrector plate.")
            imgui.same_line()
            imgui.set_next_item_width(S(56))
            ch, v = imgui.input_float("##focus_screw_phase", focus_ui['screw_phase'], 0.0, 0.0, "%.0f", ENTER)
            if ch:
                focus_ui['screw_phase'] = float(v)
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "deg")
            imgui.text("Turn sensitivity:")
            _tip("Empirical: collimation error (skew, in microradians -- pixel-scale-free, so it's the "
                 "same across cameras) removed by one full screw turn. Calibrate once: note how far a screw "
                 "turn moves the skew. Make it negative if the turn arrows point the wrong way.")
            imgui.same_line()
            imgui.set_next_item_width(S(56))
            ch, v = imgui.input_float("##focus_screw_sens", focus_ui['rad_per_turn'] * 1e6, 0.0, 0.0, "%.0f", ENTER)
            if ch and abs(v) > 1e-6:                       # entered in urad/turn (signed -> flip)
                focus_ui['rad_per_turn'] = v * 1e-6
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "µrad/turn")
            imgui.text("Invert screws:")
            _tip("Flip the screw-turn direction per axis to match your camera's image parity (mirror/flip); "
                 "set these empirically alongside the sensitivity. A future optics-DB collimator type will "
                 "preset them correctly.")
            imgui.same_line()
            ch, v = imgui.checkbox("X##focus_inv_x", focus_ui['invert_x'])
            if ch:
                focus_ui['invert_x'] = v
            imgui.same_line()
            ch, v = imgui.checkbox("Y##focus_inv_y", focus_ui['invert_y'])
            if ch:
                focus_ui['invert_y'] = v
            imgui.separator()
            # --- Focus sweep: walk a focuser range, fit the HFD V-curve, report best focus.
            imgui.text("Sweep:")
            _tip("Focus sweep with YOU as the actuator: give it a focuser range (any units -- knob "
                 "marks, mm, motor steps) and it prompts for each position; press OK once the "
                 "focuser is there. Every unsaturated frame's peak brightness is least-squares "
                 "fit (1/peak is quadratic in focuser position); the vertex is best focus. Keep "
                 "the star UNSATURATED (50-80% full well) -- clipped frames are excluded. "
                 "An electronic focuser will later answer the same prompts unattended.")
            # Plain text boxes: nothing reads them until Start Sweep, which parses them then.
            imgui.same_line()
            imgui.set_next_item_width(S(56))
            _ch, ui['txt']['sweep_start'] = imgui.input_text('##sweep_start', ui['txt']['sweep_start'])
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "to")
            imgui.same_line()
            imgui.set_next_item_width(S(56))
            _ch, ui['txt']['sweep_end'] = imgui.input_text('##sweep_end', ui['txt']['sweep_end'])
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "step")
            imgui.same_line()
            imgui.set_next_item_width(S(56))
            _ch, ui['txt']['sweep_step'] = imgui.input_text('##sweep_step', ui['txt']['sweep_step'])
            sw_running = bool((st.get('sweep') or {}).get('running'))
            if imgui.button('Abort Sweep' if sw_running else 'Start Sweep', (S(200), 0)):
                _sweep_abort() if sw_running else _sweep_start()
            stt = sweep_ui['state'] or {}
            if sw_running and stt and not stt.get('done'):
                # Fixed three-line layout -- the OK button is ALWAYS present (disabled while
                # collecting) so nothing reflows as the sweep advances.
                awaiting = stt.get('awaiting') == 'position'
                conf = sweep_ui.get('confirmed')
                imgui.text(f"Current focus: {'--' if conf is None else f'{conf:g}'}")
                imgui.text(f"Commanded focus: {stt.get('want_pos', 0):g}")
                imgui.same_line()
                imgui.begin_disabled(not awaiting)
                if imgui.button("OK##sweep_ok", (S(64), 0)):
                    _sweep_ok(stt.get('want_pos', 0.0))
                imgui.end_disabled()
                imgui.text(f"Step {stt.get('step', '?')}/{stt.get('of', '?')}"
                           + ("" if awaiting else
                              f" -- collecting {stt.get('collected', 0)}/{stt.get('need', 0)}"))
            elif stt.get('done'):
                if stt.get('aborted'):
                    _grey("Sweep aborted.")
                elif 'p0' in stt:
                    imgui.text(f"Best focus: {stt['p0']:g}  (peak {stt.get('peak0', 0):.2f})")
                    if not stt.get('bracketed', True):
                        imgui.text_colored(C4(235, 180, 90),
                                           "best focus is OUTSIDE the swept range -- re-sweep around it")
                    if stt.get('sat_frac', 0) > 0.2:
                        imgui.text_colored(C4(235, 180, 90),
                                           f"{stt['sat_frac']:.0%} of frames saturated -- "
                                           "reduce exposure and re-sweep")
                elif stt.get('error'):
                    imgui.text_colored(C4(235, 120, 120), stt['error'])
            if stt.get('points'):
                _grey("The sweep curve draws on the star (main pane), under the focus graph.")
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Detection"):
            det_st = st.get('detectors') or {}
            for role in roles:
                if role != roles[0]:
                    imgui.separator()
                if not imgui.tree_node_ex(f"{role.capitalize()} Detection", OPEN):
                    continue
                cur = det_st.get(role) or {}
                imgui.text("detector:")
                _tip("Acquisition detector for this camera -- the surface that finds candidate "
                     "targets full-frame before a lock. Switching relaunches the detector "
                     "process (a couple of seconds; avoid mid-track).")
                imgui.same_line()
                idx = _DETECTORS.index(cur.get('detector')) if cur.get('detector') in _DETECTORS else -1
                imgui.set_next_item_width(-1)
                ch, nidx = imgui.combo(f"##det_{role}", idx, _DETECTORS)
                if ch and 0 <= nidx < len(_DETECTORS):
                    _send({'type': 'set_detector', 'role': role, 'detector': _DETECTORS[nidx]})
                # The single-target detectors (extended/circmean) always run full-frame and
                # never receive a tracking ROI -- the ROI-phase choice below is inert for them.
                single = cur.get('detector') in ('extended', 'circmean')
                imgui.begin_disabled(single)
                imgui.text("tracking:")
                _tip("Tracking-phase detector: the single-answer search inside the predicted ROI "
                     "once locked. 'peak' = surface peak with a found/lost gate; 'matched' = "
                     "stateless matched filter + centre pull, no gate. (Greyed out for the "
                     "single-target detectors -- extended/circmean run full-frame and never "
                     "use a tracking ROI.)")
                imgui.same_line()
                idx = (_TRACK_DETECTORS.index(cur.get('track'))
                       if cur.get('track') in _TRACK_DETECTORS else -1)
                imgui.set_next_item_width(-1)
                ch, nidx = imgui.combo(f"##tdet_{role}", idx, _TRACK_DETECTORS)
                if ch and 0 <= nidx < len(_TRACK_DETECTORS):
                    _send({'type': 'set_detector', 'role': role,
                           'track_detector': _TRACK_DETECTORS[nidx]})
                imgui.end_disabled()
                imgui.tree_pop()
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Tracking"):
            cap = st.get('capturing') or {}
            det = set(st.get('detect_roles') or [])
            cur_pref = st.get('track_pref') or track_ui['pref']
            avail = {'guide': bool(cap.get('guide')), 'main': bool(cap.get('main')) and 'main' in det,
                     'auto': True}
            imgui.text("Track:")
            _tip("Which camera drives tracking: Guide only, the Main cam, or Auto (acquire on the guide, "
                 "hand off to the main once it locks). A camera that isn't connected is greyed out.")
            for val, lbl in _TRACK_PREFS:
                imgui.same_line()
                imgui.begin_disabled(not avail.get(val, True))
                if imgui.button(('* ' if cur_pref == val else '  ') + lbl + f"##trk_pref_{val}", (S(56), 0)):
                    _set_track_pref(val)
                imgui.end_disabled()
            ch, v = imgui.checkbox("Auto switch to main pane##auto_switch", track_ui['auto_switch'])
            if ch:
                track_ui['auto_switch'] = v
            _tip("When tracking hands off between cameras, bring the newly-active camera into the main "
                 "pane (as if you'd pressed its ^ button).")
            locked = bool(st.get('tracking'))
            ch, v = imgui.checkbox("Follow target##follow_chk", bool(st.get('follow_enabled', True)))
            if ch:
                _send({'type': 'follow', 'on': v})
            _tip("Slew the mount to hold a locked target on the boresight. A persistent setting, usable "
                 "any time: uncheck BEFORE locking and tracking engages watch-only (the tracker estimates, "
                 "the mount holds still); re-check to start following.")
            imgui.text("Track delay:")
            _tip("After a new lock, hold the mount still for this many seconds while the tracker learns "
                 "the target's angular velocity -- a better estimate before the catch-up slew means a "
                 "better chance of re-acquiring if the slew loses the target.")
            imgui.same_line()

            def _commit_delay(txt):
                track_ui['delay'] = max(0.0, _flt(txt, track_ui['delay']))
                _send({'type': 'set_track_delay', 'value': track_ui['delay']})
                return f"{track_ui['delay']:g}"
            _input_commit('track_delay', S(48), _commit_delay)
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "s")
            imgui.begin_disabled(not locked)
            if imgui.button("Unlock", (S(200), 0)):
                _send({'type': 'untrack'})
            imgui.end_disabled()
            _tip("Drop the target lock entirely and halt the mount.")
            imgui.text("Rate scale:")
            _tip("Trim multiplier on the tracker's commanded rates, applied just before the mount "
                 "(manual slews are NOT scaled). If the mount executes rates slightly slow, the "
                 "target rides a constant offset ahead of centre -- nudge this up while locked "
                 "until it centres. Validate in sim against --sim-mount-rate-scale-alt. Applies live.")
            imgui.same_line()

            def _rate_scale_set(v):
                track_ui['rate_scale'] = max(0.5, min(2.0, v))
                ui['txt']['track_rate_scale'] = f"{track_ui['rate_scale']:g}"
                _send({'type': 'set_track_rate_scale', 'value': track_ui['rate_scale']})

            def _commit_rate_scale(txt):
                _rate_scale_set(_flt(txt, track_ui['rate_scale']))
                return ui['txt']['track_rate_scale']
            _input_commit('track_rate_scale', S(48), _commit_rate_scale)
            imgui.same_line()
            imgui.text_colored(C4(140, 145, 160), "x")
            for label, delta in (('<<', -0.01), ('<', -0.002), ('>', +0.002), ('>>', +0.01)):
                if label != '<<':
                    imgui.same_line()
                if imgui.button(f"  {label}  ##rate_scale_nudge{delta}"):
                    _rate_scale_set(track_ui['rate_scale'] + delta)
                _tip(f"Nudge the commanded-rate trim by {delta:+g} ({delta * 100:+g}%).")

            def _send_horizons():
                _send({'type': 'set_track_horizons', 'min_s': track_ui['min_horizon'],
                       'max_s': track_ui['max_horizon']})

            def _send_pid():
                _send({'type': 'set_track_pid', 'on': track_ui['pid_on'],
                       'kp': track_ui['pid_kp'], 'ki': track_ui['pid_ki'],
                       'kd': track_ui['pid_kd'],
                       'integral_limit_deg_s': track_ui['pid_integral_limit']})

            if imgui.tree_node_ex("Feedforward", OPEN):
                ch, v = imgui.checkbox("Enabled##feedforward_enable", track_ui['feedforward'])
                if ch:
                    track_ui['feedforward'] = v
                    _send({'type': 'set_track_feedforward', 'on': v})
                _tip("The feedforward intercept servo: aim at the model's predicted target position "
                     "and command the rate that gets there. Off: it contributes no rates -- only "
                     "the PID trim (if enabled) drives the mount. Applies live.")
                cur_model = st.get('track_model') or track_ui['model']
                imgui.text("Model:")
                _tip("Target motion model (applies at the NEXT lock). Sky: constant angular velocity "
                     "across the sky -- right for anything far (stars, planes at range). Great "
                     "Circle: a constant-altitude great circle about the Earth's centre -- right "
                     "for LEO passes, where the zenith speed-up and horizon slow-down are "
                     "perspective the Sky model has to chase but the orbit geometry produces for "
                     "free.")
                for val, lbl in (('ema', 'Sky'), ('greatcircle', 'Great Circle')):
                    imgui.same_line()
                    if imgui.button(('* ' if cur_model == val else '  ') + lbl + f"##trk_model_{val}"):
                        track_ui['model'] = val
                        _send({'type': 'set_track_model', 'model': val, 'alt_km': track_ui['alt_km']})
                imgui.begin_disabled(cur_model != 'greatcircle')
                imgui.text("Height:")
                _tip("Assumed target altitude for the Great Circle model.")
                imgui.same_line()

                def _commit_alt(txt):
                    track_ui['alt_km'] = max(1.0, _flt(txt, track_ui['alt_km']))
                    _send({'type': 'set_track_model', 'model': track_ui['model'],
                           'alt_km': track_ui['alt_km']})
                    return f"{track_ui['alt_km']:g}"
                _input_commit('track_alt', S(48), _commit_alt)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "km")
                imgui.end_disabled()
                imgui.text("Rate smoothing:")
                _tip("EMA time constant for the target's angular-velocity estimate. Bigger = smoother "
                     "(rides out sub-pixel detection jitter) but laggier on real acceleration; smaller = "
                     "snappier but noisier. Applies live and to the next lock.")
                imgui.same_line()

                def _commit_smooth(txt):
                    track_ui['smoothing'] = max(0.0, _flt(txt, track_ui['smoothing']))
                    _send({'type': 'set_track_smoothing', 'value': track_ui['smoothing']})
                    return f"{track_ui['smoothing']:g}"
                _input_commit('track_smooth', S(48), _commit_smooth)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "s")
                imgui.text("Position smoothing:")
                _tip("EMA time constant for blending each position measurement into the model's "
                     "extrapolated position, instead of snapping the anchor to every measurement. "
                     "Warmup-corrected: the first fix snaps, then the weight relaxes toward the "
                     "steady EMA. 0 = always snap (the original behavior). Applies live and to "
                     "the next lock.")
                imgui.same_line()

                def _commit_position_smoothing(txt):
                    track_ui['position_smoothing'] = max(0.0, _flt(txt, track_ui['position_smoothing']))
                    _send({'type': 'set_track_position_smoothing',
                           'value': track_ui['position_smoothing']})
                    return f"{track_ui['position_smoothing']:g}"
                _input_commit('track_position_smoothing', S(48), _commit_position_smoothing)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "s")
                imgui.text("Min horizon:")
                _tip("Earliest allowed intercept time. Doubles as the servo's position stiffness "
                     "(P ~ 1/this): smaller = stiffer chase, bigger = gentler. Applies live.")
                imgui.same_line()

                def _commit_min_horizon(txt):
                    track_ui['min_horizon'] = max(0.1, min(10.0, _flt(txt, track_ui['min_horizon'])))
                    if track_ui['max_horizon'] < track_ui['min_horizon']:
                        track_ui['max_horizon'] = track_ui['min_horizon']
                        ui['txt']['track_max_horizon'] = f"{track_ui['max_horizon']:g}"
                    _send_horizons()
                    return f"{track_ui['min_horizon']:g}"
                _input_commit('track_min_horizon', S(48), _commit_min_horizon)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "s")
                imgui.text("Max horizon:")
                _tip("Declare the target uncatchable if no intercept is reachable within this long. "
                     "Applies live.")
                imgui.same_line()

                def _commit_max_horizon(txt):
                    track_ui['max_horizon'] = max(1.0, min(60.0, _flt(txt, track_ui['max_horizon'])))
                    if track_ui['min_horizon'] > track_ui['max_horizon']:
                        track_ui['min_horizon'] = track_ui['max_horizon']
                        ui['txt']['track_min_horizon'] = f"{track_ui['min_horizon']:g}"
                    _send_horizons()
                    return f"{track_ui['max_horizon']:g}"
                _input_commit('track_max_horizon', S(48), _commit_max_horizon)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "s")
                imgui.text("Command latency:")
                _tip("Assumed delay before a mount rate command takes effect (serial pickup + wire + "
                     "controller processing). The servo aims its intercept this far ahead; an "
                     "under-modeled delay leaves the target trailing under acceleration. Try "
                     "~0.05-0.3 s on the NexStar serial link; nudge while locked and watch the "
                     "target re-centre. Negative is allowed: on the ideal sim mount it reproduces "
                     "an UNDER-modeled latency. Applies live and to the next lock.")
                imgui.same_line()

                def _latency_set(v):
                    track_ui['latency'] = max(-5.0, min(5.0, v))
                    ui['txt']['track_latency'] = f"{track_ui['latency']:g}"
                    _send({'type': 'set_track_latency', 'value': track_ui['latency']})

                def _commit_latency(txt):
                    _latency_set(_flt(txt, track_ui['latency']))
                    return ui['txt']['track_latency']
                _input_commit('track_latency', S(48), _commit_latency)
                imgui.same_line()
                imgui.text_colored(C4(140, 145, 160), "s")
                for label, delta in (('<<', -0.1), ('<', -0.02), ('>', +0.02), ('>>', +0.1)):
                    if label != '<<':
                        imgui.same_line()
                    if imgui.button(f"  {label}  ##latency_nudge{delta}"):
                        _latency_set(track_ui['latency'] + delta)
                    _tip(f"Nudge the assumed command latency by {delta:+g} s.")
                imgui.tree_pop()
            if imgui.tree_node_ex("PID", OPEN):
                ch, v = imgui.checkbox("Enabled##pid_enable", track_ui['pid_on'])
                if ch:
                    track_ui['pid_on'] = v
                    _send_pid()
                _tip("PID trim on the MEASURED centering error -- the detected target direction vs "
                     "the hold point, in sky-vector space -- added to the feedforward servo's "
                     "rates. In practice a small I term: it grinds a persistent offset (mount "
                     "rate execution error, model lag) to zero, e-fold time 1/(ki x min "
                     "horizon). Runs only once settled: the servo must intercept at the min "
                     "horizon for 5 consecutive min-horizons (waits out the approach transient "
                     "-- not during catch-up slews or over the top); unsettled or disabled "
                     "zeroes it. Applies live.")
                for key, label, unit, tip in (
                        ('pid_kp', 'kp:', '1/s',
                         "Proportional gain on the measured centering error. The feedforward servo "
                         "already carries a P term (~1/min-horizon), so usually 0."),
                        ('pid_ki', 'ki:', '1/s^2',
                         "Integral gain: drives a persistent centering offset to zero, e-fold "
                         "time 1/(ki x min horizon) -- ~2 s at the defaults. The workhorse "
                         "term. Applies only to NEW accumulation (the stored trim is not "
                         "rescaled), so the clamp is the fast-acting knob, this is the slow one."),
                        ('pid_kd', 'kd:', '',
                         "Derivative gain on the measured error's rate of change. Amplifies "
                         "detection jitter, so usually 0."),
                        ('pid_integral_limit', 'Integral limit:', 'deg/s',
                         "Windup limit: the integral term's magnitude is clamped to this rate. "
                         "Times 1/stiffness (= min horizon) it is also the max static offset a "
                         "wound integral can park us at; keep it just above the largest genuine "
                         "rate disturbance the trim must cancel. Re-applied to the STORED trim "
                         "on the next detection -- the fast-acting rescue knob.")):
                    imgui.text(label)
                    _tip(tip)
                    imgui.same_line()

                    def _commit_pid(txt, key=key):
                        value = _flt(txt, track_ui[key])          # gains may be negative (sign tests)
                        track_ui[key] = max(0.0, value) if key == 'pid_integral_limit' else value
                        _send_pid()
                        return f"{track_ui[key]:g}"
                    _input_commit(f'track_{key}', S(48), _commit_pid)
                    if unit:
                        imgui.same_line()
                        imgui.text_colored(C4(140, 145, 160), unit)
                imgui.tree_pop()
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Settings"):
            items = ui['settings_items']
            sidx = items.index(ui['settings_sel']) if ui['settings_sel'] in items else -1
            imgui.set_next_item_width(-1)
            ch, nidx = imgui.combo("##settings_combo", sidx, items)
            if ch and 0 <= nidx < len(items):
                ui['settings_sel'] = items[nidx]
            _tip("A saved settings file captures the layout, display prefs, optics + owned gear, cameras, "
                 "and boresight.")
            if imgui.button("Load##settings") and ui['settings_sel']:
                apply_settings(settings_store.load(ui['settings_sel']))
            imgui.same_line()
            if imgui.button("Delete##settings") and ui['settings_sel']:
                settings_store.delete(ui['settings_sel'])
                _settings_refresh('')
            imgui.same_line()
            if imgui.button("Show folder##settings"):
                _open_folder(settings_store.settings_dir())
            _tip("Open the settings folder in the file browser.")
            imgui.spacing()
            imgui.set_next_item_width(-S(56))
            ch, txt = imgui.input_text("##settings_name", ui['txt']['settings_name'],
                                       imgui.InputTextFlags_.none)
            ui['txt']['settings_name'] = txt
            imgui.same_line()
            if imgui.button("Save##settings"):
                name = (ui['txt']['settings_name'] or ui['settings_sel'] or '').strip()
                if name:
                    _settings_refresh(settings_store.save(name, gather_settings()))
                    ui['txt']['settings_name'] = ''
            imgui.tree_pop()
        imgui.separator()
        if imgui.tree_node_ex("Checklist", OPEN):
            _tip("Pre-capture checklist: every line states the truth right now -- green = go, "
                 "orange = fix it in its own panel above (nothing here changes settings). "
                 "Greyed checkboxes are automatic; live ones are things only YOU can verify, "
                 "and they reset every GUI launch, so re-confirm each session.")
            GOOD, FIX = C4(120, 220, 140), C4(255, 90, 25)

            def _row(key, ok, text, manual=False):
                """One checklist line: a checkbox (greyed automatic truth, or a live
                attestation box) + status text that always states the current truth."""
                if manual:
                    ch, v = imgui.checkbox(f"##chk_{key}", checklist_ui[key])
                    if ch:
                        checklist_ui[key] = v
                else:
                    imgui.begin_disabled(True)
                    imgui.checkbox(f"##chk_{key}", bool(ok))
                    imgui.end_disabled()
                imgui.same_line()
                _ctext(GOOD if ok else FIX, text)

            off_ns = st.get('utc_offset_ns')
            if off_ns is None:
                _row('time', False, "Time: waiting for backend!")
            else:
                drift_s = (off_ns + session_mod.mono_ns() - time.time_ns()) * 1e-9
                _row('time', abs(drift_s) < 2.0,
                     f"Time: {drift_s:+.1f} s vs system clock"
                     + ("" if abs(drift_s) < 2.0 else " -- not real time!"))

            mount_url = st.get('mount_url') or 'sim'
            if not st.get('mount_connected'):
                _row('mount', False, "Mount: not connected!")
            elif mount_url == 'sim':
                _row('mount', False, "Mount: connected but not real (sim)")
            else:
                _row('mount', True, f"Mount: connected ({mount_url})")

            capturing = st.get('capturing') or {}
            src = st.get('sources') or {}
            cam_roles = [r for r in ('guide', 'main') if r in roles]
            for role in cam_roles:
                name = f"{role.capitalize()} cam"
                if not capturing.get(role):
                    _row(f'{role}_cam', False, f"{name}: not connected!")
                elif src.get(role) != 'zwo':
                    _row(f'{role}_cam', False,
                         f"{name}: connected but not real ({src.get(role)})")
                else:
                    cc = (st.get('camera_caps') or {}).get(role) or {}
                    vals = {c['name']: c.get('value') for c in cc.get('controls') or []}
                    bits = [cc.get('camera') or 'zwo']
                    if vals.get('exposure') is not None:
                        bits.append(f"{float(vals['exposure']):g} ms")
                    if vals.get('gain') is not None:
                        bits.append(f"gain {float(vals['gain']):g}")
                    _row(f'{role}_cam', True, f"{name}: connected ({'  '.join(bits)})")

            opt_sel = st.get('optics_sel') or {}
            for role in cam_roles:
                sel = ' + '.join(n for n in (opt_sel.get(role) or []) if n)
                if not sel:
                    _row(f'{role}_optics', False,
                         f"{role.capitalize()} optics: not set!", manual=True)
                else:
                    _row(f'{role}_optics', checklist_ui[f'{role}_optics'],
                         f"{role.capitalize()} optics: {sel}", manual=True)
            _row('focus', checklist_ui['focus'],
                 "Focus: checked" if checklist_ui['focus'] else "Focus: not checked!",
                 manual=True)
            bm = (list(st.get('boresight_mrad') or []) + [0.0, 0.0, 0.0])[:3]
            if not any(bm):
                # A zero boresight is almost certainly "never measured" -- complain even if
                # the box is ticked.
                _row('boresight', False, "Boresight: still zero!", manual=True)
            else:
                _row('boresight', checklist_ui['boresight'],
                     f"Boresight: ({bm[0]:g}, {bm[1]:g}, roll {bm[2]:g}) mrad", manual=True)

            auto_record = bool((st.get('record_auto') or {}).get('main'))
            _row('record', auto_record,
                 "Main cam auto record: on (records while tracking)" if auto_record
                 else "Main cam auto record: off!")
            pref = st.get('track_pref') or '?'
            _row('pref', pref == 'auto',
                 "Tracking source: Auto" if pref == 'auto'
                 else f"Tracking source: {pref} -- pinned, no handoff!")
            _row('follow', bool(st.get('follow_enabled')),
                 "Follow target: on" if st.get('follow_enabled')
                 else "Follow target: off -- mount will not move!")
            ff, pid = bool(st.get('track_feedforward')), bool(st.get('track_pid'))
            enabled = [n for n, on in (('feedforward', ff), ('PID', pid)) if on]
            _row('controller', bool(enabled),
                 f"Controller: {' + '.join(enabled)}" if enabled
                 else "Controller: feedforward and PID both off!")
            detect_roles = set(st.get('detect_roles') or [])
            missing = [r for r in cam_roles if r not in detect_roles]
            _row('detectors', not missing,
                 f"Detectors: {', '.join(sorted(detect_roles))}" if not missing
                 else f"Detectors: missing {', '.join(missing)}!")
            imgui.tree_pop()

    # ---- main loop -----------------------------------------------------------------------------
    wake = {'paths': (), 'probes': (), 'state_slot': None}
    waker_stop = threading.Event()
    threading.Thread(target=_waker, args=(glfw, wake, waker_stop), daemon=True).start()

    while not glfw.window_should_close(window):
        # Tell the watcher which files to watch (the currently-followed .ser + detection sidecars +
        # the backend state file), then block until it (or an OS input event) wakes us. The 0.25s
        # timeout is the idle heartbeat: blink text, meter sampling, tooltip delays.
        # v3: frames commit in RAM, so the waker PROBES shm headers directly (pure memory
        # reads, no syscalls) and stat-watches only the head files (segment/ended events),
        # detection sidecars, and the state file.
        wake['probes'] = tuple(followers.values()) + tuple(det_fos.values())
        # (followers expose committed() as a pure memory read of the live ring's header)
        paths = [framestream.head_path(args.session, name) for name in followers]
        paths += [framestream.head_path(args.session, f'{r}_det') for r in det_fos]
        wake['paths'] = paths
        wake['state_slot'] = ctrl.get('slot')
        glfw.wait_events_timeout(0.25)

        update_control()
        for role in roles:
            update_cam(role)
        layout['pip_map'] = {}
        pip_names = _pip_streams() if layout['pip_open'] else []
        layout['pip_slots'] = [f"pip{i}" for i in range(len(pip_names))]
        layout['pip_map'] = {f"pip{i}": s for i, s in enumerate(pip_names)}
        for name in _active_slots():                     # advance any derived surface a slot is showing
            stream = _slot_stream(name)
            if stream not in roles:
                update_cam(stream)

        imgui.backends.opengl3_new_frame()
        imgui.backends.glfw_new_frame()
        imgui.new_frame()

        vw, vh = io.display_size.x, io.display_size.y
        bm = S(6)                                    # bottom margin: keep panes off the viewport edge
        pw = layout['panel_w'] if layout['panel_open'] else 0
        vsp = S(6) if layout['panel_open'] else 0
        usable_h = vh - bm
        left_w = max(S(120), vw - pw - vsp)
        strip = bool(pip_names)
        hsp = S(6) if strip else 0
        ph = layout['pip_h'] if strip else 0
        ph = max(0, min(ph, usable_h - hsp - S(200)))   # keep the big pane >= ~200 tall
        big_h = usable_h - hsp - ph

        imgui.set_next_window_pos((0, 0))
        imgui.set_next_window_size((vw, vh))
        imgui.push_style_var(imgui.StyleVar_.window_padding, (0, 0))
        imgui.push_style_var(imgui.StyleVar_.window_rounding, 0.0)
        imgui.push_style_var(imgui.StyleVar_.window_border_size, 0.0)
        imgui.begin("##root", None, ROOT_WF)
        root_dl = imgui.get_window_draw_list()

        _tdraw = time.perf_counter()
        _pane('big', 0, 0, left_w, big_h)
        if strip and ph > S(20):
            # Horizontal divider between the big pane and the PIP strip (drag to resize the strip).
            imgui.set_cursor_screen_pos((0, big_h))
            imgui.invisible_button("##hsplitter", (left_w, hsp))
            if imgui.is_item_hovered() or imgui.is_item_active():
                imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
            hcol = (130, 135, 150) if imgui.is_item_active() else \
                ((105, 110, 124) if imgui.is_item_hovered() else (70, 74, 84))
            root_dl.add_rect_filled((0, big_h), (left_w, big_h + hsp), C(hcol))
            if imgui.is_item_active():
                layout['pip_h'] = int(max(S(120), min(vh - S(300), (vh - S(6)) - S(6) - io.mouse_pos.y)))
            n = len(pip_names)
            gap = S(4)
            col_w = (left_w - gap * (n - 1)) / n         # equal columns across the strip
            for i in range(n):
                x = round(i * (col_w + gap))
                w_ = round(col_w) if i < n - 1 else left_w - x    # last pane absorbs the rounding remainder
                _pane(f"pip{i}", x, big_h + hsp, w_, ph)
        if layout['panel_open']:
            # Vertical divider between the big+PIP column and the right panel (drag to resize the panel).
            imgui.set_cursor_screen_pos((vw - pw - vsp, 0))
            imgui.invisible_button("##vsplitter", (vsp, vh))
            if imgui.is_item_hovered() or imgui.is_item_active():
                imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
            vcol = (130, 135, 150) if imgui.is_item_active() else \
                ((105, 110, 124) if imgui.is_item_hovered() else (70, 74, 84))
            root_dl.add_rect_filled((vw - pw - vsp, 0), (vw - pw, vh), C(vcol))
            if imgui.is_item_active():
                layout['panel_w'] = int(max(S(PANEL_MIN_W), min(vw - S(320), vw - io.mouse_pos.x)))
            imgui.set_cursor_screen_pos((vw - pw, 0))
            imgui.push_style_var(imgui.StyleVar_.window_padding, (S(8), S(8)))
            imgui.begin_child("##panel", (pw, vh), 0, 0)
            _panel()
            imgui.end_child()
            imgui.pop_style_var()
        _prof('draw', (time.perf_counter() - _tdraw) * 1e3)      # all active panes + panel this frame

        imgui.end()
        imgui.pop_style_var(3)

        imgui.render()
        fbw, fbh = glfw.get_framebuffer_size(window)
        ctx.screen.use()
        ctx.viewport = (0, 0, fbw, fbh)
        ctx.clear(0.05, 0.055, 0.065)
        imgui.backends.opengl3_render_draw_data(imgui.get_draw_data())
        glfw.swap_buffers(window)

    waker_stop.set()
    _shutdown()      # tell the backend we're closing, then exit immediately


if __name__ == '__main__':
    main()
