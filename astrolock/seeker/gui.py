"""
astrolock_seeker_gui: Dear PyGui viewer.

A fixed, self-tiling layout (not free-floating windows): a large 'big' pane top-left showing a
selected camera, a docked settings/telemetry panel on the right (drag-resizable), and a strip of
PIP panes along the bottom (the other camera; focus/boresight zooms later). Roles (guide, main)
are decoupled from slots -- switch which camera is 'big' and the other drops to the PIP. Each pane
letterboxes its camera (preserve aspect, centred) at a power-of-two scale that fits with room to
spare; you can zoom in further (crops, with edge indicators). Everything reflows on viewport resize.

Same follower code path for live tailing and historical playback -- it just reads growing-or-
complete files.

    python -m astrolock.seeker.gui --session sessions/<ts>

Requires `dearpygui` (pip install dearpygui).
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

from astrolock.seeker import bayer, control, ser
from astrolock.seeker import optics as optics_db
from astrolock.seeker import settings as settings_store
from astrolock.seeker.follower import SerFollower
from astrolock.seeker.sidecar import JsonlTailer


def _newest(session_dir, suffix):
    matches = sorted(glob.glob(os.path.join(session_dir, '*' + suffix)))
    return matches[-1] if matches else None


def _color_name(cid):
    try:
        return ser.ColorId(int(cid)).name
    except ValueError:
        return str(cid)


_DEVICE = torch.device('cpu')        # switch to 'cuda' once a CUDA torch is installed
_LUT_CACHE = {}


def _gamma_lut(white_int, gain, gamma, device):
    """Cached torch LUT: raw value [0..white_int] -> display [0,1] (WB gain + gamma). A table
    lookup instead of a per-pixel pow -- the per-frame hot path."""
    key = (white_int, round(gain, 4), round(gamma, 4), str(device))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        v = (torch.arange(white_int + 1, dtype=torch.float32, device=device) * (gain / white_int)).clamp_(0.0, 1.0)
        if gamma and gamma != 1.0:
            v = v.pow_(1.0 / gamma)
        lut = _LUT_CACHE[key] = v
    return lut


def prepare_rgba(frame_raw, color_id, gamma, wb=(1.0, 1.0), device=None):
    """
    Raw frame (mosaic or mono) -> (w, h, (h,w,4) float32 RGBA on CPU, for the dpg texture).
    All compute is torch and device-parameterized (GPU-ready); torch has no uint16, so the frame is
    cast to int32 at the single ingest boundary, then everything stays in torch until the final
    .cpu().numpy() for the upload. Debayers Bayer to half-res RGB (4-plane split), applies
    display-only WB (R,B gains -- stored data stays pristine), maps the full container range to
    [0,1] with NO auto-stretch, applies gamma -- WB+gamma via a cached LUT, not a per-pixel pow.
    """
    device = device or _DEVICE
    white_int = int(np.iinfo(frame_raw.dtype).max) if np.issubdtype(frame_raw.dtype, np.integer) else 1
    frame = torch.from_numpy(np.ascontiguousarray(frame_raw).astype(np.int32, copy=False)).to(device)

    if bayer.is_bayer(color_id):
        planes = (frame[0::2, 0::2], frame[0::2, 1::2], frame[1::2, 0::2], frame[1::2, 1::2])
        ri, (g0, g1), bi = bayer.rgb_plane_indices(color_id)
        chans = ((planes[ri], wb[0]), ((planes[g0] + planes[g1]) // 2, 1.0), (planes[bi], wb[1]))
    else:
        chans = ((frame, 1.0),)

    h, w = chans[0][0].shape
    rgba = torch.ones((h, w, 4), dtype=torch.float32, device=device)    # alpha pre-filled to 1.0
    for c, (idx, gain) in enumerate(chans):
        disp = _gamma_lut(white_int, gain, gamma, device)[idx.clamp(0, white_int).long()]
        if len(chans) == 1:                                            # mono -> gray
            rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = disp
        else:
            rgba[..., c] = disp
    return w, h, rgba.cpu().numpy()                                     # CPU only at the end, for dpg


def draw_box(rgba, cx, cy, half, color):
    """Draw a hollow square (work-image coords) into an (h,w,4) rgba array. The live GUI draws
    overlays as dpg vectors, but this stays as a pure helper for baking boxes into saved frames."""
    h, w = rgba.shape[:2]
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x1 <= x0 or y1 <= y0:
        return
    rgba[y0, x0:x1 + 1] = color
    rgba[y1, x0:x1 + 1] = color
    rgba[y0:y1 + 1, x0] = color
    rgba[y0:y1 + 1, x1] = color


_MOVING = np.array([0.2, 1.0, 0.2, 1.0], dtype=np.float32)
_STATIC = np.array([1.0, 0.8, 0.2, 1.0], dtype=np.float32)


ROLES = ('guide', 'main')      # the two fixed roles: a wide guide cam that points a narrow main cam.
                               # Either may be absent/unconfigured; we don't add roles dynamically.


# --- Fixed tiled layout --------------------------------------------------------------------
# The viewport is tiled, not free-floating: a 'big' pane top-left, a docked settings panel on the
# right (drag-resizable), and a strip of PIP panes along the bottom. Roles are decoupled from
# slots -- the big pane shows a selected camera and the PIP shows the other. Everything is
# positioned by relayout() from the viewport size, so it reflows on resize.
PANEL_W = 320                    # default right-panel width (logical px, pre-DPI)
PANEL_MIN_W = 220
ZOOM_MULTS = (1, 2, 4, 8, 16)    # zoom is a multiplier over the auto power-of-two fit (1 = fit-to-pane)


def _default_settings():
    return {'zoom': 1, 'reticles': True, 'histogram': False}


def _zoom_label(z):
    return "fit" if z == 1 else f"{int(z)}×"


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
# A square az/alt rate plane: drag to drive the mount, log-scaled so it's fine near zero and full
# slew at the edge, with a centre dead-zone that reads as exactly zero.
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


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker GUI viewer")
    p.add_argument('--session', required=True, help="session directory to view")
    p.add_argument('--roles', default=None,
                   help="playback override: view a subset of an old session (default: guide,main)")
    p.add_argument('--display-width', type=int, default=640, help="(unused: panes now letterbox-fit)")
    p.add_argument('--gamma', type=float, default=2.2, help="display gamma (1 = linear)")
    p.add_argument('--wb-r', type=float, default=1.24, help="display-only WB gain for red")
    p.add_argument('--wb-b', type=float, default=1.98, help="display-only WB gain for blue")
    p.add_argument('--slew-rate', type=float, default=3.0, help="slew rate while a button is held (deg/s)")
    p.add_argument('--ui-scale', type=float, default=0.0,
                   help="UI/DPI scale factor (0 = auto-detect from the OS; e.g. 1.5 for a 150%% display)")
    p.add_argument('--device', default='cpu', help="torch device for image processing (cpu / cuda)")
    args = p.parse_args(argv)
    wb = (args.wb_r, args.wb_b)
    device = torch.device(args.device)

    import dearpygui.dearpygui as dpg

    # Declare per-monitor DPI awareness *before* the viewport exists, so Windows gives us a
    # native-resolution framebuffer instead of bitmap-upscaling a low-res window (which blurs
    # the text). Then read the monitor scale so we can size the UI up to match. Must precede
    # create_context.
    ui_scale = args.ui_scale
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        except Exception:
            pass  # older Windows, awareness already set via manifest, etc. -- purely cosmetic, never fatal
        if ui_scale <= 0:
            try:
                ui_scale = ctypes.windll.user32.GetDpiForSystem() / 96.0
            except Exception:
                ui_scale = 0.0
    if ui_scale <= 0:
        ui_scale = 1.0

    def S(v):
        """Scale a pixel dimension by the display's DPI factor, rounded to an int."""
        return int(round(v * ui_scale))

    # The two fixed roles (guide, main). --roles stays only as an optional playback override.
    roles = ([r.strip() for r in args.roles.split(',') if r.strip()] if args.roles else list(ROLES))
    followers = {}
    cams = {}                 # role -> live camera data (texture + frames + detections); lazily created
    view_settings = {}        # role -> display prefs {zoom, reticles, histogram}; persists across cams
    cam_ctrl_val = {}         # (role, control name) -> current value; the GUI owns it once a control is shown
    layout = {'panel_open': True, 'pip_open': True, 'pip_debug': False, 'panel_w': S(PANEL_W),
              'pip_h': S(200), 'big_role': ROLES[0], '_sig': None}

    dpg.create_context()
    dpg.set_global_font_scale(ui_scale)   # crisp text at the right size (ImGui 1.92 re-rasterizes)

    with dpg.theme() as slot_theme:            # camera panes: dim letterbox bars (so pane edges read), no padding/border
        with dpg.theme_component(dpg.mvAll):
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 32, 38, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize, 0, category=dpg.mvThemeCat_Core)

    # ---- small helpers -------------------------------------------------------------------
    def _item_rect(tag):
        """(rect_min, rect_size) for an item, or (None, None) if missing / not yet rendered (dpg has
        no rect state until an item's first frame; a hidden item has none either)."""
        if not tag or not dpg.does_item_exist(tag):
            return None, None
        st = dpg.get_item_state(tag)
        rmin, rsz = st.get('rect_min'), st.get('rect_size')
        return (rmin, rsz) if (rmin is not None and rsz is not None) else (None, None)

    def _hit(tag, mx, my):
        rmin, rsz = _item_rect(tag)
        if rmin is None:
            return False
        return rmin[0] <= mx <= rmin[0] + rsz[0] and rmin[1] <= my <= rmin[1] + rsz[1]

    def _other(role):
        return ROLES[1] if role == ROLES[0] else ROLES[0]

    def _slot_role(name):
        return layout['big_role'] if name == 'big' else _other(layout['big_role'])

    def _slot_stream(name):
        """The stream a slot DISPLAYS -- normally its role, but with Dbg on the pip shows the big pane's
        detector debug surface (<role>_debug.ser, written when the backend has --debug-detect-ser). Only
        the display + follower path uses this; toolbar/target-pick actions keep the real role."""
        if name == 'pipother' and layout.get('pip_debug'):
            return layout['big_role'] + '_debug'
        return _slot_role(name)

    def _toggle_dbg():
        layout['pip_debug'] = not layout['pip_debug']
        if layout['pip_debug']:
            layout['pip_open'] = True                    # no point showing the debug surface with the pip hidden

    def _active_slots():
        return ['big'] + (['pipother'] if layout['pip_open'] else [])

    def _zoom_step(role, delta):
        s = view_settings.setdefault(role, _default_settings())
        i = ZOOM_MULTS.index(s['zoom']) if s['zoom'] in ZOOM_MULTS else 0
        s['zoom'] = ZOOM_MULTS[max(0, min(len(ZOOM_MULTS) - 1, i + delta))]

    ctrl = {'client': None, 'tailer': None, 'state': None, 'last_rate': None}

    def _send(obj):
        if ctrl['client'] is not None:
            ctrl['client'].send(obj)

    def _shutdown(*_):
        """Tell the backend we're closing (it stops as soon as it drains this), then drop the
        process immediately -- os._exit skips interpreter/atexit teardown (and any auto-attached
        debugger) that could otherwise keep us -- and so the backend -- alive after the window
        closes. Wired to dpg's exit callback and the loop end, so it runs no matter how we leave."""
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

    # ---- per-role camera data (textures + frames + detections) ---------------------------
    def update_cam(role):
        """Advance a role's follower: upload the newest frame to its texture, poll detections, and
        refresh the histogram. Textures/data are per-role (a slot draws whichever role it shows).
        Returns True if a new frame was uploaded."""
        f = followers.get(role) or followers.setdefault(role, SerFollower(args.session, role))
        res = f.read_latest()
        if res is None or f.header is None:
            return False
        idx, frame = res
        fh, fw = frame.shape[0], frame.shape[1]
        cam = cams.get(role)
        if cam is None:
            w, h, _rgba = prepare_rgba(frame, f.header.color_id, args.gamma, wb, device=device)
            tex = f"tex_{role}_0"
            if not dpg.does_item_exist(tex):
                with dpg.texture_registry():
                    dpg.add_raw_texture(w, h, np.zeros(w * h * 4, dtype=np.float32),
                                        format=dpg.mvFormat_Float_rgba, tag=tex)
            det_path = f.ser_path[:-len('.ser')] + '.detections.jsonl'
            cam = cams[role] = dict(tex=tex, texver=0, w=w, h=h, fw=fw, fh=fh, ox=w / fw, oy=h / fh,
                                    color_id=f.header.color_id, blobs=[], det_idx=-1, last_idx=-1,
                                    hist=None, det_tailer=JsonlTailer(det_path), ser_path=f.ser_path)
        # segment rollover / source switch -> re-point the detections tailer
        if f.ser_path != cam['ser_path']:
            cam['det_tailer'].close()
            cam['det_tailer'] = JsonlTailer(f.ser_path[:-len('.ser')] + '.detections.jsonl')
            cam['ser_path'] = f.ser_path
            cam['last_idx'] = cam['det_idx'] = -1
        for rec in cam['det_tailer'].poll():
            cam['blobs'] = rec.get('blobs', [])
            cam['det_idx'] = rec.get('index', cam['det_idx'])
        if idx == cam['last_idx']:
            return False
        w, h, rgba = prepare_rgba(frame, f.header.color_id, args.gamma, wb, device=device)
        if (w, h) != (cam['w'], cam['h']):          # frame size changed (source/optics switch) -> a
            old = cam['tex']                          # fresh texture. Use a new tag so the string alias
            cam['texver'] += 1                        # never collides -- delete_item leaves the alias
            cam['tex'] = f"tex_{role}_{cam['texver']}"  # registered while a draw_image still references it.
            with dpg.texture_registry():
                dpg.add_raw_texture(w, h, np.zeros(w * h * 4, dtype=np.float32),
                                    format=dpg.mvFormat_Float_rgba, tag=cam['tex'])
            if dpg.does_item_exist(old):
                dpg.delete_item(old)                  # safe: draw_slot re-points its draw_image this frame
            cam.update(w=w, h=h, fw=fw, fh=fh, ox=w / fw, oy=h / fh, color_id=f.header.color_id)
        dpg.set_value(cam['tex'], rgba.ravel())
        cam['last_idx'] = idx
        # The histogram inset is off by default; only pay for it (a full-frame subsample + np.histogram
        # every frame) when it's actually enabled for this role. Otherwise skip it entirely.
        if view_settings.get(role, {}).get('histogram'):
            samp = rgba[::4, ::4, :3].mean(axis=2)        # luminance of the *displayed* image (WYSIWYG)
            counts, _ = np.histogram(samp, bins=64, range=(0.0, 1.0))
            m = counts.max()
            cam['hist'] = np.sqrt(counts / m) if m > 0 else None
        else:
            cam['hist'] = None
        return True

    # ---- slots (fixed camera panes) ------------------------------------------------------
    _LAYERS = ('img', 'box', 'fov', 'ret', 'trk', 'cut', 'hist', 'warn', 'tb')  # tb = toolbar, on top

    def make_slot(name):
        with dpg.window(tag=f"slot_{name}", no_title_bar=True, no_move=True, no_resize=True,
                        no_scrollbar=True, no_collapse=True, no_bring_to_front_on_focus=True):
            with dpg.drawlist(width=10, height=10, tag=f"dl_{name}"):
                for L in _LAYERS:
                    dpg.add_draw_layer(tag=f"L_{L}_{name}")
        dpg.bind_item_theme(f"slot_{name}", slot_theme)

    def _toolbar_defs(name):
        """(label, action) for a pane's top-left buttons. Actions resolve the slot's role at call
        time, so they follow a Swap. Drawn into the drawlist + hit-tested in on_left_click (real
        child-window buttons don't reliably capture clicks over the drawlist)."""
        if name == 'big':
            return [('Panel', lambda: layout.__setitem__('panel_open', not layout['panel_open'])),
                    ('Swap',  lambda: layout.__setitem__('big_role', _other(layout['big_role']))),
                    ('PIP',   lambda: layout.__setitem__('pip_open', not layout['pip_open'])),
                    ('Dbg',   _toggle_dbg),               # pip shows this pane's detector surface
                    ('-',     lambda: _zoom_step(_slot_role('big'), -1)),
                    ('+',     lambda: _zoom_step(_slot_role('big'), +1))]
        return [('Swap', lambda n=name: layout.__setitem__('big_role', _slot_role(n))),
                ('-',    lambda n=name: _zoom_step(_slot_role(n), -1)),
                ('+',    lambda n=name: _zoom_step(_slot_role(n), +1))]

    def _toolbar_layout(name):
        """[(label, (x0,y0,x1,y1) in drawlist coords, action), ...] laid left-to-right from top-left."""
        x, y, h, pad, gap = S(6), S(6), S(24), S(9), S(5)
        out = []
        for label, action in _toolbar_defs(name):
            bw = max(S(26), len(label) * S(8) + 2 * pad)
            out.append((label, (x, y, x + bw, y + h), action))
            x += bw + gap
        return out

    def _draw_toolbar(name, rmin):
        mx, my = dpg.get_mouse_pos(local=False)
        mlx, mly = mx - rmin[0], my - rmin[1]                # mouse in drawlist coords (for hover)
        for label, (x0, y0, x1, y1), _a in _toolbar_layout(name):
            hot = x0 <= mlx <= x1 and y0 <= mly <= y1
            dpg.draw_rectangle((x0, y0), (x1, y1), rounding=S(3), color=(95, 100, 115, 235),
                               fill=(44, 48, 60, 235) if hot else (24, 26, 34, 205), parent=f"L_tb_{name}")
            dpg.draw_text((x0 + S(6), y0 + S(5)), label, size=S(13), color=(235, 235, 240, 255),
                          parent=f"L_tb_{name}")

    PIP_R = S(15)      # pipper circle radius; a view's centre crosshairs stop here so a centred pipper joins them

    def _draw_pipper(name, tx, ty, col, box):
        """Circle at the target, clamped to stay inside the visible camera view `box` (x0,y0,x1,y1),
        + 4 short lines from the circle pointing toward the *true* target centre. In-frame that's the
        symmetric inward ticks (circle -> halfway to centre); when the target is off-frame the circle
        sits at the view edge and the lines point the way to it."""
        x0, y0, x1, y1 = box
        m = PIP_R + S(2)
        ccx = min(max(tx, x0 + m), max(x0 + m, x1 - m))   # inner max guards a view narrower than 2m
        ccy = min(max(ty, y0 + m), max(y0 + m, y1 - m))
        dpg.draw_circle((ccx, ccy), PIP_R, color=col, thickness=1.0, parent=f"L_trk_{name}")
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sx, sy = ccx + dx * PIP_R, ccy + dy * PIP_R      # a point on the circle
            vx, vy = tx - sx, ty - sy                         # toward the true target centre
            n = math.hypot(vx, vy)
            if n < 1e-6:
                continue
            dpg.draw_line((sx, sy), (sx + vx / n * PIP_R * 0.5, sy + vy / n * PIP_R * 0.5),
                          color=col, thickness=1.0, parent=f"L_trk_{name}")

    # Both dividers are thin dpg windows, but a dpg window has a ~32px minimum size, so a thin one's
    # body overflows past its visible sliver. Each is created *behind* the pane it borders and kept
    # there (no_bring_to_front_on_focus) so that pane covers the overflow; only the sliver in the gap
    # is the grab handle. Creation order sets who's behind whom: big, hsplitter, pip, vsplitter, panel.
    with dpg.theme() as splitter_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 74, 84, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (105, 110, 124, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (130, 135, 150, 255), category=dpg.mvThemeCat_Core)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 0, category=dpg.mvThemeCat_Core)

    make_slot('big')
    # Horizontal divider between the big pane and the PIP strip (drag to resize the strip). Behind
    # the PIP so its 32px-min body is hidden there; this lives only in the left column, so the
    # full-height settings panel is unaffected.
    with dpg.window(tag="hsplitter", no_title_bar=True, no_move=True, no_resize=True,
                    no_scrollbar=True, no_collapse=True, no_bring_to_front_on_focus=True):
        dpg.add_button(tag="hsplitter_btn", label="", width=S(40), height=S(6))
    dpg.bind_item_theme("hsplitter", slot_theme)
    dpg.bind_item_theme("hsplitter_btn", splitter_theme)
    make_slot('pipother')
    # Vertical divider between the big+PIP column and the right panel (drag to resize the panel).
    with dpg.window(tag="splitter", no_title_bar=True, no_move=True, no_resize=True,
                    no_scrollbar=True, no_collapse=True, no_bring_to_front_on_focus=True):
        dpg.add_button(tag="splitter_btn", label="", width=S(6), height=S(40))
    dpg.bind_item_theme("splitter", slot_theme)
    dpg.bind_item_theme("splitter_btn", splitter_theme)

    def _draw_placeholder(name, SW, SH, role):
        msg = f"{role} — no data"
        dpg.draw_text((SW / 2.0 - S(70), SH / 2.0 - S(10)), msg, size=S(18),
                      color=(140, 145, 160, 255), parent=f"L_warn_{name}")

    def draw_slot(name):
        """Draw the slot's assigned stream letterboxed + centred, with overlays, at the pane's size."""
        role = _slot_stream(name)                        # display stream (may be the <role>_debug surface)
        rmin, dlsz = _item_rect(f"dl_{name}")
        if dlsz is None:                       # not laid out / rendered yet -> next frame
            return
        SW, SH = dlsz
        for L in _LAYERS:
            dpg.delete_item(f"L_{L}_{name}", children_only=True)
        _draw_toolbar(name, rmin)              # always available, even before the camera has data
        cam = cams.get(role)
        if cam is None or not dpg.does_item_exist(cam['tex']):
            _draw_placeholder(name, SW, SH, role)
            return
        w, h = cam['w'], cam['h']
        sset = view_settings.setdefault(role, _default_settings())
        zoom = sset['zoom']
        # Default scale = largest power of two that fits with room to spare; zoom-in crops (centred).
        scale = _floor_pow2(min(SW / w, SH / h) * 0.95) * zoom
        dw, dh = w * scale, h * scale
        offx, offy = (SW - dw) / 2.0, (SH - dh) / 2.0
        cx, cy = SW / 2.0, SH / 2.0

        def T(fx, fy):                          # frame (detect) px -> pane screen px
            return offx + fx * cam['ox'] * scale, offy + fy * cam['oy'] * scale

        dpg.draw_image(cam['tex'], (offx, offy), (offx + dw, offy + dh), parent=f"L_img_{name}")

        # Detection boxes (green = moving, amber = static; faded if detect lags the shown frame).
        a = 255 if cam['det_idx'] >= cam['last_idx'] else 70
        for b in cam['blobs']:
            X, Y = T(b['px'][0], b['px'][1])
            half = max(S(4), b.get('size_px', 4) * cam['ox'] * scale) + S(3)
            col = (60, 255, 60, a) if b.get('moving') else (255, 200, 40, a)
            dpg.draw_rectangle((X - half, Y - half), (X + half, Y + half), color=col,
                               thickness=1.0, parent=f"L_box_{name}")

        # --- Reticles + target pipper (alpha'd red unless noted; all FoV via the pinhole tan-ratio) ---
        RED = (255, 70, 70, 160)                  # alpha'd red (the default for reticle geometry)
        il, ir, it, ib = offx, offx + dw, offy, offy + dh   # image edges (not the letterbox bars)
        if sset['reticles']:
            optx = (ctrl['state'] or {}).get('optics', {})
            me = optx.get(role)
            inner = None                          # a narrower co-aligned cam nested in this view (main in guide)
            if me:
                for r2, fv2 in optx.items():
                    if r2 != role and fv2['fov_x_deg'] < me['fov_x_deg'] and fv2['fov_y_deg'] < me['fov_y_deg']:
                        inner = fv2
                        break
            if inner is not None:
                # Main-cam FoV half-size (pinhole: a ray at half-angle th lands at focal_px*tan(th),
                # so the inner edge sits at (dw/2)*tan(inner_half)/tan(my_half)).
                hw = math.tan(math.radians(inner['fov_x_deg'] / 2)) / \
                    math.tan(math.radians(me['fov_x_deg'] / 2)) * (dw / 2.0)
                hh = math.tan(math.radians(inner['fov_y_deg'] / 2)) / \
                    math.tan(math.radians(me['fov_y_deg'] / 2)) * (dh / 2.0)
                mrcx, mrcy = cx, cy               # main-rect centre = boresight; centre until configurable
                gh, gv = dw / 2.0 - hw, dh / 2.0 - hh   # image-edge -> centred-rect-edge gaps
                # Centre crosshairs: from each image edge, 90% of the way to the *centred* rect edge.
                dpg.draw_line((il, cy), (il + 0.9 * gh, cy), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                dpg.draw_line((ir, cy), (ir - 0.9 * gh, cy), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                dpg.draw_line((cx, it), (cx, it + 0.9 * gv), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                dpg.draw_line((cx, ib), (cx, ib - 0.9 * gv), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                # Main-cam rect + its own crosshair stubs (outside it), each the last 10% of the gap
                # drawn from the rect edge -> they meet the centre crosshairs *iff* the rect is centred,
                # so a boresight offset will show up as a visible break (once boresight is configurable).
                dpg.draw_rectangle((mrcx - hw, mrcy - hh), (mrcx + hw, mrcy + hh), color=RED,
                                   thickness=1.0, parent=f"L_fov_{name}")
                dpg.draw_line((mrcx - hw, mrcy), (mrcx - hw - 0.1 * gh, mrcy), color=RED, thickness=1.0, parent=f"L_fov_{name}")
                dpg.draw_line((mrcx + hw, mrcy), (mrcx + hw + 0.1 * gh, mrcy), color=RED, thickness=1.0, parent=f"L_fov_{name}")
                dpg.draw_line((mrcx, mrcy - hh), (mrcx, mrcy - hh - 0.1 * gv), color=RED, thickness=1.0, parent=f"L_fov_{name}")
                dpg.draw_line((mrcx, mrcy + hh), (mrcx, mrcy + hh + 0.1 * gv), color=RED, thickness=1.0, parent=f"L_fov_{name}")
            else:
                # Narrowest cam (main): crosshairs from each image edge to the pipper radius, so a
                # centred target's pipper circle connects them.
                dpg.draw_line((il, cy), (cx - PIP_R, cy), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                dpg.draw_line((ir, cy), (cx + PIP_R, cy), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                dpg.draw_line((cx, it), (cx, cy - PIP_R), color=RED, thickness=1.0, parent=f"L_ret_{name}")
                dpg.draw_line((cx, ib), (cx, cy + PIP_R), color=RED, thickness=1.0, parent=f"L_ret_{name}")

        # Target pipper: green where THIS view is doing the tracking (amber while coasting), red where
        # the target is tracked via the *other* camera (its direction mapped in via the pinhole tan-ratio).
        stt = ctrl['state']
        pip = pcol = None
        if stt and stt.get('tracking') and stt.get('target_px'):
            tr = stt.get('track_role')
            if tr == role:
                pip = T(stt['target_px'][0], stt['target_px'][1])
                pcol = (255, 180, 40, 185) if stt.get('mode') == 'coast' else (70, 230, 100, 185)
            elif tr and tr != role:
                optx = stt.get('optics') or {}
                src, sf, mf = cams.get(tr), optx.get(tr), optx.get(role)
                if src is not None and sf and mf:
                    gtx, gty = stt['target_px'][0] * src['ox'], stt['target_px'][1] * src['oy']
                    sfx = (src['w'] / 2.0) / math.tan(math.radians(sf['fov_x_deg'] / 2.0))
                    sfy = (src['h'] / 2.0) / math.tan(math.radians(sf['fov_y_deg'] / 2.0))
                    mfx = (cam['w'] / 2.0) / math.tan(math.radians(mf['fov_x_deg'] / 2.0))
                    mfy = (cam['h'] / 2.0) / math.tan(math.radians(mf['fov_y_deg'] / 2.0))
                    mtx = cam['w'] / 2.0 + (gtx - src['w'] / 2.0) / sfx * mfx     # tan(angle) preserved across cams
                    mty = cam['h'] / 2.0 + (gty - src['h'] / 2.0) / sfy * mfy
                    pip, pcol = (offx + mtx * scale, offy + mty * scale), (255, 80, 80, 195)
        if pip is not None:
            # Clamp box = the visible camera view (image ∩ pane): the letterboxed image when zoomed
            # out, the pane itself when zoomed in past fit.
            box = (max(0.0, il), max(0.0, it), min(float(SW), ir), min(float(SH), ib))
            _draw_pipper(name, pip[0], pip[1], pcol, box)

        # Tracker ROI: the detect search window around the predicted target, drawn in the tracked view.
        if stt and stt.get('track_roi') and stt.get('track_role') == role:
            rcx, rcy, rsz = stt['track_roi']
            x0, y0 = T(rcx - rsz / 2.0, rcy - rsz / 2.0)
            x1, y1 = T(rcx + rsz / 2.0, rcy + rsz / 2.0)
            dpg.draw_rectangle((x0, y0), (x1, y1), color=(70, 230, 100, 150), thickness=1.0, parent=f"L_box_{name}")

        # Cut-off indicators: when zoomed past fit the image overflows -> arrows on the cropped edges.
        if dw > SW + 1:
            for ex, sx in ((-1, S(12)), (1, SW - S(12))):
                dpg.draw_triangle((sx, cy - S(9)), (sx, cy + S(9)), (sx + ex * S(11), cy),
                                  color=(255, 170, 40, 220), fill=(255, 170, 40, 170), parent=f"L_cut_{name}")
        if dh > SH + 1:
            for ey, sy in ((-1, S(12)), (1, SH - S(12))):
                dpg.draw_triangle((cx - S(9), sy), (cx + S(9), sy), (cx, sy + ey * S(11)),
                                  color=(255, 170, 40, 220), fill=(255, 170, 40, 170), parent=f"L_cut_{name}")

        # Histogram inset (toggle), bottom-right, fixed UI size (not zoomed) -- judge exposure/clipping.
        if sset['histogram'] and cam.get('hist') is not None:
            bars = cam['hist']
            HW, HH, mgn = min(S(180), max(S(60), SW - S(20))), S(70), S(10)
            hx1, hy1 = SW - mgn, SH - mgn
            hx0, hy0 = hx1 - HW, hy1 - HH
            dpg.draw_rectangle((hx0 - S(4), hy0 - S(4)), (hx1 + S(4), hy1 + S(4)), color=(0, 0, 0, 150),
                               fill=(0, 0, 0, 150), parent=f"L_hist_{name}")
            bw = HW / len(bars)
            for i, hgt in enumerate(bars):
                bx0 = hx0 + i * bw
                dpg.draw_rectangle((bx0, hy1 - float(hgt) * HH), (bx0 + bw, hy1), color=(205, 215, 235, 230),
                                   fill=(205, 215, 235, 230), parent=f"L_hist_{name}")
            dpg.draw_rectangle((hx0, hy0), (hx1, hy1), color=(180, 180, 180, 200), thickness=1.0,
                               parent=f"L_hist_{name}")

        # Status line (bottom-left) + a blinking NOT RECORDING warning if tracking without recording.
        st_now = ctrl['state'] or {}
        recording = bool((st_now.get('recording') or {}).get(role)) and bool(st_now.get('capturing', {}).get(role))
        status = (f"{role}  f{cam['last_idx']}  {_color_name(cam['color_id'])}  "
                  f"blobs {len(cam['blobs'])}  zoom {_zoom_label(zoom)}" + ("  REC" if recording else ""))
        dpg.draw_text((S(8), SH - S(20)), status, size=S(13), color=(200, 205, 220, 230), parent=f"L_warn_{name}")
        if st_now.get('tracking') and not recording and int(time.perf_counter() * 1.5) % 2 == 0:
            msg = "NOT RECORDING"
            dpg.draw_text((cx - len(msg) * S(40) * 0.30, cy + S(20)), msg, size=S(40),
                          color=(255, 40, 40, 255), parent=f"L_warn_{name}")

    # ---- click handling ------------------------------------------------------------------
    def _slot_at(mx, my):
        for name in _active_slots():
            rmin, rsz = _item_rect(f"dl_{name}")
            if rmin is None:
                continue
            if rmin[0] <= mx <= rmin[0] + rsz[0] and rmin[1] <= my <= rmin[1] + rsz[1]:
                return name, rmin, rsz
        return None, None, None

    def on_left_click():
        """Chrome (panel widgets, splitter, pane toolbars) handles its own clicks; otherwise a click
        in a pane locks the nearest blob (or the bare point) and tracks."""
        mx, my = dpg.get_mouse_pos(local=False)
        if _hit("win_panel", mx, my) or _hit("splitter", mx, my):
            return
        name, rmin, rsz = _slot_at(mx, my)
        if name is None:
            return
        lx, ly = mx - rmin[0], my - rmin[1]           # local to this pane's drawlist
        for _label, (x0, y0, x1, y1), action in _toolbar_layout(name):   # a toolbar button?
            if x0 <= lx <= x1 and y0 <= ly <= y1:
                action()
                return
        role = _slot_role(name)
        cam = cams.get(role)
        if cam is None:
            return
        SW, SH = rsz
        w, h = cam['w'], cam['h']
        scale = _floor_pow2(min(SW / w, SH / h) * 0.95) * view_settings.setdefault(role, _default_settings())['zoom']
        offx, offy = (SW - w * scale) / 2.0, (SH - h * scale) / 2.0
        fx = ((mx - rmin[0]) - offx) / scale / cam['ox']    # pane screen -> texture px -> frame (detect) px
        fy = ((my - rmin[1]) - offy) / scale / cam['oy']
        best, bd = None, 1e18
        for b in cam['blobs']:
            dx, dy = b['px'][0] - fx, b['px'][1] - fy
            d = dx * dx + dy * dy
            if d < bd:
                bd, best = d, b
        px = best['px'] if (best is not None and bd <= 40 * 40) else [fx, fy]
        _send({'type': 'track', 'role': role, 'px': [float(px[0]), float(px[1])]})

    def on_right_click():
        _send({'type': 'untrack'})

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_left_click)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=on_right_click)

    # ---- right settings/telemetry panel (docked; retires the old floating Control window) -----
    def _tip(text, item=None):
        """Attach a hover tooltip to `item` (default: the item just added)."""
        with dpg.tooltip(item if item is not None else dpg.last_item()):
            dpg.add_text(text, wrap=S(240))

    def _combo_row(label, items, tag, cb, parent=None, default='', right=0, group_tag=None):
        """A 'label: <dropdown>' row; the dropdown fills the remaining width (minus `right` px, when a
        trailing widget follows). parent=None -> the current container. group_tag tags the row's group
        so a caller can show/hide the whole row (label + dropdown together)."""
        kw = {'parent': parent} if parent else {}
        if group_tag:
            kw['tag'] = group_tag
        with dpg.group(horizontal=True, **kw):
            dpg.add_text(f"{label}:")
            dpg.add_combo(items, tag=tag, default_value=default, width=(-right if right else -1), callback=cb)

    def _toggle_connect(role):
        """Connect/Disconnect: fully start or stop this role's cam process. We drive it off the
        backend's actual capture state (telemetry), so the button is a plain toggle -- no local
        'intent' to get out of sync -- and connecting is always an explicit user action."""
        on = bool(((ctrl['state'] or {}).get('capturing') or {}).get(role))
        _send({'type': 'capture', 'role': role, 'on': not on})

    def render_camera_settings(role, parent):
        """One camera's connection/capture/display settings under `parent`. Widgets are intent;
        update_control reconciles capture/record to the backend, and the display toggles write
        view_settings."""
        sset = view_settings.setdefault(role, _default_settings())
        _combo_row("driver", ['synthetic', 'zwo', 'sky'], f"src_{role}",
                   lambda _s, a: _send({'type': 'set_source', 'role': role, 'source': a}), parent)
        _tip("Frame source: 'sky' = the ISS sim, 'synthetic' = a moving blob (no deps), 'zwo' = a real camera.")
        _combo_row("camera", ['(auto)'], f"chooser_{role}",
                   lambda _s, a: _on_camera_pick(role, a), parent, default='(auto)',
                   group_tag=f"camrow_{role}")
        _tip("Which physical ZWO camera to use (by model). '(auto)' = the first one. Press Rescan after plugging in.")
        dpg.add_group(tag=f"ctrls_{role}", parent=parent)   # caps-driven controls (exposure/gain/...), filled live
        dpg.add_button(label="Connect", tag=f"conn_{role}", parent=parent, user_data=role,
                       callback=lambda _s, _a, r: _toggle_connect(r))
        _tip("Start/stop this camera's capture process. Nothing connects until you press this -- pick "
             "the driver and (for zwo) the camera first, so two roles don't both grab '(auto)' and "
             "wedge the USB bus.")
        dpg.add_checkbox(label="Recording", tag=f"rec_{role}", parent=parent)
        _tip("Keep this camera's frames (mark them important so they aren't auto-deleted).")
        dpg.add_checkbox(label="Auto record", tag=f"autorec_{role}", parent=parent)
        _tip("Automatically record this camera whenever tracking is engaged.")
        dpg.add_separator(parent=parent)
        dpg.add_text("Display", parent=parent, color=(160, 170, 190))
        dpg.add_checkbox(label="Reticles", tag=f"ret_{role}", parent=parent, default_value=sset['reticles'],
                         callback=lambda _s, a: view_settings[role].__setitem__('reticles', a))
        _tip("Show the centre crosshairs + the main-cam FoV box on this camera's pane.")
        dpg.add_checkbox(label="Histogram", tag=f"hist_{role}", parent=parent, default_value=sset['histogram'],
                         callback=lambda _s, a: view_settings[role].__setitem__('histogram', a))
        _tip("Show a luminance histogram inset on this camera's pane (judge exposure/clipping).")

    # Optics: per-role sensor/optic/reducer pickers driven by the DB. Owned gear is pinned to the top
    # of each dropdown (before a divider) and remembered in the settings store.
    _SENS, _OPT, _RED = optics_db.load_db()
    _GEAR = {'sensor': sorted(_SENS), 'optic': sorted(_OPT), 'reducer': ['(none)'] + sorted(_RED)}
    owned = {'sensor': set(), 'optic': set(), 'reducer': set()}
    _DIV = '-' * 14                               # a (non-selectable) divider row in the combos

    def _gear_items(kind):
        own = [n for n in _GEAR[kind] if n in owned[kind]]
        rest = [n for n in _GEAR[kind] if n not in owned[kind]]
        return (own + [_DIV] + rest) if own else rest

    def _rebuild_gear(kind):                           # refresh both roles' combos of this kind, keep selection
        for r in roles:
            tag = f"opt_{r}_{kind}"
            if dpg.does_item_exist(tag):
                sel = dpg.get_value(tag)
                dpg.configure_item(tag, items=_gear_items(kind))
                dpg.set_value(tag, sel)

    def _send_optics(role):
        sen, opt = dpg.get_value(f"opt_{role}_sensor"), dpg.get_value(f"opt_{role}_optic")
        red = dpg.get_value(f"opt_{role}_reducer")
        if _DIV in (sen, opt, red):
            return
        _send({'type': 'set_optics', 'role': role, 'sensor': sen or None, 'optic': opt or None,
               'reducer': None if red in (None, '', '(none)') else red})

    def _on_gear_change(role, kind):
        tag = f"opt_{role}_{kind}"
        val = dpg.get_value(tag)
        if val == _DIV:                                # divider picked -> revert to the last valid choice
            dpg.set_value(tag, ctrl.get(tag, ''))
            return
        ctrl[tag] = val
        if dpg.does_item_exist(f"own_{role}_{kind}"):
            dpg.set_value(f"own_{role}_{kind}", val in owned[kind])
        _send_optics(role)

    def _toggle_owned(role, kind):
        val = dpg.get_value(f"opt_{role}_{kind}")
        if not val or val == _DIV:
            dpg.set_value(f"own_{role}_{kind}", False)
            return
        (owned[kind].add if dpg.get_value(f"own_{role}_{kind}") else owned[kind].discard)(val)
        _rebuild_gear(kind)

    def _automatch_optics(role, url):
        """When a ZWO camera whose model is a known DB sensor is picked, point this role's Optics
        sensor at it (so the plate scale matches the actual chip)."""
        if not (url and url.startswith('zwo:')):
            return
        model = url[len('zwo:'):].rsplit('#', 1)[0]
        if model in _SENS and dpg.does_item_exist(f"opt_{role}_sensor"):
            dpg.set_value(f"opt_{role}_sensor", model)
            ctrl[f"opt_{role}_sensor"] = model
            if dpg.does_item_exist(f"own_{role}_sensor"):
                dpg.set_value(f"own_{role}_sensor", model in owned['sensor'])
            _send_optics(role)

    def _on_camera_pick(role, url):
        _send({'type': 'set_camera', 'role': role, 'url': None if url in (None, '', '(auto)') else url})
        _automatch_optics(role, url)

    # Caps-driven camera controls (exposure/gain/...). The cam publishes each control's kind/range/value;
    # we render a "[<<][<] value [>][>>]" stepper per number control (single = small step, double = large;
    # log scale multiplies, linear scale adds) and push changes live. The GUI owns the value once shown
    # (the user is the source of truth), so we only rebuild when the *set* of controls changes -- e.g. the
    # source switched and a different camera's caps arrived.
    def _fmt_ctrl(v):
        return f"{v:.4g}"

    def _set_cam_ctrl(role, desc, value):
        value = min(desc['max'], max(desc['min'], float(value)))
        cam_ctrl_val[(role, desc['name'])] = value
        tag = f"cinp_{role}_{desc['name']}"
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, _fmt_ctrl(value))
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

    def _num_input(role, desc):
        try:
            v = float(dpg.get_value(f"cinp_{role}_{desc['name']}"))
        except (ValueError, TypeError):
            v = cam_ctrl_val.get((role, desc['name']), desc.get('value', 0.0))
        _set_cam_ctrl(role, desc, v)

    def _set_cam_choice(role, desc, value):            # relaunch-tier controls (binning/ROI): send as-is
        cam_ctrl_val[(role, desc['name'])] = value
        _send({'type': 'set_cam_control', 'role': role, 'name': desc['name'], 'value': value})

    def build_cam_controls(role, caps):
        """(Re)build a role's caps-driven control widgets into ctrls_<role>."""
        parent = f"ctrls_{role}"
        if not dpg.does_item_exist(parent):
            return
        dpg.delete_item(parent, children_only=True)
        for desc in (caps or {}).get('controls', []):
            cam_ctrl_val[(role, desc['name'])] = desc.get('value', 0.0)
            if desc.get('kind') == 'number':
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_text(f"{desc.get('label', desc['name'])}:")
                    for lbl, k in (('<<', 'ld'), ('<', 'sd')):
                        dpg.add_button(label=lbl, width=S(20), user_data=(role, desc, k),
                                       callback=lambda _s, _a, u: _num_step(u[0], u[1], u[2]))
                    dpg.add_input_text(tag=f"cinp_{role}_{desc['name']}", width=S(62), on_enter=True,
                                       default_value=_fmt_ctrl(desc.get('value', 0.0)),
                                       user_data=(role, desc), callback=lambda _s, _a, u: _num_input(u[0], u[1]))
                    for lbl, k in (('>', 'su'), ('>>', 'lu')):
                        dpg.add_button(label=lbl, width=S(20), user_data=(role, desc, k),
                                       callback=lambda _s, _a, u: _num_step(u[0], u[1], u[2]))
                    if desc.get('unit'):
                        dpg.add_text(desc['unit'], color=(140, 145, 160))
            elif desc.get('kind') == 'choice':         # relaunch-tier (binning/ROI): a dropdown; each pick relaunches
                with dpg.group(horizontal=True, parent=parent):
                    dpg.add_text(f"{desc.get('label', desc['name'])}:")
                    dpg.add_combo(desc.get('choices', []), tag=f"cinp_{role}_{desc['name']}",
                                  default_value=str(desc.get('value', '')), width=S(72),
                                  user_data=(role, desc), callback=lambda _s, a, u: _set_cam_choice(u[0], u[1], a))
            # (bool / file kinds -> later slices)

    # Settings persistence: gather the current settings into a JSON-able dict / apply a loaded one.
    # Grows as mount/camera/optics land; today it covers the layout + per-role display + optics prefs.
    def gather_settings():
        return {
            'version': 1,
            'layout': {k: layout[k] for k in ('panel_w', 'pip_h', 'panel_open', 'pip_open', 'pip_debug', 'big_role')},
            'display': {role: dict(view_settings.get(role, _default_settings())) for role in roles},
            'optics': {
                'owned': {k: sorted(v) for k, v in owned.items()},
                'selection': {role: [dpg.get_value(f"opt_{role}_{k}") for k in ('sensor', 'optic', 'reducer')]
                              for role in roles if dpg.does_item_exist(f"opt_{role}_sensor")},
            },
            'cameras': {role: dpg.get_value(f"chooser_{role}")
                        for role in roles if dpg.does_item_exist(f"chooser_{role}")},
        }

    def apply_settings(data):
        for k, v in (data.get('layout') or {}).items():
            if k in layout:
                layout[k] = v
        layout['_sig'] = None                          # force a relayout next frame
        for role, s in (data.get('display') or {}).items():
            vs = view_settings.setdefault(role, _default_settings())
            for k in ('zoom', 'reticles', 'histogram'):
                if k in s:
                    vs[k] = s[k]
            if dpg.does_item_exist(f"ret_{role}"):      # keep the panel checkboxes in sync
                dpg.set_value(f"ret_{role}", vs['reticles'])
            if dpg.does_item_exist(f"hist_{role}"):
                dpg.set_value(f"hist_{role}", vs['histogram'])
        opt = data.get('optics') or {}
        for k, names in (opt.get('owned') or {}).items():
            if k in owned:
                owned[k] = set(names)
        for k in owned:
            _rebuild_gear(k)
        for role, sel in (opt.get('selection') or {}).items():
            if not (sel and dpg.does_item_exist(f"opt_{role}_sensor")):
                continue
            for k, v in zip(('sensor', 'optic', 'reducer'), sel):
                if v and dpg.does_item_exist(f"opt_{role}_{k}"):
                    dpg.set_value(f"opt_{role}_{k}", v)
                    ctrl[f"opt_{role}_{k}"] = v
                    if dpg.does_item_exist(f"own_{role}_{k}"):
                        dpg.set_value(f"own_{role}_{k}", v in owned[k])
            _send_optics(role)                          # push the loaded optics to the backend
        for role, url in (data.get('cameras') or {}).items():
            if url and dpg.does_item_exist(f"chooser_{role}"):
                dpg.set_value(f"chooser_{role}", url if url in (ctrl.get('cam_items') or ['(auto)']) else '(auto)')
                _on_camera_pick(role, dpg.get_value(f"chooser_{role}"))   # push the loaded camera to the backend

    def _settings_refresh(select=None):
        dpg.configure_item('settings_combo', items=settings_store.list_settings())
        if select is not None:
            dpg.set_value('settings_combo', select)

    def _settings_save():
        name = (dpg.get_value('settings_name') or dpg.get_value('settings_combo') or '').strip()
        if name:
            _settings_refresh(settings_store.save(name, gather_settings()))
            dpg.set_value('settings_name', '')

    def _settings_load():
        if dpg.get_value('settings_combo'):
            apply_settings(settings_store.load(dpg.get_value('settings_combo')))

    def _settings_delete():
        if dpg.get_value('settings_combo'):
            settings_store.delete(dpg.get_value('settings_combo'))
            _settings_refresh('')

    with dpg.window(tag="win_panel", no_title_bar=True, no_move=True, no_resize=True, no_collapse=True):
        with dpg.collapsing_header(label="Mount", default_open=True):
            _combo_row("mount", ['Simulated mount'], 'mount_combo', None, default='Simulated mount')
            _tip("Which mount to drive. Live connect (real Celestron / Stellarium) is still WIP.")
            # 2D slew pad: a log-scaled az/alt rate plane. Drag = drive the mount (momentary override
            # of tracking); the circle shows the current rate (readout in update_control).
            dpg.add_text("Slew", color=(160, 170, 190))     # a drawlist can't host a tooltip; label it
            _tip(f"Drag the pad to drive the mount (log scale, max {SLEW_MAX:g} deg/s; Az = right, "
                 f"Alt = up). Centre = stop. Momentarily overrides tracking, resumes on release.")
            with dpg.drawlist(width=S(200), height=S(200), tag='slew_pad'):
                pad_bg = dpg.add_draw_layer()
                pad_fg = dpg.add_draw_layer()
            _P, _c, _H = S(200), S(100), S(100)                  # static grid (log scale)
            dpg.draw_rectangle((1, 1), (_P - 1, _P - 1), color=(80, 86, 100, 220),
                               fill=(18, 20, 26, 220), parent=pad_bg)     # box; border edge = SLEW_MAX
            for _g in SLEW_GRID:
                _d = _rate_to_u(_g) * _H
                for _s in (-1, 1):
                    dpg.draw_line((_c + _s * _d, 2), (_c + _s * _d, _P - 2), color=(80, 86, 100, 110), parent=pad_bg)
                    dpg.draw_line((2, _c + _s * _d), (_P - 2, _c + _s * _d), color=(80, 86, 100, 110), parent=pad_bg)
            dpg.draw_line((_c, 2), (_c, _P - 2), color=(150, 156, 172, 220), thickness=1.5, parent=pad_bg)  # az=0
            dpg.draw_line((2, _c), (_P - 2, _c), color=(150, 156, 172, 220), thickness=1.5, parent=pad_bg)  # alt=0
        with dpg.collapsing_header(label="Cameras", default_open=True):
            dpg.add_button(label="Rescan", callback=lambda: _send({'type': 'rescan_cameras'}))
            _tip("Re-enumerate attached ZWO cameras (after plugging one in).")
            for role in roles:
                with dpg.collapsing_header(label=role.capitalize(), default_open=True) as hdr:
                    pass
                render_camera_settings(role, hdr)
        with dpg.collapsing_header(label="Optics"):
            dpg.add_text("Sensor + optic set the pixel scale.", color=(150, 150, 150), wrap=S(210))
            for role in roles:
                dpg.add_separator()
                dpg.add_text(role.capitalize(), color=(160, 170, 190))
                for kind, has_owned in (('sensor', True), ('optic', True), ('reducer', False)):
                    with dpg.group(horizontal=True):
                        dpg.add_text(f"{kind}:")
                        dpg.add_combo(_gear_items(kind), tag=f"opt_{role}_{kind}",
                                      width=(-S(30) if has_owned else -1), user_data=(role, kind),
                                      callback=lambda _s, _a, u: _on_gear_change(u[0], u[1]))
                        _tip(f"{role.capitalize()} {kind}. Owned gear is pinned to the top of the list.")
                        if has_owned:
                            dpg.add_checkbox(tag=f"own_{role}_{kind}", user_data=(role, kind),
                                             callback=lambda _s, _a, u: _toggle_owned(u[0], u[1]))
                            _tip("I own this — pin it to the top of the list (in every dropdown).")
        with dpg.collapsing_header(label="Settings"):
            dpg.add_combo(settings_store.list_settings(), tag='settings_combo', width=-1)
            _tip("A saved settings file captures the layout, display prefs, optics + owned gear, and cameras.")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load", callback=_settings_load)
                dpg.add_button(label="Delete", callback=_settings_delete)
                dpg.add_button(label="Show folder", callback=lambda: _open_folder(settings_store.settings_dir()))
                _tip("Open the settings folder in the file browser.")
            dpg.add_spacer(height=S(4))
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag='settings_name', hint="save as...", width=-S(56))
                dpg.add_button(label="Save", callback=_settings_save)

    def update_control():
        # Connect to the backend command socket once its port file appears.
        if ctrl['client'] is None:
            bj = _newest(args.session, '_backend.json')
            if bj:
                try:
                    info = json.load(open(bj))
                    ctrl['client'] = control.CommandClient(info['command_host'], info['command_port'])
                except (OSError, ValueError, KeyError):
                    ctrl['client'] = None
        if ctrl['tailer'] is None:
            sp = _newest(args.session, '_state.jsonl')
            if sp:
                ctrl['tailer'] = JsonlTailer(sp)
        if ctrl['tailer'] is not None:
            for rec in ctrl['tailer'].poll():
                ctrl['state'] = rec
        st = ctrl['state']
        rec_st = (st or {}).get('recording') or {}     # per-role: {role: bool}
        cap_st = (st or {}).get('capturing') or {}
        src_st = (st or {}).get('sources') or {}
        tracking_st = bool((st or {}).get('tracking'))
        rec_sent = ctrl.setdefault('rec_sent', {})
        src_init = ctrl.setdefault('src_init', set())
        autorec_init = ctrl.setdefault('autorec_init', set())
        for role in roles:
            if st is None or not dpg.does_item_exist(f"src_{role}"):
                continue
            # One-time init of the intent widgets from the backend's actual state.
            if role not in src_init and src_st.get(role):
                dpg.set_value(f"src_{role}", src_st[role]); src_init.add(role)
            if role not in autorec_init and src_st.get(role):     # default ON for a real cam, off for sim
                dpg.set_value(f"autorec_{role}", src_st[role] == 'zwo'); autorec_init.add(role)
            # Connect/Disconnect: the button label just mirrors the backend's actual capture state;
            # the click (in _toggle_connect) starts/stops the cam. Explicit only -- no auto-connect.
            if dpg.does_item_exist(f"conn_{role}"):
                dpg.configure_item(f"conn_{role}", label="Disconnect" if cap_st.get(role) else "Connect")
            # The camera chooser only applies to the zwo driver; hide the whole row otherwise (it lists
            # the attached ZWO cameras, which mean nothing for the sim / synthetic sources).
            if dpg.does_item_exist(f"camrow_{role}"):
                dpg.configure_item(f"camrow_{role}", show=(dpg.get_value(f"src_{role}") == 'zwo'))
            # Recording = the manual box OR (Auto-record AND tracking), per camera.
            want_rec = bool(dpg.get_value(f"rec_{role}")) or (bool(dpg.get_value(f"autorec_{role}")) and tracking_st)
            if want_rec != bool(rec_st.get(role)) and want_rec != rec_sent.get(role):
                _send({'type': 'record', 'role': role, 'on': want_rec}); rec_sent[role] = want_rec

        # One-time init of the Optics tab dropdowns from the backend's current selection.
        opt_sel = (st or {}).get('optics_sel') or {}
        opt_init = ctrl.setdefault('opt_init', set())
        for role in roles:
            if role in opt_init or role not in opt_sel or not dpg.does_item_exist(f"opt_{role}_sensor"):
                continue
            sen, opt_, red = (list(opt_sel[role]) + [None, None, None])[:3]
            for k, v in (('sensor', sen), ('optic', opt_), ('reducer', red or '(none)')):
                dpg.set_value(f"opt_{role}_{k}", v or '')
                ctrl[f"opt_{role}_{k}"] = v or ''
                if dpg.does_item_exist(f"own_{role}_{k}"):
                    dpg.set_value(f"own_{role}_{k}", bool(v) and v in owned[k])
            opt_init.add(role)

        # Camera chooser: keep dropdown items in sync with the detected cameras + init selection once.
        cam_items = ['(auto)'] + ((st or {}).get('cameras_available') or [])
        if ctrl.get('cam_items') != cam_items:
            ctrl['cam_items'] = cam_items
            for role in roles:
                if dpg.does_item_exist(f"chooser_{role}"):
                    sel = dpg.get_value(f"chooser_{role}")
                    dpg.configure_item(f"chooser_{role}", items=cam_items)
                    dpg.set_value(f"chooser_{role}", sel if sel in cam_items else '(auto)')
        cam_sel_st = (st or {}).get('camera') or {}
        cam_init = ctrl.setdefault('cam_init', set())
        for role in roles:
            if role in cam_init or st is None or not dpg.does_item_exist(f"chooser_{role}"):
                continue
            val = cam_sel_st.get(role)
            dpg.set_value(f"chooser_{role}", val if val in cam_items else '(auto)')
            cam_init.add(role)

        # Caps-driven camera controls: rebuild a role's stepper widgets whenever the *set* of controls
        # changes (source switch -> a different camera's caps). Values track the GUI's own state after that.
        caps_st = (st or {}).get('camera_caps') or {}
        ctrl_sig = ctrl.setdefault('ctrl_sig', {})
        for role in roles:
            caps = caps_st.get(role)
            sig = (caps.get('source'), tuple(c['name'] for c in caps.get('controls', []))) if caps else None
            if sig != ctrl_sig.get(role) and dpg.does_item_exist(f"ctrls_{role}"):
                build_cam_controls(role, caps)
                ctrl_sig[role] = sig

        # 2D slew pad: drag inside it to drive the mount at a log-scaled az/alt rate (centre dead-zone
        # = zero). Momentarily overrides tracking, and resumes it on release. Drag latches, so it
        # keeps driving even if the cursor leaves the pad (release to stop).
        pad_rmin, pad_rsz = _item_rect('slew_pad')
        mx, my = dpg.get_mouse_pos(local=False)
        over = (pad_rmin is not None and pad_rmin[0] <= mx <= pad_rmin[0] + pad_rsz[0]
                and pad_rmin[1] <= my <= pad_rmin[1] + pad_rsz[1])
        if (dpg.is_mouse_button_down(dpg.mvMouseButton_Left) and pad_rmin is not None
                and (ctrl.get('slew_active') or over)):
            if not ctrl.get('slew_active'):          # drag start -> remember a track to resume on release
                ctrl['slew_active'] = True
                ctrl['slew_resume'] = ((st or {}).get('track_role'), (st or {}).get('target_px')) \
                    if (st and st.get('tracking')) else None
            hx, hy = pad_rsz[0] / 2.0, pad_rsz[1] / 2.0
            az = _u_to_rate((mx - (pad_rmin[0] + hx)) / max(1.0, hx))
            alt = _u_to_rate(-(my - (pad_rmin[1] + hy)) / max(1.0, hy))       # screen y down -> alt up
            if ctrl['client'] is not None and (az, alt) != ctrl['last_rate']:
                ctrl['client'].send({'type': 'set_rate', 'az': az, 'alt': alt})
                ctrl['last_rate'] = (az, alt)
        elif ctrl.get('slew_active'):                # released -> stop, then resume any prior track
            ctrl['slew_active'] = False
            if ctrl['client'] is not None:
                ctrl['client'].send({'type': 'stop'})
                ctrl['last_rate'] = (0.0, 0.0)
                res = ctrl.get('slew_resume')
                if res and res[0] and res[1]:
                    ctrl['client'].send({'type': 'track', 'role': res[0],
                                         'px': [float(res[1][0]), float(res[1][1])]})
            ctrl['slew_resume'] = None

        # Current-rate circle on the pad (drawlist-local coords), from telemetry.
        dpg.delete_item(pad_fg, children_only=True)
        if st and pad_rsz is not None:
            hx, hy = pad_rsz[0] / 2.0, pad_rsz[1] / 2.0
            cxp = hx + _rate_to_u(st.get('rate_az_deg_s', 0.0)) * hx
            cyp = hy - _rate_to_u(st.get('rate_alt_deg_s', 0.0)) * hy
            dpg.draw_circle((cxp, cyp), S(6), color=(70, 230, 100, 235), thickness=2.0, parent=pad_fg)

    # ---- layout: position everything from the viewport size ------------------------------
    def relayout():
        vw = max(S(200), dpg.get_viewport_client_width())
        vh = max(S(200), dpg.get_viewport_client_height())
        bm = S(6)                                    # bottom margin: keep panes off the viewport edge
        pw = layout['panel_w'] if layout['panel_open'] else 0
        vsp = S(6) if layout['panel_open'] else 0
        hsp = S(6) if layout['pip_open'] else 0
        usable_h = vh - bm
        left_w = max(S(120), vw - pw - vsp)
        ph = layout['pip_h'] if layout['pip_open'] else 0
        ph = max(0, min(ph, usable_h - hsp - S(200)))   # keep the big pane >= ~200 tall
        big_h = usable_h - hsp - ph                      # big_h + hsp + ph == usable_h (== vh - bm)
        # Inset the drawlists a few px inside their windows: a dpg window's content region can be a
        # touch smaller than its frame, and a drawlist sized to the full window gets its far edge clipped.
        inx, iny = S(4), S(6)
        dpg.configure_item("slot_big", pos=(0, 0), width=left_w, height=big_h)
        dpg.configure_item("dl_big", width=max(S(40), left_w - inx), height=max(S(40), big_h - iny))
        if layout['pip_open'] and ph > S(20):
            dpg.configure_item("hsplitter", show=True, pos=(0, big_h), width=left_w, height=hsp)
            dpg.configure_item("hsplitter_btn", width=left_w, height=hsp)
            pipw = max(S(80), min(left_w, int(ph * 16 / 9)))
            dpg.configure_item("slot_pipother", show=True, pos=(0, big_h + hsp), width=pipw, height=ph)
            dpg.configure_item("dl_pipother", width=max(S(40), pipw - inx), height=max(S(40), ph - iny))
        else:
            dpg.configure_item("hsplitter", show=False)
            dpg.configure_item("slot_pipother", show=False)
        if layout['panel_open']:
            dpg.configure_item("win_panel", show=True, pos=(vw - pw, 0), width=pw, height=vh)
            dpg.configure_item("splitter", show=True, pos=(vw - pw - vsp, 0), width=vsp, height=vh)
            dpg.configure_item("splitter_btn", width=vsp, height=vh)
        else:
            dpg.configure_item("win_panel", show=False)
            dpg.configure_item("splitter", show=False)

    dpg.create_viewport(title="AstroLock Seeker", width=S(1400), height=S(900))
    dpg.setup_dearpygui()
    try:
        dpg.set_exit_callback(_shutdown)    # fires the instant the window is closed
    except Exception:
        pass                                # older dpg: fall back to the loop-end + poll paths
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        vw = max(S(200), dpg.get_viewport_client_width())
        vh = max(S(200), dpg.get_viewport_client_height())
        # Divider drags: while a handle is held, resize the panel / PIP strip to follow the cursor.
        if layout['panel_open'] and dpg.is_item_active("splitter_btn"):
            mx, _ = dpg.get_mouse_pos(local=False)
            layout['panel_w'] = int(max(S(PANEL_MIN_W), min(vw - S(320), vw - mx)))
        if layout['pip_open'] and dpg.is_item_active("hsplitter_btn"):
            _, my = dpg.get_mouse_pos(local=False)
            layout['pip_h'] = int(max(S(120), min(vh - S(300), (vh - S(6)) - S(6) - my)))
        sig = (vw, vh, layout['panel_open'], layout['pip_open'], layout['panel_w'], layout['pip_h'])
        if sig != layout['_sig']:
            relayout()
            layout['_sig'] = sig

        update_control()

        new_work = False
        for role in roles:
            if update_cam(role):
                new_work = True
        for name in _active_slots():                     # advance any debug surface a pip is showing
            stream = _slot_stream(name)
            if stream not in roles and update_cam(stream):
                new_work = True
        for name in _active_slots():
            draw_slot(name)

        dpg.render_dearpygui_frame()
        if not new_work:
            time.sleep(0.005)            # idle: keep UI responsive without pegging a core

    _shutdown()      # belt-and-suspenders: if the exit callback didn't fire, still exit immediately


if __name__ == '__main__':
    main()
