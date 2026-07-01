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
import sys
import time

import numpy as np
import torch

from astrolock.seeker import bayer, control, ser
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
    layout = {'panel_open': True, 'pip_open': True, 'panel_w': S(PANEL_W), 'pip_h': S(200),
              'big_role': ROLES[0], '_sig': None}

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
            tex = f"tex_{role}"
            if not dpg.does_item_exist(tex):
                with dpg.texture_registry():
                    dpg.add_raw_texture(w, h, np.zeros(w * h * 4, dtype=np.float32),
                                        format=dpg.mvFormat_Float_rgba, tag=tex)
            det_path = f.ser_path[:-len('.ser')] + '.detections.jsonl'
            cam = cams[role] = dict(tex=tex, w=w, h=h, fw=fw, fh=fh, ox=w / fw, oy=h / fh,
                                    color_id=f.header.color_id, blobs=[], det_idx=-1, last_idx=-1,
                                    peak=0, hist=None, det_tailer=JsonlTailer(det_path), ser_path=f.ser_path)
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
        if (w, h) != (cam['w'], cam['h']):          # frame size changed (source switch) -> new texture
            if dpg.does_item_exist(cam['tex']):
                dpg.delete_item(cam['tex'])
            with dpg.texture_registry():
                dpg.add_raw_texture(w, h, np.zeros(w * h * 4, dtype=np.float32),
                                    format=dpg.mvFormat_Float_rgba, tag=cam['tex'])
            cam.update(w=w, h=h, fw=fw, fh=fh, ox=w / fw, oy=h / fh, color_id=f.header.color_id)
        dpg.set_value(cam['tex'], rgba.ravel())
        cam['last_idx'], cam['peak'] = idx, int(frame.max())
        # Luminance histogram of the *displayed* image (WYSIWYG); subsampled + sqrt-scaled.
        samp = rgba[::4, ::4, :3].mean(axis=2)
        counts, _ = np.histogram(samp, bins=64, range=(0.0, 1.0))
        m = counts.max()
        cam['hist'] = np.sqrt(counts / m) if m > 0 else None
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
        """Draw the slot's assigned role letterboxed + centred, with overlays, at the pane's size."""
        role = _slot_role(name)
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

        # Reticles (toggle): boresight crosshair at the pane centre + nested narrower-cam FoV rects.
        if sset['reticles']:
            rcol, cr = (0, 220, 220, 200), S(10)
            dpg.draw_circle((cx, cy), cr, color=rcol, thickness=1.0, parent=f"L_ret_{name}")
            for ex, ey in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                dpg.draw_line((cx + ex * cr * 2.2, cy + ey * cr * 2.2),
                              (cx + ex * cr * 0.8, cy + ey * cr * 0.8), color=rcol,
                              thickness=1.0, parent=f"L_ret_{name}")
            optx = (ctrl['state'] or {}).get('optics', {})
            me = optx.get(role)
            if me:
                for r2, fv2 in optx.items():
                    if r2 == role or not (fv2['fov_x_deg'] < me['fov_x_deg']
                                          and fv2['fov_y_deg'] < me['fov_y_deg']):
                        continue
                    hw = math.tan(math.radians(fv2['fov_x_deg'] / 2)) / \
                        math.tan(math.radians(me['fov_x_deg'] / 2)) * (dw / 2.0)
                    hh = math.tan(math.radians(fv2['fov_y_deg'] / 2)) / \
                        math.tan(math.radians(me['fov_y_deg'] / 2)) * (dh / 2.0)
                    col2 = (120, 180, 255, 220)
                    dpg.draw_rectangle((cx - hw, cy - hh), (cx + hw, cy + hh), color=col2,
                                       thickness=1.0, parent=f"L_fov_{name}")
                    dpg.draw_text((cx - hw, cy - hh - S(14)), r2, size=S(13), color=col2, parent=f"L_fov_{name}")

        # Locked-target marker (magenta; amber while coasting). A cursor -> constant size, position zooms.
        stt = ctrl['state']
        if stt and stt.get('tracking') and stt.get('track_role') == role and stt.get('target_px'):
            X, Y = T(stt['target_px'][0], stt['target_px'][1])
            col = (255, 180, 40, 255) if stt.get('mode') == 'coast' else (255, 60, 220, 255)
            r = S(14)
            dpg.draw_circle((X, Y), r, color=col, thickness=1.0, parent=f"L_trk_{name}")
            for ex, ey in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                dpg.draw_line((X + ex * (r + S(6)), Y + ey * (r + S(6))), (X + ex * S(4), Y + ey * S(4)),
                              color=col, thickness=1.0, parent=f"L_trk_{name}")

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
        status = (f"{role}  f{cam['last_idx']}  {_color_name(cam['color_id'])}  peak {cam['peak']}  "
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
    def render_camera_settings(role, parent):
        """One camera's connection/capture/display settings under `parent` -- a standalone helper so
        the same block could later be dropped into a pane. Widgets are intent; update_control
        reconciles capture/record to the backend, and the display toggles write view_settings."""
        sset = view_settings.setdefault(role, _default_settings())
        dpg.add_combo(['synthetic', 'zwo', 'sky'], tag=f"src_{role}", parent=parent, width=S(120),
                      label="driver",
                      callback=lambda _s, a, role=role: _send({'type': 'set_source', 'role': role, 'source': a}))
        dpg.add_combo(['(auto)'], default_value='(auto)', tag=f"chooser_{role}", parent=parent,
                      width=S(120), label="camera", enabled=False)   # physical-cam chooser: enumeration TBD
        dpg.add_checkbox(label="Running", tag=f"run_{role}", parent=parent)
        dpg.add_checkbox(label="Recording", tag=f"rec_{role}", parent=parent)
        dpg.add_checkbox(label="Auto record", tag=f"autorec_{role}", parent=parent)
        dpg.add_separator(parent=parent)
        dpg.add_text("Display", parent=parent, color=(160, 170, 190))
        dpg.add_checkbox(label="Reticles", tag=f"ret_{role}", parent=parent, default_value=sset['reticles'],
                         callback=lambda _s, a, role=role: view_settings[role].__setitem__('reticles', a))
        dpg.add_checkbox(label="Histogram", tag=f"hist_{role}", parent=parent, default_value=sset['histogram'],
                         callback=lambda _s, a, role=role: view_settings[role].__setitem__('histogram', a))
        dpg.add_text("zoom: +/- on the pane", parent=parent, color=(120, 122, 132))

    with dpg.window(tag="win_panel", no_title_bar=True, no_move=True, no_resize=True, no_collapse=True):
        state_text = dpg.add_text("backend: connecting...")
        dpg.add_separator()
        dpg.add_text("Cameras", color=(160, 170, 190))
        for role in roles:
            with dpg.collapsing_header(label=role.capitalize(), default_open=True) as hdr:
                pass
            render_camera_settings(role, hdr)
        dpg.add_separator()
        dpg.add_text("Slew", color=(160, 170, 190))
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=S(72))
            btn_alt_up = dpg.add_button(label="Alt +", width=S(64), height=S(40))
        with dpg.group(horizontal=True):
            btn_az_dn = dpg.add_button(label="Az -", width=S(64), height=S(40))
            btn_stop = dpg.add_button(label="Stop", width=S(64), height=S(40))
            btn_az_up = dpg.add_button(label="Az +", width=S(64), height=S(40))
        with dpg.group(horizontal=True):
            dpg.add_spacer(width=S(72))
            btn_alt_dn = dpg.add_button(label="Alt -", width=S(64), height=S(40))
        dpg.add_text("hold a direction to slew", color=(150, 150, 150))
        dpg.add_text("click a pane to track, right-click to stop", color=(150, 150, 150))

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
        cap_sent = ctrl.setdefault('cap_sent', {})
        rec_sent = ctrl.setdefault('rec_sent', {})
        src_init = ctrl.setdefault('src_init', set())
        run_init = ctrl.setdefault('run_init', set())
        autorec_init = ctrl.setdefault('autorec_init', set())
        for role in roles:
            if st is None or not dpg.does_item_exist(f"src_{role}"):
                continue
            # One-time init of the intent widgets from the backend's actual state.
            if role not in src_init and src_st.get(role):
                dpg.set_value(f"src_{role}", src_st[role]); src_init.add(role)
            if role not in run_init and role in cap_st:
                dpg.set_value(f"run_{role}", bool(cap_st[role])); run_init.add(role)
            if role not in autorec_init and src_st.get(role):     # default ON for a real cam, off for sim
                dpg.set_value(f"autorec_{role}", src_st[role] == 'zwo'); autorec_init.add(role)
            # Running = capture on/off; reconcile the checkbox intent (debounced) to the backend.
            want_run = bool(dpg.get_value(f"run_{role}"))
            if want_run != bool(cap_st.get(role)) and want_run != cap_sent.get(role):
                _send({'type': 'capture', 'role': role, 'on': want_run}); cap_sent[role] = want_run
            # Recording = the manual box OR (Auto-record AND tracking), per camera.
            want_rec = bool(dpg.get_value(f"rec_{role}")) or (bool(dpg.get_value(f"autorec_{role}")) and tracking_st)
            if want_rec != bool(rec_st.get(role)) and want_rec != rec_sent.get(role):
                _send({'type': 'record', 'role': role, 'on': want_rec}); rec_sent[role] = want_rec

        if st:
            caps = ' '.join(f"{r}:{'on' if v else 'off'}" for r, v in cap_st.items())
            recs = ' '.join(f"{r}:{'on' if v else 'off'}" for r, v in rec_st.items())
            dpg.set_value(state_text,
                          f"{st.get('mode', '?')}   az {st.get('enc_az_deg', 0):.2f}   "
                          f"alt {st.get('enc_alt_deg', 0):.2f}\n"
                          f"rate {st.get('rate_az_deg_s', 0):.2f}, {st.get('rate_alt_deg_s', 0):.2f} deg/s\n"
                          f"record [{recs}]   capture [{caps}]")
        else:
            dpg.set_value(state_text,
                          "backend: connecting..." if ctrl['client'] is None else "backend: waiting for state")

        # Press-and-hold slew: send the rate implied by the currently-held buttons, on change.
        sr = args.slew_rate
        if dpg.is_item_active(btn_stop):
            az = alt = 0.0
        else:
            az = (sr if dpg.is_item_active(btn_az_up) else 0.0) - (sr if dpg.is_item_active(btn_az_dn) else 0.0)
            alt = (sr if dpg.is_item_active(btn_alt_up) else 0.0) - (sr if dpg.is_item_active(btn_alt_dn) else 0.0)
        if ctrl['client'] is not None and (az, alt) != ctrl['last_rate']:
            ctrl['client'].send({'type': 'set_rate', 'az': az, 'alt': alt})
            ctrl['last_rate'] = (az, alt)

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
        for name in _active_slots():
            draw_slot(name)

        dpg.render_dearpygui_frame()
        if not new_work:
            time.sleep(0.005)            # idle: keep UI responsive without pegging a core

    _shutdown()      # belt-and-suspenders: if the exit callback didn't fire, still exit immediately


if __name__ == '__main__':
    main()
