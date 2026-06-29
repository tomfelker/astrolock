"""
astrolock_seeker_gui: Dear PyGui viewer.

Hello-world scope: follow each role's .ser in a session directory and show the live (or
replayed) frame, updating every render frame. Same code path for live tailing and
historical playback -- it's just a follower reading growing-or-complete files.

Bayer frames are debayered to half-res RGB for display (the 4-plane split, see bayer.py);
mono frames are shown as grayscale. Each frame maps the full container range to [0,1] (no
per-frame auto-stretch -- brightness stays stable, so expose/gain the camera) then applies
gamma. If a <role>.detections.jsonl is present, candidate blobs are overlaid as square boxes
(green = moving, amber = static) in the frame's image space, so they align directly.

    python -m astrolock.seeker.gui --session sessions/<ts>
    python -m astrolock.seeker.gui --session sessions/<ts> --roles guide,main

Requires `dearpygui` (pip install dearpygui).
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

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


def discover_roles(session_dir):
    roles = []
    for path in sorted(glob.glob(os.path.join(session_dir, '*_*.ser'))):
        stem = os.path.basename(path)[:-len('.ser')]  # filenames are <ts>_<role>.ser
        roles.append(stem.split('_', 1)[1])
    return roles


def prepare_rgba(frame_raw, color_id, gamma, wb=(1.0, 1.0)):
    """
    Raw frame (mosaic or mono) -> (w, h, (h,w,4) float32 RGBA).
    Debayers Bayer frames to half-res RGB; applies display-only WB (R,B gains -- the stored
    data stays pristine raw); maps the full container range to [0,1] with NO per-frame
    auto-stretch (so brightness is stable -- expose/gain the camera instead); applies gamma.
    """
    white = (float(np.iinfo(frame_raw.dtype).max)
             if np.issubdtype(frame_raw.dtype, np.integer) else 1.0)
    if bayer.is_bayer(color_id):
        rgb = bayer.debayer_to_rgb(frame_raw, color_id)          # (h/2, w/2, 3)
        rgb[..., 0] *= wb[0]                                      # display WB on R
        rgb[..., 2] *= wb[1]                                      # display WB on B
    else:
        f = frame_raw.astype(np.float32)
        rgb = np.repeat(f[:, :, None], 3, axis=2)

    norm = rgb / white
    if gamma and gamma != 1.0:
        norm = np.clip(norm, 0.0, 1.0) ** (1.0 / gamma)

    h, w = norm.shape[:2]
    rgba = np.empty((h, w, 4), dtype=np.float32)
    rgba[..., 0:3] = np.clip(norm, 0.0, 1.0)
    rgba[..., 3] = 1.0
    return w, h, rgba


def draw_box(rgba, cx, cy, half, color):
    """Draw a hollow square (work-image coords) into an (h,w,4) rgba array."""
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


def main(argv=None):
    p = argparse.ArgumentParser(description="AstroLock Seeker GUI viewer")
    p.add_argument('--session', required=True, help="session directory to view")
    p.add_argument('--roles', default=None, help="comma-separated roles (default: auto-detect)")
    p.add_argument('--display-width', type=int, default=640, help="on-screen width per view")
    p.add_argument('--gamma', type=float, default=2.2, help="display gamma (1 = linear)")
    p.add_argument('--wb-r', type=float, default=1.24, help="display-only WB gain for red")
    p.add_argument('--wb-b', type=float, default=1.98, help="display-only WB gain for blue")
    p.add_argument('--slew-rate', type=float, default=3.0, help="slew rate while a button is held (deg/s)")
    p.add_argument('--ui-scale', type=float, default=0.0,
                   help="UI/DPI scale factor (0 = auto-detect from the OS; e.g. 1.5 for a 150%% display)")
    args = p.parse_args(argv)
    wb = (args.wb_r, args.wb_b)

    import dearpygui.dearpygui as dpg

    # Declare per-monitor DPI awareness *before* the viewport exists, so Windows gives us a
    # native-resolution framebuffer instead of bitmap-upscaling a low-res window (which blurs
    # the text). Then read the monitor scale so we can size the UI up to match -- without that
    # the text renders crisp but tiny. Must precede create_context.
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

    fixed_roles = ([r.strip() for r in args.roles.split(',') if r.strip()]
                   if args.roles else None)
    followers = {}
    views = {}      # role -> dict(tex, status, det_tailer, blobs, ...)
    view_pos = {}   # role -> [x, y], remembered across rebuilds so panes don't jump

    dpg.create_context()
    dpg.set_global_font_scale(ui_scale)   # crisp text at the right size (ImGui 1.92 re-rasterizes)

    def ensure_view(role):
        """Create a view lazily, once a frame exists (we need its post-debayer size)."""
        if role in views:
            return
        f = followers.get(role) or followers.setdefault(role, SerFollower(args.session, role))
        res = f.read_latest()
        if res is None or f.header is None:
            return
        _, frame = res
        fh, fw = frame.shape[0], frame.shape[1]       # full frame size (detect coords' space)
        w, h, _rgba = prepare_rgba(frame, f.header.color_id, args.gamma, wb)
        tex_tag = f"tex_{role}"
        with dpg.texture_registry():
            dpg.add_raw_texture(w, h, np.zeros(w * h * 4, dtype=np.float32),
                                format=dpg.mvFormat_Float_rgba, tag=tex_tag)
        disp_w = S(args.display_width)
        disp_h = int(round(disp_w * h / w))
        win_kwargs = {'pos': view_pos[role]} if role in view_pos else {}
        with dpg.window(label=role, tag=f"win_{role}", width=disp_w + S(30), height=disp_h + S(70),
                        **win_kwargs):
            status = dpg.add_text("waiting...")
            # Draw the image and overlay boxes as crisp vector rectangles (Dear PyGui draws
            # them at display resolution, so no aliasing from the downscaled texture).
            with dpg.drawlist(width=disp_w, height=disp_h) as drawlist:
                dpg.draw_image(tex_tag, (0, 0), (disp_w, disp_h))
                box_layer = dpg.add_draw_layer()
                track_layer = dpg.add_draw_layer()      # the locked-target marker (on top)
        det_path = f.ser_path[:-len('.ser')] + '.detections.jsonl'
        views[role] = dict(tex=tex_tag, status=status, det_tailer=JsonlTailer(det_path),
                           ser_path=f.ser_path, blobs=[], box_layer=box_layer,
                           track_layer=track_layer, drawlist=drawlist,
                           ox=disp_w / fw, oy=disp_h / fh, w=w, h=h, last_idx=-1, det_idx=-1, peak=0)

    def rebuild_view(role):
        """Drop a role's view (window + texture) so ensure_view recreates it -- used when the
        frame dimensions change, e.g. switching source sky(1080) <-> synthetic(720)."""
        v = views.pop(role, None)
        if v is None:
            return
        v['det_tailer'].close()
        if dpg.does_item_exist(f"win_{role}"):
            try:
                view_pos[role] = dpg.get_item_pos(f"win_{role}")   # keep the pane in place
            except Exception:
                pass
            dpg.delete_item(f"win_{role}")
        if dpg.does_item_exist(v['tex']):
            dpg.delete_item(v['tex'])

    ctrl = {'client': None, 'tailer': None, 'state': None, 'last_rate': None}

    def _send(obj):
        if ctrl['client'] is not None:
            ctrl['client'].send(obj)

    def _view_at(mx, my):
        """Which view's image contains viewport point (mx,my)? -> (role, view, rect_min)."""
        for role, v in views.items():
            dl = v.get('drawlist')
            if dl is None or not dpg.does_item_exist(dl):
                continue
            rmin = dpg.get_item_rect_min(dl)
            rsz = dpg.get_item_rect_size(dl)
            if rmin[0] <= mx <= rmin[0] + rsz[0] and rmin[1] <= my <= rmin[1] + rsz[1]:
                return role, v, rmin
        return None, None, None

    def on_left_click():
        """Click the image to lock the nearest blob (or the bare click point) and start tracking."""
        mx, my = dpg.get_mouse_pos(local=False)
        role, v, rmin = _view_at(mx, my)
        if v is None:
            return
        ix = (mx - rmin[0]) / v['ox']                  # -> frame image-space pixels (detect's space)
        iy = (my - rmin[1]) / v['oy']
        best, bd = None, 1e18
        for b in v['blobs']:
            dx, dy = b['px'][0] - ix, b['px'][1] - iy
            d = dx * dx + dy * dy
            if d < bd:
                bd, best = d, b
        px = best['px'] if (best is not None and bd <= 40 * 40) else [ix, iy]
        _send({'type': 'track', 'role': role, 'px': [float(px[0]), float(px[1])]})

    def on_right_click():
        _send({'type': 'untrack'})

    with dpg.handler_registry():
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_left_click)
        dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Right, callback=on_right_click)

    def on_start():
        _send({'type': 'capture', 'role': dpg.get_value('cam_combo'), 'on': True})

    def on_stop_cap():
        _send({'type': 'capture', 'role': dpg.get_value('cam_combo'), 'on': False})

    def on_record(_sender, app_data):
        _send({'type': 'record', 'on': bool(app_data)})

    def on_source(_sender, app_data):
        _send({'type': 'set_source', 'role': dpg.get_value('cam_combo'), 'source': app_data})

    # Control + telemetry panel: camera select / capture / record, a slew pad, and state.
    with dpg.window(label="Control", tag="win_control", width=S(320), height=S(360), pos=(10, 10)):
        state_text = dpg.add_text("backend: connecting...")
        dpg.add_separator()
        dpg.add_combo([], tag='cam_combo', width=S(150), label="camera")
        dpg.add_combo(['synthetic', 'zwo', 'sky'], tag='src_combo', width=S(150),
                      label="source", callback=on_source)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Start", width=S(64), callback=lambda: on_start())
            dpg.add_button(label="Stop", width=S(64), callback=lambda: on_stop_cap())
            dpg.add_checkbox(label="Record", tag='rec_chk', callback=on_record)
        dpg.add_separator()
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
        dpg.add_text("click image to track, right-click to stop", color=(150, 150, 150))

    def update_control():
        # Keep the camera selector populated with the roles present in the session.
        roles_now = discover_roles(args.session)
        if list(dpg.get_item_configuration('cam_combo').get('items', [])) != roles_now:
            dpg.configure_item('cam_combo', items=roles_now)
        if not dpg.get_value('cam_combo') and roles_now:
            dpg.set_value('cam_combo', roles_now[0])

        # Connect to the backend command socket once its port file appears.
        if ctrl['client'] is None:
            bj = _newest(args.session, '_backend.json')
            if bj:
                try:
                    info = json.load(open(bj))
                    ctrl['client'] = control.CommandClient(info['command_host'], info['command_port'])
                except (OSError, ValueError, KeyError):
                    ctrl['client'] = None
        # Tail backend state for the telemetry display.
        if ctrl['tailer'] is None:
            sp = _newest(args.session, '_state.jsonl')
            if sp:
                ctrl['tailer'] = JsonlTailer(sp)
        if ctrl['tailer'] is not None:
            for rec in ctrl['tailer'].poll():
                ctrl['state'] = rec
        st = ctrl['state']
        active = dpg.get_value('cam_combo')
        if st and active and not dpg.get_value('src_combo'):     # init source combo from state
            cur = st.get('sources', {}).get(active)
            if cur:
                dpg.set_value('src_combo', cur)
        if st:
            caps = ' '.join(f"{r}:{'on' if v else 'off'}" for r, v in st.get('capturing', {}).items())
            srcs = ' '.join(f"{r}:{s}" for r, s in st.get('sources', {}).items())
            dpg.set_value(state_text,
                          f"{st.get('mode', '?')}   az {st.get('enc_az_deg', 0):.2f}   "
                          f"alt {st.get('enc_alt_deg', 0):.2f}\n"
                          f"rate {st.get('rate_az_deg_s', 0):.2f}, {st.get('rate_alt_deg_s', 0):.2f} deg/s\n"
                          f"record {'ON' if st.get('recording') else 'off'}   capture [{caps}]\n"
                          f"source [{srcs}]")
        else:
            dpg.set_value(state_text,
                          "backend: connecting..." if ctrl['client'] is None else "backend: waiting for state")

        # Press-and-hold: send the rate implied by the currently-held buttons, on change.
        sr = args.slew_rate
        if dpg.is_item_active(btn_stop):
            az = alt = 0.0
        else:
            az = (sr if dpg.is_item_active(btn_az_up) else 0.0) - (sr if dpg.is_item_active(btn_az_dn) else 0.0)
            alt = (sr if dpg.is_item_active(btn_alt_up) else 0.0) - (sr if dpg.is_item_active(btn_alt_dn) else 0.0)
        if ctrl['client'] is not None and (az, alt) != ctrl['last_rate']:
            ctrl['client'].send({'type': 'set_rate', 'az': az, 'alt': alt})
            ctrl['last_rate'] = (az, alt)

    dpg.create_viewport(title="AstroLock Seeker", width=S(1400), height=S(900))
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        update_control()
        active = fixed_roles if fixed_roles is not None else discover_roles(args.session)
        new_work = False
        for role in active:
            ensure_view(role)
            if role not in views:
                continue
            v = views[role]
            f = followers[role]
            res = f.read_latest()
            if res is None:
                continue
            idx, frame = res

            # Follow segment rollover / source switch: re-point the detections tailer.
            if f.ser_path != v['ser_path']:
                v['det_tailer'].close()
                v['det_tailer'] = JsonlTailer(f.ser_path[:-len('.ser')] + '.detections.jsonl')
                v['ser_path'] = f.ser_path
                v['blobs'] = []
                v['last_idx'] = -1
                v['det_idx'] = -1

            # Detections are cheap to poll every loop.
            for rec in v['det_tailer'].poll():
                v['blobs'] = rec.get('blobs', [])
                v['det_idx'] = rec.get('index', v['det_idx'])

            # The expensive part (debayer + texture upload) only runs on a *new* frame, so
            # the loop stays responsive (input + redraw) while waiting for the next one.
            if idx != v['last_idx']:
                w, h, rgba = prepare_rgba(frame, f.header.color_id, args.gamma, wb)
                if (w, h) != (v['w'], v['h']):    # frame size changed (source switch) -> rebuild
                    rebuild_view(role)
                    continue
                dpg.set_value(v['tex'], rgba.ravel())
                v['last_idx'] = idx
                v['peak'] = int(frame.max())
                new_work = True

            # Cheap every loop: vector box overlay + status. Fade boxes whose detection is
            # for an older frame than the one currently shown (detect lagging behind).
            dpg.delete_item(v['box_layer'], children_only=True)
            a = 255 if v['det_idx'] >= v['last_idx'] else 70
            for b in v['blobs']:
                cx, cy = b['px']
                half = max(4.0 * ui_scale, b.get('size_px', 4) * v['ox']) + 3.0 * ui_scale
                x, y = cx * v['ox'], cy * v['oy']
                color = (60, 255, 60, a) if b.get('moving') else (255, 200, 40, a)
                dpg.draw_rectangle((x - half, y - half), (x + half, y + half),
                                   color=color, thickness=max(1.0, ui_scale), parent=v['box_layer'])
            dpg.set_value(v['status'],
                          f"frame {v['last_idx']}  {_color_name(f.header.color_id)}  "
                          f"peak {v.get('peak', 0)}  blobs {len(v['blobs'])}")

            # Locked-target marker (magenta crosshair) from the backend's tracking state.
            dpg.delete_item(v['track_layer'], children_only=True)
            stt = ctrl['state']
            if (stt and stt.get('tracking') and stt.get('track_role') == role
                    and stt.get('target_px')):
                tx, ty = stt['target_px']
                X, Y = tx * v['ox'], ty * v['oy']
                col = (255, 60, 220, 255)
                r = S(14)
                th = max(1.0, 2.0 * ui_scale)
                dpg.draw_circle((X, Y), r, color=col, thickness=th, parent=v['track_layer'])
                for ex, ey in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    dpg.draw_line((X + ex * (r + S(6)), Y + ey * (r + S(6))),
                                  (X + ex * S(4), Y + ey * S(4)), color=col, thickness=th,
                                  parent=v['track_layer'])

        dpg.render_dearpygui_frame()
        if not new_work:
            time.sleep(0.005)            # idle: keep UI responsive without pegging a core

    if ctrl['client'] is not None:
        ctrl['client'].close()
    for f in followers.values():
        f.close()
    dpg.destroy_context()


if __name__ == '__main__':
    main()
