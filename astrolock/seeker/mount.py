"""
Mount drivers for the Seeker backend.

The backend commands axis *rates* and reads back an *encoder pose* + the observing *site*
(GPS location + time), through one interface, so the real mount and the simulator are
interchangeable -- the backend treats both as "real":

    set_rates(az_rad_s, alt_rad_s)               # command
    get_state() -> {az_rad, alt_rad, rate_az_rad_s, rate_alt_rad_s}
    get_site()  -> {lat_deg, lon_deg, elev_m, epoch_utc}   # like a mount's GPS

- SimMount is a *driver*, not just an integrator: it runs its own update loop at a realistic
  rate with speed + acceleration limits (periodic error etc. can follow), and reports a test
  site/clock. The backend feeds that site to the sky-sim camera, so the simulated sky matches
  where/when the (simulated) mount thinks it is. Everything runs in real time (sim time =
  epoch + elapsed wall-clock); a global time-scale is deferred.
- CelestronMount drives the real NexStar mount on a single dedicated serial thread (the
  Prolific USB-serial drivers BSOD on multi-threaded access -- only that thread touches the
  port). Its GPS is read on request (request_gps/take_gps_result) on that same thread;
  get_site() still reports the configured fallback (the backend owns live time/site).

Pick one with make_mount(); --mount selects sim vs celestron.
"""

import datetime
import math
import threading
import time

from astrolock.seeker.sidecar import JsonlWriter
from astrolock.seeker.session import mono_ns


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Mount:
    _gps_result = None

    def set_rates(self, az_rad_s, alt_rad_s):
        raise NotImplementedError

    def get_state(self):
        """-> dict(az_rad, alt_rad, rate_az_rad_s, rate_alt_rad_s)."""
        raise NotImplementedError

    def get_site(self):
        """-> dict(lat_deg, lon_deg, elev_m, epoch_utc)."""
        raise NotImplementedError

    def request_gps(self):
        """Ask this mount's GPS for a fix; the answer lands later via take_gps_result()
        (the Celestron read happens on its serial thread). Base: no GPS."""
        self._gps_result = {'ok': False, 'error': 'this mount has no GPS'}

    def take_gps_result(self):
        """Return-and-clear the last GPS answer: None (still pending / never asked), or
        {'ok': True, lat_deg, lon_deg, elev_m[, utc_ns, t_mono_ns]} -- utc_ns is unix UTC ns
        and t_mono_ns the system time it was valid at -- or {'ok': False, 'error': str}."""
        r, self._gps_result = self._gps_result, None
        return r

    def close(self):
        pass


class SimMount(Mount):
    """
    Simulated mount driver: its own ~update_hz loop integrates commanded rates subject to
    speed + acceleration limits, and reports a test site/clock. Runs in real time (sim time =
    epoch + elapsed wall-clock).
    """

    def __init__(self, az0_rad, alt0_rad, site, max_rate_rad_s=math.radians(8.0),
                 accel_rad_s2=math.radians(20.0), update_hz=10.0, sidecar_path=None):
        self._site = dict(site)
        self._az, self._alt = az0_rad, alt0_rad
        self._cmd = [0.0, 0.0]                    # commanded axis rates (rad/s)
        self._rate = [0.0, 0.0]                   # actual rates after accel limiting
        self._max = max_rate_rad_s
        self._accel = accel_rad_s2
        self._period = 1.0 / update_hz if update_hz > 0 else 0.1
        self._t0 = datetime.datetime.fromisoformat(site['epoch_utc'].replace('Z', '+00:00'))
        self._lock = threading.Lock()
        self._stop = False
        self._last = time.perf_counter()
        self._wall0 = self._last
        self._angle_t_ns = mono_ns()                # when the reported angles were valid
        # Ground-truth trajectory sidecar for the sim camera: piecewise-linear anchors
        # {t, angle, rate}, one whenever the actual rate changes. This is the mount's *real* plan
        # (continuous by construction), as opposed to the backend's reconstructed estimate -- so the
        # sim camera observes truth, not belief, and never sees a reconstruction discontinuity.
        self._writer = JsonlWriter(sidecar_path) if sidecar_path else None
        self._write_anchor_locked()                 # initial anchor (start pose, zero rate)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _write_anchor_locked(self):
        """Append one trajectory anchor: the pose now and the rate that holds forward from it.
        Caller holds the lock (or is __init__, before the thread starts)."""
        if self._writer is None:
            return
        self._writer.append({'t_mono_ns': self._angle_t_ns,
                             'az_deg': math.degrees(self._az), 'alt_deg': math.degrees(self._alt),
                             'rate_az_deg_s': math.degrees(self._rate[0]),
                             'rate_alt_deg_s': math.degrees(self._rate[1])})

    def _loop(self):
        while not self._stop:
            now = time.perf_counter()
            dt = now - self._last
            self._last = now
            with self._lock:
                # Advance the pose with the rate in effect over [last, now] FIRST, then stamp, then
                # ramp the rate for the next interval. This ordering makes each emitted anchor a
                # clean *forward* anchor: angle + rate*(future - t) reproduces the next integration
                # step exactly, so the camera's linear extrapolation is continuous across every rate
                # change (no accel*dt^2 step from pairing an angle with the rate that just changed).
                # Both axes rotate freely (no limits, clutches): altitude can tip past the zenith
                # and keep going, so a near-zenith meridian crossing tips over rather than whipping az.
                self._az = (self._az + self._rate[0] * dt) % (2 * math.pi)
                self._alt = (self._alt + self._rate[1] * dt) % (2 * math.pi)
                self._angle_t_ns = mono_ns()
                changed = False
                for ax in (0, 1):
                    dv = _clamp(self._cmd[ax] - self._rate[ax], -self._accel * dt, self._accel * dt)
                    nr = _clamp(self._rate[ax] + dv, -self._max, self._max)
                    if nr != self._rate[ax]:
                        self._rate[ax] = nr
                        changed = True
                if changed:                          # new constant-rate segment -> new anchor
                    self._write_anchor_locked()
            time.sleep(self._period)

    def set_rates(self, az_rad_s, alt_rad_s):
        with self._lock:
            self._cmd = [_clamp(az_rad_s, -self._max, self._max),
                         _clamp(alt_rad_s, -self._max, self._max)]

    def get_state(self):
        with self._lock:
            return {'az_rad': self._az, 'alt_rad': self._alt,
                    'rate_az_rad_s': self._rate[0], 'rate_alt_rad_s': self._rate[1],
                    't_mono_ns': self._angle_t_ns}

    def get_site(self):
        return dict(self._site)

    def now_utc(self):
        """Current simulated UTC (epoch + elapsed wall-clock)."""
        elapsed = time.perf_counter() - self._wall0
        return (self._t0 + datetime.timedelta(seconds=elapsed)).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

    def request_gps(self):
        """The sim mount's 'GPS': its configured site + clock (epoch + elapsed), instantly --
        so the GUI's Set-from-Mount-GPS buttons are exercisable end-to-end in sim."""
        utc = self._t0 + datetime.timedelta(seconds=time.perf_counter() - self._wall0)
        self._gps_result = {'ok': True, 'lat_deg': self._site['lat_deg'],
                            'lon_deg': self._site['lon_deg'], 'elev_m': self._site['elev_m'],
                            'utc_ns': int(utc.timestamp() * 1e9), 't_mono_ns': mono_ns()}

    def close(self):
        self._stop = True
        self._thread.join(timeout=2.0)
        if self._writer is not None:
            self._writer.close()


class CelestronMount(Mount):
    """
    Real Celestron mount via the NexStar hand controller. One thread owns the serial port and
    runs the ~7 Hz send-rates / read-positions loop, reusing the existing driver's protocol.
    GPS reads (request_gps) run between iterations on that same thread.

    UNTESTED against hardware in this milestone (including the GPS read). get_site() returns
    the configured fallback; live time/site are backend-owned, fed by take_gps_result().
    """

    def __init__(self, url, az0_rad=0.0, alt0_rad=0.0, site=None, max_rate_rad_s=math.radians(8.0)):
        from astrolock.model.telescope_connections.celestron_nexstar_hc import (
            CelestronNexstarHCConnection)
        self._conn = CelestronNexstarHCConnection(url, tracker=None)
        self._site = dict(site) if site else {}
        self._max = max_rate_rad_s
        self._lock = threading.Lock()
        self._desired = [0.0, 0.0]
        self._angles = [az0_rad, alt0_rad]
        self._rates = [0.0, 0.0]
        self._angle_t_ns = mono_ns()
        self._gps_requested = False
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        rad_to_arcsec = 180.0 / math.pi * 3600.0
        conn = self._conn
        last_a = [None, None]
        last_t = [None, None]
        with conn._open_serial_stream():
            while not self._stop:
                with self._lock:
                    d0, d1 = self._desired
                conn._serial_send_axis_rate_cmd(0, d0 * rad_to_arcsec)
                conn._serial_send_axis_rate_cmd(1, d1 * rad_to_arcsec)
                for axis in range(2):
                    a = conn._serial_read_axis_position_radians(axis)
                    t = conn.last_message_thinking_mid_time_ns
                    with self._lock:
                        if last_t[axis] is not None and t != last_t[axis]:
                            self._rates[axis] = _wrap_pi(a - last_a[axis]) / ((t - last_t[axis]) * 1e-9)
                        self._angles[axis] = a
                        self._angle_t_ns = t          # serial measurement time of the angle
                    last_a[axis], last_t[axis] = a, t
                if self._gps_requested:              # on THIS thread: only it touches the port
                    self._gps_requested = False
                    try:
                        self._gps_result = self._read_gps()
                    except Exception as e:
                        self._gps_result = {'ok': False, 'error': f'GPS read failed: {e}'}

    def _read_gps(self):
        """One GPS fix via the hand controller's 'P' passthrough to the GPS module (0xb0) --
        the legacy driver's _read_gps sequence, without the astropy wrapping. Serial thread only."""
        from astrolock.model.telescope_connections.celestron_nexstar_hc import (
            CelestronNexstarDeviceIds as DEV, CelestronNexstarCommands as CMD,
            bytes_to_uint, bytes_to_radians)
        conn = self._conn

        def ask(msg_id, response_len):
            return conn._send_and_receive_via_hc(DEV.DEV_ID_GPS, msg_id, response_len=response_len)

        linked, = ask(CMD.GPS_LINKED, 1)
        if not linked:
            return {'ok': False, 'error': 'GPS not linked (no satellite fix yet)'}
        out = {'ok': True,
               'lat_deg': math.degrees(_wrap_pi(bytes_to_radians(ask(CMD.GPS_GET_LAT, 3)))),
               'lon_deg': math.degrees(_wrap_pi(bytes_to_radians(ask(CMD.GPS_GET_LONG, 3)))),
               'elev_m': float(bytes_to_uint(ask(CMD.GPS_GET_HEIGHT, 2)))}
        time_valid, = ask(CMD.GPS_TIME_VALID, 1)
        if time_valid:
            year = bytes_to_uint(ask(CMD.GPS_GET_YEAR, 2))
            month, day = ask(CMD.GPS_GET_DATE, 2)
            hour, minute, second = ask(CMD.GPS_GET_TIME, 3)
            t_mono_ns = mono_ns()                # the moment the reported second is valid for...
            utc = datetime.datetime(year, month, day, hour, minute, second,
                                    tzinfo=datetime.timezone.utc)
            # ...plus the HC's ~200 ms answer lag after the 1PPS edge (legacy driver's correction).
            out['utc_ns'] = int(utc.timestamp() * 1e9) + 200_000_000
            out['t_mono_ns'] = t_mono_ns
        return out

    def request_gps(self):
        self._gps_requested = True           # picked up by the serial loop

    def set_rates(self, az_rad_s, alt_rad_s):
        with self._lock:
            self._desired = [_clamp(az_rad_s, -self._max, self._max),
                             _clamp(alt_rad_s, -self._max, self._max)]

    def get_state(self):
        with self._lock:
            return {'az_rad': self._angles[0], 'alt_rad': self._angles[1],
                    'rate_az_rad_s': self._rates[0], 'rate_alt_rad_s': self._rates[1],
                    't_mono_ns': self._angle_t_ns}

    def get_site(self):
        return dict(self._site)        # TODO: read the mount's GPS (lat/lon/time)

    def close(self):
        self._stop = True
        self._thread.join(timeout=2.0)


class NullMount(Mount):
    """Disconnected placeholder: holds its last pose, ignores rate commands, reports a site. Lets the
    backend's control loop keep running with no mount attached -- nothing moves."""

    def __init__(self, az0_rad=0.0, alt0_rad=0.0, site=None):
        self._az, self._alt = az0_rad, alt0_rad
        self._site = dict(site) if site else {}
        self._t_ns = mono_ns()

    def set_rates(self, az_rad_s, alt_rad_s):
        pass

    def get_state(self):
        return {'az_rad': self._az, 'alt_rad': self._alt, 'rate_az_rad_s': 0.0,
                'rate_alt_rad_s': 0.0, 't_mono_ns': self._t_ns}

    def get_site(self):
        return dict(self._site)

    def request_gps(self):
        self._gps_result = {'ok': False, 'error': 'no mount connected'}


def available_mount_urls():
    """Best-effort list of connectable real-mount URLs (Celestron on detected COM ports); [] if the
    driver import or serial enumeration fails (e.g. no serial devices / no pyserial)."""
    try:
        from astrolock.model.telescope_connections.celestron_nexstar_hc import CelestronNexstarHCConnection
        return list(CelestronNexstarHCConnection.get_urls())
    except Exception as e:
        print(f"[mount] could not enumerate mounts: {e}", flush=True)
        return []


def make_mount(kind, az0_rad, alt0_rad, site, max_rate_rad_s=math.radians(8.0),
               accel_rad_s2=math.radians(20.0), update_hz=10.0, url=None, sidecar_path=None):
    if kind == 'celestron':
        if not url:
            raise SystemExit("--mount celestron requires --mount-url celestron_nexstar_hc:COMx")
        return CelestronMount(url, az0_rad, alt0_rad, site, max_rate_rad_s)
    return SimMount(az0_rad, alt0_rad, site, max_rate_rad_s, accel_rad_s2, update_hz,
                    sidecar_path=sidecar_path)
