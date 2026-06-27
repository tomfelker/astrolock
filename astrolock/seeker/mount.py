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
  port). Real GPS read is still TODO; it reports a configured fallback site for now.

Pick one with make_mount(); --mount selects sim vs celestron.
"""

import datetime
import math
import threading
import time


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Mount:
    def set_rates(self, az_rad_s, alt_rad_s):
        raise NotImplementedError

    def get_state(self):
        """-> dict(az_rad, alt_rad, rate_az_rad_s, rate_alt_rad_s)."""
        raise NotImplementedError

    def get_site(self):
        """-> dict(lat_deg, lon_deg, elev_m, epoch_utc)."""
        raise NotImplementedError

    def close(self):
        pass


class SimMount(Mount):
    """
    Simulated mount driver: its own ~update_hz loop integrates commanded rates subject to
    speed + acceleration limits, and reports a test site/clock. Runs in real time (sim time =
    epoch + elapsed wall-clock).
    """

    def __init__(self, az0_rad, alt0_rad, site, max_rate_rad_s=math.radians(8.0),
                 accel_rad_s2=math.radians(20.0), update_hz=10.0):
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
        self._angle_t_ns = time.perf_counter_ns()   # when the reported angles were valid
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            now = time.perf_counter()
            dt = now - self._last
            self._last = now
            with self._lock:
                for ax in (0, 1):
                    dv = _clamp(self._cmd[ax] - self._rate[ax], -self._accel * dt, self._accel * dt)
                    self._rate[ax] = _clamp(self._rate[ax] + dv, -self._max, self._max)
                # Both axes rotate freely (no limits, clutches): altitude can tip past the zenith
                # and keep going, so a near-zenith meridian crossing is tracked by tipping over
                # rather than a 180-deg azimuth whip.
                self._az = (self._az + self._rate[0] * dt) % (2 * math.pi)
                self._alt = (self._alt + self._rate[1] * dt) % (2 * math.pi)
                self._angle_t_ns = time.perf_counter_ns()
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

    def close(self):
        self._stop = True
        self._thread.join(timeout=2.0)


class CelestronMount(Mount):
    """
    Real Celestron mount via the NexStar hand controller. One thread owns the serial port and
    runs the ~7 Hz send-rates / read-positions loop, reusing the existing driver's protocol.

    UNTESTED against hardware in this milestone. Real GPS read is TODO -- get_site() returns a
    configured fallback for now.
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
        self._angle_t_ns = time.perf_counter_ns()
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


def make_mount(kind, az0_rad, alt0_rad, site, max_rate_rad_s=math.radians(8.0),
               accel_rad_s2=math.radians(20.0), update_hz=10.0, url=None):
    if kind == 'celestron':
        if not url:
            raise SystemExit("--mount celestron requires --mount-url celestron_nexstar_hc:COMx")
        return CelestronMount(url, az0_rad, alt0_rad, site, max_rate_rad_s)
    return SimMount(az0_rad, alt0_rad, site, max_rate_rad_s, accel_rad_s2, update_hz)
