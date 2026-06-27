"""
Mount abstraction for the Seeker backend.

The backend commands axis *rates* and reads back an *encoder pose*; everything else (PID,
target association, sim camera) talks to this small interface so the real mount and the
simulator are interchangeable:

    set_rates(az_rad_s, alt_rad_s)   # command
    get_state() -> {az_rad, alt_rad, rate_az_rad_s, rate_alt_rad_s}

- SimMount integrates commanded rates into an encoder estimate (what the backend used to do
  inline); the sky-sim camera follows the published estimate, closing the loop in simulation.
- CelestronMount drives the real mount via the existing NexStar hand-controller driver. A
  single dedicated thread owns the serial port -- the Prolific USB-serial drivers hang/BSOD
  on multi-threaded access, so nothing else may ever touch it (see the driver's own warning).

Pick one with make_mount(); --mount selects sim vs celestron.
"""

import math
import threading
import time


def _wrap_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Mount:
    def set_rates(self, az_rad_s, alt_rad_s):
        raise NotImplementedError

    def get_state(self):
        """-> dict(az_rad, alt_rad, rate_az_rad_s, rate_alt_rad_s)."""
        raise NotImplementedError

    def close(self):
        pass


class SimMount(Mount):
    """Integrate commanded rates into an encoder estimate (lazily, on each access)."""

    def __init__(self, az0_rad=0.0, alt0_rad=0.0, max_rate_rad_s=math.radians(8.0)):
        self._az, self._alt = az0_rad, alt0_rad
        self._raz, self._ralt = 0.0, 0.0
        self._max = max_rate_rad_s
        self._t = time.perf_counter()

    def _integrate(self):
        now = time.perf_counter()
        dt = now - self._t
        self._t = now
        self._az = (self._az + self._raz * dt) % (2 * math.pi)
        self._alt = max(-math.pi / 2, min(math.pi / 2, self._alt + self._ralt * dt))

    def set_rates(self, az_rad_s, alt_rad_s):
        self._integrate()
        self._raz = max(-self._max, min(self._max, az_rad_s))
        self._ralt = max(-self._max, min(self._max, alt_rad_s))

    def get_state(self):
        self._integrate()
        return {'az_rad': self._az, 'alt_rad': self._alt,
                'rate_az_rad_s': self._raz, 'rate_alt_rad_s': self._ralt}


class CelestronMount(Mount):
    """
    Real Celestron mount via the NexStar hand controller. One thread owns the serial port and
    runs the ~7 Hz send-rates / read-positions loop, reusing the existing driver's protocol.

    UNTESTED against hardware in this milestone -- exercise at the scope before trusting it.
    """

    def __init__(self, url, az0_rad=0.0, alt0_rad=0.0, max_rate_rad_s=math.radians(8.0)):
        from astrolock.model.telescope_connections.celestron_nexstar_hc import (
            CelestronNexstarHCConnection)
        # tracker is unused by the low-level serial methods we call.
        self._conn = CelestronNexstarHCConnection(url, tracker=None)
        self._max = max_rate_rad_s
        self._lock = threading.Lock()
        self._desired = [0.0, 0.0]
        self._angles = [az0_rad, alt0_rad]
        self._rates = [0.0, 0.0]
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
                    last_a[axis], last_t[axis] = a, t

    def set_rates(self, az_rad_s, alt_rad_s):
        with self._lock:
            self._desired = [max(-self._max, min(self._max, az_rad_s)),
                             max(-self._max, min(self._max, alt_rad_s))]

    def get_state(self):
        with self._lock:
            return {'az_rad': self._angles[0], 'alt_rad': self._angles[1],
                    'rate_az_rad_s': self._rates[0], 'rate_alt_rad_s': self._rates[1]}

    def close(self):
        self._stop = True
        self._thread.join(timeout=2.0)


def make_mount(kind, az0_rad=0.0, alt0_rad=0.0, max_rate_rad_s=math.radians(8.0), url=None):
    if kind == 'celestron':
        if not url:
            raise SystemExit("--mount celestron requires --mount-url celestron_nexstar_hc:COMx")
        return CelestronMount(url, az0_rad, alt0_rad, max_rate_rad_s)
    return SimMount(az0_rad, alt0_rad, max_rate_rad_s)
