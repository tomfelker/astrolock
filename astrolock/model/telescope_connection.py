import math
import time

import astropy
import astropy.units as u
import numpy as np

from astrolock.model.util import *


_connections_loaded = False


def _load_connection_subclasses():
    """Import the connection submodules so ``TelescopeConnection.__subclasses__()`` sees them.

    Done lazily (called from the factory methods) rather than as a bottom-of-file ``import *``, to
    break a circular import: importing a connection submodule pulls in ``threaded`` ->
    ``telescope_connection`` -> (the old bottom-line import) -> the submodules again, and a submodule's
    ``class X(threaded.ThreadedConnection)`` then runs before ``threaded`` has finished defining
    ``ThreadedConnection``. Deferring to first factory use means threaded is fully loaded by then.
    """
    global _connections_loaded
    if _connections_loaded:
        return
    _connections_loaded = True
    import importlib
    from astrolock.model import telescope_connections as pkg
    for name in getattr(pkg, '__all__', []):
        importlib.import_module(f'{pkg.__name__}.{name}')


class TelescopeConnection(object):
    def __init__(self, url, tracker):
        self.tracker = tracker
        if not url.startswith(self.__class__.get_url_scheme()):
            raise RuntimeError("This connection type can't handle this URL scheme")
        self.url = url
        self.url_path = url[len(self.__class__.get_url_scheme()):]

        # rad
        self.axis_angles = np.zeros(2)
        ns = time.perf_counter_ns()
        self.axis_measurement_times_ns = np.array([ns, ns], dtype=np.int64)

        # astropy.Time
        self.gps_time = None
        self.gps_location = None        
        self.gps_measurement_time_ns = None
        self.gps_requested = True

        # rad/s
        self.desired_axis_rates = np.zeros(2)
        # rad/s
        self.estimated_axis_rates = np.zeros(2)
        
        # could be false if you're using a radio telescope,
        # or Stellarium, which can't tell us the current angles _without_ refaction, so we need to turn it off
        self.want_atmospheric_refaction = True
        # it'd be neat if the hand controller could tell us...
        self.current_temperature_C = 10.0

        self.want_sleep_inhibited = True

        # state for record_loop_rate()
        self.last_loop_performance_time_ns = None
        self.loop_time_s = 0
        self.loop_time_smoothed_s = 0
        self.loop_time_smoothing = .9


    # override this to return something like "my_device_type:"
    @classmethod
    def get_url_scheme(cls):
        raise NotImplementedError()

    @classmethod
    def get_connection_class(cls, url):
        _load_connection_subclasses()          # ensure every connection type is registered as a subclass
        try:
            if url.startswith(cls.get_url_scheme()):
                return cls
        except NotImplementedError:
            for subclass in cls.__subclasses__():
                connection_class = subclass.get_connection_class(url)
                if connection_class is not None:
                    return connection_class

    
    @classmethod
    def get_connection(cls, url, tracker):
        return cls.get_connection_class(url)(url, tracker)

    @classmethod
    def get_urls_for_subclasses(cls):
        _load_connection_subclasses()          # ensure every connection type is registered as a subclass
        connections = []
        for subclass in cls.__subclasses__():
            try:                
                connections += subclass.get_urls_for_subclasses()
                connections += subclass.get_urls()
            except NotImplementedError:
                pass
        return connections

    # override this to return a list of URL strings you could be constructed with, e.g., [ "my_device_type:port1", "my_device_type:port2" ]
    @classmethod
    def get_urls(cls):
        raise NotImplementedError
    
    def get_status(self):
        return (
            f"\t                     Azimuth   Altitude\n"
            f"\tLast Angles:      [{math.degrees(self.axis_angles[0]): 9.3f}, {math.degrees(self.axis_angles[1]): 9.3f}] deg\n"
            f"\tDesired rates:    [{math.degrees(self.desired_axis_rates[0]): 9.3f}, {math.degrees(self.desired_axis_rates[1]): 9.3f}] deg/s\n"
            f"\tEstimated rates:  [{math.degrees(self.estimated_axis_rates[0]): 9.3f}, {math.degrees(self.estimated_axis_rates[1]): 9.3f}] deg/s\n"
            f"\tSmooth loop time: {self.loop_time_smoothed_s * 1e3: 6.3f} ms\n"
        )

    def record_loop_rate(self):
        cur_time_ns = time.perf_counter_ns()
        if self.last_loop_performance_time_ns is not None:
            loop_time_ns = cur_time_ns - self.last_loop_performance_time_ns
            self.loop_time_s = loop_time_ns * 1e-9
            self.loop_time_smoothed_s = lerp(self.loop_time_s, self.loop_time_smoothed_s or self.loop_time_s, self.loop_time_smoothing)
            
        self.last_loop_performance_time_ns = cur_time_ns        
        
    def get_estimated_axis_angles_and_time(self, estimate_time_ns):
        current_ap_time = self.tracker.get_time()
        current_time_ns = time.perf_counter_ns()
        estimate_age = (current_time_ns - estimate_time_ns) * 1e-9 * u.s
        estimate_time_ap = current_ap_time - estimate_age

        time_s_since_measurement = (estimate_time_ns - self.axis_measurement_times_ns) * 1e-9
        estimated_current_axis_angles = self.axis_angles + self.desired_axis_rates * time_s_since_measurement
        return estimated_current_axis_angles, estimate_time_ap
    
    def request_gps(self):
        self.gps_requested = True

    def get_time(self):
        time_since_measurement = u.Quantity(time.perf_counter_ns() - self.gps_measurement_time_ns, unit=u.ns)
        return self.gps_time + time_since_measurement

    def _set_gps_time_with_inferred_seconds_fraction(self, new_gps_time, new_gps_measurement_time_ns):
        """
        The idea here is the telescope gives us a time with only one-second precision, but with higher accuracy.
        We will guess the fractional part of the seconds, in such a way that the result of get_time() changes as
        little as possible.  If we read the telescope's time many times, the offset should converge so that we are
        never surprised by the time we read from the telescope, because our time rolls over to the next second
        just as its time does.
        """
        if self.gps_measurement_time_ns is not None:
            time_between_measurements = u.Quantity(new_gps_measurement_time_ns - self.gps_measurement_time_ns, unit=u.ns)
            predicted_time_at_measurement = self.gps_time + time_between_measurements
            new_to_predicted_s = (predicted_time_at_measurement - new_gps_time).to_value(u.s)
            fudge_time = .1
            if new_to_predicted_s < -fudge_time:
                print(f"Telescope clock was faster than expected, adjusting ours by {-new_to_predicted_s} s to match.  Keep syncing till this stops happening.")
                new_to_predicted_s = 0.0
            elif new_to_predicted_s > 1.0 + fudge_time:
                print(f"Telescope clock was slower than expected, adjusting ours by {1.0 - new_to_predicted_s} s to match.  Keep syncing till this stops happening.")
                new_to_predicted_s = 1.0

            new_gps_time = new_gps_time + new_to_predicted_s * u.s

        self.gps_time = new_gps_time
        self.gps_measurement_time_ns = new_gps_measurement_time_ns



