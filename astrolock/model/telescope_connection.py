import astropy.units as u
import astropy.time
import numpy as np
import time
from astrolock.model.util import *

class TelescopeConnection(object):
    def __init__(self, url, tracker):
        self.tracker = tracker
        if not url.startswith(self.__class__.get_url_scheme()):
            raise RuntimeError("This connection type can't handle this URL scheme")
        self.url = url
        self.url_path = url[len(self.__class__.get_url_scheme()):]
        self.axis_angles = np.zeros(2) * u.rad
        self.axis_angles_measurement_time = astropy.time.Time(np.zeros(2), format = 'unix')
        self.gps_time = None
        self.gps_location = None
        self.gps_measurement_time = None
        self.desired_axis_rates = np.zeros(2) * u.rad / u.s
        
        # could be false if you're using a radio telescope,
        # or Stellarium, which can't tell us the current angles _without_ refaction, so we need to turn it off
        self.want_atmospheric_refaction = True
        # it'd be neat if the hand controller could tell us...
        self.current_temperature_C = 10.0

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
        s  = "\tDesired rates: " + str(self.desired_axis_rates.to(u.deg / u.s)) + "\n"
        s += "\tAngles:        " + str(self.axis_angles.to(u.deg)) + "\n"
        s += "\tLoop time:     " + str((self.loop_time_smoothed_s * u.s).to(u.ms)) + "\n"
        return s

    def record_loop_rate(self):
        cur_time_ns = time.perf_counter_ns()
        if self.last_loop_performance_time_ns is not None:
            loop_time_ns = cur_time_ns - self.last_loop_performance_time_ns
            self.loop_time_s = loop_time_ns * 1e-9
            self.loop_time_smoothed_s = lerp(self.loop_time_s, self.loop_time_smoothed_s or self.loop_time_s, self.loop_time_smoothing)
            
        self.last_loop_performance_time_ns = cur_time_ns
        

    def set_axis_rate(self, axis, angular_rate_deg_per_s):
        self.desired_axis_rates[axis] = angular_rate_deg_per_s * (u.deg / u.s)
        
    def get_estimated_axis_angles_and_time(self):
        current_time = astropy.time.Time.now()
        times_since_measurement = current_time - self.axis_angles_measurement_time
        estimated_current_axis_angles = self.axis_angles + self.desired_axis_rates * times_since_measurement
        return estimated_current_axis_angles, current_time


from astrolock.model.telescope_connections import *
