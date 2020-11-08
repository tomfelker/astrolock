import astropy.units as u
import numpy as np

class TelescopeConnection(object):
    def __init__(self, url, tracker):
        self.tracker = tracker
        if not url.startswith(self.__class__.get_url_scheme()):
            raise RuntimeError("This connection type can't handle this URL scheme")
        self.url = url
        self.url_path = url[len(self.__class__.get_url_scheme()):]
        self.axis_angles = np.zeros(2) * u.rad
        self.desired_axis_rates = np.zeros(2) * u.rad / u.s


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
        s = "Connected to " + self.url + "\n"
        s += "Desired rates: " + str(self.desired_axis_rates.to(u.deg / u.s)) + "\n"
        s += "Angles:        " + str(self.axis_angles.to(u.deg)) + "\n"
        return s

    def set_axis_rate(self, axis, angular_rate_deg_per_s):
        self.desired_axis_rates[axis] = angular_rate_deg_per_s * (u.deg / u.s)

from astrolock.model.telescope_connections import *
