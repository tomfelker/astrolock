from astrolock.model.telescope_connection import TelescopeConnection
from astropy import units as u
import time
import numpy as np

class TrackerInput(object):
    def __init__(self):
        self.rate_scale = 1
        self.accel_scale = 1
        # these are instantaneous, i.e., they don't affect momentum
        self.rate = np.zeros(2)
        # these affect momentum
        self.accel = np.zeros(2)
        self.braking = 0


class Tracker(object):
    
    def __init__(self):
        self.primary_telescope_connection = None

        self.monotonic_time_ns_of_last_input = None
        self.momentum = np.zeros(2)
        self.rates = np.zeros(2)
        self.tracker_input = TrackerInput()
    
    def get_recommended_connection_urls(self):
        return TelescopeConnection.get_urls_for_subclasses()

    def connect_to_telescope(self, url):
        if (self.primary_telescope_connection):
            self.disconnect_from_telescope()
        self.primary_telescope_connection = TelescopeConnection.get_connection(url, self)
        if self.primary_telescope_connection:
            self.primary_telescope_connection.start()

    def disconnect_from_telescope(self):
        if self.primary_telescope_connection:
            self.primary_telescope_connection.stop()
            self.primary_telescope_connection = None

    def get_status(self):
        s = ""
        s += "Input: \n"
        s += "\tRate:  " + str(self.tracker_input.rate) + "\n"
        s += "\tAccel: " + str(self.tracker_input.accel) + "\n"
        s += "Momentum: " + str(self.momentum) + "\n"
        if self.primary_telescope_connection:
            s += "Connected to telescope at " + self.primary_telescope_connection.url + "\n"
            s += self.primary_telescope_connection.get_status()
        else:
            s += "Not connected to telescope.\n"
        return s

    def set_input(self, tracker_input):
        self.tracker_input = tracker_input
        rates = self._compute_rates(True)
        if self.primary_telescope_connection is not None:
            self.primary_telescope_connection.set_axis_rate(0, rates[0])
            self.primary_telescope_connection.set_axis_rate(1, rates[1])
            

    def get_rates(self):
        return self._compute_rates(False) * (u.deg / u.s)

    def _compute_rates(self, store):
        monotonic_time_ns = time.monotonic_ns()
        elapsed_since_last_input = 0
        if self.monotonic_time_ns_of_last_input is not None:
            elapsed_since_last_input_ns = monotonic_time_ns - self.monotonic_time_ns_of_last_input
            elapsed_since_last_input = elapsed_since_last_input_ns * 1e-9

        momentum = self.momentum + self.tracker_input.accel * (self.tracker_input.accel_scale * elapsed_since_last_input)
        max_braking = elapsed_since_last_input * self.tracker_input.rate_scale * self.tracker_input.braking
        momentum = np.maximum(np.abs(momentum) - max_braking, 0) * np.sign(momentum)

        if store:
            self.momentum = momentum    
            self.monotonic_time_ns_of_last_input = monotonic_time_ns

        return momentum + self.tracker_input.rate * self.tracker_input.rate_scale



