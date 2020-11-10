from astrolock.model.telescope_connection import TelescopeConnection
from astropy import units as u
import time
import numpy as np
import threading
import copy

class TrackerInput(object):
    def __init__(self):
        self.rate_scale = 1
        self.accel_scale = 1
        self.rate = np.zeros(2)
        self.slowdown = 0
        self.speedup = 0

class TrackerInputState(object):
    def __init__(self):
        self.monotonic_time_ns_of_last_input = None
        self.momentum = np.zeros(2)
        self.smooth_input_mag = 0

class Tracker(object):
    
    def __init__(self):
        self.lock = threading.RLock()
        self.primary_telescope_connection = None
        self.monotonic_time_ns_of_last_input = None
        self.momentum = np.zeros(2)
        self.rates = np.zeros(2)
        self.tracker_input = TrackerInput()
        self.smooth_input_mag = 0.0
    
    def get_recommended_connection_urls(self):
        with self.lock:
            return TelescopeConnection.get_urls_for_subclasses()

    def connect_to_telescope(self, url):
        with self.lock:
            if (self.primary_telescope_connection):
                self.disconnect_from_telescope()
            self.primary_telescope_connection = TelescopeConnection.get_connection(url, self)
            if self.primary_telescope_connection:
                self.primary_telescope_connection.start()

    def disconnect_from_telescope(self):
        with self.lock:
            if self.primary_telescope_connection:
                self.primary_telescope_connection.stop()
                self.primary_telescope_connection = None

    def get_status(self):
        with self.lock:
            s = ""
            s += "Input: \n"
            s += "\tRate:  " + str(self.tracker_input.rate) + "\n"
            s += "\tSlowdown: " + str(self.tracker_input.slowdown) + "\n"
            s += "\tSpeedup: " + str(self.tracker_input.speedup) + "\n"
            s += "Momentum: " + str(self.momentum) + "\n"

            if self.primary_telescope_connection:
                s += "Connected to telescope at " + self.primary_telescope_connection.url + "\n"
                s += self.primary_telescope_connection.get_status()
            else:
                s += "Not connected to telescope.\n"
            return s

    def set_input(self, tracker_input):
        with self.lock:
            self.tracker_input = tracker_input
            rates = self._compute_rates(True)
            if self.primary_telescope_connection is not None:
                self.primary_telescope_connection.set_axis_rate(0, rates[0])
                self.primary_telescope_connection.set_axis_rate(1, rates[1])
            

    def get_rates(self):
        with self.lock:
            return self._compute_rates(False) * (u.deg / u.s)

    def _compute_rates(self, store):
        with self.lock:
            monotonic_time_ns = time.monotonic_ns()
            dt = 0
            if self.monotonic_time_ns_of_last_input is not None:
                elapsed_since_last_input_ns = monotonic_time_ns - self.monotonic_time_ns_of_last_input
                dt = elapsed_since_last_input_ns * 1e-9

            rate = None

            base_rc = 2.0
            rc = base_rc / (1.0 + self.tracker_input.speedup)

            alpha = dt / (rc + dt)
            
            # here, we're treating both accel and rate together:
            input_rate = self.tracker_input.rate * self.tracker_input.rate_scale
            momentum = self.momentum
            momentum = (momentum + input_rate) * alpha + momentum * (1.0 - alpha)
            rate = momentum

            # hmm, todo: make this based on magnitude, instead of per-axis
            max_braking = dt * self.tracker_input.rate_scale * self.tracker_input.slowdown
            #momentum = np.maximum(np.abs(momentum) - max_braking, 0) * np.sign(momentum)
            momentum_mag = np.sqrt(np.dot(momentum, momentum))
            if momentum_mag > 0:
                momentum_desired_mag = np.maximum(0, momentum_mag - max_braking)
                momentum *= momentum_desired_mag / momentum_mag

            if store:
                self.momentum = momentum    
                self.monotonic_time_ns_of_last_input = monotonic_time_ns

            return rate



