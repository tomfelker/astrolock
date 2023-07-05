from astrolock.model.telescope_connection import TelescopeConnection
from astrolock.model.pid import PIDController
import astrolock.model.target_sources.opensky
import astrolock.model.target_sources.kml
import astrolock.model.target_sources.skyfield

from astropy import units as u
import astropy.time
import skyfield
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
        self.target = None

        self.target_source_map = {
            'OpenSky': astrolock.model.target_sources.opensky.OpenSkyTargetSource(self),
            'KML': astrolock.model.target_sources.kml.KmlTargetSource(self),
            'Skyfield': astrolock.model.target_sources.skyfield.SkyfieldTargetSource(self),
        }

        self.pid_controllers = []
        for i in range(2):
            pid_controller = PIDController()
            self.pid_controllers.append(pid_controller)

        # todo: this should come from
        # the hand controller of the telescope if available, or
        # from the PC if there were any decent APIs for it, or
        # the GUI, but for now, hardcode:
        lat_deg, lon_deg, height_m = 37.51089, -122.2719388888889, 60
        self.location_ap = astropy.coordinates.EarthLocation.from_geodetic(lat = lat_deg * u.deg, lon = lon_deg * u.deg, height = height_m * u.m)
        self.location_sf = skyfield.api.wgs84.latlon(latitude_degrees = lat_deg, longitude_degrees = lon_deg, elevation_m = height_m)
       
    
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
            s += "Input:\n"
            s += "\tRate:  " + str(self.tracker_input.rate) + "\n"
            s += "\tSlowdown: " + str(self.tracker_input.slowdown) + "\n"
            s += "\tSpeedup: " + str(self.tracker_input.speedup) + "\n"
            s += "Momentum: " + str(self.momentum) + "\n"
            s += "Target:\n"
            if self.target is not None:
                s += self.target.get_status()
            else:
                s += "\tNo Target\n"

            if self.primary_telescope_connection:
                s += "Connected to telescope at " + self.primary_telescope_connection.url + "\n"
                s += self.primary_telescope_connection.get_status()
            else:
                s += "Not connected to telescope.\n"
            return s

    def set_input(self, tracker_input):
        with self.lock:
            self.tracker_input = tracker_input            
            if self.primary_telescope_connection is not None:
                rates = self._compute_rates(True)
                self.primary_telescope_connection.set_axis_rate(0, rates[0])
                self.primary_telescope_connection.set_axis_rate(1, rates[1])

    def set_target(self, new_target):
        with self.lock:
            if new_target is None:
                self.target = None
            elif self.target is not None and self.target.url == new_target.url and self.target != new_target:
                self.target = self.target.updated_with(new_target)
            else:
                self.target = new_target

    def update_targets(self, target_map): 
        with self.lock:
            if self.target is not None and self.target.url in target_map:
                self.target = self.target.updated_with(target_map[self.target.url])

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

            if self.target is not None:
                return self._compute_rates_with_target(dt, store)
            else:
                return self._compute_rates_momentum(dt, store)

    def _compute_rates_with_target(self, dt, store):
        now = astropy.time.Time.now()
        altaz, altaz_rates = self.target.altaz_and_rates_at_time(tracker = self, time = now)
        
        desired_axis_positions = [
            altaz.az,
            altaz.alt
        ] 
        # swizzle?
        desired_axis_rates = altaz_rates

        rates = np.zeros(2)
        if self.primary_telescope_connection is not None:
            for axis_index, pid_controller in enumerate(self.pid_controllers):            
                control_rate = pid_controller.compute_control_rate(
                    desired_position = desired_axis_positions[axis_index],
                    desired_rate = desired_axis_rates[axis_index],
                    desired_time = now,
                    commanded_rate = self.primary_telescope_connection.desired_axis_rates[axis_index],
                    measured_position = self.primary_telescope_connection.axis_angles[axis_index],
                    measured_position_time = self.primary_telescope_connection.axis_angles_measurement_time[axis_index],
                    store_state = store
                ).to_value(u.deg / u.s)
                rates[axis_index] = control_rate
        return rates

    def _compute_rates_momentum(self, dt, store):

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
                self.monotonic_time_ns_of_last_input = time.monotonic_ns()

            return rate



