from astrolock.model.telescope_connection import TelescopeConnection
from astrolock.model.pid import PIDController
import astrolock.model.target_sources.opensky
import astrolock.model.target_sources.kml
import astrolock.model.target_sources.skyfield
import astrolock.model.alignment
from astrolock.model.util import *

from astropy import units as u
import astropy.time
import skyfield
import time
import numpy as np
import threading
import copy
import math

class TrackerInput(object):
    def __init__(self):
        self.last_input_time_ns = time.perf_counter_ns()
        self.unconsumed_dt = 0.0
        self.sensitivity = -2
        self.sensitivity_scale = 1.0
        self.last_rates = np.zeros(2, dtype=np.float32)
        self.integrated_rates = np.zeros(2, dtype=np.float32)
        self.last_braking = 0.0
        self.integrated_braking = 0.0
        self.emergency_stop_command = False
        self.align_command = False
        self.reset_command = False
        self.sensitivity_decrease_button = False
        self.sensitivity_increase_button = False
        self.align_button = False
        self.reset_offset_button = False


    def consume_input_time_rates_and_braking(self):
        ns = time.perf_counter_ns()

        # integrate up however much time since the last input, since it hasn't been added yet:
        dt = (ns - self.last_input_time_ns) * 1e-9
        self.last_input_time_ns = ns
        self.integrated_rates += self.last_rates * dt
        self.integrated_braking += self.last_braking * dt
        self.unconsumed_dt += dt

        # we will return the sum over all the time since the previous consume call.
        integrated_rates = self.integrated_rates
        integrated_braking = self.integrated_braking

        self.integrated_rates = np.zeros_like(self.integrated_rates)
        self.integrated_braking = np.zeros_like(self.integrated_braking)
        self.unconsumed_dt = 0

        return dt, integrated_rates, integrated_braking

class Tracker(object):
    
    def __init__(self):
        
        self.primary_telescope_connection = None
        self.primary_telescope_alignment = astrolock.model.alignment.AlignmentModel()
        # todo: this should come from
        # the hand controller of the telescope if available, or
        # from the PC if there were any decent APIs for it, or
        # the GUI, but for now, hardcode:
        lat_deg, lon_deg, height_m = 37.51089, -122.2719388888889, 60
        self.location_ap = astropy.coordinates.EarthLocation.from_geodetic(lat = lat_deg * u.deg, lon = lon_deg * u.deg, height = height_m * u.m)
        self.location_sf = skyfield.api.wgs84.latlon(latitude_degrees = lat_deg, longitude_degrees = lon_deg, elevation_m = height_m)
       
        # add yourself here to be notified when any of the above change
        self.settings_observers = []

        self.momentum = np.zeros(2)
        self.rates = np.zeros(2)
        self.tracker_input = TrackerInput()
        self.smooth_input_mag = 0.0
        self.target = None
        self.use_telescope_time = True
        self.target_offset_using_time = True
        self.target_offset_max_lead_time = 30.0
        self.user_time_offset = astropy.time.TimeDelta(0 * u.s)
        self.target_offset_image_space = np.zeros(2)
        self.target_offset_lead_time = 0.0

        self.status_observers = []

        self.idle_observers = []

        self.add_alignment_observation_callback = None

        self.target_source_map = {
            'Skyfield': astrolock.model.target_sources.skyfield.SkyfieldTargetSource(self),
            'OpenSky': astrolock.model.target_sources.opensky.OpenSkyTargetSource(self),
            'KML': astrolock.model.target_sources.kml.KmlTargetSource(self),
        }

        self.pid_controllers = []
        for i in range(2):
            pid_controller = PIDController()
            self.pid_controllers.append(pid_controller)
    
    def notify_settings_changed(self):
        for observer in self.settings_observers:
            observer.tracker_settings_changed()

    def notify_status_changed(self):
        for observer in self.status_observers:
            observer.on_tracker_status_changed()

    def notify_idle(self):
        """
        Telescope connections will call this when they are about to wait for a response for the telescope, which is a good time
        for us to do GIL-hogging things like GUI updates, garbage collecting, or updating caches.
        
        TODO: tracker should have its own thread to service these (rather than, as now, relying on Tkinter to schedule the work)
        TODO: callers could specify how long they expect to be idle (best case can be calculated from baud rate)
        """
        for observer in self.idle_observers:
            observer.on_tracker_idle()
    
    def get_recommended_connection_urls(self):
        return TelescopeConnection.get_urls_for_subclasses()

    def connect_to_telescope(self, url):
        if (self.primary_telescope_connection):
            self.disconnect_from_telescope()
        self.primary_telescope_connection = TelescopeConnection.get_connection(url, self)

        # so we don't immediately process all the stick inputs from before we were connected
        self.tracker_input = TrackerInput()

        if self.primary_telescope_connection:
            self.primary_telescope_connection.start()

    def disconnect_from_telescope(self):
        if self.primary_telescope_connection:
            self.primary_telescope_connection.stop()
            self.primary_telescope_connection = None

    def get_status(self):
        if self.primary_telescope_connection:
            telescope_string = 'Connected to ' + self.primary_telescope_connection.url + '\n' + self.primary_telescope_connection.get_status()
        else:
            telescope_string = 'Not connected'
        return (
            f"Target:\n"
            f"{self.target.get_status() if self.target is not None else 'No target'}\n"
            f"Input:\n"
            f"\tSensitivity:    {self.tracker_input.sensitivity}\n"
            f"\tLast Rates:     [{math.degrees(self.tracker_input.last_rates[0]): 9.3f}, {math.degrees(self.tracker_input.last_rates[1]): 9.3f}] {'deg/s^2' if self.target is None else 'deg/s'}\n"
            f"\tMomentum:       [{math.degrees(self.momentum[0]): 9.3f}, {math.degrees(self.momentum[1]): 9.3f}] deg/s\n"
            f"\tSpatial offset: [{math.degrees(self.target_offset_image_space[0]): 9.3f}, {math.degrees(self.target_offset_image_space[1]): 9.3f}] deg\n"
            f"\tLead time:      {self.target_offset_lead_time: 6.3f} s\n"
            f"Telescope:\n"
            f"\t{telescope_string}\n"
        )

    def set_target(self, new_target):
        if new_target is None:
            self.target = None
        elif self.target is not None and self.target.url == new_target.url and self.target != new_target:
            self.target = self.target.updated_with(new_target)
        else:
            self.target = new_target

    def update_targets(self, target_map): 
        if self.target is not None and self.target.url in target_map:
            self.target = self.target.updated_with(target_map[self.target.url])

    def consume_input_and_calculate_raw_axis_rates(self):

        if self.tracker_input.emergency_stop_command:
            self.tracker_input.emergency_stop_command = False

            self.set_target(None)
            self.momentum *= 0
            self.target_offset_lead_time = 0
            self.target_offset_image_space *= 0
            return np.zeros(2)
        
        if self.tracker_input.reset_command:
            self.tracker_input.reset_command = False
            self.target_offset_lead_time = 0
            self.target_offset_image_space *= 0            
            return np.zeros(2)
        
        if self.tracker_input.align_command:
            self.tracker_input.align_command = False
            self.add_alignment_observation(self.tracker_input.last_input_time_ns)

        if self.target is not None:
            return self._compute_rates_with_target()
        else:
            return self._compute_rates_momentum()

    def add_alignment_observation(self, observation_time_ns):
        if self.primary_telescope_connection is not None:
            estimated_current_axis_angles, current_time = self.primary_telescope_connection.get_estimated_axis_angles_and_time(observation_time_ns)
            new_datum = astrolock.model.alignment.AlignmentDatum(None, current_time, estimated_current_axis_angles)             
            if self.add_alignment_observation_callback is not None:
                self.add_alignment_observation_callback(new_datum)
        
    def _compute_rates_with_target(self):
        dir, rates = self.target.dir_and_rates_at_time(tracker = self, time = self.get_time())
        dir_norm = np.linalg.norm(dir)
        dir /= dir_norm
        rates /= dir_norm

        image_left = np.cross(np.array([0.0, 0.0, 1.0]), dir)
        image_left /= np.linalg.norm(image_left)
        image_up = np.cross(dir, image_left)
        image_up /= np.linalg.norm(image_up)

        input_dt, integrated_rates, integrated_braking = self.tracker_input.consume_input_time_rates_and_braking()
        # Integrated_rates are units of joystick deflection * time.  Here we're thinking that your instantaneous deflection
        # commands an angular velocity, which when integrated, becomes an angle (which we treat as radians)

        self.target_offset_image_space += integrated_rates

        # todo: braking, aka recentering - (a bit complex with this time thing...)

        if self.target_offset_using_time:
            image_space_rates = np.array([np.dot(rates, image_left), np.dot(rates, image_up)])
            image_space_rates_norm = np.linalg.norm(rates)
            if image_space_rates_norm > 0.0:
                desired_delta_lead_time = np.dot(self.target_offset_image_space, image_space_rates) / np.square(image_space_rates_norm)
                old_lead_time = self.target_offset_lead_time
                self.target_offset_lead_time = np.clip(old_lead_time + desired_delta_lead_time, -self.target_offset_max_lead_time, self.target_offset_max_lead_time)
                delta_lead_time = self.target_offset_lead_time - old_lead_time
                self.target_offset_image_space -= delta_lead_time * image_space_rates

        tweaked_dir = dir + (self.target_offset_image_space[0] * image_left + self.target_offset_image_space[1] * image_up) + rates * self.target_offset_lead_time

        desired_dir = np.array(tweaked_dir, dtype=np.float32)
        desired_rates = np.array(rates, dtype=np.float32)

        desired_raw_axis_positions = self.primary_telescope_alignment.raw_axis_values_given_numpy_dir(desired_dir)

        # haha, just inverting was hard enough, let's differentiate numerically...
        if input_dt > 0:
            desired_true_axis_positions_after_dt = desired_dir + desired_rates * input_dt
            desired_raw_axis_positions_after_dt = self.primary_telescope_alignment.raw_axis_values_given_numpy_dir(desired_true_axis_positions_after_dt)
            desired_raw_axis_rates = wrap_angle_plus_minus_pi_radians(desired_raw_axis_positions_after_dt - desired_raw_axis_positions) / input_dt
        else:
            desired_raw_axis_rates = np.zeros(2)

        ns = time.perf_counter_ns()
        rates = np.zeros(2)
        if self.primary_telescope_connection is not None:
            for axis_index, pid_controller in enumerate(self.pid_controllers):            
                control_rate = pid_controller.compute_control_rate(
                    desired_position = desired_raw_axis_positions[axis_index],
                    desired_rate = desired_raw_axis_rates[axis_index],
                    commanded_rate = self.primary_telescope_connection.desired_axis_rates[axis_index],
                    measured_position = self.primary_telescope_connection.axis_angles[axis_index],
                    measurement_seconds_ago = (ns - self.primary_telescope_connection.axis_measurement_times_ns[axis_index]) * 1e-9,
                )
                rates[axis_index] = control_rate

        return rates

    def _compute_rates_momentum(self):
            input_dt, integrated_rates, integrated_braking = self.tracker_input.consume_input_time_rates_and_braking()
            # Integrated_rates are units of joystick deflection * time.  Here we're thinking that your instantaneous deflection
            # commands an angular acceleration, which becomes a delta angular velocity when integrated (and we'll think of it in rad/s).
            if input_dt > 0:
                rc = 2.0
                alpha = input_dt / (rc + input_dt)            

                self.momentum = (self.momentum + integrated_rates / input_dt) * alpha + self.momentum * (1.0 - alpha)
                momentum_mag = np.sqrt(np.dot(self.momentum, self.momentum))
                if momentum_mag > 0:
                    momentum_desired_mag = np.maximum(0, momentum_mag - integrated_braking / input_dt)
                    self.momentum *= momentum_desired_mag / momentum_mag

            return self.momentum

    def get_time(self):
        if self.use_telescope_time and self.primary_telescope_connection is not None and self.primary_telescope_connection.gps_time is not None and self.primary_telescope_connection.gps_measurement_time_ns is not None:
            return self.primary_telescope_connection.get_time()
        return astropy.time.Time.now()


