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
import math
import datetime

class TrackerInput(object):
    def __init__(self):
        self.last_input_time_ns = time.perf_counter_ns()
        self.unconsumed_dt = 0.0
        self.sensitivity = -5
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
        self.search_forward_button = False
        self.search_backward_button = False
        self.integrated_search_time = 0.0


    def integrate_up_to(self, time_ns):
        if time_ns <= self.last_input_time_ns:
            return
        dt = (time_ns - self.last_input_time_ns) * 1e-9
        self.last_input_time_ns = time_ns

        self.unconsumed_dt += dt
        self.integrated_rates += self.last_rates * dt
        self.integrated_braking += self.last_braking * dt        
        if self.search_forward_button:
            self.integrated_search_time += dt
        if self.search_backward_button:
            self.integrated_search_time -= dt

    def consume_input_time_rates_and_braking(self):        
        # we will return the sum over all the time since the previous consume call.
        unconsumed_dt = self.unconsumed_dt
        integrated_rates = self.integrated_rates
        integrated_braking = self.integrated_braking
        integrated_search_time = self.integrated_search_time

        self.integrated_rates = np.zeros_like(self.integrated_rates)
        self.integrated_braking = np.zeros_like(self.integrated_braking)
        self.integrated_search_time = 0.0
        self.unconsumed_dt = 0.0        

        return unconsumed_dt, integrated_rates, integrated_braking, integrated_search_time

class TelescopeInfo:
    def __init__(self):
        self.native_focal_length = 2800 * u.mm
        self.aperture_diameter = 279.4 * u.mm
        self.barlow_zoom = 1.0

        # Celestron 93325 40mm Omni
        self.eyepiece_focal_length = 40.0 * u.mm
        self.eyepiece_afov = 43.0 * u.deg

        # ASI678MC
        self.sensor_size = np.array([7.7, 4.3]) * u.mm
        self.sensor_pixel_size = 2 * u.um

        self.update_derived()
        
    def update_derived(self):
        self.focal_length = self.native_focal_length * self.barlow_zoom
        self.f_number = self.focal_length / self.aperture_diameter
        self.eyepiece_zoom = self.focal_length / self.eyepiece_focal_length
        self.eyepiece_fov = np.arctan(np.tan(self.eyepiece_afov / 2.0) / self.eyepiece_zoom) * 2.0
        self.camera_fov = np.arctan(self.sensor_size / (2.0 * self.focal_length)) * 2.0
        # Note: this doesn't match Stellarium because they make this small angle approximation:
        #self.eyepiece_fov = ((self.eyepiece_afov / 2.0) / self.eyepiece_zoom) * 2.0
        #self.camera_fov = (self.sensor_size / (2.0 * self.focal_length)) * 2.0 * u.rad
        



class Tracker(object):
    
    def __init__(self):
        
        self.primary_telescope_connection = None
        self.primary_telescope_alignment = astrolock.model.alignment.AlignmentModel()

        self.use_telescope_location = True
        # TODO: load and save this from settings - for now, it's hardcoded to my backyard.
        self.user_location = astropy.coordinates.EarthLocation.from_geodetic(lat=37.51089 * u.deg, lon=-122.2719388888889 * u.deg, height=60 * u.m)

        self.telescope_info = TelescopeInfo()

        
        self.tracker_input = TrackerInput()
        self.smooth_input_mag = 0.0
        self.target = None
        self.use_telescope_time = True
        self.target_offset_max_lead_time = 300.0
        self.user_time_offset = 0 * u.s


        self.modes =  [
            # these ones require a target
            'target_with_time_offset',
            'target_with_spatial_offset',

            # these ones require alignment
            'sidereal',
            # TODO: 'angular_momentum'

            # these ones work at any time:
            'axis_momentum',
            'slew'
        ]


        # tracking state
        # TODO: these should perhaps be separate classes
        self.default_mode = 'axis_momentum'
        self.current_mode = self.default_mode
        self.momentum = np.zeros(2)
        self.target_offset_image_space = np.zeros(2)
        self.target_offset_lead_time = 0.0
        self.search_time = 0.0


        # todo: maths
        self.search_overlap_factor = 1.5
        self.search_fov_rad = self.telescope_info.camera_fov.to_value(u.rad) / self.search_overlap_factor
        self.search_fovs_per_sec = 1.0

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

        self.update_location()
    
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
            f"Tracking:\n"
            f"\tSensitivity:    {self.tracker_input.sensitivity}\n"
            f"\tInput:          [{math.degrees(self.tracker_input.last_rates[0]): 9.3f}, {math.degrees(self.tracker_input.last_rates[1]): 9.3f}] {'deg/s^2' if self.target is None else 'deg/s'}\n"
            f"\tMomentum:       [{math.degrees(self.momentum[0]): 9.3f}, {math.degrees(self.momentum[1]): 9.3f}] deg/s\n"
            f"\tSpatial offset: [{math.degrees(self.target_offset_image_space[0]): 9.3f}, {math.degrees(self.target_offset_image_space[1]): 9.3f}] deg\n"
            f"\tLead time:      {self.target_offset_lead_time: 6.3f} s\n"
            f"\tSearch time:    {self.search_time: 6.3f} s\n"
            f"Telescope:\n"
            f"\t{telescope_string}\n"
        )


    def is_mode_allowed(self, mode):
        if mode == 'target_with_time_offset' or mode == 'target_with_spatial_offset':
            return self.target is not None and self.primary_telescope_alignment.valid
        if mode == 'sidereal' or mode == 'angular_momentum':
            return self.primary_telescope_alignment.valid
        if mode == 'axis_momentum' or mode == 'slew':
            return True
        return False

    def set_target(self, new_target):
        if new_target is None:
            self.target = None
        elif self.target is not None and self.target.url == new_target.url and self.target != new_target:
            self.target = self.target.updated_with(new_target)
        else:
            self.target = new_target
            self.current_mode = 'target_with_time_offset'


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
            self.target_offset_lead_time = 0.0
            self.target_offset_image_space *= 0
            self.search_time = 0.0
            return np.zeros(2)
        
        if self.tracker_input.align_command:
            self.tracker_input.align_command = False
            self.tracker_input.integrated_rates += self.compute_image_space_search_offset()
            self.search_time = 0.0
            self.add_alignment_observation(self.tracker_input.last_input_time_ns)

        if not self.is_mode_allowed(self.current_mode):
            self.current_mode = self.default_mode
            assert(self.is_mode_allowed(self.current_mode))

        if self.current_mode == 'target_with_time_offset' or self.current_mode == 'target_with_spatial_offset':
            return self._compute_rates_with_target()
        elif self.current_mode == 'sidereal':
            return self._compute_rates_sidereal()
        elif self.current_mode == 'axis_momentum':
            return self._compute_rates_axis_momentum()
        elif self.current_mode == 'slew':
            return self._compute_rates_slew()
            

    def add_alignment_observation(self, observation_time_ns):
        if self.primary_telescope_connection is not None:
            estimated_current_axis_angles, current_time = self.primary_telescope_connection.get_estimated_axis_angles_and_time(observation_time_ns)
            new_datum = astrolock.model.alignment.AlignmentDatum(None, current_time, estimated_current_axis_angles)             
            if self.add_alignment_observation_callback is not None:
                self.add_alignment_observation_callback(new_datum)
    
    def compute_image_space_search_offset(self):
        arc_length = self.search_fovs_per_sec * self.search_time
        # b is how much we move out per radian of rotation        
        b = 1.0 / (2.0 * math.pi)
        theta = math.sqrt(2 * b * arc_length) / b
        # r is the current radius, given theta
        r = b * theta
        return np.array([
            r * math.cos(theta),
            r * math.sin(theta)
        ]) * self.search_fov_rad

    def _compute_rates_with_target(self):
        dir, rates = self.target.dir_and_rates_at_time(tracker = self, time = self.get_time())
        dir_norm = np.linalg.norm(dir)
        if dir_norm > 0.0:
            dir /= dir_norm
            rates /= dir_norm

        
        image_left, image_up = dir_to_image_left_and_up(dir)
        
        self.tracker_input.integrate_up_to(time.perf_counter_ns())
        input_dt, input_offset, input_braking, input_search_time = self.tracker_input.consume_input_time_rates_and_braking()
        # input_offset are units of joystick deflection * time.  Here we're thinking that your instantaneous deflection
        # commands an angular velocity, which when integrated, becomes an angle (which we treat as radians)
       
        self.search_time = max(0.0, self.search_time + input_search_time)

        # todo: braking, aka recentering - (a bit complex with this time thing...)

        if self.current_mode == 'target_with_time_offset':
            image_space_rates = np.array([np.dot(rates, image_left), np.dot(rates, image_up)])
            image_space_rates_norm = np.linalg.norm(image_space_rates)
            if image_space_rates_norm > 0.0:
                image_space_dir = image_space_rates / image_space_rates_norm
                desired_delta_lead_time = np.dot(input_offset, image_space_dir) / image_space_rates_norm
                old_lead_time = self.target_offset_lead_time
                self.target_offset_lead_time = np.clip(old_lead_time + desired_delta_lead_time, -self.target_offset_max_lead_time, self.target_offset_max_lead_time)
                delta_lead_time = self.target_offset_lead_time - old_lead_time
                input_offset -= delta_lead_time * image_space_rates

                # rotate this to be always perpendicular to motion
                old_target_offset_len = np.linalg.norm(self.target_offset_image_space)
                self.target_offset_image_space -= image_space_dir * np.dot(self.target_offset_image_space, image_space_dir)
                target_offset_len = np.linalg.norm(self.target_offset_image_space)
                if target_offset_len > 0.0:
                    self.target_offset_image_space *= old_target_offset_len / target_offset_len
        
        self.target_offset_image_space += input_offset

        

        search_offset_image_space = self.compute_image_space_search_offset()
        offset_image_space = self.target_offset_image_space + search_offset_image_space

        tweaked_dir = dir + (offset_image_space[0] * image_left + offset_image_space[1] * image_up) + rates * self.target_offset_lead_time

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

    def _compute_rates_sidereal(self):
        dir = self.primary_telescope_alignment.dir_given_numpy_raw_axis_values(self.primary_telescope_connection.axis_angles)

        azalt = np_xyz_to_azalt(dir)       
        alt_deg = np.rad2deg(azalt[1])
        az_deg = np.rad2deg(azalt[0])

        dt_s = 1.0
        time_sf = self.ts.from_astropy(self.get_time())
        time_plus_dt_sf = time_sf + datetime.timedelta(seconds=dt_s)

        apparent = self.location_sf.at(time_sf).from_altaz(alt_degrees=alt_deg, az_degrees=az_deg)
        radec = apparent.radec() # hmm, of time?
        star = skyfield.starlib.Star(ra = radec[0], dec=radec[1])

        observatory_barycentric = self.home_planet_sf + self.location_sf        
        altaz0 = observatory_barycentric.at(time_sf).observe(star).apparent().altaz()
        altaz1 = observatory_barycentric.at(time_plus_dt_sf).observe(star).apparent().altaz()

        azalts = np.array([
            [altaz0[1].radians, altaz0[0].radians],
            [altaz1[1].radians, altaz1[0].radians]
        ], dtype=np.float32)

        dirs = np_azalt_to_xyz(azalts)
        image_left, image_up = dir_to_image_left_and_up(dirs[0])

        self.tracker_input.integrate_up_to(time.perf_counter_ns())
        input_dt, input_offset, input_braking, input_search_time = self.tracker_input.consume_input_time_rates_and_braking()

        # TODO: integrate spiral search in somehow... a bit strange since really we're just commanding rates, not positions
        # also we need to do that in dir-space, not azalt space
        #self.search_time = max(0.0, self.search_time + input_search_time)
        #search_offset_image_space = self.compute_image_space_search_offset()

        dirs[1] += (input_offset[0] * image_left + input_offset[1] * image_up) * dt_s

        raws = self.primary_telescope_alignment.raw_axis_values_given_numpy_dir(dirs)
        rates = wrap_angle_plus_minus_pi_radians(raws[1] - raws[0]) / dt_s
       
        rates += input_offset

        return rates

    def _compute_rates_axis_momentum(self):            
        self.tracker_input.integrate_up_to(time.perf_counter_ns())
        input_dt, integrated_rates, integrated_braking, integrated_search_time = self.tracker_input.consume_input_time_rates_and_braking()
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
    
    def _compute_rates_slew(self):
        self.tracker_input.integrate_up_to(time.perf_counter_ns())
        input_dt, integrated_rates, integrated_braking, integrated_search_time = self.tracker_input.consume_input_time_rates_and_braking()
        return integrated_rates

    def get_time(self):
        if self.use_telescope_time and self.primary_telescope_connection is not None and self.primary_telescope_connection.gps_time is not None and self.primary_telescope_connection.gps_measurement_time_ns is not None:
            base_time = self.primary_telescope_connection.get_time()
        else:
            base_time = astropy.time.Time.now()
        return base_time + self.user_time_offset

    def update_location(self):
        if self.use_telescope_location and self.primary_telescope_connection is not None and self.primary_telescope_connection.gps_location is not None:
            self.location_ap = self.primary_telescope_connection.gps_location
        else:
            self.location_ap = self.user_location

        # note weird order
        lon, lat, height = self.location_ap.to_geodetic()
        self.location_sf = skyfield.api.wgs84.latlon(latitude_degrees=lat.to_value(u.deg), longitude_degrees=lon.to_value(u.deg), elevation_m=height.to_value(u.m))

        self.notify_status_changed()


