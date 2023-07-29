import requests
import time
import astropy.units as u
import astropy.time
import math
import astrolock.model.alignment
from astrolock.model.util import *
import numpy as np
import torch
import gc

import astrolock.model.telescope_connections.threaded as threaded

class StellariumConnection(threaded.ThreadedConnection):
    @staticmethod
    def get_url_scheme():
        return 'stellarium:'

    @classmethod
    def get_urls(cls):
        return [ cls.get_url_scheme() + '//localhost:8090' ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # this API can't easily tell us the AltAz without refraction, so we'll just have to turn off refraction in the GUI
        # and request refraction not be used when giving us coordinates
        self.want_atmospheric_refaction = False

        self.want_sleep_inhibited = False

        # my telescope can slew nearly 4 deg/s, probably dependent on battery voltage - let's simulate this to see how it acts near the poles
        self.fake_max_rates = np.deg2rad(np.array([3.0, 3.0]))

        

        self.fake_misalignment = astrolock.model.alignment.AlignmentModel()
        do_fake_misalignment = False
        if do_fake_misalignment:
            self.fake_misalignment.randomize()
            print(f"Stellarium connection using random fake misalignment: {self.fake_misalignment}")
        else:
            self.tracker.primary_telescope_alignment.valid = True


    def loop(self):
        while not self.want_to_stop:
            try:
                #first we read some information, both to display, and because it's needed to set the rates

                self.tracker.notify_idle()

                status = requests.get('http:' + self.url_path + '/api/main/status')
                measurement_time_ns = time.perf_counter_ns()

                status_json = status.json()
                fov_deg = float(status_json['view']['fov'])

                # Stellarium seems to have a bug where they missed a format specifier
                self.last_update_utc_str = status_json['time']['utc'].replace('.%1', '.0')
                gps_time = astropy.time.Time(self.last_update_utc_str, format='isot', scale='utc')

                self._set_gps_time_with_inferred_seconds_fraction(
                    gps_time,
                    measurement_time_ns
                )

                if self.gps_requested:
                    self.gps_requested = False
                    self.gps_location = astropy.coordinates.EarthLocation.from_geodetic(lat = status_json['location']['latitude'] * u.deg, lon = status_json['location']['longitude'] * u.deg, height = status_json['location']['altitude'] * u.m)
                    self.tracker.update_location()

                view = requests.get('http:' + self.url_path + '/api/main/view?ref=on')
                measurement_time_ns = time.perf_counter_ns()
                axis_dt = (measurement_time_ns - self.axis_measurement_times_ns[0]) * 1e-9
                self.axis_measurement_times_ns[0] = measurement_time_ns
                self.axis_measurement_times_ns[1] = measurement_time_ns

                view_json = view.json()
                terrestrial_dir_str = view_json['altAz']
                terrestrial_dir_vec_str = terrestrial_dir_str.strip('[]').split(',')
                terrestrial_dir_vec = list(map(lambda s: float(s), terrestrial_dir_vec_str))
                # terrestrial_dir_vec is (South?, East, Up) ?

                terrestrial_dir_vec = np.array(terrestrial_dir_vec, dtype=np.float32) * np.array([-1.0, 1.0, 1.0], dtype=np.float32)

                old_axis_angles = self.axis_angles.copy()
                self.axis_angles = self.fake_misalignment.raw_axis_values_given_numpy_dir(terrestrial_dir_vec)
                self.estimated_axis_rates = wrap_angle_plus_minus_pi_radians(self.axis_angles - old_axis_angles) / axis_dt

                # now we will set our rates, which requires knowing the FOV

                self.desired_axis_rates = self.tracker.consume_input_and_calculate_raw_axis_rates()

                # this is what controls the move speed, which we are inverting:
                #
                # https://fossies.org/linux/stellarium/plugins/RemoteControl/src/MainService.cpp
                #
                #       double currentFov = mvmgr->getCurrentFov();
                #       // the more it is zoomed, the lower the moving speed is (in angle)
                #       //0.0004 is the default key move speed
                #       double depl=0.0004 / 30 *deltaTime*1000*currentFov;
                #       double deltaAz = moveX*depl;
                #       double deltaAlt = moveY*depl;
                #
                # hence these hard-coded numbers:

                limited_desired_axis_rates = np.clip(self.desired_axis_rates, -self.fake_max_rates, self.fake_max_rates)

                depl = 0.0004 / 30 * 1000 * fov_deg
                move_x = limited_desired_axis_rates[0] / depl
                move_y = limited_desired_axis_rates[1] / depl

                requests.post('http:' + self.url_path + '/api/main/move', data = {'x': move_x, 'y': move_y}) 
                
                sleep_start_ns = time.perf_counter_ns()
                sleep_time = .1
                time.sleep(sleep_time)
                slept_for = (time.perf_counter_ns() - sleep_start_ns) * 1e-9
                warn_time = .02
                if slept_for > sleep_time + warn_time:
                    print(f"Warning!  Tracker thread overslept by {(slept_for - sleep_time) * 1000} ms, somebody's hogging the GIL!")

                # NOTE: Stellarium takes much longer to process each request (~8 ms -> ~40 ms) when moving, so it's not completely
                # our fault that the loop rate tanks when we start tracking stuff.

                self.tracker.notify_status_changed()
                self.record_loop_rate()                

            except ConnectionError:
                self.want_to_stop = True
                pass