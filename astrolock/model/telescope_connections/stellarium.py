import requests
import time
import astropy.units as u
import astropy.time
import math
import astrolock.model.alignment
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

        self.fake_misalignment = astrolock.model.alignment.AlignmentModel()
        if False:
            self.fake_misalignment.randomize()
            print(f"Stellarium connection using random fake misalignment: {self.fake_misalignment}")


    def loop(self):
        while not self.want_to_stop:
            try:
                #first we read some information, both to display, and because it's needed to set the rates

                self.tracker.notify_idle()

                status = requests.get('http:' + self.url_path + '/api/main/status')
                measurement_time = astropy.time.Time.now()

                status_json = status.json()
                fov_deg = float(status_json['view']['fov'])

                # Stellarium seems to have a bug where they missed a format specifier
                self.last_update_utc_str = status_json['time']['utc'].replace('.%1', '.0')
                gps_time = astropy.time.Time(self.last_update_utc_str, format='isot', scale='utc')

                if self.gps_time is None or np.abs((self.gps_time - gps_time).to_value(u.s)) > 10:
                    self.gps_time = gps_time
                    # a bit of a hack - since they only give second precision for this, only record the measurement time when the second changes,
                    # so that it should be right to within our update loop time.
                    self.gps_measurement_time = measurement_time

                view = requests.get('http:' + self.url_path + '/api/main/view?ref=on')
                measurement_time = astropy.time.Time.now()
                self.axis_angles_measurement_time[0] = measurement_time
                self.axis_angles_measurement_time[1] = measurement_time

                view_json = view.json()
                terrestrial_dir_str = view_json['altAz']
                terrestrial_dir_vec_str = terrestrial_dir_str.strip('[]').split(',')
                terrestrial_dir_vec = list(map(lambda s: float(s), terrestrial_dir_vec_str))
                # terrestrial_dir_vec is (South?, East, Up) ?

                terrestrial_dir_vec = np.array(terrestrial_dir_vec, dtype=np.float32) * np.array([-1.0, 1.0, 1.0], dtype=np.float32)

                self.axis_angles = u.Quantity(self.fake_misalignment.raw_axis_values_given_numpy_dir(terrestrial_dir_vec), unit=u.rad)
               
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

                depl = 0.0004 / 30 * 1000 * fov_deg
                move_x = self.desired_axis_rates[0].to_value(u.rad / u.s) / depl
                move_y = self.desired_axis_rates[1].to_value(u.rad / u.s) / depl

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

                self.record_loop_rate()
                
            except ConnectionError:
                self.want_to_stop = True
                pass