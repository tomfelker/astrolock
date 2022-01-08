import requests
import time
import astropy.units as u
import astropy.time
import math


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

    def loop(self):
        while not self.want_to_stop:
            try:
                #first we read some information, both to display, and because it's needed to set the rates
                
                status = requests.get('http:' + self.url_path + '/api/main/status')
                status_json = status.json()
                fov_deg = float(status_json['view']['fov'])

                self.last_update_utc_str = status_json['time']['utc']

                view = requests.get('http:' + self.url_path + '/api/main/view')
                
                measurement_time = astropy.time.Time.now()
                self.axis_angles_measurement_time[0] = measurement_time
                self.axis_angles_measurement_time[1] = measurement_time

                view_json = view.json()
                terrestrial_dir_str = view_json['altAz']
                terrestrial_dir_vec_str = terrestrial_dir_str.strip('[]').split(',')
                terrestrial_dir_vec = list(map(lambda s: float(s), terrestrial_dir_vec_str))
                alt_rad = math.asin(terrestrial_dir_vec[2])
                az_rad = math.atan2(terrestrial_dir_vec[1], -terrestrial_dir_vec[0])
                self.axis_angles[0] = az_rad * u.rad
                self.axis_angles[1] = alt_rad * u.rad
                
                # now we will set our rates, which requires knowing the FOV

                #self.tracker.update_gui_callback()
                self.desired_axis_rates = self.tracker.get_rates()


                # this is what controls the move speed, which we are inverting:
                # https://fossies.org/linux/stellarium/plugins/RemoteControl/src/MainService.cpp
                #double currentFov = mvmgr->getCurrentFov();
                #// the more it is zoomed, the lower the moving speed is (in angle)
                #//0.0004 is the default key move speed
                #double depl=0.0004 / 30 *deltaTime*1000*currentFov;
                #double deltaAz = moveX*depl;
                #double deltaAlt = moveY*depl;

                depl = 0.0004 / 30 * 1000 * fov_deg
                move_x = self.desired_axis_rates[0].to_value(u.rad / u.s) / depl
                move_y = self.desired_axis_rates[1].to_value(u.rad / u.s) / depl

                requests.post('http:' + self.url_path + '/api/main/move', data = {'x': move_x, 'y': move_y}) 

                #self.tracker.update_gui_callback()
                time.sleep(.1)
                self.record_loop_rate()
            except ConnectionError:
                self.want_to_stop = True
                pass