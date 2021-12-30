import math
import requests
import threading
import time
import astropy.coordinates
import astropy.constants
import astropy.units as u
import numpy as np

import astrolock.model.target_source as target_source
import astrolock.model.target

# this is largely adapted from https://github.com/openskynetwork/opensky-api/blob/master/python/opensky_api.py

class OpenSkyTargetSource(target_source.TargetSource):
    def __init__(self, tracker = None):
        self.want_to_stop = True
        self.query_range = 15 * u.km
        self.api_url = "https://opensky-network.org/api"
        self.targets = []
        self.tracker = tracker

        
    def get_targets(self):
        return self.targets

    def loop(self):
        while not self.want_to_stop:
            url = self.api_url + "/states/all"
            params = {'time': 0}

            if self.query_range > 0:
                #todo: cool stuff like this
                lat = self.tracker.location.lat
                lon = self.tracker.location.lon
                #cart = self.tracker.location.geocentric
                #r = np.dot(cart, cart)
                #print(r)
                # todo: date line bugs...
                earth_circumference = astropy.constants.R_earth * 2.0 * math.pi;
                lat_range = self.query_range * (360.0 * u.deg / earth_circumference)
                lon_range = lat_range / np.cos(lat)
                params['lamin'] = (lat - lat_range).to_value(u.deg)
                params['lamax'] = (lat + lat_range).to_value(u.deg)
                params['lomin'] = (lon - lon_range).to_value(u.deg)
                params['lomax'] = (lon + lon_range).to_value(u.deg)

            r = requests.get(url, params = params, timeout = 15)
            if r.status_code == 200:
                self.targets = self.json_to_targets(r.json())                
            time.sleep(10)

    def start(self):
        self.want_to_stop = False
        self.loop_thread = threading.Thread(target = self.loop)
        self.loop_thread.daemon = True
        self.loop_thread.start()

    def stop(self):
        self.want_to_stop = True

    # must be in the order provided by the API
    json_keys = ["icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude", "on_ground",
            "velocity", "heading", "vertical_rate", "sensors",
            "geo_altitude", "squawk", "spi", "position_source"]

    def json_to_targets(self, json):
        targets = []

        if json["states"] is not None:
            for state_vector_array in json["states"]:
                props = dict(zip(self.json_keys, state_vector_array))
                # can't do much if it doesn't at least have these...
                if props['latitude'] is not None and props['longitude'] is not None:
                    new_target = astrolock.model.target.Target()
                    new_target.display_name = props["callsign"]
                    new_target.url = 'astrolock://skyvector/icao24/' + props['icao24']
                    new_target.skyvector_props = props

                    new_target.latitude_deg = float(props['latitude'])
                    new_target.longitude_deg = float(props['longitude'])
                    new_target.altitude_m = 0
                    if props['geo_altitude'] is not None:
                        new_target.altitude_m = float(props['geo_altitude']) 
                    if props['baro_altitude'] is not None:
                        new_target.altitude_m = float(props['baro_altitude']) 

                    # want this, but it's too slow to do one-by-one                
                    #new_target.location = astropy.coordinates.EarthLocation.from_geodetic(lon = new_target.longitudes_deg * u.deg, lat = new_target.latitudes_deg * u.deg, height = new_target.altitudes_m * u.m)
                    # so we'll do it below in a big array

                    targets.append(new_target)
            
            # doing this in one big array for performance reasons
            latitudes_deg = np.zeros(len(targets))
            longitudes_deg = np.zeros(len(targets))
            altitudes_m = np.zeros(len(targets))
            for target_index, target in enumerate(targets):
                latitudes_deg[target_index] = target.latitude_deg
                longitudes_deg[target_index] = target.longitude_deg
                altitudes_m[target_index] = target.altitude_m
            coordinates = astropy.coordinates.EarthLocation.from_geodetic(lon = longitudes_deg * u.deg, lat = latitudes_deg * u.deg, height = altitudes_m * u.m)
            for target_index, target in enumerate(targets):
                target.location = coordinates[target_index]
            
        return targets
        

