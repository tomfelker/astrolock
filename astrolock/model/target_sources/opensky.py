import math
import requests
import threading
import time
import astropy.coordinates
import astropy.constants
import astropy.units as u
import numpy as np

import skyfield
import skyfield.api

import astrolock.model.target_source as target_source
import astrolock.model.target

# this is largely adapted from https://github.com/openskynetwork/opensky-api/blob/master/python/opensky_api.py

class OpenSkyTargetSource(target_source.TargetSource):
    def __init__(self, tracker = None):
        self.want_to_stop = True
        self.query_range = 20 * u.km
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
                lat = self.tracker.location_ap.lat
                lon = self.tracker.location_ap.lon
                # todo: international date line and north/south pole bugs...
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

        json_states = json["states"] or []
        
        print(f'Parsing {len(json_states)} targets')
        start_perf_counter = time.perf_counter()

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
        
        print(f'took {(time.perf_counter() - start_perf_counter) * 1e3} ms')
        
        # doing this in one big array for performance reasons
        latitudes_deg = np.zeros(len(targets))
        longitudes_deg = np.zeros(len(targets))
        altitudes_m = np.zeros(len(targets))
        for target_index, target in enumerate(targets):
            latitudes_deg[target_index] = target.latitude_deg
            longitudes_deg[target_index] = target.longitude_deg
            altitudes_m[target_index] = target.altitude_m

        print(f'Astropy processing {len(targets)} targets')
        start_perf_counter = time.perf_counter()

        # astropy, with direct model
        if True:
            last_known_locations = astropy.coordinates.EarthLocation.from_geodetic(lon = longitudes_deg * u.deg, lat = latitudes_deg * u.deg, height = altitudes_m * u.m)
            
            # really shouldn't need to specify a time here, but astropy will crash if we don't - presumably it's trying to transform through a solar system barycentric frame
            tracker_altaz = astropy.coordinates.AltAz(location = self.tracker.location_ap, obstime = 'J2000')

            #25 ms, wtf?
            #and our fast version is still 6 ms
            #target_altaz = target.location.itrs.transform_to(tracker_altaz)
            # maybe that was just due to pathing to find the appropriate transform?
            # even that's still 5ish ms

            last_known_locations_itrs = last_known_locations.itrs
            altazs_from_tracker = astrolock.model.astropy_util.itrs_to_altaz_direct(last_known_locations_itrs, tracker_altaz)
            
            scores = altazs_from_tracker.alt.to_value(u.deg)

            #even this stuff is so slow we have to vectorize...
            display_columns = {}
            display_columns['latitude'] = last_known_locations.lat.to_string(decimal = True)
            display_columns['longitude'] = last_known_locations.lon.to_string(decimal = True)
            display_columns['altitude'] = altazs_from_tracker.alt.to_string(decimal = True)
            display_columns['azimuth'] = altazs_from_tracker.az.to_string(decimal = True)
            display_columns['distance'] = altazs_from_tracker.distance.to(u.km)
        
        #skyfield, broken vectorization, not going through earth position
        if False:
            # alas here too we need an arbitray time:
            ts = skyfield.api.load.timescale()
            t = ts.utc(2014, 1, 23, 11, 18, 7)
            times = skyfield.api.Time(ts, [t])

            # ...and, we can't vectorize, or at() will complain...  blah
            display_columns = {}
            display_columns['latitude'] = []
            display_columns['longitude'] = []
            display_columns['altitude'] = []
            display_columns['azimuth'] = []
            display_columns['distance'] = []
            scores = []

            for target_index, target in enumerate(targets):
                last_known_locations_geodetic = skyfield.api.wgs84.latlon(latitudes_deg[target_index], longitudes_deg[target_index], elevation_m = altitudes_m[target_index])
                tracker_geodetic = self.tracker.location_sf
                thingies = last_known_locations_geodetic - tracker_geodetic
                topocentric = thingies.at(t)
                alts, azs, dists = topocentric.altaz()
                scores.append(alts.degrees)

                display_columns['latitude'].append(latitudes_deg[target_index])
                display_columns['longitude'].append(longitudes_deg[target_index])
                display_columns['altitude'].append(alts.degrees)
                display_columns['azimuth'].append(azs.degrees)
                display_columns['distance'].append(dists.m)

        # skyfield, going through earth position
        if False:
            display_columns = {}

            ts = skyfield.api.load.timescale()
            t = ts.utc(2014, 1, 23, 11, 18, 7)
            planets = skyfield.api.load('de440s.bsp')
            earth = planets['earth']

            last_known_locations_geodetic = skyfield.api.wgs84.latlon(latitudes_deg, longitudes_deg, elevation_m = altitudes_m)
            last_known_locations_astrometric = earth + last_known_locations_geodetic
            
            tracker_geodetic = self.tracker.location_sfa
            tracker = earth + tracker_geodetic
            tracker_astrometric = tracker.at(t)

            last_known_locations_topocentric = tracker_astrometric.observe(last_known_locations_astrometric)
            
            alts, azs, dists = last_known_locations_topocentric.altaz()
            scores = alts.degrees

            display_columns['latitude'].append(latitudes_deg[target_index])
            display_columns['longitude'].append(longitudes_deg[target_index])
            display_columns['altitude'].append(alts.degrees)
            display_columns['azimuth'].append(azs.degrees)
            display_columns['distance'].append(dists.m)
            
        print(f'took {(time.perf_counter() - start_perf_counter) * 1e3} ms')

        for target_index, target in enumerate(targets):
            #target.last_known_location = last_known_locations[target_index]
            #target.last_known_location_itrs = last_known_locations_itrs[target_index]
            #target.altaz_from_tracker = altazs_from_tracker[target_index]
            target.score = scores[target_index]
            for column in display_columns.keys():
                target.display_columns[column] = display_columns[column][target_index]
        return targets
        

