import astrolock.model.target_source as target_source
import astrolock.model.target as target

import skyfield
import skyfield.api
import skyfield.data
import skyfield.data.hipparcos

import astropy.coordinates


class SkyfieldTarget(target.Target):
    def __init__(self, sf_target):
        super().__init__()
        self.sf_target = sf_target

    def altaz_at_time(self, tracker, time):
        observatory = tracker.home_planet_sf + tracker.location_sf
        time_sf = tracker.ts.from_astropy(time)
        observatory_barycentric = observatory.at(time_sf)
        target_astrometric = observatory_barycentric.observe(self.sf_target)
        target_apparent = target_astrometric.apparent()

        
        if tracker.primary_telescope_connection is not None:
            if tracker.primary_telescope_connection.want_atmospheric_refaction:
                temperature_C = tracker.primary_telescope_connection.temperature_C
            else:
                temperature_C = None
        else:
            temperature_C = 'standard'

        alt, az, distance = target_apparent.altaz(temperature_C=temperature_C)

        altaz = astropy.coordinates.AltAz(alt=alt.to(astropy.units.rad), az=az.to(astropy.units.rad), distance=distance.to(astropy.units.au))
        return altaz


class SkyfieldTargetSource(target_source.TargetSource):

    def __init__(self, tracker):
        super().__init__()
        self.target_map = {}
        self.tracker = tracker
        self.use_for_alignment = True
        self.started = False

    def start(self):
        self.load_targets()

    def load_targets(self, mag_limit = 3.0):

        loader = skyfield.api.Loader('./data/skyfield_cache')
        with loader.open(skyfield.data.hipparcos.URL) as f:
            stars_df = skyfield.data.hipparcos.load_dataframe(f)

        stars_df = stars_df[stars_df['magnitude'] <= mag_limit]

        star_names_url = 'https://www.pas.rochester.edu/~emamajek/WGSN/IAU-CSN.txt'
        hip_to_name = {}
        with loader.open(star_names_url) as f:
            # Sigh, have they not heard the gospel of JSON?  Or even CSV?  It's {datetime.date.today().year}!
            # The encoding is based on fixed columns, yet also it's UTF-8 encoded, so even that's complex!
            f_str = f.read().decode('UTF-8')
            for line in f_str.split('\n'):
                # omfg
                if len(line) < 1 or line[0] in ('#', '$'):
                    continue
                name = line[18:36].rstrip()
                hip = line[90:96]
                if '_' in hip:
                    continue
                hip = int(hip)
                hip_to_name[hip] = name


        planet_ephemeris = 'de440.bsp'
        planets = loader(planet_ephemeris)

        # 'global' variables heh heh
        self.tracker.home_planet_sf = planets['EARTH']
        self.tracker.ts = skyfield.api.load.timescale()

        for star_index in stars_df.index:
            star = skyfield.api.Star.from_dataframe(stars_df.loc[star_index])

            url = f'skyfield://stars/hip_main.dat/hip_{star_index}'

            new_target = SkyfieldTarget(star)
            new_target.url = url

            new_target.display_name = f"HIP {star_index}"
            if star_index in hip_to_name:
                new_target.display_name = f'{hip_to_name[star_index]} ({new_target.display_name})'


            self.target_map[url] = new_target
            
        planet_id_to_names = planets.names()
        for planet_id in planet_id_to_names.keys():
            if planet_id == 0:
                continue
            planet_names = planet_id_to_names[planet_id]
            planet_name = planet_names[0]
            planet_displayname = planet_names[-1]

            if 'EARTH' in planet_displayname:
                continue
            
            url = f'skyfield://planets/{planet_ephemeris}/{planet_name}'
            
            new_target = SkyfieldTarget(planets[planet_name])
            new_target.url = url
            new_target.display_name = planet_displayname

            self.target_map[url] = new_target

        # when we have both a planet and its barycenter, discard the barycenter
        # so we don't end up with two nearly identical targets, one of which is imaginary

        for url in list(self.target_map.keys()):
            if url.startswith('skyfield://planets/'):
                url_with_barycenter = url + '_BARYCENTER'
                if url_with_barycenter in self.target_map:
                    del self.target_map[url_with_barycenter]


            


    # maps urls to  targets    
    def get_target_map(self):
        return self.target_map