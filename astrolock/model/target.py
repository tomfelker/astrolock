import astrolock.model.astropy_util
import astropy.coordinates
import astropy.units as u
import numpy as np

class Target:
    def __init__(self):
        self.display_name = ''
        self.url = ''
        self.altaz_from_tracker = None
        self.last_known_location = None
        self.last_known_location_time = None
        self.prev_target = None
        self.score = -float('inf')
        self.display_columns = {}

    # callers will do  target = target.updated_with(new_target)
    def updated_with(self, new_target):
        self.prev_target = None
        new_target.prev_target = self
        return new_target

    def altaz_and_rates_at_time(self, tracker, time, hack = True):
        if self.prev_target is not None:
            time_between_updates = self.last_known_location_time - self.prev_target.last_known_location_time
            if time_between_updates > 0 * u.s:
                time_since_penultimate_update = time - self.last_known_location_time

                prev_to_current_location_itrs = self.last_known_location_itrs.cartesian - self.prev_target.last_known_location_itrs.cartesian
                velocity_itrs = prev_to_current_location_itrs / time_between_updates
                location_itrs_at_time = astropy.coordinates.ITRS(self.prev_target.last_known_location_itrs.cartesian + velocity_itrs * time_since_penultimate_update)

                # todo: fix this...
                # really shouldn't need to specify a time here, but astropy will crash if we don't - presumably it's trying to transform through a solar system barycentric frame
                #todo: cache this
                tracker_altaz = astropy.coordinates.AltAz(location = tracker.location_ap, obstime = 'J2000')

                altaz = astrolock.model.astropy_util.itrs_to_altaz_direct(location_itrs_at_time, tracker_altaz)
                
                rates = np.zeros(2) * u.deg / u.s
                if hack:
                    dt = 1.0 * u.s
                    future_altaz, dummy_rates = self.altaz_and_rates_at_time(tracker = tracker, time = time + dt, hack = False)
                    rates = [
                        (future_altaz.az - altaz.az) / dt,
                        (future_altaz.alt - altaz.alt) / dt
                    ]
                    
                return altaz, rates
        return self.altaz_from_tracker, np.zeros(2) * u.deg / u.s

                

                
    