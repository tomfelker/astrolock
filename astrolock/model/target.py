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
        self.extrapolated_velocity_itrs = None
        self.extrapolated_acceleration_itrs = None
        self.prev_target = None
        self.score = -float('inf')
        self.display_columns = {}

    # callers will do  target = target.updated_with(new_target)
    def updated_with(self, new_target):
        old_target = self
        old_target.prev_target = None
        new_target.prev_target = old_target
        
        

        time_between_updates = new_target.last_known_location_time - old_target.last_known_location_time
        if time_between_updates > 0 * u.s:
            delta_location_itrs = new_target.last_known_location_itrs.cartesian - old_target.last_known_location_itrs.cartesian
            new_target.extrapolated_velocity_itrs = delta_location_itrs / time_between_updates
            if old_target.extrapolated_velocity_itrs is not None:
                delta_velocity_itrs = new_target.extrapolated_velocity_itrs - old_target.extrapolated_velocity_itrs
                new_target.extrapolated_acceleration_itrs = delta_velocity_itrs / time_between_updates

        return new_target

    def altaz_and_rates_at_time(self, tracker, time, dt = 1.0 * u.s, hack = True):
        time_since_update = time - self.last_known_location_time
        location_itrs_at_time = self.last_known_location_itrs.cartesian
        
        if self.extrapolated_velocity_itrs is not None:
            location_itrs_at_time += self.extrapolated_velocity_itrs * time_since_update

        if self.extrapolated_acceleration_itrs is not None:
            location_itrs_at_time += 0.5 * self.extrapolated_acceleration_itrs * time_since_update * time_since_update

        location_itrs_at_time = astropy.coordinates.ITRS(location_itrs_at_time)

        # todo: fix this...
        # really shouldn't need to specify a time here, but astropy will crash if we don't - presumably it's trying to transform through a solar system barycentric frame
        #todo: cache this
        tracker_altaz = astropy.coordinates.AltAz(location = tracker.location_ap, obstime = 'J2000')

        altaz = astrolock.model.astropy_util.itrs_to_altaz_direct(location_itrs_at_time, tracker_altaz)
        
        rates = np.zeros(2) * u.deg / u.s
        if hack:
            future_altaz, dummy_rates = self.altaz_and_rates_at_time(tracker = tracker, time = time + dt, hack = False)
            rates = [
                (future_altaz.az - altaz.az).wrap_at(180 * u.deg) / dt,
                (future_altaz.alt - altaz.alt).wrap_at(180 * u.deg) / dt
            ]
            
        return altaz, rates
                

                
    