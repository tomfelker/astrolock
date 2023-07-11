import astrolock.model.astropy_util
import astropy.coordinates
import astropy.units as u
import astropy.units.imperial
import astropy.time
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

    @classmethod
    def from_gps(cls, lat_deg, lon_deg, alt_m = 0):
        ret = cls()
        ret.last_known_location = astropy.coordinates.EarthLocation.from_geodetic(lon = lon_deg * u.deg, lat = lat_deg * u.deg, height = alt_m * u.m)
        #todo: is this right with the craziness?
        ret.last_known_location_itrs = ret.last_known_location.itrs
        return ret

    def get_status(self):
        return (
            f'\t{self.display_name}\n'
            f'\t{self.url}\n'
            f'\textrapolated speed: {self.extrapolated_velocity_itrs.norm().to(u.imperial.kt) if self.extrapolated_velocity_itrs is not None else ""}\n'
            f'\textrapolated accel: {self.extrapolated_acceleration_itrs.norm().to(u.m / (u.s * u.s)) if self.extrapolated_acceleration_itrs is not None else ""}\n'
            f'\t\n'
        )

    # callers will do  target = target.updated_with(new_target)
    def updated_with(self, new_target):
        old_target = self
        old_target.prev_target = None
        new_target.prev_target = old_target
        
        if old_target.last_known_location_time is None or new_target.last_known_location_time is None:
            return new_target

        time_between_updates = new_target.last_known_location_time - old_target.last_known_location_time
        if time_between_updates > 0 * u.s:
            delta_location_itrs = new_target.last_known_location_itrs.cartesian - old_target.last_known_location_itrs.cartesian
            new_target.extrapolated_velocity_itrs = delta_location_itrs / time_between_updates
            if old_target.extrapolated_velocity_itrs is not None:
                delta_velocity_itrs = new_target.extrapolated_velocity_itrs - old_target.extrapolated_velocity_itrs
                new_target.extrapolated_acceleration_itrs = delta_velocity_itrs / time_between_updates

        return new_target

    def altaz_at_time(self, tracker, time):
        if self.last_known_location_time is not None:
            time_since_update = time - self.last_known_location_time
        else:
            time_since_update = 0 * u.s
        location_itrs_at_time = self.last_known_location_itrs.cartesian
        
        if self.extrapolated_velocity_itrs is not None:
            location_itrs_at_time += self.extrapolated_velocity_itrs * time_since_update

        if self.extrapolated_acceleration_itrs is not None:
            location_itrs_at_time += 0.5 * self.extrapolated_acceleration_itrs * time_since_update * time_since_update

        location_itrs_at_time = astropy.coordinates.ITRS(location_itrs_at_time)

        obstime = astropy.time.Time('J2000')

        # todo: fix this...
        # really shouldn't need to specify a time here, but astropy will crash if we don't - presumably it's trying to transform through a solar system barycentric frame
        #todo: cache this
        tracker_altaz = astropy.coordinates.AltAz(location = tracker.location_ap, obstime = obstime)

        #altaz = astrolock.model.astropy_util.itrs_to_altaz_direct(location_itrs_at_time, tracker_altaz)
        altaz = astrolock.model.astropy_util.itrs_to_altaz_new_recommended(location_itrs_at_time, tracker_altaz, t=obstime)

        return altaz

    def altaz_and_rates_at_time(self, tracker, time, dt = 1.0 * u.s):
        
        altaz = self.altaz_at_time(tracker=tracker, time=time)
        future_altaz = self.altaz_at_time(tracker=tracker, time=time + dt)

        rates = [
            (future_altaz.az - altaz.az).wrap_at(180 * u.deg) / dt,
            (future_altaz.alt - altaz.alt).wrap_at(180 * u.deg) / dt
        ]
        
        return altaz, rates
    
    def dir_and_rates_at_time(self, tracker, time, dt = 1.0 * u.s):

        altaz = self.altaz_at_time(tracker=tracker, time=time)
        future_altaz = self.altaz_at_time(tracker=tracker, time=time + dt)

        dir = altaz.cartesian.xyz.to_value()
        future_dir = future_altaz.cartesian.xyz.to_value()

        rates = (future_dir - dir) / dt.to_value(u.s)

        return dir, rates
                
    