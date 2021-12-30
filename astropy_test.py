import astropy.coordinates
#from astropy.coordinates import frame_transform_graph
import numpy as np

import astropy.units as u
import time

home = astropy.coordinates.EarthLocation.from_geodetic(lat = "37d30'39.02\"", lon = "-122d16'19.33\"", height = 64)
obj = astropy.coordinates.EarthLocation.from_geodetic(lat = 37.45, lon = -122.32, height = 60000 * u.imperial.ft)

reference_obj_altaz = obj.itrs.transform_to(astropy.coordinates.AltAz(location = home, obstime = 'J2000'))
print(f"reference_obj_altaz{reference_obj_altaz}")

# this will fix it
import astrolock.model.astropy_util

#astropy.coordinates.TransformGraph().add_transform(astropy.coordinates.ITRS, astropy.coordinates.AltAz, itrs_to_altaz)

start_time_ns = time.monotonic_ns()

# crashes: AttributeError: 'NoneType' object has no attribute 'scale'
#obj_altaz = obj.itrs.transform_to(astropy.coordinates.AltAz(location = home))

# 500 ms
obj_altaz = obj.itrs.transform_to(astropy.coordinates.AltAz(location = home, obstime = 'J2000'))
            

#obj_altaz = obj.transform_to(home)

elapsed_ns = time.monotonic_ns() - start_time_ns
print(f"took {elapsed_ns * 1e-6} ms")
print(f"obj_altaz: {obj_altaz}")