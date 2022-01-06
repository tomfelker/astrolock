
import skyfield
import skyfield.api

ts = skyfield.api.load.timescale()
t = ts.J2000
planets = skyfield.api.load('de440s.bsp')
earth = planets['earth']

obj_geodetic = skyfield.api.wgs84.latlon(52, -1, elevation_m = 1000)
obj_geometric = earth + obj_geodetic

home_geodetic = skyfield.api.wgs84.latlon(52, -1, elevation_m = 0)
home_geometric = earth + home_geodetic

home_astrometric = home_geometric.at(t)
obj_topocentric = home_astrometric.observe(obj_geometric)
obj_apparent = obj_topocentric.apparent()
obj_altaz = obj_apparent.altaz()

print(obj_altaz)