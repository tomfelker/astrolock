import astropy.coordinates
import numpy as np

print("Adding simple ITRS to AltAz transform.")

@astropy.coordinates.frame_transform_graph.transform(astropy.coordinates.FunctionTransform, astropy.coordinates.ITRS, astropy.coordinates.AltAz)
def itrs_to_altaz(obj_itrs_coord, home_altaz_frame):
    # what is the Alt and Az for me, standing at home, to view the object?
    home_itrs = home_altaz_frame.location.itrs.cartesian
    home_up = home_itrs / home_itrs.norm()
    home_east = astropy.coordinates.CartesianRepresentation(-home_itrs.y, home_itrs.x, home_itrs.z * 0)
    home_east /= home_east.norm()
    home_north = home_up.cross(home_east)
    
    home_to_obj = obj_itrs_coord.cartesian - home_itrs

    home_to_obj_dot_up = home_to_obj.dot(home_up)
    home_to_obj_dot_east = home_to_obj.dot(home_east)
    home_to_obj_dot_north = home_to_obj.dot(home_north)

    az = astropy.coordinates.Longitude(np.arctan2(home_to_obj_dot_east, home_to_obj_dot_north))
    home_to_obj_forward = np.sqrt(np.square(home_to_obj_dot_east) + np.square(home_to_obj_dot_north))
    alt = astropy.coordinates.Latitude(np.arctan2(home_to_obj_dot_up, home_to_obj_forward))
    distance = home_to_obj.norm()
    return astropy.coordinates.AltAz(alt = alt, az = az, distance = distance)
