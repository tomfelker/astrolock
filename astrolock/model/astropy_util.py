import astropy.coordinates
import numpy as np

# one could use this globally like this:
#@astropy.coordinates.frame_transform_graph.transform(astropy.coordinates.FunctionTransform, astropy.coordinates.ITRS, astropy.coordinates.AltAz)
# but it changes the behavior globally, in a way that I'm not sure is correct.
def itrs_to_altaz_direct(obj_itrs_coord, home_altaz_frame):
    # what is the Alt and Az for me, standing at home, to view the object?
    home_itrs = home_altaz_frame.location.itrs.cartesian

    # hmm, not sure about this... we're taking "up" to mean "straight away from the center of the earth",
    # but presumably you really want "what you'd measure with a plumb line", i.e., perpendicular to the geoid,
    # which then might be approximated by the WGS84 ellipsoid.
    
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

def itrs_to_altaz_doc_recommended(obj_itrs_coord, home_altaz_frame):
    # https://docs.astropy.org/en/stable/coordinates/common_errors.html#altaz-calculations-for-earth-based-objects
    # https://github.com/astropy/astropy/issues/12678
    
    # The correct way to construct a SkyCoord object for a source that is directly overhead for a topocentric observer is as follows:

    t = astropy.time.Time('J2010')
    obj = obj_itrs_coord
    home = home_altaz_frame.location

    # First we make an ITRS vector of a straight overhead object
    itrs_vec = obj.get_itrs(t).cartesian - home.get_itrs(t).cartesian

    # Now we create a topocentric ITRS frame with this data
    itrs_topo = astropy.coordinates.ITRS(itrs_vec, obstime=t, location=home)

    # convert to AltAz
    aa = itrs_topo.transform_to(astropy.coordinates.AltAz(obstime=t, location=home))
    
    return aa

def itrs_to_altaz_new_recommended(obj_itrs_coord, home_altaz_frame, t):
    # https://docs.astropy.org/en/stable/whatsnew/5.2.html#topocentric-itrs-frame
    
    #A topocentric ITRS frame has been added that makes dealing with near-Earth objects easier and more intuitive.:

    obj = obj_itrs_coord
    home = home_altaz_frame.location

    # Direction of object from GEOCENTER
    itrs_geo = obj.cartesian

    # now get the Geocentric ITRS position of observatory
    obsrepr = home.get_itrs(t).cartesian

    # topocentric ITRS position of a straight overhead object
    itrs_repr = itrs_geo - obsrepr

    # create an ITRS object that appears straight overhead for a TOPOCENTRIC OBSERVER
    itrs_topo = astropy.coordinates.ITRS(itrs_repr, obstime=t, location=home)

    # convert to AltAz
    aa = itrs_topo.transform_to(astropy.coordinates.AltAz(obstime=t, location=home))
    
    return aa