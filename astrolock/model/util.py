import math
import numpy as np

def lerp(a, b, t):
    return (1 - t) * a + t * b


def wrap_angle_plus_minus_pi_radians(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def location_to_string(location):
    if location is not None:
        return f'{location.geodetic.lat}, {location.geodetic.lon}, {location.geodetic.height}'
    return None