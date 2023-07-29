import math
import numpy as np
import os
import ctypes
import enum

def lerp(a, b, t):
    return (1 - t) * a + t * b


def wrap_angle_plus_minus_pi_radians(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

def np_xyz_to_azalt(xyz):
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]    
    xy_len = np.hypot(x, y)
    alt = np.arctan2(z, xy_len)
    az = np.arctan2(y, x)
    return np.stack([az, alt], axis = -1)


def np_azalt_to_xyz(azalt):
    az = azalt[..., 0]
    alt = azalt[..., 1]
    cos_alt = np.cos(alt)
    xyz = np.stack([
        np.cos(az) * cos_alt,
        np.sin(az) * cos_alt,
        np.sin(alt)
    ], axis=-1)
    return xyz

def dir_to_image_left_and_up(dir):
    image_left = np.cross(np.array([0.0, 0.0, 1.0]), dir)
    image_left_norm = np.linalg.norm(image_left)
    if image_left_norm > 0.0:
        image_left /= image_left_norm
    else:
        # if we're looking straight up or down, arbitrary dir
        image_left = np.array([0.0, -1.0, 0.0])
    image_up = np.cross(dir, image_left)
    image_up_norm = np.linalg.norm(image_up)
    if image_up_norm > 0.0:
        image_up /= image_up_norm
    else:
        # if dir was zero, another arbitrary dir
        image_up = np.array([-1.0, 0.0, 0.0])
    return image_left, image_up

def location_to_string(location):
    if location is not None:
        return f'{location.geodetic.lat}, {location.geodetic.lon}, {location.geodetic.height}'
    return None


class ExecutionState(enum.IntEnum):
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001


def sleep_inhibit():
    if os.name == 'nt':        
        ctypes.windll.kernel32.SetThreadExecutionState(ExecutionState.ES_CONTINUOUS | ExecutionState.ES_SYSTEM_REQUIRED)

def sleep_uninhibit():
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(ExecutionState.ES_CONTINUOUS)