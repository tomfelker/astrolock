import math
import numpy as np
import os
import ctypes
import enum

def lerp(a, b, t):
    return (1 - t) * a + t * b


def wrap_angle_plus_minus_pi_radians(theta):
    return np.arctan2(np.sin(theta), np.cos(theta))

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