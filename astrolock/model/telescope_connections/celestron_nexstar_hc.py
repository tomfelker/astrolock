"""
This can talk to a Celestron hand controller.

Many thanks for the great docs here:  http://www.paquettefamily.ca/nexstar/NexStar_AUX_Commands_10.pdf

I've tested it mostly with my CPC 1100.  Some notes:

- Max slew rates:
    - azimuth - just by sound, seems to achieve 4 deg/s
    - altitude - 

"""


import serial
import serial.tools.list_ports
import astropy.units as u
import astropy
import astropy.time
from astrolock.model.util import *
import time
import math

import astrolock.model.telescope_connections.com_port

def bytes_to_uint(bytes):
    ret = 0
    for byte in bytes:
        ret = (ret << 8) + byte
    return ret

def bytes_to_radians(bytes):
    assert(len(bytes) == 3)
    # python is kind of annoying this way... ints are magic, rather than fixed-size two's complement as every other language
    # but luckily, the circle wraps around the same wa y as two's complement does!    
    return (bytes_to_uint(bytes) << 8) / 0x100000000 * 2 * math.pi

class CelestronNexstarHCConnection(astrolock.model.telescope_connections.com_port.ComPortConnection):
    @staticmethod
    def get_url_scheme():
        return 'celestron_nexstar_hc:'

    def get_baud_rate(self):
        return 9600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # request it once on first connect, and then again if we are kicked.
        self.gps_requested = True

    def loop(self):
        rad_to_arcsec = u.Quantity(1.0, unit=u.rad).to_value(u.arcsec)

        with self._open_serial_stream():
            while not self.want_to_stop:
                
                self.desired_axis_rates = self.tracker.consume_input_and_calculate_raw_axis_rates()

                if self.gps_requested:
                    self._read_gps()

                self.tracker.notify_idle()
                for axis in range(2):
                    self._serial_send_axis_rate_cmd(axis, self.desired_axis_rates[axis] * rad_to_arcsec)
    
                for axis in range(2):
                    old_axis_angle = self.axis_angles[axis]
                    old_measurement_time_ns = self.axis_measurement_times_ns[axis]

                    self.axis_angles[axis] = self._serial_read_axis_position_radians(axis) 
                    self.axis_measurement_times_ns[axis] = time.perf_counter_ns()

                    axis_dt = (self.axis_measurement_times_ns[axis] - old_measurement_time_ns) * 1e-9
                    self.estimated_axis_rates[axis] = wrap_angle_plus_minus_pi_radians(self.axis_angles[axis] - old_axis_angle) / axis_dt

                self.tracker.notify_status_changed()
                self.record_loop_rate()
                

    # NOTE! Windows Prolific serial drivers suck, and will hang/BSOD if you call this from different threads - be careful

    # axis 0 is azimuth, yaw, or right ascension (the one which needs acceleration when the other is near 90 degrees)
    # axis 1 is altitude, pitch, or declination (which needs no compensation)
    def _serial_send_axis_rate_cmd(self, axis, arcseconds_per_second):
        if axis > 1 or axis < 0:
            raise RuntimeError("bad axis") 
        rate_int = int(round(abs(arcseconds_per_second) * 4))
        rate_int_clamped = min(rate_int, 255 * 256 - 1) #sic - rates above this, though representable, make the controller stop
        rate_hi = rate_int_clamped // 256
        rate_lo = rate_int_clamped % 256 
        cmd = bytes([
            ord('P'), # passthru
            3,
            16 + axis,
            6 if arcseconds_per_second > 0 else 7,
            rate_hi,
            rate_lo,
            0,
            0
            ])
      
        self.serial_stream.write(cmd)
        reply = self.serial_stream.read(1)
        if len(reply) != 1 or reply[0] != ord('#'):
            raise ConnectionError("read error setting rate")

        # the rate we actually set, taking clamping and rounding into account
        return math.copysign(rate_int_clamped, arcseconds_per_second) / 4

    def _serial_read_axis_position_radians(self, axis):
        cmd = bytes([
            # passthru
            ord('P'),
            # length
            3,
            # destination id
            16 + axis,
            # msgId, MC_GET_POSITION
            1,
            # data 0 to 3
            0,
            0,
            0,
            # response bytes
            3
            ])
        self.serial_stream.write(cmd)
        angle_bytes = self.serial_stream.read(4)
        if len(angle_bytes) != 4 or angle_bytes[3] != ord('#'):
            raise ConnectionError("read error getting axis position")
        angle_int = angle_bytes[0] * 65536 + angle_bytes[1] * 256 + angle_bytes[2]
        angle_radians = angle_int / 16777216 * math.pi * 2
        return angle_radians
    
    def request_gps(self):
        self.gps_requested = True

    def _read_gps(self):
        self.gps_requested = False
        self.gps_location = None
        self.gps_time = None
        self.gps_measurement_time_ns = None

        linked, = self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_LINKED, response_len = 1)
        if linked != 0:
            lat_rad = bytes_to_radians(self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_GET_LAT, response_len = 3))
            lon_rad = bytes_to_radians(self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_GET_LONG, response_len = 3))
            height_m = bytes_to_uint(self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_GET_HEIGHT, response_len = 2))
            self.gps_location = astropy.coordinates.EarthLocation.from_geodetic(lat = lat_rad * u.rad, lon = lon_rad * u.rad, height=height_m * u.m)
            time_valid, = self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_TIME_VALID, response_len = 1)
            if time_valid != 0:
                year = bytes_to_uint(self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_GET_YEAR, response_len = 2))
                month, day = self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_GET_DATE, response_len = 2)
                hour, minute, second = self._send_and_receive_via_hc(self.DEV_ID_GPS, self.GPS_GET_TIME, response_len = 3)
                self.gps_measurement_time_ns = time.perf_counter_ns()                
                self.gps_time = astropy.time.Time({'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second}, scale='utc') # TODO: 'tai'?
            


    def _send_and_receive_via_hc(self, dest_id, msg_id, data = (), response_len = 0):
        assert(len(data) <= 3)
        msg_len = len(data) + 1        
        cmd = bytes([
            0x50, # 'P'
            msg_len,
            dest_id,
            msg_id,
            data[0] if len(data) > 0 else 0,
            data[1] if len(data) > 1 else 0,
            data[2] if len(data) > 2 else 0,
            response_len
        ])
        self.serial_stream.write(cmd)
        response = self.serial_stream.read(response_len + 1)
        if len(response) != response_len + 1 or response[response_len] != 0x23:  # '#'
            raise ConnectionError("Error reading {msg_id} from {dest_id}.")
        return response[:-1]

    
    DEV_ID_MAIN = 0x01
    DEV_ID_HC = 0x04
    DEV_ID_MC_AZM = 0x10
    DEV_ID_MC_ALT = 0x11
    DEV_ID_GPS = 0xb0    

    HC_PASSTHROUGH = 0x50 # 'P'
    HC_OK = 0x23 # '#'

    MC_GET_POSITION = 0x01
    MC_SET_POS_GUIDERATE = 0x06
    MC_SET_NEG_GUIDERATE = 0x07

    GPS_GET_LAT = 0x01
    GPS_GET_LONG = 0x02
    GPS_GET_DATE = 0x03
    GPS_GET_YEAR = 0x04
    GPS_GET_SAT_INFO = 0x07
    GPS_GET_RCVR_STATUS = 0x08
    GPS_GET_HEIGHT = 0x09  # this one wasn't in the docs, but seems to work for me.
    GPS_GET_TIME = 0x33
    GPS_TIME_VALID = 0x36
    GPS_LINKED = 0x37
    GPS_GET_HW_VER = 0x55
    GPS_GET_COMPASS = 0xa0
    GPS_GET_VER = 0xfe
