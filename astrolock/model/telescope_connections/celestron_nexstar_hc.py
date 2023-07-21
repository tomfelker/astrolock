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
import enum

import astrolock.model.telescope_connections.com_port

class CelestronNexstarDeviceIds(enum.IntEnum):
    DEV_ID_MAIN = 0x01
    DEV_ID_HC = 0x04
    DEV_ID_MC_AZM = 0x10
    DEV_ID_MC_ALT = 0x11
    DEV_ID_GPS = 0xb0

    @classmethod
    def MC_for_axis(cls, axis):
        assert(axis >= 0)
        assert(axis < 2)
        return cls.DEV_ID_MC_AZM + axis

class CelestronNexstarResponseCodes(enum.IntEnum):
    HC_OK = 0x23 # '#'

class CelestronNexstarCommands(enum.IntEnum):
    HC_PASSTHROUGH = 0x50 # 'P'

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
    
    @classmethod
    def filter_comport(self, comport):
        if comport.vid is not None and comport.vid == 0x067B:
            return True
        return False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # request it once on first connect, and then again if we are kicked.
        self.gps_requested = True

        self.last_message_thinking_mid_time_ns = None
        self.last_message_thinking_end_time_ns = None

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
                    self.axis_measurement_times_ns[axis] = self.last_message_thinking_mid_time_ns

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

        self._send_and_receive_via_hc(
            dest_id=CelestronNexstarDeviceIds.MC_for_axis(axis),
            msg_id=(CelestronNexstarCommands.MC_SET_POS_GUIDERATE if arcseconds_per_second > 0 else CelestronNexstarCommands.MC_SET_NEG_GUIDERATE),
            data=(rate_hi, rate_lo),
            response_len=0
        )

        # the rate we actually set, taking clamping and rounding into account
        return math.copysign(rate_int_clamped, arcseconds_per_second) / 4

    def _serial_read_axis_position_radians(self, axis):
        angle_bytes = self._send_and_receive_via_hc(
            dest_id=CelestronNexstarDeviceIds.MC_for_axis(axis),
            msg_id=CelestronNexstarCommands.MC_GET_POSITION,
            data=(),
            response_len=3
        )
        angle_radians = bytes_to_radians(angle_bytes)
        return angle_radians
    
    def request_gps(self):
        self.gps_requested = True

    def _read_gps(self):
        self.gps_requested = False
        self.gps_location = None
        self.gps_time = None
        self.gps_measurement_time_ns = None

        linked, = self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_LINKED, response_len = 1)
        if linked != 0:
            lat_rad = bytes_to_radians(self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_GET_LAT, response_len = 3))
            lon_rad = bytes_to_radians(self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_GET_LONG, response_len = 3))
            height_m = bytes_to_uint(self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_GET_HEIGHT, response_len = 2))
            self.gps_location = astropy.coordinates.EarthLocation.from_geodetic(lat = lat_rad * u.rad, lon = lon_rad * u.rad, height=height_m * u.m)
            time_valid, = self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_TIME_VALID, response_len = 1)
            if time_valid != 0:
                year = bytes_to_uint(self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_GET_YEAR, response_len = 2))
                month, day = self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_GET_DATE, response_len = 2)
                hour, minute, second = self._send_and_receive_via_hc(CelestronNexstarDeviceIds.DEV_ID_GPS, CelestronNexstarCommands.GPS_GET_TIME, response_len = 3)

                # Note that we don't need to use _set_gps_time_with_inferred_seconds_fraction() here... it appears that the hand controller delays
                # responding to these commands by up to a second, so presumably it's synchronizing things for us.  However, it seems to consistently
                # be slightly late, so perhaps they're checking the wrong edge of a clock signal.
                #
                # https://www.cnssys.com/files/M12+UsersGuide.pdf says:
                #   The rising edge of the 1PPS signal is the time reference. The falling edge will occur
                #   approximately 200 ms (+/-1 ms) after the rising edge. The falling edge should not be used for
                #   accurate time keeping. 
                #
                # so it's probably that.  Could also just be the length of serial output from the module, which starts ~50 ms after the pulse,
                # but we can't really predict its length.

                second += .2
                self.gps_time = astropy.time.Time({'year': year, 'month': month, 'day': day, 'hour': hour, 'minute': minute, 'second': second}, scale='utc')
                self.gps_measurement_time_ns = self.last_message_thinking_end_time_ns
            


    def _send_and_receive_via_hc(self, dest_id, msg_id, data = (), response_len = 0, expect_delay = False):
        assert(len(data) <= 3)
        msg_len = len(data) + 1        
        cmd = bytes([
            CelestronNexstarCommands.HC_PASSTHROUGH, # 'P'
            msg_len,
            dest_id,
            msg_id,
            data[0] if len(data) > 0 else 0,
            data[1] if len(data) > 1 else 0,
            data[2] if len(data) > 2 else 0,
            response_len
        ])
        write_start_time_ns = time.perf_counter_ns()
        self.serial_stream.write(cmd)
        response = self.serial_stream.read(response_len + 1)
        read_finish_time_ns = time.perf_counter_ns()
        if len(response) != response_len + 1 or response[response_len] != CelestronNexstarResponseCodes.HC_OK:  # '#'
            raise ConnectionError(f"Error reading {msg_id.name} from {dest_id.name}.")
        
        total_time = (read_finish_time_ns - write_start_time_ns) * 1e-9
        write_time = len(cmd) * 8 / self.get_baud_rate()
        read_time = len(response) * 8 / self.get_baud_rate()
        thinking_time = total_time - write_time - read_time
        if thinking_time < 0:
            print(f"Hmm, telescope responded too soon by {-thinking_time} s for {dest_id.name} {msg_id.name}")
        if not expect_delay and thinking_time > .1:
            print(f"Hmm, either telescope responded late or someone's hogging the GIL, thinking_time {thinking_time} s for {dest_id.name} {msg_id.name}")
        self.last_message_thinking_mid_time_ns = write_start_time_ns + int(write_time + thinking_time / 2)
        self.last_message_thinking_end_time_ns = write_start_time_ns + int(thinking_time)

        return response[:-1]

    

