import serial
import serial.tools.list_ports
import astropy.units as u
import astropy.time
import math

import astrolock.model.telescope_connections.com_port

class CelestronNexstarHCConnection(astrolock.model.telescope_connections.com_port.ComPortConnection):
    @staticmethod
    def get_url_scheme():
        return 'celestron_nexstar_hc:'

    def get_baud_rate(self):
        return 9600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loop(self):
        with self._open_serial_stream():
            while not self.want_to_stop:
                
                self.desired_axis_rates = self.tracker.get_rates()
                self.tracker.notify_idle()
                self._serial_send_axis_rate_cmd(0, self.desired_axis_rates[0].to_value(u.arcsec / u.s))
                #self.desired_axis_rates = self.tracker.get_rates()
                self._serial_send_axis_rate_cmd(1, self.desired_axis_rates[1].to_value(u.arcsec / u.s))
                self.axis_angles[0] = self._serial_read_axis_position_radians(0) * u.rad
                self.axis_angles_measurement_time[0] = astropy.time.Time.now()
                #self.tracker.notify_status_changed()
                self.axis_angles[1] = self._serial_read_axis_position_radians(1) * u.rad
                self.axis_angles_measurement_time[1] = astropy.time.Time.now()
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
