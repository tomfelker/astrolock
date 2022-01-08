import serial
import serial.tools.list_ports
import threading
import time

import astrolock.model.telescope_connections.com_port

def formatForSerial(x):
    return str(round(x, 4))

class AngleClockConnection(astrolock.model.telescope_connections.com_port.ComPortConnection):
    
    @classmethod
    def get_url_scheme(cls):
        return "angle_clock:"

    def get_baud_rate(self):
        return 115200

    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

    def loop(self):
        self._open_serial_stream()

        while not self.want_to_stop:
            self.send_command(.1, .1, .1)
            self.update_callback()
            time.sleep(1)
            self.record_loop_rate()
            
        self.update_callback()

    def send_command(self, a, b, c):
           command = "a" + formatForSerial(a) + "b" + formatForSerial(b) + "c" + formatForSerial(c) + "\n"
           print(f'sending {command}')
           self.serial_stream.write(command.encode("utf-8"))
