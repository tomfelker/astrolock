import serial
import serial.tools.list_ports
import threading

import astrolock.model.telescope_connections.threaded as threaded

class ComPortConnection(threaded.ThreadedConnection):

    def get_baud_rate(self):
        raise NotImplementedError

    @classmethod
    def get_urls(cls):
        print(f'get_urls for {cls}')
        url_scheme = cls.get_url_scheme()
        urls = []
        comports = serial.tools.list_ports.comports()
        for comport in comports:
            url = url_scheme + comport.device
            urls.append(url)
        return urls

    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

    def _open_serial_stream(self):
        self.serial_stream = serial.Serial(self.url_path, self.get_baud_rate())
        return self.serial_stream
