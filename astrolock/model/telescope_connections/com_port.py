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
        filtered_comports = list(filter(cls.filter_comport, comports))
        if len(comports) > 0 and len(filtered_comports) == 0:
            print("Found COM ports but none matched known devices.  Listing them all anyway.  If you connect and it works, please share this log:")
            for comport in comports:
                print(f'\t{comport.device}: {comport.description}, {comport.hwid}, {comport.manufacturer}, {comport.product}')
            filtered_comports = comports
        for comport in filtered_comports:
            url = url_scheme + comport.device
            urls.append(url)
        return urls
    
    @classmethod
    def filter_comport(self, comport):
        return True

    def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)

    def _open_serial_stream(self):
        self.serial_stream = serial.Serial(self.url_path, self.get_baud_rate())
        return self.serial_stream
